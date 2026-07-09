import torch, os, sys, glob
import argparse
import math
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())
sys.path.append("./DepthAnything3/src")

from DepthAnything3.src.depth_anything_3.api import DepthAnything3

from diffsynth.core import ModelConfig
from diffsynth import load_state_dict
from src.MetaView_pipeline import MetaViewPipeline
import torch.nn.functional as F

def compute_target_extrinsic(yaw_deg, pitch_deg, radius):
    """
    Compute the camera extrinsic matrix (World-to-Camera) for rotation
    around a sphere center in front of the camera.
    Supports simultaneous yaw (left-right) and pitch (up-down) angles.

    Args:
        yaw_deg (float): Yaw angle in degrees.
        pitch_deg (float): Pitch angle in degrees.
        radius (float): Distance from the rotation sphere center to the camera
                        (typically the depth of the target object).
    Returns:
        numpy.ndarray: 4x4 extrinsic matrix.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    
    # Rotation matrix around Y axis (Yaw)
    R_y = np.array([
        [np.cos(yaw),  0, np.sin(yaw)],
        [0,            1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # Rotation matrix around X axis (Pitch)
    R_x = np.array([
        [1, 0,             0            ],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch) ]
    ])
    
    # Combined rotation (pitch first, then yaw)
    R = R_y @ R_x
    
    # Set sphere center coordinates C
    C = np.array([0.0, 0.0, radius])
    
    # Compute translation vector t = C - R * C
    t = C - R @ C
    
    # Construct 4x4 extrinsic matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def main():
    parser = argparse.ArgumentParser(description="MetaView Interactive Inference CLI")
    
    # Core interactive parameters
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="./output_novel_view.png", help="Path to save the generated image")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw angle in degrees (e.g., 60 for right, -60 for left)")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch angle in degrees (e.g., 30 for top, -30 for bottom)")
    parser.add_argument("--radius", type=float, default=None, help="Rotation radius. If None, auto-calculated from center depth.")
    
    # Model path parameters
    parser.add_argument("--da3_giant_path", type=str, default="../../Depth-Anything-3/model/DA3-GIANT-1.1", help="Path to DA3 Giant")
    parser.add_argument("--da3_depth_path", type=str, default="../../Depth-Anything-3/model/DA3NESTED-GIANT-LARGE-1.1", help="Path to DA3 Depth model")
    parser.add_argument("--qwen_path", type=str, default=None, help="Base path to Qwen-Image-Edit models")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained MetaView checkpoint (.safetensors)")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Global parameter configuration
    export_3D_feat_layers = [19, 27, 33, 39]
    prope_dim_arrange = [64, 20, 20, 24]
    add_depth = (len(prope_dim_arrange) == 4)
    merge_3D = True
    prompt = ["镜头视角转到指定位置"]

    # 1. Load input image
    print(f"[*] Loading input image from {args.image_path}...")
    original_image = Image.open(args.image_path).convert("RGB")
    edit_image = original_image.resize((960, 528))
    
    # =========================================================================
    # 2. Depth and geometry prior extraction (Depth & Intrinsics)
    # =========================================================================
    print("[*] Loading DepthAnything3 Prior Models...")
    
    with torch.inference_mode():
        # Load feature extraction model (GIANT)
        model_3D = DepthAnything3.from_pretrained(args.da3_giant_path).to(device=device)
        
        print("    -> Extracting 3D Features and Intrinsics...")
        feat_3D_output = model_3D.inference([edit_image], export_feat_layers=export_3D_feat_layers, process_res=840)
        
        # Process intrinsics Ks
        intri = feat_3D_output.intrinsics[0]
        width = intri[0, 2] * 2
        height = intri[1, 2] * 2
        Ks_matrix = [
            [intri[0, 0] / width, 0.0,                  0.0],
            [0.0,                 intri[1, 1] / height, 0.0],
            [0.0,                 0.0,                  1.0],
        ]
        Ks = torch.Tensor(Ks_matrix)
        Ks = torch.stack([Ks, Ks], dim=0).unsqueeze(0) # Shape: (1, 2, 3, 3)
        
        # Process features feat_3D
        feats = [torch.from_numpy(feat_3D_output.aux[f"feat_layer_{layer}"]) for layer in export_3D_feat_layers]
        feat_3D = torch.cat(feats, dim=-1).to(dtype=torch.bfloat16, device=device)
        
        # Release feature extraction model
        model_3D.to("cpu")
        del model_3D
        torch.cuda.empty_cache()

        # Load depth extraction model (NESTED)
        print("    -> Extracting Depth Map...")
        model_depth = DepthAnything3.from_pretrained(args.da3_depth_path).to(device=device)
        prediction = model_depth.inference([edit_image], process_res=840)
        
        depth_edit = torch.Tensor(prediction.depth).unsqueeze(0)
        depth_edit = F.interpolate(depth_edit, size=(528, 960), mode='bilinear', align_corners=False)[0]
        depth_latent = torch.zeros_like(depth_edit)
        depth = torch.cat([depth_latent, depth_edit], dim=0).unsqueeze(0) # Shape: (1, 2, H, W)
        
        # Release depth model
        model_depth.to("cpu")
        del model_depth
        torch.cuda.empty_cache()

    # =========================================================================
    # 3. Target pose calculation
    # =========================================================================
    # Auto-derive radius: if user did not specify radius, use center depth from the depth map
    if args.radius is None:
        depth_squeeze = depth[0, 1] # Get real depth channel
        z_c = depth_squeeze[depth_squeeze.shape[0]//2, depth_squeeze.shape[1]//2].item()
        args.radius = z_c
        print(f"[*] Auto-calculated rotation radius from center depth: {args.radius:.4f}")
        
    print(f"[*] Calculating Target Pose -> Yaw: {args.yaw}°, Pitch: {args.pitch}°, Radius: {args.radius}")
    extrinsic_target = compute_target_extrinsic(args.yaw, args.pitch, args.radius)
    extrinsic_source = np.eye(4)
    
    # Construct viewmats tensor: Shape (1, 2, 4, 4) -> [Target, Source]
    viewmats = torch.Tensor(np.stack((extrinsic_target, extrinsic_source), axis=0)).unsqueeze(0)

    # =========================================================================
    # 4. Generation model loading and inference (DiT Pipeline)
    # =========================================================================
    print("[*] Loading Qwen-Image-Edit Pipeline...")

    if args.qwen_path:
        print(f"[*] Loading Qwen-Image-Edit from {args.qwen_path}")
        pipe = MetaViewPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=glob.glob(f"{args.qwen_path}/Qwen-Image-Edit/transformer/diffusion_pytorch_model*.safetensors")),
                ModelConfig(path=glob.glob(f"{args.qwen_path}/Qwen-Image-Edit/text_encoder/model*.safetensors")),
                ModelConfig(path=glob.glob(f"{args.qwen_path}/Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors")),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(path=f"{args.qwen_path}/Qwen-Image-Edit/processor/"),
        )
    else:   # Auto download
        pipe = MetaViewPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    print(f"[*] Loading MetaView Weights from {args.ckpt_path}...")
    state_dict = load_state_dict(args.ckpt_path)
    pipe.dit.load_state_dict(state_dict, strict=False)

    print("[*] Starting Generation (40 Steps)...")
    with torch.inference_mode():
        generated_image = pipe(
            prompt, edit_image=edit_image, edit_image_auto_resize=False,
            seed=0,
            viewmats=viewmats.to(device=device, dtype=torch.bfloat16),
            Ks=Ks.to(device=device, dtype=torch.bfloat16),
            prope_dim_arrange=prope_dim_arrange,
            add_attn=True,
            add_3D=True,
            feat_3D=feat_3D,
            depth=depth.to(device=device, dtype=torch.bfloat16) if add_depth else None,
            merge_3D=merge_3D,
            val=True,
            num_inference_steps=40,
            height=528, width=960,
        )

    # =========================================================================
    # 5. Save result (stitch source and generated images for comparison)
    # =========================================================================
    stitched_image = Image.new('RGB', (960 * 2, 528), (255, 255, 255))
    stitched_image.paste(edit_image, (0, 0))
    stitched_image.paste(generated_image, (960, 0))
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    stitched_image.save(args.output_path)
    print(f"Success! Result saved to {args.output_path}")

if __name__ == '__main__':
    main()