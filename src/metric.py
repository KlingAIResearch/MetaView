import cv2
import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
from tqdm import tqdm
import argparse

from einops import rearrange

import lpips
from PIL import Image

import torch.nn.functional as F

sys.path.append(os.getcwd())
sys.path.append("./UFM")

from uniflowmatch.utils.geometry import get_meshgrid_torch

from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def warp_image_with_flow(source_image, source_mask, target_image, flow, thresh) -> np.ndarray:
    """
    Warp the target to source image using the given flow vectors.
    Flow vectors indicate the displacement from source to target.

    Args:
    source_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    target_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    flow: np.ndarray of shape (H, W, 2)
    source_mask: non_occluded mask represented in source image.

    Returns:
    warped_image: target_image warped according to flow into frame of source image
    np.ndarray of shape (H, W, 3), normalized to [0, 1]

    """
    # assert source_image.shape[-1] == 3
    # assert target_image.shape[-1] == 3

    assert flow.shape[-1] == 2

    # Get the shape of the source image
    height, width = source_image.shape[:2]
    target_height, target_width = target_image.shape[:2]

    # Create mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply flow displacements
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    x_new = np.clip(x + flow_x, 0, target_width - 1) + 0.5
    y_new = np.clip(y + flow_y, 0, target_height - 1) + 0.5

    x_new = (x_new / target_image.shape[1]) * 2 - 1
    y_new = (y_new / target_image.shape[0]) * 2 - 1

    warped_image = F.grid_sample(
        torch.from_numpy(target_image).permute(2, 0, 1)[None, ...].float(),
        torch.from_numpy(np.stack([x_new, y_new], axis=-1)).float()[None, ...],
        mode="bilinear",
        align_corners=False,
    )

    warped_image = warped_image[0].permute(1, 2, 0).numpy()

    if source_mask is not None:
        warped_image = warped_image * (source_mask > thresh)

    return warped_image

def compute_distance_vectorized(flow_output, covisibility_src, covisibility, 
                                    src_thresh, thresh, target_image_shape):
    height, width = target_image_shape[0], target_image_shape[1]
    
    src_mask = covisibility_src > src_thresh
    valid_mask = src_mask & (covisibility > thresh)
    penalty_mask = src_mask & ~valid_mask
    
    cnt = np.sum(src_mask)
    
    flow_normalized = flow_output / np.array([width, height])[:, None, None]
    distances = np.sqrt(np.sum(flow_normalized**2, axis=0))
    
    # total_dist = valid_dist + penalty
    total_dist = np.sum(distances[valid_mask]) + np.sum(penalty_mask) * np.sqrt(2)
    
    return total_dist, int(cnt)



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='None', help='Image file path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

thresh = 0.2 # generate 
src_thresh = 0.3 # src

refinement_model = True
resolution = 560

model = UniFlowMatchClassificationRefinement.from_pretrained(
    "infinity1096/UFM-Refine" if resolution == 560 else "infinity1096/UFM-Refine-980"
).to(device)

model.eval()

data_path = args.data_path

save = False
files = os.listdir(data_path)

LPIPS = lpips.LPIPS(net="vgg").to(device)
total_psnr = 0
total_ssim = 0
total_lpips = 0
total_dist = 0

count = len(files)
invalid = 0
for idx, file in tqdm(enumerate(files), total=count):
    print(file)

    combined_img = Image.open(os.path.join(data_path, file)).resize((960 * 3, 528))

    width, height = combined_img.size

    length = width // 3

    src_img = np.array(combined_img.crop((length * 0, 0, length * 1, height)))
    generated_img = np.array(combined_img.crop((length * 1, 0, length * 2, height)))
    gt_img = np.array(combined_img.crop((length * 2, 0, length * 3, height)))

    psnr = peak_signal_noise_ratio(generated_img, gt_img)
    ssim = structural_similarity(generated_img, gt_img, channel_axis=-1)

    img1 = rearrange(torch.Tensor(generated_img).unsqueeze(0), 'b h w c -> b c h w').to(device) / 127.5 - 1.0
    img2 = rearrange(torch.Tensor(gt_img).unsqueeze(0), 'b h w c -> b c h w').to(device) / 127.5 - 1.0
    lpips = LPIPS(img1, img2)

    total_psnr += psnr
    total_ssim += ssim
    total_lpips += lpips

    source_image = gt_img
    target_image = generated_img
    # === Predict Correspondences ===
    result = model.predict_correspondences_batched(
        source_image=torch.from_numpy(gt_img).to(device),
        target_image=torch.from_numpy(generated_img).to(device),
    )

    result2 = model.predict_correspondences_batched(
        source_image=torch.from_numpy(gt_img).to(device),
        target_image=torch.from_numpy(src_img).to(device),
    )
    covisibility_src = result2.covisibility.mask[0].cpu().numpy()

    flow_output = result.flow.flow_output[0].cpu().numpy()  # 2 H W
    covisibility = result.covisibility.mask[0].cpu().numpy()


    dist, cnt = compute_distance_vectorized(
        flow_output, covisibility_src, covisibility, 
        src_thresh, thresh, target_image.shape
    )

    if cnt:
        dist /= cnt
        dist = dist / np.sqrt(2) * 100 # normalized to 0~100
    else:
        # non-covisible between src and gt
        dist = 0
        invalid += 1
        
    total_dist += dist
    print('')
    print(f"PSNR: {psnr}")
    print(f"SSIM: {ssim}")
    print(f"LPIPS: {lpips}")
    print(f"dist: {dist} {cnt}")

    if not save:
        continue

    # === Visualize Results ===
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    axs[0, 0].imshow(source_image)
    axs[0, 0].set_title("Source Image")

    axs[0, 1].imshow(target_image)
    axs[0, 1].set_title("Target Image")

    # Warp the image using flow
    warped_image = warp_image_with_flow(source_image, None, target_image, flow_output.transpose(1, 2, 0), thresh)
    warped_image = covisibility[..., None] * warped_image + (1 - covisibility[..., None]) * 255 * np.ones_like(
        warped_image
    )
    warped_image /= 255.0

    axs[0, 2].imshow(warped_image)
    axs[0, 2].set_title("Warped Image")

    # Flow visualization
    flow_vis_image = flow_vis.flow_to_color(flow_output.transpose(1, 2, 0))
    axs[1, 0].imshow(flow_vis_image)
    axs[1, 0].set_title(f"Flow Output (Valid at covisible region) {np.round(dist, decimals=2)}, psnr:{psnr:.2f}, ssim:{ssim:.2f}")

    # Covisibility mask
    axs[1, 1].imshow(covisibility > thresh, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title(f"Covisibility Mask ({thresh})")

    heatmap = axs[1, 2].imshow(covisibility, cmap="gray", vmin=0, vmax=1)
    axs[1, 2].set_title("Covisibility Mask")
    plt.colorbar(heatmap, ax=axs[1, 2])

    if not os.path.exists(f"result_{save}"):
        os.mkdir(f"result_{save}")

    plt.tight_layout()
    plt.savefig(f"result_{save}/{file}.png")

print(f"PSNR: {total_psnr / len(files):.2f} dB")
print(f"SSIM: {total_ssim / len(files):.4f}")
print(f"LPIPS: {total_lpips.item() / count:.4f}")
print(f"dist: {total_dist / (len(files) - invalid):.4f}")