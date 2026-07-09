from diffsynth.core.data.operators import *
import torch, json, pandas, sys, os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

sys.path.append(os.getcwd())
sys.path.append("./DepthAnything3/src")

from DepthAnything3.src.depth_anything_3.api import DepthAnything3

from openexr_numpy import imread, imwrite
import torch.nn.functional as F

import pandas as pd

class MetaViewUnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        prope=False,
        debug=False,
        mode="train",
        norm_scale=1.0,
        path_3D=None,
        export_3D_feat_layers=None,
        anno_src=None,
        add_depth=False,
        subset=None,
        base_model = "qwen",
    ):
        self.base_model = base_model
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []      
        self.cached_data = []

        paths = base_path.split(";")

        if subset is None:
            self.videos = os.listdir(base_path)
        elif len(paths) > 1:
            self.videos = []
            for path in paths:
                dirs = os.listdir(path)
                for sub in subset:
                    if sub in dirs:
                        p = os.path.join(path, sub)
                        folders = os.listdir(p)
                        for folder in folders:
                            self.videos.append(os.path.join(p, folder))
        else:
            self.videos = []
            for sub in subset:
                p = os.path.join(base_path, sub)
                dirs = os.listdir(p)
                for d in dirs:
                    self.videos.append(os.path.join(sub, d))
        self.total_length = len(self.videos)
        self.prope = prope
        self.mode = mode
        self.norm_scale = norm_scale
        self.anno_src = anno_src
        self.add_depth = add_depth
        self.subset = subset

        if prope:
            self.load_from_cache = False
        
        self.model_3D = None
        if path_3D is not None:
            device = torch.device("cuda")
            self.model_3D = DepthAnything3.from_pretrained(path_3D)
            self.model_3D = self.model_3D.to(device=device)
            export_3D_feat_layers = export_3D_feat_layers.split(",")
            self.export_3D_feat_layers = [int(s) for s in export_3D_feat_layers]
        
        if debug:
            self.max_dist()
            exit(0)

    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    

    def __getitem__(self, data_id):
        if self.prope:
            if len(self.base_path.split(";")) > 1:
                return self.getitem_prope_all(data_id)
            elif "DL3DV" in self.base_path and self.metadata_path is None:
                return self.getitem_prope_DL3DV(data_id)
            elif "DL3DV" in self.base_path and self.metadata_path:
                return self.getitem_metadata_DL3DV(data_id)
            elif self.metadata_path:
                return self.getitem_metadata(data_id)
        
        return data
    
    def quick_check(self, extrinsics: torch.Tensor) -> bool:
        """check extrinsics"""
        if extrinsics.shape[-2:] != (4, 4):
            print("Extrinsics wrong shape!")
            return False
        
        t_norm = torch.norm(extrinsics[:, :3, 3], dim=1)
        if t_norm.max() > 100:  # 阈值根据应用设定
            print(f"Extrinsics shift too large! {t_norm.max()}")
            return False

        R = extrinsics[..., :3, :3]
        # 检查行列式接近1
        det_R = torch.det(R)
        if not torch.allclose(det_R, torch.ones_like(det_R), atol=1e-4):
            print("Extrinsics not ortho!")
            return False
        
        # 检查最后一行
        last_row = extrinsics[..., 3, :]
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsics.device)
        if not torch.allclose(last_row, expected.expand_as(last_row), atol=1e-4):
            print("Extrinsics wrong row!")
            return False
        
        return True


    def getitem_prope_DL3DV(self, index):
        video = self.videos[index % self.total_length]
        if not "Evaluation" in self.base_path:
            while not os.path.isdir(os.path.join(self.base_path, video)) or not os.path.exists(os.path.join(self.base_path, video, "transforms.json")):
                index += 1
                video = self.videos[index % self.total_length]
        
        blender2opencv = np.array(
            [[1, 0, 0, 0], 
            [0, -1, 0, 0], 
            [0, 0, -1, 0], 
            [0, 0, 0, 1]]
        )

        data = {}
        # if "Evaluation" in self.base_path:
        #     video = f"{video}/{video}/nerfstudio"
        with open(os.path.join(self.base_path, video, "transforms.json"), 'r', encoding='utf-8') as file:
            json_str = file.read()
            meta = json.loads(json_str)
            # frames = meta["frames"]
            # frames = sorted(frames, key=lambda x: x["colmap_im_id"])
            frames = sorted(os.listdir(os.path.join(self.base_path, video, "images_4")))
            import random

            interval = min(40, len(frames))
            extrinsics_check = False
            edit_idx = None
            while not extrinsics_check:
                extrinsics_check = True

                edit_idx = random.randint(0, len(frames) - interval)
                if "val" in self.mode:
                    edit_idx = 10
                # edit_image = os.path.join(video, frames[edit_idx]["file_path"].replace("images", "images_4"))
                edit_image = os.path.join(video, "images_4", frames[edit_idx])
                # print(edit_idx, frames[edit_idx])
                # if "Evaluation" in self.base_path and self.anno_src is not None and "vipe" in self.anno_src:
                #     vipe_pose = np.load(os.path.join(self.base_path, video.split('/')[0], "pose/video.npz"))
                #     edit_viewmats = vipe_pose["data"][edit_idx]
                if self.anno_src is not None and "vipe-DA3" in self.anno_src:
                    vipe_pose = np.load(os.path.join(self.base_path, video, "vipe-DA3/pose/video.npz"))
                    edit_viewmats = vipe_pose["data"][edit_idx]
                elif self.anno_src is not None and self.anno_src == "vipe":
                    vipe_pose = np.load(os.path.join(self.base_path, video, "pose/video.npz"))
                    edit_viewmats = vipe_pose["data"][edit_idx]
                else:
                    edit_viewmats = np.array(frames[edit_idx]["transform_matrix"], dtype=np.float32) @ blender2opencv # c2w! and invert y z axis! 
                edit_viewmats = torch.Tensor(edit_viewmats).unsqueeze(0)
                
                max_idx = min(40, len(frames) - edit_idx)
                target_idx = random.randint(edit_idx + max_idx // 2, edit_idx + max_idx - 1)
                if "val" in self.mode:
                    target_idx = 30
                # target_image = os.path.join(video, frames[target_idx]["file_path"].replace("images", "images_4"))
                target_image = os.path.join(video, "images_4", frames[target_idx])
                # if "Evaluation" in self.base_path and self.anno_src is not None and "vipe" in self.anno_src:
                #     vipe_pose = np.load(os.path.join(self.base_path, video.split('/')[0], "pose/video.npz"))
                #     target_viewmats = vipe_pose["data"][target_idx]
                if self.anno_src is not None and "vipe-DA3" in self.anno_src:
                    vipe_pose = np.load(os.path.join(self.base_path, video, "vipe-DA3/pose/video.npz"))
                    target_viewmats = vipe_pose["data"][target_idx]
                elif self.anno_src is not None and self.anno_src == "vipe":
                    vipe_pose = np.load(os.path.join(self.base_path, video, "pose/video.npz"))
                    target_viewmats = vipe_pose["data"][target_idx]
                else:
                    target_viewmats = np.array(frames[target_idx]["transform_matrix"], dtype=np.float32) @ blender2opencv
                target_viewmats = torch.Tensor(target_viewmats).unsqueeze(0)
            
                edit_c2w = edit_viewmats
                target_c2w = target_viewmats
                in_c2ws = torch.cat([target_c2w, edit_c2w], dim=0)
                
                # normalize
                c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(edit_c2w[0]), in_c2ws)  # shift to src coord(edit_image)
                c2ws[:, :3, 3] /= self.norm_scale #20.0 # 10.0 # translation normalized
                # print(c2ws)
                
                if not self.quick_check(c2ws[0:1, :, :]):
                    extrinsics_check = False

            #transform c2w to w2c align with PRoPE implementation
            viewmats = torch.linalg.inv(c2ws)


            s = 0
            ks = [  [meta["fl_x"],            s,  meta["cx"]],
                    [           0, meta["fl_y"],  meta["cy"]],
                    [           0,            0,           1]]
            ks = torch.Tensor(ks).unsqueeze(0)
            image_height = meta["h"]
            image_width = meta["w"]
            ks[..., 0, 0] = ks[..., 0, 0] / image_width
            ks[..., 1, 1] = ks[..., 1, 1] / image_height
            ks[..., 0, 2] = ks[..., 0, 2] / image_width - 0.5
            ks[..., 1, 2] = ks[..., 1, 2] / image_height - 0.5
            ks[..., 2, 2] = 1.0
            # ks has been normalized!!


            data["edit_image"] = self.main_data_operator(edit_image).resize((960, 528))
            data["image"] = self.main_data_operator(target_image).resize((960, 528))
            data["viewmats"] = viewmats
            data["Ks"] = torch.cat([ks, ks], dim=0)

            # print("original size : ",data["image"].size, data["edit_image"].size)
            if torch.isnan(data["viewmats"]).any() or torch.isnan(data["Ks"]).any():
                print("!!!camera param has NaN!!!")
                print(target_image, edit_image)
                exit(0)

            if "qwen" in self.base_model:
                data["prompt"] = "镜头视角转到指定位置"
            elif "flux" in self.base_model:
                data["prompt"] = "Turn to the target view"
            data["name"] = video

            # if data["edit_image"].size[0] != 960:   #7103edc158a862dbfa3c3454e4de584dad59c3c30055919f1dfa7fd7acfdd5c9
            #     print(f"{video} has different size!!")

            if self.model_3D is not None and "val" not in self.mode:
                feat_3D = self.model_3D.inference(
                    [data["edit_image"].resize((960, 528))],                                                 # (1, 33, 60)
                    export_feat_layers=self.export_3D_feat_layers,   # (1, 20, 36, 1536) H, W, C  (1, 20, 36, 1024) 960 528
                    process_res=840,
                )
                feats = []
                for layer in self.export_3D_feat_layers:
                    feats.append(torch.from_numpy(feat_3D.aux[f"feat_layer_{layer}"]))
                data["feat_3D"] = torch.cat(feats, dim=-1)[0] # (20, 36, 1536) H, W, C
                if torch.isnan(data["feat_3D"]).any():
                    print("!!!feat 3D has NaN!!!")
                    print(video, edit_idx)
                    exit(0)
            
            # target_z = imread(os.path.join(self.base_path, video, f"vipe-DA3/depth/{(target_idx):05d}.exr"), "Z")
            # target_z[np.isnan(target_z)] = 1000
            # target_z[(target_z > 1000) | np.isinf(target_z)] = 1000
            # target_depth = torch.Tensor(target_z)
            # data["target_depth"] = target_depth

            if self.add_depth:
                if "vipe-DA3" in self.anno_src:
                    z_channel = imread(os.path.join(self.base_path, video, f"vipe-DA3/depth/{(edit_idx):05d}.exr"), "Z")
                elif "vipe" == self.anno_src:
                    z_channel = imread(os.path.join(self.base_path, video, f"depth/{(edit_idx):05d}.exr"), "Z")
                z_channel[np.isnan(z_channel)] = 0
                z_channel[(z_channel > 1000) | np.isinf(z_channel)] = 1000
                depth_edit = torch.Tensor(z_channel).unsqueeze(0).unsqueeze(0)
                depth_edit = F.interpolate(depth_edit, size=(528, 960), mode='bilinear', align_corners=False)[0]
                # print(torch.max(depth_edit), torch.min(depth_edit))
                depth_latent = torch.zeros_like(depth_edit)
                depth = torch.cat([depth_latent, depth_edit], dim=0) # n, h, w
                if torch.isnan(depth).any():
                    print("!!!depth has NaN!!!")
                    print(video, edit_idx)
                    exit(0)

                # src_depth = np.array(depth_edit[0])
                # mx = np.max(src_depth)
                # mn = np.min(src_depth)
                # K = [  [meta["fl_x"] / image_width * 960,            0,  960 // 2],
                #     [              0, meta["fl_y"] / image_height * 528,  528 // 2],
                #     [              0,            0,           1]]
                # T = np.array(viewmats[0])
                # tgt_depth = self.transform_depth(
                #     src_depth,
                #     K=np.array(K),
                #     T=T
                # )
                # depth_latent = torch.Tensor(tgt_depth).unsqueeze(0)
                # depth = torch.cat([depth_latent, depth_edit], dim=0) # n, h, w

                data["depth"] = depth
                
                #if "val" in self.mode:
                #    tgt_depth = (tgt_depth / np.max(tgt_depth) * 255).astype(np.uint8) 
                #    src_depth = (src_depth / np.max(src_depth) * 255).astype(np.uint8) 

                #    z_channel = imread(os.path.join(self.base_path, video.split('/')[0], f"depth/{target_idx:05d}.exr"), "Z")
                #    depth_gt = torch.Tensor(z_channel).unsqueeze(0).unsqueeze(0)
                #    depth_gt = F.interpolate(depth_gt, size=(528, 960), mode='bilinear', align_corners=False)[0]
                #    depth_gt = np.array(depth_gt[0])
                #    depth_gt = (depth_gt / np.max(depth_gt) * 255).astype(np.uint8) 

                #    depth_vis = np.concatenate((src_depth, tgt_depth, depth_gt), axis=1)
                #    im = Image.fromarray(depth_vis)
                #    im.save(f"depth_vis/{index}_{mx:.2f}_{mn:.2f}.png")


        return data

    def getitem_metadata_DL3DV(self, index):

        csv = pd.read_csv(self.metadata_path)
        row = csv.iloc[index] 
        video = str(row['video'])
        edit_idx = int(row['edit_idx'])
        target_idx = int(row['target_idx'])

        data = {}

        with open(os.path.join(self.base_path, video, "transforms.json"), 'r', encoding='utf-8') as file:
            json_str = file.read()
            meta = json.loads(json_str)

            frames = sorted(os.listdir(os.path.join(self.base_path, video, "images_4")))
            import random

            edit_image = os.path.join(video, "images_4", frames[edit_idx])

            if self.anno_src is not None and "vipe-DA3" in self.anno_src:
                vipe_pose = np.load(os.path.join(self.base_path, video, "vipe-DA3/pose/video.npz"))
                edit_viewmats = vipe_pose["data"][edit_idx]
            elif self.anno_src is not None and self.anno_src == "vipe":
                vipe_pose = np.load(os.path.join(self.base_path, video, "pose/video.npz"))
                edit_viewmats = vipe_pose["data"][edit_idx]
            else:
                edit_viewmats = np.array(frames[edit_idx]["transform_matrix"], dtype=np.float32) @ blender2opencv # c2w! and invert y z axis! 
            edit_viewmats = torch.Tensor(edit_viewmats).unsqueeze(0)


            target_image = os.path.join(video, "images_4", frames[target_idx])
            if self.anno_src is not None and "vipe-DA3" in self.anno_src:
                vipe_pose = np.load(os.path.join(self.base_path, video, "vipe-DA3/pose/video.npz"))
                target_viewmats = vipe_pose["data"][target_idx]
            elif self.anno_src is not None and self.anno_src == "vipe":
                vipe_pose = np.load(os.path.join(self.base_path, video, "pose/video.npz"))
                target_viewmats = vipe_pose["data"][target_idx]
            else:
                target_viewmats = np.array(frames[target_idx]["transform_matrix"], dtype=np.float32) @ blender2opencv
            target_viewmats = torch.Tensor(target_viewmats).unsqueeze(0)
        
            edit_c2w = edit_viewmats
            target_c2w = target_viewmats
            in_c2ws = torch.cat([target_c2w, edit_c2w], dim=0)
            
            # normalize
            c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(edit_c2w[0]), in_c2ws)  # shift to src coord(edit_image)
            c2ws[:, :3, 3] /= self.norm_scale #20.0 # 10.0 # translation normalized

            #transform c2w to w2c align with PRoPE implementation
            viewmats = torch.linalg.inv(c2ws)

            s = 0
            ks = [  [meta["fl_x"],            s,  meta["cx"]],
                    [           0, meta["fl_y"],  meta["cy"]],
                    [           0,            0,           1]]
            ks = torch.Tensor(ks).unsqueeze(0)
            image_height = meta["h"]
            image_width = meta["w"]
            ks[..., 0, 0] = ks[..., 0, 0] / image_width
            ks[..., 1, 1] = ks[..., 1, 1] / image_height
            ks[..., 0, 2] = ks[..., 0, 2] / image_width - 0.5
            ks[..., 1, 2] = ks[..., 1, 2] / image_height - 0.5
            ks[..., 2, 2] = 1.0
            # ks has been normalized!!


            data["edit_image"] = self.main_data_operator(edit_image).resize((960, 528))
            data["image"] = self.main_data_operator(target_image).resize((960, 528))
            data["viewmats"] = viewmats
            data["Ks"] = torch.cat([ks, ks], dim=0)

            # print("original size : ",data["image"].size, data["edit_image"].size)
            if torch.isnan(data["viewmats"]).any() or torch.isnan(data["Ks"]).any():
                print("!!!camera param has NaN!!!")
                print(target_image, edit_image)
                exit(0)

            if "qwen" in self.base_model:
                data["prompt"] = "镜头视角转到指定位置"
            elif "flux" in self.base_model:
                data["prompt"] = "Turn to the target view"
            data["name"] = video

            if self.model_3D is not None and "val" not in self.mode:
                feat_3D = self.model_3D.inference(
                    [data["edit_image"].resize((960, 528))],                                                 # (1, 33, 60)
                    export_feat_layers=self.export_3D_feat_layers,   # (1, 20, 36, 1536) H, W, C  (1, 20, 36, 1024) 960 528
                    process_res=840,
                )
                feats = []
                for layer in self.export_3D_feat_layers:
                    feats.append(torch.from_numpy(feat_3D.aux[f"feat_layer_{layer}"]))
                data["feat_3D"] = torch.cat(feats, dim=-1) # (1, 20, 36, 1536) B, H, W, C
                if torch.isnan(data["feat_3D"]).any():
                    print("!!!feat 3D has NaN!!!")
                    print(video, edit_idx)
                    exit(0)
            
            target_z = imread(os.path.join(self.base_path, video, f"vipe-DA3/depth/{(target_idx):05d}.exr"), "Z")
            target_z[np.isnan(target_z)] = 1000
            target_z[(target_z > 1000) | np.isinf(target_z)] = 1000
            target_depth = torch.Tensor(target_z)
            data["target_depth"] = target_depth

            if self.add_depth:
                if "vipe-DA3" in self.anno_src:
                    z_channel = imread(os.path.join(self.base_path, video, f"vipe-DA3/depth/{(edit_idx):05d}.exr"), "Z")
                elif "vipe" == self.anno_src:
                    z_channel = imread(os.path.join(self.base_path, video, f"depth/{(edit_idx):05d}.exr"), "Z")
                z_channel[np.isnan(z_channel)] = 0
                z_channel[(z_channel > 1000) | np.isinf(z_channel)] = 1000
                depth_edit = torch.Tensor(z_channel).unsqueeze(0).unsqueeze(0)
                depth_edit = F.interpolate(depth_edit, size=(528, 960), mode='bilinear', align_corners=False)[0]
                # print(torch.max(depth_edit), torch.min(depth_edit))
                depth_latent = torch.zeros_like(depth_edit)
                depth = torch.cat([depth_latent, depth_edit], dim=0) # n, h, w
                if torch.isnan(depth).any():
                    print("!!!depth has NaN!!!")
                    print(video, edit_idx)
                    exit(0)

                data["depth"] = depth

        return data

    def getitem_metadata(self, index):
        csv = pd.read_csv(self.metadata_path)
        row = csv.iloc[index] 
        video = str(row['video'])
        edit_idx = int(row['edit_idx'])
        target_idx = int(row['target_idx'])

        vipe_pose = np.load(os.path.join(self.base_path, video, "vipe-DA3/pose/video.npz"))
        vipe_intr = np.load(os.path.join(self.base_path, video, "vipe-DA3/intrinsics/video.npz"))

        data = {}

        cap = cv2.VideoCapture(os.path.join(self.base_path, video, "video.mp4"))
        len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 12    # Re10K shot change
            
        import random

        interval = min(40, len_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, edit_idx)
        ret, f = cap.read()
        edit_image = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        edit_viewmats = vipe_pose["data"][edit_idx]
        edit_viewmats = torch.Tensor(edit_viewmats).unsqueeze(0)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, f = cap.read()
        target_image = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        target_viewmats = vipe_pose["data"][target_idx]
        target_viewmats = torch.Tensor(target_viewmats).unsqueeze(0)
    
        edit_c2w = edit_viewmats
        target_c2w = target_viewmats
        print(edit_idx, target_idx, edit_c2w.shape, target_c2w.shape)
        in_c2ws = torch.cat([target_c2w, edit_c2w], dim=0)
        
        # normalize
        c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(edit_c2w[0]), in_c2ws)  # shift to src coord(edit_image)
        c2ws[:, :3, 3] /= self.norm_scale # translation normalized
        
        cap.release()  # release video

        #transform c2w to w2c align with PRoPE implementation
        viewmats = torch.linalg.inv(c2ws)

        intri = vipe_intr["data"][edit_idx]
        s = 0
        ks = [  [intri[0],        s,  intri[2]],
                [       0, intri[1],  intri[3]],
                [       0,        0,        1.]]
        image_width = torch.tensor(intri[2]) * 2
        image_height = torch.tensor(intri[3]) * 2
        ks = torch.Tensor(ks).unsqueeze(0)
        ks[..., 0, 0] = ks[..., 0, 0] / image_width
        ks[..., 1, 1] = ks[..., 1, 1] / image_height
        ks[..., 0, 2] = ks[..., 0, 2] / image_width - 0.5
        ks[..., 1, 2] = ks[..., 1, 2] / image_height - 0.5


        data["edit_image"] = edit_image.resize((960, 528))
        data["image"] = target_image.resize((960, 528))
        data["viewmats"] = viewmats
        data["Ks"] = torch.cat([ks, ks], dim=0)
        # print(data["Ks"])

        if torch.isnan(data["viewmats"]).any() or torch.isnan(data["Ks"]).any():
            print("!!!camera param has NaN!!!")
            print(video, edit_idx, target_idx)
            exit(0)

        if "qwen" in self.base_model:
            data["prompt"] = "镜头视角转到指定位置"
        elif "flux" in self.base_model:
            data["prompt"] = "Turn to the target view"
        data["name"] = video

        if self.model_3D is not None and "val" not in self.mode:
            feat_3D = self.model_3D.inference(
                [data["edit_image"].resize((960, 528))],                                                 # (1, 33, 60)
                export_feat_layers=self.export_3D_feat_layers,   # (1, 20, 36, 1536) H, W, C  (1, 20, 36, 1024) 960 528
                process_res=840,
            )
            feats = []
            for layer in self.export_3D_feat_layers:
                feats.append(torch.from_numpy(feat_3D.aux[f"feat_layer_{layer}"]))
            data["feat_3D"] = torch.cat(feats, dim=-1)[0] # (20, 36, 1536) H, W, C
            if torch.isnan(data["feat_3D"]).any():
                print("!!!feat 3D has NaN!!!")
                print(video, edit_idx)
                exit(0)

        if self.add_depth:
            z_channel = imread(os.path.join(self.base_path, video, f"vipe-DA3/depth/{(edit_idx):05d}.exr"), "Z")
            z_channel[np.isnan(z_channel)] = 0
            z_channel[(z_channel > 1000) | np.isinf(z_channel)] = 1000
            depth_edit = torch.Tensor(z_channel).unsqueeze(0).unsqueeze(0)
            depth_edit = F.interpolate(depth_edit, size=(528, 960), mode='bilinear', align_corners=False)[0]
            # print(torch.max(depth_edit), torch.min(depth_edit))
            depth_latent = torch.zeros_like(depth_edit)
            depth = torch.cat([depth_latent, depth_edit], dim=0) # n, h, w
            if torch.isnan(depth).any():
                print("!!!depth has NaN!!!")
                print(video, edit_idx)
                exit(0)

            data["depth"] = depth


        return data

    def get_frame(self, video_path, frame_num):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            print(f'error video {video_path} {frame_num}')
            exit(0)

    def getitem_prope_all(self, index):
        video = self.videos[index % self.total_length]
        if not "Evaluation" in self.base_path:
            while not os.path.isdir(video):
                index += 1
                video = self.videos[index % self.total_length]

        vipe_pose = np.load(os.path.join(video, "vipe-DA3/pose/video.npz"))
        vipe_intr = np.load(os.path.join(video, "vipe-DA3/intrinsics/video.npz"))

        data = {}
        if 'DL3DV' in video:
            frames = sorted(os.listdir(os.path.join(video, "images_4")))
            len_frames = len(frames)
        else:
            cap = cv2.VideoCapture(os.path.join(video, "video.mp4"))
            len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 12    # Re10K shot change
            

        import random

        interval = min(40, len_frames - 1)
        extrinsics_check = False
        edit_idx = None
        while not extrinsics_check:
            if "val" in self.mode:
                edit_idx = 10
                target_idx = 30
            else:
                edit_idx = random.randint(0, len_frames - interval)
                max_idx = min(40, len_frames - edit_idx) # 0 -12
                target_idx = random.randint(edit_idx + max_idx // 2, edit_idx + max_idx - 1)
                # print(len_frames, edit_idx, target_idx) 

            if 'DL3DV' in video:    
                edit_image = os.path.join(video, "images_4", frames[edit_idx])
                edit_image = Image.open(edit_image)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, edit_idx)
                ret, f = cap.read()
                edit_image = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            edit_viewmats = vipe_pose["data"][edit_idx]
            edit_viewmats = torch.Tensor(edit_viewmats).unsqueeze(0)
            
            if 'DL3DV' in video:
                target_image = os.path.join(video, "images_4", frames[target_idx])
                target_image = Image.open(target_image)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, f = cap.read()
                target_image = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            target_viewmats = vipe_pose["data"][target_idx]
            target_viewmats = torch.Tensor(target_viewmats).unsqueeze(0)
        
            edit_c2w = edit_viewmats
            target_c2w = target_viewmats
            in_c2ws = torch.cat([target_c2w, edit_c2w], dim=0)
            
            # normalize
            c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(edit_c2w[0]), in_c2ws)  # shift to src coord(edit_image)
            c2ws[:, :3, 3] /= self.norm_scale # translation normalized
            
            if self.quick_check(c2ws[0:1, :, :]):
                extrinsics_check = True
                if 'DL3DV' not in video:
                    cap.release()  # release video
                break

        #transform c2w to w2c align with PRoPE implementation
        viewmats = torch.linalg.inv(c2ws)

        intri = vipe_intr["data"][edit_idx]
        s = 0
        ks = [  [intri[0],        s,  intri[2]],
                [       0, intri[1],  intri[3]],
                [       0,        0,        1.]]
        image_width = torch.tensor(intri[2]) * 2
        image_height = torch.tensor(intri[3]) * 2
        ks = torch.Tensor(ks).unsqueeze(0)
        ks[..., 0, 0] = ks[..., 0, 0] / image_width
        ks[..., 1, 1] = ks[..., 1, 1] / image_height
        ks[..., 0, 2] = ks[..., 0, 2] / image_width - 0.5
        ks[..., 1, 2] = ks[..., 1, 2] / image_height - 0.5

        # ks = [  [meta["fl_x"],            s,  meta["cx"]],
        #         [           0, meta["fl_y"],  meta["cy"]],
        #         [           0,            0,           1]]
        # ks = torch.Tensor(ks).unsqueeze(0)
        # image_height = meta["h"]
        # image_width = meta["w"]
        # ks[..., 0, 0] = ks[..., 0, 0] / image_width
        # ks[..., 1, 1] = ks[..., 1, 1] / image_height
        # ks[..., 0, 2] = ks[..., 0, 2] / image_width - 0.5
        # ks[..., 1, 2] = ks[..., 1, 2] / image_height - 0.5
        # ks[..., 2, 2] = 1.0
        # ks has been normalized!!


        data["edit_image"] = edit_image.resize((960, 528))
        data["image"] = target_image.resize((960, 528))
        data["viewmats"] = viewmats
        data["Ks"] = torch.cat([ks, ks], dim=0)
        # print(data["Ks"])

        if torch.isnan(data["viewmats"]).any() or torch.isnan(data["Ks"]).any():
            print("!!!camera param has NaN!!!")
            print(video, edit_idx, target_idx)
            exit(0)

        if "qwen" in self.base_model:
            data["prompt"] = "镜头视角转到指定位置"
        elif "flux" in self.base_model:
            data["prompt"] = "Turn to the target view"
        data["name"] = video

        if self.model_3D is not None and "val" not in self.mode:
            feat_3D = self.model_3D.inference(
                [data["edit_image"].resize((960, 528))],                                                 # (1, 33, 60)
                export_feat_layers=self.export_3D_feat_layers,   # (1, 20, 36, 1536) H, W, C  (1, 20, 36, 1024) 960 528
                process_res=840,
            )
            feats = []
            for layer in self.export_3D_feat_layers:
                feats.append(torch.from_numpy(feat_3D.aux[f"feat_layer_{layer}"]))
            data["feat_3D"] = torch.cat(feats, dim=-1)[0] # (20, 36, 1536) H, W, C
            if torch.isnan(data["feat_3D"]).any():
                print("!!!feat 3D has NaN!!!")
                print(video, edit_idx)
                exit(0)
        
        # target_z = imread(os.path.join(self.base_path, video, f"vipe-DA3/depth/{(target_idx):05d}.exr"), "Z")
        # target_z[np.isnan(target_z)] = 1000
        # target_z[(target_z > 1000) | np.isinf(target_z)] = 1000
        # target_depth = torch.Tensor(target_z)
        # data["target_depth"] = target_depth

        if self.add_depth:
            z_channel = imread(os.path.join(video, f"vipe-DA3/depth/{(edit_idx):05d}.exr"), "Z")
            z_channel[np.isnan(z_channel)] = 0
            z_channel[(z_channel > 1000) | np.isinf(z_channel)] = 1000
            depth_edit = torch.Tensor(z_channel).unsqueeze(0).unsqueeze(0)
            depth_edit = F.interpolate(depth_edit, size=(528, 960), mode='bilinear', align_corners=False)[0]
            # print(torch.max(depth_edit), torch.min(depth_edit))
            depth_latent = torch.zeros_like(depth_edit)
            depth = torch.cat([depth_latent, depth_edit], dim=0) # n, h, w
            if torch.isnan(depth).any():
                print("!!!depth has NaN!!!")
                print(video, edit_idx)
                exit(0)

            data["depth"] = depth


        return data

    def __len__(self):
        if self.prope:
            return self.total_length * self.repeat
        

if __name__ == '__main__':

    metadata_path = '../RealEstate10K/meta_view/hard.csv'
    base_path = "../RealEstate10K/JiaHWang/Re10K/test"
    # 48aaed5a44005bccd51d529ab90335b144fe5e7f3c8a22ba399f4ee3b3fb6728 
    dataset = MetaViewUnifiedDataset(
        base_path=base_path,
        metadata_path=metadata_path,
        # subset=['1K', '2K', '3K', '4K', '5K', '6K', '7K', '9K', '10K', '11K', 'train', 'Sekai-Real-Walking-HQ-split'], 
        repeat=1,
        data_file_keys="image,edit_image".split(","),
        prope=True,
        debug=False,
        mode="train",
        norm_scale=1.0,
        # path_3D="../Depth-Anything-3/model/DA3-GIANT-1.1",
        # export_3D_feat_layers="19,39",
        anno_src="vipe-DA3",
        add_depth=True,
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=base_path,
            max_pixels=1048576,
            height=None,
            width=None,
            height_division_factor=16,
            width_division_factor=16,
        )
    )

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=1)
    print(len(dataset))
    cnt = 0
    for data in dataloader:
        # print(data["name"])
        cnt += 1
        print(data['name'])
        #if cnt > 11:
        #    exit(0)
    print(cnt)