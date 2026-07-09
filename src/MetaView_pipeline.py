# Adapted from https://github.com/modelscope/DiffSynth-Studio

import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np

from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput

from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.models.qwen_image_vae import QwenImageVAE
from diffsynth.models.qwen_image_controlnet import QwenImageBlockWiseControlNet

from src.PRoPE import PropeDotProductAttention
from src.MetaView_dit import MetaViewDiT

import torch.nn.functional as F

class MetaViewPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        from transformers import Qwen2Tokenizer, Qwen2VLProcessor
        
        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: MetaViewDiT = None
        self.vae: QwenImageVAE = None
        self.blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None
        self.tokenizer: Qwen2Tokenizer = None
        self.processor: Qwen2VLProcessor = None
        self.in_iteration_models = ("dit", "blockwise_controlnet")
        self.units = [
            MetaViewUnit_ShapeChecker(),
            MetaViewUnit_NoiseInitializer(),
            MetaViewUnit_InputImageEmbedder(),
            MetaViewUnit_EditImageEmbedder(),
            MetaViewUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_MetaView
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = MetaViewPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("metaview_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        pipe.blockwise_controlnet = QwenImageBlockwiseMultiControlNet(model_pool.fetch_model("qwen_image_blockwise_controlnet", index="all"))
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Inpaint
        inpaint_mask: Image.Image = None,
        inpaint_blur_size: int = None,
        inpaint_blur_sigma: float = None,
        # Shape
        height: int = 1328,
        width: int = 1328,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        exponential_shift_mu: float = None,
        # Blockwise ControlNet
        blockwise_controlnet_inputs: list[ControlNetInput] = None,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        # Qwen-Image-Edit
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        edit_rope_interpolation: bool = False,
        # In-context control
        context_image: Image.Image = None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
        # added prope
        viewmats = None,  # [b, 2, 4, 4] order (target, edit)
        Ks = None,  # [b, 2, 3, 3]
        prope_dim_arrange = [16, 56, 56],
        add_attn = True,
        add_3D = False,
        feat_3D = None,
        depth = None,
        merge_3D = False,
        val = False,
        batch_size = 1,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=(height // 16) * (width // 16), exponential_shift_mu=exponential_shift_mu)
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": [negative_prompt],
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "inpaint_mask": inpaint_mask, "inpaint_blur_size": inpaint_blur_size, "inpaint_blur_sigma": inpaint_blur_sigma,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "blockwise_controlnet_inputs": blockwise_controlnet_inputs,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize, "edit_rope_interpolation": edit_rope_interpolation, 
            "context_image": context_image,
            # add camera param
            "viewmats": viewmats,
            "Ks": Ks,
            "prope_dim_arrange": prope_dim_arrange,
            "add_attn": add_attn,
            "add_3D": add_3D,
            "feat_3D": feat_3D,
            "depth": depth,
            "merge_3D": merge_3D,
            "val": val,
            "batch_size": batch_size,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
            # print(inputs_shared["latents"])

        # Decode
        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class QwenImageBlockwiseMultiControlNet(torch.nn.Module):
    def __init__(self, models: list[QwenImageBlockWiseControlNet]):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = torch.nn.ModuleList(models)
        for model in models:
            if hasattr(model, "vram_management_enabled") and getattr(model, "vram_management_enabled"):
                self.vram_management_enabled = True

    def preprocess(self, controlnet_inputs: list[ControlNetInput], conditionings: list[torch.Tensor], **kwargs):
        processed_conditionings = []
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            conditioning = rearrange(conditioning, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
            model_output = self.models[controlnet_input.controlnet_id].process_controlnet_conditioning(conditioning)
            processed_conditionings.append(model_output)
        return processed_conditionings

    def blockwise_forward(self, image, conditionings: list[torch.Tensor], controlnet_inputs: list[ControlNetInput], progress_id, num_inference_steps, block_id, **kwargs):
        res = 0
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            progress = (num_inference_steps - 1 - progress_id) / max(num_inference_steps - 1, 1)
            if progress > controlnet_input.start + (1e-4) or progress < controlnet_input.end - (1e-4):
                continue
            model_output = self.models[controlnet_input.controlnet_id].blockwise_forward(image, conditioning, block_id)
            res = res + model_output * controlnet_input.scale
        return res


class MetaViewUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: MetaViewPipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class MetaViewUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "batch_size"),
            output_params=("noise",),
        )

    def process(self, pipe: MetaViewPipeline, height, width, seed, rand_device, batch_size):
        noise = pipe.generate_noise((batch_size, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}



class MetaViewUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: MetaViewPipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])

        if isinstance(input_image, list):
            input_latents = []
            for input_img in input_image:
                img = pipe.preprocess_image(input_img).to(device=pipe.device, dtype=pipe.torch_dtype)
                input_latent = pipe.vae.encode(img, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                input_latents.append(input_latent)
            input_latents = torch.cat(input_latents, dim=0) # B C H W
        else:
            # single PIL img, ret [1, c, h, w]
            image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        assert noise.shape[0] == input_latents.shape[0]

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class MetaViewUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "tiled", "tile_size", "tile_stride", "edit_image_auto_resize"),
            output_params=("edit_latents", "edit_image"),
            onload_model_names=("vae",)
        )


    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height


    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        return edit_image.resize((calculated_width, calculated_height))


    def process(self, pipe: MetaViewPipeline, edit_image, tiled, tile_size, tile_stride, edit_image_auto_resize=False):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, Image.Image):
            # resized_edit_image = self.edit_image_auto_resize(edit_image) if edit_image_auto_resize else edit_image
            resized_edit_image = edit_image # skip resize
            edit_image = pipe.preprocess_image(resized_edit_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            edit_latents = pipe.vae.encode(edit_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        else:
            resized_edit_image, edit_latents = [], []
            for image in edit_image:
                # if edit_image_auto_resize:
                #     image = self.edit_image_auto_resize(image)
                resized_edit_image.append(image)
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                edit_latents.append(latents)
            edit_latents = torch.cat(edit_latents, dim=0) # B C H W
        return {"edit_latents": edit_latents, "edit_image": resized_edit_image}



class MetaViewUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )
        
    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def calculate_dimensions(self, target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def resize_image(self, image, target_area=384*384):
        width, height = self.calculate_dimensions(target_area, image.size[0] / image.size[1])
        return image.resize((width, height))
    
    def encode_prompt(self, pipe: MetaViewPipeline, prompt):
        template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.tokenizer(txt, max_length=4096+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
        if model_inputs.input_ids.shape[1] >= 1024:
            print(f"Warning!!! QwenImage model was trained on prompts up to 512 tokens. Current prompt requires {model_inputs['input_ids'].shape[1] - drop_idx} tokens, which may lead to unpredictable behavior.")
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
        
    def encode_prompt_edit(self, pipe: MetaViewPipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        # print(txt)
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
    
    def encode_prompt_edit_batch(self, pipe: MetaViewPipeline, prompt, edit_image):
        # list batch
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        split_hidden_states_list = []
        for i in range(len(prompt)):
            model_inputs = pipe.processor(text=[txt[i]], images=[edit_image[i]], padding=True, return_tensors="pt").to(pipe.device)
            hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
            split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask) #tuple (1)
            # print(type(split_hidden_states[0]))
            # print(len(split_hidden_states[0]))
            split_hidden_states_list.append(split_hidden_states[0])
        
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states_list]

        return split_hidden_states
    
    def encode_prompt_edit_multi(self, pipe: MetaViewPipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        base_img_prompt = "".join([img_prompt_template.format(i + 1) for i in range(len(edit_image))])
        txt = [template.format(base_img_prompt + e) for e in prompt]
        edit_image = [self.resize_image(image) for image in edit_image]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states

    def process(self, pipe: MetaViewPipeline, prompt, edit_image=None) -> dict:
        #prompt [n] str list
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is not None:
            # prompt = [prompt]
            if edit_image is None:
                split_hidden_states = self.encode_prompt(pipe, prompt)
            elif isinstance(edit_image, Image.Image):
                split_hidden_states = self.encode_prompt_edit(pipe, prompt, edit_image)
            elif isinstance(edit_image, list): # batch
                split_hidden_states = self.encode_prompt_edit_batch(pipe, prompt, edit_image)
            # else:
            #     split_hidden_states = self.encode_prompt_edit_multi(pipe, prompt, edit_image)
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
        else:
            return {}



def model_fn_MetaView(
    dit: MetaViewDiT = None,
    blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    height=None,
    width=None,
    blockwise_controlnet_conditioning=None,
    blockwise_controlnet_inputs=None,
    progress_id=0,
    num_inference_steps=1,
    entity_prompt_emb=None,
    entity_prompt_emb_mask=None,
    entity_masks=None,
    edit_latents=None,
    context_latents=None,
    enable_fp8_attention=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    edit_rope_interpolation=False,
    viewmats=None,  # camera param
    Ks=None,
    feat_3D=None,
    prope_dim_arrange=None,
    add_attn=False,
    add_3D=False,
    depth=None,
    merge_3D=False,
    decode_3D=False,
    val=False,
    **kwargs
):
    img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)]
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
    timestep = timestep / 1000
    
    image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
    image_seq_len = image.shape[1]


    if edit_latents is not None:    # only single edit imgß
        e = edit_latents    # B C H W
        img_shapes += [(1, e.shape[2]//2, e.shape[3]//2)]
        edit_image = [rearrange(e, "B C (H P) (W Q) -> B (H W) (C P Q)", H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2)]
        image = torch.cat([image] + edit_image, dim=1)

    # print(img_shapes)
    # print(image.shape)
    # print(prompt_emb.shape)
    # print(txt_seq_lens)
    
    # order tgt(latent, gt), src(edit_image ref)
    # resize to 1024*1024
    # print("image ",image.shape)  # ([1, 8184 (62 * 66 * 2), 64])
    # print("latents ",latents.shape)     #[1, 16, 124, 132]
    
    # [(1, 33, 60), (1, 33, 60)]
    # 960 528
    # [(1, 33, 60), (1, 33, 60)]
    # 960 528
    # print(img_shapes)   # (1, 62, 66), (1, 62, 66)
    # print(width, height)

    image = dit.img_in(image)
    conditioning = dit.time_text_embed(timestep, image.dtype)

    text = dit.txt_in(dit.txt_norm(prompt_emb))
    if edit_rope_interpolation:
        image_rotary_emb = dit.pos_embed.forward_sampling(img_shapes, txt_seq_lens, device=latents.device)
    else:
        image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        # add prope
        if viewmats is not None:
            if depth is not None: # b n h w
                depth = F.interpolate(depth, size=(height // 16, width // 16), mode='bilinear', align_corners=False)
                depth = depth.to(image.device)
                # print("depth:", depth.shape)
                # print("image:", image.shape)

                # depth_np = depth[0, 1].detach().to(torch.float).cpu().numpy()
                # depth_min, depth_max = depth_np.min(), depth_np.max()
                # depth_norm = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
                # depth_norm = depth_norm.astype(np.uint8)
                # depth_save = Image.fromarray(depth_norm, 'L')
                # depth_save.save(f"tmp/{depth_max}.png")


            dit.PRoPE = PropeDotProductAttention(
                head_dim=128,
                patches_x=width // 16,
                patches_y=height // 16,
                image_width=width,
                image_height=height,
                freq_base=10000,        #TODO 100?
                dim_arrange=prope_dim_arrange,
            )
            dit.PRoPE = dit.PRoPE.to(image.device)
            dit.PRoPE._precompute_and_cache_apply_fns(viewmats.to(image.device), Ks.to(image.device), depth) #  b, frames, h, w

            if feat_3D is not None:
                dit.add_PRoPE = PropeDotProductAttention(
                    head_dim=128,
                    patches_x=width // 16,
                    patches_y=height // 16,
                    image_width=width,
                    image_height=height,
                    freq_base=10000,
                    dim_arrange=prope_dim_arrange,
                )
                dit.add_PRoPE = dit.add_PRoPE.to(image.device)
                if depth is not None:
                    dit.add_PRoPE._precompute_and_cache_apply_fns(viewmats[:, 1:2, :, :].to(image.device), Ks[:, 1:2, :, :].to(image.device), depth[:, 1:2, :, :])
                else:
                    dit.add_PRoPE._precompute_and_cache_apply_fns(viewmats[:, 1:2, :, :].to(image.device), Ks[:, 1:2, :, :].to(image.device))
    attention_mask = None

    if feat_3D is not None:
        h_3D, w_3D = feat_3D.shape[1], feat_3D.shape[2]
        feat_3D = rearrange(feat_3D, 'b h w d -> b (h w) d')
    
    if merge_3D:
        feat_3D = dit._3D_in(feat_3D)

    for block_id, block in enumerate(dit.transformer_blocks):
        if merge_3D:
            text, image, feat_3D = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
                prope=dit.PRoPE,  # prope
                add_prope=dit.add_PRoPE,
                add_attn=add_attn,
                feat_3D=feat_3D,
                block_id=block_id, 
            )
        else:
            text, image = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
                prope=dit.PRoPE,  # prope
                add_prope=dit.add_PRoPE,
                add_attn=add_attn,
                feat_3D=feat_3D,
                block_id=block_id, 
            )

    image = dit.norm_out(image, conditioning)
    image = dit.proj_out(image)
    image = image[:, :image_seq_len]

    latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)

    if val:
        return latents

    if decode_3D:
        feat_3D = dit.norm_3D_out(feat_3D)
        feat_3D = dit.proj_3D_out(feat_3D)
        latents_3D = feat_3D.unsqueeze(0).unsqueeze(0)
        latents_3D = list(torch.chunk(latents_3D, chunks=4, dim=-1))
        return latents, latents_3D

    return latents, None
