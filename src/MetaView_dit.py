# Adapted from https://github.com/modelscope/DiffSynth-Studio

import torch, math
import torch.nn as nn
from typing import Tuple, Optional, Union, List
from einops import rearrange
import matplotlib.pyplot as plt
from diffsynth.models.general_modules import TimestepEmbeddings, RMSNorm, AdaLayerNorm

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

from src.PRoPE import PropeDotProductAttention
from src.lora import LoRALinearLayer

def visualize_attention_heads(attn_probs, name=None, layer=None):
    seq_len, _ = attn_probs.shape
    
    plt.figure(figsize=(4, 3))
    
    attn_map = attn_probs.clone().to(torch.float32).detach().cpu().numpy()
    
    im = plt.imshow(attn_map, cmap='viridis', aspect='auto', vmin=0, vmax=0.005)
    plt.title(f'Attention Maps {name} layer {layer}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    #cbar.set_ticks([0, 0.001, 0.002, 0.003, 0.004, 0.005])
    #cbar.set_ticklabels(['0.0', '0.001', '0.002', '0.003', '0.004', '0.005']) 
    # plt.tight_layout()
    
    save_path = f"./attn_maps/attn_map_merge/{name}_layer_{layer:02d}"
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')



def qwen_image_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attention_mask = None, enable_fp8_attention: bool = False, seq_order=None, layer=None):
    if FLASH_ATTN_3_AVAILABLE and attention_mask is None:
        print("flash attn!!!")
        if not enable_fp8_attention:
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v)
            if isinstance(x, tuple):
                x = x[0]
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            origin_dtype = q.dtype
            q_std, k_std, v_std = q.std(), k.std(), v.std()
            q, k, v = (q / q_std).to(torch.float8_e4m3fn), (k / k_std).to(torch.float8_e4m3fn), (v / v_std).to(torch.float8_e4m3fn)
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=q_std * k_std / math.sqrt(q.size(-1)))
            if isinstance(x, tuple):
                x = x[0]
            x = x.to(origin_dtype) * v_std
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)

    if 0 and seq_order is not None and layer % 10 ==0:
        img_seq = seq_order[0] // 2
        _3D_seq = seq_order[1]
        # print(seq_order)

        batch_size, _, seq_len, head_dim = q.shape

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)       # b n s s
        # print(attn_probs.shape)
        # lantent_attn_map = attn_probs[0, 0, :img_seq, :]
        # src_attn_map = attn_probs[0, 0, img_seq : 2*img_seq, :]
        # _3D_attn_map = attn_probs[0, 0, 2*img_seq: , :] # n s s

        lantent_attn_map = attn_probs[0, :, :img_seq, :]
        src_attn_map = attn_probs[0, :, img_seq : 2*img_seq, :]
        _3D_attn_map = attn_probs[0, :, 2*img_seq: , :] # n s s

        lantent_attn_map = torch.mean(lantent_attn_map, dim=0)  # s s
        src_attn_map = torch.mean(src_attn_map, dim=0)
        _3D_attn_map = torch.mean(_3D_attn_map, dim=0)

        visualize_attention_heads(lantent_attn_map, name="latent", layer=layer)
        visualize_attention_heads(src_attn_map, name="src", layer=layer)
        visualize_attention_heads(_3D_attn_map, name="3D", layer=layer)
    
    return x


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)

def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
):
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.rope_cache = {}
        self.scale_rope = scale_rope
        
    def rope_params(self, index, dim, theta=10000):
        """
            Args:
                index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


    def _expand_pos_freqs_if_needed(self, video_fhw, txt_seq_lens):
        if isinstance(video_fhw, list):
            video_fhw = tuple(max([i[j] for i in video_fhw]) for j in range(3))
        _, height, width = video_fhw
        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)
        required_len = max_vid_index + max(txt_seq_lens)
        cur_max_len = self.pos_freqs.shape[0]
        if required_len <= cur_max_len:
            return

        new_max_len = math.ceil(required_len / 512) * 512
        pos_index = torch.arange(new_max_len)
        neg_index = torch.arange(new_max_len).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        return


    def forward(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


    def forward_sampling(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if idx > 0 and f"{0}_{height}_{width}" not in self.rope_cache:
                frame_0, height_0, width_0 = video_fhw[0]

                rope_key_0 = f"0_{height_0}_{width_0}"
                spatial_freqs_0 = self.rope_cache[rope_key_0].reshape(frame_0, height_0, width_0, -1)
                h_indices = torch.linspace(0, height_0 - 1, height).long()
                w_indices = torch.linspace(0, width_0 - 1, width).long()
                h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
                sampled_rope = spatial_freqs_0[:, h_grid, w_grid, :]

                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                sampled_rope[:, :, :, :freqs_frame.shape[-1]] = freqs_frame

                seq_lens = frame * height * width
                self.rope_cache[rope_key] = sampled_rope.reshape(seq_lens, -1).clone()
            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone()
            vid_freqs.append(self.rope_cache[rope_key].contiguous())

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


class QwenFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(ApproximateGELU(dim, inner_dim))
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MetaViewSelfAttention3D(nn.Module):
    def __init__(
        self,
        dim_a,  # 3072
        dim_b,
        num_heads,
        head_dim,
        merge_3D = False,
    ):
        super().__init__()
        self.merge_3D = merge_3D

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.to_q_3D = nn.Linear(dim_b, dim_a)
        self.to_k_3D = nn.Linear(dim_b, dim_a)
        self.to_v_3D = nn.Linear(dim_b, dim_a)
        
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        # zero init linear out
        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(
        self,
        image: torch.FloatTensor,
        feat_3D: torch.FloatTensor, # (1, 20, 36, 1536)
        attention_mask: Optional[torch.FloatTensor] = None,
        enable_fp8_attention: bool = False,
        prope: Optional[PropeDotProductAttention] = None,
        add_prope: Optional[PropeDotProductAttention] = None,
        block_id=None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # feat_3D = rearrange(feat_3D, 'b h w d -> b (h w) d')

        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        _3D_q, _3D_k, _3D_v = self.to_q_3D(feat_3D), self.to_k_3D(feat_3D), self.to_v_3D(feat_3D)
        seq_img = img_q.shape[1]
        seq_3D = _3D_q.shape[1]

        img_q = rearrange(img_q, 'b s (h d) -> b h s d', h=self.num_heads)
        img_k = rearrange(img_k, 'b s (h d) -> b h s d', h=self.num_heads)
        img_v = rearrange(img_v, 'b s (h d) -> b h s d', h=self.num_heads)
        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)

        _3D_q = rearrange(_3D_q, 'b s (h d) -> b h s d', h=self.num_heads)
        _3D_k = rearrange(_3D_k, 'b s (h d) -> b h s d', h=self.num_heads)
        _3D_v = rearrange(_3D_v, 'b s (h d) -> b h s d', h=self.num_heads)
        _3D_q, _3D_k = self.norm_added_q(_3D_q), self.norm_added_k(_3D_k)
        
        if prope is not None: # standard prope in q k v out 
            # print("Inject Prope in q k v out in added attn layer! ")
            img_q = prope._apply_to_q(img_q)
            img_k = prope._apply_to_kv(img_k)
            img_v = prope._apply_to_kv(img_v)

            _3D_q = add_prope._apply_to_q(_3D_q)
            _3D_k = add_prope._apply_to_kv(_3D_k)
            _3D_v = add_prope._apply_to_kv(_3D_v)

        joint_q = torch.cat([img_q, _3D_q], dim=2)
        joint_k = torch.cat([img_k, _3D_k], dim=2)
        joint_v = torch.cat([img_v, _3D_v], dim=2)

        num_heads = img_q.shape[1]
        joint_attn_out = qwen_image_flash_attention(joint_q, joint_k, joint_v, num_heads=num_heads, layer=block_id, seq_order=[seq_img, seq_3D], attention_mask=attention_mask, enable_fp8_attention=enable_fp8_attention).to(img_q.dtype)
        # [b, s, (n d)]

        img_attn_output = joint_attn_out[:, :seq_img, :]    # discard 3D tokens
        _3D_attn_output = joint_attn_out[:, seq_img:, :]

        # reshape back and apply prope
        if prope is not None:
            img_attn_output = rearrange(img_attn_output, "b s (n d) -> b n s d", n=num_heads)
            img_attn_output = prope._apply_to_o(img_attn_output)
            img_attn_output = rearrange(img_attn_output, "b n s d -> b s (n d) ", n=num_heads)

        img_attn_output = self.to_out(img_attn_output)
        
        if add_prope is not None and self.merge_3D:
            _3D_attn_output = rearrange(_3D_attn_output, "b s (n d) -> b n s d", n=num_heads)
            _3D_attn_output = add_prope._apply_to_o(_3D_attn_output)
            _3D_attn_output = rearrange(_3D_attn_output, "b n s d -> b s (n d) ", n=num_heads)
        
        if self.merge_3D:
            _3D_attn_output = self.to_out(_3D_attn_output)

        if self.merge_3D:
            return img_attn_output, _3D_attn_output
        else:
            return img_attn_output, None


class QwenDoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        enable_fp8_attention: bool = False,
        prope: Optional[PropeDotProductAttention] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, 'b s (h d) -> b h s d', h=self.num_heads)
        img_k = rearrange(img_k, 'b s (h d) -> b h s d', h=self.num_heads)
        img_v = rearrange(img_v, 'b s (h d) -> b h s d', h=self.num_heads)

        txt_q = rearrange(txt_q, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_k = rearrange(txt_k, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_v = rearrange(txt_v, 'b s (h d) -> b h s d', h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)
        
        if prope is not None and image_rotary_emb is not None: # prope with original Qwen backbone
            # print("!! modified PRoPE in original Qwen backbone")
            _, txt_freqs = image_rotary_emb
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)
            img_q = prope._apply_to_q(img_q)
            img_k = prope._apply_to_kv(img_k)
            # img_v = prope._apply_to_kv(img_v)
        elif image_rotary_emb is not None:
            # print("!!! No PRoPE in original Qwen backbone !!")
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        num_heads = joint_q.shape[1]
        joint_attn_out = qwen_image_flash_attention(joint_q, joint_k, joint_v, num_heads=joint_q.shape[1], attention_mask=attention_mask, enable_fp8_attention=enable_fp8_attention).to(joint_q.dtype)
        # [b, s, (n d)]

        txt_attn_output = joint_attn_out[:, :seq_txt, :]
        img_attn_output = joint_attn_out[:, seq_txt:, :]

        # reshape back and apply prope
        # if prope is not None:
        #     img_attn_output = rearrange(img_attn_output, "b s (n d) -> b n s d", n=num_heads)
        #     img_attn_output = prope._apply_to_o(img_attn_output)
        #     img_attn_output = rearrange(img_attn_output, "b n s d -> b s (n d) ", n=num_heads)

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class MetaViewTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_attention_heads: int, 
        attention_head_dim: int, 
        eps: float = 1e-6,
        add_attn = False,
        add_attn_type = None,
        add_in_dim = None,
        lora_rank = None,
        merge_3D = False,
    ):    
        super().__init__()
        
        self.merge_3D = merge_3D

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim), 
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )
        if add_attn:
            self.prope_attn = MetaViewSelfAttention3D(
                dim_a=dim,
                dim_b=add_in_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                merge_3D=merge_3D,
            )


        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenFeedForward(dim=dim, dim_out=dim)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True), 
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim)
    
    def _modulate(self, x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)    

    def forward(
        self,
        image: torch.Tensor,  
        text: torch.Tensor,
        temb: torch.Tensor, 
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_fp8_attention = False,
        prope: Optional[PropeDotProductAttention] = None,
        add_prope: Optional[PropeDotProductAttention] = None,
        add_attn = False,
        feat_3D = None, # (1, 20, 36, 1536)
        block_id = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("feat_3D:", feat_3D.shape)
        # print("image: ", image.shape)
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        if self.merge_3D:   # ref to edit image (no noise), uncond by timestep
            # feat_3D_mod_attn, feat_3D_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
            feat_3D_normed = self.img_norm1(feat_3D)
            # feat_3D_modulated, feat_3D_gate = self._modulate(feat_3D_normed, feat_3D_mod_attn)
            feat_3D_modulated = feat_3D_normed
        else:
            feat_3D_modulated = feat_3D


        if add_attn and prope is not None: 
            img_prope_out, _3D_prope_out = self.prope_attn( # self_attn_3D
                image=img_modulated,
                feat_3D=feat_3D_modulated,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
                prope=prope,
                add_prope=add_prope,
                block_id=block_id,
            )

            img_attn_out, txt_attn_out = self.attn(
                image=img_modulated,
                text=txt_modulated,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
                prope=None,
            ) 
            img_attn_out = img_attn_out + img_prope_out
        elif not add_attn or prope is None:
            img_attn_out, txt_attn_out = self.attn(
                image=img_modulated,
                text=txt_modulated,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
                prope=prope,
            )
        
        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        if self.merge_3D:   # ref to edit image (no noise), uncond by timestep
            feat_3D = feat_3D + _3D_prope_out
            feat_3D_normed_2 = self.img_norm2(feat_3D)
            feat_3D_mlp_out = self.img_mlp(feat_3D_normed_2)
            feat_3D = feat_3D + feat_3D_mlp_out

            return text, image, feat_3D
        else:
            return text, image


class MetaViewDiT(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        add_attn_type=None,
        add_in_dim=None,
        lora_rank=None,
        merge_3D=False,
        decode_3D=False,
        _3d_dim=None,
    ):
        super().__init__()
        
        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16,56,56], scale_rope=True) # only for text embed

        self.time_text_embed = TimestepEmbeddings(256, 3072, diffusers_compatible_format=True, scale=1000, align_dtype_to_timestep=True)
        self.txt_norm = RMSNorm(3584, eps=1e-6)

        self.img_in = nn.Linear(64, 3072)
        self.txt_in = nn.Linear(3584, 3072)

        self.transformer_blocks = nn.ModuleList(
            [
                MetaViewTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                    ## add args
                    add_attn=True,
                    add_attn_type=add_attn_type,
                    add_in_dim=add_in_dim,
                    lora_rank=lora_rank,
                    merge_3D=merge_3D,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, single=True)
        self.proj_out = nn.Linear(3072, 64)

        if merge_3D:        
            self._3D_in = nn.Linear(_3d_dim, add_in_dim)
            if decode_3D:
                self.norm_3D_out = nn.LayerNorm(add_in_dim, elementwise_affine=False, eps=1e-6)
                self.proj_3D_out = nn.Linear(add_in_dim, _3d_dim)

        self.PRoPE = None
        self.add_PRoPE = None
        self.add_attn = True


    def process_entity_masks(self, latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask, entity_masks, height, width, image, img_shapes):
        # prompt_emb
        all_prompt_emb = entity_prompt_emb + [prompt_emb]
        all_prompt_emb = [self.txt_in(self.txt_norm(local_prompt_emb)) for local_prompt_emb in all_prompt_emb]
        all_prompt_emb = torch.cat(all_prompt_emb, dim=1)

        # image_rotary_emb
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        entity_seq_lens = [emb_mask.sum(dim=1).tolist() for emb_mask in entity_prompt_emb_mask]
        entity_rotary_emb = [self.pos_embed(img_shapes, entity_seq_len, device=latents.device)[1] for entity_seq_len in entity_seq_lens]
        txt_rotary_emb = torch.cat(entity_rotary_emb + [image_rotary_emb[1]], dim=0)
        image_rotary_emb = (image_rotary_emb[0], txt_rotary_emb)

        # attention_mask
        repeat_dim = latents.shape[1]
        max_masks = entity_masks.shape[1]
        entity_masks = entity_masks.repeat(1, 1, repeat_dim, 1, 1)
        entity_masks = [entity_masks[:, i, None].squeeze(1) for i in range(max_masks)]
        global_mask = torch.ones_like(entity_masks[0]).to(device=latents.device, dtype=latents.dtype)
        entity_masks = entity_masks + [global_mask]

        N = len(entity_masks)
        batch_size = entity_masks[0].shape[0]
        seq_lens = [mask_.sum(dim=1).item() for mask_ in entity_prompt_emb_mask] + [prompt_emb_mask.sum(dim=1).item()]
        total_seq_len = sum(seq_lens) + image.shape[1]
        patched_masks = []
        for i in range(N):
            patched_mask = rearrange(entity_masks[i], "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
            patched_masks.append(patched_mask)
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool).to(device=entity_masks[0].device)

        # prompt-image attention mask
        image_start = sum(seq_lens)
        image_end = total_seq_len
        cumsum = [0]
        single_image_seq = image_end - image_start
        for length in seq_lens:
            cumsum.append(cumsum[-1] + length)
        for i in range(N):
            prompt_start = cumsum[i]
            prompt_end = cumsum[i+1]
            image_mask = torch.sum(patched_masks[i], dim=-1) > 0
            image_mask = image_mask.unsqueeze(1).repeat(1, seq_lens[i], 1)
            # repeat image mask to match the single image sequence length
            repeat_time = single_image_seq // image_mask.shape[-1]
            image_mask = image_mask.repeat(1, 1, repeat_time)
            # prompt update with image
            attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask
            # image update with prompt
            attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask.transpose(1, 2)
        # prompt-prompt attention mask, let the prompt tokens not attend to each other
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                start_i, end_i = cumsum[i], cumsum[i+1]
                start_j, end_j = cumsum[j], cumsum[j+1]
                attention_mask[:, start_i:end_i, start_j:end_j] = False

        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = float('-inf')
        attention_mask[attention_mask == 1] = 0
        attention_mask = attention_mask.to(device=latents.device, dtype=latents.dtype).unsqueeze(1)

        return all_prompt_emb, image_rotary_emb, attention_mask


    def forward(
        self,
        latents=None,
        timestep=None,
        prompt_emb=None,
        prompt_emb_mask=None,
        height=None,
        width=None,
    ):
        img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
        # print(latentimg_shapes)
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        
        image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(prompt_emb))

        conditioning = self.time_text_embed(timestep, image.dtype)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

        for block in self.transformer_blocks:
            text, image = block(
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
            )
        
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)
        
        latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)
        return image
