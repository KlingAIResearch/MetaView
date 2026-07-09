# MIT License
#
# Adapted from the official implementation of PRoPE
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F

from einops import rearrange


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 10000.0, # qwen 10000
        freq_scale: float = 1.0,
        dim_arrange = [16, 56, 56], # (frame ,height, width) default for qwen. 
        depth = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.freq_base = freq_base
        self.freq_scale = freq_scale

        self.use_PRoPE = False
        self.dim_arrange = dim_arrange

        # fit Qwen scale-rope
        pos_index_x = torch.arange(patches_x)
        neg_index_x = torch.arange(patches_x).flip(0) * -1 - 1
        index_x = torch.cat([neg_index_x[-(patches_x - patches_x // 2) :], pos_index_x[: patches_x // 2]], dim=0)
        # print(index_x)
        ## Qwen rope apply order is frame, height, width!
        # coeffs_x
        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(  #
            # torch.tile(torch.arange(patches_x), (patches_y,)),
            torch.tile(index_x, (patches_y,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            # feat_dim=head_dim // 4,
            feat_dim=dim_arrange[2],
        )

        # fit Qwen scale-rope
        pos_index_y = torch.arange(patches_y)
        neg_index_y = torch.arange(patches_y).flip(0) * -1 - 1
        index_y = torch.cat([neg_index_y[-(patches_y - patches_y // 2) :], pos_index_y[: patches_y // 2]], dim=0)
        # print(index_y)
        ## Qwen rope apply order is frame, height, width!
        # coeffs_y
        coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            # torch.repeat_interleave(torch.arange(patches_y), patches_x),
            torch.repeat_interleave(index_y, patches_x),
            freq_base=freq_base,
            freq_scale=freq_scale,
            # feat_dim=head_dim // 4,
            feat_dim=dim_arrange[1],
        )

        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)




    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        # remove coeffs from state_dict
        state_dict.pop("coeffs_x_0", None)
        state_dict.pop("coeffs_x_1", None)
        state_dict.pop("coeffs_y_0", None)
        state_dict.pop("coeffs_y_1", None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        return prope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            **kwargs,
        )

    def _precompute_and_cache_apply_fns(
        self, 
        viewmats: torch.Tensor, 
        Ks: Optional[torch.Tensor],
        depth = None,
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        self.cameras = cameras
        self.use_PRoPE = True

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            dim_arrange=self.dim_arrange,
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            depth=depth
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = q.shape
        # print("!!!", q.shape)
        # print(self.cameras, self.patches_x, self.patches_y)
        assert seqlen == self.cameras * self.patches_x * self.patches_y, f"seqlen:{seqlen}, {self.cameras}, {self.patches_x}, {self.patches_y}"
        assert head_dim == self.head_dim
        assert q.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = kv.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y, f"seqlen:{seqlen}, {self.cameras}, {self.patches_x}, {self.patches_y}"
        assert head_dim == self.head_dim
        assert kv.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_kv is not None
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = o.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert o.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)


def prope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
    )

    out = F.scaled_dot_product_attention(
        query=apply_fn_q(q),
        key=apply_fn_kv(k),
        value=apply_fn_kv(v),
        **kwargs,
    )
    out = apply_fn_o(out)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
    coeffs_z: Optional[torch.Tensor] = None,
    dim_arrange = None,
    freq_base = None,
    freq_scale = None,
    depth = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    viewmats = viewmats.to(torch.float32)
    Ks = Ks.to(torch.float32)

    # Normalize camera intrinsics.
    if Ks is not None:  
        # Ks has been normalized in the dataset getitem !!
        Ks_norm = Ks
        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Precompute cos/sin terms for RoPE. We use tiles/repeats for 'row-major'
    # broadcasting.

    assert coeffs_x is not None
    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
            freq_base=100.0,
            freq_scale=1.0,
            # feat_dim=head_dim // 4,
            feat_dim=dim_arrange[1],
        )
    assert coeffs_y is not None
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(
                    torch.arange(patches_y, device=device), patches_x
                ),
                (cameras,),
            ),
            freq_base=100.0,
            freq_scale=1.0,
            # feat_dim=head_dim // 4,
            feat_dim=dim_arrange[2],
        )

    if torch.isnan(P_inv).any():
        print("!!P_inv has NaN!!!")
        exit(0)
    if torch.isnan(P_T).any():
        print("!!P_T has NaN!!!")
        exit(0)
    if torch.isnan(coeffs_x[0]).any() or torch.isnan(coeffs_x[1]).any():
        print("!!coeffs_x has NaN!!!")
        exit(0)
    if torch.isnan(coeffs_y[0]).any() or torch.isnan(coeffs_y[1]).any():
        print("!!coeffs_y has NaN!!!")
        exit(0)


    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0

    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), dim_arrange[0]),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), dim_arrange[1]),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), dim_arrange[2]),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), dim_arrange[0]),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), dim_arrange[1]),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), dim_arrange[2]),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), dim_arrange[0]),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), dim_arrange[1]),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), dim_arrange[2]),
    ]

    if len(dim_arrange) == 4:
        index_z = rearrange(depth, 'b n h w -> b n (h w)') # (batch, frame, seq_len)
        coeffs_z: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs_z(
            index_z,
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=dim_arrange[3],
        )
        coeffs_z_0 = coeffs_z[0]
        coeffs_z_1 = coeffs_z[1]
        coeffs_z = (coeffs_z_0, coeffs_z_1)

        transforms_q += [(partial(_rope_apply_coeffs_z, coeffs=coeffs_z), dim_arrange[3])]
        transforms_kv += [(partial(_rope_apply_coeffs_z, coeffs=coeffs_z), dim_arrange[3])]
        transforms_o += [(partial(_rope_apply_coeffs_z, coeffs=coeffs_z, inverse=True), dim_arrange[3])]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)

    matrix = matrix.to(feats.dtype)

    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    # print(matrix.device, feats.device)
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _rope_apply_coeffs_z(
    feats: torch.Tensor,  # (batch, num_heads, seqlen_total, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],  # (batch, 1, frame, seqlen_img, num_freqs)
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""

    # print("Inject z rope!!")
    #TODO change to interleaved same as Qwen?
    cos, sin = coeffs 

    batch, num_heads, total_seq_len, feat_dim = feats.shape
    _, __, frames, seq_len_per_img, num_freqs = cos.shape

    cos = cos.to(feats.dtype)   
    sin = sin.to(feats.dtype)
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    feats = feats.reshape((batch, num_heads, frames, seq_len_per_img, feat_dim))
    assert feats.shape[3] * frames == total_seq_len
    # if cos.shape[2] != feats.shape[2]:
    #     n_repeats = feats.shape[2] // cos.shape[2]
    #     cos = cos.repeat(1, 1, n_repeats, 1)
    #     sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(cos.shape) == len(sin.shape) == len(feats.shape) == 5 
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2

    # cos (batch, 1, frame, seqlen_img, feat_dim)
    x_in = feats[..., ::2]  # even  # (batch, num_heads, frames, seqlen_img, feat_dim)
    y_in = feats[..., 1::2]

    if inverse == False: # for qkv
        x_out = cos * x_in - sin * y_in # broadcast on "num_heads"
        y_out = sin * x_in + cos * y_in
    else:   # for out 
        x_out = cos * x_in + sin * y_in
        y_out = -sin * x_in + cos * y_in

    res = torch.stack((x_out, y_out), dim=-1).flatten(start_dim=-2)
    res = rearrange(res, 'b n f s d -> b n (f s) d')
    # print(res.shape)
    return res

def _rope_precompute_coeffs_z(
    positions: torch.Tensor,  # (batch, frame, seq_len)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 3
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    # print(freqs.shape)
    # print(positions[:128])
    angles = positions[:, None, :, :, None] * freqs
    # Shape should be: `(batch, num_heads, frame, seqlen, num_freqs)`; we're
    # broadcasting across `num_heads`.
    assert angles.shape == (positions.shape[0], 1, positions.shape[1], positions.shape[2], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    # print(freqs.shape)
    # print(positions[:128])
    angles = positions[None, None, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)


if __name__ == '__main__':

    patches_x = 64
    patches_y = 64
    freq_base = 1
    freq_scale = 10000
    head_dim = 128

    pos_index = torch.arange(patches_x)
    neg_index = torch.arange(patches_x).flip(0) * -1 - 1
    index = torch.cat([neg_index[-(patches_x - patches_x // 2) :], pos_index[: patches_x // 2]], dim=0)
    print(index)
    print(torch.arange(patches_x))
    coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            # torch.tile(torch.arange(patches_x), (patches_y,)),
            torch.tile(index, (patches_y,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            # feat_dim=head_dim // 4,
            feat_dim=56,
        )

    coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y), patches_x),
            freq_base=freq_base,
            freq_scale=freq_scale,
            # feat_dim=head_dim // 4,
            feat_dim=56,
        )

def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""

    #TODO change to interleaved same as Qwen?
    cos, sin = coeffs

    cos = cos.to(feats.dtype)
    sin = sin.to(feats.dtype)
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2

    x_in = feats[..., ::2]  # even  # (batch, num_heads, seqlen, feat_dim)
    y_in = feats[..., 1::2]

    if inverse == False: # for qkv
        x_out = cos * x_in - sin * y_in
        y_out = sin * x_in + cos * y_in
    else:   # for out 
        x_out = cos * x_in + sin * y_in
        y_out = -sin * x_in + cos * y_in

    res = torch.stack((x_out, y_out), dim=-1).flatten(start_dim=-2)
    # print(res.shape)
    return res



def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out