import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from typing import Tuple
import sys
import os
import torch.utils.checkpoint as checkpoint

import torch
import torch.nn as nn
from typing import Tuple

class RetNetRelPos(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        """
        Construct 3D relative positional encodings for RetNet.

        Components:
        - angle:  frequency vector used for sinusoidal (RoPE-style) encoding,
                  shape = (head_dim,)
        - decay:  per-head decay coefficients (in log form), shape = (num_heads,)

        Args:
            embed_dim:     model hidden dimension
            num_heads:     number of attention heads
            initial_value: initial decay value
            heads_range:   controls decay differences across heads
        """
        super().__init__()
        head_dim = embed_dim // num_heads

        # Construct RoPE frequencies: half frequencies duplicated to match head_dim
        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()  # (head_dim,)

        # Construct per-head decay factors (stored in log-space)
        head_idx = torch.arange(num_heads, dtype=torch.float)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * head_idx / num_heads)
        )  # (num_heads,)

        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    # -------------------------------------------------------
    # 1D decay (unchanged from 2D version)
    # -------------------------------------------------------
    def generate_1d_decay(self, length: int):
        """
        Generate a 1D distance decay matrix.

        Returns:
            shape = (num_heads, length, length)
        """
        index = torch.arange(length, device=self.decay.device)
        dist = (index[:, None] - index[None, :]).abs()  # (length, length)
        mask = dist * self.decay[:, None, None]         # (num_heads, length, length)
        return mask

    # -------------------------------------------------------
    # New: 3D decay
    # -------------------------------------------------------
    def generate_3d_decay(self, D: int, H: int, W: int):
        """
        Generate a 3D decay mask based on 3D Manhattan distance.

        Total sequence length L = D * H * W

        Returns:
            mask: (num_heads, L, L)
                  L1 distance = |dz| + |dh| + |dw|
        """
        index_d = torch.arange(D, device=self.decay.device)
        index_h = torch.arange(H, device=self.decay.device)
        index_w = torch.arange(W, device=self.decay.device)

        # 3D coordinate grid, each with shape (D, H, W)
        grid_d, grid_h, grid_w = torch.meshgrid(
            index_d, index_h, index_w, indexing="ij"
        )

        # Flatten to (L, 3): each entry is (d, h, w)
        grid = torch.stack([grid_d, grid_h, grid_w], dim=-1).reshape(D * H * W, 3)

        # Pairwise difference to L1 distance
        diff = grid[:, None, :] - grid[None, :, :]  # (L, L, 3)
        dist = diff.abs().sum(dim=-1)               # (L, L)

        # Apply per-head decay: (num_heads, L, L)
        mask = dist * self.decay[:, None, None]
        return mask

    # -------------------------------------------------------
    # forward: supports full 3D input (D, H, W)
    # -------------------------------------------------------
    def forward(
        self,
        slen: Tuple[int, int, int],   # (D, H, W)
        activate_recurrent: bool = False,
        chunkwise_recurrent: bool = False,
    ):
        """
        Compute 3D sinusoidal relative position encodings + decay masks.

        Modes:
        1. activate_recurrent=True:
            - returns only the final-step sin/cos + per-head decay factor
              used for streaming RetNet
        2. chunkwise_recurrent=True:
            - returns factored 1D decay masks (D, H, W separately) 
            - much cheaper than full 3D (L * L) matrix
        3. else (default):
            - returns full 3D pairwise decay matrix (num_heads, L, L)

        Returns:
            ((sin, cos), mask)

            sin, cos:
                (D, H, W, head_dim)
            mask:
                either:
                    (num_heads, L, L)
                or:
                    (mask_d, mask_h, mask_w)
        """
        D, H, W = slen
        L = D * H * W

        # ---------------------------------------------
        # Case 1: Recurrent mode (only one-step state)
        # ---------------------------------------------
        if activate_recurrent:
            idx_last = L - 1
            sin = torch.sin(self.angle * idx_last)  # (head_dim,)
            cos = torch.cos(self.angle * idx_last)  # (head_dim,)
            decay_per_head = self.decay.exp()       # (num_heads,)
            return (sin, cos), decay_per_head

        # ---------------------------------------------
        # Case 2 & 3: Compute full sequence RoPE encoding
        # ---------------------------------------------
        index = torch.arange(L, device=self.decay.device)  # (L,)
        sin = torch.sin(index[:, None] * self.angle[None, :])  # (L, head_dim)
        cos = torch.cos(index[:, None] * self.angle[None, :])  # (L, head_dim)

        # Reshape to 3D grid: (D, H, W, head_dim)
        sin = sin.reshape(D, H, W, -1)
        cos = cos.reshape(D, H, W, -1)

        # ---------------------------------------------
        # Case 2: Chunkwise recurrent (D, H, W factored)
        # ---------------------------------------------
        if chunkwise_recurrent:
            mask_d = self.generate_1d_decay(D)
            mask_h = self.generate_1d_decay(H)
            mask_w = self.generate_1d_decay(W)
            return (sin, cos), (mask_d, mask_h, mask_w)

        # ---------------------------------------------
        # Case 3: Full 3D pairwise distance mask
        # ---------------------------------------------
        mask = self.generate_3d_decay(D, H, W)
        return (sin, cos), mask

def rotate_every_two(x: torch.Tensor):
    # x: (..., d)
    x1 = x[..., ::2]   # even dims: (..., d/2)
    x2 = x[..., 1::2]  # odd dims:  (..., d/2)
    x_rot = torch.stack([-x2, x1], dim=-1)  # (..., d/2, 2)
    return x_rot.flatten(-2)               # (..., d)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv3d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        # Depthwise 3D conv: groups=dim means one conv per channel
        self.conv = nn.Conv3d(
            dim, dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape (B, D, H, W, C)
                channels-last format

        Returns:
            Tensor of shape (B, D, H, W, C)
        """
        # Convert to channels-first for Conv3d
        x = x.permute(0, 4, 1, 2, 3)   # (B, C, D, H, W)

        # Apply depthwise 3D convolution
        x = self.conv(x)               # (B, C, D, H, W)

        # Convert back to channels-last
        x = x.permute(0, 2, 3, 4, 1)   # (B, D, H, W, C)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn

        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.dropout_module = nn.Dropout(dropout)

        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)

        # Optional LayerNorm in FFN hidden dimension
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

        # Optional depthwise 3D conv in FFN hidden dimension
        self.dwconv = DWConv3d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, D, H, W, C)
        """
        # First linear projection to FFN hidden dim
        x = self.fc1(x)                          # (B, D, H, W, ffn_dim)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)

        residual = x

        # Optional depthwise 3D conv in hidden space
        if self.dwconv is not None:
            x = self.dwconv(x)                   # (B, D, H, W, ffn_dim)

        # Optional LayerNorm in hidden space
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)            # norm over last dim

        # Local residual connection in hidden space
        x = x + residual                         # (B, D, H, W, ffn_dim)

        # Project back to embed_dim
        x = self.fc2(x)                          # (B, D, H, W, C)
        x = self.dropout_module(x)
        return x

class RMTBlock(nn.Module):
    """ A basic RMT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, embed_dim, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, 
                 chunkwise_recurrent=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.block = RetBlock(flag, embed_dim, num_heads, ffn_dim, drop_path, layerscale, layer_init_values)

    def forward(self, x):
        b, d, h, w, c = x.size()
        rel_pos = self.Relpos((d, h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        
        x = self.block(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
        
        return x
        
        
class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv3d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)

    def forward(
            self,
            x: torch.Tensor, 
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x
        
class VisionRetentionChunk(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)

        # Local positional encoding via depthwise 3D convolution
        # NOTE: if value_factor != 1, consider changing to embed_dim * self.factor
        self.lepe = DWConv3d(embed_dim * self.factor, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos,
        chunkwise_recurrent: bool = False,
        incremental_state=None,
    ):
        """
        Args:
            x: (B, D, H, W, C)
            rel_pos: ((sin, cos), (mask_d, mask_h, mask_w))
                sin, cos: (D, H, W, head_dim)
                mask_d:   (num_heads, D, D)
                mask_h:   (num_heads, H, H)
                mask_w:   (num_heads, W, W)

        Returns:
            output: (B, D, H, W, C)
        """
        B, D, H, W, _ = x.size()

        (sin, cos), (mask_d, mask_h, mask_w) = rel_pos  # chunkwise 3D masks

        # Linear projections
        q = self.q_proj(x)                            # (B, D, H, W, C)
        k = self.k_proj(x)                            # (B, D, H, W, C)
        v = self.v_proj(x)                            # (B, D, H, W, C * factor)

        # Local positional encoding on v
        lepe = self.lepe(v)                           # (B, D, H, W, C * factor)

        # Reshape into heads
        k = k * self.scaling
        q = q.view(B, D, H, W, self.num_heads, self.key_dim)   # (B,D,H,W,n,d1)
        k = k.view(B, D, H, W, self.num_heads, self.key_dim)   # (B,D,H,W,n,d1)

        # (B, n, D, H, W, d1)
        q = q.permute(0, 4, 1, 2, 3, 5)
        k = k.permute(0, 4, 1, 2, 3, 5)

        # Apply rotary / theta shift
        # sin, cos: (D, H, W, d1) to broadcast to (B, n, D, H, W, d1)
        qr = theta_shift(q, sin, cos)                 # (B, n, D, H, W, d1)
        kr = theta_shift(k, sin, cos)                 # (B, n, D, H, W, d1)

        # v: (B, D, H, W, n * d2)
        d2 = self.head_dim
        v = v.view(B, D, H, W, self.num_heads, d2)    # (B,D,H,W,n,d2)

        # ------------------------------------------------
        # 1) Width-wise retention (along W)
        # ------------------------------------------------
        # qr_w, kr_w, v_w: (B, D, H, n, W, d*)
        qr_w = qr.permute(0, 2, 3, 1, 4, 5)           # (B,D,H,n,W,d1)
        kr_w = kr.permute(0, 2, 3, 1, 4, 5)           # (B,D,H,n,W,d1)
        v_w  = v.permute(0, 1, 2, 4, 3, 5)           # (B,D,H,n,W,d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)      # (B,D,H,n,W,W)
        qk_mat_w = qk_mat_w + mask_w[None, None, None, :, :, :]  # broadcast (n,W,W)
        qk_mat_w = torch.softmax(qk_mat_w, dim=-1)    # (B,D,H,n,W,W)
        v_w = torch.matmul(qk_mat_w, v_w)             # (B,D,H,n,W,d2)

        # ------------------------------------------------
        # 2) Height-wise retention (along H)
        # ------------------------------------------------
        # qr_h, kr_h: (B, D, W, n, H, d1)
        qr_h = qr.permute(0, 2, 4, 1, 3, 5)           # (B,D,W,n,H,d1)
        kr_h = kr.permute(0, 2, 4, 1, 3, 5)           # (B,D,W,n,H,d1)

        # v_h: (B, D, W, n, H, d2)
        v_h = v_w.permute(0, 1, 4, 3, 2, 5)          # (B,D,W,n,H,d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)      # (B,D,W,n,H,H)
        qk_mat_h = qk_mat_h + mask_h[None, None, None, :, :, :]
        qk_mat_h = torch.softmax(qk_mat_h, dim=-1)    # (B,D,W,n,H,H)
        v_h = torch.matmul(qk_mat_h, v_h)             # (B,D,W,n,H,d2)

        # ------------------------------------------------
        # 3) Depth-wise retention (along D)
        # ------------------------------------------------
        # qr_d, kr_d: (B, H, W, n, D, d1)
        qr_d = qr.permute(0, 3, 4, 1, 2, 5)           # (B,H,W,n,D,d1)
        kr_d = kr.permute(0, 3, 4, 1, 2, 5)           # (B,H,W,n,D,d1)

        # v_d: (B, H, W, n, D, d2)
        v_d = v_h.permute(0, 4, 2, 3, 1, 5)          # (B,H,W,n,D,d2)

        qk_mat_d = qr_d @ kr_d.transpose(-1, -2)      # (B,H,W,n,D,D)
        qk_mat_d = qk_mat_d + mask_d[None, None, None, :, :, :]
        qk_mat_d = torch.softmax(qk_mat_d, dim=-1)    # (B,H,W,n,D,D)
        out_d = torch.matmul(qk_mat_d, v_d)           # (B,H,W,n,D,d2)

        # ------------------------------------------------
        # Re-arrange back to (B, D, H, W, n * d2)
        # ------------------------------------------------
        out = out_d.permute(0, 4, 1, 2, 3, 5)        # (B,D,H,W,n,d2)
        out = out.reshape(B, D, H, W, self.num_heads * d2)  # (B,D,H,W,n*d2)

        # Add local positional encoding
        out = out + lepe                              # (B,D,H,W,n*d2)

        # Final projection back to embed_dim
        out = self.out_proj(out)                      # (B,D,H,W,C)
        return out

class VisionRetentionAll(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)

        # Local positional encoding via depthwise 3D convolution
        self.lepe = DWConv3d(embed_dim * self.factor, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos,
        chunkwise_recurrent: bool = False,
        incremental_state=None,
    ):
        """
        Full 3D retention version (non-chunkwise).

        Args:
            x: (B, D, H, W, C)
            rel_pos: ((sin, cos), mask)
                sin, cos: (D, H, W, head_dim)
                mask:     (num_heads, L, L), where L = D * H * W

        Returns:
            output: (B, D, H, W, C)
        """
        B, D, H, W, _ = x.size()
        (sin, cos), mask = rel_pos

        L = D * H * W
        assert mask.size(1) == L and mask.size(2) == L, "Mask shape must match D*H*W"

        # Linear projections
        q = self.q_proj(x)                            # (B,D,H,W,C)
        k = self.k_proj(x)                            # (B,D,H,W,C)
        v = self.v_proj(x)                            # (B,D,H,W,C*factor)

        # Local positional encoding on v
        lepe = self.lepe(v)                           # (B,D,H,W,C*factor)

        # Prepare for multi-head
        k = k * self.scaling
        q = q.view(B, D, H, W, self.num_heads, -1)    # (B,D,H,W,n,d1)
        k = k.view(B, D, H, W, self.num_heads, -1)    # (B,D,H,W,n,d1)

        # Move heads forward: (B, n, D, H, W, d1)
        q = q.permute(0, 4, 1, 2, 3, 5)
        k = k.permute(0, 4, 1, 2, 3, 5)

        # Apply rotary / theta shift
        # sin, cos: (D,H,W,d1) to broadcast to (B,n,D,H,W,d1) inside theta_shift
        qr = theta_shift(q, sin, cos)                 # (B,n,D,H,W,d1)
        kr = theta_shift(k, sin, cos)                 # (B,n,D,H,W,d1)

        # Flatten spatial dims into sequence length L = D * H * W
        qr = qr.flatten(2, 4)                         # (B,n,L,d1)
        kr = kr.flatten(2, 4)                         # (B,n,L,d1)

        # v: (B,D,H,W,n*d2) to (B,n,L,d2)
        d2 = self.head_dim
        vr = v.view(B, D, H, W, self.num_heads, d2)   # (B,D,H,W,n,d2)
        vr = vr.permute(0, 4, 1, 2, 3, 5)             # (B,n,D,H,W,d2)
        vr = vr.flatten(2, 4)                         # (B,n,L,d2)

        # Global retention over the 3D volume
        qk_mat = qr @ kr.transpose(-1, -2)            # (B,n,L,L)
        # mask: (n,L,L) to broadcast to (B,n,L,L)
        qk_mat = qk_mat + mask[None, :, :, :]         # (B,n,L,L)
        qk_mat = torch.softmax(qk_mat, dim=-1)        # (B,n,L,L)

        output = torch.matmul(qk_mat, vr)             # (B,n,L,d2)

        # Back to (B,D,H,W,n*d2)
        output = output.transpose(1, 2)               # (B,L,n,d2)
        output = output.reshape(B, D, H, W, -1)       # (B,D,H,W,n*d2)

        # Add local positional encoding
        output = output + lepe                        # (B,D,H,W,n*d2)

        # Final projection back to embed_dim
        output = self.out_proj(output)                # (B,D,H,W,C)
        return output
        

def run_test(chunkwise_recurrent: bool):
    """
    Run a forward pass for RMTBlock to verify shapes & correctness.
    """
    torch.manual_seed(42)

    # ----------------- Hyper-parameters -----------------
    B = 2                  # batch size
    D, H, W = 6, 8, 9      # 3D volume size
    embed_dim = 96
    num_heads = 4
    ffn_dim = 192
    init_value = 1.0
    heads_range = 3.0
    drop_path = 0.0
    layerscale = False

    # ----------------- Build RMTBlock -----------------
    layer = RMTBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        init_value=init_value,
        heads_range=heads_range,
        ffn_dim=ffn_dim,
        drop_path=drop_path,
        norm_layer=nn.LayerNorm,
        chunkwise_recurrent=chunkwise_recurrent,
        layerscale=layerscale,
        layer_init_values=1e-5,
    )

    # You can enable CUDA if desired:
    # layer = layer.cuda()

    # ----------------- Fake input ----------------------
    x = torch.randn(B, D, H, W, embed_dim)
    # x = x.cuda()

    # ----------------- Forward -------------------------
    with torch.no_grad():
        y = layer(x)

    print(f"\n==== Test: chunkwise_recurrent = {chunkwise_recurrent} ====")
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")
    print("OK!\n")


if __name__ == "__main__":
    # Test full 3D attention (global retention)
    run_test(chunkwise_recurrent=False) # D, H, W = 6, 8, 9 is ok! 

    # Test chunk-wise 1D * 1D * 1D retention (D / H / W axes)
    run_test(chunkwise_recurrent=True)
