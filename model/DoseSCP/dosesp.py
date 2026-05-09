import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from einops.layers.torch import Rearrange

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")

import sys
sys.path.append("./model/DoseSCP/")
from rmt_module import RMTBlock
from prototype import PrototypeLightDecoderNet3D, scribble_loss_3class

__all__ = [
    "DoseSP",
    "RMT",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
]


class DoseSP(nn.Module):

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        layerscale: Sequence[bool] = (False, False, True, True),
        chunkwise_recurrent: Sequence[bool] = (True, True, False, False),        
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        dummy = False
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")
        '''
        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")
        '''
        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.beta_raw  = nn.Parameter(torch.tensor(0.0))

        self.rmt = RMT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            chunkwise_recurrent=chunkwise_recurrent,
            layerscale=layerscale,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            dummy=dummy
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.scseg = PrototypeLightDecoderNet3D(in_channels=3, feat_dim=64, num_classes=3)

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
    
    def forward(self, x_in):
        """
        Forward pass of DoseSP with structure-guided attention.

        Inputs:
            x_in: Tensor of shape [B, C, D, H, W]
                  Multi-channel input including structure masks and imaging features.

        Returns:
            logits         : Dose prediction output
            coarse_seg     : Segmentation logits (PTV / OAR / BG)
            compact_loss   : Prototype compactness regularization
            sep_loss       : Prototype separation regularization
            scribble_loss  : Scribble supervision loss
        """

        # ======================================================
        # Segmentation Branch (Prototype-based segmentation)
        # ======================================================

        # Extract segmentation-related channels
        # (Assuming channels 4~6 correspond to image + beam + angle features)
        seg_inp = x_in[:, 4:7].detach()

        # Extract scribble annotations
        ptv = x_in[:, 1]   # PTV scribble
        oar = x_in[:, 2]   # OAR scribble
        bg  = x_in[:, 8]   # Background scribble

        # Prototype-based segmentation forward pass
        coarse_seg, compact_loss, sep_loss = self.scseg(
            seg_inp, ptv, oar, bg
        )

        # Scribble supervision loss
        scribble_loss = scribble_loss_3class(
            coarse_seg, ptv, oar, bg
        )

        # ======================================================
        # Structure-Guided Attention Mechanism
        # ======================================================

        # Convert segmentation logits to probabilities
        # Detach to prevent dose loss from backpropagating into segmentation
        prob = torch.softmax(coarse_seg.detach(), dim=1)

        # Extract class-specific attention maps
        ptv_att = prob[:, 0:1]  # positive attention (dose promotion)
        oar_att = prob[:, 1:2]  # negative attention (dose suppression)


        # Residual-style structure attention
        # F_dose' = F_dose * (1 + a*PTV ? b*OAR)
        alpha = F.softplus(self.alpha_raw)
        beta  = F.softplus(self.beta_raw)

        raw_attention = 1.0 + alpha * ptv_att - beta * oar_att
        structure_attention = 2.0 * torch.sigmoid(raw_attention)


        # ======================================================
        # Dose Prediction Backbone (RMT Encoder)
        # ======================================================

        hidden_states_out = self.rmt(x_in[:, :8], self.normalize)
        attended_hidden_states = [] 

        # Apply structure-guided attention to multi-scale features
        for h in hidden_states_out:
            att_resized = F.interpolate(
                structure_attention,
                size=h.shape[2:],   # match spatial size
                mode='trilinear',
                align_corners=True
            )
            attended_hidden_states.append(h * att_resized)

        hidden_states_out = attended_hidden_states

        # ======================================================
        # UNet-style Decoder for Dose Prediction
        # ======================================================

        enc0 = self.encoder1(x_in[:, :8])
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        # Handle potential spatial mismatch
        dec4_depth, dec4_height, dec4_width = dec4.shape[2:]
        h3 = hidden_states_out[3]
        h3_depth, h3_height, h3_width = h3.shape[2:]

        if (
            h3_depth != 2 * dec4_depth or
            h3_height != 2 * dec4_height or
            h3_width != 2 * dec4_width
        ):
            new_size = (2 * dec4_depth, 2 * dec4_height, 2 * dec4_width)
            ori_size = (h3_depth, h3_height, h3_width)

            h3_resized = F.interpolate(
                h3,
                size=new_size,
                mode='trilinear',
                align_corners=True
            )

            temp_dec3 = self.decoder5(dec4, h3_resized)

            dec3 = F.interpolate(
                temp_dec3,
                size=ori_size,
                mode='trilinear',
                align_corners=True
            )
        else:
            dec3 = self.decoder5(dec4, h3)

        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out  = self.decoder1(dec0, enc0)

        # Final dose prediction
        logits = self.out(out)

        # ======================================================
        # Return outputs
        # ======================================================

        return (
            logits,         # Dose prediction
            coarse_seg,     # Segmentation logits
            compact_loss,
            sep_loss,
            scribble_loss
        )


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


class BasicLayer(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        layerscale: list,
        chunkwise_recurrent: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [   
                RMTBlock(
                    embed_dim=dim,
                    num_heads=num_heads,
                    init_value=1.0,
                    heads_range=3.0,
                    ffn_dim=int(mlp_ratio*dim),
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    chunkwise_recurrent=chunkwise_recurrent[i] if isinstance(chunkwise_recurrent, list) else chunkwise_recurrent,
                    layerscale=layerscale[i] if isinstance(layerscale, list) else layerscale,
                    layer_init_values=1e-5,
                )  
                
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x):
        x_shape = x.size()
        x = rearrange(x, "b c d h w -> b d h w c")
        
        for blk in self.blocks:
            x = blk(x)
            
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class RMT(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        layerscale: Sequence[bool],
        chunkwise_recurrent: Sequence[bool],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        dummy = False
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            if dummy:
                layer = nn.Sequential(
                    Rearrange("n c d h w -> n d h w c"),
                    PatchMerging(dim=int(embed_dim * 2**i_layer), spatial_dims=spatial_dims),
                    Rearrange("n d h w c -> n c d h w")
                )
            else:
                layer = BasicLayer(
                    dim=int(embed_dim * 2**i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=self.window_size,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    layerscale=layerscale[i_layer],
                    chunkwise_recurrent=chunkwise_recurrent[i_layer],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    downsample=down_sample_mod,
                    use_checkpoint=use_checkpoint,
                )
                
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------
    # Define input dimensions
    # ----------------------------------
    B = 1
    C_in = 9   # Must match your real channel count
    D, H, W = 96, 128, 128

    # Create dummy multi-channel input
    x = torch.zeros((B, C_in, D, H, W), device=device)

    # Add fake scribbles
    x[:, 1, 30:35, 40:50, 40:50] = 1   # PTV
    x[:, 2, 50:55, 60:70, 60:70] = 1   # OAR
    x[:, 8, 10:15, 20:30, 20:30] = 1   # BG

    # ----------------------------------
    # Initialize model
    # ----------------------------------
    model = DoseSP(
        img_size=[D, H, W],
        in_channels=C_in-1,
        out_channels=7,
        dummy=False
    ).to(device)

    model.eval()

    # ----------------------------------
    # Forward pass
    # ----------------------------------
    with torch.no_grad():
        dose_logits, coarse_seg, compact_loss, sep_loss, scribble_loss = model(x)

    # ----------------------------------
    # Print shapes
    # ----------------------------------
    print("Dose output shape:", dose_logits.shape)
    print("Segmentation shape:", coarse_seg.shape)
    print("Compactness loss:", compact_loss.item())
    print("Separation loss:", sep_loss.item())
    print("Scribble loss:", scribble_loss.item())

    print("Forward test successful.")

