"""
Implementation of https://arxiv.org/abs/2410.17242
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from dppe.torch import PropeDotProductAttention, _invert_SE3
from dppe.utils.functional import (
    Camera,
    camera_to_raymap,
    patchify,
    raymap_to_plucker,
    unpatchify,
)
from dppe.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)


@dataclass
class LVSMDecoderOnlyModelConfig:
    ref_views: int = 2
    tar_views: int = 1

    encoder: TransformerEncoderConfig = field(
        default_factory=lambda: TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=768,
                nhead=16,
                dim_feedforward=3072,
                dropout=0.0,
                activation=F.relu,
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True,
                bias=False,
                elementwise_affine=True,
                norm_type="layer_norm",
                modulation_activation=None,
                qk_norm=False,
            ),
            num_layers=6,
            input_norm=True,
            output_norm=True,
            checkpointing=False,
        ),
    )

    img_shape: list[int] = field(default_factory=lambda: [256, 256, 3])
    cam_shape: list[int] = field(default_factory=lambda: [256, 256, 6])
    patch_size: int = 8

    # How the input rays are encoded.
    ray_encoding: Literal["plucker", "camray", "none", "raymap"] = "plucker"
    qk_pe: str = "p2d"
    vo_pe: str = "p2d"


class LVSMDecoderOnlyModel(nn.Module):
    def __init__(self, config: LVSMDecoderOnlyModelConfig):
        super().__init__()
        self.config = config

        # Create single attention module for all layers
        attn = PropeDotProductAttention(
            head_dim=config.encoder.layer.d_model // config.encoder.layer.nhead,
            patches_x=config.img_shape[1] // config.patch_size,
            patches_y=config.img_shape[0] // config.patch_size,
            image_width=config.img_shape[1],
            image_height=config.img_shape[0],
            qk_pe=config.qk_pe,
            vo_pe=config.vo_pe,
        )

        num_layers = config.encoder.num_layers
        self.attentions = nn.ModuleList([attn for _ in range(num_layers)])

        assert config.cam_shape[:2] == config.img_shape[:2], (
            f"{config.cam_shape[:2]} != {config.img_shape[:2]}"
        )

        if config.ray_encoding == "none":
            shared_rays = torch.zeros(config.cam_shape)
            self.register_buffer("shared_rays", shared_rays, persistent=False)

        # query tokenizer encodes tar_cam
        self.query_tokenizer = nn.Linear(
            config.cam_shape[-1] * config.patch_size**2,
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )
        # input tokenizer encodes ref_img and ref_cam
        self.input_tokenizer = nn.Linear(
            (
                config.img_shape[-1] * config.patch_size**2
                + config.cam_shape[-1] * config.patch_size**2
            ),
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )

        self.encoder = self.config.encoder.setup()

        self.output_layer = nn.Linear(
            config.encoder.layer.d_model,
            config.img_shape[-1] * config.patch_size**2,
            bias=config.encoder.layer.bias,
        )
        self.init_weights()

    def init_weights(self):
        for idx, layer in enumerate(self.encoder.layers):
            layer.apply(self.init_layer_weights(idx))

    def init_layer_weights(self, idx):
        # LVMS Paper A.1:
        # "We initialize the model weights with a normal distribution of zero-mean
        # and standard deviation of 0.02/(2 * (idx+ 1)) ** 0.5, where idx means
        # transform layer index."
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02 / (2 * (idx + 1)) ** 0.5)

        return _init_weights

    def create_rays(self, cams: Camera) -> Tensor:
        """Convert cameras to raymaps.

        Returns:
            rays: [B, V, H, W, C]
        """
        config = self.config
        batch_size, v = cams.camtoworld.shape[:2]
        cam_dtype = cams.camtoworld.dtype
        device = cams.camtoworld.device

        if config.ray_encoding == "none":
            rays = repeat(self.shared_rays, "h w c -> b v h w c", b=batch_size, v=v)
        else:
            # Preprocess cameras into rays.
            downscale = config.img_shape[0] // config.cam_shape[0]
            rays = camera_to_raymap(
                Ks=cams.K,
                camtoworlds=(
                    torch.eye(4, dtype=cam_dtype, device=device).broadcast_to(
                        cams.camtoworld.shape
                    )
                    if config.ray_encoding == "camray"
                    else cams.camtoworld
                ),
                height=cams.height,
                width=cams.width,
                downscale=downscale,
            )
            if config.ray_encoding in ["plucker", "camray"]:
                rays = raymap_to_plucker(rays)
            else:
                assert config.ray_encoding == "raymap"
        return rays

    def forward(
        self,
        ref_imgs: Tensor,
        ref_cams: Camera,
        tar_cams: Camera,
    ) -> Tensor:
        # ref_imgs: [B, V1, H, W, C]
        # tar_imgs: [B, V2, H, W, C]
        batch_size, v2 = tar_cams.camtoworld.shape[:2]
        config = self.config

        # Create rays.
        # ref_rays: [B, V1, H, W, C]
        # tar_rays: [B, V2, H, W, C]
        ref_rays = self.create_rays(ref_cams)
        tar_rays = self.create_rays(tar_cams)

        # ref_imgs: [B, V1, N1, DIM1]
        ref_imgs = patchify(ref_imgs, config.patch_size)
        # ref_rays: [B, V1, N2, DIM2]
        ref_rays = patchify(ref_rays, config.patch_size)
        # tar_rays: [B, V2, N2, DIM2]
        tar_rays = patchify(tar_rays, config.patch_size)

        # Tokenize into
        # x: [B*V2, V1*N1, DIM1]
        # q: [B*V2, N2, DIM2]
        x = self.input_tokenizer(torch.cat([ref_imgs, ref_rays], dim=-1))
        x = repeat(x, "b v1 n d -> (b v2) (v1 n) d", v2=v2)
        q = self.query_tokenizer(tar_rays)
        q = rearrange(q, "b v2 n d -> (b v2) n d")
        q_tokens = q.shape[1]

        # --- Prepare data for geomtry-aware self-attention ---
        ref_c2ws = repeat(ref_cams.camtoworld, "b v1 x y -> (b v2) v1 x y", v2=v2)
        ref_Ks = repeat(ref_cams.K, "b v1 x y -> (b v2) v1 x y", v2=v2)
        tar_c2ws = rearrange(tar_cams.camtoworld, "b v2 x y -> (b v2) 1 x y", v2=v2)
        tar_Ks = rearrange(tar_cams.K, "b v2 x y -> (b v2) 1 x y")
        c2ws = torch.cat([ref_c2ws, tar_c2ws], dim=1)  # [B, N, 4, 4] per camera
        Ks = torch.cat([ref_Ks, tar_Ks], dim=1)  # [B, N, 3, 3] per camera
        viewmats = _invert_SE3(c2ws)

        # run attentions
        xq = torch.cat([x, q], dim=1)  # [B, N * n, d]
        xq = self.encoder(xq, viewmats, Ks, sdpa_fns=self.attentions)
        q = xq[:, -q_tokens:, :]
        q = rearrange(q, "(b v) n d -> b v n d", b=batch_size, v=v2)

        # output layer
        o = self.output_layer(q)
        o = unpatchify(
            o,
            height=config.img_shape[0],
            width=config.img_shape[1],
            patch_size=config.patch_size,
        )
        return o
