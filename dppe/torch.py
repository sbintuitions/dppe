# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
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


"""Projective positional encoding module for multi-view attention."""

import logging
import warnings
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from flash_attn_interface import flash_attn_func

    logger.info("flash_attn_func found.")
    FLASH_ATTN_ENABLED = True
except ImportError:
    logger.info("flash_attn_func not found.")
    FLASH_ATTN_ENABLED = False


# --- One-time warning flag for handling non-divisible dimensions ---
_WARNING_SHOWN_SPLIT_6D = False


class PropeDotProductAttention(torch.nn.Module):
    """Positional encoding for multi-view attention with precomputed RoPE coefficients.

    Named PE configurations and their (qk_pe, vo_pe) mappings:
        RoPE:     (2d, none)
        CAPE:     (Rt, none)
        GTA:      (Rt2d, Rt2d)
        PRoPE:    (p2d, p2d)
        DPPEdual: (pIT2d, pIT2d)
        DPPEtAdd: (p2d, KRtAdd2d)
    """

    # These can cause an error of torch.compile
    # coeffs_x_0: torch.Tensor
    # coeffs_x_1: torch.Tensor
    # coeffs_y_0: torch.Tensor
    # coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        qk_pe: Literal[
            "p",
            "2d",
            "p2d",
            "none",
            "Rt",
            "KR2d",
            "R2d",
            "Rt2d",
            "pT2d",
            "pIT2d",
            "KRtAdd2d",
            "Kt2d",
            "t2d",
            "K2d",
        ] = "p2d",
        vo_pe: Literal[
            "p",
            "2d",
            "p2d",
            "none",
            "Rt",
            "KR2d",
            "R2d",
            "Rt2d",
            "pT2d",
            "pIT2d",
            "KRtAdd2d",
            "Kt2d",
            "t2d",
            "K2d",
        ] = "p2d",
        num_special_tokens: int = 0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.qk_pe = qk_pe
        self.vo_pe = vo_pe
        self.image_width = image_width
        self.image_height = image_height
        self.num_special_tokens = num_special_tokens

        # Whether use 2d rope somewhere
        if qk_pe in [
            "2d",
            "p2d",
            "KR2d",
            "R2d",
            "Rt2d",
            "pT2d",
            "pIT2d",
            "KRtAdd2d",
            "Kt2d",
            "t2d",
            "K2d",
        ] or vo_pe in [
            "2d",
            "p2d",
            "KR2d",
            "R2d",
            "Rt2d",
            "pT2d",
            "pIT2d",
            "KRtAdd2d",
            "Kt2d",
            "t2d",
            "K2d",
        ]:
            self.patches_x = patches_x
            self.patches_y = patches_y
        else:
            self.patches_x = None
            self.patches_y = None

        # --- 2d rope in query and key ---
        if qk_pe in [
            "2d",
            "p2d",
            "KR2d",
            "R2d",
            "Rt2d",
            "pT2d",
            "pIT2d",
            "KRtAdd2d",
            "Kt2d",
            "t2d",
            "K2d",
        ]:
            if qk_pe in ["2d"]:
                feat_dim = head_dim // 2
            elif qk_pe in [
                "p2d",
                "KR2d",
                "R2d",
                "Rt2d",
                "pT2d",
                "pIT2d",
                "KRtAdd2d",
                "Kt2d",
                "t2d",
                "K2d",
            ]:
                feat_dim = head_dim // 4

            pos_x = torch.tile(torch.arange(patches_x), (patches_y,))
            pos_y = torch.repeat_interleave(torch.arange(patches_y), patches_x)
            if self.num_special_tokens > 0:
                special_pos = torch.zeros(self.num_special_tokens, dtype=pos_x.dtype)
                pos_x = torch.cat([special_pos, pos_x])
                pos_y = torch.cat([special_pos, pos_y])

            coeffs_x: tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
                pos_x,
                freq_base=freq_base,
                freq_scale=freq_scale,
                feat_dim=feat_dim,
            )
            coeffs_y: tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
                pos_y,
                freq_base=freq_base,
                freq_scale=freq_scale,
                feat_dim=feat_dim,
            )
            self.register_buffer("qk_coeffs_x_0", coeffs_x[0], persistent=False)
            self.register_buffer("qk_coeffs_x_1", coeffs_x[1], persistent=False)
            self.register_buffer("qk_coeffs_y_0", coeffs_y[0], persistent=False)
            self.register_buffer("qk_coeffs_y_1", coeffs_y[1], persistent=False)
        else:
            self.qk_coeffs_x_0 = None
            self.qk_coeffs_x_1 = None
            self.qk_coeffs_y_0 = None
            self.qk_coeffs_y_1 = None

        # --- 2d rope in value and output ---
        if vo_pe in [
            "2d",
            "p2d",
            "KR2d",
            "R2d",
            "Rt2d",
            "pT2d",
            "pIT2d",
            "KRtAdd2d",
            "Kt2d",
            "t2d",
            "K2d",
        ]:
            if vo_pe in ["2d"]:
                feat_dim = head_dim // 2
            elif vo_pe in [
                "p2d",
                "KR2d",
                "R2d",
                "Rt2d",
                "pT2d",
                "pIT2d",
                "KRtAdd2d",
                "Kt2d",
                "t2d",
                "K2d",
            ]:
                feat_dim = head_dim // 4

            pos_x = torch.tile(torch.arange(patches_x), (patches_y,))
            pos_y = torch.repeat_interleave(torch.arange(patches_y), patches_x)
            if self.num_special_tokens > 0:
                special_pos = torch.zeros(self.num_special_tokens, dtype=pos_x.dtype)
                pos_x = torch.cat([special_pos, pos_x])
                pos_y = torch.cat([special_pos, pos_y])

            coeffs_x: tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
                pos_x,
                freq_base=freq_base,
                freq_scale=freq_scale,
                feat_dim=feat_dim,
            )
            coeffs_y: tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
                pos_y,
                freq_base=freq_base,
                freq_scale=freq_scale,
                feat_dim=feat_dim,
            )
            self.register_buffer("vo_coeffs_x_0", coeffs_x[0], persistent=False)
            self.register_buffer("vo_coeffs_x_1", coeffs_x[1], persistent=False)
            self.register_buffer("vo_coeffs_y_0", coeffs_y[0], persistent=False)
            self.register_buffer("vo_coeffs_y_1", coeffs_y[1], persistent=False)
        else:
            self.vo_coeffs_x_0 = None
            self.vo_coeffs_x_1 = None
            self.vo_coeffs_y_0 = None
            self.vo_coeffs_y_1 = None

        self.precomputed = False

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict, ignoring precomputed RoPE coefficient buffers."""
        state_dict.pop("qk_coeffs_x_0", None)
        state_dict.pop("qk_coeffs_x_1", None)
        state_dict.pop("qk_coeffs_y_0", None)
        state_dict.pop("qk_coeffs_y_1", None)
        state_dict.pop("vo_coeffs_x_0", None)
        state_dict.pop("vo_coeffs_x_1", None)
        state_dict.pop("vo_coeffs_y_0", None)
        state_dict.pop("vo_coeffs_y_1", None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, seqlen, num_heads, head_dim)
        k: torch.Tensor,  # (batch, seqlen, num_heads, head_dim)
        v: torch.Tensor,  # (batch, seqlen, num_heads, head_dim)
        **kwargs,
    ) -> torch.Tensor:
        """Apply positional encoding and compute flash attention."""
        (batch, seqlen, num_heads, head_dim) = q.shape
        dtype = q.dtype

        q = self.apply_fn_q(q)
        k = self.apply_fn_k(k)
        v = self.apply_fn_v(v)

        assert kwargs.get("dropout_p", 0.0) == 0.0, (
            "Current flash-attn-3 dosen't support dropout_p."
        )

        if FLASH_ATTN_ENABLED:
            out = flash_attn_func(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
            )
        else:
            # PyTorch sdpa assumes (batch, num_heads, seqlen, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            dropout_p = kwargs.get("dropout_p", 0.0)
            out = F.scaled_dot_product_attention(
                q.bfloat16(),
                k.bfloat16(),
                v.bfloat16(),
                dropout_p=dropout_p,
            )
            # returns back to (batch, seqlen, num_heads, head_dim)
            out = out.transpose(1, 2).contiguous()

        out = self.apply_fn_o(out.to(dtype))

        assert out.shape == (
            batch,
            seqlen,
            num_heads,
            head_dim,
        ), f"Output shape should be (batch, seqlen, num_heads, head_dim), but got {out.shape}"
        return out

    def _precompute_and_cache_apply_fns(self, viewmats: torch.Tensor, Ks: torch.Tensor | None):
        """Precompute and cache the PE application functions from camera parameters."""
        if self.precomputed:
            return

        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (
            batch,
            cameras,
            4,
            4,
        ), f"viewmats should have shape (batch, cameras, 4, 4), but got {viewmats.shape}"
        assert Ks.shape == (
            batch,
            cameras,
            3,
            3,
        ), f"Ks should have shape (batch, cameras, 3, 3), but got {Ks.shape}"
        self.cameras = cameras

        self.apply_fn_q, self.apply_fn_k, self.apply_fn_v, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            image_width=self.image_width,
            image_height=self.image_height,
            qk_pe=self.qk_pe,
            vo_pe=self.vo_pe,
            qk_coeffs_x=(self.qk_coeffs_x_0, self.qk_coeffs_x_1),
            qk_coeffs_y=(self.qk_coeffs_y_0, self.qk_coeffs_y_1),
            vo_coeffs_x=(self.vo_coeffs_x_0, self.vo_coeffs_x_1),
            vo_coeffs_y=(self.vo_coeffs_y_0, self.vo_coeffs_y_1),
        )
        if self.qk_pe == "none":
            self.apply_fn_q = self.apply_fn_k = lambda x: x
        if self.vo_pe == "none":
            self.apply_fn_v = self.apply_fn_o = lambda x: x

        self.precomputed = True

    def reset_apply_fns(self):
        """Clear cached application functions so they are recomputed on next use."""
        self.apply_fn_q = self.apply_fn_k = self.apply_fn_v = self.apply_fn_o = None
        self.precomputed = False


def _prepare_apply_fns(
    head_dim: int,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    image_width: int,
    image_height: int,
    qk_pe: str,
    vo_pe: str,
    qk_coeffs_x: torch.Tensor | None,
    qk_coeffs_y: torch.Tensor | None,
    vo_coeffs_x: torch.Tensor | None,
    vo_coeffs_y: torch.Tensor | None,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Build per-tensor apply functions for q, k, v, o based on PE configuration."""
    assert head_dim % 4 == 0, (
        f"Currently only support head_dim that is divisible by 4, but got {head_dim}"
    )

    is_p = qk_pe in ["p", "p2d", "pT2d", "pIT2d", "KRtAdd2d"] or vo_pe in [
        "p",
        "p2d",
        "pT2d",
        "pIT2d",
        "KRtAdd2d",
    ]
    is_kso3 = qk_pe in ["KR2d", "KRtAdd2d"] or vo_pe in ["KR2d", "KRtAdd2d"]
    is_so3 = qk_pe == "R2d" or vo_pe == "R2d"
    is_se3 = qk_pe in ["Rt2d", "Rt"] or vo_pe in ["Rt2d", "Rt"]
    is_kt = qk_pe == "Kt2d" or vo_pe == "Kt2d"
    is_t = qk_pe == "t2d" or vo_pe == "t2d"
    is_k = qk_pe == "K2d" or vo_pe == "K2d"

    qk_p_dim = None
    qk_2d_dim = None
    if qk_pe == "p" or qk_pe == "Rt":
        qk_p_dim = head_dim
    elif qk_pe == "2d":
        qk_2d_dim = head_dim // 2
    elif qk_pe in [
        "p2d",
        "KR2d",
        "R2d",
        "Rt2d",
        "pT2d",
        "pIT2d",
        "KRtAdd2d",
        "Kt2d",
        "t2d",
        "K2d",
    ]:
        qk_p_dim = head_dim // 2
        qk_2d_dim = head_dim // 4
    elif qk_pe == "none":
        pass
    else:
        raise ValueError(f"Unsupported qk_pe type: {qk_pe}")

    vo_p_dim = None
    vo_2d_dim = None
    if vo_pe == "p" or vo_pe == "Rt":
        vo_p_dim = head_dim
    elif vo_pe == "2d":
        vo_2d_dim = head_dim // 2
    elif vo_pe in [
        "p2d",
        "KR2d",
        "R2d",
        "Rt2d",
        "pT2d",
        "pIT2d",
        "KRtAdd2d",
        "Kt2d",
        "t2d",
        "K2d",
    ]:
        vo_p_dim = head_dim // 2
        vo_2d_dim = head_dim // 4
    elif vo_pe == "none":
        pass
    else:
        raise ValueError(f"Unsupported vo_pe type: {vo_pe}")

    transforms_q = []
    transforms_k = []
    transforms_v = []
    transforms_o = []

    # --- Projection PE ---
    if qk_p_dim is not None or vo_p_dim is not None:
        (batch, cameras, _, _) = viewmats.shape
        if is_p or is_kso3 or is_kt or is_k:
            Ks_norm = torch.zeros_like(Ks)
            Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
            Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
            Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
            Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
            Ks_norm[..., 2, 2] = 1.0
        del Ks

        if is_p:
            P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
            P_T = P.transpose(-1, -2)
            P_inv = torch.einsum(
                "...ij,...jk->...ik",
                _invert_SE3(viewmats),
                _lift_K(_invert_K(Ks_norm)),
            )
            assert P.shape == P_inv.shape == (batch, cameras, 4, 4), (
                f"P and P_inv should have shape (batch, cameras, 4, 4), but got {P.shape} and {P_inv.shape}"
            )

        if is_kso3:
            viewmats_rot_only = viewmats.clone()
            viewmats_rot_only[..., :3, 3] = 0.0  # Zero out translation

            kso3 = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats_rot_only)
            kso3_T = kso3.transpose(-1, -2)
            kso3_inv = torch.einsum(
                "...ij,...jk->...ik",
                _invert_SE3(viewmats_rot_only),  # Invert the zero-translation version
                _lift_K(_invert_K(Ks_norm)),
            )
            assert kso3.shape == kso3_inv.shape == (batch, cameras, 4, 4), (
                f"kso3 and kso3_inv should have shape (batch, cameras, 4, 4), but got {kso3.shape} and {kso3_inv.shape}"
            )

        if is_se3:
            se3 = viewmats
            se3_T = se3.transpose(-1, -2)
            se3_inv = _invert_SE3(viewmats)
            assert se3.shape == se3_inv.shape == (batch, cameras, 4, 4), (
                f"se3 and se3_inv should have shape (batch, cameras, 4, 4), but got {se3.shape} and {se3_inv.shape}"
            )

        if is_so3:
            viewmats_rot_only = viewmats.clone()
            viewmats_rot_only[..., :3, 3] = 0.0  # Zero out translation
            so3 = viewmats_rot_only
            so3_T = so3.transpose(-1, -2)
            so3_inv = _invert_SE3(viewmats_rot_only)
            assert so3.shape == so3_inv.shape == (batch, cameras, 4, 4), (
                f"so3 and so3_inv should have shape (batch, cameras, 4, 4), but got {so3.shape} and {so3_inv.shape}"
            )

        if is_kt:
            # Create viewmats with R replaced by identity I
            viewmats_t_only = torch.eye(4, device=viewmats.device, dtype=viewmats.dtype).repeat(
                batch, cameras, 1, 1
            )
            viewmats_t_only[..., :3, 3] = viewmats[..., :3, 3]

            # P = K * [I | t]
            kt = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats_t_only)
            kt_T = kt.transpose(-1, -2)
            kt_inv = torch.einsum(
                "...ij,...jk->...ik",
                _invert_SE3(viewmats_t_only),
                _lift_K(_invert_K(Ks_norm)),
            )

        if is_t:
            # Create viewmats with R replaced by identity I
            viewmats_t_only = torch.eye(4, device=viewmats.device, dtype=viewmats.dtype).repeat(
                batch, cameras, 1, 1
            )
            viewmats_t_only[..., :3, 3] = viewmats[..., :3, 3]

            # P = [I | t]
            t_mat = viewmats_t_only
            t_mat_T = t_mat.transpose(-1, -2)
            t_mat_inv = _invert_SE3(viewmats_t_only)

        if is_k:
            K_mat = _lift_K(Ks_norm)
            K_mat_T = K_mat.transpose(-1, -2)
            # Inverse of lift(K) is simply lift(K^-1)
            K_mat_inv = _lift_K(_invert_K(Ks_norm))
            assert K_mat.shape == K_mat_inv.shape == (batch, cameras, 4, 4), (
                f"K_mat and K_mat_inv should have shape (batch, cameras, 4, 4), but got {K_mat.shape} and {K_mat_inv.shape}"
            )

        if qk_p_dim is not None:
            if qk_pe in ["p", "p2d"]:
                transforms_q.append((partial(_apply_tiled_projmat, matrix=P_T), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=P_inv), qk_p_dim))
            elif qk_pe == "Rt":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=se3_T), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=se3_inv), qk_p_dim))
            elif qk_pe == "pT2d":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=P), qk_p_dim))
                transforms_k.append(
                    (partial(_apply_tiled_projmat, matrix=P_inv.transpose(-1, -2)), qk_p_dim)
                )
            elif qk_pe == "pIT2d":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=P_inv), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=P_T), qk_p_dim))
            elif qk_pe == "KR2d":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=kso3_T), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=kso3_inv), qk_p_dim))
            elif qk_pe == "R2d":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=so3_T), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=so3_inv), qk_p_dim))
            elif qk_pe == "Rt2d":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=se3_T), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=se3_inv), qk_p_dim))
            elif qk_pe == "KRtAdd2d":
                transforms_q.append(
                    (
                        partial(
                            _apply_split_6d,
                            R_mat=kso3_T[..., :3, :3],
                            t_vec=-P[..., :3, 3],
                            op="add",
                        ),
                        qk_p_dim,
                    )
                )
                transforms_k.append(
                    (
                        partial(
                            _apply_split_6d,
                            R_mat=kso3_inv[..., :3, :3],
                            t_vec=-P[..., :3, 3],
                            op="add",
                        ),
                        qk_p_dim,
                    )
                )
            elif qk_pe == "Kt2d":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=kt_T), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=kt_inv), qk_p_dim))
            elif qk_pe == "t2d":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=t_mat_T), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=t_mat_inv), qk_p_dim))
            elif qk_pe == "K2d":
                transforms_q.append((partial(_apply_tiled_projmat, matrix=K_mat_T), qk_p_dim))
                transforms_k.append((partial(_apply_tiled_projmat, matrix=K_mat_inv), qk_p_dim))
            else:
                raise ValueError(f"Unsupported qk_pe type: {qk_pe}")

        if vo_p_dim is not None:
            if vo_pe in ["p", "p2d"]:
                transforms_v.append((partial(_apply_tiled_projmat, matrix=P_inv), vo_p_dim))
                transforms_o.append((partial(_apply_tiled_projmat, matrix=P), vo_p_dim))
            elif vo_pe == "Rt":
                transforms_v.append((partial(_apply_tiled_projmat, matrix=se3_inv), vo_p_dim))
                transforms_o.append((partial(_apply_tiled_projmat, matrix=se3), vo_p_dim))
            elif vo_pe == "pT2d":
                transforms_v.append(
                    (partial(_apply_tiled_projmat, matrix=P_inv.transpose(-1, -2)), vo_p_dim)
                )
                transforms_o.append((partial(_apply_tiled_projmat, matrix=P_T), vo_p_dim))
            elif vo_pe == "pIT2d":
                transforms_v.append((partial(_apply_tiled_projmat, matrix=P_T), vo_p_dim))
                transforms_o.append(
                    (partial(_apply_tiled_projmat, matrix=P_inv.transpose(-1, -2)), vo_p_dim)
                )
            elif vo_pe == "KR2d":
                transforms_v.append((partial(_apply_tiled_projmat, matrix=kso3_inv), vo_p_dim))
                transforms_o.append((partial(_apply_tiled_projmat, matrix=kso3), vo_p_dim))
            elif vo_pe == "R2d":
                transforms_v.append((partial(_apply_tiled_projmat, matrix=so3_inv), vo_p_dim))
                transforms_o.append((partial(_apply_tiled_projmat, matrix=so3), vo_p_dim))
            elif vo_pe == "Rt2d":
                transforms_v.append((partial(_apply_tiled_projmat, matrix=se3_inv), vo_p_dim))
                transforms_o.append((partial(_apply_tiled_projmat, matrix=se3), vo_p_dim))
            elif vo_pe == "KRtAdd2d":
                transforms_v.append(
                    (
                        partial(
                            _apply_split_6d,
                            R_mat=kso3_inv[..., :3, :3],
                            t_vec=-P[..., :3, 3],
                            op="add",
                        ),
                        vo_p_dim,
                    )
                )
                transforms_o.append(
                    (
                        partial(
                            _apply_split_6d, R_mat=kso3[..., :3, :3], t_vec=P[..., :3, 3], op="add"
                        ),
                        vo_p_dim,
                    )
                )
            elif vo_pe == "Kt2d":
                transforms_v.append((partial(_apply_tiled_projmat, matrix=kt_inv), vo_p_dim))
                transforms_o.append((partial(_apply_tiled_projmat, matrix=kt), vo_p_dim))
            elif vo_pe == "t2d":
                transforms_v.append((partial(_apply_tiled_projmat, matrix=t_mat_inv), vo_p_dim))
                transforms_o.append((partial(_apply_tiled_projmat, matrix=t_mat), vo_p_dim))
            elif vo_pe == "K2d":
                transforms_v.append((partial(_apply_tiled_projmat, matrix=K_mat_inv), vo_p_dim))
                transforms_o.append((partial(_apply_tiled_projmat, matrix=K_mat), vo_p_dim))
            else:
                raise ValueError(f"Unsupported vo_pe type: {vo_pe}")

    # --- 2d patch RoPE ---
    if qk_2d_dim is not None:
        transforms_q.append((partial(_rope_apply_coeffs, coeffs=qk_coeffs_x), qk_2d_dim))
        transforms_q.append((partial(_rope_apply_coeffs, coeffs=qk_coeffs_y), qk_2d_dim))
        transforms_k.append((partial(_rope_apply_coeffs, coeffs=qk_coeffs_x), qk_2d_dim))
        transforms_k.append((partial(_rope_apply_coeffs, coeffs=qk_coeffs_y), qk_2d_dim))
    if vo_2d_dim is not None:
        transforms_v.append((partial(_rope_apply_coeffs, coeffs=vo_coeffs_x), vo_2d_dim))
        transforms_v.append((partial(_rope_apply_coeffs, coeffs=vo_coeffs_y), vo_2d_dim))
        transforms_o.append(
            (partial(_rope_apply_coeffs, coeffs=vo_coeffs_x, inverse=True), vo_2d_dim)
        )
        transforms_o.append(
            (partial(_rope_apply_coeffs, coeffs=vo_coeffs_y, inverse=True), vo_2d_dim)
        )

    apply_fns = []
    for transforms in [transforms_q, transforms_k, transforms_v, transforms_o]:
        if len(transforms) > 0:
            apply_fns.append(partial(_apply_block_diagonal, func_size_pairs=transforms))
        else:
            apply_fns.append(lambda x: x)

    return apply_fns


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, seqlen, num_heads, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply a per-camera projection matrix to feature blocks via tiled einsum."""
    (batch, seqlen, num_heads, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0, (
        f"seqlen should be greater than cameras and divisible by cameras, but got seqlen={seqlen} and cameras={cameras}"
    )
    D = matrix.shape[-1]
    assert matrix.shape == (
        batch,
        cameras,
        D,
        D,
    ), f"matrix should have shape (batch, cameras, D, D), but got {matrix.shape} with D={D}"
    assert feat_dim % D == 0, (
        f"feat_dim must be divisible by {D}, but got feat_dim={feat_dim} and D={D}"
    )

    return torch.einsum(
        "bcij,bcpnkj->bcpnki",
        matrix,
        feats.reshape((batch, cameras, -1, num_heads, feat_dim // D, D)),
    ).reshape(feats.shape)


def _apply_split_6d(
    feats: torch.Tensor,
    R_mat: torch.Tensor,  # (B, C, 3, 3)
    t_vec: torch.Tensor,  # (B, C, 3)
    op: Literal["add", "mul"],
) -> torch.Tensor:
    """Apply function for 6D blocks (3D rotation + 3D translation)."""
    global _WARNING_SHOWN_SPLIT_6D
    (batch, seqlen, num_heads, feat_dim) = feats.shape
    cameras = R_mat.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0, (
        f"seqlen should be greater than cameras and divisible by cameras, but got seqlen={seqlen} and cameras={cameras}"
    )

    D = 6
    # Compute remainder when dividing by 6
    remainder = feat_dim % D

    if remainder != 0:
        if not _WARNING_SHOWN_SPLIT_6D:
            warnings.warn(
                f"feat_dim ({feat_dim}) is not divisible by {D}. "
                f"Applying split_6d to the first {feat_dim - remainder} dimensions "
                f"and appending the remaining {remainder} dimensions unchanged."
            )
            _WARNING_SHOWN_SPLIT_6D = True

        # Split into divisible and remainder dimensions
        feat_dim_div = feat_dim - remainder
        feats_div = feats[..., :feat_dim_div]
        feats_rem = feats[..., feat_dim_div:]
    else:
        feat_dim_div = feat_dim
        feats_div = feats
        feats_rem = None

    # Decompose into 6D blocks (applied only to divisible portion)
    feats_reshaped = feats_div.reshape(batch, cameras, -1, num_heads, feat_dim_div // D, D)
    f_rot = feats_reshaped[..., 0:3]
    f_trans = feats_reshaped[..., 3:6]

    # Apply K+SO3 to the first 3 dimensions
    out_rot = torch.einsum("bcij,bcpnkj->bcpnki", R_mat, f_rot)

    # Apply translation (t) to the last 3 dimensions
    # Reshape to 4D to match _apply_t_op input format
    f_trans_4d = f_trans.reshape(batch, seqlen, num_heads, feat_dim_div // 2)
    out_trans_4d = _apply_t_op(f_trans_4d, t_vec, op=op)
    # Reshape back to decomposed form for concatenation
    out_trans = out_trans_4d.reshape(batch, cameras, -1, num_heads, feat_dim_div // D, 3)

    out = torch.cat([out_rot, out_trans], dim=-1)

    # Reshape back to (batch, seqlen, num_heads, feat_dim_div)
    out = out.reshape(batch, seqlen, num_heads, feat_dim_div)

    # Append remainder dimensions if they exist
    if feats_rem is not None:
        out = torch.cat([out, feats_rem], dim=-1)

    return out


def _apply_t_op(
    feats: torch.Tensor,
    t_vec: torch.Tensor,  # (B, C, 3)
    op: Literal["add", "mul"] = "add",
) -> torch.Tensor:
    """Apply translation (t) operation (add or mul) to all 3D blocks."""
    (batch, seqlen, num_heads, feat_dim) = feats.shape
    cameras = t_vec.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0, (
        f"seqlen should be greater than cameras and divisible by cameras, but got seqlen={seqlen} and cameras={cameras}"
    )

    D = 3
    assert feat_dim % D == 0, f"feat_dim must be divisible by {D}"

    # Decompose into 3D blocks
    feats_reshaped = feats.reshape(batch, cameras, -1, num_heads, feat_dim // D, D)

    # Expand t_vec shape for broadcasting
    t_expanded = t_vec.view(batch, cameras, 1, 1, 1, 3)

    # Apply t to all 3D blocks via broadcasting
    if op == "add":
        out = feats_reshaped + t_expanded
    elif op == "mul":
        out = feats_reshaped * t_expanded
    else:
        raise ValueError(f"Unknown op: {op}")

    return out.reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin RoPE coefficients for given positions."""
    assert len(positions.shape) == 1, (
        f"Expected positions to be a 1D tensor, but got shape {positions.shape}"
    )
    assert feat_dim % 2 == 0, f"feat_dim must be divisible by 2 for RoPE, but got {feat_dim}"
    num_freqs = feat_dim // 2

    freqs = freq_scale * (
        freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs)
    )

    angles = positions[:, None] * freqs[None, :]
    angles = angles.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, num_freqs)
    assert angles.shape == (
        1,
        positions.shape[0],
        1,
        num_freqs,
    ), f"Expected angles to have shape (1, seqlen, 1, num_freqs), but got {angles.shape}"

    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,
    coeffs: tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply precomputed RoPE coefficients to features, or their inverse."""
    cos, sin = coeffs

    if cos.shape[1] != feats.shape[1]:
        n_repeats = feats.shape[1] // cos.shape[1]
        cos = cos.repeat(1, n_repeats, 1, 1)
        sin = sin.repeat(1, n_repeats, 1, 1)

    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4, (
        f"Expected feats, cos, sin to all have 4 dimensions, but got {feats.shape}, {cos.shape}, {sin.shape}"
    )
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2, (
        f"Last dimension of coeffs should be half of feat_dim, but got {cos.shape[-1]} and {feats.shape[-1]}"
    )

    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply separate functions to contiguous blocks of the feature dimension."""
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes), (
        f"Sum of block sizes should match feat_dim, but got {sum(block_sizes)} and {feats.shape[-1]}"
    )
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, (
        f"Input/output shapes should match, but got {out.shape} and {feats.shape}"
    )
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of SE(3) transformation matrices."""
    assert transforms.shape[-2:] == (
        4,
        4,
    ), f"Expected transforms to have shape (..., 4, 4), but got {transforms.shape}"
    Rinv = transforms[..., :3, :3].transpose(-1, -2)  # (..., 3, 3)
    T = transforms[..., :3, 3:]
    T_converted = -torch.matmul(Rinv, T)  # (..., 3, 1)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3:] = T_converted
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift a 3x3 intrinsic matrix to a 4x4 homogeneous matrix."""
    assert Ks.shape[-2:] == (3, 3), f"Expected Ks to have shape (..., 3, 3), but got {Ks.shape}"
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert a 3x3 upper-triangular intrinsic matrix analytically."""
    assert Ks.shape[-2:] == (3, 3), f"Expected Ks to have shape (..., 3, 3), but got {Ks.shape}"
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out
