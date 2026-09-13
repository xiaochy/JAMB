"""
JAMB Policy

Joint action-and-3D-track diffusion for bimanual manipulation. DINOv2
visual patch tokens are fused per-patch with track query tokens (concat +
projection, bridge_mode=concat_selfattn) and denoised together with the
action chunk in one self-attention stack; 4D RoPE (world-frame xyz + time)
is applied to vision/state tokens and, via a noise-blended interpolation
with the current known EEF pose, to the action tokens as well. See the
top-level README for the full method description.

Decoder design (aligned with ACT-DP-TP):
- Custom decoder layer supporting separate query_pos and memory_pos
- Learnable position embedding for action queries
- Diffusion timestep added to memory (not query)

Feature extraction mode:
- Pre-extracted DINOv2 features (from scripts/add_dinov2_to_tracks.py /
  scripts/convert_tracks_to_zarr.py) are used directly, no runtime
  extraction during training.
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
from torch import Tensor

from jamb_policy.model.common.normalizer import LinearNormalizer
from jamb_policy.model.common.rope4d import RoPE4D
from jamb_policy.policy.base_policy import BasePolicy
from jamb_policy.common.pytorch_util import dict_apply
from jamb_policy.common.model_util import print_params


# ============================================================================
# Custom Transformer Decoder supporting separate positional encodings
# (Aligned with ACT-DP-TP implementation)
# ============================================================================

class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head attention with optional 4D RoPE applied to q/k after
    projection. Drop-in replacement for nn.MultiheadAttention(batch_first=True)
    in this file: when no rope arguments are passed it computes plain MHA.

    RoPE args are (cos, sin) tables from RoPE4D.build_cos_sin plus an optional
    rope_mask ([B, N] bool; False = leave that token unrotated). Query and key
    get independent tables so cross-attention can rotate by different
    coordinate sets.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # [B, h, N, hd]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        rope=None,
        q_cos_sin=None,
        k_cos_sin=None,
        q_rope_mask: Optional[Tensor] = None,
        k_rope_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))

        if rope is not None:
            if q_cos_sin is not None:
                q = rope.apply(q, q_cos_sin[0], q_cos_sin[1], rope_mask=q_rope_mask)
            if k_cos_sin is not None:
                k = rope.apply(k, k_cos_sin[0], k_cos_sin[1], rope_mask=k_rope_mask)

        mask = None
        if attn_mask is not None:
            mask = attn_mask if attn_mask.dtype == torch.bool else attn_mask
        if key_padding_mask is not None:
            kp = key_padding_mask[:, None, None, :]  # [B,1,1,N_k], True = ignore
            kp_bias = torch.zeros_like(kp, dtype=q.dtype).masked_fill(kp, float("-inf"))
            mask = kp_bias if mask is None else mask + kp_bias

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # [B, h, N_q, hd]
        B, _, N_q, _ = out.shape
        out = out.transpose(1, 2).reshape(B, N_q, self.nhead * self.head_dim)
        return self.out_proj(out)


class TransformerDecoderLayerWithPE(nn.Module):
    """
    Custom Transformer Decoder Layer that supports separate positional encodings
    for query and memory, following ACT-DP-TP's design.

    Position encodings are only added to Q and K in attention, not to V.
    Optionally applies 4D RoPE (rotary over world-frame x,y,z + time) to q/k:
    self-attention rotates by the query tokens' coordinates; cross-attention
    rotates q by query coords and k by memory coords.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        """Add position embedding to tensor"""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        # Position encodings (separate from features)
        memory_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        # 4D RoPE (optional)
        rope=None,
        query_cos_sin=None,
        memory_cos_sin=None,
        query_rope_mask: Optional[Tensor] = None,
        memory_rope_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt: [B, N_query, D]
            memory: [B, N_memory, D]
            memory_pos: [B, N_memory, D] - Memory positional encoding
            query_pos: [B, N_query, D] - Query positional encoding
            rope / query_cos_sin / memory_cos_sin / *_rope_mask: 4D RoPE
                tables built from token coordinates (see RoPE4D)
        """
        if self.norm_first:
            # Pre-norm (used in GAP)
            # Self-attention: RoPE by query-token coordinates
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(
                q, k, tgt2,  # value不加位置编码
                rope=rope,
                q_cos_sin=query_cos_sin,
                k_cos_sin=query_cos_sin,
                q_rope_mask=query_rope_mask,
                k_rope_mask=query_rope_mask,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
            tgt = tgt + self.dropout1(tgt2)

            # Cross-attention: q rotated by query coords, k by memory coords
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),  # query加位置
                key=self.with_pos_embed(memory, memory_pos),  # key加位置
                value=memory,  # value不加位置！
                rope=rope,
                q_cos_sin=query_cos_sin,
                k_cos_sin=memory_cos_sin,
                q_rope_mask=query_rope_mask,
                k_rope_mask=memory_rope_mask,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = tgt + self.dropout2(tgt2)

            # FFN
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Post-norm (ACT-DP-TP uses this)
            # Self-attention
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(
                q, k, tgt,
                rope=rope,
                q_cos_sin=query_cos_sin,
                k_cos_sin=query_cos_sin,
                q_rope_mask=query_rope_mask,
                k_rope_mask=query_rope_mask,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # Cross-attention
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, memory_pos),
                value=memory,
                rope=rope,
                q_cos_sin=query_cos_sin,
                k_cos_sin=memory_cos_sin,
                q_rope_mask=query_rope_mask,
                k_rope_mask=memory_rope_mask,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # FFN
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt


class TransformerDecoderWithPE(nn.Module):
    """Custom Transformer Decoder supporting separate positional encodings"""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        # Position encodings
        memory_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        # 4D RoPE (optional)
        rope=None,
        query_cos_sin=None,
        memory_cos_sin=None,
        query_rope_mask: Optional[Tensor] = None,
        memory_rope_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt: [B, N_query, D]
            memory: [B, N_memory, D]
            memory_pos: [B, N_memory, D]
            query_pos: [B, N_query, D]
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_pos=memory_pos,
                query_pos=query_pos,
                rope=rope,
                query_cos_sin=query_cos_sin,
                memory_cos_sin=memory_cos_sin,
                query_rope_mask=query_rope_mask,
                memory_rope_mask=memory_rope_mask,
            )

        return output


# ============================================================================
# Transformer Encoder (from ACT-DP-TP) for context encoding
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer with positional encoding support.
    Adapted from ACT-DP-TP.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        """Add position embedding to tensor"""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        rope=None,
        rope_cos_sin=None,
        rope_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: [B, N, D]
            pos: [B, N, D] - positional encoding
            rope / rope_cos_sin / rope_mask: 4D RoPE tables (see RoPE4D)
        """
        if self.norm_first:
            # Pre-norm
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(
                q, k, src2,  # value不加位置编码
                rope=rope,
                q_cos_sin=rope_cos_sin,
                k_cos_sin=rope_cos_sin,
                q_rope_mask=rope_mask,
                k_rope_mask=rope_mask,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self.dropout1(src2)

            # FFN
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Post-norm
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(
                q, k, src,
                rope=rope,
                q_cos_sin=rope_cos_sin,
                k_cos_sin=rope_cos_sin,
                q_rope_mask=rope_mask,
                k_rope_mask=rope_mask,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # FFN
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """Standard Transformer Encoder stack"""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        rope=None,
        rope_cos_sin=None,
        rope_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: [B, N, D]
            pos: [B, N, D]
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
                rope=rope,
                rope_cos_sin=rope_cos_sin,
                rope_mask=rope_mask,
            )

        return output


class TrackTemporalEncoder(nn.Module):
    """Embed a noisy displacement trajectory [B, N, T, 3] into one token per
    point [B, N, out_dim].

    Flattens the T steps into a single (T*3)-d vector and projects with a
    2-layer MLP: no per-step weight sharing, no time embedding, no
    attention/pooling over T. Whatever temporal structure exists must be
    learned implicitly from the fixed positions within the flattened
    vector — only valid because T is fixed at track_horizon for every
    sample. The simplest of the three TrackTemporalEncoder variants.
    """

    def __init__(self, track_horizon, temporal_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(track_horizon * 3, temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, out_dim),
        )

    def forward(self, tracks):
        B, N, T, _ = tracks.shape
        flat = tracks.reshape(B, N, T * 3)
        return self.mlp(flat)


class DualStreamDecoderLayer(nn.Module):
    """One joint-diffusion decoder layer with action and track as two
    SEPARATE token streams (each with its own self-attention and its own
    cross-attention to the observation memory), followed by an explicit
    action<->track "bridge" cross-attention and per-stream FFN.

    H_A~ = SelfAttn_A(H_A) + CrossAttn_{A<-o}(H_A, H_o)
    H_T~ = SelfAttn_T(H_T) + CrossAttn_{T<-o}(H_T, H_o)

    bridge="none" (control: no interaction at all — two fully independent
    streams that only ever share the observation memory):
        H_A' = H_A~
        H_T' = H_T~
    bridge="uni" (track -> action only):
        H_A' = H_A~ + g_TA * CrossAttn_{A<-T}(H_A~, H_T~)
        H_T' = H_T~
    bridge="bi" (bidirectional):
        H_A' = H_A~ + g_TA * CrossAttn_{A<-T}(H_A~, H_T~)
        H_T' = H_T~ + g_AT * CrossAttn_{T<-A}(H_T~, H_A~)

    Both bridge directions read the PRE-bridge tilde values (not each
    other's post-bridge output), so the two directions are computed in
    parallel, not sequentially dependent. Bridge gates are Flamingo-style
    zero-init learnable scalars so the bridge starts as a no-op and the
    model has to learn to open it. bridge="none" doesn't just zero-gate —
    it never constructs the cross-modal attention module at all, so it's
    a clean ablation for "does explicit interaction help beyond just
    giving each modality its own stream capacity".
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        bridge: str = "uni",  # "none" | "uni" | "bi"
    ):
        super().__init__()
        assert bridge in ("none", "uni", "bi"), bridge
        self.bridge = bridge

        def mk_attn():
            return RoPEMultiheadAttention(d_model, nhead, dropout=dropout)

        # Per-stream self-attention + cross-attention to observation memory
        self.self_attn_a = mk_attn()
        self.self_attn_t = mk_attn()
        self.obs_attn_a = mk_attn()
        self.obs_attn_t = mk_attn()

        # Explicit action<->track bridge (bridge="none": no cross-modal
        # module at all, not even a zero-gated one — the two streams never
        # interact beyond sharing the observation memory)
        if bridge in ("uni", "bi"):
            self.bridge_a_from_t = mk_attn()
            self.gate_ta = nn.Parameter(torch.zeros(1))
            self.norm_t_as_kv = nn.LayerNorm(d_model)  # track, normed as bridge key/value
        if bridge == "bi":
            self.bridge_t_from_a = mk_attn()
            self.gate_at = nn.Parameter(torch.zeros(1))
            self.norm_a_as_kv = nn.LayerNorm(d_model)  # action, normed as bridge key/value

        def mk_ffn():
            act = nn.GELU() if activation == 'gelu' else nn.ReLU()
            return nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                act,
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            )
        self.ffn_a = mk_ffn()
        self.ffn_t = mk_ffn()

        # Pre-norm layernorms, 4 sublayers per stream: self-attn, obs
        # cross-attn, bridge (query side), FFN.
        self.norm_a = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.norm_t = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.drop_a = nn.ModuleList([nn.Dropout(dropout) for _ in range(4)])
        self.drop_t = nn.ModuleList([nn.Dropout(dropout) for _ in range(4)])

    def forward(
        self,
        h_a: Tensor,
        h_t: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
        pos_a: Tensor,
        pos_t: Tensor,
        rope=None,
        cos_sin_a=None,
        cos_sin_t=None,
        memory_cos_sin=None,
        rope_mask_a: Optional[Tensor] = None,
        rope_mask_t: Optional[Tensor] = None,
        memory_rope_mask: Optional[Tensor] = None,
    ):
        # 1. Self-attention, per stream (rotated by each stream's own coords)
        a2 = self.norm_a[0](h_a)
        qk = a2 + pos_a
        a2 = self.self_attn_a(
            qk, qk, a2, rope=rope, q_cos_sin=cos_sin_a, k_cos_sin=cos_sin_a,
            q_rope_mask=rope_mask_a, k_rope_mask=rope_mask_a,
        )
        h_a = h_a + self.drop_a[0](a2)

        t2 = self.norm_t[0](h_t)
        qk = t2 + pos_t
        t2 = self.self_attn_t(
            qk, qk, t2, rope=rope, q_cos_sin=cos_sin_t, k_cos_sin=cos_sin_t,
            q_rope_mask=rope_mask_t, k_rope_mask=rope_mask_t,
        )
        h_t = h_t + self.drop_t[0](t2)

        # 2. Cross-attention to observation memory, per stream
        a2 = self.norm_a[1](h_a)
        a2 = self.obs_attn_a(
            query=a2 + pos_a, key=memory + memory_pos, value=memory,
            rope=rope, q_cos_sin=cos_sin_a, k_cos_sin=memory_cos_sin,
            q_rope_mask=rope_mask_a, k_rope_mask=memory_rope_mask,
        )
        h_a = h_a + self.drop_a[1](a2)

        t2 = self.norm_t[1](h_t)
        t2 = self.obs_attn_t(
            query=t2 + pos_t, key=memory + memory_pos, value=memory,
            rope=rope, q_cos_sin=cos_sin_t, k_cos_sin=memory_cos_sin,
            q_rope_mask=rope_mask_t, k_rope_mask=memory_rope_mask,
        )
        h_t = h_t + self.drop_t[1](t2)

        # H_A~, H_T~: post self-attn + obs cross-attn, pre-bridge
        h_a_tilde, h_t_tilde = h_a, h_t

        # 3. Explicit action<->track bridge (skipped entirely for bridge="none")
        if self.bridge in ("uni", "bi"):
            a_q = self.norm_a[2](h_a_tilde)
            t_kv = self.norm_t_as_kv(h_t_tilde)
            bridge_a = self.bridge_a_from_t(
                query=a_q + pos_a, key=t_kv + pos_t, value=t_kv,
                rope=rope, q_cos_sin=cos_sin_a, k_cos_sin=cos_sin_t,
                q_rope_mask=rope_mask_a, k_rope_mask=rope_mask_t,
            )
            h_a = h_a_tilde + self.drop_a[2](self.gate_ta * bridge_a)
        else:
            h_a = h_a_tilde  # no bridge: action stream unaffected by track

        if self.bridge == "bi":
            t_q = self.norm_t[2](h_t_tilde)
            a_kv = self.norm_a_as_kv(h_a_tilde)
            bridge_t = self.bridge_t_from_a(
                query=t_q + pos_t, key=a_kv + pos_a, value=a_kv,
                rope=rope, q_cos_sin=cos_sin_t, k_cos_sin=cos_sin_a,
                q_rope_mask=rope_mask_t, k_rope_mask=rope_mask_a,
            )
            h_t = h_t_tilde + self.drop_t[2](self.gate_at * bridge_t)
        else:
            h_t = h_t_tilde  # uni/none: track stream unaffected by bridge

        # 4. Per-stream FFN
        a2 = self.norm_a[3](h_a)
        a2 = self.ffn_a(a2)
        h_a = h_a + self.drop_a[3](a2)

        t2 = self.norm_t[3](h_t)
        t2 = self.ffn_t(t2)
        h_t = h_t + self.drop_t[3](t2)

        return h_a, h_t


class DualStreamDecoder(nn.Module):
    """Stack of DualStreamDecoderLayer, mirroring TransformerDecoderWithPE's
    deepcopy-from-template construction pattern."""

    def __init__(self, decoder_layer: DualStreamDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, h_a, h_t, memory, memory_pos, pos_a, pos_t, **rope_kwargs):
        for layer in self.layers:
            h_a, h_t = layer(h_a, h_t, memory, memory_pos, pos_a, pos_t, **rope_kwargs)
        return h_a, h_t


class JAMBPolicy(BasePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        # Separate scheduler for the track diffusion branch (identical
        # alpha_bar schedule as noise_scheduler, only clip_sample_range
        # differs - see JAMB.yaml track_noise_scheduler comment). None
        # falls back to sharing noise_scheduler (old behavior).
        track_noise_scheduler: Optional[DDPMScheduler] = None,
        # Feature dimensions (from pre-extracted features)
        # Pi3 feature config
        use_pi3_features=True,    # Whether to use pre-extracted pi3 features
        pi3_feature_dim=1024,     # Pi3 point_hidden feature dimension
        pi3_num_views=1,          # Number of camera views for Pi3 (head, left_wrist, right_wrist)
        pi3_embed_dim=1024,       # Embedding dimension for pi3 features (match with vision_feat_dim)
        # DinoV2 2D-semantic feature config (same 17x23 grid as Pi3; shares
        # patch_centers coordinates for 4D RoPE)
        use_dino_features=False,  # Requires dino_features in the dataset
        dino_feature_dim=1024,    # DinoV2 ViT-L/14 feature dimension
        # Auxiliary task: "track" = 16-step 3D displacement regression;
        # "pmp" = predict pi3 features at the chunk's last frame (old GAP)
        aux_task="track",
        # Track prediction config
        track_horizon=16,
        # Joint diffusion (DiPT plan §9): tracks become a diffusion variable
        # denoised together with actions (same timestep k, same scheduler)
        # instead of a one-shot regression from clean learned queries.
        # Requires the dataset normalizer to scale 3d_track (global
        # isotropic std) — raw cm-scale displacements would be drowned by
        # unit-variance scheduler noise.
        joint_diffusion=False,
        joint_track_loss_weight=0.003, # lambda: track contribution = ~(1/10)*action_loss
        track_temporal_dim=512,        # width of the per-point temporal encoder
        # Decoder architecture for joint_diffusion:
        #   "shared": action+track tokens concatenated into ONE sequence,
        #       one shared self-attention stack (original design).
        #   "separate": two SEPARATE streams (own self-attn + own cross-attn
        #       to observation), NO cross-modal bridge at all — control for
        #       isolating whether separation alone (vs. shared attention)
        #       matters, independent of any explicit interaction.
        #   "unidirectional": same as "separate" but with an explicit
        #       bridge where action reads track (track stream unaffected).
        #   "bidirectional": same as unidirectional, but track also reads
        #       action through its own bridge direction.
        bridge_mode="shared",
        # Movement-weighted track loss: downweight static background patches
        # weight = sigmoid(k*(||endpoint_disp||-tau)); alpha=base weight for static
        movement_loss_tau=0.01,
        movement_loss_alpha=0.0,
        # 4D RoPE config (world-frame x,y,z in meters + action-step time axis)
        use_rope4d=True,
        rope_spatial_base=100.0,
        rope_time_base=100.0,
        rope_spatial_scale=0.5,   # xyz normalization: workspace spans ~±0.5 m
        rope_time_scale=16.0,     # t normalization: action horizon
        # State encoder config
        state_dim=16,             # [L pose7, L grip, R pose7, R grip] — split per arm
        state_embed_dim=1024,     # Must match feature dimensions
        # Transformer encoder config (for context encoding - ACT-DP-TP style)
        encoder_depth=2,
        encoder_heads=8,
        encoder_dim_feedforward=2048,
        encoder_dropout=0.1,
        # Transformer decoder config (for denoising)
        decoder_depth=4,
        decoder_heads=8,
        decoder_dim_feedforward=2048,
        decoder_dropout=0.1,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # Parse action shape
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        # Store feature dimensions (no vision encoder, use pre-extracted features)
        self.use_dino_features = use_dino_features
        self.dino_feature_dim = dino_feature_dim
        vision_feat_dim = dino_feature_dim  # 1024 for vitl14

        # Per-arm tokenization: state and action are split into two tokens per
        # step (one per arm) so each token has a single EEF coordinate for
        # RoPE. Bimanual coordination is handled by full self-attention over
        # all tokens — one joint denoiser, not per-arm denoising.
        assert state_dim % 2 == 0 and action_dim % 2 == 0, \
            f"state_dim/action_dim must split across 2 arms, got {state_dim}/{action_dim}"
        self.arm_state_dim = state_dim // 2   # 8: pose7 + gripper1
        self.arm_action_dim = action_dim // 2

        # State encoder (per-arm, shared weights) + arm identity embedding
        self.state_encoder = nn.Linear(self.arm_state_dim, state_embed_dim)
        self.arm_embed = nn.Embedding(2, state_embed_dim)  # left=0, right=1

        # Pi3 feature encoder (if using pre-extracted features)
        self.use_pi3_features = use_pi3_features
        if use_pi3_features:
            self.pi3_num_views = pi3_num_views
            self.pi3_feature_dim = pi3_feature_dim
            # Learned position embeddings for pi3 features (one per view)
            self.pi3_view_pos_embed = nn.Embedding(pi3_num_views, pi3_embed_dim)
            assert pi3_embed_dim == vision_feat_dim, \
                f"Pi3 embed dim ({pi3_embed_dim}) must match vision dim ({vision_feat_dim})"

        # Check dimensions match for token concatenation
        assert vision_feat_dim == state_embed_dim, \
            f"Vision dim ({vision_feat_dim}) must match state dim ({state_embed_dim})"

        # Feature dimension for decoder (same as vision/state dim)
        self.feature_dim = vision_feat_dim  # 1024

        # CLS token (learnable global context token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        # Learned position embeddings
        self.cls_pos_embed = nn.Embedding(1, self.feature_dim)  # CLS token position
        self.state_pos_embed = nn.Embedding(2, self.feature_dim)  # per-arm state token position

        # ========== Context Encoder (ACT-DP-TP style) ==========
        # Context encoder processes [vision_patches, state] -> enriched features
        # No CLS token - following ACT-DP-TP diffusion encoder design
        encoder_layer = TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            activation='gelu',
            norm_first=True,
        )
        self.context_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_depth,
        )

        # Action query position embedding (learnable, per timestep; the arm
        # identity is added via arm_embed, and 4D RoPE carries the geometry)
        self.action_pos_embed = nn.Embedding(horizon, self.feature_dim)

        # Action embedding (for noised actions, per-arm 8D tokens)
        self.action_embed = nn.Sequential(
            nn.Linear(self.arm_action_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # Timestep embedding (for diffusion timestep, will be concat as token to memory)
        self.timestep_embed = nn.Sequential(
            nn.Linear(256, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # Timestep position embedding (learned, for timestep token - ACT-DP-TP style)
        self.timestep_pos_embed = nn.Embedding(1, self.feature_dim)

        # Custom Transformer decoder (supports separate query_pos and memory_pos)
        assert bridge_mode in (
            "shared", "separate", "unidirectional", "bidirectional", "concat_selfattn",
        ), bridge_mode
        assert bridge_mode == "shared" or joint_diffusion, \
            "bridge_mode != 'shared' requires joint_diffusion=True (dual-stream " \
            "decoder needs the track stream to be a diffusion variable)"
        if bridge_mode == "concat_selfattn":
            assert not (use_dino_features and use_pi3_features), \
                "concat_selfattn fuses ONE 391-patch scene-feature slice per " \
                "track token; having both dino and pi3 features active would " \
                "make that slice ambiguous (782 patch tokens vs. 391 track " \
                "queries) -- pick exactly one visual feature source."
        self.bridge_mode = bridge_mode
        if bridge_mode == "shared":
            decoder_layer = TransformerDecoderLayerWithPE(
                d_model=self.feature_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim_feedforward,
                dropout=decoder_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.transformer_decoder = TransformerDecoderWithPE(
                decoder_layer,
                num_layers=decoder_depth,
            )
        elif bridge_mode == "concat_selfattn":
            # Per-patch fusion: concat(contextualized scene feature, noised-
            # track embedding) -> Linear back to D (mirrors the reference
            # track-only DiT's track_fusion_proj). The fused track tokens,
            # noisy action tokens, contextualized state tokens, and the
            # timestep token are then denoised together in ONE
            # self-attention-only stack -- no separate cross-attention memory.
            self.track_fusion_proj = nn.Linear(2 * self.feature_dim, self.feature_dim)
            joint_dit_layer = TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim_feedforward,
                dropout=decoder_dropout,
                activation='gelu',
                norm_first=True,
            )
            self.joint_dit = TransformerEncoder(
                joint_dit_layer,
                num_layers=decoder_depth,
            )
            cprint(f"  [Concat self-attn decoder] scene+track fused per-patch "
                   f"(concat+proj); action+track+state+timestep denoised in "
                   f"one self-attention stack, no cross-attention", "yellow")
        else:
            layer_bridge = {"separate": "none", "unidirectional": "uni", "bidirectional": "bi"}[bridge_mode]
            dual_layer = DualStreamDecoderLayer(
                d_model=self.feature_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim_feedforward,
                dropout=decoder_dropout,
                activation='gelu',
                bridge=layer_bridge,
            )
            self.dual_stream_decoder = DualStreamDecoder(
                dual_layer,
                num_layers=decoder_depth,
            )
            cprint(f"  [Dual-stream decoder] bridge_mode={bridge_mode}: action and track "
                   f"are separate token streams with an explicit cross-modal bridge", "yellow")

        # Output projection for actions (per-arm 8D per token)
        self.action_head = nn.Linear(self.feature_dim, self.arm_action_dim)

        # 4D RoPE over (x, y, z, t) — shared by encoder and decoder attention
        self.use_rope4d = use_rope4d
        if use_rope4d:
            assert self.feature_dim % encoder_heads == 0 and self.feature_dim % decoder_heads == 0
            enc_head_dim = self.feature_dim // encoder_heads
            dec_head_dim = self.feature_dim // decoder_heads
            assert enc_head_dim == dec_head_dim, \
                "encoder/decoder head dims must match to share one RoPE4D module"
            self.rope = RoPE4D(
                head_dim=dec_head_dim,
                spatial_base=rope_spatial_base,
                time_base=rope_time_base,
                spatial_scale=rope_spatial_scale,
                time_scale=rope_time_scale,
            )
        else:
            self.rope = None

        # ========== Track Prediction ==========
        # Add pi3 feature queries for predicting future 3D tracks
        # if use_pi3_features:
            # Pi3 spatial query structure: 17x23 grid (matching pi3 patch grid)
        self.track_query_height = 17
        self.track_query_width = 23
        self.num_track_queries = self.track_query_height * self.track_query_width  # 17*23 = 391 queries

        # Pi3 feature query embedding (learned queries in 2D structure)
        # Shape: [height, width, feature_dim]
        self.track_query_embed = nn.Parameter(
            torch.randn(self.track_query_height, self.track_query_width, self.feature_dim)
        )

        # Pi3 query position embedding (learnable, different for each query)
        # Shape: [height, width, feature_dim]
        self.track_query_pos_embed = nn.Parameter(
            torch.randn(self.track_query_height, self.track_query_width, self.feature_dim)
        )

        # Auxiliary head (the 391 learned queries are shared; only the
        # readout differs)
        assert aux_task in ("track", "pmp"), aux_task
        self.aux_task = aux_task
        self.track_horizon = track_horizon
        if aux_task == "track":
            # Each query predicts track points for track_horizon steps
            self.track_head = nn.Linear(self.feature_dim, self.track_horizon * 3)
            cprint(f"  [Aux] 3D track prediction: {self.track_query_height}x{self.track_query_width}"
                   f" = {self.num_track_queries} queries, horizon {self.track_horizon}", "yellow")
        else:
            # Each query predicts that patch's pi3 feature at the chunk end
            self.pmp_head = nn.Linear(self.feature_dim, pi3_num_views * pi3_feature_dim)
            cprint(f"  [Aux] PMP (future pi3 feature) prediction: "
                   f"{self.num_track_queries} queries -> {pi3_num_views * pi3_feature_dim}d", "yellow")

        # ========== Joint scene-action diffusion ==========
        assert not joint_diffusion or aux_task == "track", \
            "joint_diffusion requires aux_task='track'"
        self.joint_diffusion = joint_diffusion
        self.joint_track_loss_weight = joint_track_loss_weight
        self.movement_loss_tau = movement_loss_tau
        self.movement_loss_alpha = movement_loss_alpha
        if joint_diffusion:
            # Track token input = temporal encoding of the NOISY trajectory
            # (track_query_embed is bypassed; track_query_pos_embed is kept
            # as the per-patch positional identity). track_head keeps its
            # shape but its output now feeds the DDIM update every step.
            self.track_temporal_encoder = TrackTemporalEncoder(
                track_horizon, track_temporal_dim, self.feature_dim)
            cprint(f"  [Joint diffusion] tracks denoised with actions: "
                   f"temporal encoder dim {track_temporal_dim}, "
                   f"track loss weight {joint_track_loss_weight}", "yellow")

        # Noise scheduler (track branch gets its own instance so its
        # clip_sample_range can differ from action's without touching
        # action's - see track_noise_scheduler docstring above)
        self.noise_scheduler = noise_scheduler
        self.track_noise_scheduler = (
            track_noise_scheduler if track_noise_scheduler is not None else noise_scheduler
        )

        # Normalizer
        self.normalizer = LinearNormalizer()

        # Store config
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        cprint("[JAMB] Configuration:", "cyan")
        cprint(f"  Vision feature dim: {vision_feat_dim}", "cyan")
        cprint(f"  State embed dim: {state_embed_dim}", "cyan")
        cprint(f"  Encoder depth: {encoder_depth} (Feature encoding)", "cyan")
        cprint(f"  Decoder depth: {decoder_depth} (Action denoising)", "cyan")
        cprint(f"  Action dim: {action_dim}", "cyan")
        cprint(f"  Horizon: {horizon}", "cyan")
        cprint(f"  N obs steps: {n_obs_steps}", "cyan")
        cprint(f"  N action steps: {n_action_steps}", "cyan")
        _vision_backbone = "DINOv2" if use_dino_features else ("Pi3" if use_pi3_features else "none")
        cprint(f"  [Architecture] {_vision_backbone} + Transformer Encoder + Diffusion Decoder + Track head", "yellow")
        cprint(f"  [Tokens] Per-arm state/action tokens (2 state, 2x{horizon} action)", "yellow")
        if use_rope4d:
            cprint(f"  [RoPE4D] world-frame (x,y,z) + step t; spatial_scale={rope_spatial_scale}m, "
                   f"time_scale={rope_time_scale}, bases=({rope_spatial_base}, {rope_time_base})", "yellow")
        cprint(f"  [Feature Encoding] Encoder processes [vision, state] -> enriched features", "yellow")
        cprint(f"  [Positional Encoding] Vision: 2D sinusoidal (separate from features)", "green")
        cprint(f"  [Positional Encoding] State: Learned (separate from features)", "green")
        cprint(f"  [Positional Encoding] Action query: Learnable embedding", "green")
        cprint(f"  [Positional Encoding] Timestep: Learned (for timestep token)", "green")
        cprint(f"  [Diffusion Timestep] Concat as token to memory (ACT-DP-TP 'cat' mode)", "green")

        print_params(self)

    def get_sinusoidal_timestep_embedding(self, timesteps, embedding_dim=256):
        """
        Generate sinusoidal timestep embeddings (for diffusion timestep)
        Args:
            timesteps: [B] tensor of timesteps
            embedding_dim: dimension of embedding
        Returns:
            embeddings: [B, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb

    def get_sinusoidal_positional_encoding(self, seq_len, embedding_dim, device):
        """
        Generate sinusoidal positional encodings (for 1D sequence position)
        Args:
            seq_len: length of sequence (horizon)
            embedding_dim: dimension of embedding (feature_dim)
            device: torch device
        Returns:
            positional_encoding: [seq_len, embedding_dim]
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float, device=device)
            * -(torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        pe = torch.zeros(seq_len, embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def get_2d_sinusoidal_positional_encoding(self, height, width, embedding_dim, device, temperature=10000):
        """
        Generate 2D sinusoidal positional encodings for vision patches.
        Similar to DETR's PositionEmbeddingSine.

        Args:
            height: height of patch grid
            width: width of patch grid
            embedding_dim: dimension of embedding (must be even)
            device: torch device
            temperature: temperature for sinusoidal encoding

        Returns:
            pos: [height*width, embedding_dim] positional encodings
        """
        # Create coordinate grids
        y_embed = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, width)
        x_embed = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0).repeat(height, 1)

        # Normalize to [0, 1]
        y_embed = y_embed / height
        x_embed = x_embed / width

        # Scale to [0, 2π]
        y_embed = y_embed * 2 * torch.pi
        x_embed = x_embed * 2 * torch.pi

        # Generate sinusoidal embeddings
        # Split embedding_dim between x and y coordinates
        dim_t = torch.arange(embedding_dim // 4, dtype=torch.float32, device=device)
        dim_t = temperature ** (2 * dim_t / (embedding_dim // 2))

        pos_x = x_embed[:, :, None] / dim_t  # [H, W, D//4]
        pos_y = y_embed[:, :, None] / dim_t  # [H, W, D//4]

        # Interleave sin and cos: [H, W, D//4] -> [H, W, D//2]
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)

        # Concatenate x and y encodings: [H, W, D//2] + [H, W, D//2] -> [H, W, D]
        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, D]
        pos = pos.view(-1, embedding_dim)  # [H*W, D]

        return pos

    def encode_observations(
        self,
        obs_dict: Dict[str, torch.Tensor],
        state_xyz: Optional[torch.Tensor] = None,
        patch_centers: Optional[torch.Tensor] = None,
    ):
        """
        Encode observations to memory context with transformer encoder.
        Uses pre-extracted Pi3 features.

        Token layout: [CLS, state_left, state_right, DinoV2 patches (opt),
        Pi3 patches]. The 16-D state is split into two 8-D per-arm tokens so
        each token has a single EEF coordinate for 4D RoPE. DinoV2 and Pi3
        share the same 17x23 grid, so both reuse patch_centers coordinates.

        Args:
            obs_dict: normalized observations ('agent_pos' [B, 16],
                'pi3_features' [B, N_views, N, 1024])
            state_xyz: [B, 2, 3] UN-normalized world-frame EEF positions
                (left, right) — RoPE coordinates must be raw meters,
                consistent with patch_centers
            patch_centers: [B, N, 3] world-frame patch-center positions

        Returns:
            memory, memory_pos: [B, N_tokens, D]
            memory_coords: [B, N_tokens, 4] or None
            memory_rope_mask: [B, N_tokens] bool or None
        """
        agent_pos = obs_dict['agent_pos']  # [B, 16]
        B = agent_pos.shape[0]
        device = agent_pos.device

        # 1. Process Pi3 features (if available)
        pi3_encoded = None
        if self.use_pi3_features and 'pi3_features' in obs_dict:
            pi3_features = obs_dict['pi3_features']  # [B, N_views, num_patches, 1024] or [B, num_patches, 1024]
            if pi3_features.dim() == 4:
                B_, N_views, N_patches, D = pi3_features.shape
                pi3_encoded = pi3_features.reshape(B_, N_views * N_patches, D)
            else:
                pi3_encoded = pi3_features  # [B, num_patches, D]

        # 1b. Process DinoV2 features (optional, same grid as Pi3)
        dino_encoded = None
        if self.use_dino_features and 'dino_features' in obs_dict:
            dino_features = obs_dict['dino_features']
            if dino_features.dim() == 4:
                B_, N_views, N_patches, D = dino_features.shape
                dino_encoded = dino_features.reshape(B_, N_views * N_patches, D)
            else:
                dino_encoded = dino_features

        # 2. Encode state as two per-arm tokens (shared encoder + arm id)
        arm_states = agent_pos.reshape(B, 2, self.arm_state_dim)  # [B, 2, 8]
        state_features = self.state_encoder(arm_states)  # [B, 2, D]
        state_features = state_features + self.arm_embed.weight.unsqueeze(0)  # [B, 2, D]

        # 3. Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        # 4. Concatenate all features: [CLS, State_L, State_R, Dino, Pi3]
        features_list = [cls_tokens, state_features]
        if dino_encoded is not None:
            features_list.append(dino_encoded)
        if pi3_encoded is not None:
            features_list.append(pi3_encoded)

        encoder_input = torch.cat(features_list, dim=1)  # [B, N_total_tokens, D]

        # 5. Generate position embeddings for all tokens
        # CLS position
        cls_pos = self.cls_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
        # Per-arm state positions
        state_pos = self.state_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 2, D]

        encoder_pos = torch.cat([cls_pos, state_pos], dim=1)  # [B, 3, D]

        grid_pos = self.get_2d_sinusoidal_positional_encoding(
            height=17,
            width=23,
            embedding_dim=self.feature_dim,
            device=device
        ).unsqueeze(0).expand(B, -1, -1)  # [B, 391, D]
        if dino_encoded is not None:
            encoder_pos = torch.cat([encoder_pos, grid_pos], dim=1)
        if pi3_encoded is not None:
            encoder_pos = torch.cat([encoder_pos, grid_pos], dim=1)  # [B, N_tokens, D]

        # 6. Assemble 4D RoPE coordinates (world-frame meters, t=0 for all
        # observation tokens); CLS has no coordinate -> rope_mask False
        memory_coords = None
        memory_rope_mask = None
        rope_cos_sin = None
        if self.use_rope4d and state_xyz is not None:
            N_tokens = encoder_input.shape[1]
            memory_coords = encoder_input.new_zeros(B, N_tokens, 4)
            memory_rope_mask = torch.zeros(B, N_tokens, dtype=torch.bool, device=device)
            # state tokens at indices 1, 2
            memory_coords[:, 1:3, :3] = state_xyz
            memory_rope_mask[:, 1:3] = True
            # visual patch tokens (dino then pi3, both on the same grid)
            idx = 3
            for encoded in (dino_encoded, pi3_encoded):
                if encoded is None:
                    continue
                assert patch_centers is not None, \
                    "use_rope4d requires obs['patch_centers'] for visual tokens"
                n = patch_centers.shape[1]
                memory_coords[:, idx:idx + n, :3] = patch_centers
                memory_rope_mask[:, idx:idx + n] = True
                idx += n
            rope_cos_sin = self.rope.build_cos_sin(memory_coords)

        # 7. Pass through transformer encoder
        encoder_output = self.context_encoder(
            encoder_input,
            pos=encoder_pos,
            rope=self.rope if rope_cos_sin is not None else None,
            rope_cos_sin=rope_cos_sin,
            rope_mask=memory_rope_mask,
        )  # [B, N_tokens, D]

        # 8. Use entire encoder output as memory
        memory = encoder_output  # [B, N_tokens, D]
        memory_pos = encoder_pos  # [B, N_tokens, D]

        return memory, memory_pos, memory_coords, memory_rope_mask

    def forward_diffusion(
        self,
        noised_actions: torch.Tensor,
        timestep: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        memory_coords: Optional[torch.Tensor] = None,
        memory_rope_mask: Optional[torch.Tensor] = None,
        state_xyz: Optional[torch.Tensor] = None,
        patch_centers: Optional[torch.Tensor] = None,
        noised_tracks: Optional[torch.Tensor] = None,
        action_xyz: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of denoising model (ACT-DP-TP 'cat' mode)

        action_xyz: [B, H, 2, 3] world-frame per-(step,arm) RoPE coordinate
            for action tokens -- see _blend_action_xyz: tau-blended between
            the fixed current EEF pose and the position implied by the
            CURRENT NOISY action chunk itself (never ground truth, never a
            model prediction -- same computation at train and inference,
            so no exposure-bias mismatch). Falls back to state_xyz
            broadcast (old behavior: same xyz for all H steps) when None.

        Timestep is concatenated as an additional token to memory (not added),
        with its own learned position embedding.

        Args:
            noised_actions: [B, horizon, action_dim] noised actions
            timestep: [B] diffusion timestep
            memory: [B, N_memory, D] memory features (without position)
            memory_pos: [B, N_memory, D] memory position encodings

        Returns:
            model_output: [B, horizon, action_dim] predicted output
                (noise if prediction_type='epsilon', clean sample if 'sample')
        """
        B = noised_actions.shape[0]
        device = noised_actions.device
        H = self.horizon

        # 1. Embed noised actions as per-arm tokens.
        # [B, H, 16] -> [B, 2H, 8]: first H tokens = left arm, next H = right.
        left = noised_actions[..., : self.arm_action_dim]   # [B, H, 8]
        right = noised_actions[..., self.arm_action_dim :]  # [B, H, 8]
        actions_per_arm = torch.cat([left, right], dim=1)   # [B, 2H, 8]
        action_tgt = self.action_embed(actions_per_arm)     # [B, 2H, D]

        # 1b. Add track queries. The track/pmp aux head is always constructed
        # (see __init__) — it is independent of use_pi3_features, which only
        # controls whether pi3 features are encoded as context tokens.
        if self.joint_diffusion:
            # Joint diffusion: the track token carries the NOISY
            # trajectory (the thing being denoised), not a clean
            # learned query.
            assert noised_tracks is not None, \
                "joint_diffusion=True requires noised_tracks"
            track_tgt = self.track_temporal_encoder(noised_tracks)  # [B, N, D]
        else:
            # Flatten 2D pi3 queries: [height, width, D] -> [height*width, D]
            track_tgt_flat = self.track_query_embed.reshape(-1, self.feature_dim)  # [num_track_queries, D]
            # Expand for batch
            track_tgt = track_tgt_flat.unsqueeze(0).expand(B, -1, -1)  # [B, num_track_queries, D]
        # Concatenate action queries and track queries (only used by the
        # "shared" decoder path; dual-stream keeps action_tgt/track_tgt apart)
        tgt = torch.cat([action_tgt, track_tgt], dim=1)  # [B, 2H+num_track_queries, D]

        # 2. Embed diffusion timestep as a token (ACT-DP-TP 'cat' mode)
        t_emb = self.get_sinusoidal_timestep_embedding(timestep)  # [B, 256]
        t_emb = self.timestep_embed(t_emb)  # [B, D]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, D]

        # Get timestep position embedding
        t_pos = self.timestep_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        # Concat timestep token to memory (as an additional token)
        memory = torch.cat([memory, t_emb], dim=1)  # [B, N_memory+1, D]
        memory_pos = torch.cat([memory_pos, t_pos], dim=1)  # [B, N_memory+1, D]

        # 3. Get learnable query position embeddings.
        # Per-timestep embedding repeated for both arms + arm identity.
        step_pos = self.action_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, H, D]
        arm_pos = self.arm_embed.weight  # [2, D]
        action_query_pos = torch.cat(
            [step_pos + arm_pos[0], step_pos + arm_pos[1]], dim=1
        )  # [B, 2H, D]

        # Learned position embeddings for track queries (different for each query)
        # Flatten 2D position embeddings: [height, width, D] -> [height*width, D]
        track_query_pos_flat = self.track_query_pos_embed.reshape(-1, self.feature_dim)  # [num_track_queries, D]
        # Expand for batch
        track_query_pos = track_query_pos_flat.unsqueeze(0).expand(B, -1, -1)  # [B, num_track_queries, D]
        query_pos = torch.cat([action_query_pos, track_query_pos], dim=1)  # [B, 2H+num_track_queries, D]

        if self.bridge_mode == "shared":
            # 3b. 4D RoPE tables. Action token (arm a, step h): xyz = arm a's
            # CURRENT EEF position (coords must be known at denoising time — the
            # future pose is the thing being predicted), t = h+1. Track queries:
            # their patch center, t = 0 (one token predicts all H steps).
            # The appended diffusion-timestep memory token gets no rotation.
            rope = None
            query_cos_sin = None
            memory_cos_sin = None
            query_rope_mask = None
            mem_rope_mask = None
            if self.use_rope4d and state_xyz is not None and memory_coords is not None:
                rope = self.rope
                N_q = tgt.shape[1]
                query_coords = tgt.new_zeros(B, N_q, 4)
                query_rope_mask = torch.zeros(B, N_q, dtype=torch.bool, device=device)
                t_steps = torch.arange(1, H + 1, device=device, dtype=tgt.dtype)  # [H]
                for arm in (0, 1):
                    sl = slice(arm * H, (arm + 1) * H)
                    if action_xyz is not None:
                        query_coords[:, sl, :3] = action_xyz[:, :, arm, :]  # [B,H,3] per-step estimate
                    else:
                        query_coords[:, sl, :3] = state_xyz[:, arm : arm + 1, :]  # broadcast [B,1,3]
                    query_coords[:, sl, 3] = t_steps
                query_rope_mask[:, : 2 * H] = True
                if patch_centers is not None:
                    query_coords[:, 2 * H :, :3] = patch_centers
                    query_rope_mask[:, 2 * H :] = True
                query_cos_sin = self.rope.build_cos_sin(query_coords)

                # Memory grew by the timestep token: extend coords/mask with a
                # masked-out entry
                mem_coords_ext = torch.cat(
                    [memory_coords, memory_coords.new_zeros(B, 1, 4)], dim=1
                )
                mem_rope_mask = torch.cat(
                    [memory_rope_mask,
                     torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1
                )
                memory_cos_sin = self.rope.build_cos_sin(mem_coords_ext)

            # 4. Pass through custom transformer decoder with separate positional encodings
            # Position encodings are only added in Q/K, not in V (see TransformerDecoderLayerWithPE)
            decoded = self.transformer_decoder(
                tgt=tgt,              # action + pi3 features without position
                memory=memory,        # memory with timestep token concatenated
                memory_pos=memory_pos,  # memory position (includes timestep position)
                query_pos=query_pos,    # query position (action + pi3, separate, learnable)
                rope=rope,
                query_cos_sin=query_cos_sin,
                memory_cos_sin=memory_cos_sin,
                query_rope_mask=query_rope_mask,
                memory_rope_mask=mem_rope_mask,
            )  # [B, 2H+num_track_queries, D]

            # 5. Split decoded features and project to respective spaces.
            # Reassemble per-arm outputs back to [B, H, 16].
            action_decoded = decoded[:, : 2 * H, :]  # [B, 2H, D]
            per_arm_out = self.action_head(action_decoded)  # [B, 2H, 8]
            model_output = torch.cat(
                [per_arm_out[:, :H, :], per_arm_out[:, H:, :]], dim=-1
            )  # [B, H, 16]

            track_decoded = decoded[:, 2 * H :, :]  # [B, num_track_queries, D]
        elif self.bridge_mode == "concat_selfattn":
            # Concat-fusion decoder: each of the num_track_queries patches
            # gets ONE token built by concatenating its contextualized scene
            # feature (memory's patch slice, computed once by
            # encode_observations) with its noised-track embedding
            # (track_tgt), then projecting back to D (track_fusion_proj) —
            # same fusion as the reference track-only DiT design. That fused
            # token, the noisy action tokens, the contextualized state
            # tokens (memory's state slice), and the timestep token are all
            # concatenated into ONE sequence and denoised with a single
            # self-attention stack — no cross-attention to any separate
            # memory at all (context already lives inside the fused token).
            patch_scene_feat = memory[:, 3 : 3 + self.num_track_queries, :]  # [B, num_track_queries, D]
            fused_track = self.track_fusion_proj(
                torch.cat([patch_scene_feat, track_tgt], dim=-1)
            )  # [B, num_track_queries, D]
            state_tokens = memory[:, 1:3, :]  # [B, 2, D] contextualized per-arm state
            state_pos = memory_pos[:, 1:3, :]  # [B, 2, D]

            seq = torch.cat([action_tgt, fused_track, state_tokens, t_emb], dim=1)
            seq_pos = torch.cat([action_query_pos, track_query_pos, state_pos, t_pos], dim=1)
            N_total = seq.shape[1]  # 2H + num_track_queries + 2 + 1

            rope = None
            rope_cos_sin = None
            rope_mask = None
            if self.use_rope4d and state_xyz is not None and patch_centers is not None:
                rope = self.rope
                coords = seq.new_zeros(B, N_total, 4)
                rope_mask = torch.zeros(B, N_total, dtype=torch.bool, device=device)

                # Action tokens: ESTIMATED clean action xyz per step (ground
                # truth at train time, previous denoising step's estimate at
                # inference), falling back to arm's current EEF xyz
                # broadcast across all H steps when no estimate is available
                # yet (first denoising step). t = step index.
                t_steps = torch.arange(1, H + 1, device=device, dtype=seq.dtype)  # [H]
                for arm in (0, 1):
                    sl = slice(arm * H, (arm + 1) * H)
                    if action_xyz is not None:
                        coords[:, sl, :3] = action_xyz[:, :, arm, :]  # [B,H,3]
                    else:
                        coords[:, sl, :3] = state_xyz[:, arm : arm + 1, :]
                    coords[:, sl, 3] = t_steps
                rope_mask[:, : 2 * H] = True

                # Fused track tokens: patch center, t = 0.
                track_sl = slice(2 * H, 2 * H + self.num_track_queries)
                coords[:, track_sl, :3] = patch_centers
                rope_mask[:, track_sl] = True

                # State tokens: arm's current EEF xyz, t = 0.
                state_sl = slice(
                    2 * H + self.num_track_queries, 2 * H + self.num_track_queries + 2
                )
                coords[:, state_sl, :3] = state_xyz
                rope_mask[:, state_sl] = True

                # Timestep token (last position) keeps rope_mask=False: no
                # natural coordinate, passes through unrotated.
                rope_cos_sin = self.rope.build_cos_sin(coords)

            decoded = self.joint_dit(
                seq,
                pos=seq_pos,
                rope=rope,
                rope_cos_sin=rope_cos_sin,
                rope_mask=rope_mask,
            )  # [B, N_total, D] -- self-attention only, no cross-attention

            action_decoded = decoded[:, : 2 * H, :]  # [B, 2H, D]
            per_arm_out = self.action_head(action_decoded)  # [B, 2H, 8]
            model_output = torch.cat(
                [per_arm_out[:, :H, :], per_arm_out[:, H:, :]], dim=-1
            )  # [B, H, 16]

            track_decoded = decoded[:, 2 * H : 2 * H + self.num_track_queries, :]  # [B, num_track_queries, D]
        else:
            # Dual-stream decoder: action and track NEVER get concatenated
            # into one sequence — each keeps its own tokens, own position
            # embeddings, own 4D-RoPE coordinates, and reads the other
            # stream only through DualStreamDecoderLayer's explicit bridge.
            rope = None
            cos_sin_a = cos_sin_t = memory_cos_sin = None
            rope_mask_a = rope_mask_t = mem_rope_mask = None
            if self.use_rope4d and state_xyz is not None and memory_coords is not None:
                rope = self.rope
                N_t = track_tgt.shape[1]

                coords_a = action_tgt.new_zeros(B, 2 * H, 4)
                rope_mask_a = torch.ones(B, 2 * H, dtype=torch.bool, device=device)
                t_steps = torch.arange(1, H + 1, device=device, dtype=action_tgt.dtype)  # [H]
                for arm in (0, 1):
                    sl = slice(arm * H, (arm + 1) * H)
                    if action_xyz is not None:
                        coords_a[:, sl, :3] = action_xyz[:, :, arm, :]
                    else:
                        coords_a[:, sl, :3] = state_xyz[:, arm : arm + 1, :]
                    coords_a[:, sl, 3] = t_steps
                cos_sin_a = self.rope.build_cos_sin(coords_a)

                coords_t = track_tgt.new_zeros(B, N_t, 4)
                rope_mask_t = torch.zeros(B, N_t, dtype=torch.bool, device=device)
                if patch_centers is not None:
                    coords_t[:, :, :3] = patch_centers
                    rope_mask_t[:, :] = True
                cos_sin_t = self.rope.build_cos_sin(coords_t)

                mem_coords_ext = torch.cat(
                    [memory_coords, memory_coords.new_zeros(B, 1, 4)], dim=1
                )
                mem_rope_mask = torch.cat(
                    [memory_rope_mask,
                     torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1
                )
                memory_cos_sin = self.rope.build_cos_sin(mem_coords_ext)

            action_decoded, track_decoded = self.dual_stream_decoder(
                h_a=action_tgt,
                h_t=track_tgt,
                memory=memory,
                memory_pos=memory_pos,
                pos_a=action_query_pos,
                pos_t=track_query_pos,
                rope=rope,
                cos_sin_a=cos_sin_a,
                cos_sin_t=cos_sin_t,
                memory_cos_sin=memory_cos_sin,
                rope_mask_a=rope_mask_a,
                rope_mask_t=rope_mask_t,
                memory_rope_mask=mem_rope_mask,
            )  # [B, 2H, D], [B, num_track_queries, D]

            per_arm_out = self.action_head(action_decoded)  # [B, 2H, 8]
            model_output = torch.cat(
                [per_arm_out[:, :H, :], per_arm_out[:, H:, :]], dim=-1
            )  # [B, H, 16]

        if self.aux_task == "track":
            # Each query predicts track points: [B, N, track_horizon*3]
            track_output_flat = self.track_head(track_decoded)
            aux_output = track_output_flat.reshape(
                B, self.num_track_queries, self.track_horizon, 3)
        else:
            # PMP: each query predicts the future pi3 feature of its patch
            aux_output = self.pmp_head(track_decoded)  # [B, N, N_views*pi3_dim]
        return model_output, aux_output

    # ========= inference  ============
    def conditional_sample(
        self,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        memory_coords: Optional[torch.Tensor] = None,
        memory_rope_mask: Optional[torch.Tensor] = None,
        state_xyz: Optional[torch.Tensor] = None,
        patch_centers: Optional[torch.Tensor] = None,
        generator=None,
        **kwargs,
    ):
        """
        Sample actions using DDPM

        Args:
            memory: [B, N_memory, D] memory features
            memory_pos: [B, N_memory, D] memory position encodings

        Returns:
            actions: [B, horizon, action_dim] denoised actions
        """
        B = memory.shape[0]
        device = memory.device
        dtype = memory.dtype

        # Start from pure noise
        actions = torch.randn(
            (B, self.horizon, self.action_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Joint diffusion: tracks start from noise too and are denoised in
        # lockstep with the actions (normalized space; caller unnormalizes)
        tracks = None
        if self.joint_diffusion:
            tracks = torch.randn(
                (B, self.num_track_queries, self.track_horizon, 3),
                device=device,
                dtype=dtype,
                generator=generator,
            )

        # Set timesteps (track_noise_scheduler has the identical alpha_bar
        # schedule as noise_scheduler, so this produces the same .timesteps
        # sequence - action and track stay in lockstep at each step)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        if self.track_noise_scheduler is not self.noise_scheduler:
            self.track_noise_scheduler.set_timesteps(self.num_inference_steps)

        # Iterative denoising. action_xyz is recomputed EVERY step from the
        # CURRENT noisy `actions` tensor itself (tau-blended with the fixed
        # current EEF position via _blend_action_xyz) -- same function,
        # same logic as compute_loss's training-time noised_actions case,
        # so there's no train/inference asymmetry. At the first step
        # `actions` is pure noise and tau≈0, so the blend naturally
        # collapses to ~the fixed current position with no special-casing
        # needed; no dependence on any model prediction at all (matches the
        # cited paper's a^tau definition, which reads off the noisy input,
        # not a denoised estimate).
        track_pred = None
        for t in self.noise_scheduler.timesteps:
            # Prepare timestep
            timestep = torch.full((B,), t, device=device, dtype=torch.long)

            action_xyz = None
            if state_xyz is not None:
                action_xyz = self._blend_action_xyz(state_xyz, actions, timestep)

            # Predict noise (and track if enabled)
            forward_output = self.forward_diffusion(
                noised_actions=actions,
                timestep=timestep,
                memory=memory,
                memory_pos=memory_pos,
                memory_coords=memory_coords,
                memory_rope_mask=memory_rope_mask,
                state_xyz=state_xyz,
                patch_centers=patch_centers,
                noised_tracks=tracks,
                action_xyz=action_xyz,
            )

            model_output, track_pred = forward_output

            # Denoise (DDIMScheduler.step is stateless — safe to call twice)
            actions = self.noise_scheduler.step(
                model_output,
                t,
                actions,
            ).prev_sample
            if self.joint_diffusion:
                tracks = self.track_noise_scheduler.step(
                    track_pred,
                    t,
                    tracks,
                ).prev_sample

        if self.joint_diffusion:
            track_pred = tracks  # fully denoised trajectory (normalized)

        return actions, track_pred

    def _extract_rope_coords(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Pull 4D-RoPE coordinates from UN-normalized observations.

        Returns:
            state_xyz: [B, 2, 3] world-frame EEF xyz (left, right) or None
            patch_centers: [B, N, 3] world-frame patch centers or None
        """
        if not self.use_rope4d:
            return None, None
        agent_pos = obs_dict["agent_pos"]  # [B, 16] raw EEF poses
        state_xyz = torch.stack(
            [agent_pos[:, 0:3], agent_pos[:, self.arm_state_dim : self.arm_state_dim + 3]],
            dim=1,
        )  # [B, 2, 3]
        patch_centers = obs_dict.get("patch_centers", None)  # [B, N, 3]
        if patch_centers is not None and patch_centers.dim() == 4:
            patch_centers = patch_centers.squeeze(1)  # drop N_views dim
        return state_xyz, patch_centers

    def _blend_action_xyz(
        self,
        state_xyz: torch.Tensor,
        noised_actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        tau-blended action-token RoPE coordinate (see plan doc for the
        cited-paper derivation): interpolates from the fixed CURRENT EEF
        position toward the position implied by the CURRENT NOISY action
        chunk itself, weighted by how far along denoising we are. NEVER
        uses ground truth -- noised_actions is either the training-time
        noised sample at the randomly-sampled timestep, or the
        inference-time trajectory's current (partially-denoised) state,
        so this is identical logic and identical statistics at train and
        test time (no exposure-bias mismatch).

        Args:
            state_xyz: [B, 2, 3] world-frame current EEF xyz (left, right)
            noised_actions: [B, H, action_dim] NORMALIZED noisy action
                chunk at `timesteps` (the actual diffusion input, not a
                model prediction)
            timesteps: [B] int diffusion timesteps (higher = noisier)

        Returns:
            action_xyz: [B, H, 2, 3] world-frame per-(step,arm) coordinate
        """
        est_action = self.normalizer["action"].unnormalize(noised_actions)  # [B, H, action_dim]
        est_xyz = torch.stack(
            [
                est_action[..., 0:3],
                est_action[..., self.arm_action_dim : self.arm_action_dim + 3],
            ],
            dim=2,
        )  # [B, H, 2, 3]

        # tau=0 at max noise (fully on the fixed current position) -> tau=1
        # at zero noise (fully on the dynamic estimate). Linear in the raw
        # integer timestep; simplest faithful reading of the cited paper's
        # "interpolates from the gripper position toward the estimated
        # future position as the action chunk denoises".
        T = self.noise_scheduler.config.num_train_timesteps
        tau = 1.0 - timesteps.to(est_xyz.dtype) / T  # [B]
        tau = tau.view(-1, 1, 1, 1)  # broadcast over [B, H, 2, 3]

        g = state_xyz.unsqueeze(1).expand(-1, est_xyz.shape[1], -1, -1)  # [B, H, 2, 3]
        return (1.0 - tau) * g + tau * est_xyz

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """
        Predict actions from observations (single frame)

        Args:
            obs_dict: dict with keys:
                - 'dino_features': [B, N_views, num_patches, D] pre-extracted DinoV2 features (optional)
                - 'pi3_features': [B, N_views, num_patches, 1024] pre-extracted Pi3 features (optional)
                - 'agent_pos': [B, 14] robot state
            generator: optional torch.Generator for the diffusion sampling noise
                (conditional_sample's initial noise + any stochastic scheduler
                steps). Without this, eval-time rollouts are NOT reproducible
                run-to-run even with a fixed environment seed, since the
                denoising trajectory itself is drawn from unseeded torch.randn.

        Returns:
            result: dict with 'action' key
        """
        # RoPE coordinates come from the RAW (unnormalized) observations:
        # world-frame meters, consistent with patch centers
        state_xyz, patch_centers = self._extract_rope_coords(obs_dict)

        # Normalize input
        nobs = self.normalizer.normalize(obs_dict)

        # Encode observations to memory context (single frame)
        memory, memory_pos, memory_coords, memory_rope_mask = self.encode_observations(
            nobs, state_xyz=state_xyz, patch_centers=patch_centers
        )

        # Sample actions (and track if enabled)
        sample_output = self.conditional_sample(
            memory=memory,
            memory_pos=memory_pos,
            memory_coords=memory_coords,
            memory_rope_mask=memory_rope_mask,
            state_xyz=state_xyz,
            patch_centers=patch_centers,
            generator=generator,
            **self.kwargs,
        )

        naction_pred, track_pred = sample_output

        # Unnormalize
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Extract action steps to execute
        action = action_pred[:, :self.n_action_steps]

        result = {
            "action": action,
            "action_pred": action_pred,
        }

        if self.aux_task == "track":
            if self.joint_diffusion and track_pred is not None:
                # sampled in normalized space -> back to meters
                _tnorm_key = "gt_3d_track" if "gt_3d_track" in self.normalizer.params_dict else "3d_track"
                track_pred = self.normalizer[_tnorm_key].unnormalize(track_pred)
            result["track_pred"] = track_pred
        else:
            result["pi3_features_pred"] = track_pred

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Compute diffusion loss (single frame per batch)

        Args:
            batch: dict with 'obs' and 'action'
                obs: dict with keys:
                    - 'dino_features': [B, N_views, num_patches, D] pre-extracted DinoV2 features (optional)
                    - 'pi3_features': [B, N_views, num_patches, 1024] pre-extracted Pi3 features (optional)
                    - 'agent_pos': [B, 14] robot state
                action: [B, horizon, action_dim]

        Returns:
            loss: scalar tensor
            loss_dict: dict with loss components
        """
        # RoPE coordinates from RAW observations (world-frame meters)
        state_xyz, patch_centers = self._extract_rope_coords(batch["obs"])

        # Normalize
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        B = nactions.shape[0]

        # Encode observations to memory (single frame)
        memory, memory_pos, memory_coords, memory_rope_mask = self.encode_observations(
            nobs, state_xyz=state_xyz, patch_centers=patch_centers
        )  # [B, N_memory, D]

        # Sample timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=nactions.device,
        ).long()

        # Sample noise
        noise = torch.randn_like(nactions)

        # Add noise to actions
        noised_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)

        # Action RoPE coordinate: tau-blended between the fixed current EEF
        # position and the position implied by THIS noised action chunk
        # itself (never ground truth -- see _blend_action_xyz).
        action_xyz = None
        if state_xyz is not None:
            action_xyz = self._blend_action_xyz(state_xyz, noised_actions, timesteps)

        # Joint diffusion: noise the (normalized) tracks with the SAME
        # timesteps so both modalities sit at the same noise level
        noised_tracks = None
        ntracks = None
        track_noise = None
        if self.joint_diffusion and "gt_3d_track" in batch:
            _tnorm_key = "gt_3d_track" if "gt_3d_track" in self.normalizer.params_dict else "3d_track"
            ntracks = self.normalizer[_tnorm_key].normalize(batch["gt_3d_track"])
            track_noise = torch.randn_like(ntracks)
            noised_tracks = self.track_noise_scheduler.add_noise(
                ntracks, track_noise, timesteps)

        # Predict noise (and pi3 features if enabled)
        forward_output = self.forward_diffusion(
            noised_actions=noised_actions,
            timestep=timesteps,
            memory=memory,
            memory_pos=memory_pos,
            memory_coords=memory_coords,
            memory_rope_mask=memory_rope_mask,
            state_xyz=state_xyz,
            patch_centers=patch_centers,
            noised_tracks=noised_tracks,
            action_xyz=action_xyz,
        )

        pred, track_pred = forward_output

        # Action loss
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = nactions
        elif pred_type == "v_prediction":
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * nactions
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        action_loss = F.mse_loss(pred, target, reduction="none")
        action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
        action_loss = action_loss.mean()

        loss_dict = {
            "action_loss": action_loss.item(),
        }

        # Auxiliary loss
        if self.aux_task == "pmp" and "future_pi3_features" in batch:
            future_pi3 = batch["future_pi3_features"]  # [B, N_views, N, pi3_dim]
            B_p, N_views, num_patches, pi3_dim = future_pi3.shape
            future_flat = future_pi3.permute(0, 2, 1, 3).reshape(
                B_p, num_patches, N_views * pi3_dim)
            pmp_loss = F.mse_loss(track_pred, future_flat, reduction="mean")
            loss_dict["pi3_loss"] = pmp_loss.item()
            pi3_loss_weight = 0.1  # same weighting as the original GAP PMP
            loss = action_loss + pi3_loss_weight * pmp_loss
            loss_dict["total_loss"] = loss.item()
            return loss, loss_dict

        # Track prediction loss (if enabled)
        if self.aux_task == "track" and "gt_3d_track" in batch:
            if self.joint_diffusion:
                # Diffusion target in NORMALIZED space, following the same
                # prediction_type convention as the action branch
                if pred_type == "sample":
                    track_target = ntracks
                elif pred_type == "epsilon":
                    track_target = track_noise
                else:
                    raise ValueError(
                        f"joint_diffusion unsupported for prediction_type={pred_type}")
                # After normalization the two losses are on comparable
                # scales; weight starts at parity (config joint_track_loss_weight)
                track_loss_weight = self.joint_track_loss_weight
            else:
                # One-shot regression against raw (meter-scale) displacements
                track_target = batch["gt_3d_track"]  # [B, N_patches, H_action, 3]
                track_loss_weight = 0.1  # auxiliary-task weight (unnormalized scale)

            # Movement-weighted track loss: patches that move more get higher
            # weight; weight computed from RAW (meter-scale) GT track so that
            # the threshold (tau=1cm) is meaningful regardless of normalization.
            gt_raw = batch["gt_3d_track"]  # [B, N, H, 3] meters, always available
            tau = self.movement_loss_tau    # default 0.01 m
            k = 5.0 / tau
            disp_norm = gt_raw[:, :, -1, :].norm(dim=-1)          # [B, N]
            move_weight = torch.sigmoid(k * (disp_norm - tau))     # [B, N]
            alpha = self.movement_loss_alpha                        # default 0.0
            per_patch_weight = alpha + (1 - alpha) * move_weight   # [B, N]

            track_loss = F.mse_loss(track_pred, track_target, reduction="none")  # [B, N, H, 3]
            pw = per_patch_weight.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
            track_loss = (track_loss * pw).sum() / pw.sum().clamp(min=1e-6) / 3

            loss_dict["track_loss"] = track_loss.item()
            loss_dict["move_weight_mean"] = move_weight.mean().item()

            # Combined loss with weighting
            loss = action_loss + track_loss_weight * track_loss
            loss_dict["total_loss"] = loss.item()
        else:
            loss = action_loss

        return loss, loss_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
