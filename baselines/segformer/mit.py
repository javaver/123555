"""Mix Transformer (MiT) hierarchical encoder for SegFormer.

Faithful re-implementation of the encoder proposed in:

    Xie et al. "SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers." NeurIPS 2021.

Design (per stage, strides 1/4, 1/8, 1/16, 1/32):
  - OverlapPatchEmbed: overlapping patch embedding (7x7/s4 for stage 1,
    3x3/s2 for the remaining stages);
  - Efficient Self-Attention: key/value Spatial Reduction (SR) conv with
    per-stage ratios (8, 4, 2, 1);
  - Mix-FFN: Linear -> 3x3 depth-wise conv -> GELU -> Linear (the depth-wise
    conv supplies the explicit position information);

Module names (patch_embed1..4 / block1..4 / norm1..4) deliberately match the
official NVlabs/SegFormer release, so released ImageNet-1k checkpoints
(``mit_b0.pth`` .. ``mit_b5.pth``) load directly via ``load_state_dict``.

NOTE: current timm releases ship no SegFormer/MiT models, hence the encoder is
implemented in-repo instead of relying on ``timm.create_model``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# MiT-B0 .. MiT-B5 configurations (Sec. 4.1 / official configs.json)
MIT_CONFIGS = {
    "mit_b0": dict(embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2]),
    "mit_b1": dict(embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2]),
    "mit_b2": dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3]),
    "mit_b3": dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3]),
    "mit_b4": dict(embed_dims=[64, 128, 320, 512], depths=[3, 8, 27, 3]),
    "mit_b5": dict(embed_dims=[64, 128, 320, 512], depths=[3, 6, 40, 3]),
}
MLP_RATIOS = [4, 4, 4, 4]
NUM_HEADS = [1, 2, 5, 8]
SR_RATIOS = [8, 4, 2, 1]


class DropPath(nn.Module):
    """Stochastic depth per sample (identity on the residual when dropped)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class DWConv(nn.Module):
    """3x3 depth-wise conv used inside Mix-FFN (position-aware token mixing)."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    """Mix-FFN: Linear -> DWConv -> GELU -> Linear."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.drop(self.act(x))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    """Efficient Self-Attention with Spatial Reduction (SR) on keys/values."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Transformer block: SR-Attention + Mix-FFN, both with DropPath."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        sr_ratio: int = 1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image-to-patch embedding with overlapping patches + LayerNorm."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        patch_size: int = 7,
        stride: int = 4,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class MiTEncoder(nn.Module):
    """Mix Transformer (MiT) hierarchical encoder.

    Returns the 4-stage pyramid (1/4, 1/8, 1/16, 1/32) like a
    ``timm`` ``features_only=True`` backbone; use :meth:`channels` for the
    per-stage widths.
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: List[int] = (64, 128, 320, 512),
        depths: List[int] = (3, 4, 6, 3),
        num_heads: List[int] = NUM_HEADS,
        mlp_ratios: List[int] = MLP_RATIOS,
        sr_ratios: List[int] = SR_RATIOS,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_stages = len(depths)
        embed_dims, depths = list(embed_dims), list(depths)

        # stochastic depth decay rule across all blocks
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        cur = 0
        for i in range(self.num_stages):
            if i == 0:
                patch_embed = OverlapPatchEmbed(
                    in_chans=in_chans, embed_dim=embed_dims[i],
                    patch_size=7, stride=4,
                )
            else:
                patch_embed = OverlapPatchEmbed(
                    in_chans=embed_dims[i - 1], embed_dim=embed_dims[i],
                    patch_size=3, stride=2,
                )
            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=True,
                        qk_scale=None,
                        drop_path=dpr[cur + j],
                        sr_ratio=sr_ratios[i],
                        norm_layer=lambda c: nn.LayerNorm(c, eps=1e-6),
                    )
                    for j in range(depths[i])
                ]
            )
            norm = nn.LayerNorm(embed_dims[i], eps=1e-6)
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            cur += depths[i]

        self.embed_dims = embed_dims

    def channels(self) -> List[int]:
        """Per-stage output widths (timm feature_info-compatible helper)."""
        return list(self.embed_dims)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs: List[torch.Tensor] = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


def mit_backbone(name: str, in_chans: int = 3, drop_path_rate: float = 0.1) -> MiTEncoder:
    """Build a MiT-Bx encoder from its canonical name (mit_b0..mit_b5)."""
    key = name.lower().replace("-", "_").strip()
    if key not in MIT_CONFIGS:
        raise ValueError(
            f"Unknown MiT backbone: {name!r}. Available: {sorted(MIT_CONFIGS)}"
        )
    cfg = MIT_CONFIGS[key]
    return MiTEncoder(
        in_chans=in_chans, drop_path_rate=drop_path_rate, **cfg
    )


# ---------------------------------------------------------------------------
# ImageNet 预训练权重加载: 支持三种发布格式
#   1. NVlabs/SegFormer 官方 mit_bX.pth (键名与本地实现一致, 直载);
#   2. HF transformers 旧版布局 (nvidia/mit-b0 2021 年上传的 pytorch_model.bin:
#      segformer.patch_embeddings.{s}.* / segformer.block.{s}.{j}.* / attention.self.*);
#   3. HF transformers 新版布局 (segformer.stages.{s}.blocks.{j}.* / attention.q_proj.*);
# 加载后做 100% 键覆盖率硬校验, 任何不匹配立即报错, 杜绝静默半加载随机初始化。
# ---------------------------------------------------------------------------

def _map_hf_block_key(rest: str, out_stage: int, j: int) -> Tuple[str, Tuple[str, str] | None]:
    """把 HF block 内部子键映射为 NVlabs 子键; k/v 返回待融合标记 (which, 目标键)。"""
    pre = f"block{out_stage}.{j}."
    # ---- 注意力投影: 新版 q_proj/k_proj/v_proj/o_proj, 旧版 self.query/self.key/self.value/output.dense ----
    m = re.match(r"attention\.(?:self\.query|q_proj)\.(weight|bias)$", rest)
    if m:
        return pre + f"attn.q.{m[1]}", None
    m = re.match(r"attention\.(?:self\.key|k_proj)\.(weight|bias)$", rest)
    if m:
        return pre + f"attn.kv.{m[1]}", ("k", pre + f"attn.kv.{m[1]}")
    m = re.match(r"attention\.(?:self\.value|v_proj)\.(weight|bias)$", rest)
    if m:
        return pre + f"attn.kv.{m[1]}", ("v", pre + f"attn.kv.{m[1]}")
    m = re.match(r"attention\.(?:output\.dense|o_proj)\.(weight|bias)$", rest)
    if m:
        return pre + f"attn.proj.{m[1]}", None
    # ---- 空间缩减 SR: 新版 sequence_reduction.sequence_reduction / sequence_reduction.layer_norm,
    #      旧版 self.sr / self.layer_norm (仅 sr_ratio>1 的阶段存在) ----
    m = re.match(r"attention\.(?:self\.sr|sequence_reduction\.sequence_reduction)\.(weight|bias)$", rest)
    if m:
        return pre + f"attn.sr.{m[1]}", None
    m = re.match(r"attention\.(?:self\.layer_norm|sequence_reduction\.layer_norm)\.(weight|bias)$", rest)
    if m:
        return pre + f"attn.norm.{m[1]}", None
    # ---- LayerNorm: 新版 layernorm_before/after, 旧版 layer_norm_1/2 ----
    m = re.match(r"(?:layernorm_before|layer_norm_1)\.(weight|bias)$", rest)
    if m:
        return pre + f"norm1.{m[1]}", None
    m = re.match(r"(?:layernorm_after|layer_norm_2)\.(weight|bias)$", rest)
    if m:
        return pre + f"norm2.{m[1]}", None
    # ---- Mix-FFN: 新版 fc1/fc2, 旧版 dense1/dense2; dwconv 两代同名 ----
    m = re.match(r"mlp\.(?:fc1|dense1)\.(weight|bias)$", rest)
    if m:
        return pre + f"mlp.fc1.{m[1]}", None
    m = re.match(r"mlp\.(?:fc2|dense2)\.(weight|bias)$", rest)
    if m:
        return pre + f"mlp.fc2.{m[1]}", None
    m = re.match(r"mlp\.dwconv\.dwconv\.(weight|bias)$", rest)
    if m:
        return pre + f"mlp.dwconv.dwconv.{m[1]}", None
    return "", None  # 无法识别 (dropout 等无参模块)


def _convert_hf_segformer(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """HF transformers Segformer state_dict (新旧两代布局) -> NVlabs MiT 键名。

    新版 (transformers 5.x): segformer.stages.{s}.patch_embeddings.* /
        segformer.stages.{s}.blocks.{j}.* / segformer.stages.{s}.layer_norm.*
    旧版 (transformers 4.x, 即 nvidia/mit-bX 实际上传的文件):
        segformer.encoder.patch_embeddings.{s}.* / segformer.encoder.block.{s}.{j}.* /
        segformer.encoder.layer_norm.{s}.*
    """
    out: Dict[str, torch.Tensor] = {}
    kv_parts: Dict[str, Dict[str, torch.Tensor]] = {}

    def emit_block(s: int, j: int, rest: str, v: torch.Tensor) -> None:
        mapped, kv_tag = _map_hf_block_key(rest, s + 1, j)
        if not mapped:
            return
        if kv_tag is None:
            out[mapped] = v
        else:
            which, target = kv_tag
            kv_parts.setdefault(target, {})[which] = v

    for k, v in sd.items():
        k0 = k
        for p in ("segformer.", "encoder."):  # 剥离可选的容器前缀
            if k0.startswith(p):
                k0 = k0[len(p):]
        if k0.startswith(("classifier.", "decode_head.", "head.")):
            continue

        # ---- blocks: 新版 stages.{s}.blocks.{j}.rest / 旧版 block.{s}.{j}.rest ----
        m = re.match(r"stages\.(\d+)\.blocks\.(\d+)\.(.+)$", k0)
        if m:
            emit_block(int(m[1]), int(m[2]), m[3], v)
            continue
        m = re.match(r"block\.(\d+)\.(\d+)\.(.+)$", k0)
        if m:
            emit_block(int(m[1]), int(m[2]), m[3], v)
            continue

        # ---- patch embedding: 新版 stages.{s}.patch_embeddings.*, 旧版 patch_embeddings.{s}.* ----
        m = re.match(r"stages\.(\d+)\.patch_embeddings\.(proj|layer_norm)\.(weight|bias)$", k0)
        if m:
            out[f"patch_embed{int(m[1]) + 1}.{'proj' if m[2] == 'proj' else 'norm'}.{m[3]}"] = v
            continue
        m = re.match(r"patch_embeddings\.(\d+)\.(proj|layer_norm)\.(weight|bias)$", k0)
        if m:
            out[f"patch_embed{int(m[1]) + 1}.{'proj' if m[2] == 'proj' else 'norm'}.{m[3]}"] = v
            continue

        # ---- 每阶段末尾 LayerNorm: 新版 stages.{s}.layer_norm.*, 旧版 layer_norm.{s}.* ----
        m = re.match(r"stages\.(\d+)\.layer_norm\.(weight|bias)$", k0)
        if m:
            out[f"norm{int(m[1]) + 1}.{m[2]}"] = v
            continue
        m = re.match(r"layer_norm\.(\d+)\.(weight|bias)$", k0)
        if m:
            out[f"norm{int(m[1]) + 1}.{m[2]}"] = v
            continue

    # 融合 k/v -> kv (NVlabs: kv = Linear(dim, 2*dim), 前半 k 后半 v)
    for target, parts in kv_parts.items():
        if "k" in parts and "v" in parts:
            out[target] = torch.cat([parts["k"], parts["v"]], dim=0)
        else:
            raise RuntimeError(f"HF 权重缺少成对的 k/v 投影: {target}")
    return out


def _load_state_any(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file  # timm 依赖自带

        return load_file(str(path))
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def _coverage(encoder: MiTEncoder, state: Dict[str, torch.Tensor]) -> Tuple[int, List[str]]:
    enc_sd = encoder.state_dict()
    missing = [
        k for k, t in enc_sd.items()
        if k not in state or state[k].shape != t.shape
    ]
    return len(missing), missing


def load_mit_pretrained(encoder: MiTEncoder, path: str | Path) -> None:
    """加载 ImageNet 预训练 MiT 权重 (NVlabs 原版或 HF nvidia/mit-bX, 自动识别转换)。

    覆盖率必须 100% (编码器所有键), 否则 RuntimeError —— 防止形状不匹配的权重
    被 strict=False 静默丢弃, 造成"以为加载了预训练实际随机初始化"的隐患。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MiT 预训练权重不存在: {path}")

    state = _load_state_any(path)

    # 1) NVlabs 原版: 键名直接匹配
    n_missing, missing = _coverage(encoder, state)
    if n_missing == 0:
        encoder.load_state_dict(state, strict=False)
        print(f"[MiT] 已加载 NVlabs 原版 ImageNet 权重: {path}")
        return

    # 2) HF transformers 布局 (nvidia/mit-b0 等): 转换后再校验
    converted = _convert_hf_segformer(state)
    n_missing, missing = _coverage(encoder, converted)
    if n_missing == 0:
        encoder.load_state_dict(converted, strict=False)
        print(f"[MiT] 已加载 HF SegFormer 权重并完成键名转换: {path}")
        return

    total = len(encoder.state_dict())
    raise RuntimeError(
        f"MiT 预训练权重加载失败: {path}\n"
        f"  编码器共 {total} 个参数键, 缺失/不匹配 {n_missing} 个 "
        f"(覆盖率 {100 * (total - n_missing) / total:.1f}%)\n"
        f"  缺失示例: {missing[:6]}\n"
        f"  支持: NVlabs mit_bX.pth / HF nvidia/mit-bX (pytorch_model.bin 或 model.safetensors);\n"
        f"  请确认权重文件与所选骨干 ({getattr(encoder, 'embed_dims', None)}) 尺寸一致。"
    )
