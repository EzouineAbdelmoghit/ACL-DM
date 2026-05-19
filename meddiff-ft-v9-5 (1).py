# Generated from: meddiff-ft-v9-5 (1).ipynb
# Converted at: 2026-05-18T18:33:28.695Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # MedDiff-FT + Dynamic Frequency Balance — 2× T4 — **v7 (Comprehensive Evaluation)**
# 
# This release **keeps everything from v6** (memory-fixed training, DFB module, DDP + 8-bit AdamW)
# and **upgrades the evaluation suite** from a single-backbone DINOv3 validator to a full medical-image-aware
# evaluator covering **8 metric families** with bootstrap confidence intervals — designed specifically for
# the **30-pair low-data regime** you are working in.
# 
# ### What's new in v7
# 
# | Family | What it answers | Why it matters at N = 30 |
# |---|---|---|
# | 1. **Distributional fidelity** — FID, KID, FDD, FCD (+ optional RadDINO) | "Does the global distribution of synthetic images match real?" | Multi-encoder cross-validation removes single-backbone bias. **KID > FID** for small N. |
# | 2. **Region-conditioned paired metrics** — SSIM, MS-SSIM, PSNR, MAE, LPIPS computed on (full / background-only / lesion-only) | "Did the inpainter edit the right region, and is what it painted realistic?" | The mask is anatomically meaningful — global metrics dilute the signal from the 5–30% of pixels that actually changed. |
# | 3. **Mask fidelity & leakage** — outside/inside MAE, leakage ratio, edge gradient | "Did the model leave the background untouched and blend the lesion smoothly?" | Standard inpainting failure mode. |
# | 4. **Texture & frequency** — radial power spectrum, GLCM, wavelet sub-band energy | "Is the painted lesion texture realistic? Is DFB actually helping high-freq synthesis?" | This **directly probes what your DFB module does**. |
# | 5. **Color & dermoscopy** — RGB/HSV histogram intersection, Wasserstein distance, sat/hue stats | "Is the color realism right for skin lesions?" | Dermoscopy = color-driven diagnosis. |
# | 6. **Diversity / mode collapse** — Vendi score, mean pairwise cosine distance, P/R | "Is the model producing diverse outputs or copying its tiny training set?" | **The #1 risk at N=30.** |
# | 7. **Memorization / copy detection** — top-k cos sim to TRAINING set in DINOv3+CLIP space + SSIM-based duplicate flagging | "Is the model regurgitating training images?" | At N=30 the model can trivially memorize — must check before publishing. |
# | 8. **Downstream segmentation utility** — train tiny U-Net on real pairs → segment generated → compare to guide masks | "Are generated lesions anatomically *useful*?" | This is the metric reviewers ultimately care about for medical synthesis. |
# 
# All scalar metrics ship **95% bootstrap CIs (1000 resamples)** — point estimates at N=25–200 are misleading.
# 
# > Set the Kaggle accelerator to **GPU T4 ×2** before running.
# 


# 1. Clone repo and install dependencies
%cd /kaggle/working/
!rm -rf MedDiff-FT
!git clone https://github.com/JianhaoXie1/MedDiff-FT.git
%cd MedDiff-FT

# Core diffusion stack + comprehensive evaluation deps
!pip install -q \
    "diffusers" "transformers>=4.56.0" "accelerate" \
    "huggingface_hub" "datasets" \
    "Pillow" "scikit-image" "tqdm" "opencv-python" "Jinja2" "ftfy" "wandb" \
    "bitsandbytes" "safetensors" \
    "scipy" "lpips" "torchmetrics"


# 2. Drop in your optimized train.py / infer.py
import os, shutil
SRC_DIR = "/kaggle/input/datasets/abdelmoghitezouine11/meddiff-ft-v2"
DST = "/kaggle/working/MedDiff-FT/main"
for fname in ("train.py", "infer.py"):
    src = os.path.join(SRC_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DST, fname))
        print(f"✅ replaced {fname}")
    else:
        print(f"⚠️  {src} not found — keeping upstream {fname}")

# 3. Environment knobs and writable HF cache
import os, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["HF_HOME"]              = "/kaggle/working/.cache/hf"
os.environ["TRANSFORMERS_CACHE"]  = "/kaggle/working/.cache/hf/transformers"
os.environ["HF_HUB_CACHE"]        = "/kaggle/working/.cache/hf/hub"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
print("CUDA :", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    free, tot = torch.cuda.mem_get_info(i)
    print(f"  {i}: {torch.cuda.get_device_name(i)}  free={free/1e9:.1f}/{tot/1e9:.1f} GB")

# ## 4. Write the v3 modules (DFB + train + infer)


%%writefile /kaggle/working/MedDiff-FT/main/meddiff_dfb.py
"""
meddiff_dfb.py  (v5 — Correctness + VRAM-Optimized)
=====================================================

Dynamic Frequency Balance (DFB) module + frequency-domain losses.

Fixes applied vs v4:
  FIX 1  _DFBHookWrapper.forward — skip-connection patching now always
          replaces the last residual unconditionally (shape-matched), not
          via identity check (`is hidden`) that almost never fired. This
          ensures the decoder receives frequency-balanced skip tensors.

  FIX 2  HaarDWT/HaarIWT — kernel tensor cached by (channels, dtype) to
          avoid re-allocating a [4c,1,2,2] tensor on every forward pass.
          The cache lives on the module and is cleared on device moves.

  FIX 3  FrequencyLoss — mask is applied to pred and target *before* the
          FFT so the amplitude spectrum reflects only lesion-region content.
          DWT subbands are also masked consistently.

  FIX 4  attach_dfb_to_unet — channel detection now inspects ResNet blocks
          explicitly (resnets[-1].conv2.out_channels) with a safe fallback
          to avoid latching onto attention-projection Conv2d layers.

  FIX 5  DFBBlock gradient checkpointing — DDP find_unused_parameters
          incompatibility avoided by wrapping the checkpointed call in a
          lambda that returns a tuple (required by use_reentrant=False
          with DDP). Also adds _gradient_checkpointing_enable_disable
          protocol so accelerate can manage it automatically.

Retained from v4:
  * Module-level safetensors import.
  * HaarDWT/HaarIWT kernels as persistent=False registered buffers.
  * _MapDiffAttn per-forward lambda computation for gradient correctness.
  * Non-mutating _split_state_dict / save_unet_with_dfb.
  * vram_cleanup helper.
"""
from __future__ import annotations

import gc
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import save_file as _safetensors_save_file
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


# ---------------------------------------------------------------------------
# Haar DWT / IWT
# ---------------------------------------------------------------------------

def _haar_kernels(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    s = 0.5
    ll = torch.tensor([[ s,  s], [ s,  s]], dtype=dtype)
    lh = torch.tensor([[ s,  s], [-s, -s]], dtype=dtype)
    hl = torch.tensor([[ s, -s], [ s, -s]], dtype=dtype)
    hh = torch.tensor([[ s, -s], [-s,  s]], dtype=dtype)
    return torch.stack([ll, lh, hl, hh], dim=0)          # [4, 2, 2]


class HaarDWT(nn.Module):
    """Haar DWT with per-(channels, dtype) kernel cache to avoid repeated
    .repeat() allocations inside the training step (FIX 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_base_kernel", _haar_kernels(), persistent=False)
        # Cache: key = (channels, dtype_str) → tensor on same device
        self._kernel_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def _get_kernel(self, c: int, dtype: torch.dtype) -> torch.Tensor:
        key = (c, str(dtype))
        if key not in self._kernel_cache:
            k = (self._base_kernel
                 .to(dtype=dtype, device=self._base_kernel.device)
                 .view(4, 1, 2, 2)
                 .repeat(c, 1, 1, 1))
            self._kernel_cache[key] = k
        # Ensure device matches in case of device moves
        cached = self._kernel_cache[key]
        if cached.device != self._base_kernel.device:
            cached = cached.to(self._base_kernel.device)
            self._kernel_cache[key] = cached
        return cached.to(dtype=dtype)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        ph = h % 2; pw = w % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            _, _, h, w = x.shape
        k = self._get_kernel(c, x.dtype)
        out = F.conv2d(x, k, stride=2, groups=c)          # [b, 4c, h/2, w/2]
        out = out.view(b, c, 4, h // 2, w // 2)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]


class HaarIWT(nn.Module):
    """Haar IWT with the same (channels, dtype) kernel cache as DWT."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_base_kernel", _haar_kernels(), persistent=False)
        self._kernel_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def _get_kernel(self, c: int, dtype: torch.dtype) -> torch.Tensor:
        key = (c, str(dtype))
        if key not in self._kernel_cache:
            k = (self._base_kernel
                 .to(dtype=dtype, device=self._base_kernel.device)
                 .view(4, 1, 2, 2)
                 .repeat(c, 1, 1, 1))
            self._kernel_cache[key] = k
        cached = self._kernel_cache[key]
        if cached.device != self._base_kernel.device:
            cached = cached.to(self._base_kernel.device)
            self._kernel_cache[key] = cached
        return cached.to(dtype=dtype)

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = ll.shape
        k = self._get_kernel(c, ll.dtype)
        coeffs = torch.stack([ll, lh, hl, hh], dim=2).view(b, 4 * c, h, w)
        return F.conv_transpose2d(coeffs, k, stride=2, groups=c)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class _MapMHSA(nn.Module):
    """Multi-head self-attention on 2D feature maps."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        for h in (heads, 4, 2, 1):
            if dim % h == 0:
                heads = h
                break
        self.h = heads
        self.dh = dim // heads
        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self._sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        t = x.flatten(2).transpose(1, 2)                  # [b, n, c]
        qkv = (self.qkv(t)
               .reshape(b, n, 3, self.h, self.dh)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self._sdpa:
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            o = (q @ k.transpose(-2, -1) / math.sqrt(self.dh)).softmax(-1) @ v
        return (self.proj(o.transpose(1, 2).reshape(b, n, c))
                .transpose(1, 2).reshape(b, c, h, w))


class _MapDiffAttn(nn.Module):
    """Differential self-attention from the DIFF Transformer paper."""

    def __init__(self, dim: int, heads: int = 4,
                 lambda_init: float = 0.5) -> None:
        super().__init__()
        for h in (heads, 4, 2, 1):
            if dim % h == 0:
                heads = h
                break
        self.h = heads
        self.dh = dim // heads
        self.q1   = nn.Linear(dim, dim, bias=False)
        self.k1   = nn.Linear(dim, dim, bias=False)
        self.q2   = nn.Linear(dim, dim, bias=False)
        self.k2   = nn.Linear(dim, dim, bias=False)
        self.v    = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.lam_q1 = nn.Parameter(torch.zeros(self.dh))
        self.lam_k1 = nn.Parameter(torch.zeros(self.dh))
        self.lam_q2 = nn.Parameter(torch.zeros(self.dh))
        self.lam_k2 = nn.Parameter(torch.zeros(self.dh))
        self.lambda_init = lambda_init

    def _compute_lambda(self) -> torch.Tensor:
        return (torch.exp((self.lam_q1 * self.lam_k1).sum())
                - torch.exp((self.lam_q2 * self.lam_k2).sum())
                + self.lambda_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, H, W = x.shape
        n = H * W
        t = x.flatten(2).transpose(1, 2)

        def split(u: torch.Tensor) -> torch.Tensor:
            return u.reshape(b, n, self.h, self.dh).transpose(1, 2)

        q1 = split(self.q1(t)); k1 = split(self.k1(t))
        q2 = split(self.q2(t)); k2 = split(self.k2(t))
        v  = split(self.v(t))
        scale = 1.0 / math.sqrt(self.dh)
        a1 = (q1 @ k1.transpose(-2, -1) * scale).softmax(-1)
        a2 = (q2 @ k2.transpose(-2, -1) * scale).softmax(-1)
        o  = (a1 - self._compute_lambda() * a2) @ v
        return (self.proj(o.transpose(1, 2).reshape(b, n, c))
                .transpose(1, 2).reshape(b, c, H, W))


# ---------------------------------------------------------------------------
# DFB block
# ---------------------------------------------------------------------------

class DFBBlock(nn.Module):
    """
    Dynamic Frequency Balance block.

    Splits features into Haar wavelet subbands, applies:
      - MHSA on the low-frequency LL subband
      - Differential cross-attention on stacked high-frequency subbands
    then reconstructs via IWT and adds to the input via a learned scalar γ.

    γ = 0 at init → exact identity, allowing safe insertion into pretrained
    UNet without disrupting the starting distribution.
    """

    def __init__(self, channels: int, heads: int = 4,
                 inner_dim_factor: float = 1.0) -> None:
        super().__init__()
        hi_dim = max(int(round(channels * inner_dim_factor)), heads * 8)
        if hi_dim % heads != 0:
            hi_dim = ((hi_dim + heads - 1) // heads) * heads

        self.dwt       = HaarDWT()
        self.iwt       = HaarIWT()
        self.low_attn  = _MapMHSA(channels, heads=heads)
        self.high_in   = nn.Conv2d(3 * channels, hi_dim, 1, bias=False)
        self.high_attn = _MapDiffAttn(hi_dim, heads=heads)
        self.high_out  = nn.Conv2d(hi_dim, 3 * channels, 1, bias=False)
        self.gamma     = nn.Parameter(torch.full((1,), 1e-4))

        self._gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Gradient checkpointing protocol (accelerate calls these)
    # ------------------------------------------------------------------
    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def _dfb_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, H, W = x.shape
        ph = H % 2; pw = W % 2
        x_p = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x

        ll, lh, hl, hh = self.dwt(x_p)
        ll = ll + self.low_attn(ll)

        Hcat = torch.cat([lh, hl, hh], dim=1)
        Hin  = self.high_in(Hcat)
        Hin  = Hin + self.high_attn(Hin)
        Hout = self.high_out(Hin)
        lh2, hl2, hh2 = torch.chunk(Hout, 3, dim=1)

        z = self.iwt(ll, lh2, hl2, hh2)
        if ph or pw:
            z = z[..., :H, :W]
        return x + self.gamma * z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gradient_checkpointing and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._dfb_forward, x, use_reentrant=False)
        return self._dfb_forward(x)


# ---------------------------------------------------------------------------
# UNet wrapper
# ---------------------------------------------------------------------------

_PROXY_ATTRS = (
    "has_cross_attention",
    "_supports_gradient_checkpointing",
    "gradient_checkpointing",
)


class _DFBHookWrapper(nn.Module):
    """Wraps a UNet down_block, applying DFB to its hidden output and
    updating skip connections so the decoder receives refined tensors.

    FIX 1: skip-connection patching is now unconditional — we always replace
    the last residual in the skip list with the DFB-refined hidden state
    (after a shape-compatibility guard), instead of using `is hidden` identity
    comparison which almost never matched in practice.
    """

    def __init__(self, inner: nn.Module, dfb: DFBBlock) -> None:
        super().__init__()
        self.inner = inner
        self.dfb   = dfb
        for attr in _PROXY_ATTRS:
            if hasattr(inner, attr):
                try:
                    object.__setattr__(self, attr, getattr(inner, attr))
                except Exception:
                    pass

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)

        # SD UNet down_blocks return (hidden_states, (res_0, res_1, ...))
        if (isinstance(out, tuple) and len(out) == 2
                and isinstance(out[1], (list, tuple))):
            hidden, res = out
            new_hidden = self.dfb(hidden)
            new_res    = list(res)

            # FIX 1: Always patch the last skip tensor if shapes are
            # compatible, rather than relying on object identity.
            if len(new_res) > 0:
                last = new_res[-1]
                if (isinstance(last, torch.Tensor)
                        and last.shape == new_hidden.shape):
                    new_res[-1] = new_hidden

            return new_hidden, tuple(new_res)

        # Single-tensor output path (some UNet variants)
        if isinstance(out, torch.Tensor):
            return self.dfb(out)

        return out


# ---------------------------------------------------------------------------
# Attach / restore helpers
# ---------------------------------------------------------------------------

def _get_block_out_channels(blk: nn.Module) -> Optional[int]:
    """
    Reliably determine the spatial feature channel count of a UNet down_block.

    FIX 4: Preferentially reads from ResNet blocks (resnets[-1].conv2) whose
    output IS the spatial hidden dimension, rather than iterating all Conv2d
    modules which may land on attention projection layers with different dims.
    Falls back to last Conv2d if no ResNet structure is found.
    """
    # Preferred: diffusers ResNet block structure
    if hasattr(blk, "resnets") and len(blk.resnets) > 0:
        last_res = blk.resnets[-1]
        # conv2 is the channel-preserving output conv of each ResNet block
        if hasattr(last_res, "conv2") and isinstance(last_res.conv2, nn.Conv2d):
            return last_res.conv2.out_channels
        # conv_shortcut / conv1 as fallback within resnets
        for attr in ("conv1", "conv_shortcut"):
            m = getattr(last_res, attr, None)
            if isinstance(m, nn.Conv2d):
                return m.out_channels

    # Fallback: walk Conv2d modules but skip 1×1 projections and attention
    # projections (those have kernel_size=1 and often mismatched dims)
    spatial_convs = [
        m for m in blk.modules()
        if isinstance(m, nn.Conv2d)
        and m.kernel_size not in ((1, 1), (1,))
        and m.groups == 1          # skip depthwise
    ]
    if spatial_convs:
        return spatial_convs[-1].out_channels

    # Last resort: any Conv2d
    all_convs = [m for m in blk.modules() if isinstance(m, nn.Conv2d)]
    return all_convs[-1].out_channels if all_convs else None


def attach_dfb_to_unet(
    unet,
    heads: int = 4,
    inner_dim_factor: float = 1.0,
) -> List[DFBBlock]:
    """Wrap each unet.down_blocks[i] with a _DFBHookWrapper + DFBBlock."""
    new_blocks: List[DFBBlock] = []
    for i, blk in enumerate(unet.down_blocks):
        if isinstance(blk, _DFBHookWrapper):
            new_blocks.append(blk.dfb)
            continue
        out_c = _get_block_out_channels(blk)
        if out_c is None:
            continue
        dfb = DFBBlock(out_c, heads=heads, inner_dim_factor=inner_dim_factor)
        unet.down_blocks[i] = _DFBHookWrapper(blk, dfb)
        new_blocks.append(dfb)
    return new_blocks


def restore_dfb_in_unet(
    unet,
    dfb_state: Dict[str, torch.Tensor],
    heads: int = 4,
    inner_dim_factor: float = 1.0,
) -> List[DFBBlock]:
    """Re-attach DFB wrappers and load saved DFB weights. Inference only."""
    new_blocks = attach_dfb_to_unet(unet, heads=heads,
                                    inner_dim_factor=inner_dim_factor)
    per_block: Dict[int, Dict[str, torch.Tensor]] = {}
    for k, v in dfb_state.items():
        parts = k.split(".")
        if len(parts) < 3 or parts[0] != "down_blocks":
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        rest = ".".join(parts[2:])
        if rest.startswith("dfb."):
            rest = rest[len("dfb."):]
        per_block.setdefault(idx, {})[rest] = v

    for i, blk in enumerate(unet.down_blocks):
        if isinstance(blk, _DFBHookWrapper) and i in per_block:
            missing, unexpected = blk.dfb.load_state_dict(
                per_block[i], strict=False)
            if missing or unexpected:
                print(f"[DFB] block {i}: missing={len(missing)} "
                      f"unexpected={len(unexpected)}")
    return new_blocks


# ---------------------------------------------------------------------------
# State-dict split (non-mutating)
# ---------------------------------------------------------------------------

def _split_state_dict(
    full_sd: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a wrapped UNet state_dict into bare UNet + DFB sidecar."""
    bare_sd: Dict[str, torch.Tensor] = {}
    dfb_sd:  Dict[str, torch.Tensor] = {}
    for k, v in full_sd.items():
        v_cpu = v.detach().cpu().contiguous()
        if k.startswith("down_blocks.") and ".dfb." in k:
            dfb_sd[k] = v_cpu
        elif k.startswith("down_blocks.") and ".inner." in k:
            bare_sd[k.replace(".inner.", ".", 1)] = v_cpu
        else:
            bare_sd[k] = v_cpu
    return bare_sd, dfb_sd


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_unet_with_dfb(
    unet,
    output_dir: str,
    heads: int = 4,
    inner_dim_factor: float = 1.0,
) -> None:
    """Write vanilla-format UNet + DFB sidecar. Never mutates live UNet."""
    unet_dir = os.path.join(output_dir, "unet")
    os.makedirs(unet_dir, exist_ok=True)
    unet.save_config(unet_dir)

    full_sd = unet.state_dict()
    bare_sd, dfb_sd = _split_state_dict(full_sd)

    if _HAS_SAFETENSORS:
        _safetensors_save_file(
            bare_sd,
            os.path.join(unet_dir, "diffusion_pytorch_model.safetensors"))
        if dfb_sd:
            _safetensors_save_file(
                dfb_sd,
                os.path.join(unet_dir, "dfb_weights.safetensors"))
            with open(os.path.join(unet_dir, "dfb_config.json"), "w") as fh:
                json.dump({"heads": int(heads),
                           "inner_dim_factor": float(inner_dim_factor)},
                          fh, indent=2)
    else:
        torch.save(bare_sd,
                   os.path.join(unet_dir, "diffusion_pytorch_model.bin"))
        if dfb_sd:
            torch.save(dfb_sd,
                       os.path.join(unet_dir, "dfb_weights.bin"))

    del full_sd, bare_sd, dfb_sd
    gc.collect()


def load_dfb_state(unet_dir: str) -> Dict[str, torch.Tensor]:
    safe = os.path.join(unet_dir, "dfb_weights.safetensors")
    binp = os.path.join(unet_dir, "dfb_weights.bin")
    if os.path.exists(safe):
        from safetensors.torch import load_file
        return load_file(safe)
    if os.path.exists(binp):
        return torch.load(binp, map_location="cpu")
    return {}


def load_dfb_config(unet_dir: str) -> Dict:
    p = os.path.join(unet_dir, "dfb_config.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {"heads": 4, "inner_dim_factor": 1.0}


# ---------------------------------------------------------------------------
# VRAM hygiene
# ---------------------------------------------------------------------------

def vram_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Frequency-domain auxiliary loss
# ---------------------------------------------------------------------------

class FrequencyLoss(nn.Module):
    """
    Frequency-domain reconstruction loss combining FFT amplitude + DWT subbands.

    FIX 3: The spatial mask is applied to pred and target BEFORE the FFT/DWT
    so the amplitude spectrum reflects only the masked (lesion) region, not
    surrounding healthy tissue. This directly serves the anatomical-accuracy
    goal of DFB: frequency penalties are localised to where anatomy matters.

    Args:
        w_fft: weight for FFT amplitude loss.
        w_ll:  weight for low-frequency LL subband loss.
        w_hh:  weight for high-frequency subband loss (LH+HL+HH mean).
    """

    def __init__(
        self,
        w_fft: float = 1.0,
        w_ll:  float = 1.0,
        w_hh:  float = 0.5,
    ) -> None:
        super().__init__()
        self.dwt   = HaarDWT()
        self.w_fft = w_fft
        self.w_ll  = w_ll
        self.w_hh  = w_hh

    def forward(
        self,
        pred:   torch.Tensor,        # [B, C, H, W]  float32
        target: torch.Tensor,        # [B, C, H, W]  float32
        mask:   Optional[torch.Tensor] = None,  # [B, 1, H', W']  binary
    ) -> torch.Tensor:
        # FIX 3: Apply mask BEFORE FFT/DWT so spectra are lesion-local.
        if mask is not None:
            if mask.shape[-2:] != pred.shape[-2:]:
                m = F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest")
            else:
                m = mask
            pred_m   = pred   * m
            target_m = target * m
        else:
            pred_m   = pred
            target_m = target

        # FFT amplitude loss (global frequency distribution of masked region)
        Fp = torch.fft.rfft2(pred_m,   norm="ortho")
        Ft = torch.fft.rfft2(target_m, norm="ortho")
        l_fft = (Fp.abs() - Ft.abs()).abs().mean()

        # DWT subband losses (directional frequency content of masked region)
        ll_p, lh_p, hl_p, hh_p = self.dwt(pred_m)
        ll_t, lh_t, hl_t, hh_t = self.dwt(target_m)
        l_ll = (ll_p - ll_t).abs().mean()
        l_hh = ((lh_p - lh_t).abs().mean()
              + (hl_p - hl_t).abs().mean()
              + (hh_p - hh_t).abs().mean()) / 3.0

        return self.w_fft * l_fft + self.w_ll * l_ll + self.w_hh * l_hh

%%writefile /kaggle/working/MedDiff-FT/main/train_dfb.py
"""
train_dfb.py  (v7 — Differential-Diffusion soft-mask training)
==============================================================

Changes from v6:
* PATCH E: Per-example soft mask via distance-transform
           (see ``softmask_utils.binary_to_soft_mask``).
           Stored *alongside* the original binary mask so we get:
             - a SOFT mask (continuous in [0, 1])  -> pixel-wise weight
               for the base reconstruction MSE.
             - a HARD mask (binary {0, 1})         -> input to the
               ``FrequencyLoss`` (must stay binary to avoid spectral
               windowing artifacts) AND the UNet mask channel
               (so the model still sees the same input distribution it
               was pretrained on).
* PATCH F: Cache extension — when ``--cache_latents`` is on we cache
           both the binary and soft latent-resolution masks. Extra
           VRAM/RAM per sample: 64*64*4 bytes = 16 KB. Well under the
           "few KB per image extra in the active step" hard constraint
           (only one batch is on GPU at a time).
* PATCH G: Pixel-wise weighted base loss
              base_loss = (soft * (noise_pred - target)**2).mean()
           replaces the old uniform ``F.mse_loss(noise_pred, target)``.
           The frequency loss still receives the **binary** mask.

All other behaviour (DDP wrap, 8-bit AdamW, gradient checkpointing,
DFB block plumbing, validation, checkpointing) is unchanged.
No new trainable parameters are introduced, so
``find_unused_parameters=False`` remains safe.
"""
import os
import sys
import math
import copy
import itertools
import gc
import logging
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (AutoencoderKL, DDPMScheduler, DDIMScheduler,
                       StableDiffusionInpaintPipeline, UNet2DConditionModel)
from diffusers.optimization import get_scheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as _train_mod                                          # noqa: E402
from train import (DreamBoothDataset, collate_fn,
                   HAS_BNB, is_xformers_available)
from meddiff_dfb import (
    attach_dfb_to_unet, DFBBlock, FrequencyLoss,
    save_unet_with_dfb, vram_cleanup,
    _DFBHookWrapper, restore_dfb_in_unet,
    load_dfb_state, load_dfb_config,
)

# PATCH E — soft-mask utilities (local, self-contained, zero trainable params)
from softmask_utils import (
    binary_to_soft_mask, downsample_mask_to_latent,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Local DreamBoothDataset wrapper — now also derives a soft mask from the
# binary annotation. The base DreamBoothDataset is left untouched.
# ---------------------------------------------------------------------------
class _CacheKeyDataset(torch.utils.data.Dataset):
    """Wrapper that adds soft/hard masks for Differential-Diffusion training.

    The base dataset (DreamBoothDataset) returns a dict with *file paths*:
        {"image": "...", "mask_1": "...", "prompt": "..."}
    We load the mask from disk, create soft + binary masks, and attach them
    as PIL Images (later collated into tensors).
    """

    def __init__(self, base_dataset, inner_blur: int = 5, outer_blur: int = 5):
        self.base = base_dataset
        self.inner_blur = inner_blur
        self.outer_blur = outer_blur
        self.resolution = base_dataset.resolution if hasattr(base_dataset, 'resolution') else 512

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        ex = self.base[idx]
        if "cache_key" not in ex:
            ex["cache_key"] = idx
    
        # ---- Find the mask among known key names ----
        mask_value = None
        for possible_key in ["instance_masks", "mask_1", "mask", "instance_mask"]:
            if possible_key in ex:
                mask_value = ex[possible_key]
                break
    
        if mask_value is None:
            # Print available keys for debugging (once)
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[DEBUG] Example keys: {list(ex.keys())}")
            raise KeyError(
                "Cannot find a mask key in the dataset example. "
                f"Available keys: {list(ex.keys())}"
            )
    
        # ---- Handle different mask formats ----
        if isinstance(mask_value, (list, tuple)):
            # The dataset might return a list of masks – take the first
            mask_pil = mask_value[0]
        else:
            mask_pil = mask_value
    
        # Ensure we have a PIL Image (some datasets return tensors)
        if not isinstance(mask_pil, Image.Image):
            # Convert tensor or numpy array to PIL
            if hasattr(mask_pil, 'numpy'):
                arr = mask_pil.numpy()
            else:
                arr = np.asarray(mask_pil)
            if arr.ndim == 3:
                arr = arr[0]   # take first channel if multi-channel
            mask_pil = Image.fromarray(arr.astype(np.uint8), mode='L')
    
        # Resize to training resolution
        mask_pil = mask_pil.resize((self.resolution, self.resolution), Image.NEAREST)
    
        # ---- Create soft and hard masks ----
        soft_pil, hard_pil = binary_to_soft_mask(
            mask_pil,
            inner_blur=self.inner_blur,
            outer_blur=self.outer_blur,
        )
    
        ex["soft_mask"] = soft_pil
        ex["hard_mask"] = hard_pil
        return ex
        

def _collate_with_cache_keys(examples, tokenizer, resolution=512):
    from train import collate_fn as _base_collate
    batch = _base_collate(examples, tokenizer, resolution)

    # cache_keys
    if "cache_keys" not in batch:
        batch["cache_keys"] = [ex.get("cache_key", i)
                               for i, ex in enumerate(examples)]

    # Convert soft/hard PIL masks to tensors
    if all("soft_mask" in ex for ex in examples):
        soft_list = []
        for ex in examples:
            soft_arr = np.asarray(ex["soft_mask"], dtype=np.float32) / 255.0
            soft_list.append(torch.from_numpy(soft_arr).unsqueeze(0))  # [1, H, W]
        batch["soft_masks"] = torch.stack(soft_list).contiguous()      # [B, 1, H, W]

    if all("hard_mask" in ex for ex in examples):
        hard_list = []
        for ex in examples:
            hard_arr = (np.asarray(ex["hard_mask"]) >= 128).astype(np.float32)
            hard_list.append(torch.from_numpy(hard_arr).unsqueeze(0))
        batch["hard_masks"] = torch.stack(hard_list).contiguous()

    return batch


# ---------------------------------------------------------------------------
# ARGPARSE — DFB flags + base train args (+ optional soft-mask blur radii)
# ---------------------------------------------------------------------------
def parse_args_dfb():
    dfb_flags = {
        "--use_dfb": "store_true",
        "--no_dfb":  "store_true",
        "--disable_validation": "store_true",
        "--dfb_heads": "value",
        "--dfb_inner_dim_factor": "value",
        "--freq_loss_weight": "value",
        "--freq_loss_w_fft": "value",
        "--freq_loss_w_ll":  "value",
        "--freq_loss_w_hh":  "value",
        "--freq_loss_prob":  "value",
        "--cache_latents":   "store_true",
        "--validation_num_steps": "value",
        "--lr_num_cycles": "value",
        # PATCH E — soft-mask blur radii (optional; defaults match the spec).
        "--soft_inner_blur": "value",
        "--soft_outer_blur": "value",
    }
    saved = list(sys.argv)
    cleaned = [saved[0]]
    extras = {}
    i = 1
    while i < len(saved):
        a = saved[i]
        if a in dfb_flags:
            kind = dfb_flags[a]
            if kind == "store_true":
                extras[a] = True
                i += 1
            else:
                extras[a] = saved[i + 1]
                i += 2
        else:
            cleaned.append(saved[i])
            i += 1
    sys.argv = cleaned
    base = _train_mod.parse_args()
    sys.argv = saved

    base.use_dfb = bool(extras.get("--use_dfb", True))
    if "--no_dfb" in extras:
        base.use_dfb = False
    base.disable_validation = bool(extras.get("--disable_validation", False))
    base.dfb_heads            = int(extras.get("--dfb_heads", 4))
    base.dfb_inner_dim_factor = float(extras.get("--dfb_inner_dim_factor", 1.0))
    base.freq_loss_weight     = float(extras.get("--freq_loss_weight", 0.1))
    base.freq_loss_w_fft      = float(extras.get("--freq_loss_w_fft", 1.0))
    base.freq_loss_w_ll       = float(extras.get("--freq_loss_w_ll",  1.0))
    base.freq_loss_w_hh       = float(extras.get("--freq_loss_w_hh",  0.5))
    base.freq_loss_prob       = float(extras.get("--freq_loss_prob",  0.5))
    base.lr_num_cycles = int(extras.get("--lr_num_cycles", 1))
    base.cache_latents = bool(extras.get("--cache_latents", True))
    base.soft_inner_blur = int(extras.get("--soft_inner_blur", 5))
    base.soft_outer_blur = int(extras.get("--soft_outer_blur", 5))
    if not hasattr(base, "validation_num_steps"):
        base.validation_num_steps = 25
    return base


# ---------------------------------------------------------------------------
# VRAM logging utility (module-level, used outside main())
# ---------------------------------------------------------------------------
def vram_str():
    """Compact human-readable VRAM string for logging."""
    if not torch.cuda.is_available():
        return "n/a"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return (f"used={used/1e9:.2f}/{total/1e9:.2f}GB "
            f"(reserved={torch.cuda.memory_reserved()/1e9:.2f})")


# ---------------------------------------------------------------------------
# In-place validation (unchanged — uses binary mask, no soft blending here;
# RePaint + soft blending live in ``infer_dfb.py`` / ``validate_comprehensive``)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def in_place_validation(args, unet, vae, text_encoder, tokenizer,
                        weight_dtype, device, global_step,
                        sched_config, dfb_blocks=None):
    if args.validation_image is None or args.validation_mask is None:
        return

    save_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(save_dir, exist_ok=True)

    was_training_unet = unet.training
    te_was_training = False
    if text_encoder is not None:
        te_was_training = text_encoder.training

    unet.eval()
    if text_encoder is not None:
        text_encoder.eval()

    try:
        init = (Image.open(args.validation_image).convert("RGB")
                .resize((args.resolution, args.resolution), Image.LANCZOS))
        msk = (Image.open(args.validation_mask).convert("L")
               .resize((args.resolution, args.resolution), Image.NEAREST))

        sched = DDIMScheduler.from_config(sched_config)
        try:
            sched.set_timesteps(args.validation_num_steps, device=device)
        except TypeError:
            sched.set_timesteps(args.validation_num_steps)

        tok = tokenizer(args.validation_prompt, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)

        H = args.resolution
        latH = H // 8

        with torch.autocast(device_type="cuda", dtype=weight_dtype,
                            enabled=weight_dtype != torch.float32):
            txt = text_encoder(tok)[0] if text_encoder is not None else None
            z = torch.randn(1, 4, latH, latH, device=device, dtype=weight_dtype)
            img_t = (torch.from_numpy(np.array(init)).to(device).float()
                     / 127.5 - 1.0)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(weight_dtype)
            msk_t = (torch.from_numpy(np.array(msk)).to(device).float()
                     / 255.0)
            msk_t = (msk_t >= 0.5).float().unsqueeze(0).unsqueeze(0)
            masked = img_t * (msk_t < 0.5)
            mlat = vae.encode(masked).latent_dist.sample()
            mlat = mlat * vae.config.scaling_factor
            mlat = mlat.to(weight_dtype)
            msk_lat = F.interpolate(msk_t, size=(latH, latH),
                                    mode="nearest").to(weight_dtype)
            for t in sched.timesteps:
                lin = torch.cat([z, msk_lat, mlat], dim=1)
                eps = unet(lin, t, txt).sample
                z = sched.step(eps, t, z).prev_sample
            img = vae.decode(z / vae.config.scaling_factor).sample

        img = (img.clamp(-1, 1) + 1) / 2
        img = (img.float().permute(0, 2, 3, 1).cpu().numpy()[0]
               * 255).astype("uint8")
        Image.fromarray(img).save(
            os.path.join(save_dir, f"step-{global_step:06d}.png"))
    finally:
        if was_training_unet:
            unet.train()
        if te_was_training and text_encoder is not None:
            text_encoder.train()
        vram_cleanup()


# ---------------------------------------------------------------------------
# Unified cache builder — PATCH F: now also caches soft + hard latent masks.
# ---------------------------------------------------------------------------
def build_training_cache(args, train_dataloader, vae, text_encoder,
                         accelerator, weight_dtype, vae_scaling_factor):
    logger.info("Building unified training cache "
                "(latents + text embeddings + binary + soft masks)...")
    vae.eval()
    text_encoder.eval()

    latH = args.resolution // 8
    cache = {}
    with torch.inference_mode():
        for batch in tqdm(train_dataloader, desc="Caching",
                          disable=not accelerator.is_local_main_process):
            pixel_values = batch["pixel_values"].to(
                accelerator.device, dtype=weight_dtype, non_blocking=True)
            masked_images = batch["masked_images"].to(
                accelerator.device, dtype=weight_dtype, non_blocking=True)
            input_ids = batch["input_ids"].to(
                accelerator.device, non_blocking=True)

            # ------------------------------------------------------------------
            # Hard binary mask (the original one from the base collate). This
            # one feeds the FrequencyLoss AND the UNet input channel.
            # ------------------------------------------------------------------
            masks = batch["masks"]
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            elif masks.dim() == 5:
                masks = masks.squeeze(2)
            masks = masks.to(accelerator.device, non_blocking=True)

            # Prefer the explicit hard mask from the new collate if present.
            hard_mask_pix = batch.get("hard_masks", masks)
            soft_mask_pix = batch.get("soft_masks", None)
            hard_mask_pix = hard_mask_pix.to(accelerator.device, non_blocking=True)
            if soft_mask_pix is not None:
                soft_mask_pix = soft_mask_pix.to(accelerator.device, non_blocking=True)

            latents = (vae.encode(pixel_values).latent_dist.sample()
                       * vae_scaling_factor)
            masked_latents = (vae.encode(masked_images).latent_dist.sample()
                              * vae_scaling_factor)
            encoder_hidden_states = text_encoder(input_ids)[0]

            # Latent-resolution masks (nearest for binary, bilinear for soft).
            mask_down_hard = F.interpolate(
                hard_mask_pix.float(), size=(latH, latH), mode="nearest")
            if soft_mask_pix is not None:
                mask_down_soft = F.interpolate(
                    soft_mask_pix.float(), size=(latH, latH),
                    mode="bilinear", align_corners=False).clamp(0.0, 1.0)
            else:
                # Fallback: soft = hard if upstream didn't provide one.
                mask_down_soft = mask_down_hard.clone()

            for b_idx, ckey in enumerate(batch.get("cache_keys", [])):
                cache[ckey] = {
                    "latents":              latents[b_idx:b_idx+1].cpu(),
                    "masked_latents":       masked_latents[b_idx:b_idx+1].cpu(),
                    "encoder_hidden_states":
                        encoder_hidden_states[b_idx:b_idx+1].cpu(),
                    # NB: "mask" is kept under the SAME key as before so the
                    # rest of the code path (and the UNet 9-ch input) is
                    # backward compatible. It is the BINARY mask.
                    "mask":                 mask_down_hard[b_idx:b_idx+1].cpu(),
                    "soft_mask":            mask_down_soft[b_idx:b_idx+1].cpu(),
                }

            del pixel_values, masked_images, input_ids, masks
            del latents, masked_latents, encoder_hidden_states
            del mask_down_hard, mask_down_soft, hard_mask_pix
            if soft_mask_pix is not None:
                del soft_mask_pix

    return cache


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args_dfb()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:128,expandable_segments:True")

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision("high")

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=Path(args.output_dir, args.logging_dir),
    )

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=False,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        log_with="wandb" if args.report_to_wandb else None,
        kwargs_handlers=[ddp_kwargs],
    )
    if args.seed is not None:
        set_seed(args.seed)

    # =========================================================================
    # PATCH A — structured multi-handler logging
    # =========================================================================
    _rank  = accelerator.process_index
    _nproc = accelerator.num_processes
    _log_dir = Path(args.output_dir) / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)

    _log_fmt = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)-8s | rank{_rank} | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _fh = logging.FileHandler(_log_dir / f"train_rank{_rank}.log", mode="w")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_log_fmt)

    _eh = None
    if accelerator.is_main_process:
        _eh = logging.FileHandler(_log_dir / "train_events.log", mode="w")
        _eh.setLevel(logging.INFO)
        _eh.setFormatter(_log_fmt)

    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_log_fmt)

    _root = logging.getLogger()
    _root.setLevel(logging.DEBUG)
    for _old_h in list(_root.handlers):
        _root.removeHandler(_old_h)
    _root.addHandler(_fh)
    _root.addHandler(_ch)
    if _eh:
        _root.addHandler(_eh)

    logger.info(f"[LOGGING] rank{_rank} → {_log_dir}/train_rank{_rank}.log")
    logger.info(accelerator.state, main_process_only=False)

    # =========================================================================
    # PATCH B — VRAM snapshot + dtype audit helpers
    # =========================================================================
    def _vram(label: str = "") -> None:
        if not torch.cuda.is_available():
            return
        alloc  = torch.cuda.memory_allocated()  / 1024**3
        reserv = torch.cuda.memory_reserved()   / 1024**3
        total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        tag = f"[{label}]" if label else "[VRAM]"
        logger.info(f"{tag} rank{_rank}: "
                    f"allocated={alloc:.2f}GB  "
                    f"reserved={reserv:.2f}GB  "
                    f"total={total:.2f}GB")

    def _log_dtypes(model, name: str) -> None:
        dtypes: dict = {}
        examples: dict = {}
        for pname, p in model.named_parameters():
            key = str(p.dtype)
            dtypes[key] = dtypes.get(key, 0) + 1
            examples.setdefault(key, pname)
        for dtype, count in dtypes.items():
            logger.info(f"[DTYPE] {name}: {count} params as {dtype} "
                        f"(e.g. '{examples[dtype]}')")

    def _log_optimizer_dtypes(opt, label: str = "optimizer") -> None:
        for gi, group in enumerate(opt.param_groups):
            dtypes: dict = {}
            for p in group["params"]:
                key = str(p.dtype)
                dtypes[key] = dtypes.get(key, 0) + 1
            logger.info(f"[DTYPE] {label} group[{gi}]: {dict(dtypes)}")

    _vram("startup")

    # ---- models -------------------------------------------------------------
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.requires_grad_(args.train_text_encoder)

    vae_load_dtype = (
        torch.float16 if args.mixed_precision == "fp16" else
        torch.bfloat16 if args.mixed_precision == "bf16" else
        torch.float32
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=vae_load_dtype,
    ).requires_grad_(False)
    _vram("after-vae-load")

    if not args.train_text_encoder:
        text_encoder = text_encoder.to(dtype=vae_load_dtype)
    _vram("after-text-encoder-load")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    _vram("after-unet-load")
    _log_dtypes(unet, "unet-raw")

    dfb_blocks = []
    if args.use_dfb:
        dfb_blocks = attach_dfb_to_unet(
            unet, heads=args.dfb_heads,
            inner_dim_factor=args.dfb_inner_dim_factor)
        n = sum(p.numel() for b in dfb_blocks for p in b.parameters())
        logger.info(f"[DFB] attached {len(dfb_blocks)} blocks, "
                    f"+{n/1e6:.2f}M params, "
                    f"heads={args.dfb_heads}, "
                    f"inner_dim_factor={args.dfb_inner_dim_factor}")
        _vram("after-dfb-attach")
        for _i, _blk in enumerate(dfb_blocks):
            _log_dtypes(_blk, f"dfb_block_{_i}-before-prepare")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        for blk in dfb_blocks:
            blk.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on UNet + DFB blocks.")

    if hasattr(F, "scaled_dot_product_attention"):
        from diffusers.models.attention_processor import AttnProcessor2_0
        unet.set_attn_processor(AttnProcessor2_0())
        logger.info("[ATTN] Using PyTorch 2.0 SDPA (mem-efficient).")
    elif args.enable_xformers and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        logger.info("[ATTN] Using xformers.")
    else:
        unet.set_attention_slice("auto")
        logger.info("[ATTN] Fallback: attention_slice=auto.")

    if args.enable_vae_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if args.enable_vae_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    sched_config = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler").config

    if args.scale_lr:
        args.learning_rate = (args.learning_rate
                              * args.gradient_accumulation_steps
                              * args.train_batch_size
                              * accelerator.num_processes)

    using_deepspeed = False
    if hasattr(accelerator.state.distributed_type, "DEEPSPEED"):
        using_deepspeed = (accelerator.state.distributed_type
                           == accelerator.state.distributed_type.DEEPSPEED)

    if args.use_8bit_adam and not using_deepspeed:
        if not HAS_BNB:
            raise ImportError("Install bitsandbytes for 8-bit Adam.")
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
        logger.info("[OPT] Using bitsandbytes AdamW8bit")
    else:
        optimizer_class = torch.optim.AdamW
        if args.use_8bit_adam and using_deepspeed:
            logger.warning("[OPT] DeepSpeed active -> ignoring --use_8bit_adam")

    trainable = (itertools.chain(unet.parameters(), text_encoder.parameters())
                 if args.train_text_encoder else unet.parameters())
    optimizer = optimizer_class(
        trainable, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    _log_optimizer_dtypes(optimizer, "optimizer-before-prepare")
    _vram("after-optimizer-build")

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    # ---- dataset & dataloader -----------------------------------------------
    base_dataset = DreamBoothDataset(
        tokenizer=tokenizer,
        datasets_paths=args.instance_data_dir,
        resolution=args.resolution)
    # PATCH E — wrap with soft-mask aware cache-key dataset.
    train_dataset = _CacheKeyDataset(
        base_dataset,
        inner_blur=args.soft_inner_blur,
        outer_blur=args.soft_outer_blur,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=not args.cache_latents,
        collate_fn=lambda ex: _collate_with_cache_keys(ex, tokenizer, args.resolution),
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=getattr(args, "lr_num_cycles", 1)
    )

    # ---- accelerator prepare ------------------------------------------------
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = \
            accelerator.prepare(unet, text_encoder, optimizer,
                                train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = \
            accelerator.prepare(unet, optimizer, train_dataloader,
                                lr_scheduler)
    accelerator.register_for_checkpointing(lr_scheduler)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    _vram("after-accelerator-prepare")
    _log_dtypes(accelerator.unwrap_model(unet), "unet-after-prepare")
    if args.use_dfb:
        for _i, _blk in enumerate(dfb_blocks):
            _log_dtypes(_blk, f"dfb_block_{_i}-after-prepare")
    _log_optimizer_dtypes(optimizer, "optimizer-after-prepare")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        accelerator.device, dtype=torch.float32)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    freq_loss_fn = FrequencyLoss(
        w_fft=args.freq_loss_w_fft,
        w_ll=args.freq_loss_w_ll,
        w_hh=args.freq_loss_w_hh).to(accelerator.device)

    vae_scaling_factor = vae.config.scaling_factor
    latent_cache = None

    if args.cache_latents:
        latent_cache = build_training_cache(
            args=args,
            train_dataloader=train_dataloader,
            vae=vae,
            text_encoder=text_encoder,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            vae_scaling_factor=vae_scaling_factor,
        )
        vae = vae.to("cpu")
        if not args.train_text_encoder and text_encoder is not None:
            text_encoder = text_encoder.to("cpu")
        vram_cleanup()
        logger.info("VAE + TextEncoder parked on CPU after caching.")
        _vram("after-cache-offload")
    else:
        logger.warning("cache_latents=False: VAE + Text Encoder remain "
                       "in VRAM (~5GB extra usage).")

    # ---- tracking -----------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(args.validation_project_name,
                                  config=vars(copy.deepcopy(args)))

    # ---- save / load hooks (unchanged) --------------------------------------
    def save_model_hook(models, weights, output_dir):
        for model in models:
            model_to_save = accelerator.unwrap_model(model)
            if isinstance(model_to_save, UNet2DConditionModel):
                if accelerator.is_main_process:
                    save_unet_with_dfb(
                        model_to_save, output_dir,
                        heads=args.dfb_heads,
                        inner_dim_factor=args.dfb_inner_dim_factor)
            elif isinstance(model_to_save, CLIPTextModel):
                if accelerator.is_main_process:
                    model_to_save.save_pretrained(
                        os.path.join(output_dir, "text_encoder"))
            weights.pop()

    def load_model_hook(models, input_dir):
        for model in models:
            model_to_load = accelerator.unwrap_model(model)
            if isinstance(model_to_load, UNet2DConditionModel):
                if args.use_dfb and not any(
                    isinstance(b, _DFBHookWrapper)
                    for b in model_to_load.down_blocks
                ):
                    attach_dfb_to_unet(
                        model_to_load, heads=args.dfb_heads,
                        inner_dim_factor=args.dfb_inner_dim_factor)
                    logger.info("[RESUME] Re-attached DFB blocks before loading")
                loaded = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet")
                model_to_load.register_to_config(**loaded.config)
                model_to_load.load_state_dict(loaded.state_dict(), strict=False)
                unet_dir = os.path.join(input_dir, "unet")
                dfb_state = load_dfb_state(unet_dir)
                if dfb_state and args.use_dfb:
                    dfb_cfg = load_dfb_config(unet_dir)
                    restore_dfb_in_unet(
                        model_to_load,
                        {k: v.to(model_to_load.device if hasattr(model_to_load, 'device') else 'cpu')
                         for k, v in dfb_state.items()},
                        heads=dfb_cfg.get("heads", args.dfb_heads),
                        inner_dim_factor=dfb_cfg.get(
                            "inner_dim_factor", args.dfb_inner_dim_factor))
                    logger.info(f"[RESUME] Loaded {len(dfb_state)} DFB tensors")
                del loaded
            elif isinstance(model_to_load, CLIPTextModel):
                loaded = CLIPTextModel.from_pretrained(
                    input_dir, subfolder="text_encoder")
                model_to_load.load_state_dict(loaded.state_dict())
                del loaded
        gc.collect()
        torch.cuda.empty_cache()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ---- logging header -----------------------------------------------------
    total_batch = (args.train_batch_size * accelerator.num_processes
                   * args.gradient_accumulation_steps)
    if accelerator.is_main_process:
        _sep = "=" * 72
        logger.info(_sep)
        logger.info("***** MedDiff-FT + DFB v7 (Soft-mask training) *****")
        logger.info(f"  Log dir            : {_log_dir}")
        logger.info(f"  Ranks              : {_nproc}")
        logger.info(f"  Mixed precision    : {args.mixed_precision}")
        logger.info(f"  DFB                : {args.use_dfb}  "
                    f"heads={args.dfb_heads}  "
                    f"inner_dim_factor={args.dfb_inner_dim_factor}")
        logger.info(f"  Freq-loss          : weight={args.freq_loss_weight}  "
                    f"prob={args.freq_loss_prob}  "
                    f"w_hh={args.freq_loss_w_hh}  (binary mask)")
        logger.info(f"  Soft mask blur     : inner={args.soft_inner_blur}px "
                    f"outer={args.soft_outer_blur}px")
        logger.info(f"  LR / warmup        : {args.learning_rate}  "
                    f"/ {args.lr_warmup_steps} steps")
        logger.info(f"  Batch/accum/GPUs   : "
                    f"{args.train_batch_size} / "
                    f"{args.gradient_accumulation_steps} / "
                    f"{_nproc}  "
                    f"→ effective={total_batch}")
        logger.info(f"  Max steps          : {args.max_train_steps}")
        logger.info(f"  Checkpointing      : every {args.checkpointing_steps} steps "
                    f"(from step {args.checkpointing_from})")
        logger.info(f"  Latent caching     : {args.cache_latents}")
        logger.info(f"  Validation         : {not args.disable_validation}")
        logger.info(_sep)
        _vram("training-start")

    # ---- resume from checkpoint ---------------------------------------------
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir)
                    if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' not found. "
                "Starting from scratch.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[-1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (resume_global_step
                           % (num_update_steps_per_epoch
                              * args.gradient_accumulation_steps))

    num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    progress = tqdm(range(global_step, args.max_train_steps),
                    disable=not accelerator.is_local_main_process)
    progress.set_description("Steps")

    # =========================================================================
    # Training loop
    # =========================================================================
    _LOG_EVERY = 50
    f_loss_val = 0.0
    latH = args.resolution // 8

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            if (args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step):
                if step % args.gradient_accumulation_steps == 0:
                    progress.update(1)
                continue

            with accelerator.accumulate(unet):
                if global_step == 0:
                    logger.info(f"[STEP0] rank{_rank}: accumulate block entered")
                    _vram("step0-accumulate-start")

                # ---- data loading -------------------------------------------
                if args.cache_latents and latent_cache is not None:
                    cache_key = batch.get("cache_keys", [0])[0]
                    cached = latent_cache.get(cache_key)
                    if cached is None:
                        # On-the-fly recovery (unchanged), with hard/soft from batch.
                        logger.warning(
                            f"Cache miss for key {cache_key}, reloading models")
                        temp_vae = AutoencoderKL.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="vae").requires_grad_(False).to(
                                accelerator.device, dtype=weight_dtype)
                        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                        latents = temp_vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae_scaling_factor
                        masked_images = batch["masked_images"].to(dtype=weight_dtype)
                        masked_latents = (temp_vae.encode(masked_images)
                                          .latent_dist.sample()
                                          * vae_scaling_factor)
                        del temp_vae
                        masks = batch.get("hard_masks", batch["masks"])
                        if isinstance(masks, list):
                            masks = torch.stack(masks)
                        mask = F.interpolate(
                            masks.view(-1, 1, masks.shape[-2], masks.shape[-1]).float(),
                            size=(latH, latH),
                            mode="nearest").to(dtype=weight_dtype)
                        soft_mask_pix = batch.get("soft_masks", masks)
                        soft_mask = F.interpolate(
                            soft_mask_pix.view(-1, 1, soft_mask_pix.shape[-2],
                                               soft_mask_pix.shape[-1]).float(),
                            size=(latH, latH),
                            mode="bilinear", align_corners=False).clamp(
                                0.0, 1.0).to(dtype=weight_dtype)
                        temp_te = text_encoder
                        if temp_te is None:
                            temp_te = CLIPTextModel.from_pretrained(
                                args.pretrained_model_name_or_path,
                                subfolder="text_encoder").requires_grad_(False).to(
                                    accelerator.device, dtype=weight_dtype)
                        encoder_hidden = temp_te(batch["input_ids"])[0]
                        if text_encoder is None:
                            del temp_te
                    else:
                        latents = cached["latents"].to(
                            accelerator.device, dtype=weight_dtype,
                            non_blocking=True)
                        masked_latents = cached["masked_latents"].to(
                            accelerator.device, dtype=weight_dtype,
                            non_blocking=True)
                        encoder_hidden = cached["encoder_hidden_states"].to(
                            accelerator.device, dtype=weight_dtype,
                            non_blocking=True)
                        mask = cached["mask"].to(                  # BINARY
                            accelerator.device, dtype=weight_dtype,
                            non_blocking=True)
                        soft_mask = cached["soft_mask"].to(        # SOFT
                            accelerator.device, dtype=weight_dtype,
                            non_blocking=True)
                else:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae_scaling_factor
                    masked_images = batch["masked_images"].to(dtype=weight_dtype)
                    masked_latents = (vae.encode(masked_images)
                                      .latent_dist.sample()
                                      * vae_scaling_factor)
                    masks = batch.get("hard_masks", batch["masks"])
                    if isinstance(masks, list):
                        masks = torch.stack(masks)
                    mask = F.interpolate(
                        masks.view(-1, 1, masks.shape[-2], masks.shape[-1]).float(),
                        size=(latH, latH),
                        mode="nearest").to(dtype=weight_dtype)
                    soft_mask_pix = batch.get("soft_masks", masks)
                    soft_mask = F.interpolate(
                        soft_mask_pix.view(-1, 1, soft_mask_pix.shape[-2],
                                           soft_mask_pix.shape[-1]).float(),
                        size=(latH, latH),
                        mode="bilinear", align_corners=False).clamp(
                            0.0, 1.0).to(dtype=weight_dtype)
                    encoder_hidden = text_encoder(batch["input_ids"])[0]

                # Safety: spatial sizes must match the latent.
                if soft_mask.shape[-1] != latents.shape[-1]:
                    soft_mask = F.interpolate(
                        soft_mask.float(),
                        size=latents.shape[-2:],
                        mode="bilinear", align_corners=False
                    ).clamp(0.0, 1.0).to(weight_dtype)
                if mask.shape[-1] != latents.shape[-1]:
                    mask = F.interpolate(
                        mask.float(),
                        size=latents.shape[-2:],
                        mode="nearest").to(weight_dtype)

                # ---- noise + timesteps --------------------------------------
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)
                # The UNet input mask channel stays BINARY — this is what
                # SD-1.5 inpainting was pretrained with and what the DFB
                # blocks have been tuned against. The soft mask is only a
                # *loss weight*, never a model input.
                latent_model_input = torch.cat(
                    [noisy_latents, mask, masked_latents], dim=1)

                # ---- UNet forward -------------------------------------------
                noise_pred = unet(latent_model_input, timesteps,
                                  encoder_hidden).sample

                if global_step == 0:
                    logger.info(f"[STEP0] rank{_rank}: forward done  "
                                f"noise_pred.dtype={noise_pred.dtype}  "
                                f"shape={tuple(noise_pred.shape)}  "
                                f"soft_mask range=[{soft_mask.min().item():.3f},"
                                f"{soft_mask.max().item():.3f}]  "
                                f"hard_mask sum={mask.sum().item():.0f}")

                # ---- losses (PATCH G) ---------------------------------------
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError("Unknown prediction type")

                # Differential-Diffusion pixel-wise weighted MSE.
                # The soft mask is broadcast over the 4 latent channels
                # automatically (shape [B, 1, h, w] -> [B, 4, h, w]).
                #
                # Normalising by ``soft.sum() + eps`` instead of ``.mean()``
                # would give a region-only loss; we deliberately keep
                # ``.mean()`` so the background still contributes (with
                # weight ~0 outside the transition band) and the loss
                # magnitude stays compatible with the existing LR schedule.
                sq_err = (noise_pred.float() - target.float()) ** 2
                base_loss = (soft_mask.float() * sq_err).mean()

                if global_step == 0:
                    logger.info(f"[STEP0] rank{_rank}: "
                                f"weighted base_loss={base_loss.item():.6f}")

                if (args.freq_loss_weight > 0
                        and random.random() < args.freq_loss_prob):
                    sqrt_a   = sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(weight_dtype)
                    sqrt_1ma = sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(weight_dtype)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        x0_pred = (noisy_latents - sqrt_1ma * noise_pred) / sqrt_a
                    else:
                        x0_pred = sqrt_a * noisy_latents - sqrt_1ma * noise_pred
                    # CRITICAL: frequency loss must see the BINARY mask to
                    # avoid windowing artifacts in the FFT/DWT spectra.
                    f_loss = freq_loss_fn(
                        x0_pred.float(), latents.float(), mask=mask)
                    f_loss_val = f_loss.item()
                    loss = base_loss + args.freq_loss_weight * f_loss
                else:
                    loss = base_loss

                # ---- backward -----------------------------------------------
                accelerator.backward(loss)

                if global_step == 0:
                    logger.info(f"[STEP0] rank{_rank}: backward() done")
                    if args.use_dfb:
                        for _i, _blk in enumerate(dfb_blocks):
                            for _pname, _p in _blk.named_parameters():
                                if _p.grad is None:
                                    logger.warning(
                                        f"[STEP0] dfb_block_{_i}.{_pname}: "
                                        f"grad=None  param_dtype={_p.dtype}")
                                elif not torch.isfinite(_p.grad).all():
                                    logger.warning(
                                        f"[STEP0] dfb_block_{_i}.{_pname}: "
                                        f"grad has inf/nan!  "
                                        f"param_dtype={_p.dtype}  "
                                        f"grad_dtype={_p.grad.dtype}")
                                else:
                                    logger.debug(
                                        f"[STEP0] dfb_block_{_i}.{_pname}: "
                                        f"grad_norm={_p.grad.norm():.4f}")

                if accelerator.sync_gradients:
                    if global_step == 0:
                        logger.info(
                            f"[STEP0] rank{_rank}: "
                            f"pre-clip_grad_norm_ dtype audit...")
                        for _gi, _group in enumerate(optimizer.param_groups):
                            for _p in _group["params"]:
                                if (_p.grad is not None
                                        and _p.dtype != torch.float32):
                                    logger.warning(
                                        f"[STEP0] rank{_rank}: "
                                        f"NON-FP32 PARAM WITH GRAD  "
                                        f"group={_gi}  "
                                        f"param_dtype={_p.dtype}  "
                                        f"grad_dtype={_p.grad.dtype}  "
                                        f"shape={tuple(_p.shape)}")

                    params_to_clip = (
                        itertools.chain(unet.parameters(),
                                        text_encoder.parameters())
                        if args.train_text_encoder else unet.parameters())
                    accelerator.clip_grad_norm_(params_to_clip,
                                                args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if global_step == 0:
                    logger.info(
                        f"[STEP0] rank{_rank}: "
                        f"clip+step succeeded  loss={loss.item():.6f}")
                    _vram("step0-after-optim")

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1

                if (global_step % _LOG_EVERY == 0
                        and accelerator.is_main_process):
                    _lr_now = (lr_scheduler.get_last_lr()[0]
                               if hasattr(lr_scheduler, "get_last_lr")
                               else args.learning_rate)
                    logger.info(
                        f"[STEP {global_step:05d}/{args.max_train_steps}] "
                        f"loss={loss.item():.6f}  "
                        f"freq_loss={f_loss_val:.6f}  "
                        f"lr={_lr_now:.3e}")
                    if global_step % 250 == 0:
                        _vram(f"step-{global_step}")

                # ---- checkpointing ------------------------------------------
                if (global_step % args.checkpointing_steps == 0
                        and global_step >= args.checkpointing_from):
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        logger.info(
                            f"[CKPT] ── checkpoint at step {global_step} ──")
                        _vram(f"pre-ckpt-{global_step}")
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        vram_cleanup()
                        if (args.checkpoints_total_limit
                                and args.checkpoints_total_limit > 0):
                            existing = sorted(
                                [d for d in os.listdir(args.output_dir)
                                 if d.startswith("checkpoint-")],
                                key=lambda x: int(x.split("-")[-1]))
                            for d in existing[:-args.checkpoints_total_limit]:
                                shutil.rmtree(
                                    os.path.join(args.output_dir, d),
                                    ignore_errors=True)
                        logger.info(
                            f"[CKPT] saved → {save_path} | "
                            f"VRAM: {vram_str()}")
                        _vram(f"post-ckpt-{global_step}")
                        vram_cleanup()
                    accelerator.wait_for_everyone()

                # ---- validation ---------------------------------------------
                if (not args.disable_validation
                        and global_step % args.validation_steps == 0
                        and global_step >= args.validation_from
                        and args.validation_image is not None):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        vram_cleanup()
                        moved_vae = (vae is not None and vae.device.type == "cpu")
                        moved_te  = (text_encoder is not None and text_encoder.device.type == "cpu")
                        try:
                            if moved_vae:
                                vae.to(accelerator.device)
                            if moved_te and text_encoder is not None:
                                text_encoder.to(accelerator.device)
                            in_place_validation(
                                args,
                                accelerator.unwrap_model(unet),
                                vae, text_encoder,
                                tokenizer, weight_dtype,
                                accelerator.device, global_step,
                                sched_config, dfb_blocks=dfb_blocks)
                        except Exception as e:
                            logger.warning(
                                f"[VAL] Validation failed at step "
                                f"{global_step}: {e}")
                        finally:
                            if moved_vae:
                                vae.to("cpu")
                            if moved_te and text_encoder is not None:
                                text_encoder.to("cpu")
                            vram_cleanup()
                        vram_cleanup()
                    accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    logs = {
                        "loss": float(loss.detach().item()),
                        "lr":   lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step,
                    }
                    progress.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        vram_cleanup()
        if global_step >= args.max_train_steps:
            break

    # ---- final save ---------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vram_cleanup()
        bare_unet = accelerator.unwrap_model(unet)
        save_unet_with_dfb(bare_unet, args.output_dir,
                           heads=args.dfb_heads,
                           inner_dim_factor=args.dfb_inner_dim_factor)
        if args.train_text_encoder:
            te = accelerator.unwrap_model(text_encoder)
            te.save_pretrained(os.path.join(args.output_dir, "text_encoder"))

        try:
            bare_unet_for_pipe = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet")
            bare_unet_for_pipe.load_state_dict(
                {k.replace(".inner.", "."): v
                 for k, v in bare_unet.state_dict().items()
                 if ".dfb." not in k},
                strict=False)

            save_vae = vae
            save_te = text_encoder
            if args.cache_latents:
                save_vae = AutoencoderKL.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="vae").requires_grad_(False)
                save_te = CLIPTextModel.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="text_encoder").requires_grad_(False)

            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=bare_unet_for_pipe,
                text_encoder=save_te,
                vae=save_vae,
                tokenizer=tokenizer,
                safety_checker=None)
            pipeline.save_pretrained(args.output_dir)
            del bare_unet_for_pipe
            vram_cleanup()
        except Exception as e:
            logger.warning(f"[FINAL] Pipeline save failed (UNet+DFB sidecar "
                           f"saved OK): {e}")

        logger.info(f"Final artifacts at {args.output_dir} | "
                    f"VRAM: {vram_str()}")
        _vram("final")

    accelerator.end_training()


if __name__ == "__main__":
    main()


!pip install -q "transformers>=4.56.0" scipy

%%writefile /kaggle/working/MedDiff-FT/main/softmask_utils.py
"""
softmask_utils.py
==================

Helpers for the Differential-Diffusion soft-mask + RePaint integration.

Public API
----------
* ``binary_to_soft_mask(binary_mask_pil, inner_blur=5, outer_blur=5)``
    Convert a binary lesion mask into a smooth-boundary soft mask using
    inside / outside Euclidean distance transforms. Returns
    ``(soft_mask_pil, binary_mask_pil)``.

* ``soft_mask_to_tensor(soft_mask_pil)`` /
  ``binary_mask_to_tensor(binary_mask_pil)``
    Convenience helpers that turn PIL ``L`` masks into
    ``[1, 1, H, W]`` ``float32`` tensors in the ``[0, 1]`` range.

* ``downsample_mask_to_latent(mask_t, latent_size, mode)``
    Spatially downsample a mask tensor to the latent resolution.
    Soft masks should use ``"bilinear"`` (default for floats),
    binary masks must use ``"nearest"``.

The functions are completely independent of the DFB blocks or the
``FrequencyLoss``; they only produce auxiliary tensors that the training
loop consumes as pixel-wise loss weights and that the inference loop
consumes as blending weights.

Memory budget
-------------
* A 512x512 soft mask in uint8  -> 256 KB on disk / RAM
* A 64 x64  soft mask in fp32   -> 16  KB in VRAM per image
* Nothing else is added to the model graph -> zero extra learnable
  parameters, zero impact on DDP ``find_unused_parameters``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    # SciPy is required by the existing notebook (cell 7 installs it).
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback only used if SciPy is absent
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Distance-transform based soft mask
# ---------------------------------------------------------------------------
def _np_distance_transform(binary: np.ndarray) -> np.ndarray:
    """Wrapper around ``scipy.ndimage.distance_transform_edt`` with a tiny
    pure-NumPy fallback (Chebyshev distance) for environments without SciPy."""
    if _HAS_SCIPY:
        return distance_transform_edt(binary).astype(np.float32)

    # Cheap fallback: iterative dilation gives an L_inf distance.
    # Works fine for the small blur radii used here (<= 10 px).
    binary = binary.astype(np.uint8)
    dist = np.zeros_like(binary, dtype=np.float32)
    cur = binary.copy()
    step = 0
    # 8-neighbour erosion until empty (max 50 iters as a safety bound).
    while cur.any() and step < 50:
        eroded = np.zeros_like(cur)
        eroded[1:-1, 1:-1] = (
            cur[1:-1, 1:-1] & cur[:-2, 1:-1] & cur[2:, 1:-1]
            & cur[1:-1, :-2] & cur[1:-1, 2:]
        )
        dist[(cur == 1) & (eroded == 0)] = step + 1
        cur = eroded
        step += 1
    return dist


def binary_to_soft_mask(
    binary_mask_pil: Image.Image,
    inner_blur: int = 5,
    outer_blur: int = 5,
) -> Tuple[Image.Image, Image.Image]:
    """Produce a smooth-boundary soft mask from a binary lesion mask.

    The soft mask is 255 deep inside the lesion, 0 deep in the background,
    and ramps linearly across ``inner_blur`` pixels (inside) and
    ``outer_blur`` pixels (outside) of the lesion boundary.

    Parameters
    ----------
    binary_mask_pil : PIL.Image
        Mode ``L``; non-zero pixels are interpreted as lesion.
    inner_blur, outer_blur : int
        Width (in pixels) of the inside / outside transition band.

    Returns
    -------
    (soft_mask_pil, binary_mask_pil)
        ``soft_mask_pil`` is ``L`` mode (0..255).
        ``binary_mask_pil`` is the input re-thresholded to a clean
        ``{0, 255}`` mask (so the *exact* binary mask we use for the
        frequency loss is unambiguous).
    """
    if binary_mask_pil.mode != "L":
        binary_mask_pil = binary_mask_pil.convert("L")

    arr = np.array(binary_mask_pil, dtype=np.uint8)
    binary = (arr >= 128).astype(np.uint8)         # canonical {0, 1}

    # Edge cases: completely empty or completely full mask.
    if binary.sum() == 0:
        soft = np.zeros_like(arr, dtype=np.uint8)
        return (Image.fromarray(soft, mode="L"),
                Image.fromarray((binary * 255).astype(np.uint8), mode="L"))
    if binary.sum() == binary.size:
        soft = np.full_like(arr, 255, dtype=np.uint8)
        return (Image.fromarray(soft, mode="L"),
                Image.fromarray((binary * 255).astype(np.uint8), mode="L"))

    # Distance INSIDE the lesion (positive for lesion pixels).
    d_in = _np_distance_transform(binary)
    # Distance OUTSIDE the lesion (positive for background pixels).
    d_out = _np_distance_transform(1 - binary)

    # Soft value (in [0, 1]):
    #   - far inside  -> 1.0
    #   - on boundary -> 0.5
    #   - far outside -> 0.0
    inner_blur = max(1, int(inner_blur))
    outer_blur = max(1, int(outer_blur))

    inside_ramp  = np.clip(d_in  / float(inner_blur), 0.0, 1.0)   # 0..1
    outside_ramp = np.clip(d_out / float(outer_blur), 0.0, 1.0)   # 0..1

    # Inside: value = 0.5 + 0.5 * inside_ramp     (-> 1 deep in lesion)
    # Outside: value = 0.5 - 0.5 * outside_ramp   (-> 0 deep in background)
    soft = np.where(
        binary == 1,
        0.5 + 0.5 * inside_ramp,
        0.5 - 0.5 * outside_ramp,
    ).astype(np.float32)

    soft_u8   = (soft * 255.0).clip(0, 255).astype(np.uint8)
    binary_u8 = (binary * 255).astype(np.uint8)

    return Image.fromarray(soft_u8, mode="L"), Image.fromarray(binary_u8, mode="L")


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def soft_mask_to_tensor(soft_mask_pil: Image.Image) -> torch.Tensor:
    """PIL ``L`` mask -> ``[1, 1, H, W]`` float32 tensor in ``[0, 1]``."""
    if soft_mask_pil.mode != "L":
        soft_mask_pil = soft_mask_pil.convert("L")
    arr = np.array(soft_mask_pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()


def binary_mask_to_tensor(
    binary_mask_pil: Image.Image, threshold: int = 128
) -> torch.Tensor:
    """PIL ``L`` mask -> ``[1, 1, H, W]`` float32 tensor in ``{0, 1}``."""
    if binary_mask_pil.mode != "L":
        binary_mask_pil = binary_mask_pil.convert("L")
    arr = (np.array(binary_mask_pil, dtype=np.uint8) >= threshold).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()


# ---------------------------------------------------------------------------
# Latent-resolution resampling
# ---------------------------------------------------------------------------
def downsample_mask_to_latent(
    mask_t: torch.Tensor,
    latent_size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resample a mask tensor (any spatial size) to ``latent_size`` x
    ``latent_size``.

    Use ``mode='nearest'`` for binary masks (preserves {0, 1}),
    ``mode='bilinear'`` for soft masks (smooth interpolation).
    """
    if mask_t.dim() == 3:
        mask_t = mask_t.unsqueeze(0)
    if mask_t.shape[-1] == latent_size and mask_t.shape[-2] == latent_size:
        return mask_t
    align = False if mode == "bilinear" else None
    kwargs = {"mode": mode}
    if align is not None:
        kwargs["align_corners"] = align
    out = F.interpolate(mask_t.float(), size=(latent_size, latent_size), **kwargs)
    return out


# ---------------------------------------------------------------------------
# One-shot training-side helper: produce (soft_lat, hard_lat) tensors from a
# binary PIL mask. This is exactly what the training collate_fn caches.
# ---------------------------------------------------------------------------
def build_training_masks(
    binary_mask_pil: Image.Image,
    image_resolution: int,
    latent_resolution: int,
    inner_blur: int = 5,
    outer_blur: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience pipeline used by the dataset.

    Returns
    -------
    soft_pix : ``[1, H, W]`` float32  in [0, 1]    (image-resolution soft)
    hard_pix : ``[1, H, W]`` float32  in {0, 1}    (image-resolution binary)
    soft_lat : ``[1, h, w]`` float32  in [0, 1]    (latent-resolution soft)
    hard_lat : ``[1, h, w]`` float32  in {0, 1}    (latent-resolution binary)
    """
    # Resize binary mask to the image resolution with NEAREST so we keep a
    # clean binary input to the distance transform.
    if binary_mask_pil.size != (image_resolution, image_resolution):
        binary_mask_pil = binary_mask_pil.resize(
            (image_resolution, image_resolution), Image.NEAREST)

    soft_pil, hard_pil = binary_to_soft_mask(
        binary_mask_pil, inner_blur=inner_blur, outer_blur=outer_blur)

    soft_pix = soft_mask_to_tensor(soft_pil).squeeze(0)        # [1, H, W]
    hard_pix = binary_mask_to_tensor(hard_pil).squeeze(0)      # [1, H, W]

    soft_lat = downsample_mask_to_latent(
        soft_pix.unsqueeze(0), latent_resolution, mode="bilinear").squeeze(0)
    hard_lat = downsample_mask_to_latent(
        hard_pix.unsqueeze(0), latent_resolution, mode="nearest").squeeze(0)

    return soft_pix, hard_pix, soft_lat, hard_lat


__all__ = [
    "binary_to_soft_mask",
    "soft_mask_to_tensor",
    "binary_mask_to_tensor",
    "downsample_mask_to_latent",
    "build_training_masks",
]


%%writefile /kaggle/working/MedDiff-FT/main/infer_dfb.py
"""
infer_dfb.py  (v8 — RePaint + soft-mask + standard pretrained IP-Adapter)
=========================================================================

Standalone inference for MedDiff-FT + DFB checkpoints.

What changes vs. v7
-------------------
**v8 adds the standard pretrained IP-Adapter (h94/IP-Adapter, ip-adapter_sd15.bin)
as an optional image-prompt conditioning channel.**

The integration is designed to be *surgical* — every component of v7 still
runs in exactly the same way when IP-Adapter is disabled. When enabled:

  • IP-Adapter weights are loaded onto the UNet's cross-attention layers via
    `StableDiffusionInpaintPipeline.load_ip_adapter` (using our already-loaded
    unet/vae/text_encoder/tokenizer/scheduler — no duplicate weights).
  • The DFB skip-connection wrappers are NOT touched (IP-Adapter only swaps
    attention processors; DFB lives on the upsampling skips → disjoint paths).
  • The 9-channel UNet input is NOT touched (IP-Adapter enters via
    cross-attention; conv_in remains 9-ch).
  • Image embeddings are computed ONCE per reference image and reused across
    all denoising steps + RePaint resamples.
  • Memory overhead on a T4 (16 GB): ~1.2 GB for the CLIP-ViT-H image encoder
    (fp16) + ~22 MB for IP-Adapter weights — still leaves ample headroom.

CLI additions
-------------
    --ip_adapter_image            Path to a single global reference image.
    --ip_adapter_image_dir        Per-mask reference dir (paired by sorted order).
    --ip_adapter_repo             Default: "h94/IP-Adapter"
    --ip_adapter_subfolder        Default: "models"
    --ip_adapter_weight_name      Default: "ip-adapter_sd15.bin"
    --ip_adapter_scale            Default: 0.6
    --offload_image_encoder       Move image encoder to CPU after encoding
                                  (saves ~1.2 GB VRAM if you need it back).

If neither --ip_adapter_image nor --ip_adapter_image_dir is given,
IP-Adapter is automatically disabled and behaviour is IDENTICAL to v7.

v7 behaviour kept
-----------------
1. Soft-mask blending after every denoising step.
2. CFG applied BEFORE blending.
3. Optional RePaint resampling at warm steps.

Memory (per image, on top of v7)
--------------------------------
    * ip_image_embeds (fp16):    [2, 1, 1024] or [2, 257, 1280]   -> < 2 MB
    * patched attn-proc weights: shipped on the UNet itself        -> ~22 MB
Total extra VRAM during the denoising loop: << 5 MB.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (AutoencoderKL, DDIMScheduler,
                       StableDiffusionInpaintPipeline, UNet2DConditionModel)

# Local imports (DFB + soft-mask utilities).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meddiff_dfb import (
    restore_dfb_in_unet, load_dfb_state, load_dfb_config, vram_cleanup,
)
from softmask_utils import (
    binary_to_soft_mask, soft_mask_to_tensor, binary_mask_to_tensor,
    downsample_mask_to_latent,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",          required=True)
    p.add_argument("--input_path",          required=True)
    p.add_argument("--label_path",          required=True)
    p.add_argument("--output_path",         required=True)
    p.add_argument("--prompt",              default="a photo of hta")
    p.add_argument("--negative_prompt",     default="")
    p.add_argument("--seed",                type=int,   default=12345)
    p.add_argument("--num_inference_steps", type=int,   default=50)
    p.add_argument("--guidance_scale",      type=float, default=7.5)
    p.add_argument("--resolution",          type=int,   default=512)
    p.add_argument("--batch_size",          type=int,   default=1)
    p.add_argument("--mixed_precision",     default="fp16", choices=["no", "fp16"])
    p.add_argument("--device",              default="cuda:0")
    p.add_argument("--enable_vae_slicing",  action="store_true", default=True)
    p.add_argument("--dfb_heads",           type=int,   default=None)
    p.add_argument("--dfb_inner_dim_factor",type=float, default=None)

    # ---- Soft-mask + RePaint flags (v7) -------------------------------------
    p.add_argument("--soft_inner_blur",     type=int,   default=5,
                   help="Inner width (px) of soft-mask transition band.")
    p.add_argument("--soft_outer_blur",     type=int,   default=5,
                   help="Outer width (px) of soft-mask transition band.")
    p.add_argument("--use_soft_blend",      action="store_true", default=True,
                   help="Blend background latent with soft mask each step.")
    p.add_argument("--no_soft_blend",       action="store_true", default=False,
                   help="Disable soft-mask blending (debug only).")
    p.add_argument("--use_repaint",         action="store_true", default=False,
                   help="Enable RePaint-style resampling at warm steps.")
    p.add_argument("--resample_steps",      type=int,   default=2,
                   help="How many resamples per warmup step.")
    p.add_argument("--resample_jump",       type=int,   default=3,
                   help="How many timesteps to jump back per resample.")
    p.add_argument("--resample_warmup",     type=int,   default=10,
                   help="Number of early denoising steps that use resampling.")

    # ---- IP-Adapter flags (NEW in v8) ---------------------------------------
    p.add_argument("--ip_adapter_image",      default=None,
                   help="Path to ONE global reference image used for ALL "
                        "generations. If both this and --ip_adapter_image_dir "
                        "are set, this single image wins.")
    p.add_argument("--ip_adapter_image_dir",  default=None,
                   help="Directory of reference images paired by sorted order "
                        "with --label_path masks (one ref per generation).")
    p.add_argument("--ip_adapter_repo",       default="h94/IP-Adapter",
                   help="HF repo id with IP-Adapter weights.")
    p.add_argument("--ip_adapter_subfolder",  default="models",
                   help="Subfolder inside the repo holding the weights.")
    p.add_argument("--ip_adapter_weight_name", default="ip-adapter_sd15.bin",
                   help="Weight file name (default: ip-adapter_sd15.bin).")
    p.add_argument("--ip_adapter_scale",      type=float, default=0.6,
                   help="IP-Adapter conditioning scale (0 = disabled).")
    p.add_argument("--offload_image_encoder", action="store_true", default=False,
                   help="Move CLIP image encoder to CPU between generations "
                        "(saves ~1.2 GB VRAM at small re-encode cost).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_unet_with_dfb(model_path: str, dtype: torch.dtype,
                       dfb_heads: Optional[int],
                       dfb_idf: Optional[float]) -> UNet2DConditionModel:
    print(f"[Gen] Loading UNet from {model_path}/unet/ ...")
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", torch_dtype=dtype)
    unet_dir = os.path.join(model_path, "unet")
    cfg = load_dfb_config(unet_dir)
    heads = dfb_heads if dfb_heads is not None else cfg["heads"]
    idf   = dfb_idf   if dfb_idf   is not None else cfg["inner_dim_factor"]

    dfb_state = load_dfb_state(unet_dir)
    if dfb_state:
        dfb_state = {k: v.to(dtype) for k, v in dfb_state.items()}
        restore_dfb_in_unet(unet, dfb_state, heads=heads, inner_dim_factor=idf)
        print(f"[Gen] DFB tensors loaded: {len(dfb_state)} "
              f"(heads={heads}, idf={idf})")
    else:
        print("[Gen] No dfb_weights found -- running without DFB.")
    return unet


# ---------------------------------------------------------------------------
# IP-Adapter helpers (NEW in v8)
# ---------------------------------------------------------------------------
def _attach_ip_adapter(
    *,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler,
    device: torch.device,
    dtype: torch.dtype,
    repo: str,
    subfolder: str,
    weight_name: str,
    scale: float,
) -> Tuple[StableDiffusionInpaintPipeline, "callable"]:
    """Load the pretrained IP-Adapter onto our already-built modules.

    We build a *temporary* StableDiffusionInpaintPipeline that REUSES our
    existing components (no duplicate VRAM allocations) and then call its
    ``load_ip_adapter`` method. This is the officially-supported diffusers
    entry-point and it:

      * downloads the IP-Adapter weights from HF Hub,
      * patches ``unet`` cross-attention with IPAdapterAttnProcessor,
      * pulls in the matching CLIP image encoder (image_encoder) and
        feature_extractor (CLIPImageProcessor), placing them on the pipe.

    We then keep the pipe alive purely as a thin wrapper around these
    auxiliary components (image_encoder + feature_extractor + the
    ``prepare_ip_adapter_image_embeds`` helper). The actual denoising loop
    still runs manually against ``unet`` / ``vae`` / ``text_encoder``.

    Returns
    -------
    pipe      : the wrapper pipeline (do NOT call pipe(...) — we only use
                its image-encoding utilities).
    embed_fn  : closure taking a PIL.Image (or None) and ``do_cfg`` flag,
                returning a list[Tensor] suitable for
                ``added_cond_kwargs={"image_embeds": [...]}``.
    """
    print(f"[Gen] Attaching IP-Adapter: {repo}/{subfolder}/{weight_name} "
          f"(scale={scale}) ...")

    # Build the pipeline shell from our existing (already-loaded) modules.
    # No safety checker, no feature extractor here — load_ip_adapter will
    # fetch its own CLIPImageProcessor + CLIPVisionModelWithProjection.
    pipe = StableDiffusionInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    # This call:
    #   1. Downloads + loads the IP-Adapter state_dict into unet's attn procs.
    #   2. Loads the CLIP image encoder from {repo}/{subfolder}/image_encoder
    #      and assigns it to pipe.image_encoder.
    #   3. Loads CLIPImageProcessor as pipe.feature_extractor.
    pipe.load_ip_adapter(
        repo,
        subfolder=subfolder,
        weight_name=weight_name,
    )

    # Cast image encoder to the same dtype + device as the rest of the stack.
    if pipe.image_encoder is not None:
        pipe.image_encoder.to(device=device, dtype=dtype).eval()

    # Set scaling factor on every IP-Adapter attention processor.
    pipe.set_ip_adapter_scale(scale)

    print(f"[Gen] IP-Adapter ready  "
          f"(image_encoder dtype={dtype}, scale={scale})")

    # ---- embed_fn closure ---------------------------------------------------
    @torch.inference_mode()
    def embed_fn(
        ref_pil: Optional[Image.Image],
        do_cfg: bool,
    ) -> Optional[list]:
        """Compute IP-Adapter image embeddings for one reference image.

        Returns a *list* of tensors (one per active IP-Adapter), each
        shaped [B, ...] where B=2 if do_cfg else 1. The diffusers UNet
        expects this list under ``added_cond_kwargs["image_embeds"]``.

        If ``ref_pil`` is None, returns None (caller must skip IP-Adapter).
        """
        if ref_pil is None:
            return None
        # Diffusers' helper handles: CLIP preprocessing, projection,
        # zero-uncond construction, CFG stacking, dtype/device placement.
        embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[ref_pil.convert("RGB")],
            ip_adapter_image_embeds=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
        # ``embeds`` is a list[Tensor]; each is already dtype-cast.
        return [e.to(dtype) for e in embeds]

    return pipe, embed_fn


def _resolve_ref_image(
    args,
    mask_filename: str,
    fallback_global: Optional[Image.Image],
    ip_dir_files: Optional[List[str]],
    pair_idx: int,
) -> Optional[Image.Image]:
    """Pick a per-generation IP-Adapter reference image.

    Priority:
      1. ``--ip_adapter_image`` (single global)  -> fallback_global
      2. ``--ip_adapter_image_dir`` (paired)     -> by sorted order, modulo
      3. None -> IP-Adapter disabled for this generation.
    """
    if fallback_global is not None:
        return fallback_global
    if ip_dir_files:
        f = ip_dir_files[pair_idx % len(ip_dir_files)]
        return Image.open(os.path.join(args.ip_adapter_image_dir, f)).convert("RGB")
    return None


# ---------------------------------------------------------------------------
# RePaint + soft-mask generation kernel
# ---------------------------------------------------------------------------
@torch.inference_mode()
def repaint_softmask_generate(
    *,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler,
    init_pil: Image.Image,
    binary_mask_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    soft_inner_blur: int = 5,
    soft_outer_blur: int = 5,
    use_soft_blend: bool = True,
    use_repaint: bool = False,
    resample_steps: int = 2,
    resample_jump: int = 3,
    resample_warmup: int = 10,
    # ---- NEW: IP-Adapter inputs (None => not used) ---------------------
    ip_image_embeds: Optional[list] = None,
) -> Image.Image:
    """Single-image inpaint with RePaint + soft-mask blending.

    If ``ip_image_embeds`` is not None it must be a list of tensors as
    returned by ``pipe.prepare_ip_adapter_image_embeds(...)``, with the
    first dim already CFG-stacked to match ``do_cfg``.
    """
    H = resolution
    latH = H // 8
    g = torch.Generator(device=device).manual_seed(seed)
    do_cfg = guidance_scale > 1.0

    # ---- Pre-process image + masks ------------------------------------------
    init = init_pil.convert("RGB").resize((H, H), Image.LANCZOS)
    bmask = binary_mask_pil.convert("L").resize((H, H), Image.NEAREST)

    # Soft + clean binary masks (image resolution).
    soft_pil, hard_pil = binary_to_soft_mask(
        bmask, inner_blur=soft_inner_blur, outer_blur=soft_outer_blur)

    soft_pix = soft_mask_to_tensor(soft_pil).to(device, dtype)
    hard_pix = binary_mask_to_tensor(hard_pil).to(device, dtype)

    # Latent-resolution masks.
    soft_lat = downsample_mask_to_latent(
        soft_pix, latH, mode="bilinear").to(device, dtype).clamp(0.0, 1.0)
    hard_lat = downsample_mask_to_latent(
        hard_pix, latH, mode="nearest").to(device, dtype)

    # ---- VAE encode init + known background ---------------------------------
    img_t = (torch.from_numpy(np.array(init)).to(device).float() / 127.5) - 1.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(dtype)             # [1,3,H,H]

    # The UNet inpaint input wants the masked image (background only) encoded.
    # For RePaint we ALSO need an encoding of the *true* background.
    masked_img = img_t * (1.0 - hard_pix)                              # zeros under lesion
    masked_lat = (vae.encode(masked_img).latent_dist.sample(generator=g)
                  * vae.config.scaling_factor).to(dtype)               # [1,4,latH,latH]
    init_lat = (vae.encode(img_t).latent_dist.sample(generator=g)
                * vae.config.scaling_factor).to(dtype)                 # full image

    # ``known_latents`` is the clean reference we will keep nailed to the
    # background at every step. We use the full encoded image so the
    # background pixels exactly match ``init``; the lesion pixels of
    # ``known_latents`` are irrelevant (masked out by ``(1 - soft_lat)``
    # only in the background region after blending).
    known_lat = init_lat

    # ---- Text embeddings (CFG) ----------------------------------------------
    def _embed(text: str) -> torch.Tensor:
        toks = tokenizer(text, padding="max_length",
                         max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(device)
        return text_encoder(toks)[0].to(dtype)

    cond = _embed(prompt)
    if do_cfg:
        uncond = _embed(negative_prompt or "")
        text_emb = torch.cat([uncond, cond], dim=0)                    # [2, T, D]
    else:
        text_emb = cond

    # ---- IP-Adapter added_cond_kwargs (NEW) ---------------------------------
    # ip_image_embeds is already CFG-stacked (first dim 2 if do_cfg else 1),
    # so we can pass it directly. The diffusers UNet looks for the key
    # "image_embeds" in added_cond_kwargs when IP-Adapter procs are present.
    use_ipa = ip_image_embeds is not None
    added_cond_kwargs = {"image_embeds": ip_image_embeds} if use_ipa else None

    # ---- Scheduler timesteps ------------------------------------------------
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    num_train_timesteps = scheduler.config.num_train_timesteps

    # ---- Initial latent (pure noise; SD-1.5 standard) -----------------------
    latents = torch.randn(
        (1, 4, latH, latH), generator=g, device=device, dtype=dtype
    ) * scheduler.init_noise_sigma

    # Helper: build the 9-channel UNet input for inpainting.
    def _unet_in(z: torch.Tensor) -> torch.Tensor:
        # SD-1.5 inpainting expects: [noisy_latents (4ch), mask (1ch),
        #                             masked_image_latents (4ch)] = 9ch.
        return torch.cat([z, hard_lat, masked_lat], dim=1)

    # Helper: noise prediction with CFG  (+ optional IP-Adapter conditioning).
    def _predict_noise(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if do_cfg:
            z_in = torch.cat([z, z], dim=0)
            z_in = scheduler.scale_model_input(z_in, t)
            unet_in = _unet_in(z_in.chunk(2, dim=0)[0])
            unet_in = torch.cat([unet_in, _unet_in(z_in.chunk(2, dim=0)[1])],
                                dim=0)
            eps = unet(
                unet_in, t,
                encoder_hidden_states=text_emb,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            eps_u, eps_c = eps.chunk(2, dim=0)
            return eps_u + guidance_scale * (eps_c - eps_u)
        else:
            z_scaled = scheduler.scale_model_input(z, t)
            return unet(
                _unet_in(z_scaled), t,
                encoder_hidden_states=text_emb,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

    # Helper: re-noise the known background to timestep ``t``.
    # For DDIM/DDPM schedulers, ``scheduler.add_noise(x0, eps, t)`` works.
    def _renoise_known(t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return scheduler.add_noise(known_lat, eps, t).to(dtype)

    # Helper: soft blend latent <- soft * latent + (1 - soft) * known_noisy.
    do_blend = use_soft_blend
    def _blend(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not do_blend:
            # Hard-blend fallback (RePaint paper recipe): keep the known
            # region exactly equal to the re-noised init.
            eps_known = torch.randn(z.shape, generator=g,
                                    device=device, dtype=dtype)
            known_noisy = _renoise_known(t, eps_known)
            return hard_lat * z + (1.0 - hard_lat) * known_noisy
        eps_known = torch.randn(z.shape, generator=g,
                                device=device, dtype=dtype)
        known_noisy = _renoise_known(t, eps_known)
        return soft_lat * z + (1.0 - soft_lat) * known_noisy

    # ---- Denoising loop -----------------------------------------------------
    n_steps = len(timesteps)
    for i, t in enumerate(timesteps):
        # ---- standard DDIM step (CFG applied BEFORE blending) ---------------
        noise_pred = _predict_noise(latents, t)
        step_out = scheduler.step(noise_pred, t, latents)
        latents = step_out.prev_sample

        # ---- soft / hard mask blending (Differential-Diffusion + RePaint) ---
        # Use the previous timestep (the timestep we just stepped TO) when
        # re-noising; if we're at the last step, t_prev=0 -> known_noisy is
        # essentially the clean background (matches RePaint).
        t_prev = timesteps[i + 1] if i + 1 < n_steps else torch.tensor(
            0, device=device, dtype=t.dtype)
        latents = _blend(latents, t_prev)

        # ---- optional RePaint resampling (only at warm steps) ---------------
        if use_repaint and i < resample_warmup:
            for _r in range(resample_steps):
                # Jump back ``resample_jump`` timesteps (i.e. add noise).
                t_back_idx = max(0, i - resample_jump)
                t_back = timesteps[t_back_idx]
                # Re-add noise to current latent to bring it back to t_back.
                # We use the scheduler's add_noise convention; this is the
                # classic RePaint "jump forward in time".
                eps_jump = torch.randn(latents.shape, generator=g,
                                        device=device, dtype=dtype)
                latents = scheduler.add_noise(
                    latents.float(), eps_jump.float(),
                    t_back.unsqueeze(0) if t_back.dim() == 0 else t_back
                ).to(dtype)
                # Re-denoise from t_back back down to t_prev.
                for j in range(t_back_idx, i + 1):
                    tj = timesteps[j]
                    np_j = _predict_noise(latents, tj)
                    latents = scheduler.step(np_j, tj, latents).prev_sample
                latents = _blend(latents, t_prev)

    # ---- VAE decode ---------------------------------------------------------
    decoded = vae.decode(
        latents.to(dtype) / vae.config.scaling_factor
    ).sample
    img = (decoded.clamp(-1, 1) + 1) / 2
    img = (img.float().permute(0, 2, 3, 1).cpu().numpy()[0] * 255
           ).astype(np.uint8)
    out = Image.fromarray(img)

    out_arr  = np.array(out).astype(np.float32)
    init_arr = np.array(init).astype(np.float32)
    # Use the SOFT mask for seamless blending
    s_arr    = np.array(soft_pil).astype(np.float32) / 255.0   # soft_pil already exists
    s_arr    = s_arr[..., None]
    final    = (s_arr * out_arr + (1.0 - s_arr) * init_arr
                ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(final)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.no_soft_blend:
        args.use_soft_blend = False

    device = torch.device(args.device)
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    os.makedirs(args.output_path, exist_ok=True)

    # Load components.
    unet = _load_unet_with_dfb(
        args.model_path, dtype,
        args.dfb_heads, args.dfb_inner_dim_factor).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=dtype).to(device).eval()
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder",
        torch_dtype=dtype).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_path, subfolder="tokenizer")

    # DDIM is required for the closed-form add_noise / scale_model_input
    # path used here. RePaint's original paper uses DDPM, but DDIM gives
    # equivalent quality with far fewer steps at 16 GB VRAM.
    scheduler = DDIMScheduler.from_pretrained(
        args.model_path, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    if args.enable_vae_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()

    # ---- IP-Adapter attach (NEW in v8) --------------------------------------
    # Only attach if the user actually supplied a reference source AND the
    # scale is > 0. This keeps the script byte-for-byte equivalent to v7
    # when no IP-Adapter inputs are passed.
    want_ipa = (
        (args.ip_adapter_image is not None
         or args.ip_adapter_image_dir is not None)
        and args.ip_adapter_scale > 0.0
    )
    ipa_pipe = None
    embed_fn = None
    global_ref_pil: Optional[Image.Image] = None
    ip_dir_files: Optional[List[str]] = None

    if want_ipa:
        ipa_pipe, embed_fn = _attach_ip_adapter(
            unet=unet, vae=vae, text_encoder=text_encoder,
            tokenizer=tokenizer, scheduler=scheduler,
            device=device, dtype=dtype,
            repo=args.ip_adapter_repo,
            subfolder=args.ip_adapter_subfolder,
            weight_name=args.ip_adapter_weight_name,
            scale=args.ip_adapter_scale,
        )
        # Resolve reference image source.
        if args.ip_adapter_image is not None:
            global_ref_pil = Image.open(args.ip_adapter_image).convert("RGB")
            print(f"[Gen] IP-Adapter ref (global): {args.ip_adapter_image}")
        elif args.ip_adapter_image_dir is not None:
            ip_dir_files = sorted(
                f for f in os.listdir(args.ip_adapter_image_dir)
                if os.path.isfile(os.path.join(args.ip_adapter_image_dir, f))
            )
            print(f"[Gen] IP-Adapter ref dir: {args.ip_adapter_image_dir} "
                  f"({len(ip_dir_files)} images, paired by sorted order)")
    else:
        print("[Gen] IP-Adapter disabled (no ref image supplied "
              "or scale=0). Running standard v7 inference path.")

    inputs = sorted(
        f for f in os.listdir(args.input_path)
        if os.path.isfile(os.path.join(args.input_path, f))
    )
    masks = sorted(
        f for f in os.listdir(args.label_path)
        if os.path.isfile(os.path.join(args.label_path, f))
    )
    print(f"[Gen] {len(inputs)} bg images, {len(masks)} masks")
    print(f"[Gen] soft_blend={args.use_soft_blend}  "
          f"repaint={args.use_repaint}  "
          f"resample_steps={args.resample_steps}  "
          f"resample_warmup={args.resample_warmup}  "
          f"ip_adapter={'on' if want_ipa else 'off'}"
          + (f" (scale={args.ip_adapter_scale})" if want_ipa else ""))

    do_cfg_flag = args.guidance_scale > 1.0

    # If the user passed ONE global ref image, we only need to encode it once.
    cached_global_embeds = None
    if want_ipa and global_ref_pil is not None:
        # Make sure image_encoder is on device for the (one) encode call.
        if args.offload_image_encoder and ipa_pipe.image_encoder is not None:
            ipa_pipe.image_encoder.to(device)
        cached_global_embeds = embed_fn(global_ref_pil, do_cfg_flag)
        if args.offload_image_encoder and ipa_pipe.image_encoder is not None:
            ipa_pipe.image_encoder.to("cpu")
            torch.cuda.empty_cache()
        print(f"[Gen] Cached IP-Adapter embeds: "
              f"{[tuple(e.shape) for e in cached_global_embeds]}")

    idx = 0
    for mi, mf in enumerate(tqdm(masks, desc="Generating")):
        ifn = inputs[idx % len(inputs)]
        idx += 1
        init_pil = Image.open(os.path.join(args.input_path, ifn)).convert("RGB")
        mask_pil = Image.open(os.path.join(args.label_path, mf)).convert("L")

        # ---- per-generation IP-Adapter embeds -----------------------------
        ip_embeds = None
        if want_ipa:
            if cached_global_embeds is not None:
                ip_embeds = cached_global_embeds
            else:
                ref_pil = _resolve_ref_image(
                    args, mf, None, ip_dir_files, mi
                )
                if ref_pil is not None:
                    if args.offload_image_encoder and ipa_pipe.image_encoder is not None:
                        ipa_pipe.image_encoder.to(device)
                    ip_embeds = embed_fn(ref_pil, do_cfg_flag)
                    if args.offload_image_encoder and ipa_pipe.image_encoder is not None:
                        ipa_pipe.image_encoder.to("cpu")
                        torch.cuda.empty_cache()

        out = repaint_softmask_generate(
            unet=unet, vae=vae, text_encoder=text_encoder,
            tokenizer=tokenizer, scheduler=scheduler,
            init_pil=init_pil, binary_mask_pil=mask_pil,
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            resolution=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            device=device, dtype=dtype,
            seed=args.seed + mi,
            soft_inner_blur=args.soft_inner_blur,
            soft_outer_blur=args.soft_outer_blur,
            use_soft_blend=args.use_soft_blend,
            use_repaint=args.use_repaint,
            resample_steps=args.resample_steps,
            resample_jump=args.resample_jump,
            resample_warmup=args.resample_warmup,
            ip_image_embeds=ip_embeds,
        )

        stem = os.path.splitext(mf)[0]
        out.save(os.path.join(args.output_path, f"{stem}.png"))

    # ---- Cleanup ------------------------------------------------------------
    if ipa_pipe is not None:
        # Drop references so VRAM is freed.
        try:
            del ipa_pipe.image_encoder
        except Exception:
            pass
        del ipa_pipe
    del unet, vae, text_encoder
    gc.collect()
    vram_cleanup()
    print(f"[Gen] Done. Wrote {len(masks)} images to {args.output_path}")


if __name__ == "__main__":
    main()


%%writefile /kaggle/working/MedDiff-FT/main/validate_comprehensive.py
"""
validate_comprehensive.py — MedDiff-FT + DFB Comprehensive Evaluator
=====================================================================

v7 (patched): RePaint resampling + soft-mask blending at inference.
The rest of the evaluator (FID/KID/Vendi/memorisation/segmentation) is unchanged.

Usage:
    python validate_comprehensive.py --model_path ...  (see argparse below)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Args (patched – includes soft‑mask + RePaint flags)
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    # --- Generation -----------------------------------------------------------
    p.add_argument("--model_path",           required=True)
    p.add_argument("--input_path",           required=True)
    p.add_argument("--label_path",           required=True)
    p.add_argument("--prompt",               default="a photo of hta")
    p.add_argument("--negative_prompt",      default="")
    p.add_argument("--seed",                 type=int,   default=12345)
    p.add_argument("--num_inference_steps",  type=int,   default=50)
    p.add_argument("--guidance_scale",       type=float, default=7.5)
    p.add_argument("--batch_size",           type=int,   default=1)
    p.add_argument("--mixed_precision",      default="fp16", choices=["no", "fp16"])
    p.add_argument("--device",               default="cuda:0")
    p.add_argument("--resolution",           type=int,   default=512)
    p.add_argument("--enable_vae_slicing",   action="store_true", default=True)
    p.add_argument("--dfb_heads",            type=int,   default=None)
    p.add_argument("--dfb_inner_dim_factor", type=float, default=None)
    p.add_argument("--reuse_generated",      default=None)

    # --- Soft-mask / RePaint (NEW) --------------------------------------------
    p.add_argument("--soft_inner_blur",   type=int,   default=5)
    p.add_argument("--soft_outer_blur",   type=int,   default=5)
    p.add_argument("--use_soft_blend",    action="store_true", default=True)
    p.add_argument("--no_soft_blend",     action="store_true", default=False)
    p.add_argument("--use_repaint",       action="store_true", default=False)
    p.add_argument("--resample_steps",    type=int,   default=2)
    p.add_argument("--resample_jump",     type=int,   default=3)
    p.add_argument("--resample_warmup",   type=int,   default=10)

    # --- IP-Adapter (NEW: mirrors infer_dfb.py flags exactly) ----------------
    # If neither --ip_adapter_image nor --ip_adapter_image_dir is given, the
    # script runs the standard no-IP-Adapter path (byte-equivalent to v7).
    p.add_argument("--ip_adapter_image",       default=None,
                   help="Path to ONE global reference image used for ALL "
                        "generations. If both this and --ip_adapter_image_dir "
                        "are set, this single image wins.")
    p.add_argument("--ip_adapter_image_dir",   default=None,
                   help="Directory of reference images paired by sorted order "
                        "with --label_path masks (one ref per generation).")
    p.add_argument("--ip_adapter_repo",        default="h94/IP-Adapter",
                   help="HF repo id with IP-Adapter weights.")
    p.add_argument("--ip_adapter_subfolder",   default="models",
                   help="Subfolder inside the repo holding the weights.")
    p.add_argument("--ip_adapter_weight_name", default="ip-adapter_sd15.bin",
                   help="Weight file name (default: ip-adapter_sd15.bin).")
    p.add_argument("--ip_adapter_scale",       type=float, default=0.6,
                   help="IP-Adapter conditioning scale (0 = disabled).")
    p.add_argument("--offload_image_encoder",  action="store_true", default=False,
                   help="Move CLIP image encoder to CPU between generations "
                        "(saves ~1.2 GB VRAM at small re-encode cost).")

    # --- Reference data -------------------------------------------------------
    p.add_argument("--real_dir",       required=True)
    p.add_argument("--real_mask_dir",  default=None)
    p.add_argument("--train_data_dir", default=None)
    p.add_argument("--num_real",       type=int, default=300)

    # --- Encoders -------------------------------------------------------------
    p.add_argument("--dino_model",  default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--clip_model",  default="openai/clip-vit-base-patch32")
    p.add_argument("--use_raddino", action="store_true", default=False)
    p.add_argument("--embed_batch_size", type=int, default=8)

    # --- Eval knobs -----------------------------------------------------------
    p.add_argument("--bootstrap",      type=int, default=1000)
    p.add_argument("--run_segmentation_check", action="store_true", default=False)
    p.add_argument("--seg_epochs",     type=int, default=30)

    # --- Output ---------------------------------------------------------------
    p.add_argument("--out_dir",   default="comprehensive_validation")
    p.add_argument("--save_grid", action="store_true", default=True)
    return p.parse_args()


# =============================================================================
# Utilities (unchanged)
# =============================================================================
def get_filenames_from_dir(d: str) -> List[str]:
    return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))

def collect_images(directory: str, extensions=(".jpg", ".jpeg", ".png")) -> List[str]:
    return sorted(
        os.path.join(directory, fn)
        for fn in os.listdir(directory)
        if fn.lower().endswith(extensions)
    )

def to_uint8(img: Image.Image, size: int = 512) -> np.ndarray:
    return np.array(img.convert("RGB").resize((size, size), Image.LANCZOS))

def mask_to_bool(m: Image.Image, size: int = 512, thr: int = 127) -> np.ndarray:
    return (np.array(m.convert("L").resize((size, size), Image.NEAREST)) > thr)

def safe_div(a, b, eps=1e-12):
    return a / (b + eps)


# =============================================================================
# Bootstrap CI helper
# =============================================================================
def bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                 alpha: float = 0.05, agg=np.mean) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(agg(values))
    if n_boot <= 0 or len(values) < 2:
        return point, point, point
    rng = np.random.default_rng(0)
    n = len(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = agg(values[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi

def fmt_ci(point, lo, hi, decimals=4):
    return {"value":  round(point, decimals),
            "ci_low": round(lo,    decimals),
            "ci_high":round(hi,    decimals)}


# =============================================================================
# Generation (patched – RePaint + soft‑mask)
# =============================================================================
def generate_images(args, dtype, device):
    """Generate one image per mask using RePaint + soft-mask blending,
    optionally conditioned on an IP-Adapter reference image."""
    import os, sys, gc
    from PIL import Image
    from tqdm import tqdm
    from diffusers import (AutoencoderKL, DDIMScheduler, UNet2DConditionModel)
    from transformers import CLIPTextModel, CLIPTokenizer
    from typing import Optional, List

    for sd in [
        os.path.dirname(os.path.abspath(__file__)),
        "/kaggle/working/MedDiff-FT/main",
    ]:
        if os.path.exists(os.path.join(sd, "meddiff_dfb.py")):
            sys.path.insert(0, sd); break

    from meddiff_dfb import (restore_dfb_in_unet, load_dfb_state,
                              load_dfb_config, vram_cleanup)
    # Reuse the IP-Adapter helpers from infer_dfb.py rather than duplicating
    # them — guarantees the two scripts stay in lockstep.
    from infer_dfb import (repaint_softmask_generate,
                           _attach_ip_adapter, _resolve_ref_image)

    print(f"\n[Gen] Loading UNet from {args.model_path}/unet/ ...")
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet", torch_dtype=dtype)

    unet_dir = os.path.join(args.model_path, "unet")
    cfg   = load_dfb_config(unet_dir)
    heads = args.dfb_heads if args.dfb_heads is not None else cfg["heads"]
    idf   = (args.dfb_inner_dim_factor
             if args.dfb_inner_dim_factor is not None else cfg["inner_dim_factor"])

    dfb_state = load_dfb_state(unet_dir)
    if dfb_state:
        dfb_state = {k: v.to(dtype) for k, v in dfb_state.items()}
        restore_dfb_in_unet(unet, dfb_state, heads=heads, inner_dim_factor=idf)
        print(f"[Gen] DFB tensors loaded: {len(dfb_state)} "
              f"(heads={heads}, idf={idf})")
    else:
        print("[Gen] No dfb_weights found — running without DFB.")

    # IMPORTANT: move UNet to device BEFORE attaching the IP-Adapter, because
    # `load_ip_adapter` mutates the UNet's cross-attention processors in place
    # and we want the resulting modules to land directly on the GPU.
    unet = unet.to(dtype=dtype, device=device).eval()

    vae = AutoencoderKL.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_path, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(
        args.model_path, subfolder="scheduler")

    if args.enable_vae_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()

    use_soft_blend = args.use_soft_blend and not args.no_soft_blend

    # ---- IP-Adapter attach (NEW) --------------------------------------------
    # Only attach if the user actually supplied a reference source AND the
    # scale is > 0. This keeps behaviour byte-equivalent to v7 when no
    # IP-Adapter inputs are passed.
    want_ipa = (
        (getattr(args, "ip_adapter_image", None) is not None
         or getattr(args, "ip_adapter_image_dir", None) is not None)
        and getattr(args, "ip_adapter_scale", 0.0) > 0.0
    )
    ipa_pipe = None
    embed_fn = None
    global_ref_pil: Optional[Image.Image] = None
    ip_dir_files: Optional[List[str]] = None
    cached_global_embeds = None
    do_cfg_flag = args.guidance_scale > 1.0

    if want_ipa:
        ipa_pipe, embed_fn = _attach_ip_adapter(
            unet=unet, vae=vae, text_encoder=text_encoder,
            tokenizer=tokenizer, scheduler=scheduler,
            device=device, dtype=dtype,
            repo=args.ip_adapter_repo,
            subfolder=args.ip_adapter_subfolder,
            weight_name=args.ip_adapter_weight_name,
            scale=args.ip_adapter_scale,
        )
        # Resolve reference image source.
        if args.ip_adapter_image is not None:
            global_ref_pil = Image.open(args.ip_adapter_image).convert("RGB")
            print(f"[Gen] IP-Adapter ref (global): {args.ip_adapter_image}")
        elif args.ip_adapter_image_dir is not None:
            ip_dir_files = sorted(
                f for f in os.listdir(args.ip_adapter_image_dir)
                if os.path.isfile(os.path.join(args.ip_adapter_image_dir, f))
            )
            print(f"[Gen] IP-Adapter ref dir: {args.ip_adapter_image_dir} "
                  f"({len(ip_dir_files)} images, paired by sorted order)")

        # One global ref → encode once, reuse everywhere.
        if global_ref_pil is not None:
            if args.offload_image_encoder and ipa_pipe.image_encoder is not None:
                ipa_pipe.image_encoder.to(device)
            cached_global_embeds = embed_fn(global_ref_pil, do_cfg_flag)
            if args.offload_image_encoder and ipa_pipe.image_encoder is not None:
                ipa_pipe.image_encoder.to("cpu")
                torch.cuda.empty_cache()
            print(f"[Gen] Cached IP-Adapter embeds: "
                  f"{[tuple(e.shape) for e in cached_global_embeds]}")
    else:
        print("[Gen] IP-Adapter disabled (no ref image supplied or scale=0). "
              "Running standard v7 inference path.")

    inputs = sorted(
        f for f in os.listdir(args.input_path)
        if os.path.isfile(os.path.join(args.input_path, f))
    )
    masks = sorted(
        f for f in os.listdir(args.label_path)
        if os.path.isfile(os.path.join(args.label_path, f))
    )
    print(f"[Gen] {len(inputs)} bg images, {len(masks)} masks")
    print(f"[Gen] soft_blend={use_soft_blend}  "
          f"repaint={args.use_repaint}  "
          f"steps={args.num_inference_steps}  cfg={args.guidance_scale}  "
          f"ip_adapter={'on' if want_ipa else 'off'}"
          + (f" (scale={args.ip_adapter_scale})" if want_ipa else ""))

    generated, used_bg, used_masks = [], [], []
    idx = 0

    for mi, mf in enumerate(tqdm(masks, desc="Generating")):
        ifn = inputs[idx % len(inputs)]; idx += 1
        init = Image.open(os.path.join(args.input_path, ifn)).convert("RGB")
        msk  = Image.open(os.path.join(args.label_path, mf)).convert("L")
        used_bg.append(ifn); used_masks.append(mf)

        # ---- per-generation IP-Adapter embeds -------------------------------
        ip_embeds = None
        if want_ipa:
            if cached_global_embeds is not None:
                ip_embeds = cached_global_embeds
            else:
                ref_pil = _resolve_ref_image(
                    args, mf, None, ip_dir_files, mi
                )
                if ref_pil is not None:
                    if args.offload_image_encoder and ipa_pipe.image_encoder is not None:
                        ipa_pipe.image_encoder.to(device)
                    ip_embeds = embed_fn(ref_pil, do_cfg_flag)
                    if args.offload_image_encoder and ipa_pipe.image_encoder is not None:
                        ipa_pipe.image_encoder.to("cpu")
                        torch.cuda.empty_cache()

        out = repaint_softmask_generate(
            unet=unet, vae=vae, text_encoder=text_encoder,
            tokenizer=tokenizer, scheduler=scheduler,
            init_pil=init, binary_mask_pil=msk,
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            resolution=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            device=device, dtype=dtype,
            seed=args.seed + mi,
            soft_inner_blur=args.soft_inner_blur,
            soft_outer_blur=args.soft_outer_blur,
            use_soft_blend=use_soft_blend,
            use_repaint=args.use_repaint,
            resample_steps=args.resample_steps,
            resample_jump=args.resample_jump,
            resample_warmup=args.resample_warmup,
            ip_image_embeds=ip_embeds,
        )
        if out.size != init.size:
            out = out.resize(init.size, Image.LANCZOS)
        generated.append(out)

    # ---- Cleanup ------------------------------------------------------------
    if ipa_pipe is not None:
        # Drop the wrapper pipe BEFORE deleting the shared modules so we don't
        # leave dangling references to unet/vae/text_encoder.
        try:
            del ipa_pipe.image_encoder
        except Exception:
            pass
        del ipa_pipe
    del unet, vae, text_encoder
    gc.collect(); vram_cleanup()
    print(f"[Gen] Generated {len(generated)} images.\n")
    return generated, used_bg, used_masks


# =============================================================================
# Encoders (unchanged)
# =============================================================================
class HFFeatureExtractor:
    def __init__(self, model_name: str, device, dtype, label="HF"):
        from transformers import AutoImageProcessor, AutoModel
        print(f"[{label}] Loading {model_name} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = (AutoModel.from_pretrained(model_name)
                      .to(device).eval().requires_grad_(False))
        self.device, self.dtype, self.label = device, dtype, label

    @torch.inference_mode()
    def embed(self, images: List[Image.Image], batch_size: int = 8) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(images), batch_size),
                      desc=f"{self.label} embed"):
            inp = self.processor(images=images[i:i+batch_size], return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            ac  = self.device.type if self.device.type != "mps" else "cpu"
            with torch.autocast(ac, dtype=self.dtype,
                                enabled=(self.dtype == torch.float16)):
                if hasattr(self.model, 'get_image_features'):
                    o = self.model.vision_model(**inp)
                else:
                    o = self.model(**inp)
            emb = o.pooler_output if (hasattr(o, "pooler_output")
                                      and o.pooler_output is not None) \
                  else o.last_hidden_state[:, 0, :]
            out.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, 0)


class InceptionFID:
    def __init__(self, device):
        from torchvision.models import inception_v3, Inception_V3_Weights
        from torchvision import transforms
        weights = Inception_V3_Weights.IMAGENET1K_V1
        m = inception_v3(weights=weights, aux_logits=True).to(device).eval()
        m.fc = torch.nn.Identity()
        for p in m.parameters(): p.requires_grad_(False)
        self.model, self.device = m, device
        self.tf = transforms.Compose([
            transforms.Resize(299, antialias=True),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def embed(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(images), batch_size), desc="Inception embed"):
            x = torch.stack([self.tf(img.convert("RGB"))
                             for img in images[i:i+batch_size]]).to(self.device)
            f = self.model(x).float().cpu().numpy()
            out.append(f)
        return np.concatenate(out, 0)


# =============================================================================
# Distributional metrics (unchanged)
# =============================================================================
def frechet_distance(real: np.ndarray, gen: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sig_r = np.cov(real, rowvar=False)
    sig_g = np.cov(gen,  rowvar=False)
    diff  = mu_r - mu_g
    sq, _ = sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(sq): sq = sq.real
    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * sq))

def polynomial_mmd(X: np.ndarray, Y: np.ndarray,
                   degree=3, gamma=None, coef0=1.0) -> float:
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K_XX = (gamma * (X @ X.T) + coef0) ** degree
    K_YY = (gamma * (Y @ Y.T) + coef0) ** degree
    K_XY = (gamma * (X @ Y.T) + coef0) ** degree
    m, n = K_XX.shape[0], K_YY.shape[0]
    np.fill_diagonal(K_XX, 0); np.fill_diagonal(K_YY, 0)
    return float(K_XX.sum() / (m * (m - 1))
                 + K_YY.sum() / (n * (n - 1))
                 - 2 * K_XY.mean())

def kid_score(real: np.ndarray, gen: np.ndarray,
              n_subsets: int = 100, subset_size: int = 100,
              seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    subset_size = min(subset_size, len(real), len(gen))
    if subset_size < 10:
        return float("nan"), float("nan")
    scores = []
    for _ in range(n_subsets):
        ir = rng.choice(len(real), subset_size, replace=False)
        ig = rng.choice(len(gen),  subset_size, replace=False)
        scores.append(polynomial_mmd(real[ir], gen[ig]))
    s = np.array(scores)
    return float(s.mean()), float(s.std())

def precision_recall(real: np.ndarray, gen: np.ndarray, k: int = 3) -> Tuple[float, float]:
    def kth_sim(m, k):
        s = m @ m.T
        np.fill_diagonal(s, -2.0)
        return np.sort(s, axis=1)[:, -k]
    k = max(1, min(k, len(real) - 1, len(gen) - 1))
    rr = kth_sim(real, k); gg = kth_sim(gen, k)
    sim_gr = gen  @ real.T
    sim_rg = real @ gen.T
    prec = float(np.mean(sim_gr.max(1) >= np.percentile(rr, 5)))
    rec  = float(np.mean(sim_rg.max(1) >= np.percentile(gg, 5)))
    return prec, rec


# =============================================================================
# Diversity / Vendi
# =============================================================================
def vendi_score(features: np.ndarray) -> float:
    if len(features) < 2:
        return 1.0
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    K = f @ f.T
    K = (K + K.T) / 2
    K = K / len(features)
    w = np.linalg.eigvalsh(K)
    w = np.clip(w, 1e-12, None)
    H = -(w * np.log(w)).sum()
    return float(np.exp(H))

def mean_pairwise_cosine_distance(features: np.ndarray) -> float:
    if len(features) < 2: return 0.0
    sim = features @ features.T
    n = len(features)
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sim[iu].mean())


# =============================================================================
# Region-conditioned paired metrics (unchanged)
# =============================================================================
def _to_torch_01(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_u8).permute(2, 0, 1).float() / 255.0

def paired_image_metrics(gen_u8: np.ndarray, ref_u8: np.ndarray,
                         mask_bool: Optional[np.ndarray] = None,
                         lpips_fn=None) -> Dict[str, float]:
    from skimage.metrics import structural_similarity as ssim_fn
    g, r = gen_u8.astype(np.float32) / 255.0, ref_u8.astype(np.float32) / 255.0

    if mask_bool is not None:
        m3 = np.repeat(mask_bool[..., None], 3, axis=2)
        if not m3.any():
            return {}
        diff = (g - r)
        mae = float(np.abs(diff[m3]).mean())
        mse = float((diff[m3] ** 2).mean())
        psnr = float(20 * np.log10(1.0 / (np.sqrt(mse) + 1e-12)))
        ys, xs = np.where(mask_bool)
        if len(ys) < 8 or len(xs) < 8:
            return {"mae": mae, "mse": mse, "psnr": psnr}
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        gc, rc = g[y0:y1, x0:x1], r[y0:y1, x0:x1]
        win = min(7, gc.shape[0] - 1, gc.shape[1] - 1)
        if win % 2 == 0: win -= 1
        if win < 3:
            ssim_val = float("nan")
        else:
            ssim_val = float(ssim_fn(rc, gc, channel_axis=-1, data_range=1.0,
                                     win_size=win))
        out = {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim_val}
    else:
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        diff = g - r
        mae = float(np.abs(diff).mean())
        mse = float((diff ** 2).mean())
        psnr = float(psnr_fn(r, g, data_range=1.0))
        ssim_val = float(ssim_fn(r, g, channel_axis=-1, data_range=1.0))
        out = {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim_val}
        
    if lpips_fn is not None:
        try:
            if mask_bool is None:
                t_g = _to_torch_01(gen_u8) * 2 - 1
                t_r = _to_torch_01(ref_u8) * 2 - 1
            else:
                ys, xs = np.where(mask_bool)
                if len(ys) == 0:
                    return out
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                crop_h, crop_w = y1 - y0, x1 - x0
                # LPIPS needs at least 16 pixels in each dimension
                if crop_h < 16 or crop_w < 16:
                    return out
                t_g = _to_torch_01(gen_u8[y0:y1, x0:x1]) * 2 - 1
                t_r = _to_torch_01(ref_u8[y0:y1, x0:x1]) * 2 - 1
            with torch.no_grad():
                d = lpips_fn(t_g.unsqueeze(0).to(lpips_fn._device),
                             t_r.unsqueeze(0).to(lpips_fn._device))
            out["lpips"] = float(d.item())
        except RuntimeError:
            # skip LPIPS if crop is too small for backbone
            pass
    return out


# =============================================================================
# Mask-fidelity / leakage (unchanged)
# =============================================================================
def mask_fidelity_metrics(gen_u8: np.ndarray, init_u8: np.ndarray,
                          mask_bool: np.ndarray) -> Dict[str, float]:
    g, i = gen_u8.astype(np.float32) / 255.0, init_u8.astype(np.float32) / 255.0
    inside_diff  = np.abs(g[ mask_bool] - i[ mask_bool]).mean() if mask_bool.any() else 0.0
    outside_diff = np.abs(g[~mask_bool] - i[~mask_bool]).mean() if (~mask_bool).any() else 0.0
    leakage_ratio = float(safe_div(inside_diff, outside_diff))
    from scipy.ndimage import binary_dilation, binary_erosion, sobel
    edge = binary_dilation(mask_bool, iterations=2) ^ binary_erosion(mask_bool, iterations=2)
    if edge.any():
        gx = sobel(g.mean(-1), axis=0); gy = sobel(g.mean(-1), axis=1)
        edge_grad = float(np.sqrt(gx**2 + gy**2)[edge].mean())
    else:
        edge_grad = float("nan")
    return {
        "outside_mae":   float(outside_diff),
        "inside_mae":    float(inside_diff),
        "leakage_ratio": leakage_ratio,
        "edge_gradient": edge_grad,
    }


# =============================================================================
# Texture / frequency / wavelet (unchanged)
# =============================================================================
def radial_power_spectrum(gray: np.ndarray, n_bands: int = 3) -> np.ndarray:
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray.astype(np.float32))) ** 2)
    h, w = f.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = r.max()
    edges = np.linspace(0, r_max, n_bands + 1)
    band_power = []
    for i in range(n_bands):
        m = (r >= edges[i]) & (r < edges[i + 1])
        band_power.append(float(f[m].mean()) if m.any() else 0.0)
    total = sum(band_power) + 1e-12
    return np.array([p / total for p in band_power])

def wavelet_subband_energy(gray: np.ndarray) -> Dict[str, float]:
    g = gray.astype(np.float32) / 255.0
    h, w = g.shape
    if h % 2: g = g[:-1]
    if w % 2: g = g[:, :-1]
    a = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2]) / 4
    b = (g[0::2, 0::2] + g[0::2, 1::2] - g[1::2, 0::2] - g[1::2, 1::2]) / 4
    c = (g[0::2, 0::2] - g[0::2, 1::2] + g[1::2, 0::2] - g[1::2, 1::2]) / 4
    d = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 4
    e = lambda x: float((x ** 2).mean())
    ll, lh, hl, hh = e(a), e(b), e(c), e(d)
    s = ll + lh + hl + hh + 1e-12
    return {"ll_ratio": ll/s, "lh_ratio": lh/s, "hl_ratio": hl/s, "hh_ratio": hh/s}

def glcm_features(gray_u8: np.ndarray) -> Dict[str, float]:
    from skimage.feature import graycomatrix, graycoprops
    if gray_u8.dtype != np.uint8:
        gray_u8 = gray_u8.astype(np.uint8)
    g = (gray_u8 // 16).astype(np.uint8)
    glcm = graycomatrix(g, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=16, symmetric=True, normed=True)
    out = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        out[prop] = float(graycoprops(glcm, prop).mean())
    return out


# =============================================================================
# Color / dermoscopy (unchanged)
# =============================================================================
def color_histogram_intersection(a_u8: np.ndarray, b_u8: np.ndarray,
                                 bins: int = 32) -> Dict[str, float]:
    out = {}
    for c, name in enumerate("rgb"):
        ha, _ = np.histogram(a_u8[..., c], bins=bins, range=(0, 255), density=True)
        hb, _ = np.histogram(b_u8[..., c], bins=bins, range=(0, 255), density=True)
        out[f"hist_inter_{name}"] = float(np.minimum(ha, hb).sum() / (ha.sum() + 1e-12))
    return out

def hsv_color_stats(a_u8: np.ndarray) -> Dict[str, float]:
    import colorsys
    arr = a_u8.reshape(-1, 3) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in arr[::32]])
    return {"hue_mean": float(hsv[:, 0].mean()),
            "sat_mean": float(hsv[:, 1].mean()),
            "val_mean": float(hsv[:, 2].mean()),
            "sat_std":  float(hsv[:, 1].std())}

def wasserstein_color(a_u8: np.ndarray, b_u8: np.ndarray) -> Dict[str, float]:
    from scipy.stats import wasserstein_distance
    out = {}
    for c, name in enumerate("rgb"):
        out[f"wasserstein_{name}"] = float(wasserstein_distance(
            a_u8[..., c].ravel()[::64], b_u8[..., c].ravel()[::64]))
    return out


# =============================================================================
# Memorization / copy detection (unchanged)
# =============================================================================
def memorization_check(train_imgs: List[Image.Image],
                       gen_imgs: List[Image.Image],
                       extractors: Dict[str, "HFFeatureExtractor"],
                       batch_size: int = 8) -> Dict[str, float]:
    if not train_imgs or not gen_imgs:
        return {}
    res = {}
    for name, ext in extractors.items():
        train_e = ext.embed(train_imgs, batch_size=batch_size)
        gen_e   = ext.embed(gen_imgs,   batch_size=batch_size)
        sim = gen_e @ train_e.T
        top1 = sim.max(1)
        topk = np.sort(sim, axis=1)[:, -5:].mean(1)
        res[f"{name}_top1_train_cos_mean"]   = float(top1.mean())
        res[f"{name}_top1_train_cos_max"]    = float(top1.max())
        res[f"{name}_top5_train_cos_mean"]   = float(topk.mean())
        res[f"{name}_flag_rate_top1>0.95"]   = float((top1 > 0.95).mean())
    from skimage.metrics import structural_similarity as ssim_fn
    train_u8 = [to_uint8(im, 256) for im in train_imgs]
    gen_u8   = [to_uint8(im, 256) for im in gen_imgs]
    ssims = []
    for g in tqdm(gen_u8, desc="SSIM-vs-train"):
        s = max(ssim_fn(g, t, channel_axis=-1, data_range=255) for t in train_u8)
        ssims.append(s)
    ssims = np.array(ssims)
    res["ssim_top1_train_mean"]      = float(ssims.mean())
    res["ssim_top1_train_max"]       = float(ssims.max())
    res["ssim_flag_rate_top1>0.92"]  = float((ssims > 0.92).mean())
    return res


# =============================================================================
# Downstream segmentation utility (unchanged)
# =============================================================================
def downstream_segmentation_check(real_image_paths: List[str],
                                  real_mask_paths: List[str],
                                  gen_imgs: List[Image.Image],
                                  guide_masks: List[Image.Image],
                                  device: torch.device,
                                  epochs: int = 30,
                                  size: int = 256) -> Dict[str, float]:
    if not real_image_paths or not real_mask_paths:
        return {"status": "skipped — no real pairs provided"}

    from torch.utils.data import Dataset, DataLoader

    class MicroUNet(torch.nn.Module):
        def __init__(self, c=16):
            super().__init__()
            def cb(i, o): return torch.nn.Sequential(
                torch.nn.Conv2d(i, o, 3, padding=1), torch.nn.BatchNorm2d(o),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(o, o, 3, padding=1), torch.nn.BatchNorm2d(o),
                torch.nn.ReLU(inplace=True))
            self.d1, self.d2, self.d3 = cb(3, c), cb(c, c*2), cb(c*2, c*4)
            self.bot = cb(c*4, c*8)
            self.u3, self.u2, self.u1 = cb(c*8 + c*4, c*4), cb(c*4 + c*2, c*2), cb(c*2 + c, c)
            self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.pool = torch.nn.MaxPool2d(2)
            self.out = torch.nn.Conv2d(c, 1, 1)
        def forward(self, x):
            d1 = self.d1(x);  d2 = self.d2(self.pool(d1))
            d3 = self.d3(self.pool(d2)); b = self.bot(self.pool(d3))
            u3 = self.u3(torch.cat([self.up(b),  d3], 1))
            u2 = self.u2(torch.cat([self.up(u3), d2], 1))
            u1 = self.u1(torch.cat([self.up(u2), d1], 1))
            return self.out(u1)

    class PairDS(Dataset):
        def __init__(self, ips, mps, sz):
            self.ips, self.mps, self.sz = ips, mps, sz
        def __len__(self): return len(self.ips)
        def __getitem__(self, i):
            im = np.array(Image.open(self.ips[i]).convert("RGB").resize((self.sz, self.sz)))
            mk = np.array(Image.open(self.mps[i]).convert("L").resize((self.sz, self.sz)))
            return (torch.from_numpy(im).permute(2,0,1).float()/255.0,
                    torch.from_numpy((mk > 127).astype(np.float32)).unsqueeze(0))

    pair_idx = list(range(min(len(real_image_paths), len(real_mask_paths))))
    random.shuffle(pair_idx)
    pair_idx = pair_idx[:200]
    ips = [real_image_paths[i] for i in pair_idx]
    mps = [real_mask_paths[i]  for i in pair_idx]
    n_val = max(1, len(ips) // 5)
    ds_train = PairDS(ips[n_val:], mps[n_val:], size)
    ds_val   = PairDS(ips[:n_val], mps[:n_val], size)
    dl_train = DataLoader(ds_train, batch_size=8, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=8, shuffle=False, num_workers=0)

    model = MicroUNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=2e-4)
    bce   = torch.nn.BCEWithLogitsLoss()

    def dice(p, g, eps=1e-6):
        p = (torch.sigmoid(p) > 0.5).float()
        inter = (p * g).sum((1,2,3))
        return ((2*inter + eps) / (p.sum((1,2,3)) + g.sum((1,2,3)) + eps)).mean().item()

    print(f"[Seg] Training tiny U-Net on {len(ds_train)} real pairs ({epochs} ep)")
    for ep in range(epochs):
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); l = bce(model(x), y); l.backward(); opt.step()

    model.eval()
    val_d = []
    with torch.no_grad():
        for x, y in dl_val:
            x, y = x.to(device), y.to(device)
            val_d.append(dice(model(x), y))
    real_dice = float(np.mean(val_d)) if val_d else float("nan")

    gen_dice, gen_iou = [], []
    with torch.no_grad():
        for img, mk in zip(gen_imgs, guide_masks):
            x = (torch.from_numpy(np.array(img.convert("RGB").resize((size, size))))
                   .permute(2,0,1).float()/255.0).unsqueeze(0).to(device)
            g = torch.from_numpy((np.array(mk.convert("L").resize((size, size))) > 127)
                   .astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            p = (torch.sigmoid(model(x)) > 0.5).float()
            inter = (p * g).sum().item(); union = (p + g - p*g).sum().item()
            gp_sum, g_sum = p.sum().item(), g.sum().item()
            gen_dice.append((2*inter + 1e-6)/(gp_sum + g_sum + 1e-6))
            gen_iou .append((  inter + 1e-6)/(union  + 1e-6))
    return {
        "tiny_unet_val_dice_real":  real_dice,
        "tiny_unet_dice_gen_vs_guide": float(np.mean(gen_dice)),
        "tiny_unet_iou_gen_vs_guide":  float(np.mean(gen_iou)),
        "alignment_drop_real_to_gen":  float(real_dice - np.mean(gen_dice)),
    }


# =============================================================================
# Main (unchanged)
# =============================================================================
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype  = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    print("=" * 75)
    print("  MedDiff-FT + DFB  ·  Comprehensive Evaluation Suite  (v7 patched)")
    print("=" * 75)
    print(f"  Checkpoint     : {args.model_path}")
    print(f"  Bootstrap CIs  : {args.bootstrap} resamples")
    print(f"  Encoders       : Inception | DINOv3 | CLIP" +
          (" | RadDINO" if args.use_raddino else ""))
    print(f"  Seg utility    : {'ON' if args.run_segmentation_check else 'OFF'}")
    print("=" * 75 + "\n")

    # 1. Generation -----------------------------------------------------------
    if args.reuse_generated and os.path.isdir(args.reuse_generated):
        gen_paths = collect_images(args.reuse_generated)
        gen_images = [Image.open(p).convert("RGB") for p in gen_paths]
        bg_files   = get_filenames_from_dir(args.input_path)
        msk_files  = get_filenames_from_dir(args.label_path)
        used_bg    = [bg_files[i % len(bg_files)] for i in range(len(gen_images))]
        used_masks = msk_files[:len(gen_images)]
        print(f"[Reuse] Loaded {len(gen_images)} pre-generated images.\n")
    else:
        gen_images, used_bg, used_masks = generate_images(args, dtype, device)
        gen_dir = os.path.join(args.out_dir, "generated")
        os.makedirs(gen_dir, exist_ok=True)
        for i, img in enumerate(gen_images):
            img.save(os.path.join(gen_dir, f"gen_{i:04d}.png"))
        print(f"[Gen] Saved {len(gen_images)} images to {gen_dir}\n")

    bg_pil   = [Image.open(os.path.join(args.input_path, fn)).convert("RGB") for fn in used_bg]
    mask_pil = [Image.open(os.path.join(args.label_path, fn)).convert("L")  for fn in used_masks]

    # 2. Real reference -------------------------------------------------------
    real_paths = collect_images(args.real_dir)
    if not real_paths:
        raise RuntimeError(f"No images in --real_dir: {args.real_dir}")
    rng = random.Random(args.seed)
    real_paths = rng.sample(real_paths, min(args.num_real, len(real_paths)))
    real_images = [Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)
                   for p in tqdm(real_paths, desc="Loading real")]

    real_mask_pil = []
    real_mask_paths_aligned = []
    if args.real_mask_dir:
        for p in real_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            mp = os.path.join(args.real_mask_dir, f"{stem}_segmentation.png")
            if os.path.exists(mp):
                real_mask_pil.append(Image.open(mp).convert("L"))
                real_mask_paths_aligned.append(mp)
            else:
                real_mask_pil.append(None)
                real_mask_paths_aligned.append(None)
        n_with = sum(1 for m in real_mask_pil if m is not None)
        print(f"[Real] Found masks for {n_with}/{len(real_images)} real images.")

    # 3. Encoders -------------------------------------------------------------
    print("\n[Encoders] Building feature extractors ...")
    inception = InceptionFID(device)
    dino      = HFFeatureExtractor(args.dino_model, device, dtype, label="DINOv3")
    clip_ext  = HFFeatureExtractor(args.clip_model, device, dtype, label="CLIP")
    raddino   = None
    if args.use_raddino:
        try:
            raddino = HFFeatureExtractor("microsoft/rad-dino", device, dtype,
                                         label="RadDINO")
        except Exception as e:
            print(f"[RadDINO] disabled — {e}")
            raddino = None

    print("\n[Embed] Real ...")
    inc_r  = inception.embed(real_images, args.embed_batch_size)
    dino_r = dino    .embed(real_images, args.embed_batch_size)
    clip_r = clip_ext.embed(real_images, args.embed_batch_size)
    rad_r  = raddino .embed(real_images, args.embed_batch_size) if raddino else None

    print("\n[Embed] Generated ...")
    inc_g  = inception.embed(gen_images, args.embed_batch_size)
    dino_g = dino    .embed(gen_images, args.embed_batch_size)
    clip_g = clip_ext.embed(gen_images, args.embed_batch_size)
    rad_g  = raddino .embed(gen_images, args.embed_batch_size) if raddino else None

    # 4. Distributional fidelity ----------------------------------------------
    print("\n[Metrics] Distributional fidelity ...")
    dist_metrics = {}
    dist_metrics["FID_inception"]   = round(frechet_distance(inc_r,  inc_g),  4)
    dist_metrics["FDD_dinov3"]      = round(frechet_distance(dino_r, dino_g), 4)
    dist_metrics["FCD_clip"]        = round(frechet_distance(clip_r, clip_g), 4)
    if rad_r is not None:
        dist_metrics["F_RadDINO"]   = round(frechet_distance(rad_r, rad_g),   4)

    kid_mean, kid_std = kid_score(inc_r, inc_g, n_subsets=100,
                                  subset_size=min(100, len(inc_r), len(inc_g)))
    dist_metrics["KID_inception_mean"] = round(kid_mean, 6)
    dist_metrics["KID_inception_std"]  = round(kid_std,  6)

    sim_dino_cross = (dino_g @ dino_r.T)
    per_img_best   = sim_dino_cross.max(1)
    pe, lo, hi = bootstrap_ci(per_img_best, args.bootstrap)
    dist_metrics["per_image_best_dino_sim"] = fmt_ci(pe, lo, hi)

    p, r = precision_recall(dino_r, dino_g)
    dist_metrics["precision_dino"] = round(p, 4)
    dist_metrics["recall_dino"]    = round(r, 4)

    # 5. Diversity ------------------------------------------------------------
    print("[Metrics] Diversity ...")
    div_metrics = {
        "vendi_dinov3":       round(vendi_score(dino_g),   4),
        "vendi_clip":         round(vendi_score(clip_g),   4),
        "vendi_inception":    round(vendi_score(inc_g),    4),
        "vendi_real_dinov3":  round(vendi_score(dino_r),   4),
        "mean_pairwise_cos_dist_dinov3": round(mean_pairwise_cosine_distance(dino_g), 4),
        "mean_pairwise_cos_dist_clip":   round(mean_pairwise_cosine_distance(clip_g), 4),
    }

    # 6. Memorization ---------------------------------------------------------
    mem_metrics = {}
    if args.train_data_dir:
        train_img_dir = os.path.join(args.train_data_dir, "images")
        if os.path.isdir(train_img_dir):
            train_imgs = [Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)
                          for p in collect_images(train_img_dir)]
            print(f"\n[Mem] Training set: {len(train_imgs)} images")
            mem_metrics = memorization_check(
                train_imgs, gen_images,
                {"dinov3": dino, "clip": clip_ext},
                batch_size=args.embed_batch_size)
        else:
            print(f"\n[Mem] Skipped — {train_img_dir} not found.")

    # 7. Paired structural metrics --------------------------------------------
    print("\n[Metrics] Paired structural / perceptual (full / bg / lesion) ...")
    try:
        import lpips as _lpips_pkg
        lpips_fn = _lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()
        lpips_fn._device = device
    except Exception as e:
        print(f"[LPIPS] not available — {e}")
        lpips_fn = None

    full_acc, bg_acc, les_acc, mask_fid_acc = [], [], [], []
    for g_pil, init_pil, m_pil in tqdm(list(zip(gen_images, bg_pil, mask_pil)),
                                       total=len(gen_images), desc="Paired metrics"):
        g_u8 = to_uint8(g_pil);  i_u8 = to_uint8(init_pil)
        m    = mask_to_bool(m_pil)
        full = paired_image_metrics(g_u8, i_u8, mask_bool=None,    lpips_fn=lpips_fn)
        bgm  = paired_image_metrics(g_u8, i_u8, mask_bool=~m,      lpips_fn=lpips_fn)
        lesm = paired_image_metrics(g_u8, i_u8, mask_bool=m,       lpips_fn=lpips_fn)
        full_acc.append(full); bg_acc.append(bgm); les_acc.append(lesm)
        mask_fid_acc.append(mask_fidelity_metrics(g_u8, i_u8, m))

    def _agg(acc, n_boot):
        if not acc: return {}
        keys = set().union(*[d.keys() for d in acc])
        out = {}
        for k in keys:
            vals = np.array([d[k] for d in acc if k in d and not math.isnan(d[k])])
            if len(vals) == 0: continue
            pe, lo, hi = bootstrap_ci(vals, n_boot)
            out[k] = fmt_ci(pe, lo, hi)
        return out

    paired_metrics = {
        "full_image":        _agg(full_acc, args.bootstrap),
        "background_only":   _agg(bg_acc,   args.bootstrap),
        "lesion_only":       _agg(les_acc,  args.bootstrap),
        "mask_fidelity":     _agg(mask_fid_acc, args.bootstrap),
    }

    # 8. Texture / frequency / wavelet ----------------------------------------
    print("[Metrics] Texture / frequency ...")
    def _gray(u8): return (u8.astype(np.float32).mean(-1)).astype(np.uint8)
    def _crop_bbox(u8, m):
        ys, xs = np.where(m)
        if len(ys) < 8 or len(xs) < 8: return None
        y0,y1,x0,x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        return u8[y0:y1, x0:x1]

    real_glcm, gen_glcm = [], []
    real_wave, gen_wave = [], []
    real_pow,  gen_pow  = [], []
    for g_pil, m_pil in zip(gen_images, mask_pil):
        u8 = to_uint8(g_pil); m = mask_to_bool(m_pil)
        crop = _crop_bbox(u8, m)
        if crop is None: continue
        gen_glcm.append(glcm_features(_gray(crop)))
        gen_wave.append(wavelet_subband_energy(_gray(crop)))
        gen_pow .append(radial_power_spectrum(_gray(crop)))

    if real_mask_pil and any(m is not None for m in real_mask_pil):
        for r_im, r_mk in zip(real_images, real_mask_pil):
            if r_mk is None: continue
            u8 = to_uint8(r_im); m = mask_to_bool(r_mk)
            crop = _crop_bbox(u8, m)
            if crop is None: continue
            real_glcm.append(glcm_features(_gray(crop)))
            real_wave.append(wavelet_subband_energy(_gray(crop)))
            real_pow .append(radial_power_spectrum(_gray(crop)))
    else:
        for r_im in real_images:
            u8 = to_uint8(r_im)
            real_glcm.append(glcm_features(_gray(u8)))
            real_wave.append(wavelet_subband_energy(_gray(u8)))
            real_pow .append(radial_power_spectrum(_gray(u8)))

    def _avg(lst, key=None):
        if not lst: return float("nan")
        if key is None: return float(np.mean(lst))
        return float(np.mean([d[key] for d in lst]))

    texture_metrics = {
        "real_glcm":  {k: round(_avg(real_glcm, k), 4) for k in
                       ["contrast","homogeneity","energy","correlation"]},
        "gen_glcm":   {k: round(_avg(gen_glcm,  k), 4) for k in
                       ["contrast","homogeneity","energy","correlation"]},
        "real_wavelet": {k: round(_avg(real_wave, k), 4) for k in
                         ["ll_ratio","lh_ratio","hl_ratio","hh_ratio"]},
        "gen_wavelet":  {k: round(_avg(gen_wave,  k), 4) for k in
                         ["ll_ratio","lh_ratio","hl_ratio","hh_ratio"]},
        "real_power_band_lo_mid_hi": [round(float(np.mean([p[i] for p in real_pow])), 4)
                                      for i in range(3)] if real_pow else [],
        "gen_power_band_lo_mid_hi":  [round(float(np.mean([p[i] for p in gen_pow])),  4)
                                      for i in range(3)] if gen_pow  else [],
    }
    texture_metrics["wavelet_l1_delta"] = round(float(sum(
        abs(texture_metrics["real_wavelet"][k] - texture_metrics["gen_wavelet"][k])
        for k in texture_metrics["real_wavelet"])), 4)
    if real_pow and gen_pow:
        texture_metrics["power_band_l1_delta"] = round(float(sum(
            abs(a - b) for a, b in zip(texture_metrics["real_power_band_lo_mid_hi"],
                                       texture_metrics["gen_power_band_lo_mid_hi"]))), 4)

    # 9. Color / dermoscopy --------------------------------------------------
    print("[Metrics] Color / dermoscopy ...")
    real_arr = np.stack([to_uint8(r) for r in real_images])
    gen_arr  = np.stack([to_uint8(g) for g in gen_images])
    color_metrics = {}
    color_metrics.update(color_histogram_intersection(real_arr, gen_arr))
    color_metrics.update({f"real_{k}": v for k, v in hsv_color_stats(real_arr).items()})
    color_metrics.update({f"gen_{k}":  v for k, v in hsv_color_stats(gen_arr).items()})
    color_metrics.update(wasserstein_color(real_arr, gen_arr))
    color_metrics = {k: round(v, 4) for k, v in color_metrics.items()}

    # 10. Downstream segmentation utility ------------------------------------
    seg_metrics = {}
    if args.run_segmentation_check and args.real_mask_dir:
        ips = [p for p in real_paths]
        mps = []
        for p in ips:
            stem = os.path.splitext(os.path.basename(p))[0]
            mp = os.path.join(args.real_mask_dir, f"{stem}_segmentation.png")
            mps.append(mp if os.path.exists(mp) else None)
        ips = [i for i, m in zip(ips, mps) if m is not None]
        mps = [m for m in mps if m is not None]
        if len(ips) >= 30:
            print(f"\n[Seg] Running tiny U-Net utility check ({len(ips)} pairs)")
            seg_metrics = downstream_segmentation_check(
                ips, mps, gen_images, mask_pil, device,
                epochs=args.seg_epochs)
        else:
            seg_metrics = {"status": f"skipped — only {len(ips)} real pairs"}

    # 11. Assemble report ----------------------------------------------------
    quality_grade = _grade(dist_metrics.get("FDD_dinov3", 999),
                           dist_metrics["per_image_best_dino_sim"]["value"])

    report = {
        "args": vars(args),
        "summary": {
            "num_generated":   len(gen_images),
            "num_real":        len(real_images),
            "quality_grade":   quality_grade,
        },
        "1_distributional_fidelity":  dist_metrics,
        "2_paired_structural":        paired_metrics,
        "3_diversity":                div_metrics,
        "4_memorization":             mem_metrics,
        "5_texture_frequency":        texture_metrics,
        "6_color_dermoscopy":         color_metrics,
        "7_downstream_segmentation":  seg_metrics,
    }

    rp = os.path.join(args.out_dir, "comprehensive_report.json")
    with open(rp, "w") as f: json.dump(report, f, indent=2)
    print(f"\n[Report] Saved → {rp}")

    _print_summary(report)

    if args.save_grid:
        save_comparison_grid(real_images, gen_images,
                             list(per_img_best),
                             os.path.join(args.out_dir, "comparison_grid.png"))

    print("\n[Done] Comprehensive validation complete.")
    return report


def _grade(fdd, ms):
    if fdd < 50  and ms > 0.80: return "Excellent — very close to real distribution."
    if fdd < 150 and ms > 0.65: return "Good — generally realistic."
    if fdd < 300 and ms > 0.50: return "Fair — some gap exists."
    return "Poor — large distribution gap. More training steps or data recommended."


def _print_summary(r):
    print("\n" + "=" * 75)
    print("  COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 75)
    d = r["1_distributional_fidelity"]
    print(f"  [1] Distributional fidelity")
    for k in ("FID_inception", "KID_inception_mean", "FDD_dinov3",
              "FCD_clip", "F_RadDINO", "precision_dino", "recall_dino"):
        if k in d: print(f"        {k:30s} {d[k]}")
    pibd = d["per_image_best_dino_sim"]
    print(f"        per_image_best_dino_sim       "
          f"{pibd['value']:.4f}  [95% CI {pibd['ci_low']:.4f}, {pibd['ci_high']:.4f}]")

    print(f"\n  [2] Paired structural (means; full / background / lesion):")
    for sect in ("full_image", "background_only", "lesion_only"):
        s = r["2_paired_structural"].get(sect, {})
        if not s: continue
        print(f"      {sect}:")
        for k in ("ssim", "psnr", "lpips", "mae"):
            if k in s:
                v = s[k]
                print(f"        {k:8s} {v['value']:.4f}  "
                      f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    mf = r["2_paired_structural"].get("mask_fidelity", {})
    if mf:
        print(f"      mask_fidelity:")
        for k, v in mf.items():
            print(f"        {k:15s} {v['value']:.4f}  "
                  f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

    print(f"\n  [3] Diversity (Vendi >> 1 = diverse, ~1 = mode collapse):")
    for k, v in r["3_diversity"].items(): print(f"        {k:35s} {v}")

    if r["4_memorization"]:
        print(f"\n  [4] Memorization (training-set leakage check):")
        for k, v in r["4_memorization"].items(): print(f"        {k:35s} {v}")

    print(f"\n  [5] Texture / frequency (DFB sanity check):")
    print(f"        real wavelet  : {r['5_texture_frequency']['real_wavelet']}")
    print(f"        gen  wavelet  : {r['5_texture_frequency']['gen_wavelet']}")
    print(f"        wavelet L1Δ   : {r['5_texture_frequency'].get('wavelet_l1_delta','-')}")
    if 'power_band_l1_delta' in r['5_texture_frequency']:
        print(f"        power-band L1Δ: {r['5_texture_frequency']['power_band_l1_delta']}")

    print(f"\n  [6] Color / dermoscopy (selected):")
    for k in ("hist_inter_r","hist_inter_g","hist_inter_b",
              "wasserstein_r","wasserstein_g","wasserstein_b"):
        if k in r["6_color_dermoscopy"]:
            print(f"        {k:18s} {r['6_color_dermoscopy'][k]}")

    if r["7_downstream_segmentation"]:
        print(f"\n  [7] Downstream segmentation utility:")
        for k, v in r["7_downstream_segmentation"].items():
            print(f"        {k:35s} {v}")

    print(f"\n  Overall grade: {r['summary']['quality_grade']}")
    print("=" * 75 + "\n")


def save_comparison_grid(real_imgs, gen_imgs, sims, out_path, n_cols=4):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Viz] matplotlib not available — skipping grid."); return
    n = min(len(real_imgs), len(gen_imgs), len(sims))
    n_show = min(n, n_cols * 4)
    n_rows = math.ceil(n_show / n_cols)
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols*3, n_rows*6))
    axes = np.array(axes).reshape(n_rows * 2, n_cols)
    for idx in range(n_show):
        r, c, s = (idx // n_cols)*2, idx % n_cols, sims[idx]
        axes[r,   c].imshow(real_imgs[idx % len(real_imgs)]); axes[r,   c].axis("off")
        axes[r,   c].set_title("Real", fontsize=8, color="steelblue")
        col = "green" if s > 0.75 else ("orange" if s > 0.55 else "red")
        axes[r+1, c].imshow(gen_imgs[idx]); axes[r+1, c].axis("off")
        axes[r+1, c].set_title(f"Gen  sim={s:.3f}", fontsize=8, color=col)
    for idx in range(n_show, n_rows * n_cols):
        axes[(idx//n_cols)*2,   idx%n_cols].axis("off")
        axes[(idx//n_cols)*2+1, idx%n_cols].axis("off")
    fig.legend(handles=[
        mpatches.Patch(color="green",  label="best-sim > 0.75 high"),
        mpatches.Patch(color="orange", label="0.55–0.75 medium"),
        mpatches.Patch(color="red",    label="< 0.55 low"),
    ], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("MedDiff-FT + DFB · Comprehensive Validation", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[Viz] Grid saved → {out_path}")


if __name__ == "__main__":
    main()

# 6. Download SD-1.5 inpainting (only once)
import os, torch
from diffusers import StableDiffusionInpaintPipeline
save_path = "/kaggle/working/sd15-inpaint"
if not os.path.isdir(save_path) or not os.listdir(save_path):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False)
    pipe.save_pretrained(save_path); del pipe; torch.cuda.empty_cache()
print("Model at", save_path)

# ## 7. Build mini ISIC-2018 dataset (30 pairs)
# import os, shutil, json, random
# from glob import glob
# SRC_IMG = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
# SRC_MSK = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth"
# DST     = "/kaggle/working/MedDiff-FT/data/train"
# os.makedirs(f"{DST}/images", exist_ok=True); os.makedirs(f"{DST}/masks", exist_ok=True)
# for f in glob(f"{DST}/images/*"): os.remove(f)
# for f in glob(f"{DST}/masks/*"):  os.remove(f)
# imgs = sorted(glob(f"{SRC_IMG}/*.jpg")); random.seed(42); random.shuffle(imgs); imgs = imgs[:30]
# entries = {"image": [], "mask_1": [], "prompt": []}
# for ip in imgs:
#     name = os.path.splitext(os.path.basename(ip))[0]
#     mp = f"{SRC_MSK}/{name}_segmentation.png"
#     if not os.path.exists(mp): continue
#     shutil.copy(ip, f"{DST}/images/{name}.jpg")
#     shutil.copy(mp, f"{DST}/masks/{name}_segmentation.png")
#     entries["image"].append(f"../data/train/images/{name}.jpg")
#     entries["mask_1"].append(f"../data/train/masks/{name}_segmentation.png")
#     entries["prompt"].append("a photo of hta")
# with open(f"{DST}/data.json","w") as f: json.dump(entries, f, indent=2)
# print("Pairs:", len(entries["image"]))

# # ============================================================================
# # Build a class-balanced, mask-verified, training-disjoint curated set
# # ============================================================================
# # Strategy:
# #   1. Find HAM10000_metadata.csv (tries common Kaggle paths, then falls back
# #      to a structure-based diversity sample if metadata isn't available).
# #   2. Keep only IDs that actually have a Task-1 segmentation mask on disk.
# #   3. Exclude the 30 IDs already in /kaggle/working/MedDiff-FT/data/train.
# #   4. Sample N per class across {df, akiec, bcc, mel, bkl, nv, vasc}
# #      using a fixed seed so the run is reproducible.
# # ============================================================================
# import os, json, shutil, random
# import pandas as pd
# from glob import glob

# SRC_IMG    = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
# SRC_MSK    = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth"
# TRAIN_JSON = "/kaggle/working/MedDiff-FT/data/train/data.json"
# DST        = "/kaggle/working/MedDiff-FT/data/train"
# os.makedirs(f"{DST}/images", exist_ok=True)
# os.makedirs(f"{DST}/masks",  exist_ok=True)

# # --- 1. Locate HAM10000 metadata (attach the dataset to your notebook first) -
# META_CANDIDATES = [
#     "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv",
#     "/kaggle/input/ham10000-metadatacsv/HAM10000_metadata.csv",
#     "/kaggle/input/datasets/xxc025/isic2018/HAM10000_metadata.csv",
#     "/kaggle/input/datasets/abdelmoghitezouine11/meddiff-ft-v2/ISIC2018_Task3_Test_GroundTruth.csv",
#     "/kaggle/input/datasets/abdelmoghitezouine11/meddiff-ft-v2/HAM10000_metadata.csv"
# ]
# meta_path = next((p for p in META_CANDIDATES if os.path.exists(p)), None)

# # --- 2. Build the pool of mask-verified IDs ---------------------------------
# available = {
#     os.path.basename(p).replace("_segmentation.png", "")
#     for p in glob(f"{SRC_MSK}/*_segmentation.png")
# }
# with open(TRAIN_JSON) as f:
#     train_data = json.load(f)
# train_ids = {os.path.basename(p).replace("_segmentation.png", "")
#              for p in train_data["mask_1"]}
# candidates = available - train_ids
# print(f"Task-1 masks on disk : {len(available)}")
# print(f"Already in training  : {len(train_ids)}")
# print(f"Candidate pool       : {len(candidates)}")

# # --- 3. Class-balanced sampling --------------------------------------------
# # Per-class quota (sums to 30). Tuned to give the rare classes meaningful
# # representation while still over-sampling NV (the majority class) modestly.
# QUOTA = {"df": 5, "akiec": 5, "bcc": 5, "mel": 6, "bkl": 5, "nv": 4, "vasc": 0}
# random.seed(20260512)

# selected, by_class = [], {}

# if meta_path:
#     print(f"\nUsing metadata: {meta_path}")
#     meta = pd.read_csv(meta_path)

#     # Drop rows whose image_id does NOT appear in the mask list
#     meta = meta[meta["image_id"].isin(candidates)]

#     # If no IDs survived, switch to the fallback immediately
#     if meta.empty:
#         print("⚠️  No metadata IDs matched the mask files – switching to evenly‑spread sample.")
#         meta_path = None   # force fallback below
#     else:
#         for cls, n in QUOTA.items():
#             pool = meta[meta["dx"] == cls]["image_id"].tolist()
#             random.shuffle(pool)
#             picked = pool[:n]
#             by_class[cls] = picked
#             selected += picked
#             print(f"  {cls:6s}: pool={len(pool):4d}  selected={len(picked)}")

#         # Top‑up from biggest remaining pools
#         shortfall = 30 - len(selected)
#         if shortfall > 0:
#             # Safe way to build {class: [remaining_image_ids]} without FutureWarning
#             leftover = (
#                 meta[~meta["image_id"].isin(selected)]
#                 .groupby("dx")["image_id"]
#                 .apply(list)
#                 .to_dict()
#             )
#             if leftover:
#                 for cls in sorted(leftover, key=lambda c: -len(leftover[c])):
#                     need = 30 - len(selected)
#                     if need <= 0:
#                         break
#                     add = leftover[cls][:need]
#                     by_class.setdefault(cls, []).extend(add)
#                     selected += add
#                     print(f"  top-up {cls}: +{len(add)}")
#             else:
#                 print("  (top-up pool empty)")

# # ---- Fallback (no metadata or empty match) ----
# if not meta_path or not selected:
#     print(
#         "\n⚠️  Falling back to evenly‑spread diversity sample "
#         "(no per‑class guarantee)."
#     )
#     pool = sorted(candidates)
#     step = max(1, len(pool) // 30)
#     selected = pool[::step][:30]
#     by_class = {"unknown": selected}

# # --- 4. Copy files & write data.json ---------------------------------------
# entries = {"image": [], "mask_1": [], "prompt": []}
# for name in selected:
#     ip = f"{SRC_IMG}/{name}.jpg"
#     mp = f"{SRC_MSK}/{name}_segmentation.png"
#     if not (os.path.exists(ip) and os.path.exists(mp)):
#         print(f"  skipping {name} (file missing)")
#         continue
#     shutil.copy(ip, f"{DST}/images/{name}.jpg")
#     shutil.copy(mp, f"{DST}/masks/{name}_segmentation.png")
#     entries["image"].append(f"../data/train/images/{name}.jpg")
#     entries["mask_1"].append(f"../data/train/masks/{name}_segmentation.png")
#     entries["prompt"].append("a photo of hta")

# with open(f"{DST}/data.json", "w") as f:
#     json.dump(entries, f, indent=2)

# print(f"\n✅ Wrote {len(entries['image'])} pairs to {DST}")
# print("\nClass distribution of the curated 30:")
# for cls, ids in by_class.items():
#     print(f"  {cls:6s}: {len(ids):2d}  {ids}")

%ls /kaggle/input/datasets/xxc025/isic2018/

# # 7. Build mini ISIC-2018 dataset — curated IDs with Task 1 mask verification
# import os, shutil, json
# from glob import glob

# SRC_IMG = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
# SRC_MSK = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth"
# DST     = "/kaggle/working/MedDiff-FT/data/train"

# os.makedirs(f"{DST}/images", exist_ok=True)
# os.makedirs(f"{DST}/masks",  exist_ok=True)
# for f in glob(f"{DST}/images/*"): os.remove(f)
# for f in glob(f"{DST}/masks/*"):  os.remove(f)

# # ── Step 1: find out which Task-1 masks actually exist ──────────────────────
# available_masks = {
#     os.path.basename(p).replace("_segmentation.png", "")
#     for p in glob(f"{SRC_MSK}/*_segmentation.png")
# }
# print(f"Task-1 masks on disk: {len(available_masks)}")

# # ── Step 2: curated list (chosen for diagnostic diversity) ──────────────────
# CURATED_IDS = [
#     # Dermatofibroma (DF)
#     "ISIC_0001130", "ISIC_0001114", "ISIC_0011410", "ISIC_0011433",
#     "ISIC_0011677", "ISIC_0011478", "ISIC_0011865",
#     # Actinic Keratosis (AKIEC)
#     "ISIC_0010512", "ISIC_0010889",
#     # Basal Cell Carcinoma (BCC)
#     "ISIC_0012057", "ISIC_0024280", "ISIC_0024272", "ISIC_0024235",
#     "ISIC_0024232", "ISIC_0011894", "ISIC_0024248",
#     # Melanoma (MEL)
#     "ISIC_0012382", "ISIC_0013775", "ISIC_0014325", "ISIC_0011239",
#     "ISIC_0000291", "ISIC_0001148", "ISIC_0000516", "ISIC_0013815",
#     # Benign Keratosis (BKL)
#     "ISIC_0012719", "ISIC_0011380", "ISIC_0014634",
#     # Melanocytic Nevus (NV)
#     "ISIC_0003280", "ISIC_0002913", "ISIC_0000229",
# ]

# # ── Step 3: check which curated IDs have masks ──────────────────────────────
# have_mask    = [n for n in CURATED_IDS if n in available_masks]
# missing_mask = [n for n in CURATED_IDS if n not in available_masks]

# print(f"\nCurated IDs with a Task-1 mask : {len(have_mask)}")
# print(f"Curated IDs WITHOUT a mask     : {len(missing_mask)}")
# if missing_mask:
#     print("  Missing:", missing_mask)

# # ── Step 4: if too few curated IDs survive, backfill from Task-1 pool ───────
# TARGET = 30
# entries = {"image": [], "mask_1": [], "prompt": []}

# selected = list(have_mask)  # start with whatever curated IDs have masks

# if len(selected) < TARGET:
#     print(f"\nOnly {len(selected)} curated IDs have masks — backfilling from Task-1 pool...")
#     # exclude already-selected, then fill up to TARGET
#     already = set(selected)
#     backfill_pool = sorted(available_masks - already)  # deterministic order
#     backfill      = backfill_pool[:TARGET - len(selected)]
#     selected      += backfill
#     print(f"Backfilled {len(backfill)} images from Task-1 pool")

# # ── Step 5: copy and build data.json ────────────────────────────────────────
# for name in selected:
#     ip = f"{SRC_IMG}/{name}.jpg"
#     mp = f"{SRC_MSK}/{name}_segmentation.png"
#     if not os.path.exists(ip):
#         print(f"  WARNING: image file missing for {name}"); continue
#     shutil.copy(ip, f"{DST}/images/{name}.jpg")
#     shutil.copy(mp, f"{DST}/masks/{name}_segmentation.png")
#     entries["image"].append(f"../data/train/images/{name}.jpg")
#     entries["mask_1"].append(f"../data/train/masks/{name}_segmentation.png")
#     entries["prompt"].append("a photo of hta")

# with open(f"{DST}/data.json", "w") as f:
#     json.dump(entries, f, indent=2)

# print(f"\nFinal pairs written: {len(entries['image'])} / {TARGET}")
# print(f"  From curated list : {len(have_mask)}")
# print(f"  From Task-1 backfill: {len(selected) - len(have_mask)}")

# # =============================================================================
# # CELL 10 — Stage selected files for MedDiff-FT training
# # =============================================================================
# # Copies chosen images + masks into a curated folder and writes data.json.
# # VERIFY paths and the data.json schema against your training repo before use.
# import json, shutil
# import pandas as pd

# selection = pd.read_csv("/kaggle/input/datasets/abdelmoghitezouine11/meddiff-ft-v2/meddiff_ft_selection.csv")

# DST = "/kaggle/working/MedDiff-FT/data/train"

# DATASET_BASE = "/kaggle/input/datasets/xxc025/isic2018"
# IMG_DIR  = f"{DATASET_BASE}/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
# MASK_DIR = f"{DATASET_BASE}/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth"


# os.makedirs(f"{DST}/images", exist_ok=True)
# os.makedirs(f"{DST}/masks",  exist_ok=True)

# PROMPTS = ["a photo of hta",
#            "a dermoscopy photo of hta",
#            "a clinical close-up of hta"]

# entries = {"image": [], "mask_1": [], "prompt": []}
# missing = []
# for i, (_, r) in enumerate(selection.iterrows()):
#     iid = r["image_id"]
#     si = os.path.join(IMG_DIR, iid + ".jpg")
#     sm = os.path.join(MASK_DIR, iid + "_segmentation.png")
#     if not (os.path.exists(si) and os.path.exists(sm)):
#         missing.append(iid); continue
#     shutil.copy(si, f"{DST}/images/{iid}.jpg")
#     shutil.copy(sm, f"{DST}/masks/{iid}_segmentation.png")
#     entries["image"].append(f"../data/train/images/{iid}.jpg")
#     entries["mask_1"].append(f"../data/train/masks/{iid}_segmentation.png")
#     entries["prompt"].append(PROMPTS[i % len(PROMPTS)])

# with open(f"{DST}/data.json", "w") as f:
#     json.dump(entries, f, indent=2)

# print(f"Staged {len(entries['image'])} pairs -> {DST}")
# if missing:
#     print(f"Skipped (missing image or mask): {missing}")
# print("\nTraining launch:  --instance_data_dir ../data/train")

## 7. Build mini ISIC-2018 dataset (30 pairs)
import os, shutil, json, random
from glob import glob
SRC_IMG = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
SRC_MSK = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth"
DST     = "/kaggle/working/MedDiff-FT/data/train"
os.makedirs(f"{DST}/images", exist_ok=True); os.makedirs(f"{DST}/masks", exist_ok=True)
for f in glob(f"{DST}/images/*"): os.remove(f)
for f in glob(f"{DST}/masks/*"):  os.remove(f)
imgs = sorted(glob(f"{SRC_IMG}/*.jpg")); random.seed(42); random.shuffle(imgs); imgs = imgs[:30]
entries = {"image": [], "mask_1": [], "prompt": []}
for ip in imgs:
    name = os.path.splitext(os.path.basename(ip))[0]
    mp = f"{SRC_MSK}/{name}_segmentation.png"
    if not os.path.exists(mp): continue
    shutil.copy(ip, f"{DST}/images/{name}.jpg")
    shutil.copy(mp, f"{DST}/masks/{name}_segmentation.png")
    entries["image"].append(f"../data/train/images/{name}.jpg")
    entries["mask_1"].append(f"../data/train/masks/{name}_segmentation.png")
    entries["prompt"].append("a photo of hta")
with open(f"{DST}/data.json","w") as f: json.dump(entries, f, indent=2)
print("Pairs:", len(entries["image"]))

# 8. Accelerate config — pure DDP for 2× T4 (the recommended path)
import os
ACC_CFG = "/kaggle/working/.cache/huggingface/accelerate"
os.makedirs(ACC_CFG, exist_ok=True)
cfg_path = f"{ACC_CFG}/default_config.yaml"
with open(cfg_path,"w") as f:
    f.write('''compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
gpu_ids: all
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
''')
os.environ["ACCELERATE_CONFIG_FILE"] = cfg_path
print("Wrote", cfg_path)

# %cat /kaggle/working/MedDiff-FT/data/train/data.json

# import subprocess, time, os
# log_path = "/kaggle/working/checkpoints/isic2018_dfb/logs/train_rank0.log"
# while not os.path.exists(log_path):
#     time.sleep(2)
# proc = subprocess.Popen(["tail", "-n", "50", "-f", log_path],
#     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
# try:
#     for line in proc.stdout:
#         print(line, end="", flush=True)
# except KeyboardInterrupt:
#     proc.terminate()

!pip install -q -U accelerate

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 9. Launch training — DDP + 8-bit AdamW + gradient checkpointing
#
# Per-T4 VRAM budget at 512² fp16 (measured):
#   • UNet + DFB fp16 weights ........ ~2.0 GB
#   • Text encoder + VAE fp16 ........ ~0.7 GB
#   • Activations (grad-checkpointed). ~3-4 GB
#   • 8-bit AdamW state .............. ~1.0 GB
#   • Working buffers + NCCL ......... ~1-2 GB
#   ----------------------------------------------
#   Total ............................ ~8-10 GB     (T4 has 16 GB)
#
# Validation is DISABLED in-training by default (see --disable_validation).
# Use the inference cell at the bottom of the notebook for visual checks.

%cd /kaggle/working/MedDiff-FT/main
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 TOKENIZERS_PARALLELISM=false \
 accelerate launch --config_file=$ACCELERATE_CONFIG_FILE train_dfb.py \
    --pretrained_model_name_or_path=/kaggle/working/sd15-inpaint \
    --instance_data_dir ../data \
    --output_dir=/kaggle/working/checkpoints/isic2018_dfb_v7 \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=800 \
    --mixed_precision=fp16 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --adam_weight_decay 1e-3 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-8 \
    --enable_vae_slicing \
    --cache_latents \
    --checkpointing_steps=400 \
    --checkpointing_from=400 \
    --checkpoints_total_limit=1 \
    --seed=42 \
    --use_dfb \
    --dfb_heads 4 \
    --dfb_inner_dim_factor 0.6 \
    --disable_validation \
    --freq_loss_weight 0.08 \
    --freq_loss_prob 0.30 \
    --freq_loss_w_hh 0.20 \
    --freq_loss_w_fft 1.0 \
    --freq_loss_w_ll 1.0 \
    --learning_rate 1.5e-5 \
    --lr_warmup_steps 40 \
    --lr_num_cycles 2 \
    --set_grads_to_none \
    --lr_scheduler cosine_with_restart \
    --soft_inner_blur 5 \
    --soft_outer_blur 10

# # 9. Launch training — DDP + 8-bit AdamW + gradient checkpointing
# #
# # Per-T4 VRAM budget at 512² fp16 (measured):
# #   • UNet + DFB fp16 weights ........ ~2.0 GB
# #   • Text encoder + VAE fp16 ........ ~0.7 GB
# #   • Activations (grad-checkpointed). ~3-4 GB
# #   • 8-bit AdamW state .............. ~1.0 GB
# #   • Working buffers + NCCL ......... ~1-2 GB
# #   ----------------------------------------------
# #   Total ............................ ~8-10 GB     (T4 has 16 GB)
# #
# # Validation is DISABLED in-training by default (see --disable_validation).
# # Use the inference cell at the bottom of the notebook for visual checks.

# %cd /kaggle/working/MedDiff-FT/main
# !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#  TOKENIZERS_PARALLELISM=false \
#  accelerate launch --config_file=$ACCELERATE_CONFIG_FILE train_dfb.py \
#     --pretrained_model_name_or_path=/kaggle/working/sd15-inpaint \
#     --instance_data_dir ../data \
#     --output_dir=/kaggle/working/checkpoints/isic2018_dfb_v7 \
#     --resolution=512 \
#     --train_batch_size=1 \
#     --gradient_accumulation_steps=4 \
#     --max_train_steps=1200 \
#     --mixed_precision=fp16 \
#     --gradient_checkpointing \
#     --use_8bit_adam \
#     --adam_weight_decay 1e-3 \
#     --max_grad_norm 1.0 \
#     --adam_epsilon 1e-8 \
#     --enable_vae_slicing \
#     --cache_latents \
#     --checkpointing_steps=600 \
#     --checkpointing_from=600 \
#     --checkpoints_total_limit=1 \
#     --seed=42 \
#     --use_dfb \
#     --dfb_heads 4 \
#     --dfb_inner_dim_factor 1.0 \
#     --disable_validation \
#     --freq_loss_weight 0.10 \
#     --freq_loss_prob 0.40 \
#     --freq_loss_w_hh 0.25 \
#     --freq_loss_w_fft 1.0 \
#     --freq_loss_w_ll 1.0 \
#     --learning_rate 3e-5 \
#     --lr_warmup_steps 100 \
#     --lr_num_cycles 2 \
#     --set_grads_to_none \
#     --lr_scheduler cosine_with_restarts \
#     --soft_inner_blur 5 \
#     --soft_outer_blur 5

# log_path = "/kaggle/working/checkpoints/isic2018_dfb/logs/train_rank0.log"

# with open(log_path) as f:
#     print(f.read())

%ls -R /kaggle/working/checkpoints/

# 10. Inspect the saved checkpoint format
import os
OUT = "/kaggle/working/checkpoints/isic2018_dfb_v7"
print("Final dir contents:", sorted(os.listdir(OUT)))
unet_dir = os.path.join(OUT, "unet")
print("\nUNet dir contents:")
for f in sorted(os.listdir(unet_dir)):
    p = os.path.join(unet_dir, f)
    print(f"  {f:50s}  {os.path.getsize(p)/1e6:7.1f} MB")
# You should see:
#   config.json
#   diffusion_pytorch_model.safetensors   (~1.7 GB, vanilla UNet)
#   dfb_weights.safetensors                (~120-240 MB, DFB sidecar)
#   dfb_config.json                        (heads + inner_dim_factor)



# 11. Inference inputs
import os, random, shutil
from glob import glob
from PIL import Image
os.makedirs("/kaggle/working/bg_images",   exist_ok=True)
os.makedirs("/kaggle/working/guide_masks", exist_ok=True)
input_folder  = "/kaggle/input/datasets/abdelmoghitezouine11/bg-images"
output_folder = "/kaggle/working/bg_images"
for fn in os.listdir(input_folder):
    if fn.lower().endswith((".png",".jpg",".jpeg")):
        Image.open(os.path.join(input_folder,fn)).convert("RGB")\
             .save(os.path.join(output_folder,fn))
SRC_MSK = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth"
SRC_IMG = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
imgs = sorted(glob(f"{SRC_IMG}/*.jpg")); random.seed(1); random.shuffle(imgs)
for ip in imgs[:25]:
    name = os.path.splitext(os.path.basename(ip))[0]
    mp = f"{SRC_MSK}/{name}_segmentation.png"
    if os.path.exists(mp):
        shutil.copy(mp, f"/kaggle/working/guide_masks/{name}_segmentation.png")
print("BGs :", len(os.listdir("/kaggle/working/bg_images")))
print("Msks:", len(os.listdir("/kaggle/working/guide_masks")))

import os, shutil, random
from glob import glob

# Paths
SRC_MSK = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth"
DST     = "/kaggle/working/guide_masks"
os.makedirs(DST, exist_ok=True)

# All available masks
all_masks = sorted(glob(f"{SRC_MSK}/*_segmentation.png"))
# Exclude the 30 training IDs (retrieve them from your training data.json)
import json
with open("/kaggle/working/MedDiff-FT/data/train/data.json") as f:
    train_data = json.load(f)
train_ids = {os.path.basename(p).replace("_segmentation.png","") for p in train_data["mask_1"]}

# Filter out training masks
candidate_paths = [p for p in all_masks if os.path.basename(p).replace("_segmentation.png","") not in train_ids]

# Randomly pick 100 (or all if fewer)
random.seed(42)
random.shuffle(candidate_paths)
for p in candidate_paths[:100]:
    shutil.copy(p, DST)

print(f"Copied {len(os.listdir(DST))} masks to {DST}")

# # 7b. Build IP-Adapter reference directory (refs ∩ training = ∅)
# # ============================================================================
# # Picks N reference images from the ISIC-2018 pool that are GUARANTEED to NOT
# # overlap with the 30 images chosen by cell 12 for training. Use this dir at
# # validation time via:  --ip_adapter_image_dir /kaggle/working/MedDiff-FT/data/ip_refs
# # ============================================================================
# import os, json, random, shutil
# from glob import glob

# SRC_IMG       = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
# TRAIN_DATA    = "/kaggle/working/MedDiff-FT/data/train/data.json"
# TRAIN_IMG_DIR = "/kaggle/working/MedDiff-FT/data/train/images"
# TRAIN_MSK_DIR = "/kaggle/working/MedDiff-FT/data/train/masks"
# REF_DIR       = "/kaggle/working/MedDiff-FT/data/ip_refs"
# NUM_REFS      = 30      # one ref per mask gives every generation a unique ref
# REF_SEED      = 1337    # MUST differ from cell 12's seed (42)

# # ---- 1) collect filenames we must exclude ----------------------------------
# train_names = set()
# if os.path.exists(TRAIN_DATA):
#     with open(TRAIN_DATA) as f:
#         d = json.load(f)
#     train_names = {os.path.basename(p) for p in d["image"]}
# elif os.path.isdir(TRAIN_IMG_DIR):                       # fallback
#     train_names = set(os.listdir(TRAIN_IMG_DIR))

# if not train_names:
#     print("⚠️  No training set found — run cell 12 first, otherwise refs may "
#           "overlap with a future training selection!")
# else:
#     print(f"Training images to EXCLUDE from refs : {len(train_names)}")

# # ---- 2) candidate pool = all ISIC images − training set --------------------
# all_imgs   = sorted(glob(f"{SRC_IMG}/*.jpg"))
# candidates = [p for p in all_imgs if os.path.basename(p) not in train_names]
# print(f"Total ISIC images on disk            : {len(all_imgs)}")
# print(f"Candidates after excluding training  : {len(candidates)}")

# assert len(candidates) >= NUM_REFS, (
#     f"Only {len(candidates)} non-training images available, "
#     f"need {NUM_REFS}. Reduce NUM_REFS or point SRC_IMG at a larger pool.")

# # ---- 3) deterministic random sample (separate RNG, separate seed) ----------
# rng       = random.Random(REF_SEED)
# ref_paths = rng.sample(candidates, NUM_REFS)

# # ---- 4) write the ref directory --------------------------------------------
# os.makedirs(REF_DIR, exist_ok=True)
# for f in glob(f"{REF_DIR}/*"):
#     os.remove(f)                                          # clean previous run
# for p in ref_paths:
#     shutil.copy(p, os.path.join(REF_DIR, os.path.basename(p)))

# # ---- 5) hard sanity check: no overlap at all -------------------------------
# ref_names = {os.path.basename(p) for p in ref_paths}
# overlap   = ref_names & train_names
# assert not overlap, f"BUG: refs overlap training set: {sorted(overlap)}"

# print(f"\n✓ Wrote {len(ref_paths)} reference images → {REF_DIR}")
# print(f"✓ Zero overlap with the {len(train_names)}-image training set")
# print(f"\nFirst 5 refs:")
# for p in ref_paths[:5]:
#     print(f"  {os.path.basename(p)}")

# =============================================================================
# Build hue-targeted IP-Adapter refs (excludes training 25, applies same gates)
# =============================================================================
import os, shutil, cv2, pandas as pd

# --- paths -------------------------------------------------------------------
IMG_DIR     = "/kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
RANKING_CSV = "/kaggle/input/datasets/abdelmoghitezouine11/meddiff-ft-v2/meddiff_ft_full_ranking (1).csv"
SELECT_CSV  = "/kaggle/input/datasets/abdelmoghitezouine11/meddiff-ft-v2/meddiff_ft_selection.csv"
DST         = "/kaggle/working/MedDiff-FT/data/ip_refs"

# --- exclusions: training set + same blocklist as v2 notebook ----------------
train_ids = set(pd.read_csv(SELECT_CSV)["image_id"])
BLOCKLIST = {
    "ISIC_0013527", "ISIC_0015953", "ISIC_0013167", "ISIC_0012330",
    "ISIC_0000065", "ISIC_0000028", "ISIC_0012390", "ISIC_0012447",
    "ISIC_0012323", "ISIC_0012314", "ISIC_0000169", "ISIC_0000210",
    "ISIC_0000170", "ISIC_0013684", "ISIC_0012705", "ISIC_0014599",
    "ISIC_0012207", "ISIC_0012693",
}

# --- load full ranking and re-apply the v2 gates -----------------------------
df = pd.read_csv(RANKING_CSV)
def passes(r):
    if r["marker_n"]    < 0.95: return False
    if r["glare_n"]     < 0.90: return False
    if r["letterbox_n"] < 0.99: return False
    if r["ruler_n"]     < 0.40: return False
    if r["class"] in ("NV", "MEL"):
        return r["sharpness_n"] >= 0.70 and r["hair_n"] >= 0.70 and r["vignette_n"] >= 0.65
    return r["sharpness_n"] >= 0.60 and r["hair_n"] >= 0.50 and r["vignette_n"] >= 0.55

pool = df[~df["image_id"].isin(train_ids | BLOCKLIST)
          & df["class"].isin(["NV","MEL","BKL"])].copy()
pool = pool[pool.apply(passes, axis=1)]
print(f"Candidates after gates + exclusions: {len(pool)}  "
      f"by class: {dict(pool['class'].value_counts())}")

# --- compute per-image HSV hue mean (the metric we are targeting) -----------
TARGET_HUE = 0.189   # real_hue_mean from your validation report
hues = {}
for iid in pool["image_id"]:
    img = cv2.imread(f"{IMG_DIR}/{iid}.jpg")
    if img is None: continue
    img = cv2.resize(img, (256, 192))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hues[iid] = float(hsv[:, :, 0].mean() / 180.0)   # OpenCV hue 0-180 -> normalise
pool["hue_mean"] = pool["image_id"].map(hues)
pool["hue_dist"] = (pool["hue_mean"] - TARGET_HUE).abs()

# --- hue-aware ranking: trade off quality vs hue proximity ------------------
# ip_score = quality - 2 * |hue - real_hue|. Pick top-N per class by ip_score.
pool["ip_score"] = pool["quality_score"] - 2.0 * pool["hue_dist"]

QUOTA = {"NV": 8, "MEL": 8, "BKL": 8}     # = 24 refs
picked = pd.concat([pool[pool["class"] == c].nlargest(n, "ip_score")
                    for c, n in QUOTA.items()])

print(f"\nSelected {len(picked)} ip_refs.")
print(f"  mean hue of refs : {picked['hue_mean'].mean():.3f}   "
      f"(target {TARGET_HUE})")
print(f"  mean hue of train: "
      f"{df[df['image_id'].isin(train_ids)]['image_id'].map(hues).dropna().mean():.3f}")
print(f"  by class: {dict(picked['class'].value_counts())}")

# --- wipe old refs and copy new ones ----------------------------------------
os.makedirs(DST, exist_ok=True)
for f in os.listdir(DST):
    os.remove(os.path.join(DST, f))
for iid in picked["image_id"]:
    shutil.copy(f"{IMG_DIR}/{iid}.jpg", f"{DST}/{iid}.jpg")
print(f"\nWrote {len(picked)} refs to {DST}")

# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================
# Replaces the old DINOv3-only validate cell with the 8-family evaluator.
#
# Outputs to /kaggle/working/comprehensive_validation/:
#   - generated/                       (all synthesized PNGs)
#   - comprehensive_report.json        (every metric + 95% CIs)
#   - comparison_grid.png              (visual real-vs-gen grid)
#
# Argument tips for the 30-pair regime:
#   --num_real 300            real reference set size for FID/KID
#   --bootstrap 1000          number of bootstrap resamples for CIs
#   --use_raddino             enables medical-domain RadDINO encoder (optional)
#   --run_segmentation_check  trains tiny U-Net for downstream utility (~5–10 min)

!python -u /kaggle/working/MedDiff-FT/main/validate_comprehensive.py \
    --model_path           /kaggle/working/checkpoints/isic2018_dfb_v7 \
    --input_path           /kaggle/working/bg_images \
    --label_path           /kaggle/working/guide_masks \
    --real_dir             /kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input \
    --real_mask_dir        /kaggle/input/datasets/xxc025/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth \
    --train_data_dir       /kaggle/working/MedDiff-FT/data/train \
    --out_dir              /kaggle/working/comprehensive_validation \
    --prompt               "a photo of hta" \
    --seed                 97452 \
    --num_inference_steps  75 \
    --guidance_scale       4.5 \
    --batch_size           1 \
    --mixed_precision      fp16 \
    --num_real             300 \
    --embed_batch_size     8 \
    --bootstrap            1000 \
    --dino_model           facebook/dinov3-vitb16-pretrain-lvd1689m \
    --clip_model           openai/clip-vit-base-patch32 \
    --run_segmentation_check \
    --seg_epochs           30 \
    --device               cuda:0 \
    --use_soft_blend \
    --soft_inner_blur 5 \
    --soft_outer_blur 10 \
    --use_repaint \
    --resample_steps 2 \
    --resample_jump 2 \
    --resample_warmup 6 \
    --ip_adapter_image_dir  /kaggle/working/MedDiff-FT/data/ip_refs \
    --ip_adapter_scale      0.30

# 12. Run DFB-aware inference (loads bare UNet + dfb_weights sidecar)
# !python /kaggle/working/MedDiff-FT/main/infer_dfb.py \
#     --model_path  /kaggle/working/checkpoints/isic2018_dfb \
#     --input_path  /kaggle/working/bg_images \
#     --label_path  /kaggle/working/guide_masks \
#     --out_path    /kaggle/working/synthetic_isic2018_dfb_3 \
#     --prompt      "a photo of hta" \
#     --num_inference_steps 100 \
#     --guidance_scale 12 \
#     --batch_size 1 \
#     --mixed_precision fp16 \
#     --device cuda:0 \
#     --enable_vae_slicing \
#     --seed 150

%ls -R /kaggle/working/comprehensive_validation


# Visual sanity check on generated images
import os, math
from PIL import Image
import matplotlib.pyplot as plt

OUT = "/kaggle/working/comprehensive_validation/generated"
files = sorted(os.listdir(OUT))[:125]
n = len(files); cols = 4; rows = math.ceil(n/cols)
fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
ax = ax.flatten() if n > 1 else [ax]
for a, fn in zip(ax, files):
    a.imshow(Image.open(os.path.join(OUT, fn))); a.axis("off")
    a.set_title(fn, fontsize=8)
for a in ax[len(files):]: a.axis("off")
plt.tight_layout(); plt.show()


# ============================================================================
# READ + PRETTY-PRINT THE COMPREHENSIVE REPORT
# ============================================================================
import json, os
from pprint import pprint

REPORT = "/kaggle/working/comprehensive_validation/comprehensive_report.json"
with open(REPORT) as f:
    R = json.load(f)

def banner(t, ch="="):
    line = ch * 78
    print(f"\n{line}\n  {t}\n{line}")

def fmt_ci(d):
    if isinstance(d, dict) and "value" in d:
        return f"{d['value']:.4f}  [95% CI {d['ci_low']:.4f}, {d['ci_high']:.4f}]"
    return str(d)

# ---------- Header ----------
banner("MedDiff-FT + DFB · Comprehensive Evaluation Report")
print(f"  Generated samples : {R['summary']['num_generated']}")
print(f"  Real reference    : {R['summary']['num_real']}")
print(f"  Overall grade     : {R['summary']['quality_grade']}")

# ---------- 1. Distributional fidelity ----------
banner("1. DISTRIBUTIONAL FIDELITY (multi-encoder, lower = better for distances)", "-")
d = R["1_distributional_fidelity"]
print(f"  FID  (Inception-V3)         : {d.get('FID_inception','-')}")
print(f"  KID  (Inception, mean ± std): {d.get('KID_inception_mean','-')}  ± {d.get('KID_inception_std','-')}")
print(f"  FDD  (DINOv3)               : {d.get('FDD_dinov3','-')}")
print(f"  FCD  (CLIP)                 : {d.get('FCD_clip','-')}")
if 'F_RadDINO' in d:
    print(f"  F-RadDINO (medical ViT)     : {d['F_RadDINO']}")
print(f"  Per-image best DINOv3 sim   : {fmt_ci(d['per_image_best_dino_sim'])}")
print(f"  Precision (DINOv3)          : {d.get('precision_dino','-')}")
print(f"  Recall    (DINOv3)          : {d.get('recall_dino','-')}")

# ---------- 2. Paired structural / perceptual ----------
banner("2. PAIRED STRUCTURAL / PERCEPTUAL — full / background-only / lesion-only", "-")
P = R["2_paired_structural"]
for region in ("full_image", "background_only", "lesion_only"):
    s = P.get(region, {})
    if not s: continue
    print(f"\n  {region}:")
    for k in ("ssim", "psnr", "lpips", "mae", "mse"):
        if k in s:
            print(f"    {k:8s} {fmt_ci(s[k])}")

mf = P.get("mask_fidelity", {})
if mf:
    print(f"\n  mask_fidelity (inpainting correctness):")
    for k, v in mf.items():
        print(f"    {k:18s} {fmt_ci(v)}")

# ---------- 3. Diversity ----------
banner("3. DIVERSITY  (Vendi >> 1 = diverse · ~1 = mode collapse)", "-")
for k, v in R["3_diversity"].items():
    print(f"  {k:38s} {v}")

# Compute Vendi efficiency: gen_vendi / real_vendi → 1.0 = matches real diversity
div = R["3_diversity"]
if "vendi_dinov3" in div and "vendi_real_dinov3" in div and div["vendi_real_dinov3"] > 0:
    eff = div["vendi_dinov3"] / div["vendi_real_dinov3"]
    print(f"\n  Diversity efficiency vs real (DINOv3): {eff:.3f}  "
          f"({'good' if eff > 0.6 else 'mode collapse risk'})")

# ---------- 4. Memorization ----------
if R["4_memorization"]:
    banner("4. MEMORIZATION  (top-1 cos > 0.95 OR ssim > 0.92 = suspicious copy)", "-")
    M = R["4_memorization"]
    for k, v in M.items():
        flag = ""
        if "flag_rate" in k and v > 0.05: flag = "  ← INVESTIGATE"
        print(f"  {k:42s} {v}{flag}")

# ---------- 5. Texture / frequency ----------
banner("5. TEXTURE & FREQUENCY  (DFB sanity check — is high-freq synthesis right?)", "-")
T = R["5_texture_frequency"]
print(f"  GLCM (real)        : {T.get('real_glcm','-')}")
print(f"  GLCM (generated)   : {T.get('gen_glcm','-')}")
print(f"  Wavelet (real)     : {T.get('real_wavelet','-')}")
print(f"  Wavelet (generated): {T.get('gen_wavelet','-')}")
print(f"  Wavelet L1 delta   : {T.get('wavelet_l1_delta','-')}    (lower = closer)")
if 'real_power_band_lo_mid_hi' in T:
    print(f"  Power band  real   : {T['real_power_band_lo_mid_hi']}  (lo / mid / hi)")
    print(f"  Power band  gen    : {T['gen_power_band_lo_mid_hi']}")
    print(f"  Power band L1 delta: {T.get('power_band_l1_delta','-')}")

# ---------- 6. Color ----------
banner("6. COLOR / DERMOSCOPY  (histogram intersection 0–1, higher = closer)", "-")
C = R["6_color_dermoscopy"]
print(f"  Hist intersection RGB    : R={C.get('hist_inter_r','-')}  "
      f"G={C.get('hist_inter_g','-')}  B={C.get('hist_inter_b','-')}")
print(f"  Wasserstein distance RGB : R={C.get('wasserstein_r','-')}  "
      f"G={C.get('wasserstein_g','-')}  B={C.get('wasserstein_b','-')}")
print(f"  HSV mean (real)  : H={C.get('real_hue_mean','-')}  "
      f"S={C.get('real_sat_mean','-')}  V={C.get('real_val_mean','-')}")
print(f"  HSV mean (gen)   : H={C.get('gen_hue_mean','-')}  "
      f"S={C.get('gen_sat_mean','-')}  V={C.get('gen_val_mean','-')}")

# ---------- 7. Downstream segmentation ----------
if R["7_downstream_segmentation"]:
    banner("7. DOWNSTREAM SEGMENTATION UTILITY (medical-task transferability)", "-")
    S = R["7_downstream_segmentation"]
    for k, v in S.items():
        print(f"  {k:38s} {v}")
    if "alignment_drop_real_to_gen" in S:
        drop = S["alignment_drop_real_to_gen"]
        verdict = ("USEFUL — generated lesions are anatomically valid"
                   if drop < 0.15 else
                   "PARTIAL — generated lesions deviate anatomically"
                   if drop < 0.30 else
                   "WEAK — generated lesions don't match guide masks well")
        print(f"\n  → Verdict: {verdict}")

# ---------- TL;DR ----------
banner("TL;DR  (what to report in your paper / writeup)")
print(f"  • FID / KID / FDD / FCD : "
      f"{d.get('FID_inception','-')} / {d.get('KID_inception_mean','-')} / "
      f"{d.get('FDD_dinov3','-')} / {d.get('FCD_clip','-')}")
print(f"  • Lesion SSIM / LPIPS   : "
      f"{P.get('lesion_only', {}).get('ssim',{}).get('value','-')} / "
      f"{P.get('lesion_only', {}).get('lpips',{}).get('value','-')}")
print(f"  • Vendi (DINOv3)        : {div.get('vendi_dinov3','-')}  "
      f"(real = {div.get('vendi_real_dinov3','-')})")
print(f"  • Memorization flag     : "
      f"{R['4_memorization'].get('ssim_flag_rate_top1>0.92','-') if R['4_memorization'] else '-'}")
print(f"  • Mask leakage ratio    : "
      f"{P.get('mask_fidelity',{}).get('leakage_ratio',{}).get('value','-')}")
print(f"  • Grade                 : {R['summary']['quality_grade']}")
print()


# Display the comparison grid (real vs. generated, color-coded by similarity)
from PIL import Image as PILImage
import matplotlib.pyplot as plt

grid = PILImage.open("/kaggle/working/comprehensive_validation/comparison_grid.png")
plt.figure(figsize=(16, 20))
plt.imshow(grid); plt.axis("off"); plt.tight_layout(); plt.show()


# ## 📊 Rich Visualization of the Comprehensive Report
# 
# The cells below render the JSON report into the figures you will most likely include in a paper / writeup.


# ============================================================================
# RICH PLOTS FROM THE COMPREHENSIVE REPORT
# ============================================================================
# Bar charts with 95% CIs for the headline paired metrics, comparing
# full-image / background-only / lesion-only — making it visually obvious
# whether DFB is helping the LESION region (which is what we care about).

import json, numpy as np, matplotlib.pyplot as plt

with open("/kaggle/working/comprehensive_validation/comprehensive_report.json") as f:
    R = json.load(f)

P = R["2_paired_structural"]
regions = ["full_image", "background_only", "lesion_only"]
metrics = ["ssim", "psnr", "lpips", "mae"]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
colors = {"full_image": "steelblue", "background_only": "seagreen",
          "lesion_only": "indianred"}

for ax, m in zip(axes, metrics):
    means, lows, highs, labels, cs = [], [], [], [], []
    for region in regions:
        d = P.get(region, {}).get(m)
        if d is None: continue
        means.append(d["value"])
        lows .append(d["value"] - d["ci_low"])
        highs.append(d["ci_high"] - d["value"])
        labels.append(region.replace("_", "\n"))
        cs.append(colors[region])
    if not means: continue
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[lows, highs], color=cs, capsize=6, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{m.upper()} (95% CI)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

plt.suptitle("Region-Conditioned Paired Metrics  ·  full vs. background-only vs. lesion-only",
             fontsize=12)
plt.tight_layout()
plt.savefig("/kaggle/working/comprehensive_validation/region_metrics_bar.png",
            dpi=120, bbox_inches="tight")
plt.show()


# ============================================================================
# DFB SANITY PLOT — wavelet + frequency band comparison
# ============================================================================
# This is the plot you should put in a DFB ablation table. If DFB is doing
# its job, gen wavelet ratios and gen high-frequency power should be CLOSE
# to real (small L1 delta) — otherwise DFB is not learning useful frequency
# priors.

import json, numpy as np, matplotlib.pyplot as plt

with open("/kaggle/working/comprehensive_validation/comprehensive_report.json") as f:
    R = json.load(f)
T = R["5_texture_frequency"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ---- Wavelet sub-band energy ----
keys = ["ll_ratio", "lh_ratio", "hl_ratio", "hh_ratio"]
real = [T["real_wavelet"][k] for k in keys]
gen  = [T["gen_wavelet" ][k] for k in keys]
x = np.arange(len(keys)); w = 0.35
axes[0].bar(x - w/2, real, w, label="Real",      color="steelblue", alpha=0.85)
axes[0].bar(x + w/2, gen,  w, label="Generated", color="indianred", alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(keys)
axes[0].set_title(f"Haar Wavelet Sub-band Energy  (L1 Δ = {T.get('wavelet_l1_delta','?')})")
axes[0].set_ylabel("Energy ratio"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# ---- Radial power spectrum ----
if "real_power_band_lo_mid_hi" in T and T["real_power_band_lo_mid_hi"]:
    bands = ["low", "mid", "high"]
    real_p = T["real_power_band_lo_mid_hi"]
    gen_p  = T["gen_power_band_lo_mid_hi"]
    x = np.arange(len(bands))
    axes[1].bar(x - w/2, real_p, w, label="Real",      color="steelblue", alpha=0.85)
    axes[1].bar(x + w/2, gen_p,  w, label="Generated", color="indianred", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(bands)
    axes[1].set_title(f"Radial Power Spectrum  (L1 Δ = {T.get('power_band_l1_delta','?')})")
    axes[1].set_ylabel("Normalized power")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("Frequency-Domain Validation  ·  is DFB synthesizing realistic high-freq content?",
             fontsize=12)
plt.tight_layout()
plt.savefig("/kaggle/working/comprehensive_validation/frequency_validation.png",
            dpi=120, bbox_inches="tight")
plt.show()


# ============================================================================
# DIVERSITY / MODE-COLLAPSE PLOT
# ============================================================================
# Vendi score = effective number of modes in the generated set (1 = total
# collapse, N = perfect diversity). We compare gen-Vendi to real-Vendi.

import json, numpy as np, matplotlib.pyplot as plt

with open("/kaggle/working/comprehensive_validation/comprehensive_report.json") as f:
    R = json.load(f)
D = R["3_diversity"]

fig, ax = plt.subplots(figsize=(8, 4))
labels = ["DINOv3", "CLIP", "Inception"]
gen_v  = [D.get("vendi_dinov3"), D.get("vendi_clip"), D.get("vendi_inception")]
real_v = [D.get("vendi_real_dinov3"), None, None]    # only DINOv3 has real baseline

x = np.arange(len(labels)); w = 0.35
gen_arr  = [v if v is not None else 0 for v in gen_v]
real_arr = [v if v is not None else 0 for v in real_v]
ax.bar(x - w/2, gen_arr,  w, label="Generated", color="indianred", alpha=0.85)
ax.bar(x + w/2, real_arr, w, label="Real",      color="steelblue", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Vendi score (effective # of modes)")
ax.set_title("Diversity per Encoder  ·  gen Vendi closer to real Vendi = better coverage")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/kaggle/working/comprehensive_validation/diversity_vendi.png",
            dpi=120, bbox_inches="tight")
plt.show()

if D.get("vendi_real_dinov3") and D.get("vendi_dinov3"):
    eff = D["vendi_dinov3"] / D["vendi_real_dinov3"]
    print(f"\nDiversity efficiency vs real (DINOv3): {eff:.3f}")
    print("  > 0.8  : excellent diversity coverage")
    print("  0.6–0.8: acceptable for low-data regime")
    print("  < 0.6  : likely mode collapse — increase training data, lower CFG, or add prompt variation")


# ============================================================================
# MEMORIZATION CHECK PLOT (critical at N=30 training)
# ============================================================================
# A scatter of every generated image's nearest-training-image similarity
# in DINOv3 space + SSIM. Bars on the right = aggregated stats with thresholds.

import json, numpy as np, matplotlib.pyplot as plt

with open("/kaggle/working/comprehensive_validation/comprehensive_report.json") as f:
    R = json.load(f)
M = R.get("4_memorization", {})

if not M:
    print("Memorization check was skipped (no --train_data_dir).")
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    keys, values, colors = [], [], []
    for k, v in M.items():
        if not isinstance(v, (int, float)): continue
        if "flag_rate" in k:
            colors.append("crimson" if v > 0.05 else "seagreen")
        elif "max" in k:
            colors.append("orange")
        else:
            colors.append("steelblue")
        keys.append(k); values.append(v)
    y = np.arange(len(keys))
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=9)
    ax.axvline(0.95, ls="--", color="crimson", alpha=0.5, label="cos > 0.95 threshold")
    ax.axvline(0.92, ls="--", color="orangered", alpha=0.5, label="ssim > 0.92 threshold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Memorization Check  ·  similarity to TRAINING set "
                 "(should be moderate, not extreme)")
    plt.tight_layout()
    plt.savefig("/kaggle/working/comprehensive_validation/memorization_check.png",
                dpi=120, bbox_inches="tight")
    plt.show()

    # Verdict
    flag_dino = M.get("dinov3_flag_rate_top1>0.95", 0)
    flag_ssim = M.get("ssim_flag_rate_top1>0.92",   0)
    if flag_dino > 0.05 or flag_ssim > 0.05:
        print(f"\n⚠  Memorization concern: {flag_dino*100:.1f}% of gens have "
              f"DINOv3 cos>0.95 vs train, {flag_ssim*100:.1f}% have SSIM>0.92.")
        print("   Recommend: more diverse prompts, lower training steps, or stronger augmentation.")
    else:
        print(f"\n✓ No memorization detected at the 5% flag-rate threshold.")


# ## 📋 One-Page Publishable Summary Table


# ============================================================================
# ONE-PAGE PUBLISHABLE SUMMARY TABLE
# ============================================================================
import json, pandas as pd

with open("/kaggle/working/comprehensive_validation/comprehensive_report.json") as f:
    R = json.load(f)

rows = []
def add(family, name, value, lower_is_better=None, note=""):
    rows.append({"Family": family, "Metric": name, "Value": value,
                 "Direction": ("↓ lower=better" if lower_is_better is True
                               else "↑ higher=better" if lower_is_better is False
                               else ""),
                 "Note": note})

D = R["1_distributional_fidelity"]
add("Distribution",  "FID (Inception)",       D.get("FID_inception"),         True,  "ImageNet prior")
add("Distribution",  "KID (Inception)",       D.get("KID_inception_mean"),    True,  "robust at low N")
add("Distribution",  "FDD (DINOv3)",          D.get("FDD_dinov3"),            True,  "self-sup features")
add("Distribution",  "FCD (CLIP)",            D.get("FCD_clip"),              True,  "semantic features")
if "F_RadDINO" in D:
    add("Distribution","F-RadDINO",           D["F_RadDINO"],                 True,  "medical ViT")
add("Distribution",  "Precision (DINOv3)",    D.get("precision_dino"),        False, "fidelity")
add("Distribution",  "Recall (DINOv3)",       D.get("recall_dino"),           False, "coverage")

P = R["2_paired_structural"]
for region, label in [("full_image","full"),("background_only","bg"),("lesion_only","lesion")]:
    s = P.get(region, {})
    if "ssim"  in s: add("Paired-"+label, "SSIM",  s["ssim"]["value"],   False)
    if "psnr"  in s: add("Paired-"+label, "PSNR",  s["psnr"]["value"],   False)
    if "lpips" in s: add("Paired-"+label, "LPIPS", s["lpips"]["value"],  True)
    if "mae"   in s: add("Paired-"+label, "MAE",   s["mae"]["value"],    True)

mf = P.get("mask_fidelity", {})
if mf:
    add("Mask",     "Outside-mask MAE",  mf["outside_mae"]["value"],   True,  "should be ~0")
    add("Mask",     "Inside-mask MAE",   mf["inside_mae"]["value"],   None,  "should be > outside")
    add("Mask",     "Leakage ratio",     mf["leakage_ratio"]["value"], False, ">>1 means inpainter respected mask")
    add("Mask",     "Edge gradient",     mf["edge_gradient"]["value"], True,  "smaller = smoother blend")

div = R["3_diversity"]
add("Diversity",  "Vendi (DINOv3)",  div.get("vendi_dinov3"),       False)
add("Diversity",  "Vendi (Real)",    div.get("vendi_real_dinov3"),  None,  "reference")
add("Diversity",  "Pairwise cos dist", div.get("mean_pairwise_cos_dist_dinov3"), False)

M = R.get("4_memorization", {})
if M:
    add("Memorization", "Top-1 train cos (DINOv3) max", M.get("dinov3_top1_train_cos_max"), True)
    add("Memorization", "Top-1 train SSIM max",         M.get("ssim_top1_train_max"),       True)
    add("Memorization", "Flag rate cos>0.95 (DINOv3)",  M.get("dinov3_flag_rate_top1>0.95"),True)
    add("Memorization", "Flag rate SSIM>0.92",          M.get("ssim_flag_rate_top1>0.92"),  True)

T = R["5_texture_frequency"]
add("Frequency",  "Wavelet L1Δ (real-gen)", T.get("wavelet_l1_delta"),     True)
if "power_band_l1_delta" in T:
    add("Frequency","Power-band L1Δ",       T.get("power_band_l1_delta"), True)

C = R["6_color_dermoscopy"]
add("Color",      "Hist intersection R",    C.get("hist_inter_r"), False)
add("Color",      "Hist intersection G",    C.get("hist_inter_g"), False)
add("Color",      "Hist intersection B",    C.get("hist_inter_b"), False)

S = R.get("7_downstream_segmentation", {}) or {}
if S and "tiny_unet_dice_gen_vs_guide" in S:
    add("Downstream",   "U-Net val Dice (real)",       S["tiny_unet_val_dice_real"],       False)
    add("Downstream",   "U-Net Dice on gen vs guide",  S["tiny_unet_dice_gen_vs_guide"],   False)
    add("Downstream",   "U-Net IoU on gen vs guide",   S["tiny_unet_iou_gen_vs_guide"],    False)
    add("Downstream",   "Alignment drop (real→gen)",   S["alignment_drop_real_to_gen"],    True)

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("/kaggle/working/comprehensive_validation/summary_table.csv", index=False)
print("\n✓ Summary table → /kaggle/working/comprehensive_validation/summary_table.csv")


# # 13. Visual sanity check
# import os, math
# from PIL import Image
# import matplotlib.pyplot as plt
# OUT = "/kaggle/working/synthetic_isic2018_dfb_2"
# files = sorted(os.listdir(OUT))[:7]
# n = len(files); cols = 3; rows = math.ceil(n/cols)
# fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
# ax = ax.flatten() if n>1 else [ax]
# for a, fn in zip(ax, files):
#     a.imshow(Image.open(os.path.join(OUT, fn))); a.axis("off"); a.set_title(fn, fontsize=8)
# for a in ax[len(files):]: a.axis("off")
# plt.tight_layout(); plt.show()

# # 13. Visual sanity check
# import os, math
# from PIL import Image
# import matplotlib.pyplot as plt
# OUT = "/kaggle/working/synthetic_isic2018_dfb_3"
# files = sorted(os.listdir(OUT))[:7]
# n = len(files); cols = 3; rows = math.ceil(n/cols)
# fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
# ax = ax.flatten() if n>1 else [ax]
# for a, fn in zip(ax, files):
#     a.imshow(Image.open(os.path.join(OUT, fn))); a.axis("off"); a.set_title(fn, fontsize=8)
# for a in ax[len(files):]: a.axis("off")
# plt.tight_layout(); plt.show()

# ---
# 
# ## (Optional) DeepSpeed ZeRO-2 path
# 
# Only use this if you specifically need it. For SD-1.5 inpainting at 512² with
# DDP + 8-bit AdamW + grad-checkpointing, you already have ~6 GB of headroom on
# each T4 — DeepSpeed adds complexity and host-↔-GPU transfer overhead without
# giving you anything practical here.
# 
# **If you do enable DeepSpeed:**
# - DROP `--use_8bit_adam` from the launch command — DeepSpeed manages
#   optimizer-state partitioning itself in fp32, and bnb's 8-bit AdamW state
#   format is incompatible with ZeRO partitioning.
# - Keep `--gradient_checkpointing` and `--enable_vae_slicing`.
# - The CPU offload path is slow on Kaggle (the host ↔ GPU PCIe is shared
#   between both T4s).
# 
# The cells below are *commented out* — uncomment to use DeepSpeed instead of DDP.
# 


# # ----- DeepSpeed install + accelerate config -----
# !pip install -q deepspeed
#
# import os
# ACC_CFG = "/kaggle/working/.cache/huggingface/accelerate"
# os.makedirs(ACC_CFG, exist_ok=True)
# cfg_path = f"{ACC_CFG}/default_config.yaml"
# with open(cfg_path, "w") as f:
#     f.write('''compute_environment: LOCAL_MACHINE
# debug: false
# deepspeed_config:
#   deepspeed_config_file: /kaggle/working/ds_config.json
#   zero3_init_flag: false
# distributed_type: DEEPSPEED
# downcast_bf16: 'no'
# machine_rank: 0
# main_training_function: main
# num_machines: 1
# num_processes: 2
# rdzv_backend: static
# same_network: true
# use_cpu: false
# ''')
# os.environ["ACCELERATE_CONFIG_FILE"] = cfg_path
# print("DeepSpeed accelerate config written.")

# # ----- DeepSpeed ZeRO-2 config (CPU optimizer offload) -----
# %%writefile /kaggle/working/ds_config.json
# {
#     "train_batch_size": "auto",
#     "train_micro_batch_size_per_gpu": 1,
#     "gradient_accumulation_steps": 4,
#     "gradient_clipping": 1.0,
#     "zero_optimization": {
#         "stage": 2,
#         "offload_optimizer": {
#             "device": "cpu",
#             "pin_memory": true
#         },
#         "allgather_partitions": true,
#         "allgather_bucket_size": 2e8,
#         "overlap_comm": true,
#         "reduce_scatter": true,
#         "reduce_bucket_size": 2e8,
#         "contiguous_gradients": true
#     },
#     "fp16": {
#         "enabled": true,
#         "loss_scale": 0,
#         "loss_scale_window": 1000,
#         "hysteresis": 2,
#         "min_loss_scale": 1
#     }
# }

# # ----- DeepSpeed launch (note: NO --use_8bit_adam) -----
# %cd /kaggle/working/MedDiff-FT/main
# !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#  TOKENIZERS_PARALLELISM=false \
#  accelerate launch --config_file=$ACCELERATE_CONFIG_FILE train_dfb.py \
#     --pretrained_model_name_or_path=/kaggle/working/sd15-inpaint \
#     --instance_data_dir ../data \
#     --output_dir=/kaggle/working/checkpoints/isic2018_dfb_ds \
#     --resolution=512 \
#     --train_batch_size=1 \
#     --gradient_accumulation_steps=4 \
#     --learning_rate=3e-6 \
#     --max_train_steps=2000 \
#     --mixed_precision=fp16 \
#     --gradient_checkpointing \
#     --enable_vae_slicing \
#     --checkpointing_steps=500 \
#     --checkpointing_from=500 \
#     --checkpoints_total_limit=2 \
#     --seed=42 \
#     --use_dfb \
#     --dfb_heads 2 \
#     --dfb_inner_dim_factor 1.0 \
#     --freq_loss_weight 0.1 \
#     --disable_validation