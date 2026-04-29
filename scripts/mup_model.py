"""
μP (Maximal Update Parameterization) GPT — Part 3 of the SVG scaling study.

Key differences from model.py (Standard Parameterization):
  1. Attention logit scale: 1/head_dim  (SP uses 1/sqrt(head_dim))
  2. Output layer: MuReadout            (SP uses plain nn.Linear; MuReadout scales LR by 1/width)
  3. No input↔output weight tying       (μP assigns different LR multipliers to each)
  4. set_base_shapes() registered       (tells MuAdamW which dims are "infinite" → per-layer LR)

Usage:
    from scripts.mup_model import MupGPT, build_mup_model
    from scripts.model import ModelConfig

    model = build_mup_model(config)          # properly set-up μP model
    optimizer = MuAdamW(model.parameters())  # LR-transfer-aware optimizer
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from mup import MuReadout, set_base_shapes

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.model import ModelConfig, MLP


# ---------------------------------------------------------------------------
# Reference width used to define μP base shapes (must be small enough that
# every target model is strictly wider).
# ---------------------------------------------------------------------------
_BASE_D_MODEL  = 64    # base model width
_DELTA_D_MODEL = 128   # delta model width (2× base)
_BASE_N_HEADS  = 4     # divides both 64 and 128
_BASE_D_FF_MULTIPLIER = 4  # d_ff = d_model * 4  (same ratio as target configs)


# ---------------------------------------------------------------------------
# Attention with μP scaling
# ---------------------------------------------------------------------------

class MupCausalSelfAttention(nn.Module):
    """Causal self-attention with μP logit scaling: 1/head_dim (not 1/√head_dim)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads  = config.n_heads
        self.d_model  = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        self.c_attn   = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj   = nn.Linear(config.d_model, config.d_model,     bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

        # μP key change: scale by 1/head_dim instead of 1/√head_dim
        self._mup_scale = 1.0 / self.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Try flash attention with explicit μP scale (PyTorch ≥ 2.1)
        if hasattr(F, "scaled_dot_product_attention"):
            try:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                    scale=self._mup_scale,
                )
            except TypeError:
                # scale kwarg not supported → manual fallback
                y = self._manual_attn(q, k, v, T, x.device)
        else:
            y = self._manual_attn(q, k, v, T, x.device)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

    def _manual_attn(self, q, k, v, T, device):
        att = (q @ k.transpose(-2, -1)) * self._mup_scale  # μP: 1/head_dim
        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.training and self.dropout_p > 0:
            att = F.dropout(att, p=self.dropout_p)
        return att @ v


class MupBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = MupCausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# μP GPT
# ---------------------------------------------------------------------------

class MupGPT(nn.Module):
    """
    Decoder-only GPT with μP.

    Differences from GPT in model.py:
      - MupCausalSelfAttention (1/head_dim scale)
      - MuReadout output layer (LR multiplied by base_width/target_width via MuAdamW)
      - No weight tying between wte and lm_head
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.d_model),
            wpe  = nn.Embedding(config.block_size, config.d_model),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([MupBlock(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model, bias=config.bias),
        ))
        # MuReadout: output linear whose LR is automatically scaled by MuAdamW
        # NOT tied to wte — μP requires separate embedding and readout LR scaling
        self.lm_head = MuReadout(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 style: scale residual branch projections by 1/√(2·n_layers)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        return self.lm_head(x[:, [-1], :]), None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config_file(cls, path: str) -> "MupGPT":
        with open(path) as f:
            cfg = json.load(f)
        return cls(ModelConfig.from_dict(cfg))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_id: int = 2,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = logits.scatter(1, sorted_idx, sorted_logits)

            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break

        return idx


# ---------------------------------------------------------------------------
# Factory: create a properly μP-initialized model
# ---------------------------------------------------------------------------

def _make_base_config(target_config: ModelConfig, d_model: int) -> ModelConfig:
    """Build a minimal-width config with the same depth as the target."""
    return ModelConfig(
        vocab_size  = target_config.vocab_size,
        block_size  = target_config.block_size,
        d_model     = d_model,
        n_layers    = target_config.n_layers,   # must match target so param names align
        n_heads     = _BASE_N_HEADS,            # 4 divides both 64 and 128
        d_ff        = d_model * _BASE_D_FF_MULTIPLIER,
        dropout     = target_config.dropout,
        bias        = target_config.bias,
    )


def build_mup_model(config: ModelConfig) -> MupGPT:
    """
    Create a MupGPT with μP base shapes registered.

    Builds a base (d_model=64) and delta (d_model=128) model with the same
    depth as `config`, then calls set_base_shapes so that MuAdamW can apply
    the correct per-parameter LR multiplier: lr × (base_width / current_width).
    """
    base_cfg  = _make_base_config(config, _BASE_D_MODEL)
    delta_cfg = _make_base_config(config, _DELTA_D_MODEL)

    base_model  = MupGPT(base_cfg)
    delta_model = MupGPT(delta_cfg)
    model       = MupGPT(config)

    # rescale_params=False: keep GPT-2 style init; MuAdamW handles LR scaling
    set_base_shapes(model, base_model, delta=delta_model, rescale_params=False)

    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/tiny.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = ModelConfig.from_dict(cfg_dict)

    model = build_mup_model(config)
    n = model.count_parameters()
    cfg = model.config
    print(f"μP model: {cfg_path}")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"  Parameters: {n:,} ({n/1e6:.2f}M)")
    print(f"  Attention scale: 1/{cfg.d_model // cfg.n_heads} (μP) "
          f"vs 1/√{cfg.d_model // cfg.n_heads} (SP)")
    print("  Base shapes registered: OK")
