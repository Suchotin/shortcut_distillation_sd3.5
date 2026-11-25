"""Shortcut models (Frans et al., arXiv 2410.12557) on top of a pretrained SD3.5 transformer.

SD3 convention: x_sigma = (1 - sigma) * x_data + sigma * noise (sigma = 1 is noise), the network
predicts v = noise - x_data, and an Euler step is x' = x + (sigma' - sigma) * v. The paper's
formulas are rewritten in this convention.

Step sizes d live on the uniform grid u that SD3 maps to sigma(u) = shift*u / (1 + (shift-1)*u),
so d = 1/n means one of n sampling steps.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

# MMDiT-X attention / MLP projections; `attn2` exists only in the dual-attention blocks, and the
# adaLN projections `norm1*.linear` are where the step-size embedding enters each block.
DEFAULT_LORA_TARGETS = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
    "norm1.linear", "norm1_context.linear",
]


def shift_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * u / (1.0 + (shift - 1.0) * u)


def sampling_sigmas(num_steps: int, shift: float, device=None) -> torch.Tensor:
    """num_steps + 1 sigmas from 1 (noise) to 0 (data)."""
    u = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64, device=device)
    return shift_sigma(u, shift).float()


@dataclass
class ShortcutConfig:
    base_steps: int = 32          # M: the finest grid, d = 0 below it
    shift: float = 3.0            # SD3.5 scheduler shift
    cfg_scale: float = 4.5        # guidance distilled into the student
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_targets: list = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    fm_target: str = "data"       # d = 0 target: "data" (noise - x) or "teacher" (guided velocity)

    @property
    def num_levels(self) -> int:
        k = int(round(math.log2(self.base_steps)))
        assert 2 ** k == self.base_steps, "base_steps must be a power of two"
        return k

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ShortcutConfig":
        with open(path) as f:
            return cls(**json.load(f))


class StepSizeEmbedder(nn.Module):
    """Sinusoidal embedding of d + MLP with a zero-initialised output layer."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.mlp = TimestepEmbedding(freq_dim, dim)
        nn.init.zeros_(self.mlp.linear_2.weight)
        nn.init.zeros_(self.mlp.linear_2.bias)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        e = self.proj(d.float() * 1000.0)
        return self.mlp(e.to(self.mlp.linear_1.weight.dtype))


class ShortcutSD3(nn.Module):
    """SD3 transformer (+ LoRA) conditioned on the step size d.

    The d-embedding is added to the timestep + pooled-text embedding through a forward hook, so
    the transformer's modules and LoRA key names stay unchanged. With a zero d-embedding and zero
    LoRA the model is exactly the teacher.
    """

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer
        cfg = transformer.config
        self.d_embedder = StepSizeEmbedder(cfg.num_attention_heads * cfg.attention_head_dim)
        self._d_emb = None
        transformer.time_text_embed.register_forward_hook(self._add_d_embedding)

    def _add_d_embedding(self, module, inputs, output):
        return output if self._d_emb is None else output + self._d_emb.to(output.dtype)

    def forward(self, x, sigma, d, prompt_embeds, pooled, use_d: bool = True):
        if use_d and d is not None:
            self._d_emb = self.d_embedder(d)
        try:
            return self.transformer(hidden_states=x, timestep=sigma * 1000.0,
                                    encoder_hidden_states=prompt_embeds,
                                    pooled_projections=pooled, return_dict=False)[0]
        finally:
            self._d_emb = None


@contextmanager
def teacher_mode(model: ShortcutSD3):
    """Run the frozen pretrained model: LoRA disabled, no d-embedding."""
    model.transformer.disable_adapters()
    try:
        yield
    finally:
        model.transformer.enable_adapters()


def predict(model, x, sigma, d, pe, pp, cfg_scale=None, null_pe=None, null_pp=None,
            teacher=False):
    """Velocity, optionally with classifier-free guidance against the empty prompt."""
    dtype = pe.dtype

    def run(x_, s_, d_, pe_, pp_):
        if teacher:
            with teacher_mode(model):
                return model(x_.to(dtype), s_, d_, pe_, pp_, use_d=False)
        return model(x_.to(dtype), s_, d_, pe_, pp_)

    if cfg_scale is None:
        return run(x, sigma, d, pe, pp).float()
    b = x.shape[0]
    cond, uncond = run(
        torch.cat([x, x]), torch.cat([sigma, sigma]), torch.cat([d, d]),
        torch.cat([pe, null_pe.expand(b, -1, -1)]), torch.cat([pp, null_pp.expand(b, -1)]),
    ).float().chunk(2)
    return uncond + cfg_scale * (cond - uncond)


def sample_flow_matching(n, cfg: ShortcutConfig, generator=None, device=None):
    """sigma for d = 0 samples, u on the finest grid {1, 1 - 1/M, ..., 1/M}."""
    k = torch.randint(0, cfg.base_steps, (n,), generator=generator)
    sigma = shift_sigma(1.0 - k.double() / cfg.base_steps, cfg.shift)
    return sigma.float().to(device), torch.zeros(n, device=device)


def sample_bootstrap(n, cfg: ShortcutConfig, levels=None, generator=None, device=None,
                     anchor_fraction=None):
    """Self-consistency samples (paper Alg. 1, discrete time sampling).

    Level j trains the step d = 2^j / M from two steps of 2^(j-1) / M, starting at multiples of d.
    Levels are round-robin by default; with `anchor_fraction` level 1, the only level whose target
    comes from the base case d = 0, gets that share and the other levels split the rest.
    """
    levels = list(levels) if levels is not None else list(range(1, cfg.num_levels + 1))
    if anchor_fraction is None:
        offset = int(torch.randint(0, len(levels), (1,), generator=generator))
        j = torch.tensor([levels[(i + offset) % len(levels)] for i in range(n)])
    else:
        rest = [lv for lv in levels if lv != 1]
        w = torch.tensor([anchor_fraction] + [(1 - anchor_fraction) / max(len(rest), 1)] * len(rest),
                         dtype=torch.float64)
        j = torch.tensor([1] + rest)[torch.multinomial(w, n, replacement=True, generator=generator)]
    d = (2.0 ** j.double()) / cfg.base_steps
    k = (torch.rand(n, generator=generator, dtype=torch.float64) * (cfg.base_steps // 2 ** j)).floor()
    u0 = 1.0 - k * d

    def s(u):
        return shift_sigma(u, cfg.shift).float().to(device)

    return dict(sigma0=s(u0), sigma1=s(u0 - d / 2), sigma2=s(u0 - d), level=j.to(device),
                d_student=d.float().to(device),
                d_half=torch.where(j == 1, 0.0, d / 2).float().to(device))


@torch.no_grad()
def self_consistency_target(model, x0, b, pe, pp, null_pe, null_pp, cfg: ShortcutConfig,
                            base_from_teacher=True):
    """Two half-steps of d/2, averaged with their sigma lengths D1, D2 (the grid is not uniform):

        s1 = s(x, sigma0, d/2),  x' = x - D1 * s1,  s2 = s(x', sigma1, d/2)
        target = (D1 * s1 + D2 * s2) / (D1 + D2)

    Level 1 uses the guided base case d = 0, from the frozen teacher or the model itself, so the
    guidance is distilled into every d > 0. Call with the EMA weights swapped in.
    """
    base = b["level"] == 1
    d1 = (b["sigma0"] - b["sigma1"]).view(-1, 1, 1, 1)
    d2 = (b["sigma1"] - b["sigma2"]).view(-1, 1, 1, 1)

    def half_step(x, sigma):
        v = torch.empty_like(x, dtype=torch.float32)
        if base.any():
            i = base.nonzero().squeeze(1)
            v[i] = predict(model, x[i], sigma[i], b["d_half"][i], pe[i], pp[i],
                           cfg_scale=cfg.cfg_scale, null_pe=null_pe, null_pp=null_pp,
                           teacher=base_from_teacher)
        if (~base).any():
            i = (~base).nonzero().squeeze(1)
            v[i] = predict(model, x[i], sigma[i], b["d_half"][i], pe[i], pp[i])
        return v

    s1 = half_step(x0, b["sigma0"])
    s2 = half_step(x0 - d1 * s1, b["sigma1"])
    return (d1 * s1 + d2 * s2) / (d1 + d2)


@torch.no_grad()
def shortcut_sample(model, noise, pe, pp, num_steps, cfg: ShortcutConfig, null_pe=None,
                    null_pp=None, flow_cfg_scale=None):
    """Paper Alg. 2. Fewer than M steps: shortcuts with d = 1/num_steps, one pass per step.

    With M or more steps the d = 0 head is integrated like the original model, with CFG unless
    the guidance was distilled into it (fm_target = "teacher").
    """
    b = noise.shape[0]
    sigmas = sampling_sigmas(num_steps, cfg.shift, device=noise.device)
    shortcut = num_steps < cfg.base_steps
    if shortcut and cfg.base_steps % num_steps:
        warnings.warn(f"{num_steps} steps is not a trained step size", stacklevel=2)
    d = torch.full((b,), 1.0 / num_steps if shortcut else 0.0, device=noise.device)
    if shortcut:
        w = None
    elif flow_cfg_scale is not None:
        w = flow_cfg_scale
    else:
        w = None if cfg.fm_target == "teacher" else cfg.cfg_scale
    x = noise.float()
    for i in range(num_steps):
        v = predict(model, x, sigmas[i].expand(b), d, pe, pp, cfg_scale=w,
                    null_pe=null_pe, null_pp=null_pp)
        x = x + (sigmas[i + 1] - sigmas[i]) * v
    return x


class EMA:
    """EMA of the trainable parameters (LoRA + d-embedder), swapped in place when needed."""

    def __init__(self, params, decay: float):
        self.params = list(params)
        self.decay = decay
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.params], 1.0 - self.decay)

    @contextmanager
    def swapped(self):
        self._swap()
        try:
            yield
        finally:
            self._swap()

    def _swap(self):
        for p, s in zip(self.params, self.shadow):
            p.data, s.data = s.data, p.data

    def state_dict(self):
        return {"decay": self.decay, "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        for s, v in zip(self.shadow, sd["shadow"]):
            s.copy_(v)


def add_lora(transformer, cfg: ShortcutConfig):
    from peft import LoraConfig

    transformer.add_adapter(LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                       init_lora_weights="gaussian",
                                       target_modules=cfg.lora_targets))


def save_student(path: str, model: ShortcutSD3, cfg: ShortcutConfig):
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    lora = {k: v.detach().to(torch.float16).cpu().contiguous()
            for k, v in get_peft_model_state_dict(model.transformer).items()}
    save_file(lora, os.path.join(path, "lora.safetensors"))
    d_emb = {k: v.detach().float().cpu().contiguous()
             for k, v in model.d_embedder.state_dict().items()}
    save_file(d_emb, os.path.join(path, "d_embedder.safetensors"))
    cfg.save(os.path.join(path, "shortcut_config.json"))


def load_student(path: str, transformer) -> tuple[ShortcutSD3, ShortcutConfig]:
    """Attach a trained LoRA + step-size embedder (a directory written by save_student)."""
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    cfg = ShortcutConfig.load(os.path.join(path, "shortcut_config.json"))
    add_lora(transformer, cfg)
    res = set_peft_model_state_dict(transformer, load_file(os.path.join(path, "lora.safetensors")))
    if getattr(res, "unexpected_keys", None):
        raise RuntimeError(f"unexpected LoRA keys: {res.unexpected_keys[:5]}")
    model = ShortcutSD3(transformer)
    model.d_embedder.load_state_dict(load_file(os.path.join(path, "d_embedder.safetensors")))
    p = next(transformer.parameters())
    model.d_embedder.to(device=p.device, dtype=p.dtype)
    return model.eval(), cfg
