"""Shortcut models (Frans et al., arXiv 2410.12557) on top of a pretrained SD3.5 transformer.

Conventions (diffusers / SD3):
    x_sigma = (1 - sigma) * x_data + sigma * noise,   sigma = 1 is pure noise
    the network predicts the velocity  v = noise - x_data
    an Euler step from sigma to sigma' < sigma is  x' = x + (sigma' - sigma) * v

The paper uses the opposite time direction (t = 0 is noise, v = x1 - x0); every formula
below is the paper's formula rewritten in the SD3 convention.

Time grid. SD3 samples on a shifted grid sigma(u) = shift*u / (1 + (shift-1)*u) with u uniform.
Step sizes d live in u-space ("grid units"), so d = 1/n means "one of n sampling steps".
Because the sigma-grid is non-uniform, the self-consistency target is the average of the two
half-step velocities weighted by the sigma-length of each half-step (for a uniform grid this
reduces exactly to the paper's (s1 + s2) / 2).

Step-size conditioning follows SAM 3D (arXiv 2511.16624, App. C.4): a new embedder for d whose
final linear layer is zero-initialised, so at initialisation the model equals the teacher. Its
output is added to SD3's timestep+pooled-text embedding (`temb`), which drives all adaLN
modulations, exactly like DiT shortcut models add the d-embedding to the t-embedding.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

# LoRA targets for SD3.5 Medium (MMDiT-X). `attn2` is the extra image self-attention that
# exists only in the first dual-attention blocks; adaLN modulation layers (`norm1*.linear`) are
# included because they are where the d-embedding enters every block.
DEFAULT_LORA_TARGETS = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
    "norm1.linear", "norm1_context.linear",
]


# --------------------------------------------------------------------------------------------
# Time grid
# --------------------------------------------------------------------------------------------
def shift_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    """Map uniform time u in [0, 1] (1 = noise) to SD3's shifted sigma."""
    return shift * u / (1.0 + (shift - 1.0) * u)


def sampling_sigmas(num_steps: int, shift: float, device=None) -> torch.Tensor:
    """num_steps + 1 sigmas from 1 (noise) to 0 (data) on the shifted uniform grid."""
    u = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64, device=device)
    return shift_sigma(u, shift).float()


@dataclass
class ShortcutConfig:
    base_steps: int = 32          # M: finest grid; d_min = 1/M is the flow-matching base case
    shift: float = 3.0            # SD3.5 scheduler shift
    cfg_scale: float = 4.5        # CFG baked into the smallest bootstrap level
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_targets: list = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))

    @property
    def num_levels(self) -> int:  # number of bootstrap levels: student d in {2/M, ..., 1}
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


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
class StepSizeEmbedder(nn.Module):
    """Sinusoidal embedding of d followed by an MLP; final layer zero-initialised (SAM 3D)."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        # same embedding family as SD3's own timestep embedder
        self.proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.mlp = TimestepEmbedding(freq_dim, dim)
        nn.init.zeros_(self.mlp.linear_2.weight)
        nn.init.zeros_(self.mlp.linear_2.bias)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        e = self.proj(d.float() * 1000.0)  # d in [0, 1], scaled like SD3 timesteps
        return self.mlp(e.to(self.mlp.linear_1.weight.dtype))


class ShortcutSD3(nn.Module):
    """s_theta(x, sigma, d): SD3 transformer (+LoRA) conditioned on step size d.

    d = 0 is the instantaneous flow (the pretrained model's regime); d > 0 are shortcuts.
    The d-embedding is added to `time_text_embed`'s output via a forward hook, so the
    transformer's module tree (and LoRA key names) stays untouched.
    """

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer
        cfg = transformer.config
        dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.d_embedder = StepSizeEmbedder(dim)
        self._d_emb = None
        self._hook = transformer.time_text_embed.register_forward_hook(self._add_d_embedding)

    def _add_d_embedding(self, module, inputs, output):
        if self._d_emb is None:
            return output
        return output + self._d_emb.to(output.dtype)

    def forward(self, x, sigma, d, prompt_embeds, pooled, use_d: bool = True):
        """Returns the shortcut velocity (noise - data convention), same shape as x.

        x: (B, C, H, W); sigma: (B,) in [0, 1]; d: (B,) step size in grid units or None.
        """
        if use_d and d is not None:
            self._d_emb = self.d_embedder(d)
        try:
            return self.transformer(
                hidden_states=x,
                timestep=sigma * 1000.0,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]
        finally:
            self._d_emb = None


@contextmanager
def teacher_mode(model: ShortcutSD3):
    """The frozen pretrained model: LoRA adapters disabled and no d-embedding."""
    model.transformer.disable_adapters()
    try:
        yield
    finally:
        model.transformer.enable_adapters()


def predict(model, x, sigma, d, pe, pp, cfg_scale=None, null_pe=None, null_pp=None,
            teacher=False):
    """Velocity with optional classifier-free guidance (uncond = empty-prompt embeddings)."""
    dtype = pe.dtype
    use_d = not teacher

    def run(x_, s_, d_, pe_, pp_):
        if teacher:
            with teacher_mode(model):
                return model(x_.to(dtype), s_, d_, pe_, pp_, use_d=False)
        return model(x_.to(dtype), s_, d_, pe_, pp_, use_d=use_d)

    if cfg_scale is None:
        return run(x, sigma, d, pe, pp).float()
    b = x.shape[0]
    out = run(
        torch.cat([x, x]), torch.cat([sigma, sigma]), torch.cat([d, d]),
        torch.cat([pe, null_pe.expand(b, -1, -1)]), torch.cat([pp, null_pp.expand(b, -1)]),
    ).float()
    cond, uncond = out.chunk(2)
    return uncond + cfg_scale * (cond - uncond)


# --------------------------------------------------------------------------------------------
# Training targets
# --------------------------------------------------------------------------------------------
def sample_flow_matching(n, cfg: ShortcutConfig, generator=None, device=None):
    """Flow-matching (d = 0) samples: u on the finest grid {1, 1 - 1/M, ..., 1/M}."""
    k = torch.randint(0, cfg.base_steps, (n,), generator=generator)
    u = 1.0 - k.double() / cfg.base_steps
    sigma = shift_sigma(u, cfg.shift).float().to(device)
    return sigma, torch.zeros(n, device=device)


def sample_bootstrap(n, cfg: ShortcutConfig, levels=None, generator=None, device=None):
    """Self-consistency samples (paper Alg. 1 + "discrete time sampling").

    Level j in 1..K trains the student step d = 2^j / M from two half-steps of 2^(j-1) / M.
    Start times are multiples of the student step: u in {1, 1 - d, ..., d}.
    Levels are assigned round-robin (as in the official code) so every batch covers them evenly.
    Returns per-sample sigma at start / middle / end, student d, half-step d, and the level.
    """
    levels = list(levels) if levels is not None else list(range(1, cfg.num_levels + 1))
    offset = int(torch.randint(0, len(levels), (1,), generator=generator))
    j = torch.tensor([levels[(i + offset) % len(levels)] for i in range(n)])
    d_student = (2.0 ** j.double()) / cfg.base_steps
    n_slots = (cfg.base_steps // (2 ** j)).long()
    k = (torch.rand(n, generator=generator, dtype=torch.float64) * n_slots).floor()
    u0 = 1.0 - k * d_student
    u1 = u0 - d_student / 2
    u2 = u0 - d_student
    s = lambda u: shift_sigma(u, cfg.shift).float().to(device)  # noqa: E731
    d_half = torch.where(j == 1, torch.zeros_like(d_student), d_student / 2)  # level 1 -> d = 0
    return dict(sigma0=s(u0), sigma1=s(u1), sigma2=s(u2), level=j.to(device),
                d_student=d_student.float().to(device), d_half=d_half.float().to(device))


@torch.no_grad()
def self_consistency_target(model, x0, b, pe, pp, null_pe, null_pp, cfg: ShortcutConfig,
                            base_from_teacher=False, clip=None):
    """s_target = (D1 * s(x, sigma0, d/2) + D2 * s(x', sigma1, d/2)) / (D1 + D2),
    with x' = x - D1 * s(x, sigma0, d/2), D1 = sigma0 - sigma1, D2 = sigma1 - sigma2.

    The second half-step is queried at the *next* time sigma1 (paper Alg. 1; Eq. 5 has a typo).
    Level-1 half-steps are the base case d = 0 and use CFG (guidance distilled into all d > 0,
    as in the official code, MAGI-1 and SAM 3D). Optionally the base case can be taken from the
    frozen teacher instead of the student's own (EMA) d = 0 prediction.
    `model` should carry the EMA weights when called (paper: EMA targets).
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
    x_mid = x0 - d1 * s1
    if clip is not None:
        x_mid = x_mid.clamp(-clip, clip)
    s2 = half_step(x_mid, b["sigma1"])
    target = (d1 * s1 + d2 * s2) / (d1 + d2)
    if clip is not None:
        target = target.clamp(-clip, clip)
    return target


# --------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def shortcut_sample(model, noise, pe, pp, num_steps, cfg: ShortcutConfig, null_pe=None,
                    null_pp=None, flow_cfg_scale=None):
    """Paper Alg. 2 on the shifted grid.

    num_steps < M: shortcut mode, d = 1/num_steps, one forward pass per step, no CFG.
    num_steps >= M: the d = 0 flow model with CFG (behaves like the original SD3.5).
    """
    b = noise.shape[0]
    sigmas = sampling_sigmas(num_steps, cfg.shift, device=noise.device)
    shortcut = num_steps < cfg.base_steps
    if shortcut and cfg.base_steps % num_steps != 0:
        print(f"[warn] {num_steps} steps is not a trained shortcut level (powers of two < M)")
    d = torch.full((b,), 1.0 / num_steps if shortcut else 0.0, device=noise.device)
    w = None if shortcut else (flow_cfg_scale or cfg.cfg_scale)
    x = noise.float()
    for i in range(num_steps):
        sigma = sigmas[i].expand(b)
        v = predict(model, x, sigma, d, pe, pp, cfg_scale=w, null_pe=null_pe, null_pp=null_pp)
        x = x + (sigmas[i + 1] - sigmas[i]) * v
    return x


# --------------------------------------------------------------------------------------------
# EMA over trainable params only (LoRA + d-embedder); EMA of LoRA deltas == EMA of the model
# --------------------------------------------------------------------------------------------
class EMA:
    def __init__(self, params, decay: float):
        self.params = list(params)
        self.decay = decay
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.params], 1.0 - self.decay)

    @contextmanager
    def swapped(self):
        """Temporarily run the model with EMA weights (pointer swap, no copies)."""
        self._swap()
        try:
            yield
        finally:
            self._swap()

    def _swap(self):
        for p, s in zip(self.params, self.shadow):
            tmp = p.data
            p.data = s.data
            s.data = tmp

    def state_dict(self):
        return {"decay": self.decay, "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        for s, v in zip(self.shadow, sd["shadow"]):
            s.copy_(v)


# --------------------------------------------------------------------------------------------
# LoRA setup, save / load
# --------------------------------------------------------------------------------------------
def add_lora(transformer, cfg: ShortcutConfig):
    from peft import LoraConfig

    transformer.add_adapter(LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, init_lora_weights="gaussian",
        target_modules=cfg.lora_targets,
    ))


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
    """Attach a trained shortcut LoRA + d-embedder to a (frozen) SD3 transformer."""
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    cfg = ShortcutConfig.load(os.path.join(path, "shortcut_config.json"))
    add_lora(transformer, cfg)
    lora = load_file(os.path.join(path, "lora.safetensors"))
    res = set_peft_model_state_dict(transformer, lora)
    unexpected = getattr(res, "unexpected_keys", None)
    if unexpected:
        raise RuntimeError(f"unexpected LoRA keys: {unexpected[:5]}")
    model = ShortcutSD3(transformer)
    model.d_embedder.load_state_dict(load_file(os.path.join(path, "d_embedder.safetensors")))
    p = next(transformer.parameters())
    model.d_embedder.to(device=p.device, dtype=p.dtype)
    return model.eval(), cfg
