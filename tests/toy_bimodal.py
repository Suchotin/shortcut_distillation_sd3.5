"""Toy check (CPU, ~3 min) that the objective learns shortcuts.

A tiny SD3-architecture teacher is trained on bimodal data (all latents +1 or -1), where naive
1-step Euler collapses to the mean (paper Fig. 2). It is then distilled with scripts/train.py
and the few-step samples are compared with the teacher's 32-step ODE solution.

Run:  python tests/toy_bimodal.py [--base_velocity teacher|ema]
"""

import argparse
import json
import os
import sys
import tempfile

import torch
from diffusers import SD3Transformer2DModel
from safetensors.torch import save_file

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.dirname(__file__)]
import test_smoke as T  # noqa: E402
import train  # noqa: E402
from shortcut_sd35.shortcut import (ShortcutConfig, ShortcutSD3, add_lora,  # noqa: E402
                                    load_student, save_student, shortcut_sample)


def bimodal(n, g):
    sign = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return (sign.view(-1, 1, 1, 1) * torch.ones(n, T.C, T.H, T.H)
            + 0.05 * torch.randn(n, T.C, T.H, T.H, generator=g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_steps", type=int, default=1500)
    ap.add_argument("--distill_steps", type=int, default=800)
    ap.add_argument("--base_velocity", default="teacher")
    ap.add_argument("--lr", default="3e-4")
    args = ap.parse_args()
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()
    model_dir, data = T.make_fake_setup(tmp)
    for f in os.listdir(data):
        if f.startswith("shard_"):
            os.remove(os.path.join(data, f))
    g = torch.Generator().manual_seed(0)
    n = 512
    pe1, pp1 = T.rand_cond(1)
    npe, npp = T.rand_cond(1)
    save_file({"latents": bimodal(n, g).half(),
               "prompt_embeds": pe1.expand(n, -1, -1).half().contiguous(),
               "pooled": pp1.expand(n, -1).half().contiguous()},
              os.path.join(data, "shard_00000.safetensors"))
    save_file({"prompt_embeds": npe.half(), "pooled": npp.half()},
              os.path.join(data, "null_embeds.safetensors"))

    # teacher: plain flow matching, logit-normal sigma, 10% prompt dropout
    tr = T.tiny_transformer()
    opt = torch.optim.AdamW(tr.parameters(), lr=1e-3)
    for _ in range(args.pretrain_steps):
        x1 = bimodal(64, g)
        eps = torch.randn_like(x1)
        s = torch.sigmoid(torch.randn(64))
        xs = (1 - s.view(-1, 1, 1, 1)) * x1 + s.view(-1, 1, 1, 1) * eps
        drop = torch.rand(64) < 0.1
        pe = torch.where(drop.view(-1, 1, 1), npe, pe1)
        pp = torch.where(drop.view(-1, 1), npp, pp1)
        v = tr(hidden_states=xs, timestep=s * 1000, encoder_hidden_states=pe,
               pooled_projections=pp, return_dict=False)[0]
        loss = ((v - (eps - x1)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    tr.save_pretrained(os.path.join(model_dir, "transformer"))
    mk = lambda: SD3Transformer2DModel.from_pretrained(os.path.join(model_dir, "transformer"))  # noqa: E731

    X = torch.randn(256, T.C, T.H, T.H, generator=torch.Generator().manual_seed(5))
    PE, PP = pe1.expand(256, -1, -1), pp1.expand(256, -1)

    def on_mode(x):  # share of samples on one of the two modes
        return ((x.mean(dim=(1, 2, 3)).abs() - 1).abs() < 0.25).float().mean().item()

    def report(st, cfg, ref):
        with torch.no_grad():
            r = {}
            for k in (1, 2, 4):
                xs = shortcut_sample(st, X, PE, PP, k, cfg)
                r[f"{k} step"] = {"mse_to_teacher_32step": round(((xs - ref) ** 2).mean().item(), 4),
                                  "frac_on_mode": round(on_mode(xs), 3)}
        return r

    # zero LoRA and zero d-embedding: the teacher itself, sampled with naive Euler
    c0 = ShortcutConfig(base_steps=8, shift=3.0, cfg_scale=1.0, lora_rank=16, lora_alpha=16)
    t0 = mk()
    add_lora(t0, c0)
    for name, p in t0.named_parameters():
        if "lora_B" in name:
            torch.nn.init.zeros_(p)
    save_student(os.path.join(tmp, "init"), ShortcutSD3(t0), c0)
    teacher, _ = load_student(os.path.join(tmp, "init"), mk())
    with torch.no_grad():
        ref = shortcut_sample(teacher, X, PE, PP, 32, c0, null_pe=npe, null_pp=npp)
    before = report(teacher, c0, ref)

    out = os.path.join(tmp, "run")
    train.main(["--model", model_dir, "--data_dir", data, "--output_dir", out,
                "--base_steps", "8", "--lora_rank", "16", "--lora_alpha", "16",
                "--batch_size", "16", "--grad_accum", "1", "--bootstrap_fraction", "0.5",
                "--anchor_fraction", "0", "--fm_target", "data", "--num_workers", "0",
                "--log_every", "200", "--val_every", "100000", "--save_every", "100000",
                "--lr", args.lr, "--warmup_steps", "0", "--ema_decay", "0.99",
                "--cfg_scale", "1.0", "--weight_decay", "0.0", "--mixed_precision", "no",
                "--max_steps", str(args.distill_steps), "--base_velocity", args.base_velocity])
    student, cfg = load_student(os.path.join(out, "final", "student_ema"), mk())
    after = report(student, cfg, ref)
    print(f"teacher 32-step ODE: frac_on_mode {on_mode(ref):.3f}")
    print("teacher, naive few-step Euler:", json.dumps(before))
    print(f"shortcut student ({args.base_velocity}):", json.dumps(after))


if __name__ == "__main__":
    main()
