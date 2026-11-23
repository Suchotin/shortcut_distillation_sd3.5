"""How much classifier-free guidance a trained student has absorbed.

States x_sigma are built from the teacher data exactly as in training (level-1 start times on
the shifted grid). At each state the frozen teacher gives v_c (prompt) and v_u (empty prompt),
and the student's d = 0 head and its finest shortcut head d = 2/M (level 1, the only level
trained directly against the teacher) are projected onto the guidance direction:

    w_eff(s) = 1 + <s - v_c, v_c - v_u> / ||v_c - v_u||^2        (pooled over all samples)

w_eff = 1: the student behaves like the conditional teacher (no guidance learned);
w_eff = cfg_scale: the guidance is fully distilled. The level-1 target itself is reported as a
reference (two guided Euler half-steps, so close to cfg_scale). Also printed: the level-1
self-consistency loss (what training logs as d=1/16) next to the same loss of the conditional
teacher, which is the loss of the untrained student.

Usage:
  python scripts/diagnose_guidance.py --student runs/<run>/final/student_ema --data_dir data/<set>
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shortcut_sd35.data import LatentDataset, load_null_embeds  # noqa: E402
from shortcut_sd35.shortcut import (load_student, predict, sample_bootstrap,  # noqa: E402
                                    self_consistency_target)

BANDS = [("high noise  (sigma >= 0.75)", 0.75, 1.01), ("mid  (0.4 <= sigma < 0.75)", 0.4, 0.75),
         ("low noise   (sigma < 0.4)", -0.01, 0.4)]


@torch.no_grad()
def guidance_stats(model, cfg, x_data, pe, pp, null_pe, null_pp, generator, batch_size=4,
                   autocast=None):
    """Pooled w_eff of the d = 0 head, the finest shortcut head and the level-1 target, overall
    and per noise band, plus level-1 losses. Tensors may live on any device (moved to pe's)."""
    dev = pe.device
    ctx = autocast or nullcontext
    d_fine = 2.0 / cfg.base_steps
    acc = defaultdict(float)
    for i in range(0, x_data.shape[0], batch_size):
        x0 = x_data[i:i + batch_size].to(dev).float()
        e, p = pe[i:i + batch_size], pp[i:i + batch_size]
        n = x0.shape[0]
        b = sample_bootstrap(n, cfg, levels=[1], generator=generator, device=dev)
        noise = torch.randn(x0.shape, generator=generator).to(dev)
        s0 = b["sigma0"].view(-1, 1, 1, 1)
        x = (1 - s0) * x0 + s0 * noise
        zeros = torch.zeros(n, device=dev)
        with ctx():
            vc = predict(model, x, b["sigma0"], zeros, e, p, teacher=True)
            vu = predict(model, x, b["sigma0"], zeros, null_pe.expand(n, -1, -1),
                         null_pp.expand(n, -1), teacher=True)
            heads = {"d=0": predict(model, x, b["sigma0"], zeros, e, p),
                     f"d={d_fine:g}": predict(model, x, b["sigma0"], torch.full((n,), d_fine,
                                                                                device=dev), e, p)}
            tgt = self_consistency_target(model, x, b, e, p, null_pe, null_pp, cfg,
                                          base_from_teacher=True)
        delta = (vc - vu).flatten(1)
        dd = (delta ** 2).sum(1)
        sig = b["sigma0"]
        for name, s in list(heads.items()) + [("level-1 target", tgt)]:
            dot = ((s - vc).flatten(1) * delta).sum(1)
            acc[(name, "all", "dot")] += float(dot.sum())
            acc[(name, "all", "dd")] += float(dd.sum())
            for band, lo, hi in BANDS:
                m = (sig >= lo) & (sig < hi)
                acc[(name, band, "dot")] += float(dot[m].sum())
                acc[(name, band, "dd")] += float(dd[m].sum())
        h1 = heads[f"d={d_fine:g}"]
        acc["loss_student"] += float(((h1 - tgt) ** 2).flatten(1).mean(1).sum())
        acc["loss_teacher_cond"] += float(((vc - tgt) ** 2).flatten(1).mean(1).sum())
        acc["n"] += n
    names = list(dict.fromkeys(k[0] for k in acc if isinstance(k, tuple)))
    out = {"cfg_scale": cfg.cfg_scale, "n": int(acc["n"]),
           "level1_loss_student": acc["loss_student"] / acc["n"],
           "level1_loss_cond_teacher": acc["loss_teacher_cond"] / acc["n"], "w_eff": {}}
    for name in names:  # None: no samples in that noise band
        out["w_eff"][name] = {band: (1.0 + acc[(name, band, "dot")] / acc[(name, band, "dd")]
                                     if acc[(name, band, "dd")] > 0 else None)
                              for band in ["all"] + [b[0] for b in BANDS]}
    return out


def to_markdown(r, title):
    bands = ["all"] + [b[0] for b in BANDS]
    lines = [f"### {title}", "",
             f"Effective guidance scale w_eff (1 = none learned, {r['cfg_scale']:g} = fully "
             f"distilled), pooled over {r['n']} training states:", "",
             "| | " + " | ".join(bands) + " |", "|---|" + "---:|" * len(bands)]
    for name, w in r["w_eff"].items():
        lines.append(f"| {name} | " + " | ".join("—" if w[b] is None else f"{w[b]:.2f}"
                                                  for b in bands) + " |")
    lines += ["", f"Level-1 self-consistency loss: student {r['level1_loss_student']:.4f}, "
                  f"conditional teacher (= untrained student) {r['level1_loss_cond_teacher']:.4f}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--student", required=True, help="dir with lora/d_embedder/shortcut_config")
    ap.add_argument("--data_dir", required=True, help="teacher data (latents + text embeddings)")
    ap.add_argument("--n", type=int, default=64, help="number of training states")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    args = ap.parse_args(argv)

    from diffusers import SD3Transformer2DModel

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tr = SD3Transformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype)
    model, cfg = load_student(args.student, tr.to(args.device))
    ds = LatentDataset(args.data_dir)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[: args.n].tolist()
    items = [ds[i] for i in idx]
    x = torch.stack([it["latents"] for it in items])
    pe = torch.stack([it["prompt_embeds"] for it in items]).to(args.device, dtype)
    pp = torch.stack([it["pooled"] for it in items]).to(args.device, dtype)
    npe, npp = (t.to(args.device, dtype) for t in load_null_embeds(args.data_dir))
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.device.startswith("cuda") else None
    r = guidance_stats(model, cfg, x, pe, pp, npe, npp, g, batch_size=args.batch_size, autocast=ac)
    r.update(student=args.student, data_dir=args.data_dir, fm_target=cfg.fm_target)
    print(to_markdown(r, args.student))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
    return r


if __name__ == "__main__":
    main()
