"""How much classifier-free guidance a student has absorbed.

At training states x_sigma the frozen teacher gives v_c (prompt) and v_u (empty prompt). A
prediction s is projected onto the guidance direction:

    w_eff(s) = 1 + <s - v_c, v_c - v_u> / ||v_c - v_u||^2

w_eff = 1 means no guidance was learned, w_eff = cfg_scale means all of it. Reported for the
d = 0 head, the finest shortcut head d = 2/M and its training target, overall and per noise band.

Usage:
  python scripts/diagnose_guidance.py --student runs/shortcut_1024/final/student_ema --data_dir DATA
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

BANDS = {"high (sigma >= 0.75)": (0.75, 1.01), "mid": (0.4, 0.75), "low (sigma < 0.4)": (-0.01, 0.4)}


@torch.no_grad()
def guidance_stats(model, cfg, x_data, pe, pp, null_pe, null_pp, generator, batch_size=4,
                   autocast=None):
    dev = pe.device
    d_fine = 2.0 / cfg.base_steps
    acc = defaultdict(float)
    for i in range(0, x_data.shape[0], batch_size):
        x0 = x_data[i:i + batch_size].to(dev).float()
        e, p = pe[i:i + batch_size], pp[i:i + batch_size]
        n = x0.shape[0]
        b = sample_bootstrap(n, cfg, levels=[1], generator=generator, device=dev)
        s0 = b["sigma0"].view(-1, 1, 1, 1)
        x = (1 - s0) * x0 + s0 * torch.randn(x0.shape, generator=generator).to(dev)
        zeros = torch.zeros(n, device=dev)
        with (autocast or nullcontext)():
            vc = predict(model, x, b["sigma0"], zeros, e, p, teacher=True)
            vu = predict(model, x, b["sigma0"], zeros, null_pe.expand(n, -1, -1),
                         null_pp.expand(n, -1), teacher=True)
            heads = {"d=0": predict(model, x, b["sigma0"], zeros, e, p),
                     f"d={d_fine:g}": predict(model, x, b["sigma0"], torch.full((n,), d_fine, device=dev),
                                              e, p),
                     "target": self_consistency_target(model, x, b, e, p, null_pe, null_pp, cfg)}
        delta = (vc - vu).flatten(1)
        dd = (delta ** 2).sum(1)
        bands = {"all": torch.ones(n, dtype=torch.bool, device=dev)}
        bands.update({k: (b["sigma0"] >= lo) & (b["sigma0"] < hi) for k, (lo, hi) in BANDS.items()})
        for name, s in heads.items():
            dot = ((s - vc).flatten(1) * delta).sum(1)
            for band, m in bands.items():
                acc[(name, band, "dot")] += float(dot[m].sum())
                acc[(name, band, "dd")] += float(dd[m].sum())
        tgt = heads["target"]
        acc["loss_student"] += float(((heads[f"d={d_fine:g}"] - tgt) ** 2).flatten(1).mean(1).sum())
        acc["loss_teacher"] += float(((vc - tgt) ** 2).flatten(1).mean(1).sum())
        acc["n"] += n
    w_eff = {name: {band: 1 + acc[(name, band, "dot")] / acc[(name, band, "dd")]
                    if acc[(name, band, "dd")] > 0 else None for band in ["all", *BANDS]}
             for name in heads}
    return {"cfg_scale": cfg.cfg_scale, "n": int(acc["n"]), "w_eff": w_eff,
            "level1_loss_student": acc["loss_student"] / acc["n"],
            "level1_loss_cond_teacher": acc["loss_teacher"] / acc["n"]}


def to_markdown(r):
    bands = ["all", *BANDS]
    lines = ["| head | " + " | ".join(bands) + " |", "|---|" + "---:|" * len(bands)]
    for name, w in r["w_eff"].items():
        cells = ("-" if w[b] is None else f"{w[b]:.2f}" for b in bands)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append(f"\nlevel-1 loss: student {r['level1_loss_student']:.4f}, "
                 f"conditional teacher {r['level1_loss_cond_teacher']:.4f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--student", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=64, help="number of training states")
    ap.add_argument("--batch_size", type=int, default=4, help="changes which states are drawn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args(argv)

    from diffusers import SD3Transformer2DModel

    cuda = args.device.startswith("cuda")
    dtype = torch.bfloat16 if cuda else torch.float32
    tr = SD3Transformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype)
    model, cfg = load_student(args.student, tr.to(args.device))
    ds = LatentDataset(args.data_dir)
    g = torch.Generator().manual_seed(args.seed)
    items = [ds[i] for i in torch.randperm(len(ds), generator=g)[:args.n].tolist()]
    x = torch.stack([it["latents"] for it in items])
    pe = torch.stack([it["prompt_embeds"] for it in items]).to(args.device, dtype)
    pp = torch.stack([it["pooled"] for it in items]).to(args.device, dtype)
    npe, npp = (t.to(args.device, dtype) for t in load_null_embeds(args.data_dir))
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if cuda else None
    r = guidance_stats(model, cfg, x, pe, pp, npe, npp, g, batch_size=args.batch_size, autocast=ac)
    print(to_markdown(r))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
    return r


if __name__ == "__main__":
    main()
