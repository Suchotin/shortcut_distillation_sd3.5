"""Shortcut distillation of SD3.5 Medium with LoRA (Frans et al. 2410.12557, SAM 3D-style init).

Each batch (paper Eq. 5 / Alg. 1):
  * (1 - k) of the samples: flow matching at d = 0 on teacher-generated data
        loss = || s(x_sigma, sigma, 0) - (noise - x_data) ||^2       (+ prompt dropout for CFG)
  * k of the samples: self-consistency at a random level j (student step d = 2^j / M)
        loss = || s(x_sigma, sigma, d) - stopgrad(EMA two half-steps of d/2) ||^2
    Level 1's half-steps are the d = 0 base case evaluated with CFG, which distills guidance
    into every d > 0, so shortcut sampling needs one forward pass per step. By default that
    base case is the frozen teacher (--base_velocity teacher); --base_velocity ema uses the
    student's own EMA d = 0 prediction exactly as in the from-scratch paper.

Usage (single GPU):
  accelerate launch scripts/train.py --data_dir data/sd35m_512 --output_dir runs/shortcut
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shortcut_sd35.data import (LatentDataset, load_meta, load_null_embeds,  # noqa: E402
                                load_showcase)
from shortcut_sd35.shortcut import (EMA, ShortcutConfig, ShortcutSD3, add_lora,  # noqa: E402
                                    sample_bootstrap, sample_flow_matching, save_student,
                                    self_consistency_target, shortcut_sample)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="runs/shortcut")
    # shortcut
    ap.add_argument("--base_steps", type=int, default=32, help="M; d_min = 1/M (paper: 128)")
    ap.add_argument("--cfg_scale", type=float, default=None, help="default: teacher CFG from meta")
    ap.add_argument("--bootstrap_fraction", type=float, default=0.25, help="k (paper: 1/4)")
    ap.add_argument("--min_shortcut_steps", type=int, default=1,
                    help="train levels down to this many sampling steps (1 = all levels)")
    ap.add_argument("--base_velocity", choices=["ema", "teacher"], default="teacher",
                    help="d=0 half-steps of level 1 (base case of the recursion): frozen teacher "
                         "(distillation; avoids drift of the student's own d=0 head and double "
                         "CFG on CFG-generated data) or the student's own EMA (paper, from scratch)")
    ap.add_argument("--prompt_dropout", type=float, default=0.1)
    ap.add_argument("--target_clip", type=float, default=None, help="official code uses 4")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    # LoRA / optimisation
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--mixed_precision", default="bf16")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    # logging
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--val_steps", default="1,2,4,8")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--keep_train_states", type=int, default=1,
                    help="keep optimizer/EMA state (~2 GB each) only in the last N checkpoints; "
                         "student weights are kept in all of them")
    ap.add_argument("--resume", default=None, help="checkpoint dir with train_state.pt")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def decode(vae, latents):
    x = latents / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(x.to(vae.dtype), return_dict=False)[0]
    img = (img.float() / 2 + 0.5).clamp(0, 1)
    return (img.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


def save_grid(rows, path):
    """rows: list of uint8 arrays [n, H, W, 3] -> one image, rows stacked vertically."""
    grid = np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)
    Image.fromarray(grid).save(path)


@torch.no_grad()
def validate(model, ema, vae, showcase, null, cfg, steps_list, latent_shape, path, accelerator):
    pe, pp = showcase
    g = torch.Generator("cpu").manual_seed(1234)
    noise = torch.randn((pe.shape[0], *latent_shape), generator=g).to(pe.device)
    rows = []
    with ema.swapped(), accelerator.autocast():
        for n in steps_list:
            x = shortcut_sample(model, noise, pe, pp, n, cfg, null_pe=null[0], null_pp=null[1])
            rows.append(decode(vae, x))
    save_grid(rows, path)


def main(argv=None):
    args = parse_args(argv)
    torch.set_float32_matmul_precision("high")  # TF32 for the remaining fp32 matmuls (Ampere+)
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=args.grad_accum)
    set_seed(args.seed, device_specific=True)
    device = accelerator.device
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (
        torch.float16 if args.mixed_precision == "fp16" else torch.float32)

    meta = load_meta(args.data_dir)
    cfg = ShortcutConfig(base_steps=args.base_steps, shift=meta["shift"],
                         cfg_scale=args.cfg_scale if args.cfg_scale is not None else meta["cfg"],
                         lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    levels = [j for j in range(1, cfg.num_levels + 1)
              if 2 ** j <= cfg.base_steps // args.min_shortcut_steps]

    # ---- model: frozen SD3.5 + LoRA + zero-init step-size embedder ----
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=weight_dtype)
    transformer.requires_grad_(False)
    add_lora(transformer, cfg)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    model = ShortcutSD3(transformer).to(device)
    for p in model.parameters():  # trainable params in fp32 (LoRA + d-embedder)
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n)
    n_d = sum(p.numel() for p in model.d_embedder.parameters())
    accelerator.print(f"trainable: LoRA {n_lora / 1e6:.1f}M + d-embedder {n_d / 1e6:.1f}M; "
                      f"levels {levels} (student d = {[2 ** j / cfg.base_steps for j in levels]})")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_scheduler("constant_with_warmup", opt,
                          num_warmup_steps=args.warmup_steps * accelerator.num_processes,
                          num_training_steps=args.max_steps * accelerator.num_processes)
    ds = LatentDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, pin_memory=True,
                    persistent_workers=args.num_workers > 0)
    model, opt, dl, sched = accelerator.prepare(model, opt, dl, sched)
    net = accelerator.unwrap_model(model)
    ema = EMA(params, args.ema_decay)

    null_pe, null_pp = (t.to(device, weight_dtype) for t in load_null_embeds(args.data_dir))
    step = 0
    if args.resume:
        st = torch.load(os.path.join(args.resume, "train_state.pt"), map_location="cpu")
        for p, v in zip(params, st["params"]):
            p.data.copy_(v)
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["lr_scheduler"])
        step = st["step"]
        accelerator.print(f"resumed from {args.resume} at step {step}")

    vae = showcase = None
    steps_list = [int(s) for s in args.val_steps.split(",")]
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae",
                                            torch_dtype=weight_dtype).to(device)
        _, sc_pe, sc_pp = load_showcase(args.data_dir)
        showcase = (sc_pe.to(device, weight_dtype), sc_pp.to(device, weight_dtype))

    def run_validation():
        if accelerator.is_main_process:
            validate(net, ema, vae, showcase, (null_pe, null_pp), cfg, steps_list,
                     meta["latent_shape"],
                     os.path.join(args.output_dir, "samples", f"step_{step:06d}.png"),
                     accelerator)

    def save(tag):
        if not accelerator.is_main_process:
            return
        d = os.path.join(args.output_dir, tag)
        save_student(os.path.join(d, "student"), net, cfg)
        with ema.swapped():
            save_student(os.path.join(d, "student_ema"), net, cfg)
        torch.save({"step": step, "params": [p.detach().cpu() for p in params],
                    "ema": ema.state_dict(), "optimizer": opt.state_dict(),
                    "lr_scheduler": sched.state_dict()}, os.path.join(d, "train_state.pt"))
        accelerator.print(f"saved {d}")
        ckpts = sorted((int(n.split("-")[1]), n) for n in os.listdir(args.output_dir)
                       if n.startswith("checkpoint-") and n.split("-")[1].isdigit())
        for _, n in ckpts[: max(len(ckpts) - args.keep_train_states, 0)]:
            old = os.path.join(args.output_dir, n, "train_state.pt")
            if os.path.exists(old):
                os.remove(old)

    if step == 0:
        run_validation()  # step 0 == the teacher sampled with few steps and no CFG

    gen = torch.Generator("cpu").manual_seed(args.seed + accelerator.process_index)
    stats, t_last = defaultdict(lambda: [0.0, 0]), time.time()
    log_f = open(os.path.join(args.output_dir, "log.jsonl"), "a") if accelerator.is_main_process else None
    base_from_teacher = args.base_velocity == "teacher"
    n_epochs = math.ceil(args.max_steps * args.grad_accum / len(dl)) + 1

    for _ in range(n_epochs):
        for batch in dl:
            with accelerator.accumulate(model):
                x_data = batch["latents"].float()
                pe = batch["prompt_embeds"].to(weight_dtype)
                pp = batch["pooled"].to(weight_dtype)
                bsz = x_data.shape[0]
                nb = int(round(bsz * args.bootstrap_fraction))
                noise = torch.randn_like(x_data)

                # flow-matching part (d = 0), with prompt dropout to keep the uncond branch
                s_fm, d_fm = sample_flow_matching(bsz - nb, cfg, gen, device)
                drop = (torch.rand(bsz - nb, generator=gen) < args.prompt_dropout).nonzero()
                if len(drop):
                    i = nb + drop.squeeze(1).to(device)
                    pe, pp = pe.clone(), pp.clone()
                    pe[i], pp[i] = null_pe.expand(len(i), -1, -1), null_pp.expand(len(i), -1)
                sv = s_fm.view(-1, 1, 1, 1)
                x_fm = (1 - sv) * x_data[nb:] + sv * noise[nb:]
                tgt_fm = noise[nb:] - x_data[nb:]

                # self-consistency part (d > 0), EMA two-half-step targets
                if nb > 0:
                    b = sample_bootstrap(nb, cfg, levels, gen, device)
                    s0 = b["sigma0"].view(-1, 1, 1, 1)
                    x_b = (1 - s0) * x_data[:nb] + s0 * noise[:nb]
                    with ema.swapped(), accelerator.autocast():
                        tgt_b = self_consistency_target(
                            net, x_b, b, pe[:nb], pp[:nb], null_pe, null_pp, cfg,
                            base_from_teacher=base_from_teacher, clip=args.target_clip)
                    x_in = torch.cat([x_b, x_fm])
                    sigma = torch.cat([b["sigma0"], s_fm])
                    d = torch.cat([b["d_student"], d_fm])
                    target = torch.cat([tgt_b, tgt_fm])
                else:
                    x_in, sigma, d, target = x_fm, s_fm, d_fm, tgt_fm

                with accelerator.autocast():
                    pred = model(x_in.to(weight_dtype), sigma, d, pe, pp).float()
                per_sample = ((pred - target) ** 2).mean(dim=(1, 2, 3))
                loss = per_sample.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gn = accelerator.clip_grad_norm_(params, args.max_grad_norm)
                    if gn is not None:  # total norm before clipping
                        stats["grad_norm"][0] += float(gn)
                        stats["grad_norm"][1] += 1
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            ps = per_sample.detach().cpu()
            stats["loss"][0] += ps.mean().item()
            stats["loss"][1] += 1
            stats["fm"][0] += ps[nb:].sum().item()
            stats["fm"][1] += bsz - nb
            if nb > 0:
                for lv, v in zip(b["level"].cpu().tolist(), ps[:nb].tolist()):
                    stats[f"d=1/{cfg.base_steps // 2 ** lv}"][0] += v
                    stats[f"d=1/{cfg.base_steps // 2 ** lv}"][1] += 1

            if not accelerator.sync_gradients:
                continue
            ema.update()
            step += 1

            if step % args.log_every == 0 and accelerator.is_main_process:
                rec = {"step": step, "lr": sched.get_last_lr()[0],
                       "sec/step": (time.time() - t_last) / args.log_every}
                if torch.cuda.is_available():
                    rec["mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                rec.update({k: s / max(c, 1) for k, (s, c) in sorted(stats.items())})
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items()), flush=True)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                stats.clear()
                t_last = time.time()
            if step % args.val_every == 0:
                run_validation()
            if step % args.save_every == 0:
                save(f"checkpoint-{step}")
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    save("final")
    run_validation()
    accelerator.end_training()


if __name__ == "__main__":
    main()
