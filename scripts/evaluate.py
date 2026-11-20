"""Evaluate the shortcut student on the first N PartiPrompts (+ the 4 prompts from the task).

Methods (same initial noise per prompt for all of them):
  teacher   SD3.5 Medium, scheduler default grid, `--teacher_steps` (40) steps, CFG 4.5
  naive     SD3.5 Medium with only `--naive_steps` (4) steps, CFG 4.5   (what we must beat)
  student   shortcut model, n in `--steps` (1,2,4,8), no CFG (CFG is distilled)

Metrics on PartiPrompts:
  clip_score     100 * cos(CLIP-L/14 image, text)            prompt alignment
  pickscore      PickScore_v1 (optional, --pickscore)         human-preference proxy
  clip_sim_teacher  cos(CLIP image emb, teacher image emb)    fidelity to the teacher
  sec/img, NFE per image

Usage:
  python scripts/evaluate.py --student runs/shortcut/final/student_ema --out eval/run1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shortcut_sd35.data import SHOWCASE_PROMPTS  # noqa: E402
from shortcut_sd35.shortcut import load_student, shortcut_sample  # noqa: E402


@torch.no_grad()
def teacher_sample(transformer, sched_config, noise, pe, pp, null_pe, null_pp, steps, cfg):
    """Exactly the StableDiffusion3Pipeline denoising loop (Euler, shift, CFG)."""
    sched = FlowMatchEulerDiscreteScheduler.from_config(sched_config)
    kw = {}
    if sched.config.get("use_dynamic_shifting", False):  # same as the pipeline (off for SD3.5)
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift
        c = sched.config
        kw["mu"] = calculate_shift((noise.shape[-2] // 2) * (noise.shape[-1] // 2),
                                   c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096),
                                   c.get("base_shift", 0.5), c.get("max_shift", 1.16))
    sched.set_timesteps(steps, device=noise.device, **kw)
    b = noise.shape[0]
    x = noise.to(pe.dtype)
    for t in sched.timesteps:
        v = transformer(hidden_states=torch.cat([x, x]), timestep=t.expand(2 * b),
                        encoder_hidden_states=torch.cat([pe, null_pe.expand(b, -1, -1)]),
                        pooled_projections=torch.cat([pp, null_pp.expand(b, -1)]),
                        return_dict=False)[0]
        vc, vu = v.chunk(2)
        x = sched.step(vu + cfg * (vc - vu), t, x, return_dict=False)[0]
    return x


@torch.no_grad()
def decode(pipe, latents):
    x = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    img = pipe.vae.decode(x.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img)


def labeled_grid(rows, col_labels, row_labels, cell=256, pad=24):
    w = cell * len(col_labels) + 220
    h = pad + cell * len(rows)
    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    for j, lab in enumerate(col_labels):
        d.text((220 + j * cell + 4, 4), lab, fill="black")
    for i, (row, rl) in enumerate(zip(rows, row_labels)):
        y = pad + i * cell
        d.multiline_text((4, y + 4), "\n".join(rl[k:k + 30] for k in range(0, len(rl), 30)),
                         fill="black")
        for j, im in enumerate(row):
            canvas.paste(im.resize((cell, cell)), (220 + j * cell, y))
    return canvas


class Scorer:
    def __init__(self, device, pickscore=False):
        from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.pick = None
        if pickscore:
            self.pick_proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.pick = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()

    @torch.no_grad()
    def _emb(self, model, proc, images, texts):
        ii = proc(images=images, return_tensors="pt").to(self.device)
        ti = proc(text=texts, padding=True, truncation=True, max_length=77,
                  return_tensors="pt").to(self.device)
        ie = model.get_image_features(**ii)
        te = model.get_text_features(**ti)
        ie = getattr(ie, "pooler_output", ie)
        te = getattr(te, "pooler_output", te)
        return torch.nn.functional.normalize(ie, dim=-1), torch.nn.functional.normalize(te, dim=-1)

    def score(self, images, texts, bs=16):
        out = {"clip_score": [], "img_emb": []}
        if self.pick is not None:
            out["pickscore"] = []
        for k in range(0, len(images), bs):
            im, tx = images[k:k + bs], texts[k:k + bs]
            ie, te = self._emb(self.clip, self.clip_proc, im, tx)
            out["clip_score"] += (100 * (ie * te).sum(-1).clamp(min=0)).tolist()
            out["img_emb"].append(ie.cpu())
            if self.pick is not None:
                pie, pte = self._emb(self.pick, self.pick_proc, im, tx)
                out["pickscore"] += (self.pick.logit_scale.exp() * (pie * pte).sum(-1)).tolist()
        out["img_emb"] = torch.cat(out["img_emb"])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--student", required=True, help="dir with lora/d_embedder/shortcut_config")
    ap.add_argument("--out", default="eval/run")
    ap.add_argument("--num_prompts", type=int, default=100)
    ap.add_argument("--steps", default="1,2,4,8")
    ap.add_argument("--teacher_steps", type=int, default=40)
    ap.add_argument("--naive_steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--max_sequence_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pickscore", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    dt = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    parti = load_dataset("nateraw/parti-prompts", split="train")["Prompt"][: args.num_prompts]
    prompts = list(parti) + SHOWCASE_PROMPTS
    n_parti = len(parti)

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dt).to(device)
    with torch.no_grad():
        enc = [pipe.encode_prompt(prompt=prompts[k:k + 16], prompt_2=None, prompt_3=None,
                                  device=device, do_classifier_free_guidance=False,
                                  max_sequence_length=args.max_sequence_length)
               for k in range(0, len(prompts), 16)]
        pe = torch.cat([e[0] for e in enc])
        pp = torch.cat([e[2] for e in enc])
        null_pe, _, null_pp, _ = pipe.encode_prompt(
            prompt=[""], prompt_2=None, prompt_3=None, device=device,
            do_classifier_free_guidance=False, max_sequence_length=args.max_sequence_length)
    for te in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        te.to("cpu")
    torch.cuda.empty_cache()

    c = pipe.transformer.config.in_channels
    shape = (c, args.height // pipe.vae_scale_factor, args.width // pipe.vae_scale_factor)
    noise = torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(args.seed * 7919 + i))
                         for i in range(len(prompts))]).to(device)

    def run(name, fn, nfe):
        d = os.path.join(args.out, "images", name)
        os.makedirs(d, exist_ok=True)
        paths = [os.path.join(d, f"{i:03d}.png") for i in range(len(prompts))]
        if all(os.path.exists(p) for p in paths):  # cached (e.g. teacher from a previous run)
            print(f"{name}: cached")
            return [Image.open(p).convert("RGB") for p in paths], None
        imgs, t_total = [], 0.0
        for k in range(0, len(prompts), args.batch_size):
            sl = slice(k, k + args.batch_size)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=dt):
                lat = fn(noise[sl], pe[sl], pp[sl])
            torch.cuda.synchronize()
            t_total += time.time() - t0
            imgs += decode(pipe, lat)
        for im, p in zip(imgs, paths):
            im.save(p)
        sec = t_total / len(prompts)
        print(f"{name}: {sec:.3f} s/img (transformer only), NFE {nfe}")
        return imgs, sec

    sc = pipe.scheduler.config
    results = {}
    results[f"teacher_{args.teacher_steps}step_cfg"] = run(
        f"teacher_{args.teacher_steps}step_cfg",
        lambda n, e, p: teacher_sample(pipe.transformer, sc, n, e, p, null_pe, null_pp,
                                       args.teacher_steps, args.cfg), 2 * args.teacher_steps)
    results[f"naive_{args.naive_steps}step_cfg"] = run(
        f"naive_{args.naive_steps}step_cfg",
        lambda n, e, p: teacher_sample(pipe.transformer, sc, n, e, p, null_pe, null_pp,
                                       args.naive_steps, args.cfg), 2 * args.naive_steps)

    student, scfg = load_student(args.student, pipe.transformer)
    for n in [int(s) for s in args.steps.split(",")]:
        nfe = n if n < scfg.base_steps else 2 * n
        results[f"student_{n}step"] = run(
            f"student_{n}step",
            lambda x, e, p, n=n: shortcut_sample(student, x, e, p, n, scfg,
                                                 null_pe=null_pe, null_pp=null_pp), nfe)
    del student
    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()

    # ---- metrics on PartiPrompts ----
    scorer = Scorer(device, pickscore=args.pickscore)
    teacher_key = f"teacher_{args.teacher_steps}step_cfg"
    scored = {k: scorer.score(v[0][:n_parti], prompts[:n_parti]) for k, v in results.items()}
    t_emb = scored[teacher_key]["img_emb"]
    metrics = {}
    for k, s in scored.items():
        m = {"clip_score": float(np.mean(s["clip_score"])),
             "clip_sim_teacher": float((s["img_emb"] * t_emb).sum(-1).mean()),
             "sec_per_img": results[k][1]}
        if "pickscore" in s:
            m["pickscore"] = float(np.mean(s["pickscore"]))
        metrics[k] = m
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    cols = ["clip_score", "pickscore", "clip_sim_teacher", "sec_per_img"]
    lines = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k, m in metrics.items():
        lines.append(f"| {k} | " + " | ".join(
            f"{m[c]:.4f}" if isinstance(m.get(c), float) else "-" for c in cols) + " |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "metrics.md"), "w") as f:
        f.write(f"PartiPrompts first {n_parti}, {args.height}x{args.width}\n\n{table}\n")
    print(table)

    # ---- grids: the 4 task prompts, and the first 8 PartiPrompts ----
    names = list(results)
    for tag, idx in (("showcase", range(n_parti, len(prompts))), ("parti", range(min(8, n_parti)))):
        rows = [[results[k][0][i] for k in names] for i in idx]
        labeled_grid(rows, names, [prompts[i] for i in idx]).save(
            os.path.join(args.out, f"grid_{tag}.png"))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
