"""Generate the distillation data with the SD3.5 Medium teacher (40 steps, CFG 4.5).

Stores the clean latents with the text embeddings in shards, so training needs neither the text
encoders nor the VAE. Existing shards are skipped, so the script can be restarted.

Usage:
  python scripts/generate_data.py --prompts data/prompts_v2.txt --out data/sd35m_1024
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shortcut_sd35.data import SHOWCASE_PROMPTS  # noqa: E402


def encode(pipe, prompts, max_len, device):
    pe, _, pp, _ = pipe.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None, device=device,
                                      do_classifier_free_guidance=False, max_sequence_length=max_len)
    return pe, pp


def noise_for(indices, shape, seed):
    """Per-prompt noise, independent of the batch size."""
    return torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(seed * 1_000_003 + i))
                        for i in indices])


def half(**tensors):
    return {k: v.half().cpu().contiguous() for k, v in tensors.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--prompts", default="data/prompts_v2.txt")
    ap.add_argument("--out", default="data/sd35m_1024")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--max_sequence_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--num_previews", type=int, default=32, help="decode the first N to PNG")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1, help="one process per GPU")
    ap.add_argument("--worker_id", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    os.makedirs(os.path.join(args.out, "previews"), exist_ok=True)
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    shape = (pipe.transformer.config.in_channels, args.height // pipe.vae_scale_factor,
             args.width // pipe.vae_scale_factor)

    with torch.no_grad():
        null_pe, null_pp = encode(pipe, [""], args.max_sequence_length, device)
        if args.worker_id == 0 and not os.path.exists(os.path.join(args.out, "meta.json")):
            sc_pe, sc_pp = encode(pipe, SHOWCASE_PROMPTS, args.max_sequence_length, device)
            save_file(half(prompt_embeds=null_pe, pooled=null_pp),
                      os.path.join(args.out, "null_embeds.safetensors"))
            save_file(half(prompt_embeds=sc_pe, pooled=sc_pp),
                      os.path.join(args.out, "showcase.safetensors"))
            with open(os.path.join(args.out, "showcase.json"), "w") as f:
                json.dump(SHOWCASE_PROMPTS, f, indent=1)
            meta = dict(model=args.model, height=args.height, width=args.width, steps=args.steps,
                        cfg=args.cfg, max_sequence_length=args.max_sequence_length,
                        seed=args.seed, n=len(prompts),
                        shift=float(pipe.scheduler.config.get("shift", 3.0)),
                        latent_shape=list(shape))
            with open(os.path.join(args.out, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

    def shard_path(s):
        return os.path.join(args.out, f"shard_{s:05d}.safetensors")

    todo = [s for s in range(math.ceil(len(prompts) / args.shard_size))
            if s % args.num_workers == args.worker_id and not os.path.exists(shard_path(s))]
    n_todo = sum(min(args.shard_size, len(prompts) - s * args.shard_size) for s in todo)
    print(f"worker {args.worker_id}: {len(todo)} shards, {n_todo} images")
    t0, done = time.time(), 0
    for s in todo:
        lo, hi = s * args.shard_size, min((s + 1) * args.shard_size, len(prompts))
        lat, pes, pps = [], [], []
        for b in range(lo, hi, args.batch_size):
            idx = list(range(b, min(b + args.batch_size, hi)))
            with torch.no_grad():
                pe, pp = encode(pipe, [prompts[i] for i in idx], args.max_sequence_length, device)
                latents = pipe(prompt_embeds=pe, pooled_prompt_embeds=pp,
                               negative_prompt_embeds=null_pe.expand(len(idx), -1, -1),
                               negative_pooled_prompt_embeds=null_pp.expand(len(idx), -1),
                               latents=noise_for(idx, shape, args.seed).to(device, torch.bfloat16),
                               num_inference_steps=args.steps, guidance_scale=args.cfg,
                               height=args.height, width=args.width, output_type="latent").images
                for j, i in enumerate(idx):
                    if i < args.num_previews:
                        x = latents[j:j + 1] / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
                        img = pipe.vae.decode(x.to(pipe.vae.dtype), return_dict=False)[0]
                        pipe.image_processor.postprocess(img)[0].save(
                            os.path.join(args.out, "previews", f"{i:05d}.png"))
            lat.append(latents)
            pes.append(pe)
            pps.append(pp)
            done += len(idx)
            rate = done / (time.time() - t0)
            print(f"[worker {args.worker_id}] {done}/{n_todo} {rate:.2f} img/s, "
                  f"eta {(n_todo - done) / rate / 60:.0f} min", flush=True)
        with open(shard_path(s).replace(".safetensors", ".json"), "w") as f:
            json.dump(prompts[lo:hi], f, indent=0)
        tmp = shard_path(s) + ".tmp"
        save_file(half(latents=torch.cat(lat), prompt_embeds=torch.cat(pes), pooled=torch.cat(pps)), tmp)
        os.replace(tmp, shard_path(s))


if __name__ == "__main__":
    main()
