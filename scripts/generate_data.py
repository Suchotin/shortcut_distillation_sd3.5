"""Generate the distillation dataset with the SD3.5 Medium teacher.

For every prompt: teacher sampling (default 40 steps, CFG 4.5, the model card's settings),
and we store the clean latent (model space, i.e. what the transformer is trained on) together
with the text embeddings, so training needs neither the text encoders nor the VAE.
Shards that already exist are skipped, so the script can be restarted.

Usage:
  python scripts/generate_data.py --prompts data/prompts.txt --out data/sd35m_512 \
      --height 512 --width 512 --batch_size 8
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
    pe, _, pp, _ = pipe.encode_prompt(
        prompt=prompts, prompt_2=None, prompt_3=None, device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=False, max_sequence_length=max_len,
    )
    return pe, pp


def noise_for(indices, shape, seed):
    """Per-sample deterministic noise, independent of batch size."""
    out = []
    for i in indices:
        g = torch.Generator("cpu").manual_seed(seed * 1_000_003 + i)
        out.append(torch.randn(shape, generator=g))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--prompts", default="data/prompts.txt")
    ap.add_argument("--out", default="data/sd35m_512")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--max_sequence_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    ap.add_argument("--num_previews", type=int, default=32, help="decode first N to PNG")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu_offload", action="store_true", help="for GPUs with <24GB")
    ap.add_argument("--num_workers", type=int, default=1, help="parallel processes, one per GPU")
    ap.add_argument("--worker_id", type=int, default=0, help="this process handles shards "
                    "with index %% num_workers == worker_id")
    args = ap.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    os.makedirs(os.path.join(args.out, "previews"), exist_ok=True)
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if pipe.scheduler.config.get("use_dynamic_shifting", False):
        print("[warn] scheduler uses dynamic shifting; the shortcut grid assumes a fixed shift")
    c = pipe.transformer.config.in_channels
    shape = (c, args.height // pipe.vae_scale_factor, args.width // pipe.vae_scale_factor)

    with torch.no_grad():
        null_pe, null_pp = encode(pipe, [""], args.max_sequence_length, device)
    if args.worker_id == 0:  # shared files are written once
        with torch.no_grad():
            sc_pe, sc_pp = encode(pipe, SHOWCASE_PROMPTS, args.max_sequence_length, device)
        save_file({"prompt_embeds": null_pe.half().cpu().contiguous(),
                   "pooled": null_pp.half().cpu().contiguous()},
                  os.path.join(args.out, "null_embeds.safetensors"))
        save_file({"prompt_embeds": sc_pe.half().cpu().contiguous(),
                   "pooled": sc_pp.half().cpu().contiguous()},
                  os.path.join(args.out, "showcase.safetensors"))
        with open(os.path.join(args.out, "showcase.json"), "w") as f:
            json.dump(SHOWCASE_PROMPTS, f, indent=1)
        meta = dict(model=args.model, height=args.height, width=args.width, steps=args.steps,
                    cfg=args.cfg, max_sequence_length=args.max_sequence_length, seed=args.seed,
                    n=len(prompts), shift=float(pipe.scheduler.config.get("shift", 3.0)),
                    latent_shape=list(shape))
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    n_shards = math.ceil(len(prompts) / args.shard_size)
    shard_range = lambda s: (s * args.shard_size, min((s + 1) * args.shard_size, len(prompts)))  # noqa: E731
    shard_path = lambda s: os.path.join(args.out, f"shard_{s:05d}.safetensors")  # noqa: E731
    todo = [s for s in range(n_shards)
            if s % args.num_workers == args.worker_id and not os.path.exists(shard_path(s))]
    n_todo = sum(hi - lo for lo, hi in map(shard_range, todo))
    print(f"worker {args.worker_id}/{args.num_workers}: {len(todo)} shards, {n_todo} images to generate")
    t0, done = time.time(), 0
    for s in todo:
        path = shard_path(s)
        lo, hi = shard_range(s)
        lat, pes, pps = [], [], []
        for b in range(lo, hi, args.batch_size):
            idx = list(range(b, min(b + args.batch_size, hi)))
            batch = [prompts[i] for i in idx]
            with torch.no_grad():
                pe, pp = encode(pipe, batch, args.max_sequence_length, device)
                n = len(batch)
                latents = pipe(
                    prompt_embeds=pe, pooled_prompt_embeds=pp,
                    negative_prompt_embeds=null_pe.expand(n, -1, -1),
                    negative_pooled_prompt_embeds=null_pp.expand(n, -1),
                    latents=noise_for(idx, shape, args.seed).to(device, torch.bfloat16),
                    num_inference_steps=args.steps, guidance_scale=args.cfg,
                    height=args.height, width=args.width, output_type="latent",
                ).images
                for j, i in enumerate(idx):  # decode a few previews for visual sanity checks
                    if i < args.num_previews:
                        x = latents[j: j + 1] / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
                        img = pipe.vae.decode(x.to(pipe.vae.dtype), return_dict=False)[0]
                        pipe.image_processor.postprocess(img)[0].save(
                            os.path.join(args.out, "previews", f"{i:05d}.png"))
            lat.append(latents.half().cpu())
            pes.append(pe.half().cpu())
            pps.append(pp.half().cpu())
            done += n
            rate = done / (time.time() - t0)
            print(f"[worker {args.worker_id}] {done}/{n_todo} {rate:.2f} img/s, "
                  f"eta {(n_todo - done) / rate / 60:.1f} min", flush=True)
        with open(path.replace(".safetensors", ".json"), "w") as f:
            json.dump(prompts[lo:hi], f, indent=0)
        tmp = path + ".tmp"  # atomic: an interrupted run never leaves a truncated shard behind
        save_file({"latents": torch.cat(lat).contiguous(),
                   "prompt_embeds": torch.cat(pes).contiguous(),
                   "pooled": torch.cat(pps).contiguous()}, tmp)
        os.replace(tmp, path)
        print(f"saved {path} (peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
