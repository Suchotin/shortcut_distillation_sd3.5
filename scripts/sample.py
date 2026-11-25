"""Generate images with a trained shortcut student.

Usage:
  python scripts/sample.py --prompt "A castle made of cardboard." --steps 4
  python scripts/sample.py --student runs/shortcut_1024/final/student_ema --prompt "..." "..."
"""

import argparse
import os
import sys

import torch
from diffusers import StableDiffusion3Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shortcut_sd35.shortcut import load_student, shortcut_sample  # noqa: E402

DEFAULT_STUDENT = "suchotino/sd3.5-medium-shortcut-lora"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", nargs="+", required=True)
    ap.add_argument("--student", default=DEFAULT_STUDENT, help="local directory or Hub repo id")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--steps", type=int, default=4, help="1, 2, 4 or 8")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="samples")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    student_dir = args.student
    if not os.path.isdir(student_dir):
        from huggingface_hub import snapshot_download
        student_dir = snapshot_download(args.student)

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model, cfg = load_student(student_dir, pipe.transformer)
    with torch.no_grad():
        pe, _, pp, _ = pipe.encode_prompt(prompt=args.prompt, prompt_2=None, prompt_3=None,
                                          device=device, do_classifier_free_guidance=False)
    shape = (len(args.prompt), pipe.transformer.config.in_channels,
             args.height // pipe.vae_scale_factor, args.width // pipe.vae_scale_factor)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(args.seed)).to(device)

    with torch.no_grad(), torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
        latents = shortcut_sample(model, noise, pe, pp, args.steps, cfg)
        x = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        images = pipe.image_processor.postprocess(pipe.vae.decode(x.to(dtype), return_dict=False)[0])

    os.makedirs(args.out, exist_ok=True)
    for i, im in enumerate(images):
        path = os.path.join(args.out, f"{i:02d}_{args.steps}steps.png")
        im.save(path)
        print(path)


if __name__ == "__main__":
    main()
