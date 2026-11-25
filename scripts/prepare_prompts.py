"""Build the training prompts, disjoint from all PartiPrompts (the benchmark).

Mix of 4,000 prompts: COCO captions (35%), open-image-preferences (30%), Lexica SD prompts (15%),
and templated text-in-image prompts for the rest (signs, chalkboards, mugs...).

Usage:
  python scripts/prepare_prompts.py --out data/prompts_v2.txt --sources_out data/prompts_v2_sources.jsonl
  python scripts/prepare_prompts.py --out data/prompts_v3.txt --sources_out data/prompts_v3_sources.jsonl \
      --seed 1 --exclude data/prompts.txt data/prompts_v2.txt
"""

import argparse
import collections
import json
import os
import random
import re

from datasets import load_dataset

TEXT_SUBJECTS = [
    "a cat", "a golden retriever", "a robot", "an old fisherman", "a penguin", "a fox",
    "a little girl", "a chef", "a raccoon", "an owl", "a businessman", "a koala",
]
TEXT_SURFACES = [
    "a wooden sign", "a neon sign", "a chalkboard", "a coffee mug", "a t-shirt", "a birthday cake",
    "a street sign", "a book cover", "a storefront", "a banner", "a sticky note", "a whiteboard",
    "a cardboard box", "a vintage poster", "a license plate", "a graffiti wall",
]
TEXT_WORDS = [
    "HELLO", "OPEN", "Good Vibes", "SALE", "Coffee Time", "Welcome Home", "STOP", "Dream Big",
    "No Signal", "Fresh Bread", "GAME OVER", "Stay Wild", "Lucky 7", "Moon Base", "Hot Soup",
    "PIXEL", "Night Shift", "Free Hugs", "TACO", "Go Team", "Make Art", "Bonjour", "LOADING",
    "Zero Gravity", "Keep Calm", "Byte Me", "Sunny Days", "ROCKET", "Tea Party", "Low Battery",
]
TEXT_STYLES = ["photo", "photorealistic", "digital art", "watercolor illustration",
               "3d render", "cinematic shot", "oil painting", "studio photo"]
TEXT_TEMPLATES = [
    "{subj} holding a sign that says '{word}', {style}",
    "{surf} with the words \"{word}\" written on it, {style}",
    "a {style} of {surf} that reads '{word}'",
    "{subj} writing {word} on a chalkboard, {style}",
]


def read_column(repo: str, column: str) -> list[str]:
    """One column of a parquet dataset, without downloading the image columns."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    out = []
    for path in sorted(fs.glob(f"datasets/{repo}/**/*.parquet")):
        with fs.open(path, "rb", block_size=1 << 20) as fh:
            out += pq.read_table(fh, columns=[column]).column(column).to_pylist()
    return out


def clean(p: str) -> str:
    return re.sub(r"\s+", " ", p.replace("\n", " ")).strip().strip('"“”')


def norm(p: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", p.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prompts_v2.txt")
    ap.add_argument("--sources_out", default="data/prompts_v2_sources.jsonl")
    ap.add_argument("--total", type=int, default=4000)
    ap.add_argument("--frac_coco", type=float, default=0.35)
    ap.add_argument("--frac_oip", type=float, default=0.30)
    ap.add_argument("--frac_lexica", type=float, default=0.15)
    ap.add_argument("--max_words", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="prompt files not to reuse")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    banned = {norm(p) for p in load_dataset("nateraw/parti-prompts", split="train")["Prompt"]}
    for path in args.exclude:
        with open(path) as f:
            banned |= {norm(line) for line in f if line.strip()}

    def take(pool, n):
        pool = [clean(p) for p in pool]
        pool = [p for p in pool if 3 <= len(p.split()) <= args.max_words and norm(p) not in banned]
        rng.shuffle(pool)
        return pool[:n]

    n_coco = int(args.total * args.frac_coco)
    n_oip = int(args.total * args.frac_oip)
    n_lex = int(args.total * args.frac_lexica)
    coco = load_dataset("yerevann/coco-karpathy", split="train")
    coco_pool = [rng.choice(s) for s in coco.shuffle(seed=args.seed).select(range(4 * n_coco))["sentences"]]
    oip_pool = read_column("data-is-better-together/open-image-preferences-v1-binarized", "prompt")
    lex = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
    lex_pool = lex.shuffle(seed=args.seed).select(range(4 * n_lex))["Prompt"] if n_lex else []

    prompts, sources = [], []
    for name, pool, n in (("coco-karpathy", coco_pool, n_coco),
                          ("open-image-preferences", oip_pool, n_oip),
                          ("lexica-sd-prompts", lex_pool, n_lex)):
        got = take(pool, n)
        prompts += got
        sources += [name] * len(got)
    while len(prompts) < args.total:
        t = rng.choice(TEXT_TEMPLATES)
        prompts.append(t.format(subj=rng.choice(TEXT_SUBJECTS), surf=rng.choice(TEXT_SURFACES),
                                word=rng.choice(TEXT_WORDS), style=rng.choice(TEXT_STYLES)))
        sources.append("text-in-image-template")

    seen, uniq = set(), []
    for p, s in zip(prompts, sources):
        if norm(p) not in seen and norm(p) not in banned:
            seen.add(norm(p))
            uniq.append((p, s))
    rng.shuffle(uniq)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(p for p, _ in uniq) + "\n")
    with open(args.sources_out, "w") as f:
        for p, s in uniq:
            f.write(json.dumps({"prompt": p, "source": s}, ensure_ascii=False) + "\n")
    print(f"{len(uniq)} prompts -> {args.out}: {dict(collections.Counter(s for _, s in uniq))}")


if __name__ == "__main__":
    main()
