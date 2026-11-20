"""Build the training prompt list (disjoint from PartiPrompts, which is the benchmark).

Mix (defaults for 4000 prompts):
  * 40% COCO captions (yerevann/coco-karpathy)          - simple everyday scenes, close to Parti
  * 35% open-image-preferences (data-is-better-together) - diverse, high-quality T2I prompts
  * 15% Lexica SD prompts (Gustavosta/Stable-Diffusion-Prompts) - artistic styles
  * ~10% templated "text in image" prompts               - keeps typography (signs, chalkboards)
No content filtering by default; --filter drops prompts matching BLOCKLIST.

Usage:
  python scripts/prepare_prompts.py --out data/prompts.txt --total 4000
"""

import argparse
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
    """Read one column of a parquet dataset without downloading the other (image) columns."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    out = []
    for path in sorted(fs.glob(f"datasets/{repo}/**/*.parquet")):
        with fs.open(path, "rb", block_size=1 << 20) as fh:
            out += pq.read_table(fh, columns=[column]).column(column).to_pylist()
    return out


# Optional content filter (--filter, off by default): word-boundary match, case-insensitive.
BLOCKLIST = re.compile(r"\b(" + "|".join([
    r"nude\w*", r"naked", r"nsfw", r"sex\w*", r"sensual", r"erotic\w*", r"lingerie", r"bikini",
    r"topless", r"breasts?", r"boobs?", r"cleavage", r"busty", r"seductive\w*", r"underwear",
    r"porn\w*", r"hentai", r"fetish\w*", r"lewd", r"provocative", r"kiss\w*", r"bra",
    r"blood\w*", r"gore", r"gory", r"corpses?", r"dead body", r"decapitat\w*", r"severed",
    r"tortur\w*", r"suicide", r"murder\w*", r"kill\w*", r"violen\w*", r"terroris\w*", r"isis",
    r"guns?", r"rifles?", r"pistols?", r"firearms?", r"weapons?", r"bombs?", r"explosives?",
    r"drugs?", r"cocaine", r"weed", r"marijuana", r"heroin",
    r"nazi\w*", r"swastika", r"hitler", r"ss military", r"communis\w*", r"propaganda",
    r"putin", r"trump", r"biden",
    r"obama", r"politic\w*", r"president", r"ritual", r"sacrific\w*",
    # frequent real people in prompt datasets (not exhaustive)
    r"emma watson", r"emma stone", r"scarlett johansson", r"gal gadot", r"megan fox",
    r"taylor swift", r"elon musk", r"kim kardashian", r"margot robbie", r"angelina jolie",
    r"zendaya", r"ana de armas", r"natalie portman", r"kate beckinsale", r"keanu reeves",
    r"tom cruise", r"beyonce", r"rihanna", r"celebrity", r"billie eilish", r"jennifer lawrence",
    r"emilia clarke", r"anya taylor-joy", r"lady gaga", r"ariana grande", r"selena gomez",
    r"brad pitt", r"leonardo dicaprio", r"will smith", r"johnny depp", r"mark zuckerberg",
    r"bill gates", r"steve jobs", r"jeff bezos", r"kanye", r"dwayne johnson", r"the rock",
]) + r")\b", re.IGNORECASE)


def clean(p: str) -> str:
    p = re.sub(r"\s+", " ", p.replace("\n", " ")).strip().strip('"“”')
    return p


def norm(p: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", p.lower()).strip()


def ok(p: str, max_words: int, use_filter: bool = False) -> bool:
    n = len(p.split())
    return 3 <= n <= max_words and not (use_filter and BLOCKLIST.search(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prompts.txt")
    ap.add_argument("--sources_out", default="data/prompts_sources.jsonl")
    ap.add_argument("--total", type=int, default=4000)
    ap.add_argument("--frac_coco", type=float, default=0.40)
    ap.add_argument("--frac_oip", type=float, default=0.35)
    ap.add_argument("--frac_lexica", type=float, default=0.15)
    ap.add_argument("--filter", action="store_true",
                    help="drop prompts matching BLOCKLIST (off by default)")
    ap.add_argument("--frac_text", type=float, default=0.10)
    ap.add_argument("--max_words", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    parti = {norm(p) for p in load_dataset("nateraw/parti-prompts", split="train")["Prompt"]}

    def take(pool, n):
        pool = [clean(p) for p in pool]
        pool = [p for p in pool if ok(p, args.max_words, args.filter) and norm(p) not in parti]
        rng.shuffle(pool)
        return pool[:n]

    n_coco = int(args.total * args.frac_coco)
    n_oip = int(args.total * args.frac_oip)
    n_lex = int(args.total * args.frac_lexica)

    coco = load_dataset("yerevann/coco-karpathy", split="train")
    coco_pool = [rng.choice(s) for s in coco.shuffle(seed=args.seed).select(range(4 * n_coco))["sentences"]]
    # this dataset is 2 GB because of image columns; read only the prompts
    oip_pool = read_column("data-is-better-together/open-image-preferences-v1-binarized", "prompt")
    lex_pool = []
    if n_lex > 0:
        lex = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
        lex_pool = lex.shuffle(seed=args.seed).select(range(4 * n_lex))["Prompt"]

    prompts, sources = [], []
    for name, pool, n in (("coco-karpathy", coco_pool, n_coco),
                          ("open-image-preferences", oip_pool, n_oip),
                          ("lexica-sd-prompts", lex_pool, n_lex)):
        got = take(pool, n)
        prompts += got
        sources += [name] * len(got)
    while len(prompts) < args.total:  # templated typography prompts fill the rest
        t = rng.choice(TEXT_TEMPLATES)
        prompts.append(t.format(subj=rng.choice(TEXT_SUBJECTS), surf=rng.choice(TEXT_SURFACES),
                                word=rng.choice(TEXT_WORDS), style=rng.choice(TEXT_STYLES)))
        sources.append("text-in-image-template")

    seen, uniq = set(), []
    for p, s in zip(prompts, sources):  # dedupe (templates may repeat) while keeping order
        if norm(p) not in seen and norm(p) not in parti:
            seen.add(norm(p))
            uniq.append((p, s))
    rng.shuffle(uniq)

    import collections
    import json
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(p for p, _ in uniq) + "\n")
    with open(args.sources_out, "w") as f:  # provenance: one {"prompt", "source"} per line
        for p, s in uniq:
            f.write(json.dumps({"prompt": p, "source": s}, ensure_ascii=False) + "\n")
    print(f"wrote {len(uniq)} prompts to {args.out}: "
          f"{dict(collections.Counter(s for _, s in uniq))}")


if __name__ == "__main__":
    main()
