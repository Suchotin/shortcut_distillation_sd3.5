"""Turn a short benchmark (setup_env.sh bench) into an ETA / cost estimate for the full run."""

import argparse
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_log")
    ap.add_argument("train_jsonl")
    ap.add_argument("--images", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--price", type=float, default=1.57, help="USD per GPU-hour")
    ap.add_argument("--teacher_steps", type=int, default=40)
    args = ap.parse_args()

    rates = [float(x) for x in re.findall(r"([\d.]+) img/s", open(args.gen_log).read())]
    if not rates:
        raise SystemExit("no 'img/s' lines in the generation log")
    img_s = rates[-1]  # cumulative average incl. warm-up -> slightly pessimistic
    recs = [json.loads(line) for line in open(args.train_jsonl)]
    steady = recs[1:] or recs  # first interval includes warm-up / step-0 validation
    sec_step = sum(r["sec/step"] for r in steady) / len(steady)
    mem = max(r.get("mem_gb", 0.0) for r in recs)

    fwd_per_s = img_s * args.teacher_steps * 2  # generation: steps x (cond + uncond)
    eval_fwd = 104 * (2 * args.teacher_steps + 8 + 15)  # teacher + naive 4-step + student 1/2/4/8
    rows = [
        ("teacher data", args.images / img_s / 3600),
        ("training", args.steps * sec_step / 3600),
        ("evaluation (+~5 min model loading)", eval_fwd / fwd_per_s / 3600 + 5 / 60),
    ]
    total = sum(h for _, h in rows)
    print(f"\nmeasured: generation {img_s:.2f} img/s | training {sec_step:.2f} s/step | "
          f"peak training memory {mem:.1f} GiB")
    for name, h in rows:
        print(f"  {name:38s} {h * 60:6.0f} min")
    print(f"  {'total':38s} {total * 60:6.0f} min  ->  ~${total * args.price:.2f} at ${args.price}/h")


if __name__ == "__main__":
    main()
