"""Mirror a run's logs into TensorBoard while it is running (reads files only, never touches
the training process). Standalone on purpose: needs only torch.utils.tensorboard, numpy, PIL,
so it can run in any Python env that has tensorboard installed (it does not need the training
env); point a TensorBoard server at the parent of --logdir.

What it logs:
  training   loss/total, loss/flow_matching_d0, self_consistency/<N>_steps (per shortcut level),
             train/grad_norm, train/lr, perf/sec_per_step, perf/peak_mem_gb     <- log.jsonl
  samples    validation grids (rows = 1/2/4/8 steps, cols = the 4 prompts from the task)
  data       teacher generation progress and img/s                              <- logs/gen_*.log
  eval       metrics table (text + scalars) and final grids                      <- eval dir

Usage:
  python scripts/tb_sidecar.py --run_dir RUN --gen_logs 'logs/gen_*.log' --eval_dir EVAL \
      --logdir /root/tf-logs/shortcut_512
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

SCALAR_TAGS = {
    "loss": "loss/total",
    "fm": "loss/flow_matching_d0",
    "grad_norm": "train/grad_norm",
    "lr": "train/lr",
    "sec/step": "perf/sec_per_step",
    "mem_gb": "perf/peak_mem_gb",
}
GEN_RE = re.compile(r"\[worker (\d+)\] (\d+)/(\d+) ([\d.]+) img/s")


def tag_for(key: str) -> str:
    m = re.fullmatch(r"d=1/(\d+)", key)  # shortcut level d = 1/N  ->  N-step sampling
    if m:
        return f"self_consistency/{int(m.group(1)):02d}_steps"
    return SCALAR_TAGS.get(key, "other/" + re.sub(r"[^A-Za-z0-9_]", "_", key))


def load_image(path: str, max_px: int):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px))
    return np.asarray(img)


def stable(path: str, age: float = 5.0) -> bool:  # skip files that are still being written
    return time.time() - os.path.getmtime(path) > age


class Sidecar:
    def __init__(self, args):
        self.a = args
        self.w = SummaryWriter(args.logdir)
        self.train_pos = 0
        self.seen_imgs = set()
        self.gen_last = {}
        self.eval_done = False

    def training(self):
        path = os.path.join(self.a.run_dir, "log.jsonl")
        if not os.path.exists(path):
            return
        with open(path) as f:
            f.seek(self.train_pos)
            while True:
                line = f.readline()
                if not line or not line.endswith("\n"):  # nothing new / line still being written
                    break
                self.train_pos = f.tell()
                rec = json.loads(line)
                step = int(rec.pop("step"))
                for k, v in rec.items():
                    if isinstance(v, (int, float)):
                        self.w.add_scalar(tag_for(k), v, step)

    def samples(self):
        for p in sorted(glob.glob(os.path.join(self.a.run_dir, "samples", "step_*.png"))):
            key = (p, os.path.getmtime(p))
            if key in self.seen_imgs or not stable(p):
                continue
            step = int(re.search(r"step_(\d+)", p).group(1))
            self.w.add_image("samples/rows_1_2_4_8_steps", load_image(p, self.a.image_px), step,
                             dataformats="HWC")
            self.seen_imgs.add(key)

    def generation(self):
        for p in glob.glob(self.a.gen_logs):
            with open(p, errors="ignore") as f:
                for m in GEN_RE.finditer(f.read()):
                    worker, done, total, rate = int(m[1]), int(m[2]), int(m[3]), float(m[4])
                    if done <= self.gen_last.get(worker, 0):
                        continue
                    self.gen_last[worker] = done
                    self.w.add_scalar(f"data/images_done_worker{worker}", done, done)
                    self.w.add_scalar(f"data/img_per_s_worker{worker}", rate, done)
                    self.w.add_scalar(f"data/fraction_done_worker{worker}", done / total, done)

    def evaluation(self):
        if self.eval_done or not self.a.eval_dir:
            return
        mpath = os.path.join(self.a.eval_dir, "metrics.json")
        if not (os.path.exists(mpath) and stable(mpath)):
            return
        metrics = json.load(open(mpath))
        md = os.path.join(self.a.eval_dir, "metrics.md")
        if os.path.exists(md):
            self.w.add_text("eval/metrics", open(md).read(), 0)
        for method, m in metrics.items():
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    self.w.add_scalar(f"eval_{k}/{method}", v, 0)
        for name in ("grid_showcase", "grid_parti"):
            p = os.path.join(self.a.eval_dir, f"{name}.png")
            if os.path.exists(p):
                self.w.add_image(f"eval/{name}", load_image(p, 2 * self.a.image_px), 0,
                                 dataformats="HWC")
        self.eval_done = True

    def tick(self):
        for fn in (self.generation, self.training, self.samples, self.evaluation):
            try:
                fn()
            except Exception as e:  # noqa: BLE001  a bad line must not kill the sidecar
                print(f"[tb_sidecar] {fn.__name__}: {type(e).__name__}: {e}", flush=True)
        self.w.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--gen_logs", default="logs/gen_*.log")
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--image_px", type=int, default=1024)
    ap.add_argument("--once", action="store_true", help="single pass (for tests)")
    args = ap.parse_args()
    # the run's log files are the source of truth: rebuild from scratch on every start so a
    # restarted sidecar never duplicates points
    for old in glob.glob(os.path.join(args.logdir, "events.out.tfevents.*")):
        os.remove(old)
    s = Sidecar(args)
    print(f"[tb_sidecar] writing to {args.logdir}", flush=True)
    while True:
        s.tick()
        if args.once or s.eval_done:
            break
        time.sleep(args.interval)
    s.w.close()


if __name__ == "__main__":
    main()
