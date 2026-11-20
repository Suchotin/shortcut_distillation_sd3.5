"""Teacher-generated dataset: clean latents (model space) + text embeddings, stored in shards.

Layout written by scripts/generate_data.py:
    meta.json                     generation settings
    null_embeds.safetensors       empty-prompt embeddings (unconditional branch of CFG)
    showcase.safetensors/.json    embeddings of the 4 prompts from the task (for validation)
    shard_XXXXX.safetensors       latents [n,16,h,w] fp16, prompt_embeds [n,L,4096] fp16,
                                  pooled [n,2048] fp16
    shard_XXXXX.json              prompts of the shard
"""

from __future__ import annotations

import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import Dataset

SHOWCASE_PROMPTS = [
    "A realistic cheetah holding a sign that says: 'Fast as hell!'",
    "An astronaut riding a horse in a photorealistic style",
    "A dog wearing a baseball cap backwards and writing BONEZ on a chalkboard",
    "A castle made of cardboard.",
]


def shard_paths(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "shard_*.safetensors")))


class LatentDataset(Dataset):
    """Reads one sample at a time from the shards (lazy slicing, nothing held in RAM)."""

    def __init__(self, data_dir: str):
        self.paths = shard_paths(data_dir)
        if not self.paths:
            raise FileNotFoundError(f"no shard_*.safetensors in {data_dir}")
        self.index = []
        for si, p in enumerate(self.paths):
            with safe_open(p, framework="pt") as f:
                n = f.get_slice("latents").get_shape()[0]
            self.index += [(si, i) for i in range(n)]
        self._handles = {}

    def __len__(self):
        return len(self.index)

    def _handle(self, si):
        if si not in self._handles:  # opened lazily per dataloader worker
            self._handles[si] = safe_open(self.paths[si], framework="pt")
        return self._handles[si]

    def __getitem__(self, idx):
        si, i = self.index[idx]
        f = self._handle(si)
        return {
            "latents": f.get_slice("latents")[i],
            "prompt_embeds": f.get_slice("prompt_embeds")[i],
            "pooled": f.get_slice("pooled")[i],
        }

    def __getstate__(self):  # file handles are not picklable; reopen in workers
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


def load_null_embeds(data_dir: str):
    t = load_file(os.path.join(data_dir, "null_embeds.safetensors"))
    return t["prompt_embeds"], t["pooled"]


def load_showcase(data_dir: str):
    t = load_file(os.path.join(data_dir, "showcase.safetensors"))
    with open(os.path.join(data_dir, "showcase.json")) as f:
        prompts = json.load(f)
    return prompts, t["prompt_embeds"], t["pooled"]


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)
