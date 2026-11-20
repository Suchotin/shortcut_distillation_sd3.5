"""CPU smoke test with a tiny random SD3 (MMDiT-X with dual attention) — no GPU / HF access needed.

Run:  python tests/test_smoke.py
"""

import json
import math
import os
import sys
import tempfile

import torch
from diffusers import AutoencoderKL, SD3Transformer2DModel
from safetensors.torch import save_file

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from shortcut_sd35.data import SHOWCASE_PROMPTS, LatentDataset  # noqa: E402
from shortcut_sd35.shortcut import (EMA, ShortcutConfig, ShortcutSD3, add_lora,  # noqa: E402
                                    load_student, predict, sample_bootstrap,
                                    sample_flow_matching, sampling_sigmas,
                                    self_consistency_target, shift_sigma, shortcut_sample,
                                    teacher_mode)

L, JD, PD, C, H = 12, 32, 64, 16, 8  # text tokens, text dim, pooled dim, latent ch, latent hw


def tiny_transformer():
    torch.manual_seed(0)
    return SD3Transformer2DModel(
        sample_size=H, patch_size=2, in_channels=C, out_channels=C, num_layers=3,
        attention_head_dim=8, num_attention_heads=4, joint_attention_dim=JD,
        caption_projection_dim=32, pooled_projection_dim=PD, pos_embed_max_size=32,
        dual_attention_layers=(0, 1), qk_norm="rms_norm")


def tiny_vae():
    torch.manual_seed(0)
    return AutoencoderKL(in_channels=3, out_channels=3, latent_channels=C,
                         down_block_types=["DownEncoderBlock2D"] * 4,
                         up_block_types=["UpDecoderBlock2D"] * 4,
                         block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=8,
                         scaling_factor=1.5305, shift_factor=0.0609,
                         use_quant_conv=False, use_post_quant_conv=False)


def rand_cond(b):
    return torch.randn(b, L, JD), torch.randn(b, PD)


def test_zero_init_equals_teacher():
    tr = tiny_transformer()
    cfg = ShortcutConfig(base_steps=8, lora_rank=4, lora_alpha=4)
    add_lora(tr, cfg)
    names = [n for n, _ in tr.named_modules() if n.endswith("lora_A")]
    assert any(".attn2." in n for n in names), "attn2 (dual attention) not LoRA-wrapped"
    assert any("norm1.linear" in n for n in names), "adaLN not LoRA-wrapped"
    for n, p in tr.named_parameters():  # make LoRA non-trivial, teacher must ignore it
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)
    m = ShortcutSD3(tr).eval()
    x = torch.randn(2, C, H, H)
    pe, pp = rand_cond(2)
    s = torch.tensor([0.9, 0.3])
    with torch.no_grad():
        with teacher_mode(m):
            t1 = m(x, s, None, pe, pp, use_d=False)
        tr.disable_adapters()
        t2 = tr(hidden_states=x, timestep=s * 1000, encoder_hidden_states=pe,
                pooled_projections=pp, return_dict=False)[0]
        tr.enable_adapters()
        a = m(x, s, torch.tensor([0.0, 0.0]), pe, pp)
        b = m(x, s, torch.tensor([0.5, 1.0]), pe, pp)
    assert torch.allclose(t1, t2), "teacher_mode != base model"
    assert torch.allclose(a, b), "zero-init d-embedder must make output independent of d"
    assert not torch.allclose(a, t1), "LoRA should change the student output"
    with torch.no_grad():  # after the d-embedder learns something, d must matter
        torch.nn.init.normal_(m.d_embedder.mlp.linear_2.weight, std=0.1)
        c = m(x, s, torch.tensor([0.5, 1.0]), pe, pp)
    assert not torch.allclose(a, c)
    print("ok zero-init / teacher_mode / LoRA targets")


def test_grid_and_sampling():
    cfg = ShortcutConfig(base_steps=32, shift=3.0)
    sig = sampling_sigmas(4, 3.0)
    assert abs(sig[0] - 1) < 1e-6 and abs(sig[-1]) < 1e-6 and torch.all(sig[:-1] > sig[1:])
    # 4-step grid is a subset of the 32-step grid (shortcut targets compose exactly)
    fine = sampling_sigmas(32, 3.0)
    assert torch.allclose(sig, fine[::8])
    g = torch.Generator().manual_seed(0)
    b = sample_bootstrap(1000, cfg, generator=g)
    assert torch.all(b["sigma0"] > b["sigma1"]) and torch.all(b["sigma1"] > b["sigma2"])
    assert torch.all(b["sigma2"] >= 0)
    assert set(b["level"].tolist()) == set(range(1, 6))
    assert torch.all(b["d_half"][b["level"] == 1] == 0)
    # start times are multiples of the student step
    u0 = (b["sigma0"].double() / (3 - 2 * b["sigma0"].double()))  # inverse shift for s=3
    k = (1 - u0) / b["d_student"].double()
    assert torch.allclose(k, k.round(), atol=1e-4)
    s_fm, d_fm = sample_flow_matching(100, cfg, g)
    assert torch.all(d_fm == 0) and torch.all(s_fm > 0) and torch.all(s_fm <= 1)
    # restricting levels (e.g. only >= 4 steps)
    b4 = sample_bootstrap(50, cfg, levels=[1, 2, 3], generator=g)
    assert b4["d_student"].max() <= 0.25 + 1e-6
    print("ok grid / bootstrap sampling")


class ConstVel(torch.nn.Module):
    """Fake model with a known velocity field v(x, sigma) = a * x + c(sigma) (linear ODE)."""

    def forward(self, x, sigma, d, pe, pp, use_d=True):
        return 0.5 * x + sigma.view(-1, 1, 1, 1)


def test_target_formula():
    cfg = ShortcutConfig(base_steps=8, shift=3.0, cfg_scale=1.0)  # cfg=1 -> CFG is a no-op
    m = ConstVel()
    g = torch.Generator().manual_seed(1)
    b = sample_bootstrap(16, cfg, generator=g)
    x = torch.randn(16, C, H, H)
    pe, pp = rand_cond(16)
    tgt = self_consistency_target(m, x, b, pe, pp, pe[:1], pp[:1], cfg)
    # manual two Euler half-steps, then average velocity over the full step
    d1 = (b["sigma0"] - b["sigma1"]).view(-1, 1, 1, 1)
    d2 = (b["sigma1"] - b["sigma2"]).view(-1, 1, 1, 1)
    v1 = m(x, b["sigma0"], None, None, None)
    xm = x - d1 * v1
    v2 = m(xm, b["sigma1"], None, None, None)
    x2 = xm - d2 * v2
    ref = (x - x2) / (d1 + d2)
    assert torch.allclose(tgt, ref, atol=1e-5)
    # uniform grid (shift = 1) -> plain average as in the paper
    cfg1 = ShortcutConfig(base_steps=8, shift=1.0, cfg_scale=1.0)
    b1 = sample_bootstrap(16, cfg1, generator=g)
    t1 = self_consistency_target(m, x, b1, pe, pp, pe[:1], pp[:1], cfg1)
    v1 = m(x, b1["sigma0"], None, None, None)
    v2 = m(x - (b1["sigma0"] - b1["sigma1"]).view(-1, 1, 1, 1) * v1, b1["sigma1"], None, None, None)
    assert torch.allclose(t1, (v1 + v2) / 2, atol=1e-5)
    print("ok self-consistency target")


def test_cfg_in_predict():
    tr = tiny_transformer()
    add_lora(tr, ShortcutConfig(lora_rank=4, lora_alpha=4))
    m = ShortcutSD3(tr).eval()
    x = torch.randn(2, C, H, H)
    pe, pp = rand_cond(2)
    npe, npp = rand_cond(1)
    s, d = torch.tensor([0.7, 0.2]), torch.zeros(2)
    with torch.no_grad():
        c = predict(m, x, s, d, pe, pp)
        u = predict(m, x, s, d, npe.expand(2, -1, -1), npp.expand(2, -1))
        g = predict(m, x, s, d, pe, pp, cfg_scale=4.5, null_pe=npe, null_pp=npp)
    assert torch.allclose(g, u + 4.5 * (c - u), atol=1e-4)
    print("ok CFG")


def test_ema_swap():
    p = [torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.zeros(2))]
    ema = EMA(p, 0.9)
    with torch.no_grad():
        p[0].add_(1.0)
    ema.update()
    assert torch.allclose(ema.shadow[0], torch.full((3,), 1.1))
    with ema.swapped():
        assert torch.allclose(p[0], torch.full((3,), 1.1))
    assert torch.allclose(p[0], torch.full((3,), 2.0))
    print("ok EMA")


def make_fake_setup(tmp):
    model_dir = os.path.join(tmp, "model")
    tiny_transformer().save_pretrained(os.path.join(model_dir, "transformer"))
    tiny_vae().save_pretrained(os.path.join(model_dir, "vae"))
    data = os.path.join(tmp, "data")
    os.makedirs(data)
    for s in range(2):
        n = 10
        save_file({"latents": torch.randn(n, C, H, H).half(),
                   "prompt_embeds": torch.randn(n, L, JD).half(),
                   "pooled": torch.randn(n, PD).half()},
                  os.path.join(data, f"shard_{s:05d}.safetensors"))
    npe, npp = rand_cond(1)
    save_file({"prompt_embeds": npe.half(), "pooled": npp.half()},
              os.path.join(data, "null_embeds.safetensors"))
    spe, spp = rand_cond(4)
    save_file({"prompt_embeds": spe.half(), "pooled": spp.half()},
              os.path.join(data, "showcase.safetensors"))
    with open(os.path.join(data, "showcase.json"), "w") as f:
        json.dump(SHOWCASE_PROMPTS, f)
    with open(os.path.join(data, "meta.json"), "w") as f:
        json.dump({"shift": 3.0, "cfg": 4.5, "latent_shape": [C, H, H]}, f)
    return model_dir, data


def test_train_end_to_end(mixed_precision):
    import train
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)  # allow another precision in-process
    with tempfile.TemporaryDirectory() as tmp:
        model_dir, data = make_fake_setup(tmp)
        assert len(LatentDataset(data)) == 20
        out = os.path.join(tmp, "run")
        common = ["--model", model_dir, "--data_dir", data, "--output_dir", out,
                  "--base_steps", "8", "--lora_rank", "4", "--lora_alpha", "4",
                  "--batch_size", "4", "--bootstrap_fraction", "0.5", "--num_workers", "0",
                  "--log_every", "2", "--val_every", "3", "--save_every", "3",
                  "--val_steps", "1,2,4,8", "--mixed_precision", mixed_precision,
                  "--gradient_checkpointing", "--lr", "1e-3", "--warmup_steps", "0"]
        train.main(common + ["--max_steps", "6", "--base_velocity", "teacher"])
        assert os.path.exists(os.path.join(out, "final", "student_ema", "lora.safetensors"))
        # old optimizer states are pruned, student weights are kept
        assert not os.path.exists(os.path.join(out, "checkpoint-3", "train_state.pt"))
        assert os.path.exists(os.path.join(out, "checkpoint-3", "student_ema", "lora.safetensors"))
        assert os.path.exists(os.path.join(out, "checkpoint-6", "train_state.pt"))
        assert os.path.exists(os.path.join(out, "samples", "step_000006.png"))
        logs = [json.loads(line) for line in open(os.path.join(out, "log.jsonl"))]
        assert all(math.isfinite(v) for r in logs for v in r.values() if isinstance(v, float))
        # resume and continue
        train.main(common + ["--max_steps", "8", "--resume", os.path.join(out, "checkpoint-6")])
        st = torch.load(os.path.join(out, "final", "train_state.pt"))
        assert st["step"] == 8

        # reload the student into a fresh transformer: outputs must match the EMA model
        tr = SD3Transformer2DModel.from_pretrained(os.path.join(model_dir, "transformer"))
        student, scfg = load_student(os.path.join(out, "final", "student_ema"), tr)
        x = torch.randn(4, C, H, H)
        pe, pp = rand_cond(4)
        npe, npp = rand_cond(1)
        with torch.no_grad():
            outs = {n: shortcut_sample(student, x, pe, pp, n, scfg, null_pe=npe, null_pp=npp)
                    for n in (1, 2, 4, 8)}
            d = torch.full((4,), 0.25)
            v_d = student(x, torch.ones(4), d, pe, pp)
            v_0 = student(x, torch.ones(4), torch.zeros(4), pe, pp)
        assert all(torch.isfinite(v).all() for v in outs.values())
        assert not torch.allclose(v_d, v_0), "after training the output should depend on d"
    print(f"ok train end-to-end ({mixed_precision}) + resume + load_student + sampling")


def test_eval_helpers():
    from diffusers import FlowMatchEulerDiscreteScheduler
    from PIL import Image

    import evaluate

    tr = tiny_transformer().eval()
    sc = FlowMatchEulerDiscreteScheduler(shift=3.0).config
    x = torch.randn(2, C, H, H)
    pe, pp = rand_cond(2)
    npe, npp = rand_cond(1)
    out = evaluate.teacher_sample(tr, sc, x, pe, pp, npe, npp, steps=4, cfg=4.5)
    # reference: manual Euler with CFG on the scheduler's sigmas
    s = FlowMatchEulerDiscreteScheduler.from_config(sc)
    s.set_timesteps(4)
    ref = x.clone()
    with torch.no_grad():
        for i, t in enumerate(s.timesteps):
            vc = tr(hidden_states=ref, timestep=t.expand(2), encoder_hidden_states=pe,
                    pooled_projections=pp, return_dict=False)[0]
            vu = tr(hidden_states=ref, timestep=t.expand(2), encoder_hidden_states=npe.expand(2, -1, -1),
                    pooled_projections=npp.expand(2, -1), return_dict=False)[0]
            ref = ref + (s.sigmas[i + 1] - s.sigmas[i]) * (vu + 4.5 * (vc - vu))
    assert torch.allclose(out, ref, atol=1e-4)
    im = Image.new("RGB", (64, 64))
    g = evaluate.labeled_grid([[im, im], [im, im]], ["a", "b"], ["p1", "p2"], cell=32)
    assert g.size == (220 + 64, 24 + 64)
    print("ok evaluate helpers")


if __name__ == "__main__":
    test_eval_helpers()
    test_zero_init_equals_teacher()
    test_grid_and_sampling()
    test_target_formula()
    test_cfg_in_predict()
    test_ema_swap()
    test_train_end_to_end("no")
    test_train_end_to_end("bf16")
    print("ALL OK")
