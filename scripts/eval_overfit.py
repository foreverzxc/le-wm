"""Offline evaluation of the overfit model on LIBERO data.

Usage:
    python scripts/eval_overfit.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import stable_worldmodel as swm
from libero_data import LiberoDataset
from utils import get_img_preprocessor


def load_model(ckpt_dir, run_name="lewm"):
    ckpt_dir = Path(ckpt_dir)
    weight_files = sorted(ckpt_dir.glob(f"{run_name}_weights_epoch_*.pt"))
    weight_path = weight_files[-1]
    cfg_stem = weight_path.stem.replace("_weights_epoch_", "_config_epoch_")
    cfg_path = weight_path.parent / f"{cfg_stem}.json"
    cfg = OmegaConf.create(json.loads(cfg_path.read_text()))
    model = hydra.utils.instantiate(cfg)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    model.requires_grad_(False)
    print(f"Loaded: {weight_path.name}")
    return model


def main():
    cache = Path(swm.data.utils.get_cache_dir())
    model = load_model(cache / "checkpoints", "lewm")

    img_proc = get_img_preprocessor(source="pixels", target="pixels", img_size=128)

    ds = LiberoDataset(
        path="/home/cyborg/WM/LIBERO-datasets/libero_10",
        frameskip=5,
        num_steps=4,  # num_preds + history_size
        image_key="agentview_rgb",
        max_episodes=1,
        transform=img_proc,
    )

    ctx_len = 3
    n_preds = 1
    n_items = len(ds)

    # ---- Test 1: Batch prediction accuracy ----
    print("\n=== Test 1: Batch prediction accuracy ===")
    batch_items = [ds[i] for i in range(min(20, n_items))]
    batch = {k: torch.stack([item[k] for item in batch_items]) for k in batch_items[0]}
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model.encode(batch)
        emb = out["emb"]           # (B, T, D)
        act_emb = out["act_emb"]    # (B, T, D)

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        pred_emb = model.predict(ctx_emb, ctx_act)  # (B, ctx_len, D)
        tgt_emb = emb[:, n_preds:]  # (B, ctx_len, D)

    mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1)  # (B, ctx_len)
    cos = torch.nn.functional.cosine_similarity(
        pred_emb.flatten(0, 1), tgt_emb.flatten(0, 1), dim=-1
    ).reshape(pred_emb.shape[:2])

    print(f"  MSE:        {mse.mean().item():.6f} ± {mse.std().item():.6f}")
    print(f"  Cos sim:    {cos.mean().item():.4f} ± {cos.std().item():.4f}")
    print(f"  MSE range:  [{mse.min().item():.6f}, {mse.max().item():.6f}]")

    # ---- Test 2: Trajectory sweep — 1-step-ahead errors over time ----
    print("\n=== Test 2: Trajectory sweep ===")
    n_sweep = min(50, n_items)
    traj_errors = []
    traj_cos = []

    for i in range(n_sweep):
        item = ds[i]
        single = {k: v.unsqueeze(0).cuda() for k, v in item.items()}
        with torch.no_grad():
            out = model.encode(single)
            e = out["emb"]
            a = out["act_emb"]
            pred = model.predict(e[:, :ctx_len], a[:, :ctx_len])[:, -1]  # (1, D) — last step prediction
            tgt = e[:, ctx_len]  # (1, D) — actual next step embedding
            m = (pred - tgt).pow(2).mean().item()
            c = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1).item()
            traj_errors.append(m)
            traj_cos.append(c)

    print(f"  Sweep MSE:  {np.mean(traj_errors):.6f} ± {np.std(traj_errors):.6f}")
    print(f"  Sweep cos:  {np.mean(traj_cos):.4f} ± {np.std(traj_cos):.4f}")

    # ---- Test 3: Latent planning check ----
    # For each context, predict next embedding with correct vs random actions.
    # Correct action should produce lower prediction error.
    print("\n=== Test 3: Action-conditioned prediction ===")
    action_dim = ds.get_dim("action")  # 7
    raw_act_dim = ds.frameskip * action_dim  # 35, input to action_encoder
    n_tests = 20
    correct_gap = []

    for idx in range(min(n_tests, n_items)):
        item = ds[idx]
        single = {k: v.unsqueeze(0).cuda() for k, v in item.items()}
        B = 1

        with torch.no_grad():
            out = model.encode(single)
            ctx = out["emb"][:, :ctx_len]
            ctx_act = out["act_emb"][:, :ctx_len]

            # Correct action → prediction error
            pred_correct = model.predict(ctx, ctx_act)
            err_correct = (pred_correct - out["emb"][:, n_preds:]).pow(2).mean().item()

            # Random action → prediction error (average over 50 random)
            err_random = 0.0
            for _ in range(50):
                rand_act = torch.randn(B, ctx_len, raw_act_dim, device="cuda") * 2.0
                rand_act_emb = model.action_encoder(rand_act)
                pred_rand = model.predict(ctx, rand_act_emb)
                err_random += (pred_rand - out["emb"][:, n_preds:]).pow(2).mean().item()
            err_random /= 50
            correct_gap.append(err_random - err_correct)

    mean_gap = np.mean(correct_gap)
    print(f"  Error(correct action): lower by {mean_gap:.6f} vs random actions")
    print(f"  Gap per sample: {[f'{g:.4f}' for g in correct_gap[:10]]}{'...' if len(correct_gap) > 10 else ''}")
    print(f"  All gaps {'positive' if all(g > 0 for g in correct_gap) else 'NOT all positive'} "
          f"({sum(1 for g in correct_gap if g > 0)}/{len(correct_gap)} above zero)")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Per-sample MSE
    ax = axes[0, 0]
    mse_flat = mse.flatten().cpu().numpy()
    ax.bar(range(len(mse_flat)), mse_flat, color="steelblue", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Sample × step")
    ax.set_ylabel("MSE")
    ax.set_title(f"Per-sample prediction MSE\n(mean={mse.mean().item():.6f})")

    # 2. Trajectory sweep
    ax = axes[0, 1]
    ax.plot(traj_errors, color="steelblue", alpha=0.7, linewidth=1)
    ax.axhline(y=np.mean(traj_errors), color="red", linestyle="--", alpha=0.5, label=f"mean={np.mean(traj_errors):.6f}")
    ax.set_xlabel("Trajectory step")
    ax.set_ylabel("1-step pred MSE")
    ax.set_title(f"Prediction error along trajectory\n({n_sweep} steps)")
    ax.legend(fontsize=8)

    # 3. Action gap histogram
    ax = axes[1, 0]
    ax.bar(range(len(correct_gap)), correct_gap, color=["green" if g > 0 else "red" for g in correct_gap], alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Error(random) − Error(correct)")
    ax.set_title(f"Action-conditioned gap\n(mean={mean_gap:.6f}, >0 is good)")

    # 4. Embedding prediction vs actual scatter
    ax = axes[1, 1]
    pred_np = pred_emb.flatten(0, 1).cpu().numpy()[:, :2]  # first 2 PCA-like dims
    tgt_np = tgt_emb.flatten(0, 1).cpu().numpy()[:, :2]
    ax.scatter(pred_np[:, 0], pred_np[:, 1], c="red", alpha=0.6, s=30, label="predicted")
    ax.scatter(tgt_np[:, 0], tgt_np[:, 1], c="blue", alpha=0.6, s=30, label="actual")
    for i in range(min(20, len(pred_np))):
        ax.plot([pred_np[i, 0], tgt_np[i, 0]], [pred_np[i, 1], tgt_np[i, 1]],
                color="gray", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Embedding dim 0")
    ax.set_ylabel("Embedding dim 1")
    ax.set_title("Predicted vs actual (first 2 dims)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "output" / "eval_overfit.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
