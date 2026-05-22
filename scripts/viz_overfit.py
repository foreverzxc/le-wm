"""Overfit visualization: compare predicted vs actual embeddings.

Usage:
    python scripts/viz_overfit.py
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
import stable_pretraining as spt
from libero_data import LiberoDataset
from utils import get_img_preprocessor


def load_model(ckpt_dir, run_name="lewm", epoch=None):
    ckpt_dir = Path(ckpt_dir)
    weight_files = sorted(ckpt_dir.glob(f"{run_name}_weights_epoch_*.pt"))
    if not weight_files:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    if epoch is not None:
        weight_files = [f for f in weight_files if f"epoch_{epoch}" in f.stem]
        if not weight_files:
            raise FileNotFoundError(f"No epoch {epoch} checkpoint")
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

    # Dataset with training transforms
    img_prep = get_img_preprocessor(source="pixels", target="pixels", img_size=128)

    ds = LiberoDataset(
        path="/home/cyborg/WM/LIBERO-datasets/libero_10",
        frameskip=5,
        num_steps=4,  # num_preds + history_size
        image_key="agentview_rgb",
        max_episodes=1,
        transform=img_prep,
    )

    # Get properly transformed training samples via __getitem__
    n_samples = 10
    batch_items = [ds[i] for i in range(n_samples)]
    batch = {k: torch.stack([item[k] for item in batch_items]) for k in batch_items[0]}
    batch = {k: v.cuda() for k, v in batch.items()}

    ctx_len = 3
    n_preds = 1

    with torch.no_grad():
        output = model.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        pred_emb = model.predict(ctx_emb, ctx_act)
        tgt_emb = emb[:, n_preds:]

    # --- Metrics ---
    mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1)  # per-sample
    cos_sim = torch.nn.functional.cosine_similarity(
        pred_emb.flatten(1), tgt_emb.flatten(1), dim=-1
    )

    mse_flat = mse.flatten().cpu().tolist()
    n_total = len(mse_flat)
    print(f"\n  Prediction MSE (mean ± std): {mse.mean().item():.6f} ± {mse.std().item():.6f}")
    print(f"  Minimum MSE: {mse.min().item():.6f}  Maximum MSE: {mse.max().item():.6f}")

    # --- Autoregressive rollout on one sample ---
    item = ds[0]
    rollout_batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()}
    # pixels: (1, span=4, C, H, W), action: (1, num_steps=4, frameskip*D)

    with torch.no_grad():
        out = model.encode(rollout_batch)
        embs = out["emb"]  # (1, 4, D)

        # Autoregressive 1-step-ahead predictions along the trajectory
        ctx = embs[:, :ctx_len].clone()  # start context
        pred_errors = []
        cos_sims = []
        for t in range(ctx_len, embs.shape[1]):
            # Use the last ctx_len frames of actual embeddings as context
            actual_ctx = embs[:, t - ctx_len : t]
            # Predict next
            pred = model.predict(actual_ctx, out["act_emb"][:, t - ctx_len : t])[:, -1:]
            actual = embs[:, t : t + 1]
            err = (pred - actual).pow(2).mean().item()
            cos = torch.nn.functional.cosine_similarity(pred.flatten(), actual.flatten(), dim=0).item()
            pred_errors.append(err)
            cos_sims.append(cos)

    print(f"\n  Rollout errors (step 3→4): {pred_errors}")
    print(f"  Rollout cos sims:           {[f'{x:.3f}' for x in cos_sims]}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Per-sample MSE bar chart
    ax = axes[0]
    x = range(n_total)
    ax.bar(x, mse_flat, color="steelblue", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("MSE")
    ax.set_title(f"Prediction error per sample\n(mean={mse.mean().item():.6f})")

    # 2. Embedding dim comparison (first 32 dims, first sample)
    ax = axes[1]
    dims = min(32, pred_emb.shape[-1])
    x_dims = np.arange(dims)
    ax.bar(x_dims - 0.15, pred_emb[0, 0, :dims].cpu().numpy(), 0.3, label="predicted", alpha=0.8)
    ax.bar(x_dims + 0.15, tgt_emb[0, 0, :dims].cpu().numpy(), 0.3, label="actual", alpha=0.8)
    ax.set_xlabel("Embedding dimension")
    ax.set_title(f"First {dims} embedding dims (sample 0)")
    ax.legend(fontsize=8)

    # 3. Rollout
    ax = axes[2]
    if pred_errors:
        ax.plot(range(len(pred_errors)), pred_errors, "o-", color="red", markersize=8, label="MSE")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("MSE")
        ax.set_title("1-step ahead prediction error\nalong trajectory")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Rollout not available", ha="center", va="center")

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "output" / "overfit_viz.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
