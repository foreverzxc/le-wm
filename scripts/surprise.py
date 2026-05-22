"""Surprise (prediction error) along a trajectory.

Usage:
    python scripts/surprise.py                           # LIBERO overfit model, first episode
    python scripts/surprise.py --dataset pusht --ep 42   # PushT, episode 42
    python scripts/surprise.py --ep 0 --save output/surprise.png
"""
import argparse, json, sys
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
from utils import get_img_preprocessor


def load_model(ckpt_name="lewm", epoch=None):
    ckpt_dir = Path(swm.data.utils.get_cache_dir()) / "checkpoints"
    weight_files = sorted(ckpt_dir.glob(f"{ckpt_name}_weights_epoch_*.pt"))
    if not weight_files:
        raise FileNotFoundError(f"No checkpoints for '{ckpt_name}' in {ckpt_dir}")
    if epoch is not None:
        weight_files = [f for f in weight_files if f"epoch_{epoch}" in f.stem]
    weight_path = weight_files[-1]
    cfg_stem = weight_path.stem.replace("_weights_epoch_", "_config_epoch_")
    cfg = OmegaConf.create(json.loads((weight_path.parent / f"{cfg_stem}.json").read_text()))
    model = hydra.utils.instantiate(cfg)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    model.requires_grad_(False)
    return model


def load_libero_episode(ep_idx=0):
    from libero_data import LiberoDataset
    ds = LiberoDataset(
        path="/home/cyborg/WM/LIBERO-datasets/libero_10",
        frameskip=5, num_steps=4, image_key="agentview_rgb", max_episodes=10,
        transform=get_img_preprocessor("pixels", "pixels", 128),
    )
    return ds


def load_pusht_episode(ep_idx=0):
    ds = swm.data.HDF5Dataset(
        "pusht_expert_train", frameskip=5, num_steps=4,
        transform=get_img_preprocessor("pixels", "pixels", 224),
    )
    return ds


def compute_surprise(model, dataset, ep_idx=0, ctx_len=3):
    """Slide through an episode, compute 1-step-ahead prediction error at each position."""
    ep_len = dataset._episode_lengths[ep_idx]

    # Build trajectory by sliding window
    num_steps = dataset.num_steps  # 4 = ctx_len + n_preds
    n_samples = max(0, ep_len // dataset.frameskip - num_steps)

    surprise = []
    for i in range(n_samples):
        idx = int(dataset.offsets[ep_idx] // dataset.frameskip + i)
        if idx >= len(dataset):
            break
        item = dataset[idx]
        batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()}

        with torch.no_grad():
            out = model.encode(batch)
            emb = out["emb"]
            act_emb = out["act_emb"]

            ctx_emb = emb[:, :ctx_len]
            ctx_act = act_emb[:, :ctx_len]
            pred = model.predict(ctx_emb, ctx_act)
            tgt = emb[:, 1:ctx_len+1]  # next-step targets

            err = (pred - tgt).pow(2).mean().item()

        surprise.append(err)

    return np.array(surprise)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="libero", choices=["libero", "pusht"])
    parser.add_argument("--ep", type=int, default=0, help="Episode index to analyze")
    parser.add_argument("--save", type=str, default=None, help="Save plot to path")
    parser.add_argument("--ckpt", default="lewm", help="Checkpoint name")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch")
    parser.add_argument("--threshold", type=float, default=3.0, help="Surprise threshold (std multiples)")
    args = parser.parse_args()

    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt, args.epoch)

    print(f"Loading dataset: {args.dataset}, episode {args.ep}")
    if args.dataset == "libero":
        ds = load_libero_episode(args.ep)
    else:
        ds = load_pusht_episode(args.ep)

    surprise = compute_surprise(model, ds, args.ep)
    if len(surprise) == 0:
        print("No samples in this episode!")
        return

    mu, sigma = surprise.mean(), surprise.std()
    anomalies = surprise > mu + args.threshold * sigma

    print(f"\n  Episode {args.ep}: {len(surprise)} steps")
    print(f"  Surprise: mean={mu:.6f}, std={sigma:.6f}, max={surprise.max():.6f}")
    print(f"  Anomalies (>{args.threshold}σ): {anomalies.sum()}/{len(surprise)} steps")

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1, 1]})

    ax = axes[0]
    x = np.arange(len(surprise))
    ax.plot(x, surprise, color="steelblue", linewidth=1, alpha=0.8)
    ax.axhline(y=mu, color="gray", linestyle="--", alpha=0.5, label=f"mean={mu:.4f}")
    ax.axhline(y=mu + args.threshold * sigma, color="red", linestyle="--", alpha=0.5,
               label=f"{args.threshold}σ threshold")
    if anomalies.any():
        ax.scatter(x[anomalies], surprise[anomalies], color="red", s=40, zorder=5, label="anomalies")
    ax.set_ylabel("Surprise (MSE)")
    ax.set_title(f"Surprise along trajectory — {args.dataset} episode {args.ep} ({len(surprise)} steps)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.fill_between(x, surprise, mu, color="steelblue", alpha=0.3)
    ax.fill_between(x, surprise, mu, where=anomalies, color="red", alpha=0.6)
    ax.set_ylabel("Surprise residual")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    ax = axes[2]
    ax.hist(surprise, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(x=mu, color="gray", linestyle="--", alpha=0.5, label=f"μ={mu:.6f}")
    ax.axvline(x=mu + args.threshold * sigma, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Surprise (MSE)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = args.save or f"output/surprise_{args.dataset}_ep{args.ep}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
