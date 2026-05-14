#!/usr/bin/env python3
"""Embedding dimension sweep experiment for LeWM.

Trains LeWM on PushT with different embed_dim values and evaluates isotropy.

CONSERVATIVE resource settings for RTX 3050 6GB + 16GB RAM:
  - batch_size=32, num_workers=2, no persistent_workers
  - Train one model at a time, clean up between runs
  - max_epochs=3

Usage:
    python run_embed_dim_sweep.py
"""

import gc
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from test_isotropy import load_model, extract_embeddings


def train_one(embed_dim: int, output_dir: Path, max_epochs: int = 3):
    """Train a single LeWM model with given embed_dim on PushT.
    
    Very conservative resource settings to avoid OOM on RTX 3050 + 16GB RAM.
    """
    
    print(f"\n{'='*70}")
    print(f"  Training embed_dim={embed_dim} | epochs={max_epochs} | batch=32")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Dataset ──
    dataset = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels", "action", "proprio", "state"],
        keys_to_cache=["action", "proprio", "state"],
        transform=None,
    )
    
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=224)]
    for col in ["action", "proprio", "state"]:
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)
    
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    
    rnd_gen = torch.Generator().manual_seed(3072)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[0.9, 0.1], generator=rnd_gen
    )
    
    # CONSERVATIVE: batch=32, workers=2, no persistent_workers
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True, 
        drop_last=True, generator=rnd_gen,
        num_workers=2, persistent_workers=False, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=32, shuffle=False,
        num_workers=2, persistent_workers=False, pin_memory=True,
    )
    
    # ── Model ──
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    effective_act_dim = 5 * 2  # frameskip=5, action_dim=2
    
    predictor = ARPredictor(
        num_frames=3, input_dim=embed_dim, hidden_dim=embed_dim,
        output_dim=embed_dim, depth=6, heads=16, mlp_dim=2048,
        dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d,
    )
    
    predictor_proj = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d,
    )
    
    world_model = JEPA(
        encoder=encoder, predictor=predictor,
        action_encoder=action_encoder, projector=projector,
        pred_proj=predictor_proj,
    )
    
    # ── Config for forward ──
    class Cfg:
        class wm:
            history_size = 3
            num_preds = 1
            action_dim = 2
        class loss:
            class sigreg:
                weight = 0.09
    
    sigreg = SIGReg(knots=17, num_proj=1024)
    
    from train import lejepa_forward
    
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": {"type": "AdamW", "lr": 5e-5, "weight_decay": 1e-3},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    world_model = spt.Module(
        model=world_model, sigreg=sigreg,
        forward=partial(lejepa_forward, cfg=Cfg()),
        optim=optimizers,
    )
    
    # ── Train ──
    ckpt_dir = output_dir / f"embed_dim_{embed_dim}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    object_dump_callback = ModelObjectCallBack(
        dirpath=ckpt_dir, filename="lewm_object", epoch_interval=1,
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs, devices=1, accelerator="gpu",
        precision="bf16-mixed", gradient_clip_val=1.0,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=10,  # Don't validate on full dataset
    )
    
    start_time = time.time()
    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data_module,
    )
    manager()
    elapsed = time.time() - start_time
    
    print(f"  Training completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    
    # Find the saved checkpoint
    ckpt_files = list(ckpt_dir.glob("lewm_object*.ckpt"))
    if ckpt_files:
        ckpt_path = sorted(ckpt_files)[-1]
        print(f"  Checkpoint: {ckpt_path}")
        return ckpt_path
    else:
        print(f"  WARNING: No checkpoint found in {ckpt_dir}")
        return None


def evaluate_isotropy(ckpt_path: Path, label: str):
    """Run isotropy evaluation on a trained model."""
    print(f"\n  Evaluating isotropy for {label}...")
    
    model = load_model(str(ckpt_path))
    embeddings, pred_embeddings = extract_embeddings(
        model, "data", n_samples=500,
        dataset_name="pusht_expert_train",
        keys_to_load=["pixels", "action", "proprio", "state"],
        keys_to_cache=["action", "proprio", "state"],
        normalize_cols=["action", "proprio", "state"],
    )
    
    emb_flat = embeddings.reshape(-1, embeddings.shape[-1]).numpy()
    D = emb_flat.shape[1]
    
    # Compute metrics
    emb_centered = emb_flat - emb_flat.mean(axis=0)
    cov = np.cov(emb_centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    
    condition_number = eigenvalues.max() / max(eigenvalues.min(), 1e-10)
    effective_dim = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    max_avg_ratio = eigenvalues.max() / eigenvalues.mean()
    eigenvalue_cv = eigenvalues.std() / eigenvalues.mean()
    variance_cv = emb_centered.var(axis=0).std() / emb_centered.var(axis=0).mean()
    
    # Correlation stats
    std = np.sqrt(np.diag(cov))
    corr = cov / (std[:, None] * std[None, :])
    mask = ~np.eye(corr.shape[0], dtype=bool)
    off_diag = corr[mask]
    corr_mean = np.abs(off_diag).mean()
    
    # Prediction loss
    if pred_embeddings is not None:
        pred_flat = pred_embeddings.reshape(-1, pred_embeddings.shape[-1]).numpy()
        min_len = min(len(emb_flat), len(pred_flat))
        mse = np.mean((pred_flat[:min_len] - emb_flat[:min_len]) ** 2)
    else:
        mse = float('nan')
    
    metrics = {
        "label": label,
        "embed_dim": D,
        "condition_number": float(condition_number),
        "effective_dim": float(effective_dim),
        "eff_dim_pct": float(effective_dim / D * 100),
        "max_avg_ratio": float(max_avg_ratio),
        "eigenvalue_cv": float(eigenvalue_cv),
        "variance_cv": float(variance_cv),
        "corr_mean": float(corr_mean),
        "pred_mse": float(mse),
        "eigenvalues": eigenvalues.tolist(),
    }
    
    print(f"    Condition number: {condition_number:.1f}")
    print(f"    Effective dim:    {effective_dim:.1f} / {D} ({effective_dim/D*100:.1f}%)")
    print(f"    Pred MSE:         {mse:.6f}")
    
    return metrics


def cleanup():
    """Force garbage collection and free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  [cleanup] GPU cache cleared, garbage collected")


def generate_comparison_plot(all_metrics: dict, output_dir: Path):
    """Generate comparison plots for the embed_dim sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    keys = list(all_metrics.keys())
    labels = []
    for k in keys:
        m = all_metrics[k]
        if "pretrained" in k:
            labels.append("192 (pretrained\n100 epochs)")
        else:
            labels.append(f"{m['embed_dim']}D\n(3 epochs)")
    
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"][:len(keys)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Effective dimensionality
    ax = axes[0, 0]
    eff_dims = [all_metrics[k]["effective_dim"] for k in keys]
    full_dims = [all_metrics[k]["embed_dim"] for k in keys]
    x = np.arange(len(keys))
    width = 0.35
    ax.bar(x - width/2, full_dims, width, label='Full dim',
           color=[colors[i] for i in range(len(keys))], alpha=0.4)
    ax.bar(x + width/2, eff_dims, width, label='Effective dim',
           color=[colors[i] for i in range(len(keys))])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Dimensionality")
    ax.set_title("Full vs Effective Dimensionality")
    ax.legend()
    for i, (f, e) in enumerate(zip(full_dims, eff_dims)):
        ax.text(i + width/2, e + 1, f'{e:.1f}\n({e/f*100:.0f}%)', ha='center', fontsize=9)
    
    # 2. Condition number
    ax = axes[0, 1]
    cond_nums = [all_metrics[k]["condition_number"] for k in keys]
    bars = ax.bar(x, cond_nums, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Condition Number")
    ax.set_title("Condition Number (lower = more isotropic)")
    ax.set_yscale("log")
    for bar, val in zip(bars, cond_nums):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
    
    # 3. Effective dim percentage
    ax = axes[0, 2]
    eff_pcts = [all_metrics[k]["eff_dim_pct"] for k in keys]
    bars = ax.bar(x, eff_pcts, color=colors)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% (isotropic)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Effective Dim %")
    ax.set_title("Effective Dimensionality / Full Dim")
    ax.legend()
    for bar, val in zip(bars, eff_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 4. Eigenvalue spectrum
    ax = axes[1, 0]
    for i, k in enumerate(keys):
        ev = np.array(all_metrics[k]["eigenvalues"])
        ax.plot(ev, label=labels[i].replace('\n', ' '), linewidth=1.5, color=colors[i])
    ax.set_xlabel("Dimension index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Spectrum")
    ax.legend(fontsize=8)
    
    # 5. Prediction MSE
    ax = axes[1, 1]
    mses = [all_metrics[k]["pred_mse"] for k in keys]
    bars = ax.bar(x, mses, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Prediction MSE")
    ax.set_title("L2 Prediction Loss (lower = better)")
    for bar, val in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    
    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    headers = ["Metric"] + [l.replace('\n', ' ') for l in labels]
    rows = [
        ("Full dim", "embed_dim"),
        ("Eff dim", "effective_dim"),
        ("Eff dim %", "eff_dim_pct"),
        ("Condition #", "condition_number"),
        ("Max/Avg ratio", "max_avg_ratio"),
        ("Variance CoV", "variance_cv"),
        ("|r| mean", "corr_mean"),
        ("Pred MSE", "pred_mse"),
    ]
    for label_str, key in rows:
        row = [label_str]
        for k in keys:
            val = all_metrics[k][key]
            if key == "eff_dim_pct":
                row.append(f"{val:.1f}%")
            elif val >= 100:
                row.append(f"{val:.0f}")
            elif val >= 1:
                row.append(f"{val:.1f}")
            else:
                row.append(f"{val:.4f}")
        table_data.append(row)
    
    table = ax.table(
        cellText=table_data, colLabels=headers,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary Table", fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle("LeWM Embed Dimension Sweep: Isotropy & Prediction Quality\n"
                 "(PushT dataset, 3 epochs each vs 100 epochs pretrained)",
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {output_dir / 'sweep_comparison.png'}")


def main():
    output_dir = Path("sweep_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    
    # ── Phase 1: Train and evaluate each embed_dim SEQUENTIALLY ──
    # Train one, evaluate, cleanup, then next
    for embed_dim in [64, 96]:
        print(f"\n{'#'*70}")
        print(f"  PHASE: embed_dim={embed_dim}")
        print(f"{'#'*70}")
        
        # Train
        ckpt = train_one(embed_dim, output_dir, max_epochs=3)
        
        # Cleanup training resources
        cleanup()
        
        if ckpt is None:
            print(f"  Skipping evaluation for embed_dim={embed_dim} (no checkpoint)")
            continue
        
        # Evaluate
        label = f"embed_dim_{embed_dim}"
        metrics = evaluate_isotropy(ckpt, label)
        all_metrics[str(embed_dim)] = metrics
        
        # Cleanup evaluation resources
        cleanup()
    
    # ── Phase 2: Evaluate the original pretrained 192D model ──
    original_ckpt = Path("data/pusht/lewm_object.ckpt")
    if original_ckpt.exists():
        print(f"\n{'#'*70}")
        print(f"  PHASE: Pretrained 192D evaluation")
        print(f"{'#'*70}")
        metrics = evaluate_isotropy(original_ckpt, "embed_dim_192_pretrained")
        all_metrics["192_pretrained"] = metrics
        cleanup()
    
    # ── Phase 3: Generate comparison plot ──
    if len(all_metrics) >= 2:
        generate_comparison_plot(all_metrics, output_dir)
    
    # ── Save metrics ──
    metrics_clean = {}
    for k, v in all_metrics.items():
        metrics_clean[k] = {kk: vv for kk, vv in v.items() if kk != "eigenvalues"}
    
    with open(output_dir / "sweep_metrics.json", "w") as f:
        json.dump(metrics_clean, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"  Results: {output_dir}/")
    print(f"  Metrics: sweep_metrics.json")
    print(f"  Plot:    sweep_comparison.png")


if __name__ == "__main__":
    main()