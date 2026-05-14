#!/usr/bin/env python3
"""Test isotropy of LeWM embeddings on PushT dataset.

Runs four tests:
1. Covariance eigenvalue spectrum
2. Condition number + anisotropy metrics
3. Random 1D projection histograms vs Gaussian
4. Correlation matrix (off-diagonal independence)

Usage:
    python test_isotropy.py
    python test_isotropy.py --model data/pusht/lewm_object.ckpt
    python test_isotropy.py --n-samples 500
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── ensure project root is importable ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from jepa import JEPA


# ============================================================
#  Helper: load model
# ============================================================

def load_model(ckpt_path: str) -> JEPA:
    """Load a JEPA model from an _object.ckpt checkpoint."""
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.eval()
    model.requires_grad_(False)
    return model


# ============================================================
#  Helper: extract embeddings from dataset
# ============================================================

def extract_embeddings(model: JEPA, cache_dir: str, n_samples: int = 500,
                      dataset_name: str = "pusht_expert_train",
                      keys_to_load: list = None,
                      keys_to_cache: list = None,
                      frameskip: int = 5,
                      num_steps: int = 4,
                      normalize_cols: list = None):
    """Run the encoder on n_samples from dataset and collect embeddings.
    
    Uses the exact same dataset initialization and transforms as training.
    """
    import stable_worldmodel as swm
    import stable_pretraining as spt
    from utils import get_column_normalizer, get_img_preprocessor

    if keys_to_load is None:
        keys_to_load = ["pixels", "action", "proprio", "state"]
    if keys_to_cache is None:
        keys_to_cache = ["action", "proprio", "state"]
    if normalize_cols is None:
        normalize_cols = ["action", "proprio", "state"]

    dataset = swm.data.HDF5Dataset(
        name=dataset_name,
        frameskip=frameskip,
        num_steps=num_steps,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        cache_dir=cache_dir,
    )

    # Apply same transforms as train.py
    transforms_list = [get_img_preprocessor(source='pixels', target='pixels', img_size=224)]
    for col in normalize_cols:
        if col in keys_to_load:
            normalizer = get_column_normalizer(dataset, col, col)
            transforms_list.append(normalizer)
    
    transform = spt.data.transforms.Compose(*transforms_list)
    dataset.transform = transform

    all_embeddings = []
    all_pred_embeddings = []
    count = 0

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    with torch.no_grad():
        for batch in loader:
            if count >= n_samples:
                break

            # Replace NaN values with 0 (same as training)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

            # Encode
            output = model.encode(batch)
            emb = output["emb"]  # (B, T, D)
            all_embeddings.append(emb)

            # Also get predicted embeddings
            if "act_emb" in output:
                ctx_len = 3  # history_size from config
                ctx_emb = emb[:, :ctx_len]
                ctx_act = output["act_emb"][:, :ctx_len]
                pred_emb = model.predict(ctx_emb, ctx_act)  # (B, T, D)
                all_pred_embeddings.append(pred_emb)

            count += emb.size(0)

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, T, D)
    embeddings = embeddings[:n_samples]

    if all_pred_embeddings:
        pred_embeddings = torch.cat(all_pred_embeddings, dim=0)[:n_samples]
    else:
        pred_embeddings = None

    return embeddings, pred_embeddings


# ============================================================
#  Test 1: Eigenvalue spectrum
# ============================================================

def test_eigenvalue_spectrum(emb_flat: np.ndarray, save_dir: Path):
    """Plot eigenvalue spectrum of the covariance matrix."""
    emb_centered = emb_flat - emb_flat.mean(axis=0)
    cov = np.cov(emb_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1a. Eigenvalue spectrum
    axes[0].plot(eigenvalues, linewidth=1.5)
    axes[0].axhline(y=eigenvalues.mean(), color='r', linestyle='--',
                    label=f'mean={eigenvalues.mean():.4f}')
    axes[0].set_xlabel("Dimension index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Covariance Eigenvalue Spectrum")
    axes[0].legend()

    # 1b. Explained variance ratio
    var_ratio = eigenvalues / eigenvalues.sum()
    axes[1].bar(range(len(eigenvalues)), var_ratio, width=1.0)
    axes[1].set_xlabel("Dimension index")
    axes[1].set_ylabel("Variance Ratio")
    axes[1].set_title("Variance Explained per Dimension")

    # 1c. Cumulative variance
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    axes[2].plot(cumvar, linewidth=1.5)
    axes[2].axhline(y=0.9, color='r', linestyle='--', label='90%')
    axes[2].axhline(y=0.95, color='g', linestyle='--', label='95%')
    axes[2].axhline(y=0.99, color='orange', linestyle='--', label='99%')
    axes[2].set_xlabel("Number of Components")
    axes[2].set_ylabel("Cumulative Variance Explained")
    axes[2].set_title("Cumulative Variance")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "01_eigenvalue_spectrum.png", dpi=150, bbox_inches='tight')
    plt.close()

    return eigenvalues


# ============================================================
#  Test 2: Quantitative anisotropy metrics
# ============================================================

def test_anisotropy_metrics(eigenvalues: np.ndarray, D: int):
    """Compute and print anisotropy metrics from eigenvalues."""
    condition_number = eigenvalues.max() / max(eigenvalues.min(), 1e-10)
    max_avg_ratio = eigenvalues.max() / eigenvalues.mean()
    effective_dim = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    variance_cv = eigenvalues.std() / eigenvalues.mean()  # coefficient of variation

    print("\n" + "=" * 60)
    print("  ANISOTROPY METRICS")
    print("=" * 60)
    print(f"  Embedding dimensionality:       {D}")
    print(f"  Condition number (λ_max/λ_min): {condition_number:.2f}")
    print(f"    → Isotropic: ≈1.0 | Acceptable: <10 | Anisotropic: >100")
    print(f"  Max/Avg eigenvalue ratio:       {max_avg_ratio:.2f}")
    print(f"    → Isotropic: ≈1.0")
    print(f"  Effective dimensionality:       {effective_dim:.1f} / {D}")
    print(f"    → Isotropic: ≈{D} | Collapsed: ≪{D}")
    print(f"  Eigenvalue CoV (std/mean):      {variance_cv:.4f}")
    print(f"    → Isotropic: ≈0.0")
    print(f"  λ_max:                          {eigenvalues.max():.4f}")
    print(f"  λ_min:                          {eigenvalues.min():.6f}")
    print(f"  λ_mean:                         {eigenvalues.mean():.4f}")
    print("=" * 60)

    return {
        "condition_number": condition_number,
        "max_avg_ratio": max_avg_ratio,
        "effective_dim": effective_dim,
        "variance_cv": variance_cv,
    }


# ============================================================
#  Test 3: Random 1D projections vs Gaussian
# ============================================================

def test_random_projections(emb_flat: np.ndarray, save_dir: Path, n_projs: int = 8):
    """Project embeddings onto random directions and compare to Gaussian."""
    emb_centered = emb_flat - emb_flat.mean(axis=0)
    D = emb_centered.shape[1]
    sigma = np.sqrt(np.mean(np.var(emb_centered, axis=0)))  # average std

    ncols = min(4, n_projs)
    nrows = (n_projs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i in range(n_projs):
        # Random unit vector
        v = np.random.randn(D)
        v = v / np.linalg.norm(v)

        proj = emb_centered @ v

        axes[i].hist(proj, bins=60, density=True, alpha=0.7,
                     color='steelblue', label='Projected emb')
        x = np.linspace(proj.min(), proj.max(), 200)
        axes[i].plot(x, np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi)),
                     'r-', linewidth=2, label=f'N(0, {sigma:.2f}²)')

        # Shapiro-Wilk test for normality (on a subsample if too large)
        from scipy.stats import shapiro
        sample = np.random.choice(proj, min(5000, len(proj)), replace=False)
        stat, pval = shapiro(sample)
        axes[i].set_title(f'Proj {i + 1}  (SW p={pval:.3f})', fontsize=10)
        axes[i].legend(fontsize=8)

    for j in range(n_projs, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("1D Random Projections vs Isotropic Gaussian", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "03_random_projections.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
#  Test 4: Correlation matrix (off-diagonal independence)
# ============================================================

def test_correlation_matrix(emb_flat: np.ndarray, save_dir: Path):
    """Visualize the correlation matrix of embedding dimensions."""
    emb_centered = emb_flat - emb_flat.mean(axis=0)
    cov = np.cov(emb_centered, rowvar=False)

    # Correlation matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / (std[:, None] * std[None, :])

    # Off-diagonal statistics
    mask = ~np.eye(corr.shape[0], dtype=bool)
    off_diag = corr[mask]
    print("\n  Correlation matrix off-diagonal stats:")
    print(f"    mean:  {off_diag.mean():.4f}  (isotropic → ≈0)")
    print(f"    std:   {off_diag.std():.4f}")
    print(f"    max:   {np.abs(off_diag).max():.4f}")
    print(f"    |r|>0.3: {(np.abs(off_diag) > 0.3).sum()} / {off_diag.size}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full correlation matrix
    im = axes[0].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title("Correlation Matrix")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Dimension")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Histogram of off-diagonal values
    axes[1].hist(off_diag, bins=100, density=True, color='steelblue', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel("Correlation value")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Off-diagonal Correlation Distribution")
    axes[1].axvline(x=off_diag.mean(), color='orange', linestyle='--',
                    label=f'mean={off_diag.mean():.4f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "04_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
#  Test 5: Per-dimension statistics
# ============================================================

def test_per_dimension_stats(emb_flat: np.ndarray, save_dir: Path):
    """Check per-dimension mean and variance."""
    means = emb_flat.mean(axis=0)
    variances = emb_flat.var(axis=0)
    D = emb_flat.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(range(D), means, width=1.0, color='steelblue', alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Mean")
    axes[0].set_title("Per-dimension Mean (isotropic → ≈0)")

    axes[1].bar(range(D), variances, width=1.0, color='steelblue', alpha=0.7)
    axes[1].axhline(y=variances.mean(), color='r', linestyle='--',
                    label=f'mean={variances.mean():.4f}')
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Variance")
    axes[1].set_title("Per-dimension Variance (isotropic → equal)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "05_per_dimension_stats.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Per-dimension mean:  min={means.min():.4f}, max={means.max():.4f}, std={means.std():.4f}")
    print(f"  Per-dimension var:   min={variances.min():.4f}, max={variances.max():.4f}, std={variances.std():.4f}")
    print(f"  Variance CoV (std/mean): {variances.std() / variances.mean():.4f}  (isotropic → ≈0)")


# ============================================================
#  Test 6: CDF of L2 norms vs chi distribution
# ============================================================

def test_norm_distribution(emb_flat: np.ndarray, save_dir: Path):
    """Compare the distribution of L2 norms to the chi distribution."""
    from scipy.stats import chi

    emb_centered = emb_flat - emb_flat.mean(axis=0)
    D = emb_centered.shape[1]

    # Compute L2 norms
    norms = np.linalg.norm(emb_centered, axis=1)

    # Under isotropic N(0, σ²I), the norm follows chi(D) scaled by σ
    sigma_est = np.sqrt(np.mean(np.var(emb_centered, axis=0)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram vs chi PDF
    axes[0].hist(norms, bins=60, density=True, alpha=0.7, color='steelblue', label='Empirical')
    x = np.linspace(0, norms.max(), 200)
    chi_pdf = chi.pdf(x / sigma_est, D) / sigma_est
    axes[0].plot(x, chi_pdf, 'r-', linewidth=2, label=f'χ({D}) × σ={sigma_est:.2f}')
    axes[0].set_xlabel("L2 Norm")
    axes[0].set_ylabel("Density")
    axes[0].set_title("L2 Norm Distribution vs Chi")
    axes[0].legend()

    # Q-Q style: CDF comparison
    from scipy.stats import norm as sp_norm
    sorted_norms = np.sort(norms)
    ecdf = np.arange(1, len(sorted_norms) + 1) / len(sorted_norms)
    theoretical_cdf = chi.cdf(sorted_norms / sigma_est, D)

    axes[1].plot(ecdf, theoretical_cdf, linewidth=1.5)
    axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect match')
    axes[1].set_xlabel("Empirical CDF")
    axes[1].set_ylabel("Theoretical χ CDF")
    axes[1].set_title("Q-Q: L2 Norm CDF vs Chi CDF")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "06_norm_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
#  Test 7: Predicted vs target embedding isotropy comparison
# ============================================================

def test_pred_vs_target(embeddings: np.ndarray, pred_embeddings: np.ndarray, save_dir: Path):
    """Compare isotropy of predicted embeddings vs target embeddings."""
    # Ensure matching shapes (predictions may have fewer timesteps)
    min_len = min(len(embeddings), len(pred_embeddings))
    embeddings = embeddings[:min_len]
    pred_embeddings = pred_embeddings[:min_len]

    emb_centered = embeddings - embeddings.mean(axis=0)
    pred_centered = pred_embeddings - pred_embeddings.mean(axis=0)

    cov_emb = np.cov(emb_centered, rowvar=False)
    cov_pred = np.cov(pred_centered, rowvar=False)

    ev_emb = np.sort(np.linalg.eigvalsh(cov_emb))[::-1]
    ev_pred = np.sort(np.linalg.eigvalsh(cov_pred))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(ev_emb, label='Target (encoder output)', linewidth=1.5)
    axes[0].plot(ev_pred, label='Predicted', linewidth=1.5, linestyle='--')
    axes[0].set_xlabel("Dimension index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Eigenvalue Spectrum: Target vs Predicted")
    axes[0].legend()

    # Also plot residual isotropy
    residual = pred_embeddings - embeddings
    res_centered = residual - residual.mean(axis=0)
    cov_res = np.cov(res_centered, rowvar=False)
    ev_res = np.sort(np.linalg.eigvalsh(cov_res))[::-1]

    axes[1].plot(ev_emb, label='Target', linewidth=1.5)
    axes[1].plot(ev_res, label='Residual (pred - target)', linewidth=1.5, linestyle='--')
    axes[1].set_xlabel("Dimension index")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].set_title("Eigenvalue Spectrum: Target vs Residual")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "07_pred_vs_target_isotropy.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Print metrics
    cn_emb = ev_emb.max() / max(ev_emb.min(), 1e-10)
    cn_pred = ev_pred.max() / max(ev_pred.min(), 1e-10)
    cn_res = ev_res.max() / max(ev_res.min(), 1e-10)
    ed_emb = (ev_emb.sum() ** 2) / (ev_emb ** 2).sum()
    ed_pred = (ev_pred.sum() ** 2) / (ev_pred ** 2).sum()
    ed_res = (ev_res.sum() ** 2) / (ev_res ** 2).sum()

    print("\n  Target vs Predicted isotropy:")
    print(f"  {'':20s} {'Target':>12s} {'Predicted':>12s} {'Residual':>12s}")
    print(f"  {'Condition number':20s} {cn_emb:12.2f} {cn_pred:12.2f} {cn_res:12.2f}")
    print(f"  {'Effective dim':20s} {ed_emb:12.1f} {ed_pred:12.1f} {ed_res:12.1f}")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test isotropy of LeWM embeddings")
    parser.add_argument("--model", type=str, default="data/pusht/lewm_object.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="pusht_expert_train",
                        help="Dataset name")
    parser.add_argument("--cache-dir", type=str, default="data",
                        help="Cache directory for dataset")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples to use")
    parser.add_argument("--output-dir", type=str, default="isotropy_results",
                        help="Output directory for plots and results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LeWM Embedding Isotropy Test")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Dataset:     {args.data}")
    print(f"  Cache dir:   {args.cache_dir}")
    print(f"  N samples:   {args.n_samples}")
    print(f"  Output dir:  {args.output_dir}")

    # ── Dataset-specific config ──
    dataset_configs = {
        "pusht_expert_train": {
            "dataset_name": "pusht_expert_train",
            "keys_to_load": ["pixels", "action", "proprio", "state"],
            "keys_to_cache": ["action", "proprio", "state"],
            "normalize_cols": ["action", "proprio", "state"],
        },
        "reacher": {
            "dataset_name": "reacher",
            "keys_to_load": ["pixels", "action", "observation"],
            "keys_to_cache": ["action", "observation"],
            "normalize_cols": ["action", "observation"],
        },
        "tworoom": {
            "dataset_name": "tworoom",
            "keys_to_load": ["pixels", "action"],
            "keys_to_cache": ["action"],
            "normalize_cols": ["action"],
        },
    }
    ds_cfg = dataset_configs.get(args.data, {
        "dataset_name": args.data,
        "keys_to_load": None,
        "keys_to_cache": None,
        "normalize_cols": None,
    })

    # ── Load model ──
    print("\n[1/8] Loading model...")
    model = load_model(args.model)
    print(f"  Model loaded. Type: {type(model).__name__}")

    # ── Extract embeddings ──
    print(f"\n[2/8] Extracting embeddings (n={args.n_samples})...")
    embeddings, pred_embeddings = extract_embeddings(
        model, args.cache_dir, n_samples=args.n_samples,
        frameskip=5, num_steps=4, **ds_cfg)
    print(f"  Embeddings shape: {embeddings.shape}")  # (N, T, D)

    # Flatten (B, T, D) → (B*T, D)
    emb_flat = embeddings.reshape(-1, embeddings.shape[-1]).numpy()
    D = emb_flat.shape[1]
    print(f"  Flattened shape: {emb_flat.shape}")
    print(f"  Embedding dim:   {D}")

    # ── Test 1: Eigenvalue spectrum ──
    print("\n[4/8] Test 1: Eigenvalue spectrum...")
    eigenvalues = test_eigenvalue_spectrum(emb_flat, output_dir)
    print("  → Saved 01_eigenvalue_spectrum.png")

    # ── Test 2: Anisotropy metrics ──
    print("\n[5/8] Test 2: Anisotropy metrics...")
    metrics = test_anisotropy_metrics(eigenvalues, D)

    # ── Test 3: Random projections ──
    print("\n[6/8] Test 3: Random projections vs Gaussian...")
    test_random_projections(emb_flat, output_dir, n_projs=8)
    print("  → Saved 03_random_projections.png")

    # ── Test 4: Correlation matrix ──
    print("\n[7/8] Test 4: Correlation matrix...")
    test_correlation_matrix(emb_flat, output_dir)
    print("  → Saved 04_correlation_matrix.png")

    # ── Test 5: Per-dimension stats ──
    test_per_dimension_stats(emb_flat, output_dir)
    print("  → Saved 05_per_dimension_stats.png")

    # ── Test 6: Norm distribution ──
    test_norm_distribution(emb_flat, output_dir)
    print("  → Saved 06_norm_distribution.png")

    # ── Test 7: Predicted vs Target ──
    if pred_embeddings is not None:
        pred_flat = pred_embeddings.reshape(-1, pred_embeddings.shape[-1]).numpy()
        test_pred_vs_target(emb_flat, pred_flat, output_dir)
        print("  → Saved 07_pred_vs_target_isotropy.png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    is_isotropic = metrics["condition_number"] < 10 and metrics["effective_dim"] > 0.5 * D

    if metrics["condition_number"] < 5:
        verdict = "✅ HIGHLY ISOTROPIC — eigenvalues are nearly uniform"
    elif metrics["condition_number"] < 20:
        verdict = "⚠️  MODERATELY ISOTROPIC — some dimensions dominate"
    elif metrics["condition_number"] < 100:
        verdict = "❌ WEAKLY ISOTROPIC — significant anisotropy detected"
    else:
        verdict = "❌ ANISOTROPIC — embedding space is highly non-uniform"

    print(f"\n  Condition number: {metrics['condition_number']:.2f}")
    print(f"  Effective dim:    {metrics['effective_dim']:.1f} / {D}")
    print(f"  Verdict:          {verdict}")

    print(f"\n  All results saved to: {output_dir}/")
    print("  Plots: 01-07_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()