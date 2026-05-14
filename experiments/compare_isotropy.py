#!/usr/bin/env python3
"""Minimal isotropy comparison across 4 trained models.

Usage: python experiments/compare_isotropy.py
"""
import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def load_model(ckpt_path):
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.eval()
    model.requires_grad_(False)
    return model

def extract_embeddings_small(model, n_batches=5, batch_size=8):
    """Extract a small batch of embeddings using stable_worldmodel data loader."""
    import stable_worldmodel as swm
    import stable_pretraining as spt
    from utils import get_img_preprocessor
    
    dataset = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels"],
        keys_to_cache=[],
        cache_dir=str(Path.home() / ".stable_worldmodel"),
    )
    
    transform = get_img_preprocessor(source='pixels', target='pixels', img_size=224)
    dataset.transform = transform
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=False,
    )
    
    embeddings_list = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        pixels = batch["pixels"].float()
        b, t, c, h, w = pixels.shape
        pixels = pixels.reshape(b * t, c, h, w)
        
        with torch.no_grad():
            output = model.encoder(pixels, interpolate_pos_encoding=True)
            pixel_emb = output.last_hidden_state[:, 0]
            emb = model.projector(pixel_emb)
            emb = model.whitening(emb)
            emb = emb.reshape(b, t, -1)
            embeddings_list.append(emb)
    
    return torch.cat(embeddings_list, dim=0)[:80]

def compute_isometry(emb):
    """Compute isotropy metrics from flattened embeddings."""
    flat = emb.reshape(-1, emb.shape[-1]).numpy()
    centered = flat - flat.mean(0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    
    condition_num = eigenvalues.max() / max(eigenvalues.min(), 1e-10)
    effective_dim = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    D = eigenvalues.shape[0]
    
    # Top-K variance explained
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    k_90 = int((cumvar < 0.90).sum()) + 1
    k_95 = int((cumvar < 0.95).sum()) + 1
    
    return {
        "eigenvalues": eigenvalues,
        "condition_number": condition_num,
        "effective_dim": effective_dim,
        "D": D,
        "k_90": k_90,
        "k_95": k_95,
        "max_eigval": eigenvalues.max(),
        "min_eigval": eigenvalues.min(),
    }

def main():
    results = {}
    
    experiments = {
        "A: Baseline (SIGReg)": "baseline/lewm_baseline_epoch_3_object.ckpt",
        "B: Whitening": "whitening/lewm_whitening_epoch_3_object.ckpt",
        "C: Noise": "noise/lewm_noise_epoch_3_object.ckpt",
        "D: Whitening+Noise": "whitening_noise/lewm_whitening_noise_epoch_3_object.ckpt",
    }
    
    base = Path.home() / ".stable_worldmodel" / "experiments"
    
    for name, rel_path in experiments.items():
        ckpt = base / rel_path
        if not ckpt.exists():
            print(f"  {name}: checkpoint not found at {ckpt}")
            continue
        
        print(f"Loading {name}...")
        model = load_model(ckpt)
        
        print(f"  Extracting embeddings...")
        emb = extract_embeddings_small(model)
        print(f"  Shape: {emb.shape}")
        
        metrics = compute_isometry(emb)
        results[name] = metrics
        
        print(f"  Condition number: {metrics['condition_number']:.2f}")
        print(f"  Effective dim: {metrics['effective_dim']:.1f} / {metrics['D']}")
        print(f"  K@90%: {metrics['k_90']}, K@95%: {metrics['k_95']}")
        print(f"  Max eigval: {metrics['max_eigval']:.4f}, Min eigval: {metrics['min_eigval']:.6f}")
        print()
    
    # Summary table
    print("=" * 80)
    print(f"{'Experiment':40s} {'Cond#':>10s} {'Eff.Dim':>10s} {'K@90%':>8s} {'K@95%':>8s}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:40s} {m['condition_number']:>10.2f} {m['effective_dim']:>8.1f}/{m['D']:<3d} {m['k_90']:>6d}  {m['k_95']:>6d}")
    print("=" * 80)

if __name__ == "__main__":
    main()
