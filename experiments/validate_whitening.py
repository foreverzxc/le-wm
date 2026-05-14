#!/usr/bin/env python3
"""综合性验证：白化微调模型 vs 预训练基线。

测试项:
  1. 各向同性：特征值谱、条件数、有效维度
  2. 预测误差：embedding-level MSE
  3. 残差分析：预测残差的各向同性
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import stable_worldmodel as swm
import stable_pretraining as spt
from utils import get_img_preprocessor, get_column_normalizer


def collect_embeddings(model, n_batches=50, batch_size=32):
    """收集 encoder 输出和白化后输出 + 预测值."""
    dataset = swm.data.HDF5Dataset(
        name="pusht_expert_train", frameskip=5, num_steps=4,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        cache_dir=str(Path.home() / ".stable_worldmodel"),
    )
    transform = get_img_preprocessor(source='pixels', target='pixels', img_size=224)
    normalizer = get_column_normalizer(dataset, "action", "action")
    dataset.transform = spt.data.transforms.Compose(transform, normalizer)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )

    model.eval()
    device = next(model.parameters()).device

    enc = []    # encoder output (after projector)
    pred = []   # predictor output
    tgt = []    # target (ground truth encoder output)
    act = []    # actions

    with torch.no_grad():
        collected = 0
        for batch in loader:
            if collected >= n_batches:
                break

            pixels = batch["pixels"].float().to(device)
            actions = torch.nan_to_num(batch["action"].float().to(device))
            b, t, c, h, w = pixels.shape

            # Encode
            output = model.encode({"pixels": pixels, "action": actions})
            emb = output["emb"]  # (B, T, D) — whitened if applicable
            act_emb = output["act_emb"]

            # Predict next embedding
            ctx = emb[:, :3]  # history_size = 3
            ctx_act = act_emb[:, :3]
            pred_emb = model.predict(ctx, ctx_act)  # (B, 1, D)

            enc.append(emb[:, -1:].cpu())   # last timestep (what predictor predicts)
            pred.append(pred_emb.cpu())
            tgt.append(emb[:, 1:].cpu())    # align with pred_emb

            collected += 1

    enc = torch.cat(enc, dim=0).reshape(-1, emb.shape[-1])
    pred = torch.cat(pred, dim=0).reshape(-1, emb.shape[-1])
    tgt = torch.cat(tgt, dim=0).reshape(-1, emb.shape[-1])

    print(f"  Collected {enc.shape[0]} embeddings, dim={enc.shape[1]}")
    return enc.numpy(), pred.numpy(), tgt.numpy()


def compute_isotropy(data, name=""):
    """Compute isotropy metrics."""
    c = data - data.mean(0)
    cov = np.cov(c, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]

    cn = ev.max() / max(ev.min(), 1e-10)
    ed = (ev.sum() ** 2) / (ev ** 2).sum()
    D = ev.shape[0]
    cumvar = np.cumsum(ev) / ev.sum()
    k90 = int((cumvar < 0.90).sum()) + 1
    k95 = int((cumvar < 0.95).sum()) + 1

    # Correlation stats
    std = np.sqrt(np.diag(cov))
    corr = cov / (std[:, None] * std[None, :])
    mask = ~np.eye(D, dtype=bool)
    off = corr[mask]

    print(f"  [{name:>20s}] cond#={cn:>12.1f}  eff_dim={ed:>6.1f}/{D}  "
          f"K@90={k90:>3d}  K@95={k95:>3d}  "
          f"|r|_mean={np.abs(off).mean():.4f}  |r|_max={np.abs(off).max():.4f}")

    return {"cn": cn, "ed": ed, "k90": k90, "k95": k95, "ev": ev}


def main():
    ckpt_base = Path.home() / ".stable_worldmodel"
    pretrained = ckpt_base / "pusht" / "lewm_object.ckpt"
    finetuned = ckpt_base / "experiments" / "finetune_whitening" / "finetune_whitening_final_object.ckpt"

    print("Loading models...")
    model_pt = torch.load(pretrained, map_location="cuda", weights_only=False)
    model_ft = torch.load(finetuned, map_location="cuda", weights_only=False)

    print("\n=== Model A: Pre-trained (SIGReg) ===")
    enc_pt, pred_pt, tgt_pt = collect_embeddings(model_pt)

    print("\n=== Model B: Whitening Fine-tuned ===")
    enc_ft, pred_ft, tgt_ft = collect_embeddings(model_ft)

    print("\n" + "=" * 80)
    print("  TEST 1: EMBEDDING ISOTROPY")
    print("=" * 80)
    compute_isotropy(enc_pt, "Pre-trained enc")
    compute_isotropy(enc_ft, "Whitening enc")

    print("\n" + "=" * 80)
    print("  TEST 2: PREDICTION ERROR")
    print("=" * 80)
    mse_pt = np.mean((pred_pt - tgt_pt) ** 2)
    mse_ft = np.mean((pred_ft - tgt_ft) ** 2)
    print(f"  Pre-trained  pred MSE: {mse_pt:.6f}")
    print(f"  Whitening    pred MSE: {mse_ft:.6f}")
    print(f"  Improvement:           {mse_pt/mse_ft:.1f}x lower")

    print("\n" + "=" * 80)
    print("  TEST 3: PREDICTION RESIDUAL ISOTROPY")
    print("=" * 80)
    residual_pt = pred_pt - tgt_pt
    residual_ft = pred_ft - tgt_ft
    compute_isotropy(residual_pt, "Pre-trained res")
    compute_isotropy(residual_ft, "Whitening res")

    print("\n" + "=" * 80)
    print("  TEST 4: PER-DIMENSION MSE")
    print("=" * 80)
    dim_mse_pt = ((pred_pt - tgt_pt) ** 2).mean(0)
    dim_mse_ft = ((pred_ft - tgt_ft) ** 2).mean(0)
    print(f"  Pre-trained:  mean={dim_mse_pt.mean():.6f}  std={dim_mse_pt.std():.6f}  "
          f"min={dim_mse_pt.min():.6f}  max={dim_mse_pt.max():.6f}")
    print(f"  Whitening:    mean={dim_mse_ft.mean():.6f}  std={dim_mse_ft.std():.6f}  "
          f"min={dim_mse_ft.min():.6f}  max={dim_mse_ft.max():.6f}")


if __name__ == "__main__":
    main()
