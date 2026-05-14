#!/usr/bin/env python3
"""开环预测对比：预训练 vs 白化微调。

在真实轨迹上做多步预测，对比累积误差。
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import stable_worldmodel as swm
import stable_pretraining as spt
from utils import get_img_preprocessor, get_column_normalizer


@torch.no_grad()
def openloop_mse(model, steps=10, n_samples=200):
    """Run open-loop prediction and compute cumulative MSE."""
    dataset = swm.data.HDF5Dataset(
        name="pusht_expert_train", frameskip=5, num_steps=steps + 3,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        cache_dir=str(Path.home() / ".stable_worldmodel"),
    )
    transform = get_img_preprocessor(source='pixels', target='pixels', img_size=224)
    normalizer = get_column_normalizer(dataset, "action", "action")
    dataset.transform = spt.data.transforms.Compose(transform, normalizer)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True,
    )

    model.eval()
    device = next(model.parameters()).device
    all_errors = []

    for i, batch in enumerate(loader):
        if i * 16 >= n_samples:
            break
        pixels = batch["pixels"].float().to(device)
        actions = torch.nan_to_num(batch["action"].float()).to(device)

        # Encode all frames at once
        output = model.encode({"pixels": pixels[:, :3+steps], "action": actions[:, :3+steps]})
        all_embs = output["emb"]  # (B, 3+steps, D)

        # Multi-step open-loop prediction: at step t, predict from all_embs[:, t:t+3]
        pred_list = []
        for t in range(steps):
            ctx = all_embs[:, t:t+3]  # ground truth embeddings as context
            act_emb = output["act_emb"][:, t:t+3]  # ground truth actions
            pred = model.predict(ctx, act_emb)
            pred_list.append(pred[:, -1:])  # (B, 1, D)

        pred_embs = torch.cat(pred_list, dim=1)  # (B, steps, D)
        tgt_embs = all_embs[:, 3:]  # ground truth for frames 3..3+steps

        step_errors = (pred_embs - tgt_embs).pow(2).mean(dim=-1)  # (B, steps)
        all_errors.append(step_errors.cpu())

    all_errors = torch.cat(all_errors, dim=0)
    mean_per_step = all_errors.mean(dim=0)
    cum_errors = all_errors.cumsum(dim=1).mean(dim=0)
    return mean_per_step.numpy(), cum_errors.numpy()


def main():
    base = Path.home() / ".stable_worldmodel"
    models = {
        "Pre-trained (SIGReg)": base / "pusht" / "lewm_object.ckpt",
        "Whitening (unfrozen enc)": base / "experiments" / "finetune_whitening" / "finetune_whitening_object.ckpt",
    }

    results = {}
    for label, ckpt in models.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        model = torch.load(ckpt, map_location="cuda", weights_only=False)
        step_err, cum_err = openloop_mse(model, steps=10, n_samples=200)
        results[label] = (step_err, cum_err)
        for t in range(10):
            print(f"  Step {t+1:2d}: MSE={step_err[t]:.6f}  cum={cum_err[t]:.6f}")
        print(f"  Final cumulative MSE: {cum_err[-1]:.6f}")

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    labels = list(results.keys())
    for t in range(10):
        vals = [results[l][0][t] for l in labels]
        ratio = vals[1] / vals[0] if vals[0] > 0 else float('inf')
        print(f"  Step {t+1:2d}:  {labels[0][:20]:20s}={vals[0]:.6f}  {labels[1][:20]:20s}={vals[1]:.6f}  ratio={ratio:.2f}x")
    cum_vals = [results[l][1][-1] for l in labels]
    print(f"  Final cumulative: {labels[0][:20]:20s}={cum_vals[0]:.6f}  {labels[1][:20]:20s}={cum_vals[1]:.6f}  ratio={cum_vals[1]/cum_vals[0]:.2f}x")


if __name__ == "__main__":
    main()
