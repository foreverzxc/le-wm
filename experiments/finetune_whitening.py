#!/usr/bin/env python3
"""微调：从预训练模型加载，附加白化层，仅用 MSE 损失微调。

策略：
  1. 先用预训练模型的 encoder 收集大量 embedding，计算全局白化统计量
  2. 固定白化层（不更新 running 统计量），使白化变换成为定常映射
  3. 微调预测器，使其适应固定的白化空间

用法:
    # 白化微调（推荐）
    python experiments/finetune_whitening.py --mode whitening
    # 白化 + 噪声微调
    python experiments/finetune_whitening.py --mode whitening_noise
    # 控制组：直接 SIGReg 微调
    python experiments/finetune_whitening.py --mode baseline
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import torch
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jepa import JEPA
from module import SIGReg, WhiteningLayer, NoiseInjection
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def make_forward(sigreg_enabled, sigreg_weight, noise_enabled):
    def forward_fn(self, batch, stage, cfg):
        ctx_len = cfg.wm.history_size
        n_preds = cfg.wm.num_preds
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        output = self.model.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]

        if noise_enabled and hasattr(self, "noise_injection") and self.noise_injection is not None:
            emb = self.noise_injection(emb)

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]
        pred_emb = self.model.predict(ctx_emb, ctx_act)

        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

        if sigreg_enabled and hasattr(self, "sigreg") and self.sigreg is not None:
            output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
            output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
        else:
            output["loss"] = output["pred_loss"]

        losses = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
        self.log_dict(losses, on_step=True, sync_dist=True)
        return output

    return forward_fn


def compute_whitening_stats(model, n_samples=2000, batch_size=64):
    """Collect pre-trained model embeddings and compute global mean + cov."""
    dataset = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        frameskip=5, num_steps=4,
        keys_to_load=["pixels"],
        keys_to_cache=[],
        cache_dir=str(Path.home() / ".stable_worldmodel"),
    )
    transform = get_img_preprocessor(source='pixels', target='pixels', img_size=224)
    dataset.transform = transform

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )

    model.eval()
    all_embs = []
    device = next(model.parameters()).device

    with torch.no_grad():
        collected = 0
        for batch in loader:
            pixels = batch["pixels"].float().to(device)
            b, t, c, h, w = pixels.shape
            pixels = pixels.reshape(b * t, c, h, w)

            output = model.encoder(pixels, interpolate_pos_encoding=True)
            pixel_emb = output.last_hidden_state[:, 0]
            emb = model.projector(pixel_emb)
            all_embs.append(emb.cpu())
            collected += b * t
            if collected >= n_samples:
                break

    all_embs = torch.cat(all_embs, dim=0)
    mean = all_embs.mean(0)
    centered = all_embs - mean
    cov = (centered.T @ centered) / (centered.size(0) - 1)
    print(f"Collected {all_embs.size(0)} embeddings. Cov shape: {cov.shape}")
    return mean, cov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "whitening", "whitening_noise"], default="whitening")
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--pretrained-ckpt", type=str,
                        default=str(Path.home() / ".stable_worldmodel" / "pusht" / "lewm_object.ckpt"))
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--limit-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n-stats-samples", type=int, default=2000)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="冻结 encoder 权重，只训练 predictor")
    args = parser.parse_args()

    mode = args.mode
    is_whitening = "whitening" in mode
    is_noise = "noise" in mode
    is_baseline = mode == "baseline"
    exp_name = f"finetune_{mode}"

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Pretrained: {args.pretrained_ckpt}")
    print(f"{'='*60}\n")

    # ── Load pre-trained model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa_model = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)

    # Freeze encoder if requested
    if args.freeze_encoder:
        print("Freezing encoder (only predictor will be trained)...")
        for name, param in jepa_model.encoder.named_parameters():
            param.requires_grad_(False)
        print(f"  Frozen {sum(1 for _ in jepa_model.encoder.parameters())} encoder parameters")

    if is_whitening:
        embed_dim = jepa_model.projector.net[-1].out_features
        print("Computing global whitening statistics from pre-trained model...")
        mean, cov = compute_whitening_stats(jepa_model, n_samples=args.n_stats_samples)

        # Initialize WhiteningLayer with pre-computed statistics and FREEZE them
        whitening = WhiteningLayer(dim=embed_dim, frozen=True)
        whitening.running_mean.copy_(mean)
        whitening.running_cov.copy_(cov)
        jepa_model.whitening = whitening

        print(f"  WhiteningLayer initialized with global stats (frozen)")
        print(f"  mean norm: {mean.norm():.4f}, cov trace: {cov.trace():.4f}")
    else:
        jepa_model.whitening = torch.nn.Identity()

    # ── Dataset ──
    print("Setting up dataset...")
    dataset = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        frameskip=5, num_steps=4,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        cache_dir=str(Path.home() / ".stable_worldmodel"),
    )
    transform = get_img_preprocessor(source='pixels', target='pixels', img_size=224)
    normalizer = get_column_normalizer(dataset, "action", "action")
    dataset.transform = spt.data.transforms.Compose(transform, normalizer)

    rnd = torch.Generator().manual_seed(42)
    train_set, val_set = spt.data.random_split(dataset, lengths=[0.9, 0.1], generator=rnd)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=2, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    # ── Lightning module ──
    sigreg = SIGReg(knots=17, num_proj=1024) if is_baseline else None
    noise = NoiseInjection(std=args.noise_std) if is_noise else None
    sigreg_weight = 0.09

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 1e-3},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    forward_fn = make_forward(sigreg_enabled=is_baseline, sigreg_weight=sigreg_weight, noise_enabled=is_noise)
    cfg = OmegaConf.create({"wm": {"history_size": 3, "num_preds": 1}})

    pl_module = spt.Module(
        model=jepa_model,
        sigreg=sigreg,
        noise_injection=noise,
        forward=partial(forward_fn, cfg=cfg),
        optim=optimizers,
    )

    # ── Trainer ──
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    run_dir = Path.home() / ".stable_worldmodel" / "experiments" / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    object_cb = ModelObjectCallBack(dirpath=run_dir, filename=exp_name, epoch_interval=1)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches // 2,
        accelerator="gpu",
        devices=1,
        precision="bf16",
        gradient_clip_val=1.0,
        callbacks=[object_cb],
        enable_checkpointing=False,
        logger=False,
    )

    manager = spt.Manager(trainer=trainer, module=pl_module, data=data_module)
    manager()

    # Save final model in AutoCostModel-compatible format
    suffix = "_frozen_enc" if args.freeze_encoder else ""
    final = run_dir / f"{exp_name}{suffix}_object.ckpt"
    torch.save(jepa_model, final)
    print(f"\nFinal model saved to {final}")
    print(f"  AutoCostModel load: swm.policy.AutoCostModel('{run_dir}')")


if __name__ == "__main__":
    main()
