"""Train Planner on top of frozen WM.

Usage:
    python train_planner.py data=pusht planner.ckpt=pusht
"""
import json
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

from planner import PlannerDecoder, PlannerLoss, planner_rollout
from utils import get_img_preprocessor


def load_frozen_wm(ckpt_name: str):
    """Load a pretrained WM checkpoint (frozen, eval mode)."""
    cache_dir = Path(swm.data.utils.get_cache_dir())

    # Try our checkpoint format first, then pretrained format
    ckpt_dir = cache_dir / "checkpoints"
    weight_files = sorted(ckpt_dir.glob(f"{ckpt_name}_weights_epoch_*.pt"))
    if weight_files:
        weight_path = weight_files[-1]
        cfg_path = weight_path.parent / weight_path.name.replace(
            "_weights_epoch_", "_config_epoch_"
        ).replace(".pt", ".json")
    else:
        weight_path = cache_dir / f"{ckpt_name}_weights.pt"
        cfg_path = cache_dir / f"{ckpt_name}_config.json"

    cfg = OmegaConf.create(json.loads(cfg_path.read_text()))

    # Remap targets
    def remap(d):
        if isinstance(d, dict):
            if "_target_" in d:
                t = d["_target_"]
                t = t.replace("stable_worldmodel.wm.lewm.LeWM", "jepa.JEPA")
                t = t.replace("stable_worldmodel.wm.lewm.module.Predictor", "module.ARPredictor")
                t = t.replace("stable_worldmodel.wm.lewm.module.Embedder", "module.Embedder")
                t = t.replace("stable_worldmodel.wm.lewm.module.MLP", "module.MLP")
                d["_target_"] = t
            for v in d.values():
                remap(v)
        return d

    cfg = remap(OmegaConf.to_container(cfg, resolve=True))
    cfg = OmegaConf.create(cfg)

    wm = hydra.utils.instantiate(cfg)
    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    wm.load_state_dict(sd, strict=False)
    wm = wm.cuda().eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


def get_goal_pair(dataset, idx, goal_offset=25):
    """Return (ctx_item, goal_item, ctx_frame, goal_frame)."""
    ctx_item = dataset[idx]
    goal_idx = min(idx + goal_offset // dataset.frameskip, len(dataset) - 1)
    goal_item = dataset[goal_idx]
    return ctx_item, goal_item


def planner_forward(self, batch, stage, cfg):
    """Training forward for Planner. Stage: "sft" (action imitation) or "ft" (WM rollout)."""
    ctx_batch = {k: v for k, v in batch.items() if not k.startswith("goal")}
    hs = cfg.wm.history_size
    train_stage = cfg.planner.get("stage", "ft")

    # Encode context (frozen WM, no grad)
    with torch.no_grad():
        ctx_out = self.wm.encode(ctx_batch)
        ctx_emb = ctx_out["emb"]  # (B, num_steps, D)

    # Planner: full history ctx + final-frame goal
    history_ctx = ctx_emb[:, :hs]
    goal_emb = ctx_emb[:, -1:]
    actions, conf = self.model(history_ctx, goal_emb)  # (B, N, T, A), (B, N, 1)

    if train_stage == "sft":
        # ── SFT: best-of-N action imitation + DETR conf + diversity ─
        gt_future = ctx_batch["action"][:, hs : hs + cfg.planner.horizon]
        gt_expanded = gt_future.unsqueeze(1).expand(-1, actions.shape[1], -1, -1)

        # Per-query cost: MSE between planner actions and GT
        costs = (actions - gt_expanded).pow(2).mean(dim=(-2, -1))  # (B, N)
        best_cost, best_idx = costs.min(dim=-1)

        # Diversity loss
        B = actions.size(0); N = actions.shape[1]
        best_acts = actions[torch.arange(B), best_idx]
        div_loss = torch.tensor(0.0, device=actions.device); count = 0
        for i in range(N):
            mask = (best_idx != i)
            if mask.any():
                other = actions[mask, i].reshape(mask.sum(), -1)
                best_for = best_acts[mask].reshape(mask.sum(), -1)
                sim = torch.nn.functional.cosine_similarity(other, best_for, dim=-1)
                div_loss += torch.nn.functional.relu(sim - 0.0).mean(); count += 1
        if count > 0: div_loss = div_loss / count

        # DETR confidence loss: matched→1, unmatched→0 (logits, safe for bf16)
        conf_target = torch.zeros_like(conf)
        conf_target[torch.arange(B), best_idx] = 1.0
        conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            conf.squeeze(-1), conf_target.squeeze(-1))

        conf_weight = cfg.planner.get("conf_weight", 0.5)
        total_loss = best_cost.mean() \
            + cfg.planner.diversity_weight * div_loss \
            + conf_weight * conf_loss
        loss_info = {
            "loss": total_loss.detach(),
            "best_cost": best_cost.mean().detach(),
            "cost_mean": costs.mean().detach(),
            "diversity": div_loss.detach(),
            "conf_loss": conf_loss.detach(),
        }

    else:
        # ── FT: WM rollout + best-of-N + diversity + conf ────────────
        hist_actions = ctx_batch["action"][:, :hs]
        info = {"pixels": ctx_batch["pixels"][:, :hs]}
        pred_embs, _ = planner_rollout(
            self.wm, actions, info, history_size=hs,
            hist_actions=hist_actions, goal_emb=goal_emb,
        )
        total_loss, loss_info = self.planner_loss(
            actions, pred_embs, goal_emb, conf=conf)

    self.log_dict(
        {f"{stage}/{k}": v for k, v in loss_info.items()},
        on_step=True, sync_dist=True,
    )
    return {"loss": total_loss}


@hydra.main(version_base=None, config_path="./config/train", config_name="planner_train")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    if cfg.data.get("dataset_class"):
        dataset = hydra.utils.instantiate(cfg.data.dataset)
    else:
        dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
        dataset_name = dataset_cfg.pop("name")
        dataset_name = os.path.splitext(dataset_name)[0]
        cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
        dataset = swm.data.HDF5Dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )

    img_proc = get_img_preprocessor("pixels", "pixels", cfg.img_size)
    dataset.transform = img_proc

    # Optional: limit to first N episodes
    max_ep = cfg.planner.get("max_episodes", 0)
    if max_ep and max_ep > 0:
        keep = [i for i in range(len(dataset))
                if dataset.clip_indices[i][0] < max_ep]
        dataset = torch.utils.data.Subset(dataset, keep)
        dataset.frameskip = dataset.dataset.frameskip
        dataset.num_steps = dataset.dataset.num_steps

    # Optional: limit to first N samples (for single-sample overfitting)
    max_samples = cfg.planner.get("max_samples", 0)
    if max_samples and max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
        dataset.frameskip = dataset.dataset.frameskip
        dataset.num_steps = dataset.dataset.num_steps

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_split = cfg.get("train_split", 0.9)
    if train_split >= 1.0:
        # Deterministic: use all data for training, dummy val to avoid scheduler error
        train_set = dataset
        val_set = torch.utils.data.Subset(dataset, [0])  # first sample as dummy val
        shuffle = False
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[train_split, 1 - train_split], generator=rnd_gen)
        shuffle = True

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=shuffle, drop_last=shuffle,
        generator=rnd_gen if shuffle else None,
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    # Load frozen WM
    wm = load_frozen_wm(cfg.planner.ckpt)

    # Ensure planner horizon does not exceed available future frames
    future_frames = dataset.num_steps - cfg.wm.history_size
    if cfg.planner.horizon > future_frames:
        raise ValueError(
            f"planner.horizon ({cfg.planner.horizon}) exceeds available future "
            f"frames ({future_frames}). Set data.dataset.num_steps to at least "
            f"{cfg.wm.history_size + cfg.planner.horizon}, or reduce horizon."
        )

    # Auto-detect action_dim from WM if not set in config
    action_dim = cfg.planner.action_dim
    if action_dim is None:
        # WM action_encoder input_dim = frameskip * raw_action_dim
        action_dim = wm.action_encoder.patch_embed.in_channels // dataset.frameskip

    # Planner decoder
    planner = PlannerDecoder(
        embed_dim=cfg.wm.embed_dim,
        num_queries=cfg.planner.num_queries,
        num_layers=cfg.planner.num_layers,
        num_heads=cfg.planner.num_heads,
        mlp_dim=cfg.planner.mlp_dim,
        horizon=cfg.planner.horizon,
        action_dim=action_dim,
        action_substeps=dataset.frameskip,
        action_range=cfg.planner.get("action_range", 1.0),
        dropout=cfg.planner.dropout,
    )

    print(f"Planner: {cfg.planner.num_queries} queries × "
          f"{cfg.planner.horizon} coarse steps × {dataset.frameskip} substeps × "
          f"{action_dim}D = {cfg.planner.num_queries}×{cfg.planner.horizon}×"
          f"{dataset.frameskip * action_dim} scalars")

    planner_loss = PlannerLoss(diversity_weight=cfg.planner.diversity_weight)

    optimizers = {
        "planner_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    planner_module = spt.Module(
        model=planner,
        wm=wm,
        planner_loss=planner_loss,
        forward=partial(planner_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), "checkpoints", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "planner_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        num_sanity_val_steps=1,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"planner_{cfg.planner.ckpt}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=planner_module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
