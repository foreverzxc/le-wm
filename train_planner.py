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
    """Training forward for Planner."""
    ctx_batch = {k: v for k, v in batch.items() if not k.startswith("goal")}
    goal_batch = {k: v for k, v in batch.items() if k.startswith("goal")}

    # Encode context and goal (frozen WM, no grad — just starting state)
    with torch.no_grad():
        ctx_out = self.wm.encode(ctx_batch)
        ctx_emb = ctx_out["emb"]  # (B, T, D)

        goal_pixels = goal_batch.get("goal_pixels", ctx_batch["pixels"][:, -1:])
        if goal_pixels.ndim == 4:
            goal_pixels = goal_pixels.unsqueeze(1)
        goal_out = self.wm.encode({"pixels": goal_pixels})
        goal_emb = goal_out["emb"][:, -1:]  # (B, 1, D)

    # Planner produces N action candidates (has grad)
    history_ctx = ctx_emb[:, :cfg.wm.history_size]
    actions = self.model(history_ctx[:, -1:], goal_emb)  # (B, N, T, A)

    # Extract real historical actions from batch
    hs = cfg.wm.history_size
    raw_act = actions.shape[-1]
    fs = self.wm.action_encoder.patch_embed.in_channels // raw_act
    hist_actions = ctx_batch["action"][:, :hs]  # (B, HS, fs * raw_dim)
    hist_actions = hist_actions.reshape(B, hs, fs, raw_act).mean(2)  # (B, HS, raw_dim)

    # Rollout through frozen WM
    info = {"pixels": ctx_batch["pixels"][:, :hs]}
    pred_embs, _ = planner_rollout(
        self.wm, actions, info, history_size=hs,
        hist_actions=hist_actions, goal_emb=goal_emb,
    )

    # Compute loss
    total_loss, loss_info = self.planner_loss(actions, pred_embs, goal_emb)

    self.log_dict(
        {f"{stage}/{k}": v for k, v in loss_info.items()},
        on_step=True, sync_dist=True,
    )
    return {"loss": total_loss}


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
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

    # Optional: limit to first N episodes for overfitting tests
    max_ep = cfg.planner.get("max_episodes", 0)
    base_dataset = dataset  # keep reference for frameskip etc.
    if max_ep and max_ep > 0:
        indices = [i for i in range(len(dataset))
                   if dataset.clip_indices[i][0] < max_ep]
        dataset = torch.utils.data.Subset(dataset, indices)
        dataset.frameskip = base_dataset.frameskip  # propagate key attrs
        dataset.num_steps = base_dataset.num_steps

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    # Load frozen WM
    wm = load_frozen_wm(cfg.planner.ckpt)

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
        dropout=cfg.planner.dropout,
    )

    print(f"Planner: {cfg.planner.num_queries} queries × "
          f"{cfg.planner.horizon} steps × {action_dim}D action "
          f"(total output: {cfg.planner.num_queries}×{cfg.planner.horizon}×{action_dim})")

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
