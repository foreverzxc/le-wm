"""Planner overfit on 1 PushT episode, T=1 horizon.

Shows context/goal from the SAME episode, planner vs GT action comparison,
and training loss curve. No env simulation (state-matching not available).

Usage:
    python scripts/planner_overfit_t1.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, torch
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")
from omegaconf import OmegaConf
import hydra, stable_worldmodel as swm
from planner import PlannerDecoder, PlannerLoss, planner_rollout
from utils import get_img_preprocessor

OUT = Path(__file__).parent.parent / "output" / "viz"
OUT.mkdir(parents=True, exist_ok=True)


def load_wm():
    cache = Path(swm.data.utils.get_cache_dir())
    def remap(d):
        if isinstance(d, dict):
            if "_target_" in d:
                t = d["_target_"]
                for o, n in [
                    ("stable_worldmodel.wm.lewm.LeWM", "jepa.JEPA"),
                    ("stable_worldmodel.wm.lewm.module.Predictor", "module.ARPredictor"),
                    ("stable_worldmodel.wm.lewm.module.Embedder", "module.Embedder"),
                    ("stable_worldmodel.wm.lewm.module.MLP", "module.MLP"),
                ]:
                    t = t.replace(o, n)
                d["_target_"] = t
            for v in d.values():
                remap(v)
        return d

    cfg = OmegaConf.create(remap(json.loads((cache / "pusht_config.json").read_text())))
    wm = hydra.utils.instantiate(cfg)
    sd = torch.load(cache / "pusht_weights.pt", map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    wm.load_state_dict(sd, strict=False)
    wm = wm.cuda().eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


def main():
    wm = load_wm()

    # Data: 1 episode
    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds_raw = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                                   transform=None)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                               transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices)
    ds_raw = torch.utils.data.Subset(ds_raw, indices)

    N, T, ctx_len, raw_act = 4, 1, 3, 2
    fs = wm.action_encoder.patch_embed.in_channels // raw_act
    planner = PlannerDecoder(embed_dim=192, num_queries=N, horizon=T,
                              action_dim=raw_act, num_layers=3).cuda()
    loss_fn = PlannerLoss(diversity_weight=0.0)
    opt = torch.optim.AdamW(planner.parameters(), lr=1e-3)

    # Train
    n_epochs = 20
    loss_history = []
    print(f"T={T}, training {n_epochs} epochs on {len(ds)} samples ...")
    for epoch in range(n_epochs):
        epoch_loss = 0
        for idx in range(len(ds)):
            item = ds[idx]
            batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
                     if torch.is_tensor(v)}
            B = 1
            with torch.no_grad():
                out = wm.encode(batch)
                ctx_emb = out["emb"][:, :ctx_len]
                goal_emb = out["emb"][:, -1:]

            actions = planner(ctx_emb[:, -1:], goal_emb)
            info = {"pixels": batch["pixels"][:, :ctx_len]}
            hist_act = batch["action"][:, :ctx_len].reshape(B, ctx_len, fs, raw_act).mean(2)
            pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len,
                                             hist_actions=hist_act, goal_emb=goal_emb)
            loss, _ = loss_fn(actions, pred_embs, goal_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(ds))
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}: loss={epoch_loss/len(ds):.6f}")

    # ── Evaluate on one sample ──
    sample_idx = len(ds) // 2
    item = ds[sample_idx]
    raw_item = ds_raw[sample_idx]
    batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
             if torch.is_tensor(v)}

    with torch.no_grad():
        out = wm.encode(batch)
        ctx_emb = out["emb"][:, :ctx_len]
        goal_emb = out["emb"][:, -1:]
        info = {"pixels": batch["pixels"][:, :ctx_len]}

    hist_actions = batch["action"][:, :ctx_len].reshape(1, ctx_len, fs, raw_act).mean(2)

    actions = planner(ctx_emb[:, -1:], goal_emb).detach()
    pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len,
                                     hist_actions=hist_actions, goal_emb=goal_emb)
    goal = goal_emb.expand(-1, N, -1)
    costs = (pred_embs - goal).pow(2).mean(dim=-1)[0]
    best_idx = costs.argmin().item()
    best_action = actions[0, best_idx].cpu().numpy()[0]

    # GT
    gt_raw = batch["action"][:, ctx_len:]
    gt_action = gt_raw.reshape(1, 1, fs, raw_act).mean(2)
    gt_actions_t = gt_action.unsqueeze(1).expand(-1, N, -1, -1).float().cuda()
    gt_embs, _ = planner_rollout(wm, gt_actions_t, info, history_size=ctx_len,
                                   hist_actions=hist_actions, goal_emb=goal_emb)
    gt_cost = (gt_embs - goal).pow(2).mean(dim=-1)[0, 0].item()
    natural_dist = (ctx_emb - goal_emb).pow(2).mean().item()

    print(f"\n{'='*50}")
    print(f"Planner best cost:  {costs[best_idx].item():.6f}")
    print(f"GT action cost:     {gt_cost:.6f}")
    print(f"Natural ctx→goal:   {natural_dist:.6f}")
    print(f"Planner action:     {best_action}")
    print(f"GT action:          {gt_action[0, 0].cpu().numpy()}")

    # ── Visualization ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: context frames from the dataset episode
    for i in range(3):
        img = raw_item["pixels"][i].permute(1, 2, 0).numpy().astype(np.uint8)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Context frame {i}")
        axes[0, i].axis("off")

    # Goal image from the SAME episode
    axes[1, 0].imshow(raw_item["pixels"][3].permute(1, 2, 0).numpy().astype(np.uint8))
    gt_np = gt_action[0, 0].cpu().numpy()
    axes[1, 0].set_title(f"Goal (frame 3 — same episode)\n"
                         f"Planner: {np.round(best_action, 3)}\n"
                         f"GT:      {np.round(gt_np, 3)}")
    axes[1, 0].axis("off")

    # Action comparison
    ax = axes[1, 1]
    ax.bar([0, 1], best_action, 0.35, color='#58a6ff', label='Planner', alpha=0.85)
    ax.bar([0.35, 1.35], gt_np, 0.35, color='gray', label='GT', alpha=0.85)
    ax.set_xticks([0.175, 1.175])
    ax.set_xticklabels(['Action X', 'Action Y'])
    ax.set_ylabel('Action value')
    ax.set_title(f'Action: planner (blue) vs GT (gray)\n'
                 f'Planner cost={costs[best_idx].item():.4f}  '
                 f'GT cost={gt_cost:.4f}')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)

    # Loss curve
    axes[1, 2].plot(loss_history, color="#3fb950", linewidth=1.5)
    axes[1, 2].axhline(y=gt_cost, color="gray", linestyle="--",
                       label=f"GT cost={gt_cost:.4f}")
    axes[1, 2].axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_title(f"Training loss\nfinal={loss_history[-1]:.6f}")
    axes[1, 2].legend(fontsize=8)

    plt.suptitle(f"Planner T=1 Overfit (1 PushT episode, {n_epochs} epochs)\n"
                 f"cost: planner={costs[best_idx].item():.4f}  gt={gt_cost:.4f}  "
                 f"natural={natural_dist:.4f}  |  ",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT / "planner_t1_overfit.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"All viz: {OUT}/")


if __name__ == "__main__":
    main()
