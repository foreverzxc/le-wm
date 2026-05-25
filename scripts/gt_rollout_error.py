"""Compare WM training pred_loss vs planner_rollout cost.

Key question: does planner_rollout with GT actions produce the same cost
as the WM's own prediction? If not, where does the gap come from?

Tests:
  1. WM pred_loss (lejepa_forward) — average over 3 positions
  2. WM pred_loss for LAST position only (the goal prediction)
  3. planner_rollout T=1 with GT action + real history + correct goal
  4. planner_rollout T=1 with GT action + zero history + correct goal

Usage:
    python scripts/gt_rollout_error.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, torch
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")
from omegaconf import OmegaConf
import hydra, stable_worldmodel as swm
from planner import planner_rollout
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

    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                               transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices[:30])
    ctx_len = 3
    raw_act = 2
    fs = wm.action_encoder.patch_embed.in_channels // raw_act

    # Collect per-position WM pred_loss + rollout costs
    wm_pos_mse = {0: [], 1: [], 2: []}  # per-position WM error
    rollout_gt_costs = []
    rollout_gt_zero_hist = []

    for sample_idx in range(8):
        item = ds[sample_idx]
        batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
                 if torch.is_tensor(v)}
        B = 1

        with torch.no_grad():
            out = wm.encode(batch)
            emb = out["emb"]          # (B, 4, D)
            act_emb = out["act_emb"]  # (B, 4, D)

        # 1. WM pred_loss per position
        ctx_emb = emb[:, :ctx_len]      # (B, 3, D)
        ctx_act = act_emb[:, :ctx_len]  # (B, 3, D)
        pred_emb = wm.predict(ctx_emb, ctx_act)  # (B, 3, D)
        tgt_emb = emb[:, 1:ctx_len+1]            # (B, 3, D)

        for pos in range(3):
            mse = (pred_emb[:, pos] - tgt_emb[:, pos]).pow(2).mean().item()
            wm_pos_mse[pos].append(mse)

        # 2. planner_rollout with GT action
        goal_emb = emb[:, -1:]  # (B, 1, D) — frame 3
        info = {"pixels": batch["pixels"][:, :ctx_len]}

        # Real history actions: frames 0,1,2 → raw actions
        hist_raw = batch["action"][:, :ctx_len]  # (B, 3, 10)
        hist_actions = hist_raw.reshape(B, ctx_len, fs, raw_act).mean(2)  # (B, 3, 2)

        # GT action for step 3
        gt_raw = batch["action"][:, ctx_len:]  # (B, 1, 10)
        gt_action = gt_raw.reshape(B, 1, fs, raw_act).mean(2)  # (B, 1, 2)
        gt_t = gt_action.unsqueeze(1).float().cuda()  # (B, 1, 1, 2)

        # With real history
        gt_embs, _ = planner_rollout(wm, gt_t, info, history_size=ctx_len,
                                       hist_actions=hist_actions, goal_emb=goal_emb)
        rollout_gt_costs.append((gt_embs - goal_emb).pow(2).mean().item())

        # With zero history
        gt_embs_z, _ = planner_rollout(wm, gt_t, info, history_size=ctx_len,
                                         hist_actions=None, goal_emb=goal_emb)
        rollout_gt_zero_hist.append((gt_embs_z - goal_emb).pow(2).mean().item())

    # ── Summary ──
    wm_all = np.mean([np.mean(wm_pos_mse[p]) for p in range(3)])
    wm_last = np.mean(wm_pos_mse[2])
    gt_cost = np.mean(rollout_gt_costs)
    gt_zero = np.mean(rollout_gt_zero_hist)

    print(f"\n{'='*60}")
    print(f"WM pred_loss (avg 3 pos):  {wm_all:.6f}")
    print(f"WM pred_loss (last only):   {wm_last:.6f}  ← should match rollout")
    print(f"Rollout GT + real hist:     {gt_cost:.6f}")
    print(f"Rollout GT + zero hist:     {gt_zero:.6f}")
    print(f"Gap (rollout / wm_last):    {gt_cost/wm_last:.2f}x")

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["WM pred_loss\n(avg 3 pos)", "WM pred_loss\n(last = goal)",
             "Rollout\n(GT + real hist)", "Rollout\n(GT + zero hist)"]
    means = [wm_all, wm_last, gt_cost, gt_zero]
    colors = ["#3fb950", "#58a6ff", "#f85149", "#d2991d"]
    ax.bar(names, means, color=colors, alpha=0.85)
    for i, m in enumerate(means):
        ax.text(i, m + max(means)*0.02, f"{m:.6f}", ha="center",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Where does the Rollout cost come from?\n"
                 f"Rollout / WM last = {gt_cost/wm_last:.2f}x",
                 fontsize=13)
    ax.grid(axis="y", alpha=0.2)
    out = OUT / "rollout_vs_wm_cost.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")
    print(f"All viz in: {OUT}/")


if __name__ == "__main__":
    main()
