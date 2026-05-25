"""Compare planner vs GT actions in PushT env — two GIFs side by side.

Usage:
    python scripts/compare_actions.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, torch, imageio, gymnasium as gym
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


def get_state(ds, step):
    base = ds.dataset if hasattr(ds, 'dataset') else ds
    s = base.get_row_data(step)["state"]
    return s[0] if s.ndim > 1 else s


def run_sim(env, init_state, action_series, frameskip=5, is_gt=False):
    """Run action series in env from init_state, return frames.

    Args:
        action_series: if is_gt, shape (T, frameskip, raw_dim) — all individual actions.
                       if not, shape (T, raw_dim) — each repeated frameskip times.
    """
    env.reset()
    env.unwrapped._set_state(init_state)
    frames = []
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    frames.append(env.render())
    if is_gt:
        # GT: all individual actions, apply one by one
        for a in action_series.reshape(-1, action_series.shape[-1]):
            a = np.clip(a.astype(np.float32), -1, 1)
            obs, _, terminated, truncated, _ = env.step(a)
            frames.append(env.render())
            if terminated or truncated:
                return frames
    else:
        # Planner: repeat each raw action frameskip times
        for a in action_series:
            a = np.clip(a.astype(np.float32), -1, 1)
            for _ in range(frameskip):
                obs, _, terminated, truncated, _ = env.step(a)
                frames.append(env.render())
                if terminated or truncated:
                    return frames
    return frames


def main():
    wm = load_wm()

    ctx_len, T, raw_act = 3, 5, 2
    fs = wm.action_encoder.patch_embed.in_channels // raw_act
    total_frames = ctx_len + T

    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds_raw = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                                   num_steps=total_frames, transform=None)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                               num_steps=total_frames, transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices)
    ds_raw = torch.utils.data.Subset(ds_raw, indices)

    # Train planner
    N = 8
    planner = PlannerDecoder(embed_dim=192, num_queries=N, horizon=T,
                              action_dim=raw_act, num_layers=3).cuda()
    loss_fn = PlannerLoss(diversity_weight=0.1)
    opt = torch.optim.AdamW(planner.parameters(), lr=1e-3)

    for epoch in range(50):
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

    # Pick a sample
    sample_idx = len(ds) // 2
    item = ds[sample_idx]
    batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
             if torch.is_tensor(v)}
    with torch.no_grad():
        out = wm.encode(batch)
        ctx_emb = out["emb"][:, :ctx_len]
        goal_emb = out["emb"][:, -1:]
        info = {"pixels": batch["pixels"][:, :ctx_len]}
    hist_actions = batch["action"][:, :ctx_len].reshape(1, ctx_len, fs, raw_act).mean(2)

    # Best planner action
    actions = planner(ctx_emb[:, -1:], goal_emb).detach()
    pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len,
                                     hist_actions=hist_actions, goal_emb=goal_emb)
    goal = goal_emb.expand(-1, N, -1)
    costs = (pred_embs - goal).pow(2).mean(dim=-1)[0]
    best_idx = costs.argmin().item()
    planner_acts = actions[0, best_idx].cpu().numpy()  # (T, 2)

    # GT actions: use ALL individual actions (T, frameskip, raw_dim) = (5, 5, 2)
    gt_raw = batch["action"][:, ctx_len:ctx_len+T]  # (1, T, 10)
    gt_acts_full = gt_raw.reshape(1, T, fs, raw_act)[0].cpu().numpy()  # (T, fs, 2)
    # Flatten for display
    gt_acts_avg = gt_acts_full.mean(1)  # (T, 2) — for printing only

    # Get starting state
    base_ds = ds_raw.dataset
    orig_idx = ds_raw.indices[sample_idx]
    start_frame = base_ds.clip_indices[orig_idx][1]
    init_state = get_state(ds_raw, start_frame)

    print(f"Planner actions (mean per step): {np.round(planner_acts, 3).tolist()}")
    print(f"GT actions (mean per step):      {np.round(gt_acts_avg, 3).tolist()}")

    # Run both simulations
    env = gym.make("swm/PushT-v1", max_episode_steps=200, render_mode="rgb_array")
    planner_frames = run_sim(env, init_state, planner_acts, is_gt=False)
    env.close()
    env = gym.make("swm/PushT-v1", max_episode_steps=200, render_mode="rgb_array")
    gt_frames = run_sim(env, init_state, gt_acts_full, is_gt=True)
    env.close()

    # Match frame counts
    n = min(len(planner_frames), len(gt_frames))
    planner_frames = planner_frames[:n]
    gt_frames = gt_frames[:n]

    # Save individual GIFs
    imageio.mimsave(OUT / "compare_planner.gif", planner_frames, fps=6, loop=0)
    imageio.mimsave(OUT / "compare_gt.gif", gt_frames, fps=6, loop=0)

    # Side-by-side GIF
    combined = [np.hstack([p, g]) for p, g in zip(planner_frames, gt_frames)]
    imageio.mimsave(OUT / "compare_side_by_side.gif", combined, fps=6, loop=0)

    # Label frame
    h, w = planner_frames[0].shape[:2]
    label_h = 30
    for i in range(len(planner_frames)):
        labeled_p = np.pad(planner_frames[i], ((label_h, 0), (0, 0), (0, 0)),
                           constant_values=30)
        labeled_g = np.pad(gt_frames[i], ((label_h, 0), (0, 0), (0, 0)),
                           constant_values=30)
        combined[i] = np.hstack([labeled_p, labeled_g])

    imageio.mimsave(OUT / "compare_labeled.gif", combined, fps=6, loop=0)

    print(f"\nSaved:")
    print(f"  {OUT}/compare_planner.gif  ({len(planner_frames)} frames)")
    print(f"  {OUT}/compare_gt.gif       ({len(gt_frames)} frames)")
    print(f"  {OUT}/compare_side_by_side.gif")
    print(f"  {OUT}/compare_labeled.gif")


if __name__ == "__main__":
    main()
