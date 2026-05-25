"""Planner T=5 overfit: goal is 5 rollout steps ahead (25 env steps, 5 frames).

Compares planner action series vs GT action series, both rolled out through WM
to the same goal embedding (frame 8).

Usage:
    python scripts/planner_overfit_t5.py
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


def get_state_from_dataset(ds, step=0):
    base = ds.dataset if hasattr(ds, 'dataset') else ds
    row = base.get_row_data(step)
    state = row["state"]
    if state.ndim > 1:
        state = state[0]
    return state


def set_env_state(env, state):
    if state.ndim > 1:
        state = state[0]
    env.unwrapped._set_state(state)
    return True


def main():
    wm = load_wm()

    # num_steps=8: 3 ctx + 5 future frames for T=5 rollout
    ctx_len, T, raw_act = 3, 5, 2
    fs = wm.action_encoder.patch_embed.in_channels // raw_act  # 5
    total_frames = ctx_len + T  # 8

    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds_raw = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                                   num_steps=total_frames, transform=None)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                               num_steps=total_frames, transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices)
    ds_raw = torch.utils.data.Subset(ds_raw, indices)

    N = 8
    planner = PlannerDecoder(embed_dim=192, num_queries=N, horizon=T,
                              action_dim=raw_act, num_layers=3).cuda()
    loss_fn = PlannerLoss(diversity_weight=0.1)
    opt = torch.optim.AdamW(planner.parameters(), lr=1e-3)

    n_epochs = 50
    loss_history = []
    print(f"T={T}, goal at frame {total_frames-1} (25 env steps ahead)")
    print(f"Training {n_epochs} epochs on {len(ds)} samples ...")
    for epoch in range(n_epochs):
        epoch_loss = 0
        for idx in range(len(ds)):
            item = ds[idx]
            batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
                     if torch.is_tensor(v)}
            B = 1
            with torch.no_grad():
                out = wm.encode(batch)
                ctx_emb = out["emb"][:, :ctx_len]      # (1, 3, D)
                goal_emb = out["emb"][:, -1:]           # (1, 1, D) — frame 7

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
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}: loss={epoch_loss/len(ds):.6f}")

    # ── Evaluate ──
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

    # Planner
    actions = planner(ctx_emb[:, -1:], goal_emb).detach()
    pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len,
                                     hist_actions=hist_actions, goal_emb=goal_emb)
    goal = goal_emb.expand(-1, N, -1)
    costs = (pred_embs - goal).pow(2).mean(dim=-1)[0]
    best_idx = costs.argmin().item()
    best_actions = actions[0, best_idx].cpu().numpy()

    # GT actions for T=5 steps
    gt_raw = batch["action"][:, ctx_len:ctx_len+T]  # (1, 5, 10) — 5 future steps
    gt_actions_avg = gt_raw.reshape(1, T, fs, raw_act).mean(2)  # (1, 5, 2)
    gt_actions_t = gt_actions_avg.unsqueeze(1).expand(-1, N, -1, -1).float().cuda()
    gt_embs, _ = planner_rollout(wm, gt_actions_t, info, history_size=ctx_len,
                                   hist_actions=hist_actions, goal_emb=goal_emb)
    gt_cost = (gt_embs - goal).pow(2).mean(dim=-1)[0, 0].item()

    natural_dist = (ctx_emb - goal_emb).pow(2).mean().item()

    print(f"\n{'='*50}")
    print(f"Planner best cost:  {costs[best_idx].item():.6f}")
    print(f"GT action cost:     {gt_cost:.6f}")
    print(f"Natural ctx→goal:   {natural_dist:.6f}")
    print(f"All query costs:    {[f'{c:.4f}' for c in costs.cpu().tolist()]}")
    print(f"Best action series (5 steps):")
    for i, a in enumerate(best_actions):
        print(f"  step {i}: {a}")

    # ── Simulation from dataset state ──
    base_ds = ds_raw.dataset
    orig_idx = ds_raw.indices[sample_idx]
    ctx_start_frame = base_ds.clip_indices[orig_idx][1]
    init_state = get_state_from_dataset(ds_raw, ctx_start_frame)

    env = gym.make("swm/PushT-v1", max_episode_steps=200, render_mode="rgb_array")
    env.reset()
    set_env_state(env, init_state)

    frames = []
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    frames.append(env.render())

    # Planner actions: each repeated frameskip=5 times
    for t in range(T):
        a = best_actions[t].astype(np.float32)
        a = np.clip(a, -1, 1)
        for _ in range(fs):
            obs, _, terminated, truncated, _ = env.step(a)
            frames.append(env.render())
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    env.close()

    # Also run GT simulation (all individual actions)
    env = gym.make("swm/PushT-v1", max_episode_steps=200, render_mode="rgb_array")
    env.reset()
    set_env_state(env, init_state)
    gt_frames = []
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    gt_frames.append(env.render())
    gt_actions_full = gt_raw.reshape(1, T, fs, raw_act)[0].cpu().numpy()  # (T, 5, 2)
    for a in gt_actions_full.reshape(-1, raw_act):
        a = np.clip(a.astype(np.float32), -1, 1)
        obs, _, terminated, truncated, _ = env.step(a)
        gt_frames.append(env.render())
        if terminated or truncated:
            break
    env.close()

    # ── Visualization ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Context frames
    for i in range(3):
        img = raw_item["pixels"][i].permute(1, 2, 0).numpy().astype(np.uint8)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Context {i} (t={i*5})")
        axes[0, i].axis("off")

    # Goal frame (frame 7, 25 env steps ahead)
    axes[1, 0].imshow(raw_item["pixels"][-1].permute(1, 2, 0).numpy().astype(np.uint8))
    axes[1, 0].set_title(f"Goal (t=40, 25 env steps ahead)")
    axes[1, 0].axis("off")

    # Action trajectories: planner vs GT (cumulative)
    ax = axes[1, 1]
    planner_cum = np.cumsum(best_actions, axis=0)
    gt_cum = np.cumsum(gt_actions_avg[0].cpu().numpy(), axis=0)
    # All planner queries
    all_acts = actions[0].cpu().numpy()
    cmap = plt.cm.tab10
    for i in range(N):
        cum = np.cumsum(all_acts[i], axis=0)
        alpha = 0.9 if i == best_idx else 0.2
        lw = 2.5 if i == best_idx else 0.8
        ax.plot(cum[:, 0], cum[:, 1], 'o-', color=cmap(i % 10),
                linewidth=lw, alpha=alpha, markersize=5,
                label=f'Q{i}★ cost={costs[i]:.3f}' if i == best_idx else None)
    ax.plot(gt_cum[:, 0], gt_cum[:, 1], 's--', color='black',
            linewidth=2.5, markersize=7, label=f'GT cost={gt_cost:.3f}')
    ax.set_xlabel("Cumulative X")
    ax.set_ylabel("Cumulative Y")
    ax.set_title(f"Action space (T=5)\nPlanner best={costs[best_idx]:.4f}  GT={gt_cost:.4f}")
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    ax.legend(fontsize=6, loc='best')

    # Loss curve
    axes[1, 2].plot(loss_history, color="#3fb950", linewidth=1.5)
    axes[1, 2].axhline(y=gt_cost, color="gray", linestyle="--",
                       label=f"GT cost={gt_cost:.4f}")
    axes[1, 2].axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_title(f"Training loss\nfinal={loss_history[-1]:.6f}")
    axes[1, 2].legend(fontsize=8)

    plt.suptitle(f"Planner T={T} Overfit (goal = {25} env steps / {5} frames ahead)\n"
                 f"best_cost={costs[best_idx].item():.4f}  "
                 f"gt_cost={gt_cost:.4f}  natural={natural_dist:.4f}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT / "planner_t5_overfit.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")

    # GIFs
    gif_out = OUT / "planner_t5_sim.gif"
    imageio.mimsave(gif_out, frames, fps=6, loop=0)
    print(f"Saved: {gif_out}  ({len(frames)} frames)")
    gt_gif_out = OUT / "planner_t5_gt_sim.gif"
    imageio.mimsave(gt_gif_out, gt_frames, fps=6, loop=0)
    print(f"Saved: {gt_gif_out}  ({len(gt_frames)} frames)")

    print(f"\nAll viz: {OUT}/")


if __name__ == "__main__":
    main()
