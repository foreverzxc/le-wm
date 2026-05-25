"""Simulate planner action in PushT environment from dataset starting state.

Usage:
    python scripts/sim_planner_action.py
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


def get_state_from_dataset(ds, ep_idx=0, step=0):
    """Get PushT state (7,) from dataset for a given episode and step index."""
    base = ds.dataset if hasattr(ds, 'dataset') else ds
    row = base.get_row_data(step)
    state = row["state"]  # (7,) float32
    if state.ndim > 1:
        state = state[0]
    return state


def set_env_state(env, state):
    """Try to set PushT env state. Handles version differences."""
    uw = env.unwrapped
    # Try different API versions
    if hasattr(uw, '_set_state'):
        try:
            uw._set_state(state)
            return True
        except Exception:
            pass
    # Fallback: set via physics directly
    if hasattr(uw, 'space') and hasattr(uw, 'block'):
        try:
            pos_agent = state[:2].tolist()
            pos_block = state[2:4].tolist()
            rot_block = float(state[4])
            uw.block.position = pos_block
            uw.block.angle = rot_block
            uw.space.step(uw.dt)
            return True
        except Exception:
            pass
    return False


def main():
    wm = load_wm()

    # Data: 1 episode, raw (no transform) for state + display
    ds_raw = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                                   transform=None)
    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                               transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices)
    ds_raw = torch.utils.data.Subset(ds_raw, indices)

    N, T, ctx_len, raw_act = 4, 1, 3, 2
    fs = wm.action_encoder.patch_embed.in_channels // raw_act

    # Train planner
    planner = PlannerDecoder(embed_dim=192, num_queries=N, horizon=T,
                              action_dim=raw_act, num_layers=3).cuda()
    loss_fn = PlannerLoss(diversity_weight=0.0)
    opt = torch.optim.AdamW(planner.parameters(), lr=1e-3)

    n_epochs = 30
    print(f"Training {n_epochs} epochs ...")
    for epoch in range(n_epochs):
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
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}: loss={loss.item():.6f}")

    # ── Get best action on a sample ──
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

    print(f"Best action: {best_action}  (cost={costs[best_idx].item():.4f})")

    # ── Simulation from dataset starting state ──
    # Get the state at the context start
    base_ds = ds_raw.dataset
    orig_idx = ds_raw.indices[sample_idx]
    ctx_start_frame = base_ds.clip_indices[orig_idx][1]  # first frame of context
    init_state = get_state_from_dataset(ds_raw, 0, ctx_start_frame)

    env = gym.make("swm/PushT-v1", max_episode_steps=200, render_mode="rgb_array")
    obs, _ = env.reset()
    state_ok = set_env_state(env, init_state)
    print(f"State set: {state_ok}")

    frames = []
    # Step once to render correct state after physics step
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    frames.append(env.render())

    a = best_action.astype(np.float32)
    a = np.clip(a, -1, 1)
    for _ in range(5):  # frameskip = 5
        obs, _, terminated, truncated, _ = env.step(a)
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()

    # ── Save GIF ──
    gif_out = OUT / "planner_t1_sim.gif"
    imageio.mimsave(gif_out, frames, fps=5, loop=0)
    print(f"Saved: {gif_out}  ({len(frames)} frames)")

    # ── Side-by-side: context + sim frames ──
    ctx_frame = raw_item["pixels"][2].permute(1, 2, 0).numpy().astype(np.uint8)
    goal_frame = raw_item["pixels"][3].permute(1, 2, 0).numpy().astype(np.uint8)
    sim_start = frames[0]
    sim_end = frames[-1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(ctx_frame)
    axes[0].set_title("Context (last frame)")
    axes[0].axis("off")
    axes[1].imshow(goal_frame)
    axes[1].set_title("Goal (dataset)")
    axes[1].axis("off")
    axes[2].imshow(sim_start)
    axes[2].set_title(f"Sim start\n(state from dataset)")
    axes[2].axis("off")
    axes[3].imshow(sim_end)
    axes[3].set_title(f"Sim after action\n{np.round(best_action, 3)}")
    axes[3].axis("off")

    cmp_out = OUT / "planner_t1_sim_comparison.png"
    plt.tight_layout()
    plt.savefig(cmp_out, dpi=120)
    print(f"Saved: {cmp_out}")
    print(f"All viz: {OUT}/")


if __name__ == "__main__":
    main()
