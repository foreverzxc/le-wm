"""Overfit planner on 1 PushT episode, then simulate best action + show goal.

Usage:
    python scripts/sim_planner.py
"""
import json, sys, imageio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import hydra, numpy as np, torch
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")
from omegaconf import OmegaConf
import stable_worldmodel as swm
from planner import PlannerDecoder, PlannerLoss, planner_rollout
from utils import get_img_preprocessor


def load_wm():
    cache = Path(swm.data.utils.get_cache_dir())
    def remap(d):
        if isinstance(d, dict):
            if '_target_' in d:
                t = d['_target_']
                for o, n in [('stable_worldmodel.wm.lewm.LeWM','jepa.JEPA'),
                    ('stable_worldmodel.wm.lewm.module.Predictor','module.ARPredictor'),
                    ('stable_worldmodel.wm.lewm.module.Embedder','module.Embedder'),
                    ('stable_worldmodel.wm.lewm.module.MLP','module.MLP')]: t = t.replace(o, n)
                d['_target_'] = t
            for v in d.values(): remap(v)
        return d
    cfg = OmegaConf.create(remap(json.loads((cache/'pusht_config.json').read_text())))
    wm = hydra.utils.instantiate(cfg)
    sd = torch.load(cache/'pusht_weights.pt', map_location='cpu', weights_only=True)
    sd = {k.replace('_orig_mod.',''): v for k,v in sd.items()}
    wm.load_state_dict(sd, strict=False)
    wm = wm.cuda().eval()
    for p in wm.parameters(): p.requires_grad_(False)
    return wm


def main():
    wm = load_wm()

    # 1 episode, no transforms initially (need raw pixels for display)
    ds_raw = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                                   transform=None)
    indices = [i for i in range(len(ds_raw)) if ds_raw.clip_indices[i][0] == 0]
    ds_raw = torch.utils.data.Subset(ds_raw, indices)

    # With transform for WM input
    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                               transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices)

    N, T, ctx_len = 8, 5, 3
    planner = PlannerDecoder(embed_dim=192, num_queries=N, horizon=T,
                              action_dim=2, num_layers=3).cuda()
    loss_fn = PlannerLoss(diversity_weight=0.1)
    opt = torch.optim.AdamW(planner.parameters(), lr=1e-4)

    # ── Train on 1 episode ──
    n_epochs = 50
    print(f"Training {n_epochs} epochs on {len(ds)} samples...")
    for epoch in range(n_epochs):
        epoch_loss = 0
        for item_idx in range(len(ds)):
            item = ds[item_idx]
            batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
                     if torch.is_tensor(v)}

            with torch.no_grad():
                out = wm.encode(batch)
                ctx_emb = out["emb"][:, :ctx_len]
                goal_emb = out["emb"][:, -1:]

            actions = planner(ctx_emb[:, -1:], goal_emb)
            info = {"pixels": batch["pixels"][:, :ctx_len]}
            pred_embs, goal_emb_r = planner_rollout(wm, actions, info, history_size=ctx_len)
            loss, _ = loss_fn(actions, pred_embs, goal_emb_r)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}: loss={epoch_loss/len(ds):.6f}")

    # ── Run on training sample, get actions + goal ──
    sample_idx = len(ds) // 2  # middle of episode
    item = ds[sample_idx]
    raw_item = ds_raw[sample_idx]
    batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
             if torch.is_tensor(v)}

    with torch.no_grad():
        out = wm.encode(batch)
        ctx_emb = out["emb"][:, :ctx_len]
        goal_emb = out["emb"][:, -1:]

    actions = planner(ctx_emb[:, -1:], goal_emb).detach()
    info = {"pixels": batch["pixels"][:, :ctx_len]}
    with torch.no_grad():
        pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len)
        goal = goal_emb.expand(-1, N, -1)
        costs = (pred_embs - goal).pow(2).mean(dim=-1)[0]

    best_idx = costs.argmin().item()
    best_actions = actions[0, best_idx].cpu().numpy()
    best_cost = costs[best_idx].item()

    print(f"\nBest query: Q{best_idx}, cost={best_cost:.4f}")
    print(f"All costs: {[f'{c:.4f}' for c in costs.cpu().tolist()]}")

    # ── Simulation ──
    env = gym.make("swm/PushT-v1", max_episode_steps=200, render_mode="rgb_array")
    frames = []
    obs, _ = env.reset()
    frames.append(env.render())

    for t in range(T):
        action = best_actions[t].astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        if terminated or truncated:
            break

    env.close()

    # ── Figure: Goal vs Context ──
    ctx_pixels = raw_item["pixels"]  # (4, 3, 224, 224) — already torch, TCHW
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    for i in range(4):
        img = ctx_pixels[i].permute(1, 2, 0).numpy().astype(np.uint8)
        axes[i].imshow(img)
        axes[i].set_title(f'Context frame {i}' if i < 3 else 'Goal frame (t+15)')
        axes[i].axis('off')

    # Last panel: first + last sim frame overlaid
    axes[4].imshow(frames[0])
    axes[4].imshow(frames[-1], alpha=0.5)
    axes[4].set_title('Sim: start (opaque) + end (50%)')
    axes[4].axis('off')

    fig_path = Path(__file__).parent.parent / "output" / "planner_overfit_goal.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    print(f"Saved: {fig_path}")

    # ── Save sim video ──
    gif_path = Path(__file__).parent.parent / "output" / "planner_overfit_sim.gif"
    imageio.mimsave(gif_path, frames, fps=10, loop=0)
    print(f"Saved: {gif_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
