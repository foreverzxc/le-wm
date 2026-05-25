"""Train planner on 1 PushT episode (T=5), then compare GT vs planner vs dataset.

Generates:
  - output/viz/compare_3way.gif  (dataset | GT sim | planner sim, side-by-side)
  - output/viz/compare_3way.png  (goal + action space + loss)

Usage:
    python scripts/train_and_compare.py
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


def run_sim(env, init_state, actions, is_gt=False):
    """Run actions in env, return frames. is_gt: actions shape (T, fs, 2)."""
    env.reset()
    env.unwrapped._set_state(init_state)
    frames = []
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    # Don't include blank step — go straight to first action
    if is_gt:
        for a in actions.reshape(-1, actions.shape[-1]):
            a = np.clip(a.astype(np.float32), -1, 1)
            obs, _, terminated, truncated, _ = env.step(a)
            frames.append(env.render())
            if terminated or truncated:
                break
    else:
        for a in actions:
            a = np.clip(a.astype(np.float32), -1, 1)
            for _ in range(5):
                obs, _, terminated, truncated, _ = env.step(a)
                frames.append(env.render())
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
    return frames


def main():
    wm = load_wm()

    ctx_len, T, raw_act, fs = 3, 5, 2, wm.action_encoder.patch_embed.in_channels // 2
    total_frames = ctx_len + T

    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds_raw = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                                   num_steps=total_frames, transform=None)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                               num_steps=total_frames, transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices)
    ds_raw = torch.utils.data.Subset(ds_raw, indices)

    # Planner with action smoothness penalty
    N = 8
    planner = PlannerDecoder(embed_dim=192, num_queries=N, horizon=T,
                              action_dim=raw_act, num_layers=3).cuda()
    loss_fn = PlannerLoss(diversity_weight=0.1)
    opt = torch.optim.AdamW(planner.parameters(), lr=1e-3)

    # Train
    n_epochs = 50
    smoothness_weight = 2.0  # L2 penalty keeps actions from exploding

    # Initialize action head to output small values
    with torch.no_grad():
        last_linear = planner.action_head[-1]
        last_linear.weight.data *= 0.1
        last_linear.bias.data.zero_()
    loss_history = []
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
                ctx_emb = out["emb"][:, :ctx_len]
                goal_emb = out["emb"][:, -1:]

            actions = torch.clamp(planner(ctx_emb[:, -1:], goal_emb), -1, 1)
            info = {"pixels": batch["pixels"][:, :ctx_len]}
            hist_act = batch["action"][:, :ctx_len].reshape(B, ctx_len, fs, raw_act).mean(2)
            pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len,
                                             hist_actions=hist_act, goal_emb=goal_emb)
            loss, _ = loss_fn(actions, pred_embs, goal_emb)
            # Action smoothness: penalize large actions
            smooth_loss = actions.pow(2).mean()
            total_loss = loss + smoothness_weight * smooth_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            epoch_loss += total_loss.item()
        loss_history.append(epoch_loss / len(ds))
        if epoch % 20 == 0 or epoch == n_epochs - 1:
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

    actions = torch.clamp(planner(ctx_emb[:, -1:], goal_emb).detach(), -1, 1)
    pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len,
                                     hist_actions=hist_actions, goal_emb=goal_emb)
    goal = goal_emb.expand(-1, N, -1)
    costs = (pred_embs - goal).pow(2).mean(dim=-1)[0]
    best_idx = costs.argmin().item()
    planner_acts = np.clip(actions[0, best_idx].cpu().numpy(), -1, 1)

    gt_raw = batch["action"][:, ctx_len:ctx_len+T]
    gt_acts_full = gt_raw.reshape(1, T, fs, raw_act)[0].cpu().numpy()
    gt_acts_avg = gt_acts_full.mean(1)
    gt_actions_t = torch.from_numpy(gt_acts_avg).unsqueeze(0).unsqueeze(0).expand(-1, N, -1, -1).float().cuda()
    gt_embs, _ = planner_rollout(wm, gt_actions_t, info, history_size=ctx_len,
                                   hist_actions=hist_actions, goal_emb=goal_emb)
    gt_cost = (gt_embs - goal).pow(2).mean(dim=-1)[0, 0].item()

    natural_dist = (ctx_emb - goal_emb).pow(2).mean().item()

    print(f"\n{'='*50}")
    print(f"Planner best cost:  {costs[best_idx].item():.6f}")
    print(f"GT action cost:     {gt_cost:.6f}")
    print(f"Natural ctx→goal:   {natural_dist:.6f}")
    print(f"Planner actions: {np.round(planner_acts, 3).tolist()}")
    print(f"GT actions:      {np.round(gt_acts_avg, 3).tolist()}")

    # ── Simulations ──
    base_ds = ds_raw.dataset
    orig_idx = ds_raw.indices[sample_idx]
    start_frame = base_ds.clip_indices[orig_idx][1]
    init_state = base_ds.get_row_data(start_frame)["state"]
    if init_state.ndim > 1:
        init_state = init_state[0]

    # GT sim
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    gt_sim = run_sim(env, init_state, gt_acts_full, is_gt=True)
    env.close()

    # Planner sim
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    planner_sim = run_sim(env, init_state, planner_acts, is_gt=False)
    env.close()

    # Dataset frames
    ds_pixels = []
    for i in range(ctx_len + T):
        img = raw_item["pixels"][i].permute(1, 2, 0).numpy().astype(np.uint8)
        ds_pixels.append(img)

    # ── 3-way GIF: dataset | GT sim | planner sim ──
    target_size = ds_pixels[0].shape[:2]
    from PIL import Image

    def resize_to(img, h, w):
        if img.shape[:2] != (h, w):
            return np.array(Image.fromarray(img).resize((w, h), Image.NEAREST))
        return img

    # Align: dataset frame i → sim frame i*fs
    combined_frames = []
    n_frames = min(len(ds_pixels), max(len(gt_sim)//fs, len(planner_sim)//fs) + 1)
    for i in range(n_frames):
        ds_img = ds_pixels[min(i, len(ds_pixels)-1)]
        gt_i = min(i * fs, len(gt_sim) - 1)
        pl_i = min(i * fs, len(planner_sim) - 1)
        gt_img = resize_to(gt_sim[gt_i], *target_size)
        pl_img = resize_to(planner_sim[pl_i], *target_size)

        row = np.hstack([ds_img, gt_img, pl_img])
        combined_frames.append(row)

    imageio.mimsave(OUT / "compare_3way.gif", combined_frames, fps=2, loop=0)
    print(f"\nSaved: {OUT}/compare_3way.gif  ({len(combined_frames)} frames)")

    # ── PNG summary ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for i in range(3):
        axes[0, i].imshow(ds_pixels[i])
        axes[0, i].set_title(f"Context {i}")
        axes[0, i].axis("off")

    axes[1, 0].imshow(ds_pixels[-1])
    axes[1, 0].set_title(f"Goal (frame {total_frames-1})")
    axes[1, 0].axis("off")

    # Action comparison
    ax = axes[1, 1]
    planner_cum = np.cumsum(planner_acts, axis=0)
    gt_cum = np.cumsum(gt_acts_avg, axis=0)
    all_acts = actions[0].cpu().numpy()
    cmap = plt.cm.tab10
    for i in range(N):
        cum = np.cumsum(all_acts[i], axis=0)
        alpha = 0.9 if i == best_idx else 0.2
        lw = 2.5 if i == best_idx else 0.8
        ax.plot(cum[:, 0], cum[:, 1], 'o-', color=cmap(i % 10),
                linewidth=lw, alpha=alpha, markersize=5,
                label=f'Q{i}' if i == best_idx else None)
    ax.plot(gt_cum[:, 0], gt_cum[:, 1], 's--', color='black', linewidth=2.5, label='GT')
    ax.set_xlabel("Cumulative X"); ax.set_ylabel("Cumulative Y")
    ax.set_title(f"Action space\nPlanner={costs[best_idx]:.4f}  GT={gt_cost:.4f}")
    ax.axhline(0, color='gray', ls=':', alpha=0.3); ax.axvline(0, color='gray', ls=':', alpha=0.3)
    ax.legend(fontsize=6)

    axes[1, 2].plot(loss_history, color="#3fb950", linewidth=1.5)
    axes[1, 2].axhline(y=gt_cost, color="gray", linestyle="--", label=f"GT={gt_cost:.4f}")
    axes[1, 2].set_xlabel("Epoch"); axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_title(f"Loss (final={loss_history[-1]:.4f})")
    axes[1, 2].legend(fontsize=8)

    plt.suptitle(f"Planner T={T} (1 episode, {n_epochs} epochs)  |  "
                 f"Planner={costs[best_idx].item():.4f}  GT={gt_cost:.4f}  natural={natural_dist:.4f}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "compare_3way.png", dpi=120, bbox_inches="tight")
    print(f"Saved: {OUT}/compare_3way.png")
    print(f"\nfirefox {OUT}/compare_3way.gif")
    print(f"firefox {OUT}/compare_3way.png")


if __name__ == "__main__":
    main()
