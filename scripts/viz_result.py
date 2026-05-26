"""Compare multiple action sequences from the same initial state in PushT simulator.

Generic API: provide a state + list of named action sequences → renders all
side-by-side with dataset frames.

Usage as script:
    python scripts/viz_overfit_result.py [--ckpt PATH]

Usage as module:
    from scripts.viz_overfit_result import visualize_actions, run_sim
    visualize_actions(state, ds_frames, action_list, fs, ctx_len, out_dir)
"""

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch, imageio, gymnasium as gym
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")

OUT = Path(__file__).parent.parent / "output" / "viz"
OUT.mkdir(parents=True, exist_ok=True)

RAW_ACT = 2  # PushT action dim


# ── Simulation ───────────────────────────────────────────────────────────

def run_sim(state, actions_2d):
    """Run a sequence of (N, raw_act) actions in PushT env from initial state.

    Args:
        state: (7,) starting state
        actions_2d: (N_steps, raw_act) individual actions

    Returns:
        frames: list of rendered images; frames[0] = starting state
    """
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped._set_state(state)
    env.step(np.zeros(RAW_ACT, dtype=np.float32))
    frames = [env.render()]
    for a in actions_2d:
        env.step(np.clip(a.astype(np.float32), -1, 1))
        frames.append(env.render())
    env.close()
    return frames


# ── Visualization ────────────────────────────────────────────────────────

def visualize_actions(state, ds_frames, action_list, fs, ctx_len, out_dir,
                      filename="compare_actions"):
    """Render multiple action sequences from the same state, compare with dataset.

    Args:
        state: (7,) PushT state
        ds_frames: list of HWC uint8 dataset frames (ctx_len + horizon total)
        action_list: list of dicts [{"name": str, "actions": (total_steps, 2)}]
        fs: frameskip (actions per dataset frame)
        ctx_len: number of context frames (comparison starts at ds_frames[ctx_len])
        out_dir: pathlib.Path for output
        filename: base name for output files

    Each action dict uses actions in individual-step format (already expanded from
    coarse steps to fs*raw_dim then reshaped to (-1, raw_act)).
    """
    horizon = len(ds_frames) - ctx_len
    ds_h, ds_w = ds_frames[0].shape[:2]

    # Run all action sequences through simulator
    sim_results = {}
    for entry in action_list:
        name = entry["name"]
        acts = entry["actions"].reshape(-1, RAW_ACT).astype(np.float32)
        frames = run_sim(state, acts)
        sim_results[name] = frames
        print(f"  {name}: {len(frames)} sim frames")

    # ── GIF: all sequences side-by-side + goal ────────────────────────
    goal_img = ds_frames[-1]  # last frame = goal, static column
    combined = []

    for k in range(horizon):
        ds_i = ctx_len + k
        sim_step = k * fs

        row_parts = [ds_frames[ds_i]]
        for entry in action_list:
            frames = sim_results[entry["name"]]
            s = min(sim_step, len(frames) - 1)
            img = frames[s]
            if img.shape[:2] != (ds_h, ds_w):
                img = np.array(Image.fromarray(img).resize(
                    (ds_w, ds_h), Image.NEAREST))
            row_parts.append(img)
        row_parts.append(goal_img)  # static goal column

        combined.append(np.hstack(row_parts))

    gif_path = out_dir / f"{filename}.gif"
    imageio.mimsave(gif_path, combined, fps=2, loop=0)
    print(f"Saved: {gif_path}")

    # ── Action plot ──────────────────────────────────────────────────
    _plot_actions(action_list, fs, horizon, out_dir, filename)

    # ── Save action values ──────────────────────────────────────────
    actions_path = out_dir / f"{filename}_actions.npz"
    save_dict = {}
    for entry in action_list:
        key = entry["name"].replace(" ", "_").replace("(", "").replace(")", "")
        save_dict[key] = entry["actions"]
    np.savez(actions_path, **save_dict)
    print(f"Saved: {actions_path}")
    for entry in action_list:
        print(f"  {entry['name']}:")
        print(f"    {entry['actions']}")

    return sim_results


def _plot_actions(action_list, fs, horizon, out_dir, filename):
    """Plot each action sequence: one row per sequence, one column per coarse step."""
    n_actions = len(action_list)
    fig, axes = plt.subplots(n_actions, horizon,
                              figsize=(horizon * 3, n_actions * 2 + 1))
    if horizon == 1:
        axes = axes.reshape(n_actions, 1)

    for row_idx, entry in enumerate(action_list):
        name = entry["name"]
        acts = entry["actions"].reshape(horizon, fs, RAW_ACT)
        for t in range(horizon):
            ax = axes[row_idx, t] if n_actions > 1 else axes[t]
            for d in range(RAW_ACT):
                ax.plot(range(fs), acts[t, :, d], 'o-', markersize=3, alpha=0.7)
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(0, color='gray', ls=':', alpha=0.3)
            if row_idx == 0:
                ax.set_title(f"Step {t}")
            if t == 0:
                ax.set_ylabel(name, fontsize=9)

    fig.suptitle("Action sequences (each point = one raw action)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename}_actions.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir}/{filename}_actions.png")


# ── Overfit-specific helpers (for script mode) ───────────────────────────

def load_wm():
    from omegaconf import OmegaConf; import hydra; import stable_worldmodel as swm
    cache = Path(swm.data.utils.get_cache_dir())
    def _remap(d):
        if isinstance(d, dict):
            if "_target_" in d:
                t = d["_target_"]
                for o, n in [("stable_worldmodel.wm.lewm.LeWM", "jepa.JEPA"),
                             ("stable_worldmodel.wm.lewm.module.Predictor", "module.ARPredictor"),
                             ("stable_worldmodel.wm.lewm.module.Embedder", "module.Embedder"),
                             ("stable_worldmodel.wm.lewm.module.MLP", "module.MLP")]:
                    t = t.replace(o, n)
                d["_target_"] = t
            for v in d.values(): _remap(v)
        return d
    cfg = OmegaConf.create(_remap(json.loads((cache / "pusht_config.json").read_text())))
    wm = hydra.utils.instantiate(cfg)
    sd = torch.load(cache / "pusht_weights.pt", map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    wm.load_state_dict(sd, strict=False)
    return wm.cuda().eval()


def load_planner(ckpt_path):
    from planner import PlannerDecoder
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd_pl = {}
    for k, v in sd.items():
        if k.startswith("model."):
            sd_pl[k[len("model."):]] = v
    num_layers = sum(1 for k in sd_pl
                     if k.startswith("decoder.layers.") and "self_attn.in_proj_weight" in k)
    num_queries = sd_pl["query_embed"].shape[1]
    embed_dim = sd_pl["query_embed"].shape[2]
    planner = PlannerDecoder(embed_dim=embed_dim, num_queries=num_queries,
                             num_layers=num_layers, horizon=5, action_dim=2,
                             action_substeps=5)
    planner.load_state_dict(sd_pl)
    return planner.cuda().eval()


def main(ckpt_path=None, sample_idx=0, episode=0):
    if ckpt_path is None:
        # Find latest overfit checkpoint (sort by version number)
        ckpts = list(Path("output/lightning/lightning_logs").glob(
            "version_*/checkpoints/epoch=*-step=*.ckpt"))
        if ckpts:
            ckpt_path = str(sorted(ckpts, key=lambda p: int(
                p.parent.parent.name.split("_")[1]))[-1])
        else:
            raise FileNotFoundError("No checkpoint found")
    ctx_len, horizon, fs = 3, 5, 5

    wm = load_wm()
    for p in wm.parameters(): p.requires_grad_(False)
    planner = load_planner(ckpt_path)

    # Load data
    from utils import get_img_preprocessor
    import stable_worldmodel as swm
    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds_raw = swm.data.HDF5Dataset("pusht_expert_train", frameskip=fs,
                                   num_steps=ctx_len+horizon, transform=None)
    ep_idx = [i for i in range(len(ds_raw)) if ds_raw.clip_indices[i][0] == episode]
    ds_raw = torch.utils.data.Subset(ds_raw, ep_idx)
    ds_tf = swm.data.HDF5Dataset("pusht_expert_train", frameskip=fs,
                                  num_steps=ctx_len+horizon, transform=transform)
    ds_tf = torch.utils.data.Subset(ds_tf, ep_idx)

    item_raw = ds_raw[sample_idx]
    item_tf = ds_tf[sample_idx]
    ds_frames = [item_raw["pixels"][i].permute(1,2,0).numpy().astype(np.uint8)
                 for i in range(ctx_len + horizon)]
    batch = {k: v.unsqueeze(0).cuda() for k, v in item_tf.items() if torch.is_tensor(v)}

    # Get state at future-window start
    base = ds_raw.dataset
    orig_idx = ds_raw.indices[sample_idx]
    start_row = base.clip_indices[orig_idx][1]
    state_row = start_row + ctx_len * fs
    state = base.get_row_data(state_row)["state"]
    if state.ndim > 1: state = state[0]
    state = np.array(state).astype(np.float64)
    print(f"Start state: agent=({state[0]:.1f},{state[1]:.1f}) "
          f"block=({state[2]:.1f},{state[3]:.1f})")

    # Encode context + goal
    with torch.no_grad():
        out = wm.encode(batch)
        ctx_emb = out["emb"][:, :ctx_len]
        goal_emb = out["emb"][:, -1:]

    # Run planner through WM
    from planner import planner_rollout
    hist_actions = batch["action"][:, :ctx_len]
    info = {"pixels": batch["pixels"][:, :ctx_len]}

    with torch.no_grad():
        actions, conf = planner(ctx_emb, goal_emb)
        pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len,
                                       hist_actions=hist_actions, goal_emb=goal_emb)
        goal = goal_emb.reshape(1, 1, 1, -1).expand(-1, actions.shape[1], pred_embs.shape[2], -1)
        costs = (pred_embs - goal).pow(2).mean(dim=-1).mean(dim=-1)[0].cpu().numpy()
        best_idx = int(costs.argmin())
        conf_vals = torch.sigmoid(conf).squeeze().cpu().numpy()

    # GT cost
    gt_raw = item_raw["action"][ctx_len:ctx_len+horizon].numpy()
    gt_t = torch.from_numpy(gt_raw).float().cuda().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        gt_embs, _ = planner_rollout(wm, gt_t, info, history_size=ctx_len,
                                     hist_actions=hist_actions, goal_emb=goal_emb)
        gt_cost = (gt_embs - goal).pow(2).mean().item()

    print(f"Planner cost: {costs[best_idx]:.6f}  (best of {costs.argmin()})")
    print(f"GT cost:      {gt_cost:.6f}")
    print(f"All costs:    {costs}")
    print(f"Confidences:  {np.round(conf_vals, 3)}")

    # Prepare action list for visualization
    action_list = [
        {"name": "GT", "actions": gt_raw},
        {"name": f"Planner (cost={costs[best_idx]:.3f})",
         "actions": actions[0, best_idx].cpu().numpy()},
    ]

    visualize_actions(state, ds_frames, action_list, fs, ctx_len, OUT,
                      filename="planner_overfit_sim")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    main(ckpt_path=args.ckpt, sample_idx=args.sample, episode=args.episode)
