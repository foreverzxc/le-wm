"""Compare GT action simulation pixels vs dataset pixels.

Runs all individual GT actions in the PushT simulator and compares
rendered frames with dataset pixels at corresponding timesteps.

Usage:
    python scripts/compare_gt_sim_pixels.py [--episode EP] [--ctx_len C] [--horizon T]
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import imageio, gymnasium as gym
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")
import stable_worldmodel as swm

OUT = Path(__file__).parent.parent / "output" / "viz"
OUT.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────

def load_dataset(frameskip=5, ctx_len=3, horizon=5, episode=0):
    """Load PushT dataset, subset to one episode (matches diag_state_match.py)."""
    ds = swm.data.HDF5Dataset(
        "pusht_expert_train", frameskip=frameskip,
        num_steps=ctx_len + horizon, transform=None,
    )
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == episode]
    from torch.utils.data import Subset
    return Subset(ds, indices)


def get_state_at_frame(ds, sample_idx, frame_offset=0):
    """Extract PushT state (7,) at a specific frame offset within the clip.

    frame_offset=0 → state at clip start (dataset frame 0)
    frame_offset=ctx_len → state at beginning of future window

    Uses Subset→dataset indirection (matches diag).
    """
    base = ds.dataset
    orig_idx = ds.indices[sample_idx]
    start_frame = base.clip_indices[orig_idx][1]
    hdf5_row = start_frame + frame_offset * ds.dataset.frameskip
    state = base.get_row_data(hdf5_row)["state"]
    if state.ndim > 1:
        state = state[0]
    return np.array(state).astype(np.float64)


# ── Simulation ───────────────────────────────────────────────────────────

def run_gt_simulation(state, gt_actions, fs, raw_act):
    """Run all individual GT actions in the PushT env.

    Returns sim_frames aligned with dataset frames:
        sim_frames[0]      = initial state (after zero-step sync)
        sim_frames[i*fs]   = state after i dataset frames' worth of actions

    Args:
        state: (7,) starting state from dataset
        gt_actions: (T, fs*raw_act) raw dataset actions (concatenated)
        fs: frameskip
        raw_act: action dimension per step (2 for PushT)

    Returns:
        sim_frames: list of rendered np arrays, index 0 = starting state
    """
    actions_2d = gt_actions.reshape(-1, raw_act).astype(np.float32)

    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped._set_state(state)

    # Zero step to sync PyMunk physics with render after _set_state
    env.step(np.zeros(raw_act, dtype=np.float32))
    sim_frames = [env.render()]  # frame 0 = starting state, matches dataset frame 0

    for a in actions_2d:
        a = np.clip(a, -1, 1)
        env.step(a)
        sim_frames.append(env.render())

    env.close()
    return sim_frames


# ── Pixel comparison ─────────────────────────────────────────────────────

def compare_mse(sim_img, ds_img):
    """Per-pixel MSE between simulation and dataset frames (both HWC uint8)."""
    diff = sim_img.astype(np.float32) - ds_img.astype(np.float32)
    return float((diff ** 2).mean()), diff


def build_comparison_gif(ds_frames, sim_frames, fs, ctx_len, horizon):
    """Build side-by-side comparison: only the horizon future frames.

    Sim starts from state at dataset frame ctx_len, so:
        ds_frame[ctx_len + k]  ↔  sim[k * fs]

    Returns:
        combined: list of side-by-side HWC uint8 arrays
        metrics: list of dicts with mse per frame
    """
    ds_h, ds_w = ds_frames[0].shape[:2]
    combined = []
    metrics = []

    for k in range(horizon):  # include the starting state + T future frames
        ds_i = ctx_len + k
        ds_img = ds_frames[ds_i]
        sim_step = k * fs

        if sim_step >= len(sim_frames):
            sim_step = len(sim_frames) - 1

        sim_img = sim_frames[sim_step]

        if sim_img.shape[:2] != (ds_h, ds_w):
            sim_img = np.array(Image.fromarray(sim_img).resize(
                (ds_w, ds_h), Image.NEAREST))

        mse, _ = compare_mse(sim_img, ds_img)
        metrics.append({"ds_frame": ds_i, "sim_step": sim_step, "mse": mse})

        label = np.ones((24, ds_w * 2, 3), dtype=np.uint8) * 40
        row = np.hstack([ds_img, sim_img])
        combined.append(np.vstack([label, row]))

    return combined, metrics


# ── Visualization ─────────────────────────────────────────────────────────

def debug_initial_state(state, ds_frame_0, raw_act, out_dir):
    """Set state in fresh env, render, and compare with dataset frame 0.

    Saves a side-by-side image and returns pixel-level diff info.
    """
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped._set_state(state)

    # Zero step to sync render
    env.step(np.zeros(raw_act, dtype=np.float32))
    sim_img = env.render()

    # Read back actual env state
    uw = env.unwrapped
    actual = np.array([
        uw.agent.position[0], uw.agent.position[1],
        uw.block.position[0], uw.block.position[1],
        uw.block.angle,
    ])
    env.close()

    print(f"\n  Dataset state:  agent=({state[0]:.3f},{state[1]:.3f})  "
          f"block=({state[2]:.3f},{state[3]:.3f})  angle={state[4]:.4f}")
    print(f"  Env after set:  agent=({actual[0]:.3f},{actual[1]:.3f})  "
          f"block=({actual[2]:.3f},{actual[3]:.3f})  angle={actual[4]:.4f}")
    print(f"  State delta:    agent=({actual[0]-state[0]:.4f},{actual[1]-state[1]:.4f})  "
          f"block=({actual[2]-state[2]:.4f},{actual[3]-state[3]:.4f})  "
          f"angle={actual[4]-state[4]:.6f}")

    # Pixel diff
    diff = np.abs(sim_img.astype(float) - ds_frame_0.astype(float))
    print(f"  Pixel diff: mean={diff.mean():.2f}  max={diff.max():.0f}  "
          f"pct>10={(diff>10).mean()*100:.1f}%")

    # Side-by-side image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    ax1.imshow(ds_frame_0)
    ax1.set_title("Dataset frame 0")
    ax1.axis("off")
    ax2.imshow(sim_img)
    ax2.set_title("Sim after _set_state + zero step")
    ax2.axis("off")
    ax3.imshow(diff / max(diff.max(), 1))
    ax3.set_title(f"Diff  (mean={diff.mean():.1f}, max={diff.max():.0f})")
    ax3.axis("off")
    fig.tight_layout()
    out_path = out_dir / "debug_initial_state.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metrics(metrics, out_path):
    """Plot per-frame MSE."""
    fig, ax = plt.subplots(figsize=(8, 3))
    xs = [m["ds_frame"] for m in metrics]
    ys = [m["mse"] for m in metrics]
    ax.bar(xs, ys, color="#58a6ff", alpha=0.85)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("MSE (pixel space)")
    ax.set_title("MSE: Simulation vs Dataset pixels")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    for x, y in zip(xs, ys):
        ax.text(x, y + max(ys) * 0.03, f"{y:.1f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index (default: 0)")
    parser.add_argument("--ctx_len", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--frameskip", type=int, default=5)
    args = parser.parse_args()

    ctx_len = args.ctx_len
    T = args.horizon
    fs = args.frameskip
    raw_act = 2

    print(f"Loading PushT episode {args.episode} (ctx={ctx_len}, T={T}, fs={fs})...")
    ds = load_dataset(frameskip=fs, ctx_len=ctx_len, horizon=T, episode=args.episode)

    sample_idx = len(ds) // 2  # mid-episode sample
    item = ds[sample_idx]

    # Dataset frames: (ctx_len + T) frames, each (C, H, W)
    ds_frames = [
        item["pixels"][i].permute(1, 2, 0).numpy().astype(np.uint8)
        for i in range(ctx_len + T)
    ]

    # GT actions for the horizon window: (T, fs*raw_act)
    # action[i] spans pixel[i] → pixel[i+1]; we need actions[ctx_len : ctx_len+T]
    gt_raw = item["action"][ctx_len:ctx_len + T].numpy()

    # Starting state at the beginning of the action window (frame ctx_len)
    # Sim[0] must match ds_frame[ctx_len], NOT ds_frame[0]
    state = get_state_at_frame(ds, sample_idx, frame_offset=ctx_len)
    print(f"Start state (frame {ctx_len}): agent=({state[0]:.1f},{state[1]:.1f}) "
          f"block=({state[2]:.1f},{state[3]:.1f}) angle={state[4]:.2f}")

    # Debug: render initial state and compare with dataset frame ctx_len
    debug_initial_state(state, ds_frames[ctx_len], raw_act, OUT)

    # Run simulation
    print(f"Running {T*fs} individual GT actions...")
    sim_frames = run_gt_simulation(state, gt_raw, fs, raw_act)
    print(f"  simulated {len(sim_frames)} steps")

    # Compare
    combined, metrics = build_comparison_gif(ds_frames, sim_frames, fs,
                                             ctx_len, T)
    gif_path = OUT / "compare_gt_sim_pixels.gif"
    imageio.mimsave(gif_path, combined, fps=2, loop=0)
    print(f"Saved: {gif_path}  ({len(combined)} frames, left=dataset right=sim)")

    plot_metrics(metrics, OUT / "compare_gt_sim_pixels_mse.png")

    # Summary
    total_mse = sum(m["mse"] for m in metrics)
    print(f"\nSummary:")
    print(f"  Total MSE across {len(metrics)} frames: {total_mse:.1f}")
    print(f"  Mean MSE per frame: {total_mse / len(metrics):.1f}")
    print(f"  Max MSE: {max(m['mse'] for m in metrics):.1f}")
    for m in metrics:
        print(f"    ds_frame {m['ds_frame']:2d}  (sim step {m['sim_step']:3d}):  "
              f"MSE = {m['mse']:7.1f}")


if __name__ == "__main__":
    main()
