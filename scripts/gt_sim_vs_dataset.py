"""GT simulation vs dataset pixels, side-by-side GIF.

Each dataset frame (5 env steps) is compared to the corresponding sim frame.

Usage:
    python scripts/gt_sim_vs_dataset.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, torch, imageio, gymnasium as gym
from PIL import Image
import stable_worldmodel as swm

OUT = Path(__file__).parent.parent / "output" / "viz"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ctx_len, T, raw_act, fs = 3, 5, 2, 5

    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                               num_steps=ctx_len + T, transform=None)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices)

    sample_idx = len(ds) // 2
    item = ds[sample_idx]

    # Dataset frames: 8 frames (3 ctx + 5 future), each HWC
    ds_frames = [item["pixels"][i].permute(1,2,0).numpy().astype(np.uint8)
                 for i in range(ctx_len + T)]

    # GT actions: (T=5, fs=5, raw_act=2) = 25 individual actions
    gt_raw = item["action"][ctx_len:ctx_len+T].numpy()
    gt_actions = gt_raw.reshape(T, fs, raw_act)

    # Starting state
    base = ds.dataset
    orig_idx = ds.indices[sample_idx]
    start_frame = base.clip_indices[orig_idx][1]
    state = base.get_row_data(start_frame)["state"]
    if state.ndim > 1:
        state = state[0]

    # Run simulation
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped._set_state(state)

    sim_frames = []
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    sim_frames.append(env.render())  # frame 0 = starting state

    for a in gt_actions.reshape(-1, raw_act):
        a = np.clip(a.astype(np.float32), -1, 1)
        obs, _, terminated, truncated, _ = env.step(a)
        sim_frames.append(env.render())
        if terminated or truncated:
            break
    env.close()

    # Align: dataset frame i (starting at 0) corresponds to sim frame i*fs
    # Dataset frame 0 = state before any action = sim frame 0
    # Dataset frame 3 = after 3*5=15 actions = sim frame 15
    target_h, target_w = ds_frames[0].shape[:2]

    combined = []
    for ds_i in range(ctx_len + T):
        # Sim frame index: each dataset frame is 5 env steps later
        sim_i = ds_i * fs
        if sim_i >= len(sim_frames):
            sim_img = sim_frames[-1]
        else:
            sim_img = sim_frames[sim_i]

        # Resize sim to match dataset if sizes differ
        if sim_img.shape[:2] != (target_h, target_w):
            sim_img = np.array(Image.fromarray(sim_img).resize(
                (target_w, target_h), Image.NEAREST))

        # Side by side
        ds_img = ds_frames[ds_i]
        row = np.hstack([ds_img, sim_img])
        combined.append(row)

    imageio.mimsave(OUT / "gt_sim_vs_dataset.gif", combined, fps=2, loop=0)
    print(f"Saved: {OUT}/gt_sim_vs_dataset.gif  ({len(combined)} frames)")
    print("Left = dataset pixels  |  Right = GT simulation from same state")


if __name__ == "__main__":
    main()
