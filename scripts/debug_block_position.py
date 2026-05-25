"""Compare block position in dataset vs simulation frame by frame.

Usage:
    python scripts/debug_block_position.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, torch, gymnasium as gym
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")
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
    base = ds.dataset
    orig_idx = ds.indices[sample_idx]

    # Get state for EACH dataset frame
    print("Dataset frame states (from get_row_data):")
    ds_states = []
    for i in range(ctx_len + T):
        frame_num = base.clip_indices[orig_idx][1] + i * fs
        s = base.get_row_data(frame_num)["state"]
        if s.ndim > 1:
            s = s[0]
        ds_states.append(s)
        label = "CTX" if i < ctx_len else f"FUT+{(i-ctx_len)*fs}"
        print(f"  frame {i:2d} ({label:6s}): "
              f"agent=({s[0]:7.1f},{s[1]:7.1f})  "
              f"block=({s[2]:7.1f},{s[3]:7.1f})  angle={s[4]:.2f}")

    # Run simulation from frame 0 state
    init_state = ds_states[0]
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped._set_state(init_state)

    gt_raw = item["action"][ctx_len:ctx_len+T].numpy()
    gt_actions = gt_raw.reshape(T, fs, raw_act)  # (T, 5, 2)

    sim_states = []
    uw = env.unwrapped
    # Record initial state
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    sim_states.append(np.array([
        uw.agent.position[0], uw.agent.position[1],
        uw.block.position[0], uw.block.position[1],
        uw.block.angle, 0.0, 0.0
    ]))

    for a in gt_actions.reshape(-1, raw_act):
        a = np.clip(a.astype(np.float32), -1, 1)
        obs, _, _, _, _ = env.step(a)
        sim_states.append(np.array([
            uw.agent.position[0], uw.agent.position[1],
            uw.block.position[0], uw.block.position[1],
            uw.block.angle, 0.0, 0.0
        ]))
    env.close()

    # Compare: dataset frame i should match sim frame i*fs
    print("\nComparison (dataset frame vs sim at i*fs):")
    for ds_i in range(ctx_len + T):
        sim_i = ds_i * fs
        if sim_i >= len(sim_states):
            break
        ds_s = ds_states[ds_i]
        sim_s = sim_states[sim_i]
        agent_diff = np.sqrt(((ds_s[:2] - sim_s[:2])**2).sum())
        block_diff = np.sqrt(((ds_s[2:4] - sim_s[2:4])**2).sum())
        angle_diff = abs(ds_s[4] - sim_s[4])
        label = "CTX" if ds_i < ctx_len else f"FUT+{(ds_i-ctx_len)*fs}"
        print(f"  frame {ds_i:2d} ({label:6s}): "
              f"agent Δ={agent_diff:.1f}px  "
              f"block Δ={block_diff:.1f}px  "
              f"angle Δ={angle_diff:.3f}rad")

    # Visual comparison at key frames
    ds_pixels = [item["pixels"][i].permute(1,2,0).numpy().astype(np.uint8)
                 for i in range(ctx_len + T)]

    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped._set_state(init_state)
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    sim_pixels = [env.render()]
    for a in gt_actions.reshape(-1, raw_act):
        a = np.clip(a.astype(np.float32), -1, 1)
        obs, _, _, _, _ = env.step(a)
        sim_pixels.append(env.render())
    env.close()

    # Show key frames side by side
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    key_frames = [0, 1, 2, 3, 5, 7]  # ctx 0,1,2 + future 0,2,4
    for j, fi in enumerate(key_frames):
        if j < 3:
            ax = axes[0, j]
        else:
            ax = axes[1, j-3]
        sim_i = fi * fs
        ds_img = ds_pixels[fi]
        sim_img = sim_pixels[min(sim_i, len(sim_pixels)-1)]
        from PIL import Image
        if sim_img.shape[:2] != ds_img.shape[:2]:
            sim_img = np.array(Image.fromarray(sim_img).resize(
                (ds_img.shape[1], ds_img.shape[0]), Image.NEAREST))
        combined = np.hstack([ds_img, sim_img])
        ax.imshow(combined)
        ax.set_title(f"Frame {fi} (sim step {sim_i}): ds (left) vs sim (right)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "debug_block_position.png", dpi=120)
    print(f"\nSaved: {OUT}/debug_block_position.png")


if __name__ == "__main__":
    main()
