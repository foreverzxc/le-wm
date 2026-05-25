"""Debug: compare dataset pixel vs env render after _set_state.

Usage:
    python scripts/debug_state_match.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, gymnasium as gym
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")
import stable_worldmodel as swm

OUT = Path(__file__).parent.parent / "output" / "viz"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                               transform=None)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = swm.data.Subset(ds, indices) if hasattr(swm.data, 'Subset') else ds
    if not hasattr(ds, 'clip_indices'):
        from torch.utils.data import Subset
        ds = Subset(ds, indices)

    sample_idx = len(ds) // 2
    item = ds[sample_idx]
    ctx_img = item["pixels"][0].permute(1, 2, 0).numpy().astype(np.uint8)

    # Get state at context start
    base = ds.dataset if hasattr(ds, 'dataset') else ds
    orig_idx = ds.indices[sample_idx] if hasattr(ds, 'indices') else sample_idx
    start_frame = base.clip_indices[orig_idx][1]
    state = base.get_row_data(start_frame)["state"]
    if state.ndim > 1:
        state = state[0]
    print(f"Dataset state: agent=({state[0]:.1f},{state[1]:.1f}) "
          f"block=({state[2]:.1f},{state[3]:.1f}) angle={state[4]:.2f}")

    # Env
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    uw = env.unwrapped

    print(f"After reset:   agent=({uw.agent.position[0]:.1f},{uw.agent.position[1]:.1f}) "
          f"block=({uw.block.position[0]:.1f},{uw.block.position[1]:.1f}) "
          f"angle={uw.block.angle:.2f}")

    uw._set_state(state)
    print(f"After set:     agent=({uw.agent.position[0]:.1f},{uw.agent.position[1]:.1f}) "
          f"block=({uw.block.position[0]:.1f},{uw.block.position[1]:.1f}) "
          f"angle={uw.block.angle:.2f}")

    # Step once with zero action to render
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    sim_img = env.render()
    env.close()

    # Compare
    diff = np.abs(ctx_img.astype(float) - sim_img.astype(float))
    print(f"\nPixel comparison:")
    print(f"  Mean abs diff:     {diff.mean():.2f}  (max possible=255)")
    print(f"  Max abs diff:      {diff.max():.0f}")
    print(f"  Pixels diff > 10:  {(diff > 10).mean()*100:.1f}%")
    print(f"  Pixels diff > 50:  {(diff > 50).mean()*100:.1f}%")
    print(f"  Dataset bg (corner): {ctx_img[0,:5,0].mean():.0f}")
    print(f"  Sim bg (corner):     {sim_img[0,:5,0].mean():.0f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    ax1.imshow(ctx_img)
    ax1.set_title("Dataset pixel (ctx frame 0)")
    ax1.axis("off")
    ax2.imshow(sim_img)
    ax2.set_title("Env render after _set_state + zero step")
    ax2.axis("off")
    ax3.imshow(diff / max(diff.max(), 1))
    ax3.set_title(f"Difference (mean={diff.mean():.1f}, max={diff.max():.0f})")
    ax3.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "debug_state_match.png", dpi=120)
    print(f"Saved: {OUT}/debug_state_match.png")


if __name__ == "__main__":
    main()
