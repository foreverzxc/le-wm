"""Check dataset action values and simulation divergence cause.

Usage:
    python scripts/debug_action_values.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, torch, gymnasium as gym
import stable_worldmodel as swm

OUT = Path(__file__).parent.parent / "output" / "viz"


def main():
    ctx_len, T, raw_act, fs = 3, 5, 2, 5

    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                               num_steps=ctx_len + T, transform=None)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices)
    sample_idx = len(ds) // 2
    item = ds[sample_idx]

    # Check action values: are they in [-1, 1]?
    all_actions = item["action"].numpy()
    print(f"Action shape: {all_actions.shape}  (num_steps={ctx_len+T}, frameskip*raw_dim={fs*raw_act})")
    print(f"Action range: [{all_actions.min():.4f}, {all_actions.max():.4f}]")
    print(f"Action mean:  {all_actions.mean():.4f}")
    print()

    # Show each step's 5 individual actions
    for step in range(ctx_len + T):
        acts = all_actions[step].reshape(fs, raw_act)
        print(f"Step {step}: {np.round(acts, 4).tolist()}")

    # Now test: does a single zero step cause big divergence?
    base = ds.dataset
    orig_idx = ds.indices[sample_idx]
    start_frame = base.clip_indices[orig_idx][1]
    state = base.get_row_data(start_frame)["state"]
    if state.ndim > 1:
        state = state[0]

    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    uw = env.unwrapped

    # Set state, check before ANY step
    uw._set_state(state)
    pos_before = (uw.agent.position[0], uw.agent.position[1])
    print(f"\nAfter _set_state: agent=({pos_before[0]:.1f}, {pos_before[1]:.1f}) "
          f"block=({uw.block.position[0]:.1f},{uw.block.position[1]:.1f})")

    # One zero step
    env.step(np.zeros(2, dtype=np.float32))
    pos_after_zero = (uw.agent.position[0], uw.agent.position[1])
    delta_zero = np.sqrt((pos_after_zero[0]-pos_before[0])**2 +
                         (pos_after_zero[1]-pos_before[1])**2)
    print(f"After 1 zero step: agent=({pos_after_zero[0]:.1f}, {pos_after_zero[1]:.1f})  "
          f"Δ={delta_zero:.1f}px")

    # Apply first GT action and compare with dataset
    first_act = all_actions[ctx_len].reshape(fs, raw_act)  # first future action, 5 steps
    for i, a in enumerate(first_act):
        a = np.clip(a.astype(np.float32), -1, 1)
        env.step(a)
    pos_after_5 = (uw.agent.position[0], uw.agent.position[1])

    # Dataset agent position after 5 steps
    state_after_5 = base.get_row_data(start_frame + fs)["state"]
    if state_after_5.ndim > 1:
        state_after_5 = state_after_5[0]
    ds_agent = state_after_5[:2]

    delta = np.sqrt((pos_after_5[0]-ds_agent[0])**2 + (pos_after_5[1]-ds_agent[1])**2)
    print(f"After 5 GT actions: agent=({pos_after_5[0]:.1f},{pos_after_5[1]:.1f})")
    print(f"Dataset agent:      agent=({ds_agent[0]:.1f},{ds_agent[1]:.1f})")
    print(f"Δ={delta:.1f}px")

    # KEY TEST: what if we DON'T do the zero step?
    env.close()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    uw = env.unwrapped
    uw._set_state(state)
    # Directly apply first GT action, no zero step
    for i, a in enumerate(first_act):
        a = np.clip(a.astype(np.float32), -1, 1)
        env.step(a)
    pos_no_zero = (uw.agent.position[0], uw.agent.position[1])
    delta_no_zero = np.sqrt((pos_no_zero[0]-ds_agent[0])**2 +
                            (pos_no_zero[1]-ds_agent[1])**2)
    print(f"\nWithout zero step:")
    print(f"After 5 GT actions: agent=({pos_no_zero[0]:.1f},{pos_no_zero[1]:.1f})")
    print(f"Δ={delta_no_zero:.1f}px (vs zero-step: {delta:.1f}px)")
    env.close()


if __name__ == "__main__":
    main()
