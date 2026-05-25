"""Check PushT env state structure and fix set_env_state if needed.

Usage:
    python scripts/debug_pusht_state.py
"""
import gymnasium as gym
import numpy as np
import stable_worldmodel as swm


def main():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()

    uw = env.unwrapped
    print("PushT env attributes after reset:")
    attrs = [a for a in dir(uw) if not a.startswith("_")]
    for a in sorted(attrs):
        if not callable(getattr(uw, a)):
            val = getattr(uw, a)
            if hasattr(val, 'position'):
                print(f"  {a}.position = {val.position}")
            print(f"  {a} = {type(val).__name__}")

    print()

    # Check _set_state source
    import inspect
    print("_set_state source:")
    print(inspect.getsource(uw._set_state))

    # Check what's different: dataset state format
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4)
    for step in [0, 100, 500, 1000]:
        s = ds.get_row_data(step)["state"]
        if s.ndim > 1:
            s = s[0]
        print(f"  step {step:5d}: agent=({s[0]:.1f},{s[1]:.1f})  "
              f"block=({s[2]:.1f},{s[3]:.1f})  angle={s[4]:.2f}  vel=({s[5]:.1f},{s[6]:.1f})")

    # Try to find where 'agent' is in the env
    print()
    print("Looking for agent in env tree...")
    for attr in dir(uw):
        obj = getattr(uw, attr)
        if hasattr(obj, 'position') and hasattr(obj, 'velocity'):
            print(f"  Found position+velocity on: {attr}")
            print(f"    position: {obj.position}")
            print(f"    velocity: {obj.velocity}")
    env.close()


if __name__ == "__main__":
    main()
