"""Check action ordering in HDF5 dataset and compare sim trajectories.

Usage:
    python scripts/debug_action_order.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, torch, gymnasium as gym, h5py
import stable_worldmodel as swm


def main():
    ctx_len, T, raw_act, fs = 3, 5, 2, 5

    # Open HDF5 directly to check raw action storage
    h5_path = Path(swm.data.utils.get_cache_dir()) / "pusht_expert_train.h5"
    with h5py.File(h5_path, "r") as f:
        # Check raw action shape
        raw_actions = f["action"][:]
        print(f"Raw HDF5 action shape: {raw_actions.shape}")  # should be (N, 2)
        print(f"Raw action[0:5]: {raw_actions[:5]}")  # first 5 actions
        print(f"Raw action[5:10]: {raw_actions[5:10]}")  # next 5 actions

        # Check what dataset.__getitem__ returns
        ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5,
                                   num_steps=ctx_len+T, transform=None)
        indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
        ds = torch.utils.data.Subset(ds, indices)
        sample_idx = len(ds) // 2
        item = ds[sample_idx]

        item_actions = item["action"].numpy()  # (8, 10)
        print(f"\nDataset __getitem__ action shape: {item_actions.shape}")
        print(f"Step 3 (first future action): {item_actions[3]}")

        # Get the raw actions from HDF5 that correspond to this item's frames
        base = ds.dataset
        orig_idx = ds.indices[sample_idx]
        start_frame = base.clip_indices[orig_idx][1]
        span = (ctx_len + T) * fs  # 8 steps × 5 = 40 env steps
        h5_actions = raw_actions[start_frame:start_frame + span]
        print(f"\nHDF5 actions for this sample (frame {start_frame} to {start_frame+span}):")
        print(f"  Shape: {h5_actions.shape}")
        print(f"  First 10 raw actions: {h5_actions[:10].tolist()}")

        # Reshape HDF5 actions the way dataset does: (8, 5, 2)
        h5_reshaped = h5_actions.reshape(ctx_len+T, fs, raw_act)
        print(f"\n  Reshaped (8,5,2) step 3: {h5_reshaped[3].tolist()}")
        print(f"  Dataset step 3:                 {item_actions[3].reshape(fs, raw_act).tolist()}")

        # Are they the SAME?
        match = np.allclose(h5_reshaped[3], item_actions[3].reshape(fs, raw_act))
        print(f"  Match: {match}")

        if not match:
            # Try different reshape: maybe (8, 2, 5)?
            h5_alt = h5_actions.reshape(ctx_len+T, raw_act, fs).transpose(0, 2, 1)
            match2 = np.allclose(h5_alt[3], item_actions[3].reshape(fs, raw_act))
            print(f"  Alt reshape (8,2,5→transpose) match: {match2}")


if __name__ == "__main__":
    main()
