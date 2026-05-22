"""Multi-dataset wrapper for cross-domain world model training.

Combines HDF5Datasets, pads actions to a unified dimension, and provides
weighted sampling to balance large (Pusht 18K eps) and small datasets.
"""
from pathlib import Path

import numpy as np
import torch


class MultiDomainDataset:
    """Wrapper that concatenates multiple HDF5-like datasets.

    - Pads action to ``max_action_dim`` (zero-pad right side)
    - Provides ``get_col_data`` by querying only the first dataset (used for
      image normalizer init — per-column normalizers are skipped in multi-dataset
      training since they're not critical for embedding-space losses)
    - ``get_dim`` delegates to the first dataset for action dim (used by train.py
      to set ``action_encoder.input_dim``)
    """

    @staticmethod
    def _unwrap(ds):
        """Get underlying dataset attributes through Subset wrappers."""
        return ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds

    def __init__(self, datasets: list, max_action_dim: int):
        self.datasets = datasets
        self.max_action_dim = max_action_dim
        self.lengths = np.array([len(d) for d in datasets])
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths[:-1])])
        base = self._unwrap(datasets[0])
        self.frameskip = base.frameskip
        self.num_steps = base.num_steps

    # ── torch Dataset interface ──────────────────────────────────────

    def __len__(self):
        return int(self.lengths.sum())

    def __getitem__(self, idx: int) -> dict:
        ds_idx = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        ds_idx = max(0, min(ds_idx, len(self.datasets) - 1))
        local_idx = idx - int(self.offsets[ds_idx])
        item = self.datasets[ds_idx][local_idx]

        # Pad action to unified dimension
        action = item["action"]
        pad_needed = self.max_action_dim - action.shape[-1]
        if pad_needed > 0:
            pad = torch.zeros(*action.shape[:-1], pad_needed, dtype=action.dtype)
            action = torch.cat([action, pad], dim=-1)

        # Return only keys needed by JEPA model — different datasets
        # have different auxiliary keys that break DataLoader collation.
        return {"pixels": item["pixels"], "action": action}

    # ── swm.Dataset compatibility ────────────────────────────────────

    def get_col_data(self, col: str) -> np.ndarray:
        """Return sample column from first dataset (used for img normalizer init)."""
        return self._unwrap(self.datasets[0]).get_col_data(col)

    def get_dim(self, col: str) -> int:
        """Return unified raw action dim, delegate others."""
        if col == "action":
            return self.max_action_dim // self.frameskip
        return self._unwrap(self.datasets[0]).get_dim(col)

    @property
    def transform(self):
        return self._unwrap(self.datasets[0]).transform

    @transform.setter
    def transform(self, t):
        for ds in self.datasets:
            self._unwrap(ds).transform = t


def build_cross4_dataset(max_samples: int = 10000):
    """Build the 4-dataset training set (Pusht + Cube + Reacher + TwoRooms).

    Args:
        max_samples: Max samples per dataset (subsampled for fast iteration).
    """
    import os
    import stable_worldmodel as swm
    from utils import get_img_preprocessor

    transform = get_img_preprocessor("pixels", "pixels", 224)

    ds_configs = [
        ("pusht_expert_train", 2),     # 2D action
        ("cube_single_expert", 5),     # 5D action (Cube = max)
        ("reacher", 2),                 # 2D action
        ("tworoom", 2),                 # 2D action
    ]

    datasets = []
    for name, act_dim in ds_configs:
        ds = swm.data.HDF5Dataset(
            name, frameskip=5, num_steps=4,
            keys_to_cache=["action"],
            transform=transform,
        )
        # Subsample evenly across episodes for diversity
        if max_samples and len(ds) > max_samples:
            n_eps = len(ds.lengths)
            indices = []
            per_ep = max(1, max_samples // n_eps)
            for ep in range(n_eps):
                lo = int(ds.offsets[ep] // ds.frameskip)
                hi = int((ds.offsets[ep] + ds.lengths[ep]) // ds.frameskip) - ds.num_steps
                hi = max(lo, hi)
                ep_idx = np.linspace(lo, hi, min(per_ep, hi - lo + 1), dtype=int)
                indices.extend(ep_idx.tolist())
            indices = np.array(indices[:max_samples])
            ds = torch.utils.data.Subset(ds, indices)
        datasets.append(ds)
        n_eps = len(ds.dataset.lengths) if hasattr(ds, 'dataset') else len(ds.lengths)
        print(f"  {name}: {len(ds)} samples, {n_eps} episodes, "
              f"action={act_dim * 5}D")

    return MultiDomainDataset(datasets, max_action_dim=25)
