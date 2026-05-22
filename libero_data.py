"""LIBERO dataset adapter — reads native LIBERO HDF5 format (demo-grouped)
and presents it in the flat episode-based interface expected by the training pipeline.
"""

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from stable_worldmodel.data.dataset import Dataset


_OBS_KEY_MAP = {
    "agentview_rgb": "pixels",
    "eye_in_hand_rgb": "pixels_wrist",
    "ee_pos": "ee_pos",
    "ee_ori": "ee_ori",
    "ee_states": "ee_states",
    "gripper_states": "gripper_states",
    "joint_states": "joint_states",
}

# Non-image proprioceptive keys get concatenated into "proprio" for the model.
_PROPRIO_KEYS = ["ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states"]


class LiberoDataset(Dataset):
    """Dataset that reads LIBERO HDF5 files (one file per task, demo groups inside).

    Each ``.hdf5`` file contains ``data/demo_0, demo_1, ...`` groups.  Each demo
    is treated as one episode.  All demos across all files are concatenated into a
    flat episode index.

    Args:
        path: Directory containing LIBERO ``.hdf5`` files.
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample sequence.
        transform: Optional transform callable applied to each item.
        image_key: Which observation key to use as ``pixels`` (default ``agentview_rgb``).
        max_episodes: Limit to first N episodes (for debugging/overfitting).
    """

    def __init__(
        self,
        path: str | Path,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        image_key: str = "agentview_rgb",
        max_episodes: int | None = None,
        **kwargs,
    ):
        self.root = Path(path)
        self.image_key = image_key
        self._files: list[Path] = sorted(self.root.glob("*.hdf5"))

        # Scan all files and demos to build episode table.
        episode_lengths: list[int] = []
        episode_meta: list[tuple[Path, str]] = []  # (file, demo_key)

        for fp in self._files:
            with h5py.File(fp, "r") as f:
                for demo_key in f["data"]:
                    grp = f["data"][demo_key]
                    n = grp["actions"].shape[0]
                    episode_lengths.append(n)
                    episode_meta.append((fp, demo_key))
                    if max_episodes and len(episode_lengths) >= max_episodes:
                        break
            if max_episodes and len(episode_lengths) >= max_episodes:
                break

        self._episode_lengths = np.array(episode_lengths)
        self._episode_meta = episode_meta
        offsets = np.cumsum(np.concatenate([[0], self._episode_lengths[:-1]]))

        # Compute max dims for variable columns across files (states differ across scenes).
        self._max_dims: dict[str, int] = {}
        seen_files: set[Path] = set()
        for fp, demo_key in self._episode_meta:
            if fp in seen_files:
                continue
            seen_files.add(fp)
            with h5py.File(fp, "r") as f:
                demo = f["data"][demo_key]
                for col in ["states"]:
                    if col in demo:
                        self._max_dims[col] = max(self._max_dims.get(col, 0), demo[col].shape[-1])

        # Cache for column data and file handles.
        self._col_cache: dict[str, np.ndarray] = {}
        self._file_handles: dict[Path, h5py.File] = {}

        super().__init__(
            lengths=self._episode_lengths,
            offsets=offsets,
            frameskip=frameskip,
            num_steps=num_steps,
            transform=transform,
        )

    # ------------------------------------------------------------------
    #  File handle cache
    # ------------------------------------------------------------------

    def _get_file_handle(self, fp: Path) -> h5py.File:
        """Return a cached HDF5 file handle, opening once and reusing."""
        if fp not in self._file_handles:
            self._file_handles[fp] = h5py.File(fp, "r")
        return self._file_handles[fp]

    def close(self):
        """Release all cached file handles."""
        for f in self._file_handles.values():
            f.close()
        self._file_handles.clear()

    # ------------------------------------------------------------------
    #  Column metadata
    # ------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        return ["pixels", "action", "proprio", "state", "episode_idx", "step_idx"]

    def get_dim(self, col: str) -> int:
        if col in ("episode_idx", "step_idx"):
            return 1

        f = self._get_file_handle(self._files[0])
        first_demo_key = sorted(f["data"].keys())[0]
        demo = f["data"][first_demo_key]

        if col == "pixels":
            return demo["obs"][self.image_key].shape[-1]  # channels
        if col == "action":
            return demo["actions"].shape[-1]
        if col == "proprio":
            obs = demo["obs"]
            return sum(obs[k].shape[-1] for k in _PROPRIO_KEYS if k in obs)
        if col == "state":
            return self._max_dims.get("states", demo["states"].shape[-1] if "states" in demo else 0)

        obs = demo["obs"]
        if col in obs:
            return obs[col].shape[-1]
        if col in demo:
            return demo[col].shape[-1]

        raise KeyError(f"Unknown column {col!r}")

    # ------------------------------------------------------------------
    #  Data loading helpers
    # ------------------------------------------------------------------

    def _load_demo_slice(self, demo: h5py.Group, start: int, end: int) -> dict:
        """Load a contiguous slice from one demo group.

        All columns *except* action are frameskip-sub-sampled.  Action keeps
        every frame so the base-class ``__getitem__`` can reshape
        ``(span, D) → (num_steps, frameskip * D)``.
        """
        step = self.frameskip
        sl = slice(start, end, step) if step > 1 else slice(start, end)

        obs_grp = demo["obs"]
        pixels = obs_grp[self.image_key][sl]  # (T/step, H, W, 3)

        # Proprio: concatenate pre-defined keys (frameskip sub-sampled)
        proprio_parts = [obs_grp[k][sl] for k in _PROPRIO_KEYS if k in obs_grp]
        proprio = (
            np.concatenate(proprio_parts, axis=-1)
            if proprio_parts
            else np.zeros((pixels.shape[0], 0))
        )

        # State (frameskip sub-sampled): pad to max dim across scenes
        if "states" in demo:
            state = demo["states"][sl]
            max_dim = self._max_dims.get("states", state.shape[-1])
            if state.shape[-1] < max_dim:
                state = np.concatenate(
                    [state, np.zeros((state.shape[0], max_dim - state.shape[-1]), dtype=state.dtype)],
                    axis=-1,
                )
        else:
            state = np.zeros((pixels.shape[0], 0))

        # Action: load full span (no frameskip) — base __getitem__ handles reshape
        action = demo["actions"][start:end]  # (span, D)

        return {
            "pixels": pixels,
            "action": action,
            "proprio": proprio,
            "state": state,
        }

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        fp, demo_key = self._episode_meta[ep_idx]
        f = self._get_file_handle(fp)
        demo = f["data"][demo_key]
        steps = self._load_demo_slice(demo, start, end)

        # Convert to torch and handle channel layout.
        for col, val in steps.items():
            if isinstance(val, np.ndarray):
                data = torch.from_numpy(val.copy())
                # Permute image-like columns: (T, H, W, C) → (T, C, H, W)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    data = data.permute(0, 3, 1, 2)
                steps[col] = data

        return steps

    # ------------------------------------------------------------------
    #  Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)
        if self.transform:
            steps = self.transform(steps)
        if "action" in steps:
            steps["action"] = steps["action"].reshape(self.num_steps, -1)
        return steps

    def load_episode(self, episode_idx: int) -> dict:
        fp, demo_key = self._episode_meta[episode_idx]
        f = self._get_file_handle(fp)
        demo = f["data"][demo_key]
        return self._load_demo_slice(demo, 0, demo["actions"].shape[0])

    # ------------------------------------------------------------------
    #  Column-level access (for normalizer computation)
    # ------------------------------------------------------------------

    def get_col_data(self, col: str) -> np.ndarray:
        if col in self._col_cache:
            return self._col_cache[col]

        if col in ("episode_idx", "step_idx"):
            raise KeyError(f"Virtual column {col!r} — use get_row_data for row-level access.")

        data = self._build_col(col)
        self._col_cache[col] = data
        return data

    def _build_col(self, col: str) -> np.ndarray:
        """Concatenate `col` across all episodes into a single flat array."""
        parts = []
        for fp, demo_key in self._episode_meta:
            f = self._get_file_handle(fp)
            demo = f["data"][demo_key]
            if col == "pixels":
                parts.append(demo["obs"][self.image_key][:])
            elif col == "action":
                parts.append(demo["actions"][:])
            elif col == "proprio":
                obs = demo["obs"]
                pp = [obs[k][:] for k in _PROPRIO_KEYS if k in obs]
                parts.append(np.concatenate(pp, axis=-1) if pp else np.zeros((demo["actions"].shape[0], 0)))
            elif col == "state":
                if "states" in demo:
                    arr = demo["states"][:]
                    max_dim = self._max_dims.get("states", arr.shape[-1])
                    if arr.shape[-1] < max_dim:
                        pad = np.zeros((arr.shape[0], max_dim - arr.shape[-1]), dtype=arr.dtype)
                        arr = np.concatenate([arr, pad], axis=-1)
                    parts.append(arr)
                else:
                    n = demo["actions"].shape[0]
                    parts.append(np.zeros((n, 0)))
            else:
                obs = demo["obs"]
                if col in obs:
                    parts.append(obs[col][:])
                elif col in demo:
                    parts.append(demo[col][:])
                else:
                    raise KeyError(f"Unknown column {col!r}")
        return np.concatenate(parts, axis=0)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        idx = np.atleast_1d(np.asarray(row_idx))

        # Map flat row index to (ep_idx, local_step).
        offsets = self.offsets
        ep_idx = np.searchsorted(offsets, idx, side="right") - 1
        ep_idx = np.clip(ep_idx, 0, len(offsets) - 1)
        local_step = idx - offsets[ep_idx]

        result: dict = {}
        for i, (ep, step) in enumerate(zip(ep_idx, local_step)):
            fp, demo_key = self._episode_meta[ep]
            f = self._get_file_handle(fp)
            demo = f["data"][demo_key]
            row = self._load_demo_slice(demo, int(step), int(step) + 1)
            if i == 0:
                result = {k: [v] for k, v in row.items()}
            else:
                for k, v in row.items():
                    result[k].append(v)

        result = {k: np.concatenate(v, axis=0) for k, v in result.items()}
        result["episode_idx"] = ep_idx[:, None]
        result["step_idx"] = local_step[:, None]
        return result


# ------------------------------------------------------------------
#  Factory function — called from config
# ------------------------------------------------------------------

def create_libero_dataset(cfg, image_key: str = "agentview_rgb"):
    """Instantiate a LiberoDataset from an OmegaConf dict config.

    Expected config keys: ``path``, ``frameskip``, ``num_steps``.
    """
    return LiberoDataset(
        path=cfg.path,
        frameskip=cfg.get("frameskip", 1),
        num_steps=cfg.get("num_steps", 1),
        image_key=image_key,
    )
