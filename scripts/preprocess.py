"""Precompute WM embeddings for PushT dataset.

Saves (ctx_emb, goal_emb, gt_actions) per clip to a .pt file.
Training can then use these directly, skipping HDF5 reads and ViT encoding.

Usage:
    python scripts/preprocess.py                     # full dataset
    python scripts/preprocess.py --data_fraction 0.1  # 10%
    python scripts/preprocess.py --max_episodes 50    # first 50 episodes
"""

import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import hydra, stable_worldmodel as swm
from omegaconf import OmegaConf
from tqdm import tqdm


def load_wm(ckpt_name="pusht"):
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
            for v in d.values():
                _remap(v)
        return d
    cfg = OmegaConf.create(_remap(json.loads((cache / f"{ckpt_name}_config.json").read_text())))
    wm = hydra.utils.instantiate(cfg)
    sd = torch.load(cache / f"{ckpt_name}_weights.pt", map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    wm.load_state_dict(sd, strict=False)
    return wm.cuda().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of clips to encode per WM call")
    parser.add_argument("--out", type=str,
                        default="pusht_precomputed.pt")
    args = parser.parse_args()

    ctx_len, horizon, fs = 3, 5, 5
    num_steps = ctx_len + horizon

    print("Loading WM...")
    wm = load_wm("pusht")
    for p in wm.parameters():
        p.requires_grad_(False)

    print("Loading dataset...")
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=fs,
                               num_steps=num_steps,
                               keys_to_cache=[], transform=None)
    from utils import get_img_preprocessor
    ds.transform = get_img_preprocessor("pixels", "pixels", 224)

    # Select clips
    indices = list(range(len(ds)))
    if args.max_episodes > 0:
        indices = [i for i in indices
                   if ds.clip_indices[i][0] < args.max_episodes]
    if args.data_fraction < 1.0:
        n = max(1, int(len(indices) * args.data_fraction))
        indices = indices[:n]

    print(f"Preprocessing {len(indices):,} clips...")

    all_ctx = []      # list of (ctx_len, D)
    all_goal = []     # list of (1, D)
    all_actions = []  # list of (horizon, fs*raw_dim)

    t0 = time.time()
    bs = args.batch_size

    for start in tqdm(range(0, len(indices), bs)):
        batch_indices = indices[start:start + bs]
        # Collect batch of pixels
        pixels_list = []
        actions_list = []
        for idx in batch_indices:
            item = ds[idx]
            pixels_list.append(item["pixels"])       # (num_steps, C, H, W)
            actions_list.append(item["action"])      # (num_steps, fs*raw_dim)

        pixels = torch.stack(pixels_list).cuda()  # (B, num_steps, C, H, W)
        actions = torch.stack(actions_list)        # (B, num_steps, fs*raw_dim)

        with torch.no_grad():
            out = wm.encode({"pixels": pixels})
            emb = out["emb"]  # (B, num_steps, D)

        # Store: ctx_emb (first ctx_len frames), goal_emb (last frame), gt_actions
        all_ctx.append(emb[:, :ctx_len].cpu())           # (B, ctx_len, D)
        all_goal.append(emb[:, -1:].cpu())               # (B, 1, D)
        # GT actions: actions from ctx_len-1 to ctx_len-1+horizon
        all_actions.append(actions[:, ctx_len-1:ctx_len-1+horizon].cpu())

        del pixels, actions, out, emb

    # Concatenate and save
    ctx_data = torch.cat(all_ctx, dim=0)       # (N, ctx_len, D)
    goal_data = torch.cat(all_goal, dim=0)     # (N, 1, D)
    action_data = torch.cat(all_actions, dim=0)  # (N, horizon, fs*raw_dim)

    out_path = Path(args.out)
    torch.save({
        "ctx_emb": ctx_data,
        "goal_emb": goal_data,
        "actions": action_data,
        "num_clips": len(indices),
        "ctx_len": ctx_len,
        "horizon": horizon,
        "action_dim": action_data.shape[-1],
    }, out_path)

    size_mb = out_path.stat().st_size / 1024 / 1024
    elapsed = time.time() - t0
    print(f"Saved: {out_path} ({size_mb:.0f} MB)")
    print(f"Time: {elapsed/60:.1f} min ({elapsed/len(indices)*1000:.1f} ms/clip)")


if __name__ == "__main__":
    main()
