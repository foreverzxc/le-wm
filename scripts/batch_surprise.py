"""Cross-dataset surprise: two tables — with-action and without-action.

With-action:  uses ground-truth actions (only where model.action_dim == dataset.action_dim)
Without-action: uses zero-vector actions (full 4×4 matrix, pure visual surprise)

Usage:
    python scripts/batch_surprise.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate

import stable_worldmodel as swm
from utils import get_img_preprocessor


MODELS = {
    "pusht":    {"weights": "pusht_weights.pt",    "config": "pusht_config.json"},
    "cube":     {"weights": "cube_weights.pt",     "config": "cube_config.json"},
    "reacher":  {"weights": "reacher_weights.pt",  "config": "reacher_config.json"},
    "tworooms": {"weights": "tworooms_weights.pt", "config": "tworooms_config.json"},
}

DATASETS = {
    "pusht":    {"h5": "pusht_expert_train",   "img": 224},
    "cube":     {"h5": "cube_single_expert",    "img": 224},
    "reacher":  {"h5": "reacher",               "img": 224},
    "tworooms": {"h5": "tworoom",               "img": 224},
}


def _remap_targets(cfg_dict):
    if isinstance(cfg_dict, dict):
        if "_target_" in cfg_dict:
            t = cfg_dict["_target_"]
            for old, new in [
                ("stable_worldmodel.wm.lewm.LeWM", "jepa.JEPA"),
                ("stable_worldmodel.wm.lewm.module.Predictor", "module.ARPredictor"),
                ("stable_worldmodel.wm.lewm.module.Embedder", "module.Embedder"),
                ("stable_worldmodel.wm.lewm.module.MLP", "module.MLP"),
            ]:
                t = t.replace(old, new)
            cfg_dict["_target_"] = t
        for v in cfg_dict.values():
            _remap_targets(v)
    return cfg_dict


def load_pretrained_model(cache_dir, model_name):
    info = MODELS[model_name]
    cfg_path = cache_dir / info["config"]
    weights_path = cache_dir / info["weights"]
    cfg_dict = _remap_targets(json.loads(cfg_path.read_text()))
    cfg = OmegaConf.create(cfg_dict)
    model = hydra.utils.instantiate(cfg)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()
    model.requires_grad_(False)
    return model


def episode_indices(dataset, ep_idx, max_steps):
    n = len(dataset)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        ep, _ = dataset.clip_indices[mid]
        if ep < ep_idx: lo = mid + 1
        else: hi = mid
    start_idx = lo
    lo, hi = start_idx, n
    while lo < hi:
        mid = (lo + hi) // 2
        ep, _ = dataset.clip_indices[mid]
        if ep <= ep_idx: lo = mid + 1
        else: hi = mid
    idxs = list(range(start_idx, lo))
    if len(idxs) > max_steps:
        step = len(idxs) // max_steps
        idxs = idxs[::step][:max_steps]
    return idxs


def compute_surprise(model, dataset, ep_idx, ctx_len, max_steps, use_real_action, model_act_dim):
    idxs = episode_indices(dataset, ep_idx, max_steps)
    errors = []
    for idx in idxs:
        try:
            item = dataset[idx]
        except Exception:
            continue

        pixels = item["pixels"].unsqueeze(0).cuda()
        B, T = pixels.shape[:2]

        if use_real_action and "action" in item:
            action = item["action"].unsqueeze(0).cuda()
            if action.shape[-1] != model_act_dim:
                continue  # dim mismatch, skip
        else:
            action = torch.zeros(B, T, model_act_dim, device="cuda")

        with torch.no_grad():
            out = model.encode({"pixels": pixels, "action": action})
            emb = out["emb"]
            act_emb = out["act_emb"]
            ctx_emb = emb[:, :ctx_len]
            ctx_act = act_emb[:, :ctx_len]
            pred = model.predict(ctx_emb, ctx_act)
            tgt = emb[:, 1:ctx_len+1]
            err = (pred - tgt).pow(2).mean().item()
        errors.append(err)

    return np.array(errors) if errors else np.array([float("nan")])


def main():
    cache_dir = Path(swm.data.utils.get_cache_dir())

    all_models = list(MODELS.keys())
    all_datasets = list(DATASETS.keys())

    # Pre-load models, get action dims
    print("Loading models...")
    loaded = {}
    model_act_dims = {}
    for m in all_models:
        loaded[m] = load_pretrained_model(cache_dir, m)
        model_act_dims[m] = loaded[m].action_encoder.patch_embed.in_channels
        print(f"  {m}: action_dim={model_act_dims[m]}")

    # Get dataset action dims
    ds_act_dims = {}
    for ds_name in all_datasets:
        ds_info = DATASETS[ds_name]
        transform = get_img_preprocessor("pixels", "pixels", ds_info["img"])
        ds = swm.data.HDF5Dataset(ds_info["h5"], frameskip=5, num_steps=4,
                                   transform=transform)
        ds_act_dims[ds_name] = ds.get_dim("action") * ds.frameskip
        print(f"  dataset {ds_name}: action_dim={ds_act_dims[ds_name]}")

    # ── Compute both matrices ──
    ctx_len = 3
    max_steps = 100
    n_test = 5

    results_action = {}    # (model, dataset) -> mean
    results_noaction = {}  # (model, dataset) -> mean

    for ds_name in all_datasets:
        ds_info = DATASETS[ds_name]
        transform = get_img_preprocessor("pixels", "pixels", ds_info["img"])
        ds = swm.data.HDF5Dataset(ds_info["h5"], frameskip=5, num_steps=4,
                                   transform=transform)
        n_ep = min(n_test, len(ds.lengths))

        for m_name in all_models:
            model = loaded[m_name]
            m_ad = model_act_dims[m_name]
            d_ad = ds_act_dims[ds_name]
            dims_match = (m_ad == d_ad)

            # ── With-action (only if dims match) ──
            if dims_match:
                means = []
                for ep in range(n_ep):
                    errs = compute_surprise(model, ds, ep, ctx_len, max_steps, use_real_action=True, model_act_dim=m_ad)
                    if len(errs) > 0 and not np.isnan(errs[0]):
                        means.append(errs.mean())
                results_action[(m_name, ds_name)] = np.mean(means) if means else None

            # ── Without-action (always) ──
            means = []
            for ep in range(n_ep):
                errs = compute_surprise(model, ds, ep, ctx_len, max_steps, use_real_action=False, model_act_dim=m_ad)
                if len(errs) > 0 and not np.isnan(errs[0]):
                    means.append(errs.mean())
            results_noaction[(m_name, ds_name)] = np.mean(means) if means else None

            dim_tag = "✓" if dims_match else "✗"
            a_val = f"{results_action.get((m_name, ds_name), 0):.4f}" if dims_match else "·"
            n_val = f"{results_noaction[(m_name, ds_name)]:.4f}"
            print(f"  {m_name:10s} × {ds_name:10s}  dims={dim_tag}  with-action={a_val}  no-action={n_val}")

    # ═══════════════════════════════════════════════════════════════════
    #  TABLE 1: With-action (only matching dims)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("TABLE 1: SURPRISE WITH GROUND-TRUTH ACTIONS (· = dim mismatch)")
    print(f"{'='*90}")
    headers = ["Model \\ Dataset"] + [d.upper() for d in all_datasets]
    rows = []
    for m in all_models:
        row = [m.upper()]
        for d in all_datasets:
            v = results_action.get((m, d))
            row.append(f"{v:.4f}" if v is not None else "·")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # ═══════════════════════════════════════════════════════════════════
    #  TABLE 2: Without-action (full matrix, visual surprise)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("TABLE 2: SURPRISE WITH ZERO-ACTIONS (visual-only, full matrix)")
    print(f"{'='*90}")
    headers2 = ["Model \\ Dataset"] + [d.upper() for d in all_datasets] + ["AVG"]
    rows2 = []
    col_sums = {d: [] for d in all_datasets}
    for m in all_models:
        row = [m.upper()]
        rv = []
        for d in all_datasets:
            v = results_noaction.get((m, d), 0)
            row.append(f"{v:.4f}")
            rv.append(v)
            col_sums[d].append(v)
        row.append(f"{np.mean(rv):.4f}")
        rows2.append(row)

    # Complexity row
    crow = ["COMPLEXITY"]
    for d in all_datasets:
        crow.append(f"{np.mean(col_sums[d]):.4f}")
    crow.append("")
    rows2.append(crow)
    print(tabulate(rows2, headers=headers2, tablefmt="grid"))

    # ═══════════════════════════════════════════════════════════════════
    #  Analysis
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("ANALYSIS")
    print(f"{'='*90}")

    # Dataset complexity (from no-action table)
    print("\nDataset complexity (avg surprise across all models, higher = visually harder):")
    ranking = sorted([(d, np.mean(col_sums[d])) for d in all_datasets],
                     key=lambda x: -x[1])
    for i, (ds, score) in enumerate(ranking):
        print(f"  {i+1}. {ds.upper():12s}  {score:.4f}")

    # Model generalization (avg off-diagonal surprise, no-action)
    print("\nModel generalization (avg off-diag no-action surprise, lower = better):")
    for m in all_models:
        off = [results_noaction[(m, d)] for d in all_datasets if d != m]
        diag = results_noaction.get((m, m), 0)
        print(f"  {m.upper():12s}  diag={diag:.4f}  off-diag avg={np.mean(off):.4f}  gap={np.mean(off)-diag:+.4f}")

    # Action contribution (with - without for diagonal)
    print("\nAction contribution on diagonal (no-action - with-action, higher = actions matter more):")
    for m in all_datasets:
        wa = results_action.get((m, m))
        na = results_noaction.get((m, m))
        if wa is not None and na is not None:
            print(f"  {m.upper():12s}  with-action={wa:.4f}  no-action={na:.4f}  Δ={na-wa:+.4f}")

    # CSV
    print(f"\nCSV (no-action matrix):")
    print(",".join(headers2))
    for r in rows2:
        print(",".join(str(x) for x in r))


if __name__ == "__main__":
    main()
