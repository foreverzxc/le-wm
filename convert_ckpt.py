#!/usr/bin/env python3
"""Convert .pt weight files to .ckpt model checkpoints.

Scans the data/ directory for *_weights.pt files and corresponding *_config.json
files, builds the JEPA model, loads weights, and saves the full model object
as lewm_object.ckpt in the appropriate subdirectory.

Usage:
    python convert_ckpt.py              # Convert all models in data/
    python convert_ckpt.py --name pusht # Convert specific model only
"""

import argparse
import json
import sys
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def strip_hydra(d):
    """Remove Hydra-specific keys (starting with '_') from config dict."""
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            v = strip_hydra(v)
        out[k] = v
    return out


def find_models(data_dir: Path):
    """Find all model pairs (*_weights.pt + *_config.json) in data_dir."""
    weight_files = sorted(data_dir.glob("*_weights.pt"))
    models = []
    for wf in weight_files:
        name = wf.stem.replace("_weights", "")
        config_file = data_dir / f"{name}_config.json"
        if config_file.exists():
            models.append((name, wf, config_file))
        else:
            print(f"WARNING: Config file not found for {wf.name}: {config_file}")
    return models


def convert_model(name: str, weights_path: Path, config_path: Path, data_dir: Path):
    """Convert a single .pt model to .ckpt format.

    Args:
        name: Model name (e.g., 'pusht', 'cube', 'reacher', 'tworooms')
        weights_path: Path to the *_weights.pt file
        config_path: Path to the *_config.json file
        data_dir: Base data directory for output
    """
    print(f"\n{'='*60}")
    print(f"Converting: {name}")
    print(f"  Config:  {config_path}")
    print(f"  Weights: {weights_path}")
    print(f"{'='*60}")

    # Load config
    cfg = json.loads(config_path.read_text())
    print("  Config loaded.")

    # Build encoder
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    # MLP factory
    mlp = lambda k: MLP(
        input_dim=cfg[k]["input_dim"],
        output_dim=cfg[k]["output_dim"],
        hidden_dim=cfg[k]["hidden_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )

    # Strip hydra keys from nested configs
    predictor_cfg = strip_hydra(cfg["predictor"])
    action_enc_cfg = strip_hydra(cfg["action_encoder"])

    print(f"  Predictor config: {predictor_cfg}")
    print(f"  Action encoder config: {action_enc_cfg}")

    # Build full JEPA model
    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**predictor_cfg),
        action_encoder=Embedder(**action_enc_cfg),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )

    # Load weights
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    print(f"  Loaded state dict with {len(sd)} keys")

    result = model.load_state_dict(sd, strict=True)
    print(f"  Load result: {result}")

    # Save checkpoint
    out_dir = data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lewm_object.ckpt"

    torch.save(model, out_path)
    print(f"  Saved checkpoint to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert .pt models to .ckpt checkpoints")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory containing *_weights.pt and *_config.json files. "
             "Defaults to the project's data/ directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Convert only the model with this name (e.g., pusht, cube, reacher, tworooms). "
             "If not specified, all models are converted.",
    )
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Find models to convert
    if args.name:
        # Convert specific model
        weights_path = data_dir / f"{args.name}_weights.pt"
        config_path = data_dir / f"{args.name}_config.json"
        if not weights_path.exists():
            print(f"ERROR: Weights file not found: {weights_path}")
            sys.exit(1)
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        models = [(args.name, weights_path, config_path)]
    else:
        # Convert all models
        models = find_models(data_dir)

    if not models:
        print("No models found to convert.")
        sys.exit(0)

    print(f"Found {len(models)} model(s) to convert:")
    for name, wp, cp in models:
        print(f"  - {name}: {wp.name} + {cp.name}")

    # Convert each model
    success = 0
    for name, weights_path, config_path in models:
        try:
            convert_model(name, weights_path, config_path, data_dir)
            success += 1
        except Exception as e:
            print(f"  Failed to convert {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Conversion complete: {success}/{len(models)} model(s) converted successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()