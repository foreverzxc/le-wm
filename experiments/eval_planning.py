#!/usr/bin/env python3
"""下游规划评估：白化微调模型 vs 预训练基线。

用法:
    # 评估所有可用模型
    python experiments/eval_planning.py
    # 指定具体模型目录
    python experiments/eval_planning.py --model-dir ~/.stable_worldmodel/experiments/finetune_whitening
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import stable_worldmodel as swm
import stable_pretraining as spt
from omegaconf import OmegaConf


def evaluate_model(model_dir, label, n_episodes=50, seed=42, horizon=5):
    """Run MPC planning evaluation on PushT."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {label}")
    print(f"  Model: {model_dir}")
    print(f"{'='*70}")

    # Load model via AutoCostModel
    model = swm.policy.AutoCostModel(str(model_dir))
    model = model.to("cuda")
    model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # Set up preprocessors
    process = {}
    dataset = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=str(Path.home() / ".stable_worldmodel"),
    )
    from sklearn import preprocessing
    for col in ["action", "proprio", "state"]:
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler

    # Image transform
    from torchvision.transforms import v2 as transforms
    transform = {
        "pixels": transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]),
        "goal": transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]),
    }

    # Configure solver
    config = swm.PlanConfig(
        horizon=horizon,
        receding_horizon=horizon,
        action_block=5,
        history_len=1,
        warm_start=True,
    )
    solver = swm.solver.CEMSolver(
        model=model, device="cuda",
        num_samples=500, n_steps=3, topk=25,
    )

    # Create world model policy
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )

    # Create environment
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=n_episodes,
        max_episode_steps=2 * 50,
        image_shape=(224, 224),
    )

    # Sample evaluation episodes from dataset
    col_name = "episode_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = _get_episode_lengths(dataset, ep_indices)
    max_start = episode_len - 25 - 1
    valid_mask = dataset.get_col_data("step_idx") <= np.array(
        [{ep: ms for ep, ms in zip(ep_indices, max_start)}[ep]
         for ep in dataset.get_col_data(col_name)]
    )
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"  Valid starting points: {len(valid_indices)}")

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(valid_indices), n_episodes, replace=False)
    selected = np.sort(valid_indices[selected])

    start_steps = dataset.get_row_data(selected)["step_idx"].tolist()
    episodes_idx = dataset.get_row_data(selected)[col_name].tolist()

    # Run evaluation
    world.set_policy(policy)
    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=start_steps,
        goal_offset_steps=25,
        eval_budget=50,
        episodes_idx=episodes_idx,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    )
    elapsed = time.time() - start_time

    print(f"\n  Results ({label}):")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    print(f"  Time: {elapsed:.1f}s")
    return metrics


def _get_episode_lengths(dataset, episodes):
    ep_idx = dataset.get_col_data("episode_idx")
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[ep_idx == ep_id]) + 1)
    return np.array(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    base = Path.home() / ".stable_worldmodel" / "experiments"
    pretrained_dir = Path.home() / ".stable_worldmodel" / "pusht"

    if args.model_dir:
        dirs = [(args.model_dir, "Custom model")]
    else:
        dirs = [
            (str(pretrained_dir), "Pre-trained (SIGReg)"),
        ]
        if (base / "finetune_whitening").exists():
            dirs.append((str(base / "finetune_whitening"), "Whitening fine-tuned"))

    all_results = {}
    for d, label in dirs:
        try:
            m = evaluate_model(d, label, n_episodes=args.n_episodes, horizon=args.horizon)
            all_results[label] = m
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for label, m in all_results.items():
        sr = m.get("success_rate", "N/A")
        print(f"  {label:40s}: success_rate={sr}")


if __name__ == "__main__":
    main()
