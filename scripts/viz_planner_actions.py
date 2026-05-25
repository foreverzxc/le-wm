"""Visualize planner action sequences: each query's trajectory in action space.

Usage:
    python scripts/viz_planner_actions.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra, numpy as np, torch
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")
from omegaconf import OmegaConf
import stable_worldmodel as swm
from planner import PlannerDecoder, planner_rollout
from utils import get_img_preprocessor


def load_wm():
    cache = Path(swm.data.utils.get_cache_dir())
    def remap(d):
        if isinstance(d, dict):
            if '_target_' in d:
                t = d['_target_']
                for o, n in [('stable_worldmodel.wm.lewm.LeWM','jepa.JEPA'),
                    ('stable_worldmodel.wm.lewm.module.Predictor','module.ARPredictor'),
                    ('stable_worldmodel.wm.lewm.module.Embedder','module.Embedder'),
                    ('stable_worldmodel.wm.lewm.module.MLP','module.MLP')]: t = t.replace(o, n)
                d['_target_'] = t
            for v in d.values(): remap(v)
        return d
    cfg = OmegaConf.create(remap(json.loads((cache/'pusht_config.json').read_text())))
    wm = hydra.utils.instantiate(cfg)
    sd = torch.load(cache/'pusht_weights.pt', map_location='cpu', weights_only=True)
    sd = {k.replace('_orig_mod.',''): v for k,v in sd.items()}
    wm.load_state_dict(sd, strict=False)
    wm = wm.cuda().eval()
    for p in wm.parameters(): p.requires_grad_(False)
    return wm


def main():
    wm = load_wm()

    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                               transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices[:30])

    N, T = 8, 5
    planner = PlannerDecoder(embed_dim=192, num_queries=N, horizon=T,
                              action_dim=2, num_layers=3).cuda()

    # Take 4 different starting positions in the episode
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for sample_idx in range(8):
        item = ds[sample_idx * 3]  # spaced samples
        batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
                 if torch.is_tensor(v)}

        ctx_len = 3
        with torch.no_grad():
            out = wm.encode(batch)
            ctx_emb = out["emb"][:, :ctx_len]
            goal_emb = out["emb"][:, -1:]

        # Planner forward
        actions = planner(ctx_emb[:, -1:], goal_emb).detach()  # (1, N, T, 2)

        # Rollout costs
        info = {"pixels": batch["pixels"][:, :ctx_len]}
        with torch.no_grad():
            pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len)
            goal = goal_emb.expand(-1, N, -1)
            costs = (pred_embs - goal).pow(2).mean(dim=-1)[0]

        # Ground truth
        gt_raw = batch["action"][:, 1:1+T].detach()  # (1, T, 10)
        T_gt = gt_raw.shape[1]
        gt_actions = gt_raw.reshape(-1, T_gt, 5, 2).mean(2)[0]  # (T_gt, 2)
        gt_cum = np.cumsum(gt_actions.detach().cpu().numpy(), axis=0)

        ax = axes[sample_idx]
        acts = actions[0].detach().cpu().numpy()  # (N, T, 2)

        # Plot each query's action trajectory
        best_idx = costs.argmin().item()
        cmap = plt.cm.tab10

        for i in range(N):
            # Cumulative sum = position in action space
            cum = np.cumsum(acts[i], axis=0)
            alpha = 0.9 if i == best_idx else 0.35
            lw = 2.5 if i == best_idx else 1.0
            color = cmap(i % 10)
            ax.plot(cum[:, 0], cum[:, 1], 'o-', color=color,
                    linewidth=lw, alpha=alpha, markersize=5,
                    label=f'Q{i}' if i == best_idx else None)
            # Start marker
            ax.scatter([cum[0, 0]], [cum[0, 1]], color=color, s=30, alpha=alpha, marker='s')
            # End marker
            ax.scatter([cum[-1, 0]], [cum[-1, 1]], color=color, s=50, alpha=alpha, marker='*')

        # Ground truth trajectory (if available)
        if T_gt >= 2:
            ax.plot(gt_cum[:min(T, T_gt), 0], gt_cum[:min(T, T_gt), 1],
                    's--', color='black', linewidth=2.5, markersize=6, label='GT',
                    zorder=10)

        ax.set_xlabel('Action x (cumulative)')
        ax.set_ylabel('Action y (cumulative)')
        ax.set_title(f'Step {sample_idx*3}  |  best=Q{best_idx} (cost={costs[best_idx]:.3f})'
                     f'\nplanner min={costs.min():.3f}  max={costs.max():.3f}')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
        if sample_idx == 0:
            ax.legend(fontsize=6, loc='best')

    plt.suptitle(f'Planner Action Sequences — {N} queries × {T} steps each\n'
                 f'Star = final position, Square = start, ★ = best query (lowest cost)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = Path(__file__).parent.parent / "output" / "planner_actions.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
