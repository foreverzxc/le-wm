"""Visualize planner outputs: action sequences, rollout embeddings, cost distribution.

Usage:
    python scripts/viz_planner.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra, numpy as np, torch
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("Agg")
from omegaconf import OmegaConf
import stable_worldmodel as swm
from planner import PlannerDecoder, PlannerLoss, planner_rollout
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

    # Dataset: 1 Pusht episode
    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds = swm.data.HDF5Dataset("pusht_expert_train", frameskip=5, num_steps=4,
                               transform=transform)
    indices = [i for i in range(len(ds)) if ds.clip_indices[i][0] == 0]
    ds = torch.utils.data.Subset(ds, indices[:20])

    # Planner (untrained)
    N, T = 8, 5
    planner = PlannerDecoder(embed_dim=192, num_queries=N, horizon=T,
                              action_dim=2, num_layers=3).cuda()

    # Get a sample
    item = ds[10]
    batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()
             if torch.is_tensor(v)}

    ctx_len = 3
    with torch.no_grad():
        out = wm.encode(batch)
        ctx_emb = out["emb"][:, :ctx_len]
        goal_emb = out["emb"][:, -1:]  # last frame as goal

    # Planner forward
    actions = planner(ctx_emb[:, -1:], goal_emb)  # (1, N, T, 2)

    # Rollout
    info = {"pixels": batch["pixels"][:, :ctx_len]}
    pred_embs, _ = planner_rollout(wm, actions, info, history_size=ctx_len)

    # Random actions for comparison
    rand_actions = torch.randn(1, N, T, 2, device='cuda') * 2.0
    rand_embs, _ = planner_rollout(wm, rand_actions, info, history_size=ctx_len)

    # Costs
    goal = goal_emb.expand(-1, N, -1)
    planner_costs = (pred_embs - goal).pow(2).mean(dim=-1)[0].detach().cpu().numpy()
    rand_costs = (rand_embs - goal).pow(2).mean(dim=-1)[0].detach().cpu().numpy()
    best_idx = planner_costs.argmin()

    # Ground truth action for comparison
    gt_actions_raw = batch["action"][:, 1:].detach().cpu().numpy()  # (1, steps, 10)
    T_gt = gt_actions_raw.shape[1]
    gt_actions = gt_actions_raw.reshape(-1, T_gt, 5, 2).mean(2)  # (1, T_gt, 2)

    # ── Plots ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Action sequences (first 2 queries)
    ax = axes[0, 0]
    acts = actions[0].cpu().detach().numpy()  # (N, T, 2)
    for i in range(min(4, N)):
        alpha = 1.0 if i == best_idx else 0.3
        ax.plot(acts[i, :, 0], acts[i, :, 1], 'o-', alpha=alpha,
                label=f'Q{i} {"★" if i==best_idx else ""}')
    if gt_actions is not None:
        ax.plot(gt_actions[0, :, 0], gt_actions[0, :, 1], 's--', color='black',
                linewidth=2, label='GT')
    ax.set_xlabel('Action dim 0 (x)')
    ax.set_ylabel('Action dim 1 (y)')
    ax.set_title(f'Action sequences (best=Q{best_idx})')
    ax.legend(fontsize=7)

    # 2. Cost comparison: planner vs random
    ax = axes[0, 1]
    x = np.arange(N)
    w = 0.35
    ax.bar(x - w/2, planner_costs, w, label='Planner', alpha=0.8)
    ax.bar(x + w/2, rand_costs, w, label='Random', alpha=0.8)
    ax.set_xlabel('Query index')
    ax.set_ylabel('MSE to goal')
    ax.set_title(f'Cost: planner vs random\n'
                 f'(planner min={planner_costs.min():.4f}, rand min={rand_costs.min():.4f})')
    ax.legend(fontsize=7)

    # 3. Embedding space: predicted vs goal (2D PCA)
    ax = axes[0, 2]
    all_embs = torch.cat([pred_embs[0], rand_embs[0], goal_emb[0]]).detach().cpu().numpy()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(all_embs)
    pred_2d = pca.transform(pred_embs[0].detach().cpu().numpy())
    rand_2d = pca.transform(rand_embs[0].detach().cpu().numpy())
    goal_2d = pca.transform(goal_emb[0].detach().cpu().numpy())
    ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c='blue', label='Planner', s=60)
    ax.scatter(rand_2d[:, 0], rand_2d[:, 1], c='gray', label='Random', s=30, alpha=0.5)
    ax.scatter(goal_2d[:1, 0], goal_2d[:1, 1], c='red', marker='*', s=200, label='Goal')
    ax.scatter(pred_2d[best_idx:best_idx+1, 0], pred_2d[best_idx:best_idx+1, 1],
               c='cyan', marker='o', s=120, label='Best planner')
    for i in range(N):
        ax.plot([pred_2d[i, 0], goal_2d[0, 0]], [pred_2d[i, 1], goal_2d[0, 1]],
                'blue', alpha=0.15, linewidth=0.5)
    ax.set_title('Embedding space (PCA)')
    ax.legend(fontsize=6, loc='lower left')

    # 4. Action dimension distribution
    ax = axes[1, 0]
    ax.hist(acts[:, :, 0].flatten(), bins=20, alpha=0.5, label='dim 0 (x)', color='blue')
    ax.hist(acts[:, :, 1].flatten(), bins=20, alpha=0.5, label='dim 1 (y)', color='orange')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Action value')
    ax.set_title('Action value distribution (all queries)')
    ax.legend(fontsize=7)

    # 5. Per-step action evolution
    ax = axes[1, 1]
    for i in range(min(4, N)):
        ax.plot(range(T), acts[i, :, 0], 'o-', alpha=0.5, label=f'Q{i} x' if i==0 else '')
        ax.plot(range(T), acts[i, :, 1], 's-', alpha=0.5, label=f'Q{i} y' if i==0 else '')
    ax.set_xlabel('Step t')
    ax.set_ylabel('Action')
    ax.set_title('Action over planning horizon')
    ax.legend(fontsize=7)

    # 6. Query diversity
    ax = axes[1, 2]
    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_matrix[i, j] = np.dot(acts[i].flatten(), acts[j].flatten()) / (
                np.linalg.norm(acts[i]) * np.linalg.norm(acts[j]) + 1e-8)
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xlabel('Query j')
    ax.set_ylabel('Query i')
    ax.set_title(f'Query cosine similarity\n(mean off-diag={sim_matrix[~np.eye(N,dtype=bool)].mean():.3f})')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = Path(__file__).parent.parent / "output" / "planner_viz.png"
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    print(f"\nSummary:")
    print(f"  Planner best cost:  {planner_costs.min():.4f} (query {best_idx})")
    print(f"  Random best cost:   {rand_costs.min():.4f}")
    print(f"  Planner mean cost:  {planner_costs.mean():.4f}")
    print(f"  Random mean cost:   {rand_costs.mean():.4f}")
    print(f"  Query diversity:    {sim_matrix[~np.eye(N,dtype=bool)].mean():.3f}")


if __name__ == "__main__":
    main()
