"""Planner: action-sequence generator on top of frozen WM.

Given context + goal images, outputs N candidate action sequences via learned
queries (Perceiver-style self-attention). Best candidate is selected by
rollout cost. Diversity loss pushes non-best queries away from the best one
(DETR-style regularisation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PlannerDecoder(nn.Module):
    """Self-attention decoder over [goal_emb, ctx_emb, learnable_queries].

    Args:
        embed_dim: WM embedding dimension (192).
        num_queries: Number of action-plan candidates (N).
        num_layers: Transformer decoder layers.
        num_heads: Attention heads.
        mlp_dim: FFN hidden dimension.
        horizon: Number of future steps to plan.
        action_dim: Raw action dimension per step.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_queries: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        horizon: int = 5,
        action_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.horizon = horizon
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Learnable query tokens
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)

        # Input projection for goal + ctx (already embed_dim, optional refine)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Transformer decoder (self-attention only — no cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head: query embedding → action sequence
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, horizon * action_dim),
        )

    def forward(self, ctx_emb: torch.Tensor, goal_emb: torch.Tensor):
        """Produce N action sequences.

        Args:
            ctx_emb:  (B, 1, D)  context embedding (encoded from pixels)
            goal_emb: (B, 1, D)  goal embedding

        Returns:
            actions: (B, N, horizon * action_dim)  raw action sequences
        """
        B = ctx_emb.size(0)
        D = self.embed_dim

        # Build input sequence: [goal, ctx, queries]
        goal = self.input_proj(goal_emb)       # (B, 1, D)
        ctx = self.input_proj(ctx_emb)          # (B, 1, D)
        queries = self.query_embed.expand(B, -1, -1)  # (B, N, D)

        # Triplet input: concat along sequence dimension
        decoder_input = torch.cat([goal, ctx, queries], dim=1)  # (B, 2+N, D)

        # Self-attention (no separate memory — queries attend to goal and ctx)
        out = self.decoder(decoder_input, decoder_input)  # (B, 2+N, D)

        # Extract query outputs
        query_out = out[:, 2:, :]  # (B, N, D)

        # Predict action sequences
        actions = self.action_head(query_out)  # (B, N, horizon * action_dim)
        actions = actions.reshape(B, self.num_queries, self.horizon, self.action_dim)

        return actions


class PlannerLoss(nn.Module):
    """Best-of-N + diversity loss.

    - ``cost``: MSE between rollout embedding and goal embedding, per query.
    - ``loss``: cost of the *best* query (min cost).
    - ``diversity``: pushes non-best action sequences away from the best one.
    """

    def __init__(self, diversity_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight

    def forward(
        self,
        action_seqs: torch.Tensor,  # (B, N, T, action_dim)
        pred_embs: torch.Tensor,    # (B, N, D)  — final rollout embeddings
        goal_emb: torch.Tensor,     # (B, 1, D)  — goal embedding
    ):
        """
        Returns:
            loss: scalar
            info: dict with cost, best_idx, diversity term
        """
        B, N, D = pred_embs.shape
        goal = goal_emb.expand(-1, N, -1)        # (B, N, D)

        # Per-query cost
        costs = (pred_embs - goal).pow(2).mean(dim=-1)  # (B, N)

        # Best query per sample
        best_cost, best_idx = costs.min(dim=-1)  # (B,), (B,)

        # ── Diversity loss (DETR-style) ──
        # Penalise cosine similarity between non-best and best action sequences.
        best_actions = action_seqs[torch.arange(B), best_idx]  # (B, T, A)
        div_loss = torch.tensor(0.0, device=action_seqs.device)
        count = 0
        for i in range(N):
            mask = (best_idx != i)
            if mask.any():
                other = action_seqs[mask, i]               # (B', T, A)
                best_for_those = best_actions[mask]         # (B', T, A)
                # Flatten action sequence for similarity
                sim = F.cosine_similarity(
                    other.reshape(other.size(0), -1),
                    best_for_those.reshape(best_for_those.size(0), -1),
                    dim=-1,
                )  # (B',)
                # Push towards dissimilarity (penalise positive sim)
                div_loss += F.relu(sim - 0.0).mean()
                count += 1
        if count > 0:
            div_loss = div_loss / count

        loss = best_cost.mean() + self.diversity_weight * div_loss

        info = {
            "loss": loss.detach(),
            "best_cost": best_cost.mean().detach(),
            "cost_mean": costs.mean().detach(),
            "cost_std": costs.std(dim=-1).mean().detach(),
            "diversity": div_loss.detach(),
        }
        return loss, info


def planner_rollout(wm, actions, info_dict, history_size=3):
    """Run WM rollout using planner actions across N candidates.

    Args:
        wm: frozen JEPA model (encoder + predictor + action_encoder)
        actions: (B, N, T, raw_action_dim)  from planner
        info_dict: dict with ``pixels`` (B, T, C, H, W) and optional ``goal``
        history_size: WM context length

    Returns:
        pred_embs: (B, N, D)  final step embedding for each candidate
    """
    B, N, T, _ = actions.shape

    # Encode context
    with torch.no_grad():
        out = wm.encode({k: v for k, v in info_dict.items()
                          if torch.is_tensor(v) and k != "goal"})
        ctx_emb = out["emb"]  # (B, T_ctx, D)

        # Goal embedding
        if "goal" in info_dict:
            goal_img = info_dict["goal"]
            if goal_img.ndim == 4:
                goal_img = goal_img.unsqueeze(1)
            goal_out = wm.encode({"pixels": goal_img[:, :1]})
            goal_emb = goal_out["emb"][:, -1:]  # (B, 1, D)
        else:
            goal_emb = ctx_emb[:, -1:]

        # Expand embeddings for N candidates
        ctx_emb_exp = ctx_emb.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, T_ctx, D)
        emb = rearrange(ctx_emb_exp, "b n t d -> (b n) t d")
        emb = emb.clone()

        # Rollout
        HS = history_size
        act_raw = rearrange(actions, "b n t d -> (b n) t d")

        for t in range(T):
            act_full = torch.cat([
                torch.zeros(B * N, HS, actions.shape[-1], device=actions.device),
                act_raw,
            ], dim=1)
            act_emb = wm.action_encoder(act_full[:, :HS + t + 1])
            emb_trunc = emb[:, -(HS + t):]
            act_trunc = act_emb[:, -(HS + t):]

            # Predict next step
            pred = wm.predict(emb_trunc[:, :HS], act_trunc[:, :HS])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)

        # Take the final step embedding
        final_emb = emb[:, -1]  # (B*N, D)
        pred_embs = rearrange(final_emb, "(b n) d -> b n d", b=B, n=N)

    return pred_embs, goal_emb
