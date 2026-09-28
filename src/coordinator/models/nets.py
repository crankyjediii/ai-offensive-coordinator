"""PyTorch networks: Model A (defensive sequence encoder) and Model B (multi-task outcome model)."""

from __future__ import annotations

import torch
from torch import nn

from coordinator.features.sequence import CAT_CARDINALITY, NUM_FIELDS

A_HEADS = {"box": 6, "rush": 5, "blitz": 2, "coverage": 8, "man": 2, "pressure": 2}
SIT_CARD = [4, 4, 4, 5]  # down, distance band, field zone, quarter (5 = overtime)
B_EVENTS = ["success", "explosive", "sack", "turnover", "first_down_or_td"]


class QueryEncoder(nn.Module):
    def __init__(self, n_num: int, d: int, emb: int = 8):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, emb) for c in SIT_CARD)
        self.proj = nn.Sequential(nn.Linear(emb * len(SIT_CARD) + n_num, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, sit_cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        e = [m(sit_cat[:, i]) for i, m in enumerate(self.embs)]
        return self.proj(torch.cat([*e, num], dim=-1))


class DefenseSequenceModel(nn.Module):
    """Model A. ``arch`` is 'transformer', 'pooled' (masked mean of snap tokens) or 'query_only'."""

    def __init__(self, n_query_num: int, d: int = 128, n_heads: int = 4, n_layers: int = 2, ff: int = 512,
                 dropout: float = 0.1, max_len: int = 128, arch: str = "transformer", use_position: bool = True,
                 emb: int = 8, prior_offset: bool = False):
        super().__init__()
        self.arch, self.use_position, self.max_len, self.prior_offset = arch, use_position, max_len, prior_offset
        self.tok_embs = nn.ModuleList(nn.Embedding(c, emb, padding_idx=0) for c in CAT_CARDINALITY)
        n_num = len(NUM_FIELDS)
        self.tok_proj = nn.Linear(emb * len(CAT_CARDINALITY) + 2 * n_num, d)
        self.query = QueryEncoder(n_query_num, d)
        self.pos = nn.Embedding(max_len + 1, d)
        self.type_emb = nn.Embedding(2, d)
        if arch == "transformer":
            layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(d)
        elif arch == "pooled":
            self.pool_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({k: nn.Linear(d, n) for k, n in A_HEADS.items()})
        if prior_offset:
            # Heads learn corrections to the smoothed prior: at initialization the output is the prior.
            for head in self.heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def tokens(self, cat: torch.Tensor, num: torch.Tensor, miss: torch.Tensor) -> torch.Tensor:
        e = [m(cat[..., i]) for i, m in enumerate(self.tok_embs)]
        return self.tok_proj(torch.cat([*e, num, miss.float()], dim=-1))

    def forward(self, hist_cat, hist_num, hist_miss, pad_mask, sit_cat, query_num,
                prior_logp: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        q = self.query(sit_cat, query_num)
        if self.arch == "query_only":
            h = q
        else:
            t = self.tokens(hist_cat, hist_num, hist_miss)
            b, length, _ = t.shape
            if self.arch == "pooled":
                keep = (~pad_mask).float().unsqueeze(-1)
                pooled = (t * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # zero-history rows -> zeros
                h = self.pool_mlp(torch.cat([q, pooled], dim=-1))
            else:
                x = torch.cat([t, q.unsqueeze(1)], dim=1)
                types = torch.cat([torch.zeros(b, length, dtype=torch.long, device=x.device),
                                   torch.ones(b, 1, dtype=torch.long, device=x.device)], dim=1)
                x = x + self.type_emb(types)
                if self.use_position:
                    pos = torch.arange(length + 1, device=x.device).flip(0).clamp(max=self.max_len)
                    x = x + self.pos(pos).unsqueeze(0)
                mask = torch.cat([pad_mask, torch.zeros(b, 1, dtype=torch.bool, device=x.device)], dim=1)
                out = self.encoder(self.drop(x), src_key_padding_mask=mask)
                h = self.final_norm(out[:, -1])
        out = {k: head(h) for k, head in self.heads.items()}
        if self.prior_offset:
            if prior_logp is None:
                raise ValueError("prior_offset model requires prior_logp")
            out = {k: v + prior_logp[k] for k, v in out.items()}
        return out


class OutcomeModel(nn.Module):
    """Model B: candidate-conditioned EPA mean and unconditional event probabilities."""

    def __init__(self, n_num: int, n_actions: int = 2, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.query = QueryEncoder(n_num + n_actions, hidden)
        self.body = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.epa_mean = nn.Linear(hidden, 1)
        self.epa_robust = nn.Linear(hidden, 1)
        self.events = nn.Linear(hidden, len(B_EVENTS))
        self.n_actions = n_actions

    def forward(self, sit_cat, num, action_idx) -> dict[str, torch.Tensor]:
        onehot = nn.functional.one_hot(action_idx, self.n_actions).float()
        h = self.body(self.query(sit_cat, torch.cat([num, onehot], dim=-1)))
        return {"epa_mean": self.epa_mean(h).squeeze(-1), "epa_robust": self.epa_robust(h).squeeze(-1),
                "event_logits": self.events(h)}
