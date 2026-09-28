"""Training loops for Model A and Model B with masked multi-task losses and early stopping."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from coordinator.features.sequence import HistoryBatch
from coordinator.models.nets import A_HEADS, B_EVENTS, DefenseSequenceModel, OutcomeModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


@dataclass
class FitLog:
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")
    seconds: float = 0.0
    device: str = ""
    n_parameters: int = 0


# ---------------------------------------------------------------- Model A


@dataclass
class AData:
    ctx_idx: np.ndarray
    sit_cat: np.ndarray
    query_num: np.ndarray
    targets: dict[str, np.ndarray]
    prior_logp: dict[str, np.ndarray] | None = None


def masked_multitask_ce(out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                        weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """L_A = sum_j lambda_j [sum_i m_ij CE / max(1, sum_i m_ij)]; empty heads contribute zero."""
    total = out[next(iter(out))].new_zeros(())
    parts = {}
    for h, logits in out.items():
        y = targets[h]
        m = y >= 0
        if m.any():
            loss = nn.functional.cross_entropy(logits[m], y[m], reduction="sum") / m.sum().clamp(min=1)
            total = total + (weights or {}).get(h, 1.0) * loss
            parts[h] = float(loss.detach())
    return total, parts


class HistoryStore:
    """History tensors on device, gathered per batch by context index."""

    def __init__(self, hist: HistoryBatch, device: torch.device):
        self.cat = torch.as_tensor(hist.cat, device=device)
        self.num = torch.as_tensor(hist.num, device=device)
        self.miss = torch.as_tensor(hist.num_missing, device=device)
        self.pad = torch.as_tensor(hist.padding_mask, device=device)

    def gather(self, idx: torch.Tensor, shuffle_order: bool = False):
        cat, num, miss, pad = self.cat[idx], self.num[idx], self.miss[idx], self.pad[idx]
        if shuffle_order:  # ablation: permute real tokens within each prefix
            perm = torch.argsort(torch.rand(pad.shape, device=pad.device) + pad.float() * -2.0, dim=1)
            gather = perm.unsqueeze(-1)
            cat = torch.gather(cat, 1, gather.expand_as(cat))
            num = torch.gather(num, 1, gather.expand_as(num))
            miss = torch.gather(miss, 1, gather.expand_as(miss))
            pad = torch.gather(pad, 1, perm)
        return cat, num, miss, pad


def _a_batches(data: AData, batch: int, shuffle: bool, rng: np.random.Generator):
    n = len(data.ctx_idx)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch):
        yield order[s:s + batch]


def _a_tensors(data: AData, ix: np.ndarray, device: torch.device):
    prior = None if data.prior_logp is None else {h: torch.as_tensor(data.prior_logp[h][ix], device=device) for h in A_HEADS}
    targets = {h: torch.as_tensor(data.targets[h][ix], device=device) for h in A_HEADS} if data.targets else {}
    return (
        torch.as_tensor(data.ctx_idx[ix], device=device),
        torch.as_tensor(data.sit_cat[ix], device=device),
        torch.as_tensor(data.query_num[ix], device=device),
        targets,
        prior,
    )


def predict_a_logits(model: DefenseSequenceModel, store: HistoryStore, data: AData, device: torch.device,
                     batch: int = 1024, shuffle_order: bool = False) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {h: [] for h in A_HEADS}
    with torch.inference_mode():
        for ix in _a_batches(data, batch, False, np.random.default_rng(0)):
            ci, sc, qn, _, pr = _a_tensors(data, ix, device)
            out = model(*store.gather(ci, shuffle_order), sc, qn, pr)
            for h in A_HEADS:
                outs[h].append(out[h].float().cpu().numpy())
    return {h: np.concatenate(v) for h, v in outs.items()}


def fit_model_a(train: AData, val: AData, hist: HistoryBatch, seed: int, arch: str = "transformer",
                use_position: bool = True, shuffle_order: bool = False, max_epochs: int = 12, patience: int = 3,
                lr: float = 7e-4, batch: int = 512, d: int = 128, n_layers: int = 2, n_heads: int = 4,
                history_length: int = 128, prior_offset: bool = False) -> tuple[DefenseSequenceModel, FitLog]:
    seed_everything(seed)
    device = train_device()
    model = DefenseSequenceModel(train.query_num.shape[1], d=d, n_layers=n_layers, n_heads=n_heads,
                                 arch=arch, use_position=use_position, max_len=history_length,
                                 prior_offset=prior_offset).to(device)
    store = HistoryStore(hist, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max_epochs * int(np.ceil(len(train.ctx_idx) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    log = FitLog(device=str(device), n_parameters=sum(p.numel() for p in model.parameters()))
    rng = np.random.default_rng(seed)
    best_state, bad = None, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        tr_loss, nb = 0.0, 0
        for ix in _a_batches(train, batch, True, rng):
            ci, sc, qn, tg, pr = _a_tensors(train, ix, device)
            out = model(*store.gather(ci, shuffle_order), sc, qn, pr)
            loss, _ = masked_multitask_ce(out, tg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_loss += float(loss.detach())
            nb += 1
        logits = predict_a_logits(model, store, val, device, shuffle_order=shuffle_order)
        val_parts = {}
        for h in A_HEADS:
            y = val.targets[h]
            m = y >= 0
            if m.any():
                z = torch.as_tensor(logits[h][m])
                val_parts[h] = float(nn.functional.cross_entropy(z, torch.as_tensor(y[m])))
        val_loss = sum(val_parts.values())
        if not np.isfinite(val_loss) or not np.isfinite(tr_loss):
            raise FloatingPointError("non-finite loss in Model A training")
        log.epochs.append({"epoch": epoch, "train_loss": tr_loss / max(nb, 1), "val_loss": val_loss, "val_parts": val_parts})
        if val_loss < log.best_val - 1e-4:
            log.best_val, log.best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    log.seconds = time.time() - t0
    return model.cpu().eval(), log


# ---------------------------------------------------------------- Model B


@dataclass
class BData:
    sit_cat: np.ndarray
    num: np.ndarray
    action: np.ndarray
    epa: np.ndarray
    events: np.ndarray  # nan = masked


def b_loss(out: dict[str, torch.Tensor], epa: torch.Tensor, ev: torch.Tensor, huber_weight: float = 0.5) -> torch.Tensor:
    mse = nn.functional.mse_loss(out["epa_mean"], epa)
    hub = nn.functional.huber_loss(out["epa_robust"], epa, delta=1.0)
    m = ~torch.isnan(ev)
    bce = nn.functional.binary_cross_entropy_with_logits(out["event_logits"][m], ev[m], reduction="none")
    head = m.nonzero()[:, 1]
    per_head = torch.zeros(ev.shape[1], device=ev.device).index_add(0, head, bce)
    counts = m.sum(0).clamp(min=1)
    return mse + huber_weight * hub + (per_head / counts).sum()


def predict_b(model: OutcomeModel, data: BData, action: np.ndarray | None = None, batch: int = 4096) -> dict[str, np.ndarray]:
    model.eval()
    act = data.action if action is None else action
    mean, robust, logits = [], [], []
    with torch.inference_mode():
        for s in range(0, len(act), batch):
            out = model(torch.as_tensor(data.sit_cat[s:s + batch]), torch.as_tensor(data.num[s:s + batch]),
                        torch.as_tensor(act[s:s + batch]))
            mean.append(out["epa_mean"].numpy())
            robust.append(out["epa_robust"].numpy())
            logits.append(out["event_logits"].numpy())
    return {"epa_mean": np.concatenate(mean), "epa_robust": np.concatenate(robust),
            "event_logits": np.concatenate(logits)}


def fit_model_b(train: BData, val: BData, seed: int, max_epochs: int = 30, patience: int = 4, lr: float = 1e-3,
                batch: int = 512, hidden: int = 128) -> tuple[OutcomeModel, FitLog]:
    seed_everything(seed)
    model = OutcomeModel(train.num.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log = FitLog(device="cpu", n_parameters=sum(p.numel() for p in model.parameters()))
    rng = np.random.default_rng(seed)
    t = {k: torch.as_tensor(v) for k, v in vars(train).items()}
    best_state, bad = None, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(train.action))
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch):
            ix = torch.as_tensor(order[s:s + batch])
            out = model(t["sit_cat"][ix], t["num"][ix], t["action"][ix])
            loss = b_loss(out, t["epa"][ix], t["events"][ix])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        pv = predict_b(model, val)
        with torch.no_grad():
            vl = float(b_loss({k: torch.as_tensor(v) for k, v in pv.items()}, torch.as_tensor(val.epa),
                              torch.as_tensor(val.events), huber_weight=0.0))
        log.epochs.append({"epoch": epoch, "train_loss": tot / nb, "val_loss": vl})
        if vl < log.best_val - 1e-5:
            log.best_val, log.best_epoch, bad = vl, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    log.seconds = time.time() - t0
    return model.eval(), log


__all__ = ["AData", "BData", "fit_model_a", "fit_model_b", "predict_a_logits", "predict_b", "HistoryStore",
           "masked_multitask_ce", "b_loss", "B_EVENTS", "train_device", "seed_everything"]
