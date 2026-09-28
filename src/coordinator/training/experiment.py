"""Main research run: R01 prior ablations, R03 sequence ablations, Model B feature-contract
comparison, a prespecified promotion gate, final refit, calibration and one locked test.

Decision rule (frozen before the test is opened):
  A neural challenger replaces the incumbent baseline only if its primary loss improves by at
  least 1% relative on BOTH development windows (2024 wk1-12 and 2024 wk13-18) and the game-
  clustered 95% interval of the paired difference on 2024 wk1-12 lies below zero.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import platform
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from coordinator import config
from coordinator.data.snapshot import code_commit, load_snapshot
from coordinator.features.builder import FeatureBuilder
from coordinator.features.context import PriorParams, compute_priors
from coordinator.models.bundle import write_bundle
from coordinator.models.calibration import conformal_quantile, fit_platt, fit_temperature, sigmoid, softmax
from coordinator.models.nets import A_HEADS, B_EVENTS
from coordinator.training.data import (
    Standardizer,
    Workspace,
    a_targets,
    b_targets,
    inputs_for,
    make_workspace,
    split_rows,
)
from coordinator.training.fit import (
    AData,
    BData,
    HistoryStore,
    fit_model_a,
    fit_model_b,
    predict_a_logits,
    predict_b,
)
from coordinator.training.metrics import (
    calibration_in_the_large_and_slope,
    calibration_table,
    cluster_bootstrap_mean,
    log_loss,
    paired_difference,
)

PRIOR_HEAD_FOR_A = {"box": "box", "rush": "rush", "blitz": "blitz", "coverage": "coverage", "man": "man",
                    "pressure": "pressure"}
PROMOTION_MIN_RELATIVE = 0.01
CONFORMAL_COVERAGE = 0.90


def _games(rows: pl.DataFrame) -> np.ndarray:
    return rows["game_id"].to_numpy()


# ------------------------------------------------------------------ R01: priors

def prior_probs(ws: Workspace, rows: pl.DataFrame, params: PriorParams, heads=("box", "five_plus", "blitz", "coverage")) -> dict:
    priors, _, _ = compute_priors(ws.builder.team_games, ws.contexts, params)
    ci = rows["context_idx"].to_numpy()
    cell = rows["cell_idx"].to_numpy()
    return {h: priors[h][ci, cell] for h in heads}


PRIOR_TARGETS = {"box": "t_box", "five_plus": "t_five_plus", "blitz": "t_blitz", "coverage": "t_coverage"}


def _prior_losses(ws: Workspace, rows: pl.DataFrame, params: PriorParams) -> dict[str, np.ndarray]:
    probs = prior_probs(ws, rows, params)
    out = {}
    for h, col in PRIOR_TARGETS.items():
        y = rows[col].cast(pl.Float64).to_numpy()
        m = ~np.isnan(y)
        out[h] = (log_loss(probs[h][m], y[m].astype(int)), m)
    return out


def run_r01(ws: Workspace, quick: bool) -> tuple[PriorParams, dict]:
    tune = ws.rows.filter(pl.col("season") == 2023)
    grid = []
    kappas = (10, 50, 200) if quick else (0, 10, 25, 50, 100, 200)
    a_teams = (0.0, 0.5, 1.0) if quick else (0.0, 0.25, 0.5, 0.75, 1.0)
    taus = (365.0,) if quick else (180.0, 365.0, 730.0)
    for tau in taus:
        for k in kappas:
            for a in a_teams:
                p = PriorParams(tau_hist_days=tau, kappa=float(k), a_team=a)
                losses = _prior_losses(ws, tune, p)
                # coverage has no 2023 history (NGS-era 2022 excluded), so it cannot inform tuning
                score = sum(float(v[0].mean()) for h, v in losses.items() if h != "coverage")
                grid.append({"tau_hist_days": tau, "kappa": k, "a_team": a, "tuning_score": score})
    best = min(grid, key=lambda g: g["tuning_score"])
    chosen = PriorParams(tau_hist_days=best["tau_hist_days"], kappa=float(best["kappa"]), a_team=best["a_team"])

    variants = {
        "league_only": dataclasses.replace(chosen, a_team=0.0, kappa=1e9),
        "team_only": dataclasses.replace(chosen, a_team=1.0, kappa=1e9),
        "prior_mixture_no_current": dataclasses.replace(chosen, kappa=1e9),
        "current_only": dataclasses.replace(chosen, kappa=0.0),
        "mixture_with_current": chosen,
    }
    report: dict = {"grid": grid, "chosen": chosen.to_dict(), "coordinator_component": "unavailable (no verified playcaller registry)",
                    "evaluation": {}}
    val = ws.rows.filter((pl.col("season") == 2024) & (pl.col("week") <= 12))
    for slice_name, expr in {"weeks_1_4": pl.col("week") <= 4, "weeks_5_12": pl.col("week") > 4}.items():
        r = val.filter(expr)
        res = {name: _prior_losses(ws, r, p) for name, p in variants.items()}
        games = _games(r)
        out = {}
        for h in PRIOR_TARGETS:
            base_loss, m = res["league_only"][h]
            out[h] = {name: float(res[name][h][0].mean()) for name in variants}
            out[h]["n"] = int(m.sum())
            out[h]["mixture_vs_league"] = paired_difference(res["mixture_with_current"][h][0], base_loss, games[m])
        report["evaluation"][slice_name] = out
    return chosen, report


# ------------------------------------------------------------------ Model A

def _a_data(ws: Workspace, rows: pl.DataFrame, std: Standardizer | None = None) -> tuple[AData, Standardizer]:
    inp = inputs_for(ws, rows, "charted")
    x = np.concatenate([inp.sit_num, inp.ctx_num], axis=1)
    std = std or Standardizer().fit(x)
    cell = rows["cell_idx"].to_numpy()
    prior_logp = {h: np.log(np.clip(ws.cf.priors[PRIOR_HEAD_FOR_A[h]][inp.ctx_idx, cell], 1e-6, 1.0)).astype(np.float32)
                  for h in A_HEADS}
    return AData(inp.ctx_idx, inp.sit_cat, std.transform(x), a_targets(rows), prior_logp), std


def _a_loss_table(probs: dict[str, np.ndarray], data: AData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for h in A_HEADS:
        y = data.targets[h]
        m = y >= 0
        out[h] = (log_loss(probs[h][m], y[m]), m)
    return out


def _prior_a_probs(ws: Workspace, rows: pl.DataFrame, params: PriorParams) -> dict[str, np.ndarray]:
    priors, _, _ = compute_priors(ws.builder.team_games, ws.contexts, params)
    ci, cell = rows["context_idx"].to_numpy(), rows["cell_idx"].to_numpy()
    return {h: priors[PRIOR_HEAD_FOR_A[h]][ci, cell] for h in A_HEADS}


def _sk_a(train: AData, evals: list[AData], kind: str) -> list[dict[str, np.ndarray]]:
    xs = lambda d: np.concatenate([d.query_num, d.sit_cat], axis=1)  # noqa: E731
    outs = [{} for _ in evals]
    for h, k in A_HEADS.items():
        y = train.targets[h]
        m = y >= 0
        est = (LogisticRegression(max_iter=2000, C=0.5) if kind == "logistic"
               else HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                                   early_stopping=True, random_state=0))
        est.fit(xs(train)[m], y[m])
        for o, d in zip(outs, evals, strict=True):
            p = np.full((len(d.ctx_idx), k), 1e-6)
            p[:, est.classes_] = est.predict_proba(xs(d))
            o[h] = p / p.sum(1, keepdims=True)
    return outs


def _summ(losses: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    per = {h: float(v[0].mean()) if len(v[0]) else None for h, v in losses.items()}
    return {"per_head_log_loss": per, "primary_sum_log_loss": float(sum(v for v in per.values() if v is not None))}


def _stacked_loss(losses: dict, heads=A_HEADS) -> np.ndarray:
    """Per-row sum over heads of per-head-normalized losses, for paired comparisons."""
    n = len(next(iter(losses.values()))[1])
    tot = np.zeros(n)
    for h in heads:
        loss, m = losses[h]
        full = np.zeros(n)
        full[m] = loss / m.mean()  # rescale so each head contributes its mean loss in expectation
        tot += full
    return tot


# ------------------------------------------------------------------ Model B

def _b_data(ws: Workspace, rows: pl.DataFrame, contract: str, std: Standardizer | None = None) -> tuple[BData, Standardizer]:
    inp = inputs_for(ws, rows, contract)
    x = np.concatenate([inp.sit_num, inp.ctx_num], axis=1)
    std = std or Standardizer().fit(x)
    epa, ev, act = b_targets(rows)
    return BData(inp.sit_cat, std.transform(x), act, epa, ev), std


def _b_eval(epa_pred: np.ndarray, ev_prob: np.ndarray, d: BData) -> dict:
    out = {"epa_mse": float(((epa_pred - d.epa) ** 2).mean()), "epa_bias": float((epa_pred - d.epa).mean()),
           "epa_mae": float(np.abs(epa_pred - d.epa).mean())}
    for j, e in enumerate(B_EVENTS):
        m = ~np.isnan(d.events[:, j])
        if not m.any():
            out[f"{e}_log_loss"] = out[f"{e}_brier"] = None
            continue
        p = np.clip(ev_prob[m, j], 1e-6, 1 - 1e-6)
        y = d.events[m, j]
        out[f"{e}_log_loss"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out[f"{e}_brier"] = float(((p - y) ** 2).mean())
    return out


def _situational_b(train_rows: pl.DataFrame, eval_rows: list[pl.DataFrame], shrink: float = 50.0):
    keys = ["action", "down", "band_idx", "zone_idx"]
    cols = ["y_epa"] + [f"y_{e}" for e in B_EVENTS]
    t = train_rows.with_columns([pl.col(c).cast(pl.Float64) for c in cols])
    top = t.group_by("action").agg([pl.col(c).mean().alias(f"top_{c}") for c in cols])
    cell = t.group_by(keys).agg([pl.col(c).sum().alias(f"s_{c}") for c in cols] + [pl.col(c).count().alias(f"n_{c}") for c in cols])
    outs = []
    for r in eval_rows:
        j = r.select(keys).with_row_index("_i").join(cell, on=keys, how="left").join(top, on="action", how="left").sort("_i")
        vals = {c: ((j[f"s_{c}"].fill_null(0) + shrink * j[f"top_{c}"]) / (j[f"n_{c}"].fill_null(0) + shrink)).to_numpy() for c in cols}
        ev = np.stack([vals[f"y_{e}"] for e in B_EVENTS], axis=1)
        ev[(r["action"] == "designed_rush").to_numpy(), B_EVENTS.index("sack")] = 0.0
        outs.append((vals["y_epa"], ev))
    return outs


def _sk_b(train: BData, evals: list[BData], kind: str):
    xs = lambda d: np.concatenate([d.num, np.eye(2)[d.action], d.sit_cat], axis=1)  # noqa: E731
    if kind == "linear":
        reg = Ridge(alpha=10.0)
        clf = lambda: LogisticRegression(max_iter=2000, C=0.5)  # noqa: E731
    else:
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            early_stopping=True, random_state=0)
        clf = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,  # noqa: E731
                                                     early_stopping=True, random_state=0)
    reg.fit(xs(train), train.epa)
    models = {"epa": reg}
    for j, e in enumerate(B_EVENTS):
        m = ~np.isnan(train.events[:, j])
        c = clf()
        c.fit(xs(train)[m], train.events[m, j].astype(int))
        models[e] = c
    outs = []
    for d in evals:
        ev = np.stack([models[e].predict_proba(xs(d))[:, 1] for e in B_EVENTS], axis=1)
        ev[d.action == 0, B_EVENTS.index("sack")] = 0.0
        outs.append((models["epa"].predict(xs(d)), ev))
    return outs, models


def _mlp_b_probs(models, d: BData, action=None):
    preds = [predict_b(m, d, action) for m in models]
    epa = np.mean([p["epa_mean"] for p in preds], axis=0)
    ev = sigmoid(np.mean([p["event_logits"] for p in preds], axis=0))
    act = d.action if action is None else action
    ev[act == 0, B_EVENTS.index("sack")] = 0.0
    return epa, ev, preds


# ------------------------------------------------------------------ main

def run(snapshot_id: str, quick: bool = False, seeds: tuple[int, ...] = (0, 1, 2), prior_offset: bool = True,
        note: str | None = None) -> dict:
    t_start = time.time()
    exp_id = f"exp_{time.strftime('%Y%m%dT%H%M%S')}{'_quick' if quick else ''}"
    out_dir = config.EXPERIMENTS_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = load_snapshot(snapshot_id)
    history_length = 32 if quick else 128
    ws = make_workspace(FeatureBuilder(snap), history_length=history_length)
    if quick:
        seeds = (0,)
    max_ep_a = 1 if quick else 10
    max_ep_b = 2 if quick else 30
    report: dict = {
        "experiment_id": exp_id, "snapshot_id": snapshot_id, "quick": quick, "code_commit": code_commit(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "torch": torch.__version__, "train_device": "mps" if torch.backends.mps.is_available() else "cpu"},
        "splits": {"select_train": "2023 wk1-14", "select_early_stop": "2023 wk15-18",
                   "select_val (dev window 1)": "2024 wk1-12", "calibration (dev window 2)": "2024 wk13-18",
                   "final_train": "2023 + 2024 wk1-8", "final_early_stop": "2024 wk9-12",
                   "locked_test": "2025 regular season"},
        "temporal_evidence": snap.temporal_mode,
        "decision_rule": __doc__.split("Decision rule")[1].strip(),
        "model_a_prior_offset": prior_offset,
        "note": note,
    }

    # ---- R01 priors
    chosen_prior, r01 = run_r01(ws, quick)
    report["R01_priors"] = r01
    ws.builder.prior_params = chosen_prior
    ws.cf = ws.builder.context_features(ws.contexts)

    rows = {s: split_rows(ws, s) for s in ("select_train", "select_early_stop", "select_val", "calibration",
                                            "final_train", "final_early_stop", "test")}
    if quick:
        rows = {k: v.sample(fraction=0.15, seed=0).sort(["kickoff_at", "game_id", "play_order"]) for k, v in rows.items()}

    # ---- Model A selection (single exploratory seed)
    a_tr, a_std = _a_data(ws, rows["select_train"])
    a_es, _ = _a_data(ws, rows["select_early_stop"], a_std)
    a_val, _ = _a_data(ws, rows["select_val"], a_std)
    a_cal, _ = _a_data(ws, rows["calibration"], a_std)
    a_sel: dict = {}
    losses_val: dict = {}
    losses_cal: dict = {}
    for name, p in {"prior_league": dataclasses.replace(chosen_prior, a_team=0.0, kappa=1e9),
                    "prior_mixture": chosen_prior}.items():
        losses_val[name] = _a_loss_table(_prior_a_probs(ws, rows["select_val"], p), a_val)
        losses_cal[name] = _a_loss_table(_prior_a_probs(ws, rows["calibration"], p), a_cal)
    for kind in ("logistic", "hgb"):
        pv, pc = _sk_a(a_tr, [a_val, a_cal], kind)
        losses_val[kind], losses_cal[kind] = _a_loss_table(pv, a_val), _a_loss_table(pc, a_cal)
    po = {"prior_offset": prior_offset}
    variants = {
        "query_only_mlp": {"arch": "query_only", **po},
        "pooled_history": {"arch": "pooled", **po},
        "transformer": {"arch": "transformer", **po},
        "transformer_shuffled": {"arch": "transformer", "shuffle_order": True, **po},
        "transformer_no_position": {"arch": "transformer", "use_position": False, **po},
    }
    if not quick:
        variants["transformer_len32"] = {"arch": "transformer", "history_length": 32, **po}
        variants["transformer_len64"] = {"arch": "transformer", "history_length": 64, **po}
        if prior_offset:
            variants["transformer_without_prior_offset"] = {"arch": "transformer", "prior_offset": False}
    fit_logs = {}
    for name, kw in variants.items():
        hl = kw.pop("history_length", history_length)
        hist = ws.hist if hl == history_length else ws.builder.histories(ws.contexts, hl)
        model, log = fit_model_a(a_tr, a_es, hist, seed=seeds[0], max_epochs=max_ep_a, history_length=hl, **kw)
        store = HistoryStore(hist, torch.device("cpu"))
        so = kw.get("shuffle_order", False)
        pv = {h: softmax(v) for h, v in predict_a_logits(model, store, a_val, torch.device("cpu"), shuffle_order=so).items()}
        pc = {h: softmax(v) for h, v in predict_a_logits(model, store, a_cal, torch.device("cpu"), shuffle_order=so).items()}
        losses_val[name], losses_cal[name] = _a_loss_table(pv, a_val), _a_loss_table(pc, a_cal)
        fit_logs[name] = {"seconds": log.seconds, "best_epoch": log.best_epoch, "n_parameters": log.n_parameters,
                          "device": log.device, "epochs": log.epochs}
    for name in losses_val:
        a_sel[name] = {"dev_window_1": _summ(losses_val[name]), "dev_window_2": _summ(losses_cal[name])}
    baselines_a = ["prior_league", "prior_mixture", "logistic", "hgb"]
    incumbent_a = min(baselines_a, key=lambda n: a_sel[n]["dev_window_1"]["primary_sum_log_loss"])
    games_val, games_cal = _games(rows["select_val"]), _games(rows["calibration"])
    for name in a_sel:
        a_sel[name]["vs_incumbent_window_1"] = paired_difference(
            _stacked_loss(losses_val[name]), _stacked_loss(losses_val[incumbent_a]), games_val)
        a_sel[name]["vs_incumbent_window_2"] = paired_difference(
            _stacked_loss(losses_cal[name]), _stacked_loss(losses_cal[incumbent_a]), games_cal)
    neural_a = [n for n in variants if n in ("transformer", "pooled_history", "query_only_mlp")]
    best_neural_a = min(neural_a, key=lambda n: a_sel[n]["dev_window_1"]["primary_sum_log_loss"])

    def passes(sel: dict) -> bool:
        w1, w2 = sel["vs_incumbent_window_1"], sel["vs_incumbent_window_2"]
        if w1["relative_change"] is None or w2["relative_change"] is None:
            return False
        return (w1["relative_change"] <= -PROMOTION_MIN_RELATIVE
                and w2["relative_change"] <= -PROMOTION_MIN_RELATIVE and w1["ci_high"] < 0)

    a_decision = {"incumbent": incumbent_a, "best_neural": best_neural_a, "neural_passes_gate": passes(a_sel[best_neural_a])}
    a_decision["served"] = best_neural_a if a_decision["neural_passes_gate"] else incumbent_a
    report["model_A_selection"] = {"results": a_sel, "fit_logs": fit_logs, "decision": a_decision}

    # ---- Model B selection: feature contracts x model families
    b_sel: dict = {}
    b_rows = {k: v.filter(pl.col("in_reg_scrimmage_penalty_free_v1")) for k, v in rows.items()}
    for contract in ("pbp", "pbp_ftn", "charted"):
        b_tr, b_std = _b_data(ws, b_rows["select_train"], contract)
        b_es, _ = _b_data(ws, b_rows["select_early_stop"], contract, b_std)
        b_val, _ = _b_data(ws, b_rows["select_val"], contract, b_std)
        b_cal, _ = _b_data(ws, b_rows["calibration"], contract, b_std)
        res = {}
        if contract == "pbp":
            (sv, sc) = _situational_b(b_rows["select_train"], [b_rows["select_val"], b_rows["calibration"]])
            res["situational"] = (sv, sc)
        for kind in ("linear", "hgb"):
            outs, _ = _sk_b(b_tr, [b_val, b_cal], kind)
            res[kind] = tuple(outs)
        models = [fit_model_b(b_tr, b_es, seed=seeds[0], max_epochs=max_ep_b)[0]]
        e_v, p_v, _ = _mlp_b_probs(models, b_val)
        e_c, p_c, _ = _mlp_b_probs(models, b_cal)
        res["mlp"] = ((e_v, p_v), (e_c, p_c))
        for kind, ((ev_, pv_), (ec_, pc_)) in res.items():
            b_sel[f"{contract}:{kind}"] = {
                "dev_window_1": _b_eval(ev_, pv_, b_val), "dev_window_2": _b_eval(ec_, pc_, b_cal),
                "_sq_val": (ev_ - b_val.epa) ** 2, "_sq_cal": (ec_ - b_cal.epa) ** 2,
            }
    gv, gc = _games(b_rows["select_val"]), _games(b_rows["calibration"])
    # Servable incumbents only; the situational table is reported as a reference comparator.
    baselines_b = ["pbp:linear", "pbp:hgb"]
    incumbent_b = min(baselines_b, key=lambda k: b_sel[k]["dev_window_1"]["epa_mse"])
    for k in b_sel:
        b_sel[k]["vs_incumbent_window_1"] = paired_difference(b_sel[k]["_sq_val"], b_sel[incumbent_b]["_sq_val"], gv)
        b_sel[k]["vs_incumbent_window_2"] = paired_difference(b_sel[k]["_sq_cal"], b_sel[incumbent_b]["_sq_cal"], gc)
    for k in b_sel:
        b_sel[k].pop("_sq_val")
        b_sel[k].pop("_sq_cal")
    # Served pbp_baseline bundle must use the 'pbp' contract (no charted inputs).
    b_mlp_pass = passes(b_sel["pbp:mlp"])
    b_decision = {"incumbent": incumbent_b, "pbp_mlp_passes_gate": b_mlp_pass,
                  "served": "pbp:mlp" if b_mlp_pass else incumbent_b}
    report["model_B_selection"] = {"results": b_sel, "decision": b_decision}

    # ---- Final refit on 2023 + 2024 wk1-8, calibrate on 2024 wk13-18, locked test on 2025
    final: dict = {}
    fa_tr, fa_std = _a_data(ws, rows["final_train"])
    fa_es, _ = _a_data(ws, rows["final_early_stop"], fa_std)
    fa_cal, _ = _a_data(ws, rows["calibration"], fa_std)
    fa_te, _ = _a_data(ws, rows["test"], fa_std)
    a_members, a_logs, temps = [], [], {}
    store = HistoryStore(ws.hist, torch.device("cpu"))
    arch = a_decision["served"]
    arch_kw = {"transformer": {"arch": "transformer", **po}, "pooled_history": {"arch": "pooled", **po},
               "query_only_mlp": {"arch": "query_only", **po}}
    test_losses_a = {}
    if arch in arch_kw:
        for s in seeds:
            m, log = fit_model_a(fa_tr, fa_es, ws.hist, seed=s, max_epochs=max_ep_a, history_length=history_length, **arch_kw[arch])
            a_members.append(m)
            a_logs.append({"seed": s, "seconds": log.seconds, "best_epoch": log.best_epoch, "n_parameters": log.n_parameters})
        cal_logits = [predict_a_logits(m, store, fa_cal, torch.device("cpu")) for m in a_members]
        te_logits = [predict_a_logits(m, store, fa_te, torch.device("cpu")) for m in a_members]
        for h in A_HEADS:
            y = fa_cal.targets[h]
            msk = y >= 0
            temps[h] = fit_temperature(np.mean([c[h] for c in cal_logits], axis=0)[msk], y[msk]) if msk.sum() > 50 else 1.0
        served_test = {h: np.mean([softmax(t[h] / temps[h]) for t in te_logits], axis=0) for h in A_HEADS}
    else:
        pp = chosen_prior if arch == "prior_mixture" else dataclasses.replace(chosen_prior, a_team=0.0, kappa=1e9)
        served_test = _prior_a_probs(ws, rows["test"], pp)
    test_losses_a["served:" + arch] = _a_loss_table(served_test, fa_te)
    for name, p in {"prior_league": dataclasses.replace(chosen_prior, a_team=0.0, kappa=1e9), "prior_mixture": chosen_prior}.items():
        test_losses_a[name] = _a_loss_table(_prior_a_probs(ws, rows["test"], p), fa_te)
    for kind in ("logistic", "hgb"):
        (pt,) = _sk_a(fa_tr, [fa_te], kind)
        test_losses_a[kind] = _a_loss_table(pt, fa_te)
    gt = _games(rows["test"])
    final["model_A_test"] = {n: _summ(v) for n, v in test_losses_a.items()}
    final["model_A_test"]["served_vs_incumbent"] = paired_difference(
        _stacked_loss(test_losses_a["served:" + arch]), _stacked_loss(test_losses_a[incumbent_a]), gt)
    final["model_A_members"] = a_logs
    final["model_A_temperatures"] = temps
    cov_y = fa_te.targets["coverage"]
    cm = cov_y >= 0
    if cm.any():
        pred = served_test["coverage"][cm].argmax(1)
        conf = np.zeros((8, 8), dtype=int)
        np.add.at(conf, (cov_y[cm], pred), 1)
        final["model_A_test_coverage_confusion"] = conf.tolist()
        final["model_A_test_coverage_accuracy"] = float((pred == cov_y[cm]).mean())
        final["model_A_test_coverage_majority_class_rate"] = float(np.bincount(cov_y[cm]).max() / cm.sum())
    for h in ("blitz", "pressure", "man"):
        y = fa_te.targets[h]
        mm = y >= 0
        if mm.any():
            final.setdefault("model_A_test_calibration", {})[h] = {
                "table": calibration_table(served_test[h][mm, 1], y[mm]),
                **calibration_in_the_large_and_slope(served_test[h][mm, 1], y[mm])}

    # Model B final (pbp contract)
    fb_rows = {k: b_rows[k] for k in ("final_train", "final_early_stop", "calibration", "test")}
    fb_tr, fb_std = _b_data(ws, fb_rows["final_train"], "pbp")
    fb_es, _ = _b_data(ws, fb_rows["final_early_stop"], "pbp", fb_std)
    fb_cal, _ = _b_data(ws, fb_rows["calibration"], "pbp", fb_std)
    fb_te, _ = _b_data(ws, fb_rows["test"], "pbp", fb_std)
    served_b = b_decision["served"].split(":")[1]
    b_members, b_logs, sk_models = [], [], None
    if served_b == "mlp":
        for s in seeds:
            m, log = fit_model_b(fb_tr, fb_es, seed=s, max_epochs=max_ep_b)
            b_members.append(m)
            b_logs.append({"seed": s, "seconds": log.seconds, "best_epoch": log.best_epoch, "n_parameters": log.n_parameters})
        e_cal, _, pred_cal = _mlp_b_probs(b_members, fb_cal)
        logit_cal = np.mean([p["event_logits"] for p in pred_cal], axis=0)
        e_te, _, pred_te = _mlp_b_probs(b_members, fb_te)
        logit_te = np.mean([p["event_logits"] for p in pred_te], axis=0)
    elif served_b in ("hgb", "linear"):
        (oc, ot), sk_models = _sk_b(fb_tr, [fb_cal, fb_te], served_b)
        e_cal, e_te = oc[0], ot[0]
        logit_cal = np.log(np.clip(oc[1], 1e-6, 1 - 1e-6) / np.clip(1 - oc[1], 1e-6, 1))
        logit_te = np.log(np.clip(ot[1], 1e-6, 1 - 1e-6) / np.clip(1 - ot[1], 1e-6, 1))
    platt = {}
    for j, e in enumerate(B_EVENTS):
        mm = ~np.isnan(fb_cal.events[:, j])
        platt[e] = fit_platt(logit_cal[mm, j], fb_cal.events[mm, j]) if mm.sum() > 50 else (1.0, 0.0)
    cal_probs_te = np.stack([sigmoid(platt[e][0] * logit_te[:, j] + platt[e][1]) for j, e in enumerate(B_EVENTS)], axis=1)
    cal_probs_te[fb_te.action == 0, B_EVENTS.index("sack")] = 0.0
    conformal = {}
    for a, aname in enumerate(("designed_rush", "dropback")):
        m = fb_cal.action == a
        conformal[aname] = conformal_quantile(np.abs(fb_cal.epa[m] - e_cal[m]), CONFORMAL_COVERAGE)
    q_te = np.where(fb_te.action == 1, conformal["dropback"], conformal["designed_rush"])
    covered = np.abs(fb_te.epa - e_te) <= q_te

    test_b = {"served:" + b_decision["served"]: _b_eval(e_te, cal_probs_te, fb_te)}
    (sit_te,) = _situational_b(fb_rows["final_train"], [fb_rows["test"]])
    test_b["pbp:situational"] = _b_eval(sit_te[0], sit_te[1], fb_te)
    for kind in ("linear", "hgb"):
        if b_decision["served"] == f"pbp:{kind}":
            continue
        (ot,), _ = _sk_b(fb_tr, [fb_te], kind)
        test_b[f"pbp:{kind}"] = _b_eval(ot[0], ot[1], fb_te)
    gtb = _games(fb_rows["test"])
    inc_pred = {"pbp:situational": sit_te[0]}.get(incumbent_b)
    if inc_pred is None:
        kind = incumbent_b.split(":")[1]
        (ot,), _ = _sk_b(fb_tr, [fb_te], kind)
        inc_pred = ot[0]
    test_b["served_vs_incumbent_epa_sq_error"] = paired_difference((e_te - fb_te.epa) ** 2, (inc_pred - fb_te.epa) ** 2, gtb)
    test_b["by_action"] = {
        aname: {"n": int((fb_te.action == a).sum()), **_b_eval(e_te[fb_te.action == a], cal_probs_te[fb_te.action == a],
                                                             BData(fb_te.sit_cat[fb_te.action == a], fb_te.num[fb_te.action == a],
                                                                   fb_te.action[fb_te.action == a], fb_te.epa[fb_te.action == a],
                                                                   fb_te.events[fb_te.action == a]))}
        for a, aname in enumerate(("designed_rush", "dropback"))}
    test_b["interval"] = {"method": "split_conformal_residual", "nominal_coverage": CONFORMAL_COVERAGE,
                          "half_width": conformal,
                          "empirical_coverage": {aname: float(covered[fb_te.action == a].mean()) for a, aname in enumerate(("designed_rush", "dropback"))},
                          "empirical_coverage_all": cluster_bootstrap_mean(covered.astype(float), gtb)}
    test_b["event_calibration"] = {e: {"table": calibration_table(cal_probs_te[m, j], fb_te.events[m, j]),
                                       **calibration_in_the_large_and_slope(cal_probs_te[m, j], fb_te.events[m, j])}
                                   for j, e in enumerate(B_EVENTS) if (m := ~np.isnan(fb_te.events[:, j])).sum() > 0}
    # Recommendation-level diagnostics on test situations: score both candidates.
    alt = {}
    for a in (0, 1):
        act = np.full(len(fb_te.action), a)
        if served_b == "mlp":
            alt[a] = np.stack([predict_b(m, fb_te, act)["epa_mean"] for m in b_members])
        else:
            x = np.concatenate([fb_te.num, np.eye(2)[act], fb_te.sit_cat], axis=1)
            alt[a] = sk_models["epa"].predict(x)[None, :]
    margin = alt[1].mean(0) - alt[0].mean(0)
    spread = np.sqrt(alt[1].std(0) ** 2 + alt[0].std(0) ** 2) if alt[1].shape[0] > 1 else np.zeros_like(margin)
    rank_agree = float(np.mean(np.all(np.sign(alt[1] - alt[0]) == np.sign(margin), axis=0))) if alt[1].shape[0] > 1 else None
    test_b["recommendation_diagnostics"] = {
        "share_dropback_ranked_first": float((margin > 0).mean()),
        "median_abs_margin_epa": float(np.median(np.abs(margin))),
        "share_margin_below_spread": float((np.abs(margin) < spread).mean()),
        "seed_rank_agreement": rank_agree,
        "note": "Factual outcomes exist only for the chosen action; these are not value-of-recommendation estimates.",
    }
    final["model_B_test"] = test_b
    final["model_B_members"] = b_logs
    report["final"] = final
    report["runtime_seconds"] = time.time() - t_start

    # ---- Bundle
    ma = {"kind": arch, "arch": arch_kw.get(arch, {}).get("arch"), "history_length": history_length,
          "n_query_num": int(fa_tr.query_num.shape[1]), "member_files": [], "temperatures": temps,
          "standardizer": fa_std.to_dict() if a_members else None, "feature_contract": "charted",
          "heads": list(A_HEADS), "use_position": True, "prior_offset": prior_offset}
    files: dict = {}
    for i, m in enumerate(a_members):
        files[f"model_a_seed{seeds[i]}.pt"] = m.state_dict()
        ma["member_files"].append(f"model_a_seed{seeds[i]}.pt")
    mb = {"kind": served_b, "feature_contract": "pbp", "n_num": int(fb_tr.num.shape[1]), "hidden": 128,
          "standardizer": fb_std.to_dict(), "platt": platt, "conformal_half_width": conformal,
          "conformal_nominal_coverage": CONFORMAL_COVERAGE, "member_files": []}
    if served_b == "mlp":
        for i, m in enumerate(b_members):
            files[f"model_b_seed{seeds[i]}.pt"] = m.state_dict()
            mb["member_files"].append(f"model_b_seed{seeds[i]}.pt")
    else:
        files["model_b_sklearn.joblib"] = sk_models
        mb["member_files"].append("model_b_sklearn.joblib")
    spec = {
        "bundle_name": "broad_v1" + ("_quick" if quick else ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_snapshot_id": snapshot_id, "experiment_id": exp_id, "evaluation_report_id": exp_id,
        "calibration_artifact_id": f"{exp_id}_calibration", "ranking_policy_id": "broad_policy_v1",
        "taxonomy_version": config.TAXONOMY_VERSION, "feature_schema_version": config.FEATURE_SCHEMA_VERSION,
        "outcome_cohort_id": config.PRIMARY_OUTCOME_COHORT, "response_contract_version": config.RESPONSE_CONTRACT_VERSION,
        "modes": ["pbp_baseline", "charted_history"], "candidates": ["designed_rush", "dropback"],
        "prior_params": chosen_prior.to_dict(),
        "strength_params": dataclasses.asdict(ws.builder.strength_params),
        "training_boundary": "2024 week 8 (early stop through week 12)", "calibration_boundary": "2024 weeks 13-18",
        "model_a": ma, "model_b": mb,
        "is_quick_smoke_bundle": quick,
    }
    bundle_id = write_bundle(spec, files)
    report["model_bundle_id"] = bundle_id
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    _write_metric_csv(report, out_dir / "metrics.csv")
    return report


def _write_metric_csv(report: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "model", "window", "metric", "value"])
        for name, r in report["model_A_selection"]["results"].items():
            for win in ("dev_window_1", "dev_window_2"):
                for h, v in r[win]["per_head_log_loss"].items():
                    w.writerow(["A_selection", name, win, f"{h}_log_loss", v])
        for name, r in report["model_B_selection"]["results"].items():
            for win in ("dev_window_1", "dev_window_2"):
                for k, v in r[win].items():
                    w.writerow(["B_selection", name, win, k, v])
        for name, r in report["final"]["model_A_test"].items():
            if "per_head_log_loss" in r:
                for h, v in r["per_head_log_loss"].items():
                    w.writerow(["A_test", name, "test_2025", f"{h}_log_loss", v])
        for name, r in report["final"]["model_B_test"].items():
            if isinstance(r, dict) and "epa_mse" in r:
                for k, v in r.items():
                    w.writerow(["B_test", name, "test_2025", k, v])
