"""馬場 what-if 感度診断: pred_rank*_baba* 列からシナリオ間差分を定量評価する。

Usage:
  python model_training/src/diagnostics_going_sensitivity.py
  python model_training/src/diagnostics_going_sensitivity.py --parquet main/results/today_predictions_with_bets.parquet
  python model_training/src/diagnostics_going_sensitivity.py --models-dir model_training/models/ensemble_v5
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from feature_groups import going_feature_names

DEFAULT_PARQUET = ROOT / "main" / "results" / "today_predictions_with_bets.parquet"
DEFAULT_MODELS = ROOT / "model_training" / "models" / "ensemble_v5"
LOG_DIR = ROOT / "model_training" / "logs" / "going_diagnostics"

SCENARIO_SUFFIXES = {1: "_baba1", 2: "_baba2", 3: "_baba3", 4: "_baba4"}
SCENARIO_LABELS = {1: "馬場_良", 2: "馬場_稍重", 3: "馬場_重", 4: "馬場_不良"}


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"予測 Parquet/CSV が見つかりません: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _score_col(df: pd.DataFrame, jv: int) -> str:
    return f"pred_rank1{SCENARIO_SUFFIXES[jv]}"


def _race_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "race_id" not in df.columns:
        raise ValueError("race_id 列が必要です")
    return df.groupby("race_id", sort=False)


def compute_scenario_sensitivity(df: pd.DataFrame) -> dict:
    """良(1)基準のシナリオ感度指標。"""
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]
    race_stats = []
    for rid, g in _race_groups(df):
        if len(g) < 2:
            continue
        hn = pd.to_numeric(g.get("horse_num", g.index), errors="coerce")
        row_base: dict = {"race_id": rid, "n_horses": len(g)}
        for a, b in pairs:
            ca, cb = _score_col(g, a), _score_col(g, b)
            if ca not in g.columns or cb not in g.columns:
                continue
            sa = pd.to_numeric(g.set_index(hn)[ca], errors="coerce")
            sb = pd.to_numeric(g.set_index(hn)[cb], errors="coerce")
            aligned = pd.concat([sa, sb], axis=1).dropna()
            if aligned.empty:
                continue
            diff = (aligned.iloc[:, 1] - aligned.iloc[:, 0]).abs()
            key = f"{SCENARIO_LABELS[a]}_vs_{SCENARIO_LABELS[b]}"
            row_base[f"{key}_max_diff"] = float(diff.max())
            row_base[f"{key}_mean_diff"] = float(diff.mean())
            row_base[f"{key}_all_identical"] = bool((aligned.iloc[:, 0] == aligned.iloc[:, 1]).all())
            row_base[f"{key}_top1_same"] = bool(
                aligned.iloc[:, 0].idxmax() == aligned.iloc[:, 1].idxmax()
            )
        race_stats.append(row_base)

    rs = pd.DataFrame(race_stats)
    if rs.empty:
        return {"n_races": 0, "error": "pred_rank1_baba* 列が見つかりません"}

    out: dict = {"n_races": int(len(rs))}
    for a, b in pairs:
        key = f"{SCENARIO_LABELS[a]}_vs_{SCENARIO_LABELS[b]}"
        col_max = f"{key}_max_diff"
        col_id = f"{key}_all_identical"
        col_top1 = f"{key}_top1_same"
        if col_max in rs.columns:
            out[f"{key}_max_diff_mean"] = float(rs[col_max].mean())
            out[f"{key}_max_diff_median"] = float(rs[col_max].median())
            out[f"{key}_max_diff_max"] = float(rs[col_max].max())
        if col_id in rs.columns:
            out[f"{key}_all_identical_rate"] = float(rs[col_id].mean())
        if col_top1 in rs.columns:
            out[f"{key}_top1_same_rate"] = float(rs[col_top1].mean())

    if "馬場_良_vs_馬場_不良_top1_same" in rs.columns:
        out["good_vs_bad_top1_flip_rate"] = float(1.0 - rs["馬場_良_vs_馬場_不良_top1_same"].mean())
    return out


def compute_model_going_gain_share(models_dir: Path) -> dict:
    pkl = models_dir / "lgbm_model_rank1_seed42.pkl"
    if not pkl.exists():
        pkls = sorted(models_dir.glob("lgbm_model_rank1_seed*.pkl"))
        if not pkls:
            return {"error": f"rank1 model not found in {models_dir}"}
        pkl = pkls[0]
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    fn = list(model.feature_name())
    imp = model.feature_importance(importance_type="gain")
    total = float(imp.sum()) or 1.0
    going_names = set(going_feature_names(fn))
    going_gain = float(imp[[i for i, n in enumerate(fn) if n in going_names]].sum())
    top_going = sorted(
        [(fn[i], float(imp[i])) for i in range(len(fn)) if fn[i] in going_names],
        key=lambda x: -x[1],
    )[:10]
    return {
        "models_dir": str(models_dir),
        "n_features": len(fn),
        "n_going_features": len(going_names),
        "going_gain_share_pct": round(going_gain / total * 100, 3),
        "top_going_features": [{"name": n, "gain": g} for n, g in top_going],
    }


def run(
    parquet_path: Path = DEFAULT_PARQUET,
    models_dir: Path = DEFAULT_MODELS,
    output_json: Path | None = None,
) -> dict:
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parquet_path": str(parquet_path),
    }
    try:
        df = _load_predictions(parquet_path)
        report["sensitivity"] = compute_scenario_sensitivity(df)
    except FileNotFoundError as e:
        report["sensitivity"] = {"error": str(e)}

    report["feature_importance"] = compute_model_going_gain_share(models_dir)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output_json or LOG_DIR / f"going_sensitivity_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="馬場 what-if 感度診断")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.parquet, args.models_dir, args.output)


if __name__ == "__main__":
    main()
