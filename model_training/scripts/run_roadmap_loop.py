"""ロードマップ全ステップ実行: 計画→実装→評価→（不合格なら次計画）。

各実験は 1 パラメータ/1 特徴量セットのみ。不合格時はモデル復元。

実行:
    python model_training/scripts/run_roadmap_loop.py
    python model_training/scripts/run_roadmap_loop.py --from-step form_momentum
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "model_training" / "scripts"
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from roadmap_eval import evaluate_experiment, load_results, save_verdict_report  # noqa: E402

CONFIG_PATH = ROOT / "model_training" / "config" / "train_config.json"
MODELS_DIR = ROOT / "model_training" / "models"
CHAMPION_DIR = MODELS_DIR / "champion_going_v1"
BASELINE_RESULTS = MODELS_DIR / "backtest_results_baseline_going_v1.json"
SPRINT_CANDIDATE = MODELS_DIR / "backtest_results_sprint_v1.json"

INITIAL_PLAN: list[dict[str, Any]] = [
    {
        "id": "rebaseline_going_v1",
        "type": "backtest_only",
        "desc": "going_v1 現行モデルでベースライン再計測",
        "result_name": "backtest_results_baseline_going_v1.json",
    },
    {
        "id": "form_momentum",
        "type": "feature_train",
        "desc": "going_v1 + form_momentum 3列",
        "build_set": "form_momentum",
        "feature_file": "features_v6_going_form_momentum_v1.parquet",
    },
    {
        "id": "track_cond_streak",
        "type": "feature_train",
        "desc": "going_v1 + track_cond_streak 2列",
        "build_set": "track_cond_streak",
        "feature_file": "features_v6_going_track_cond_streak_v1.parquet",
    },
    {
        "id": "rival_strength",
        "type": "feature_train",
        "desc": "going_v1 + rival_strength 3列",
        "build_set": "rival_strength",
        "feature_file": "features_v6_going_rival_strength_v1.parquet",
    },
    {
        "id": "temp_070",
        "type": "calibration_only",
        "desc": "Temperature=0.70（再学習なし）",
        "calibration_patch": {"temperature": 0.70},
    },
    {
        "id": "temp_065",
        "type": "calibration_only",
        "desc": "Temperature=0.65（再学習なし）",
        "calibration_patch": {"temperature": 0.65},
    },
    {
        "id": "temp_080",
        "type": "calibration_only",
        "desc": "Temperature=0.80（再学習なし）",
        "calibration_patch": {"temperature": 0.80},
    },
    {
        "id": "isotonic_on",
        "type": "calibration_only",
        "desc": "Isotonic較正 ON（再学習なし）",
        "calibration_patch": {"isotonic": True},
    },
    {
        "id": "reg_lambda_3",
        "type": "hp_train",
        "desc": "reg_lambda 2.0→3.0",
        "hp_patch": {"reg_lambda": 3.0},
    },
    {
        "id": "min_child_80",
        "type": "hp_train",
        "desc": "min_child_samples 50→80",
        "hp_patch": {"min_child_samples": 80},
    },
]


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _backup_models(tag: str) -> Path:
    backup_dir = MODELS_DIR / f"backup_roadmap_{tag}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("lgbm_binary_fold*.txt", "lgbm_binary_fold*.json", "lgbm_binary_fold*.joblib"):
        for p in MODELS_DIR.glob(pattern):
            shutil.copy2(p, backup_dir / p.name)
    return backup_dir


def _restore_models(backup_dir: Path) -> None:
    for p in backup_dir.glob("lgbm_binary_fold*"):
        shutil.copy2(p, MODELS_DIR / p.name)


def _save_champion_snapshot() -> None:
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("lgbm_binary_fold*.txt", "lgbm_binary_fold*.json"):
        for p in MODELS_DIR.glob(pattern):
            shutil.copy2(p, CHAMPION_DIR / p.name)


def _run_backtest(result_name: str) -> list[dict]:
    from backtest import run_full_backtest

    results = run_full_backtest()
    src = MODELS_DIR / "backtest_results.json"
    dst = MODELS_DIR / result_name
    if src.exists():
        shutil.copy2(src, dst)
    return results


def _run_ensemble_train() -> None:
    from train import run_ensemble_training

    run_ensemble_training(seeds=[42, 43, 44, 45, 46])


def _build_feature(build_set: str) -> None:
    from build_going_plus_features import build

    build(build_set)


def _register_latent_cols(cfg: dict, col_names: list[str]) -> None:
    """新特徴量を features.latent に登録（get_feature_cols が拾う）。"""
    latent = cfg.setdefault("features", {}).setdefault("latent", [])
    for c in col_names:
        if c not in latent:
            latent.append(c)


def _cols_for_step(step: dict) -> list[str]:
    if step.get("build_set"):
        from build_going_plus_features import SETS

        return SETS[step["build_set"]]["cols_fn"]()
    if step.get("feature_file") == "features_v6_sprint_v1.parquet":
        return [
            "horse_sprint_win_rate",
            "horse_sprint_top3_rate",
            "sprint_agari3f_mean",
            "distance_fit_score",
            "distance_bucket",
        ]
    return step.get("extra_latent_cols") or []


def _persist_hp_adoption(cfg: dict, step: dict) -> None:
    if step.get("hp_patch"):
        hp = cfg["training"].setdefault("backtest_conservative_params", {})
        hp.update(step["hp_patch"])


def _patch_config(cfg: dict, step: dict) -> dict:
    orig = copy.deepcopy(cfg)
    t = cfg["training"]
    if step.get("feature_file"):
        t["backtest_feature_file"] = step["feature_file"]
    if step.get("calibration_patch"):
        t.setdefault("calibration", {}).update(step["calibration_patch"])
    if step.get("hp_patch"):
        t.setdefault("backtest_conservative_params", {}).update(step["hp_patch"])
    return orig


def _print_summary(results: list[dict], label: str) -> None:
    print(f"\n=== {label} ===")
    for r in results:
        tm = r.get("test_metrics", {})
        dm = r.get("drawdown_metrics", {})
        print(
            f"F{r.get('fold')}: ROI={tm.get('roi', 0):.1%} "
            f"Sharpe={dm.get('sharpe_ratio', 0):.3f} n={tm.get('n_bets', 0)}"
        )


def _replan_steps(baseline: list[dict]) -> list[dict[str, Any]]:
    f2 = next((r for r in baseline if r.get("fold") == 2), {})
    base_sh = (f2.get("drawdown_metrics") or {}).get("sharpe_ratio", 0)
    extra: list[dict[str, Any]] = []
    print(f"\n[REPLAN] F2 Sharpe 0.10 not reached (baseline={base_sh:.3f})")

    if SPRINT_CANDIDATE.exists():
        sprint = load_results(SPRINT_CANDIDATE)
        sp_f2 = next((r for r in sprint if r.get("fold") == 2), {})
        sp_sh = (sp_f2.get("drawdown_metrics") or {}).get("sharpe_ratio", 0)
        if sp_sh >= base_sh:
            extra.append(
                {
                    "id": "adopt_sprint_v1",
                    "type": "feature_train",
                    "desc": "A2 sprint_v1 再学習（前回 F2 Sharpe=0.098）",
                    "feature_file": "features_v6_sprint_v1.parquet",
                    "skip_build": True,
                }
            )
    extra.append(
        {
            "id": "temp_068",
            "type": "calibration_only",
            "desc": "Temperature=0.68（再計画）",
            "calibration_patch": {"temperature": 0.68},
        }
    )
    return extra


def _execute_step(
    step: dict,
    cfg: dict,
    baseline_results: list[dict],
    champion_backup: Path,
) -> tuple[list[dict], Any, Path | None, dict]:
    """1 ステップ実行 → (results, verdict, pre_train_backup, orig_cfg)"""
    orig_cfg = copy.deepcopy(cfg)
    pre_train_backup: Path | None = None

    if step["type"] != "backtest_only":
        orig_cfg = _patch_config(cfg, step)
        _save_config(cfg)

    if step["type"] == "backtest_only":
        results = _run_backtest(step["result_name"])
        verdict = evaluate_experiment(step["id"], results, results, is_feature_experiment=False)
        verdict.verdict = "BASELINE"
        verdict.adopted = True
        verdict.reason = "ベースライン確定"
        _save_champion_snapshot()

    elif step["type"] == "feature_train":
        if not step.get("skip_build") and step.get("build_set"):
            _build_feature(step["build_set"])
        extra_cols = _cols_for_step(step)
        if extra_cols:
            _register_latent_cols(cfg, extra_cols)
            _save_config(cfg)
        pre_train_backup = _backup_models(step["id"])
        _run_ensemble_train()
        results = _run_backtest(f"backtest_results_{step['id']}.json")
        base = baseline_results or load_results(BASELINE_RESULTS)
        verdict = evaluate_experiment(step["id"], results, base)

    elif step["type"] == "calibration_only":
        pre_train_backup = champion_backup
        results = _run_backtest(f"backtest_results_{step['id']}.json")
        base = baseline_results or load_results(BASELINE_RESULTS)
        verdict = evaluate_experiment(step["id"], results, base, is_feature_experiment=False)

    elif step["type"] == "hp_train":
        pre_train_backup = _backup_models(step["id"])
        _run_ensemble_train()
        results = _run_backtest(f"backtest_results_{step['id']}.json")
        base = baseline_results or load_results(BASELINE_RESULTS)
        verdict = evaluate_experiment(step["id"], results, base, is_feature_experiment=False)

    else:
        raise ValueError(f"unknown step type: {step['type']}")

    return results, verdict, pre_train_backup, orig_cfg


def run_loop(from_step: str | None = None) -> None:
    cfg = _load_config()
    champion_backup = _backup_models(f"champion_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    baseline_results: list[dict] = []
    best_partial: tuple[str, float] | None = None
    adopted_id: str | None = None
    full_pass = False

    steps = list(INITIAL_PLAN)
    if from_step:
        ids = [s["id"] for s in steps]
        if from_step in ids:
            steps = steps[ids.index(from_step) :]

    state: dict = {"started": datetime.now().isoformat(), "steps": []}
    replanned = False
    step_idx = 0

    while step_idx < len(steps):
        step = steps[step_idx]
        print("\n" + "=" * 60)
        print(f"[PLAN] {step_idx + 1}/{len(steps)}: {step['id']} - {step['desc']}")
        print("=" * 60)

        result_name = step.get("result_name") or f"backtest_results_{step['id']}.json"
        pre_train_backup: Path | None = None

        orig_cfg = copy.deepcopy(cfg)
        verdict = None
        try:
            results, verdict, pre_train_backup, orig_cfg = _execute_step(
                step, cfg, baseline_results, champion_backup
            )
            _print_summary(results, step["id"])
            report = save_verdict_report(
                verdict, step["desc"], adopted_id or "going_v1", MODELS_DIR / result_name
            )
            print(f"[EVAL] {verdict.verdict}: {verdict.reason}")
            print(f"[REPORT] {report}")

            f2 = next((f for f in verdict.folds if f.fold == 2), None)
            if f2 and verdict.verdict == "PASS_PARTIAL":
                if best_partial is None or f2.sharpe > best_partial[1]:
                    best_partial = (step["id"], f2.sharpe)

            if verdict.verdict == "PASS_FULL" or step["type"] == "backtest_only":
                if step["type"] != "backtest_only":
                    full_pass = verdict.verdict == "PASS_FULL"
                    adopted_id = step["id"]
                    champion_backup = _backup_models(f"adopted_{step['id']}")
                    baseline_results = results
                    shutil.copy2(MODELS_DIR / result_name, BASELINE_RESULTS)
                    cfg = _load_config()
                    if step.get("feature_file"):
                        cfg["training"]["backtest_feature_file"] = step["feature_file"]
                    _persist_hp_adoption(cfg, step)
                    _save_config(cfg)
                    print(f"[ADOPT] champion: {step['id']}")
                else:
                    baseline_results = results
                    adopted_id = step["id"]
            elif pre_train_backup and step["type"] in ("feature_train", "hp_train"):
                _restore_models(pre_train_backup)
                print(f"[RESTORE] {pre_train_backup.name}")

            state["steps"].append(
                {"id": step["id"], "verdict": verdict.verdict, "adopted": verdict.adopted}
            )

        finally:
            if step["type"] != "backtest_only" and orig_cfg is not None:
                # calibration / HP は常に元に戻す
                if step.get("calibration_patch") or step.get("hp_patch"):
                    _save_config(orig_cfg)
                    cfg = orig_cfg
                elif verdict is not None and not verdict.adopted and step.get("feature_file"):
                    orig_cfg["training"]["backtest_feature_file"] = cfg["training"].get(
                        "backtest_feature_file", "features_v6_going_v1.parquet"
                    )
                    if adopted_id == "rebaseline_going_v1" or not adopted_id:
                        orig_cfg["training"]["backtest_feature_file"] = "features_v6_going_v1.parquet"
                    _save_config(orig_cfg)
                    cfg = orig_cfg

        step_idx += 1

        # 一次計画終了時に再計画
        if step_idx == len(INITIAL_PLAN) and not full_pass and not replanned:
            replanned = True
            steps.extend(_replan_steps(baseline_results))

    state_path = MODELS_DIR / "roadmap_state.json"
    state["finished"] = datetime.now().isoformat()
    state["full_pass"] = full_pass
    state["adopted"] = adopted_id
    state["best_partial"] = (
        {"id": best_partial[0], "f2_sharpe": best_partial[1]} if best_partial else None
    )
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("[DONE] ロードマップループ完了")
    print(f"  FULL PASS: {full_pass}")
    print(f"  採用 ID: {adopted_id}")
    print(f"  ベスト部分改善: {best_partial}")
    print(f"  state: {state_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", default=None)
    args = parser.parse_args()
    run_loop(from_step=args.from_step)


if __name__ == "__main__":
    main()
