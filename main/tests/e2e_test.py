"""E2Eテスト — 本番パイプラインの動作検証。

deployment-evaluatorフェーズ。
実行: python main/tests/e2e_test.py

テスト項目:
  1. 出馬表取得・必須カラム確認
  2. 馬体重の異常値・未発表処理
  3. 出走取消（スクラッチ）処理
  4. モデルロード・推論実行
  5. 結果保存の整合性（必須ファイル・カラム）
  6. EV/Kelly計算の数値整合性
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))
sys.path.insert(0, str(ROOT / "main"))
sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "main" / "results"

_PASSED = []
_FAILED = []


def _ok(name: str) -> None:
    print(f"  [PASS] {name}")
    _PASSED.append(name)


def _fail(name: str, reason: str) -> None:
    print(f"  [FAIL] {name}: {reason}")
    _FAILED.append(name)


# ---------------------------------------------------------------------------
# テスト1: 馬体重処理
# ---------------------------------------------------------------------------

def test_horse_weight_handling() -> None:
    print("\nTest 1: 馬体重処理")
    # NOTE: bare module import (sys.path に main/ を追加済み)。
    # `from main import ...` は e2e_test が main/ を sys.path に入れ
    # main.py モジュールが main パッケージを shadow するため失敗する。
    from race_runtime import validate_horse_weight

    cases = [
        (450, -4, "normal"),
        # 未発表（None/NaN）は除外扱い。validate_horse_weight の契約は
        # None→"excluded"（前走補完ではなくスキップ）。
        (None, None, "excluded"),
        (200, None, "abnormal"),
        (700, None, "abnormal"),
        (450, -30, "large_change"),
        (500, 2, "normal"),
    ]

    for weight, diff, expected in cases:
        result = validate_horse_weight(weight, diff)
        status = result["status"]
        name = f"体重={weight}, diff={diff} → {status}"
        if status == expected or (expected == "large_change" and status == "large_change"):
            _ok(name)
        else:
            _fail(name, f"期待={expected}, 実際={status}")


# ---------------------------------------------------------------------------
# テスト2: 出走取消処理
# ---------------------------------------------------------------------------

def test_scratch_handling() -> None:
    print("\nTest 2: 出走取消処理")
    from race_runtime import filter_scratched

    recs = pd.DataFrame({
        "race_id": ["R001", "R001", "R001"],
        "horse_id": ["H001", "H002", "H003"],
        "is_recommended": [True, True, True],
        "kelly_bet_yen": [500, 300, 200],
    })

    scratched = ["H002"]
    result = filter_scratched(recs, scratched)

    if "H002" not in result["horse_id"].values:
        _ok("取消馬H002が除外された")
    else:
        _fail("取消馬除外", "H002が残っている")

    if len(result) == 2:
        _ok(f"除外後の頭数が正しい: {len(result)}頭")
    else:
        _fail("除外後頭数", f"期待=2, 実際={len(result)}")

    # 取消なしの場合
    result2 = filter_scratched(recs, [])
    if len(result2) == 3:
        _ok("取消なしの場合は全馬保持")
    else:
        _fail("取消なし処理", f"期待=3, 実際={len(result2)}")


# ---------------------------------------------------------------------------
# テスト3: EV計算の数値整合性
# ---------------------------------------------------------------------------

def test_ev_calculation() -> None:
    print("\nTest 3: EV計算")
    from ev_calculator import calculate_ev, enrich_predictions

    # 勝率40%, オッズ3.0 → EV = 0.4 * 3.0 = 1.20
    ev = calculate_ev(0.40, 3.0)
    if abs(ev - 1.20) < 1e-9:
        _ok(f"EV計算正確: {ev:.4f} (期待=1.2000)")
    else:
        _fail("EV計算", f"期待=1.2000, 実際={ev:.4f}")

    # 負期待値: 勝率10%, オッズ5.0 → EV = 0.5 (負期待値)
    ev2 = calculate_ev(0.10, 5.0)
    if ev2 < 1.0:
        _ok(f"負期待値を正しく識別: EV={ev2:.4f}")
    else:
        _fail("負期待値識別", f"EV={ev2:.4f}が1.0以上（異常）")

    # enrich_predictions
    df = pd.DataFrame({
        "race_id": ["R001", "R001"],
        "horse_id": ["H001", "H002"],
        "model_prob": [0.30, 0.15],
        "odds": [4.0, 8.0],
    })
    enriched = enrich_predictions(df)
    required_cols = ["ev_rate", "implied_prob", "model_edge"]
    missing = [c for c in required_cols if c not in enriched.columns]
    if not missing:
        _ok("enrich_predictions: 必須カラム追加確認")
    else:
        _fail("enrich_predictions", f"カラム不足: {missing}")


# ---------------------------------------------------------------------------
# テスト4: Kelly計算の数値整合性
# ---------------------------------------------------------------------------

def test_kelly_sizing() -> None:
    print("\nTest 4: Kelly基準")
    from kelly_sizer import kelly_bet_amount, kelly_fraction

    # p=0.35, b=4.0 → full_kelly = (4*0.35 - 0.65)/4 = (1.4-0.65)/4 = 0.1875
    # fractional(0.08) = 0.1875 * 0.08 = 0.015
    frac = kelly_fraction(0.35, 5.0, kelly_frac=0.08, max_bet_ratio=0.05)
    expected_full = (4.0 * 0.35 - 0.65) / 4.0
    expected_frac = min(expected_full * 0.08, 0.05)
    if abs(frac - expected_frac) < 1e-9:
        _ok(f"Kelly比率計算正確: {frac:.6f} (期待={expected_frac:.6f})")
    else:
        _fail("Kelly比率計算", f"期待={expected_frac:.6f}, 実際={frac:.6f}")

    # 負のKellyはゼロになること
    frac_neg = kelly_fraction(0.05, 2.0, kelly_frac=0.08, max_bet_ratio=0.05)
    if frac_neg == 0.0:
        _ok("負のKellyは0に切り捨て")
    else:
        _fail("負のKelly処理", f"期待=0.0, 実際={frac_neg:.6f}")

    # max_bet_ratio制約
    frac_capped = kelly_fraction(0.80, 2.0, kelly_frac=0.08, max_bet_ratio=0.05)
    if frac_capped <= 0.05:
        _ok(f"max_bet_ratio上限適用: {frac_capped:.4f} <= 0.05")
    else:
        _fail("max_bet_ratio制約", f"{frac_capped:.4f} > 0.05")


# ---------------------------------------------------------------------------
# テスト5: Plackett-Luce確率変換
# ---------------------------------------------------------------------------

def test_plackett_luce() -> None:
    print("\nTest 5: Plackett-Luce確率")
    from plackett_luce import win_probabilities, place_probabilities_harville

    # 3頭のスコア
    scores = np.array([2.0, 1.0, 0.0])
    win_probs = win_probabilities(scores)

    if abs(win_probs.sum() - 1.0) < 1e-9:
        _ok(f"勝率の和=1.0: {win_probs.sum():.6f}")
    else:
        _fail("勝率の和", f"期待=1.0, 実際={win_probs.sum():.6f}")

    if win_probs[0] > win_probs[1] > win_probs[2]:
        _ok(f"順序が正しい: {win_probs[0]:.3f} > {win_probs[1]:.3f} > {win_probs[2]:.3f}")
    else:
        _fail("勝率順序", f"期待: 降順, 実際: {win_probs}")

    place_probs = place_probabilities_harville(win_probs)
    if all(0 <= p <= 1 for p in place_probs):
        _ok("複勝確率が[0,1]範囲内")
    else:
        _fail("複勝確率範囲", f"範囲外: {place_probs}")

    if all(p >= w for p, w in zip(place_probs, win_probs)):
        _ok("複勝確率 >= 単勝確率")
    else:
        _fail("複勝>=単勝", f"place={place_probs}, win={win_probs}")


# ---------------------------------------------------------------------------
# テスト6: 結果保存ファイルの確認
# ---------------------------------------------------------------------------

def test_result_storage() -> None:
    print("\nTest 6: 結果保存ファイル確認")
    required_files = [
        "today_recommendations.parquet",
        "today_recommendations.csv",
    ]

    for fname in required_files:
        fpath = RESULT_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            if size > 0:
                _ok(f"ファイル存在・非空: {fname} ({size}bytes)")
            else:
                _fail(f"空ファイル", fname)
        else:
            # まだパイプライン未実行の場合は情報扱い
            print(f"  [INFO] ファイル未生成（パイプライン未実行）: {fname}")

    # Parquetのカラム確認（存在する場合）
    pq_path = RESULT_DIR / "today_recommendations.parquet"
    if pq_path.exists():
        try:
            recs = pd.read_parquet(pq_path)
            # 本番出力（inference_pipeline）の実スキーマに合わせる。
            # 旧スキーマ（horse_id/model_prob/odds/ev_rate/kelly_bet_yen/is_recommended）は廃止済み。
            required_cols = [
                "ticket_type", "race_id", "horse_num", "pred_prob",
                "odds_raw", "odds_effective", "expected_value",
                "edge", "kelly_fraction", "suggested_stake", "is_executable",
            ]
            missing = [c for c in required_cols if c not in recs.columns]
            if not missing:
                _ok(f"推奨ファイルの必須カラム確認: {len(recs)}行")
            else:
                _fail("必須カラム確認", f"不足: {missing}")
        except Exception as e:
            _fail("Parquet読み込み", str(e))


# ---------------------------------------------------------------------------
# テスト7: モデルロード（モデルが存在する場合）
# ---------------------------------------------------------------------------

def _load_booster_crlf_safe(path: "Path", lgb) -> "object":
    """LightGBM モデルを CRLF 耐性ありでロードする。

    Windows / git autocrlf 経由でチェックアウトされた .txt モデルは行末が
    CRLF になることがあり、`lgb.Booster(model_file=...)` のツリーパーサが
    "expect a tree here" でネイティブ abort する（exit 127、try/except 不可）。
    バイト列を読んで LF 正規化し model_str で渡すと同一モデルを安全にロードできる。
    """
    raw = path.read_bytes()
    text = raw.replace(b"\r\n", b"\n").decode("utf-8")
    return lgb.Booster(model_str=text)


def test_model_loading() -> None:
    print("\nTest 7: モデルロード")
    try:
        from pipeline_common import MODELS_DIR
    except ImportError:
        from model_training.src.pipeline_common import MODELS_DIR
    import lightgbm as lgb

    # 現行の本番モデル: binary 5シードアンサンブル（バックテストと同一セット）
    for fold in [1, 2, 3]:
        seed_paths = sorted(MODELS_DIR.glob(f"lgbm_binary_fold{fold}_seed*.txt"))
        if not seed_paths:
            print(f"  [INFO] Fold {fold} binaryアンサンブル未学習（train.py実行後に確認）")
            continue
        if len(seed_paths) != 5:
            _fail(
                f"Fold {fold} アンサンブル構成",
                f"シードモデル数が期待値5と不一致: {len(seed_paths)}個",
            )
            continue
        try:
            models = [_load_booster_crlf_safe(p, lgb) for p in seed_paths]
            trees = [m.num_trees() for m in models]
            _ok(f"Fold {fold} binaryアンサンブルロード成功: {len(models)}モデル (trees={trees})")
        except Exception as e:
            _fail(f"Fold {fold} binaryアンサンブルロード", str(e))

    # strategy_engine.py は inference_common 経由で binary アンサンブルを使用する
    # （旧 lambdarank パスは廃止済み。残存ファイルがあっても本番では使われない）


# ---------------------------------------------------------------------------
# テスト8: strategy_config 整合性（Phase 0）
# ---------------------------------------------------------------------------

def test_strategy_config_alignment() -> None:
    """P0-1/3/4: strategy_config.json が train_config と整合していること。"""
    print("\nTest 8: strategy_config 整合性")
    cfg_path = ROOT / "strategy" / "config" / "strategy_config.json"
    train_path = ROOT / "model_training" / "config" / "train_config.json"

    if not cfg_path.exists():
        _fail("strategy_config 存在", str(cfg_path))
        return

    import json

    with cfg_path.open(encoding="utf-8") as f:
        sc = json.load(f)
    with train_path.open(encoding="utf-8") as f:
        tc = json.load(f)["training"]

    checks = [
        ("ev_threshold", sc.get("ev_threshold"), tc.get("ev_threshold"), 1.05),
        ("min_odds", sc.get("min_odds"), tc.get("min_odds"), 2.0),
        ("max_picks_per_race", sc.get("max_picks_per_race"), tc.get("max_picks_per_race"), 2),
        ("max_selections_per_race", sc.get("max_selections_per_race"), tc.get("max_picks_per_race"), 2),
        ("kelly_fraction", sc.get("kelly_fraction"), tc.get("kelly_fraction"), 0.08),
    ]
    for name, s_val, t_val, expected in checks:
        if s_val == expected and t_val == expected:
            _ok(f"{name}={s_val} (strategy=train={expected})")
        else:
            _fail(name, f"strategy={s_val}, train={t_val}, 期待={expected}")

    if "conditional_ev_overrides" in sc:
        _fail("キー名", "conditional_ev_overrides が残存（condition_ev_overrides に統一すること）")
    elif "condition_ev_overrides" in sc:
        _ok("condition_ev_overrides キー名（s なし）")

    combo_keys = ("pair_top_n", "wide_top_n", "rank2_blend", "wide_min_edge", "max_expected_value")
    for key in combo_keys:
        if key in sc:
            _ok(f"combo {key}={sc[key]}")
        else:
            _fail(f"combo {key}", "strategy_config に未定義")

    bands = sc.get("dynamic_edge_bands") or []
    high_band = next((b for b in bands if float(b.get("odds_min", 0)) >= 12.0), None)
    if high_band and float(high_band.get("min_edge", 0)) >= 0.12:
        _ok(f"12倍超 min_edge={high_band['min_edge']}")
    else:
        _fail("12倍超 dynamic_edge", str(high_band))

    from main.pipeline.strategy_pipeline import resolve_strategy_calibration_path

    cal_path = resolve_strategy_calibration_path(ROOT, cfg_path)
    if cal_path.is_file():
        _ok(f"calibrator resolved: {cal_path.name}")
    else:
        _fail("calibrator path", str(cal_path))

    if "specv2" in str(sc.get("calibration_path", "")):
        _ok("calibration_path=specv2")
    else:
        _fail("calibration_path", sc.get("calibration_path"))

    if sc.get("online_phase") == "phase1_5":
        _ok("online_phase=phase1_5 (C1 wide EV)")
    else:
        _fail("online_phase", sc.get("online_phase"))

    if sc.get("wide_bets_enabled") is True and sc.get("quinella_bets_enabled") is False:
        _ok("C1: wide ON / quinella OFF")
    else:
        _fail("combo ticket flags", f"wide={sc.get('wide_bets_enabled')} quinella={sc.get('quinella_bets_enabled')}")


def test_wide_phase15_recommendation_smoke() -> None:
    """phase1_5 + O3 オッズでワイド EV 推奨が生成されること。"""
    print("\nTest 11: wide phase1_5 recommendation smoke")
    from strategy.src.betting_framework import ProbabilityCalibrator, run_today_recommendation
    from main.pipeline.strategy_pipeline import (
        load_strategy_runtime_config,
        resolve_strategy_calibration_path,
        strategy_config_from_runtime,
    )

    runtime = load_strategy_runtime_config(ROOT / "strategy" / "config" / "strategy_config.json")
    cfg = strategy_config_from_runtime(runtime)
    cal_path = resolve_strategy_calibration_path(ROOT)
    cal = ProbabilityCalibrator.from_json(cal_path) if cal_path.is_file() else None

    pred = pd.DataFrame({
        "race_id": ["202506170101", "202506170101", "202506170101", "202506170101"],
        "horse_num": [1, 2, 3, 4],
        "pred_rank1": [0.35, 0.25, 0.20, 0.10],
        "pred_rank2": [0.30, 0.28, 0.22, 0.12],
        "pred_rank3": [0.28, 0.26, 0.24, 0.14],
        "odds": [3.5, 5.0, 8.0, 12.0],
        "n_horses": [16, 16, 16, 16],
    })
    wide_odds = {
        ("202506170101", "01", "02"): 8.0,
        ("202506170101", "01", "03"): 10.0,
    }
    rec = run_today_recommendation(
        pred,
        config=cfg,
        calibrator=cal,
        phase="phase1_5",
        wide_bets_enabled=True,
        quinella_bets_enabled=False,
        place_bets_enabled=False,
        wide_min_edge=0.05,
        max_expected_value=float(runtime.get("max_expected_value", 5.0)),
        wide_odds_dict=wide_odds,
    )
    wide = rec[rec["ticket_type"] == "ワイド"] if not rec.empty else rec
    win = rec[rec["ticket_type"] == "単勝"] if not rec.empty else rec
    if not wide.empty and wide["expected_value"].notna().any():
        _ok(f"wide rows={len(wide)} ev>0 sample={wide['expected_value'].max():.2f}")
    else:
        _fail("wide phase1_5", f"rows={len(rec)} wide={len(wide)}")
    quinella = rec[rec["ticket_type"] == "馬連"] if not rec.empty else rec
    if len(quinella) == 0:
        _ok("quinella suppressed (C1)")
    else:
        _fail("quinella should be off", str(len(quinella)))
    if not win.empty:
        _ok(f"win rows={len(win)} preserved (Track A)")
    else:
        _fail("win bets missing", str(rec.columns.tolist()))


def test_portfolio_kelly_smoke() -> None:
    """C2: portfolio_kelly 有効時、同一レース win+wide の stake 合計が cap 以内。"""
    print("\nTest 12: portfolio Kelly win+wide smoke")
    from strategy.src.betting_framework import ProbabilityCalibrator, run_today_recommendation
    from main.pipeline.strategy_pipeline import (
        load_strategy_runtime_config,
        resolve_strategy_calibration_path,
        strategy_config_from_runtime,
    )

    runtime = load_strategy_runtime_config(ROOT / "strategy" / "config" / "strategy_config.json")
    cfg = strategy_config_from_runtime(runtime)
    cal_path = resolve_strategy_calibration_path(ROOT)
    cal = ProbabilityCalibrator.from_json(cal_path) if cal_path.is_file() else None
    max_invest = int(runtime.get("max_invest_per_race", 50_000))

    pred = pd.DataFrame({
        "race_id": ["202506170101"] * 4,
        "horse_num": [1, 2, 3, 4],
        "pred_rank1": [0.35, 0.25, 0.20, 0.10],
        "pred_rank2": [0.30, 0.28, 0.22, 0.12],
        "pred_rank3": [0.28, 0.26, 0.24, 0.14],
        "odds": [3.5, 5.0, 8.0, 12.0],
        "n_horses": [16, 16, 16, 16],
    })
    wide_odds = {
        ("202506170101", "01", "02"): 8.0,
        ("202506170101", "01", "03"): 10.0,
    }
    rec = run_today_recommendation(
        pred,
        config=cfg,
        calibrator=cal,
        phase="phase1_5",
        wide_bets_enabled=True,
        quinella_bets_enabled=False,
        place_bets_enabled=False,
        wide_min_edge=0.05,
        max_expected_value=float(runtime.get("max_expected_value", 5.0)),
        wide_odds_dict=wide_odds,
        portfolio_kelly_enabled=bool(runtime.get("portfolio_kelly_enabled", True)),
        portfolio_kelly_mode=str(runtime.get("portfolio_kelly_mode", "portfolio_kelly_fractional")),
        portfolio_growth_ratio_min=float(runtime.get("portfolio_growth_ratio_min", 0.5)),
        portfolio_ind_cap_ratio=float(runtime.get("portfolio_ind_cap_ratio", 0.85)),
    )
    if rec.empty:
        _fail("portfolio kelly", "no recommendations")
        return
    race_total = int(rec.groupby("race_id")["suggested_stake"].sum().max())
    if race_total <= max_invest:
        _ok(f"race stake total={race_total} <= cap={max_invest}")
    else:
        _fail("portfolio cap", f"total={race_total} cap={max_invest}")
    has_win = (rec["ticket_type"] == "単勝").any()
    has_wide = (rec["ticket_type"] == "ワイド").any()
    if has_win and has_wide:
        _ok("win+wide same race")
    else:
        _fail("win+wide mix", f"win={has_win} wide={has_wide}")


def test_combo_anchor_kpi_smoke() -> None:
    """ワイド/馬連 anchor KPI が evaluation モジュールで計算できること。"""
    print("\nTest 10: combo anchor KPI smoke")
    from evaluation import calculate_ranking_metrics

    df = pd.DataFrame({
        "race_id": ["R1", "R1", "R1", "R2", "R2", "R2"],
        "horse_num": [1, 2, 3, 1, 2, 3],
        "finish_rank": [1, 2, 5, 3, 1, 2],
        "model_prob": [0.5, 0.3, 0.2, 0.4, 0.35, 0.25],
    })
    m = calculate_ranking_metrics(df)
    if m.get("wide_anchor_any", 0) >= 0.5:
        _ok(f"wide_anchor_any={m['wide_anchor_any']:.0%}")
    else:
        _fail("wide_anchor_any", str(m))
    if "quinella_anchor_any" in m:
        _ok(f"quinella_anchor_any={m['quinella_anchor_any']:.0%}")
    else:
        _fail("quinella_anchor_any", "列なし")


# ---------------------------------------------------------------------------
# テスト9: monthly_dd_tracker（Phase 0 P0-5）
# ---------------------------------------------------------------------------

def test_monthly_dd_tracker() -> None:
    """P0-5: 月次DDトラッカーの読み書き・閾値判定。"""
    print("\nTest 9: monthly_dd_tracker")
    import tempfile
    from datetime import date, timedelta
    from main.pipeline import monthly_dd_tracker as mdt

    ym = date.today().strftime("%Y-%m")
    d1 = f"{ym}-01"
    d2 = f"{ym}-02"
    d3 = f"{ym}-03"

    with tempfile.TemporaryDirectory() as tmp:
        tracker = Path(tmp) / "monthly_pnl_tracker.json"
        orig = mdt.TRACKER_FILE
        mdt.TRACKER_FILE = tracker
        try:
            mdt.record_daily_pnl(d1, invested=10000.0, returned=0.0, n_recommendations=4, n_hits=1)
            mdt.record_daily_pnl(d2, invested=5000.0, returned=8000.0, n_recommendations=2, n_hits=0)
            exceeded, rate = mdt.check_monthly_dd_limit(
                initial_bankroll=100_000.0,
                monthly_drawdown_limit=-0.08,
            )
            if not exceeded:
                _ok(f"通常損益でDD未超過 (rate={rate:.2%})")
            else:
                _fail("DD判定", f"想定外超過 rate={rate}")

            # 大損失で超過
            mdt.record_daily_pnl(d3, invested=20000.0, returned=0.0)
            exceeded2, rate2 = mdt.check_monthly_dd_limit(
                initial_bankroll=100_000.0,
                monthly_drawdown_limit=-0.08,
            )
            if exceeded2:
                _ok(f"大損失でDD超過検知 (rate={rate2:.2%})")
            else:
                _fail("DD超過検知", f"rate={rate2}")

            alert_info = mdt.check_hit_rate_and_roi_alerts(
                hit_rate_floor=0.25, roi_floor=1.15, min_bets_for_alert=3
            )
            if alert_info.get("alerts"):
                _ok(f"的中率/ROIアラート検知: {alert_info['alerts']}")
            else:
                _fail("的中率/ROIアラート", "期待する警告が発生しなかった")

            summary = mdt.get_monthly_summary(ym)
            if len(summary) >= 3 and "cumulative_profit" in summary.columns:
                _ok(f"月次サマリー {len(summary)} 日分")
            else:
                _fail("月次サマリー", str(summary.columns.tolist()))
        finally:
            mdt.TRACKER_FILE = orig


def test_baba_scenario_feature_recompute() -> None:
    """Phase 1: what-if 馬場シナリオで imputed / change 特徴量が更新される。"""
    print("\nTest 13: baba scenario feature recompute")
    import numpy as np
    import pandas as pd
    from main.pipeline.inference_pipeline import apply_uniform_baba_jv_code

    df = pd.DataFrame(
        [
            {
                "track_code": 11,
                "turf_condition": 3.0,
                "dirt_condition": 0.0,
                "track_condition_code": 3.0,
                "horse_turf_heavy_win_rate": 0.10,
                "horse_turf_very_heavy_win_rate": 0.20,
                "horse_turf_light_win_rate": 0.30,
                "horse_turf_soft_win_rate": 0.15,
                "going_match_score_turf_imputed": 1.5,
                "going_change_lag1": 1.0,
                "going_worsening_flag": 0,
            }
        ]
    )
    out = apply_uniform_baba_jv_code(df, 4)
    if not np.isclose(out["going_match_score_turf_imputed"].iloc[0], 0.20):
        _fail("imputed sync", str(out["going_match_score_turf_imputed"].iloc[0]))
    else:
        _ok("going_match_score_turf_imputed synced on jv=4")
    if not np.isclose(out["going_change_lag1"].iloc[0], 2.0):
        _fail("going_change_lag1", str(out["going_change_lag1"].iloc[0]))
    else:
        _ok("going_change_lag1 updated on scenario shift")


def test_baba_inference_latency_soft() -> None:
    """Phase 1-5: 4シナリオ apply_uniform_baba_jv_code の soft レイテンシ監視。"""
    print("\nTest 14: baba inference latency (soft)")
    import time
    import pandas as pd
    from main.pipeline.inference_pipeline import apply_uniform_baba_jv_code

    n = 500
    df = pd.DataFrame(
        {
            "track_code": [11] * n,
            "turf_condition": [0.0] * n,
            "dirt_condition": [0.0] * n,
            "horse_turf_heavy_win_rate": [0.1] * n,
            "horse_turf_very_heavy_win_rate": [0.2] * n,
            "horse_turf_light_win_rate": [0.3] * n,
            "horse_turf_soft_win_rate": [0.15] * n,
            "going_match_score_turf_imputed": [1.0] * n,
            "going_change_lag1": [0.0] * n,
            "going_worsening_flag": [0] * n,
        }
    )
    t0 = time.perf_counter()
    for jv in (1, 2, 3, 4):
        apply_uniform_baba_jv_code(df, jv)
    elapsed = time.perf_counter() - t0
    if elapsed > 30.0:
        print(f"  [WARN] baba recompute 4x500 rows took {elapsed:.2f}s (>30s soft limit)")
    _ok(f"baba recompute latency {elapsed:.3f}s (soft gate 30s for feature-only path)")


# ---------------------------------------------------------------------------
# テストランナー
# ---------------------------------------------------------------------------

def run_all_tests() -> bool:
    print("=" * 50)
    print("E2Eテスト開始")
    print("=" * 50)

    test_horse_weight_handling()
    test_scratch_handling()
    test_ev_calculation()
    test_kelly_sizing()
    test_plackett_luce()
    test_result_storage()
    test_model_loading()
    test_strategy_config_alignment()
    test_wide_phase15_recommendation_smoke()
    test_portfolio_kelly_smoke()
    test_combo_anchor_kpi_smoke()
    test_monthly_dd_tracker()
    test_baba_scenario_feature_recompute()
    test_baba_inference_latency_soft()

    print("\n" + "=" * 50)
    print(f"結果: [PASS]{len(_PASSED)}件 / [FAIL]{len(_FAILED)}件")
    if _FAILED:
        print("失敗テスト:")
        for f in _FAILED:
            print(f"  - {f}")
    print("=" * 50)

    return len(_FAILED) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
