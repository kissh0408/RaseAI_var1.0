"""Walk-Forward Validationバックテスト実行エンジン。

backtest-evaluatorフェーズ。
binaryアンサンブルモデルの各フォールドをテスト期間で評価し、
合格基準（ROI>=1.05, MDD>=-0.20, Sharpe>=0.10）を確認する。

実行: python strategy/src/backtest.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from bet_tuning import tune_bet_params_on_valid
from calibration import get_raw_scores
from evaluation import (
    analyze_by_conditions,
    calculate_drawdown,
    calculate_log_loss,
    calculate_ranking_metrics,
    calculate_residual_ic,
    calculate_roi_metrics,
    detect_overfitting,
    detect_ranking_overfitting,
    generate_evaluation_report,
    grade_metrics,
)
from ev_calculator import apply_ev_filters, enrich_predictions
from ev_filters import ev_filter_config_from_mapping, effective_min_edge
from inference_common import (
    apply_condition_overrides,
    apply_max_picks_per_race,
    apply_race_budget_cap,
    apply_var1_market_blend_probs,
    load_ensemble_models,
    normalize_within_race,
    predict_model_probs,
)
from kelly_sizer import apply_kelly_sizing
# NOTE: get_db_connection は撤去。オッズは parquet 同梱（_odds）を使うため DB 非依存。
from pipeline_common import FEATURES_DIR, MODELS_DIR, load_config
from plackett_luce import tune_temperature
from train import get_feature_cols


def load_features_with_odds() -> pd.DataFrame:
    """active feature_file（無ければ features_v4>v3>v2 後方互換）を読み込み、
    SE のオッズと is_win を付与して返す。

    NOTE: 旧実装は features_v6/v4/v3/v2 のみ探索していたが、現行リポジトリの実体は
    train_config.json の training.feature_file が指す features_past_v*.parquet。
    ImportError 復旧時に参照崩れを防ぐため、active file を最優先で読む。
    features_v5 はバックテスト悪化によりリジェクト済み（train.py 参照）。
    """
    feat_path = None
    t_cfg = load_config().get("training", {})
    # 1) backtest 専用の特徴量ファイルを最優先で解決する。
    #    理由: training.feature_file は rank/本番(main/pipeline)が掴む 352列ファイル
    #    (features_past_v25_odds)であり、binary 検証モデル(57 feature_cols)とは別系統。
    #    両者は意図的に共存する2系統(CLAUDE.md「確定アーキテクチャ」)のため、binary 検証
    #    パスは training.backtest_feature_file で独立解決し、rank パスの feature_file 解決には
    #    一切影響を与えない。
    backtest_feature_file = t_cfg.get("backtest_feature_file")
    if backtest_feature_file:
        p = FEATURES_DIR / backtest_feature_file
        if p.exists():
            feat_path = p
    # 2) backtest 専用キーが無い場合のみ、従来どおり active feature_file を参照
    if feat_path is None:
        feature_file = t_cfg.get("feature_file")
        if feature_file:
            p = FEATURES_DIR / feature_file
            if p.exists():
                feat_path = p
    # 3) 後方互換: 旧 features_v* を探索（v8/v7/v5 はリジェクト）
    if feat_path is None:
        for ver in ["v6", "v4", "v3", "v2"]:
            p = FEATURES_DIR / f"features_{ver}.parquet"
            if p.exists():
                feat_path = p
                break
    if feat_path is None:
        raise FileNotFoundError(
            "feature parquet が見つかりません。train_config.json の "
            "training.feature_file を確認するか、先に特徴量生成を実行してください。"
        )

    df = pd.read_parquet(feat_path)
    print(f"  {feat_path.name} loaded: {len(df)} rows")

    # スキーマ整合: 現行 active ファイル(features_past_v25_odds)は race_date/horse_id を
    # 直接持たず、date(datetime) と ketto_num を持つ。旧 features_v4 スキーマ
    # (race_date/horse_id) を前提とする backtest ランタイムに合わせて列を導出する。
    # 理由: ローダ層で列名を正規化し、EV/Kelly/正規化ロジックには一切触れない。
    if "race_date" not in df.columns:
        if "date" in df.columns:
            df["race_date"] = pd.to_datetime(df["date"])
        elif "year" in df.columns and "month_day" in df.columns:
            # month_day は MDD 形式の int（例: 104 -> 1月4日）。year と合成して日付化
            md = df["month_day"].astype(int)
            df["race_date"] = pd.to_datetime(
                {
                    "year": df["year"].astype(int),
                    "month": md // 100,
                    "day": md % 100,
                },
                errors="coerce",
            )
        else:
            raise KeyError("race_date を導出できません（date / year+month_day が無い）")
    else:
        df["race_date"] = pd.to_datetime(df["race_date"])

    if "horse_id" not in df.columns and "ketto_num" in df.columns:
        # ketto_num（血統登録番号）が馬の一意キー。本番 rank 経路と同一の識別子。
        df["horse_id"] = df["ketto_num"].astype(str)

    # オッズ取得元: ファイル名が _odds の通り、単勝オッズは parquet に同梱済み。
    # 旧実装は DB(SE テーブル)へ問い合わせていたが、common/data/JVData.db は本環境に
    # 存在せず KeyError/接続失敗の原因だった。parquet 内の odds を直接使うことで
    # DB 非依存にする（DB 接続コードは撤去）。
    if "odds" not in df.columns:
        raise KeyError(
            f"{feat_path.name} に odds 列がありません。オッズ同梱の特徴量ファイルが必要です"
        )

    # base_margin（市場 log-odds）はオッズから生成。parquet に market_log_odds が
    # 無い場合のみ inference_common と同一ロジックで補完（builders/basic.py と同式）。
    if "market_log_odds" not in df.columns:
        from inference_common import compute_market_log_odds

        df["race_id"] = df["race_id"].astype(str)
        df = compute_market_log_odds(df, odds_col="odds")

    df["is_win"] = (df["finish_rank"] == 1).astype(int) if "finish_rank" in df.columns else 0

    return df.sort_values(["race_date", "race_id"]).reset_index(drop=True)


def run_fold_backtest(
    fold: int,
    df_all: pd.DataFrame,
    fold_cfg: dict,
    feature_cols: list[str],
    cfg: dict,
) -> dict:
    """1フォールドのバックテストを実行し、評価指標を返す。"""
    t_cfg = cfg["training"]
    train_end = pd.Timestamp(fold_cfg["train_end"])
    valid_start = pd.Timestamp(fold_cfg["valid_start"])
    valid_end = pd.Timestamp(fold_cfg.get("valid_end", fold_cfg.get("test_start", "")))
    test_start = pd.Timestamp(fold_cfg["test_start"])
    test_end = pd.Timestamp(fold_cfg["test_end"])

    train_df = df_all[df_all["race_date"] <= train_end].copy()
    # non-overlapping: バリデーション期間は学習期間外
    valid_df = df_all[
        (df_all["race_date"] >= valid_start) & (df_all["race_date"] <= valid_end)
    ].copy()
    test_df = df_all[(df_all["race_date"] >= test_start) & (df_all["race_date"] <= test_end)].copy()

    if len(test_df) == 0:
        return {"error": f"Fold {fold}: テスト期間にデータなし"}

    models = load_ensemble_models(fold)
    model = models[0]  # 代表モデル（temperature tune 用）
    # binary 検証モデルは学習時の feature_cols(57列)を model file 内に保持している。
    # 現行 train_config から再導出した feature_cols(52列)とは一致せず、その差分を
    # predict に渡すと LightGBM が shape mismatch (52 != 57) で落ちる。
    # ground truth は「各モデルが学習に使った列」なので、model.feature_name() を
    # 一次ソースとして available を解決する（推論の数式・前処理には一切触れない）。
    model_feature_cols = list(model.feature_name()) if hasattr(model, "feature_name") else []
    resolved_cols = model_feature_cols or feature_cols
    available = [c for c in resolved_cols if c in df_all.columns]
    base_margin_col = t_cfg.get("base_margin_col")

    def _apply_init_score_and_normalize(df_period: pd.DataFrame) -> pd.Series:
        """共有推論パス（inference_common）でアンサンブル確率を計算する。"""
        return predict_model_probs(
            models, df_period, available, base_margin_col, temperature=temperature, t_cfg=t_cfg
        )

    # フェーズ5: 不確実性Skip。閾値は学習期間の候補ベットから導出（テスト期間で後出ししない）
    unc_cfg = t_cfg.get("uncertainty_skip", {})
    unc_state = {"threshold": None}

    # 温度パラメータ: バリデーションデータで最適化
    cal_cfg = t_cfg.get("calibration", {})
    temperature = float(cal_cfg.get("temperature", 0.8))
    if cal_cfg.get("temperature_tune", False) and len(valid_df) > 0:
        try:
            valid_scores = get_raw_scores(model, valid_df, available)
            valid_with_scores = valid_df.copy()
            valid_with_scores["raw_score"] = valid_scores
            valid_with_scores["is_win"] = (valid_with_scores["finish_rank"] == 1).astype(int)
            t_range = tuple(cal_cfg.get("temperature_range", [0.8, 3.0]))
            temperature = tune_temperature(
                valid_with_scores, score_col="raw_score", label_col="is_win", t_range=t_range
            )
        except Exception as e:
            print(f"  [WARN] 温度チューニング失敗: {e}。デフォルト T={temperature} を使用")

    print(f"  温度パラメータ T={temperature:.3f}")

    # Isotonic Regression 較正器: バリデーションデータのみで学習（テスト期間リーク防止）
    _iso_calibrator = None
    if len(valid_df) > 0 and cal_cfg.get("isotonic", True):
        try:
            from sklearn.isotonic import IsotonicRegression as _IR
            _iso_calibrator = _IR(out_of_bounds="clip")
            _prob_norm = _apply_init_score_and_normalize(valid_df).values
            _is_win_v = (valid_df["finish_rank"] == 1).astype(int).values
            _iso_calibrator.fit(_prob_norm, _is_win_v)
            print(f"  Isotonic較正器学習完了 (valid n={len(valid_df)})")
        except Exception as _e:
            print(f"  [WARN] Isotonic較正失敗: {_e}")
            _iso_calibrator = None

    def _predict_and_recommend(
        df_period: pd.DataFrame,
        *,
        ev_threshold: float | None = None,
        max_picks: int | None = None,
    ) -> pd.DataFrame:
        if len(df_period) == 0:
            return pd.DataFrame()
        df_period = df_period.copy()

        ev_thr = float(ev_threshold if ev_threshold is not None else t_cfg["ev_threshold"])
        max_pick = int(max_picks if max_picks is not None else t_cfg.get("max_picks_per_race", 2))

        # binary / group_softmax モデル: init_score 補正 + レース内正規化
        if unc_cfg.get("enabled", False):
            from inference_common import predict_with_uncertainty
            probs, unc = predict_with_uncertainty(models, df_period, available, base_margin_col)
            df_period["model_prob"] = probs
            df_period["pred_uncertainty"] = unc / probs.clip(lower=1e-6)
        else:
            df_period["model_prob"] = _apply_init_score_and_normalize(df_period)

        # Isotonic 較正をテスト期間に適用してレース内再正規化
        if _iso_calibrator is not None:
            try:
                _cal_prob = _iso_calibrator.predict(df_period["model_prob"].values)
                df_period["model_prob"] = normalize_within_race(_cal_prob, df_period)
            except Exception:
                pass

        # var1 市場残差ブレンド → EV 計算用 ev_prob（ランキング指標は model_prob のまま）
        blend_cfg = t_cfg.get("var1_market_blend", {})
        prob_col = "model_prob"
        if blend_cfg.get("enabled", False) and "var1_pure_score_z" in df_period.columns:
            beta = float(blend_cfg.get("beta", 0.30))
            df_period["ev_prob"] = apply_var1_market_blend_probs(
                df_period, beta=beta, z_col="var1_pure_score_z"
            )
            prob_col = "ev_prob"

        df_period = enrich_predictions(df_period, model_prob_col=prob_col, odds_col="odds")
        df_period = apply_kelly_sizing(
            df_period,
            bankroll=10_000_000,
            kelly_frac=t_cfg["kelly_fraction"],
            max_bet_ratio=t_cfg["max_bet_ratio"],
        )
        base_mask = apply_ev_filters(
            df_period,
            ev_threshold=ev_thr,
            min_odds=t_cfg["min_odds"],
            max_odds=t_cfg["max_odds"],
            min_model_prob=t_cfg["min_model_prob"],
            model_prob_col=prob_col,
        ) & (df_period["kelly_bet_yen"] > 0)

        if t_cfg.get("dynamic_edge_enabled", False):
            ev_cfg = ev_filter_config_from_mapping(t_cfg)
            req_edge = effective_min_edge(df_period["odds"].values, ev_cfg)
            base_mask = base_mask & pd.Series(
                df_period["model_edge"].values >= req_edge, index=df_period.index
            )

        base_mask = apply_condition_overrides(
            df_period, base_mask, cfg.get("condition_ev_overrides", []), ev_thr
        )

        if max_pick > 0:
            base_mask = apply_max_picks_per_race(df_period, base_mask, max_pick)

        if unc_cfg.get("enabled", False) and "pred_uncertainty" in df_period.columns:
            if unc_state["threshold"] is None:
                cand = df_period.loc[base_mask, "pred_uncertainty"].dropna()
                q = float(unc_cfg.get("quantile", 0.9))
                unc_state["threshold"] = float(cand.quantile(q)) if len(cand) else np.inf
                print(f"  不確実性Skip閾値（学習期間 q={q}）: {unc_state['threshold']:.4f}")
            base_mask = base_mask & (df_period["pred_uncertainty"] <= unc_state["threshold"])

        excl_surface = t_cfg.get("exclude_surface_codes", [3])
        if excl_surface and "surface_code" in df_period.columns:
            base_mask = base_mask & ~df_period["surface_code"].isin(excl_surface)

        excl_abnormal = t_cfg.get("exclude_abnormal_codes", [])
        if excl_abnormal and "abnormal_code" in df_period.columns:
            base_mask = base_mask & ~df_period["abnormal_code"].isin(excl_abnormal)
        min_hc = t_cfg.get("min_horse_count")
        if min_hc and "horse_count" in df_period.columns:
            base_mask = base_mask & (df_period["horse_count"] >= min_hc)

        df_period["is_recommended"] = base_mask
        df_period = apply_race_budget_cap(
            df_period, base_mask, t_cfg["max_bet_ratio"], bankroll=10_000_000
        )
        return df_period

    # VALID のみで EV / max_picks をチューニング（Rule 3）
    bet_tune_cfg = t_cfg.get("bet_tuning", {})
    tuned_params: dict = {
        "ev_threshold": t_cfg["ev_threshold"],
        "max_picks_per_race": int(t_cfg.get("max_picks_per_race", 2)),
        "tuning_fallback": True,
    }
    if bet_tune_cfg.get("enabled", False) and len(valid_df) > 0:
        min_by_fold = bet_tune_cfg.get("min_valid_bets_by_fold") or {}
        min_n = int(min_by_fold.get(str(fold), min_by_fold.get(fold, bet_tune_cfg.get("min_valid_bets", 100))))
        tuned_params = tune_bet_params_on_valid(
            valid_df,
            _predict_and_recommend,
            default_ev_threshold=t_cfg["ev_threshold"],
            default_max_picks=int(t_cfg.get("max_picks_per_race", 2)),
            ev_threshold_grid=bet_tune_cfg.get(
                "ev_threshold_grid", [1.0, 1.05, 1.08, 1.10, 1.15, 1.20]
            ),
            max_picks_grid=bet_tune_cfg.get("max_picks_grid", [1, 2, 3]),
            min_valid_bets=min_n,
            objective=str(bet_tune_cfg.get("objective", "roi")),
            sharpe_tie_delta=float(bet_tune_cfg.get("sharpe_tie_delta", 0.01)),
        )
        print(
            f"  [bet_tuning] VALID 選定 ({bet_tune_cfg.get('objective', 'roi')}): "
            f"ev_threshold={tuned_params['ev_threshold']}, "
            f"max_picks={tuned_params['max_picks_per_race']}, "
            f"valid_sharpe={tuned_params.get('valid_sharpe')}, "
            f"valid_roi={tuned_params.get('valid_roi')}, "
            f"n={tuned_params.get('valid_n_bets')}, "
            f"min_n={min_n}, fallback={tuned_params.get('tuning_fallback')}"
        )

    train_pred = _predict_and_recommend(train_df)
    valid_pred = _predict_and_recommend(
        valid_df,
        ev_threshold=tuned_params["ev_threshold"],
        max_picks=tuned_params["max_picks_per_race"],
    )
    test_pred = _predict_and_recommend(
        test_df,
        ev_threshold=tuned_params["ev_threshold"],
        max_picks=tuned_params["max_picks_per_race"],
    )

    train_metrics = calculate_roi_metrics(train_pred) if len(train_pred) > 0 else {}
    valid_metrics = calculate_roi_metrics(valid_pred) if len(valid_pred) > 0 else {}
    test_metrics = calculate_roi_metrics(test_pred)
    drawdown_metrics = calculate_drawdown(test_pred)
    overfitting = detect_overfitting(train_metrics, valid_metrics, test_metrics)
    conditions = analyze_by_conditions(test_pred) if "track_condition_code" in test_pred.columns else {}

    # 戦略なし順位指標（全頭に model_prob 付与済み → EV フィルタ前）
    train_rank = calculate_ranking_metrics(train_pred) if len(train_pred) > 0 else {}
    valid_rank = calculate_ranking_metrics(valid_pred) if len(valid_pred) > 0 else {}
    test_rank = calculate_ranking_metrics(test_pred) if len(test_pred) > 0 else {}
    ranking_overfit = detect_ranking_overfitting(train_rank, valid_rank, test_rank)

    # Residual IC + Log Loss
    ic = calculate_residual_ic(test_pred, win_col="is_win") if len(test_pred) > 0 else {}
    ll = calculate_log_loss(test_pred, win_col="is_win") if len(test_pred) > 0 else {}

    report = generate_evaluation_report(
        fold, test_metrics, drawdown_metrics, overfitting, conditions,
        residual_ic=ic, log_loss=ll,
    )
    print(report)
    print()

    grades = grade_metrics(test_metrics, drawdown_metrics)
    print(
        f"  [Ranking] test top3={test_rank.get('top3_overlap_rate', float('nan')):.1%} "
        f"top1={test_rank.get('top1_win_rate', float('nan')):.1%} "
        f"box3={test_rank.get('top3_box_rate', float('nan')):.1%} | "
        f"wide_anchor={test_rank.get('wide_anchor_any', float('nan')):.1%} "
        f"quinella_anchor={test_rank.get('quinella_anchor_any', float('nan')):.1%} | "
        f"logloss={ll.get('model_log_loss', float('nan')):.4f}"
    )
    return {
        "fold": fold,
        "temperature": temperature,
        "uncertainty_threshold": unc_state["threshold"],
        "test_metrics": test_metrics,
        "drawdown_metrics": drawdown_metrics,
        "overfitting_check": overfitting,
        "ranking_metrics": {
            "train": train_rank,
            "valid": valid_rank,
            "test": test_rank,
        },
        "ranking_overfitting_check": ranking_overfit,
        "condition_analysis": conditions,
        "residual_ic": ic,
        "log_loss": ll,
        "grades": grades,
        "bet_tuning": tuned_params,
        "var1_market_blend": t_cfg.get("var1_market_blend", {}),
        "test_predictions": test_pred,
    }


def run_full_backtest() -> list[dict]:
    """全フォールドのバックテストを実行し、結果レポートを出力する。"""
    cfg = load_config()
    feature_cols = get_feature_cols(cfg)

    print("Loading data with odds...")
    df_all = load_features_with_odds()
    print(f"  Total: {len(df_all)} rows, odds available: {df_all['odds'].notna().sum()}")

    all_results = []
    for fold_cfg in cfg["training"]["walkforward_folds"]:
        fold = fold_cfg["fold"]
        print(f"\n{'='*50}")
        print(f"Fold {fold} バックテスト開始")
        print(f"{'='*50}")
        result = run_fold_backtest(fold, df_all, fold_cfg, feature_cols, cfg)
        all_results.append(result)

    # 全フォールドの合算サマリ
    print("\n" + "=" * 50)
    print("全フォールドサマリ")
    print("=" * 50)
    for r in all_results:
        fold = r.get("fold", "?")
        tm = r.get("test_metrics", {})
        dm = r.get("drawdown_metrics", {})
        ic = r.get("residual_ic", {})
        ll = r.get("log_loss", {})
        rk = r.get("ranking_metrics", {}).get("test", {})
        grades = r.get("grades", {})
        print(
            f"Fold {fold}: ROI={tm.get('roi', 0):.1%} [{grades.get('roi','?')}] | "
            f"hit={tm.get('hit_rate', 0):.1%} [{grades.get('hit_rate','?')}] | "
            f"MDD={dm.get('max_drawdown_rate', 0):.1%} [{grades.get('mdd','?')}] | "
            f"Sharpe={dm.get('sharpe_ratio', 0):.2f} [{grades.get('sharpe','?')}] | "
            f"n={tm.get('n_bets', 0)} [{grades.get('n_bets','?')}] | "
            f"top3={rk.get('top3_overlap_rate', float('nan')):.1%} | "
            f"top1={rk.get('top1_win_rate', float('nan')):.1%} | "
            f"wide_anchor={rk.get('wide_anchor_any', float('nan')):.1%} | "
            f"quinella_anchor={rk.get('quinella_anchor_any', float('nan')):.1%} | "
            f"logloss={ll.get('model_log_loss', float('nan')):.4f} | "
            f"IC={ic.get('residual_ic', float('nan')):.4f} | "
            f"T={r.get('temperature', 0.8):.3f} | "
            f"総合={grades.get('overall','?')}"
        )

    # 結果をJSONで保存
    save_path = MODELS_DIR / "backtest_results.json"
    serializable = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k != "test_predictions"}
        serializable.append(s)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    stamped_path = MODELS_DIR / f"backtest_results_{stamp}.json"
    shutil.copy2(save_path, stamped_path)
    print(f"\nバックテスト結果保存: {save_path}")
    print(f"  (日付スタンプコピー: {stamped_path.name})")

    return all_results


if __name__ == "__main__":
    run_full_backtest()
