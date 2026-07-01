"""
train.py — RaceAI_var1.0 LambdaRank 学習スクリプト

使用方法:
    python pure_rank/src/train.py              # seed=42 のみ、fold 3 のみ
    python pure_rank/src/train.py --ensemble   # 5 seeds × 3 folds = 15 モデル

モデル保存先:
    pure_rank/models/lambdarank_fold{1,2,3}_seed{42-46}.txt

禁止事項:
- init_score に市場オッズ由来の値を使わない
- categorical_feature を lgb.Dataset に未指定にしない
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

# EV 重み付き学習用: evaluate.py・predict.py・simulate_ev.py の既存関数を再利用
# （コード重複禁止）
from evaluate import ensemble_predict, load_models
from predict import _best_wide_pair, compute_race_probabilities, softmax_with_temperature
from simulate_ev import _build_wide_odds_lookup, _normalize_pair

# ─── パス解決 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "pure_rank" / "config" / "train_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── 特徴量列の選択 ────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    """学習に使う特徴量列を返す。

    ID 列・ラベル列・禁止列を除外する。
    """
    id_cols = set(cfg["features"]["id_cols"])
    forbidden = {
        # 市場情報（絶対禁止）
        "odds", "popularity", "win_odds", "place_odds",
        "quinella_odds", "market_prob", "market_log_odds",
        "init_score", "ninki",
        # 一時作業列
        "_time_dev",
        # RA / SE のメタ列（特徴量として不要）
        "year", "month_day", "kai", "nichi", "race_num",
        "horse_num", "registered_count", "finish_count",
        "race_type_code", "weight_type", "race_condition_code",
        "race_level", "race_age_type", "course_kubun",
        "track_code",
        "obstacle_mile_time_sec",
        "dead_heat_flag", "dead_heat_count",
        "breed_code", "region_code",
        # 血統 ID（文字列。特徴量としては派生した win_rate 系を使う）
        "sire_id", "bms_id",
        # ─── レース後にしか判明しない後出し情報（特徴量にしてはならない） ───
        # 走破タイム・上がり3F（結果。hist_ 系経由で過去走データは使用可）
        "racetime", "time_3f_after",
        # コーナー通過順（レース中の位置情報。結果）
        "corner_1", "corner_2", "corner_3", "corner_4",
        # 脚質判定（レース後判定）
        "running_style_code",
        # 異常区分（レース後確定）
        "abnormal_code",
        # 賞金（レース後確定。hist_ 系経由で過去走データは使用可）
        "hon_shokin", "fuka_shokin",
        # 生ラベル（全てレース後確定）
        "finish_rank", "is_win", "is_place", "lr_label",
    }
    exclude = id_cols | forbidden

    # 残った数値・カテゴリ列を特徴量とする
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype not in ["object", "string"]
    ]
    return feature_cols


def get_group_sizes(df: pd.DataFrame, race_id_col: str = "race_id") -> list[int]:
    """LightGBM LambdaRank 用 group 配列（レースごとの頭数リスト）を返す。

    前提: df は (race_date, race_id, horse_num) 順に並んでいなければならない。
    sort=False は行順を尊重するため、parquet の行順序が正しい場合のみ正確な
    グループ割り当てになる。create_features.py でこのソートを保証している。
    """
    return df.groupby(race_id_col, sort=False).size().tolist()


# ─── 時系列 Fold 定義 ─────────────────────────────────────────────────────────

def get_fold_split(
    df: pd.DataFrame, fold: int, fold_valid_years: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fold 番号（1-indexed）に対応する train / valid を返す。

    fold_valid_years は train_config.json の training.fold_valid_years から渡す。
    各 year は "YYYY" 形式で、valid 期間は "{year}-01-01" 〜 "{year}-12-31" とする。

    Parameters
    ----------
    df : 全学習対象データ（テスト期間を含まない）
    fold : 1, 2, 3
    fold_valid_years : config の training.fold_valid_years（例: ["2022", "2023", "2024"]）

    Returns
    -------
    (train_df, valid_df)
    """
    year = fold_valid_years[fold - 1]
    valid_start_ts = pd.Timestamp(f"{year}-01-01")
    valid_end_ts = pd.Timestamp(f"{year}-12-31")

    train_df = df[df["race_date"] < valid_start_ts].copy()
    valid_df = df[
        (df["race_date"] >= valid_start_ts) & (df["race_date"] <= valid_end_ts)
    ].copy()
    return train_df, valid_df


# ─── EV サンプル重み計算 ────────────────────────────────────────────────────────

def compute_ev_weight(ev: float, k: float, max_weight: float = 2.0) -> float:
    """EV をシグモイド関数で重みに変換する。

    Parameters
    ----------
    ev         : 当該レースの最大 EV ペアの EV 値（NaN の場合は 1.0 を返す）
    k          : シグモイドの急峻さパラメータ（感度分析: 5, 10, 20）
    max_weight : 重みの上限（感度分析: 1.5, 2.0, 3.0）

    Returns
    -------
    weight in [1.0, max_weight]
        - EV << 1.0 → weight ≈ 1.0
        - EV == 1.0 → weight = (1.0 + max_weight) / 2
        - EV >> 1.0 → weight ≈ max_weight
    """
    if np.isnan(ev):
        return 1.0  # EV 不明（オッズ未取得）→ デフォルト重み
    return 1.0 + (max_weight - 1.0) / (1.0 + np.exp(-k * (ev - 1.0)))


def compute_train_weights(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    odds_dir: Path,
    T_opt: float,
    k: float,
    max_weight: float,
    cfg: dict,
) -> np.ndarray:
    """学習データ全行に対する EV サンプル重みを計算して返す。

    2段学習プロセス:
    1. 現行モデルで df の Wide EV を予測（Harville 確率 × WideOdds 事前オッズ）
    2. EV に基づくシグモイド重みを計算
    3. レース内で重みの合計 = 1 に正規化して返す

    Parameters
    ----------
    df          : 学習プール全行（race_date <= valid_end）
    feature_cols: 学習に使う特徴量列
    models_dir  : 現行モデルの保存ディレクトリ
    odds_dir    : WideOdds CSV のディレクトリ
    T_opt       : Softmax 温度パラメータ（train_config.json から）
    k           : シグモイド急峻さ（感度分析パラメータ）
    max_weight  : 重みの上限（感度分析パラメータ）
    cfg         : train_config.json の内容

    Returns
    -------
    np.ndarray: shape = (len(df),)。df の行順と一致する。
    """
    print(f"  [compute_train_weights] Loading current models from {models_dir}")
    models = load_models(models_dir)
    preds = ensemble_predict(models, df[feature_cols])
    df = df.copy()
    df["pred_score"] = preds

    # WideOdds CSV の読み込み（学習プールの全年）
    train_years = sorted(df["race_date"].dt.year.unique().tolist())
    print(f"  [compute_train_weights] Loading WideOdds for years: {train_years}")
    wide_odds_lookup = _build_wide_odds_lookup(train_years, odds_dir)

    # レースごとに best wide pair の EV を計算
    race_ev_map: dict = {}
    for race_id, grp in df.groupby("race_id"):
        if len(grp) < 2:
            race_ev_map[race_id] = float("nan")
            continue
        rid = str(race_id)
        grp_s = grp.sort_values("pred_score", ascending=False).reset_index(drop=True)
        horse_nums = grp_s["horse_num"].astype(int).values
        scores = grp_s["pred_score"].values
        probs = compute_race_probabilities(scores, T_opt)
        wi, wj = _best_wide_pair(probs["wide_matrix"])
        wide_key = _normalize_pair(int(horse_nums[wi]), int(horse_nums[wj]))
        p_wide = float(probs["wide_matrix"][wi, wj])
        prior = wide_odds_lookup.get(rid, {}).get(wide_key, None)
        ev = (p_wide * prior) if prior is not None else float("nan")
        race_ev_map[race_id] = ev

    df["race_ev"] = df["race_id"].map(race_ev_map)

    # シグモイド重み（行単位）
    df["weight_raw"] = df["race_ev"].apply(lambda ev: compute_ev_weight(ev, k, max_weight))

    # レース内正規化: 各レース内で重みの合計 = 1
    race_weight_sum = df.groupby("race_id")["weight_raw"].transform("sum")
    df["weight_norm"] = df["weight_raw"] / race_weight_sum

    w_arr = df["weight_norm"].values
    print(f"  [compute_train_weights] weight range: {w_arr.min():.4f} - {w_arr.max():.4f}")
    nan_ev_races = int(df["race_ev"].isna().sum())
    print(f"  [compute_train_weights] races with EV=NaN: {nan_ev_races}/{len(df)}")
    return w_arr


# ─── LambdaRank 学習 ───────────────────────────────────────────────────────────

def train_lambdarank(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: list[int],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    group_valid: list[int],
    feature_cols: list[str],
    cat_features: list[str],
    params_cfg: dict,
    training_cfg: dict,
    seed: int,
    weight_train: Optional[np.ndarray] = None,
) -> lgb.Booster:
    """LambdaRank モデルを学習して返す。

    Parameters
    ----------
    init_score は使わない（RaceAI_var2.0.0 との根本的な違い）
    categorical_feature を lgb.Dataset に必ず指定する
    weight_train : EV サンプル重み（None = 均等重み）。group_train と同じ行順
    """
    # cat_features のうち実際に feature_cols に含まれるものだけ指定
    valid_cat = [c for c in cat_features if c in feature_cols]

    params = {
        "objective": params_cfg["objective"],
        "metric": params_cfg["metric"],
        "ndcg_eval_at": params_cfg["ndcg_eval_at"],
        "label_gain": params_cfg["label_gain"],
        "num_leaves": params_cfg["num_leaves"],
        "min_child_samples": params_cfg["min_child_samples"],
        "reg_alpha": params_cfg["reg_alpha"],
        "reg_lambda": params_cfg["reg_lambda"],
        "learning_rate": params_cfg["learning_rate"],
        "seed": seed,
        "verbose": -1,
    }

    # init_score は使わない（市場オッズ由来の残差学習は禁止）
    lgb_train = lgb.Dataset(
        X_train[feature_cols],
        label=y_train,
        group=group_train,
        categorical_feature=valid_cat,
        weight=weight_train,  # None なら均等重み（後方互換）
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        X_valid[feature_cols],
        label=y_valid,
        group=group_valid,
        categorical_feature=valid_cat,
        reference=lgb_train,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=params_cfg["n_estimators"],
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(training_cfg["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(training_cfg["log_eval_period"]),
        ],
    )
    return model


# ─── メイン ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RaceAI_var1.0 LambdaRank Training")
    parser.add_argument(
        "--ensemble", action="store_true",
        help="5 seeds × 3 folds の全モデルを学習する（省略時は seed=42 + fold 3 のみ）"
    )
    parser.add_argument(
        "--use-ev-weight",
        action="store_true",
        help="EV ベースのサンプル重み付きで学習する",
    )
    parser.add_argument(
        "--ev-weight-k",
        type=float,
        default=10.0,
        help="シグモイド重みの急峻さパラメータ（感度分析: 5, 10, 20）",
    )
    parser.add_argument(
        "--ev-weight-max",
        type=float,
        default=2.0,
        help="サンプル重みの上限（感度分析: 1.5, 2.0, 3.0）",
    )
    args = parser.parse_args()

    cfg = load_config()
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    feat_cfg = cfg["features"]

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"

    # モデル保存先: --use-ev-weight 時は専用ディレクトリに保存（既存モデルを上書きしない）
    if args.use_ev_weight:
        k_label = str(int(args.ev_weight_k)) if args.ev_weight_k == int(args.ev_weight_k) else str(args.ev_weight_k)
        max_label = str(args.ev_weight_max).replace(".", "")
        models_dir = PROJECT_ROOT / f"pure_rank/models_weighted_k{k_label}_max{max_label}"
        print(f"  EV weighted mode: k={args.ev_weight_k}, max_weight={args.ev_weight_max}")
        print(f"  Models will be saved to: {models_dir}")
    else:
        models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print(f"Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)
    print(f"  rows={len(df):,}, cols={len(df.columns)}")

    # 特徴量列を決定
    feature_cols = get_feature_cols(df, cfg)
    cat_features = feat_cfg["categorical"]
    print(f"  Feature cols: {len(feature_cols)}")
    print(f"  Cat features: {cat_features}")

    # テスト期間を除外（学習・バリデーション用のみ）
    valid_end_ts = pd.Timestamp(training_cfg["valid_end"])
    df_train_pool = df[df["race_date"] <= valid_end_ts].copy()
    df_test = df[df["race_date"] > valid_end_ts].copy()
    print(f"  Train pool: {len(df_train_pool):,} rows | Test: {len(df_test):,} rows")

    # 学習対象のシード・フォールドを決定
    if args.ensemble:
        seeds = training_cfg["seeds"]
        folds = list(range(1, training_cfg["folds"] + 1))
    else:
        seeds = [training_cfg["seeds"][0]]  # 42 のみ
        folds = [training_cfg["folds"]]     # fold 3 のみ

    print(f"\nTraining: seeds={seeds}, folds={folds}")
    print(f"Total models: {len(seeds) * len(folds)}")

    # EV 重み付き学習: 学習プール全行に対して重みを事前計算
    weight_series: Optional[pd.Series] = None
    if args.use_ev_weight:
        T_opt = float(cfg.get("plackett_luce", {}).get("T_opt", 1.0))
        odds_dir = PROJECT_ROOT / "common" / "data" / "output" / "odds"
        # 均等重みモデル（既存の pure_rank/models/）で EV を予測するため、
        # models_dir を一時的に既存モデルディレクトリに向ける
        existing_models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
        print(f"\nComputing EV-based sample weights (k={args.ev_weight_k}, max_weight={args.ev_weight_max})...")
        weight_array = compute_train_weights(
            df=df_train_pool,
            feature_cols=feature_cols,
            models_dir=existing_models_dir,
            odds_dir=odds_dir,
            T_opt=T_opt,
            k=args.ev_weight_k,
            max_weight=args.ev_weight_max,
            cfg=cfg,
        )
        print(f"  EV weight range: {weight_array.min():.4f} - {weight_array.max():.4f}")
        # pandas Series でインデックス対応付け（fold 分割後の行選択に使う）
        weight_series = pd.Series(weight_array, index=df_train_pool.index)

    trained_models = []
    for seed in seeds:
        for fold in folds:
            model_path = models_dir / f"lambdarank_fold{fold}_seed{seed}.txt"

            print(f"\n--- Fold {fold} / Seed {seed} ---")
            train_df, valid_df = get_fold_split(df_train_pool, fold, training_cfg["fold_valid_years"])
            print(f"  Train: {len(train_df):,} rows, {train_df['race_id'].nunique():,} races "
                  f"({train_df['race_date'].min().date()} - {train_df['race_date'].max().date()})")
            print(f"  Valid: {len(valid_df):,} rows, {valid_df['race_id'].nunique():,} races "
                  f"({valid_df['race_date'].min().date()} - {valid_df['race_date'].max().date()})")

            if len(valid_df) == 0:
                print("  [SKIP] valid_df が空です")
                continue

            # LambdaRank ラベル（lr_label）
            y_train = train_df[feat_cfg["lr_label"]]
            y_valid = valid_df[feat_cfg["lr_label"]]

            # group 配列（レースごとの頭数）
            # 注意: race_id の順序を保持するため sort=False
            group_train = get_group_sizes(train_df)
            group_valid = get_group_sizes(valid_df)

            # fold ごとに train_df のインデックスに対応する重みを選択
            # weight_series は df_train_pool と同じインデックスを持つ
            fold_weight_train: Optional[np.ndarray] = None
            if weight_series is not None:
                fold_weight_train = weight_series.loc[train_df.index].values

            model = train_lambdarank(
                X_train=train_df,
                y_train=y_train,
                group_train=group_train,
                X_valid=valid_df,
                y_valid=y_valid,
                group_valid=group_valid,
                feature_cols=feature_cols,
                cat_features=cat_features,
                params_cfg=params_cfg,
                training_cfg=training_cfg,
                seed=seed,
                weight_train=fold_weight_train,
            )

            model.save_model(str(model_path))
            print(f"  Saved: {model_path}")
            trained_models.append(model)

    print(f"\n[train] Done. {len(trained_models)} models trained.")

    # 特徴量重要度サマリー（最後のモデル）
    if trained_models:
        last_model = trained_models[-1]
        importance = pd.Series(
            last_model.feature_importance(importance_type="gain"),
            index=feature_cols,
        ).sort_values(ascending=False)
        print("\nTop 20 features by gain:")
        print(importance.head(20).to_string())


if __name__ == "__main__":
    main()
