"""features_v2.parquet に 3 つの新特徴量を追加し features_v3.parquet を生成する。

追加特徴量:
  jra_tm_orthogonalized  : JRA TM示唆確率から市場log-oddsの影響を除去した直交化残差
  weight_relative_z      : レース内馬体重の相対 Z スコア（当日レース内比較）
  jockey_trainer_synergy : 騎手×調教師コンビの通算勝率（時系列リーク防止 shift(1)）

使用方法:
  python model_training/src/create_features_v3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pipeline_common import FEATURES_DIR, save_features, validate_no_leakage


# ---------------------------------------------------------------------------
# A-1. JRA TM 直交化特徴量
# ---------------------------------------------------------------------------

def add_jra_tm_orthogonalized(df: pd.DataFrame) -> pd.DataFrame:
    """jra_tm_implied_prob から market_log_odds の線形成分を除去した残差を追加する。

    OLS: jra_tm_implied_prob = alpha + beta * market_log_odds + residual
    feature = residual（市場に織り込まれていない JRA TM の純粋な情報）
    """
    src_col = "jra_tm_implied_prob"
    margin_col = "market_log_odds"

    if src_col not in df.columns or margin_col not in df.columns:
        print(f"[WARN] {src_col} または {margin_col} が見つかりません。jra_tm_orthogonalized をスキップします。")
        df["jra_tm_orthogonalized"] = np.nan
        return df

    mask = df[src_col].notna() & df[margin_col].notna()
    X = df.loc[mask, margin_col].values.reshape(-1, 1)
    y = df.loc[mask, src_col].values

    # OLS係数を全期間データで推定（feature transformation のみ、is_win 不使用）
    reg = LinearRegression().fit(X, y)
    alpha, beta = float(reg.intercept_), float(reg.coef_[0])

    fitted = alpha + beta * df[margin_col]
    df["jra_tm_orthogonalized"] = df[src_col] - fitted

    residual_mean = df.loc[mask, "jra_tm_orthogonalized"].mean()
    residual_std = df.loc[mask, "jra_tm_orthogonalized"].std()
    print(
        f"  jra_tm_orthogonalized: alpha={alpha:.4f}, beta={beta:.4f}, "
        f"R2={reg.score(X, y):.3f}, residual mean={residual_mean:.4f} std={residual_std:.4f}"
    )
    return df


# ---------------------------------------------------------------------------
# A-2. レース内馬体重相対 Z スコア
# ---------------------------------------------------------------------------

def add_weight_relative_z(df: pd.DataFrame) -> pd.DataFrame:
    """当日レース内での馬体重の相対 Z スコアを追加する。

    z = (horse_weight - race_mean_weight) / race_std_weight
    レース内 std が 0 の場合は 0 を設定する。
    """
    if "horse_weight" not in df.columns:
        print("[WARN] horse_weight が見つかりません。weight_relative_z をスキップします。")
        df["weight_relative_z"] = np.nan
        return df

    race_mean = df.groupby("race_id")["horse_weight"].transform("mean")
    race_std = df.groupby("race_id")["horse_weight"].transform("std").fillna(0)
    # std が 0（全頭同じ体重）の場合は z=0
    df["weight_relative_z"] = np.where(
        race_std > 0,
        (df["horse_weight"] - race_mean) / race_std,
        0.0,
    )

    print(
        f"  weight_relative_z: mean={df['weight_relative_z'].mean():.4f}, "
        f"std={df['weight_relative_z'].std():.4f}, "
        f"NaN rate={df['weight_relative_z'].isna().mean():.1%}"
    )
    return df


# ---------------------------------------------------------------------------
# A-3. 騎手×調教師シナジー率
# ---------------------------------------------------------------------------

def add_jockey_trainer_synergy(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """騎手×調教師コンビの通算勝率（時系列リーク防止 shift(1)）を追加する。

    min_count 件未満のコンビは NaN（信号不足）とする。
    """
    if "jockey_id" not in df.columns or "trainer_id" not in df.columns:
        print("[WARN] jockey_id または trainer_id が見つかりません。jockey_trainer_synergy をスキップします。")
        df["jockey_trainer_synergy"] = np.nan
        return df

    if "is_win" not in df.columns:
        df["is_win"] = (df["finish_rank"] == 1).astype(int)

    df = df.sort_values("race_date").reset_index(drop=True)

    # shift(1).expanding().sum/count と等価: 累積和から当該行を引く / 過去出走数はcumcount
    # （is_winはNaNなしの0/1なので厳密に一致する。コンビ数が多いためlambdaより大幅に高速）
    grp = df.groupby(["jockey_id", "trainer_id"])["is_win"]
    cum_wins = grp.cumsum() - df["is_win"]
    cum_count = grp.cumcount()

    synergy = cum_wins / cum_count
    # min_count 件未満は NaN
    synergy = synergy.where(cum_count >= min_count, other=np.nan)
    df["jockey_trainer_synergy"] = synergy

    nan_rate = df["jockey_trainer_synergy"].isna().mean()
    mean_val = df["jockey_trainer_synergy"].mean()
    print(
        f"  jockey_trainer_synergy: mean={mean_val:.4f}, NaN rate={nan_rate:.1%} "
        f"(新規コンビ・件数不足を除外)"
    )
    return df


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

NEW_FEATURES = ["jra_tm_orthogonalized", "weight_relative_z", "jockey_trainer_synergy"]


def main() -> None:
    v2_path = FEATURES_DIR / "features_v2.parquet"
    if not v2_path.exists():
        raise FileNotFoundError(f"features_v2.parquet が見つかりません: {v2_path}")

    print("=== features_v3 生成開始 ===")
    print(f"  入力: {v2_path}")

    df = pd.read_parquet(v2_path)
    df["race_date"] = pd.to_datetime(df["race_date"])
    print(f"  読み込み完了: {len(df)} rows, {len(df.columns)} cols")

    print("\n[A-1] JRA TM 直交化特徴量...")
    df = add_jra_tm_orthogonalized(df)

    print("\n[A-2] レース内馬体重相対 Z スコア...")
    df = add_weight_relative_z(df)

    print("\n[A-3] 騎手×調教師シナジー率...")
    df = add_jockey_trainer_synergy(df)

    # バリデーション
    print("\n=== NaN 率チェック ===")
    validate_no_leakage(df, NEW_FEATURES)
    for col in NEW_FEATURES:
        nan_rate = df[col].isna().mean()
        print(f"  {col}: NaN={nan_rate:.1%}")

    # 保存
    print("\n=== features_v3 保存 ===")
    save_features(df, "features_v3")
    print(f"=== 完了: {len(df)} rows, {len(df.columns)} cols ===")


if __name__ == "__main__":
    main()
