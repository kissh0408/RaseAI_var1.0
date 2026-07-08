"""確率キャリブレーション: Isotonic Regression / Platt Scaling。

model-strategy-generatorフェーズ。
lambdarankの生スコアを真の勝率に変換する。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from pipeline_common import FEATURES_DIR, MODELS_DIR, load_config


def _load_booster_crlf_safe(path: Path) -> lgb.Booster:
    """Windows CRLF 改行でも LightGBM テキストモデルを読めるようにする。"""
    raw = path.read_bytes()
    if b"\r\n" in raw:
        model_str = raw.replace(b"\r\n", b"\n").decode("utf-8")
        return lgb.Booster(model_str=model_str)
    return lgb.Booster(model_file=str(path))


def load_model(fold: int) -> lgb.Booster:
    binary_path = MODELS_DIR / f"lgbm_binary_fold{fold}.txt"
    if binary_path.exists():
        return _load_booster_crlf_safe(binary_path)
    return _load_booster_crlf_safe(MODELS_DIR / f"lgbm_lambdarank_fold{fold}.txt")


def get_raw_scores(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """lambdarankの生スコア（高いほど順位が高いことを示す）を返す。"""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    return model.predict(X, num_iteration=model.best_iteration)


def scores_to_win_prob(
    scores: np.ndarray,
    race_ids: pd.Series,
    method: str = "softmax",
) -> pd.Series:
    """レース内でのSoftmax変換により単勝確率を計算する。

    Plackett-Luceモデルに基づく: P(i wins) = exp(s_i) / Σ exp(s_k)
    数値安定化のためレース内最大値を引いてからexp。
    """
    result = pd.Series(np.nan, index=range(len(scores)))
    df_tmp = pd.DataFrame({"score": scores, "race_id": race_ids.values})

    for race_id, group in df_tmp.groupby("race_id"):
        s = group["score"].values
        # 数値安定化
        s_stable = s - s.max()
        exp_s = np.exp(s_stable)
        prob = exp_s / exp_s.sum()
        result.iloc[group.index] = prob

    return result


class ProbabilityCalibrator:
    """Isotonic Regressionによる確率キャリブレーター。

    lambdarankの生スコア（softmax後）と実際の勝率を一致させる。
    90日のホールドアウト期間で学習する。
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.scaler = MinMaxScaler()
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        elif method == "platt":
            self.calibrator = LogisticRegression(C=1e5)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, is_win: np.ndarray) -> "ProbabilityCalibrator":
        """ホールドアウト期間で校正器を学習する。"""
        X = raw_probs.reshape(-1, 1)
        if self.method == "isotonic":
            self.calibrator.fit(raw_probs, is_win)
        elif self.method == "platt":
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.calibrator.fit(X_scaled, is_win)
        self.is_fitted = True
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """校正済み確率を返す。"""
        assert self.is_fitted, "calibrator is not fitted yet"
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(raw_probs), 1e-6, 1.0)
        elif self.method == "platt":
            X = raw_probs.reshape(-1, 1)
            X_scaled = self.scaler.transform(X)
            return self.calibrator.predict_proba(X_scaled)[:, 1]

    def save(self, path: Path) -> None:
        # 自パイプラインが生成したオブジェクトのみ保存。外部起源のpickleは絶対に読み込まない。
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        # 信頼できるパス（model_training/models/）からのみロード。
        # 外部ファイルをここに渡さないこと。
        assert str(path).replace("\\", "/").find("model_training/models") != -1, (
            f"安全でないパスからのロードを阻止: {path}"
        )
        return joblib.load(path)


def build_calibrator(fold: int, df_holdout: pd.DataFrame, feature_cols: list[str]) -> ProbabilityCalibrator:
    """ホールドアウト期間でキャリブレーターを構築して保存する。"""
    cfg = load_config()
    cal_cfg = cfg["training"]["calibration"]
    # train_config は isotonic フラグのみの場合もある（MS-1）
    method = cal_cfg.get("method")
    if method is None:
        method = "isotonic" if cal_cfg.get("isotonic", False) else "platt"
    _holdout_days = cal_cfg.get("holdout_days", 90)  # 将来の holdout 切り出し用

    model = load_model(fold)
    scores = get_raw_scores(model, df_holdout, feature_cols)
    raw_probs = scores_to_win_prob(scores, df_holdout["race_id"])

    is_win = (df_holdout["finish_rank"] == 1).astype(int).values
    raw_probs_arr = raw_probs.values

    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(raw_probs_arr, is_win)

    save_path = MODELS_DIR / f"calibrator_fold{fold}.joblib"
    calibrator.save(save_path)
    print(f"Calibrator saved: {save_path}")
    return calibrator


def apply_calibration(
    model: lgb.Booster,
    calibrator: ProbabilityCalibrator,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """推論時にモデルスコア → 校正済み勝率を返す。"""
    scores = get_raw_scores(model, df, feature_cols)
    raw_probs = scores_to_win_prob(scores, df["race_id"])
    calibrated = calibrator.predict(raw_probs.values)
    return pd.Series(calibrated, index=df.index, name="model_prob")
