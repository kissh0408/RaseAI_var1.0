"""本番ランタイムのイレギュラー処理を、テスト可能な純粋関数として集約するモジュール。

これまで出走取消除外は main.py / pipeline/strategy_pipeline.py にインライン実装され、
馬体重の異常値ガードはどこにも無かった（NaN 伝播のみ）。E2E テストが
`from main import validate_horse_weight, filter_scratched` を要求するため、
ロジックをここに切り出し、main/__init__.py から再エクスポートする。
odds==0（取消）と odds==NaN（未取得）の両方を検出し、win_prob_est がある場合は
レース内再正規化 + expected_return 再計算まで行う。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# 馬体重の物理的に妥当な範囲（kg）。サラブレッドの実測はおおむね 380〜560kg で、
# 300未満 / 650超は入力誤り・データ破損とみなして異常扱いにする。
HORSE_WEIGHT_MIN_KG = 300
HORSE_WEIGHT_MAX_KG = 650
# 前走比 ±20kg 超は大幅増減。除外はしないがフラグで下流に注意喚起する。
HORSE_WEIGHT_LARGE_DIFF_KG = 20

# 出走取消の標準マーカー（JRA ではオッズ 0 が取消を表す）。
SCRATCH_ODDS_MARKER = 0

# 推奨フレーム内で馬を一意に識別する候補列（先に見つかったものを使う）。
_HORSE_ID_CANDIDATES = ("horse_id", "ketto_num")


def validate_horse_weight(weight: Any, diff: Any) -> dict:
    """馬体重と前走差の妥当性を判定する純粋関数。

    deployment-evaluator 仕様に準拠:
      - weight が None/NaN（未発表）         -> status="excluded"（補完は呼び出し側方針）
      - weight が範囲外（<300 or >650）       -> status="abnormal"（値は破棄）
      - |diff| > 20                          -> status="large_change"（値は保持しフラグ）
      - それ以外                              -> status="normal"

    なぜ excluded を既定にするか: 未発表体重を 0 や前走値で無条件補完すると
    本番で誤った特徴量を生むため、まず除外候補としてマークし補完方針を呼び出し側に委ねる。
    """
    if weight is None or (isinstance(weight, float) and pd.isna(weight)):
        return {"status": "excluded", "value": None}

    weight_num = pd.to_numeric(weight, errors="coerce")
    if pd.isna(weight_num):
        # 数値化できない値（文字列ゴミ等）も未発表と同じく除外候補にする。
        return {"status": "excluded", "value": None}

    if weight_num < HORSE_WEIGHT_MIN_KG or weight_num > HORSE_WEIGHT_MAX_KG:
        return {"status": "abnormal", "value": None}

    if diff is not None:
        diff_num = pd.to_numeric(diff, errors="coerce")
        if not pd.isna(diff_num) and abs(diff_num) > HORSE_WEIGHT_LARGE_DIFF_KG:
            return {"status": "large_change", "value": float(weight_num), "flag": True}

    return {"status": "normal", "value": float(weight_num)}


def clamp_horse_weight(weight: Any) -> float | None:
    """前処理層用の値域ガード。異常値は NaN(None) 化し、正常値はそのまま返す。

    validate_horse_weight と同じ閾値で「異常値のみ」処理する。正常な体重は
    数値変換以外の変更をしない（特徴量の意味・既存正常値の挙動を変えないため）。
    """
    result = validate_horse_weight(weight, None)
    if result["status"] in ("excluded", "abnormal"):
        return None
    return result["value"]


def filter_scratched(
    recommendations: pd.DataFrame,
    scratched_horses: list | None = None,
) -> pd.DataFrame:
    """出走取消馬を推奨フレームから除外する純粋関数。

    odds==0（JRA 取消マーカー）と odds==NaN（未取得・レース前取消）の両方を検出する。
    明示的な取消馬 ID リスト（horse_id / ketto_num）でも除外できる。
    win_prob_est がある場合は残存馬へレース内再正規化し expected_return も再計算する。
    """
    if recommendations is None or len(recommendations) == 0:
        return recommendations

    out = recommendations.copy()
    drop_mask = pd.Series(False, index=out.index)

    # 1) 明示リストによる除外（E2E 経路）: 最初に見つかった ID 列で照合する。
    if scratched_horses:
        scratched_set = set(scratched_horses)
        for id_col in _HORSE_ID_CANDIDATES:
            if id_col in out.columns:
                drop_mask = drop_mask | out[id_col].isin(scratched_set)
                break

    # 2) odds==0 または NaN による除外（main 本番修正 68994941/22e8cfc7 を統合）。
    if "odds" in out.columns:
        odds_num = pd.to_numeric(out["odds"], errors="coerce")
        drop_mask = drop_mask | (odds_num == SCRATCH_ODDS_MARKER) | odds_num.isna()

    if not drop_mask.any():
        return out.reset_index(drop=True)

    out = out.loc[~drop_mask].reset_index(drop=True)

    if "win_prob_est" in out.columns and "race_id" in out.columns and len(out) > 0:
        race_sum = out.groupby("race_id")["win_prob_est"].transform("sum")
        out["win_prob_est"] = out["win_prob_est"] / race_sum.clip(lower=1e-9)
        if "odds" in out.columns and "expected_return" in out.columns:
            o = pd.to_numeric(out["odds"], errors="coerce")
            out["expected_return"] = out["win_prob_est"] * o

    return out
