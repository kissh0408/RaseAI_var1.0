"""_build_hc_norm_features の単体テスト（v50_hc_norm）。

検証項目:
- 基準時計差 = 同日×同調教場 median との差
- as-of が training_date < race_date を厳密に満たす（当日セッション除外）
- recent5 平均 / キャリア best(cummin) / accel best(cummax + ffill)
- 調教履歴なし馬は NaN、行数不変
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from create_features import _build_hc_norm_features


def _make_hc() -> pd.DataFrame:
    rows = []
    # 2024-01-01 center0: 馬1=52.0, 馬2=54.0, 馬3=56.0 → median=54.0
    for k, t, a in [(1, 52.0, 0.5), (2, 54.0, np.nan), (3, 56.0, np.nan)]:
        rows.append(dict(ketto_num=k, training_date=pd.Timestamp("2024-01-01"),
                         training_center=0, hc_4f_sec=t, hc_accel_sec=a))
    # 2024-01-08 center0: 馬1=53.0, 馬2=53.0 → median=53.0
    for k, t, a in [(1, 53.0, 0.2), (2, 53.0, np.nan)]:
        rows.append(dict(ketto_num=k, training_date=pd.Timestamp("2024-01-08"),
                         training_center=0, hc_4f_sec=t, hc_accel_sec=a))
    return pd.DataFrame(rows)


def _make_races() -> pd.DataFrame:
    # レースA: 2024-01-08（当日調教は使えない）、レースB: 2024-01-15
    rows = []
    for rid, date in [("A", "2024-01-08"), ("B", "2024-01-15")]:
        for k in [1, 4]:  # 馬4 は調教履歴なし
            rows.append(dict(race_id=rid, ketto_num=np.int64(k),
                             race_date=pd.Timestamp(date), dummy=1.0))
    return pd.DataFrame(rows)


def test_basediff_asof_strictly_before():
    out = _build_hc_norm_features(_make_races(), _make_hc())
    row = out[(out.race_id == "A") & (out.ketto_num == 1)].iloc[0]
    # 01-08 レースは 01-01 セッションのみ参照: basediff = 52-54 = -2.0
    assert row.trn_hc_basediff_recent5 == -2.0
    assert row.trn_hc_basediff_best == -2.0
    assert row.trn_hc_accel_best == 0.5


def test_recent5_and_career_best():
    out = _build_hc_norm_features(_make_races(), _make_hc())
    row = out[(out.race_id == "B") & (out.ketto_num == 1)].iloc[0]
    # 01-15 レースは両セッション参照: basediff = [-2.0, 0.0]
    assert row.trn_hc_basediff_recent5 == -1.0   # mean(-2, 0)
    assert row.trn_hc_basediff_best == -2.0      # cummin
    assert row.trn_hc_accel_best == 0.5          # max(0.5, 0.2)


def test_no_history_is_nan_and_rowcount_preserved():
    races = _make_races()
    out = _build_hc_norm_features(races, _make_hc())
    assert len(out) == len(races)
    row = out[(out.race_id == "B") & (out.ketto_num == 4)].iloc[0]
    assert np.isnan(row.trn_hc_basediff_recent5)
    assert np.isnan(row.trn_hc_basediff_best)
    assert np.isnan(row.trn_hc_accel_best)


def test_accel_nan_session_ffilled():
    # accel が NaN のセッションが最後でも、直前までの cummax が引き継がれる
    hc = _make_hc()
    races = pd.DataFrame([dict(race_id="C", ketto_num=np.int64(2),
                               race_date=pd.Timestamp("2024-01-15"), dummy=1.0)])
    out = _build_hc_norm_features(races, hc)
    # 馬2 は accel 全 NaN → NaN のまま
    assert np.isnan(out.iloc[0].trn_hc_accel_best)


def test_empty_hc_returns_nan_columns():
    races = _make_races()
    out = _build_hc_norm_features(races, pd.DataFrame(
        columns=["ketto_num", "training_date", "training_center", "hc_4f_sec", "hc_accel_sec"]))
    assert len(out) == len(races)
    assert out["trn_hc_basediff_recent5"].isna().all()
