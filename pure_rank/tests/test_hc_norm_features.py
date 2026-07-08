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


def test_accel_all_nan_stays_nan():
    # accel が全セッション NaN の馬は NaN のまま（ffill で捏造されない）
    hc = _make_hc()
    races = pd.DataFrame([dict(race_id="C", ketto_num=np.int64(2),
                               race_date=pd.Timestamp("2024-01-15"), dummy=1.0)])
    out = _build_hc_norm_features(races, hc)
    # 馬2 は accel 全 NaN → NaN のまま
    assert np.isnan(out.iloc[0].trn_hc_accel_best)


def test_accel_valid_then_nan_ffilled_to_latest_session():
    # ffill 退行検知: accel が「有効 → NaN」の順で、as-of が拾う最新セッションが
    # NaN 行になるケース。cummax は NaN 行で NaN を返すため、ffill が無いと
    # trn_hc_accel_best が NaN になりこのテストが落ちる。
    hc = pd.DataFrame([
        dict(ketto_num=5, training_date=pd.Timestamp("2024-01-01"),
             training_center=0, hc_4f_sec=53.0, hc_accel_sec=0.8),
        dict(ketto_num=5, training_date=pd.Timestamp("2024-01-08"),
             training_center=0, hc_4f_sec=54.0, hc_accel_sec=np.nan),
    ])
    races = pd.DataFrame([dict(race_id="D", ketto_num=np.int64(5),
                               race_date=pd.Timestamp("2024-01-15"), dummy=1.0)])
    out = _build_hc_norm_features(races, hc)
    assert out.iloc[0].trn_hc_accel_best == 0.8


def test_baseline_median_is_per_training_center():
    # 同日でも調教場ごとに別 median: groupby キーから training_center が落ちると
    # プール median（[52,54,56,60,62,64] → 58）になり、両馬のアサートが崩れる。
    rows = []
    # 2024-01-10 center0: 馬1=52, 馬2=54, 馬3=56 → median=54
    for k, t in [(1, 52.0), (2, 54.0), (3, 56.0)]:
        rows.append(dict(ketto_num=k, training_date=pd.Timestamp("2024-01-10"),
                         training_center=0, hc_4f_sec=t, hc_accel_sec=np.nan))
    # 2024-01-10 center1: 馬5=60, 馬6=62, 馬7=64 → median=62
    for k, t in [(5, 60.0), (6, 62.0), (7, 64.0)]:
        rows.append(dict(ketto_num=k, training_date=pd.Timestamp("2024-01-10"),
                         training_center=1, hc_4f_sec=t, hc_accel_sec=np.nan))
    hc = pd.DataFrame(rows)
    races = pd.DataFrame([
        dict(race_id="E", ketto_num=np.int64(1),
             race_date=pd.Timestamp("2024-01-15"), dummy=1.0),
        dict(race_id="E", ketto_num=np.int64(5),
             race_date=pd.Timestamp("2024-01-15"), dummy=1.0),
    ])
    out = _build_hc_norm_features(races, hc)
    row1 = out[out.ketto_num == 1].iloc[0]
    row5 = out[out.ketto_num == 5].iloc[0]
    # 馬1: 52 - 54 = -2.0（center0 median 基準。プール median なら 52-58=-6.0）
    assert row1.trn_hc_basediff_recent5 == -2.0
    assert row1.trn_hc_basediff_best == -2.0
    # 馬5: 60 - 62 = -2.0（center1 median 基準。プール median なら 60-58=+2.0）
    assert row5.trn_hc_basediff_recent5 == -2.0
    assert row5.trn_hc_basediff_best == -2.0


def test_empty_hc_returns_nan_columns():
    races = _make_races()
    out = _build_hc_norm_features(races, pd.DataFrame(
        columns=["ketto_num", "training_date", "training_center", "hc_4f_sec", "hc_accel_sec"]))
    assert len(out) == len(races)
    assert out["trn_hc_basediff_recent5"].isna().all()
