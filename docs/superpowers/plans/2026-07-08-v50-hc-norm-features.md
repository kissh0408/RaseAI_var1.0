# v50_hc_norm（坂路 基準時計差 特徴量）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 坂路調教タイムを「同日・同調教場の全馬 median との差（基準時計差）」で正規化したキャリア集約3特徴量を pure_rank に追加し、v39_course_slim（Top-1=30.24%）と A/B 比較する。

**Architecture:** `src/training.py`（ルート）の 基準時計差・坂路スピード・坂路瞬発力 の概念を、pure_rank の既存パイプライン（HC_preprocessed.parquet → create_features.py → LambdaRank）にベクトル化して移植する。既存の `trn_*` 特徴量（14日窓・生タイム）はそのまま維持し、新規3列のみ features_version = `v50_hc_norm` でゲート付き追加する。WC（ウッド）はデータが2021年以降しかなく学習期間（〜2023）の大半が欠損するため対象外。

**Tech Stack:** Python 3.12 / pandas / LightGBM LambdaRank / pytest

## Global Constraints

- 市場情報（odds / popularity / ninki / market_log_odds / init_score）を pure_rank の特徴量に一切使わない（CLAUDE.md 憲法1）
- 時系列リーク防止: 新特徴量は `training_date < race_date` を**厳密に**（`allow_exact_matches=False`）満たすこと（憲法2）
- 後出し禁止: 合否ゲート・フォールバックは本計画に事前登録済み。テスト結果を見た後のパラメータ・特徴量調整は禁止（憲法3）
- リーク停止閾値: Top-1 > 40% または Spearman > 0.6 → 即停止・報告（憲法4）
- **合格ゲート（事前登録）**: TEST Top-1 > 30.24%（v39_course_slim）。Top-3/NDCG@3/Spearman が明確に劣化（-0.5pp / -0.003 超）していないこと
- **相関ゲートフォールバック（事前登録）**: `_run_correlation_gate` で NG（|r|>=0.7）が**新規列同士のみ**（trn_hc_basediff_recent5 vs trn_hc_basediff_best）の場合 → `trn_hc_basediff_best` を除外した2列構成で再実行。**既存列との NG** の場合 → 実験中止し planner（指揮官）へ報告
- `features_*.parquet` / `HC_preprocessed.parquet` / `pure_rank/models/*.txt` は上書き前に必ずバックアップ（憲法・規約4）
- 設定値は `pure_rank/config/train_config.json` に集約。ハードコード禁止
- git commit は**タスクで触ったファイルのみ** `git add <path>` で個別指定（リポジトリに無関係な未コミット変更が大量にあるため `git add -A` 禁止）
- LambdaRank パラメータ・分割・シードは一切変更しない（実験は特徴量追加のみ。1変更/実験の規約）

## 新規特徴量定義（3列）

| 列名 | 定義 | src/training.py 対応 |
|------|------|---------------------|
| `trn_hc_basediff_recent5` | race_date より厳密に前の坂路セッションのうち直近5本の 基準時計差 平均 | 坂路スピード_直近5本平均 |
| `trn_hc_basediff_best` | 同・キャリア全期間の 基準時計差 最小値（expanding min） | 坂路スピード |
| `trn_hc_accel_best` | 同・キャリア全期間の 加速（lap_400_200 − lap_200_0）最大値 | 坂路瞬発力 |

基準時計差 = `hc_4f_sec − median(hc_4f_sec | 同 training_date × training_center)`（全出走馬の調教データで計算。レース結果情報は不使用のためリークなし）

---

### Task 1: preprocess.py に hc_accel_sec を追加し HC_preprocessed を再生成

**Files:**
- Modify: `pure_rank/src/preprocess.py:190-238`（`preprocess_hc`）

**Interfaces:**
- Produces: `HC_preprocessed.parquet` に `hc_accel_sec` (float32, NaN許容) 列。既存6列（ketto_num, training_date, training_center, hc_3f_sec, hc_4f_sec, hc_200_sec）と行フィルタは**変更しない**（既存 trn_* 特徴量の再現性を保つため）

- [ ] **Step 1: 既存 parquet をバックアップ**

```bash
cd "C:/Users/syugo/AI/RaceAI_var1.0"
cp pure_rank/data/01_preprocessed/HC_preprocessed.parquet pure_rank/data/01_preprocessed/HC_preprocessed.bak.parquet
```

（パスが違う場合は `train_config.json` の `data.preprocessed_dir` を確認）

- [ ] **Step 2: preprocess_hc を修正**

`USE_COLS` に `lap_time_400_200` を追加:

```python
    USE_COLS = [
        "ketto_num", "training_date", "training_center",
        "time_4f_total", "time_3f_total", "lap_time_400_200", "lap_time_200_0",
    ]
```

数値変換ループの対象にも `lap_time_400_200` を追加:

```python
    for col in ["ketto_num", "time_4f_total", "time_3f_total", "lap_time_400_200", "lap_time_200_0"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
```

**行フィルタは変更しない。** タイム変換ブロックの直後に加速列を追加（lap_400_200 が無効な行は NaN のまま）:

```python
    # 加速 = lap(400-200m) − lap(200-0m)。正なら終い加速型（v50_hc_norm 用）。
    # lap_time_400_200 無効行を除外すると既存 trn_* 特徴量の行集合が変わるため、
    # 行は残して NaN にする。
    lap_42 = df["lap_time_400_200"].where(df["lap_time_400_200"] > 0)
    df["hc_accel_sec"] = ((lap_42 - df["lap_time_200_0"]) / 10.0).astype("float32")
```

`out_cols` に追加:

```python
    out_cols = ["ketto_num", "training_date", "training_center", "hc_3f_sec", "hc_4f_sec", "hc_200_sec", "hc_accel_sec"]
```

- [ ] **Step 3: HC のみ再生成**

```bash
cd "C:/Users/syugo/AI/RaceAI_var1.0"
python -c "
import sys; sys.path.insert(0, 'pure_rank/src')
from common import PROJECT_ROOT, load_config
from preprocess import preprocess_hc
cfg = load_config()
preprocess_hc(PROJECT_ROOT / cfg['data']['hc_dir'],
              PROJECT_ROOT / cfg['data']['preprocessed_dir'] / 'HC_preprocessed.parquet')
"
```

Expected: `[preprocess_hc] saved: ... | rows=N` （N はバックアップと同一行数）

- [ ] **Step 4: 検証**

```bash
python -c "
import pandas as pd
new = pd.read_parquet('pure_rank/data/01_preprocessed/HC_preprocessed.parquet')
old = pd.read_parquet('pure_rank/data/01_preprocessed/HC_preprocessed.bak.parquet')
assert len(new) == len(old), f'行数不一致: {len(new)} vs {len(old)}'
assert 'hc_accel_sec' in new.columns
assert (new[old.columns.tolist()].dtypes == old.dtypes).all()
print('rows:', len(new))
print('hc_accel_sec: NaN率 =', new.hc_accel_sec.isna().mean().round(4),
      '| mean =', new.hc_accel_sec.mean().round(3),
      '| p5/p95 =', new.hc_accel_sec.quantile([0.05, 0.95]).round(2).tolist())
"
```

Expected: 行数一致、NaN率 < 0.1 程度、mean が ±1.5 秒以内の常識的な値（加速は概ね -1〜+1 秒台）。異常なら Step 2 を見直す。

- [ ] **Step 5: Commit**

```bash
git add pure_rank/src/preprocess.py
git commit -m "feat(v50): add hc_accel_sec to HC preprocessing for basediff features"
```

---

### Task 2: _build_hc_norm_features を TDD で実装

**Files:**
- Create: `pure_rank/tests/test_hc_norm_features.py`
- Modify: `pure_rank/src/create_features.py`（SECTION 6 `_add_training_features` の直後に関数追加）

**Interfaces:**
- Consumes: Task 1 の `hc_accel_sec` 列を含む HC DataFrame
- Produces: `_build_hc_norm_features(df: pd.DataFrame, hc: pd.DataFrame) -> pd.DataFrame` — df（`race_id`/`ketto_num`(int64)/`race_date` 必須）に新規3列を付与して返す。行数不変。Task 3 の main() 統合がこの名前で呼ぶ

- [ ] **Step 1: 失敗するテストを書く**

`pure_rank/tests/test_hc_norm_features.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd "C:/Users/syugo/AI/RaceAI_var1.0"
python -m pytest pure_rank/tests/test_hc_norm_features.py -v
```

Expected: FAIL（`ImportError: cannot import name '_build_hc_norm_features'`）。pytest 未導入なら `pip install pytest`。

- [ ] **Step 3: 実装**

`pure_rank/src/create_features.py` の `_add_training_features` 定義の直後（SECTION 7 の前）に追加:

```python
def _build_hc_norm_features(df: pd.DataFrame, hc: pd.DataFrame) -> pd.DataFrame:
    """坂路 基準時計差 特徴量（v50_hc_norm 専用）。

    生の坂路タイムは調教場・日ごとの馬場差を含むため、同日×同調教場の
    全馬 median を基準にした相対値（基準時計差）に正規化する
    （src/training.py の 坂路スピード/瞬発力 と同じ発想）。
    集約は expanding（キャリア全期間）で、merge_asof の
    allow_exact_matches=False により training_date < race_date を厳密に保証する。
    """
    new_cols = ["trn_hc_basediff_recent5", "trn_hc_basediff_best", "trn_hc_accel_best"]
    if len(hc) == 0:
        for col in new_cols:
            df[col] = np.nan
        return df

    hc = hc.copy()
    baseline = hc.groupby(["training_date", "training_center"])["hc_4f_sec"].transform("median")
    hc["basediff"] = hc["hc_4f_sec"] - baseline

    hc = hc.sort_values(["ketto_num", "training_date"], kind="stable").reset_index(drop=True)
    g = hc.groupby("ketto_num")
    hc["trn_hc_basediff_best"] = g["basediff"].cummin()
    hc["trn_hc_basediff_recent5"] = (
        g["basediff"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    # cummax は NaN 行で NaN を返すため、馬内 ffill で直前までのベストを引き継ぐ
    hc["trn_hc_accel_best"] = g["hc_accel_sec"].cummax()
    hc["trn_hc_accel_best"] = hc.groupby("ketto_num")["trn_hc_accel_best"].ffill()

    hc_sorted = hc.sort_values("training_date", kind="stable").reset_index(drop=True)
    keys = (
        df[["race_id", "ketto_num", "race_date"]]
        .sort_values("race_date", kind="stable")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        keys,
        hc_sorted[["ketto_num", "training_date"] + new_cols],
        left_on="race_date",
        right_on="training_date",
        by="ketto_num",
        direction="backward",
        allow_exact_matches=False,
    )
    df = df.merge(
        merged[["race_id", "ketto_num"] + new_cols],
        on=["race_id", "ketto_num"], how="left",
    )
    return df
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python -m pytest pure_rank/tests/test_hc_norm_features.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pure_rank/tests/test_hc_norm_features.py pure_rank/src/create_features.py
git commit -m "feat(v50): add _build_hc_norm_features (HC baseline-diff, TDD)"
```

---

### Task 3: main() 統合・config 切替・特徴量生成（相関ゲート判定）

**Files:**
- Modify: `pure_rank/src/create_features.py`（main() の SECTION [6] 直後、`NEW_FEATURE_COLS_BY_VERSION`、列数 assert チェーン）
- Modify: `pure_rank/config/train_config.json`（`data.features_version`）

**Interfaces:**
- Consumes: Task 2 の `_build_hc_norm_features(df, hc)`
- Produces: `pure_rank/data/02_features/features_v50_hc_norm.parquet`（135列 = v39 の 132 + 3）と `manifest_v50_hc_norm.json`。Task 4 の train.py がこれを読む

- [ ] **Step 1: main() に version ゲート付きで呼び出しを追加**

`print("\n[7] Building labels (lr_label)...")` の直前（SECTION [6] の `df = _add_training_features(df, hc, wc)` の後）に:

```python
    if version == "v50_hc_norm":
        print("\n[6.5] Building HC baseline-diff features (v50_hc_norm)...")
        df = _build_hc_norm_features(df, hc)
```

- [ ] **Step 2: NEW_FEATURE_COLS_BY_VERSION にエントリ追加**

```python
    "v50_hc_norm": [
        "trn_hc_basediff_recent5",
        "trn_hc_basediff_best",
        "trn_hc_accel_best",
    ],
```

- [ ] **Step 3: 列数 assert チェーンに v50 分岐を追加**

既存の `elif version == ...` チェーン（v40 等と同形式）に:

```python
    elif version == "v50_hc_norm":
        assert len(df.columns) == 135, (
            f"v50_hc_norm は 135 列（v39_course_slim の 132 + 新規3列）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        for col in NEW_FEATURE_COLS_BY_VERSION["v50_hc_norm"]:
            assert col in df.columns, f"{col} がありません"
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 135")
```

（チェーンの実際の構造・他バージョンの書式に合わせること。135 が実際と食い違って assert に落ちた場合、まず自分の追加列以外の混入を疑う）

- [ ] **Step 4: config を v50 に切替**

`pure_rank/config/train_config.json` の `data.features_version` を `"v49_six_lap"` → `"v50_hc_norm"` に変更。**他のキーは一切変更しない。**

- [ ] **Step 5: 特徴量生成を実行**

```bash
cd "C:/Users/syugo/AI/RaceAI_var1.0"
python pure_rank/src/create_features.py 2>&1 | tee pure_rank/data/02_features/create_v50_hc_norm_log.txt
```

Expected:
- `[6.5] Building HC baseline-diff features (v50_hc_norm)...` が出る
- `[8.2] Correlation gate` で新規3列すべて `max|r| < 0.7 OK` → `相関ゲート: 全列 PASS`
- `Column count assert PASS: 135 == 135`
- `features_v50_hc_norm.parquet` 保存

**相関ゲート NG の場合（事前登録フォールバック）:**
- NG が `trn_hc_basediff_recent5` vs `trn_hc_basediff_best` の新規列同士のみ → `_build_hc_norm_features` から `trn_hc_basediff_best` の出力を外し（new_cols から削除、cummin 行を削除）、NEW_FEATURE_COLS_BY_VERSION と assert を 134 列に合わせて再実行
- 既存列との NG（例: trn_hc_best_3f_14d と r>=0.7）→ **ここで停止**し、NG ペアと r 値を報告して指揮官の判断を仰ぐ

- [ ] **Step 6: NaN 率と分布の妥当性確認**

生成ログの `[8] NaN rate report` で新規3列の NaN 率を確認。HC データは 2015 年以降のため、既存 `trn_hc_*` 列と同水準（学習期間全体で概ね 30〜60%）であること。既存列より極端に高い場合は merge キーの dtype 不一致を疑う。

- [ ] **Step 7: Commit**

```bash
git add pure_rank/src/create_features.py pure_rank/config/train_config.json
git commit -m "feat(v50): integrate hc_norm features into pipeline, switch features_version"
```

---

### Task 4: 学習（5シード×3フォールド）と評価

**Files:**
- 生成物のみ（コード変更なし）: `pure_rank/models/lambdarank_fold*_seed*.txt`、評価ログ

**Interfaces:**
- Consumes: `features_v50_hc_norm.parquet`（Task 3）
- Produces: TEST 指標（Top-1 / Top-3 / NDCG@3 / Spearman）を `pure_rank/data/02_features/evaluate_v50_hc_norm_log.txt` に記録

- [ ] **Step 1: 現行モデルをバックアップ**

```bash
cd "C:/Users/syugo/AI/RaceAI_var1.0"
mkdir -p pure_rank/models_backup_before_v50
cp pure_rank/models/lambdarank_fold*_seed*.txt pure_rank/models_backup_before_v50/
```

- [ ] **Step 2: アンサンブル学習（15モデル・長時間）**

```bash
python pure_rank/src/train.py --ensemble 2>&1 | tee pure_rank/data/02_features/train_v50_hc_norm_log.txt
```

30分以上かかる可能性があるためバックグラウンド実行し完了を待つこと。Expected: 15 モデルすべて `lambdarank_fold{1..3}_seed{42..46}.txt` として保存、early_stopping が 200〜400 本程度で発火。

- [ ] **Step 3: 評価**

```bash
python pure_rank/src/evaluate.py 2>&1 | tee pure_rank/data/02_features/evaluate_v50_hc_norm_log.txt
```

Expected: TEST（2025-01-01 以降、4,775 レース）の Top-1 / Top-3 / NDCG@3 / Spearman が出力される。

- [ ] **Step 4: リーク停止チェック**

Top-1 > 40% または Spearman > 0.6 の場合は**合格ではなく危険信号**。即座に停止し指揮官へ報告。

- [ ] **Step 5: 結果報告（コミットはしない）**

新規3列の feature importance（train ログ or モデルから gain 上位を確認）と、v39_course_slim（Top-1=30.24% / Top-3=61.76% / NDCG@3=0.5359 / Spearman=0.5048）との差分を報告する。**合否判定・採用判断はしない**（evaluator / 指揮官の役割）。

---

### Task 5: 合否判定後の後始末（指揮官の判定を受けて実行）

**Files:**
- Modify: `pure_rank/config/train_config.json`（不合格時のみ）

**分岐A: 合格（Top-1 > 30.24% かつ他指標が劣化なし）**
- evaluator エージェントによる独立検証（市場情報混入チェック・リーク確認）を経て採用。config は v50_hc_norm のまま、CLAUDE.md のベースライン記述更新は指揮官が行う

**分岐B: 不合格**

- [ ] **Step 1: config を v39_course_slim に戻す**

`data.features_version` → `"v39_course_slim"`。

- [ ] **Step 2: ベースライン再学習・再現確認**

```bash
python pure_rank/src/train.py --ensemble 2>&1 | tee pure_rank/data/02_features/train_v39_restore_after_v50_log.txt
python pure_rank/src/evaluate.py
```

Expected: Top-1 = 30.24% ±0.1pp（features_v39_course_slim.parquet は既存のため create_features 再実行は不要）

- [ ] **Step 3: 実装コードは記録として残す**（v40/v42 等と同じ扱い。version ゲートで無効化されているため削除不要）

```bash
git add pure_rank/config/train_config.json
git commit -m "chore(v50): restore v39_course_slim baseline after v50 experiment"
```

---

## Self-Review 済み事項

- 憲法1: 新規列は調教データのみ由来。`_check_no_market_features` が main() で2回走る
- 憲法2: `allow_exact_matches=False` + テスト `test_basediff_asof_strictly_before` で担保
- 憲法3: 合否ゲート・相関ゲートフォールバックとも本計画に事前登録
- 型整合: `_build_hc_norm_features(df, hc)` は Task 2 で定義し Task 3 の main() が同シグネチャで呼ぶ。`ketto_num` は SECTION [6] の `_add_training_features` が int64 化済みのため、[6.5] の配置で dtype 前提が成立する
