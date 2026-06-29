# Phase 2 実装仕様書: 調教特徴量

**日付**: 2026-06-29
**フェーズ**: Phase 2 — 調教データ（HC/WC）特徴量
**対象**: RaceAI_var1.0 — 市場情報なし純粋能力LambdaRank
**ベースライン**: Phase 1完了 (Top-1=25.5%)
**出力**: features_v2.parquet

---

## 目的

調教タイムは「馬が今どのような状態にあるか」を直接示す、市場情報を一切含まない純粋な能力指標である。
1番人気基準（Top-1≈30〜33%）への接近を目指し、Phase 1の過去走成績ベースライン（Top-1=25.5%）に対して調教状態の信号を追加する。

調教特徴量の設計思想:
- 調教セッション単体の絶対値（馬の現時点の仕上がり）
- 同レース出走馬との相対比較（同一条件下での優劣）
- 前レース前の調教との差分（状態の変化トレンド）

---

## 禁止特徴量の確認

- [ ] オッズ・人気を一切含まない（HC/WCデータには存在しないが、結合後に誤混入しないこと）
- [ ] 人気順位を含まない
- [ ] 当該レース以降の結果（着順・賞金）を調教特徴量の計算に使わない
- [ ] training_date >= race_date のデータを使わない（結合キーの不等号方向を必ず確認）

---

## 1. データソース確認

### HC（坂路調教）

| 項目 | 値 |
|------|-----|
| ファイルパス | `C:\Users\syugo\AI\RaceAI_var1.0\common\data\output\slop_hc\slop_hc_{year}.csv` |
| 年範囲 | 2015〜2026 |
| 2024年行数 | 約 482,296 行 |

**実際のカラム（CSVヘッダー確認済み）:**

| カラム名 | 型 | 内容 | 備考 |
|---------|-----|------|------|
| `record_id` | str | "HC" 固定 | 不要 |
| `data_kubun` | int | データ区分 | 不要 |
| `training_center` | int8 | トレセン区分 | 実データ: 0 or 1（注記参照） |
| `training_date` | str | 調教年月日 YYYYMMDD | datetime変換必須 |
| `training_time` | str | 調教時刻 HHMM | 不使用 |
| `ketto_num` | int64 | 血統登録番号（馬ID） | 結合キー |
| `time_4f_total` | int | 4F合計タイム（1/10秒単位） | 例: 660 = 66.0秒 |
| `lap_time_800_600` | int | 800-600m区間ラップ | 不使用 |
| `time_3f_total` | int | 3F合計タイム（1/10秒単位） | 例: 485 = 48.5秒 |
| `lap_time_600_400` | int | 600-400m区間ラップ | 不使用 |
| `time_2f_total` | int | 2F合計タイム | 不使用（3F/4Fで代替） |
| `lap_time_400_200` | int | 400-200m区間ラップ | 不使用 |
| `lap_time_200_0` | int | 最終200m区間ラップ（1/10秒） | 例: 160 = 16.0秒。重要 |
| `record_separator` | str | セパレータ | 不要 |
| `raw_hex` | str | 生データ | 不要 |

**training_center コードの注記（実データと仕様書の相違）:**
ユーザー仕様では「1=美浦, 2=栗東」と記載されているが、実際のCSVデータでは 0 と 1 の二値が観測される（0: 約233,603件、1: 約248,693件 / 2024年）。
WCデータとの照合から、training_center=0 が栗東、training_center=1 が美浦と推定されるが、
実装時に馬の所属（美浦/栗東）と照合して確認すること。
本仕様書では training_center をカテゴリ変数として そのまま使用（意味を断定しない）。

### WC（コース調教: CW・ウッドチップ等）

| 項目 | 値 |
|------|-----|
| ファイルパス | `C:\Users\syugo\AI\RaceAI_var1.0\common\data\output\wood_wc\wood_wc_{year}.csv` |
| 年範囲 | 2021〜2026（HC より範囲が短い） |
| 2024年行数 | 約 148,322 行 |

**実際のカラム（CSVヘッダー確認済み）:**

| カラム名 | 型 | 内容 | 備考 |
|---------|-----|------|------|
| `ketto_num` | int64 | 馬ID | 結合キー |
| `training_date` | str | 調教年月日 YYYYMMDD | datetime変換必須 |
| `training_center` | int8 | 0 or 1 | 0=栗東(CW), 1=美浦(ウッドチップ) |
| `course` | int8 | コース種別 | 3=栗東CW, 2=美浦ウッドチップ（実データ確認済み） |
| `track_direction` | int8 | 周回方向 | 不使用 |
| `time_4f_total` | int | 4F合計タイム（1/10秒） | |
| `lap_time_4f_3f` | int | 4F-3F区間 | 不使用 |
| `time_3f_total` | int | 3F合計タイム（1/10秒） | 主要指標 |
| `lap_time_3f_2f` | int | 3F-2F区間 | 不使用 |
| `time_2f_total` | int | 2F合計タイム | 不使用 |
| `lap_time_2f_1f` | int | 2F-1F区間 | 不使用 |
| `lap_time_1f_0f` | int | 最終1Fラップ（1/10秒） | 例: 147 = 14.7秒。重要 |
| `time_10f_total` 〜 `time_5f_total` | int | 5F〜10F合計 | 全て 0 の場合多い。不使用 |

---

## 2. 前処理仕様

### 2-1. HC_preprocessed.parquet

**読み込み列:**
```
ketto_num, training_date, training_center,
time_4f_total, time_3f_total, lap_time_200_0
```

**型変換・パース:**

```python
# training_date: YYYYMMDD文字列 → datetime
df['training_date'] = pd.to_datetime(df['training_date'].astype(str), format='%Y%m%d')

# タイム変換: 1/10秒単位 → 実秒 (float32)
# 例: 485 → 48.5秒
df['hc_3f_sec']  = (df['time_3f_total']  / 10.0).astype('float32')
df['hc_4f_sec']  = (df['time_4f_total']  / 10.0).astype('float32')
df['hc_200_sec'] = (df['lap_time_200_0'] / 10.0).astype('float32')
```

**除外条件（タイム無効行）:**

```python
# time_3f_total または time_4f_total が 0 の行は無効
df = df[
    (df['time_3f_total'] > 0) &
    (df['time_4f_total'] > 0) &
    (df['lap_time_200_0'] > 0)
].copy()
```

**出力 Parquet 列定義:**

| 列名 | dtype | 内容 |
|------|-------|------|
| `ketto_num` | int64 | 馬ID |
| `training_date` | datetime64[ns] | 調教日 |
| `training_center` | int8 | トレセン区分 (0/1) |
| `hc_3f_sec` | float32 | 3Fタイム（秒） |
| `hc_4f_sec` | float32 | 4Fタイム（秒） |
| `hc_200_sec` | float32 | 最終200mラップ（秒） |

**全年ファイルの結合方法:**

```python
import glob
files = glob.glob('C:/Users/syugo/AI/RaceAI_var1.0/common/data/output/slop_hc/slop_hc_*.csv')
hc_list = [pd.read_csv(f, encoding='utf-8-sig', usecols=[...]) for f in sorted(files)]
hc = pd.concat(hc_list, ignore_index=True)
```

### 2-2. WC_preprocessed.parquet

**読み込み列:**
```
ketto_num, training_date, training_center, course,
time_4f_total, time_3f_total, lap_time_1f_0f
```

**型変換・パース:**

```python
df['training_date'] = pd.to_datetime(df['training_date'].astype(str), format='%Y%m%d')

df['wc_3f_sec'] = (df['time_3f_total']   / 10.0).astype('float32')
df['wc_4f_sec'] = (df['time_4f_total']   / 10.0).astype('float32')
df['wc_1f_sec'] = (df['lap_time_1f_0f'] / 10.0).astype('float32')
```

**除外条件:**

```python
df = df[
    (df['time_3f_total'] > 0) &
    (df['time_4f_total'] > 0) &
    (df['lap_time_1f_0f'] > 0)
].copy()
```

**出力 Parquet 列定義:**

| 列名 | dtype | 内容 |
|------|-------|------|
| `ketto_num` | int64 | 馬ID |
| `training_date` | datetime64[ns] | 調教日 |
| `training_center` | int8 | 0=栗東CW, 1=美浦ウッドチップ |
| `course` | int8 | コース種別 (2 or 3) |
| `wc_3f_sec` | float32 | 3Fタイム（秒） |
| `wc_4f_sec` | float32 | 4Fタイム（秒） |
| `wc_1f_sec` | float32 | 最終1Fラップ（秒） |

---

## 3. 調教特徴量の結合方法

### 結合の基本構造

```
SE（race_date + ketto_num）
    ↓ merge_asof (最近接セッション取得)
HC_preprocessed / WC_preprocessed
    ↓ window aggregation (14日以内集計)
カテゴリA特徴量（絶対値系）
    ↓ groupby(race_id) rank/zscore
カテゴリB特徴量（同レース内相対比較系）
    ↓ groupby(ketto_num).shift(1)
カテゴリC特徴量（過去走との比較系）
```

### merge_asof: 最近接セッション取得

`pd.merge_asof` を使い、レース当日より前の直近1セッションを取得する。

```python
# 前提: 両DataFrameをソート済みにすること（sort_valuesが必須）
hc_sorted = hc.sort_values('training_date')
se_keys = main_df[['ketto_num', 'race_id', 'race_date']].sort_values('race_date')

last_hc = pd.merge_asof(
    se_keys,
    hc_sorted[['ketto_num', 'training_date', 'hc_3f_sec', 'hc_4f_sec', 'hc_200_sec']],
    left_on='race_date',
    right_on='training_date',
    by='ketto_num',
    direction='backward',        # race_date 以前の最直近セッション
    tolerance=pd.Timedelta(days=14),  # 14日超え前は NaN
)
# 得られる列: trn_hc_last_3f_sec, trn_hc_last_4f_sec, trn_hc_last_200_sec
```

**注意**: `merge_asof` に渡す DataFrame は必ず `by` キー（ketto_num）でも安定ソートすること。
`sort_values(['ketto_num', 'training_date'])` を使い、その上で `left_on`/`right_on` の列でもソートする。

### 14日ウィンドウ集計: 最速タイム・セッション数

`merge_asof` は1行（最直近）しか取得できない。14日以内の全セッションを集計するには、
以下の「マージ後フィルタ」方式を使用する。

```python
# 1. ketto_num で内部結合（cross-join 相当）
merged = main_df[['ketto_num', 'race_id', 'race_date']].merge(
    hc[['ketto_num', 'training_date', 'hc_3f_sec', 'hc_200_sec']],
    on='ketto_num',
    how='left'
)

# 2. ウィンドウフィルタ: 0 < race_date - training_date <= 14日
diff_days = (merged['race_date'] - merged['training_date']).dt.days
mask = (diff_days > 0) & (diff_days <= 14)
window = merged[mask].copy()

# 3. 集計
hc_agg = window.groupby(['race_id', 'ketto_num']).agg(
    trn_hc_best_3f_14d  =('hc_3f_sec',  'min'),  # 最速（最小値）
    trn_hc_best_200_14d =('hc_200_sec', 'min'),
    trn_hc_count_14d    =('training_date', 'count'),
).reset_index()
```

**メモリ注意**: HC全期間（2015-2026）は約5百万行規模になる。
ketto_numでのマージ前に、SEに存在するketto_numのみに絞り込んでからマージすること。

```python
active_horses = set(main_df['ketto_num'].unique())
hc_filtered = hc[hc['ketto_num'].isin(active_horses)]
```

---

## 4. 特徴量リスト（全22特徴量）

### カテゴリ A: 絶対値系（13特徴量）

| 列名 | ソース | 計算方法 | リーク防止 | 欠損条件 |
|------|--------|---------|-----------|---------|
| `trn_hc_last_3f_sec` | HC | merge_asof、14日以内直近3Fタイム（秒） | training_date < race_date | 14日以内HC調教なし |
| `trn_hc_last_4f_sec` | HC | merge_asof、14日以内直近4Fタイム（秒） | 同上 | 同上 |
| `trn_hc_last_200_sec` | HC | merge_asof、14日以内直近最終200mラップ（秒） | 同上 | 同上 |
| `trn_hc_best_3f_14d` | HC | 14日以内最速HC 3Fタイム（秒）= min | 同上 | 同上 |
| `trn_hc_best_200_14d` | HC | 14日以内最速HC 最終200mラップ（秒）= min | 同上 | 同上 |
| `trn_hc_count_14d` | HC | 14日以内HCセッション数（整数） | 同上 | 0の場合はNaNではなく0 |
| `trn_wc_last_3f_sec` | WC | merge_asof、14日以内直近WC 3Fタイム（秒） | 同上 | 2015-2020 は全NaN |
| `trn_wc_last_4f_sec` | WC | merge_asof、14日以内直近WC 4Fタイム（秒） | 同上 | 同上 |
| `trn_wc_last_1f_sec` | WC | merge_asof、14日以内直近WC 最終1Fラップ（秒） | 同上 | 同上 |
| `trn_wc_best_3f_14d` | WC | 14日以内最速WC 3Fタイム（秒）= min | 同上 | 同上 |
| `trn_wc_best_1f_14d` | WC | 14日以内最速WC 最終1Fラップ（秒）= min | 同上 | 同上 |
| `trn_wc_count_14d` | WC | 14日以内WCセッション数（整数） | 同上 | 0の場合は0 |
| `trn_total_count_14d` | HC+WC | HC + WC 合計セッション数（14日以内） | 同上 | 欠損はNaN（不明扱い） |

**タイムの解釈**: 小さいほど速い（良い仕上がり）。LightGBMに解釈を委ねる。

### カテゴリ B: 同レース内相対比較系（5特徴量）

同レース出走馬の `trn_hc_best_3f_14d` / `trn_wc_best_3f_14d` を使い、
レース内での相対的な仕上がり優劣を数値化する。

| 列名 | ソース | 計算方法 | 欠損処理 |
|------|--------|---------|---------|
| `trn_hc_rank_3f` | HC(A) | `groupby('race_id')['trn_hc_best_3f_14d'].rank(method='min', ascending=True, na_option='bottom')` | 欠損馬は最下位（最も遅いとみなす） |
| `trn_hc_zscore_3f` | HC(A) | `groupby('race_id')['trn_hc_best_3f_14d'].transform(lambda x: (x - x.mean()) / x.std())` | 欠損はNaN（Zスコア計算から除外） |
| `trn_hc_rank_200` | HC(A) | `groupby('race_id')['trn_hc_best_200_14d'].rank(method='min', ascending=True, na_option='bottom')` | 同上 |
| `trn_wc_rank_3f` | WC(A) | `groupby('race_id')['trn_wc_best_3f_14d'].rank(method='min', ascending=True, na_option='bottom')` | 欠損はNaN（2021+のみ有効） |
| `trn_wc_zscore_3f` | WC(A) | `groupby('race_id')['trn_wc_best_3f_14d'].transform(lambda x: (x - x.mean()) / x.std())` | 欠損はNaN |

**Zスコアの実装例:**

```python
def zscore_within_race(series: pd.Series) -> pd.Series:
    """NaNを除いた有効馬のみでZスコア計算。欠損馬はNaN。"""
    mean = series.mean()  # skipna=True がデフォルト
    std  = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std

df['trn_hc_zscore_3f'] = (
    df.groupby('race_id')['trn_hc_best_3f_14d']
    .transform(zscore_within_race)
)
```

**同レース内のデータ欠損率について**: WCは2021+のみのため、
2021年以前のレースでは`trn_wc_rank_3f`/`trn_wc_zscore_3f`が全馬NaNになる（許容）。
HC系は2015+のデータがあるため、欠損率は調教データ未記録馬のみ。

### カテゴリ C: 過去走との比較系（4特徴量）

「今回の調教準備」と「前レース前の調教準備」を比較し、仕上がりの変化トレンドを示す。
リーク防止: `shift(1)` で前レース（直前の出走日）のA特徴量と差分を計算する。

**計算順序:**
1. カテゴリA特徴量を全行について計算済みの DataFrame を使用する
2. `sort_values(['ketto_num', 'race_date'])` でソート
3. `groupby('ketto_num')` で `shift(1)` を適用

| 列名 | 計算式 | 解釈 |
|------|--------|------|
| `trn_hc_3f_delta` | `trn_hc_best_3f_14d - shift(1)[trn_hc_best_3f_14d]` | 負 = 今回の方が速い（好調） |
| `trn_hc_200_delta` | `trn_hc_best_200_14d - shift(1)[trn_hc_best_200_14d]` | 負 = 末脚が鋭くなった |
| `trn_wc_3f_delta` | `trn_wc_best_3f_14d - shift(1)[trn_wc_best_3f_14d]` | 負 = 今回の方が速い |
| `trn_count_delta` | `trn_total_count_14d - shift(1)[trn_total_count_14d]` | 正 = 調教量が増えた |

**実装例:**

```python
df = df.sort_values(['ketto_num', 'race_date'])
grp = df.groupby('ketto_num')

df['trn_hc_3f_delta']   = df['trn_hc_best_3f_14d']   - grp['trn_hc_best_3f_14d'].shift(1)
df['trn_hc_200_delta']  = df['trn_hc_best_200_14d']   - grp['trn_hc_best_200_14d'].shift(1)
df['trn_wc_3f_delta']   = df['trn_wc_best_3f_14d']    - grp['trn_wc_best_3f_14d'].shift(1)
df['trn_count_delta']   = df['trn_total_count_14d']    - grp['trn_total_count_14d'].shift(1)
```

**リーク確認**: カテゴリCの `shift(1)` は当該レース自身のA特徴量（=前レース前の調教まとめ）を除外し、
前レース前の調教データを参照する。`training_date < race_date` の制約は既にAで保証されているため、
Cは追加のリーク防止は不要。

---

## 5. 欠損値ポリシー

| 欠損パターン | 対処 | 理由 |
|------------|------|------|
| WCの2015-2020レース | NaN のまま | WCデータが存在しない年。LightGBMが自動処理 |
| 14日以内に調教なし（休養直後など） | NaN のまま | 調教情報がないことも情報。0埋めしない |
| カテゴリC: 初レース（前レースなし） | NaN のまま | shift(1)で自然にNaN。問題なし |
| Zスコア: レース内全馬NaN | NaN のまま | 全馬が調教データなしの場合。稀 |
| `trn_hc_count_14d` = 0 | **0として記録**（NaNではない） | 0セッションは情報（調教なし）を意味する |

**LightGBMへの渡し方**: 全てのNaNをそのまま渡す（fillnaしない）。
LightGBMは内部でNaNを特別扱いするため、欠損パターン自体が学習される。

---

## 6. create_features.py への組み込み方針

### 追加するセクション（既存の SECTION 3〜6 の後に挿入）

```
SECTION 3: HISTORICAL FEATURES（既存）
SECTION 4: CURRENT FEATURES（既存）
SECTION 5: BLOODLINE FEATURES（既存）
SECTION 6: LABELS（既存）
SECTION 7: TRAINING FEATURES（新規）  ← ここに追加
    7-1: HC/WC Parquetの読み込み
    7-2: カテゴリA: 最近接セッション (merge_asof)
    7-3: カテゴリA: 14日ウィンドウ集計
    7-4: カテゴリA: 合計セッション数
    7-5: カテゴリB: 同レース内相対比較
    7-6: カテゴリC: 過去走との差分 (shift(1))
```

### 関数シグネチャ（実装者への参考）

```python
def _load_hc(cfg: dict) -> pd.DataFrame:
    """HC_preprocessed.parquet を読み込む。"""

def _load_wc(cfg: dict) -> pd.DataFrame:
    """WC_preprocessed.parquet を読み込む。なければ空DataFrameを返す。"""

def _add_training_features(
    df: pd.DataFrame,
    hc: pd.DataFrame,
    wc: pd.DataFrame,
) -> pd.DataFrame:
    """調教特徴量を df に追加して返す。
    
    Parameters
    ----------
    df : フィルタ済み・SE+RA+SK結合済みの DataFrame
    hc : HC_preprocessed.parquet
    wc : WC_preprocessed.parquet（2021+のみ）
    
    Returns
    -------
    pd.DataFrame : trn_ プレフィックスの22列が追加された DataFrame
    """
```

---

## 7. preprocess.py への追加仕様

既存の `preprocess.py` は `horse_data.parquet`（RaceAI_var2.0.0由来）から SE/RA/SK を生成している。
HC/WC は独立した CSV ファイル群であるため、以下の関数を **新規追加** する。

```python
def preprocess_hc(hc_dir: Path, dst_parquet: Path) -> pd.DataFrame:
    """HC（坂路調教）CSV全年をまとめて前処理して保存する。
    
    Parameters
    ----------
    hc_dir : slop_hc_{year}.csv が入ったディレクトリ
    dst_parquet : 保存先（HC_preprocessed.parquet）
    """

def preprocess_wc(wc_dir: Path, dst_parquet: Path) -> pd.DataFrame:
    """WC（コース調教）CSV全年をまとめて前処理して保存する。
    
    Parameters
    ----------
    wc_dir : wood_wc_{year}.csv が入ったディレクトリ
    dst_parquet : 保存先（WC_preprocessed.parquet）
    """
```

`preprocess.py` の `main()` からこれらを呼び出し、
`pure_rank/data/01_preprocessed/HC_preprocessed.parquet` と
`pure_rank/data/01_preprocessed/WC_preprocessed.parquet` を生成する。

---

## 8. train_config.json への変更事項

| キー | 現在値 | 変更後 | 変更理由 |
|------|--------|--------|---------|
| `data.features_version` | `"v1"` | `"v2"` | features_v2.parquet を出力するため |
| `data.hc_dir` | （なし） | `"C:/Users/syugo/AI/RaceAI_var1.0/common/data/output/slop_hc"` | HC CSVのパス |
| `data.wc_dir` | （なし） | `"C:/Users/syugo/AI/RaceAI_var1.0/common/data/output/wood_wc"` | WC CSVのパス |

カテゴリ特徴量リストへの追加は不要（trn_ 系は全て連続値として使用）。

---

## 9. ディレクトリ構造（Phase 2完了後）

```
pure_rank/
├── config/
│   └── train_config.json             ← features_version: "v2", hc_dir/wc_dir追加
├── data/
│   ├── 01_preprocessed/
│   │   ├── SE_preprocessed.parquet   （既存）
│   │   ├── RA_preprocessed.parquet   （既存）
│   │   ├── SK_preprocessed.parquet   （既存）
│   │   ├── HC_preprocessed.parquet   ← 新規（2015-2026）
│   │   └── WC_preprocessed.parquet   ← 新規（2021-2026）
│   └── 02_features/
│       ├── features_v1.parquet        （Phase 1、バックアップ保持）
│       └── features_v2.parquet        ← 新規出力
├── models/
│   └── lambdarank_fold*_seed*.txt    （Phase 2学習後に更新）
└── src/
    ├── preprocess.py                  ← preprocess_hc / preprocess_wc を追加
    ├── create_features.py             ← SECTION 7（調教特徴量）を追加
    ├── train.py                       （変更なし）
    └── evaluate.py                    （変更なし）
```

---

## 10. 評価基準

| 指標 | Phase 1基準 | Phase 2目標 | リーク停止閾値 |
|------|------------|------------|--------------|
| Top-1 的中率 | 25.5% | >27% | >40%（即停止） |
| NDCG@3 | — | >0.50 | — |
| Spearman相関 | — | >0.48 | >0.60（即停止） |
| テスト件数 | — | 500レース以上 | — |

Phase 1の Top-1=25.5% を下回った場合は調教特徴量の効果がマイナスとみなし、
evaluatorの判断で `trn_` 系列を全削除してPhase 1状態に戻す。

---

## 11. implementer への引き渡し事項

以下をこの順序で実装・実行すること。

### タスク 1: preprocess.py への追加（前処理）

- `preprocess_hc(hc_dir, dst)`: HC CSV全年読み込み → タイム変換 → 除外フィルタ → Parquet保存
- `preprocess_wc(wc_dir, dst)`: WC CSV全年読み込み → タイム変換 → 除外フィルタ → Parquet保存
- `main()` から両関数を呼び出す

確認ポイント:
- `time_3f_total == 0` の行が除外されていること
- `training_date` が datetime 型になっていること
- HC: 約2〜5百万行規模が出力されること（全年合計）

### タスク 2: train_config.json の更新

- `features_version`: `"v1"` → `"v2"`
- `hc_dir` / `wc_dir` の追加

### タスク 3: create_features.py への追加（SECTION 7）

- `_load_hc(cfg)` / `_load_wc(cfg)` 関数
- `_add_training_features(df, hc, wc)` 関数（A → B → C の順で計算）
- `main()` の `[5] Building bloodline features` の直後に呼び出し

実装順序（関数内部）:
1. カテゴリA: merge_asof で最近接セッション取得
2. カテゴリA: 14日ウィンドウ集計（マージ後フィルタ方式）
3. カテゴリA: 合計セッション数計算
4. カテゴリB: groupby(race_id) での rank/zscore（A完了後に計算）
5. カテゴリC: sort_values + groupby(ketto_num).shift(1)（A完了後に計算）

### タスク 4: 実行と検証

```bash
# 前処理実行
python pure_rank/src/preprocess.py

# 特徴量生成（v2）
python pure_rank/src/create_features.py

# 市場情報混入チェック（必須）
grep -rn "odds\|popularity\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"

# 学習
python pure_rank/src/train.py --ensemble

# 評価
python pure_rank/src/evaluate.py
```

### タスク 5: 評価後の確認

- evaluator に `features_v2.parquet` の NaN率レポートを提出する
- `trn_wc_*` 系は 2015-2020 で高NaN率（正常）
- `trn_hc_*` 系の NaN率が 50% 超の場合は merge ロジックの見直しを行う

---

## 12. 禁止事項の確認チェックリスト

- [ ] 調教データに odds / popularity カラムが存在しないことを確認した
- [ ] `training_date >= race_date` の行が結合後に残っていないことを確認した（`diff_days > 0` を必ず使う）
- [ ] カテゴリB（同レース内比較）はレース結果（finish_rank）を使っていないことを確認した
- [ ] カテゴリCのshift(1)で当該レースのA特徴量が除外されていることを確認した
- [ ] `features_v1.parquet` のバックアップが存在することを確認してから `features_v2.parquet` を生成した
- [ ] Top-1 > 40% が出た場合は即座に停止してevaluatorに連絡した

---

*以上が Phase 2 調教特徴量の完全な実装仕様書である。*
*planner担当: 2026-06-29*
