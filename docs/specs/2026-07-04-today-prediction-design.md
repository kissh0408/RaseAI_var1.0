# 実装仕様書: 当日レース予測機能 — 2026-07-04

## 0. 目的

現状 RaceAI_var1.0 は「過去データに対する学習・評価」までしか対応しておらず、
**当日実際に行われるレースに対して事前予測を出す**機能が存在しない。

本仕様は、当日の出走馬データを JV-Link から取得し、既存の
`features_v39_course_slim`（132列）特徴量パイプライン・15モデル
（3fold×5seed）LambdaRank アンサンブルに通し、開催場所ごと・
馬場状態シナリオ（良/稍重/重/不良）ごとに着順予測を出力する
Jupyter notebook 実行フローを新設する。

**市場情報不使用の原則は当日予測でも変わらない。** 当日データ取得フローには
単勝オッズ取得が付随するが、特徴量化・推論には一切使わない。

### なぜ4馬場状態シナリオが必要か

発走前（特にレース前日〜当日朝の時点）では、その日最終的にどの馬場状態
（良/稍重/重/不良）が発表されるか確定していない。天候変化・散水・時間経過で
悪化/回復しうる。本モデルは `track_condition_code` に依存する特徴量を持つため、
「もし良馬場なら」「もし重馬場なら」という4つの仮定でそれぞれ推論し、
的中判断・馬券判断はユーザー自身が直前の馬場発表を見て該当フォルダを参照する
運用を想定する。

---

## 1. 禁止特徴量の確認

- [x] オッズ系データ（`odds`, `win_odds`, `place_odds`, `quinella_odds` 等）を特徴量に含めない
- [x] 人気順位（`popularity`, `ninki`）を含めない
- [x] `run_today_se_ra_and_realtime_merge` は内部で `refresh_today_odds_data`
      （0B31 速報単勝オッズ取得）を実行し `race_se.csv` の `odds` 列を更新するが、
      この列は当日予測パイプラインの特徴量化ステップで **明示的に drop** する
      （後述 4-B）
- [x] `FORBIDDEN_MARKET_COLS`（`common.py`）による混入チェックを当日パイプラインの
      最終出力にも必ず適用する

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ main/ --include="*.py"
```

---

## 2. 現状資産の要点（実装前に必ず確認済みの事実）

implementer が実装を始める前に誤解しないよう、調査で確認した事実を明記する。

### 2-1. `create_features.py`（132列, `v39_course_slim`）の構造

`_load_data()` → `_apply_filters()` → `_check_no_market_features()` →
`_build_course_geometry_features()` → `_build_hist_features()` →
`_build_current_features()` → `_build_sire_features()` →
`_build_jockey_trainer_features()` → `_build_speed_index_features()` →
`_build_relative_features()` → `_add_training_features()`（HC/WC）→
`_build_labels()` の順で実行され、最後に列数アサート
（`v39_course_slim` は **132列固定**）・相関ゲート・行順ソート
（`race_date, race_id, horse_num`）を経て parquet 保存される。

当日予測はこの **同一関数群を同一順序で** 通す必要がある。独自に特徴量を
再実装すると学習時との計算方式のズレ（=精度劣化・再現性喪失）が起きるため、
`create_features.py` の関数を import して再利用する設計とする
（後述 4-C）。

### 2-2. `track_condition_code` に依存する特徴量（4シナリオで値が変わる列）

コード全文を精査した結果、132列のうち **track_condition_code の値そのもの**
（またはそれから導出される `surface_condition = surface_code*10 + track_condition_code`）
を **groupby キーに使っている**列は以下の3列のみ。この3列は「今日のレースが
どの馬場状態か」によって、対象馬の過去走のうち集計対象になるレースの
サブセットが変わるため、シナリオごとに値が変わる。

| 列名 | 生成関数 | 依存理由 |
|------|---------|---------|
| `hist_same_condition_win_rate` | `_build_hist_features` | `groupby(["ketto_num", "track_condition_code"])` → 「この馬のこの馬場状態での過去勝率」。今日の想定馬場状態によって参照する過去走の集合が変わる |
| `hist_surface_condition_win_rate` | `_build_hist_features` | `groupby(["ketto_num", "surface_condition"])`。`surface_condition` は `surface_code*10+track_condition_code` なので同様に依存 |
| `hist_best_time_same_cond` | `_build_hist_features` | `groupby(["ketto_num", "distance", "surface_code", "track_condition_code"])` → 「同条件での自己ベストタイム」 |

これ以外の `hist_*` 列（`hist_win_rate`, `hist_avg_rank_3`,
`hist_speed_idx_last/best/avg3/cond_best`, `hist_sire_*`, `hist_jockey_*`,
`hist_trainer_*` 等 129列）は、当該馬・産駒・騎手・調教師の**自分の時系列上
での shift(1)** のみに依存し、`track_condition_code` をグループキーに使わない。
そのため今日の想定馬場状態を変えても値は変わらない。

`_build_speed_index_features` 内の `cond_avg_time`/`cond_std_time`（レース日次
集計、`track_condition_code` を含むグループキー）は中間変数であり、
`_speed_idx` を経由して `hist_speed_idx_*` に反映されるが、`_speed_idx` 自体は
その馬自身の**過去**レースの `racetime`（既に確定済みの実測値）に対して
計算されたものであり、今日のレース行の `racetime` は NaN（未実施）なので
`hist_speed_idx_*` は今日の想定馬場状態と無関係に確定する（自馬の過去実績の
集計のため）。

**設計原則**: 132列中 129列は「馬場シナリオに依存しない共通計算」であり、
1回計算すれば4シナリオで使い回せる。再計算が必要なのは上記3列のみ。
4シナリオ分すべてについて 1736行のパイプライン全体（sire/jockey/trainer
集計を含む重い処理）を4回走らせるのは非効率かつ「同一入力から同一出力が
出るはず」という不変条件のテストがしにくくなるため、**共通部分は1回だけ計算し、
シナリオ依存3列だけを軽量に差し替える** 設計を implementer に指示する
（詳細は 4-D）。

### 2-3. データ除外条件フィルタの当日レースへの不適用

`_apply_filters()` は `finish_rank > 0`（着順確定済み）を必須条件にしている。
当日の未実施レースは `finish_rank` が存在しない（0 or NaN）ため、このフィルタを
そのまま当日行に適用すると全馬が除外されてしまう。

→ **当日行にはこのフィルタを適用しない**。ただし当日行を含めて
`_build_hist_features` 等を計算する母集団（過去の全履歴行）には、
学習時と同じ `_apply_filters()` を適用し続ける（学習時の特徴量分布とズレさせない
ため）。実装方法は 4-C 参照。

`abnormal_code`（当日行は未確定 → NaN）はフィルタ条件 `~isin([1,3,4])` に対して
NaN は False 扱いにならず「含まれない」＝ True 判定になる（pandas の `isin` は
NaN を False として扱うため `~isin` は True になる）ため、当日行が誤って
除外される心配はない。`grade_code`・`horse_count`（`running_count` 由来、
出馬表確定時点で判明）は当日行にも正しく入るため通常通りフィルタしてよい。

### 2-4. race_id の構築方法

`race_id` は var2.0.0 の `horse_data.parquet` に由来する列で、
`preprocess.py` の `_make_race_id()`（HR前処理専用関数だが命名規則は共通）を見ると

```
race_id = year(4桁) + month_day(4桁) + course_code(2桁) + kai(2桁) + nichi(2桁) + race_num(2桁)
```

の16桁文字列である。当日データにも `year, month_day, course_code, kai, nichi,
race_num` は RA から直接取得できるため、同じ規則で今日の `race_id` を組み立てれば
既存パイプラインの `race_id` キー（merge・group化に使用）と整合する。

### 2-5. `course_code` → 開催場所名の対応表

`common/data/src/clean_cushion_data.py` の `COURSE_NAME_TO_CODE` が
JV-Link/前処理と同一のコード体系であることが確認できた。

| course_code | 開催場所名 |
|---|---|
| 1 | 札幌 |
| 2 | 函館 |
| 3 | 福島 |
| 4 | 新潟 |
| 5 | 東京 |
| 6 | 中山 |
| 7 | 中京 |
| 8 | 京都 |
| 9 | 阪神 |
| 10 | 小倉 |

この逆引き辞書（`COURSE_CODE_TO_NAME`）をどこか1箇所（`common.py` 推奨）に
定義し、フォルダ名生成に使う。

### 2-6. `track_condition_code` → 日本語ラベル対応

| track_condition_code | ラベル |
|---|---|
| 1 | 良 |
| 2 | 稍重 |
| 3 | 重 |
| 4 | 不良 |

（`0` = コード無し/障害だが、`_apply_filters` で障害 [grade_code=9] は除外済みのため
当日の平地レースでは出現しない想定。ただし implementer は防御的に 0 が来た場合の
挙動＝スキップまたは警告ログを実装すること）

### 2-7. HC/WC（調教データ）は当日予測に必須

`train_config.json` の `features_version=v39_course_slim` には
`trn_hc_*` / `trn_wc_*`（坂路調教・ウッドチップ調教、直近14日）が **12列**
含まれる（`_add_training_features`）。これは `pure_rank/data/01_preprocessed/`
の `HC_preprocessed.parquet` / `WC_preprocessed.parquet` を読み込むが、
これらは JV-Link の SLOP/WOOD dataspec から **別途取得**する必要があり、
`run_today_se_ra_and_realtime_merge` ではカバーされない
（RA/SE + 速報 + オッズのみ取得する関数のため）。

→ 当日予測を行うには、レース当日までの直近調教データを含む
最新の HC/WC を **別途 `fetch_hc_only` / `fetch_wc_only` で当年分を再取得**し、
`preprocess.py` の `preprocess_hc` / `preprocess_wc` で
`HC_preprocessed.parquet` / `WC_preprocessed.parquet` を更新しておく必要がある
（4-A 手順内で明記）。

なお **TM（タイム指数）特徴量は現行 `v39_course_slim` の132列に含まれていない**
（`train_config.json` に `tm_dir` が存在せず、`create_features.py` にも
TM 由来の列がない）。プロジェクト憲法ドキュメントの「Phase 5: TM追加」は
将来の別Phaseの話であり、当日予測機能の実装ではTMデータ取得は不要。

### 2-8. `preprocess.py` の `src_parquet_dir` はプロジェクト外（var2.0.0）を参照している

`train_config.json` の `data.src_parquet_dir` は
`C:/Users/syugo/AI/RaceAI_var2.0.0/model_training/data/01_preprocessed` を指しており、
`preprocess.py`（通常モード）はここの `horse_data.parquet`（var2.0.0側の統合済み
テーブル、オッズ列を含む）を読んで var1.0 用に絞り込む設計になっている。

当日データは var1.0 内の JV-Link 取得フロー
（`run_today_se_ra_and_realtime_merge` → `main/data/race/race_ra.csv`,
`race_se.csv`）でのみ得られ、var2.0.0 の `horse_data.parquet` には
**当日レースはまだ反映されていない**（var2.0.0 側のバッチ更新を待つ必要があり、
当日中の即時性が失われる）。

→ 当日予測パイプラインは var2.0.0 の `horse_data.parquet` を経由せず、
`main/data/race/race_ra.csv` / `race_se.csv`（当日生データ）を
**直接** `SE_preprocessed.parquet` / `RA_preprocessed.parquet` と同じ列名・
dtype に変換するアダプタを新規に持つ（4-Bで詳細）。既存の
`preprocess_se()` / `preprocess_ra()` をそのまま使い回すことはできない
（入力ソースのスキーマが異なるため）。ただし変換後の**出力スキーマ**は
完全に一致させ、`create_features.py` の関数がそのまま使えるようにする。

---

## 3. データフロー設計

```
[ステップ0: 事前準備（ユーザー実行、当日朝など）]
  0-1. HC/WC 当年分を再取得: fetch_hc_only(cur_year, cur_year), fetch_wc_only(cur_year, cur_year)
  0-2. preprocess.py 相当の HC/WC 前処理のみ再実行し
       pure_rank/data/01_preprocessed/{HC,WC}_preprocessed.parquet を更新

[ステップ1: 当日データ取得]
  1-1. run_today_se_ra_and_realtime()（32bit委譲込み、既存）
       → main/data/race/race_ra.csv, race_se.csv 更新（RA/SE/速報/オッズ反映済み）
  1-2. 出馬取消・除外馬（当日発表分）があれば abnormal_code に反映されている前提

[ステップ2: 当日データ → 既存前処理スキーマへの変換（新規）]
  2-1. race_ra.csv, race_se.csv を読み込み
  2-2. race_id を生成（year+month_day+course_code+kai+nichi+race_num）
  2-3. SE_preprocessed.parquet / RA_preprocessed.parquet と同一の列名・dtypeに変換
       （odds, popularity 等の市場列はこの時点で明示的に drop）
  2-4. SK（血統）は ketto_num で SK_preprocessed.parquet と結合（既存馬なら
       sire_id/bms_id が引ける。新規登録馬で欠損の場合は NaN のまま許容）

[ステップ3: 履歴データとの結合]
  3-1. SE_preprocessed.parquet / RA_preprocessed.parquet をロード
  3-2. 2で作った当日行を末尾に concat（race_date は当日日付になるため
       時系列ソートで自然に最後尾に来る）
  3-3. 過去データ側にのみ _apply_filters() の finish_rank>0 等を適用し、
       当日行は「フィルタ対象外」として無条件で残す
       （2-3節の設計に基づくマスク処理。詳細は4-C）

[ステップ4: 特徴量パイプライン適用（共通部分・1回のみ）]
  4-1. 当日行の track_condition_code に暫定値（例: 1=良）を仮置きする
       （3列以外は暫定値に依存しないため、どの値でも129列は正しく計算される）
  4-2. create_features.py の内部関数を順番に呼び出す
       （_build_course_geometry_features → _build_hist_features →
        _build_current_features → _build_sire_features →
        _build_jockey_trainer_features → _build_speed_index_features →
        _build_relative_features → _add_training_features）
  4-3. 当日行のみを抽出し、129列（シナリオ非依存）を確保する

[ステップ5: シナリオ依存3列の差し替え（4パターン × 軽量計算）]
  5-1. track_condition_code ∈ {1,2,3,4} それぞれについて、
       hist_same_condition_win_rate / hist_surface_condition_win_rate /
       hist_best_time_same_cond を当該馬の過去走履歴（既にロード済み・
       フィルタ済みの履歴df）から直接集計し直す
       （重いパイプライン全体の再実行ではなく、対象3列のみのgroupby集計）
  5-2. 4-3の129列 + 5-1の3列を結合し、132列の当日特徴量DataFrameを
       シナリオごとに4つ作る

[ステップ6: 推論]
  6-1. get_feature_cols() と同じロジックで特徴量列を選択
  6-2. load_models()で15モデル（3fold×5seed）をロード
  6-3. ensemble_predict()でシナリオごとにスコアを算出
  6-4. レース内順位（1位予想〜）に変換。Softmax/Harville変換の要否は
       6-5節参照

[ステップ7: 出力]
  7-1. course_code × track_condition_code ごとにフォルダを作成
  7-2. レースごとの予想順位CSV・開催サマリーを保存
```

### 6-5. Softmax/Harville確率変換の要否について

`predict.py` には `softmax_with_temperature` + `harville_place_probs`
（温度 T=0.76）が既に実装されているが、これは**複勝・ワイド・馬連の期待値
計算**（RaceAI_var2.0.0 由来の馬券最適化文脈）のために作られたものである。

当日予測機能の一次目的は「純粋能力ベースの着順予想」であり、
ROIやオッズは扱わない。したがって：

- **出力の主軸は生スコア降順の順位（予想着順）**とする
- 参考情報として Softmax(T=0.76) による「相対的な強さ」を0-100スケール等で
  併記するのは可（解釈補助）。ただし Harville/Stern のワイド確率変換は
  今回のスコープ外（市場オッズが絡む確率較正はvar2.0.0領域であり、
  var1.0で出す必要はない）
- evaluator が過去の的中率検証で使っている評価軸（Top-1/Top-3/NDCG/Spearman）
  とも整合させるため、CSV出力には「ランキングスコア」と「予想着順」を
  必須列とし、Softmax確率は任意列とする

---

## 4. 実装詳細設計（implementerへの指示）

### 4-A. `main/notebook_bootstrap.py`（新規）

var2.0.0 の `notebook_bootstrap.py` をベースに、var1.0 向けに以下を変更して移植する。

**移植してよいもの（市場情報と無関係な汎用インフラ）**:
- `_ensure_paths`, `run_with_32bit_python`, `_interpreter_is_64bit`
- `update_jra_data_32bit`, `get_race_data_32bit`
- `run_today_se_ra_and_realtime_32bit`, `run_today_se_ra_and_realtime`
  （`_run_realtime_we_v2_snapshot_inproc` 含む）
- `fetch_hc_only`, `fetch_wc_only` 等の `fetch_*_only` 系 import 転記

**除外・作り直すもの**:
- 末尾の `load_models` / `predict` / `recommend_bets` / `format_recommendations`
  を `main.main` から import する部分（var2.0.0固有の市場残差ロジック）は
  **削除**。var1.0版は代わりに `pure_rank/src/predict_today.py`
  （4-C節）の関数を import する
- `model_training.src.*` への参照（`create_main_features` 等）は
  var1.0に存在しないモジュールなので削除し、代わりに
  `pure_rank.src.create_features` / `pure_rank.src.preprocess` を import する

**新規に追加が必要なもの**:
- HC/WC当年再取得のラッパー関数（`refresh_today_training_data()` 等の名前で、
  `fetch_hc_only(cur_year, cur_year)` → `fetch_wc_only(cur_year, cur_year)` →
  `preprocess_hc` / `preprocess_wc` 再実行までを一括するショートカット）

### 4-B. `common/data/src/jv_subprocess.py`（新規）

var2.0.0/var3.0 の実装をそのまま移植する（32bit子プロセス起動のみの
汎用インフラで、市場情報とは無関係）。差分確認のみで良い。

### 4-C. 当日データ変換アダプタ（新規モジュール、例: `pure_rank/src/today_adapter.py`）

**目的**: `main/data/race/race_ra.csv` / `race_se.csv`（JV生データ形式）を
`SE_preprocessed.parquet` / `RA_preprocessed.parquet` と同一スキーマの
DataFrame に変換する。

**実装前提として implementer が必ず検証すべきこと**:
実際に `run_today_se_ra_and_realtime()` を1回実行し、出力される
`race_ra.csv` / `race_se.csv` の実列名をダンプして、`preprocess.py` の
`_SE_SOURCE_COLS` / `_RA_SOURCE_COLS_FROM_HD` にある列名（`wakuban`,
`horse_num`, `ketto_num`, `sex_code`, `trainer_code`, `jockey_code`,
`burden_weight`, `horse_weight`, `abnormal_code`, `grade_code`,
`track_code`, `weather_code`, `turf_condition`, `dirt_condition`,
`running_count` 等）と実際に一致するか突き合わせる。
本仕様書は静的コード解析（`legacy_get_data_impl.py` は8000行超あり、
CSV書き出し列定義箇所を仕様策定段階で全数特定できていない）に基づく
設計であり、**列名の実地確認は実装フェーズの必須タスク**とする。

処理内容:
1. `race_ra.csv` / `race_se.csv` を読み込み（`dtype=str` で一旦読み、
   `preprocess_se`/`preprocess_ra` と同じ型変換ロジックを適用）
2. `race_id` を 2-4節の規則で生成
3. `race_date` を `year + month_day` から生成（`_make_race_date` と同ロジック）
4. `surface_code`, `track_condition_code`, `surface_condition`,
   `distance_category`, `horse_count`（=`running_count`）を
   `preprocess_ra` と同じ計算式で導出
5. **`odds` / `popularity` 列がもし存在すれば明示的に `drop`**
   （`refresh_today_odds_data` が `race_se.csv` に反映するため）
6. `finish_rank`, `racetime`, `time_3f_after`, `is_win`, `is_place`,
   `abnormal_code`, `hon_shokin`, `fuka_shokin`, `running_style_code`,
   `corner_1..4` は当日未実施につき **NaN埋め**（`is_win=0`, `is_place=0`
   としてもよいが、後述の理由で値自体は計算結果に影響しない。実装は
   NaNで統一し、型エラーが出る箇所のみ0で明示的に埋める）
7. SK と `ketto_num` で left join（`sire_id`, `bms_id`）
8. 出力 DataFrame の列集合が `SE_preprocessed.parquet` ∪
   `RA_preprocessed.parquet`（`_load_data()` がマージ後に持つ列集合）と
   完全一致することをアサートする

### 4-D. 特徴量生成（`pure_rank/src/predict_today.py`、新規）

`create_features.py` の非 `main()` 関数（`_build_hist_features` 等）を
`from create_features import _build_hist_features, ...` で **直接 import**
して再利用する（コード重複禁止の原則。`_` プレフィックスの private 関数だが
同一プロジェクト内なので import して構わない。もし将来 refactorer が
public API 化するのは別タスクとする）。

```
def build_today_features(today_raw: DataFrame, cfg: dict) -> dict[int, DataFrame]:
    """
    戻り値: {track_condition_code: 132列DataFrame(当日行のみ)} の辞書（4件）
    """
    hist_se = pd.read_parquet(...)  # SE_preprocessed
    hist_ra = pd.read_parquet(...)  # RA_preprocessed
    hist_sk = pd.read_parquet(...)  # SK_preprocessed

    # 1. 履歴データを _load_data 相当のロジックでマージ・フィルタ
    hist_df = ...  # 既存 _apply_filters 適用済み

    # 2. 当日行（today_raw）は track_condition_code=1(良) 暫定値で結合
    #    （3列以外はこの値に依存しないため何でもよいが、コードの意図を
    #      明確にするため「良」をプレースホルダに固定する）
    combined = pd.concat([hist_df, today_raw_with_provisional_code], ...)
    combined = _build_course_geometry_features(combined)
    combined = _build_hist_features(combined)
    combined = _build_current_features(combined)
    combined = _build_sire_features(combined)
    combined = _build_jockey_trainer_features(combined)
    combined = _build_speed_index_features(combined)
    combined = _build_relative_features(combined)
    combined = _add_training_features(combined, hc, wc)

    today_rows_129cols = combined[combined["race_date"] == today_date] のうち
        シナリオ非依存126+3列から3列を除いたもの

    # 3. シナリオ依存3列を4パターン分軽量再計算
    result = {}
    for code in [1, 2, 3, 4]:
        scenario_3cols = _recompute_condition_dependent_cols(
            hist_df, today_raw, track_condition_code=code
        )
        result[code] = today_rows_129cols と scenario_3cols を列結合
    return result
```

`_recompute_condition_dependent_cols` は 132列の内訳のうち
`hist_same_condition_win_rate` / `hist_surface_condition_win_rate` /
`hist_best_time_same_cond` の3列だけを、対象馬ごとに `hist_df`
（過去走・フィルタ済み・real な `track_condition_code`）から
`groupby(ketto_num, track_condition_code=code)` 等で直接集計する軽量関数
として新規実装する。

**列数・列名の完全一致を保証するassert必須**（`create_features.py` の
`main()` にある132列アサートと同じ思想を、当日パイプラインの出力にも
必ず入れる）。列が1つでもズレると `get_feature_cols()` の集合が
学習時と食い違い、サイレントに精度が壊れるため。

### 4-E. カテゴリ特徴量の扱い（重要な実装上の注意）

`train.py` は `lgb.Dataset(..., categorical_feature=valid_cat, ...)` で
LightGBM 側にカテゴリ列を指定して学習している。LightGBM Booster の
`predict()` はカテゴリ列を **学習時に保存されたモデル内のカテゴリ境界
情報**に基づいて分岐するため、pandas の `category` dtype を使わず
素の整数コード列（`surface_code`, `track_condition_code`, `course_code`,
`grade_code`, `distance_category`, `sex_code`, `weather_code` は
現状すべて int 系そのまま）で学習・保存されている前提であれば、
当日データも同じ int dtype で渡す限り追加のエンコーディング作業は不要と
推測される。ただし **train.py 側の実際の dtype 変換ロジック（category化の
有無）を implementer が再確認**し、当日側で完全に同じ dtype
変換を再現すること（この点は本仕様書のスコープでは未検証で、
実装時の確認事項として明記する）。

---

## 5. 出力フォーマット・フォルダ構造

```
main/predictions/{YYYYMMDD}/{開催場所名}/{良|稍重|重|不良}/
    race_{race_id}_pred.csv     # レースごとの予想順位
    summary.csv                  # その開催場所・馬場状態でのレース一覧サマリー
```

例:
```
main/predictions/20260704/東京/良/race_20260704050101_pred.csv
main/predictions/20260704/東京/稍重/race_20260704050101_pred.csv
main/predictions/20260704/東京/重/race_20260704050101_pred.csv
main/predictions/20260704/東京/不良/race_20260704050101_pred.csv
main/predictions/20260704/東京/良/summary.csv
```

`race_{race_id}_pred.csv` の必須列:

| 列名 | 内容 |
|------|------|
| `race_id` | 16桁レースID |
| `race_num` | レース番号 |
| `horse_num` | 馬番 |
| `ketto_num` | 血統登録番号 |
| `pred_score` | アンサンブル生スコア（降順） |
| `pred_rank` | 予想着順（1始まり、pred_scoreの降順順位） |
| `pred_softmax_prob`（任意） | Softmax(T=0.76)による参考勝率 |

`summary.csv`: 開催場所・馬場状態内の全レースの `race_id`,
`race_num`, `distance`, `surface_code`, `1位予想馬番`, `1位予想スコア` 等の
一覧（目視確認用）。

出力前に **市場情報混入チェック**（`FORBIDDEN_MARKET_COLS` との突合）を
DataFrame に対して実行し、通過しなければ CSV を書き出さずエラー停止する。

---

## 6. 評価基準（この機能自体の検証）

当日予測は「本番運用機能」であり、Top-1/NDCG等のオフライン指標では
直接評価できない（結果が出るのは後日）。実装検証は以下で行う:

- **回帰テスト**: 既に着順が確定している過去の1レース（例えばテスト期間内の
  実在レース）を「あたかも当日であるかのように」当日パイプラインに通し、
  出力される `pred_score` が `evaluate.py` で計算した既存のオフライン予測
  スコアと**完全一致**することを確認する（4シナリオ中、実際の
  `track_condition_code` に対応するシナリオの出力が、既存パイプラインの
  出力と一致するはず）。これが当日パイプラインの正しさを保証する最有力の
  検証方法である
- 4シナリオ間で `hist_same_condition_win_rate` 等3列のみが異なり、
  他129列が完全に一致することを assert で確認する
- 市場情報列（odds等）がいかなる中間DataFrameにも出現しないことを
  `_check_no_market_features` 相当のチェックで確認する

**リーク停止閾値は本機能には直接適用されない**（オフライン評価ではないため）。
ただし回帰テストでの完全一致が取れない場合は、学習時パイプラインとの
計算式の相違＝実装バグの可能性が高いため、evaluatorへの相談ではなく
implementer側での原因究明を優先する。

---

## 7. implementerへの引き渡し事項（実装ステップ）

段階的に検証できるよう、以下の順に実装することを推奨する。

1. **Step 1**: `common/data/src/jv_subprocess.py` を var2.0.0 から移植（差分のみ）
2. **Step 2**: `main/notebook_bootstrap.py` を新規作成（4-A節の移植・除外・追加方針）
3. **Step 3**: 実際に `run_today_se_ra_and_realtime()` を1回実行し、
   `main/data/race/race_ra.csv` / `race_se.csv` の実列名をダンプして
   4-C節の想定列名とのズレを確認・記録する
4. **Step 4**: `pure_rank/src/today_adapter.py` を新規作成し、
   Step 3で確認した実列名に基づいて変換ロジックを実装。
   単体テストとして「変換後DataFrameの列名・dtypeが
   `SE_preprocessed.parquet`/`RA_preprocessed.parquet` と一致する」ことを検証
5. **Step 5**: `pure_rank/src/predict_today.py` に
   `build_today_features()` を実装。まず track_condition_code=1（良）
   固定の単一シナリオで動作確認（129列+3列=132列が揃うこと、
   列数アサートが通ることを確認）
6. **Step 6**: 4シナリオ対応（`_recompute_condition_dependent_cols` 実装）。
   3列以外が4シナリオ間で完全一致することをassertで検証
7. **Step 7**: 回帰テスト（6節）を実施。過去の実在レース1件で
   既存 `evaluate.py` のスコアと当日パイプラインのスコアが一致することを確認
8. **Step 8**: 推論・出力部分（`ensemble_predict` 呼び出し、
   フォルダ構造生成、CSV出力）を実装
9. **Step 9**: `main/Notebook/main.ipynb` を作成し、
   Step 1〜8の関数をセルに分けて呼び出す実行フローに組み立てる
10. **Step 10**: 市場情報混入チェックコマンドを実行し、
    全ステップで新規作成したモジュールに odds/popularity 系列が
    一切現れないことを最終確認する

### 実装時の注意（ユーザー実行が必要な工程）

Step 3・Step 7以降の当日データ取得を伴う検証は、**JV-Link接続（実機の
Windows環境・32bit Python・JRA-VANの契約回線）が必要**であり、
implementerが自動で完結できない可能性が高い。JV-Link接続を要する箇所は
ユーザーに実行を依頼し、出力（CSV・ログ）を implementer が確認する
という分担を想定する。過去レースを使った回帰テスト（Step 7）は
JV-Link接続不要（既存の `01_preprocessed` データのみで完結）なので、
先にこちらを実装・検証してから当日接続を伴う部分に進むことを推奨する。

---

## 8. 未確定事項・実装フェーズでの要確認事項一覧

- `race_ra.csv` / `race_se.csv` の実際の列名（4-C節、Step 3で確定）
- `train.py` での categorical 列の dtype 処理詳細（4-E節）
- 新規登録馬（`sire_id`/`bms_id` がSKに存在しない）・出走取消馬の扱い
  （エラーにせず NaN で通し、モデルの欠損値分岐に任せる方針を推奨するが、
  implementerが実装時に最終決定してよい）
- Softmax確率を出力CSVに含めるか否か（6-5節、推奨は「任意列として含める」）
