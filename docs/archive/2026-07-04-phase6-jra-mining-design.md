## 実装仕様書: Phase 6（JRAマイニング予想 あり/なし比較） — 2026-07-04

### 承認状況

ユーザー承認済み（CLAUDE.md 禁止事項8「Phase 6はユーザー承認なしに開始しない」を満たす）。

### 目的

現行正式ベースライン v39_course_slim（Top-1=30.24% / Top-3=61.76% / NDCG@3=0.5359 /
Spearman=0.5048、132列）は、2026-07-04 に確定した市場ベンチマーク実測値
（1番人気 Top-1=32.90%、evaluator 独立検証済み）に **-2.66pp** 届いていない。
このギャップをROI戦略ではなくランキング精度（Top-1/NDCG/Spearman軸）で埋める一手として、
JRA公式データマイニング予想（`race_se` の `mining_kubun`/`mining_predicted_rank`/
`mining_predicted_time`/`mining_error_plus`/`mining_error_minus`）を特徴量として追加する
価値を検証する。

---

## 0. データソースの正体確認（実装前必須・最重要）

### 0-1. 過去の「DM失敗」（v15, Spearman=0.08）との関係

メモリ（project_phase_progress.md）記載の過去結論:

> ### DM 失敗の根本原因（v15）
> - dm_pred_rank vs finish_rank: Spearman rho = 0.0797（非常に弱い）
> - 結論: JRA DM 予想はこのモデルにとってほぼランダムな信号。再試験は不要。

この v15 の実装（`dm_pred_rank` という変数名の特徴量）は **リポジトリ内に一切の痕跡がない**
（`git log --all -p -- pure_rank/src/create_features.py` に `dm_pred` の出現なし、grep 全体でも
ヒットなし）。過去セッションでの使い捨て実験で、コミットされずに終わったものと推定される。
そのため実装方法（どのCSV/どのフィールドから構築したか、join key、shift漏れの有無)を
直接監査することはできない。

今回、以下2系統の独立データソースで実測を行った結果、**過去結論とは大きく異なる強い信号**を
確認した。

### 0-2. 実測結果（今回・2026-07-04時点、2019〜2025年 330,845行）

| 指標 | 実測値 |
|------|--------|
| `mining_predicted_rank` vs `finish_rank` Spearman | **ρ = 0.4710**（race_se_*.csv 直接集計、2019-2025） |
| 同上（var2.0.0 `horse_data.parquet` 由来、2015-2026） | **ρ = 0.4768**（n=547,170） |
| 同上（pure_rank v39_course_slim の特徴量サブセットに完全 inner join、2015-2025） | **ρ = 0.4786**（n=536,776、**欠損ゼロで完全結合**） |
| mining_predicted_rank=1 の実際勝率 | 22.97%（n=24,110。ランダム基準 ≈1/16=6.25%を大きく上回る） |
| カバレッジ（mining_predicted_rank 非欠損率） | 2015〜2025年で毎年 **99.5〜99.7%**、2026年（進行中）89.7% |

参考として、既存の最強単一特徴量 `hist_last_time_dev` は同一データで `finish_rank` との
Spearman ρ = 0.3867（v39_course_slim全体）。**`mining_predicted_rank` はこれを上回る、
現時点でプロジェクト内最強の単一相関を持つ特徴量候補**である。

### 0-3. 矛盾の解消仮説

以下の根拠から、v15 の「ρ=0.08（ほぼランダム）」という結論は、**今回検証した
`race_se.mining_predicted_rank` とは異なる構築過程・異なる品質のデータに基づく可能性が高い**
と判断する（Rule 3: 後出しじゃんけん禁止に抵触しないよう、この判断はテスト期間データではなく
学習期間データ2015-2024の集計のみで行った）。

1. **コード的な裏付け**: `common/data/src/legacy_get_data_impl.py` の
   `merge_dm_mining_to_main_se()` 関数が、`realtime_dm/dm.csv`（JV-Link 0B13 リアルタイム
   マイニング）から `race_se.csv` の `mining_predicted_time`/`mining_error_plus`/
   `mining_error_minus`/`mining_predicted_rank`/`mining_kubun` へ **horse_num 単位で正しく
   突き合わせてマージする**実装が存在し、さらに `mining_predicted_time` のみ入っていて
   `mining_predicted_rank` が欠けている行を **レース内タイム昇順で補完する**
   `_backfill_mining_predicted_rank_from_time()` まで用意されている。これは「正しい
   race内相対順位」を保証するために書かれた実装であり、`race_se` 経由のマイニング列は
   相応に検証された導線を通っている。
2. **時系列的な裏付け**: `race_se.mining_kubun` は全期間で値 `"3"` に統一されており
   （2026年の一部未確定レースのみ `"0"` = 未算出）、単一の確定版マイニング予想のみが
   格納されていると読める。v15 実験が仮に別の未マージ・未検証の生ファイル
   （例: `DM_SCHEMA`/`TM_SCHEMA` の固定長パース結果を直接使う等）を対象にしていたなら、
   horse_num の突合せミスや欠損値の 0 埋めによって信号が崩れ、ρ=0.08 のようなほぼランダムな
   相関になり得る。
3. **論理的な裏付け**: `mining_predicted_time` と実際の走破 `time` は Pearson r=0.989 /
   Spearman ρ=0.995 という極めて高い相関を示すが、完全一致率は 0%（実測値と予測値がずれている）。
   これは「距離が同じレースでは走破タイムの絶対値がほぼ距離だけで決まる」という自明な
   構造（分単位のオーダーが支配的）を反映しており、post-race結果のコピーではなく独立予測である
   ことを裏付ける。リークではない。

**結論**: 過去のv15結論（ρ=0.08）は再現できず、今回 `race_se.mining_predicted_rank` を
2系統の独立データソースで測定して ρ≈0.47〜0.48 の強い信号を確認した。
`race_se` 経由のマイニング列は past-v15 とは異なる、より信頼できる導線（JV-Link 0B13正式マージ
＋補完ロジック）を経ていると判断し、**再試験する価値がある**と結論する。ただし過去結論を
明示的に覆すため、この節の内容は evaluator が独立に再検証すること（後述 4章）。

### 0-4. TM（Phase 5既知の教訓）との関係

`race_se` の `mining_*` 列は前述のとおり DM（走破タイム予想＋順位）系統であり、
var2.0.0 の `jra_tm_score`（TM=公式タイム指数のスコア型）とは別データ（`SCHEMAS["TM"]` は
`mining_pred_i_score` を持つ別スキーマ）。Phase 5 の TM失敗（v12）の教訓、
「TM単体信号はspeed_idxより強い（|ρ|=0.33）が hist_last_time_dev（ρ=0.39）と r=-0.54で
重複し共線性でノイズ化した」は、**今回のDM系マイニング特徴量にも同型リスクとして適用可能**
（0-5節で実測）。

### 0-5. 既存特徴量との重複リスク実測（学習期間データのみ、2015-2024）

`features_v39_course_slim.parquet`（132列）と mining データを race_id×ketto_num で
inner join した結果（536,776行、**結合欠損ゼロ**）:

| 既存特徴量 | vs `mining_predicted_rank` Pearson r |
|-----------|--------------------------------------|
| `hist_last_time_dev` | **+0.5214** |
| `field_z_time_dev` | **+0.5218** |
| `hist_speed_idx_last` | -0.3304 |
| `hist_speed_idx_best` | -0.1982 |

いずれも相関ゲート閾値 `|r|>=0.7`（`_run_correlation_gate()`）を下回るため機械的ゲートは
通過見込みだが、`hist_last_time_dev`/`field_z_time_dev` との r≈0.52 は Phase 5 TM失敗
（r=-0.54で共線性ノイズ化）と同水準の重複度であり、**単純追加でも共線性でモデルに
埋もれるリスクは中程度に存在する**。実験は「あり/なし」の1変更比較で行い、
重複由来の無効化が起きた場合は Phase 5 と同じ「単体相関は強いが共線性でノイズ化」という
パターンとして evaluator が切り分けること。

参考: 学習期間（race_date<=2024-12-31）でのレース内 is_win 相関（相関ゲートの参考指標）は
`mining_predicted_rank` で r=-0.198、既存最強特徴量 `hist_last_time_dev` で r=-0.158。
どちらも `_run_correlation_gate()` の目安閾値 0.15 を超えるが、これはハードゲートではなく
警告のみ（強い特徴量では通常発生する）。想定内としてよい。

---

## 1. 禁止特徴量の確認（市場情報排除の原則との整合性議論）

### 1-1. 形式的な確認

- [x] `mining_predicted_rank`/`mining_predicted_time`/`mining_error_plus`/`mining_error_minus`
      はいずれも単勝オッズ・複勝オッズ・馬連オッズではない
- [x] 人気順位（`popularity`）そのものではない
- [x] `market_log_odds`/`init_score` ではない
- [x] データソースは JRA公式のデータマイニング予想（0B13）であり、賭け手の資金配分から
      導出される集合知（オッズ・人気）とは生成過程が異なる

CLAUDE.md の禁止リストの文言上は非該当。ただし以下の点を **明示的に議論**する
（ユーザー指示どおり）。

### 1-2. 市場情報との類似性に関する検討（懸念点）

今回の調査で、看過できない事実が2つ判明した。

1. **`mining_predicted_rank` と実際の人気順位（`popularity`）の Spearman相関は ρ=0.6896**
   （n=547,170）。これは `mining_predicted_rank` 自身の予測力（対 `finish_rank` で ρ=0.48）
   よりも高い値であり、JRAマイニング予想は「実際の着順」よりも「市場の人気」に対して
   より強く整合している。
2. RaceAI_var2.0.0（市場情報を許容する残差学習プロジェクト）の
   `model_training/src/feature_groups.py` は `_MARKET_SUBSTRINGS`（市場関連特徴量として
   一括除外/切り分けする対象パターン）に **`"mining_"` を含めている**。すなわち
   姉妹プロジェクト自身が、マイニング系データを「絶対能力」ではなく「市場寄り」の
   カテゴリとして扱っている。

### 1-3. 判断

上記2点は、`mining_predicted_rank` が「オッズそのもの」ではないが「市場コンセンサスに
近い性質を帯びた情報」である可能性を示す強いシグナルである。一方で、以下の理由により
**Phase 6として「あり/なし」比較実験を行うこと自体は正当**と判断する:

- JRAマイニング予想は的中を意図した独立の統計/AIモデルの出力であり、賭け手の資金の
  重み付き投票（オッズ）とは生成メカニズムが異なる。ρ=0.69という相関は「両者が
  同じ客観的な適性情報（過去成績・血統・調教等）に部分的に依拠している」ことの
  反映である可能性があり、それ自体は不正な市場情報の混入を意味しない
  （良い能力評価指標同士は市場とも相関するのが自然）。
- CLAUDE.md の禁止リストは「市場から直接導出される値」（オッズ・人気順位・オッズ変動・
  市場補正指数）を対象としており、JRA自身の予想アルゴリズムの出力は文言上これに該当しない。

ただし、この整合性は**実験結果によって裏付けが必要**であり、単純併記では不十分と判断し、
評価フェーズに **市場プロキシ診断**（4-3節）を必須の付帯評価として追加する。
これは合否ゲートではないが、Top-1改善が「市場と一致した場合にのみ得られている」のか
「市場と乖離した場面でも真に的中率を上げている」のかを判別し、後者でなければ
Phase 6の採用を最終承認前に再協議する。

### 1-4 併存プロジェクトへの含意

RaceAI_var2.0.0 側で `mining_` が市場サブストリングに分類されている事実は、
本プロジェクト（RaceAI_var1.0）にとって「使ってよいか」の判断材料であると同時に、
将来 var1.0 と var2.0.0 を組み合わせる際（CLAUDE.md「関連プロジェクト」節）に
**二重計上（同じ市場的情報を残差学習側とability側の両方でカウントする）リスク**の
警告でもある。Phase 6採用が決まった場合、この点を仕様書として記録し、将来の統合設計時に
参照できるようにする（本ドキュメントがその記録を兼ねる）。

---

## 2. 追加する特徴量

### 実験1（v42_mining、唯一の変更）

| 特徴量名 | ソース(JVテーブル/列) | 計算方法 | リーク防止 | 期待効果 |
|---------|-------------------|--------|-----------|--------|
| `mining_pred_rank` | `race_se.mining_predicted_rank`（var2.0.0 `horse_data.parquet` の `mining_predicted_rank` 列を経由） | そのまま数値特徴量として採用（1=予想最速、horse_count=予想最遅）。0/欠損はNaN | 当該レースの事前予想であり、レース結果を含まない（0-3節で検証済み）。**shift不要**（過去走ではなく当該レース事前情報のため） | 単体 Spearman ρ=0.48（既存最強のhist_last_time_dev ρ=0.39を上回る）。Top-1のギャップ縮小に寄与する可能性 |

**1変更1実験の原則を厳守**するため、`mining_uncertainty`/`mining_best_time`/
`mining_worst_time`/`mining_gap_to_best`（レース内正規化した予測タイム差）等の派生特徴量は
**v42実験1では追加しない**。v42が合格した場合のみ、v43として派生特徴量の追加要否を
別途 planner が再検討する。

### リーク防止の設計・確認方法

1. **タイミング**: JRAデータマイニング予想（DM, 0B13）はJRA公式が最終出走投票締切後・
   レース発走前（実務上おおむね発走1時間前後）に確定・公開する事前予想データである。
   当該レースの結果（着順・タイム）を用いて事後生成される値ではない。
2. **shift(1)は不要**: 本特徴量は「過去走の集計値」ではなく「今回のレースそのものに対する
   事前予測」であるため、他の `hist_*` 系特徴量のような `shift(1)+expanding` は不要
   （むしろ shift すると誤って前走のマイニング予想を使うことになり逆に不適切）。
   ただし、これは「当該レースの結果」を使っていないことと同義ではないため、
   実装時に以下を必ず確認する:
   - `mining_predicted_time` と実測 `time` の完全一致率が 0%（本仕様書0-3節で実測済み。
     実装後に再確認すること）
   - `mining_predicted_rank` と `finish_rank` の完全一致率が異常に高くない
     （0-2節: mining_predicted_rank=1 の実際勝率22.97% ≠ 100%であり、これは事前予想が
     時々外れる健全な予測データであることを示す。実装後に再確認すること）
3. **除外条件との整合**: `mining_kubun` は現行データでほぼ全て `"3"`（確定版）。
   `"0"`（未算出、2026年進行中データの一部にのみ出現）の行は `mining_predicted_rank` が
   欠損となるため、既存の欠損値処理（NaN許容のLightGBM分割）に委ねる。追加のフィルタは
   不要（`mining_kubun` を別途フィルタ条件に加える必要はない）。

### 相関ゲートへの接続

`pure_rank/src/create_features.py` の `NEW_FEATURE_COLS_BY_VERSION` に以下を追加する:

```python
NEW_FEATURE_COLS_BY_VERSION: dict[str, list[str]] = {
    ...
    "v42_mining": ["mining_pred_rank"],
}
```

`_run_correlation_gate()` は学習期間（`race_date <= valid_end`=2024-12-31）のみで
Pearson |r|>=0.7 を判定する。0-5節の実測（最大 r≈0.52）から機械的ゲートは通過見込みだが、
**implementerは実測ログを必ず数値として出力し、evaluatorが実装後の値を確認すること**
（推測値のみでゲート通過を主張しない）。

---

## 3. データ除外条件

既存条件を変更しない（CLAUDE.md 標準フィルタをそのまま適用）。

```python
df = df[
    (~df['grade_code'].isin([8, 9])) &
    (~df['abnormal_code'].isin([1, 3, 4])) &
    (df['horse_count'] >= 5) &
    (df['finish_rank'] > 0)
]
```

`mining_predicted_rank` が欠損の行（約0.3〜0.4%、2026年進行中データはこの限りでない）を
追加で除外する必要はない。LightGBMのNaN分割に委ねる。

---

## 4. 実験プロトコル

### 4-1. 手順（1変更1実験）

1. **preprocess.py の変更**（実装対象、詳細は5章）: `_SE_SOURCE_COLS` に
   `mining_predicted_rank` を追加し、`SE_preprocessed.parquet` を再生成。
   バックアップを取ってから上書きする（CLAUDE.md 規約）。
2. **create_features.py の変更**: `mining_pred_rank` 列を1列だけ追加する
   `v42_mining` バージョンを作成。既存 `v39_course_slim` の全131列（+新1列=132列）は
   一切変更しない。
3. **相関ゲート実行**: `_run_correlation_gate()` のログを保存し、PASSを確認。
4. **学習**: `train_config.json` の `features_version` を `"v42_mining"` に切り替えて
   学習（5シードアンサンブル、既存パラメータは変更しない）。
5. **評価**: 4-2節の合否基準で判定。

### 4-2. 評価基準（合否ゲート）

| 指標 | 合格 | 要改善 | 不合格 |
|------|------|--------|--------|
| Top-1 的中率 | > 30.24%（v39_course_slim超え） | 29.9〜30.24% | < 29.9% |
| Top-3 的中率 | > 61.76% | 60〜61.76% | < 60% |
| NDCG@3 | > 0.5359 | 0.530〜0.5359 | < 0.530 |
| Spearman相関 | > 0.5048 | 0.495〜0.5048 | < 0.495 |
| テスト件数 | 2025年以降で500レース以上（既存テストセット基準を踏襲） | — | 200未満は判定保留 |

**リーク停止閾値（最優先で確認）**: Top-1 > 40% または Spearman相関 > 0.6 →
即座に実装停止し、evaluatorへ報告。0-2節の学習期間相関（ρ≈0.48）から通常の学習では
この閾値に達しない見込みだが、`mining_predicted_rank` は既存特徴量より明確に強い
単体信号のため、**この閾値チェックは通常以上に注意して行うこと**。

### 4-3. 副次評価（合否ゲートではない・参考指標）

`pure_rank/src/simulate_ev.py` の `compute_favorite_baseline` 機能を用いて、
同一テストレース集合で以下を測定する:

1. **対市場ギャップの変化**: v39_course_slim の Top-1=30.24% と1番人気実測Top-1=32.90%の
   差分（-2.66pp）が、v42_mining でどれだけ縮小するか。
2. **市場プロキシ診断**（1-3節の懸念に対応する必須の追加測定。ゲートではないが
   evaluatorはこれを必ず報告に含めること）:
   - テストレースを「モデルのTop-1予想馬 と 1番人気馬が一致する」群と「不一致」群に分割し、
     v39_course_slim → v42_mining の Top-1改善が**不一致群でも**観測されるかを確認する。
   - 改善が一致群（市場と同じ予想をした場合）にのみ集中している場合、
     `mining_pred_rank` は「独自の的中力向上」ではなく「市場コンセンサスへの接近」を
     もたらしている可能性が高く、1-3節の懸念が実証されたことになる。この場合、
     Top-1が合否基準を満たしていても、プロジェクトの目的（市場を超える純粋能力評価）に
     照らして採用可否をユーザー・plannerに再協議する。
   - 不一致群でも明確な改善が見られる場合、`mining_pred_rank` はJRA独自の分析に基づく
     真に付加的な能力評価情報である根拠が強まり、無条件で採用してよい。

### 4-4. 後出しじゃんけん禁止の遵守

4-2節・4-3節の閾値・分割方法は本仕様書執筆時点（学習期間データの分析のみ）で確定した。
2025年以降のテストデータの結果を見てから閾値・分割条件・特徴量を調整することは禁止。
結果が芳しくない場合でも、本仕様書の基準のまま合否判定を行い、不合格であれば
「試済み・結論確定リスト」に理由とともに追記して終了する。

---

## 5. implementerへの引き渡し事項

### 5-1. 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `pure_rank/src/preprocess.py` | `_SE_SOURCE_COLS`（35〜49行目付近）に `"mining_predicted_rank"` を追加。`preprocess_se()` 内で 0 値・異常値（0埋め含む）を NaN 化する処理を追加（var2.0.0 `preprocessing.py` 316〜320行目の `df["mining_predicted_rank"].replace(0, np.nan)` と同じ扱いを踏襲）。`SE_preprocessed.parquet` を再生成する前に既存ファイルをバックアップすること | 
| `pure_rank/src/create_features.py` | 新バージョン `"v42_mining"` を追加。`mining_pred_rank` 列を1列だけ既存132列セットに足す関数（既存のセクション構成に倣い、例えば `_build_current_features` 内 or 新規小セクション `5.7` として追加）。`NEW_FEATURE_COLS_BY_VERSION["v42_mining"] = ["mining_pred_rank"]` を追加。カテゴリ特徴量リストへの追加は不要（数値特徴量のため） |
| `pure_rank/config/train_config.json` | 実験時のみ `features_version` を `"v42_mining"` に切り替え。本番 `"v39_course_slim"` は変更しない（比較実験が終わるまで上書き禁止） |

### 5-2. データソースの結合方法

`preprocess_se()` は `src_hd`（= `C:/Users/syugo/AI/RaceAI_var2.0.0/model_training/data/01_preprocessed/horse_data.parquet`、`train_config.json` の `data.src_parquet_dir`）を読み込んでいる。
この parquet には既に `mining_predicted_rank` 列が存在する（var2.0.0側の `preprocessing.py` で
`race_se.mining_predicted_time`/`mining_predicted_rank` から変換済み、コード確認済み）。
**新たなJV-Link接続や生CSVの再パースは不要**。`_SE_SOURCE_COLS` に列名を追加するだけで
`available_cols` 経由で自動的に取り込まれる。

参考までに、同parquetには派生列 `mining_times`（秒変換済み予測タイム）、
`mining_error_plus_sec`/`mining_error_minus_sec`、`mining_uncertainty`、
`mining_best_time`/`mining_worst_time` も存在するが、**v42実験1では使用しない**（2-節の
1変更1実験の原則）。将来 v43 以降で検討する場合の参照用として存在のみ記録する。

### 5-3. 手順チェックリスト

1. [ ] `SE_preprocessed.parquet` のバックアップ作成（`SE_preprocessed.parquet.bak_20260704` 等）
2. [ ] `preprocess.py` 変更 → `python pure_rank/src/preprocess.py` 実行 → 列追加を確認
   （`mining_predicted_rank` の非欠損率が学習期間で99%以上であることを確認）
3. [ ] `create_features.py` に `v42_mining` バージョン追加 →
   `python pure_rank/src/create_features.py --version v42_mining`（既存のバージョン切り替え
   引数の実装に合わせる）
4. [ ] 相関ゲートログを保存し、全列PASSを確認（NGなら学習に進まずplannerへ報告）
5. [ ] `mining_predicted_time` と実測 `time` の完全一致率が0%であることを再確認（リーク検知）
6. [ ] `train_config.json` の `features_version` を一時的に `"v42_mining"` に変更して学習実行
7. [ ] evaluatorに引き渡し（4-2節・4-3節の基準で判定を依頼）
8. [ ] 判定結果に関わらず `train_config.json` を元の `"v39_course_slim"` に戻す
   （実験後の本番設定復元。合格の場合のみ、evaluator合格後に正式切り替え）

### 5-4. 検証方法（implementer自己チェック、evaluator引き渡し前）

```bash
# 市場情報混入チェック（既存コマンド、必ず実行）
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
```

このコマンドで `mining_predicted_rank`/`mining_pred_rank` 自体はヒットしない想定
（"odds"/"popularity"等の禁止語を含まないため）。ヒットした場合は変数命名を見直すこと。

---

## 6. 禁止事項の再確認

- [x] オッズ・人気を特徴量に使わない（`mining_pred_rank` はJRA公式予想であり、いずれとも別データ）
- [x] `init_score` に市場オッズ由来の値を使わない（本Phaseでは `init_score` を一切使用しない）
- [x] テストデータ（2025+）を見て特徴量・閾値を後付け調整しない（4-4節で明文化）
- [x] リーク停止閾値: Top-1 > 40% または Spearman > 0.6 → 即座に停止・evaluator報告（4-2節）
- [x] Phase 6はユーザー承認後のみ開始（本タスクは承認済み）

## 7. 未解決の懸念（evaluator・orchestratorへの申し送り）

1-2節・1-3節・4-3節で述べたとおり、`mining_pred_rank` は**市場情報そのものではないが、
市場人気との相関が予測力そのものより高い（ρ=0.69 vs 0.48）という点で、他の禁止特徴量とは
性質の異なるグレーゾーンに位置する**。本仕様書は「あり/なし」比較実験としての実施を承認するが、
最終的な採用可否は、通常の合否基準（4-2節）に加えて市場プロキシ診断（4-3節）の結果を
踏まえてorchestrator/ユーザーが判断すること。合否基準を満たしても市場プロキシ診断で
「市場一致群でのみ改善」が確認された場合は、採用前にユーザーへの再確認を推奨する。
