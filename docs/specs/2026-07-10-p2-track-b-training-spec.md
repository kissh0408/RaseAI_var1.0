# 実装仕様書: P2 Track B — 調教時系列 5 候補の αゲート審査 — 2026-07-10

**作成者**: planner
**承認済み計画**: `docs/specs/2026-07-08-alpha-gate-data-expansion-plan.md` Track B（§B-1〜B-4）、
`docs/specs/2026-07-09-next-performance-improvement-proposal.md` §3 P2
**先行 Phase**: P1（`docs/specs/2026-07-09-p1-alpha-segments-spec.md`）は全セグメント α=0 で終了。
提案書 §6 のとおり「P1 の結果いかんに関わらず P2 は実施する」に従い本 Phase を開始する。
**実装先**: `pure_rank/experiments/track_b_training/`（隔離実験・本番非接触）
**実装担当**: implementer（本仕様書はコードを含まない）

---

## 1. 目的

市場は調教の「印・評価・パドック」を織り込むが、**個体内の時系列変化**（強度トレンド・
頻度変化・加速度プロファイルの変化・失速率・使用パターン切替）は集約された形でしか
流通していない可能性がある。本 Phase では HC（坂路、511万行、2015〜）・WC（ウッド、
74万行、2021-07-27〜）の生時系列から**事前登録済み 5 候補**（B-1〜B-5）を 1 本ずつ生成し、
稼働済み αゲート（`evaluation/alpha_gate.py`、γ の LRT）で市場条件付き尤度改善を審査する。

- **判定軸は Top-1 ではなく γ**（市場条件付きで尤度を足すか）。Top-1 は退行防止条件のみ。
- 成功時: γ 有意な候補は L2 の追加信号スロット（γ 項）として直接本番化できる
  （L1 再学習不要 — 元計画 B-3 規約）。ΔLL/race > 0 が EV の源泉になる。
- 全滅時: 「**JV-Link 調教データからの市場超え信号は現行粒度では不成立**」と記録して
  終了する（元計画 B-4 撤退基準）。候補の後出し追加・再定義・符号反転による延長は禁止。
- **v50 の失敗（水準の正規化 = 基準時計差）との差分**: 本 Phase は「水準」を一切使わない。
  **変化・頻度・形状**の軸のみ。「基準時計との差」「調教場平均との差」等の水準系
  正規化値そのものを候補にすることを禁止する（個体内の差分・傾き・比率は可）。

---

## 2. 禁止特徴量の確認

- [x] 入力データは HC/WC（JRA 公式計測の調教タイム）と、レースキー
  （`race_id, horse_num, ketto_num, race_date`）のみ。HC/WC は市場情報ではない
  （元計画 §6-4）。オッズ・人気・払戻・`market_log_odds`・`init_score` を候補生成
  コードで一切参照しない。
- [x] 市場確率 q（`ln_market_q`）が現れるのは `alpha_gate.py` 内部
  （L2 条件付きロジットの統合変数）のみ。候補生成側には現れない。
- [x] z の二重使用なし（`pure_score_z` は αゲート内部の α·z の一箇所のみ）。

---

## 3. 事前登録判定基準（共通。TEST を見る前に固定・変更禁止）

| 項目 | 固定内容 |
|---|---|
| 使用スコア | `scores_v39_course_slim_fold2_oos.parquet`（fold2 のみ。15 モデル平均は使用禁止） |
| 検定プロトコル | αゲート既定: fit=2023 → eval=2024（`run_alpha_gate` が自動実行）。TEST(2025+) は一次通過候補のみ各 1 回 |
| 候補定義 | §5 の B-1〜B-5 の 5 本で確定。追加・削除・計算式変更・**符号反転**の後出しは禁止 |
| 一次判定（成功） | γ の LRT **p < 0.01** かつ **ΔLL/race > 0**（`alpha_gate.py` 既定ゲート） |
| 多重性の扱い | **Bonferroni 補正は行わない**（各候補 p < 0.01 のまま）。根拠: (i) 元計画 A-3 の事前登録プロトコルどおり 5 本は固定された独立ファミリで後出し追加がない、(ii) γ は下限境界 (0,5) 上の片側検定で χ²(1) の p 値は保守的、(iii) 一次通過候補には TEST(2025+) の独立な二次確認が必ず入るため偽陽性はそこで落ちる。本判断は実装前に固定し、結果を見た後の変更を禁止する |
| 二次判定（TEST、1 回のみ） | fit 期間（2023-01-01〜2024-12-31）で H1(α,β,γ)/H0(γ=0) を再フィットし、TEST(2025+) で `test_logloss_h1 < test_logloss_h0` かつ TEST Top-1 ≤ 40%・Spearman ≤ 0.60 |
| Top-1 退行防止 | eval_top1 ≥ 0.2999（`alpha_gate.py` の `TOP1_REGRESSION_FLOOR`。向上は不要） |
| リーク停止 | Top-1 > 40% / Spearman > 0.6 / 候補と `finish_rank`・`ln_market_q`・`pure_score_z` の \|r\| ≥ 0.7（`leak_warning`）→ 即停止・evaluator 報告（合格ではなく危険信号） |
| 撤退基準 | 5 候補全てが一次不合格 → 「JV-Link 調教データからの市場超え信号は現行粒度では不成立」と `results/track_b_summary.json` に verdict `"training_data_no_market_signal_at_current_granularity"` を記録して P2 終了（元計画 B-4） |
| 1変更1候補 | 5 本は順番に独立審査する。複数候補を混合・合成した特徴量は作らない |

### γ 符号の事前登録（重要な統計的制約）

`fit_fusion_mle` の `gamma_bounds=(0.0, 5.0)` は**片側**（γ≥0）である。したがって各候補は
「**cand_score が大きいほど好走期待が高い**」向きに符号を揃えて出力しなければならない。
符号は §5 で候補ごとに事前登録し、**結果を見た後の符号反転は後出しとして禁止**する
（符号が逆なら γ=0 に張り付いて不合格になる。それはその候補の結論である）。

### NaN 方針（全候補共通・事前登録）

`alpha_gate.py` の `attach_candidate_z`（レース内 z 標準化）は pandas の mean/std が NaN を
スキップするため NaN は NaN のまま残り、`build_race_tuples` → `fusion_probs` に NaN が流れて
**尤度計算が壊れる**（NaN を落とすとレース内頭数が変わり、これも不可）。よって:

1. 候補生成側で raw 値を計算した後、**レース内の非 NaN 値の平均で NaN を埋める**。
   埋めた馬はレース平均と一致するため、αゲートのレース内 z 標準化後に **z = 0（情報なし
   = 平均扱い）**となる。統計的に「その馬について候補は何も言わない」に対応する。
2. レース内全馬 NaN の場合は定数 0.0 で埋める（`alpha_gate._race_zscore` が分散ゼロ
   レースを全馬 z=0 として処理することを確認済み）。
3. 出力 parquet の `cand_score` に **NaN が 1 行も残らないこと**をテストで保証する。
4. 埋め込み前の raw NaN 率（全体・2023・2024 別）を `data/cand_b{n}_*.meta.json` に記録する
   （NaN 率が高い候補は検出力が低い、という解釈情報として残す。判定基準には使わない）。

---

## 4. データと既存 API（実測確認済み）

### 4.1 入力ファイル

| データ | パス | 実測 |
|---|---|---|
| 坂路調教 | `C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\data\01_preprocessed\HC_preprocessed.parquet` | 5,113,087 行。列: `ketto_num, training_date, training_center, hc_3f_sec, hc_4f_sec, hc_200_sec, hc_accel_sec`。期間 2015-01-02〜2026-07-04。`hc_accel_sec` は平均 +0.16（大きいほど加速良好、v50 で cummax を「ベスト」とした符号） |
| ウッド調教 | 同 `WC_preprocessed.parquet` | 741,428 行。列: `ketto_num, training_date, training_center, course, wc_3f_sec, wc_4f_sec, wc_1f_sec`。期間 **2021-07-27**〜2026-07-04 |
| fold2 OOS スコア | `pure_rank\data\03_scores\scores_v39_course_slim_fold2_oos.parquet` | 列: `race_id, race_date, ketto_num, horse_num, horse_number, course_code, finish_rank, pure_score, pure_score_z`。158,180 行 / 11,561 レース。期間 **2023-01-05〜2026-05-24**（fit/eval/TEST を全て含む） |
| 特徴量（馬の全レース履歴） | `pure_rank\data\02_features\features_v39_course_slim.parquet` | `ketto_num`・`race_date` を含むことを確認済み。B-2 の「前走日」はここから取る（scores は 2023+ のみで前走が欠けるため） |

### 4.2 レースキーと候補 parquet の形式

- **レースキー**: scores parquet から `(race_id, horse_num, ketto_num, race_date)` のユニーク
  行を取り出す。候補はこの 11,561 レース分だけ生成すればよい（αゲートは
  `load_candidate_dataset` で scores と `race_id + horse_num` inner merge する）。
- **候補 parquet**: 列は **`race_id`（str 化前の型で可、gate 側で str 化される）,
  `horse_num`, `cand_score` の 3 列のみ**。`(race_id, horse_num)` はユニークであること。
  出力先: `pure_rank/experiments/track_b_training/data/cand_b{n}_{name}.parquet`。

| ID | 出力ファイル名 |
|----|---------------|
| B-1 | `cand_b1_intensity_trend.parquet` |
| B-2 | `cand_b2_freq_change.parquet` |
| B-3 | `cand_b3_accel_profile.parquet` |
| B-4 | `cand_b4_fade_trend.parquet` |
| B-5 | `cand_b5_wc_switch.parquet` |

### 4.3 αゲート呼び出し（各候補共通）

```python
from evaluation.alpha_gate import run_alpha_gate
report = run_alpha_gate(
    candidate_path,                      # data/cand_b{n}_{name}.parquet
    scores_path=SCORES_FOLD2_OOS,        # 4.1 の fold2 OOS スコア
    features_path=FEATURES_V39,          # 4.1 の features parquet
    candidate_name="b1_intensity_trend",
    out_dir=EXP_DIR / "results",         # 本番 evaluation/reports/ を汚さない（必須）
)
```

- `out_dir` を必ず実験内 `results/` に指定する。`evaluation/reports/` および
  `gate_summary.json` への反映は evaluator 判定後の別タスク。
- レポートには `gates`（`gamma_lrt_p_lt_0_01` / `delta_ll_per_race_positive` /
  `top1_regression_ok` / `leak_warning_clear`）と `pass` が自動で入る。
- **Spearman 停止条件の補完**: `alpha_gate.py` は Spearman を計算しないため、
  `run_gate.py`（実験側ラッパー）で eval(2024) の fusion 確率順位 vs `finish_rank` の
  レース内 Spearman 平均を追加計算し、> 0.60 なら `leak_stop: true` を結果に記録して停止する。

### 4.4 リーク防止の共通実装規約

- 全候補で **`training_date < race_date` を厳守**（当日朝の調教も含めない）。
  実装は (a) 事前計算した per-workout 累積統計に対する
  `pd.merge_asof(..., by="ketto_num", direction="backward", allow_exact_matches=False)`
  （`_build_hc_norm_features` と同型）、または (b) レースキー×調教行の期間結合後に
  `training_date < race_date` で明示フィルタ、のいずれでもよいが、
  **当日境界（training_date == race_date の行が除外されること）をテストで固定**する。
- `race_date` は scores（または features）から取得する。HC/WC 側に着順・レース情報は無い。

---

## 5. 候補定義（B-1〜B-5。各 10 行程度の事前登録。以後変更禁止）

数値パラメータ（窓幅・最小本数）は全て `config.json` に集約する（ハードコード禁止）。
以下の値が登録値である。

### B-1: 調教強度トレンド `cand_b1_intensity_trend`

1. **仮説**: 直近 30 日で坂路 4F タイムが個体内で速くなっている馬は仕上がり途上の
   情報を持ち、市場の集約（調教印）より細かい粒度でオッズに未反映である。
2. **対象データ**: HC のみ。`hc_4f_sec` が非 NaN の行。
3. **窓**: `race_date - 30日 <= training_date < race_date`（上限は厳密に未満）。
4. **計算式**: 窓内の各調教について `t = (training_date - race_date).days`（負値）を説明
   変数、`hc_4f_sec` を目的変数とする OLS 単回帰の傾き `slope`（単位: **秒/日**）。
   `cand_score = -slope`（タイム減少 = 改善 = 正。符号登録済み）。
5. **NaN 規約**: 窓内の有効本数 < 3、または `training_date` が全て同一日
   （説明変数の分散ゼロ）→ raw NaN。§3 の共通 NaN 方針で埋める。
6. **v50 との差分**: v50 は「同日×調教場 median との差」＝水準。本候補は個体内の
   時間傾きのみで、調教場間・日間の水準差は差分に落ちる（切片に吸収される）。
7. **却下条件**: γ LRT p ≥ 0.01 または ΔLL/race ≤ 0 → 不採用として記録し B-2 へ。

### B-2: 調教頻度の個体内変化 `cand_b2_freq_change`

1. **仮説**: 同一馬の「前走からの調教本数」が自身の通常値より多い（厩舎の勝負気配・
   順調さ）ことは新聞の調教欄に本数としては載るが、個体基準化された形では市場に
   流通していない。
2. **対象データ**: HC のみ（WC は 2021+ で期間非対称のため除外。WC 使用は B-5 が担当）。
   本数 = HC 行数（タイム列の NaN 有無は問わない。1 行 = 1 本）。
3. **前走日**: features parquet から馬ごと（`ketto_num`）に `race_date` を昇順ソートし
   `shift(1)` で前走日 `prev_race_date` を得る（features は 2015+ の全履歴を持つ）。
4. **計算式**: `n_interval = #{HC行: prev_race_date <= training_date < race_date}`。
   個体基準 `baseline = 過去レースの n_interval の中央値`（当該レースを除く
   shift(1)+expanding median）。`cand_score = n_interval / baseline`（多い = 正方向。
   符号登録済み: 比が大きいほど好走期待が高い、と登録する）。
5. **NaN 規約**: `prev_race_date` が NaN（初出走）、過去レースの n_interval 標本数 < 3、
   または `baseline == 0` → raw NaN。
6. **既知の交絡（登録時に許容）**: レース間隔が長いほど本数は増える。間隔正規化は
   行わない（元計画 B-2 の定義どおり raw 本数比で固定。間隔情報は L1 既存特徴量
   `days_since_last_race` 系が既に持つ）。
7. **却下条件**: B-1 と同一（γ LRT p ≥ 0.01 または ΔLL/race ≤ 0）。

### B-3: 加速度プロファイル変化 `cand_b3_accel_profile`

1. **仮説**: 坂路のラスト加速（`hc_accel_sec`、大きいほど良好）が自身のキャリア平均
   より直近で上振れしている馬は状態上昇局面にあり、水準（ベスト値 = v50 で審査済み）
   と違って市場に集約されていない。
2. **対象データ**: HC のみ。`hc_accel_sec` が非 NaN の行。`training_date < race_date`。
3. **計算式**: `recent3 = 直近 3 本（training_date 降順の先頭 3 行）の hc_accel_sec 平均`、
   `career = race_date より前の全有効本の hc_accel_sec 平均`（expanding、recent3 を含む）。
   `cand_score = recent3 - career`（直近が自己基準を上回る = 正。符号登録済み）。
4. **NaN 規約**: 有効本数（career）< 6、または直近 3 本が確保できない → raw NaN
   （career ≥ 6 により recent3 以外に ≥3 本のベースラインが必ず存在する）。
5. **v50 との差分**: v50 の `trn_hc_accel_best` は cummax の**水準**。本候補は
   個体内の**直近 vs キャリアの差分（形状変化）**であり水準を含まない。
6. **却下条件**: B-1 と同一。

### B-4: ラスト1F失速率トレンド `cand_b4_fade_trend`

1. **仮説**: 坂路のラスト 200m が序盤ペース比で失速しなくなってきている馬
   （フィニッシュの持続力改善）は、タイム水準に現れない仕上がり情報を持つ。
2. **対象データ**: HC のみ。`hc_200_sec` と `hc_3f_sec` がともに非 NaN かつ
   `hc_3f_sec > 0` の行。
3. **失速率（本ごと）**: `fade = hc_200_sec / (hc_3f_sec / 3)`（1 超 = ラストが平均
   ラップより遅い = 失速。元計画 B-4 の定義式）。
4. **窓と計算式**: B-1 と同一の 30 日窓（`race_date - 30日 <= training_date < race_date`）。
   窓内の `fade` を `t = (training_date - race_date).days` に OLS 回帰した傾き
   `slope_fade`（単位: 1/日）。`cand_score = -slope_fade`（失速率が低下傾向 = 改善 = 正。
   符号登録済み）。
5. **NaN 規約**: 窓内の有効本数 < 3、または `training_date` 全同一 → raw NaN。
6. **v50 との差分**: 失速率という**形状指標**の**トレンド**であり、水準系は不使用。
7. **却下条件**: B-1 と同一。

### B-5: WC×HC 併用切替 `cand_b5_wc_switch`

1. **仮説**: 坂路主体の馬にウッド追いを増やす（またはその逆の）使用パターン変化は
   厩舎の仕上げ意図の変更を示し、調教印より粒度の細かい情報である。
2. **対象データ**: HC + WC。**両者とも `training_date >= WC_START` の行のみ使用**
   （`WC_START` = WC parquet の実測最小日 2021-07-27。config に記録し、実行時に
   parquet から再確認して一致をログに残す）。比率の分母分子の期間を揃えるため。
3. **計算式**: 直近 30 日窓（B-1 と同一定義）で
   `wc_share_recent = n_WC / (n_WC + n_HC)`。個体基準
   `wc_share_career = race_date より前（かつ WC_START 以後）の全期間の n_WC / (n_WC + n_HC)`。
   `cand_score = wc_share_recent - wc_share_career`（ウッド比率の増加 = 強め・実戦的
   調教への切替 = 正。符号登録済み）。
4. **NaN 規約**: 窓内本数（HC+WC）< 2、または career 本数 < 5 → raw NaN。
5. **範囲の限定（1変更1候補）**: 元計画の「調教場・コース使用パターン」のうち
   **WC/HC 比率の変化のみ**を候補とする。調教場（training_center）切替や WC の
   `course` 別パターンは別変数になるため本候補に**含めない**（後出し追加も禁止）。
6. **サブセット性の確認**: WC は 2021-07-27 以降だが、αゲートの fit=2023 / eval=2024 /
   TEST=2025+ は全て WC カバー範囲内であり、期間サブセットの問題は生じない。
   2021 年後半〜のキャリアしか見えないことによる career 比率の打ち切りは許容
   ノイズとして登録する。raw NaN 率は meta.json で報告する。
7. **却下条件**: B-1 と同一。5 本目のため、不合格なら §3 の撤退基準を発動する。

---

## 6. 実装構成（隔離実験・P1 alpha_segments 構成に準拠）

```
pure_rank/experiments/track_b_training/
├── README.md            # 目的・隔離宣言・市場情報境界・実行手順
├── config.json          # 窓幅30日・最小本数(B-1:3, B-2:3, B-3:6, B-4:3, B-5:2/5)・
│                        #   WC_START・符号規約・判定閾値（p=0.01, top1_floor=0.2999,
│                        #   leak: top1>0.40/spearman>0.60/|r|>=0.7)・入出力パス
├── training_lib.py      # 純関数のみ: OLS傾き・頻度比・recent-career差・失速率・
│                        #   WC比率差・レース内mean埋め。市場情報列に一切触れない
├── build_candidates.py  # HC/WC + レースキー → data/cand_b{n}_{name}.parquet + meta.json
│                        #   （--candidate b1..b5 で1本ずつ生成。1実行1候補）
├── run_gate.py          # run_alpha_gate(out_dir=results/) + eval Spearman 補完計算 +
│                        #   track_b_summary.json への追記
├── data/                # cand_b{n}_*.parquet / *.meta.json
├── results/             # alpha_gate_b{n}_*.json / track_b_summary.json /
│                        #   （二次判定時のみ）test_b{n}_*.json
└── tests/               # §8 の TDD テスト（合成データのみ）
```

### 隔離宣言（README に明記すること）

本実験は上記ディレクトリに完結する。以下は**書き換えない**:
`pure_rank/models/`、`pure_rank/data/02_features/*.parquet`、`pure_rank/data/03_scores/`、
`pure_rank/config/train_config.json`、`prob_fusion/data/`、`prob_fusion/src/`、
`evaluation/reports/`（`gate_summary.json` を含む）。
`gate_summary.json` への結果反映は **evaluator 判定後に別タスクとして**
`evaluation/update_gate_summary.py` 経由で行い、本実験のスクリプトは実験内
`results/` のみに出力する（`run_alpha_gate` は必ず `out_dir` 指定で呼ぶ）。

---

## 7. 手順（候補ごとの逐次サイクル。B-1 → B-2 → … → B-5 の順で 1 本ずつ）

各候補 n について:

1. `build_candidates.py --candidate b{n}`: HC/WC + scores のレースキーから
   `data/cand_b{n}_{name}.parquet`（3 列、NaN なし、(race_id, horse_num) ユニーク）と
   `data/cand_b{n}_{name}.meta.json`（raw NaN 率・行数・生成パラメータ・生成日時）を出力。
2. `run_gate.py --candidate b{n}`: `run_alpha_gate(..., out_dir=results/)` を実行し、
   eval(2024) レース内 Spearman を補完計算。結果を `results/track_b_summary.json` に
   追記（候補 ID・gates・pass・leak_stop・タイムスタンプ）。
3. `leak_warning` または Top-1 > 40% または Spearman > 0.60 → **即停止・evaluator 報告**
   （後続候補にも進まない。合格ではなく危険信号）。
4. 一次判定（p < 0.01 かつ ΔLL/race > 0 かつ top1_floor かつ leak clear）の結果を記録し、
   合否にかかわらず次の候補へ（**TEST には触れない**）。
5. 5 本完了後、一次通過候補が存在すれば evaluator の検証を経て二次判定へ。

### 二次判定（TEST。一次通過候補のみ、各 1 回。事前登録済み手順）

一次通過候補が 0 の場合、**実行しない**（TEST 完全非接触のまま撤退基準を発動）。
実行する場合は候補ごとに以下を固定手順とする:

1. `load_candidate_dataset` と同一経路でデータセットを構築し、
   `prob_fusion/src/oos_protocol.py::split_oos_periods` で
   fit(2023-01-01〜2024-12-31) / TEST(2025-01-01〜) に分割。
2. fit 期間で H1: `fit_fusion_mle(fit_races_with_x)`（α, β, γ）と
   H0: `gamma_fixed_zero=True`（α, β のみ）を再フィット。
3. TEST で `test_ll_h1 = mean_logloss(test_df, α, β, x_col="cand_score_z", gamma=γ)` と
   `test_ll_h0 = mean_logloss(test_df, α_h0, β_h0)` を各 1 回だけ計算。
4. 合格 = `test_ll_h1 < test_ll_h0` かつ TEST Top-1 ≤ 0.40 かつ TEST Spearman ≤ 0.60。
5. `results/test_b{n}_{name}.json` に出力。最終判定は evaluator が行う。

### 撤退時の終了処理

5 候補すべて一次不合格の場合、`results/track_b_summary.json` の `verdict` に
`"training_data_no_market_signal_at_current_granularity"` を記録して P2 終了。
gate_summary への反映は evaluator 判定後の別タスク。次アクションは提案書 §4 の P4。

---

## 8. TDD テスト項目（テストファースト。合成データのみで実データ不要で走ること)

`pure_rank/experiments/track_b_training/tests/` に実装:

1. **B-1 傾き**: 合成調教列（既知の傾き -0.1 秒/日）で `cand = +0.1` を回復。
   有効 2 本 → NaN。全本同一日 → NaN。傾きゼロ → 0.0。
2. **B-2 頻度比**: 合成の前走間隔と本数で `n_interval / median` を検証。
   過去 n_interval 標本 2 件 → NaN。baseline=0 → NaN。初出走（prev NaN）→ NaN。
   baseline の expanding median が**当該レースの n_interval を含まない**（shift(1)）こと。
3. **B-3 プロファイル差**: recent3 − career の値検証。career 5 本 → NaN、6 本 → 有効。
4. **B-4 失速率**: `fade = hc_200 / (hc_3f/3)` の式検証（hc_3f=36, hc_200=13 → fade=13/12）。
   `hc_3f_sec` が 0 または NaN の行が計算から除外されること。傾き符号（改善 → 正）。
5. **B-5 WC 比率差**: 合成 HC/WC 列で share 差を検証。窓内 1 本 → NaN、career 4 本 → NaN。
   `WC_START` より前の行が分母分子とも除外されること。
6. **当日除外境界（全候補共通・最重要）**: `training_date == race_date` の調教行が
   使われないこと（前日の行は使われること）。merge_asof 実装なら
   `allow_exact_matches=False` の効果、フィルタ実装なら `<` 境界をテストで固定。
7. **NaN 埋め規約**: レース内一部 NaN → 埋め値がレース内非 NaN 平均と一致し、
   `alpha_gate.attach_candidate_z` を通すと当該馬の z が 0 になること。
   全馬 NaN レース → 全馬 0.0 埋め・z 全馬 0。出力 parquet に NaN が残らないこと。
8. **出力形式**: 列が正確に `{race_id, horse_num, cand_score}` の 3 列であること。
   `(race_id, horse_num)` の重複が無いこと。
9. **市場列ガード**: `training_lib.py` / `build_candidates.py` のソース文字列に
   `odds / popularity / ninki / market_log_odds / init_score / market_q / ln_market` が
   現れないことをテストで自動検証（`pure_rank/src/common.py` の
   `FORBIDDEN_MARKET_COLS` / `SUSPICIOUS_MARKET_NAME_PATTERN` が使えるなら併用）。
10. **符号規約**: 「改善している合成馬」の cand_score が全候補で正になること
    （B-1: タイム短縮、B-3: 加速度上振れ、B-4: 失速率低下、B-5: WC 比率増）。
11. **再現性**: 合成データテストは seed 固定で決定的に通ること。

---

## 9. 市場情報混入チェック手順

1. **候補生成層（最も厳格）** — 純関数層と生成スクリプトは市場列に触れないこと:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score\|market_q\|ln_market" \
  pure_rank/experiments/track_b_training/training_lib.py \
  pure_rank/experiments/track_b_training/build_candidates.py
# → 0 件であること
```

2. **実験全体** — 市場情報は `run_gate.py` の `run_alpha_gate` import 経由
   （αゲート内部の L2 統合変数）のみ:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
  pure_rank/experiments/track_b_training/ --include="*.py"
# → ヒットは run_gate.py の alpha_gate import・コメントのみであること。
#   候補の計算式に market 系列が現れたら即修正
```

3. **本番非接触の確認**:

```bash
git status --short
# → 変更が pure_rank/experiments/track_b_training/ と docs/specs/ に限られること。
#   特に evaluation/reports/ 配下に新規ファイルが無いこと（out_dir 指定漏れの検出）
```

---

## 10. implementerへの引き渡し事項（順序付きタスクリスト）

セッションが途中で打ち切られても、完了した候補の結果が `results/` に残る構成
（1 候補 = 生成 → ゲート → 記録 の完結サイクル）で進めること。

1. `pure_rank/experiments/track_b_training/` を作成し、README（隔離宣言・市場情報境界・
   実行手順）と `config.json`（§5 の窓幅・最小本数・WC_START・符号規約・§3 の閾値を
   全て集約。ハードコード禁止）を書く。
2. **テストを先に書く**（§8 の 1〜11。
   `python -m pytest pure_rank/experiments/track_b_training/tests/ -v` で RED を確認）。
3. `training_lib.py`（純関数）を実装しテストを通す（GREEN）。
4. `build_candidates.py` と `run_gate.py` を実装する（`run_alpha_gate` は必ず
   `out_dir=results/` で呼ぶ。eval Spearman の補完計算を含む）。
5. **B-1**: 生成（meta.json の raw NaN 率を確認・記録）→ ゲート実行 → 結果を
   `track_b_summary.json` に記録。リーク停止条件該当なら即停止・evaluator 報告。
6. **B-2**: 同上（前走日は features parquet から。scores からではない点に注意）。
7. **B-3**: 同上。
8. **B-4**: 同上。
9. **B-5**: 同上（WC_START の実測確認ログを残す）。
10. §9 の市場情報混入チェック（grep 3 種）を実行しログを残す。
11. 5 本の一次判定結果を evaluator へ引き渡す。一次通過候補が存在し evaluator が
    承認した場合のみ、二次判定スクリプト（§7 の TEST 手順）を実装・実行（各候補 1 回）。
12. 全結果を evaluator の最終判定に回す。gate_summary への反映は evaluator 判定後の
    別タスク（本実験スクリプトからは書き込まない）。

---

## 11. 評価基準（evaluator 向けサマリ）

- 一次: γ LRT p < 0.01（fit=2023）かつ ΔLL/race > 0（eval=2024）。Bonferroni なし
  （§3 で事前固定済み）。Top-1 ≥ 0.2999 は退行防止条件。
- 二次（TEST 1 回のみ・一次通過候補のみ）: TEST logloss(H1) < TEST logloss(H0)、
  かつ Top-1 ≤ 40%・Spearman ≤ 0.60。
- リーク停止: どの段階でも Top-1 > 40% / Spearman > 0.6 / \|r\| ≥ 0.7（対 finish_rank・
  ln_market_q・pure_score_z）→ 即停止・evaluator 報告。
- 本 Phase の評価軸は γ / logloss であり、Top-1 向上や ROI では判定しない。
- **Top-1 が上がっても γ 非有意なら不採用。Top-1 不変でも γ 有意なら採用**（元計画 B-3）。
- 5 本全滅時の verdict: `"training_data_no_market_signal_at_current_granularity"`
  （= JV-Link 調教データからの市場超え信号は現行粒度では不成立）。

---

## 12. 結果追記欄（実行後に orchestrator/planner が追記。追記後は変更禁止）

（2026-07-10 追記。出典: `pure_rank/experiments/track_b_training/results/track_b_summary.json`、
evaluator 独立検証 PASS 2026-07-10）

| ID | γ | LRT p | ΔLL/race | eval Top-1 | leak | 一次判定 |
|----|---|-------|----------|-----------|------|---------|
| B-1 intensity_trend | 0.0090 | 0.6649 | -7.8e-05 | 34.76% | なし | **不合格** |
| B-2 freq_change | 0.0413 | 0.0639 | -6.65e-04 | 34.58% | なし | **不合格** |
| B-3 accel_profile | 0.0418 | 0.0342 | -4.98e-04 | 34.46% | なし | **不合格** |
| B-4 fade_trend | 0.0000 | 1.0000 | +7.7e-10 (≈0) | 34.85% | なし | **不合格** |
| B-5 wc_switch | 0.0000 | ≈1.0 | +2.0e-10 (≈0) | 34.85% | なし | **不合格** |

**5 ファミリ全滅（事前登録の一次ゲート p<0.01 かつ ΔLL/race>0 を満たす候補なし）**。
B-2/B-3 は p<0.05 だが事前登録閾値未達かつ ΔLL/race が負であり不採用（後出しの閾値緩和は禁止）。
リーク停止条件（Top-1>40% / Spearman>0.6 / |r|≥0.7）への該当なし（eval Spearman は 0.5537〜0.5540）。
一次通過候補が存在しないため二次判定（TEST 2025+）は不実行 = **TEST 完全非接触のまま終了**。
verdict = `"training_data_no_market_signal_at_current_granularity"`
（元計画 B-4 撤退基準どおり「JV-Link 調教データからの市場超え信号は現行粒度では不成立」を正式記録。
`evaluation/reports/gate_summary.json` の track_b_training 節に反映済み）。
事前登録どおり候補ファミリの後出し追加は行わず **P2（Track B）を終了**する。
次アクションは提案書（`2026-07-09-next-performance-improvement-proposal.md` §4）の優先順位に従い P4（券種間整合性の系統的乖離測定）。

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-07-10 | 初版（P2 Track B 仕様確定。B-1〜B-5 の計算式・符号・NaN 規約・判定基準を事前登録） |
| 2026-07-10 | §12 最終結果を追記（5 候補全滅・verdict=training_data_no_market_signal_at_current_granularity・evaluator PASS・TEST 非接触）。**P2 終了** |
