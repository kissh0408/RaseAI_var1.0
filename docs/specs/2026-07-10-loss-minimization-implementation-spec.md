# 実装仕様書: 損失最小化運用（flat top-1 単勝）＋ L4 パイプライン復旧 — 2026-07-10

**種別**: implementer 向け実装仕様書（planner 作成）
**上位指示書**: `docs/specs/2026-07-10-goal-redefinition-and-loss-minimization.md`（§3 が本書のスコープ、§4 が禁止事項）
**目標**: 黒字化ではなく **1番人気ベースラインに対するROI優位（損失最小化）**。fold2 OOS 既知実測: モデル予測1位 flat bet ROI 81.89% vs 1番人気 77.89%（+4.0pp）。

---

## 0. 前提・禁止事項の確認

- [ ] L1（pure_rank）に市場情報を一切追加しない。本仕様はL1・L2に**一切変更を加えない**。オッズはL3の除外条件・サイジング・決済計算のみに使う。
- [ ] 「黒字化」を主張・示唆する文言をコード・ログ・CSV・レポートに一切書かない。期待損失（fold2 OOS実測で元本の約18%）を必ず明記する。
- [ ] EV閾値のこじつけをしない。本仕様の新ロジックはEV閾値フィルタを**使わない**（正のEVは存在しないという確定結論と整合させるため）。
- [ ] Rule 3: 固定比率等の全パラメータはVALID期間（2024）のみで決定・凍結してから、TEST期間（2025+）のバックテストを**1回だけ**実行する。
- [ ] 既存のEVパス（`betting/src/backtest.py::simulate_bets`, `run_backtest_oos.py`）は**削除・変更しない**（過去レポートの再現性維持。新ロジックは新モジュールに分離する）。

---

## 1. L3: flat top-1 ベッティングロジック

### 1.1 選定ルール

各レースで**モデル予測1位を無条件で単勝1点**選ぶ。EV閾値フィルタは使わない。

- **スコア列**: `pure_score_z`（L1のレース内zスコア）。バックテストでは `pure_rank/src/score_utils.py::attach_pure_score_z` で fold2 OOSスコアから生成、当日運用ではL1予測（`rank_preds`）の `pred_score`/`ensemble_score` から同関数で生成する。**融合確率 `p_win` ではなくL1スコアを使う**（既知実測81.89%は「モデル予測1位」＝L1スコア1位に対するもの。α=0のため融合確率1位は市場1位に一致し、ベースラインと同一になってしまう）。
- **タイブレーク**: `pure_score_z` 同値の場合は `horse_num` 昇順で最小の馬（決定的挙動の保証。事前登録）。
- **オッズ除外**: 選定された1位馬のオッズが `min_odds`（2.0）未満または `max_odds`（50.0）超の場合、**そのレースは見送り**（2位馬への繰り下げは行わない。繰り下げは未検証の別戦略であり後出しになるため禁止）。
- **オッズ欠損**: 1位馬のオッズが取得できないレースも見送り。見送り件数はログ・レポートに記録する。
- **取消・除外馬**: 当日運用では `main/race_runtime.py::filter_scratched` 適用後のフレームで選定する（既存挙動を踏襲）。

### 1.2 サイジング: 固定比率 0.1%（R2改訂・凍結値）

> **R2改訂（2026-07-10）**: 初版の決定規則（グリッド {0.0025, 0.005, 0.01}、月次MDD≤0.15）は、implementerのVALID 2024実測（`evaluation/reports/flat_fraction_valid_2024.json`: 2,608ベット、的中率24.2%、100円flat ROI 79.1%）で**全候補不合格**となった（f=0.0025でもworst月2024-02のMDD=0.1596）。原因は初版導出の設計誤り: 月次損失を「連敗テール」としてのみ見積もり、**全レース購入×負の期待値戦略では月次損失が構造的ドリフトである**こと — 期待月次損失/初期bankroll = f × 月間ベット数 × 損失率 ≈ f × 217 × 0.209 ≈ f × 45.4、つまりf=0.0025で**期待値だけで11.4%** — を見落としていた。本改訂は**TESTデータを一切参照していない**（Rule 3の保護対象はTEST。VALID結果に基づく設計改訂は正当）。旧規則v1は変更履歴として本節末尾に残す。

**1レースあたり初期bankrollの 0.1%**（`stake_fraction = 0.001`。初期bankroll=100,000円なら1点100円＝JRA最低購入単位）。分数ケリーは使わない（ケリーは正の期待値を前提とした成長率最適化であり、期待値が負である本戦略には数学的前提が成立しない）。

**決定規則 v2（恣意的緩和を構造的に防ぐ形で再定義）**:

flat betting ではドローダウン・エクスポージャが f に**厳密に線形**であることを利用する（VALID実測でもグリッド3点のMDDが完全比例: 0.1596 / 0.399 / 0.798）。よって f は実測値の線形スケーリングで機械的に定まる:

1. `f_scale = monthly_mdd_limit / (worst_month_dd@f0 ÷ f0)` — 観測worst月がちょうど上限に達する限界値。VALID実測: `0.15 / (0.1596/0.0025) = 0.00235`。
2. **安全係数 k = 0.5** を掛ける: `f ≤ 0.5 × f_scale = 0.001175`。f_scaleをそのまま採用すると「観測worst月＝上限ぴったり」でヘッドルームゼロとなり、実質的な後付け合格になるため禁止。kは保守側（fを小さくする方向）にのみ働く固定値であり、VALID結果を見て緩めることは不可。
3. グリッドを**下方にのみ**拡張した {0.001, 0.0025, 0.005, 0.01} から条件を満たす最大値を採用 → **f = 0.001**。上方拡張は禁止。
4. 併せて `busiest_day_exposure@f ≤ 0.5 × max_daily_exposure` を確認する。

**f = 0.001 の検証（VALID実測の線形スケーリング。TESTは未参照）**:

| 項目 | f=0.001 での値 | 上限 | 判定 |
|---|---|---|---|
| worst月次損失（2024-02） | 0.1596 × 0.4 = **6.4%** | 15% | 合格（ヘッドルーム約2.3倍） |
| 期待月次損失（構造分） | 0.001 × 217 × 0.209 ≈ **4.5%** | — | 月次stop発動は稀と見込める水準 |
| 最繁忙日エクスポージャ（2024-05-05, 33R） | 0.0825 × 0.4 = **3.3%** | 25%（設計上界12.5%） | 合格 |

**改訂で採らなかった選択肢と理由（恣意的緩和の防止記録）**:

1. **f=0.0025の近接不合格（0.96pp超過）を許容する** — 閾値観測後の事後的な例外承認そのものであり却下。閾値 `monthly_mdd_limit = 0.15` は**不変**とする。
2. **月次MDDのウィンドウ定義変更（暦月→ローリング等）** — 構造的ドリフトという根本問題を解決せず、本番 `risk_limits.py` / `monthly_dd_tracker.py` の運用セマンティクス（implementerが整合させた暦月・対初期bankroll定義）との乖離を生むため却下。
3. **既存サーキットブレーカー（月次stop・連敗stop）を根拠に高いfを正当化する** — stopは実運用の防御層（defense in depth）であり、サイジングの代替ではない。むしろ**stopが頻繁に発動する f は本戦略に不適**（月の途中で停止すると「全レースで市場より損の少ない賭け方をする」という比較目的自体が果たせない）。f=0.001はworst月6.4%でstop発動が稀となる水準であり、この観点とも整合する。

**運用意義の注記**: 対市場優位（+4.0pp）は比率戦略のため f に依存しない。100円/点が小さすぎる場合は f ではなく**初期bankroll側を増やす**（例: 1,000,000円なら1,000円/点）。最低運用可能bankrollは `stake_rounding_yen / f = 100,000円`。それ未満は最低購入単位の制約で実効fが設計値を超えるため**運用非対応**とし、`flat_top1.py` でエラーにする。

**バックテストと運用のstake定義（R2で運用側も定額に統一）**:
- バックテスト（§4）: `stake = 初期bankroll × f`（定額）。ROI = Σpayout/Σstake は定額なら重み一様となり、既知実測81.89%（100円均等賭け）と直接比較可能。
- 当日運用: `stake = 運用開始時点の初期bankroll × f`（**定額**。100円単位切り捨て、最低100円）。初版の「現在bankroll比」は f=0.001 では100円単位への丸めにより0円へ退化する病理があるため定額へ変更。下方防御は既存stop（`monthly_mdd_limit`, `consecutive_loss_stop`）が担う。

<details>
<summary>旧・決定規則 v1（R2で置換。記録のため保持）</summary>

初版はグリッド {0.0025, 0.005, 0.01} に対し「VALID実測月次MDD≤0.15かつ最繁忙日≤0.25を満たす最大f（既定想定0.005）」としていた。導出は期待最大連敗長 `log(n)/log(1/q) ≈ 22連敗` に基づくテールリスク見積りのみで、構造的ドリフト項（f×月間ベット数×損失率）を欠いていたため、VALID実測で全候補不合格となった。
</details>

### 1.3 除外条件・リスク上限（既存値を維持、変更禁止）

`betting/config/betting_config.json` の既存値をそのまま使う:

| 項目 | 値 | 適用 |
|---|---|---|
| `min_odds` / `max_odds` | 2.0 / 50.0 | 選定馬のオッズ範囲外→レース見送り（バックテスト・運用共通） |
| `max_daily_exposure` | 0.25 | 運用時: 日次合計stakeがこれを超える場合、以降のレースを見送り |
| `monthly_mdd_limit` | 0.15 | 運用時: `betting/src/risk_limits.py::RiskLimits` で月次停止 |
| `consecutive_loss_stop` | 10 | 運用時: 同上 |

**バックテストでは停止規則（MDD・連敗stop）を適用しない**（測定純度のため。既知実測81.89%・市場77.89%はいずれも停止なしのflat betであり、停止を入れると比較が壊れる）。オッズ除外（min/max_odds）のみバックテストにも適用する。この方針は事前登録とする。

### 1.4 出力仕様（定型注記は必須）

以下の定型文を**定数**として新モジュールに定義し、(a) 推奨CSVの全行の `note` 列、(b) 実行ログ（print/logging）、(c) バックテストレポートJSONの `disclaimer` キー、の3箇所すべてに埋め込む:

```
本推奨は市場に対する相対的な損失最小化を目的とし、黒字化を保証するものではない（fold2 OOS実測: ROI 81.89%、元本の約18%の期待損失）
```

推奨CSV（`main/results/{YYYYMMDD}/today_recommendations.csv`）の列:

```
race_id, bet_type(=win), selection(馬番), pure_score_z, odds_used, odds_source,
stake, stake_fraction, mode(=loss_min_top1), skipped_reason(見送り時のみ別ファイル),
odds_timestamp, note(定型文)
```

- `odds_source`: `"race_se_csv"`（確定/前日オッズ）か `"realtime_o1"`（当日取得）かを必ず記録する（§3.4参照）。
- 見送りレースは `main/results/{YYYYMMDD}/skipped_races.csv`（race_id, top1_horse_num, reason ∈ {odds_below_min, odds_above_max, odds_missing, daily_exposure, risk_stop}）に出力する。

### 1.5 モジュール配置・config 変更

**新規モジュール**: `betting/src/flat_top1.py`

```
DISCLAIMER: str                         # 1.4の定型文
select_top1_bets(df, *, cfg) -> pd.DataFrame
    # race_idごとにpure_score_z最大の1行を選び、オッズ除外を適用。
    # 戻り値にskipped情報（attrs または別DataFrame返却のタプル）を含める
apply_flat_sizing(picks, *, bankroll, stake_fraction, rounding_yen=100) -> pd.DataFrame
settle_win_bets(picks) -> pd.DataFrame  # finish_rank==1 で payout=stake*odds、pnl列付与（バックテスト用）
run_loss_min_recommendations(rank_preds_df, odds_df, *, cfg, odds_timestamp, bankroll) -> pd.DataFrame
    # 当日運用エントリ。1.4の列仕様でCSV行を構築
```

**config 追加**（`betting/config/betting_config.json`。既存キーは変更しない）:

```json
{
  "mode": "loss_min_top1",
  "loss_min": {
    "selection": "model_top1",
    "score_col": "pure_score_z",
    "stake_fraction": 0.001,
    "stake_fraction_frozen": true,
    "stake_rounding_yen": 100
  }
}
```

- `mode`: `"loss_min_top1"`（新既定）| `"ev_filter"`（旧ロジック。後方互換のため残す）。
- `stake_fraction = 0.001` は決定規則v2（§1.2）による**凍結値**。§2 の再実行で規則v2の機械適用結果が0.001と一致することを確認した上で `stake_fraction_frozen: true` とする。
- `betting/src/recommend.py::run_recommendations` は変更せず、L4側（`main/unified_pipeline.py`）で `mode` により `run_loss_min_recommendations` と `run_recommendations` を分岐させる。

**単体テスト**: `betting/tests/test_flat_top1.py` を新規作成。最低限: (1) 各レース1点のみ選定される、(2) タイブレークが馬番昇順、(3) オッズ範囲外レースが見送りになる、(4) stakeが100円単位・bankroll×fractionと一致、(5) 決済計算（的中・不的中）、(6) `note` 列に定型文が入る、(7) `mode="ev_filter"` で旧パスが従来通り動く。

---

## 2. VALID（2024）でのパラメータ確認スクリプト（Rule 3）

**新規スクリプト**: `betting/src/derive_flat_fraction.py`

- 入力: `pure_rank/data/03_scores/scores_v39_course_slim_fold2_oos.parquet` + `pure_rank/data/02_features/features_v39_course_slim.parquet` + オッズ（`evaluation/odds_loader.py::attach_odds_from_se_parquet`。`betting/src/backtest.py::load_scored_odds_frame` を再利用してよい）。
- **期間: 2024-01-01〜2024-12-31 のみ**（`run_backtest_oos.py` の `VALID_START/VALID_END` と同一）。TEST期間（2025+）のデータを読み込んでも集計に**含めてはならない**。
- 処理: §1.1の選定ルール＋オッズ除外を適用したflat top-1系列を構築し、以下を算出:
  - n_bets、的中率、VALID ROI（参考値）
  - 最大連敗長
  - 候補グリッド f ∈ {0.001, 0.0025, 0.005, 0.01}（R2で下方拡張）それぞれの資金曲線・**月次最大MDD**（暦月ごとの独立P&L対初期bankroll比。`risk_limits.py` の運用セマンティクスと同一定義）・最繁忙日エクスポージャ
  - §1.2 の**決定規則v2**（f_scale線形スケーリング × 安全係数0.5 → グリッド内最大値）を機械的に適用した採用 f と、その中間値（f_scale, 0.5×f_scale）
- 出力: `evaluation/reports/flat_fraction_valid_2024.json`（採用f、決定規則の各判定値、`disclaimer` キー含む。初回実行結果は規則v1版として上書きされるため、必要ならgit履歴で参照）。
- 採用 f（規則v2の期待値: 0.001）を `betting_config.json` の `loss_min.stake_fraction` に反映し `stake_fraction_frozen: true` としてコミットする。**この時点までTESTバックテストを実行してはならない。**
- 規則v2の適用結果が0.001と**一致しない**場合は実装かデータの不整合を疑い、fを変えずplannerへ差し戻す（その場でのグリッド・係数変更は禁止）。

---

## 3. L4 パイプライン復旧

### 3.1 既知バグの修正（実運用パス上）

調査で確認済みの2件を修正する:

1. **`main/unified_pipeline.py` L284 の KeyError**: `--odds` 指定時、`load_realtime_odds()`（`main/race_id_utils.py`）の戻り値は `race_id, horse_num, odds` の3列だが、呼び出し側は `rt[["race_id", "horse_id", "horse_num", "odds"]]` を選択しており **`horse_id` が存在せず必ず落ちる**。修正方針: `_odds_from_se` 側の下流利用を確認し、`horse_id` が必須なら `load_realtime_odds` にダミー列（または `horse_num` 文字列）を補い、不要なら選択列から外す。
2. **`main/pipeline/data_pipeline.py` L68**: `from main.jv_subprocess import run_with_32bit_python` — アーカイブ済みパス。`common.data.src.jv_subprocess` へ修正（view_pipeline.py で実施済みの修正と同一）。ただし §3.3 で data_pipeline.py 自体が隔離対象になる場合は隔離が優先。

### 3.2 エンドツーエンド検証手順

```
python main/unified_pipeline.py
# 既定パス: main/data/race/race_ra.csv, race_se.csv（実データ配置済み）
```

確認項目（この順に実施し、失敗した箇所を修正して再実行）:

1. `run_unified_today()` が例外なく最後まで到達し、summary dict が返ること。
2. L1: `main/predictions/` に馬場シナリオ別予測が出力されること（`pure_rank/src/predict_today.py` 経由。`main/build_today_features.py` はアーカイブ済みだが、unified_pipeline が使う `build_today_features` は `main/notebook_bootstrap.py` L493 で `pure_rank/src/predict_today.py` から束縛される**別物**であることを確認済み。`pure_rank/models/lambdarank_fold*_seed*.txt` は存在する）。
3. L2: 融合確率（`win_prob_est_baba*`）が export に含まれること（参考情報として出力継続。推奨には使わない）。
4. L3: `mode="loss_min_top1"` 分岐で §1.4 仕様の `today_recommendations.csv` と `skipped_races.csv` が生成されること。`note` 列の定型文を目視確認。
5. `main/results/today_predictions_with_bets.csv` と場別exportが生成されること。
6. 各レースの推奨が**1点以下**であること（機械チェック: race_idごとの行数≤1）。

過去日の実データで代用してよい（当日データである必要はない）。検証に使った `race_ra.csv`/`race_se.csv` の日付を実行ログに記録する。

### 3.3 アーカイブ参照コードの隔離

**到達性の確認方法**: 実運用エントリポイントを `main/unified_pipeline.py::run_unified_today`（CLI含む）と `main/notebook_bootstrap.py` の `__all__` 公開関数（JVデータ更新系）の2つと定義し、そこからの import 連鎖（遅延importを含む。`grep -n "import"` で関数内importも追う）に含まれないモジュールを「実運用パス外」と判定する。

**判断基準**:

| 条件 | 処置 |
|---|---|
| 存在しないモジュール（`strategy.src.*`, `main.jv_subprocess` 等）を参照し、かつ実運用パス外 | `main/archive/` へ移動（削除ではなく隔離。git履歴と将来の参照可能性のため） |
| 実運用パスから到達するが壊れたimportを含む | §3.1 の通り修正して存置 |
| 実運用パス上で正常 | 存置 |

**確認済みの隔離候補**（implementerは到達性を再確認の上で処置）:
- `main/main.py` — L320等で存在しない `strategy.src.betting_framework` を遅延import。var2.0.0系の市場残差ロジック前提。
- `main/pipeline/strategy_pipeline.py` — L337/L563/L607/L768 で `strategy.src.*` を遅延import。
- `main/pipeline/data_pipeline.py`, `main/pipeline/inference_pipeline.py`, `main/pipeline/view_pipeline.py`, `main/pipeline/baba_scenario.py`, `main/pipeline/monthly_dd_tracker.py` — main.py からのみ参照されるものは main.py と共に隔離。ただし `main/pipeline/export_utils.py` は unified_pipeline（L200）から到達するため**存置必須**。
- 隔離時は `main/archive/README.md` に「なぜ隔離したか・復帰条件」を1行ずつ記録する。
- 隔離後に §3.2 のE2Eを再実行し、退行がないことを確認する。

### 3.4 当日リアルタイムオッズ（O1/O2/O3）の確認

1. `common/data/src/get_data.py` / `jv_run.py` / `jv_subprocess.py` にリアルタイムオッズ（O1系）取得関数が存在するか、`main/notebook_bootstrap.py` の公開関数から呼べるかをコードレベルで確認する。
2. JV-Link実接続テストが可能なら、O1取得→CSV→`load_realtime_odds()`→`run_unified_today(odds_csv=...)` の経路を1回通す。
3. **機能しない・接続確認ができない場合**: 修復を試みず、`docs/2026-07-09-current-system-guide.md` §7 と本仕様の実装報告に「リアルタイムオッズ取得は未確認・運用範囲外」と明記し、運用範囲を「`race_se.csv` ベースのオッズ（確定/前日水準）による推奨」に絞る。この場合 `odds_source="race_se_csv"` が常に記録され、CSVの注記でオッズ時点の限界が読み手に伝わる状態を維持する。
4. いずれの場合も、結果（機能する/しない/未確認）を成果レポートに事実として記載する。「動くはず」という推定表現は禁止。

---

## 4. TEST検証（fold2 OOS、1回だけ実行）

### 4.1 実行手順

**新規スクリプト**: `betting/src/run_backtest_oos_flat.py`（`run_backtest_oos.py` を雛形に流用）

- データ: `scores_v39_course_slim_fold2_oos.parquet` + `features_v39_course_slim.parquet` + `attach_odds_from_se_parquet`（既存 `load_scored_odds_frame` を再利用）。
- 期間: TEST = 2025-01-01以降（`prob_fusion/src/oos_protocol.py::TEST_START`）。
- **実行条件**: §1.5 の単体テスト合格、§2 の f 凍結コミット、§3.2 のE2E成功、の3つが揃ってから**1回だけ**実行する。結果を見てのパラメータ変更・再実行は禁止（バグ修正による再実行はレポートに理由を明記した場合のみ可）。
- 1回の実行で以下を**すべて**算出する（事前登録済み出力。複数回実行の口実を作らないため）:
  1. **本番設定**: flat top-1、オッズ除外あり、f凍結値 → n_bets, hit率, ROI, 総期待損失額
  2. **再現性確認用**: flat top-1、オッズ除外**なし**（100円均等）→ 既知実測 81.89% と一致するか（±0.1pp以内でなければデータ・選定ロジックのバグを疑い、evaluatorへ報告）
  3. **ベースライン**: 同一レース集合で1番人気（オッズ最小、同オッズは馬番昇順等の決定的タイブレーク。`pure_rank/src/simulate_ev.py::compute_favorite_baseline` と整合する定義）への同額flat bet → 除外あり/なし両方
  4. **ペアドブートストラップ**: レース単位リサンプリング（B=10,000、seed=42固定）で ROI差（モデル−ベースライン、除外あり同士）の95%CI

### 4.2 合否ゲート

出力: `evaluation/reports/betting_backtest_oos_flat.json`（`disclaimer` キー必須）。`evaluation/update_gate_summary.py` 経由で `gate_summary.json` に登録する。

| ゲート | 基準 | 意味 |
|---|---|---|
| `n_bets_gte_200` | 本番設定の n_bets ≥ 200 | 有意性主張の最低標本数（憲法） |
| `reproduction_ok` | 除外なしflat ROI が 81.89% ±0.1pp | 選定ロジックが既知実測を再現 |
| `roi_above_market_point` | 本番設定ROI > ベースラインROI（点推定） | **運用開始の主判定** |
| `roi_above_market_ci95` | ブートストラップ95%CIの下限 > 0 | 統計的有意判定 |

- **合格（pass）**: 上4つすべて成立。
- **条件付き合格（pass_point_only）**: `roi_above_market_ci95` のみ不成立。運用は開始可能だが、レポート・CSV注記・CLAUDE.md追記のいずれにおいても「点推定で優位、統計的有意性は未達（95%CI: [x, y]pp）」と正確に記載し、優位性を断定的に主張しない。
- **不合格**: `roi_above_market_point` 不成立 → 運用開始せず、結果をそのまま記録してplannerへ差し戻し（結果を見た後のパラメータ再調整は不可）。
- `reproduction_ok` 不成立 → 合否判定以前の問題としてimplementerがバグ調査（この場合の再実行はバグ修正としてレポート明記の上で可）。
- どの結果でも「黒字化」に関する記述は行わない（ROIが仮に100%を超えても、それはノイズであり主張しない。リーク停止閾値と同様に危険信号として扱いevaluatorへ報告する）。

### 4.3 evaluator への引き渡し

TEST実行後、evaluatorが独立に (1) 市場情報混入チェック（`grep -rn "odds|popularity|market_log_odds|init_score" pure_rank/src/` が増分ゼロ）、(2) ゲート判定の再計算、(3) `gate_summary.json` 整合、を確認する。

---

## 5. 退行確認

1. 既存テストが全て通ること: `python -m pytest betting/tests evaluation/tests prob_fusion/tests main/tests pure_rank -q`（pure_rank配下にテストがある場合。収集エラーは隔離ミスのシグナル）。
2. §3.3 の隔離後、`python -c "import main.unified_pipeline, main.notebook_bootstrap"` が成功すること。
3. 旧EVパス: `mode="ev_filter"` で `run_recommendations` が従来出力を返すこと（test_flat_top1.py の後方互換テストでカバー）。
4. `evaluation/reports/betting_backtest_oos.json`（旧EVバックテスト結果）を上書き・削除しないこと（新レポートは別ファイル名）。
5. L1成果物（`features_*.parquet`, `scores_*.parquet`, モデル）への書き込みが一切ないこと。

---

## 6. implementer タスクリスト（推奨実施順）

1. `betting/src/flat_top1.py` 新規作成（§1.5）＋ `betting/tests/test_flat_top1.py`（TDD推奨）
2. `betting_config.json` に `mode` / `loss_min` 追加（§1.5）
3. `betting/src/derive_flat_fraction.py` 作成・VALID実行（R2: 規則v2＋拡張グリッドで再実行）→ `flat_fraction_valid_2024.json` 更新 → f=0.001凍結（§2）
4. `main/unified_pipeline.py` 既知バグ2件修正（§3.1）＋ `mode` 分岐追加（L3呼び出しを `run_loss_min_recommendations` へ）
5. E2E実行・修復（§3.2）
6. アーカイブ参照コードの到達性確認・隔離（§3.3）＋ E2E再実行
7. リアルタイムオッズ経路の確認・結果記録（§3.4）
8. 退行確認（§5）
9. **最後に1回だけ** `betting/src/run_backtest_oos_flat.py` 実行（§4.1）→ ゲート判定 → evaluatorへ引き渡し（§4.3）

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-07-10 | 初版。指示書§3を実装仕様化（flat top-1 単勝・固定比率0.5%・L4復旧・1回限りOOS検証） |
| 2026-07-10 R2 | VALID 2024実測で決定規則v1が全候補不合格となったことを受け §1.2 を改訂。構造的ドリフト（f×月間ベット数×損失率）を明示した規則v2を導入し **f=0.001 を凍結**。運用stakeを定額に統一、最低運用bankroll（100,000円）を規定。閾値0.15自体は不変、TESTデータは未参照 |
| 2026-07-10 evaluator sign-off | 独立検証を実施。判定は **条件付き合格（pass_point_only, 運用開始可・黒字化主張禁止）**。詳細を以下に記録する。 |

---

## 7. evaluator 独立検証記録（2026-07-10）

### 7.1 数値の再計算（bit-for-bit再現）

- `betting/src/derive_flat_fraction.py` を独立に再実行し、`evaluation/reports/flat_fraction_valid_2024.json` と完全一致（`adopted_stake_fraction=0.001`、`f_scale=0.0023496...`、`f_capped=0.0011748...`）。決定規則v2の手計算（`0.15 / (0.1596/0.0025) = 0.002350`、`×0.5 = 0.001175`、グリッド内最大適格値 `0.001`）とも一致。
- `betting/src/run_backtest_oos_flat.py` を独立に再実行し、`evaluation/reports/betting_backtest_oos_flat.json` と完全一致（本番ROI 83.35%, 再現性確認ROI 81.887%（既知実測81.89%と±0.1pp以内）, ベースラインROI 78.35%, ブートストラップ95%CI=[-2.16, +10.18]pp）。乱数シード固定（42）による決定的処理のため完全再現した。
- `favorite_baseline_no_odds_exclusion.hit_rate = 0.3290` は CLAUDE.md に既記載の市場ベンチマーク実測（32.90%）と一致し、クロスチェックとして妥当。

### 7.2 市場情報混入チェック

- `grep -rn "odds|popularity|ninki|market_log_odds|init_score" pure_rank/src/ --include="*.py"` は本フェーズ開始前後で**増分ゼロ**（`pure_rank/src/` に本フェーズでの変更なし。変更は `betting/`, `main/`, `evaluation/` に限定）。既存マッチは全て `simulate_ev.py`（betting-layer専用のEVシミュレーション。L1特徴量ではない）・`predict.py` の market_blend（評価専用）等、既存の意図された用途であり、L1特徴量への新規混入はなし。
- `betting/src/flat_top1.py` はオッズをEV計算・除外条件・決済のみに使用しており（L3の許可範囲内）、`pure_score_z`（L1由来のzスコア）とオッズの二重使用（L2 z の禁止事項）には該当しない。

### 7.3 TDDテストの再実行

- `python -m pytest betting/tests -q` → **43 passed**（新規 `test_flat_top1.py` 11件を含む）。
- `python -m pytest evaluation/tests prob_fusion/tests main/tests -q` → **43 passed, 4 skipped**（skipはPortfolio Kelly/ワイド関連の既知アーカイブ理由によるもの、退行ではない）。
- `python -c "import main.unified_pipeline, main.notebook_bootstrap"` → 成功（§5退行確認2項目、独立確認済み）。

### 7.4 Rule 3 遵守確認

- `derive_flat_fraction.py` は `VALID_START/VALID_END = 2024-01-01/2024-12-31` に限定されており、TEST期間（2025+）を読み込まないことをコードレベルで確認した。
- `run_backtest_oos_flat.py` の実行はコミット履歴・ファイル更新時刻から**1回のみ**であることを確認した（`betting_backtest_oos_flat.json` は今回の独立再実行で完全一致した数値のみが存在し、複数バージョンの痕跡なし）。

### 7.5 手順違反の評価: E2E未実施のままTEST実行

報告された手順違反（§4.1の「単体テスト合格・f凍結・E2E成功」の3条件が揃う前にTESTバックテストを実行）について、ファイルのタイムスタンプを独立に検査した結果、**単なる順序の入れ替えではなく、実質的なゲート未達**と判定する（上記(a)ではなく(b)）。

根拠（ファイル更新時刻、2026-07-10）:

| 時刻 | イベント |
|---|---|
| 22:21 | `main/unified_pipeline.py` 修正完了 |
| 22:29 | `main/archive/README.md`（隔離記録） |
| **22:52** | **E2E実行**（`main/results/20260710/today_recommendations.csv` 生成） |
| 23:04 | `derive_flat_fraction.py` 実行（VALID f 導出） |
| 23:05 | `betting_config.json` に `stake_fraction=0.001` を凍結 |
| 23:06 | `betting/src/flat_top1.py` 修正（bankroll/最低stake検証を追加） |
| **23:07** | **TESTバックテスト実行**（`betting_backtest_oos_flat.json`） |

22:52のE2E出力を検査すると、`today_recommendations.csv` の `stake_fraction` 列は **0.005** であり、後に凍結された本番値 **0.001** ではない（`stake=500円`、`bankroll×0.005=500`と整合）。つまりこのE2E実行は:

1. Rule 3 で凍結される前の暫定 `stake_fraction=0.005` に対して行われており、
2. `flat_top1.py` の bankroll下限検証（23:06追加）が存在しない版のコードに対して行われている。

したがって「E2E成功」は事実として存在するが、**TESTバックテストが実際に使った凍結済み設定（f=0.001）・最終版コード（`apply_flat_sizing` の検証込み）に対しては一度もE2Eが実行されていない**。これは§4.1のゲート条件の字義通りの意味（3条件が全て成立した状態でのTEST実行）を満たしておらず、順序の入れ替えという軽微な話ではなく、**ゲート未達のままTESTを実行した**という事実に相当する。

一方で、統計的妥当性（Rule 3のリーク防止）への影響は限定的と判断する。理由:

- `run_backtest_oos_flat.py` は `betting_config.json`（凍結後の値）と `pure_rank/data/03_scores/scores_v39_course_slim_fold2_oos.parquet` / `features_v39_course_slim.parquet` を直接読み込むオフライン処理であり、`main/unified_pipeline.py`（L4当日パイプライン配線）とはコード経路上完全に独立している。E2Eの成否はTESTバックテストの入力・ロジック・数値に一切影響しない。
- 上記7.1の独立再計算により、TESTバックテストの数値自体（凍結後のf=0.001を使用）は再現性がありRule 3にも整合することを確認済み。

**結論**: 統計的検証結果（7.1〜7.4）は有効であり差し戻し不要。しかし手順違反は(b)「看過できない規約違反」に該当し、**是正措置が必要**:

1. **記録**: 本節に事実を記載済み（本項目で完了）。
2. **追加検証（必須・未完了）**: 凍結済み `betting_config.json`（`stake_fraction=0.001`）および現行版 `betting/src/flat_top1.py` に対して `main/unified_pipeline.py::run_unified_today()` を**再実行**し、§3.2の確認項目（1〜6）を満たすことを確認する。evaluatorは本レポート作成と並行してこの再実行を独立に開始したが、実データでの推論に時間を要するため本レポート確定時点では完了を確認できていない。**implementerまたはorchestratorが再実行を完了し、結果（`main/results/{日付}/today_recommendations.csv` の `stake_fraction` 列が0.001であること、行数チェック等）を本仕様書に追記するまで、本Phaseの「E2E検証済み」ステータスは暫定扱いとする。**

### 7.6 追加で確認した未完了項目（手順違反とは別、軽微）

- **§4.2 のゲート登録が未実施**: `evaluation/gate_summary.json` を確認したところ、`loss_min`/flat-top1 バックテストに対応するセクションが存在しない（`evaluation/update_gate_summary.py` に本フェーズの登録ロジックが未追加）。統計的判定には影響しないが、仕様書の記載通りではないため実施を推奨する。
- `docs/2026-07-09-current-system-guide.md` §7（未解決・未着手の領域）に「L4当日パイプラインの完全復旧」が未着手として残っている。本フェーズでL4復旧・隔離作業は実施済みのため、状況に応じて当該記述の更新を推奨する（必須ではない）。

### 7.7 最終verdict

| ゲート | 基準 | 結果 | 判定 |
|---|---|---|---|
| `n_bets_gte_200` | n≥200 | 3,758 | 合格 |
| `reproduction_ok` | 81.89%±0.1pp | 81.887% | 合格 |
| `roi_above_market_point` | 本番ROI(83.35%) > ベースラインROI(78.35%) | +4.99pp（点推定） | 合格 |
| `roi_above_market_ci95` | ブートストラップ95%CI下限>0 | [-2.16, +10.18]pp（ゼロを跨ぐ） | **不成立** |

**Verdict: `pass_point_only`（条件付き合格）** — 仕様書§4.2の基準通り。運用開始は可能だが、点推定でのみ優位（+3.83〜4.99pp、算出方法により微差）であり、統計的有意性（95%CI下限>0）は未達であることを、レポート・CSV注記・関連ドキュメントのいずれでも「優位性を断定的に主張しない」形で正確に記載すること。「黒字化」を示唆する記述は一切確認されなかった（`DISCLAIMER` 定数が3箇所すべてに実装されていることを確認済み）。

手順違反（E2E未実施のままTEST実行）は(b)の扱いとし、7.5に記録した追加検証（凍結済み設定でのE2E再実行・結果確認）が完了するまで、L4本番運用の開始は保留することを推奨する。TESTバックテストの数値自体は独立検証により有効と確認したため、**再実行の対象はE2Eのみ**であり、TESTバックテストの再実行は不要（するべきでもない。Rule 3の「1回のみ」原則を守る）。

### 7.8 是正措置の完了（2026-07-11）

§7.5で必須とされた追加検証を実施した。

- `python -c "from main.unified_pipeline import run_unified_today; run_unified_today(...)"` を凍結済み `betting_config.json`（`stake_fraction=0.001`, `stake_fraction_frozen=true`）・現行版 `betting/src/flat_top1.py` に対して実行し、エラーなく完走した（`main/data/race/race_ra.csv`, `race_se.csv` の実データ、36レース・476頭を使用）。
- 出力 `main/results/20260711/today_recommendations.csv` を検査し、`stake_fraction` 列が全行で **0.001**（凍結値。7.5で問題視された暫定値0.005ではない）であることを確認した。`mode=loss_min_top1`、`stake=100.0`（最低購入単位）、`note`列に「本推奨は市場に対する相対的な損失最小化を目的とし、黒字化を保証するものではない（fold2 OOS実測: ROI 81.89%、元本の約18%の期待損失）」という定型注記が全行に含まれることを確認した。
- `python -m pytest -q`（プロジェクト全体）: **265 passed, 8 skipped**、退行なし。

**§4.1の3条件（単体テスト合格・f凍結・E2E成功）が全て凍結後の最終設定に対して満たされたことを確認した。** これにより7.5の保留条件は解消され、本Phaseの「E2E検証済み」ステータスは暫定ではなく確定とする。7.6で指摘された`gate_summary.json`未登録は引き続き軽微な未完了事項として残る（統計的判定には影響しない）。
