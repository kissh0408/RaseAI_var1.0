# RaceAI 再構築マスタープラン — Benter型確率統合アーキテクチャ

**作成日**: 2026-07-08
**種別**: 実装指示書（本書に基づき実装する。本書自体は実装を含まない）
**前提決定**（ユーザー承認済み）:
1. アーキテクチャは **Benter型統合**（純能力モデル × 市場オッズの条件付きロジット統合）
2. 旧 Layer 2（`model_training/` + `strategy/` の binary 残差・R-6 系）は **廃止して再構築**
3. モデルバックアップ等の遺物は **リポ外アーカイブへ退避**

---

## 0. なぜ作り直すか — 理想設計を先に置く

### 0.1 世界で実証された唯一の設計

長期・大規模に市場控除率（JRA 単勝20% / 複勝20% / ワイド22.5% / 馬連22.5%）を超えて
黒字化した競馬AIの公開実例は、香港の Bill Benter モデル系統（Benter 1994,
"Computer Based Horse Race Handicapping and Wagering Systems"）に集約される。
その核心は次の一行に尽きる：

```
最終勝率 p_i = softmax( α · s_i + β · ln q_i )
  s_i : ファンダメンタル（純能力）モデルのスコア（市場情報を含まない）
  q_i : 市場オッズから導いた市場確率（控除率補正済み）
  α,β : 学習データ上の最尤推定で決める結合係数
```

- 市場は「倒す敵」ではなく「最強の事前分布」。モデルの仕事は市場確率に
  **直交する残り情報を上乗せ**すること。
- α > 0 が統計的に有意である限り、統合確率は市場単体より必ず良い
  （logloss で市場ベースラインを下回る）。これが賭けエッジの源泉になる。

### 0.2 現行システムの失敗はすべて「統合層の不在」に帰着する

`docs/2026-07-05-current-problems-detailed.md`（P-01〜P-49）の要約：

| 失敗 | 根本原因 |
|------|---------|
| ワイド/馬連 ROI 80〜90% 頭打ち（P-11〜P-13） | Harville/Stern を **無校正の LambdaRank スコア**から作っている。市場情報ゼロの確率でEVを切っても控除率に勝てない |
| Harville 確率の系統的過大推定（P-16） | 同上。ランキングスコアは確率ではない |
| Platt/Isotonic/ROI-T 全滅（P-19） | スコア→確率の事後校正では市場の情報を取り込めない |
| 条件EV探索が TEST で全滅（P-23） | エッジ不在のまま条件を掘る＝後出しじゃんけん |
| R-6 統合が TEST 悪化（P-32〜P-40） | 残差回帰＋z二重使用＋bet_tuning という**未実証の複雑さ**の積み上げ |
| 市場ブレンド β=0.30 の馬連が ROI 104.8%（P-18） | **唯一の正の信号**。ただし n=66。→ これを本体に据えるのが本計画 |

結論：L1（純能力 LambdaRank）は資産（Top-1 30.24%、市場と独立）。
捨てるのは L2 以降の「市場を使わない確率化」と「残差回帰」の全系統。

### 0.3 正直な期待値（合意事項として明記）

- **的中率軸**: 統合確率の Top-1 は構造上ほぼ確実に 1番人気（32.90%）以上になる
  （β だけでも市場を再現でき、α>0 なら上回る）。目標 **Top-1 ≥ 33.5%**。
- **回収率軸**: ROI > 100% は保証されない。市場に対する logloss 改善幅が小さければ、
  EVフィルタ後のベット数が確保できない。だから各 Phase に **合格ゲートと撤退基準**を置き、
  ゲートを通らない限り次へ進まない。数字を盛らない。これが本計画の憲法である。

---

## 1. 理想アーキテクチャ（To-Be）

### 1.1 4層構造

```
┌──────────────────────────────────────────────────────────┐
│ L4 運用層  main/            当日: データ取得→L1→L2→L3→推奨CSV │
├──────────────────────────────────────────────────────────┤
│ L3 ベッティング層  betting/   EV計算・分数Kelly・資金/リスク管理  │
│    入力: L2確率 + 直前オッズ   出力: 買い目リスト（券種/金額）      │
├──────────────────────────────────────────────────────────┤
│ L2 確率統合層  prob_fusion/   条件付きロジット p∝exp(αz+βlnq)   │
│    入力: L1スコアz + 市場確率q  出力: キャリブ済み勝率/複勝率        │
├──────────────────────────────────────────────────────────┤
│ L1 能力層  pure_rank/        LambdaRank（市場情報 完全排除）     │
│    入力: JV-Link特徴量        出力: レース内標準化スコア z          │
├──────────────────────────────────────────────────────────┤
│ L0 データ層  common/data/     JV-Link 取得・パース・格納          │
└──────────────────────────────────────────────────────────┘
```

**市場情報の境界線（改定憲法）**
- L0/L1: 市場情報（オッズ・人気）**完全禁止**（現行憲法そのまま。grepゲート継続）
- L2: 市場確率 q を **明示的な統合変数としてのみ**使用（LightGBM等の特徴量には入れない）
- L3: オッズは EV 計算・ベットサイズにのみ使用
- 「z の二重使用」（P-35）の再発禁止: L1スコアが L2 に入る経路は α·z の一箇所のみ

### 1.2 各層のインターフェイス（実装の契約）

| 層 | 成果物ファイル | スキーマ（最低限） |
|----|--------------|------------------|
| L1 | `pure_rank/data/03_scores/scores_{version}.parquet` | race_id, horse_number, pure_score, pure_score_z（レース内標準化） |
| L2 | `prob_fusion/data/probs_{version}.parquet` | race_id, horse_number, p_win, p_place, alpha, beta, model_version |
| L3 | `betting/data/bets_{version}.parquet` / `main/results/today_recommendations.csv` | race_id, bet_type, selection, ev, kelly_fraction, stake, odds_used, odds_timestamp |

- 各層は**下位層の成果物ファイルだけ**を入力にする（コードの直接import不可、疎結合）。
- 全成果物に `manifest.json`（生成日時・入力ハッシュ・config スナップショット）を付ける。

### 1.3 L2 確率統合層の設計

**モデル**: 条件付きロジット（レースを1グループとする多項ロジット）

```
p_i = exp(α·z_i + β·ln q_i) / Σ_j exp(α·z_j + β·ln q_j)
```

- `z_i`: L1 の 5シード×3fold アンサンブル平均スコアをレース内で z 標準化
- `q_i`: 単勝オッズ→ `q_raw_i = 1/odds_i` → 控除率補正。補正方式は
  (a) 比例正規化 `q_i = q_raw_i / Σq_raw` を第1実装、
  (b) power法（本命-大穴バイアス補正）を第2実装として VALID で比較
- 推定: scipy/statsmodels で負の対数尤度最小化（勝ち馬の logloss）。
  実装は 50 行程度で済む。**LightGBM を使わない**（過学習経路を作らない）
- 複勝率 `p_place`: 統合勝率から Stern モデル（λ は VALID で推定）で導出。
  ただし Phase 3 の複勝EVで実測校正カーブを必須確認
- 拡張スロット（Phase 5 以降）: `α·z + β·ln q + γ·x` の x に少数の追加信号
  （例: 斤量変化・休養明け）を**1本ずつ**追加し尤度比検定で採否

**合格ゲート（L2）**
| 指標 | 基準 |
|------|------|
| TEST logloss(統合) | < TEST logloss(市場のみ β推定) — **必達。落ちたら α=0 と同義で計画停止** |
| α の推定値 | > 0 かつ尤度比検定 p < 0.01（fold毎） |
| 統合 Top-1 | ≥ 33.0%（1番人気 32.90% 以上） |
| キャリブレーション | 10-bin で |predicted − actual| 最大 3pp 以内（単勝） |

### 1.4 L3 ベッティング層の設計

- **券種の順序**: 単勝・複勝 → （ゲート通過後）ワイド・馬連。
  ペア券種は確率導出（Stern）の校正が難しく、現行の失敗もそこで起きた。
  勝率が直接使える単勝・複勝で先にエッジの存在を証明する。
- **EV**: `EV_i = p_i × odds_i`。閾値は VALID のみで決定（Rule 3 継続）。
  実運用オッズずれ対策として **EV ヘアカット**（例: EV×0.95 で判定）を config 化
- **サイズ**: 分数 Kelly（初期 1/8〜1/4、config）。レース内・日次の最大エクスポージャ制約。
  同時複数ベットは共分散を無視しない（同一レース内は排反近似で Kelly を減衰）
- **リスク管理**: 月次 MDD 限度、連敗停止ルールを backtest 段階から組み込む

**合格ゲート（L3、Phase 3）**
| 指標 | 基準 |
|------|------|
| walk-forward ROI | 全 fold で > 100%（VALID で閾値決定 → TEST 検証） |
| ベット数 | 各 fold n ≥ 200（憲法継続） |
| Sharpe（ベット単位） | > 0 |
| 撤退基準 | 上記を満たす (閾値, 券種) が存在しない → **Phase 3 不合格として停止し、L2 の logloss 改善（γ拡張・L1改善）に戻る。条件を後掘りしない** |

### 1.5 評価プロトコル（全層共通）

- **共通テスト集合**: TRAIN ≤2023-12-31 / VALID 2024 / TEST 2025-01-01〜（4,775レース）を
  全層で共有。`evaluation/` に唯一の分割定義モジュールを置き、各層はそれを import
- **市場ベースラインの常設**: 1番人気 Top-1=32.90%・単勝ROI=77.94%・市場logloss を
  ベースラインファイルとして固定保存し、全評価レポートに併記
- **リーク検知**: Top-1 > 40% / Spearman > 0.6 即停止（継続）。
  L2 では「TEST logloss が市場比で異常に良い（>5%改善）」も停止シグナルに追加
- **オッズの時点管理**: バックテストは確定オッズを使用している事実を全レポートに明記。
  当日運用では発走 N 分前オッズを取得時刻付きで記録し、確定オッズとの乖離を月次で監視

---

## 2. 現状（As-Is）とのギャップ / 資産の仕分け

| 現状資産 | 判定 | 行き先 |
|---------|------|--------|
| `pure_rank/`（v39_course_slim、Top-1 30.24%） | **維持** | L1 としてそのまま。改善トラック継続 |
| `pure_rank/src/simulate_ev.py` の Harville/Stern・favorite baseline | 部分移植 | Stern 係数推定と市場ベースライン計算は `evaluation/`・`prob_fusion/` へ |
| `model_training/`（binary残差・v2系特徴量40ファイル超） | **廃止** | アーカイブ。復活させない |
| `strategy/src/` | 選別移植 | `kelly_sizer.py`, `ev_calculator.py`, `race_filters.py` はレビューの上 `betting/` へ移植。`backtest.py`, `bet_tuning.py`, `binary_recommendation.py`, `market_bias_corrector.py` 等の残差系はアーカイブ |
| `main/`（unified_pipeline, Notebook） | 改修 | L4 として一本化（binary系統の呼び出しを除去） |
| `common/data/`（JV-Link） | 維持 | L0 |
| リポ直下 `src/`（ped/race/time/training.py） | 廃止 | 外部参照ゼロ（grep確認済）。アーカイブ |
| `o1_raw_debug.txt`, `backup_before_unified_integration_20260705/` | 廃止 | アーカイブ |
| `pure_rank/models_backup_*` 24ディレクトリ（約400MB） | 退避 | リポ外アーカイブ。`pure_rank/models`（現行本番15モデル）のみ残す |

---

## 3. Phase 0 — リポジトリ整理（実装前の必須作業）

> 実施エージェント: refactorer。**精度に影響する変更を含めないこと。**
> 完了ゲート: `python pure_rank/src/evaluate.py` で Top-1=30.24% が再現し、
> `main/unified_pipeline.py` がドライランで通ること。

### 3.1 アーカイブ退避（リポ外 `C:/Users/syugo/AI/_archive/RaceAI_var1.0/`）

1. `pure_rank/models_backup_*`, `pure_rank/models_v11_backup`,
   `pure_rank/models_v30_backup_*`, `pure_rank/models_weighted_*` → `_archive/models/`
2. `backup_before_unified_integration_20260705/` → `_archive/`
3. `o1_raw_debug.txt` → `_archive/`
4. リポ直下 `src/` → `_archive/legacy_src/`（移動前に再度 grep で参照ゼロ確認）
5. `model_training/` → `_archive/layer2_legacy/model_training/`
   `strategy/` → 移植（§2 の3ファイル）完了後に `_archive/layer2_legacy/strategy/`
   ※ git 履歴には残るため物理退避で良い。`git rm -r` でリポから除去

### 3.2 ディレクトリ再編（To-Be）

```
RaceAI_var1.0/
├── common/data/          # L0 JV-Link（現状維持）
├── pure_rank/            # L1（現状維持。ただし↓を整理）
│   ├── src/              #   create_features / train / predict / evaluate のみ
│   └── analysis/         #   analyze_*.py, plot_*.py, export_scores.py を移動
├── prob_fusion/          # L2 新設
│   ├── config/fusion_config.json
│   ├── src/  (fit_fusion.py, market_prob.py, place_prob.py, predict_fusion.py)
│   ├── data/
│   └── tests/
├── betting/              # L3 新設
│   ├── config/betting_config.json
│   ├── src/  (ev_engine.py, kelly_sizer.py, risk_limits.py, backtest.py)
│   ├── data/
│   └── tests/
├── evaluation/           # 全層共通の評価基盤 新設
│   ├── splits.py         #   TRAIN/VALID/TEST 分割の唯一の定義
│   ├── market_baseline.py
│   └── reports/
├── main/                 # L4（binary系統の除去、Step構成の更新）
├── config/paths.json
├── docs/
│   ├── specs/            # 本書 + 今後のPhase仕様のみ
│   └── archive/          # 既存の30本超の旧spec・旧レポートを移動
├── tests/                # リポ横断の統合テスト（残差系テストはアーカイブ）
├── scripts/
└── CLAUDE.md
```

- `pure_rank/src/simulate_ev.py`・`wide_probability.py` は Phase 3 完了まで凍結扱いで残置
  （市場ベースライン再計算に使うため）。Phase 4 で `betting/` に吸収して除去
- 旧テスト（`tests/test_strategy_pipeline_*`, `test_inference_common_pure.py` 等の
  残差系依存分）はアーカイブへ。移植モジュールのテストは `betting/tests/` に付け直す

### 3.3 .gitignore / 衛生

- `*.parquet`, `models/`, `data/` 系の生成物が追跡されていないか棚卸しし、
  生成物は ignore + manifest のみ追跡に統一
- `__pycache__/`, `*.head` などの一時ファイルを除去
- 未コミットの変更（現在 working tree に多数）は Phase 0 開始前に
  ユーザー確認の上でコミットまたは破棄して**クリーンな状態から始める**

### 3.4 CLAUDE.md 改定

- 層定義を本書 §1.1 の4層に差し替え（L2/L3 の市場情報境界を明記）
- 「本番凍結」節から var2 系（`var1_init_score.beta=0.15` 等）を削除し、
  L2/L3 の凍結パラメータは各 Phase 合格時に追記する方式へ
- 禁止事項 10〜14（var2系）を L2/L3 版に書き換え
  （例: 「q を LightGBM 特徴量に入れない」「EV閾値のTEST後付け調整禁止」「n<200での有意主張禁止」は継続）
- 5エージェント体制・リーク停止閾値・Rule 3 は変更しない

---

## 4. Phase 1 — 評価基盤（evaluation/）

> 実施: implementer（TDD）。設計判断が要る場合のみ planner。

1. `evaluation/splits.py`: 分割定義の単一ソース化。既存 config の値を移し、
   pure_rank 側もこれを参照するよう変更（値は不変）
2. `evaluation/market_baseline.py`: `compute_favorite_baseline`（既存 simulate_ev.py から移植）
   + 市場確率 q の構築（比例正規化 / power法）+ 市場 logloss の算出。
   結果を `evaluation/reports/market_baseline.json` に固定保存
3. レポート様式: 全 Phase の評価は「モデル値 vs 市場ベースライン」併記を必須化

**合格ゲート**: 1番人気 Top-1=32.90% / 単勝ROI=77.94% が再現。市場 logloss が記録される。

## 5. Phase 2 — L2 確率統合層（prob_fusion/）

1. `market_prob.py`: 単勝オッズ→q（2方式）。WinOdds は既存
   `fetch_win_odds_yearly()`（`legacy_get_data_impl.py`）を利用
2. `fit_fusion.py`: 条件付きロジット MLE。fold 毎に (α, β) を推定し
   walk-forward で TEST に適用。出力は §1.2 スキーマ
3. `place_prob.py`: Stern による複勝率導出（λ は VALID 推定）
4. 診断: キャリブレーションカーブ・残差IC・尤度比検定を `evaluation/reports/` へ

**合格ゲート**: §1.3 の表。**logloss で市場に勝てなければ Phase 3 に進まない。**

## 6. Phase 3 — L3 ベッティング層（betting/）単勝・複勝

1. `ev_engine.py`: EV 計算 + ヘアカット + VALID 閾値選択
2. `kelly_sizer.py`: strategy/ から移植・レビュー（分数Kelly・排反減衰・上限制約）
3. `risk_limits.py`: 月次MDD・連敗停止
4. `backtest.py`: walk-forward（fold=2023/2024/2025+）。単勝・複勝それぞれ独立に評価

**合格ゲート**: §1.4 の表。不合格なら撤退基準に従い L2 改善へ戻る。

## 7. Phase 4 — ペア券種拡張（ワイド・馬連）※Phase 3 合格が前提

1. L2 統合勝率から Stern でペア確率を導出し、実測ペア的中率で 10-bin 校正確認
   （現行 P-16 の最大誤差19ppが、市場統合後にどこまで縮むかを最初に測る）
2. 校正が基準内（最大 5pp）に入った券種のみ EV ベット対象に追加
3. `pure_rank/src/simulate_ev.py`・`wide_probability.py` を `betting/` に吸収し除去

## 8. Phase 5 — 当日運用統合（main/）

1. `unified_pipeline.py` を L1→L2→L3 の直列に改修（binary/R-6 呼び出し除去 → P-46/P-47 解消）
2. 直前オッズ取得（取得時刻の記録付き）と推奨CSV（§1.2 L3スキーマ）
3. Notebook Step 構成を新パイプラインに同期
4. 運用ログ: 推奨と実結果の突合を月次レポート化（predicted p vs actual、ROI追跡）

## 9. 並行トラック — L1 継続改善（既存ロードマップ継続）

- 福島・小倉弱点（P-02/03）、多頭数（P-06）、短距離（P-05）は従来どおり
  planner→implementer→evaluator サイクルで継続。合否軸も従来どおり（Top-1/NDCG/Spearman）
- L1 が +1pp 改善すれば L2 の α が増え、L3 のエッジが直接増える。
  **L1 改善は本計画で初めて ROI に接続される**（従来は P-15 のとおり無関係だった）

---

## 10. 実装規約（全 Phase 共通）

1. superpowers:test-driven-development に従う（テスト先行）
2. 1パラメータずつ変更・1特徴量ずつ追加（憲法継続）
3. 生成物上書き前のバックアップ（憲法継続）
4. 各 Phase 完了時: evaluator の独立検証 → refactorer の市場情報混入チェック
   （grep 対象に `prob_fusion/` は q・ln_q 経路のみ許可のホワイトリスト方式を追加）
5. 乱数 seed 固定（42〜46 系列を継承）

## 11. リスクと未決事項

| リスク | 対応 |
|-------|------|
| α が有意でも EV フィルタ後のベット数が不足 | Phase 3 撤退基準で停止。γ 拡張・L1 改善で logloss を積み増してから再挑戦 |
| 確定オッズと購入時オッズの乖離で実運用 ROI が backtest を下回る | EV ヘアカット + 月次乖離監視（§1.5） |
| 複勝オッズの取得可否（JV-Link 複勝レンジ表示） | Phase 2 で調査タスク化。取得不可なら Phase 3 は単勝のみで判定 |
| q の控除率補正方式で結果が変わる | 2方式を VALID 比較（Phase 2 内、TEST は最終1回のみ） |
| 過去の「後出し」文化の再発 | すべての閾値・方式選択は VALID 限定。TEST は各 Phase 1回のみ実行し結果を manifest に封印 |

---

## 12. 完了の定義

- **的中率**: L2 統合確率の Top-1 ≥ 33.0%（市場 32.90% 超え）— Phase 2 で判定
- **回収率**: walk-forward 全 fold ROI > 100%（n≥200/fold、VALID決定→TEST検証）— Phase 3 で判定
- **運用**: 当日パイプライン一本化、推奨CSVが L2/L3 経由 — Phase 5 で判定
- **衛生**: リポにバックアップ遺物ゼロ、docs/specs は現行計画のみ、全生成物に manifest

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-07-08 | 初版（ユーザー承認: Benter型統合 / 旧L2廃止 / 遺物アーカイブ） |
