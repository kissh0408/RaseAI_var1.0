# デプロイ後定量戦略ロードマップ v2

**確定日**: 2026-06-17  
**ステータス**: **最高意思決定（確定）** — v1 を supersede  
**対象システム**: ensemble_v5 + specv2 cal + C1 ワイド本番

---

## 1. エグゼクティブ・サマリー

2025 OOF において単勝 ROI **151%** / MDD **-19.9%**、ワイド anchor ROI **162%** / 的中 **23.4%** を確認。一方、2019・2020・2022 年は MDD **-86〜-98%** とレジーム依存が顕著。

本番の成功定義・研究配分・凍結期限を以下で確定する。

| # | 決定事項 | 選択 |
|---|----------|------|
| ① | 北極星 KPI | **案1改** — Floor ROI ≥ **115%** / Target ≥ **130%**、MDD ≤ -20%、合算的中 ≥ 20% |
| ② | Track B | **A** — 並行 R&D（20%）、3 連続失敗で構造限界と明記 |
| ③ | Track C2 | **a** — win+wide 合算 MDD → 多変量 Kelly 配分 |
| ④ | Track D | **a** — D-1 ヒートマップ → **HMM 動的レジーム検知** |
| ⑤ | 凍結終了 | **2026-07-15**（デプロイから 4 週） |

---

## 2. 北極星 KPI（本番）

### 3 段階（ポートフォリオ = 単勝 + ワイド合算）

| Tier | ROI | MDD | 用途 |
|------|-----|-----|------|
| **リジェクト** | **< 100%** | — | 変更ロールバック・リリース不可 |
| **Floor（維持）** | **≥ 115%** | ≤ -20% | 週次モニタ・アラート。本番北極星の下限 |
| **Target（正常）** | **≥ 130%** | ≤ -20% | OOF / 年次の「健全」目標 |
| **Stretch** | **≥ 145%** | ≤ -20% | 2025 単勝 baseline（151%）付近の維持 |

| その他 | 目標 |
|--------|------|
| 合算的中率 | ≥ 20% |
| n_bets（年） | ≥ 500 |
| 月次 DD 停止 | -8%（`monthly_dd_tracker`） |
| 2025 単勝単体 | ROI ≥ 145%（C2 後 ±5pp 以内） |

**105% との関係**: `standard_eval_pipeline_v1` / CLAUDE.md の **単勝 OOF ゲート（105%）** は変更しない。北極星はポートフォリオ合算で **115% / 130%** を用いる。

**本番に含めない KPI**: binary 単勝的中 25%（Track B 専用・別ゲート）。

**baseline 参照**: `baseline_standard_eval.json`（2025 production_live）

---

## 3. 研究リソース配分

| 比率 | Track | 内容 |
|------|-------|------|
| **40%** | A + C1-6 | 本番モニタ・実績 vs OOF ギャップ |
| **30%** | C2 + D-1 | 多変量 Kelly・レジーム分析/HMM |
| **20%** | B | 別 config R&D（単勝25% / duration 移行） |
| **10%** | E | MLOps 監視（ドリフト・OOD） |

---

## 4. タイムライン

### Phase 0: 完全凍結（〜 **2026-07-15**）

| 禁止 | 許可例外 |
|------|----------|
| モデル・HP・特徴量パイプライン変更 | O3/JV-Link 等 **致命的バグ** |
| ev_threshold / Kelly / calibrator 変更 | ログ・モニタ追加（本番出力に影響なし） |
| condition_ev 追加（テスト由来） | Track E 監視の **本番外** 構築 |

**成果物**: 4 週実績レポート（単勝・ワイド別 ROI/MDD/的中/n）

### Phase 1: C2（2026-07-15 〜 2 ヶ月）

多変量フラクショナル Kelly による win+wide 配分。詳細: `domain_planner_spec_c2_multivariate_kelly_v1.md`

| ID | 内容 | ゲート |
|----|------|--------|
| C2-1 | OOF win+wide 合算 MDD シミュレーション | 2025 MDD ≤ -20% |
| C2-2 | 多変量 Kelly + fractional 制約実装 | 単勝 ROI 151%±2pp 維持 |
| C2-3 | ベイズ CLV スキップ回路（任意・Phase 1b） | EV 閾値未満で棄却 |
| C2-4 | 本番反映（1 変数: 配分のみ） | 合算 ROI ≥ 115%（Target 130%）+ E2E |

**既存資産**: `strategy/src/betting_framework.py` の `simultaneous_kelly_fractions` を拡張起点とする。

### Phase 2: D（並行 1〜3 ヶ月）

レジーム耐性。詳細: `domain_planner_spec_d_regime_detection_v1.md`

| ID | 内容 |
|----|------|
| D-1 | 学習期間のみ: 年×月×馬場 ROI ヒートマップ（**静的除外ルールは採用しない**） |
| D-2 | HMM/GMM レジーム推定（Brier MA・市場効率・環境ノイズ） |
| D-3 | レジーム別 Kelly 乗数（Risk-On 1.0 / Risk-Off 0.25 / Crash 0） |
| D-4 | condition_ev 候補（D-1 根拠・年別安定性確認後のみ） |

**全期間 OOF 目標**: MDD **-25% 以内**（現 -31.5%）。2025 ゲートは維持。

### Phase 3: E（凍結中〜並行）

MLOps 監視。詳細: `domain_planner_spec_e_mlops_monitoring_v1.md`

| 機構 | 用途 |
|------|------|
| ADWIN | 急激な概念ドリフト（LogLoss/Brier） |
| Page-Hinkley | 緩やかなドリフト |
| KS 検定 | レース前 OOD → ベットスキップ |

**本番 merge**: calibrator 再 fit トリガーを **日付ではなく drift 検知**に委譲（Phase 0 中はアラートのみ）。

### Phase 4: B（3〜6 ヶ月・別ゲート）

| ルール | 内容 |
|--------|------|
| B-1〜B-3 | max_odds grid / 特徴量 1 本 / Rank v24（各 1 変数） |
| **打切り** | 3 実験連続で valid 的中 +3pt 未満 → 構造限界を仕様に明記 |
| **パラダイム移行** | `domain_planner_spec_b_duration_forecasting_v1.md` — 走破時間回帰 + MC シミュレーション |

---

## 5. レジーム状態と自動対応（D-2 確定案）

| 状態 | 観測の目安 | Kelly 乗数 |
|------|------------|------------|
| **Risk-On** | Brier 安定、オッズ変動小 | C2 の $a_f$ × **1.0** |
| **Risk-Off** | 予測誤差拡大、馬場/オッズ不安定 | × **0.25** |
| **Crash** | 構造崩壊・月次 DD -8% 兆候 | × **0**（モニタのみ） |

---

## 6. やらないこと（v1 継承 + 追加）

- specv2 モデル本番 promotion（v5≈specv2）
- テスト fold から condition_ev / 静的月別除外
- win/wide 独立 Kelly の単純合算（C2 禁止）
- Track B 成果の無ゲート本番 merge
- 凍結期間中の calibrator 再 fit（drift アラートのみ）

---

## 7. エージェント分担

| Track | domain-planner | model-strategy-generator | backtest-evaluator | data-generator | deployment-evaluator |
|-------|----------------|--------------------------|--------------------|----------------|----------------------|
| A/C1-6 | — | — | 週次実績 vs baseline | — | ログ・DD |
| C2 | 仕様 | Kelly 実装 | OOF 合算 MDD | — | E2E |
| D | HMM 仕様 | Kelly 乗数 | 年別・全期間 | 観測変数 | レジーム API |
| E | 監視仕様 | — | — | 特徴量分布 | ダッシュボード |
| B | duration 仕様 | 回帰/MC | 別 baseline | 特徴量 | — |

---

## 8. 関連ドキュメント

| ファイル | 内容 |
|---------|------|
| [post_deploy_roadmap_v1.md](post_deploy_roadmap_v1.md) | 旧版（参照のみ） |
| [domain_planner_spec_c2_multivariate_kelly_v1.md](domain_planner_spec_c2_multivariate_kelly_v1.md) | C2 数学・実装要件 |
| [domain_planner_spec_d_regime_detection_v1.md](domain_planner_spec_d_regime_detection_v1.md) | D-1/HMM |
| [domain_planner_spec_e_mlops_monitoring_v1.md](domain_planner_spec_e_mlops_monitoring_v1.md) | ドリフト・OOD |
| [domain_planner_spec_b_duration_forecasting_v1.md](domain_planner_spec_b_duration_forecasting_v1.md) | B パラダイム移行 |
| [standard_eval_pipeline_v1.md](standard_eval_pipeline_v1.md) | 評価土俵 |
