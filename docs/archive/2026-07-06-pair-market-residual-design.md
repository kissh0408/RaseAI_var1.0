# 実装仕様書: ワイド市場乖離ペア戦略（Step 3 正式仕様）— 2026-07-06

## 禁止特徴量の確認

- [x] WideOdds を `features_*.parquet` にマージしない（ベッティング層のみ）
- [x] implied_probability を学習特徴量に使わない
- [x] Layer 1 `init_score` に市場オッズを使わない
- [x] P-35: `var1_market_blend.enabled=false` を Layer 2 本番で維持

統合元:

- `docs/specs/2026-07-01-wide-odds-ev-integration-design.md`（真 EV）
- `docs/specs/2026-07-01-market-divergence-betting-design.md`（乖離スコア）

---

## 1. 目的

単一確率モデルから P_wide を導出し、WideOdds との **真 EV** と **市場乖離** でペアを選ぶ。
Step 1（真 EV）→ Step 2（bracket Isotonic）→ Step 3（Strategy D）を本番統合する。

---

## 2. 確率ソース（L1 / L2 両方比較）

| ID | ソース | P_win 入力 | P_wide 導出 | 本番用途 |
|----|--------|-----------|-------------|----------|
| **L1** | pure_rank | ensemble_score → T → softmax | Harville + bracket Isotonic | 能力純粋＋キャリブレーション |
| **L2** | binary layer | `model_prob`（init_score 済み） | `harville_wide_pair_prob` | 単勝パスと整合 |

実装:

- L1: `pure_rank/src/wide_probability.py` → `compute_calibrated_wide_probs`
- L2: `wide_probability.wide_probs_from_model_prob_frame` / `strategy/src/ev_filters.wide_probs_from_win_probs`

---

## 3. 市場側（全ソース共通）

```python
p_implied_raw(i, j) = 1.0 / wide_odds(i, j)
overround = sum(p_implied_raw) over valid pairs in race
p_implied(i, j) = p_implied_raw / overround
```

共有モジュール: `strategy/src/wide_ev_core.py`

---

## 4. EV と乖離

```python
EV_raw(i, j) = P_wide_model(i, j) × wide_odds(i, j)
log_divergence = log(P_wide_model × wide_odds × overround)
             = log(P_wide_model / p_implied)
edge ⟺ log_divergence > 0 ⟺ P_model > p_implied
```

---

## 5. 戦略定義

| Strategy | ペア選択 | ベット条件 |
|----------|----------|------------|
| A | `argmax P_wide` | `EV >= ev_threshold` |
| B | `argmax log_divergence` | `EV >= ev_threshold` |
| C | `argmax log_divergence` | `log_divergence > div_threshold` |
| **D（本番）** | `argmax log_divergence` | `EV >= ev_threshold AND log_divergence > div_threshold` |

---

## 6. 閾値・ソース選定（VALID のみ）

```python
ev_threshold ∈ {1.0, 1.05, 1.1, 1.2}
div_threshold ∈ {0.0, 0.1, 0.2}
score = ROI × sqrt(n_bets)   # n_bets >= 100
```

- VALID 2024 でグリッド探索 → `model_training/config/train_config.json` の `wide_betting` に記録
- TEST 2025+ は選定後 **1 回のみ** 報告（後出しじゃんけん禁止）

---

## 7. 合格基準

| 指標 | VALID 選定 | TEST 報告 |
|------|-----------|-----------|
| Strategy D vs A | ROI +3pp 以上 | 参考 |
| bracket MAE | < 0.06 | — |
| bracket ROI @ EV≥1.05 | +2pp vs 非 bracket | — |
| L1 vs L2 | `ROI×√n` 最大を採用 | 両方記録 |

Step 3 合格 = Strategy D が Strategy A より VALID ROI **+3pp 以上**（100% ROI は必須としない）。

---

## 8. Kelly / リスク

- ¼ Kelly（単勝と同系）: `kelly_fraction = max((b×p - q)/b, 0) / 4`
- 1 レース最大 **1 ワイドペア**（初期）。`strategy_config.json` で 2 ペアまで拡張可
- 参照: `docs/specs/2026-07-01-risk-adjusted-evaluation-design.md`

---

## 9. 実装マップ

| ファイル | 役割 |
|----------|------|
| `strategy/src/wide_ev_core.py` | odds ロード、overround、EV、乖離、閾値探索 |
| `pure_rank/src/wide_probability.py` | L1 キャリブレーション推論 |
| `pure_rank/src/simulate_ev.py` | `--divergence-compare`, bracket VALID gate |
| `strategy/src/combo_backtest.py` | `compare_l1_l2_wide_divergence` |
| `strategy/src/strategy_engine.py` | Strategy D 本番推奨 |
| `main/unified_pipeline.py` | venue CSV へ wide 列出力 |
| `main/notebook_bootstrap.py` | Step 9 表示 |

---

## 10. P-35 遵守

- Layer 2 本番: `var1_market_blend.enabled=false`
- L1 `market_blend`（`predict.py --fit-market-blend`）は **eval 比較サブセクションのみ**。Step 3 本番には入れない（二重カウント防止）

---

## 11. データ要件

| データ | パス | 用途 |
|--------|------|------|
| WideOdds 歴史 | `common/data/output/odds/WideOdds_{year}.csv` | VALID/TEST EV |
| O3 当日 | `common/data/output/realtime_odds/o3_odds.csv` | 当日 unified 表示 |
| race_id | 16 桁 → 先頭 14 桁で突合 | features / rank_preds |

取得: `python model_training/scripts/fetch_wide_odds_yearly.py`（JV-Link 32bit 必須）
