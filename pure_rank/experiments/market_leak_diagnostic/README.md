# market_leak_diagnostic — 一回限りの診断実験

**目的**: L1（LambdaRank）に市場情報（確定単勝オッズ・人気・ln(市場確率)）を
直接特徴量として与えたら Top-1 / ROI がどう変わるかを見る「上限診断」。

**位置づけ**: これは CLAUDE.md の L1 市場情報禁止ルールに意図的に反する
一回限りの隔離実験。本番 `pure_rank/src/`, `pure_rank/config/train_config.json`,
`pure_rank/models/` には一切触れない。結果は `evaluation/reports/gate_summary.json`
の合否判定には反映しない。

**既知の制約**: このリポジトリには真の「レース前オッズ」データが存在しない。
`common/data/output/odds/WinOdds_*.csv` は SE レコード由来の**確定（確定後）
オッズ・人気**であり、投票締切時点のオッズではない。したがってこの実験は
「レース前情報として使える市場情報の効果」ではなく、「確定オッズ相当の情報が
特徴量に入った場合の理論的上限」を測る診断である。

## 実行手順

```bash
# 1. 実験用特徴量を構築（本番 features_v39_course_slim.parquet + 確定オッズ列）
python pure_rank/experiments/market_leak_diagnostic/build_features.py

# 2. fold2 のみ 5 シード学習（本番 fold2 OOS 測定と同一プロトコル）
python pure_rank/experiments/market_leak_diagnostic/train_fold2.py

# 3. fold2 OOS スコアをエクスポート
python pure_rank/experiments/market_leak_diagnostic/export_scores.py

# 4. L2 Benter 統合 + L3 単勝バックテスト（本番 evaluation/reports/ は汚さない設計）
python pure_rank/experiments/market_leak_diagnostic/run_oos_backtest.py
```

## 追加特徴量（本番 FORBIDDEN_COLS を意図的に迂回するための別名）

| 列名 | 内容 |
|------|------|
| `exp_win_odds` | 確定単勝オッズ |
| `exp_ln_odds` | ln(確定単勝オッズ) |
| `exp_popularity` | 確定単勝人気順位 |
| `exp_market_log_odds` | ln(市場確率)。1/odds をレース内正規化（proportional 法、prob_fusion と同一式） |
