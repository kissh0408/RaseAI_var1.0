# error_analysis_popularity_margin — 診断専用（本番非適用）

**目的**: 2つの診断を行う。

1. **人気帯別の的中傾向**: モデル予測の的中率・見逃し率を実際の単勝人気帯（1人気, 2人気,
   3人気, 4人気, 5人気以上）で分解する。「5人気以下の穴馬が上位入賞したとき、モデルは
   拾えていたか」を測る。
2. **僅差ニアミス分布**: 予測1位馬が外れたケースのうち、勝ち馬とのタイム差がどの程度の
   僅差だったかを分布として見る（鼻差級のニアミスがどれだけあるか）。

**位置づけ**: 市場情報（オッズ・人気）は**評価レイヤーの診断のみ**に使用し、学習・特徴量
には一切投入しない（CLAUDE.md Rule 1 は特徴量禁止であり、事後評価での使用は
`evaluation/market_baseline.py` の favorite baseline 測定と同じ扱いで許容される）。

本スクリプトは `pure_rank/src/`, `pure_rank/config/`, `pure_rank/models/`,
`evaluation/reports/gate_summary.json` のいずれにも一切書き込まない。結果は
`results/` 配下のみに出力する。

対象データ: fold2 OOS スコア（`scores_v39_course_slim_fold2_oos.parquet`）、
TEST期間（2025-01-01以降。`prob_fusion/src/oos_protocol.py::TEST_START` と同一）。

## 実行

```bash
python pure_rank/experiments/error_analysis_popularity_margin/run_analysis.py
```
