# train_config_tuning — 学習設定（ハイパーパラメータ）の再検討

## 目的

特徴量ではなく **学習設定（LightGBMパラメータ）** 側で、まだ一度も検証していない軸を
1パラメータずつテストする（CLAUDE.md「実験は1パラメータずつ変更する」に準拠）。
ROIではなく的中率（Top-1/Top-3/NDCG@3/Spearman）向上が目的（ユーザー指示によりROI評価は対象外）。

対象は `pure_rank/config/train_config.json` に明示指定が無い、つまり
LightGBMのデフォルト値のまま放置されているパラメータ:

| 変数 | 現状 | 検証内容 |
|------|------|---------|
| `lambdarank_truncation_level` | 未指定（デフォルト=30） | 上位のみに勾配を集中させる値に変更 |
| `feature_fraction`/`bagging_fraction`/`bagging_freq` | 未指定（デフォルト=1.0, 0） | バギング型正則化の追加 |
| seed数 | 5（42-46） | 7-10に拡張し分散削減効果を確認 |

## プロトコル

- 特徴量: `pure_rank/data/02_features/features_v39_course_slim.parquet`（本番と同一、v39_course_slim）
- 分割: 本番 `pure_rank/src/train.py::get_fold_split` と同一の fold2（train<2023-01-01 / valid=2023）
- 評価: fold2の5モデル（またはvariant Cのみ拡張seed）でTEST期間（race_date>=2023-01-01、実質2023-2026含む。
  既存ベースライン `scores_v39_course_slim_fold2_oos.parquet` と同一範囲）をスコアリングし、
  `pure_rank/src/evaluate.py::compute_metrics` で Top-1/Top-3/NDCG@3/Spearman を算出
- 比較対象: 既存の `pure_rank/data/03_scores/scores_v39_course_slim_fold2_oos.parquet`
  （本番 train_config.json のパラメータで学習済みのfold2 OOSベースライン、再学習不要で流用）
- 本番ファイル（`pure_rank/config/`, `pure_rank/models/`, `pure_rank/src/`）には一切書き込まない

## 実行順

1. `evaluate_baseline.py` — 既存 v39 fold2 OOS ベースラインの指標を算出
2. `train_variant.py --variant truncation --truncation-level 3` — Variant A
3. `train_variant.py --variant bagging` — Variant B（feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1）
4. `train_variant.py --variant seeds10` — Variant C（seed 42-51の10本）
5. 各variantで `export_scores.py --variant <name>` → 指標算出 → `reports/comparison.json` に集約
