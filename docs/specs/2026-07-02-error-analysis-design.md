# 実装仕様書: エラー分析スクリプト — 2026-07-02

## 目的

テストセット（4,775レース、2025-01-05〜2026-05-24）で「予測1位が実際に3着以下になった」
レースを条件別に集計し、モデルが特に弱い条件を特定する。

現行ベースライン（v33_jt_ext、Top-1=30.37%）からの出発点として、次に追加すべき
特徴量の根拠を数値で示す。コードを書かずにデータ駆動で設計する。

## 禁止特徴量の確認

- [x] 分析スクリプトは既存モデル（models/）を使うのみ。新規特徴量は追加しない
- [x] オッズ・人気を参照しない
- [x] テストデータの結果を見て特徴量を後付け選択しない（分析後に仕様変更する場合は
      planner を通す）

## 実装対象ファイル

```
pure_rank/src/analyze_errors.py   # 新規作成
```

evaluate.py から以下を import して再利用する（コピーしない）：

```python
from evaluate import load_config, load_models, ensemble_predict, get_feature_cols
```

## データ仕様

### テストセット抽出

```python
valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])  # = 2024-12-31
df_test = df[df["race_date"] > valid_end_ts].copy()
# 期待値: 4,775 レース / 66,020 サンプル
```

### データフィルタ（train_config.json の filters と同一）

```python
df = df[
    (~df["grade_code"].isin(cfg["filters"]["exclude_grade_codes"])) &
    (~df["abnormal_code"].isin(cfg["filters"]["exclude_abnormal_codes"])) &
    (df["horse_count"] >= cfg["filters"]["min_horse_count"]) &
    (df["finish_rank"] > 0)
]
```

## 分析軸の定義

### テストセット条件別レース数（確認済み）

| 分析軸 | カラム | 値と件数 |
|--------|--------|---------|
| 馬場種別 | surface_code | 1=芝(2,224) / 2=ダート(2,380) / 5=その他(171) |
| 馬場状態 | track_condition_code | 0=不明(106) / 1=良(3,458) / 2=稍重(834) / 3=重(296) / 4=不良(81) |
| 距離カテゴリ | distance_category | 0=短距離(1,675) / 1=マイル(2,010) / 2=中距離(727) / 3=長距離(363) |
| 頭数バケット | horse_count | 5-8頭(277) / 9-12頭(1,100) / 13-16頭(2,977) / 17頭+(421) |
| グレード | grade_code | 1=一般(3,562) / 5=OP(1,101) / 7=重賞(98) / 2-4=特別(14合計) |
| 競馬場 | course_code | コードごと集計（全コード） |

### 注意事項

- `surface_code=5` は 171 レース存在する。除外はしないが「その他」としてラベルを付ける
- `track_condition_code=0` は 106 レース。「不明」としてラベルを付ける
- `grade_code` が 2/3/4 は合計 14 レースのみ。個別に出力するが「サンプル不足」フラグを付ける

## 計算する指標

各条件区分ごとに以下を計算する：

```python
def analyze_by_condition(df_eval: pd.DataFrame, col: str) -> dict:
    """
    df_eval には pred_score（予測スコア）と finish_rank が含まれる。
    col で条件を区分し、Top-1 的中率と件数を返す。
    """
    results = {}
    overall_top1 = 0.0  # 後で全体平均と比較するために使用

    for val, group in df_eval.groupby("race_id"):
        ...
    # race_id でグループ化してから、条件値でサブグループ化する
```

実際の実装は race_id 単位でループし、各レースのメタ情報（surface_code 等）は
レース内で一意のため先頭行から取得する：

```python
meta_cols = ["surface_code", "track_condition_code", "distance_category",
             "grade_code", "horse_count", "course_code"]

for race_id, grp in df_eval.groupby("race_id"):
    pred_best_idx = grp["pred_score"].idxmax()
    is_hit = int(grp.loc[pred_best_idx, "finish_rank"] == 1)
    meta = grp.iloc[0][meta_cols]
    # 各条件軸の辞書に is_hit を追記
```

### 頭数バケット変換

```python
horse_count_bucket_label = pd.cut(
    df_eval["horse_count"],
    bins=[4, 8, 12, 16, 100],
    labels=["5-8", "9-12", "13-16", "17+"]
)
```

## 出力フォーマット

ファイル: `pure_rank/data/02_features/error_analysis_v33.json`

```json
{
  "model_version": "v33_jt_ext",
  "test_period": {"start": "2025-01-05", "end": "2026-05-24"},
  "overall": {
    "top1_rate": 0.3037,
    "n_races": 4775
  },
  "by_surface_code": {
    "1": {"top1_rate": 0.0, "n_races": 2224, "label": "芝"},
    "2": {"top1_rate": 0.0, "n_races": 2380, "label": "ダート"},
    "5": {"top1_rate": 0.0, "n_races": 171, "label": "その他"}
  },
  "by_track_condition_code": {
    "0": {"top1_rate": 0.0, "n_races": 106, "label": "不明"},
    "1": {"top1_rate": 0.0, "n_races": 3458, "label": "良"},
    "2": {"top1_rate": 0.0, "n_races": 834, "label": "稍重"},
    "3": {"top1_rate": 0.0, "n_races": 296, "label": "重"},
    "4": {"top1_rate": 0.0, "n_races": 81, "label": "不良"}
  },
  "by_distance_category": {
    "0": {"top1_rate": 0.0, "n_races": 1675, "label": "短距離(~1400m)"},
    "1": {"top1_rate": 0.0, "n_races": 2010, "label": "マイル(1401-1800m)"},
    "2": {"top1_rate": 0.0, "n_races": 727, "label": "中距離(1801-2200m)"},
    "3": {"top1_rate": 0.0, "n_races": 363, "label": "長距離(2201m+)"}
  },
  "by_horse_count_bucket": {
    "5-8":  {"top1_rate": 0.0, "n_races": 277},
    "9-12": {"top1_rate": 0.0, "n_races": 1100},
    "13-16":{"top1_rate": 0.0, "n_races": 2977},
    "17+":  {"top1_rate": 0.0, "n_races": 421}
  },
  "by_grade_code": {
    "1": {"top1_rate": 0.0, "n_races": 3562, "label": "一般"},
    "5": {"top1_rate": 0.0, "n_races": 1101, "label": "オープン"},
    "7": {"top1_rate": 0.0, "n_races": 98, "label": "重賞"},
    "2-4": {"top1_rate": 0.0, "n_races": 14, "label": "特別(小サンプル)", "warning": "n<100"}
  },
  "by_course_code": {},
  "worst_conditions": [
    {
      "axis": "track_condition_code",
      "value": "4",
      "label": "不良",
      "top1_rate": 0.0,
      "n_races": 81,
      "gap_vs_overall": -0.05
    }
  ]
}
```

`worst_conditions` には Top-1 が全体（30.37%）より **5pp 以上低く**、かつ **n_races >= 50** の
条件をすべてリストアップする。

## コンソール出力

スクリプト実行時に以下を出力する：

```
=== Error Analysis: v33_jt_ext ===
Overall Top-1: 30.37%  (4,775 races)

--- by surface_code ---
  芝(1)       : XX.X%  (2,224 races)  [diff: +X.Xpp]
  ダート(2)   : XX.X%  (2,380 races)  [diff: +X.Xpp]
  その他(5)   : XX.X%  (  171 races)  [diff: +X.Xpp]

--- by track_condition_code ---
  良(1)       : XX.X%  (3,458 races)  [diff: +X.Xpp]
  稍重(2)     : XX.X%  (  834 races)  [diff: +X.Xpp]
  重(3)       : XX.X%  (  296 races)  [diff: +X.Xpp]
  不良(4)     : XX.X%  (   81 races)  [diff: +X.Xpp]  ← 不良馬場での弱点

...

=== WORST CONDITIONS (gap > -5pp, n >= 50) ===
  1. [track_condition_code=4/不良]  XX.X%  (81 races)   gap=-X.Xpp
  2. ...

Results saved: pure_rank/data/02_features/error_analysis_v33.json
```

## evaluatorへの依頼事項

error_analysis_v33.json の結果を受け取ったら、以下を判断して planner に報告する：

1. **track_condition 別格差**: 不良馬場（code=4）の Top-1 が全体より 5pp 以上低い場合
   → `hist_sire_track_condition_win_rate_ts` の優先度を高める

2. **course_code 別格差**: 特定の競馬場コードで Top-1 が 20% を下回る場合
   → その競馬場専用の特徴量が必要

3. **大頭数格差**: 17頭+で Top-1 が全体より 5pp 以上低い場合
   → フィールドサイズ調整特徴量の追加を検討

4. **距離別格差**: 長距離（distance_category=3）で大きな格差がある場合
   → 長距離適性特徴量の不足

## 評価基準

- スクリプト実行後の全体 Top-1 が 30.37% と一致すること（±0.1pp 以内）
- n_races の合計が 4,775 と一致すること
- JSON が正常に保存されること

## implementerへの引き渡し事項

1. `pure_rank/src/analyze_errors.py` を新規作成する
2. `evaluate.py` の `load_config`, `load_models`, `ensemble_predict`, `get_feature_cols` を import して再利用する
3. 分析軸は上記の 6 軸（surface / track_condition / distance_category / horse_count / grade_code / course_code）
4. 全体 Top-1 が 30.37% と一致することを確認してから JSON を保存する
5. `analyze_phase_b.py` の構造（race_id ループ、defaultdict の利用）を参考にしてよい
6. 実行コマンド: `python pure_rank/src/analyze_errors.py`
