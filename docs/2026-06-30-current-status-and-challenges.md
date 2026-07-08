# RaceAI_var1.0 — 現状サマリーと課題（2026-06-30）

## 1. プロジェクト概要

市場情報（オッズ・人気）を一切使わない純粋能力ベースの競馬着順予想AI。  
JV-Link から取得した馬の能力・血統・調教・騎手データのみで着順を予測する。

- **主モデル**: LightGBM LambdaRank（5シード × 3フォールド = 15モデル）
- **特徴量バージョン**: v29_fixed（98列）
- **テストセット**: 4,775レース（2025-01-05〜2026-05-24）

---

## 2. ランキング精度の現状

### 最終スコア（v29_fixed + label_gain A-3）

| 指標 | 結果 | 合格基準 | 判定 |
|------|------|---------|------|
| Top-1 的中率 | **30.18%** | >30% | ✅ 合格 |
| Top-3 的中率 | **62.05%** | >55% | ✅ 合格 |
| NDCG@3 | **0.5377** | >0.52 | ✅ 合格 |
| Spearman 相関 | **0.5063** | >0.50 | ✅ 合格 |

Phase 7 ベースライン（Top-1=28.5%）を **+1.68pp** 上回り、目標の >30% を達成。

### 精度推移

| バージョン | Top-1 | 主な変更 |
|-----------|-------|---------|
| v6 | 27.9% | 騎手・調教師追加（速度指数バグあり） |
| v11 | 28.6% | 速度指数バグ修正（distance_category→distance） |
| v17 | 29.5% | 間隔カテゴリ別勝率（hist_interval_cat_win_rate）追加 |
| v23 | 29.7% | 季節別勝率（hist_season_win_rate）追加 |
| v29_fixed | 29.9% | **LambdaRank 行順序バグ修正** |
| **v29_fixed + A-3** | **30.2%** | **label_gain [0,1,3,7,15,31,100]（1位重み強化）** |

---

## 3. 今セッションで発見・修正した重大バグ

### LambdaRank グループ割り当てバグ（最重要）

**症状**: v11 を含む全バージョンで `get_group_sizes(sort=False)` が正しく機能していなかった。

**原因**:  
`create_features.py` が `sort_values(["ketto_num", "race_date"])` でソートして parquet を保存していた。  
`get_group_sizes(sort=False)` は行順序どおりにグループ境界を決めるため、異なるレースの馬が同一グループとして扱われていた。

```
バグ時:  RL=[1,1,1,1,1]  vs  正しい groupby=[12,16,14,17,...]
```

LightGBM は「異なるレースの馬同士」を比較してランキングを学習していた。

**修正**:  
`create_features.py` の `to_parquet()` 直前に以下を追加：

```python
df = df.sort_values(["race_date", "race_id", "horse_num"])
```

**影響**: 修正後 Top-1 が +0.2pp 改善（29.7% → 29.9%）。

---

## 4. 実装済みコンポーネント

### 4-1. ランキングモデル（pure_rank/src/）

| ファイル | 内容 |
|---------|------|
| `create_features.py` | 特徴量生成パイプライン（98列、v29_fixed） |
| `train.py` | LambdaRank 学習（5シード × 3フォールド） |
| `evaluate.py` | Top-N/NDCG/Spearman 評価 |
| `preprocess.py` | JV-Link SE/RA/HC/WC 前処理 |

### 4-2. Plackett-Luce 確率変換（Phase A — 評価済み合格）

`pure_rank/src/predict.py`

- **Softmax 温度キャリブレーション**: T=0.76（2024年バリデーションで Log-Loss 最小化）
- **Harville 公式**: 勝率 p_i から 2着・3着・ワイド・馬連確率を解析的に導出
- **評価指標（テストセット）**:

| 指標 | 値 |
|------|-----|
| top3_coverage_rate | 61.78% |
| wide_pair_coverage_rate | 28.80% |
| quinella_pair_coverage_rate | 13.97% |
| wide_harville_coverage_rate | 28.80% |

### 4-3. EV シミュレーション（pure_rank/src/simulate_ev.py）

HR 払戻データを用いた EV 分析・閾値スイープ・条件別 ROI 計算を実装。  
**ただし EV 計算に根本的な問題あり（後述）。**

### 4-4. キャリブレーション（feature/calibration-improvement ブランチ）

3手法を実装・評価済み（不採用）：

| 手法 | EV>=1.0 wide ROI | 判定 |
|------|-----------------|------|
| ベースライン全件 | 80.99% | — |
| Platt スケーリング | 26.76% | 不採用 |
| ROI-T=0.30 最適化 | 80.84% | 不採用（実質フィルタなし） |
| Isotonic wide | 39.56% | 不採用 |

---

## 5. 現在の課題

### 課題1（最重要）: EV 計算に事前オッズが未使用

**現在の実装**:
```python
EV = p_wide × HR払戻（結果） / 100
```

**問題**:  
`HR` レコードは**レース結果の払戻金**であり、**レース前の事前オッズではない**。  
外れレースには払戻データが存在しないため、全体平均で代替している。これは：
- 当該レース・当該ペアの真の市場評価を反映しない
- EV > 1.0 フィルタが機能しない（外れレースに一律の平均値を使うため）

**正しい計算**:
```python
EV = p_wide × OR事前ワイドオッズ / 100
```

**解決策**:  
JV-Link の **OR レコード**（事前オッズデータ）を取得して pipeline に組み込む。  
`common/data/src/` に OR 取得スクリプトを追加する必要がある。

---

### 課題2: 市場はすでに純粋能力を織り込み済み

キャリブレーション分析の結果、次の構造的課題が判明：

```
モデル p_wide ≈ 28%
市場の平均ワイド払戻 ≈ 350円
EV ≈ 0.28 × 3.50 = 0.98
```

純粋能力モデルの予測は市場（オッズ）にほぼ織り込まれており、  
同じ情報だけでは EV > 1.0 を継続して作るのが困難。

**突破口の候補**:
1. **OR 事前オッズで正しい EV を計算** → 市場が過小評価している馬を発見
2. **展開・ペース特徴量の追加** → 市場が反映しにくい相互作用情報を活用
3. **特定レース条件への絞り込み** → 得意条件でのみベット

---

### 課題3: Phase B 相対特徴量が有効でなかった

v30_relative（within-race z-score 9列追加）を試験したが、Top-1 が 30.2% → 29.86% に後退。  
既存特徴量（`hist_last_time_dev` など）が相対差を暗黙的にカバーしており、  
重複情報がノイズとなった可能性が高い。

---

## 6. 採用見送り・不採用一覧

| 施策 | 理由 |
|------|------|
| TM（対戦型マイニング予想） | hist_last_time_dev と強い共線性（r=-0.54） |
| DM（タイム型マイニング予想） | finish_rank との Spearman=0.08（ほぼランダム信号） |
| 着差コード（margin_code） | SE 前処理で除外済み。time_diff で代替可能だが hist_last_time_dev と r=0.837 で重複 |
| v30_relative（相対特徴量） | Top-1 -0.34pp 後退。情報の重複が原因 |
| Platt スケーリング | EV ROI 26.76%（悪化） |
| ROI-T=0.30 最適化 | 実質フィルタなし（99.7%通過）、改善なし |
| Isotonic wide キャリブレーション | EV ROI 39.56%（悪化） |

---

## 7. 次フェーズ候補

### 優先度 高

#### A. OR レコード取得 → 正しい EV 計算
- `common/data/src/` に OR 取得スクリプトを追加
- `simulate_ev.py` の EV 計算を事前オッズベースに修正
- EV > 1.0 フィルタが機能するか再評価

#### B. 展開・ペース特徴量（Phase B-2 相対特徴量再設計）
- 先行馬密度（`field_front_runner_density`）を再設計
- running_style_code の過去走集計を shift(1) で正しく実装
- v30 の失敗を踏まえ、真に新規情報のみ追加

### 優先度 中

#### C. レース条件フィルタリング
- 芝・短距離・特定コースで ROI が高い条件を特定
- evaluate.py に条件別 Top-1 の出力を追加して分析

#### D. Phase 3〜5 特徴量（仕様書は作成済み）

| Phase | 追加特徴量 | 目標 Top-1 | 仕様書 |
|-------|----------|-----------|-------|
| Phase 3 | 血統（父適性・母父・ニックス） | >31% | `docs/superpowers/specs/2026-06-29-phase3-bloodline-class-features-design.md` |
| Phase 4 | 騎手・調教師（直近30日成績） | >32% | `docs/superpowers/specs/2026-06-29-phase4-jockey-trainer-design.md` |
| Phase 5 | TMタイム指数・通算賞金 | >33% | `docs/superpowers/specs/2026-06-29-phase5-speed-index-design.md` |

---

## 8. 現在のファイル構成

```
pure_rank/
├── config/
│   └── train_config.json          # features_version=v29_fixed, label_gain=[0,1,3,7,15,31,100], T_opt=0.76
├── data/
│   ├── 01_preprocessed/           # SE/RA/HC/WC 前処理済み Parquet
│   └── 02_features/
│       ├── features_v29_fixed.parquet   # 現行最良特徴量（98列）
│       ├── eval_results.json            # Top-1=30.18%, NDCG@3=0.5377
│       ├── ev_results.json              # EV シミュレーション結果
│       └── charts/
│           ├── hit_rate_by_score_summary.json
│           ├── calibration_wide.json    # キャリブレーション誤差=8.71%
│           └── calibration_comparison.json
├── models/
│   ├── lambdarank_fold*_seed*.txt  # 15モデル（v29_fixed + A-3）
│   └── calibration/               # Platt/Isotonic/T_roi モデル
└── src/
    ├── create_features.py          # 特徴量生成（行順序バグ修正済み）
    ├── train.py                    # LambdaRank 学習
    ├── evaluate.py                 # Top-N/NDCG/Spearman 評価 + 補助指標
    ├── predict.py                  # Softmax T=0.76 + Harville 確率変換
    └── simulate_ev.py              # EV シミュレーション（※事前オッズ未使用）
```

---

## 9. Git 状態

| ブランチ | 内容 | 状態 |
|---------|------|------|
| master | v29_fixed + A-3 + predict.py + simulate_ev.py | push 済み |
| feature/calibration-improvement | 3手法キャリブレーション + refactorer 整理 | push 済み・未マージ |

---

## 10. 判定サマリー

| 項目 | 状態 |
|------|------|
| ランキング精度（Top-1 >30%） | ✅ **達成（30.18%）** |
| 市場情報（オッズ・人気）の排除 | ✅ 完全排除 |
| 時系列リーク防止 | ✅ shift(1) で全特徴量対応済み |
| Phase 7 ベースライン超え | ✅ +1.68pp |
| 市場ベンチマーク超え（≈31-33%） | 🔄 残り 0.8〜2.8pp |
| ROI 100% 超え | ❌ **未達成（最良 80.99%）** |
| EV 計算の正確性 | ❌ **OR 事前オッズ未取得** |
