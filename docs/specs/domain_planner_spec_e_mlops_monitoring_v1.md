# Domain Planner Spec: E MLOps 監視（ドリフト・OOD）v1

**日付**: 2026-06-17  
**ステータス**: 確定（Phase 0 中: 本番外構築、アラートのみ）  
**担当**: domain-planner → deployment-evaluator → backtest-evaluator

---

## 1. 目的

カレンダー定期再学習を廃止し、**データ駆動トリガー**で calibrator 再 fit・運用判断を行う。

凍結期間（〜2026-07-15）中は **アラートのみ**（本番ベットロジック変更禁止）。

---

## 2. 概念ドリフト（レース後）

### ADWIN

- **入力**: レースごと LogLoss または Brier score
- **出力**: drift 検知時刻 + アラート
- **用途**: 急激な精度崩れ

### Page-Hinkley

- **入力**: 同上時系列
- **パラメータ**: $\delta$（許容ノイズ）、閾値 $\lambda$
- **用途**: 緩やかな劣化

**トリガー（凍結後）**: ADWIN **または** PH が 7 日以内に発火 → calibrator 再 fit **検討**（valid のみで fit）。

---

## 3. 共変量シフト / OOD（レース前）

### KS 検定（二標本）

- **比較**: 学習特徴量分布 $F_{train}$ vs 当日レース $F_{live}$
- **対象列**: 重要度上位 10 列（SHAP または固定リスト manifest）
- **有意水準**: 1%
- **アクション**: OOD フラグ → **当該レースベット額 0**（スキップ）

**リーク**: 学習分布は train split のみ。テスト期間分布を live 比較に混ぜない。

---

## 4. ダッシュボード（Track E 成果物）

| パネル | 内容 |
|--------|------|
| 週次 ROI | 単勝 / ワイド / 合算 vs baseline |
| Drift | ADWIN/PH 状態 |
| OOD 率 | 日次 OOD レース割合 |
| レジーム | D-2 の $P(S_t)$（実装後） |

**保存先**: `main/Resulut/` 以外の監視用 JSON（例: `logs/mlops/`）。

---

## 5. 本番 merge ゲート

1. 凍結終了
2. OOD スキップが 2025 OOF で ROI **+0pp 以上**（過剰スキップで n_bets 500 割れなら NG）
3. E2E PASS

---

## 6. リソース

Track E = 全体 **10%**。Phase 0 は ADWIN + 週次 JSON ログのみ MVP。
