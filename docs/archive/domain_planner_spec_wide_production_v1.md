# Domain Planner Spec: ワイド本番統合 v1（Track C1）

**日付**: 2026-06-17  
**ステータス**: ✅ 本番反映済み（2026-06-17）  
**前提**: Track A デプロイ完了（specv2 cal + 2025 ゲート合格）

---

## 1. 背景

OOF 評価（`baseline_standard_eval.json`）では **wide anchor** が単勝より高い的中率・ROI を示す。

| プロファイル | 2025 wide anchor | 単勝 |
|-------------|------------------|------|
| production + specv2 cal | 的中 23.4% / ROI 162% | 的中 11.7% / ROI 151% |

本番は `online_phase=phase1`（ヒューリスティック連複・ワイド、EV なし）のままだったため、**phase1_5 + O3 オッズ** に切り替える。

---

## 2. スコープ（1変数ルール）

| 変更 | 内容 |
|------|------|
| ✅ | `online_phase` → `phase1_5` |
| ✅ | `wide_bets_enabled=true` |
| ✅ | `wide_min_edge=0.05`（valid sweep 採用値） |
| ❌ | 単勝 EV / Kelly / calibrator（Track A 凍結） |
| ❌ | 馬連本番（`quinella_bets_enabled=false`） |
| ❌ | 複勝本番（`place_bets_enabled=false`） |

---

## 3. 本番設定

```json
{
  "online_phase": "phase1_5",
  "wide_bets_enabled": true,
  "quinella_bets_enabled": false,
  "place_bets_enabled": false,
  "wide_min_edge": 0.05,
  "wide_top_n": 2
}
```

**確率**: Harville（正規化勝率）× O3 最低オッズ  
**サイジング**: Kelly 0.08（単勝と同一 `max_invest_per_race` で合算キャップ）

---

## 4. 合格基準（OOF 2025）

| 指標 | 合格 |
|------|------|
| ROI（wide anchor bet） | ≥ 115% |
| n_bets | ≥ 200 |
| 的中率 | 参考（目標 20%+） |

MDD は win+wide 合算ポートフォリオで Phase C1-6 以降に評価。

---

## 5. 検証コマンド

```bash
python model_training/scripts/run_wide_production_gates.py
python main/tests/e2e_test.py
```

---

## 6. ロールバック

`strategy_config.json` で以下に戻す:

```json
"online_phase": "phase1",
"wide_bets_enabled": false
```

単勝推奨のみ継続（Track A）。
