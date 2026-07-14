# main/archive/ — 隔離済みモジュール

**隔離日**: 2026-07-10（損失最小化運用実装＋L4パイプライン復旧、`docs/specs/2026-07-10-loss-minimization-implementation-spec.md` §3.3）

## 到達性の判定方法

実運用エントリポイントを `main/unified_pipeline.py::run_unified_today`（CLI含む）と
`main/notebook_bootstrap.py` の `__all__` 公開関数の2つと定義し、そこからの import 連鎖
（関数内の遅延importを含む）を `grep -rn "import"` で追跡した。到達しないモジュールをここに
隔離した（削除ではない。git 履歴で追跡可能・将来の参照可能性のため）。

## 隔離したファイルと理由

- **`main/main.py`**
  実運用パス（unified_pipeline / notebook_bootstrap）から一切importされていない。
  L320等で存在しない `strategy.src.betting_framework` を遅延importしており、
  そもそも実行不能（`strategy/` パッケージ自体がリポジトリに存在しない）。
  var2.0.0系の市場残差ロジック（本プロジェクトでは禁止のinit_score等）を前提にした設計。

- **`main/pipeline/data_pipeline.py`**
  `main.main` からのみ参照。L68で `from main.jv_subprocess import run_with_32bit_python`
  を実行しており、`main.jv_subprocess` はアーカイブ済みで存在しない
  （正: `common.data.src.jv_subprocess`）。main.py 隔離に伴い道連れで隔離。

- **`main/pipeline/strategy_pipeline.py`**
  `main.main` からのみ参照。L337/L563/L607/L768 で存在しない `strategy.src.*` を遅延import。

- **`main/pipeline/view_pipeline.py`**
  `main.main` からのみ参照。単体では import 可能（`common.data.src.jv_subprocess` は
  実在する）が、呼び出し元（main.main）が到達不能なため道連れで隔離。

- **`main/pipeline/monthly_dd_tracker.py`**
  `main.main` からのみ参照（本番向け月次P&Lファイルトラッカー）。
  損失最小化運用の月次MDD確認は `betting/src/derive_flat_fraction.py`（VALID凍結用）と
  `betting/src/risk_limits.py::RiskLimits`（運用ガード）で別途担う。
  `main/tests/e2e_test.py::test_monthly_dd_tracker` は隔離に合わせて skip 化した。

## 隔離を見送ったもの（再確認の結果、存置が必要と判明）

- **`main/pipeline/inference_pipeline.py`**, **`main/pipeline/baba_scenario.py`**
  当初は「`main.main` からのみ参照」として隔離候補に挙げられていたが、実装時の到達性
  再確認で `tests/test_inference_baba_sync.py`（トップレベル import）と
  `main/tests/e2e_test.py`（Test 13/14）が `apply_uniform_baba_jv_code` を直接
  テストしていることが判明した。両モジュール自体は壊れたimportを含まず
  （`inference_pipeline.py` は `main.pipeline.baba_scenario` に委譲するだけの薄いラッパー）、
  実運用パス外ではあるが「存在しないモジュールを参照」の条件を満たさないため、
  判断基準表の隔離条件（存在しないモジュール参照 かつ 実運用パス外）に該当しない。
  退行（既存テストの collection error）を避けるため `main/pipeline/` に存置した。

## 存置したもの（誤って隔離しないよう明記）

- **`main/pipeline/export_utils.py`** — `main/unified_pipeline.py` L200
  （`export_unified_by_venue_baba`）から到達するため存置必須。
- **`main/pipeline/inference_pipeline.py`**, **`main/pipeline/baba_scenario.py`** — 上記参照。

## 復帰条件

- `strategy/src/*` パッケージが本リポジトリに実装され、かつ本プロジェクトの
  「市場情報境界」「ROIで合否判定しない」等の憲法条項に適合すると evaluator が
  確認した場合にのみ、個別モジュール単位で `main/pipeline/` へ復帰を検討する。
- それまでは `main/unified_pipeline.py`（L1→L2→L3の当日パイプライン、
  §3.2 でE2E検証済み）が唯一の実運用エントリポイントである。
