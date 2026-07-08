"""ロードマップ実験の合格判定・ベースライン比較・レポート生成。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "docs" / "experiments"
MODELS_DIR = ROOT / "model_training" / "models"

# H-1 リリースゲート（Binary）
F2_SHARPE_PASS = 0.10
F2_SHARPE_WARN = 0.05
F2_ROI_PASS = 1.05
F1_ROI_DEGRADATION_PP = 0.05
F3_N_BETS_PROVISIONAL = 200
FEATURE_SHARPE_DELTA = 0.03


@dataclass
class FoldMetrics:
    fold: int
    roi: float
    mdd: float
    sharpe: float
    n_bets: int
    hit_rate: float

    @classmethod
    def from_result(cls, r: dict) -> FoldMetrics:
        tm = r.get("test_metrics") or {}
        dm = r.get("drawdown_metrics") or {}
        return cls(
            fold=int(r.get("fold", 0)),
            roi=float(tm.get("roi", 0)),
            mdd=float(dm.get("max_drawdown_rate", 0)),
            sharpe=float(dm.get("sharpe_ratio", 0)),
            n_bets=int(tm.get("n_bets", 0)),
            hit_rate=float(tm.get("hit_rate", 0)),
        )


@dataclass
class EvalVerdict:
    experiment_id: str
    adopted: bool
    verdict: str  # PASS_FULL | PASS_PARTIAL | FAIL | SKIP
    reason: str
    folds: list[FoldMetrics] = field(default_factory=list)
    vs_baseline: dict[str, Any] = field(default_factory=dict)
    next_action: str = ""

    def to_markdown(self, change_desc: str, baseline_name: str) -> str:
        lines = [
            f"# 実験: {self.experiment_id}",
            "",
            f"- 実施日: {date.today().isoformat()}",
            f"- 変更内容: {change_desc}",
            f"- ベースライン: {baseline_name}",
            f"- 判定: **{self.verdict}** — {self.reason}",
            "",
            "## 結果",
            "",
            "| Fold | ROI | MDD | Sharpe | n_bets | hit_rate |",
            "|------|-----|-----|--------|--------|----------|",
        ]
        for f in self.folds:
            lines.append(
                f"| F{f.fold} | {f.roi:.1%} | {f.mdd:.1%} | {f.sharpe:.3f} | {f.n_bets} | {f.hit_rate:.1%} |"
            )
        if self.vs_baseline:
            lines.extend(
                [
                    "",
                    "## ベースライン比",
                    "",
                    "| Fold | ROI diff | Sharpe diff | n_bets diff |",
                    "|------|----------|-------------|-------------|",
                ]
            )
            for k, v in self.vs_baseline.items():
                lines.append(
                    f"| {k} | {v.get('roi_diff_pp', 0):+.1f}pp | "
                    f"{v.get('sharpe_diff', 0):+.3f} | {v.get('n_bets_diff', 0):+d} |"
                )
        lines.extend(
            [
                "",
                "## 考察",
                "",
                self.reason,
                "",
                "## 次のアクション",
                "",
                self.next_action or "—",
                "",
                f"- [{'x' if self.adopted else ' '}] 採用",
                f"- [{'x' if not self.adopted else ' '}] 棄却",
            ]
        )
        return "\n".join(lines) + "\n"


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_folds(results: list[dict]) -> list[FoldMetrics]:
    return [FoldMetrics.from_result(r) for r in sorted(results, key=lambda x: x.get("fold", 0))]


def compare_folds(exp: list[FoldMetrics], base: list[FoldMetrics]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    base_map = {f.fold: f for f in base}
    for f in exp:
        b = base_map.get(f.fold)
        if not b:
            continue
        key = f"F{f.fold}"
        out[key] = {
            "roi_diff_pp": (f.roi - b.roi) * 100,
            "sharpe_diff": f.sharpe - b.sharpe,
            "n_bets_diff": f.n_bets - b.n_bets,
        }
    return out


def evaluate_experiment(
    experiment_id: str,
    results: list[dict],
    baseline_results: list[dict],
    *,
    is_feature_experiment: bool = True,
) -> EvalVerdict:
    folds = extract_folds(results)
    base_folds = extract_folds(baseline_results)
    vs = compare_folds(folds, base_folds)

    f1 = next((f for f in folds if f.fold == 1), None)
    f2 = next((f for f in folds if f.fold == 2), None)
    f3 = next((f for f in folds if f.fold == 3), None)
    b1 = next((f for f in base_folds if f.fold == 1), None)
    b2 = next((f for f in base_folds if f.fold == 2), None)

    if not f2:
        return EvalVerdict(
            experiment_id=experiment_id,
            adopted=False,
            verdict="FAIL",
            reason="F2 結果なし",
            folds=folds,
            vs_baseline=vs,
            next_action="データ・モデル整合性を確認して再実行",
        )

    reasons: list[str] = []
    f2_sharpe_ok = f2.sharpe >= F2_SHARPE_PASS
    f2_roi_ok = f2.roi >= F2_ROI_PASS
    f1_ok = True
    if f1 and b1:
        f1_ok = f1.roi >= b1.roi - F1_ROI_DEGRADATION_PP
        if not f1_ok:
            reasons.append(f"F1 ROI 非劣化違反 ({f1.roi:.1%} vs baseline {b1.roi:.1%})")

    f3_ok = True
    if f3:
        f3_ok = f3.n_bets >= F3_N_BETS_PROVISIONAL
        if not f3_ok:
            reasons.append(f"F3 n_bets={f3.n_bets} < {F3_N_BETS_PROVISIONAL}（暫定）")

    sharpe_delta = f2.sharpe - (b2.sharpe if b2 else 0)
    partial_feature = is_feature_experiment and sharpe_delta >= FEATURE_SHARPE_DELTA and f1_ok

    if f2_sharpe_ok and f2_roi_ok and f1_ok:
        verdict = "PASS_FULL"
        adopted = True
        reason = (
            f"F2 Sharpe={f2.sharpe:.3f}>={F2_SHARPE_PASS}, ROI={f2.roi:.1%}, "
            f"Sharpe改善={sharpe_delta:+.3f}"
        )
        next_action = "チャンピオンとして採用。次実験のベースラインを更新"
    elif partial_feature or (f2.sharpe >= F2_SHARPE_WARN and sharpe_delta > 0 and f1_ok):
        verdict = "PASS_PARTIAL"
        adopted = False
        reason = (
            f"F2 Sharpe={f2.sharpe:.3f}（目標{F2_SHARPE_PASS}未達だが baseline 比 {sharpe_delta:+.3f}）。"
            + ("; ".join(reasons) if reasons else "部分改善")
        )
        next_action = "棄却して次候補へ。ベスト候補として記録"
    else:
        verdict = "FAIL"
        adopted = False
        fail_parts = [
            f"F2 Sharpe={f2.sharpe:.3f}",
            f"ROI={f2.roi:.1%}",
            f"ΔSharpe={sharpe_delta:+.3f}",
        ]
        reason = "不合格: " + ", ".join(fail_parts)
        if reasons:
            reason += "; " + "; ".join(reasons)
        next_action = "モデル復元。計画キューの次ステップへ"

    return EvalVerdict(
        experiment_id=experiment_id,
        adopted=adopted,
        verdict=verdict,
        reason=reason,
        folds=folds,
        vs_baseline=vs,
        next_action=next_action,
    )


def save_verdict_report(
    verdict: EvalVerdict,
    change_desc: str,
    baseline_name: str,
    results_path: Path,
) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EXPERIMENTS_DIR / f"{date.today().isoformat()}-{verdict.experiment_id}.md"
    report_path.write_text(
        verdict.to_markdown(change_desc, baseline_name),
        encoding="utf-8",
    )
    summary_path = MODELS_DIR / f"roadmap_state.json"
    state: dict = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            state = json.load(f)
    state.setdefault("experiments", []).append(
        {
            "id": verdict.experiment_id,
            "verdict": verdict.verdict,
            "adopted": verdict.adopted,
            "f2_sharpe": next((f.sharpe for f in verdict.folds if f.fold == 2), None),
            "results_file": str(results_path.name),
            "report": str(report_path.name),
        }
    )
    summary_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def evaluate_phase3_experiment(
    experiment_id: str,
    results: list[dict],
    baseline_results: list[dict],
) -> EvalVerdict:
    """Phase 3: F3 Sharpe 改善を主ゲート、F2 非劣化を維持ゲートとする。"""
    verdict = evaluate_experiment(
        experiment_id, results, baseline_results, is_feature_experiment=False
    )
    f2 = next((f for f in verdict.folds if f.fold == 2), None)
    f3 = next((f for f in verdict.folds if f.fold == 3), None)
    b2 = next((f for f in extract_folds(baseline_results) if f.fold == 2), None)
    b3 = next((f for f in extract_folds(baseline_results) if f.fold == 3), None)
    if not f3 or not f2 or not b2 or not b3:
        return verdict

    f3_pass = f3.sharpe >= F2_SHARPE_PASS
    f2_maintain = f2.sharpe >= b2.sharpe - 0.02
    f3_improve = f3.sharpe >= b3.sharpe + 0.02

    if f3_pass and f2_maintain and f2.sharpe >= F2_SHARPE_PASS:
        verdict.verdict = "PASS_FULL"
        verdict.adopted = True
        verdict.reason = (
            f"F3 Sharpe={f3.sharpe:.3f}>={F2_SHARPE_PASS}, "
            f"F2 Sharpe={f2.sharpe:.3f} maintained, F3 delta={f3.sharpe - b3.sharpe:+.3f}"
        )
        verdict.next_action = "Phase3 adopt. Proceed to Rank eval"
    elif f3_improve and f2_maintain:
        verdict.verdict = "PASS_PARTIAL"
        verdict.adopted = False
        verdict.reason = (
            f"F3 Sharpe {b3.sharpe:.3f}->{f3.sharpe:.3f} (+{f3.sharpe - b3.sharpe:.3f}) "
            f"but F3 target {F2_SHARPE_PASS} not reached"
        )
    else:
        verdict.verdict = "FAIL"
        verdict.adopted = False
        verdict.reason = (
            f"F3 Sharpe={f3.sharpe:.3f}, F2 Sharpe={f2.sharpe:.3f} "
            f"(baseline F3={b3.sharpe:.3f})"
        )
    return verdict
