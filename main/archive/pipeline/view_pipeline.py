"""Notebook 向け表示・オッズ取得・特徴量チェック（view 系）。

main.main から re-export され、後方互換の公開 API を維持する。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.data.src.jv_subprocess import run_with_32bit_python

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "model_training" / "models"
PREDICTION_OUTPUT_PATH = PROJECT_ROOT / "main" / "results" / "today_predictions_with_bets.csv"
O2_ODDS_PATH = PROJECT_ROOT / "common" / "data" / "output" / "realtime_odds" / "o2_odds.csv"
O3_ODDS_PATH = PROJECT_ROOT / "common" / "data" / "output" / "realtime_odds" / "o3_odds.csv"


def fetch_today_tan_odds() -> object:
    """全レース分の単勝オッズ(O1/0B31)を取得して o1_odds.csv に保存し race_se.csv に反映する。"""
    result = run_with_32bit_python(
        PROJECT_ROOT,
        "from common.data.src.get_data import fetch_odds_0b31_for_main_races, merge_odds_to_main_se; "
        "r = fetch_odds_0b31_for_main_races(); print(r); merge_odds_to_main_se()",
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[ERROR] returncode:", result.returncode)
        print(result.stderr)
    return result


def check_today_tan_odds() -> None:
    """o1_odds.csv の当日分サマリーをコンソールに出力する。"""
    import datetime

    course_names = {
        1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
        6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
    }
    o1_path = PROJECT_ROOT / "common" / "data" / "output" / "realtime_odds" / "o1_odds.csv"
    if not o1_path.exists():
        print("[NG] o1_odds.csv が存在しません")
        return
    df = pd.read_csv(o1_path)
    today_md = int(datetime.date.today().strftime("%m%d"))
    today = df[df["month_day"] == today_md]
    groups = today.groupby(["course_code", "race_num"])
    print(f"[o1 単勝] 今日({today_md})分: {len(today)}件 / {groups.ngroups}レース")
    if groups.ngroups == 0:
        print("  ⚠️  今日分のオッズがありません。fetch_today_tan_odds() を実行してください。")
        return
    summary = (
        today.groupby(["course_code", "race_num"])
        .agg(頭数=("horse_num", "count"), オッズ最小=("odds_raw", "min"), オッズ最大=("odds_raw", "max"))
        .reset_index()
    )
    summary["競馬場"] = summary["course_code"].map(course_names).fillna(summary["course_code"].astype(str))
    summary["オッズ最小"] = (pd.to_numeric(summary["オッズ最小"], errors="coerce") / 10).map("{:.1f}".format)
    summary["オッズ最大"] = (pd.to_numeric(summary["オッズ最大"], errors="coerce") / 10).map("{:.1f}".format)
    print(
        summary[["競馬場", "race_num", "頭数", "オッズ最小", "オッズ最大"]]
        .rename(columns={"race_num": "R"})
        .sort_values(["競馬場", "R"])
        .to_string(index=False)
    )


def fetch_today_pair_odds() -> object:
    """O2(馬連)/O3(ワイド)速報を 0B32/0B33 で取得して realtime_odds/ に保存する。"""
    result = run_with_32bit_python(
        PROJECT_ROOT,
        "from common.data.src.get_data import fetch_pairwide_odds_0b31_for_main_races; "
        "r = fetch_pairwide_odds_0b31_for_main_races(); print(r)",
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[ERROR] returncode:", result.returncode)
        print(result.stderr)
    return result


def create_main_pastfeatures() -> None:
    """v21モデル用の過去特徴量を生成する（main_features_past.parquet を更新）。"""
    from model_training.src.create_pastfeatures import (
        create_main_pastfeatures as _create_main_pastfeatures_impl,
    )

    _create_main_pastfeatures_impl()


def check_model_features() -> None:
    """v21モデルの特徴量充足チェック。不足列を v21 追加分と既存分に分類して表示する。"""
    import pickle

    v21_new = {
        "youshiba_win_rate", "youshiba_top3_rate", "sire_youshiba_win_rate",
        "horse_youshiba_exp_count", "kokai_koban_win_rate",
        "horse_soft_turf_win_rate", "horse_soft_turf_top3_rate",
        "sire_soft_turf_win_rate", "going_soft_exp_count",
        "speed_index_3run_avg", "speed_index_trend", "pace_dist_style_win_rate",
    }
    model_pkl = MODELS_DIR / "lgbm_model_rank1_all_non_leak.pkl"
    with open(model_pkl, "rb") as f:
        m = pickle.load(f)
    feat_names = m.feature_name()
    main_feat_pq = PROJECT_ROOT / "model_training" / "data" / "02_features" / "main_features_past.parquet"
    main_feat_csv = PROJECT_ROOT / "model_training" / "data" / "02_features" / "main_features_past.csv"
    if main_feat_pq.exists():
        df = pd.read_parquet(main_feat_pq)
        src = "parquet"
    elif main_feat_csv.exists():
        df = pd.read_csv(main_feat_csv, nrows=1)
        src = "csv"
    else:
        raise FileNotFoundError("main_features_past が見つかりません")
    missing = [f for f in feat_names if f not in df.columns]
    ok_count = len(feat_names) - len(missing)
    print(f"[{src}] OK features: {ok_count}/{len(feat_names)}")
    missing_v21 = [f for f in missing if f in v21_new]
    missing_other = [f for f in missing if f not in v21_new]
    if missing_other:
        print(f"[WARN] v21以外の不足特徴量 ({len(missing_other)}列): {missing_other}")
    if missing_v21:
        print(f"[INFO] v21追加特徴量はNaN補完で対応 ({len(missing_v21)}列): {missing_v21}")
    if not missing:
        print("全特徴量が揃っています。run_predict_and_recommend_workflow() を実行できます。")
    else:
        print(f"\n[OK] {ok_count}/{len(feat_names)} 特徴量 利用可能。不足分はLightGBMのNaN分岐で自動補完。")
        print("run_predict_and_recommend_workflow() を実行できます。")


def display_pair_odds_view(top_n: int = 3) -> None:
    """
    O2(馬連)/O3(ワイド)オッズと予測上位馬ペアをコンソールに出力する。

    today_predictions_with_bets.parquet と o2/o3_odds.csv を参照する。
    発売期間中のレースのみオッズが取得済みのため「未取得」表示は正常。
    """
    course_names = {
        1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
        6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
    }
    pred_path = PREDICTION_OUTPUT_PATH.with_suffix(".parquet")
    if not pred_path.exists():
        print(
            "[NG] today_predictions_with_bets.parquet が見つかりません。"
            "先に run_predict_and_recommend_workflow() を実行してください。"
        )
        return

    pred = pd.read_parquet(pred_path)
    today_str = str(pred["race_id"].iloc[0])[:8]
    today_md = int(today_str[4:])

    def _load_odds(path: Path, md: int) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, dtype={"race_id": str})
        return df[df["month_day"] == md].copy()

    o2 = _load_odds(O2_ODDS_PATH, today_md)
    o3 = _load_odds(O3_ODDS_PATH, today_md)
    print(f"[o2 馬連] 今日({today_md})分: {len(o2)}件")
    print(f"[o3 ワイド] 今日({today_md})分: {len(o3)}件")
    if len(o2) == 0:
        print("\n⚠️  馬連・ワイドオッズが未取得です。fetch_today_pair_odds() を実行してください。")

    top = (
        pred.sort_values("win_prob_est", ascending=False)
        .groupby("race_id", sort=False)
        .head(top_n)
        .copy()
    )
    top["rank_in_race"] = (
        top.groupby("race_id")["win_prob_est"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    def _make_pairs(group: pd.DataFrame) -> pd.DataFrame:
        horses = group.sort_values("rank_in_race").head(top_n)
        nums = horses["horse_num"].tolist()
        probs = horses["win_prob_est"].tolist()
        rows = [
            {
                "race_id": group["race_id"].iloc[0],
                "h1": min(nums[i], nums[j]),
                "h2": max(nums[i], nums[j]),
                "prob_sum": probs[i] + probs[j],
                "rank_label": f"{i + 1}位-{j + 1}位",
            }
            for i in range(len(nums))
            for j in range(i + 1, len(nums))
        ]
        return pd.DataFrame(rows)

    pairs = (
        top.groupby("race_id", group_keys=False)
        .apply(_make_pairs)
        .reset_index(drop=True)
    )
    race_info = pred[["race_id", "course_code", "race_num", "month_day"]].drop_duplicates("race_id")
    pairs = pairs.merge(race_info, on="race_id", how="left")
    pairs["競馬場"] = pairs["course_code"].map(course_names).fillna(pairs["course_code"].astype(str))
    pairs["R"] = pairs["race_num"].astype(str) + "R"
    today_pairs = pairs[pairs["month_day"] == today_md]

    def _join_pair_odds(pairs_df: pd.DataFrame, odds_df: pd.DataFrame, col: str) -> pd.DataFrame:
        if odds_df.empty:
            pairs_df = pairs_df.copy()
            pairs_df["odds"] = float("nan")
            return pairs_df
        o = odds_df.copy()
        o["h1"] = o[["horse_num_1", "horse_num_2"]].min(axis=1)
        o["h2"] = o[["horse_num_1", "horse_num_2"]].max(axis=1)
        merged = pairs_df.merge(o[["race_id", "h1", "h2", col]], on=["race_id", "h1", "h2"], how="left")
        merged.rename(columns={col: "odds"}, inplace=True)
        return merged

    pairs_o2 = _join_pair_odds(today_pairs.copy(), o2, "odds_raw")
    pairs_o2["odds"] = pd.to_numeric(pairs_o2["odds"], errors="coerce") / 10

    if not o3.empty:
        o3_work = o3.copy()
        o3_work["h1"] = o3_work[["horse_num_1", "horse_num_2"]].min(axis=1)
        o3_work["h2"] = o3_work[["horse_num_1", "horse_num_2"]].max(axis=1)
        pairs_o3 = today_pairs.merge(
            o3_work[["race_id", "h1", "h2", "odds_min_raw", "odds_max_raw"]],
            on=["race_id", "h1", "h2"],
            how="left",
        )
        pairs_o3["odds_min"] = pd.to_numeric(pairs_o3["odds_min_raw"], errors="coerce") / 10
        pairs_o3["odds_max"] = pd.to_numeric(pairs_o3["odds_max_raw"], errors="coerce") / 10
    else:
        pairs_o3 = today_pairs.copy()
        pairs_o3["odds_min"] = float("nan")
        pairs_o3["odds_max"] = float("nan")

    out_o2 = pairs_o2[["競馬場", "R", "rank_label", "h1", "h2", "prob_sum", "odds"]].copy().rename(
        columns={
            "rank_label": "組合せ",
            "h1": "馬番1",
            "h2": "馬番2",
            "prob_sum": "勝率合計",
            "odds": "馬連オッズ",
        }
    ).sort_values("勝率合計", ascending=False).reset_index(drop=True)
    out_o2["勝率合計"] = out_o2["勝率合計"].map("{:.1%}".format)
    out_o2["馬連オッズ"] = out_o2["馬連オッズ"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "未取得")

    out_o3 = pairs_o3[["競馬場", "R", "rank_label", "h1", "h2", "prob_sum", "odds_min", "odds_max"]].copy().rename(
        columns={
            "rank_label": "組合せ",
            "h1": "馬番1",
            "h2": "馬番2",
            "prob_sum": "勝率合計",
            "odds_min": "ワイド下限",
            "odds_max": "ワイド上限",
        }
    ).sort_values("勝率合計", ascending=False).reset_index(drop=True)
    out_o3["勝率合計"] = out_o3["勝率合計"].map("{:.1%}".format)
    out_o3["ワイド下限"] = out_o3["ワイド下限"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "未取得")
    out_o3["ワイド上限"] = out_o3["ワイド上限"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "未取得")

    out_dir = PREDICTION_OUTPUT_PATH.parent
    o2_csv = out_dir / "today_quinella_view.csv"
    o3_csv = out_dir / "today_wide_view.csv"
    out_o2.to_csv(o2_csv, index=False, encoding="utf-8-sig")
    out_o3.to_csv(o3_csv, index=False, encoding="utf-8-sig")
    print(f"保存: {o2_csv.name} / {o3_csv.name}")

    try:
        from IPython.display import display as _display

        print("\n■ 馬連 買い目（勝率合計降順）")
        _display(out_o2)
        print("\n■ ワイド 買い目（勝率合計降順）")
        _display(out_o3)
    except ImportError:
        print("\n■ 馬連 買い目（勝率合計降順）")
        print(out_o2.to_string(index=False))
        print("\n■ ワイド 買い目（勝率合計降順）")
        print(out_o3.to_string(index=False))
