"""
JRA クッション値・含水率 Excel を CSV にクリーニングする。

出力列は race_ra / preprocessing と同じキー体系:
  year, month_day, race_date, course_code, kai,
  surface_code, measure_point_code, moisture_pct, cushion_value

使い方:
  python common/data/src/clean_cushion_data.py
  python common/data/src/clean_cushion_data.py --input-dir common/data/output/cushion
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# JV-Link / preprocessing と同一の競馬場コード
COURSE_NAME_TO_CODE: dict[str, int] = {
    "札幌": 1,
    "函館": 2,
    "福島": 3,
    "新潟": 4,
    "東京": 5,
    "中山": 6,
    "中京": 7,
    "京都": 8,
    "阪神": 9,
    "小倉": 10,
}

# 2025年以降 Excel のコース列が A/B/C/D になる形式判定用（出力には含めない）
NEW_FORMAT_COURSE_LABELS: frozenset[str] = frozenset({"A", "A1", "A2", "B", "C", "D", "E"})

SURFACE_NAME_TO_CODE: dict[str, int] = {"芝": 1, "ダート": 2}
MEASURE_POINT_NAME_TO_CODE: dict[str, int] = {
    "ゴール前": 1,
    "4コーナー": 2,
}

OUTPUT_COLUMNS: tuple[str, ...] = (
    "year",
    "month_day",
    "race_date",
    "course_code",
    "kai",
    "surface_code",
    "measure_point_code",
    "moisture_pct",
    "cushion_value",
)

GROUP_KEYS: tuple[str, ...] = (
    "year",
    "month_day",
    "race_date",
    "course_code",
    "kai",
    "surface_code",
    "measure_point_code",
)

_VENUE_NOISE_PREFIX = re.compile(r"^(?:Roll|epic)+", re.IGNORECASE)
_KAI_PATTERN = re.compile(r"(\d+)")
_COMBINED_LOCATION = re.compile(r"^(芝|ダート)[・･](.+)$")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return text.replace(" ", "")


def _normalize_venue_name(value: object) -> str:
    text = _normalize_text(value)
    text = _VENUE_NOISE_PREFIX.sub("", text)
    return text


def _parse_kai(value: object) -> int | None:
    text = _normalize_text(value)
    match = _KAI_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(1))


def _parse_measure_point(location: str) -> int | None:
    normalized = _normalize_text(location)
    return MEASURE_POINT_NAME_TO_CODE.get(normalized)


def _parse_surface(surface: str) -> int | None:
    normalized = _normalize_text(surface)
    return SURFACE_NAME_TO_CODE.get(normalized)


def _parse_combined_location(location: object) -> tuple[int | None, int | None]:
    text = _normalize_text(location)
    match = _COMBINED_LOCATION.match(text)
    if not match:
        return None, None
    surface_code = _parse_surface(match.group(1))
    measure_code = _parse_measure_point(match.group(2))
    return surface_code, measure_code


def _is_new_format(df: pd.DataFrame) -> bool:
    course_values = {_normalize_text(v) for v in df["コース"].dropna().unique()}
    location_values = {_normalize_text(v) for v in df["場所"].dropna().unique()}
    has_course_letters = course_values.issubset(NEW_FORMAT_COURSE_LABELS)
    has_combined_locations = any("・" in v or "･" in v for v in location_values)
    return has_course_letters and has_combined_locations


def _read_raw_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    expected = {"競馬場", "開催", "コース", "場所", "日付", "含水率"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: 必須列が不足しています: {sorted(missing)}")
    if "クッション値" not in df.columns:
        df["クッション値"] = pd.NA
    return df


def _build_output_frame(raw: pd.DataFrame, file_year: int) -> pd.DataFrame:
    dt = pd.to_datetime(raw["日付"], errors="coerce")
    new_format = _is_new_format(raw)

    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        venue = _normalize_venue_name(row["競馬場"])
        course_code = COURSE_NAME_TO_CODE.get(venue)
        kai = _parse_kai(row["開催"])
        race_ts = dt.iloc[idx] if isinstance(idx, int) else dt.loc[idx]

        if course_code is None:
            continue
        if kai is None or pd.isna(race_ts):
            continue
        if int(race_ts.year) != file_year:
            continue

        course_text = _normalize_text(row["コース"])
        location_text = _normalize_text(row["場所"])

        if new_format:
            surface_code, measure_point_code = _parse_combined_location(location_text)
        else:
            surface_code = _parse_surface(course_text)
            measure_point_code = _parse_measure_point(location_text)

        if surface_code is None or measure_point_code is None:
            continue

        moisture = pd.to_numeric(row["含水率"], errors="coerce")
        cushion = pd.to_numeric(row["クッション値"], errors="coerce")
        if pd.notna(cushion) and surface_code != 1:
            cushion = pd.NA

        rows.append(
            {
                "year": int(race_ts.year),
                "month_day": int(race_ts.strftime("%m%d")),
                "race_date": int(race_ts.strftime("%Y%m%d")),
                "course_code": course_code,
                "kai": kai,
                "surface_code": surface_code,
                "measure_point_code": measure_point_code,
                "moisture_pct": moisture,
                "cushion_value": cushion,
            }
        )

    out = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    if out.empty:
        return out

    for col in GROUP_KEYS:
        out[col] = out[col].astype("Int64")

    # 2025+ は A/B/C/D コースが同一キーに複数行あるため平均化（course_kubun は使わない）
    out = (
        out.groupby(list(GROUP_KEYS), as_index=False)
        .agg(
            moisture_pct=("moisture_pct", "mean"),
            cushion_value=("cushion_value", "mean"),
        )
        .sort_values(list(GROUP_KEYS))
        .reset_index(drop=True)
    )
    return out


def clean_cushion_file(path: Path, output_dir: Path | None = None) -> pd.DataFrame:
    file_year = int(path.stem.split("_")[-1])
    raw = _read_raw_excel(path)
    cleaned = _build_output_frame(raw, file_year=file_year)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"cushion_{file_year}.csv"
        cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")

    return cleaned


def clean_all_cushion_files(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    write_combined: bool = True,
) -> pd.DataFrame:
    if output_dir is None:
        output_dir = input_dir

    frames: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob("cushion_*.xlsx")):
        frames.append(clean_cushion_file(path, output_dir=output_dir))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if write_combined and not combined.empty:
        combined.to_csv(output_dir / "cushion_all.csv", index=False, encoding="utf-8-sig")
    return combined


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="クッション値 Excel を統合用 CSV に変換")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "common" / "data" / "output" / "cushion",
        help="cushion_YYYY.xlsx の入力ディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV 出力先（省略時は input-dir と同じ）",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="cushion_all.csv を出力しない",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir

    combined = clean_all_cushion_files(
        input_dir,
        output_dir,
        write_combined=not args.no_combined,
    )
    print(f"入力: {input_dir}")
    print(f"出力: {output_dir}")
    print(f"行数: {len(combined):,}")
    if not combined.empty:
        print(
            "期間:",
            combined["race_date"].min(),
            "-",
            combined["race_date"].max(),
        )
        turf = combined[combined["surface_code"] == 1]
        print(
            "芝クッション値あり:",
            turf["cushion_value"].notna().sum(),
            "/",
            len(turf),
        )


if __name__ == "__main__":
    main()
