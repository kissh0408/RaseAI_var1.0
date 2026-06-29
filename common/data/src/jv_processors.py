import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    from .jv_log import jv_err, jv_info, jv_section, jv_verbose, jv_warn
except ImportError:
    from jv_log import jv_err, jv_info, jv_section, jv_verbose, jv_warn


@dataclass
class ProcessorContext:
    client_cls: object
    tqdm_fn: object
    parse_fixed_width_fn: object
    schemas: dict
    extract_record_key_fn: object
    load_existing_dates_fn: object
    save_to_csv_fn: object
    get_schema_fieldnames_fn: object
    load_race_year_merge_map_fn: object
    race_stream_try_merge_fn: object
    race_kubuns: object


class BaseProcessor(ABC):
    def __init__(self, context: ProcessorContext):
        self.ctx = context

    @abstractmethod
    def run_yearly(self, **_kwargs):
        raise NotImplementedError


class RaceProcessor(BaseProcessor):

    def _run_single_year(
        self,
        *,
        year,
        task,
        output_dir,
        start_date_str,
        end_date_str,
        start_year,
        end_year,
    ):
        year_start = f"{year}0101000000"
        year_end = f"{year}1231235959"
        if year == start_year:
            year_start = max(year_start, start_date_str)
        if year == end_year:
            year_end = min(year_end, end_date_str)

        client = self.ctx.client_cls()
        try:
            client.login()
        except Exception:
            jv_warn("Login failed - skip this task.")
            try:
                client.close()
            except Exception:
                pass
            return

        ds = task["dataspec"]
        opt = task["option"]
        targets = set(task["target_ids"])
        task_start_date = year_start

        jv_section(
            f"{ds} (year={year})",
            f"JVOpen {task_start_date} .. {year_end}  opt={opt}",
        )
        time.sleep(2)

        existing_data_keys = {}
        ra_merge_map = {}
        se_merge_map = {}
        for target_id in targets:
            subdir = f"race_{target_id.lower()}"
            fname = f"race_{target_id.lower()}_{year}.csv"

            subdir_path = os.path.join(output_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)

            fpath = os.path.join(subdir_path, fname)
            if target_id == "RA":
                ra_merge_map = self.ctx.load_race_year_merge_map_fn(fpath, "RA")
            elif target_id == "SE":
                se_merge_map = self.ctx.load_race_year_merge_map_fn(fpath, "SE")
            else:
                existing_keys = self.ctx.load_existing_dates_fn(fpath, target_id)
                existing_data_keys[target_id] = existing_keys

        ex_parts: list[str] = []
        for tid in sorted(targets):
            if tid == "RA" and ra_merge_map:
                ex_parts.append(f"RA={len(ra_merge_map)}")
            elif tid == "SE" and se_merge_map:
                ex_parts.append(f"SE={len(se_merge_map)}")
            elif tid not in ("RA", "SE"):
                ek = existing_data_keys.get(tid) or set()
                if ek:
                    ex_parts.append(f"{tid}={len(ek)}")
        if ex_parts:
            jv_info("  existing keys: " + "  ".join(ex_parts))

        records_buffer = {}
        total_count = 0
        skipped_count = 0
        chunk_count = 0
        pbar = self.ctx.tqdm_fn(
            desc=f"Fetching {ds} ({year})",
            unit="chunks",
            position=0,
            leave=True,
        )

        try:
            for raw_chunk in client.get_data(ds, task_start_date, opt, year_end):
                chunk_count += 1
                pbar.update(1)
                if isinstance(raw_chunk, bytes):
                    try:
                        raw_chunk = raw_chunk.decode("cp932", "replace")
                    except Exception:
                        continue

                if not raw_chunk:
                    continue

                for line in raw_chunk.split("\n"):
                    line = line.rstrip("\r\n")
                    if not line or len(line) < 2:
                        continue
                    rec_id = line[:2]
                    if targets and rec_id not in targets:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    if len(line_bytes) < 2:
                        continue

                    if rec_id not in self.ctx.schemas:
                        continue
                    parsed = self.ctx.parse_fixed_width_fn(
                        line_bytes, self.ctx.schemas[rec_id]
                    )

                    if "year" in parsed:
                        try:
                            rec_year = int(parsed["year"])
                            if rec_year != year:
                                skipped_count += 1
                                continue
                        except Exception:
                            continue

                    if rec_id in {"RA", "SE"}:
                        parsed["raw_hex"] = line_bytes.hex()
                        merge = ra_merge_map if rec_id == "RA" else se_merge_map
                        if self.ctx.race_stream_try_merge_fn(
                            merge,
                            parsed,
                            rec_id,
                            self.ctx.race_kubuns,
                        ):
                            total_count += 1
                        else:
                            skipped_count += 1
                        continue

                    if rec_id == "HR":
                        existing_keys = existing_data_keys.get("HR", set())
                        record_key = self.ctx.extract_record_key_fn(parsed, "HR")
                        if not record_key or record_key in existing_keys:
                            skipped_count += 1
                            continue
                        parsed["raw_hex"] = line_bytes.hex()
                        records_buffer.setdefault("HR", []).append(parsed)
                        existing_keys.add(record_key)
                        total_count += 1
                        hr_buf = records_buffer["HR"]
                        if len(hr_buf) >= 100000:
                            subdir_path = os.path.join(output_dir, "race_hr")
                            os.makedirs(subdir_path, exist_ok=True)
                            save_path = os.path.join(subdir_path, f"race_hr_{year}.csv")
                            fields = self.ctx.get_schema_fieldnames_fn("HR") + ["raw_hex"]
                            self.ctx.save_to_csv_fn(hr_buf, save_path, fields)
                            records_buffer["HR"] = []
                        continue

                    skipped_count += 1

                if chunk_count % 10 == 0:
                    pbar.set_postfix({"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"})
        except Exception as e:
            jv_err(f"{ds} ({year}): {e}")
        finally:
            pbar.close()
            try:
                client.close()
            except Exception:
                pass

        if ra_merge_map:
            subdir_path = os.path.join(output_dir, "race_ra")
            os.makedirs(subdir_path, exist_ok=True)
            save_path = os.path.join(subdir_path, f"race_ra_{year}.csv")
            rows = [t[1] for t in ra_merge_map.values()]
            fields = self.ctx.get_schema_fieldnames_fn("RA") + ["raw_hex"]
            self.ctx.save_to_csv_fn(rows, save_path, fields, append=False)
        if se_merge_map:
            subdir_path = os.path.join(output_dir, "race_se")
            os.makedirs(subdir_path, exist_ok=True)
            save_path = os.path.join(subdir_path, f"race_se_{year}.csv")
            rows = [t[1] for t in se_merge_map.values()]
            fields = self.ctx.get_schema_fieldnames_fn("SE") + ["raw_hex"]
            self.ctx.save_to_csv_fn(rows, save_path, fields, append=False)
        for rid, data_list in records_buffer.items():
            if rid == "HR" and data_list:
                subdir_path = os.path.join(output_dir, "race_hr")
                os.makedirs(subdir_path, exist_ok=True)
                save_path = os.path.join(subdir_path, f"race_hr_{year}.csv")
                fields = self.ctx.get_schema_fieldnames_fn("HR") + ["raw_hex"]
                self.ctx.save_to_csv_fn(data_list, save_path, fields)

        if total_count == 0:
            if skipped_count > 0:
                jv_info(
                    f"  no new rows for {ds} ({year}); skipped={skipped_count:,} (already in file)"
                )
            else:
                jv_info(f"  no stream data for {ds} ({year})")
        else:
            jv_info(
                f"  +{total_count:,} new, skipped {skipped_count:,} for {ds} ({year})"
            )

    def run_yearly(
        self,
        *,
        task,
        output_dir,
        start_date_str,
        end_date_str,
        start_year,
        end_year,
    ):
        for year in task["years"]:
            self._run_single_year(
                year=year,
                task=task,
                output_dir=output_dir,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                start_year=start_year,
                end_year=end_year,
            )


class TrainingProcessor(BaseProcessor):
    def _paths(self, ds: str, rec_id: str, year: int):
        if ds == "SLOP":
            subdir = f"slop_{rec_id.lower()}"
            fname = f"slop_{rec_id.lower()}_{year}.csv"
        elif ds == "WOOD":
            subdir = f"wood_{rec_id.lower()}"
            fname = f"wood_{rec_id.lower()}_{year}.csv"
        else:
            subdir = f"{ds.lower()}_{rec_id.lower()}"
            fname = f"{ds.lower()}_{rec_id.lower()}_{year}.csv"
        return subdir, fname

    def _run_single_year(
        self,
        *,
        year,
        task,
        output_dir,
        start_date_str,
        end_date_str,
        start_year,
        end_year,
    ):
        ds = task["dataspec"]
        opt = task["option"]
        targets = set(task["target_ids"])

        year_start = f"{year}0101000000"
        year_end = f"{year}1231235959"
        if year == start_year:
            year_start = max(year_start, start_date_str)
        if year == end_year:
            year_end = min(year_end, end_date_str)

        task_start_date = year_start
        if ds == "WOOD" and year == 2021:
            task_start_date = max("20210727000000", year_start)

        client = self.ctx.client_cls()
        try:
            client.login()
        except Exception:
            jv_warn("Login failed - skip this task.")
            try:
                client.close()
            except Exception:
                pass
            return

        jv_section(
            f"{ds} (year={year})",
            f"JVOpen {task_start_date} .. {year_end}  opt={opt}",
        )
        time.sleep(2)

        existing_data_keys = {}
        for target_id in targets:
            subdir, fname = self._paths(ds, target_id, year)
            subdir_path = os.path.join(output_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            fpath = os.path.join(subdir_path, fname)
            existing_keys = self.ctx.load_existing_dates_fn(fpath, target_id)
            existing_data_keys[target_id] = existing_keys

        ex_parts = [
            f"{tid}={len(existing_data_keys[tid])}"
            for tid in sorted(targets)
            if existing_data_keys.get(tid)
        ]
        if ex_parts:
            jv_info("  existing keys: " + "  ".join(ex_parts))

        records_buffer = {}
        total_count = 0
        skipped_count = 0
        chunk_count = 0

        pbar = self.ctx.tqdm_fn(
            desc=f"Fetching {ds} ({year})",
            unit="chunks",
            position=0,
            leave=True,
        )

        try:
            for raw_chunk in client.get_data(ds, task_start_date, opt, year_end):
                chunk_count += 1
                pbar.update(1)
                if isinstance(raw_chunk, bytes):
                    try:
                        raw_chunk = raw_chunk.decode("cp932", "replace")
                    except Exception:
                        continue
                if not raw_chunk:
                    continue

                for line in raw_chunk.split("\n"):
                    line = line.rstrip("\r\n")
                    if not line or len(line) < 2:
                        continue
                    rec_id = line[:2]
                    if targets and rec_id not in targets:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    if len(line_bytes) < 2 or rec_id not in self.ctx.schemas:
                        continue
                    parsed = self.ctx.parse_fixed_width_fn(
                        line_bytes, self.ctx.schemas[rec_id]
                    )

                    training_date = str(parsed.get("training_date", "")).strip()
                    if len(training_date) != 8 or not training_date.isdigit():
                        skipped_count += 1
                        continue
                    training_year = int(training_date[:4])
                    if training_year != year:
                        skipped_count += 1
                        continue
                    if ds == "WOOD" and training_year == 2021 and training_date[4:8] < "0727":
                        skipped_count += 1
                        continue

                    existing_keys = existing_data_keys.get(rec_id, set())
                    record_key = (
                        str(parsed.get("ketto_num", "")),
                        training_date,
                    )
                    if record_key in existing_keys:
                        skipped_count += 1
                        continue

                    parsed["raw_hex"] = line_bytes.hex()
                    records_buffer.setdefault(rec_id, []).append(parsed)
                    existing_keys.add(record_key)
                    total_count += 1

                    if len(records_buffer[rec_id]) >= 100000:
                        subdir, fname = self._paths(ds, rec_id, year)
                        subdir_path = os.path.join(output_dir, subdir)
                        os.makedirs(subdir_path, exist_ok=True)
                        save_path = os.path.join(subdir_path, fname)
                        fields = self.ctx.get_schema_fieldnames_fn(rec_id) + ["raw_hex"]
                        self.ctx.save_to_csv_fn(records_buffer[rec_id], save_path, fields)
                        records_buffer[rec_id] = []

                if chunk_count % 10 == 0:
                    pbar.set_postfix({"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"})
        except Exception as e:
            jv_err(f"{ds} ({year}): {e}")
        finally:
            pbar.close()
            try:
                client.close()
            except Exception:
                pass

        for rid, data_list in records_buffer.items():
            if data_list:
                subdir, fname = self._paths(ds, rid, year)
                subdir_path = os.path.join(output_dir, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                save_path = os.path.join(subdir_path, fname)
                fields = self.ctx.get_schema_fieldnames_fn(rid) + ["raw_hex"]
                self.ctx.save_to_csv_fn(data_list, save_path, fields)

        if total_count == 0:
            if skipped_count > 0:
                jv_info(
                    f"  no new rows for {ds} ({year}); skipped={skipped_count:,} (already in file)"
                )
            else:
                jv_info(f"  no stream data for {ds} ({year})")
        else:
            jv_info(
                f"  +{total_count:,} new, skipped {skipped_count:,} for {ds} ({year})"
            )

    def run_yearly(
        self,
        *,
        task,
        output_dir,
        start_date_str,
        end_date_str,
        start_year,
        end_year,
    ):
        for year in task["years"]:
            self._run_single_year(
                year=year,
                task=task,
                output_dir=output_dir,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                start_year=start_year,
                end_year=end_year,
            )


class FullPeriodProcessor(BaseProcessor):
    def _paths(self, ds: str, rec_id: str):
        if ds == "BLDN":
            return f"blod_{rec_id.lower()}", f"blod_{rec_id.lower()}.csv"
        if ds == "MING":
            return f"ming_{rec_id.lower()}", f"ming_{rec_id.lower()}.csv"
        return f"{ds.lower()}_{rec_id.lower()}", f"{ds.lower()}_{rec_id.lower()}.csv"

    def run_task(self, *, task, output_dir, start_date_str, end_date_str, end_year):
        ds = task["dataspec"]
        opt = task["option"]
        targets = set(task["target_ids"])
        task_start_date = max(task["start_date"], start_date_str)

        try:
            client = self.ctx.client_cls()
        except Exception as e:
            jv_err(f"JV-Link init failed for {ds}: {e}")
            jv_warn("Skipping this task and all subsequent tasks.")
            raise
        try:
            client.login()
        except Exception:
            jv_warn("Login failed - skip this task.")
            try:
                client.close()
            except Exception:
                pass
            return

        jv_section(
            ds,
            f"JVOpen {task_start_date} .. {end_date_str}  opt={opt}",
        )
        time.sleep(2)

        existing_data_keys = {}
        for target_id in targets:
            subdir, fname = self._paths(ds, target_id)
            subdir_path = os.path.join(output_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            fpath = os.path.join(subdir_path, fname)
            existing_keys = self.ctx.load_existing_dates_fn(fpath, target_id)
            existing_data_keys[target_id] = existing_keys

        ex_parts = [
            f"{tid}={len(existing_data_keys[tid])}"
            for tid in sorted(targets)
            if existing_data_keys.get(tid)
        ]
        if ex_parts:
            jv_info("  existing keys: " + "  ".join(ex_parts))

        records_buffer = {}
        total_count = 0
        skipped_count = 0
        chunk_count = 0

        pbar = self.ctx.tqdm_fn(desc=f"Fetching {ds}", unit="chunks", position=0, leave=True)

        try:
            for raw_chunk in client.get_data(ds, task_start_date, opt, end_date_str):
                chunk_count += 1
                pbar.update(1)
                if isinstance(raw_chunk, bytes):
                    try:
                        raw_chunk = raw_chunk.decode("cp932", "replace")
                    except Exception:
                        continue
                if not raw_chunk:
                    continue

                for line in raw_chunk.split("\n"):
                    line = line.rstrip("\r\n")
                    if not line or len(line) < 2:
                        continue
                    rec_id = line[:2]
                    if targets and rec_id not in targets:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    if len(line_bytes) < 2 or rec_id not in self.ctx.schemas:
                        continue

                    parsed = self.ctx.parse_fixed_width_fn(line_bytes, self.ctx.schemas[rec_id])

                    is_master_like = rec_id in {"HN", "SK", "BT", "HC", "WC", "UM"}
                    if not is_master_like and ds not in ["BLDN", "MING"]:
                        if "year" in parsed:
                            try:
                                rec_year = int(parsed["year"])
                                if rec_year > end_year:
                                    continue
                            except Exception:
                                pass

                    existing_keys = existing_data_keys.get(rec_id, set())
                    record_key = self.ctx.extract_record_key_fn(parsed, rec_id)
                    if record_key is not None and record_key in existing_keys:
                        skipped_count += 1
                        continue

                    parsed["raw_hex"] = line_bytes.hex()
                    records_buffer.setdefault(rec_id, []).append(parsed)
                    if record_key:
                        existing_keys.add(record_key)
                    total_count += 1

                    if len(records_buffer[rec_id]) >= 100000:
                        subdir, fname = self._paths(ds, rec_id)
                        subdir_path = os.path.join(output_dir, subdir)
                        os.makedirs(subdir_path, exist_ok=True)
                        save_path = os.path.join(subdir_path, fname)
                        fields = self.ctx.get_schema_fieldnames_fn(rec_id) + ["raw_hex"]
                        self.ctx.save_to_csv_fn(records_buffer[rec_id], save_path, fields)
                        records_buffer[rec_id] = []

                if chunk_count % 10 == 0:
                    pbar.set_postfix({"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"})
        except Exception as e:
            jv_err(f"{ds}: {e}")
        finally:
            pbar.close()
            try:
                client.close()
            except Exception:
                pass

        for rid, data_list in records_buffer.items():
            if data_list:
                subdir, fname = self._paths(ds, rid)
                subdir_path = os.path.join(output_dir, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                save_path = os.path.join(subdir_path, fname)
                fields = self.ctx.get_schema_fieldnames_fn(rid) + ["raw_hex"]
                self.ctx.save_to_csv_fn(data_list, save_path, fields)
            else:
                if rid in targets:
                    jv_warn(
                        f"{rid}: stream seen but no rows buffered for save (check filters / JVOpen)"
                    )

        for target_id in targets:
            count = len(records_buffer.get(target_id, []))
            if count == 0 and target_id not in records_buffer:
                jv_warn(f"{target_id}: no records processed (JVOpen skip or empty stream)")
            elif count > 0:
                jv_verbose(f"{target_id}: {count} rows in buffer before save")

        if total_count == 0:
            if skipped_count > 0:
                jv_info(
                    f"  no new rows for {ds}; skipped={skipped_count:,} (already in file)"
                )
            else:
                jv_info(f"  no stream data for {ds}")
        else:
            jv_info(
                f"  +{total_count:,} new, skipped {skipped_count:,} for {ds}"
            )

    def run_yearly(self, **_kwargs):
        # Full-period processors are task-oriented (single span),
        # so this is intentionally unsupported.
        raise NotImplementedError("FullPeriodProcessor uses run_task(), not run_yearly().")


PROCESSOR_MAP = {
    "BLDN": FullPeriodProcessor,
    "MING": FullPeriodProcessor,
    "RACE": RaceProcessor,
    "SLOP": TrainingProcessor,
    "WOOD": TrainingProcessor,
}


def get_processor_class(dataspec: str):
    proc = PROCESSOR_MAP.get(dataspec)
    if proc is None:
        raise KeyError(f"No processor registered for dataspec: {dataspec}")
    return proc
