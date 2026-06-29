import csv
import hashlib
import os
import sqlite3

try:
    from .jv_log import jv_saved, jv_verbose, jv_warn
except ImportError:
    from jv_log import jv_saved, jv_verbose, jv_warn
from pathlib import Path
from typing import Iterable


def _usecols_for_rec_id(rec_id: str):
    if rec_id == "SE":
        return [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
            "horse_num",
        ]
    if rec_id in {"RA", "HR", "DM", "TM", "WE", "AV", "TC", "CC", "O2", "O3"}:
        return ["year", "month_day", "course_code", "kai", "nichi", "race_num"]
    if rec_id in {"WH", "JC"}:
        return [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
            "horse_num",
        ]
    if rec_id in {"HN", "SK"}:
        return ["ketto_num"]
    if rec_id == "BT":
        return ["breeding_reg_num"]
    if rec_id in {"HC", "WC"}:
        return ["ketto_num", "training_date"]
    return None


def _normalize_key(values: Iterable[str]):
    vals = [str(v or "").strip() for v in values]
    if len(vals) == 1:
        return vals[0]
    return tuple(vals)


def _extract_keys_from_csv(filepath: str, rec_id: str):
    keys = set()
    usecols = _usecols_for_rec_id(rec_id)
    if not os.path.isfile(filepath) or os.path.getsize(filepath) <= 1024:
        return keys
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return keys
        if not usecols:
            for row in reader:
                key = tuple(str(row.get(k, "")).strip() for k in sorted(row.keys()))
                if any(key):
                    keys.add(key)
            return keys
        for row in reader:
            raw = [row.get(c, "") for c in usecols]
            norm = _normalize_key(raw)
            if isinstance(norm, tuple):
                if any(norm):
                    keys.add(norm)
            elif norm:
                keys.add(norm)
    return keys


class SQLiteKeyIndex:
    def __init__(self, output_file: str):
        csv_path = Path(output_file).resolve()
        output_root = csv_path.parent.parent if csv_path.parent.parent.exists() else csv_path.parent
        state_dir = output_root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = state_dir / "jv_key_index.sqlite3"
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS file_meta (
                file_path TEXT PRIMARY KEY,
                rec_id TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS keys_index (
                file_path TEXT NOT NULL,
                rec_id TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                key_text TEXT NOT NULL,
                PRIMARY KEY (file_path, rec_id, key_hash)
            )
            """
        )
        self.conn.commit()

    def _serialize_key(self, key) -> str:
        if isinstance(key, tuple):
            return "|".join(str(v) for v in key)
        return str(key)

    def _hash_key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _stat(self, file_path: str):
        st = os.stat(file_path)
        return st.st_mtime_ns, st.st_size

    def is_fresh(self, file_path: str, rec_id: str) -> bool:
        if not os.path.exists(file_path):
            return False
        mtime_ns, size_bytes = self._stat(file_path)
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT mtime_ns, size_bytes FROM file_meta WHERE file_path = ? AND rec_id = ?",
            (file_path, rec_id),
        ).fetchone()
        return bool(row and row[0] == mtime_ns and row[1] == size_bytes)

    def load_keys(self, file_path: str, rec_id: str):
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT key_text FROM keys_index WHERE file_path = ? AND rec_id = ?",
            (file_path, rec_id),
        ).fetchall()
        keys = set()
        for (key_text,) in rows:
            if "|" in key_text and rec_id not in {"HN", "SK", "BT"}:
                keys.add(tuple(key_text.split("|")))
            else:
                keys.add(key_text)
        return keys

    def refresh_keys(self, file_path: str, rec_id: str, keys):
        if not os.path.exists(file_path):
            return
        mtime_ns, size_bytes = self._stat(file_path)
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM keys_index WHERE file_path = ? AND rec_id = ?",
            (file_path, rec_id),
        )
        payload = []
        for key in keys:
            text = self._serialize_key(key)
            payload.append((file_path, rec_id, self._hash_key(text), text))
        if payload:
            cur.executemany(
                """
                INSERT OR REPLACE INTO keys_index (file_path, rec_id, key_hash, key_text)
                VALUES (?, ?, ?, ?)
                """,
                payload,
            )
        cur.execute(
            """
            INSERT INTO file_meta(file_path, rec_id, mtime_ns, size_bytes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                rec_id=excluded.rec_id,
                mtime_ns=excluded.mtime_ns,
                size_bytes=excluded.size_bytes,
                updated_at=CURRENT_TIMESTAMP
            """,
            (file_path, rec_id, mtime_ns, size_bytes),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


def _load_existing_dates_without_pandas(filepath, rec_id):
    return _extract_keys_from_csv(filepath, rec_id)


def load_existing_dates(filepath, rec_id):
    if not os.path.exists(filepath) or os.path.getsize(filepath) <= 1024:
        return set()
    idx = SQLiteKeyIndex(filepath)
    try:
        if idx.is_fresh(filepath, rec_id):
            return idx.load_keys(filepath, rec_id)
        keys = _extract_keys_from_csv(filepath, rec_id)
        idx.refresh_keys(filepath, rec_id, keys)
        return keys
    except Exception as e:
        jv_warn(f"SQLite key index failed for {filepath}: {e}")
        return _extract_keys_from_csv(filepath, rec_id)
    finally:
        idx.close()


def save_to_csv(data_list, filepath, fieldnames, *, append: bool = True):
    if not data_list:
        jv_verbose(f"save_to_csv: no rows for {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)

    mode = "a" if append else "w"
    if not append:
        file_exists = False
    with open(filepath, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_list)
    jv_saved(filepath, len(data_list))
