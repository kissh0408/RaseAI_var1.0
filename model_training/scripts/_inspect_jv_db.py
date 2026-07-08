import sqlite3
from pathlib import Path

db = Path("common/data/JVData.db")
c = sqlite3.connect(db)
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("tables:", tables)
for t in tables[:5]:
    cols = [r[1] for r in c.execute(f"PRAGMA table_info({t})").fetchall()]
    n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: n={n}, cols={cols[:8]}...")
c.close()
