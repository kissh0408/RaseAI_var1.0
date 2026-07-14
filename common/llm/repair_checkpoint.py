"""チェックポイントの重複除去 + 整合性確認。"""
import shutil
from pathlib import Path
import pandas as pd

CHECKPOINT = Path(r"C:\Users\syugo\AI\RaceAI_var3.0\simulator\cache\llm_cache_checkpoint.parquet")
BACKUP     = Path(r"C:\Users\syugo\AI\RaceAI_var3.0\simulator\cache\llm_cache_checkpoint_bak.parquet")

# バックアップ
shutil.copy2(CHECKPOINT, BACKUP)
print(f"backup -> {BACKUP}")

df = pd.read_parquet(CHECKPOINT)
print(f"before: {len(df):,} rows / {df['race_id'].nunique():,} unique races")

# 重複除去: race_id × horse_num の最初の出現を残す
df_clean = df.drop_duplicates(subset=["race_id", "horse_num"], keep="first")
print(f"after : {len(df_clean):,} rows / {df_clean['race_id'].nunique():,} unique races")
print(f"removed: {len(df) - len(df_clean):,} dup rows")

# ev_score sum=1 の確認
race_sum = df_clean.groupby("race_id")["llm_ev_score"].sum()
bad = (race_sum - 1.0).abs() > 0.01
print(f"ev_score sum != 1.0 (bad races): {bad.sum()}")

# 保存
df_clean.to_parquet(CHECKPOINT, index=False)
print(f"saved -> {CHECKPOINT}")

# 最終確認
df_check = pd.read_parquet(CHECKPOINT)
dup_remaining = df_check.duplicated(subset=["race_id", "horse_num"]).sum()
print(f"dup remaining after repair: {dup_remaining}")
print(f"unique races: {df_check['race_id'].nunique():,}")
