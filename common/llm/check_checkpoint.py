import pandas as pd
import numpy as np

df = pd.read_parquet(r'C:\Users\syugo\AI\RaceAI_var3.0\simulator\cache\llm_cache_checkpoint.parquet')

print("=== 基本情報 ===")
print("rows:", len(df))
print("unique races:", df["race_id"].nunique())

# horse_num の重複チェック
dup = df.duplicated(subset=["race_id", "horse_num"])
print("\n=== race_id x horse_num 重複 ===")
print("重複行数:", dup.sum())
if dup.sum() > 0:
    print("サンプル:")
    print(df[dup].head(10))

# horse_num の実値範囲チェック（JRA最大18頭）
print("\n=== horse_num 実値範囲 ===")
print("max horse_num:", df["horse_num"].max())
print("horse_num > 18 の行数:", (df["horse_num"] > 18).sum())
if (df["horse_num"] > 18).sum() > 0:
    print("サンプル:")
    print(df[df["horse_num"] > 18].head(10))

# horses_per_race の正確な分布（実馬頭数ではなく行数）
horses_per_race = df.groupby("race_id")["horse_num"].count()
print("\n=== horses_per_race 分布（18超は異常） ===")
print(horses_per_race[horses_per_race > 18].describe())
print("18超のレース数:", (horses_per_race > 18).sum())

# 18超レースの race_id サンプル
bad_races = horses_per_race[horses_per_race > 18].head(5).index.tolist()
for rid in bad_races:
    sub = df[df["race_id"] == rid]
    print(f"\n  race_id={rid}: {len(sub)}行")
    print(sub[["horse_num","llm_ev_score","llm_rank_score"]].to_string())
