import pandas as pd, datetime
cp = pd.read_parquet(r'C:\Users\syugo\AI\RaceAI_var3.0\simulator\cache\llm_cache_checkpoint.parquet')
done = cp['race_id'].nunique()
total = 11442
remaining = total - done
speed = 4.98
eta_min = remaining * speed / 60
eta_time = datetime.datetime.now() + datetime.timedelta(minutes=eta_min)
print("done:", done, "/", total, f"({done/total*100:.1f}%)")
print("remaining:", remaining, "races")
print("ETA:", round(eta_min), "min ->", eta_time.strftime("%H:%M"))
