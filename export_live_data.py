import sqlite3
import pandas as pd

conn = sqlite3.connect("news.db")

df = pd.read_sql("SELECT content, label FROM news", conn)

df.to_csv("data/live/verified_live.csv", index=False)

print("Exported live rows:", len(df))
