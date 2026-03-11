import pandas as pd

base = pd.read_csv("data/processed/combined_v2.csv")
live = pd.read_csv("data/live/verified_live.csv")

final = pd.concat([base, live]).drop_duplicates()

final.to_csv("data/processed/final_training_data.csv", index=False)

print("Final dataset size:", len(final))
