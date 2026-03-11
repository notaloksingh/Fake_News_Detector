import pandas as pd
import glob
import os

base = pd.read_csv("data/processed/base_clean.csv")

# gather isot files: prefer data/raw/isot.csv, otherwise any isot_*.csv
raw_single = "data/raw/isot.csv"
isot_paths = []
if os.path.exists(raw_single):
    isot_paths = [raw_single]
else:
    isot_paths = sorted(glob.glob("data/raw/isot*.csv"))

if not isot_paths:
    raise SystemExit("No ISOT files found in data/raw — place isot.csv or isot_*.csv there.")

parts = []
for p in isot_paths:
    df_i = pd.read_csv(p, engine="python", on_bad_lines="skip", encoding="utf-8")
    df_i.columns = df_i.columns.str.lower().str.strip()
    # if label column missing, infer from filename (isot_true / isot_fake)
    if "label" not in df_i.columns:
        fname = os.path.basename(p).lower()
        if "_true" in fname or "_real" in fname:
            df_i["label"] = "real"
        elif "_fake" in fname:
            df_i["label"] = "fake"
    # build content and normalize
    df_i["content"] = df_i.get("title", "").fillna("") + " " + df_i.get("text", df_i.get("article", "")).fillna("")
    parts.append(df_i)

isot = pd.concat(parts, ignore_index=True)
isot["label"] = isot["label"].astype(str).str.lower().map({"fake": 0, "real": 1, "true": 1, "false": 0})
isot = isot[["content", "label"]].dropna()

combined = pd.concat([base, isot], ignore_index=True).drop_duplicates()

combined.to_csv("data/processed/combined_v2.csv", index=False)

print("Combined dataset size:", len(combined))
