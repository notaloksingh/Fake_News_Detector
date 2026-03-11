import os
import pandas as pd
import sqlite3
from train_model import train_model


def _load_main():
    # prefer canonical raw news.csv, otherwise fall back to processed combined_v2 or base_clean
    candidates = [
        "data/news.csv",
        "data/processed/combined_v2.csv",
        "data/processed/base_clean.csv",
        "data/processed/combined.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"Loading main dataset from {p}")
            df = pd.read_csv(p, engine="python", on_bad_lines="skip", encoding="utf-8")
            df.columns = df.columns.str.lower().str.strip()
            print(f"Raw read -> shape: {df.shape}, columns: {df.columns.tolist()}")
            # build content if needed
            if "content" not in df.columns:
                title = None
                for c in ["title", "headline", "head"]:
                    if c in df.columns:
                        title = c
                        break
                text = None
                for c in ["text", "article", "body"]:
                    if c in df.columns:
                        text = c
                        break
                if title and text:
                    df["content"] = df[title].fillna("") + " " + df[text].fillna("")
                elif title:
                    df["content"] = df[title].fillna("")
                elif text:
                    df["content"] = df[text].fillna("")
            # normalize label column
            label_col = None
            for c in ["label", "truth", "class", "category", "subject"]:
                if c in df.columns:
                    label_col = c
                    break
            if label_col:
                # robust mapping: numeric coercion first (handles 0.0/1.0), then textual mapping
                numeric = pd.to_numeric(df[label_col], errors="coerce")
                mapped = pd.Series(index=df.index, dtype="float64")
                num_mask = numeric.isin([0, 1])
                if num_mask.any():
                    mapped.loc[num_mask] = numeric.loc[num_mask]
                # textual mapping for remaining rows
                text_mask = ~num_mask
                if text_mask.any():
                    txt = df.loc[text_mask, label_col].astype(str).str.lower().str.strip()
                    txt_mapped = txt.map({"fake": 0, "false": 0, "real": 1, "true": 1, "0": 0, "1": 1})
                    # heuristics for tokens
                    def _heur(s):
                        s = str(s)
                        if "fake" in s or "false" in s:
                            return 0
                        if "real" in s or "true" in s:
                            return 1
                        return None
                    heur = txt.apply(_heur)
                    # prefer exact map, otherwise heuristic
                    txt_mapped.fillna(heur, inplace=True)
                    mapped.loc[text_mask] = txt_mapped.values
                df["label"] = mapped
                print("Label value counts after mapping:")
                try:
                    print(df["label"].value_counts(dropna=False).to_dict())
                except Exception:
                    print("Could not print label counts")
            # keep only content/label
            if "content" in df.columns and "label" in df.columns:
                out = df[["content", "label"]].dropna()
                print(f"Loaded {len(out)} rows from {p}")
                return out
            else:
                print(f"File {p} does not contain usable content/label columns; skipping.")
    raise SystemExit("No suitable main dataset found. Please provide data/news.csv or data/processed/combined_v2.csv")


def _load_new():
    db_path = "news.db"
    if not os.path.exists(db_path):
        print("No news.db found — continuing without new verified data.")
        return None
    try:
        conn = sqlite3.connect(db_path)
        df_new = pd.read_sql("SELECT content, label FROM news", conn)
        df_new.columns = df_new.columns.str.lower().str.strip()
        df_new = df_new[["content", "label"]].dropna()
        print(f"Loaded {len(df_new)} rows from news.db")
        return df_new
    except Exception as e:
        print("Failed to load news.db:", e)
        return None


df_main = _load_main()
df_new = _load_new()

if df_new is not None:
    df_combined = pd.concat([df_main, df_new], ignore_index=True)
else:
    df_combined = df_main

print(f"Total rows for retraining: {len(df_combined)}")

# Diagnostics: ensure we have enough data to train
if len(df_combined) < 20:
    print("Not enough rows to retrain (need >=20). Aborting. Rows available:", len(df_combined))
    # show a sample of available data
    print("Sample rows:")
    print(df_combined.head(10).to_dict())
    raise SystemExit(1)

# Retrain
train_model(df_combined)
