import pandas as pd

# =========================
# Load dataset (try to auto-detect delimiter)
# =========================
csv_path = "data/base/kaggle_base.csv"
try:
    df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="skip", encoding="utf-8")
except Exception:
    # fallback to default comma separator
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", encoding="utf-8")

df.columns = df.columns.str.lower().str.strip()
print("Detected columns:", df.columns.tolist())
# Print a few rows for diagnosis if columns look suspicious
preview_cols = [c for c in ["title", "text", "subject", "label"] if c in df.columns]
if len(preview_cols) > 0:
    print("Sample rows (first 5) for:", preview_cols)
    print(df[preview_cols].head(5).to_string())

# =========================
# Build content
# =========================
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# =========================
# CREATE LABELS FROM SUBJECT (THE REAL FIX)
# =========================
REAL_SUBJECTS = {
    "politicsnews", "worldnews", "news", "middle-east",
    "us_news", "europe", "global"
}

FAKE_SUBJECTS = {
    "politics", "left-news", "government-news", "usnews"
}

def subject_to_label(s):
    s = str(s).lower().replace(" ", "")
    if s in REAL_SUBJECTS:
        return 1
    if s in FAKE_SUBJECTS:
        return 0
    return None

# If an existing numeric 0/1 `label` column exists, prefer it instead of overwriting
use_subject_mapping = True
if "label" in df.columns:
    # try to coerce existing labels to numeric 0/1
    df["_label_orig"] = pd.to_numeric(df["label"], errors="coerce")
    valid_count = df["_label_orig"].isin([0, 1]).sum()
    if valid_count > 0:
        print(f"Found existing 'label' column with {valid_count} valid 0/1 entries. Keeping existing labels.")
        df["label"] = df["_label_orig"]
        use_subject_mapping = False

if use_subject_mapping:
    df["label"] = df["subject"].apply(subject_to_label)
    # Print unmapped subject examples to help extend mapping
    unmapped = df[df["label"].isnull()]["subject"].value_counts().head(20)
    if len(unmapped) > 0:
        print("Unmapped subject examples (top 20):")
        print(unmapped)

# =========================
# Final cleanup
# =========================
df = df[["content", "label"]]
df.dropna(inplace=True)

print("Final usable rows:", len(df))
print("Label distribution:")
print(df["label"].value_counts())

# =========================
# Save cleaned data
# =========================
df.to_csv("data/processed/base_clean.csv", index=False)

# If nothing was produced, try a robust tab-split fallback parser
if len(df) == 0:
    print("No usable rows after initial parse — trying tab-split fallback parser...")
    import io

    def parse_tab_fallback(path):
        out = {"title": [], "text": [], "subject": [], "date": [], "label": []}
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i == 0:
                    # skip header if it contains known column names
                    hdr = line.strip().lower()
                    if any(k in hdr for k in ["title", "text", "subject", "label"]):
                        continue
                # split from the right: expect last fields to be subject, date, label
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    # try comma fallback
                    parts = line.rstrip("\n").split(",")
                if len(parts) >= 5:
                    title = parts[0]
                    label = parts[-1]
                    date = parts[-2]
                    subject = parts[-3]
                    text = "\t".join(parts[1:-3])
                    out["title"].append(title)
                    out["text"].append(text)
                    out["subject"].append(subject)
                    out["date"].append(date)
                    out["label"].append(label)
        return pd.DataFrame(out)

    raw_df = parse_tab_fallback(csv_path)
    if raw_df.empty:
        print("Fallback parser produced no rows — please inspect the raw CSV file formatting.")
    else:
        print("Fallback parser produced rows:", len(raw_df))
        # rebuild content and try mapping again
        raw_df["content"] = raw_df["title"].fillna("") + " " + raw_df["text"].fillna("")
        # try to coerce existing label values like '2017\tFAKE' -> extract FAKE/TRUE or the final word
        def normalize_label(v):
            if pd.isna(v):
                return None
            s = str(v).strip()
            # common patterns: '2017\tFAKE' or '2017FAKE' or 'FAKE' or 'TRUE'
            if s.upper().endswith("FAKE"):
                return 0
            if s.upper().endswith("TRUE") or s.upper().endswith("REAL"):
                return 1
            if s in ("0", "1"):
                return int(s)
            return None

        raw_df["label"] = raw_df["label"].apply(normalize_label)
        # if still many missing, try mapping from subject
        if raw_df["label"].isnull().all():
            raw_df["label"] = raw_df["subject"].apply(subject_to_label)
        cleaned = raw_df[["content", "label"]].dropna()
        print("Fallback cleaned usable rows:", len(cleaned))
        print("Fallback label distribution:")
        print(cleaned["label"].value_counts())
        cleaned.to_csv("data/processed/base_clean.csv", index=False)