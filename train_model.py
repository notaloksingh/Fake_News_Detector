import pandas as pd
import re
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def train_model(df):
    df["content"] = df["content"].apply(normalize)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_vec))
    print("Accuracy:", acc)

    # Save artifacts expected by app
    joblib.dump(model, "model/model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

if __name__ == "__main__":
    df = pd.read_csv ("data/processed/combined_v2.csv",
    engine="python",
    on_bad_lines="skip",
    encoding="utf-8"
    )

    df.columns = df.columns.str.lower().str.strip()

    # find likely columns
    def _find(col_names):
        for c in col_names:
            if c in df.columns:
                return c
        return None

    title_col = _find(["title", "headline", "head"])
    text_col = _find(["text", "article", "content", "body"])
    label_col = _find(["label", "truth", "class", "category", "subject"])

    if not (title_col or text_col):
        print("No title/text column found. Columns:", df.columns.tolist())
        raise SystemExit(1)
    if not label_col:
        print("No label column found. Columns:", df.columns.tolist())
        raise SystemExit(1)

    # build content from available columns
    if title_col and text_col:
        df["content"] = df[title_col].fillna("") + " " + df[text_col].fillna("")
    elif title_col:
        df["content"] = df[title_col].fillna("")
    else:
        df["content"] = df[text_col].fillna("")

    # many files have extra tab-separated tokens like "2016\tFAKE" in label column
    df["label_raw"] = df[label_col].astype(str).str.split("\t").str[-1]

    # Attempt robust mapping: first coerce numeric-like strings (e.g. '0', '1', '0.0'),
    # then map textual labels, then apply heuristics for remaining unmapped values.
    label_map = {"fake": 0, "false": 0, "0": 0, "real": 1, "true": 1, "1": 1}

    # try numeric coercion first (handles '0.0', '1.0', etc.)
    numeric_vals = pd.to_numeric(df["label_raw"].str.strip(), errors="coerce")
    mapped = pd.Series(index=df.index, dtype="float64")
    num_mask = numeric_vals.isin([0, 1])
    if num_mask.any():
        mapped.loc[num_mask] = numeric_vals.loc[num_mask]

    # For the remaining rows, map textual tokens
    remaining_mask = ~num_mask
    if remaining_mask.any():
        mapped.loc[remaining_mask] = (
            df.loc[remaining_mask, "label_raw"].str.lower().str.strip().map(label_map)
        )

    # simple heuristics for still-unmapped labels (return np.nan for unknowns)
    def _heuristic(s):
        s = str(s).lower()
        if "fake" in s or "false" in s or "fraud" in s:
            return 0
        if "real" in s or "true" in s or "true story" in s:
            return 1
        return np.nan

    mask = mapped.isna()
    if mask.any():
        heur = df.loc[mask, "label_raw"].apply(_heuristic)
        mapped.loc[mask] = heur.values

    # coerce to numeric (ints) and keep NaN for unknowns
    df["label"] = pd.to_numeric(mapped, errors="coerce")

    before = len(df)
    df.dropna(subset=["content", "label"], inplace=True)
    after = len(df)

    if after == 0:
        print("No usable rows after processing.\nColumns:", df.columns.tolist())
        print("Sample raw label values (top 20):")
        print(df["label_raw"].astype(str).value_counts().head(20).to_dict())
        raise SystemExit(1)

    if before != after:
        print(f"Dropped {before-after} rows with missing content/label. Using {after} rows.")

    train_model(df)
