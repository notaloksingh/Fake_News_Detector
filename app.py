from flask import Flask, render_template, request
import joblib
import requests
import sqlite3
from database import init_db

# =========================
# App setup
# =========================
app = Flask(__name__)

# Initialize database
init_db()

# =========================
# Load trained ML artifacts
# =========================
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# =========================
# News API configuration
# =========================
NEWS_API_KEY = "5f00c860ff7c4baa954f9803bf592ae0"

def get_live_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception:
        return []

    articles = []
    if data.get("status") == "ok":
        for item in data.get("articles", []):
            articles.append({
                "title": item.get("title", ""),
                "description": item.get("description") or ""
            })

    return articles


# =========================
# Save checked news to DB
# =========================
def save_news(text, label):
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO news (content, label) VALUES (?, ?)",
        (text, label)
    )
    conn.commit()
    conn.close()


# =========================
# Main route
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    # Fetch live news once per request
    live_news = get_live_news()

    if request.method == "POST":
        news_text = request.form.get("news", "").strip()

        if news_text:
            # Transform input
            X_input = vectorizer.transform([news_text])

            # Predict
            prediction = model.predict(X_input)[0]
            probabilities = model.predict_proba(X_input)[0]

            confidence = int(max(probabilities) * 100)
            result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

            # Save to database (REAL=1, FAKE=0)
            save_news(news_text, prediction)

    # ✅ RETURN MUST BE INSIDE FUNCTION
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        live_news=live_news
    )


# =========================
# Run app
# =========================
if __name__ == "__main__":
    app.run(debug=True)
