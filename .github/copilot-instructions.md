# Copilot instructions for fake-news-detection

Purpose
- Help an AI coding agent become immediately productive in this repository.

Big picture
- This is a single-process Flask web app that serves a pre-trained scikit-learn model. Training and serving are separated:
  - `train_model.py` reads `data/news.csv`, trains a `TfidfVectorizer` + `LogisticRegression`, and writes artifacts to `model/model.pkl` and `model/vectorizer.pkl`.
  - `app.py` loads those artifacts and renders predictions via `templates/index.html`.

Key files and what they show
- `train_model.py`: canonical data-loading, label-mapping rules, cleaning, vectorizer and model hyperparameters (see `TfidfVectorizer(stop_words='english', max_df=0.7)` and `LogisticRegression(max_iter=1000)`).
- `app.py`: Flask app that expects `model/model.pkl` and `model/vectorizer.pkl`, returns `prediction` and `confidence` to the template.
- `templates/index.html`: UI that expects `prediction`, `confidence` and references `bar_class` in a progress bar (note: `app.py` currently does not supply `bar_class`).
- `data/news.csv`: canonical dataset location. `train_model.py` accepts several alternate column names (title/headline, text/article/content, label/truth/class).
- `requirements.txt`: pinned Python packages and versions to use in the environment.

Developer workflows (concrete commands)
- Create a venv and install deps:
  - `python -m venv .venv` (Windows: `python -m venv .venv`)
  - Activate the venv (PowerShell): `.\.venv\Scripts\Activate` (Cmd.exe: `\.venv\Scripts\activate.bat`)
  - After activation your prompt will look like `(.venv) PS C:\Users\alok\OneDrive\Pictures\Desktop\fake-news-detection>` and then run: `pip install -r requirements.txt`
- Train model (produces `model/` artifacts):
  - `python train_model.py`
- Run app locally (Flask debug enabled in `app.py`):
  - `python app.py` → open http://127.0.0.1:5000

Project-specific conventions and gotchas
- Model artifacts live in `model/` and are loaded by relative paths in `app.py`. Agents modifying paths must keep these runtime locations synced.
- `train_model.py` normalizes column names and maps many textual label variants into 0 (fake) / 1 (real). If label mapping yields many unmapped values, training will stop with diagnostics printed to console. Use the printed examples to adjust dataset values.
- Text cleaning is aggressive: `clean_text()` lowercases and strips non a-z characters. Searching for English words only.
- Template mismatch: `templates/index.html` uses `bar_class` to style the progress bar, but `app.py` only passes `prediction` and `confidence`. Fix options:
  - add `bar_class` in `app.py` (e.g., `'bar-success'` / `'bar-danger'` based on predicted class), or
  - remove `bar_class` reference from the template.

Integration points and debugging notes
- If `app.py` fails with a joblib load error, re-run `python train_model.py` to recreate `model/model.pkl` and `model/vectorizer.pkl`.
- Version compatibility: `joblib` / `scikit-learn` versions in `requirements.txt` matter for loading model artifacts created in other environments.
- Data problems: `train_model.py` prints examples of raw labels and the top unmapped values — use these logs to update `data/news.csv` or extend `map_label_value()`.

What an agent should do first (recommended tasks)
- Verify `model/` contains `model.pkl` and `vectorizer.pkl`. If missing, run `python train_model.py`.
- Add a brief health endpoint (e.g., `/health`) or unit tests to validate model loading.
- Fix the `bar_class` mismatch between `app.py` and `templates/index.html`.

Where to look for more context
- Serve logic: `app.py`
- Training & preprocessing: `train_model.py`
- UI: `templates/index.html` and `static/style.css`
- Dataset: `data/news.csv`
- Dependencies: `requirements.txt`

Questions for the repo owner
- Do you want `bar_class` logic added to `app.py` or should the template be simplified?
- Are there target deployment constraints (Docker, cloud provider) and CI workflows to document?

If changes are made to training or vectorizer config, retrain and ensure `model/` is updated before deploying the server.
