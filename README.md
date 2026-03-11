"# Fake_News_Detector" 
📰 Fake News Detection System
An AI-powered Fake News Detection Web Application that classifies news articles as REAL or FAKE using Natural Language Processing (NLP) and Machine Learning.
The system integrates a Flask web interface, live news API, automated retraining pipeline, and database storage to continuously improve the model.
🚀 Features
🧠 Machine Learning based Fake News Detection
📊 Confidence score visualization
🌐 Live news integration using NewsAPI
🗄 Database storage of checked news articles
🔄 Automated model retraining pipeline
🌙 Dark mode UI
🧪 Automated testing dataset
📈 Continuous learning from real user data

🛠 Technologies Used

Programming
Python
Machine Learning
Scikit-learn
Logistic Regression
TF-IDF Vectorization
Natural Language Processing (NLP)
Web Development
Flask
HTML
CSS
JavaScript
Data & Automation
SQLite Database
Joblib (model serialization)
NewsAPI (live news)
Windows Task Scheduler (automated retraining)

🧠 How It Works

User enters a news article in the web interface.
Text is cleaned and normalized using NLP preprocessing.
The trained ML model converts text to TF-IDF vectors.
Logistic Regression predicts whether the news is REAL or FAKE.
A confidence score is displayed to the user.
The article and prediction are stored in a database.
Stored data is exported and used for future model retraining.

📂 Project Structure
fake-news-detection
│
├── app.py
├── train_model.py
├── retrain_model.py
├── run_tests.py
├── prepare_base_data.py
├── export_live_data.py
│
├── model
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── templates
│   └── index.html
│
├── static
│   └── style.css
│
├── data
│   ├── base
│   ├── processed
│   ├── live
│   └── tests
│
├── news.db
└── README.md

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
2️⃣ Create Virtual Environment
python -m venv venv
Activate it:
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running the Application
Start the Flask server:
python app.py
Open in browser:
http://127.0.0.1:5000

🧪 Run Automated Tests

python run_tests.py
This evaluates the model using a predefined test dataset.

🔄 Retraining the Model

Manual retraining:
python retrain_model.py
The project also supports automatic retraining using Windows Task Scheduler.

📊 Example Prediction

Input News:
Scientists confirm that drinking boiled banana peels cures cancer in 30 days.
Prediction
FAKE NEWS
Confidence: 92%

🧪 Dataset

The model is trained on a combined dataset of:
Kaggle Fake & Real News Dataset
Additional curated datasets
Live user-submitted news articles

📈 Future Improvements

Transformer models (BERT
Source credibility scoring
Explainable AI (word importance visualization)
Admin verification dashboard
Cloud deployment

👨‍💻 Author
Alok Singh.

⭐ If you like this project

Give it a ⭐ on GitHub to support the work.
