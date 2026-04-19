# Fake News Detection System

A simple machine learning project for classifying news as `REAL` or `FAKE` using live Times of India data.

## Tech Stack

- Python
- Scikit-learn
- NLTK
- Streamlit
- SQLite

## What This Project Includes

- a small local labeled dataset
- live Times of India data from RSS
- text cleaning and stemming
- TF-IDF feature extraction
- Logistic Regression model
- Streamlit app for prediction
- SQLite storage for fetched articles

## Folder Structure

```text
Fake News Detection System/
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   ├── fake_news_dataset.csv
│   ├── live_news_dataset.csv
│   └── fake_news.db
├── models/
│   ├── best_model.pkl
│   ├── model_metadata.json
│   └── training_metrics.json
├── src/
│   └── fake_news_detector/
│       ├── __init__.py
│       ├── config.py
│       ├── database.py
│       ├── dataset.py
│       ├── live_data.py
│       ├── modeling.py
│       └── preprocessing.py
└── tests/
    └── test_pipeline.py
```

## Simple Flow

1. Fetch live Times of India articles
2. Clean the text
3. Convert text into TF-IDF features
4. Train Logistic Regression
5. Predict in Streamlit

## Setup

```bash
pip3 install -r requirements.txt
```

## Train

Train with current local and live data:

```bash
python3 train.py
```

Refresh Times of India live data and train:

```bash
python3 train.py --refresh-live-data --live-limit 25
```

## Predict

```bash
python3 predict.py --title "Government approves education grant" --text "Officials announced funding support for rural schools."
```

## Run the App

```bash
python3 -m streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Notes

- The local CSV provides both `REAL` and `FAKE` examples.
- The live Times of India feed provides recent `REAL` examples.
- This is a simple educational project, not a production fact-checking system.
