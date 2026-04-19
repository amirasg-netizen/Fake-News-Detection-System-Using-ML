from __future__ import annotations

import json
import os
import random
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from fake_news_detector.config import BASE_DATASET_PATH, BEST_MODEL_PATH, DATABASE_PATH, MODEL_METADATA_PATH
from fake_news_detector.database import load_article_table
from fake_news_detector.dataset import prepare_user_text
from fake_news_detector.modeling import load_model_bundle, predict_with_model


@st.cache_resource
def get_model_bundle() -> dict[str, object]:
    return load_model_bundle(BEST_MODEL_PATH)


@st.cache_data
def get_metadata() -> dict[str, object]:
    if not os.path.exists(MODEL_METADATA_PATH):
        return {}
    with open(MODEL_METADATA_PATH, encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


@st.cache_data
def get_live_articles() -> pd.DataFrame:
    rows = load_article_table(DATABASE_PATH, limit=100)
    return pd.DataFrame(rows)


@st.cache_data
def get_fake_samples() -> pd.DataFrame:
    if not os.path.exists(BASE_DATASET_PATH):
        return pd.DataFrame(columns=["title", "text", "label"])
    dataframe = pd.read_csv(BASE_DATASET_PATH)
    dataframe["label"] = dataframe["label"].astype(str).str.upper().str.strip()
    return dataframe[dataframe["label"] == "FAKE"].reset_index(drop=True)


def set_article_fields(title: str, text: str) -> None:
    st.session_state["headline"] = title
    st.session_state["article_text"] = text


def load_random_real_article() -> None:
    live_articles = get_live_articles()
    if live_articles.empty:
        st.session_state["fetch_message"] = (
            "No live Times of India articles are available yet. Run `python3 train.py --refresh-live-data` first."
        )
        return
    selected = live_articles.sample(1, random_state=random.randint(1, 100000)).iloc[0]
    set_article_fields(
        str(selected.get("title", "")),
        str(selected.get("text", "")),
    )
    st.session_state["fetch_message"] = "Loaded a real Times of India article."


def load_random_fake_article() -> None:
    fake_samples = get_fake_samples()
    if fake_samples.empty:
        st.session_state["fetch_message"] = "No fake samples are available in the local dataset."
        return
    selected = fake_samples.sample(1, random_state=random.randint(1, 100000)).iloc[0]
    set_article_fields(
        str(selected.get("title", "")),
        str(selected.get("text", "")),
    )
    st.session_state["fetch_message"] = "Loaded a fake sample from the dataset."


def main() -> None:
    st.set_page_config(page_title="Fake News Detection System", page_icon="📰", layout="wide")
    if "headline" not in st.session_state:
        st.session_state["headline"] = ""
    if "article_text" not in st.session_state:
        st.session_state["article_text"] = ""
    if "fetch_message" not in st.session_state:
        st.session_state["fetch_message"] = ""

    st.title("Fake News Detection System")
    st.write(
        "A simple fake news detection app using live Times of India data, TF-IDF, and logistic regression."
    )

    metadata = get_metadata()
    sidebar = st.sidebar
    sidebar.header("Training Summary")
    if metadata:
        sidebar.metric("Accuracy", f"{metadata.get('accuracy', 0.0):.2%}")
        sidebar.metric("Precision", f"{metadata.get('precision', 0.0):.2%}")
        sidebar.metric("Recall", f"{metadata.get('recall', 0.0):.2%}")
        sidebar.metric("F1 Score", f"{metadata.get('f1_score', 0.0):.2%}")
        sidebar.caption(f"Model: {metadata.get('model_name', 'N/A')}")
    else:
        sidebar.warning("No trained model metadata found yet. Run `python3 train.py` first.")

    st.subheader("Check a News Article")
    button_cols = st.columns(2)

    with button_cols[0]:
        st.button(
            "Fetch Real News",
            use_container_width=True,
            on_click=load_random_real_article,
        )

    with button_cols[1]:
        st.button(
            "Fetch Fake News",
            use_container_width=True,
            on_click=load_random_fake_article,
        )

    if st.session_state["fetch_message"]:
        st.caption(st.session_state["fetch_message"])

    with st.form("prediction_form", clear_on_submit=False):
        title = st.text_input(
            "Headline",
            key="headline",
            placeholder="Enter the article headline",
        )
        text = st.text_area(
            "Article Text",
            key="article_text",
            placeholder="Paste the article body",
            height=240,
        )
        predict_clicked = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        if not os.path.exists(BEST_MODEL_PATH):
            st.error("Train the model first with `python3 train.py`.")
        elif not title.strip() and not text.strip():
            st.warning("Please enter a headline or article body.")
        else:
            prepared = prepare_user_text(title, text)
            if not prepared.strip():
                st.warning(
                    "The edited text does not contain enough supported English words after preprocessing. "
                    "Please enter a fuller news headline or article body."
                )
            else:
                try:
                    model_bundle = get_model_bundle()
                    predictions, probabilities = predict_with_model(model_bundle, [prepared])
                    label = model_bundle["label_mapping"][int(predictions[0])]
                    fake_probability = float(probabilities[0][0])
                    real_probability = float(probabilities[0][1])

                    if label == "REAL":
                        st.success(f"Prediction: {label}")
                    else:
                        st.error(f"Prediction: {label}")

                    result_cols = st.columns(2)
                    result_cols[0].metric("REAL probability", f"{real_probability:.2%}")
                    result_cols[1].metric("FAKE probability", f"{fake_probability:.2%}")
                except Exception as error:
                    st.error(f"Prediction failed: {error}")
                    st.info("Try refreshing the page or retraining with `python3 train.py`.")

    st.divider()
    st.subheader("How It Works")
    st.write("1. `Fetch Real News` loads a recent Times of India article from the website feed.")
    st.write("2. `Fetch Fake News` loads a fake example from the local labeled dataset.")
    st.write("3. The text is cleaned, converted into TF-IDF features, and scored by Logistic Regression.")

    st.divider()
    st.subheader("Live Times of India Data")
    live_articles = get_live_articles()
    if live_articles.empty:
        st.info("No Times of India live articles found in SQLite yet. Run `python3 train.py --refresh-live-data`.")
    else:
        visible_columns = [column for column in ["source", "title", "published_at", "link"] if column in live_articles.columns]
        st.dataframe(live_articles[visible_columns], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
