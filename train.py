from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from fake_news_detector.config import BASE_DATASET_PATH, DATABASE_PATH, LIVE_DATASET_PATH
from fake_news_detector.database import initialize_database, replace_live_articles, save_training_run
from fake_news_detector.dataset import combine_datasets, label_distribution, load_csv_dataset
from fake_news_detector.live_data import fetch_default_live_dataset, save_live_dataset
from fake_news_detector.modeling import final_evaluation, save_model_artifacts, split_dataset, tune_best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fake news detection project.")
    parser.add_argument(
        "--refresh-live-data",
        action="store_true",
        help="Fetch recent Times of India RSS items before training.",
    )
    parser.add_argument(
        "--live-limit",
        type=int,
        default=25,
        help="Number of RSS items to fetch per source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_database(DATABASE_PATH)

    live_articles: list[object] = []
    if args.refresh_live_data:
        live_articles = fetch_default_live_dataset(limit_per_source=args.live_limit)
        save_live_dataset(live_articles, os.path.join(PROJECT_ROOT, "data", "live_news_dataset.csv"))
        replace_live_articles(DATABASE_PATH, live_articles)
        print(f"Fetched and stored {len(live_articles)} live articles.")

    dataframe = combine_datasets(include_live_data=True)
    X_train, X_test, y_train, y_test = split_dataset(dataframe)
    base_dataset_count = len(load_csv_dataset(BASE_DATASET_PATH))
    live_dataset_count = len(load_csv_dataset(LIVE_DATASET_PATH)) if os.path.exists(LIVE_DATASET_PATH) else 0

    best_model_bundle, tuning_summary = tune_best_model(X_train, y_train)
    evaluation = final_evaluation(best_model_bundle, X_test, y_test)

    metadata = {
        "dataset_size": int(len(dataframe)),
        "base_dataset_samples": int(base_dataset_count),
        "live_dataset_samples": int(live_dataset_count),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "label_distribution": label_distribution(dataframe),
        "model_name": "Logistic Regression + TF-IDF",
        "feature_extraction": {
            "tfidf": "Weights important words higher and common words lower.",
        },
        "tuning_summary": tuning_summary,
        **evaluation,
    }

    save_model_artifacts(best_model_bundle, metadata)
    save_training_run(DATABASE_PATH, metadata)

    print("Training complete")
    print(json.dumps(
        {
            "model": metadata["model_name"],
            "accuracy": round(metadata["accuracy"], 4),
            "precision": round(metadata["precision"], 4),
            "recall": round(metadata["recall"], 4),
            "f1_score": round(metadata["f1_score"], 4),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
