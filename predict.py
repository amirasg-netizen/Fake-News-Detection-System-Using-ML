from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from fake_news_detector.config import BEST_MODEL_PATH
from fake_news_detector.dataset import prepare_user_text
from fake_news_detector.modeling import load_model_bundle, predict_with_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a news article is REAL or FAKE.")
    parser.add_argument("--title", required=True, help="Article headline")
    parser.add_argument("--text", required=True, help="Article content")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {BEST_MODEL_PATH}. Run `python3 train.py` first.")

    model_bundle = load_model_bundle()
    prepared_text = prepare_user_text(args.title, args.text)
    predictions, probabilities = predict_with_model(model_bundle, [prepared_text])
    label_mapping = model_bundle["label_mapping"]
    label = label_mapping[int(predictions[0])]

    print(f"Prediction: {label}")
    print(f"FAKE probability: {probabilities[0][0]:.2%}")
    print(f"REAL probability: {probabilities[0][1]:.2%}")


if __name__ == "__main__":
    main()
