"""
Production Training Script for Emotion and Sentiment Classification using MELD Dataset.
Dataset directory: data/meld/
"""

import json
import re
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "meld"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Normalize text and remove special characters."""
    if not text:
        return ""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(split_name: str):
    """Load and flatten dialogue turns from JSON split."""
    json_path = DATA_DIR / f"meld_{split_name}_dialogues.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing file: {json_path}. Run ingest_meld.py first.")

    with open(json_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    texts = []
    emotions = []
    sentiments = []

    for d_id, turns in dialogues.items():
        turns_sorted = sorted(turns, key=lambda x: x.get("utterance_id", 0))
        for i, turn in enumerate(turns_sorted):
            cleaned = clean_text(turn.get("utterance", ""))
            if not cleaned:
                continue

            # Context enrichment: previous turn speaker and text
            if i > 0:
                prev_turn = turns_sorted[i - 1]
                prev_text = clean_text(prev_turn.get("utterance", ""))
                enriched = f"[{prev_turn.get('speaker', 'Other')}: {prev_text[:60]}] {turn.get('speaker', 'Speaker')}: {cleaned}"
            else:
                enriched = f"{turn.get('speaker', 'Speaker')}: {cleaned}"

            texts.append(enriched)
            emotions.append(turn.get("emotion", "neutral").lower())
            sentiments.append(turn.get("sentiment", "neutral").lower())

    return texts, emotions, sentiments


def train_and_evaluate():
    print("[*] Loading MELD train, dev, and test sets...")
    train_texts, train_emotions, train_sentiments = load_split("train")
    dev_texts, dev_emotions, dev_sentiments = load_split("dev")
    test_texts, test_emotions, test_sentiments = load_split("test")

    print(f"[+] Loaded {len(train_texts)} train turns, {len(dev_texts)} dev turns, {len(test_texts)} test turns.")

    all_train_texts = train_texts + dev_texts
    all_train_emotions = train_emotions + dev_emotions
    all_train_sentiments = train_sentiments + dev_sentiments

    report_data = {
        "dataset": "MELD Conversational Dataset (data/meld)",
        "train_samples": len(all_train_texts),
        "test_samples": len(test_texts),
        "models": {}
    }

    # ==========================================
    # 1. EMOTION CLASSIFIER (7 Classes)
    # ==========================================
    print("\n" + "="*50)
    print("[*] Training Emotion Classifier (7 classes with balanced class weights)...")
    
    from sklearn.pipeline import FeatureUnion

    emotion_pipeline = Pipeline([
        ("union", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 3),
                min_df=2,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=5,
                sublinear_tf=True,
            )),
        ])),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.5,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    emotion_pipeline.fit(all_train_texts, all_train_emotions)

    pred_emotions = emotion_pipeline.predict(test_texts)
    classes_emotion = list(emotion_pipeline.classes_)

    acc_emo = accuracy_score(test_emotions, pred_emotions)
    macro_f1_emo = f1_score(test_emotions, pred_emotions, average="macro")
    weighted_f1_emo = f1_score(test_emotions, pred_emotions, average="weighted")

    print(f"[+] Emotion Test Accuracy: {acc_emo:.4f}")
    print(f"[+] Emotion Test Macro-F1: {macro_f1_emo:.4f}")
    print(f"[+] Emotion Test Weighted-F1: {weighted_f1_emo:.4f}")
    print("\nClassification Report (Emotion):")
    print(classification_report(test_emotions, pred_emotions, zero_division=0))

    report_data["models"]["emotion"] = {
        "accuracy": round(float(acc_emo), 4),
        "macro_f1": round(float(macro_f1_emo), 4),
        "weighted_f1": round(float(weighted_f1_emo), 4),
        "classes": classes_emotion,
        "classification_report": classification_report(test_emotions, pred_emotions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test_emotions, pred_emotions, labels=classes_emotion).tolist()
    }

    emotion_model_path = MODELS_DIR / "emotion_pipeline.joblib"
    joblib.dump(emotion_pipeline, emotion_model_path)
    print(f"[OK] Saved Emotion Model to {emotion_model_path}")

    # ==========================================
    # 2. SENTIMENT CLASSIFIER (3 Classes)
    # ==========================================
    print("\n" + "="*50)
    print("[*] Training Sentiment Classifier (3 classes with balanced class weights)...")

    sentiment_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=16000,
            sublinear_tf=True,
            min_df=2
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=2.0,
            solver="lbfgs"
        ))
    ])

    sentiment_pipeline.fit(all_train_texts, all_train_sentiments)

    pred_sentiments = sentiment_pipeline.predict(test_texts)
    classes_sentiment = list(sentiment_pipeline.classes_)

    acc_sent = accuracy_score(test_sentiments, pred_sentiments)
    macro_f1_sent = f1_score(test_sentiments, pred_sentiments, average="macro")
    weighted_f1_sent = f1_score(test_sentiments, pred_sentiments, average="weighted")

    print(f"[+] Sentiment Test Accuracy: {acc_sent:.4f}")
    print(f"[+] Sentiment Test Macro-F1: {macro_f1_sent:.4f}")
    print(f"[+] Sentiment Test Weighted-F1: {weighted_f1_sent:.4f}")
    print("\nClassification Report (Sentiment):")
    print(classification_report(test_sentiments, pred_sentiments, zero_division=0))

    report_data["models"]["sentiment"] = {
        "accuracy": round(float(acc_sent), 4),
        "macro_f1": round(float(macro_f1_sent), 4),
        "weighted_f1": round(float(weighted_f1_sent), 4),
        "classes": classes_sentiment,
        "classification_report": classification_report(test_sentiments, pred_sentiments, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test_sentiments, pred_sentiments, labels=classes_sentiment).tolist()
    }

    sentiment_model_path = MODELS_DIR / "sentiment_pipeline.joblib"
    joblib.dump(sentiment_pipeline, sentiment_model_path)
    print(f"[OK] Saved Sentiment Model to {sentiment_model_path}")

    # Save benchmark report
    report_file = REPORTS_DIR / "emotion_benchmark_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "="*50)
    print(f"[OK] Models and evaluation reports saved to:")
    print(f"     - {emotion_model_path}")
    print(f"     - {sentiment_model_path}")
    print(f"     - {report_file}")


if __name__ == "__main__":
    train_and_evaluate()
