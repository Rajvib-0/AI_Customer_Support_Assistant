"""Module 1 - customer query classification (TF-IDF + Linear SVM)."""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config as cfg

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_NUM = re.compile(r"#?\b\d{4,}\b")
_NON_ALNUM = re.compile(r"[^a-z0-9' ]+")


def clean_text(text: str) -> str:
    """Lowercase, mask URLs/emails/long numbers (order IDs), strip punctuation. Stop-words are kept on purpose."""
    t = _URL.sub(" ", str(text).lower())
    t = _EMAIL.sub(" ", t)
    t = _NUM.sub(" longnumber ", t)
    return re.sub(r"\s+", " ", _NON_ALNUM.sub(" ", t)).strip()


TEXT_COLUMNS = ("query", "text", "message", "customer_query", "question", "utterance")
LABEL_COLUMNS = ("category", "label", "intent", "class", "target")


def train_and_save(csv_path, out_path=cfg.CLASSIFIER_PATH, seed: int = 42) -> dict:
    """Train TF-IDF + (calibrated) Linear SVM on a CSV and save the pipeline. Returns test metrics."""
    df = pd.read_csv(csv_path).dropna()
    cols = {c.lower(): c for c in df.columns}
    text_col = next((cols[c] for c in TEXT_COLUMNS if c in cols), None)
    label_col = next((cols[c] for c in LABEL_COLUMNS if c in cols), None)
    if not text_col or not label_col:
        raise ValueError(f"Could not find text/label columns in {list(df.columns)}")
    df = df.drop_duplicates(text_col)

    def make():
        return Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)),
            # calibration lets the SVM expose real probabilities for the confidence score
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3)),
        ])

    X_tr, X_te, y_tr, y_te = train_test_split(df[text_col], df[label_col], test_size=0.2,
                                              stratify=df[label_col], random_state=seed)
    report = classification_report(y_te, make().fit(X_tr, y_tr).predict(X_te), output_dict=True, zero_division=0)
    final = make().fit(df[text_col], df[label_col])          # refit on all data for deployment
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "name": "TF-IDF + Linear SVM (calibrated)"}, out_path)
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
            "rows": len(df), "classes": sorted(df[label_col].unique())}


class ClassifierService:
    def __init__(self, path=cfg.CLASSIFIER_PATH):
        obj = joblib.load(path)
        self.vectorizer = None
        if isinstance(obj, dict) and "model" in obj:                 # {"model": pipeline_or_clf, ["vectorizer": tfidf]}
            self.model, self.vectorizer = obj["model"], obj.get("vectorizer")
            self.name = obj.get("name")
        elif isinstance(obj, (tuple, list)) and len(obj) == 2:       # (vectorizer, classifier)
            self.vectorizer, self.model = obj
            self.name = None
        elif hasattr(obj, "predict"):                                # a bare sklearn Pipeline
            self.model, self.name = obj, None
        else:
            raise ValueError("Unsupported classifier file: expected a Pipeline, (vectorizer, model) or a dict")
        if not self.name:
            final = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
            self.name = type(final).__name__
        self.classes = [str(c) for c in self.model.classes_]

    def _scores(self, text: str):
        x = self.vectorizer.transform([text]) if self.vectorizer is not None else [text]
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(x)[0], dtype=float), "probability"
        margins = np.asarray(self.model.decision_function(x), dtype=float)
        margins = np.stack([-margins, margins], axis=1)[0] if margins.ndim == 1 else margins[0]
        e = np.exp(margins - margins.max())
        return e / e.sum(), "softmax_of_margin"

    def predict(self, query: str) -> dict:
        text = clean_text(query) if cfg.CLASSIFIER_PREPROCESS else query
        probs, kind = self._scores(text)
        i = int(probs.argmax())
        return {"query": query, "category": self.classes[i], "confidence": round(float(probs[i]), 4),
                "confidence_type": kind, "model": self.name,
                "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs)}}
