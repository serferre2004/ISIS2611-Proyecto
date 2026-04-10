from __future__ import annotations

import re
import time
from dataclasses import dataclass

import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


SEED = 42


def preprocess_agresivo(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-záéíóúñüç\s]", " ", text)
    text = re.sub(r"\b(?![aeiouyáéíóú])[a-záéíóúñüç]\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class Experiment:
    name: str
    text_key: str
    vectorizer: BaseEstimator
    model: BaseEstimator


def feature_union_char_word(
    *,
    analyzer: str = "char_wb",
    char_ngram: tuple[int, int] = (3, 5),
    char_min_df: int = 3,
    char_max_features: int = 80000,
    word_ngram: tuple[int, int] = (1, 2),
    word_min_df: int = 3,
    word_max_features: int = 30000,
) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=char_ngram,
                    lowercase=True,
                    min_df=char_min_df,
                    sublinear_tf=True,
                    max_features=char_max_features,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=word_ngram,
                    lowercase=True,
                    min_df=word_min_df,
                    sublinear_tf=True,
                    max_features=word_max_features,
                ),
            ),
        ]
    )


def char_vectorizer(
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    min_df: int,
    max_features: int,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=True,
        min_df=min_df,
        sublinear_tf=True,
        max_features=max_features,
    )


def run_experiment(
    exp: Experiment,
    X_tr_map: dict[str, pd.Series],
    X_val_map: dict[str, pd.Series],
    y_tr,
    y_val,
) -> dict[str, object]:
    started = time.perf_counter()
    X_tr = X_tr_map[exp.text_key]
    X_val = X_val_map[exp.text_key]

    X_tr_feat = exp.vectorizer.fit_transform(X_tr)
    X_val_feat = exp.vectorizer.transform(X_val)

    if sparse.isspmatrix_csr(X_tr_feat):
        train_shape = X_tr_feat.shape
    else:
        train_shape = sparse.csr_matrix(X_tr_feat).shape

    exp.model.fit(X_tr_feat, y_tr)
    pred = exp.model.predict(X_val_feat)
    acc = accuracy_score(y_val, pred)
    elapsed = time.perf_counter() - started
    return {
        "name": exp.name,
        "accuracy": acc,
        "seconds": elapsed,
        "n_features": train_shape[1],
    }


def main() -> None:
    df = pd.read_csv("data/train.csv")
    df["clean_text"] = df["text"].apply(preprocess_agresivo)

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=0.15,
        random_state=SEED,
        stratify=df["decade"],
    )
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)

    X_tr_map = {
        "raw": df.loc[train_idx, "text"],
        "clean": df.loc[train_idx, "clean_text"],
    }
    X_val_map = {
        "raw": df.loc[val_idx, "text"],
        "clean": df.loc[val_idx, "clean_text"],
    }
    y_tr = df.loc[train_idx, "decade"].to_numpy()
    y_val = df.loc[val_idx, "decade"].to_numpy()

    experiments = [
        Experiment(
            name="clean_char_wb_3_5_linsvc_c1",
            text_key="clean",
            vectorizer=char_vectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_features=80000,
            ),
            model=LinearSVC(C=1.0, dual="auto"),
        ),
        Experiment(
            name="clean_char_3_5_linsvc_c1",
            text_key="clean",
            vectorizer=char_vectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=3,
                max_features=100000,
            ),
            model=LinearSVC(C=1.0, dual="auto"),
        ),
        Experiment(
            name="clean_union_charwb_word_linsvc_c1",
            text_key="clean",
            vectorizer=feature_union_char_word(
                analyzer="char_wb",
                char_ngram=(3, 5),
                char_min_df=3,
                char_max_features=80000,
                word_ngram=(1, 2),
                word_min_df=3,
                word_max_features=30000,
            ),
            model=LinearSVC(C=1.0, dual="auto"),
        ),
        Experiment(
            name="raw_char_wb_3_5_linsvc_c1",
            text_key="raw",
            vectorizer=char_vectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_features=100000,
            ),
            model=LinearSVC(C=1.0, dual="auto"),
        ),
        Experiment(
            name="raw_char_3_5_linsvc_c1",
            text_key="raw",
            vectorizer=char_vectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=3,
                max_features=120000,
            ),
            model=LinearSVC(C=1.0, dual="auto"),
        ),
        Experiment(
            name="raw_union_char_word_linsvc_c1",
            text_key="raw",
            vectorizer=feature_union_char_word(
                analyzer="char",
                char_ngram=(3, 5),
                char_min_df=3,
                char_max_features=120000,
                word_ngram=(1, 2),
                word_min_df=3,
                word_max_features=30000,
            ),
            model=LinearSVC(C=1.0, dual="auto"),
        ),
        Experiment(
            name="clean_char_3_5_sgd_logloss",
            text_key="clean",
            vectorizer=char_vectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=3,
                max_features=100000,
            ),
            model=SGDClassifier(
                loss="log_loss",
                alpha=1e-6,
                penalty="l2",
                max_iter=50,
                tol=1e-3,
                random_state=SEED,
            ),
        ),
        Experiment(
            name="clean_char_wb_3_5_logreg_c4",
            text_key="clean",
            vectorizer=char_vectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_features=60000,
            ),
            model=LogisticRegression(
                C=4.0,
                solver="saga",
                max_iter=300,
                verbose=0,
            ),
        ),
    ]

    results = []
    print("Baseline notebook (Ridge ordinal): accuracy=0.0684")
    for exp in experiments:
        print(f"\n>>> Running {exp.name}")
        result = run_experiment(exp, X_tr_map, X_val_map, y_tr, y_val)
        results.append(result)
        print(
            f"{result['name']}: accuracy={result['accuracy']:.4f} "
            f"features={result['n_features']:,} time={result['seconds']:.1f}s"
        )

    print("\n=== Sorted by accuracy ===")
    for row in sorted(results, key=lambda r: r["accuracy"], reverse=True):
        print(
            f"{row['accuracy']:.4f} | {row['seconds']:6.1f}s | "
            f"{row['n_features']:7,d} | {row['name']}"
        )


if __name__ == "__main__":
    main()
