"""Decoder training for side-specific binary LDA models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from .features import r2_feature_ranking, zscore_from_train


@dataclass
class DecoderModel:
    """Serialized decoder contents mirroring MATLAB-like snapshot structure."""

    lda: LinearDiscriminantAnalysis
    selected_idx: np.ndarray
    norm_mean: np.ndarray
    norm_std: np.ndarray
    threshold: float
    margin: float
    metrics: dict[str, float]


def train_side_decoder(
    X: np.ndarray,
    y_binary: np.ndarray,
    threshold: float = 0.5,
    margin: float = 0.05,
    top_k: int = 30,
) -> DecoderModel:
    """Train one binary decoder (L/R/N) with r² feature selection + LDA."""

    selector = r2_feature_ranking(X, y_binary, top_k=top_k)
    Xs = X[:, selector.indices]
    mean = Xs.mean(axis=0, keepdims=True)
    std = Xs.std(axis=0, keepdims=True) + 1e-8
    Xn = (Xs - mean) / std

    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(Xn, y_binary)
    prob = lda.predict_proba(Xn)[:, 1]
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_binary, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_binary, pred)),
        "auprc": float(average_precision_score(y_binary, prob)),
        "tpr": float(tp / max(tp + fn, 1)),
        "tnr": float(tn / max(tn + fp, 1)),
    }
    return DecoderModel(
        lda=lda,
        selected_idx=selector.indices,
        norm_mean=mean,
        norm_std=std,
        threshold=threshold,
        margin=margin,
        metrics=metrics,
    )


def predict_decoder(decoder: DecoderModel, X: np.ndarray) -> np.ndarray:
    """Return posterior P(class=1) for one decoder."""

    Xs = X[:, decoder.selected_idx]
    Xn = (Xs - decoder.norm_mean) / decoder.norm_std
    return decoder.lda.predict_proba(Xn)[:, 1]


def classify_three_way(
    p_right: np.ndarray,
    p_left: np.ndarray,
    p_none: np.ndarray,
    thr_right: float,
    thr_left: float,
    thr_none: float,
    margin: float,
) -> np.ndarray:
    """Apply threshold + margin to produce classes {1=no, 2=distractor, 3=ambivalent}."""

    stack = np.column_stack([p_none, p_right, p_left])
    thr = np.array([thr_none, thr_right, thr_left])[None, :]
    passed = stack - thr

    best = np.argmax(passed, axis=1)
    sorted_scores = np.sort(passed, axis=1)
    confident = (sorted_scores[:, -1] - sorted_scores[:, -2]) >= margin

    out = np.full(len(best), 3, dtype=int)
    out[(best == 0) & confident] = 1
    out[((best == 1) | (best == 2)) & confident] = 2
    return out


def cross_validated_side_decoding(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> dict[str, np.ndarray | float]:
    """Cross-validated one-vs-rest decoding with train-fold normalization/selection.

    y coding convention: 0=no distractor, 1=right, 2=left.
    """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    post = np.zeros((len(y), 3), dtype=float)

    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]

        for cls in (0, 1, 2):
            yb_tr = (ytr == cls).astype(int)
            selector = r2_feature_ranking(Xtr, yb_tr, top_k=30)
            Xtr_s = Xtr[:, selector.indices]
            Xte_s = Xte[:, selector.indices]
            Xtr_n, Xte_n, _ = zscore_from_train(Xtr_s, Xte_s)

            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(Xtr_n, yb_tr)
            post[te, cls] = lda.predict_proba(Xte_n)[:, 1]

    pred = np.argmax(post, axis=1)
    return {"posteriors": post, "predicted_class": pred, "accuracy": float((pred == y).mean())}
