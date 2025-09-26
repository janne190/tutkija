"""Model-based screening helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

INCLUDED = "included"
EXCLUDED = "excluded"
DEFAULT_THRESHOLD = 0.5
_POSITIVE_LABELS = {"included", "relevant", "positive", "1", "true", "yes"}
_NEGATIVE_LABELS = {"excluded", "irrelevant", "negative", "0", "false", "no"}


@dataclass
class ScreeningResult:
    frame: pd.DataFrame
    threshold: float
    engine: str
    metadata: dict[str, object] = field(default_factory=dict)


def pick_threshold_for_recall(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    target_recall: float = 0.9,
) -> float:
    """Return the smallest threshold meeting the desired recall."""

    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if y_true_arr.size == 0 or y_prob_arr.size == 0:
        raise ValueError("y_true and y_prob must be non-empty")
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")
    if y_true_arr.sum() == 0:
        return float(DEFAULT_THRESHOLD)

    candidates = sorted(set(y_prob_arr.tolist() + [0.0, 1.0]))
    best_threshold: float | None = None
    for threshold in candidates:
        predictions = (y_prob_arr >= threshold).astype(int)
        recall = recall_score(y_true_arr, predictions, zero_division=0)
        if recall >= target_recall:
            best_threshold = float(threshold)
            break
    if best_threshold is None:
        return float(DEFAULT_THRESHOLD)
    return best_threshold


def score_and_label(
    df: pd.DataFrame,
    *,
    target_recall: float = 0.9,
    seed: int = 7,
    use_asreview: bool = False,
    seeds: Sequence[str] | None = None,
) -> ScreeningResult:
    """Score records and assign labels using either scikit-learn or ASReview."""

    frame = df.copy()
    if use_asreview:
        from .asreview_wrapper import score_with_asreview

        return score_with_asreview(
            frame, target_recall=target_recall, seed=seed, seeds=seeds
        )
    return _score_with_scikit(
        frame, target_recall=target_recall, seed=seed, seeds=seeds
    )


def _score_with_scikit(
    frame: pd.DataFrame,
    *,
    target_recall: float,
    seed: int,
    seeds: Sequence[str] | None,
    engine_name: str = "scikit",
) -> ScreeningResult:
    text_series = _combine_text_columns(frame)
    metadata: dict[str, object] = {"trained": False, "strategy": "default_threshold"}
    if frame.empty:
        frame["probability"] = pd.Series(dtype=float)
        frame["label"] = pd.Series(dtype=str)
        metadata["strategy"] = "empty"
        return ScreeningResult(
            frame=frame, threshold=DEFAULT_THRESHOLD, engine=engine_name, metadata=metadata
        )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(random_state=seed, class_weight="balanced")),
        ]
    )

    y_prob: np.ndarray | None = None
    threshold = DEFAULT_THRESHOLD

    if "gold_label" in frame.columns and frame["gold_label"].notna().any():
        gold_set = frame[frame["gold_label"].notna()].copy()
        y_true = gold_set["gold_label"].apply(_to_binary_label).values
        if np.any(y_true == 1):
            pipeline.fit(
                _combine_text_columns(gold_set),
                y_true,
            )
            y_prob = pipeline.predict_proba(text_series)[:, 1]
            threshold = pick_threshold_for_recall(
                y_true, pipeline.predict_proba(gold_set)[:, 1], target_recall
            )
            metadata["trained"] = True
            metadata["strategy"] = "recall_optimized"
        else:
            logger.warning(
                "Gold set found but contains no positive labels, using default threshold"
            )
            # Fall through to pseudo-labels or untrained
    
    if y_prob is None:
        # Pseudo-fit with rules and seeds if available
        pseudo_labels = _create_pseudo_labels(frame, seeds)
        if np.any(pseudo_labels == 1):
            pipeline.fit(text_series, pseudo_labels)
            y_prob = pipeline.predict_proba(text_series)[:, 1]
            metadata["trained"] = True
            metadata["strategy"] = "pseudo_labels"
        else:
            # If no positive pseudo-labels, just fit the vectorizer
            # and use the untrained classifier.
            logger.warning(
                "No gold set or positive seeds found. Model is not trained, using default threshold."
            )
            # Fit only the vectorizer, then create default probabilities
            pipeline.named_steps["tfidf"].fit(text_series)
            y_prob = np.full(len(text_series), 0.5)  # Create a 1D array
            metadata["trained"] = False
            metadata["strategy"] = "untrained_default"

    frame["probability"] = y_prob
    frame["label"] = np.where(
        frame["probability"] >= threshold, INCLUDED, EXCLUDED
    ).tolist()

    # Records with reasons are always excluded
    if "reasons" in frame.columns:
        has_reasons = frame["reasons"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        frame.loc[has_reasons, "label"] = EXCLUDED

    return ScreeningResult(
        frame=frame, threshold=threshold, engine=engine_name, metadata=metadata
    )


def _combine_text_columns(df: pd.DataFrame) -> pd.Series:
    """Safely combine title and abstract into a single text series."""
    title = df["title"].fillna("").astype(str)
    abstract = df["abstract"].fillna("").astype(str)
    return (title + " " + abstract).str.strip()


def _to_binary_label(label: Any) -> int:
    """Convert a label to a binary 0 or 1."""
    if pd.isna(label):
        return 0
    label_str = str(label).lower().strip()
    if label_str in _POSITIVE_LABELS:
        return 1
    return 0

def _create_pseudo_labels(df: pd.DataFrame, seeds: Sequence[str] | None) -> np.ndarray:
    """Create pseudo-labels for training based on rules and seeds."""
    labels = np.zeros(len(df))
    
    # Seeds are positive examples
    if seeds:
        seed_ids = set(seeds)
        for idx, row in df.iterrows():
            if row.get("id") in seed_ids or row.get("doi") in seed_ids:
                labels[idx] = 1

    # Rule-based exclusions are negative examples
    if "reasons" in df.columns:
        has_reasons = df["reasons"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        labels[has_reasons] = 0 # Explicitly set to 0, even if it was a seed

    return labels
