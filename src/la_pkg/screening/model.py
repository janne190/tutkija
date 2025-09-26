"""Model-based screening helpers."""

from __future__ import annotations

import logging
from typing import Any, Sequence, TypedDict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline


__all__ = ["score_and_label", "ScreenStats"]

logger = logging.getLogger(__name__)

# Try importing ASReview to check if it's available
try:
    import asreview  # noqa

    ASREVIEW_AVAILABLE = True
except ImportError:
    ASREVIEW_AVAILABLE = False

INCLUDED = "included"
EXCLUDED = "excluded"
DEFAULT_THRESHOLD = 0.5
_POSITIVE_LABELS = {"included", "relevant", "positive", "1", "true", "yes"}
_NEGATIVE_LABELS = {"excluded", "irrelevant", "negative", "0", "false", "no"}


class ScreenStats(TypedDict):
    """Statistics about the screening process."""

    identified: int  # Total papers found
    screened: int  # Papers processed
    excluded_rules: int  # Pre-filter exclusions
    excluded_model: int  # Model-based exclusions
    included: int  # Papers passing both stages
    engine: str  # "scikit" or "asreview"
    recall_target: float  # Target recall setting
    threshold_used: float  # Classification threshold
    seeds_count: int  # Number of seed papers used
    version: str  # Version string
    random_state: int  # Random seed used
    fallback: str  # Fallback mode used, if any


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
) -> tuple[pd.DataFrame, ScreenStats]:
    """Score records and assign labels using either scikit-learn or ASReview."""

    frame = df.copy()

    if use_asreview:
        if not ASREVIEW_AVAILABLE:
            raise RuntimeError(
                "ASReview is not installed. Install with 'pip install tutkija[asreview]' "
                "or 'uv pip install asreview'. Alternatively, use --engine scikit."
            )
        from .asreview_wrapper import score_with_asreview

        frame, stats = score_with_asreview(
            frame, target_recall=target_recall, seed=seed, seeds=seeds
        )
        return frame, stats

    frame, stats = _score_with_scikit(
        frame, target_recall=target_recall, seed=seed, seeds=seeds
    )
    return frame, stats


def _score_with_scikit(
    frame: pd.DataFrame,
    *,
    target_recall: float,
    seed: int,
    seeds: Sequence[str] | None,
    engine_name: str = "scikit",
) -> tuple[pd.DataFrame, ScreenStats]:
    text_series = _combine_text_columns(frame)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(random_state=seed, class_weight="balanced")),
        ]
    )

    if frame.empty:
        frame["probability"] = pd.Series(dtype=float)
        frame["label"] = pd.Series(dtype=str)
        empty_stats: ScreenStats = {
            "identified": 0,
            "screened": 0,
            "excluded_rules": 0,
            "excluded_model": 0,
            "included": 0,
            "engine": engine_name,
            "recall_target": target_recall,
            "threshold_used": DEFAULT_THRESHOLD,
            "seeds_count": len(seeds or []),
            "version": asreview.__version__ if ASREVIEW_AVAILABLE else "none",
            "random_state": seed,
            "fallback": "none",
        }
        return frame, empty_stats

    y_prob: np.ndarray | None = None
    threshold = DEFAULT_THRESHOLD

    # Always fit TfidfVectorizer on all text
    vectorizer = pipeline.named_steps["tfidf"]
    vectorizer.fit(text_series)

    # Try to get training data from gold labels first
    y_prob = None
    if "gold_label" in frame.columns and frame["gold_label"].notna().any():
        gold_set = frame[frame["gold_label"].notna()].copy()
        y_true = gold_set["gold_label"].apply(_to_binary_label).values
        if np.any(y_true == 1) and np.any(y_true == 0):
            # Only fit classifier if we have both positive and negative examples
            gold_text = _combine_text_columns(gold_set)
            pipeline.fit(gold_text, y_true)
            y_prob = pipeline.predict_proba(text_series)[:, 1]
            # Calculate threshold based on probabilities for gold set
            gold_probs = pipeline.predict_proba(_combine_text_columns(gold_set))[:, 1]
            threshold = pick_threshold_for_recall(y_true, gold_probs, target_recall)
        else:
            logger.warning(
                "Gold set found but contains only one class, using default threshold"
            )

    # If no gold labels, try pseudo-labels from rules and seeds
    if y_prob is None:
        pseudo_labels = _create_pseudo_labels(frame, seeds)
        if np.any(pseudo_labels == 1):
            # Get text features for all papers
            X = vectorizer.transform(text_series)

            # Identify seed papers to compute similarity with
            seed_mask = pseudo_labels == 1
            seed_vectors = X[seed_mask]

            # Compute cosine similarity with seed papers
            similarities = (X @ seed_vectors.T).toarray()
            # Take max similarity per paper and rescale
            max_sim = similarities.max(axis=1)
            # Scale up similarities to make seeded papers have more influence
            y_prob = 0.5 + 0.5 * max_sim
        else:
            logger.warning(
                "No gold set or seeds found. Using default probabilities 0.5."
            )
            y_prob = np.full(len(text_series), 0.5)  # Create a 1D array

    # Use higher threshold for untrained model with high recall target
    if y_prob is None or np.all(y_prob == 0.5):
        y_prob = np.full(len(frame), 0.5)  # Ensure array for consistent return type
        # For high recall, use higher threshold to exclude more aggressively
        threshold = 0.6 if target_recall > 0.8 else 0.5

    frame["probability"] = y_prob
    frame["label"] = np.where(
        frame["probability"] >= threshold, INCLUDED, EXCLUDED
    ).tolist()

    # Records with reasons are always excluded
    if "reasons" in frame.columns:
        has_reasons = frame["reasons"].apply(
            lambda x: isinstance(x, list) and len(x) > 0
        )
        frame.loc[has_reasons, "label"] = EXCLUDED
    identified = len(frame)
    screened = frame["label"].notna().sum()
    excluded_rules = 0 if "reasons" not in frame.columns else has_reasons.sum()
    excluded_model = (frame["label"] == EXCLUDED).sum() - excluded_rules
    included = (frame["label"] == INCLUDED).sum()

    stats: ScreenStats = {
        "identified": identified,
        "screened": screened,
        "excluded_rules": excluded_rules,
        "excluded_model": excluded_model,
        "included": included,
        "engine": engine_name,
        "recall_target": target_recall,
        "threshold_used": threshold,
        "seeds_count": len(seeds or []),
        "version": asreview.__version__ if ASREVIEW_AVAILABLE else "1.0.0",
        "random_state": seed,
        "fallback": "default_prob_0.5"
        if y_prob is None or np.all(y_prob == 0.5)
        else "none",
    }
    return frame, stats


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
    """Create pseudo-labels for training based on rules and seeds.

    Returns:
        A numpy array where:
        - 1 indicates a positive example (from seeds)
        - 0 indicates a negative example (from rules or unlabeled)
    """
    labels = np.zeros(len(df), dtype=int)  # Start with all as negative examples

    # First apply seed-based positive examples
    if seeds:
        seed_ids = set(seeds)
        for idx, row in df.iterrows():
            doc_id = str(row.get("id", ""))  # Ensure string comparison
            doc_doi = str(row.get("doi", "")).lower()  # Normalize DOI comparison
            if doc_id in seed_ids or doc_doi in seed_ids:
                labels[idx] = 1

    # Then override with rule-based negative examples
    if "reasons" in df.columns:
        has_reasons = df["reasons"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        labels[has_reasons] = 0  # Rules override seeds

    return labels
