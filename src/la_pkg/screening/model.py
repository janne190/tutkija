"""Model-based screening helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

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
    if "reasons" not in frame.columns:
        frame["reasons"] = [[] for _ in range(len(frame))]
    else:
        frame["reasons"] = frame["reasons"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    text_series = _combine_text_columns(frame)
    metadata: dict[str, object] = {"trained": False, "strategy": "uniform"}
    if frame.empty:
        frame["probability"] = []
        frame["label"] = []
        metadata["strategy"] = "empty"
        return ScreeningResult(
            frame=frame,
            threshold=float(DEFAULT_THRESHOLD),
            engine=engine_name,
            metadata=metadata,
        )

    probabilities = np.full(frame.shape[0], DEFAULT_THRESHOLD, dtype=float)
    threshold = float(DEFAULT_THRESHOLD)

    gold_labels = _extract_gold_labels(frame)
    pipeline: Pipeline | None = None

    if gold_labels is not None and not gold_labels.empty:
        pipeline = _build_pipeline(seed)
        train_text = text_series.loc[gold_labels.index]
        try:
            pipeline.fit(train_text, gold_labels.to_numpy())
        except ValueError as exc:
            logger.warning("Unable to train logistic model with gold labels: %s", exc)
            pipeline = None
        else:
            metadata["trained"] = True
            metadata["strategy"] = "gold"
            probabilities = pipeline.predict_proba(text_series)[:, 1]
            train_probs = pipeline.predict_proba(train_text)[:, 1]
            threshold = pick_threshold_for_recall(
                gold_labels.to_numpy(), train_probs, target_recall
            )

    elif seeds:
        seeds_set = {str(seed).strip() for seed in seeds if str(seed).strip()}
        if seeds_set and "id" in frame.columns:
            id_series = frame["id"].astype(str)
            positives = id_series.isin(seeds_set)
            positives_count = int(positives.sum())
            if 0 < positives_count < frame.shape[0]:
                pipeline = _build_pipeline(seed)
                y_train = positives.astype(int).to_numpy()
                try:
                    pipeline.fit(text_series, y_train)
                except ValueError as exc:
                    logger.warning("Unable to train logistic model with seeds: %s", exc)
                    pipeline = None
                else:
                    metadata["trained"] = True
                    metadata["strategy"] = "seeds"
                    probabilities = pipeline.predict_proba(text_series)[:, 1]
                    threshold = pick_threshold_for_recall(
                        y_train, probabilities, target_recall
                    )
            else:
                logger.info(
                    "Seed ids provided but none matched or negatives missing; skipping supervised training."
                )
        elif seeds_set:
            logger.info(
                "Seeds provided but 'id' column missing; skipping supervised training."
            )

    if pipeline is None:
        metadata.setdefault("strategy", "uniform")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    prob_series = pd.Series(probabilities, index=frame.index, name="probability")
    frame["probability"] = prob_series
    frame["label"] = np.where(prob_series > threshold, INCLUDED, EXCLUDED)

    # Set label to EXCLUDED for records with reasons
    frame.loc[frame["reasons"].str.len() > 0, "label"] = EXCLUDED

    # Add stats to metadata
    excluded_rules = int(frame["reasons"].str.len().gt(0).sum())
    metadata.update(
        {
            "excluded_rules": excluded_rules,
            "identified": len(frame),
            "screened": len(frame),
        }
    )

    return ScreeningResult(
        frame=frame, threshold=float(threshold), engine=engine_name, metadata=metadata
    )


def _build_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    C=0.5,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _combine_text_columns(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("title")
    if title is None:
        title_series = pd.Series("", index=frame.index)
    else:
        title_series = title.fillna("").astype(str)
    abstract = frame.get("abstract")
    if abstract is None:
        abstract_series = pd.Series("", index=frame.index)
    else:
        abstract_series = abstract.fillna("").astype(str)
    combined = title_series.str.strip().str.cat(abstract_series.str.strip(), sep=" ")
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()
    return combined


def _extract_gold_labels(frame: pd.DataFrame) -> pd.Series | None:
    if "gold_label" not in frame.columns:
        return None
    series = frame["gold_label"].dropna()
    if series.empty:
        return None
    try:
        converted = series.apply(_label_to_int)
    except ValueError as exc:
        logger.warning("Failed to interpret gold labels: %s", exc)
        return None
    if converted.nunique() < 2:
        logger.warning(
            "Gold labels contain a single class; skipping supervised training."
        )
        return None
    return converted


def _label_to_int(value: object) -> int:
    if value is None:
        raise ValueError("gold label cannot be None")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
        raise ValueError(f"unsupported gold label value: {value}")
    text = str(value).strip().lower()
    if text in _POSITIVE_LABELS:
        return 1
    if text in _NEGATIVE_LABELS:
        return 0
    raise ValueError(f"unsupported gold label value: {value}")
