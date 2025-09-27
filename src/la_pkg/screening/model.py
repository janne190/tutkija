"""Model-based screening helpers."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, TypedDict

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import cosine_similarity

from la_pkg import __version__

logger = logging.getLogger(__name__)

INCLUDED = "included"
EXCLUDED = "excluded"
DEFAULT_THRESHOLD = 0.5
_POSITIVE_LABELS = {"included", "relevant", "positive", "1", "true", "yes"}
_NEGATIVE_LABELS = {"excluded", "irrelevant", "negative", "0", "false", "no"}


class ScreenStats(TypedDict):
    """Summary statistics emitted by the screening pipeline."""

    identified: int
    screened: int
    excluded_rules: int
    excluded_model: int
    included: int
    engine: str
    recall_target: float
    threshold_used: float
    seeds_count: int
    version: str
    random_state: int
    fallback: str
    out_path: str


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

    candidates = sorted(set(y_prob_arr.tolist()))
    for threshold in candidates:
        predictions = (y_prob_arr >= threshold).astype(int)
        recall = recall_score(y_true_arr, predictions, zero_division=0)
        if recall >= target_recall:
            return float(threshold)
    return float(DEFAULT_THRESHOLD)


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
        try:
            from .asreview_wrapper import score_with_asreview
        except ImportError as exc:  # pragma: no cover - import error path
            raise RuntimeError(
                "ASReview is not installed. Install with 'pip install tutkija[asreview]' or use --engine scikit."
            ) from exc
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
) -> tuple[pd.DataFrame, ScreenStats]:
    if "reasons" not in frame.columns:
        frame["reasons"] = [[] for _ in range(len(frame))]
    else:
        frame["reasons"] = frame["reasons"].apply(_ensure_reason_list)

    text_series = _combine_text_columns(frame)
    seeds_list = [seed_value.strip() for seed_value in (seeds or []) if seed_value and seed_value.strip()]
    probabilities = np.full(frame.shape[0], DEFAULT_THRESHOLD, dtype=float)
    threshold = float(DEFAULT_THRESHOLD)
    fallback = "default_prob_0.5"

    if frame.empty:
        stats = _build_stats(
            frame,
            engine=engine_name,
            recall_target=target_recall,
            threshold=threshold,
            seeds_count=len(seeds_list),
            fallback=fallback,
            random_state=seed,
        )
        frame["probability"] = pd.Series(dtype=float)
        frame["label"] = pd.Series(dtype=str)
        return frame, stats

    gold_labels = _extract_gold_labels(frame)

    if gold_labels is not None and not gold_labels.empty:
        try:
            pipeline = _build_pipeline(seed)
            train_text = text_series.loc[gold_labels.index]
            pipeline.fit(train_text, gold_labels.to_numpy())
            probabilities = pipeline.predict_proba(text_series)[:, 1]
            train_probs = pipeline.predict_proba(train_text)[:, 1]
            threshold = pick_threshold_for_recall(
                gold_labels.to_numpy(), train_probs, target_recall
            )
            fallback = "model"
        except (ValueError, NotFittedError) as exc:
            logger.warning("Unable to train logistic model with gold labels: %s", exc)

    elif seeds_list:
        vectorizer = _build_vectorizer()
        matrix = vectorizer.fit_transform(text_series)
        seed_indices = _match_seed_indices(frame, seeds_list)
        if seed_indices:
            seed_matrix = matrix[seed_indices]
            similarities = cosine_similarity(matrix, seed_matrix)
            max_similarity = similarities.max(axis=1)
            probabilities = np.clip(0.5 + 0.5 * max_similarity, 0.0, 1.0)
            fallback = "seed_similarity"
        else:
            fallback = "default_prob_0.5"

    probabilities = np.clip(probabilities, 0.0, 1.0)
    prob_series = pd.Series(probabilities, index=frame.index, name="probability")
    frame["probability"] = prob_series
    frame["label"] = np.where(prob_series > threshold, INCLUDED, EXCLUDED)

    has_reasons = frame["reasons"].apply(len).gt(0)
    frame.loc[has_reasons, "label"] = EXCLUDED
    frame.loc[frame["label"].eq(INCLUDED), "reasons"] = frame.loc[
        frame["label"].eq(INCLUDED), "reasons"
    ].apply(lambda _: [])

    stats = _build_stats(
        frame,
        engine=engine_name,
        recall_target=target_recall,
        threshold=threshold,
        seeds_count=len(seeds_list),
        fallback=fallback,
        random_state=seed,
    )
    logger.info(
        "screened=%s included=%s excluded_rules=%s fallback=%s threshold=%.3f engine=%s",
        stats["screened"],
        stats["included"],
        stats["excluded_rules"],
        stats["fallback"],
        stats["threshold_used"],
        stats["engine"],
    )
    return frame, stats


def _build_pipeline(seed: int) -> LogisticRegressionPipeline:
    vectorizer = _build_vectorizer()
    classifier = LogisticRegression(
        solver="liblinear",
        penalty="l1",
        C=0.5,
        random_state=seed,
        class_weight="balanced",
    )
    return LogisticRegressionPipeline(vectorizer=vectorizer, classifier=classifier)


class LogisticRegressionPipeline:
    """Lightweight pipeline to couple TF-IDF and logistic regression."""

    def __init__(self, *, vectorizer: TfidfVectorizer, classifier: LogisticRegression) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self._is_fitted = False

    def fit(self, text: pd.Series, y: np.ndarray) -> None:
        features = self.vectorizer.fit_transform(text)
        self.classifier.fit(features, y)
        self._is_fitted = True

    def predict_proba(self, text: pd.Series) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("Pipeline must be fitted before predicting")
        features = self.vectorizer.transform(text)
        return self.classifier.predict_proba(features)


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_df=0.9, stop_words="english"
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


def _ensure_reason_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if hasattr(value, "tolist"):
        try:
            return [
                str(item)
                for item in value.tolist()  # type: ignore[call-arg]
                if str(item).strip()
            ]
        except Exception:  # pragma: no cover - defensive
            pass
    if value in (None, ""):
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    return [str(value)] if str(value).strip() else []


def _match_seed_indices(frame: pd.DataFrame, seeds: Sequence[str]) -> list[int]:
    if not seeds:
        return []

    indices: set[int] = set()
    for seed in seeds:
        key, value = _split_seed(seed)
        if value is None:
            continue
        columns = _seed_columns_for_key(key)
        for column in columns:
            if column not in frame.columns:
                continue
            matches = frame[column].fillna("").astype(str).str.strip().str.lower()
            mask = matches == value.lower()
            if mask.any():
                positions = np.flatnonzero(mask.to_numpy())
                indices.update(int(pos) for pos in positions)
    return sorted(indices)


def _split_seed(seed: str) -> tuple[str | None, str | None]:
    if ":" in seed:
        prefix, value = seed.split(":", 1)
        prefix = prefix.strip().lower()
        value = value.strip()
        if not value:
            return None, None
        return prefix or None, value
    cleaned = seed.strip()
    if not cleaned:
        return None, None
    return None, cleaned


def _seed_columns_for_key(key: str | None) -> Iterable[str]:
    if key is None:
        return ("id",)
    mapping = {
        "id": ("id",),
        "doi": ("doi",),
        "pmid": ("pmid", "pubmed_id"),
        "pmcid": ("pmcid", "pmc_id"),
    }
    return mapping.get(key, ("id",))


def _build_stats(
    frame: pd.DataFrame,
    *,
    engine: str,
    recall_target: float,
    threshold: float,
    seeds_count: int,
    fallback: str,
    random_state: int,
) -> ScreenStats:
    reasons_len = (
        frame["reasons"].apply(len)
        if "reasons" in frame.columns
        else pd.Series(0, index=frame.index, dtype=int)
    )
    excluded_rules = int(reasons_len.gt(0).sum())
    labels = frame.get("label", pd.Series(dtype=str))
    excluded_model = (
        int(((labels == EXCLUDED) & (reasons_len == 0)).sum()) if not labels.empty else 0
    )
    included = int(labels.eq(INCLUDED).sum()) if not labels.empty else 0

    return ScreenStats(
        identified=int(frame.shape[0]),
        screened=int(frame.shape[0]),
        excluded_rules=excluded_rules,
        excluded_model=excluded_model,
        included=included,
        engine=engine,
        recall_target=float(recall_target),
        threshold_used=float(threshold),
        seeds_count=int(seeds_count),
        version=__version__,
        random_state=int(random_state),
        fallback=fallback,
        out_path="",
    )
