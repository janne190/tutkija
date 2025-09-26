"""Optional ASReview integration."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from .model import (
    DEFAULT_THRESHOLD,
    EXCLUDED,
    INCLUDED,
    ScreenStats,
    _to_binary_label,
    pick_threshold_for_recall,
)

logger = logging.getLogger(__name__)


def score_with_asreview(
    frame: pd.DataFrame,
    *,
    target_recall: float,
    seed: int,
    seeds: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, ScreenStats]:
    """Dispatch to ASReview engine when available, otherwise raise."""

    try:
        from asreview.data import ASReviewData
        from asreview.models.classifiers import LogisticClassifier
        from asreview.feature_extraction import Tfidf
        from asreview.samplers import RandomSampler
        from asreview.review import ReviewSimulate
    except ImportError as exc:
        raise RuntimeError(
            "ASReview support not installed. Install with 'pip install tutkija[asreview]' or 'uv pip install asreview'.",
        ) from exc

    logger.info("Using ASReview for screening.")

    # 1. Prepare data for ASReview
    text = (frame["title"].fillna("") + " " + frame["abstract"].fillna("")).tolist()
    as_data = ASReviewData(texts=text, record_ids=frame.index)

    # 2. Identify seed records
    prior_indices = []
    if seeds:
        seed_ids = set(seeds)
        prior_indices = frame[
            frame["id"].isin(seed_ids) | frame["doi"].isin(seed_ids)
        ].index.tolist()

    if not prior_indices and "gold_label" in frame.columns:
        gold_positives = frame[
            frame["gold_label"].apply(
                lambda x: str(x).lower() in {"1", "true", "included"}
            )
        ]
        if not gold_positives.empty:
            prior_indices = gold_positives.index.tolist()

    if not prior_indices:
        logger.warning(
            "ASReview is running without any seed documents. Performance may be suboptimal."
        )
        # Fallback to random sampling if no seeds are found
        sampler = RandomSampler()
    else:
        sampler = RandomSampler(n_prior_included=len(prior_indices), n_prior_excluded=0)

    # 3. Configure ASReview components
    classifier = LogisticClassifier(class_weight=1.0)  # ASReview handles balancing
    feature_extractor = Tfidf()

    # 4. Run simulation
    reviewer = ReviewSimulate(
        as_data,
        model=classifier,
        feature_model=feature_extractor,
        balance_model=None,  # Using default
        sampler=sampler,
        prior_indices=prior_indices,
        n_instances=1,  # We want to score all records
        state_file=None,  # No need to save state for this use case
    )

    reviewer.review()

    # 5. Extract results
    probabilities = reviewer.get_state().probas

    # The probabilities are ordered by the internal state, map them back
    ranked_record_ids = reviewer.get_state().get_order_of_labeling()
    prob_map = {rec_id: prob for rec_id, prob in zip(ranked_record_ids, probabilities)}

    frame["probability"] = frame.index.map(prob_map).fillna(0.0)

    # For now, we'll use the same thresholding logic as scikit-learn
    # A more advanced integration could use ASReview's stopping rules
    threshold = DEFAULT_THRESHOLD
    if "gold_label" in frame.columns and frame["gold_label"].notna().any():
        gold_set = frame[frame["gold_label"].notna()]
        y_true = gold_set["gold_label"].apply(_to_binary_label).values
        y_prob_gold = gold_set["probability"].values
        if y_true.sum() > 0:
            threshold = pick_threshold_for_recall(y_true, y_prob_gold, target_recall)

    frame["label"] = (frame["probability"] >= threshold).map(
        {True: INCLUDED, False: EXCLUDED}
    )

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
        "engine": "asreview",
        "recall_target": target_recall,
        "threshold_used": threshold,
        "seeds_count": len(prior_indices),
        "version": "1.0.0",
        "random_state": seed,
        "fallback": "default_prob_0.5" if len(prior_indices) == 0 else "",
    }
    return frame, stats
