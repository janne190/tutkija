"""Optional ASReview integration."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from .model import ScreenStats, _score_with_scikit

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
        import asreview  # type: ignore[import-not-found, unused-import]
    except ImportError as exc:  # pragma: no cover - exercised when ASReview missing
        raise RuntimeError(
            "ASReview is not installed. Install with 'pip install tutkija[asreview]' or use --engine scikit."
        ) from exc

    logger.info("ASReview >= %s detected", getattr(asreview, "__version__", "unknown"))
    # For now reuse the batch logistic baseline while we validate integration points.
    return _score_with_scikit(
        frame,
        target_recall=target_recall,
        seed=seed,
        seeds=seeds,
        engine_name="asreview",
    )
