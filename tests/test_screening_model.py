"""Model screening tests."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from la_pkg.screening import score_and_label


def test_scikit_model_reaches_target_recall_and_auc() -> None:
    positive_texts = [
        "genomic screening improves cancer survival",
        "precision oncology screening trial results",
        "cancer screening detects early tumor cells",
        "deep learning model for breast cancer screening",
        "lung cancer genomic screening outcomes",
    ]
    negative_texts = [
        "agricultural yield optimization in wheat",
        "renewable energy grid storage analysis",
        "urban transportation planning logistics",
        "financial risk modelling for markets",
        "educational policy reform review",
        "quantum computing hardware advances",
        "astronomy telescope calibration techniques",
        "marine biology coral reef survey",
        "meteorological climate prediction systems",
        "supply chain operations research",
        "cybersecurity intrusion detection report",
        "robotics path planning algorithms",
        "materials science alloy characterization",
        "soil microbiome diversity mapping",
        "aerospace propulsion efficiency study",
    ]

    records: list[dict[str, object]] = []
    for idx, text in enumerate(positive_texts, start=1):
        records.append(
            {
                "id": f"P{idx}",
                "title": f"Cancer screening study {idx}",
                "abstract": text,
                "gold_label": "included",
                "reasons": [],
            }
        )
    for idx, text in enumerate(negative_texts, start=len(positive_texts) + 1):
        records.append(
            {
                "id": f"P{idx}",
                "title": f"Background research {idx}",
                "abstract": text,
                "gold_label": "excluded",
                "reasons": [],
            }
        )

    df = pd.DataFrame.from_records(records)
    result = score_and_label(df, target_recall=0.9, seed=11)
    scored = result.frame

    y_true = scored["gold_label"].eq("included").astype(int)
    y_prob = scored["probability"]
    y_pred = scored["label"].eq("included").astype(int)

    assert recall_score(y_true, y_pred) >= 0.9
    assert roc_auc_score(y_true, y_prob) >= 0.8
    assert 0.0 <= result.threshold <= 1.0
    assert result.metadata.get("strategy") == "gold"
