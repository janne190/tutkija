# 0003. Screening engine architecture

Date: 2025-09-26

## Status
Accepted

## Context
A document screening workflow needs to balance two requirements:
1. Reliable identification of relevant papers (measured as recall)
2. Reduction of manual review workload (achieved via filtering and ranking)

The primary design questions were:
- Choice of ML engine(s)
- Definition of "included" vs "excluded"
- Rules vs Model division
- Treatment of reasons for exclusion

## Decision

1. Two-stage pipeline architecture:
   - Stage 1: Rule-based pre-screening (language, year, document type)
   - Stage 2: Model-based screening (ranking/classification)

2. Built-in scikit-learn baseline using:
   - TF-IDF vectorization for text features
   - Balanced logistic regression for binary classification
   - Threshold tuning to achieve target recall
   - Default to 0.5 probability when no training data

3. Optional ASReview integration:
   - Available via `--engine asreview`
   - Must be installed separately (`pip install tutkija[asreview]`)
   - Uses ASReview's default components (features, classifier)

4. Consistent `reasons` handling:
   - Empty list (`[]`) for included papers
   - Non-empty list explains exclusions
   - Rules add reasons before model step
   - Runtime validation ensures consistency

## Consequences

### Positive
- Clear separation between rule-based and model-based screening
- Simple default workflow with scikit-learn
- Transparent exclusion tracking via reasons
- Optional integration with research tools (ASReview)

### Negative
- Two engines may give different results
- Training data quality impacts ranking
- Default threshold may need tuning

### Neutral
- Rules are conservative to avoid false negatives
- Recall target forces wider inclusion net
- Seeds can improve but not perfect

## Implementation Notes
The stages are reflected in key files:
- `src/la_pkg/screening/rules.py`: Pre-screening rules
- `src/la_pkg/screening/model.py`: ML engine abstraction
- `src/la_pkg/screening/asreview_wrapper.py`: Optional ASReview support

Guard-rails:
- Audit script verifies metrics CSV
- Nightly tests verify workflow
- Pre-commit enforces types
