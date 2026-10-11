from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def stsb_direct_score_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask for the original ordered rating on the STS 0-5 similarity scale using varied but semantically equivalent level descriptions.
    
        Target derivation: Target is original score level.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='score'; state={"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; instruction='Assign the appropriate semantic similarity level using this scale.'; options=["0","1","2","3","4","5"]; target=0.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_direct_score_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_exact_score_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one score level k and ask whether the item belongs exactly at level k.
    
        Target derivation: Positive iff original score == k.
        Rows used: 1. Selection: Balance exact-match vs non-match.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"input":{"sentence1":"Sentence A 0","sentence2":"Sentence B 0"},"proposed_level":0,"level_description":"level 0"}; instruction='Does this item belong exactly at score level 0?'; options=["correct","incorrect"]; target='correct'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_exact_score_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample threshold t and ask whether the score is at least t; separately generate at-most variants.
    
        Target derivation: For at-least: positive iff score >= t. For at-most: positive iff score <= t.
        Rows used: 1. Selection: Cover all nontrivial thresholds.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; instruction='Is the score at least 4?'; options=["meets the condition","does not meet the condition"]; target='does not meet the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_interval_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a contiguous interval [a,b] and ask whether the score lies within it.
    
        Target derivation: Positive iff a <= score <= b.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; instruction='Is the score within the inclusive interval [3, 5]?'; options=["inside the set","outside the set"]; target='outside the set'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_interval_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_coarse_bucket(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Randomly coarsen adjacent score levels into 2-4 named buckets and ask which bucket contains the item.
    
        Target derivation: Target is the bucket containing original score.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Buckets must be exhaustive and non-overlapping.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; instruction='Which temporary score bucket contains the correct score?'; options=["Bucket U","Group Beta","Set Two"]; target='Bucket U'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_coarse_bucket", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_pairwise_compare(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two sentence pairss and ask which has higher semantic similarity, with an equal option when labels are equal.
    
        Target derivation: Compare the two original scores.
        Rows used: 2. Selection: Include greater, less, and equal cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"sentence1":"Sentence A 0","sentence2":"Sentence B 0"},"second":{"sentence1":"Sentence A 3","sentence2":"Sentence B 3"}}; instruction='Which sentence pair has higher semantic similarity?'; options=["A","second","tie"]; target='second'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_pairwise_compare", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_same_score_or_band(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether two items have the same exact score or, in a separate variant, the same configured coarse band.
    
        Target derivation: Positive iff the compared score/band values are equal.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"sentence1":"Sentence A 0","sentence2":"Sentence B 0"},"second":{"sentence1":"Sentence A 8","sentence2":"Sentence B 8"}}; instruction='Do the two items have the same coarse band split at level 3?'; options=["equivalent","not equivalent"]; target='equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_same_score_or_band", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_score_difference_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the absolute difference between two scores is at least d.
    
        Target derivation: Positive iff abs(score1-score2) >= d.
        Rows used: 2. Selection: Use valid d values and balance outcomes.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"sentence1":"Sentence A 0","sentence2":"Sentence B 0"},"second":{"sentence1":"Sentence A 3","sentence2":"Sentence B 3"}}; instruction='Is the absolute difference between the two score levels at least 3?'; options=["meets the condition","does not meet the condition"]; target='meets the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_score_difference_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_other_score_subset(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert the score task to a choice over a subset of exact score levels plus OTHER. Omit the true level in OTHER-positive examples.
    
        Target derivation: Target is true score if offered, else OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; instruction='Choose the exact score if it is listed; otherwise choose the unlisted-score option.'; options=["1","2","3","some other category"]; target='some other category'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_other_score_subset", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required first or second sentence; offer score levels plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Only remove information required by the rating definition; do not call disagreement UNKNOWN.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"Sentence A 0","sentence2":"[UNAVAILABLE]"}; instruction='Rate only from the supplied input; choose insufficient information if a required field is missing.'; options=["0","1","2","3","4","5"]; target='cannot be determined'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use a subset of score levels plus OTHER and UNKNOWN, generating answer-present, answer-missing, and input-missing cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"Sentence A 0","sentence2":"[MISSING]"}; instruction='Distinguish an omitted score level from genuinely insufficient input.'; options=["0","1","5","some other category","unknown"]; target='unknown'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_opaque_bucket_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map temporary contiguous score buckets to opaque names with definitions supplied for that item.
    
        Target derivation: Target is opaque name of the bucket containing original score.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; instruction='Which temporary score bucket contains the correct score?'; options=["DR","Z4","HY"]; target='DR'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_opaque_bucket_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def stsb_reference_similarity_order(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one sentence pair as reference score and two candidate sentence pairs; ask which candidate has similarity closer to the reference score.
    
        Target derivation: Target is candidate with smaller absolute score difference; include equal-distance option on ties.
        Rows used: 3. Selection: Prefer non-tie examples unless intentionally sampling equality.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"Sentence A 0","sentence2":"Sentence B 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"reference":{"sentence1":"Sentence A 0","sentence2":"Sentence B 0"},"candidate_A":{"sentence1":"Sentence A 7","sentence2":"Sentence B 7"...; instruction='Which candidate sentence pair has a similarity score closer to the reference pair?'; options=["item A","B","no meaningful difference"]; target='B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("stsb_reference_similarity_order", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "stsb_direct_score_rephrase": stsb_direct_score_rephrase,
    "stsb_exact_score_verify": stsb_exact_score_verify,
    "stsb_threshold": stsb_threshold,
    "stsb_interval_membership": stsb_interval_membership,
    "stsb_coarse_bucket": stsb_coarse_bucket,
    "stsb_pairwise_compare": stsb_pairwise_compare,
    "stsb_same_score_or_band": stsb_same_score_or_band,
    "stsb_score_difference_threshold": stsb_score_difference_threshold,
    "stsb_other_score_subset": stsb_other_score_subset,
    "stsb_unknown_missing_required_input": stsb_unknown_missing_required_input,
    "stsb_other_unknown_joint": stsb_other_unknown_joint,
    "stsb_opaque_bucket_codebook": stsb_opaque_bucket_codebook,
    "stsb_reference_similarity_order": stsb_reference_similarity_order,
}
