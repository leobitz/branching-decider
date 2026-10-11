from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def sst5_direct_score_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask for the original ordered rating on the five-level sentiment scale using varied but semantically equivalent level descriptions.
    
        Target derivation: Target is original score level.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='score'; state="Text 0 for sst5"; instruction='Rate the sentiment of this review sentence on the ordered levels below.'; options=["0","1","2","3","4"]; target=0.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_direct_score_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_exact_score_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one score level k and ask whether the item belongs exactly at level k.
    
        Target derivation: Positive iff original score == k.
        Rows used: 1. Selection: Balance exact-match vs non-match.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state={"input":"Text 0 for sst5","proposed_level":1,"level_description":"level 1"}; instruction='Does this item belong exactly at score level 1?'; options=["right","wrong"]; target='wrong'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_exact_score_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample threshold t and ask whether the score is at least t; separately generate at-most variants.
    
        Target derivation: For at-least: positive iff score >= t. For at-most: positive iff score <= t.
        Rows used: 1. Selection: Cover all nontrivial thresholds.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for sst5"; instruction='Is the score at most 1?'; options=["meets the condition","does not meet the condition"]; target='meets the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_interval_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a contiguous interval [a,b] and ask whether the score lies within it.
    
        Target derivation: Positive iff a <= score <= b.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for sst5"; instruction='Is the score within the inclusive interval [0, 3]?'; options=["included","not included"]; target='included'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_interval_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_coarse_bucket(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Randomly coarsen adjacent score levels into 2-4 named buckets and ask which bucket contains the item.
    
        Target derivation: Target is the bucket containing original score.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Buckets must be exhaustive and non-overlapping.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for sst5"; instruction='Which temporary score bucket contains the correct score?'; options=["Group Beta","Group Delta"]; target='Group Beta'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_coarse_bucket", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_pairwise_compare(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two review sentencess and ask which has more positive sentiment, with an equal option when labels are equal.
    
        Target derivation: Compare the two original scores.
        Rows used: 2. Selection: Include greater, less, and equal cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Text 0 for sst5","second":"Text 8 for sst5"}; instruction='Which review sentence has higher sentiment?'; options=["first item","B","same"]; target='B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_pairwise_compare", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_same_score_or_band(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether two items have the same exact score or, in a separate variant, the same configured coarse band.
    
        Target derivation: Positive iff the compared score/band values are equal.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Text 0 for sst5","second":"Text 1 for sst5"}; instruction='Do the two items have the same coarse band split at level 2?'; options=["same","different"]; target='same'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_same_score_or_band", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_score_difference_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the absolute difference between two scores is at least d.
    
        Target derivation: Positive iff abs(score1-score2) >= d.
        Rows used: 2. Selection: Use valid d values and balance outcomes.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Text 0 for sst5","second":"Text 6 for sst5"}; instruction='Is the absolute difference between the two score levels at least 1?'; options=["meets the condition","does not meet the condition"]; target='meets the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_score_difference_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_other_score_subset(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert the score task to a choice over a subset of exact score levels plus OTHER. Omit the true level in OTHER-positive examples.
    
        Target derivation: Target is true score if offered, else OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for sst5"; instruction='Choose the exact score if it is listed; otherwise choose the unlisted-score option.'; options=["0","1","4","an unlisted option"]; target='0'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_other_score_subset", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required review sentence; offer score levels plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Only remove information required by the rating definition; do not call disagreement UNKNOWN.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state="The required information was not supplied."; instruction='Rate only from the supplied input; choose insufficient information if a required field is missing.'; options=["0","1","2","3","4","need more information"]; target='need more information'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use a subset of score levels plus OTHER and UNKNOWN, generating answer-present, answer-missing, and input-missing cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for sst5"; instruction='Distinguish an omitted score level from genuinely insufficient input.'; options=["0","1","3","none of the provided labels","more information is needed"]; target='0'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sst5_opaque_bucket_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map temporary contiguous score buckets to opaque names with definitions supplied for that item.
    
        Target derivation: Target is opaque name of the bucket containing original score.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for sst5"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for sst5"; instruction='Which temporary score bucket contains the correct score?'; options=["AX","CQ"]; target='AX'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sst5_opaque_bucket_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "sst5_direct_score_rephrase": sst5_direct_score_rephrase,
    "sst5_exact_score_verify": sst5_exact_score_verify,
    "sst5_threshold": sst5_threshold,
    "sst5_interval_membership": sst5_interval_membership,
    "sst5_coarse_bucket": sst5_coarse_bucket,
    "sst5_pairwise_compare": sst5_pairwise_compare,
    "sst5_same_score_or_band": sst5_same_score_or_band,
    "sst5_score_difference_threshold": sst5_score_difference_threshold,
    "sst5_other_score_subset": sst5_other_score_subset,
    "sst5_unknown_missing_required_input": sst5_unknown_missing_required_input,
    "sst5_other_unknown_joint": sst5_other_unknown_joint,
    "sst5_opaque_bucket_codebook": sst5_opaque_bucket_codebook,
}
