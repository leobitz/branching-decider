from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def helpsteer2_verbosity_direct_score_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask for the original ordered rating on the HelpSteer2 verbosity 0-4 scale using varied but semantically equivalent level descriptions.
    
        Target derivation: Target is original score level.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='score'; state={"prompt":"Prompt 0","response":"Response 0"}; instruction='Rate the verbosity relative to the request of this prompt-response pair on the ordered levels below.'; options=["0","1","2","3","4"]; target=0.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_direct_score_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_exact_score_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one score level k and ask whether the item belongs exactly at level k.
    
        Target derivation: Positive iff original score == k.
        Rows used: 1. Selection: Balance exact-match vs non-match.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"input":{"prompt":"Prompt 0","response":"Response 0"},"proposed_level":4,"level_description":"level 4"}; instruction='Does this item belong exactly at score level 4?'; options=["right","wrong"]; target='wrong'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_exact_score_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample threshold t and ask whether the score is at least t; separately generate at-most variants.
    
        Target derivation: For at-least: positive iff score >= t. For at-most: positive iff score <= t.
        Rows used: 1. Selection: Cover all nontrivial thresholds.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"prompt":"Prompt 0","response":"Response 0"}; instruction='Is the score at least 4?'; options=["meets the condition","does not meet the condition"]; target='does not meet the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_interval_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a contiguous interval [a,b] and ask whether the score lies within it.
    
        Target derivation: Positive iff a <= score <= b.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"prompt":"Prompt 0","response":"Response 0"}; instruction='Is the score within the inclusive interval [0, 4]?'; options=["inside the set","outside the set"]; target='inside the set'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_interval_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_coarse_bucket(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Randomly coarsen adjacent score levels into 2-4 named buckets and ask which bucket contains the item.
    
        Target derivation: Target is the bucket containing original score.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Buckets must be exhaustive and non-overlapping.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"prompt":"Prompt 0","response":"Response 0"}; instruction='Which temporary score bucket contains the correct score?'; options=["Set Two","Group Gamma"]; target='Set Two'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_coarse_bucket", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_pairwise_compare(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two prompt-response pairss and ask which has greater verbosity relative to the request, with an equal option when labels are equal.
    
        Target derivation: Compare the two original scores.
        Rows used: 2. Selection: Include greater, less, and equal cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"prompt":"Prompt 0","response":"Response 0"},"second":{"prompt":"Prompt 4","response":"Response 4"}}; instruction='Which prompt-response pair has higher verbosity relative to the request?'; options=["the first one","second","equal"]; target='second'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_pairwise_compare", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_same_score_or_band(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether two items have the same exact score or, in a separate variant, the same configured coarse band.
    
        Target derivation: Positive iff the compared score/band values are equal.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"prompt":"Prompt 0","response":"Response 0"},"second":{"prompt":"Prompt 6","response":"Response 6"}}; instruction='Do the two items have the same coarse band split at level 2?'; options=["equivalent","not equivalent"]; target='equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_same_score_or_band", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_score_difference_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the absolute difference between two scores is at least d.
    
        Target derivation: Positive iff abs(score1-score2) >= d.
        Rows used: 2. Selection: Use valid d values and balance outcomes.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"prompt":"Prompt 0","response":"Response 0"},"second":{"prompt":"Prompt 6","response":"Response 6"}}; instruction='Is the absolute difference between the two score levels at least 1?'; options=["meets the condition","does not meet the condition"]; target='meets the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_score_difference_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_other_score_subset(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert the score task to a choice over a subset of exact score levels plus OTHER. Omit the true level in OTHER-positive examples.
    
        Target derivation: Target is true score if offered, else OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"prompt":"Prompt 0","response":"Response 0"}; instruction='Choose the exact score if it is listed; otherwise choose the unlisted-score option.'; options=["2","3","4","not among these"]; target='not among these'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_other_score_subset", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required prompt or response; offer score levels plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Only remove information required by the rating definition; do not call disagreement UNKNOWN.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"prompt":"Prompt 0","response":"[MISSING]"}; instruction='Rate only from the supplied input; choose insufficient information if a required field is missing.'; options=["0","1","2","3","4","required information is missing"]; target='required information is missing'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use a subset of score levels plus OTHER and UNKNOWN, generating answer-present, answer-missing, and input-missing cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"prompt":"Prompt 0","response":"<missing>"}; instruction='Distinguish an omitted score level from genuinely insufficient input.'; options=["0","1","3","none of the provided labels","required information is missing"]; target='required information is missing'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def helpsteer2_verbosity_opaque_bucket_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map temporary contiguous score buckets to opaque names with definitions supplied for that item.
    
        Target derivation: Target is opaque name of the bucket containing original score.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"prompt":"Prompt 0","response":"Response 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"prompt":"Prompt 0","response":"Response 0"}; instruction='Which temporary score bucket contains the correct score?'; options=["FW","DR","KR"]; target='FW'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("helpsteer2_verbosity_opaque_bucket_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "helpsteer2_verbosity_direct_score_rephrase": helpsteer2_verbosity_direct_score_rephrase,
    "helpsteer2_verbosity_exact_score_verify": helpsteer2_verbosity_exact_score_verify,
    "helpsteer2_verbosity_threshold": helpsteer2_verbosity_threshold,
    "helpsteer2_verbosity_interval_membership": helpsteer2_verbosity_interval_membership,
    "helpsteer2_verbosity_coarse_bucket": helpsteer2_verbosity_coarse_bucket,
    "helpsteer2_verbosity_pairwise_compare": helpsteer2_verbosity_pairwise_compare,
    "helpsteer2_verbosity_same_score_or_band": helpsteer2_verbosity_same_score_or_band,
    "helpsteer2_verbosity_score_difference_threshold": helpsteer2_verbosity_score_difference_threshold,
    "helpsteer2_verbosity_other_score_subset": helpsteer2_verbosity_other_score_subset,
    "helpsteer2_verbosity_unknown_missing_required_input": helpsteer2_verbosity_unknown_missing_required_input,
    "helpsteer2_verbosity_other_unknown_joint": helpsteer2_verbosity_other_unknown_joint,
    "helpsteer2_verbosity_opaque_bucket_codebook": helpsteer2_verbosity_opaque_bucket_codebook,
}
