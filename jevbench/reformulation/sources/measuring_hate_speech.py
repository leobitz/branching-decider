from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def measuring_hate_speech_direct_score_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask for the original ordered rating on the three-level hate-speech scale using varied but semantically equivalent level descriptions.
    
        Target derivation: Target is original score level.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='score'; state="Text 0 for measuring_hate_speech"; instruction='Assign the appropriate hate-speech severity level using this scale.'; options=["0","1","2"]; target=0.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_direct_score_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_exact_score_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one score level k and ask whether the item belongs exactly at level k.
    
        Target derivation: Positive iff original score == k.
        Rows used: 1. Selection: Balance exact-match vs non-match.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: If full score distribution exists, P(positive)=p(score=k).
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state={"input":"Text 0 for measuring_hate_speech","proposed_level":0,"level_description":"level 0"}; instruction='Does this item belong exactly at score level 0?'; options=["correct","incorrect"]; target='correct'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_exact_score_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample threshold t and ask whether the score is at least t; separately generate at-most variants.
    
        Target derivation: For at-least: positive iff score >= t. For at-most: positive iff score <= t.
        Rows used: 1. Selection: Cover all nontrivial thresholds.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: P(at_least_t)=sum_{k>=t} p(k); analogous for at-most.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for measuring_hate_speech"; instruction='Is the score at least 1?'; options=["meets the condition","does not meet the condition"]; target='does not meet the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_interval_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a contiguous interval [a,b] and ask whether the score lies within it.
    
        Target derivation: Positive iff a <= score <= b.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: P(positive)=sum_{a<=k<=b} p(k).
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for measuring_hate_speech"; instruction='Is the score within the inclusive interval [1, 2]?'; options=["inside the set","outside the set"]; target='outside the set'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_interval_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_coarse_bucket(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Randomly coarsen adjacent score levels into 2-4 named buckets and ask which bucket contains the item.
    
        Target derivation: Target is the bucket containing original score.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Buckets must be exhaustive and non-overlapping.
        Soft-label rule: Bucket probabilities are sums of fine-level probabilities.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for measuring_hate_speech"; instruction='Which temporary score bucket contains the correct score?'; options=["Bucket T","Bucket S","Bucket R"]; target='Bucket T'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_coarse_bucket", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_pairwise_compare(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two commentss and ask which has higher hate-speech severity, with an equal option when labels are equal.
    
        Target derivation: Compare the two original scores.
        Rows used: 2. Selection: Include greater, less, and equal cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Text 0 for measuring_hate_speech","second":"Text 4 for measuring_hate_speech"}; instruction='Which comment has higher hate-speech severity?'; options=["first item","second","tie"]; target='second'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_pairwise_compare", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_same_score_or_band(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether two items have the same exact score or, in a separate variant, the same configured coarse band.
    
        Target derivation: Positive iff the compared score/band values are equal.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Text 0 for measuring_hate_speech","second":"Text 3 for measuring_hate_speech"}; instruction='Do the two items have the same exact score?'; options=["same","different"]; target='same'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_same_score_or_band", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_score_difference_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the absolute difference between two scores is at least d.
    
        Target derivation: Positive iff abs(score1-score2) >= d.
        Rows used: 2. Selection: Use valid d values and balance outcomes.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: P(at_least_t)=sum_{k>=t} p(k); analogous for at-most.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Text 0 for measuring_hate_speech","second":"Text 5 for measuring_hate_speech"}; instruction='Is the absolute difference between the two score levels at least 2?'; options=["meets the condition","does not meet the condition"]; target='meets the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_score_difference_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_other_score_subset(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert the score task to a choice over a subset of exact score levels plus OTHER. Omit the true level in OTHER-positive examples.
    
        Target derivation: Target is true score if offered, else OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for measuring_hate_speech"; instruction='Choose the exact score if it is listed; otherwise choose the unlisted-score option.'; options=["0","1","2","other / unlisted"]; target='0'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_other_score_subset", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required comment text; offer score levels plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Only remove information required by the rating definition; do not call disagreement UNKNOWN.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state="[MISSING]"; instruction='Rate only from the supplied input; choose insufficient information if a required field is missing.'; options=["0","1","2","required information is missing"]; target='required information is missing'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use a subset of score levels plus OTHER and UNKNOWN, generating answer-present, answer-missing, and input-missing cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for measuring_hate_speech"; instruction='Distinguish an omitted score level from genuinely insufficient input.'; options=["1","2","an unlisted category","not sure"]; target='an unlisted category'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_opaque_bucket_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map temporary contiguous score buckets to opaque names with definitions supplied for that item.
    
        Target derivation: Target is opaque name of the bucket containing original score.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for measuring_hate_speech"; instruction='Which temporary score bucket contains the correct score?'; options=["AX","HY"]; target='AX'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_opaque_bucket_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_agreement_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the leading hate-speech level receives at least threshold t of annotator probability.
    
        Target derivation: Positive iff max soft probability >= t.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state="Text 0 for measuring_hate_speech"; instruction='Did the leading human label receive at least 60% of the annotation probability?'; options=["meets the condition","does not meet the condition"]; target='meets the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_agreement_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def measuring_hate_speech_compare_agreement(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Compare two comments by human agreement strength on their leading level, with an approximately-equal option.
    
        Target derivation: Compare max soft probabilities using a fixed tie epsilon.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Text 0 for measuring_hate_speech"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Text 0 for measuring_hate_speech","second":"Text 7 for measuring_hate_speech"}; instruction='Which item has stronger human agreement on its leading label?'; options=["first","item B","approximately equal"]; target='approximately equal'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("measuring_hate_speech_compare_agreement", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "measuring_hate_speech_direct_score_rephrase": measuring_hate_speech_direct_score_rephrase,
    "measuring_hate_speech_exact_score_verify": measuring_hate_speech_exact_score_verify,
    "measuring_hate_speech_threshold": measuring_hate_speech_threshold,
    "measuring_hate_speech_interval_membership": measuring_hate_speech_interval_membership,
    "measuring_hate_speech_coarse_bucket": measuring_hate_speech_coarse_bucket,
    "measuring_hate_speech_pairwise_compare": measuring_hate_speech_pairwise_compare,
    "measuring_hate_speech_same_score_or_band": measuring_hate_speech_same_score_or_band,
    "measuring_hate_speech_score_difference_threshold": measuring_hate_speech_score_difference_threshold,
    "measuring_hate_speech_other_score_subset": measuring_hate_speech_other_score_subset,
    "measuring_hate_speech_unknown_missing_required_input": measuring_hate_speech_unknown_missing_required_input,
    "measuring_hate_speech_other_unknown_joint": measuring_hate_speech_other_unknown_joint,
    "measuring_hate_speech_opaque_bucket_codebook": measuring_hate_speech_opaque_bucket_codebook,
    "measuring_hate_speech_agreement_threshold": measuring_hate_speech_agreement_threshold,
    "measuring_hate_speech_compare_agreement": measuring_hate_speech_compare_agreement,
}
