from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def civil_comments_direct_binary_surface(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the original binary decision about whether a comment is toxic, but sample from semantically compatible binary surface pairs.
    
        Target derivation: Target is original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: Preserve original P(positive) and map to the chosen positive/negative surfaces.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for civil_comments"; instruction='For this comment, is it true that the comment is toxic?'; options=["affirmative","negative"]; target='negative'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_direct_binary_surface", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_inverted_binary_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the complement of the original binary decision, e.g. ask whether the answer is negative/refuted/non-matching instead of positive.
    
        Target derivation: Target is 1-original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: P(new positive)=1-P(original positive).
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for civil_comments"; instruction='Is it false that the comment is toxic?'; options=["yes","no"]; target='yes'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_inverted_binary_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_same_answer_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two commentss and ask whether their original binary decisions are the same.
    
        Target derivation: Positive iff label1 == label2.
        Rows used: 2. Selection: Balance same and different.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for civil_comments","second":"Message 9 for civil_comments"}; instruction='Do the two items have the same binary outcome?'; options=["equivalent","not equivalent"]; target='not equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_same_answer_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_pair_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two commentss and ask which are positive under the original criterion: first only, second only, both, or neither.
    
        Target derivation: Derive from the two original binary labels.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for civil_comments","second":"Message 4 for civil_comments"}; instruction='Which of the two original propositions are positive?'; options=["only the first","B only","A and B","none of the two"]; target='none of the two'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_pair_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_logical_and(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether both original propositions are positive.
    
        Target derivation: Positive iff label1=1 and label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: Only under explicit independent-rater framing: p=p1*p2.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for civil_comments","second":"Message 6 for civil_comments"}; instruction='Are both original propositions positive?'; options=["T","F"]; target='F'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_logical_and", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_logical_or(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether at least one original proposition is positive.
    
        Target derivation: Positive iff label1=1 or label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: Only under explicit independent-rater framing: p=p1+p2-p1*p2.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for civil_comments","second":"Message 8 for civil_comments"}; instruction='Is at least one of the two original propositions positive?'; options=["T","F"]; target='F'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_logical_or", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_logical_xor(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether exactly one original proposition is positive.
    
        Target derivation: Positive iff label1 != label2.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: Only under explicit independent-rater framing: p=p1*(1-p2)+(1-p1)*p2.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for civil_comments","second":"Message 5 for civil_comments"}; instruction='Is exactly one of the two original propositions positive?'; options=["true","false"]; target='true'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_logical_xor", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_other_target_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to a choice task; omit the true binary answer, retain the opposite answer, and add OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Original item remains answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for civil_comments"; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["true","another option"]; target='another option'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_other_target_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_other_target_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to choice with both binary answers plus OTHER.
    
        Target derivation: Target is the original binary answer; OTHER is incorrect.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for civil_comments"; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["negative","affirmative","an unlisted answer"]; target='negative'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_other_target_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required comment text; offer both binary answers plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Instruction must state that the decision is based on supplied information when outside knowledge could otherwise answer it.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state="[MISSING]"; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["yes","no","no basis to decide"]; target='no basis to decide'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer binary answers plus OTHER and UNKNOWN and generate the three global joint cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state="<missing>"; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["yes","other","there is not enough to decide"]; target='there is not enough to decide'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_opaque_binary_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map positive and negative to two temporary opaque labels with definitions given in the question.
    
        Target derivation: Target is opaque label mapped to original binary answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for civil_comments"; instruction='Use the temporary binary codebook and select the code matching the original decision.'; options=["FW","CQ"]; target='FW'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_opaque_binary_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_toxicity_share_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether at least threshold t of annotators judged the comment toxic.
    
        Target derivation: Positive iff soft toxic share >= t.
        Rows used: 1. Selection: Use several thresholds and balance outcomes.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for civil_comments"; instruction='Did at least 20% of annotators judge the comment toxic?'; options=["meets the condition","does not meet the condition"]; target='does not meet the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_toxicity_share_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def civil_comments_compare_toxicity_share(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask which of two comments received the higher toxic vote share, with approximately-equal option.
    
        Target derivation: Compare soft positive shares with fixed tie epsilon.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for civil_comments"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for civil_comments","second":"Message 6 for civil_comments"}; instruction='Which comment received the higher toxic vote share?'; options=["first","second item","equal"]; target='equal'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("civil_comments_compare_toxicity_share", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "civil_comments_direct_binary_surface": civil_comments_direct_binary_surface,
    "civil_comments_inverted_binary_question": civil_comments_inverted_binary_question,
    "civil_comments_same_answer_pair": civil_comments_same_answer_pair,
    "civil_comments_pair_truth_pattern": civil_comments_pair_truth_pattern,
    "civil_comments_logical_and": civil_comments_logical_and,
    "civil_comments_logical_or": civil_comments_logical_or,
    "civil_comments_logical_xor": civil_comments_logical_xor,
    "civil_comments_other_target_removed": civil_comments_other_target_removed,
    "civil_comments_other_target_present": civil_comments_other_target_present,
    "civil_comments_unknown_missing_required_input": civil_comments_unknown_missing_required_input,
    "civil_comments_other_unknown_joint": civil_comments_other_unknown_joint,
    "civil_comments_opaque_binary_codebook": civil_comments_opaque_binary_codebook,
    "civil_comments_toxicity_share_threshold": civil_comments_toxicity_share_threshold,
    "civil_comments_compare_toxicity_share": civil_comments_compare_toxicity_share,
}
