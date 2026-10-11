from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def boolq_direct_binary_surface(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the original binary decision about whether the passage implies a yes answer, but sample from semantically compatible binary surface pairs.
    
        Target derivation: Target is original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; instruction='For this passage-question pair, is it true that the answer is yes?'; options=["yes","no"]; target='no'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_direct_binary_surface", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_inverted_binary_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the complement of the original binary decision, e.g. ask whether the answer is negative/refuted/non-matching instead of positive.
    
        Target derivation: Target is 1-original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; instruction='Is it false that the answer is yes?'; options=["Y","N"]; target='Y'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_inverted_binary_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_same_answer_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two passage-question pairss and ask whether their original binary decisions are the same.
    
        Target derivation: Positive iff label1 == label2.
        Rows used: 2. Selection: Balance same and different.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"},"second":{"passage":"Passage sentence one...; instruction='Do the two items have the same binary outcome?'; options=["same","different"]; target='same'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_same_answer_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_pair_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two passage-question pairss and ask which are positive under the original criterion: first only, second only, both, or neither.
    
        Target derivation: Derive from the two original binary labels.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"},"second":{"passage":"Passage sentence one...; instruction='Which of the two original propositions are positive?'; options=["the first but not the second","B only","A and B","neither matches"]; target='neither matches'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_pair_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_logical_and(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether both original propositions are positive.
    
        Target derivation: Positive iff label1=1 and label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"},"second":{"passage":"Passage sentence one...; instruction='Are both original propositions positive?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_logical_and", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_logical_or(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether at least one original proposition is positive.
    
        Target derivation: Positive iff label1=1 or label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"},"second":{"passage":"Passage sentence one...; instruction='Is at least one of the two original propositions positive?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_logical_or", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_logical_xor(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether exactly one original proposition is positive.
    
        Target derivation: Positive iff label1 != label2.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"},"second":{"passage":"Passage sentence one...; instruction='Is exactly one of the two original propositions positive?'; options=["true","false"]; target='true'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_logical_xor", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_other_target_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to a choice task; omit the true binary answer, retain the opposite answer, and add OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Original item remains answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["true","another answer"]; target='another answer'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_other_target_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_other_target_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to choice with both binary answers plus OTHER.
    
        Target derivation: Target is the original binary answer; OTHER is incorrect.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["false","true","none of the provided options"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_other_target_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required question or required passage; offer both binary answers plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Instruction must state that the decision is based on supplied information when outside knowledge could otherwise answer it.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"The required information was not supplied.","question":"Question 0?"}; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["yes","no","I don't know"]; target="I don't know".
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer binary answers plus OTHER and UNKNOWN and generate the three global joint cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["no","yes","no offered label fits","need more information"]; target='no'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_opaque_binary_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map positive and negative to two temporary opaque labels with definitions given in the question.
    
        Target derivation: Target is opaque label mapped to original binary answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; instruction='Use the temporary binary codebook and select the code matching the original decision.'; options=["DR","EV"]; target='DR'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_opaque_binary_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_irrelevant_passage_distractor(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Keep the original passage and question, append a clearly separated unrelated passage from another row as distractor context.
    
        Target derivation: Target remains original BoolQ label.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: Do not remove original passage.; Mark passages as separate sources.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?","unrelated_passage":{"passage":"Passage sentence on...; instruction='For this passage-question pair, is it true that the answer is yes?'; options=["yes","no"]; target='no'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_irrelevant_passage_distractor", row, pool=pool, seed=seed, validator=validator, **kwargs)

def boolq_partial_passage_ablation_optional(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove one or more passage sentences and include UNKNOWN.
    
        Target derivation: UNKNOWN only when a validator confirms the remaining passage no longer supports a unique yes/no answer under passage-only instructions.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: Yes: pass validator(transformed_row, original_row) -> bool.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"passage":"Passage sentence one 0. Passage sentence two 0.","question":"Question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"passage":"[UNAVAILABLE]","question":"Question 0?"}; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["1 - yes","0 - no","not knowable from what is provided"]; target='not knowable from what is provided'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("boolq_partial_passage_ablation_optional", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "boolq_direct_binary_surface": boolq_direct_binary_surface,
    "boolq_inverted_binary_question": boolq_inverted_binary_question,
    "boolq_same_answer_pair": boolq_same_answer_pair,
    "boolq_pair_truth_pattern": boolq_pair_truth_pattern,
    "boolq_logical_and": boolq_logical_and,
    "boolq_logical_or": boolq_logical_or,
    "boolq_logical_xor": boolq_logical_xor,
    "boolq_other_target_removed": boolq_other_target_removed,
    "boolq_other_target_present": boolq_other_target_present,
    "boolq_unknown_missing_required_input": boolq_unknown_missing_required_input,
    "boolq_other_unknown_joint": boolq_other_unknown_joint,
    "boolq_opaque_binary_codebook": boolq_opaque_binary_codebook,
    "boolq_irrelevant_passage_distractor": boolq_irrelevant_passage_distractor,
    "boolq_partial_passage_ablation_optional": boolq_partial_passage_ablation_optional,
}
