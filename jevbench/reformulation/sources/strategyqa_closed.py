from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def strategyqa_closed_direct_binary_surface(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the original binary decision about whether the answer to the world-knowledge question is yes, but sample from semantically compatible binary surface pairs.
    
        Target derivation: Target is original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"World question 0?"}; instruction='For this StrategyQA question, is it true that the answer is yes?'; options=["Y","N"]; target='N'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_direct_binary_surface", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_inverted_binary_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the complement of the original binary decision, e.g. ask whether the answer is negative/refuted/non-matching instead of positive.
    
        Target derivation: Target is 1-original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"World question 0?"}; instruction='Is it false that the answer is yes?'; options=["affirmative","negative"]; target='affirmative'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_inverted_binary_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_same_answer_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two StrategyQA questionss and ask whether their original binary decisions are the same.
    
        Target derivation: Positive iff label1 == label2.
        Rows used: 2. Selection: Balance same and different.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"World question 0?"},"second":{"question":"World question 6?"}}; instruction='Do the two items have the same binary outcome?'; options=["same","different"]; target='same'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_same_answer_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_pair_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two StrategyQA questionss and ask which are positive under the original criterion: first only, second only, both, or neither.
    
        Target derivation: Derive from the two original binary labels.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"World question 0?"},"second":{"question":"World question 10?"}}; instruction='Which of the two original propositions are positive?'; options=["only the first","only the second","both","none of the two"]; target='none of the two'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_pair_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_logical_and(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether both original propositions are positive.
    
        Target derivation: Positive iff label1=1 and label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"World question 0?"},"second":{"question":"World question 4?"}}; instruction='Are both original propositions positive?'; options=["T","F"]; target='F'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_logical_and", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_logical_or(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether at least one original proposition is positive.
    
        Target derivation: Positive iff label1=1 or label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"World question 0?"},"second":{"question":"World question 6?"}}; instruction='Is at least one of the two original propositions positive?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_logical_or", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_logical_xor(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether exactly one original proposition is positive.
    
        Target derivation: Positive iff label1 != label2.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"World question 0?"},"second":{"question":"World question 5?"}}; instruction='Is exactly one of the two original propositions positive?'; options=["true","false"]; target='true'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_logical_xor", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_other_target_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to a choice task; omit the true binary answer, retain the opposite answer, and add OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Original item remains answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"World question 0?"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["Y","the correct answer is not included"]; target='the correct answer is not included'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_other_target_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_other_target_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to choice with both binary answers plus OTHER.
    
        Target derivation: Target is the original binary answer; OTHER is incorrect.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"World question 0?"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["0 - no","1 - yes","a choice not shown"]; target='0 - no'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_other_target_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required question text; offer both binary answers plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Instruction must state that the decision is based on supplied information when outside knowledge could otherwise answer it.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"[MISSING]"}; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["Y","N","need more information"]; target='need more information'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer binary answers plus OTHER and UNKNOWN and generate the three global joint cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"World question 0?"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["affirmative","something else","not established"]; target='something else'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_opaque_binary_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map positive and negative to two temporary opaque labels with definitions given in the question.
    
        Target derivation: Target is opaque label mapped to original binary answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"World question 0?"}; instruction='Use the temporary binary codebook and select the code matching the original decision.'; options=["HY","BZ"]; target='HY'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_opaque_binary_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_closed_two_question_meta(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two independent StrategyQA questions and ask whether yes is correct for first only, second only, both, or neither.
    
        Target derivation: Derive from the two gold binary labels.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"World question 0?"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"World question 0?"},"second":{"question":"World question 2?"}}; instruction='Which of the two original propositions are positive?'; options=["the first but not the second","only the second","both are true","neither is true"]; target='neither is true'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_closed_two_question_meta", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "strategyqa_closed_direct_binary_surface": strategyqa_closed_direct_binary_surface,
    "strategyqa_closed_inverted_binary_question": strategyqa_closed_inverted_binary_question,
    "strategyqa_closed_same_answer_pair": strategyqa_closed_same_answer_pair,
    "strategyqa_closed_pair_truth_pattern": strategyqa_closed_pair_truth_pattern,
    "strategyqa_closed_logical_and": strategyqa_closed_logical_and,
    "strategyqa_closed_logical_or": strategyqa_closed_logical_or,
    "strategyqa_closed_logical_xor": strategyqa_closed_logical_xor,
    "strategyqa_closed_other_target_removed": strategyqa_closed_other_target_removed,
    "strategyqa_closed_other_target_present": strategyqa_closed_other_target_present,
    "strategyqa_closed_unknown_missing_required_input": strategyqa_closed_unknown_missing_required_input,
    "strategyqa_closed_other_unknown_joint": strategyqa_closed_other_unknown_joint,
    "strategyqa_closed_opaque_binary_codebook": strategyqa_closed_opaque_binary_codebook,
    "strategyqa_closed_two_question_meta": strategyqa_closed_two_question_meta,
}
