from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def strategyqa_grounded_direct_binary_surface(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the original binary decision about whether the answer is yes given the supplied supporting facts, but sample from semantically compatible binary surface pairs.
    
        Target derivation: Target is original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; instruction='For this grounded StrategyQA item, is it true that the answer is yes from the supplied facts?'; options=["Y","N"]; target='N'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_direct_binary_surface", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_inverted_binary_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the complement of the original binary decision, e.g. ask whether the answer is negative/refuted/non-matching instead of positive.
    
        Target derivation: Target is 1-original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; instruction='Is it false that the answer is yes from the supplied facts?'; options=["1 - yes","0 - no"]; target='1 - yes'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_inverted_binary_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_same_answer_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two grounded StrategyQA question-fact bundless and ask whether their original binary decisions are the same.
    
        Target derivation: Positive iff label1 == label2.
        Rows used: 2. Selection: Balance same and different.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]},"second":{"question":"Grounded question 7?","facts":["Fact 7...; instruction='Do the two items have the same binary outcome?'; options=["same","different"]; target='different'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_same_answer_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_pair_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two grounded StrategyQA question-fact bundless and ask which are positive under the original criterion: first only, second only, both, or neither.
    
        Target derivation: Derive from the two original binary labels.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]},"second":{"question":"Grounded question 8?","facts":["Fact 8...; instruction='Which of the two original propositions are positive?'; options=["A only","B only","both apply","neither"]; target='neither'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_pair_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_logical_and(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether both original propositions are positive.
    
        Target derivation: Positive iff label1=1 and label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]},"second":{"question":"Grounded question 9?","facts":["Fact 9...; instruction='Are both original propositions positive?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_logical_and", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_logical_or(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether at least one original proposition is positive.
    
        Target derivation: Positive iff label1=1 or label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]},"second":{"question":"Grounded question 8?","facts":["Fact 8...; instruction='Is at least one of the two original propositions positive?'; options=["T","F"]; target='F'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_logical_or", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_logical_xor(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether exactly one original proposition is positive.
    
        Target derivation: Positive iff label1 != label2.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]},"second":{"question":"Grounded question 10?","facts":["Fact ...; instruction='Is exactly one of the two original propositions positive?'; options=["T","F"]; target='F'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_logical_xor", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_other_target_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to a choice task; omit the true binary answer, retain the opposite answer, and add OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Original item remains answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["true","none of the provided labels"]; target='none of the provided labels'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_other_target_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_other_target_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to choice with both binary answers plus OTHER.
    
        Target derivation: Target is the original binary answer; OTHER is incorrect.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["negative","affirmative","none of the listed choices"]; target='negative'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_other_target_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required question or supporting facts; offer both binary answers plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Instruction must state that the decision is based on supplied information when outside knowledge could otherwise answer it.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":"<missing>"}; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["true","false","answer cannot be inferred"]; target='answer cannot be inferred'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer binary answers plus OTHER and UNKNOWN and generate the three global joint cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":"<missing>"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["Y","no offered label fits","not knowable from what is provided"]; target='not knowable from what is provided'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_opaque_binary_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map positive and negative to two temporary opaque labels with definitions given in the question.
    
        Target derivation: Target is opaque label mapped to original binary answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; instruction='Use the temporary binary codebook and select the code matching the original decision.'; options=["V4","GX"]; target='V4'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_opaque_binary_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_remove_all_facts(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove all supporting facts and explicitly require answering only from supplied facts; offer yes/no/UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":[]}; instruction='Based only on the supplied facts, is the answer yes? Choose insufficient information when the facts are absent.'; options=["true","false","insufficient evidence"]; target='insufficient evidence'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_remove_all_facts", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_add_distractor_facts(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Keep all original supporting facts and append clearly separated unrelated facts from another row.
    
        Target derivation: Target remains original answer.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: Retain all original supporting facts.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"],"unrelated_facts":["Fact 9 a","Fact 9 b"]}; instruction='Given the original supplied facts, is the answer to the original question yes? Ignore the separately marked unrelated facts.'; options=["1 - yes","0 - no"]; target='0 - no'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_add_distractor_facts", row, pool=pool, seed=seed, validator=validator, **kwargs)

def strategyqa_grounded_partial_fact_ablation_optional(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove one or more supporting facts and include UNKNOWN.
    
        Target derivation: UNKNOWN only if validator confirms remaining facts are insufficient under fact-only instructions.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: Yes: pass validator(transformed_row, original_row) -> bool.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Grounded question 0?","facts":["Fact 0 a","Fact 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"question":"Grounded question 0?","facts":"[UNAVAILABLE]"}; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["yes","no","not sure"]; target='not sure'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("strategyqa_grounded_partial_fact_ablation_optional", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "strategyqa_grounded_direct_binary_surface": strategyqa_grounded_direct_binary_surface,
    "strategyqa_grounded_inverted_binary_question": strategyqa_grounded_inverted_binary_question,
    "strategyqa_grounded_same_answer_pair": strategyqa_grounded_same_answer_pair,
    "strategyqa_grounded_pair_truth_pattern": strategyqa_grounded_pair_truth_pattern,
    "strategyqa_grounded_logical_and": strategyqa_grounded_logical_and,
    "strategyqa_grounded_logical_or": strategyqa_grounded_logical_or,
    "strategyqa_grounded_logical_xor": strategyqa_grounded_logical_xor,
    "strategyqa_grounded_other_target_removed": strategyqa_grounded_other_target_removed,
    "strategyqa_grounded_other_target_present": strategyqa_grounded_other_target_present,
    "strategyqa_grounded_unknown_missing_required_input": strategyqa_grounded_unknown_missing_required_input,
    "strategyqa_grounded_other_unknown_joint": strategyqa_grounded_other_unknown_joint,
    "strategyqa_grounded_opaque_binary_codebook": strategyqa_grounded_opaque_binary_codebook,
    "strategyqa_grounded_remove_all_facts": strategyqa_grounded_remove_all_facts,
    "strategyqa_grounded_add_distractor_facts": strategyqa_grounded_add_distractor_facts,
    "strategyqa_grounded_partial_fact_ablation_optional": strategyqa_grounded_partial_fact_ablation_optional,
}
