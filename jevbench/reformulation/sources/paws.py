from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def paws_direct_binary_surface(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the original binary decision about whether the two sentences are paraphrases, but sample from semantically compatible binary surface pairs.
    
        Target derivation: Target is original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"S1 0","sentence2":"S2 0"}; instruction='For this sentence pair, is it true that the sentences have the same meaning?'; options=["affirmative","negative"]; target='negative'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_direct_binary_surface", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_inverted_binary_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the complement of the original binary decision, e.g. ask whether the answer is negative/refuted/non-matching instead of positive.
    
        Target derivation: Target is 1-original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"S1 0","sentence2":"S2 0"}; instruction='Is it false that the sentences have the same meaning?'; options=["Y","N"]; target='Y'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_inverted_binary_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_same_answer_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two sentence pairss and ask whether their original binary decisions are the same.
    
        Target derivation: Positive iff label1 == label2.
        Rows used: 2. Selection: Balance same and different.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"sentence1":"S1 0","sentence2":"S2 0"},"second":{"sentence1":"S1 10","sentence2":"S2 10"}}; instruction='Do the two items have the same binary outcome?'; options=["same","different"]; target='same'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_same_answer_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_pair_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two sentence pairss and ask which are positive under the original criterion: first only, second only, both, or neither.
    
        Target derivation: Derive from the two original binary labels.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"sentence1":"S1 0","sentence2":"S2 0"},"second":{"sentence1":"S1 10","sentence2":"S2 10"}}; instruction='Which of the two original propositions are positive?'; options=["A only","the second but not the first","each of them","neither is true"]; target='neither is true'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_pair_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_logical_and(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether both original propositions are positive.
    
        Target derivation: Positive iff label1=1 and label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"sentence1":"S1 0","sentence2":"S2 0"},"second":{"sentence1":"S1 5","sentence2":"S2 5"}}; instruction='Are both original propositions positive?'; options=["T","F"]; target='F'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_logical_and", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_logical_or(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether at least one original proposition is positive.
    
        Target derivation: Positive iff label1=1 or label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"sentence1":"S1 0","sentence2":"S2 0"},"second":{"sentence1":"S1 3","sentence2":"S2 3"}}; instruction='Is at least one of the two original propositions positive?'; options=["true","false"]; target='true'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_logical_or", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_logical_xor(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether exactly one original proposition is positive.
    
        Target derivation: Positive iff label1 != label2.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"sentence1":"S1 0","sentence2":"S2 0"},"second":{"sentence1":"S1 8","sentence2":"S2 8"}}; instruction='Is exactly one of the two original propositions positive?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_logical_xor", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_other_target_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to a choice task; omit the true binary answer, retain the opposite answer, and add OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Original item remains answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"S1 0","sentence2":"S2 0"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["Y","none of the provided options"]; target='none of the provided options'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_other_target_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_other_target_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to choice with both binary answers plus OTHER.
    
        Target derivation: Target is the original binary answer; OTHER is incorrect.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"S1 0","sentence2":"S2 0"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["false","true","elsewhere"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_other_target_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required first or second sentence; offer both binary answers plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Instruction must state that the decision is based on supplied information when outside knowledge could otherwise answer it.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"S1 0","sentence2":"[NOT PROVIDED]"}; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["affirmative","negative","undetermined"]; target='undetermined'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer binary answers plus OTHER and UNKNOWN and generate the three global joint cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"S1 0","sentence2":"<missing>"}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["Y","an unlisted option","indeterminate"]; target='indeterminate'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_opaque_binary_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map positive and negative to two temporary opaque labels with definitions given in the question.
    
        Target derivation: Target is opaque label mapped to original binary answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"S1 0","sentence2":"S2 0"}; instruction='Use the temporary binary codebook and select the code matching the original decision.'; options=["CQ","Q9"]; target='CQ'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_opaque_binary_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_difference_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the two sentences differ in meaning rather than whether they are paraphrases.
    
        Target derivation: Target is complement of original paraphrase label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"sentence1":"S1 0","sentence2":"S2 0"}; instruction='Do the two sentences differ in meaning?'; options=["affirmative","negative"]; target='affirmative'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_difference_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def paws_relation_reference_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one sentence pair as a reference relation and two candidate pairs; ask which candidate has the same paraphrase/non-paraphrase status.
    
        Target derivation: Target is the unique candidate with the same binary label as the reference.
        Rows used: 3. Selection: Require exactly one match.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"sentence1":"S1 0","sentence2":"S2 0"}; gold=0.
          Reformulated output: primitive='choice'; state={"reference":{"sentence1":"S1 0","sentence2":"S2 0"},"A":{"sentence1":"S1 11","sentence2":"S2 11"},"B":{"sentence1":"S1 4","sentence2":"S...; instruction='Which candidate has the same binary relation/class as the reference?'; options=["A","B"]; target='B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("paws_relation_reference_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "paws_direct_binary_surface": paws_direct_binary_surface,
    "paws_inverted_binary_question": paws_inverted_binary_question,
    "paws_same_answer_pair": paws_same_answer_pair,
    "paws_pair_truth_pattern": paws_pair_truth_pattern,
    "paws_logical_and": paws_logical_and,
    "paws_logical_or": paws_logical_or,
    "paws_logical_xor": paws_logical_xor,
    "paws_other_target_removed": paws_other_target_removed,
    "paws_other_target_present": paws_other_target_present,
    "paws_unknown_missing_required_input": paws_unknown_missing_required_input,
    "paws_other_unknown_joint": paws_other_unknown_joint,
    "paws_opaque_binary_codebook": paws_opaque_binary_codebook,
    "paws_difference_question": paws_difference_question,
    "paws_relation_reference_match": paws_relation_reference_match,
}
