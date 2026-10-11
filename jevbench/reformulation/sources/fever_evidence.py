from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def fever_evidence_direct_binary_surface(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the original binary decision about whether evidence supports the claim, but sample from semantically compatible binary surface pairs.
    
        Target derivation: Target is original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; instruction='For this claim-evidence pair, is it true that the evidence supports the claim?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_direct_binary_surface", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_inverted_binary_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the complement of the original binary decision, e.g. ask whether the answer is negative/refuted/non-matching instead of positive.
    
        Target derivation: Target is 1-original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; instruction='Is it false that the evidence supports the claim?'; options=["Y","N"]; target='Y'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_inverted_binary_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_same_answer_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two claim-evidence pairss and ask whether their original binary decisions are the same.
    
        Target derivation: Positive iff label1 == label2.
        Rows used: 2. Selection: Balance same and different.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]},"second":{"claim":"Claim 7","evidence":["Evidence 7 a","Evidence ...; instruction='Do the two items have the same binary outcome?'; options=["equivalent","not equivalent"]; target='not equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_same_answer_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_pair_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two claim-evidence pairss and ask which are positive under the original criterion: first only, second only, both, or neither.
    
        Target derivation: Derive from the two original binary labels.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]},"second":{"claim":"Claim 6","evidence":["Evidence 6 a","Evidence ...; instruction='Which of the two original propositions are positive?'; options=["A only","B only","both match","neither applies"]; target='neither applies'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_pair_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_logical_and(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether both original propositions are positive.
    
        Target derivation: Positive iff label1=1 and label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]},"second":{"claim":"Claim 11","evidence":["Evidence 11 a","Evidenc...; instruction='Are both original propositions positive?'; options=["T","F"]; target='F'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_logical_and", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_logical_or(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether at least one original proposition is positive.
    
        Target derivation: Positive iff label1=1 or label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]},"second":{"claim":"Claim 10","evidence":["Evidence 10 a","Evidenc...; instruction='Is at least one of the two original propositions positive?'; options=["T","F"]; target='F'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_logical_or", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_logical_xor(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether exactly one original proposition is positive.
    
        Target derivation: Positive iff label1 != label2.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"first":{"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]},"second":{"claim":"Claim 7","evidence":["Evidence 7 a","Evidence ...; instruction='Is exactly one of the two original propositions positive?'; options=["T","F"]; target='T'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_logical_xor", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_other_target_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to a choice task; omit the true binary answer, retain the opposite answer, and add OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Original item remains answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["yes","a choice not shown"]; target='a choice not shown'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_other_target_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_other_target_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to choice with both binary answers plus OTHER.
    
        Target derivation: Target is the original binary answer; OTHER is incorrect.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["false","true","not represented by these choices"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_other_target_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required evidence or claim; offer both binary answers plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Instruction must state that the decision is based on supplied information when outside knowledge could otherwise answer it.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":"The required information was not supplied."}; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["1 - yes","0 - no","the available information does not decide it"]; target='the available information does not decide it'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer binary answers plus OTHER and UNKNOWN and generate the three global joint cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["yes","not listed","required information is missing"]; target='not listed'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_opaque_binary_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map positive and negative to two temporary opaque labels with definitions given in the question.
    
        Target derivation: Target is opaque label mapped to original binary answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; instruction='Use the temporary binary codebook and select the code matching the original decision.'; options=["V4","N1"]; target='V4'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_opaque_binary_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_refutation_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the evidence refutes the claim rather than supports it.
    
        Target derivation: Positive iff original support label is negative, assuming the source contains only supported/refuted rows.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; instruction='Does the supplied evidence refute the claim?'; options=["Y","N"]; target='Y'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_refutation_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_add_unrelated_evidence(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Append clearly separated evidence from an unrelated row while retaining all original evidence; ask about the original claim.
    
        Target derivation: Target remains the original support/refute label.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: Keep original evidence intact.; Do not merge unrelated evidence into the same sentence.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"],"unrelated_evidence":["Evidence 3 a","Evidence 3 b"]}; instruction='Considering the original evidence for the original claim, does it support the claim? Treat the separately marked unrelated evidence as a distractor.'; options=["affirmative","negative"]; target='negative'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_add_unrelated_evidence", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_evidence_ablation(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove all evidence and explicitly require a judgment from supplied evidence only; offer supported, refuted, and UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":[]}; instruction='Judge only from the supplied evidence. If no evidence determines the claim, choose insufficient information.'; options=["supported","refuted","the evidence is insufficient"]; target='the evidence is insufficient'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_evidence_ablation", row, pool=pool, seed=seed, validator=validator, **kwargs)

def fever_evidence_partial_ablation_optional(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove a subset of evidence sentences and include UNKNOWN.
    
        Target derivation: UNKNOWN only when a validator confirms remaining evidence neither supports nor refutes the claim.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: Yes: pass validator(transformed_row, original_row) -> bool.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"claim":"Claim 0","evidence":["Evidence 0 a","Evidence 0 b"]}; gold=0.
          Reformulated output: primitive='choice'; state={"claim":"Claim 0","evidence":"The required information was not supplied."}; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["yes","no","not enough evidence"]; target='not enough evidence'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("fever_evidence_partial_ablation_optional", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "fever_evidence_direct_binary_surface": fever_evidence_direct_binary_surface,
    "fever_evidence_inverted_binary_question": fever_evidence_inverted_binary_question,
    "fever_evidence_same_answer_pair": fever_evidence_same_answer_pair,
    "fever_evidence_pair_truth_pattern": fever_evidence_pair_truth_pattern,
    "fever_evidence_logical_and": fever_evidence_logical_and,
    "fever_evidence_logical_or": fever_evidence_logical_or,
    "fever_evidence_logical_xor": fever_evidence_logical_xor,
    "fever_evidence_other_target_removed": fever_evidence_other_target_removed,
    "fever_evidence_other_target_present": fever_evidence_other_target_present,
    "fever_evidence_unknown_missing_required_input": fever_evidence_unknown_missing_required_input,
    "fever_evidence_other_unknown_joint": fever_evidence_other_unknown_joint,
    "fever_evidence_opaque_binary_codebook": fever_evidence_opaque_binary_codebook,
    "fever_evidence_refutation_question": fever_evidence_refutation_question,
    "fever_evidence_add_unrelated_evidence": fever_evidence_add_unrelated_evidence,
    "fever_evidence_evidence_ablation": fever_evidence_evidence_ablation,
    "fever_evidence_partial_ablation_optional": fever_evidence_partial_ablation_optional,
}
