from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def sms_spam_direct_binary_surface(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the original binary decision about whether an SMS is spam, but sample from semantically compatible binary surface pairs.
    
        Target derivation: Target is original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for sms_spam"; instruction='For this SMS message, is it true that the message is spam?'; options=["yes","no"]; target='no'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_direct_binary_surface", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_inverted_binary_question(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask the complement of the original binary decision, e.g. ask whether the answer is negative/refuted/non-matching instead of positive.
    
        Target derivation: Target is 1-original binary label.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for sms_spam"; instruction='Is it false that the message is spam?'; options=["yes","no"]; target='yes'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_inverted_binary_question", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_same_answer_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two SMS messagess and ask whether their original binary decisions are the same.
    
        Target derivation: Positive iff label1 == label2.
        Rows used: 2. Selection: Balance same and different.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for sms_spam","second":"Message 1 for sms_spam"}; instruction='Do the two items have the same binary outcome?'; options=["same","different"]; target='different'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_same_answer_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_pair_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two SMS messagess and ask which are positive under the original criterion: first only, second only, both, or neither.
    
        Target derivation: Derive from the two original binary labels.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for sms_spam","second":"Message 4 for sms_spam"}; instruction='Which of the two original propositions are positive?'; options=["A only","only the second","the two of them","neither applies"]; target='neither applies'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_pair_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_logical_and(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether both original propositions are positive.
    
        Target derivation: Positive iff label1=1 and label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for sms_spam","second":"Message 11 for sms_spam"}; instruction='Are both original propositions positive?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_logical_and", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_logical_or(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether at least one original proposition is positive.
    
        Target derivation: Positive iff label1=1 or label2=1.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for sms_spam","second":"Message 4 for sms_spam"}; instruction='Is at least one of the two original propositions positive?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_logical_or", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_logical_xor(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether exactly one original proposition is positive.
    
        Target derivation: Positive iff label1 != label2.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state={"first":"Message 0 for sms_spam","second":"Message 10 for sms_spam"}; instruction='Is exactly one of the two original propositions positive?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_logical_xor", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_other_target_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to a choice task; omit the true binary answer, retain the opposite answer, and add OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Original item remains answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for sms_spam"; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["true","another answer"]; target='another answer'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_other_target_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_other_target_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Convert binary decision to choice with both binary answers plus OTHER.
    
        Target derivation: Target is the original binary answer; OTHER is incorrect.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for sms_spam"; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["no","yes","not among these"]; target='no'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_other_target_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_unknown_missing_required_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove or explicitly mark unavailable the required SMS message text; offer both binary answers plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Instruction must state that the decision is based on supplied information when outside knowledge could otherwise answer it.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state="<missing>"; instruction='Decide only from the supplied input; choose insufficient information if a required field is missing.'; options=["yes","no","more information is needed"]; target='more information is needed'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_unknown_missing_required_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer binary answers plus OTHER and UNKNOWN and generate the three global joint cases.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for sms_spam"; instruction='Choose the binary answer when it is offered. Use the unlisted-answer option only if the answer is known but omitted; use insufficient information only when the input cannot decide it.'; options=["N","Y","a different answer","cannot be resolved"]; target='N'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_opaque_binary_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map positive and negative to two temporary opaque labels with definitions given in the question.
    
        Target derivation: Target is opaque label mapped to original binary answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state="Message 0 for sms_spam"; instruction='Use the temporary binary codebook and select the code matching the original decision.'; options=["X8","V4"]; target='X8'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_opaque_binary_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def sms_spam_reference_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one SMS as a reference and two candidate SMS messages; ask which candidate has the same spam/ham class as the reference.
    
        Target derivation: Target is unique candidate with same binary label.
        Rows used: 3. Selection: Require exactly one match.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Message 0 for sms_spam"; gold=0.
          Reformulated output: primitive='choice'; state={"reference":"Message 0 for sms_spam","A":"Message 3 for sms_spam","B":"Message 8 for sms_spam"}; instruction='Which candidate has the same binary relation/class as the reference?'; options=["A","B"]; target='B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("sms_spam_reference_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "sms_spam_direct_binary_surface": sms_spam_direct_binary_surface,
    "sms_spam_inverted_binary_question": sms_spam_inverted_binary_question,
    "sms_spam_same_answer_pair": sms_spam_same_answer_pair,
    "sms_spam_pair_truth_pattern": sms_spam_pair_truth_pattern,
    "sms_spam_logical_and": sms_spam_logical_and,
    "sms_spam_logical_or": sms_spam_logical_or,
    "sms_spam_logical_xor": sms_spam_logical_xor,
    "sms_spam_other_target_removed": sms_spam_other_target_removed,
    "sms_spam_other_target_present": sms_spam_other_target_present,
    "sms_spam_unknown_missing_required_input": sms_spam_unknown_missing_required_input,
    "sms_spam_other_unknown_joint": sms_spam_other_unknown_joint,
    "sms_spam_opaque_binary_codebook": sms_spam_opaque_binary_codebook,
    "sms_spam_reference_match": sms_spam_reference_match,
}
