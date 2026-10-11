from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def banking77_direct_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rephrase the instruction while preserving the same customer message and full original label set.
    
        Target derivation: Target is the original gold class.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Do not paraphrase the content being classified unless semantics are guaranteed unchanged.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state="Example 0 for banking77"; instruction='Identify the most appropriate option from the supplied labels. Original task: Choose the class.'; options=["card_arrival","cash_withdrawal","transfer_fee","refund","payment","fear"]; target='card_arrival'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_direct_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_candidate_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one candidate intent; ask whether that candidate is the correct intent for the customer message.
    
        Target derivation: Positive iff candidate equals original gold class.
        Rows used: 1. Selection: 50% gold candidate, 50% non-gold candidate; prefer hard/nearby negatives when available.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for banking77","candidate_label":"card_arrival","candidate_description":"card arrival"}; instruction='Is the proposed banking support intent the correct one for this customer message?'; options=["right","wrong"]; target='right'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_candidate_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_subset_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a subset of intent labels; ask whether the correct intent is contained in that set.
    
        Target derivation: Positive iff original gold class is in sampled subset.
        Rows used: 1. Selection: Vary subset size from 1 to min(8, K-1). Balance include-gold vs exclude-gold.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for banking77","candidate_set":{"support":"support","transfer_fee":"transfer fee","notice":"notice","surprise":"surpr...; instruction='Is the correct banking support intent for this customer message contained in `candidate_set`?'; options=["included","not included"]; target='not included'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_subset_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_partition_group(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Partition a sampled candidate label set into 2-4 temporary groups and ask which group contains the correct intent.
    
        Target derivation: Target is the group containing the original gold class.
        Rows used: 1. Selection: Ensure gold is included in exactly one group.; Randomize group names and group order.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state="Example 0 for banking77"; instruction='Which temporary group contains the correct banking support intent for this customer message?'; options=["Group Delta","Set Four","Group Beta","Set Two"]; target='Set Two'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_partition_group", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_other_gold_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer a subset of ordinary intent labels that deliberately excludes the gold class, plus a sampled OTHER surface form.
    
        Target derivation: Target is canonical OTHER.
        Rows used: 1. Selection: Keep enough plausible distractors that OTHER is not trivial.
        Validity guard: Original problem must remain fully answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state="Example 0 for banking77"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["support","payment","fear","refund","transfer_fee","different from all listed options"]; target='different from all listed options'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_other_gold_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_other_gold_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer the gold intent, several distractors, and a sampled OTHER surface form.
    
        Target derivation: Target remains the ordinary gold class; OTHER is a distractor.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state="Example 0 for banking77"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["cash_withdrawal","transfer_fee","card_arrival","payment","fear","other"]; target='card_arrival'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_other_gold_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_unknown_missing_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Replace the required customer message with an explicit missing marker or omit that required field; offer ordinary candidates plus UNKNOWN.
    
        Target derivation: Target is canonical UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: The question must explicitly restrict the decision to the supplied input so outside knowledge cannot repair the missing field.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state="The required information was not supplied."; instruction='Decide only from the supplied input. If the required information is unavailable, choose the insufficient-information option.'; options=["card_arrival","cash_withdrawal","refund","fear","transfer_fee","surprise"]; target='there is not enough to decide'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_unknown_missing_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Generate the three global joint cases using ordinary intent labels plus both OTHER and UNKNOWN.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: Balance answer-present, answer-missing, and unanswerable cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state="Example 0 for banking77"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["fear","transfer_fee","refund","notice","payment","none of the provided labels"]; target='none of the provided labels'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_opaque_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rename a sampled set of intent labels with temporary opaque keys while preserving readable descriptions, then ask for the key matching the customer message.
    
        Target derivation: Target is the temporary key mapped to the original gold class.
        Rows used: 1. Selection: Regenerate key mapping per example or family.; Do not let alphabetical order encode the original label.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state="Example 0 for banking77"; instruction='Use the temporary codebook below and select the code whose description best matches the input.'; options=["AX","X8","HY","M2","T3","V4"]; target='M2'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_opaque_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_same_class_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows and ask whether they share the same intent. Present both customer messages.
    
        Target derivation: Positive iff the two original gold classes are equal.
        Rows used: 2. Selection: Balance same-class and different-class pairs.; Use different raw examples even for same-class pairs.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state={"first":"Example 0 for banking77","second":"Example 6 for banking77"}; instruction='Do the first and second customer messages have the same banking support intent?'; options=["equivalent","not equivalent"]; target='equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_same_class_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_reference_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one row as a reference and 2-4 other rows as candidates; ask which candidate has the same intent as the reference.
    
        Target derivation: Target is the unique candidate whose original gold class equals the reference gold class.
        Rows used: 3-5. Selection: Require exactly one matching candidate.; All rows from same split.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state={"reference":"Example 0 for banking77","candidates":{"A":"Example 3 for banking77","B":"Example 6 for banking77","C":"Example 5 for banki...; instruction='Which candidate belongs to the same underlying class as the reference?'; options=["A","B","C"]; target='B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_reference_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_odd_one_out(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows sharing one intent and one row with a different intent; ask which customer message is the odd one out by class.
    
        Target derivation: Target is the row whose gold class differs.
        Rows used: 3. Selection: Randomize item order.; Avoid duplicate text.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state={"items":{"A":"Example 8 for banking77","B":"Example 0 for banking77","C":"Example 6 for banking77"}}; instruction='Two items share the same underlying class. Which item is the odd one out?'; options=["A","B","C"]; target='A'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_odd_one_out", row, pool=pool, seed=seed, validator=validator, **kwargs)

def banking77_two_message_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose a candidate intent X and present two independent customer messages. Ask whether X applies to the first only, second only, both, or neither.
    
        Target derivation: Derive the four-way target by comparing X to each row gold intent.
        Rows used: 2. Selection: Sample across all four truth patterns when possible.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for banking77"; gold="card_arrival".
          Reformulated output: primitive='choice'; state={"candidate":"card_arrival","first":"Example 0 for banking77","second":"Example 11 for banking77"}; instruction='For which of the two inputs does the proposed label apply?'; options=["first only","only the second","each of them","none of the two"]; target='first only'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("banking77_two_message_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "banking77_direct_rephrase": banking77_direct_rephrase,
    "banking77_candidate_verify": banking77_candidate_verify,
    "banking77_subset_membership": banking77_subset_membership,
    "banking77_partition_group": banking77_partition_group,
    "banking77_other_gold_removed": banking77_other_gold_removed,
    "banking77_other_gold_present": banking77_other_gold_present,
    "banking77_unknown_missing_input": banking77_unknown_missing_input,
    "banking77_other_unknown_joint": banking77_other_unknown_joint,
    "banking77_opaque_codebook": banking77_opaque_codebook,
    "banking77_same_class_pair": banking77_same_class_pair,
    "banking77_reference_match": banking77_reference_match,
    "banking77_odd_one_out": banking77_odd_one_out,
    "banking77_two_message_truth_pattern": banking77_two_message_truth_pattern,
}
