from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def massive_direct_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rephrase the instruction while preserving the same assistant utterance and full original label set.
    
        Target derivation: Target is the original gold class.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Do not paraphrase the content being classified unless semantics are guaranteed unchanged.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state="Example 0 for massive"; instruction='Identify the most appropriate option from the supplied labels. Original task: Choose the class.'; options=["play_music","timer_set","weather_query","refund","payment","fear"]; target='play_music'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_direct_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_candidate_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one candidate intent; ask whether that candidate is the correct intent for the assistant utterance.
    
        Target derivation: Positive iff candidate equals original gold class.
        Rows used: 1. Selection: 50% gold candidate, 50% non-gold candidate; prefer hard/nearby negatives when available.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for massive","candidate_label":"play_music","candidate_description":"play music"}; instruction='Is the proposed intent the correct one for this assistant utterance?'; options=["correct","incorrect"]; target='correct'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_candidate_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_subset_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a subset of intent labels; ask whether the correct intent is contained in that set.
    
        Target derivation: Positive iff original gold class is in sampled subset.
        Rows used: 1. Selection: Vary subset size from 1 to min(8, K-1). Balance include-gold vs exclude-gold.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for massive","candidate_set":{"surprise":"surprise","payment":"payment","timer_set":"timer set","notice":"notice"}}; instruction='Is the correct intent for this assistant utterance contained in `candidate_set`?'; options=["inside the set","outside the set"]; target='outside the set'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_subset_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_partition_group(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Partition a sampled candidate label set into 2-4 temporary groups and ask which group contains the correct intent.
    
        Target derivation: Target is the group containing the original gold class.
        Rows used: 1. Selection: Ensure gold is included in exactly one group.; Randomize group names and group order.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state="Example 0 for massive"; instruction='Which temporary group contains the correct intent for this assistant utterance?'; options=["Bucket U","Bucket R","Group Gamma","Bucket T"]; target='Bucket R'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_partition_group", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_other_gold_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer a subset of ordinary intent labels that deliberately excludes the gold class, plus a sampled OTHER surface form.
    
        Target derivation: Target is canonical OTHER.
        Rows used: 1. Selection: Keep enough plausible distractors that OTHER is not trivial.
        Validity guard: Original problem must remain fully answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state="Example 0 for massive"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["timer_set","fear","surprise","refund","notice","no offered label fits"]; target='no offered label fits'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_other_gold_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_other_gold_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer the gold intent, several distractors, and a sampled OTHER surface form.
    
        Target derivation: Target remains the ordinary gold class; OTHER is a distractor.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state="Example 0 for massive"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["refund","fear","timer_set","play_music","surprise","outside the listed labels"]; target='play_music'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_other_gold_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_unknown_missing_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Replace the required assistant utterance with an explicit missing marker or omit that required field; offer ordinary candidates plus UNKNOWN.
    
        Target derivation: Target is canonical UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: The question must explicitly restrict the decision to the supplied input so outside knowledge cannot repair the missing field.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state="[NOT PROVIDED]"; instruction='Decide only from the supplied input. If the required information is unavailable, choose the insufficient-information option.'; options=["timer_set","payment","notice","surprise","play_music","support"]; target='unspecified'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_unknown_missing_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Generate the three global joint cases using ordinary intent labels plus both OTHER and UNKNOWN.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: Balance answer-present, answer-missing, and unanswerable cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state="[UNAVAILABLE]"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["weather_query","notice","timer_set","fear","refund","other"]; target='cannot be resolved'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_opaque_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rename a sampled set of intent labels with temporary opaque keys while preserving readable descriptions, then ask for the key matching the assistant utterance.
    
        Target derivation: Target is the temporary key mapped to the original gold class.
        Rows used: 1. Selection: Regenerate key mapping per example or family.; Do not let alphabetical order encode the original label.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state="Example 0 for massive"; instruction='Use the temporary codebook below and select the code whose description best matches the input.'; options=["N1","R6","FW","JQ","HY","T3"]; target='T3'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_opaque_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_same_class_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows and ask whether they share the same intent. Present both assistant utterances.
    
        Target derivation: Positive iff the two original gold classes are equal.
        Rows used: 2. Selection: Balance same-class and different-class pairs.; Use different raw examples even for same-class pairs.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state={"first":"Example 0 for massive","second":"Example 8 for massive"}; instruction='Do the first and second assistant utterances have the same intent?'; options=["equivalent","not equivalent"]; target='equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_same_class_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_reference_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one row as a reference and 2-4 other rows as candidates; ask which candidate has the same intent as the reference.
    
        Target derivation: Target is the unique candidate whose original gold class equals the reference gold class.
        Rows used: 3-5. Selection: Require exactly one matching candidate.; All rows from same split.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state={"reference":"Example 0 for massive","candidates":{"A":"Example 1 for massive","B":"Example 7 for massive","C":"Example 2 for massive"}}; instruction='Which candidate belongs to the same underlying class as the reference?'; options=["A","B","C"]; target='C'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_reference_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_odd_one_out(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows sharing one intent and one row with a different intent; ask which assistant utterance is the odd one out by class.
    
        Target derivation: Target is the row whose gold class differs.
        Rows used: 3. Selection: Randomize item order.; Avoid duplicate text.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state={"items":{"A":"Example 5 for massive","B":"Example 8 for massive","C":"Example 0 for massive"}}; instruction='Two items share the same underlying class. Which item is the odd one out?'; options=["A","B","C"]; target='A'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_odd_one_out", row, pool=pool, seed=seed, validator=validator, **kwargs)

def massive_two_utterance_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose candidate intent X and ask whether X applies to the first utterance only, second only, both, or neither.
    
        Target derivation: Derive from equality of X with the two gold intents.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for massive"; gold="play_music".
          Reformulated output: primitive='choice'; state={"candidate":"surprise","first":"Example 0 for massive","second":"Example 10 for massive"}; instruction='For which of the two inputs does the proposed label apply?'; options=["only the first","only the second","A and B","neither A nor B"]; target='neither A nor B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("massive_two_utterance_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "massive_direct_rephrase": massive_direct_rephrase,
    "massive_candidate_verify": massive_candidate_verify,
    "massive_subset_membership": massive_subset_membership,
    "massive_partition_group": massive_partition_group,
    "massive_other_gold_removed": massive_other_gold_removed,
    "massive_other_gold_present": massive_other_gold_present,
    "massive_unknown_missing_input": massive_unknown_missing_input,
    "massive_other_unknown_joint": massive_other_unknown_joint,
    "massive_opaque_codebook": massive_opaque_codebook,
    "massive_same_class_pair": massive_same_class_pair,
    "massive_reference_match": massive_reference_match,
    "massive_odd_one_out": massive_odd_one_out,
    "massive_two_utterance_truth_pattern": massive_two_utterance_truth_pattern,
}
