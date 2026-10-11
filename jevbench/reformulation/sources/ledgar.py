from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def ledgar_direct_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rephrase the instruction while preserving the same contract clause and full original label set.
    
        Target derivation: Target is the original gold class.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Do not paraphrase the content being classified unless semantics are guaranteed unchanged.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state="Example 0 for ledgar"; instruction='Select the single best matching label for the input. Original task: Choose the class.'; options=["confidentiality","governing_law","termination","refund","payment","fear"]; target='termination'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_direct_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_candidate_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one candidate provision category; ask whether that candidate is the correct provision category for the contract clause.
    
        Target derivation: Positive iff candidate equals original gold class.
        Rows used: 1. Selection: 50% gold candidate, 50% non-gold candidate; prefer hard/nearby negatives when available.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for ledgar","candidate_label":"termination","candidate_description":"termination"}; instruction='Is the proposed provision category the correct one for this contract clause?'; options=["right","wrong"]; target='right'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_candidate_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_subset_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a subset of provision category labels; ask whether the correct provision category is contained in that set.
    
        Target derivation: Positive iff original gold class is in sampled subset.
        Rows used: 1. Selection: Vary subset size from 1 to min(8, K-1). Balance include-gold vs exclude-gold.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for ledgar","candidate_set":{"governing_law":"governing law","refund":"refund","termination":"termination","confident...; instruction='Is the correct provision category for this contract clause contained in `candidate_set`?'; options=["included","not included"]; target='included'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_subset_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_partition_group(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Partition a sampled candidate label set into 2-4 temporary groups and ask which group contains the correct provision category.
    
        Target derivation: Target is the group containing the original gold class.
        Rows used: 1. Selection: Ensure gold is included in exactly one group.; Randomize group names and group order.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state="Example 0 for ledgar"; instruction='Which temporary group contains the correct provision category for this contract clause?'; options=["Group Beta","Bucket S"]; target='Group Beta'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_partition_group", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_other_gold_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer a subset of ordinary provision category labels that deliberately excludes the gold class, plus a sampled OTHER surface form.
    
        Target derivation: Target is canonical OTHER.
        Rows used: 1. Selection: Keep enough plausible distractors that OTHER is not trivial.
        Validity guard: Original problem must remain fully answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state="Example 0 for ledgar"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["support","confidentiality","refund","fear","governing_law","other"]; target='other'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_other_gold_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_other_gold_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer the gold provision category, several distractors, and a sampled OTHER surface form.
    
        Target derivation: Target remains the ordinary gold class; OTHER is a distractor.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state="Example 0 for ledgar"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["refund","surprise","confidentiality","termination","governing_law","none of the provided labels"]; target='termination'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_other_gold_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_unknown_missing_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Replace the required contract clause with an explicit missing marker or omit that required field; offer ordinary candidates plus UNKNOWN.
    
        Target derivation: Target is canonical UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: The question must explicitly restrict the decision to the supplied input so outside knowledge cannot repair the missing field.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state="[NOT PROVIDED]"; instruction='Decide only from the supplied input. If the required information is unavailable, choose the insufficient-information option.'; options=["notice","fear","termination","payment","surprise","confidentiality"]; target='unspecified'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_unknown_missing_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Generate the three global joint cases using ordinary provision category labels plus both OTHER and UNKNOWN.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: Balance answer-present, answer-missing, and unanswerable cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state="Example 0 for ledgar"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["fear","governing_law","payment","refund","termination","not represented by these choices"]; target='termination'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_opaque_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rename a sampled set of provision category labels with temporary opaque keys while preserving readable descriptions, then ask for the key matching the contract clause.
    
        Target derivation: Target is the temporary key mapped to the original gold class.
        Rows used: 1. Selection: Regenerate key mapping per example or family.; Do not let alphabetical order encode the original label.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state="Example 0 for ledgar"; instruction='Use the temporary codebook below and select the code whose description best matches the input.'; options=["T3","GX","CQ","M2","Z4","K7"]; target='FW'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_opaque_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_same_class_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows and ask whether they share the same provision category. Present both contract clauses.
    
        Target derivation: Positive iff the two original gold classes are equal.
        Rows used: 2. Selection: Balance same-class and different-class pairs.; Use different raw examples even for same-class pairs.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state={"first":"Example 0 for ledgar","second":"Example 1 for ledgar"}; instruction='Do the first and second contract clauses have the same provision category?'; options=["equivalent","not equivalent"]; target='not equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_same_class_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_reference_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one row as a reference and 2-4 other rows as candidates; ask which candidate has the same provision category as the reference.
    
        Target derivation: Target is the unique candidate whose original gold class equals the reference gold class.
        Rows used: 3-5. Selection: Require exactly one matching candidate.; All rows from same split.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state={"reference":"Example 0 for ledgar","candidates":{"A":"Example 10 for ledgar","B":"Example 9 for ledgar","C":"Example 8 for ledgar"}}; instruction='Which candidate belongs to the same underlying class as the reference?'; options=["A","B","C"]; target='C'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_reference_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_odd_one_out(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows sharing one provision category and one row with a different provision category; ask which contract clause is the odd one out by class.
    
        Target derivation: Target is the row whose gold class differs.
        Rows used: 3. Selection: Randomize item order.; Avoid duplicate text.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state={"items":{"A":"Example 0 for ledgar","B":"Example 2 for ledgar","C":"Example 10 for ledgar"}}; instruction='Two items share the same underlying class. Which item is the odd one out?'; options=["A","B","C"]; target='C'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_odd_one_out", row, pool=pool, seed=seed, validator=validator, **kwargs)

def ledgar_two_clause_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose one provision category X and present two clauses; ask whether X describes first only, second only, both, or neither.
    
        Target derivation: Derive from each row gold category.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for ledgar"; gold="termination".
          Reformulated output: primitive='choice'; state={"candidate":"termination","first":"Example 0 for ledgar","second":"Example 5 for ledgar"}; instruction='For which of the two inputs does the proposed label apply?'; options=["the first but not the second","second only","both match","neither A nor B"]; target='the first but not the second'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("ledgar_two_clause_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "ledgar_direct_rephrase": ledgar_direct_rephrase,
    "ledgar_candidate_verify": ledgar_candidate_verify,
    "ledgar_subset_membership": ledgar_subset_membership,
    "ledgar_partition_group": ledgar_partition_group,
    "ledgar_other_gold_removed": ledgar_other_gold_removed,
    "ledgar_other_gold_present": ledgar_other_gold_present,
    "ledgar_unknown_missing_input": ledgar_unknown_missing_input,
    "ledgar_other_unknown_joint": ledgar_other_unknown_joint,
    "ledgar_opaque_codebook": ledgar_opaque_codebook,
    "ledgar_same_class_pair": ledgar_same_class_pair,
    "ledgar_reference_match": ledgar_reference_match,
    "ledgar_odd_one_out": ledgar_odd_one_out,
    "ledgar_two_clause_truth_pattern": ledgar_two_clause_truth_pattern,
}
