from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def mnli_direct_relation(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present premise and hypothesis and ask for entailment, contradiction, or neutral/undetermined relation using varied label descriptions.
    
        Target derivation: Target is original NLI relation.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Which NLI relation best describes this premise-hypothesis pair?'; options=["entailment","neutral","contradiction"]; target='entailment'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_direct_relation", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_candidate_relation_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one of the three NLI relations and ask whether it is the relation between premise and hypothesis.
    
        Target derivation: Positive iff sampled relation equals gold relation.
        Rows used: 1. Selection: Balance correct and incorrect relation candidates.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"input":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"proposed_relation":"entailment","relation_definition":"The hypothesis follo...; instruction='Is `proposed_relation` the correct relation for this premise-hypothesis pair?'; options=["right","wrong"]; target='right'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_candidate_relation_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_relation_subset_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the true relation belongs to a sampled subset of NLI relations.
    
        Target derivation: Positive iff gold relation is in subset.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"input":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"relation_set":{"entailment":"The hypothesis follows from the premise.","neu...; instruction='Is the true NLI relation one of the relations in `relation_set`?'; options=["inside the set","outside the set"]; target='inside the set'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_relation_subset_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_same_relation_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two premise-hypothesis pairs and ask whether the two pairs have the same NLI relation.
    
        Target derivation: Positive iff gold_relation_1 == gold_relation_2.
        Rows used: 2. Selection: Balance same and different relations.; Prefer reasonably short pairs when available.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"first":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"second":{"premise":"Premise 9","hypothesis":"Hypothesis 9"}}; instruction='Do the first and second inputs have the same label?'; options=["equivalent","not equivalent"]; target='equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_same_relation_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_specific_relation_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose a relation R, present two NLI pairs, and ask whether R describes first only, second only, both, or neither.
    
        Target derivation: Compare R to each pair gold relation.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"candidate":"neutral","first":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"second":{"premise":"Premise 6","hypothesis":"Hypothes...; instruction='For which of the two inputs does the proposed label apply?'; options=["only the first","the second but not the first","both apply","neither applies"]; target='neither applies'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_specific_relation_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_reference_relation_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one NLI pair as reference and several candidate pairs; ask which candidate has the same relation as the reference.
    
        Target derivation: Target is the unique candidate with the same gold relation.
        Rows used: 3-5. Selection: Require exactly one relation match.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"reference":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"candidates":{"A":{"premise":"Premise 5","hypothesis":"Hypothesis 5"},"B...; instruction='Which candidate belongs to the same underlying class as the reference?'; options=["A","B","C"]; target='C'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_reference_relation_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_relation_odd_one_out(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose two NLI pairs with the same relation and one with a different relation; ask for the odd pair.
    
        Target derivation: Target is the pair with different gold relation.
        Rows used: 3. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"items":{"A":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"B":{"premise":"Premise 7","hypothesis":"Hypothesis 7"},"C":{"premise":...; instruction='Two items share the same underlying class. Which item is the odd one out?'; options=["A","B","C"]; target='B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_relation_odd_one_out", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_other_gold_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer only two non-gold NLI relations plus OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["neutral","contradiction","another answer"]; target='another answer'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_other_gold_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_other_gold_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer the gold relation, at least one distractor relation, and OTHER.
    
        Target derivation: Target is the gold relation.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["contradiction","entailment","neutral","none of the above"]; target='entailment'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_other_gold_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_unknown_missing_pair_member(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove either the premise or hypothesis and explicitly mark it unavailable; offer the three relations plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: The missing sentence must not be reconstructable from other fields.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"[NOT PROVIDED]"}; instruction='Determine the relation only from the supplied pair; use the insufficient-information option when premise or hypothesis is missing.'; options=["entailment","neutral","contradiction","cannot determine"]; target='cannot determine'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_unknown_missing_pair_member", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Mix answer-present, answer-missing, and missing-premise/hypothesis cases with both OTHER and UNKNOWN present.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"The required information was not supplied.","hypothesis":"Hypothesis 0"}; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["neutral","contradiction","an unlisted answer","more information is needed"]; target='more information is needed'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_opaque_relation_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map the three relations to temporary opaque names with definitions supplied in the question.
    
        Target derivation: Target is temporary name mapped to gold relation.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Use the temporary codebook below and select the code whose description best matches the input.'; options=["V4","K7","FW"]; target='FW'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_opaque_relation_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mnli_relation_boolean_composition(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Build a logical claim such as both pairs are entailments, at least one is contradiction, or exactly one is neutral.
    
        Target derivation: Evaluate the logical expression from the two gold relations.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"first":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"second":{"premise":"Premise 9","hypothesis":"Hypothesis 9"},"relation":"neu...; instruction='Is it true that exactly one pair has relation `neutral`?'; options=["true","false"]; target='false'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mnli_relation_boolean_composition", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "mnli_direct_relation": mnli_direct_relation,
    "mnli_candidate_relation_verify": mnli_candidate_relation_verify,
    "mnli_relation_subset_membership": mnli_relation_subset_membership,
    "mnli_same_relation_pair": mnli_same_relation_pair,
    "mnli_specific_relation_truth_pattern": mnli_specific_relation_truth_pattern,
    "mnli_reference_relation_match": mnli_reference_relation_match,
    "mnli_relation_odd_one_out": mnli_relation_odd_one_out,
    "mnli_other_gold_removed": mnli_other_gold_removed,
    "mnli_other_gold_present": mnli_other_gold_present,
    "mnli_unknown_missing_pair_member": mnli_unknown_missing_pair_member,
    "mnli_other_unknown_joint": mnli_other_unknown_joint,
    "mnli_opaque_relation_codebook": mnli_opaque_relation_codebook,
    "mnli_relation_boolean_composition": mnli_relation_boolean_composition,
}
