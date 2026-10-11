from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def chaosnli_direct_relation(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present premise and hypothesis and ask for entailment, contradiction, or neutral/undetermined relation using varied label descriptions.
    
        Target derivation: Target is original NLI relation.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Classify whether the hypothesis is entailed, contradicted, or left undecided by the premise.'; options=["entailment","neutral","contradiction"]; target='entailment'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_direct_relation", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_candidate_relation_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one of the three NLI relations and ask whether it is the relation between premise and hypothesis.
    
        Target derivation: Positive iff sampled relation equals gold relation.
        Rows used: 1. Selection: Balance correct and incorrect relation candidates.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: P(positive)=soft_label[candidate relation].
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"input":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"proposed_relation":"contradiction","relation_definition":"The premise contr...; instruction='Is `proposed_relation` the correct relation for this premise-hypothesis pair?'; options=["correct","incorrect"]; target='incorrect'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_candidate_relation_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_relation_subset_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the true relation belongs to a sampled subset of NLI relations.
    
        Target derivation: Positive iff gold relation is in subset.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: P(positive)=sum of soft probabilities in subset.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"input":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"relation_set":{"entailment":"The hypothesis follows from the premise.","con...; instruction='Is the true NLI relation one of the relations in `relation_set`?'; options=["inside the set","outside the set"]; target='inside the set'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_relation_subset_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_same_relation_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two premise-hypothesis pairs and ask whether the two pairs have the same NLI relation.
    
        Target derivation: Positive iff gold_relation_1 == gold_relation_2.
        Rows used: 2. Selection: Balance same and different relations.; Prefer reasonably short pairs when available.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: Optional independent-rater framing: P(same)=sum_r p1(r)*p2(r).
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"first":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"second":{"premise":"Premise 11","hypothesis":"Hypothesis 11"}}; instruction='Do the first and second inputs have the same label?'; options=["equivalent","not equivalent"]; target='not equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_same_relation_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_specific_relation_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose a relation R, present two NLI pairs, and ask whether R describes first only, second only, both, or neither.
    
        Target derivation: Compare R to each pair gold relation.
        Rows used: 2. Selection: Sample all four patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"candidate":"entailment","first":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"second":{"premise":"Premise 9","hypothesis":"Hypot...; instruction='For which of the two inputs does the proposed label apply?'; options=["A only","only the second","A and B","none of the two"]; target='A and B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_specific_relation_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_reference_relation_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one NLI pair as reference and several candidate pairs; ask which candidate has the same relation as the reference.
    
        Target derivation: Target is the unique candidate with the same gold relation.
        Rows used: 3-5. Selection: Require exactly one relation match.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"reference":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"candidates":{"A":{"premise":"Premise 10","hypothesis":"Hypothesis 10"},...; instruction='Which candidate belongs to the same underlying class as the reference?'; options=["A","B","C"]; target='C'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_reference_relation_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_relation_odd_one_out(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose two NLI pairs with the same relation and one with a different relation; ask for the odd pair.
    
        Target derivation: Target is the pair with different gold relation.
        Rows used: 3. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"items":{"A":{"premise":"Premise 10","hypothesis":"Hypothesis 10"},"B":{"premise":"Premise 3","hypothesis":"Hypothesis 3"},"C":{"premise...; instruction='Two items share the same underlying class. Which item is the odd one out?'; options=["A","B","C"]; target='A'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_relation_odd_one_out", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_other_gold_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer only two non-gold NLI relations plus OTHER.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["neutral","contradiction","not listed"]; target='not listed'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_other_gold_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_other_gold_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer the gold relation, at least one distractor relation, and OTHER.
    
        Target derivation: Target is the gold relation.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["neutral","contradiction","entailment","no listed option is correct"]; target='entailment'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_other_gold_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_unknown_missing_pair_member(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove either the premise or hypothesis and explicitly mark it unavailable; offer the three relations plus UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: The missing sentence must not be reconstructable from other fields.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"[UNAVAILABLE]"}; instruction='Determine the relation only from the supplied pair; use the insufficient-information option when premise or hypothesis is missing.'; options=["entailment","neutral","contradiction","unclear from the information given"]; target='unclear from the information given'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_unknown_missing_pair_member", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Mix answer-present, answer-missing, and missing-premise/hypothesis cases with both OTHER and UNKNOWN present.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"[NOT PROVIDED]"}; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["contradiction","neutral","a category not shown","unresolvable from the given data"]; target='unresolvable from the given data'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_opaque_relation_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Map the three relations to temporary opaque names with definitions supplied in the question.
    
        Target derivation: Target is temporary name mapped to gold relation.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Use the temporary codebook below and select the code whose description best matches the input.'; options=["R6","CQ","GX"]; target='R6'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_opaque_relation_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_relation_boolean_composition(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Build a logical claim such as both pairs are entailments, at least one is contradiction, or exactly one is neutral.
    
        Target derivation: Evaluate the logical expression from the two gold relations.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"first":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"second":{"premise":"Premise 7","hypothesis":"Hypothesis 7"},"relation":"neu...; instruction='Is it true that at least one pair has relation `neutral`?'; options=["true","false"]; target='true'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_relation_boolean_composition", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_majority_strength_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the leading relation received at least threshold t of annotator votes.
    
        Target derivation: Positive iff max(soft_label) >= t.
        Rows used: 1. Selection: Sample thresholds with balance.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"premise":"Premise 0","hypothesis":"Hypothesis 0"}; instruction='Did the leading human label receive at least 70% of the annotation probability?'; options=["meets the condition","does not meet the condition"]; target='meets the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_majority_strength_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def chaosnli_compare_annotator_agreement(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask which NLI pair has stronger annotator agreement on its leading relation, allowing an approximately-equal option.
    
        Target derivation: Compare max soft-label shares with a configured tie epsilon.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"premise":"Premise 0","hypothesis":"Hypothesis 0"}; gold="entailment".
          Reformulated output: primitive='choice'; state={"first":{"premise":"Premise 0","hypothesis":"Hypothesis 0"},"second":{"premise":"Premise 4","hypothesis":"Hypothesis 4"}}; instruction='Which item has stronger human agreement on its leading label?'; options=["first item","B","equal"]; target='equal'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("chaosnli_compare_annotator_agreement", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "chaosnli_direct_relation": chaosnli_direct_relation,
    "chaosnli_candidate_relation_verify": chaosnli_candidate_relation_verify,
    "chaosnli_relation_subset_membership": chaosnli_relation_subset_membership,
    "chaosnli_same_relation_pair": chaosnli_same_relation_pair,
    "chaosnli_specific_relation_truth_pattern": chaosnli_specific_relation_truth_pattern,
    "chaosnli_reference_relation_match": chaosnli_reference_relation_match,
    "chaosnli_relation_odd_one_out": chaosnli_relation_odd_one_out,
    "chaosnli_other_gold_removed": chaosnli_other_gold_removed,
    "chaosnli_other_gold_present": chaosnli_other_gold_present,
    "chaosnli_unknown_missing_pair_member": chaosnli_unknown_missing_pair_member,
    "chaosnli_other_unknown_joint": chaosnli_other_unknown_joint,
    "chaosnli_opaque_relation_codebook": chaosnli_opaque_relation_codebook,
    "chaosnli_relation_boolean_composition": chaosnli_relation_boolean_composition,
    "chaosnli_majority_strength_threshold": chaosnli_majority_strength_threshold,
    "chaosnli_compare_annotator_agreement": chaosnli_compare_annotator_agreement,
}
