from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def clinc150_direct_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rephrase the instruction while preserving the same assistant query and full original label set.
    
        Target derivation: Target is the original gold class.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Do not paraphrase the content being classified unless semantics are guaranteed unchanged.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="Example 0 for clinc150"; instruction='Select the single best matching label for the input. Original task: Choose the class.'; options=["oos","set_alarm","weather","refund","payment","fear"]; target='set_alarm'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_direct_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_candidate_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one candidate intent; ask whether that candidate is the correct intent for the assistant query.
    
        Target derivation: Positive iff candidate equals original gold class.
        Rows used: 1. Selection: 50% gold candidate, 50% non-gold candidate; prefer hard/nearby negatives when available.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for clinc150","candidate_label":"notice","candidate_description":"notice"}; instruction='Is the proposed intent the correct one for this assistant query?'; options=["right","wrong"]; target='wrong'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_candidate_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_subset_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a subset of intent labels; ask whether the correct intent is contained in that set.
    
        Target derivation: Positive iff original gold class is in sampled subset.
        Rows used: 1. Selection: Vary subset size from 1 to min(8, K-1). Balance include-gold vs exclude-gold.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for clinc150","candidate_set":{"surprise":"surprise","refund":"refund","payment":"payment","notice":"notice","weather...; instruction='Is the correct intent for this assistant query contained in `candidate_set`?'; options=["included","not included"]; target='not included'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_subset_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_partition_group(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Partition a sampled candidate label set into 2-4 temporary groups and ask which group contains the correct intent.
    
        Target derivation: Target is the group containing the original gold class.
        Rows used: 1. Selection: Ensure gold is included in exactly one group.; Randomize group names and group order.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="Example 0 for clinc150"; instruction='Which temporary group contains the correct intent for this assistant query?'; options=["Set Three","Group Alpha"]; target='Set Three'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_partition_group", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_other_gold_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer a subset of ordinary intent labels that deliberately excludes the gold class, plus a sampled OTHER surface form.
    
        Target derivation: Target is canonical OTHER.
        Rows used: 1. Selection: Keep enough plausible distractors that OTHER is not trivial.
        Validity guard: Original problem must remain fully answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="Example 0 for clinc150"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["support","surprise","payment","weather","notice","outside the provided options"]; target='outside the provided options'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_other_gold_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_other_gold_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer the gold intent, several distractors, and a sampled OTHER surface form.
    
        Target derivation: Target remains the ordinary gold class; OTHER is a distractor.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="Example 0 for clinc150"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["refund","set_alarm","fear","oos","surprise","not represented by these choices"]; target='set_alarm'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_other_gold_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_unknown_missing_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Replace the required assistant query with an explicit missing marker or omit that required field; offer ordinary candidates plus UNKNOWN.
    
        Target derivation: Target is canonical UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: The question must explicitly restrict the decision to the supplied input so outside knowledge cannot repair the missing field.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="The required information was not supplied."; instruction='Decide only from the supplied input. If the required information is unavailable, choose the insufficient-information option.'; options=["support","payment","set_alarm","surprise","oos","fear"]; target='cannot tell from this context'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_unknown_missing_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Generate the three global joint cases using ordinary intent labels plus both OTHER and UNKNOWN.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: Balance answer-present, answer-missing, and unanswerable cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="Example 0 for clinc150"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["set_alarm","fear","weather","payment","oos","no offered label fits"]; target='set_alarm'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_opaque_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rename a sampled set of intent labels with temporary opaque keys while preserving readable descriptions, then ask for the key matching the assistant query.
    
        Target derivation: Target is the temporary key mapped to the original gold class.
        Rows used: 1. Selection: Regenerate key mapping per example or family.; Do not let alphabetical order encode the original label.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="Example 0 for clinc150"; instruction='Use the temporary codebook below and select the code whose description best matches the input.'; options=["AX","CQ","Z4","R6","HY","P5"]; target='HY'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_opaque_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_same_class_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows and ask whether they share the same intent. Present both assistant querys.
    
        Target derivation: Positive iff the two original gold classes are equal.
        Rows used: 2. Selection: Balance same-class and different-class pairs.; Use different raw examples even for same-class pairs.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state={"first":"Example 0 for clinc150","second":"Example 1 for clinc150"}; instruction='Do the first and second assistant querys have the same intent?'; options=["equivalent","not equivalent"]; target='not equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_same_class_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_reference_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one row as a reference and 2-4 other rows as candidates; ask which candidate has the same intent as the reference.
    
        Target derivation: Target is the unique candidate whose original gold class equals the reference gold class.
        Rows used: 3-5. Selection: Require exactly one matching candidate.; All rows from same split.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state={"reference":"Example 0 for clinc150","candidates":{"A":"Example 1 for clinc150","B":"Example 6 for clinc150","C":"Example 7 for clinc150"}}; instruction='Which candidate belongs to the same underlying class as the reference?'; options=["A","B","C"]; target='B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_reference_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_odd_one_out(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows sharing one intent and one row with a different intent; ask which assistant query is the odd one out by class.
    
        Target derivation: Target is the row whose gold class differs.
        Rows used: 3. Selection: Randomize item order.; Avoid duplicate text.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state={"items":{"A":"Example 1 for clinc150","B":"Example 0 for clinc150","C":"Example 8 for clinc150"}}; instruction='Two items share the same underlying class. Which item is the odd one out?'; options=["A","B","C"]; target='A'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_odd_one_out", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_in_scope_vs_oos(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether the query belongs to one of the named in-scope intents rather than the native out-of-scope class.
    
        Target derivation: Positive iff gold class is not native oos.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="Example 0 for clinc150"; instruction='Does this query belong to one of the named in-scope intents?'; options=["belongs","does not belong"]; target='belongs'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_in_scope_vs_oos", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_oos_candidate_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask directly whether the query is out of scope for the known intent inventory.
    
        Target derivation: Positive iff gold class is native oos.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state="Example 0 for clinc150"; instruction='Is this query out of scope for the known intent inventory?'; options=["belongs","does not belong"]; target='does not belong'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_oos_candidate_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def clinc150_two_query_truth_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose candidate intent X and ask whether X describes the first query only, second only, both, or neither.
    
        Target derivation: Derive from equality of X with each gold label.
        Rows used: 2. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for clinc150"; gold="set_alarm".
          Reformulated output: primitive='choice'; state={"candidate":"set_alarm","first":"Example 0 for clinc150","second":"Example 1 for clinc150"}; instruction='For which of the two inputs does the proposed label apply?'; options=["A only","second only","both match","neither applies"]; target='A only'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("clinc150_two_query_truth_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "clinc150_direct_rephrase": clinc150_direct_rephrase,
    "clinc150_candidate_verify": clinc150_candidate_verify,
    "clinc150_subset_membership": clinc150_subset_membership,
    "clinc150_partition_group": clinc150_partition_group,
    "clinc150_other_gold_removed": clinc150_other_gold_removed,
    "clinc150_other_gold_present": clinc150_other_gold_present,
    "clinc150_unknown_missing_input": clinc150_unknown_missing_input,
    "clinc150_other_unknown_joint": clinc150_other_unknown_joint,
    "clinc150_opaque_codebook": clinc150_opaque_codebook,
    "clinc150_same_class_pair": clinc150_same_class_pair,
    "clinc150_reference_match": clinc150_reference_match,
    "clinc150_odd_one_out": clinc150_odd_one_out,
    "clinc150_in_scope_vs_oos": clinc150_in_scope_vs_oos,
    "clinc150_oos_candidate_verify": clinc150_oos_candidate_verify,
    "clinc150_two_query_truth_pattern": clinc150_two_query_truth_pattern,
}
