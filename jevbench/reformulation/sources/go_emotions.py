from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def go_emotions_direct_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rephrase the instruction while preserving the same comment and full original label set.
    
        Target derivation: Target is the original gold class.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Do not paraphrase the content being classified unless semantics are guaranteed unchanged.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state="Example 0 for go_emotions"; instruction='Which provided category most accurately describes the input? Original task: Choose the class.'; options=["joy","neutral","sadness","refund","payment","fear"]; target='joy'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_direct_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_candidate_verify(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample one candidate primary emotion; ask whether that candidate is the correct primary emotion for the comment.
    
        Target derivation: Positive iff candidate equals original gold class.
        Rows used: 1. Selection: 50% gold candidate, 50% non-gold candidate; prefer hard/nearby negatives when available.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: If soft_label distribution exists, P(positive)=p(candidate emotion).
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for go_emotions","candidate_label":"fear","candidate_description":"fear"}; instruction='Is the proposed primary emotion the correct one for this comment?'; options=["correct","incorrect"]; target='incorrect'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_candidate_verify", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_subset_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample a subset of primary emotion labels; ask whether the correct primary emotion is contained in that set.
    
        Target derivation: Positive iff original gold class is in sampled subset.
        Rows used: 1. Selection: Vary subset size from 1 to min(8, K-1). Balance include-gold vs exclude-gold.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: P(positive)=sum of probabilities for emotions in subset.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state={"input":"Example 0 for go_emotions","candidate_set":{"refund":"refund","payment":"payment","joy":"joy","neutral":"neutral","sadness":"sa...; instruction='Is the correct primary emotion for this comment contained in `candidate_set`?'; options=["inside the set","outside the set"]; target='inside the set'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_subset_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_partition_group(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Partition a sampled candidate label set into 2-4 temporary groups and ask which group contains the correct primary emotion.
    
        Target derivation: Target is the group containing the original gold class.
        Rows used: 1. Selection: Ensure gold is included in exactly one group.; Randomize group names and group order.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state="Example 0 for go_emotions"; instruction='Which temporary group contains the correct primary emotion for this comment?'; options=["Set Four","Set One"]; target='Set Four'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_partition_group", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_other_gold_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer a subset of ordinary primary emotion labels that deliberately excludes the gold class, plus a sampled OTHER surface form.
    
        Target derivation: Target is canonical OTHER.
        Rows used: 1. Selection: Keep enough plausible distractors that OTHER is not trivial.
        Validity guard: Original problem must remain fully answerable.
        Soft-label rule: For a choice over retained emotions plus OTHER, map all omitted emotion probability mass into OTHER.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state="Example 0 for go_emotions"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["surprise","refund","sadness","notice","support","an unlisted option"]; target='an unlisted option'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_other_gold_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_other_gold_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Offer the gold primary emotion, several distractors, and a sampled OTHER surface form.
    
        Target derivation: Target remains the ordinary gold class; OTHER is a distractor.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: For a choice over retained emotions plus OTHER, map all omitted emotion probability mass into OTHER.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state="Example 0 for go_emotions"; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["joy","payment","notice","neutral","refund","the correct answer is not included"]; target='joy'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_other_gold_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_unknown_missing_input(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Replace the required comment with an explicit missing marker or omit that required field; offer ordinary candidates plus UNKNOWN.
    
        Target derivation: Target is canonical UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: The question must explicitly restrict the decision to the supplied input so outside knowledge cannot repair the missing field.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state="[MISSING]"; instruction='Decide only from the supplied input. If the required information is unavailable, choose the insufficient-information option.'; options=["joy","surprise","sadness","refund","support","neutral"]; target='answer cannot be inferred'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_unknown_missing_input", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Generate the three global joint cases using ordinary primary emotion labels plus both OTHER and UNKNOWN.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: Balance answer-present, answer-missing, and unanswerable cases.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state="The required information was not supplied."; instruction='Choose the best answer from the supplied options. Use the unlisted-answer option only when the problem is answerable but its answer is absent; use the insufficient-information option only when the supplied input cannot determine an answer.'; options=["surprise","notice","refund","fear","neutral","none apply"]; target='cannot be determined'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_opaque_codebook(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Rename a sampled set of primary emotion labels with temporary opaque keys while preserving readable descriptions, then ask for the key matching the comment.
    
        Target derivation: Target is the temporary key mapped to the original gold class.
        Rows used: 1. Selection: Regenerate key mapping per example or family.; Do not let alphabetical order encode the original label.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state="Example 0 for go_emotions"; instruction='Use the temporary codebook below and select the code whose description best matches the input.'; options=["EV","R6","FW","Z4","HY","KR"]; target='V4'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_opaque_codebook", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_same_class_pair(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows and ask whether they share the same primary emotion. Present both comments.
    
        Target derivation: Positive iff the two original gold classes are equal.
        Rows used: 2. Selection: Balance same-class and different-class pairs.; Use different raw examples even for same-class pairs.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state={"first":"Example 0 for go_emotions","second":"Example 2 for go_emotions"}; instruction='Do the first and second comments have the same primary emotion?'; options=["equivalent","not equivalent"]; target='equivalent'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_same_class_pair", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_reference_match(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use one row as a reference and 2-4 other rows as candidates; ask which candidate has the same primary emotion as the reference.
    
        Target derivation: Target is the unique candidate whose original gold class equals the reference gold class.
        Rows used: 3-5. Selection: Require exactly one matching candidate.; All rows from same split.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state={"reference":"Example 0 for go_emotions","candidates":{"A":"Example 3 for go_emotions","B":"Example 2 for go_emotions","C":"Example 9 for...; instruction='Which candidate belongs to the same underlying class as the reference?'; options=["A","B","C"]; target='B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_reference_match", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_odd_one_out(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Sample two rows sharing one primary emotion and one row with a different primary emotion; ask which comment is the odd one out by class.
    
        Target derivation: Target is the row whose gold class differs.
        Rows used: 3. Selection: Randomize item order.; Avoid duplicate text.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state={"items":{"A":"Example 4 for go_emotions","B":"Example 2 for go_emotions","C":"Example 0 for go_emotions"}}; instruction='Two items share the same underlying class. Which item is the odd one out?'; options=["A","B","C"]; target='A'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_odd_one_out", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_agreement_threshold(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Use the human vote distribution and ask whether the leading emotion receives at least threshold t of votes.
    
        Target derivation: Positive iff max(soft_label) >= t.
        Rows used: 1. Selection: Sample t from a fixed grid such as 0.5, 0.6, 0.7, 0.8 with class balancing.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state="Example 0 for go_emotions"; instruction='Did the leading human label receive at least 60% of the annotation probability?'; options=["meets the condition","does not meet the condition"]; target='meets the condition'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_agreement_threshold", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_compare_agreement(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Present two comments and ask which has stronger human agreement on its leading emotion, or whether they are approximately tied.
    
        Target derivation: Compare max soft-label probabilities; include tie only when within configured epsilon.
        Rows used: 2. Selection: Require a minimum margin for non-tie examples.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state={"first":"Example 0 for go_emotions","second":"Example 3 for go_emotions"}; instruction='Which item has stronger human agreement on its leading label?'; options=["first item","second","tie"]; target='tie'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_compare_agreement", row, pool=pool, seed=seed, validator=validator, **kwargs)

def go_emotions_same_primary_pair_soft(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Ask whether two comments have the same primary emotion.
    
        Target derivation: Hard target uses argmax emotions.
        Rows used: 2. Selection: Balance same and different argmax emotion.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: Optional independent-rater framing: P(same)=sum_k p1(k)*p2(k).
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: "Example 0 for go_emotions"; gold="joy".
          Reformulated output: primitive='choice'; state={"first":"Example 0 for go_emotions","second":"Example 11 for go_emotions"}; instruction='Do the first and second comments have the same primary emotion?'; options=["same","different"]; target='different'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("go_emotions_same_primary_pair_soft", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "go_emotions_direct_rephrase": go_emotions_direct_rephrase,
    "go_emotions_candidate_verify": go_emotions_candidate_verify,
    "go_emotions_subset_membership": go_emotions_subset_membership,
    "go_emotions_partition_group": go_emotions_partition_group,
    "go_emotions_other_gold_removed": go_emotions_other_gold_removed,
    "go_emotions_other_gold_present": go_emotions_other_gold_present,
    "go_emotions_unknown_missing_input": go_emotions_unknown_missing_input,
    "go_emotions_other_unknown_joint": go_emotions_other_unknown_joint,
    "go_emotions_opaque_codebook": go_emotions_opaque_codebook,
    "go_emotions_same_class_pair": go_emotions_same_class_pair,
    "go_emotions_reference_match": go_emotions_reference_match,
    "go_emotions_odd_one_out": go_emotions_odd_one_out,
    "go_emotions_agreement_threshold": go_emotions_agreement_threshold,
    "go_emotions_compare_agreement": go_emotions_compare_agreement,
    "go_emotions_same_primary_pair_soft": go_emotions_same_primary_pair_soft,
}
