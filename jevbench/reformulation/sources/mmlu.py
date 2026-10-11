from __future__ import annotations

from typing import Any, Mapping, Sequence
from ..engine import apply_strategy
from ..record import Record

def mmlu_direct_mcq_rephrase(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Keep the MMLU question stem and answer texts; vary only the instruction and option key surface.
    
        Target derivation: Target is the original correct answer text after remapping option keys.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Never preserve original option letters as the only identity of an answer.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"question":"Synthetic question 0?","subject":"science"}; instruction='Select the correct answer to the question.'; options=["FW","Z4","EV","X8"]; target='FW'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_direct_mcq_rephrase", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_candidate_correctness(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Select one answer choice and ask whether it is the correct answer to the question.
    
        Target derivation: Positive iff selected choice is the original gold choice.
        Rows used: 1. Selection: 50% correct candidate, 50% incorrect candidate.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"input":{"question":"Synthetic question 0?","subject":"science"},"proposed_answer":"alpha answer"}; instruction='Is `proposed_answer` the correct answer to the question?'; options=["right","wrong"]; target='right'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_candidate_correctness", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_answer_subset_membership(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Select a subset of answer choices and ask whether the correct answer is among them.
    
        Target derivation: Positive iff original gold answer is in the subset.
        Rows used: 1. Selection: Balance inclusion/exclusion; vary subset size.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"input":{"question":"Synthetic question 0?","subject":"science"},"answer_subset":["alpha answer","delta answer"]}; instruction='Is the correct answer contained in `answer_subset`?'; options=["included","not included"]; target='included'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_answer_subset_membership", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_answer_partition(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Partition the original answer choices into temporary groups and ask which group contains the correct answer.
    
        Target derivation: Target is the group containing the gold answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"question":"Synthetic question 0?","subject":"science"}; instruction='Which temporary answer group contains the correct answer?'; options=["Set Two","Bucket U"]; target='Bucket U'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_answer_partition", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_other_gold_removed(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove the correct answer from the displayed choices, retain plausible distractors, and add an OTHER surface.
    
        Target derivation: Target is OTHER.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Question stem remains unchanged and answerable.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"question":"Synthetic question 0?","subject":"science"}; instruction='Answer the question. Distinguish a missing answer choice from genuinely insufficient question information.'; options=["B","C","D","a different answer"]; target='a different answer'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_other_gold_removed", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_other_gold_present(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Keep the correct answer and distractors, add OTHER as an extra option.
    
        Target derivation: Target remains the original correct answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"question":"Synthetic question 0?","subject":"science"}; instruction='Answer the question. Distinguish a missing answer choice from genuinely insufficient question information.'; options=["A","B","C","D","none match"]; target='A'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_other_gold_present", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_unknown_missing_stem(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Hide the question stem or replace the decisive content with an explicit missing marker while retaining answer options and UNKNOWN.
    
        Target derivation: Target is UNKNOWN.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Use only deterministic missingness such as removing the full stem; partial ablation needs validation.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"question":"[UNAVAILABLE]","subject":"science"}; instruction='Choose an answer only if the supplied question is sufficient; otherwise choose the insufficient-information option.'; options=["A","B","C","D","unanswerable as stated"]; target='unanswerable as stated'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_unknown_missing_stem", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_other_unknown_joint(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Generate ordinary-answer, OTHER-correct, and UNKNOWN-correct cases with both special labels present.
    
        Target derivation: Use global joint_other_unknown_template.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"question":"Synthetic question 0?","subject":"science"}; instruction='Answer the question. Distinguish a missing answer choice from genuinely insufficient question information.'; options=["A","B","C","D","none of the above","not enough context"]; target='A'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_other_unknown_joint", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_two_question_candidate_pattern(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        For two independent MCQs choose one candidate answer from each. Ask which candidate assertions are correct: first only, second only, both, or neither.
    
        Target derivation: Each candidate assertion is true iff that selected answer equals that row gold answer.
        Rows used: 2. Selection: Sample all four truth patterns.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"first":{"question":"Synthetic question 0?","subject":"science"},"first_candidate":"gamma answer","second":{"question":"Synthetic questi...; instruction='Which proposed answers are correct?'; options=["A only","B only","both apply","neither A nor B"]; target='neither A nor B'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_two_question_candidate_pattern", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_same_candidate_truth_status(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Choose one candidate answer for each of two questions and ask whether the two candidate assertions have the same truth status.
    
        Target derivation: Positive iff both candidates are correct or both are incorrect.
        Rows used: 2. Selection: Balance same vs different truth status.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"first":{"question":"Synthetic question 0?","subject":"science"},"first_candidate":"alpha answer","second":{"question":"Synthetic questi...; instruction='Do the two proposed answers have the same correctness status?'; options=["same","different"]; target='different'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_same_candidate_truth_status", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_opaque_answer_keys(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Replace A/B/C/D or source option keys with random opaque keys while preserving answer text.
    
        Target derivation: Target is the random key attached to the original gold answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: Randomize mapping independently; do not encode old position in the key.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: No.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"question":"Synthetic question 0?","subject":"science"}; instruction='Which candidate best answers the question?'; options=["N1","BZ","EV","JQ"]; target='N1'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_opaque_answer_keys", row, pool=pool, seed=seed, validator=validator, **kwargs)

def mmlu_partial_stem_ablation_optional(row: Mapping[str, Any] | Record, *, pool: Sequence[Mapping[str, Any] | Record] | None = None, seed: int = 0, validator=None, **kwargs: Any) -> dict[str, Any]:
    """
        Remove one information-bearing span from the stem and ask the original question with UNKNOWN available.
    
        Target derivation: UNKNOWN only if a validator confirms the remaining text no longer determines a unique answer.
        Rows used: 1. Selection: No special row-selection rule.
        Validity guard: No additional semantic guard beyond preserving the source label semantics.
        Soft-label rule: No special soft-label propagation rule is required.
        Validator required: Yes: pass validator(transformed_row, original_row) -> bool.
    
        Typical transformation example (generated from a synthetic row using this function):
          Input example: {"question":"Synthetic question 0?","subject":"science"}; gold="A".
          Reformulated output: primitive='choice'; state={"question":"[MISSING INFORMATION]","subject":"science"}; instruction='Answer from the remaining question text; choose insufficient information if it no longer determines a unique answer.'; options=["A","B","C","D","information is missing"]; target='information is missing'.
    
        Returns a JEV-Bench-compatible serialized row dictionary. Multi-row
        strategies sample required comparison rows from ``pool``.
    """
    return apply_strategy("mmlu_partial_stem_ablation_optional", row, pool=pool, seed=seed, validator=validator, **kwargs)

STRATEGIES = {
    "mmlu_direct_mcq_rephrase": mmlu_direct_mcq_rephrase,
    "mmlu_candidate_correctness": mmlu_candidate_correctness,
    "mmlu_answer_subset_membership": mmlu_answer_subset_membership,
    "mmlu_answer_partition": mmlu_answer_partition,
    "mmlu_other_gold_removed": mmlu_other_gold_removed,
    "mmlu_other_gold_present": mmlu_other_gold_present,
    "mmlu_unknown_missing_stem": mmlu_unknown_missing_stem,
    "mmlu_other_unknown_joint": mmlu_other_unknown_joint,
    "mmlu_two_question_candidate_pattern": mmlu_two_question_candidate_pattern,
    "mmlu_same_candidate_truth_status": mmlu_same_candidate_truth_status,
    "mmlu_opaque_answer_keys": mmlu_opaque_answer_keys,
    "mmlu_partial_stem_ablation_optional": mmlu_partial_stem_ablation_optional,
}
