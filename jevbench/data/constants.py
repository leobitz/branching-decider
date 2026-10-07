"""Shared dataset and task constants."""

DATASET_NAME_DEFAULT = "Praveenrajus/jev-bench"
PROCESSING_VERSION = "jev_bench_text_candidates_v1"
PROCESSED_SPLITS = ("train", "validation", "test")

NOUL_SURFACE_PAIRS = (
    ("yes", "no"),
    ("true", "false"),
    ("correct", "incorrect"),
    ("affirmative", "negative"),
    ("supported", "not supported"),
    ("yes / true", "no / false"),
)

SCORE_SURFACE_STYLES = (
    "key_description",
    "score_key_description",
    "level_key_description",
    "rating_position_description",
    "description_only",
    "option_position_description",
)

CONFIG_DOMAINS = {
    "banking77": "customer-support",
    "clinc150": "virtual-assistant",
    "massive": "virtual-assistant",
    "ledgar": "legal",
    "go_emotions": "social",
    "mmlu": "knowledge",
    "arc_challenge": "knowledge",
    "mnli": "nli",
    "chaosnli": "nli",
    "sst5": "reviews",
    "yelp5": "reviews",
    "helpsteer2_helpfulness": "llm-judging",
    "helpsteer2_verbosity": "llm-judging",
    "stsb": "nli",
    "measuring_hate_speech": "safety",
    "boolq": "reading-comprehension",
    "fever_evidence": "fact-checking",
    "paws": "nli",
    "civil_comments": "safety",
    "sms_spam": "messaging",
    "strategyqa_closed": "knowledge",
    "strategyqa_grounded": "knowledge",
}

TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}
TASK_PROMPTS = {
    "choice": "Choose the best option.",
    "score": "Rate it on the given scale.",
    "noul": "Answer yes or no.",
}
