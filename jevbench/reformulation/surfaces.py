"""Diverse, semantically compatible output-label surfaces.

Canonical targets (OTHER, UNKNOWN, FIRST_ONLY...) stay in metadata.  Only the
user-facing surface changes; unrelated yes/no and same/different labels never
mix into one binary pair.
"""
from __future__ import annotations
import random

BINARY_PAIRS = [
    ('yes','no'), ('true','false'), ('correct','incorrect'), ('right','wrong'),
    ('affirmative','negative'), ('applies','does not apply'), ('matches','does not match'),
    ('same','different'), ('equivalent','not equivalent'), ('supported','not supported'),
    ('entailed','not entailed'), ('consistent','inconsistent'), ('included','not included'),
    ('inside the set','outside the set'), ('belongs','does not belong'),
    ('present','absent'), ('meets the condition','does not meet the condition'),
    ('valid','invalid'), ('agrees','disagrees'), ('Y','N'), ('T','F'),
    ('1 - yes','0 - no'), ('accepted','rejected'),
]
BINARY_GROUPS = {
    'generic': (0,1,2,3,4,19,20,21),
    'verify': (0,1,2,3,4,19,20,21),
    'membership': (0,1,5,12,13,14,15,16,19,20),
    'same': (0,1,7,8,17,19,20),
    'evidence': (0,1,9,10,11,19,20),
    'threshold': (0,1,5,16,19,20),
}
OTHER = [
    'other','something else','another answer','a different answer','another option',
    'none of these','none of the above','none of the listed choices',
    'none of the provided options','none of the provided labels','none apply',
    'no listed option is correct','not among these','not listed',
    'an unlisted answer','an unlisted option','an unlisted category',
    'a category not shown','a choice not shown','outside the provided options',
    'outside the listed labels','not represented here','not represented by these choices',
    'different from all listed options','the correct answer is not included',
    'the correct label is missing','some other category','some other choice',
    'other / unlisted','none match','no offered label fits','elsewhere',
]
UNKNOWN = [
    'unknown',"I don't know",'not sure','cannot determine','cannot be determined',
    'not enough information','insufficient information','insufficient evidence',
    'not enough evidence','undetermined','indeterminate','unclear',
    'unclear from the information given','cannot answer from the provided information',
    'cannot conclude from the provided information','not established',
    'not decidable from the context','information is missing','required information is missing',
    'need more information','more information is needed','unable to tell',
    'not enough context','insufficient context','unresolvable from the given data',
    'answer cannot be inferred','the evidence is insufficient','no basis to decide',
    'not knowable from what is provided','cannot conclude','unanswerable as stated',
    'unspecified','not provided','cannot be resolved','cannot tell from this context',
    'the available information does not decide it','there is not enough to decide',
]
PATTERN = {
    'first_only': ['first only','A only','only the first','the first but not the second'],
    'second_only': ['second only','B only','only the second','the second but not the first'],
    'both': ['both','both are true','both apply','both match','A and B','each of them','the two of them'],
    'neither': ['neither','neither is true','neither applies','neither matches','neither A nor B','none of the two'],
}
GROUP_NAMES = ['Group Alpha','Group Beta','Group Gamma','Group Delta','Set One','Set Two','Set Three','Set Four','Bucket R','Bucket S','Bucket T','Bucket U']
OPAQUE_KEYS = ['K7','M2','Q9','V4','R6','T3','X8','P5','N1','Z4','AX','BZ','CQ','DR','EV','FW','GX','HY','JQ','KR']


def binary(rng: random.Random, kind: str = 'generic') -> tuple[str, str]:
    return BINARY_PAIRS[rng.choice(BINARY_GROUPS.get(kind, BINARY_GROUPS['generic']))]


def surface(rng: random.Random, forms: list[str], used=()) -> str:
    existing = {str(x).casefold() for x in used}
    available = [x for x in forms if x.casefold() not in existing]
    if not available:
        raise ValueError('No unique surface label available')
    return rng.choice(available)


def truth_pattern(rng: random.Random) -> dict[str, str]:
    used=[]
    result={}
    for key, options in PATTERN.items():
        result[key]=surface(rng, options, used)
        used.append(result[key])
    return result


def codes(rng: random.Random, count: int) -> list[str]:
    keys = OPAQUE_KEYS + [f'CODE_{i}' for i in range(max(0, count-len(OPAQUE_KEYS)))]
    return rng.sample(keys, count)
