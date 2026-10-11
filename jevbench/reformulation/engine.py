"""Executable transformation operators for all JEV-Bench source strategy functions.

Input and output use the native JEV-Bench row wire format. This module is only
imported by the dedicated reformulation script, not by training or any models.

The operators derive labels from seed gold annotations; they do not use an LLM
as an unverified gold-label generator. Partial semantic ablations need a caller
validator and are skipped by the automatic CLI.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from .record import Record
from .surfaces import BINARY_PAIRS, GROUP_NAMES, OTHER, UNKNOWN, PATTERN, binary, codes, surface, truth_pattern

CHOICE_SOURCES = {'banking77','clinc150','massive','ledgar','go_emotions'}
MCQ_SOURCES = {'mmlu','arc_challenge'}
NLI_SOURCES = {'mnli','chaosnli'}
SCORE_SOURCES = {'sst5','yelp5','helpsteer2_helpfulness','helpsteer2_verbosity','stsb','measuring_hate_speech'}
BINARY_SOURCES = {'boolq','fever_evidence','paws','civil_comments','sms_spam','strategyqa_closed','strategyqa_grounded'}

class InsufficientPoolError(ValueError):
    """No eligible other row in the same source and split."""

class ValidationRequiredError(ValueError):
    """Partial ablation requires semantic validation of insufficient evidence."""


def hashed_seed(*items: Any) -> int:
    v=json.dumps(items,sort_keys=True,ensure_ascii=False,default=str).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(v,digest_size=8).digest(),'big')


class PoolIndex:
    """Indexed read-only pool for fast multi-row selection within one source/split."""
    def __init__(self, rows: Sequence[Mapping[str,Any]]):
        self.rows=[Record.from_row(x) for x in rows]
        self.by_label: dict[str,list[Record]]=defaultdict(list)
        for x in self.rows:
            self.by_label[x.label].append(x)

    def choose(self, anchor: Record, rng: random.Random, *, same: bool | None = None,
               exclude: set[str] | None = None, has_soft: bool=False) -> Record:
        exclude=set(exclude or ()) | {anchor.id}
        if same is True:
            candidates=self.by_label.get(anchor.label, [])
        elif same is False:
            # Sample label first to avoid scanning huge pools at every call.
            labels=[k for k,v in self.by_label.items() if k!=anchor.label and len(v)>0]
            rng.shuffle(labels)
            candidates=[]
            for k in labels:
                possible=[r for r in self.by_label[k] if r.id not in exclude and (not has_soft or r.soft_label is not None)]
                if possible:
                    return rng.choice(possible)
        else:
            candidates=self.rows
        if not candidates:
            raise InsufficientPoolError('No comparison row with the required class relationship')
        # Rejection sampling is constant time for ordinary pools.
        for _ in range(32):
            selected=rng.choice(candidates)
            if selected.id not in exclude and (not has_soft or selected.soft_label is not None) and selected.split==anchor.split and selected.source==anchor.source:
                return selected
        allowed=[r for r in candidates if r.id not in exclude and (not has_soft or r.soft_label is not None) and r.source==anchor.source and r.split==anchor.split]
        if not allowed:
            raise InsufficientPoolError('No valid distinct comparison row in this source/split')
        return rng.choice(allowed)


def pool_index(raw: PoolIndex | Sequence[Mapping[str,Any]] | None) -> PoolIndex:
    if isinstance(raw,PoolIndex):
        return raw
    if raw is None:
        raise InsufficientPoolError('Multi-row strategy requires a same-source/same-split pool')
    return PoolIndex(raw)


def _gold_boolean(r: Record) -> bool:
    text=str(r.label).strip().casefold()
    if text in {'1','true','yes','y','positive'}: return True
    if text in {'0','false','no','n','negative'}: return False
    raise ValueError(f'Invalid binary source label: {r.label!r}')


def _description(value: Any) -> str:
    if isinstance(value,dict):
        for key in ('description','text','name','label'):
            if value.get(key): return str(value[key])
    return str(value) if value is not None else ''


def _normalized_soft(record: Record) -> dict[str,float] | None:
    p=record.probabilities()
    if p is None: return None
    if record.primitive=='noul':
        out={}
        for k,v in p.items():
            kl=str(k).lower()
            if kl in {'1','true','yes','positive'}: out['1']=out.get('1',0)+v
            elif kl in {'0','false','no','negative'}: out['0']=out.get('0',0)+v
        return out or None
    return p


def _soft_sum(record: Record, keys: Sequence[str]) -> float | None:
    p=_normalized_soft(record)
    return None if p is None else sum(p.get(str(x),0.0) for x in keys)


def _state_with(record: Record, **kwargs: Any) -> Any:
    return {'input':copy.deepcopy(record.state),**kwargs}


def _missing_state() -> dict[str,str]:
    # Intentional full omission: the gold label cannot be deduced from *supplied*
    # evidence even where a world-knowledge guess would have been possible.
    return {'required_input':'[NOT PROVIDED]','notice':'Required facts/text/question intentionally omitted.'}


def _pattern(a: bool, b: bool) -> str:
    return 'both' if a and b else 'first_only' if a else 'second_only' if b else 'neither'


def _make(strategy: str, parents: Sequence[Record], rng: random.Random, seed: int,
          state: Any, instruction: str, options: Mapping[str,Any] | None, label: str | int,
          *, primitive: str='choice', soft: dict[str,float] | list[float] | None=None,
          canonical: str | None=None, detail: dict[str,Any] | None=None) -> dict[str,Any]:
    r=parents[0]
    if primitive=='score':
        criteria=list(options.values()) if isinstance(options,dict) else list(options or [])
    else:
        if not options or len(options)<2:
            raise ValueError(f'{strategy}: requires at least two choices')
        items=list(options.items())
        rng.shuffle(items)
        criteria=dict(items)
        if str(label) not in criteria:
            raise ValueError(f'{strategy}: target {label!r} missing from candidate set')
    q={'type':primitive,'instructions':instruction,'criteria':criteria}
    parent_ids=[x.id for x in parents]
    fingerprint=hashlib.sha1(json.dumps([strategy,parent_ids,seed,state,q,str(label)],sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()[:16]
    meta=dict(r.meta)
    meta.update({
        'reformulation_strategy':strategy,
        'parent_ids':parent_ids,
        'parent_labels':[x.label for x in parents],
        'canonical_target':canonical if canonical is not None else str(label),
        'details':detail or {},
    })
    out=Record(
        id=f'aug:{r.source}:{r.split}:{fingerprint}', source=r.source,split=r.split,
        primitive=primitive,state=state,question=q,label=str(label),soft_label=soft,meta=meta,
    )
    return out.to_row()


def _binary(strategy: str, parents: Sequence[Record], rng: random.Random, seed: int,
            state: Any, instruction: str, correct: bool, *, kind: str='generic',
            probability: float | None=None, detail: dict[str,Any] | None=None) -> dict[str,Any]:
    yes,no=binary(rng,kind)
    p={yes:float(probability),no:1-float(probability)} if probability is not None else None
    return _make(strategy,parents,rng,seed,state,instruction,
                 {yes:'The stated condition holds.',no:'The stated condition does not hold.'},
                 yes if correct else no,soft=p,canonical='TRUE' if correct else 'FALSE',
                 detail={**(detail or {}),'positive_surface':yes,'negative_surface':no})


def _extra_choices(rng: random.Random, ordinary: dict[str,Any], *, add_other: bool, add_unknown: bool) -> tuple[dict[str,Any],str|None,str|None]:
    c=dict(ordinary)
    other=surface(rng,OTHER,c) if add_other else None
    if other:
        c[other]='The answer is determinable, but its correct ordinary label was not offered.'
    unknown=surface(rng,UNKNOWN,c) if add_unknown else None
    if unknown:
        c[unknown]='The necessary information was not provided; no determinate answer follows.'
    return c,other,unknown


def _other_choice(strategy: str, r: Record, rng: random.Random, seed: int,
                  *, case: str='answer_missing', use_unknown: bool=False, score: bool=False) -> dict[str,Any]:
    ordinary=r.criteria()
    if not ordinary:
        raise ValueError('A choice or score source must provide options')
    if case not in {'answer_present','answer_missing','unanswerable'}:
        raise ValueError('Unknown answerability case')
    keys=list(ordinary)
    gold=r.label
    if gold not in ordinary: raise ValueError('Gold not in original class inventory')
    if case=='answer_missing':
        # Large-K source inventories (e.g. 151 CLINC intents) should not
        # explode to 151-way variants. Keep diverse negative subsets instead.
        distractors=[k for k in keys if k!=gold]
        count=rng.randint(1,min(12,len(distractors)))
        keep=rng.sample(distractors,count)
    elif case=='answer_present':
        # Keep gold plus a variable-sized diverse subset of distractors.
        distractors=[k for k in keys if k!=gold]
        count=rng.randint(1,min(12,len(distractors)))
        keep=[gold]+rng.sample(distractors,count)
    else:
        keep=rng.sample(keys,min(len(keys),6))
    # Include OTHER as distractor when known answer offered.
    c,other,unknown=_extra_choices(rng,{k:ordinary[k] for k in keep},add_other=True,add_unknown=use_unknown or case=='unanswerable')
    chosen=unknown if case=='unanswerable' else other if case=='answer_missing' else gold
    state=_missing_state() if case=='unanswerable' else copy.deepcopy(r.state)
    instr='Answer using only the supplied input. Choose an ordinary label if present; otherwise choose the unlisted-answer label only when the answer is known but omitted, or the insufficient-information label only when the input does not determine an answer.'
    soft=None
    p=_normalized_soft(r)
    if p is not None:
        if case=='unanswerable':
            soft={k:0.0 for k in c}; soft[unknown]=1.0
        else:
            soft={k:p.get(k,0.0) for k in keep}
            soft[other]=max(0.0,1-sum(soft.values()))
            if unknown: soft[unknown]=0.0
    return _make(strategy,[r],rng,seed,state,instr,c,chosen,soft=soft,
                 canonical={'answer_present':gold,'answer_missing':'OTHER','unanswerable':'UNKNOWN'}[case],
                 detail={'joint_case':case,'other_surface':other,'unknown_surface':unknown,'gold_original':gold,'retained_labels':keep})


def _unknown_choice(strategy: str,r:Record,rng:random.Random,seed:int, *, use_other:bool=False) -> dict[str,Any]:
    c=dict(r.criteria())
    if r.primitive=='noul':
        yes,no=binary(rng); c={yes:'Yes / positive.',no:'No / negative.'}
    if len(c)>7:
        gold=r.label
        keys=rng.sample(list(c),min(6,len(c)))
        if gold in c and gold not in keys: keys[0]=gold
        c={k:c[k] for k in keys}
    c,other,unknown=_extra_choices(rng,c,add_other=use_other,add_unknown=True)
    return _make(strategy,[r],rng,seed,_missing_state(),
                 'The required input was withheld. Decide only from supplied information; do not guess from prior knowledge. Choose the insufficient-information answer.',
                 c,unknown,soft={k:float(k==unknown) for k in c},canonical='UNKNOWN',
                 detail={'missing_entire_required_input':True,'unknown_surface':unknown,'other_surface':other})


def _coded_choice(strategy:str,r:Record,rng:random.Random,seed:int, *, include_all:bool=False) -> dict[str,Any]:
    crit=r.criteria(); keys=list(crit)
    if not crit: raise ValueError('No classes for codebook')
    if not include_all and len(keys)>12:
        keys=[r.label]+rng.sample([k for k in keys if k!=r.label],11)
    replacements=codes(rng,len(keys))
    mapping=dict(zip(keys,replacements))
    options={mapping[k]:_description(crit[k]) for k in keys}
    p=_normalized_soft(r)
    soft=None
    if p is not None:
        soft={mapping[k]:p.get(k,0.0) for k in keys}
        total=sum(soft.values())
        if total: soft={k:v/total for k,v in soft.items()}
    return _make(strategy,[r],rng,seed,copy.deepcopy(r.state),
                 'Use only this temporary codebook; return the code whose definition best describes the input.',
                 options,mapping[r.label],soft=soft,canonical=r.label,detail={'codebook':mapping})


def _subset(strategy:str,r:Record,rng:random.Random,seed:int, *, mcq:bool=False, include_gold:bool|None=None) -> dict[str,Any]:
    crit=r.criteria(); keys=list(crit)
    if len(keys)<2: raise ValueError('Not enough classes')
    include_gold=bool(rng.randrange(2)) if include_gold is None else include_gold
    options=[k for k in keys if k!=r.label]
    size=rng.randint(1,min(len(keys)-1,8))
    subset=([r.label]+rng.sample(options,min(size-1,len(options)))) if include_gold else rng.sample(options,min(size,len(options)))
    state=_state_with(r,selected_answers=[crit[k] for k in subset] if mcq else {k:crit[k] for k in subset})
    return _binary(strategy,[r],rng,seed,state,
                   'Is the correct semantic answer included in the displayed set of candidates?',
                   r.label in subset,kind='membership',probability=_soft_sum(r,subset),detail={'selected_keys':subset})


def _partition(strategy:str,r:Record,rng:random.Random,seed:int, *, mcq:bool=False) -> dict[str,Any]:
    crit=r.criteria(); keys=list(crit)
    if len(keys)<2: raise ValueError('Not enough labels for partition')
    if len(keys)>12: keys=[r.label]+rng.sample([x for x in keys if x!=r.label],11)
    rng.shuffle(keys)
    n=rng.randint(2,min(4,len(keys)))
    names=rng.sample(GROUP_NAMES,n)
    buckets={name:[] for name in names}
    for i,k in enumerate(keys): buckets[names[i%n]].append(k)
    target=next(name for name,group in buckets.items() if r.label in group)
    descriptions={g:'; '.join(f'{k}: {_description(crit[k])}' for k in group) for g,group in buckets.items()}
    p=_normalized_soft(r)
    soft={g:sum(p.get(k,0) for k in group) for g,group in buckets.items()} if p else None
    if soft:
        total=sum(soft.values()); soft={k:v/total for k,v in soft.items()} if total else None
    return _make(strategy,[r],rng,seed,copy.deepcopy(r.state),
                 'Which temporary group contains the correct answer?',descriptions,target,
                 soft=soft,canonical=r.label,detail={'partition':buckets})


def _verify(strategy:str,r:Record,rng:random.Random,seed:int, *, mcq:bool=False, candidate:str|None=None, positive:bool|None=None) -> dict[str,Any]:
    crit=r.criteria(); keys=list(crit)
    if len(keys)<2: raise ValueError('No selectable distractors')
    if positive is None: positive=bool(rng.randrange(2))
    if candidate is None: candidate=r.label if positive else rng.choice([k for k in keys if k!=r.label])
    stated={'candidate_label':candidate,'candidate_definition':crit[candidate]}
    return _binary(strategy,[r],rng,seed,_state_with(r,**stated),
                   'Is this proposed candidate the correct answer to the supplied question?',candidate==r.label,
                   kind='verify',probability=_soft_sum(r,[candidate]),detail={'candidate':candidate})


def _ref_match(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int, *, n:int=2) -> dict[str,Any]:
    match=pool.choose(r,rng,same=True)
    candidates=[match]
    used={match.id}
    for _ in range(n-1):
        other=pool.choose(r,rng,same=False,exclude=used)
        candidates.append(other); used.add(other.id)
    rng.shuffle(candidates)
    labels=[f'Candidate {chr(65+i)}' for i in range(len(candidates))]
    state={'reference':copy.deepcopy(r.state),'candidates':{lab:copy.deepcopy(x.state) for lab,x in zip(labels,candidates)}}
    target=labels[candidates.index(match)]
    return _make(strategy,[r]+candidates,rng,seed,state,'Which candidate has the same category/relation as the reference?',{x:'Select this candidate' for x in labels},target,
                 detail={'parent_to_candidate':{lab:x.id for lab,x in zip(labels,candidates)}})


def _odd_one(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int) -> dict[str,Any]:
    match=pool.choose(r,rng,same=True)
    diff=pool.choose(r,rng,same=False,exclude={match.id})
    items=[r,match,diff];rng.shuffle(items)
    state={'items':{f'Item {i+1}':x.state for i,x in enumerate(items)}}
    target=f'Item {items.index(diff)+1}'
    return _make(strategy,items,rng,seed,state,'Two items share the same class/relation. Which one is the odd one out?',{f'Item {i+1}':'Pick this item' for i in range(3)},target)


def _same_class(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int,*,desired:bool|None=None,soft:bool=False) -> dict[str,Any]:
    if desired is None:desired=bool(rng.randrange(2))
    r2=pool.choose(r,rng,same=desired)
    p=None
    p1=_normalized_soft(r);p2=_normalized_soft(r2)
    if soft and p1 and p2:
        p=sum(v*p2.get(k,0.0) for k,v in p1.items())
    return _binary(strategy,[r,r2],rng,seed,{'first':r.state,'second':r2.state},
                   'Do the first and second items belong to the same semantic class/relation?',
                   r.label==r2.label,kind='same',probability=p)


def _truth_pattern(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int, *, chosen_label:str|None=None, mcq:bool=False) -> dict[str,Any]:
    r2=pool.choose(r,rng)
    if mcq:
        crit1=r.criteria();crit2=r2.criteria()
        # For MCQs, answer letters from separate questions have NO shared meaning.
        positive1=bool(rng.randrange(2));positive2=bool(rng.randrange(2))
        k1=r.label if positive1 else rng.choice([k for k in crit1 if k!=r.label])
        k2=r2.label if positive2 else rng.choice([k for k in crit2 if k!=r2.label])
        state={'first':r.state,'first_candidate':crit1[k1],'second':r2.state,'second_candidate':crit2[k2]}
        instr='Which of these two proposed MCQ answers is correct?'
        flag1=(k1==r.label);flag2=(k2==r2.label)
    elif r.primitive=='noul':
        state={'first':r.state,'second':r2.state}
        instr='Which of the two propositions are true under the original binary questions?'
        flag1=_gold_boolean(r);flag2=_gold_boolean(r2)
    else:
        crit=r.criteria()
        if chosen_label is None: chosen_label=r.label if bool(rng.randrange(2)) else rng.choice([k for k in crit if k!=r.label])
        state={'relation_or_class_to_check':chosen_label,'first':r.state,'second':r2.state}
        instr='For which input does the provided candidate relation/category apply?'
        flag1=r.label==chosen_label;flag2=r2.label==chosen_label
    forms=truth_pattern(rng)
    return _make(strategy,[r,r2],rng,seed,state,instr,
                 {label:f'Condition: {key}' for key,label in forms.items()},forms[_pattern(flag1,flag2)],
                 canonical=_pattern(flag1,flag2),detail={'truth_flags':[flag1,flag2]})


def _bool_pair(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int, *, op:str) -> dict[str,Any]:
    r2=pool.choose(r,rng)
    a=_gold_boolean(r);b=_gold_boolean(r2)
    truth={'and':a and b,'or':a or b,'xor':a!=b,'same':a==b}[op]
    instruction={'and':'Are both propositions true?','or':'Is at least one proposition true?',
                 'xor':'Is exactly one proposition true?','same':'Do the two original questions have the same true/false outcome?'}[op]
    p=None
    if r.source=='civil_comments' and op in {'and','or','xor'}:
        p1=_normalized_soft(r);p2=_normalized_soft(r2)
        if p1 and p2:
            v1=p1.get('1',0.0);v2=p2.get('1',0.0)
            p={'and':v1*v2,'or':v1+v2-v1*v2,'xor':v1*(1-v2)+(1-v1)*v2}[op]
    return _binary(strategy,[r,r2],rng,seed,{'first':r.state,'second':r2.state},instruction,truth,
                   kind='same' if op=='same' else 'generic',probability=p,detail={'operator':op,'independent_rater_soft':p is not None})


def _direct(strategy:str,r:Record,rng:random.Random,seed:int,*,kind:str) -> dict[str,Any]:
    if kind=='score':
        q=r.question
        prompt=rng.choice([
            'Assign the best fitting level on the ordered scale below.',
            'Rate the provided item using the supplied ordered levels.',
            'Which of these ordered ratings best represents the input?',
        ])
        return _make(strategy,[r],rng,seed,copy.deepcopy(r.state),prompt,r.criteria(),r.label,
                     primitive='score',soft=copy.deepcopy(r.soft_label),canonical=r.label)
    if kind=='binary':
        return _bool_direct(strategy,r,rng,seed)
    if kind=='mcq':
        return _coded_choice(strategy,r,rng,seed,include_all=True)
    if kind=='nli':
        instr='Which logical relation holds from premise to hypothesis (entailment, neutral, or contradiction)?'
    else:
        instr=rng.choice(['Select the best-matching category for the input.',
                          'Which provided label describes the supplied input?',
                          'Choose exactly one category using the label definitions.'])
    return _make(strategy,[r],rng,seed,copy.deepcopy(r.state),instr,r.criteria(),r.label,
                 soft=copy.deepcopy(r.soft_label),canonical=r.label)


def _mcq_rephrase(strategy:str,r:Record,rng:random.Random,seed:int) -> dict[str,Any]:
    """Same MCQ content and gold answer with a conventional new option order."""
    crit=r.criteria()
    keys=list(crit)
    if r.label not in crit: raise ValueError('Gold missing from original MCQ')
    rng.shuffle(keys)
    newkeys=[f'Option {i+1}' for i in range(len(keys))]
    mapping=dict(zip(keys,newkeys))
    options={mapping[k]:crit[k] for k in keys}
    instruction=rng.choice([
        'Select the correct answer to the question.',
        'Which of these options correctly answers the supplied question?',
        'Choose the one best answer using the candidate descriptions.',
    ])
    return _make(strategy,[r],rng,seed,copy.deepcopy(r.state),instruction,options,
                 mapping[r.label],canonical=r.label,detail={'remapped_option_keys':mapping})


def _bool_direct(strategy:str,r:Record,rng:random.Random,seed:int,*,invert:bool=False,statement:str|None=None) -> dict[str,Any]:
    truth=_gold_boolean(r)
    if invert:truth=not truth
    instruction=statement or ("Is the original binary proposition false (rather than true)?" if invert else
                              "Does the original binary proposition hold for the supplied input?")
    p=_normalized_soft(r)
    prob=(p.get('1',0.0) if not invert else p.get('0',0.0)) if p else None
    return _binary(strategy,[r],rng,seed,copy.deepcopy(r.state),instruction,truth,
                   kind='evidence' if r.source=='fever_evidence' else 'generic',probability=prob,
                   detail={'inverted':invert})


def _binary_other(strategy:str,r:Record,rng:random.Random,seed:int,*,case:str='answer_missing',use_unknown:bool=False) -> dict[str,Any]:
    yes,no=binary(rng)
    ordinary={yes:'The original question is true / yes.', no:'The original question is false / no.'}
    gold=yes if _gold_boolean(r) else no
    if case=='answer_missing':
        ordinary.pop(gold)
    c,other,unknown=_extra_choices(rng,ordinary,add_other=True,add_unknown=use_unknown or case=='unanswerable')
    label=gold if case=='answer_present' else other if case=='answer_missing' else unknown
    state=_missing_state() if case=='unanswerable' else copy.deepcopy(r.state)
    return _make(strategy,[r],rng,seed,state,
                 'Answer the original yes/no proposition using only the supplied input. If the correct response is not offered use the unlisted-answer choice; if the input is unavailable use insufficient information.',
                 c,label,canonical={'answer_present':str(r.label),'answer_missing':'OTHER','unanswerable':'UNKNOWN'}[case],
                 detail={'joint_case':case,'other_surface':other,'unknown_surface':unknown,'positive_surface':yes,'negative_surface':no})


def _score_levels(r:Record) -> list[str]:
    crit=r.criteria()
    if not crit:raise ValueError('No ordered score levels')
    return [_description(x) for x in crit.values()]


def _score_exact(strategy:str,r:Record,rng:random.Random,seed:int, *, k:int|None=None,positive:bool|None=None) -> dict[str,Any]:
    levels=_score_levels(r); g=int(r.label)
    if positive is None:positive=bool(rng.randrange(2))
    if k is None: k=g if positive else rng.choice([i for i in range(len(levels)) if i!=g])
    return _binary(strategy,[r],rng,seed,_state_with(r,proposed_level=k,level_description=levels[k]),
                   f'Is the rating exactly level {k}?',g==k,kind='verify',probability=_soft_sum(r,[str(k)]),detail={'exact_level':k})


def _score_threshold(strategy:str,r:Record,rng:random.Random,seed:int, *, t:int|None=None,direction:str|None=None) -> dict[str,Any]:
    levels=_score_levels(r); g=int(r.label); direction=direction or rng.choice(['at_least','at_most'])
    t=(rng.randint(1,len(levels)-1) if direction=='at_least' else rng.randint(0,len(levels)-2)) if t is None else int(t)
    keys=[str(i) for i in range(len(levels)) if (i>=t if direction=='at_least' else i<=t)]
    truth=g>=t if direction=='at_least' else g<=t
    return _binary(strategy,[r],rng,seed,copy.deepcopy(r.state),
                   f'Is the ordered rating {"at least" if direction=="at_least" else "at most"} {t}?',truth,
                   kind='threshold',probability=_soft_sum(r,keys),detail={'direction':direction,'threshold':t})


def _score_interval(strategy:str,r:Record,rng:random.Random,seed:int,*,a:int|None=None,b:int|None=None) -> dict[str,Any]:
    n=len(_score_levels(r));g=int(r.label)
    if a is None or b is None:a,b=sorted(rng.sample(range(n),2))
    a=int(a);b=int(b)
    return _binary(strategy,[r],rng,seed,copy.deepcopy(r.state),
                   f'Is the ordered score between {a} and {b}, inclusive?',a<=g<=b,
                   kind='membership',probability=_soft_sum(r,[str(i) for i in range(a,b+1)]),detail={'interval':[a,b]})


def _score_buckets(strategy:str,r:Record,rng:random.Random,seed:int, *, opaque:bool=False) -> dict[str,Any]:
    levels=_score_levels(r);n=len(levels)
    count=rng.randint(2,min(4,n))
    cuts=sorted(rng.sample(range(1,n),count-1))
    bounds=[0]+cuts+[n]
    names=codes(rng,count) if opaque else rng.sample(GROUP_NAMES,count)
    bucket={name:list(range(bounds[i],bounds[i+1])) for i,name in enumerate(names)}
    c={k:'; '.join(f'{idx}: {levels[idx]}' for idx in indices) for k,indices in bucket.items()}
    gold=int(r.label);target=next(k for k,inds in bucket.items() if gold in inds)
    p=_normalized_soft(r)
    soft={k:sum(p.get(str(i),0) for i in inds) for k,inds in bucket.items()} if p else None
    return _make(strategy,[r],rng,seed,copy.deepcopy(r.state),'Choose the temporary rating bucket containing the correct ordered rating.',
                 c,target,soft=soft,canonical=r.label,detail={'score_buckets':bucket,'opaque_codebook':opaque})


def _score_compare(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int, *, task:str) -> dict[str,Any]:
    r2=pool.choose(r,rng)
    a=int(r.label);b=int(r2.label)
    state={'first':copy.deepcopy(r.state),'second':copy.deepcopy(r2.state)}
    if task=='difference':
        threshold=rng.randint(1,len(_score_levels(r))-1)
        return _binary(strategy,[r,r2],rng,seed,state,
                       f'Is the absolute difference between the two ratings at least {threshold}?',abs(a-b)>=threshold,
                       kind='threshold',detail={'difference_threshold':threshold,'absolute_difference':abs(a-b)})
    if task in {'same','band'}:
        border=len(_score_levels(r))//2
        match=(a==b if task=='same' else (a>=border)==(b>=border))
        return _binary(strategy,[r,r2],rng,seed,state,
                       'Do the two inputs have the same '+ ('exact' if task=='same' else 'coarsened')+' score?',match,
                       kind='same',detail={'mode':task,'band_boundary':border})
    first,second,equal='first','second','equal'
    target=first if a>b else second if a<b else equal
    c={first:'First item has higher score.',second:'Second item has higher score.',equal:'Same ordered score.'}
    return _make(strategy,[r,r2],rng,seed,state,'Which input has a higher ordered score?',c,target,
                 canonical=target,detail={'score_first':a,'score_second':b})


def _score_other(strategy:str,r:Record,rng:random.Random,seed:int,*,case:str='answer_missing',use_unknown:bool=False) -> dict[str,Any]:
    return _other_choice(strategy,r,rng,seed,case=case,use_unknown=use_unknown,score=True)


def _score_reference(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int) -> dict[str,Any]:
    r2=pool.choose(r,rng);r3=pool.choose(r,rng,exclude={r2.id})
    a=abs(int(r.label)-int(r2.label));b=abs(int(r.label)-int(r3.label))
    target='first' if a<b else 'second' if b<a else 'equal'
    return _make(strategy,[r,r2,r3],rng,seed,{'reference':r.state,'first':r2.state,'second':r3.state},
                 'Which sentence-pair similarity score is closer to the reference score?',
                 {'first':'First candidate is closer.','second':'Second candidate is closer.','equal':'Both candidate scores are equally close.'},target,
                 detail={'score_distance_first':a,'score_distance_second':b})


def _agreement(r:Record) -> float:
    p=_normalized_soft(r)
    if not p:
        raise ValueError('This strategy requires source-provided human label probabilities')
    return max(p.values())


def _agreement_threshold(strategy:str,r:Record,rng:random.Random,seed:int,*,t:float|None=None,share:bool=False) -> dict[str,Any]:
    if share and r.source=='civil_comments':
        p=_normalized_soft(r)
        if p is None: raise ValueError('No toxicity share')
        value=p.get('1',0)
    else:
        value=_agreement(r)
    t=float(t if t is not None else rng.choice([0.25,0.4,0.5,0.6,0.7,0.8]))
    return _binary(strategy,[r],rng,seed,copy.deepcopy(r.state),
                   f'Is the observed {"positive/toxic vote share" if share else "leading human vote share"} at least {t:.0%}?',
                   value>=t,kind='threshold',detail={'threshold':t,'observed_share':value})


def _agreement_compare(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int, *, share:bool=False) -> dict[str,Any]:
    r2=pool.choose(r,rng,has_soft=True)
    def strength(x):
        p=_normalized_soft(x)
        if not p:raise ValueError('No human probabilities')
        return p.get('1',0) if share else max(p.values())
    a=strength(r);b=strength(r2);epsilon=0.03
    target='equal' if abs(a-b)<=epsilon else 'first' if a>b else 'second'
    return _make(strategy,[r,r2],rng,seed,{'first':r.state,'second':r2.state},
                 'Which item has the larger human '+('positive/toxic vote share?' if share else 'agreement on its leading class?'),
                 {'first':'First item.','second':'Second item.','equal':f'Tied within {epsilon:.2f} share.'},target,
                 detail={'first_share':a,'second_share':b,'tie_epsilon':epsilon})


def _mcq_truth_pair(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int, *, same_status:bool=False) -> dict[str,Any]:
    r2=pool.choose(r,rng)
    c1=r.criteria();c2=r2.criteria()
    t1=bool(rng.randrange(2));t2=t1 if same_status and bool(rng.randrange(2)) else (not t1 if same_status else bool(rng.randrange(2)))
    k1=r.label if t1 else rng.choice([k for k in c1 if k!=r.label])
    k2=r2.label if t2 else rng.choice([k for k in c2 if k!=r2.label])
    state={'first_question':r.state,'first_proposed_answer':c1[k1],'second_question':r2.state,'second_proposed_answer':c2[k2]}
    if same_status:
        return _binary(strategy,[r,r2],rng,seed,state,'Are the proposed answers equally correct (either both correct or both incorrect)?',t1==t2,kind='same')
    form=truth_pattern(rng);kind=_pattern(t1,t2)
    return _make(strategy,[r,r2],rng,seed,state,'Which of these proposed answers are correct?',
                 {v:k.replace('_',' ') for k,v in form.items()},form[kind],canonical=kind)


def _nli_composition(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int) -> dict[str,Any]:
    other=pool.choose(r,rng)
    relation=rng.choice(list(r.criteria()))
    operator=rng.choice(['and','or','xor'])
    a=r.label==relation;b=other.label==relation
    truth={'and':a and b,'or':a or b,'xor':a!=b}[operator]
    prompt={'and':'both pairs','or':'at least one pair','xor':'exactly one pair'}[operator]
    return _binary(strategy,[r,other],rng,seed,{'first_pair':r.state,'second_pair':other.state,'relation':relation},
                   f'Is it true that {prompt} has the relation {relation}?',truth,
                   kind='generic',detail={'relation':relation,'logical_operator':operator})


def _extra_distractor(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int,*,source_key:str) -> dict[str,Any]:
    other=pool.choose(r,rng)
    original=copy.deepcopy(r.state)
    extra=copy.deepcopy(other.state)
    state={'original':original,'unrelated_material':extra}
    instruction='Decide the original item using its original evidence. The separately marked unrelated material is a distractor and does not replace the original evidence.'
    return _binary(strategy,[r,other],rng,seed,state,instruction,_gold_boolean(r),probability=_soft_sum(r,['1']))


def _partial_ablation(strategy:str,r:Record,rng:random.Random,seed:int, validator:Callable | None) -> dict[str,Any]:
    if validator is None:
        raise ValidationRequiredError('Cannot infer UNKNOWN from a partial ablation without a trusted validator')
    state=copy.deepcopy(r.state)
    if isinstance(state,dict):
        candidates=[k for k,v in state.items() if isinstance(v,(str,list)) and len(v)>3]
        if not candidates: raise ValueError('No partially ablatable field')
        k=rng.choice(candidates)
        value=state[k]
        if isinstance(value,list):
            state[k]=[x for i,x in enumerate(value) if i!=rng.randrange(len(value))]
        else:
            pieces=value.split(' ')
            start=rng.randrange(len(pieces))
            state[k]=' '.join(pieces[:start]+['[MISSING SPAN]']+pieces[min(len(pieces),start+max(1,len(pieces)//4)):])
    else:
        raise ValueError('Partial ablation requires a structured state')
    orig_crit=r.criteria()
    if r.primitive=='noul':
        pos,neg=binary(rng);orig_crit={pos:'Yes',neg:'No'}
    c,_,unknown=_extra_choices(rng,orig_crit,add_other=False,add_unknown=True)
    out=_make(strategy,[r],rng,seed,state,'Choose insufficient information if the remaining context is not sufficient to decide.',c,unknown,canonical='UNKNOWN',detail={'validated_partial_ablation':True})
    if not bool(validator(out,r.to_row())):
        raise ValueError('Validator rejected the claim that the ablated item is unanswerable')
    return out


def _binary_reference(strategy:str,r:Record,pool:PoolIndex,rng:random.Random,seed:int) -> dict[str,Any]:
    return _ref_match(strategy,r,pool,rng,seed,n=2)


def _binary_opaque(strategy:str,r:Record,rng:random.Random,seed:int) -> dict[str,Any]:
    labels=codes(rng,2); mapping={'0':labels[0],'1':labels[1]}
    p=_normalized_soft(r)
    soft={mapping[k]:v for k,v in p.items() if k in mapping} if p else None
    return _make(strategy,[r],rng,seed,copy.deepcopy(r.state),
                 'Use the temporary binary codebook: choose the code matching the original yes/no proposition.',
                 {mapping['0']:'The original proposition is false / no.',mapping['1']:'The original proposition is true / yes.'},
                 mapping['1' if _gold_boolean(r) else '0'],soft=soft,canonical=r.label,detail={'codebook':mapping})


def _other_joint(strategy:str,r:Record,rng:random.Random,seed:int,*,case:str|None=None) -> dict[str,Any]:
    case=case or rng.choice(['answer_present','answer_missing','unanswerable'])
    if r.primitive=='noul':
        return _binary_other(strategy,r,rng,seed,case=case,use_unknown=True)
    return _other_choice(strategy,r,rng,seed,case=case,use_unknown=True)


def apply_strategy(strategy_id:str,row:Mapping[str,Any]|Record,*,pool:PoolIndex|Sequence[Mapping[str,Any]]|None=None,
                   seed:int=0,validator:Callable|None=None,**kw:Any) -> dict[str,Any]:
    """Run one of the 295 explicitly exported reformulation strategies.

    `pool` must be restricted to the anchor's original source and split; the
    worker builds/uses an index once for all tasks, so pairwise sampling is fast.
    """
    r=Record.from_row(row)
    if not strategy_id.startswith(r.source+'_'):
        raise ValueError(f'Strategy {strategy_id!r} is not for source {r.source!r}')
    suffix=strategy_id[len(r.source)+1:]
    rng=random.Random(hashed_seed(strategy_id,r.id,seed))

    if suffix.endswith('_ablation_optional') or suffix in {'partial_stem_ablation_optional','partial_passage_ablation_optional'}:
        return _partial_ablation(strategy_id,r,rng,seed,validator)

    if r.source in MCQ_SOURCES:
        if suffix=='direct_mcq_rephrase':
            return _mcq_rephrase(strategy_id,r,rng,seed)
        if suffix=='opaque_answer_keys':
            return _coded_choice(strategy_id,r,rng,seed,include_all=True)
        if suffix=='candidate_correctness':
            return _verify(strategy_id,r,rng,seed,mcq=True,positive=kw.get('positive'))
        if suffix=='answer_subset_membership':
            return _subset(strategy_id,r,rng,seed,mcq=True,include_gold=kw.get('include_gold'))
        if suffix=='answer_partition':return _partition(strategy_id,r,rng,seed,mcq=True)
        if suffix=='other_gold_removed':return _other_choice(strategy_id,r,rng,seed,case='answer_missing')
        if suffix=='other_gold_present':return _other_choice(strategy_id,r,rng,seed,case='answer_present')
        if suffix=='unknown_missing_stem':return _unknown_choice(strategy_id,r,rng,seed)
        if suffix=='other_unknown_joint':return _other_joint(strategy_id,r,rng,seed,case=kw.get('case'))
        if suffix=='two_question_candidate_pattern':return _mcq_truth_pair(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='same_candidate_truth_status':return _mcq_truth_pair(strategy_id,r,pool_index(pool),rng,seed,same_status=True)

    if r.source in CHOICE_SOURCES:
        if suffix=='direct_rephrase':return _direct(strategy_id,r,rng,seed,kind='choice')
        if suffix=='candidate_verify':return _verify(strategy_id,r,rng,seed,positive=kw.get('positive'))
        if suffix=='subset_membership':return _subset(strategy_id,r,rng,seed,include_gold=kw.get('include_gold'))
        if suffix=='partition_group':return _partition(strategy_id,r,rng,seed)
        if suffix=='other_gold_removed':return _other_choice(strategy_id,r,rng,seed,case='answer_missing')
        if suffix=='other_gold_present':return _other_choice(strategy_id,r,rng,seed,case='answer_present')
        if suffix=='unknown_missing_input':return _unknown_choice(strategy_id,r,rng,seed)
        if suffix=='other_unknown_joint':return _other_joint(strategy_id,r,rng,seed,case=kw.get('case'))
        if suffix=='opaque_codebook':return _coded_choice(strategy_id,r,rng,seed)
        if suffix=='same_class_pair':return _same_class(strategy_id,r,pool_index(pool),rng,seed,desired=kw.get('same'))
        if suffix=='reference_match':return _ref_match(strategy_id,r,pool_index(pool),rng,seed,n=3)
        if suffix=='odd_one_out':return _odd_one(strategy_id,r,pool_index(pool),rng,seed)
        if suffix in {'two_message_truth_pattern','two_query_truth_pattern','two_utterance_truth_pattern','two_clause_truth_pattern'}:
            return _truth_pattern(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='in_scope_vs_oos':
            return _binary(strategy_id,[r],rng,seed,r.state,'Does this query belong to any named in-scope intent?',r.label!='oos',kind='membership')
        if suffix=='oos_candidate_verify':
            return _binary(strategy_id,[r],rng,seed,r.state,'Is this query out of scope for the known intent inventory?',r.label=='oos',kind='verify')
        if suffix=='agreement_threshold':return _agreement_threshold(strategy_id,r,rng,seed)
        if suffix=='compare_agreement':return _agreement_compare(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='same_primary_pair_soft':return _same_class(strategy_id,r,pool_index(pool),rng,seed,soft=True)

    if r.source in NLI_SOURCES:
        if suffix=='direct_relation':return _direct(strategy_id,r,rng,seed,kind='nli')
        if suffix=='candidate_relation_verify':return _verify(strategy_id,r,rng,seed,positive=kw.get('positive'))
        if suffix=='relation_subset_membership':return _subset(strategy_id,r,rng,seed,include_gold=kw.get('include_gold'))
        if suffix=='same_relation_pair':return _same_class(strategy_id,r,pool_index(pool),rng,seed,desired=kw.get('same'),soft=r.source=='chaosnli')
        if suffix=='specific_relation_truth_pattern':return _truth_pattern(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='reference_relation_match':return _ref_match(strategy_id,r,pool_index(pool),rng,seed,n=3)
        if suffix=='relation_odd_one_out':return _odd_one(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='other_gold_removed':return _other_choice(strategy_id,r,rng,seed,case='answer_missing')
        if suffix=='other_gold_present':return _other_choice(strategy_id,r,rng,seed,case='answer_present')
        if suffix=='unknown_missing_pair_member':return _unknown_choice(strategy_id,r,rng,seed)
        if suffix=='other_unknown_joint':return _other_joint(strategy_id,r,rng,seed,case=kw.get('case'))
        if suffix=='opaque_relation_codebook':return _coded_choice(strategy_id,r,rng,seed,include_all=True)
        if suffix=='relation_boolean_composition':return _nli_composition(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='majority_strength_threshold':return _agreement_threshold(strategy_id,r,rng,seed)
        if suffix=='compare_annotator_agreement':return _agreement_compare(strategy_id,r,pool_index(pool),rng,seed)

    if r.source in SCORE_SOURCES:
        if suffix=='direct_score_rephrase':return _direct(strategy_id,r,rng,seed,kind='score')
        if suffix=='exact_score_verify':return _score_exact(strategy_id,r,rng,seed,positive=kw.get('positive'))
        if suffix=='threshold':return _score_threshold(strategy_id,r,rng,seed)
        if suffix=='interval_membership':return _score_interval(strategy_id,r,rng,seed)
        if suffix=='coarse_bucket':return _score_buckets(strategy_id,r,rng,seed)
        if suffix=='pairwise_compare':return _score_compare(strategy_id,r,pool_index(pool),rng,seed,task='compare')
        if suffix=='same_score_or_band':return _score_compare(strategy_id,r,pool_index(pool),rng,seed,task=rng.choice(['same','band']))
        if suffix=='score_difference_threshold':return _score_compare(strategy_id,r,pool_index(pool),rng,seed,task='difference')
        if suffix=='other_score_subset':return _score_other(strategy_id,r,rng,seed,case=rng.choice(['answer_present','answer_missing']))
        if suffix=='unknown_missing_required_input':return _unknown_choice(strategy_id,r,rng,seed)
        if suffix=='other_unknown_joint':return _other_joint(strategy_id,r,rng,seed,case=kw.get('case'))
        if suffix=='opaque_bucket_codebook':return _score_buckets(strategy_id,r,rng,seed,opaque=True)
        if suffix=='reference_similarity_order':return _score_reference(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='agreement_threshold':return _agreement_threshold(strategy_id,r,rng,seed)
        if suffix=='compare_agreement':return _agreement_compare(strategy_id,r,pool_index(pool),rng,seed)

    if r.source in BINARY_SOURCES:
        if suffix=='direct_binary_surface':return _bool_direct(strategy_id,r,rng,seed)
        if suffix=='inverted_binary_question':return _bool_direct(strategy_id,r,rng,seed,invert=True)
        if suffix=='same_answer_pair':return _bool_pair(strategy_id,r,pool_index(pool),rng,seed,op='same')
        if suffix in {'pair_truth_pattern','two_question_meta'}:return _truth_pattern(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='logical_and':return _bool_pair(strategy_id,r,pool_index(pool),rng,seed,op='and')
        if suffix=='logical_or':return _bool_pair(strategy_id,r,pool_index(pool),rng,seed,op='or')
        if suffix=='logical_xor':return _bool_pair(strategy_id,r,pool_index(pool),rng,seed,op='xor')
        if suffix=='other_target_removed':return _binary_other(strategy_id,r,rng,seed,case='answer_missing')
        if suffix=='other_target_present':return _binary_other(strategy_id,r,rng,seed,case='answer_present')
        if suffix=='unknown_missing_required_input':return _unknown_choice(strategy_id,r,rng,seed)
        if suffix=='other_unknown_joint':return _other_joint(strategy_id,r,rng,seed,case=kw.get('case'))
        if suffix=='opaque_binary_codebook':return _binary_opaque(strategy_id,r,rng,seed)
        if suffix=='reference_match' or suffix=='relation_reference_match':return _binary_reference(strategy_id,r,pool_index(pool),rng,seed)
        if suffix=='irrelevant_passage_distractor':return _extra_distractor(strategy_id,r,pool_index(pool),rng,seed,source_key='passage')
        if suffix=='refutation_question':return _bool_direct(strategy_id,r,rng,seed,invert=True,statement='Does the original evidence refute the claim?')
        if suffix=='add_unrelated_evidence':return _extra_distractor(strategy_id,r,pool_index(pool),rng,seed,source_key='evidence')
        if suffix=='evidence_ablation':return _unknown_choice(strategy_id,r,rng,seed)
        if suffix=='difference_question':return _bool_direct(strategy_id,r,rng,seed,invert=True,statement='Do these two sentences differ in meaning?')
        if suffix=='toxicity_share_threshold':return _agreement_threshold(strategy_id,r,rng,seed,share=True)
        if suffix=='compare_toxicity_share':return _agreement_compare(strategy_id,r,pool_index(pool),rng,seed,share=True)
        if suffix=='remove_all_facts':return _unknown_choice(strategy_id,r,rng,seed)
        if suffix=='add_distractor_facts':return _extra_distractor(strategy_id,r,pool_index(pool),rng,seed,source_key='facts')

    raise NotImplementedError(f'No transform for source={r.source} strategy={strategy_id}')
