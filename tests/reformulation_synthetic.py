"""Small JEV-Bench-shaped fixtures covering all 22 source configs."""
from __future__ import annotations
from jevbench.reformulation.record import Record

CHOICE_LABELS={
 'banking77':['card_arrival','cash_withdrawal','transfer_fee'],
 'clinc150':['set_alarm','weather','oos'],
 'massive':['play_music','weather_query','timer_set'],
 'ledgar':['termination','governing_law','confidentiality'],
 'go_emotions':['joy','sadness','neutral'],
}
CHOICE_EXTRA=['refund','payment','fear','surprise','notice','support']
BINARY_SOURCES={'boolq','fever_evidence','paws','civil_comments','sms_spam','strategyqa_closed','strategyqa_grounded'}
SCORE_SOURCES={'sst5','yelp5','helpsteer2_helpfulness','helpsteer2_verbosity','stsb','measuring_hate_speech'}


def make_pool(source,n=12,split='train'):
    out=[]
    for i in range(n):
        soft=None
        if source in CHOICE_LABELS:
            labels=CHOICE_LABELS[source]
            gold=labels[i%len(labels)]
            criteria={x:x.replace('_',' ') for x in labels+CHOICE_EXTRA}
            state={'text':f'{source} message {i} with semantic content'}
            primitive='choice'
            q={'type':'choice','instructions':'Choose the best semantic category for the input.','criteria':criteria}
            if source=='go_emotions':
                soft={k:(.8 if k==gold else .1/(len(criteria)-1)) for k in criteria}
        elif source in {'mmlu','arc_challenge'}:
            criteria={'A':'the sun','B':'the moon','C':'the earth','D':'mars'}
            gold=list(criteria)[i%len(criteria)]
            state={'question':f'Which option answers science question number {i}?','topic':'science'}
            primitive='choice'
            q={'type':'choice','instructions':'Choose the correct option.','criteria':criteria}
        elif source in {'mnli','chaosnli'}:
            criteria={'entailment':'follows from the premise','neutral':'not determined by the premise','contradiction':'denied by the premise'}
            gold=list(criteria)[i%len(criteria)]
            state={'premise':f'Someone {i} works in a store.','hypothesis':f'The person {i} works for money.'}
            primitive='choice'
            q={'type':'choice','instructions':'Determine the premise hypothesis relation.','criteria':criteria}
            if source=='chaosnli':
                soft={k:(.8 if k==gold else .1) for k in criteria}
        elif source in SCORE_SOURCES:
            length=6 if source=='stsb' else 3 if source=='measuring_hate_speech' else 5
            criteria=[f'level {j}' for j in range(length)]
            gold=str(i%length)
            if source.startswith('helpsteer2'):
                state={'prompt':f'Prompt {i} requiring short answer','response':f'Answer {i}'}
            elif source=='stsb':
                state={'sentence1':f'Sentence one {i}.','sentence2':f'Sentence two {i}.'}
            else: state={'text':f'Rate this {source} example {i}'}
            primitive='score'
            q={'type':'score','instructions':'Assign an ordered rating.','criteria':criteria}
            if source=='measuring_hate_speech':
                soft={str(k):(.8 if k==int(gold) else .1) for k in range(length)}
        elif source in BINARY_SOURCES:
            primitive='noul'
            gold=str(i%2)
            if source=='boolq':state={'question':f'Is statement {i} true?','passage':f'Passage A {i}. Passage B {i}.'}
            elif source=='fever_evidence':state={'claim':f'A claim {i}.','evidence':[f'First evidence {i}',f'Second evidence {i}']}
            elif source=='paws':state={'sentence1':f'The dog chased cat {i}.','sentence2':f'The cat chased dog {i}.'}
            elif source=='strategyqa_grounded':state={'question':f'Question {i}?','facts':[f'Fact A {i}',f'Fact B {i}']}
            elif source=='strategyqa_closed':state={'question':f'Can animals {i} read?'}
            else:state={'text':f'This is message number {i}, used as a text sample.'}
            q={'type':'noul','instructions':'Is this original statement true?'}
            if source=='civil_comments':soft=.8 if int(gold) else .2
        else:raise KeyError(source)
        out.append(Record(id=f'{source}:{split}:{i}',source=source,split=split,primitive=primitive,
                          state=state,question=q,label=gold,soft_label=soft,meta={}).to_row())
    return out
