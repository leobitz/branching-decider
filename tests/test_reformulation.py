"""Reformulation-only tests. No models or HF download required."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import pytest
from jevbench.data.constants import CONFIG_DOMAINS
from jevbench.data.processing import process_jev_row
from jevbench.reformulation import get_strategy, get_specs, list_strategies, Record
from jevbench.reformulation.engine import PoolIndex, ValidationRequiredError
from reformulate_dataset import allocate_targets, build, generate_parallel
from reformulation_synthetic import make_pool


def _valid_partial(transformed,original):
    return True


def test_all_295_strategies_can_generate_valid_processed_rows():
    failures=[]
    all_specs={x for source in CONFIG_DOMAINS for x in list_strategies(source)}
    assert len(all_specs)==295
    for source in CONFIG_DOMAINS:
        rows=make_pool(source)
        index=PoolIndex(rows)
        for sid in list_strategies(source):
            try:
                out=get_strategy(sid)(rows[0],pool=index,seed=42,validator=_valid_partial)
                processed=process_jev_row(out,source,'train',42)
                assert processed['source_config']==source
                assert out['source']==source
                assert processed['labels'][processed['target_index']]
            except Exception as exc:
                failures.append((sid,type(exc).__name__,str(exc)))
    assert not failures, '\n'.join(map(str,failures))


def test_unsafe_partial_ablation_requires_validator():
    rows=make_pool('mmlu')
    with pytest.raises(ValidationRequiredError):
        get_strategy('mmlu_partial_stem_ablation_optional')(rows[0],pool=rows,seed=1)


def test_other_vs_unknown_vs_answer_present():
    rows=make_pool('mnli')
    fn=get_strategy('mnli_other_unknown_joint')
    for case,expected in [('answer_present','entailment'),('answer_missing','OTHER'),('unanswerable','UNKNOWN')]:
        out=Record.from_row(fn(rows[0],pool=rows,seed=7,case=case))
        assert out.meta['canonical_target']==expected
        p=process_jev_row(out.to_row(),'mnli','train',42)
        assert p['labels'][p['target_index']]
        if case=='unanswerable':
            assert 'NOT PROVIDED' in json.dumps(out.state)


def test_quota_total_and_per_source():
    c={('mmlu','train'):80,('mmlu','test'):20,('mnli','train'):40,('mnli','test'):10}
    a=allocate_targets(c,num_generated=75)
    assert sum(a.values())==75
    assert a[('mmlu','train')]>a[('mmlu','test')]
    b=allocate_targets(c,per_source=14)
    assert sum(x for (s,_),x in b.items() if s=='mmlu')==14
    assert sum(x for (s,_),x in b.items() if s=='mnli')==14
    assert allocate_targets(c,ratio=1.0)==c


def test_parallel_exact_quota_and_original_merging(tmp_path):
    pools={source:{split:make_pool(source,8,split) for split in ('train','validation','test')} for source in ('mmlu','mnli')}
    def mock_loader(dataset_name,config_name,revision,cache_dir):
        return pools[config_name]
    path=build(output_dir=tmp_path/'dataset',configs=['mmlu','mnli'],splits=('train','validation','test'),
               num_generated=24,workers=2,seed=19,loader=mock_loader,strict=True,save_raw=True)
    mf=json.loads((path/'manifest.json').read_text())
    assert sum(sum(x.values()) for x in mf['generated_split_counts'].values())==24
    originals=generated=0
    for split in ('train','validation','test'):
        records=[json.loads(line) for line in (path/f'{split}.jsonl').read_text().splitlines()]
        originals+=sum(r['original'] is True for r in records)
        generated+=sum(r['original'] is False for r in records)
        assert all(isinstance(r['generation'],dict) for r in records if not r['original'])
        assert all(r['generation'] is None for r in records if r['original'])
        for rec in records:
            assert isinstance(rec['labels'],list)
            assert 0<=rec['target_index']<len(rec['labels'])
            assert rec['source_split']==split
    assert originals==48
    assert generated==24
    assert mf['processing_version']
    assert (path/'raw'/'mnli'/'train.jsonl').exists()


def test_existing_data_loader_reads_merged_cache_without_changes(tmp_path):
    from types import SimpleNamespace
    from jevbench.data.constants import DATASET_NAME_DEFAULT
    from jevbench.data.dataset import prepare_jev_bench_cache, _read_jsonl, JevBenchDecisionDataset

    def mock_loader(dataset_name,config_name,revision,cache_dir):
        return {s:make_pool(config_name,6,s) for s in ('train','validation','test')}
    out=build(output_dir=tmp_path/'merged',configs=['mmlu'],num_generated=5,
              workers=1,loader=mock_loader)
    args=SimpleNamespace(
        processed_data_dir=out, configs='mmlu', hf_dataset=DATASET_NAME_DEFAULT,
        dataset_revision='main', surface_seed=42, force_reprocess=False,
    )
    assert prepare_jev_bench_cache(args)==out
    records=_read_jsonl(out/'train.jsonl')
    assert len(records)>=6
    ds=JevBenchDecisionDataset(records,'train',max_examples=None,seed=42)
    assert len(ds)==len(records)
    assert all('target' in item for item in ds)


def test_parallel_runs_are_worker_count_invariant(tmp_path):
    def mock_loader(dataset_name, config_name, revision, cache_dir):
        return {s:make_pool(config_name,12,s) for s in ('train','validation','test')}
    results=[]
    for workers in (1,2):
        path=build(output_dir=tmp_path/f'merged_{workers}',configs=['banking77'],num_generated=18,
                   workers=workers,seed=123,loader=mock_loader)
        by_split={}
        for s in ('train','validation','test'):
            by_split[s]=[
                (r['context'],r['query'],r['labels'],r['target_index'],r['generation'])
                for r in (json.loads(x) for x in (path/f'{s}.jsonl').read_text().splitlines())
                if not r['original']
            ]
        results.append(by_split)
    assert results[0]==results[1]
