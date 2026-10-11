#!/usr/bin/env python3
"""Download JEV-Bench, generate exactly N reformulations in parallel, merge, and save.

Unmodified model/training code can load these files by pointing its existing
--processed-data-dir at the output directory. Normal training and the default
JEV-Bench downloader are NOT modified by this script.

Examples:
    python reformulate_dataset.py --num-generated 10000 --workers 8
    python reformulate_dataset.py --per-source 1000 --workers 4
    python reformulate_dataset.py --ratio 1.0 --workers 8 --output-dir processed_data_jev_hierarchical-1.7B-reform
    python reformulate_dataset.py --configs mnli,mmlu --splits train --num-generated 1000
    python reformulate_dataset.py --num-generated 10000 --workers 8 --output-dir processed_data_jev_hierarchical-1.7B-reform
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from jevbench.data.collators import hierarchical_token_length
from jevbench.data.constants import DATASET_NAME_DEFAULT, PROCESSING_VERSION, PROCESSED_SPLITS, NOUL_SURFACE_PAIRS, SCORE_SURFACE_STYLES
from jevbench.data.processing import process_jev_row
from jevbench.reformulation import get_specs, get_strategy, list_strategies
from jevbench.reformulation.engine import PoolIndex, hashed_seed
from jevbench.reformulation.record import Record, decode

# Read-only globals initialized once per worker for cheap indexed multi-row lookup.
_CTX: dict[str, Any] = {}


@functools.lru_cache(maxsize=2)
def _load_tokenizer(name:str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _write(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def _rows(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _safe_name(name: str) -> str:
    return re.sub(r'[^-a-zA-Z0-9_.]', '_', name)[:150]


def _normalize_row(row: dict[str,Any], source: str, split: str, i: int) -> dict[str, Any]:
    result=dict(row)
    result['source']=source
    result['split']=split
    result['id']=str(result.get('id') or f'{source}:{split}:{i}')
    return result


def _quota(weights: dict[tuple[str,str],int], target: int) -> dict[tuple[str,str],int]:
    """Largest-remainder allocation; sum of allocated counts is exactly target."""
    total=sum(max(0,v) for v in weights.values())
    if target<0:raise ValueError('Requested generated count must be nonnegative')
    if target and not total:raise ValueError('No valid seed rows in the requested sources/splits')
    if not total:return {k:0 for k in weights}
    exact={k:target*max(0,v)/total for k,v in weights.items()}
    out={k:math.floor(x) for k,x in exact.items()}
    missing=target-sum(out.values())
    for key in sorted(weights,key=lambda k:(-(exact[k]-out[k]), k))[:missing]:out[key]+=1
    return out


def allocate_targets(group_counts: dict[tuple[str,str],int],*,ratio:float=1.0,num_generated:int|None=None,
                     per_source:int|None=None,per_source_split:int|None=None) -> dict[tuple[str,str],int]:
    modes=sum(x is not None for x in (num_generated,per_source,per_source_split))
    if modes>1:raise ValueError('Only one of --num-generated, --per-source, --per-source-split may be specified')
    if ratio<0:raise ValueError('Ratio must be >= 0')
    if num_generated is not None:return _quota(group_counts,num_generated)
    if per_source is not None:
        if per_source<0:raise ValueError('per_source must be nonnegative')
        out={k:0 for k in group_counts}
        for source in sorted({s for s,_ in group_counts}):
            subset={k:v for k,v in group_counts.items() if k[0]==source}
            if sum(subset.values())>0:out.update(_quota(subset,per_source))
        return out
    if per_source_split is not None:
        if per_source_split<0:raise ValueError('per_source_split must be nonnegative')
        return {k:(per_source_split if v>0 else 0) for k,v in group_counts.items()}
    return {k:int(math.floor(v*ratio+0.5)) for k,v in group_counts.items()}


def _init_worker(raw_rows:list[dict[str,Any]], source:str, split:str, strategies:list[str], seed:int, surface_seed:int, validator=None,
                 length_limit:dict[str,Any]|None=None):
    global _CTX
    rng=random.Random(hashed_seed('worker-index',source,split,seed))
    order=list(range(len(raw_rows)))
    rng.shuffle(order)
    strategies=list(strategies)
    rng.shuffle(strategies)
    _CTX={
        'pool':PoolIndex(raw_rows),
        'rows':raw_rows,
        'order':order,
        'source':source,
        'split':split,
        'strategies':strategies,
        'seed':seed,
        'surface_seed':surface_seed,
        'specs':{s['id']:s for s in get_specs(source)},
        'validator':_import_validator(validator) if isinstance(validator,str) else validator,
        'length_limit':length_limit,
        'tokenizer':_load_tokenizer(length_limit['tokenizer_name']) if length_limit else None,
    }


def _generation_meta(raw:dict[str,Any],sid:str,slot:int,seed:int,source:str,split:str) -> dict[str,Any]:
    meta=decode(raw.get('meta')) or {}
    spec=_CTX['specs'].get(sid,{})
    canonical=meta.get('canonical_target')
    return {
        'strategy':sid,
        'description':spec.get('construction'),
        'target_rule':spec.get('target_rule'),
        'row_usage':spec.get('row_usage'),
        'source':source,
        'split':split,
        'parent_ids':meta.get('parent_ids',[]),
        'parent_labels':meta.get('parent_labels',[]),
        'canonical_target':canonical,
        'answerability':False if canonical=='UNKNOWN' else True,
        'gold_offered':False if canonical=='OTHER' else None if canonical=='UNKNOWN' else True,
        'seed':seed,
        'sample_slot':slot,
        'raw_id':raw.get('id'),
        'details':meta.get('details',{}),
    }


def _balanced_kwargs(sid:str,slot:int) -> dict[str,Any]:
    # Ensure OTHER, UNKNOWN and ordinary correct answers all appear as targets.
    if sid.endswith('other_unknown_joint'):
        return {'case':('answer_present','answer_missing','unanswerable')[slot%3]}
    if sid.endswith(('candidate_verify','candidate_correctness','candidate_relation_verify','exact_score_verify')):
        return {'positive':bool(slot%2)}
    if sid.endswith(('subset_membership','answer_subset_membership','relation_subset_membership')):
        return {'include_gold':bool(slot%2)}
    if sid.endswith(('same_relation_pair','same_class_pair')):
        return {'same':bool(slot%2)}
    return {}


def _worker_generate(slot:int) -> tuple[dict[str,Any]|None,dict[str,Any]|None,str|None]:
    ctx=_CTX
    raw_rows=ctx['rows'];strategies=ctx['strategies'];source=ctx['source'];split=ctx['split']
    if not raw_rows or not strategies:return None,None,'No seed rows or strategies'
    anchor=raw_rows[ctx['order'][slot % len(raw_rows)]]
    # Each pass over the anchor pool shifts the chosen strategy; the initial
    # mapping is a balanced permutation rather than always choosing strategy 0.
    # Choose a stride coprime to strategy count so the schedule visits every
    # strategy before repeating, even for 12/14/15/16-strategy sources.
    m=len(strategies)
    stride=next((k for k in range(2,m) if math.gcd(k,m)==1),1)
    first=(slot * stride + slot//len(raw_rows)) % m
    errors=[]
    for attempt in range(min(18,len(strategies))):
        sid=strategies[(first+attempt)%len(strategies)]
        seed=hashed_seed('generate',ctx['seed'],source,split,slot,attempt,sid) % (2**32)
        try:
            fn=get_strategy(sid)
            raw=fn(anchor,pool=ctx['pool'],seed=seed,validator=ctx['validator'],**_balanced_kwargs(sid,slot))
            processed=process_jev_row(raw,source,split,ctx['surface_seed'])
            limit=ctx['length_limit']
            if limit:
                n_tokens=hierarchical_token_length(ctx['tokenizer'],processed['context'],processed['query'],
                                                   processed['labels'],limit['max_label_length'])
                if n_tokens>limit['max_length']:
                    raise ValueError(f'too long ({n_tokens}>{limit["max_length"]} tokens)')
            generation=_generation_meta(raw,sid,slot,seed,source,split)
            processed['original']=False
            processed['generation']=generation
            return processed,raw,None
        except (ValueError,NotImplementedError,KeyError,TypeError,IndexError) as exc:
            errors.append(f'{sid}: {type(exc).__name__}: {exc}')
    return None,None,'; '.join(errors[:3])


def _fingerprint(processed:dict[str,Any]) -> str:
    # Exact content deduplication across independent jobs; omit IDs, metadata.
    blob=json.dumps([processed['context'],processed['query'],processed['labels'],processed['target_index']],ensure_ascii=False,sort_keys=True).encode('utf-8')
    return hashlib.blake2b(blob,digest_size=16).hexdigest()


def _round_batch(executor,lo:int,hi:int,chunksize:int):
    for result in executor.map(_worker_generate,range(lo,hi),chunksize=chunksize):
        yield result


def generate_parallel(*,source:str,split:str,raw_rows:list[dict[str,Any]],target:int,
                      strategies:list[str],workers:int,seed:int,surface_seed:int,
                      chunksize:int=16,deduplicate:bool=True,strict:bool=True,
                      max_attempt_factor:int=8,validator=None,raw_handle=None,output_handle=None,
                      error_handle=None,length_limit:dict[str,Any]|None=None) -> dict[str,Any]:
    """Process pool generation with bounded batches and exact requested quota."""
    if not raw_rows and target:raise RuntimeError(f'{source}/{split} has no valid originals')
    if not strategies and target:raise RuntimeError(f'{source}/{split} has no eligible generators')
    seen:set[str]=set()
    written=0;attempts=0;errors=0;duplicates=0
    usage=Counter()
    max_slots=max(target*max_attempt_factor,target+workers*32)
    batch=max(32,workers*chunksize*4)
    def consume(items):
        nonlocal written,attempts,errors,duplicates
        for processed,raw,error in items:
            attempts+=1
            if error:
                errors+=1
                if error_handle and errors<=100:
                    _write(error_handle,{'source':source,'split':split,'attempt':attempts,'error':error})
                continue
            if processed is None:
                errors+=1
                continue
            fingerprint=_fingerprint(processed)
            if deduplicate and fingerprint in seen:
                duplicates+=1
                continue
            seen.add(fingerprint)
            _write(output_handle,processed)
            if raw_handle is not None:
                raw=dict(raw)
                raw['original']=False
                raw['generation']=processed['generation']
                _write(raw_handle,raw)
            usage[processed['generation']['strategy']]+=1
            written+=1
            if written>=target:break

    if workers==1:
        _init_worker(raw_rows,source,split,strategies,seed,surface_seed,validator,length_limit)
        for lo in range(0,max_slots,batch):
            consume(_worker_generate(i) for i in range(lo,min(max_slots,lo+batch)))
            if written>=target:break
    else:
        # Spawn/forkserver avoids unsafe fork-from-Jupyter multithreaded state.
        context=mp.get_context('forkserver' if 'forkserver' in mp.get_all_start_methods() else 'spawn')
        with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=_init_worker,
                                 initargs=(raw_rows,source,split,strategies,seed,surface_seed,validator,length_limit)) as executor:
            for lo in range(0,max_slots,batch):
                consume(_round_batch(executor,lo,min(max_slots,lo+batch),chunksize))
                if written>=target:break

    if strict and written!=target:
        raise RuntimeError(
            f'{source}/{split}: generated {written:,} unique rows, requested {target:,} '
            f'after {attempts:,} attempts (errors={errors}, duplicates={duplicates}). '
            'Reduce the requested count, disable deduplication, or use --allow-shortfall.'
        )
    return {'requested':target,'generated':written,'attempted':attempts,'errors':errors,
            'duplicates':duplicates,'strategies':dict(usage)}


def _load_hub(dataset_name:str,configs:list[str]|None,revision:str,cache_dir:Path|None):
    try:
        from datasets import get_dataset_config_names,load_dataset
    except ImportError as exc:
        raise ImportError('Install dataset dependencies first: pip install -r requirements.txt') from exc
    if configs is None:
        configs=list(get_dataset_config_names(dataset_name,revision=revision,cache_dir=str(cache_dir) if cache_dir else None))
    return configs,load_dataset


def _import_validator(dotted:str|None):
    if not dotted:return None
    if ':' not in dotted:raise ValueError('Validator must be module_path:function_name')
    module,func=dotted.split(':',1)
    validator=getattr(importlib.import_module(module),func)
    if not callable(validator):raise ValueError('Validator must be callable')
    return validator


def build(*,dataset_name:str=DATASET_NAME_DEFAULT,revision:str='main',cache_dir:Path|None=None,
          output_dir:Path=Path('jev_bench_processed_reformulated'),configs:list[str]|None=None,
          splits:tuple[str,...]=PROCESSED_SPLITS, ratio:float=1.0,num_generated:int|None=None,
          per_source:int|None=None,per_source_split:int|None=None,workers:int=4,seed:int=42,
          surface_seed:int=42,strategies:list[str]|None=None,strict:bool=True,deduplicate:bool=True,
          save_raw:bool=False,validator=None,force:bool=False,chunksize:int=16,
          loader=None,config_discovery=None,length_limit:dict[str,Any]|None=None) -> Path:
    """Build original + generated cache; no model or training changes required.

    Optional loader and config_discovery hooks are for offline integration tests.
    """
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.glob('*.jsonl')) and not force:
        raise FileExistsError(f'Output {output_dir} already contains JSONL files; supply --force to rebuild')
    if workers<1:raise ValueError('workers must be >= 1')
    if chunksize<1:raise ValueError('chunksize must be >= 1')
    bad=set(splits)-set(PROCESSED_SPLITS)
    if bad:raise ValueError(f'Unsupported splits: {bad}')
    if loader is None:
        configs,loader=_load_hub(dataset_name,configs,revision,cache_dir)
    elif configs is None:
        configs=list(config_discovery(dataset_name)) if config_discovery else list_strategies() # pragma: no cover
    configs=list(configs)
    print(f'Building merged JEV-Bench dataset: {dataset_name} configs={len(configs)} workers={workers}')

    output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.jev_reform_work_',dir=output_dir) as temporary:
        stage=Path(temporary)
        counts={}
        original_summary={s:Counter() for s in PROCESSED_SPLITS}
        original_primitive={s:Counter() for s in PROCESSED_SPLITS}
        errors=Counter()
        raw_paths={};stage_originals={}
        # Stage each config/split to bounded disk storage. This is necessary to
        # allocate --num-generated globally without holding all 22 configs in RAM.
        for ci,source in enumerate(configs,1):
            print(f'[{ci}/{len(configs)}] Download/process {source}',flush=True)
            ds=loader(dataset_name,source,revision=revision,cache_dir=str(cache_dir) if cache_dir else None)
            for split in PROCESSED_SPLITS:
                key=(source,split)
                counts[key]=0
                if split not in ds:continue
                stem=stage/(_safe_name(source)+'_'+split)
                raw_path=stem.with_suffix('.raw.jsonl')
                original_path=stem.with_suffix('.original.jsonl')
                raw_paths[key]=raw_path
                stage_originals[key]=original_path
                with raw_path.open('w',encoding='utf-8') as rh,original_path.open('w',encoding='utf-8') as oh:
                    for i,data in enumerate(ds[split]):
                        raw=_normalize_row(data,source,split,i)
                        try:
                            processed=process_jev_row(raw,source,split,surface_seed)
                        except (ValueError,KeyError,TypeError) as exc:
                            errors[(source,split,type(exc).__name__)]+=1
                            continue
                        processed['original']=True
                        processed['generation']=None
                        _write(rh,raw)
                        _write(oh,processed)
                        counts[key]+=1
                        original_summary[split][source]+=1
                        original_primitive[split][processed['primitive']]+=1
                print(f'  {split:10s} valid originals={counts[key]:,}',flush=True)
            del ds

        active={k:v for k,v in counts.items() if k[1] in splits}
        targets=allocate_targets(active,ratio=ratio,num_generated=num_generated,per_source=per_source,per_source_split=per_source_split)
        processed_tmp={split:stage/f'{split}.merged.jsonl' for split in PROCESSED_SPLITS}
        handles={split:p.open('w',encoding='utf-8') for split,p in processed_tmp.items()}
        usage={s:Counter() for s in PROCESSED_SPLITS}
        final_counts={s:Counter() for s in PROCESSED_SPLITS}
        generated_summary={s:Counter() for s in PROCESSED_SPLITS}
        reports={}
        error_path=stage/'generation_errors.jsonl'
        error_handle=error_path.open('w',encoding='utf-8')
        try:
            # Original rows are *always* preserved in their original split.
            for (source,split),path in stage_originals.items():
                for row in _rows(path):
                    _write(handles[split],row)
                    final_counts[split][source]+=1

            if save_raw:
                # Preserve *all* original raw splits, even if generation was
                # requested only for train (or the generation budget is zero).
                for (source,split),path in raw_paths.items():
                    raw_dir=output_dir/'raw'/source
                    raw_dir.mkdir(parents=True,exist_ok=True)
                    with (raw_dir/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
                        for row in _rows(path):
                            _write(handle,{**row,'original':True,'generation':None})

            for (source,split),target in targets.items():
                if target<=0:continue
                rows=list(_rows(raw_paths[(source,split)]))
                allowed=list_strategies(source)
                if strategies is not None:allowed=[name for name in allowed if name in strategies]
                # Without a validator, skip semantic ablations whose remaining
                # context may still determine the answer.  With a supplied
                # validator those strategies can participate normally.
                if validator is None:
                    unsafe={x['id'] for x in get_specs(source) if x.get('requires_validator')}
                    allowed=[name for name in allowed if name not in unsafe]
                if not allowed:
                    raise ValueError(f'No automatic strategies available for {source}; check --strategies')
                raw_handle=None
                if save_raw:
                    raw_dir=output_dir/'raw'/source
                    raw_handle=(raw_dir/f'{split}.jsonl').open('a',encoding='utf-8')
                print(f'Generating {target:,} for {source}/{split} using {workers} worker(s), {len(allowed)} strategies',flush=True)
                try:
                    report=generate_parallel(
                        source=source,split=split,raw_rows=rows,target=target,strategies=allowed,
                        workers=workers,seed=seed,surface_seed=surface_seed,chunksize=chunksize,
                        deduplicate=deduplicate,strict=strict,validator=validator,
                        output_handle=handles[split],raw_handle=raw_handle,error_handle=error_handle,
                        length_limit=length_limit,
                    )
                finally:
                    if raw_handle:raw_handle.close()
                reports[f'{source}/{split}']=report
                generated_summary[split][source]+=report['generated']
                final_counts[split][source]+=report['generated']
                usage[split].update(report['strategies'])
                print(f'  wrote {report["generated"]:,}/{target:,}, attempts={report["attempted"]:,}, duplicates={report["duplicates"]:,}',flush=True)
        finally:
            for handle in handles.values():handle.close()
            error_handle.close()

        # Only publish split files once every requested quota was reached.
        for split,path in processed_tmp.items():
            path.replace(output_dir/f'{split}.jsonl')
        if error_path.stat().st_size:
            shutil.copyfile(error_path,output_dir/'generation_errors.jsonl')
        elif (output_dir/'generation_errors.jsonl').exists():
            (output_dir/'generation_errors.jsonl').unlink()

        primitive_counts={split:Counter() for split in PROCESSED_SPLITS}
        # Source-specific format is preserved by the upstream processor, and
        # generated tasks can change primitive score/noul into choice.
        for split in PROCESSED_SPLITS:
            for row in _rows(output_dir/f'{split}.jsonl'):
                primitive_counts[split][row['primitive']]+=1
        manifest={
            # These five fields are intentionally identical to the original
            # loader's expected cache manifest: existing train.py works as-is.
            'processing_version':PROCESSING_VERSION,
            'hf_dataset':dataset_name,
            'dataset_revision':revision,
            'configs':configs,
            'surface_seed':surface_seed,
            'split_counts':{s:dict(final_counts[s]) for s in PROCESSED_SPLITS},
            'original_split_counts':{s:dict(original_summary[s]) for s in PROCESSED_SPLITS},
            'generated_split_counts':{s:dict(generated_summary[s]) for s in PROCESSED_SPLITS},
            'primitive_counts':{s:dict(primitive_counts[s]) for s in PROCESSED_SPLITS},
            'strategy_counts':{s:dict(usage[s]) for s in PROCESSED_SPLITS},
            'generation_reports':reports,
            'original':True,
            'generation':{'column_type':'object/null','description':'Strategy ID, derivation, parents, seed, answerability and surfaces'},
            'reformulation':{
                'enabled':True,'ratio':ratio,'num_generated':num_generated,
                'per_source':per_source,'per_source_split':per_source_split,
                'splits':list(splits),'seed':seed,'workers':workers,
                'deduplicate':deduplicate,'strict':strict,'save_raw':save_raw,
                'partial_ablation_requires_validator':True,
                'validator_supplied':validator is not None,
                'length_limit':length_limit,
            },
            'skipped_original_rows':{f'{source}/{split}/{reason}':n for (source,split,reason),n in errors.items()},
            'noul_surface_pairs':[list(x) for x in NOUL_SURFACE_PAIRS],
            'score_surface_styles':list(SCORE_SURFACE_STYLES),
        }
        (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return output_dir


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--hf-dataset',default=DATASET_NAME_DEFAULT,help='Hugging Face dataset ID (original repository default)')
    p.add_argument('--dataset-revision',default='main')
    p.add_argument('--hf-cache-dir',type=Path,default=None)
    p.add_argument('--output-dir',type=Path,default=Path('jev_bench_processed_reformulated'))
    p.add_argument('--configs',default=None,help='Comma-separated source/config names; default = all configs')
    p.add_argument('--splits',default='train,validation,test',help='Splits to augment; unselected splits still retain originals')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--num-generated','--amount',type=int,default=None,help='Exact TOTAL generated examples across all selected sources and splits')
    group.add_argument('--per-source',type=int,default=None,help='Exact new examples per source/config, allocated across selected splits')
    group.add_argument('--per-source-split',type=int,default=None,help='Exact new examples per source AND split')
    group.add_argument('--ratio',type=float,default=None,help='Generated examples per valid original example, source/split-wise')
    p.add_argument('--workers','--num-workers',type=int,default=min(4,os.cpu_count() or 1),help='ProcessPool worker processes')
    p.add_argument('--chunksize',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--surface-seed',type=int,default=42)
    p.add_argument('--strategies',default=None,help='Comma-separated strategy IDs; default all safe automatic strategies')
    p.add_argument('--validator',default=None,help='Optional semantic validator module:function for partial-ablation strategies')
    p.add_argument('--max-length',type=int,default=None,help='Reject generated rows needing more than this many tokens (prefix + longest label); same meaning as train.py --max-length. Default: no check')
    p.add_argument('--max-label-length',type=int,default=96,help='Per-label token cap used for the length check; match train.py --max-label-length')
    p.add_argument('--tokenizer-name',default='Qwen/Qwen3-1.7B-Base',help='Tokenizer used for the length check; match train.py --model-name')
    p.add_argument('--save-raw',action='store_true',help='Also save merged raw JEV rows under raw/<source>/<split>.jsonl')
    p.add_argument('--allow-duplicates',action='store_true',help='Allow exactly duplicated final training inputs')
    p.add_argument('--allow-shortfall',action='store_true',help='Do not fail if a requested count cannot be reached')
    p.add_argument('--force',action='store_true',help='Replace existing split files at output directory')
    return p


def main(argv:list[str]|None=None) -> Path:
    args=parser().parse_args(argv)
    splits=tuple(x.strip() for x in args.splits.split(',') if x.strip())
    configs=[x.strip() for x in args.configs.split(',') if x.strip()] if args.configs else None
    strategies=[x.strip() for x in args.strategies.split(',') if x.strip()] if args.strategies else None
    result=build(dataset_name=args.hf_dataset,revision=args.dataset_revision,cache_dir=args.hf_cache_dir,
                 output_dir=args.output_dir,configs=configs,splits=splits,
                 ratio=1.0 if args.ratio is None else args.ratio,
                 num_generated=args.num_generated,per_source=args.per_source,
                 per_source_split=args.per_source_split,workers=args.workers,seed=args.seed,
                 surface_seed=args.surface_seed,strategies=strategies,strict=not args.allow_shortfall,
                 deduplicate=not args.allow_duplicates,save_raw=args.save_raw,
                 validator=args.validator,force=args.force,chunksize=args.chunksize,
                 length_limit={'max_length':args.max_length,'max_label_length':args.max_label_length,
                               'tokenizer_name':args.tokenizer_name} if args.max_length else None)
    print(f'Merged dataset saved: {result.resolve()}')
    return result


if __name__=='__main__':
    main()
