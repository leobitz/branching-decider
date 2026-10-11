"""Read and write native JEV-Bench rows, including JSON-encoded state/question."""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


def decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(value)
        except (ValueError, SyntaxError, TypeError):
            pass
    return value


@dataclass
class Record:
    id: str
    source: str
    split: str
    primitive: str
    state: Any
    question: dict[str, Any]
    label: str
    soft_label: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Mapping[str, Any] | 'Record') -> 'Record':
        if isinstance(row, cls):
            return row
        question = decode(row.get('question')) or {}
        if not isinstance(question, dict):
            question = {'instructions': str(question)}
        meta = decode(row.get('meta')) or {}
        if not isinstance(meta, dict):
            meta = {'source_meta': meta}
        return cls(
            id=str(row.get('id') or 'unknown'),
            source=str(row.get('source') or ''),
            split=str(row.get('split') or ''),
            primitive=str(row.get('primitive') or question.get('type') or 'choice'),
            state=decode(row.get('state')),
            question=question,
            label=str(row.get('label')),
            soft_label=decode(row.get('soft_label')),
            meta=meta,
        )

    def criteria(self) -> dict[str, Any]:
        c = self.question.get('criteria')
        if c is None:
            c = self.question.get('options', self.question.get('choices', self.question.get('labels')))
        if isinstance(c, dict):
            return {str(k): v for k, v in c.items()}
        if isinstance(c, list):
            return {str(i): x for i, x in enumerate(c)}
        return {}

    def to_row(self) -> dict[str, Any]:
        return dict(
            id=self.id, source=self.source, split=self.split, primitive=self.primitive,
            state=json.dumps(self.state, ensure_ascii=False),
            question=json.dumps(self.question, ensure_ascii=False),
            label=str(self.label),
            soft_label=None if self.soft_label is None else json.dumps(self.soft_label, ensure_ascii=False),
            meta=json.dumps(self.meta, ensure_ascii=False),
        )

    def probabilities(self) -> dict[str, float] | None:
        """Canonical class probabilities; never infer unobserved labels."""
        soft = self.soft_label
        if isinstance(soft, dict):
            try:
                p = {str(k): float(v) for k, v in soft.items()}
            except (TypeError, ValueError):
                return None
        elif isinstance(soft, list) and self.primitive in {'score', 'choice'}:
            keys = list(self.criteria())
            if len(soft) != len(keys):
                return None
            p = {key: float(x) for key, x in zip(keys, soft)}
        elif self.primitive == 'noul' and isinstance(soft, (float, int)):
            f = float(soft)
            p = {'0': 1-f, '1': f}
        else:
            return None
        if not p or any(x < 0 for x in p.values()):
            return None
        total = sum(p.values())
        return {k: x/total for k, x in p.items()} if total > 0 else None
