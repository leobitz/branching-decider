"""Source-specific strategy lookup."""
from __future__ import annotations

from importlib import import_module
from .strategy_catalog import CATALOG


def list_strategies(source: str | None = None) -> list[str]:
    return sorted(spec['id'] for item in CATALOG if source is None or item['source'] == source for spec in item['strategies'])


def get_strategy(strategy_id: str):
    source = next((item['source'] for item in CATALOG if any(s['id'] == strategy_id for s in item['strategies'])), None)
    if source is None:
        raise KeyError(f'Unknown strategy {strategy_id!r}')
    return getattr(import_module(f'{__package__}.sources.{source}'), strategy_id)


def get_specs(source: str) -> list[dict]:
    return next((o['strategies'] for o in CATALOG if o['source'] == source), [])
