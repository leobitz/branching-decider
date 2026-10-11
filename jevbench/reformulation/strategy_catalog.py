"""Lightweight access to the strategy plan, independent of torch or HF."""
from __future__ import annotations
import json
from pathlib import Path
CATALOG = json.loads(Path(__file__).with_name('strategy_plan.json').read_text(encoding='utf-8'))
