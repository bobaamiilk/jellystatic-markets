"""Macro event calendar stub — tagging only, never used for trading decisions.

Events are loaded from a YAML file (list of {date, label} mappings) or from
the built-in APAC defaults. The engine attaches event labels to report output
so analysts can see which drawdown dates coincided with a BOJ/Fed print.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class MacroEvent:
    date: str    # ISO 8601, e.g. "2024-07-31"
    label: str   # human-readable, e.g. "BOJ surprise hike"


_APAC_DEFAULTS: list[dict[str, str]] = [
    {"date": "2024-01-23", "label": "BOJ Jan policy meeting"},
    {"date": "2024-03-19", "label": "BOJ ends negative rates"},
    {"date": "2024-03-20", "label": "Fed FOMC (hold)"},
    {"date": "2024-04-26", "label": "BOJ Apr policy meeting"},
    {"date": "2024-06-12", "label": "Fed FOMC (hold)"},
    {"date": "2024-07-31", "label": "BOJ surprise hike"},
    {"date": "2024-08-05", "label": "JPY carry unwind"},
    {"date": "2024-09-18", "label": "Fed first cut (−50 bp)"},
    {"date": "2024-10-31", "label": "BOJ Oct policy meeting"},
    {"date": "2024-11-07", "label": "Fed cut (−25 bp)"},
    {"date": "2024-12-18", "label": "Fed Dec cut; dot-plot hawkish"},
    {"date": "2025-01-29", "label": "Fed FOMC (hold)"},
    {"date": "2025-02-24", "label": "China NPC budget signal"},
    {"date": "2025-03-19", "label": "BOJ hike to 0.5%"},
    {"date": "2025-04-02", "label": "US tariff announcement"},
    {"date": "2025-05-07", "label": "RBA cut"},
    {"date": "2025-06-11", "label": "Fed FOMC"},
]


def load_events(path: str | Path | None = None) -> list[MacroEvent]:
    """Load macro events from a YAML file, or return the built-in APAC defaults.

    Expected YAML shape::

        events:
          - date: "2024-07-31"
            label: "BOJ surprise hike"
    """
    if path is None:
        raw = _APAC_DEFAULTS
    else:
        with open(path) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        raw = data.get("events", [])
    return [MacroEvent(**e) for e in raw]


def events_on_date(
    events: list[MacroEvent], date: str | pd.Timestamp
) -> list[MacroEvent]:
    """Filter events that fall on a specific date."""
    target = pd.Timestamp(date).date().isoformat()
    return [e for e in events if e.date == target]


def events_in_range(
    events: list[MacroEvent],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> list[MacroEvent]:
    """Filter events falling within [start, end] inclusive."""
    s = pd.Timestamp(start).date().isoformat()
    e = pd.Timestamp(end).date().isoformat()
    return [ev for ev in events if s <= ev.date <= e]
