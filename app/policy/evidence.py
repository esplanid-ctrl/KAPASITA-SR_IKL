"""
Evidence registry access for the Policy/Evidence layer.

data/POLICY_SOURCE_REGISTER.csv is the single source of truth for policy
evidence (MASTER_PROMPT.md rule #6: source -> year -> ... traceability).
A rule may only cite a source_id that actually exists in this registry —
citing anything else is treated as inventing a source (rule #5) and is
rejected, not warned about.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REGISTER_PATH = Path(__file__).resolve().parents[2] / "data" / "POLICY_SOURCE_REGISTER.csv"
REQUIRED_COLUMNS = {"source_id", "title", "institution", "year", "scope", "role", "status"}


class EvidenceError(RuntimeError):
    """Raised when code (or a rule) references policy evidence that does
    not exist in the registry."""


def load_policy_register(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or REGISTER_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Policy source register not found at {path}. No policy "
            "recommendation may be produced without a real evidence registry."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Policy source register missing required columns: {sorted(missing)}")
    return df


def get_sources(source_ids: List[str], register: pd.DataFrame) -> List[Dict]:
    """Returns evidence records for the given source_ids, in the order given.

    Raises EvidenceError if any id is not present in the registry — a rule
    must never cite evidence that doesn't exist.
    """
    if not source_ids:
        return []
    found = register[register["source_id"].isin(source_ids)]
    found_ids = set(found["source_id"])
    missing = set(source_ids) - found_ids
    if missing:
        raise EvidenceError(
            f"Unknown policy source_id(s): {sorted(missing)}. A rule must "
            "never cite evidence that is not in POLICY_SOURCE_REGISTER.csv."
        )
    # preserve caller-specified order for readable citations
    ordered = sorted(found.to_dict(orient="records"), key=lambda r: source_ids.index(r["source_id"]))
    return ordered


def is_effective(source_record: Dict) -> bool:
    """Low-level helper: is the REGISTRY's own status field 'official'?

    This is deliberately narrow and must not be confused with a rule's
    overall `rule_status` or `policy_effective_status` (see
    app/policy/engine.py), which separate:
      - source_verified: has a human confirmed this rule's citation is
        actually relevant (section/page) -- independent of registry status.
      - policy_effective_status: what the registry status actually means
        ("official" -> effective, "verify" -> pending_verification,
        "secondary" -> supporting_context) -- 'verify' is NEVER collapsed
        into "not effective".
      - rule_status: the combined taxonomy value exposed to the user
        (candidate / verified / rejected / pending_policy_verification /
        data_unavailable / unlinked_policy / policy_inactive).
    """
    return str(source_record.get("status", "")).strip().lower() == "official"

