"""
Centralized configuration for KAPASITA-MVP.

Rule (MASTER_PROMPT.md #F/#G, docs/11, docs/12, AT-09):
No source URL and no API key may ever be hard-coded in application code.
Every value here is read from environment variables / Streamlit secrets at
runtime. This module is the ONLY place allowed to call os.environ for these
names, so a single audit point exists for "no secret in Git" checks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


class ConfigError(RuntimeError):
    """Raised when a required piece of configuration is missing."""


def _get(name: str, required: bool, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, default)
    if required and not value:
        raise ConfigError(
            f"Missing required configuration '{name}'. "
            "Set it as an environment variable or in .streamlit/secrets.toml "
            "(see .env.example). It must never be hard-coded in source code."
        )
    return value


@dataclass(frozen=True)
class DriveConfig:
    data_folder_id: str
    policy_folder_id: str
    service_account_json: str


@dataclass(frozen=True)
class RegionSourceConfig:
    """Region source is kept SEPARATE from DATA and POLICY (project-owner
    decision): a different Drive folder, a different secret name, and a
    different lifecycle (built once as a canonical artifact, not
    downloaded/reparsed on every request)."""
    region_folder_id: str
    service_account_json: str


@dataclass(frozen=True)
class CopilotConfig:
    api_key: Optional[str]
    model: str
    enabled: bool


def load_drive_config(required: bool = True) -> DriveConfig:
    """Loads Google Drive access configuration.

    required=False allows callers (e.g. tests, or a dashboard boot check)
    to detect that Drive is not configured yet instead of crashing.
    """
    return DriveConfig(
        data_folder_id=_get("GOOGLE_DRIVE_DATA_FOLDER_ID", required) or "",
        policy_folder_id=_get("GOOGLE_DRIVE_POLICY_FOLDER_ID", required) or "",
        service_account_json=_get("GOOGLE_SERVICE_ACCOUNT_JSON", required) or "",
    )


def load_region_source_config(required: bool = False) -> RegionSourceConfig:
    """Loads the region-master Drive source configuration.

    Separate from load_drive_config() on purpose: the region master
    (cahyadsn/wilayah) is a distinct artifact lifecycle -- it should be
    fetched and preprocessed into a canonical CSV (see
    app.data.region_master.build_canonical_region_master +
    save_canonical_artifact) as an offline/preprocessing step, not
    downloaded or re-parsed on every Streamlit page load. Defaults to
    required=False since the dashboard can still run (without a region
    join / choropleth) while this isn't configured yet.
    """
    return RegionSourceConfig(
        region_folder_id=_get("GOOGLE_DRIVE_REGION_FOLDER_ID", required) or "",
        service_account_json=_get("GOOGLE_SERVICE_ACCOUNT_JSON", required) or "",
    )


def load_copilot_config() -> CopilotConfig:
    """Gemini is optional at runtime (docs/07 AT-10: dashboard must remain
    usable without Gemini), so this never raises — it reports enabled=False
    instead."""
    api_key = _get("GEMINI_API_KEY", required=False)
    model = _get("GEMINI_MODEL", required=False, default="gemini-2.5-flash")
    return CopilotConfig(api_key=api_key, model=model, enabled=bool(api_key))


def assert_no_hardcoded_secrets_placeholder() -> None:
    """Documentation-as-code reminder: this function intentionally does
    nothing at runtime. Its purpose is to be the single named symbol that
    AT-09 acceptance testing / code review greps for, to confirm secret
    handling is centralized here and not duplicated ad hoc elsewhere."""
    return None
