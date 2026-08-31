#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reproducible, dependency-honest advanced solver benchmark harnesses."""

from .adapters.base import CaseSpec, Tolerances
from .campaign import build_cases, CampaignConfig, PRESETS
from .compare import compare_reports, IncomparableReportsError
from .harness import execute_case, run_campaign
from .problems import default_problems
from .schema import SCHEMA_VERSION, SchemaError, validate_report, validate_row


__all__ = [
    "CampaignConfig",
    "CaseSpec",
    "IncomparableReportsError",
    "PRESETS",
    "SCHEMA_VERSION",
    "SchemaError",
    "Tolerances",
    "build_cases",
    "compare_reports",
    "default_problems",
    "execute_case",
    "run_campaign",
    "validate_report",
    "validate_row",
]
