#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic external-representation interchange contracts."""

from ._inspection import (
    HostInspectionConversion,
    HostInspectionField,
    HostInspectionFrame,
)
from ._report import (
    AdapterCapability,
    AdapterError,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterNegotiationResult,
    AdapterReport,
    AdapterRequirement,
    AdapterStatus,
    AdapterWaiver,
    compose_adapter_reports,
    negotiate_adapter,
    require_lossless,
)


__all__ = [
    "HostInspectionConversion",
    "HostInspectionField",
    "HostInspectionFrame",
    "AdapterCapability",
    "AdapterError",
    "AdapterFormatProfile",
    "AdapterLoss",
    "AdapterNegotiationResult",
    "AdapterReport",
    "AdapterStatus",
    "AdapterRequirement",
    "require_lossless",
    "AdapterWaiver",
    "compose_adapter_reports",
    "negotiate_adapter",
]
