#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic external-representation interchange contracts."""

from . import energy_runtime, fmi, helics, opticstudio
from ._inspection import (
    HostInspectionConversion,
    HostInspectionField,
    HostInspectionFrame,
)
from ._mesh_arrays import (
    MeshArrayArtifact,
    MeshArrayAssociation,
    MeshArrayBlock,
    MeshArrayField,
    MeshArraySelection,
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
from ._resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceManifest,
    ResourceReadError,
)


__all__ = [
    "energy_runtime",
    "fmi",
    "helics",
    "opticstudio",
    "HostInspectionConversion",
    "HostInspectionField",
    "HostInspectionFrame",
    "MeshArrayArtifact",
    "MeshArrayAssociation",
    "MeshArrayBlock",
    "MeshArrayField",
    "MeshArraySelection",
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
    "BoundedResource",
    "ResourceLimits",
    "ResourceManifest",
    "ResourceReadError",
    "account_bounded_resource",
    "bounded_resource_from_bytes",
    "read_bounded_resource",
    "compose_adapter_reports",
    "negotiate_adapter",
]
