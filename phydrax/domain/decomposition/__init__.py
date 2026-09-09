#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._adapters import (
    CoverAdapterEvidence,
    geometry_subdomain_patch,
    validate_atlas_cover_adapter,
    validate_cell_partition_cover,
)
from ._cartesian import (
    AxisPartition,
    BoxPartition,
    cartesian_subdomain_cover,
    CartesianCoverPlan,
    normalized_patch_coordinate,
)
from ._cover import (
    PairedSupport,
    PairedSupportEvidence,
    PairingTopology,
    SubdomainCover,
    SubdomainCoverEvidence,
    SubdomainPatch,
)
from ._fields import (
    broken_field,
    BrokenField,
    LocalFieldFamily,
    LocalFieldRef,
    partition_of_unity_family,
    partition_of_unity_field,
)
from ._geometry import (
    MappedCoverEvidence,
    MappedCoverValidationPlan,
    validate_mapped_cover,
)
from ._hierarchy import SubdomainHierarchy, SubdomainLevel
from ._ownership import (
    cover_integration_ownership,
    IntegrationOwnership,
    IntegrationOwnershipEvidence,
)
from ._routing import prepare_field_routing, PreparedFieldRouting


__all__ = [
    "AxisPartition",
    "BoxPartition",
    "CoverAdapterEvidence",
    "BrokenField",
    "CartesianCoverPlan",
    "IntegrationOwnership",
    "IntegrationOwnershipEvidence",
    "LocalFieldFamily",
    "LocalFieldRef",
    "MappedCoverEvidence",
    "MappedCoverValidationPlan",
    "PairedSupport",
    "PairedSupportEvidence",
    "PairingTopology",
    "PreparedFieldRouting",
    "SubdomainCover",
    "SubdomainCoverEvidence",
    "SubdomainHierarchy",
    "SubdomainLevel",
    "SubdomainPatch",
    "broken_field",
    "cartesian_subdomain_cover",
    "geometry_subdomain_patch",
    "cover_integration_ownership",
    "normalized_patch_coordinate",
    "partition_of_unity_field",
    "partition_of_unity_family",
    "prepare_field_routing",
    "validate_mapped_cover",
    "validate_atlas_cover_adapter",
    "validate_cell_partition_cover",
]
