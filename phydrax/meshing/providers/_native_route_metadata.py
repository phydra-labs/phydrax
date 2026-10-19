#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Named organization and association result for native mesh routes."""

from typing import assert_never, final

import numpy as np

from phydrax._strict import StrictModule
from phydrax.discretization import CellMesh
from phydrax.meshing._association_state import GeometryAssociation
from phydrax.meshing._contracts import (
    CurveMeshingSpec,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from phydrax.meshing._organization_records import MeshLabel, MeshPatch, MeshZone
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.typing import checked


@final
class NativeRouteMetadata(StrictModule):
    zones: tuple[MeshZone, ...]
    patches: tuple[MeshPatch, ...]
    labels: tuple[MeshLabel, ...]
    associations: tuple[GeometryAssociation, ...]


@checked
def mesh_entity_scope(
    mesh: CellMesh, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    """Bind mesh entity identifiers to their dimension and revision."""
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        identifiers,
    )


@checked
def specification_scope(
    specification: CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec, /
) -> MeshingScope:
    """Return the declared meshing scope for a surface, curve, or volume request."""
    match specification:
        case CurveMeshingSpec() | SurfaceMeshingSpec():
            return specification.scope
        case VolumeMeshingSpec():
            return specification.boundary_scope
        case invalid:
            assert_never(invalid)
