#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import BoundarySurfaceTrace
from ...discretization.finite_volume import (
    HybridMimeticDiffusion,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ...interchange import GeospatialContract
from ...meshing import CellMeshingResult, MeshingDerivativeMode


class QualifiedVoroCrustPorousMesh(StrictModule, NonTrainableState):
    """VoroCrust result admitted for native hybrid porous operators.

    The qualification certifies the geometry consumed by Phydrax. It explicitly
    does not claim retained Voronoi generators, Delaunay-dual orthogonality,
    source material transfer, TPFA consistency, or differentiable meshing.
    """

    result: CellMeshingResult
    discretization: UnstructuredFiniteVolumeDiscretization
    coordinates: GeospatialContract
    surface_trace: BoundarySurfaceTrace | None
    generator_dual_available: bool = eqx.field(static=True)
    tpfa_certified: bool = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)

    def require_surface_trace(self) -> BoundarySurfaceTrace:
        if self.surface_trace is None:
            raise ValueError("No physical surface faces were explicitly qualified.")
        return self.surface_trace

    def require_tpfa(self) -> None:
        raise ValueError(
            "VoroCrust provider identity does not certify TPFA for the consumed geometry and tensor."
        )


def qualify_vorocrust_porous_mesh(
    result: CellMeshingResult,
    coordinates: GeospatialContract,
    /,
    *,
    surface_faces: ArrayLike | None = None,
) -> QualifiedVoroCrustPorousMesh:
    if not isinstance(result, CellMeshingResult):
        raise TypeError(
            "VoroCrust qualification requires a successful CellMeshingResult."
        )
    if result.provider.name != "vorocrust":
        raise ValueError(
            "Physical VoroCrust qualification cannot be attached to another provider."
        )
    if result.derivative_mode is not MeshingDerivativeMode.NONDIFFERENTIABLE:
        raise ValueError(
            "The VoroCrust topology and geometry must remain explicitly nondifferentiable."
        )
    if not isinstance(coordinates, GeospatialContract):
        raise TypeError("Physical mesh coordinates require a GeospatialContract.")
    coordinates.require_cartesian(result.coordinate_contract, dimensions=3)
    if result.zones or result.labels or result.associations:
        raise ValueError(
            "Current VoroCrust import does not qualify source regions, labels, or associations for porous materials."
        )
    discretization = UnstructuredFiniteVolumePlan.from_cell_mesh(result.mesh).prepare()
    # Construction is the numerical geometry certificate consumed by Richards,
    # heat and transport; no backend mesh-quality adjective substitutes for it.
    HybridMimeticDiffusion(discretization)
    trace = (
        None
        if surface_faces is None
        else BoundarySurfaceTrace(discretization, surface_faces)
    )
    identifier = canonical_fingerprint(
        {
            "kind": "vorocrust-porous-mesh-qualification",
            "result": result.result_id,
            "geometry": discretization.geometry_id,
            "coordinates": coordinates.geospatial_id,
            "surface_trace": None if trace is None else trace.trace_id,
            "generator_dual_available": False,
            "tpfa_certified": False,
        }
    )
    return QualifiedVoroCrustPorousMesh(
        result,
        discretization,
        coordinates,
        trace,
        False,
        False,
        identifier,
    )


__all__ = ["QualifiedVoroCrustPorousMesh", "qualify_vorocrust_porous_mesh"]
