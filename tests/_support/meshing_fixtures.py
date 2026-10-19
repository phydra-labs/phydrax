"""Identity-bound meshing requests and small native source fixtures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phydrax import SpatialCoordinateContract
from phydrax._meshcore import TetMeshBoundaryPolicy
from phydrax.discretization import AdaptiveSimplexPolicy, CellMesh
from phydrax.geometry import MeshingDomain
from phydrax.geometry._domain_coverage import DomainCoverageCertificate
from phydrax.geometry._embedding_state import MeshFindingStatus
from phydrax.geometry._global_embedding import GlobalEmbeddingCertificate
from phydrax.geometry._source_fidelity import SourceFidelityCertificate
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingResult,
    CellMeshingTarget,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshingEntityKind,
    MeshingLimits,
    MeshingScope,
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativePlcSource,
    NativeSurfaceSource,
    PiecewiseLinearComplex,
    RegionSeed,
    SizeControlStrength,
    SurfaceMeshingSpec,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._mesh_part import MeshPart
from phydrax.typing import ConvertibleToArray


# Corner i uses x=i&1, y=(i>>1)&1, z=(i>>2)&1; loops face the exterior.
BOX_LOOPS = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)


def mesh_scope(
    mesh: CellMesh, dimension: int, entity_ids: ConvertibleToArray | None = None
) -> MeshingScope:
    """Bind stable entity IDs to the mesh's current topology set and numeric revision."""
    entities = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        entities.entity_ids
        if entity_ids is None
        else np.asarray(entity_ids, dtype=np.int64),
    )


def domain_scope(
    domain: MeshingDomain, dimension: int, entity_ids: ConvertibleToArray
) -> MeshingScope:
    """Bind domain entity indices to the declared geometry source and revision."""
    return MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        domain.entity_set_id(dimension),
        np.asarray(entity_ids, dtype=np.int64),
    )


def surface_specification(
    domain: MeshingDomain,
    size: float,
    *,
    limits: MeshingLimits | None = None,
    strength: SizeControlStrength = SizeControlStrength.SOFT,
) -> SurfaceMeshingSpec:
    """Request triangular surface cells over all authored patches, with one bound size control."""
    scope = domain_scope(domain, 2, np.arange(len(domain.patches), dtype=np.int64))
    return SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, size, strength=strength),),
        limits=limits,
    )


def volume_specification(
    source: NativePlcSource,
    *,
    size: float,
    limits: MeshingLimits | None = None,
    region_seeds: tuple[RegionSeed, ...] = (),
    strength: SizeControlStrength = SizeControlStrength.SOFT,
) -> VolumeMeshingSpec:
    """Request tetrahedra inside the PLC with its complete, revision-bound facet scope."""
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        f"{source.source_id}-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, size, strength=strength),),
        region_seeds=region_seeds,
        limits=limits,
    )


def surface_mesh(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    *,
    coordinate_contract: SpatialCoordinateContract = SpatialCoordinateContract.si(),
) -> CellMeshingResult:
    """Execute the native parametric-surface route and retain its certified result."""
    provider = NativeMeshingProvider(NativeMeshingOptions("parametric_surface"))
    return provider.plan(
        NativeSurfaceSource(domain),
        specification,
        coordinate_contract=coordinate_contract,
    ).execute()


def volume_mesh(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    *,
    coordinate_contract: SpatialCoordinateContract = SpatialCoordinateContract.si(),
) -> CellMeshingResult:
    """Execute the native PLC-tetrahedral route without discarding source identity or evidence."""
    provider = NativeMeshingProvider(NativeMeshingOptions("plc_tetrahedral"))
    return provider.plan(
        source, specification, coordinate_contract=coordinate_contract
    ).execute()


def certified_carrier(part: MeshPart) -> CellMeshingResult:
    """Return a part's certified cell producer; refuse a different producer kind."""
    match part.carrier:
        case CellMeshingResult() as carrier:
            return carrier
        case _:
            raise TypeError(
                f"meshing_fixtures.part.carrier must be CellMeshingResult; got {type(part.carrier).__name__}"
            )


def box_corners(lower: Sequence[float], upper: Sequence[float]) -> NDArray[np.float64]:
    """Return binary-indexed corners compatible with the outward BOX_LOOPS."""
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if (
        lo.shape != (3,)
        or hi.shape != (3,)
        or not np.all(np.isfinite((lo, hi)))
        or np.any(hi <= lo)
    ):
        raise ValueError(
            f"meshing_fixtures.box must have finite increasing 3D bounds; got {lo}, {hi}"
        )
    return np.asarray(
        [[(hi if i >> axis & 1 else lo)[axis] for axis in range(3)] for i in range(8)],
        dtype=np.float64,
    )


def box_plc(
    lower: Sequence[float] = (0.0, 0.0, 0.0),
    upper: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    region_id: str = "solid",
    boundary: TetMeshBoundaryPolicy = "conforming",
) -> PiecewiseLinearComplex:
    """Return a closed outward six-facet PLC; negative sides belong to the named region."""
    return PiecewiseLinearComplex(
        box_corners(lower, upper),
        tuple(np.asarray(loop, dtype=np.int64) for loop in BOX_LOOPS),
        np.arange(6, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
        (region_id,),
        boundary=boundary,
    )


def adaptation_policy(
    route: MeshAdaptationRoute,
    *,
    device_policy: AdaptiveSimplexPolicy | None = None,
    limits: MeshingLimits | None = None,
    protected_scopes: tuple[MeshingScope, ...] = (),
) -> MeshAdaptationPolicy:
    """Supply a capacity policy exactly for device routes; preserve the requested route."""
    match route:
        case MeshAdaptationRoute.DEVICE_BISECTION | MeshAdaptationRoute.DEVICE_METRIC_2D:
            capacity = AdaptiveSimplexPolicy() if device_policy is None else device_policy
        case _:
            capacity = device_policy
    return MeshAdaptationPolicy(
        route, device_policy=capacity, limits=limits, protected_scopes=protected_scopes
    )


def finding_checks(
    certificate: GlobalEmbeddingCertificate
    | DomainCoverageCertificate
    | SourceFidelityCertificate,
    status: MeshFindingStatus,
) -> set[str]:
    """Return check codes at one finding status without substituting overall certificate status."""
    return {value.check for value in certificate.findings if value.status == status}
