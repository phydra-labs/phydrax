"""Author curved swept source controls before target generation."""

from __future__ import annotations

import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    swept_coordinate_element,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._source_fidelity import MappedDomainBoundarySource
from phydrax.meshing import LayerSchedule, SweepControl, SweepMapKind
from tests._support.coordinate_references import coordinate_reference_budget


def twisted_curved_source(
    degree: int = 2,
    /,
    *,
    angle: float = 0.15,
    cap_offset: tuple[float, float, float] = (0.1, -0.05, 1.0),
) -> tuple[phx.meshing.NativeSweepSource, phx.meshing.VolumeMeshingSpec]:
    """Author a non-box source independently of target construction."""
    profile_element = coordinate_lagrange_element("quadrilateral", degree)
    nodes = np.asarray(profile_element.reference_nodes)
    u, v = nodes[:, 0], nodes[:, 1]
    controls = np.column_stack((u + 0.25 * v, v + 0.125 * u * (1.0 - u)))
    corners = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.25, 1.0), (0.25, 1.0)), dtype=np.float64
    )
    profile = CellMesh(
        corners,
        (
            CellBlock(
                "profile", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    profile_geometry = CellGeometrySpec(
        {"profile": profile_element},
        {"profile": np.arange(profile_element.local_dof_count, dtype=np.int32)[None, :]},
        controls,
    )
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    frames = np.stack((np.eye(3, dtype=np.float64), rotation))
    offsets = np.asarray(((0.0, 0.0, 0.0), cap_offset), dtype=np.float64)
    control = SweepControl(
        SweepMapKind.FRAMES,
        LayerSchedule(np.asarray((1.0,), dtype=np.float64)),
        "authored-piecewise-affine-twist",
        frames=frames,
        translations=offsets,
    )
    # Root reference topology and all physical controls preexist generation.
    cube = reference_cell_topology("hexahedron")
    reference_points = np.asarray(cube.vertices, dtype=np.float64)
    reference_mesh = CellMesh(
        reference_points,
        (CellBlock("root", "hexahedron", np.arange(8, dtype=np.int32)[None, :]),),
    )
    facets = np.asarray(
        tuple(
            triangle
            for a, b, c, d in cube.entities[2]
            for triangle in ((a, b, c), (a, c, d))
        ),
        dtype=np.int64,
    )
    reference_domain = PiecewiseLinearDomain(
        reference_points,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (facets.shape[0], 1)),
        ("solid",),
        source_id="twisted-source-reference",
    )
    source_points = np.pad(controls, ((0, 0), (0, 1)))
    root_controls = np.concatenate(
        (source_points, source_points @ rotation.T + offsets[1])
    )
    with coordinate_reference_budget() as reference_budget:
        root_element = (
            coordinate_lagrange_element("hexahedron", 1)
            if degree == 1
            else swept_coordinate_element(profile_element, budget=reference_budget)
        )
    root_geometry = CellGeometrySpec(
        {"root": root_element},
        {"root": np.arange(root_element.local_dof_count, dtype=np.int32)[None, :]},
        root_controls,
    )
    revision = canonical_fingerprint(
        {
            "kind": "independently-authored-twisted-source",
            "profile_basis": profile_element.element_id,
            "profile_controls": controls,
            "frames": frames,
            "offsets": offsets,
        }
    )
    domain = MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        root_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="twisted-source",
        source_revision=revision,
    )
    query = MappedDomainBoundarySource(domain, np.asarray((0,), dtype=np.int64))
    correspondence = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        reference_mesh.topology_id,
        {"sweep:profile": np.asarray((0,), dtype=np.int64)},
        {"sweep:profile": np.arange(8, dtype=np.int64)[None, :]},
    )
    source = phx.meshing.NativeSweepSource(
        profile,
        control,
        domain.source_id,
        revision,
        fidelity_source=query,
        maximum_deviation=1e-10,
        domain=domain,
        profile_geometry=profile_geometry,
        root_correspondence=correspondence,
    )
    scope = phx.meshing.MeshingScope(
        domain.source_id,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "twisted-source-boundary",
        np.arange(facets.shape[0], dtype=np.int64),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("hexahedron",)),
            geometry_order=degree,
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 1.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    return source, specification


def stored_sweep_control(kind: SweepMapKind, /) -> SweepControl:
    """Build sweep controls with explicit state for restoration checks."""
    schedule = LayerSchedule(np.asarray((0.25, 0.25), dtype=np.float64))
    match kind:
        case SweepMapKind.FRAMES:
            return SweepControl(
                kind,
                schedule,
                "stored-frame-map",
                origin=np.asarray((0.25, 0.0, 0.0), dtype=np.float64),
                frames=np.broadcast_to(np.eye(3, dtype=np.float64), (3, 3, 3)),
                translations=np.asarray(
                    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.0, 0.0, 0.5)),
                    dtype=np.float64,
                ),
            )
        case SweepMapKind.EXTRUSION | SweepMapKind.REVOLUTION:
            return SweepControl(
                kind,
                schedule,
                "stored-analytic-map",
                origin=np.asarray((0.25, 0.0, 0.0), dtype=np.float64),
            )
        case _:
            raise ValueError(
                f"sweep_fixture.kind must be a supported SweepMapKind; got {kind}"
            )
