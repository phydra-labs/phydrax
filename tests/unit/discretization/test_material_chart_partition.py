#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Algorithm contracts for rational chart partitions and native clipping."""

import math
from contextlib import AbstractContextManager, nullcontext
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._chart_certification import certify_chart_endpoint
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._surface_chart_deformation import (
    _certify_surface_endpoints,
    _exact_chart_partition_proof,
    _renew_restricted_surface_validity,
    prepare_exact_material_pieces,
)
from phydrax.discretization._surface_chart_records import (
    material_point,
    SurfaceChartWitness,
)
from phydrax.geometry import MeshingDomain
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.meshing.qualification import folded_plates_surface
from tests._support.coordinate_references import coordinate_reference_budget


def test_material_partitions_retain_nonrepresentable_rational_coordinates() -> None:

    zero = Fraction(0)
    a, b, c, d = (
        (zero, zero),
        (Fraction(1, 3), zero),
        (Fraction(1, 3), Fraction(1, 7)),
        (zero, Fraction(1, 7)),
    )
    source, target = ((a, b, c), (a, c, d)), ((a, b, d), (b, c, d))
    with (
        NativeExecutionScope(
            NativeExecutionLimits(2 * 1_000_000, 0, 0, 64 * 1024**2, math.inf)
        ) as scope,
        scope.batch(2 * 1_000_000, bytes_upper=64 * 1024**2) as ticket,
    ):
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=1_000_000,
            maximum_memory_bytes=64 * 1024**2,
        )
        with budget.activate(), ticket.native_target():
            pieces, work, pairs, retained = prepare_exact_material_pieces(
                source,
                target,
                np.asarray((17, 91), dtype=np.int64),
                np.asarray((53, 211), dtype=np.int64),
                16,
                1_000_000,
                64 * 1024**2,
                budget=budget,
            )
    assert pieces and work > 0 and pairs > 0 and retained > 0
    old_areas, new_areas = [zero, zero], [zero, zero]
    for piece in pieces:
        old_areas[piece.source_row] += piece.source_reference_area
        new_areas[piece.target_row] += piece.target_reference_area
        assert tuple(
            material_point(source[piece.source_row], point)
            for point in piece.exact_source_reference_vertices
        ) == tuple(
            material_point(target[piece.target_row], point)
            for point in piece.exact_target_reference_vertices
        )
        assert piece.source_cell_global_id == (17, 91)[piece.source_row]
        assert piece.target_cell_global_id == (53, 211)[piece.target_row]
    assert old_areas == new_areas == [Fraction(1, 2), Fraction(1, 2)]
    assert (
        Fraction(float(b[0])) != b[0]
    )  # Native topology retains the nonrepresentable rational value.


def test_material_partition_refuses_uncovered_reference_area() -> None:

    source = (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
    target = (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1, 2), Fraction(0)),
            (Fraction(0), Fraction(1, 2)),
        ),
    )
    with (
        NativeExecutionScope(
            NativeExecutionLimits(2 * 1_000_000, 0, 0, 64 * 1024**2, math.inf)
        ) as scope,
        scope.batch(2 * 1_000_000, bytes_upper=64 * 1024**2) as ticket,
    ):
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=1_000_000,
            maximum_memory_bytes=64 * 1024**2,
        )
        with budget.activate(), ticket.native_target():
            with pytest.raises(
                ValueError, match=r"surface_chart\.prepare_exact_material_pieces\.pieces"
            ):
                prepare_exact_material_pieces(
                    source,
                    target,
                    np.asarray((17,), dtype=np.int64),
                    np.asarray((53,), dtype=np.int64),
                    4,
                    1_000_000,
                    64 * 1024**2,
                    budget=budget,
                )


def test_material_partition_rejects_balanced_gap_and_double_cover() -> None:

    source = (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
    half = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1, 2)),
    )
    # Net material area matches the source, but one half is doubled and the
    # other is absent. Matching total/piece reference areas alone is not proof.
    with (
        NativeExecutionScope(
            NativeExecutionLimits(2 * 1_000_000, 0, 0, 64 * 1024**2, math.inf)
        ) as scope,
        scope.batch(2 * 1_000_000, bytes_upper=64 * 1024**2) as ticket,
    ):
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=1_000_000,
            maximum_memory_bytes=64 * 1024**2,
        )
        with budget.activate(), ticket.native_target():
            with pytest.raises(
                ValueError, match=r"surface_chart\.prepare_exact_material_pieces\.pieces"
            ):
                prepare_exact_material_pieces(
                    source,
                    (half, half),
                    np.asarray((17,), dtype=np.int64),
                    np.asarray((53, 211), dtype=np.int64),
                    16,
                    1_000_000,
                    64 * 1024**2,
                    budget=budget,
                )


@pytest.mark.parametrize(
    "target_charts,visits,outcome",
    [
        pytest.param(
            np.asarray(
                (
                    ((0.0, 0.0), (1.0, 0.0), (0.25, 0.25)),
                    ((1.0, 0.0), (0.0, 1.0), (0.25, 0.25)),
                    ((0.0, 1.0), (0.0, 0.0), (0.25, 0.25)),
                ),
                dtype=np.float64,
            ),
            12,
            nullcontext(),
            id="interior-split",
        ),
        pytest.param(
            np.asarray(
                (
                    ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5)),
                    ((0.0, 0.0), (0.5, 0.5), (0.0, 1.0)),
                ),
                dtype=np.float64,
            ),
            9,
            nullcontext(),
            id="boundary-midpoint-split",
        ),
        pytest.param(
            np.asarray(
                (
                    ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5 + 2**-20)),
                    ((0.0, 0.0), (0.5, 0.5 + 2**-20), (0.0, 1.0)),
                ),
                dtype=np.float64,
            ),
            None,
            pytest.raises(
                ValueError, match=r"surface_chart\.exact_chart_partition_proof\.pieces"
            ),
            id="off-boundary-split",
        ),
        pytest.param(
            np.asarray(
                (
                    ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5)),
                    ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5)),
                ),
                dtype=np.float64,
            ),
            None,
            pytest.raises(
                ValueError, match=r"surface_chart\.exact_chart_partition_proof\.pieces"
            ),
            id="balanced-gap-and-overlap",
        ),
    ],
)
def test_restricted_partition_preserves_signed_area_and_boundary_chain(
    target_charts: NDArray[np.float64],
    visits: int | None,
    outcome: AbstractContextManager[object],
) -> None:
    source = SurfaceChartWitness(
        geometry_id="source-geometry",
        topology_id="source-topology",
        domain_id="material-domain",
        cell_global_ids=np.asarray((17,), dtype=np.int64),
        patches=np.asarray((0,), dtype=np.int32),
        charts=np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),), dtype=np.float64),
        geometry_entity_ids=("patch",),
        occurrence_paths=(("surface",),),
    )
    count = target_charts.shape[0]
    target = SurfaceChartWitness(
        geometry_id="target-geometry",
        topology_id="target-topology",
        domain_id="material-domain",
        cell_global_ids=np.arange(23, 23 + count, dtype=np.int64),
        patches=np.zeros(count, dtype=np.int32),
        charts=target_charts,
        geometry_entity_ids=("patch",) * count,
        occurrence_paths=(("surface",),) * count,
    )
    with outcome:
        assert _exact_chart_partition_proof(source, target) == visits


class _FoldedEndpointCase(NamedTuple):
    domain: MeshingDomain
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_mesh: CellMesh
    target_geometry: CellGeometrySpec
    source_witness: SurfaceChartWitness
    target_witness: SurfaceChartWitness


def _folded_endpoint_case(
    restricted: bool, budget: CoordinateEnclosureBudget
) -> _FoldedEndpointCase:
    """Split two planar triangles sharing the right-angle plate interface."""
    domain = folded_plates_surface().domain
    coordinates = np.asarray(
        ((0, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1)), dtype=np.float64
    )
    source_cells = np.asarray(((2, 0, 1), (0, 3, 1)), dtype=np.int32)
    target_cells = np.asarray(
        ((2, 0, 4), (0, 1, 4), (0, 3, 5), (0, 5, 1)), dtype=np.int32
    )
    target_coordinates = np.concatenate(
        (coordinates, np.asarray(((-0.5, 0.5, 0), (0, 0.5, 0.5)), dtype=np.float64))
    )
    source_names, target_names = ("first", "second"), ("a", "b", "c", "d")
    source_mesh = CellMesh(
        coordinates,
        tuple(
            CellBlock(
                name,
                "triangle",
                cells[None],
                global_ids=np.asarray((17 + row,), dtype=np.int64),
            )
            for row, (name, cells) in enumerate(
                zip(source_names, source_cells, strict=True)
            )
        ),
    )
    target_mesh = CellMesh(
        target_coordinates,
        tuple(
            CellBlock(
                name,
                "triangle",
                cells[None],
                global_ids=np.asarray((31 + row,), dtype=np.int64),
            )
            for row, (name, cells) in enumerate(
                zip(target_names, target_cells, strict=True)
            )
        ),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh, budget=budget)
    target_geometry = CellGeometrySpec.affine(target_mesh, budget=budget)
    if restricted:
        root_id = cell_geometry_id(source_geometry)
        source_origin = CellGeometryRestrictionSource(
            root_id,
            source_mesh.topology_id,
            {
                name: np.asarray((17 + row,), dtype=np.int64)
                for row, name in enumerate(source_names)
            },
            {name: source_cells[row : row + 1] for row, name in enumerate(source_names)},
        )
        source_geometry = CellGeometrySpec(
            dict(zip(source_geometry.block_names, source_geometry.elements, strict=True)),
            dict(
                zip(
                    source_geometry.block_names,
                    source_geometry.geometry_dofs,
                    strict=True,
                )
            ),
            source_geometry.coordinates,
            restriction_source=source_origin,
            budget=budget,
        )
        references = np.asarray(
            (
                ((0, 0), (1, 0), (0, 0.5)),
                ((1, 0), (0, 1), (0, 0.5)),
                ((0, 0), (1, 0), (0.5, 0.5)),
                ((0, 0), (0.5, 0.5), (0, 1)),
            ),
            dtype=np.float64,
        )
        element = coordinate_lagrange_element("triangle", 1)
        target_geometry = CellGeometrySpec(
            {
                name: RestrictedCellGeometryElement(
                    element,
                    "triangle",
                    (references[row, 1:, :] - references[row, 0, :]).T,
                    references[row, 0, :],
                )
                for row, name in enumerate(target_names)
            },
            {
                name: source_cells[row // 2 : row // 2 + 1]
                for row, name in enumerate(target_names)
            },
            coordinates,
            restriction_source=CellGeometryRestrictionSource(
                root_id,
                source_mesh.topology_id,
                {
                    name: np.asarray((17 + row // 2,), dtype=np.int64)
                    for row, name in enumerate(target_names)
                },
                {
                    name: source_cells[row // 2 : row // 2 + 1]
                    for row, name in enumerate(target_names)
                },
            ),
            budget=budget,
        )

    def witness(
        mesh: CellMesh, geometry: CellGeometrySpec, patches: list[int]
    ) -> SurfaceChartWitness:
        charts = np.asarray(mesh.coordinates)[
            np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
        ]
        charts = np.stack(
            [
                points[:, (0, 1) if patch == 0 else (2, 1)]
                for points, patch in zip(charts, patches, strict=True)
            ]
        )
        return SurfaceChartWitness(
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            domain_id=domain.domain_id,
            cell_global_ids=np.concatenate(
                [np.asarray(block.global_ids) for block in mesh.blocks]
            ),
            patches=np.asarray(patches, dtype=np.int32),
            charts=charts,
            geometry_entity_ids=tuple(domain.entity_id(2, patch) for patch in patches),
            occurrence_paths=tuple(
                domain.source_occurrences[2][patch] for patch in patches
            ),
        )

    return _FoldedEndpointCase(
        domain,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness(source_mesh, source_geometry, [0, 1]),
        witness(target_mesh, target_geometry, [0, 0, 1, 1]),
    )


@pytest.mark.parametrize("restricted", [False, True], ids=["independent", "restricted"])
def test_retained_surface_certificates_accept_two_patch_target_contracts(
    restricted: bool,
) -> None:
    policy, limits = CellValidityPolicy(), MeshCertificateLimits()
    with coordinate_reference_budget(limits) as budget:
        case = _folded_endpoint_case(restricted, budget)
        source = certify_chart_endpoint(
            case.source_mesh,
            case.source_geometry,
            policy,
            limits,
            None,
            budget=budget,
            stale_message="stale source",
        )
        result = _certify_surface_endpoints(
            case.source_mesh,
            case.source_geometry,
            case.target_mesh,
            case.target_geometry,
            case.source_witness,
            case.target_witness,
            case.domain,
            limits,
            policy,
            (source.validity, source.embedding),
            budget,
        )
        result.target_validity.require_bound(case.target_geometry, mesh=case.target_mesh)
        result.embeddings[1].binding.require(case.target_mesh, case.target_geometry)
    assert result.source_validity is source.validity
    assert result.embeddings[0] is source.embedding
    assert result.target_validity.all_certified
    assert result.embeddings[1].status == "certified"
    # Each child has physical area 1/4, hence squared Jacobian determinant 1/4.
    assert np.all(np.asarray(result.target_validity.determinant_lower) <= 0.25)
    assert np.all(np.asarray(result.target_validity.determinant_lower) > 0.0)
    assert np.all(np.asarray(result.target_validity.determinant_upper) >= 0.25)
    if restricted:
        assert result.target_measure_routes == (
            (0, Fraction(1, 2)),
            (0, Fraction(1, 2)),
            (1, Fraction(1, 2)),
            (1, Fraction(1, 2)),
        )
    else:
        assert result.target_measure_routes is None
        assert (
            case.source_geometry.coordinates.shape
            != case.target_geometry.coordinates.shape
        )


@pytest.mark.parametrize(
    "fault",
    [
        "bound-coordinates",
        "bound-topology",
        "policy",
        "source-status",
        "source-restriction",
        "target-restriction",
        "root-geometry",
        "root-topology",
        "coordinates",
    ],
)
def test_restricted_surface_validity_rejects_invalid_source_contract(fault: str) -> None:
    policy = CellValidityPolicy()
    with coordinate_reference_budget() as budget:
        case = _folded_endpoint_case(True, budget)
        source_mesh, source_geometry = case.source_mesh, case.source_geometry
        target_geometry = case.target_geometry
        validity = certify_cell_geometry_validity(
            source_geometry, mesh=source_mesh, policy=policy, budget=budget
        )
        match fault:
            case "bound-coordinates":
                source_geometry = source_geometry.with_coordinates(
                    np.asarray(source_geometry.coordinates) + 0.25, budget=budget
                )
            case "bound-topology":
                source_mesh = CellMesh.from_triangles(
                    source_mesh.coordinates,
                    np.concatenate(
                        [np.asarray(block.vertices) for block in source_mesh.blocks]
                    ),
                    cell_global_ids=np.asarray((71, 72), dtype=np.int64),
                )
            case "policy":
                policy = CellValidityPolicy(relative_determinant_floor=1e-8)
            case "source-status":
                validity = CellValidityCertificate(
                    np.full(2, CellValidityStatus.INVALID, dtype=np.int32),
                    validity.determinant_lower,
                    validity.determinant_upper,
                    validity.depth,
                    block_names=validity.block_names,
                    block_offsets=validity.block_offsets,
                    unsupported_block_names=(),
                    unresolved_reasons=(),
                    geometry_id=validity.geometry_id,
                    geometry_layout_id=validity.geometry_layout_id,
                    topology_id=validity.topology_id,
                    policy_id=validity.policy_id,
                )
            case "source-restriction":
                source_geometry = CellGeometrySpec.affine(source_mesh, budget=budget)
                validity = certify_cell_geometry_validity(
                    source_geometry, mesh=source_mesh, policy=policy, budget=budget
                )
            case "target-restriction":
                target_geometry = CellGeometrySpec(
                    dict(
                        zip(
                            target_geometry.block_names,
                            target_geometry.elements,
                            strict=True,
                        )
                    ),
                    dict(
                        zip(
                            target_geometry.block_names,
                            target_geometry.geometry_dofs,
                            strict=True,
                        )
                    ),
                    target_geometry.coordinates,
                    budget=budget,
                )
            case "root-geometry" | "root-topology":
                origin = target_geometry.restriction_source
                assert origin is not None
                target_geometry = CellGeometrySpec(
                    dict(
                        zip(
                            target_geometry.block_names,
                            target_geometry.elements,
                            strict=True,
                        )
                    ),
                    dict(
                        zip(
                            target_geometry.block_names,
                            target_geometry.geometry_dofs,
                            strict=True,
                        )
                    ),
                    target_geometry.coordinates,
                    restriction_source=CellGeometryRestrictionSource(
                        "foreign-geometry"
                        if fault == "root-geometry"
                        else origin.source_geometry_id,
                        "foreign-topology"
                        if fault == "root-topology"
                        else origin.source_topology_id,
                        origin.block_parent_cell_ids,
                        origin.block_parent_vertex_ids,
                    ),
                    budget=budget,
                )
            case "coordinates":
                target_geometry = target_geometry.with_coordinates(
                    np.asarray(target_geometry.coordinates) + 0.25, budget=budget
                )
            case _:
                raise ValueError(
                    f"surface_validity_test.fault must select a declared source-contract violation; got {fault!r}"
                )
        with pytest.raises(ValueError):
            _renew_restricted_surface_validity(
                source_mesh,
                source_geometry,
                validity,
                case.target_mesh,
                target_geometry,
                policy,
                case.source_witness,
                case.target_witness,
                budget=budget,
            )


@pytest.mark.parametrize("fault", ["degenerate", "overlap"])
def test_retained_surface_certificates_refuse_invalid_independent_targets(
    fault: str,
) -> None:
    policy, limits = CellValidityPolicy(), MeshCertificateLimits()
    with coordinate_reference_budget(limits) as budget:
        case = _folded_endpoint_case(False, budget)
        source = certify_chart_endpoint(
            case.source_mesh,
            case.source_geometry,
            policy,
            limits,
            None,
            budget=budget,
            stale_message="stale source",
        )
        coordinates = np.asarray(case.target_mesh.coordinates).copy()
        if fault == "degenerate":
            coordinates[:, 0] = 0.0
        else:
            coordinates[:, 0] -= coordinates[:, 2]
            coordinates[:, 2] = 0.0
        target_mesh = case.target_mesh.with_coordinates(
            coordinates, numeric_version="invalid"
        )
        target_geometry = CellGeometrySpec.affine(target_mesh, budget=budget)
        witness = SurfaceChartWitness(
            geometry_id=cell_geometry_id(target_geometry),
            topology_id=target_mesh.topology_id,
            domain_id=case.domain.domain_id,
            cell_global_ids=case.target_witness.cell_global_ids,
            patches=case.target_witness.patches,
            charts=case.target_witness.charts,
            geometry_entity_ids=case.target_witness.geometry_entity_ids,
            occurrence_paths=case.target_witness.occurrence_paths,
        )
        with pytest.raises(ValueError):
            _certify_surface_endpoints(
                case.source_mesh,
                case.source_geometry,
                target_mesh,
                target_geometry,
                case.source_witness,
                witness,
                case.domain,
                limits,
                policy,
                (source.validity, source.embedding),
                budget,
            )
