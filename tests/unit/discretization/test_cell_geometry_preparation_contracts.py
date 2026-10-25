#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coordinate preparation multiplicity and stored-field reconstruction contracts."""

from collections.abc import Iterator
from typing import assert_never

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    cell_geometry_preparation_bounds,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    coordinate_lagrange_element,
    coordinate_tabulation,
    CoordinateTabulator,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    prepare_barycentric_source_snapshot,
    prepare_spline_source_snapshot,
    RationalComposedCellGeometryElement,
    SplineCellGeometryElement,
    SweptCoordinateTabulator,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_validity_preparation_bounds,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)
from tests._support.coordinate_references import coordinate_reference_budget


type RestorableCoordinateSource = (
    CellGeometrySpec
    | CoordinateTabulator
    | SweptCoordinateTabulator
    | LayerColumnCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
)


@pytest.fixture(params=("spec", "lattice", "swept", "layer", "polynomial", "rational"))
def stored_source(request: pytest.FixtureRequest) -> Iterator[RestorableCoordinateSource]:
    triangle = coordinate_lagrange_element("triangle", 1)
    with coordinate_reference_budget() as budget:
        match request.param:
            case "spec":
                yield CellGeometrySpec(
                    {"cells": triangle},
                    {"cells": np.arange(3, dtype=np.int32)[None]},
                    triangle.reference_nodes,
                )
            case "lattice":
                tabulator = triangle.tabulator
                assert isinstance(tabulator, CoordinateTabulator)
                yield tabulator
            case "swept":
                yield SweptCoordinateTabulator(triangle)
            case "layer":
                yield LayerColumnCellGeometryElement(triangle, budget=budget)
            case "polynomial":
                yield PolynomialComposedCellGeometryElement(
                    triangle,
                    triangle,
                    np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.int64),
                    np.ones((3, 2), dtype=np.int64),
                    budget=budget,
                )
            case "rational":
                yield RationalComposedCellGeometryElement(
                    coordinate_lagrange_element("tetrahedron", 1),
                    coordinate_lagrange_element("pyramid", 1),
                    np.asarray(
                        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)),
                        dtype=np.int64,
                    ),
                    np.asarray(
                        ((1, 1, 1), (2, 1, 1), (2, 2, 1), (1, 2, 1), (1, 1, 1)),
                        dtype=np.int64,
                    ),
                )
            case _:
                raise ValueError(
                    "stored_source.case must name a coordinate record; got an unknown fixture case."
                )


def _restore_source(value: RestorableCoordinateSource) -> RestorableCoordinateSource:
    match value:
        case CellGeometrySpec():
            return CellGeometrySpec.from_stored_fields(value)
        case CoordinateTabulator():
            return CoordinateTabulator.from_stored_fields(value)
        case SweptCoordinateTabulator():
            return SweptCoordinateTabulator.from_stored_fields(value)
        case LayerColumnCellGeometryElement():
            return LayerColumnCellGeometryElement.from_stored_fields(value)
        case PolynomialComposedCellGeometryElement():
            return PolynomialComposedCellGeometryElement.from_stored_fields(value)
        case RationalComposedCellGeometryElement():
            return RationalComposedCellGeometryElement.from_stored_fields(value)
        case invalid:
            assert_never(invalid)


def test_coordinate_owner_reconstruction_preserves_fields_and_dynamic_leaves(
    stored_source: RestorableCoordinateSource,
) -> None:
    register_meshing_source_artifacts()
    restored = _restore_source(stored_source)
    assert type(restored) is type(stored_source)
    assert array_tree_fingerprint(restored) == array_tree_fingerprint(stored_source)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(
        stored_source
    )
    validate_meshing_source_closure(restored)


def test_rational_composed_coordinate_tabulation_preserves_chart_law() -> None:
    element = RationalComposedCellGeometryElement(
        coordinate_lagrange_element("tetrahedron", 1),
        coordinate_lagrange_element("pyramid", 1),
        np.asarray(
            ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)),
            dtype=np.int64,
        ),
        np.asarray(
            ((1, 1, 1), (2, 1, 1), (2, 2, 1), (1, 2, 1), (1, 1, 1)),
            dtype=np.int64,
        ),
    )
    points = np.asarray(((0.2, 0.3, 0.25), (0.5, 0.5, 0.4)), dtype=np.float64)
    canonical = coordinate_tabulation(element).tabulate(points)
    retained = element.tabulate(points)

    np.testing.assert_array_equal(canonical[0], retained[0])
    np.testing.assert_array_equal(canonical[1], retained[1])


def _corrupt_stored_source(
    stored_source: RestorableCoordinateSource,
) -> RestorableCoordinateSource:
    match stored_source:
        case CellGeometrySpec():
            changed = eqx.tree_at(
                lambda value: value.geometry_dofs,
                stored_source,
                (stored_source.geometry_dofs[0].at[0, 0].set(-1),),
            )
        case CoordinateTabulator():
            changed = CoordinateTabulator("triangle", 0)
        case SweptCoordinateTabulator():
            changed = eqx.tree_at(
                lambda value: value.source.reference_nodes,
                stored_source,
                stored_source.source.reference_nodes.at[0, 0].add(0.125),
            )
        case LayerColumnCellGeometryElement():
            changed = eqx.tree_at(
                lambda value: value.wall_element.reference_nodes,
                stored_source,
                stored_source.wall_element.reference_nodes.at[0, 0].add(0.125),
            )
        case (
            PolynomialComposedCellGeometryElement()
            | RationalComposedCellGeometryElement()
        ):
            changed = eqx.tree_at(
                lambda value: value.chart_coordinates,
                stored_source,
                stored_source.chart_coordinates.at[0, 0].add(0.125),
            )
        case invalid:
            assert_never(invalid)
    return changed


def test_coordinate_owner_reconstruction_refuses_corrupted_stored_fields(
    stored_source: RestorableCoordinateSource,
) -> None:
    register_meshing_source_artifacts()
    changed = _corrupt_stored_source(stored_source)
    # The perturbation is a corruption probe, not a numerical tolerance.
    with pytest.raises(ValueError, match="inputs|stored_fields|degree"):
        validate_meshing_source_closure(changed)


def test_changed_barycentric_snapshot_does_not_reuse_equal_cached_element_id() -> None:
    root = coordinate_lagrange_element("triangle", 1)
    weights = np.eye(3, dtype=np.float64)
    element = BarycentricCellGeometryElement(root, weights)
    with coordinate_reference_budget() as budget:
        snapshot = prepare_barycentric_source_snapshot(element, budget=budget)
        changed = eqx.tree_at(
            lambda source: source.barycentric_weights,
            element,
            element.barycentric_weights.at[0, 0].set(0.5),
        )
        assert changed.element_id == element.element_id
        changed_snapshot = prepare_barycentric_source_snapshot(changed, budget=budget)
        np.testing.assert_array_equal(
            changed_snapshot.barycentric_weights, np.asarray(changed.barycentric_weights)
        )
        np.testing.assert_array_equal(snapshot.barycentric_weights, weights)
        assert not snapshot.barycentric_weights.flags.writeable
        assert not changed_snapshot.barycentric_weights.flags.writeable
        assert isinstance(element.barycentric_weights, jax.Array)


def test_host_snapshot_refusal_retains_preparation_measurement() -> None:
    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1), np.eye(3, dtype=np.float64)
    )
    with coordinate_reference_budget() as budget:
        budget.maximum_memory_bytes = 1
        with pytest.raises(
            CoordinateEnclosureResourceError, match="CoordinateEnclosureBudget"
        ) as refusal:
            with budget.measurement():
                prepare_barycentric_source_snapshot(element, budget=budget)
        assert refusal.value.resource == "polynomial_storage"
        evidence = refusal.value.phase_evidence
        assert evidence is not None
        # Refusal reports completed work, not work that failed admission.
        assert evidence.work_units == budget.work_units
        assert evidence.native_charged_work_units == budget.native_charged_work_units == 0
        assert evidence.peak_bytes_upper == budget.peak_bytes_upper == 0
        assert refusal.value.completed == 0
        assert refusal.value.requested > refusal.value.limit == 1
        assert budget.retained_basis_bytes == 0


def test_geometry_preparation_bounds_retain_route_multiplicity() -> None:
    element = coordinate_lagrange_element("tetrahedron", 2)
    points = np.asarray(element.reference_nodes, dtype=np.float64)
    policy = CellValidityPolicy(maximum_subdivision_depth=0, maximum_piece_count=1)
    bounds = []
    validity_bounds = []
    for count in (1, 2, 3):
        geometry = CellGeometrySpec(
            {"cells": element},
            {
                "cells": np.tile(
                    np.arange(element.local_dof_count, dtype=np.int32), (count, 1)
                )
            },
            points,
        )
        bounds.append(cell_geometry_preparation_bounds(geometry))
        validity_bounds.append(cell_validity_preparation_bounds(geometry, policy))
        certificate = certify_cell_geometry_validity(geometry, policy=policy)
        assert certificate.all_certified
        assert certificate.status.shape == (count,)
        assert certificate.block_offsets == (0, count)
    for axis in (0, 1):
        assert bounds[2][axis] - bounds[1][axis] == bounds[1][axis] - bounds[0][axis] > 0
        assert (
            validity_bounds[2][axis] - validity_bounds[1][axis]
            == validity_bounds[1][axis] - validity_bounds[0][axis]
            > 0
        )


def test_spline_host_snapshot_preserves_execution_knots_and_weight_leaves() -> None:
    knots = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    weights = np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float64)
    element = SplineCellGeometryElement(
        knots, knots, weights, 1, 1, (1, 1), "surface", "revision"
    )
    before = tuple(jax.tree_util.tree_leaves(element))
    with coordinate_reference_budget() as budget:
        snapshot = prepare_spline_source_snapshot(element, budget=budget)
        np.testing.assert_array_equal(snapshot.u_knots, knots)
        np.testing.assert_array_equal(snapshot.v_knots, knots)
        np.testing.assert_array_equal(snapshot.weights, weights)
        assert not snapshot.u_knots.flags.writeable
        assert not snapshot.v_knots.flags.writeable
        assert not snapshot.weights.flags.writeable
    after = tuple(jax.tree_util.tree_leaves(element))
    assert len(before) == len(after) == 3
    assert all(first is second for first, second in zip(before, after, strict=True))
    assert all(isinstance(value, jax.Array) for value in after)


@pytest.fixture(
    params=(False, True), ids=("implicit-source-blocks", "authored-source-blocks")
)
def restriction_source(request: pytest.FixtureRequest) -> CellGeometryRestrictionSource:
    return CellGeometryRestrictionSource(
        "source-geometry",
        "source-topology",
        {
            "zeta": np.asarray((7, 9), dtype=np.int64),
            "alpha": np.asarray((4,), dtype=np.int64),
        },
        {
            "zeta": np.asarray(((30, 10, 20), (90, 70, 80)), dtype=np.int64),
            "alpha": np.asarray(((6, 4, 5),), dtype=np.int64),
        },
        block_source_blocks={"zeta": "root-z", "alpha": "root-a"}
        if request.param
        else None,
    )


def test_restriction_reconstruction_preserves_ordered_parent_banks_and_identity(
    restriction_source: CellGeometryRestrictionSource,
) -> None:
    register_meshing_source_artifacts()
    restored = CellGeometryRestrictionSource.from_stored_fields(restriction_source)
    assert restored.block_names == restriction_source.block_names == ("alpha", "zeta")
    assert restored.source_block_names == restriction_source.source_block_names
    assert restored.restriction_source_id == restriction_source.restriction_source_id
    assert restored.source_geometry_id == restriction_source.source_geometry_id
    assert restored.source_topology_id == restriction_source.source_topology_id
    for name in restored.block_names:
        np.testing.assert_array_equal(
            restored.block_parent_cell_ids[name],
            restriction_source.block_parent_cell_ids[name],
        )
        np.testing.assert_array_equal(
            restored.block_parent_vertex_ids[name],
            restriction_source.block_parent_vertex_ids[name],
        )
    validate_meshing_source_closure(restored)


def test_restriction_reconstruction_refuses_repeated_parent_vertex_id(
    restriction_source: CellGeometryRestrictionSource,
) -> None:
    alpha, zeta = restriction_source.parent_vertex_ids
    changed = eqx.tree_at(
        lambda value: value.parent_vertex_ids,
        restriction_source,
        (alpha.at[0, 1].set(alpha[0, 0]), zeta),
    )
    with pytest.raises(
        ValueError, match=r"CellGeometryRestrictionSource\.inputs.*distinct identities"
    ):
        CellGeometryRestrictionSource.from_stored_fields(changed)
