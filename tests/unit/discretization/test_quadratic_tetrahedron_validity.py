#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact native P2 tetrahedron roots preserve the independent Python proof."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from phydrax import _meshcore as meshcore_owner
from phydrax.discretization import _cell_geometry_validity as validity_owner
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from tests._support.coordinate_references import coordinate_reference_budget


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_owner.meshcore_provider_installed(),
        reason="native phydrax-meshcore provider is not installed",
    ),
]


type Transform = Callable[[np.ndarray], np.ndarray]


def _geometry(transform: Transform) -> CellGeometrySpec:
    element = coordinate_lagrange_element("tetrahedron", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = np.asarray(transform(reference), dtype=np.float64)
    routes = np.arange(10, dtype=np.int32)[None, :]
    return CellGeometrySpec({"cells": element}, {"cells": routes}, coordinates)


def _identity(points: np.ndarray) -> np.ndarray:
    return points.copy()


def _inverted(points: np.ndarray) -> np.ndarray:
    return points * np.asarray((-1.0, 1.0, 1.0), dtype=np.float64)


def _actual_degree_cancellation(points: np.ndarray) -> np.ndarray:
    x, y, z = points.T
    return np.column_stack((x + 0.25 * y * z, y, z))


def _nonnegative_interior_zero(points: np.ndarray) -> np.ndarray:
    x, y, z = points.T
    return np.column_stack((x - x * x, y * (1.0 - 2.0 * x), z))


def _floor_equality(points: np.ndarray) -> np.ndarray:
    jacobian = np.asarray(
        ((1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return points @ jacobian.T


def _generic_root(
    geometry: CellGeometrySpec,
    policy: validity_owner.CellValidityPolicy,
) -> validity_owner._SubdivisionResult:
    element = geometry.elements[0]
    assert isinstance(element, FiniteElementSpec)
    route = np.asarray(geometry.geometry_dofs[0])[0]
    with coordinate_reference_budget() as budget:
        source = geometry.source_coordinates(budget=budget)
        local = tuple(source[int(index)] for index in route)
        reference = validity_owner._prepare_polynomial_reference_domain(
            "tetrahedron", 3, element.degree, False
        )
        assert reference is not None
        return validity_owner._certify_polynomial_cell(
            element, local, reference, 3, False, policy, budget
        )


def _native_arguments(
    geometry: CellGeometrySpec, policy: validity_owner.CellValidityPolicy
) -> tuple[np.ndarray, np.ndarray]:
    """Bind node order and exact derivative coefficients to the same native element."""
    with coordinate_reference_budget() as budget:
        prepared = validity_owner._quadratic_tetrahedron_native_input(
            geometry,
            geometry.elements[0],
            "tetrahedron",
            np.asarray(geometry.geometry_dofs[0]),
            np.asarray(geometry.coordinates, dtype=np.float64),
            policy,
            {},
            budget,
        )
        assert prepared is not None
        coordinates, routes, template = prepared
        return coordinates[routes], np.asarray(template, dtype=np.int32)


@pytest.mark.parametrize(
    ("name", "transform", "policy", "root_decision"),
    (
        (
            "valid",
            _identity,
            validity_owner.CellValidityPolicy(),
            meshcore_owner.QuadraticTetrahedronValidityDecision.ROOT_VALID,
        ),
        (
            "inverted",
            _inverted,
            validity_owner.CellValidityPolicy(),
            meshcore_owner.QuadraticTetrahedronValidityDecision.ROOT_INVALID,
        ),
        (
            "actual-degree-cancellation",
            _actual_degree_cancellation,
            validity_owner.CellValidityPolicy(maximum_bernstein_nodes=4),
            meshcore_owner.QuadraticTetrahedronValidityDecision.ROOT_VALID,
        ),
        (
            "node-limit-below",
            _nonnegative_interior_zero,
            validity_owner.CellValidityPolicy(
                maximum_bernstein_nodes=9, maximum_subdivision_depth=1
            ),
            meshcore_owner.QuadraticTetrahedronValidityDecision.BERNSTEIN_NODE_LIMIT,
        ),
        (
            "node-limit-boundary",
            _nonnegative_interior_zero,
            validity_owner.CellValidityPolicy(
                maximum_bernstein_nodes=10, maximum_subdivision_depth=0
            ),
            meshcore_owner.QuadraticTetrahedronValidityDecision.DEPTH_ZERO_UNRESOLVED,
        ),
        (
            "depth-zero",
            _nonnegative_interior_zero,
            validity_owner.CellValidityPolicy(maximum_subdivision_depth=0),
            meshcore_owner.QuadraticTetrahedronValidityDecision.DEPTH_ZERO_UNRESOLVED,
        ),
        (
            "depth-fallback",
            _nonnegative_interior_zero,
            validity_owner.CellValidityPolicy(maximum_subdivision_depth=1),
            meshcore_owner.QuadraticTetrahedronValidityDecision.NEEDS_SUBDIVISION,
        ),
        (
            "floor-equality",
            _floor_equality,
            validity_owner.CellValidityPolicy(
                relative_determinant_floor=0.5, maximum_subdivision_depth=0
            ),
            meshcore_owner.QuadraticTetrahedronValidityDecision.DEPTH_ZERO_UNRESOLVED,
        ),
    ),
    ids=(
        "valid",
        "invalid",
        "actual-degree-cancellation",
        "node-limit-S-minus-one",
        "node-limit-S",
        "depth-zero",
        "same-ledger-depth-fallback",
        "relative-floor-equality",
    ),
)
def test_native_root_matches_independent_exact_generic_owner(
    name: str,
    transform: Transform,
    policy: validity_owner.CellValidityPolicy,
    root_decision: meshcore_owner.QuadraticTetrahedronValidityDecision,
) -> None:
    geometry = _geometry(transform)
    generic = _generic_root(geometry, policy)
    nodes, weights = _native_arguments(geometry, policy)
    root = meshcore_owner.quadratic_tetrahedron_validity(
        nodes, weights, policy.relative_determinant_floor,
        policy.maximum_bernstein_nodes, policy.maximum_subdivision_depth,
        1 << 20, 1 << 20,
    )
    with coordinate_reference_budget() as budget:
        certificate = validity_owner.certify_cell_geometry_validity(
            geometry, policy=policy, budget=budget
        )

    assert int(root.decisions[0]) == root_decision, name
    assert int(certificate.status[0]) == int(generic.status), name
    assert int(certificate.depth[0]) == generic.depth, name
    for actual, expected in (
        (certificate.determinant_lower[0], generic.lower),
        (certificate.determinant_upper[0], generic.upper),
    ):
        if math.isnan(expected):
            assert math.isnan(float(actual)), name
        else:
            assert np.asarray(actual).view(np.uint64) == np.asarray(
                np.float64(expected)
            ).view(np.uint64), name
    assert certificate.unresolved_reasons == tuple(
        ("cells", reason) for reason in dict.fromkeys(generic.reasons)
    ), name


def test_native_work_refusal_propagates_and_settles_partial_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry(_identity)
    native = meshcore_owner.quadratic_tetrahedron_validity
    work_before = 0

    with coordinate_reference_budget() as budget:
        def refuse_at_native_cap(
            nodes: np.ndarray,
            derivative_weights: np.ndarray,
            relative_floor: float,
            maximum_bernstein_nodes: int,
            maximum_subdivision_depth: int,
            maximum_work_units: int,
            maximum_scratch_bytes: int,
            /,
        ) -> meshcore_owner.QuadraticTetrahedronValidityResult:
            nonlocal work_before
            work_before = budget.work_units
            return native(
                nodes,
                derivative_weights,
                relative_floor,
                maximum_bernstein_nodes,
                maximum_subdivision_depth,
                min(maximum_work_units, 4 * nodes.size + 1),
                maximum_scratch_bytes,
            )

        monkeypatch.setattr(
            meshcore_owner, "quadratic_tetrahedron_validity", refuse_at_native_cap
        )
        with pytest.raises(meshcore_owner.QuadraticTetrahedronValidityError) as caught:
            validity_owner.certify_cell_geometry_validity(
                geometry,
                policy=validity_owner.CellValidityPolicy(),
                budget=budget,
            )

        error = caught.value
        assert error.status is meshcore_owner.MeshcoreStatus.CAPACITY_EXCEEDED
        assert error.evidence.shape == (6,)
        assert error.evidence.dtype == np.dtype(np.int64)
        assert error.evidence[0] > 0
        assert error.evidence[1] >= 0
        assert budget.work_units == work_before + int(error.evidence[0])


def test_native_binding_reports_zero_work_limit_refusal_evidence() -> None:
    geometry = _geometry(_identity)
    nodes, weights = _native_arguments(geometry, validity_owner.CellValidityPolicy())

    with pytest.raises(meshcore_owner.QuadraticTetrahedronValidityError) as caught:
        meshcore_owner.quadratic_tetrahedron_validity(
            nodes,
            weights,
            0.0,
            20,
            0,
            0,
            1 << 20,
        )
    assert caught.value.status is meshcore_owner.MeshcoreStatus.CAPACITY_EXCEEDED
    assert caught.value.evidence.shape == (6,)
    assert caught.value.evidence.dtype == np.dtype(np.int64)
    assert caught.value.evidence[4] in (-1, 0)
    assert caught.value.evidence[5] in (1, 2, 3)
