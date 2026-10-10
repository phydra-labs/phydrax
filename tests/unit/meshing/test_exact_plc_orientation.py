#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Validity and global embedding of exact PLC source points, not their carrier.

The constrained vertex lies on the tilted PLC edge ``A -> B`` (inside the plane
``2 z = x + y``) at the decimal parameter ``t = 0.1``. Its exact source point is
``S = (2 t, t, 3 t / 2)``; ``3 t / 2`` is a binary64 tie, so the carrier
``P = RNE(S)`` sits ``2**-56`` above ``S`` in ``z`` and above the plane. The
oracles below use 200-digit decimal arithmetic on the binary64 inputs, which is
exact for these dyadic values and independent of the rational implementation.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx


_T = 0.1
_A = (0.0, 0.0, 0.0)
_B = (2.0, 1.0, 1.5)
_C = (0.0, 2.0, 1.0)
_S = tuple(
    Fraction(a) + Fraction(_T) * (Fraction(b) - Fraction(a))
    for a, b in zip(_A, _B, strict=True)
)
_P = tuple(float(value) for value in _S)


def _decimal_source() -> tuple[Decimal, ...]:
    with localcontext() as context:
        context.prec = 200
        return tuple(
            Decimal(a) + Decimal(_T) * (Decimal(b) - Decimal(a))
            for a, b in zip(_A, _B, strict=True)
        )


def _decimal_determinant(corners: tuple[tuple[Decimal, ...], ...]) -> Decimal:
    with localcontext() as context:
        context.prec = 200
        origin = corners[0]
        u, v, w = (
            tuple(point[axis] - origin[axis] for axis in range(3))
            for point in corners[1:]
        )
        return (
            u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
        )


def _decimal(point: tuple[float, ...]) -> tuple[Decimal, ...]:
    return tuple(Decimal(value) for value in point)


def _plc(
    points: tuple[tuple[float, ...], ...],
    cells: tuple[tuple[int, ...], ...],
    constrained: int,
    maximum_bits: int = 4096,
) -> tuple[Any, Any]:
    count = len(points)
    strata = np.zeros((count,), dtype=np.int64)
    rows = np.full((count,), -1, dtype=np.int64)
    parameters = np.zeros((count, 2), dtype=np.float64)
    strata[constrained], rows[constrained], parameters[constrained, 0] = 1, 0, _T
    source = phx.discretization.ExactPlcCellGeometrySource(
        np.asarray((_A, _B), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        strata,
        rows,
        parameters,
        domain_source_id="tilted-plc-edge",
        maximum_bits=maximum_bits,
        domain_source_revision="tilted-plc-edge-authored",
        source_triangle_ids=np.empty((0,), dtype=np.int64),
        source_triangle_bounds=np.empty((0,), dtype=np.float64),
        source_segment_ids=np.asarray([0], dtype=np.int64),
        source_segment_bounds=np.asarray([1e-15], dtype=np.float64),
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int64)
    )
    return mesh, phx.discretization.CellGeometrySpec.plc(mesh, source)


def _checks(certificate: Any, status: str) -> set[str]:
    return {value.check for value in certificate.findings if value.status == status}


def _single_tetrahedron() -> tuple[Any, Any]:
    return _plc(
        (_P, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), ((0, 1, 2, 3),), 0
    )


def _hidden_contact(maximum_bits: int = 4096) -> tuple[Any, Any]:
    # T1 = (A, C, B, D) lies below the plane with face ABC in it; T2 = (S, E, F, G)
    # lies above it and touches edge AB only at the exact source point S.
    points = (
        _A,
        _B,
        _C,
        (1.0, 1.0, 0.0),
        _P,
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 1.0),
        (0.0, 0.5, 1.0),
    )
    return _plc(points, ((0, 2, 1, 3), (4, 5, 6, 7)), 4, maximum_bits)


def test_source_point_is_the_authority_of_its_rounded_carrier() -> None:
    mesh, geometry = _single_tetrahedron()
    vertex = geometry.source_coordinates()[0]

    assert vertex == _S
    assert vertex != tuple(Fraction(value) for value in _P)
    assert tuple(float(value) for value in vertex) == _P
    assert np.array_equal(np.asarray(mesh.coordinates)[0], np.asarray(_P))


def test_source_tetrahedron_measure_is_enclosed_and_certified() -> None:
    mesh, geometry = _single_tetrahedron()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    carrier = phx.discretization.certify_cell_geometry_validity(mesh)
    determinant = _decimal_determinant(
        (
            _decimal_source(),
            *(
                _decimal(point)
                for point in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            ),
        )
    )

    assert validity.all_certified
    assert (
        Decimal(float(validity.determinant_lower[0]))
        <= determinant
        <= Decimal(float(validity.determinant_upper[0]))
    )
    assert validity.geometry_id != carrier.geometry_id

    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    assert {"exact_plc_source_coordinates", "boundary_contact", "exterior_degree"} <= set(
        embedding.evaluated_checks
    )
    assert embedding.source_expression_work_units > 0


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(0.99, "CERTIFIED_VALID"), (1.01, "INVALID")],
    ids=["below-floor-margin", "above-floor-margin"],
)
def test_relative_floor_scales_the_source_jacobian(factor: float, expected: str) -> None:
    mesh, geometry = _single_tetrahedron()
    corners = (
        _decimal_source(),
        *(
            _decimal(point)
            for point in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ),
    )
    with localcontext() as context:
        context.prec = 200
        norms = Decimal(1)
        for point in corners[1:]:
            norms *= sum(
                ((point[axis] - corners[0][axis]) ** 2 for axis in range(3)), Decimal(0)
            ).sqrt()
        relative = _decimal_determinant(corners) / norms
    policy = phx.discretization.CellValidityPolicy(
        relative_determinant_floor=float(relative * Decimal(factor))
    )

    validity = phx.discretization.certify_cell_geometry_validity(
        geometry, mesh=mesh, policy=policy
    )

    assert phx.discretization.CellValidityStatus(int(validity.status[0])).name == expected


def test_carrier_positive_but_source_degenerate_tetrahedron_is_refused() -> None:
    mesh, geometry = _plc((_A, _B, _C, _P), ((0, 1, 2, 3),), 3)
    corners = (_decimal(_A), _decimal(_B), _decimal(_C))
    # Oracle premise: the rounded carrier is positive, the source is coplanar.
    assert _decimal_determinant((*corners, _decimal(_P))) > 0
    assert _decimal_determinant((*corners, _decimal_source())) == 0
    policy = phx.discretization.CellValidityPolicy(relative_determinant_floor=0.0)

    carrier = phx.discretization.certify_cell_geometry_validity(
        phx.discretization.CellGeometrySpec.affine(mesh), mesh=mesh, policy=policy
    )
    validity = phx.discretization.certify_cell_geometry_validity(
        geometry, mesh=mesh, policy=policy
    )
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert carrier.all_certified
    assert int(validity.status[0]) == phx.discretization.CellValidityStatus.INVALID
    assert embedding.status == "violated"
    assert "invalid_cell" in _checks(embedding, "violated")


def test_source_contact_hidden_by_carrier_rounding_is_a_violation() -> None:
    mesh, geometry = _hidden_contact()
    carrier_geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    carrier = phx.geometry.certify_global_embedding(
        mesh,
        carrier_geometry,
        phx.discretization.certify_cell_geometry_validity(carrier_geometry, mesh=mesh),
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)

    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert carrier.status == "certified"
    assert validity.all_certified
    assert embedding.status == "violated"
    assert "boundary_contact" in _checks(embedding, "violated")


@pytest.mark.parametrize(
    "limits",
    [{"maximum_work_units": 1}, {"maximum_scratch_bytes": 1}],
    ids=["work", "scratch"],
)
def test_exhausted_request_ledger_is_unresolved_not_a_violation(
    limits: dict[str, int],
) -> None:
    mesh, geometry = _hidden_contact()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)

    embedding = phx.geometry.certify_global_embedding(
        mesh, geometry, validity, limits=phx.geometry.MeshCertificateLimits(**limits)
    )

    assert embedding.status == "unresolved"
    assert _checks(embedding, "unresolved") == {"exact_source_resource_budget"}
    assert not _checks(embedding, "violated")


def test_source_integer_bit_budget_bounds_the_scaled_bank() -> None:
    # Every source value fits 57 bits (denominator 2**56 of 3 t / 2); scaling
    # by D = 2**56 needs 58 bits for the vertex coordinate 2, so a 57-bit source
    # budget refuses the scaled bank before forming it.
    mesh, geometry = _hidden_contact(maximum_bits=57)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)

    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert embedding.status == "unresolved"
    assert _checks(embedding, "unresolved") == {"exact_source_bit_budget"}
