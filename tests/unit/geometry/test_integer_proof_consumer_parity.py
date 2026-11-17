from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np
import pytest

from phydrax.discretization import (
    _cell_geometry_validity as validity,
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_geometry import PerCellRestrictedCellGeometryElement
from phydrax.discretization._coordinate_enclosure import (
    Expression,
    linear_combinations,
    RationalPolynomial,
)
from phydrax.discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from phydrax.discretization.fem._reference import (
    lagrange_element,
    lagrange_reference_node_data,
)
from phydrax.geometry import _embedding_cells, _mapped_embedding
from phydrax.meshing._curving import _exact_coordinate_bank_source_map


class _GenericBasis:
    """The pre-optimization exact expression combination route."""

    def __init__(self, basis: Sequence[Expression], /) -> None:
        self.basis = basis

    def combine(
        self, coefficients: Sequence[Sequence[ExactScalar]], /
    ) -> tuple[Expression, ...]:
        return linear_combinations(self.basis, coefficients)


def _triangle_geometry() -> tuple[CellMesh, CellGeometrySpec]:
    element = lagrange_element("triangle", 2)
    root = RadicalElement.sqrt(2)
    rows: list[tuple[tuple[ExactScalar, ...], ...]] = []
    for index, factor in enumerate((Fraction(-1), -root, Fraction(1))):
        local = []
        for node in lagrange_reference_node_data(element):
            assert node.exact_point is not None
            u, v = node.exact_point
            local.append((factor * (u + u * u / 4) + 4 * index, v + v * v / 4))
        rows.append(tuple(local))
    vertices = np.asarray(
        [
            [[4 * i, 0], [4 * i + float(factor) * 1.25, 0], [4 * i, 1.25]]
            for i, factor in enumerate((Fraction(-1), -root, Fraction(1)))
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    mesh = CellMesh(
        vertices, (CellBlock("cells", "triangle", np.arange(9).reshape(-1, 3)),)
    )
    bank = tuple(point for row in rows for point in row)
    execution = np.asarray([[float(x) for x in point] for point in bank])
    source = _exact_coordinate_bank_source_map(
        mesh, "proof-parity", "proof-parity-r1", ("cells",), bank, execution
    )
    routes = np.arange(len(bank)).reshape(len(rows), -1)
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": routes},
        execution,
        source_map=source,
        source_mesh=mesh,
    )
    return mesh, geometry


def _interval_geometry(
    indices: tuple[int, ...], *, rational_chart: bool = True
) -> CellGeometrySpec:
    element = lagrange_element("interval", 2)
    chart: tuple[Expression, ...] = (
        RationalPolynomial({(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(1, 2)})
        if rational_chart
        else {(1,): Fraction(1)},
    )
    restricted = PerCellRestrictedCellGeometryElement(
        element,
        "interval",
        tuple(10 + i for i in indices),
        tuple(20 + i for i in indices),
        (chart,) * len(indices),
    )
    root = RadicalElement.sqrt(2)
    factors: tuple[ExactScalar, ...] = (Fraction(1), root, Fraction(-1))
    rows = []
    for index in indices:
        for node in lagrange_reference_node_data(element):
            assert node.exact_point is not None
            u = node.exact_point[0]
            rows.append((factors[index] * (u + u * u / 4),))
    bank = tuple(rows)
    execution = np.asarray([[float(point[0])] for point in bank])
    routes = np.arange(len(bank)).reshape(len(indices), -1)
    endpoint = Fraction(7, 9) if rational_chart else Fraction(5, 4)
    vertices = np.asarray(
        [[[0.0], [float(factors[i] * endpoint)]] for i in indices], dtype=np.float64
    ).reshape(-1, 1)
    mesh = CellMesh(
        vertices,
        (CellBlock("cells", "interval", np.arange(2 * len(indices)).reshape(-1, 2)),),
    )
    source = _exact_coordinate_bank_source_map(
        mesh, "validity-parity", "validity-parity-r1", ("cells",), bank, execution
    )
    return CellGeometrySpec(
        {"cells": restricted},
        {"cells": routes},
        execution,
        source_map=source,
        source_mesh=mesh,
    )


def _assert_certificate_rows_equal(
    actual: validity.CellValidityCertificate,
    expected: validity.CellValidityCertificate,
) -> None:
    for name in (
        "status",
        "source_status",
        "execution_status",
        "determinant_lower",
        "determinant_upper",
        "source_determinant_lower",
        "source_determinant_upper",
        "depth",
        "source_depth",
    ):
        left, right = (
            np.asarray(getattr(actual, name)),
            np.asarray(getattr(expected, name)),
        )
        if left.dtype == np.float64:
            left, right = left.view(np.uint64), right.view(np.uint64)
        np.testing.assert_array_equal(left, right)
    assert actual.cell_map_digests == expected.cell_map_digests


def test_mapped_injectivity_witnesses_match_generic_and_batch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, geometry = _triangle_geometry()
    cells = _embedding_cells.prepare_embedding_cells(mesh, geometry)
    actual = _mapped_embedding.prove_mapped_cells_injective(cells)
    for witness in actual:
        assert isinstance(witness, _mapped_embedding.MappedInjectivityWitness)
        assert witness.principal_minor_controls
    first = actual[0]
    assert isinstance(first, _mapped_embedding.MappedInjectivityWitness)
    assert first.projection_matrix[0][0] < 0
    with monkeypatch.context() as reference:
        reference.setattr(
            _mapped_embedding, "PreparedLinearCombinationBasis", _GenericBasis
        )
        reference.setattr(
            _embedding_cells, "PreparedLinearCombinationBasis", _GenericBasis
        )
        generic_cells = _embedding_cells.prepare_embedding_cells(mesh, geometry)
        expected = _mapped_embedding.prove_mapped_cells_injective(generic_cells)
    assert actual == expected
    assert actual == tuple(
        _mapped_embedding.prove_mapped_cell_injective(cell) for cell in cells
    )
    assert _mapped_embedding.prove_mapped_cells_injective(cells[::-1]) == actual[::-1]


@pytest.mark.parametrize("rational_chart", (False, True))
def test_restricted_validity_certificate_matches_generic_and_batch_rows(
    monkeypatch: pytest.MonkeyPatch,
    rational_chart: bool,
) -> None:
    if not rational_chart:
        # Exercise the independently composed per-chart proof when root
        # composition is unavailable, without changing its exact inputs.
        def no_root(
            self: validity.RestrictedCoordinateMapPreparation, row: int, /
        ) -> None:
            return None

        monkeypatch.setattr(
            validity.RestrictedCoordinateMapPreparation, "expressions", no_root
        )
    geometry = _interval_geometry((0, 1, 2), rational_chart=rational_chart)
    actual = certify_cell_geometry_validity(geometry)
    with monkeypatch.context() as reference:
        reference.setattr(validity, "PreparedLinearCombinationBasis", _GenericBasis)
        expected = certify_cell_geometry_validity(geometry)
    _assert_certificate_rows_equal(actual, expected)
    assert actual.certificate_id == expected.certificate_id
    assert tuple(np.asarray(actual.status)) == (
        CellValidityStatus.CERTIFIED_VALID,
        CellValidityStatus.CERTIFIED_VALID,
        CellValidityStatus.INVALID,
    )
    singles = tuple(
        certify_cell_geometry_validity(
            _interval_geometry((i,), rational_chart=rational_chart)
        )
        for i in range(3)
    )
    for name in (
        "status",
        "source_status",
        "execution_status",
        "determinant_lower",
        "determinant_upper",
        "source_determinant_lower",
        "source_determinant_upper",
        "depth",
        "source_depth",
    ):
        combined = np.asarray(getattr(actual, name))
        separate = np.concatenate([np.asarray(getattr(item, name)) for item in singles])
        if combined.dtype == np.float64:
            combined, separate = combined.view(np.uint64), separate.view(np.uint64)
        np.testing.assert_array_equal(combined, separate)
    batch_digests = np.frombuffer(actual.cell_map_digests, dtype=np.uint8).reshape(
        -1, 2, 16
    )
    single_digests = np.concatenate(
        [
            np.frombuffer(item.cell_map_digests, dtype=np.uint8).reshape(-1, 2, 16)
            for item in singles
        ]
    )
    np.testing.assert_array_equal(batch_digests[:, 0], single_digests[:, 0])
    np.testing.assert_array_equal(batch_digests[1, 1], single_digests[1, 1])
    reversed_certificate = certify_cell_geometry_validity(
        _interval_geometry((2, 1, 0), rational_chart=rational_chart)
    )
    for name in (
        "status",
        "source_status",
        "execution_status",
        "determinant_lower",
        "determinant_upper",
        "source_determinant_lower",
        "source_determinant_upper",
        "depth",
        "source_depth",
    ):
        left = np.asarray(getattr(reversed_certificate, name))
        right = np.asarray(getattr(actual, name))[::-1]
        if left.dtype == np.float64:
            left, right = left.view(np.uint64), right.view(np.uint64)
        np.testing.assert_array_equal(left, right)
    reversed_digests = np.frombuffer(
        reversed_certificate.cell_map_digests, dtype=np.uint8
    ).reshape(-1, 2, 16)
    np.testing.assert_array_equal(reversed_digests, batch_digests[::-1])
