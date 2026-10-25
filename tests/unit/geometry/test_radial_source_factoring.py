# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Keep exact affine/proxy factoring separate from radial proof verdicts."""

import sys
from collections.abc import Mapping
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import (
    CellGeometryElement,
    coordinate_lagrange_element,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import (
    CoordinateSourceBank,
    ExactScalar,
    Polynomial,
)
from phydrax.geometry import _radial_source_fidelity as owner, MeshCertificateLimits
from tests._support.coordinate_references import coordinate_reference_budget


def _inputs(curved: bool) -> tuple[CellGeometryElement, CoordinateSourceBank]:
    element = coordinate_lagrange_element("triangle", 2 if curved else 1)
    local = tuple(
        (
            Fraction(float(u)), Fraction(float(v)),
            Fraction(1) + (Fraction(float(u)) * Fraction(float(v)) / 8 if curved else 0),
        )
        for u, v in np.asarray(element.reference_nodes)
    )
    return element, local


def _prepare(
    element: CellGeometryElement,
    local: CoordinateSourceBank,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, np.ndarray]:
    findings = []
    result = owner._prepare_radial_cell(
        element, local, np.asarray((0, 1, 2), dtype=np.int64), 17,
        (np.zeros(3), 1.0, 1.0), 1.0, MeshCertificateLimits(),
        findings, owner._RadialProgress(), budget=budget,
    )
    assert result is not None
    assert findings == []
    assert np.isfinite(result.bound) and 0 <= result.bound <= 1.0
    return result.bound, result.corners


@pytest.mark.parametrize("curved,products", [(False, 3), (True, 6)])
def test_radial_reuses_only_coefficientwise_identical_proxy(
    monkeypatch: pytest.MonkeyPatch, curved: bool, products: int
) -> None:
    element, local = _inputs(curved)
    multiply = owner.algebra.multiply
    calls = 0

    def counted(
        first: Mapping[tuple[int, ...], ExactScalar],
        second: Mapping[tuple[int, ...], ExactScalar],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> Polynomial:
        nonlocal calls
        frame = sys._getframe(1)
        if (
            frame.f_code.co_name == "<genexpr>"
            and frame.f_back is not None
            and frame.f_back.f_code.co_name == "_prepare_radial_cell"
        ):
            calls += 1
        return multiply(first, second, budget=budget)

    monkeypatch.setattr(owner.algebra, "multiply", counted)
    with coordinate_reference_budget() as budget:
        _prepare(element, local, budget)
    # The quadratic bump vanishes at every corner, but must not borrow the
    # affine proxy's polynomial merely because its corner samples match.
    assert calls == products


@pytest.mark.parametrize("curved", [False, True])
def test_radial_factoring_observed_exact_work_boundary(curved: bool) -> None:
    element, local = _inputs(curved)
    with coordinate_reference_budget() as observed:
        bound, corners = _prepare(element, local, observed)
        required = observed.required_work_units
    assert required > 0
    with coordinate_reference_budget() as admitted:
        with admitted.bound_stage(required, 10**8):
            actual_bound, actual_corners = _prepare(element, local, admitted)
    assert actual_bound == bound
    np.testing.assert_array_equal(actual_corners, corners)
    with coordinate_reference_budget() as refused:
        with pytest.raises(CoordinateEnclosureResourceError):
            with refused.bound_stage(required - 1, 10**8):
                _prepare(element, local, refused)
