from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._coordinate_enclosure import RationalPolynomial
from phydrax.discretization._nested_reference import NestedCellReferenceMap
from phydrax.discretization.fem._de_rham import FiniteElementTransferResult
from phydrax.discretization.fem._exact_form_moments import exact_reference_form_moments
from phydrax.discretization.fem._form_elements import form_element
from phydrax.geometry._common_root_volume_overlap import CurvedOverlapIntegrationPolicy
from phydrax.linalg import AbstractLinearOperator
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator
from tests._support.rational_form_transfer import make_rational_form_transfer_case


def _dense(operator: AbstractLinearOperator) -> np.ndarray:
    if not isinstance(operator, SparseCoordinateOperator):
        raise TypeError("Expected a sparse finite-element coordinate action.")
    result = np.zeros((operator.target.size, operator.source.size), dtype=np.float64)
    relation = operator.relation
    if not isinstance(relation, EdgeRelation):
        raise TypeError("Expected edge-relation sparse finite-element coordinates.")
    valid = np.asarray(relation.valid, dtype=np.bool_)
    np.add.at(
        result,
        (
            np.asarray(relation.target_indices, dtype=np.int64)[valid],
            np.asarray(relation.source_indices, dtype=np.int64)[valid],
        ),
        np.asarray(operator.coefficients, dtype=np.float64)[valid],
    )
    return result


def test_nonconstant_rational_reference_map_retains_piola_and_commuting_bounds() -> None:
    result = make_rational_form_transfer_case()
    assert isinstance(result, FiniteElementTransferResult)
    assert result.integration_policy is not None
    assert result.exact_integral_count > 0
    assert result.enclosed_integral_count > 0
    assert len(result.entry_evidence) == 2
    np.testing.assert_allclose(_dense(result.complex_map.map(0)), np.eye(2), atol=1e-12)

    top_action = _dense(result.complex_map.map(1))
    assert top_action.shape == (1, 1)
    top_error = np.asarray(result.entry_evidence[1].absolute_error_bounds)
    assert top_error.shape == (1,)
    assert top_error[0] > 0.0
    assert abs(top_action[0, 0] - 1.0) <= top_error[0] + 2.0e-15
    assert np.any(np.asarray(result.entry_evidence[1].enclosed_integral_counts) > 0)

    assert len(result.commuting_evidence) == 1
    residual = result.commuting_evidence[0]
    assert np.all(
        np.abs(np.asarray(residual.observed_defects))
        <= np.asarray(residual.integration_error_bounds)
        + np.asarray(residual.roundoff_error_bounds)
    )

    adjoint = result.complex_map.adjoint()
    random = np.random.default_rng(37)
    for degree in range(2):
        source_space = result.complex_map.source.space(degree)
        target_space = result.complex_map.target.space(degree)
        source_values = random.normal(size=source_space.size)
        target_values = random.normal(size=target_space.size)
        np.testing.assert_allclose(
            target_space.inner(
                result.complex_map.map(degree).mv(source_values), target_values
            ),
            source_space.inner(source_values, adjoint.map(degree).mv(target_values)),
            rtol=2e-10,
            atol=2e-10,
        )
    assert result.evidence_id


def test_nonconstant_rational_moment_requires_an_integration_policy() -> None:
    argument = RationalPolynomial(
        {(1,): Fraction(2)},
        {(0,): Fraction(1), (1,): Fraction(1)},
    )
    reference_map = NestedCellReferenceMap(
        20,
        10,
        "interval",
        "interval",
        ((Fraction(0),), (Fraction(1),)),
        (argument,),
    )
    source = form_element("interval", 1, 1, family="trimmed", proxy="circulation")
    target = form_element("interval", 1, 1, family="trimmed", proxy="circulation")
    with pytest.raises(ValueError, match="explicit CurvedOverlapIntegrationPolicy"):
        exact_reference_form_moments(
            source,
            target,
            reference_map,
            np.eye(1, dtype=np.float64),
            np.eye(1, dtype=np.float64),
            None,
        )


def test_rational_hdiv_form_uses_oriented_two_form_piola_moments() -> None:
    denominator: dict[tuple[int, ...], Fraction] = {
        (0, 0, 0): Fraction(1),
        (1, 0, 0): Fraction(1),
        (0, 1, 0): Fraction(1),
        (0, 0, 1): Fraction(1),
    }

    def rational_argument(coordinate: int, /) -> RationalPolynomial:
        numerator: dict[tuple[int, ...], Fraction] = {
            tuple(int(axis == coordinate) for axis in range(3)): Fraction(2)
        }
        return RationalPolynomial(numerator, denominator)

    arguments = tuple(rational_argument(coordinate) for coordinate in range(3))
    corners = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    reference_map = NestedCellReferenceMap(
        20,
        10,
        "tetrahedron",
        "tetrahedron",
        corners,
        arguments,
    )
    source = form_element(
        "tetrahedron",
        2,
        1,
        family="trimmed",
        twist="untwisted",
        proxy="flux",
    )
    target = form_element(
        "tetrahedron",
        2,
        1,
        family="trimmed",
        twist="untwisted",
        proxy="flux",
    )
    policy = CurvedOverlapIntegrationPolicy(
        relative_tolerance=1.0e-7, absolute_tolerance=1.0e-10
    )
    identity = np.eye(source.local_dof_count, dtype=np.float64)
    moments = exact_reference_form_moments(
        source,
        target,
        reference_map,
        identity,
        identity,
        policy,
    )
    np.testing.assert_array_less(
        np.abs(moments.values - np.eye(target.local_dof_count)),
        moments.errors + 2.0e-12,
    )
    assert moments.enclosed_count > 0
    assert np.all(np.isfinite(moments.errors))
