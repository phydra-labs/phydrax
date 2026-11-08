from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._coordinate_enclosure import RationalPolynomial
from phydrax.discretization._nested_reference import (
    nested_cell_reference_maps,
    NestedCellReferenceMap,
)
from phydrax.discretization._nested_reference_types import (
    ReferenceCellMapWitnesses,
    ReferenceCellPartition,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._de_rham import FiniteElementTransferResult
from phydrax.discretization.fem._exact_form_moments import (
    exact_partitioned_reference_form_moments,
    exact_reference_form_moments,
)
from phydrax.discretization.fem._form_elements import form_element
from phydrax.discretization.fem._partition_form_moments import (
    prepare_partitioned_reference_entities,
)
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


@pytest.mark.parametrize("degree", range(4))
def test_prism_exact_reference_identity_includes_product_entities(degree: int) -> None:
    element = form_element(
        "prism",
        degree,
        1,
        family="prism-trimmed",
        twist="untwisted",
        proxy="components",
    )
    identity = np.eye(element.local_dof_count)
    moments = exact_reference_form_moments(
        element, element, None, identity, identity, None
    )
    np.testing.assert_allclose(moments.values, identity, atol=2e-12)
    assert np.all(moments.exact)


@pytest.mark.parametrize("degree", range(4))
def test_partitioned_prism_moments_recover_coarse_forms(degree: int) -> None:
    element = form_element(
        "prism",
        degree,
        1,
        family="prism-trimmed",
        twist="untwisted",
        proxy="components",
    )
    assert element.form_basis is not None
    vertices = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("prism").vertices
    )
    children = tuple(
        tuple((x, y, z / 2 + offset) for x, y, z in vertices)
        for offset in (Fraction(0), Fraction(1, 2))
    )
    witnesses = ReferenceCellMapWitnesses(
        np.asarray((0, 1)),
        np.asarray((41, 41)),
        "prism",
        "prism",
        np.asarray(children, dtype=float),
        children,
    )
    records = prepare_partitioned_reference_entities(
        ReferenceCellPartition((witnesses,)), 41, element.form_basis
    )
    maps = nested_cell_reference_maps(witnesses)
    identity = np.eye(element.local_dof_count)
    result = np.zeros_like(identity)
    for record, reference_map in zip(records, maps, strict=True):
        prolong = exact_reference_form_moments(
            element, element, reference_map, identity, identity, None
        )
        restrict = exact_partitioned_reference_form_moments(
            element, element, record, identity, identity, None
        )
        assert np.all(restrict.exact)
        result += restrict.values @ prolong.values
    np.testing.assert_allclose(result, identity, atol=2e-11)


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


def test_prepared_chart_moment_signature_preserves_all_entry_evidence() -> None:
    from dataclasses import replace

    from phydrax.discretization.fem._exact_form_moments import (
        _prepare_reference_chart_orientation,
    )

    element = form_element(
        "prism",
        1,
        2,
        family="prism-trimmed",
        twist="untwisted",
        proxy="circulation",
    )
    assert element.form_basis is not None
    vertices = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("prism").vertices
    )
    first = tuple((x, y, z / 2) for x, y, z in vertices)
    second = tuple((x, y, z / 2 + Fraction(1, 2)) for x, y, z in vertices)
    witnesses = ReferenceCellMapWitnesses(
        np.asarray((11, 12)),
        np.asarray((21, 21)),
        "prism",
        "prism",
        np.asarray((first, second), dtype=float),
        (first, second),
    )
    first_map, wrong_chart = nested_cell_reference_maps(witnesses)
    same_chart = replace(first_map, target_cell_id=31, source_cell_id=41)
    prepared = _prepare_reference_chart_orientation(
        first_map, 3, element.form_basis.family, {}
    )
    identity = np.eye(element.local_dof_count)
    direct = exact_reference_form_moments(
        element, element, first_map, identity, identity, None
    )
    replay = exact_reference_form_moments(
        element,
        element,
        same_chart,
        identity,
        identity,
        None,
        prepared_orientation=prepared,
    )
    for name in ("values", "errors", "exact", "enclosed"):
        np.testing.assert_array_equal(getattr(direct, name), getattr(replay, name))
    with pytest.raises(ValueError, match="different exact map"):
        exact_reference_form_moments(
            element,
            element,
            wrong_chart,
            identity,
            identity,
            None,
            prepared_orientation=prepared,
        )
