from dataclasses import replace
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
    _ReferenceMomentCache,
    exact_partitioned_reference_form_moments,
    exact_reference_form_moments,
    MomentIntegralMatrix,
)
from phydrax.discretization.fem._form_elements import form_element
from phydrax.discretization.fem._partition_form_moments import (
    PartitionedReferenceEntities,
    prepare_partitioned_reference_entities,
)
from phydrax.discretization.fem._reference import FiniteElementSpec
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
    _assert_moment_evidence_equal(direct, replay)
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
    cache = _ReferenceMomentCache()
    exact_identity = _exact_identity(element.local_dof_count)
    cached = cache.nested(
        element,
        element,
        same_chart,
        exact_identity,
        exact_identity,
        None,
        prepared_orientation=prepared,
    )
    _assert_moment_evidence_equal(cached, direct)
    with pytest.raises(ValueError):
        cache.nested(
            element,
            element,
            wrong_chart,
            exact_identity,
            exact_identity,
            None,
            prepared_orientation=prepared,
        )


def _partition_moment_fixture(
    element: FiniteElementSpec,
    *,
    cut: Fraction = Fraction(1, 2),
    ids: tuple[int, int] = (11, 12),
) -> tuple[PartitionedReferenceEntities, ...]:
    vertices = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("prism").vertices
    )
    children = (
        tuple((x, y, cut * z) for x, y, z in vertices),
        tuple((x, y, cut + (1 - cut) * z) for x, y, z in vertices),
    )
    witnesses = ReferenceCellMapWitnesses(
        np.asarray(ids, dtype=np.int64),
        np.asarray((21, 21), dtype=np.int64),
        "prism",
        "prism",
        np.asarray(children, dtype=np.float64),
        children,
    )
    assert element.form_basis is not None
    return prepare_partitioned_reference_entities(
        ReferenceCellPartition((witnesses,)), 21, element.form_basis
    )


def _exact_identity(size: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(int(i == j)) for j in range(size)) for i in range(size))


def _assert_moment_evidence_equal(
    actual: MomentIntegralMatrix, expected: MomentIntegralMatrix
) -> None:
    def assert_array(actual_array: np.ndarray, expected_array: np.ndarray) -> None:
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        assert actual_array.tobytes() == expected_array.tobytes()

    assert_array(actual.values, expected.values)
    assert_array(actual.errors, expected.errors)
    assert_array(actual.exact, expected.exact)
    assert_array(actual.enclosed, expected.enclosed)


@pytest.mark.parametrize(
    "facet", ("global-binding", "geometry", "source-orientation", "target-orientation")
)
def test_partition_moment_cache_respects_geometry_orientation_and_bindings(
    facet: str,
) -> None:
    element = form_element(
        "prism", 1, 1, family="prism-trimmed", twist="untwisted", proxy="components"
    )
    identity = _exact_identity(element.local_dof_count)
    flipped = tuple(
        tuple(-value if row == 0 else value for value in values)
        for row, values in enumerate(identity)
    )
    first, second = _partition_moment_fixture(element)
    rebound_map = replace(first.coarse_to_fine, target_cell_id=121, source_cell_id=111)
    rebound = replace(
        first, fine_cell_id=111, coarse_cell_id=121, coarse_to_fine=rebound_map
    )
    record = (
        rebound if facet == "global-binding" else second if facet == "geometry" else first
    )
    source_transform = flipped if facet == "source-orientation" else identity
    target_transform = flipped if facet == "target-orientation" else identity
    cache = _ReferenceMomentCache()
    baseline = cache.partitioned(element, element, first, identity, identity, None)
    expected = exact_partitioned_reference_form_moments(
        element, element, record, source_transform, target_transform, None
    )
    actual = cache.partitioned(
        element, element, record, source_transform, target_transform, None
    )
    _assert_moment_evidence_equal(actual, expected)
    if facet == "global-binding":
        _assert_moment_evidence_equal(baseline, actual)
    else:
        assert not np.array_equal(baseline.values, actual.values)


@pytest.mark.parametrize("retention", ("eviction", "oversized"))
def test_partition_moment_cache_retention_bounds_preserve_results(retention: str) -> None:
    element = form_element(
        "prism", 1, 1, family="prism-trimmed", twist="untwisted", proxy="components"
    )
    identity = _exact_identity(element.local_dof_count)
    records = _partition_moment_fixture(element)
    expected = tuple(
        exact_partitioned_reference_form_moments(
            element, element, record, identity, identity, None
        )
        for record in records
    )
    cache = (
        _ReferenceMomentCache(max_entries=1, max_bytes=32 << 20)
        if retention == "eviction"
        else _ReferenceMomentCache(max_entries=2, max_bytes=1)
    )
    for index in (0, 1, 0):
        result = cache.partitioned(
            element, element, records[index], identity, identity, None
        )
        _assert_moment_evidence_equal(result, expected[index])
        info = cache.table.cache_info()
        assert info.currsize <= info.max_entries
        assert info.current_bytes <= info.max_bytes
    if retention == "oversized":
        assert cache.table.cache_info().currsize == 0
    else:
        assert cache.table.cache_info().currsize == 1


def test_partition_moment_cache_does_not_mask_invalid_cell_kind() -> None:
    element = form_element(
        "prism", 1, 1, family="prism-trimmed", twist="untwisted", proxy="components"
    )
    identity = _exact_identity(element.local_dof_count)
    record = _partition_moment_fixture(element)[0]
    cache = _ReferenceMomentCache()
    cache.partitioned(element, element, record, identity, identity, None)
    invalid = replace(record, fine_cell_kind="tetrahedron")
    with pytest.raises(ValueError):
        exact_partitioned_reference_form_moments(
            element, element, invalid, identity, identity, None
        )
    with pytest.raises(ValueError):
        cache.partitioned(element, element, invalid, identity, identity, None)


def test_moment_cache_rational_map_policy_and_failures() -> None:
    argument = RationalPolynomial(
        {(1,): Fraction(2)},
        {(0,): Fraction(1), (1,): Fraction(1)},
    )
    reference_map = NestedCellReferenceMap(
        20, 10, "interval", "interval", ((Fraction(0),), (Fraction(1),)), (argument,)
    )
    source = form_element("interval", 1, 1, family="trimmed", proxy="circulation")
    transform = _exact_identity(source.local_dof_count)
    policies = (
        CurvedOverlapIntegrationPolicy(
            relative_tolerance=1.0e-5, absolute_tolerance=1.0e-8
        ),
        CurvedOverlapIntegrationPolicy(
            relative_tolerance=1.0e-7, absolute_tolerance=1.0e-10
        ),
    )
    cache = _ReferenceMomentCache()
    for policy in policies:
        expected = exact_reference_form_moments(
            source, source, reference_map, transform, transform, policy
        )
        actual = cache.nested(
            source,
            source,
            reference_map,
            transform,
            transform,
            policy,
            prepared_orientation=None,
        )
        _assert_moment_evidence_equal(actual, expected)
    with pytest.raises(ValueError):
        cache.nested(
            source,
            source,
            reference_map,
            transform,
            transform,
            None,
            prepared_orientation=None,
        )


def test_moment_cache_distinguishes_rational_chart_arguments_and_elements() -> None:
    element = form_element("interval", 0, 2, family="trimmed", proxy="scalar")
    lower_order = form_element("interval", 0, 1, family="trimmed", proxy="scalar")
    argument = RationalPolynomial(
        {(1,): Fraction(2)},
        {(0,): Fraction(1), (1,): Fraction(1)},
    )
    affine = NestedCellReferenceMap(
        20,
        10,
        "interval",
        "interval",
        ((Fraction(0),), (Fraction(1),)),
        ({(1,): Fraction(1)},),
    )
    curved = replace(affine, arguments=(argument,))
    policy = CurvedOverlapIntegrationPolicy(
        relative_tolerance=1.0e-7, absolute_tolerance=1.0e-10
    )
    cache = _ReferenceMomentCache()
    results = []
    for current_element, reference_map in (
        (element, affine),
        (element, curved),
        (lower_order, affine),
    ):
        transform = _exact_identity(current_element.local_dof_count)
        expected = exact_reference_form_moments(
            current_element, current_element, reference_map, transform, transform, policy
        )
        actual = cache.nested(
            current_element,
            current_element,
            reference_map,
            transform,
            transform,
            policy,
            prepared_orientation=None,
        )
        _assert_moment_evidence_equal(actual, expected)
        results.append(actual)
    assert not np.array_equal(results[0].values, results[1].values)
    assert results[0].values.shape != results[2].values.shape


def test_partition_moment_cache_distinguishes_owned_pieces_with_same_map() -> None:
    element = form_element("triangle", 0, 1, family="trimmed", proxy="scalar")
    basis = element.form_basis
    assert basis is not None
    zero = Fraction(0)
    one = Fraction(1)
    half = Fraction(1, 2)
    children = (
        ((zero, zero), (one, zero), (half, half)),
        ((zero, zero), (half, half), (zero, one)),
    )

    def first_child(ids: tuple[int, int]) -> PartitionedReferenceEntities:
        witnesses = ReferenceCellMapWitnesses(
            np.asarray(ids, dtype=np.int64),
            np.asarray((21, 21), dtype=np.int64),
            "triangle",
            "triangle",
            np.asarray(children, dtype=np.float64),
            children,
        )
        records = prepare_partitioned_reference_entities(
            ReferenceCellPartition((witnesses,)), 21, basis
        )
        return next(record for record in records if record.fine_cell_id == ids[0])

    owned = first_child((11, 12))
    unowned = first_child((12, 11))
    assert owned.coarse_to_fine.source_reference_vertices == (
        unowned.coarse_to_fine.source_reference_vertices
    )
    assert owned.pieces != unowned.pieces
    transform = _exact_identity(element.local_dof_count)
    cache = _ReferenceMomentCache()
    results = []
    for record in (owned, unowned):
        expected = exact_partitioned_reference_form_moments(
            element, element, record, transform, transform, None
        )
        actual = cache.partitioned(element, element, record, transform, transform, None)
        _assert_moment_evidence_equal(actual, expected)
        results.append(actual)
    assert not np.array_equal(results[0].values, results[1].values)


def test_moment_cache_distinguishes_same_size_form_bases() -> None:
    scalar = form_element("interval", 0, 1, family="trimmed", proxy="scalar")
    circulation = form_element("interval", 1, 2, family="trimmed", proxy="circulation")
    assert scalar.local_dof_count == circulation.local_dof_count
    transform = _exact_identity(scalar.local_dof_count)
    reference_map = NestedCellReferenceMap(
        20,
        10,
        "interval",
        "interval",
        ((Fraction(0),), (Fraction(1, 2),)),
        ({(1,): Fraction(1, 2)},),
    )
    cache = _ReferenceMomentCache()
    results = []
    for element in (scalar, circulation):
        expected = exact_reference_form_moments(
            element, element, reference_map, transform, transform, None
        )
        actual = cache.nested(
            element,
            element,
            reference_map,
            transform,
            transform,
            None,
            prepared_orientation=None,
        )
        _assert_moment_evidence_equal(actual, expected)
        results.append(actual)
    assert not np.array_equal(results[0].values, results[1].values)
