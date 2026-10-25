#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Setting-invariant metrics, Cartesian realization, Niggli reduction and sublattices.

Controls: the metric degrees of freedom of each crystal family, the
setting-specific free metric entries of monoclinic unique axes and
rhombohedral axes, orthogonality of every realized operation, the uniqueness
of Niggli metrics across equivalent bases, and the number of index-n
sublattices of Z^3 (OEIS A001001).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.atomistic.crystal import (
    admitted_hall_numbers,
    enumerate_sublattice_transforms,
    metric_from_cell_parameters,
    niggli_reduce,
    SettingMetricChart,
    SpaceGroupSetting,
    standard_space_group_setting,
    sublattice_count,
)


HALL_NUMBERS = tuple(range(1, 531))
_FAMILY_DIMENSION = {
    "triclinic": 6,
    "monoclinic": 4,
    "orthorhombic": 3,
    "tetragonal": 2,
    "trigonal": 2,
    "hexagonal": 2,
    "cubic": 1,
}
# Metric coordinate order (G00, G11, G22, G12, G02, G01).
_OFF_DIAGONAL = {"a": 3, "b": 4, "c": 5}


def _generic_metric(seed: int) -> np.ndarray:
    factor = np.random.default_rng(seed).normal(size=(3, 3))
    return factor @ factor.T + 3.0 * np.eye(3)


@pytest.mark.parametrize("hall_number", HALL_NUMBERS, ids=lambda n: f"hall-{n}")
def test_projected_cell_realizes_every_operation_as_an_isometry(hall_number: int) -> None:
    setting = SpaceGroupSetting(hall_number)
    chart = SettingMetricChart(setting)
    assert chart.dimension == _FAMILY_DIMENSION[setting.crystal_system]
    assert chart.constraint_rank == 6 - chart.dimension
    coefficients = chart.coefficients(jnp.asarray(_generic_metric(hall_number)))
    result = chart.cell(coefficients)
    assert bool(result.successful)
    assert bool(result.property_evidence.positive_definite)
    assert bool(result.factorization_valid)
    vectors = np.asarray(result.vectors)
    np.testing.assert_allclose(vectors @ vectors.T, np.asarray(result.metric), atol=1e-12)
    cartesian = np.asarray(result.cartesian_rotations)
    assert cartesian.shape == (setting.operation_count, 3, 3)
    np.testing.assert_allclose(
        cartesian @ np.swapaxes(cartesian, -1, -2),
        np.broadcast_to(np.eye(3), cartesian.shape),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.det(cartesian),
        np.asarray(setting.determinants, dtype=np.float64),
        atol=1e-10,
    )
    # Cartesian action agrees with the fractional action on a generic point.
    point = np.asarray([0.21, 0.47, 0.83])
    fractional = np.asarray(setting.apply(point, dtype=np.float64))
    np.testing.assert_allclose(
        fractional @ vectors,
        (cartesian @ (vectors.T @ point)) + np.asarray(result.cartesian_translations),
        atol=1e-10,
    )


@pytest.mark.parametrize("axis", ("a", "b", "c"))
def test_monoclinic_unique_axis_frees_only_its_perpendicular_angle(axis: str) -> None:
    hall = next(
        h for h in admitted_hall_numbers(14) if SpaceGroupSetting(h).unique_axis == axis
    )
    chart = SettingMetricChart(SpaceGroupSetting(hall))
    free = {
        index for vector in chart.basis for index, value in enumerate(vector) if value
    }
    assert free == {0, 1, 2, _OFF_DIAGONAL[axis]}


def test_rhombohedral_axes_constrain_equal_edges_and_angles() -> None:
    hall = next(
        h
        for h in admitted_hall_numbers(166)
        if SpaceGroupSetting(h).rhombohedral_axes == "rhombohedral"
    )
    chart = SettingMetricChart(SpaceGroupSetting(hall))
    assert chart.dimension == 2
    metric = np.asarray(chart.metric(jnp.asarray([1.3, -0.4])))
    diagonal = np.diag(metric)
    off = metric[np.triu_indices(3, 1)]
    np.testing.assert_allclose(diagonal, diagonal[0])
    np.testing.assert_allclose(off, off[0])
    hexagonal = SettingMetricChart(standard_space_group_setting(166))
    cell = np.asarray(hexagonal.metric(jnp.asarray([2.0, 1.0])))
    # Hexagonal axes: |a| = |b| and gamma = 120 degrees, c free.
    assert cell[0, 0] == pytest.approx(cell[1, 1])
    assert cell[0, 1] / cell[0, 0] == pytest.approx(-0.5)
    assert cell[0, 2] == 0.0 and cell[1, 2] == 0.0


def test_symmetrization_is_an_idempotent_projection_onto_invariant_metrics() -> None:
    chart = SettingMetricChart(standard_space_group_setting(152))
    metric = jnp.asarray(_generic_metric(7))
    once = chart.symmetrize(metric)
    np.testing.assert_allclose(
        np.asarray(chart.symmetrize(once)), np.asarray(once), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(chart.metric(chart.coefficients(once))), np.asarray(once), atol=1e-12
    )
    rotations = standard_space_group_setting(152).rotation_array().astype(np.float64)
    residual = np.swapaxes(rotations, -1, -2) @ np.asarray(once) @ rotations - np.asarray(
        once
    )
    assert np.max(np.abs(residual)) < 1e-12


def test_cell_gradient_follows_the_exact_metric_chart() -> None:
    chart = SettingMetricChart(standard_space_group_setting(62))
    coefficients = jnp.asarray([4.0, 9.0, 16.0])

    def squared_edges(values: jax.Array) -> jax.Array:
        return jnp.sum(chart.cell(values).vectors ** 2)

    # sum(H**2) = trace(H H^T) = sum_k c_k trace(B_k).
    gradient = np.asarray(jax.grad(squared_edges)(coefficients))
    traces = [sum(vector[:3]) for vector in chart.basis]
    np.testing.assert_allclose(gradient, traces, atol=1e-10)


def test_indefinite_chart_coordinates_fail_with_evidence() -> None:
    chart = SettingMetricChart(standard_space_group_setting(225))
    result = chart.cell(jnp.asarray([-1.0]))
    assert not bool(result.successful)
    assert not bool(result.property_evidence.positive_definite)
    with pytest.raises(ValueError, match="positive"):
        chart.periodic_cell(jnp.asarray([-1.0]))


def test_periodic_cell_volume_matches_the_invariant_metric() -> None:
    chart = SettingMetricChart(standard_space_group_setting(14))
    metric = metric_from_cell_parameters(
        jnp.asarray([5.0, 7.0, 9.0]), jnp.asarray([90.0, 104.0, 90.0])
    )
    cell = chart.periodic_cell(chart.coefficients(metric))
    expected = 5.0 * 7.0 * 9.0 * np.sin(np.deg2rad(104.0))
    assert cell.volume == pytest.approx(expected, rel=1e-12)


def test_cell_parameter_metric_matches_lattice_geometry() -> None:
    hexagonal = np.asarray(
        metric_from_cell_parameters(
            jnp.asarray([2.0, 2.0, 3.0]), jnp.asarray([90.0, 90.0, 120.0])
        )
    )
    np.testing.assert_allclose(
        hexagonal, [[4.0, -2.0, 0.0], [-2.0, 4.0, 0.0], [0.0, 0.0, 9.0]], atol=1e-12
    )


def test_niggli_reduces_a_skewed_cubic_basis_to_the_unit_cube() -> None:
    skew = np.asarray([[1, 0, 0], [3, 1, 0], [-2, 5, 1]], dtype=np.float64)
    reduction = niggli_reduce(skew)
    assert reduction.successful
    assert reduction.status == "reduced"
    assert reduction.determinant == 1
    np.testing.assert_allclose(reduction.metric, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(
        reduction.vectors,
        reduction.transformation.T.astype(np.float64) @ skew,
        atol=1e-12,
    )


def test_equivalent_face_centred_bases_share_one_niggli_metric() -> None:
    primitive = 0.5 * np.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    unimodular = np.asarray([[2, 1, 0], [1, 1, 0], [3, -2, 1]], dtype=np.float64)
    first = niggli_reduce(primitive)
    second = niggli_reduce(unimodular @ primitive)
    assert first.successful and second.successful
    np.testing.assert_allclose(first.metric, second.metric, atol=1e-9)
    # Face-centred cubic: A = B = C and xi = eta = zeta = A (type I).
    metric = first.metric
    np.testing.assert_allclose(np.diag(metric), 0.5, atol=1e-12)
    np.testing.assert_allclose(2.0 * metric[np.triu_indices(3, 1)], 0.5, atol=1e-12)


def test_niggli_iteration_bound_is_reported_not_hidden() -> None:
    skew = np.asarray([[1, 0, 0], [40, 1, 0], [-25, 33, 1]], dtype=np.float64)
    reduction = niggli_reduce(skew, maximum_iterations=1)
    assert reduction.status == "iteration-limit"
    assert not reduction.successful
    with pytest.raises(ValueError, match="positive-definite"):
        niggli_reduce(np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def test_sublattice_counts_match_the_divisor_formula_sequence() -> None:
    assert [sublattice_count(n) for n in range(1, 7)] == [1, 7, 13, 35, 31, 91]


@pytest.mark.parametrize("index", (2, 3, 4, 6))
def test_sublattice_enumeration_is_complete_and_duplicate_free(index: int) -> None:
    transforms = enumerate_sublattice_transforms(index, maximum_count=200)
    assert transforms.shape == (sublattice_count(index), 3, 3)
    determinants = np.rint(np.linalg.det(transforms.astype(np.float64))).astype(np.int64)
    assert np.all(determinants == index)
    # An index-n sublattice contains n Z^3, so it is determined by its residue
    # subgroup in (Z / n)^3; distinct residue sets are distinct lattices.
    residues = {_residue_subgroup(transform, index) for transform in transforms}
    assert len(residues) == len(transforms)
    assert all(len(group) == index**2 for group in residues)


def _residue_subgroup(transform: np.ndarray, index: int) -> frozenset[tuple[int, ...]]:
    generators = [tuple(int(v) % index for v in column) for column in transform.T]
    members: set[tuple[int, ...]] = {(0, 0, 0)}
    frontier: list[tuple[int, ...]] = [(0, 0, 0)]
    while frontier:
        discovered: list[tuple[int, ...]] = []
        for member in frontier:
            for generator in generators:
                candidate = tuple(
                    (a + b) % index for a, b in zip(member, generator, strict=True)
                )
                if candidate not in members:
                    members.add(candidate)
                    discovered.append(candidate)
        frontier = discovered
    return frozenset(members)


def test_sublattice_enumeration_admits_its_count_before_allocation() -> None:
    with pytest.raises(ValueError, match="maximum_count"):
        enumerate_sublattice_transforms(6, maximum_count=90)
