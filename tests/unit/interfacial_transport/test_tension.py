#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix


def _k23_metric() -> np.ndarray:
    """Graph metric of K_{2,3}: a metric that is not of negative type."""
    left = np.array([1, 1, 0, 0, 0], dtype=bool)
    same = left[:, None] == left[None, :]
    values = np.where(same, 2.0, 1.0)
    np.fill_diagonal(values, 0.0)
    return values


def test_herring_matrix_is_admitted_for_energy_dissipation() -> None:
    root = np.sqrt(2.0)
    tension = InterfaceTensionMatrix(
        ("a", "b", "c"), np.array([[0.0, 1.0, 1.0], [1.0, 0.0, root], [1.0, root, 0.0]])
    )
    evidence = jax.jit(lambda matrix: matrix.admissibility())(tension)

    assert bool(evidence.admissible)
    assert bool(evidence.strict_triangle_inequality)
    np.testing.assert_allclose(evidence.triangle_excess, root - 2.0)
    assert bool(evidence.conditionally_negative_semidefinite)
    assert bool(evidence.energy_dissipation_admitted)


def test_metric_without_negative_type_is_admissible_but_not_dissipative() -> None:
    tension = InterfaceTensionMatrix(tuple("abcde"), _k23_metric())
    evidence = tension.admissibility()

    assert bool(evidence.triangle_inequality)
    assert bool(evidence.admissible)
    assert float(evidence.conditional_minimum_eigenvalue) < 0.0
    assert not bool(evidence.conditionally_negative_semidefinite)
    assert not bool(evidence.energy_dissipation_admitted)


def test_triangle_violation_is_reported_and_tracks_updated_values() -> None:
    tension = InterfaceTensionMatrix(
        ("a", "b", "c"), np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    )
    evaluate = jax.jit(lambda matrix: matrix.admissibility())
    violated = eqx.tree_at(
        lambda matrix: matrix.values,
        tension,
        jnp.asarray([[0.0, 3.0, 1.0], [3.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
    )

    assert bool(evaluate(tension).admissible)
    evidence = evaluate(violated)
    np.testing.assert_allclose(evidence.triangle_excess, 1.0)
    assert not bool(evidence.triangle_inequality)
    assert not bool(evidence.admissible)


def test_uniform_structure_gathers_pairs_without_a_dense_matrix() -> None:
    labels = tuple(f"cell{index}" for index in range(10_000))
    tension = InterfaceTensionMatrix(labels, 2.0, structure="uniform")
    evidence = tension.admissibility()

    np.testing.assert_allclose(
        tension.pair_values(jnp.array([0, 5, 9_999]), jnp.array([0, 7, 3])),
        [0.0, 2.0, 2.0],
    )
    assert bool(evidence.energy_dissipation_admitted)
    np.testing.assert_allclose(evidence.triangle_excess, -2.0)
    assert tension.label_index("cell42") == 42


def test_pairwise_gather_uses_declared_label_order() -> None:
    values = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    tension = InterfaceTensionMatrix(("gas", "oil", "water"), values)
    first = jnp.array([[0, 1], [2, 2]])
    second = jnp.array([[2, 2], [1, 2]])

    np.testing.assert_allclose(
        tension.pair_values(first, second), [[2.0, 1.5], [1.5, 0.0]]
    )
    np.testing.assert_allclose(tension.dense_values(), values)


@pytest.mark.parametrize(
    ("labels", "values", "structure"),
    [
        (("a",), 1.0, "uniform"),
        (("a", "a"), 1.0, "uniform"),
        (("a", "b"), -1.0, "uniform"),
        (("a", "b"), np.nan, "uniform"),
        (("a", "b"), np.array([[0.0, 1.0], [2.0, 0.0]]), "pairwise"),
        (("a", "b"), np.array([[1.0, 1.0], [1.0, 0.0]]), "pairwise"),
        (("a", "b"), np.array([[0.0, -1.0], [-1.0, 0.0]]), "pairwise"),
        (("a", "b", "c"), np.zeros((2, 2)), "pairwise"),
    ],
)
def test_invalid_tension_representation_is_refused(
    labels: tuple[str, ...], values: object, structure: str
) -> None:
    with pytest.raises(ValueError):
        InterfaceTensionMatrix(labels, values, structure=structure)  # ty: ignore[invalid-argument-type]


def test_mobility_requires_positive_pairs() -> None:
    mobility = InterfaceMobilityMatrix(
        ("a", "b", "c"), np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 0.5], [2.0, 0.5, 0.0]])
    )

    assert bool(mobility.admissibility().admissible)
    np.testing.assert_allclose(mobility.admissibility().minimum_pair_value, 0.5)
    with pytest.raises(ValueError, match="positive"):
        InterfaceMobilityMatrix(("a", "b"), np.array([[0.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="positive"):
        InterfaceMobilityMatrix(("a", "b"), 0.0, structure="uniform")
