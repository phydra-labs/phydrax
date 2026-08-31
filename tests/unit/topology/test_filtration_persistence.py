#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests.unit.topology._fixtures import (
    filled_triangle_filtration,
    filled_triangle_topology,
    filled_triangle_vertex_support,
)


def test_vertex_support_requires_explicit_geometric_closure():
    topology = filled_triangle_topology()
    support = filled_triangle_vertex_support(topology)
    assert support.topology_id == topology.topology_id

    with pytest.raises(ValueError, match="non-empty vertex support"):
        phx.topology.cell_vertex_support(
            topology,
            (
                np.asarray([[0], [1], [2]], dtype=np.int32),
                np.full((3, 2), -1, dtype=np.int32),
                np.asarray([[0, 1, 2]], dtype=np.int32),
            ),
        )


def test_explicit_filtration_rejects_face_monotonicity_violation():
    topology = filled_triangle_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    with pytest.raises(ValueError, match="face monotonicity"):
        phx.topology.CellFiltration(
            complex,
            (
                jnp.asarray([0.0, 2.0, 0.0]),
                jnp.asarray([1.0, 1.0, 1.0]),
                jnp.asarray([3.0]),
            ),
            source_id="invalid",
        )


def test_selected_filtration_values_must_be_finite():
    topology = filled_triangle_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    with pytest.raises(ValueError, match="finite"):
        phx.topology.CellFiltration(
            complex,
            (
                jnp.asarray([0.0, jnp.nan, 1.0]),
                jnp.asarray([0.5, 1.0, 1.0]),
                jnp.asarray([2.0]),
            ),
            source_id="nonfinite",
        )


def test_lower_and_upper_star_builders_preserve_face_order():
    topology = filled_triangle_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    support = filled_triangle_vertex_support(topology)
    lower = phx.topology.lower_star_filtration(
        complex,
        support,
        jnp.asarray([0.0, 1.0, 2.0]),
        source_id="lower",
    )
    upper = phx.topology.upper_star_filtration(
        complex,
        support,
        jnp.asarray([0.0, 1.0, 2.0]),
        source_id="upper",
    )

    np.testing.assert_allclose(lower.values[2], [2.0])
    np.testing.assert_allclose(upper.values[2], [0.0])
    assert np.all(np.diff(np.asarray(lower.canonical_order_values)) >= 0)
    assert np.all(np.diff(np.asarray(upper.canonical_order_values)) >= 0)


def test_prepared_vertex_filtration_is_jittable_and_batched():
    topology = filled_triangle_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    support = filled_triangle_vertex_support(topology)
    prepared = phx.topology.PreparedVertexFiltration(
        complex,
        support,
        direction="sublevel",
    )
    values = jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    result = eqx.filter_jit(prepared.cell_values)(values)

    assert result[0].shape == (2, 3)
    assert result[1].shape == (2, 3)
    assert result[2].shape == (2, 1)
    np.testing.assert_allclose(result[2], [[2.0], [2.0]])


def test_triangle_persistence_has_essential_component_and_finite_loop():
    _, _, filtration = filled_triangle_filtration()
    result = phx.topology.compute_persistence(
        filtration,
        coefficients=phx.topology.PrimeField(2),
        representatives="cycles",
    )
    diagram = result.diagram()
    raw = result.diagram(include_zero_length=True)

    np.testing.assert_array_equal(diagram.degrees, [0, 1])
    np.testing.assert_allclose(diagram.birth_values, [0.0, 1.0])
    np.testing.assert_allclose(diagram.death_values, [0.0, 2.0])
    np.testing.assert_array_equal(diagram.has_finite_death, [False, True])
    assert raw.interval_count == 4
    assert result.pairing.representatives is not None
    assert result.pairing.representatives.pair_count == result.pairing.pair_count


def test_induced_relative_persistence_uses_quotient_boundary():
    topology, complex, filtration = filled_triangle_filtration()
    boundary = phx.topology.CellSubcomplex.from_subsets(topology, "boundary")
    result = phx.topology.compute_persistence(
        filtration,
        coefficients=phx.topology.PrimeField(3),
        relative_to=boundary,
    )
    diagram = result.diagram()

    np.testing.assert_array_equal(diagram.degrees, [2])
    np.testing.assert_allclose(diagram.birth_values, [2.0])
    np.testing.assert_array_equal(diagram.has_finite_death, [False])
    assert result.pairing.layout_id != complex.layout.layout_id


def test_relative_subcomplex_must_share_filtration_topology():
    _, _, filtration = filled_triangle_filtration()
    other = phx.geometry.simplicial.TriangleTopology(
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        num_vertices=4,
    ).cell_complex_topology()
    other_boundary = phx.topology.CellSubcomplex.from_subsets(other, "boundary")
    with pytest.raises(ValueError, match="exact topology"):
        phx.topology.compute_persistence(
            filtration,
            coefficients=phx.topology.PrimeField(2),
            relative_to=other_boundary,
        )


def test_packed_diagram_separates_padding_and_essential_bars():
    _, _, filtration = filled_triangle_filtration()
    result = phx.topology.compute_persistence(
        filtration,
        coefficients=phx.topology.PrimeField(2),
    )
    packed = result.pack(4)

    np.testing.assert_array_equal(packed.active_mask, [True, True, False, False])
    np.testing.assert_array_equal(
        packed.has_finite_death,
        [False, True, False, False],
    )
    assert int(packed.interval_count) == 2
    with pytest.raises(phx.topology.TopologyResourceError, match="capacity"):
        result.pack(1)


def test_frozen_pairing_evaluates_batches_and_detects_full_order_change():
    _, _, filtration = filled_triangle_filtration()
    result = phx.topology.compute_persistence(
        filtration,
        coefficients=phx.topology.PrimeField(2),
    )
    frozen = phx.topology.freeze_persistence_pairing(result, filtration)
    values = tuple(jnp.stack((value, value + 0.1)) for value in filtration.values)
    evaluation = eqx.filter_jit(frozen.evaluate)(values)

    np.testing.assert_array_equal(evaluation.ordering_valid, [True, True])
    assert evaluation.birth_values.shape == (2, result.pairing.pair_count)

    changed = (
        jnp.asarray([1.5, 0.5, 1.0]),
        filtration.values[1],
        filtration.values[2],
    )
    invalid = frozen.evaluate(changed)
    assert not bool(invalid.ordering_valid)


def test_frozen_pairing_endpoint_gather_has_local_gradient():
    _, _, filtration = filled_triangle_filtration()
    result = phx.topology.compute_persistence(
        filtration,
        coefficients=phx.topology.PrimeField(2),
    )
    frozen = phx.topology.freeze_persistence_pairing(result, filtration)

    def objective(vertices):
        evaluated = frozen.evaluate(
            (vertices, filtration.values[1], filtration.values[2])
        )
        return jnp.sum(evaluated.birth_values) + jnp.sum(evaluated.death_values)

    gradient = jax.grad(objective)(filtration.values[0])
    assert gradient.shape == (3,)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0)


def test_tied_relabeling_preserves_diagram_not_cell_pairing():
    topology = filled_triangle_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    tied = phx.topology.CellFiltration(
        complex,
        (
            jnp.zeros((3,)),
            jnp.zeros((3,)),
            jnp.ones((1,)),
        ),
        source_id="ties",
    )
    diagram = phx.topology.compute_persistence(
        tied,
        coefficients=phx.topology.PrimeField(2),
    ).diagram()

    np.testing.assert_array_equal(diagram.degrees, [0, 1])
    np.testing.assert_allclose(diagram.birth_values, [0.0, 0.0])
    np.testing.assert_allclose(diagram.death_values, [0.0, 1.0])
