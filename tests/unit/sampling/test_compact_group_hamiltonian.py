#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _triangle():
    return phx.discretization.polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)


def _u1_kernel(step_size=0.12):
    action = phx.operators.path_integral.CompactU1GaugeMeasure(_triangle(), beta=0.7)
    target = phx.operators.path_integral.compact_geometric_target_from_lattice_action(
        action
    )
    kernel = phx.sampling.prepare_compact_group_hamiltonian_kernel(
        target,
        step_size=step_size,
        leapfrog_steps=4,
    )
    return action, kernel


def _su2_kernel(step_size=0.08):
    topology = _triangle()
    paths = phx.discretization.prepare_cell_boundary_paths(topology)
    space = phx.graph.MatrixGaugeLinkSpace(
        topology,
        phx.metrix.SpecialUnitaryGroup(2),
    )
    action = phx.operators.path_integral.WilsonGaugeAction(
        space,
        paths,
        plaquette_couplings=0.9,
    )
    target = phx.operators.path_integral.compact_geometric_target_from_lattice_action(
        action
    )
    kernel = phx.sampling.prepare_compact_group_hamiltonian_kernel(
        target,
        step_size=step_size,
        leapfrog_steps=3,
    )
    return action, kernel


def test_flat_torus_hmc_preserves_membership_and_replays_semantic_keys():
    action, kernel = _u1_kernel()
    positions = jnp.stack(
        (
            jnp.zeros(action.configuration_shape),
            jnp.linspace(-0.2, 0.3, action.num_edges),
        )
    )
    state = phx.sampling.initialize_compact_group_hamiltonian_state(kernel, positions)
    first = phx.sampling.sample_compact_group_hamiltonian(
        kernel,
        state,
        key=jax.random.key(2),
        num_draws=8,
    )
    replay = phx.sampling.sample_compact_group_hamiltonian(
        kernel,
        state,
        key=jax.random.key(2),
        num_draws=8,
    )

    assert first.samples.shape == (2, 8, action.num_edges)
    assert jnp.array_equal(first.samples, replay.samples)
    assert jnp.all(jax.vmap(jax.vmap(action.geometry.contains))(first.samples))
    assert jnp.all(first.final_state.valid)
    assert jnp.all(jnp.isfinite(first.energy_error))


def test_su2_hmc_preserves_group_and_uses_per_group_metric():
    action, kernel = _su2_kernel()
    identity = action.link_space.identity()
    coordinates = jnp.full(action.local_coordinate_shape, 0.03)
    perturbed = action.geometry.retract(identity, coordinates)
    state = phx.sampling.initialize_compact_group_hamiltonian_state(
        kernel,
        jnp.stack((identity, perturbed)),
    )
    result = phx.sampling.sample_compact_group_hamiltonian(
        kernel,
        state,
        key=jax.random.key(5),
        num_draws=5,
    )

    assert kernel.coordinate_metric is not None
    assert kernel.coordinate_metric.gram.shape == (3, 3)
    assert result.samples.shape == (2, 5) + action.configuration_shape
    assert jnp.all(jax.vmap(jax.vmap(action.geometry.contains))(result.samples))
    assert jnp.all(result.final_state.valid)
    assert not jnp.any(result.membership_failure)


def test_compact_group_hmc_adaptation_returns_frozen_kernel():
    action, kernel = _u1_kernel(step_size=0.1)
    state = phx.sampling.initialize_compact_group_hamiltonian_state(
        kernel,
        jnp.zeros((2,) + action.configuration_shape),
    )
    adaptation = phx.sampling.adapt_compact_group_hamiltonian(
        kernel,
        state,
        phx.sampling.HamiltonianAdaptationPlan(
            warmup_steps=3,
            minimum_step_size=0.01,
            maximum_step_size=0.4,
        ),
        key=jax.random.key(7),
    )

    assert adaptation.frozen
    assert jnp.all(adaptation.valid)
    assert adaptation.step_size_history.shape == (3,)
    assert adaptation.acceptance_history.shape == (3,)
    assert 0.01 < adaptation.kernel.step_size < 0.4


def test_compact_group_hmc_rejects_wrong_measure_and_nonmembers():
    scalar = phx.operators.path_integral.Phi4LatticeAction(
        phx.discretization.StructuredCochainBridge(
            phx.discretization.TensorGridPlan(
                (phx.discretization.UniformCellAxisSpec(3, periodic=True),),
                axis_names=("x",),
            ).prepare(jnp.asarray([[0.0], [1.0]]))
        ).cochain
    )
    with pytest.raises(ValueError, match="reference measure"):
        phx.sampling.CompactGeometricTarget(
            lambda value: -scalar.action(value),
            scalar.geometry,
            configuration_shape=scalar.configuration_shape,
            local_coordinate_shape=scalar.local_coordinate_shape,
            reference_measure="lebesgue",
            target_id="scalar",
        )
    wrong_geometry = phx.sampling.CompactGeometricTarget(
        lambda value: -scalar.action(value),
        scalar.geometry,
        configuration_shape=scalar.configuration_shape,
        local_coordinate_shape=scalar.local_coordinate_shape,
        reference_measure="flat-torus",
        target_id="incorrectly-declared",
    )
    with pytest.raises(TypeError, match="supports FlatTorus"):
        phx.sampling.prepare_compact_group_hamiltonian_kernel(
            wrong_geometry,
            step_size=0.1,
        )

    u1_action, _ = _u1_kernel()
    mismatched_measure = phx.sampling.CompactGeometricTarget(
        lambda value: -u1_action.action(value),
        u1_action.geometry,
        configuration_shape=u1_action.configuration_shape,
        local_coordinate_shape=u1_action.local_coordinate_shape,
        reference_measure="product-haar",
        target_id="mismatched-u1-measure",
    )
    with pytest.raises(ValueError, match="flat-torus reference"):
        phx.sampling.prepare_compact_group_hamiltonian_kernel(
            mismatched_measure,
            step_size=0.1,
        )

    action, kernel = _su2_kernel()
    invalid = jnp.zeros((1,) + action.configuration_shape, dtype=complex)
    with pytest.raises(Exception, match="finite group members"):
        phx.sampling.initialize_compact_group_hamiltonian_state(kernel, invalid)
