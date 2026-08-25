#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _grid(count=4):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _compiled_entropy_problem(
    *,
    entropy_pair=None,
    system=None,
    method=None,
    source=None,
    capacity=None,
):
    system_ = phx.equations.EulerSystem() if system is None else system
    grid = _grid()
    discretization = phx.discretization.FiniteVolumePlan(
        grid,
        component_names=system_.component_names,
    ).prepare()
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
    method_ = (
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        )
        if method is None
        else method
    )
    problem = phx.equations.ConservationProblemIR(
        "entropy-test",
        "state",
        system_,
        boundaries,
        source=source,
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method_,
        entropy_pair=entropy_pair,
        capacity=capacity,
    )
    return compiled, discretization, system_


def test_conservation_compiler_accepts_entropy_pair_and_fingerprints_it():
    system = phx.equations.EulerSystem()
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    without_pair, _, _ = _compiled_entropy_problem(system=system)
    with_pair, _, _ = _compiled_entropy_problem(system=system, entropy_pair=pair)

    assert with_pair.dynamics.entropy_pair is pair
    assert with_pair.dynamics.dynamics_id != without_pair.dynamics.dynamics_id
    assert with_pair.compilation_id != without_pair.compilation_id


def test_finite_volume_entropy_diagnostics_are_volume_weighted_and_source_separated():
    system = phx.equations.EulerSystem()
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    source = lambda time, state, coordinates, args: jnp.broadcast_to(
        jnp.asarray([0.1, 0.0, 0.0], dtype=state.dtype),
        state.shape,
    )
    compiled, discretization, system = _compiled_entropy_problem(
        entropy_pair=pair,
        system=system,
        source=source,
    )
    primitive = jnp.asarray([1.0, 0.4, 1.0])
    state = jnp.broadcast_to(
        system.primitive_to_conserved(primitive),
        (discretization.cell_shape[0], system.component_count),
    )

    residual, diagnostics = compiled.residual_with_diagnostics(jnp.asarray(0.0), state)
    entropy = diagnostics.entropy
    assert entropy is not None
    assert bool(entropy.admissible)
    assert jnp.allclose(
        residual,
        jnp.broadcast_to(
            jnp.asarray([0.1, 0.0, 0.0], dtype=state.dtype),
            state.shape,
        ),
    )
    assert jnp.allclose(
        entropy.total_entropy,
        jnp.sum(discretization.cell_volumes * pair.entropy(state)),
    )
    expected_source_rate = jnp.sum(
        discretization.cell_volumes
        * jnp.sum(pair.entropy_variables(state) * source(0.0, state, None, None), axis=-1)
    )
    assert entropy.semidiscrete_entropy_rate.shape == ()
    assert entropy.source_entropy_rate.shape == ()
    assert entropy.convective_entropy_rate.shape == ()
    assert jnp.allclose(entropy.semidiscrete_entropy_rate, expected_source_rate)
    assert jnp.allclose(entropy.source_entropy_rate, expected_source_rate)
    assert jnp.allclose(entropy.convective_entropy_rate, 0.0)


def test_entropy_total_uses_capacity_weighted_effective_volumes():
    system = phx.equations.EulerSystem()
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    capacity = jnp.asarray([1.0, 1.5, 2.0, 2.5])
    compiled, discretization, _ = _compiled_entropy_problem(
        entropy_pair=pair,
        system=system,
        capacity=capacity,
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray([1.0, 0.2, 1.0])),
        (discretization.cell_shape[0], system.component_count),
    )

    residual, diagnostics = compiled.residual_with_diagnostics(jnp.asarray(0.0), state)
    assert jnp.allclose(residual, 0.0)
    assert diagnostics.entropy is not None
    assert jnp.allclose(
        diagnostics.entropy.total_entropy,
        jnp.sum(discretization.cell_volumes * capacity * pair.entropy(state)),
    )


def test_entropy_pair_can_be_disabled_without_entropy_work():
    compiled, discretization, system = _compiled_entropy_problem()
    primitive = jnp.asarray([1.0, 0.0, 1.0])
    state = jnp.broadcast_to(
        system.primitive_to_conserved(primitive),
        (discretization.cell_shape[0], system.component_count),
    )
    _, diagnostics = compiled.residual_with_diagnostics(jnp.asarray(0.0), state)
    assert diagnostics.entropy is None


def test_entropy_pair_mismatch_and_viscous_combinations_fail_at_compilation():
    system = phx.equations.EulerSystem()
    pair = phx.equations.ideal_gas_euler_entropy_pair(phx.equations.EulerSystem(2))
    with pytest.raises(ValueError, match="must target the conservation problem system"):
        _compiled_entropy_problem(system=system, entropy_pair=pair)

    viscous_system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.1)
    )
    viscous_pair = phx.equations.ideal_gas_euler_entropy_pair(viscous_system)
    viscous_method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
        viscous=phx.discretization.ViscousFluxPlan(),
    )
    with pytest.raises(ValueError, match="unsupported with viscous"):
        _compiled_entropy_problem(
            system=viscous_system,
            method=viscous_method,
            entropy_pair=viscous_pair,
        )


def test_integrated_relative_entropy_uses_effective_cell_volumes():
    system = phx.equations.EulerSystem()
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    _, discretization, _ = _compiled_entropy_problem(system=system)
    left = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray([1.0, 0.0, 1.0])),
        (discretization.cell_shape[0], system.component_count),
    )
    right = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray([0.8, 0.2, 0.7])),
        left.shape,
    )
    actual = phx.discretization.integrated_finite_volume_relative_entropy(
        pair,
        left,
        right,
        discretization.cell_volumes,
    )
    expected = jnp.sum(discretization.cell_volumes * pair.relative_entropy(left, right))
    assert jnp.allclose(actual, expected)
    assert actual > 0.0
