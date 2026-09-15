import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import nonequilibrium_field as nef


def test_closed_time_path_and_free_keldysh_identities():
    grid = nef.ClosedTimePathPlan(
        jnp.linspace(0.0, 1.0, 11), maximum_two_point_elements=10_000
    ).prepare()
    np.testing.assert_allclose(jnp.sum(grid.contour_weights), 0.0, atol=1.0e-15)
    np.testing.assert_array_equal(grid.contour_times[:11], grid.contour_times[11:][::-1])

    frequencies = jnp.asarray([1.0, 2.0])
    occupations = jnp.asarray([0.0, 1.0])
    free = nef.FreeKeldyshPlan(grid, frequencies)
    functions = jax.jit(free.evaluate)(occupations)
    evidence = free.identity_evidence(functions)
    assert bool(evidence.satisfied)
    expected_equal_time = (occupations + 0.5) / frequencies
    np.testing.assert_allclose(functions.statistical[0, 0], expected_equal_time)
    np.testing.assert_allclose(
        functions.spectral[4, 2],
        jnp.sin(0.2 * frequencies) / frequencies,
    )
    np.testing.assert_allclose(functions.retarded[jnp.triu_indices(11, 1)], 0.0, atol=0.0)


def test_finite_kadanoff_baym_memory_is_causal_and_free_energy_is_conserved():
    grid = nef.ClosedTimePathPlan(
        jnp.linspace(0.0, 1.0, 11), maximum_two_point_elements=10_000
    ).prepare()
    free = nef.KadanoffBaym2PIPlan(
        grid,
        coupling=0.0,
        memory_steps=4,
        truncation="free",
        energy_tolerance=1.0e-11,
    ).prepare(jnp.asarray([1.0, 1.7]))
    free_result = jax.jit(free.evolve)(jnp.asarray([0.0, 0.5]))
    assert bool(free_result.diagnostics.conserved)
    assert free_result.diagnostics.relative_energy_drift < 1.0e-11
    np.testing.assert_allclose(
        free_result.diagnostics.discrete_energy,
        free_result.diagnostics.discrete_energy[0],
        rtol=0.0,
        atol=1.0e-11,
    )

    interacting = nef.KadanoffBaym2PIPlan(
        grid,
        coupling=0.05,
        memory_steps=3,
        truncation="basketball",
        energy_tolerance=5.0e-3,
    ).prepare(jnp.asarray([1.0]))
    result = jax.jit(interacting.evolve)(jnp.asarray([0.0]))
    assert bool(result.diagnostics.memory.causal)
    assert bool(result.diagnostics.conserved)
    assert result.diagnostics.relative_energy_drift < 5.0e-3
    outside = ~result.self_energy.support_mask
    np.testing.assert_allclose(result.self_energy.statistical[outside], 0.0, atol=0.0)
    np.testing.assert_allclose(result.self_energy.spectral[outside], 0.0, atol=0.0)
    upper = jnp.triu(jnp.ones((11, 11), dtype=bool), 1)
    np.testing.assert_allclose(result.self_energy.retarded[upper], 0.0, atol=0.0)


def test_gauss_constrained_symplectic_yang_mills_and_ward_identity():
    plan = nef.ClassicalYangMillsPlan(
        (2, 2),
        (1.0, 1.0),
        time_step=0.05,
        step_count=8,
        gauss_tolerance=1.0e-10,
    )
    vacuum = plan.vacuum_state()
    electric = jnp.zeros_like(vacuum.electric_fields)
    electric = electric.at[..., 0, 0].set(0.2)
    electric = electric.at[..., 1, 1].set(-0.15)
    initial = nef.ClassicalYangMillsState(
        vacuum.links,
        electric,
        jnp.asarray(0, dtype=jnp.int32),
        plan.plan_id,
    )
    prepared = plan.prepare(initial)
    result = jax.jit(prepared.run)()
    assert bool(result.diagnostics.gauss_conserved)
    assert result.diagnostics.maximum_gauss_residual < 1.0e-10
    assert result.diagnostics.maximum_link_norm_residual < 1.0e-12
    assert result.diagnostics.relative_energy_drift < 1.0e-4

    angles = 0.13 * jnp.arange(4.0).reshape((2, 2))
    transformations = jnp.stack(
        (
            jnp.cos(angles),
            jnp.sin(angles),
            jnp.zeros_like(angles),
            jnp.zeros_like(angles),
        ),
        axis=-1,
    )
    ward = nef.yang_mills_ward_evidence(prepared, result.final_state, transformations)
    assert bool(ward.satisfied)
    assert ward.energy_residual < 1.0e-10
    assert ward.gauss_covariance_residual < 1.0e-10


def test_nonequilibrium_resource_and_constraint_guards():
    with pytest.raises(ValueError, match="resource budget"):
        nef.ClosedTimePathPlan(jnp.linspace(0.0, 1.0, 20), maximum_contour_points=10)
    grid = nef.ClosedTimePathPlan(jnp.linspace(0.0, 1.0, 11)).prepare()
    with pytest.raises(ValueError, match="stability or allocation bounds"):
        nef.KadanoffBaym2PIPlan(
            grid, coupling=0.0, memory_steps=3, truncation="free"
        ).prepare(jnp.asarray([100.0]))

    plan = nef.ClassicalYangMillsPlan((2, 2), (1.0, 1.0), time_step=0.05, step_count=2)
    vacuum = plan.vacuum_state()
    violating_electric = vacuum.electric_fields.at[0, 0, 0, 0].set(1.0)
    invalid = nef.ClassicalYangMillsState(
        vacuum.links,
        violating_electric,
        vacuum.step_index,
        plan.plan_id,
    )
    with pytest.raises(ValueError, match="Gauss constraint"):
        plan.prepare(invalid)
