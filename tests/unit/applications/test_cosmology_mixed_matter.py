import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._coupled import (
    ComovingEulerPlan,
    CosmologicalGasParticleGravityPlan,
    CosmologicalGasParticleState,
)
from phydrax.applications.cosmology._mixed_matter import (
    WaveParticleCosmologyPlan,
    WaveParticleCosmologyState,
    WaveParticleGasCosmologyPlan,
    WaveParticleGasCosmologyState,
)
from phydrax.applications.cosmology._particle_mesh import (
    CosmologicalParticleMeshPlan,
)
from phydrax.applications.cosmology._particles import CosmologicalKDKPlan
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)


def _case(
    *,
    count=8,
    dimension=1,
    gas=False,
    schedule=(1.0, 1.0002),
    policy=None,
):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(count, periodic=True)
        for _ in range(dimension)
    )
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(axes, axis_names=names).prepare(
        jnp.asarray([[0.0] * dimension, [1.0] * dimension])
    )
    system = phx.equations.EulerSystem(dimension)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "mixed-cosmology-test",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity_owner = phx.solver.NewtonianSelfGravityPlan(0.05).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )

    coordinates = jnp.meshgrid(
        *(grid.structured_axes[axis].interval_centers for axis in range(dimension)),
        indexing="ij",
    )
    positions = jnp.stack(tuple(value.reshape((-1,)) for value in coordinates), axis=-1)
    capacity = positions.shape[0]
    particle_support = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity),
        jnp.full((capacity,), 1.0 / capacity),
        ambient_dimension=dimension,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particle_support)
    particle_gravity = phx.solver.ParticleMeshGravityPlan(gravity_owner, transfer)
    kdk = CosmologicalKDKPlan(particle_support, (1.0,) * dimension)

    half_cell = 0.5 / count
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(dimension)),
        axis_names=names,
        field_name="psi",
    ).prepare(
        tuple(
            phx.discretization.AxisDomain.periodic(half_cell, 1.0 + half_cell)
            for _ in range(dimension)
        )
    )
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    selected_policy = (
        WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
            norm_relative_tolerance=1.0e-8,
        )
        if policy is None
        else policy
    )
    wave = WaveDarkMatterPlan(
        1.0,
        jnp.asarray(schedule),
        gravitational_constant=0.05,
        reduced_planck_constant=0.05,
        step_policy=selected_policy,
    ).prepare(space, background)
    wave_state = wave.initialize(jnp.ones(space.physical_shape, dtype=jnp.complex128))
    particle_state = kdk.initialize(
        positions,
        jnp.zeros_like(positions),
        schedule[0],
    )

    result = {
        "space": space,
        "grid": grid,
        "background": background,
        "wave": wave,
        "wave_state": wave_state,
        "particle_gravity": particle_gravity,
        "kdk": kdk,
        "particle_state": particle_state,
    }
    if gas:
        gas_plan = ComovingEulerPlan(
            dynamics,
            adiabatic_index=5.0 / 3.0,
            expansion_dimension=3,
            substeps=8,
        )
        cell_average = jnp.zeros(grid.shape + (dimension + 2,))
        cell_average = cell_average.at[..., 0].set(1.0)
        cell_average = cell_average.at[..., -1].set(1.0)
        gas_state = gas_plan.initialize(cell_average, schedule[0])
        result.update(gas=gas_plan, gas_state=gas_state)
    return result


def test_uniform_mixed_density_has_one_mean_removal_and_zero_force():
    case = _case()
    prepared = WaveParticleCosmologyPlan(
        case["wave"], case["kdk"], case["particle_gravity"]
    ).prepare()
    state = WaveParticleCosmologyState(case["wave_state"], case["particle_state"])

    assembly = prepared.density.assemble(state)
    shared = prepared.gravity.solve(assembly)

    np.testing.assert_allclose(assembly.component_mass, [1.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(assembly.total_density, 2.0, atol=1e-12)
    assert shared.mean_removal_count == 1
    assert bool(shared.successful)
    assert bool(assembly.density_nonnegative)
    np.testing.assert_allclose(shared.source_integral, 0.0, atol=1e-12)
    np.testing.assert_allclose(shared.potential, 0.0, atol=1e-12)
    np.testing.assert_allclose(shared.component_force, 0.0, atol=1e-12)
    np.testing.assert_allclose(shared.total_force, 0.0, atol=1e-12)
    assert shared.particle_force_adjoint_defect < 1e-12


def test_uniform_component_limits_reproduce_wave_and_particle_owners():
    case = _case(schedule=(1.0, 1.0001, 1.0002))
    prepared = WaveParticleCosmologyPlan(
        case["wave"], case["kdk"], case["particle_gravity"]
    ).prepare()
    initial = WaveParticleCosmologyState(case["wave_state"], case["particle_state"])

    mixed = prepared.rollout(initial)
    wave_only = case["wave"].solve(case["wave_state"])
    particle_only = CosmologicalParticleMeshPlan(
        case["kdk"], case["particle_gravity"], case["wave"].scale_factors
    ).rollout(case["background"], case["particle_state"])

    assert bool(mixed.successful)
    np.testing.assert_allclose(mixed.state.wave.psi, wave_only.state.psi, atol=2e-11)
    np.testing.assert_allclose(
        mixed.state.particles.positions, particle_only.state.positions, atol=2e-12
    )
    np.testing.assert_allclose(
        mixed.state.particles.canonical_momenta,
        particle_only.state.canonical_momenta,
        atol=2e-12,
    )
    assert bool(jnp.all(mixed.diagnostics.time_level_consistent))
    assert bool(jnp.all(mixed.diagnostics.wave_norm_conserved))
    np.testing.assert_allclose(mixed.diagnostics.total_gravity_work, 0.0, atol=1e-12)


def test_uniform_wave_particle_gas_limit_reproduces_gas_particle_owner():
    case = _case(gas=True)
    prepared = WaveParticleGasCosmologyPlan(
        case["wave"], case["kdk"], case["gas"], case["particle_gravity"]
    ).prepare()
    initial = WaveParticleGasCosmologyState(
        case["wave_state"], case["particle_state"], case["gas_state"]
    )
    owner = CosmologicalGasParticleGravityPlan(
        case["gas"],
        case["kdk"],
        case["particle_gravity"],
        case["wave"].scale_factors,
    )

    mixed = prepared.rollout(initial)
    gas_particle = owner.rollout(
        case["background"],
        CosmologicalGasParticleState(case["gas_state"], case["particle_state"]),
    )

    assert bool(mixed.successful)
    assert bool(gas_particle.successful)
    np.testing.assert_allclose(
        mixed.state.gas.cell_average, gas_particle.state.gas.cell_average, rtol=2e-10
    )
    np.testing.assert_allclose(
        mixed.state.particles.positions,
        gas_particle.state.particles.positions,
        atol=2e-12,
    )
    assert bool(jnp.all(mixed.diagnostics.gas_density_positive))
    assert bool(jnp.all(mixed.diagnostics.gas_pressure_positive))
    assert bool(jnp.all(mixed.diagnostics.gas_homogeneous_successful))
    np.testing.assert_allclose(mixed.diagnostics.component_gravity_work, 0.0, atol=1e-11)


def test_rejected_mixed_interval_rolls_back_the_whole_state():
    policy = WaveDarkMatterStepPolicy(
        maximum_phase_radians=1.0e-12,
        minimum_de_broglie_cells=2.0,
        norm_relative_tolerance=1.0e-8,
    )
    case = _case(gas=True, policy=policy, schedule=(1.0, 1.001, 1.002))
    x = case["space"].axes[0].nodes
    psi = jnp.sqrt(1.0 + 0.2 * jnp.cos(2.0 * jnp.pi * x)).astype(jnp.complex128)
    wave_state = case["wave"].initialize(psi)
    initial = WaveParticleGasCosmologyState(
        wave_state, case["particle_state"], case["gas_state"]
    )
    prepared = WaveParticleGasCosmologyPlan(
        case["wave"], case["kdk"], case["gas"], case["particle_gravity"]
    ).prepare()

    result = prepared.rollout(initial)

    assert not bool(result.successful)
    assert bool(result.diagnostics.rolled_back[0])
    assert not bool(result.diagnostics.attempted[1])
    assert int(result.diagnostics.first_failed_step) == 0
    np.testing.assert_array_equal(result.state.wave.psi, initial.wave.psi)
    np.testing.assert_array_equal(
        result.state.particles.positions, initial.particles.positions
    )
    np.testing.assert_array_equal(
        result.state.particles.canonical_momenta,
        initial.particles.canonical_momenta,
    )
    np.testing.assert_array_equal(result.state.gas.cell_average, initial.gas.cell_average)
    assert result.state.wave.scale_factor == initial.wave.scale_factor
    assert result.state.particles.scale_factor == initial.particles.scale_factor
    assert result.state.gas.scale_factor == initial.gas.scale_factor


def test_mixed_plan_rejects_a_second_gravitational_constant_owner():
    case = _case()
    mismatched_owner = phx.solver.NewtonianSelfGravityPlan(0.051).prepare(
        case["particle_gravity"].gravity.transport
    )
    mismatched_gravity = phx.solver.ParticleMeshGravityPlan(
        mismatched_owner,
        case["particle_gravity"].transfer,
    )

    with pytest.raises(ValueError, match="exactly one gravitational constant"):
        WaveParticleCosmologyPlan(case["wave"], case["kdk"], mismatched_gravity)


def test_mixed_rollout_rejects_nan_component_time_levels_before_canonicalization():
    case = _case(gas=True)
    prepared = WaveParticleGasCosmologyPlan(
        case["wave"], case["kdk"], case["gas"], case["particle_gravity"]
    ).prepare()
    wave_type = type(case["wave_state"])
    particle_type = type(case["particle_state"])
    gas_type = type(case["gas_state"])
    nan = jnp.asarray(jnp.nan, dtype=case["wave_state"].scale_factor.dtype)
    invalid_states = (
        WaveParticleGasCosmologyState(
            wave_type(case["wave_state"].psi, nan),
            case["particle_state"],
            case["gas_state"],
        ),
        WaveParticleGasCosmologyState(
            case["wave_state"],
            particle_type(
                case["particle_state"].positions,
                case["particle_state"].canonical_momenta,
                nan,
            ),
            case["gas_state"],
        ),
        WaveParticleGasCosmologyState(
            case["wave_state"],
            case["particle_state"],
            gas_type(case["gas_state"].cell_average, nan),
        ),
    )

    for invalid in invalid_states:
        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="finite, positive",
        ):
            jax.block_until_ready(prepared.rollout(invalid).successful)
