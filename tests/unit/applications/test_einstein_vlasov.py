#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._physical import RelativityScaleContract
from phydrax.applications.numerical_relativity._boundaries import PeriodicBoundary
from phydrax.applications.numerical_relativity._derivatives import FourthOrderDerivatives
from phydrax.applications.numerical_relativity._einstein_vlasov import (
    adm_geodesic_rates,
    EinsteinVlasovConstraintSolveEvidence,
    EinsteinVlasovConstraintSolveResult,
    EinsteinVlasovMatterPlan,
    EinsteinVlasovMatterState,
    EinsteinVlasovStatus,
)
from phydrax.applications.numerical_relativity._enforcement import (
    Z4cAlgebraicEnforcement,
)
from phydrax.applications.numerical_relativity._gauge import GeodesicGauge
from phydrax.applications.numerical_relativity._grid import FixedGridGeometry
from phydrax.applications.numerical_relativity._state import flat_z4c_state
from phydrax.applications.numerical_relativity._z4c import (
    evaluate_z4c_rhs,
    z4c_adm_geometry,
    Z4cSystem,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticStressDepositPlan,
)
from phydrax.metrix import RelativityConvention
from phydrax.units import KILOGRAM


def _geodesic_payload(inverse):
    count = inverse.shape[0]
    return jnp.concatenate(
        (
            jnp.ones((count, 1)),
            jnp.zeros((count, 3)),
            inverse.reshape((count, 9)),
            jnp.zeros((count, 3 + 9 + 27)),
        ),
        axis=-1,
    )


def _fixture(*, weights=None, momenta=None, minimum_lapse=1.0e-4):
    shape = (5, 5, 5)
    fixed = FixedGridGeometry(shape, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), periodic=True)
    target = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))))
    positions = jnp.asarray(
        (
            (2.5, 2.0, 2.0),
            (1.5, 2.0, 2.0),
            (2.0, 2.5, 2.0),
            (2.0, 1.5, 2.0),
            (2.0, 2.0, 2.5),
            (2.0, 2.0, 1.5),
        )
    )
    particle_support = phx.discretization.ParticleSetPlan(
        jnp.arange(10, 16, dtype=jnp.int64),
        jnp.ones((6,)),
        ambient_dimension=3,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(target).prepare(particle_support)
    scale = RelativityScaleContract.geometric(KILOGRAM)
    convention = RelativityConvention.canonical()
    units = RelativisticUnitContract(scale, convention)
    stress = RelativisticStressDepositPlan(
        transfer,
        units,
        jnp.asarray((7,), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        topology_id=fixed.grid_id,
        mass_shell_relative_tolerance=1.0e-5,
        conservation_tolerance=1.0e-5,
        frame_momentum_relative_tolerance=1.0e-5,
    )
    system = Z4cSystem(
        scale,
        convention,
        chart_id="cartesian",
        constraint_tolerance=1.0e-3,
    )
    coordinates = jnp.moveaxis(fixed.coordinates, 0, -1)
    observer_coordinates = jnp.concatenate(
        (jnp.zeros(shape + (1,)), coordinates), axis=-1
    )

    def frame_provider(geometry, time, scale_factor):
        spacetime_coordinates = observer_coordinates.at[..., 0].set(time)
        return LocalRelativisticFramePlan.from_adm(
            geometry,
            units,
            spacetime_coordinates,
            time,
            scale_factor,
            observer_id="einstein-vlasov-eulerian-observer",
            orientation_id="right-handed-future",
        )

    z4c = flat_z4c_state(shape, grid_id=fixed.grid_id)
    geometry = z4c_adm_geometry(system, fixed, z4c, snapshot_token=0)
    frame = frame_provider(geometry, jnp.asarray(0.0), jnp.asarray(1.0))
    if momenta is None:
        momenta = 0.2 * jnp.asarray(
            (
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        )
    particle_state = stress.initialize(
        jnp.full((6,), 1.0e-30) if weights is None else weights,
        positions,
        momenta,
        jnp.full((6,), 7, dtype=jnp.int32),
        frame,
    )
    plan = EinsteinVlasovMatterPlan(
        system,
        fixed,
        FourthOrderDerivatives(shape, fixed.spacing),
        GeodesicGauge(),
        PeriodicBoundary(),
        Z4cAlgebraicEnforcement(maximum_correction=1.0),
        stress,
        frame_provider,
        frame_provider_id="flat-eulerian-frame-provider",
        time_step=0.01,
        source_tolerance=1.0e-4,
        mass_shell_tolerance=1.0e-4,
        energy_condition_tolerance=1.0e-6,
        minimum_lapse=minimum_lapse,
        maximum_consecutive_failures=2,
    )
    return plan, z4c, particle_state, frame


def _constraint_solver(*, converged=True):
    def solve(z4c, particles, stress, geometry):
        del particles, stress, geometry
        evidence = EinsteinVlasovConstraintSolveEvidence(
            0.0,
            jnp.zeros((3,)),
            0.0,
            jnp.zeros((3,)),
            1,
            converged,
            True,
            solver_id="manufactured-constraint-solve",
        )
        return EinsteinVlasovConstraintSolveResult(z4c, evidence)

    return solve


def test_minkowski_and_homogeneous_flrw_geodesic_rates():
    momentum = jnp.asarray(((0.3, -0.2, 0.1), (0.0, 0.4, 0.0)))
    mass = jnp.asarray((1.0, 2.0))
    identity = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
    energy = jnp.sqrt(mass**2 + jnp.sum(momentum**2, axis=-1))
    minkowski = jax.jit(adm_geodesic_rates)(
        momentum,
        energy,
        _geodesic_payload(identity),
        jnp.ones((2,), dtype="bool"),
    )
    np.testing.assert_allclose(minkowski.position_rate, momentum / energy[:, None])
    np.testing.assert_allclose(minkowski.covariant_momentum_rate, 0.0)
    assert bool(minkowski.successful)

    scale_factor = 2.5
    inverse_flrw = identity / scale_factor**2
    flrw_energy = jnp.sqrt(
        mass**2 + jnp.einsum("pij,pi,pj->p", inverse_flrw, momentum, momentum)
    )
    flrw = adm_geodesic_rates(
        momentum,
        flrw_energy,
        _geodesic_payload(inverse_flrw),
        jnp.ones((2,), dtype="bool"),
    )
    np.testing.assert_allclose(
        flrw.position_rate,
        momentum / (scale_factor**2 * flrw_energy[:, None]),
    )
    np.testing.assert_allclose(flrw.covariant_momentum_rate, 0.0)


def test_single_shell_source_is_spherical_and_mass_shell_admissible():
    plan, _, particles, frame = _fixture(weights=jnp.full((6,), 0.25))
    deposited = plan.stress.deposit(particles, frame)

    assert bool(deposited.successful)
    np.testing.assert_allclose(
        deposited.source_integrals.momentum_covector,
        0.0,
        atol=1.0e-7,
    )
    diagonal = jnp.diag(deposited.source_integrals.stress_covariant)
    np.testing.assert_allclose(diagonal, jnp.mean(diagonal), rtol=2.0e-6)
    np.testing.assert_allclose(
        deposited.source_integrals.stress_covariant - jnp.diag(diagonal),
        0.0,
        atol=1.0e-7,
    )
    assert bool(deposited.mass_shell_valid)


def test_manufactured_source_enters_hamiltonian_and_momentum_constraints():
    weights = jnp.full((6,), 2.0e-6)
    plan, z4c, particles, frame = _fixture(weights=weights)
    deposited = plan.stress.deposit(particles, frame)
    evaluated = evaluate_z4c_rhs(
        plan.system,
        plan.grid,
        plan.derivatives,
        plan.gauge,
        z4c,
        snapshot_token=0,
        stress_energy=deposited.projection,
    )

    np.testing.assert_allclose(
        evaluated.constraints.hamiltonian,
        -2.0 * plan.system.einstein_coupling * deposited.projection.energy_density,
        rtol=2.0e-5,
        atol=2.0e-8,
    )
    assert bool(evaluated.source_valid)
    assert float(jnp.max(jnp.abs(evaluated.constraints.momentum))) > 0.0


def test_initial_data_requires_constraint_solve_evidence():
    plan, z4c, particles, _ = _fixture()
    refused = plan.admit_initial_data(z4c, particles, _constraint_solver(converged=False))
    assert not bool(refused.evidence.admitted)
    with pytest.raises(ValueError, match="constraint solve"):
        plan.initialize(z4c, particles, _constraint_solver(converged=False))


def test_z4c_particle_step_recomputes_endpoint_source_and_commits_atomically():
    plan, z4c, particles, _ = _fixture()
    state = plan.initialize(z4c, particles, _constraint_solver())
    result = plan.advance(state)

    assert bool(result.successful)
    assert int(result.accepted.accepted_steps) == 1
    assert float(result.accepted.time) == pytest.approx(0.01)
    assert (
        result.start_stress.projection.snapshot_token
        != result.endpoint_stress.projection.snapshot_token
    )
    assert bool(
        result.endpoint_stress.projection.compatible_with(result.endpoint_geometry)
    )
    assert not np.array_equal(
        np.asarray(result.accepted.particles.positions),
        np.asarray(state.particles.positions),
    )
    assert bool(result.evidence.qualified)


def test_strong_field_refusal_rolls_back_geometry_and_particle_state():
    plan, z4c, particles, _ = _fixture(minimum_lapse=0.9)
    admitted = plan.initialize(z4c, particles, _constraint_solver())
    low_lapse = admitted.z4c.with_values(admitted.z4c.values.at[18].set(0.5))
    source = EinsteinVlasovMatterState(
        low_lapse,
        admitted.particles,
        admitted.time,
        admitted.accepted_steps,
        admitted.rejected_steps,
        admitted.consecutive_failures,
        admitted.terminal,
        runtime_id=admitted.runtime_id,
    )
    result = plan.advance(source)

    assert not bool(result.successful)
    assert int(result.status) & int(EinsteinVlasovStatus.STRONG_FIELD_REFUSED)
    np.testing.assert_array_equal(result.accepted.z4c.values, source.z4c.values)
    np.testing.assert_array_equal(
        result.accepted.particles.positions, source.particles.positions
    )
    assert int(result.accepted.rejected_steps) == 1
