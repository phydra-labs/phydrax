#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._observables import (
    DiffusionFitPlan,
    fit_diffusion,
    lagged_msd_vacf,
    LaggedCorrelationPlan,
    LaggedCorrelationResult,
    static_structure_factor,
    StaticStructureFactorPlan,
)
from phydrax.atomistic._soft_matter import (
    langevin_fdt_report,
    SoftMatterAtomisticProtocol,
    SoftMatterProtocolKind,
)


def test_static_structure_factor_resolves_forward_peak_and_extinction():
    positions = jnp.asarray([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
    plan = StaticStructureFactorPlan(
        [[0.0, 0.0, 0.0], [2.0 * jnp.pi, 0.0, 0.0]],
        maximum_frames=2,
        maximum_particles=2,
    )

    result = static_structure_factor(plan, positions)

    assert result.successful
    np.testing.assert_allclose(result.values, [2.0, 0.0], atol=1.0e-14)
    with pytest.raises(ValueError, match="resource bounds"):
        static_structure_factor(plan, jnp.zeros((3, 2, 3)))


def test_unwrapped_all_origin_msd_and_vacf_are_lag_resolved():
    times = jnp.asarray([0.0, 1.0, 2.0])
    positions = jnp.asarray(
        [
            [[0.9, 0.0, 0.0]],
            [[1.1, 0.0, 0.0]],
            [[1.3, 0.0, 0.0]],
        ]
    )
    velocities = jnp.asarray([[[0.2, 0.0, 0.0]]] * 3)
    plan = LaggedCorrelationPlan(
        [0, 1, 2],
        maximum_frames=3,
        maximum_particles=1,
    )

    result = lagged_msd_vacf(plan, times, positions, velocities)

    assert result.successful
    np.testing.assert_allclose(result.mean_squared_displacement, [0.0, 0.04, 0.16])
    np.testing.assert_allclose(result.velocity_autocorrelation, 0.04)
    np.testing.assert_array_equal(result.origin_counts, [3, 2, 1])


def test_diffusion_fit_retains_einstein_and_green_kubo_evidence():
    diffusion = 0.2
    lag_times = jnp.arange(5.0)
    msd = 6.0 * diffusion * lag_times
    vacf = jnp.asarray([6.0 * diffusion, 0.0, 0.0, 0.0, 0.0])
    correlation = LaggedCorrelationResult(
        jnp.arange(5, dtype=jnp.int32),
        lag_times,
        msd,
        vacf,
        jnp.asarray([20, 19, 18, 17, 16], dtype=jnp.int32),
        jnp.asarray([20, 19, 18, 17, 16], dtype=jnp.int32),
        jnp.asarray(True),
        "synthetic-diffusion-correlation",
    )
    plan = DiffusionFitPlan(
        1,
        4,
        minimum_origins=8,
        minimum_r_squared=0.999,
        maximum_relative_standard_error=1.0e-10,
        maximum_einstein_green_kubo_relative_error=1.0e-12,
    )

    evidence = fit_diffusion(plan, correlation)

    assert evidence.successful
    np.testing.assert_allclose(evidence.diffusion_coefficient, diffusion, atol=1.0e-14)
    np.testing.assert_allclose(evidence.green_kubo_diffusion, diffusion, atol=1.0e-14)
    assert evidence.einstein_green_kubo_relative_error <= 1.0e-12


def _colloid_dynamics(*, shift_energy_at_cutoff=True):
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 10.0)
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20],
        [0, 0],
        [1.0, 2.0],
        units,
        atom_type_ids=[0, 0],
        element_mask=[False, False],
        molecule_ids=[0, 1],
        cell=cell,
    ).prepare()
    cutoff = 2.0 ** (1.0 / 6.0)
    potential = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential(
                [1.0],
                [1.0],
                cutoff,
                shift_energy_at_cutoff=shift_energy_at_cutoff,
            )
        ]
    ).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.BAOABLangevinPlan(0.005, 1.0, realization_id=7),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="nvt",
        temperature=2.0,
    ).prepare(dynamics)
    return dynamics, thermodynamic


def test_langevin_colloid_protocol_binds_existing_runtime_and_fdt_identity():
    dynamics, thermodynamic = _colloid_dynamics()
    protocol = SoftMatterAtomisticProtocol(
        dynamics,
        SoftMatterProtocolKind.LANGEVIN_COLLOID,
        production_steps=100,
        maximum_particles=2,
    )

    evidence = langevin_fdt_report(protocol, thermodynamic)

    assert evidence.successful
    np.testing.assert_allclose(evidence.identity_residual, 0.0, atol=0.0)
    assert evidence.noise_addressing == "stable-particle-id/step/operator/realization"
    with pytest.raises(ValueError, match="particle capacity"):
        SoftMatterAtomisticProtocol(
            dynamics,
            SoftMatterProtocolKind.LANGEVIN_COLLOID,
            production_steps=100,
            maximum_particles=1,
        )
    unshifted, _ = _colloid_dynamics(shift_energy_at_cutoff=False)
    with pytest.raises(ValueError, match="WCA"):
        SoftMatterAtomisticProtocol(
            unshifted,
            SoftMatterProtocolKind.LANGEVIN_COLLOID,
            production_steps=100,
            maximum_particles=2,
        )
