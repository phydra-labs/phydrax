#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.lattice_boltzmann import (
    BGKCollisionPlan,
    D2Q9,
    LatticeBoltzmannMethodPlan,
    LatticeBoltzmannPrecisionPolicy,
    LatticeBoltzmannSmagorinskyEvidence,
    SmagorinskyCollisionPlan,
)
from phydrax.discretization.lattice_boltzmann._collision import (
    macroscopic_raw_moments,
    quadratic_equilibrium,
)
from tools.lattice_boltzmann_smagorinsky_qualification import qualification


def _nonequilibrium_state():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    density = jnp.asarray(1.0)
    velocity = jnp.asarray((0.03, -0.01))
    equilibrium = quadratic_equilibrium(density, velocity, lattice, precision)
    amplitude = 2.0e-4
    perturbation = amplitude * jnp.asarray(
        (0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0)
    )
    return lattice, precision, density, velocity, equilibrium, equilibrium + perturbation


def _collide(coefficient, populations, density, velocity, lattice, precision, rate):
    return LatticeBoltzmannMethodPlan(SmagorinskyCollisionPlan(coefficient)).collide(
        populations,
        density,
        velocity,
        jnp.zeros_like(velocity),
        jnp.asarray(rate),
        lattice,
        precision,
    )


def test_nonzero_smagorinsky_uses_local_nonequilibrium_stress_and_reports_evidence():
    lattice, precision, density, velocity, equilibrium, populations = (
        _nonequilibrium_state()
    )
    coefficient = 0.16
    rate = 1.25
    result = _collide(
        coefficient, populations, density, velocity, lattice, precision, rate
    )
    evidence = result.smagorinsky_evidence

    assert isinstance(evidence, LatticeBoltzmannSmagorinskyEvidence)
    nonequilibrium = np.asarray(populations - equilibrium)
    velocities = np.asarray(lattice.velocities)
    stress = np.einsum("q,qa,qb->ab", nonequilibrium, velocities, velocities)
    stress_norm = np.sqrt(np.sum(stress**2))
    rho = float(jnp.sum(populations))
    cs2 = float(lattice.sound_speed_squared)
    tau0 = 1.0 / rate
    expected_tau = 0.5 * (
        tau0
        + np.sqrt(
            tau0**2 + 2.0 * np.sqrt(2.0) * coefficient**2 * stress_norm / (rho * cs2**2)
        )
    )

    np.testing.assert_allclose(evidence.nonequilibrium_stress_norm, stress_norm)
    np.testing.assert_allclose(evidence.effective_relaxation_time, expected_tau)
    np.testing.assert_allclose(
        result.diagnostics.relaxation_rate_minimum, 1.0 / expected_tau
    )
    np.testing.assert_allclose(
        evidence.effective_kinematic_viscosity,
        cs2 * (expected_tau - 0.5),
    )
    assert bool(evidence.coefficient_active)
    assert bool(evidence.finite)
    assert bool(evidence.successful)
    assert bool(evidence.support_satisfied)
    assert evidence.coefficient_lower_bound == 0.0
    assert evidence.coefficient_requires_finite
    assert evidence.base_relaxation_rate_bounds == (0.0, 2.0)
    assert evidence.base_relaxation_rate_bounds_exclusive
    assert evidence.density_lower_bound == 0.0
    assert evidence.density_lower_bound_exclusive
    assert evidence.filter_width_in_lattice_units == 1.0

    old_mass, old_momentum = macroscopic_raw_moments(populations, lattice, precision)
    new_mass, new_momentum = macroscopic_raw_moments(
        result.populations, lattice, precision
    )
    np.testing.assert_allclose(new_mass, old_mass, rtol=0.0, atol=5.0e-16)
    np.testing.assert_allclose(new_momentum, old_momentum, rtol=0.0, atol=5.0e-16)
    assert float(evidence.conserved_moment_defect) <= 5.0e-16


def test_zero_coefficient_parity_equilibrium_limit_and_monotonic_response():
    lattice, precision, density, velocity, equilibrium, populations = (
        _nonequilibrium_state()
    )
    rate = 1.25
    zero = _collide(0.0, populations, density, velocity, lattice, precision, rate)
    bgk = LatticeBoltzmannMethodPlan(BGKCollisionPlan()).collide(
        populations,
        density,
        velocity,
        jnp.zeros_like(velocity),
        jnp.asarray(rate),
        lattice,
        precision,
    )
    np.testing.assert_array_equal(zero.candidate_populations, bgk.candidate_populations)
    np.testing.assert_array_equal(zero.populations, bgk.populations)
    assert zero.smagorinsky_evidence is not None
    assert not bool(zero.smagorinsky_evidence.coefficient_active)
    np.testing.assert_array_equal(
        zero.smagorinsky_evidence.effective_relaxation_time,
        jnp.asarray(1.0 / rate),
    )
    np.testing.assert_array_equal(
        zero.smagorinsky_evidence.eddy_kinematic_viscosity,
        jnp.asarray(0.0),
    )

    equilibrium_result = _collide(
        0.2, equilibrium, density, velocity, lattice, precision, rate
    )
    equilibrium_evidence = equilibrium_result.smagorinsky_evidence
    assert equilibrium_evidence is not None
    np.testing.assert_array_equal(equilibrium_result.populations, equilibrium)
    np.testing.assert_array_equal(
        equilibrium_evidence.effective_relaxation_time, jnp.asarray(1.0 / rate)
    )
    assert not bool(equilibrium_evidence.coefficient_active)

    lower = _collide(
        0.1, populations, density, velocity, lattice, precision, rate
    ).smagorinsky_evidence
    higher = _collide(
        0.2, populations, density, velocity, lattice, precision, rate
    ).smagorinsky_evidence
    assert lower is not None
    assert higher is not None
    assert float(higher.effective_relaxation_time) > float(
        lower.effective_relaxation_time
    )
    assert float(higher.effective_kinematic_viscosity) > float(
        lower.effective_kinematic_viscosity
    )


def test_smagorinsky_rejects_invalid_coefficient_and_relaxation_rate():
    for coefficient in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            SmagorinskyCollisionPlan(coefficient)

    lattice, precision, density, velocity, _, populations = _nonequilibrium_state()
    for rate in (0.0, 2.0, np.nan):
        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError),
            match=r"base relaxation rate must lie in \(0, 2\)",
        ):
            result = _collide(
                0.16, populations, density, velocity, lattice, precision, rate
            )
            jax.block_until_ready(result.populations)


def test_smagorinsky_eager_jit_and_jvp_are_consistent_and_finite():
    lattice, precision, density, velocity, _, populations = _nonequilibrium_state()
    prepared = LatticeBoltzmannMethodPlan(SmagorinskyCollisionPlan(0.16)).prepare(
        lattice, precision
    )
    rate = jnp.asarray(1.25)
    force = jnp.zeros_like(velocity)

    def collide(values):
        return prepared.collide(
            values, density, velocity, force, rate, lattice, precision
        )

    eager = collide(populations)
    compiled = eqx.filter_jit(collide)(populations)
    np.testing.assert_allclose(compiled.populations, eager.populations)
    assert compiled.smagorinsky_evidence is not None
    assert eager.smagorinsky_evidence is not None
    np.testing.assert_allclose(
        compiled.smagorinsky_evidence.effective_relaxation_time,
        eager.smagorinsky_evidence.effective_relaxation_time,
    )

    direction = jnp.asarray((0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0))
    _, tangent = jax.jvp(
        lambda values: collide(values).populations,
        (populations,),
        (direction,),
    )
    epsilon = 1.0e-6
    finite_difference = (
        collide(populations + epsilon * direction).populations
        - collide(populations - epsilon * direction).populations
    ) / (2.0 * epsilon)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert bool(jnp.any(jnp.abs(tangent) > 0.0))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-7, atol=2.0e-9)


def test_smagorinsky_decaying_shear_qualification_is_observable_and_scoped():
    record = qualification(
        resolution=12,
        steps=8,
        amplitude=0.06,
        base_relaxation_rate=1.25,
        coefficient=0.25,
    )

    assert record["case"] == "periodic-decaying-shear-d2q9"
    assert record["passed"]
    assert record["smagorinsky"]["coefficient_active"]
    assert record["smagorinsky"]["successful"]
    assert record["molecular"]["successful"]
    assert record["additional_amplitude_decay"] > 0.0
    assert record["reference"]["relative_error"] < 0.05
    assert record["smagorinsky"]["support"] == record["molecular"]["support"]
