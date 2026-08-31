#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.lattice_boltzmann import (
    BGKCollisionPlan,
    CentralMomentCollisionPlan,
    certified_nearest_neighbor_velocity_set,
    CumulantCollisionPlan,
    D2Q9,
    D3Q27,
    EntropicCollisionPlan,
    GuoForcingPlan,
    KBCCollisionPlan,
    LatticeBoltzmannMethodPlan,
    LatticeBoltzmannPrecisionPolicy,
    MomentBasisPlan,
    MRTCollisionPlan,
    RegularizedCollisionPlan,
    RelaxationSpectrumPlan,
    SmagorinskyCollisionPlan,
)
from phydrax.discretization.lattice_boltzmann._collision import (
    macroscopic_raw_moments,
    quadratic_equilibrium,
)


def _state(lattice):
    precision = LatticeBoltzmannPrecisionPolicy()
    density = jnp.asarray([[1.0, 1.1], [0.9, 1.05]])
    velocity = jnp.broadcast_to(
        jnp.asarray((0.02,) * lattice.dimension), density.shape + (lattice.dimension,)
    )
    equilibrium = quadratic_equilibrium(density, velocity, lattice, precision)
    return precision, density, velocity, equilibrium


def test_d3q27_and_custom_lattice_certification_are_explicit():
    lattice = D3Q27()
    assert lattice.population_count == 27
    assert lattice.supports("kbc")
    custom = certified_nearest_neighbor_velocity_set(
        "permuted-d3q27",
        np.asarray(lattice.velocities)[::-1],
        np.asarray(lattice.weights)[::-1],
        tuple(26 - np.asarray(lattice.opposite)[::-1]),
    )
    assert custom.supports("athermal-hydrodynamics")
    assert custom.capability_evidence.hydrodynamic_isotropy_order == 4


def test_mrt_regularized_smagorinsky_and_moment_collisions_conserve():
    lattice = D2Q9()
    precision, density, velocity, equilibrium = _state(lattice)
    perturbation = jnp.asarray((0.0, 1.0, 1.0, -1.0, -1.0, 0.5, 0.5, -0.5, -0.5)) * 1e-5
    populations = equilibrium + perturbation
    basis = MomentBasisPlan()
    spectrum = RelaxationSpectrumPlan(default_rate=1.1)
    plans = (
        MRTCollisionPlan(basis, spectrum),
        RegularizedCollisionPlan(),
        SmagorinskyCollisionPlan(0.0),
        CentralMomentCollisionPlan(basis, spectrum),
        CumulantCollisionPlan(basis, spectrum),
    )
    old_mass, old_momentum = macroscopic_raw_moments(populations, lattice, precision)
    for collision in plans:
        method = LatticeBoltzmannMethodPlan(collision)
        result = method.collide(
            populations,
            density,
            velocity,
            jnp.zeros_like(velocity),
            jnp.asarray(1.2),
            lattice,
            precision,
        )
        assert result.successful
        new_mass, new_momentum = macroscopic_raw_moments(
            result.populations, lattice, precision
        )
        np.testing.assert_allclose(new_mass, old_mass, atol=2e-12)
        np.testing.assert_allclose(new_momentum, old_momentum, atol=2e-12)


def test_mrt_guo_and_entropic_equilibrium_paths_are_certified():
    lattice = D2Q9()
    precision, density, velocity, equilibrium = _state(lattice)
    force = jnp.broadcast_to(jnp.asarray((1e-6, -2e-6)), velocity.shape)
    mrt = LatticeBoltzmannMethodPlan(
        MRTCollisionPlan(MomentBasisPlan(), RelaxationSpectrumPlan()),
        forcing=GuoForcingPlan(),
    ).collide(
        equilibrium,
        density,
        velocity,
        force,
        jnp.asarray(1.0),
        lattice,
        precision,
    )
    assert mrt.successful
    for collision in (KBCCollisionPlan(), EntropicCollisionPlan()):
        result = LatticeBoltzmannMethodPlan(collision).collide(
            equilibrium,
            density,
            velocity,
            jnp.zeros_like(velocity),
            jnp.asarray(1.0),
            lattice,
            precision,
        )
        assert result.successful
        np.testing.assert_allclose(result.populations, equilibrium, atol=2e-12)


def test_float32_float64_and_mixed_storage_keep_explicit_population_dtype():
    lattice = D2Q9()
    density = jnp.ones((2, 2))
    velocity = jnp.zeros((2, 2, 2))
    policies = (
        LatticeBoltzmannPrecisionPolicy(population_dtype=jnp.float32),
        LatticeBoltzmannPrecisionPolicy(population_dtype=jnp.float64),
        LatticeBoltzmannPrecisionPolicy(
            population_dtype=jnp.float32,
            compute_dtype=jnp.float64,
            accumulation_dtype=jnp.float64,
            certification_dtype=jnp.float64,
            mixed_storage=True,
        ),
    )
    for precision in policies:
        equilibrium = quadratic_equilibrium(density, velocity, lattice, precision)
        result = LatticeBoltzmannMethodPlan(BGKCollisionPlan()).collide(
            equilibrium,
            density,
            velocity,
            jnp.zeros_like(velocity),
            jnp.asarray(1.0),
            lattice,
            precision,
        )

        assert result.successful
        assert equilibrium.dtype == jnp.dtype(precision.compute_dtype)
        assert result.populations.dtype == jnp.dtype(precision.population_dtype)


def test_collision_basis_and_spectrum_prepare_once_for_compiled_method():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    method = LatticeBoltzmannMethodPlan(
        MRTCollisionPlan(MomentBasisPlan(), RelaxationSpectrumPlan())
    )
    prepared = method.prepare(lattice, precision)

    assert prepared.collision.basis is not None
    assert prepared.collision.spectrum is not None
    assert prepared.collision.spectrum.basis_id == prepared.collision.basis.basis_id
    assert prepared.collision.lattice_id == lattice.lattice_id
    assert prepared.collision.precision_policy_id == precision.policy_id
