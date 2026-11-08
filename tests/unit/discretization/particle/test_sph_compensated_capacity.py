#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._numerics._compensated import compensated_sum
from phydrax.discretization.particle._sph_operators import sph_morris_viscous_force
from phydrax.discretization.particle._stabilization import (
    MolteniColagrossiDensityDiffusionPlan,
    MonaghanArtificialViscosityPlan,
    sph_artificial_viscosity_force,
    sph_density_diffusion_rate,
)


# Padded capacities (particle or pair slots) of one physical problem with
# `count` logical entries; each is compared against the unpadded capacity.
_CASES = (
    (8, 12, False),
    (8, 12, True),
    (60, 70, False),
    (60, 128, False),
    (60, 70, True),
    (60, 128, True),
)
# Speed at which the Monaghan pair power has an odd significand below 4/3, so
# a power and its half meet exactly at a rounding tie.
_ARTIFICIAL_SPEED = 1.0390625


def _stream(count: int) -> np.ndarray:
    # 60 terms sum sequentially to exactly 1.0 but to 1 + 2**-52 when zero
    # padding past 64 slots makes the unmasked sum blocked.
    return np.asarray([1.0, 2.0**-53] + [2.0**-108] * (count - 2), dtype=np.float64)


def _slots(count: int, capacity: int, interleave: bool) -> np.ndarray:
    if not interleave:
        return np.arange(count)
    return np.round(np.linspace(0.0, capacity - 1.0, count)).astype(np.int64)


def _scatter(
    values: np.ndarray, slots: np.ndarray, capacity: int, fill: float
) -> np.ndarray:
    output = np.full((capacity,) + values.shape[1:], fill, dtype=np.float64)
    output[slots] = values
    return output


def _bits(*values: Any) -> tuple[bytes, ...]:
    return tuple(np.asarray(value).tobytes() for value in values)


def _star(
    count: int, capacity: int, interleave: bool
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Pairs from `count` leaves to one hub, padded to `capacity` pair slots.

    Particle slot `s` of a leaf matches its pair slot; the hub is the last of
    `capacity + 1` particle slots. Padding pairs are invalid.
    """
    slots = _slots(count, capacity, interleave)
    hub = capacity
    left = np.zeros((capacity,), dtype=np.int32)
    right = np.zeros((capacity,), dtype=np.int32)
    valid = np.zeros((capacity,), dtype=np.bool_)
    left[slots] = slots
    right[slots] = hub
    valid[slots] = True
    relation = phx.sparse.EdgeRelation(
        jnp.asarray(left),
        jnp.asarray(right),
        source_size=capacity + 1,
        target_size=capacity + 1,
        valid=jnp.asarray(valid),
    )
    pairs = phx.discretization.particle.ParticlePairRelation(
        relation,
        jnp.asarray(left, dtype=jnp.int64),
        jnp.asarray(right, dtype=jnp.int64),
        source_support_id="star-particles",
        target_support_id="star-particles",
        same_set=True,
        unordered=True,
        relation_schema_id=f"star-{capacity}",
    )
    return pairs, slots, valid


def _star_vectors(
    leaf: np.ndarray, hub: np.ndarray, slots: np.ndarray, capacity: int
) -> np.ndarray:
    output = np.zeros((capacity + 1,) + leaf.shape[1:], dtype=np.float64)
    output[slots] = leaf
    output[capacity] = hub
    return output


def _ledger(count: int, capacity: int, interleave: bool, dimension: int) -> Any:
    pairs, slots, _ = _star(count, capacity, interleave)
    unit = np.zeros((dimension,), dtype=np.float64)
    first = unit.copy()
    first[0] = 1.0
    second = unit.copy()
    second[1] = 1.0
    position = _star_vectors(np.tile(first, (count, 1)), unit, slots, capacity)
    velocity = _star_vectors(np.tile(second, (count, 1)), unit, slots, capacity)
    exchange = np.full((capacity, dimension), 7.0, dtype=np.float64)
    exchange[slots] = 0.0
    exchange[slots, 1] = _stream(count)
    geometry = phx.discretization.particle.particle_pair_geometry(
        jnp.asarray(position), pairs
    )
    return phx.discretization.particle.ParticleExchangeLedger.from_exchange(
        pairs,
        geometry,
        jnp.asarray(exchange),
        jnp.zeros((capacity + 1, dimension), dtype=jnp.float64),
        velocities=jnp.asarray(velocity),
    )


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize(("count", "capacity", "interleave"), _CASES)
def test_exchange_ledger_torque_and_power_are_pair_capacity_independent(
    count: int, capacity: int, interleave: bool, dimension: int
) -> None:
    reference = _ledger(count, count, False, dimension)
    padded = _ledger(count, capacity, interleave, dimension)

    assert _bits(padded.torque, padded.relative_power) == _bits(
        reference.torque, reference.relative_power
    )
    assert int(padded.pair_count) == count
    if count == 60:
        expected = np.asarray(1.0, dtype=np.float64)
        assert _bits(padded.relative_power) == _bits(expected)


def _particles(
    masses: np.ndarray, slots: np.ndarray, capacity: int, dimension: int
) -> Any:
    active = np.zeros((capacity,), dtype=np.bool_)
    active[slots] = True
    return phx.discretization.ParticleSetPlan(
        np.arange(capacity),
        _scatter(masses, slots, capacity, 3.0),
        ambient_dimension=dimension,
        active_mask=active,
    ).prepare()


def _isolated_positions(count: int, slots: np.ndarray, capacity: int) -> np.ndarray:
    # Particles 10 apart on y = 1 have no physical pairs at smoothing length 1.
    leaf = np.stack(
        (10.0 * np.arange(1, count + 1, dtype=np.float64), np.ones((count,))), axis=-1
    )
    return _scatter(leaf, slots, capacity, 0.5)


def _barotropic(count: int, capacity: int, interleave: bool) -> Any:
    slots = _slots(count, capacity, interleave)
    stream = _stream(count)
    particles = _particles(stream, slots, capacity, 2)
    compiled = phx.equations.compile_barotropic_sph_problem(
        phx.equations.BarotropicFluidProblemIR(
            "capacity-fluid", phx.equations.TaitBarotropicMaterial(1.0, 1.0)
        ),
        particles,
        phx.discretization.BarotropicSPHMethodPlan(
            phx.discretization.WendlandC2SPHKernel(2), 1.0
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            capacity * (capacity - 1) // 2
        ),
    )
    momentum = np.stack((-stream, np.zeros((count,))), axis=-1)
    return compiled.dynamics.diagnostics(
        0.0,
        jnp.asarray(_isolated_positions(count, slots, capacity)),
        jnp.asarray(_scatter(momentum, slots, capacity, 5.0)),
        None,
    )


def _barotropic_outputs(diagnostics: Any) -> tuple[Any, ...]:
    return (
        diagnostics.total_mass,
        diagnostics.linear_momentum,
        diagnostics.kinetic_energy,
        diagnostics.angular_momentum,
        diagnostics.net_internal_torque,
    )


@pytest.mark.parametrize(("count", "capacity", "interleave"), _CASES)
def test_barotropic_sph_diagnostics_are_particle_capacity_independent(
    count: int, capacity: int, interleave: bool
) -> None:
    reference = _barotropic(count, count, False)
    padded = _barotropic(count, capacity, interleave)

    assert _bits(*_barotropic_outputs(padded)) == _bits(*_barotropic_outputs(reference))
    if count == 60:
        assert float(padded.total_mass) == 1.0
        assert float(padded.kinetic_energy) == 0.5


def _wcsph(count: int, capacity: int, interleave: bool) -> Any:
    slots = _slots(count, capacity, interleave)
    stream = _stream(count)
    particles = _particles(stream, slots, capacity, 2)

    def push(time: Any, position: Any, velocity: Any, density: Any, args: Any) -> Any:
        del time, velocity, density, args
        return jnp.zeros_like(position).at[:, 0].set(1.0)

    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        phx.equations.WeaklyCompressibleFluidProblemIR(
            "capacity-wcsph",
            phx.equations.TaitBarotropicMaterial(1.0, 1.0),
            external_acceleration=push,
            external_acceleration_id="acceleration:capacity",
        ),
        particles,
        phx.discretization.WeaklyCompressibleSPHMethodPlan(
            phx.discretization.WendlandC2SPHKernel(2),
            1.0,
            density=phx.discretization.ContinuityDensityPlan(),
            physical_viscosity=phx.discretization.MorrisViscosityPlan(0.01),
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            capacity * (capacity - 1) // 2
        ),
    )
    velocity = np.stack((-np.ones((count,)), np.zeros((count,))), axis=-1)
    state = compiled.initialize_state(
        jnp.asarray(_isolated_positions(count, slots, capacity)),
        jnp.asarray(_scatter(velocity, slots, capacity, 5.0)),
        jnp.ones((capacity,), dtype=jnp.float64),
    )
    return compiled.dynamics.diagnostics(0.0, state, None)


def _wcsph_outputs(diagnostics: Any) -> tuple[Any, ...]:
    return (
        diagnostics.total_mass,
        diagnostics.linear_momentum,
        diagnostics.angular_momentum,
        diagnostics.kinetic_energy,
        diagnostics.internal_energy,
        diagnostics.total_energy,
        diagnostics.pressure_kinetic_power,
        diagnostics.internal_energy_rate,
        diagnostics.density_mean,
        diagnostics.external_force,
        diagnostics.external_power,
        diagnostics.net_viscous_torque,
    )


@pytest.mark.parametrize(("count", "capacity", "interleave"), _CASES)
def test_wcsph_diagnostics_are_particle_capacity_independent(
    count: int, capacity: int, interleave: bool
) -> None:
    reference = _wcsph(count, count, False)
    padded = _wcsph(count, capacity, interleave)

    assert _bits(*_wcsph_outputs(padded)) == _bits(*_wcsph_outputs(reference))
    if count == 60:
        assert float(padded.total_mass) == 1.0
        assert float(padded.external_power) == -1.0


def _morris(count: int, capacity: int, interleave: bool) -> Any:
    pairs, slots, valid = _star(count, capacity, interleave)
    position = _star_vectors(np.ones((count, 1)), np.zeros((1,)), slots, capacity)
    velocity = _star_vectors(np.ones((count, 1)), np.zeros((1,)), slots, capacity)
    masses = _star_vectors(2.0 * _stream(count), np.ones(()), slots, capacity)
    geometry = phx.discretization.particle.particle_pair_geometry(
        jnp.asarray(position), pairs
    )
    # Unit distance and smoothing length with regularization 7/8 make every
    # pair power exactly minus half the leaf mass.
    return sph_morris_viscous_force(
        jnp.asarray(masses),
        jnp.ones((capacity + 1,), dtype=jnp.float64),
        jnp.asarray(velocity),
        pairs,
        geometry,
        jnp.asarray(valid),
        phx.discretization.WendlandC2SPHKernel(1),
        1.0,
        1.0,
        0.875,
        particle_count=capacity + 1,
        execution=phx.discretization.ParticleExecutionPolicy(
            accumulation="deterministic"
        ),
        precision=phx.discretization.ParticlePrecisionPolicy(),
    )


@pytest.mark.parametrize(("count", "capacity", "interleave"), _CASES)
def test_morris_viscous_power_is_pair_capacity_independent(
    count: int, capacity: int, interleave: bool
) -> None:
    reference = _morris(count, count, False)
    padded = _morris(count, capacity, interleave)

    assert _bits(padded.dissipation_rate, padded.positive_power_defect) == _bits(
        reference.dissipation_rate, reference.positive_power_defect
    )
    if count == 60:
        assert float(padded.dissipation_rate) == 1.0


def _artificial(count: int, capacity: int, interleave: bool) -> Any:
    pairs, slots, valid = _star(count, capacity, interleave)
    leaf_masses = np.asarray([1.0, 0.5] + [2.0**-108] * (count - 2), dtype=np.float64)
    masses = _star_vectors(leaf_masses, np.ones(()), slots, capacity)
    active = np.zeros((capacity + 1,), dtype=np.bool_)
    active[slots] = True
    active[capacity] = True
    particles = phx.discretization.ParticleSetPlan(
        np.arange(capacity + 1), masses, ambient_dimension=1, active_mask=active
    ).prepare()
    position = _star_vectors(np.ones((count, 1)), np.zeros((1,)), slots, capacity)
    velocity = _star_vectors(
        np.full((count, 1), _ARTIFICIAL_SPEED), np.zeros((1,)), slots, capacity
    )
    geometry = phx.discretization.particle.particle_pair_geometry(
        jnp.asarray(position), pairs
    )
    return sph_artificial_viscosity_force(
        MonaghanArtificialViscosityPlan(1.0, activation="always"),
        particles,
        jnp.ones((capacity + 1,), dtype=jnp.float64),
        jnp.ones((capacity + 1,), dtype=jnp.float64),
        jnp.asarray(velocity),
        pairs,
        geometry,
        jnp.asarray(valid),
        phx.discretization.WendlandC2SPHKernel(1),
        1.0,
        phx.discretization.ParticleExecutionPolicy(accumulation="deterministic"),
    )


@pytest.mark.parametrize(("count", "capacity", "interleave"), _CASES)
def test_artificial_viscosity_power_is_pair_capacity_independent(
    count: int, capacity: int, interleave: bool
) -> None:
    reference = _artificial(count, count, False)
    padded = _artificial(count, capacity, interleave)

    assert _bits(padded.dissipation_rate, padded.positive_power_defect) == _bits(
        reference.dissipation_rate, reference.positive_power_defect
    )
    if count == 60:
        # The logical pair powers discriminate sequential from blocked sums.
        power = reference.pair_power
        assert _bits(compensated_sum(power)) != _bits(
            compensated_sum(jnp.pad(power, (0, 10)))
        )


def _diffusion(count: int, capacity: int, interleave: bool) -> Any:
    pairs, slots, valid = _star(count, capacity, interleave)
    active = np.zeros((capacity + 1,), dtype=np.bool_)
    active[slots] = True
    active[capacity] = True
    masses = _star_vectors(np.ones((count,)), np.ones(()), slots, capacity)
    particles = phx.discretization.ParticleSetPlan(
        np.arange(capacity + 1), masses, ambient_dimension=1, active_mask=active
    ).prepare()
    leaf_density = 1.0 + _stream(count)
    leaf_density[1:] += 2.0**-20 * np.arange(1, count, dtype=np.float64)
    density = _star_vectors(leaf_density, np.ones(()), slots, capacity)
    position = _star_vectors(np.ones((count, 1)), np.zeros((1,)), slots, capacity)
    geometry = phx.discretization.particle.particle_pair_geometry(
        jnp.asarray(position), pairs
    )
    return sph_density_diffusion_rate(
        MolteniColagrossiDensityDiffusionPlan(0.1),
        particles,
        jnp.asarray(density),
        jnp.ones((capacity + 1,), dtype=jnp.float64),
        pairs,
        geometry,
        jnp.asarray(valid),
        phx.discretization.WendlandC2SPHKernel(1),
        1.0,
        phx.discretization.ParticleExecutionPolicy(accumulation="deterministic"),
    )


@pytest.mark.parametrize(("count", "capacity", "interleave"), _CASES)
def test_density_diffusion_variance_rate_is_particle_capacity_independent(
    count: int, capacity: int, interleave: bool
) -> None:
    reference = _diffusion(count, count, False)
    padded = _diffusion(count, capacity, interleave)

    assert _bits(padded.variance_rate, padded.positive_variance_defect) == _bits(
        reference.variance_rate, reference.positive_variance_defect
    )


def _residuals(count: int, capacity: int, interleave: bool) -> Any:
    slots = _slots(count, capacity, interleave)
    excess = np.zeros((count,), dtype=np.float64)
    excess[:2] = 1.0
    active = np.zeros((capacity,), dtype=np.bool_)
    active[slots] = True
    return phx.discretization.particle_constraint_residuals(
        jnp.asarray(_scatter(1.0 + excess, slots, capacity, 3.0)),
        1.0,
        jnp.asarray(_scatter(_stream(count), slots, capacity, 5.0)),
        density_rate=jnp.asarray(_scatter(excess, slots, capacity, 7.0)),
        active_mask=jnp.asarray(active),
    )


def _residual_bits(residuals: Any) -> tuple[bytes, ...]:
    return _bits(
        residuals.relative_density_linf,
        residuals.relative_density_l2,
        residuals.relative_divergence_linf,
        residuals.relative_divergence_l2,
        residuals.pressure_complementarity,
    )


@pytest.mark.parametrize(("count", "capacity", "interleave"), _CASES)
def test_qualification_residuals_are_particle_capacity_independent(
    count: int, capacity: int, interleave: bool
) -> None:
    reference = _residuals(count, count, False)
    padded = _residuals(count, capacity, interleave)
    # The L2 gate sits one ulp below the logical density residual of 1.0.
    profile = phx.discretization.ParticleQualificationProfile(
        density_linf_tolerance=10.0,
        density_l2_tolerance=float(np.nextafter(1.0, 0.0)),
        divergence_linf_tolerance=10.0,
        divergence_l2_tolerance=10.0,
    )

    assert _residual_bits(padded) == _residual_bits(reference)
    assert _bits(profile.constraints_satisfied(padded)) == _bits(
        profile.constraints_satisfied(reference)
    )
    if count == 60:
        assert float(padded.relative_density_l2) == 1.0
        assert not bool(profile.constraints_satisfied(padded))


@pytest.mark.parametrize("count", (8, 60))
def test_qualification_default_mask_matches_explicit_full_mask(count: int) -> None:
    excess = np.zeros((count,), dtype=np.float64)
    excess[:2] = 1.0
    arguments = (jnp.asarray(1.0 + excess), 1.0, jnp.asarray(_stream(count)))
    default = phx.discretization.particle_constraint_residuals(
        *arguments, density_rate=jnp.asarray(excess)
    )

    assert _residual_bits(default) == _residual_bits(_residuals(count, count, False))
