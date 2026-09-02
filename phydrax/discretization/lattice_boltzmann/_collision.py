#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._lattice import LatticeBoltzmannVelocitySet
from ._moments import (
    central_moments,
    central_moments_from_cumulants,
    cumulants_from_central_moments,
    MomentBasisPlan,
    populations_from_central_moments,
    populations_from_raw_moments,
    PreparedMomentBasis,
    PreparedRelaxationSpectrum,
    raw_moments,
    RelaxationSpectrumPlan,
)
from ._precision import LatticeBoltzmannPrecisionPolicy


class LatticeBoltzmannCollisionDiagnostics(StrictModule):
    entropy_before: Array
    entropy_after: Array
    entropy_residual: Array
    minimum_population: Array
    positivity_margin: Array
    mass_error: Array
    momentum_error: Array
    relaxation_rate_minimum: Array
    relaxation_rate_maximum: Array
    stabilization_parameter: Array
    root_iterations: Array
    root_residual: Array


class LatticeBoltzmannCollisionResult(StrictModule):
    candidate_populations: Array
    populations: Array
    successful: Array
    diagnostics: LatticeBoltzmannCollisionDiagnostics


class BGKCollisionPlan(StrictModule, NonTrainableState):
    family: str = "bgk"
    collision_id: str = "lattice-boltzmann-collision:bgk"


class TRTCollisionPlan(StrictModule, NonTrainableState):
    magic_parameter: float = eqx.field(static=True)
    collision_id: str = eqx.field(static=True)
    family: str = "trt"

    def __init__(self, magic_parameter: float = 3.0 / 16.0, /):
        value = float(magic_parameter)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("TRT magic_parameter must be finite and positive.")
        self.magic_parameter = value
        self.collision_id = canonical_fingerprint(
            {"kind": "lattice-boltzmann-collision-trt", "magic_parameter": value}
        )

    def odd_relaxation_rate(self, even_rate: Array, /) -> Array:
        rate = jnp.asarray(even_rate)
        denominator = 1.0 / rate - 0.5
        odd = 1.0 / (0.5 + self.magic_parameter / denominator)
        invalid = (
            ~jnp.isfinite(rate)
            | (rate <= 0.0)
            | (rate >= 2.0)
            | ~jnp.isfinite(odd)
            | (odd <= 0.0)
            | (odd >= 2.0)
        )
        return eqx.error_if(odd, invalid, "TRT relaxation rates must lie in (0, 2).")


class MRTCollisionPlan(StrictModule, NonTrainableState):
    basis: MomentBasisPlan
    spectrum: RelaxationSpectrumPlan
    collision_id: str = eqx.field(static=True)
    family: str = "mrt"

    def __init__(self, basis: MomentBasisPlan, spectrum: RelaxationSpectrumPlan, /):
        if not isinstance(basis, MomentBasisPlan) or not isinstance(
            spectrum, RelaxationSpectrumPlan
        ):
            raise TypeError("MRT requires moment-basis and relaxation-spectrum plans.")
        self.basis = basis
        self.spectrum = spectrum
        self.collision_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-collision-mrt",
                "basis": basis.plan_id,
                "spectrum": spectrum.plan_id,
            }
        )


class RegularizedCollisionPlan(StrictModule, NonTrainableState):
    family: str = "regularized-second-order"
    collision_id: str = "lattice-boltzmann-collision:regularized-second-order"


class SmagorinskyCollisionPlan(StrictModule, NonTrainableState):
    coefficient: float = eqx.field(static=True)
    collision_id: str = eqx.field(static=True)
    family: str = "smagorinsky"

    def __init__(self, coefficient: float = 0.16, /):
        value = float(coefficient)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Smagorinsky coefficient must be finite and nonnegative.")
        self.coefficient = value
        self.collision_id = canonical_fingerprint(
            {"kind": "lattice-boltzmann-collision-smagorinsky", "coefficient": value}
        )


class CentralMomentCollisionPlan(StrictModule, NonTrainableState):
    basis: MomentBasisPlan
    spectrum: RelaxationSpectrumPlan
    collision_id: str = eqx.field(static=True)
    family: str = "central-moment"

    def __init__(self, basis: MomentBasisPlan, spectrum: RelaxationSpectrumPlan, /):
        self.basis = basis
        self.spectrum = spectrum
        self.collision_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-collision-central",
                "basis": basis.plan_id,
                "spectrum": spectrum.plan_id,
            }
        )


class CumulantCollisionPlan(StrictModule, NonTrainableState):
    basis: MomentBasisPlan
    spectrum: RelaxationSpectrumPlan
    collision_id: str = eqx.field(static=True)
    family: str = "cumulant"

    def __init__(self, basis: MomentBasisPlan, spectrum: RelaxationSpectrumPlan, /):
        self.basis = basis
        self.spectrum = spectrum
        self.collision_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-collision-cumulant",
                "basis": basis.plan_id,
                "spectrum": spectrum.plan_id,
            }
        )


class KBCCollisionPlan(StrictModule, NonTrainableState):
    basis: MomentBasisPlan
    collision_id: str = eqx.field(static=True)
    family: str = "kbc"

    def __init__(self, basis: MomentBasisPlan | None = None, /):
        selected = MomentBasisPlan() if basis is None else basis
        self.basis = selected
        self.collision_id = canonical_fingerprint(
            {"kind": "lattice-boltzmann-collision-kbc", "basis": selected.plan_id}
        )


class EntropicCollisionPlan(StrictModule, NonTrainableState):
    iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    collision_id: str = eqx.field(static=True)
    family: str = "entropic"

    def __init__(self, /, *, iterations: int = 12, tolerance: float = 1.0e-11):
        count = int(iterations)
        tol = float(tolerance)
        if count < 2 or not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("Entropic iterations and tolerance are invalid.")
        self.iterations = count
        self.tolerance = tol
        self.collision_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-collision-entropic",
                "iterations": count,
                "tolerance": tol,
            }
        )


LatticeBoltzmannCollisionPlan: TypeAlias = (
    BGKCollisionPlan
    | TRTCollisionPlan
    | MRTCollisionPlan
    | RegularizedCollisionPlan
    | SmagorinskyCollisionPlan
    | CentralMomentCollisionPlan
    | CumulantCollisionPlan
    | KBCCollisionPlan
    | EntropicCollisionPlan
)


class PreparedLatticeBoltzmannCollision(StrictModule, NonTrainableState):
    """Collision plan bound once to one lattice, precision, basis, and spectrum."""

    plan: LatticeBoltzmannCollisionPlan
    basis: PreparedMomentBasis | None
    spectrum: PreparedRelaxationSpectrum | None
    lattice_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def prepare_lattice_boltzmann_collision(
    plan: LatticeBoltzmannCollisionPlan,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> PreparedLatticeBoltzmannCollision:
    if not isinstance(
        plan,
        (
            BGKCollisionPlan,
            TRTCollisionPlan,
            MRTCollisionPlan,
            RegularizedCollisionPlan,
            SmagorinskyCollisionPlan,
            CentralMomentCollisionPlan,
            CumulantCollisionPlan,
            KBCCollisionPlan,
            EntropicCollisionPlan,
        ),
    ):
        raise TypeError("plan must be a lattice-Boltzmann collision plan.")
    capability = {
        "cumulant": "cumulant-unforced",
        "entropic": "entropic-unforced",
    }.get(plan.family, plan.family)
    velocity_set.require(capability)
    basis = None
    spectrum = None
    if isinstance(
        plan, (MRTCollisionPlan, CentralMomentCollisionPlan, CumulantCollisionPlan)
    ):
        basis = plan.basis.prepare(velocity_set, precision)
        spectrum = plan.spectrum.prepare(basis)
    elif isinstance(plan, KBCCollisionPlan):
        basis = plan.basis.prepare(velocity_set, precision)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-lattice-boltzmann-collision",
            "collision": plan.collision_id,
            "lattice": velocity_set.lattice_id,
            "precision": precision.policy_id,
            "basis": None if basis is None else basis.basis_id,
            "spectrum": None if spectrum is None else spectrum.spectrum_id,
        }
    )
    return PreparedLatticeBoltzmannCollision(
        plan,
        basis,
        spectrum,
        velocity_set.lattice_id,
        precision.policy_id,
        prepared_id,
    )


def macroscopic_raw_moments(
    populations: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> tuple[Array, Array]:
    values = precision.accumulation(populations)
    velocities = precision.accumulation(velocity_set.velocities)
    return jnp.sum(values, axis=-1), ein.contract("...q,qd->...d", values, velocities)


def quadratic_equilibrium(
    density: Array,
    velocity: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    rho = precision.compute(density)
    u = precision.compute(velocity)
    c = precision.coefficient(velocity_set.velocities)
    weights = precision.coefficient(velocity_set.weights)
    cs2 = precision.coefficient(velocity_set.sound_speed_squared)
    cu = ein.contract("...d,qd->...q", u, c)
    u2 = ein.contract("...d,...d->...", u, u)
    return precision.compute(
        weights
        * rho[..., None]
        * (1.0 + cu / cs2 + 0.5 * cu**2 / cs2**2 - 0.5 * u2[..., None] / cs2)
    )


def _population_rate(rate: Array, populations: Array, /) -> Array:
    value = jnp.asarray(rate, dtype=populations.dtype)
    return value if value.ndim == 0 else value[..., None]


def collide_bgk(
    populations: Array,
    equilibrium: Array,
    raw_force_source: Array,
    relaxation_rate: Array,
    /,
) -> Array:
    rate = _population_rate(relaxation_rate, populations)
    return (
        populations
        - rate * (populations - equilibrium)
        + (1.0 - 0.5 * rate) * raw_force_source
    )


def collide_trt(
    populations: Array,
    equilibrium: Array,
    raw_force_source: Array,
    even_rate: Array,
    odd_rate: Array,
    opposite: Array,
    /,
) -> Array:
    opposite_populations = populations[..., opposite]
    opposite_equilibrium = equilibrium[..., opposite]
    opposite_force = raw_force_source[..., opposite]
    even_population = 0.5 * (populations + opposite_populations)
    odd_population = 0.5 * (populations - opposite_populations)
    even_equilibrium = 0.5 * (equilibrium + opposite_equilibrium)
    odd_equilibrium = 0.5 * (equilibrium - opposite_equilibrium)
    even_force = 0.5 * (raw_force_source + opposite_force)
    odd_force = 0.5 * (raw_force_source - opposite_force)
    even = _population_rate(even_rate, populations)
    odd = _population_rate(odd_rate, populations)
    return (
        populations
        - even * (even_population - even_equilibrium)
        - odd * (odd_population - odd_equilibrium)
        + (1.0 - 0.5 * even) * even_force
        + (1.0 - 0.5 * odd) * odd_force
    )


def regularized_nonequilibrium(
    populations: Array,
    equilibrium: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    nonequilibrium = precision.compute(populations - equilibrium)
    c = precision.coefficient(velocity_set.velocities)
    weights = precision.coefficient(velocity_set.weights)
    cs2 = precision.coefficient(velocity_set.sound_speed_squared)
    identity = jnp.eye(velocity_set.dimension, dtype=nonequilibrium.dtype)
    stress = ein.contract("...q,qa,qb->...ab", nonequilibrium, c, c)
    hermite = ein.contract("qa,qb->qab", c, c) - cs2 * identity
    return weights * ein.contract("qab,...ab->...q", hermite, stress) / (2.0 * cs2**2)


def _entropy(populations: Array, weights: Array) -> Array:
    positive = populations > 0.0
    safe = jnp.where(positive, populations, 1.0)
    return jnp.sum(jnp.where(positive, safe * jnp.log(safe / weights), jnp.inf), axis=-1)


def _entropic_candidate(
    populations: Array,
    equilibrium: Array,
    weights: Array,
    beta: Array,
    plan: EntropicCollisionPlan,
) -> tuple[Array, Array, Array]:
    delta = equilibrium - populations
    negative = delta < 0.0
    upper = jnp.min(jnp.where(negative, -populations / delta, jnp.inf), axis=-1)
    upper = jnp.minimum(upper * (1.0 - 16.0 * jnp.finfo(populations.dtype).eps), 4.0)
    entropy0 = _entropy(populations, weights)

    def iteration(_, alpha):
        trial = populations + alpha[..., None] * delta
        safe = jnp.maximum(trial, jnp.finfo(populations.dtype).tiny)
        residual = jnp.sum(safe * jnp.log(safe / weights), axis=-1) - entropy0
        derivative = jnp.sum(delta * (jnp.log(safe / weights) + 1.0), axis=-1)
        newton = alpha - residual / jnp.where(jnp.abs(derivative) > 0.0, derivative, 1.0)
        return jnp.clip(newton, 1.0, upper)

    alpha = jax.lax.fori_loop(
        0,
        plan.iterations,
        iteration,
        jnp.minimum(jnp.asarray(2.0, populations.dtype), upper),
    )
    pre_relaxed = populations + alpha[..., None] * delta
    candidate = populations + beta[..., None] * (pre_relaxed - populations)
    residual = jnp.abs(_entropy(pre_relaxed, weights) - entropy0)
    return candidate, alpha, residual


def _collision_diagnostics(
    old: Array,
    candidate: Array,
    raw_force: Array,
    lattice: LatticeBoltzmannVelocitySet,
    rates: Array,
    stabilization: Array,
    iterations: Array,
    root_residual: Array,
) -> LatticeBoltzmannCollisionDiagnostics:
    weights = jnp.asarray(lattice.weights, dtype=old.dtype)
    old_mass, old_momentum = macroscopic_raw_moments(
        old,
        lattice,
        LatticeBoltzmannPrecisionPolicy(
            population_dtype=old.dtype,
            compute_dtype=old.dtype,
            accumulation_dtype=old.dtype,
            certification_dtype=old.dtype,
        ),
    )
    new_mass, new_momentum = macroscopic_raw_moments(
        candidate,
        lattice,
        LatticeBoltzmannPrecisionPolicy(
            population_dtype=old.dtype,
            compute_dtype=old.dtype,
            accumulation_dtype=old.dtype,
            certification_dtype=old.dtype,
        ),
    )
    force_momentum = ein.contract(
        "...q,qd->...d", raw_force, jnp.asarray(lattice.velocities, dtype=old.dtype)
    )
    before = _entropy(old, weights)
    after = _entropy(candidate, weights)
    return LatticeBoltzmannCollisionDiagnostics(
        entropy_before=before,
        entropy_after=after,
        entropy_residual=after - before,
        minimum_population=jnp.min(candidate),
        positivity_margin=jnp.min(candidate),
        mass_error=jnp.max(jnp.abs(new_mass - old_mass)),
        momentum_error=jnp.max(jnp.abs(new_momentum - old_momentum - force_momentum)),
        relaxation_rate_minimum=jnp.min(rates),
        relaxation_rate_maximum=jnp.max(rates),
        stabilization_parameter=stabilization,
        root_iterations=iterations,
        root_residual=root_residual,
    )


def collide_detailed(
    plan: LatticeBoltzmannCollisionPlan | PreparedLatticeBoltzmannCollision,
    populations: Array,
    equilibrium: Array,
    raw_force_source: Array,
    even_rate: Array,
    velocity: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> LatticeBoltzmannCollisionResult:
    prepared = (
        plan
        if isinstance(plan, PreparedLatticeBoltzmannCollision)
        else prepare_lattice_boltzmann_collision(plan, velocity_set, precision)
    )
    if (
        prepared.lattice_id != velocity_set.lattice_id
        or prepared.precision_policy_id != precision.policy_id
    ):
        raise ValueError("Prepared collision, lattice, and precision do not match.")
    plan = prepared.plan
    rate = jnp.asarray(even_rate, dtype=populations.dtype)
    rates = rate[..., None]
    stabilization = jnp.asarray(1.0, dtype=populations.dtype)
    iterations = jnp.asarray(0, dtype=jnp.int32)
    root_residual = jnp.asarray(0.0, dtype=populations.dtype)

    if isinstance(plan, BGKCollisionPlan):
        candidate = collide_bgk(populations, equilibrium, raw_force_source, rate)
    elif isinstance(plan, TRTCollisionPlan):
        odd = plan.odd_relaxation_rate(rate)
        rates = jnp.stack((rate, odd), axis=-1)
        candidate = collide_trt(
            populations, equilibrium, raw_force_source, rate, odd, velocity_set.opposite
        )
    elif isinstance(plan, MRTCollisionPlan):
        basis = prepared.basis
        spectrum = prepared.spectrum
        if basis is None or spectrum is None:
            raise RuntimeError("Prepared MRT collision lacks basis or spectrum.")
        rates = spectrum.relaxation_rates(rate)
        moments = raw_moments(populations, basis, precision)
        eq_moments = raw_moments(equilibrium, basis, precision)
        source_moments = raw_moments(raw_force_source, basis, precision)
        candidate = populations_from_raw_moments(
            moments
            - rates * (moments - eq_moments)
            + (1.0 - 0.5 * rates) * source_moments,
            basis,
            precision,
        )
    elif isinstance(plan, RegularizedCollisionPlan):
        projected = regularized_nonequilibrium(
            populations, equilibrium, velocity_set, precision
        )
        candidate = (
            equilibrium
            + (1.0 - rate)[..., None] * projected
            + (1.0 - 0.5 * rate)[..., None] * raw_force_source
        )
    elif isinstance(plan, SmagorinskyCollisionPlan):
        projected = regularized_nonequilibrium(
            populations, equilibrium, velocity_set, precision
        )
        stress_norm = jnp.sqrt(jnp.sum(projected**2, axis=-1))
        tau0 = 1.0 / rate
        tau_eff = 0.5 * (
            tau0 + jnp.sqrt(tau0**2 + 36.0 * plan.coefficient**2 * stress_norm)
        )
        effective = 1.0 / tau_eff
        rates = effective[..., None]
        candidate = collide_bgk(populations, equilibrium, raw_force_source, effective)
    elif isinstance(plan, CentralMomentCollisionPlan):
        basis = prepared.basis
        spectrum = prepared.spectrum
        if basis is None or spectrum is None:
            raise RuntimeError(
                "Prepared central-moment collision lacks basis or spectrum."
            )
        rates = spectrum.relaxation_rates(rate)
        moments = central_moments(populations, velocity, velocity_set, basis, precision)
        eq_moments = central_moments(
            equilibrium, velocity, velocity_set, basis, precision
        )
        source_moments = central_moments(
            raw_force_source, velocity, velocity_set, basis, precision
        )
        candidate = populations_from_central_moments(
            moments
            - rates * (moments - eq_moments)
            + (1.0 - 0.5 * rates) * source_moments,
            velocity,
            basis,
            precision,
        )
    elif isinstance(plan, CumulantCollisionPlan):
        basis = prepared.basis
        spectrum = prepared.spectrum
        if basis is None or spectrum is None:
            raise RuntimeError("Prepared cumulant collision lacks basis or spectrum.")
        rates = spectrum.relaxation_rates(rate)
        cumulants = cumulants_from_central_moments(
            central_moments(populations, velocity, velocity_set, basis, precision), basis
        )
        eq_cumulants = cumulants_from_central_moments(
            central_moments(equilibrium, velocity, velocity_set, basis, precision), basis
        )
        relaxed = cumulants - rates * (cumulants - eq_cumulants)
        candidate = populations_from_central_moments(
            central_moments_from_cumulants(relaxed, basis), velocity, basis, precision
        )
    elif isinstance(plan, KBCCollisionPlan):
        regular = regularized_nonequilibrium(
            populations, equilibrium, velocity_set, precision
        )
        higher = populations - equilibrium - regular
        denominator = jnp.sum(
            higher**2 / jnp.maximum(equilibrium, jnp.finfo(populations.dtype).tiny),
            axis=-1,
        )
        numerator = jnp.sum(
            regular
            * higher
            / jnp.maximum(equilibrium, jnp.finfo(populations.dtype).tiny),
            axis=-1,
        )
        beta = 0.5 * rate
        gamma = jnp.where(
            denominator > 0.0,
            1.0 / beta - (2.0 - 1.0 / beta) * numerator / denominator,
            2.0,
        )
        stabilization = gamma
        candidate = populations - beta[..., None] * (
            2.0 * regular + gamma[..., None] * higher
        )
    else:
        beta = 0.5 * rate
        candidate, alpha, root_residual = _entropic_candidate(
            populations,
            equilibrium,
            jnp.asarray(velocity_set.weights, dtype=populations.dtype),
            beta,
            plan,
        )
        stabilization = alpha
        iterations = jnp.asarray(plan.iterations, dtype=jnp.int32)

    candidate = precision.population(candidate)
    diagnostics = _collision_diagnostics(
        populations,
        candidate,
        raw_force_source,
        velocity_set,
        rates,
        stabilization,
        iterations,
        root_residual,
    )
    finite = jnp.all(jnp.isfinite(candidate))
    positivity = (
        diagnostics.minimum_population > 0.0
        if isinstance(plan, (KBCCollisionPlan, EntropicCollisionPlan))
        else jnp.asarray(True)
    )
    root_tolerance = (
        plan.tolerance if isinstance(plan, EntropicCollisionPlan) else jnp.inf
    )
    successful = finite & positivity & jnp.all(root_residual <= root_tolerance)
    accepted = precision.population(jnp.where(successful, candidate, populations))
    return LatticeBoltzmannCollisionResult(candidate, accepted, successful, diagnostics)


__all__ = [
    "BGKCollisionPlan",
    "CentralMomentCollisionPlan",
    "CumulantCollisionPlan",
    "EntropicCollisionPlan",
    "KBCCollisionPlan",
    "LatticeBoltzmannCollisionDiagnostics",
    "LatticeBoltzmannCollisionPlan",
    "LatticeBoltzmannCollisionResult",
    "MRTCollisionPlan",
    "RegularizedCollisionPlan",
    "SmagorinskyCollisionPlan",
    "TRTCollisionPlan",
    "collide_bgk",
    "collide_detailed",
    "collide_trt",
    "macroscopic_raw_moments",
    "quadratic_equilibrium",
    "regularized_nonequilibrium",
]
