#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._kinetic_entropy import (
    KineticEntropyRootPlan,
    solve_kinetic_entropy_root,
)
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


class LatticeBoltzmannSmagorinskyEvidence(StrictModule):
    """Collision-local evidence for the athermal, unit-filter-width closure."""

    base_relaxation_time: Array
    effective_relaxation_time: Array
    molecular_kinematic_viscosity: Array
    effective_kinematic_viscosity: Array
    eddy_kinematic_viscosity: Array
    nonequilibrium_stress_norm: Array
    coefficient_active: Array
    conserved_moment_defect: Array
    finite: Array
    successful: Array
    support_satisfied: Array
    coefficient: float = eqx.field(static=True)
    coefficient_lower_bound: float = eqx.field(static=True)
    coefficient_requires_finite: bool = eqx.field(static=True)
    base_relaxation_rate_bounds: tuple[float, float] = eqx.field(static=True)
    base_relaxation_rate_bounds_exclusive: bool = eqx.field(static=True)
    density_lower_bound: float = eqx.field(static=True)
    density_lower_bound_exclusive: bool = eqx.field(static=True)
    filter_width_in_lattice_units: float = eqx.field(static=True)


class LatticeBoltzmannCollisionResult(StrictModule):
    candidate_populations: Array
    populations: Array
    successful: Array
    diagnostics: LatticeBoltzmannCollisionDiagnostics
    smagorinsky_evidence: LatticeBoltzmannSmagorinskyEvidence | None


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


KBCVariant = Literal["a", "b", "c", "d"]
KBCStabilizerKind = Literal["quadratic", "exact", "hybrid"]


class KBCCollisionPlan(StrictModule, NonTrainableState):
    basis: MomentBasisPlan
    root: KineticEntropyRootPlan
    variant: KBCVariant = eqx.field(static=True)
    stabilizer: KBCStabilizerKind = eqx.field(static=True)
    collision_id: str = eqx.field(static=True)
    family: str = "kbc"

    def __init__(
        self,
        basis: MomentBasisPlan | None = None,
        /,
        *,
        variant: KBCVariant = "b",
        stabilizer: KBCStabilizerKind = "quadratic",
        root: KineticEntropyRootPlan | None = None,
    ):
        if variant not in ("a", "b", "c", "d"):
            raise ValueError(f"Unknown KBC variant {variant!r}.")
        if stabilizer not in ("quadratic", "exact", "hybrid"):
            raise ValueError(f"Unknown KBC stabilizer {stabilizer!r}.")
        selected_basis = MomentBasisPlan() if basis is None else basis
        selected_root = KineticEntropyRootPlan() if root is None else root
        if not isinstance(selected_basis, MomentBasisPlan):
            raise TypeError("basis must be a MomentBasisPlan.")
        if not isinstance(selected_root, KineticEntropyRootPlan):
            raise TypeError("root must be a KineticEntropyRootPlan.")
        self.basis = selected_basis
        self.root = selected_root
        self.variant = variant
        self.stabilizer = stabilizer
        self.collision_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-collision-kbc",
                "basis": selected_basis.plan_id,
                "variant": variant,
                "stabilizer": stabilizer,
                "root": selected_root.plan_id,
            }
        )


class EntropicCollisionPlan(StrictModule, NonTrainableState):
    root: KineticEntropyRootPlan
    collision_id: str = eqx.field(static=True)
    family: str = "entropic"

    def __init__(
        self,
        root: KineticEntropyRootPlan | None = None,
        /,
        *,
        iterations: int = 24,
        tolerance: float = 1.0e-11,
        strategy: str = "exact",
    ):
        selected = (
            KineticEntropyRootPlan(
                strategy=strategy,
                maximum_steps=iterations,
                residual_tolerance=tolerance,
            )
            if root is None
            else root
        )
        if not isinstance(selected, KineticEntropyRootPlan):
            raise TypeError("root must be a KineticEntropyRootPlan.")
        self.root = selected
        self.collision_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-collision-entropic",
                "root": selected.plan_id,
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


def _smagorinsky_relaxation(
    populations: Array,
    equilibrium: Array,
    base_relaxation_rate: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    coefficient: float,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Solve the local athermal SRT closure from the raw nonequilibrium stress.

    The quadratic uses ``Pi_neq = sum_q(c_q c_q (f_q - f_eq_q))`` and the
    lattice relation ``nu = cs2 * (tau - 1 / 2)`` at unit filter width.
    """
    rate = jnp.asarray(base_relaxation_rate, dtype=populations.dtype)
    rate = eqx.error_if(
        rate,
        jnp.any(~jnp.isfinite(rate) | (rate <= 0.0) | (rate >= 2.0)),
        "Smagorinsky base relaxation rate must lie in (0, 2).",
    )
    nonequilibrium = precision.compute(populations - equilibrium)
    velocities = precision.coefficient(velocity_set.velocities)
    stress = ein.contract("...q,qa,qb->...ab", nonequilibrium, velocities, velocities)
    stress_squared = jnp.sum(stress**2, axis=(-2, -1))
    positive_stress = stress_squared > 0.0
    stress_norm = jnp.where(
        positive_stress,
        jnp.sqrt(jnp.where(positive_stress, stress_squared, 1.0)),
        0.0,
    )
    density = jnp.sum(precision.accumulation(populations), axis=-1)
    density_supported = jnp.isfinite(density) & (density > 0.0)
    safe_density = jnp.where(density_supported, density, 1.0)
    cs2 = precision.compute(velocity_set.sound_speed_squared)
    tau0 = 1.0 / rate
    coefficient_value = jnp.asarray(coefficient, dtype=stress_norm.dtype)
    if coefficient == 0.0:
        tau_eff = tau0
    else:
        radicand = tau0**2 + (
            2.0
            * jnp.sqrt(jnp.asarray(2.0, dtype=stress_norm.dtype))
            * coefficient_value**2
            * stress_norm
            / (safe_density * cs2**2)
        )
        tau_eff = jnp.where(positive_stress, 0.5 * (tau0 + jnp.sqrt(radicand)), tau0)
    molecular_viscosity = cs2 * (tau0 - 0.5)
    effective_viscosity = cs2 * (tau_eff - 0.5)
    eddy_viscosity = effective_viscosity - molecular_viscosity
    coefficient_active = (coefficient_value > 0.0) & positive_stress & density_supported
    support_satisfied = jnp.all(density_supported)
    return (
        1.0 / tau_eff,
        tau0,
        tau_eff,
        molecular_viscosity,
        effective_viscosity,
        eddy_viscosity,
        stress_norm,
        coefficient_active,
        support_satisfied,
    )


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
) -> tuple[Array, Array, Array, Array, Array]:
    root = solve_kinetic_entropy_root(
        plan.root,
        populations,
        equilibrium - populations,
        base_measure=weights,
    )
    candidate = populations + beta[..., None] * (root.mirror_populations - populations)
    evidence = root.evidence
    return (
        candidate,
        evidence.alpha,
        evidence.residual,
        evidence.iterations,
        evidence.successful,
    )


def _kbc_components(
    populations: Array,
    equilibrium: Array,
    velocity: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    basis: PreparedMomentBasis,
    precision: LatticeBoltzmannPrecisionPolicy,
    variant: KBCVariant,
    /,
) -> tuple[Array, Array]:
    nonequilibrium_moments = central_moments(
        populations, velocity, velocity_set, basis, precision
    ) - central_moments(equilibrium, velocity, velocity_set, basis, precision)
    exponents = np.asarray(basis.exponents, dtype=np.int32)
    degrees = np.sum(exponents, axis=1)
    include_third = variant in ("c", "d")
    selected = jnp.where(
        jnp.asarray(
            (degrees == 2) | ((degrees == 3) & include_third),
            dtype=jnp.bool_,
        ),
        nonequilibrium_moments,
        0.0,
    )
    if variant in ("a", "c"):
        diagonal_indices = tuple(
            index
            for index, exponent in enumerate(basis.exponents)
            if sum(exponent) == 2 and max(exponent) == 2
        )
        if len(diagonal_indices) != velocity_set.dimension:
            raise RuntimeError("KBC moment basis lacks the diagonal stress modes.")
        trace = jnp.mean(selected[..., jnp.asarray(diagonal_indices)], axis=-1)
        for index in diagonal_indices:
            selected = selected.at[..., index].add(-trace)
    shear = populations_from_central_moments(
        selected,
        velocity,
        basis,
        precision,
    )
    higher = populations - equilibrium - shear
    return shear, higher


def _kbc_candidate(
    populations: Array,
    equilibrium: Array,
    raw_force_source: Array,
    beta: Array,
    shear: Array,
    higher: Array,
    weights: Array,
    plan: KBCCollisionPlan,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    safe_equilibrium = jnp.maximum(equilibrium, jnp.finfo(populations.dtype).tiny)
    denominator = jnp.sum(higher**2 / safe_equilibrium, axis=-1)
    numerator = jnp.sum(shear * higher / safe_equilibrium, axis=-1)
    quadratic_gamma = jnp.where(
        denominator > 0.0,
        1.0 / beta - (2.0 - 1.0 / beta) * numerator / denominator,
        2.0,
    )
    base = populations - 2.0 * beta[..., None] * shear
    direction = -beta[..., None] * higher
    inactive = denominator <= (
        32.0
        * jnp.finfo(populations.dtype).eps
        * jnp.maximum(jnp.max(populations, axis=-1), 1.0)
    )

    def derivative_at(gamma: Array) -> Array:
        candidate = base + gamma[..., None] * direction
        safe = jnp.where(candidate > 0.0, candidate, 1.0)
        return jnp.sum(
            direction * (jnp.log(safe / weights) + 1.0),
            axis=-1,
        )

    approximate = base + quadratic_gamma[..., None] * direction
    approximate_residual = jnp.abs(derivative_at(quadratic_gamma))
    approximate_valid = (
        jnp.all(jnp.isfinite(approximate), axis=-1)
        & (jnp.min(approximate, axis=-1) > 0.0)
        & jnp.isfinite(approximate_residual)
        & (approximate_residual <= plan.root.approximation_tolerance)
    )
    if plan.stabilizer == "quadratic":
        gamma = quadratic_gamma
        residual = approximate_residual
        successful = approximate_valid | inactive
        iterations = jnp.zeros(gamma.shape, dtype=jnp.int32)
    else:
        ratios = jnp.where(direction < 0.0, -base / direction, jnp.inf)
        upper = jnp.minimum(
            jnp.min(ratios, axis=-1) * (1.0 - plan.root.positivity_margin),
            plan.root.maximum_root,
        )
        lower = jnp.zeros_like(upper)
        lower_value = derivative_at(lower)
        upper_value = derivative_at(upper)
        bracketed = (
            jnp.all(jnp.isfinite(base), axis=-1)
            & (jnp.min(base, axis=-1) > 0.0)
            & (upper > 0.0)
            & jnp.isfinite(lower_value)
            & jnp.isfinite(upper_value)
            & (lower_value <= 0.0)
            & (upper_value >= 0.0)
        )
        use_approximation = (plan.stabilizer == "hybrid") & approximate_valid
        active = bracketed & ~use_approximation & ~inactive
        current = jnp.minimum(jnp.maximum(quadratic_gamma, lower), upper)
        counts = jnp.zeros(current.shape, dtype=jnp.int32)

        def iteration(_, state):
            value, lo, hi, current_active, step_counts = state
            function = derivative_at(value)
            candidate_populations = base + value[..., None] * direction
            safe = jnp.where(candidate_populations > 0.0, candidate_populations, 1.0)
            derivative = jnp.sum(direction * direction / safe, axis=-1)
            next_lo = jnp.where(current_active & (function <= 0.0), value, lo)
            next_hi = jnp.where(current_active & (function > 0.0), value, hi)
            newton = value - function / jnp.where(derivative > 0.0, derivative, 1.0)
            use_newton = (
                current_active
                & jnp.isfinite(newton)
                & (derivative > 0.0)
                & (newton > next_lo)
                & (newton < next_hi)
            )
            proposed = jnp.where(use_newton, newton, 0.5 * (next_lo + next_hi))
            converged = current_active & (
                jnp.abs(derivative_at(proposed)) <= plan.root.residual_tolerance
            )
            return (
                jnp.where(current_active, proposed, value),
                next_lo,
                next_hi,
                current_active & ~converged,
                step_counts + current_active.astype(jnp.int32),
            )

        exact_gamma, _, _, _, exact_iterations = jax.lax.fori_loop(
            0,
            plan.root.maximum_steps,
            iteration,
            (current, lower, upper, active, counts),
        )
        gamma = jnp.where(use_approximation, quadratic_gamma, exact_gamma)
        gamma = jnp.where(inactive, 2.0, gamma)
        residual = jnp.abs(derivative_at(gamma))
        exact_success = bracketed & (residual <= plan.root.residual_tolerance)
        successful = inactive | use_approximation | exact_success
        iterations = jnp.where(use_approximation | inactive, 0, exact_iterations)

    candidate = base + gamma[..., None] * direction
    candidate = candidate + (1.0 - beta[..., None]) * raw_force_source
    successful &= jnp.all(jnp.isfinite(candidate), axis=-1) & (
        jnp.min(candidate, axis=-1) > 0.0
    )
    return candidate, gamma, residual, iterations, successful


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
    root_successful = jnp.asarray(True)
    smagorinsky_values = None

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
        (
            effective,
            tau0,
            tau_eff,
            molecular_viscosity,
            effective_viscosity,
            eddy_viscosity,
            stress_norm,
            coefficient_active,
            support_satisfied,
        ) = _smagorinsky_relaxation(
            populations,
            equilibrium,
            rate,
            velocity_set,
            precision,
            plan.coefficient,
        )
        rates = effective[..., None]
        candidate = collide_bgk(populations, equilibrium, raw_force_source, effective)
        smagorinsky_values = (
            tau0,
            tau_eff,
            molecular_viscosity,
            effective_viscosity,
            eddy_viscosity,
            stress_norm,
            coefficient_active,
            support_satisfied,
        )
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
        basis = prepared.basis
        if basis is None:
            raise RuntimeError("Prepared KBC collision lacks a moment basis.")
        shear, higher = _kbc_components(
            populations,
            equilibrium,
            velocity,
            velocity_set,
            basis,
            precision,
            plan.variant,
        )
        beta = 0.5 * rate
        (
            candidate,
            gamma,
            root_residual,
            iterations,
            root_successful,
        ) = _kbc_candidate(
            populations,
            equilibrium,
            raw_force_source,
            beta,
            shear,
            higher,
            jnp.asarray(velocity_set.weights, dtype=populations.dtype),
            plan,
        )
        stabilization = gamma
    else:
        beta = 0.5 * rate
        (
            candidate,
            alpha,
            root_residual,
            iterations,
            root_successful,
        ) = _entropic_candidate(
            populations,
            equilibrium,
            jnp.asarray(velocity_set.weights, dtype=populations.dtype),
            beta,
            plan,
        )
        stabilization = alpha

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
        plan.root.residual_tolerance
        if isinstance(plan, EntropicCollisionPlan)
        else jnp.inf
    )
    successful = (
        finite
        & positivity
        & jnp.all(root_successful)
        & jnp.all(root_residual <= root_tolerance)
    )
    smagorinsky_evidence = None
    if smagorinsky_values is not None:
        (
            tau0,
            tau_eff,
            molecular_viscosity,
            effective_viscosity,
            eddy_viscosity,
            stress_norm,
            coefficient_active,
            support_satisfied,
        ) = smagorinsky_values
        evidence_finite = finite & jnp.all(
            jnp.isfinite(tau_eff)
            & jnp.isfinite(effective_viscosity)
            & jnp.isfinite(stress_norm)
        )
        successful = successful & evidence_finite & support_satisfied
        smagorinsky_evidence = LatticeBoltzmannSmagorinskyEvidence(
            base_relaxation_time=tau0,
            effective_relaxation_time=tau_eff,
            molecular_kinematic_viscosity=molecular_viscosity,
            effective_kinematic_viscosity=effective_viscosity,
            eddy_kinematic_viscosity=eddy_viscosity,
            nonequilibrium_stress_norm=stress_norm,
            coefficient_active=coefficient_active,
            conserved_moment_defect=jnp.maximum(
                diagnostics.mass_error, diagnostics.momentum_error
            ),
            finite=evidence_finite,
            successful=successful,
            support_satisfied=support_satisfied,
            coefficient=plan.coefficient,
            coefficient_lower_bound=0.0,
            coefficient_requires_finite=True,
            base_relaxation_rate_bounds=(0.0, 2.0),
            base_relaxation_rate_bounds_exclusive=True,
            density_lower_bound=0.0,
            density_lower_bound_exclusive=True,
            filter_width_in_lattice_units=1.0,
        )
    accepted = precision.population(
        jnp.where(successful[..., None], candidate, populations)
    )
    return LatticeBoltzmannCollisionResult(
        candidate, accepted, successful, diagnostics, smagorinsky_evidence
    )


__all__ = [
    "BGKCollisionPlan",
    "CentralMomentCollisionPlan",
    "CumulantCollisionPlan",
    "EntropicCollisionPlan",
    "KBCCollisionPlan",
    "KBCStabilizerKind",
    "KBCVariant",
    "LatticeBoltzmannCollisionDiagnostics",
    "LatticeBoltzmannCollisionPlan",
    "LatticeBoltzmannCollisionResult",
    "LatticeBoltzmannSmagorinskyEvidence",
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
