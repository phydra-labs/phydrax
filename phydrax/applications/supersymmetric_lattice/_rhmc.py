#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""End-to-end phase-quenched RHMC composition for regulated 2D twisted SYM."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import SpectralInterval
from ...operators.path_integral import (
    dirac_normal_operator,
    FractionalPowerPseudofermionTerm,
    generate_minimax_rational_approximation,
    plan_minimax_rational_approximation,
    power_rational_target,
    PseudofermionSolveRoles,
)
from ...sampling import (
    initialize_rhmc_state,
    NestedForcePartition,
    NestedForcePlan,
    plan_rhmc,
    prepare_rhmc,
    PreparedRHMCKernel,
    RHMCResourcePolicy,
    RHMCSampleResult,
    sample_rhmc,
    SeparableActionRegistry,
    SeparableActionTerm,
)
from ._actions import PreparedTwistedSYMAction
from ._fermions import RegulatedTwistedDiracOperator, TwistedKahlerDiracOperator
from ._twisted_n2 import (
    BoundedEuclideanStateGeometry,
    TwistedN2SYMPlan,
    TwistedSYMCoordinateLayout,
)


class PreparedTwistedN2RHMC(StrictModule):
    """Fully bound regulated finite theory, fermions, rationals, and sampler."""

    theory: TwistedN2SYMPlan = eqx.field(static=True)
    bosonic_action: PreparedTwistedSYMAction
    layout: TwistedSYMCoordinateLayout
    kahler_dirac: TwistedKahlerDiracOperator
    regulated_dirac: RegulatedTwistedDiracOperator
    spectral_interval: SpectralInterval
    pseudofermion: FractionalPowerPseudofermionTerm
    registry: SeparableActionRegistry
    kernel: PreparedRHMCKernel
    phase_quenched_power: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class TwistedN2RHMCEvidence(StrictModule):
    acceptance_rate: Array
    maximum_energy_error: Array
    divergent_count: Array
    nonfinite_count: Array
    membership_failure_count: Array
    force_failure_count: Array
    maximum_force_solve_error: Array
    solve_error_bounds_available: Array
    solve_error_bounds_certified: Array
    finite: Array
    successful: Array
    sample_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class TwistedN2RHMCRun(StrictModule):
    samples: RHMCSampleResult
    evidence: TwistedN2RHMCEvidence
    prepared_id: str = eqx.field(static=True)


def _scalar_regulator(
    layout: TwistedSYMCoordinateLayout,
    coordinates: Array,
    coefficient: float,
    /,
) -> Array:
    configuration = layout.unpack(coordinates)
    forward = configuration.links.values
    reverse = configuration.reverse_links.values
    reverse_adjoint = jnp.swapaxes(jnp.conj(reverse), -1, -2)
    difference = forward - reverse_adjoint
    return jnp.asarray(coefficient**2, dtype=jnp.real(coordinates).dtype) * jnp.sum(
        jnp.real(difference * jnp.conj(difference))
    )


def _u1_regulator(
    layout: TwistedSYMCoordinateLayout,
    coordinates: Array,
    coefficient: float,
    /,
) -> Array:
    configuration = layout.unpack(coordinates)
    forward = configuration.links.values
    reverse = configuration.reverse_links.values
    forward_det = jnp.linalg.det(forward)
    reverse_det = jnp.linalg.det(reverse)
    residual = jnp.abs(forward_det - 1.0) ** 2 + jnp.abs(reverse_det - 1.0) ** 2
    return jnp.asarray(coefficient**2, dtype=jnp.real(coordinates).dtype) * jnp.sum(
        residual
    )


def prepare_twisted_n2_rhmc(
    theory: TwistedN2SYMPlan,
    configuration_template: ArrayLike,
    /,
    *,
    step_size: float,
    trajectory_steps: int,
    rational_poles: int = 8,
    rational_verification_points: int = 4097,
    rational_tolerance: float | None = None,
    bosonic_substeps: int = 2,
    solves: PseudofermionSolveRoles | None = None,
    resources: RHMCResourcePolicy | None = None,
) -> PreparedTwistedN2RHMC:
    """Prepare the finite regulated phase-quenched determinant target."""
    if not isinstance(theory, TwistedN2SYMPlan):
        raise TypeError("theory must be TwistedN2SYMPlan.")
    coordinates = jnp.asarray(configuration_template)
    bosonic = theory.prepare_bosonic()
    layout = TwistedSYMCoordinateLayout(bosonic)
    if coordinates.shape != layout.coordinate_shape:
        raise ValueError(
            f"configuration_template must have shape {layout.coordinate_shape}."
        )
    if jnp.iscomplexobj(coordinates) or not jnp.issubdtype(
        coordinates.dtype, jnp.floating
    ):
        raise TypeError(
            "Twisted-SYM RHMC configuration coordinates must be real floating arrays."
        )
    geometry = BoundedEuclideanStateGeometry(
        theory.coordinate_bound,
        geometry_id=f"twisted-n2-bounded-euclidean:{theory.plan_id}",
    )
    if not bool(geometry.contains(coordinates)):
        raise ValueError(
            "configuration_template lies outside the admitted coordinate domain."
        )

    kahler = TwistedKahlerDiracOperator(theory, layout, coordinates)
    regulated = RegulatedTwistedDiracOperator(kahler, theory.fermion_mass)
    normal = dirac_normal_operator(regulated)
    interval = SpectralInterval(
        normal,
        theory.normal_spectral_lower,
        theory.normal_spectral_upper,
        evidence="construction",
        scope="structural",
    )
    power = 0.25
    action_plan = plan_minimax_rational_approximation(
        power_rational_target(-power),
        num_poles=rational_poles,
        verification_points=rational_verification_points,
        error_metric="relative",
        requested_tolerance=rational_tolerance,
    )
    refresh_plan = plan_minimax_rational_approximation(
        power_rational_target(0.5 * power),
        num_poles=rational_poles,
        verification_points=rational_verification_points,
        error_metric="relative",
        requested_tolerance=rational_tolerance,
    )
    action_approximation = generate_minimax_rational_approximation(interval, action_plan)
    refresh_approximation = generate_minimax_rational_approximation(
        interval, refresh_plan
    )
    pseudofermion = FractionalPowerPseudofermionTerm(
        regulated,
        interval,
        action_approximation,
        refresh_approximation,
        determinant_power=power,
        solves=solves,
    )

    action_terms = [
        SeparableActionTerm(
            lambda value: bosonic.action(layout.unpack(value)),
            term_id=f"twisted-n2-bosonic:{bosonic.prepared_id}",
        )
    ]
    if theory.scalar_mass > 0.0:
        action_terms.append(
            SeparableActionTerm(
                lambda value: _scalar_regulator(layout, value, theory.scalar_mass),
                term_id=f"twisted-n2-scalar-regulator:{theory.plan_id}",
            )
        )
    if theory.u1_mass > 0.0:
        action_terms.append(
            SeparableActionTerm(
                lambda value: _u1_regulator(layout, value, theory.u1_mass),
                term_id=f"twisted-n2-u1-regulator:{theory.plan_id}",
            )
        )
    registry = SeparableActionRegistry(tuple(action_terms), (pseudofermion,))
    pseudofermion_index = len(action_terms)
    force_plan = NestedForcePlan(
        (
            NestedForcePartition((pseudofermion_index,), substeps=1),
            NestedForcePartition(
                tuple(range(len(action_terms))), substeps=int(bosonic_substeps)
            ),
        )
    )
    rhmc_plan = plan_rhmc(
        step_size=step_size,
        trajectory_steps=trajectory_steps,
        force_plan=force_plan,
        resources=resources,
    )
    kernel = prepare_rhmc(
        registry,
        rhmc_plan,
        coordinates,
        geometry=geometry,
        local_coordinate_shape=layout.coordinate_shape,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-twisted-n2-phase-quenched-rhmc",
            "theory": theory.plan_id,
            "bosonic": bosonic.prepared_id,
            "layout": layout.layout_id,
            "dirac": regulated.operator_id,
            "interval": interval.certificate_id,
            "action_rational": action_approximation.certificate_id,
            "refresh_rational": refresh_approximation.certificate_id,
            "pseudofermion": pseudofermion.term_id,
            "registry": registry.registry_id,
            "kernel": kernel.kernel_id,
        }
    )
    return PreparedTwistedN2RHMC(
        theory=theory,
        bosonic_action=bosonic,
        layout=layout,
        kahler_dirac=kahler,
        regulated_dirac=regulated,
        spectral_interval=interval,
        pseudofermion=pseudofermion,
        registry=registry,
        kernel=kernel,
        phase_quenched_power=power,
        prepared_id=prepared_id,
        claim="finite-regulated-phase-quenched-two-dimensional-twisted-n2-rhmc",
    )


def sample_twisted_n2_rhmc(
    prepared: PreparedTwistedN2RHMC,
    initial_configuration: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    num_draws: int,
) -> TwistedN2RHMCRun:
    if not isinstance(prepared, PreparedTwistedN2RHMC):
        raise TypeError("prepared must be PreparedTwistedN2RHMC.")
    initial = initialize_rhmc_state(prepared.kernel, initial_configuration, key=key)
    samples = sample_rhmc(prepared.kernel, initial, num_draws=num_draws)
    maximum_energy = jnp.max(jnp.abs(samples.energy_error))
    maximum_force_error = jnp.max(
        samples.maximum_force_shifted_solution_error_upper_bound
    )
    available = (
        jnp.all(samples.refresh_solve_error_bound_available)
        & jnp.all(samples.initial_term_solve_error_bound_available)
        & jnp.all(samples.proposed_term_solve_error_bound_available)
        & jnp.all(samples.force_solve_error_bound_available)
    )
    certified = available & (
        jnp.all(samples.refresh_solve_error_bound_certified)
        & jnp.all(samples.initial_term_solve_error_bound_certified)
        & jnp.all(samples.proposed_term_solve_error_bound_certified)
        & jnp.all(samples.force_solve_error_bound_certified)
    )
    finite = (
        jnp.all(jnp.isfinite(samples.configurations))
        & jnp.all(jnp.isfinite(samples.energy_error))
        & jnp.where(available, jnp.isfinite(maximum_force_error), True)
    )
    divergent_count = jnp.sum(samples.divergent.astype(jnp.int32))
    nonfinite_count = jnp.sum(samples.nonfinite.astype(jnp.int32))
    membership_count = jnp.sum(samples.membership_failure.astype(jnp.int32))
    force_failure_count = jnp.sum((~samples.force_successful).astype(jnp.int32))
    successful = (
        finite
        & (divergent_count == 0)
        & (nonfinite_count == 0)
        & (membership_count == 0)
        & (force_failure_count == 0)
    )
    evidence = TwistedN2RHMCEvidence(
        acceptance_rate=samples.acceptance_rate,
        maximum_energy_error=maximum_energy,
        divergent_count=divergent_count,
        nonfinite_count=nonfinite_count,
        membership_failure_count=membership_count,
        force_failure_count=force_failure_count,
        maximum_force_solve_error=maximum_force_error,
        solve_error_bounds_available=available,
        solve_error_bounds_certified=certified,
        finite=finite,
        successful=successful,
        sample_count=samples.num_draws,
        prepared_id=prepared.prepared_id,
        claim="finite-regulated-phase-quenched-chain-no-continuum-or-pfaffian-phase-claim",
    )
    return TwistedN2RHMCRun(
        samples=samples,
        evidence=evidence,
        prepared_id=prepared.prepared_id,
    )


__all__ = [
    "PreparedTwistedN2RHMC",
    "TwistedN2RHMCEvidence",
    "TwistedN2RHMCRun",
    "prepare_twisted_n2_rhmc",
    "sample_twisted_n2_rhmc",
]
