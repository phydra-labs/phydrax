#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._bounds import Bounds
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import (
    NonlinearTermination,
    prepare_variational_inequality,
    PreparedVariationalInequalitySolve,
    refresh_variational_inequality,
    SemismoothNewton,
    solve_prepared_variational_inequality,
    VariationalInequalityProblem,
    VariationalInequalityResult,
)


if TYPE_CHECKING:
    from ._rod_tendon import PreparedTendonRoute


def _real_vector(name: str, value: ArrayLike, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _route_identity(route: Any, /) -> str:
    identifier = getattr(route, "prepared_id", None)
    if not isinstance(identifier, str) or not identifier:
        raise TypeError(
            "route must be a PreparedTendonRoute with a nonempty prepared_id."
        )
    return identifier


def _route_span_count(route: Any, /) -> int:
    count = getattr(route, "span_count", None)
    if count is None:
        plan = getattr(route, "plan", None)
        station_count = getattr(plan, "station_count", None)
        count = None if station_count is None else station_count - 1
    if not isinstance(count, (int, np.integer)) or int(count) < 2:
        raise ValueError("A capstan route must contain at least two fixed spans.")
    return int(count)


class CapstanTendonFrictionState(StrictModule, NonTrainableState):
    """Accepted per-span tendon history for one immutable prepared route."""

    tensions: Array
    stress_free_lengths: Array
    slip: Array

    def __init__(
        self,
        tensions: ArrayLike,
        stress_free_lengths: ArrayLike,
        slip: ArrayLike,
        /,
    ):
        tension = _real_vector("tensions", tensions)
        free = _real_vector("stress_free_lengths", stress_free_lengths)
        slip_ = _real_vector("slip", slip)
        if tension.size < 2:
            raise ValueError("Capstan state must contain at least two spans.")
        if free.shape != tension.shape or slip_.shape != (tension.size - 1,):
            raise ValueError(
                "State requires one stress-free length per span and one slip per eyelet."
            )
        dtype = np.dtype(
            jax.dtypes.canonicalize_dtype(
                jnp.result_type(tension.dtype, free.dtype, slip_.dtype)
            )
        )
        with np.errstate(over="ignore", invalid="ignore"):
            tension = tension.astype(dtype, copy=False)
            free = free.astype(dtype, copy=False)
            slip_ = slip_.astype(dtype, copy=False)
        if not all(np.all(np.isfinite(value)) for value in (tension, free, slip_)):
            raise ValueError("Capstan state must be finite in the active JAX precision.")
        if np.any(tension < 0.0):
            raise ValueError("tensions must be non-negative.")
        if np.any(free <= 0.0):
            raise ValueError("stress_free_lengths must be strictly positive.")
        self.tensions = jnp.asarray(tension)
        self.stress_free_lengths = jnp.asarray(free)
        self.slip = jnp.asarray(slip_)


def _state_with_values(
    state: CapstanTendonFrictionState,
    tensions: Array,
    stress_free_lengths: Array,
    slip: Array,
    /,
) -> CapstanTendonFrictionState:
    return eqx.tree_at(
        lambda value: (
            value.tensions,
            value.stress_free_lengths,
            value.slip,
        ),
        state,
        (tensions, stress_free_lengths, slip),
    )


class _CapstanVIArguments(StrictModule):
    stress_free_lengths: Array
    span_lengths: Array
    axial_rigidity: Array
    friction_factors: Array


class CapstanTendonFrictionEvidence(StrictModule):
    """Independent complementarity, dissipation, and power acceptance evidence."""

    finite: Array
    converged: Array
    vi_certified: Array
    capstan_complementary: Array
    capstan_violation: Array
    dissipation_nonnegative: Array
    interface_dissipation: Array
    dissipation_power: Array
    rod_power: Array
    stored_energy: Array
    stored_energy_rate: Array
    power_residual: Array
    power_balanced: Array
    accepted: Array
    rollback_applied: Array


class CapstanTendonFrictionEvaluation(StrictModule):
    """Candidate VI result and atomic accepted-or-rolled-back tendon history."""

    previous_state: CapstanTendonFrictionState
    candidate_state: CapstanTendonFrictionState
    accepted_state: CapstanTendonFrictionState
    variational_inequality: VariationalInequalityResult
    directional_slip_increment: Array
    span_lengths: Array
    span_length_rates: Array
    forward_capstan_margin: Array
    reverse_capstan_margin: Array
    evidence: CapstanTendonFrictionEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.accepted


def _span_tensions(
    span_lengths: Array, stress_free_lengths: Array, axial_rigidity: Array, /
) -> Array:
    extension = jnp.maximum(span_lengths - stress_free_lengths, 0.0)
    return axial_rigidity * extension / stress_free_lengths


def _stored_energy(
    span_lengths: Array, stress_free_lengths: Array, axial_rigidity: Array, /
) -> Array:
    extension = jnp.maximum(span_lengths - stress_free_lengths, 0.0)
    return jnp.sum(0.5 * axial_rigidity * extension * extension / stress_free_lengths)


def _material_force(
    span_lengths: Array, stress_free_lengths: Array, axial_rigidity: Array, /
) -> Array:
    stretch = span_lengths / stress_free_lengths
    return jnp.where(
        span_lengths > stress_free_lengths,
        0.5 * axial_rigidity * (stretch * stretch - 1.0),
        0.0,
    )


def _free_length_increment(directional_increment: Array, span_count: int, /) -> Array:
    net = directional_increment[0] - directional_increment[1]
    increment = jnp.zeros((span_count,), dtype=directional_increment.dtype)
    increment = increment.at[:-1].add(-net)
    return increment.at[1:].add(net)


def _slip_upper_bounds(
    stress_free_lengths: Array, maximum_slip_fraction: float, /
) -> Array:
    upper = maximum_slip_fraction * jnp.minimum(
        stress_free_lengths[:-1], stress_free_lengths[1:]
    )
    return eqx.error_if(
        upper,
        ~jnp.all(jnp.isfinite(upper)) | jnp.any(upper <= 0.0),
        "stress_free_lengths must yield finite positive capstan slip bounds.",
    )


def _vi_operator(
    directional_increment: Array, arguments: _CapstanVIArguments, /
) -> Array:
    span_count = arguments.stress_free_lengths.shape[0]
    free = arguments.stress_free_lengths + _free_length_increment(
        directional_increment, span_count
    )
    tension = _span_tensions(arguments.span_lengths, free, arguments.axial_rigidity)
    forward = arguments.friction_factors * tension[:-1] - tension[1:]
    reverse = arguments.friction_factors * tension[1:] - tension[:-1]
    return jnp.stack((forward, reverse))


class CapstanTendonFrictionPlan(StrictModule, NonTrainableState):
    """Fixed-route elastic capstan calibration solved as a nonnegative VI.

    The two nonnegative variables at every interior eyelet represent opposing
    material-slip directions.  Tensions remain physical variables; in
    particular, no logarithm is taken when a span is slack.
    """

    friction_coefficients: Array
    wrap_angles: Array
    axial_rigidity: Array
    friction_factors: Array
    method: SemismoothNewton
    termination: NonlinearTermination
    maximum_slip_fraction: float = eqx.field(static=True)
    dissipation_tolerance: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        friction_coefficients: ArrayLike,
        wrap_angles: ArrayLike,
        axial_rigidity: ArrayLike,
        /,
        *,
        maximum_slip_fraction: float = 0.2,
        dissipation_tolerance: float = 1.0e-8,
        power_tolerance: float = 1.0e-8,
        method: SemismoothNewton | None = None,
        termination: NonlinearTermination | None = None,
        plan_id: str | None = None,
    ):
        friction = _real_vector("friction_coefficients", friction_coefficients)
        angles = _real_vector("wrap_angles", wrap_angles)
        rigidity = _real_vector("axial_rigidity", axial_rigidity)
        if friction.size < 1 or angles.shape != friction.shape:
            raise ValueError(
                "friction_coefficients and wrap_angles must identify every eyelet."
            )
        if rigidity.shape != (friction.size + 1,):
            raise ValueError("axial_rigidity must identify every tendon span.")
        if np.any(friction < 0.0) or np.any(angles < 0.0):
            raise ValueError(
                "Friction coefficients and wrap angles must be non-negative."
            )
        if np.any(rigidity <= 0.0):
            raise ValueError("axial_rigidity must be strictly positive.")
        slip_fraction = float(maximum_slip_fraction)
        dissipation = float(dissipation_tolerance)
        power = float(power_tolerance)
        if not isfinite(slip_fraction) or not 0.0 < slip_fraction < 0.5:
            raise ValueError(
                "maximum_slip_fraction must lie strictly between zero and 0.5."
            )
        if (
            not isfinite(dissipation)
            or dissipation < 0.0
            or not isfinite(power)
            or power < 0.0
        ):
            raise ValueError("Evidence tolerances must be finite and non-negative.")
        method_ = (
            SemismoothNewton(
                formulation="fischer-burmeister",
                feasibility="preserve-box",
            )
            if method is None
            else method
        )
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(method_, SemismoothNewton):
            raise TypeError("method must be SemismoothNewton or None.")
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        if method_.feasibility != "preserve-box":
            raise ValueError("Capstan VI method must preserve its nonnegative box.")
        dtype = np.dtype(
            jax.dtypes.canonicalize_dtype(
                jnp.result_type(friction.dtype, angles.dtype, rigidity.dtype)
            )
        )
        with np.errstate(over="ignore", invalid="ignore"):
            arrays = {
                "friction_coefficients": friction.astype(dtype, copy=False),
                "wrap_angles": angles.astype(dtype, copy=False),
                "axial_rigidity": rigidity.astype(dtype, copy=False),
            }
            factors = np.exp(arrays["friction_coefficients"] * arrays["wrap_angles"])
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError(
                "Capstan calibration is not finite in the active JAX precision."
            )
        if not np.all(np.isfinite(factors)):
            raise ValueError(
                "friction_coefficients times wrap_angles must yield finite "
                "capstan factors in the active JAX precision."
            )
        generated = canonical_fingerprint(
            {
                "kind": "fixed-route-capstan-tendon-friction",
                "calibration": array_tree_fingerprint(arrays),
                "maximum_slip_fraction": slip_fraction,
                "dissipation_tolerance": dissipation,
                "power_tolerance": power,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.friction_coefficients = jnp.asarray(arrays["friction_coefficients"])
        self.wrap_angles = jnp.asarray(arrays["wrap_angles"])
        self.axial_rigidity = jnp.asarray(arrays["axial_rigidity"])
        self.friction_factors = jnp.asarray(factors)
        self.maximum_slip_fraction = slip_fraction
        self.dissipation_tolerance = dissipation
        self.power_tolerance = power
        self.method = method_
        self.termination = termination_
        self.plan_id = identifier

    @property
    def span_count(self) -> int:
        return int(self.axial_rigidity.shape[0])

    @property
    def eyelet_count(self) -> int:
        return int(self.friction_coefficients.shape[0])

    def prepare(
        self,
        route: PreparedTendonRoute,
        initial_state: CapstanTendonFrictionState,
        /,
    ) -> PreparedCapstanTendonFriction:
        """Bind calibration and fixed VI work topology to one prepared route."""
        route_id = _route_identity(route)
        if _route_span_count(route) != self.span_count:
            raise ValueError("Route and capstan calibration span counts must match.")
        if not isinstance(initial_state, CapstanTendonFrictionState):
            raise TypeError("initial_state must be CapstanTendonFrictionState.")
        if initial_state.tensions.shape != (self.span_count,):
            raise ValueError("Initial state and capstan calibration sizes must match.")
        arguments = _CapstanVIArguments(
            initial_state.stress_free_lengths,
            initial_state.stress_free_lengths,
            self.axial_rigidity,
            self.friction_factors,
        )
        upper = _slip_upper_bounds(
            initial_state.stress_free_lengths, self.maximum_slip_fraction
        )
        problem = VariationalInequalityProblem(
            _vi_operator,
            Bounds(
                jnp.zeros((2, self.eyelet_count)),
                jnp.broadcast_to(upper, (2, self.eyelet_count)),
            ),
            problem_id=f"{self.plan_id}/capstan-slip",
        )
        vi = prepare_variational_inequality(
            problem,
            jnp.zeros((2, self.eyelet_count), dtype=initial_state.tensions.dtype),
            method=self.method,
            termination=self.termination,
            args=arguments,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-route-capstan-tendon-friction",
                "plan": self.plan_id,
                "route": route_id,
                "vi_topology": vi.topology_id,
            }
        )
        return PreparedCapstanTendonFriction(
            self,
            route,
            vi,
            route_id=route_id,
            prepared_id=prepared_id,
        )


class PreparedCapstanTendonFriction(StrictModule, NonTrainableState):
    """Prepared fixed-work capstan solve bound to one immutable tendon route."""

    plan: CapstanTendonFrictionPlan
    route: Any
    variational_inequality: PreparedVariationalInequalitySolve
    route_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CapstanTendonFrictionPlan,
        route: PreparedTendonRoute,
        variational_inequality: PreparedVariationalInequalitySolve,
        /,
        *,
        route_id: str,
        prepared_id: str,
    ):
        self.plan = plan
        self.route = route
        self.variational_inequality = variational_inequality
        self.route_id = route_id
        self.prepared_id = prepared_id

    def evaluate(
        self,
        state: CapstanTendonFrictionState,
        span_lengths: ArrayLike,
        span_length_rates: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> CapstanTendonFrictionEvaluation:
        """Solve one fixed-route capstan update and atomically accept or roll back.

        ``span_lengths`` and ``span_length_rates`` are the ordered per-span
        kinematics of the bound prepared route.  Keeping those kinematics
        explicit separates this constitutive VI from native or reduced rod
        state representations.
        """
        if not isinstance(state, CapstanTendonFrictionState):
            raise TypeError("state must be CapstanTendonFrictionState.")
        span_count = self.plan.span_count
        if state.tensions.shape != (span_count,):
            raise ValueError("State and prepared capstan span counts must match.")
        lengths = jnp.asarray(span_lengths, dtype=state.tensions.dtype)
        rates = jnp.asarray(span_length_rates, dtype=state.tensions.dtype)
        dt = jnp.asarray(step_size, dtype=state.tensions.dtype)
        if lengths.shape != (span_count,) or rates.shape != (span_count,):
            raise ValueError(
                "span_lengths and span_length_rates must identify every span."
            )
        if dt.shape != ():
            raise ValueError("step_size must be scalar.")
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0),
            "step_size must be finite and positive.",
        )
        lengths = eqx.error_if(
            lengths,
            ~jnp.all(jnp.isfinite(lengths)) | jnp.any(lengths < 0.0),
            "span_lengths must be finite and non-negative.",
        )
        rates = eqx.error_if(
            rates,
            ~jnp.all(jnp.isfinite(rates)),
            "span_length_rates must be finite.",
        )

        factors = self.plan.friction_factors
        arguments = _CapstanVIArguments(
            state.stress_free_lengths,
            lengths,
            self.plan.axial_rigidity,
            factors,
        )
        upper = _slip_upper_bounds(
            state.stress_free_lengths, self.plan.maximum_slip_fraction
        )
        bounds = eqx.tree_at(
            lambda value: value.upper,
            self.variational_inequality.problem.bounds,
            jnp.broadcast_to(upper, (2, self.plan.eyelet_count)),
        )
        problem = VariationalInequalityProblem(
            _vi_operator,
            bounds,
            problem_id=self.variational_inequality.problem.problem_id,
        )
        refreshed = refresh_variational_inequality(
            self.variational_inequality,
            problem,
            jnp.zeros((2, self.plan.eyelet_count), dtype=lengths.dtype),
            args=arguments,
        )
        vi_result = solve_prepared_variational_inequality(refreshed)
        directional = vi_result.state
        free_increment = _free_length_increment(directional, span_count)
        candidate_free = state.stress_free_lengths + free_increment
        candidate_tension = _span_tensions(
            lengths, candidate_free, self.plan.axial_rigidity
        )
        net_slip = directional[0] - directional[1]
        candidate = _state_with_values(
            state,
            candidate_tension,
            candidate_free,
            state.slip + net_slip,
        )

        forward_margin = factors * candidate_tension[:-1] - candidate_tension[1:]
        reverse_margin = factors * candidate_tension[1:] - candidate_tension[:-1]
        capstan_violation = jnp.max(
            jnp.maximum(-jnp.concatenate((forward_margin, reverse_margin)), 0.0),
            initial=0.0,
        )
        capstan_scale = jnp.maximum(jnp.max(candidate_tension, initial=0.0), 1.0)
        capstan_complementary = vi_result.certificate.certified & (
            capstan_violation <= self.plan.method.certification_tolerance * capstan_scale
        )

        free_rate = free_increment / dt
        material_force = _material_force(
            lengths, candidate_free, self.plan.axial_rigidity
        )
        interface_dissipation = (material_force[1:] - material_force[:-1]) * (
            net_slip / dt
        )
        dissipation_power = jnp.sum(interface_dissipation)
        rod_power = -jnp.sum(candidate_tension * rates)
        stored_energy = _stored_energy(lengths, candidate_free, self.plan.axial_rigidity)
        stored_energy_rate = -rod_power - jnp.sum(material_force * free_rate)
        power_residual = stored_energy_rate + rod_power + dissipation_power
        power_scale = jnp.maximum(
            jnp.maximum(jnp.abs(stored_energy_rate), jnp.abs(rod_power)),
            jnp.maximum(jnp.abs(dissipation_power), 1.0),
        )
        finite = (
            jnp.all(jnp.isfinite(candidate.tensions))
            & jnp.all(jnp.isfinite(candidate.stress_free_lengths))
            & jnp.all(jnp.isfinite(candidate.slip))
            & jnp.all(candidate.stress_free_lengths > 0.0)
            & jnp.all(jnp.isfinite(interface_dissipation))
            & jnp.isfinite(stored_energy)
            & jnp.isfinite(power_residual)
        )
        dissipation_nonnegative = (
            dissipation_power >= -self.plan.dissipation_tolerance * power_scale
        )
        power_closed = jnp.abs(power_residual) <= self.plan.power_tolerance * power_scale
        accepted = (
            vi_result.successful
            & capstan_complementary
            & finite
            & dissipation_nonnegative
            & power_closed
        )
        accepted_state = _state_with_values(
            state,
            jnp.where(accepted, candidate.tensions, state.tensions),
            jnp.where(
                accepted,
                candidate.stress_free_lengths,
                state.stress_free_lengths,
            ),
            jnp.where(accepted, candidate.slip, state.slip),
        )
        evidence = CapstanTendonFrictionEvidence(
            finite,
            vi_result.successful,
            vi_result.certificate.certified,
            capstan_complementary,
            capstan_violation,
            dissipation_nonnegative,
            interface_dissipation,
            dissipation_power,
            rod_power,
            stored_energy,
            stored_energy_rate,
            power_residual,
            power_closed,
            accepted,
            ~accepted,
        )
        return CapstanTendonFrictionEvaluation(
            state,
            candidate,
            accepted_state,
            vi_result,
            directional,
            lengths,
            rates,
            forward_margin,
            reverse_margin,
            evidence,
            self.prepared_id,
        )


__all__ = [
    "CapstanTendonFrictionEvaluation",
    "CapstanTendonFrictionEvidence",
    "CapstanTendonFrictionPlan",
    "CapstanTendonFrictionState",
    "PreparedCapstanTendonFriction",
]
