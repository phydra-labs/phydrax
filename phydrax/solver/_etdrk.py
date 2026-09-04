#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle
from ..discretization.spectral import HermitianSpectralCoordinates
from ..linalg import DiagonalLinearOperator
from ..linalg._matrix_functions import _phi_function_value
from ._differential import DifferentialSolution
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ._semilinear_drift import SemilinearDrift
from ._temporal_method import TemporalMethodCapabilities


if TYPE_CHECKING:
    from ..equations._periodic_les import PeriodicLESStepRestriction


def _etdrk_update(
    order: Literal[2, 4],
    drift: SemilinearDrift,
    diagonal: Array,
    time: Array,
    state: Array,
    step_size: Array,
    args: Any,
    stage_forcing: tuple[Array, ...] | None = None,
    first_nonlinear: Array | None = None,
    /,
) -> Array:
    expected_stages = 2 if order == 2 else 4
    if stage_forcing is not None and len(stage_forcing) != expected_stages:
        raise ValueError(
            f"ETDRK{order} additive forcing requires {expected_stages} stage values."
        )

    def nonlinear(stage: int, stage_time: Array, stage_state: Array) -> Array:
        if stage == 0 and first_nonlinear is not None:
            value = jnp.asarray(first_nonlinear, dtype=stage_state.dtype)
            if value.shape != state.shape:
                raise ValueError(
                    "Precomputed ETDRK first nonlinear value has an incompatible shape."
                )
        else:
            value = drift.nonlinear(stage_time, stage_state, args)
        if stage_forcing is None:
            return value
        forcing = jnp.asarray(stage_forcing[stage], dtype=stage_state.dtype)
        if forcing.shape != state.shape:
            raise ValueError("ETDRK additive stage forcing has an incompatible shape.")
        return value + forcing

    z = step_size * diagonal.astype(state.dtype)
    exponential = jnp.exp(z)
    n1 = nonlinear(0, time, state)
    if order == 2:
        predictor = exponential * state + step_size * _phi_function_value(z, 1) * n1
        n2 = nonlinear(1, time + step_size, predictor)
        return predictor + step_size * _phi_function_value(z, 2) * (n2 - n1)
    half_exponential = jnp.exp(0.5 * z)
    q = 0.5 * _phi_function_value(0.5 * z, 1)
    a = half_exponential * state + step_size * q * n1
    n2 = nonlinear(1, time + 0.5 * step_size, a)
    b = half_exponential * state + step_size * q * n2
    n3 = nonlinear(2, time + 0.5 * step_size, b)
    c = half_exponential * a + step_size * q * (2.0 * n3 - n1)
    n4 = nonlinear(3, time + step_size, c)
    phi1 = _phi_function_value(z, 1)
    phi2 = _phi_function_value(z, 2)
    phi3 = _phi_function_value(z, 3)
    f1 = phi1 - 3.0 * phi2 + 4.0 * phi3
    f2 = phi2 - 2.0 * phi3
    f3 = 4.0 * phi3 - phi2
    return exponential * state + step_size * (f1 * n1 + 2.0 * f2 * (n2 + n3) + f3 * n4)


class ETDRKMethod(StrictModule, NonTrainableState):
    """Fixed-step diagonal exponential time-differencing Runge--Kutta method."""

    order: Literal[2, 4] = eqx.field(static=True)
    capabilities: TemporalMethodCapabilities
    method_id: str = eqx.field(static=True)

    def __init__(self, order: Literal[2, 4] = 4):
        order_ = int(order)
        if order_ not in (2, 4):
            raise ValueError("ETDRK order must be two or four.")
        identifier = canonical_fingerprint(
            {"kind": "etdrk-method", "order": order_, "linear_path": "diagonal"}
        )
        self.order = order_
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("additive-ode",),
            method_class="exponential",
            order=order_,
            adaptive=False,
            history_depth=1,
            stage_abscissae=(0.0, 1.0) if order_ == 2 else (0.0, 0.5, 0.5, 1.0),
            causal_stage_extent=1.0,
            noise_requirement="none",
            method_id=identifier,
        )
        self.method_id = identifier

    def _diagonal(self, drift: SemilinearDrift, /) -> Array:
        operator = drift.linear_operator
        if not isinstance(operator, DiagonalLinearOperator) or operator.batch_shape:
            raise ValueError(
                "ETDRK initially requires an unbatched DiagonalLinearOperator."
            )
        return operator.diagonal.reshape(drift.state_shape)

    def prepare(
        self,
        drift: SemilinearDrift,
        /,
        *,
        coordinates: HermitianSpectralCoordinates | None = None,
    ) -> PreparedETDRKMethod:
        """Bind the complete drift and optional real-field boundary contract."""
        if not isinstance(drift, SemilinearDrift):
            raise TypeError("drift must be a SemilinearDrift.")
        if coordinates is not None:
            if not isinstance(coordinates, HermitianSpectralCoordinates):
                raise TypeError(
                    "coordinates must be HermitianSpectralCoordinates or None."
                )
            if coordinates.state_shape != drift.state_shape:
                raise ValueError(
                    "Hermitian coordinates and semilinear drift state shapes differ."
                )
        return PreparedETDRKMethod(
            self,
            drift,
            self._diagonal(drift),
            coordinates,
        )

    def step(
        self,
        drift: SemilinearDrift,
        time: ArrayLike,
        state: ArrayLike,
        dt: ArrayLike,
        args: Any = None,
        /,
        *,
        coordinates: HermitianSpectralCoordinates | None = None,
    ) -> Array:
        """Advance through the same prepared fixed-step path used by rollouts."""
        prepared = self.prepare(drift, coordinates=coordinates)
        result = prepared.step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(time),
            jnp.asarray(state),
            jnp.asarray(dt),
            args,
        )
        return result.accepted_state


class PreparedETDRKMethod(AbstractFixedStepMethod):
    """ETDRK method bound to one complete semilinear drift identity."""

    drift: SemilinearDrift
    diagonal: Array
    coordinates: HermitianSpectralCoordinates | None
    capabilities: TemporalMethodCapabilities
    order: Literal[2, 4] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: ETDRKMethod,
        drift: SemilinearDrift,
        diagonal: Array,
        coordinates: HermitianSpectralCoordinates | None,
        /,
    ):
        self.drift = drift
        self.diagonal = jnp.asarray(diagonal)
        self.coordinates = coordinates
        self.capabilities = method.capabilities
        self.order = method.order
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-etdrk-method-v1",
                "method": method.method_id,
                "drift": drift.drift_id,
                "coordinates": None if coordinates is None else coordinates.coordinate_id,
                "live_state": "full-complex"
                if coordinates is not None
                else "declared-array",
                "acceptance": "finite-hermitian" if coordinates is not None else "finite",
            }
        )

    def _validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.drift.state_shape:
            raise ValueError(
                f"ETDRK state must have shape {self.drift.state_shape}; "
                f"got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("ETDRK state must have an inexact dtype.")
        if self.coordinates is not None:
            value = self.coordinates.validate_state(value)
        return value

    def _boundary_evidence(self, state: Array, /) -> tuple[Array, Array, Array]:
        finite = jnp.all(jnp.isfinite(state))
        if self.coordinates is None:
            return state, finite, jnp.zeros((), dtype=state.real.dtype)
        defect = self.coordinates.reality_defect(state)
        projected = self.coordinates.project(state)
        valid = finite & (defect <= self.coordinates.reality_tolerance)
        return projected, valid, defect

    def _step_with_first_nonlinear(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        first_nonlinear: Array | None,
        /,
    ) -> FixedStepResult:
        del step_index
        value = self._validate_state(state)
        step = jnp.asarray(step_size, dtype=value.real.dtype).reshape(())
        step = eqx.error_if(
            step,
            ~(jnp.isfinite(step) & (step > 0)),
            "ETDRK step size must be finite and positive.",
        )
        start = jnp.asarray(time, dtype=step.dtype).reshape(())
        _, incoming_valid, incoming_defect = self._boundary_evidence(value)
        candidate = _etdrk_update(
            self.order,
            self.drift,
            self.diagonal,
            start,
            value,
            step,
            args,
            None,
            first_nonlinear,
        )
        projected, candidate_valid, candidate_defect = self._boundary_evidence(candidate)
        successful = incoming_valid & candidate_valid
        accepted = jnp.where(successful, projected, value)
        correction = projected - candidate
        correction_norm = jnp.sqrt(jnp.sum(jnp.real(correction * jnp.conj(correction))))
        finite_transition = jnp.all(jnp.isfinite(value)) & jnp.all(
            jnp.isfinite(candidate)
        )
        residual = jnp.where(
            finite_transition,
            jnp.maximum(incoming_defect, candidate_defect),
            jnp.asarray(jnp.inf, dtype=value.real.dtype),
        )
        return FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=successful,
            residual=residual,
            iterations=jnp.asarray(0, dtype=jnp.int32),
            work=jnp.asarray(self.order, dtype=jnp.int32),
            transform_applied=successful & (correction_norm > 0.0),
            transform_correction_norm=jnp.where(
                successful,
                correction_norm,
                jnp.zeros((), dtype=correction_norm.dtype),
            ),
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        return self._step_with_first_nonlinear(
            step_index,
            time,
            state,
            step_size,
            args,
            None,
        )


class LESStabilityGuardedETDRKMethod(StrictModule, NonTrainableState):
    """ETDRK plan with an explicit current-state periodic LES stability guard."""

    base_method: ETDRKMethod
    capabilities: TemporalMethodCapabilities
    safety_factor: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        base_method: ETDRKMethod,
        /,
        *,
        safety_factor: float,
    ):
        if not isinstance(base_method, ETDRKMethod):
            raise TypeError("base_method must be an ETDRKMethod.")
        safety = float(safety_factor)
        if not np.isfinite(safety) or not 0.0 < safety <= 1.0:
            raise ValueError("safety_factor must be finite and lie in (0, 1].")
        self.base_method = base_method
        self.capabilities = base_method.capabilities
        self.safety_factor = safety
        self.method_id = canonical_fingerprint(
            {
                "kind": "les-stability-guarded-etdrk-method",
                "base_method": base_method.method_id,
                "safety_factor": safety,
                "restriction": "periodic-algebraic-les-current-state",
            }
        )

    def prepare(
        self,
        dynamics: Any,
        /,
        *,
        coordinates: HermitianSpectralCoordinates,
    ) -> PreparedLESStabilityGuardedETDRKMethod:
        from ..equations._incompressible import (
            CompiledIncompressibleSpectralDynamics,
        )

        if not isinstance(dynamics, CompiledIncompressibleSpectralDynamics):
            raise TypeError("dynamics must be CompiledIncompressibleSpectralDynamics.")
        if dynamics.algebraic_les is None:
            raise ValueError(
                "LES-stability-guarded ETDRK requires compiled algebraic LES."
            )
        if not isinstance(coordinates, HermitianSpectralCoordinates):
            raise TypeError(
                "coordinates must be HermitianSpectralCoordinates for LES ETDRK."
            )
        if (
            coordinates.discretization.prepared_id != dynamics.discretization.prepared_id
            or coordinates.state_shape != dynamics.state_shape
        ):
            raise ValueError(
                "LES ETDRK coordinates must bind the compiled spectral "
                "discretization and velocity state shape."
            )
        base = self.base_method.prepare(
            dynamics.semilinear_drift,
            coordinates=coordinates,
        )
        return PreparedLESStabilityGuardedETDRKMethod(self, base, dynamics)

    def step(
        self,
        dynamics: Any,
        time: ArrayLike,
        state: ArrayLike,
        dt: ArrayLike,
        args: Any = None,
        /,
        *,
        coordinates: HermitianSpectralCoordinates,
    ) -> Array:
        prepared = self.prepare(dynamics, coordinates=coordinates)
        result = prepared.step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(time),
            jnp.asarray(state),
            jnp.asarray(dt),
            args,
        )
        return result.accepted_state


class PreparedLESStabilityGuardedETDRKMethod(AbstractFixedStepMethod):
    """Prepared ETDRK transition guarded by one bound periodic LES realization."""

    plan: LESStabilityGuardedETDRKMethod
    base_method: PreparedETDRKMethod
    dynamics: Any
    capabilities: TemporalMethodCapabilities
    order: Literal[2, 4] = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LESStabilityGuardedETDRKMethod,
        base_method: PreparedETDRKMethod,
        dynamics: Any,
        /,
    ):
        from ..equations._incompressible import (
            CompiledIncompressibleSpectralDynamics,
        )

        if not isinstance(plan, LESStabilityGuardedETDRKMethod):
            raise TypeError("plan must be a LESStabilityGuardedETDRKMethod.")
        if not isinstance(base_method, PreparedETDRKMethod):
            raise TypeError("base_method must be a PreparedETDRKMethod.")
        if not isinstance(dynamics, CompiledIncompressibleSpectralDynamics):
            raise TypeError("dynamics must be CompiledIncompressibleSpectralDynamics.")
        if dynamics.algebraic_les is None:
            raise ValueError("Prepared guarded ETDRK requires algebraic LES.")
        if base_method.drift.drift_id != dynamics.semilinear_drift.drift_id:
            raise ValueError("Prepared ETDRK drift and compiled LES dynamics disagree.")
        self.plan = plan
        self.base_method = base_method
        self.dynamics = dynamics
        self.capabilities = base_method.capabilities
        self.order = base_method.order
        self.safety_factor = plan.safety_factor
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-les-stability-guarded-etdrk-method",
                "plan": plan.method_id,
                "base_method": base_method.method_id,
                "compiled_dynamics": dynamics.compilation_id,
                "prepared_les": dynamics.algebraic_les.prepared_id,
                "safety_factor": plan.safety_factor,
                "first_stage": "precomputed-nonlinear-reused",
            }
        )

    def step_restriction(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> PeriodicLESStepRestriction:
        value = self.base_method._validate_state(state)
        stage = self.dynamics.stage(jnp.asarray(time), value, args)
        return self.dynamics.step_restriction(
            value,
            algebraic_les_stage=stage.algebraic_les,
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        value = self.base_method._validate_state(state)
        step = jnp.asarray(step_size, dtype=value.real.dtype).reshape(())
        step = eqx.error_if(
            step,
            ~(jnp.isfinite(step) & (step > 0.0)),
            "ETDRK step size must be finite and positive.",
        )
        start = jnp.asarray(time, dtype=step.dtype).reshape(())
        stage = self.dynamics.stage(start, value, args)
        restriction = self.dynamics.step_restriction(
            value,
            algebraic_les_stage=stage.algebraic_les,
        )
        first_nonlinear = stage.rates.nonlinear_rate
        selected = restriction.etdrk_selected.astype(step.dtype)
        allowed = jnp.asarray(self.safety_factor, dtype=step.dtype) * selected
        first_nonlinear_finite = jnp.all(jnp.isfinite(first_nonlinear))
        transition_finite = restriction.finite & first_nonlinear_finite
        stable = transition_finite & (allowed > 0.0) & (step <= allowed)

        def advance(_: None) -> FixedStepResult:
            return self.base_method._step_with_first_nonlinear(
                step_index,
                start,
                value,
                step,
                args,
                first_nonlinear,
            )

        def reject(_: None) -> FixedStepResult:
            finite_limit = jnp.isfinite(allowed) & (allowed > 0.0)
            safe_limit = jnp.where(finite_limit, allowed, jnp.ones_like(allowed))
            violation = jnp.maximum(step / safe_limit - 1.0, 0.0)
            residual = jnp.where(
                transition_finite,
                jnp.where(finite_limit, violation, jnp.zeros_like(violation)),
                jnp.asarray(jnp.inf, dtype=step.dtype),
            )
            return FixedStepResult(
                candidate_state=value,
                accepted_state=value,
                successful=jnp.asarray(False),
                residual=residual,
                iterations=jnp.asarray(0, dtype=jnp.int32),
                work=jnp.asarray(1, dtype=jnp.int32),
                transform_applied=jnp.asarray(False),
                transform_correction_norm=jnp.zeros((), dtype=step.dtype),
            )

        return jax.lax.cond(stable, advance, reject, operand=None)


def solve_etdrk(
    method: ETDRKMethod,
    drift: SemilinearDrift,
    initial_state: ArrayLike,
    times: ArrayLike,
    /,
    *,
    args: Any = None,
    coordinates: HermitianSpectralCoordinates | None = None,
    discretization_bundle: DiscretizationBundle | None = None,
    problem_id: str | None = None,
) -> DifferentialSolution:
    """Integrate through the prepared ETDRK fixed-step transition."""
    if not isinstance(method, ETDRKMethod):
        raise TypeError("method must be an ETDRKMethod.")
    if not isinstance(drift, SemilinearDrift):
        raise TypeError("drift must be a SemilinearDrift.")
    saved = jnp.asarray(times)
    if jnp.iscomplexobj(saved):
        raise TypeError("ETDRK times must be real-valued.")
    if saved.ndim != 1 or saved.size < 2:
        raise ValueError("ETDRK times must be a rank-one grid with at least two values.")
    saved_host = np.asarray(saved, dtype=float)
    if np.any(~np.isfinite(saved_host)) or np.any(np.diff(saved_host) <= 0.0):
        raise ValueError("ETDRK times must be finite and strictly increasing.")
    prepared = method.prepare(drift, coordinates=coordinates)
    initial = prepared._validate_state(initial_state)
    projected_initial, initial_valid, _ = prepared._boundary_evidence(initial)
    accepted_initial = jnp.where(initial_valid, projected_initial, initial)
    starts = saved[:-1]
    durations = jnp.diff(saved)

    def advance(
        carry: tuple[Array, Array],
        data: tuple[Array, Array, Array],
    ):
        state, cumulative_valid = carry
        step_index, time, duration = data
        result = prepared.step(step_index, time, state, duration, args)
        following = jnp.where(
            cumulative_valid,
            result.accepted_state,
            state,
        )
        valid = cumulative_valid & result.successful
        return (following, valid), (following, valid)

    indices = jnp.arange(int(saved.size) - 1, dtype=jnp.int32)
    _, (advanced, advanced_valid) = jax.lax.scan(
        advance,
        (accepted_initial, initial_valid),
        (indices, starts, durations),
    )
    states = jnp.concatenate((accepted_initial[None], advanced), axis=0)
    valid = jnp.concatenate((initial_valid[None], advanced_valid), axis=0)
    name = f"ETDRK{method.order}"
    return DifferentialSolution(
        times=saved,
        states=states,
        valid=valid,
        backend_result=prepared,
        stats={"num_steps": int(saved.size - 1), "order": method.order},
        solver_name=name,
        interpretation="ito",
        solver_id=f"solver:etdrk:{method.order}",
        resolved_method=f"etdrk{method.order}:diagonal",
        discretization_bundle=discretization_bundle,
        backend_successful=jnp.all(valid),
        problem_id=problem_id,
    )


__all__ = [
    "ETDRKMethod",
    "LESStabilityGuardedETDRKMethod",
    "PreparedETDRKMethod",
    "PreparedLESStabilityGuardedETDRKMethod",
    "solve_etdrk",
]
