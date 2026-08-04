#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil, isfinite
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ..stochastic import WienerRealization
from ._differential import DifferentialSolution
from ._matrix_functions import (
    matrix_exponential_action,
    matrix_phi1_action,
    MatrixFunctionPolicy,
)
from ._noise import SpatialNoiseBasis
from ._spde import SemidiscreteSPDE


SemilinearFallback: TypeAlias = Literal["diffrax", "error"]


def _stochastic_convolution_time_factor(
    linear_eigenvalues: Array,
    step: Array,
    /,
) -> Array:
    values = jnp.asarray(linear_eigenvalues)
    h = jnp.asarray(step, dtype=values.dtype)
    threshold = jnp.sqrt(jnp.finfo(values.dtype).eps)
    safe = jnp.where(jnp.abs(values) > threshold, values, 1.0)
    quotient = jnp.expm1(2.0 * values * h) / (2.0 * safe)
    series = h + values * h**2 + (2.0 / 3.0) * values**2 * h**3
    return jnp.where(jnp.abs(values) > threshold, quotient, series)


def exact_modal_stochastic_convolution(
    noise_basis: SpatialNoiseBasis,
    linear_eigenvalues: ArrayLike,
    step: ArrayLike,
    standard_normal: ArrayLike,
    /,
) -> Array:
    """Sample one exactly filtered additive stochastic convolution increment."""
    if not isinstance(noise_basis, SpatialNoiseBasis):
        raise TypeError("noise_basis must be a SpatialNoiseBasis.")
    eigenvalues = jnp.asarray(linear_eigenvalues, dtype=float).reshape((-1,))
    normal = jnp.asarray(standard_normal, dtype=float).reshape((-1,))
    expected = (noise_basis.rank,)
    if eigenvalues.shape != expected or normal.shape != expected:
        raise ValueError(
            "linear_eigenvalues and standard_normal must contain one value per "
            f"noise mode; expected {expected}."
        )
    factor = _stochastic_convolution_time_factor(eigenvalues, jnp.asarray(step))
    variances = noise_basis.eigenvalues * factor
    tolerance = 100.0 * jnp.finfo(variances.dtype).eps
    variances = jnp.where(
        (variances < 0.0) & (variances > -tolerance),
        0.0,
        variances,
    )
    variances = jax.lax.cond(
        jnp.any(variances < 0.0),
        lambda value: jnp.full_like(value, jnp.nan),
        lambda value: value,
        variances,
    )
    coefficients = jnp.sqrt(variances) * normal
    return jnp.tensordot(noise_basis.modes, coefficients, axes=((-1,), (0,)))


def _step_schedule(
    start: float,
    end: float,
    save_times: ArrayLike,
    max_step: float,
    /,
) -> tuple[Array, Array, Array]:
    saved = np.asarray(save_times, dtype=float)
    if saved.ndim != 1 or saved.size <= 0:
        raise ValueError("save_times must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(saved)) or np.any(np.diff(saved) <= 0.0):
        raise ValueError("save_times must be finite and strictly increasing.")
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, abs(start), abs(end))
    if float(saved[0]) < start - tolerance or float(saved[-1]) > end + tolerance:
        raise ValueError("save_times must lie within the differential problem interval.")
    current = float(start)
    steps: list[float] = []
    save_indices: list[int] = []
    for target in saved:
        target_value = float(target)
        interval = target_value - current
        if interval < -tolerance:
            raise ValueError("save_times cannot precede the current integration time.")
        if abs(interval) <= tolerance:
            save_indices.append(len(steps))
            current = target_value
            continue
        count = max(1, int(ceil(interval / max_step)))
        local_step = interval / float(count)
        steps.extend((local_step,) * count)
        current = target_value
        save_indices.append(len(steps))
    return (
        jnp.asarray(steps, dtype=float),
        jnp.asarray(save_indices, dtype=jnp.int32),
        jnp.asarray(saved, dtype=float),
    )


def _unsupported_reason(
    spde: SemidiscreteSPDE,
    realization: WienerRealization | None,
    /,
) -> str | None:
    drift = spde.semilinear_drift
    if drift is None:
        return "the semidiscrete problem has no explicit semilinear decomposition"
    problem = spde.problem
    if not problem.stochastic:
        if realization is not None:
            return "deterministic problems do not accept a Wiener realization"
        return None
    if problem.interpretation != "ito":
        return "the specialized solver currently supports only Itô equations"
    if not problem.additive_noise:
        return "the specialized solver currently supports only additive noise"
    if spde.noise_basis is None:
        return "additive stochastic convolution requires a SpatialNoiseBasis"
    if realization is None:
        return "stochastic problems require a Wiener realization"
    if realization.noise_shape != problem.noise_shape:
        return "the Wiener realization noise shape does not match the problem"
    if realization.noise_id != problem.noise_id:
        return "the Wiener realization noise identity does not match the problem"
    if realization.support[0] > float(problem.t0) or realization.support[1] < float(
        problem.t1
    ):
        return "the Wiener realization support does not cover the problem interval"
    if (
        drift.compatible_noise_eigenvalues is None
        or drift.compatible_noise_basis_id != spde.noise_basis.basis_id
        or drift.compatible_noise_eigenvalues.shape != (spde.noise_basis.rank,)
    ):
        return "the linear operator and additive noise basis do not share declared modes"
    return None


def _solve_diffrax_fallback(
    spde: SemidiscreteSPDE,
    save_times: ArrayLike,
    realization: WienerRealization | None,
    /,
    *,
    dt: float,
    solver: Any | None,
    stepsize_controller: Any | None,
    adjoint: Any | None,
    rtol: float,
    atol: float,
    max_steps: int | None,
    throw: bool,
) -> DifferentialSolution:
    from ._diffrax_backend import solve_diffrax, solve_diffrax_ensemble

    kwargs = dict(
        save_times=save_times,
        realization=realization,
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        dt0=dt,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        throw=throw,
    )
    if realization is not None and realization.sample_shape:
        return solve_diffrax_ensemble(spde.problem, **kwargs)
    return solve_diffrax(spde.problem, **kwargs)


def solve_semilinear_spde(
    spde: SemidiscreteSPDE,
    /,
    *,
    save_times: ArrayLike,
    realization: WienerRealization | None = None,
    dt: float,
    matrix_function_policy: MatrixFunctionPolicy | None = None,
    fallback: SemilinearFallback = "diffrax",
    diffrax_solver: Any | None = None,
    diffrax_stepsize_controller: Any | None = None,
    diffrax_adjoint: Any | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int | None = 4096,
    throw: bool = False,
) -> DifferentialSolution:
    """Integrate a semilinear SPDE with exact compatible stochastic convolution.

    The supported specialization is fixed-step Itô exponential Euler with additive
    finite-rank noise. Unsupported problems lower to the existing Diffrax backend by
    default; ``fallback="error"`` exposes the unsupported reason instead.
    """
    if not isinstance(spde, SemidiscreteSPDE):
        raise TypeError("solve_semilinear_spde requires a SemidiscreteSPDE.")
    step_limit = float(dt)
    if not isfinite(step_limit) or step_limit <= 0.0:
        raise ValueError("dt must be finite and positive.")
    if fallback not in ("diffrax", "error"):
        raise ValueError("fallback must be 'diffrax' or 'error'.")
    policy = (
        MatrixFunctionPolicy()
        if matrix_function_policy is None
        else matrix_function_policy
    )
    if not isinstance(policy, MatrixFunctionPolicy):
        raise TypeError("matrix_function_policy must be a MatrixFunctionPolicy.")
    reason = _unsupported_reason(spde, realization)
    if reason is not None:
        if fallback == "error":
            raise ValueError(
                f"Specialized semilinear integration is unavailable: {reason}."
            )
        return _solve_diffrax_fallback(
            spde,
            save_times,
            realization,
            dt=step_limit,
            solver=diffrax_solver,
            stepsize_controller=diffrax_stepsize_controller,
            adjoint=diffrax_adjoint,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            throw=throw,
        )

    drift = spde.semilinear_drift
    assert drift is not None
    steps, save_indices, saved = _step_schedule(
        float(spde.problem.t0),
        float(spde.problem.t1),
        save_times,
        step_limit,
    )
    num_steps = int(steps.size)
    initial_state = spde.problem.initial_state
    stochastic = spde.problem.stochastic
    noise_basis = spde.noise_basis
    noise_eigenvalues = drift.compatible_noise_eigenvalues

    def one_path(path_key, path_sign):
        def advance(carry, item):
            time, state = carry
            step_value, step_index = item
            propagated = matrix_exponential_action(
                drift.linear_operator,
                state,
                step_value,
                policy=policy,
                spectral=drift.spectral_representation,
                spectral_bounds=drift.spectral_bounds,
                self_adjoint=drift.mass_self_adjoint,
                mass_weights=drift.mass_weights,
            )
            nonlinear = drift.nonlinear(time, state, spde.problem.args)
            nonlinear_update = step_value * matrix_phi1_action(
                drift.linear_operator,
                nonlinear,
                step_value,
                policy=policy,
                spectral=drift.spectral_representation,
                spectral_bounds=drift.spectral_bounds,
                self_adjoint=drift.mass_self_adjoint,
                mass_weights=drift.mass_weights,
            )
            if stochastic:
                assert noise_basis is not None
                assert noise_eigenvalues is not None
                step_key = jr.fold_in(path_key, step_index)
                normal = path_sign * jr.normal(
                    step_key,
                    (noise_basis.rank,),
                    dtype=state.dtype,
                )
                noise_update = exact_modal_stochastic_convolution(
                    noise_basis,
                    noise_eigenvalues,
                    step_value,
                    normal,
                )
            else:
                noise_update = jnp.zeros_like(state)
            next_state = propagated + nonlinear_update + noise_update
            return (time + step_value, next_state), next_state

        _, stepped = jax.lax.scan(
            advance,
            (jnp.asarray(spde.problem.t0), initial_state),
            (steps, jnp.arange(num_steps, dtype=jnp.uint32)),
        )
        complete = jnp.concatenate((initial_state[None, ...], stepped), axis=0)
        return complete[save_indices]

    if stochastic:
        assert realization is not None
        if realization.sample_shape:
            flat_keys = realization.path_keys.reshape((-1,))
            flat_signs = realization.path_signs.reshape((-1,))
            states = jax.vmap(one_path)(flat_keys, flat_signs).reshape(
                realization.sample_shape + (int(saved.size),) + spde.state_shape
            )
            times = jnp.broadcast_to(saved, realization.sample_shape + saved.shape)
            sample_shape = realization.sample_shape
        else:
            states = one_path(realization.path_keys, realization.path_signs)
            times = saved
            sample_shape = ()
    else:
        states = one_path(jr.key(0), jnp.asarray(1.0))
        times = saved
        sample_shape = ()
    state_axes = tuple(range(len(sample_shape) + 1, states.ndim))
    valid = jnp.isfinite(times) & jnp.all(jnp.isfinite(states), axis=state_axes)
    stats_steps = (
        jnp.full(sample_shape, num_steps, dtype=jnp.int32)
        if sample_shape
        else jnp.asarray(num_steps, dtype=jnp.int32)
    )
    return DifferentialSolution(
        times=times,
        states=states,
        valid=valid,
        sample_shape=sample_shape,
        backend_result="successful",
        stats={
            "num_steps": stats_steps,
            "matrix_function_method": policy.method,
            "differentiation": policy.differentiation,
            "exact_stochastic_convolution": bool(stochastic),
        },
        event_mask=None,
        realization=realization,
        wiener_term_slices=spde.problem.wiener_term_slices,
        solver_name="SemilinearExponentialEuler",
        interpretation=spde.problem.interpretation,
    )


__all__ = [
    "SemilinearFallback",
    "exact_modal_stochastic_convolution",
    "solve_semilinear_spde",
]
