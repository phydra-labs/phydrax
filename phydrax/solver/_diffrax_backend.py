#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..stochastic import WienerRealization
from ._differential import DifferentialProblem, DifferentialSolution


class _VectorizedDenseInterpolation(eqx.Module):
    """Dense Diffrax interpolation over shared arbitrarily shaped query times."""

    interpolation: dfx.DenseInterpolation
    sample_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        interpolation: dfx.DenseInterpolation,
        sample_shape: tuple[int, ...],
        /,
    ):
        samples = tuple(int(size) for size in sample_shape)
        batch_shape = tuple(jnp.shape(interpolation.t0_if_trivial))
        if batch_shape != samples:
            raise ValueError(
                "Dense interpolation batch shape must match the solution sample shape; "
                f"got {batch_shape} and {samples}."
            )
        batch_ndim = len(samples)
        self.interpolation = jax.tree.map(
            lambda value: value.reshape((-1,) + value.shape[batch_ndim:]),
            interpolation,
            is_leaf=eqx.is_array,
        )
        self.sample_shape = samples

    @eqx.filter_jit
    def evaluate(
        self,
        query_times: ArrayLike,
        /,
        *,
        left: bool = True,
    ) -> Array:
        """Evaluate every realization on one shared scalar or array of times."""
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = jnp.asarray(query_times)
        if jnp.iscomplexobj(query):
            raise TypeError("Dense interpolation query times must be real-valued.")
        if query.size == 0:
            raise ValueError("Dense interpolation query times must be non-empty.")
        query = query.astype(float)
        query = eqx.error_if(
            query,
            ~jnp.all(jnp.isfinite(query)),
            "Dense interpolation query times must be finite.",
        )
        bounds = jax.vmap(
            lambda interpolation: jnp.stack((interpolation.t0, interpolation.t1))
        )(self.interpolation)
        lower = jnp.max(jnp.minimum(bounds[:, 0], bounds[:, 1]))
        upper = jnp.min(jnp.maximum(bounds[:, 0], bounds[:, 1]))
        query = eqx.error_if(
            query,
            jnp.any((query < lower) | (query > upper)),
            "Dense interpolation query times must lie within every solution interval.",
        )
        flat_times = query.reshape(-1)
        values = jax.vmap(
            lambda interpolation: jax.vmap(
                lambda time: interpolation.evaluate(time, left=left)
            )(flat_times)
        )(self.interpolation)
        return values.reshape(self.sample_shape + query.shape + values.shape[2:])


def _save_times(problem: DifferentialProblem, values: ArrayLike, /) -> Array:
    times = jnp.asarray(values, dtype=float)
    if times.ndim != 1 or int(times.shape[0]) <= 0:
        raise ValueError("save_times must be a non-empty rank-1 array.")
    times = eqx.error_if(
        times,
        ~jnp.all(jnp.isfinite(times)),
        "save_times must be finite.",
    )
    if int(times.shape[0]) > 1:
        times = eqx.error_if(
            times,
            ~jnp.all(jnp.diff(times) > 0.0),
            "save_times must be strictly increasing.",
        )
    return eqx.error_if(
        times,
        (times[0] < problem.t0) | (times[-1] > problem.t1),
        "save_times must lie within the problem time interval.",
    )


def _levy_area(kind: str, /) -> type:
    if kind == "brownian":
        return dfx.BrownianIncrement
    if kind == "space_time":
        return dfx.SpaceTimeLevyArea
    if kind == "space_time_time":
        return dfx.SpaceTimeTimeLevyArea
    raise AssertionError(f"Unhandled Levy-area kind {kind!r}.")


def _vector_field(function):
    def evaluate(t, state, args):
        return jnp.asarray(function(t, state, args))

    return evaluate


def _combined_diffusion(problem: DifferentialProblem, path_sign: Array, /):
    def evaluate(time, state, args):
        state_shape = tuple(jnp.shape(state))
        columns = []
        for term in problem.wiener_terms:
            value = jnp.asarray(term.coefficient(time, state, args))
            expected_shape = state_shape + term.noise_shape
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"WienerTerm {term.name!r} coefficient must return shape "
                    f"{expected_shape}; got {value.shape}."
                )
            columns.append(value.reshape(state_shape + (term.noise_size,)))
        return path_sign * jnp.concatenate(columns, axis=-1)

    return evaluate


def _validated_stochastic_solver(
    problem: DifferentialProblem,
    solver: Any,
    realization: WienerRealization,
    /,
) -> None:
    is_ito = isinstance(solver, dfx.AbstractItoSolver)
    is_stratonovich = isinstance(solver, dfx.AbstractStratonovichSolver)
    if not is_ito and not is_stratonovich:
        raise ValueError(
            "A stochastic problem requires a Diffrax solver explicitly marked as "
            "Itô or Stratonovich."
        )
    if not problem.additive_noise:
        if problem.interpretation == "ito" and not is_ito:
            raise ValueError("An Itô problem requires an Itô-compatible solver.")
        if problem.interpretation == "stratonovich" and not is_stratonovich:
            raise ValueError(
                "A Stratonovich problem requires a Stratonovich-compatible solver."
            )

    if isinstance(solver, dfx.AbstractSRK):
        provided = _levy_area(realization.levy_area)
        required = solver.minimal_levy_area
        if not issubclass(provided, required):
            raise ValueError(
                f"{type(solver).__name__} requires {required.__name__}, but the "
                f"Wiener realization provides {provided.__name__}."
            )


def _validated_realization_interval(
    problem: DifferentialProblem,
    realization: WienerRealization,
    /,
) -> tuple[Array, Array]:
    if problem.noise_shape != realization.noise_shape:
        raise ValueError(
            "Wiener realization noise_shape must match the problem's combined "
            f"noise shape; got {realization.noise_shape} and {problem.noise_shape}."
        )
    if problem.noise_id is not None and realization.noise_id != problem.noise_id:
        raise ValueError(
            "Wiener realization noise_id must match the problem's stochastic basis."
        )
    support_start, support_end = realization.support
    start = eqx.error_if(
        problem.t0,
        problem.t0 < support_start,
        "DifferentialProblem t0 lies before the Wiener realization support.",
    )
    end = eqx.error_if(
        problem.t1,
        problem.t1 > support_end,
        "DifferentialProblem t1 lies after the Wiener realization support.",
    )
    return start, end


def _resolved_solver(problem: DifferentialProblem, solver: Any | None, /) -> Any:
    if solver is not None:
        return solver
    if not problem.stochastic:
        return dfx.Tsit5()
    if problem.interpretation == "ito":
        return dfx.Euler()
    return dfx.EulerHeun()


def _resolved_controller(
    problem: DifferentialProblem,
    controller: Any | None,
    /,
    *,
    rtol: float,
    atol: float,
) -> Any:
    if controller is not None:
        return controller
    if problem.stochastic:
        return dfx.ConstantStepSize()
    return dfx.PIDController(rtol=float(rtol), atol=float(atol))


def _native_solution(
    problem: DifferentialProblem,
    save_times: Array,
    *,
    realization: WienerRealization | None,
    path_key: Array | None,
    path_sign: Array | None,
    solver: Any,
    stepsize_controller: Any,
    adjoint: Any,
    dt0: ArrayLike | None,
    event: Any | None,
    dense: bool,
    max_steps: int | None,
    throw: bool,
):
    drift_term = dfx.ODETerm(_vector_field(problem.drift))
    if problem.stochastic:
        if realization is None or path_key is None or path_sign is None:
            raise ValueError("Stochastic problems require a WienerRealization.")
        if dt0 is None:
            raise ValueError("Stochastic Diffrax solves require an explicit dt0.")
        start, end = _validated_realization_interval(problem, realization)
        resolved_dt0 = jnp.asarray(dt0)
        if isinstance(stepsize_controller, dfx.ConstantStepSize):
            resolved_dt0 = eqx.error_if(
                resolved_dt0,
                jnp.abs(resolved_dt0) <= realization.tolerance,
                "WienerRealization tolerance must be strictly smaller than the "
                "fixed integration step.",
            )
        real_dtype = jnp.asarray(problem.initial_state).real.dtype
        brownian = dfx.VirtualBrownianTree(
            t0=realization.support[0],
            t1=realization.support[1],
            tol=realization.tolerance,
            shape=jax.ShapeDtypeStruct(realization.noise_shape, real_dtype),
            key=path_key,
            levy_area=_levy_area(realization.levy_area),
        )
        terms = dfx.MultiTerm(
            drift_term,
            dfx.ControlTerm(
                _combined_diffusion(
                    problem,
                    jnp.asarray(path_sign, dtype=real_dtype),
                ),
                brownian,
            ),
        )
    else:
        if realization is not None or path_key is not None or path_sign is not None:
            raise ValueError("Deterministic problems do not accept a WienerRealization.")
        start = problem.t0
        end = problem.t1
        resolved_dt0 = dt0
        terms = drift_term
    return dfx.diffeqsolve(
        terms,
        solver,
        t0=start,
        t1=end,
        dt0=resolved_dt0,
        y0=problem.initial_state,
        args=problem.args,
        saveat=dfx.SaveAt(ts=save_times, dense=dense),
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        event=event,
        max_steps=max_steps,
        throw=bool(throw),
    )


def _valid_values(times: Array, states: Array, /, *, sample_ndim: int) -> Array:
    state_axes = tuple(range(sample_ndim + 1, states.ndim))
    finite_states = jnp.isfinite(states)
    if state_axes:
        finite_states = jnp.all(finite_states, axis=state_axes)
    return jnp.isfinite(times) & finite_states


def _dense_interpolation(native: Any, sample_shape: tuple[int, ...], /) -> Any:
    interpolation = native.interpolation
    if interpolation is None:
        raise RuntimeError("Diffrax did not return the requested dense interpolation.")
    return _VectorizedDenseInterpolation(interpolation, sample_shape)


def _reshape_native_sample_shape(native: Any, sample_shape: tuple[int, ...], /) -> Any:
    count = prod(sample_shape)

    def reshape(value):
        if eqx.is_array(value):
            if value.ndim == 0 or int(value.shape[0]) != count:
                raise ValueError(
                    "Vectorized Diffrax output does not align with the realization "
                    f"sample shape {sample_shape}."
                )
            return value.reshape(sample_shape + value.shape[1:])
        return value

    return jax.tree.map(reshape, native, is_leaf=eqx.is_array)


def solve_diffrax(
    problem: DifferentialProblem,
    /,
    *,
    save_times: ArrayLike,
    realization: WienerRealization | None = None,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    event: Any | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense: bool = False,
    max_steps: int | None = 4096,
    throw: bool = False,
) -> DifferentialSolution:
    """Solve one finite-dimensional ODE or globally defined SDE realization."""
    if not isinstance(problem, DifferentialProblem):
        raise TypeError("solve_diffrax requires a DifferentialProblem.")
    if realization is not None and not isinstance(realization, WienerRealization):
        raise TypeError("realization must be a WienerRealization or None.")
    if not isinstance(dense, bool):
        raise TypeError("dense must be a bool.")
    if problem.stochastic:
        if realization is None:
            raise ValueError("Stochastic problems require a WienerRealization.")
        if realization.sample_shape:
            raise ValueError(
                "solve_diffrax requires a scalar realization; use "
                "solve_diffrax_ensemble for a realization batch."
            )
    elif realization is not None:
        raise ValueError("Deterministic problems do not accept a WienerRealization.")

    times = _save_times(problem, save_times)
    selected_solver = _resolved_solver(problem, solver)
    if realization is not None:
        _validated_stochastic_solver(problem, selected_solver, realization)
    controller = _resolved_controller(
        problem,
        stepsize_controller,
        rtol=rtol,
        atol=atol,
    )
    selected_adjoint = dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
    native = _native_solution(
        problem,
        times,
        realization=realization,
        path_key=None if realization is None else realization.path_keys,
        path_sign=None if realization is None else realization.path_signs,
        solver=selected_solver,
        stepsize_controller=controller,
        adjoint=selected_adjoint,
        dt0=dt0,
        event=event,
        dense=dense,
        max_steps=max_steps,
        throw=throw,
    )
    native_times = jnp.asarray(native.ts)
    native_states = jnp.asarray(native.ys)
    return DifferentialSolution(
        times=native_times,
        states=native_states,
        valid=_valid_values(native_times, native_states, sample_ndim=0),
        interpolation=_dense_interpolation(native, ()) if dense else None,
        backend_result=native.result,
        stats=native.stats,
        event_mask=native.event_mask,
        realization=realization,
        wiener_term_slices=problem.wiener_term_slices,
        solver_name=type(selected_solver).__name__,
        interpretation=problem.interpretation,
    )


def solve_diffrax_ensemble(
    problem: DifferentialProblem,
    /,
    *,
    save_times: ArrayLike,
    realization: WienerRealization,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike,
    event: Any | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int | None = 4096,
    dense: bool = False,
    throw: bool = False,
) -> DifferentialSolution:
    """Solve the coupled SDE batch encoded by one Wiener realization."""
    if not isinstance(problem, DifferentialProblem):
        raise TypeError("solve_diffrax_ensemble requires a DifferentialProblem.")
    if not problem.stochastic:
        raise ValueError("solve_diffrax_ensemble requires a stochastic problem.")
    if not isinstance(realization, WienerRealization):
        raise TypeError("realization must be a WienerRealization.")
    if not realization.sample_shape:
        raise ValueError(
            "solve_diffrax_ensemble requires a realization with non-empty sample_shape."
        )
    if not isinstance(dense, bool):
        raise TypeError("dense must be a bool.")
    times = _save_times(problem, save_times)
    selected_solver = _resolved_solver(problem, solver)
    _validated_stochastic_solver(problem, selected_solver, realization)
    controller = _resolved_controller(
        problem,
        stepsize_controller,
        rtol=rtol,
        atol=atol,
    )
    selected_adjoint = dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
    count = realization.num_paths
    key_shape = tuple(realization.root_key.shape)
    keys = realization.path_keys.reshape((count,) + key_shape)
    signs = realization.path_signs.reshape((count,))

    def one(key, sign):
        return _native_solution(
            problem,
            times,
            realization=realization,
            path_key=key,
            path_sign=sign,
            solver=selected_solver,
            stepsize_controller=controller,
            adjoint=selected_adjoint,
            dt0=dt0,
            event=event,
            dense=dense,
            max_steps=max_steps,
            throw=throw,
        )

    native = _reshape_native_sample_shape(
        jax.vmap(one)(keys, signs),
        realization.sample_shape,
    )
    native_times = jnp.asarray(native.ts)
    native_states = jnp.asarray(native.ys)
    return DifferentialSolution(
        times=native_times,
        states=native_states,
        valid=_valid_values(
            native_times,
            native_states,
            sample_ndim=len(realization.sample_shape),
        ),
        sample_shape=realization.sample_shape,
        interpolation=(
            _dense_interpolation(native, realization.sample_shape) if dense else None
        ),
        backend_result=native.result,
        stats=native.stats,
        event_mask=native.event_mask,
        realization=realization,
        wiener_term_slices=problem.wiener_term_slices,
        solver_name=type(selected_solver).__name__,
        interpretation=problem.interpretation,
    )


__all__ = ["solve_diffrax", "solve_diffrax_ensemble"]
