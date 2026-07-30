#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from ._differential import DifferentialProblem, DifferentialSolution, WienerDriver


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
    driver: WienerDriver | None,
    driver_key: Array | None,
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
        if driver is None or driver_key is None:
            raise ValueError("Stochastic problems require a WienerDriver.")
        if dt0 is None:
            raise ValueError("Stochastic Diffrax solves require an explicit dt0.")
        assert problem.diffusion is not None
        real_dtype = jnp.asarray(problem.initial_state).real.dtype
        brownian = dfx.VirtualBrownianTree(
            t0=problem.t0,
            t1=problem.t1,
            tol=driver.tolerance,
            shape=jax.ShapeDtypeStruct(driver.noise_shape, real_dtype),
            key=driver_key,
            levy_area=_levy_area(driver.levy_area),
        )
        terms = dfx.MultiTerm(
            drift_term,
            dfx.ControlTerm(_vector_field(problem.diffusion), brownian),
        )
    else:
        if driver is not None or driver_key is not None:
            raise ValueError("Deterministic problems do not accept a WienerDriver.")
        terms = drift_term
    return dfx.diffeqsolve(
        terms,
        solver,
        t0=problem.t0,
        t1=problem.t1,
        dt0=dt0,
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


def solve_diffrax(
    problem: DifferentialProblem,
    /,
    *,
    save_times: ArrayLike,
    driver: WienerDriver | None = None,
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
    """Solve one finite-dimensional ODE or SDE through Diffrax."""
    if not isinstance(problem, DifferentialProblem):
        raise TypeError("solve_diffrax requires a DifferentialProblem.")
    if driver is not None and not isinstance(driver, WienerDriver):
        raise TypeError("driver must be a WienerDriver or None.")
    if not isinstance(dense, bool):
        raise TypeError("dense must be a bool.")
    times = _save_times(problem, save_times)
    selected_solver = _resolved_solver(problem, solver)
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
        driver=driver,
        driver_key=None if driver is None else driver.key,
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
        driver=driver,
        realization_keys=None if driver is None else driver.key,
        solver_name=type(selected_solver).__name__,
        interpretation=problem.interpretation,
    )


def solve_diffrax_ensemble(
    problem: DifferentialProblem,
    /,
    *,
    save_times: ArrayLike,
    driver: WienerDriver,
    num_paths: int,
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
    """Solve independent SDE realizations with one leading process-sample axis."""
    if not isinstance(problem, DifferentialProblem):
        raise TypeError("solve_diffrax_ensemble requires a DifferentialProblem.")
    if not problem.stochastic:
        raise ValueError("solve_diffrax_ensemble requires a stochastic problem.")
    if not isinstance(driver, WienerDriver):
        raise TypeError("driver must be a WienerDriver.")
    if not isinstance(dense, bool):
        raise TypeError("dense must be a bool.")
    count = int(num_paths)
    if count <= 0:
        raise ValueError("num_paths must be positive.")
    times = _save_times(problem, save_times)
    selected_solver = _resolved_solver(problem, solver)
    controller = _resolved_controller(
        problem,
        stepsize_controller,
        rtol=rtol,
        atol=atol,
    )
    selected_adjoint = dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
    keys = jr.split(driver.key, count)

    def one(key):
        return _native_solution(
            problem,
            times,
            driver=driver,
            driver_key=key,
            solver=selected_solver,
            stepsize_controller=controller,
            adjoint=selected_adjoint,
            dt0=dt0,
            event=event,
            dense=dense,
            max_steps=max_steps,
            throw=throw,
        )

    native = jax.vmap(one)(keys)
    native_times = jnp.asarray(native.ts)
    native_states = jnp.asarray(native.ys)
    return DifferentialSolution(
        times=native_times,
        states=native_states,
        valid=_valid_values(native_times, native_states, sample_ndim=1),
        sample_shape=(count,),
        interpolation=_dense_interpolation(native, (count,)) if dense else None,
        backend_result=native.result,
        stats=native.stats,
        event_mask=native.event_mask,
        driver=driver,
        realization_keys=keys,
        solver_name=type(selected_solver).__name__,
        interpretation=problem.interpretation,
    )


__all__ = ["solve_diffrax", "solve_diffrax_ensemble"]
