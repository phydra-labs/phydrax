#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod
from typing import Any, Protocol

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike

from ..stochastic import WienerRealization
from ._differential import DifferentialProblem, DifferentialSolution
from ._differential_ir import lower_deterministic_problem
from ._geometric import (
    AbstractGeometricSolver,
    CommutatorFreeSolver,
    GeometricEuler,
    RKMK,
    SRKMK,
    StormerVerlet,
)
from ._save_schedule import validate_save_times
from ._split_differential import SplitDifferentialProblem
from ._ssp_runge_kutta import SSPRK33, SSPRK54
from ._temporal_method import (
    configuration_id,
    diffrax_method_capabilities,
    TemporalMethodCapabilities,
    TemporalSolveEvidence,
)
from ._temporal_precision import TemporalPrecisionPolicy


class _StochasticProblemContract(Protocol):
    @property
    def t0(self) -> Array: ...

    @property
    def t1(self) -> Array: ...

    @property
    def noise_shape(self) -> tuple[int, ...]: ...

    @property
    def noise_id(self) -> str | None: ...

    @property
    def interpretation(self) -> str: ...

    @property
    def additive_noise(self) -> bool: ...


class _VectorizedDenseInterpolation(eqx.Module):
    """Dense Diffrax interpolation over shared arbitrarily shaped query times."""

    interpolation: dfx.DenseInterpolation
    precision: TemporalPrecisionPolicy
    sample_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        interpolation: dfx.DenseInterpolation,
        sample_shape: tuple[int, ...],
        precision: TemporalPrecisionPolicy,
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
        self.precision = precision
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
        return self.precision.output(
            values.reshape(self.sample_shape + query.shape + values.shape[2:])
        )


def _levy_area(kind: str, /) -> type:
    if kind == "brownian":
        return dfx.BrownianIncrement
    if kind == "space_time":
        return dfx.SpaceTimeLevyArea
    if kind == "space_time_time":
        return dfx.SpaceTimeTimeLevyArea
    raise AssertionError(f"Unhandled Levy-area kind {kind!r}.")


def _realized_wiener_path(
    realization: WienerRealization,
    path_key: Array,
    path_sign: Array,
    dtype: jnp.dtype,
    /,
) -> tuple[dfx.VirtualBrownianTree, Array]:
    """Construct one ordinary or delayed Diffrax path from global provenance."""
    brownian = dfx.VirtualBrownianTree(
        t0=realization.support[0],
        t1=realization.support[1],
        tol=realization.tolerance,
        shape=jax.ShapeDtypeStruct(realization.noise_shape, dtype),
        key=path_key,
        levy_area=_levy_area(realization.levy_area),
    )
    return brownian, jnp.asarray(path_sign, dtype=dtype)


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
    problem: _StochasticProblemContract,
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
    if isinstance(problem, DifferentialProblem):
        capabilities = _method_capabilities(solver)
        structures = tuple(term.structure for term in problem.wiener_terms)
        if capabilities.noise_requirement == "additive" and any(
            structure != "additive" for structure in structures
        ):
            raise ValueError(
                f"{type(solver).__name__} requires explicitly additive noise."
            )
        if capabilities.noise_requirement == "commutative" and any(
            structure == "general" for structure in structures
        ):
            raise ValueError(
                f"{type(solver).__name__} requires explicitly commutative noise."
            )


def _validated_realization_interval(
    problem: _StochasticProblemContract,
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


def _validated_state_geometry_solver(
    problem: DifferentialProblem | SplitDifferentialProblem,
    solver: Any,
    /,
) -> None:
    geometry = problem.state_geometry
    geometric = isinstance(solver, AbstractGeometricSolver)
    if geometry is None:
        if geometric:
            raise ValueError(
                "A geometric solver requires DifferentialProblem state_geometry."
            )
        return
    if not geometry.trivial and not geometric:
        raise ValueError(
            "A nontrivial state_geometry requires an AbstractGeometricSolver; "
            f"got {type(solver).__name__}."
        )
    if geometric and solver.geometry.geometry_id != geometry.geometry_id:
        raise ValueError(
            "Geometric solver and DifferentialProblem must carry the same "
            "state_geometry_id."
        )
    if problem.stochastic and not geometry.trivial:
        if problem.interpretation == "ito":
            raise ValueError(
                "Nontrivial state geometry supports explicit Stratonovich dynamics "
                "only; generic Itô geometry is not implemented."
            )
        if not isinstance(solver, SRKMK):
            raise ValueError("A stochastic nontrivial state_geometry requires SRKMK.")
    if not problem.stochastic and isinstance(solver, SRKMK):
        raise ValueError("SRKMK requires a stochastic DifferentialProblem.")


def _method_capabilities(solver: Any, /) -> TemporalMethodCapabilities:
    if isinstance(solver, (SSPRK33, SSPRK54)):
        return solver.capabilities
    if isinstance(solver, GeometricEuler):
        return TemporalMethodCapabilities(
            equation_forms=("explicit-ode", "geometric-ode"),
            method_class="geometric",
            order=1,
            dense_order=1,
            adaptive=False,
            stage_abscissae=solver.stage_abscissae,
            causal_stage_extent=solver.causal_stage_extent,
            verified=True,
            method_id=solver.solver_id,
        )
    if isinstance(solver, StormerVerlet):
        return TemporalMethodCapabilities(
            equation_forms=("explicit-ode", "geometric-ode"),
            method_class="geometric",
            order=2,
            dense_order=1,
            adaptive=False,
            stage_abscissae=solver.stage_abscissae,
            causal_stage_extent=solver.causal_stage_extent,
            symplectic=True,
            reversible=True,
            verified=True,
            method_id=solver.solver_id,
        )
    if isinstance(solver, RKMK):
        return TemporalMethodCapabilities(
            equation_forms=("explicit-ode", "geometric-ode"),
            method_class="geometric",
            order=2 if solver.method == "midpoint" else 4,
            dense_order=1,
            adaptive=False,
            stage_abscissae=solver.stage_abscissae,
            causal_stage_extent=solver.causal_stage_extent,
            verified=True,
            method_id=solver.solver_id,
        )
    if isinstance(solver, CommutatorFreeSolver):
        return TemporalMethodCapabilities(
            equation_forms=("explicit-ode", "geometric-ode"),
            method_class="geometric",
            order=solver.tableau.order,
            dense_order=1,
            adaptive=False,
            stage_abscissae=solver.stage_abscissae,
            causal_stage_extent=solver.causal_stage_extent,
            verified=solver.tableau.tableau_id == "tableau:commutator-free-midpoint",
            method_id=solver.solver_id,
        )
    if isinstance(solver, SRKMK):
        return TemporalMethodCapabilities(
            equation_forms=("sde", "geometric-sde"),
            method_class="geometric",
            order=1,
            dense_order=1,
            strong_orders=(("general", 0.5),),
            adaptive=False,
            stage_abscissae=solver.stage_abscissae,
            causal_stage_extent=solver.causal_stage_extent,
            noise_requirement="general",
            verified=True,
            method_id=solver.solver_id,
        )
    return diffrax_method_capabilities(solver)


def _solver_precision(
    solver: Any,
    requested: TemporalPrecisionPolicy | None,
    /,
) -> TemporalPrecisionPolicy:
    embedded = (
        solver.precision
        if isinstance(solver, (SSPRK33, SSPRK54))
        else TemporalPrecisionPolicy()
    )
    if requested is None:
        return embedded
    if not isinstance(requested, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    if (
        isinstance(solver, (SSPRK33, SSPRK54))
        and requested.policy_id != embedded.policy_id
    ):
        raise ValueError(
            "Explicit Diffrax precision must match the precision embedded in SSPRK."
        )
    return requested


def _equation_form(problem: DifferentialProblem | SplitDifferentialProblem, /) -> str:
    if isinstance(problem, SplitDifferentialProblem):
        return "additive-ode"
    geometry = problem.state_geometry
    nontrivial_geometry = geometry is not None and not geometry.trivial
    if problem.stochastic:
        return "geometric-sde" if nontrivial_geometry else "sde"
    return "geometric-ode" if nontrivial_geometry else "explicit-ode"


def _validated_method_form(
    problem: DifferentialProblem | SplitDifferentialProblem,
    solver: Any,
    /,
) -> None:
    if not isinstance(solver, dfx.AbstractSolver):
        raise TypeError("solver must be a Diffrax AbstractSolver or None.")
    form = _equation_form(problem)
    capabilities = _method_capabilities(solver)
    if form not in capabilities.equation_forms:
        raise ValueError(
            f"{type(solver).__name__} does not accept temporal equation form "
            f"{form!r}; supported forms are {capabilities.equation_forms}."
        )


def _solver_provenance(solver: Any, /) -> tuple[str, str]:
    if isinstance(solver, AbstractGeometricSolver):
        return solver.solver_id, solver.resolved_method
    name = type(solver).__name__
    return f"solver:diffrax:{name}", name


def _resolved_solver(
    problem: DifferentialProblem | SplitDifferentialProblem,
    solver: Any | None,
    /,
) -> Any:
    if solver is not None:
        return solver
    if isinstance(problem, SplitDifferentialProblem):
        return dfx.KenCarp4(root_finder=optx.Newton(rtol=1e-8, atol=1e-10))
    if problem.state_geometry is not None and not problem.state_geometry.trivial:
        if problem.stochastic:
            if problem.interpretation == "ito":
                raise ValueError(
                    "Nontrivial state geometry supports explicit Stratonovich "
                    "dynamics only."
                )
            return SRKMK(problem.state_geometry)
        return RKMK(problem.state_geometry)
    if not problem.stochastic:
        return dfx.Tsit5()
    if problem.interpretation == "ito":
        return dfx.Euler()
    return dfx.EulerHeun()


def _resolved_controller(
    problem: DifferentialProblem | SplitDifferentialProblem,
    solver: Any,
    controller: Any | None,
    /,
    *,
    rtol: float,
    atol: float,
) -> Any:
    if controller is not None and not isinstance(
        controller, dfx.AbstractStepSizeController
    ):
        raise TypeError("stepsize_controller must be a Diffrax controller or None.")
    if (
        isinstance(solver, AbstractGeometricSolver)
        and controller is not None
        and not isinstance(controller, dfx.ConstantStepSize)
    ):
        raise ValueError("Geometric solvers require diffrax.ConstantStepSize.")
    if controller is None:
        if isinstance(solver, AbstractGeometricSolver) or not isinstance(
            solver, dfx.AbstractAdaptiveSolver
        ):
            resolved = dfx.ConstantStepSize()
        else:
            resolved = dfx.PIDController(rtol=float(rtol), atol=float(atol))
    else:
        resolved = controller
    if isinstance(resolved, dfx.AbstractAdaptiveStepSizeController) and not isinstance(
        solver, dfx.AbstractAdaptiveSolver
    ):
        raise ValueError(
            f"{type(solver).__name__} does not provide an error estimate required "
            "by an adaptive step-size controller."
        )
    return resolved


def _validate_step_configuration(
    controller: Any,
    dt0: ArrayLike | None,
    /,
    *,
    rtol: float,
    atol: float,
) -> None:
    relative = float(rtol)
    absolute = float(atol)
    if (
        not isfinite(relative)
        or not isfinite(absolute)
        or relative <= 0.0
        or absolute <= 0.0
    ):
        raise ValueError("rtol and atol must be finite and positive.")
    if (
        not isinstance(controller, dfx.AbstractAdaptiveStepSizeController)
        and not isinstance(controller, dfx.StepTo)
        and dt0 is None
    ):
        raise ValueError("Fixed-step Diffrax controllers require an explicit dt0.")


def _solve_evidence(
    problem: DifferentialProblem | SplitDifferentialProblem,
    solver: Any,
    controller: Any,
    adjoint: Any,
    event: Any | None,
    precision: TemporalPrecisionPolicy,
    /,
    *,
    rtol: float,
    atol: float,
    dt0: ArrayLike | None,
    dense: bool,
    max_steps: int | None,
    explicit_configuration_id: str | None,
) -> TemporalSolveEvidence:
    if explicit_configuration_id is not None and (
        not isinstance(explicit_configuration_id, str) or not explicit_configuration_id
    ):
        raise ValueError("solver_configuration_id must be non-empty or None.")
    controller_id = configuration_id(controller, prefix="controller")
    adjoint_id = configuration_id(adjoint, prefix="adjoint")
    event_id = None if event is None else configuration_id(event, prefix="event")
    resolved_configuration_id = (
        explicit_configuration_id
        if explicit_configuration_id is not None
        else configuration_id(
            (
                solver,
                controller,
                adjoint,
                event,
                float(rtol),
                float(atol),
                None if dt0 is None else repr(dt0),
                bool(dense),
                max_steps,
                precision.policy_id,
            ),
            prefix="temporal-configuration",
        )
    )
    return TemporalSolveEvidence(
        _method_capabilities(solver),
        equation_form=_equation_form(problem),
        backend_id="backend:diffrax",
        configuration_id=resolved_configuration_id,
        controller_id=controller_id,
        adjoint_id=adjoint_id,
        event_id=event_id,
        adaptive=isinstance(controller, dfx.AbstractAdaptiveStepSizeController),
        dense=dense,
        maximum_steps=max_steps,
        precision_evidence=precision.evidence_for(
            problem.initial_state,
            problem.t0,
        ),
    )


def _native_solution(
    problem: DifferentialProblem | SplitDifferentialProblem,
    save_times: Array,
    *,
    realization: WienerRealization | None,
    path_key: Array | None,
    path_sign: Array | None,
    solver: Any,
    precision: TemporalPrecisionPolicy,
    stepsize_controller: Any,
    adjoint: Any,
    dt0: ArrayLike | None,
    event: Any | None,
    dense: bool,
    max_steps: int | None,
    throw: bool,
):
    if isinstance(problem, SplitDifferentialProblem):
        if realization is not None or path_key is not None or path_sign is not None:
            raise ValueError("Split differential problems do not accept Wiener paths.")
        lowered = lower_deterministic_problem(problem)
        assert lowered.implicit_rhs is not None
        terms = dfx.MultiTerm(
            dfx.ODETerm(_vector_field(lowered.explicit_rhs)),
            dfx.ODETerm(_vector_field(lowered.implicit_rhs)),
        )
        start = problem.t0
        end = problem.t1
        resolved_dt0 = dt0
    elif problem.stochastic:
        if realization is None or path_key is None or path_sign is None:
            raise ValueError("Stochastic problems require a WienerRealization.")
        start, end = _validated_realization_interval(problem, realization)
        resolved_dt0 = None if dt0 is None else jnp.asarray(dt0)
        if resolved_dt0 is None:
            if not isinstance(stepsize_controller, dfx.StepTo):
                raise ValueError(
                    "Stochastic Diffrax solves require explicit dt0 unless the "
                    "controller declares every step with diffrax.StepTo."
                )
        elif isinstance(stepsize_controller, dfx.ConstantStepSize):
            resolved_dt0 = eqx.error_if(
                resolved_dt0,
                jnp.abs(resolved_dt0) <= realization.tolerance,
                "WienerRealization tolerance must be strictly smaller than the "
                "fixed integration step.",
            )
        real_dtype = jnp.asarray(problem.initial_state).real.dtype
        brownian, signed_path = _realized_wiener_path(
            realization,
            path_key,
            path_sign,
            real_dtype,
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(_vector_field(problem.drift)),
            dfx.ControlTerm(
                _combined_diffusion(problem, signed_path),
                brownian,
            ),
        )
    else:
        if realization is not None or path_key is not None or path_sign is not None:
            raise ValueError("Deterministic problems do not accept a WienerRealization.")
        start = problem.t0
        end = problem.t1
        resolved_dt0 = dt0
        lowered = lower_deterministic_problem(problem)
        terms = dfx.ODETerm(_vector_field(lowered.explicit_rhs))
    time_dtype = jnp.asarray(problem.initial_state).real.dtype
    start = precision.coefficient(jnp.asarray(start, dtype=time_dtype))
    end = precision.coefficient(jnp.asarray(end, dtype=time_dtype))
    save_times = precision.coefficient(jnp.asarray(save_times, dtype=time_dtype))
    resolved_dt0 = (
        None
        if resolved_dt0 is None
        else precision.coefficient(jnp.asarray(resolved_dt0, dtype=time_dtype))
    )
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


def _dense_interpolation(
    native: Any,
    sample_shape: tuple[int, ...],
    precision: TemporalPrecisionPolicy,
    /,
) -> Any:
    interpolation = native.interpolation
    if interpolation is None:
        raise RuntimeError("Diffrax did not return the requested dense interpolation.")
    return _VectorizedDenseInterpolation(interpolation, sample_shape, precision)


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
    problem: DifferentialProblem | SplitDifferentialProblem,
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
    solver_configuration_id: str | None = None,
    precision: TemporalPrecisionPolicy | None = None,
) -> DifferentialSolution:
    """Solve one explicit, additive-split, or stochastic differential problem."""
    if not isinstance(problem, (DifferentialProblem, SplitDifferentialProblem)):
        raise TypeError(
            "solve_diffrax requires DifferentialProblem or SplitDifferentialProblem."
        )
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

    times = validate_save_times(problem.t0, problem.t1, save_times)
    selected_solver = _resolved_solver(problem, solver)
    precision = _solver_precision(selected_solver, precision)
    precision.validate_diffrax_state(
        problem.initial_state,
        internal_precision=isinstance(selected_solver, (SSPRK33, SSPRK54)),
    )
    _validated_state_geometry_solver(problem, selected_solver)
    if isinstance(selected_solver, AbstractGeometricSolver) and dt0 is None:
        raise ValueError("Geometric solvers require an explicit fixed dt0.")
    if realization is not None:
        _validated_stochastic_solver(problem, selected_solver, realization)
    _validated_method_form(problem, selected_solver)
    controller = _resolved_controller(
        problem,
        selected_solver,
        stepsize_controller,
        rtol=rtol,
        atol=atol,
    )
    _validate_step_configuration(controller, dt0, rtol=rtol, atol=atol)
    selected_adjoint = dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
    evidence = _solve_evidence(
        problem,
        selected_solver,
        controller,
        selected_adjoint,
        event,
        precision,
        rtol=rtol,
        atol=atol,
        dt0=dt0,
        dense=dense,
        max_steps=max_steps,
        explicit_configuration_id=solver_configuration_id,
    )
    native = _native_solution(
        problem,
        times,
        realization=realization,
        path_key=None if realization is None else realization.path_keys,
        path_sign=None if realization is None else realization.path_signs,
        solver=selected_solver,
        precision=precision,
        stepsize_controller=controller,
        adjoint=selected_adjoint,
        dt0=dt0,
        event=event,
        dense=dense,
        max_steps=max_steps,
        throw=throw,
    )
    native_times = jnp.asarray(native.ts)
    native_states = precision.output(native.ys)
    solver_id, resolved_method = _solver_provenance(selected_solver)
    return DifferentialSolution(
        times=native_times,
        states=native_states,
        valid=_valid_values(native_times, native_states, sample_ndim=0),
        interpolation=_dense_interpolation(native, (), precision) if dense else None,
        backend_result=native.result,
        stats=native.stats,
        event_mask=native.event_mask,
        realization=realization,
        wiener_term_slices=problem.wiener_term_slices,
        solver_name=type(selected_solver).__name__,
        interpretation=problem.interpretation,
        state_geometry_id=problem.state_geometry_id,
        solver_id=solver_id,
        resolved_method=resolved_method,
        discretization_bundle=problem.discretization_bundle,
        backend_successful=dfx.is_okay(native.result),
        event_terminated=dfx.is_event(native.result),
        temporal_evidence=evidence,
        problem_id=problem.problem_id,
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
    solver_configuration_id: str | None = None,
    precision: TemporalPrecisionPolicy | None = None,
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
    times = validate_save_times(problem.t0, problem.t1, save_times)
    selected_solver = _resolved_solver(problem, solver)
    precision = _solver_precision(selected_solver, precision)
    precision.validate_diffrax_state(
        problem.initial_state,
        internal_precision=isinstance(selected_solver, (SSPRK33, SSPRK54)),
    )
    _validated_state_geometry_solver(problem, selected_solver)
    _validated_stochastic_solver(problem, selected_solver, realization)
    _validated_method_form(problem, selected_solver)
    controller = _resolved_controller(
        problem,
        selected_solver,
        stepsize_controller,
        rtol=rtol,
        atol=atol,
    )
    _validate_step_configuration(controller, dt0, rtol=rtol, atol=atol)
    selected_adjoint = dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
    evidence = _solve_evidence(
        problem,
        selected_solver,
        controller,
        selected_adjoint,
        event,
        precision,
        rtol=rtol,
        atol=atol,
        dt0=dt0,
        dense=dense,
        max_steps=max_steps,
        explicit_configuration_id=solver_configuration_id,
    )
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
            precision=precision,
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
    native_states = precision.output(native.ys)
    solver_id, resolved_method = _solver_provenance(selected_solver)
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
            _dense_interpolation(native, realization.sample_shape, precision)
            if dense
            else None
        ),
        backend_result=native.result,
        stats=native.stats,
        event_mask=native.event_mask,
        realization=realization,
        wiener_term_slices=problem.wiener_term_slices,
        solver_name=type(selected_solver).__name__,
        interpretation=problem.interpretation,
        state_geometry_id=problem.state_geometry_id,
        solver_id=solver_id,
        resolved_method=resolved_method,
        discretization_bundle=problem.discretization_bundle,
        backend_successful=dfx.is_okay(native.result),
        event_terminated=dfx.is_event(native.result),
        temporal_evidence=evidence,
        problem_id=problem.problem_id,
    )


__all__ = ["solve_diffrax", "solve_diffrax_ensemble"]
