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
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    matrix_exponential_action,
    matrix_phi1_action,
    MatrixFunctionPolicy,
)
from ..stochastic._spatial_noise import SpatialNoiseBasis
from ..stochastic._wiener import WienerRealization
from ._differential import DifferentialProblem, DifferentialSolution
from ._spde import SemidiscreteSPDE


SemilinearFallback: TypeAlias = Literal["diffrax", "error"]
SemilinearSPDEScheme: TypeAlias = Literal[
    "auto",
    "exponential_euler",
    "exponential_milstein",
]
_ResolvedSemilinearSPDEScheme: TypeAlias = Literal[
    "exact_additive",
    "exponential_euler",
    "exponential_milstein",
]


def _semilinear_solution_bundle(
    spde: SemidiscreteSPDE,
    realization: WienerRealization | None,
    num_steps: int,
    scheme: str,
    /,
) -> DiscretizationBundle:
    records = list(spde.discretization_bundle.records)
    temporal_key = DiscretizationKey(
        "spde_internal_time",
        DiscretizationRole.TEMPORAL,
        domain_labels=("time",),
    )
    records.append(
        DiscretizationRecord(
            temporal_key,
            "fixed-step-temporal-mesh",
            canonical_fingerprint(
                {
                    "kind": "semilinear-spde-time",
                    "problem": spde.discretization_bundle.bundle_id,
                    "num_steps": int(num_steps),
                    "scheme": str(scheme),
                }
            ),
        )
    )
    coupling_ids = list(spde.discretization_bundle.stochastic_coupling_ids)
    if realization is not None:
        driver_key = DiscretizationKey(
            "wiener_driver",
            DiscretizationRole.DRIVER,
            domain_labels=("time",),
        )
        records.append(
            DiscretizationRecord(
                driver_key,
                "wiener-realization",
                realization.realization_id,
                dependency_key_ids=(temporal_key.key_id,),
                realization_id=realization.realization_id,
            )
        )
        if realization.coupling_id is not None:
            coupling_ids.append(realization.coupling_id)
        if realization.sample_shape:
            ensemble_key = DiscretizationKey(
                "path_ensemble",
                DiscretizationRole.ENSEMBLE,
            )
            records.append(
                DiscretizationRecord(
                    ensemble_key,
                    "stochastic-path-ensemble",
                    canonical_fingerprint(
                        {
                            "kind": "path-ensemble",
                            "realization": realization.realization_id,
                            "sample_shape": list(realization.sample_shape),
                        }
                    ),
                    dependency_key_ids=(driver_key.key_id,),
                    realization_id=realization.realization_id,
                )
            )
    return DiscretizationBundle(
        records,
        transfers=spde.discretization_bundle.transfers,
        stochastic_coupling_ids=tuple(dict.fromkeys(coupling_ids)),
    )


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


def _realization_unsupported_reason(
    spde: SemidiscreteSPDE,
    realization: WienerRealization | None,
    /,
    *,
    require_increments: bool,
) -> str | None:
    problem = spde.problem
    if not problem.stochastic:
        if realization is not None:
            return "deterministic problems do not accept a Wiener realization"
        return None
    if problem.interpretation != "ito":
        return "the specialized solver currently supports only Itô equations"
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
    if require_increments and realization.levy_area != "brownian":
        return "the specialized scheme requires Brownian increments"
    return None


def _exact_additive_unsupported_reason(
    spde: SemidiscreteSPDE,
    realization: WienerRealization | None,
    /,
) -> str | None:
    reason = _realization_unsupported_reason(
        spde,
        realization,
        require_increments=False,
    )
    if reason is not None:
        return reason
    problem = spde.problem
    if not problem.stochastic:
        return "exact stochastic convolution requires a stochastic problem"
    if not problem.additive_noise:
        return "exact stochastic convolution requires additive noise"
    if spde.noise_basis is None:
        return "additive stochastic convolution requires a SpatialNoiseBasis"
    drift = spde.semilinear_drift
    assert drift is not None
    if (
        drift.compatible_noise_eigenvalues is None
        or drift.compatible_noise_basis_id != spde.noise_basis.basis_id
        or drift.compatible_noise_eigenvalues.shape != (spde.noise_basis.rank,)
    ):
        return "the linear operator and additive noise basis do not share declared modes"
    return None


def _resolved_specialization(
    spde: SemidiscreteSPDE,
    realization: WienerRealization | None,
    scheme: SemilinearSPDEScheme,
    /,
) -> tuple[_ResolvedSemilinearSPDEScheme | None, str | None]:
    if spde.semilinear_drift is None:
        return None, "the semidiscrete problem has no explicit semilinear decomposition"
    if scheme == "auto":
        if spde.problem.stochastic:
            exact_reason = _exact_additive_unsupported_reason(spde, realization)
            if exact_reason is None:
                return "exact_additive", None
        reason = _realization_unsupported_reason(
            spde,
            realization,
            require_increments=spde.problem.stochastic,
        )
        return (None, reason) if reason is not None else ("exponential_euler", None)
    reason = _realization_unsupported_reason(
        spde,
        realization,
        require_increments=spde.problem.stochastic,
    )
    if reason is not None:
        return None, reason
    if any(term.representation != "dense" for term in spde.problem.wiener_terms):
        return None, "semilinear specialization requires dense Wiener coefficients"
    if scheme == "exponential_euler":
        return scheme, None
    if not spde.problem.stochastic:
        return None, "exponential Milstein requires a stochastic problem"
    if any(
        term.structure not in ("additive", "commutative")
        for term in spde.problem.wiener_terms
    ):
        return None, "exponential Milstein requires declared commutative noise"
    return scheme, None


def _diffusion_columns(
    problem: DifferentialProblem,
    time: Array,
    state: Array,
    /,
) -> Array:
    state_shape = tuple(state.shape)
    columns = []
    for term in problem.wiener_terms:
        value = term.coefficient_array(time, state, problem.args)
        expected = state_shape + term.noise_shape
        if tuple(value.shape) != expected:
            raise ValueError(
                f"WienerTerm {term.name!r} coefficient must return shape "
                f"{expected}; got {value.shape}."
            )
        columns.append(value.reshape(state_shape + (term.noise_size,)))
    return jnp.concatenate(columns, axis=-1)


def _milstein_increment(
    problem: DifferentialProblem,
    time: Array,
    state: Array,
    step: Array,
    wiener_increment: Array,
    diffusion: Array,
    /,
) -> Array:
    """Compute the commutative Milstein correction with factor JVPs."""
    directions = jnp.moveaxis(diffusion, -1, 0)

    def differentiate(direction):
        return jax.jvp(
            lambda value: _diffusion_columns(problem, time, value),
            (state,),
            (direction,),
        )[1]

    derivatives = jax.vmap(differentiate)(directions)
    noise_size = int(wiener_increment.size)
    flattened = derivatives.reshape((noise_size, int(state.size), noise_size))
    iterated = wiener_increment[:, None] * wiener_increment[None, :] - step * jnp.eye(
        noise_size, dtype=wiener_increment.dtype
    )
    return (0.5 * oe.contract("ksj,kj->s", flattened, iterated)).reshape(state.shape)


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

    if realization is not None and realization.sample_shape:
        return solve_diffrax_ensemble(
            spde.problem,
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
    return solve_diffrax(
        spde.problem,
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


def solve_semilinear_spde(
    spde: SemidiscreteSPDE,
    /,
    *,
    save_times: ArrayLike,
    realization: WienerRealization | None = None,
    dt: float,
    scheme: SemilinearSPDEScheme = "auto",
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
    """Integrate a semilinear SPDE with a matrix-free exponential scheme.

    ``"auto"`` preserves exact modal stochastic convolution for compatible
    additive noise and otherwise selects exponential Euler. Exponential Milstein
    uses factor JVPs and requires explicitly declared commutative Itô noise.
    Unsupported problems lower to Diffrax by default.
    """
    if not isinstance(spde, SemidiscreteSPDE):
        raise TypeError("solve_semilinear_spde requires a SemidiscreteSPDE.")
    if (
        spde.problem.state_geometry is not None
        and not spde.problem.state_geometry.trivial
    ):
        raise ValueError(
            "Semilinear exponential solvers do not support nontrivial state_geometry."
        )
    step_limit = float(dt)
    if not isfinite(step_limit) or step_limit <= 0.0:
        raise ValueError("dt must be finite and positive.")
    if scheme not in ("auto", "exponential_euler", "exponential_milstein"):
        raise ValueError(
            "scheme must be 'auto', 'exponential_euler', or 'exponential_milstein'."
        )
    if fallback not in ("diffrax", "error"):
        raise ValueError("fallback must be 'diffrax' or 'error'.")
    policy = (
        MatrixFunctionPolicy()
        if matrix_function_policy is None
        else matrix_function_policy
    )
    if not isinstance(policy, MatrixFunctionPolicy):
        raise TypeError("matrix_function_policy must be a MatrixFunctionPolicy.")
    resolved_scheme, reason = _resolved_specialization(spde, realization, scheme)
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
    assert resolved_scheme is not None

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
    if isinstance(drift.linear_operator, AbstractLinearOperator) and isinstance(
        drift.linear_operator.source, ArraySpace
    ):
        initial_state = initial_state.astype(drift.linear_operator.source.dtype)
    stochastic = spde.problem.stochastic
    exact_additive = resolved_scheme == "exact_additive"
    noise_basis = spde.noise_basis
    noise_eigenvalues = drift.compatible_noise_eigenvalues

    if stochastic and not exact_additive:
        assert realization is not None
        ends = jnp.asarray(spde.problem.t0) + jnp.cumsum(steps)
        starts = jnp.concatenate((jnp.asarray(spde.problem.t0)[None], ends[:-1]))
        support_start, support_end = realization.support
        starts = jnp.clip(starts, support_start, support_end)
        ends = jnp.clip(ends, support_start, support_end)
        path_increments = realization.increments(
            starts,
            ends,
            dtype=initial_state.real.dtype,
        )
    else:
        sample_shape = () if realization is None else realization.sample_shape
        path_increments = jnp.zeros(sample_shape + (num_steps, 0))
    matrix_operator = drift.linear_operator
    if (
        not isinstance(matrix_operator, AbstractLinearOperator)
        and drift.spectral_representation is not None
    ):
        matrix_operator = drift.spectral_representation.operator

    def exponential_action(value, step_value):
        return matrix_exponential_action(
            matrix_operator,
            value,
            step_value,
            policy=policy,
            spectral=drift.spectral_representation,
            spectral_bounds=drift.spectral_bounds,
        )

    def one_path(path_key, path_sign, wiener_increments):
        def advance(carry, item):
            time, state, path_valid = carry
            step_value, step_index, wiener_increment = item
            nonlinear = drift.nonlinear(time, state, spde.problem.args)
            nonlinear_result = matrix_phi1_action(
                matrix_operator,
                nonlinear,
                step_value,
                policy=policy,
                spectral=drift.spectral_representation,
                spectral_bounds=drift.spectral_bounds,
            )
            nonlinear_update = step_value * nonlinear_result.value
            if exact_additive:
                assert noise_basis is not None
                assert noise_eigenvalues is not None
                step_key = jr.fold_in(path_key, step_index)
                normal = path_sign * jr.normal(
                    step_key,
                    (noise_basis.rank,),
                    dtype=state.real.dtype,
                )
                propagated_result = exponential_action(state, step_value)
                noise_update = exact_modal_stochastic_convolution(
                    noise_basis,
                    noise_eigenvalues,
                    step_value,
                    normal,
                )
            elif stochastic:
                diffusion = _diffusion_columns(spde.problem, time, state)
                local_noise = jnp.tensordot(
                    diffusion,
                    wiener_increment,
                    axes=((-1,), (0,)),
                )
                if resolved_scheme == "exponential_milstein":
                    local_noise = local_noise + _milstein_increment(
                        spde.problem,
                        time,
                        state,
                        step_value,
                        wiener_increment,
                        diffusion,
                    )
                propagated_result = exponential_action(
                    state + local_noise,
                    step_value,
                )
                noise_update = jnp.zeros_like(state)
            else:
                propagated_result = exponential_action(state, step_value)
                noise_update = jnp.zeros_like(state)
            next_state = propagated_result.value + nonlinear_update + noise_update
            next_valid = (
                path_valid & propagated_result.converged & nonlinear_result.converged
            )
            return (
                time + step_value,
                next_state,
                next_valid,
            ), (next_state, next_valid)

        _, (stepped, step_valid) = jax.lax.scan(
            advance,
            (
                jnp.asarray(spde.problem.t0),
                initial_state,
                jnp.asarray(True),
            ),
            (
                steps,
                jnp.arange(num_steps, dtype=jnp.uint32),
                wiener_increments,
            ),
        )
        complete = jnp.concatenate((initial_state[None, ...], stepped), axis=0)
        complete_valid = jnp.concatenate((jnp.asarray([True]), step_valid))
        return complete[save_indices], complete_valid[save_indices]

    if stochastic:
        assert realization is not None
        if realization.sample_shape:
            flat_keys = realization.path_keys.reshape((-1,))
            flat_signs = realization.path_signs.reshape((-1,))
            flat_increments = path_increments.reshape(
                (realization.num_paths, num_steps, path_increments.shape[-1])
            )
            flat_states, flat_matrix_valid = jax.vmap(one_path)(
                flat_keys,
                flat_signs,
                flat_increments,
            )
            states = flat_states.reshape(
                realization.sample_shape + (int(saved.size),) + spde.state_shape
            )
            matrix_valid = flat_matrix_valid.reshape(
                realization.sample_shape + (int(saved.size),)
            )
            times = jnp.broadcast_to(saved, realization.sample_shape + saved.shape)
            sample_shape = realization.sample_shape
        else:
            states, matrix_valid = one_path(
                realization.path_keys,
                realization.path_signs,
                path_increments,
            )
            times = saved
            sample_shape = ()
    else:
        states, matrix_valid = one_path(
            jr.key(0),
            jnp.asarray(1.0),
            path_increments,
        )
        times = saved
        sample_shape = ()
    state_axes = tuple(range(len(sample_shape) + 1, states.ndim))
    valid = (
        matrix_valid
        & jnp.isfinite(times)
        & jnp.all(jnp.isfinite(states), axis=state_axes)
    )
    stats_steps = (
        jnp.full(sample_shape, num_steps, dtype=jnp.int32)
        if sample_shape
        else jnp.asarray(num_steps, dtype=jnp.int32)
    )
    solver_name = (
        "SemilinearExponentialMilstein"
        if resolved_scheme == "exponential_milstein"
        else "SemilinearExponentialEuler"
    )
    return DifferentialSolution(
        times=times,
        states=states,
        valid=valid,
        sample_shape=sample_shape,
        backend_result="successful",
        stats={
            "num_steps": stats_steps,
            "matrix_function_converged": jnp.all(matrix_valid, axis=-1),
            "matrix_function_method": policy.method,
            "matrix_function_orthogonalization": policy.orthogonalization,
            "scheme": resolved_scheme,
            "exact_stochastic_convolution": exact_additive,
            "uses_realization_increments": bool(stochastic and not exact_additive),
        },
        event_mask=None,
        realization=realization,
        wiener_term_slices=spde.problem.wiener_term_slices,
        solver_name=solver_name,
        interpretation=spde.problem.interpretation,
        state_geometry_id=spde.problem.state_geometry_id,
        solver_id=f"solver:semilinear:{resolved_scheme}",
        resolved_method=f"{resolved_scheme}:{policy.method}",
        discretization_bundle=_semilinear_solution_bundle(
            spde,
            realization,
            num_steps,
            resolved_scheme,
        ),
    )


__all__ = [
    "SemilinearFallback",
    "SemilinearSPDEScheme",
    "exact_modal_stochastic_convolution",
    "solve_semilinear_spde",
]
