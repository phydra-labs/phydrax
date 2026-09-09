#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native differential problems and sampled observations of regional dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...ein import contract
from ...series import SampledSeries, SeriesSupport
from ...solver import (
    DelayDifferentialProblem,
    DelayHistoryWindow,
    DelayValues,
    DifferentialProblem,
    DifferentialSolution,
    FunctionalDelay,
    MemoryEquationSolution,
    solve_diffrax,
    solve_diffrax_delay,
    solve_diffrax_delay_segmented,
)
from ._bold import _balloon_valid, balloon_equilibrium, BalloonWindkessel, NeuralBOLDDrive
from ._regional import _delayed_coupling, Hopf, RegionalConnectivity, WilsonCowan


class _RegionalDynamics(StrictModule):
    connectivity: RegionalConnectivity
    model: WilsonCowan | Hopf
    history: Callable[[Array, Any], ArrayLike]
    drive: Callable[[Array, Array, Any], ArrayLike] | None
    user_args: Any
    balloon: BalloonWindkessel | None
    bold_drive: NeuralBOLDDrive | None
    initial_balloon: Array | None


def _regional_history(time: Array, context: _RegionalDynamics, /) -> Array:
    state = jnp.asarray(context.history(time, context.user_args), dtype=float)
    if state.shape != (context.connectivity.region_count, 2):
        raise ValueError(
            "Neural prehistory must return [region,2] in declared region order."
        )
    bad = ~jnp.all(jnp.isfinite(state))
    if isinstance(context.model, WilsonCowan):
        bad = bad | jnp.any((state < 0.0) | (state > 1.0))
    state = eqx.error_if(
        state,
        bad,
        "Neural prehistory must be finite and Wilson-Cowan fractions must lie in [0,1].",
    )
    if context.initial_balloon is None:
        return state
    return jnp.concatenate((state, context.initial_balloon), axis=-1)


def _functional_coupling(
    time: Array, state: Array, history: DelayHistoryWindow, context: _RegionalDynamics, /
) -> Array:
    del time
    coupling = _delayed_coupling(context.connectivity, state, history)
    # Native functional delay observations are state-shaped. Hemodynamics is
    # never queried as a coupling coordinate and needs no additional history.
    return jnp.zeros_like(state).at[:, :2].set(coupling)


def _regional_rhs(
    time: Array, state: Array, delayed: Array, context: _RegionalDynamics, /
) -> Array:
    neural = state[:, :2]
    connectivity = context.connectivity
    instant_weights = jnp.where(connectivity.delays_s == 0.0, connectivity.weights, 0.0)
    incoming = contract("ij,jc->ic", instant_weights, neural) + delayed[:, :2]
    external = (
        jnp.zeros_like(neural)
        if context.drive is None
        else jnp.asarray(context.drive(time, neural, context.user_args))
    )
    if external.shape != neural.shape:
        raise ValueError("Regional external drive must return [region,2].")
    neural_rate = context.model(
        neural, incoming, external, jnp.sum(connectivity.weights, axis=1)
    )
    if context.balloon is None:
        return neural_rate
    assert context.bold_drive is not None
    balloon_rate = context.balloon(state[:, 2:], context.bold_drive(neural))
    return jnp.concatenate((neural_rate, balloon_rate), axis=-1)


def _delay_rhs(
    time: Array, state: Array, memory: DelayValues, context: _RegionalDynamics, /
) -> Array:
    return _regional_rhs(time, state, memory["regional_coupling"], context)


def _ordinary_rhs(time: Array, state: Array, context: _RegionalDynamics, /) -> Array:
    return _regional_rhs(time, state, jnp.zeros_like(state), context)


def _problem(
    connectivity,
    model,
    history,
    *,
    t0,
    t1,
    drive,
    args,
    balloon,
    bold_drive,
    initial_balloon,
    problem_id,
):
    if not isinstance(connectivity, RegionalConnectivity):
        raise TypeError("connectivity must be RegionalConnectivity.")
    if not isinstance(model, (WilsonCowan, Hopf)):
        raise TypeError("model must be WilsonCowan or Hopf.")
    if not callable(history) or (drive is not None and not callable(drive)):
        raise TypeError("history must be callable; drive must be callable or None.")
    for owner in (model, balloon):
        for leaf in jax.tree.leaves(owner):
            if (
                eqx.is_array(leaf)
                and leaf.ndim == 1
                and leaf.size not in (1, connectivity.region_count)
            ):
                raise ValueError("Regional parameter vectors must match region_count.")
    if bold_drive is not None:
        for value in (bold_drive.component_weights, bold_drive.baseline):
            if value.ndim == 2 and value.shape[0] not in (1, connectivity.region_count):
                raise ValueError("BOLD drive arrays must match region_count.")
        if bold_drive.gain.ndim == 1 and bold_drive.gain.size not in (
            1,
            connectivity.region_count,
        ):
            raise ValueError("BOLD drive gain must match region_count.")
    context = _RegionalDynamics(
        connectivity, model, history, drive, args, balloon, bold_drive, initial_balloon
    )
    if connectivity.delayed_edge_count:
        term = FunctionalDelay(
            "regional_coupling",
            _functional_coupling,
            (connectivity.minimum_delay_s, connectivity.maximum_delay_s),
            discontinuity_lags=jnp.asarray(connectivity.propagation_lags_s),
        )
        return DelayDifferentialProblem(
            _delay_rhs,
            _regional_history,
            (term,),
            t0=t0,
            t1=t1,
            args=context,
            problem_id=problem_id,
        )
    return DifferentialProblem(
        _ordinary_rhs,
        _regional_history(jnp.asarray(t0), context),
        t0=t0,
        t1=t1,
        args=context,
        problem_id=problem_id,
    )


def regional_problem(
    connectivity: RegionalConnectivity,
    model: WilsonCowan | Hopf,
    history: Callable[[Array, Any], ArrayLike],
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    drive: Callable[[Array, Array, Any], ArrayLike] | None = None,
    args: Any = None,
    problem_id: str = "regional-neural-dynamics",
) -> DifferentialProblem | DelayDifferentialProblem:
    """Build a native problem with explicit prehistory ``history(time_s,args)``.

    Positive delays use the native accepted-interpolant method of steps; an
    entirely instantaneous map lowers to an ordinary DifferentialProblem.
    ``drive(time_s, neural[region,2], args)`` returns the model's external input.
    """
    return _problem(
        connectivity,
        model,
        history,
        t0=t0,
        t1=t1,
        drive=drive,
        args=args,
        balloon=None,
        bold_drive=None,
        initial_balloon=None,
        problem_id=problem_id,
    )


def regional_bold_problem(
    connectivity: RegionalConnectivity,
    model: WilsonCowan | Hopf,
    history: Callable[[Array, Any], ArrayLike],
    balloon: BalloonWindkessel,
    bold_drive: NeuralBOLDDrive,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    drive: Callable[[Array, Array, Any], ArrayLike] | None = None,
    args: Any = None,
    initial_balloon: ArrayLike | None = None,
    problem_id: str = "regional-neural-bold-dynamics",
) -> DifferentialProblem | DelayDifferentialProblem:
    """Joint neural/BOLD integration, not post-hoc resetting of a response kernel.

    State columns are E,I (or x,y), then s,log(f),log(v),log(q). The default
    initial Balloon state is resting equilibrium; an explicit state overrides
    it. Native segmented continuation retains the complete six-coordinate state
    and history. ``history`` describes neural prehistory only; its BOLD columns
    are never used in delayed coupling.
    """
    if not isinstance(connectivity, RegionalConnectivity):
        raise TypeError("connectivity must be RegionalConnectivity.")
    if not isinstance(balloon, BalloonWindkessel) or not isinstance(
        bold_drive, NeuralBOLDDrive
    ):
        raise TypeError(
            "balloon and bold_drive must be BalloonWindkessel and NeuralBOLDDrive."
        )
    initial = (
        balloon_equilibrium(connectivity.region_count)
        if initial_balloon is None
        else jnp.asarray(initial_balloon, dtype=float)
    )
    if initial.shape != (connectivity.region_count, 4):
        raise ValueError("initial_balloon must have shape [region,4].")
    initial = eqx.error_if(
        initial,
        ~jnp.all(_balloon_valid(initial)),
        "Initial Balloon state must represent finite strictly positive flow, volume and deoxyhemoglobin.",
    )
    return _problem(
        connectivity,
        model,
        history,
        t0=t0,
        t1=t1,
        drive=drive,
        args=args,
        balloon=balloon,
        bold_drive=bold_drive,
        initial_balloon=initial,
        problem_id=problem_id,
    )


class RegionalSolution(StrictModule):
    """Native solver evidence plus time-major labelled sampled neural/BOLD data."""

    native: DifferentialSolution | MemoryEquationSolution
    neural: SampledSeries
    bold: SampledSeries | None
    region_ids: tuple[str, ...] = eqx.field(static=True)

    @property
    def continuation(self):
        """The native segmented continuation, or None for a nonsegmented solve."""
        return (
            self.native.continuation
            if isinstance(self.native, MemoryEquationSolution)
            else None
        )


def solve_regional(
    problem: DifferentialProblem | DelayDifferentialProblem,
    /,
    *,
    save_times: ArrayLike,
    segmented: bool = False,
    **solver_options: Any,
) -> RegionalSolution:
    """Solve via native differential/delay solvers and retain their full evidence.

    ``solver_options`` are passed to exactly the selected native solver. For
    positive delays, ``segmented=True`` enables bounded native history and
    ``continuation=previous.continuation``. Supply the same original problem
    when resuming; a restarted neural initial state is not a delay continuation.
    Ordinary all-zero-delay problems use ``solve_diffrax`` and reject segmented
    options, rather than introducing a fictitious lag.
    """
    if not isinstance(
        problem, (DifferentialProblem, DelayDifferentialProblem)
    ) or not isinstance(problem.args, _RegionalDynamics):
        raise TypeError(
            "problem must be constructed by regional_problem or regional_bold_problem."
        )
    if not isinstance(segmented, bool):
        raise TypeError("segmented must be a bool.")
    if isinstance(problem, DelayDifferentialProblem):
        solve = solve_diffrax_delay_segmented if segmented else solve_diffrax_delay
    else:
        if segmented:
            raise ValueError(
                "All-zero-delay regional dynamics use the native ordinary solver, not segmented delay execution."
            )
        solve = solve_diffrax
    native = solve(problem, save_times=save_times, **solver_options)
    context = problem.args
    support = SeriesSupport(
        native.times,
        node_valid=native.valid,
        coordinate_name="time_s",
        coordinate_id="neuroscience:seconds",
    )
    neural_values = native.states[:, :, :2]
    neural_valid = jnp.isfinite(neural_values)
    if isinstance(context.model, WilsonCowan):
        neural_valid = neural_valid & (neural_values >= 0.0) & (neural_values <= 1.0)
    neural = SampledSeries(
        support,
        neural_values,
        value_valid=neural_valid,
        series_id=f"{problem.problem_id}:neural",
    )
    bold = None
    if context.balloon is not None:
        balloon_state = native.states[:, :, 2:]
        values = context.balloon.bold(balloon_state)
        valid = (
            _balloon_valid(balloon_state)
            & jnp.isfinite(values)
            & jnp.all(neural_valid, axis=-1)
        )
        bold = SampledSeries(
            support,
            values,
            value_valid=valid,
            series_id=f"{problem.problem_id}:bold-fraction",
        )
    return RegionalSolution(native, neural, bold, context.connectivity.region_ids)


__all__ = [
    "RegionalSolution",
    "regional_bold_problem",
    "regional_problem",
    "solve_regional",
]
