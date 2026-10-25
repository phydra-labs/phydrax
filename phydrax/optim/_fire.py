#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fast inertial relaxation engines (FIRE and FIRE2).

Both methods are force-driven: one evaluation per step, no line search, and an
inertial velocity whose power ``P = <v, F>`` (with ``F = -g`` restricted to free
variables) controls an adaptive time step and mixing parameter for every
optimizer group. Group reductions and displacement caps follow an
`OptimizerGroupLayout`, so a batch of independent systems, a whole reaction path
or a sharded system keeps one controller per group.

References: Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006) for FIRE and
Guénolé et al., Comput. Mater. Sci. 175, 109584 (2020) for FIRE2.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from math import isfinite, isnan
from typing import Any, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import PyTree

from .._strict import StrictModule
from ..typing import Bool, Float, Int32
from ._bounds import (
    _admit_trial,
    _bound_capabilities,
    _evaluate_force_driven,
    _force_driven_admission,
    _force_driven_inputs,
    _group_converged,
    _group_status,
    _ITERATING,
    _solve_force_driven,
    _tree_zeros_on_mask,
)
from ._group_layout import OptimizerGroupDim, OptimizerGroupLayout
from ._iterative._base import AbstractMinimizationMethod
from ._iterative._types import (
    Bounds,
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationTermination,
)


@final
class FIREState(StrictModule):
    """Complete stepwise state of `FIRE` or `FIRE2` for every optimizer group.

    ``parameters`` is the point the caller evaluates next and ``velocity`` the
    inertial velocity carried into that step. ``accepted_*`` is the last accepted
    evaluation, the reported iterate. ``time_step``, ``mixing`` and
    ``positive_steps`` are the adaptive controller (``Δt``, ``α`` and the count of
    consecutive positive-power steps). ``power`` and ``step_norm`` describe the
    latest step a group took. For the latest evaluation a group consumed,
    ``uphill`` records the non-positive-power branch, ``rolled_back`` an objective
    increase that rejected the evaluation (FIRE with ``downhill_check``), and
    ``trust_limited``/``bound_limited`` whether the maximum-step cap or the bounds
    shortened the displacement. Integer fields count events over the run.
    ``status`` is an `OptimizationStatus` per group as for
    `TrustScaledLBFGSState`.
    ``inverse_mass`` scales the FIRE force kick and is ``None`` for unit mass.
    """

    __strict_contract__ = True

    parameters: PyTree[Array]
    velocity: PyTree[Array]
    inverse_mass: PyTree[Array] | None
    accepted_parameters: PyTree[Array]
    accepted_gradient: PyTree[Array]
    accepted_value: Float[OptimizerGroupDim]
    has_accepted: Bool[OptimizerGroupDim]
    time_step: Float[OptimizerGroupDim]
    mixing: Float[OptimizerGroupDim]
    positive_steps: Int32[OptimizerGroupDim]
    power: Float[OptimizerGroupDim]
    optimality: Float[OptimizerGroupDim]
    maximum_site_gradient: Float[OptimizerGroupDim]
    initial_optimality: Float[OptimizerGroupDim]
    step_norm: Float[OptimizerGroupDim]
    status: Int32[OptimizerGroupDim]
    evaluation_accepted: Bool[OptimizerGroupDim]
    uphill: Bool[OptimizerGroupDim]
    rolled_back: Bool[OptimizerGroupDim]
    trust_limited: Bool[OptimizerGroupDim]
    bound_limited: Bool[OptimizerGroupDim]
    evaluations: Int32[OptimizerGroupDim]
    iterations: Int32[OptimizerGroupDim]
    uphill_steps: Int32[OptimizerGroupDim]
    rollbacks: Int32[OptimizerGroupDim]
    trust_limited_steps: Int32[OptimizerGroupDim]
    nonfinite_evaluations: Int32[OptimizerGroupDim]
    layout: OptimizerGroupLayout

    @property
    def accepted_step_size(self) -> Array:
        return self.time_step

    @property
    def damping(self) -> Array:
        return self.mixing

    @property
    def direction_fallbacks(self) -> Array:
        return self.uphill_steps

    @property
    def rejected_steps(self) -> Array:
        return self.rollbacks + self.nonfinite_evaluations


class _ControllerUpdate(NamedTuple):
    velocity: PyTree[Array]
    displacement: PyTree[Array]
    time_step: Array
    mixing: Array
    positive_steps: Array
    power: Array
    uphill: Array
    trust_limited: Array


def _validated_controls(
    *,
    initial_time_step: float,
    minimum_time_step: float,
    maximum_time_step: float,
    time_step_increase: float,
    time_step_decrease: float,
    initial_mixing: float,
    mixing_decay: float,
    positive_steps: int,
    maximum_step: float,
    active_tolerance: float,
    positive_steps_name: str,
) -> tuple[float, float, float, float, float, float, float, int, float, float]:
    initial, minimum, maximum, increase, decrease, mixing, decay = (
        float(value)
        for value in (
            initial_time_step,
            minimum_time_step,
            maximum_time_step,
            time_step_increase,
            time_step_decrease,
            initial_mixing,
            mixing_decay,
        )
    )
    if any(
        not isfinite(value) for value in (initial, minimum, maximum, increase, decrease)
    ):
        raise ValueError("Time-step controls must be finite.")
    if not 0.0 < minimum <= initial <= maximum:
        raise ValueError(
            "Time steps must satisfy 0 < minimum_time_step <= initial_time_step "
            "<= maximum_time_step."
        )
    if increase <= 1.0 or not 0.0 < decrease < 1.0:
        raise ValueError(
            "time_step_increase must exceed one and time_step_decrease lie in (0, 1)."
        )
    if not 0.0 < mixing < 1.0 or not 0.0 < decay < 1.0:
        raise ValueError("initial_mixing and mixing_decay must lie in (0, 1).")
    if isinstance(positive_steps, bool) or not isinstance(positive_steps, int):
        raise TypeError(f"{positive_steps_name} must be an int.")
    if positive_steps < 0:
        raise ValueError(f"{positive_steps_name} must be non-negative.")
    step = float(maximum_step)
    if isnan(step) or step <= 0.0:
        raise ValueError("maximum_step must be positive; inf disables the cap.")
    tolerance = float(active_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("active_tolerance must be finite and non-negative.")
    return (
        initial,
        minimum,
        maximum,
        increase,
        decrease,
        mixing,
        decay,
        positive_steps,
        step,
        tolerance,
    )


def _inverse_mass(
    inverse_mass: PyTree[ArrayLike] | None,
    parameters: PyTree[Array],
    /,
) -> PyTree[Array] | None:
    """Admit per-site inverse masses broadcast over trailing site axes."""

    if inverse_mass is None:
        return None
    if jax.tree.structure(inverse_mass) != jax.tree.structure(parameters):
        raise ValueError("inverse_mass must have the parameter PyTree structure.")

    def admit(value: ArrayLike, parameter: Array) -> Array:
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("inverse_mass leaves must be real floating arrays.")
        shape = parameter.shape
        if array.ndim > len(shape) or tuple(array.shape) != shape[: array.ndim]:
            raise ValueError(
                "Each inverse_mass leaf shape must be a leading prefix of its "
                "parameter leaf shape."
            )
        if not isinstance(array, core.Tracer):
            host = np.asarray(array)
            if not np.all(np.isfinite(host)) or np.any(host < 0.0):
                raise ValueError("inverse_mass must be finite and non-negative.")
        return jnp.reshape(
            array.astype(parameter.dtype),
            array.shape + (1,) * (len(shape) - array.ndim),
        )

    return jax.tree.map(admit, inverse_mass, parameters)


def _initial_fire_state(
    parameters: PyTree[ArrayLike],
    layout: OptimizerGroupLayout | None,
    value_dtype: DTypeLike | None,
    inverse_mass: PyTree[ArrayLike] | None,
    initial_time_step: float,
    initial_mixing: float,
    /,
) -> FIREState:
    parameters_, layout_, dtype, value_dtype_ = _force_driven_admission(
        parameters, layout, value_dtype
    )
    groups = layout_.group_count
    zero = jnp.zeros((groups,), dtype=dtype)
    unset = jnp.full((groups,), jnp.nan, dtype=dtype)
    no = jnp.zeros((groups,), dtype=jnp.bool_)
    none = jnp.zeros((groups,), dtype=jnp.int32)
    zero_tree = jax.tree.map(jnp.zeros_like, parameters_)
    return FIREState(
        parameters=parameters_,
        velocity=zero_tree,
        inverse_mass=_inverse_mass(inverse_mass, parameters_),
        accepted_parameters=parameters_,
        accepted_gradient=zero_tree,
        accepted_value=jnp.full((groups,), jnp.nan, dtype=value_dtype_),
        has_accepted=no,
        time_step=jnp.full((groups,), initial_time_step, dtype=dtype),
        mixing=jnp.full((groups,), initial_mixing, dtype=dtype),
        positive_steps=none,
        power=unset,
        optimality=unset,
        maximum_site_gradient=unset,
        initial_optimality=unset,
        step_norm=zero,
        status=jnp.full((groups,), _ITERATING, dtype=jnp.int32),
        evaluation_accepted=no,
        uphill=no,
        rolled_back=no,
        trust_limited=no,
        bound_limited=no,
        evaluations=none,
        iterations=none,
        uphill_steps=none,
        rollbacks=none,
        trust_limited_steps=none,
        nonfinite_evaluations=none,
        layout=layout_,
    )


def _force_scaled_speed(velocity_squared: Array, force_squared: Array, /) -> Array:
    """``sqrt(v·v / F·F)``, zero when the group has no force."""

    has_force = force_squared > 0.0
    return jnp.where(
        has_force,
        jnp.sqrt(velocity_squared / jnp.where(has_force, force_squared, 1.0)),
        0.0,
    )


def _fire_step(
    state: FIREState,
    value: ArrayLike,
    gradient: PyTree[ArrayLike],
    /,
    *,
    active_tolerance: float,
    downhill_check: bool,
    bounds: Bounds | None,
    termination: OptimizationTermination | None,
    converged: ArrayLike | None,
    active: ArrayLike | None,
    update: Callable[[FIREState, PyTree[Array], Array], _ControllerUpdate],
) -> FIREState:
    """Shared evaluation, acceptance, rollback and admission around one update."""

    if not isinstance(state, FIREState):
        raise TypeError("state must be a FIREState.")
    layout = state.layout
    x = state.parameters
    dtype = state.time_step.dtype
    value_, gradient_, active_, converged_ = _force_driven_inputs(
        layout, x, state.accepted_value.dtype, value, gradient, active, converged
    )
    evaluation = _evaluate_force_driven(
        layout, bounds, x, value_, gradient_, active_tolerance
    )
    finite_active = active_ & evaluation.finite
    initial_optimality = jnp.where(
        finite_active & ~state.has_accepted,
        evaluation.optimality,
        state.initial_optimality,
    )
    converged_ = _group_converged(
        termination, evaluation.optimality, initial_optimality, converged_
    )
    moving = finite_active & ~converged_
    if downhill_check:
        rollback = moving & state.has_accepted & (value_ > state.accepted_value)
        accepted_free = state.accepted_gradient
        if bounds is not None:
            accepted_free = _tree_zeros_on_mask(
                accepted_free,
                bounds.active_mask(
                    state.accepted_parameters,
                    state.accepted_gradient,
                    tolerance=active_tolerance,
                ),
            )
        origin = layout.select(rollback, state.accepted_parameters, x)
        free_gradient = layout.select(rollback, accepted_free, evaluation.free_gradient)
    else:
        rollback = jnp.zeros_like(moving)
        origin = x
        free_gradient = evaluation.free_gradient
    accepted = finite_active & ~rollback
    force = jax.tree.map(jnp.negative, free_gradient)
    controlled = update(state, force, rollback)

    unprojected = jax.tree.map(jnp.add, origin, controlled.displacement)
    trial, bound_limited = _admit_trial(layout, bounds, unprojected)
    velocity = controlled.velocity
    if bounds is not None:
        # A coordinate held by a bound loses the velocity that pushed it there.
        velocity = jax.tree.map(
            lambda v, admitted, proposed: jnp.where(admitted != proposed, 0.0, v),
            velocity,
            trial,
            unprojected,
        )
    failed = active_ & ~evaluation.finite
    retreat = failed & state.has_accepted
    held = layout.select(retreat, state.accepted_parameters, x)
    zero_velocity = jax.tree.map(jnp.zeros_like, state.velocity)
    held_velocity = layout.select(failed, zero_velocity, state.velocity)
    step_norm = layout.norm(jax.tree.map(jnp.subtract, trial, origin)).astype(dtype)
    uphill = moving & controlled.uphill
    trust_limited = moving & controlled.trust_limited
    bound_limited = moving & bound_limited
    return dataclasses.replace(
        state,
        parameters=layout.select(moving, trial, held),
        velocity=layout.select(moving, velocity, held_velocity),
        accepted_parameters=layout.select(accepted, x, state.accepted_parameters),
        accepted_gradient=layout.select(accepted, gradient_, state.accepted_gradient),
        accepted_value=jnp.where(accepted, value_, state.accepted_value),
        has_accepted=state.has_accepted | accepted,
        time_step=jnp.where(moving, controlled.time_step, state.time_step).astype(dtype),
        mixing=jnp.where(moving, controlled.mixing, state.mixing).astype(dtype),
        positive_steps=jnp.where(
            moving, controlled.positive_steps, state.positive_steps
        ).astype(jnp.int32),
        power=jnp.where(moving, controlled.power, state.power).astype(dtype),
        optimality=jnp.where(accepted, evaluation.optimality, state.optimality),
        maximum_site_gradient=jnp.where(
            accepted, evaluation.maximum_site_gradient, state.maximum_site_gradient
        ),
        initial_optimality=initial_optimality,
        step_norm=jnp.where(moving, step_norm, state.step_norm),
        status=_group_status(state.status, active_, evaluation.finite, converged_),
        evaluation_accepted=jnp.where(active_, accepted, state.evaluation_accepted),
        uphill=jnp.where(active_, uphill, state.uphill),
        rolled_back=jnp.where(active_, rollback, state.rolled_back),
        trust_limited=jnp.where(active_, trust_limited, state.trust_limited),
        bound_limited=jnp.where(active_, bound_limited, state.bound_limited),
        evaluations=state.evaluations + active_.astype(jnp.int32),
        iterations=state.iterations + moving.astype(jnp.int32),
        uphill_steps=state.uphill_steps + uphill.astype(jnp.int32),
        rollbacks=state.rollbacks + rollback.astype(jnp.int32),
        trust_limited_steps=state.trust_limited_steps + trust_limited.astype(jnp.int32),
        nonfinite_evaluations=state.nonfinite_evaluations + failed.astype(jnp.int32),
    )


@final
class FIRE(AbstractMinimizationMethod):
    """Fast inertial relaxation engine with mass-aware kicks and per-site caps.

    One step per evaluation, per group: measure ``P = <v, F>``,
    ``v·v`` and ``F·F`` on the incoming velocity; when ``P > 0`` count a positive
    step, mix ``v ← (1-α) v + α sqrt(v·v / F·F) F`` with the updated ``α``, and
    once ``minimum_positive_steps`` consecutive positive steps are reached grow
    ``Δt ← min(Δt·time_step_increase, maximum_time_step)`` and decay
    ``α ← α·mixing_decay``; when ``P <= 0`` zero the velocity, shrink
    ``Δt ← max(Δt·time_step_decrease, minimum_time_step)``, reset ``α`` and the
    count. Then kick ``v ← v + Δt M⁻¹ F`` and move ``Δx = Δt v`` with the incoming
    ``Δt``, shortening every site displacement longer than ``maximum_step``.

    With ``downhill_check`` an accepted point whose successor has a higher
    objective value is restored: the evaluation is rejected, the step restarts
    from the accepted point with its own force, and the non-positive-power branch
    is applied with the already shrunk ``Δt`` (so the rejected trial is not
    proposed again). Without it the method never compares objective values and
    does not claim monotone decrease.

    With bounds, the force acts on free variables only, the displacement is
    projected onto the bounds, and a coordinate held by a bound loses its velocity.
    """

    initial_time_step: float = eqx.field(static=True)
    minimum_time_step: float = eqx.field(static=True)
    maximum_time_step: float = eqx.field(static=True)
    time_step_increase: float = eqx.field(static=True)
    time_step_decrease: float = eqx.field(static=True)
    initial_mixing: float = eqx.field(static=True)
    mixing_decay: float = eqx.field(static=True)
    minimum_positive_steps: int = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    downhill_check: bool = eqx.field(static=True)
    active_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_time_step: float = 0.1,
        minimum_time_step: float = 0.01,
        maximum_time_step: float = 1.0,
        time_step_increase: float = 1.1,
        time_step_decrease: float = 0.5,
        initial_mixing: float = 0.1,
        mixing_decay: float = 0.99,
        minimum_positive_steps: int = 5,
        maximum_step: float = 0.2,
        downhill_check: bool = False,
        active_tolerance: float = 1e-10,
    ) -> None:
        if not isinstance(downhill_check, bool):
            raise TypeError("downhill_check must be a bool.")
        (
            self.initial_time_step,
            self.minimum_time_step,
            self.maximum_time_step,
            self.time_step_increase,
            self.time_step_decrease,
            self.initial_mixing,
            self.mixing_decay,
            self.minimum_positive_steps,
            self.maximum_step,
            self.active_tolerance,
        ) = _validated_controls(
            initial_time_step=initial_time_step,
            minimum_time_step=minimum_time_step,
            maximum_time_step=maximum_time_step,
            time_step_increase=time_step_increase,
            time_step_decrease=time_step_decrease,
            initial_mixing=initial_mixing,
            mixing_decay=mixing_decay,
            positive_steps=minimum_positive_steps,
            maximum_step=maximum_step,
            active_tolerance=active_tolerance,
            positive_steps_name="minimum_positive_steps",
        )
        self.downhill_check = downhill_check

    @property
    def method_id(self) -> str:
        return "fire/downhill-check" if self.downhill_check else "fire"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return _bound_capabilities()

    def init(
        self,
        parameters: PyTree[ArrayLike],
        /,
        *,
        layout: OptimizerGroupLayout | None = None,
        value_dtype: DTypeLike | None = None,
        inverse_mass: PyTree[ArrayLike] | None = None,
    ) -> FIREState:
        """Prepare a state at rest whose ``parameters`` are evaluated first.

        ``inverse_mass`` leaves are non-negative per-site inverse masses whose
        shapes are leading prefixes of the parameter leaves; zero holds a site.
        """

        return _initial_fire_state(
            parameters,
            layout,
            value_dtype,
            inverse_mass,
            self.initial_time_step,
            self.initial_mixing,
        )

    def _update(
        self, state: FIREState, force: PyTree[Array], rollback: Array, /
    ) -> _ControllerUpdate:
        layout = state.layout
        dtype = state.time_step.dtype
        velocity = state.velocity
        power = layout.inner(velocity, force).astype(dtype)
        velocity_squared = layout.squared_norm(velocity).astype(dtype)
        force_squared = layout.squared_norm(force).astype(dtype)
        positive = (power > 0.0) & ~rollback
        count = jnp.where(positive, state.positive_steps + 1, 0).astype(jnp.int32)
        grow = positive & (count >= self.minimum_positive_steps)
        time_step = state.time_step
        time_step_ = jnp.where(
            grow,
            jnp.minimum(time_step * self.time_step_increase, self.maximum_time_step),
            jnp.where(
                positive,
                time_step,
                jnp.maximum(time_step * self.time_step_decrease, self.minimum_time_step),
            ),
        ).astype(dtype)
        mixing = jnp.where(
            grow,
            state.mixing * self.mixing_decay,
            jnp.where(positive, state.mixing, self.initial_mixing),
        ).astype(dtype)
        speed = _force_scaled_speed(velocity_squared, force_squared)
        mixed = jax.tree.map(
            jnp.add,
            layout.scale(1.0 - mixing, velocity),
            layout.scale(mixing * speed, force),
        )
        velocity = layout.select(positive, mixed, jax.tree.map(jnp.zeros_like, velocity))
        # The incoming step moves the iterate; after a rollback the shrunk step is
        # used so the rejected trial point is not proposed again.
        motion_step = jnp.where(rollback, time_step_, time_step)
        kick = layout.scale(motion_step, force)
        if state.inverse_mass is not None:
            kick = jax.tree.map(jnp.multiply, kick, state.inverse_mass)
        velocity = jax.tree.map(jnp.add, velocity, kick)
        displacement, trust_limited = layout.clamp_sites(
            layout.scale(motion_step, velocity), self.maximum_step
        )
        return _ControllerUpdate(
            velocity=velocity,
            displacement=displacement,
            time_step=time_step_,
            mixing=mixing,
            positive_steps=count,
            power=power,
            uphill=~positive,
            trust_limited=trust_limited,
        )

    def step(
        self,
        state: FIREState,
        value: ArrayLike,
        gradient: PyTree[ArrayLike],
        /,
        *,
        bounds: Bounds | None = None,
        termination: OptimizationTermination | None = None,
        converged: ArrayLike | None = None,
        active: ArrayLike | None = None,
    ) -> FIREState:
        """Consume one evaluation at ``state.parameters`` and propose the next point.

        Acceptance, convergence, rollback and group masking follow
        `TrustScaledLBFGS.step`.
        """

        return _fire_step(
            state,
            value,
            gradient,
            active_tolerance=self.active_tolerance,
            downhill_check=self.downhill_check,
            bounds=bounds,
            termination=termination,
            converged=converged,
            active=active,
            update=self._update,
        )

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return _solve_force_driven(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
            globalization=(
                "inertial-power-control/downhill-rollback"
                if self.downhill_check
                else "inertial-power-control"
            ),
            notes=(
                "Force-driven inertial relaxation with unit mass; an objective "
                "increase restores the last accepted evaluation."
                if self.downhill_check
                else "Force-driven inertial relaxation with unit mass; objective "
                "decrease is not enforced."
            ),
        )


@final
class FIRE2(AbstractMinimizationMethod):
    """FIRE2 with a half-step kick, modified mixing and uphill half-step retreat.

    One step per evaluation, per group: half-kick ``v' = v + Δt F`` and measure
    ``P = <v', F>``, ``v'·v'`` and ``F·F``; when ``P > 0`` count a positive step and,
    once the count exceeds ``delay_steps``, grow
    ``Δt ← min(Δt·time_step_increase, maximum_time_step)`` and decay
    ``α ← α·mixing_decay``; when ``P <= 0`` reset the count and ``α`` and shrink
    ``Δt ← max(Δt·time_step_decrease, minimum_time_step)``. The mixed velocity is
    ``(1-α) v + ((1-α) Δt_old + α sqrt(v'·v' / F·F)) F``; the raw displacement is
    ``Δt v`` downhill and ``-Δt v / 2`` uphill, after which an uphill velocity is
    zeroed. When the largest site displacement exceeds ``maximum_step`` the whole
    group displacement and ``Δt`` are scaled by the same factor.

    Objective values are never compared; the method does not claim monotone
    decrease. With bounds, the force acts on free variables only, the displacement
    is projected onto the bounds, and a coordinate held by a bound loses its
    velocity.
    """

    initial_time_step: float = eqx.field(static=True)
    minimum_time_step: float = eqx.field(static=True)
    maximum_time_step: float = eqx.field(static=True)
    time_step_increase: float = eqx.field(static=True)
    time_step_decrease: float = eqx.field(static=True)
    initial_mixing: float = eqx.field(static=True)
    mixing_decay: float = eqx.field(static=True)
    delay_steps: int = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    active_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_time_step: float = 0.005,
        minimum_time_step: float = 0.005,
        maximum_time_step: float = 0.08,
        time_step_increase: float = 1.05,
        time_step_decrease: float = 0.75,
        initial_mixing: float = 0.09,
        mixing_decay: float = 0.985,
        delay_steps: int = 60,
        maximum_step: float = 0.1,
        active_tolerance: float = 1e-10,
    ) -> None:
        (
            self.initial_time_step,
            self.minimum_time_step,
            self.maximum_time_step,
            self.time_step_increase,
            self.time_step_decrease,
            self.initial_mixing,
            self.mixing_decay,
            self.delay_steps,
            self.maximum_step,
            self.active_tolerance,
        ) = _validated_controls(
            initial_time_step=initial_time_step,
            minimum_time_step=minimum_time_step,
            maximum_time_step=maximum_time_step,
            time_step_increase=time_step_increase,
            time_step_decrease=time_step_decrease,
            initial_mixing=initial_mixing,
            mixing_decay=mixing_decay,
            positive_steps=delay_steps,
            maximum_step=maximum_step,
            active_tolerance=active_tolerance,
            positive_steps_name="delay_steps",
        )

    @property
    def method_id(self) -> str:
        return "fire2"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return _bound_capabilities()

    def init(
        self,
        parameters: PyTree[ArrayLike],
        /,
        *,
        layout: OptimizerGroupLayout | None = None,
        value_dtype: DTypeLike | None = None,
    ) -> FIREState:
        """Prepare a state at rest whose ``parameters`` are evaluated first."""

        return _initial_fire_state(
            parameters,
            layout,
            value_dtype,
            None,
            self.initial_time_step,
            self.initial_mixing,
        )

    def _update(
        self, state: FIREState, force: PyTree[Array], rollback: Array, /
    ) -> _ControllerUpdate:
        del rollback
        layout = state.layout
        dtype = state.time_step.dtype
        velocity = state.velocity
        time_step = state.time_step
        kicked = jax.tree.map(jnp.add, velocity, layout.scale(time_step, force))
        power = layout.inner(kicked, force).astype(dtype)
        kicked_squared = layout.squared_norm(kicked).astype(dtype)
        force_squared = layout.squared_norm(force).astype(dtype)
        positive = power > 0.0
        count = jnp.where(positive, state.positive_steps + 1, 0).astype(jnp.int32)
        grow = positive & (count > self.delay_steps)
        time_step_ = jnp.where(
            grow,
            jnp.minimum(time_step * self.time_step_increase, self.maximum_time_step),
            jnp.where(
                positive,
                time_step,
                jnp.maximum(time_step * self.time_step_decrease, self.minimum_time_step),
            ),
        ).astype(dtype)
        mixing = jnp.where(
            grow,
            state.mixing * self.mixing_decay,
            jnp.where(positive, state.mixing, self.initial_mixing),
        ).astype(dtype)
        speed = _force_scaled_speed(kicked_squared, force_squared)
        mixed = jax.tree.map(
            jnp.add,
            layout.scale(1.0 - mixing, velocity),
            layout.scale((1.0 - mixing) * time_step + mixing * speed, force),
        )
        raw = layout.scale(jnp.where(positive, time_step_, -0.5 * time_step_), mixed)
        largest = layout.site_norm_max(raw).astype(dtype)
        trust_limited = largest > self.maximum_step
        factor = jnp.where(
            trust_limited,
            self.maximum_step / jnp.where(trust_limited, largest, 1.0),
            1.0,
        ).astype(dtype)
        return _ControllerUpdate(
            velocity=layout.select(positive, mixed, jax.tree.map(jnp.zeros_like, mixed)),
            displacement=layout.scale(factor, raw),
            time_step=time_step_ * factor,
            mixing=mixing,
            positive_steps=count,
            power=power,
            uphill=~positive,
            trust_limited=trust_limited,
        )

    def step(
        self,
        state: FIREState,
        value: ArrayLike,
        gradient: PyTree[ArrayLike],
        /,
        *,
        bounds: Bounds | None = None,
        termination: OptimizationTermination | None = None,
        converged: ArrayLike | None = None,
        active: ArrayLike | None = None,
    ) -> FIREState:
        """Consume one evaluation at ``state.parameters`` and propose the next point.

        Acceptance, convergence, rollback and group masking follow
        `TrustScaledLBFGS.step`.
        """

        if state.inverse_mass is not None:
            raise ValueError("FIRE2 uses unit mass; the state carries inverse masses.")
        return _fire_step(
            state,
            value,
            gradient,
            active_tolerance=self.active_tolerance,
            downhill_check=False,
            bounds=bounds,
            termination=termination,
            converged=converged,
            active=active,
            update=self._update,
        )

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return _solve_force_driven(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
            globalization="inertial-power-control/half-step-retreat",
            notes=(
                "Force-driven FIRE2 inertial relaxation with unit mass; objective "
                "decrease is not enforced."
            ),
        )


__all__ = ["FIRE", "FIRE2", "FIREState"]
