#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from operator import index
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import FaceVelocity, PreparedMACOperators
from ...equations._incompressible import IncompressibleFlowProblem
from ...equations._mac_incompressible import (
    compile_mac_incompressible_flow,
    CompiledMACIncompressibleDynamics,
)
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
)
from ...linalg.svd import svd as solve_svd, SVDProblem, SVDSolvePolicy
from ...solver._fixed_step import AbstractSSPRKFixedStepMethod
from ...solver._mac_viscous import (
    MACIMEXEulerMethod,
    MACSBDF2Method,
    MACSBDF2State,
)
from ._statistics import _mac_face_to_cell


MACFlowControlKind = Literal[
    "pressure_gradient",
    "bulk_velocity",
    "frozen_density_mass_flux",
]
MACFlowControlMethodKind = Literal["ssprk", "imex_euler", "sbdf2"]


class _ConstantMACFlowTargetSchedule(StrictModule, NonTrainableState):
    value: Array
    schedule_id: str = eqx.field(static=True)

    def __init__(self, value: ArrayLike, /):
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            values = values.reshape((1,))
        if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError(
                "A constant MAC flow-control target must be a finite vector."
            )
        self.value = jnp.asarray(values)
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "constant-mac-flow-control-target-schedule",
                "value": values.tolist(),
            }
        )

    def __call__(self, time: Array, /) -> Array:
        del time
        return self.value


class MACFlowControlTarget(StrictModule, NonTrainableState):
    """An explicitly identified pressure-gradient or integral-flow target.

    ``frozen_density_mass_flux`` is deliberately a separate target kind. Its
    positive cell-density field is frozen into both the observable and the
    pressure-gradient acceleration for the complete method-stage response map.
    A callable schedule must carry a caller-owned ``schedule_id``; constant
    vectors receive a content identity automatically.
    """

    schedule: Callable[[Array], ArrayLike]
    frozen_density: Array | None
    kind: MACFlowControlKind = eqx.field(static=True)
    axes: tuple[int, ...] = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    density_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MACFlowControlKind,
        axes: Sequence[int],
        target: ArrayLike | Callable[[Array], ArrayLike],
        /,
        *,
        reference_density: float = 1.0,
        frozen_density: ArrayLike | None = None,
        schedule_id: str | None = None,
        density_id: str | None = None,
    ):
        if kind not in (
            "pressure_gradient",
            "bulk_velocity",
            "frozen_density_mass_flux",
        ):
            raise ValueError(f"Unknown MAC flow-control target kind {kind!r}.")
        raw_axes = tuple(axes)
        if any(isinstance(axis, bool) for axis in raw_axes):
            raise TypeError("MAC flow-control axes must be integer indices.")
        axes_ = tuple(index(axis) for axis in raw_axes)
        if not axes_ or len(axes_) > 3 or len(set(axes_)) != len(axes_):
            raise ValueError("MAC flow control requires one to three distinct axes.")
        if any(axis < 0 or axis >= 3 for axis in axes_):
            raise ValueError("MAC flow-control axes must lie in [0, 3).")
        density = float(reference_density)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("reference_density must be positive and finite.")

        if callable(target):
            identifier = "" if schedule_id is None else str(schedule_id)
            if not identifier:
                raise ValueError("A callable target schedule requires schedule_id.")
            schedule = target
        else:
            if schedule_id is not None:
                raise ValueError("schedule_id is derived for a constant target vector.")
            constant = _ConstantMACFlowTargetSchedule(target)
            if constant.value.shape != (len(axes_),):
                raise ValueError("Target-vector length must match the controlled axes.")
            schedule = constant
            identifier = constant.schedule_id

        if kind == "frozen_density_mass_flux":
            if frozen_density is None:
                raise ValueError(
                    "A frozen-density mass-flux target requires frozen_density."
                )
            raw_density = np.asarray(frozen_density, dtype=float)
            if (
                raw_density.ndim not in (2, 3)
                or raw_density.size == 0
                or np.any(~np.isfinite(raw_density))
                or np.any(raw_density <= 0.0)
            ):
                raise ValueError("Frozen mass-flux density must be finite and positive.")
            frozen = jnp.asarray(raw_density)
            density_label = None if density_id is None else str(density_id)
            if density_label is not None and not density_label:
                raise ValueError("density_id must be non-empty.")
            frozen_id = canonical_fingerprint(
                {
                    "kind": "frozen-mac-flow-control-density",
                    "identity": density_label,
                    "content": array_tree_fingerprint(raw_density),
                }
            )
        else:
            if frozen_density is not None or density_id is not None:
                raise ValueError(
                    "frozen_density and density_id belong only to mass-flux targets."
                )
            frozen = None
            frozen_id = "constant-density"

        self.schedule = schedule
        self.frozen_density = frozen
        self.kind = kind
        self.axes = axes_
        self.reference_density = density
        self.schedule_id = identifier
        self.density_id = frozen_id
        self.target_id = canonical_fingerprint(
            {
                "kind": "mac-flow-control-target",
                "quantity": kind,
                "axes": axes_,
                "schedule": identifier,
                "reference_density": density,
                "density": frozen_id,
            }
        )

    @classmethod
    def prescribed_pressure_gradient(
        cls,
        pressure_gradient: ArrayLike | Callable[[Array], ArrayLike],
        /,
        *,
        axes: Sequence[int] | None = None,
        density: float = 1.0,
        schedule_id: str | None = None,
    ) -> MACFlowControlTarget:
        if axes is None:
            if callable(pressure_gradient):
                raise ValueError("Callable pressure gradients require explicit axes.")
            values = np.asarray(pressure_gradient)
            if values.ndim == 0:
                axes_ = (0,)
            elif values.ndim == 1 and 0 < values.size <= 3:
                axes_ = tuple(range(values.size))
            else:
                raise ValueError("A pressure gradient must be a scalar or short vector.")
        else:
            axes_ = tuple(axes)
        return cls(
            "pressure_gradient",
            axes_,
            pressure_gradient,
            reference_density=density,
            schedule_id=schedule_id,
        )

    @classmethod
    def bulk_velocity(
        cls,
        velocity: ArrayLike | Callable[[Array], ArrayLike],
        /,
        *,
        axes: Sequence[int] = (0,),
        density: float = 1.0,
        schedule_id: str | None = None,
    ) -> MACFlowControlTarget:
        return cls(
            "bulk_velocity",
            axes,
            velocity,
            reference_density=density,
            schedule_id=schedule_id,
        )

    @classmethod
    def frozen_density_mass_flux(
        cls,
        mass_flux_and_density: tuple[ArrayLike | Callable[[Array], ArrayLike], ArrayLike],
        /,
        *,
        axes: Sequence[int] = (0,),
        schedule_id: str | None = None,
        density_id: str | None = None,
    ) -> MACFlowControlTarget:
        if (
            not isinstance(mass_flux_and_density, tuple)
            or len(mass_flux_and_density) != 2
        ):
            raise TypeError(
                "Mass-flux control requires the distinct (target, frozen_density) tuple."
            )
        target, density = mass_flux_and_density
        return cls(
            "frozen_density_mass_flux",
            axes,
            target,
            frozen_density=density,
            schedule_id=schedule_id,
            density_id=density_id,
        )

    def evaluate(self, time: ArrayLike, /) -> Array:
        value = jnp.asarray(self.schedule(jnp.asarray(time)))
        if value.ndim == 0 and len(self.axes) == 1:
            value = value.reshape((1,))
        if value.shape != (len(self.axes),):
            raise ValueError(
                "MAC flow-control schedule output must match the controlled axes."
            )
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "MAC flow-control target schedule returned a nonfinite value.",
        )


class MACFlowControlResourceEvidence(StrictModule, NonTrainableState):
    active_controls: int = eqx.field(static=True)
    stage_map_evaluations: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    controller_peak_bytes: int = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    within_budget: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class MACFlowControlConditioningEvidence(StrictModule):
    singular_values: Array
    minimum_singular_value: Array
    singular_value_floor: Array
    condition_number: Array
    numerical_rank: Array
    solve_status: Array
    full_rank: Array
    accepted: Array
    control_space_id: str = eqx.field(static=True)


class MACFlowControlDiagnostics(StrictModule):
    target: Array
    zero_control_response: Array
    response_matrix: Array
    control: Array
    predicted: Array
    achieved: Array
    observed_flux: Array
    target_residual_norm: Array
    response_residual_norm: Array
    boundary_residual_norm: Array
    projection_residual_norm: Array
    pressure_residual_norm: Array
    finite: Array
    successful: Array
    conditioning: MACFlowControlConditioningEvidence
    resources: MACFlowControlResourceEvidence
    quantity: MACFlowControlKind = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class MACFlowControlState(StrictModule):
    """Checkpoint-complete accepted controller and multistep continuation state."""

    time: Array
    step_index: Array
    state: Array
    previous_state: Array
    previous_explicit_rate: FaceVelocity
    explicit_rate: FaceVelocity
    pressure: Array
    previous_control: Array
    control: Array
    accepted_steps: Array
    sbdf2_valid: Array
    method_status: Array
    startup_pending: Array
    method_kind: MACFlowControlMethodKind = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class MACFlowControlStepResult(StrictModule):
    attempted_time: Array
    candidate_state: MACFlowControlState
    state: MACFlowControlState
    diagnostics: MACFlowControlDiagnostics
    accepted: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class _MACPressureGradientControl(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    base_forcing: Any
    target: MACFlowControlTarget
    face_density: FaceVelocity
    control: Array
    prescribed: bool = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)
    control_space_id: str = eqx.field(static=True)

    def __call__(self, time: Array, velocity: FaceVelocity, args: Any, /) -> FaceVelocity:
        values = self.operators.validate_velocity(velocity)
        base = (
            tuple(jnp.zeros_like(value) for value in values)
            if self.base_forcing is None
            else self.operators.validate_velocity(self.base_forcing(time, values, args))
        )
        active = self.target.evaluate(time) if self.prescribed else self.control
        gradient = jnp.zeros((len(values),), dtype=values[0].dtype)
        gradient = gradient.at[jnp.asarray(self.target.axes)].set(active)
        return tuple(
            source - gradient[axis] / density.astype(source.dtype)
            for axis, (source, density) in enumerate(
                zip(base, self.face_density, strict=True)
            )
        )


def _face_density(
    operators: PreparedMACOperators,
    target: MACFlowControlTarget,
    /,
) -> FaceVelocity:
    dimension = len(operators.discretization.cell_shape)
    if target.frozen_density is None:
        return tuple(
            jnp.full(
                layout.shape,
                target.reference_density,
                dtype=operators.pressure_space.dtype,
            )
            for layout in operators.discretization.face_layouts
        )
    density = jnp.asarray(target.frozen_density, dtype=operators.pressure_space.dtype)
    if density.shape != operators.discretization.cell_shape:
        raise ValueError(
            "Frozen mass-flux density must match the MAC cell-pressure shape."
        )
    output = []
    for axis in range(dimension):
        periodic = operators.discretization.grid.structured_axes[axis].periodic
        if periodic:
            face = 0.5 * (density + jnp.roll(density, 1, axis=axis))
        else:
            lower = jnp.take(density, jnp.asarray([0]), axis=axis)
            upper = jnp.take(density, jnp.asarray([-1]), axis=axis)
            left = jnp.take(density, jnp.arange(density.shape[axis] - 1), axis=axis)
            right = jnp.take(density, jnp.arange(1, density.shape[axis]), axis=axis)
            face = jnp.concatenate((lower, 0.5 * (left + right), upper), axis=axis)
        if face.shape != operators.discretization.face_layouts[axis].shape:
            raise RuntimeError("Prepared MAC flow-control face density has wrong shape.")
        output.append(face)
    return tuple(output)


def _method_dynamics(
    method: AbstractSSPRKFixedStepMethod | MACIMEXEulerMethod | MACSBDF2Method,
    /,
) -> tuple[CompiledMACIncompressibleDynamics, MACFlowControlMethodKind]:
    if isinstance(method, AbstractSSPRKFixedStepMethod):
        if not isinstance(method.vector_field, CompiledMACIncompressibleDynamics):
            raise TypeError("MAC SSPRK flow control requires compiled MAC dynamics.")
        return method.vector_field, "ssprk"
    if isinstance(method, MACIMEXEulerMethod):
        return method.dynamics, "imex_euler"
    if isinstance(method, MACSBDF2Method):
        return method.dynamics, "sbdf2"
    raise TypeError("Unsupported MAC flow-control temporal method.")


def _bind_dynamics(
    method: AbstractSSPRKFixedStepMethod | MACIMEXEulerMethod | MACSBDF2Method,
    dynamics: CompiledMACIncompressibleDynamics,
    /,
):
    if isinstance(method, AbstractSSPRKFixedStepMethod):
        return eqx.tree_at(lambda selected: selected.vector_field, method, dynamics)
    if isinstance(method, MACIMEXEulerMethod):
        return eqx.tree_at(lambda selected: selected.dynamics, method, dynamics)
    return eqx.tree_at(
        lambda selected: (selected.dynamics, selected.startup_method.dynamics),
        method,
        (dynamics, dynamics),
    )


def _set_control(method, control: Array, /):
    if isinstance(method, AbstractSSPRKFixedStepMethod):
        return eqx.tree_at(
            lambda selected: selected.vector_field.problem.forcing.control,
            method,
            control,
        )
    if isinstance(method, MACIMEXEulerMethod):
        return eqx.tree_at(
            lambda selected: selected.dynamics.problem.forcing.control,
            method,
            control,
        )
    return eqx.tree_at(
        lambda selected: (
            selected.dynamics.problem.forcing.control,
            selected.startup_method.dynamics.problem.forcing.control,
        ),
        method,
        (control, control),
    )


class MACFlowControlPlan(StrictModule, NonTrainableState):
    """Prepare a finite method-stage response controller for bounded MAC flow."""

    method: AbstractSSPRKFixedStepMethod | MACIMEXEulerMethod | MACSBDF2Method
    target: MACFlowControlTarget
    target_absolute_tolerance: float = eqx.field(static=True)
    target_relative_tolerance: float = eqx.field(static=True)
    response_tolerance: float = eqx.field(static=True)
    boundary_tolerance: float = eqx.field(static=True)
    projection_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractSSPRKFixedStepMethod | MACIMEXEulerMethod | MACSBDF2Method,
        target: MACFlowControlTarget,
        /,
        *,
        target_absolute_tolerance: float = 1.0e-7,
        target_relative_tolerance: float = 1.0e-7,
        response_tolerance: float = 1.0e-7,
        boundary_tolerance: float = 1.0e-7,
        projection_tolerance: float = 1.0e-7,
        condition_limit: float = 1.0e10,
        maximum_resource_bytes: int = 512 * 1024**2,
    ):
        dynamics, _ = _method_dynamics(method)
        if not isinstance(target, MACFlowControlTarget):
            raise TypeError("target must be a MACFlowControlTarget.")
        dimension = dynamics.problem.spatial_dimension
        if any(axis >= dimension for axis in target.axes):
            raise ValueError("Controlled axes exceed the MAC spatial dimension.")
        tolerances = (
            float(target_absolute_tolerance),
            float(target_relative_tolerance),
            float(response_tolerance),
            float(boundary_tolerance),
            float(projection_tolerance),
        )
        condition = float(condition_limit)
        budget = int(maximum_resource_bytes)
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Flow-control tolerances must be finite and nonnegative.")
        if not math.isfinite(condition) or condition < 1.0 or budget <= 0:
            raise ValueError("Flow-control conditioning and resource limits are invalid.")
        self.method = method
        self.target = target
        (
            self.target_absolute_tolerance,
            self.target_relative_tolerance,
            self.response_tolerance,
            self.boundary_tolerance,
            self.projection_tolerance,
        ) = tolerances
        self.condition_limit = condition
        self.maximum_resource_bytes = budget
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-flow-control-plan",
                "method": method.method_id,
                "dynamics": dynamics.compilation_id,
                "target": target.target_id,
                "target_absolute_tolerance": tolerances[0],
                "target_relative_tolerance": tolerances[1],
                "response_tolerance": tolerances[2],
                "boundary_tolerance": tolerances[3],
                "projection_tolerance": tolerances[4],
                "condition_limit": condition,
                "maximum_resource_bytes": budget,
                "controller": "zero-plus-unit-method-stage-response",
            }
        )

    def prepare(self, /) -> PreparedMACFlowControl:
        return PreparedMACFlowControl(self)


class PreparedMACFlowControl(StrictModule, NonTrainableState):
    """Executable, fail-closed MAC method-stage response controller."""

    plan: MACFlowControlPlan
    method: AbstractSSPRKFixedStepMethod | MACIMEXEulerMethod | MACSBDF2Method
    dynamics: CompiledMACIncompressibleDynamics
    resources: MACFlowControlResourceEvidence
    method_kind: MACFlowControlMethodKind = eqx.field(static=True)
    control_space_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MACFlowControlPlan, /):
        if not isinstance(plan, MACFlowControlPlan):
            raise TypeError("plan must be a MACFlowControlPlan.")
        dynamics, method_kind = _method_dynamics(plan.method)
        target = plan.target
        faces = _face_density(dynamics.momentum.operators, target)
        control_space_id = canonical_fingerprint(
            {
                "kind": "mac-pressure-gradient-vector-control-space",
                "operators": dynamics.momentum.operators.prepared_id,
                "axes": target.axes,
                "density": target.density_id,
                "reference_density": target.reference_density,
                "sign": "acceleration=-pressure-gradient/density",
            }
        )
        forcing = _MACPressureGradientControl(
            operators=dynamics.momentum.operators,
            base_forcing=dynamics.problem.forcing,
            target=target,
            face_density=faces,
            control=jnp.zeros(
                (len(target.axes),),
                dtype=dynamics.momentum.operators.pressure_space.dtype,
            ),
            prescribed=target.kind == "pressure_gradient",
            forcing_id=canonical_fingerprint(
                {
                    "kind": "mac-pressure-gradient-control-forcing",
                    "base": dynamics.problem.forcing_id,
                    "target": target.target_id,
                    "control_space": control_space_id,
                }
            ),
            control_space_id=control_space_id,
        )
        problem = IncompressibleFlowProblem(
            dynamics.problem.spatial_dimension,
            dynamics.problem.viscosity,
            forcing=forcing,
            forcing_id=forcing.forcing_id,
        )
        controlled = compile_mac_incompressible_flow(
            problem, dynamics.momentum, dynamics.projection
        )
        method = _bind_dynamics(plan.method, controlled)

        dtype = np.dtype(controlled.momentum.operators.pressure_space.dtype)
        velocity_size = controlled.momentum.operators.velocity_space.size
        pressure_size = controlled.momentum.operators.pressure_space.size
        state_scalars = 4 * velocity_size + pressure_size + 2 * len(target.axes) + 9
        evaluations = 1 if target.kind == "pressure_gradient" else len(target.axes) + 2
        state_bytes = int(state_scalars * dtype.itemsize)
        peak_bytes = int(
            evaluations * state_bytes
            + dtype.itemsize * (len(target.axes) ** 2 + 8 * len(target.axes) + 8)
        )
        within_budget = peak_bytes <= plan.maximum_resource_bytes
        evidence_id = canonical_fingerprint(
            {
                "kind": "mac-flow-control-resource-evidence",
                "plan": plan.plan_id,
                "active_controls": len(target.axes),
                "stage_map_evaluations": evaluations,
                "state_bytes": state_bytes,
                "controller_peak_bytes": peak_bytes,
                "maximum_resource_bytes": plan.maximum_resource_bytes,
                "within_budget": within_budget,
            }
        )
        if not within_budget:
            raise ValueError(
                "MAC flow-control response map exceeds maximum_resource_bytes: "
                f"requires {peak_bytes}, budget is {plan.maximum_resource_bytes}."
            )
        resources = MACFlowControlResourceEvidence(
            active_controls=len(target.axes),
            stage_map_evaluations=evaluations,
            state_bytes=state_bytes,
            controller_peak_bytes=peak_bytes,
            maximum_resource_bytes=plan.maximum_resource_bytes,
            within_budget=within_budget,
            evidence_id=evidence_id,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-flow-control",
                "plan": plan.plan_id,
                "controlled_dynamics": controlled.compilation_id,
                "control_space": control_space_id,
                "resources": evidence_id,
            }
        )
        self.plan = plan
        self.method = method
        self.dynamics = controlled
        self.resources = resources
        self.method_kind = method_kind
        self.control_space_id = control_space_id
        self.prepared_id = prepared_id

    def initialize(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
    ) -> MACFlowControlState:
        velocity_state = self.dynamics.validate_state(state)
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        time_ = jnp.asarray(time, dtype=dtype).reshape(())
        pressure_ = (
            jnp.zeros(
                self.dynamics.momentum.operators.discretization.cell_shape,
                dtype=dtype,
            )
            if pressure is None
            else self.dynamics.momentum.operators.gauge_project(pressure)
        )
        velocity = self.dynamics.unpack_velocity(velocity_state)
        explicit = tuple(jnp.zeros_like(value) for value in velocity)
        controls = jnp.zeros((len(self.plan.target.axes),), dtype=dtype)
        return MACFlowControlState(
            time=time_,
            step_index=jnp.asarray(0, dtype=jnp.int32),
            state=velocity_state,
            previous_state=velocity_state,
            previous_explicit_rate=explicit,
            explicit_rate=explicit,
            pressure=pressure_,
            previous_control=controls,
            control=controls,
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            sbdf2_valid=jnp.asarray(False),
            startup_pending=jnp.asarray(self.method_kind == "sbdf2"),
            method_status=jnp.asarray(0, dtype=jnp.int32),
            method_kind=self.method_kind,
            method_id=self.plan.method.method_id,
            target_id=self.plan.target.target_id,
            control_id=self.control_space_id,
            plan_id=self.prepared_id,
        )

    def _validate_state(self, state: MACFlowControlState, /) -> None:
        if not isinstance(state, MACFlowControlState):
            raise TypeError("state must be a MACFlowControlState.")
        if (
            state.plan_id != self.prepared_id
            or state.method_id != self.plan.method.method_id
            or state.target_id != self.plan.target.target_id
            or state.control_id != self.control_space_id
            or state.method_kind != self.method_kind
        ):
            raise ValueError(
                "MAC flow-control state belongs to a different prepared plan."
            )
        self.dynamics.validate_state(state.state)
        self.dynamics.validate_state(state.previous_state)
        operators = self.dynamics.momentum.operators
        operators.validate_pressure(state.pressure)
        operators.validate_velocity(state.previous_explicit_rate)
        operators.validate_velocity(state.explicit_rate)
        control_shape = (len(self.plan.target.axes),)
        if (
            state.control.shape != control_shape
            or state.previous_control.shape != control_shape
        ):
            raise ValueError("MAC flow-control continuation has the wrong control shape.")
        scalar_leaves = (
            state.time,
            state.step_index,
            state.accepted_steps,
            state.sbdf2_valid,
            state.method_status,
            state.startup_pending,
        )
        if any(value.shape != () for value in scalar_leaves):
            raise ValueError(
                "MAC flow-control scalar continuation fields must be scalar."
            )

    def _step_size(self, value: ArrayLike | None, /) -> Array:
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        if self.method_kind == "sbdf2":
            step = jnp.asarray(self.method.step_size, dtype=dtype)
            if value is not None:
                supplied = jnp.asarray(value, dtype=dtype).reshape(())
                step = eqx.error_if(
                    step, supplied != step, "Fixed MAC SBDF2 step cannot change."
                )
            return step
        if self.method_kind == "imex_euler":
            return self.method._step_size(value)
        if value is None:
            raise ValueError("MAC SSPRK flow control requires step_size.")
        step = jnp.asarray(value, dtype=dtype).reshape(())
        return eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "MAC SSPRK flow-control step must be positive and finite.",
        )

    def _from_sbdf2_history(
        self,
        previous: MACFlowControlState,
        history: MACSBDF2State,
        control: Array,
        /,
    ) -> MACFlowControlState:
        return MACFlowControlState(
            time=history.time,
            step_index=previous.step_index + jnp.asarray(1, dtype=jnp.int32),
            state=history.state,
            previous_state=history.previous_state,
            previous_explicit_rate=history.previous_explicit_rate,
            explicit_rate=history.explicit_rate,
            pressure=history.pressure,
            previous_control=previous.control,
            control=control,
            accepted_steps=history.accepted_steps,
            sbdf2_valid=history.valid,
            startup_pending=jnp.asarray(False),
            method_status=history.status,
            method_kind=self.method_kind,
            method_id=self.plan.method.method_id,
            target_id=self.plan.target.target_id,
            control_id=self.control_space_id,
            plan_id=self.prepared_id,
        )

    def _advance(
        self,
        state: MACFlowControlState,
        step: Array,
        control: Array,
        args: Any,
        /,
    ) -> tuple[MACFlowControlState, Array]:
        method = _set_control(self.method, control)
        if self.method_kind == "ssprk":
            result = method.step(state.step_index, state.time, state.state, step, args)
            candidate = MACFlowControlState(
                time=state.time + step,
                step_index=state.step_index + jnp.asarray(1, dtype=jnp.int32),
                state=result.accepted_state,
                previous_state=state.state,
                previous_explicit_rate=state.explicit_rate,
                explicit_rate=state.explicit_rate,
                pressure=state.pressure,
                previous_control=state.control,
                control=control,
                accepted_steps=state.accepted_steps + result.successful.astype(jnp.int32),
                sbdf2_valid=state.sbdf2_valid,
                startup_pending=state.startup_pending,
                method_status=state.method_status,
                method_kind=self.method_kind,
                method_id=self.plan.method.method_id,
                target_id=self.plan.target.target_id,
                control_id=self.control_space_id,
                plan_id=self.prepared_id,
            )
            return candidate, result.successful
        if self.method_kind == "imex_euler":
            result = method.step(
                state.time,
                state.state,
                step_size=step,
                pressure=state.pressure,
                args=args,
            )
            candidate = MACFlowControlState(
                time=result.attempted_time,
                step_index=state.step_index + jnp.asarray(1, dtype=jnp.int32),
                state=result.state,
                previous_state=state.state,
                previous_explicit_rate=state.explicit_rate,
                explicit_rate=result.explicit_rate,
                pressure=result.pressure,
                previous_control=state.control,
                control=control,
                accepted_steps=state.accepted_steps + result.accepted.astype(jnp.int32),
                sbdf2_valid=state.sbdf2_valid,
                startup_pending=state.startup_pending,
                method_status=result.status,
                method_kind=self.method_kind,
                method_id=self.plan.method.method_id,
                target_id=self.plan.target.target_id,
                control_id=self.control_space_id,
                plan_id=self.prepared_id,
            )
            return candidate, result.accepted

        def startup(_):
            result = method.initialize(
                state.time, state.state, pressure=state.pressure, args=args
            )
            return self._from_sbdf2_history(
                state, result.history, control
            ), result.accepted

        def multistep(_):
            _, convection, _, forcing = method.dynamics.rate_components(
                state.time, state.state, args
            )
            controlled_explicit = tuple(
                -advective + source
                for advective, source in zip(convection, forcing, strict=True)
            )
            history = MACSBDF2State(
                time=state.time,
                previous_state=state.previous_state,
                state=state.state,
                previous_explicit_rate=state.previous_explicit_rate,
                explicit_rate=controlled_explicit,
                pressure=state.pressure,
                accepted_steps=state.accepted_steps,
                valid=state.sbdf2_valid,
                status=state.method_status,
                method_id=method.method_id,
            )
            result = method.step(history, args=args)
            return self._from_sbdf2_history(
                state, result.history, control
            ), result.accepted

        return jax.lax.cond(state.startup_pending, startup, multistep, operand=None)

    def _observable(self, state: Array, /) -> Array:
        velocity = self.dynamics.unpack_velocity(state)
        operators = self.dynamics.momentum.operators
        axes = operators.discretization.grid.structured_axes
        cell_velocity = jnp.stack(
            tuple(
                _mac_face_to_cell(value, axis, axes[axis].periodic)
                for axis, value in enumerate(velocity)
            ),
            axis=-1,
        )
        volumes = operators.discretization.cell_volumes.astype(cell_velocity.dtype)
        if self.plan.target.frozen_density is None:
            weighted = volumes[..., None] * cell_velocity
        else:
            density = self.plan.target.frozen_density.astype(cell_velocity.dtype)
            weighted = volumes[..., None] * density[..., None] * cell_velocity
        average = jnp.sum(weighted, axis=tuple(range(volumes.ndim))) / jnp.sum(volumes)
        return average[jnp.asarray(self.plan.target.axes)]

    def _conditioning(
        self,
        response: Array,
        right_hand_side: Array,
        /,
    ) -> tuple[Array, MACFlowControlConditioningEvidence]:
        count = len(self.plan.target.axes)
        operator = DenseLinearOperator(
            response, operator_id=f"{self.prepared_id}:stage-response"
        )
        spectrum = solve_svd(
            SVDProblem(operator, problem_id=f"{self.prepared_id}:response-spectrum"),
            policy=SVDSolvePolicy(
                count=count,
                failure=FailurePolicy("status"),
            ),
        )
        linear = solve_linear(
            LinearSystem(operator, problem_id=f"{self.prepared_id}:response-system"),
            right_hand_side,
            policy=LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status")),
        )
        singular_values = spectrum.singular_values
        minimum = jnp.min(singular_values)
        safe_minimum = jnp.maximum(minimum, jnp.finfo(singular_values.dtype).tiny)
        condition = jnp.max(singular_values) / safe_minimum
        singular_floor = jnp.asarray(
            64.0 * jnp.finfo(singular_values.dtype).eps,
            dtype=singular_values.dtype,
        )
        full_rank = (
            spectrum.successful
            & (spectrum.numerical_rank == count)
            & (minimum > singular_floor)
        )
        accepted = (
            full_rank
            & linear.successful
            & jnp.isfinite(condition)
            & (condition <= self.plan.condition_limit)
            & jnp.all(jnp.isfinite(linear.value))
        )
        control = jnp.where(accepted, linear.value, jnp.zeros_like(right_hand_side))
        evidence = MACFlowControlConditioningEvidence(
            singular_values=singular_values,
            minimum_singular_value=minimum,
            singular_value_floor=singular_floor,
            condition_number=condition,
            numerical_rank=spectrum.numerical_rank,
            solve_status=linear.status,
            full_rank=full_rank,
            accepted=accepted,
            control_space_id=self.control_space_id,
        )
        return control, evidence

    def step(
        self,
        state: MACFlowControlState,
        /,
        *,
        step_size: ArrayLike | None = None,
        args: Any = None,
    ) -> MACFlowControlStepResult:
        self._validate_state(state)
        step = self._step_size(step_size)
        attempted_time = state.time + step
        target = self.plan.target.evaluate(attempted_time)
        count = len(self.plan.target.axes)
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        zeros = jnp.zeros((count,), dtype=dtype)

        if self.plan.target.kind == "pressure_gradient":
            control = target.astype(dtype)
            zero_response = jnp.zeros_like(control)
            response = jnp.zeros((count, count), dtype=dtype)
            candidate, method_success = self._advance(state, step, control, args)
            observed_flux = self._observable(candidate.state)
            achieved = control
            predicted = control
            singular_values = jnp.ones((count,), dtype=dtype)
            conditioning = MACFlowControlConditioningEvidence(
                singular_values=singular_values,
                minimum_singular_value=jnp.asarray(1.0, dtype=dtype),
                singular_value_floor=jnp.asarray(0.0, dtype=dtype),
                condition_number=jnp.asarray(1.0, dtype=dtype),
                numerical_rank=jnp.asarray(count, dtype=jnp.int32),
                solve_status=jnp.asarray(0, dtype=jnp.int32),
                full_rank=jnp.asarray(True),
                accepted=jnp.asarray(True),
                control_space_id=self.control_space_id,
            )
        else:
            zero_candidate, zero_success = self._advance(state, step, zeros, args)
            zero_response = self._observable(zero_candidate.state)
            columns = []
            influence_success = zero_success
            for column in range(count):
                unit = jnp.zeros((count,), dtype=dtype).at[column].set(1.0)
                unit_candidate, unit_success = self._advance(state, step, unit, args)
                columns.append(self._observable(unit_candidate.state) - zero_response)
                influence_success = influence_success & unit_success
            response = jnp.stack(tuple(columns), axis=1)
            control, conditioning = self._conditioning(
                response, target.astype(dtype) - zero_response
            )
            candidate, final_success = self._advance(state, step, control, args)
            observed_flux = self._observable(candidate.state)
            achieved = observed_flux
            predicted = zero_response + response @ control
            method_success = influence_success & final_success & conditioning.accepted

        controlled_method = _set_control(self.method, control)
        if isinstance(controlled_method, AbstractSSPRKFixedStepMethod):
            diagnostic_dynamics = controlled_method.vector_field
        else:
            diagnostic_dynamics = controlled_method.dynamics
        flow = diagnostic_dynamics.diagnostics(attempted_time, candidate.state, args)
        target_residual = jnp.sqrt(jnp.sum((achieved - target.astype(dtype)) ** 2))
        response_residual = jnp.sqrt(jnp.sum((achieved - predicted) ** 2))
        boundary_residual = jnp.max(jnp.abs(flow.boundary_defect))
        pressure_residual = jnp.maximum(
            jnp.abs(flow.pressure_residual_norm),
            jnp.abs(flow.pressure_gauge_residual),
        )
        projection_residual = jnp.maximum(
            jnp.abs(flow.divergence_norm), pressure_residual
        )
        target_scale = jnp.sqrt(jnp.sum(target.astype(dtype) ** 2))
        achieved_scale = jnp.sqrt(jnp.sum(achieved**2))
        target_ok = target_residual <= (
            self.plan.target_absolute_tolerance
            + self.plan.target_relative_tolerance * target_scale
        )
        response_ok = response_residual <= self.plan.response_tolerance * (
            1.0 + achieved_scale
        )
        finite = (
            jnp.all(jnp.isfinite(target))
            & jnp.all(jnp.isfinite(response))
            & jnp.all(jnp.isfinite(control))
            & jnp.all(jnp.isfinite(observed_flux))
            & jnp.isfinite(target_residual)
            & jnp.isfinite(response_residual)
            & jnp.isfinite(boundary_residual)
            & jnp.isfinite(projection_residual)
            & flow.finite
        )
        successful = (
            method_success
            & finite
            & target_ok
            & response_ok
            & (boundary_residual <= self.plan.boundary_tolerance)
            & (projection_residual <= self.plan.projection_tolerance)
            & flow.successful
        )
        committed = jax.tree.map(
            lambda accepted, previous: jnp.where(successful, accepted, previous),
            candidate,
            state,
        )
        diagnostics = MACFlowControlDiagnostics(
            target=target,
            zero_control_response=zero_response,
            response_matrix=response,
            control=control,
            predicted=predicted,
            achieved=achieved,
            observed_flux=observed_flux,
            target_residual_norm=target_residual,
            response_residual_norm=response_residual,
            boundary_residual_norm=boundary_residual,
            projection_residual_norm=projection_residual,
            pressure_residual_norm=pressure_residual,
            finite=finite,
            successful=successful,
            conditioning=conditioning,
            resources=self.resources,
            quantity=self.plan.target.kind,
            target_id=self.plan.target.target_id,
            control_id=self.control_space_id,
            plan_id=self.prepared_id,
        )
        return MACFlowControlStepResult(
            attempted_time=attempted_time,
            candidate_state=candidate,
            state=committed,
            diagnostics=diagnostics,
            accepted=successful,
            plan_id=self.prepared_id,
        )


__all__ = [
    "MACFlowControlConditioningEvidence",
    "MACFlowControlDiagnostics",
    "MACFlowControlKind",
    "MACFlowControlPlan",
    "MACFlowControlResourceEvidence",
    "MACFlowControlState",
    "MACFlowControlStepResult",
    "MACFlowControlTarget",
    "PreparedMACFlowControl",
]
