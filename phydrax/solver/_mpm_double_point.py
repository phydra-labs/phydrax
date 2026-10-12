#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Saturated incompressible-constituent MPM/FLIP with a mixed pressure unknown.

The material stress is effective skeleton stress. Accepted pressure is applied
once through the *same* mixed adjoint as its increment, never independently as
an MPM stress force. Dirichlet cells are pressure reservoirs (their continuity
rows are replaced); all remaining cells satisfy mixture continuity.
"""

from __future__ import annotations

from enum import IntEnum
from math import prod
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite
from ..applications.solid_mechanics._mpm_coupled_fields import (
    effective_pore_pressure_first_piola,
)
from ..applications.solid_mechanics._mpm_double_point import (
    cell_to_faces,
    MixedMPMPressureOperator,
    PreparedMPMToMACFaceTransfer,
)
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.flip._transfer import PreparedFLIPParticleTransfer
from ..discretization.flip._types import FLIPParticleState, FLIPRuntimeState
from ..discretization.mpm._dynamics import PreparedMPMDynamics
from ..discretization.mpm._types import MPMRuntimeState, MPMStepResult
from ..discretization.splatting._structured import (
    ParticleGridSplatPlan,
    ParticleGridSplatState,
    PreparedParticleGridSplat,
)
from ..equations._cfd_dem import AbstractInterphaseDragLaw, InterphaseDragEvaluation
from ..linalg import (
    ArraySpace,
    determinant_small_linear,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    MINRES,
    OperatorProperties,
    PCG,
    prepare,
    PreparedLinearSolve,
    refresh,
    SmallLinearSolvePlan,
    solve,
    TolerancePolicy,
)


class DoublePointMPMStatus(IntEnum):
    SUCCESS = 0
    INVALID_FRACTION = 1
    TRANSFER_FAILED = 2
    DRAG_FAILED = 3
    PROJECTION_FAILED = 4
    MATERIAL_FAILED = 5
    INVALID_STATE = 6
    PREDICTOR_FAILED = 7


@final
class DoublePointMPMState(StrictModule):
    solid: MPMRuntimeState
    fluid: FLIPRuntimeState


@final
class MixedMPMProjectionResult(StrictModule):
    pressure: Array
    pressure_increment: Array
    solid_velocity: Array
    fluid_velocity: FaceVelocity
    residual_norm: Array
    divergence_norm: Array
    iterations: Array
    linear_status: Array
    pressure_datum_supported: Array
    successful: Array


def _pack(solid: Array, fluid: FaceVelocity) -> Array:
    return jnp.concatenate((solid.reshape(-1), *(value.reshape(-1) for value in fluid)))


def _unpack(
    value: Array, solid_shape: tuple[int, ...], face_shapes: tuple[tuple[int, ...], ...]
) -> tuple[Array, FaceVelocity]:
    count = int(np.prod(solid_shape))
    solid = value[:count].reshape(solid_shape)
    components = []
    for shape in face_shapes:
        end = count + int(np.prod(shape))
        components.append(value[count:end].reshape(shape))
        count = end
    return solid, tuple(components)


@final
class _IdentityAction(StrictModule):
    def __call__(self, value: Array) -> Array:
        return value


@final
class _PressureAction(StrictModule):
    mixed: MixedMPMPressureOperator
    datum: Array

    def __call__(self, value: Array) -> Array:
        free = jnp.where(self.datum, 0.0, value)
        return jnp.where(self.datum, value, self.mixed.pressure_action(free))


@final
class _DragAction(StrictModule):
    mixed: MixedMPMPressureOperator
    coefficient: FaceVelocity
    step: Array
    solid_shape: tuple[int, ...] = eqx.field(static=True)
    face_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __call__(self, value: Array) -> Array:
        solid, fluid = _unpack(value, self.solid_shape, self.face_shapes)
        sampled = self.mixed.transfer.velocity(
            jnp.where(self.mixed.solid_active, solid, 0.0)
        )
        force = tuple(
            beta * (vs - jnp.where(active, vf, 0.0))
            for beta, vs, vf, active in zip(
                self.coefficient, sampled, fluid, self.mixed.fluid_active, strict=True
            )
        )
        solid_force = self.mixed.transfer.force_adjoint(force)
        solid_mass = self.mixed.solid_mass
        solid_out = jnp.where(
            self.mixed.solid_active,
            solid_mass[..., None] * solid + self.step * solid_force,
            solid,
        )
        fluid_out = tuple(
            jnp.where(active, mass * vf - self.step * fs, vf)
            for mass, vf, fs, active in zip(
                self.mixed.fluid_mass, fluid, force, self.mixed.fluid_active, strict=True
            )
        )
        return _pack(solid_out, fluid_out)


def _linear_operator(
    action: _IdentityAction | _PressureAction | _DragAction,
    space: ArraySpace,
    identifier: str,
    *,
    pressure: bool = False,
) -> FunctionLinearOperator:
    return FunctionLinearOperator(
        action,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=not pressure,
            evidence=(
                {"self_adjoint": "construction"}
                if pressure
                else {
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                }
            ),
        ),
        operator_id=identifier,
    )


@final
class _PressureConnectivity(StrictModule):
    """Fixed sparse divergence graph; numerical zero coefficients remove edges."""

    faces: Array
    coefficients: Array
    solid: Array
    groups: Array
    cells: Array
    dofs: Array
    cell_count: int = eqx.field(static=True)
    dof_count: int = eqx.field(static=True)

    def __init__(self, transfer: PreparedMPMToMACFaceTransfer) -> None:
        operators = transfer.operators
        shape = operators.discretization.cell_shape
        source = np.asarray(transfer.relation.source_indices)
        target = np.asarray(transfer.relation.target_indices)
        valid = np.asarray(transfer.relation.valid)
        weights = np.asarray(transfer.weights)
        solid_count = transfer.relation.source_size
        routes: dict[int, list[tuple[int, float]]] = {}
        for index in np.flatnonzero(valid):
            routes.setdefault(int(target[index]), []).append(
                (int(source[index]), float(weights[index]))
            )
        cells: list[int] = []
        dofs: list[int] = []
        faces: list[int] = []
        coefficients: list[float] = []
        solid: list[bool] = []
        volume = np.asarray(operators.discretization.cell_volumes)
        for cell in np.ndindex(shape):
            cell_id = int(np.ravel_multi_index(cell, shape))
            for axis, layout in enumerate(operators.discretization.face_layouts):
                measure = np.asarray(operators.discretization.face_measures[axis])
                periodic = operators.discretization.grid.structured_axes[axis].periodic
                for side, sign in ((0, -1.0), (1, 1.0)):
                    coordinate = list(cell)
                    coordinate[axis] += side
                    if periodic:
                        coordinate[axis] %= shape[axis]
                    face_tuple = tuple(coordinate)
                    face_id = transfer.face_offsets[axis] + int(
                        np.ravel_multi_index(face_tuple, layout.shape)
                    )
                    coefficient = sign * float(measure[face_tuple] / volume[cell])
                    cells.append(cell_id)
                    dofs.append(solid_count + face_id)
                    faces.append(face_id)
                    coefficients.append(coefficient)
                    solid.append(False)
                    for dof, weight in routes.get(face_id, []):
                        cells.append(cell_id)
                        dofs.append(dof)
                        faces.append(face_id)
                        coefficients.append(coefficient * weight)
                        solid.append(True)
        pairs = np.stack((cells, dofs), axis=-1).astype(np.int32)
        unique, groups = np.unique(pairs, axis=0, return_inverse=True)
        self.faces = jnp.asarray(faces, dtype=jnp.int32)
        self.coefficients = jnp.asarray(coefficients, dtype=transfer.weights.dtype)
        self.solid = jnp.asarray(solid, dtype=jnp.bool_)
        self.groups = jnp.asarray(groups, dtype=jnp.int32)
        self.cells = jnp.asarray(unique[:, 0], dtype=jnp.int32)
        self.dofs = jnp.asarray(unique[:, 1], dtype=jnp.int32)
        self.cell_count = prod(shape)
        self.dof_count = solid_count + transfer.face_offsets[-1]

    def supported(self, mixed: MixedMPMPressureOperator, datum: Array) -> Array:
        sf = jnp.concatenate(
            tuple(value.reshape((-1,)) for value in mixed.solid_fraction)
        )
        ff = jnp.concatenate(
            tuple(value.reshape((-1,)) for value in mixed.fluid_fraction)
        )
        fraction = jnp.where(self.solid, sf[self.faces], ff[self.faces])
        values = (
            jnp.zeros(self.cells.shape, dtype=self.coefficients.dtype)
            .at[self.groups]
            .add(self.coefficients * fraction)
        )
        active = jnp.concatenate(
            (
                mixed.solid_active.reshape((-1,)),
                *(value.reshape((-1,)) for value in mixed.fluid_active),
            )
        )
        edges = (values != 0.0) & active[self.dofs]
        labels = jnp.arange(self.cell_count, dtype=jnp.int32)

        def sweep(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            iteration, previous, _ = state
            dof_labels = jnp.full((self.dof_count,), self.cell_count, dtype=jnp.int32)
            dof_labels = dof_labels.at[self.dofs].min(
                jnp.where(edges, previous[self.cells], self.cell_count)
            )
            current = previous.at[self.cells].min(
                jnp.where(edges, dof_labels[self.dofs], self.cell_count)
            )
            return iteration + 1, current, jnp.any(current != previous)

        _, labels, _ = jax.lax.while_loop(
            lambda state: (state[0] < self.cell_count) & state[2],
            sweep,
            (jnp.asarray(0, dtype=jnp.int32), labels, jnp.asarray(True)),
        )
        anchored = (
            jnp.zeros((self.cell_count,), dtype=jnp.int32)
            .at[labels]
            .max(datum.reshape((-1,)).astype(jnp.int32))
        )
        return jnp.all(anchored[labels] != 0)


@final
class DoublePointMPMPlan(StrictModule, NonTrainableState):
    """Fixed Cartesian double-point attempt with an explicit pressure datum.

    ``pressure_mask`` selects physical pressure-reservoir cells; these can be a
    free surface at atmospheric pressure. No mean-zero gauge is substituted for
    physical pressure. ``solid_free`` and ``fluid_free`` impose stationary walls.
    """

    solid: PreparedMPMDynamics
    fluid_transfer: PreparedFLIPParticleTransfer
    solid_to_mac: PreparedMPMToMACFaceTransfer
    solid_cell: PreparedParticleGridSplat
    drag_law: AbstractInterphaseDragLaw
    solid_density: Array
    fluid_density: Array
    fluid_viscosity: Array
    grain_diameter: Array
    pressure_mask: Array
    pressure_values: Array
    solid_free: Array
    fluid_free: tuple[Array, ...]
    pressure_connectivity: _PressureConnectivity
    prepared_pressure: PreparedLinearSolve
    prepared_drag: PreparedLinearSolve
    pressure_space: ArraySpace
    drag_space: ArraySpace
    determinant_plan: SmallLinearSolvePlan
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    pressure_id: str = eqx.field(static=True)
    drag_id: str = eqx.field(static=True)

    def __init__(
        self,
        solid: PreparedMPMDynamics,
        fluid_transfer: PreparedFLIPParticleTransfer,
        solid_to_mac: PreparedMPMToMACFaceTransfer,
        drag_law: AbstractInterphaseDragLaw,
        /,
        *,
        solid_density: ArrayLike,
        fluid_density: ArrayLike,
        fluid_viscosity: ArrayLike,
        grain_diameter: ArrayLike,
        pressure_mask: ArrayLike,
        pressure_values: ArrayLike = 0.0,
        biot_coefficient: ArrayLike = 1.0,
        solid_free: ArrayLike | None = None,
        fluid_free: tuple[ArrayLike, ...] | None = None,
        tolerance: float = 1e-8,
        maximum_iterations: int = 256,
    ) -> None:
        operators = fluid_transfer.plan.operators
        solid._require_split_layout()
        if solid.contact is not None:
            raise ValueError(
                "Double-point mixed projection does not support MPM contact constraints."
            )
        if not bool(jnp.asarray(biot_coefficient) == 1.0):
            raise ValueError(
                "Incompressible constituents require Biot coefficient equal to one."
            )
        if solid_to_mac.operators.prepared_id != operators.prepared_id:
            raise ValueError("Solid and fluid transfers must bind the same MAC grid.")
        if solid_to_mac.solid.prepared_id != solid.prepared_id:
            raise ValueError("Solid transfer must bind the supplied MPM dynamics.")
        mask = np.asarray(pressure_mask, dtype=np.bool_)
        if mask.shape != operators.discretization.cell_shape or not np.any(mask):
            raise ValueError(
                "Double-point MPM requires a free-surface or Dirichlet pressure datum."
            )
        values = np.broadcast_to(
            np.asarray(pressure_values, dtype=np.float64), mask.shape
        )
        properties = tuple(
            jnp.asarray(value, dtype=operators.pressure_space.dtype).reshape(())
            for value in (solid_density, fluid_density, fluid_viscosity, grain_diameter)
        )
        if any(
            not bool(jnp.isfinite(value) & (value > 0.0)) for value in properties
        ) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Intrinsic densities, viscosity and grain diameter must be positive and finite."
            )
        if tolerance <= 0.0 or maximum_iterations <= 0:
            raise ValueError("Solver tolerance and iteration capacity must be positive.")
        solid_shape = solid.nodal_shape + (solid.dimension,)
        sf = (
            np.ones(solid_shape, dtype=np.bool_)
            if solid_free is None
            else np.asarray(solid_free, dtype=np.bool_)
        )
        if sf.shape != solid_shape:
            raise ValueError("solid_free must have the solid nodal velocity shape.")
        if solid.boundary is not None:
            boundary_mask = np.asarray(solid.boundary.mask)
            if np.any(np.asarray(solid.boundary.values)[boundary_mask] != 0.0):
                raise ValueError(
                    "Mixed projection currently requires stationary prescribed solid boundaries."
                )
            sf = sf & ~boundary_mask
        layouts = operators.discretization.face_layouts
        ff = (
            tuple(np.ones(layout.shape, dtype=np.bool_) for layout in layouts)
            if fluid_free is None
            else tuple(np.asarray(value, dtype=np.bool_) for value in fluid_free)
        )
        if len(ff) != len(layouts) or any(
            value.shape != layout.shape for value, layout in zip(ff, layouts, strict=True)
        ):
            raise ValueError("fluid_free must have the MAC face shapes.")
        identifier = canonical_fingerprint(
            {
                "kind": "double-point-mpm",
                "solid": solid.prepared_id,
                "fluid": fluid_transfer.prepared_id,
                "map": solid_to_mac.prepared_id,
                "drag": drag_law.law_id,
                "pressure_mask": mask,
                "solid_free": sf,
                "fluid_free": ff,
                "tolerance": tolerance,
                "iterations": maximum_iterations,
            }
        )
        pressure_id = identifier + ":pressure"
        drag_id = identifier + ":drag"
        pressure_space = ArraySpace(mask.shape, dtype=operators.pressure_space.dtype)
        size = int(np.prod(solid_shape)) + sum(
            int(np.prod(layout.shape)) for layout in layouts
        )
        drag_space = ArraySpace((size,), dtype=operators.pressure_space.dtype)
        policy = LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=tolerance, absolute=0.0, max_steps=maximum_iterations
            ),
        )
        self.solid = solid
        self.fluid_transfer = fluid_transfer
        self.solid_to_mac = solid_to_mac
        self.solid_cell = ParticleGridSplatPlan(
            operators.discretization.grid,
            location=operators.discretization.grid.location(
                operators.discretization.cell_layout.offsets
            ),
            assignment=solid.splat.plan.assignment,
            boundary="reject",
            execution=solid.splat.plan.execution,
            precision=solid.splat.plan.precision,
            budget=solid.splat.plan.budget,
        ).prepare(solid.particles)
        self.drag_law = drag_law
        (
            self.solid_density,
            self.fluid_density,
            self.fluid_viscosity,
            self.grain_diameter,
        ) = properties
        self.pressure_mask = jnp.asarray(mask)
        self.pressure_values = jnp.asarray(values, dtype=operators.pressure_space.dtype)
        self.pressure_connectivity = _PressureConnectivity(solid_to_mac)
        self.solid_free = jnp.asarray(sf)
        self.fluid_free = tuple(jnp.asarray(value) for value in ff)
        self.pressure_space = pressure_space
        self.drag_space = drag_space
        self.determinant_plan = SmallLinearSolvePlan(solid.constitutive_dimension)
        self.prepared_pressure = prepare(
            LinearSystem(
                _linear_operator(
                    _IdentityAction(), pressure_space, pressure_id, pressure=True
                ),
                problem_id=pressure_id,
            ),
            LinearSolvePolicy(
                MINRES(),
                tolerance=TolerancePolicy(
                    relative=tolerance, absolute=0.0, max_steps=maximum_iterations
                ),
            ),
        )
        self.prepared_drag = prepare(
            LinearSystem(
                _linear_operator(_IdentityAction(), drag_space, drag_id),
                problem_id=drag_id,
            ),
            policy,
        )
        self.tolerance = tolerance
        self.plan_id = identifier
        self.pressure_id = pressure_id
        self.drag_id = drag_id

    def initialize(
        self, solid: MPMRuntimeState, fluid: FLIPRuntimeState, /
    ) -> DoublePointMPMState:
        self.fluid_transfer.plan.operators.validate_pressure(fluid.pressure)
        if not bool(
            jnp.all(
                jnp.where(
                    self.pressure_mask, fluid.pressure == self.pressure_values, True
                )
            )
        ):
            raise ValueError("Initial pressure must satisfy the physical pressure datum.")
        return DoublePointMPMState(solid, fluid)


@final
class DoublePointMPMStepResult(StrictModule):
    candidate_state: DoublePointMPMState
    accepted_state: DoublePointMPMState
    porosity: Array
    mixture_velocity: FaceVelocity
    mixture_divergence: Array
    expanded_divergence: Array
    divergence_identity_defect: Array
    drag: tuple[InterphaseDragEvaluation, ...]
    projection: MixedMPMProjectionResult
    solid_step: MPMStepResult
    drag_momentum_defect: Array
    solid_momentum_defect: Array
    fluid_momentum_defect: Array
    drag_residual_norm: Array
    effective_stress: Array
    total_stress: Array
    drag_phase_present: tuple[Array, ...]
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _phase_momentum(
    mixed: MixedMPMPressureOperator, solid: Array, fluid: FaceVelocity
) -> Array:
    solid_p = jnp.sum(
        (mixed.solid_mass[..., None] * solid).reshape((-1, solid.shape[-1])), axis=0
    )
    fluid_p = jnp.stack(
        tuple(
            jnp.sum(mass * velocity)
            for mass, velocity in zip(mixed.fluid_mass, fluid, strict=True)
        )
    )
    return solid_p + fluid_p


def _face_pair(value: Array, axis: int, periodic: bool) -> tuple[Array, Array]:
    if periodic:
        return value, jnp.roll(value, -1, axis=axis)
    lower = [slice(None)] * value.ndim
    upper = [slice(None)] * value.ndim
    lower[axis] = slice(None, -1)
    upper[axis] = slice(1, None)
    return value[tuple(lower)], value[tuple(upper)]


def _slip_speed(
    mixed: MixedMPMPressureOperator, solid: Array, fluid: FaceVelocity
) -> FaceVelocity:
    operators = mixed.transfer.operators
    sampled = mixed.transfer.velocity(jnp.where(mixed.solid_active, solid, 0.0))
    slip = tuple(
        jnp.where(mask, vf, 0.0) - vs
        for mask, vf, vs in zip(mixed.fluid_active, fluid, sampled, strict=True)
    )
    centered_square = []
    for axis, value in enumerate(slip):
        lower, upper = _face_pair(
            value, axis, operators.discretization.grid.structured_axes[axis].periodic
        )
        centered_square.append((0.5 * (lower + upper)) ** 2)
    face_square = tuple(cell_to_faces(operators, value) for value in centered_square)
    return tuple(
        jnp.sqrt(
            value**2
            + sum(face_square[other][axis] for other in range(len(slip)) if other != axis)
        )
        for axis, value in enumerate(slip)
    )


def _expanded_divergence(
    mixed: MixedMPMPressureOperator, solid: Array, fluid: FaceVelocity
) -> Array:
    """Exact centered discrete product rule including grad(n) dot phase slip."""
    operators = mixed.transfer.operators
    sampled = mixed.transfer.velocity(jnp.where(mixed.solid_active, solid, 0.0))
    expanded = jnp.zeros_like(operators.discretization.cell_volumes)
    for axis, (n, vs, vf, area, active) in enumerate(
        zip(
            mixed.fluid_fraction,
            sampled,
            fluid,
            operators.discretization.face_measures,
            mixed.fluid_active,
            strict=True,
        )
    ):
        periodic = operators.discretization.grid.structured_axes[axis].periodic
        nl, nu = _face_pair(n, axis, periodic)
        sl, su = _face_pair(area * vs, axis, periodic)
        fl, fu = _face_pair(area * jnp.where(active, vf, 0.0), axis, periodic)
        average = 0.5 * (nl + nu)
        expanded = (
            expanded
            + (
                (1.0 - average) * (su - sl)
                + average * (fu - fl)
                + (nu - nl) * 0.5 * ((fu + fl) - (su + sl))
            )
            / operators.discretization.cell_volumes
        )
    return expanded


def _advect_fluid(
    plan: DoublePointMPMPlan, position: Array, velocity: Array, step: Array
) -> tuple[Array, Array]:
    proposed = position + step * velocity
    valid = jnp.asarray(True)
    for axis, grid_axis in enumerate(
        plan.fluid_transfer.plan.operators.discretization.grid.structured_axes
    ):
        lower, upper = grid_axis.bounds
        if grid_axis.periodic:
            proposed = proposed.at[:, axis].set(
                lower + jnp.mod(proposed[:, axis] - lower, upper - lower)
            )
        else:
            valid = valid & jnp.all(
                (proposed[:, axis] >= lower) & (proposed[:, axis] <= upper)
            )
    return proposed, valid


def _drag_stage(
    plan: DoublePointMPMPlan,
    mixed: MixedMPMPressureOperator,
    solid: Array,
    fluid: FaceVelocity,
    step: Array,
) -> tuple[
    Array, FaceVelocity, tuple[InterphaseDragEvaluation, ...], Array, Array, Array
]:
    speed = _slip_speed(mixed, solid, fluid)
    evidence = []
    coefficients = []
    for n, slip, measure, mass in zip(
        mixed.fluid_fraction,
        speed,
        mixed.transfer.operators.face_dual_measures,
        mixed.fluid_mass,
        strict=True,
    ):
        present = (n > 0.0) & (n < 1.0) & (mass > 0.0)
        evaluation = plan.drag_law.evaluate(
            jnp.where(present, n, 0.5),
            jnp.where(present, slip, 0.0),
            plan.grain_diameter,
            plan.fluid_viscosity,
            plan.fluid_density,
            jnp.ones_like(n),
        )
        evaluation = eqx.tree_at(
            lambda value: (value.beta, value.particle_drag_coefficient),
            evaluation,
            (
                jnp.where(present, evaluation.beta, 0.0),
                jnp.where(present, evaluation.particle_drag_coefficient, 0.0),
            ),
        )
        evidence.append(evaluation)
        coefficients.append(jnp.where(present, evaluation.beta * measure, 0.0))
    action = _DragAction(
        mixed,
        tuple(coefficients),
        step,
        solid.shape,
        tuple(value.shape for value in fluid),
    )
    rhs = _pack(
        jnp.where(mixed.solid_active, mixed.solid_mass[..., None] * solid, solid),
        tuple(
            jnp.where(active, mass * value, value)
            for mass, value, active in zip(
                mixed.fluid_mass, fluid, mixed.fluid_active, strict=True
            )
        ),
    )
    prepared = refresh(
        plan.prepared_drag,
        LinearSystem(
            _linear_operator(action, plan.drag_space, plan.drag_id),
            problem_id=plan.drag_id,
        ),
    )
    linear = solve(prepared, rhs, initial_guess=_pack(solid, fluid))
    vs, vf = _unpack(linear.value, solid.shape, tuple(value.shape for value in fluid))
    force = tuple(
        coef * (ss - jnp.where(mask, ff, 0.0))
        for coef, ss, ff, mask in zip(
            coefficients,
            mixed.transfer.velocity(jnp.where(mixed.solid_active, vs, 0.0)),
            vf,
            mixed.fluid_active,
            strict=True,
        )
    )
    solid_force = mixed.transfer.force_adjoint(force)
    solid_wall_impulse = -step * jnp.sum(
        jnp.where(mixed.solid_active, 0.0, solid_force).reshape((-1, solid.shape[-1])),
        axis=0,
    )
    fluid_wall_impulse = step * jnp.stack(
        tuple(
            jnp.sum(jnp.where(mask, 0.0, value))
            for mask, value in zip(mixed.fluid_active, force, strict=True)
        )
    )
    before = _phase_momentum(mixed, solid, fluid)
    defect = (
        _phase_momentum(mixed, vs, vf) - before + solid_wall_impulse + fluid_wall_impulse
    )
    residual = jnp.asarray(linear.diagnostics.residual_norm)
    ledger_tolerance = jnp.sqrt(
        jnp.asarray(rhs.size, dtype=rhs.dtype)
    ) * residual + 256.0 * jnp.finfo(rhs.dtype).eps * jnp.sum(jnp.abs(rhs))
    successful = (
        linear.successful
        & jnp.all(jnp.stack(tuple(jnp.all(value.successful) for value in evidence)))
        & (jnp.max(jnp.abs(defect)) <= ledger_tolerance)
    )
    return vs, vf, tuple(evidence), defect, residual, successful


def project_double_point_velocity(
    plan: DoublePointMPMPlan,
    mixed: MixedMPMPressureOperator,
    solid: Array,
    fluid: FaceVelocity,
    pressure: Array,
    step: Array,
    /,
) -> MixedMPMProjectionResult:
    """One mixed Schur solve; no sampled-mixture MAC projection is performed."""
    action = _PressureAction(mixed, plan.pressure_mask)
    volume = mixed.transfer.operators.discretization.cell_volumes
    prescribed = jnp.where(
        plan.pressure_mask, -volume * (plan.pressure_values - pressure), 0.0
    )
    rhs = mixed.divergence(solid, fluid) / step - mixed.pressure_action(prescribed)
    rhs = jnp.where(plan.pressure_mask, prescribed, rhs)
    prepared = refresh(
        plan.prepared_pressure,
        LinearSystem(
            _linear_operator(
                action, plan.pressure_space, plan.pressure_id, pressure=True
            ),
            problem_id=plan.pressure_id,
        ),
    )
    linear = solve(prepared, rhs)
    increment = jnp.where(plan.pressure_mask, prescribed, linear.value)
    solid_out = solid - step * mixed.inverse_mass_s(mixed.solid_transpose(increment))
    fluid_out = tuple(
        value - step * delta
        for value, delta in zip(
            fluid, mixed.inverse_mass_f(mixed.fluid_transpose(increment)), strict=True
        )
    )
    divergence = mixed.divergence(solid_out, fluid_out)
    norm = jnp.max(jnp.abs(jnp.where(plan.pressure_mask, 0.0, divergence)))
    # A nonzero reservoir can drive flow from rest; its affine boundary action
    # sets the scale even when the unforced velocity has exactly zero divergence.
    scale = jnp.max(jnp.abs(mixed.divergence(solid, fluid))) + step * jnp.max(
        jnp.abs(mixed.pressure_action(prescribed))
    )
    datum_supported = plan.pressure_connectivity.supported(mixed, plan.pressure_mask)
    successful = (
        datum_supported
        & linear.successful
        & (
            norm
            <= plan.tolerance * scale + 256.0 * jnp.finfo(divergence.dtype).eps * scale
        )
    )
    pressure_increment = -increment / volume
    return MixedMPMProjectionResult(
        pressure + pressure_increment,
        pressure_increment,
        solid_out,
        fluid_out,
        jnp.asarray(linear.diagnostics.residual_norm),
        norm,
        jnp.asarray(linear.diagnostics.iterations, dtype=jnp.int32),
        jnp.asarray(linear.status, dtype=jnp.int32),
        datum_supported,
        successful,
    )


def _stress_evidence(
    plan: DoublePointMPMPlan,
    solid: MPMRuntimeState,
    routes: ParticleGridSplatState,
    pressure: Array,
) -> tuple[Array, Array, Array]:
    material_particle = solid.particles
    deformation = material_particle.deformation_gradient
    jacobian = determinant_small_linear(plan.determinant_plan, deformation)
    particle_pressure = plan.solid_cell.gather(routes, pressure)
    pore_piola = effective_pore_pressure_first_piola(
        deformation, particle_pressure.values, 1.0
    )
    effective = (
        material_particle.first_piola @ jnp.swapaxes(deformation, -1, -2)
    ) / jacobian[..., None, None]
    total = (
        effective
        + (pore_piola @ jnp.swapaxes(deformation, -1, -2)) / jacobian[..., None, None]
    )
    return effective, total, jnp.all(particle_pressure.support)


def _transaction_status(
    fraction_ok: Array,
    predictor_ok: Array,
    transfer_ok: Array,
    drag_ok: Array,
    projection_ok: Array,
    material_ok: Array,
    finite: Array,
) -> Array:
    return jnp.select(
        (
            ~fraction_ok,
            ~predictor_ok,
            ~transfer_ok,
            ~drag_ok,
            ~projection_ok,
            ~material_ok,
            ~finite,
        ),
        (
            int(DoublePointMPMStatus.INVALID_FRACTION),
            int(DoublePointMPMStatus.PREDICTOR_FAILED),
            int(DoublePointMPMStatus.TRANSFER_FAILED),
            int(DoublePointMPMStatus.DRAG_FAILED),
            int(DoublePointMPMStatus.PROJECTION_FAILED),
            int(DoublePointMPMStatus.MATERIAL_FAILED),
            int(DoublePointMPMStatus.INVALID_STATE),
        ),
        default=int(DoublePointMPMStatus.SUCCESS),
    ).astype(jnp.int32)


def advance_double_point_mpm_step(
    plan: DoublePointMPMPlan,
    state: DoublePointMPMState,
    step_size: ArrayLike,
    /,
    *,
    arguments: Any = None,
) -> DoublePointMPMStepResult:
    """Return a candidate and atomically retain both committed phases on failure."""
    dtype = state.solid.particles.position.dtype
    step = jnp.asarray(step_size, dtype=dtype).reshape(())
    predictor = plan.solid.predict_grid(state.solid, step, arguments)
    operators = plan.fluid_transfer.plan.operators
    active = (
        plan.solid.particles.active_mask
        if state.solid.lifecycle_state is None
        else state.solid.lifecycle_state.active
    )
    solid_routes = plan.solid_cell.build(
        state.solid.particles.position,
        active_mask=active,
        assignment_input=state.solid.assignment_input,
    )
    solid_masses = (
        plan.solid.particles.safe_masses
        if state.solid.lifecycle_state is None
        else state.solid.lifecycle_state.masses
    )
    deposit = plan.solid_cell.deposit_content(
        solid_routes, solid_masses / plan.solid_density
    )
    fraction = deposit.content / operators.discretization.cell_volumes
    porosity = 1.0 - fraction
    fraction_ok = jnp.all(jnp.isfinite(fraction) & (fraction >= 0.0) & (fraction <= 1.0))
    n_faces = cell_to_faces(operators, porosity)
    s_faces = tuple(1.0 - value for value in n_faces)
    routes = plan.fluid_transfer.build(state.fluid.particles.position)
    fluid_grid = plan.fluid_transfer.particle_to_grid(
        routes, state.fluid.particles.velocity, plan.fluid_density
    )
    fluid_mass = tuple(
        jnp.where(n > 0.0, mass, 0.0)
        for n, mass in zip(n_faces, fluid_grid.face_mass, strict=True)
    )
    mixed = MixedMPMPressureOperator(
        plan.solid_to_mac,
        s_faces,
        n_faces,
        predictor.grid.mass[0],
        fluid_mass,
        solid_active=plan.solid_free,
        fluid_active=plan.fluid_free,
    )
    solid_velocity = jnp.where(plan.solid_free, predictor.grid.velocity[0], 0.0)
    fluid_velocity = tuple(
        jnp.where(free, velocity, 0.0)
        for free, velocity in zip(plan.fluid_free, fluid_grid.velocity, strict=True)
    )
    # Accepted pressure is applied through the exact same D^T as its increment.
    pressure_content = operators.discretization.cell_volumes * state.fluid.pressure
    solid_velocity = solid_velocity + step * mixed.inverse_mass_s(
        mixed.solid_transpose(pressure_content)
    )
    fluid_velocity = tuple(
        value + step * correction
        for value, correction in zip(
            fluid_velocity,
            mixed.inverse_mass_f(mixed.fluid_transpose(pressure_content)),
            strict=True,
        )
    )
    vs, vf, drag, drag_defect, drag_residual, drag_ok = _drag_stage(
        plan, mixed, solid_velocity, fluid_velocity, step
    )
    projection = project_double_point_velocity(
        plan, mixed, vs, vf, state.fluid.pressure, step
    )
    solid_step = plan.solid.finalize_grid_velocity(
        predictor, projection.solid_velocity[None]
    )
    gather = plan.fluid_transfer.grid_to_particle(
        routes, fluid_grid.velocity, projection.fluid_velocity
    )
    velocity = state.fluid.particles.velocity + gather.flip_increment
    next_position, domain_ok = _advect_fluid(
        plan, state.fluid.particles.position, gather.pic_velocity, step
    )
    particles = FLIPParticleState(next_position, velocity)
    fluid_candidate = FLIPRuntimeState(
        particles,
        projection.pressure,
        state.fluid.time + step,
        state.fluid.accepted_step + 1,
        jnp.asarray(0, dtype=jnp.int32),
        state.fluid.geometry_epoch,
        state.fluid.geometry_id,
    )
    candidate = DoublePointMPMState(solid_step.candidate_state, fluid_candidate)
    effective_stress, total_stress, pressure_support = _stress_evidence(
        plan, candidate.solid, solid_routes, projection.pressure
    )
    sampled = mixed.transfer.velocity(projection.solid_velocity)
    mixture = tuple(
        s * vs_ + n * vf_
        for s, n, vs_, vf_ in zip(
            s_faces, n_faces, sampled, projection.fluid_velocity, strict=True
        )
    )
    divergence = operators.divergence(mixture)
    expanded = _expanded_divergence(
        mixed, projection.solid_velocity, projection.fluid_velocity
    )
    identity_defect = divergence - expanded
    finite = (
        tree_allfinite(candidate)
        & jnp.all(jnp.isfinite(effective_stress))
        & jnp.all(jnp.isfinite(total_stress))
        & jnp.isfinite(step)
        & (step > 0.0)
    )
    transfer_ok = (
        deposit.successful
        & fluid_grid.successful
        & gather.successful
        & mixed.evidence.successful
        & pressure_support
        & domain_ok
    )
    solid_grid_momentum = jnp.sum(
        (mixed.solid_mass[..., None] * projection.solid_velocity).reshape(
            (-1, plan.solid.dimension)
        ),
        axis=0,
    )
    solid_particle_momentum = jnp.sum(
        solid_masses[:, None] * candidate.solid.particles.velocity, axis=0
    )
    fluid_particle_impulse = jnp.sum(
        plan.fluid_transfer.particles.safe_masses[:, None] * gather.flip_increment, axis=0
    )
    fluid_grid_impulse = jnp.stack(
        tuple(
            jnp.sum(mass * (new - old))
            for mass, new, old in zip(
                fluid_mass, projection.fluid_velocity, fluid_grid.velocity, strict=True
            )
        )
    )
    solid_scale = jnp.sum(
        jnp.abs(mixed.solid_mass[..., None] * projection.solid_velocity)
    ) + jnp.sum(jnp.abs(solid_masses[:, None] * candidate.solid.particles.velocity))
    fluid_scale = jnp.sum(
        jnp.abs(
            plan.fluid_transfer.particles.safe_masses[:, None] * gather.flip_increment
        )
    ) + sum(
        jnp.sum(jnp.abs(mass * (new - old)))
        for mass, new, old in zip(
            fluid_mass, projection.fluid_velocity, fluid_grid.velocity, strict=True
        )
    )
    roundoff = 4096.0 * jnp.finfo(dtype).eps
    ledger_ok = (
        jnp.max(jnp.abs(solid_particle_momentum - solid_grid_momentum))
        <= roundoff * solid_scale
    ) & (
        jnp.max(jnp.abs(fluid_particle_impulse - fluid_grid_impulse))
        <= roundoff * fluid_scale
    )
    identity_ok = jnp.max(jnp.abs(identity_defect)) <= roundoff * (
        jnp.max(jnp.abs(divergence)) + jnp.max(jnp.abs(expanded))
    )
    transfer_ok = transfer_ok & ledger_ok & identity_ok
    successful = (
        fraction_ok
        & predictor.successful
        & transfer_ok
        & drag_ok
        & projection.successful
        & solid_step.successful
        & finite
    )
    accepted = jax.lax.cond(successful, lambda: candidate, lambda: state)
    status = _transaction_status(
        fraction_ok,
        predictor.successful,
        transfer_ok,
        drag_ok,
        projection.successful,
        solid_step.successful,
        finite,
    )
    return DoublePointMPMStepResult(
        candidate,
        accepted,
        porosity,
        mixture,
        divergence,
        expanded,
        identity_defect,
        drag,
        projection,
        solid_step,
        drag_defect,
        solid_particle_momentum - solid_grid_momentum,
        fluid_particle_impulse - fluid_grid_impulse,
        drag_residual,
        effective_stress,
        total_stress,
        tuple((n > 0.0) & (n < 1.0) for n in n_faces),
        jnp.asarray(status, dtype=jnp.int32),
        successful,
        plan.plan_id,
    )


__all__ = [
    "DoublePointMPMPlan",
    "DoublePointMPMState",
    "DoublePointMPMStatus",
    "DoublePointMPMStepResult",
    "MixedMPMProjectionResult",
    "advance_double_point_mpm_step",
    "project_double_point_velocity",
]
