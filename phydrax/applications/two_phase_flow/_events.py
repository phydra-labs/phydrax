#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._step import TwoPhaseMovingBodyPlan
from ._vof import PreparedIncompressibleTwoPhaseVOF, TwoPhaseVOFState


class TwoPhaseCapabilityEvidence(StrictModule):
    wetting_cells: Array
    drying_cells: Array
    moving_contact_cells: Array
    boundary_contact_cells: Array
    surface_piercing_cells: Array
    body_contact_cells: Array
    overturning_cells: Array
    topology_change_count: Array
    contact_angle_residual: Array
    wetting_event: Array
    drying_event: Array
    moving_contact_event: Array
    surface_piercing_event: Array
    body_contact_event: Array
    breaking_or_overturning_event: Array
    derivative_available: Array
    finite: Array
    successful: Array
    event_code: Array
    event_id: str = eqx.field(static=True)


class TwoPhaseCapabilityEventPlan(StrictModule, NonTrainableState):
    phase_threshold: float = eqx.field(static=True)
    contact_angle_tolerance: float = eqx.field(static=True)
    minimum_overturning_normal: float = eqx.field(static=True)
    maximum_topology_changes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        phase_threshold: float = 0.5,
        contact_angle_tolerance: float = 5.0e-3,
        minimum_overturning_normal: float = 0.1,
        maximum_topology_changes: int = 0,
    ):
        threshold = float(phase_threshold)
        angle = float(contact_angle_tolerance)
        normal = float(minimum_overturning_normal)
        changes = int(maximum_topology_changes)
        if (
            not 0.0 < threshold < 1.0
            or angle <= 0.0
            or not 0.0 <= normal < 1.0
            or changes < 0
        ):
            raise ValueError("Two-phase capability event policy is invalid.")
        self.phase_threshold = threshold
        self.contact_angle_tolerance = angle
        self.minimum_overturning_normal = normal
        self.maximum_topology_changes = changes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-phase-capability-event-policy",
                "phase_threshold": threshold,
                "contact_angle_tolerance": angle,
                "minimum_overturning_normal": normal,
                "maximum_topology_changes": changes,
            }
        )

    @staticmethod
    def _boundary_mask(two_phase: PreparedIncompressibleTwoPhaseVOF, /) -> Array:
        shape = two_phase.plan.discretization.cell_shape
        mask = jnp.zeros(shape, dtype=bool)
        for axis, grid_axis in enumerate(
            two_phase.plan.discretization.grid.structured_axes
        ):
            if grid_axis.periodic:
                continue
            lower = [slice(None)] * len(shape)
            upper = [slice(None)] * len(shape)
            lower[axis] = 0
            upper[axis] = -1
            mask = mask.at[tuple(lower)].set(True)
            mask = mask.at[tuple(upper)].set(True)
        return mask

    @staticmethod
    def _cell_coordinates(two_phase: PreparedIncompressibleTwoPhaseVOF, /) -> Array:
        axes = two_phase.plan.discretization.grid.structured_axes
        shape = two_phase.plan.discretization.cell_shape
        components = []
        for axis, grid_axis in enumerate(axes):
            reshape = [1] * len(shape)
            reshape[axis] = shape[axis]
            components.append(
                jnp.broadcast_to(grid_axis.interval_centers.reshape(reshape), shape)
            )
        return jnp.stack(components, axis=-1)

    def evaluate(
        self,
        two_phase: PreparedIncompressibleTwoPhaseVOF,
        state: TwoPhaseVOFState,
        /,
        *,
        previous_state: TwoPhaseVOFState | None = None,
        body: TwoPhaseMovingBodyPlan | None = None,
    ) -> TwoPhaseCapabilityEvidence:
        if not isinstance(two_phase, PreparedIncompressibleTwoPhaseVOF):
            raise TypeError("two_phase must be a PreparedIncompressibleTwoPhaseVOF.")
        if body is not None and not isinstance(body, TwoPhaseMovingBodyPlan):
            raise TypeError("body must be a TwoPhaseMovingBodyPlan or None.")
        alpha = two_phase.alpha(state)
        previous_alpha = (
            alpha if previous_state is None else two_phase.alpha(previous_state)
        )
        boundary = self._boundary_mask(two_phase)
        current_liquid = alpha >= self.phase_threshold
        previous_liquid = previous_alpha >= self.phase_threshold
        wetting = boundary & ~previous_liquid & current_liquid
        drying = boundary & previous_liquid & ~current_liquid
        mixed = (alpha > 1.0e-12) & (alpha < 1.0 - 1.0e-12)
        previous_mixed = (previous_alpha > 1.0e-12) & (previous_alpha < 1.0 - 1.0e-12)
        moving_contact = (
            boundary & (mixed | previous_mixed) & (current_liquid != previous_liquid)
        )
        boundary_contact = boundary & mixed
        plic = two_phase.plic(alpha)
        vertical_component = jnp.abs(plic.normal[..., -1])
        overturning = mixed & (vertical_component < self.minimum_overturning_normal)
        surface_piercing = jnp.zeros_like(alpha, dtype=bool)
        body_contact = jnp.zeros_like(alpha, dtype=bool)
        if body is not None:
            coordinates = self._cell_coordinates(two_phase)
            if body.center.size != coordinates.shape[-1]:
                raise ValueError("Moving body dimension does not match the VOF grid.")
            distance = jnp.linalg.norm(
                coordinates - body.center.reshape((1,) * alpha.ndim + (-1,)),
                axis=-1,
            )
            body_cells = distance <= body.radius
            body_contact = body_cells & mixed
            has_liquid = jnp.any(body_cells & current_liquid)
            has_gas = jnp.any(body_cells & ~current_liquid)
            surface_piercing = body_cells & mixed & has_liquid & has_gas
        expected_normal = jnp.abs(jnp.cos(two_phase.plan.material.contact_angle))
        angle_residual = jnp.asarray(0.0, dtype=alpha.dtype)
        for axis, grid_axis in enumerate(
            two_phase.plan.discretization.grid.structured_axes
        ):
            if grid_axis.periodic:
                continue
            on_axis_boundary = jnp.zeros_like(boundary)
            for index in (0, -1):
                location = [slice(None)] * alpha.ndim
                location[axis] = index
                on_axis_boundary = on_axis_boundary.at[tuple(location)].set(True)
            local = jnp.where(
                on_axis_boundary & mixed,
                jnp.abs(jnp.abs(plic.normal[..., axis]) - expected_normal),
                0.0,
            )
            angle_residual = jnp.maximum(angle_residual, jnp.max(local))
        topology = two_phase.topology_evidence(state, previous_alpha)
        topology_count = jnp.sum(topology.changed_cell_mask.astype(jnp.int32))
        wetting_event = jnp.any(wetting)
        drying_event = jnp.any(drying)
        moving_contact_event = jnp.any(moving_contact)
        piercing_event = jnp.any(surface_piercing)
        body_contact_event = jnp.any(body_contact)
        breaking_event = jnp.any(overturning) | (
            topology_count > self.maximum_topology_changes
        )
        derivative_available = ~(
            wetting_event
            | drying_event
            | moving_contact_event
            | piercing_event
            | body_contact_event
            | breaking_event
        )
        finite = (
            topology.finite
            & plic.finite
            & jnp.isfinite(angle_residual)
            & jnp.all(jnp.isfinite(alpha))
        )
        successful = (
            finite & plic.valid & (angle_residual <= self.contact_angle_tolerance)
        )
        event_code = (
            wetting_event.astype(jnp.int32)
            + 2 * drying_event.astype(jnp.int32)
            + 4 * moving_contact_event.astype(jnp.int32)
            + 8 * piercing_event.astype(jnp.int32)
            + 16 * body_contact_event.astype(jnp.int32)
            + 32 * breaking_event.astype(jnp.int32)
        )
        return TwoPhaseCapabilityEvidence(
            wetting_cells=wetting,
            drying_cells=drying,
            moving_contact_cells=moving_contact,
            boundary_contact_cells=boundary_contact,
            surface_piercing_cells=surface_piercing,
            body_contact_cells=body_contact,
            overturning_cells=overturning,
            topology_change_count=topology_count,
            contact_angle_residual=angle_residual,
            wetting_event=wetting_event,
            drying_event=drying_event,
            moving_contact_event=moving_contact_event,
            surface_piercing_event=piercing_event,
            body_contact_event=body_contact_event,
            breaking_or_overturning_event=breaking_event,
            derivative_available=derivative_available,
            finite=finite,
            successful=successful,
            event_code=event_code,
            event_id=canonical_fingerprint(
                {
                    "kind": "two-phase-capability-event",
                    "plan": self.plan_id,
                    "two_phase": two_phase.prepared_id,
                    "body": "none" if body is None else body.plan_id,
                }
            ),
        )


class TwoPhaseRemeshEvidence(StrictModule):
    liquid_volume_defect: Array
    scalar_content_defect: dict[str, Array]
    momentum_defect: Array
    source_coverage_residual: Array
    target_coverage_residual: Array
    topology_changed: Array
    derivative_available: Array
    finite: Array
    conservative: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)


class TwoPhaseRemeshResult(StrictModule):
    state: TwoPhaseVOFState
    evidence: TwoPhaseRemeshEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class ConservativeTwoPhaseRemeshPlan(StrictModule, NonTrainableState):
    source: PreparedIncompressibleTwoPhaseVOF
    target: PreparedIncompressibleTwoPhaseVOF
    cell_overlap_volume: Array
    face_transfer: tuple[Array, ...]
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: PreparedIncompressibleTwoPhaseVOF,
        target: PreparedIncompressibleTwoPhaseVOF,
        cell_overlap_volume: ArrayLike,
        face_transfer: tuple[ArrayLike, ...],
        /,
        *,
        tolerance: float = 1.0e-10,
    ):
        if not isinstance(source, PreparedIncompressibleTwoPhaseVOF) or not isinstance(
            target, PreparedIncompressibleTwoPhaseVOF
        ):
            raise TypeError("source and target must be prepared two-phase products.")
        overlap = np.asarray(cell_overlap_volume, dtype=float)
        source_volume = np.asarray(source.plan.discretization.cell_volumes).reshape((-1,))
        target_volume = np.asarray(target.plan.discretization.cell_volumes).reshape((-1,))
        expected = (target_volume.size, source_volume.size)
        tolerance_ = float(tolerance)
        matrices = tuple(np.asarray(value, dtype=float) for value in face_transfer)
        if (
            overlap.shape != expected
            or np.any(~np.isfinite(overlap))
            or np.any(overlap < 0.0)
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or len(matrices) != len(source.plan.discretization.face_layouts)
            or len(matrices) != len(target.plan.discretization.face_layouts)
            or not np.allclose(np.sum(overlap, axis=0), source_volume, atol=tolerance_)
            or not np.allclose(np.sum(overlap, axis=1), target_volume, atol=tolerance_)
        ):
            raise ValueError("Two-phase cell overlap is not a conservative full cover.")
        for matrix, source_layout, target_layout in zip(
            matrices,
            source.plan.discretization.face_layouts,
            target.plan.discretization.face_layouts,
            strict=True,
        ):
            if (
                matrix.shape
                != (int(np.prod(target_layout.shape)), int(np.prod(source_layout.shape)))
                or np.any(~np.isfinite(matrix))
                or np.any(matrix < 0.0)
                or not np.allclose(np.sum(matrix, axis=0), 1.0, atol=tolerance_)
            ):
                raise ValueError("Two-phase face transfer is not conservative.")
        self.source = source
        self.target = target
        self.cell_overlap_volume = jnp.asarray(overlap)
        self.face_transfer = tuple(jnp.asarray(value) for value in matrices)
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-two-phase-remesh",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "cell_shape": list(overlap.shape),
                "face_shapes": [list(value.shape) for value in matrices],
                "tolerance": tolerance_,
            }
        )

    def transfer(self, state: TwoPhaseVOFState, /) -> TwoPhaseRemeshResult:
        source_volume = self.source.plan.discretization.cell_volumes
        target_volume = self.target.plan.discretization.cell_volumes
        liquid_concentration = state.liquid_content / source_volume
        liquid_content = contract(
            "ts,s->t", self.cell_overlap_volume, liquid_concentration.reshape((-1,))
        ).reshape(target_volume.shape)
        scalars = {
            name: contract(
                "ts,s->t",
                self.cell_overlap_volume,
                (value / source_volume).reshape((-1,)),
            ).reshape(target_volume.shape)
            for name, value in state.phase_scalar_content.items()
        }
        momentum = tuple(
            contract("ts,s->t", matrix, value.reshape((-1,))).reshape(layout.shape)
            for matrix, value, layout in zip(
                self.face_transfer,
                state.momentum,
                self.target.plan.discretization.face_layouts,
                strict=True,
            )
        )
        alpha = liquid_content / target_volume
        candidate = TwoPhaseVOFState(
            liquid_content=liquid_content,
            momentum=momentum,
            phase_scalar_content=scalars,
            level_set=self.target.level_set_from_alpha(alpha),
        )
        scalar_defect = {
            name: jnp.sum(scalars[name]) - jnp.sum(value)
            for name, value in state.phase_scalar_content.items()
        }
        momentum_defect = jnp.stack(
            tuple(
                jnp.sum(target_value) - jnp.sum(source_value)
                for target_value, source_value in zip(
                    momentum, state.momentum, strict=True
                )
            )
        )
        source_coverage = jnp.max(
            jnp.abs(
                jnp.sum(self.cell_overlap_volume, axis=0) - source_volume.reshape((-1,))
            )
        )
        target_coverage = jnp.max(
            jnp.abs(
                jnp.sum(self.cell_overlap_volume, axis=1) - target_volume.reshape((-1,))
            )
        )
        liquid_defect = jnp.sum(liquid_content) - jnp.sum(state.liquid_content)
        topology = self.target.topology_evidence(candidate)
        scalar_finite = (
            jnp.asarray(True)
            if not scalars
            else jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(value)) for value in scalars.values())
                )
            )
        )
        finite = (
            topology.finite
            & jnp.all(jnp.isfinite(liquid_content))
            & jnp.all(jnp.isfinite(momentum_defect))
            & scalar_finite
        )
        scalar_max = (
            jnp.asarray(0.0, dtype=liquid_content.dtype)
            if not scalar_defect
            else jnp.max(jnp.abs(jnp.stack(tuple(scalar_defect.values()))))
        )
        conservative = (
            (jnp.abs(liquid_defect) <= self.tolerance)
            & (jnp.max(jnp.abs(momentum_defect)) <= self.tolerance)
            & (scalar_max <= self.tolerance)
            & (source_coverage <= self.tolerance)
            & (target_coverage <= self.tolerance)
        )
        successful = finite & conservative & topology.valid
        evidence = TwoPhaseRemeshEvidence(
            liquid_volume_defect=liquid_defect,
            scalar_content_defect=scalar_defect,
            momentum_defect=momentum_defect,
            source_coverage_residual=source_coverage,
            target_coverage_residual=target_coverage,
            topology_changed=jnp.asarray(
                self.source.plan.discretization.cell_shape
                != self.target.plan.discretization.cell_shape
            ),
            derivative_available=jnp.asarray(False),
            finite=finite,
            conservative=conservative,
            successful=successful,
            epoch_id=canonical_fingerprint(
                {
                    "kind": "two-phase-remesh-epoch",
                    "plan": self.plan_id,
                    "topology_change": self.source.plan.discretization.cell_shape
                    != self.target.plan.discretization.cell_shape,
                }
            ),
        )
        return TwoPhaseRemeshResult(candidate, evidence, successful, self.plan_id)


__all__ = [
    "ConservativeTwoPhaseRemeshPlan",
    "TwoPhaseCapabilityEventPlan",
    "TwoPhaseCapabilityEvidence",
    "TwoPhaseRemeshEvidence",
    "TwoPhaseRemeshResult",
]
