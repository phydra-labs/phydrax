#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._compressible_contracts import CompressibleKineticPopulationState
from ._compressible_frame import KineticFrameRemapResult, remap_kinetic_frame
from ._compressible_rules import CompressibleVelocityRule
from ._positive_kinetic import PositiveCompressibleKineticPlan


class PredictiveRefinementEvidence(StrictModule):
    current_indicator: Array
    predicted_indicator: Array
    refine: Array
    coarsen: Array
    reason_bits: Array
    successful: Array


class PredictiveKineticRefinementPlan(StrictModule, NonTrainableState):
    rule: CompressibleVelocityRule
    refine_threshold: float = eqx.field(static=True)
    coarsen_threshold: float = eqx.field(static=True)
    prediction_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rule: CompressibleVelocityRule,
        /,
        *,
        refine_threshold: float,
        coarsen_threshold: float,
        prediction_steps: int = 2,
    ):
        if not isinstance(rule, CompressibleVelocityRule):
            raise TypeError("rule must be a CompressibleVelocityRule.")
        refine = float(refine_threshold)
        coarsen = float(coarsen_threshold)
        steps = int(prediction_steps)
        if not np.isfinite(refine) or not np.isfinite(coarsen) or refine <= coarsen:
            raise ValueError(
                "Refinement thresholds must be finite with refine > coarsen."
            )
        if steps < 0:
            raise ValueError("prediction_steps must be nonnegative.")
        self.rule = rule
        self.refine_threshold = refine
        self.coarsen_threshold = coarsen
        self.prediction_steps = steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "predictive-kinetic-refinement",
                "rule": rule.rule_id,
                "refine_threshold": refine,
                "coarsen_threshold": coarsen,
                "prediction_steps": steps,
            }
        )

    def evaluate(
        self,
        indicator: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
    ) -> PredictiveRefinementEvidence:
        values = jnp.asarray(indicator)
        if values.ndim != self.rule.dimension:
            raise ValueError("indicator rank must match the velocity dimension.")
        validity = (
            jnp.ones(values.shape, dtype=jnp.bool_)
            if active is None
            else jnp.asarray(active, dtype=jnp.bool_)
        )
        if validity.shape != values.shape:
            raise ValueError("active mask must match indicator shape.")
        predicted = jnp.where(validity, values, -jnp.inf)
        seed = jnp.where(validity & jnp.isfinite(values), values, -jnp.inf)
        velocities = np.asarray(self.rule.velocities, dtype=np.int32)
        for step in range(1, self.prediction_steps + 1):
            advected = tuple(
                jnp.roll(
                    seed,
                    shift=tuple(int(step * component) for component in velocity),
                    axis=tuple(range(self.rule.dimension)),
                )
                for velocity in velocities
            )
            predicted = jnp.maximum(
                predicted, jnp.max(jnp.stack(advected, axis=0), axis=0)
            )
        finite = jnp.isfinite(values) & jnp.isfinite(predicted)
        refine = validity & finite & (predicted >= self.refine_threshold)
        coarsen = validity & finite & (values <= self.coarsen_threshold) & ~refine
        reason = refine.astype(jnp.uint32) | (
            ((predicted > values) & refine).astype(jnp.uint32) << 1
        )
        return PredictiveRefinementEvidence(
            current_indicator=values,
            predicted_indicator=predicted,
            refine=refine,
            coarsen=coarsen,
            reason_bits=reason,
            successful=jnp.all(finite | ~validity),
        )


class KineticAMRTransferEvidence(StrictModule):
    mass_defect: Array
    minimum_population: Array
    successful: Array


class KineticAMRTransferResult(StrictModule):
    state: CompressibleKineticPopulationState
    evidence: KineticAMRTransferEvidence


def _amr_transfer_result(
    source: CompressibleKineticPopulationState,
    target: CompressibleKineticPopulationState,
    /,
) -> KineticAMRTransferResult:
    source_mass = jnp.mean(jnp.sum(source.population("particle"), axis=-1))
    target_mass = jnp.mean(jnp.sum(target.population("particle"), axis=-1))
    mass_defect = target_mass - source_mass
    positive_minima = tuple(
        jnp.min(value)
        for value, field in zip(target.populations, target.layout.fields, strict=True)
        if field.positive
    )
    minimum = jnp.min(jnp.stack(positive_minima))
    finite = jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in target.populations))
    )
    tolerance = (
        512.0 * jnp.finfo(target_mass.dtype).eps * jnp.maximum(jnp.abs(source_mass), 1.0)
    )
    successful = finite & (minimum >= 0.0) & (jnp.abs(mass_defect) <= tolerance)
    return KineticAMRTransferResult(
        target,
        KineticAMRTransferEvidence(mass_defect, minimum, successful),
    )


class KineticAMRTransferPlan(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, refinement_ratio: int = 2, /):
        dim = int(dimension)
        ratio = int(refinement_ratio)
        if dim not in (1, 2, 3) or ratio < 2:
            raise ValueError("AMR dimension or refinement_ratio is invalid.")
        self.dimension = dim
        self.refinement_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {"kind": "kinetic-amr-transfer", "dimension": dim, "ratio": ratio}
        )

    def prolong(
        self, state: CompressibleKineticPopulationState, /
    ) -> KineticAMRTransferResult:
        if len(state.spatial_shape) != self.dimension:
            raise ValueError("State dimension does not match the AMR plan.")

        def repeat(value: Array) -> Array:
            result = value
            for axis in range(self.dimension):
                result = jnp.repeat(result, self.refinement_ratio, axis=axis)
            return result

        candidate = CompressibleKineticPopulationState(
            tuple(repeat(value) for value in state.populations),
            repeat(state.equilibrium_dual),
            repeat(state.stabilizer[..., None])[..., 0],
            repeat(state.frame_velocity),
            repeat(state.frame_temperature_scale[..., None])[..., 0],
            state.layout,
            state.model_id,
            state.rule_id,
        )
        return _amr_transfer_result(state, candidate)

    def restrict(
        self, state: CompressibleKineticPopulationState, /
    ) -> KineticAMRTransferResult:
        if len(state.spatial_shape) != self.dimension:
            raise ValueError("State dimension does not match the AMR plan.")
        ratio = self.refinement_ratio
        if any(size % ratio for size in state.spatial_shape):
            raise ValueError("Fine-grid extents must be divisible by refinement_ratio.")

        def average(value: Array) -> Array:
            result = value
            for axis in reversed(range(self.dimension)):
                shape = result.shape
                result = result.reshape(
                    shape[:axis] + (shape[axis] // ratio, ratio) + shape[axis + 1 :]
                ).mean(axis=axis + 1)
            return result

        candidate = CompressibleKineticPopulationState(
            tuple(average(value) for value in state.populations),
            average(state.equilibrium_dual),
            average(state.stabilizer[..., None])[..., 0],
            average(state.frame_velocity),
            average(state.frame_temperature_scale[..., None])[..., 0],
            state.layout,
            state.model_id,
            state.rule_id,
        )
        return _amr_transfer_result(state, candidate)


class MappedKineticGridPlan(StrictModule, NonTrainableState):
    jacobian: Array
    inverse_jacobian: Array
    determinant: Array
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, jacobian: ArrayLike, /):
        matrix = np.asarray(jacobian)
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError("jacobian must end in equal physical/reference dimensions.")
        dimension = matrix.shape[-1]
        if dimension not in (1, 2, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Mapped kinetic Jacobian is invalid.")
        determinant = np.linalg.det(matrix)
        if np.any(determinant <= 0.0):
            raise ValueError("Mapped kinetic Jacobian must preserve orientation.")
        inverse_result = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            np.broadcast_to(np.eye(dimension), matrix.shape),
            policy=LinearSolvePolicy(DenseLU()),
        )
        if not bool(jnp.all(inverse_result.successful)):
            raise ValueError("Mapped kinetic Jacobian factorization failed.")
        inverse = np.asarray(inverse_result.value)
        self.jacobian = jnp.asarray(matrix)
        self.inverse_jacobian = jnp.asarray(inverse)
        self.determinant = jnp.asarray(determinant)
        self.dimension = dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-kinetic-grid",
                "jacobian": matrix.tolist(),
                "inverse": inverse.tolist(),
                "determinant": determinant.tolist(),
            }
        )

    def contravariant_velocities(self, rule: CompressibleVelocityRule, /) -> Array:
        if rule.dimension != self.dimension:
            raise ValueError("Rule and mapped grid dimensions differ.")
        return ein.contract("...ab,qb->...qa", self.inverse_jacobian, rule.velocities)


class MovingKineticGeometryEvidence(StrictModule):
    covered_mass: Array
    uncovered_cells: Array
    covered_cells: Array
    successful: Array


class MovingKineticGeometryResult(StrictModule):
    candidate: CompressibleKineticPopulationState
    previous: CompressibleKineticPopulationState
    evidence: MovingKineticGeometryEvidence


class MovingKineticGeometryPlan(StrictModule, NonTrainableState):
    active_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, active_shape: tuple[int, ...], /):
        shape = tuple(int(value) for value in active_shape)
        if not shape or any(value < 1 for value in shape):
            raise ValueError("active_shape must contain positive extents.")
        self.active_shape = shape
        self.plan_id = canonical_fingerprint(
            {"kind": "moving-kinetic-geometry", "active_shape": list(shape)}
        )

    def update(
        self,
        state: CompressibleKineticPopulationState,
        old_active: ArrayLike,
        new_active: ArrayLike,
        uncovered_state: CompressibleKineticPopulationState,
        /,
    ) -> MovingKineticGeometryResult:
        old_mask = jnp.asarray(old_active, dtype=jnp.bool_)
        new_mask = jnp.asarray(new_active, dtype=jnp.bool_)
        if old_mask.shape != self.active_shape or new_mask.shape != self.active_shape:
            raise ValueError("Geometry masks must match active_shape.")
        if (
            state.spatial_shape != self.active_shape
            or uncovered_state.spatial_shape != self.active_shape
        ):
            raise ValueError("Geometry states must match active_shape.")
        if (
            state.layout.layout_id != uncovered_state.layout.layout_id
            or state.model_id != uncovered_state.model_id
            or state.rule_id != uncovered_state.rule_id
        ):
            raise ValueError(
                "Moving-geometry states have different scientific identities."
            )
        uncovered = ~old_mask & new_mask
        covered = old_mask & ~new_mask
        updated_fields = tuple(
            jnp.where(
                uncovered[..., None],
                fill,
                jnp.where(covered[..., None], jnp.zeros_like(value), value),
            )
            for value, fill in zip(
                state.populations, uncovered_state.populations, strict=True
            )
        )
        updated = CompressibleKineticPopulationState(
            updated_fields,
            jnp.where(
                uncovered[..., None],
                uncovered_state.equilibrium_dual,
                state.equilibrium_dual,
            ),
            jnp.where(uncovered, uncovered_state.stabilizer, state.stabilizer),
            jnp.where(
                uncovered[..., None], uncovered_state.frame_velocity, state.frame_velocity
            ),
            jnp.where(
                uncovered,
                uncovered_state.frame_temperature_scale,
                state.frame_temperature_scale,
            ),
            state.layout,
            state.model_id,
            state.rule_id,
        )
        covered_mass = jnp.sum(
            jnp.where(covered[..., None], state.population("particle"), 0.0)
        )
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in updated_fields))
        )
        positive = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(value >= 0.0)
                    for value, field in zip(
                        updated_fields, state.layout.fields, strict=True
                    )
                    if field.positive
                )
            )
        )
        evidence = MovingKineticGeometryEvidence(
            covered_mass=covered_mass,
            uncovered_cells=jnp.sum(uncovered),
            covered_cells=jnp.sum(covered),
            successful=finite & positive,
        )
        return MovingKineticGeometryResult(updated, state, evidence)


class KineticMultiblockInterfacePlan(StrictModule, NonTrainableState):
    source_model: PositiveCompressibleKineticPlan
    target_model: PositiveCompressibleKineticPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_model: PositiveCompressibleKineticPlan,
        target_model: PositiveCompressibleKineticPlan,
        /,
    ):
        if not isinstance(
            source_model, PositiveCompressibleKineticPlan
        ) or not isinstance(target_model, PositiveCompressibleKineticPlan):
            raise TypeError("Multiblock interface models must be positive kinetic plans.")
        self.source_model = source_model
        self.target_model = target_model
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-multiblock-interface",
                "source": source_model.model_id,
                "target": target_model.model_id,
            }
        )

    def transfer(
        self, state: CompressibleKineticPopulationState, /
    ) -> KineticFrameRemapResult:
        return remap_kinetic_frame(self.source_model, self.target_model, state)


__all__ = [
    "KineticAMRTransferEvidence",
    "KineticAMRTransferResult",
    "KineticAMRTransferPlan",
    "KineticMultiblockInterfacePlan",
    "MappedKineticGridPlan",
    "MovingKineticGeometryEvidence",
    "MovingKineticGeometryResult",
    "MovingKineticGeometryPlan",
    "PredictiveKineticRefinementPlan",
    "PredictiveRefinementEvidence",
]
