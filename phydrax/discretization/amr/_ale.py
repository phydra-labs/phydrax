#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic SSPRK(3,3) geometry stages for fixed-connectivity patch ALE motion."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._geometry import (
    _validated_revision,
    VariablePatchGeometryPlan,
    VariablePatchGeometryState,
)


class VariablePatchALEStepEvidence(StrictModule, NonTrainableState):
    """Stage-wise GCL and endpoint recurrence evidence for one attempted step."""

    maximum_stage_gcl_defect: Array
    maximum_endpoint_volume_defect: Array
    passed: Array
    proposed_reduction_factor: Array
    evidence_id: str = eqx.field(static=True)


class VariablePatchALEStepGeometry(StrictModule):
    plan: "VariablePatchALEPlan"
    source: VariablePatchGeometryState
    stage_initial: VariablePatchGeometryState
    stage_endpoint: VariablePatchGeometryState
    stage_midpoint: VariablePatchGeometryState
    evidence: VariablePatchALEStepEvidence
    successful: Array

    def committed_geometry(self, /) -> VariablePatchGeometryState:
        return jax.tree.map(
            lambda accepted, rejected: (
                jnp.where(
                    self.successful,
                    accepted,
                    rejected,
                )
                if isinstance(accepted, jax.Array)
                else accepted
            ),
            self.stage_endpoint,
            self.source,
        )


class VariablePatchALEPlan(StrictModule, NonTrainableState):
    """Fixed-connectivity ALE stage compiler with all-or-nothing geometry commit."""

    geometry: VariablePatchGeometryPlan
    endpoint_tolerance: float = eqx.field(static=True)
    minimum_reduction_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: VariablePatchGeometryPlan,
        /,
        *,
        endpoint_tolerance: float = 1.0e-10,
        minimum_reduction_factor: float = 0.1,
    ):
        tolerance = float(endpoint_tolerance)
        reduction = float(minimum_reduction_factor)
        if (
            not isinstance(geometry, VariablePatchGeometryPlan)
            or not 0.0 < reduction < 1.0
            or tolerance < 0.0
        ):
            raise ValueError("Variable patch ALE policy is invalid.")
        self.geometry = geometry
        self.endpoint_tolerance = tolerance
        self.minimum_reduction_factor = reduction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-ale-plan",
                "geometry": geometry.plan_id,
                "endpoint_tolerance": tolerance,
                "minimum_reduction_factor": reduction,
            }
        )

    def prepare_step(
        self,
        source: VariablePatchGeometryState,
        step_size: ArrayLike,
        args=None,
        /,
    ) -> VariablePatchALEStepGeometry:
        if (
            not isinstance(source, VariablePatchGeometryState)
            or source.plan.plan_id != self.geometry.plan_id
        ):
            raise ValueError("ALE source geometry does not match its plan.")
        dt = jnp.asarray(step_size, dtype=source.time.dtype)
        if dt.shape != ():
            raise ValueError("ALE step_size must be scalar.")
        revision = _validated_revision(source.revision)
        next_revision = eqx.error_if(
            revision,
            revision == jnp.iinfo(jnp.int32).max,
            "ALE geometry revision cannot advance beyond int32 capacity.",
        ) + jnp.asarray(1, dtype=jnp.int32)
        initial = self.geometry.state(
            source.time,
            args,
            revision=revision,
        )
        endpoint = self.geometry.state(
            source.time + dt,
            args,
            revision=next_revision,
        )
        midpoint = self.geometry.state(
            source.time + 0.5 * dt,
            args,
            revision=next_revision,
        )
        gcl_terms = tuple(
            jnp.max(defect)
            for state in (initial, endpoint, midpoint)
            for level in state.gcl_defects
            for defect in level
        )
        endpoint_terms = []
        for source_level, target_level, first_level, second_level, third_level in zip(
            source.cell_volumes,
            endpoint.cell_volumes,
            initial.mesh_volume_rates,
            endpoint.mesh_volume_rates,
            midpoint.mesh_volume_rates,
            strict=True,
        ):
            for old, new, first, second, third in zip(
                source_level,
                target_level,
                first_level,
                second_level,
                third_level,
                strict=True,
            ):
                predicted = old + dt * (first / 6.0 + second / 6.0 + 2.0 * third / 3.0)
                endpoint_terms.append(jnp.max(jnp.abs(new - predicted)))
        maximum_gcl = jnp.max(jnp.stack(gcl_terms))
        maximum_endpoint = jnp.max(jnp.stack(tuple(endpoint_terms)))
        finite = jnp.isfinite(dt) & (dt > 0.0)
        passed = (
            finite
            & source.valid
            & initial.valid
            & endpoint.valid
            & midpoint.valid
            & (maximum_endpoint <= self.endpoint_tolerance)
        )
        scale = jnp.maximum(maximum_endpoint / max(self.endpoint_tolerance, 1.0e-30), 1.0)
        reduction = jnp.where(
            passed,
            1.0,
            jnp.maximum(
                self.minimum_reduction_factor,
                0.8 * scale ** (-1.0 / 3.0),
            ),
        )
        evidence = VariablePatchALEStepEvidence(
            maximum_gcl,
            maximum_endpoint,
            passed,
            reduction,
            canonical_fingerprint(
                {
                    "kind": "variable-patch-ale-step-evidence",
                    "plan": self.plan_id,
                    "shape": [
                        list(value.shape)
                        for level in source.cell_volumes
                        for value in level
                    ],
                }
            ),
        )
        return VariablePatchALEStepGeometry(
            self,
            source,
            initial,
            endpoint,
            midpoint,
            evidence,
            passed,
        )


__all__ = [
    "VariablePatchALEPlan",
    "VariablePatchALEStepEvidence",
    "VariablePatchALEStepGeometry",
]
