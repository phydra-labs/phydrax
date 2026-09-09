#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nn.operator.data import (
    FunctionSamples,
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorTargetBatch,
)
from ...nn.operator.training import OperatorDataset, OperatorSplitPolicy
from ...qualification._reference import ReferenceArtifactManifest


@dataclass(frozen=True)
class CompressibleOperatorCase:
    """One accepted CFD case with leakage and training-rights identity."""

    case_id: str
    geometry_id: str
    system_id: str
    method_id: str
    flow_family_id: str
    qualification_id: str
    artifact: ReferenceArtifactManifest
    mach_number: float
    reynolds_number: float
    angle_of_attack: float
    sequence_coordinate: float | None = None

    def __post_init__(self):
        identifiers = (
            self.case_id,
            self.geometry_id,
            self.system_id,
            self.method_id,
            self.flow_family_id,
            self.qualification_id,
        )
        values = (self.mach_number, self.reynolds_number, self.angle_of_attack)
        if (
            any(not str(value) for value in identifiers)
            or not isinstance(self.artifact, ReferenceArtifactManifest)
            or not self.artifact.training_use_permitted
            or any(not np.isfinite(float(value)) for value in values)
            or self.mach_number < 0.0
            or self.reynolds_number <= 0.0
            or (
                self.sequence_coordinate is not None
                and not np.isfinite(float(self.sequence_coordinate))
            )
        ):
            raise ValueError("Compressible operator case identity or rights are invalid.")

    def provenance(self) -> OperatorCaseProvenance:
        order = {
            "mach_number": self.mach_number,
            "reynolds_number": self.reynolds_number,
            "angle_of_attack": self.angle_of_attack,
        }
        if self.sequence_coordinate is not None:
            order["sequence_coordinate"] = self.sequence_coordinate
        return OperatorCaseProvenance(
            self.case_id,
            identities={
                "geometry_id": self.geometry_id,
                "system_id": self.system_id,
                "method_id": self.method_id,
                "flow_family_id": self.flow_family_id,
                "qualification_id": self.qualification_id,
                "artifact_id": self.artifact.manifest_id,
            },
            order=order,
        )


class CompressibleOperatorDatasetPlan(StrictModule, NonTrainableState):
    """Convert accepted CFD fields to native provenance-safe operator data."""

    condition_names: tuple[str, ...] = eqx.field(static=True)
    geometry_input_name: str = eqx.field(static=True)
    condition_input_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        condition_names: Sequence[str],
        /,
        *,
        geometry_input_name: str = "geometry",
        condition_input_name: str = "conditions",
        query_name: str = "flow",
    ):
        conditions = tuple(str(name) for name in condition_names)
        geometry_name = str(geometry_input_name)
        condition_name = str(condition_input_name)
        query_name_ = str(query_name)
        if (
            not conditions
            or any(not name for name in conditions)
            or len(set(conditions)) != len(conditions)
            or not geometry_name
            or not condition_name
            or not query_name_
        ):
            raise ValueError("Operator dataset names must be unique and non-empty.")
        self.condition_names = conditions
        self.geometry_input_name = geometry_name
        self.condition_input_name = condition_name
        self.query_name = query_name_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-operator-dataset-plan",
                "condition_names": conditions,
                "geometry_input_name": geometry_name,
                "condition_input_name": condition_name,
                "query_name": query_name_,
            }
        )

    @staticmethod
    def default_split_policy(*, seed: int = 0) -> OperatorSplitPolicy:
        return OperatorSplitPolicy(group_by=("geometry_id",), seed=int(seed))

    def build(
        self,
        cases: Sequence[CompressibleOperatorCase],
        condition_values: ArrayLike,
        geometry_coordinates: ArrayLike,
        query_coordinates: ArrayLike,
        targets: Mapping[str, ArrayLike],
        /,
        *,
        geometry_mask: ArrayLike | None = None,
        query_mask: ArrayLike | None = None,
        query_quadrature_weights: ArrayLike | None = None,
    ) -> OperatorDataset:
        cases_ = tuple(cases)
        case_count = len(cases_)
        if (
            case_count == 0
            or any(not isinstance(case, CompressibleOperatorCase) for case in cases_)
            or len({case.case_id for case in cases_}) != case_count
            or not targets
        ):
            raise ValueError("Operator data requires unique accepted cases and targets.")
        conditions = jnp.asarray(condition_values)
        if conditions.shape != (case_count, len(self.condition_names)):
            raise ValueError(
                "Condition values must align with cases and named conditions."
            )
        geometry = jnp.asarray(geometry_coordinates)
        query = jnp.asarray(query_coordinates)
        if geometry.ndim == 2:
            geometry = jnp.broadcast_to(geometry, (case_count,) + geometry.shape)
        if query.ndim == 2:
            query = jnp.broadcast_to(query, (case_count,) + query.shape)
        if (
            geometry.ndim != 3
            or query.ndim != 3
            or geometry.shape[0] != case_count
            or query.shape[0] != case_count
            or geometry.shape[-1] != query.shape[-1]
        ):
            raise ValueError(
                "Geometry and query coordinates must align by case and dimension."
            )
        geometry_mask_ = (
            jnp.ones(geometry.shape[:-1], dtype=bool)
            if geometry_mask is None
            else jnp.broadcast_to(
                jnp.asarray(geometry_mask, dtype=bool), geometry.shape[:-1]
            )
        )
        query_mask_ = (
            jnp.ones(query.shape[:-1], dtype=bool)
            if query_mask is None
            else jnp.broadcast_to(jnp.asarray(query_mask, dtype=bool), query.shape[:-1])
        )
        query_weights = (
            None
            if query_quadrature_weights is None
            else jnp.broadcast_to(jnp.asarray(query_quadrature_weights), query.shape[:-1])
        )
        condition_samples = FunctionSamples(
            values=conditions[:, None, :],
            coordinates=jnp.zeros((1, 1), dtype=conditions.dtype),
            support_id="compressible-condition-point",
        )
        geometry_samples = FunctionSamples(
            values=geometry,
            coordinates=geometry,
            mask=geometry_mask_,
            support_id="compressible-geometry-points",
        )
        query_samples = FunctionSamples(
            values=None,
            coordinates=query,
            quadrature_weights=query_weights,
            mask=query_mask_,
            support_id="compressible-flow-query",
            measure_id=None if query_weights is None else "physical-query-measure",
        )
        batch = OperatorBatch(
            inputs={
                self.condition_input_name: condition_samples,
                self.geometry_input_name: geometry_samples,
            },
            queries={self.query_name: query_samples},
            case_axes=("case",),
            case_shape=(case_count,),
        )
        target_batch = OperatorTargetBatch.from_arrays(
            {name: jnp.asarray(value) for name, value in targets.items()}, batch
        )
        return OperatorDataset(
            batch,
            target_batch,
            tuple(case.provenance() for case in cases_),
        )


__all__ = ["CompressibleOperatorCase", "CompressibleOperatorDatasetPlan"]
