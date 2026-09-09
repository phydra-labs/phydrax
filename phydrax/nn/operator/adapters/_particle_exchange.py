#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.particle._assembly import ParticleExchangeLedger
from ....discretization.particle._pairwise import (
    ParticlePairGeometry,
    ParticlePairRelation,
    scatter_pair_exchange,
)
from ....discretization.particle._precision import ParticleAccumulation
from ..data import FunctionSamples, OperatorBatch
from ..training._trained_operator import TrainedOperator


PairwiseExchangeKind: TypeAlias = Literal["vector", "central_force", "scalar_flux"]


class PairwiseExchangeFeatureSchema(StrictModule, NonTrainableState):
    """Exact learned edge-feature layout for one fixed particle relation."""

    names: tuple[str, ...] = eqx.field(static=True)
    units: tuple[str, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: tuple[str, ...],
        units: tuple[str, ...],
        /,
        *,
        dtype: Any,
        relation_schema_id: str,
    ):
        names_ = tuple(str(value).strip() for value in names)
        units_ = tuple(str(value).strip() for value in units)
        dtype_ = np.dtype(dtype)
        relation = str(relation_schema_id).strip()
        if (
            not names_
            or len(names_) != len(units_)
            or len(set(names_)) != len(names_)
            or any(not value for value in (*names_, *units_))
            or not jnp.issubdtype(dtype_, jnp.floating)
            or not relation
        ):
            raise ValueError("Pairwise exchange feature schema is invalid.")
        self.names = names_
        self.units = units_
        self.dtype = dtype_.name
        self.relation_schema_id = relation
        self.schema_id = canonical_fingerprint(
            {
                "kind": "pairwise-exchange-feature-schema",
                "names": list(names_),
                "units": list(units_),
                "dtype": dtype_.name,
                "relation_schema": relation,
            }
        )

    @property
    def width(self) -> int:
        return len(self.names)


def particle_pair_operator_batch(
    features: ArrayLike,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    /,
    *,
    source_name: str = "pair_features",
    query_name: str = "pairs",
) -> OperatorBatch:
    """Represent one fixed-capacity pair realization as a point-cloud operator batch."""

    if not isinstance(pairs, ParticlePairRelation):
        raise TypeError("pairs must be ParticlePairRelation.")
    if not isinstance(geometry, ParticlePairGeometry):
        raise TypeError("geometry must be ParticlePairGeometry.")
    if geometry.relation_schema_id != pairs.relation_schema_id:
        raise ValueError("Pair relation and geometry schemas differ.")
    values = jnp.asarray(features)
    if values.ndim != 2 or values.shape[0] != pairs.capacity:
        raise ValueError("Pair features must have shape (capacity, features).")
    source = str(source_name).strip()
    query = str(query_name).strip()
    if not source or not query:
        raise ValueError("Pair operator source and query names must be non-empty.")
    valid = pairs.valid & geometry.valid
    safe = jnp.where(valid[:, None], values, jnp.zeros_like(values))
    coordinates = geometry.displacement
    return OperatorBatch(
        inputs={
            source: FunctionSamples(
                values=safe,
                coordinates=coordinates,
                mask=valid,
                support_id=pairs.relation_schema_id,
            )
        },
        queries={
            query: FunctionSamples(
                values=None,
                coordinates=coordinates,
                mask=valid,
                support_id=pairs.relation_schema_id,
            )
        },
    )


class PairwiseExchangeBindingPlan(StrictModule, NonTrainableState):
    """Artifact-bound construction for a conservative same-set learned exchange."""

    feature_schema: PairwiseExchangeFeatureSchema
    exchange_kind: PairwiseExchangeKind = eqx.field(static=True)
    model_artifact_id: str = eqx.field(static=True)
    source_name: str = eqx.field(static=True)
    target_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)
    accumulation: ParticleAccumulation = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        feature_schema: PairwiseExchangeFeatureSchema,
        /,
        *,
        exchange_kind: PairwiseExchangeKind,
        model_artifact_id: str,
        source_name: str = "pair_features",
        target_name: str = "pair_exchange",
        query_name: str = "pairs",
        accumulation: ParticleAccumulation = "deterministic",
        conservation_tolerance: float = 1e-10,
    ):
        if not isinstance(feature_schema, PairwiseExchangeFeatureSchema):
            raise TypeError("feature_schema must be PairwiseExchangeFeatureSchema.")
        kind = str(exchange_kind)
        artifact = str(model_artifact_id).strip()
        source = str(source_name).strip()
        target = str(target_name).strip()
        query = str(query_name).strip()
        tolerance = float(conservation_tolerance)
        if kind not in ("vector", "central_force", "scalar_flux"):
            raise ValueError("Unknown pairwise exchange kind.")
        if not artifact or not source or not target or not query:
            raise ValueError("Pairwise exchange binding identities must be non-empty.")
        if accumulation not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown particle accumulation policy.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conservation_tolerance must be finite and nonnegative.")
        self.feature_schema = feature_schema
        self.exchange_kind = kind
        self.model_artifact_id = artifact
        self.source_name = source
        self.target_name = target
        self.query_name = query
        self.accumulation = accumulation
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pairwise-exchange-binding-plan",
                "feature_schema": feature_schema.schema_id,
                "exchange_kind": kind,
                "model_artifact": artifact,
                "source": source,
                "target": target,
                "query": query,
                "accumulation": accumulation,
                "conservation_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        trained: TrainedOperator,
        pairs: ParticlePairRelation,
        geometry: ParticlePairGeometry,
        /,
    ) -> PreparedPairwiseExchangeBinding:
        if not isinstance(trained, TrainedOperator):
            raise TypeError("trained must be TrainedOperator.")
        if trained.artifact_id != self.model_artifact_id:
            raise ValueError("Trained pair model artifact does not match the plan.")
        if not pairs.same_set or not pairs.unordered:
            raise ValueError(
                "Conservative pair exchange requires unordered same-set relations."
            )
        if pairs.relation_schema_id != self.feature_schema.relation_schema_id:
            raise ValueError("Pair relation does not match the feature schema.")
        template = particle_pair_operator_batch(
            jnp.zeros(
                (pairs.capacity, self.feature_schema.width),
                dtype=np.dtype(self.feature_schema.dtype),
            ),
            pairs,
            geometry,
            source_name=self.source_name,
            query_name=self.query_name,
        )
        prepared = trained.prepare(template)
        return PreparedPairwiseExchangeBinding(
            trained,
            prepared.physical_batch,
            pairs,
            geometry,
            self,
        )


class LearnedPairwiseExchangeResult(StrictModule):
    pair_values: Array
    particle_values: Array
    ledger: ParticleExchangeLedger
    successful: Array
    binding_id: str = eqx.field(static=True)


class PreparedPairwiseExchangeBinding(StrictModule, NonTrainableState):
    trained: TrainedOperator
    template: OperatorBatch
    pairs: ParticlePairRelation
    geometry: ParticlePairGeometry
    plan: PairwiseExchangeBindingPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        trained: TrainedOperator,
        template: OperatorBatch,
        pairs: ParticlePairRelation,
        geometry: ParticlePairGeometry,
        plan: PairwiseExchangeBindingPlan,
        /,
    ):
        self.trained = trained
        self.template = template
        self.pairs = pairs
        self.geometry = geometry
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-pairwise-exchange-binding",
                "plan": plan.plan_id,
                "task": trained.task_fingerprint,
                "contract": trained.contract_fingerprint,
                "normalization": trained.normalization_fingerprint,
                "relation": pairs.relation_schema_id,
                "geometry": geometry.schema_id,
            }
        )

    def evaluate(
        self,
        features: ArrayLike,
        /,
        *,
        velocities: ArrayLike | None = None,
    ) -> LearnedPairwiseExchangeResult:
        values = jnp.asarray(features)
        expected = (self.pairs.capacity, self.plan.feature_schema.width)
        if values.shape != expected:
            raise ValueError(f"Pair features must have shape {expected}.")
        if np.dtype(values.dtype).name != self.plan.feature_schema.dtype:
            raise TypeError("Pair features do not match the bound dtype.")
        valid = self.pairs.valid & self.geometry.valid
        values = eqx.error_if(
            values,
            jnp.any(valid[:, None] & ~jnp.isfinite(values)),
            "Active learned pair features must be finite.",
        )
        batch = particle_pair_operator_batch(
            values,
            self.pairs,
            self.geometry,
            source_name=self.plan.source_name,
            query_name=self.plan.query_name,
        )
        prediction = self.trained.predict_prepared(
            self.trained.execution_plan.prepare_prevalidated(batch)
        )
        raw = jnp.asarray(prediction.field(self.plan.target_name).values)
        dimension = int(self.geometry.displacement.shape[-1])
        if self.plan.exchange_kind == "vector":
            if raw.shape != (self.pairs.capacity, dimension):
                raise ValueError("Vector pair exchange has the wrong shape.")
            pair_values = raw
        else:
            if raw.shape == (self.pairs.capacity, 1):
                raw = raw[:, 0]
            if raw.shape != (self.pairs.capacity,):
                raise ValueError("Scalar pair exchange has the wrong shape.")
            pair_values = (
                raw[:, None] * self.geometry.direction
                if self.plan.exchange_kind == "central_force"
                else raw
            )
        mask = valid.reshape(valid.shape + (1,) * (pair_values.ndim - 1))
        pair_values = jnp.where(mask, pair_values, jnp.zeros_like(pair_values))
        pair_values = eqx.error_if(
            pair_values,
            jnp.any(~jnp.isfinite(pair_values)),
            "Active learned pair exchange contains nonfinite values.",
        )
        particle_values = scatter_pair_exchange(
            self.pairs,
            pair_values,
            size=self.pairs.relation.source_size,
            accumulation=self.plan.accumulation,
            valid=valid,
        )
        ledger = ParticleExchangeLedger.from_exchange(
            self.pairs,
            self.geometry,
            pair_values,
            particle_values,
            velocities=velocities,
        )
        successful = ledger.finite & (
            ledger.action_reaction_defect <= self.plan.conservation_tolerance
        )
        return LearnedPairwiseExchangeResult(
            pair_values,
            particle_values,
            ledger,
            successful,
            self.prepared_id,
        )

    def __call__(
        self,
        features: ArrayLike,
        /,
        *,
        velocities: ArrayLike | None = None,
    ) -> LearnedPairwiseExchangeResult:
        return self.evaluate(features, velocities=velocities)


__all__ = [
    "LearnedPairwiseExchangeResult",
    "PairwiseExchangeBindingPlan",
    "PairwiseExchangeFeatureSchema",
    "PairwiseExchangeKind",
    "PreparedPairwiseExchangeBinding",
    "particle_pair_operator_batch",
]
