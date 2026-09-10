#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.axes as cx

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..conditions import Observation
from ..domain import DomainComponent, PointBatch
from ..domain._observation import indexed_field
from ..fidelity import FidelityDataset
from ..integration import fixed, from_samples, mean_over
from ._residual import ResidualPenalty


class PreparedFidelityObservation(StrictModule, NonTrainableState):
    """Fixed target-fidelity observation term and its complete source evidence."""

    term: ResidualPenalty
    batch: PointBatch
    targets: Array
    weights: Array
    hierarchy_fingerprint: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    level_id: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    split_group_ids: tuple[str, ...] = eqx.field(static=True)
    pair_ids: tuple[str, ...] = eqx.field(static=True)
    evaluation_ids: tuple[str, ...] = eqx.field(static=True)
    rejected_evaluation_ids: tuple[str, ...] = eqx.field(static=True)
    observation_set_id: str = eqx.field(static=True)


def prepare_fidelity_observation_penalty(
    dataset: FidelityDataset,
    level_id: str,
    /,
    *,
    field: str,
    component: DomainComponent,
    scale: ArrayLike = 1.0,
    standard_deviation: ArrayLike | None = None,
) -> PreparedFidelityObservation:
    """Prepare active fidelity rows as an exact fixed PINN observation penalty."""

    if not isinstance(dataset, FidelityDataset):
        raise TypeError("dataset must be a FidelityDataset.")
    if not isinstance(component, DomainComponent):
        raise TypeError("component must be a DomainComponent.")
    level = dataset.hierarchy.level(level_id)
    field_name = str(field)
    if not field_name:
        raise ValueError("field must be non-empty.")
    rows = dataset.evaluations_at(level.level_id)
    active = tuple(row for row in rows if bool(np.asarray(row.valid)))
    rejected = tuple(row for row in rows if not bool(np.asarray(row.valid)))
    if not active:
        raise ValueError(f"Fidelity level {level.level_id!r} has no active observations.")
    case_by_id = {case.case_id: case for case in dataset.cases}
    cases = tuple(case_by_id[row.case_id] for row in active)
    coordinates = _stack_case_coordinates(cases, component)
    base_batch = component.points(coordinates)
    axis_names = base_batch.structure.axis_names
    if axis_names is None or len(axis_names) != 1:
        raise ValueError("Fidelity point observations require one explicit sample axis.")
    index_key = f"__fidelity_observation_index__:{level.level_id}"
    index = cx.AxisArray(
        jnp.arange(len(active), dtype=jnp.int32),
        dims=(axis_names[0],),
    )
    batch = PointBatch(
        frozendict((*base_batch.points.items(), (index_key, index))),
        base_batch.structure,
        metadata=base_batch.metadata,
    )
    targets = _stack_observables(active)
    deviations = (
        jnp.ones((len(active),), dtype=targets.dtype)
        if standard_deviation is None
        else jnp.broadcast_to(
            jnp.asarray(standard_deviation, dtype=targets.dtype),
            (len(active),),
        )
    )
    if bool(jnp.any(~jnp.isfinite(deviations) | (deviations <= 0.0))):
        raise ValueError("standard_deviation must be finite and positive.")
    log_weights = jnp.stack(
        tuple(jnp.asarray(case.log_weight, dtype=targets.dtype) for case in cases)
    )
    raw_weights = jnp.exp(log_weights - jnp.max(log_weights)) / (deviations * deviations)
    weights = raw_weights / jnp.mean(raw_weights)
    target = indexed_field(
        component.domain,
        targets,
        size=len(active),
        index_key=index_key,
        owner="Fidelity observation target",
    )
    density = indexed_field(
        component.domain,
        weights,
        size=len(active),
        index_key=index_key,
        owner="Fidelity observation weights",
    )
    condition = Observation(field_name, component, target)
    realization = from_samples(mean_over(component), batch)
    term = ResidualPenalty(
        condition,
        fixed(realization),
        density=density,
        scale=scale,
    )
    observation_set_id = canonical_fingerprint(
        {
            "kind": "fidelity-point-observation-set",
            "hierarchy": dataset.hierarchy.fingerprint,
            "dataset": dataset.dataset_id,
            "level": level.level_id,
            "field": field_name,
            "component": repr(component.spec),
            "targets": array_tree_fingerprint(targets),
            "weights": array_tree_fingerprint(weights),
            "evaluations": [row.evaluation_id for row in active],
            "rejected": [row.evaluation_id for row in rejected],
        }
    )
    return PreparedFidelityObservation(
        term=term,
        batch=batch,
        targets=targets,
        weights=weights,
        hierarchy_fingerprint=dataset.hierarchy.fingerprint,
        dataset_id=dataset.dataset_id,
        level_id=level.level_id,
        field_name=field_name,
        case_ids=tuple(row.case_id for row in active),
        split_group_ids=tuple(case.split_group_id for case in cases),
        pair_ids=tuple(row.pair_id for row in active),
        evaluation_ids=tuple(row.evaluation_id for row in active),
        rejected_evaluation_ids=tuple(row.evaluation_id for row in rejected),
        observation_set_id=observation_set_id,
    )


def _stack_case_coordinates(cases, component: DomainComponent, /):
    first = cases[0].inputs
    if isinstance(first, Mapping):
        labels = tuple(label for label in component.domain.labels if label in first)
        if not labels:
            raise ValueError("Fidelity observation coordinate mappings cannot be empty.")
        if any(
            not isinstance(case.inputs, Mapping)
            or {str(label) for label in case.inputs} != set(labels)
            for case in cases
        ):
            raise ValueError(
                "Fidelity observation cases must share one coordinate mapping layout."
            )
        return {
            label: jnp.stack(
                tuple(jnp.asarray(case.inputs[label], dtype=float) for case in cases)
            )
            for label in labels
        }
    if any(isinstance(case.inputs, Mapping) for case in cases[1:]):
        raise ValueError("Fidelity observation case input layouts must match.")
    return jnp.stack(tuple(jnp.asarray(case.inputs, dtype=float) for case in cases))


def _stack_observables(rows) -> Array:
    leaves = []
    for row in rows:
        observable_leaves = tuple(jax.tree_util.tree_leaves(row.observable))
        if len(observable_leaves) != 1:
            raise ValueError(
                "PINN fidelity observations require one array observable leaf."
            )
        leaves.append(jnp.asarray(observable_leaves[0], dtype=float))
    targets = jnp.stack(tuple(leaves))
    if bool(jnp.any(~jnp.isfinite(targets))):
        raise ValueError("Active PINN fidelity observation targets must be finite.")
    return targets


__all__ = [
    "PreparedFidelityObservation",
    "prepare_fidelity_observation_penalty",
]
