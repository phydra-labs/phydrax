#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ._features import _validate_acyclic
from ._validation import (
    boolean_array,
    content_id,
    integer_array,
    labels_tuple,
    nonempty_string,
)


class BiospecimenLineage(StrictModule, NonTrainableState):
    """Typed subject-to-observation provenance with pooling and replicate identity."""

    SUBJECT = 0
    SPECIMEN = 1
    ALIQUOT = 2
    LIBRARY = 3
    RUN = 4
    OBSERVATION = 5

    entity_ids: Array
    entity_kinds: Array
    relation: EdgeRelation
    pooled: Array
    biological_replicate_group_ids: Array
    technical_replicate_group_ids: Array
    study_id: str = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        entity_ids: ArrayLike,
        entity_kinds: ArrayLike,
        parent_indices: ArrayLike,
        child_indices: ArrayLike,
        biological_replicate_group_ids: ArrayLike,
        technical_replicate_group_ids: ArrayLike,
        /,
        *,
        study_id: str,
        pooled: ArrayLike | None = None,
        valid: ArrayLike | None = None,
        labels: Sequence[str] | None = None,
    ):
        ids = integer_array("entity_ids", entity_ids, ndim=1)
        kinds = integer_array("entity_kinds", entity_kinds, ndim=1)
        entity_count = int(ids.shape[0])
        if kinds.shape != (entity_count,):
            raise ValueError(
                f"entity_kinds must have shape {(entity_count,)}; got {kinds.shape}."
            )
        ids_host = np.asarray(ids)
        kinds_host = np.asarray(kinds)
        if np.any(ids_host < 0) or np.unique(ids_host).size != ids_host.size:
            raise ValueError("Lineage entity IDs must be unique and non-negative.")
        if np.any((kinds_host < self.SUBJECT) | (kinds_host > self.OBSERVATION)):
            raise ValueError("Lineage entity kinds must lie in [SUBJECT, OBSERVATION].")

        parents = integer_array("parent_indices", parent_indices, ndim=1)
        children = integer_array("child_indices", child_indices, ndim=1)
        if parents.shape != children.shape:
            raise ValueError("Lineage parent and child index shapes must match.")
        edge_shape = (int(parents.shape[0]),)
        valid_ = boolean_array("valid", valid, edge_shape, default=True)
        parent_host = np.asarray(parents)
        child_host = np.asarray(children)
        valid_host = np.asarray(valid_)
        if parent_host.size and (
            np.any(parent_host < 0)
            or np.any(parent_host >= entity_count)
            or np.any(child_host < 0)
            or np.any(child_host >= entity_count)
        ):
            raise ValueError("Lineage indices must lie within the entity space.")
        valid_parents = parent_host[valid_host]
        valid_children = child_host[valid_host]
        if valid_parents.size:
            pairs = np.stack((valid_parents, valid_children), axis=1)
            if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
                raise ValueError("Valid lineage relations must be unique.")
        _validate_acyclic(entity_count, valid_parents, valid_children)
        allowed_transitions = {
            (self.SUBJECT, self.SPECIMEN),
            (self.SPECIMEN, self.ALIQUOT),
            (self.ALIQUOT, self.LIBRARY),
            (self.LIBRARY, self.RUN),
            (self.LIBRARY, self.OBSERVATION),
            (self.RUN, self.OBSERVATION),
        }
        for parent, child in zip(
            valid_parents.tolist(), valid_children.tolist(), strict=True
        ):
            transition = (int(kinds_host[parent]), int(kinds_host[child]))
            if transition not in allowed_transitions:
                raise ValueError(
                    "Lineage edges must follow subject→specimen→aliquot→library, "
                    "then library→run→observation or library→observation."
                )
        incoming = np.bincount(valid_children, minlength=entity_count)
        if np.any(incoming[kinds_host == self.SUBJECT] != 0):
            raise ValueError("Subject entities cannot have lineage parents.")
        if np.any(incoming[kinds_host != self.SUBJECT] == 0):
            raise ValueError("Every non-subject entity must have a lineage parent.")

        pooled_ = boolean_array("pooled", pooled, (entity_count,), default=False)
        pooled_host = np.asarray(pooled_)
        is_multi_parent = incoming > 1
        if np.any(pooled_host != is_multi_parent):
            raise ValueError(
                "pooled must mark exactly the entities with multiple lineage parents."
            )
        if np.any(
            pooled_host
            & ~np.isin(kinds_host, (self.SPECIMEN, self.ALIQUOT, self.LIBRARY))
        ):
            raise ValueError("Only specimen, aliquot, or library entities may be pooled.")

        biological = integer_array(
            "biological_replicate_group_ids",
            biological_replicate_group_ids,
            ndim=1,
        )
        technical = integer_array(
            "technical_replicate_group_ids",
            technical_replicate_group_ids,
            ndim=1,
        )
        if biological.shape != (entity_count,) or technical.shape != (entity_count,):
            raise ValueError(
                "Biological and technical replicate group IDs must match entity_ids."
            )
        biological_host = np.asarray(biological)
        technical_host = np.asarray(technical)
        if np.any(biological_host < -1) or np.any(technical_host < -1):
            raise ValueError("Replicate group IDs must be non-negative or -1.")
        observations = kinds_host == self.OBSERVATION
        if np.any(biological_host[observations] < 0) or np.any(
            technical_host[observations] < 0
        ):
            raise ValueError("Every observation must have both replicate group IDs.")
        _validate_nested_groups(
            biological_host[observations], technical_host[observations]
        )

        study_id_ = nonempty_string("study_id", study_id)
        labels_ = labels_tuple("labels", labels, entity_count)
        relation = EdgeRelation(
            parents,
            children,
            source_size=entity_count,
            target_size=entity_count,
            valid=valid_,
        )
        lineage_id = content_id(
            "biospecimen_lineage",
            {"labels": labels_, "study_id": study_id_},
            (ids, kinds, relation, pooled_, biological, technical),
        )
        self.entity_ids = ids
        self.entity_kinds = kinds
        self.relation = relation
        self.pooled = pooled_
        self.biological_replicate_group_ids = biological
        self.technical_replicate_group_ids = technical
        self.study_id = study_id_
        self.labels = labels_
        self.entity_count = entity_count
        self.lineage_id = lineage_id


class ExperimentalUnitPlan(StrictModule, NonTrainableState):
    """Observation-to-ancestral-unit assignments with condition and block identity."""

    lineage: BiospecimenLineage
    observation_indices: Array
    unit_indices: Array
    condition_ids: Array
    block_group_ids: Array
    included: Array
    capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lineage: BiospecimenLineage,
        observation_indices: ArrayLike,
        unit_indices: ArrayLike,
        condition_ids: ArrayLike,
        /,
        *,
        block_group_ids: ArrayLike | None = None,
        included: ArrayLike | None = None,
    ):
        if not isinstance(lineage, BiospecimenLineage):
            raise TypeError("lineage must be a BiospecimenLineage.")
        observations = integer_array("observation_indices", observation_indices, ndim=1)
        units = integer_array("unit_indices", unit_indices, ndim=1)
        conditions = integer_array("condition_ids", condition_ids, ndim=1)
        capacity = int(observations.shape[0])
        expected = (capacity,)
        if units.shape != expected or conditions.shape != expected:
            raise ValueError("Unit plan arrays must have matching rank-1 shapes.")
        if block_group_ids is None:
            blocks = jnp.zeros(expected, dtype=jnp.int32)
        else:
            blocks = integer_array("block_group_ids", block_group_ids, ndim=1)
            if blocks.shape != expected:
                raise ValueError(
                    f"block_group_ids must have shape {expected}; got {blocks.shape}."
                )
        included_ = boolean_array("included", included, expected, default=True)
        observation_host = np.asarray(observations)
        unit_host = np.asarray(units)
        condition_host = np.asarray(conditions)
        block_host = np.asarray(blocks)
        included_host = np.asarray(included_)
        if observation_host.size and (
            np.any(observation_host < 0)
            or np.any(observation_host >= lineage.entity_count)
            or np.any(unit_host < 0)
            or np.any(unit_host >= lineage.entity_count)
        ):
            raise ValueError("Unit plan indices must lie within the lineage.")
        kinds = np.asarray(lineage.entity_kinds)
        if np.any(kinds[observation_host[included_host]] != lineage.OBSERVATION):
            raise ValueError("Included observation_indices must reference observations.")
        allowed_units = np.isin(
            kinds[unit_host[included_host]],
            (lineage.SUBJECT, lineage.SPECIMEN, lineage.ALIQUOT),
        )
        if np.any(~allowed_units):
            raise ValueError(
                "Included unit_indices must reference subjects, specimens, or aliquots."
            )
        if np.any(condition_host[included_host] < 0) or np.any(
            block_host[included_host] < 0
        ):
            raise ValueError("Included condition and block IDs must be non-negative.")
        included_observations = observation_host[included_host]
        if np.unique(included_observations).size != included_observations.size:
            raise ValueError("Included observations must occur exactly once.")
        ancestors = _ancestor_sets(lineage)
        for observation, unit in zip(
            included_observations.tolist(),
            unit_host[included_host].tolist(),
            strict=True,
        ):
            if unit not in ancestors[observation]:
                raise ValueError(
                    "Every experimental unit must be an observation ancestor."
                )
        _validate_unit_assignments(
            unit_host[included_host], condition_host[included_host]
        )

        plan_id = content_id(
            "experimental_unit_plan",
            {"lineage_id": lineage.lineage_id},
            (observations, units, conditions, blocks, included_),
        )
        self.lineage = lineage
        self.observation_indices = observations
        self.unit_indices = units
        self.condition_ids = conditions
        self.block_group_ids = blocks
        self.included = included_
        self.capacity = capacity
        self.plan_id = plan_id


class ExchangeabilityPlan(StrictModule, NonTrainableState):
    """Restricted randomization groups over one validated experimental-unit plan."""

    experimental_units: ExperimentalUnitPlan
    exchangeability_group_ids: Array
    permutation_mask: Array
    exchangeability_id: str = eqx.field(static=True)

    def __init__(
        self,
        experimental_units: ExperimentalUnitPlan,
        exchangeability_group_ids: ArrayLike,
        /,
        *,
        permutation_mask: ArrayLike | None = None,
    ):
        if not isinstance(experimental_units, ExperimentalUnitPlan):
            raise TypeError("experimental_units must be an ExperimentalUnitPlan.")
        groups = integer_array(
            "exchangeability_group_ids", exchangeability_group_ids, ndim=1
        )
        shape = (experimental_units.capacity,)
        if groups.shape != shape:
            raise ValueError(
                f"exchangeability_group_ids must have shape {shape}; got {groups.shape}."
            )
        if permutation_mask is None:
            mask = jnp.asarray(experimental_units.included, dtype=bool)
        else:
            mask = boolean_array(
                "permutation_mask", permutation_mask, shape, default=False
            )
            if np.any(np.asarray(mask) & ~np.asarray(experimental_units.included)):
                raise ValueError(
                    "permutation_mask cannot include excluded experimental units."
                )
        group_host = np.asarray(groups)
        mask_host = np.asarray(mask)
        if np.any(group_host[mask_host] < 0):
            raise ValueError("Permutable rows require non-negative exchangeability IDs.")
        unit_host = np.asarray(experimental_units.unit_indices)[mask_host]
        block_host = np.asarray(experimental_units.block_group_ids)[mask_host]
        _validate_unit_assignments(unit_host, group_host[mask_host])
        _validate_nested_groups(block_host, group_host[mask_host])

        exchangeability_id = content_id(
            "exchangeability_plan",
            {"experimental_unit_plan_id": experimental_units.plan_id},
            (groups, mask),
        )
        self.experimental_units = experimental_units
        self.exchangeability_group_ids = groups
        self.permutation_mask = mask
        self.exchangeability_id = exchangeability_id


def _validate_nested_groups(parents: np.ndarray, children: np.ndarray, /) -> None:
    child_parent: dict[int, int] = {}
    for parent, child in zip(parents.tolist(), children.tolist(), strict=True):
        prior = child_parent.setdefault(int(child), int(parent))
        if prior != int(parent):
            raise ValueError("Each nested replicate/group ID must have one parent group.")


def _validate_unit_assignments(units: np.ndarray, assignments: np.ndarray, /) -> None:
    unit_assignment: dict[int, int] = {}
    for unit, assignment in zip(units.tolist(), assignments.tolist(), strict=True):
        prior = unit_assignment.setdefault(int(unit), int(assignment))
        if prior != int(assignment):
            raise ValueError("One experimental unit cannot have multiple assignments.")


def _ancestor_sets(lineage: BiospecimenLineage, /) -> list[set[int]]:
    parents: list[list[int]] = [[] for _ in range(lineage.entity_count)]
    valid = np.asarray(lineage.relation.valid)
    parent_indices = np.asarray(lineage.relation.source_indices)[valid]
    child_indices = np.asarray(lineage.relation.target_indices)[valid]
    for parent, child in zip(
        parent_indices.tolist(), child_indices.tolist(), strict=True
    ):
        parents[child].append(parent)
    ancestors: list[set[int]] = [set() for _ in range(lineage.entity_count)]
    order = np.argsort(np.asarray(lineage.entity_kinds), kind="stable")
    for node in order.tolist():
        for parent in parents[node]:
            ancestors[node].add(parent)
            ancestors[node].update(ancestors[parent])
    return ancestors
