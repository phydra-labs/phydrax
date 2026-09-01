#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ._validation import (
    boolean_array,
    confidence_array,
    content_id,
    integer_array,
    labels_tuple,
    nonempty_string,
    string_tuple,
)


class FeatureDictionary(StrictModule, NonTrainableState):
    """Versioned numeric feature identity space with fixed-capacity validity data."""

    feature_ids: Array
    active: Array
    confidence: Array
    namespace: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    species: str = eqx.field(static=True)
    reference: str = eqx.field(static=True)
    annotation: str = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    dictionary_id: str = eqx.field(static=True)

    def __init__(
        self,
        feature_ids: ArrayLike,
        /,
        *,
        namespace: str,
        version: str,
        species: str,
        reference: str,
        annotation: str,
        labels: Sequence[str] | None = None,
        active: ArrayLike | None = None,
        confidence: ArrayLike | None = None,
    ):
        ids = integer_array("feature_ids", feature_ids, ndim=1)
        capacity = int(ids.shape[0])
        active_ = boolean_array("active", active, (capacity,), default=True)
        confidence_ = confidence_array("confidence", confidence, (capacity,))
        ids_host = np.asarray(ids)
        active_host = np.asarray(active_)
        active_ids = ids_host[active_host]
        if np.any(active_ids < 0):
            raise ValueError("Active feature IDs must be non-negative.")
        if np.unique(active_ids).size != active_ids.size:
            raise ValueError("Active feature IDs must be unique.")

        namespace_ = nonempty_string("namespace", namespace)
        version_ = nonempty_string("version", version)
        species_ = nonempty_string("species", species)
        reference_ = nonempty_string("reference", reference)
        annotation_ = nonempty_string("annotation", annotation)
        labels_ = labels_tuple("labels", labels, capacity)
        dictionary_id = content_id(
            "feature_dictionary",
            {
                "annotation": annotation_,
                "labels": labels_,
                "namespace": namespace_,
                "reference": reference_,
                "species": species_,
                "version": version_,
            },
            (ids, active_, confidence_),
        )
        self.feature_ids = ids
        self.active = active_
        self.confidence = confidence_
        self.namespace = namespace_
        self.version = version_
        self.species = species_
        self.reference = reference_
        self.annotation = annotation_
        self.labels = labels_
        self.capacity = capacity
        self.dictionary_id = dictionary_id


class FeatureMapping(StrictModule, NonTrainableState):
    """Confidence-weighted, one-to-many routes between two feature dictionaries."""

    source: FeatureDictionary
    target: FeatureDictionary
    relation: EdgeRelation
    confidence: Array
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: FeatureDictionary,
        target: FeatureDictionary,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        confidence: ArrayLike | None = None,
    ):
        if not isinstance(source, FeatureDictionary):
            raise TypeError("source must be a FeatureDictionary.")
        if not isinstance(target, FeatureDictionary):
            raise TypeError("target must be a FeatureDictionary.")
        source_ = integer_array("source_indices", source_indices, ndim=1)
        target_ = integer_array("target_indices", target_indices, ndim=1)
        if source_.shape != target_.shape:
            raise ValueError("Mapping source and target index shapes must match.")
        route_shape = (int(source_.shape[0]),)
        valid_ = boolean_array("valid", valid, route_shape, default=True)
        confidence_ = confidence_array("confidence", confidence, route_shape)
        source_host = np.asarray(source_)
        target_host = np.asarray(target_)
        valid_host = np.asarray(valid_)
        if source_host.size and (
            np.any(source_host < 0)
            or np.any(source_host >= source.capacity)
            or np.any(target_host < 0)
            or np.any(target_host >= target.capacity)
        ):
            raise ValueError("Mapping indices must lie within their dictionaries.")
        if np.any(valid_host):
            valid_source = source_host[valid_host]
            valid_target = target_host[valid_host]
            if np.any(~np.asarray(source.active)[valid_source]) or np.any(
                ~np.asarray(target.active)[valid_target]
            ):
                raise ValueError("Valid mapping routes must reference active features.")
            pairs = np.stack((valid_source, valid_target), axis=1)
            if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
                raise ValueError("Valid feature mapping routes must be unique.")

        relation = EdgeRelation(
            source_,
            target_,
            source_size=source.capacity,
            target_size=target.capacity,
            valid=valid_,
        )
        mapping_id = content_id(
            "feature_mapping",
            {
                "source_dictionary_id": source.dictionary_id,
                "target_dictionary_id": target.dictionary_id,
            },
            (relation, confidence_),
        )
        self.source = source
        self.target = target
        self.relation = relation
        self.confidence = confidence_
        self.mapping_id = mapping_id


class OntologyGraph(StrictModule, NonTrainableState):
    """A validated ontology DAG represented only by typed child-to-parent edges."""

    features: FeatureDictionary
    relation: EdgeRelation
    relation_codes: Array
    confidence: Array
    relation_names: tuple[str, ...] = eqx.field(static=True)
    ontology_id: str = eqx.field(static=True)

    def __init__(
        self,
        features: FeatureDictionary,
        child_indices: ArrayLike,
        parent_indices: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        relation_codes: ArrayLike | None = None,
        relation_names: Sequence[str] = ("is_a",),
        confidence: ArrayLike | None = None,
    ):
        if not isinstance(features, FeatureDictionary):
            raise TypeError("features must be a FeatureDictionary.")
        children = integer_array("child_indices", child_indices, ndim=1)
        parents = integer_array("parent_indices", parent_indices, ndim=1)
        if children.shape != parents.shape:
            raise ValueError("Ontology child and parent index shapes must match.")
        edge_shape = (int(children.shape[0]),)
        valid_ = boolean_array("valid", valid, edge_shape, default=True)
        names = string_tuple("relation_names", relation_names, allow_empty=False)
        if relation_codes is None:
            codes = jnp.zeros(edge_shape, dtype=jnp.int32)
        else:
            codes = integer_array("relation_codes", relation_codes, ndim=1)
            if codes.shape != edge_shape:
                raise ValueError(
                    f"relation_codes must have shape {edge_shape}; got {codes.shape}."
                )
        confidence_ = confidence_array("confidence", confidence, edge_shape)
        child_host = np.asarray(children)
        parent_host = np.asarray(parents)
        valid_host = np.asarray(valid_)
        code_host = np.asarray(codes)
        if child_host.size and (
            np.any(child_host < 0)
            or np.any(child_host >= features.capacity)
            or np.any(parent_host < 0)
            or np.any(parent_host >= features.capacity)
        ):
            raise ValueError("Ontology indices must lie within the feature dictionary.")
        valid_children = child_host[valid_host]
        valid_parents = parent_host[valid_host]
        valid_codes = code_host[valid_host]
        if valid_codes.size and (
            np.any(valid_codes < 0) or np.any(valid_codes >= len(names))
        ):
            raise ValueError("Valid ontology relation codes are out of range.")
        active_host = np.asarray(features.active)
        if valid_children.size and (
            np.any(~active_host[valid_children]) or np.any(~active_host[valid_parents])
        ):
            raise ValueError("Valid ontology edges must reference active features.")
        if valid_children.size:
            pairs = np.stack((valid_children, valid_parents), axis=1)
            if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
                raise ValueError("Valid ontology edges must be unique.")
        _validate_acyclic(features.capacity, valid_children, valid_parents)

        relation = EdgeRelation(
            children,
            parents,
            source_size=features.capacity,
            target_size=features.capacity,
            valid=valid_,
        )
        ontology_id = content_id(
            "ontology_graph",
            {
                "feature_dictionary_id": features.dictionary_id,
                "relation_names": names,
            },
            (relation, codes, confidence_),
        )
        self.features = features
        self.relation = relation
        self.relation_codes = codes
        self.confidence = confidence_
        self.relation_names = names
        self.ontology_id = ontology_id


def _validate_acyclic(
    node_count: int,
    sources: np.ndarray[Any, Any],
    targets: np.ndarray[Any, Any],
    /,
) -> None:
    successors: list[list[int]] = [[] for _ in range(node_count)]
    indegree = [0] * node_count
    for source, target in zip(sources.tolist(), targets.tolist(), strict=True):
        successors[source].append(target)
        indegree[target] += 1
    frontier = [node for node, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for target in successors[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
    if visited != node_count:
        raise ValueError("Ontology edges must be acyclic.")
