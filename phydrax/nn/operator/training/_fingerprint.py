#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....graph import operator_topology_fingerprint
from ..data import FunctionSamples, OperatorBatch, OperatorTargetBatch
from ._dataset import OperatorDataset


_OPERATOR_DATASET_FINGERPRINT_FORMAT = "phydrax-operator-dataset-v1"


def _samples_schema(samples: FunctionSamples, /) -> dict[str, Any]:
    values = None
    if samples.values is not None:
        array = jnp.asarray(samples.values)
        values = {"shape": list(array.shape), "dtype": str(array.dtype)}
    coordinate_dim = None
    if samples.coordinates is not None:
        coordinate_dim = int(samples.coordinates.shape[-1])
    elif samples.axes:
        coordinate_dim = len(samples.axes)
    topology = None
    if samples.topology is not None:
        topology = {
            "kind": samples.topology.kind,
            "site": samples.topology.site,
            "entity": samples.topology.entity,
            "case_shape": list(samples.topology.case_shape),
            "sample_shape": list(samples.topology.sample_shape),
            "num_graphs": samples.topology.graph.num_graphs,
            "entity_count": samples.topology.entity_count,
            "edge_count": (
                int(samples.topology.graph.senders.shape[0])
                if samples.topology.graph.senders is not None
                else 0
            ),
            "fingerprint": operator_topology_fingerprint(samples.topology),
        }
    return {
        "values": values,
        "sample_shape": list(samples.sample_shape),
        "coordinate_dim": coordinate_dim,
        "geometry_case_shape": list(samples.geometry_case_shape),
        "axes": [
            {
                "name": axis.name,
                "size": axis.size,
                "basis": axis.basis,
                "periodic": axis.periodic,
            }
            for axis in samples.axes
        ],
        "has_quadrature": samples.quadrature_weights is not None
        or any(axis.quadrature_weights is not None for axis in samples.axes),
        "has_mask": samples.mask is not None,
        "topology": topology,
    }


def operator_batch_schema(
    batch: OperatorBatch,
    /,
    *,
    target: OperatorTargetBatch | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe compatibility contract for an operator batch."""
    schema = {
        "case_axes": list(batch.case_axes),
        "case_shape": list(batch.case_shape),
        "inputs": {
            name: _samples_schema(samples) for name, samples in batch.inputs.items()
        },
        "queries": {
            name: _samples_schema(samples) for name, samples in batch.queries.items()
        },
    }
    if target is not None:
        target.validate(batch)
        schema["targets"] = {
            name: {
                "shape": list(field.values.shape),
                "dtype": str(field.values.dtype),
                "query_name": field.query_name,
                "channels": field.spec.channels,
                "component_names": list(field.spec.component_names),
                **(
                    {}
                    if field.spec.classification is None
                    else {"classification": field.spec.classification.to_dict()}
                ),
            }
            for name, field in target.fields.items()
        }
    return schema


def operator_fit_schema(
    batch: OperatorBatch,
    /,
    *,
    target: OperatorTargetBatch,
) -> dict[str, Any]:
    """Return cardinality-independent model I/O semantics for fit checkpoints."""
    target.validate(batch)
    case_rank = len(batch.case_shape)

    def field_schema(samples: FunctionSamples) -> dict[str, Any]:
        value = None
        if samples.values is not None:
            array = jnp.asarray(samples.values)
            component_start = case_rank + len(samples.sample_shape)
            value = {
                "component_shape": list(array.shape[component_start:]),
                "dtype": str(array.dtype),
            }
        coordinate_dim = (
            int(samples.coordinates.shape[-1])
            if samples.coordinates is not None
            else len(samples.axes) or None
        )
        return {
            "values": value,
            "sample_rank": len(samples.sample_shape),
            "coordinate_dim": coordinate_dim,
            "has_coordinates": samples.coordinates is not None,
            "axes": [
                {
                    "name": axis.name,
                    "basis": axis.basis,
                    "periodic": axis.periodic,
                }
                for axis in samples.axes
            ],
            "has_quadrature": samples.quadrature_weights is not None
            or any(axis.quadrature_weights is not None for axis in samples.axes),
            "has_mask": samples.mask is not None,
            "topology": (
                None
                if samples.topology is None
                else {
                    "kind": samples.topology.kind,
                    "site": samples.topology.site,
                    "entity": samples.topology.entity,
                    "sample_rank": len(samples.topology.sample_shape),
                }
            ),
        }

    return {
        "case_axes": list(batch.case_axes),
        "inputs": {name: field_schema(samples) for name, samples in batch.inputs.items()},
        "queries": {
            name: field_schema(samples) for name, samples in batch.queries.items()
        },
        "targets": {
            name: {
                "dtype": str(field.values.dtype),
                "query_name": field.query_name,
                "channels": field.spec.channels,
                "component_names": list(field.spec.component_names),
                **(
                    {}
                    if field.spec.classification is None
                    else {"classification": field.spec.classification.to_dict()}
                ),
            }
            for name, field in target.fields.items()
        },
    }


def operator_dataset_fingerprint(dataset: OperatorDataset, /) -> str:
    """Hash one immutable operator dataset's batches, case metadata, and provenance."""
    if not isinstance(dataset, OperatorDataset):
        raise TypeError("dataset must be an OperatorDataset.")
    assert dataset.provenance is not None
    payload = {
        "format": _OPERATOR_DATASET_FINGERPRINT_FORMAT,
        "schema": operator_batch_schema(dataset.batch, target=dataset.targets),
        "arrays": array_tree_fingerprint(
            {
                "batch": dataset.batch,
                "targets": dataset.targets,
                "case_log_weights": dataset.case_log_weights,
                "case_mask": dataset.case_mask,
            }
        ),
        "provenance": [
            {
                "case_id": record.case_id,
                "identities": dict(record.identities),
                "order": dict(record.order),
            }
            for record in dataset.provenance
        ],
    }
    return f"sha256:{canonical_fingerprint(payload)}"


__all__ = ["operator_batch_schema", "operator_dataset_fingerprint"]
