#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._precision import FDExecutionPrecisionPolicy


_FD_CHECKPOINT_FORMAT = "phydrax-fd-checkpoint"
_FD_CHECKPOINT_VERSION = 2


class FDCheckpointPlan(StrictModule, NonTrainableState):
    """Exact compatibility identity for portable finite-difference restart state."""

    discretization_ids: tuple[str, ...] = eqx.field(static=True)
    boundary_program_id: str | None = eqx.field(static=True)
    amr_trace_id: str | None = eqx.field(static=True)
    partition_id: str | None = eqx.field(static=True)
    integrator_id: str = eqx.field(static=True)
    precision_contract_id: str = eqx.field(static=True)
    precision: FDExecutionPrecisionPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization_ids: Sequence[str],
        integrator_id: str,
        /,
        *,
        boundary_program_id: str | None = None,
        amr_trace_id: str | None = None,
        partition_id: str | None = None,
        precision: FDExecutionPrecisionPolicy | None = None,
    ):
        discretizations = tuple(str(value) for value in discretization_ids)
        integrator = str(integrator_id)
        precision_ = FDExecutionPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, FDExecutionPrecisionPolicy):
            raise TypeError("precision must be an FDExecutionPrecisionPolicy.")
        optional = tuple(
            None if value is None else str(value)
            for value in (boundary_program_id, amr_trace_id, partition_id)
        )
        if (
            not discretizations
            or any(not value for value in discretizations)
            or not integrator
            or not precision_.policy_id
            or any(value == "" for value in optional)
        ):
            raise ValueError("FD checkpoint identities must be non-empty when supplied.")
        self.discretization_ids = discretizations
        self.boundary_program_id = optional[0]
        self.amr_trace_id = optional[1]
        self.partition_id = optional[2]
        self.integrator_id = integrator
        self.precision_contract_id = precision_.policy_id
        self.precision = precision_
        self.plan_id = canonical_fingerprint(self.manifest_identity())

    def manifest_identity(self, /) -> dict[str, Any]:
        return {
            "discretization_ids": list(self.discretization_ids),
            "boundary_program_id": self.boundary_program_id,
            "amr_trace_id": self.amr_trace_id,
            "partition_id": self.partition_id,
            "integrator_id": self.integrator_id,
            "precision_contract_id": self.precision_contract_id,
        }


class FDCheckpoint(StrictModule):
    """Validated portable time, named fields, and auxiliary solver arrays."""

    time: Array
    field_names: tuple[str, ...] = eqx.field(static=True)
    fields: tuple[Array, ...]
    auxiliary_names: tuple[str, ...] = eqx.field(static=True)
    auxiliary: tuple[Array, ...]
    plan_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        time: ArrayLike,
        fields: Mapping[str, ArrayLike],
        auxiliary: Mapping[str, ArrayLike],
        plan_id: str,
        checkpoint_id: str,
        /,
    ):
        field_names = tuple(sorted(fields))
        auxiliary_names = tuple(sorted(auxiliary))
        if (
            not field_names
            or any(not _portable_name(value) for value in field_names + auxiliary_names)
            or not plan_id
            or not checkpoint_id
        ):
            raise ValueError("FD checkpoint fields and identities are invalid.")
        self.time = jnp.asarray(time)
        self.field_names = field_names
        self.fields = tuple(jnp.asarray(fields[name]) for name in field_names)
        self.auxiliary_names = auxiliary_names
        self.auxiliary = tuple(jnp.asarray(auxiliary[name]) for name in auxiliary_names)
        self.plan_id = str(plan_id)
        self.checkpoint_id = str(checkpoint_id)

    def field(self, name: str, /) -> Array:
        if name not in self.field_names:
            raise KeyError(f"Checkpoint has no field {name!r}.")
        return self.fields[self.field_names.index(name)]

    def auxiliary_value(self, name: str, /) -> Array:
        if name not in self.auxiliary_names:
            raise KeyError(f"Checkpoint has no auxiliary value {name!r}.")
        return self.auxiliary[self.auxiliary_names.index(name)]


def _portable_name(name: str, /) -> bool:
    return bool(name) and "/" not in name and "\\" not in name


def write_fd_checkpoint(
    path: str | os.PathLike[str],
    plan: FDCheckpointPlan,
    time: ArrayLike,
    fields: Mapping[str, ArrayLike],
    /,
    *,
    auxiliary: Mapping[str, ArrayLike] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    if not isinstance(plan, FDCheckpointPlan):
        raise TypeError("plan must be FDCheckpointPlan.")
    field_values = dict(fields)
    auxiliary_values = {} if auxiliary is None else dict(auxiliary)
    if not field_values or any(
        not _portable_name(str(value))
        for value in tuple(field_values) + tuple(auxiliary_values)
    ):
        raise ValueError(
            "FD checkpoint array names must be portable and fields non-empty."
        )
    expected_field_dtype = jnp.dtype(plan.precision.field_dtype)
    for name, value in field_values.items():
        array = jnp.asarray(value)
        if array.dtype != expected_field_dtype:
            raise TypeError(
                f"FD checkpoint field {name!r} has dtype {array.dtype}; "
                f"expected {expected_field_dtype}."
            )
    for name, value in auxiliary_values.items():
        array = jnp.asarray(value)
        if (
            jnp.issubdtype(array.dtype, jnp.inexact)
            and array.dtype != expected_field_dtype
        ):
            raise TypeError(
                f"FD checkpoint auxiliary {name!r} has dtype {array.dtype}; "
                f"expected {expected_field_dtype} for inexact state."
            )
    arrays = {f"field/{name}": value for name, value in field_values.items()}
    arrays.update(
        {f"auxiliary/{name}": value for name, value in auxiliary_values.items()}
    )
    arrays["time"] = np.asarray(time)
    metadata_ = {} if metadata is None else dict(metadata)
    checkpoint_id = canonical_fingerprint(
        {
            "kind": "fd-checkpoint",
            "plan": plan.plan_id,
            "field_names": sorted(field_values),
            "auxiliary_names": sorted(auxiliary_values),
            "metadata": metadata_,
        }
    )
    manifest = {
        "format": _FD_CHECKPOINT_FORMAT,
        "version": _FD_CHECKPOINT_VERSION,
        "plan_id": plan.plan_id,
        "identity": plan.manifest_identity(),
        "checkpoint_id": checkpoint_id,
        "field_names": sorted(field_values),
        "auxiliary_names": sorted(auxiliary_values),
        "metadata": metadata_,
    }
    return write_array_archive(path, manifest=manifest, arrays=arrays)


def read_fd_checkpoint(
    path: str | os.PathLike[str],
    expected_plan: FDCheckpointPlan,
    /,
) -> FDCheckpoint:
    if not isinstance(expected_plan, FDCheckpointPlan):
        raise TypeError("expected_plan must be FDCheckpointPlan.")
    manifest, arrays = read_array_archive(path)
    expected_keys = {
        "format",
        "version",
        "plan_id",
        "identity",
        "checkpoint_id",
        "field_names",
        "auxiliary_names",
        "metadata",
        "arrays",
    }
    if set(manifest) != expected_keys:
        raise ArrayArchiveCorruptionError("FD checkpoint manifest fields are invalid.")
    if (
        manifest["format"] != _FD_CHECKPOINT_FORMAT
        or manifest["version"] != _FD_CHECKPOINT_VERSION
    ):
        raise ArrayArchiveCorruptionError("Archive is not a supported FD checkpoint.")
    if (
        manifest["plan_id"] != expected_plan.plan_id
        or manifest["identity"] != expected_plan.manifest_identity()
    ):
        raise ValueError(
            "FD checkpoint is incompatible with the expected execution plan."
        )
    field_names = manifest["field_names"]
    auxiliary_names = manifest["auxiliary_names"]
    if not isinstance(field_names, list) or not isinstance(auxiliary_names, list):
        raise ArrayArchiveCorruptionError("FD checkpoint array name lists are invalid.")
    expected_arrays = {"time"}
    expected_arrays.update(f"field/{name}" for name in field_names)
    expected_arrays.update(f"auxiliary/{name}" for name in auxiliary_names)
    if set(arrays) != expected_arrays:
        raise ArrayArchiveCorruptionError("FD checkpoint array payload is incomplete.")
    return FDCheckpoint(
        arrays["time"],
        {name: arrays[f"field/{name}"] for name in field_names},
        {name: arrays[f"auxiliary/{name}"] for name in auxiliary_names},
        expected_plan.plan_id,
        str(manifest["checkpoint_id"]),
    )


__all__ = [
    "FDCheckpoint",
    "FDCheckpointPlan",
    "read_fd_checkpoint",
    "write_fd_checkpoint",
]
