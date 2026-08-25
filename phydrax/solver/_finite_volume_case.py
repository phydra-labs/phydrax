#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._fv_precision import (
    FiniteVolumePrecisionPolicy,
    PrecisionDType,
)
from ..discretization.finite_volume import UnstructuredFiniteVolumeDiscretization
from ._finite_volume_runtime import (
    FiniteVolumeStepPolicy,
    PreparedFiniteVolumeRuntime,
)


class FiniteVolumeExecutionSpec(StrictModule, NonTrainableState):
    end_time: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    step_policy: FiniteVolumeStepPolicy
    execution_id: str = eqx.field(static=True)

    def __init__(
        self,
        end_time: float,
        maximum_steps: int,
        /,
        *,
        step_policy: FiniteVolumeStepPolicy | None = None,
    ):
        end = float(end_time)
        steps = int(maximum_steps)
        policy = FiniteVolumeStepPolicy() if step_policy is None else step_policy
        if not np.isfinite(end) or end <= 0.0 or steps <= 0:
            raise ValueError("Execution end time and maximum steps must be positive.")
        if not isinstance(policy, FiniteVolumeStepPolicy):
            raise TypeError("step_policy must be FiniteVolumeStepPolicy.")
        self.end_time = end
        self.maximum_steps = steps
        self.step_policy = policy
        self.execution_id = canonical_fingerprint(
            {
                "kind": "finite-volume-execution",
                "end_time": end,
                "maximum_steps": steps,
                "step_policy": policy.policy_id,
            }
        )


class FiniteVolumeCaseSpec(StrictModule, NonTrainableState):
    """Versioned normalized identity for one prepared FV simulation."""

    schema_version: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    mesh_kind: str = eqx.field(static=True)
    mesh_topology_id: str = eqx.field(static=True)
    mesh_geometry_id: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    mesh_vertices: Array
    vertex_global_ids: Array
    cell_global_ids: Array
    precision: FiniteVolumePrecisionPolicy
    execution: FiniteVolumeExecutionSpec
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        runtime: PreparedFiniteVolumeRuntime,
        execution: FiniteVolumeExecutionSpec,
        /,
        *,
        precision: FiniteVolumePrecisionPolicy | None = None,
        schema_version: int = 2,
    ):
        name_ = str(name)
        version = int(schema_version)
        if not name_ or version != 2:
            raise ValueError(
                "Finite-volume case name must be non-empty and schema version 2."
            )
        if not isinstance(runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("runtime must be PreparedFiniteVolumeRuntime.")
        precision_ = runtime.precision if precision is None else precision
        if not isinstance(execution, FiniteVolumeExecutionSpec):
            raise TypeError("execution must be FiniteVolumeExecutionSpec.")
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy.")
        if precision_.policy_id != runtime.precision.policy_id:
            raise ValueError(
                "Finite-volume case precision must match the prepared runtime."
            )
        dynamics = runtime.dynamics
        discretization = dynamics.discretization
        if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            mesh_kind = "unstructured"
            topology_id = discretization.topology_id
            geometry_id = discretization.geometry_id
            vertices = discretization.vertices
            vertex_ids = discretization.vertex_global_ids
            cell_ids = discretization.cell_global_ids
        else:
            mesh_kind = "structured"
            topology_id = discretization.prepared_id
            geometry_id = discretization.prepared_id
            vertices = jnp.empty((0, 0))
            vertex_ids = jnp.empty((0,), dtype=jnp.int64)
            cell_ids = jnp.empty((0,), dtype=jnp.int64)
        self.schema_version = version
        self.name = name_
        self.runtime_id = runtime.runtime_id
        self.system_id = dynamics.system.system_id
        self.discretization_id = discretization.prepared_id
        self.method_id = dynamics.method.method_id
        self.boundary_id = dynamics.boundaries.boundary_set_id
        self.mesh_kind = mesh_kind
        self.mesh_topology_id = topology_id
        self.mesh_geometry_id = geometry_id
        self.state_shape = discretization.state_shape
        self.mesh_vertices = jnp.asarray(vertices)
        self.vertex_global_ids = jnp.asarray(vertex_ids)
        self.cell_global_ids = jnp.asarray(cell_ids)
        self.precision = precision_
        self.execution = execution
        self.case_id = canonical_fingerprint(self.to_dict(include_case_id=False))

    def to_dict(self, /, *, include_case_id: bool = True) -> dict[str, Any]:
        output = {
            "schema_version": self.schema_version,
            "name": self.name,
            "runtime_id": self.runtime_id,
            "system_id": self.system_id,
            "discretization_id": self.discretization_id,
            "method_id": self.method_id,
            "boundary_id": self.boundary_id,
            "mesh": {
                "kind": self.mesh_kind,
                "topology_id": self.mesh_topology_id,
                "geometry_id": self.mesh_geometry_id,
                "vertex_count": int(self.mesh_vertices.shape[0]),
                "cell_count": int(self.state_shape[0]),
            },
            "state_shape": list(self.state_shape),
            "precision": {
                "storage_dtype": self.precision.storage_dtype,
                "reconstruction_dtype": self.precision.reconstruction_dtype,
                "flux_dtype": self.precision.flux_dtype,
                "reduction_dtype": self.precision.reduction_dtype,
                "output_dtype": self.precision.output_dtype,
                "checkpoint_dtype": self.precision.checkpoint_dtype,
            },
            "execution": {
                "end_time": self.execution.end_time,
                "maximum_steps": self.execution.maximum_steps,
                "step_policy_id": self.execution.step_policy.policy_id,
            },
        }
        if include_case_id:
            output["case_id"] = self.case_id
        return output

    @staticmethod
    def validate_dict(payload: dict[str, Any], /) -> None:
        required = {
            "schema_version",
            "name",
            "runtime_id",
            "system_id",
            "discretization_id",
            "method_id",
            "boundary_id",
            "mesh",
            "state_shape",
            "precision",
            "execution",
            "case_id",
        }
        unknown = set(payload).difference(required)
        missing = required.difference(payload)
        if unknown or missing:
            raise ValueError(
                f"Finite-volume case schema has unknown={sorted(unknown)!r}, "
                f"missing={sorted(missing)!r}."
            )
        if payload["schema_version"] != 2:
            raise ValueError("Unsupported finite-volume case schema version.")
        mesh_keys = {
            "kind",
            "topology_id",
            "geometry_id",
            "vertex_count",
            "cell_count",
        }
        if not isinstance(payload["mesh"], dict) or set(payload["mesh"]) != mesh_keys:
            raise ValueError("Finite-volume mesh identity schema fields changed.")
        if payload["mesh"]["kind"] not in ("structured", "unstructured"):
            raise ValueError("Finite-volume mesh identity kind is invalid.")
        if (
            not isinstance(payload["state_shape"], list)
            or not payload["state_shape"]
            or any(int(value) <= 0 for value in payload["state_shape"])
        ):
            raise ValueError("Finite-volume state_shape is invalid.")

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        runtime: PreparedFiniteVolumeRuntime,
        execution: FiniteVolumeExecutionSpec,
        /,
    ) -> "FiniteVolumeCaseSpec":
        cls.validate_dict(payload)
        precision_payload = payload["precision"]
        precision_keys = {
            "storage_dtype",
            "reconstruction_dtype",
            "flux_dtype",
            "reduction_dtype",
            "output_dtype",
            "checkpoint_dtype",
        }
        if set(precision_payload) != precision_keys:
            raise ValueError("Finite-volume precision schema fields changed.")
        precision = FiniteVolumePrecisionPolicy(
            precision_payload["storage_dtype"],
            reconstruction_dtype=precision_payload["reconstruction_dtype"],
            flux_dtype=precision_payload["flux_dtype"],
            reduction_dtype=precision_payload["reduction_dtype"],
            output_dtype=precision_payload["output_dtype"],
            checkpoint_dtype=precision_payload["checkpoint_dtype"],
        )
        if payload["execution"] != {
            "end_time": execution.end_time,
            "maximum_steps": execution.maximum_steps,
            "step_policy_id": execution.step_policy.policy_id,
        }:
            raise ValueError("Finite-volume execution schema is incompatible.")
        case = cls(
            payload["name"],
            runtime,
            execution,
            precision=precision,
            schema_version=payload["schema_version"],
        )
        if case.to_dict() != payload:
            raise ValueError(
                "Finite-volume case identities do not match prepared runtime."
            )
        return case


__all__ = [
    "FiniteVolumeCaseSpec",
    "FiniteVolumeExecutionSpec",
    "FiniteVolumePrecisionPolicy",
    "PrecisionDType",
]
