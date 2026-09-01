#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..splatting import ParticleGridSplatState, PreparedParticleGridSplat
from ._contact_kway import KWayMPMContactPlan, MPMContactGraph, MPMKWayContactResult


class MPMDeterminismMode(IntEnum):
    FAST = 0
    DETERMINISTIC = 1
    COMPENSATED = 2


class MPMKernelRealization(IntEnum):
    REFERENCE = 0
    FUSED_JAX = 1
    CUSTOM_ACCELERATOR = 2


class MPMExecutionPlan(StrictModule, NonTrainableState):
    backend: str = eqx.field(static=True)
    device_mesh: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    determinism: MPMDeterminismMode = eqx.field(static=True)
    realization: MPMKernelRealization = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    grid_capacity: int = eqx.field(static=True)
    route_capacity: int = eqx.field(static=True)
    field_capacity: int = eqx.field(static=True)
    block_capacity: int = eqx.field(static=True)
    contact_pair_capacity: int = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        backend: str,
        device_mesh: str,
        precision_policy_id: str,
        determinism: MPMDeterminismMode,
        realization: MPMKernelRealization,
        particle_capacity: int,
        grid_capacity: int,
        route_capacity: int,
        field_capacity: int,
        block_capacity: int,
        contact_pair_capacity: int,
    ):
        identifiers = (str(backend), str(device_mesh), str(precision_policy_id))
        capacities = tuple(
            int(value)
            for value in (
                particle_capacity,
                grid_capacity,
                route_capacity,
                field_capacity,
                block_capacity,
                contact_pair_capacity,
            )
        )
        if any(not value for value in identifiers) or any(
            value <= 0 for value in capacities
        ):
            raise ValueError("MPM execution identity/capacities must be positive.")
        self.backend, self.device_mesh, self.precision_policy_id = identifiers
        self.determinism = MPMDeterminismMode(determinism)
        self.realization = MPMKernelRealization(realization)
        (
            self.particle_capacity,
            self.grid_capacity,
            self.route_capacity,
            self.field_capacity,
            self.block_capacity,
            self.contact_pair_capacity,
        ) = capacities
        self.execution_id = canonical_fingerprint(
            {
                "kind": "mpm-execution-plan",
                "backend": self.backend,
                "device_mesh": self.device_mesh,
                "precision_policy_id": self.precision_policy_id,
                "determinism": int(self.determinism),
                "realization": int(self.realization),
                "capacities": capacities,
            }
        )

    def admit(
        self,
        *,
        particles: int,
        grid_nodes: int,
        routes: int,
        fields: int,
        blocks: int,
        contact_pairs: int,
    ):
        requested = (particles, grid_nodes, routes, fields, blocks, contact_pairs)
        admitted = (
            self.particle_capacity,
            self.grid_capacity,
            self.route_capacity,
            self.field_capacity,
            self.block_capacity,
            self.contact_pair_capacity,
        )
        if any(
            int(value) > limit for value, limit in zip(requested, admitted, strict=True)
        ):
            raise ValueError("MPM execution capacity envelope exceeded.")
        return self.execution_id


class MPMCapacityCertificate(StrictModule, NonTrainableState):
    execution_id: str = eqx.field(static=True)
    source_commit: str = eqx.field(static=True)
    toolchain: str = eqx.field(static=True)
    hardware: str = eqx.field(static=True)
    cold_compile_seconds: float = eqx.field(static=True)
    peak_memory_bytes: int = eqx.field(static=True)
    routes_per_second: float = eqx.field(static=True)
    step_seconds_p95: float = eqx.field(static=True)
    gradient_seconds_p95: float = eqx.field(static=True)
    checkpoint_bytes_per_second: float = eqx.field(static=True)
    numerical_defect_p99: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(self, execution: MPMExecutionPlan, /, **metrics):
        if not isinstance(execution, MPMExecutionPlan):
            raise TypeError("execution must be MPMExecutionPlan.")
        required = (
            "source_commit",
            "toolchain",
            "hardware",
            "cold_compile_seconds",
            "peak_memory_bytes",
            "routes_per_second",
            "step_seconds_p95",
            "gradient_seconds_p95",
            "checkpoint_bytes_per_second",
            "numerical_defect_p99",
        )
        if set(metrics) != set(required):
            raise ValueError("Capacity certificate metric inventory changed.")
        text = tuple(str(metrics[name]) for name in required[:3])
        numeric = tuple(
            float(metrics[name]) for name in required[3:] if name != "peak_memory_bytes"
        )
        memory = int(metrics["peak_memory_bytes"])
        if (
            any(not value for value in text)
            or memory <= 0
            or any(not np.isfinite(value) or value < 0.0 for value in numeric)
        ):
            raise ValueError("Capacity certificate metrics are invalid.")
        self.execution_id = execution.execution_id
        for name in required:
            setattr(self, name, memory if name == "peak_memory_bytes" else metrics[name])
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "mpm-capacity-certificate",
                "execution": execution.execution_id,
                **metrics,
            }
        )


def deterministic_global_sum(values: ArrayLike, /, *, compensated: bool = True) -> Array:
    value = jnp.asarray(values)
    flattened = value.reshape((-1,) + value.shape[1:])
    return (
        compensated_sum(flattened, axis=0) if compensated else jnp.sum(flattened, axis=0)
    )


def fused_route_reduction(
    prepared: PreparedParticleGridSplat,
    routes: ParticleGridSplatState,
    mass_payload: ArrayLike,
    vector_payload: ArrayLike,
    /,
):
    mass = prepared.scatter_route_payload(routes, mass_payload)
    vector = prepared.scatter_route_payload(routes, vector_payload)
    return mass, vector


def fused_contact_projection(
    plan: KWayMPMContactPlan,
    mass: ArrayLike,
    velocity: ArrayLike,
    graph: MPMContactGraph,
    step_size: ArrayLike,
    /,
    *,
    essential_mask: ArrayLike | None = None,
    essential_values: ArrayLike | None = None,
) -> MPMKWayContactResult:
    return plan.solve(
        mass,
        velocity,
        graph,
        step_size,
        essential_mask=essential_mask,
        essential_values=essential_values,
    )


__all__ = [
    "MPMCapacityCertificate",
    "MPMDeterminismMode",
    "MPMExecutionPlan",
    "MPMKernelRealization",
    "deterministic_global_sum",
    "fused_contact_projection",
    "fused_route_reduction",
]
