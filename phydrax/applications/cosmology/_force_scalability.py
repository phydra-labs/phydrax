#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver._particle_gravity import (
    PeriodicEwaldEvidence,
    PeriodicEwaldForcePlan,
    PeriodicEwaldResult,
)
from ._closure import ScientificArtifactEnvelope


class MeshMatchedNearFieldGate(StrictModule, NonTrainableState):
    cutoff: float = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_relative_error: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        cutoff: float,
        maximum_pairs: int,
        maximum_relative_error: float,
    ):
        cutoff_ = float(cutoff)
        pairs = int(maximum_pairs)
        error = float(maximum_relative_error)
        if (
            not np.isfinite(cutoff_)
            or cutoff_ <= 0.0
            or pairs <= 0
            or not np.isfinite(error)
            or error <= 0.0
        ):
            raise ValueError("Near-field gate policy is invalid.")
        self.cutoff = cutoff_
        self.maximum_pairs = pairs
        self.maximum_relative_error = error

    def evaluate(
        self,
        positions: ArrayLike,
        reference_acceleration: ArrayLike,
        candidate_acceleration: ArrayLike,
        /,
    ) -> dict[str, Array]:
        position = jnp.asarray(positions)
        reference = jnp.asarray(reference_acceleration, dtype=position.dtype)
        candidate = jnp.asarray(candidate_acceleration, dtype=position.dtype)
        if reference.shape != position.shape or candidate.shape != position.shape:
            raise ValueError("Near-field gate acceleration shapes must match positions.")
        displacement = position[:, None, :] - position[None, :, :]
        distances = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
        pair_count = jnp.sum((distances > 0.0) & (distances <= self.cutoff)) // 2
        absolute = jnp.sqrt(jnp.sum((candidate - reference) ** 2, axis=-1))
        reference_norm = jnp.sqrt(jnp.sum(reference**2, axis=-1))
        relative = absolute / jnp.maximum(reference_norm, 1.0e-12)
        maximum_relative = jnp.max(relative)
        capacity_ok = pair_count <= self.maximum_pairs
        error_ok = maximum_relative <= self.maximum_relative_error
        return {
            "pair_count": pair_count,
            "maximum_relative_error": maximum_relative,
            "capacity_ok": capacity_ok,
            "error_ok": error_ok,
            "approved": capacity_ok & error_ok,
        }


class CosmologySnapshotProduct(StrictModule, NonTrainableState):
    particle_ids: Array
    positions: Array
    canonical_momenta: Array
    masses: Array
    scale_factor: Array
    box_size: tuple[float, ...] = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_ids: ArrayLike,
        positions: ArrayLike,
        canonical_momenta: ArrayLike,
        masses: ArrayLike,
        scale_factor: ArrayLike,
        box_size: tuple[float, ...],
        artifact: ScientificArtifactEnvelope,
        /,
    ):
        ids = jax.lax.stop_gradient(jnp.asarray(particle_ids))
        position = jax.lax.stop_gradient(jnp.asarray(positions))
        momentum = jax.lax.stop_gradient(
            jnp.asarray(canonical_momenta, dtype=position.dtype)
        )
        mass = jax.lax.stop_gradient(jnp.asarray(masses, dtype=position.dtype))
        scale = jax.lax.stop_gradient(jnp.asarray(scale_factor, dtype=position.dtype))
        if (
            ids.ndim != 1
            or position.shape != momentum.shape
            or position.shape[0] != ids.size
            or mass.shape != ids.shape
            or position.shape[1] != len(box_size)
            or scale.shape != ()
        ):
            raise ValueError("Cosmology snapshot shapes are inconsistent.")
        self.particle_ids = ids
        self.positions = position
        self.canonical_momenta = momentum
        self.masses = mass
        self.scale_factor = scale
        self.box_size = tuple(float(value) for value in box_size)
        self.artifact = artifact
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "cosmology-snapshot",
                "artifact": artifact.artifact_id,
                "arrays": array_tree_fingerprint((ids, position, momentum, mass, scale)),
            }
        )

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        /,
        *,
        id_dataset: str,
        position_dataset: str,
        velocity_dataset: str,
        mass_dataset: str,
        scale_factor: float,
        box_size: tuple[float, ...],
        artifact: ScientificArtifactEnvelope,
    ) -> CosmologySnapshotProduct:
        with h5py.File(Path(path), "r") as handle:
            ids = np.asarray(handle[id_dataset])
            positions = np.asarray(handle[position_dataset])
            velocities = np.asarray(handle[velocity_dataset])
            masses = np.asarray(handle[mass_dataset])
        canonical_momenta = masses[:, None] * scale_factor * velocities
        return cls(
            ids,
            positions,
            canonical_momenta,
            masses,
            scale_factor,
            box_size,
            artifact,
        )


class DistributedPMFeasibilityEvidence(StrictModule, NonTrainableState):
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    device_mesh_shape: tuple[int, ...] = eqx.field(static=True)
    particle_capacity_per_device: int = eqx.field(static=True)
    estimated_mesh_bytes_per_device: int = eqx.field(static=True)
    estimated_transpose_bytes: int = eqx.field(static=True)
    divisible: bool = eqx.field(static=True)
    feasible: bool = eqx.field(static=True)

    def __init__(
        self,
        mesh_shape: tuple[int, ...],
        device_mesh_shape: tuple[int, ...],
        particle_capacity_per_device: int,
        /,
        *,
        dtype_bytes: int = 8,
        arrays_per_device: int = 8,
        byte_budget_per_device: int,
    ):
        mesh = tuple(int(value) for value in mesh_shape)
        devices = tuple(int(value) for value in device_mesh_shape)
        capacity = int(particle_capacity_per_device)
        divisible = len(mesh) == len(devices) and all(
            count > 0 and parts > 0 and count % parts == 0
            for count, parts in zip(mesh, devices, strict=True)
        )
        local_cells = (
            int(
                np.prod(
                    tuple(
                        count // parts for count, parts in zip(mesh, devices, strict=True)
                    )
                )
            )
            if divisible
            else 0
        )
        mesh_bytes = local_cells * int(dtype_bytes) * int(arrays_per_device)
        transpose_bytes = local_cells * int(dtype_bytes) * 2
        feasible = (
            divisible
            and capacity > 0
            and mesh_bytes + transpose_bytes <= int(byte_budget_per_device)
        )
        self.mesh_shape = mesh
        self.device_mesh_shape = devices
        self.particle_capacity_per_device = capacity
        self.estimated_mesh_bytes_per_device = mesh_bytes
        self.estimated_transpose_bytes = transpose_bytes
        self.divisible = divisible
        self.feasible = feasible


__all__ = [
    "CosmologySnapshotProduct",
    "DistributedPMFeasibilityEvidence",
    "MeshMatchedNearFieldGate",
    "PeriodicEwaldEvidence",
    "PeriodicEwaldForcePlan",
    "PeriodicEwaldResult",
]
