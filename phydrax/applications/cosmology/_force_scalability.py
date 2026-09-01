#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import ScientificArtifactEnvelope


class PeriodicEwaldEvidence(StrictModule):
    real_space_acceleration: Array
    reciprocal_acceleration: Array
    net_force: Array
    finite: Array
    successful: Array


class PeriodicEwaldResult(StrictModule):
    acceleration: Array
    evidence: PeriodicEwaldEvidence
    successful: Array


class PeriodicEwaldForcePlan(StrictModule, NonTrainableState):
    """Small-N softened-neutral periodic Ewald acceleration reference."""

    box_size: tuple[float, ...] = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    real_offsets: Array
    wavevectors: Array
    volume: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        box_size: tuple[float, ...],
        gravitational_constant: float,
        /,
        *,
        softening: float,
        alpha: float,
        real_shells: int = 2,
        reciprocal_modes: int = 4,
    ):
        lengths = tuple(float(value) for value in box_size)
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        alpha_ = float(alpha)
        real = int(real_shells)
        reciprocal = int(reciprocal_modes)
        if (
            not lengths
            or any(not np.isfinite(value) or value <= 0.0 for value in lengths)
            or not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not np.isfinite(alpha_)
            or alpha_ <= 0.0
            or real < 0
            or reciprocal < 1
        ):
            raise ValueError("Periodic Ewald policy is invalid.")
        dimension = len(lengths)
        integer_offsets = np.asarray(
            tuple(product(range(-real, real + 1), repeat=dimension)), dtype=float
        )
        reciprocal_indices = np.asarray(
            tuple(
                index
                for index in product(range(-reciprocal, reciprocal + 1), repeat=dimension)
                if any(value != 0 for value in index)
            ),
            dtype=float,
        )
        wavevectors = 2.0 * np.pi * reciprocal_indices / np.asarray(lengths)[None, :]
        self.box_size = lengths
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.alpha = alpha_
        self.real_offsets = jnp.asarray(integer_offsets * np.asarray(lengths)[None, :])
        self.wavevectors = jnp.asarray(wavevectors)
        self.volume = float(np.prod(lengths))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-force",
                "box_size": list(lengths),
                "gravitational_constant": gravity,
                "softening": epsilon,
                "alpha": alpha_,
                "real_shells": real,
                "reciprocal_modes": reciprocal,
            }
        )

    def evaluate(self, positions: ArrayLike, masses: ArrayLike, /) -> PeriodicEwaldResult:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        if (
            position.ndim != 2
            or position.shape[1] != len(self.box_size)
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError("Periodic Ewald positions/masses have incompatible shapes.")
        position = eqx.error_if(
            position,
            jnp.any(~jnp.isfinite(position))
            | jnp.any(~jnp.isfinite(mass))
            | jnp.any(mass <= 0.0),
            "Periodic Ewald inputs must be finite with positive masses.",
        )
        target = position[:, None, None, :]
        source = position[None, :, None, :] + self.real_offsets[None, None, :, :]
        displacement = source - target
        distance_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        distance = jnp.sqrt(distance_squared)
        zero_offset = jnp.all(self.real_offsets == 0.0, axis=-1)
        self_pair = (
            jnp.eye(position.shape[0], dtype=bool)[:, :, None]
            & zero_offset[None, None, :]
        )
        screening = jax.scipy.special.erfc(self.alpha * distance) + (
            2.0
            * self.alpha
            * distance
            / jnp.sqrt(jnp.pi)
            * jnp.exp(-((self.alpha * distance) ** 2))
        )
        inverse_cube = jnp.where(self_pair, 0.0, screening / distance**3)
        real_acceleration = jnp.sum(
            self.gravitational_constant
            * mass[None, :, None, None]
            * displacement
            * inverse_cube[..., None],
            axis=(1, 2),
        )
        k = self.wavevectors.astype(position.dtype)
        k_squared = jnp.sum(k**2, axis=-1)
        source_phase = contract("kd,nd->kn", k, position)
        density_real = contract("n,kn->k", mass, jnp.cos(source_phase))
        density_imag = -contract("n,kn->k", mass, jnp.sin(source_phase))
        target_phase = source_phase.T
        real_product = -density_real[None, :] * jnp.sin(target_phase) - density_imag[
            None, :
        ] * jnp.cos(target_phase)
        coefficient = (
            4.0
            * jnp.pi
            * self.gravitational_constant
            / self.volume
            * jnp.exp(-k_squared / (4.0 * self.alpha**2))
            / k_squared
        )
        reciprocal_acceleration = contract("k,nk,kd->nd", coefficient, real_product, k)
        acceleration = real_acceleration + reciprocal_acceleration
        net_force = jnp.sum(mass[:, None] * acceleration, axis=0)
        finite = jnp.all(jnp.isfinite(acceleration))
        evidence = PeriodicEwaldEvidence(
            real_acceleration,
            reciprocal_acceleration,
            net_force,
            finite,
            finite,
        )
        return PeriodicEwaldResult(acceleration, evidence, finite)


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
