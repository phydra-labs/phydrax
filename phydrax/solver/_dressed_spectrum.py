#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import HermitianSpectrum, MaterializationPolicy
from ..linalg.eigen import (
    HermitianEigenspaceTrackingPlan,
    HermitianEigenspaceTrackingPolicy,
    plan_hermitian_eigenspace_tracking,
    track_hermitian_eigenspaces,
)
from ..operators.quantum import DenseQuantumSubspace
from ._circuit_qed import PreparedCircuitQEDDevice
from ._local_hamiltonian import materialize_local_hamiltonian


class DressedStateLabel(StrictModule):
    """One product-state level tuple retained as a dressed-state identity."""

    levels: tuple[int, ...] = eqx.field(static=True)
    label_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[int],
        /,
        *,
        label_id: str | None = None,
    ):
        raw_levels = tuple(levels)
        if not raw_levels or any(
            isinstance(level, bool) or not isinstance(level, Integral)
            for level in raw_levels
        ):
            raise TypeError("levels must contain non-negative integers.")
        levels_ = tuple(int(level) for level in raw_levels)
        if any(level < 0 for level in levels_):
            raise ValueError("levels must contain non-negative integers.")
        identifier = (
            canonical_fingerprint(
                {"kind": "dressed-state-label", "levels": list(levels_)}
            )
            if label_id is None
            else str(label_id)
        )
        if not identifier:
            raise ValueError("label_id must be nonempty.")
        self.levels = levels_
        self.label_id = identifier


class DressedSpectrumPolicy(StrictModule):
    """Dense resource and correspondence policy for a dressed spectrum."""

    tracking: HermitianEigenspaceTrackingPolicy
    maximum_hilbert_dimension: int = eqx.field(static=True)
    maximum_dense_entries: int = eqx.field(static=True)
    maximum_dense_bytes: int = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    eigen_residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_hilbert_dimension: int = 4096,
        maximum_dense_entries: int = 1 << 24,
        maximum_dense_bytes: int = 1 << 29,
        hermiticity_tolerance: float = 1e-10,
        eigen_residual_tolerance: float = 1e-8,
        tracking: HermitianEigenspaceTrackingPolicy | None = None,
    ):
        for name, value in (
            ("maximum_hilbert_dimension", maximum_hilbert_dimension),
            ("maximum_dense_entries", maximum_dense_entries),
            ("maximum_dense_bytes", maximum_dense_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a positive integer.")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive.")
        if any(
            not isfinite(float(value)) or float(value) < 0.0
            for value in (hermiticity_tolerance, eigen_residual_tolerance)
        ):
            raise ValueError(
                "Dressed-spectrum tolerances must be finite and non-negative."
            )
        tracking_ = HermitianEigenspaceTrackingPolicy() if tracking is None else tracking
        if not isinstance(tracking_, HermitianEigenspaceTrackingPolicy):
            raise TypeError(
                "tracking must be a HermitianEigenspaceTrackingPolicy or None."
            )
        self.tracking = tracking_
        self.maximum_hilbert_dimension = int(maximum_hilbert_dimension)
        self.maximum_dense_entries = int(maximum_dense_entries)
        self.maximum_dense_bytes = int(maximum_dense_bytes)
        self.hermiticity_tolerance = float(hermiticity_tolerance)
        self.eigen_residual_tolerance = float(eigen_residual_tolerance)


class DressedSpectrumCostEstimate(StrictModule):
    """Dense eigensystem storage and work dimensions."""

    hilbert_dimension: int = eqx.field(static=True)
    label_count: int = eqx.field(static=True)
    dense_entries: int = eqx.field(static=True)
    dense_bytes: int = eqx.field(static=True)


class DressedSpectrumPlan(StrictModule):
    """Product-state reference and tracking plan for repeated dressed spectra."""

    labels: tuple[DressedStateLabel, ...]
    reference_vectors: Array
    tracking_plan: HermitianEigenspaceTrackingPlan
    policy: DressedSpectrumPolicy
    cost: DressedSpectrumCostEstimate
    device_plan_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class DressedSpectrumDiagnostics(StrictModule):
    """Dense eigensolve, assignment, and source-device evidence."""

    hermiticity_residual: Array
    eigen_residual: Array
    minimum_overlap: Array
    assignment_margin: Array
    device_valid: Array
    finite: Array
    valid: Array


class PreparedDressedSpectrum(StrictModule):
    """Dressed energies and vectors preserved in declared product-label order."""

    plan: DressedSpectrumPlan
    energies: Array
    vectors: Array
    full_eigenvalues: Array
    subspace: DenseQuantumSubspace
    diagnostics: DressedSpectrumDiagnostics
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def energy(self, label: DressedStateLabel | Sequence[int], /) -> Array:
        levels = label.levels if isinstance(label, DressedStateLabel) else tuple(label)
        for index, candidate in enumerate(self.plan.labels):
            if candidate.levels == tuple(levels):
                return self.energies[index]
        raise KeyError(f"Unknown dressed-state label {tuple(levels)!r}.")


def _flat_index(levels: tuple[int, ...], dimensions: tuple[int, ...], /) -> int:
    index = 0
    for level, dimension in zip(levels, dimensions, strict=True):
        index = index * dimension + level
    return index


def _reference_energies(
    device: PreparedCircuitQEDDevice,
    labels: tuple[DressedStateLabel, ...],
    /,
) -> Array:
    values = []
    for label in labels:
        energy = jnp.asarray(0.0, dtype=device.reductions[0].energies.dtype)
        for level, reduction in zip(label.levels, device.reductions, strict=True):
            energy = energy + reduction.energies[level]
        values.append(energy)
    return jnp.stack(values)


def plan_dressed_spectrum(
    device: PreparedCircuitQEDDevice,
    labels: Sequence[DressedStateLabel | Sequence[int]],
    policy: DressedSpectrumPolicy | None = None,
    /,
) -> DressedSpectrumPlan:
    """Plan bounded dense diagonalization and one-to-one product-state matching."""

    if not isinstance(device, PreparedCircuitQEDDevice):
        raise TypeError("device must be a PreparedCircuitQEDDevice.")
    selected_labels = tuple(
        label if isinstance(label, DressedStateLabel) else DressedStateLabel(label)
        for label in labels
    )
    if not selected_labels:
        raise ValueError("labels must be nonempty.")
    if len({label.levels for label in selected_labels}) != len(selected_labels):
        raise ValueError("Dressed-state labels must be unique.")
    dimensions = device.plan.layout.local_dimensions
    for label in selected_labels:
        if len(label.levels) != len(dimensions):
            raise ValueError("Every label must cover every device mode.")
        if any(
            level >= dimension
            for level, dimension in zip(label.levels, dimensions, strict=True)
        ):
            raise ValueError("A dressed-state label contains an out-of-range level.")
    selected = DressedSpectrumPolicy() if policy is None else policy
    if not isinstance(selected, DressedSpectrumPolicy):
        raise TypeError("policy must be a DressedSpectrumPolicy or None.")
    dimension = device.plan.layout.dimension
    if dimension > selected.maximum_hilbert_dimension:
        raise ValueError("Dressed spectrum exceeds maximum_hilbert_dimension.")
    entries = dimension * dimension
    itemsize = np.dtype(device.drift.terms[0].generator.dtype).itemsize
    byte_count = entries * itemsize
    if (
        entries > selected.maximum_dense_entries
        or byte_count > selected.maximum_dense_bytes
    ):
        raise ValueError("Dressed spectrum exceeds dense resource limits.")
    indices = tuple(_flat_index(label.levels, dimensions) for label in selected_labels)
    reference_vectors = jnp.eye(
        dimension,
        dtype=device.drift.terms[0].generator.dtype,
    )[:, jnp.asarray(indices, dtype=jnp.int32)]
    reference_energies = _reference_energies(device, selected_labels)
    tracking_plan = plan_hermitian_eigenspace_tracking(
        np.asarray(reference_energies),
        policy=selected.tracking,
    )
    cost = DressedSpectrumCostEstimate(
        dimension,
        len(selected_labels),
        entries,
        byte_count,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "dressed-spectrum-plan",
            "device": device.plan.plan_id,
            "labels": [label.label_id for label in selected_labels],
            "tracking": tracking_plan.plan_id,
            "limits": {
                "maximum_hilbert_dimension": selected.maximum_hilbert_dimension,
                "maximum_dense_entries": selected.maximum_dense_entries,
                "maximum_dense_bytes": selected.maximum_dense_bytes,
            },
        }
    )
    return DressedSpectrumPlan(
        selected_labels,
        reference_vectors,
        tracking_plan,
        selected,
        cost,
        device.plan.plan_id,
        plan_id,
    )


def _prepare_dressed_spectrum(
    device: PreparedCircuitQEDDevice,
    plan: DressedSpectrumPlan,
    /,
    *,
    numeric_version: ArrayLike,
) -> PreparedDressedSpectrum:
    if device.plan.plan_id != plan.device_plan_id:
        raise ValueError("Prepared device does not match the dressed-spectrum plan.")
    matrix = materialize_local_hamiltonian(
        device.drift,
        policy=MaterializationPolicy(
            max_entries=plan.policy.maximum_dense_entries,
            max_bytes=plan.policy.maximum_dense_bytes,
        ),
    )
    spectrum = HermitianSpectrum(
        matrix,
        tolerance=plan.policy.hermiticity_tolerance,
    )
    tracking = track_hermitian_eigenspaces(
        plan.tracking_plan,
        plan.reference_vectors,
        spectrum.eigenvalues,
        spectrum.eigenvectors,
    )
    residual = matrix @ tracking.vectors - tracking.vectors * tracking.values[None, :]
    eigen_residual = jnp.max(jnp.abs(residual))
    finite = (
        jnp.all(jnp.isfinite(matrix))
        & jnp.all(jnp.isfinite(tracking.values))
        & jnp.all(jnp.isfinite(tracking.vectors))
        & jnp.isfinite(eigen_residual)
    )
    valid = (
        device.diagnostics.valid
        & spectrum.valid
        & tracking.successful
        & finite
        & (eigen_residual <= plan.policy.eigen_residual_tolerance)
    )
    subspace = DenseQuantumSubspace(
        tracking.vectors,
        tolerance=plan.policy.tracking.orthogonality_tolerance,
        maximum_entries=plan.cost.hilbert_dimension * plan.cost.label_count,
        subspace_id=canonical_fingerprint(
            {"kind": "dressed-quantum-subspace", "plan": plan.plan_id}
        ),
    )
    diagnostics = DressedSpectrumDiagnostics(
        spectrum.hermiticity_residual,
        eigen_residual,
        jnp.min(tracking.diagnostics.cluster_minimum_overlaps),
        tracking.diagnostics.assignment_margin,
        device.diagnostics.valid,
        finite,
        valid & subspace.evidence.valid,
    )
    return PreparedDressedSpectrum(
        plan,
        tracking.values,
        tracking.vectors,
        spectrum.eigenvalues,
        subspace,
        diagnostics,
        jnp.asarray(numeric_version, dtype=jnp.int32),
        canonical_fingerprint(
            {"kind": "prepared-dressed-spectrum", "plan": plan.plan_id}
        ),
    )


def prepare_dressed_spectrum(
    device: PreparedCircuitQEDDevice,
    plan: DressedSpectrumPlan | None = None,
    /,
    *,
    labels: Sequence[DressedStateLabel | Sequence[int]] | None = None,
    policy: DressedSpectrumPolicy | None = None,
) -> PreparedDressedSpectrum:
    """Prepare a product-labelled dressed eigensystem."""

    if plan is None:
        if labels is None:
            raise ValueError("labels are required when plan is omitted.")
        selected = plan_dressed_spectrum(device, labels, policy)
    else:
        if labels is not None or policy is not None:
            raise ValueError("Specify plan or labels/policy, not both.")
        selected = plan
    return _prepare_dressed_spectrum(
        device,
        selected,
        numeric_version=jnp.asarray(0, dtype=jnp.int32),
    )


def refresh_dressed_spectrum(
    prepared: PreparedDressedSpectrum,
    device: PreparedCircuitQEDDevice,
    /,
) -> PreparedDressedSpectrum:
    """Refresh dressed energies while retaining the original product labels."""

    if not isinstance(prepared, PreparedDressedSpectrum):
        raise TypeError("prepared must be a PreparedDressedSpectrum.")
    return _prepare_dressed_spectrum(
        device,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def dressed_quantum_subspace(
    prepared: PreparedDressedSpectrum,
    labels: Sequence[DressedStateLabel | Sequence[int]],
    /,
) -> DenseQuantumSubspace:
    """Select an ordered dressed logical subspace from one prepared spectrum."""

    indices: list[int] = []
    for label in labels:
        levels = label.levels if isinstance(label, DressedStateLabel) else tuple(label)
        matches = [
            index
            for index, candidate in enumerate(prepared.plan.labels)
            if candidate.levels == tuple(levels)
        ]
        if not matches:
            raise KeyError(f"Unknown dressed-state label {tuple(levels)!r}.")
        indices.append(matches[0])
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("labels must select unique prepared dressed states.")
    return DenseQuantumSubspace(
        prepared.vectors[:, jnp.asarray(indices, dtype=jnp.int32)],
        tolerance=prepared.plan.policy.tracking.orthogonality_tolerance,
        subspace_id=canonical_fingerprint(
            {
                "kind": "selected-dressed-quantum-subspace",
                "plan": prepared.plan.plan_id,
                "indices": indices,
            }
        ),
    )


__all__ = [
    "DressedSpectrumCostEstimate",
    "DressedSpectrumDiagnostics",
    "DressedSpectrumPlan",
    "DressedSpectrumPolicy",
    "DressedStateLabel",
    "PreparedDressedSpectrum",
    "dressed_quantum_subspace",
    "plan_dressed_spectrum",
    "prepare_dressed_spectrum",
    "refresh_dressed_spectrum",
]
