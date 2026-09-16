#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit tensor-structure crossing matrices and independent algebra audits."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._data import ConformalDataPlan


class CrossingSectorPlan(StrictModule):
    """Caller-supplied crossing matrices in one explicit tensor basis gauge."""

    data: ConformalDataPlan
    matrices: Array
    component_labels: tuple[str, ...] = eqx.field(static=True)
    basis_gauge_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_matrix_entries: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        data: ConformalDataPlan,
        component_labels: Sequence[str],
        matrices: ArrayLike,
        /,
        *,
        basis_gauge_id: str,
        tolerance: float = 1e-10,
        maximum_matrix_entries: int = 1_000_000,
    ):
        if not isinstance(data, ConformalDataPlan):
            raise TypeError("data must be ConformalDataPlan.")
        labels = tuple(str(value) for value in component_labels)
        gauge = str(basis_gauge_id)
        tolerance_value = float(tolerance)
        maximum = int(maximum_matrix_entries)
        raw = np.asarray(matrices)
        if (
            not labels
            or any(not value for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("Crossing component labels must be unique and non-empty.")
        if len(labels) > data.maximum_tensor_structures:
            raise ValueError("Crossing tensor structures exceed data-plan admission.")
        expected = (len(data.channels), len(labels), len(labels))
        if raw.shape != expected:
            raise ValueError(f"Crossing matrices must have shape {expected}.")
        if raw.size > maximum:
            raise ValueError("Crossing matrices exceed maximum_matrix_entries.")
        if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
            raise ValueError("Crossing matrices must be finite numeric arrays.")
        if not gauge:
            raise ValueError("basis_gauge_id must be non-empty.")
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0 or maximum < 1:
            raise ValueError("Crossing tolerance/resources are invalid.")
        self.data = data
        self.matrices = jnp.asarray(raw, dtype=jnp.complex128)
        self.component_labels = labels
        self.basis_gauge_id = gauge
        self.tolerance = tolerance_value
        self.maximum_matrix_entries = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "crossing-sector-plan",
                "data": data.plan_id,
                "component_labels": labels,
                "matrices": array_tree_fingerprint(raw),
                "basis_gauge_id": gauge,
                "tolerance": tolerance_value,
                "maximum_matrix_entries": maximum,
            }
        )


class CrossingSectorEvidence(StrictModule):
    """Finite crossing-matrix algebra evidence in the supplied basis gauge."""

    involution_residuals: Array
    determinant_magnitudes: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    basis_gauge_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class PreparedCrossingSectors(StrictModule):
    """Prepared fixed matrices with label-indexed crossing action."""

    plan: CrossingSectorPlan
    inverse_matrices: Array
    evidence: CrossingSectorEvidence
    prepared_id: str = eqx.field(static=True)

    def channel_index(self, label: str, /) -> int:
        target = str(label)
        for index, channel in enumerate(self.plan.data.channels):
            if channel.label == target:
                return index
        raise ValueError("Crossing-channel label is outside the prepared plan.")

    def apply(self, channel_label: str, components: ArrayLike, /) -> Array:
        """Apply a crossing matrix to the leading component axis."""
        index = self.channel_index(channel_label)
        values = jnp.asarray(components, dtype=jnp.complex128)
        if values.ndim < 1 or values.shape[0] != len(self.plan.component_labels):
            raise ValueError("Crossing components must begin with the component axis.")
        return ein.contract("ab,b...->a...", self.plan.matrices[index], values)

    def inverse_apply(self, channel_label: str, components: ArrayLike, /) -> Array:
        index = self.channel_index(channel_label)
        values = jnp.asarray(components, dtype=jnp.complex128)
        if values.ndim < 1 or values.shape[0] != len(self.plan.component_labels):
            raise ValueError("Crossing components must begin with the component axis.")
        return ein.contract("ab,b...->a...", self.inverse_matrices[index], values)

    def relation_residual(
        self,
        channel_label: str,
        direct: ArrayLike,
        crossed: ArrayLike,
        /,
    ) -> Array:
        direct_values = jnp.asarray(direct, dtype=jnp.complex128)
        crossed_values = jnp.asarray(crossed, dtype=jnp.complex128)
        transformed = self.apply(channel_label, crossed_values)
        if direct_values.shape != transformed.shape:
            raise ValueError("Direct and crossed component arrays must match.")
        scale = jnp.maximum(1.0, jnp.linalg.norm(direct_values))
        return jnp.linalg.norm(direct_values - transformed) / scale


def prepare_crossing_sectors(plan: CrossingSectorPlan, /) -> PreparedCrossingSectors:
    if not isinstance(plan, CrossingSectorPlan):
        raise TypeError("plan must be CrossingSectorPlan.")
    matrices = np.asarray(plan.matrices, dtype=np.complex128)
    identity = np.eye(len(plan.component_labels), dtype=np.complex128)
    inverses = np.empty_like(matrices)
    residuals = np.zeros((len(plan.data.channels),), dtype=np.float64)
    determinants = np.zeros_like(residuals)
    for index, channel in enumerate(plan.data.channels):
        matrix = matrices[index]
        determinant = np.linalg.det(matrix)
        determinants[index] = abs(determinant)
        if not np.isfinite(determinant) or abs(determinant) <= plan.tolerance:
            raise ValueError(f"Crossing matrix {channel.label!r} is singular.")
        inverses[index] = np.linalg.inv(matrix)
        if channel.involutive:
            residuals[index] = float(
                np.linalg.norm(matrix @ matrix - identity)
                / max(1.0, float(np.linalg.norm(identity)))
            )
            if residuals[index] > plan.tolerance:
                raise ValueError(
                    f"Crossing matrix {channel.label!r} violates its involution."
                )
    finite = bool(
        np.all(np.isfinite(inverses))
        and np.all(np.isfinite(residuals))
        and np.all(np.isfinite(determinants))
    )
    accepted = finite and bool(np.all(residuals <= plan.tolerance))
    evidence = CrossingSectorEvidence(
        involution_residuals=jnp.asarray(residuals),
        determinant_magnitudes=jnp.asarray(determinants),
        finite=jnp.asarray(finite),
        accepted=jnp.asarray(accepted),
        plan_id=plan.plan_id,
        basis_gauge_id=plan.basis_gauge_id,
        claim="finite-caller-supplied-crossing-basis-algebra-only",
    )
    return PreparedCrossingSectors(
        plan=plan,
        inverse_matrices=jnp.asarray(inverses),
        evidence=evidence,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-crossing-sectors",
                "plan": plan.plan_id,
                "inverse_matrices": array_tree_fingerprint(inverses),
            }
        ),
    )


__all__ = [
    "CrossingSectorEvidence",
    "CrossingSectorPlan",
    "PreparedCrossingSectors",
    "prepare_crossing_sectors",
]
