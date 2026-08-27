#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractLinearOperator
from ._generic import IntegrationDomain


class EmbeddedQuadrature(StrictModule, NonTrainableState):
    """Fixed-capacity cut-cell points, weights, and validity over a cell domain."""

    domain: IntegrationDomain
    reference_points: Array
    reference_weights: Array
    valid: Array
    classification_version: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: IntegrationDomain,
        reference_points: ArrayLike,
        reference_weights: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        classification_version: str,
    ):
        if not isinstance(domain, IntegrationDomain) or domain.kind != "cell":
            raise ValueError("Embedded quadrature requires a cell IntegrationDomain.")
        points = np.asarray(reference_points, dtype=float)
        weights = np.asarray(reference_weights, dtype=float)
        valid_ = np.asarray(valid, dtype=bool)
        if (
            points.ndim != 3
            or weights.shape != points.shape[:2]
            or valid_.shape != weights.shape
        ):
            raise ValueError("Embedded points/weights/validity have incompatible shapes.")
        if points.shape[0] != domain.entity_indices.size:
            raise ValueError(
                "Embedded quadrature requires one point bank per domain cell."
            )
        if (
            np.any(~np.isfinite(points[valid_]))
            or np.any(~np.isfinite(weights[valid_]))
            or np.any(weights[valid_] < 0.0)
        ):
            raise ValueError(
                "Active embedded quadrature entries must be finite and non-negative."
            )
        version = str(classification_version)
        if not version:
            raise ValueError("classification_version must be non-empty.")
        self.domain = domain
        self.reference_points = jnp.asarray(points)
        self.reference_weights = jnp.asarray(weights)
        self.valid = jnp.asarray(valid_)
        self.classification_version = version
        self.quadrature_id = canonical_fingerprint(
            {
                "kind": "embedded-finite-element-quadrature",
                "domain": domain.domain_id,
                "points": array_tree_fingerprint(points),
                "weights": array_tree_fingerprint(weights),
                "valid": array_tree_fingerprint(valid_),
                "classification_version": version,
            }
        )


class FiniteElementEnrichment(StrictModule, NonTrainableState):
    """Explicit cell activation and pure reference enrichment evaluator."""

    evaluator: Callable
    active_cells: Array
    local_enrichment_count: int = eqx.field(static=True)
    enrichment_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable,
        active_cells: ArrayLike,
        local_enrichment_count: int,
        /,
        *,
        enrichment_id: str,
    ):
        active = np.asarray(active_cells, dtype=bool)
        count = int(local_enrichment_count)
        identifier = str(enrichment_id)
        if not callable(evaluator) or active.ndim != 1 or count <= 0 or not identifier:
            raise ValueError("Enrichment evaluator, activation, count, or ID is invalid.")
        self.evaluator = evaluator
        self.active_cells = jnp.asarray(active)
        self.local_enrichment_count = count
        self.enrichment_id = identifier

    def evaluate(self, points: ArrayLike, /) -> Array:
        values = jnp.asarray(self.evaluator(jnp.asarray(points)))
        if values.shape[-1] != self.local_enrichment_count:
            raise ValueError("Enrichment evaluator returned the wrong local width.")
        return values


class MultiscaleFiniteElementBasis(StrictModule, NonTrainableState):
    """Offline coarse-to-fine basis represented by an existing linear operator."""

    prolongation: AbstractLinearOperator
    coefficient_version: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        prolongation: AbstractLinearOperator,
        coefficient_version: str,
        /,
    ):
        if not isinstance(prolongation, AbstractLinearOperator):
            raise TypeError("prolongation must be AbstractLinearOperator.")
        version = str(coefficient_version)
        if not version:
            raise ValueError("coefficient_version must be non-empty.")
        self.prolongation = prolongation
        self.coefficient_version = version
        self.basis_id = canonical_fingerprint(
            {
                "kind": "multiscale-finite-element-basis",
                "prolongation": prolongation.operator_id,
                "coefficient_version": version,
            }
        )


__all__ = [
    "EmbeddedQuadrature",
    "FiniteElementEnrichment",
    "MultiscaleFiniteElementBasis",
]
