#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._angular_quadrature import CertifiedSlabAngularQuadrature


TransportBoundaryKind = Literal["vacuum", "incident", "reflecting"]


class SlabTransportBoundaryPlan(StrictModule, NonTrainableState):
    left_kind: TransportBoundaryKind = eqx.field(static=True)
    right_kind: TransportBoundaryKind = eqx.field(static=True)
    left_incident: Array
    right_incident: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        group_count: int,
        quadrature: CertifiedSlabAngularQuadrature,
        /,
        *,
        left_kind: TransportBoundaryKind = "vacuum",
        right_kind: TransportBoundaryKind = "vacuum",
        left_incident: ArrayLike | None = None,
        right_incident: ArrayLike | None = None,
    ):
        groups = int(group_count)
        if left_kind not in ("vacuum", "incident", "reflecting") or right_kind not in (
            "vacuum",
            "incident",
            "reflecting",
        ):
            raise ValueError("Unknown slab transport boundary kind.")
        shape = (groups, quadrature.angle_count)
        left = (
            np.zeros(shape)
            if left_incident is None
            else np.asarray(left_incident, dtype=float)
        )
        right = (
            np.zeros(shape)
            if right_incident is None
            else np.asarray(right_incident, dtype=float)
        )
        if (
            groups < 1
            or left.shape != shape
            or right.shape != shape
            or np.any(~np.isfinite(left))
            or np.any(left < 0.0)
            or np.any(~np.isfinite(right))
            or np.any(right < 0.0)
        ):
            raise ValueError("Slab incident boundary arrays are invalid.")
        positive = np.asarray(quadrature.ordinates) > 0.0
        negative = ~positive
        if np.any(left[:, negative] != 0.0) or np.any(right[:, positive] != 0.0):
            raise ValueError("Incident data may be supplied only on incoming ordinates.")
        if left_kind != "incident" and np.any(left != 0.0):
            raise ValueError("Left incident data requires left_kind='incident'.")
        if right_kind != "incident" and np.any(right != 0.0):
            raise ValueError("Right incident data requires right_kind='incident'.")
        self.left_kind = left_kind
        self.right_kind = right_kind
        self.left_incident = jnp.asarray(left)
        self.right_incident = jnp.asarray(right)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "slab-transport-boundaries",
                "left_kind": left_kind,
                "right_kind": right_kind,
                "left": array_tree_fingerprint(left),
                "right": array_tree_fingerprint(right),
                "quadrature": quadrature.quadrature_id,
            }
        )


class MultigroupSlabTransportProblem(StrictModule, NonTrainableState):
    cell_edges: Array
    cell_widths: Array
    total_cross_section: Array
    scattering_cross_section: Array
    fixed_isotropic_source: Array
    quadrature: CertifiedSlabAngularQuadrature
    boundaries: SlabTransportBoundaryPlan
    group_sets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_edges: ArrayLike,
        total_cross_section: ArrayLike,
        scattering_cross_section: ArrayLike,
        fixed_isotropic_source: ArrayLike,
        quadrature: CertifiedSlabAngularQuadrature,
        boundaries: SlabTransportBoundaryPlan,
        /,
        *,
        group_sets: tuple[tuple[int, ...], ...] | None = None,
    ):
        edges = np.asarray(cell_edges, dtype=float)
        total = np.asarray(total_cross_section, dtype=float)
        scattering = np.asarray(scattering_cross_section, dtype=float)
        source = np.asarray(fixed_isotropic_source, dtype=float)
        if not isinstance(quadrature, CertifiedSlabAngularQuadrature):
            raise TypeError("quadrature must be CertifiedSlabAngularQuadrature.")
        if not isinstance(boundaries, SlabTransportBoundaryPlan):
            raise TypeError("boundaries must be SlabTransportBoundaryPlan.")
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("Slab cell edges must be finite and increasing.")
        cells = edges.size - 1
        if total.ndim != 2 or total.shape[0] != cells:
            raise ValueError("Total cross section must have shape (cell, group).")
        groups = total.shape[1]
        if (
            scattering.shape != (cells, groups, groups)
            or source.shape != (cells, groups)
            or np.any(~np.isfinite(total))
            or np.any(total <= 0.0)
            or np.any(~np.isfinite(scattering))
            or np.any(scattering < 0.0)
            or np.any(~np.isfinite(source))
            or np.any(source < 0.0)
            or np.any(np.sum(scattering, axis=-1) > total + 1.0e-12)
            or boundaries.left_incident.shape != (groups, quadrature.angle_count)
        ):
            raise ValueError(
                "Multigroup slab material, source, or boundaries are invalid."
            )
        sets = (
            (tuple(range(groups)),)
            if group_sets is None
            else tuple(tuple(int(group) for group in value) for value in group_sets)
        )
        membership = tuple(group for value in sets for group in value)
        if (
            not sets
            or any(not value for value in sets)
            or sorted(membership) != list(range(groups))
            or len(set(membership)) != groups
        ):
            raise ValueError(
                "Transport group_sets must partition every group exactly once."
            )
        self.cell_edges = jnp.asarray(edges)
        self.cell_widths = jnp.asarray(np.diff(edges))
        self.total_cross_section = jnp.asarray(total)
        self.scattering_cross_section = jnp.asarray(scattering)
        self.fixed_isotropic_source = jnp.asarray(source)
        self.quadrature = quadrature
        self.boundaries = boundaries
        self.group_sets = sets
        self.problem_id = canonical_fingerprint(
            {
                "kind": "multigroup-slab-linear-boltzmann",
                "edges": array_tree_fingerprint(edges),
                "total": array_tree_fingerprint(total),
                "scattering": array_tree_fingerprint(scattering),
                "source": array_tree_fingerprint(source),
                "quadrature": quadrature.quadrature_id,
                "boundaries": boundaries.boundary_id,
                "group_sets": sets,
            }
        )

    @property
    def cell_count(self) -> int:
        return int(self.total_cross_section.shape[0])

    @property
    def group_count(self) -> int:
        return int(self.total_cross_section.shape[1])

    @property
    def absorption_cross_section(self) -> Array:
        return self.total_cross_section - jnp.sum(self.scattering_cross_section, axis=-1)


__all__ = [
    "MultigroupSlabTransportProblem",
    "SlabTransportBoundaryPlan",
    "TransportBoundaryKind",
]
