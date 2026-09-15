#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified adaptive sampling policy for implicit block-AMR cut geometry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._canonical import (
    BlockAMRResourcePlan,
    canonicalize_patch_hierarchy,
    CanonicalPatchHierarchy,
)
from ._core import BlockHierarchyTopology
from ._cut_complex import (
    EmbeddedLevelSetBody,
    EmbeddedLevelSetBodySet,
    MultivaluedCutCellPlan,
)
from ._cut_complex_2d import MultivaluedCutCell2DPlan
from ._mapped_geometry import PatchCoordinateMapSet
from ._variable import VariablePatchHierarchyTopology


IntervalBound = Callable[[Array, Array, Array, Any], tuple[ArrayLike, ArrayLike]]
SimplexCertificate = Callable[[Array, Array, Array, Any], ArrayLike]


class CertifiedImplicitBody(StrictModule, NonTrainableState):
    """Level set with physical interval bounds and local PL-topology certificate."""

    body: EmbeddedLevelSetBody
    interval_bound: IntervalBound = eqx.field(static=True)
    simplex_certificate: SimplexCertificate = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        body: EmbeddedLevelSetBody,
        interval_bound: IntervalBound,
        simplex_certificate: SimplexCertificate,
        certificate_id: str,
        /,
    ):
        identifier = str(certificate_id)
        if (
            not isinstance(body, EmbeddedLevelSetBody)
            or not callable(interval_bound)
            or not callable(simplex_certificate)
            or not identifier
        ):
            raise ValueError(
                "Certified implicit body requires body and callable evidence."
            )
        self.body = body
        self.interval_bound = interval_bound
        self.simplex_certificate = simplex_certificate
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "certified-implicit-body",
                "body": body.body_id,
                "certificate": identifier,
            }
        )


class AdaptiveImplicitSamplingEvidence(StrictModule, NonTrainableState):
    """Resolved cell counts and finite global subdivision selected by certificates."""

    leaf_cell_count: int = eqx.field(static=True)
    certified_uniform_cells: int = eqx.field(static=True)
    certified_cut_cells: int = eqx.field(static=True)
    refined_boxes: int = eqx.field(static=True)
    maximum_depth_used: int = eqx.field(static=True)
    subdivision: int = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedAdaptiveImplicitSampling(StrictModule, NonTrainableState):
    """Certified finite subdivision reusable by 2-D or 3-D cut preparation."""

    hierarchy: CanonicalPatchHierarchy
    bodies: tuple[CertifiedImplicitBody, ...]
    subdivision: int = eqx.field(static=True)
    evidence: AdaptiveImplicitSamplingEvidence
    plan_id: str = eqx.field(static=True)


class AdaptiveImplicitSamplingPlan(StrictModule, NonTrainableState):
    """Recursively certify uniform or locally piecewise-linear implicit cells."""

    hierarchy: CanonicalPatchHierarchy
    coordinate_map: Any = eqx.field(static=True)
    coordinate_map_id: str = eqx.field(static=True)
    bodies: tuple[CertifiedImplicitBody, ...]
    maximum_depth: int = eqx.field(static=True)
    interval_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology | VariablePatchHierarchyTopology,
        coordinate_map: Any,
        coordinate_map_id: str,
        bodies: Sequence[CertifiedImplicitBody],
        /,
        *,
        maximum_depth: int = 6,
        interval_tolerance: float = 1.0e-12,
    ):
        hierarchy = canonicalize_patch_hierarchy(topology)
        bodies_ = tuple(bodies)
        depth = int(maximum_depth)
        tolerance = float(interval_tolerance)
        if not callable(coordinate_map) and not isinstance(
            coordinate_map, PatchCoordinateMapSet
        ):
            raise TypeError("Adaptive implicit sampling requires a coordinate map.")
        if (
            not bodies_
            or not all(isinstance(body, CertifiedImplicitBody) for body in bodies_)
            or len({body.body.body_tag for body in bodies_}) != len(bodies_)
        ):
            raise ValueError("Adaptive implicit bodies require unique tags.")
        if depth < 0 or not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Adaptive implicit depth/tolerance is invalid.")
        self.hierarchy = hierarchy
        self.coordinate_map = coordinate_map
        self.coordinate_map_id = str(coordinate_map_id)
        self.bodies = tuple(sorted(bodies_, key=lambda value: value.body.body_tag))
        self.maximum_depth = depth
        self.interval_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-implicit-sampling-plan",
                "hierarchy": hierarchy.hierarchy_id,
                "map": self.coordinate_map_id,
                "bodies": [body.certificate_id for body in self.bodies],
                "maximum_depth": depth,
                "interval_tolerance": tolerance,
            }
        )

    def _leaf_cells(self):
        cells = []
        for level in self.hierarchy.levels:
            for bucket in level.buckets:
                active = np.asarray(bucket.leaf_active, dtype=bool)
                for lane, box in enumerate(bucket.boxes):
                    if box is None:
                        continue
                    for local in np.argwhere(active[lane]):
                        cells.append(
                            (
                                level.level,
                                tuple(
                                    int(start) + int(offset)
                                    for start, offset in zip(
                                        box.lower, local, strict=True
                                    )
                                ),
                                box.box_id,
                            )
                        )
        return tuple(sorted(cells))

    def prepare(
        self,
        time: ArrayLike = 0.0,
        args: Any = None,
        /,
    ) -> PreparedAdaptiveImplicitSampling:
        time_ = jnp.asarray(time)
        if time_.shape != ():
            raise ValueError("Adaptive implicit sampling time must be scalar.")
        dimension = self.hierarchy.dimension
        lower_bounds = np.asarray(
            [
                axis.bounds[0]
                for axis in self.hierarchy.topology.plan.grid.structured_axes
            ],
            dtype=float,
        )
        uniform_count = 0
        cut_count = 0
        refined_count = 0
        maximum_used = 0

        def certify_box(
            patch_id: str,
            reference_lower: np.ndarray,
            reference_upper: np.ndarray,
            depth: int,
        ) -> None:
            nonlocal uniform_count, cut_count, refined_count, maximum_used
            corners = np.asarray(
                tuple(
                    tuple(
                        reference_upper[axis] if side else reference_lower[axis]
                        for axis, side in enumerate(bits)
                    )
                    for bits in product((0, 1), repeat=dimension)
                )
            )
            mapped = (
                self.coordinate_map.map_for(patch_id)
                if isinstance(self.coordinate_map, PatchCoordinateMapSet)
                else self.coordinate_map
            )
            points = jnp.asarray(mapped(jnp.asarray(corners), time_, args))
            physical = np.asarray(points, dtype=float)
            if physical.shape != corners.shape or np.any(~np.isfinite(physical)):
                raise ValueError("Adaptive coordinate map returned invalid points.")
            physical_lower = jnp.asarray(np.min(physical, axis=0))
            physical_upper = jnp.asarray(np.max(physical, axis=0))
            unresolved = False
            crossing = False
            for certified in self.bodies:
                interval_lower, interval_upper = certified.interval_bound(
                    physical_lower,
                    physical_upper,
                    time_,
                    args,
                )
                lower = float(np.asarray(interval_lower))
                upper = float(np.asarray(interval_upper))
                if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
                    raise ValueError(
                        "Implicit interval certificate returned invalid bounds."
                    )
                if lower > self.interval_tolerance or upper < -self.interval_tolerance:
                    continue
                values = jnp.asarray(certified.body.level_set(points, time_, args))
                if values.shape != (corners.shape[0],) or not bool(
                    jnp.all(jnp.isfinite(values))
                ):
                    raise ValueError(
                        "Certified level set returned invalid corner values."
                    )
                signs_cross = bool(jnp.any(values > 0.0) & jnp.any(values < 0.0))
                locally_certified = bool(
                    jnp.asarray(
                        certified.simplex_certificate(points, values, time_, args)
                    )
                )
                crossing = crossing or signs_cross
                unresolved = unresolved or not (signs_cross and locally_certified)
            if not unresolved:
                if crossing:
                    cut_count += 1
                else:
                    uniform_count += 1
                maximum_used = max(maximum_used, depth)
                return
            if depth >= self.maximum_depth:
                raise ValueError(
                    "Implicit topology remains unresolved at maximum adaptive depth."
                )
            refined_count += 1
            midpoint = 0.5 * (reference_lower + reference_upper)
            for bits in product((0, 1), repeat=dimension):
                child_lower = np.asarray(
                    [
                        midpoint[axis] if side else reference_lower[axis]
                        for axis, side in enumerate(bits)
                    ]
                )
                child_upper = np.asarray(
                    [
                        reference_upper[axis] if side else midpoint[axis]
                        for axis, side in enumerate(bits)
                    ]
                )
                certify_box(patch_id, child_lower, child_upper, depth + 1)

        cells = self._leaf_cells()
        for level_index, coordinate, patch_id in cells:
            spacing = np.asarray(self.hierarchy.levels[level_index].spacing, dtype=float)
            reference_lower = lower_bounds + spacing * np.asarray(coordinate)
            certify_box(
                patch_id,
                reference_lower,
                reference_lower + spacing,
                0,
            )
        subdivision = 2**maximum_used
        evidence = AdaptiveImplicitSamplingEvidence(
            leaf_cell_count=len(cells),
            certified_uniform_cells=uniform_count,
            certified_cut_cells=cut_count,
            refined_boxes=refined_count,
            maximum_depth_used=maximum_used,
            subdivision=subdivision,
            valid=True,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "adaptive-implicit-sampling-evidence",
                    "plan": self.plan_id,
                    "leaf_cells": len(cells),
                    "uniform": uniform_count,
                    "cut": cut_count,
                    "refined": refined_count,
                    "maximum_depth": maximum_used,
                    "subdivision": subdivision,
                }
            ),
        )
        return PreparedAdaptiveImplicitSampling(
            self.hierarchy,
            self.bodies,
            subdivision,
            evidence,
            self.plan_id,
        )

    def cut_plan(
        self,
        prepared: PreparedAdaptiveImplicitSampling,
        resources: BlockAMRResourcePlan,
        /,
        *,
        predicate_tolerance: float | None = None,
    ) -> MultivaluedCutCellPlan | MultivaluedCutCell2DPlan:
        if (
            not isinstance(prepared, PreparedAdaptiveImplicitSampling)
            or prepared.plan_id != self.plan_id
            or not isinstance(resources, BlockAMRResourcePlan)
        ):
            raise ValueError("Adaptive implicit cut-plan inputs are incompatible.")
        tolerance = (
            self.interval_tolerance
            if predicate_tolerance is None
            else float(predicate_tolerance)
        )
        bodies = EmbeddedLevelSetBodySet(
            tuple(certified.body for certified in self.bodies)
        )
        plan_type = (
            MultivaluedCutCell2DPlan
            if self.hierarchy.dimension == 2
            else MultivaluedCutCellPlan
        )
        return plan_type(
            self.hierarchy.topology,
            self.coordinate_map,
            self.coordinate_map_id,
            bodies,
            resources,
            subdivision=prepared.subdivision,
            predicate_tolerance=tolerance,
        )


__all__ = [
    "AdaptiveImplicitSamplingEvidence",
    "AdaptiveImplicitSamplingPlan",
    "CertifiedImplicitBody",
    "IntervalBound",
    "PreparedAdaptiveImplicitSampling",
    "SimplexCertificate",
]
