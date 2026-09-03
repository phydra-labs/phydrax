#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import equinox as eqx
import numpy as np

from phydrax.domain import (
    AbstractGeometry,
    Boundary,
    ComponentSum,
    DomainComponent,
    GeometryDomain,
)
from phydrax.geometry import BoundaryAtlas, GeometryCapability
from phydrax.geometry._capabilities import ClosestPointProvider

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class BoundarySide(str, Enum):
    """Orientation role of one boundary patch."""

    BOUNDARY = "boundary"
    MINUS = "minus"
    PLUS = "plus"


class JunctionKind(str, Enum):
    """Exact narrow-phase evidence used for one closure intersection."""

    ENTITY_ADJACENCY = "entity_adjacency"
    CHART_TRANSITION = "chart_transition"
    COMMON_REFINEMENT = "common_refinement"
    SUPPLIED_TOPOLOGY = "supplied_topology"


class BoundaryPatch(StrictModule, NonTrainableState):
    """One stable, oriented boundary support in its represented geometry.

    ``predicate_id`` names a filtered support but does not itself prove the
    existence of a collar.  Such a proof is carried separately by
    ``collar_certificate_id``.  This prevents a sampled predicate or a BVH
    proximity result from being promoted to exact continuum topology.
    """

    component: DomainComponent
    atlas: BoundaryAtlas | None
    collar_provider: Any | None
    variable: str = eqx.field(static=True)
    side: BoundarySide = eqx.field(static=True)
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    tags: tuple[str, ...] = eqx.field(static=True)
    orientation: tuple[int, ...] = eqx.field(static=True)
    predicate_id: str | None = eqx.field(static=True)
    collar_certificate_id: str | None = eqx.field(static=True)
    topology_certificate_id: str = eqx.field(static=True)
    represented_geometry_id: str = eqx.field(static=True)
    physical_geometry_id: str | None = eqx.field(static=True)
    exact_to_physical: bool = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: DomainComponent,
        /,
        *,
        variable: str,
        side: BoundarySide | str = BoundarySide.BOUNDARY,
        atlas: BoundaryAtlas | None = None,
        collar_provider: Any | None = None,
        predicate_id: str | None = None,
        collar_certificate_id: str | None = None,
        topology_certificate_id: str,
        represented_geometry_id: str,
        physical_geometry_id: str | None = None,
        exact_to_physical: bool = False,
    ):
        if not isinstance(component, DomainComponent):
            raise TypeError("BoundaryPatch.component must be a DomainComponent.")
        variable_ = str(variable)
        if variable_ not in component.domain.labels:
            raise KeyError(f"Boundary variable {variable_!r} is outside the component.")
        selection = component.spec.selection_for(variable_)
        if not isinstance(selection, Boundary):
            raise ValueError("BoundaryPatch variable must select Boundary().")
        factor = component.domain.factor(variable_)
        if not isinstance(factor, AbstractGeometry):
            raise TypeError("BoundaryPatch variable must name a geometry factor.")
        side_ = BoundarySide(side)
        topology_id = str(topology_certificate_id)
        represented_id = str(represented_geometry_id)
        if not topology_id or not represented_id:
            raise ValueError("Topology and represented-geometry IDs must be non-empty.")
        predicate = None if predicate_id is None else str(predicate_id)
        if predicate is not None and not predicate:
            raise ValueError("predicate_id must be non-empty when provided.")
        has_filter = bool(component.where) or component.where_all is not None
        if has_filter != (predicate is not None):
            raise ValueError(
                "Filtered boundary patches require exactly one explicit stable "
                "predicate_id; unfiltered patches must not declare one."
            )
        collar_id = None if collar_certificate_id is None else str(collar_certificate_id)
        if collar_id is not None and not collar_id:
            raise ValueError("collar_certificate_id must be non-empty when provided.")
        if collar_provider is not None:
            if not isinstance(collar_provider, ClosestPointProvider):
                raise TypeError(
                    "collar_provider must implement the closest-point capability."
                )
            if collar_id is None:
                raise ValueError("A collar provider requires a collar certificate ID.")
        physical_id = None if physical_geometry_id is None else str(physical_geometry_id)
        if physical_id is not None and not physical_id:
            raise ValueError("physical_geometry_id must be non-empty when provided.")
        physical_exact = bool(exact_to_physical)
        if physical_exact and physical_id is None:
            raise ValueError("exact_to_physical requires a physical geometry ID.")

        if atlas is None:
            entity_ids = tuple(
                () if selection.entity_ids is None else selection.entity_ids
            )
            tags = tuple(() if selection.tags is None else selection.tags)
            orientation = (1,)
        else:
            if not isinstance(atlas, BoundaryAtlas):
                raise TypeError("BoundaryPatch.atlas must be a BoundaryAtlas or None.")
            entity_ids = tuple(
                int(value) for value in np.asarray(atlas.source_entity_ids)
            )
            tags = tuple(atlas.physical_tags)
            orientation = tuple(int(value) for value in np.asarray(atlas.orientation))
            if not entity_ids:
                raise ValueError(
                    "A boundary patch atlas must contain at least one chart."
                )
            if side_ is BoundarySide.PLUS:
                orientation = tuple(-value for value in orientation)

        support_id = canonical_fingerprint(
            {
                "kind": "boundary-patch-v1",
                "variable": variable_,
                "side": side_.value,
                "represented_geometry": represented_id,
                "physical_geometry": physical_id,
                "entity_ids": entity_ids,
                "tags": tags,
                "orientation": orientation,
                "predicate": predicate,
                "topology": topology_id,
                "collar": collar_id,
            }
        )
        self.component = component
        self.atlas = atlas
        self.collar_provider = collar_provider
        self.variable = variable_
        self.side = side_
        self.entity_ids = entity_ids
        self.tags = tags
        self.orientation = orientation
        self.predicate_id = predicate
        self.collar_certificate_id = collar_id
        self.topology_certificate_id = topology_id
        self.represented_geometry_id = represented_id
        self.physical_geometry_id = physical_id
        self.exact_to_physical = physical_exact
        self.support_id = support_id


class BoundaryJunction(StrictModule, NonTrainableState):
    """Certified closure intersection among two or more boundary patches."""

    patch_indices: tuple[int, ...] = eqx.field(static=True)
    kind: JunctionKind = eqx.field(static=True)
    topology_certificate_id: str = eqx.field(static=True)
    transition_ids: tuple[str, ...] = eqx.field(static=True)
    orientation_signs: tuple[int, ...] = eqx.field(static=True)
    compatibility_operator_id: str = eqx.field(static=True)
    junction_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_indices: Sequence[int],
        /,
        *,
        kind: JunctionKind | str,
        topology_certificate_id: str,
        compatibility_operator_id: str,
        transition_ids: Sequence[str] = (),
        orientation_signs: Sequence[int] = (),
    ):
        indices = tuple(int(value) for value in patch_indices)
        if len(indices) < 2 or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError(
                "BoundaryJunction.patch_indices must contain at least two unique "
                "non-negative indices."
            )
        kind_ = JunctionKind(kind)
        topology_id = str(topology_certificate_id)
        compatibility_id = str(compatibility_operator_id)
        transitions = tuple(str(value) for value in transition_ids)
        if (
            not topology_id
            or not compatibility_id
            or any(not value for value in transitions)
        ):
            raise ValueError("Junction evidence IDs must be non-empty.")
        signs = tuple(int(value) for value in orientation_signs)
        if not signs:
            signs = tuple(1 for _ in indices)
        if len(signs) != len(indices) or any(value not in (-1, 1) for value in signs):
            raise ValueError("Junction orientation signs must be ±1 per patch.")
        self.patch_indices = indices
        self.kind = kind_
        self.topology_certificate_id = topology_id
        self.transition_ids = transitions
        self.orientation_signs = signs
        self.compatibility_operator_id = compatibility_id
        self.junction_id = canonical_fingerprint(
            {
                "kind": "boundary-junction-v1",
                "patch_indices": indices,
                "evidence_kind": kind_.value,
                "topology": topology_id,
                "transitions": transitions,
                "orientation": signs,
                "compatibility": compatibility_id,
            }
        )


class BoundarySupportEvidence(StrictModule, NonTrainableState):
    """Preparation evidence for one exact represented boundary cover."""

    support_ids: tuple[str, ...] = eqx.field(static=True)
    junction_ids: tuple[str, ...] = eqx.field(static=True)
    represented_geometry_ids: tuple[str, ...] = eqx.field(static=True)
    physical_geometry_ids: tuple[str | None, ...] = eqx.field(static=True)
    coverage_certificate_id: str = eqx.field(static=True)
    topology_certificate_ids: tuple[str, ...] = eqx.field(static=True)
    orientation_valid: bool = eqx.field(static=True)
    intersections_resolved: bool = eqx.field(static=True)
    coverage_complete: bool = eqx.field(static=True)
    physical_exact: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        support_ids: Sequence[str],
        junction_ids: Sequence[str],
        represented_geometry_ids: Sequence[str],
        physical_geometry_ids: Sequence[str | None],
        coverage_certificate_id: str,
        topology_certificate_ids: Sequence[str],
        orientation_valid: bool,
        intersections_resolved: bool,
        coverage_complete: bool,
        physical_exact: bool,
    ):
        supports = tuple(str(value) for value in support_ids)
        junctions = tuple(str(value) for value in junction_ids)
        represented = tuple(str(value) for value in represented_geometry_ids)
        physical = tuple(
            None if value is None else str(value) for value in physical_geometry_ids
        )
        topology = tuple(str(value) for value in topology_certificate_ids)
        coverage = str(coverage_certificate_id)
        if (
            not supports
            or any(
                not value for value in (*supports, *junctions, *represented, *topology)
            )
            or any(value == "" for value in physical if value is not None)
        ):
            raise ValueError("Boundary support evidence IDs must be non-empty.")
        if len(represented) != len(supports) or len(physical) != len(supports):
            raise ValueError("Geometry evidence must contain one entry per support.")
        if len(topology) != len(supports) + len(junctions):
            raise ValueError(
                "Topology evidence must contain one entry per support and junction."
            )
        if not coverage:
            raise ValueError("coverage_certificate_id must be non-empty.")
        self.support_ids = supports
        self.junction_ids = junctions
        self.represented_geometry_ids = represented
        self.physical_geometry_ids = physical
        self.coverage_certificate_id = coverage
        self.topology_certificate_ids = topology
        self.orientation_valid = bool(orientation_valid)
        self.intersections_resolved = bool(intersections_resolved)
        self.coverage_complete = bool(coverage_complete)
        self.physical_exact = bool(physical_exact)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "boundary-support-evidence-v1",
                "supports": supports,
                "junctions": junctions,
                "represented": represented,
                "physical": physical,
                "coverage": coverage,
                "topology": topology,
                "orientation_valid": self.orientation_valid,
                "intersections_resolved": self.intersections_resolved,
                "coverage_complete": self.coverage_complete,
                "physical_exact": self.physical_exact,
            }
        )


class BoundaryCover(StrictModule, NonTrainableState):
    """Finite exact cover with explicitly resolved closure intersections."""

    patches: tuple[BoundaryPatch, ...]
    junctions: tuple[BoundaryJunction, ...]
    disjoint_pairs: tuple[tuple[int, int, str], ...] = eqx.field(static=True)
    evidence: BoundarySupportEvidence = eqx.field(static=True)
    cover_id: str = eqx.field(static=True)

    def __init__(
        self,
        patches: Sequence[BoundaryPatch],
        junctions: Sequence[BoundaryJunction],
        /,
        *,
        disjoint_pairs: Sequence[tuple[int, int, str]],
        evidence: BoundarySupportEvidence,
    ):
        patches_ = tuple(patches)
        junctions_ = tuple(junctions)
        if not patches_:
            raise ValueError("BoundaryCover requires at least one boundary patch.")
        if not all(isinstance(value, BoundaryPatch) for value in patches_):
            raise TypeError("BoundaryCover patches must be BoundaryPatch values.")
        if not all(isinstance(value, BoundaryJunction) for value in junctions_):
            raise TypeError("BoundaryCover junctions must be BoundaryJunction values.")
        patch_ids = tuple(patch.support_id for patch in patches_)
        junction_ids = tuple(junction.junction_id for junction in junctions_)
        if len(set(patch_ids)) != len(patch_ids):
            raise ValueError("BoundaryCover patches must have distinct support IDs.")
        if len(set(junction_ids)) != len(junction_ids):
            raise ValueError("BoundaryCover junctions must have distinct junction IDs.")
        domain = patches_[0].component.domain
        if any(not domain.same_support(patch.component.domain) for patch in patches_[1:]):
            raise ValueError(
                "BoundaryCover patches must share compatible domain support."
            )
        for junction in junctions_:
            if max(junction.patch_indices) >= len(patches_):
                raise ValueError("A junction references a patch outside this cover.")
        disjoint = tuple(
            (int(first), int(second), str(certificate))
            for first, second, certificate in disjoint_pairs
        )
        for first, second, certificate in disjoint:
            if not (0 <= first < second < len(patches_)) or not certificate:
                raise ValueError(
                    "Disjoint-pair evidence must name a valid ordered pair and ID."
                )
        disjoint_pair_set = {(first, second) for first, second, _ in disjoint}
        if len(disjoint_pair_set) != len(disjoint):
            raise ValueError("Disjoint patch pairs must be certified exactly once.")
        junction_pairs = {
            tuple(sorted((first, second)))
            for junction in junctions_
            for offset, first in enumerate(junction.patch_indices)
            for second in junction.patch_indices[offset + 1 :]
        }
        if junction_pairs & disjoint_pair_set:
            raise ValueError("A patch pair cannot be both intersecting and disjoint.")
        required_pairs = {
            (first, second)
            for first in range(len(patches_))
            for second in range(first + 1, len(patches_))
        }
        if junction_pairs | disjoint_pair_set != required_pairs:
            raise ValueError("BoundaryCover requires evidence for every patch pair.")
        if not isinstance(evidence, BoundarySupportEvidence):
            raise TypeError("BoundaryCover.evidence must be BoundarySupportEvidence.")
        expected_topology = tuple(
            patch.topology_certificate_id for patch in patches_
        ) + tuple(junction.topology_certificate_id for junction in junctions_)
        orientation_valid = all(
            value in (-1, 1) for patch in patches_ for value in patch.orientation
        ) and all(
            value in (-1, 1)
            for junction in junctions_
            for value in junction.orientation_signs
        )
        if (
            evidence.support_ids != patch_ids
            or evidence.junction_ids != junction_ids
            or evidence.represented_geometry_ids
            != tuple(patch.represented_geometry_id for patch in patches_)
            or evidence.physical_geometry_ids
            != tuple(patch.physical_geometry_id for patch in patches_)
            or evidence.topology_certificate_ids != expected_topology
            or evidence.orientation_valid != orientation_valid
            or not evidence.intersections_resolved
            or not evidence.coverage_complete
            or evidence.physical_exact
            != all(patch.exact_to_physical for patch in patches_)
        ):
            raise ValueError("BoundaryCover evidence does not certify this exact cover.")
        self.patches = patches_
        self.junctions = junctions_
        self.disjoint_pairs = disjoint
        self.evidence = evidence
        self.cover_id = canonical_fingerprint(
            {
                "kind": "boundary-cover-v1",
                "patches": tuple(value.support_id for value in patches_),
                "junctions": tuple(value.junction_id for value in junctions_),
                "disjoint": disjoint,
                "evidence": evidence.evidence_id,
            }
        )


def _boundary_variable(component: DomainComponent, requested: str | None, /) -> str:
    selected = tuple(
        label
        for label in component.domain.labels
        if isinstance(component.spec.selection_for(label), Boundary)
    )
    if requested is None:
        if len(selected) != 1:
            raise ValueError(
                "Boundary support must select exactly one geometry variable or "
                "prepare_boundary_cover(variable=...) must disambiguate it."
            )
        return selected[0]
    variable = str(requested)
    if variable not in selected:
        raise ValueError(f"Variable {variable!r} is not a selected boundary factor.")
    return variable


def _factor_identity(factor: AbstractGeometry, /) -> tuple[str, BoundaryAtlas | None]:
    if isinstance(factor, GeometryDomain):
        atlas = (
            factor.boundary_atlas
            if factor.has_geometry_capability(GeometryCapability.BOUNDARY_ATLAS)
            else None
        )
        if atlas is not None:
            return atlas.source_id, atlas
    bounds = np.asarray(factor.bounds)
    identity = canonical_fingerprint(
        {
            "kind": "domain-geometry-representation-v1",
            "type": f"{type(factor).__module__}.{type(factor).__qualname__}",
            "bounds": array_tree_fingerprint(bounds),
        }
    )
    return identity, None


def prepare_boundary_cover(
    support: DomainComponent | ComponentSum,
    /,
    *,
    variable: str | None = None,
    side: BoundarySide | str = BoundarySide.BOUNDARY,
    predicate_ids: Mapping[int, str] | None = None,
    collar_providers: Mapping[int, Any] | None = None,
    collar_certificate_ids: Mapping[int, str] | None = None,
    topology_certificate_ids: Mapping[int, str] | None = None,
    physical_geometry_ids: Mapping[int, str] | None = None,
    exact_to_physical: Mapping[int, bool] | None = None,
    junctions: Sequence[BoundaryJunction] = (),
    disjoint_pairs: Sequence[tuple[int, int, str]] = (),
    coverage_certificate_id: str | None = None,
) -> BoundaryCover:
    """Normalize a boundary component or sum into one exact support cover.

    Every pair of patches must be classified by exact narrow-phase junction
    evidence or by an explicit exact-disjointness certificate.  In particular,
    ``ComponentSum.assume_disjoint`` and AABB/BVH overlap results are never used
    as pointwise topology certificates.
    """
    if isinstance(support, DomainComponent):
        terms = (support,)
    elif isinstance(support, ComponentSum):
        terms = support.terms
    else:
        raise TypeError("support must be DomainComponent or ComponentSum.")
    predicate_map = {} if predicate_ids is None else dict(predicate_ids)
    collar_map = {} if collar_providers is None else dict(collar_providers)
    collar_id_map = {} if collar_certificate_ids is None else dict(collar_certificate_ids)
    topology_map = (
        {} if topology_certificate_ids is None else dict(topology_certificate_ids)
    )
    physical_map = {} if physical_geometry_ids is None else dict(physical_geometry_ids)
    physical_exact_map = {} if exact_to_physical is None else dict(exact_to_physical)
    known_indices = set(range(len(terms)))
    for name, mapping in (
        ("predicate_ids", predicate_map),
        ("collar_providers", collar_map),
        ("collar_certificate_ids", collar_id_map),
        ("topology_certificate_ids", topology_map),
        ("physical_geometry_ids", physical_map),
        ("exact_to_physical", physical_exact_map),
    ):
        if not set(mapping).issubset(known_indices):
            raise KeyError(f"{name} contains a term index outside the support.")

    patches: list[BoundaryPatch] = []
    for index, component in enumerate(terms):
        variable_ = _boundary_variable(component, variable)
        factor = component.domain.factor(variable_)
        if not isinstance(factor, AbstractGeometry):
            raise TypeError("Selected boundary factor is not a geometry.")
        represented_id, atlas = _factor_identity(factor)
        selection = component.spec.selection_for(variable_)
        if atlas is not None and isinstance(selection, Boundary):
            if selection.tags is not None or selection.entity_ids is not None:
                atlas = atlas.select(
                    tags=selection.tags,
                    entity_ids=selection.entity_ids,
                )
        topology_id = topology_map.get(index)
        if topology_id is None:
            topology_id = canonical_fingerprint(
                {
                    "kind": "authoritative-boundary-selection-v1",
                    "represented_geometry": represented_id,
                    "entity_ids": (
                        ()
                        if atlas is None
                        else tuple(
                            int(value) for value in np.asarray(atlas.source_entity_ids)
                        )
                    ),
                    "tags": () if atlas is None else atlas.physical_tags,
                }
            )
        patches.append(
            BoundaryPatch(
                component,
                variable=variable_,
                side=side,
                atlas=atlas,
                collar_provider=collar_map.get(index),
                predicate_id=predicate_map.get(index),
                collar_certificate_id=collar_id_map.get(index),
                topology_certificate_id=topology_id,
                represented_geometry_id=represented_id,
                physical_geometry_id=physical_map.get(index),
                exact_to_physical=physical_exact_map.get(index, False),
            )
        )

    junctions_ = tuple(junctions)
    junction_pairs = {
        tuple(sorted((first, second)))
        for junction in junctions_
        for offset, first in enumerate(junction.patch_indices)
        for second in junction.patch_indices[offset + 1 :]
    }
    disjoint_pairs_ = tuple(
        (min(int(first), int(second)), max(int(first), int(second)), str(certificate))
        for first, second, certificate in disjoint_pairs
    )
    disjoint_pair_set = {(first, second) for first, second, _ in disjoint_pairs_}
    required_pairs = {
        (first, second)
        for first in range(len(patches))
        for second in range(first + 1, len(patches))
    }
    resolved_pairs = junction_pairs | disjoint_pair_set
    missing = tuple(sorted(required_pairs - resolved_pairs))
    if missing:
        raise ValueError(
            "Boundary patch closure intersections are unresolved for pairs "
            f"{missing}; ComponentSum measure disjointness is insufficient."
        )
    if junction_pairs & disjoint_pair_set:
        raise ValueError("A patch pair cannot be both intersecting and exactly disjoint.")
    coverage_id = coverage_certificate_id
    has_filtered = any(patch.predicate_id is not None for patch in patches)
    if coverage_id is None:
        if has_filtered:
            raise ValueError(
                "Filtered boundary covers require an explicit coverage_certificate_id."
            )
        coverage_id = canonical_fingerprint(
            {
                "kind": "structural-boundary-cover-v1",
                "patches": tuple(patch.support_id for patch in patches),
                "resolved_pairs": tuple(sorted(resolved_pairs)),
            }
        )
    coverage_id = str(coverage_id)
    if not coverage_id:
        raise ValueError("coverage_certificate_id must be non-empty.")
    orientation_valid = all(
        value in (-1, 1) for patch in patches for value in patch.orientation
    ) and all(
        value in (-1, 1)
        for junction in junctions_
        for value in junction.orientation_signs
    )
    evidence = BoundarySupportEvidence(
        support_ids=tuple(patch.support_id for patch in patches),
        junction_ids=tuple(value.junction_id for value in junctions_),
        represented_geometry_ids=tuple(
            patch.represented_geometry_id for patch in patches
        ),
        physical_geometry_ids=tuple(patch.physical_geometry_id for patch in patches),
        coverage_certificate_id=coverage_id,
        topology_certificate_ids=tuple(patch.topology_certificate_id for patch in patches)
        + tuple(value.topology_certificate_id for value in junctions_),
        orientation_valid=orientation_valid,
        intersections_resolved=resolved_pairs == required_pairs,
        coverage_complete=True,
        physical_exact=all(patch.exact_to_physical for patch in patches),
    )
    return BoundaryCover(
        patches,
        junctions_,
        disjoint_pairs=disjoint_pairs_,
        evidence=evidence,
    )


__all__ = [
    "BoundaryCover",
    "BoundaryJunction",
    "BoundaryPatch",
    "BoundarySide",
    "BoundarySupportEvidence",
    "JunctionKind",
    "prepare_boundary_cover",
]
