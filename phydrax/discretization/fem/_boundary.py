#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._conservation_boundary import AbstractConservationBoundary
from .._integration_domain import IntegrationDomain
from ._generic import FiniteElementDiscretization
from ._reference_topology import FacetOrientationAction


def _canonical_patch_name(value: object, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Finite-element boundary patch names must be canonical strings.")
    return value


def _facet_id(value: object, name: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer exterior-facet ID.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _subset_domain(
    exterior: IntegrationDomain,
    positions: np.ndarray,
    /,
    *,
    selection_id: str,
) -> IntegrationDomain:
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("Finite-element boundary patches must contain exterior facets.")
    return IntegrationDomain(
        "exterior_facet",
        np.asarray(exterior.entity_indices)[positions],
        exterior.support_id,
        exterior.entity_set_id,
        owner_cells=np.asarray(exterior.owner_cells)[positions],
        neighbour_cells=np.asarray(exterior.neighbour_cells)[positions],
        owner_local_entities=np.asarray(exterior.owner_local_entities)[positions],
        neighbour_local_entities=np.asarray(exterior.neighbour_local_entities)[positions],
        selection_id=selection_id,
    )


def tensor_local_face(cell_kind: str, local_facet: int, /) -> tuple[int, int]:
    mappings = {
        "quadrilateral": ((1, 0), (0, 1), (1, 1), (0, 0)),
        "hexahedron": ((2, 0), (2, 1), (1, 0), (0, 1), (1, 1), (0, 0)),
    }
    if cell_kind not in mappings:
        raise ValueError("Unsupported tensor-cell kind.")
    facet = int(local_facet)
    if facet < 0 or facet >= len(mappings[cell_kind]):
        raise ValueError("Unsupported tensor-cell local facet.")
    return mappings[cell_kind][facet]


class FiniteElementBoundaryPatch(StrictModule, NonTrainableState):
    """One exhaustively owned finite-element exterior-facet patch."""

    name: str = eqx.field(static=True)
    domain: IntegrationDomain
    boundary: AbstractConservationBoundary
    patch_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        domain: IntegrationDomain,
        boundary: AbstractConservationBoundary,
        /,
    ):
        patch_name = _canonical_patch_name(name)
        if not isinstance(domain, IntegrationDomain) or domain.kind != "exterior_facet":
            raise TypeError("Finite-element boundary patches require an exterior domain.")
        if not isinstance(boundary, AbstractConservationBoundary):
            raise TypeError("Finite-element patches require conservation boundaries.")
        self.name = patch_name
        self.domain = domain
        self.boundary = boundary
        self.patch_id = canonical_fingerprint(
            {
                "kind": "finite-element-boundary-patch",
                "name": patch_name,
                "domain": domain.domain_id,
                "boundary": boundary.boundary_id,
            }
        )


class FiniteElementPeriodicTransform(StrictModule, NonTrainableState):
    """Affine coordinate, facet-orientation, and component periodic transform."""

    coordinate_matrix: jnp.ndarray
    coordinate_offset: jnp.ndarray
    component_matrix: jnp.ndarray
    orientation: FacetOrientationAction = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_matrix,
        coordinate_offset,
        orientation: FacetOrientationAction,
        /,
        *,
        component_matrix=None,
        tolerance: float = 1.0e-10,
    ):
        matrix = np.asarray(coordinate_matrix, dtype=float)
        offset = np.asarray(coordinate_offset, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or offset.shape != (matrix.shape[0],)
            or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(offset))
            or not isinstance(orientation, FacetOrientationAction)
        ):
            raise ValueError("Periodic coordinate transform is invalid.")
        components = (
            np.eye(1, dtype=float)
            if component_matrix is None
            else np.asarray(component_matrix, dtype=float)
        )
        if (
            components.ndim != 2
            or components.shape[0] != components.shape[1]
            or not np.all(np.isfinite(components))
        ):
            raise ValueError("Periodic component transform must be finite and square.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Periodic transform tolerance must be positive.")
        coordinate_orthogonality = matrix.T @ matrix
        component_orthogonality = components.T @ components
        if (
            np.max(np.abs(coordinate_orthogonality - np.eye(matrix.shape[0])))
            > tolerance_
            or np.max(np.abs(component_orthogonality - np.eye(components.shape[0])))
            > tolerance_
        ):
            raise ValueError("Periodic transforms must be orthogonal affine actions.")
        self.coordinate_matrix = jnp.asarray(matrix)
        self.coordinate_offset = jnp.asarray(offset)
        self.component_matrix = jnp.asarray(components)
        self.orientation = orientation
        self.tolerance = tolerance_
        self.transform_id = canonical_fingerprint(
            {
                "kind": "finite-element-periodic-transform",
                "coordinate_matrix": array_tree_fingerprint(matrix),
                "coordinate_offset": array_tree_fingerprint(offset),
                "component_matrix": array_tree_fingerprint(components),
                "orientation": orientation.orientation_id,
                "tolerance": tolerance_,
            }
        )

    def map_coordinates(self, coordinates, /):
        values = jnp.asarray(coordinates)
        return (
            ein.contract("ij,...j->...i", self.coordinate_matrix, values, backend="jax")
            + self.coordinate_offset
        )

    def map_components(self, values, /):
        data = jnp.asarray(values)
        if self.component_matrix.shape == (1, 1):
            return data
        if data.shape[-1] != self.component_matrix.shape[1]:
            raise ValueError("Periodic component transform shape is incompatible.")
        return ein.contract("ij,...j->...i", self.component_matrix, data, backend="jax")


class FiniteElementPeriodicFacetPair(StrictModule, NonTrainableState):
    """One explicit pair of exterior facets forming a periodic interface."""

    owner_facet: int = eqx.field(static=True)
    neighbour_facet: int = eqx.field(static=True)
    transform: FiniteElementPeriodicTransform | None
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner_facet: int,
        neighbour_facet: int,
        /,
        *,
        transform: FiniteElementPeriodicTransform | None = None,
    ):
        owner = _facet_id(owner_facet, "owner_facet")
        neighbour = _facet_id(neighbour_facet, "neighbour_facet")
        if owner == neighbour:
            raise ValueError("A periodic facet cannot be paired with itself.")
        if transform is not None and not isinstance(
            transform, FiniteElementPeriodicTransform
        ):
            raise TypeError("transform must be FiniteElementPeriodicTransform or None.")
        self.owner_facet = owner
        self.neighbour_facet = neighbour
        self.transform = transform
        self.pair_id = canonical_fingerprint(
            {
                "kind": "finite-element-periodic-facet-pair",
                "owner": owner,
                "neighbour": neighbour,
                "transform": (None if transform is None else transform.transform_id),
            }
        )


class FiniteElementBoundarySet(StrictModule, NonTrainableState):
    """Exhaustive physical and periodic ownership of FE exterior facets."""

    patch_names: tuple[str, ...] = eqx.field(static=True)
    patches: tuple[FiniteElementBoundaryPatch, ...]
    periodic_pairs: tuple[FiniteElementPeriodicFacetPair, ...]
    support_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        physical: Mapping[str, tuple[Sequence[int], AbstractConservationBoundary]],
        /,
        *,
        periodic_pairs: Sequence[FiniteElementPeriodicFacetPair] = (),
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be a FiniteElementDiscretization.")
        if not isinstance(physical, Mapping):
            raise TypeError("physical must map patch names to facets and policies.")
        pairs = tuple(periodic_pairs)
        if not all(isinstance(pair, FiniteElementPeriodicFacetPair) for pair in pairs):
            raise TypeError("periodic_pairs must contain periodic facet pairs.")

        exterior = discretization.exterior_facet_domain
        exterior_ids = np.asarray(exterior.entity_indices, dtype=np.int64)
        if exterior_ids.ndim != 1 or np.unique(exterior_ids).size != exterior_ids.size:
            raise ValueError("Prepared exterior facet IDs must be unique.")
        positions_by_id = {
            int(facet): position for position, facet in enumerate(exterior_ids)
        }
        exterior_set = set(positions_by_id)

        periodic_ids: list[int] = []
        for pair in pairs:
            periodic_ids.extend((pair.owner_facet, pair.neighbour_facet))
        if len(periodic_ids) != len(set(periodic_ids)):
            raise ValueError("Periodic exterior facets must be owned exactly once.")
        unknown_periodic = set(periodic_ids) - exterior_set
        if unknown_periodic:
            raise ValueError("Periodic pairs reference non-exterior facets.")

        names = tuple(sorted(_canonical_patch_name(name) for name in physical))
        if len(names) != len(set(names)):
            raise ValueError("Finite-element boundary patch names must be unique.")
        patches = []
        physical_ids: list[int] = []
        for name in names:
            value = physical[name]
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError(
                    "Physical boundary entries must be (facet_ids, boundary) tuples."
                )
            raw_facets, boundary = value
            if not isinstance(boundary, AbstractConservationBoundary):
                raise TypeError("Physical patches require conservation boundaries.")
            facets = tuple(
                _facet_id(facet, f"physical[{name!r}]") for facet in raw_facets
            )
            if not facets:
                raise ValueError("Physical boundary patches cannot be empty.")
            if len(facets) != len(set(facets)):
                raise ValueError("A physical boundary patch repeats an exterior facet.")
            unknown = set(facets) - exterior_set
            if unknown:
                raise ValueError("Physical patches reference non-exterior facets.")
            physical_ids.extend(facets)
            positions = np.asarray(
                [positions_by_id[facet] for facet in facets], dtype=np.int32
            )
            domain = _subset_domain(
                exterior,
                positions,
                selection_id=canonical_fingerprint(
                    {
                        "kind": "finite-element-boundary-selection",
                        "name": name,
                        "facets": facets,
                    }
                ),
            )
            patches.append(FiniteElementBoundaryPatch(name, domain, boundary))

        if len(physical_ids) != len(set(physical_ids)):
            raise ValueError("Physical boundary patches overlap.")
        if set(physical_ids) & set(periodic_ids):
            raise ValueError("Physical and periodic exterior ownership overlaps.")
        owned = set(physical_ids) | set(periodic_ids)
        if owned != exterior_set:
            missing = tuple(sorted(exterior_set - owned))
            raise ValueError(
                "Finite-element boundary ownership must be exhaustive; "
                f"missing exterior facets {missing}."
            )

        self.patch_names = names
        self.patches = tuple(patches)
        self.periodic_pairs = pairs
        self.support_id = exterior.support_id
        self.entity_set_id = exterior.entity_set_id
        self.boundary_set_id = canonical_fingerprint(
            {
                "kind": "finite-element-boundary-set",
                "support": exterior.support_id,
                "entity_set": exterior.entity_set_id,
                "patches": tuple(patch.patch_id for patch in patches),
                "periodic": tuple(pair.pair_id for pair in pairs),
            }
        )

    def patch(self, name: str, /) -> FiniteElementBoundaryPatch:
        patch_name = _canonical_patch_name(name)
        for patch in self.patches:
            if patch.name == patch_name:
                return patch
        raise ValueError(f"Unknown finite-element boundary patch {patch_name!r}.")


__all__ = [
    "FiniteElementBoundaryPatch",
    "FiniteElementBoundarySet",
    "FiniteElementPeriodicFacetPair",
    "FiniteElementPeriodicTransform",
    "tensor_local_face",
]
