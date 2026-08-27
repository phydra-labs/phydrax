#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation, RowRelation, SparseLinearMap, SparseRelation
from ._core import nonempty_identifier, resolved_identifier


def _array_digest(value: object, /) -> dict[str, object]:
    return array_tree_fingerprint(value)


def _bool_array(name: str, value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return jnp.asarray(array)


class EntitySubset(StrictModule, NonTrainableState):
    """Named subset of one fixed-capacity entity set."""

    name: str = eqx.field(static=True)
    mask: Array
    subset_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        mask: ArrayLike,
        /,
        *,
        subset_id: str | None = None,
    ):
        name_ = nonempty_identifier("name", name)
        mask_ = np.asarray(mask, dtype=bool)
        if mask_.ndim != 1:
            raise ValueError("Entity subset masks must be rank-1.")
        self.name = name_
        self.mask = jnp.asarray(mask_)
        self.subset_id = resolved_identifier(
            "subset_id",
            subset_id,
            {
                "kind": "entity-subset",
                "name": name_,
                "mask": _array_digest(mask_),
            },
        )


class EntitySet(StrictModule, NonTrainableState):
    """Fixed-capacity, identity-bearing topological entities."""

    name: str = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    count: int = eqx.field(static=True)
    entity_ids: Array
    active_mask: Array
    subsets: tuple[EntitySubset, ...]
    entity_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        intrinsic_dimension: int,
        entity_ids: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        subsets: Sequence[EntitySubset] = (),
        entity_set_id: str | None = None,
    ):
        name_ = nonempty_identifier("name", name)
        dimension = int(intrinsic_dimension)
        if dimension < 0:
            raise ValueError("intrinsic_dimension must be non-negative.")
        identifiers = np.asarray(entity_ids)
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("entity_ids must be one rank-1 integer array.")
        identifiers = identifiers.astype(np.int64, copy=False)
        count = int(identifiers.shape[0])
        active = (
            np.ones((count,), dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (count,):
            raise ValueError(
                f"active_mask must have shape {(count,)}; got {active.shape}."
            )
        active_ids = identifiers[active]
        if np.any(active_ids < 0):
            raise ValueError("Active entity IDs must be non-negative.")
        if np.unique(active_ids).size != active_ids.size:
            raise ValueError("Active entity IDs must be unique.")
        subsets_ = tuple(subsets)
        if not all(isinstance(subset, EntitySubset) for subset in subsets_):
            raise TypeError("subsets must contain EntitySubset values.")
        names = tuple(subset.name for subset in subsets_)
        if len(set(names)) != len(names):
            raise ValueError("Entity subset names must be unique.")
        for subset in subsets_:
            subset_mask = np.asarray(subset.mask, dtype=bool)
            if subset_mask.shape != (count,):
                raise ValueError(
                    f"Subset {subset.name!r} must have shape {(count,)}; "
                    f"got {subset_mask.shape}."
                )
            if np.any(subset_mask & ~active):
                raise ValueError("Entity subsets cannot include inactive entities.")
        self.name = name_
        self.intrinsic_dimension = dimension
        self.count = count
        self.entity_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.active_mask = jnp.asarray(active)
        self.subsets = subsets_
        self.entity_set_id = resolved_identifier(
            "entity_set_id",
            entity_set_id,
            {
                "kind": "entity-set",
                "name": name_,
                "intrinsic_dimension": dimension,
                "entity_ids": _array_digest(identifiers),
                "active_mask": _array_digest(active),
                "subsets": [subset.subset_id for subset in subsets_],
            },
        )

    @property
    def num_active(self) -> int:
        return int(np.count_nonzero(np.asarray(self.active_mask)))

    def subset(self, name: str, /) -> EntitySubset:
        for subset in self.subsets:
            if subset.name == name:
                return subset
        raise KeyError(f"Unknown entity subset {name!r} on {self.name!r}.")


class EntitySelection(StrictModule, NonTrainableState):
    """Canonical composable selection over one exact entity set."""

    entity_set_id: str = eqx.field(static=True)
    mask: Array
    active_mask: Array
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        entities: EntitySet | str,
        mask: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        selection_id: str | None = None,
    ):
        if isinstance(entities, EntitySet):
            entity_set_id = entities.entity_set_id
            active = np.asarray(entities.active_mask, dtype=bool)
        else:
            entity_set_id = nonempty_identifier("entity_set_id", entities)
            if active_mask is None:
                raise ValueError(
                    "active_mask is required when constructing from an entity-set ID."
                )
            active = np.asarray(active_mask, dtype=bool)
        mask_ = np.asarray(mask, dtype=bool)
        if active.ndim != 1 or mask_.shape != active.shape:
            raise ValueError(
                "Entity selection and active masks must share one rank-1 shape."
            )
        if np.any(mask_ & ~active):
            raise ValueError("Entity selections cannot include inactive entities.")
        self.entity_set_id = entity_set_id
        self.mask = jnp.asarray(mask_)
        self.active_mask = jnp.asarray(active)
        self.selection_id = resolved_identifier(
            "selection_id",
            selection_id,
            {
                "kind": "entity-selection",
                "entity_set": entity_set_id,
                "mask": _array_digest(mask_),
            },
        )

    @classmethod
    def from_subset(
        cls,
        entities: EntitySet,
        subset_name: str,
        /,
    ) -> EntitySelection:
        return cls(entities, entities.subset(subset_name).mask)

    def _binary(
        self,
        other: EntitySelection,
        operation: str,
        /,
    ) -> EntitySelection:
        if not isinstance(other, EntitySelection):
            raise TypeError("Selection operands must be EntitySelection values.")
        if self.entity_set_id != other.entity_set_id:
            raise ValueError("Entity selections must share one entity set.")
        if operation == "union":
            mask = self.mask | other.mask
        elif operation == "intersection":
            mask = self.mask & other.mask
        elif operation == "difference":
            mask = self.mask & ~other.mask
        else:
            raise ValueError("Unknown entity selection operation.")
        return _selection_from_masks(
            self.entity_set_id,
            mask,
            self.active_mask,
            operation,
            (self.selection_id, other.selection_id),
        )

    def union(self, other: EntitySelection, /) -> EntitySelection:
        return self._binary(other, "union")

    def intersection(self, other: EntitySelection, /) -> EntitySelection:
        return self._binary(other, "intersection")

    def difference(self, other: EntitySelection, /) -> EntitySelection:
        return self._binary(other, "difference")

    def complement(self, /) -> EntitySelection:
        return _selection_from_masks(
            self.entity_set_id,
            self.active_mask & ~self.mask,
            self.active_mask,
            "complement",
            (self.selection_id,),
        )


def _selection_from_masks(
    entity_set_id: str,
    mask: ArrayLike,
    active_mask: ArrayLike,
    operation: str,
    operands: tuple[str, ...],
    /,
) -> EntitySelection:
    return EntitySelection(
        entity_set_id,
        mask,
        active_mask=active_mask,
        selection_id=canonical_fingerprint(
            {
                "kind": "composed-entity-selection",
                "entity_set": entity_set_id,
                "operation": operation,
                "operands": list(operands),
                "mask": _array_digest(mask),
            }
        ),
    )


class OrientedIncidence(StrictModule, NonTrainableState):
    """Sparse signed boundary incidence from lower to upper entities."""

    degree: int = eqx.field(static=True)
    lower_entity_set_id: str = eqx.field(static=True)
    upper_entity_set_id: str = eqx.field(static=True)
    relation: EdgeRelation
    signs: Array
    incidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        lower: EntitySet,
        upper: EntitySet,
        relation: EdgeRelation,
        signs: ArrayLike,
        /,
        *,
        incidence_id: str | None = None,
    ):
        degree_ = int(degree)
        if degree_ <= 0:
            raise ValueError("Incidence degree must be positive.")
        if lower.intrinsic_dimension != degree_ - 1:
            raise ValueError("Lower entity dimension must equal degree - 1.")
        if upper.intrinsic_dimension != degree_:
            raise ValueError("Upper entity dimension must equal degree.")
        if not isinstance(relation, EdgeRelation):
            raise TypeError("Oriented incidence requires an EdgeRelation.")
        if relation.source_size != lower.count or relation.target_size != upper.count:
            raise ValueError("Incidence relation sizes must match the entity sets.")
        coefficients = np.asarray(signs)
        if coefficients.shape != relation.route_shape:
            raise ValueError(
                f"Incidence signs must have shape {relation.route_shape}; "
                f"got {coefficients.shape}."
            )
        valid = np.asarray(relation.valid, dtype=bool)
        active_coefficients = coefficients[valid]
        if np.any(~np.isfinite(active_coefficients)):
            raise ValueError("Active incidence signs must be finite.")
        if np.any(np.abs(active_coefficients) != 1):
            raise ValueError("Active incidence signs must be ±1.")
        source = np.asarray(relation.source_indices)[valid]
        target = np.asarray(relation.target_indices)[valid]
        pairs = np.stack((source, target), axis=1) if source.size else np.empty((0, 2))
        if pairs.shape[0] and np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError("Active incidence pairs must be unique.")
        lower_ids = np.asarray(lower.entity_ids, dtype=np.int64)[source]
        upper_ids = np.asarray(upper.entity_ids, dtype=np.int64)[target]
        canonical_incidence = np.stack(
            (lower_ids, upper_ids, active_coefficients.astype(np.int64)),
            axis=1,
        )
        if canonical_incidence.shape[0]:
            order = np.lexsort(
                (
                    canonical_incidence[:, 2],
                    canonical_incidence[:, 1],
                    canonical_incidence[:, 0],
                )
            )
            canonical_incidence = canonical_incidence[order]
        self.degree = degree_
        self.lower_entity_set_id = lower.entity_set_id
        self.upper_entity_set_id = upper.entity_set_id
        self.relation = relation
        self.signs = jnp.asarray(coefficients, dtype=float)
        self.incidence_id = resolved_identifier(
            "incidence_id",
            incidence_id,
            {
                "kind": "oriented-incidence",
                "degree": degree_,
                "lower": lower.entity_set_id,
                "upper": upper.entity_set_id,
                "canonical_incidence": _array_digest(canonical_incidence),
            },
        )

    def exterior_derivative(self, /) -> SparseLinearMap:
        """Return the lower-to-upper transpose-boundary action."""
        return SparseLinearMap(
            self.relation,
            self.signs,
            operator_id=canonical_fingerprint(
                {"kind": "exterior-derivative", "incidence": self.incidence_id}
            ),
        )

    def boundary(self, /) -> SparseLinearMap:
        """Return the upper-to-lower boundary action."""
        return SparseLinearMap(
            self.relation.transpose(),
            self.signs,
            operator_id=canonical_fingerprint(
                {"kind": "boundary", "incidence": self.incidence_id}
            ),
        )

    def scipy_boundary(self, /) -> sp.csr_matrix:
        """Return a host-side sparse lower-by-upper boundary matrix."""
        valid = np.asarray(self.relation.valid, dtype=bool)
        return sp.coo_matrix(
            (
                np.asarray(self.signs)[valid],
                (
                    np.asarray(self.relation.source_indices)[valid],
                    np.asarray(self.relation.target_indices)[valid],
                ),
            ),
            shape=(self.relation.source_size, self.relation.target_size),
        ).tocsr()


class TensorTopology(StrictModule, NonTrainableState):
    """Implicit tensor-product topology without materialized graph routes."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    axis_sizes: tuple[int, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    active_mask: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        axis_sizes: Sequence[int],
        /,
        *,
        periodic: Sequence[bool] | None = None,
        active_mask: ArrayLike | None = None,
        topology_id: str | None = None,
    ):
        names = tuple(str(name) for name in axis_names)
        sizes = tuple(int(size) for size in axis_sizes)
        if not names or any(not name for name in names):
            raise ValueError("Tensor topology requires non-empty axis names.")
        if len(set(names)) != len(names):
            raise ValueError("Tensor topology axis names must be unique.")
        if len(sizes) != len(names) or any(size <= 0 for size in sizes):
            raise ValueError("Tensor topology requires one positive size per axis.")
        periodic_ = (
            (False,) * len(names)
            if periodic is None
            else tuple(bool(value) for value in periodic)
        )
        if len(periodic_) != len(names):
            raise ValueError("periodic must provide one value per axis.")
        active = (
            np.ones(sizes, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != sizes:
            raise ValueError(f"active_mask must have shape {sizes}; got {active.shape}.")
        if not np.any(active):
            raise ValueError("Tensor topology requires at least one active site.")
        self.axis_names = names
        self.axis_sizes = sizes
        self.periodic = periodic_
        self.active_mask = jnp.asarray(active)
        self.topology_id = resolved_identifier(
            "topology_id",
            topology_id,
            {
                "kind": "tensor-topology",
                "axis_names": list(names),
                "axis_sizes": list(sizes),
                "periodic": list(periodic_),
                "active_mask": _array_digest(active),
            },
        )


class CellComplexTopology(StrictModule, NonTrainableState):
    """Validated oriented finite cell complex with one entity set per degree."""

    entity_sets: tuple[EntitySet, ...]
    incidences: tuple[OrientedIncidence, ...]
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        entity_sets: Sequence[EntitySet],
        incidences: Sequence[OrientedIncidence],
        /,
        *,
        topology_id: str | None = None,
        validate: bool = True,
    ):
        entities = tuple(entity_sets)
        if not entities or not all(isinstance(value, EntitySet) for value in entities):
            raise TypeError("entity_sets must contain one or more EntitySet values.")
        dimensions = tuple(value.intrinsic_dimension for value in entities)
        if dimensions != tuple(range(len(entities))):
            raise ValueError(
                "Cell-complex entity sets must cover contiguous dimensions from 0."
            )
        incidences_ = tuple(incidences)
        if len(incidences_) != len(entities) - 1:
            raise ValueError("One incidence is required between each consecutive degree.")
        for degree, incidence in enumerate(incidences_, start=1):
            if not isinstance(incidence, OrientedIncidence):
                raise TypeError("incidences must contain OrientedIncidence values.")
            if incidence.degree != degree:
                raise ValueError("Incidence degrees must be contiguous and ordered.")
            if (
                incidence.lower_entity_set_id != entities[degree - 1].entity_set_id
                or incidence.upper_entity_set_id != entities[degree].entity_set_id
            ):
                raise ValueError(
                    "Incidence endpoints must match consecutive entity sets."
                )
        if validate:
            self._validate_chain(incidences_)
        self.entity_sets = entities
        self.incidences = incidences_
        self.topology_id = resolved_identifier(
            "topology_id",
            topology_id,
            {
                "kind": "cell-complex-topology",
                "entity_sets": [value.entity_set_id for value in entities],
                "incidences": [value.incidence_id for value in incidences_],
            },
        )

    @staticmethod
    def _validate_chain(incidences: tuple[OrientedIncidence, ...], /) -> None:
        for lower, upper in zip(incidences[:-1], incidences[1:], strict=True):
            product = lower.scipy_boundary() @ upper.scipy_boundary()
            product.eliminate_zeros()
            if product.nnz:
                raise ValueError(
                    "Cell-complex incidences violate boundary-of-boundary zero."
                )

    @property
    def dimension(self) -> int:
        return len(self.entity_sets) - 1

    def entities(self, degree: int, /) -> EntitySet:
        index = int(degree)
        if index < 0 or index >= len(self.entity_sets):
            raise ValueError(f"degree must lie in [0, {self.dimension}].")
        return self.entity_sets[index]


class PointTopology(StrictModule, NonTrainableState):
    """Point support with an optional fixed-capacity neighborhood relation."""

    points: EntitySet
    neighborhoods: SparseRelation | None
    refreshable_neighborhoods: bool = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: EntitySet,
        /,
        *,
        neighborhoods: SparseRelation | None = None,
        refreshable_neighborhoods: bool = False,
        topology_id: str | None = None,
    ):
        if not isinstance(points, EntitySet) or points.intrinsic_dimension != 0:
            raise TypeError("points must be a zero-dimensional EntitySet.")
        if neighborhoods is not None:
            if not isinstance(neighborhoods, (EdgeRelation, RowRelation)):
                raise TypeError(
                    "neighborhoods must be an EdgeRelation, RowRelation, or None."
                )
            if neighborhoods.source_size != points.count:
                raise ValueError("Neighborhood sources must index the point entity set.")
            if (
                isinstance(neighborhoods, EdgeRelation)
                and neighborhoods.target_size != points.count
            ):
                raise ValueError(
                    "Edge neighborhoods must target the same point entity set."
                )
        neighborhood_payload = None
        if neighborhoods is not None:
            neighborhood_payload = {
                "source": _array_digest(neighborhoods.source_indices),
                "valid": _array_digest(neighborhoods.valid),
                "source_size": neighborhoods.source_size,
                "output_shape": list(neighborhoods.output_shape),
            }
            if isinstance(neighborhoods, EdgeRelation):
                neighborhood_payload["target"] = _array_digest(
                    neighborhoods.target_indices
                )
        self.points = points
        self.neighborhoods = neighborhoods
        self.refreshable_neighborhoods = bool(refreshable_neighborhoods)
        self.topology_id = resolved_identifier(
            "topology_id",
            topology_id,
            {
                "kind": "point-topology",
                "points": points.entity_set_id,
                "neighborhoods": neighborhood_payload,
                "refreshable": bool(refreshable_neighborhoods),
            },
        )


DiscreteTopology: TypeAlias = TensorTopology | CellComplexTopology | PointTopology


__all__ = [
    "CellComplexTopology",
    "DiscreteTopology",
    "EntitySet",
    "EntitySubset",
    "OrientedIncidence",
    "PointTopology",
    "TensorTopology",
]
