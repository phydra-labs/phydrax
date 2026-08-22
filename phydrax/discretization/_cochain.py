#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import ArraySpace, DiagonalPairing
from ._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from ._lifecycle import AbstractPreparedDiscretization, validate_prepared_metadata
from ._measure import DiscreteMeasure
from ._spaces import DiscreteFieldSpace, EntityDofLayout
from ._support import DiscreteSupport
from ._topology import CellComplexTopology


CochainBoundaryKind: TypeAlias = Literal["absolute", "relative"]

CochainSide: TypeAlias = Literal["primal", "dual"]
CochainCellOrientation: TypeAlias = Literal["invariant", "signed"]
CochainSampling: TypeAlias = Literal[
    "point_value",
    "cell_average",
    "cell_integral",
]


class CochainFieldSpec(StrictModule, NonTrainableState):
    """Discrete differential-form semantics shared by fields and operators."""

    degree: int = eqx.field(static=True)
    complex_side: CochainSide = eqx.field(static=True)
    cell_orientation: CochainCellOrientation = eqx.field(static=True)
    sampling: CochainSampling = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        complex_side: CochainSide = "primal",
        cell_orientation: CochainCellOrientation,
        sampling: CochainSampling,
    ):
        resolved_degree = int(degree)
        if resolved_degree < 0:
            raise ValueError("Cochain degree must be non-negative.")
        if complex_side not in ("primal", "dual"):
            raise ValueError("complex_side must be 'primal' or 'dual'.")
        if cell_orientation not in ("invariant", "signed"):
            raise ValueError("cell_orientation must be 'invariant' or 'signed'.")
        if sampling not in ("point_value", "cell_average", "cell_integral"):
            raise ValueError(
                "sampling must be 'point_value', 'cell_average', or 'cell_integral'."
            )
        self.degree = resolved_degree
        self.complex_side = complex_side
        self.cell_orientation = cell_orientation
        self.sampling = sampling

    def to_dict(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "complex_side": self.complex_side,
            "cell_orientation": self.cell_orientation,
            "sampling": self.sampling,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "CochainFieldSpec":
        return cls(
            int(value["degree"]),
            complex_side=value.get("complex_side", "primal"),
            cell_orientation=value["cell_orientation"],
            sampling=value["sampling"],
        )


class CochainBoundaryPolicy(StrictModule, NonTrainableState):
    """Boundary realization used by metric cochain operators."""

    kind: CochainBoundaryKind = eqx.field(static=True)

    def __init__(self, kind: CochainBoundaryKind = "absolute"):
        if kind not in ("absolute", "relative"):
            raise ValueError("Cochain boundary policy must be 'absolute' or 'relative'.")
        self.kind = kind

    @property
    def code(self) -> int:
        return 0 if self.kind == "absolute" else 1


class CochainDiscretization(AbstractPreparedDiscretization):
    """Prepared metric cochain spaces over one canonical oriented cell complex."""

    topology: CellComplexTopology
    hodge_stars: tuple[Array, ...]
    primal_measures: tuple[Array, ...]
    dual_measures: tuple[Array, ...]
    boundary_masks: tuple[Array, ...]
    coordinates: tuple[Array | None, ...]
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    def __init__(
        self,
        topology: CellComplexTopology,
        hodge_stars: Sequence[ArrayLike],
        /,
        *,
        primal_measures: Sequence[ArrayLike] | None = None,
        dual_measures: Sequence[ArrayLike] | None = None,
        boundary_masks: Sequence[ArrayLike] | None = None,
        coordinates: Sequence[ArrayLike | None] | None = None,
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
        numeric_version: str = "0",
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be a CellComplexTopology.")
        counts = tuple(entity_set.count for entity_set in topology.entity_sets)
        stars = self._degree_arrays("hodge_stars", hodge_stars, counts, positive=True)
        primal = self._degree_arrays(
            "primal_measures",
            tuple(np.ones((count,)) for count in counts)
            if primal_measures is None
            else primal_measures,
            counts,
            positive=True,
        )
        dual = self._degree_arrays(
            "dual_measures",
            tuple(
                np.asarray(primal_value) * np.asarray(star)
                for primal_value, star in zip(primal, stars, strict=True)
            )
            if dual_measures is None
            else dual_measures,
            counts,
            positive=True,
        )
        boundaries = self._degree_arrays(
            "boundary_masks",
            tuple(np.zeros((count,), dtype=bool) for count in counts)
            if boundary_masks is None
            else boundary_masks,
            counts,
            dtype=bool,
        )
        coordinates_ = self._coordinates(coordinates, counts)
        key_ = (
            DiscretizationKey(
                "cochain",
                DiscretizationRole.PHYSICAL,
                domain_labels=("entity",),
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        embedding_payload = {
            "kind": "cochain-embedding",
            "coordinates": [
                None if value is None else array_tree_fingerprint(value)
                for value in coordinates_
            ],
        }
        dimensions = {int(value.shape[1]) for value in coordinates_ if value is not None}
        ambient_dimension = (
            next(iter(dimensions)) if dimensions else max(1, topology.dimension)
        )
        support = DiscreteSupport(
            topology,
            ambient_dimension,
            canonical_fingerprint(embedding_payload),
        )
        spaces = []
        measures = []
        for degree, (entity_set, star, primal_value) in enumerate(
            zip(topology.entity_sets, stars, primal, strict=True)
        ):
            layout = EntityDofLayout(
                entity_set.entity_set_id,
                entity_set.count,
                entity_set.count,
            )
            pairing = DiagonalPairing(star)
            spaces.append(
                DiscreteFieldSpace(
                    f"cochain_{degree}",
                    support.support_id,
                    layout,
                    ArraySpace(
                        (entity_set.count,),
                        pairing=pairing,
                        space_id=canonical_fingerprint(
                            {
                                "kind": "cochain-coordinate-space",
                                "support": support.support_id,
                                "degree": degree,
                            }
                        ),
                    ),
                    representation="cochain",
                    conformity="cochain",
                )
            )
            measures.append(
                DiscreteMeasure(
                    f"primal_{degree}",
                    support.support_id,
                    entity_set.entity_set_id,
                    primal_value,
                    normalization="physical",
                )
            )
        capabilities = (
            DiscretizationCapability.ENTITY_INCIDENCE,
            DiscretizationCapability.STRONG_DERIVATIVE,
            DiscretizationCapability.SPECTRAL_TRANSFORM,
            DiscretizationCapability.SPARSE_ASSEMBLY,
        )
        preparation = PreparationReport(
            capabilities=capabilities,
            resource_counts={
                "degrees": len(counts),
                "entities": sum(counts),
                "incidences": sum(
                    int(np.count_nonzero(np.asarray(incidence.relation.valid)))
                    for incidence in topology.incidences
                ),
            },
        )
        spaces_, measures_, capabilities_ = validate_prepared_metadata(
            key=key_,
            support=support,
            field_spaces=spaces,
            measures=measures,
            capabilities=capabilities,
            preparation=preparation,
        )
        plan_identifier = (
            canonical_fingerprint(
                {
                    "kind": "cochain-plan",
                    "topology": topology.topology_id,
                    "key": key_.key_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        version = str(numeric_version)
        if not plan_identifier or not version:
            raise ValueError("plan_id and numeric_version must be non-empty.")
        prepared_identifier = canonical_fingerprint(
            {
                "kind": "prepared-cochain",
                "plan": plan_identifier,
                "topology": topology.topology_id,
                "hodge": [array_tree_fingerprint(value) for value in stars],
                "primal": [array_tree_fingerprint(value) for value in primal],
                "dual": [array_tree_fingerprint(value) for value in dual],
                "boundary": [array_tree_fingerprint(value) for value in boundaries],
                "embedding": support.embedding_id,
            }
        )
        self.topology = topology
        self.hodge_stars = stars
        self.primal_measures = primal
        self.dual_measures = dual
        self.boundary_masks = boundaries
        self.coordinates = coordinates_
        self.key = key_
        self.support = support
        self.field_spaces = spaces_
        self.measures = measures_
        self.capabilities = capabilities_
        self.plan_id = plan_identifier
        self.prepared_id = prepared_identifier
        self.numeric_version = version
        self.preparation = preparation

    @staticmethod
    def _degree_arrays(
        name: str,
        values: Sequence[ArrayLike],
        counts: tuple[int, ...],
        /,
        *,
        positive: bool = False,
        dtype: Any = float,
    ) -> tuple[Array, ...]:
        resolved = tuple(values)
        if len(resolved) != len(counts):
            raise ValueError(f"{name} must provide one array per cochain degree.")
        arrays = []
        for degree, (value, count) in enumerate(zip(resolved, counts, strict=True)):
            array = np.asarray(value, dtype=dtype)
            if array.shape != (count,):
                raise ValueError(f"{name}[{degree}] must have shape ({count},).")
            if np.any(~np.isfinite(array)):
                raise ValueError(f"{name}[{degree}] must be finite.")
            if positive and np.any(array <= 0):
                raise ValueError(f"{name}[{degree}] must be strictly positive.")
            arrays.append(jnp.asarray(array))
        return tuple(arrays)

    @staticmethod
    def _coordinates(
        values: Sequence[ArrayLike | None] | None,
        counts: tuple[int, ...],
        /,
    ) -> tuple[Array | None, ...]:
        resolved = (None,) * len(counts) if values is None else tuple(values)
        if len(resolved) != len(counts):
            raise ValueError("coordinates must provide one entry per cochain degree.")
        arrays: list[Array | None] = []
        dimensions = set()
        for degree, (value, count) in enumerate(zip(resolved, counts, strict=True)):
            if value is None:
                arrays.append(None)
                continue
            array = np.asarray(value, dtype=float)
            if array.ndim != 2 or array.shape[0] != count or np.any(~np.isfinite(array)):
                raise ValueError(
                    f"coordinates[{degree}] must be finite with leading size {count}."
                )
            dimensions.add(int(array.shape[1]))
            arrays.append(jnp.asarray(array))
        if len(dimensions) > 1 or (dimensions and any(value is None for value in arrays)):
            raise ValueError(
                "Coordinates must be present with one ambient dimension at every degree."
            )
        return tuple(arrays)

    @property
    def max_degree(self) -> int:
        return self.topology.dimension

    @property
    def cell_counts(self) -> tuple[int, ...]:
        return tuple(entity_set.count for entity_set in self.topology.entity_sets)

    def space(self, degree: int, /) -> DiscreteFieldSpace:
        value = int(degree)
        if value < 0 or value > self.max_degree:
            raise ValueError(f"degree must lie in [0, {self.max_degree}].")
        return self.field_spaces[value]

    def active_mask(
        self,
        degree: int,
        boundary_policy: CochainBoundaryKind = "absolute",
        /,
    ) -> Array:
        value = int(degree)
        if value < 0 or value > self.max_degree:
            raise ValueError(f"degree must lie in [0, {self.max_degree}].")
        if boundary_policy == "absolute":
            return jnp.ones((self.cell_counts[value],), dtype=bool)
        if boundary_policy == "relative":
            return ~self.boundary_masks[value]
        raise ValueError("boundary_policy must be 'absolute' or 'relative'.")


__all__ = [
    "CochainBoundaryKind",
    "CochainBoundaryPolicy",
    "CochainCellOrientation",
    "CochainDiscretization",
    "CochainFieldSpec",
    "CochainSampling",
    "CochainSide",
]
