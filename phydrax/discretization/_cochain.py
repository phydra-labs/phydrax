#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core, ensure_compile_time_eval
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..exterior._complex import ComplexBoundary
from ..exterior._form_type import FormTwist, FormType
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    apply_real_map_componentwise,
    ArraySpace,
)
from ..linalg._complexes import (
    HarmonicSubspace,
    HilbertComplex,
    hodge_decomposition,
    HodgeDecomposition,
    HodgeDecompositionPolicy,
    HodgeLaplacianPart,
)
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ..sparse._linear import _SparseStoragePlan
from ..typing import parse
from ._cell_de_rham import AbstractCellDeRhamComplex
from ._cochain_hodge import CochainHodge, DiagonalHodge, SparseHodge
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


def _degree_arrays(
    name: str, values: Sequence[ArrayLike], counts: tuple[int, ...], /
) -> tuple[Array, ...]:
    resolved = tuple(values)
    if len(resolved) != len(counts):
        raise ValueError(f"{name} must provide one array per degree.")
    arrays = tuple(jnp.asarray(value, dtype=jnp.float64) for value in resolved)
    for degree, (array, count) in enumerate(zip(arrays, counts, strict=True)):
        if array.shape != (count,):
            raise ValueError(f"{name}[{degree}] must have shape ({count},).")
        if not isinstance(array, jax_core.Tracer):
            host = np.asarray(array)
            if np.any(~np.isfinite(host)) or np.any(host <= 0):
                raise ValueError(f"{name}[{degree}] must be finite and positive.")
    return arrays


def _coordinate_arrays(
    values: Sequence[ArrayLike | None] | None, counts: tuple[int, ...], /
) -> tuple[Array | None, ...]:
    resolved = (None,) * len(counts) if values is None else tuple(values)
    if len(resolved) != len(counts):
        raise ValueError("coordinates must provide one entry per degree.")
    arrays = tuple(
        None if value is None else jnp.asarray(value, dtype=jnp.float64)
        for value in resolved
    )
    dimensions: set[int] = set()
    for degree, (array, count) in enumerate(zip(arrays, counts, strict=True)):
        if array is None:
            continue
        if array.ndim != 2 or array.shape[0] != count:
            raise ValueError(f"coordinates[{degree}] must have leading size {count}.")
        dimensions.add(array.shape[1])
        if not isinstance(array, jax_core.Tracer) and np.any(
            ~np.isfinite(np.asarray(array))
        ):
            raise ValueError("Coordinates must be finite.")
    if len(dimensions) > 1 or (dimensions and any(value is None for value in arrays)):
        raise ValueError(
            "Coordinates must be present at every degree with one ambient dimension."
        )
    return arrays


def _metric_values(hodge: CochainHodge, /) -> Array:
    return hodge.weights if isinstance(hodge, DiagonalHodge) else hodge.upper_values


def _binding_id(plan_id: str, revision: str, /) -> str:
    return canonical_fingerprint(
        {"kind": "prepared-cochain", "plan": plan_id, "numeric_revision": revision}
    )


def _bind_differentials(
    operators: tuple[AbstractLinearOperator, ...],
    spaces: tuple[AbstractVectorSpace, ...],
    /,
) -> tuple[AbstractLinearOperator, ...]:
    return tuple(
        eqx.tree_at(
            lambda operator: (operator.source, operator.target),
            operator,
            (spaces[degree], spaces[degree + 1]),
        )
        for degree, operator in enumerate(operators)
    )


def _admit_layout(
    topology: CellComplexTopology,
    hodges: Sequence[CochainHodge],
    boundary_masks: Sequence[ArrayLike] | None,
    /,
) -> tuple[
    tuple[int, ...],
    tuple[CochainHodge, ...],
    tuple[np.ndarray, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    if not isinstance(topology, CellComplexTopology):
        raise TypeError("topology must be a CellComplexTopology.")
    counts = tuple(entity.count for entity in topology.entity_sets)
    metrics = tuple(hodges)
    if len(metrics) != len(counts) or not all(
        isinstance(hodge, (DiagonalHodge, SparseHodge)) for hodge in metrics
    ):
        raise TypeError(
            "hodges must provide one DiagonalHodge or SparseHodge per degree."
        )
    if any(hodge.size != count for hodge, count in zip(metrics, counts, strict=True)):
        raise ValueError("Hodge dimensions must match topology entity counts.")
    boundaries = (
        tuple(np.zeros((count,), dtype=np.bool_) for count in counts)
        if boundary_masks is None
        else tuple(np.asarray(mask, dtype=np.bool_) for mask in boundary_masks)
    )
    if len(boundaries) != len(counts) or any(
        mask.shape != (count,) for mask, count in zip(boundaries, counts, strict=True)
    ):
        raise ValueError("Boundary masks must match topology entity counts.")
    entity_masks = tuple(
        np.asarray(entity.active_mask, dtype=np.bool_) for entity in topology.entity_sets
    )
    absolute = tuple(tuple(np.flatnonzero(mask).tolist()) for mask in entity_masks)
    relative = tuple(
        tuple(np.flatnonzero(mask & ~boundary).tolist())
        for mask, boundary in zip(entity_masks, boundaries, strict=True)
    )
    return counts, metrics, boundaries, absolute, relative


def _plan_identity(
    topology: CellComplexTopology,
    metrics: tuple[CochainHodge, ...],
    boundaries: tuple[np.ndarray, ...],
    coordinates: tuple[Array | None, ...],
    key: DiscretizationKey,
    differentials: Sequence[AbstractLinearOperator] | None,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "cochain-plan",
            "topology": topology.topology_id,
            "key": key.key_id,
            "boundary": [array_tree_fingerprint(mask) for mask in boundaries],
            "hodges": [hodge.layout_id for hodge in metrics],
            "coordinate_shapes": [
                None if value is None else value.shape for value in coordinates
            ],
            "differentials": None
            if differentials is None
            else [operator.operator_id for operator in differentials],
        }
    )


def _resolve_numeric_binding(
    numerical_leaves: tuple[Array, ...], numeric_revision: str | None, /
) -> str:
    if numeric_revision is not None:
        if not isinstance(numeric_revision, str) or not numeric_revision:
            raise ValueError("numeric_revision must be a non-empty binding identity.")
        return numeric_revision
    if any(isinstance(value, jax_core.Tracer) for value in numerical_leaves):
        raise ValueError(
            "Traced numerical cochain values require an explicit numeric_revision binding."
        )
    return canonical_fingerprint(
        {
            "kind": "cochain-numeric-snapshot",
            "values": [array_tree_fingerprint(value) for value in numerical_leaves],
        }
    )


def _prepare_embedding(
    topology: CellComplexTopology,
    coordinates: tuple[Array | None, ...],
    revision: str,
    explicit_binding: bool,
    /,
) -> DiscreteSupport:
    dimensions = {value.shape[1] for value in coordinates if value is not None}
    ambient = next(iter(dimensions)) if dimensions else max(1, topology.dimension)
    if explicit_binding:
        embedding = canonical_fingerprint(
            {
                "kind": "cochain-embedding",
                "binding": revision,
                "shapes": [
                    None if value is None else value.shape for value in coordinates
                ],
            }
        )
    else:
        embedding = canonical_fingerprint(
            {
                "kind": "cochain-embedding",
                "coordinates": [
                    None if value is None else array_tree_fingerprint(value)
                    for value in coordinates
                ],
            }
        )
    return DiscreteSupport(topology, ambient, embedding)


def _prepare_identity(
    topology: CellComplexTopology,
    metrics: tuple[CochainHodge, ...],
    boundaries: tuple[np.ndarray, ...],
    coordinates: tuple[Array | None, ...],
    numerical_leaves: tuple[Array, ...],
    key: DiscretizationKey | None,
    numeric_revision: str | None,
    differentials: Sequence[AbstractLinearOperator] | None,
    /,
) -> tuple[DiscretizationKey, str, str, str, DiscreteSupport]:
    key_ = (
        DiscretizationKey(
            "cochain", DiscretizationRole.PHYSICAL, domain_labels=("entity",)
        )
        if key is None
        else key
    )
    if not isinstance(key_, DiscretizationKey):
        raise TypeError("key must be a DiscretizationKey.")
    plan = _plan_identity(topology, metrics, boundaries, coordinates, key_, differentials)
    revision = _resolve_numeric_binding(numerical_leaves, numeric_revision)
    prepared = _binding_id(plan, revision)
    support = _prepare_embedding(
        topology, coordinates, revision, numeric_revision is not None
    )
    return key_, plan, revision, prepared, support


def _metric_spaces(
    metrics: tuple[CochainHodge, ...],
    active: tuple[tuple[int, ...], ...],
    candidates: tuple[AbstractVectorSpace, ...],
    prepared_id: str,
    boundary: ComplexBoundary,
    /,
) -> tuple[AbstractVectorSpace, ...]:
    # Relative indices are a subset of absolute indices; equal extents mean the
    # same ordered coordinates. Reuse their numerical preparation exactly.
    return tuple(
        candidate
        if candidate.size == len(indices)
        else hodge.restrict(np.asarray(indices, dtype=np.int32)).make_space(
            space_id=f"{prepared_id}:{boundary}:{degree}"
        )[0]
        for degree, (hodge, indices, candidate) in enumerate(
            zip(metrics, active, candidates, strict=True)
        )
    )


def _prepare_complex(
    topology: CellComplexTopology,
    metrics: tuple[CochainHodge, ...],
    active: tuple[tuple[int, ...], ...],
    prepared_id: str,
    boundary: ComplexBoundary,
    candidates: tuple[AbstractVectorSpace, ...],
    /,
) -> HilbertComplex:
    spaces = _metric_spaces(metrics, active, candidates, prepared_id, boundary)
    operators: list[AbstractLinearOperator] = []
    for degree, incidence in enumerate(topology.incidences):
        relation = incidence.relation
        lower_map = np.full((relation.source_size,), -1, dtype=np.int32)
        upper_map = np.full((relation.target_size,), -1, dtype=np.int32)
        lower_map[np.asarray(active[degree], dtype=np.int32)] = np.arange(
            len(active[degree]), dtype=np.int32
        )
        upper_map[np.asarray(active[degree + 1], dtype=np.int32)] = np.arange(
            len(active[degree + 1]), dtype=np.int32
        )
        lower = np.asarray(relation.source_indices)
        upper = np.asarray(relation.target_indices)
        valid = (
            np.asarray(relation.valid) & (lower_map[lower] >= 0) & (upper_map[upper] >= 0)
        )
        with ensure_compile_time_eval():
            restricted_relation = EdgeRelation(
                lower_map[lower[valid]],
                upper_map[upper[valid]],
                source_size=len(active[degree]),
                target_size=len(active[degree + 1]),
            )
            storage_plan = _SparseStoragePlan(restricted_relation)
        operators.append(
            SparseCoordinateOperator(
                restricted_relation,
                incidence.signs[jnp.asarray(np.flatnonzero(valid), dtype=jnp.int32)],
                source=spaces[degree],
                target=spaces[degree + 1],
                operator_id=f"{prepared_id}:{boundary}:d:{degree}",
                storage_plan=storage_plan,
            )
        )
    return HilbertComplex(
        spaces, tuple(operators), complex_id=f"{prepared_id}:{boundary}"
    )


def _admit_differentials(
    topology: CellComplexTopology,
    differentials: Sequence[AbstractLinearOperator] | None,
    spaces: tuple[ArraySpace, ...],
    absolute: HilbertComplex,
    /,
) -> HilbertComplex:
    if differentials is None:
        return absolute
    overrides = tuple(differentials)
    if len(overrides) != topology.dimension or not all(
        isinstance(operator, AbstractLinearOperator) for operator in overrides
    ):
        raise ValueError(
            "differentials must provide one linear operator per incidence degree."
        )
    for degree, (override, incidence) in enumerate(
        zip(overrides, topology.incidences, strict=True)
    ):
        if (
            override.source.size != spaces[degree].size
            or override.target.size != spaces[degree + 1].size
            or not override.capabilities.transpose
        ):
            raise ValueError(
                "Differential override has incompatible dimensions or no transpose."
            )
        probes = np.stack(
            (
                np.ones(spaces[degree].size, dtype=np.float64),
                np.cos(np.arange(spaces[degree].size, dtype=np.float64)),
                np.sin(np.arange(spaces[degree].size, dtype=np.float64)),
            )
        )
        canonical = incidence.exterior_derivative()
        for probe in probes:
            if not np.allclose(
                np.asarray(override.mv(jnp.asarray(probe))),
                np.asarray(canonical.mv(jnp.asarray(probe))),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    "Differential override disagrees with canonical incidence."
                )
    if any(
        space.size != active.size
        for space, active in zip(spaces, absolute.spaces, strict=True)
    ):
        return absolute
    return HilbertComplex(
        absolute.spaces,
        _bind_differentials(overrides, absolute.spaces),
        complex_id=absolute.complex_id,
    )


def _prepare_metadata(
    topology: CellComplexTopology,
    spaces: tuple[ArraySpace, ...],
    primal: tuple[Array, ...],
    key: DiscretizationKey,
    support: DiscreteSupport,
    prepared_id: str,
    active: tuple[tuple[int, ...], ...],
    metrics: tuple[CochainHodge, ...],
    /,
) -> tuple[
    tuple[DiscreteFieldSpace, ...],
    tuple[DiscreteMeasure, ...],
    tuple[DiscretizationCapability, ...],
    PreparationReport,
]:
    field_spaces = tuple(
        DiscreteFieldSpace(
            f"cochain_{degree}",
            support.support_id,
            EntityDofLayout(entity.entity_set_id, entity.count, entity.count),
            spaces[degree],
            representation="cochain",
            conformity="unrestricted",
            form_type=FormType(topology.dimension, degree, twist="untwisted"),
        )
        for degree, entity in enumerate(topology.entity_sets)
    )
    measures = tuple(
        DiscreteMeasure(
            f"primal_{degree}",
            support.support_id,
            entity.entity_set_id,
            primal[degree],
            normalization="physical",
            active_mask=entity.active_mask,
            measure_id=f"{prepared_id}:primal-measure:{degree}",
        )
        for degree, entity in enumerate(topology.entity_sets)
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
            "degrees": len(spaces),
            "entities": sum(space.size for space in spaces),
            "incidences": sum(
                incidence.relation.capacity for incidence in topology.incidences
            ),
            "relative_entities": sum(len(indices) for indices in active),
            "hodge_entries": sum(_metric_values(hodge).size for hodge in metrics),
        },
    )
    fields_, measures_, capabilities_ = validate_prepared_metadata(
        key=key,
        support=support,
        field_spaces=field_spaces,
        measures=measures,
        capabilities=capabilities,
        preparation=preparation,
    )
    return fields_, measures_, capabilities_, preparation


@final
class CochainDiscretization(AbstractPreparedDiscretization, AbstractCellDeRhamComplex):
    """Metric-only cell cochains with prepared absolute and relative complexes.

    Relative pairings are the principal restrictions R M R^T. Their inverse is
    solved in active coordinates, never a mask of the full inverse. Explicit
    numerical revisions identify a binding, not changing numerical leaf values;
    ``with_metric`` is therefore safe inside compiled scans.
    """

    topology: CellComplexTopology
    hodges: tuple[CochainHodge, ...]
    primal_measures: tuple[Array, ...]
    dual_measures: tuple[Array, ...]
    boundary_masks: tuple[Array, ...]
    coordinates: tuple[Array | None, ...]
    time: Array
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    numeric_revision: str = eqx.field(static=True)
    primal_twist: FormTwist = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    preparation: PreparationReport
    _active: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    _absolute_active: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    _absolute: HilbertComplex
    _relative: HilbertComplex
    _masses: tuple[AbstractLinearOperator, ...]
    _differential_overrides: tuple[AbstractLinearOperator, ...] | None

    def __init__(
        self,
        topology: CellComplexTopology,
        hodges: Sequence[CochainHodge],
        /,
        *,
        boundary_masks: Sequence[ArrayLike] | None = None,
        coordinates: Sequence[ArrayLike | None] | None = None,
        primal_measures: Sequence[ArrayLike] | None = None,
        dual_measures: Sequence[ArrayLike] | None = None,
        key: DiscretizationKey | None = None,
        numeric_revision: str | None = None,
        time: ArrayLike = 0.0,
        differentials: Sequence[AbstractLinearOperator] | None = None,
    ) -> None:
        counts, metrics, boundaries, absolute_active, active = _admit_layout(
            topology, hodges, boundary_masks
        )
        coords = _coordinate_arrays(coordinates, counts)
        primal = _degree_arrays(
            "primal_measures",
            tuple(jnp.ones((count,), dtype=jnp.float64) for count in counts)
            if primal_measures is None
            else primal_measures,
            counts,
        )
        dual = _degree_arrays(
            "dual_measures",
            tuple(
                primal_value * hodge.weights
                if isinstance(hodge, DiagonalHodge)
                else primal_value
                for primal_value, hodge in zip(primal, metrics, strict=True)
            )
            if dual_measures is None
            else dual_measures,
            counts,
        )
        leaves = (
            *tuple(_metric_values(hodge) for hodge in metrics),
            *primal,
            *dual,
            *tuple(value for value in coords if value is not None),
        )
        key_, plan, revision, prepared, support = _prepare_identity(
            topology,
            metrics,
            boundaries,
            coords,
            leaves,
            key,
            numeric_revision,
            differentials,
        )
        spaces_and_mass = tuple(
            hodge.make_space(space_id=f"{prepared}:degree:{degree}")
            for degree, hodge in enumerate(metrics)
        )
        spaces = tuple(pair[0] for pair in spaces_and_mass)
        masses = tuple(pair[1] for pair in spaces_and_mass)
        absolute = _admit_differentials(
            topology,
            differentials,
            spaces,
            _prepare_complex(
                topology, metrics, absolute_active, prepared, "absolute", spaces
            ),
        )
        relative = _prepare_complex(
            topology, metrics, active, prepared, "relative", absolute.spaces
        )
        fields_, measures_, capabilities_, preparation = _prepare_metadata(
            topology, spaces, primal, key_, support, prepared, active, metrics
        )
        time_ = jnp.asarray(time, dtype=jnp.float64)
        if time_.shape != ():
            raise ValueError("time must be a scalar.")
        self.topology = topology
        self.hodges = metrics
        self.primal_measures = primal
        self.dual_measures = dual
        self.boundary_masks = tuple(jnp.asarray(mask) for mask in boundaries)
        self.coordinates = coords
        self.time = time_
        self.key = key_
        self.support = support
        self.field_spaces = fields_
        self.measures = measures_
        self.capabilities = capabilities_
        self.plan_id = plan
        self.prepared_id = prepared
        self.numeric_version = revision
        self.numeric_revision = revision
        self.primal_twist = "untwisted"
        self.dimension = topology.dimension
        self.realization_id = prepared
        self.preparation = preparation
        self._active = active
        self._absolute_active = absolute_active
        self._absolute = absolute
        self._relative = relative
        self._masses = masses
        self._differential_overrides = (
            None if differentials is None else tuple(differentials)
        )

    @property
    def max_degree(self) -> int:
        return self.dimension

    @property
    def cell_counts(self) -> tuple[int, ...]:
        return tuple(entity.count for entity in self.topology.entity_sets)

    def _degree(self, degree: int, /) -> int:
        if degree < 0 or degree > self.dimension:
            raise ValueError(f"degree must lie in [0, {self.dimension}].")
        return degree

    def space(self, degree: int, /) -> DiscreteFieldSpace:
        return self.field_spaces[self._degree(degree)]

    def hilbert_complex(
        self, /, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        boundary = parse(boundary, ComplexBoundary, "boundary")
        return self._absolute if boundary == "absolute" else self._relative

    def active_indices(
        self, degree: int, /, *, boundary: ComplexBoundary = "relative"
    ) -> Array:
        degree = self._degree(degree)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        indices = (
            self._absolute_active[degree]
            if boundary == "absolute"
            else self._active[degree]
        )
        return jnp.asarray(indices, dtype=jnp.int32)

    def active_mask(
        self, degree: int, /, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        degree = self._degree(degree)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        active = self.topology.entity_sets[degree].active_mask
        return active if boundary == "absolute" else active & ~self.boundary_masks[degree]

    def _values(self, degree: int, values: ArrayLike, /) -> Array:
        degree = self._degree(degree)
        value = jnp.asarray(values)
        if value.shape != (self.cell_counts[degree],):
            raise ValueError(
                f"Degree-{degree} cochain must have shape ({self.cell_counts[degree]},)."
            )
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.float64)
        return value

    def _compact(self, degree: int, values: Array, boundary: ComplexBoundary, /) -> Array:
        indices = (
            self._absolute_active[degree]
            if boundary == "absolute"
            else self._active[degree]
        )
        return (
            values
            if len(indices) == values.size
            else values[jnp.asarray(indices, dtype=jnp.int32)]
        )

    def _extend(self, degree: int, values: Array, boundary: ComplexBoundary, /) -> Array:
        if values.size == self.cell_counts[degree]:
            return values
        return (
            jnp.zeros((self.cell_counts[degree],), dtype=values.dtype)
            .at[self.active_indices(degree, boundary=boundary)]
            .set(values)
        )

    def exterior_derivative(
        self, degree: int, values: ArrayLike, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        value = self._values(degree, values)
        complex_ = self.hilbert_complex(boundary=boundary)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        if degree >= self.dimension:
            raise ValueError("Exterior derivative degree must be below dimension.")
        active = self._compact(degree, value, boundary)
        output = apply_real_map_componentwise(complex_.differential(degree).mv, active)
        return self._extend(degree + 1, output, boundary)

    def hodge_operator(self, degree: int, /) -> AbstractLinearOperator:
        return self._masses[self._degree(degree)]

    def hodge_diagonal(self, degree: int, /) -> Array:
        hodge = self.hodges[self._degree(degree)]
        if not isinstance(hodge, DiagonalHodge):
            raise TypeError("hodge_diagonal requires a DiagonalHodge.")
        return hodge.weights

    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        value = self._values(degree, values)
        return apply_real_map_componentwise(
            self.field_spaces[degree].vector_space.riesz, value
        )

    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        value = self._values(degree, values)
        return apply_real_map_componentwise(
            self.field_spaces[degree].vector_space.inverse_riesz, value
        )

    def codifferential(
        self, degree: int, values: ArrayLike, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        value = self._values(degree, values)
        complex_ = self.hilbert_complex(boundary=boundary)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        if degree <= 0:
            raise ValueError("Codifferential degree must be positive.")
        active = self._compact(degree, value, boundary)
        weighted = apply_real_map_componentwise(complex_.space(degree).riesz, active)
        transposed = apply_real_map_componentwise(
            complex_.differential(degree - 1).transpose_mv, weighted
        )
        output = apply_real_map_componentwise(
            complex_.space(degree - 1).inverse_riesz, transposed
        )
        return self._extend(degree - 1, output, boundary)

    def dual_exterior_derivative(self, degree: int, values: ArrayLike, /) -> Array:
        value = self._values(degree, values)
        if degree <= 0:
            raise ValueError(
                "Dual exterior derivative requires a positive primal degree."
            )
        active = value[self.active_indices(degree, boundary="absolute")]
        output = apply_real_map_componentwise(
            self._absolute.differential(degree - 1).transpose_mv, active
        )
        return (-1) ** degree * self._extend(degree - 1, output, "absolute")

    def hodge_laplacian(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        part: HodgeLaplacianPart = "complete",
    ) -> Array:
        value = self._values(degree, values)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        part = parse(part, HodgeLaplacianPart, "part")
        if (part == "lower" and degree == 0) or (
            part == "upper" and degree == self.dimension
        ):
            raise ValueError("Requested Hodge Laplacian part is absent at this degree.")
        result = jnp.zeros_like(value)
        if part != "lower" and degree < self.dimension:
            result = result + self.codifferential(
                degree + 1,
                self.exterior_derivative(degree, value, boundary=boundary),
                boundary=boundary,
            )
        if part != "upper" and degree > 0:
            result = result + self.exterior_derivative(
                degree - 1,
                self.codifferential(degree, value, boundary=boundary),
                boundary=boundary,
            )
        return result

    def hodge_decomposition(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        harmonic: HarmonicSubspace,
        lower_harmonic: HarmonicSubspace | None = None,
        policy: HodgeDecompositionPolicy | None = None,
    ) -> HodgeDecomposition:
        value = self._values(degree, values)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        result = hodge_decomposition(
            self.hilbert_complex(boundary=boundary),
            degree,
            self._compact(degree, value, boundary),
            harmonic=harmonic,
            lower_harmonic=lower_harmonic,
            policy=policy,
        )
        exact = self._extend(degree, jnp.asarray(result.exact), boundary)
        coexact = self._extend(degree, jnp.asarray(result.coexact), boundary)
        harmonic_value = self._extend(degree, jnp.asarray(result.harmonic), boundary)
        potential = (
            None
            if result.exact_potential is None
            else self._extend(degree - 1, jnp.asarray(result.exact_potential), boundary)
        )
        return eqx.tree_at(
            lambda decomposition: (
                decomposition.exact,
                decomposition.coexact,
                decomposition.harmonic,
                decomposition.exact_potential,
            ),
            result,
            (exact, coexact, harmonic_value, potential),
            is_leaf=lambda leaf: leaf is None,
        )

    @property
    def metric_valid(self) -> Array:
        return jnp.all(jnp.stack(tuple(hodge.valid for hodge in self.hodges)))

    def admit(self, /) -> CochainDiscretization:
        for hodge in self.hodges:
            hodge.admit()
        return self

    def with_metric(
        self, hodges: Sequence[CochainHodge], /, *, numeric_revision: str
    ) -> CochainDiscretization:
        metrics = tuple(hodges)
        if len(metrics) != len(self.hodges) or not all(
            isinstance(hodge, (DiagonalHodge, SparseHodge)) for hodge in metrics
        ):
            raise TypeError("hodges must provide one metric per degree.")
        if tuple(hodge.layout_id for hodge in metrics) != tuple(
            hodge.layout_id for hodge in self.hodges
        ):
            raise ValueError(
                "with_metric requires unchanged metric layout; prepare a new discretization for a new pattern."
            )
        if not isinstance(numeric_revision, str) or not numeric_revision:
            raise ValueError("numeric_revision must be a non-empty binding identity.")
        if numeric_revision != self.numeric_revision:
            return CochainDiscretization(
                self.topology,
                metrics,
                boundary_masks=self.boundary_masks,
                coordinates=self.coordinates,
                primal_measures=self.primal_measures,
                dual_measures=self.dual_measures,
                key=self.key,
                numeric_revision=numeric_revision,
                time=self.time,
                differentials=self._differential_overrides,
            )
        identifier = _binding_id(self.plan_id, numeric_revision)
        space_mass = tuple(
            hodge.make_space(space_id=f"{identifier}:degree:{degree}")
            for degree, hodge in enumerate(metrics)
        )
        spaces = tuple(pair[0] for pair in space_mass)
        absolute_spaces = _metric_spaces(
            metrics, self._absolute_active, spaces, identifier, "absolute"
        )
        relative_spaces = _metric_spaces(
            metrics, self._active, absolute_spaces, identifier, "relative"
        )
        absolute = HilbertComplex(
            absolute_spaces,
            _bind_differentials(self._absolute.differentials, absolute_spaces),
            complex_id=f"{identifier}:absolute",
        )
        relative = HilbertComplex(
            relative_spaces,
            _bind_differentials(self._relative.differentials, relative_spaces),
            complex_id=f"{identifier}:relative",
        )
        fields = tuple(
            eqx.tree_at(lambda field: field.vector_space, field, spaces[degree])
            for degree, field in enumerate(self.field_spaces)
        )
        return eqx.tree_at(
            lambda value: (
                value.hodges,
                value._absolute,
                value._relative,
                value._masses,
                value.field_spaces,
            ),
            self,
            (metrics, absolute, relative, tuple(pair[1] for pair in space_mass), fields),
        )


__all__ = ["CochainDiscretization"]
