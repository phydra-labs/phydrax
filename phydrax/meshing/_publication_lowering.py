#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity scientific publication joins before any owner-local construction."""

from __future__ import annotations

from functools import partial
from itertools import combinations
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometryStorageProjection,
    coordinate_lagrange_element,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ._organization import _organization_definition_id, MeshAttribute


if TYPE_CHECKING:
    from ._distribution import SimplexNeighborhoodWorkset
    from ._initial_certification import InitialCollectiveMeshEvidence
    from ._result import CellMeshingResult, CollectiveMeshEvidence


def _lookup(table: Array, queries: Array, count: int, /) -> tuple[Array, Array]:
    """Lexicographic lower bounds without query-by-global-bank dense products."""
    size = table.shape[0]
    low = jnp.zeros(queries.shape[:-1], dtype=jnp.int32)
    high = jnp.full(low.shape, count, dtype=jnp.int32)

    def step(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        left, right = bounds
        middle = (left + right) // 2
        row = table[jnp.minimum(middle, size - 1)]
        less = jnp.zeros(left.shape, dtype=jnp.bool_)
        equal = jnp.ones(left.shape, dtype=jnp.bool_)
        for column in range(table.shape[-1]):
            less |= equal & (row[..., column] < queries[..., column])
            equal &= row[..., column] == queries[..., column]
        running = left < right
        return jnp.where(running & less, middle + 1, left), jnp.where(
            running & ~less, middle, right
        )

    low, _ = jax.lax.fori_loop(0, max(1, count.bit_length()), step, (low, high))
    safe = jnp.minimum(low, size - 1)
    return safe, (low < count) & jnp.all(table[safe] == queries, axis=-1)


def _compact(keys: Array, valid: Array, capacity: int, /) -> tuple[Array, Array, Array]:
    sentinel = jnp.iinfo(jnp.int64).max
    masked = jnp.where(valid[:, None], keys, sentinel)
    order = jnp.lexsort(
        tuple(masked[:, column] for column in range(keys.shape[1] - 1, -1, -1))
    )
    ordered = masked[order]
    fresh = (ordered[:, 0] != sentinel) & jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            jnp.any(ordered[1:] != ordered[:-1], axis=1),
        )
    )
    count = jnp.sum(fresh, dtype=jnp.int32)
    selected = jnp.nonzero(fresh, size=capacity, fill_value=0)[0]
    active = jnp.arange(capacity) < count
    return jnp.where(active[:, None], ordered[selected], -1), active, count <= capacity


@partial(jax.jit, static_argnames=("counts", "vertex_capacity", "organization_degrees"))
def _project(
    arrays: dict[str, Array],
    tables: tuple[Array, ...],
    identifiers: tuple[Array, ...],
    owners: tuple[Array, ...],
    *,
    counts: tuple[int, ...],
    vertex_capacity: int,
    organization_degrees: tuple[tuple[str, int], ...],
) -> tuple[dict[str, Array], Array]:
    corners = arrays["closure/cell_vertices"]
    cell_ids = arrays["closure/cell_ids"]
    cell_valid = arrays["closure/cell_valid"]
    parts, cells, width = corners.shape
    dimension = width - 1
    output: dict[str, Array] = {}
    for name, value in arrays.items():
        if name.startswith("closure/"):
            output[name] = value
    passed = jnp.all(~cell_valid | ((cell_ids >= 0) & jnp.all(corners >= 0, axis=-1)))
    passed &= jnp.all(
        ~cell_valid | jnp.all(jnp.diff(jnp.sort(corners, axis=-1), axis=-1) > 0, axis=-1)
    )
    positions: list[Array] = []
    masks: list[Array] = []
    for degree in range(dimension + 1):
        if degree == dimension:
            queries = cell_ids[..., None]
            valid = cell_valid
            capacity = cells
        else:
            columns = jnp.asarray(
                tuple(combinations(range(width), degree + 1)), dtype=jnp.int32
            )
            queries = jnp.sort(corners[:, :, columns], axis=-1).reshape(
                (parts, -1, degree + 1)
            )
            valid = jnp.broadcast_to(
                cell_valid[..., None], (parts, cells, columns.shape[0])
            ).reshape((parts, -1))
            capacity = vertex_capacity if degree == 0 else queries.shape[1]
        keys, mask, fits = jax.vmap(lambda key, active: _compact(key, active, capacity))(
            queries, valid
        )
        row, found = _lookup(tables[degree], keys, counts[degree])
        owner = owners[degree][row]
        passed &= jnp.all(fits) & jnp.all(
            ~mask | (found & (owner >= 0) & (owner < parts))
        )
        table = tables[degree]
        prefix = jnp.arange(table.shape[0]) < counts[degree]
        # Canonical accepted banks are strictly ordered, complete owned occurrences.
        ordered = jnp.zeros((table.shape[0] - 1,), dtype=jnp.bool_)
        equal = jnp.ones(ordered.shape, dtype=jnp.bool_)
        for column in range(table.shape[-1]):
            ordered |= equal & (table[:-1, column] < table[1:, column])
            equal &= table[:-1, column] == table[1:, column]
        passed &= jnp.all(
            ~prefix
            | (
                (identifiers[degree] >= 0)
                & (owners[degree] >= 0)
                & (owners[degree] < parts)
            )
        )
        passed &= jnp.all((jnp.arange(ordered.shape[0]) >= counts[degree] - 1) | ordered)
        sorted_ids = jnp.sort(
            jnp.where(prefix, identifiers[degree], jnp.iinfo(jnp.int64).max)
        )
        passed &= jnp.all(
            (jnp.arange(sorted_ids.shape[0] - 1) >= counts[degree] - 1)
            | (sorted_ids[:-1] < sorted_ids[1:])
        )
        output[f"entity/{degree}/keys"] = keys
        output[f"entity/{degree}/valid"] = mask
        output[f"entity/{degree}/ids"] = jnp.where(mask, identifiers[degree][row], -1)
        output[f"entity/{degree}/owners"] = jnp.where(mask, owner, -1)
        if degree == dimension:
            passed &= jnp.all(jnp.sum(mask, axis=1) == jnp.sum(cell_valid, axis=1))
            original_rows, original_found = _lookup(
                tables[degree], cell_ids[..., None], counts[degree]
            )
            passed &= jnp.all(
                ~cell_valid
                | (
                    original_found
                    & (owners[degree][original_rows] == arrays["closure/cell_owner"])
                )
            )
        positions.append(row)
        masks.append(mask)
    output["coordinates"] = arrays["coordinates"][positions[0]]
    if "closure/cell_coordinates" in arrays:
        corner_rows, found = _lookup(tables[0], corners[..., None], counts[0])
        accepted = arrays["coordinates"][corner_rows]
        passed &= jnp.all(~cell_valid[..., None] | found)
        passed &= jnp.all(
            ~cell_valid[..., None, None]
            | (accepted == arrays["closure/cell_coordinates"])
        )
    for name, degree in organization_degrees:
        values = arrays[name]
        mask = masks[degree]
        rows = positions[degree]
        selected = values[rows]
        broadcast_mask = mask.reshape((*mask.shape, *((1,) * (values.ndim - 1))))
        output[name] = jnp.where(
            broadcast_mask, selected, jnp.zeros((), dtype=values.dtype)
        )
        if name.endswith("/membership"):
            passed &= jnp.all(~values[counts[degree] :])
    return output, passed


def _addressable(value: Array, partition_index: int, /) -> Array:
    for shard in value.addressable_shards:
        leading = shard.index[0]
        if isinstance(leading, slice):
            first = 0 if leading.start is None else leading.start
            last = value.shape[0] if leading.stop is None else leading.stop
            if first <= partition_index < last:
                return shard.data[partition_index - first]
    raise ValueError("Publication projection partition is not process-addressable.")


@final
class PublicationProjection(StrictModule, NonTrainableState):
    """Numerical global receipts bound to the exact accepted scientific banks."""

    source_arrays: tuple[tuple[str, Array], ...]
    entity_tables: tuple[Array, ...]
    entity_ids: tuple[Array, ...]
    entity_owners: tuple[Array, ...]
    projected_arrays: tuple[tuple[str, Array], ...]
    geometry: CellGeometryStorageProjection
    source_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    coordinate_geometry_id: str = eqx.field(static=True)
    organization_id: str = eqx.field(static=True)
    definitions_id: str = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)

    def require_source(self, logical_arrays: tuple[tuple[str, Array], ...], /) -> None:
        actual = dict(logical_arrays)
        if any(
            name not in actual or actual[name] is not value
            for name, value in self.source_arrays
            if not name.startswith("closure/")
        ):
            raise ValueError(
                "Publication projection does not belong to these immutable numerical banks."
            )
        self.geometry.require_source(logical_arrays, self.coordinate_geometry_id)

    def addressable_arrays(
        self, partition_index: int, /
    ) -> tuple[tuple[str, Array], ...]:
        if not 0 <= partition_index < self.partition_count:
            raise ValueError("Publication projection partition is outside its placement.")
        return tuple(
            (name, _addressable(value, partition_index))
            for name, value in self.projected_arrays
        )


@final
class PreparedPublicationLowering(StrictModule, NonTrainableState):
    """One globally shaped query operation prepared from actual received closure packets."""

    source_arrays: tuple[tuple[str, Array], ...]
    entity_tables: tuple[Array, ...]
    entity_ids: tuple[Array, ...]
    entity_owners: tuple[Array, ...]
    source_elements: tuple[tuple[str, CellGeometryElement], ...]
    counts: tuple[int, ...] = eqx.field(static=True)
    vertex_capacity: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    organization_degrees: tuple[tuple[str, int], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    coordinate_geometry_id: str = eqx.field(static=True)
    organization_id: str = eqx.field(static=True)
    definitions_id: str = eqx.field(static=True)

    def __init__(
        self,
        original: CellMeshingResult | InitialCollectiveMeshEvidence | None,
        evidence: CollectiveMeshEvidence | InitialCollectiveMeshEvidence,
        neighborhood: SimplexNeighborhoodWorkset | None,
        /,
        *,
        vertex_capacity: int,
    ) -> None:
        from ._initial_certification import InitialCollectiveMeshEvidence

        if isinstance(evidence, InitialCollectiveMeshEvidence):
            if original is not None or neighborhood is not None:
                raise ValueError(
                    "Initial publication consumes its authored source, not a predecessor result."
                )
            evidence.require_current()
            arrays = dict(evidence.logical_arrays)
            queries = arrays["closure/cell_vertices"]
            if queries.ndim != 3 or queries.shape[-1] != 3:
                raise ValueError(
                    "Initial publication requires complete fixed-capacity surface closure packets."
                )
            degrees = []
            for name in arrays:
                if name.startswith("initial/source_"):
                    degrees.append((name, 0))
                elif name.startswith("initial/edge_"):
                    degrees.append((name, 1))
                elif name.startswith("initial/cell_"):
                    degrees.append((name, 2))
            from ._collective_organization import initial_organization_definitions

            for family, records in initial_organization_definitions(evidence):
                for index, record in enumerate(records):
                    name = f"organization/{family}/{index}/membership"
                    membership = arrays[name]
                    degree = record.scope.entity_dimension
                    if (
                        membership.shape != evidence.entity_ids[degree].shape
                        or membership.dtype != jnp.bool_
                    ):
                        raise ValueError(
                            "Initial organization membership differs from its scientific source entity axis."
                        )
                    degrees.append((name, degree))
            self.source_arrays = evidence.logical_arrays
            self.entity_tables = evidence.entity_keys
            self.entity_ids = evidence.entity_ids
            self.entity_owners = evidence.entity_owners
            self.counts = evidence.global_entity_counts
            self.vertex_capacity = vertex_capacity
            self.coordinate_count = evidence.global_entity_counts[0]
            self.source_elements = (
                (
                    "surface",
                    coordinate_lagrange_element(
                        evidence.cell_kind, evidence.specification.target.geometry_order
                    ),
                ),
            )
            self.organization_degrees = tuple(degrees)
            self.source_id = evidence.compiled.domain.domain_id
            self.topology_id = evidence.topology_id
            self.coordinate_geometry_id = evidence.coordinate_geometry_id
            self.organization_id = evidence.global_organization_id
            self.definitions_id = canonical_fingerprint(
                (evidence.compiled.compiled_id, evidence.specification.specification_id)
            )
            return
        if original is None or neighborhood is None:
            raise ValueError(
                "Subdivision publication requires its actual predecessor and neighborhood."
            )
        if isinstance(vertex_capacity, bool) or vertex_capacity <= 0:
            raise ValueError("Publication vertex capacity must be positive.")
        arrays = dict(evidence.logical_arrays)
        fields = (
            "cell_ids",
            "cell_vertices",
            "cell_valid",
            "cell_owner",
            "cell_coordinates",
            "root_cell_ids",
        )
        for field in fields:
            arrays[f"closure/{field}"] = getattr(neighborhood, field)
        queries = neighborhood.cell_vertices
        if (
            queries.ndim != 3
            or queries.dtype != jnp.int64
            or queries.shape[:2] != neighborhood.cell_ids.shape
        ):
            raise ValueError(
                "Publication requires actual part-leading fixed-capacity simplex packets."
            )
        counts = evidence.global_entity_counts
        if len(counts) != queries.shape[-1]:
            raise ValueError(
                "Publication simplex dimension differs from the accepted entity banks."
            )
        evidence.require_passed()
        if isinstance(original, InitialCollectiveMeshEvidence):
            from ._collective_organization import initial_organization_definitions

            source_geometry_id = original.coordinate_geometry_id
            source_topology_id = original.topology_id
            coordinate_count = original.global_entity_counts[0]
            source_id = original.evidence_id
            families = initial_organization_definitions(original)
            source_elements = (
                (
                    "surface",
                    coordinate_lagrange_element(
                        original.cell_kind, original.specification.target.geometry_order
                    ),
                ),
            )
        else:
            source_geometry_id = cell_geometry_id(original.geometry)
            source_topology_id = original.mesh.topology_id
            coordinate_count = original.geometry.coordinates.shape[0]
            source_id = original.result_id
            families = (
                ("patch", original.patches),
                ("zone", original.zones),
                ("label", original.labels),
                ("attribute", original.attributes),
            )
            source_elements = tuple(
                zip(
                    original.geometry.block_names, original.geometry.elements, strict=True
                )
            )
        for name, identity in (
            ("geometry/source_geometry_id", source_geometry_id),
            ("geometry/source_topology_id", source_topology_id),
        ):
            expected = np.frombuffer(
                bytes.fromhex(canonical_fingerprint(identity)), dtype=np.uint8
            )
            if (
                name not in arrays
                or arrays[name].shape != (32,)
                or arrays[name].dtype != jnp.uint8
            ):
                raise ValueError(
                    "Publication requires immutable scientific source identity banks."
                )
            if not bool(jax.device_get(jnp.all(arrays[name] == jnp.asarray(expected)))):
                raise ValueError(
                    "Publication scientific source identity differs from its authored source."
                )
        definitions = []
        degrees = []
        for family, records in families:
            for index, record in enumerate(records):
                degree = record.scope.entity_dimension
                definitions.append(
                    (family, index, degree, _organization_definition_id(record))
                )
                membership = f"organization/{family}/{index}/membership"
                if (
                    arrays[membership].shape != evidence.entity_ids[degree].shape
                    or arrays[membership].dtype != jnp.bool_
                ):
                    raise ValueError(
                        "Publication organization membership has incompatible scientific axes."
                    )
                degrees.append((membership, degree))
                if family == "attribute":
                    if not isinstance(record, MeshAttribute):
                        raise TypeError(
                            "Attribute publication requires its actual scientific attribute definition."
                        )
                    name = f"organization/attribute/{index}/values"
                    if arrays[name].shape != (
                        *arrays[membership].shape,
                        *record.component_shape,
                    ):
                        raise ValueError(
                            "Publication attribute values have incompatible scientific component axes."
                        )
                    degrees.append((name, degree))
        for degree, (keys, ids, owners, count) in enumerate(
            zip(
                evidence.entity_keys,
                evidence.entity_ids,
                evidence.entity_owners,
                counts,
                strict=True,
            )
        ):
            width = 1 if degree == len(counts) - 1 else degree + 1
            if (
                keys.ndim != 2
                or keys.shape[-1] != width
                or keys.shape[0] < count
                or count < 0
                or keys.shape[0] == 0
            ):
                raise ValueError(
                    "Publication entity keys have incompatible scientific capacity."
                )
            if (
                keys.dtype != jnp.int64
                or ids.shape != keys.shape[:1]
                or ids.dtype != jnp.int64
                or owners.shape != ids.shape
                or owners.dtype != jnp.int32
            ):
                raise ValueError(
                    "Publication entity identifiers and ownership have incompatible scientific axes."
                )
        self.source_arrays = tuple(sorted(arrays.items()))
        self.entity_tables = evidence.entity_keys
        self.entity_ids = evidence.entity_ids
        self.entity_owners = evidence.entity_owners
        self.counts = counts
        self.vertex_capacity = vertex_capacity
        self.coordinate_count = coordinate_count
        self.source_elements = source_elements
        self.organization_degrees = tuple(degrees)
        self.source_id = source_id
        self.topology_id = evidence.topology_id
        self.coordinate_geometry_id = evidence.coordinate_geometry_id
        self.organization_id = evidence.global_organization_id
        self.definitions_id = canonical_fingerprint(definitions)

    def execute(self) -> PublicationProjection:
        arrays, accepted = _project(
            dict(self.source_arrays),
            self.entity_tables,
            self.entity_ids,
            self.entity_owners,
            counts=self.counts,
            vertex_capacity=self.vertex_capacity,
            organization_degrees=self.organization_degrees,
        )
        if not bool(jax.device_get(accepted)):
            raise ValueError(
                "Publication projection has missing or nonunique ownership, corrupt banks, or exceeded closure capacity."
            )
        query = dict(self.source_arrays)["closure/cell_ids"]
        placement = query.sharding
        if isinstance(placement, NamedSharding):
            placement = NamedSharding(
                placement.mesh,
                PartitionSpec(placement.spec[0] if placement.spec else None),
            )
        projected = tuple(
            (name, jax.device_put(value, placement))
            for name, value in sorted(arrays.items())
        )
        geometry = CellGeometryStorageProjection(
            self.source_arrays,
            self.coordinate_geometry_id,
            self.coordinate_count,
            source_elements=dict(self.source_elements),
        )
        return PublicationProjection(
            self.source_arrays,
            self.entity_tables,
            self.entity_ids,
            self.entity_owners,
            projected,
            geometry,
            self.source_id,
            self.topology_id,
            self.coordinate_geometry_id,
            self.organization_id,
            self.definitions_id,
            query.shape[0],
        )
