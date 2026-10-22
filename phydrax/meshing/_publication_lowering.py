#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity scientific publication joins before any owner-local construction."""

from __future__ import annotations

from functools import partial
from itertools import combinations
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec

from phydrax.discretization._reference_cell_types import CellGeometryElement

from .._execution_array import _project_local_leading_axes
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    CellGeometryStorageProjection,
    coordinate_lagrange_element,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..typing import checked

# Module references defer symbol resolution across mutually dependent execution owners.
from . import (
    _collective_organization as collective_organization_owner,
    _distribution as distribution_owner,
    _entity_keys as entity_keys_owner,
    _initial_certification as initial_certification_owner,
    _result as result_owner,
)
from ._collective_failure import agreed_local_step
from ._organization_records import MeshAttribute, organization_definition_id


def _lookup(table: Array, queries: Array, count: int, /) -> tuple[Array, Array]:
    """Lexicographic lower bounds without query-by-global-bank dense products."""
    size = table.shape[0]
    low = jnp.zeros(queries.shape[:-1], dtype=jnp.int32)
    high = jnp.full(low.shape, count, dtype=jnp.int32)

    def step(bounds: tuple[Array, Array]) -> tuple[Array, Array]:
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

    low, _ = jax.lax.while_loop(
        lambda bounds: jnp.any(bounds[0] < bounds[1]), step, (low, high)
    )
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


@jax.jit
def logical_boundary_masks(
    cell_rows: jax.Array,
    cells: int,
    entity_tables: tuple[jax.Array, ...],
    counts: tuple[int, ...],
    /,
) -> tuple[jax.Array, ...]:
    dimension = cell_rows.shape[1] - 1
    columns = jnp.asarray(
        tuple(
            tuple(column for column in range(dimension + 1) if column != opposite)
            for opposite in range(dimension + 1)
        ),
        dtype=jnp.int32,
    )
    cell_valid = jnp.arange(cell_rows.shape[0]) < cells
    facet_keys = jnp.sort(cell_rows[:, columns], axis=2).reshape((-1, dimension))
    facet_valid = jnp.repeat(cell_valid, dimension + 1)
    facet_table = jnp.where(
        entity_tables[-2] >= 0,
        entity_tables[-2],
        jnp.iinfo(jnp.int64).max,
    )
    facet_rows = entity_keys_owner.key_positions(facet_table, facet_keys)
    incidence = (
        jnp.zeros((facet_table.shape[0],), dtype=jnp.int32)
        .at[jnp.maximum(facet_rows, 0)]
        .add(facet_valid.astype(jnp.int32))
    )
    facet_boundary = (jnp.arange(facet_table.shape[0]) < counts[-2]) & (incidence == 1)
    masks = []
    for degree in range(dimension - 1):
        local = np.asarray(
            tuple(combinations(range(dimension), degree + 1)), dtype=np.int32
        )
        keys = entity_tables[-2][:, local].reshape((-1, degree + 1))
        valid = jnp.repeat(facet_boundary, local.shape[0])
        table = jnp.where(
            entity_tables[degree] >= 0,
            entity_tables[degree],
            jnp.iinfo(jnp.int64).max,
        )
        rows = entity_keys_owner.key_positions(table, keys)
        membership = (
            jnp.zeros((table.shape[0],), dtype=jnp.int32)
            .at[jnp.maximum(rows, 0)]
            .add(valid.astype(jnp.int32))
        )
        masks.append((jnp.arange(table.shape[0]) < counts[degree]) & (membership > 0))
    masks.append(facet_boundary)
    masks.append(jnp.zeros((cell_rows.shape[0],), dtype=jnp.bool_))
    return tuple(masks)


@partial(jax.jit, static_argnames=("vertex_capacity", "organization_degrees"))
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
    boundary_masks = logical_boundary_masks(
        arrays["cell_vertices"], counts[-1], tables, counts
    )
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
        # accepted banks are strictly ordered, full owned occurrences.
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
        output[f"entity/{degree}/physical_boundary"] = mask & boundary_masks[degree][row]
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
            tail = (jnp.arange(values.shape[0]) >= counts[degree]).reshape(
                (values.shape[0], *((1,) * (values.ndim - 1)))
            )
            passed &= jnp.all(jnp.where(tail, ~values, jnp.ones((), dtype=values.dtype)))
    return output, passed


def _addressable_payload(
    value: Array, partition_index: int, /
) -> tuple[Array, int, jax.Device]:
    for shard in value.addressable_shards:
        leading = shard.index[0]
        if isinstance(leading, slice):
            first = 0 if leading.start is None else leading.start
            last = value.shape[0] if leading.stop is None else leading.stop
            if first <= partition_index < last:
                return shard.data, partition_index - first, shard.device
    raise ValueError(
        "PublicationProjection.partition_index must be process-addressable; got a remote partition"
    )


def addressable(value: Array, partition_index: int, /) -> Array:
    return _addressable_arrays((value,), partition_index)[0]


def _addressable_arrays(
    values: tuple[Array, ...], partition_index: int, /
) -> tuple[Array, ...]:
    payloads = tuple(_addressable_payload(value, partition_index) for value in values)
    return _project_local_leading_axes(payloads)


@final
class PublicationProjection(StrictModule, NonTrainableState):
    """Bind global numerical projections to accepted scientific banks."""

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
                "PublicationProjection.source_arrays must retain the accepted numerical banks; got different bank objects"
            )
        type(self.geometry).require_source(
            self.geometry, logical_arrays, self.coordinate_geometry_id
        )

    def addressable_arrays(
        self, partition_index: int, /
    ) -> tuple[tuple[str, Array], ...]:
        if not 0 <= partition_index < self.partition_count:
            raise ValueError(
                "PublicationProjection.partition_index must belong to its placement; got an out-of-range partition"
            )
        values = _addressable_arrays(
            tuple(value for _, value in self.projected_arrays), partition_index
        )
        return tuple(
            (entry[0], value)
            for entry, value in zip(self.projected_arrays, values, strict=True)
        )


@final
class PreparedPublicationLowering(StrictModule, NonTrainableState):
    """Prepare globally shaped queries from received fixed-capacity closure packets."""

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

    @checked
    def __init__(
        self,
        original: result_owner.CellMeshingResult
        | initial_certification_owner.InitialCollectiveMeshEvidence
        | None,
        evidence: result_owner.CollectiveMeshEvidence
        | initial_certification_owner.InitialCollectiveMeshEvidence,
        neighborhood: distribution_owner.SimplexNeighborhoodWorkset | None,
        /,
        *,
        vertex_capacity: int,
    ) -> None:

        match evidence:
            case initial_certification_owner.InitialCollectiveMeshEvidence():
                preparation = _prepare_initial_publication(
                    original, evidence, neighborhood, vertex_capacity
                )
            case result_owner.CollectiveMeshEvidence():
                preparation = _prepare_subdivision_publication(
                    original, evidence, neighborhood, vertex_capacity
                )
            case _:
                raise TypeError(
                    "PreparedPublicationLowering.evidence must be initial or collective mesh evidence; "
                    f"got {type(evidence).__name__}"
                )
        self.source_arrays = preparation.source_arrays
        self.entity_tables = preparation.entity_tables
        self.entity_ids = preparation.entity_ids
        self.entity_owners = preparation.entity_owners
        self.source_elements = preparation.source_elements
        self.counts = preparation.counts
        self.vertex_capacity = preparation.vertex_capacity
        self.coordinate_count = preparation.coordinate_count
        self.organization_degrees = preparation.organization_degrees
        self.source_id = preparation.source_id
        self.topology_id = preparation.topology_id
        self.coordinate_geometry_id = preparation.coordinate_geometry_id
        self.organization_id = preparation.organization_id
        self.definitions_id = preparation.definitions_id

    def execute(self) -> PublicationProjection:
        with agreed_local_step():
            projected_names = {
                "coordinates",
                "cell_vertices",
                *(name for name, _ in self.organization_degrees),
            }
            projected_inputs = {
                name: value
                for name, value in self.source_arrays
                if name.startswith("closure/") or name in projected_names
            }
        arrays, accepted = _project(
            projected_inputs,
            self.entity_tables,
            self.entity_ids,
            self.entity_owners,
            counts=self.counts,
            vertex_capacity=self.vertex_capacity,
            organization_degrees=self.organization_degrees,
        )
        passed = bool(jax.device_get(accepted))
        with agreed_local_step():
            if not passed:
                raise ValueError(
                    "PublicationProjection.projected_arrays must have unique ownership and fit closure capacity; got missing ownership, corrupt banks, or exceeded capacity"
                )
        with agreed_local_step():
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
        with agreed_local_step():
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


class _PublicationPreparation(StrictModule):
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


class _PublicationOrganization(StrictModule):
    definitions: tuple[tuple[str, int, int, str], ...] = eqx.field(static=True)
    degrees: tuple[tuple[str, int], ...] = eqx.field(static=True)


def _prepare_publication_organization(
    arrays: dict[str, Array],
    source_geometry_id: str,
    source_topology_id: str,
    families: collective_organization_owner.OrganizationDefinitions,
    evidence: result_owner.CollectiveMeshEvidence,
    counts: tuple[int, ...],
) -> _PublicationOrganization:
    for name, identity in (
        ("geometry/source_geometry_id", source_geometry_id),
        ("geometry/source_topology_id", source_topology_id),
    ):
        with agreed_local_step():
            expected = np.frombuffer(
                bytes.fromhex(canonical_fingerprint(identity)), dtype=np.uint8
            )
            if (
                name not in arrays
                or arrays[name].shape != (32,)
                or arrays[name].dtype != jnp.uint8
            ):
                raise ValueError(
                    "PreparedPublicationLowering.source_arrays must retain source identity banks; got absent or incompatible identity axes"
                )
        accepted = bool(jax.device_get(jnp.all(arrays[name] == jnp.asarray(expected))))
        with agreed_local_step():
            if not accepted:
                raise ValueError(
                    "PreparedPublicationLowering.source_arrays must match authored source identities; got a different identity"
                )
    with agreed_local_step():
        definitions = []
        degrees = []
        for family, records in families:
            for index, record in enumerate(records):
                degree = record.scope.entity_dimension
                definitions.append(
                    (family, index, degree, organization_definition_id(record))
                )
                membership = f"organization/{family}/{index}/membership"
                if (
                    arrays[membership].shape != evidence.entity_ids[degree].shape
                    or arrays[membership].dtype != jnp.bool_
                ):
                    raise ValueError(
                        "PreparedPublicationLowering.membership must match its entity bank; got incompatible membership axes"
                    )
                degrees.append((membership, degree))
                if family == "attribute":
                    match record:
                        case MeshAttribute():
                            name = f"organization/attribute/{index}/values"
                            if arrays[name].shape != (
                                *arrays[membership].shape,
                                *record.component_shape,
                            ):
                                raise ValueError(
                                    "PreparedPublicationLowering.attribute_values must match entity and component axes; got incompatible component axes"
                                )
                            degrees.append((name, degree))
                        case _:
                            raise TypeError(
                                "PreparedPublicationLowering.attribute must be MeshAttribute; got a different definition kind"
                            )
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
                    "PreparedPublicationLowering.entity_keys must have compatible capacity; got incompatible key axes"
                )
            if (
                keys.dtype != jnp.int64
                or ids.shape != keys.shape[:1]
                or ids.dtype != jnp.int64
                or owners.shape != ids.shape
                or owners.dtype != jnp.int32
            ):
                raise ValueError(
                    "PreparedPublicationLowering.entity_ids must match key axes and integer dtypes together with entity_owners; got incompatible axes"
                )
        return _PublicationOrganization(tuple(definitions), tuple(degrees))


def _prepare_initial_publication(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence
    | None,
    evidence: initial_certification_owner.InitialCollectiveMeshEvidence,
    neighborhood: distribution_owner.SimplexNeighborhoodWorkset | None,
    vertex_capacity: int,
) -> _PublicationPreparation:
    if original is not None or neighborhood is not None:
        raise ValueError(
            "PreparedPublicationLowering.original must be None together with neighborhood for initial publication; got a predecessor"
        )
    evidence.require_current()
    arrays = dict(evidence.logical_arrays)
    queries = arrays["closure/cell_vertices"]
    if queries.ndim != 3 or queries.shape[-1] != 3:
        raise ValueError(
            "PreparedPublicationLowering.closure must contain fixed-capacity triangle packets; got incompatible packet axes"
        )
    degrees = []
    for name in arrays:
        if name.startswith("initial/source_"):
            degrees.append((name, 0))
        elif name.startswith("initial/edge_"):
            degrees.append((name, 1))
        elif name.startswith("initial/cell_"):
            degrees.append((name, 2))

    for (
        family,
        records,
    ) in collective_organization_owner.initial_organization_definitions(evidence):
        for index, record in enumerate(records):
            name = f"organization/{family}/{index}/membership"
            membership = arrays[name]
            degree = record.scope.entity_dimension
            if (
                membership.shape != evidence.entity_ids[degree].shape
                or membership.dtype != jnp.bool_
            ):
                raise ValueError(
                    "PreparedPublicationLowering.membership must match its source entity axis; got incompatible membership axes"
                )
            degrees.append((name, degree))
    source_arrays = evidence.logical_arrays
    entity_tables = evidence.entity_keys
    entity_ids = evidence.entity_ids
    entity_owners = evidence.entity_owners
    counts = evidence.global_entity_counts
    coordinate_count = evidence.global_entity_counts[0]
    premise = evidence.premise
    source_elements = (
        (
            premise.block_name,
            coordinate_lagrange_element(
                evidence.cell_kind, premise.specification.target.geometry_order
            ),
        ),
    )
    organization_degrees = tuple(degrees)
    source_id = premise.source_domain_id
    topology_id = evidence.topology_id
    coordinate_geometry_id = evidence.coordinate_geometry_id
    organization_id = evidence.global_organization_id
    definitions_id = premise.definitions_id
    return _PublicationPreparation(
        source_arrays=source_arrays,
        entity_tables=entity_tables,
        entity_ids=entity_ids,
        entity_owners=entity_owners,
        source_elements=source_elements,
        counts=counts,
        vertex_capacity=vertex_capacity,
        coordinate_count=coordinate_count,
        organization_degrees=organization_degrees,
        source_id=source_id,
        topology_id=topology_id,
        coordinate_geometry_id=coordinate_geometry_id,
        organization_id=organization_id,
        definitions_id=definitions_id,
    )


def _prepare_subdivision_publication(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence
    | None,
    evidence: result_owner.CollectiveMeshEvidence,
    neighborhood: distribution_owner.SimplexNeighborhoodWorkset | None,
    vertex_capacity: int,
) -> _PublicationPreparation:
    with agreed_local_step():
        if original is None or neighborhood is None:
            raise ValueError(
                "PreparedPublicationLowering.neighborhood must bind subdivision inputs together with the predecessor; got a missing predecessor or neighborhood"
            )
        if isinstance(vertex_capacity, bool) or vertex_capacity <= 0:
            raise ValueError(
                "PreparedPublicationLowering.vertex_capacity must be positive; got nonpositive capacity"
            )
        arrays = dict(evidence.logical_arrays)
        arrays.update(
            {
                "closure/cell_ids": neighborhood.cell_ids,
                "closure/cell_vertices": neighborhood.cell_vertices,
                "closure/cell_valid": neighborhood.cell_valid,
                "closure/cell_owner": neighborhood.cell_owner,
                "closure/cell_coordinates": neighborhood.cell_coordinates,
                "closure/root_cell_ids": neighborhood.root_cell_ids,
            }
        )
        queries = neighborhood.cell_vertices
        if (
            queries.ndim != 3
            or queries.dtype != jnp.int64
            or queries.shape[:2] != neighborhood.cell_ids.shape
        ):
            raise ValueError(
                "PreparedPublicationLowering.neighborhood must have part-leading simplex packet axes; got incompatible packets"
            )
        counts = evidence.global_entity_counts
        if len(counts) != queries.shape[-1]:
            raise ValueError(
                "PreparedPublicationLowering.neighborhood must match the entity bank dimension; got a different simplex dimension"
            )
    with agreed_local_step():
        type(evidence).require_passed(evidence)
    with agreed_local_step():
        if isinstance(original, initial_certification_owner.InitialCollectiveMeshEvidence):
            source_geometry_id = original.coordinate_geometry_id
            source_topology_id = original.topology_id
            coordinate_count = original.global_entity_counts[0]
            source_id = original.evidence_id
            families = collective_organization_owner.initial_organization_definitions(
                original
            )
            source_elements = (
                (
                    original.premise.block_name,
                    coordinate_lagrange_element(
                        original.cell_kind,
                        original.premise.specification.target.geometry_order,
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
                zip(original.geometry.block_names, original.geometry.elements, strict=True)
            )
    organization = _prepare_publication_organization(
        arrays, source_geometry_id, source_topology_id, families, evidence, counts
    )
    with agreed_local_step():
        definitions, degrees = organization.definitions, organization.degrees
        source_arrays = tuple(sorted(arrays.items()))
        entity_tables = evidence.entity_keys
        entity_ids = evidence.entity_ids
        entity_owners = evidence.entity_owners
        organization_degrees = tuple(degrees)
        topology_id = evidence.topology_id
        coordinate_geometry_id = evidence.coordinate_geometry_id
        organization_id = evidence.global_organization_id
        # Identity encoding retains the list-of-tuples definition payload.
        definitions_id = canonical_fingerprint(list(definitions))
        return _PublicationPreparation(
            source_arrays=source_arrays,
            entity_tables=entity_tables,
            entity_ids=entity_ids,
            entity_owners=entity_owners,
            source_elements=source_elements,
            counts=counts,
            vertex_capacity=vertex_capacity,
            coordinate_count=coordinate_count,
            organization_degrees=organization_degrees,
            source_id=source_id,
            topology_id=topology_id,
            coordinate_geometry_id=coordinate_geometry_id,
            organization_id=organization_id,
            definitions_id=definitions_id,
        )
