#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Logical owner-local migration packets and transport evidence."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import fields
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import AdaptiveSimplexState
from ..typing import (
    AnyDim,
    Bool,
    checked,
    Dim,
    Float,
    Float64,
    Int32,
    Int64,
    parse,
    Scope,
    Shaped,
    validate,
    VariadicDim,
)


if TYPE_CHECKING:
    from ._distribution import SimplexNeighborhoodWorkset
    from ._distribution_migration import (
        OwnerLocalMigrationEvidence,
        OwnerLocalMigrationState,
    )


def _migration_local_validation(collective: bool) -> AbstractContextManager[None]:
    """Keep local reconstruction checks outside the next numerical collective."""
    if collective:
        from ..lifecycle._meshing_sources import meshing_source_local_restore_phase

        return meshing_source_local_restore_phase()
    return nullcontext()


def _migration_optional_owner(
    collective: bool, present: bool, field: str, /
) -> None:
    """Agree optional numerical branches before any rank can skip their kernels."""
    if collective and jax.process_count() > 1:
        from jax.experimental import multihost_utils

        values = np.asarray(
            multihost_utils.process_allgather(np.int32(present), tiled=False)
        )
        with _migration_local_validation(True):
            if not np.all(values == np.int32(present)):
                raise ValueError(
                    f"{field} must have the same presence on every reconstruction rank."
                )


def _require_migration_arrays_live(value: object, /) -> None:
    """Refuse deleted current buffers before any other rank enters a kernel."""
    if any(
        isinstance(leaf, Array) and leaf.is_deleted()
        for leaf in jax.tree_util.tree_leaves(value)
    ):
        raise ValueError("Migration reconstruction must retain live numerical arrays.")


@jax.jit
def _resident_owner_status_invalid(
    owners: Array, valid: Array, status: Array, parts: int
) -> Array:
    return jnp.any(valid & ((owners < 0) | (owners >= parts))) | jnp.any(status < 0)


@jax.jit
def _corner_owner_invalid(owners: Array, valid: Array, parts: int) -> Array:
    return jnp.any(valid[..., None] & ((owners < 0) | (owners >= parts)))


@jax.jit
def _forest_owner_invalid(
    cell_ids: Array,
    vertex_ids: Array,
    cell_owners: Array,
    vertex_owners: Array,
    parts: int,
) -> Array:
    return jnp.any(
        (cell_ids >= 0) & ((cell_owners < 0) | (cell_owners >= parts))
    ) | jnp.any(
        (vertex_ids >= 0) & ((vertex_owners < 0) | (vertex_owners >= parts))
    )


@jax.jit
def _transport_decision_invalid(
    status: Array, accepted: Array, counts: tuple[Array, ...]
) -> Array:
    invalid = jnp.any(status < 0) | jnp.any(accepted != (status == 0))
    for count in counts:
        invalid |= jnp.any(count < 0)
    return invalid


@jax.jit
def _transport_owned_invalid(owned: Array, resident: Array) -> Array:
    return jnp.any(owned > resident)


@jax.jit
def _transport_capacity_invalid(
    accepted: Array, count: Array, capacity: int
) -> Array:
    return jnp.any(accepted & (count > capacity))


class _MigrationPartDim(Dim):
    pass


class _MigrationCellDim(Dim):
    pass


class _MigrationCornerDim(Dim):
    pass


class _MigrationAmbientDim(Dim):
    pass


class _MigrationSourceCellDim(Dim):
    pass


class _MigrationForestCellDim(Dim):
    pass


class _MigrationForestVertexDim(Dim):
    pass


class _MigrationCellValueDims(VariadicDim):
    pass


class _MigrationVertexValueDims(VariadicDim):
    pass


@final
class MigrationWorkset(StrictModule, NonTrainableState):
    """Retain the twelve neighborhood arrays without execution placement."""

    __strict_contract__ = True

    cell_ids: Int64[_MigrationPartDim, _MigrationCellDim]
    cell_vertices: Int64[_MigrationPartDim, _MigrationCellDim, _MigrationCornerDim]
    cell_coordinates: Float[
        _MigrationPartDim, _MigrationCellDim, _MigrationCornerDim, _MigrationAmbientDim
    ]
    cell_owner: Int32[_MigrationPartDim, _MigrationCellDim]
    cell_classes: Int32[_MigrationPartDim, _MigrationCellDim]
    cell_exterior: Bool[_MigrationPartDim, _MigrationCellDim, _MigrationCornerDim]
    cell_valid: Bool[_MigrationPartDim, _MigrationCellDim]
    root_cell_ids: Int64[_MigrationPartDim, _MigrationCellDim]
    source_cell_ids: Int64[_MigrationPartDim, _MigrationCellDim, _MigrationSourceCellDim]
    source_vertex_ids: Int64[
        _MigrationPartDim, _MigrationCellDim, _MigrationCornerDim, _MigrationCornerDim
    ]
    source_weights: Float64[
        _MigrationPartDim, _MigrationCellDim, _MigrationCornerDim, _MigrationCornerDim
    ]
    status: Int32[_MigrationPartDim]

    @checked
    def __init__(
        self,
        cell_ids: Int64[_MigrationPartDim, _MigrationCellDim],
        cell_vertices: Int64[_MigrationPartDim, _MigrationCellDim, _MigrationCornerDim],
        cell_coordinates: Float[
            _MigrationPartDim,
            _MigrationCellDim,
            _MigrationCornerDim,
            _MigrationAmbientDim,
        ],
        cell_owner: Int32[_MigrationPartDim, _MigrationCellDim],
        cell_classes: Int32[_MigrationPartDim, _MigrationCellDim],
        cell_exterior: Bool[_MigrationPartDim, _MigrationCellDim, _MigrationCornerDim],
        cell_valid: Bool[_MigrationPartDim, _MigrationCellDim],
        root_cell_ids: Int64[_MigrationPartDim, _MigrationCellDim],
        source_cell_ids: Int64[
            _MigrationPartDim, _MigrationCellDim, _MigrationSourceCellDim
        ],
        source_vertex_ids: Int64[
            _MigrationPartDim, _MigrationCellDim, _MigrationCornerDim, _MigrationCornerDim
        ],
        source_weights: Float64[
            _MigrationPartDim, _MigrationCellDim, _MigrationCornerDim, _MigrationCornerDim
        ],
        status: Int32[_MigrationPartDim],
    ) -> None:
        self.cell_ids = cell_ids
        self.cell_vertices = cell_vertices
        self.cell_coordinates = cell_coordinates
        self.cell_owner = cell_owner
        self.cell_classes = cell_classes
        self.cell_exterior = cell_exterior
        self.cell_valid = cell_valid
        self.root_cell_ids = root_cell_ids
        self.source_cell_ids = source_cell_ids
        self.source_vertex_ids = source_vertex_ids
        self.source_weights = source_weights
        self.status = status
        self._require_invariants()

    def _require_invariants(self, *, collective: bool = False) -> None:
        with _migration_local_validation(collective):
            parts, cells, corners = self.cell_vertices.shape
            ambient = self.cell_coordinates.shape[-1]
            if (
                parts < 1
                or cells < 1
                or corners not in (3, 4)
                or ambient not in range(corners - 1, 4)
            ):
                raise ValueError(
                    f"MigrationWorkset.extents must describe nonempty triangle or tetrahedron packets; got {(parts, cells, corners, ambient)}"
                )
            if self.source_cell_ids.shape[-1] < 1:
                raise ValueError(
                    f"MigrationWorkset.source_cell_ids must retain a nonempty predecessor capacity; got {self.source_cell_ids.shape}"
                )
            if self.cell_coordinates.dtype not in (
                np.dtype(np.float32),
                np.dtype(np.float64),
            ):
                raise TypeError(
                    f"MigrationWorkset.cell_coordinates must have float32 or float64 dtype; got {self.cell_coordinates.dtype}"
                )
            # Refused packets remain representable; status, not ancestry repair,
            # governs whether the live transaction may publish them.
            owners, valid, status = self.cell_owner, self.cell_valid, self.status
        invalid = bool(jax.device_get(
            _resident_owner_status_invalid(owners, valid, status, parts)
        ))
        with _migration_local_validation(collective):
            if invalid:
                raise ValueError(
                    "MigrationWorkset.owner_status must contain valid resident owners and nonnegative status flags; got an out-of-range owner or negative status"
                )

    def validate_restored(self, *, collective: bool = False) -> None:
        """Validate the packet contract and logical extents."""
        with _migration_local_validation(collective):
            validate(self)
            if collective:
                _require_migration_arrays_live(self)
        self._require_invariants(collective=collective)

    @classmethod
    def from_stored_fields(
        cls, value: MigrationWorkset, /, *, collective: bool = False
    ) -> MigrationWorkset:
        with _migration_local_validation(collective):
            if type(value) is not cls:
                raise TypeError("MigrationWorkset.stored_fields must retain its exact owner type.")
            validate(value)
        if collective:
            value.validate_restored(collective=True)
            with _migration_local_validation(True):
                from .._model._structure import restore_registered_fields

                return restore_registered_fields(
                    cls, {field.name: getattr(value, field.name) for field in fields(value)}
                )
        return cls(
            value.cell_ids,
            value.cell_vertices,
            value.cell_coordinates,
            value.cell_owner,
            value.cell_classes,
            value.cell_exterior,
            value.cell_valid,
            value.root_cell_ids,
            value.source_cell_ids,
            value.source_vertex_ids,
            value.source_weights,
            value.status,
        )

    @classmethod
    def from_live(cls, value: SimplexNeighborhoodWorkset, /) -> MigrationWorkset:
        from ._distribution import SimplexNeighborhoodWorkset

        match value:
            case SimplexNeighborhoodWorkset():
                return cls(
                    value.cell_ids,
                    value.cell_vertices,
                    value.cell_coordinates,
                    value.cell_owner,
                    value.cell_classes,
                    value.cell_exterior,
                    value.cell_valid,
                    value.root_cell_ids,
                    value.source_cell_ids,
                    value.source_vertex_ids,
                    value.source_weights,
                    value.status,
                )
            case _:
                raise TypeError(
                    f"MigrationWorkset.from_live.value must be SimplexNeighborhoodWorkset; got {type(value).__qualname__}"
                )

    def to_live(self) -> SimplexNeighborhoodWorkset:
        from ._distribution import SimplexNeighborhoodWorkset

        self.validate_restored()
        return SimplexNeighborhoodWorkset(
            self.cell_ids,
            self.cell_vertices,
            self.cell_coordinates,
            self.cell_owner,
            self.cell_classes,
            self.cell_exterior,
            self.cell_valid,
            self.root_cell_ids,
            self.source_cell_ids,
            self.source_vertex_ids,
            self.source_weights,
            self.status,
        )


@final
class MigrationWorkState(StrictModule, NonTrainableState):
    """Retain migration fields, epochs, forest records, and raw-slot history."""

    __strict_contract__ = True

    workset: MigrationWorkset
    cell_fields: Shaped[_MigrationPartDim, _MigrationCellDim, _MigrationCellValueDims]
    vertex_fields: Shaped[
        _MigrationPartDim,
        _MigrationCellDim,
        _MigrationCornerDim,
        _MigrationVertexValueDims,
    ]
    cell_epochs: Int64[_MigrationPartDim, _MigrationCellDim]
    vertex_owners: Int32[_MigrationPartDim, _MigrationCellDim, _MigrationCornerDim]
    forest: AdaptiveSimplexState | None = None
    forest_cell_owners: Int32[_MigrationPartDim, _MigrationForestCellDim] | None = None
    forest_vertex_owners: Int32[_MigrationPartDim, _MigrationForestVertexDim] | None = (
        None
    )
    cell_history: tuple[Array, ...] = ()
    vertex_history: tuple[Array, ...] = ()

    @checked
    def __init__(
        self,
        workset: MigrationWorkset,
        cell_fields: Shaped[
            _MigrationPartDim, _MigrationCellDim, _MigrationCellValueDims
        ],
        vertex_fields: Shaped[
            _MigrationPartDim,
            _MigrationCellDim,
            _MigrationCornerDim,
            _MigrationVertexValueDims,
        ],
        cell_epochs: Int64[_MigrationPartDim, _MigrationCellDim],
        vertex_owners: Int32[_MigrationPartDim, _MigrationCellDim, _MigrationCornerDim],
        forest: AdaptiveSimplexState | None = None,
        forest_cell_owners: Int32[_MigrationPartDim, _MigrationForestCellDim]
        | None = None,
        forest_vertex_owners: Int32[_MigrationPartDim, _MigrationForestVertexDim]
        | None = None,
        cell_history: tuple[Array, ...] = (),
        vertex_history: tuple[Array, ...] = (),
    ) -> None:
        self.workset = workset
        self.cell_fields = cell_fields
        self.vertex_fields = vertex_fields
        self.cell_epochs = cell_epochs
        self.vertex_owners = vertex_owners
        self.forest = forest
        self.forest_cell_owners = forest_cell_owners
        self.forest_vertex_owners = forest_vertex_owners
        self.cell_history = cell_history
        self.vertex_history = vertex_history
        self._require_invariants()

    def _require_invariants(self, *, collective: bool = False) -> None:
        self.workset._require_invariants(collective=collective)
        with _migration_local_validation(collective):
            shape = self.workset.cell_ids.shape
            if (
                self.cell_fields.shape[:2] != shape
                or self.vertex_fields.shape[:3] != self.workset.cell_vertices.shape
            ):
                raise ValueError(
                    f"MigrationWorkState.fields must align with workset cells and corners; got {self.cell_fields.shape}/{self.vertex_fields.shape}"
                )
            if (
                self.cell_epochs.shape != shape
                or self.vertex_owners.shape != self.workset.cell_vertices.shape
            ):
                raise ValueError(
                    f"MigrationWorkState.epochs_owners must align with workset cells and corners; got {self.cell_epochs.shape}/{self.vertex_owners.shape}"
                )
            owners, valid = self.vertex_owners, self.workset.cell_valid
        invalid = bool(jax.device_get(_corner_owner_invalid(owners, valid, shape[0])))
        with _migration_local_validation(collective):
            if invalid:
                raise ValueError(
                    "MigrationWorkState.vertex_owners must assign resident corners to logical parts; got an out-of-range owner"
                )
            if type(self.cell_history) is not tuple or type(self.vertex_history) is not tuple:
                raise TypeError(
                    "MigrationWorkState.history must use tuples of numerical leaves; got a non-tuple history"
                )
            if self.forest is None:
                if (
                    self.forest_cell_owners is not None
                    or self.forest_vertex_owners is not None
                    or self.cell_history
                    or self.vertex_history
                ):
                    raise ValueError(
                        "MigrationWorkState.forest must accompany forest owners and raw-slot history; got missing forest"
                    )
                return
            if self.forest_cell_owners is None or self.forest_vertex_owners is None:
                raise ValueError(
                    "MigrationWorkState.forest_owners must accompany retained forest records; got missing owners"
                )
            cell_shape = self.forest.mesh.cell_ids.shape
            vertex_shape = self.forest.mesh.vertex_ids.shape
            if (
                len(cell_shape) != 2
                or len(vertex_shape) != 2
                or cell_shape[0] != shape[0]
                or vertex_shape[0] != shape[0]
            ):
                raise ValueError(
                    f"MigrationWorkState.forest must share logical part extent; got {cell_shape}/{vertex_shape}"
                )
            if (
                self.forest_cell_owners.shape != cell_shape
                or self.forest_vertex_owners.shape != vertex_shape
            ):
                raise ValueError(
                    f"MigrationWorkState.forest_owners must align with raw forest capacities; got {self.forest_cell_owners.shape}/{self.forest_vertex_owners.shape}"
                )
            if (
                self.forest.mesh.cells.shape[-1] != self.workset.cell_vertices.shape[-1]
                or self.forest.mesh.coordinates.shape[-1]
                != self.workset.cell_coordinates.shape[-1]
            ):
                raise ValueError(
                    "MigrationWorkState.forest must share simplex and ambient extents with packets; got incompatible geometry extents"
                )
            # Variable-length history tuples require leaf-level contracts: each
            # leaf may have a different dtype and trailing value shape.
            for name, history, expected in (
                ("cell_history", self.cell_history, cell_shape),
                ("vertex_history", self.vertex_history, vertex_shape),
            ):
                for index, leaf in enumerate(history):
                    parse(
                        leaf,
                        Shaped[_MigrationPartDim, AnyDim, _MigrationCellValueDims],
                        f"MigrationWorkState.{name}[{index}]",
                        scope=Scope(),
                    )
                    if leaf.shape[:2] != expected:
                        raise ValueError(
                            f"MigrationWorkState.{name} must align with raw forest slots {expected}; got shape={leaf.shape} at index={index}"
                        )
            parts = shape[0]
            cell_ids, vertex_ids = self.forest.mesh.cell_ids, self.forest.mesh.vertex_ids
            cell_owners, vertex_owners = self.forest_cell_owners, self.forest_vertex_owners
        invalid = bool(jax.device_get(
            _forest_owner_invalid(
                cell_ids, vertex_ids, cell_owners, vertex_owners, parts
            )
        ))
        with _migration_local_validation(collective):
            if invalid:
                raise ValueError(
                    "MigrationWorkState.forest_owners must assign allocated records to logical parts; got an out-of-range owner"
                )

    def validate_restored(self, *, collective: bool = False) -> None:
        """Validate all numerical contracts and raw-slot alignment."""
        with _migration_local_validation(collective):
            validate(self)
            if collective:
                _require_migration_arrays_live(self)
        _migration_optional_owner(
            collective, self.forest is not None, "MigrationWorkState.forest"
        )
        self._require_invariants(collective=collective)

    @classmethod
    def from_stored_fields(
        cls, value: MigrationWorkState, /, *, collective: bool = False
    ) -> MigrationWorkState:
        with _migration_local_validation(collective):
            if type(value) is not cls:
                raise TypeError("MigrationWorkState.stored_fields must retain its exact owner type.")
            validate(value)
        if collective:
            value.validate_restored(collective=True)
            with _migration_local_validation(True):
                from .._model._structure import restore_registered_fields

                return restore_registered_fields(
                    cls, {field.name: getattr(value, field.name) for field in fields(value)}
                )
        return cls(
            MigrationWorkset.from_stored_fields(value.workset),
            value.cell_fields,
            value.vertex_fields,
            value.cell_epochs,
            value.vertex_owners,
            value.forest,
            value.forest_cell_owners,
            value.forest_vertex_owners,
            value.cell_history,
            value.vertex_history,
        )

    @classmethod
    def from_live(cls, value: OwnerLocalMigrationState, /) -> MigrationWorkState:
        from ._distribution_migration import OwnerLocalMigrationState

        match value:
            case OwnerLocalMigrationState():
                return cls(
                    MigrationWorkset.from_live(value.workset),
                    value.cell_fields,
                    value.vertex_fields,
                    value.cell_epochs,
                    value.vertex_owners,
                    value.forest,
                    value.forest_cell_owners,
                    value.forest_vertex_owners,
                    value.cell_history,
                    value.vertex_history,
                )
            case _:
                raise TypeError(
                    f"MigrationWorkState.from_live.value must be OwnerLocalMigrationState; got {type(value).__qualname__}"
                )

    def to_live(self) -> OwnerLocalMigrationState:
        from ._distribution_migration import OwnerLocalMigrationState

        self.validate_restored()
        return OwnerLocalMigrationState(
            self.workset.to_live(),
            self.cell_fields,
            self.vertex_fields,
            self.cell_epochs,
            self.vertex_owners,
            self.forest,
            self.forest_cell_owners,
            self.forest_vertex_owners,
            self.cell_history,
            self.vertex_history,
        )


@final
class MigrationTransportEvidence(StrictModule, NonTrainableState):
    """Retain collective decisions, measured counts, and transport controls."""

    __strict_contract__ = True

    status: Int32[_MigrationPartDim]
    accepted: Bool[_MigrationPartDim]
    owned_before: Int32[_MigrationPartDim]
    owned_after: Int32[_MigrationPartDim]
    resident_after: Int32[_MigrationPartDim]
    vertices_after: Int32[_MigrationPartDim]
    sent_cells: Int32[_MigrationPartDim]
    received_cells: Int32[_MigrationPartDim]
    forest_cells_before: Int32[_MigrationPartDim]
    forest_cells_after: Int32[_MigrationPartDim]
    forest_vertices_before: Int32[_MigrationPartDim]
    forest_vertices_after: Int32[_MigrationPartDim]
    halo_width: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    vertex_capacity: int = eqx.field(static=True)
    forest_cell_capacity: int = eqx.field(static=True)
    forest_vertex_capacity: int = eqx.field(static=True)
    route_metadata_bytes: int = eqx.field(static=True)

    def __post_init__(self) -> None:
        self.validate_restored()

    def _require_invariants(self, *, collective: bool = False) -> None:
        with _migration_local_validation(collective):
            controls = (
                self.halo_width,
                self.cell_capacity,
                self.vertex_capacity,
                self.forest_cell_capacity,
                self.forest_vertex_capacity,
                self.route_metadata_bytes,
            )
            if any(type(value) is not int for value in controls):
                raise TypeError(
                    f"MigrationTransportEvidence.controls must be plain integers; got {controls!r}"
                )
            if (
                min(controls) < 0
                or self.cell_capacity < 1
                or self.vertex_capacity < 1
                or self.status.shape[0] < 1
            ):
                raise ValueError(
                    f"MigrationTransportEvidence.controls must have nonnegative controls and nonempty part/cell/vertex capacities; got {controls!r}"
                )
            counts = (
                self.owned_before,
                self.owned_after,
                self.resident_after,
                self.vertices_after,
                self.sent_cells,
                self.received_cells,
                self.forest_cells_before,
                self.forest_cells_after,
                self.forest_vertices_before,
                self.forest_vertices_after,
            )
            accepted = self.accepted
        invalid = bool(jax.device_get(_transport_decision_invalid(
            self.status, accepted, counts
        )))
        with _migration_local_validation(collective):
            if invalid:
                raise ValueError(
                    "MigrationTransportEvidence.decision_counts must retain nonnegative flags/counts and acceptance equal to zero status; got inconsistent evidence"
                )
        invalid = bool(jax.device_get(_transport_owned_invalid(
            self.owned_after, self.resident_after
        )))
        with _migration_local_validation(collective):
            if invalid:
                raise ValueError(
                    "MigrationTransportEvidence.owned_after must not exceed resident_after; got inconsistent cell counts"
                )
        for count, capacity in (
            (self.resident_after, self.cell_capacity),
            (self.vertices_after, self.vertex_capacity),
            (self.forest_cells_after, self.forest_cell_capacity),
            (self.forest_vertices_after, self.forest_vertex_capacity),
        ):
            invalid = bool(jax.device_get(_transport_capacity_invalid(
                accepted, count, capacity
            )))
            with _migration_local_validation(collective):
                if invalid:
                    raise ValueError(
                        f"MigrationTransportEvidence.accepted_counts must fit recorded capacities; got an accepted count above {capacity}"
                    )

    def validate_restored(self, *, collective: bool = False) -> None:
        """Validate counter contracts and the recorded acceptance decision."""
        with _migration_local_validation(collective):
            validate(self)
            if collective:
                _require_migration_arrays_live(self)
        self._require_invariants(collective=collective)

    @classmethod
    def from_stored_fields(
        cls, value: MigrationTransportEvidence, /, *, collective: bool = False
    ) -> MigrationTransportEvidence:
        with _migration_local_validation(collective):
            if type(value) is not cls:
                raise TypeError("MigrationTransportEvidence.stored_fields must retain its exact owner type.")
            validate(value)
        if collective:
            value.validate_restored(collective=True)
            with _migration_local_validation(True):
                from .._model._structure import restore_registered_fields

                return restore_registered_fields(
                    cls, {field.name: getattr(value, field.name) for field in fields(value)}
                )
        return cls(
            value.status,
            value.accepted,
            value.owned_before,
            value.owned_after,
            value.resident_after,
            value.vertices_after,
            value.sent_cells,
            value.received_cells,
            value.forest_cells_before,
            value.forest_cells_after,
            value.forest_vertices_before,
            value.forest_vertices_after,
            value.halo_width,
            value.cell_capacity,
            value.vertex_capacity,
            value.forest_cell_capacity,
            value.forest_vertex_capacity,
            value.route_metadata_bytes,
        )

    @classmethod
    def from_live(
        cls, value: OwnerLocalMigrationEvidence, /
    ) -> MigrationTransportEvidence:
        from ._distribution_migration import OwnerLocalMigrationEvidence

        match value:
            case OwnerLocalMigrationEvidence():
                return cls(
                    value.status,
                    value.accepted,
                    value.owned_before,
                    value.owned_after,
                    value.resident_after,
                    value.vertices_after,
                    value.sent_cells,
                    value.received_cells,
                    value.forest_cells_before,
                    value.forest_cells_after,
                    value.forest_vertices_before,
                    value.forest_vertices_after,
                    value.halo_width,
                    value.cell_capacity,
                    value.vertex_capacity,
                    value.forest_cell_capacity,
                    value.forest_vertex_capacity,
                    value.route_metadata_bytes,
                )
            case _:
                raise TypeError(
                    f"MigrationTransportEvidence.from_live.value must be OwnerLocalMigrationEvidence; got {type(value).__qualname__}"
                )

    def to_live(self) -> OwnerLocalMigrationEvidence:
        from ._distribution_migration import OwnerLocalMigrationEvidence

        self.validate_restored()
        return OwnerLocalMigrationEvidence(
            self.status,
            self.accepted,
            self.owned_before,
            self.owned_after,
            self.resident_after,
            self.vertices_after,
            self.sent_cells,
            self.received_cells,
            self.forest_cells_before,
            self.forest_cells_after,
            self.forest_vertices_before,
            self.forest_vertices_after,
            self.halo_width,
            self.cell_capacity,
            self.vertex_capacity,
            self.forest_cell_capacity,
            self.forest_vertex_capacity,
            self.route_metadata_bytes,
        )


MIGRATION_STATE_ARTIFACT_TYPES: tuple[type, ...] = (
    MigrationWorkset,
    MigrationWorkState,
    MigrationTransportEvidence,
)
