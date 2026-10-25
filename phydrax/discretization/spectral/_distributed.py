#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from math import prod
from operator import index
from typing import Any, cast, final, Literal, TypeAlias, TypedDict, Unpack

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from ..._execution_collectives import CollectiveBinding, require_collective_lease
from ..._execution_device import device_platform
from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import canonical_fingerprint
from ..._jax_context import inside_jax_transformation
from ..._spectral._fourier import resize_fourier_axis
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...typing import checked, parse
from ._cudecomp import (
    CudecompTransposeEvidence,
    prepare_cudecomp_execution,
    PreparedCudecompExecution,
)
from ._precision import SpectralPrecisionPolicy


SpectralSchedule: TypeAlias = Literal["slab", "pencil", "channel"]
SpectralRepresentation: TypeAlias = Literal["physical", "modal"]
SpectralTransposeProvider: TypeAlias = Literal["native", "cudecomp"]
_SpectralStageOperation: TypeAlias = Literal["fft", "transpose"]
type _SpectralTransposePacking = Literal["direct", "pencil-to-slab", "slab-to-pencil"]


class _ForwardedPlanOptions(TypedDict, total=False):
    padded_shape: Sequence[int] | None
    state_shape: Sequence[int]
    maximum_bytes: int
    horizontal_axes: Sequence[int]
    transpose_provider: SpectralTransposeProvider
    collective_binding: CollectiveBinding | None
    maximum_external_scratch_bytes: int


class _FromDiscretizationOptions(_ForwardedPlanOptions, total=False):
    schedule: SpectralSchedule


def _positive_shape(shape: Sequence[int], owner: str, /) -> tuple[int, ...]:
    result = tuple(index(value) for value in shape)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{owner} must contain positive dimensions.")
    return result


def _payload_shapes(
    shapes: Sequence[Sequence[int]], state_shape: tuple[int, ...], /
) -> tuple[tuple[int, ...], ...]:
    canonical = tuple(
        sorted(
            {
                state_shape,
                *(tuple(index(dimension) for dimension in shape) for shape in shapes),
            }
        )
    )
    if any(any(dimension <= 0 for dimension in shape) for shape in canonical):
        raise ValueError(
            "admitted_payload_shapes must contain exact shapes with positive dimensions."
        )
    return canonical


def _require_precision_available(precision: SpectralPrecisionPolicy, /) -> None:
    requested = (
        precision.physical_dtype,
        precision.coefficient_dtype,
        precision.transform_dtype,
        precision.nonlinear_dtype,
        precision.reduction_dtype,
        precision.certification_dtype,
        precision.output_dtype,
        precision.checkpoint_dtype,
    )
    unavailable = tuple(
        dtype
        for dtype in requested
        if np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype))).name != dtype
    )
    if unavailable:
        raise ValueError(
            "The active JAX dtype policy cannot honor requested spectral precision: "
            + ", ".join(unavailable)
            + "."
        )


def _mesh_entry_size(entry: str | tuple[str, ...] | None, shape: dict[str, int]) -> int:
    if entry is None:
        return 1
    if isinstance(entry, str):
        return shape[entry]
    return prod(shape[name] for name in entry)


def _partition_entry(entry: Any, /) -> str | tuple[str, ...] | None:
    if entry is None or isinstance(entry, str):
        return entry
    value = tuple(str(name) for name in entry)
    if not value or any(not name for name in value):
        raise ValueError("Tuple PartitionSpec entries must contain non-empty names.")
    return value


@final
class SpectralMeshTopology(StrictModule, NonTrainableState):
    """Caller-visible device mesh with a stable, hardware-exact identity.

    ``lanes`` declares a lane-reference topology: one device whose single mesh
    axis is executed as ``lanes`` named vmap lanes by the caller's owner region
    (``DistributedOwnershipPlan(owner_lanes=...)``). Its layouts, transposes and
    resource reports describe the ``lanes``-way partition exactly, and
    ``forward_local``/``inverse_local`` run unchanged inside the lane map with the
    same all-to-all transposes. It is a numerical oracle, not distributed
    through :meth:`require_available`.

    ``owner_axis`` declares an owner-region topology (:meth:`owner_region`):
    its mesh coordinates FACTOR one bound owner axis of size ``prod(mesh_shape)``
    with the first mesh axis fastest (owner ``r`` is coordinate
    ``(r % px, r // px)`` of a ``(px, py)`` grid). That is exactly the
    flattened order of the canonical full-mesh pencil all-to-all over
    ``(axis_1, axis_0)``, so a ``DistributedOwnershipPlan`` region whose single
    owner axis carries this name executes the canonical pencil sequence with
    the identical collective. It runs only inside that owner region.
    """

    mesh: Mesh = eqx.field(static=True)
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    mesh_axis_names: tuple[str, ...] = eqx.field(static=True)
    device_ids: tuple[int, ...] = eqx.field(static=True)
    platform: str = eqx.field(static=True)
    device_keys: tuple[tuple[int, int], ...] = eqx.field(static=True)
    execution_group_id: str | None = eqx.field(static=True)
    lanes: int | None = eqx.field(static=True)
    owner_axis: str | None = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    # Caller meshes, physical owner regions and lane references have distinct
    # identity rules. Keep their exhaustive admission and device ordering together
    # so an equal integer shape cannot silently substitute one topology for another.
    def __init__(
        self,
        mesh_or_shape: Mesh | Sequence[int] = (1,),
        /,
        *,
        devices: Sequence[jax.Device] | None = None,
        axis_names: Sequence[str] | None = None,
        execution_group_id: str | None = None,
        lanes: int | None = None,
        owner_axis: str | None = None,
    ) -> None:
        owner_axis_: str | None = None
        if owner_axis is not None:
            if not isinstance(owner_axis, str) or not owner_axis:
                raise TypeError("owner_axis must be a non-empty string.")
            if isinstance(mesh_or_shape, Mesh):
                raise ValueError(
                    "An owner-region topology factors an owner axis; it takes a grid "
                    "shape and owner-ordered devices, not a caller-owned Mesh."
                )
            if devices is None:
                raise ValueError("An owner-region topology needs its owner devices.")
            owner_axis_ = owner_axis
        lane_count: int | None = None
        if lanes is not None:
            lane_count = index(lanes)
            if lane_count <= 0:
                raise ValueError("lanes must be positive.")
            if isinstance(mesh_or_shape, Mesh):
                raise ValueError(
                    "A lane-reference topology is the one-axis shape (lanes,)."
                )
            lane_shape = tuple(index(value) for value in mesh_or_shape)
            if owner_axis_ is None and lane_shape != (lane_count,):
                raise ValueError(
                    "A lane-reference topology is the one-axis shape (lanes,)."
                )
            if owner_axis_ is not None and prod(lane_shape) != lane_count:
                raise ValueError(
                    "An owner-region lane topology factors exactly its lane count."
                )
            if devices is None or len(tuple(devices)) != 1:
                raise ValueError("A lane-reference topology executes on one device.")
        if isinstance(mesh_or_shape, Mesh):
            if devices is not None or axis_names is not None:
                raise ValueError(
                    "A caller-owned Mesh already fixes devices and axis names."
                )
            mesh = mesh_or_shape
            names = tuple(str(name) for name in mesh.axis_names)
            mesh_shape = tuple(mesh.shape[name] for name in names)
            selected = tuple(mesh.devices.flat)
        else:
            mesh_shape = _positive_shape(mesh_or_shape, "mesh_shape")
            names = (
                tuple(f"spectral_{axis}" for axis in range(len(mesh_shape)))
                if axis_names is None
                else tuple(str(name) for name in axis_names)
            )
            if (
                len(names) != len(mesh_shape)
                or any(not name for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError("axis_names must uniquely name every mesh dimension.")
            if owner_axis_ is not None and owner_axis_ in names:
                raise ValueError(
                    "The owner axis is bound by the owner region; it cannot also name "
                    "a factored spectral mesh axis."
                )
            available = tuple(jax.devices())
            requested = 1 if lane_count is not None else prod(mesh_shape)
            if devices is None:
                if requested > len(available):
                    raise RuntimeError(
                        f"Distributed spectral topology requests {requested} devices, "
                        f"but this process exposes {len(available)}. Configure JAX devices "
                        "before importing JAX; distribution is never simulated."
                    )
                selected = available[:requested]
            else:
                selected = tuple(devices)
            if len(selected) != requested:
                raise ValueError(
                    "mesh_shape product must equal the selected device count."
                )
            if len({(device.process_index, device.id) for device in selected}) != len(
                selected
            ):
                raise ValueError("A spectral topology cannot contain a device twice.")
            if lane_count is not None:
                arranged = np.asarray(selected, dtype=object).reshape(
                    (1,) * len(mesh_shape)
                )
            elif owner_axis_ is not None:
                # Owner r sits at the coordinate whose FIRST axis varies fastest.
                arranged = (
                    np.asarray(selected, dtype=object)
                    .reshape(tuple(reversed(mesh_shape)))
                    .transpose(tuple(reversed(range(len(mesh_shape)))))
                )
            else:
                arranged = np.asarray(selected, dtype=object).reshape(mesh_shape)
            mesh = Mesh(arranged, names)
        if not selected:
            raise ValueError("A spectral topology requires at least one device.")
        platforms = {device.platform for device in selected}
        if len(platforms) != 1:
            raise ValueError("A spectral topology must use devices from one platform.")
        ids = tuple(int(device.id) for device in selected)
        keys = tuple((int(device.process_index), int(device.id)) for device in selected)
        platform = selected[0].platform
        self.mesh = mesh
        self.mesh_shape = mesh_shape
        self.mesh_axis_names = names
        self.device_ids = ids
        self.device_keys = keys
        self.execution_group_id = execution_group_id
        self.lanes = lane_count
        self.owner_axis = owner_axis_
        self.platform = platform
        self.topology_id = canonical_fingerprint(
            {
                "kind": "spectral-mesh-topology",
                "mesh_shape": list(mesh_shape),
                "axis_names": list(names),
                "platform": platform,
                "device_ids": list(ids),
                "device_keys": [list(key) for key in keys],
                "execution_group_id": execution_group_id,
                **({} if lane_count is None else {"lane_reference": lane_count}),
                **(
                    {}
                    if owner_axis_ is None
                    else {
                        "owner_axis": owner_axis_,
                        "owner_order": "first-mesh-axis-fastest",
                    }
                ),
            }
        )

    @classmethod
    def lane_reference(
        cls,
        lanes: int,
        /,
        *,
        axis_name: str,
        device: jax.Device | None = None,
        execution_group_id: str | None = None,
    ) -> "SpectralMeshTopology":
        """One device whose axis ``axis_name`` runs as ``lanes`` named vmap lanes."""
        selected = jax.devices()[0] if device is None else device
        return cls(
            (lanes,),
            devices=(selected,),
            axis_names=(axis_name,),
            execution_group_id=execution_group_id,
            lanes=lanes,
        )

    @classmethod
    def owner_region(
        cls,
        grid: Sequence[int],
        /,
        *,
        owner_axis: str,
        devices: Sequence[jax.Device],
        lanes: bool = False,
        axis_names: Sequence[str] | None = None,
        execution_group_id: str | None = None,
    ) -> "SpectralMeshTopology":
        """Spectral grid factoring one bound owner axis (first axis fastest).

        ``devices`` are the owner devices in owner order (one per owner), or
        the single device of a lane-reference owner region (``lanes=True``,
        ``prod(grid)`` named vmap lanes). Owner ``r`` holds the layout block
        of coordinate ``(r % grid[0], r // grid[0])`` for a two-axis grid.
        """
        shape = _positive_shape(grid, "grid")
        selected = tuple(devices)
        if not lanes and len(selected) != prod(shape):
            raise ValueError("An owner-region topology needs one device per owner.")
        names = (
            tuple(f"{owner_axis}_spectral_{axis}" for axis in range(len(shape)))
            if axis_names is None
            else tuple(axis_names)
        )
        return cls(
            shape,
            devices=selected,
            axis_names=names,
            execution_group_id=execution_group_id,
            lanes=prod(shape) if lanes else None,
            owner_axis=owner_axis,
        )

    @classmethod
    def from_execution_group(
        cls,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_names: Sequence[str] | None = None,
    ) -> SpectralMeshTopology:
        mesh = execution_group.mesh
        if axis_names is not None:
            shape = tuple(mesh.shape[name] for name in mesh.axis_names)
            mesh = Mesh(
                np.asarray(execution_group.devices, dtype=object).reshape(shape),
                axis_names,
            )
        return cls(
            mesh,
            execution_group_id=execution_group.spec.group_id,
        )

    @classmethod
    def one_device(cls, device: jax.Device | None = None, /) -> "SpectralMeshTopology":
        selected = jax.devices()[0] if device is None else device
        return cls((1,), devices=(selected,), axis_names=("spectral",))

    @property
    def device_count(self) -> int:
        return prod(self.mesh_shape)

    def require_available(self, /) -> None:
        if self.lanes is not None:
            raise ValueError(
                "A lane-reference spectral topology has no device mesh: it executes "
                "only inside a lane owner region through forward_local/inverse_local."
            )
        if self.owner_axis is not None:
            raise ValueError(
                "An owner-region spectral topology factors the owner axis "
                f"{self.owner_axis!r}: it executes only inside that owner region "
                "through forward_local/inverse_local."
            )
        available = {
            (device.platform, int(device.process_index), int(device.id))
            for device in jax.devices(self.platform)
        }
        missing = tuple(
            key
            for key in self.device_keys
            if (self.platform, key[0], key[1]) not in available
        )
        if missing:
            raise RuntimeError(
                "Spectral topology devices are unavailable in the current JAX "
                f"process: {missing}."
            )


@final
class SpectralLayout(StrictModule, NonTrainableState):
    """Global array shape and its exact named-mesh partition contract."""

    global_shape: tuple[int, ...] = eqx.field(static=True)
    partition: tuple[str | tuple[str, ...] | None, ...] = eqx.field(static=True)
    representation: SpectralRepresentation = eqx.field(static=True)
    padded: bool = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        global_shape: Sequence[int],
        partition: Sequence[str | tuple[str, ...] | None],
        representation: SpectralRepresentation,
        topology: SpectralMeshTopology,
        /,
        *,
        padded: bool = False,
    ) -> None:
        shape = _positive_shape(global_shape, "global_shape")
        entries = tuple(_partition_entry(value) for value in partition)
        if len(entries) != len(shape):
            raise ValueError("partition must explicitly describe every array dimension.")
        representation = parse(representation, SpectralRepresentation, "representation")
        mesh_sizes = dict(zip(topology.mesh_axis_names, topology.mesh_shape, strict=True))
        used: list[str] = []
        for entry in entries:
            names = () if entry is None else (entry,) if isinstance(entry, str) else entry
            if any(name not in mesh_sizes for name in names):
                raise ValueError("partition refers to an axis outside the spectral mesh.")
            used.extend(names)
        if len(used) != len(set(used)):
            raise ValueError(
                "Each nontrivial mesh axis must partition exactly one dimension."
            )
        if any(
            size % _mesh_entry_size(entry, mesh_sizes)
            for size, entry in zip(shape, entries, strict=True)
        ):
            raise ValueError("Every partition factor must divide its global dimension.")
        self.global_shape = shape
        self.partition = entries
        self.representation = representation
        self.padded = bool(padded)
        self.topology_id = topology.topology_id
        self.layout_id = canonical_fingerprint(
            {
                "kind": "distributed-spectral-layout",
                "shape": list(shape),
                "partition": [
                    None
                    if value is None
                    else list(value)
                    if isinstance(value, tuple)
                    else value
                    for value in entries
                ],
                "representation": representation,
                "padded": bool(padded),
                "topology": topology.topology_id,
            }
        )

    @property
    def partition_spec(self) -> PartitionSpec:
        return PartitionSpec(*self.partition)

    @property
    def used_mesh_axes(self) -> tuple[str, ...]:
        result: list[str] = []
        for entry in self.partition:
            if isinstance(entry, str):
                result.append(entry)
            elif isinstance(entry, tuple):
                result.extend(entry)
        return tuple(result)

    def sharding(self, topology: SpectralMeshTopology, /) -> NamedSharding:
        if topology.topology_id != self.topology_id:
            raise ValueError("Spectral layout/topology identity mismatch.")
        return NamedSharding(topology.mesh, self.partition_spec)

    def local_shape(self, topology: SpectralMeshTopology, /) -> tuple[int, ...]:
        if topology.topology_id != self.topology_id:
            raise ValueError("Spectral layout/topology identity mismatch.")
        mesh_sizes = dict(zip(topology.mesh_axis_names, topology.mesh_shape, strict=True))
        return tuple(
            size // _mesh_entry_size(entry, mesh_sizes)
            for size, entry in zip(self.global_shape, self.partition, strict=True)
        )


@final
class SpectralTranspose(StrictModule, NonTrainableState):
    """One all-to-all redistribution derived from declared layouts and topology.

    ``collective_axis`` is the bound axis that executes it: the moved mesh
    axis (or the full-mesh ``(axis_1, axis_0)`` pair), or the owner axis of an
    owner-region topology, whose flattened order is the same.
    """

    source: SpectralLayout
    target: SpectralLayout
    mesh_axis: str | tuple[str, ...] = eqx.field(static=True)
    collective_axis: str | tuple[str, ...] = eqx.field(static=True)
    split_axis: int = eqx.field(static=True)
    concat_axis: int = eqx.field(static=True)
    packing: _SpectralTransposePacking = eqx.field(static=True)
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    transpose_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: SpectralLayout,
        target: SpectralLayout,
        topology: SpectralMeshTopology,
        /,
    ) -> None:
        if not isinstance(source, SpectralLayout) or not isinstance(
            target, SpectralLayout
        ):
            raise TypeError("source and target must be SpectralLayout values.")
        if not isinstance(topology, SpectralMeshTopology):
            raise TypeError("topology must be a SpectralMeshTopology.")
        if (
            source.topology_id != target.topology_id
            or source.topology_id != topology.topology_id
        ):
            raise ValueError(
                "Spectral transpose layouts must share the supplied topology."
            )
        if source.global_shape != target.global_shape or source.padded != target.padded:
            raise ValueError(
                "Spectral transpose layouts must describe the same global array."
            )

        def entries(layout: SpectralLayout) -> tuple[tuple[str, ...], ...]:
            return tuple(
                () if value is None else (value,) if isinstance(value, str) else value
                for value in layout.partition
            )

        source_entries, target_entries = entries(source), entries(target)
        source_axes = {
            name: axis for axis, names in enumerate(source_entries) for name in names
        }
        target_axes = {
            name: axis for axis, names in enumerate(target_entries) for name in names
        }
        if source_axes.keys() != target_axes.keys():
            raise ValueError(
                "An all-to-all transpose must preserve participating mesh axes."
            )
        moved = tuple(
            name
            for name in topology.mesh_axis_names
            if name in source_axes and source_axes[name] != target_axes[name]
        )
        packing: _SpectralTransposePacking
        mesh_axis: str | tuple[str, ...]
        mesh_shape: tuple[int, ...]
        if len(moved) == 1:
            mesh_axis = moved[0]
            split, concat = target_axes[mesh_axis], source_axes[mesh_axis]
            expected = list(source_entries)
            if expected[concat][-1] != mesh_axis:
                raise ValueError("A direct transpose moves the innermost partition axis.")
            expected[concat] = expected[concat][:-1]
            expected[split] = (*expected[split], mesh_axis)
            if tuple(expected) != target_entries:
                raise ValueError("The target layout does not match a direct all-to-all.")
            packing, mesh_shape = "direct", ()
        elif (
            len(moved) == 2 and len(topology.mesh_shape) == 2 and len(source_entries) >= 3
        ):
            if len(source_entries[0]) == len(source_entries[1]) == 1:
                first, second = source_entries[0][0], source_entries[1][0]
                packing, split, concat = "pencil-to-slab", 3, 0
            elif len(target_entries[0]) == len(target_entries[1]) == 1:
                first, second = target_entries[0][0], target_entries[1][0]
                packing, split, concat = "slab-to-pencil", 0, 3
            else:
                raise ValueError(
                    "A full-mesh transpose requires a declared xy-pencil layout."
                )
            physical = ((first,), (second,), ()) + ((),) * (len(source_entries) - 3)
            modal = ((), (), (second, first)) + ((),) * (len(source_entries) - 3)
            expected_source, expected_target = (
                (physical, modal) if packing == "pencil-to-slab" else (modal, physical)
            )
            if source_entries != expected_source or target_entries != expected_target:
                raise ValueError(
                    "A full-mesh transpose requires canonical pencil and z-slab layouts."
                )
            mesh_axis = (second, first)
            sizes = dict(zip(topology.mesh_axis_names, topology.mesh_shape, strict=True))
            mesh_shape = (sizes[first], sizes[second])
        else:
            raise ValueError(
                "Layouts must describe a direct or full-mesh pencil transpose."
            )
        collective_axis: str | tuple[str, ...] = mesh_axis
        if topology.owner_axis is not None:
            names = topology.mesh_axis_names
            full_mesh = (
                mesh_axis == names[0]
                if len(names) == 1
                else len(names) == 2 and mesh_axis == (names[1], names[0])
            )
            if not full_mesh:
                raise ValueError(
                    "An owner-region topology executes only full-mesh transposes whose "
                    "flattened order is its owner order."
                )
            collective_axis = topology.owner_axis

        identity: dict[str, Any] = {
            "kind": "spectral-all-to-all-transpose",
            "source": source.layout_id,
            "target": target.layout_id,
            "mesh_axis": mesh_axis if isinstance(mesh_axis, str) else list(mesh_axis),
            "split_axis": split,
            "concat_axis": concat,
        }
        if packing != "direct":
            identity["packing"] = packing
            identity["mesh_shape"] = list(mesh_shape)
        if topology.owner_axis is not None:
            identity["collective_axis"] = topology.owner_axis
        self.source = source
        self.target = target
        self.mesh_axis = mesh_axis
        self.collective_axis = collective_axis
        self.split_axis = split
        self.concat_axis = concat
        self.packing = packing
        self.mesh_shape = mesh_shape
        self.transpose_id = canonical_fingerprint(identity)

    def apply_local(self, local_value: Array, /) -> Array:
        if self.packing == "direct":
            return jax.lax.all_to_all(
                local_value,
                axis_name=self.collective_axis,
                split_axis=self.split_axis,
                concat_axis=self.concat_axis,
                tiled=True,
            )
        px, py = self.mesh_shape
        nx, ny, nz, *payload = local_value.shape
        if self.packing == "pencil-to-slab":
            packed = jax.lax.all_to_all(
                local_value[None],
                axis_name=self.collective_axis,
                split_axis=self.split_axis,
                concat_axis=self.concat_axis,
                tiled=True,
            )
            packed = packed.reshape(py, px, nx, ny, nz // (px * py), *payload)
            packed = packed.transpose(1, 2, 0, 3, 4, *range(5, packed.ndim))
            return packed.reshape(px * nx, py * ny, nz // (px * py), *payload)
        if self.packing != "slab-to-pencil":
            raise ValueError(f"Invalid spectral transpose packing {self.packing!r}.")
        packed = local_value.reshape(px, nx // px, py, ny // py, nz, *payload)
        packed = packed.transpose(2, 0, 1, 3, 4, *range(5, packed.ndim))
        packed = packed.reshape(px * py, nx // px, ny // py, nz, *payload)
        return jax.lax.all_to_all(
            packed,
            axis_name=self.collective_axis,
            split_axis=self.split_axis,
            concat_axis=self.concat_axis,
            tiled=True,
        )[0]

    def execute(self, values: ArrayLike, topology: SpectralMeshTopology, /) -> Array:
        if topology.topology_id != self.source.topology_id:
            raise ValueError("Spectral transpose/topology identity mismatch.")
        value = jnp.asarray(values)
        if value.shape != self.source.global_shape:
            raise ValueError(
                f"Transpose input must have shape {self.source.global_shape}; got {value.shape}."
            )
        topology.require_available()
        placed = jax.device_put(value, self.source.sharding(topology))
        mapped = jax.shard_map(
            self.apply_local,
            mesh=topology.mesh,
            in_specs=self.source.partition_spec,
            out_specs=self.target.partition_spec,
            check_vma=False,
        )
        return mapped(placed)


@final
class _SpectralStage(StrictModule, NonTrainableState):
    operation: _SpectralStageOperation = eqx.field(static=True)
    axes: tuple[int, ...] | None = eqx.field(static=True)
    transpose: SpectralTranspose | None
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        operation: _SpectralStageOperation,
        /,
        *,
        axes: Sequence[int] | None = None,
        transpose: SpectralTranspose | None = None,
    ) -> None:
        operation_ = parse(operation, _SpectralStageOperation, "operation")
        if operation_ == "fft":
            axes_ = () if axes is None else tuple(index(axis) for axis in axes)
            if (
                not axes_
                or transpose is not None
                or any(axis < 0 for axis in axes_)
                or len(set(axes_)) != len(axes_)
            ):
                raise ValueError(
                    "An FFT stage requires distinct ordered nonnegative axes."
                )
        else:
            if axes is not None or not isinstance(transpose, SpectralTranspose):
                raise ValueError("A transpose stage requires exactly one transpose.")
            axes_ = None
        self.operation = operation_
        self.axes = axes_
        self.transpose = transpose
        self.stage_id = canonical_fingerprint(
            {
                "kind": "distributed-spectral-stage",
                "operation": operation_,
                "axes": None if axes_ is None else list(axes_),
                "transpose": (None if transpose is None else transpose.transpose_id),
            }
        )

    def apply_local(self, value: Array, /, *, inverse: bool) -> Array:
        if self.operation == "transpose":
            if self.transpose is None:
                raise RuntimeError("Prepared transpose stage is incomplete.")
            return self.transpose.apply_local(value)
        if self.axes is None:
            raise RuntimeError("Prepared FFT stage is incomplete.")
        transform = jnp.fft.ifft if inverse else jnp.fft.fft
        result = value
        for axis in self.axes:
            result = transform(result, axis=axis, norm="ortho")
        return result


def _sequence_id(
    sequence: tuple[_SpectralStage, ...],
    direction: str,
    source: SpectralLayout,
    target: SpectralLayout,
    precision: SpectralPrecisionPolicy,
    scale: float,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "distributed-spectral-stage-sequence",
            "direction": direction,
            "source": source.layout_id,
            "target": target.layout_id,
            "padded": source.padded,
            "transform_dtype": precision.transform_dtype,
            "normalization": "ortho",
            "scale": scale,
            "stages": [stage.stage_id for stage in sequence],
        }
    )


def _reverse_sequence(
    forward: tuple[_SpectralStage, ...],
    transpose_pairs: Sequence[tuple[SpectralTranspose, SpectralTranspose]],
    /,
    *,
    symmetric_suffix_count: int = 0,
) -> tuple[_SpectralStage, ...]:
    suffix_count = index(symmetric_suffix_count)
    if suffix_count < 0 or suffix_count > len(forward):
        raise ValueError("symmetric_suffix_count lies outside the stage sequence.")
    core = forward if suffix_count == 0 else forward[:-suffix_count]
    suffix = () if suffix_count == 0 else forward[-suffix_count:]
    reverse: list[_SpectralStage] = []
    for stage in reversed(core):
        if stage.operation == "fft":
            reverse.append(stage)
            continue
        if stage.transpose is None:
            raise RuntimeError("Prepared transpose stage is incomplete.")
        matches = tuple(
            inverse
            for direct, inverse in transpose_pairs
            if direct.transpose_id == stage.transpose.transpose_id
        )
        if len(matches) != 1:
            raise RuntimeError("Prepared stage sequence lacks one reverse transpose.")
        reverse.append(_SpectralStage("transpose", transpose=matches[0]))
    reverse.extend(suffix)
    return tuple(reverse)


def _slab_sequence_pair(
    physical: SpectralLayout,
    modal: SpectralLayout,
    topology: SpectralMeshTopology,
    spatial_rank: int,
    /,
) -> tuple[tuple[_SpectralStage, ...], tuple[_SpectralStage, ...]]:
    forward_transpose = SpectralTranspose(physical, modal, topology)
    inverse_transpose = SpectralTranspose(modal, physical, topology)
    forward = (
        _SpectralStage("fft", axes=range(1, spatial_rank)),
        _SpectralStage("transpose", transpose=forward_transpose),
        _SpectralStage("fft", axes=(0,)),
    )
    inverse = _reverse_sequence(forward, ((forward_transpose, inverse_transpose),))
    return forward, inverse


def _pencil_sequence_pair(
    physical: SpectralLayout,
    modal: SpectralLayout,
    topology: SpectralMeshTopology,
    spatial_rank: int,
    /,
) -> tuple[tuple[_SpectralStage, ...], tuple[_SpectralStage, ...]]:
    forward_transpose = SpectralTranspose(physical, modal, topology)
    inverse_transpose = SpectralTranspose(modal, physical, topology)
    suffix = (
        () if spatial_rank == 3 else (_SpectralStage("fft", axes=range(3, spatial_rank)),)
    )
    forward = (
        _SpectralStage("fft", axes=(2,)),
        _SpectralStage("transpose", transpose=forward_transpose),
        _SpectralStage("fft", axes=(1,)),
        _SpectralStage("fft", axes=(0,)),
        *suffix,
    )
    inverse = _reverse_sequence(
        forward,
        ((forward_transpose, inverse_transpose),),
        symmetric_suffix_count=len(suffix),
    )
    return forward, inverse


@dataclass(frozen=True, slots=True)
class _LayoutPreparation:
    physical: SpectralLayout
    modal: SpectralLayout
    padded_physical: SpectralLayout
    padded_modal: SpectralLayout
    forward: tuple[_SpectralStage, ...]
    inverse: tuple[_SpectralStage, ...]
    padded_forward: tuple[_SpectralStage, ...]
    padded_inverse: tuple[_SpectralStage, ...]
    horizontal_axes: tuple[int, int] | None
    reasons: tuple[str, ...]


def _prepare_layouts(
    topology: SpectralMeshTopology,
    shape: tuple[int, ...],
    padded: tuple[int, ...],
    trailing: tuple[int, ...],
    schedule: SpectralSchedule,
    horizontal_axes: Sequence[int],
    /,
) -> _LayoutPreparation:
    mesh_names = topology.mesh_axis_names
    mesh_shape = topology.mesh_shape
    rank = len(shape)
    full_shape = shape + trailing
    full_padded = padded + trailing
    reasons: list[str] = []
    horizontal: tuple[int, int] | None = None
    if schedule == "pencil":
        if len(mesh_shape) != 2 or rank < 3:
            reasons.append(
                "pencil execution requires a two-dimensional mesh and spatial "
                "rank at least three"
            )
        if len(mesh_shape) == 2 and rank >= 3:
            px, py = mesh_shape
            if shape[0] % px or shape[1] % py or shape[2] % (px * py):
                reasons.append(
                    "canonical pencil dimensions are not divisible by their "
                    "transform partitions"
                )
            if padded[0] % px or padded[1] % py or padded[2] % (px * py):
                reasons.append(
                    "padded pencil dimensions are not divisible by their "
                    "transform partitions"
                )
            physical_partition = (
                mesh_names[0],
                mesh_names[1],
                None,
            ) + (None,) * (rank - 3 + len(trailing))
            modal_partition = (
                None,
                None,
                (mesh_names[1], mesh_names[0]),
            ) + (None,) * (rank - 3 + len(trailing))
        else:
            physical_partition = (None,) * len(full_shape)
            modal_partition = physical_partition
    elif schedule == "channel":
        axes = tuple(index(axis) for axis in horizontal_axes)
        if (
            len(axes) != 2
            or len(set(axes)) != 2
            or any(axis < 0 or axis >= rank for axis in axes)
        ):
            raise ValueError("horizontal_axes must name two distinct spatial axes.")
        first_axis, second_axis = axes
        horizontal = (first_axis, second_axis)
        if rank != 3 or tuple(axis for axis in range(rank) if axis not in axes) != (1,):
            reasons.append(
                "channel execution requires exactly Fourier-Chebyshev-Fourier "
                "axes with replicated Chebyshev axis 1"
            )
        if rank == 3 and padded[1] != shape[1]:
            reasons.append("channel padding cannot resize the replicated Chebyshev axis")
        if len(mesh_shape) not in (1, 2):
            reasons.append("channel execution requires a one- or two-dimensional mesh")
        channel_entries: list[str | tuple[str, ...] | None] = [None] * len(full_shape)
        for mesh_axis, spatial_axis in enumerate(axes[: len(mesh_shape)]):
            count = mesh_shape[mesh_axis]
            if shape[spatial_axis] % count or padded[spatial_axis] % count:
                reasons.append(
                    "channel horizontal dimensions are not divisible by their "
                    "mesh partitions"
                )
            channel_entries[spatial_axis] = mesh_names[mesh_axis]
        physical_partition = tuple(channel_entries)
        modal_partition = physical_partition
    else:
        if len(mesh_shape) != 1:
            reasons.append("slab execution requires a one-dimensional mesh")
        if rank < 2:
            reasons.append("slab execution requires spatial rank at least two")
        first_axis, second_axis = (0, 1)
        count = mesh_shape[0] if len(mesh_shape) == 1 else 1
        if rank >= 2:
            if shape[first_axis] % count or shape[second_axis] % count:
                reasons.append(
                    "canonical slab dimensions are not divisible by the mesh size"
                )
            if padded[first_axis] % count or padded[second_axis] % count:
                reasons.append(
                    "padded slab dimensions are not divisible by the mesh size"
                )
        physical_entries: list[str | tuple[str, ...] | None] = [None] * len(full_shape)
        modal_entries: list[str | tuple[str, ...] | None] = [None] * len(full_shape)
        if len(mesh_shape) == 1 and rank >= 2:
            physical_entries[first_axis] = mesh_names[0]
            modal_entries[second_axis] = mesh_names[0]
        physical_partition = tuple(physical_entries)
        modal_partition = tuple(modal_entries)
    physical = SpectralLayout(full_shape, physical_partition, "physical", topology)
    modal = SpectralLayout(full_shape, modal_partition, "modal", topology)
    padded_physical = SpectralLayout(
        full_padded, physical_partition, "physical", topology, padded=True
    )
    padded_modal = SpectralLayout(
        full_padded, modal_partition, "modal", topology, padded=True
    )
    if schedule == "pencil" and len(mesh_shape) == 2 and rank >= 3:
        forward, inverse = _pencil_sequence_pair(
            physical,
            modal,
            topology,
            rank,
        )
        padded_forward, padded_inverse = _pencil_sequence_pair(
            padded_physical,
            padded_modal,
            topology,
            rank,
        )
    elif schedule == "slab" and len(mesh_shape) == 1 and rank >= 2:
        forward, inverse = _slab_sequence_pair(physical, modal, topology, rank)
        padded_forward, padded_inverse = _slab_sequence_pair(
            padded_physical, padded_modal, topology, rank
        )
    else:
        forward = inverse = padded_forward = padded_inverse = ()
    return _LayoutPreparation(
        physical,
        modal,
        padded_physical,
        padded_modal,
        forward,
        inverse,
        padded_forward,
        padded_inverse,
        horizontal,
        tuple(reasons),
    )


@final
class SpectralResourceReport(StrictModule, NonTrainableState):
    """Fail-closed logical FFT buffer and traffic admission.

    Compiler temporaries, allocator alignment, runtime metadata, and composed
    consumer workflows require separate evidence; this is not a ceiling on
    total compiled-executable or device memory.
    """

    canonical_storage_bytes: int = eqx.field(static=True)
    padded_storage_bytes: int = eqx.field(static=True)
    transform_workspace_bytes: int = eqx.field(static=True)
    collective_payload_bytes: int = eqx.field(static=True)
    peak_live_bytes: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    transpose_packing_bytes: int = eqx.field(static=True)
    external_device_scratch_bytes: int = eqx.field(static=True)
    external_resident_device_bytes: int | None = eqx.field(static=True)
    external_host_bytes: int | None = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        canonical_storage_bytes: int,
        padded_storage_bytes: int,
        transform_workspace_bytes: int,
        collective_payload_bytes: int,
        peak_live_bytes: int,
        maximum_bytes: int,
        reasons: Sequence[str] = (),
        transpose_packing_bytes: int = 0,
        external_device_scratch_bytes: int = 0,
        external_resident_device_bytes: int | None = 0,
        external_host_bytes: int | None = 0,
    ) -> None:
        (
            canonical,
            padded,
            workspace,
            collective,
            peak,
        ) = (
            index(canonical_storage_bytes),
            index(padded_storage_bytes),
            index(transform_workspace_bytes),
            index(collective_payload_bytes),
            index(peak_live_bytes),
        )
        maximum = index(maximum_bytes)
        packing = index(transpose_packing_bytes)
        external_scratch = index(external_device_scratch_bytes)
        if packing < 0 or external_scratch < 0:
            raise ValueError(
                "Transpose packing and external scratch bytes must be nonnegative."
            )
        if any(
            value is not None and value < 0
            for value in (external_resident_device_bytes, external_host_bytes)
        ):
            raise ValueError(
                "External storage bytes must be nonnegative or explicitly unknown."
            )
        if (
            any(value < 0 for value in (canonical, padded, workspace, collective, peak))
            or maximum <= 0
        ):
            raise ValueError(
                "Spectral resource byte counts must be non-negative and bounded."
            )
        if peak != 2 * padded + workspace + external_scratch:
            raise ValueError(
                "peak_live_bytes must count source, target, transform workspace, "
                "and known retained external scratch."
            )
        reasons_ = tuple(str(reason) for reason in reasons)
        if any(not reason for reason in reasons_):
            raise ValueError("Resource refusal reasons must be non-empty.")
        if peak > maximum:
            reasons_ += (f"peak live bytes {peak} exceed maximum_bytes {maximum}",)
        accepted = not reasons_
        self.canonical_storage_bytes = canonical
        self.padded_storage_bytes = padded
        self.transform_workspace_bytes = workspace
        self.collective_payload_bytes = collective
        self.peak_live_bytes = peak
        self.maximum_bytes = maximum
        self.accepted = accepted
        self.transpose_packing_bytes = packing
        self.external_device_scratch_bytes = external_scratch
        self.external_resident_device_bytes = external_resident_device_bytes
        self.external_host_bytes = external_host_bytes
        self.reasons = reasons_
        self.report_id = canonical_fingerprint(
            {
                "kind": "distributed-spectral-resource-report",
                "canonical_storage_bytes": canonical,
                "padded_storage_bytes": padded,
                "transform_workspace_bytes": workspace,
                "collective_payload_bytes": collective,
                "peak_live_bytes": peak,
                "maximum_bytes": maximum,
                "accepted": accepted,
                "reasons": list(reasons_),
                "transpose_packing_bytes": packing,
                "external_device_scratch_bytes": external_scratch,
                "external_resident_device_bytes": external_resident_device_bytes,
                "external_host_bytes": external_host_bytes,
            }
        )


class SpectralResourceError(MemoryError):
    """Typed refusal retaining the exact preflight report."""

    report: SpectralResourceReport

    def __init__(self, report: SpectralResourceReport, /) -> None:
        self.report = report
        super().__init__(
            "Distributed spectral resource preflight refused: "
            + "; ".join(report.reasons)
        )


def _sequence_ids(
    layouts: _LayoutPreparation,
    precision: SpectralPrecisionPolicy,
    scale: float,
    padded_scale: float,
    /,
) -> tuple[str, str, str, str]:
    return (
        _sequence_id(
            layouts.forward,
            "physical-to-modal",
            layouts.physical,
            layouts.modal,
            precision,
            scale,
        ),
        _sequence_id(
            layouts.inverse,
            "modal-to-physical",
            layouts.modal,
            layouts.physical,
            precision,
            scale,
        ),
        _sequence_id(
            layouts.padded_forward,
            "padded-physical-to-modal",
            layouts.padded_physical,
            layouts.padded_modal,
            precision,
            padded_scale,
        ),
        _sequence_id(
            layouts.padded_inverse,
            "padded-modal-to-physical",
            layouts.padded_modal,
            layouts.padded_physical,
            precision,
            padded_scale,
        ),
    )


def _prepare_fft_resource(
    shape: tuple[int, ...],
    padded: tuple[int, ...],
    admitted: tuple[tuple[int, ...], ...],
    precision: SpectralPrecisionPolicy,
    forward: tuple[_SpectralStage, ...],
    maximum: int,
    reasons: Sequence[str],
    /,
) -> tuple[SpectralResourceReport, int, int]:
    payload_elements = max(prod(payload) if payload else 1 for payload in admitted)
    coefficient_itemsize = np.dtype(precision.coefficient_dtype).itemsize
    transform_itemsize = np.dtype(precision.transform_dtype).itemsize
    canonical_storage = prod(shape) * payload_elements * coefficient_itemsize
    padded_storage = prod(padded) * payload_elements * coefficient_itemsize
    has_fft_stage = any(stage.operation == "fft" for stage in forward)
    transform_payload = prod(padded) * payload_elements * transform_itemsize
    # A packed transpose can keep its communication buffer live alongside the
    # reassembled FFT input. Reserve one additional compute-precision buffer.
    packed = any(
        stage.transpose is not None and stage.transpose.packing != "direct"
        for stage in forward
    )
    transform_workspace = transform_payload * (1 + packed) if has_fft_stage else 0
    collective_count = sum(stage.operation == "transpose" for stage in forward)
    collective_payload = transform_payload * collective_count
    resource = SpectralResourceReport(
        canonical_storage_bytes=canonical_storage,
        padded_storage_bytes=padded_storage,
        transform_workspace_bytes=transform_workspace,
        collective_payload_bytes=collective_payload,
        peak_live_bytes=2 * padded_storage + transform_workspace,
        maximum_bytes=maximum,
        reasons=reasons,
    )
    return resource, collective_count, collective_payload


def _with_cudecomp_storage(
    resource: SpectralResourceReport,
    packing_bytes: int,
    external_scratch_bytes: int,
    *,
    reasons: tuple[str, ...] = (),
) -> SpectralResourceReport:
    return SpectralResourceReport(
        canonical_storage_bytes=resource.canonical_storage_bytes,
        padded_storage_bytes=resource.padded_storage_bytes,
        transform_workspace_bytes=resource.transform_workspace_bytes + packing_bytes,
        collective_payload_bytes=resource.collective_payload_bytes,
        peak_live_bytes=resource.peak_live_bytes + packing_bytes + external_scratch_bytes,
        maximum_bytes=resource.maximum_bytes,
        transpose_packing_bytes=packing_bytes,
        external_device_scratch_bytes=external_scratch_bytes,
        external_resident_device_bytes=None,
        external_host_bytes=None,
        reasons=(*resource.reasons, *reasons),
    )


def _prepare_cudecomp_transpose(
    binding: CollectiveBinding,
    topology: SpectralMeshTopology,
    shape: tuple[int, ...],
    padded: tuple[int, ...],
    precision: SpectralPrecisionPolicy,
    admitted: tuple[tuple[int, ...], ...],
    resource: SpectralResourceReport,
    maximum_external_scratch_bytes: int,
    sequence_ids: tuple[str, str, str, str],
) -> tuple[
    PreparedCudecompExecution,
    SpectralResourceReport,
    tuple[str, str, str, str],
]:
    payload_elements = max(prod(payload) for payload in admitted)
    packing = (
        2 * prod(padded) * payload_elements * np.dtype(precision.transform_dtype).itemsize
        if payload_elements > 1
        else 0
    )
    available = resource.maximum_bytes - resource.peak_live_bytes - packing
    scratch_limit = min(
        maximum_external_scratch_bytes, available // topology.device_count
    )
    if scratch_limit <= 0:
        raise SpectralResourceError(
            _with_cudecomp_storage(
                resource,
                packing,
                0,
                reasons=(
                    "no byte budget remains for required positive cuDecomp scratch",
                ),
            )
        )
    execution = prepare_cudecomp_execution(
        binding,
        shape,
        padded,
        topology.mesh,
        np.dtype(precision.transform_dtype),
        scratch_limit,
    )
    external_scratch = (
        topology.device_count * execution.evidence.external_device_scratch_bytes_per_rank
    )
    resource = _with_cudecomp_storage(resource, packing, external_scratch)
    if not resource.accepted:
        raise SpectralResourceError(resource)
    forward, inverse, padded_forward, padded_inverse = (
        canonical_fingerprint(
            {
                "kind": "provider-bound-spectral-sequence",
                "native_sequence": sequence_id,
                "transpose_execution": execution.evidence.execution_id,
            }
        )
        for sequence_id in sequence_ids
    )
    return execution, resource, (forward, inverse, padded_forward, padded_inverse)


def _plan_identities(
    owner: str,
    topology: SpectralMeshTopology,
    shape: tuple[int, ...],
    padded: tuple[int, ...],
    lengths: tuple[float, ...],
    schedule: SpectralSchedule,
    horizontal: tuple[int, int] | None,
    precision: SpectralPrecisionPolicy,
    scale: float,
    padded_scale: float,
    layouts: _LayoutPreparation,
    admitted: tuple[tuple[int, ...], ...],
    trailing: tuple[int, ...],
    resource: SpectralResourceReport,
    sequence_ids: tuple[str, str, str, str],
    transpose_provider: SpectralTransposeProvider,
    maximum_external_scratch_bytes: int,
    /,
) -> tuple[str, str, str, str]:
    transform_families = (
        ("fourier", "chebyshev", "fourier")
        if schedule == "channel"
        else ("fourier",) * len(shape)
    )
    numerical_id = canonical_fingerprint(
        {
            "kind": "distributed-spectral-numerical-plan",
            "spatial_shape": list(shape),
            "padded_shape": list(padded),
            "transform_families": list(transform_families),
            "horizontal_axes": None if horizontal is None else list(horizontal),
            "domain_lengths": list(lengths),
            "normalization": "ortho",
            "transform_scale": scale,
            "padded_transform_scale": padded_scale,
            "representation": "full-complex-c2c",
            "coefficient_storage": precision.coefficient_dtype,
            "transform_compute": precision.transform_dtype,
            "reduction": precision.reduction_dtype,
            "precision": precision.policy_id,
        }
    )
    execution_id = canonical_fingerprint(
        {
            "kind": "distributed-spectral-execution",
            "numerical": numerical_id,
            "topology": topology.topology_id,
            "schedule": schedule,
            "transpose_provider": transpose_provider,
            "maximum_external_scratch_bytes": maximum_external_scratch_bytes,
            "layouts": [
                layouts.physical.layout_id,
                layouts.modal.layout_id,
                layouts.padded_physical.layout_id,
                layouts.padded_modal.layout_id,
            ],
            "stage_sequences": list(sequence_ids),
            "state_shape": list(trailing),
            "admitted_payload_shapes": [list(payload) for payload in admitted],
            "resource": resource.report_id,
        }
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "owner-bound-distributed-spectral-plan",
            "owner": owner,
            "numerical": numerical_id,
            "execution": execution_id,
        }
    )
    report_id = canonical_fingerprint(
        {
            "kind": "distributed-spectral-preparation-report",
            "owner": owner,
            "numerical": numerical_id,
            "execution": execution_id,
            "precision": precision.policy_id,
            "resource": resource.report_id,
        }
    )
    return numerical_id, execution_id, plan_id, report_id


@final
class DistributedSpectralPreparationReport(StrictModule, NonTrainableState):
    owner_id: str = eqx.field(static=True)
    numerical_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    schedule: SpectralSchedule = eqx.field(static=True)
    transpose_provider: SpectralTransposeProvider = eqx.field(static=True)
    transpose_evidence: CudecompTransposeEvidence | None
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    admitted_payload_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    physical_layout_id: str = eqx.field(static=True)
    modal_layout_id: str = eqx.field(static=True)
    padded_physical_layout_id: str = eqx.field(static=True)
    padded_modal_layout_id: str = eqx.field(static=True)
    local_physical_shape: tuple[int, ...] = eqx.field(static=True)
    local_modal_shape: tuple[int, ...] = eqx.field(static=True)
    collective_count: int = eqx.field(static=True)
    collective_payload_bytes: int = eqx.field(static=True)
    forward_sequence_id: str = eqx.field(static=True)
    inverse_sequence_id: str = eqx.field(static=True)
    padded_forward_sequence_id: str = eqx.field(static=True)
    padded_inverse_sequence_id: str = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    host_gather: bool = eqx.field(static=True)
    zero_mode_atomic: bool = eqx.field(static=True)
    resource: SpectralResourceReport
    report_id: str = eqx.field(static=True)


@final
class SpectralExecutionResult(StrictModule):
    value: Array
    layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


@final
class SpectralGlobalDiagnostics(StrictModule):
    total: Array
    maximum_absolute: Array
    l2_norm: Array
    finite: Array
    accumulation_dtype: str = eqx.field(static=True)
    reduction_axes: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


@final
class DistributedSpectralExecutionPlan(StrictModule, NonTrainableState):
    """Prepared full-complex slab/pencil FFT or action-only channel execution."""

    topology: SpectralMeshTopology
    owner_id: str = eqx.field(static=True)
    precision: SpectralPrecisionPolicy
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    admitted_payload_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    schedule: SpectralSchedule = eqx.field(static=True)
    transpose_provider: SpectralTransposeProvider = eqx.field(static=True)
    maximum_external_scratch_bytes: int = eqx.field(static=True)
    _cudecomp_execution: PreparedCudecompExecution | None
    physical_layout: SpectralLayout
    modal_layout: SpectralLayout
    padded_physical_layout: SpectralLayout
    padded_modal_layout: SpectralLayout
    _forward_sequence: tuple[_SpectralStage, ...] = eqx.field(static=True)
    _inverse_sequence: tuple[_SpectralStage, ...] = eqx.field(static=True)
    _padded_forward_sequence: tuple[_SpectralStage, ...] = eqx.field(static=True)
    _padded_inverse_sequence: tuple[_SpectralStage, ...] = eqx.field(static=True)
    domain_lengths: tuple[float, ...] = eqx.field(static=True)
    transform_scale: float = eqx.field(static=True)
    padded_transform_scale: float = eqx.field(static=True)
    horizontal_axes: tuple[int, int] | None = eqx.field(static=True)
    report: DistributedSpectralPreparationReport
    numerical_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        topology: SpectralMeshTopology,
        spatial_shape: Sequence[int],
        /,
        *,
        owner_id: str,
        admitted_payload_shapes: Sequence[Sequence[int]],
        precision: SpectralPrecisionPolicy | None = None,
        schedule: SpectralSchedule = "slab",
        padded_shape: Sequence[int] | None = None,
        state_shape: Sequence[int] = (),
        domain_lengths: Sequence[float] | None = None,
        transform_scale: float = 1.0,
        padded_transform_scale: float | None = None,
        maximum_bytes: int = 2 * 1024**3,
        horizontal_axes: Sequence[int] = (0, 2),
        transpose_provider: SpectralTransposeProvider = "native",
        collective_binding: CollectiveBinding | None = None,
        maximum_external_scratch_bytes: int = 256 * 1024**2,
    ) -> None:
        owner = canonical_identifier(owner_id, "owner_id")
        precision_ = (
            SpectralPrecisionPolicy(jnp.float32) if precision is None else precision
        )
        if not isinstance(precision_, SpectralPrecisionPolicy):
            raise TypeError("precision must be a SpectralPrecisionPolicy or None.")
        _require_precision_available(precision_)
        shape = _positive_shape(spatial_shape, "spatial_shape")
        padded = (
            shape
            if padded_shape is None
            else _positive_shape(padded_shape, "padded_shape")
        )
        trailing = tuple(index(value) for value in state_shape)
        if any(value <= 0 for value in trailing):
            raise ValueError("state_shape dimensions must be positive.")
        admitted = _payload_shapes(admitted_payload_shapes, trailing)
        if len(padded) != len(shape) or any(
            padded_size < size for size, padded_size in zip(shape, padded, strict=True)
        ):
            raise ValueError("padded_shape must componentwise contain spatial_shape.")
        schedule_ = parse(schedule, SpectralSchedule, "schedule")
        lengths = (
            (2.0 * np.pi,) * len(shape)
            if domain_lengths is None
            else tuple(float(value) for value in domain_lengths)
        )
        if len(lengths) != len(shape) or any(
            not np.isfinite(value) or value <= 0.0 for value in lengths
        ):
            raise ValueError(
                "domain_lengths must contain one finite positive value per axis."
            )
        scale = float(transform_scale)
        padded_scale = (
            scale if padded_transform_scale is None else float(padded_transform_scale)
        )
        if (
            not np.isfinite(scale)
            or scale <= 0.0
            or not np.isfinite(padded_scale)
            or padded_scale <= 0.0
        ):
            raise ValueError("Transform scales must be finite and positive.")
        maximum = index(maximum_bytes)
        if maximum <= 0:
            raise ValueError("maximum_bytes must be positive.")
        provider = parse(
            transpose_provider, SpectralTransposeProvider, "transpose_provider"
        )
        external_maximum = index(maximum_external_scratch_bytes)
        if external_maximum <= 0:
            raise ValueError("maximum_external_scratch_bytes must be positive.")
        cudecomp_binding = None
        if provider == "native" and collective_binding is not None:
            raise ValueError(
                "Native spectral transposes do not consume a collective binding."
            )
        if provider == "cudecomp":
            if schedule_ != "slab" or len(shape) != 3 or len(topology.mesh_shape) != 1:
                raise ValueError(
                    "cuDecomp supports canonical direct three-dimensional slabs only; no pencil fallback."
                )
            if not isinstance(collective_binding, CollectiveBinding):
                raise TypeError("cuDecomp requires an explicit CollectiveBinding.")
            cudecomp_binding = collective_binding
            if any(
                device_platform(device) != "cuda" for device in topology.mesh.devices.flat
            ):
                raise ValueError(
                    "cuDecomp requires CUDA devices on every participating rank."
                )
        layouts = _prepare_layouts(
            topology,
            shape,
            padded,
            trailing,
            schedule_,
            horizontal_axes,
        )
        physical = layouts.physical
        modal = layouts.modal
        padded_physical = layouts.padded_physical
        padded_modal = layouts.padded_modal
        forward = layouts.forward
        inverse = layouts.inverse
        padded_forward = layouts.padded_forward
        padded_inverse = layouts.padded_inverse
        horizontal = layouts.horizontal_axes
        sequence_ids = _sequence_ids(layouts, precision_, scale, padded_scale)
        resource, collective_count, collective_payload = _prepare_fft_resource(
            shape,
            padded,
            admitted,
            precision_,
            padded_forward,
            maximum,
            layouts.reasons,
        )
        if not resource.accepted:
            raise SpectralResourceError(resource)
        cudecomp_execution = None
        if cudecomp_binding is not None:
            cudecomp_execution, resource, sequence_ids = _prepare_cudecomp_transpose(
                cudecomp_binding,
                topology,
                shape,
                padded,
                precision_,
                admitted,
                resource,
                external_maximum,
                sequence_ids,
            )
        forward_id, inverse_id, padded_forward_id, padded_inverse_id = sequence_ids
        sequence_ids = (forward_id, inverse_id, padded_forward_id, padded_inverse_id)
        numerical_id, execution_id, plan_id, report_id = _plan_identities(
            owner,
            topology,
            shape,
            padded,
            lengths,
            schedule_,
            horizontal,
            precision_,
            scale,
            padded_scale,
            layouts,
            admitted,
            trailing,
            resource,
            sequence_ids,
            provider,
            external_maximum,
        )
        self.topology = topology
        self.owner_id = owner
        self.precision = precision_
        self.spatial_shape = shape
        self.padded_shape = padded
        self.state_shape = trailing
        self.admitted_payload_shapes = admitted
        self.schedule = schedule_
        self.transpose_provider = provider
        self.maximum_external_scratch_bytes = external_maximum
        self._cudecomp_execution = cudecomp_execution
        self.physical_layout = physical
        self.modal_layout = modal
        self.padded_physical_layout = padded_physical
        self.padded_modal_layout = padded_modal
        self._forward_sequence = forward
        self._inverse_sequence = inverse
        self._padded_forward_sequence = padded_forward
        self._padded_inverse_sequence = padded_inverse
        self.domain_lengths = lengths
        self.transform_scale = scale
        self.padded_transform_scale = padded_scale
        self.horizontal_axes = horizontal
        self.numerical_id = numerical_id
        self.execution_id = execution_id
        self.plan_id = plan_id
        self.report = DistributedSpectralPreparationReport(
            owner_id=owner,
            numerical_id=numerical_id,
            execution_id=execution_id,
            precision_policy_id=precision_.policy_id,
            schedule=schedule_,
            transpose_provider=provider,
            transpose_evidence=None
            if cudecomp_execution is None
            else cudecomp_execution.evidence,
            spatial_shape=shape,
            padded_shape=padded,
            state_shape=trailing,
            admitted_payload_shapes=admitted,
            topology_id=topology.topology_id,
            physical_layout_id=physical.layout_id,
            modal_layout_id=modal.layout_id,
            padded_physical_layout_id=padded_physical.layout_id,
            padded_modal_layout_id=padded_modal.layout_id,
            local_physical_shape=physical.local_shape(topology),
            local_modal_shape=modal.local_shape(topology),
            collective_count=collective_count,
            collective_payload_bytes=collective_payload,
            forward_sequence_id=forward_id,
            inverse_sequence_id=inverse_id,
            padded_forward_sequence_id=padded_forward_id,
            padded_inverse_sequence_id=padded_inverse_id,
            differentiable=True,
            host_gather=False,
            zero_mode_atomic=True,
            resource=resource,
            report_id=report_id,
        )

    @classmethod
    def from_discretization(
        cls,
        topology: SpectralMeshTopology,
        discretization: Any,
        /,
        *,
        admitted_payload_shapes: Sequence[Sequence[int]],
        **kwargs: Unpack[_FromDiscretizationOptions],
    ) -> "DistributedSpectralExecutionPlan":
        axes = tuple(discretization.axes)
        families = tuple(axis.family for axis in axes)
        schedule = kwargs.pop(
            "schedule",
            "channel" if families == ("fourier", "chebyshev", "fourier") else "slab",
        )
        if schedule == "channel":
            if families != ("fourier", "chebyshev", "fourier"):
                raise ValueError(
                    "Channel execution requires exactly Fourier-Chebyshev-Fourier axes."
                )
        elif any(family != "fourier" for family in families):
            raise ValueError("Distributed C2C plans require all-Fourier discretizations.")
        scale = prod(float(jnp.sqrt(axis.quadrature_weights[0])) for axis in axes)
        padded_shape = kwargs.get("padded_shape")
        if padded_shape is None:
            padded_scale = scale
        else:
            padded = _positive_shape(padded_shape, "padded_shape")
            if len(padded) != len(axes):
                raise ValueError("padded_shape must match the discretization rank.")
            padded_scale = prod(
                float(jnp.sqrt(axis.length / count))
                for axis, count in zip(axes, padded, strict=True)
            )
        forwarded = cast(_ForwardedPlanOptions, kwargs)
        return cls(
            topology,
            discretization.modal_shape,
            owner_id=discretization.prepared_id,
            precision=discretization.plan.precision,
            admitted_payload_shapes=admitted_payload_shapes,
            schedule=schedule,
            domain_lengths=tuple(float(axis.length) for axis in axes),
            transform_scale=scale,
            padded_transform_scale=padded_scale,
            **forwarded,
        )

    @property
    def collective_binding(self) -> CollectiveBinding | None:
        return (
            None if self._cudecomp_execution is None else self._cudecomp_execution.binding
        )

    def prepare(self, /) -> "DistributedSpectralExecutionPlan":
        self.topology.require_available()
        binding = self.collective_binding
        if binding is None:
            return self
        if (
            inside_jax_transformation()
            or isinstance(binding.lease_id, Tracer)
            or isinstance(binding.invocation_ticket, Tracer)
        ):
            raise ValueError(
                "cuDecomp recipe re-admission is an explicit host phase, not a JIT operation."
            )
        # Restoration keeps a portable recipe, never an old native generation.
        # Re-admission also rebuilds consumer-visible IDs if the site build changed.
        lease = require_collective_lease(binding.spec)
        return DistributedSpectralExecutionPlan(
            self.topology,
            self.spatial_shape,
            owner_id=self.owner_id,
            admitted_payload_shapes=self.admitted_payload_shapes,
            precision=self.precision,
            schedule=self.schedule,
            padded_shape=self.padded_shape,
            state_shape=self.state_shape,
            domain_lengths=self.domain_lengths,
            transform_scale=self.transform_scale,
            padded_transform_scale=self.padded_transform_scale,
            maximum_bytes=self.report.resource.maximum_bytes,
            horizontal_axes=(0, 2)
            if self.horizontal_axes is None
            else self.horizontal_axes,
            transpose_provider=self.transpose_provider,
            collective_binding=lease.binding,
            maximum_external_scratch_bytes=self.maximum_external_scratch_bytes,
        )

    def _layout(
        self, representation: SpectralRepresentation, padded: bool, /
    ) -> SpectralLayout:
        if representation == "physical":
            return self.padded_physical_layout if padded else self.physical_layout
        return self.padded_modal_layout if padded else self.modal_layout

    def _batched_layout(
        self,
        global_shape: Sequence[int],
        representation: SpectralRepresentation,
        padded: bool,
        /,
    ) -> SpectralLayout:
        shape = tuple(global_shape)
        spatial = self.padded_shape if padded else self.spatial_shape
        rank = len(spatial)
        if len(shape) < rank or shape[:rank] != spatial:
            raise ValueError(
                f"Batched spectral values must begin with shape {spatial}; got {shape}."
            )
        payload = shape[rank:]
        if payload not in self.admitted_payload_shapes:
            raise ValueError(
                f"Spectral payload shape {payload} is not admitted; exact admitted "
                f"shapes are {self.admitted_payload_shapes}."
            )
        base = self._layout(representation, padded)
        partition = base.partition[:rank] + (None,) * len(payload)
        return SpectralLayout(
            shape,
            partition,
            representation,
            self.topology,
            padded=padded,
        )

    def _validate_batched(
        self,
        values: ArrayLike,
        representation: SpectralRepresentation,
        padded: bool,
        owner: str,
        /,
    ) -> tuple[Array, SpectralLayout]:
        input_dtype = jax.dtypes.result_type(values)
        if not (
            jnp.issubdtype(input_dtype, jnp.number)
            or jnp.issubdtype(input_dtype, jnp.bool_)
        ):
            raise TypeError(f"{owner} must be a numeric ArrayLike value.")
        static_shape = np.shape(values)
        layout = self._batched_layout(static_shape, representation, padded)
        value = jnp.asarray(values, dtype=jnp.dtype(self.precision.coefficient_dtype))
        self.topology.require_available()
        return jax.device_put(value, layout.sharding(self.topology)), layout

    def _validate(
        self, values: ArrayLike, layout: SpectralLayout, owner: str, /
    ) -> Array:
        value = jnp.asarray(values, dtype=jnp.dtype(self.precision.coefficient_dtype))
        if value.shape != layout.global_shape:
            raise ValueError(
                f"{owner} must have global shape {layout.global_shape}; got {value.shape}."
            )
        self.topology.require_available()
        return jax.device_put(value, layout.sharding(self.topology))

    def place(
        self,
        values: ArrayLike,
        /,
        *,
        representation: SpectralRepresentation,
        padded: bool = False,
    ) -> Array:
        return self._validate(
            values, self._layout(representation, padded), "Spectral value"
        )

    def place_batched(
        self,
        values: ArrayLike,
        /,
        *,
        representation: SpectralRepresentation,
        padded: bool = False,
    ) -> Array:
        """Place one exact admitted payload shape without changing spatial sharding."""
        placed, _ = self._validate_batched(
            values, representation, padded, "Batched spectral value"
        )
        return placed

    def _execute_local_sequence(
        self,
        local: Array,
        sequence: tuple[_SpectralStage, ...],
        scale: float,
        /,
        *,
        inverse: bool,
        padded: bool = False,
    ) -> Array:
        if self.schedule == "channel":
            raise ValueError(
                "Channel layouts use execute_channel; the Chebyshev axis is not "
                "a C2C transform."
            )
        transform_dtype = jnp.dtype(self.precision.transform_dtype)
        coefficient_dtype = jnp.dtype(self.precision.coefficient_dtype)
        value = local.astype(transform_dtype)
        scale_value = jnp.asarray(scale, dtype=value.real.dtype)
        if not inverse:
            value = value * scale_value
        for stage in sequence:
            if stage.operation == "transpose" and self._cudecomp_execution is not None:
                value = self._cudecomp_execution.apply_local(
                    value, inverse=inverse, padded=padded
                )
            else:
                value = stage.apply_local(value, inverse=inverse)
        if inverse:
            value = value / scale_value
        return value.astype(coefficient_dtype)

    def forward_local(self, local: Array, /, *, padded: bool = False) -> Array:
        """Physical→modal transform of this device's slab/pencil block.

        Must run inside a mapped region over ``topology.mesh`` whose axes carry
        the physical (``padded``) layout partition; the transposes are native
        all-to-all collectives whose JAX transpose is the exact adjoint.
        """
        if padded:
            return self._execute_local_sequence(
                local,
                self._padded_forward_sequence,
                self.padded_transform_scale,
                inverse=False,
                padded=True,
            )
        return self._execute_local_sequence(
            local,
            self._forward_sequence,
            self.transform_scale,
            inverse=False,
        )

    def inverse_local(self, local: Array, /, *, padded: bool = False) -> Array:
        """Modal→physical transform of this device's block (inverse of ``forward_local``)."""
        if padded:
            return self._execute_local_sequence(
                local,
                self._padded_inverse_sequence,
                self.padded_transform_scale,
                padded=True,
                inverse=True,
            )
        return self._execute_local_sequence(
            local,
            self._inverse_sequence,
            self.transform_scale,
            inverse=True,
        )

    def to_modal(self, values: ArrayLike, /, *, padded: bool = False) -> Array:
        source = self._layout("physical", padded)
        target = self._layout("modal", padded)
        placed = self._validate(values, source, "Physical spectral state")
        action = partial(self.forward_local, padded=padded)
        mapped = jax.shard_map(
            action,
            mesh=self.topology.mesh,
            in_specs=source.partition_spec,
            out_specs=target.partition_spec,
            check_vma=self.transpose_provider == "cudecomp",
        )
        return mapped(placed)

    def to_physical(self, coefficients: ArrayLike, /, *, padded: bool = False) -> Array:
        source = self._layout("modal", padded)
        target = self._layout("physical", padded)
        placed = self._validate(coefficients, source, "Modal spectral state")
        action = partial(self.inverse_local, padded=padded)
        mapped = jax.shard_map(
            action,
            mesh=self.topology.mesh,
            in_specs=source.partition_spec,
            out_specs=target.partition_spec,
            check_vma=self.transpose_provider == "cudecomp",
        )
        return mapped(placed)

    def to_modal_batched(self, values: ArrayLike, /, *, padded: bool = False) -> Array:
        """Transform one exact admitted trailing payload in a distributed C2C batch."""
        placed, source = self._validate_batched(
            values, "physical", padded, "Batched physical spectral state"
        )
        target = self._batched_layout(placed.shape, "modal", padded)
        action = partial(self.forward_local, padded=padded)
        mapped = jax.shard_map(
            action,
            mesh=self.topology.mesh,
            in_specs=source.partition_spec,
            out_specs=target.partition_spec,
            check_vma=self.transpose_provider == "cudecomp",
        )
        return mapped(placed)

    def to_physical_batched(
        self, coefficients: ArrayLike, /, *, padded: bool = False
    ) -> Array:
        """Invert one exact admitted trailing payload without a global materialization."""
        placed, source = self._validate_batched(
            coefficients, "modal", padded, "Batched modal spectral state"
        )
        target = self._batched_layout(placed.shape, "physical", padded)
        action = partial(self.inverse_local, padded=padded)
        mapped = jax.shard_map(
            action,
            mesh=self.topology.mesh,
            in_specs=source.partition_spec,
            out_specs=target.partition_spec,
            check_vma=self.transpose_provider == "cudecomp",
        )
        return mapped(placed)

    def execute_transform(
        self,
        values: ArrayLike,
        /,
        *,
        direction: Literal["physical_to_modal", "modal_to_physical"],
        padded: bool = False,
    ) -> SpectralExecutionResult:
        if direction == "physical_to_modal":
            result = self.to_modal(values, padded=padded)
            layout = self._layout("modal", padded)
        elif direction == "modal_to_physical":
            result = self.to_physical(values, padded=padded)
            layout = self._layout("physical", padded)
        else:
            raise ValueError("Unknown distributed spectral transform direction.")
        return SpectralExecutionResult(result, layout.layout_id, self.plan_id)

    def pad_modal(self, coefficients: ArrayLike, /) -> Array:
        value = self._validate(coefficients, self.modal_layout, "Canonical modal state")
        result = value
        for axis, target in enumerate(self.padded_shape):
            result = resize_fourier_axis(result, axis, target)
        return jax.device_put(result, self.padded_modal_layout.sharding(self.topology))

    def unpad_modal(self, coefficients: ArrayLike, /) -> Array:
        value = self._validate(
            coefficients, self.padded_modal_layout, "Padded modal state"
        )
        result = value
        for axis, target in enumerate(self.spatial_shape):
            result = resize_fourier_axis(result, axis, target)
        return jax.device_put(result, self.modal_layout.sharding(self.topology))

    def pad_modal_batched(self, coefficients: ArrayLike, /) -> Array:
        """Embed a retained modal field with one exact admitted payload shape."""
        value, _ = self._validate_batched(
            coefficients, "modal", False, "Batched canonical modal state"
        )
        result = value
        for axis, target in enumerate(self.padded_shape):
            result = resize_fourier_axis(result, axis, target)
        target_layout = self._batched_layout(result.shape, "modal", True)
        return jax.device_put(result, target_layout.sharding(self.topology))

    def unpad_modal_batched(self, coefficients: ArrayLike, /) -> Array:
        """Restrict padded modal fields while retaining distributed sharding."""
        value, _ = self._validate_batched(
            coefficients, "modal", True, "Batched padded modal state"
        )
        result = value
        for axis, target in enumerate(self.spatial_shape):
            result = resize_fourier_axis(result, axis, target)
        target_layout = self._batched_layout(result.shape, "modal", False)
        return jax.device_put(result, target_layout.sharding(self.topology))

    def modal_derivative(
        self,
        coefficients: ArrayLike,
        axis: int,
        /,
        *,
        order: int = 1,
        padded: bool = False,
    ) -> Array:
        layout = self._layout("modal", padded)
        value = self._validate(coefficients, layout, "Modal derivative state")
        axis_ = int(axis)
        order_ = int(order)
        shape = self.padded_shape if padded else self.spatial_shape
        if axis_ < 0 or axis_ >= len(shape) or order_ < 0:
            raise ValueError("Derivative axis/order is invalid.")
        if order_ == 0:
            return value
        real_dtype = value.real.dtype
        modes = jnp.fft.fftfreq(shape[axis_]).astype(real_dtype) * shape[axis_]
        wave = (
            2.0
            * jnp.asarray(jnp.pi, dtype=real_dtype)
            * modes
            / self.domain_lengths[axis_]
        )
        multiplier_shape = [1] * value.ndim
        multiplier_shape[axis_] = shape[axis_]
        return value * ((1j * wave) ** order_).reshape(tuple(multiplier_shape))

    def modal_derivative_batched(
        self,
        coefficients: ArrayLike,
        axis: int,
        /,
        *,
        order: int = 1,
        padded: bool = False,
    ) -> Array:
        """Differentiate a modal field with one exact admitted payload shape."""
        value, _ = self._validate_batched(
            coefficients, "modal", padded, "Batched modal derivative state"
        )
        axis_ = int(axis)
        order_ = int(order)
        shape = self.padded_shape if padded else self.spatial_shape
        if axis_ < 0 or axis_ >= len(shape) or order_ < 0:
            raise ValueError("Derivative axis/order is invalid.")
        if order_ == 0:
            return value
        real_dtype = value.real.dtype
        modes = jnp.fft.fftfreq(shape[axis_]).astype(real_dtype) * shape[axis_]
        wave = (
            2.0
            * jnp.asarray(jnp.pi, dtype=real_dtype)
            * modes
            / self.domain_lengths[axis_]
        )
        multiplier_shape = [1] * value.ndim
        multiplier_shape[axis_] = shape[axis_]
        return value * ((1j * wave) ** order_).reshape(tuple(multiplier_shape))

    def project(self, projector: Any, coefficients: ArrayLike, /) -> Array:
        value = self._validate(coefficients, self.modal_layout, "Projected modal state")
        projected = projector.project(value)
        return self._validate(projected, self.modal_layout, "Projector result")

    def etdrk_step(
        self,
        stepper: Callable[..., ArrayLike],
        coefficients: ArrayLike,
        /,
        *args: object,
        **kwargs: object,
    ) -> Array:
        value = self._validate(coefficients, self.modal_layout, "ETDRK modal state")
        result = stepper(value, *args, **kwargs)
        return self._validate(result, self.modal_layout, "ETDRK result")

    def rotational_nonlinear(
        self, velocity: ArrayLike, /, *, projector: Any | None = None
    ) -> Array:
        if (
            len(self.spatial_shape) != 3
            or self.state_shape != (3,)
            or self.schedule == "channel"
        ):
            raise ValueError(
                "Rotational evaluation requires a three-dimensional periodic vector plan."
            )
        modal = self._validate(velocity, self.modal_layout, "Modal velocity")
        padded_modal = self.pad_modal(modal)
        physical_velocity = self.to_physical(padded_modal, padded=True)
        derivatives = tuple(
            self.to_physical(
                self.modal_derivative(padded_modal, axis, padded=True), padded=True
            )
            for axis in range(3)
        )
        curl = jnp.stack(
            (
                derivatives[1][..., 2] - derivatives[2][..., 1],
                derivatives[2][..., 0] - derivatives[0][..., 2],
                derivatives[0][..., 1] - derivatives[1][..., 0],
            ),
            axis=-1,
        )
        rotational = jnp.cross(physical_velocity, curl, axis=-1)
        result = self.unpad_modal(self.to_modal(rotational, padded=True))
        return result if projector is None else self.project(projector, result)

    def _global_reductions(
        self, value: Array, layout: SpectralLayout, /
    ) -> tuple[Array, Array, Array, Array]:
        reduction_dtype = jnp.dtype(self.precision.reduction_dtype)
        sum_dtype = np.dtype(
            jnp.complex128 if reduction_dtype.itemsize > 4 else jnp.complex64
        )
        axes = layout.used_mesh_axes

        def reduce_local(local: Array) -> tuple[Array, Array, Array, Array]:
            reduced = local.astype(sum_dtype)
            magnitude = jnp.abs(reduced)
            total = jnp.sum(reduced)
            squared = jnp.sum(magnitude * magnitude, dtype=reduction_dtype)
            maximum = jax.lax.stop_gradient(
                jnp.max(magnitude, initial=jnp.asarray(0.0, dtype=reduction_dtype))
            )
            finite = jnp.all(jnp.isfinite(reduced)).astype(jnp.int32)
            if axes:
                total = jax.lax.psum(total, axes)
                squared = jax.lax.psum(squared, axes)
                maximum = jax.lax.pmax(maximum, axes)
                finite = jax.lax.pmin(finite, axes)
            return total, maximum, jnp.sqrt(squared), finite.astype(jnp.bool_)

        mapped = jax.shard_map(
            reduce_local,
            mesh=self.topology.mesh,
            in_specs=layout.partition_spec,
            out_specs=(PartitionSpec(),) * 4,
            check_vma=False,
        )
        return mapped(value)

    def diagnostics(
        self,
        values: ArrayLike,
        /,
        *,
        representation: SpectralRepresentation = "modal",
        padded: bool = False,
    ) -> SpectralGlobalDiagnostics:
        layout = self._layout(representation, padded)
        value = self._validate(values, layout, "Diagnostic state")
        total, maximum, norm, finite = self._global_reductions(value, layout)
        return SpectralGlobalDiagnostics(
            total,
            maximum,
            norm,
            finite,
            self.precision.reduction_dtype,
            layout.used_mesh_axes,
            self.plan_id,
        )

    def diagnostics_batched(
        self,
        values: ArrayLike,
        /,
        *,
        representation: SpectralRepresentation = "modal",
        padded: bool = False,
    ) -> SpectralGlobalDiagnostics:
        """Reduce one exact admitted payload shape over every spatial shard."""
        value, layout = self._validate_batched(
            values, representation, padded, "Batched diagnostic state"
        )
        total, maximum, norm, finite = self._global_reductions(value, layout)
        return SpectralGlobalDiagnostics(
            total,
            maximum,
            norm,
            finite,
            self.precision.reduction_dtype,
            layout.used_mesh_axes,
            self.plan_id,
        )

    def global_inner_product(
        self,
        left: ArrayLike,
        right: ArrayLike,
        /,
        *,
        representation: SpectralRepresentation = "modal",
        padded: bool = False,
    ) -> Array:
        """Return a shard-reduced inner product for one admitted payload shape."""
        left_value, layout = self._validate_batched(
            left, representation, padded, "Left distributed inner-product state"
        )
        right_value, right_layout = self._validate_batched(
            right, representation, padded, "Right distributed inner-product state"
        )
        if right_layout.global_shape != layout.global_shape:
            raise ValueError("Distributed inner-product operands must have equal shape.")
        reduction_dtype = np.dtype(self.precision.reduction_dtype)
        sum_dtype = np.dtype(
            jnp.complex128 if reduction_dtype.itemsize > 4 else jnp.complex64
        )
        axes = layout.used_mesh_axes

        def inner_local(left_local: Array, right_local: Array) -> Array:
            total = jnp.vdot(left_local.astype(sum_dtype), right_local.astype(sum_dtype))
            return jax.lax.psum(total, axes) if axes else total

        mapped = jax.shard_map(
            inner_local,
            mesh=self.topology.mesh,
            in_specs=(layout.partition_spec, right_layout.partition_spec),
            out_specs=PartitionSpec(),
            check_vma=False,
        )
        return mapped(left_value, right_value)

    def global_all(
        self,
        predicates: ArrayLike,
        /,
        *,
        representation: SpectralRepresentation = "modal",
        padded: bool = False,
    ) -> Array:
        """Return a replicated all-shard conjunction without a host transfer."""
        value = jnp.asarray(predicates, dtype=jnp.bool_)
        layout = self._batched_layout(value.shape, representation, padded)
        self.topology.require_available()
        placed = jax.device_put(value, layout.sharding(self.topology))
        axes = layout.used_mesh_axes

        def all_local(local: Array) -> Array:
            result = jnp.all(local).astype(jnp.int32)
            if axes:
                result = jax.lax.pmin(result, axes)
            return result.astype(jnp.bool_)

        mapped = jax.shard_map(
            all_local,
            mesh=self.topology.mesh,
            in_specs=layout.partition_spec,
            out_specs=PartitionSpec(),
            check_vma=False,
        )
        return mapped(placed)

    def execute_channel(
        self,
        action: Callable[..., ArrayLike],
        state: ArrayLike,
        /,
        *args: object,
        **kwargs: object,
    ) -> Array:
        if self.schedule != "channel":
            raise ValueError("execute_channel requires a channel execution plan.")
        value = self._validate(state, self.modal_layout, "Channel modal state")
        result = action(value, *args, **kwargs)
        return self._validate(result, self.modal_layout, "Channel action result")

    def channel_zero_mode(self, state: ArrayLike, /) -> Array:
        if self.schedule != "channel" or self.horizontal_axes is None:
            raise ValueError("channel_zero_mode requires a channel execution plan.")
        value = self._validate(state, self.modal_layout, "Channel modal state")
        first, second = self.horizontal_axes
        selection: list[Any] = [slice(None)] * value.ndim
        selection[first] = 0
        selection[second] = 0
        return value[tuple(selection)]


__all__ = [
    "DistributedSpectralExecutionPlan",
    "DistributedSpectralPreparationReport",
    "SpectralExecutionResult",
    "SpectralGlobalDiagnostics",
    "SpectralLayout",
    "SpectralMeshTopology",
    "SpectralResourceError",
    "SpectralResourceReport",
    "SpectralTranspose",
    "SpectralTransposeProvider",
]
