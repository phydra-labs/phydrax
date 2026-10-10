#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from operator import index
from typing import Any, cast, final, Literal, TypeAlias, TypedDict, Unpack

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import canonical_fingerprint
from ..._spectral._fourier import resize_fourier_axis
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...typing import checked, parse
from ._precision import SpectralPrecisionPolicy


SpectralSchedule: TypeAlias = Literal["slab", "pencil", "channel"]
SpectralRepresentation: TypeAlias = Literal["physical", "modal"]
_SpectralStageOperation: TypeAlias = Literal["fft", "transpose"]


class _ForwardedPlanOptions(TypedDict, total=False):
    padded_shape: Sequence[int] | None
    state_shape: Sequence[int]
    maximum_bytes: int
    horizontal_axes: Sequence[int]


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
    """Caller-visible device mesh with a stable, hardware-exact identity."""

    mesh: Mesh = eqx.field(static=True)
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    mesh_axis_names: tuple[str, ...] = eqx.field(static=True)
    device_ids: tuple[int, ...] = eqx.field(static=True)
    platform: str = eqx.field(static=True)
    device_keys: tuple[tuple[int, int], ...] = eqx.field(static=True)
    execution_group_id: str | None = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh_or_shape: Mesh | Sequence[int] = (1,),
        /,
        *,
        devices: Sequence[jax.Device] | None = None,
        axis_names: Sequence[str] | None = None,
        execution_group_id: str | None = None,
    ) -> None:
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
            available = tuple(jax.devices())
            requested = prod(mesh_shape)
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
            mesh = Mesh(np.asarray(selected, dtype=object).reshape(mesh_shape), names)
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
            }
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
        available = {
            (device.platform, int(device.process_index), int(device.id))
            for device in jax.devices()
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
    """One differentiable all-to-all redistribution between spectral layouts."""

    source: SpectralLayout
    target: SpectralLayout
    mesh_axis: str = eqx.field(static=True)
    split_axis: int = eqx.field(static=True)
    concat_axis: int = eqx.field(static=True)
    transpose_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: SpectralLayout,
        target: SpectralLayout,
        mesh_axis: str,
        split_axis: int,
        concat_axis: int,
        /,
    ) -> None:
        if not isinstance(source, SpectralLayout) or not isinstance(
            target, SpectralLayout
        ):
            raise TypeError("source and target must be SpectralLayout values.")
        if source.topology_id != target.topology_id:
            raise ValueError("Spectral transpose layouts must share one topology.")
        name = str(mesh_axis)
        split = int(split_axis)
        concat = int(concat_axis)
        if not name or name not in source.used_mesh_axes + target.used_mesh_axes:
            raise ValueError("mesh_axis must participate in the transpose layouts.")
        if (
            split < 0
            or concat < 0
            or split >= len(source.global_shape)
            or concat >= len(source.global_shape)
        ):
            raise ValueError("Transpose axes lie outside the distributed array rank.")
        if split == concat:
            raise ValueError("all_to_all split and concatenation axes must differ.")
        self.source = source
        self.target = target
        self.mesh_axis = name
        self.split_axis = split
        self.concat_axis = concat
        self.transpose_id = canonical_fingerprint(
            {
                "kind": "spectral-all-to-all-transpose",
                "source": source.layout_id,
                "target": target.layout_id,
                "mesh_axis": name,
                "split_axis": split,
                "concat_axis": concat,
            }
        )

    def apply_local(self, local_value: Array, /) -> Array:
        return jax.lax.all_to_all(
            local_value,
            axis_name=self.mesh_axis,
            split_axis=self.split_axis,
            concat_axis=self.concat_axis,
            tiled=True,
        )

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
    mesh_axis: str,
    spatial_rank: int,
    /,
) -> tuple[tuple[_SpectralStage, ...], tuple[_SpectralStage, ...]]:
    forward_transpose = SpectralTranspose(physical, modal, mesh_axis, 1, 0)
    inverse_transpose = SpectralTranspose(modal, physical, mesh_axis, 0, 1)
    forward = (
        _SpectralStage("fft", axes=range(1, spatial_rank)),
        _SpectralStage("transpose", transpose=forward_transpose),
        _SpectralStage("fft", axes=(0,)),
    )
    inverse = _reverse_sequence(forward, ((forward_transpose, inverse_transpose),))
    return forward, inverse


def _pencil_sequence_pair(
    physical: SpectralLayout,
    middle: SpectralLayout,
    modal: SpectralLayout,
    mesh_axes: tuple[str, str],
    spatial_rank: int,
    /,
) -> tuple[tuple[_SpectralStage, ...], tuple[_SpectralStage, ...]]:
    first, second = mesh_axes
    first_forward = SpectralTranspose(physical, middle, second, 2, 1)
    second_forward = SpectralTranspose(middle, modal, first, 2, 0)
    first_inverse = SpectralTranspose(middle, physical, second, 1, 2)
    second_inverse = SpectralTranspose(modal, middle, first, 0, 2)
    suffix = (
        () if spatial_rank == 3 else (_SpectralStage("fft", axes=range(3, spatial_rank)),)
    )
    forward = (
        _SpectralStage("fft", axes=(2,)),
        _SpectralStage("transpose", transpose=first_forward),
        _SpectralStage("fft", axes=(1,)),
        _SpectralStage("transpose", transpose=second_forward),
        _SpectralStage("fft", axes=(0,)),
        *suffix,
    )
    inverse = _reverse_sequence(
        forward,
        (
            (first_forward, first_inverse),
            (second_forward, second_inverse),
        ),
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
        middle_partition = (mesh_names[0], None, mesh_names[1]) + (None,) * (
            rank - 3 + len(trailing)
        )
        middle = SpectralLayout(full_shape, middle_partition, "modal", topology)
        middle_padded = SpectralLayout(
            full_padded, middle_partition, "modal", topology, padded=True
        )
        forward, inverse = _pencil_sequence_pair(
            physical,
            middle,
            modal,
            (mesh_names[0], mesh_names[1]),
            rank,
        )
        padded_forward, padded_inverse = _pencil_sequence_pair(
            padded_physical,
            middle_padded,
            padded_modal,
            (mesh_names[0], mesh_names[1]),
            rank,
        )
    elif schedule == "slab" and len(mesh_shape) == 1 and rank >= 2:
        forward, inverse = _slab_sequence_pair(physical, modal, mesh_names[0], rank)
        padded_forward, padded_inverse = _slab_sequence_pair(
            padded_physical, padded_modal, mesh_names[0], rank
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
    """Fail-closed FFT storage, workspace, liveness, and traffic evidence."""

    canonical_storage_bytes: int = eqx.field(static=True)
    padded_storage_bytes: int = eqx.field(static=True)
    transform_workspace_bytes: int = eqx.field(static=True)
    collective_payload_bytes: int = eqx.field(static=True)
    peak_live_bytes: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
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
        if (
            any(value < 0 for value in (canonical, padded, workspace, collective, peak))
            or maximum <= 0
        ):
            raise ValueError(
                "Spectral resource byte counts must be non-negative and bounded."
            )
        if peak != 2 * padded + workspace:
            raise ValueError(
                "peak_live_bytes must count one source, one target, and one "
                "transform workspace buffer."
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
    transform_workspace = (
        prod(padded) * payload_elements * transform_itemsize if has_fft_stage else 0
    )
    collective_count = sum(stage.operation == "transpose" for stage in forward)
    collective_payload = transform_workspace * collective_count
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
        (
            forward_id,
            inverse_id,
            padded_forward_id,
            padded_inverse_id,
        ) = sequence_ids
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
        )
        self.topology = topology
        self.owner_id = owner
        self.precision = precision_
        self.spatial_shape = shape
        self.padded_shape = padded
        self.state_shape = trailing
        self.admitted_payload_shapes = admitted
        self.schedule = schedule_
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

    def prepare(self, /) -> "DistributedSpectralExecutionPlan":
        self.topology.require_available()
        return self

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
            value = stage.apply_local(value, inverse=inverse)
        if inverse:
            value = value / scale_value
        return value.astype(coefficient_dtype)

    def _forward_local(self, local: Array, /) -> Array:
        return self._execute_local_sequence(
            local,
            self._forward_sequence,
            self.transform_scale,
            inverse=False,
        )

    def _forward_padded_local(self, local: Array, /) -> Array:
        return self._execute_local_sequence(
            local,
            self._padded_forward_sequence,
            self.padded_transform_scale,
            inverse=False,
        )

    def _inverse_local(self, local: Array, /) -> Array:
        return self._execute_local_sequence(
            local,
            self._inverse_sequence,
            self.transform_scale,
            inverse=True,
        )

    def _inverse_padded_local(self, local: Array, /) -> Array:
        return self._execute_local_sequence(
            local,
            self._padded_inverse_sequence,
            self.padded_transform_scale,
            inverse=True,
        )

    def to_modal(self, values: ArrayLike, /, *, padded: bool = False) -> Array:
        source = self._layout("physical", padded)
        target = self._layout("modal", padded)
        placed = self._validate(values, source, "Physical spectral state")
        action = self._forward_padded_local if padded else self._forward_local
        mapped = jax.shard_map(
            action,
            mesh=self.topology.mesh,
            in_specs=source.partition_spec,
            out_specs=target.partition_spec,
            check_vma=False,
        )
        return mapped(placed)

    def to_physical(self, coefficients: ArrayLike, /, *, padded: bool = False) -> Array:
        source = self._layout("modal", padded)
        target = self._layout("physical", padded)
        placed = self._validate(coefficients, source, "Modal spectral state")
        action = self._inverse_padded_local if padded else self._inverse_local
        mapped = jax.shard_map(
            action,
            mesh=self.topology.mesh,
            in_specs=source.partition_spec,
            out_specs=target.partition_spec,
            check_vma=False,
        )
        return mapped(placed)

    def to_modal_batched(self, values: ArrayLike, /, *, padded: bool = False) -> Array:
        """Transform one exact admitted trailing payload in a distributed C2C batch."""
        placed, source = self._validate_batched(
            values, "physical", padded, "Batched physical spectral state"
        )
        target = self._batched_layout(placed.shape, "modal", padded)
        action = self._forward_padded_local if padded else self._forward_local
        mapped = jax.shard_map(
            action,
            mesh=self.topology.mesh,
            in_specs=source.partition_spec,
            out_specs=target.partition_spec,
            check_vma=False,
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
        action = self._inverse_padded_local if padded else self._inverse_local
        mapped = jax.shard_map(
            action,
            mesh=self.topology.mesh,
            in_specs=source.partition_spec,
            out_specs=target.partition_spec,
            check_vma=False,
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
]
