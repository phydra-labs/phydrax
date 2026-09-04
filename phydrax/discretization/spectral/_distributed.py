#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from operator import index
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._spectral._fourier import resize_fourier_axis
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


SpectralSchedule: TypeAlias = Literal["slab", "pencil", "channel"]
SpectralRepresentation: TypeAlias = Literal["physical", "modal"]


def _positive_shape(shape: Sequence[int], owner: str, /) -> tuple[int, ...]:
    result = tuple(int(value) for value in shape)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{owner} must contain positive dimensions.")
    return result


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


def _complex_accumulation_dtype(dtype: np.dtype, /) -> np.dtype:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return np.dtype(jnp.complex128 if dtype.itemsize > 8 else jnp.complex64)
    return dtype


class SpectralMeshTopology(StrictModule, NonTrainableState):
    """Caller-visible device mesh with a stable, hardware-exact identity."""

    mesh: Mesh = eqx.field(static=True)
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    mesh_axis_names: tuple[str, ...] = eqx.field(static=True)
    device_ids: tuple[int, ...] = eqx.field(static=True)
    platform: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh_or_shape: Mesh | Sequence[int] = (1,),
        /,
        *,
        devices: Sequence[jax.Device] | None = None,
        axis_names: Sequence[str] | None = None,
    ):
        if isinstance(mesh_or_shape, Mesh):
            if devices is not None or axis_names is not None:
                raise ValueError(
                    "A caller-owned Mesh already fixes devices and axis names."
                )
            mesh = mesh_or_shape
            names = tuple(str(name) for name in mesh.axis_names)
            mesh_shape = tuple(int(mesh.shape[name]) for name in names)
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
            if len({device.id for device in selected}) != len(selected):
                raise ValueError("A spectral topology cannot contain a device twice.")
            mesh = Mesh(np.asarray(selected, dtype=object).reshape(mesh_shape), names)
        if not selected:
            raise ValueError("A spectral topology requires at least one device.")
        platforms = {device.platform for device in selected}
        if len(platforms) != 1:
            raise ValueError("A spectral topology must use devices from one platform.")
        ids = tuple(int(device.id) for device in selected)
        platform = selected[0].platform
        self.mesh = mesh
        self.mesh_shape = mesh_shape
        self.mesh_axis_names = names
        self.device_ids = ids
        self.platform = platform
        self.topology_id = canonical_fingerprint(
            {
                "kind": "spectral-mesh-topology",
                "mesh_shape": list(mesh_shape),
                "axis_names": list(names),
                "platform": platform,
                "device_ids": list(ids),
            }
        )

    @classmethod
    def one_device(cls, device: jax.Device | None = None, /) -> "SpectralMeshTopology":
        selected = jax.devices()[0] if device is None else device
        return cls((1,), devices=(selected,), axis_names=("spectral",))

    @property
    def device_count(self) -> int:
        return prod(self.mesh_shape)

    def require_available(self, /) -> None:
        available = {(device.platform, int(device.id)) for device in jax.devices()}
        missing = tuple(
            device_id
            for device_id in self.device_ids
            if (self.platform, device_id) not in available
        )
        if missing:
            raise RuntimeError(
                "Spectral topology devices are unavailable in the current JAX process: "
                f"{missing}."
            )


class SpectralLayout(StrictModule, NonTrainableState):
    """Global array shape and its exact named-mesh partition contract."""

    global_shape: tuple[int, ...] = eqx.field(static=True)
    partition: tuple[str | tuple[str, ...] | None, ...] = eqx.field(static=True)
    representation: SpectralRepresentation = eqx.field(static=True)
    padded: bool = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        global_shape: Sequence[int],
        partition: Sequence[str | tuple[str, ...] | None],
        representation: SpectralRepresentation,
        topology: SpectralMeshTopology,
        /,
        *,
        padded: bool = False,
    ):
        shape = _positive_shape(global_shape, "global_shape")
        entries = tuple(_partition_entry(value) for value in partition)
        if len(entries) != len(shape):
            raise ValueError("partition must explicitly describe every array dimension.")
        if representation not in ("physical", "modal"):
            raise ValueError("representation must be 'physical' or 'modal'.")
        if not isinstance(topology, SpectralMeshTopology):
            raise TypeError("topology must be SpectralMeshTopology.")
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
    ):
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


class SpectralResourceReport(StrictModule, NonTrainableState):
    """Fail-closed memory preflight for one distributed spectral execution plan."""

    canonical_bytes: int = eqx.field(static=True)
    padded_bytes: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    stage_bytes: int = eqx.field(static=True)
    collective_bytes: int = eqx.field(static=True)
    checkpoint_bytes: int = eqx.field(static=True)
    closure_bytes: int = eqx.field(static=True)
    total_bytes: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        canonical_bytes: int,
        padded_bytes: int,
        state_bytes: int,
        stage_bytes: int,
        collective_bytes: int,
        checkpoint_bytes: int,
        closure_bytes: int = 0,
        maximum_bytes: int,
        reasons: Sequence[str] = (),
    ):
        values = tuple(
            index(value)
            for value in (
                canonical_bytes,
                padded_bytes,
                state_bytes,
                stage_bytes,
                collective_bytes,
                checkpoint_bytes,
                closure_bytes,
            )
        )
        maximum = index(maximum_bytes)
        if any(value < 0 for value in values) or maximum <= 0:
            raise ValueError(
                "Spectral resource byte counts must be non-negative and bounded."
            )
        total = sum(values)
        reasons_ = tuple(str(reason) for reason in reasons)
        if any(not reason for reason in reasons_):
            raise ValueError("Resource refusal reasons must be non-empty.")
        if total > maximum:
            reasons_ += (f"required bytes {total} exceed maximum_bytes {maximum}",)
        accepted = not reasons_
        (
            self.canonical_bytes,
            self.padded_bytes,
            self.state_bytes,
            self.stage_bytes,
            self.collective_bytes,
            self.checkpoint_bytes,
            self.closure_bytes,
        ) = values
        self.total_bytes = total
        self.maximum_bytes = maximum
        self.accepted = accepted
        self.reasons = reasons_
        self.report_id = canonical_fingerprint(
            {
                "kind": "distributed-spectral-resource-report",
                "bytes": list(values),
                "total": total,
                "maximum": maximum,
                "accepted": accepted,
                "reasons": list(reasons_),
            }
        )


class SpectralResourceError(MemoryError):
    """Typed refusal retaining the exact preflight report."""

    report: SpectralResourceReport

    def __init__(self, report: SpectralResourceReport, /):
        self.report = report
        super().__init__(
            "Distributed spectral resource preflight refused: "
            + "; ".join(report.reasons)
        )


class DistributedSpectralPreparationReport(StrictModule, NonTrainableState):
    schedule: SpectralSchedule = eqx.field(static=True)
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    local_physical_shape: tuple[int, ...] = eqx.field(static=True)
    local_modal_shape: tuple[int, ...] = eqx.field(static=True)
    collective_count: int = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    host_gather: bool = eqx.field(static=True)
    zero_mode_atomic: bool = eqx.field(static=True)
    resource: SpectralResourceReport
    report_id: str = eqx.field(static=True)


class SpectralExecutionResult(StrictModule):
    value: Array
    layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class SpectralGlobalDiagnostics(StrictModule):
    total: Array
    maximum_absolute: Array
    l2_norm: Array
    finite: Array
    accumulation_dtype: str = eqx.field(static=True)
    reduction_axes: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class DistributedSpectralExecutionPlan(StrictModule, NonTrainableState):
    """Prepared slab/pencil full-complex FFT execution on a real JAX mesh."""

    topology: SpectralMeshTopology
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    schedule: SpectralSchedule = eqx.field(static=True)
    physical_layout: SpectralLayout
    modal_layout: SpectralLayout
    padded_physical_layout: SpectralLayout
    padded_modal_layout: SpectralLayout
    physical_to_modal: tuple[SpectralTranspose, ...]
    modal_to_physical: tuple[SpectralTranspose, ...]
    padded_physical_to_modal: tuple[SpectralTranspose, ...]
    padded_modal_to_physical: tuple[SpectralTranspose, ...]
    domain_lengths: tuple[float, ...] = eqx.field(static=True)
    transform_scale: float = eqx.field(static=True)
    padded_transform_scale: float = eqx.field(static=True)
    coefficient_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    horizontal_axes: tuple[int, int] | None = eqx.field(static=True)
    report: DistributedSpectralPreparationReport
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: SpectralMeshTopology,
        spatial_shape: Sequence[int],
        /,
        *,
        schedule: SpectralSchedule = "slab",
        padded_shape: Sequence[int] | None = None,
        state_shape: Sequence[int] = (),
        domain_lengths: Sequence[float] | None = None,
        coefficient_dtype: Any = jnp.complex64,
        accumulation_dtype: Any | None = None,
        transform_scale: float = 1.0,
        padded_transform_scale: float | None = None,
        stage_count: int = 1,
        checkpoint_count: int = 0,
        closure_workspace_bytes: int = 0,
        maximum_bytes: int = 2 * 1024**3,
        horizontal_axes: Sequence[int] = (0, 2),
    ):
        if not isinstance(topology, SpectralMeshTopology):
            raise TypeError("topology must be SpectralMeshTopology.")
        shape = _positive_shape(spatial_shape, "spatial_shape")
        padded = (
            shape
            if padded_shape is None
            else _positive_shape(padded_shape, "padded_shape")
        )
        trailing = tuple(int(value) for value in state_shape)
        if any(value <= 0 for value in trailing):
            raise ValueError("state_shape dimensions must be positive.")
        if len(padded) != len(shape) or any(
            b < a for a, b in zip(shape, padded, strict=True)
        ):
            raise ValueError("padded_shape must componentwise contain spatial_shape.")
        if schedule not in ("slab", "pencil", "channel"):
            raise ValueError("schedule must be 'slab', 'pencil', or 'channel'.")
        dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(coefficient_dtype)))
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            raise TypeError(
                "Distributed C2C execution requires a complex coefficient dtype."
            )
        coefficient_real_dtype = np.dtype(jnp.empty((), dtype=dtype).real.dtype)
        accumulation = np.dtype(
            jax.dtypes.canonicalize_dtype(
                coefficient_real_dtype
                if accumulation_dtype is None
                else np.dtype(accumulation_dtype)
            )
        )
        if not jnp.issubdtype(accumulation, jnp.floating):
            raise TypeError("accumulation_dtype must be real floating point.")
        if accumulation.itemsize < coefficient_real_dtype.itemsize:
            raise ValueError(
                "accumulation precision cannot be narrower than coefficient precision."
            )
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
        stages = index(stage_count)
        checkpoints = index(checkpoint_count)
        closure_workspace = index(closure_workspace_bytes)
        maximum = index(maximum_bytes)
        if stages <= 0 or checkpoints < 0 or closure_workspace < 0 or maximum <= 0:
            raise ValueError(
                "stage_count, checkpoint_count, closure_workspace_bytes, and "
                "maximum_bytes are invalid."
            )
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
                    "pencil execution requires a two-dimensional mesh and "
                    "spatial rank at least three"
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
            axes = tuple(int(value) for value in horizontal_axes)
            if (
                len(axes) != 2
                or len(set(axes)) != 2
                or any(value < 0 or value >= rank for value in axes)
            ):
                raise ValueError("horizontal_axes must name two distinct spatial axes.")
            first_axis, second_axis = axes
            horizontal = (first_axis, second_axis)
            if rank != 3 or tuple(axis for axis in range(rank) if axis not in axes) != (
                1,
            ):
                reasons.append(
                    "channel distribution requires rank three with replicated "
                    "Chebyshev axis 1"
                )
            if rank == 3 and padded[1] != shape[1]:
                reasons.append(
                    "channel padding cannot resize the replicated Chebyshev axis"
                )
            if len(mesh_shape) not in (1, 2):
                reasons.append(
                    "channel execution requires a one- or two-dimensional mesh"
                )
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
            first_axis, second_axis = (0, 1)
            count = mesh_shape[0] if len(mesh_shape) == 1 else 1
            if shape[first_axis] % count or shape[second_axis] % count:
                reasons.append(
                    "canonical slab dimensions are not divisible by the mesh size"
                )
            if padded[first_axis] % count or padded[second_axis] % count:
                reasons.append(
                    "padded slab dimensions are not divisible by the mesh size"
                )
            physical_entries: list[str | tuple[str, ...] | None] = [None] * len(
                full_shape
            )
            modal_entries: list[str | tuple[str, ...] | None] = [None] * len(full_shape)
            if len(mesh_shape) == 1:
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
            forward = (
                SpectralTranspose(physical, middle, mesh_names[1], 2, 1),
                SpectralTranspose(middle, modal, mesh_names[0], 2, 0),
            )
            reverse = (
                SpectralTranspose(modal, middle, mesh_names[0], 0, 2),
                SpectralTranspose(middle, physical, mesh_names[1], 1, 2),
            )
            padded_forward = (
                SpectralTranspose(padded_physical, middle_padded, mesh_names[1], 2, 1),
                SpectralTranspose(middle_padded, padded_modal, mesh_names[0], 2, 0),
            )
            padded_reverse = (
                SpectralTranspose(padded_modal, middle_padded, mesh_names[0], 0, 2),
                SpectralTranspose(middle_padded, padded_physical, mesh_names[1], 1, 2),
            )
        elif schedule == "slab" and len(mesh_shape) == 1:
            forward = (
                SpectralTranspose(
                    physical,
                    modal,
                    mesh_names[0],
                    second_axis,
                    first_axis,
                ),
            )
            reverse = (
                SpectralTranspose(
                    modal,
                    physical,
                    mesh_names[0],
                    first_axis,
                    second_axis,
                ),
            )
            padded_forward = (
                SpectralTranspose(
                    padded_physical,
                    padded_modal,
                    mesh_names[0],
                    second_axis,
                    first_axis,
                ),
            )
            padded_reverse = (
                SpectralTranspose(
                    padded_modal,
                    padded_physical,
                    mesh_names[0],
                    first_axis,
                    second_axis,
                ),
            )
        else:
            forward = reverse = padded_forward = padded_reverse = ()
        components = prod(trailing) if trailing else 1
        canonical_bytes = prod(shape) * dtype.itemsize
        padded_bytes = prod(padded) * dtype.itemsize
        state_bytes = canonical_bytes * components
        stage_bytes = padded_bytes * components * stages
        collective_count = 2 if schedule == "pencil" else 1 if schedule == "slab" else 0
        collective_bytes = padded_bytes * components * collective_count
        checkpoint_bytes = state_bytes * checkpoints
        resource = SpectralResourceReport(
            canonical_bytes=canonical_bytes,
            padded_bytes=padded_bytes,
            state_bytes=state_bytes,
            stage_bytes=stage_bytes,
            collective_bytes=collective_bytes,
            checkpoint_bytes=checkpoint_bytes,
            closure_bytes=closure_workspace,
            maximum_bytes=maximum,
            reasons=reasons,
        )
        if not resource.accepted:
            raise SpectralResourceError(resource)
        report_id = canonical_fingerprint(
            {
                "kind": "distributed-spectral-preparation-report",
                "schedule": schedule,
                "shape": list(shape),
                "padded_shape": list(padded),
                "state_shape": list(trailing),
                "topology": topology.topology_id,
                "resource": resource.report_id,
            }
        )
        self.topology = topology
        self.spatial_shape = shape
        self.padded_shape = padded
        self.state_shape = trailing
        self.schedule = schedule
        self.physical_layout = physical
        self.modal_layout = modal
        self.padded_physical_layout = padded_physical
        self.padded_modal_layout = padded_modal
        self.physical_to_modal = forward
        self.modal_to_physical = reverse
        self.padded_physical_to_modal = padded_forward
        self.padded_modal_to_physical = padded_reverse
        self.domain_lengths = lengths
        self.transform_scale = scale
        self.padded_transform_scale = padded_scale
        self.coefficient_dtype = dtype.str
        self.accumulation_dtype = accumulation.str
        self.horizontal_axes = horizontal
        self.report = DistributedSpectralPreparationReport(
            schedule=schedule,
            spatial_shape=shape,
            padded_shape=padded,
            state_shape=trailing,
            local_physical_shape=physical.local_shape(topology),
            local_modal_shape=modal.local_shape(topology),
            collective_count=collective_count,
            differentiable=True,
            host_gather=False,
            zero_mode_atomic=True,
            resource=resource,
            report_id=report_id,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-spectral-execution-plan",
                "report": report_id,
                "dtype": dtype.str,
                "accumulation_dtype": accumulation.str,
                "lengths": list(lengths),
                "transform_scale": scale,
                "padded_transform_scale": padded_scale,
            }
        )

    @classmethod
    def from_discretization(
        cls,
        topology: SpectralMeshTopology,
        discretization: Any,
        /,
        **kwargs,
    ) -> "DistributedSpectralExecutionPlan":
        axes = tuple(discretization.axes)
        families = tuple(axis.family for axis in axes)
        schedule = kwargs.pop(
            "schedule",
            "channel" if families == ("fourier", "chebyshev", "fourier") else "slab",
        )
        if schedule != "channel" and any(family != "fourier" for family in families):
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
        return cls(
            topology,
            discretization.modal_shape,
            schedule=schedule,
            domain_lengths=tuple(float(axis.length) for axis in axes),
            coefficient_dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
            accumulation_dtype=jnp.dtype(discretization.plan.precision.reduction_dtype),
            transform_scale=scale,
            padded_transform_scale=padded_scale,
            **kwargs,
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
        shape = tuple(int(value) for value in global_shape)
        spatial = self.padded_shape if padded else self.spatial_shape
        rank = len(spatial)
        if len(shape) < rank or shape[:rank] != spatial:
            raise ValueError(
                f"Batched spectral values must begin with shape {spatial}; got {shape}."
            )
        base = self._layout(representation, padded)
        partition = base.partition[:rank] + (None,) * (len(shape) - rank)
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
        value = jnp.asarray(values, dtype=jnp.dtype(self.coefficient_dtype))
        layout = self._batched_layout(value.shape, representation, padded)
        self.topology.require_available()
        return jax.device_put(value, layout.sharding(self.topology)), layout

    def _validate(
        self, values: ArrayLike, layout: SpectralLayout, owner: str, /
    ) -> Array:
        value = jnp.asarray(values, dtype=jnp.dtype(self.coefficient_dtype))
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
        """Place arbitrary replicated payload axes without changing spatial sharding."""
        placed, _ = self._validate_batched(
            values, representation, padded, "Batched spectral value"
        )
        return placed

    def _forward_local_scaled(self, local: Array, scale: float, /) -> Array:
        rank = len(self.spatial_shape)
        if self.schedule == "pencil":
            value = jnp.fft.fft(local * scale, axis=2, norm="ortho")
            value = jax.lax.all_to_all(
                value, self.topology.mesh_axis_names[1], 2, 1, tiled=True
            )
            value = jnp.fft.fft(value, axis=1, norm="ortho")
            value = jax.lax.all_to_all(
                value, self.topology.mesh_axis_names[0], 2, 0, tiled=True
            )
            value = jnp.fft.fft(value, axis=0, norm="ortho")
            for axis in range(3, rank):
                value = jnp.fft.fft(value, axis=axis, norm="ortho")
            return value
        if self.schedule == "channel":
            raise ValueError(
                "Channel layouts use execute_channel; the Chebyshev axis is not a C2C transform."
            )
        value = local * scale
        for axis in range(1, rank):
            value = jnp.fft.fft(value, axis=axis, norm="ortho")
        value = jax.lax.all_to_all(
            value, self.topology.mesh_axis_names[0], 1, 0, tiled=True
        )
        return jnp.fft.fft(value, axis=0, norm="ortho")

    def _forward_local(self, local: Array, /) -> Array:
        return self._forward_local_scaled(local, self.transform_scale)

    def _forward_padded_local(self, local: Array, /) -> Array:
        return self._forward_local_scaled(local, self.padded_transform_scale)

    def _inverse_local_scaled(self, local: Array, scale: float, /) -> Array:
        rank = len(self.spatial_shape)
        if self.schedule == "pencil":
            value = jnp.fft.ifft(local, axis=0, norm="ortho")
            value = jax.lax.all_to_all(
                value, self.topology.mesh_axis_names[0], 0, 2, tiled=True
            )
            value = jnp.fft.ifft(value, axis=1, norm="ortho")
            value = jax.lax.all_to_all(
                value, self.topology.mesh_axis_names[1], 1, 2, tiled=True
            )
            value = jnp.fft.ifft(value, axis=2, norm="ortho")
            for axis in range(3, rank):
                value = jnp.fft.ifft(value, axis=axis, norm="ortho")
            return value / scale
        if self.schedule == "channel":
            raise ValueError(
                "Channel layouts use execute_channel; the Chebyshev axis is not a C2C transform."
            )
        value = jnp.fft.ifft(local, axis=0, norm="ortho")
        value = jax.lax.all_to_all(
            value, self.topology.mesh_axis_names[0], 0, 1, tiled=True
        )
        for axis in range(1, rank):
            value = jnp.fft.ifft(value, axis=axis, norm="ortho")
        return value / scale

    def _inverse_local(self, local: Array, /) -> Array:
        return self._inverse_local_scaled(local, self.transform_scale)

    def _inverse_padded_local(self, local: Array, /) -> Array:
        return self._inverse_local_scaled(local, self.padded_transform_scale)

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
        """Transform arbitrary trailing payload axes in one distributed C2C batch."""
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
        """Invert arbitrary trailing payload axes without materializing them globally."""
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
        """Embed retained modal fields with arbitrary replicated payload axes."""
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
        """Differentiate modal fields with arbitrary replicated payload axes."""
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
        *args,
        **kwargs,
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
        self, values: ArrayLike, layout: SpectralLayout, /
    ) -> tuple[Array, Array, Array, Array]:
        value = self._validate(values, layout, "Diagnostic state")
        reduction_dtype = jnp.dtype(self.accumulation_dtype)
        sum_dtype = _complex_accumulation_dtype(np.dtype(value.dtype))
        if np.dtype(sum_dtype).itemsize < reduction_dtype.itemsize:
            sum_dtype = np.dtype(
                jnp.complex128 if reduction_dtype.itemsize > 4 else jnp.complex64
            )
        axes = layout.used_mesh_axes

        def reduce_local(local):
            total = jnp.sum(local.astype(sum_dtype))
            squared = jnp.sum(jnp.square(jnp.abs(local)).astype(reduction_dtype))
            maximum = jax.lax.stop_gradient(
                jnp.max(jnp.abs(local).astype(reduction_dtype), initial=0.0)
            )
            finite = jnp.all(jnp.isfinite(local)).astype(jnp.int32)
            if axes:
                total = jax.lax.psum(total, axes)
                squared = jax.lax.psum(squared, axes)
                maximum = jax.lax.pmax(maximum, axes)
                finite = jax.lax.pmin(finite, axes)
            return total, maximum, jnp.sqrt(squared), finite.astype(bool)

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
        total, maximum, norm, finite = self._global_reductions(values, layout)
        return SpectralGlobalDiagnostics(
            total,
            maximum,
            norm,
            finite,
            self.accumulation_dtype,
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
        """Reduce arbitrary replicated payload axes over every spatial shard."""
        value, layout = self._validate_batched(
            values, representation, padded, "Batched diagnostic state"
        )
        total, maximum, norm, finite = self._global_reductions(value, layout)
        return SpectralGlobalDiagnostics(
            total,
            maximum,
            norm,
            finite,
            self.accumulation_dtype,
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
        """Return a shard-reduced complex inner product for arbitrary payload axes."""
        left_value, layout = self._validate_batched(
            left, representation, padded, "Left distributed inner-product state"
        )
        right_value, right_layout = self._validate_batched(
            right, representation, padded, "Right distributed inner-product state"
        )
        if right_layout.global_shape != layout.global_shape:
            raise ValueError("Distributed inner-product operands must have equal shape.")
        sum_dtype = _complex_accumulation_dtype(np.dtype(left_value.dtype))
        reduction_dtype = np.dtype(self.accumulation_dtype)
        if np.dtype(sum_dtype).itemsize < reduction_dtype.itemsize:
            sum_dtype = np.dtype(
                jnp.complex128 if reduction_dtype.itemsize > 4 else jnp.complex64
            )
        axes = layout.used_mesh_axes

        def inner_local(left_local, right_local):
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
        value = jnp.asarray(predicates, dtype=bool)
        layout = self._batched_layout(value.shape, representation, padded)
        self.topology.require_available()
        placed = jax.device_put(value, layout.sharding(self.topology))
        axes = layout.used_mesh_axes

        def all_local(local):
            result = jnp.all(local).astype(jnp.int32)
            if axes:
                result = jax.lax.pmin(result, axes)
            return result.astype(bool)

        mapped = jax.shard_map(
            all_local,
            mesh=self.topology.mesh,
            in_specs=layout.partition_spec,
            out_specs=PartitionSpec(),
            check_vma=False,
        )
        return mapped(placed)

    def execute_channel(
        self, action: Callable[..., ArrayLike], state: ArrayLike, /, *args, **kwargs
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
