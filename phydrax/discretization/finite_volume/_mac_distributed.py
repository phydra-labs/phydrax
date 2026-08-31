#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import FaceVelocity, PreparedMACOperators
from ._mac_momentum import PreparedMACMomentumOperators


MACDistributedPlanState: TypeAlias = Literal[
    "ready", "unsupported-topology", "unavailable-process-topology"
]
MACHaloRole: TypeAlias = Literal["pressure", "face", "momentum"]


def _partition_entries(
    specification: PartitionSpec,
    dimension: int,
    /,
) -> tuple[str | None, ...]:
    entries = tuple(specification)
    if len(entries) != dimension:
        raise ValueError("Every MAC PartitionSpec must explicitly name every dimension.")
    normalized: list[str | None] = []
    for entry in entries:
        if entry is None:
            normalized.append(None)
        elif isinstance(entry, str) and entry:
            normalized.append(entry)
        elif (
            isinstance(entry, tuple)
            and len(entry) == 1
            and isinstance(entry[0], str)
            and entry[0]
        ):
            normalized.append(entry[0])
        else:
            raise ValueError(
                "MAC spatial dimensions may use at most one named mesh axis."
            )
    return tuple(normalized)


def _mesh_identity(mesh: Mesh, /) -> tuple[tuple[str, ...], tuple[object, ...]]:
    names = tuple(str(name) for name in mesh.axis_names)
    devices = tuple(np.asarray(mesh.devices, dtype=object).reshape(-1).tolist())
    return names, devices


def _same_mesh(left: Mesh, right: Mesh, /) -> bool:
    left_names, left_devices = _mesh_identity(left)
    right_names, right_devices = _mesh_identity(right)
    return left_names == right_names and left_devices == right_devices


def _device_identity(device: jax.Device, /) -> dict[str, object]:
    return {
        "platform": str(device.platform),
        "process_index": int(device.process_index),
        "device_id": int(device.id),
        "device_kind": str(device.device_kind),
    }


def _permutation(
    count: int,
    direction: int,
    periodic: bool,
    /,
) -> tuple[tuple[int, int], ...]:
    pairs = []
    for source in range(count):
        target = source + direction
        if periodic:
            target %= count
        if 0 <= target < count:
            pairs.append((source, target))
    return tuple(pairs)


def _axis_slice(value: Array, axis: int, start: int, stop: int, /) -> Array:
    return jax.lax.slice_in_dim(value, start, stop, axis=axis)


def _all_finite(values: Sequence[Array], /) -> Array:
    result = jnp.asarray(True)
    for value in values:
        result = result & jnp.all(jnp.isfinite(value))
    return result


class MACDistributedPlanStatus(StrictModule, NonTrainableState):
    """Fail-closed hardware and topology readiness evidence."""

    state: MACDistributedPlanState = eqx.field(static=True)
    ready: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    device_count: int = eqx.field(static=True)
    addressable_device_count: int = eqx.field(static=True)
    process_index: int = eqx.field(static=True)
    process_count: int = eqx.field(static=True)
    mesh_process_indices: tuple[int, ...] = eqx.field(static=True)
    multi_host: bool = eqx.field(static=True)
    fully_addressable: bool = eqx.field(static=True)
    status_id: str = eqx.field(static=True)


class MACHaloMetadata(StrictModule, NonTrainableState):
    """One role-specific nearest-neighbor exchange route."""

    role: MACHaloRole = eqx.field(static=True)
    component_axis: int | None = eqx.field(static=True)
    spatial_axis: int = eqx.field(static=True)
    side: Literal["lower", "upper"] = eqx.field(static=True)
    mesh_axis: str = eqx.field(static=True)
    width: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    permutation: tuple[tuple[int, int], ...] = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)


class MACInterfaceFaceOwnership(StrictModule, NonTrainableState):
    """Authoritative upper-cell ownership for one partition-interface family."""

    component_axis: int = eqx.field(static=True)
    partition_axis: int = eqx.field(static=True)
    mesh_axis: str = eqx.field(static=True)
    cell_extent: int = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    local_extent: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    interface_indices: tuple[int, ...] = eqx.field(static=True)
    owner_partitions: tuple[int, ...] = eqx.field(static=True)
    transverse_face_count: int = eqx.field(static=True)
    interface_face_count: int = eqx.field(static=True)
    policy: str = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)

    def owner_partition(self, face_index: int, /) -> int:
        index = int(face_index)
        upper_cell = (
            index % self.cell_extent
            if self.periodic
            else min(max(index, 0), self.cell_extent - 1)
        )
        return min(upper_cell // self.local_extent, self.partition_count - 1)


class MACLocalStencilPlan(StrictModule, NonTrainableState):
    """Pure shard-local D/G kernels with explicit ppermute halo exchanges."""

    dimension: int = eqx.field(static=True)
    pressure_shape: tuple[int, ...] = eqx.field(static=True)
    local_pressure_shape: tuple[int, ...] = eqx.field(static=True)
    split_factors: tuple[int, ...] = eqx.field(static=True)
    axis_mesh_names: tuple[str | None, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    lower_permutations: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    upper_permutations: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def _lower_halo(self, value: Array, axis: int, /) -> Array:
        name = self.axis_mesh_names[axis]
        width = self.halo_width
        if name is None or self.split_factors[axis] == 1:
            return _axis_slice(value, axis, value.shape[axis] - width, value.shape[axis])
        upper_edge = _axis_slice(
            value, axis, value.shape[axis] - width, value.shape[axis]
        )
        return jax.lax.ppermute(
            upper_edge,
            axis_name=name,
            perm=self.lower_permutations[axis],
        )

    def _upper_halo(self, value: Array, axis: int, /) -> Array:
        name = self.axis_mesh_names[axis]
        width = self.halo_width
        if name is None or self.split_factors[axis] == 1:
            return _axis_slice(value, axis, 0, width)
        lower_edge = _axis_slice(value, axis, 0, width)
        return jax.lax.ppermute(
            lower_edge,
            axis_name=name,
            perm=self.upper_permutations[axis],
        )

    def materialize(self, value: Array, spatial_axis: int, /) -> Array:
        axis = int(spatial_axis)
        if axis < 0 or axis >= self.dimension:
            raise ValueError("MAC halo spatial_axis is out of range.")
        lower = self._lower_halo(value, axis)
        upper = self._upper_halo(value, axis)
        return jnp.concatenate((lower, value, upper), axis=axis)

    def gradient(
        self,
        pressure: Array,
        face_distances: FaceVelocity,
        /,
    ) -> FaceVelocity:
        output = []
        for axis in range(self.dimension):
            distance = face_distances[axis]
            if self.split_factors[axis] > 1:
                lower = self._lower_halo(pressure, axis)
                extended = jnp.concatenate((lower, pressure), axis=axis)
                derivative = _axis_slice(
                    extended, axis, 1, extended.shape[axis]
                ) - _axis_slice(extended, axis, 0, extended.shape[axis] - 1)
            elif self.periodic_axes[axis]:
                derivative = pressure - jnp.roll(pressure, 1, axis=axis)
            elif pressure.shape[axis] == 1:
                shape = list(pressure.shape)
                shape[axis] = 2
                derivative = jnp.zeros(tuple(shape), dtype=pressure.dtype)
            else:
                moved = jnp.moveaxis(pressure, axis, 0)
                interior = moved[1:] - moved[:-1]
                derivative = jnp.moveaxis(
                    jnp.concatenate(
                        (jnp.zeros_like(moved[:1]), interior, jnp.zeros_like(moved[:1])),
                        axis=0,
                    ),
                    0,
                    axis,
                )
            output.append(derivative / distance)
        return tuple(output)

    def divergence(
        self,
        velocity: FaceVelocity,
        face_measures: FaceVelocity,
        cell_volumes: Array,
        /,
    ) -> Array:
        result = jnp.zeros_like(cell_volumes)
        for axis in range(self.dimension):
            integrated = velocity[axis] * face_measures[axis]
            if self.split_factors[axis] > 1:
                upper = self._upper_halo(integrated, axis)
                extended = jnp.concatenate((integrated, upper), axis=axis)
                difference = _axis_slice(
                    extended, axis, 1, extended.shape[axis]
                ) - _axis_slice(extended, axis, 0, extended.shape[axis] - 1)
            elif self.periodic_axes[axis]:
                difference = jnp.roll(integrated, -1, axis=axis) - integrated
            else:
                difference = _axis_slice(
                    integrated, axis, 1, integrated.shape[axis]
                ) - _axis_slice(integrated, axis, 0, integrated.shape[axis] - 1)
            result = result + difference / cell_volumes
        return result

    def interpolate_inverse_momentum(
        self,
        inverse_momentum: Array,
        /,
    ) -> FaceVelocity:
        output = []
        for axis in range(self.dimension):
            if self.split_factors[axis] > 1:
                lower = self._lower_halo(inverse_momentum, axis)
                previous = jnp.concatenate((lower, inverse_momentum), axis=axis)
                face = 0.5 * (
                    _axis_slice(previous, axis, 0, previous.shape[axis] - 1)
                    + inverse_momentum
                )
            elif self.periodic_axes[axis]:
                face = 0.5 * (inverse_momentum + jnp.roll(inverse_momentum, 1, axis=axis))
            else:
                moved = jnp.moveaxis(inverse_momentum, axis, 0)
                interior = 0.5 * (moved[1:] + moved[:-1])
                face = jnp.moveaxis(
                    jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0),
                    0,
                    axis,
                )
            output.append(face)
        return tuple(output)


class MACDistributedState(StrictModule):
    """Pressure and authoritative, exactly-once face arrays on one MAC topology."""

    pressure: Array
    velocity: FaceVelocity
    all_finite: Array
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        pressure: Array,
        velocity: FaceVelocity,
        topology_id: str,
        layout_id: str,
        /,
    ):
        values = tuple(velocity)
        self.pressure = pressure
        self.velocity = values
        self.all_finite = _all_finite((pressure,) + values)
        self.topology_id = str(topology_id)
        self.layout_id = str(layout_id)


class MACDistributedDiagnostics(StrictModule):
    """Global finite, ownership, and weighted D/G adjoint evidence."""

    weighted_adjoint_residual: Array
    pressure_pairing: Array
    velocity_pairing: Array
    all_finite: Array
    passed: Array
    interface_face_count: int = eqx.field(static=True)
    authoritative_interface_face_count: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    diagnostics_id: str = eqx.field(static=True)


class MACDistributedTopologyPlan(StrictModule, NonTrainableState):
    """Explicit caller-mesh topology for compatible staggered MAC coordinates."""

    operators: PreparedMACOperators
    mesh: Mesh
    pressure_sharding: NamedSharding
    face_shardings: tuple[NamedSharding, ...]
    pressure_spec: PartitionSpec = eqx.field(static=True)
    face_specs: tuple[PartitionSpec, ...] = eqx.field(static=True)
    split_factors: tuple[int, ...] = eqx.field(static=True)
    axis_mesh_names: tuple[str | None, ...] = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    status: MACDistributedPlanStatus
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        mesh: Mesh,
        pressure_sharding: NamedSharding,
        face_shardings: Sequence[NamedSharding],
        /,
        *,
        halo_width: int = 1,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a caller-owned jax.sharding.Mesh.")
        if not isinstance(pressure_sharding, NamedSharding):
            raise TypeError("pressure_sharding must be NamedSharding.")
        faces = tuple(face_shardings)
        dimension = len(operators.discretization.cell_shape)
        if len(faces) != dimension or any(
            not isinstance(value, NamedSharding) for value in faces
        ):
            raise TypeError("face_shardings must contain one NamedSharding per axis.")
        if not _same_mesh(mesh, pressure_sharding.mesh) or any(
            not _same_mesh(mesh, value.mesh) for value in faces
        ):
            raise ValueError("Every MAC NamedSharding must use the exact caller mesh.")
        width = int(halo_width)
        if width != 1:
            raise ValueError("Compatible MAC D/G execution requires one-cell halos.")
        pressure_spec = pressure_sharding.spec
        face_specs = tuple(value.spec for value in faces)
        pressure_entries = _partition_entries(pressure_spec, dimension)
        face_entries = tuple(
            _partition_entries(specification, dimension) for specification in face_specs
        )
        mesh_names = tuple(str(name) for name in mesh.axis_names)
        if any(not name for name in mesh_names) or len(set(mesh_names)) != len(
            mesh_names
        ):
            raise ValueError("MAC mesh axis names must be unique nonempty strings.")
        mesh_shape = {str(name): int(mesh.shape[name]) for name in mesh.axis_names}
        if any(
            name is not None and name not in mesh_shape for name in pressure_entries
        ) or any(
            name is not None and name not in mesh_shape
            for entries in face_entries
            for name in entries
        ):
            raise ValueError("MAC PartitionSpecs refer to an axis outside the mesh.")
        split_factors = tuple(
            1 if name is None else mesh_shape[name] for name in pressure_entries
        )
        pressure_shape = operators.discretization.cell_shape
        face_shapes = tuple(
            layout.shape for layout in operators.discretization.face_layouts
        )
        reasons = []
        if any(entries != pressure_entries for entries in face_entries):
            reasons.append(
                "pressure and face PartitionSpecs must align for local MAC stencils"
            )
        used = tuple(name for name in pressure_entries if name is not None)
        nontrivial = tuple(name for name in mesh_names if mesh_shape[name] > 1)
        if len(set(used)) != len(used) or set(used) != set(nontrivial):
            reasons.append(
                "every nontrivial mesh axis must partition exactly one spatial dimension"
            )
        if any(
            size % factor
            for size, factor in zip(pressure_shape, split_factors, strict=True)
        ):
            reasons.append("pressure extents must divide exactly across the mesh")
        for shape, entries in zip(face_shapes, face_entries, strict=True):
            factors = tuple(1 if name is None else mesh_shape[name] for name in entries)
            if any(size % factor for size, factor in zip(shape, factors, strict=True)):
                reasons.append(
                    "face extents must divide exactly; nonperiodic normal faces cannot be split"
                )
                break
        devices = tuple(np.asarray(mesh.devices, dtype=object).reshape(-1).tolist())
        process_indices = tuple(sorted({int(device.process_index) for device in devices}))
        runtime_process_count = int(jax.process_count())
        runtime_process_index = int(jax.process_index())
        required_process_indices = tuple(range(runtime_process_count))
        local_devices = tuple(
            device
            for device in devices
            if int(device.process_index) == runtime_process_index
        )
        process_ready = process_indices == required_process_indices and bool(
            local_devices
        )
        if not process_ready:
            state: MACDistributedPlanState = "unavailable-process-topology"
            reason = "caller mesh must cover every initialized JAX process and the local process"
        elif reasons:
            state = "unsupported-topology"
            reason = "; ".join(reasons)
        else:
            state = "ready"
            reason = "ready"
        ready = state == "ready"
        fully_addressable = len(local_devices) == len(devices)
        topology_id = canonical_fingerprint(
            {
                "kind": "mac-distributed-topology",
                "operators": operators.prepared_id,
                "mesh_axes": list(mesh_names),
                "mesh_shape": [mesh_shape[name] for name in mesh_names],
                "devices": [_device_identity(device) for device in devices],
                "pressure_spec": [
                    None if name is None else name for name in pressure_entries
                ],
                "face_specs": [
                    [None if name is None else name for name in entries]
                    for entries in face_entries
                ],
                "process_indices": list(process_indices),
                "runtime_process_count": runtime_process_count,
                "halo_width": width,
            }
        )
        layout_id = canonical_fingerprint(
            {
                "kind": "mac-distributed-layout",
                "topology": topology_id,
                "pressure_shape": list(pressure_shape),
                "face_shapes": [list(shape) for shape in face_shapes],
                "ownership": "upper-cell-authoritative",
            }
        )
        status_id = canonical_fingerprint(
            {
                "kind": "mac-distributed-plan-status",
                "topology": topology_id,
                "state": state,
                "reason": reason,
                "process_index": runtime_process_index,
                "addressable_device_count": len(local_devices),
            }
        )
        self.operators = operators
        self.mesh = mesh
        self.pressure_sharding = pressure_sharding
        self.face_shardings = faces
        self.pressure_spec = pressure_spec
        self.face_specs = face_specs
        self.split_factors = split_factors
        self.axis_mesh_names = pressure_entries
        self.halo_width = width
        self.status = MACDistributedPlanStatus(
            state=state,
            ready=ready,
            reason=reason,
            device_count=len(devices),
            addressable_device_count=len(local_devices),
            process_index=runtime_process_index,
            process_count=runtime_process_count,
            mesh_process_indices=process_indices,
            multi_host=runtime_process_count > 1,
            fully_addressable=fully_addressable,
            status_id=status_id,
        )
        self.topology_id = topology_id
        self.layout_id = layout_id

    @classmethod
    def single_device(
        cls,
        operators: PreparedMACOperators,
        /,
        *,
        device: jax.Device | None = None,
        axis_name: str = "mac_device",
    ) -> MACDistributedTopologyPlan:
        selected = jax.devices()[0] if device is None else device
        name = str(axis_name)
        if not name:
            raise ValueError("axis_name must be nonempty.")
        mesh = Mesh(np.asarray((selected,), dtype=object), (name,))
        specification = PartitionSpec(
            *((None,) * len(operators.discretization.cell_shape))
        )
        pressure = NamedSharding(mesh, specification)
        faces = tuple(
            NamedSharding(mesh, specification)
            for _ in operators.discretization.face_layouts
        )
        return cls(operators, mesh, pressure, faces)

    def prepare(
        self,
        momentum: PreparedMACMomentumOperators | None = None,
        /,
    ) -> PreparedMACDistributedTopology:
        return PreparedMACDistributedTopology(self, momentum)


class PreparedMACDistributedTopology(StrictModule, NonTrainableState):
    """Prepared distribution, validation, halo, and local execution facade."""

    plan: MACDistributedTopologyPlan
    operators: PreparedMACOperators
    momentum_operators: PreparedMACMomentumOperators | None
    local_stencils: MACLocalStencilPlan
    halo_metadata: tuple[MACHaloMetadata, ...]
    interface_ownership: tuple[MACInterfaceFaceOwnership, ...]
    cell_volumes: Array
    face_measures: FaceVelocity
    face_distances: FaceVelocity
    face_dual_measures: FaceVelocity
    status: MACDistributedPlanStatus
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MACDistributedTopologyPlan,
        momentum: PreparedMACMomentumOperators | None = None,
        /,
    ):
        if not isinstance(plan, MACDistributedTopologyPlan):
            raise TypeError("plan must be MACDistributedTopologyPlan.")
        if momentum is not None and not isinstance(
            momentum, PreparedMACMomentumOperators
        ):
            raise TypeError("momentum must be PreparedMACMomentumOperators or None.")
        if (
            momentum is not None
            and momentum.operators.prepared_id != plan.operators.prepared_id
        ):
            raise ValueError("Distributed momentum must use the topology operators.")
        grid = plan.operators.discretization.grid
        periodic = tuple(bool(axis.periodic) for axis in grid.structured_axes)
        local_shape = tuple(
            size // factor
            for size, factor in zip(
                plan.operators.discretization.cell_shape,
                plan.split_factors,
                strict=True,
            )
        )
        lower_permutations = tuple(
            _permutation(factor, 1, periodic[axis])
            for axis, factor in enumerate(plan.split_factors)
        )
        upper_permutations = tuple(
            _permutation(factor, -1, periodic[axis])
            for axis, factor in enumerate(plan.split_factors)
        )
        local_stencils = MACLocalStencilPlan(
            dimension=len(local_shape),
            pressure_shape=plan.operators.discretization.cell_shape,
            local_pressure_shape=local_shape,
            split_factors=plan.split_factors,
            axis_mesh_names=plan.axis_mesh_names,
            periodic_axes=periodic,
            lower_permutations=lower_permutations,
            upper_permutations=upper_permutations,
            halo_width=plan.halo_width,
            plan_id=canonical_fingerprint(
                {
                    "kind": "mac-local-stencil-plan",
                    "topology": plan.topology_id,
                    "local_shape": list(local_shape),
                    "periodic": list(periodic),
                }
            ),
        )
        halo_metadata = []
        for spatial_axis, (factor, mesh_axis) in enumerate(
            zip(plan.split_factors, plan.axis_mesh_names, strict=True)
        ):
            if factor == 1 or mesh_axis is None:
                continue
            for role, component_axis in (
                (("pressure", None),)
                + tuple(("face", axis) for axis in range(len(local_shape)))
                + tuple(("momentum", axis) for axis in range(len(local_shape)))
            ):
                for side, permutation in (
                    ("lower", lower_permutations[spatial_axis]),
                    ("upper", upper_permutations[spatial_axis]),
                ):
                    metadata_id = canonical_fingerprint(
                        {
                            "kind": "mac-halo-metadata",
                            "topology": plan.topology_id,
                            "role": role,
                            "component_axis": component_axis,
                            "spatial_axis": spatial_axis,
                            "side": side,
                            "permutation": [list(pair) for pair in permutation],
                        }
                    )
                    halo_metadata.append(
                        MACHaloMetadata(
                            role=role,
                            component_axis=component_axis,
                            spatial_axis=spatial_axis,
                            side=side,
                            mesh_axis=mesh_axis,
                            width=plan.halo_width,
                            periodic=periodic[spatial_axis],
                            permutation=permutation,
                            metadata_id=metadata_id,
                        )
                    )
        ownership = []
        cell_shape = plan.operators.discretization.cell_shape
        for partition_axis, (factor, mesh_axis) in enumerate(
            zip(plan.split_factors, plan.axis_mesh_names, strict=True)
        ):
            if factor == 1 or mesh_axis is None:
                continue
            local_extent = cell_shape[partition_axis] // factor
            interfaces = tuple(
                range(
                    0 if periodic[partition_axis] else local_extent,
                    cell_shape[partition_axis],
                    local_extent,
                )
            )
            owners = tuple(min(index // local_extent, factor - 1) for index in interfaces)
            transverse = prod(
                cell_shape[axis]
                for axis in range(len(cell_shape))
                if axis != partition_axis
            )
            ownership_id = canonical_fingerprint(
                {
                    "kind": "mac-interface-face-ownership",
                    "topology": plan.topology_id,
                    "component_axis": partition_axis,
                    "partition_axis": partition_axis,
                    "interfaces": list(interfaces),
                    "owners": list(owners),
                    "policy": "upper-cell-authoritative",
                }
            )
            ownership.append(
                MACInterfaceFaceOwnership(
                    component_axis=partition_axis,
                    partition_axis=partition_axis,
                    mesh_axis=mesh_axis,
                    cell_extent=cell_shape[partition_axis],
                    partition_count=factor,
                    local_extent=local_extent,
                    periodic=periodic[partition_axis],
                    interface_indices=interfaces,
                    owner_partitions=owners,
                    transverse_face_count=transverse,
                    interface_face_count=len(interfaces) * transverse,
                    policy="upper-cell-authoritative",
                    ownership_id=ownership_id,
                )
            )
        distances = tuple(
            dual / measure
            for dual, measure in zip(
                plan.operators.face_dual_measures,
                plan.operators.discretization.face_measures,
                strict=True,
            )
        )
        if plan.status.ready:
            volumes = jax.device_put(
                plan.operators.discretization.cell_volumes,
                plan.pressure_sharding,
            )
            measures = tuple(
                jax.device_put(value, sharding)
                for value, sharding in zip(
                    plan.operators.discretization.face_measures,
                    plan.face_shardings,
                    strict=True,
                )
            )
            distances_ = tuple(
                jax.device_put(value, sharding)
                for value, sharding in zip(distances, plan.face_shardings, strict=True)
            )
            dual_measures = tuple(
                jax.device_put(value, sharding)
                for value, sharding in zip(
                    plan.operators.face_dual_measures,
                    plan.face_shardings,
                    strict=True,
                )
            )
        else:
            volumes = plan.operators.discretization.cell_volumes
            measures = plan.operators.discretization.face_measures
            distances_ = distances
            dual_measures = plan.operators.face_dual_measures
        self.plan = plan
        self.operators = plan.operators
        self.momentum_operators = momentum
        self.local_stencils = local_stencils
        self.halo_metadata = tuple(halo_metadata)
        self.interface_ownership = tuple(ownership)
        self.cell_volumes = volumes
        self.face_measures = tuple(measures)
        self.face_distances = tuple(distances_)
        self.face_dual_measures = tuple(dual_measures)
        self.status = plan.status
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-distributed-topology",
                "topology": plan.topology_id,
                "local_stencils": local_stencils.plan_id,
                "halos": [value.metadata_id for value in halo_metadata],
                "ownership": [value.ownership_id for value in ownership],
                "momentum": None if momentum is None else momentum.prepared_id,
                "status": plan.status.status_id,
            }
        )

    def _require_ready(self, /) -> None:
        if not self.status.ready:
            raise RuntimeError(
                f"MAC distributed topology is not executable: {self.status.reason}."
            )

    def _validate_pressure_sharding(self, pressure: Array, /) -> None:
        if not isinstance(pressure, jax.Array):
            raise TypeError("Distributed MAC pressure must be a jax.Array.")
        if pressure.shape != self.operators.discretization.cell_shape:
            raise ValueError("Distributed MAC pressure has the wrong global shape.")
        if pressure.dtype != self.operators.pressure_space.dtype:
            raise TypeError("Distributed MAC pressure has the wrong dtype.")
        if not pressure.sharding.is_equivalent_to(
            self.plan.pressure_sharding, pressure.ndim
        ):
            raise ValueError("Distributed MAC pressure has the wrong NamedSharding.")

    def _validate_velocity_sharding(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = tuple(velocity)
        if len(values) != len(self.plan.face_shardings):
            raise ValueError("Distributed MAC velocity has the wrong component count.")
        for value, layout, sharding in zip(
            values,
            self.operators.discretization.face_layouts,
            self.plan.face_shardings,
            strict=True,
        ):
            if not isinstance(value, jax.Array):
                raise TypeError("Distributed MAC face values must be jax.Array objects.")
            if value.shape != layout.shape:
                raise ValueError("Distributed MAC face value has the wrong global shape.")
            if value.dtype != self.operators.pressure_space.dtype:
                raise TypeError("Distributed MAC face value has the wrong dtype.")
            if not value.sharding.is_equivalent_to(sharding, value.ndim):
                raise ValueError(
                    "Distributed MAC face value has the wrong NamedSharding."
                )
        return values

    def distribute(
        self,
        pressure: ArrayLike,
        velocity: FaceVelocity,
        /,
    ) -> MACDistributedState:
        self._require_ready()
        pressure_ = self.operators.validate_pressure(pressure)
        velocity_ = self.operators.validate_velocity(velocity)
        sharded_pressure = jax.device_put(pressure_, self.plan.pressure_sharding)
        sharded_velocity = tuple(
            jax.device_put(value, sharding)
            for value, sharding in zip(velocity_, self.plan.face_shardings, strict=True)
        )
        return MACDistributedState(
            sharded_pressure,
            sharded_velocity,
            self.plan.topology_id,
            self.plan.layout_id,
        )

    def distribute_pressure(self, pressure: ArrayLike, /) -> Array:
        self._require_ready()
        value = self.operators.validate_pressure(pressure)
        return jax.device_put(value, self.plan.pressure_sharding)

    def distribute_velocity(self, velocity: FaceVelocity, /) -> FaceVelocity:
        self._require_ready()
        values = self.operators.validate_velocity(velocity)
        return tuple(
            jax.device_put(value, sharding)
            for value, sharding in zip(values, self.plan.face_shardings, strict=True)
        )

    def validate(self, state: MACDistributedState, /) -> MACDistributedState:
        self._require_ready()
        if not isinstance(state, MACDistributedState):
            raise TypeError("state must be MACDistributedState.")
        if (
            state.topology_id != self.plan.topology_id
            or state.layout_id != self.plan.layout_id
        ):
            raise ValueError("Distributed MAC state belongs to a different topology.")
        self._validate_pressure_sharding(state.pressure)
        self._validate_velocity_sharding(state.velocity)
        return state

    def gather(
        self,
        state: MACDistributedState,
        /,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        value = self.validate(state)
        if not self.status.fully_addressable:
            raise RuntimeError(
                "A global gather is unavailable for a multi-host, partially addressable mesh."
            )
        pressure = np.asarray(jax.device_get(value.pressure))
        velocity = tuple(
            np.asarray(jax.device_get(component)) for component in value.velocity
        )
        return pressure, velocity

    def materialize_pressure_halo(
        self,
        pressure: Array,
        spatial_axis: int,
        /,
    ) -> Array:
        self._require_ready()
        self._validate_pressure_sharding(pressure)
        axis = int(spatial_axis)

        def local(value):
            return self.local_stencils.materialize(value, axis)

        return jax.shard_map(
            local,
            mesh=self.plan.mesh,
            in_specs=self.plan.pressure_spec,
            out_specs=self.plan.pressure_spec,
        )(pressure)

    def materialize_face_halo(
        self,
        component_axis: int,
        spatial_axis: int,
        velocity: FaceVelocity,
        /,
    ) -> Array:
        self._require_ready()
        values = self._validate_velocity_sharding(velocity)
        component = int(component_axis)
        axis = int(spatial_axis)
        if component < 0 or component >= len(values):
            raise ValueError("component_axis is out of range.")

        def local(value):
            return self.local_stencils.materialize(value, axis)

        return jax.shard_map(
            local,
            mesh=self.plan.mesh,
            in_specs=self.plan.face_specs[component],
            out_specs=self.plan.face_specs[component],
        )(values[component])

    def gradient(self, pressure: Array, /) -> FaceVelocity:
        self._require_ready()
        self._validate_pressure_sharding(pressure)

        def local(value, distances):
            return self.local_stencils.gradient(value, distances)

        return jax.shard_map(
            local,
            mesh=self.plan.mesh,
            in_specs=(self.plan.pressure_spec, self.plan.face_specs),
            out_specs=self.plan.face_specs,
        )(pressure, self.face_distances)

    def divergence(self, velocity: FaceVelocity, /) -> Array:
        self._require_ready()
        values = self._validate_velocity_sharding(velocity)

        def local(components, measures, volumes):
            return self.local_stencils.divergence(components, measures, volumes)

        return jax.shard_map(
            local,
            mesh=self.plan.mesh,
            in_specs=(
                self.plan.face_specs,
                self.plan.face_specs,
                self.plan.pressure_spec,
            ),
            out_specs=self.plan.pressure_spec,
        )(values, self.face_measures, self.cell_volumes)

    def interpolate_inverse_momentum(self, inverse_momentum: Array, /) -> FaceVelocity:
        self._require_ready()
        self._validate_pressure_sharding(inverse_momentum)

        def local(value):
            return self.local_stencils.interpolate_inverse_momentum(value)

        return jax.shard_map(
            local,
            mesh=self.plan.mesh,
            in_specs=self.plan.pressure_spec,
            out_specs=self.plan.face_specs,
        )(inverse_momentum)

    def convection(self, velocity: FaceVelocity, /) -> FaceVelocity:
        self._require_ready()
        values = self._validate_velocity_sharding(velocity)
        if self.momentum_operators is None:
            raise RuntimeError("No PreparedMACMomentumOperators were bound at prepare().")
        action = jax.jit(
            self.momentum_operators.convection,
            in_shardings=(self.plan.face_shardings,),
            out_shardings=self.plan.face_shardings,
        )
        return action(values)

    def face_laplacian(
        self,
        velocity: FaceVelocity,
        /,
        *,
        homogeneous: bool = False,
    ) -> FaceVelocity:
        self._require_ready()
        values = self._validate_velocity_sharding(velocity)
        if self.momentum_operators is None:
            raise RuntimeError("No PreparedMACMomentumOperators were bound at prepare().")
        method = (
            self.momentum_operators.homogeneous_laplacian
            if homogeneous
            else self.momentum_operators.laplacian
        )
        action = jax.jit(
            method,
            in_shardings=(self.plan.face_shardings,),
            out_shardings=self.plan.face_shardings,
        )
        return action(values)

    def momentum_rate(
        self,
        velocity: FaceVelocity,
        viscosity: ArrayLike,
        /,
    ) -> FaceVelocity:
        coefficient = jnp.asarray(
            viscosity, dtype=self.operators.pressure_space.dtype
        ).reshape(())
        convection = self.convection(velocity)
        diffusion = self.face_laplacian(velocity)
        return tuple(
            -transport + coefficient * laplacian
            for transport, laplacian in zip(convection, diffusion, strict=True)
        )

    def diagnostics(
        self,
        pressure: Array,
        velocity: FaceVelocity,
        /,
    ) -> MACDistributedDiagnostics:
        self._require_ready()
        self._validate_pressure_sharding(pressure)
        values = self._validate_velocity_sharding(velocity)
        divergence = self.divergence(values)
        gradient = self.gradient(pressure)
        left = jnp.sum(self.cell_volumes * pressure * divergence)
        right = sum(
            jnp.sum(measure * component * derivative)
            for measure, component, derivative in zip(
                self.face_dual_measures, values, gradient, strict=True
            )
        )
        residual = jnp.abs(left + right)
        finite = _all_finite(
            (pressure, divergence) + values + tuple(gradient) + (left, right)
        )
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(left), jnp.abs(right)))
        tolerance = 4096.0 * jnp.finfo(pressure.dtype).eps * scale
        interface_count = sum(
            ownership.interface_face_count for ownership in self.interface_ownership
        )
        passed = finite & (residual <= tolerance)
        return MACDistributedDiagnostics(
            weighted_adjoint_residual=residual,
            pressure_pairing=left,
            velocity_pairing=right,
            all_finite=finite,
            passed=passed,
            interface_face_count=interface_count,
            authoritative_interface_face_count=interface_count,
            topology_id=self.plan.topology_id,
            diagnostics_id=canonical_fingerprint(
                {
                    "kind": "mac-distributed-diagnostics",
                    "prepared": self.prepared_id,
                    "ownership": "exactly-once",
                }
            ),
        )


__all__ = [
    "MACDistributedDiagnostics",
    "MACDistributedPlanStatus",
    "MACDistributedState",
    "MACDistributedTopologyPlan",
    "MACHaloMetadata",
    "MACInterfaceFaceOwnership",
    "MACLocalStencilPlan",
    "PreparedMACDistributedTopology",
]
