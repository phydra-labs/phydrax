#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dynamics import PreparedFiniteVolumeDynamics


class FiniteVolumeHaloRoute(StrictModule, NonTrainableState):
    axis: int = eqx.field(static=True)
    side: str = eqx.field(static=True)
    mesh_axis: str = eqx.field(static=True)
    neighbor_offset: int = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    route_id: str = eqx.field(static=True)


class FiniteVolumeShardingReport(StrictModule, NonTrainableState):
    global_shape: tuple[int, ...] = eqx.field(static=True)
    local_shape: tuple[int, ...] = eqx.field(static=True)
    split_factors: tuple[int, ...] = eqx.field(static=True)
    device_count: int = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class FiniteVolumeDecompositionPlan(StrictModule, NonTrainableState):
    """Cartesian named-sharding plan with explicit local extent validation."""

    global_shape: tuple[int, ...] = eqx.field(static=True)
    split_factors: tuple[int, ...] = eqx.field(static=True)
    axis_names: tuple[str, ...] = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        global_shape: Sequence[int],
        split_factors: Sequence[int],
        axis_names: Sequence[str],
        /,
        *,
        halo_width: int,
    ):
        shape = tuple(int(value) for value in global_shape)
        splits = tuple(int(value) for value in split_factors)
        names = tuple(str(value) for value in axis_names)
        width = int(halo_width)
        if (
            not shape
            or len(shape) != len(splits)
            or len(shape) != len(names)
            or any(value <= 0 for value in shape + splits)
            or any(not name for name in names)
            or len(set(names)) != len(names)
            or width < 0
        ):
            raise ValueError("Finite-volume decomposition metadata is invalid.")
        if any(size % split for size, split in zip(shape, splits, strict=True)):
            raise ValueError("Global FV shape must divide exactly by split factors.")
        local = tuple(size // split for size, split in zip(shape, splits, strict=True))
        if any(size < max(1, 2 * width) for size in local):
            raise ValueError("Local FV extent is smaller than the required halo reach.")
        self.global_shape = shape
        self.split_factors = splits
        self.axis_names = names
        self.halo_width = width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-volume-decomposition",
                "global_shape": list(shape),
                "split_factors": list(splits),
                "axis_names": list(names),
                "halo_width": width,
            }
        )

    def prepare(
        self,
        devices: Sequence[jax.Device] | None = None,
        /,
    ) -> "PreparedFiniteVolumeDecomposition":
        return PreparedFiniteVolumeDecomposition(self, devices=devices)


class PreparedFiniteVolumeDecomposition(StrictModule, NonTrainableState):
    plan: FiniteVolumeDecompositionPlan
    mesh: Mesh
    cell_sharding: NamedSharding
    local_shape: tuple[int, ...] = eqx.field(static=True)
    mesh_axis_names: tuple[str, ...] = eqx.field(static=True)
    report: FiniteVolumeShardingReport
    halo_routes: tuple[FiniteVolumeHaloRoute, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FiniteVolumeDecompositionPlan,
        /,
        *,
        devices: Sequence[jax.Device] | None = None,
    ):
        if not isinstance(plan, FiniteVolumeDecompositionPlan):
            raise TypeError("plan must be a FiniteVolumeDecompositionPlan.")
        available = tuple(jax.devices() if devices is None else devices)
        required = prod(plan.split_factors)
        if len(available) < required:
            raise ValueError(
                f"FV decomposition requires {required} devices, found {len(available)}."
            )
        selected = np.asarray(available[:required], dtype=object).reshape(
            plan.split_factors
        )
        mesh_axis_names = tuple(f"fv_{name}" for name in plan.axis_names)
        mesh = Mesh(selected, mesh_axis_names)
        partition = tuple(
            mesh_axis_names[axis] if split > 1 else None
            for axis, split in enumerate(plan.split_factors)
        ) + (None,)
        sharding = NamedSharding(mesh, PartitionSpec(*partition))
        routes = tuple(
            FiniteVolumeHaloRoute(
                axis=axis,
                side=side,
                mesh_axis=mesh_axis_names[axis],
                neighbor_offset=-1 if side == "lower" else 1,
                halo_width=plan.halo_width,
                periodic=True,
                route_id=canonical_fingerprint(
                    {
                        "kind": "finite-volume-halo-route",
                        "plan": plan.plan_id,
                        "axis": axis,
                        "side": side,
                    }
                ),
            )
            for axis, split in enumerate(plan.split_factors)
            if split > 1
            for side in ("lower", "upper")
        )
        local = tuple(
            size // split
            for size, split in zip(plan.global_shape, plan.split_factors, strict=True)
        )
        report_id = canonical_fingerprint(
            {
                "kind": "finite-volume-sharding-report",
                "plan": plan.plan_id,
                "local_shape": list(local),
                "device_count": required,
            }
        )
        report = FiniteVolumeShardingReport(
            global_shape=plan.global_shape,
            local_shape=local,
            split_factors=plan.split_factors,
            device_count=required,
            halo_width=plan.halo_width,
            report_id=report_id,
        )
        self.plan = plan
        self.mesh = mesh
        self.cell_sharding = sharding
        self.local_shape = local
        self.mesh_axis_names = mesh_axis_names
        self.report = report
        self.halo_routes = routes
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-volume-decomposition",
                "plan": plan.plan_id,
                "report": report_id,
            }
        )

    def shard_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[:-1] != self.plan.global_shape:
            raise ValueError("Distributed FV state does not match global shape.")
        return jax.device_put(value, self.cell_sharding)

    def periodic_halo(self, state: Array, axis: int, /) -> Array:
        axis_ = int(axis)
        width = self.plan.halo_width
        if not 0 <= axis_ < len(self.plan.global_shape):
            raise ValueError("Distributed FV halo axis is out of range.")
        if width == 0:
            return state
        lower = jnp.take(
            state,
            jnp.arange(state.shape[axis_] - width, state.shape[axis_]),
            axis=axis_,
        )
        upper = jnp.take(state, jnp.arange(width), axis=axis_)
        return jnp.concatenate((lower, state, upper), axis=axis_)

    def materialize_periodic_halos(self, state: Array, /) -> tuple[Array, ...]:
        return tuple(
            self.periodic_halo(state, axis) for axis in range(len(self.plan.global_shape))
        )

    def compile_residual(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        time: ArrayLike,
        args: Any = None,
        /,
    ) -> Callable[[Array], Array]:
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("dynamics must be PreparedFiniteVolumeDynamics.")
        if dynamics.discretization.cell_shape != self.plan.global_shape:
            raise ValueError("Distributed FV dynamics and decomposition shapes differ.")
        return jax.jit(
            lambda value: dynamics(jnp.asarray(time), value, args),
            in_shardings=self.cell_sharding,
            out_shardings=self.cell_sharding,
        )

    def residual(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        time: ArrayLike,
        state: Array,
        args: Any = None,
        /,
    ) -> Array:
        return self.compile_residual(dynamics, time, args)(state)


__all__ = [
    "FiniteVolumeDecompositionPlan",
    "FiniteVolumeHaloRoute",
    "FiniteVolumeShardingReport",
    "PreparedFiniteVolumeDecomposition",
]
