# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Named native layouts for bulk inventories and inventory-free surface traces."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._strict import StrictModule
from ...dynamics import DAEVariableBlock, StateLayout


class SemiconductorStateLayout(StrictModule):
    native: StateLayout
    variables: tuple[DAEVariableBlock, ...]
    names: tuple[str, ...] = eqx.field(static=True)
    routes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    bulk_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        node_count,
        *,
        electrothermal=False,
        carrier_energy=False,
        interfaces=(),
        traps=(),
    ):
        if (
            isinstance(node_count, bool)
            or not isinstance(node_count, (int, np.integer))
            or node_count < 1
        ):
            raise ValueError("node_count must be a positive integer.")
        if not isinstance(electrothermal, bool) or not isinstance(carrier_energy, bool):
            raise TypeError("State model selections must be Boolean.")
        if carrier_energy and not electrothermal:
            raise ValueError("Carrier-energy fields require lattice-energy storage.")
        if not isinstance(interfaces, tuple) or not isinstance(traps, tuple):
            raise TypeError("interfaces and traps must be immutable tuples.")
        node_count = int(node_count)
        bulk = ("potential", "electron", "hole")
        if electrothermal:
            bulk += ("lattice_energy",)
        if carrier_energy:
            bulk += ("electron_energy", "hole_energy")
        names = list(bulk)
        routes = [
            tuple(range(i, node_count * len(bulk), len(bulk))) for i in range(len(bulk))
        ]
        labels = [f"{name}:{node}" for node in range(node_count) for name in bulk]
        for index, interface in enumerate(interfaces):
            if interface.electron_law is not None:
                for name in (
                    "electron_left",
                    "electron_right",
                    "hole_left",
                    "hole_right",
                ):
                    names.append(f"interface_{index}_{name}")
                    routes.append((len(labels),))
                    labels.append(names[-1])
        for index in range(len(traps)):
            names.append(f"trap_{index}")
            routes.append((len(labels),))
            labels.append(names[-1])
        self.shape = (
            (node_count, len(bulk))
            if len(labels) == node_count * len(bulk)
            else (len(labels),)
        )
        self.native = StateLayout(self.shape, component_names=labels)
        self.variables = tuple(
            DAEVariableBlock(
                name,
                (len(route),),
                0 if name == "potential" or name.startswith("interface_") else 1,
            )
            for name, route in zip(names, routes, strict=True)
        )
        self.names, self.routes, self.bulk_names = tuple(names), tuple(routes), bulk
        self.node_count = node_count

    @property
    def size(self):
        return self.native.size

    def indices(self, name):
        return jnp.asarray(self.routes[self.names.index(name)], dtype=jnp.int32)

    def field(self, state, name):
        return jnp.asarray(state).reshape(-1)[self.indices(name)]

    def set(self, state, name, value):
        return (
            jnp.asarray(state)
            .reshape(-1)
            .at[self.indices(name)]
            .set(jnp.broadcast_to(value, (len(self.routes[self.names.index(name)]),)))
            .reshape(self.shape)
        )

    def pack(self, **fields):
        result = jnp.zeros(self.shape)
        for name, value in fields.items():
            result = self.set(result, name, value)
        return result

    def node_indices(self, nodes):
        return (
            np.asarray(nodes)[:, None] * len(self.bulk_names)
            + np.arange(len(self.bulk_names))[None, :]
        )
