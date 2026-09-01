#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._models import AbstractScatteringComponent
from ._ports import WavePort


class InstancePort(StrictModule):
    """A port address within one scattering-network definition level."""

    instance_id: str = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    def __init__(self, instance_id: str, port_id: str, /):
        instance = str(instance_id)
        port = str(port_id)
        if not instance or not port:
            raise ValueError("InstancePort IDs must be non-empty.")
        self.instance_id = instance
        self.port_id = port


class ScatteringInstance(StrictModule):
    """Named leaf component or hierarchical scattering-network instance."""

    component: AbstractScatteringComponent | ScatteringNetwork
    instance_id: str = eqx.field(static=True)

    def __init__(
        self,
        instance_id: str,
        component: AbstractScatteringComponent | ScatteringNetwork,
        /,
    ):
        identifier = str(instance_id)
        if not identifier:
            raise ValueError("instance_id must be non-empty.")
        if not isinstance(component, (AbstractScatteringComponent, ScatteringNetwork)):
            raise TypeError("component must be a scattering component or network.")
        self.component = component
        self.instance_id = identifier


class WaveConnectionMap(StrictModule):
    """Lossless coordinate map applied in both directions across one connection."""

    forward: Array
    reverse: Array
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward: ArrayLike,
        /,
        *,
        reverse: ArrayLike | None = None,
        map_id: str | None = None,
    ):
        forward_ = jnp.asarray(forward)
        if (
            forward_.ndim != 2
            or forward_.shape[0] == 0
            or forward_.shape[0] != forward_.shape[1]
            or not jnp.issubdtype(forward_.dtype, jnp.number)
        ):
            raise ValueError("Wave connection maps must be nonempty square matrices.")
        forward_ = forward_.astype(jnp.result_type(forward_, jnp.complex128))
        reverse_ = (
            jnp.conj(forward_.T)
            if reverse is None
            else jnp.asarray(reverse, dtype=forward_.dtype)
        )
        if reverse_.shape != forward_.shape:
            raise ValueError("Forward and reverse connection maps must have equal shape.")
        identity = jnp.eye(forward_.shape[0], dtype=forward_.dtype)
        tolerance = 100 * jnp.finfo(forward_.real.dtype).eps * max(1, forward_.shape[0])
        defect = jnp.maximum(
            jnp.linalg.norm(reverse_ @ forward_ - identity),
            jnp.linalg.norm(forward_ @ reverse_ - identity),
        )
        adjoint_defect = jnp.linalg.norm(reverse_ - jnp.conj(forward_.T))
        if bool(defect > tolerance) or bool(adjoint_defect > tolerance):
            raise ValueError(
                "Direct wave connection maps must be mutually inverse and lossless."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "wave-connection-map",
                    "forward": array_tree_fingerprint(forward_),
                    "reverse": array_tree_fingerprint(reverse_),
                }
            )
            if map_id is None
            else str(map_id)
        )
        if not identifier:
            raise ValueError("map_id must be non-empty.")
        self.forward, self.reverse, self.map_id = forward_, reverse_, identifier

    @property
    def size(self) -> int:
        return int(self.forward.shape[0])


class WaveConnection(StrictModule):
    """One lossless pairwise wave-port connection."""

    first: InstancePort
    second: InstancePort
    mapping: WaveConnectionMap | None

    def __init__(
        self,
        first: InstancePort,
        second: InstancePort,
        /,
        *,
        mapping: WaveConnectionMap | None = None,
    ):
        if not isinstance(first, InstancePort) or not isinstance(second, InstancePort):
            raise TypeError("WaveConnection endpoints must be InstancePort values.")
        if first.instance_id == second.instance_id and first.port_id == second.port_id:
            raise ValueError("A wave connection cannot join a port to itself.")
        if mapping is not None and not isinstance(mapping, WaveConnectionMap):
            raise TypeError("mapping must be WaveConnectionMap or None.")
        self.first, self.second, self.mapping = first, second, mapping


class WaveProbe(StrictModule):
    """Read-only gather of incident and outgoing waves at one internal port."""

    port: InstancePort
    probe_id: str = eqx.field(static=True)

    def __init__(self, probe_id: str, port: InstancePort, /):
        identifier = str(probe_id)
        if not identifier:
            raise ValueError("probe_id must be non-empty.")
        if not isinstance(port, InstancePort):
            raise TypeError("port must be InstancePort.")
        self.port = port
        self.probe_id = identifier


class ScatteringNetwork(AbstractScatteringComponent):
    """Declarative hierarchy flattened into one global wave equation at planning."""

    instances: tuple[ScatteringInstance, ...]
    connections: tuple[WaveConnection, ...]
    external_ports: tuple[InstancePort, ...]
    probes: tuple[WaveProbe, ...]
    external_port_ids: tuple[str, ...] = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        instances: Sequence[ScatteringInstance],
        connections: Sequence[WaveConnection],
        external_ports: Sequence[InstancePort],
        /,
        *,
        external_port_ids: Sequence[str] | None = None,
        probes: Sequence[WaveProbe] = (),
        network_id: str = "scattering-network",
    ):
        instance_tuple = tuple(instances)
        connection_tuple = tuple(connections)
        external_tuple = tuple(external_ports)
        probe_tuple = tuple(probes)
        if not instance_tuple or any(
            not isinstance(item, ScatteringInstance) for item in instance_tuple
        ):
            raise ValueError(
                "instances must be a non-empty sequence of ScatteringInstance values."
            )
        identifiers = tuple(item.instance_id for item in instance_tuple)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Scattering instance IDs must be unique within a network.")
        if any(not isinstance(item, WaveConnection) for item in connection_tuple):
            raise TypeError("connections must contain WaveConnection values.")
        if not external_tuple or any(
            not isinstance(item, InstancePort) for item in external_tuple
        ):
            raise ValueError(
                "external_ports must be a non-empty sequence of InstancePort values."
            )
        if any(not isinstance(item, WaveProbe) for item in probe_tuple):
            raise TypeError("probes must contain WaveProbe values.")
        probe_ids = tuple(item.probe_id for item in probe_tuple)
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError("Probe IDs must be unique within a network.")
        exposed_ids = (
            tuple(f"{port.instance_id}.{port.port_id}" for port in external_tuple)
            if external_port_ids is None
            else tuple(str(value) for value in external_port_ids)
        )
        if (
            len(exposed_ids) != len(external_tuple)
            or len(set(exposed_ids)) != len(exposed_ids)
            or any(not value for value in exposed_ids)
        ):
            raise ValueError(
                "external_port_ids must be unique, non-empty, and match external_ports."
            )
        identifier = str(network_id)
        if not identifier:
            raise ValueError("network_id must be non-empty.")
        self.instances = instance_tuple
        self.connections = connection_tuple
        self.external_ports = external_tuple
        self.probes = probe_tuple
        self.external_port_ids = exposed_ids
        self.network_id = identifier

    def _instance(self, instance_id: str) -> ScatteringInstance:
        for instance in self.instances:
            if instance.instance_id == instance_id:
                return instance
        raise KeyError(f"Unknown scattering instance {instance_id!r}.")

    def _resolved_port(self, address: InstancePort) -> WavePort:
        component = self._instance(address.instance_id).component
        for port in component.ports:
            if port.port_id == address.port_id:
                return port
        raise KeyError(
            f"Unknown port {address.port_id!r} on instance {address.instance_id!r}."
        )

    @property
    def ports(self) -> tuple[WavePort, ...]:
        return tuple(
            WavePort(
                external_id,
                self._resolved_port(address).references,
                coordinate_ids=self._resolved_port(address).coordinate_ids,
            )
            for external_id, address in zip(
                self.external_port_ids, self.external_ports, strict=True
            )
        )

    def evaluate(self, angular_frequency, /):
        from ._network import full_scattering_matrix, prepare_scattering_network

        prepared = prepare_scattering_network(self, angular_frequency)
        matrix = full_scattering_matrix(prepared)
        from ._models import ScatteringResponse

        return ScatteringResponse(
            matrix,
            tuple(reference for port in self.ports for reference in port.references),
            prepared.numeric_version,
        )


__all__ = [
    "InstancePort",
    "ScatteringInstance",
    "ScatteringNetwork",
    "WaveConnection",
    "WaveConnectionMap",
    "WaveProbe",
]
