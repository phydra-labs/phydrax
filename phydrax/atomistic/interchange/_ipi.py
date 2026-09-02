#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import socket
import struct
from enum import IntEnum
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._hybrid import AbstractExternalAtomisticProvider, ExternalAtomisticEvaluation
from .._system import PreparedAtomisticSystem


_HEADER = 12


class IPITransportStatus(IntEnum):
    READY = 0
    HAVE_DATA = 1
    CLOSED = 2
    PROTOCOL_ERROR = 3
    PROVIDER_ERROR = 4


class IPITransportPlan(StrictModule, NonTrainableState):
    mode: str = eqx.field(static=True)
    address: str = eqx.field(static=True)
    port: int | None = eqx.field(static=True)
    timeout: float = eqx.field(static=True)
    maximum_atoms: int = eqx.field(static=True)
    maximum_extra_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: str,
        address: str,
        /,
        *,
        port: int | None = None,
        timeout: float = 60.0,
        maximum_atoms: int = 1_000_000,
        maximum_extra_bytes: int = 1_000_000,
    ):
        if mode not in ("unix", "tcp"):
            raise ValueError("i-PI transport mode must be unix or tcp.")
        port_ = None if port is None else int(port)
        if mode == "tcp" and (port_ is None or not 0 < port_ < 65536):
            raise ValueError("TCP i-PI transport requires a valid port.")
        if mode == "unix" and port_ is not None:
            raise ValueError("Unix i-PI transport does not accept a port.")
        if mode == "unix" and len(str(address).encode()) >= 104:
            raise ValueError("Unix i-PI socket path exceeds the portable 103-byte limit.")
        if (
            float(timeout) <= 0.0
            or int(maximum_atoms) <= 0
            or int(maximum_extra_bytes) < 0
        ):
            raise ValueError("i-PI transport capacities must be positive.")
        self.mode = mode
        self.address = str(address)
        self.port = port_
        self.timeout = float(timeout)
        self.maximum_atoms = int(maximum_atoms)
        self.maximum_extra_bytes = int(maximum_extra_bytes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ipi-transport",
                "mode": mode,
                "address": self.address,
                "port": port_,
                "timeout": self.timeout,
                "maximum_atoms": self.maximum_atoms,
                "maximum_extra_bytes": self.maximum_extra_bytes,
            }
        )

    @classmethod
    def unix(cls, path: str, /, **kwargs):
        return cls("unix", path, **kwargs)

    @classmethod
    def tcp(cls, host: str, port: int, /, **kwargs):
        return cls("tcp", host, port=port, **kwargs)

    def connect(self) -> "IPISession":
        family = socket.AF_UNIX if self.mode == "unix" else socket.AF_INET
        connection = socket.socket(family, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        connection.connect(
            self.address if self.mode == "unix" else (self.address, self.port)
        )
        return IPISession(connection, self)

    def listen(self) -> "IPIListener":
        family = socket.AF_UNIX if self.mode == "unix" else socket.AF_INET
        server = socket.socket(family, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(self.timeout)
        server.bind(self.address if self.mode == "unix" else (self.address, self.port))
        server.listen(1)
        return IPIListener(server, self)


class IPIRequest(StrictModule):
    cell: jnp.ndarray
    inverse_cell: jnp.ndarray
    positions: jnp.ndarray
    request_id: str = eqx.field(static=True)


class IPIResponse(StrictModule):
    energy: jnp.ndarray
    forces: jnp.ndarray
    virial: jnp.ndarray
    extra: dict
    successful: jnp.ndarray
    request_id: str = eqx.field(static=True)


class IPISession:
    def __init__(self, connection: socket.socket, plan: IPITransportPlan, /):
        self.connection = connection
        self.plan = plan
        self.status = IPITransportStatus.READY
        self.pending: IPIResponse | None = None

    def _recv_exact(self, count: int) -> bytes:
        chunks = []
        remaining = count
        while remaining:
            chunk = self.connection.recv(remaining)
            if not chunk:
                raise ConnectionError("i-PI connection closed while receiving data.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def recv_command(self) -> str:
        return self._recv_exact(_HEADER).decode("ascii").strip()

    def send_command(self, command: str) -> None:
        encoded = command.encode("ascii")
        if len(encoded) > _HEADER:
            raise ValueError("i-PI command exceeds 12 bytes.")
        self.connection.sendall(encoded.ljust(_HEADER, b" "))

    def recv_positions(self) -> IPIRequest:
        cell = np.frombuffer(self._recv_exact(9 * 8), dtype="<f8").reshape((3, 3))
        inverse = np.frombuffer(self._recv_exact(9 * 8), dtype="<f8").reshape((3, 3))
        atom_count = struct.unpack("<i", self._recv_exact(4))[0]
        if atom_count <= 0 or atom_count > self.plan.maximum_atoms:
            raise ValueError("i-PI atom count exceeds transport capacity.")
        positions = np.frombuffer(
            self._recv_exact(atom_count * 3 * 8), dtype="<f8"
        ).reshape((atom_count, 3))
        request_id = canonical_fingerprint(
            {
                "kind": "ipi-request",
                "cell": cell.tolist(),
                "positions": positions.tolist(),
            }
        )
        self.status = IPITransportStatus.HAVE_DATA
        return IPIRequest(
            jnp.asarray(cell), jnp.asarray(inverse), jnp.asarray(positions), request_id
        )

    def send_force(self, response: IPIResponse) -> None:
        if self.status is not IPITransportStatus.HAVE_DATA:
            raise ValueError("i-PI force response has no pending position request.")
        if not isinstance(response, IPIResponse):
            raise TypeError("response must be IPIResponse.")
        forces = np.asarray(response.forces, dtype="<f8")
        virial = np.asarray(response.virial, dtype="<f8")
        if (
            forces.ndim != 2
            or forces.shape[1] != 3
            or forces.shape[0] > self.plan.maximum_atoms
            or virial.shape != (3, 3)
            or not bool(response.successful)
            or not np.isfinite(float(response.energy))
            or not np.all(np.isfinite(forces))
            or not np.all(np.isfinite(virial))
        ):
            raise ValueError("i-PI force response is invalid or unsuccessful.")
        extra = json.dumps(response.extra, sort_keys=True, separators=(",", ":")).encode()
        if len(extra) > self.plan.maximum_extra_bytes:
            raise ValueError("i-PI extra payload exceeds capacity.")
        self.send_command("FORCEREADY")
        self.connection.sendall(struct.pack("<d", float(response.energy)))
        self.connection.sendall(struct.pack("<i", forces.shape[0]))
        self.connection.sendall(forces.tobytes(order="C"))
        self.connection.sendall(virial.reshape((9,)).tobytes())
        self.connection.sendall(struct.pack("<i", len(extra)))
        self.connection.sendall(extra)
        self.status = IPITransportStatus.READY

    def close(self) -> None:
        self.status = IPITransportStatus.CLOSED
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class IPIListener:
    def __init__(self, server: socket.socket, plan: IPITransportPlan, /):
        self.server = server
        self.plan = plan

    def accept(self) -> IPISession:
        connection, _ = self.server.accept()
        connection.settimeout(self.plan.timeout)
        return IPISession(connection, self.plan)

    def close(self):
        self.server.close()
        if self.plan.mode == "unix":
            Path(self.plan.address).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class TransportedExternalAtomisticProvider(AbstractExternalAtomisticProvider):
    session: object
    provider_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    def __init__(
        self, session: IPISession, provider_id: str, /, *, conservative: bool = True
    ):
        self.session = session
        self.provider_id = str(provider_id)
        self.conservative = bool(conservative)
        self.differentiable = False

    def evaluate(
        self, system: PreparedAtomisticSystem, positions, cell_vectors, /
    ) -> ExternalAtomisticEvaluation:
        session = self.session
        if not isinstance(session, IPISession):
            raise TypeError("Transported provider has no live i-PI session.")
        if session.status is not IPITransportStatus.READY:
            raise ValueError("i-PI provider session is not ready for a request.")
        cell = (
            np.eye(3) if cell_vectors is None else np.asarray(cell_vectors, dtype=float)
        )
        coordinate = np.asarray(positions, dtype="<f8")
        if (
            cell.shape != (3, 3)
            or coordinate.shape != (system.capacity, 3)
            or not np.all(np.isfinite(cell))
            or not np.all(np.isfinite(coordinate))
        ):
            raise ValueError("i-PI provider cell or positions are invalid.")
        inverse = np.linalg.solve(cell, np.eye(3, dtype=cell.dtype))
        session.send_command("POSDATA")
        session.connection.sendall(np.asarray(cell, dtype="<f8").tobytes())
        session.connection.sendall(np.asarray(inverse, dtype="<f8").tobytes())
        coordinate = np.asarray(positions, dtype="<f8")
        session.connection.sendall(struct.pack("<i", coordinate.shape[0]))
        session.status = IPITransportStatus.HAVE_DATA
        session.connection.sendall(coordinate.tobytes())
        session.send_command("GETFORCE")
        if session.recv_command() != "FORCEREADY":
            raise ValueError("i-PI provider returned an unexpected command.")
        energy = struct.unpack("<d", session._recv_exact(8))[0]
        atom_count = struct.unpack("<i", session._recv_exact(4))[0]
        if atom_count != system.capacity:
            raise ValueError("i-PI force response atom count changed.")
        forces = np.frombuffer(
            session._recv_exact(atom_count * 3 * 8), dtype="<f8"
        ).reshape((atom_count, 3))
        virial = np.frombuffer(session._recv_exact(9 * 8), dtype="<f8").reshape((3, 3))
        extra_size = struct.unpack("<i", session._recv_exact(4))[0]
        if extra_size < 0 or extra_size > session.plan.maximum_extra_bytes:
            raise ValueError("i-PI extra response size is invalid.")
        extra = json.loads(session._recv_exact(extra_size) or b"{}")
        session.status = IPITransportStatus.READY
        return ExternalAtomisticEvaluation(
            jnp.asarray(energy),
            jnp.asarray(forces),
            jnp.asarray(virial),
            jnp.asarray(np.isfinite(energy) and np.isfinite(forces).all()),
            self.provider_id,
        )


def serve_ipi_once(
    session: IPISession,
    provider: AbstractExternalAtomisticProvider,
    system: PreparedAtomisticSystem,
    /,
) -> IPITransportStatus:
    request = None
    while True:
        command = session.recv_command()
        if command == "STATUS":
            session.send_command("HAVEDATA" if session.pending is not None else "READY")
        elif command == "POSDATA":
            request = session.recv_positions()
            evaluation = provider.evaluate(system, request.positions, request.cell)
            session.pending = IPIResponse(
                evaluation.energy,
                evaluation.forces,
                jnp.zeros((3, 3)) if evaluation.stress is None else evaluation.stress,
                {},
                evaluation.successful,
                request.request_id,
            )
        elif command == "GETFORCE":
            if session.pending is None:
                raise ValueError("i-PI requested force before sending positions.")
            session.send_force(session.pending)
            session.pending = None
            return IPITransportStatus.READY
        elif command == "EXIT":
            session.close()
            return IPITransportStatus.CLOSED
        else:
            raise ValueError(f"Unknown i-PI command {command!r}.")


__all__ = [
    "IPIListener",
    "IPIRequest",
    "IPIResponse",
    "IPISession",
    "IPITransportPlan",
    "IPITransportStatus",
    "TransportedExternalAtomisticProvider",
    "serve_ipi_once",
]
