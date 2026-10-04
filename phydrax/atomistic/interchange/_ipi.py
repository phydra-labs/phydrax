#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""i-PI socket transport with an explicit atomic-unit wire contract.

The i-PI driver protocol carries little-endian float64 payloads in Hartree
atomic units: lengths in bohr, energies in hartree, and forces in
hartree/bohr. Cells travel as ``h``, whose columns are lattice vectors, in C
order, followed by ``inv(h)``; Phydrax lattice vectors are rows, so
``h = cell_vectors.T``. The force reply carries the configurational virial
``W = -dE/d(strain) = -V * stress`` in hartree, transmitted as ``W.T`` in C
order. Native records only hold the tensile stress of
``ExternalAtomisticEvaluation``; the virial exists only on the wire. An
all-zero cell with an all-zero inverse is the aperiodic encoding.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import struct
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, TypeAlias, TypedDict, Unpack

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse
from ...units import (
    BOHR,
    conversion_factor,
    derived_unit,
    HARTREE,
    SI_REFERENCE_SYSTEM_ID,
)
from .._hybrid import AbstractExternalAtomisticProvider, ExternalAtomisticEvaluation
from .._system import PreparedAtomisticSystem


_HEADER = 12
_RECEIVE_CHUNK = 1 << 20
_MAXIMUM_INIT_BYTES = 1 << 20
_HARTREE_PER_BOHR = derived_unit("hartree/bohr", ((HARTREE, 1), (BOHR, -1)))
_WIRE_FLOAT = np.dtype("<f8")
_WIRE_INT = struct.Struct("<i")
_WIRE_DOUBLE = struct.Struct("<d")
_INVERSE_CELL_TOLERANCE = 1.0e-8

IPITransportMode: TypeAlias = Literal["unix", "tcp"]
IPIVirialPolicy: TypeAlias = Literal["required", "optional"]
IPIInverseCellPolicy: TypeAlias = Literal["verify", "ignore"]


class _IPITransportOptions(TypedDict, total=False):
    timeout: float
    maximum_atoms: int
    maximum_extra_bytes: int
    virial: IPIVirialPolicy
    inverse_cell: IPIInverseCellPolicy


class IPITransportStatus(IntEnum):
    READY = 0
    HAVE_DATA = 1
    CLOSED = 2
    PROTOCOL_ERROR = 3
    PROVIDER_ERROR = 4


class IPITransportPlan(StrictModule, NonTrainableState):
    """Socket endpoint, capacity bounds, and wire semantics of one i-PI link.

    ``virial="required"`` refuses a transaction whose configurational virial
    is unavailable: a serving provider must return a stress for a fully
    periodic cell and a received virial must be finite. ``"optional"``
    transmits an all-NaN virial when the stress is unavailable and maps a
    received all-NaN virial to ``stress=None``; a zero is never substituted.
    ``inverse_cell="verify"`` requires the received inverse to equal
    ``inv(h)``; ``"ignore"`` treats ``h`` as the sole geometry, for peers such
    as ASE's socket server that send a transposed legacy inverse.
    """

    mode: IPITransportMode = eqx.field(static=True)
    address: str = eqx.field(static=True)
    port: int | None = eqx.field(static=True)
    timeout: float = eqx.field(static=True)
    maximum_atoms: int = eqx.field(static=True)
    maximum_extra_bytes: int = eqx.field(static=True)
    virial: IPIVirialPolicy = eqx.field(static=True)
    inverse_cell: IPIInverseCellPolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: IPITransportMode,
        address: str,
        /,
        *,
        port: int | None = None,
        timeout: float = 60.0,
        maximum_atoms: int = 1_000_000,
        maximum_extra_bytes: int = 1_000_000,
        virial: IPIVirialPolicy = "required",
        inverse_cell: IPIInverseCellPolicy = "verify",
    ) -> None:
        mode_ = parse(mode, IPITransportMode, "mode")
        virial_ = parse(virial, IPIVirialPolicy, "virial")
        inverse_ = parse(inverse_cell, IPIInverseCellPolicy, "inverse_cell")
        if port is not None and not 0 < port < 65536:
            raise ValueError("i-PI TCP port must lie in 1..65535.")
        match mode_:
            case "tcp":
                if port is None:
                    raise ValueError("TCP i-PI transport requires a valid port.")
            case "unix":
                if port is not None:
                    raise ValueError("Unix i-PI transport does not accept a port.")
                if len(address.encode()) >= 104:
                    raise ValueError(
                        "Unix i-PI socket path exceeds the portable 103-byte limit."
                    )
        if (
            not math.isfinite(timeout)
            or timeout <= 0.0
            or maximum_atoms <= 0
            or maximum_extra_bytes < 0
        ):
            raise ValueError("i-PI transport timeout and capacities must be positive.")
        self.mode = mode_
        self.address = address
        self.port = port
        self.timeout = float(timeout)
        self.maximum_atoms = maximum_atoms
        self.maximum_extra_bytes = maximum_extra_bytes
        self.virial = virial_
        self.inverse_cell = inverse_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ipi-transport",
                "mode": mode_,
                "address": address,
                "port": port,
                "timeout": self.timeout,
                "maximum_atoms": maximum_atoms,
                "maximum_extra_bytes": maximum_extra_bytes,
                "virial": virial_,
                "inverse_cell": inverse_,
            }
        )

    @classmethod
    def unix(cls, path: str, /, **kwargs: Unpack[_IPITransportOptions]) -> Self:
        return cls("unix", path, **kwargs)

    @classmethod
    def tcp(cls, host: str, port: int, /, **kwargs: Unpack[_IPITransportOptions]) -> Self:
        return cls("tcp", host, port=port, **kwargs)

    def _socket(self) -> socket.socket:
        family = socket.AF_UNIX if self.mode == "unix" else socket.AF_INET
        return socket.socket(family, socket.SOCK_STREAM)

    def _endpoint(self) -> str | tuple[str, int]:
        if self.port is None:
            return self.address
        return (self.address, self.port)

    def connect(self) -> "IPISession":
        connection = self._socket()
        connection.settimeout(self.timeout)
        connection.connect(self._endpoint())
        return IPISession(connection, self)

    def listen(self) -> "IPIListener":
        server = self._socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(self.timeout)
        server.bind(self._endpoint())
        server.listen(1)
        return IPIListener(server, self)


class IPIRequest(StrictModule):
    """One received position request converted into native system units.

    ``cell_vectors`` holds native row lattice vectors, or ``None`` for the
    aperiodic all-zero wire cell. ``request_id`` addresses the exact received
    payload bytes.
    """

    positions: Array
    cell_vectors: Array | None
    request_id: str = eqx.field(static=True)


class _WireScale(StrictModule, NonTrainableState):
    """Exact native-to-atomic-unit factors resolved from one unit system."""

    length: float = eqx.field(static=True)
    energy: float = eqx.field(static=True)
    force: float = eqx.field(static=True)

    def __init__(self, system: PreparedAtomisticSystem, /) -> None:
        scale = system.plan.units.scale
        if scale.length_unit.reference_system_id != SI_REFERENCE_SYSTEM_ID:
            raise ValueError(
                "i-PI atomic-unit transport requires SI-convertible atomistic units."
            )
        self.length = float(conversion_factor(scale.length_unit, BOHR))
        self.energy = float(conversion_factor(scale.energy_unit, HARTREE))
        self.force = float(conversion_factor(scale.force_unit, _HARTREE_PER_BOHR))


def _transported_cell(system: PreparedAtomisticSystem, /) -> bool:
    """Return whether the system is periodic, refusing cells i-PI cannot carry."""

    cell = system.cell
    if cell is None:
        return False
    if cell.rank != 3 or cell.ambient_dimension != 3 or not cell.fully_periodic:
        raise ValueError(
            "i-PI transport carries fully periodic 3D cells or the aperiodic zero cell."
        )
    return True


def _cell_volume(cell_vectors: np.ndarray, /) -> float:
    # Host interchange of a 3x3 wire cell; the volume is the requested value.
    volume = abs(float(np.linalg.det(cell_vectors)))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("i-PI cell must be finite and nonsingular.")
    return volume


def _encode_cell(
    cell_vectors: np.ndarray | None, scale: _WireScale, /
) -> tuple[bytes, bytes]:
    if cell_vectors is None:
        zero = np.zeros((3, 3), dtype=_WIRE_FLOAT).tobytes()
        return zero, zero
    _cell_volume(cell_vectors)
    h = np.ascontiguousarray(cell_vectors.T * scale.length, dtype=_WIRE_FLOAT)
    # The inverse cell is itself the transmitted protocol value.
    inverse = np.linalg.solve(h, np.eye(3, dtype=_WIRE_FLOAT))
    return h.tobytes(order="C"), np.ascontiguousarray(inverse, dtype=_WIRE_FLOAT).tobytes(
        order="C"
    )


def _decode_cell(
    h_bytes: bytes,
    inverse_bytes: bytes,
    scale: _WireScale,
    policy: IPIInverseCellPolicy,
    /,
) -> np.ndarray | None:
    h = np.frombuffer(h_bytes, dtype=_WIRE_FLOAT).reshape((3, 3)).astype(np.float64)
    inverse = (
        np.frombuffer(inverse_bytes, dtype=_WIRE_FLOAT).reshape((3, 3)).astype(np.float64)
    )
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(inverse)):
        raise ValueError("i-PI cell payload is not finite.")
    if not np.any(h):
        if np.any(inverse):
            raise ValueError("i-PI aperiodic zero cell carries a nonzero inverse.")
        return None
    _cell_volume(h)
    match policy:
        case "verify":
            if not np.allclose(
                inverse @ h, np.eye(3), rtol=0.0, atol=_INVERSE_CELL_TOLERANCE
            ):
                raise ValueError("i-PI inverse cell is not inv(h).")
        case "ignore":
            pass
    return h.T / scale.length


def _encode_virial(
    stress: np.ndarray | None,
    cell_vectors: np.ndarray | None,
    scale: _WireScale,
    policy: IPIVirialPolicy,
    /,
) -> bytes:
    if stress is None or cell_vectors is None:
        match policy:
            case "required":
                raise ValueError(
                    "i-PI virial is required but no periodic stress is available."
                )
            case "optional":
                return np.full((3, 3), np.nan, dtype=_WIRE_FLOAT).tobytes()
    virial = -_cell_volume(cell_vectors) * stress * scale.energy
    return np.ascontiguousarray(virial.T, dtype=_WIRE_FLOAT).tobytes(order="C")


def _decode_virial(
    payload: bytes,
    cell_vectors: np.ndarray | None,
    scale: _WireScale,
    policy: IPIVirialPolicy,
    /,
) -> np.ndarray | None:
    virial = (
        np.frombuffer(payload, dtype=_WIRE_FLOAT).reshape((3, 3)).T.astype(np.float64)
    )
    unavailable = bool(np.all(np.isnan(virial)))
    if not unavailable and not np.all(np.isfinite(virial)):
        raise ValueError("i-PI virial is partially non-finite.")
    if unavailable or cell_vectors is None:
        match policy:
            case "required":
                raise ValueError(
                    "i-PI virial is required but unavailable for this transaction."
                )
            case "optional":
                return None
    return -(virial / scale.energy) / _cell_volume(cell_vectors)


class IPISession:
    """One connected i-PI peer in either the server or the driver role."""

    def __init__(self, connection: socket.socket, plan: IPITransportPlan, /) -> None:
        self.connection = connection
        self.plan = plan
        self.status = IPITransportStatus.READY
        self.pending_request: IPIRequest | None = None
        self.pending_evaluation: ExternalAtomisticEvaluation | None = None

    def _fail(self, status: IPITransportStatus, message: str, /) -> ValueError:
        self.status = status
        return ValueError(message)

    def _recv_exact(self, count: int, /) -> bytes:
        """Read exactly ``count`` already-bounded bytes across partial reads."""

        chunks: list[bytes] = []
        remaining = count
        while remaining:
            chunk = self.connection.recv(min(remaining, _RECEIVE_CHUNK))
            if not chunk:
                self.status = IPITransportStatus.CLOSED
                raise ConnectionError("i-PI connection closed while receiving data.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _recv_int(self) -> int:
        return _WIRE_INT.unpack(self._recv_exact(_WIRE_INT.size))[0]

    def recv_command(self) -> str:
        raw = self._recv_exact(_HEADER)
        if not raw.isascii():
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR, "i-PI header is not ASCII."
            )
        return raw.decode("ascii").strip()

    def send_command(self, command: str, /) -> None:
        encoded = command.encode("ascii")
        if len(encoded) > _HEADER:
            raise ValueError("i-PI command exceeds 12 bytes.")
        self.connection.sendall(encoded.ljust(_HEADER, b" "))

    def _recv_atom_count(self, system: PreparedAtomisticSystem, /) -> int:
        count = self._recv_int()
        if count <= 0 or count > self.plan.maximum_atoms:
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR,
                "i-PI atom count exceeds transport capacity.",
            )
        if count != system.capacity:
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR,
                "i-PI atom count differs from the bound atomistic system.",
            )
        return count

    def _recv_extra(self) -> bytes:
        size = self._recv_int()
        if size < 0 or size > self.plan.maximum_extra_bytes:
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR, "i-PI extra payload size is invalid."
            )
        return self._recv_exact(size)

    def recv_init(self) -> tuple[int, bytes]:
        """Read the bounded ``INIT`` payload following its header."""

        bead = self._recv_int()
        size = self._recv_int()
        if size < 0 or size > _MAXIMUM_INIT_BYTES:
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR, "i-PI INIT payload size is invalid."
            )
        return bead, self._recv_exact(size)

    def recv_positions(self, system: PreparedAtomisticSystem, /) -> IPIRequest:
        """Read a ``POSDATA`` payload, following its header, in native units."""

        if self.status is not IPITransportStatus.READY:
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR,
                "i-PI positions arrived while a response is pending.",
            )
        scale = _WireScale(system)
        h_bytes = self._recv_exact(9 * _WIRE_FLOAT.itemsize)
        inverse_bytes = self._recv_exact(9 * _WIRE_FLOAT.itemsize)
        count = self._recv_atom_count(system)
        position_bytes = self._recv_exact(3 * count * _WIRE_FLOAT.itemsize)
        cell_vectors = _decode_cell(h_bytes, inverse_bytes, scale, self.plan.inverse_cell)
        positions = (
            np.frombuffer(position_bytes, dtype=_WIRE_FLOAT).reshape((count, 3))
            / scale.length
        )
        if not np.all(np.isfinite(positions)):
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR, "i-PI positions are not finite."
            )
        digest = hashlib.sha256(
            h_bytes + inverse_bytes + _WIRE_INT.pack(count) + position_bytes
        ).hexdigest()
        dtype = system.plan.coordinate_dtype
        request = IPIRequest(
            jnp.asarray(positions, dtype=dtype),
            None if cell_vectors is None else jnp.asarray(cell_vectors, dtype=dtype),
            canonical_fingerprint({"kind": "ipi-request", "payload_sha256": digest}),
        )
        self.pending_request = request
        return request

    def send_force(
        self,
        system: PreparedAtomisticSystem,
        evaluation: ExternalAtomisticEvaluation,
        /,
        *,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        """Answer the pending request with converted energy, forces, and virial."""

        request = self.pending_request
        if self.status is not IPITransportStatus.HAVE_DATA or request is None:
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR,
                "i-PI force response has no pending position request.",
            )
        scale = _WireScale(system)
        energy = float(np.asarray(evaluation.energy, dtype=np.float64))
        forces = np.asarray(evaluation.forces, dtype=np.float64)
        stress = (
            None
            if evaluation.stress is None
            else np.asarray(evaluation.stress, dtype=np.float64)
        )
        if (
            not bool(evaluation.successful)
            or forces.shape != (system.capacity, 3)
            or not math.isfinite(energy)
            or not np.all(np.isfinite(forces))
            or (stress is not None and not np.all(np.isfinite(stress)))
        ):
            raise self._fail(
                IPITransportStatus.PROVIDER_ERROR,
                "i-PI force response is unsuccessful, non-finite, or misaligned.",
            )
        cell = (
            None
            if request.cell_vectors is None or system.cell is None
            else np.asarray(request.cell_vectors, dtype=np.float64)
        )
        virial = _encode_virial(stress, cell, scale, self.plan.virial)
        payload = json.dumps(
            {} if extra is None else extra,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(payload) > self.plan.maximum_extra_bytes:
            raise ValueError("i-PI extra payload exceeds capacity.")
        self.send_command("FORCEREADY")
        self.connection.sendall(
            _WIRE_DOUBLE.pack(energy * scale.energy)
            + _WIRE_INT.pack(forces.shape[0])
            + np.ascontiguousarray(forces * scale.force, dtype=_WIRE_FLOAT).tobytes()
            + virial
            + _WIRE_INT.pack(len(payload))
            + payload
        )
        self.status = IPITransportStatus.READY
        self.pending_request = None
        self.pending_evaluation = None

    def _expect_status(self, expected: str, /) -> None:
        self.send_command("STATUS")
        reply = self.recv_command()
        if reply == "NEEDINIT" and expected == "READY":
            self.send_command("INIT")
            self.connection.sendall(_WIRE_INT.pack(0) + _WIRE_INT.pack(0))
            self.send_command("STATUS")
            reply = self.recv_command()
        if reply != expected:
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR,
                f"i-PI driver replied {reply!r}; expected {expected!r}.",
            )

    def send_positions(
        self,
        system: PreparedAtomisticSystem,
        positions: np.ndarray,
        cell_vectors: np.ndarray | None,
        /,
    ) -> None:
        """Poll the driver and transmit one position request in atomic units."""

        if self.status is not IPITransportStatus.READY:
            raise ValueError("i-PI provider session is not ready for a request.")
        scale = _WireScale(system)
        h_bytes, inverse_bytes = _encode_cell(cell_vectors, scale)
        self._expect_status("READY")
        self.send_command("POSDATA")
        self.connection.sendall(
            h_bytes
            + inverse_bytes
            + _WIRE_INT.pack(positions.shape[0])
            + np.ascontiguousarray(positions * scale.length, dtype=_WIRE_FLOAT).tobytes()
        )
        self.status = IPITransportStatus.HAVE_DATA

    def recv_force(
        self,
        system: PreparedAtomisticSystem,
        cell_vectors: np.ndarray | None,
        provider_id: str,
        /,
    ) -> ExternalAtomisticEvaluation:
        """Collect the driver's reply and convert it into native units."""

        if self.status is not IPITransportStatus.HAVE_DATA:
            raise ValueError("i-PI provider session has no outstanding request.")
        scale = _WireScale(system)
        self._expect_status("HAVEDATA")
        self.send_command("GETFORCE")
        if self.recv_command() != "FORCEREADY":
            raise self._fail(
                IPITransportStatus.PROTOCOL_ERROR,
                "i-PI provider returned an unexpected command.",
            )
        energy = _WIRE_DOUBLE.unpack(self._recv_exact(_WIRE_DOUBLE.size))[0]
        count = self._recv_atom_count(system)
        forces = np.frombuffer(
            self._recv_exact(3 * count * _WIRE_FLOAT.itemsize), dtype=_WIRE_FLOAT
        ).reshape((count, 3))
        virial = self._recv_exact(9 * _WIRE_FLOAT.itemsize)
        self._recv_extra()
        stress = _decode_virial(virial, cell_vectors, scale, self.plan.virial)
        self.status = IPITransportStatus.READY
        dtype = system.plan.coordinate_dtype
        successful = math.isfinite(energy) and bool(np.all(np.isfinite(forces)))
        return ExternalAtomisticEvaluation(
            jnp.asarray(energy / scale.energy, dtype=dtype),
            jnp.asarray(forces / scale.force, dtype=dtype),
            None if stress is None else jnp.asarray(stress, dtype=dtype),
            jnp.asarray(successful),
            provider_id,
        )

    def close(self) -> None:
        self.status = IPITransportStatus.CLOSED
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class IPIListener:
    def __init__(self, server: socket.socket, plan: IPITransportPlan, /) -> None:
        self.server = server
        self.plan = plan

    def accept(self) -> IPISession:
        connection, _ = self.server.accept()
        connection.settimeout(self.plan.timeout)
        return IPISession(connection, self.plan)

    def close(self) -> None:
        self.server.close()
        if self.plan.mode == "unix":
            Path(self.plan.address).unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _host_cell(
    system: PreparedAtomisticSystem, cell_vectors: ArrayLike | None, /
) -> np.ndarray | None:
    if not _transported_cell(system):
        if cell_vectors is not None:
            raise ValueError("An aperiodic atomistic system cannot transmit a cell.")
        return None
    cell = system.cell
    if cell is None:
        raise RuntimeError("Internal invariant failed: periodic system has no cell.")
    vectors = np.asarray(
        cell.vectors if cell_vectors is None else cell_vectors, dtype=np.float64
    )
    if vectors.shape != (3, 3) or not np.all(np.isfinite(vectors)):
        raise ValueError("i-PI cell vectors must be a finite (3, 3) row lattice.")
    return vectors


class TransportedExternalAtomisticProvider(AbstractExternalAtomisticProvider):
    """Evaluate an external i-PI driver as a host, nondifferentiable provider."""

    session: IPISession
    provider_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    def __init__(
        self, session: IPISession, provider_id: str, /, *, conservative: bool = True
    ) -> None:
        if not isinstance(session, IPISession):
            raise TypeError("Transported provider requires a live IPISession.")
        identifier = provider_id.strip()
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        self.session = session
        self.provider_id = identifier
        self.conservative = conservative
        self.differentiable = False

    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        cell = _host_cell(system, cell_vectors)
        coordinate = np.asarray(positions, dtype=np.float64)
        if coordinate.shape != (system.capacity, 3) or not np.all(
            np.isfinite(coordinate)
        ):
            raise ValueError("i-PI provider positions are invalid.")
        self.session.send_positions(system, coordinate, cell)
        return self.session.recv_force(system, cell, self.provider_id)


def serve_ipi_once(
    session: IPISession,
    provider: AbstractExternalAtomisticProvider,
    system: PreparedAtomisticSystem,
    /,
) -> IPITransportStatus:
    """Act as an i-PI driver until one force reply is sent or the peer exits.

    A periodic system evaluates the received cell. An aperiodic system ignores
    the transmitted box, which then only serves the peer's own bookkeeping,
    and its virial follows the plan's ``virial`` policy.
    """

    periodic = _transported_cell(system)
    while True:
        command = session.recv_command()
        if command == "STATUS":
            session.send_command(
                "HAVEDATA" if session.pending_evaluation is not None else "READY"
            )
        elif command == "INIT":
            session.recv_init()
        elif command == "POSDATA":
            request = session.recv_positions(system)
            cell = request.cell_vectors if periodic else None
            if periodic and cell is None:
                raise session._fail(
                    IPITransportStatus.PROTOCOL_ERROR,
                    "i-PI sent the aperiodic zero cell to a periodic system.",
                )
            evaluation = provider.evaluate(system, request.positions, cell)
            if evaluation.provider_id != provider.provider_id:
                raise session._fail(
                    IPITransportStatus.PROVIDER_ERROR,
                    "i-PI provider changed its bound identity.",
                )
            session.pending_evaluation = evaluation
            session.status = IPITransportStatus.HAVE_DATA
        elif command == "GETFORCE":
            evaluation = session.pending_evaluation
            if evaluation is None:
                raise session._fail(
                    IPITransportStatus.PROTOCOL_ERROR,
                    "i-PI requested force before sending positions.",
                )
            session.send_force(system, evaluation)
            return IPITransportStatus.READY
        elif command == "EXIT":
            session.close()
            return IPITransportStatus.CLOSED
        else:
            raise session._fail(
                IPITransportStatus.PROTOCOL_ERROR, f"Unknown i-PI command {command!r}."
            )


__all__ = [
    "IPIInverseCellPolicy",
    "IPIListener",
    "IPIRequest",
    "IPISession",
    "IPITransportMode",
    "IPITransportPlan",
    "IPITransportStatus",
    "IPIVirialPolicy",
    "TransportedExternalAtomisticProvider",
    "serve_ipi_once",
]
