import socket
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


# Independent i-PI oracle constants (CODATA 2018), deliberately not read from
# phydrax.units: 1 bohr in angstrom and 1 hartree in electronvolt.
BOHR_ANGSTROM = 0.529177210903
HARTREE_EV = 27.211386245988

# Skew row lattice (angstrom) with a nonzero off-diagonal in every row.
CELL = np.array([[5.0, 0.0, 0.0], [1.3, 4.6, 0.0], [0.7, -0.9, 5.4]])
POSITIONS = np.array([[0.4, 0.3, 0.2], [2.9, 2.1, 3.3]])
GRADIENT_LOAD = np.array([[0.11, -0.23, 0.05], [-0.04, 0.17, -0.31]])
VOLUME_COEFFICIENT = 0.013


def _oracle(
    positions: np.ndarray, cell: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """E = c det(H) - sum_i g_i . r_i at fixed fractional coordinates.

    Under r' = (I + e) r and H' = H (I + e).T, dE/de_ab = c V delta_ab
    - sum_i g_ia r_ib, so the tensile stress is sym(dE/de) / V.
    """

    volume = abs(float(np.linalg.det(cell)))
    energy = VOLUME_COEFFICIENT * volume - float(np.sum(GRADIENT_LOAD * positions))
    strain = VOLUME_COEFFICIENT * volume * np.eye(3) - GRADIENT_LOAD.T @ positions
    return energy, GRADIENT_LOAD.copy(), 0.5 * (strain + strain.T) / volume


def _system(*, periodic: bool) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    cell = phx.discretization.PeriodicCell(jnp.asarray(CELL)) if periodic else None
    return phx.atomistic.AtomisticSystemPlan(
        jnp.asarray([0, 1]),
        jnp.asarray([1, 8]),
        jnp.asarray([1.008, 15.999]),
        units,
        cell=cell,
    ).prepare()


def _provider(*, with_stress: bool) -> Any:
    def evaluator(system: Any, positions: Any, cell_vectors: Any) -> Any:
        del system
        cell = CELL if cell_vectors is None else np.asarray(cell_vectors)
        energy, forces, stress = _oracle(np.asarray(positions), cell)
        return phx.atomistic.ExternalAtomisticEvaluation(
            jnp.asarray(energy),
            jnp.asarray(forces),
            jnp.asarray(stress) if with_stress else None,
            jnp.asarray(True),
            "oracle",
        )

    return phx.atomistic.CallableBornOppenheimerProvider(evaluator, "oracle")


def _plan(**kwargs: Any) -> Any:
    return phx.atomistic.interchange.IPITransportPlan.unix(
        "/tmp/phydrax-ipi-unused.sock", timeout=5.0, **kwargs
    )


def _header(command: str) -> bytes:
    return command.encode("ascii").ljust(12, b" ")


def _read_exact(peer: socket.socket, count: int) -> bytes:
    data = b""
    while len(data) < count:
        chunk = peer.recv(count - len(data))
        assert chunk, "peer closed early"
        data += chunk
    return data


def _posdata(cell: np.ndarray, positions: np.ndarray) -> bytes:
    h = cell.T / BOHR_ANGSTROM
    return (
        _header("POSDATA")
        + h.astype("<f8").tobytes(order="C")
        + np.linalg.inv(h).astype("<f8").tobytes(order="C")
        + struct.pack("<i", positions.shape[0])
        + (positions / BOHR_ANGSTROM).astype("<f8").tobytes(order="C")
    )


def _drive_once(
    peer: socket.socket, payload: bytes, send: Callable[[bytes], None]
) -> dict[str, Any]:
    """Act as the i-PI server: poll, send positions, and collect the reply."""

    send(_header("STATUS"))
    assert _read_exact(peer, 12) == _header("READY")
    send(payload)
    send(_header("STATUS"))
    assert _read_exact(peer, 12) == _header("HAVEDATA")
    send(_header("GETFORCE"))
    assert _read_exact(peer, 12) == _header("FORCEREADY")
    energy = struct.unpack("<d", _read_exact(peer, 8))[0]
    count = struct.unpack("<i", _read_exact(peer, 4))[0]
    forces = np.frombuffer(_read_exact(peer, 24 * count), dtype="<f8").reshape(count, 3)
    virial = np.frombuffer(_read_exact(peer, 72), dtype="<f8").reshape(3, 3)
    extra = _read_exact(peer, struct.unpack("<i", _read_exact(peer, 4))[0])
    return {"energy": energy, "forces": forces, "virial_wire": virial, "extra": extra}


def _serve(system: Any, provider: Any, plan: Any) -> tuple[socket.socket, Any]:
    driver_end, peer = socket.socketpair()
    session = phx.atomistic.interchange.IPISession(driver_end, plan)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        phx.atomistic.interchange.serve_ipi_once, session, provider, system
    )
    executor.shutdown(wait=False)
    return peer, future


def test_driver_reply_matches_independent_atomic_unit_oracle() -> None:
    system = _system(periodic=True)
    peer, future = _serve(system, _provider(with_stress=True), _plan())
    with peer:
        reply = _drive_once(peer, _posdata(CELL, POSITIONS), peer.sendall)
    assert (
        future.result(timeout=10.0) is phx.atomistic.interchange.IPITransportStatus.READY
    )

    energy, forces, stress = _oracle(POSITIONS, CELL)
    volume = abs(float(np.linalg.det(CELL)))
    np.testing.assert_allclose(reply["energy"], energy / HARTREE_EV, rtol=1e-12)
    np.testing.assert_allclose(
        reply["forces"], forces * BOHR_ANGSTROM / HARTREE_EV, rtol=1e-12
    )
    # i-PI virial: W = -V sigma in hartree, transmitted as W.T in C order.
    expected_virial = -volume * stress / HARTREE_EV
    np.testing.assert_allclose(reply["virial_wire"].T, expected_virial, rtol=1e-12)
    # The strain derivative is nonzero, so a zero or stress-as-virial reply fails.
    assert np.max(np.abs(expected_virial)) > 1e-3
    assert not np.allclose(reply["virial_wire"], stress, atol=1e-6)


def test_driver_survives_fragmented_partial_reads() -> None:
    system = _system(periodic=True)
    peer, future = _serve(system, _provider(with_stress=True), _plan())

    def trickle(data: bytes) -> None:
        for start in range(0, len(data), 5):
            peer.sendall(data[start : start + 5])
            time.sleep(0.0005)

    with peer:
        reply = _drive_once(peer, _posdata(CELL, POSITIONS), trickle)
    assert (
        future.result(timeout=10.0) is phx.atomistic.interchange.IPITransportStatus.READY
    )
    np.testing.assert_allclose(
        reply["energy"], _oracle(POSITIONS, CELL)[0] / HARTREE_EV, rtol=1e-12
    )


def test_driver_refuses_truncated_position_payload() -> None:
    system = _system(periodic=True)
    peer, future = _serve(system, _provider(with_stress=True), _plan())
    peer.sendall(_posdata(CELL, POSITIONS)[:-7])
    peer.close()
    with pytest.raises(ConnectionError, match="closed while receiving"):
        future.result(timeout=10.0)


def test_driver_refuses_unavailable_required_virial() -> None:
    system = _system(periodic=True)
    peer, future = _serve(system, _provider(with_stress=False), _plan())
    with peer:
        peer.sendall(_posdata(CELL, POSITIONS) + _header("STATUS"))
        assert _read_exact(peer, 12) == _header("HAVEDATA")
        peer.sendall(_header("GETFORCE"))
        with pytest.raises(ValueError, match="virial is required"):
            future.result(timeout=10.0)
        peer.setblocking(False)
        with pytest.raises(BlockingIOError):
            peer.recv(12)


def test_optional_virial_is_transmitted_as_explicit_nan() -> None:
    system = _system(periodic=False)
    peer, future = _serve(system, _provider(with_stress=False), _plan(virial="optional"))
    zero = (
        _header("POSDATA")
        + bytes(144)
        + struct.pack("<i", 2)
        + (POSITIONS / BOHR_ANGSTROM).astype("<f8").tobytes()
    )
    with peer:
        reply = _drive_once(peer, zero, peer.sendall)
    future.result(timeout=10.0)
    assert np.all(np.isnan(reply["virial_wire"]))


def test_driver_verifies_the_transmitted_inverse_cell() -> None:
    system = _system(periodic=True)
    peer, future = _serve(system, _provider(with_stress=True), _plan())
    h = CELL.T / BOHR_ANGSTROM
    transposed_inverse = (
        _header("POSDATA")
        + h.astype("<f8").tobytes()
        + np.linalg.inv(h).T.astype("<f8").tobytes()
        + struct.pack("<i", 2)
        + (POSITIONS / BOHR_ANGSTROM).astype("<f8").tobytes()
    )
    with peer:
        peer.sendall(transposed_inverse)
        with pytest.raises(ValueError, match=r"inverse cell is not inv\(h\)"):
            future.result(timeout=10.0)


def _fake_driver(
    peer: socket.socket, energy: float, forces: np.ndarray, virial: np.ndarray
) -> dict[str, Any]:
    """Answer one server transaction with hand-built atomic-unit bytes."""

    received: dict[str, Any] = {}
    while True:
        command = _read_exact(peer, 12).decode().strip()
        if command == "STATUS":
            peer.sendall(_header("HAVEDATA" if received else "READY"))
        elif command == "POSDATA":
            received["h"] = np.frombuffer(_read_exact(peer, 72), "<f8").reshape(3, 3)
            received["ih"] = np.frombuffer(_read_exact(peer, 72), "<f8").reshape(3, 3)
            count = struct.unpack("<i", _read_exact(peer, 4))[0]
            received["positions"] = np.frombuffer(
                _read_exact(peer, 24 * count), "<f8"
            ).reshape(count, 3)
        elif command == "GETFORCE":
            peer.sendall(
                _header("FORCEREADY")
                + struct.pack("<d", energy)
                + struct.pack("<i", forces.shape[0])
                + forces.astype("<f8").tobytes()
                + virial.T.astype("<f8").tobytes()
                + struct.pack("<i", 2)
                + b"{}"
            )
            return received
        else:
            raise AssertionError(command)


def test_server_converts_wire_virial_to_native_tensile_stress() -> None:
    system = _system(periodic=True)
    server_end, peer = socket.socketpair()
    energy_au = -0.75
    forces_au = np.array([[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]])
    virial_au = np.array(
        [[0.04, 0.01, -0.02], [0.01, -0.03, 0.005], [-0.02, 0.005, 0.06]]
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fake_driver, peer, energy_au, forces_au, virial_au)
        session = phx.atomistic.interchange.IPISession(server_end, _plan())
        remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
            session, "remote"
        )
        result = remote.evaluate(system, jnp.asarray(POSITIONS), None)
        received = future.result(timeout=10.0)
    server_end.close()
    peer.close()

    np.testing.assert_allclose(received["h"], CELL.T / BOHR_ANGSTROM, rtol=1e-12)
    np.testing.assert_allclose(received["ih"] @ received["h"], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        received["positions"], POSITIONS / BOHR_ANGSTROM, rtol=1e-12
    )
    volume = abs(float(np.linalg.det(CELL)))
    assert bool(result.successful)
    assert result.stress is not None
    np.testing.assert_allclose(result.energy, energy_au * HARTREE_EV, rtol=1e-12)
    np.testing.assert_allclose(
        result.forces, forces_au * HARTREE_EV / BOHR_ANGSTROM, rtol=1e-12
    )
    np.testing.assert_allclose(
        result.stress, -virial_au * HARTREE_EV / volume, rtol=1e-12
    )


def test_server_refuses_nan_virial_when_required() -> None:
    system = _system(periodic=True)
    server_end, peer = socket.socketpair()
    nan_virial = np.full((3, 3), np.nan)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fake_driver, peer, 0.1, np.zeros((2, 3)), nan_virial)
        session = phx.atomistic.interchange.IPISession(server_end, _plan())
        remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
            session, "remote"
        )
        with pytest.raises(ValueError, match="virial is required"):
            remote.evaluate(system, jnp.asarray(POSITIONS), None)
        future.result(timeout=10.0)
    server_end.close()
    peer.close()


def test_reduced_units_have_no_atomic_unit_mapping() -> None:
    system = phx.atomistic.AtomisticSystemPlan(
        jnp.asarray([0, 1]),
        jnp.asarray([1, 1]),
        jnp.asarray([1.0, 1.0]),
        phx.atomistic.AtomisticUnitSystem.reduced(),
    ).prepare()
    server_end, peer = socket.socketpair()
    with server_end, peer:
        session = phx.atomistic.interchange.IPISession(server_end, _plan())
        remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
            session, "remote"
        )
        with pytest.raises(ValueError, match="SI-convertible"):
            remote.evaluate(system, jnp.asarray(POSITIONS), None)


def test_unix_loopback_with_ase_socket_server_reports_oracle_stress(
    tmp_path: Path,
) -> None:
    ase = pytest.importorskip("ase")
    socketio = pytest.importorskip("ase.calculators.socketio")
    system = _system(periodic=True)
    socket_name = f"phydrax-ase-{id(tmp_path)}"
    atoms = ase.Atoms("HO", positions=POSITIONS, cell=CELL, pbc=True)
    # ASE's socket server sends a transposed legacy inverse; h stays authoritative.
    plan = phx.atomistic.interchange.IPITransportPlan.unix(
        f"/tmp/ipi_{socket_name}", timeout=10.0, inverse_cell="ignore"
    )
    results: dict[str, Any] = {}

    def run_ase_server() -> None:
        with socketio.SocketIOCalculator(unixsocket=socket_name) as calculator:
            atoms.calc = calculator
            results["energy"] = atoms.get_potential_energy()
            results["forces"] = atoms.get_forces()
            results["stress"] = atoms.get_stress(voigt=False)

    server = threading.Thread(target=run_ase_server)
    server.start()
    deadline = time.monotonic() + 10.0
    while not Path(plan.address).exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with plan.connect() as session:
        status = phx.atomistic.interchange.serve_ipi_once(
            session, _provider(with_stress=True), system
        )
        server.join(timeout=10.0)
    assert status is phx.atomistic.interchange.IPITransportStatus.READY
    energy, forces, stress = _oracle(POSITIONS, CELL)
    np.testing.assert_allclose(results["energy"], energy, rtol=1e-6)
    np.testing.assert_allclose(results["forces"], forces, rtol=1e-6)
    np.testing.assert_allclose(results["stress"], stress, rtol=1e-6, atol=1e-10)
