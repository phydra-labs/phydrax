import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax.numpy as jnp

import phydrax as phx


# i-PI exchanges Hartree atomic units, so the system needs SI-convertible units.
system = phx.atomistic.AtomisticSystemPlan(
    # ty: ignore[invalid-argument-type]
    [0, 1],
    # ty: ignore[invalid-argument-type]
    [1, 1],
    # ty: ignore[invalid-argument-type]
    [1.008, 1.008],
    phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])


def evaluator(prepared: Any, coordinate: Any, cell_vectors: Any) -> Any:
    del prepared, cell_vectors
    # A finite molecule has no cell stress; the transport declares it unavailable.
    return phx.atomistic.ExternalAtomisticEvaluation(
        jnp.sum(coordinate**2),
        -2.0 * coordinate,
        None,
        jnp.asarray(True),
        "loopback-local",
    )


provider = phx.atomistic.CallableBornOppenheimerProvider(evaluator, "loopback-local")
socket_path = os.path.join(tempfile.gettempdir(), f"phydrax-ipi-{os.getpid()}.sock")
transport = phx.atomistic.interchange.IPITransportPlan.unix(
    socket_path, timeout=5.0, virial="optional"
)
listener = transport.listen()


def serve() -> Any:
    with listener.accept() as session:
        return phx.atomistic.interchange.serve_ipi_once(session, provider, system)


with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(serve)
    with transport.connect() as session:
        remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
            session, "loopback-remote"
        )
        result = remote.evaluate(system, positions, None)
    status = future.result(timeout=10.0)
listener.close()
if status is not phx.atomistic.interchange.IPITransportStatus.READY or not bool(
    result.successful
):
    raise RuntimeError("i-PI loopback failed")
print(float(result.energy), result.forces, result.stress)
