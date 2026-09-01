import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp

import phydrax as phx


system = phx.atomistic.AtomisticSystemPlan(
    [0, 1], [1, 1], [1.0, 1.0], phx.atomistic.AtomisticUnitSystem.reduced()
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])


def evaluator(prepared, coordinate, cell_vectors):
    del prepared, cell_vectors
    return phx.atomistic.ExternalAtomisticEvaluation(
        jnp.sum(coordinate**2),
        -2.0 * coordinate,
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        "loopback-local",
    )


provider = phx.atomistic.CallableBornOppenheimerProvider(evaluator, "loopback-local")
socket_path = os.path.join(tempfile.gettempdir(), f"phydrax-ipi-{os.getpid()}.sock")
transport = phx.atomistic.interchange.IPITransportPlan.unix(socket_path, timeout=5.0)
listener = transport.listen()


def serve():
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
print(float(result.energy), result.forces)
