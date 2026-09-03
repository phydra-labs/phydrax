#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import capture_environment, logical_array_bytes, measure_repeated

import phydrax as phx


def _finite_profile(repeats: int):
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    build = phx.tensor_network.build_local_term_mpo(
        (2, 2),
        (
            phx.tensor_network.FiniteLocalTerm(0, (z,)),
            phx.tensor_network.FiniteLocalTerm(1, (z,)),
        ),
    )
    state = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128)
    )
    problem = phx.solver.FiniteDMRGProblem(state, build.operator)
    policy = phx.solver.FiniteDMRGPolicy(
        maximum_bond_dimension=2,
        maximum_sweeps=2,
        eigen_policy=phx.linalg.eigen.EigenSolvePolicy(
            phx.linalg.eigen.DenseEigh(), count=1, which="smallest-algebraic"
        ),
    )
    result, timing = measure_repeated(
        lambda: phx.solver.solve_finite_dmrg(problem, policy),
        warmup=1,
        repeats=repeats,
    )
    return {
        "execution": timing.to_milliseconds_dict(),
        "input_bytes": logical_array_bytes((state, build.operator)),
        "energy": float(result.energy),
        "status": int(result.diagnostics.status),
        "projected_residual": float(
            jnp.nanmin(result.diagnostics.projected_residual_history)
        ),
        "variance": float(jnp.nanmin(result.diagnostics.energy_variance_history)),
    }


def _tensor_train_profile(repeats: int):
    dense = jnp.arange(64.0, dtype=jnp.float64).reshape((4, 4, 4))
    result, timing = measure_repeated(
        lambda: phx.tensor_train.tt_svd(dense, max_ranks=(4, 4), relative_tolerance=0.0),
        warmup=1,
        repeats=repeats,
    )
    return {
        "execution": timing.to_milliseconds_dict(),
        "input_bytes": logical_array_bytes(dense),
        "output_bytes": logical_array_bytes(result.tensor),
        "frobenius_bound": float(result.evidence.frobenius_error_bound),
        "dense_residual": float(
            jnp.linalg.norm(result.tensor.to_dense(max_entries=dense.size) - dense)
        ),
    }


def _network_profile(repeats: int):
    leg = phx.tensor_network.ContractionLeg("shared", 4)
    topology = phx.tensor_network.ContractionStructure(
        (
            phx.tensor_network.ContractionOperand("left", (leg,)),
            phx.tensor_network.ContractionOperand("middle", (leg,)),
            phx.tensor_network.ContractionOperand("right", (leg,)),
        ),
        (),
    )
    values = (
        jnp.arange(1.0, 5.0, dtype=jnp.float64),
        jnp.arange(2.0, 6.0, dtype=jnp.float64),
        jnp.arange(3.0, 7.0, dtype=jnp.float64),
    )
    prepared = phx.tensor_network.prepare_contraction(
        phx.tensor_network.plan_contraction(topology, dtype="float64"), values
    )
    result, timing = measure_repeated(
        lambda: phx.tensor_network.execute_contraction(prepared),
        warmup=1,
        repeats=repeats,
    )
    return {
        "execution": timing.to_milliseconds_dict(),
        "input_bytes": logical_array_bytes(values),
        "value": float(result.value),
        "exact": bool(result.evidence.exact),
        "replay_id": result.evidence.replay_id,
    }


def _symmetry_profile(repeats: int):
    result, timing = measure_repeated(
        lambda: phx.tensor_network.su2_recoupling_matrix(1, 1, 1, 1)[2],
        warmup=1,
        repeats=repeats,
    )
    return {
        "execution": timing.to_milliseconds_dict(),
        "unitarity_residual": float(
            jnp.linalg.norm(result @ result.T - jnp.eye(result.shape[0]))
        ),
        "pentagon_residual": float(
            phx.tensor_network.su2_pentagon_residual(1, 1, 1, 1, 0)
        ),
    }


def _quantum_profile(repeats: int):
    zero = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex64)
    one = jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)
    instrument = phx.solver.QuantumInstrument(
        jnp.stack((zero, one))[:, None, :, :],
        jnp.ones((2, 1), dtype=bool),
        tolerance=1e-5,
    )
    state = jnp.asarray([1.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)
    result, timing = measure_repeated(
        lambda: phx.solver.apply_dense_quantum_instrument(instrument, state),
        warmup=1,
        repeats=repeats,
    )
    return {
        "execution": timing.to_milliseconds_dict(),
        "probabilities": [float(value) for value in result.probabilities],
        "probability_sum_residual": float(result.probability_sum_residual),
        "valid": bool(result.valid),
    }


def _platform_profile():
    state = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    )
    support = phx.tensor_network.TensorNetworkSupportTuple(
        representation="mps",
        workflow="finite-ground-state",
        boundary="open",
        algorithm="finite-dmrg",
        backend=jax.default_backend(),
        dtype="float32",
    )
    policy = phx.tensor_network.TensorNetworkResourcePolicy(
        maximum_compile_units=1_000,
        maximum_host_bytes=1_000_000,
        maximum_device_bytes=1_000_000,
        maximum_output_queue_bytes=1_000_000,
    )
    forecast = phx.tensor_network.forecast_tensor_network_resources(
        state, support, policy
    )
    admission = phx.tensor_network.admit_tensor_network_resources(forecast, (support,))
    return {
        "forecast_id": forecast.forecast_id,
        "storage_bytes": forecast.storage_bytes,
        "admitted": admission.admitted,
        "failure": admission.failure.value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "finite_chain": _finite_profile(arguments.repeats),
        "tensor_train": _tensor_train_profile(arguments.repeats),
        "arbitrary_network": _network_profile(arguments.repeats),
        "symmetry": _symmetry_profile(arguments.repeats),
        "quantum_instrument": _quantum_profile(arguments.repeats),
        "platform_admission": _platform_profile(),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
