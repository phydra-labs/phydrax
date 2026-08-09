#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx


def _cycle_graph(node_count: int) -> phx.graph.GraphIR:
    sources = jnp.arange(node_count, dtype=jnp.int32)
    targets = (sources + 1) % node_count
    return phx.graph.GraphIR(
        nodes=jnp.zeros((node_count, 1)),
        edges={"conductance": jnp.ones((2 * node_count,))},
        senders=jnp.concatenate((sources, targets)),
        receivers=jnp.concatenate((targets, sources)),
        n_node=jnp.asarray([node_count]),
        n_edge=jnp.asarray([2 * node_count]),
    )


def _timed_compiled(function, argument, repeats):
    compiled = jax.jit(function)
    started = time.perf_counter()
    output = jax.block_until_ready(compiled(argument))
    first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        output = jax.block_until_ready(compiled(argument))
    steady_seconds = (time.perf_counter() - started) / repeats
    return output, first_seconds, steady_seconds


def run_benchmarks(
    *,
    node_count: int = 64,
    rank: int = 15,
    repeats: int = 10,
) -> dict[str, Any]:
    """Benchmark preprocessing, finite features, and exact GP weight space."""
    if node_count < 4 or node_count % 2:
        raise ValueError("node_count must be an even integer of at least four.")
    if rank <= 0 or rank >= node_count or rank % 2 == 0:
        raise ValueError("rank must be a positive odd integer below node_count.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    graph = _cycle_graph(node_count)
    started = time.perf_counter()
    complex_ir = phx.graph.graph_to_cochain_complex(
        graph,
        edge_weight_key="conductance",
    )
    eigenbasis = phx.graph.cochain_laplacian_eigenbasis(
        complex_ir,
        0,
        num_modes=rank,
    )
    preprocessing_seconds = time.perf_counter() - started
    kernel = phx.kernels.AmplitudeKernel(
        phx.kernels.SpectralFeatureKernel(
            eigenbasis,
            phx.kernels.MaternSpectralMultiplier(0.7, 1.5),
        ),
        0.8,
    )
    entities = complex_ir.cell_entities(0)
    features, feature_first, feature_steady = _timed_compiled(
        lambda values: phx.kernels.kernel_features(kernel, values),
        entities,
        repeats,
    )
    gram, gram_first, gram_steady = _timed_compiled(
        lambda values: kernel.matrix(values, values),
        entities,
        repeats,
    )

    observation_entities = jnp.tile(entities, 3)
    latent = jnp.sin(2.0 * jnp.pi * entities / node_count)
    observations = jnp.tile(latent, 3)
    state = phx.uq.GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.05)
    model = phx.uq.ExactGaussianProcessDiscrepancy(
        observation_entities,
        observations,
    )
    started = time.perf_counter()
    feature_factor = model.factor(state=state)
    if not isinstance(feature_factor, phx.uq.FiniteFeatureGaussianProcessFactor):
        raise RuntimeError("The benchmark kernel did not select exact weight space.")
    jax.block_until_ready(feature_factor.correction_cholesky)
    feature_factor_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dense_factor = phx.uq.ExactGaussianProcessFactor(
        observation_entities,
        state=state,
    )
    jax.block_until_ready(dense_factor.cholesky)
    dense_factor_seconds = time.perf_counter() - started
    residual = model.residual(jnp.zeros_like(observations))
    feature_log_probability = feature_factor.log_probability(residual)
    dense_log_probability = dense_factor.log_probability(residual)
    feature_condition = feature_factor.condition(residual, entities)
    dense_condition = dense_factor.condition(residual, entities)
    log_probability_error = jnp.abs(feature_log_probability - dense_log_probability)
    mean_error = jnp.max(jnp.abs(feature_condition.mean - dense_condition.mean))
    covariance_error = jnp.max(
        jnp.abs(feature_condition.covariance - dense_condition.covariance)
    )
    finite_posterior = bool(
        jnp.all(jnp.isfinite(feature_condition.mean))
        & jnp.all(jnp.isfinite(feature_condition.covariance))
    )
    feature_selected = True
    lower_storage = (
        feature_factor.factor_storage_elements < dense_factor.factor_storage_elements
    )

    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "node_count": node_count,
        "rank": rank,
        "repeats": repeats,
        "spectrum": {
            "constructed": True,
            "exact": eigenbasis.report.exact,
            "method_id": eigenbasis.report.method_id,
            "next_eigenvalue": eigenbasis.report.next_eigenvalue,
            "preprocessing_seconds": preprocessing_seconds,
            "basis_storage_elements": int(
                eigenbasis.eigenfunctions.size + eigenbasis.eigenvalues.size
            ),
        },
        "feature_evaluation": {
            "compile_and_first_seconds": feature_first,
            "steady_seconds": feature_steady,
            "output_bytes": int(features.size * features.dtype.itemsize),
        },
        "gram_evaluation": {
            "compile_and_first_seconds": gram_first,
            "steady_seconds": gram_steady,
            "output_bytes": int(gram.size * gram.dtype.itemsize),
        },
        "gp_factorization": {
            "feature_space_selected": feature_selected,
            "feature_seconds": feature_factor_seconds,
            "dense_seconds": dense_factor_seconds,
            "feature_storage_elements": feature_factor.factor_storage_elements,
            "dense_storage_elements": dense_factor.factor_storage_elements,
            "lower_feature_storage": lower_storage,
            "log_probability_absolute_error": float(log_probability_error),
        },
        "conditioning": {
            "finite_posterior": finite_posterior,
            "mean_max_absolute_error": float(mean_error),
            "covariance_max_absolute_error": float(covariance_error),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Phydrax Laplacian spectral kernels and GP inference."
    )
    parser.add_argument("--node-count", type=int, default=64)
    parser.add_argument("--rank", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = run_benchmarks(
        node_count=8 if arguments.smoke else arguments.node_count,
        rank=5 if arguments.smoke else arguments.rank,
        repeats=1 if arguments.smoke else arguments.repeats,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
