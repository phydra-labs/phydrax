#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax._trainable import partition_trainable


def _array_count(tree: Any, /) -> int:
    return sum(int(leaf.size) for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf))


def _array_bytes(tree: Any, /) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if eqx.is_array(leaf)
    )


def _resource_benchmark(
    *, dimension: int, rank: int, batch_size: int, scaling: str
) -> dict[str, Any]:
    base = phx.nn.layers.Linear(
        in_size=dimension,
        out_size=dimension,
        rwf=False,
        use_bias=False,
        key=jr.key(0),
    )
    adapted, report = phx.nn.parameters.adapt_low_rank(
        base,
        {
            ".weight": phx.nn.parameters.LowRankSpec(
                rank,
                scaling=scaling,
            )
        },
        key=jr.key(1),
    )
    subspace = phx.nn.parameters.low_rank_parameter_subspace(adapted)
    dense_parameters, dense_fixed = partition_trainable(base)
    adapter_parameters = subspace.initial
    adapter_fixed = subspace.frozen
    inputs = jr.normal(jr.key(2), (batch_size, dimension))
    targets = jr.normal(jr.key(3), (batch_size, dimension))
    dense_optimizer = optax.adam(1e-3)
    adapter_optimizer = optax.adam(1e-3)
    dense_state = dense_optimizer.init(dense_parameters)
    adapter_state = adapter_optimizer.init(adapter_parameters)

    def dense_step(parameters, state):
        def objective(candidate):
            model = eqx.combine(candidate, dense_fixed)
            return jnp.mean((model(inputs) - targets) ** 2)

        loss, gradient = eqx.filter_value_and_grad(objective)(parameters)
        updates, state = dense_optimizer.update(gradient, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss

    def adapter_step(parameters, state):
        def objective(candidate):
            model = eqx.combine(candidate, adapter_fixed)
            return jnp.mean((model(inputs) - targets) ** 2)

        loss, gradient = eqx.filter_value_and_grad(objective)(parameters)
        updates, state = adapter_optimizer.update(gradient, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss

    dense_jit = eqx.filter_jit(dense_step)
    adapter_jit = eqx.filter_jit(adapter_step)
    started = time.perf_counter()
    dense_parameters, dense_state, dense_loss = dense_jit(dense_parameters, dense_state)
    jax.block_until_ready(dense_loss)
    dense_compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    adapter_parameters, adapter_state, adapter_loss = adapter_jit(
        adapter_parameters, adapter_state
    )
    jax.block_until_ready(adapter_loss)
    adapter_compile_seconds = time.perf_counter() - started

    repeats = 5
    started = time.perf_counter()
    for _ in range(repeats):
        dense_parameters, dense_state, dense_loss = dense_jit(
            dense_parameters, dense_state
        )
    jax.block_until_ready(dense_loss)
    dense_step_seconds = (time.perf_counter() - started) / repeats
    started = time.perf_counter()
    for _ in range(repeats):
        adapter_parameters, adapter_state, adapter_loss = adapter_jit(
            adapter_parameters, adapter_state
        )
    jax.block_until_ready(adapter_loss)
    adapter_step_seconds = (time.perf_counter() - started) / repeats

    trained_adapter = eqx.combine(adapter_parameters, adapter_fixed)
    merged = phx.nn.parameters.merge_low_rank(trained_adapter)
    unmerged_output = trained_adapter(inputs)
    merged_output = merged(inputs)
    with tempfile.TemporaryDirectory() as directory:
        adapter_path = Path(directory) / "adapter.phx"
        phx.nn.parameters.save_low_rank_adapter(adapter_path, trained_adapter)
        artifact_bytes = adapter_path.stat().st_size

    return {
        "dimension": dimension,
        "rank": rank,
        "batch_size": batch_size,
        "scaling": scaling,
        "dense_parameter_count": _array_count(dense_parameters),
        "adapter_parameter_count": _array_count(adapter_parameters),
        "reported_adapter_parameter_count": report.adapter_parameter_count,
        "dense_parameter_bytes": _array_bytes(dense_parameters),
        "adapter_parameter_bytes": _array_bytes(adapter_parameters),
        "dense_optimizer_state_bytes": _array_bytes(dense_state),
        "adapter_optimizer_state_bytes": _array_bytes(adapter_state),
        "dense_compile_seconds": dense_compile_seconds,
        "adapter_compile_seconds": adapter_compile_seconds,
        "dense_step_seconds": dense_step_seconds,
        "adapter_step_seconds": adapter_step_seconds,
        "adapter_artifact_bytes": artifact_bytes,
        "merged_output_max_abs_error": float(
            jax.device_get(jnp.max(jnp.abs(unmerged_output - merged_output)))
        ),
    }


def _operator_dataset(*, coefficient: float, cases: int = 8, resolution: int = 8):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, resolution),
        quadrature_weights=jnp.full((resolution,), 1.0 / resolution),
    )
    offsets = jnp.arange(cases, dtype=float)[:, None]
    values = jnp.sin((offsets + 1.0) * jnp.pi * axis.nodes[None, :])
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"solution": coefficient * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _operator_loss(model, dataset) -> float:
    prediction = model(dataset.batch)
    target = dataset.targets.field("solution").values
    return float(jax.device_get(jnp.mean((prediction - target) ** 2)))


def _transfer_benchmark() -> dict[str, Any]:
    source = _operator_dataset(coefficient=2.0)
    target = _operator_dataset(coefficient=3.0)
    latent = 6
    branch = phx.nn.models.MLP(
        in_size=8,
        out_size=latent,
        width_size=8,
        depth=1,
        rwf=False,
        key=jr.key(10),
    )
    trunk = phx.nn.models.MLP(
        in_size="scalar",
        out_size=latent,
        width_size=8,
        depth=1,
        rwf=False,
        key=jr.key(11),
    )
    base = phx.nn.operator.architectures.DeepONet(
        branch=phx.nn.operator.architectures.FixedBranchEncoder(branch, latent),
        trunk=trunk,
        coord_dim=1,
        latent_size=latent,
        out_size="scalar",
        in_size=8,
        source_key="state",
    )
    pretrained = phx.nn.operator.training.fit_operator(
        base,
        source,
        epochs=1,
        steps=4,
        batch_size=4,
        learning_rate=2e-2,
    ).last_execution_model
    frozen_loss = _operator_loss(pretrained, target)
    full_result = phx.nn.operator.training.fit_operator(
        pretrained,
        target,
        epochs=1,
        steps=4,
        batch_size=4,
        learning_rate=2e-2,
    )
    paths = phx.nn.parameters.low_rank_sites(pretrained)
    adapted, report = phx.nn.parameters.adapt_low_rank(
        pretrained,
        {path: phx.nn.parameters.LowRankSpec(1, scaling="sqrt_rank") for path in paths},
        key=jr.key(11),
    )
    subspace = phx.nn.parameters.low_rank_parameter_subspace(adapted)
    adapter_result = phx.nn.operator.training.fit_operator(
        adapted,
        target,
        epochs=1,
        steps=4,
        batch_size=4,
        learning_rate=2e-2,
        parameter_subspace=subspace,
    )
    full_parameters, _ = partition_trainable(pretrained)
    return {
        "base_coefficient": 2.0,
        "target_coefficient": 3.0,
        "frozen_loss": frozen_loss,
        "full_finetune_loss": float(full_result.final_loss),
        "low_rank_loss": float(adapter_result.final_loss),
        "full_trainable_parameters": _array_count(full_parameters),
        "low_rank_trainable_parameters": subspace.total_dimension,
        "adapted_site_count": len(report.sites),
        "rank": 1,
        "scaling": "sqrt_rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/low_rank_adaptation.json"),
    )
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--scaling",
        choices=("rank", "sqrt_rank"),
        default="sqrt_rank",
    )
    args = parser.parse_args()
    result = {
        "resource": _resource_benchmark(
            dimension=args.dimension,
            rank=args.rank,
            batch_size=args.batch_size,
            scaling=args.scaling,
        ),
        "operator_transfer": _transfer_benchmark(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
