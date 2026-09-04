#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
import tracemalloc
import urllib.request
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax._trainable import partition_trainable


RMD17_GATES = {
    "energy_mae_per_atom": 0.05,
    "force_component_mae": 0.20,
    "energy_rotation_defect": 1e-7,
    "force_rotation_defect": 1e-7,
    "maximum_net_force": 1e-7,
    "maximum_net_torque": 1e-7,
}
MODEL_NAMES = ("painn", "nequip")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a matched finite-molecule PaiNN-versus-NequIP campaign on one "
            "local or checksum-verified cached rMD17 NPZ."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", type=Path, help="Existing local rMD17 NPZ.")
    source.add_argument("--url", help="Explicit rMD17 NPZ URL to fetch into the cache.")
    parser.add_argument(
        "--sha256", help="Required SHA-256 for --url and optional verification for --npz."
    )
    parser.add_argument("--doi", help="Dataset DOI recorded verbatim in provenance.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/phydrax/rmd17"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--train-size", type=int, default=950)
    parser.add_argument("--validation-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=100.0)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--maximum-neighbors", type=int, required=True)
    parser.add_argument("--maximum-dense-atoms", type=int, required=True)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--interactions", type=int, default=3)
    parser.add_argument("--radial-basis", type=int, default=20)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_source(arguments: argparse.Namespace, /) -> tuple[Path, str]:
    expected = None if arguments.sha256 is None else arguments.sha256.lower()
    if expected is not None and (
        len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected)
    ):
        raise ValueError("--sha256 must contain exactly 64 lowercase hexadecimal digits.")
    if arguments.url is None:
        source = arguments.npz.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
    else:
        if expected is None:
            raise ValueError(
                "--url requires --sha256; unverified benchmark fetches are forbidden."
            )
        arguments.cache_dir.mkdir(parents=True, exist_ok=True)
        source = (arguments.cache_dir / f"{expected}.npz").resolve()
        if not source.exists():
            with urllib.request.urlopen(arguments.url, timeout=120) as response:
                payload = response.read()
            observed = hashlib.sha256(payload).hexdigest()
            if observed != expected:
                raise ValueError(
                    f"Downloaded rMD17 checksum {observed} does not match {expected}."
                )
            source.write_bytes(payload)
    observed = _sha256(source)
    if expected is not None and observed != expected:
        raise ValueError(f"Local rMD17 checksum {observed} does not match {expected}.")
    return source, observed


def _fingerprint(value: Any, /) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any, /) -> Any:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            label = (
                "nan"
                if np.isnan(number)
                else ("positive_infinity" if number > 0 else "negative_infinity")
            )
            return {"kind": "nonfinite", "value": label}
        return number
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _tree_bytes(tree: Any, /) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _parameter_count(potential, /) -> int:
    trainable, _ = partition_trainable(potential)
    return sum(
        int(leaf.size)
        for leaf in jax.tree_util.tree_leaves(trainable)
        if isinstance(leaf, jax.Array)
    )


def _rotation_evidence(potential, batch, execution):
    one = phx.atomistic.AtomisticBatch(
        np.asarray(batch.atomic_numbers[:1]),
        np.asarray(batch.positions[:1]),
        np.asarray(batch.masses[:1]),
        batch.scale,
        particle_ids=np.asarray(batch.particle_ids[:1]),
        atom_mask=np.asarray(batch.atom_mask[:1]),
        structure_ids=(batch.structure_ids[0],),
        coordinate_dtype=batch.positions.dtype,
    )
    rotation = jnp.asarray(
        [[0.36, -0.48, 0.80], [0.80, 0.60, 0.00], [-0.48, 0.64, 0.60]],
        dtype=batch.positions.dtype,
    )
    rotated = phx.atomistic.AtomisticBatch(
        one.atomic_numbers,
        one.positions @ rotation.T,
        one.masses,
        one.scale,
        particle_ids=one.particle_ids,
        atom_mask=one.atom_mask,
        structure_ids=(one.structure_ids[0] + "/rotated",),
        coordinate_dtype=one.positions.dtype,
    )
    reference = phx.atomistic.energy_and_forces(potential, one, execution)
    observed = phx.atomistic.energy_and_forces(potential, rotated, execution)
    force_reference = reference.forces @ rotation.T
    return {
        "energy_rotation_defect": float(
            jnp.max(jnp.abs(observed.energy - reference.energy))
        ),
        "force_rotation_defect": float(
            jnp.max(jnp.abs(observed.forces - force_reference))
        ),
    }


def _prediction_metrics(prediction, target_energy, target_forces, atom_mask):
    atom_count = jnp.sum(atom_mask, axis=1)
    energy_residual = (prediction.energy - target_energy) / atom_count
    force_mask = jnp.broadcast_to(atom_mask[:, :, None], target_forces.shape)
    force_residual = jnp.where(force_mask, prediction.forces - target_forces, 0.0)
    component_count = jnp.sum(force_mask)
    return {
        "energy_mae_per_atom": float(jnp.mean(jnp.abs(energy_residual))),
        "energy_rmse_per_atom": float(jnp.sqrt(jnp.mean(energy_residual**2))),
        "force_component_mae": float(jnp.sum(jnp.abs(force_residual)) / component_count),
        "force_component_rmse": float(
            jnp.sqrt(jnp.sum(force_residual**2) / component_count)
        ),
        "maximum_net_force": float(jnp.max(jnp.abs(prediction.net_force))),
        "maximum_net_torque": float(jnp.max(jnp.abs(prediction.net_torque))),
        "all_predictions_valid": bool(jnp.all(prediction.valid)),
        "maximum_neighbor_count": int(jnp.max(prediction.maximum_neighbor_count)),
        "neighbor_overflow_count": int(jnp.sum(prediction.neighbor_overflow)),
    }


def _gates(metrics):
    checks = {name: metrics[name] <= threshold for name, threshold in RMD17_GATES.items()}
    checks["all_predictions_valid"] = metrics["all_predictions_valid"]
    checks["no_neighbor_overflow"] = metrics["neighbor_overflow_count"] == 0
    checks["training_success"] = metrics["training_success"]
    return {"checks": checks, "passed": all(checks.values())}


def _potential(model_name, dataset, arguments, seed):
    common = {
        "cutoff": arguments.cutoff,
        "feature_count": arguments.features,
        "interaction_count": arguments.interactions,
        "radial_basis_count": arguments.radial_basis,
        "key": jr.key(seed),
    }
    if model_name == "painn":
        return phx.nn.atomistic.PaiNNPotential(dataset.scale, **common)
    if model_name == "nequip":
        return phx.nn.atomistic.NequIPPotential(dataset.scale, **common)
    raise ValueError(f"Unknown atomistic benchmark model {model_name!r}.")


def _run_model(
    model_name,
    dataset,
    arguments,
    seed,
    problem,
    policy,
    test_batch,
    test_energy,
    test_forces,
    neighborhood_work,
):
    model = _potential(model_name, dataset, arguments, seed)
    tracemalloc.start()
    training_started = time.perf_counter()
    result = phx.atomistic.fit_atomistic_potential(
        model, problem, policy, key=jr.key(seed + 1_000_000)
    )
    jax.block_until_ready(result.final_loss)
    training_seconds = time.perf_counter() - training_started
    _, peak_host_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    potential = result.best_potential
    compiled = eqx.filter_jit(phx.atomistic.energy_and_forces)
    first_started = time.perf_counter()
    prediction = compiled(potential, test_batch, problem.graph_execution)
    jax.block_until_ready(prediction.energy)
    compile_first_seconds = time.perf_counter() - first_started
    steady_started = time.perf_counter()
    for _ in range(arguments.timing_repeats):
        prediction = compiled(potential, test_batch, problem.graph_execution)
        jax.block_until_ready(prediction.energy)
    steady_seconds = (time.perf_counter() - steady_started) / arguments.timing_repeats
    metrics = _prediction_metrics(
        prediction, test_energy, test_forces, test_batch.atom_mask
    )
    metrics.update(_rotation_evidence(potential, test_batch, problem.graph_execution))
    metrics.update(
        {
            "training_success": bool(result.successful),
            "training_status": int(result.status),
            "training_termination": result.termination,
            "training_final_loss": float(result.final_loss),
            "training_best_loss": float(result.best_loss),
            "training_seconds": training_seconds,
            "compile_first_prediction_seconds": compile_first_seconds,
            "steady_prediction_seconds": steady_seconds,
            "peak_tracemalloc_host_bytes": int(peak_host_bytes),
            "model_tree_bytes": _tree_bytes(potential),
            "prediction_tree_bytes": _tree_bytes(prediction),
            "parameter_count": _parameter_count(potential),
            **neighborhood_work,
        }
    )
    model_evidence = {
        "potential_id": potential.potential_id,
        "method_id": potential.method_id,
    }
    if model_name == "nequip":
        model_evidence.update(
            {
                "maximum_degree": potential.configuration.maximum_degree,
                "tensor_product_plan_ids": list(
                    potential.configuration.tensor_product_plan_ids
                ),
                "tensor_product_parameter_count_per_interaction": [
                    interaction.tensor_product.plan.parameter_count
                    for interaction in potential.interactions
                ],
            }
        )
    return {
        "training_result_id": result.result_id,
        "normalization_id": result.normalization.normalization_id,
        "model_evidence": model_evidence,
        "metrics": metrics,
        "gates": _gates(metrics),
    }


def _run_seed(dataset, arguments, seed):
    split = phx.atomistic.split_rmd17(
        dataset,
        train_size=arguments.train_size,
        validation_size=arguments.validation_size,
        test_size=arguments.test_size,
        seed=seed,
    )
    train_batch, train_energy, train_forces = dataset.take(split.train_indices)
    validation_batch, validation_energy, validation_forces = dataset.take(
        split.validation_indices
    )
    test_batch, test_energy, test_forces = dataset.take(split.test_indices)
    execution = phx.atomistic.AtomisticGraphExecutionPlan(
        arguments.maximum_neighbors,
        maximum_dense_atoms=arguments.maximum_dense_atoms,
    )
    problem = phx.atomistic.AtomisticTrainingProblem(
        train_batch,
        execution,
        training_energy=train_energy,
        training_forces=train_forces,
        validation_batch=validation_batch,
        validation_energy=validation_energy,
        validation_forces=validation_forces,
    )
    policy = phx.atomistic.AtomisticTrainingPolicy(
        maximum_steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        energy_weight=arguments.energy_weight,
        force_weight=arguments.force_weight,
    )
    graph = phx.atomistic.realize_atomistic_graph(
        test_batch,
        execution,
        cutoff=arguments.cutoff,
    )
    candidate_per_case = test_batch.atom_capacity * (test_batch.atom_capacity - 1)
    active_total = int(jnp.sum(graph.graph.edge_mask))
    neighborhood_work = {
        "dense_candidate_directed_edges_per_case": int(candidate_per_case),
        "dense_candidate_directed_edges_total": int(
            test_batch.case_count * candidate_per_case
        ),
        "active_directed_edges_total": active_total,
        "mean_active_directed_edges_per_case": active_total / test_batch.case_count,
    }
    models = {
        model_name: _run_model(
            model_name,
            dataset,
            arguments,
            seed,
            problem,
            policy,
            test_batch,
            test_energy,
            test_forces,
            neighborhood_work,
        )
        for model_name in MODEL_NAMES
    }
    paired_names = (
        "energy_mae_per_atom",
        "force_component_mae",
        "training_seconds",
        "steady_prediction_seconds",
        "peak_tracemalloc_host_bytes",
        "parameter_count",
    )
    paired_delta = {
        name: models["nequip"]["metrics"][name] - models["painn"]["metrics"][name]
        for name in paired_names
    }
    return {
        "seed": seed,
        "split_id": split.split_id,
        "split_sizes": {
            "train": int(split.train_indices.size),
            "validation": int(split.validation_indices.size),
            "test": int(split.test_indices.size),
        },
        "models": models,
        "paired_nequip_minus_painn": paired_delta,
    }


def _aggregate(values):
    array = np.asarray(tuple(values), dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _summary(records):
    metric_names = (
        "energy_mae_per_atom",
        "energy_rmse_per_atom",
        "force_component_mae",
        "force_component_rmse",
        "energy_rotation_defect",
        "force_rotation_defect",
        "training_seconds",
        "steady_prediction_seconds",
        "peak_tracemalloc_host_bytes",
        "parameter_count",
    )
    model_summaries = {}
    for model_name in MODEL_NAMES:
        model_summaries[model_name] = {
            "all_gates_passed": all(
                record["models"][model_name]["gates"]["passed"] for record in records
            ),
            "aggregates": {
                name: _aggregate(
                    record["models"][model_name]["metrics"][name] for record in records
                )
                for name in metric_names
            },
        }
    paired_names = tuple(records[0]["paired_nequip_minus_painn"])
    return {
        "seed_count": len(records),
        "models": model_summaries,
        "paired_nequip_minus_painn": {
            name: _aggregate(
                record["paired_nequip_minus_painn"][name] for record in records
            )
            for name in paired_names
        },
    }


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.smoke:
        arguments.seeds = arguments.seeds[:1]
        arguments.train_size = min(arguments.train_size, 4)
        arguments.validation_size = min(arguments.validation_size, 2)
        arguments.test_size = min(arguments.test_size, 2)
        arguments.steps = min(arguments.steps, 2)
        arguments.timing_repeats = 1
        arguments.features = min(arguments.features, 2)
        arguments.interactions = min(arguments.interactions, 1)
        arguments.radial_basis = min(arguments.radial_basis, 3)
    if arguments.timing_repeats <= 0:
        raise ValueError("--timing-repeats must be positive.")
    source, source_sha256 = _resolved_source(arguments)
    dataset = phx.atomistic.load_rmd17_npz(source)
    configuration = {
        "campaign": "matched-painn-versus-nequip",
        "models": list(MODEL_NAMES),
        "dataset_id": dataset.dataset_id,
        "source_sha256": source_sha256,
        "source_url": arguments.url,
        "doi": arguments.doi,
        "seeds": list(arguments.seeds),
        "train_size": arguments.train_size,
        "validation_size": arguments.validation_size,
        "test_size": arguments.test_size,
        "steps": arguments.steps,
        "learning_rate": arguments.learning_rate,
        "energy_weight": arguments.energy_weight,
        "force_weight": arguments.force_weight,
        "cutoff": arguments.cutoff,
        "maximum_neighbors": arguments.maximum_neighbors,
        "maximum_dense_atoms": arguments.maximum_dense_atoms,
        "features": arguments.features,
        "interactions": arguments.interactions,
        "radial_basis": arguments.radial_basis,
        "timing_repeats": arguments.timing_repeats,
        "gates": RMD17_GATES,
    }
    records = [_run_seed(dataset, arguments, seed) for seed in arguments.seeds]
    payload = {
        "benchmark": "finite-nonperiodic-rmd17-painn-versus-nequip",
        "fingerprint": _fingerprint(configuration),
        "configuration": configuration,
        "records": records,
        "summary": _summary(records),
        "provenance": {
            "phydrax_version": importlib.metadata.version("phydrax"),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "dataset_scale": dataset.scale.to_dict(),
            "source_length_unit": dataset.source_length_unit.to_dict(),
            "source_energy_unit": dataset.source_energy_unit.to_dict(),
            "source_mass_unit": dataset.source_mass_unit.to_dict(),
            "avogadro_constant_set_id": dataset.avogadro_constant_set_id,
            "local_npz": str(source),
            "network_fetch_used": arguments.url is not None,
            "matched_split_training_policy_and_capacity": True,
        },
    }
    print(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
