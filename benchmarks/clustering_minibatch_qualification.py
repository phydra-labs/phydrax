# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from benchmarks._clustering_reference import (
    EMAConfig,
    EMAResult,
    Policy,
    reference_ema_kmeans,
    resolve_alpha0,
    resolve_batch_size,
    Rule,
    Split,
    test_mse,
)
from benchmarks._io import write_json_atomic
from benchmarks._runtime import capture_environment
from phydrax.ml import MLBatch
from phydrax.ml.clustering import HardClusterModel, KMeans, MiniBatchKMeans


type Array = NDArray[np.floating]


class Cell(TypedDict):
    regime: str
    n: int
    k: int
    d: int
    contrast: float
    epochs: float
    runs: dict[str, list[dict[str, float]]]
    arms: dict[str, dict[str, float]]


class StopEvidence(TypedDict):
    regime: str
    contrast: float
    epochs: float
    differences: dict[str, float]


@dataclass(frozen=True, slots=True)
class Regime:
    name: str
    n: int
    k: int
    d: int
    contrast: float = 0.0


def regimes(quick: bool) -> list[Regime]:
    specs = [Regime("E1", n, k, 96) for n in (100_000, 1_000_000) for k in (1024, 4096)]
    specs += [Regime(name, 100_000, k, 16) for name in ("E2", "S1") for k in (256, 1024)]
    specs += [
        Regime(name, 200_000, 1024, d, contrast)
        for name, d in (("P1", 2), ("P2", 3))
        for contrast in (10, 100, 10000)
    ]
    specs += [Regime("P3", 200_000, 1024, 2)]
    if quick:
        return [Regime(s.name, 512, min(s.k, 16), s.d, s.contrast) for s in specs]
    return specs


def _mixture(
    rng: np.random.Generator,
    count: int,
    centers: Array,
    scales: Array,
    probabilities: Array,
) -> Array:
    indices = rng.choice(len(centers), size=count, p=probabilities)
    return centers[indices] + rng.normal(size=(count, centers.shape[1])) * scales[indices]


def generate_data(spec: Regime, test_count: int = 20_000) -> tuple[Array, Array]:
    seed = {"E1": 101, "E2": 102, "S1": 102, "P1": 103, "P2": 104, "P3": 105}[spec.name]
    rng = np.random.Generator(np.random.PCG64(seed))
    count = spec.n + test_count
    if spec.name in ("E1", "E2", "S1"):
        intrinsic = 16
        clusters = 2048 if spec.name == "E1" else 512
        probs = np.arange(1, clusters + 1, dtype=np.float64) ** -1.1
        probs /= np.sum(probs)
        centers = rng.normal(size=(clusters, intrinsic)) * 3
        scales = np.exp(
            rng.uniform(math.log(0.1), math.log(1.0), size=(clusters, intrinsic))
        )
        latent = _mixture(rng, count, centers, scales, probs)
        if spec.name == "E1":
            basis, _ = np.linalg.qr(rng.normal(size=(96, intrinsic)))
            data = (latent @ basis.T + 0.02 * rng.normal(size=(count, 96))).astype(
                np.float32
            )
        else:
            data = latent
            if spec.name == "S1":
                data += 1000 * np.std(data[: spec.n], axis=0)
    elif spec.name in ("P1", "P2"):
        chunks = []
        remaining = count
        while remaining:
            proposal = rng.uniform(size=(max(remaining * 2, 1024), spec.d))
            density = 1 + (spec.contrast - 1) * np.exp(
                -np.sum((proposal - 0.5) ** 2, axis=1) / (2 * 0.1**2)
            )
            accepted = proposal[
                rng.uniform(size=len(proposal)) * spec.contrast < density
            ][:remaining]
            chunks.append(accepted)
            remaining -= len(accepted)
        data = np.concatenate(chunks)
    elif spec.name == "P3":
        centers = rng.uniform(size=(200, 2))
        widths = np.exp(rng.uniform(math.log(1e-3), math.log(3e-2), size=(200, 1)))
        probs = np.arange(1, 201, dtype=np.float64) ** -1.1
        probs /= np.sum(probs)
        data = _mixture(rng, count, centers, widths, probs)
        background = rng.uniform(size=count) < 0.1
        data[background] = rng.uniform(size=(np.count_nonzero(background), 2))
    else:
        raise ValueError(f"Unknown regime {spec.name}")
    return data[: spec.n], data[spec.n :]


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    rule: Rule
    policy: Policy
    split: Split = "radial"
    centered: bool = True


def _arms() -> list[Arm]:
    rules: tuple[Rule, ...] = ("mass", "distortion", "removal")
    # "frozen" stops revival in the closing pass; "restart" revives through it
    # and closes rewritten slots over their post-split rows; "settle" freezes
    # the closing pass after a leading pass and restarts when it is the only one.
    policies: tuple[Policy, ...] = ("mix", "frozen", "restart", "settle")
    return [
        Arm(f"{rule}_{policy}", rule, policy) for rule in rules for policy in policies
    ] + [Arm("off", "off", "mix"), Arm("paper", "mass", "mix", "multiplicative", False)]


def _run(
    x: Array, heldout: Array, spec: Regime, epochs: float, seed: int, arm: Arm
) -> dict[str, float]:
    result = reference_ema_kmeans(
        x,
        EMAConfig(
            k=spec.k,
            epochs=epochs,
            seed=seed,
            rule=arm.rule,
            policy=arm.policy,
            split=arm.split,
            centered=arm.centered,
        ),
    )
    return _metrics(result, test_mse(heldout, result.centers))


def _metrics(result: EMAResult, mse: float) -> dict[str, float]:
    return {
        "test_mse": mse,
        "closing_objective": result.closing_objective,
        "dead_count": float(result.dead_count),
        "revivals": float(result.revivals),
        "imbalance": result.imbalance,
        "closing_gain": result.closing_gain,
        "wall_seconds": result.wall_seconds,
        "batch_size": float(result.batch_size),
        "alpha0": result.alpha0,
        "updates": float(result.updates),
    }


def _summarize(cells: list[Cell], names: list[str]) -> dict[str, dict[str, float]]:
    aggregate = {name: [] for name in names}
    for cell in cells:
        runs = cell["runs"]
        seeds = len(runs[names[0]])
        minima = [min(runs[name][i]["test_mse"] for name in names) for i in range(seeds)]
        medians = {}
        for name in names:
            values = runs[name]
            regret = [
                ((values[i]["test_mse"] - minima[i]) / minima[i]) if minima[i] else 0.0
                for i in range(seeds)
            ]
            median = float(np.median(regret))
            aggregate[name].append(median)
            metrics = {
                key: float(np.median([v[key] for v in values])) for key in values[0]
            }
            metrics["median_regret"] = median
            medians[name] = metrics
        cell["arms"] = medians
    return {
        name: {"worst": max(values), "mean": float(np.mean(values))}
        for name, values in aggregate.items()
    }


type _ArmTask = tuple[Regime, float, Arm, int, bool]

# Workers run single-threaded BLAS; the parallelism is across independent runs.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _arm_runs(task: _ArmTask) -> list[dict[str, float]]:
    spec, epochs, arm, seeds, quick = task
    x, heldout = generate_data(spec, 512 if quick else 20_000)
    return [_run(x, heldout, spec, epochs, seed, arm) for seed in range(seeds)]


def _map_runs(tasks: list[_ArmTask], workers: int) -> list[list[dict[str, float]]]:
    """Run every task; results keep task order whatever the worker schedule."""
    if workers == 1:
        return [_arm_runs(task) for task in tasks]
    for name in _THREAD_VARIABLES:
        os.environ[name] = "1"
    # Longest first: work grows with samples, clusters, features and passes.
    order = sorted(
        range(len(tasks)),
        key=lambda i: -tasks[i][0].n * tasks[i][0].k * tasks[i][0].d * tasks[i][1],
    )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        results = list(pool.map(_arm_runs, [tasks[i] for i in order]))
    ordered: list[list[dict[str, float]]] = [[] for _ in tasks]
    for position, index in enumerate(order):
        ordered[index] = results[position]
    return ordered


def run_g1(quick: bool, workers: int) -> dict[str, object]:
    arms = _arms()
    epochs_values = (1.0, 2.0) if quick else (1.0, 2.0, 5.0)
    seeds = 2 if quick else 5
    grid = [(spec, epochs) for spec in regimes(quick) for epochs in epochs_values]
    runs = _map_runs(
        [(spec, epochs, arm, seeds, quick) for spec, epochs in grid for arm in arms],
        workers,
    )
    cells: list[Cell] = []
    for index, (spec, epochs) in enumerate(grid):
        cell_runs = runs[index * len(arms) : (index + 1) * len(arms)]
        cells.append(
            {
                "regime": spec.name,
                "n": spec.n,
                "k": spec.k,
                "d": spec.d,
                "contrast": spec.contrast,
                "epochs": epochs,
                "runs": {arm.name: run for arm, run in zip(arms, cell_runs)},
                "arms": {},
            }
        )
    names = [arm.name for arm in arms]
    stage_a = _summarize(cells, names)
    candidates = arms[:12]
    best = min(stage_a[a.name]["worst"] for a in candidates)
    tied = [a for a in candidates if stage_a[a.name]["worst"] <= best + 0.0005]
    selected = min(
        tied, key=lambda a: (a.policy == "mix", stage_a[a.name]["mean"], a.name)
    )
    evidence: list[StopEvidence] = []
    for cell in cells:
        if cell["d"] <= 3:
            summaries = cell["arms"]
            differences = {
                rule: summaries[f"{rule}_{selected.policy}"]["median_regret"]
                - summaries["off"]["median_regret"]
                for rule in ("mass", "distortion", "removal")
            }
            evidence.append(
                {
                    "regime": cell["regime"],
                    "contrast": cell["contrast"],
                    "epochs": cell["epochs"],
                    "differences": differences,
                }
            )
    stop = all(all(v > 0.001 for v in item["differences"].values()) for item in evidence)
    multiplicative = Arm(
        "selected_multiplicative", selected.rule, selected.policy, "multiplicative"
    )
    second = _map_runs(
        [(spec, epochs, multiplicative, seeds, quick) for spec, epochs in grid],
        workers,
    )
    for cell, run in zip(cells, second):
        cell["runs"][multiplicative.name] = run
    stage_b = _summarize(cells, names + [multiplicative.name])
    unshifted = [cell for cell in cells if cell["regime"] != "S1"]
    radial = [cell["arms"][selected.name]["median_regret"] for cell in unshifted]
    product = [cell["arms"][multiplicative.name]["median_regret"] for cell in unshifted]
    # The origin-free radial split is kept unless it costs accuracy by both the
    # worst-case and the mean median regret; one noisy cell cannot overturn it.
    split = (
        "multiplicative"
        if max(product) < max(radial) - 0.0005
        and float(np.mean(product)) < float(np.mean(radial)) - 0.0005
        else "radial"
    )
    return {
        "environment": capture_environment().to_dict(),
        "quick": quick,
        "cells": cells,
        "decision": {
            "rule": selected.rule,
            "policy": selected.policy,
            "split": split,
            "stage_a_summary": stage_a,
            "stage_b_summary": stage_b,
            "tied_candidates": [a.name for a in tied],
            "tie_threshold": 0.0005,
            "stop_for_user": stop,
            "stop_evidence": evidence,
            "unshifted_radial_worst": max(radial),
            "unshifted_multiplicative_worst": max(product),
            "unshifted_radial_mean": float(np.mean(radial)),
            "unshifted_multiplicative_mean": float(np.mean(product)),
        },
    }


class PackageRuns(TypedDict):
    test_mse: list[float]
    closing_gain: list[float]
    dead_count: list[float]
    revivals: list[float]
    batch_size: int
    initial_step: list[float]


_SEEDS = 5
_TOLERANCE_CANDIDATES = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def _package_regimes(quick: bool) -> list[Regime]:
    """Moderate-size regimes the package gates share."""
    specs = [
        Regime("E1", 100_000, 1024, 96),
        Regime("E1", 100_000, 4096, 96),
        Regime("E2", 100_000, 256, 16),
        Regime("E2", 100_000, 1024, 16),
        Regime("P1", 200_000, 1024, 2, 100.0),
        Regime("P2", 200_000, 1024, 3, 100.0),
        Regime("P3", 200_000, 1024, 2),
    ]
    if quick:
        return [Regime(s.name, 2048, min(s.k, 32), s.d, s.contrast) for s in specs]
    return specs


def _hard_model(result: Any) -> HardClusterModel:
    model = result.as_trainable()
    if not isinstance(model, HardClusterModel):
        raise TypeError("Clustering fits must produce a HardClusterModel.")
    return model


def _replicated(x: Array, seeds: int) -> jax.Array:
    """One case per seed: every case reads the same data with its own random stream."""
    return jnp.broadcast_to(jnp.asarray(x), (seeds,) + x.shape)


def package_runs(
    x: Array,
    heldout: Array,
    k: int,
    epochs: float,
    seeds: int,
    *,
    batch_size: int | None = None,
    initial_step: float | None = None,
) -> PackageRuns:
    recipe = MiniBatchKMeans(
        k, epochs=epochs, batch_size=batch_size, initial_step=initial_step
    )

    def fit(values: jax.Array) -> tuple[jax.Array, ...]:
        result = recipe.fit_batch(MLBatch(values), key=jax.random.key(0))
        evidence = result.diagnostics.minibatch
        return (
            _hard_model(result).centers,
            evidence.closing_gain,
            result.diagnostics.cluster_mass,
            evidence.revivals,
            evidence.initial_step,
        )

    centers, gain, mass, revivals, step = (
        np.asarray(value) for value in jax.jit(fit)(_replicated(x, seeds))
    )
    return {
        "test_mse": [test_mse(heldout, centers[seed]) for seed in range(seeds)],
        "closing_gain": [float(value) for value in gain],
        "dead_count": [float(np.sum(row == 0.0)) for row in mass],
        "revivals": [float(value) for value in revivals],
        "batch_size": resolve_batch_size(x.shape[0], k, epochs, "quality", batch_size),
        "initial_step": [float(value) for value in step],
    }


def lloyd_test_mse(
    x: Array, heldout: Array, k: int, iterations: int, seeds: int
) -> list[float]:
    recipe = KMeans(
        k,
        max_iterations=iterations,
        tolerance=0.0,
        initialization="random",
        empty_policy="retain",
    )

    def fit(values: jax.Array) -> jax.Array:
        result = recipe.fit_batch(MLBatch(values), key=jax.random.key(0))
        return _hard_model(result).centers

    centers = np.asarray(jax.jit(fit)(_replicated(x, seeds)))
    return [test_mse(heldout, centers[seed]) for seed in range(seeds)]


def _median(values: list[float]) -> float:
    return float(np.median(values))


def run_g2(quick: bool) -> dict[str, object]:
    """Calibrate the default convergence tolerance from closing gains."""
    epochs_values = (0.5, 1.0, 2.0, 5.0)
    seeds = 2 if quick else _SEEDS
    gains: dict[float, list[float]] = {epochs: [] for epochs in epochs_values}
    cells = []
    for spec in _package_regimes(quick):
        x, heldout = generate_data(spec, 512 if quick else 20_000)
        for epochs in epochs_values:
            runs = package_runs(x, heldout, spec.k, epochs, seeds)
            gains[epochs].extend(runs["closing_gain"])
            cells.append({"regime": spec.name, "k": spec.k, "epochs": epochs, **runs})
    percentile = float(np.percentile(gains[2.0], 99))
    tolerance = next(
        (value for value in _TOLERANCE_CANDIDATES if value >= 2.0 * percentile),
        _TOLERANCE_CANDIDATES[-1],
    )
    flagged = {
        str(epochs): float(np.mean(np.asarray(gains[epochs]) > tolerance))
        for epochs in epochs_values
    }
    return {
        "cells": cells,
        "decision": {
            "two_epoch_gain_p99": percentile,
            "tolerance": tolerance,
            "covered": tolerance >= 2.0 * percentile,
            "flagged_fraction_by_epochs": flagged,
        },
    }


def run_g3(quick: bool) -> dict[str, object]:
    """Regret of the law-chosen batch size and starting step against a local grid."""
    seeds = 2 if quick else _SEEDS
    batch_scales = (0.25, 1.0, 4.0)
    step_scales = (0.25, 0.5, 1.0, 2.0, 4.0)
    cells = []
    regrets = []
    for spec in _package_regimes(quick)[:4]:
        x, heldout = generate_data(spec, 512 if quick else 20_000)
        n = x.shape[0]
        for epochs in (2.0, 5.0):
            law_batch = resolve_batch_size(n, spec.k, epochs, "quality")
            grid: dict[str, float] = {}
            for batch_scale in batch_scales:
                batch = max(1, min(n, round(law_batch * batch_scale)))
                law_step = resolve_alpha0(n, spec.k, epochs, batch)
                for step_scale in step_scales:
                    step = min(1.0, law_step * step_scale)
                    runs = package_runs(
                        x,
                        heldout,
                        spec.k,
                        epochs,
                        seeds,
                        batch_size=batch,
                        initial_step=step,
                    )
                    grid[f"{batch_scale}x{step_scale}"] = _median(runs["test_mse"])
            best = min(grid.values())
            regret = (grid["1.0x1.0"] - best) / best
            regrets.append(regret)
            cells.append(
                {
                    "regime": spec.name,
                    "k": spec.k,
                    "epochs": epochs,
                    "law_batch_size": law_batch,
                    "median_test_mse": grid,
                    "law_regret": regret,
                }
            )
    median_regret = float(np.median(regrets))
    return {
        "cells": cells,
        "decision": {
            "median_law_regret": median_regret,
            "passed": median_regret <= 0.005,
        },
    }


_AGREEMENT_SEEDS = 20
# Standard error of a sample median, in units of sigma / sqrt(n), for normal data.
_MEDIAN_ERROR = math.sqrt(math.pi / 2.0)


def run_g4(quick: bool, workers: int, g1: dict[str, Any]) -> dict[str, object]:
    """Package against the independent reference with the selected revival design.

    The two implementations draw different random streams, so agreement is
    statistical: on the same data and seed count, the median held-out errors
    must differ by at most three standard errors of the difference of medians
    and the median dead-cluster counts by at most max(2, 10%).
    """
    seeds = 4 if quick else _AGREEMENT_SEEDS
    decision = g1["decision"]
    arm = Arm("reference", decision["rule"], decision["policy"], decision["split"])
    specs = [Regime("E1", 100_000, 1024, 96), Regime("P1", 200_000, 1024, 2, 100.0)]
    if quick:
        specs = [Regime(s.name, 2048, 32, s.d, s.contrast) for s in specs]
    grid = [(spec, epochs) for spec in specs for epochs in (1.0, 2.0, 5.0)]
    references = _map_runs(
        [(spec, epochs, arm, seeds, quick) for spec, epochs in grid], workers
    )
    rows = []
    passed = True
    for (spec, epochs), reference in zip(grid, references):
        x, heldout = generate_data(spec, 512 if quick else 20_000)
        runs = package_runs(x, heldout, spec.k, epochs, seeds)
        package_mse = np.asarray(runs["test_mse"])
        reference_mse = np.asarray([run["test_mse"] for run in reference])
        difference = float(np.median(package_mse) - np.median(reference_mse))
        error = _MEDIAN_ERROR * math.sqrt(
            (np.var(package_mse, ddof=1) + np.var(reference_mse, ddof=1)) / seeds
        )
        package_dead = _median(runs["dead_count"])
        reference_dead = _median([run["dead_count"] for run in reference])
        dead_ok = abs(package_dead - reference_dead) <= max(2.0, 0.1 * reference_dead)
        row_passed = abs(difference) <= 3.0 * error and dead_ok
        passed = passed and row_passed
        rows.append(
            {
                "regime": spec.name,
                "k": spec.k,
                "contrast": spec.contrast,
                "epochs": epochs,
                "seeds": seeds,
                "package_median_test_mse": float(np.median(package_mse)),
                "reference_median_test_mse": float(np.median(reference_mse)),
                "relative_difference": difference / float(np.median(reference_mse)),
                "difference_standard_errors": abs(difference) / error,
                "package_median_dead": package_dead,
                "reference_median_dead": reference_dead,
                "passed": row_passed,
            }
        )
    return {"rows": rows, "decision": {"passed": passed}}


def run_g5(quick: bool) -> dict[str, object]:
    """Epochs the package needs to reach the test error of 25 Lloyd iterations."""
    seeds = 2 if quick else _SEEDS
    epochs_values = (1.0, 1.5, 2.0, 3.0)
    rows = []
    passed = True
    for spec in _package_regimes(quick)[:4]:
        x, heldout = generate_data(spec, 512 if quick else 20_000)
        lloyd = _median(lloyd_test_mse(x, heldout, spec.k, 25, seeds))
        package = {
            str(epochs): _median(
                package_runs(x, heldout, spec.k, epochs, seeds)["test_mse"]
            )
            for epochs in epochs_values
        }
        reached = next(
            (epochs for epochs in epochs_values if package[str(epochs)] <= lloyd), None
        )
        row_passed = reached is not None and reached <= 3.0
        passed = passed and row_passed
        rows.append(
            {
                "regime": spec.name,
                "k": spec.k,
                "lloyd_25_median_test_mse": lloyd,
                "package_median_test_mse": package,
                "epochs_to_reach_lloyd_25": reached,
                "passed": row_passed,
            }
        )
    return {"rows": rows, "decision": {"passed": passed}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=("g1", "g2", "g3", "g4", "g5"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes for the independent reference runs of g1 and g4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/clustering_minibatch_qualification.json"),
    )
    args = parser.parse_args()
    path: Path = args.output
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    runners: dict[str, Callable[[], dict[str, object]]] = {
        "g1": lambda: run_g1(args.quick, args.workers),
        "g2": lambda: run_g2(args.quick),
        "g3": lambda: run_g3(args.quick),
        "g4": lambda: run_g4(args.quick, args.workers, existing["g1"]),
        "g5": lambda: run_g5(args.quick),
    }
    section = runners[args.gate]()
    section["environment"] = capture_environment().to_dict()
    existing[args.gate] = section
    write_json_atomic(path, existing)
    print(json.dumps(section["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
