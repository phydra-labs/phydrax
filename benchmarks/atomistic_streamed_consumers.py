# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Matched native learned-potential E/F scaling before and after streamed cutover."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from functools import partial
from operator import methodcaller
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.atomistic import (
    AbstractAtomisticPotential,
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
)
from phydrax.nn.atomistic import NequIPPotential, PaiNNPotential
from phydrax.units import ANGSTROM, ELECTRONVOLT


def _evaluate(
    potential: AbstractAtomisticPotential,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    positions: Array,
) -> tuple[Array, Array]:
    def total(coordinates: Array) -> tuple[Array, Array]:
        energy, _, _ = potential._energy_unchecked(batch, coordinates, execution)
        return jnp.sum(energy), energy

    (_, energy), gradient = jax.value_and_grad(total, has_aux=True)(positions)
    return energy, -gradient


_COMPILED_EVALUATE = eqx.filter_jit(_evaluate)


def _case(atom_count: int, features: int, repeats: int, /) -> dict[str, Any]:
    scale = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
    index = np.arange(atom_count, dtype=np.int32)
    positions = (
        np.stack((index % 4, (index // 4) % 4, index // 16), axis=-1).astype(np.float64)
        * 0.8
    )
    structure = AtomicStructure(
        np.full((atom_count,), 1, dtype=np.int32),
        positions,
        np.ones((atom_count,), dtype=np.float64),
        scale,
    )
    batch = AtomisticBatch.from_structure(structure)
    execution = AtomisticGraphExecutionPlan(
        atom_count - 1, maximum_dense_atoms=atom_count
    )
    records: dict[str, Any] = {}
    for name, family in (("painn", PaiNNPotential), ("nequip", NequIPPotential)):
        potential = family(
            scale,
            cutoff=2.5,
            feature_count=features,
            interaction_count=2,
            radial_basis_count=6,
            key=jax.random.key(7),
        )
        arguments = (potential, batch, execution, batch.positions)
        # Equinox's annotation erases its documented compiled .lower interface.
        lowering = _COMPILED_EVALUATE.lower  # ty: ignore[unresolved-attribute]
        executable, phases = measure_lower_and_compile(
            partial(lowering, *arguments), methodcaller("compile")
        )
        invocation = partial(executable, *arguments)
        result, first = measure_synchronized(invocation)
        _, warm = measure_repeated(invocation, warmup=1, repeats=repeats)
        evidence = compiler_evidence(
            executable.compiled.cost_analysis(),
            executable.compiled.memory_analysis(),
            source="xla",
        )
        energy, forces = result
        if not bool(jnp.all(jnp.isfinite(energy))) or not bool(
            jnp.all(jnp.isfinite(forces))
        ):
            raise RuntimeError(f"{name} E/F evaluation is nonfinite.")
        records[name] = {
            "architecture_id": potential.architecture_id,
            "energy": np.asarray(energy).tolist(),
            "forces": np.asarray(forces).tolist(),
            "phases": asdict(phases),
            "first_execution_seconds": first,
            "warm": warm.to_dict(unit="seconds"),
            "compiler": asdict(evidence),
            "logical_retained_bytes": logical_array_bytes(arguments),
        }
    return {"atoms": atom_count, "features": features, "models": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(count < 2 for count in args.atoms) or args.features < 1 or args.repeats < 1:
        raise ValueError(
            "Atom capacities must be at least two; features/repeats must be positive."
        )
    root = Path(__file__).resolve().parents[1]
    owners = (
        "phydrax/nn/atomistic/_painn.py",
        "phydrax/nn/atomistic/_nequip.py",
        "phydrax/atomistic/_graph.py",
    )
    record = {
        "benchmark": "matched-atomistic-streamed-consumers",
        "label": args.label,
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root, Path(__file__), ("cases", "compiler", "environment", "source_sha256")
        ).to_dict(),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in owners
        },
        "cases": [_case(count, args.features, args.repeats) for count in args.atoms],
    }
    write_json_atomic(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
