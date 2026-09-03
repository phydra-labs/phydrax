from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import capture_environment, measure_repeated

import phydrax as phx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=256)
    parser.add_argument("--beads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.atoms <= 0 or args.beads <= 0 or args.atoms % args.beads:
        raise ValueError("atoms must be a positive multiple of beads")
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    membership = jnp.repeat(jnp.arange(args.beads), args.atoms // args.beads)
    system = phx.atomistic.AtomisticSystemPlan(
        jnp.arange(args.atoms),
        jnp.ones((args.atoms,), dtype=jnp.int32),
        jnp.linspace(1.0, 2.0, args.atoms),
        units,
        atom_type_ids=jnp.ones((args.atoms,), dtype=jnp.int32),
        molecule_ids=membership,
    ).prepare()
    mapping = phx.atomistic.MolecularCoarseMapPlan(
        10_000 + jnp.arange(args.beads),
        jnp.arange(args.beads) % 4,
        membership,
    ).prepare(system)
    positions = jax.random.normal(jax.random.key(0), (args.atoms, 3))
    forces = jax.random.normal(jax.random.key(1), (args.atoms, 3))
    compiled = jax.jit(lambda x, f: mapping.evaluate(x, forces=f))
    result, elapsed = measure_repeated(
        lambda: compiled(positions, forces),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "atoms": args.atoms,
        "beads": args.beads,
        "execution_seconds": elapsed.to_seconds_dict(),
        "successful": bool(result.successful),
        "mass_residual": float(result.mass_residual),
        "charge_residual": float(result.charge_residual),
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
