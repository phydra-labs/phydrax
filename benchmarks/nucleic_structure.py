"""Sparse/dense published eRMSD execution on declared geometric fixtures.

Synthetic rings benchmark descriptor mathematics, not native RNA structure or
force-field accuracy. No external implementation/table data are copied.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

from phydrax.applications.nucleic_acid_biophysics import (
    NucleicAcidConstruct,
    NucleotideAtomMapping,
    prepare_nucleotide_binding,
)
from phydrax.applications.nucleic_acid_biophysics.structure import (
    base_frames,
    NucleotideGDescriptor,
    NucleotideTorsionProgram,
    sugar_pseudorotation,
)
from phydrax.atomistic import AtomisticSystemPlan, AtomisticUnitSystem
from phydrax.units import ANGSTROM


def run(length, repeats, smooth_width):
    construct = NucleicAcidConstruct(("rna",), ("A" * length,), ("RNA",), (False,))
    keys = construct.nucleotide_keys
    ids = tuple(100 + 17 * i for i in range(3 * length))
    mapping = NucleotideAtomMapping(
        construct,
        ids,
        tuple(key for key in keys for _ in range(3)),
        ("C2", "C6", "C4") * length,
    )
    ring = np.array([[1.0, 0.0, 0.0], [-0.5, 0.8, 0.0], [-0.5, -0.8, 0.0]])
    centers = np.zeros((length, 3))
    centers[:, 2] = 3.4 * np.arange(length)
    positions = jnp.asarray((ring[None] + centers[:, None]).reshape((-1, 3)))
    reference = positions.at[:, 0].add(jnp.repeat(0.05 * jnp.sin(jnp.arange(length)), 3))
    binding = prepare_nucleotide_binding(mapping, ids)
    # Directed fixed sparse support includes all nonzero pairs for BOTH fixtures.
    pairs = tuple(
        (keys[i], keys[j])
        for i in range(length)
        for j in range(max(0, i - 3), min(length, i + 4))
        if i != j
    )
    results, values = [], []
    for name, support in (("dense", None), ("sparse", pairs)):
        descriptor = NucleotideGDescriptor(
            binding,
            length_unit=ANGSTROM,
            pairs=support,
            image_policy="nonperiodic",
            smooth_width=smooth_width,
        )
        function = jax.jit(lambda x: descriptor.compare(x, reference).squared_distance)
        executable, timing = measure_lower_and_compile(
            lambda: function.lower(positions), lambda lowered: lowered.compile()
        )
        result, elapsed = measure_repeated(
            lambda: executable(positions), warmup=1, repeats=repeats
        )
        gradient = jax.jit(
            jax.grad(lambda x: descriptor.compare(x, reference).squared_distance)
        )(positions)
        rotation = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        moved = positions @ rotation.T + jnp.array([3.0, -2.0, 4.0])
        rigid_error = jnp.max(
            jnp.abs(
                descriptor.evaluate(moved).values - descriptor.evaluate(positions).values
            )
        )
        frame = base_frames(positions, binding, image_policy="nonperiodic")
        evidence = compiler_evidence(
            executable.cost_analysis(),
            executable.memory_analysis(),
            source="jax-compiled-executable",
        )
        results.append(
            {
                "support": name,
                "nucleotides": length,
                "atoms": 3 * length,
                "pair_capacity": descriptor.pair_indices.shape[0],
                "active_pairs": int(
                    jnp.sum(descriptor.evaluate(positions).within_cutoff)
                ),
                "descriptor_id": descriptor.descriptor_id,
                "definition": "published" if smooth_width == 0 else "distinct-C2-taper",
                "compile": asdict(timing),
                "execution_seconds": elapsed.to_seconds_dict(),
                "compiler": asdict(evidence),
                "logical_bytes": logical_array_bytes((positions, descriptor)),
                "squared_ermsd": float(result),
                "rigid_error": float(rigid_error),
                "force_translation_residual": float(
                    jnp.max(jnp.abs(jnp.sum(gradient, axis=0)))
                ),
                "frame_orthogonality_error": float(
                    jnp.max(
                        jnp.abs(
                            jnp.swapaxes(frame.axes, -1, -2) @ frame.axes - jnp.eye(3)
                        )
                    )
                ),
                "frame_valid": bool(jnp.all(frame.valid)),
            }
        )
        values.append(float(result))
    return {
        "profiles": results,
        "sparse_dense_absolute_error": abs(values[0] - values[1]),
    }


def torsion_probe(repeats):
    construct = NucleicAcidConstruct(("ring",), ("A",), ("RNA",), (False,))
    names = ("P", "O5'", "C5'", "C4'", "C3'", "O3'", "O4'", "C1'", "C2'", "N9", "C4")
    ids = tuple(13 * i + 50 for i in range(len(names)))
    mapping = NucleotideAtomMapping(
        construct, ids, (construct.nucleotide_keys[0],) * len(ids), names
    )
    units = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = AtomisticSystemPlan(
        ids, [6] * len(ids), [12.0] * len(ids), units, atom_type_ids=[0] * len(ids)
    ).prepare()
    parameter = jnp.arange(len(ids), dtype=float)
    positions = jnp.stack(
        (jnp.cos(parameter), jnp.sin(parameter), 0.2 * parameter), axis=-1
    )
    program = NucleotideTorsionProgram(mapping, system)
    function = jax.jit(lambda x: program.evaluate(x).values)
    executable, timing = measure_lower_and_compile(
        lambda: function.lower(positions), lambda lowered: lowered.compile()
    )
    values, elapsed = measure_repeated(
        lambda: executable(positions), warmup=1, repeats=repeats
    )
    rotation = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = executable(positions @ rotation.T + jnp.array([3.0, 1.0, 2.0]))
    delta = jnp.arctan2(jnp.sin(values - moved), jnp.cos(values - moved))
    pucker = sugar_pseudorotation(program.evaluate(positions))
    return {
        "atoms": len(ids),
        "compile": asdict(timing),
        "execution_seconds": elapsed.to_seconds_dict(),
        "proper_rigid_torsion_error": float(jnp.max(jnp.abs(delta))),
        "pucker_harmonic_residual": float(pucker.harmonic_residual[0]),
        "pucker_valid": bool(pucker.valid[0]),
        "qualification": "synthetic nonideal geometry; Fourier residual is reported, not suppressed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smooth-width", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(length < 2 for length in args.lengths) or args.repeats < 1:
        raise ValueError("At least two nucleotides and one timing repeat are required.")
    payload = {
        "environment": capture_environment().to_dict(),
        "qualification": "synthetic descriptor geometry, not experimental structure validation",
        "results": [
            run(length, args.repeats, args.smooth_width) for length in args.lengths
        ],
        "torsion_probe": torsion_probe(args.repeats),
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
