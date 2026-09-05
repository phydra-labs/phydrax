#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark actual mixed mechanics; numerical qualification, not molecular calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from benchmarks.nucleic_rigid import parameter_data
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.coarse import (
    nucleotide_reference_sites,
    NucleotideModelPlan,
    NucleotideParameterArtifact,
)
from phydrax.applications.protein_folding.hybrid import (
    HybridCrossInteractionPlan,
    PreparedHybridModel,
)
from phydrax.atomistic import AtomisticSystemPlan, AtomisticUnitSystem, ElasticNetworkPlan
from phydrax.qualification import ReferenceArtifactManifest


def source(payload: bytes, name: str) -> ReferenceArtifactManifest:
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Author-Owned-Numerical-Benchmark",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="numerical-fixture",
        nondimensionalization={"length": 1.0, "energy": 1.0},
        uncertainty=None,
        lineage_ids=("independently-authored-uncalibrated-model",),
    )


def prepare(count: int):
    units = AtomisticUnitSystem.reduced()
    ids = np.arange(count, dtype=np.int64)
    parameter_record = parameter_data()
    spacing = parameter_record["profiles"]["DNA"]["backbone"][1]
    reference = np.column_stack(
        (spacing * np.arange(count, dtype=float), np.zeros((count, 2)))
    )
    reference_source = source(reference.tobytes(), "reference-protein")
    system = AtomisticSystemPlan(
        ids,
        np.zeros(count, dtype=np.int32),
        np.full(count, 2.0),
        units,
        element_mask=np.zeros(count, dtype=bool),
        mobile_mask=ids != 0,
    ).prepare()
    network = ElasticNetworkPlan(1.1 * spacing, 4.0, count - 1).prepare(
        system, reference, reference_id=reference_source.manifest_id
    )
    payload = json.dumps(parameter_record, sort_keys=True).encode()
    parameters = NucleotideParameterArtifact(
        source(payload, "nucleotide-parameters"), payload, units
    )
    site_ids = np.arange(8 * count, dtype=np.int64).reshape(count, 8)
    construct = NucleicAcidConstruct(("dna",), ("A" * count,), ("DNA",), (False,))
    nucleotide = NucleotideModelPlan(
        construct,
        ids,
        site_ids,
        nucleotide_reference_sites(construct, parameters),
        np.full(count, 3.0),
        np.tile(np.eye(3), (count, 1, 1)),
        parameters,
        fixed_mask=ids == 0,
    ).prepare()
    cross_parameters = {
        "steric_energy": 0.1,
        "steric_radius": 2.0,
        "linker_stiffness": 0.5,
        "linker_length": 1.5,
        "electrostatic_prefactor": -0.03,
        "screening": 0.2,
    }
    cross = HybridCrossInteractionPlan(
        np.column_stack((ids, site_ids[:, 1])),
        units,
        source(json.dumps(cross_parameters, sort_keys=True).encode(), "cross-parameters"),
        **cross_parameters,
    )
    model = PreparedHybridModel(network, nucleotide, cross, reference_source)
    rigid = nucleotide.bodies.kinematics(
        jnp.asarray(reference + [0.0, 1.5, 0.0]),
        jnp.zeros((count, 3)),
        jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (count, 1)),
        jnp.zeros((count, 3)),
    )
    return model, model.initialize(reference, np.zeros_like(reference), rigid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.particles < 2
        or args.steps < 1
        or not np.isfinite(args.step_size)
        or args.step_size <= 0
    ):
        raise ValueError(
            "Require particles >= 2, steps >= 1 and a positive finite step size."
        )
    model, state = prepare(args.particles)
    evaluation = model.evaluate(state)
    initial_energy = evaluation.energy + model.kinetic_energy(state)
    rows = []
    for refinement in (1, 2):
        dt = args.step_size / refinement
        steps = args.steps * refinement

        def rollout(initial):
            def body(current, _):
                result = model.step(current, dt)
                return result.state, (result.total_energy, result.successful)

            return jax.lax.scan(body, initial, None, length=steps)

        lowered_function = jax.jit(rollout)
        executable, compile_time = measure_lower_and_compile(
            lambda: lowered_function.lower(state), lambda lowered: lowered.compile()
        )
        (final, (energies, successful)), elapsed = measure_repeated(
            lambda: executable(state), warmup=args.warmup, repeats=args.repeats
        )
        rows.append(
            {
                "step_size": dt,
                "steps": steps,
                "lowering_seconds": compile_time.lowering_seconds,
                "compilation_seconds": compile_time.compilation_seconds,
                "execution_seconds": elapsed.to_seconds_dict(),
                "successful": bool(jnp.all(successful)),
                "maximum_energy_drift": float(
                    jnp.max(jnp.abs(energies - initial_energy))
                ),
                "final_energy_drift": float(energies[-1] - initial_energy),
                "maximum_quaternion_norm_error": float(
                    jnp.max(
                        jnp.abs(jnp.sum(final.nucleotide.orientation**2, axis=-1) - 1)
                    )
                ),
            }
        )
    force_residual = jnp.sum(evaluation.protein_forces, axis=0) + jnp.sum(
        evaluation.nucleotide_load.force, axis=0
    )
    torque_residual = jnp.sum(
        jnp.cross(state.protein.positions, evaluation.protein_forces), axis=0
    ) + jnp.sum(
        jnp.cross(state.nucleotide.position, evaluation.nucleotide_load.force)
        + evaluation.nucleotide_load.torque,
        axis=0,
    )
    # End-particle longitudinal response has exactly one reference spring.
    tangent = jax.grad(
        lambda dx: model.protein_network.evaluate(
            state.protein.positions.at[-1, 0].add(dx)
        ).forces[-1, 0]
    )(0.0)
    payload = {
        "environment": capture_environment().to_dict(),
        "protein_capacity": args.particles,
        "protein_mobile": args.particles - 1,
        "rigid_capacity": args.particles,
        "rigid_mobile": args.particles - 1,
        "rigid_marker_capacity": int(model.nucleotide_model.marker_map.markers.capacity),
        "rigid_physical_site_count": int(
            jnp.sum(model.nucleotide_model.physical_site_mask)
        ),
        "cross_pair_count": args.particles,
        "logical_array_bytes": logical_array_bytes((model, state)),
        "support_map_id": model.support_map.fingerprint(),
        "force_balance_residual": float(jnp.sqrt(jnp.sum(force_residual**2))),
        "torque_balance_residual": float(jnp.sqrt(jnp.sum(torque_residual**2))),
        "reference_longitudinal_stiffness_error": float(jnp.abs(tangent + 4.0)),
        "initial_fixed_protein_reaction_norm": float(
            jnp.sqrt(jnp.sum(evaluation.protein_reaction_forces**2))
        ),
        "initial_fixed_rigid_reaction_norm": float(
            jnp.sqrt(jnp.sum(evaluation.nucleotide_reaction_load.force**2))
        ),
        "refinement": rows,
        "scope": (
            "Nonperiodic reference-conditioned mechanics with explicit synthetic "
            "coefficients; no sequence recognition, experimental calibration, "
            "or physical-time calibration."
        ),
        "approximation": "Native rigid KDK plus joint Cartesian KDK; fixed interaction support; no heat bath.",
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
