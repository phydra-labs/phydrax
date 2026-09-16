#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax.ein as ein
from benchmarks._runtime import capture_environment
from phydrax.applications.cosmology._dark_radiation import DarkRadiationLedgerPlan
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm_kernels import TwoBodyDifferentialKernelPlan
from phydrax.applications.cosmology._sidm_reactions import (
    DarkTwoBodyReactionPlan,
    InelasticSIDMPlan,
)
from phydrax.applications.cosmology._sidm_weighted import WeightedSIDMPacketState
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest


def _case(repetitions: int):
    species = tuple(
        DarkSectorSpeciesPlan(
            name,
            mass,
            internal_energy=internal,
            charge_names=("dark",),
            charges=(0.0,),
        )
        for name, mass, internal in (
            ("a", 1.0, 0.0),
            ("b", 1.0, 0.0),
            ("c", 0.9, 10.25),
            ("d", 0.9, 10.25),
        )
    )
    recoil_speed = np.sqrt(1.0 / 0.45)
    speeds = np.asarray((0.0, recoil_speed, 2.0, 5.0))

    def kernel(first, second, total):
        cosines = np.asarray((-1.0, 0.0, 1.0))
        differential = np.full((speeds.size, cosines.size), total / (4.0 * np.pi))
        payload = TwoBodyDifferentialKernelPlan.canonical_table_bytes(
            speeds, cosines, differential
        )
        digest = hashlib.sha256(payload).hexdigest()
        lineage = (f"synthetic:{first.species_id}:{second.species_id}",)
        manifest = ReferenceArtifactManifest(
            f"sidm-reaction-benchmark-kernel-{digest[:16]}",
            checksum_algorithm="sha256",
            checksum=digest,
            size_bytes=len(payload),
            license_id="benchmark-synthetic-kernel",
            commercial_use_permitted=True,
            redistribution_permitted=True,
            training_use_permitted=True,
            export_permitted=True,
            export_classification="unrestricted",
            nondimensionalization={"speed": 1.0, "cross_section": 1.0},
            uncertainty={"relative_cross_section": 1.0e-12},
            lineage_ids=lineage,
        )
        artifact = ScientificArtifactEnvelope(
            artifact_kind="synthetic-sidm-differential-kernel",
            content_digest=digest,
            producer="phydrax-benchmark",
            producer_version="1",
            build_id=f"sidm-reaction-benchmark-kernel-{digest[:16]}",
            license_id=manifest.license_id,
            resource_id=manifest.manifest_id,
            status="complete",
            parent_artifact_ids=lineage,
        )
        return TwoBodyDifferentialKernelPlan(
            first,
            second,
            speeds,
            cosines,
            differential,
            source_artifact=artifact,
            reference_manifest=manifest,
            commercial_use=True,
            redistribution=True,
            training_use=True,
            export=True,
        )

    channel = DarkTwoBodyReactionPlan(
        species[:2],
        species[2:],
        kernel(species[0], species[1], 1.0),
        kernel(species[2], species[3], 1.0 / 0.45),
        conserved_charge_names=("dark",),
        speed_of_light=10.0,
        maximum_speed_fraction=0.2,
        detailed_balance_tolerance=2.0e-12,
    )
    plan = InelasticSIDMPlan((channel,), DarkRadiationLedgerPlan(1, speed_of_light=10.0))
    active = jnp.asarray((True, True, False))
    microscopic = jnp.asarray((1.0, 1.0, 0.0))
    weights = jnp.asarray((1.0, 1.0, 0.0))
    macro = microscopic * weights
    velocities = jnp.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    packets = WeightedSIDMPacketState(
        jnp.zeros((3, 3)),
        microscopic,
        weights,
        macro,
        0.5 * macro[:, None] * velocities,
        active,
        jnp.asarray((1, 2, 3), dtype=jnp.int64),
        jnp.full((3,), -1, dtype=jnp.int64),
        jnp.asarray((0, 0, -1), dtype=jnp.int32),
        jnp.asarray(0.5),
    )
    initial = plan.initialize(packets, jnp.asarray((0, 1, -1), dtype=jnp.int32))
    compiled_rates = eqx.filter_jit(plan.partial_rates)
    compiled_react = eqx.filter_jit(plan.react)

    started = time.perf_counter()
    rates = compiled_rates(initial, 0, 1)
    first = compiled_react(initial, 0, 1, jr.key(0))
    jax.block_until_ready(first.accepted_state.packets.canonical_momenta)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - started)

    started = time.perf_counter()
    result = first
    for epoch in range(repetitions):
        result = compiled_react(initial, 0, 1, jr.key(epoch + 1))
    jax.block_until_ready(result.accepted_state.packets.canonical_momenta)
    execution_ms = 1000.0 * (time.perf_counter() - started) / repetitions
    return {
        "reaction_profile": "nonrelativistic-reversible-2-to-2",
        "channel_count": len(plan.channels),
        "forward_partial_rate": float(rates.partial_rates[0]),
        "reverse_partial_rate": float(rates.partial_rates[1]),
        "total_partial_rate": float(rates.total_rate),
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "reactions_per_second": 1000.0 / execution_ms,
        "selected_channel": int(result.evidence.selected_channel),
        "detailed_balance_residual": float(result.evidence.detailed_balance_residual),
        "charge_defect_norm": float(
            jnp.sqrt(
                ein.contract(
                    "i,i->", result.evidence.charge_defect, result.evidence.charge_defect
                )
            )
        ),
        "physical_momentum_defect_norm": float(
            jnp.sqrt(
                ein.contract(
                    "i,i->",
                    result.evidence.physical_momentum_defect,
                    result.evidence.physical_momentum_defect,
                )
            )
        ),
        "canonical_momentum_defect_norm": float(
            jnp.sqrt(
                ein.contract(
                    "i,i->",
                    result.evidence.canonical_momentum_defect,
                    result.evidence.canonical_momentum_defect,
                )
            )
        ),
        "rest_mass_energy_change": float(result.evidence.rest_mass_energy_change),
        "internal_energy_change": float(result.evidence.internal_energy_change),
        "kinetic_energy_change": float(result.evidence.kinetic_energy_change),
        "radiation_energy_change": float(result.evidence.radiation_energy_change),
        "total_energy_defect": float(result.evidence.total_energy_defect),
        "gravitational_mass_change": float(result.evidence.gravitational_mass_change),
        "dynamic_pm_mass_required": bool(result.evidence.dynamic_pm_mass_required),
        "successful": bool(result.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/dark_matter_sidm_reactions.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    case = _case(arguments.repeats)
    payload = {
        "environment": capture_environment().to_dict(),
        "case": case,
        "all_successful": case["successful"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
