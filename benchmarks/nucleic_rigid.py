# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Numerical rigid mechanics benchmark; synthetic pairs are not duplex calibration.

Run: JAX_ENABLE_X64=1 python -m benchmarks.nucleic_rigid --bodies 16 --steps 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.coarse import (
    nucleotide_reference_sites,
    NucleotideModelPlan,
    NucleotideParameterArtifact,
)
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.discretization.particle._rigid_body import (
    _quaternion_retract,
    RigidBodyKinematics,
)
from phydrax.qualification._reference import ReferenceArtifactManifest


def parameter_data(family="average-dna"):
    """Independently authored noncalibrated coefficients for all equation paths.

    These are not a transcription of any model table. They are useful for
    conservation/continuity regressions only, never published-scale observables.
    """
    angle = lambda theta: [1.3, float(theta), 0.6]
    morse = lambda amplitude: [amplitude, 1.0, 2.0, 0.7, 1.7, 2.0]
    harmonic = lambda amplitude: [amplitude, 1.0, 1.8, 0.25, 1.6, 1.0]
    profile = {
        "backbone": [3.0, 1.8, 0.8],
        "excluded": {
            "back-back": [0.1, 0.3, 0.28],
            "base-base": [0.1, 0.3, 0.28],
            "back-base": [0.1, 0.3, 0.28],
        },
        "hydrogen-bond": {
            "radial": morse(1.0),
            "angles": {
                "1": angle(0),
                "2": angle(0),
                "3": angle(0),
                "4": angle(np.pi),
                "7": angle(np.pi / 2),
                "8": angle(np.pi / 2),
            },
        },
        "stacking": {
            "radial": morse(0.2),
            "angles": {"4": angle(0), "5": angle(0), "6": angle(0)},
            "helicity": [[1.0, -0.5], [1.0, -0.5]],
        },
        "cross-stacking": {
            "radial": harmonic(0.1),
            "angles": {
                "1": angle(2.3),
                "2": angle(1.0),
                "3": angle(1.0),
                "4": angle(0),
                "7": angle(0.9),
                "8": angle(0.9),
            },
        },
        "coaxial-stacking": {
            "radial": harmonic(0.1),
            "angles": {
                "1": angle(np.pi - 0.6),
                "4": angle(0),
                "5": angle(0),
                "6": angle(0),
            },
            "helicity": [[1.0, -0.5], [1.0, -0.5]],
        },
        "stacking_temperature_coefficient": 0.1,
    }
    dna_geometry = {
        "backbone": [-0.2, 0.0, 0.0],
        "base": [0.1, 0.0, 0.0],
        "stack3": [0.08, 0.0, 0.0],
        "stack5": [0.08, 0.0, 0.0],
        "coax": [0.08, 0.0, 0.0],
    }
    rna_geometry = {
        "backbone": [-0.2, 0.05, 0.04],
        "base": [0.1, 0.0, 0.0],
        "stack3": [0.08, 0.0, 0.04],
        "stack5": [0.08, 0.0, -0.04],
        "coax": [0.08, 0.0, 0.0],
    }
    names = (
        ("DNA", "RNA", "HYBRID")
        if family == "dna-rna-hybrid"
        else (("RNA",) if family == "rna" else ("DNA",))
    )
    profiles, geometry, strengths = {}, {}, {}
    for name in names:
        item = deepcopy(profile)
        if name == "RNA":
            item["hydrogen-bond"]["angles"]["1"] = angle(np.pi)
            del item["cross-stacking"]["angles"]["4"]
            item["stacking"]["angles"] = {
                "5": angle(np.pi / 2),
                "6": angle(0),
                "9": angle(0),
                "10": angle(0),
            }
            item["p3"] = (np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)).tolist()
            item["p5"] = (np.array([-1.0, -2.0, 1.0]) / np.sqrt(6)).tolist()
            geometry[name] = rna_geometry
        else:
            if family != "average-dna":
                item["coaxial-stacking"].pop("helicity")
                item["coaxial-stacking"]["angles"]["1"] = angle(np.pi - 0.25)
                item["coaxial-stacking"]["f6"] = [6.5, float(np.pi - 0.1)]
            if name == "DNA":
                geometry[name] = deepcopy(dna_geometry)
                if family != "average-dna":
                    geometry[name]["backbone"][1] = 0.15
        if family in ("groove-salt-dna", "sequence-dna", "dna-rna-hybrid"):
            item["screening"] = {
                "prefactor": 0.1,
                "length_per_sqrt_temperature_over_molar": 0.2,
                "terminal_charge_factor": 0.5,
            }
        hb = np.zeros((4, 4))
        hb[[0, 1, 2, 3], [3, 2, 1, 0]] = 1.0
        if name == "RNA":
            hb[1, 3] = hb[3, 1] = 0.7
        stack = np.ones((4, 4))
        if family == "sequence-dna":
            stack[0, 3] = 1.5
        strengths[name] = {"stacking": stack.tolist(), "hydrogen-bond": hb.tolist()}
        if name == "HYBRID":
            for field in ("backbone", "stacking", "stacking_temperature_coefficient"):
                del item[field]
            del strengths[name]["stacking"]
        profiles[name] = item
    return {
        "family": family,
        "source_model": "independently-authored-equation-regression",
        "temperature": 1.0,
        "salt_concentration": 1.0,
        "salt_unit": "mole/litre",
        "geometry": geometry,
        "profiles": profiles,
        "sequence_strengths": strengths,
    }


def parameter_artifact(family="average-dna"):
    payload = json.dumps(parameter_data(family), sort_keys=True).encode()
    manifest = ReferenceArtifactManifest(
        "analytic-pair-mechanics",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"reduced-energy": 1.0, "reduced-length": 1.0},
        uncertainty=None,
        lineage_ids=("independently-authored-equation-fixture",),
    )
    return NucleotideParameterArtifact(manifest, payload, AtomisticUnitSystem.reduced())


def make_fixture(count=16):
    """Independent noninteracting pairs with nonzero orientation torques."""
    if count < 2 or count % 2:
        raise ValueError("Body count must be positive and even.")
    ids = 1001 + np.arange(count, dtype=np.int64) * 7
    parameters = parameter_artifact()
    construct = NucleicAcidConstruct(
        tuple(f"s{i}" for i in range(count)),
        tuple("A" if i % 2 == 0 else "T" for i in range(count)),
        ("DNA",) * count,
        (False,) * count,
    )
    geometry = nucleotide_reference_sites(construct, parameters)
    model = NucleotideModelPlan(
        construct,
        ids,
        100003 + np.arange(8 * count).reshape(count, 8),
        geometry,
        np.ones(count) * 2,
        np.broadcast_to(np.eye(3), (count, 3, 3)),
        parameters,
    ).prepare()
    position = np.zeros((count, 3))
    position[:, 1] = np.repeat(np.arange(count // 2) * 5.0, 2)
    position[1::2] += [1.4, 0.1, 0.1]
    orientation = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
    orientation[1::2] = [0.0, 0.0, np.cos(0.2), np.sin(0.2)]
    state = model.bodies.kinematics(
        position, np.zeros((count, 3)), orientation, np.zeros((count, 3))
    )
    return model, state


def run(count=16, steps=100, repeats=5):
    model, state = make_fixture(count)
    energy_force = jax.jit(lambda value: model.evaluate(value))
    compiled, compilation = measure_lower_and_compile(
        lambda: energy_force.lower(state), lambda lowered: lowered.compile()
    )
    evaluation, timing = measure_repeated(
        lambda: compiled(state), warmup=1, repeats=repeats
    )
    load = evaluation.loads.load
    direction = (
        jnp.broadcast_to(jnp.array([0.1, -0.2, 0.3]), state.angular_velocity.shape)
        .at[1::2]
        .multiply(-1)
    )

    def displaced(t):
        return model.energy(
            RigidBodyKinematics(
                state.position,
                state.velocity,
                _quaternion_retract(state.orientation, t * direction),
                state.angular_velocity,
            )
        )

    fd_wrench = (displaced(1e-5) - displaced(-1e-5)) / 2e-5
    adjoint_wrench = -jnp.sum(load.torque * direction)
    initial_energy = model.energy(state) + model.kinetic_energy(state)

    def trajectory(dt, n):
        def advance(q, i):
            result = model.step(q, i * dt, dt)
            energy = model.energy(result.kinematics) + model.kinetic_energy(
                result.kinematics
            )
            return result.kinematics, (energy, result.successful)

        return jax.lax.scan(advance, state, jnp.arange(n))

    coarse = jax.jit(lambda: trajectory(0.01, steps))()
    fine = jax.jit(lambda: trajectory(0.005, 2 * steps))()
    bath = model.heat_bath(2.0, 3.0)

    # Actual thermostatted conservative evolution, with explicit PRNG event keys.
    def thermal_rollout(key):
        def advance(q, inputs):
            i, event_key = inputs
            result = bath.step(
                q,
                model.mechanical_load(q),
                i * 0.005,
                0.005,
                lambda t, s, args: model.mechanical_load(s),
                event_key,
            )
            return result.kinematics, result.successful

        return jax.lax.scan(
            advance, state, (jnp.arange(steps), jax.random.split(key, steps))
        )

    thermal_state, thermal_success = jax.jit(thermal_rollout)(jax.random.key(13))
    # Independent OU draws compare with the exact finite-time covariance, not
    # an unjustified configurational-equilibrium claim for a short trajectory.
    draws = jax.jit(jax.vmap(lambda key: bath.apply(state, 10.0, key)))(
        jax.random.split(jax.random.key(21), 4096)
    )
    translation_variance = np.var(np.asarray(draws.velocity), axis=0)
    rotation_variance = np.var(np.asarray(draws.angular_velocity), axis=0)
    return {
        "environment": capture_environment().to_dict(),
        "qualification": (
            "published equations with independent analytic coefficients; "
            "duplex observables and physical clock calibration gated"
        ),
        "body_capacity": count,
        "active_bodies": count,
        "site_capacity": 8 * count,
        "physical_sites": 5 * count,
        "selected_pairs": count * (count - 1) // 2,
        "logical_array_bytes": logical_array_bytes((model, state)),
        "compilation": asdict(compilation),
        "execution_seconds": timing.to_dict(),
        "energy": float(evaluation.energy),
        "force_balance_norm": float(jnp.sqrt(jnp.sum(jnp.sum(load.force, 0) ** 2))),
        "torque_balance_norm": float(
            jnp.sqrt(
                jnp.sum(
                    jnp.sum(load.torque + jnp.cross(state.position, load.force), 0) ** 2
                )
            )
        ),
        "finite_difference_wrench_error": float(jnp.abs(fd_wrench - adjoint_wrench)),
        "max_energy_drift_dt_001": float(jnp.max(jnp.abs(coarse[1][0] - initial_energy))),
        "max_energy_drift_dt_0005": float(jnp.max(jnp.abs(fine[1][0] - initial_energy))),
        "conservative_successful": bool(jnp.all(coarse[1][1]) & jnp.all(fine[1][1])),
        "thermal_steps_successful": bool(jnp.all(thermal_success)),
        "final_thermal_kinetic_energy": float(model.kinetic_energy(thermal_state)),
        "ou_translation_variance_relative_error": float(
            np.max(np.abs(translation_variance / 0.5 - 1))
        ),
        "ou_rotation_variance_relative_error": float(
            np.max(np.abs(rotation_variance - 1))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodies", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    options = parser.parse_args()
    if options.steps < 1 or options.repeats < 1:
        parser.error("steps and repeats must be positive")
    print(json.dumps(run(options.bodies, options.steps, options.repeats), indent=2))


if __name__ == "__main__":
    main()
