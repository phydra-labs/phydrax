#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run deterministic calibration controls for candidate electronic transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import capture_environment
from phydrax.chemistry.periodic._boltzmann import (
    ConstantRelaxationTime,
    PeriodicBoltzmannPlan,
)
from phydrax.chemistry.periodic._kubo import (
    finite_frequency_kubo_response,
    KuboDiamagneticSumRule,
    KuboLinewidth,
    PeriodicKuboPlan,
)
from phydrax.operators.quantum._electronic_transport import retarded_embedding


_E = 1.602176634e-19
_KB = 1.380649e-23
_HBAR = 1.0545718176461565e-34


def qualification() -> dict:
    energy = 0.2 + 0.1j
    lead_energy = -0.4
    contact_h = jnp.asarray([[0.8]])
    contact_s = jnp.asarray([[0.1]])
    base = retarded_embedding(
        energy,
        jnp.asarray([[1.0 / (energy - lead_energy)]]),
        contact_h,
        contact_s,
    )
    shift = 1.7
    translated = retarded_embedding(
        energy + shift,
        jnp.asarray([[1.0 / ((energy + shift) - (lead_energy + shift))]]),
        contact_h + shift * contact_s,
        contact_s,
    )
    embedding_error = float(jnp.max(jnp.abs(base.self_energy - translated.self_energy)))

    gap, temperature, speed, volume = 1.2 * _E, 300.0, 1.7e5, 1.0e-28
    energies = jnp.asarray([[-0.5 * gap, 0.5 * gap]])
    velocity = jnp.zeros((1, 2, 2, 1), dtype="complex128")
    velocity = velocity.at[0, 0, 1, 0].set(speed).at[0, 1, 0, 0].set(speed)
    f_difference = np.tanh(gap / (4.0 * _KB * temperature))
    f_sum = np.pi * _E**2 / volume * f_difference / gap * speed**2
    kubo = PeriodicKuboPlan(
        energies,
        velocity,
        jnp.ones(1),
        KuboDiamagneticSumRule(
            jnp.asarray([[f_sum]]), source_id="calibration-analytic-two-level"
        ),
        chemical_potential_joule=0.0,
        temperature_kelvin=temperature,
        cell_volume_m3=volume,
    )
    raw = kubo.raw_transitions()
    frequency = gap / _HBAR
    optical = finite_frequency_kubo_response(
        kubo,
        jnp.asarray([0.75 * frequency, frequency, 1.25 * frequency]),
        KuboLinewidth(0.01 * _E, mechanism_id="calibration-declared-linewidth"),
        f_sum_tolerance=1.0e-12,
    )

    boltzmann_energies = jnp.asarray([[-0.04 * _E], [0.04 * _E]])
    boltzmann_velocity = jnp.asarray([[[1.0e5, 0.5e5]], [[-0.6e5, 1.1e5]]])

    def boltzmann(tau):
        return PeriodicBoltzmannPlan(
            boltzmann_energies,
            boltzmann_velocity,
            jnp.asarray([0.5, 0.5]),
            ConstantRelaxationTime(tau, mechanism_id="calibration-supplied-tau"),
            chemical_potential_joule=0.0,
            temperature_kelvin=temperature,
            cell_volume_m3=volume,
        ).evaluate()

    thermal = boltzmann(1.0e-14)
    thermal_double = boltzmann(2.0e-14)
    tau_error = float(
        jnp.max(
            jnp.abs(
                thermal_double.electrical_conductivity_siemens_per_m
                - 2.0 * thermal.electrical_conductivity_siemens_per_m
            )
        )
        / jnp.max(jnp.abs(thermal.electrical_conductivity_siemens_per_m))
    )
    cases = {
        "retarded_embedding": {
            "gamma_minimum_eigenvalue": float(
                base.evidence.broadening_minimum_eigenvalue
            ),
            "energy_shift_error": embedding_error,
            "passed": bool(base.evidence.causal) and embedding_error <= 1.0e-12,
        },
        "kubo_two_level": {
            "f_sum_relative_residual": float(raw.f_sum_relative_residual),
            "minimum_dissipative_eigenvalue": float(
                optical.evidence.minimum_dissipative_eigenvalue
            ),
            "drude_weight": float(raw.drude_weight[0, 0]),
            "passed": bool(optical.successful) and float(raw.drude_weight[0, 0]) == 0.0,
        },
        "boltzmann_constant_tau": {
            "tau_scaling_relative_error": tau_error,
            "onsager_relative_residual": float(
                thermal.evidence.kelvin_onsager_relative_residual
            ),
            "electrical_rank": int(thermal.evidence.electrical_rank),
            "passed": bool(jnp.all(thermal.successful)) and tau_error <= 1.0e-12,
        },
    }
    return {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/cm_transport_qualification.json"),
    )
    arguments = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError("Transport qualification requires JAX_ENABLE_X64=1.")
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
