#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run analytic synthetic checks for nuclear, tokamak, and reactor substrates."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

import phydrax as phx


def _data(name):
    payload = name.encode()
    reference = phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"identity": 1.0},
        uncertainty={"analytic": 0.0},
        lineage_ids=("synthetic-generator",),
    )
    return phx.nuclear.NuclearDataProvenance(
        reference,
        f"synthetic://{name}",
        "analytic-fixtures",
        "current",
        name,
    )


def qualify():
    line = phx.discretization.finite_volume.MetricLinePlan(
        [0.0, 0.25, 1.0], [2.0, 3.0], [1.0, 2.0, 4.0], "qualification-line"
    ).prepare()
    flux = np.asarray([[2.0], [3.0], [1.0]])
    line_evidence = line.conservation_evidence(flux)

    inductance = phx.circuit.CoupledInductancePlan(
        np.asarray([[2.0, 0.5], [0.5, 1.0]]),
        np.diag([0.1, 0.2]),
        ("active", "passive"),
    ).prepare()
    circuit = inductance.step_implicit_euler([1.0, 0.0], [1.0, 0.0], 0.1)

    parent = phx.nuclear.NuclideKey(10, 20)
    daughter = phx.nuclear.NuclideKey(10, 20, 1)
    groups = phx.nuclear.EnergyGroupStructure(
        [0.0, 1.0], phx.units.MEGAELECTRONVOLT, source_id="qualification-groups"
    )
    decay_rate = math.log(2.0) / 4.0
    transition = phx.nuclear.InventoryTransition(
        "isomeric-transition",
        daughter,
        ((parent, 1.0),),
        decay_rate,
        np.asarray([0.0]),
        1.0e-14,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _data("isomeric-transition"),
    )
    activation = phx.nuclear.ActivationNetworkPlan(
        (daughter, parent), groups, (transition,), error_tolerance=1.0e-12
    ).prepare()
    decay = activation.step(activation.inventory([1.0, 0.0]), [0.0], 2.0)
    decay_error = float(
        np.max(
            np.abs(
                np.asarray(decay.accepted.amounts_mol)
                - np.asarray(
                    [math.exp(-decay_rate * 2.0), 1.0 - math.exp(-decay_rate * 2.0)]
                )
            )
        )
    )

    kinetics_plan = phx.applications.reactor_physics.DelayedNeutronKineticsPlan(
        np.asarray([0.006]), np.asarray([0.2]), 1.0e-4, "analytic-kinetics"
    )
    kinetics_initial = kinetics_plan.equilibrium_state(1.0)
    kinetics = kinetics_plan.prepare().step(kinetics_initial, 0.0, 0.0, 0.1)
    kinetics_error = float(abs(float(kinetics.accepted_state.neutron_population) - 1.0))

    successful = (
        bool(line_evidence.successful)
        and bool(circuit.successful)
        and bool(decay.successful)
        and decay_error <= 1.0e-10
        and bool(kinetics.successful)
        and kinetics_error <= 1.0e-10
    )
    if not successful:
        raise RuntimeError("Synthetic nuclear/tokamak qualification failed.")
    return {
        "metric_line_closure": float(np.max(np.abs(line_evidence.closure_residual))),
        "circuit_energy_closure_j": float(abs(circuit.ledger.closure_residual_j)),
        "activation_analytic_error_mol": decay_error,
        "kinetics_stationary_error": kinetics_error,
        "candidate_profile_count": len(phx.nuclear.nuclear_candidate_profiles())
        + len(phx.applications.tokamak.tokamak_candidate_profiles())
        + len(phx.applications.reactor_physics.reactor_candidate_profiles()),
        "successful": True,
    }


def main():
    print(json.dumps(qualify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
