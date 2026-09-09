#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run a synthetic native tokamak-core to fusion-source to activation step."""

import hashlib
import json

import numpy as np

import phydrax as phx


def reference(name):
    payload = name.encode()
    return phx.qualification.ReferenceArtifactManifest(
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


def main():
    theta = 2.0 * np.pi * np.arange(32) / 32
    contours = np.zeros((3, 32, 2))
    contours[0, :, 0] = 2.0
    for index, radius in enumerate((0.0, 0.25, 0.5)):
        contours[index, :, 0] = 2.0 + radius * np.cos(theta)
        contours[index, :, 1] = radius * np.sin(theta)
    geometry = phx.applications.tokamak.FluxSurfaceGeometry(
        np.asarray([0.0, 0.5, 0.9]),
        contours,
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 2.0, 4.0]),
        np.asarray([2.0, 2.0, 2.0]),
        np.asarray([0.0, 0.25, 0.5]),
        np.asarray([1.0, 1.5, 2.0]),
        "synthetic-equilibrium",
    )
    transport = phx.applications.tokamak.TokamakCoreTransportPlan(geometry, 1.0).prepare()

    deuterium_key = phx.nuclear.NuclideKey(1, 2)
    tritium_key = phx.nuclear.NuclideKey(1, 3)
    helium_key = phx.nuclear.NuclideKey(2, 4)
    deuterium = phx.nuclear.NuclearSpeciesKey.from_nuclide(deuterium_key)
    tritium = phx.nuclear.NuclearSpeciesKey.from_nuclide(tritium_key)
    helium = phx.nuclear.NuclearSpeciesKey.from_nuclide(helium_key)
    neutron = phx.nuclear.NuclearSpeciesKey.from_particle(
        phx.nuclear.NuclearParticleKind.NEUTRON
    )
    species = phx.nuclear.NuclearSpeciesTable(
        (deuterium, tritium, helium, neutron),
        np.asarray(
            [3.3435837724e-27, 5.0073567446e-27, 6.6446573357e-27, 1.67492749804e-27]
        ),
        reference("synthetic-masses"),
    )
    reaction_data = phx.nuclear.NuclearDataProvenance(
        reference("synthetic-reactivity"),
        "synthetic://reactivity",
        "synthetic-reactivity",
        "current",
        "d-t",
    )
    participant = phx.nuclear.NuclearReactionParticipant
    channel = phx.nuclear.NuclearReactionChannel.from_species_table(
        "d-t",
        (participant(deuterium), participant(tritium)),
        (participant(helium), participant(neutron)),
        species,
        reaction_data,
    )
    kev = float(phx.units.conversion_factor(phx.units.KILOELECTRONVOLT, phx.units.JOULE))
    reactivity = phx.nuclear.TabulatedMaxwellianReactivity(
        kev * np.asarray([1.0, 5.0, 10.0, 20.0]),
        np.asarray([1.0e-24, 1.0e-23, 3.0e-23, 5.0e-23]),
        channel,
    )
    fusion = phx.nuclear.ThermalFusionReactionPlan(channel, reactivity, species)

    cobalt = phx.nuclear.NuclideKey(27, 59)
    cobalt60 = phx.nuclear.NuclideKey(27, 60)
    groups = phx.nuclear.EnergyGroupStructure(
        [0.0, 1.0, 20.0], phx.units.MEGAELECTRONVOLT, source_id="synthetic-groups"
    )
    capture = phx.nuclear.InventoryTransition(
        "cobalt-capture",
        cobalt,
        ((cobalt60, 1.0),),
        0.0,
        np.asarray([1.0e-28, 2.0e-28]),
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        phx.nuclear.NuclearDataProvenance(
            reference("synthetic-capture"),
            "synthetic://capture",
            "synthetic-capture",
            "current",
            "co59-n-gamma",
        ),
    )
    activation = phx.nuclear.ActivationNetworkPlan(
        (cobalt, cobalt60), groups, (capture,)
    ).prepare()
    response = phx.applications.tokamak.NeutronResponsePlan(
        np.asarray([[1.0e8, 1.0e8], [1.0e9, 1.0e9]]),
        "synthetic-linear-response",
    )
    scenario = phx.applications.tokamak.FusionActivationScenarioPlan(
        transport,
        fusion,
        activation,
        response,
        0.5,
        0.5,
        1,
    ).prepare()

    state = phx.applications.tokamak.TokamakCoreState(
        np.asarray([1.0e20, 8.0e19]),
        kev * np.asarray([8.0, 6.0]),
        kev * np.asarray([10.0, 8.0]),
    )
    faces = np.zeros((3,))
    result = scenario.step(
        state,
        activation.inventory([1.0, 0.0]),
        1.0e-3,
        phx.applications.tokamak.TokamakTransportCoefficients(faces, faces, faces),
        phx.applications.tokamak.TokamakTransportSources(
            np.zeros(2), np.zeros(2), np.zeros(2)
        ),
    )
    if not bool(result.successful):
        raise RuntimeError("Synthetic fusion-activation scenario failed.")
    print(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "fusion_power_w": float(
                    np.sum(
                        np.asarray(result.fusion.total_power_density_w_m3)
                        * np.asarray(transport.geometry.cell_volume_m3)
                    )
                ),
                "neutron_source_rate_s": float(np.sum(result.core_neutron_source_rate_s)),
                "activation_scalar_flux_m2_s": np.asarray(
                    result.activation_scalar_flux_m2_s
                ).tolist(),
                "final_inventory_mol": np.asarray(
                    result.activation.accepted.amounts_mol
                ).tolist(),
                "sensitivity_valid": bool(result.sensitivity_valid),
                "successful": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
