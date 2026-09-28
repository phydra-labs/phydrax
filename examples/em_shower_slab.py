# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Two-generation electromagnetic shower of 200 keV photons in a synthetic slab.

Photons are absorbed photoelectrically, their photoelectrons radiate
bremsstrahlung photons into the next generation, and the shower ledger closes
on the primary energy. The material tables are synthetic; nothing here is a
claim about a real medium.
"""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _manifest(name: str) -> phx.qualification.ReferenceArtifactManifest:
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum="e" * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_eV": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("synthetic:example",),
    )


def _photon_table(
    name: str, grid: phx.equations.PhotonEnergyGrid, value: float
) -> phx.equations.DiagnosticPhotonCoefficientTable:
    provenance = phx.nuclear.NuclearDataProvenance(
        _manifest(name), f"https://example.invalid/{name}", "synthetic", "example", name
    )
    unit = phx.units.derived_unit(
        "m2/kg", ((phx.units.METER, 2), (phx.units.KILOGRAM, -1))
    )
    return phx.equations.DiagnosticPhotonCoefficientTable(
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ATTENUATION,
        grid,
        ("slab",),
        jnp.asarray(((value, value),)),
        unit,
        provenance,
        phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
    )


def main() -> None:
    ev_per_joule = float(
        phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE)
    )
    grid = phx.equations.PhotonEnergyGrid(jnp.asarray((1.0e3, 2.0e6)) * ev_per_joule)
    library = phx.equations.RadiationCrossSectionLibrary(
        _photon_table("photoelectric", grid, 1.0),
        _photon_table("compton", grid, 0.0),
        _photon_table("rayleigh", grid, 0.0),
        jnp.asarray((1.0,)),
    )
    materials = phx.equations.ChargedRadiationMaterialLibrary(
        jnp.asarray((10.0, 2.0e6)),
        jnp.full((1, 2), 1.0e3),
        jnp.full((1, 2), 1.0e-3),
        jnp.full((1, 2), 2.0),
        ("slab",),
        _manifest("charged-slab"),
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.zeros((3,)),
        jnp.ones((3,)),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        material_count=1,
    )
    photons = phx.solver.PhotonTransportPlan(
        geometry,
        library,
        maximum_events=4,
        cutoff_energy=1.0e3,
        angular_sampling_attempts=64,
        electron_stack=phx.solver.SecondaryStackSpec(1, minimum_energy=10.0),
    )
    charged = phx.solver.ChargedParticleTransportPlan(
        geometry,
        materials,
        maximum_steps=64,
        maximum_step_length=0.05,
        cutoff_energy_ev=10.0,
        step_bank_capacity=64,
        photon_stack=phx.solver.SecondaryStackSpec(64, minimum_energy=1.0e3),
    )
    shower = phx.solver.EMShowerPlan(
        photons,
        charged,
        photon_capacity=256,
        charged_capacity=256,
        maximum_generations=2,
    )
    count = 256
    primaries = phx.solver.ShowerParticleBatch.photons(
        jnp.broadcast_to(jnp.asarray((0.5, 0.5, 0.0)), (count, 3)),
        jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (count, 3)),
        jnp.full((count,), 2.0e5),
        capacity=256,
    )
    result = shower.simulate(primaries, None, jr.key(0))
    print("status", phx.solver.EMShowerStatus(int(result.status)).name)
    print("primary_energy_eV", float(result.primary_energy))
    print("deposited_energy_eV", float(result.deposited_energy))
    print("escaped_energy_eV", float(result.escaped_energy))
    print("stack_remainder_energy_eV", float(result.stack_remainder_energy))
    print("ledger_residual_eV", float(result.ledger_residual))
    print("photoelectrons", int(result.generation_secondary_electron_count[0]))
    print("bremsstrahlung_photons", int(result.generation_secondary_photon_count[1]))


if __name__ == "__main__":
    main()
