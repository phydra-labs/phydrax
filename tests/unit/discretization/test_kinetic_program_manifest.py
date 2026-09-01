#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.discretization.lattice_boltzmann import (
    athermal_lattice_boltzmann_manifest,
    coupled_population_manifest,
    D2Q9,
    finite_volume_dvm_manifest,
    KineticFieldRole,
    KineticFieldSpec,
    KineticProgramManifest,
    KineticStageSpec,
    LatticeBoltzmannPrecisionPolicy,
    reactive_transport_manifest,
    transport_population_manifest,
)


def test_athermal_manifest_has_complete_ordered_state_and_exchange_contract():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    manifest = athermal_lattice_boltzmann_manifest(
        lattice.lattice_id,
        precision.policy_id,
        lattice.population_count,
        lattice.dimension,
    )

    assert manifest.field("populations").checkpoint_required
    assert manifest.field("post_collision").halo_width == 1
    assert manifest.checkpoint_fields == ("populations",)
    assert tuple(stage.order for stage in manifest.stages) == tuple(
        range(len(manifest.stages))
    )
    assert manifest.stages[2].exchange_fields == ("post_collision",)


def test_coupled_manifest_records_separate_population_conservation_channels():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    manifest = coupled_population_manifest(
        "binary-test",
        lattice.lattice_id,
        precision.policy_id,
        lattice.population_count,
        lattice.dimension,
        ("hydrodynamic", "phase"),
        (("mass", "momentum"), ("phase_mass",)),
    )

    assert manifest.field("hydrodynamic").conserved_channels == ("mass", "momentum")
    assert manifest.field("phase").conserved_channels == ("phase_mass",)
    assert set(manifest.checkpoint_fields) == {"hydrodynamic", "phase"}


def test_manifest_rejects_unavailable_reads_duplicate_orders_and_missing_halos():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    initial = KineticFieldSpec(
        "initial",
        KineticFieldRole.MACROSCOPIC,
        initialized=True,
    )
    output = KineticFieldSpec("output", KineticFieldRole.MACROSCOPIC)
    with pytest.raises(ValueError, match="reads unavailable"):
        KineticProgramManifest(
            "bad-read",
            lattice.lattice_id,
            precision.policy_id,
            (initial, output),
            (KineticStageSpec("bad", 0, reads=("output",)),),
        )
    with pytest.raises(ValueError, match="orders must be unique"):
        KineticProgramManifest(
            "bad-order",
            lattice.lattice_id,
            precision.policy_id,
            (initial, output),
            (
                KineticStageSpec("one", 0, reads=("initial",), writes=("output",)),
                KineticStageSpec("two", 0, reads=("output",)),
            ),
        )
    with pytest.raises(ValueError, match="positive halo width"):
        KineticProgramManifest(
            "bad-halo",
            lattice.lattice_id,
            precision.policy_id,
            (initial, output),
            (
                KineticStageSpec(
                    "exchange",
                    0,
                    reads=("initial",),
                    writes=("output",),
                    exchange_fields=("initial",),
                ),
            ),
        )


def test_reactive_manifest_composes_thermal_and_species_dependencies():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    thermal = transport_population_manifest(
        "thermal",
        lattice.lattice_id,
        precision.policy_id,
        "thermal_populations",
        (lattice.population_count,),
        ("energy",),
        dimension=lattice.dimension,
    )
    species = transport_population_manifest(
        "species",
        lattice.lattice_id,
        precision.policy_id,
        "species_populations",
        (2, lattice.population_count),
        ("species_amount", "element_amount"),
        dimension=lattice.dimension,
        source_component_shape=(2,),
    )
    reactive = reactive_transport_manifest(
        lattice.lattice_id,
        precision.policy_id,
        thermal.manifest_id,
        species.manifest_id,
    )

    assert reactive.dependency_manifest_ids == (
        thermal.manifest_id,
        species.manifest_id,
    )
    assert set(reactive.checkpoint_fields) == {
        "thermal_state",
        "species_state",
        "reaction_extent",
    }
    dvm = finite_volume_dvm_manifest(
        "d2v-test",
        precision.policy_id,
        17,
        ("mass", "momentum_0", "momentum_1", "kinetic_energy"),
        has_source=True,
    )
    assert tuple(stage.name for stage in dvm.stages) == (
        "face_reconstruction",
        "numerical_flux_divergence",
        "source_evaluation",
        "residual_assembly",
        "diagnostics",
    )
    assert dvm.checkpoint_fields == ("dvm_populations",)
