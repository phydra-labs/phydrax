"""Native model, prepared geometry, and derivative boundaries retain their types."""

from pathlib import Path
from typing import assert_type

from jax import Array

from phydrax.atomistic import (
    AtomicStructure,
    atomistic_energy_derivatives,
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticBatch,
    AtomisticEnergyDerivatives,
    AtomisticEnergyObjective,
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticGraphPayload,
    AtomisticModelArtifact,
    AtomisticModelArtifactManifest,
    AtomisticScaleContract,
    AtomisticTrainingProblem,
    AtomisticUnitSystem,
    NativeAtomisticProviderPlan,
    read_atomistic_model_artifact,
    write_atomistic_model_artifact,
)
from phydrax.atomistic.interchange import (
    convert_mace_checkpoint,
    MACECheckpointConversion,
    MACEProviderRuntime,
    MACESource,
)
from phydrax.export import (
    AtomisticIREEExportBundle,
    load_atomistic_iree,
    LoadedAtomisticIREE,
    save_atomistic_iree,
)
from phydrax.nn.atomistic import MACEArchitecture, MACELESPotential, MACEPotential
from phydrax.typing import PRNGKey
from phydrax.units import DALTON


def native_execution(
    scale: AtomisticScaleContract,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    key: PRNGKey,
    atomic_energies: Array,
) -> None:
    architecture = MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2),
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=2.0,
    )
    potential = MACEPotential(
        scale, architecture, atomic_energies=atomic_energies, key=key
    )
    assert_type(potential, MACEPotential)
    layout = AtomisticGraphLayout.for_batch(batch, execution, cutoff=3.0)
    assert_type(layout, AtomisticGraphLayout)
    payload = layout.prepare(batch)
    assert_type(payload, AtomisticGraphPayload)
    derivative = atomistic_energy_derivatives(potential, payload, batch.positions)
    assert_type(derivative, AtomisticEnergyDerivatives)
    loader = atomistic_training_loader(
        atomistic_training_source(batch, mass_unit=DALTON, energy=derivative.energy)
    )
    problem = AtomisticTrainingProblem(
        loader, execution, (AtomisticEnergyObjective(),), scale=scale, cutoff=3.0
    )
    assert_type(problem, AtomisticTrainingProblem)


def artifact_and_frozen_execution(
    model: MACEPotential,
    plan: NativeAtomisticProviderPlan,
    structure: AtomicStructure,
    units: AtomisticUnitSystem,
    path: Path,
) -> None:
    manifest = write_atomistic_model_artifact(path, model)
    assert_type(manifest, AtomisticModelArtifactManifest)
    restored = read_atomistic_model_artifact(
        path, numeric_revision_id=manifest.numeric_revision.revision_id
    )
    assert_type(restored, AtomisticModelArtifact)
    bundle = save_atomistic_iree(plan, structure, units, path)
    assert_type(bundle, AtomisticIREEExportBundle)
    executable = load_atomistic_iree(
        path,
        trusted_module_sha256=bundle.module_sha256,
        trusted_contract_id=bundle.contract.contract_id,
    )
    assert_type(executable, LoadedAtomisticIREE)


def source_conversion(source: MACESource, provider: MACEProviderRuntime) -> None:
    conversion = convert_mace_checkpoint(source, provider=provider)
    assert_type(conversion, MACECheckpointConversion)
    assert_type(conversion.potential, MACEPotential | MACELESPotential)
