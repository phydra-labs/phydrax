"""Deploy one native MACE model through its artifact, ASE, i-PI, and frozen IREE.

The model is a small randomly initialized native MACE, not a pretrained
potential. Every route evaluates the same scalar energy: the pickle-free
artifact restores the model, ``NativeAtomisticProvider`` serves it to i-PI in
Hartree atomic units with the configurational virial ``W = -V * stress``, the
ASE calculator reports eV/Angstrom properties, and the frozen IREE executable
consumes a host-prepared candidate graph. Optional ASE and IREE routes run only
when those packages are installed; nothing is downloaded.
"""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.backends.iree import iree_availability


UNITS = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
POSITIONS = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
CELL = np.array([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]])
NUMBERS = np.array([8, 1, 1])


def build_model() -> phx.nn.atomistic.MACEPotential:
    architecture = phx.nn.atomistic.MACEArchitecture(
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
    return phx.nn.atomistic.MACEPotential(
        UNITS.scale,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        key=jax.random.key(0),
    )


def provider_plan(
    model: phx.nn.atomistic.MACEPotential,
) -> phx.atomistic.NativeAtomisticProviderPlan:
    execution = phx.atomistic.AtomisticGraphExecutionPlan(
        32,
        backend="particle",
        image_capacity=phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=1024,
            maximum_degree=64,
            maximum_images=343,
        ),
    )
    return phx.atomistic.NativeAtomisticProviderPlan(
        model,
        execution,
        finite_neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(256),
        skin=0.4,
        deformation_margin=0.1,
    )


def periodic_system() -> phx.atomistic.PreparedAtomisticSystem:
    structure = phx.atomistic.AtomicStructure(
        jnp.asarray(NUMBERS),
        jnp.asarray(POSITIONS),
        jnp.asarray([15.999, 1.008, 1.008]),
        UNITS.scale,
        cell=jnp.asarray(CELL),
        periodic_axes=jnp.asarray([True, True, True]),
    )
    return phx.atomistic.AtomisticSystemPlan.from_structure(structure, UNITS).prepare()


def serve_over_ipi(
    provider: phx.atomistic.NativeAtomisticProvider,
    system: phx.atomistic.PreparedAtomisticSystem,
) -> phx.atomistic.ExternalAtomisticEvaluation:
    socket_path = os.path.join(tempfile.gettempdir(), f"phydrax-mace-{os.getpid()}.sock")
    transport = phx.atomistic.interchange.IPITransportPlan.unix(socket_path, timeout=60.0)
    listener = transport.listen()

    def serve() -> phx.atomistic.interchange.IPITransportStatus:
        with listener.accept() as session:
            return phx.atomistic.interchange.serve_ipi_once(session, provider, system)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve)
        with transport.connect() as session:
            remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
                session, "ipi-loopback"
            )
            result = remote.evaluate(system, jnp.asarray(POSITIONS), None)
        future.result(timeout=120.0)
    listener.close()
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = build_model()
        manifest = phx.atomistic.write_atomistic_model_artifact(root / "model", model)
        artifact = phx.atomistic.read_atomistic_model_artifact(
            root / "model", numeric_revision_id=manifest.numeric_revision.revision_id
        )
        plan = provider_plan(artifact.model)
        system = periodic_system()
        provider = plan.prepare(system)
        native = provider.evaluate_state(jnp.asarray(POSITIONS), None, None)
        if not bool(native.evaluation.successful):
            raise RuntimeError("Native MACE evaluation failed.")
        print("native energy [eV]:", float(native.evaluation.energy))
        print("native stress [eV/Ang^3]:\n", np.asarray(native.evaluation.stress))

        transported = serve_over_ipi(provider, system)
        print(
            "i-PI loopback energy difference [eV]:",
            float(transported.energy - native.evaluation.energy),
        )

        if phx.atomistic.interchange.is_ase_available():
            import ase

            atoms = ase.Atoms(numbers=NUMBERS, positions=POSITIONS, cell=CELL, pbc=True)
            atoms.calc = phx.atomistic.interchange.NativeASECalculator(
                phx.atomistic.interchange.NativeASECalculatorPlan(plan, UNITS),
                artifact_id=artifact.manifest.artifact_id,
            )
            print("ASE energy [eV]:", atoms.get_potential_energy())
            print("ASE Voigt stress [eV/Ang^3]:", atoms.get_stress())

        if iree_availability().available:
            structure = phx.atomistic.AtomicStructure(
                jnp.asarray(NUMBERS),
                jnp.asarray(POSITIONS),
                jnp.asarray([15.999, 1.008, 1.008]),
                UNITS.scale,
                cell=jnp.asarray(CELL),
                periodic_axes=jnp.asarray([True, True, True]),
            )
            # Float64 MACE needs the host C math library: a host-specific
            # system-library executable linked by the host system linker.
            bundle = phx.export.save_atomistic_iree(
                plan,
                structure,
                UNITS,
                root / "water-mace.phxiree",
                policy=phx.export.IREEExportPolicy(executable_format="system-library"),
            )
            frozen = phx.export.load_atomistic_iree(
                bundle.path,
                trusted_module_sha256=bundle.module_sha256,
                trusted_contract_id=bundle.contract.contract_id,
            )(native.request)
            print(
                "frozen IREE status:",
                frozen.status.name,
                "energy [eV]:",
                float(frozen.energy),
            )


if __name__ == "__main__":
    main()
