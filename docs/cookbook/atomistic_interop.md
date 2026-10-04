# Atomistic interoperability

## Resumable trajectory output and rerun

Write accepted frames with a reporter, reopen the H5MD file, and rescore it with the same
prepared potential.

```python
import tempfile
from pathlib import Path

import jax.numpy as jnp
import phydrax as phx

system = phx.atomistic.AtomisticSystemPlan(
    [10, 20],
    [1, 1],
    [1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=[0, 0],
).prepare()
potential = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
).prepare(system)
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
    system.particles
)
frame = phx.atomistic.AtomisticFrame(
    0.0,
    0,
    jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
    system.plan.particle_ids,
    system_id=system.plan.system_id,
    topology_id=system.topology.topology_id,
    units=system.plan.units,
    source_id="interop-cookbook-frame",
)
with tempfile.TemporaryDirectory(prefix="phydrax-doc-") as directory:
    sink = phx.atomistic.interchange.H5MDTrajectoryPlan(
        Path(directory) / "trajectory.h5"
    )
    with sink.open(append=False) as writer:
        writer.write(frame)
    rerun = phx.atomistic.AtomisticRerunPlan(
        sink,
        potential,
        neighborhood,
    ).run()
    assert bool(rerun.successful)
```

`append=True` resumes at the committed frame boundary; it does not infer simulation state.
The artifact stores its complete unit descriptor once and rejects legacy ID-only
metadata. Resume dynamics from its atomistic checkpoint, then append frames whose
system, topology, and complete unit system match the existing stream.

For analysis selections, convert an MDAnalysis selection once into an
`AtomisticSelectionPlan` and store the stable selected IDs. Do not execute selection strings
inside compiled dynamics.

## Copy a structure from ASE without losing atom identity

Make particle identity explicit before atoms can be sliced or reordered. ASE carries the
ID array with each atom, while the adapter report carries source provenance. This
fragment requires the optional `ase` package and is not part of the core executable
documentation environment.

```text
import numpy as np
from ase import Atoms
import phydrax as phx
from phydrax.units import ANGSTROM, ELECTRONVOLT

scale = phx.atomistic.AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
source = Atoms(
    numbers=[14, 14],
    positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]],
    masses=[28.085, 28.085],
    cell=[[2.7, 0.0, 0.0], [0.0, 2.7, 0.0], [0.0, 0.0, 2.7]],
    pbc=True,
    info={phx.atomistic.interchange.ASE_SOURCE_ID_INFO: "relaxed-silicon"},
)
source.new_array(
    phx.atomistic.interchange.ASE_PARTICLE_ID_ARRAY,
    np.asarray([1001, 1002], dtype=np.int64),
)

structure, report = phx.atomistic.interchange.from_ase_atoms(source, scale)
phx.interchange.require_lossless(report)

# ASE slicing reorders the reserved ID array with atomic data.
reordered, reordered_report = phx.atomistic.interchange.from_ase_atoms(
    source[[1, 0]], scale
)
phx.interchange.require_lossless(reordered_report)
assert reordered.particle_ids.tolist() == [1002, 1001]

detached, export_report = phx.atomistic.interchange.to_ase_atoms(structure)
phx.interchange.require_lossless(export_report)
assert detached.calc is None
```

If the source does not contain `ASE_PARTICLE_ID_ARRAY`, import deliberately assigns IDs
`0, 1, ...` in the current atom order and reports a synthesized semantic. Carry the
returned report and call `require_lossless` when that default is unacceptable. Velocity,
constraint, charge, calculator, and arbitrary array or info content is likewise never
silently attached to `AtomicStructure`: it is either listed in the report or rejected.

## Serve one native model through its artifact, i-PI, ASE, and IREE

Persist a native MACE model as a pickle-free artifact, restore it with its numeric
revision pinned, and evaluate the same scalar energy natively, over an i-PI loopback,
through the ASE calculator, and as a frozen IREE executable. The model here is a small
randomly initialized native MACE, not a pretrained potential. The ASE and IREE routes
run only when those optional packages are installed. The same workflow is
`examples/atomistic_mace_deployment.py`.

```python
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
MASSES = [15.999, 1.008, 1.008]

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
model = phx.nn.atomistic.MACEPotential(
    UNITS.scale,
    architecture,
    atomic_energies=jnp.asarray([[-1.0, -2.0]], dtype=jnp.float64),
    key=jax.random.key(0),
)
structure = phx.atomistic.AtomicStructure(
    jnp.asarray(NUMBERS),
    jnp.asarray(POSITIONS),
    jnp.asarray(MASSES),
    UNITS.scale,
    cell=jnp.asarray(CELL),
    periodic_axes=jnp.asarray([True, True, True]),
)
system = phx.atomistic.AtomisticSystemPlan.from_structure(structure, UNITS).prepare()

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    manifest = phx.atomistic.write_atomistic_model_artifact(root / "model", model)
    artifact = phx.atomistic.read_atomistic_model_artifact(
        root / "model", numeric_revision_id=manifest.numeric_revision.revision_id
    )

    plan = phx.atomistic.NativeAtomisticProviderPlan(
        artifact.model,
        phx.atomistic.AtomisticGraphExecutionPlan(
            32,
            backend="particle",
            image_capacity=phx.discretization.ParticleImageCapacity(
                maximum_particles_per_cell=16,
                maximum_edges=1024,
                maximum_degree=64,
                maximum_images=343,
            ),
        ),
        finite_neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(256),
        skin=0.4,
        deformation_margin=0.1,
    )
    provider = plan.prepare(system)
    native = provider.evaluate_state(jnp.asarray(POSITIONS), None, None)
    if not bool(native.evaluation.successful):
        raise RuntimeError("Native MACE evaluation failed.")
    print("native energy [eV]:", float(native.evaluation.energy))
    print("native tensile stress [eV/Ang^3]:\n", np.asarray(native.evaluation.stress))

    # i-PI: Hartree atomic units on the wire, virial W = -V * stress in the reply.
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
            transported = remote.evaluate(system, jnp.asarray(POSITIONS), None)
        future.result(timeout=120.0)
    listener.close()
    print(
        "i-PI energy difference [eV]:",
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
        print("frozen status:", frozen.status.name, "energy [eV]:", float(frozen.energy))
```

The default `virial="required"` suits this fully periodic system. A finite molecule has
no stress, so serve it with `IPITransportPlan.unix(path, virial="optional")`, as in
`examples/atomistic_ipi.py`; the reply then carries an all-NaN virial rather than a
zero. Peers that send a transposed legacy inverse cell, such as ASE's socket server,
need `inverse_cell="ignore"`. The frozen IREE export is an unreleased candidate whose
float64 MACE parity is not claimed.

## Convert an admitted mace-torch checkpoint

Conversion executes the source provider (mace-torch 0.3.16, e3nn 0.4.4, and a declared
torch release) only in a separately installed interpreter that you pin; Phydrax never
imports it and downloads nothing. This recipe asks that provider for a tiny deterministic
safe state-dict fixture, admits its bytes, converts it, compares native and
provider-side E/F/S, and writes the pickle-free native artifact. The same workflow, with
a command-line option for a local checkpoint, is `examples/mace_checkpoint_conversion.py`.

```python
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.artifacts import (
    admit_external_artifact,
    ArtifactManifest,
    ExternalArtifactPolicy,
)
from phydrax.interchange import pin_executable


PROVIDER_VENV = Path(".tmp/mace-provider/.venv")
DECLARATION = {
    "model_class": "ScaleShiftMACE",
    "r_max": 4.5,
    "num_bessel": 6,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "interaction_classes": ["RealAgnosticResidualInteractionBlock"],
    "num_interactions": 1,
    "atomic_numbers": [1, 8],
    "hidden_irreps": "8x0e",
    "MLP_irreps": None,
    "avg_num_neighbors": 2.5,
    "correlation": [3],
    "gate": None,
    "pair_repulsion": True,
    "distance_transform": "Agnesi",
    "radial_MLP": [16, 16],
    "radial_type": "bessel",
    "heads": ["Default"],
    "apply_cutoff": True,
    "use_reduced_cg": False,
    "use_agnostic_product": False,
    "use_last_readout_only": False,
    "atomic_energies": [-13.6, -2041.8],
    "atomic_inter_scale": 1.3,
    "atomic_inter_shift": 0.05,
}
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
CELL = np.array([[4.6, 0.0, 0.0], [0.9, 4.4, 0.0], [0.5, -0.6, 4.8]])

provider = phx.atomistic.interchange.MACEProviderRuntime(
    pin_executable(
        PROVIDER_VENV / "bin" / "python", version="3.12.8", license_id="PSF-2.0"
    ),
    str(PROVIDER_VENV / "lib" / "python3.12" / "site-packages"),
    "2.14.1",
)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    fixture = phx.atomistic.interchange.create_mace_provider_fixture(
        DECLARATION, root / "fixture", provider=provider, seed=20261003
    )
    policy = ExternalArtifactPolicy(
        fixture.path.parent,
        maximum_bytes=4 * 1024**3,
        allowed_license_ids=("MIT",),
        allowed_suffixes=(".model", ".pt"),
    )
    manifest = ArtifactManifest(
        artifact_id=fixture.path.stem,
        producer="mace-torch",
        version="0.3.16",
        sha256=fixture.sha256,
        byte_size=fixture.path.stat().st_size,
        source_uri=fixture.path.resolve().as_uri(),
        license_id="MIT",
        model=fixture.path.stem,
        coverage="caller-supplied local checkpoint",
    )
    source = phx.atomistic.interchange.MACESource(
        admit_external_artifact(fixture.path.name, manifest, policy=policy),
        manifest,
        policy,
        "torch-state-dict",
        architecture=dict(fixture.declaration),
    )

    conversion = phx.atomistic.interchange.convert_mace_checkpoint(
        source, provider=provider
    )
    native = conversion.potential

    configuration = phx.atomistic.interchange.MACEProviderConfiguration(
        (8, 1, 1), WATER, CELL, True
    )
    oracle = phx.atomistic.interchange.evaluate_mace_source(
        source, [configuration], provider=provider
    ).cases[0]
    structure = phx.atomistic.AtomicStructure(
        jnp.asarray([8, 1, 1]),
        jnp.asarray(WATER),
        jnp.asarray([15.999, 1.008, 1.008]),
        native.scale,
        cell=jnp.asarray(CELL),
        periodic_axes=jnp.asarray([True, True, True]),
    )
    execution = phx.atomistic.AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=4096,
            maximum_degree=128,
            maximum_images=343,
        ),
    )
    prediction = phx.atomistic.energy_and_forces(
        native, structure, execution, compute_stress=True
    )
    print("source energy [eV]:", oracle.energy)
    print("native energy [eV]:", float(prediction.energy[0]))
    print(
        "max |dF| [eV/Ang]:",
        float(np.max(np.abs(np.asarray(prediction.forces[0]) - oracle.forces))),
    )
    print(
        "max |dS| [eV/Ang^3]:",
        float(np.max(np.abs(np.asarray(prediction.stress[0]) - oracle.stress))),
    )

    model_manifest = phx.atomistic.write_atomistic_model_artifact(
        root / "native-model", native, source=conversion.provenance
    )
    print("native artifact:", model_manifest.artifact_id)
```

A state-dict source is restricted pickle metadata plus tensor storage, so it
needs its exact provider `architecture` declaration. Storage and tensor bounds
are preflighted before `torch.load(weights_only=True)`. A full-object `.model`
checkpoint is an executable pickle. Admit it with its out-of-band SHA-256 and
actual license, and authorize deserialization for exactly that digest; there
is no unsafe fallback. A multihead source must name its head. The provider
subprocess is neither a security nor a memory sandbox.

```text
source = phx.atomistic.interchange.MACESource(
    admit_external_artifact(checkpoint.name, manifest, policy=policy),
    manifest,
    policy,
    "torch-full-model",
    trust=phx.atomistic.interchange.TrustedTorchPickleSource(
        manifest.sha256, "Published by the model authors; digest verified out of band."
    ),
)
conversion = phx.atomistic.interchange.convert_mace_checkpoint(
    source, provider=provider, head="Default"
)
phx.atomistic.write_atomistic_model_artifact(
    "native-model",
    conversion.potential,
    source=conversion.provenance,
    licenses=(manifest.license_id,),
)
```

The written artifact is restored with `read_atomistic_model_artifact` and needs no
provider for inference, training, i-PI/ASE serving, or a fresh restart. Phydrax bundles
no checkpoint weights. MIT-licensed MACE-MP/MPA weights and ASL-licensed OMAT, MH, and
MACE-OFF weights carry their own terms; see the
[mace-foundations README](https://github.com/ACEsuit/mace-foundations) and the
[mace-off README](https://github.com/ACEsuit/mace-off). Converting a foundation
checkpoint is not a fidelity qualification of that model.
