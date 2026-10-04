"""Admitted MACE sources, the pinned provider runtime, and native evaluation.

The provider (mace-torch, e3nn, torch) is never a Phydrax dependency. Tests
reach it only through an explicitly pinned interpreter declared by
``PHYDRAX_MACE_PROVIDER_PYTHON``, ``PHYDRAX_MACE_PROVIDER_PYTHON_VERSION``,
``PHYDRAX_MACE_PROVIDER_SITE_PACKAGES`` and ``PHYDRAX_MACE_PROVIDER_TORCH_VERSION``;
an unset declaration is the single canonical skip boundary, and every failure
of a declared provider fails. Provider-built fixtures are lawful deterministic
test oracles, not qualified pretrained models.
"""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.artifacts import (
    admit_external_artifact,
    ArtifactManifest,
    ExternalArtifactPolicy,
)
from phydrax.atomistic.interchange import MACESourceKind
from phydrax.interchange import pin_executable


PROVIDER_ENVIRONMENT = (
    "PHYDRAX_MACE_PROVIDER_PYTHON",
    "PHYDRAX_MACE_PROVIDER_PYTHON_VERSION",
    "PHYDRAX_MACE_PROVIDER_SITE_PACKAGES",
    "PHYDRAX_MACE_PROVIDER_TORCH_VERSION",
)
# Optional: the pinned cuequivariance release of the same provider, required
# exactly when installed there and for reduced generalized-CG sources.
CUEQUIVARIANCE_ENVIRONMENT = "PHYDRAX_MACE_PROVIDER_CUEQUIVARIANCE_VERSION"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mace_torch_0_3_16"

# Standard layouts admitted by the converter. Each row exercises distinct
# provider semantics; random provider initialization supplies the weights.
# Provider architecture declarations are JSON records, typed like the
# converter's own ``Mapping[str, Any]`` declaration boundary.
_SHARED: dict[str, Any] = {
    "r_max": 3.2,
    "num_bessel": 6,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "avg_num_neighbors": 2.5,
    "radial_MLP": [8, 8],
    "radial_type": "bessel",
    "use_reduced_cg": False,
    "use_agnostic_product": False,
    "use_last_readout_only": False,
}
ONE_LAYER_INVARIANT_READOUT: dict[str, Any] = {
    **_SHARED,
    "model_class": "ScaleShiftMACE",
    "interaction_classes": ["RealAgnosticResidualInteractionBlock"],
    "num_interactions": 1,
    "atomic_numbers": [1, 8],
    "hidden_irreps": "8x0e",
    "MLP_irreps": None,
    "correlation": [3],
    "gate": None,
    "pair_repulsion": True,
    "distance_transform": "Agnesi",
    "heads": ["Default"],
    "apply_cutoff": True,
    "atomic_energies": [-13.6, -2041.8],
    "atomic_inter_scale": 1.3,
    "atomic_inter_shift": 0.05,
}
TWO_LAYER_DENSITY_MULTIHEAD: dict[str, Any] = {
    **_SHARED,
    "model_class": "MACE",
    "interaction_classes": [
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
    ],
    "num_interactions": 2,
    "atomic_numbers": [1, 6, 8],
    "hidden_irreps": "4x0e+4x1o",
    "MLP_irreps": "4x0e",
    "correlation": [2, 2],
    "gate": "silu",
    "pair_repulsion": False,
    "distance_transform": "None",
    "heads": ["pbe", "r2scan"],
    "apply_cutoff": False,
    "atomic_energies": [[-13.6, -1029.8, -2041.8], [-13.5, -1030.4, -2042.6]],
}
TWO_LAYER_RESIDUAL_AGNESI_ZBL: dict[str, Any] = {
    **_SHARED,
    "model_class": "ScaleShiftMACE",
    "interaction_classes": [
        "RealAgnosticInteractionBlock",
        "RealAgnosticResidualInteractionBlock",
    ],
    "num_interactions": 2,
    "atomic_numbers": [1, 8],
    "hidden_irreps": "4x0e+4x1o+4x2e",
    "MLP_irreps": "4x0e",
    "correlation": [3, 2],
    "gate": "silu",
    "pair_repulsion": True,
    "distance_transform": "Agnesi",
    "heads": ["Default"],
    "apply_cutoff": True,
    "atomic_energies": [-13.6, -2041.8],
    "atomic_inter_scale": 0.8,
    "atomic_inter_shift": -0.02,
}
# The same layout with the reduced generalized-CG U basis (fewer product
# parameters), built by the provider through its cuequivariance construction.
TWO_LAYER_REDUCED_CG: dict[str, Any] = {
    **TWO_LAYER_RESIDUAL_AGNESI_ZBL,
    "use_reduced_cg": True,
}
DECLARATIONS: dict[str, dict[str, Any]] = {
    "one-layer-invariant-readout": ONE_LAYER_INVARIANT_READOUT,
    "two-layer-density-multihead": TWO_LAYER_DENSITY_MULTIHEAD,
    "two-layer-residual-agnesi-zbl": TWO_LAYER_RESIDUAL_AGNESI_ZBL,
}


def provider_runtime() -> Any:
    """The declared pinned provider, or the canonical skip when undeclared."""

    python, version, site_packages, torch_version = (
        os.environ.get(name) for name in PROVIDER_ENVIRONMENT
    )
    if (
        python is None
        or version is None
        or site_packages is None
        or torch_version is None
    ):
        pytest.skip(
            "declare " + ", ".join(PROVIDER_ENVIRONMENT) + " for the pinned "
            "mace-torch 0.3.16 / e3nn 0.4.4 provider interpreter"
        )
    return phx.atomistic.interchange.MACEProviderRuntime(
        pin_executable(python, version=version, license_id="PSF-2.0"),
        site_packages,
        torch_version,
        cuequivariance_version=os.environ.get(CUEQUIVARIANCE_ENVIRONMENT),
    )


def reduced_cg_provider(provider: Any) -> Any:
    """The provider when it declares cuequivariance; otherwise the skip boundary."""

    if provider.cuequivariance_version is None:
        pytest.skip(f"declare {CUEQUIVARIANCE_ENVIRONMENT} for reduced-CG U bases")
    return provider


def admitted_source(
    path: Path,
    kind: MACESourceKind,
    *,
    sha256: str | None = None,
    license_id: str = "MIT",
    allowed_licenses: tuple[str, ...] = ("MIT",),
    trust: str | None = None,
    architecture: Mapping[str, Any] | None = None,
) -> Any:
    """Admit local source bytes under an exact digest/size/license policy."""

    digest = hashlib.sha256(path.read_bytes()).hexdigest() if sha256 is None else sha256
    policy = ExternalArtifactPolicy(
        path.parent,
        maximum_bytes=1 << 31,
        allowed_license_ids=allowed_licenses,
        allowed_suffixes=(path.suffix,),
    )
    manifest = ArtifactManifest(
        artifact_id=path.stem,
        producer="mace-torch",
        version="0.3.16",
        sha256=digest,
        byte_size=path.stat().st_size,
        source_uri=path.resolve().as_uri(),
        license_id=license_id,
        model="mace",
        coverage="provider-built test fixture",
    )
    artifact = admit_external_artifact(path.name, manifest, policy=policy)
    return phx.atomistic.interchange.MACESource(
        artifact,
        manifest,
        policy,
        kind,
        trust=None
        if trust is None
        else phx.atomistic.interchange.TrustedTorchPickleSource(digest, trust),
        architecture=architecture,
    )


def configurations(species: tuple[int, ...]) -> tuple[Any, Any]:
    """One finite and one triclinic periodic configuration of ``species``."""

    finite = np.array(
        [[0.0, 0.0, 0.0], [0.96, 0.08, -0.04], [-0.27, 0.91, 0.18], [1.9, 0.5, 0.35]]
    )
    cell = np.array([[3.6, 0.0, 0.0], [0.5, 3.3, 0.0], [0.3, 0.4, 3.8]])
    periodic = np.array([[0.1, 0.2, 0.1], [1.05, 0.35, 0.2], [0.3, 1.2, 0.55]])
    numbers = tuple(species[index % len(species)] for index in range(4))
    return (
        phx.atomistic.interchange.MACEProviderConfiguration(
            numbers, finite, np.zeros((3, 3)), False
        ),
        phx.atomistic.interchange.MACEProviderConfiguration(
            numbers[:3], periodic, cell, True
        ),
    )


def execution(configuration: Any) -> Any:
    atoms = len(configuration.numbers)
    if not configuration.periodic:
        return phx.atomistic.AtomisticGraphExecutionPlan(atoms, maximum_dense_atoms=atoms)
    return phx.atomistic.AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=4096,
            maximum_degree=512,
            maximum_images=343,
        ),
    )


def structure(potential: Any, configuration: Any) -> Any:
    """The configuration at the potential's declared coordinate precision."""

    dtype = potential.precision.coordinate_dtype
    masses = jnp.ones((len(configuration.numbers),), dtype=dtype)
    positions = jnp.asarray(configuration.positions, dtype=dtype)
    if not configuration.periodic:
        return phx.atomistic.AtomicStructure(
            jnp.asarray(configuration.numbers), positions, masses, potential.scale
        )
    return phx.atomistic.AtomicStructure(
        jnp.asarray(configuration.numbers),
        positions,
        masses,
        potential.scale,
        cell=jnp.asarray(configuration.cell, dtype=dtype),
        periodic_axes=jnp.asarray([True, True, True]),
    )


def native_case(potential: Any, configuration: Any) -> dict[str, np.ndarray]:
    """Native total energy, forces and (periodic) stress of one configuration."""

    prediction = phx.atomistic.energy_and_forces(
        potential,
        structure(potential, configuration),
        execution(configuration),
        compute_stress=configuration.periodic,
    )
    assert bool(np.asarray(prediction.successful).all())
    atoms = len(configuration.numbers)
    result = {
        "energy": np.asarray(prediction.energy)[0],
        "forces": np.asarray(prediction.forces)[0, :atoms],
    }
    if configuration.periodic:
        result["stress"] = np.asarray(prediction.stress)[0]
    return result


# Frozen float64 gates (absolute, relative) for an independent provider oracle.
FLOAT64_GATES = {
    "energy": (1.0e-9, 1.0e-12),
    "forces": (1.0e-8, 1.0e-10),
    "stress": (1.0e-9, 1.0e-10),
}
FLOAT32_GATES = {
    "energy": (1.0e-4, 1.0e-6),
    "forces": (1.0e-4, 1.0e-5),
    "stress": (1.0e-5, 1.0e-5),
}


def within(observed: Any, reference: Any, gate: tuple[float, float]) -> bool:
    absolute, relative = gate
    difference = np.abs(
        np.asarray(observed, np.float64) - np.asarray(reference, np.float64)
    )
    return bool(np.all(difference <= absolute + relative * np.abs(reference)))


def assert_matches_oracle(
    potential: Any, configurations_: Any, oracle: Any, gates: Mapping[str, Any]
) -> None:
    for configuration, case in zip(configurations_, oracle.cases, strict=True):
        native = native_case(potential, configuration)
        assert within(native["energy"], case.energy, gates["energy"]), (
            native["energy"],
            case.energy,
        )
        assert within(native["forces"], case.forces, gates["forces"])
        if configuration.periodic:
            assert within(native["stress"], case.stress, gates["stress"])


def committed_source_potential(case: str = "two_scale_shift") -> Any:
    """A native model reconstructed from a committed provider-built fixture.

    The npz holds the provider state dictionary, its Wigner 3j tables and
    sampled source harmonics (``tests/fixtures/mace_torch_0_3_16``); no
    provider is needed to rebuild the source parameterization.
    """

    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    declaration = manifest["cases"][case]
    fixture = np.load(FIXTURES / f"{case}.npz")
    kinds = {
        "RealAgnosticInteractionBlock": "real-agnostic",
        "RealAgnosticResidualInteractionBlock": "real-agnostic-residual",
        "RealAgnosticDensityInteractionBlock": "real-agnostic-density",
        "RealAgnosticDensityResidualInteractionBlock": "real-agnostic-density-residual",
    }
    count = declaration["interactions"]
    gate = declaration["normalize2mom_silu"]
    agnesi = declaration["distance_transform"] == "Agnesi"
    architecture = phx.nn.atomistic.MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=(kinds[declaration["first"]],)
        + (kinds[declaration["rest"]],) * (count - 1),
        correlations=tuple(declaration["correlation"]),
        radial_widths=(8,),
        readout_width=5 if count > 1 else None,
        radial_activation_scale=gate,
        readout_activation_scale=gate if count > 1 else None,
        average_neighbor_count=2.0,
        heads=tuple(declaration["heads"]),
        head=declaration["heads"][0],
        energy_scaling="scale-shift"
        if declaration["class"] == "ScaleShiftMACE"
        else "unscaled",
        pair_repulsion=declaration["pair_repulsion"],
        distance_transform="agnesi" if agnesi else "none",
        agnesi_parameters=(1.0805, 0.9183, 4.5791) if agnesi else None,
        cutoff_placement="embedding" if declaration["apply_cutoff"] else "weights",
    )
    native = np.asarray(
        phx.special.RealCartesianHarmonics(2, normalization="fully_normalized")(
            fixture["harmonics/vectors"]
        )
    )
    source = fixture["harmonics/values"]
    transforms = []
    for degree in range(3):
        block = slice(degree * degree, (degree + 1) ** 2)
        solution, *_ = np.linalg.lstsq(source[:, block], native[:, block], rcond=None)
        transforms.append(solution.T)
    couplings: dict[tuple[int, int, int], np.ndarray] = {}
    for name in fixture.files:
        if name.startswith("w3j/"):
            first, second, output = (int(value) for value in name[4:].split("_"))
            couplings[first, second, output] = fixture[name]
    dropped = ("output_mask", "zeroed", "num_interactions", "_w3j")
    tensors = {
        name.removeprefix("state/"): fixture[name]
        for name in fixture.files
        if name.startswith("state/")
        and fixture[name].size
        and not any(marker in name for marker in dropped)
    }
    return phx.nn.atomistic.mace_potential_from_source(
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond().scale,
        architecture,
        tensors,
        degree_transforms=transforms,
        couplings=couplings,
        source_id=f"mace-torch-0.3.16-fixture:{case}",
    )


WATER_CELL = np.array([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]])
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
WATER_NUMBERS = (8, 1, 1)
WATER_MASSES = (15.999, 1.008, 1.008)


def water_system() -> Any:
    from tests._support.mace_deployment import electronvolt_units

    units = electronvolt_units()
    structure_ = phx.atomistic.AtomicStructure(
        jnp.asarray(WATER_NUMBERS),
        jnp.asarray(WATER),
        jnp.asarray(WATER_MASSES),
        units.scale,
        cell=jnp.asarray(WATER_CELL),
        periodic_axes=jnp.asarray([True, True, True]),
    )
    return phx.atomistic.AtomisticSystemPlan.from_structure(structure_, units).prepare()


def nvt_dynamics(
    model: Any,
    system: Any,
    *,
    temperature: float = 300.0,
    skin: float = 0.4,
) -> tuple[Any, Any]:
    """Fixed-cell BAOAB Langevin NVT over the native provider recipe."""

    from tests._support.mace_deployment import provider_plan

    provider = provider_plan(model, skin=skin).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        provider.program,
        provider.neighborhood,
        phx.atomistic.BAOABLangevinPlan(0.25, 0.02),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="nvt",
        temperature=temperature,
    ).prepare(dynamics)
    return dynamics, thermodynamic


def water_template(dynamics: Any, thermodynamic: Any) -> Any:
    """Structural state template; restart replaces every leaf."""

    import jax

    return dynamics.initialize_state(
        jnp.asarray(WATER),
        thermodynamic,
        velocity=jnp.zeros((3, 3)),
        key=jax.random.key(0),
    )


def water_training_problem(*, label_shift: float = 0.0) -> Any:
    """Periodic E/F/S labels of a fixed teacher on strained cells.

    Two strains train; a third validates. ``label_shift`` alters the energy
    targets (a different problem with otherwise identical structure).
    """

    from tests._support.mace_deployment import (
        electronvolt_units,
        graph_execution,
        tiny_mace,
    )

    def batch(magnitudes: tuple[float, ...]) -> Any:
        structures = []
        for magnitude in magnitudes:
            deformation = np.eye(3) + magnitude * np.array(
                [[1.0, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, 0.7]]
            )
            structures.append(
                phx.atomistic.AtomicStructure(
                    jnp.asarray(WATER_NUMBERS),
                    jnp.asarray(WATER @ deformation.T),
                    jnp.asarray(WATER_MASSES),
                    electronvolt_units().scale,
                    cell=jnp.asarray(WATER_CELL @ deformation.T),
                    periodic_axes=jnp.asarray([True, True, True]),
                )
            )
        return phx.atomistic.AtomisticBatch.from_structures(structures)

    teacher = tiny_mace(seed=11)
    training, validation = batch((-0.015, 0.02)), batch((0.005,))
    labels = phx.atomistic.energy_and_forces(
        teacher, training, graph_execution(), compute_stress=True
    )
    held_out = phx.atomistic.energy_and_forces(
        teacher, validation, graph_execution(), compute_stress=True
    )
    return phx.atomistic.AtomisticTrainingProblem(
        training,
        graph_execution(),
        cutoff=3.0,
        training_energy=labels.energy + label_shift,
        training_forces=labels.forces,
        training_stress=labels.stress,
        validation_batch=validation,
        validation_energy=held_out.energy,
        validation_forces=held_out.forces,
        validation_stress=held_out.stress,
    )
