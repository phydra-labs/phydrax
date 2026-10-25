"""Admitted MACE sources, the pinned provider runtime, and native evaluation.

The provider (mace-torch, e3nn, torch, and LES for MACELES) is never a Phydrax
dependency. Tests reach it only through an explicitly pinned interpreter
declared by ``PHYDRAX_MACE_PROVIDER_PYTHON``,
``PHYDRAX_MACE_PROVIDER_PYTHON_VERSION``, ``PHYDRAX_MACE_PROVIDER_SITE_PACKAGES``,
``PHYDRAX_MACE_PROVIDER_TORCH_VERSION`` and
``PHYDRAX_MACE_PROVIDER_MACE_TORCH_VERSION``; an unset declaration (or a
provider of another mace-torch release than a test needs) is the single
canonical skip boundary, and every failure of a declared provider fails.
Provider-built fixtures are lawful deterministic test oracles, not qualified
pretrained models.
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
from phydrax.nn.atomistic import (
    MACEArchitecture,
    MACEDistanceTransform,
    MACEInteractionKind,
)


PROVIDER_ENVIRONMENT = (
    "PHYDRAX_MACE_PROVIDER_PYTHON",
    "PHYDRAX_MACE_PROVIDER_PYTHON_VERSION",
    "PHYDRAX_MACE_PROVIDER_SITE_PACKAGES",
    "PHYDRAX_MACE_PROVIDER_TORCH_VERSION",
    "PHYDRAX_MACE_PROVIDER_MACE_TORCH_VERSION",
)
# Optional releases of the same provider, each required exactly when installed
# there: cuequivariance (reduced generalized-CG U bases, cuEquivariance
# layouts), OpenEquivariance (its layouts) and LES (MACELES sources).
CUEQUIVARIANCE_ENVIRONMENT = "PHYDRAX_MACE_PROVIDER_CUEQUIVARIANCE_VERSION"
OPENEQUIVARIANCE_ENVIRONMENT = "PHYDRAX_MACE_PROVIDER_OPENEQUIVARIANCE_VERSION"
LES_ENVIRONMENT = "PHYDRAX_MACE_PROVIDER_LES_VERSION"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mace_torch_0_3_16"
RELEASE_FIXTURES = {
    "0.3.15": FIXTURES.parent / "mace_torch_0_3_15",
    "0.3.16": FIXTURES,
}

# Layouts admitted by the converter. Each row exercises distinct provider
# semantics; random provider initialization supplies the weights. Provider
# architecture declarations are JSON records, typed like the converter's own
# ``Mapping[str, Any]`` declaration boundary, extracted by mace-torch 0.3.16.
_SHARED: dict[str, Any] = {
    "mace_torch_release": "0.3.16",
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
    "use_so3": False,
    "edge_irreps": None,
    "use_edge_irreps_first": False,
    "embedding_specs": None,
    "use_embedding_readout": False,
    "keep_last_layer_irreps": False,
}
ONE_LAYER_INVARIANT_READOUT: dict[str, Any] = {
    **_SHARED,
    "model_class": "ScaleShiftMACE",
    "interaction_classes": ["RealAgnosticResidualInteractionBlock"],
    "num_interactions": 1,
    "atomic_numbers": [1, 8],
    "hidden_irreps": "8x0e",
    "MLP_irreps": None,
    "readout_class": None,
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
    "readout_class": "NonLinearReadoutBlock",
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
    "readout_class": "NonLinearReadoutBlock",
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

# One tiny declaration per admitted family beyond the standard rows.
_FAMILY: dict[str, Any] = {
    **_SHARED,
    "model_class": "ScaleShiftMACE",
    "interaction_classes": [
        "RealAgnosticInteractionBlock",
        "RealAgnosticResidualInteractionBlock",
    ],
    "num_interactions": 2,
    "atomic_numbers": [1, 8],
    "hidden_irreps": "4x0e+4x1o",
    "MLP_irreps": "4x0e",
    "readout_class": "NonLinearReadoutBlock",
    "correlation": [2, 2],
    "gate": "silu",
    "pair_repulsion": False,
    "distance_transform": "None",
    "heads": ["Default"],
    "apply_cutoff": True,
    "atomic_energies": [-13.6, -2041.8],
    "atomic_inter_scale": 0.9,
    "atomic_inter_shift": 0.03,
}
# The complete ``les_arguments`` MACELES records (``Les`` defaults).
LES_ARGUMENTS: dict[str, Any] = {
    "n_layers": 3,
    "n_hidden": [32, 16],
    "add_linear_nn": True,
    "output_scaling_factor": 0.1,
    "sigma": 1.0,
    "dl": 2.0,
    "remove_self_interaction": True,
    "is_periodic": None,
    "N_max": 10,
    "remove_mean": True,
    "epsilon_factor": 1.0,
    "use_atomwise": False,
    "use_fixed_atomic_charges": False,
    "fixed_atomic_charges_scaling_factor": 0.5,
    "use_atomic_alpha": False,
    "use_epsilon_r_scaling": False,
    "compute_bec": False,
    "bec_output_index": None,
}
CHARGE_SPIN_SPECS: dict[str, Any] = {
    "total_charge": {
        "type": "categorical",
        "per": "graph",
        "num_classes": 5,
        "emb_dim": 3,
        "offset": 2,
    },
    "total_spin": {
        "type": "categorical",
        "per": "graph",
        "num_classes": 4,
        "emb_dim": 2,
        "offset": 0,
    },
}
FAMILY_DECLARATIONS: dict[str, dict[str, Any]] = {
    "attention": {
        **_FAMILY,
        "interaction_classes": ["RealAgnosticAttResidualInteractionBlock"] * 2,
    },
    "nonlinear-bias-readout": {
        **_FAMILY,
        "interaction_classes": ["RealAgnosticResidualNonLinearInteractionBlock"] * 2,
        "readout_class": "NonLinearBiasReadoutBlock",
    },
    "gaussian-soft": {
        **_FAMILY,
        "radial_type": "gaussian",
        "distance_transform": "Soft",
    },
    "chebyshev-agnesi": {
        **_FAMILY,
        "num_bessel": 4,
        "radial_type": "chebyshev",
        "distance_transform": "Agnesi",
    },
    "so3": {**_FAMILY, "use_so3": True, "hidden_irreps": "4x0e+4x1e"},
    "edge-irreps-first": {
        **_FAMILY,
        "edge_irreps": "3x0e+3x1o",
        "use_edge_irreps_first": True,
    },
    "keep-last-layer-irreps": {**_FAMILY, "keep_last_layer_irreps": True},
    "joint-embedding-readout": {
        **_FAMILY,
        "embedding_specs": CHARGE_SPIN_SPECS,
        "use_embedding_readout": True,
    },
    "maceles": {**_FAMILY, "model_class": "MACELES", "les_arguments": LES_ARGUMENTS},
}


def for_release(declaration: Mapping[str, Any], release: str) -> dict[str, Any]:
    """The same declaration as extracted by another admitted mace-torch release.

    0.3.15 has no ``keep_last_layer_irreps``; only its default is expressible.
    """

    converted = {**declaration, "mace_torch_release": release}
    if release == "0.3.15":
        if converted.pop("keep_last_layer_irreps", False):
            raise ValueError("mace-torch 0.3.15 cannot keep last-layer irreps.")
    else:
        converted.setdefault("keep_last_layer_irreps", False)
    return converted


def provider_runtime(release: str = "0.3.16") -> Any:
    """The declared pinned provider of ``release``, or the canonical skip."""

    python, version, site_packages, torch_version, mace_version = (
        os.environ.get(name) for name in PROVIDER_ENVIRONMENT
    )
    if (
        python is None
        or version is None
        or site_packages is None
        or torch_version is None
        or mace_version is None
    ):
        pytest.skip(
            "declare " + ", ".join(PROVIDER_ENVIRONMENT) + " for the pinned "
            f"mace-torch {release} / e3nn 0.4.4 provider interpreter"
        )
    if mace_version != release:
        pytest.skip(f"this test needs the mace-torch {release} provider")
    return phx.atomistic.interchange.MACEProviderRuntime(
        pin_executable(python, version=version, license_id="PSF-2.0"),
        site_packages,
        torch_version,
        cuequivariance_version=os.environ.get(CUEQUIVARIANCE_ENVIRONMENT),
        openequivariance_version=os.environ.get(OPENEQUIVARIANCE_ENVIRONMENT),
        les_version=os.environ.get(LES_ENVIRONMENT),
        mace_torch_version=mace_version,
    )


def reduced_cg_provider(provider: Any) -> Any:
    """The provider when it declares cuequivariance; otherwise the skip boundary."""

    if provider.cuequivariance_version is None:
        pytest.skip(f"declare {CUEQUIVARIANCE_ENVIRONMENT} for reduced-CG U bases")
    return provider


def les_provider(provider: Any) -> Any:
    """The provider when it declares LES; otherwise the skip boundary."""

    if provider.les_version is None:
        pytest.skip(f"declare {LES_ENVIRONMENT} for MACELES sources")
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
        version="0.3.16" if architecture is None else architecture["mace_torch_release"],
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


def configurations(
    species: tuple[int, ...],
    *,
    total_charge: int | None = None,
    total_spin: int | None = None,
) -> tuple[Any, Any]:
    """One finite and one triclinic periodic configuration of ``species``."""

    finite = np.array(
        [[0.0, 0.0, 0.0], [0.96, 0.08, -0.04], [-0.27, 0.91, 0.18], [1.9, 0.5, 0.35]]
    )
    cell = np.array([[3.6, 0.0, 0.0], [0.5, 3.3, 0.0], [0.3, 0.4, 3.8]])
    periodic = np.array([[0.1, 0.2, 0.1], [1.05, 0.35, 0.2], [0.3, 1.2, 0.55]])
    numbers = tuple(species[index % len(species)] for index in range(4))
    return (
        phx.atomistic.interchange.MACEProviderConfiguration(
            numbers,
            finite,
            np.zeros((3, 3)),
            False,
            total_charge=total_charge,
            total_spin=total_spin,
        ),
        phx.atomistic.interchange.MACEProviderConfiguration(
            numbers[:3],
            periodic,
            cell,
            True,
            total_charge=total_charge,
            total_spin=total_spin,
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


# The provider data pipeline's values of absent charge/spin conditions.
_PROVIDER_CONDITION_DEFAULTS = {"total_charge": 0, "total_spin": 1}


def conditioning(potential: Any, configuration: Any) -> Any:
    """The joint-embedding conditions of one configuration (``None`` without any).

    Absent configuration values take the provider's own defaults, so native
    and provider evaluations read the same codes.
    """

    spec = potential.configuration.conditioning
    if not spec.names:
        return None
    values = {}
    for name in spec.names:
        value = getattr(configuration, name)
        values[name] = np.asarray(
            [_PROVIDER_CONDITION_DEFAULTS[name] if value is None else value],
            dtype=np.int64,
        )
    return phx.atomistic.AtomisticConditioning(
        spec, values, case_count=1, atom_capacity=len(configuration.numbers)
    )


def native_case(potential: Any, configuration: Any) -> dict[str, np.ndarray]:
    """Native total energy, forces and (periodic) stress of one configuration."""

    prediction = phx.atomistic.energy_and_forces(
        potential,
        structure(potential, configuration),
        execution(configuration),
        compute_stress=configuration.periodic,
        conditioning=conditioning(potential, configuration),
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


def latent_resources(definition: Any, cell: Any) -> Any:
    """Latent Ewald resources whose reciprocal modes cover ``cell`` (or none)."""

    from phydrax.atomistic._latent_electrostatics import LatentEwaldResources

    if cell is None or not np.any(cell):
        return LatentEwaldResources()
    length = float(np.max(np.linalg.norm(np.asarray(cell), axis=1)))
    return LatentEwaldResources(mode_extents=definition.covering_extents(length))


_INTERACTION_KINDS: dict[str, MACEInteractionKind] = {
    "RealAgnosticInteractionBlock": "real-agnostic",
    "RealAgnosticResidualInteractionBlock": "real-agnostic-residual",
    "RealAgnosticDensityInteractionBlock": "real-agnostic-density",
    "RealAgnosticDensityResidualInteractionBlock": "real-agnostic-density-residual",
    "RealAgnosticAttResidualInteractionBlock": "real-agnostic-attention-residual",
    "RealAgnosticResidualNonLinearInteractionBlock": "real-agnostic-residual-nonlinear",
}
_DISTANCE_TRANSFORMS: dict[str, MACEDistanceTransform] = {
    "None": "none",
    "Agnesi": "agnesi",
    "Soft": "soft",
}


def committed_manifest(release: str = "0.3.16") -> dict[str, Any]:
    return json.loads((RELEASE_FIXTURES[release] / "manifest.json").read_text())


def committed_architecture(declaration: Mapping[str, Any]) -> MACEArchitecture:
    """The native architecture of one committed generator case.

    Generator cases share r_max 3.0, four basis functions, cutoff power 5,
    ``l <= 2`` harmonics, four channels, ``radial_MLP=[8]`` and ``5x0e``
    readout MLPs; the manifest records each case's family fields and the
    provider's realized normalization constants.
    """

    count = declaration["interactions"]
    radial_scale = declaration["normalize2mom_silu"]
    readout_scale = declaration.get("readout_scale", radial_scale)
    transform = declaration["distance_transform"]
    hidden = declaration.get("hidden_irreps", "4x0e+4x1o")
    edge = declaration.get("edge_irreps")
    edge_channels = None if edge is None else int(edge.split("x", 1)[0])
    edge_degree = None if edge is None else len(edge.split("+")) - 1
    from phydrax.nn.atomistic._mace_embedding import MACEEmbeddingFeature

    features = tuple(
        MACEEmbeddingFeature(
            name,
            "categorical",
            level="case",
            width=spec["emb_dim"],
            classes=spec["num_classes"],
            offset=spec["offset"],
        )
        for name, spec in (declaration.get("embedding_specs") or {}).items()
    )
    gate_scales = declaration.get("gate_scales")
    return MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis=declaration.get("radial_type", "bessel"),
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=len(hidden.split("+")) - 1,
        edge_degree=2,
        parity="so3" if declaration.get("use_so3", False) else "o3",
        edge_channel_count=edge_channels,
        edge_feature_degree=edge_degree,
        edge_irreps_first=declaration.get("use_edge_irreps_first", False),
        interactions=(_INTERACTION_KINDS[declaration["first"]],)
        + (_INTERACTION_KINDS[declaration["rest"]],) * (count - 1),
        correlations=tuple(declaration["correlation"]),
        radial_widths=(8,),
        readout_width=5 if count > 1 else None,
        readout_variant="nonlinear-bias"
        if declaration.get("readout_class") == "NonLinearBiasReadoutBlock"
        else "nonlinear",
        radial_activation_scale=radial_scale,
        readout_activation_scale=readout_scale if count > 1 else None,
        gate_scales=None if gate_scales is None else tuple(gate_scales),
        average_neighbor_count=2.0,
        heads=tuple(declaration["heads"]),
        head=declaration["heads"][0],
        energy_scaling="unscaled" if declaration["class"] == "MACE" else "scale-shift",
        pair_repulsion=declaration["pair_repulsion"],
        distance_transform=_DISTANCE_TRANSFORMS[transform],
        agnesi_parameters=(1.0805, 0.9183, 4.5791) if transform == "Agnesi" else None,
        soft_alpha=4.0 if transform == "Soft" else None,
        cutoff_placement="embedding" if declaration["apply_cutoff"] else "weights",
        keep_last_layer_irreps=declaration.get("keep_last_layer_irreps", False),
        embedding_features=features,
        embedding_readout=declaration.get("use_embedding_readout", False),
    )


def _committed_inputs(
    case: str, release: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[np.ndarray], dict[Any, Any]]:
    declaration = committed_manifest(release)["cases"][case]
    fixture = np.load(RELEASE_FIXTURES[release] / f"{case}.npz")
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
    return declaration, tensors, transforms, couplings


def committed_source_potential(
    case: str = "two_scale_shift", release: str = "0.3.16"
) -> Any:
    """A native model reconstructed from a committed provider-built fixture.

    The npz holds the provider state dictionary, its Wigner 3j tables and
    sampled source harmonics (``tests/fixtures/mace_torch_<release>``); no
    provider is needed to rebuild the source parameterization.
    """

    declaration, tensors, transforms, couplings = _committed_inputs(case, release)
    return phx.nn.atomistic.mace_potential_from_source(
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond().scale,
        committed_architecture(declaration),
        tensors,
        degree_transforms=transforms,
        couplings=couplings,
        source_id=f"mace-torch-{release}-fixture:{case}",
    )


def committed_maceles_potential(
    case: str, atom_capacity: int, resources: Any, release: str = "0.3.16"
) -> Any:
    """A native MACELES model of a committed fixture, prepared for one capacity."""

    from phydrax.nn.atomistic._mace_source import maceles_potential_from_source

    declaration, tensors, transforms, couplings = _committed_inputs(case, release)
    return maceles_potential_from_source(
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond().scale,
        committed_architecture(declaration),
        tensors,
        les_arguments=declaration["les_arguments"],
        atom_capacity=atom_capacity,
        resources=resources,
        degree_transforms=transforms,
        couplings=couplings,
        source_id=f"mace-torch-{release}-fixture:{case}",
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
    """Fixed-cell BAOAB Langevin NVT over the native provider recipe.

    Ordered accumulation makes in-process restarts continue bitwise on every
    backend; fresh-process GPU restarts also need ``compiler_profile``.
    """

    from tests._support.mace_deployment import learned_program

    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        learned_program(model, system, skin=skin),
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


def water_training_problem(
    *, label_shift: float = 0.0, payload_cases: int | None = None
) -> Any:
    """Periodic E/F/S labels of a fixed teacher on strained cells.

    Two strains train (one global batch per update); a third validates from a
    lineage-disjoint loader. ``label_shift`` alters the training energy labels
    (a different problem with otherwise identical structure).
    ``payload_cases=None`` packs the whole batch into one payload;
    ``payload_cases=1`` packs one case per payload, so every update window
    accumulates two payloads (microsteps). Ordered accumulation makes labels
    and continuation bitwise within one process; fresh-process GPU restarts
    also need ``compiler_profile``.
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

    def source(split: str, magnitudes: tuple[float, ...], shift: float) -> Any:
        structures = batch(magnitudes)
        labels = phx.atomistic.energy_and_forces(
            teacher, structures, execution, compute_stress=True
        )
        return phx.atomistic.atomistic_training_source(
            structures,
            mass_unit=phx.units.DALTON,
            energy=labels.energy + shift,
            forces=labels.forces,
            stress=labels.stress,
            record_ids=tuple(f"water-{split}-{magnitude}" for magnitude in magnitudes),
        )

    teacher = tiny_mace(seed=11)
    # The reusable layout's search envelope covers every strained cell.
    execution = graph_execution(
        accumulation="deterministic", search_cell=WATER_CELL, search_margin=0.1
    )
    full = phx.atomistic.atomistic_training_loader(
        source("training", (-0.015, 0.02), label_shift)
    )
    if payload_cases is None:
        training = full
    else:
        atom_level = full.mixture.schema.atom_level
        training = phx.atomistic.AtomisticStructureLoader(
            full.mixture,
            full.bindings,
            phx.atomistic.AtomisticWorkloadBuckets(
                (
                    phx.atomistic.AtomisticWorkloadBucket(
                        "water-microstep",
                        cases=payload_cases,
                        levels=(
                            phx.atomistic.AtomisticLevelCapacity(
                                atom_level, layout="per-case", rows=len(WATER_NUMBERS)
                            ),
                        ),
                        max_bytes=1 << 30,
                    ),
                )
            ),
            batch_size=full.batch_size,
            stream=full.stream,
        )
    return phx.atomistic.AtomisticTrainingProblem(
        training,
        execution,
        (
            phx.atomistic.AtomisticEnergyObjective(),
            phx.atomistic.AtomisticForceObjective(),
            phx.atomistic.AtomisticStressObjective(),
        ),
        scale=electronvolt_units().scale,
        cutoff=3.0,
        fields=phx.atomistic.AtomisticTrainingFields(cell="cell", periodic="periodic"),
        validation=phx.atomistic.atomistic_training_loader(
            source("validation", (0.005,), 0.0), stream="validation"
        ),
    )
