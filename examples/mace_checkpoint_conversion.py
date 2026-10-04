"""Convert an admitted mace-torch checkpoint into a native, pickle-free MACE model.

The source provider (mace-torch, e3nn, torch) runs only in a separately
installed, caller-pinned interpreter; Phydrax never imports it. Nothing is
downloaded: either pass an explicit local checkpoint with its SHA-256 pin, or
let the provider build a tiny deterministic safe state-dict fixture (MIT, the
license of the provider code that generates it).

    PYTHONPATH=. python examples/mace_checkpoint_conversion.py \\
        --provider-venv .tmp/mace-provider/.venv \\
        --provider-python-version 3.12.8 --torch-version 2.14.1

A local checkpoint is a rights decision before anything else: ``--license-id``
declares the license of its weights and ``--held-license`` the licenses the
operator holds; neither defaults, and a digest of the pinned release catalog
(``tools/mace_checkpoint_campaign.py``) must carry that release's weight
license. Both are checked before the provider runs. A full-object pickle
(``--checkpoint model.model --sha256 ...``) is then deserialized only with
``--trust-statement`` naming why that exact digest is trusted; there is no
unsafe fallback. Pinning, rights and pickle trust are independent decisions.
"""

import argparse
import hashlib
import tempfile
from collections.abc import Sequence
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
from tools.mace_checkpoint_campaign import CAMPAIGN_ROWS


# Provider-generated fixture weights carry the license of the provider code.
FIXTURE_LICENSE = "MIT"

FIXTURE_DECLARATION = {
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


def _arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider-venv", type=Path, required=True)
    parser.add_argument("--provider-python-version", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--cuequivariance-version")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--license-id")
    parser.add_argument("--held-license", action="append", default=[])
    parser.add_argument("--trust-statement")
    parser.add_argument("--head")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def _source_rights(arguments: argparse.Namespace) -> tuple[str, tuple[str, ...]]:
    """The weight license and the held licenses that admit it, before any provider run.

    Generated fixtures are MIT. A caller checkpoint names both explicitly; a
    digest of the pinned release catalog must name that release's weight
    license, so restricted (ASL) releases never become MIT by omission.
    """

    if arguments.checkpoint is None:
        if arguments.license_id is not None or arguments.held_license:
            raise SystemExit("--license-id and --held-license apply to --checkpoint.")
        return FIXTURE_LICENSE, (FIXTURE_LICENSE,)
    if arguments.sha256 is None:
        raise SystemExit("--checkpoint requires its out-of-band --sha256 pin.")
    if arguments.license_id is None:
        raise SystemExit(
            "--checkpoint requires the explicit --license-id of its weights."
        )
    pinned = next((row for row in CAMPAIGN_ROWS if row.sha256 == arguments.sha256), None)
    if pinned is not None and pinned.weight_license != arguments.license_id:
        raise SystemExit(
            f"--sha256 pins the {pinned.release} release, whose weights are "
            f"{pinned.weight_license}, not {arguments.license_id}."
        )
    if arguments.license_id not in arguments.held_license:
        raise SystemExit(
            f"Weights under {arguments.license_id} require --held-license "
            f"{arguments.license_id}; a digest pin or pickle trust is not a rights grant."
        )
    return arguments.license_id, tuple(dict.fromkeys(arguments.held_license))


def _provider(
    venv: Path, python_version: str, torch_version: str, cuequivariance: str | None
) -> phx.atomistic.interchange.MACEProviderRuntime:
    major, minor = python_version.split(".")[:2]
    return phx.atomistic.interchange.MACEProviderRuntime(
        pin_executable(
            venv / "bin" / "python", version=python_version, license_id="PSF-2.0"
        ),
        str(venv / "lib" / f"python{major}.{minor}" / "site-packages"),
        torch_version,
        cuequivariance_version=cuequivariance,
    )


def _admitted_source(
    path: Path,
    sha256: str,
    license_id: str,
    held_licenses: tuple[str, ...],
    kind: phx.atomistic.interchange.MACESourceKind,
    *,
    trust: str | None,
    architecture: dict[str, object] | None,
) -> phx.atomistic.interchange.MACESource:
    policy = ExternalArtifactPolicy(
        path.parent,
        maximum_bytes=4 * 1024**3,
        allowed_license_ids=held_licenses,
        allowed_suffixes=(".model", ".pt"),
    )
    manifest = ArtifactManifest(
        artifact_id=path.stem,
        producer="mace-torch",
        version="0.3.16",
        sha256=sha256,
        byte_size=path.stat().st_size,
        source_uri=path.resolve().as_uri(),
        license_id=license_id,
        model=path.stem,
        coverage="caller-supplied local checkpoint",
    )
    artifact = admit_external_artifact(path.name, manifest, policy=policy)
    return phx.atomistic.interchange.MACESource(
        artifact,
        manifest,
        policy,
        kind,
        trust=(
            None
            if trust is None
            else phx.atomistic.interchange.TrustedTorchPickleSource(sha256, trust)
        ),
        architecture=architecture,
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    license_id, held_licenses = _source_rights(arguments)
    provider = _provider(
        arguments.provider_venv,
        arguments.provider_python_version,
        arguments.torch_version,
        arguments.cuequivariance_version,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        if arguments.checkpoint is None:
            (root / "fixture").mkdir()
            fixture = phx.atomistic.interchange.create_mace_provider_fixture(
                FIXTURE_DECLARATION, root / "fixture", provider=provider, seed=20261003
            )
            source = _admitted_source(
                fixture.path,
                fixture.sha256,
                license_id,
                held_licenses,
                "torch-state-dict",
                trust=None,
                architecture=dict(fixture.declaration),
            )
        else:
            observed = hashlib.sha256(arguments.checkpoint.read_bytes()).hexdigest()
            if observed != arguments.sha256:
                raise SystemExit("Checkpoint bytes do not match the supplied --sha256.")
            source = _admitted_source(
                arguments.checkpoint,
                arguments.sha256,
                license_id,
                held_licenses,
                "torch-full-model",
                trust=arguments.trust_statement,
                architecture=None,
            )
        conversion = phx.atomistic.interchange.convert_mace_checkpoint(
            source, provider=provider, head=arguments.head
        )
        native = conversion.potential
        configuration = phx.atomistic.interchange.MACEProviderConfiguration(
            (8, 1, 1), WATER, CELL, True
        )
        # The provider evaluates exactly the head the conversion selected.
        oracle = phx.atomistic.interchange.evaluate_mace_source(
            source, [configuration], provider=provider, head=conversion.provenance.head
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
        # The configuration is periodic, so both sides report stress.
        if prediction.stress is None or oracle.stress is None:
            raise RuntimeError("A periodic configuration must report stress.")
        print(
            "max |dS| [eV/Ang^3]:",
            float(np.max(np.abs(np.asarray(prediction.stress[0]) - oracle.stress))),
        )
        destination = (
            root / "native-model" if arguments.output is None else arguments.output
        )
        manifest = phx.atomistic.write_atomistic_model_artifact(
            destination, native, source=conversion.provenance
        )
        print("native artifact:", destination, manifest.artifact_id)


if __name__ == "__main__":
    main()
