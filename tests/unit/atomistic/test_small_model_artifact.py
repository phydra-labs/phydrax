#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import copy
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from phydrax._model._structure import model_recipe_array_inventory
from phydrax.atomistic import (
    AtomicStructure,
    atomistic_model_revision,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    energy_and_forces,
)
from phydrax.atomistic._model_artifact import (
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    AtomisticModelArtifactError,
    read_atomistic_model_artifact,
    write_atomistic_model_artifact,
)
from phydrax.nn.atomistic import NequIPPotential, PaiNNPotential
from phydrax.units import ANGSTROM, ELECTRONVOLT


REPOSITORY = Path(__file__).resolve().parents[3]
SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
FAMILIES = ("painn", "nequip")


def build_model(family: str) -> Any:
    if family == "painn":
        return PaiNNPotential(
            SCALE,
            cutoff=2.5,
            feature_count=4,
            interaction_count=2,
            radial_basis_count=4,
            maximum_species_id=8,
            key=jr.key(11),
        )
    if family == "nequip":
        return NequIPPotential(
            SCALE,
            cutoff=2.5,
            feature_count=2,
            interaction_count=2,
            radial_basis_count=4,
            maximum_species_id=8,
            key=jr.key(13),
        )
    raise ValueError(family)


def evaluate(model: Any) -> tuple[np.ndarray, np.ndarray]:
    structure = AtomicStructure(
        np.asarray([8, 1, 1], dtype=np.int32),
        np.asarray(
            [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 0.8, 0.2]], dtype=np.float64
        ),
        np.asarray([15.999, 1.008, 1.008], dtype=np.float64),
        SCALE,
    )
    plan = AtomisticGraphExecutionPlan(3, maximum_dense_atoms=4)
    result = energy_and_forces(model, structure, plan)
    return np.asarray(result.energy), np.asarray(result.forces)


def _rewrite(
    source: Path, target: Path, edit: Callable[[dict[str, Any], dict[str, Any]], None]
) -> None:
    """Re-encode an artifact after ``edit(manifest, arrays)`` as a valid archive."""

    manifest, arrays = read_array_archive(source, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS)
    manifest = copy.deepcopy(manifest)
    manifest.pop("arrays")
    arrays = {name: np.array(value) for name, value in arrays.items()}
    edit(manifest, arrays)
    write_array_archive(
        target, manifest=manifest, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS, arrays=arrays
    )


def _leaf(manifest: dict[str, Any], path: str) -> str:
    inventory = model_recipe_array_inventory(
        manifest["model_recipe"],
        prefix="model/leaves",
        limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    )
    matches = [entry.name for entry in inventory if entry.path == path]
    assert len(matches) == 1, path
    return matches[0]


def _fields(manifest: dict[str, Any], *names: str | int) -> dict[str, Any]:
    """Recipe fields of the dataclass reached through fields/tuple items."""
    node = manifest["model_recipe"]
    for name in names:
        node = node["items"][name] if isinstance(name, int) else node["fields"][name]
    return node["fields"]


@pytest.fixture(scope="module", params=FAMILIES)
def written(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    family = request.param
    model = build_model(family)
    path = tmp_path_factory.mktemp(f"{family}-artifact") / "model.phydrax"
    manifest = write_atomistic_model_artifact(path, model)
    energy, forces = evaluate(model)
    return {
        "family": family,
        "model": model,
        "path": path,
        "manifest": manifest,
        "energy": energy,
        "forces": forces,
    }


def test_round_trip_restores_identities_energy_and_forces(written: Any) -> None:
    manifest = written["manifest"]
    restored = read_atomistic_model_artifact(
        written["path"], numeric_revision_id=manifest.numeric_revision.revision_id
    )
    assert type(restored.model) is type(written["model"])
    assert restored.manifest.artifact_id == manifest.artifact_id
    assert restored.model.architecture_id == written["model"].architecture_id
    assert (
        atomistic_model_revision(restored.model).revision_id
        == atomistic_model_revision(written["model"]).revision_id
    )
    energy, forces = evaluate(restored.model)
    np.testing.assert_array_equal(energy, written["energy"])
    np.testing.assert_array_equal(forces, written["forces"])


def test_fresh_process_restore_matches_energy_forces_and_revision(
    written: Any, tmp_path: Path
) -> None:
    output = tmp_path / "restored.npz"
    script = textwrap.dedent(
        """
        import importlib.util
        import sys

        import numpy as np

        from phydrax.atomistic import atomistic_model_revision
        from phydrax.atomistic._model_artifact import read_atomistic_model_artifact

        module_path, artifact, revision, output = sys.argv[1:5]
        spec = importlib.util.spec_from_file_location("small_artifact", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        restored = read_atomistic_model_artifact(artifact, numeric_revision_id=revision)
        energy, forces = module.evaluate(restored.model)
        np.savez(
            output,
            energy=energy,
            forces=forces,
            revision=np.asarray(atomistic_model_revision(restored.model).revision_id),
            artifact=np.asarray(restored.manifest.artifact_id),
        )
        """
    )
    manifest = written["manifest"]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve()),
            str(written["path"]),
            manifest.numeric_revision.revision_id,
            str(output),
        ],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output) as restored:
        np.testing.assert_array_equal(restored["energy"], written["energy"])
        np.testing.assert_array_equal(restored["forces"], written["forces"])
        assert str(restored["revision"]) == manifest.numeric_revision.revision_id
        assert str(restored["artifact"]) == manifest.artifact_id


def _cutoff(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
    _fields(manifest, "configuration")["cutoff"] = {"kind": "literal", "value": 2.0}


def _feature_count(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
    configuration = _fields(manifest, "configuration")
    count = configuration["feature_count"]["value"]
    configuration["feature_count"] = {"kind": "literal", "value": count + 1}


def _radial_frequency(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
    name = _leaf(manifest, ".configuration.radial_frequencies")
    arrays[name] = arrays[name] * 1.5


def _activation(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
    _fields(manifest, "readout_hidden")["activation"] = {
        "kind": "callable",
        "value": "jax.artifact:tanh",
    }


def _nonfinite_interaction(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
    family = manifest["model_type"]
    path = (
        ".interactions[1].filter_in.weight"
        if family.endswith("PaiNNPotential")
        else ".interactions[1].radial_in.weight"
    )
    name = _leaf(manifest, path)
    arrays[name] = arrays[name].copy()
    arrays[name].flat[0] = np.nan


def _method(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
    manifest["model_recipe"]["fields"]["method_id"] = {
        "kind": "literal",
        "value": "negative-position-gradient-of-some-other-energy",
    }


_SCIENTIFIC_TAMPERS = {
    "cutoff": _cutoff,
    "feature-count": _feature_count,
    "radial-frequency": _radial_frequency,
    "activation": _activation,
    "nonfinite-interaction": _nonfinite_interaction,
    "method": _method,
}


@pytest.mark.parametrize("tamper", sorted(_SCIENTIFIC_TAMPERS))
def test_tampered_fixed_metadata_or_parameters_refuse(
    written: Any, tamper: str, tmp_path: Path
) -> None:
    target = tmp_path / f"{tamper}.phydrax"
    _rewrite(written["path"], target, _SCIENTIFIC_TAMPERS[tamper])
    with pytest.raises(AtomisticModelArtifactError, match="scientific validation"):
        read_atomistic_model_artifact(target)


def test_tampered_parameter_shape_refuses(written: Any, tmp_path: Path) -> None:
    target = tmp_path / "shape.phydrax"

    def edit(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        name = _leaf(manifest, ".readout_hidden.bias")
        node = _fields(manifest, "readout_hidden")["bias"]
        node["shape"] = [node["shape"][0] + 1]
        arrays[name] = np.zeros(tuple(node["shape"]), dtype=arrays[name].dtype)

    _rewrite(written["path"], target, edit)
    # Either the structural guard or the family validator refuses; never admits.
    with pytest.raises((AtomisticModelArtifactError, ArrayArchiveCorruptionError)):
        read_atomistic_model_artifact(target)


def test_nequip_tensor_product_plan_and_couplings_refuse(tmp_path: Path) -> None:
    model = build_model("nequip")
    source = tmp_path / "nequip.phydrax"
    write_atomistic_model_artifact(source, model)

    def plan_budget(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        plan = _fields(manifest, "interactions", 0, "tensor_product", "plan")
        limit = plan["maximum_parameters"]["value"]
        plan["maximum_parameters"] = {"kind": "literal", "value": limit + 1}

    def coupling(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        name = _leaf(manifest, ".interactions[0].tensor_product.prepared.coefficients[0]")
        arrays[name] = arrays[name] * 2.0

    def plan_ids(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        configuration = _fields(manifest, "configuration")
        ids = configuration["tensor_product_plan_ids"]
        ids["items"][0] = {"kind": "literal", "value": "0" * 64}

    for name, edit in (
        ("budget", plan_budget),
        ("coupling", coupling),
        ("plan-ids", plan_ids),
    ):
        target = tmp_path / f"{name}.phydrax"
        _rewrite(source, target, edit)
        with pytest.raises(AtomisticModelArtifactError, match="scientific validation"):
            read_atomistic_model_artifact(target)


@pytest.mark.parametrize("family", FAMILIES)
def test_validate_refuses_transformed_parameters(family: str) -> None:
    model = build_model(family)
    widened = eqx.tree_at(
        lambda value: value.readout_hidden.weight,
        model,
        jnp.zeros(
            (model.configuration.feature_count, model.configuration.feature_count + 1),
            dtype=model.readout_hidden.weight.dtype,
        ),
    )
    with pytest.raises(ValueError, match="shapes"):
        widened.validate()
    nonfinite = eqx.tree_at(
        lambda value: value.embedding, model, model.embedding.at[0, 0].set(jnp.inf)
    )
    with pytest.raises(ValueError, match="finite"):
        nonfinite.validate()
    lowered = eqx.tree_at(
        lambda value: value.readout_energy.weight,
        model,
        model.readout_energy.weight.astype(jnp.float32),
    )
    with pytest.raises(ValueError, match="dtype"):
        lowered.validate()
