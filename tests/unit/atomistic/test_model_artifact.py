#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import copy
import dataclasses
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
    _model_artifact as model_artifact,
    AtomicStructure,
    atomistic_potential_revision,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    energy_and_forces,
)
from phydrax.atomistic._model_artifact import (
    _registered,
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    AtomisticModelArtifactError,
    read_atomistic_model_artifact,
    write_atomistic_model_artifact,
)
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_prepare import (
    MACERadialRealization,
    prepare_mace_potential,
)
from phydrax.nn.atomistic._radial_projection import RadialTableDeclaration
from phydrax.typing import parse
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.recipes import recipe_field_names, recipe_field_position


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)


def _model(seed: int = 3) -> Any:
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
        average_neighbor_count=2.0,
        readout_width=4,
    )
    return MACEPotential(
        SCALE,
        architecture,
        atomic_energies=np.asarray([-1.0, -2.0], dtype=np.float64),
        key=jr.key(seed),
    )


def _energy(model: Any) -> float:
    structure = AtomicStructure(
        np.asarray([8, 1, 1], dtype=np.int32),
        np.asarray(
            [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 0.8, 0.2]], dtype=np.float64
        ),
        np.asarray([16.0, 1.0, 1.0], dtype=np.float64),
        SCALE,
    )
    plan = AtomisticGraphExecutionPlan(8, maximum_dense_atoms=4)
    return float(energy_and_forces(model, structure, plan).energy[0])


def _rewrite(source: Path, target: Path, edit: Any) -> None:
    """Re-encode an artifact after ``edit(manifest, arrays)`` as a valid archive."""

    manifest, arrays = read_array_archive(source, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS)
    manifest = copy.deepcopy(manifest)
    manifest.pop("arrays")
    arrays = {name: np.array(value) for name, value in arrays.items()}
    edit(manifest, arrays)
    write_array_archive(
        target, manifest=manifest, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS, arrays=arrays
    )


def _leaf(manifest: dict[str, Any], suffix: str) -> str:
    inventory = model_recipe_array_inventory(
        manifest["model_recipe"],
        prefix="model/leaves",
        limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    )
    matches = [entry.name for entry in inventory if entry.path.endswith(suffix)]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path, Any]:
    model = _model()
    path = tmp_path_factory.mktemp("artifact") / "model.phydrax"
    manifest = write_atomistic_model_artifact(path, model, licenses=["MIT"])
    return model, path, manifest


def test_round_trip_restores_identities_and_energy(written: Any) -> None:
    model, path, manifest = written
    restored = read_atomistic_model_artifact(
        path, numeric_revision_id=manifest.numeric_revision.revision_id
    )
    assert restored.manifest.artifact_id == manifest.artifact_id
    assert restored.manifest.content_id == manifest.content_id
    assert restored.manifest.structure_id == manifest.structure_id
    assert restored.manifest.licenses == ("MIT",)
    assert (
        atomistic_potential_revision(restored.model).revision_id
        == atomistic_potential_revision(model).revision_id
    )
    assert _energy(restored.model) == _energy(model)


def test_pinned_revision_refuses_other_parameters(written: Any, tmp_path: Path) -> None:
    model, _, manifest = written
    updated = eqx.tree_at(lambda value: value.embedding, model, model.embedding + 0.25)
    path = tmp_path / "updated.phydrax"
    write_atomistic_model_artifact(path, updated)
    with pytest.raises(AtomisticModelArtifactError, match="pinned revision"):
        read_atomistic_model_artifact(
            path, numeric_revision_id=manifest.numeric_revision.revision_id
        )


def test_corrupted_member_bytes_refuse(written: Any, tmp_path: Path) -> None:
    _, path, _ = written
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    corrupted = tmp_path / "corrupted.phydrax"
    corrupted.write_bytes(bytes(payload))
    with pytest.raises(ArrayArchiveCorruptionError):
        read_atomistic_model_artifact(corrupted)


def test_unregistered_model_type_refuses(written: Any, tmp_path: Path) -> None:
    _, path, _ = written
    target = tmp_path / "type.phydrax"

    def edit(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        manifest["model_type"] = "phydrax.nn.atomistic:MACEArchitecture"
        manifest["model_recipe"]["type"] = "phydrax.nn.atomistic:MACEArchitecture"

    _rewrite(path, target, edit)
    with pytest.raises(ArrayArchiveCorruptionError, match="registered atomistic model"):
        read_atomistic_model_artifact(target)


def test_extra_recipe_field_refuses(written: Any, tmp_path: Path) -> None:
    _, path, _ = written
    target = tmp_path / "field.phydrax"

    def edit(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        # An item no registered field owns.
        manifest["model_recipe"]["items"].append({"kind": "literal", "value": 1})

    _rewrite(path, target, edit)
    with pytest.raises(ArrayArchiveCorruptionError, match="recipe is invalid"):
        read_atomistic_model_artifact(target)


def test_shape_preserving_nonfinite_parameter_fails_scientific_validation(
    written: Any, tmp_path: Path
) -> None:
    _, path, _ = written
    target = tmp_path / "nonfinite.phydrax"

    def edit(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        name = _leaf(manifest, ".embedding")
        arrays[name] = np.full_like(arrays[name], np.nan)

    _rewrite(path, target, edit)
    with pytest.raises(AtomisticModelArtifactError, match="scientific validation"):
        read_atomistic_model_artifact(target)


def test_structurally_valid_invalid_cutoff_fails_scientific_validation(
    written: Any, tmp_path: Path
) -> None:
    _, path, _ = written
    target = tmp_path / "cutoff.phydrax"

    def edit(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        recipe = manifest["model_recipe"]
        configuration = recipe["items"][recipe_field_position(recipe, "configuration")]
        configuration["items"][recipe_field_position(configuration, "cutoff")] = {
            "kind": "literal",
            "value": -3.0,
        }

    _rewrite(path, target, edit)
    with pytest.raises(AtomisticModelArtifactError, match="scientific validation"):
        read_atomistic_model_artifact(target)


def test_recorded_identity_must_match_restored_model(
    written: Any, tmp_path: Path
) -> None:
    _, path, _ = written
    target = tmp_path / "identity.phydrax"

    def edit(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        manifest["identity"]["numeric_revision_id"] = "0" * 64

    _rewrite(path, target, edit)
    with pytest.raises(AtomisticModelArtifactError, match="identity"):
        read_atomistic_model_artifact(target)


def test_oversized_declared_array_refuses_before_allocation(
    written: Any, tmp_path: Path
) -> None:
    _, path, _ = written
    target = tmp_path / "oversized.phydrax"

    def edit(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        recipe = manifest["model_recipe"]
        recipe["items"][recipe_field_position(recipe, "embedding")]["shape"] = [1 << 40]

    _rewrite(path, target, edit)
    with pytest.raises(ArrayArchiveCorruptionError, match="recipe is invalid"):
        read_atomistic_model_artifact(target)


def test_failed_write_preserves_previous_artifact(written: Any, tmp_path: Path) -> None:
    model, path, manifest = written
    target = tmp_path / "published.phydrax"
    target.write_bytes(path.read_bytes())
    invalid = eqx.tree_at(
        lambda value: value.embedding, model, jnp.full_like(model.embedding, jnp.nan)
    )
    with pytest.raises(ValueError):
        write_atomistic_model_artifact(target, invalid)
    assert (
        read_atomistic_model_artifact(target).manifest.artifact_id == manifest.artifact_id
    )


def test_prepared_artifact_is_distinct_and_refuses_stale_binding(
    written: Any, tmp_path: Path
) -> None:
    model, _, manifest = written
    prepared_path = tmp_path / "prepared.phydrax"
    prepared = write_atomistic_model_artifact(
        prepared_path, prepare_mace_potential(model)
    )
    assert prepared.numeric_revision.revision_id == manifest.numeric_revision.revision_id
    assert prepared.structure_id != manifest.structure_id
    restored = read_atomistic_model_artifact(prepared_path)
    assert type(restored.model).__name__ == "PreparedMACEPotential"
    stale = tmp_path / "stale.phydrax"

    def edit(recorded: dict[str, Any], arrays: dict[str, Any]) -> None:
        name = _leaf(recorded, ".model.embedding")
        arrays[name] = arrays[name] + 0.125

    _rewrite(prepared_path, stale, edit)
    with pytest.raises(AtomisticModelArtifactError, match="scientific validation"):
        read_atomistic_model_artifact(stale)


def _readout_model() -> Any:
    """One density-normalized residual layer with an invariant readout product."""
    architecture = MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=3,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic-density-residual",),
        correlations=(2,),
        readout_correlation=2,
        radial_widths=(6,),
        average_neighbor_count=2.0,
        readout_width=4,
    )
    return MACEPotential(
        SCALE,
        architecture,
        atomic_energies=np.asarray([-1.0, -2.0], dtype=np.float64),
        key=jr.key(5),
    )


@pytest.fixture(scope="module")
def readout_model() -> Any:
    return _readout_model()


def _forged(path: Path, model: Any) -> None:
    """Publish ``model`` as an attacker would: fresh outer digests, no validation."""
    entry = dataclasses.replace(_registered(type(model)), validate=lambda _: None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model_artifact, "_registered", lambda _: entry)
        write_atomistic_model_artifact(path, model)


def _require_refused(tmp_path: Path, original: Any, changed: Any) -> None:
    """Publication refuses and keeps the previous artifact; a forged one never restores."""
    target = tmp_path / "published.phydrax"
    manifest = write_atomistic_model_artifact(target, original)
    with pytest.raises(ValueError):
        write_atomistic_model_artifact(target, changed)
    assert (
        read_atomistic_model_artifact(target).manifest.artifact_id == manifest.artifact_id
    )
    forged = tmp_path / "forged.phydrax"
    _forged(forged, changed)
    with pytest.raises(AtomisticModelArtifactError, match="scientific validation"):
        read_atomistic_model_artifact(forged)


def _scaled(values: Any) -> Any:
    return values * 2.0


def _nan_entry(values: Any) -> Any:
    return values.reshape(-1).at[0].set(jnp.nan).reshape(values.shape)


def _nudged(values: Any) -> Any:
    return values.reshape(-1).at[0].add(1.0e-3).reshape(values.shape)


_EXACT_TAMPERS = {
    "product-basis-rescaled": (
        "written",
        lambda m: m.layers[0].product.contraction.plan.bases[1][1].coefficients,
        _scaled,
    ),
    "product-basis-nan": (
        "written",
        lambda m: m.layers[1].product.contraction.plan.bases[0][1].coefficients,
        _nan_entry,
    ),
    "product-graph-rerouted": (
        "written",
        lambda m: m.layers[0].product.contraction.plan.product_graph.term_node,
        lambda values: jnp.roll(values, 1),
    ),
    "readout-product-basis-rescaled": (
        "readout",
        lambda m: m.layers[0].readout_product.contraction.plan.bases[0][1].coefficients,
        _scaled,
    ),
    "readout-product-term-map": (
        "readout",
        lambda m: m.layers[0].readout_product.contraction.plan.term_maps[0][1],
        lambda values: values[::-1],
    ),
}


@pytest.mark.parametrize("tamper", sorted(_EXACT_TAMPERS))
def test_fixed_basis_change_retaining_identities_is_refused(
    tamper: str, written: Any, readout_model: Any, tmp_path: Path
) -> None:
    owner, where, replace = _EXACT_TAMPERS[tamper]
    model = written[0] if owner == "written" else readout_model
    changed = eqx.tree_at(where, model, replace(where(model)))
    # The fixed data is not a parameter: identities and revision are unchanged.
    assert changed.architecture_id == model.architecture_id
    assert (
        atomistic_potential_revision(changed).revision_id
        == atomistic_potential_revision(model).revision_id
    )
    _require_refused(tmp_path, model, changed)


_TABLES = RadialTableDeclaration(0.2, 24, layout="projected-width")

_PREPARED_TAMPERS = {
    "merged-coefficients-finite": (
        "written",
        "exact",
        lambda p: p.layers[0].contraction.coefficients[1],
        _nudged,
    ),
    "merged-coefficients-nan": (
        "written",
        "exact",
        lambda p: p.layers[1].contraction.coefficients[0],
        _nan_entry,
    ),
    "duplicated-readout-weight": (
        "written",
        "exact",
        lambda p: p.layers[0].layer.readout.weight,
        _nudged,
    ),
    "source-rows-nan": (
        "written",
        "exact",
        lambda p: p.layers[0].source_rows,
        _nan_entry,
    ),
    "source-rows-finite": (
        "written",
        "exact",
        lambda p: p.layers[0].source_rows,
        _nudged,
    ),
    "message-maps-finite": (
        "written",
        "exact",
        lambda p: p.layers[0].message_maps[0],
        _nudged,
    ),
    "residual-rows-finite": (
        "readout",
        "exact",
        lambda p: p.layers[0].residual_rows,
        _nudged,
    ),
    "readout-merged-coefficients": (
        "readout",
        "exact",
        lambda p: p.layers[0].readout_contraction.coefficients[0],
        _nudged,
    ),
    "radial-table-values": (
        "written",
        "tabulated",
        lambda p: p.layers[1].radial_table.values,
        _nudged,
    ),
    "radial-table-slopes-nan": (
        "written",
        "tabulated",
        lambda p: p.layers[0].radial_table.slopes,
        _nan_entry,
    ),
    "density-table-slopes": (
        "readout",
        "tabulated",
        lambda p: p.layers[0].density_table.slopes,
        _nudged,
    ),
}


@pytest.mark.parametrize("tamper", sorted(_PREPARED_TAMPERS))
def test_prepared_payload_change_retaining_identities_is_refused(
    tamper: str, written: Any, readout_model: Any, tmp_path: Path
) -> None:
    owner, declared, where, replace = _PREPARED_TAMPERS[tamper]
    radial = parse(declared, MACERadialRealization, "radial")
    model = written[0] if owner == "written" else readout_model
    prepared = prepare_mace_potential(
        model, radial=radial, tables=_TABLES if radial == "tabulated" else None
    )
    changed = eqx.tree_at(where, prepared, replace(where(prepared)))
    assert changed.prepared_id == prepared.prepared_id
    _require_refused(tmp_path, prepared, changed)


def _first_basis_recipe(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if node.get("kind") == "dataclass" and {"path_count", "basis_id"} <= set(
            recipe_field_names(node)
        ):
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _first_basis_recipe(child)
        if found is not None:
            return found
    return None


def test_huge_consistent_basis_dimensions_refuse_before_replay_allocation(
    written: Any, tmp_path: Path
) -> None:
    _, path, _ = written
    target = tmp_path / "dimensions.phydrax"
    paths = 1 << 40

    def edit(manifest: dict[str, Any], arrays: dict[str, Any]) -> None:
        basis = _first_basis_recipe(manifest["model_recipe"])
        assert basis is not None
        items = basis["items"]
        components = 2 * items[recipe_field_position(basis, "output_degree")]["value"] + 1
        items[recipe_field_position(basis, "path_count")]["value"] = paths
        relation = items[recipe_field_position(basis, "relation")]
        relation["items"][recipe_field_position(relation, "target_size")]["value"] = (
            components * paths
        )

    # The tiny archive declares a dense replay of ~2**40 * terms entries; the
    # owner binds paths to the admitted W leaves and refuses before allocating.
    _rewrite(path, target, edit)
    with pytest.raises(AtomisticModelArtifactError, match="scientific validation"):
        read_atomistic_model_artifact(target)
