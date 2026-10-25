#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Crystal packing recipes: constructor replay, identities, refusal and restart.

Recipes rebuild every scientific input through its owning constructor and
verify each recorded identity; tampered, incomplete, mistyped or unreferenced
content refuses.  A recipe and a partly active generation checkpoint resume in
a fresh interpreter that builds no molecule or policy by hand.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import jax
import jax.random as jr
import numpy as np
import pytest

from phydrax._array_archive import (
    array_collection_digest,
    ArrayArchiveError,
    read_array_archive,
    write_array_archive,
)
from phydrax._jax_config import default_device
from phydrax.atomistic._generation import (
    AtomisticGenerationStopReason,
    AtomisticMemoryGenerationSink,
)
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticDenseImageSearch,
    AtomisticRoleCapacity,
)
from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    ConformerPool,
    ContactPolicy,
    CrystalCellPolicy,
    CrystalContactResources,
    CrystalPackingPlan,
    CrystalPackingRecipe,
    CrystalReliefPolicy,
    CrystalSpaceGroupPolicy,
    ElementTable,
    MolecularComponent,
    MolecularConformer,
    MolecularCrystalInput,
    read_crystal_packing_recipe,
    RigidSymmetryLayout,
    SettingTransformation,
    standard_space_group_setting,
    VolumePolicy,
    write_crystal_packing_recipe,
)
from phydrax.atomistic.crystal._recipe import _content_id
from phydrax.discretization import ParticleImageCapacity
from tests._support import crystal_recipe_restart as worker
from tests._support.crystal_geometry_molecules import trans_dichloroethene


REPOSITORY = Path(__file__).resolve().parents[4]
WORKER = Path(worker.__file__).resolve()
MASSES = ElementTable(
    "mass",
    {1: 1.008, 6: 12.011, 17: 35.45, 35: 79.904},
    provenance="CIAAW 2021 conventional standard atomic weights (test subset)",
)
RADII = ElementTable(
    "contact-radius",
    {1: 1.0, 6: 1.5, 17: 1.5, 35: 1.6},
    provenance="declared test contact radii",
)
ATOMIC_VOLUMES = ElementTable(
    "atomic-volume", {1: 5.0, 6: 15.0, 17: 20.0}, provenance="declared test volumes"
)
POOL_ARRAY = "pool-0000/admitted-positions"


@pytest.fixture
def _cpu() -> Iterator[None]:
    with default_device(jax.devices("cpu")[0]):
        yield


def _resources(lanes: int) -> CrystalContactResources:
    width = 8192
    return CrystalContactResources(
        AtomisticDenseImageSearch(maximum_dense_routes=lanes * 200_000),
        search_capacities=(
            ParticleImageCapacity(
                maximum_particles_per_cell=64,
                maximum_edges=lanes * width,
                maximum_degree=width,
                maximum_images=4096,
                maximum_case_edges=width,
            ),
        ),
        role_capacities=(
            AtomisticRoleCapacity(maximum_routes=lanes * width, maximum_degree=width),
        ),
        skin=0.5,
    )


def _pool(
    *, name: str = "trans-1,2-dichloroethene", weights: tuple[float, ...] = (2.0, 1.0)
) -> ConformerPool:
    reference = trans_dichloroethene()
    base = reference.component
    component = MolecularComponent(
        base.atomic_numbers,
        base.bonds,
        bond_orders=base.bond_orders,
        masses=MASSES,
        spin_multiplicity=1,
        double_bond_stereo=base.double_bond_stereo,
        name=name,
    )
    geometry = np.asarray(reference.conformers[0].admitted_positions)
    # A distinct second geometry (one hydrogen displaced), admitted in a
    # rotated and translated frame so its witness differs from its local frame.
    displaced = geometry.copy()
    displaced[4] += np.asarray([0.04, 0.03, 0.0])
    angle = 0.3
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    conformers = (
        MolecularConformer(component, geometry, provenance="idealized test geometry"),
        MolecularConformer(
            component,
            displaced @ rotation.T + np.asarray([0.5, -1.0, 2.0]),
            provenance="displaced idealized test geometry",
        ),
    )
    return ConformerPool(component, conformers, weights=weights)


def _recipe(
    *,
    name: str = "trans-1,2-dichloroethene",
    conformer_weights: tuple[float, ...] = (2.0, 1.0),
    campaign_id: str = "recipe",
    lanes: int = 2,
) -> CrystalPackingRecipe:
    molecular = MolecularCrystalInput(
        (_pool(name=name, weights=conformer_weights),),
        contact=ContactPolicy(RADII),
        volume=VolumePolicy(formula_unit_volume=60.0),
    )
    groups = CrystalSpaceGroupPolicy(
        [standard_space_group_setting(number) for number in (1, 2)],
        weights=(1.0, 3.0),
        provenance="declared test prior (not a CSD frequency table)",
    )
    plan = CrystalPackingPlan(
        molecular,
        groups,
        z_prime=1,
        chirality="enantiopure",
        resources=_resources(lanes),
        cell=CrystalCellPolicy(minimum_height=2.6, minimum_height_floor=2.5),
        relief=CrystalReliefPolicy(maximum_steps=10, check_interval=5),
    )
    return CrystalPackingRecipe(plan.prepare(lanes=lanes), campaign_id=campaign_id)


def _variant_recipe() -> CrystalPackingRecipe:
    """Per-atom masses, a declared contact matrix, tabulated volume, transformed P-1."""
    pool = trans_dichloroethene()
    atoms = pool.component.atom_count
    shifted = standard_space_group_setting(2).transformed(
        SettingTransformation(None, (Fraction(1, 4), Fraction(0), Fraction(0)))
    )
    plan = CrystalPackingPlan(
        MolecularCrystalInput(
            (pool,),
            contact=np.full((atoms, atoms), 3.0),
            volume=VolumePolicy(atomic_volumes=ATOMIC_VOLUMES),
        ),
        CrystalSpaceGroupPolicy.fixed(shifted),
        z_prime=1,
        chirality="enantiopure",
        resources=_resources(1),
        cell=CrystalCellPolicy(minimum_height=2.6, minimum_height_floor=2.5),
        relief=CrystalReliefPolicy(maximum_steps=10, check_interval=5),
    )
    return CrystalPackingRecipe(plan.prepare(lanes=1), campaign_id="variant")


def test_recipe_round_trip_rebuilds_every_scientific_input(
    tmp_path: Path, _cpu: None
) -> None:
    recipe = _recipe()
    written = write_crystal_packing_recipe(tmp_path / "packing.recipe", recipe)
    restored, admitted = read_crystal_packing_recipe(tmp_path / "packing.recipe")
    assert admitted.content_id == written.content_id
    assert admitted.array_digest == written.array_digest
    assert restored.recipe_id == recipe.recipe_id == admitted.recipe_id
    assert restored.plan.plan_id == recipe.plan.plan_id == admitted.plan_id
    assert restored.plan.law_id == recipe.plan.law_id == admitted.law_id
    assert written.law_id == admitted.law_id and written.plan_id == admitted.plan_id
    assert restored.stream_identity == recipe.stream_identity
    assert set(dict(recipe.stream_identity)) == {
        "stream",
        "samplers",
        "revisions",
        "executables",
    }
    assert restored.stream().stream_id == recipe.stream().stream_id
    assert [m.sampler_id for m in restored.packer.members] == [
        m.sampler_id for m in recipe.packer.members
    ]

    source, rebuilt = recipe.plan, restored.plan
    pool, pool_ = source.molecular_input.pools[0], rebuilt.molecular_input.pools[0]
    assert pool_.pool_id == pool.pool_id
    assert pool_.declared_weights == (2.0, 1.0)
    assert pool_.component.component_id == pool.component.component_id
    mass_table = pool_.component.mass_table
    assert mass_table is not None and mass_table.table_id == MASSES.table_id
    assert pool_.component.mass_provenance == MASSES.table_id
    assert pool_.component.mass_provenance == pool.component.mass_provenance
    for left, right in zip(pool.conformers, pool_.conformers, strict=True):
        assert right.conformer_id == left.conformer_id
        np.testing.assert_array_equal(right.admitted_positions, left.admitted_positions)
        np.testing.assert_array_equal(
            np.asarray(right.positions), np.asarray(left.positions)
        )
        np.testing.assert_array_equal(
            np.asarray(right.principal_moments), np.asarray(left.principal_moments)
        )
    contact = rebuilt.molecular_input.contact_policy
    assert contact is not None and contact.radii.provenance == RADII.provenance
    assert contact.policy_id == rebuilt.molecular_input.contact_provenance
    assert rebuilt.molecular_input.volume_policy.policy_id == (
        source.molecular_input.volume_policy.policy_id
    )
    assert rebuilt.groups.declared_weights == (1.0, 3.0)
    assert rebuilt.groups.provenance == source.groups.provenance
    assert [s.setting_id for s in rebuilt.groups.settings] == [
        s.setting_id for s in source.groups.settings
    ]
    assert rebuilt.cell.policy_id == source.cell.policy_id
    assert rebuilt.relief.policy_id == source.relief.policy_id
    assert rebuilt.resources.resources_id == source.resources.resources_id

    # Re-publishing the restored recipe yields the identical archive bytes.
    again = write_crystal_packing_recipe(tmp_path / "again.recipe", restored)
    assert again.content_id == written.content_id
    assert (tmp_path / "again.recipe").read_bytes() == (
        tmp_path / "packing.recipe"
    ).read_bytes()


def test_compiled_consumers_hold_no_restoration_witnesses(_cpu: None) -> None:
    recipe = _recipe()
    witnesses = (
        AsymmetricUnitPlan,
        ConformerPool,
        ElementTable,
        MolecularComponent,
        MolecularConformer,
        MolecularCrystalInput,
    )
    rigid = RigidSymmetryLayout(recipe.packer.units[1])
    for compiled in (*recipe.packer.members, rigid):
        nodes = jax.tree_util.tree_leaves(
            compiled, is_leaf=lambda node: isinstance(node, witnesses)
        )
        assert not any(isinstance(node, witnesses) for node in nodes)
    for member, unit in zip(recipe.packer.members, recipe.packer.units, strict=True):
        layout = member.layout
        assert layout.plan_id == unit.plan_id
        assert all(
            isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(layout)
        )
        np.testing.assert_array_equal(
            np.asarray(layout.conformer_table), unit.conformer_table
        )
        np.testing.assert_array_equal(np.asarray(layout.atom_copy), unit.atom_copy)


def test_transformed_setting_and_declared_inputs_round_trip(
    tmp_path: Path, _cpu: None
) -> None:
    recipe = _variant_recipe()
    write_crystal_packing_recipe(tmp_path / "variant.recipe", recipe)
    restored, _ = read_crystal_packing_recipe(tmp_path / "variant.recipe")
    (setting,) = restored.plan.groups.settings
    (original,) = recipe.plan.groups.settings
    assert not setting.transformation.is_identity
    assert setting.transformation.transformation_id == (
        original.transformation.transformation_id
    )
    assert setting.setting_id == original.setting_id
    molecular = restored.plan.molecular_input
    assert molecular.contact_policy is None
    np.testing.assert_array_equal(
        np.asarray(molecular.contact_distances),
        np.asarray(recipe.plan.molecular_input.contact_distances),
    )
    assert molecular.contact_provenance == recipe.plan.molecular_input.contact_provenance
    assert molecular.formula_unit_volume == 80.0
    volumes = molecular.volume_policy.atomic_volumes
    assert volumes is not None and volumes.table_id == ATOMIC_VOLUMES.table_id
    assert molecular.pools[0].component.mass_table is None
    assert restored.recipe_id == recipe.recipe_id


def test_identities_change_only_with_content(tmp_path: Path, _cpu: None) -> None:
    reference = _recipe()
    first = write_crystal_packing_recipe(tmp_path / "a.recipe", reference)
    repeat = write_crystal_packing_recipe(tmp_path / "b.recipe", _recipe())
    assert repeat.content_id == first.content_id
    assert repeat.recipe_id == first.recipe_id

    # A display-only name changes stored content, never a scientific identity.
    renamed = _recipe(name="display name only")
    named = write_crystal_packing_recipe(tmp_path / "c.recipe", renamed)
    assert named.content_id != first.content_id
    assert renamed.plan.law_id == reference.plan.law_id
    assert renamed.plan.plan_id == reference.plan.plan_id
    assert renamed.recipe_id == reference.recipe_id

    # Changed normalized conformer weights are new scientific content.
    weighted = _recipe(conformer_weights=(3.0, 1.0))
    assert weighted.plan.law_id != reference.plan.law_id
    assert weighted.plan.plan_id != reference.plan.plan_id
    assert weighted.recipe_id != reference.recipe_id

    # Rescaled declared weights are a restoration witness only: the
    # normalized law and every scientific identity are unchanged.
    rescaled = _recipe(conformer_weights=(4.0, 2.0))
    witness = write_crystal_packing_recipe(tmp_path / "d.recipe", rescaled)
    assert rescaled.plan.law_id == reference.plan.law_id
    assert rescaled.plan.plan_id == reference.plan.plan_id
    assert rescaled.recipe_id == reference.recipe_id
    assert witness.content_id != first.content_id

    # A new campaign keeps the plan and changes the stream and recipe.
    campaign = _recipe(campaign_id="another")
    assert campaign.plan.plan_id == reference.plan.plan_id
    assert campaign.recipe_id != reference.recipe_id


def _rewrite(
    source: Path,
    target: Path,
    mutate: Callable[[dict, dict[str, np.ndarray]], None],
    *,
    readdress: bool = True,
) -> Path:
    manifest, arrays = read_array_archive(source)
    body = copy.deepcopy(manifest["recipe"])
    values = {name: np.array(value) for name, value in arrays.items()}
    mutate(body, values)
    digest = array_collection_digest(values)
    return write_array_archive(
        target,
        manifest={
            "kind": manifest["kind"],
            "recipe": body,
            "array_digest": digest,
            "content_id": _content_id(body, digest)
            if readdress
            else manifest["content_id"],
        },
        arrays=values,
    )


def _plan(body: dict) -> dict:
    return body["plan"]


def _shift_atom(body: dict, arrays: dict[str, np.ndarray]) -> None:
    arrays[POOL_ARRAY][0, 0, 0] += 1.0e-6


def _drop(body: dict, arrays: dict[str, np.ndarray]) -> None:
    del arrays[POOL_ARRAY]


TAMPERING = {
    "unknown-field": (
        lambda body, arrays: _plan(body).update(extra=1),
        "packing plan recipe fields are not canonical",
    ),
    "missing-field": (
        lambda body, arrays: _plan(body)["relief"].pop("check_interval"),
        "relief policy recipe fields are not canonical",
    ),
    "wrong-type": (
        lambda body, arrays: _plan(body).update(z_prime="1"),
        "must be an integer",
    ),
    "conformer-coordinates": (_shift_atom, "rebuilt conformer does not reproduce"),
    "missing-array": (_drop, "is missing"),
    "unreferenced-array": (
        lambda body, arrays: arrays.update(extra=np.zeros((3,))),
        "unreferenced arrays",
    ),
    "cell-policy-value": (
        lambda body, arrays: _plan(body)["cell"].update(oversample=1.6),
        "rebuilt cell policy does not reproduce",
    ),
    "radius-provenance": (
        lambda body, arrays: _plan(body)["input"]["contact"]["radii"].update(
            provenance="another source"
        ),
        "rebuilt contact radius table does not reproduce",
    ),
    "mass-value": (
        lambda body, arrays: _plan(body)["input"]["pools"][0]["component"]["masses"][
            "table"
        ]["entries"][0].__setitem__(1, 1.5),
        "rebuilt mass table does not reproduce",
    ),
    "setting-transformation": (
        lambda body, arrays: _plan(body)["groups"]["settings"][1].update(
            origin_shift=["1/4", "0", "0"]
        ),
        "rebuilt setting transformation does not reproduce",
    ),
    "noncanonical-fraction": (
        lambda body, arrays: _plan(body)["groups"]["settings"][1].update(
            origin_shift=["0/2", "0", "0"]
        ),
        "not canonical",
    ),
    "group-weight": (
        lambda body, arrays: _plan(body)["groups"].update(declared_weights=[1.0, 2.0]),
        "rebuilt space-group policy does not reproduce",
    ),
    "lanes": (
        lambda body, arrays: body["generation"].update(lanes=1),
        "sampler and stream identities",
    ),
    "law-identity": (
        lambda body, arrays: _plan(body).update(law_id="0" * 64),
        "rebuilt packing law does not reproduce",
    ),
    "missing-law-identity": (
        lambda body, arrays: _plan(body).pop("law_id"),
        "packing plan recipe fields are not canonical",
    ),
    # A stream identity recorded before executable layouts were bound refuses.
    "missing-executable-identity": (
        lambda body, arrays: body["generation"]["stream_identity"].pop("executables"),
        "stream identity recipe fields are not canonical",
    ),
    "executable-identity": (
        lambda body, arrays: body["generation"]["stream_identity"].update(
            executables="0" * 64
        ),
        "sampler and stream identities",
    ),
    "recipe-identity": (
        lambda body, arrays: body.update(recipe_id="0" * 64),
        "rebuilt packing recipe does not reproduce",
    ),
    "kind": (
        lambda body, arrays: body.update(kind="crystal-packing-plan"),
        "does not hold a crystal packing recipe",
    ),
}


@pytest.mark.parametrize("case", sorted(TAMPERING))
def test_tampered_or_incomplete_recipes_refuse(
    case: str, tmp_path: Path, _cpu: None
) -> None:
    source = tmp_path / "source.recipe"
    write_crystal_packing_recipe(source, _recipe())
    mutate, message = TAMPERING[case]
    target = _rewrite(source, tmp_path / f"{case}.recipe", mutate)
    with pytest.raises(ValueError, match=message):
        read_crystal_packing_recipe(target)


def test_unaddressed_edits_and_corrupt_bytes_refuse(tmp_path: Path, _cpu: None) -> None:
    source = tmp_path / "source.recipe"
    write_crystal_packing_recipe(source, _recipe())
    stale = _rewrite(
        source,
        tmp_path / "stale.recipe",
        lambda body, arrays: _plan(body)["relief"].update(maximum_steps=11),
        readdress=False,
    )
    with pytest.raises(ValueError, match="content address"):
        read_crystal_packing_recipe(stale)
    payload = bytearray(source.read_bytes())
    marker = payload.index(b"\x93NUMPY")
    payload[marker + 200] ^= 0xFF
    corrupt = tmp_path / "corrupt.recipe"
    corrupt.write_bytes(bytes(payload))
    with pytest.raises(ArrayArchiveError):
        read_crystal_packing_recipe(corrupt)


def test_checkpoint_refuses_a_stream_of_another_recipe(
    tmp_path: Path, _cpu: None
) -> None:
    runtime = _recipe().stream().prepare()
    state = runtime.start(jr.key(0), target=2, attempt_budget=4)
    runtime.write_checkpoint(tmp_path / "state.npz", state)
    other = _recipe(conformer_weights=(3.0, 1.0)).stream().prepare()
    with pytest.raises(ValueError):
        other.read_checkpoint(tmp_path / "state.npz")


def test_fresh_process_recipe_continues_partly_active_generation(
    tmp_path: Path, _cpu: None
) -> None:
    recipe_path = tmp_path / "packing.recipe"
    checkpoint = tmp_path / "generation.npz"
    output = tmp_path / "continued.npz"
    recipe = _recipe(campaign_id="recipe-restart")
    written = write_crystal_packing_recipe(recipe_path, recipe)
    runtime = recipe.stream().prepare()
    state = runtime.start(jr.key(3), target=10, attempt_budget=12)
    state = runtime.run(
        state, AtomisticMemoryGenerationSink("before"), maximum_transitions=1
    )
    for _ in range(12):
        state = runtime.transition(state)
        if runtime.ready(state):
            break
    # Pending unpublished proposals and still-running lanes at the checkpoint.
    assert runtime.ready(state)
    assert state.stop_reason is AtomisticGenerationStopReason.RUNNING
    assert int(state.cursor.next_ordinal) > int(state.cursor.watermark)
    assert int(state.cursor.next_ordinal) < int(state.cursor.budget)
    runtime.write_checkpoint(checkpoint, state)
    expected = worker.continuation_arrays(runtime, state)
    assert expected["published_ids"].size > expected["pending_ids"].size > 0

    completed = subprocess.run(
        [sys.executable, str(WORKER), str(recipe_path), str(checkpoint), str(output)],
        cwd=REPOSITORY,
        env={
            **os.environ,
            "PYTHONPATH": str(REPOSITORY),
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
        },
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output) as archive:
        assert str(archive["recipe_id"]) == written.recipe_id
        assert str(archive["stream_id"]) == runtime.stream_id
        observed = {
            name: archive[name]
            for name in archive.files
            if name not in ("recipe_id", "stream_id")
        }
    assert set(observed) == set(expected)
    for name, value in expected.items():
        assert observed[name].shape == value.shape, name
        np.testing.assert_array_equal(observed[name], value, err_msg=name)
    np.testing.assert_array_equal(
        expected["published_ordinals"][: expected["pending_ordinals"].size],
        expected["pending_ordinals"],
    )
