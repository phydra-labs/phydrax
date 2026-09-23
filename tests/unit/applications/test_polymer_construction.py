import hashlib
import json

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._artifact_security import (
    admit_external_artifact,
    ExternalArtifactPolicy,
)
from phydrax.applications import polymer_construction as pc
from phydrax.artifacts import ArtifactManifest


def _source():
    return {
        "material_id": "two-chain-network",
        "bead_types": [{"id": "A", "mass": 1.0}],
        "chains": [
            {"id": "left", "sequence": [0, 0]},
            {"id": "right", "sequence": [0, 0]},
        ],
        "ports": [
            {
                "id": "left-0",
                "chain_id": "left",
                "bead_offset": 0,
                "compatibility_class": "donor",
            },
            {
                "id": "left-1",
                "chain_id": "left",
                "bead_offset": 1,
                "compatibility_class": "donor",
            },
            {
                "id": "right-0",
                "chain_id": "right",
                "bead_offset": 0,
                "compatibility_class": "acceptor",
            },
            {
                "id": "right-1",
                "chain_id": "right",
                "bead_offset": 1,
                "compatibility_class": "acceptor",
            },
        ],
        "maximum_particles": 6,
        "particle_id_start": 100,
    }


def test_recipe_adapter_and_lowering_preserve_explicit_material_semantics():
    adapted = pc.polymer_recipe_from_mapping(
        _source(), phx.atomistic.AtomisticUnitSystem.reduced()
    )
    construction = pc.lower_polymer_recipe(adapted.recipe)

    assert adapted.report.valid & construction.successful
    np.testing.assert_array_equal(construction.system.active_mask, [1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(construction.topology.bonds, [[100, 101], [102, 103]])
    np.testing.assert_allclose(construction.statistics.dispersity, 1.0)
    assert construction.lowering.chain_particle_ids == ((100, 101), (102, 103))
    assert (
        pc.polymer_recipe_to_mapping(adapted.recipe)["material_id"] == "two-chain-network"
    )


def test_admitted_recipe_requires_the_exact_trusted_manifest(tmp_path):
    payload = json.dumps(_source()).encode()
    (tmp_path / "recipe.json").write_bytes(payload)
    manifest = ArtifactManifest(
        artifact_id="polymer-recipe",
        producer="independent-test",
        version="1",
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        source_uri="https://example.invalid/polymer-recipe",
        license_id="CC-BY-4.0",
        model="polymer-recipe-json",
        coverage="unit-test",
    )
    policy = ExternalArtifactPolicy(
        tmp_path,
        maximum_bytes=4096,
        allowed_license_ids=("CC-BY-4.0",),
        allowed_suffixes=(".json",),
    )
    admitted = admit_external_artifact("recipe.json", manifest, policy=policy)
    result = pc.polymer_recipe_from_admitted_json(
        admitted,
        manifest,
        phx.atomistic.AtomisticUnitSystem.reduced(),
        policy=policy,
    )

    assert result.recipe.material_id == "two-chain-network"


def test_nonperiodic_reaction_epoch_is_atomic_and_network_observable():
    adapted = pc.polymer_recipe_from_mapping(
        _source(), phx.atomistic.AtomisticUnitSystem.reduced()
    )
    state = pc.initialize_polymer_reaction_state(pc.lower_polymer_recipe(adapted.recipe))
    template = pc.PolymerReactionTemplate("crosslink", "donor", "acceptor")

    accepted = pc.apply_polymer_reaction(state, template, "left-0", "right-0")
    refused = pc.apply_polymer_reaction(accepted.state, template, "left-0", "right-1")
    network = pc.polymer_network_observables(refused.state)

    assert accepted.successful
    assert not refused.successful
    assert refused.event.status is pc.PolymerReactionStatus.EXHAUSTED_PORT
    assert network.successful
    assert int(network.component_count) == 1
    assert int(network.accepted_events) == 1
    assert int(network.refused_events) == 1
    repair = pc.apply_polymer_reaction(
        refused.state,
        pc.PolymerReactionTemplate(
            "repair-crosslink",
            "donor",
            "acceptor",
            reaction_kind=pc.PolymerReactionKind.REPAIR,
        ),
        "left-0",
        "right-0",
    )
    repaired_network = pc.polymer_network_observables(repair.state)
    assert repair.successful
    assert repair.event.reaction_kind is pc.PolymerReactionKind.REPAIR
    assert int(repaired_network.component_count) == 2
    np.testing.assert_allclose(repaired_network.conversion, 0.0)


def test_periodic_cure_requires_explicit_winding_and_detects_spanning_cycle():
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 8.0)
    adapted = pc.polymer_recipe_from_mapping(
        _source(), phx.atomistic.AtomisticUnitSystem.reduced(), cell=cell
    )
    construction = pc.lower_polymer_recipe(adapted.recipe)
    state = pc.initialize_polymer_reaction_state(construction)
    template = pc.PolymerReactionTemplate("periodic-crosslink", "donor", "acceptor")

    refused = pc.apply_polymer_reaction(state, template, "left-0", "right-0")
    assert not refused.successful
    assert refused.event.status is pc.PolymerReactionStatus.PERIODIC_WINDING_REQUIRED

    first = pc.apply_polymer_reaction(
        state,
        template,
        "left-0",
        "right-0",
        periodic_image_shift=(1, 0, 0),
    )
    second = pc.apply_polymer_reaction(
        first.state,
        template,
        "left-1",
        "right-1",
        periodic_image_shift=(0, 0, 0),
    )
    network = pc.polymer_network_observables(second.state)

    assert first.successful & second.successful
    assert network.successful
    assert int(network.cycle_rank) == 1
    assert network.periodic_spanning
    np.testing.assert_array_equal(second.state.image_counts[2:4], [[1, 0, 0], [1, 0, 0]])
