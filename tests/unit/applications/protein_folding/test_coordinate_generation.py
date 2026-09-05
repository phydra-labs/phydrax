import hashlib
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.applications.protein_folding._construct import (
    ProteinAtomKey,
    ProteinConstruct,
)
from phydrax.applications.protein_folding._hypotheses import ProteinSourceAtom
from phydrax.applications.protein_folding.generation import (
    CoordinateGeometryPolicy,
    CoordinateProviderProvenance,
    fit_coordinate_model,
    import_protein_hypotheses,
    load_coordinate_model,
    map_protein_hypothesis,
    prepare_coordinate_sampler,
    prepare_coordinate_training_data,
    prepare_protein_coordinate_support,
    qualify_coordinate_proposals,
    sample_coordinate_proposals,
    save_coordinate_model,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.atomistic import AtomisticBatch, AtomisticScaleContract
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM, ELECTRONVOLT


def _rights(payload=b"original numerical fixture", *, training=True, export=True):
    return ReferenceArtifactManifest(
        "original-synthetic-coordinate-fixture",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Phydrax-OriginalSyntheticFixture",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=training,
        export_permitted=export,
        export_classification="test-declaration",
        nondimensionalization={"coordinate_angstrom": 1.0},
        uncertainty=None,
        lineage_ids=("original-analytic-fixture",),
    )


def _fixture():
    construct = ProteinConstruct(("chain",), ("A",))
    keys = tuple(
        ProteinAtomKey(construct.residue_keys[0], name) for name in ("CA", "N", "C", "CB")
    )
    ids = (101, 809, 405, 222)
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    batch = AtomisticBatch(
        np.array([[6, 7, 6, 6]]),
        x[None],
        np.array([[12.0, 14.0, 12.0, 12.0]]),
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        particle_ids=np.array([ids]),
    )
    geometry = CoordinateGeometryPolicy(
        ((101, 809), (101, 405), (101, 222)),
        ((0.2, 2.0), (0.2, 2.0), (0.2, 2.0)),
        ((101, 809, 405, 222),),
        (1,),
        0.01,
        "analytic-tetrahedron",
    )
    atom_ids = dict(zip(keys, ids, strict=True))
    support = prepare_protein_coordinate_support(
        construct, batch, atom_ids, gauge_atom_ids=(101, 809, 405), geometry=geometry
    )
    return construct, keys, atom_ids, support, x


def _data(support, x, rights=None):
    conditions = np.linspace(-1.0, 1.0, 12)[:, None]
    positions = np.broadcast_to(x, (12, 4, 3)).copy()
    positions[:, 3, 2] += 0.25 * conditions[:, 0]
    manifest = _rights(positions.tobytes()) if rights is None else rights
    return prepare_coordinate_training_data(
        support,
        positions,
        conditions,
        condition_names=("analytic_deformation",),
        record_ids=tuple(f"sample-{i}" for i in range(12)),
        source_manifest_ids=(manifest.manifest_id,) * 12,
        split_group_ids=tuple("heldout" if i % 4 == 0 else "train" for i in range(12)),
        validation_groups=("heldout",),
        rights=(manifest,),
        corpus_description="Independent held-out analytic parameter values; not a scientific protein corpus.",
    )


def _source(manifest, name="raw"):
    return ScientificArtifactEnvelope(
        artifact_kind="user-coordinate-output",
        content_digest=manifest.checksum,
        producer="original-fixture",
        producer_version="native",
        build_id="test-fixture",
        license_id=manifest.license_id,
        resource_id=name,
        status="complete",
    )


def test_gauge_preserves_handedness_and_proper_rigid_invariance():
    _, _, _, support, x = _fixture()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    first, valid = eqx.filter_jit(support.canonicalize)(jnp.asarray(x))
    moved, moved_valid = support.canonicalize(x @ rotation + np.array([2.0, -8.0, 3.0]))
    assert valid and moved_valid
    assert jnp.allclose(first, moved, atol=1e-12)
    reflected = x * np.array([1.0, 1.0, -1.0])
    report = eqx.filter_jit(lambda points: qualify_coordinate_proposals(support, points))(
        jnp.stack((jnp.asarray(x), jnp.asarray(reflected)))
    )
    assert np.array_equal(report.accepted, [True, False])
    assert np.array_equal(report.chirality_valid, [True, False])
    gradient = eqx.filter_grad(
        lambda points: jnp.sum(support.canonicalize(points)[0] ** 2)
    )(jnp.asarray(x))
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(jnp.sum(gradient, axis=0), 0.0, atol=1e-10)


def test_offline_provider_mapping_reorders_without_erasing_parent_rights():
    construct, keys, mapping, support, x = _fixture()
    output, restricted_weights = _rights(), _rights(b"restricted-weights", training=False)
    records = tuple(
        ProteinSourceAtom(str(i), key, "model", "A", "17", "", "", 1.0, element)
        for i, (key, element) in enumerate(zip(keys, (6, 7, 6, 6), strict=True))
    )
    order = (2, 0, 3, 1)
    provenance = CoordinateProviderProvenance(
        "offline-caller-output",
        (output,),
        weight_rights=(restricted_weights,),
        input_rights=(output,),
        input_artifact_ids=("prepared-query",),
        learned_model=True,
    )
    result = import_protein_hypotheses(
        construct,
        tuple(records[i] for i in order),
        x[np.array(order)][None],
        ANGSTROM,
        (_source(output),),
        provenance=provenance,
    )
    mapped = map_protein_hypothesis(result.hypotheses[0], support, mapping)
    assert jnp.array_equal(mapped, x)
    assert jnp.array_equal(result.hypotheses[0].positions, x[np.array(order)])
    with pytest.raises(PermissionError):
        map_protein_hypothesis(result.hypotheses[0], support, mapping, training_use=True)
    with pytest.raises(PermissionError):
        import_protein_hypotheses(
            construct,
            records,
            x[None],
            ANGSTROM,
            (_source(output),),
            provenance=replace(
                provenance, egress_destination="https://unapproved.example"
            ),
        )
    with pytest.raises(PermissionError):
        import_protein_hypotheses(
            construct,
            records,
            x[None],
            ANGSTROM,
            (_source(output),),
            provenance=replace(provenance, weight_rights=()),
        )
    bad_mapping = dict(mapping)
    bad_mapping[keys[0]], bad_mapping[keys[1]] = mapping[keys[1]], mapping[keys[0]]
    with pytest.raises(ValueError):
        map_protein_hypothesis(result.hypotheses[0], support, bad_mapping)


def test_training_refuses_restrictions_and_validation_leakage():
    _, _, _, support, x = _fixture()
    with pytest.raises(PermissionError):
        _data(support, x, _rights(training=False))
    rights = _rights()
    with pytest.raises(ValueError):
        prepare_coordinate_training_data(
            support,
            np.stack((x, x)),
            np.array([[0.0], [1.0]]),
            condition_names=("condition",),
            record_ids=("train-record", "test-record"),
            source_manifest_ids=(rights.manifest_id,) * 2,
            split_group_ids=("train", "test"),
            validation_groups=("test",),
            rights=(rights,),
            corpus_description="deliberately leaked fixture",
        )
    with pytest.raises(ValueError):
        prepare_protein_coordinate_support(
            ProteinConstruct(("chain",), ("A",)),
            support.template,
            {},
            gauge_atom_ids=(101, 809, 405),
            geometry=support.geometry,
        )


def test_actual_native_training_sampling_and_weight_admission(tmp_path):
    _, _, _, support, x = _fixture()
    data = _data(support, x)
    fit = fit_coordinate_model(
        data,
        key=jr.key(11),
        steps=120,
        width=24,
        depth=2,
        pairs_per_step=24,
        learning_rate=4e-3,
    )
    assert fit.final_training_loss < 0.85 * fit.initial_training_loss
    assert np.isfinite(fit.validation_loss)
    context = jnp.array([[-0.5], [0.5]])
    proposals = sample_coordinate_proposals(fit, jr.key(12), context)
    replay = sample_coordinate_proposals(fit, jr.key(12), context)
    assert jnp.array_equal(proposals.raw_positions, replay.raw_positions)
    assert jnp.all(proposals.solver_valid)
    assert proposals.raw_positions.shape == (2, 4, 3)
    masses = support.template.masses[0]
    assert jnp.allclose(
        jnp.sum(proposals.raw_positions * masses[None, :, None], axis=1), 0.0, atol=1e-7
    )
    sampler = prepare_coordinate_sampler(fit)
    derivative = eqx.filter_grad(
        lambda condition: jnp.sum(sampler(jr.key(7), condition)[0] ** 2)
    )(context)
    assert jnp.all(jnp.isfinite(derivative))
    assert jnp.any(jnp.abs(derivative) > 1e-7)
    destination = tmp_path / "coordinate-model.phxml"
    save_coordinate_model(destination, fit)
    weight_rights = _rights(destination.read_bytes())
    restored = load_coordinate_model(destination, support, weight_rights=weight_rights)
    restored_samples = sample_coordinate_proposals(restored, jr.key(12), context)
    assert jnp.array_equal(proposals.raw_positions, restored_samples.raw_positions)
    with pytest.raises(ValueError):
        load_coordinate_model(
            destination, support, weight_rights=_rights(b"different bytes")
        )
