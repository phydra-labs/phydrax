import hashlib
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax import ValuePort
from phydrax.applications.protein_folding._construct import (
    ProteinAtomKey,
    ProteinConstruct,
)
from phydrax.applications.protein_folding._hypotheses import ProteinSourceAtom
from phydrax.applications.protein_folding.generation import (
    CoordinateGeometryFilter,
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
from phydrax.atomistic import (
    AtomisticBatch,
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticConditionKind,
    AtomisticScaleContract,
)
from phydrax.atomistic._generation import (
    atomistic_proposal_id,
    AtomisticCellAdmission,
    AtomisticChemistryAdmission,
    AtomisticConditionSchedule,
    AtomisticGenerationStreamPlan,
    AtomisticMemoryGenerationSink,
    AtomisticProposalBlock,
    AtomisticProposalStatus,
    AtomisticTopologyAdmission,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM, DIMENSIONLESS, ELECTRONVOLT


_CAMPAIGN = "original-analytic-coordinate-campaign"


def _spec(name: str = "analytic_deformation") -> AtomisticConditioningSpec:
    return AtomisticConditioningSpec(
        (
            AtomisticConditionField(
                name,
                ValuePort(
                    "test.coordinate-generation." + name,
                    event_shape=(),
                    component_ids=("value",),
                    dimensions=(DIMENSIONLESS,),
                    representation="declared-dimensionless-deformation",
                    normalization_id="identity",
                ),
            ),
        )
    )


def _sample(
    fit: Any,
    values: Any,
    *,
    campaign_id: str = _CAMPAIGN,
    start_ordinal: int = 0,
    lanes_per_member: int = 2,
    sink: AtomisticMemoryGenerationSink | None = None,
    export: bool = False,
) -> Any:
    return sample_coordinate_proposals(
        fit,
        jr.key(12),
        {"analytic_deformation": jnp.asarray(values)},
        campaign_id=campaign_id,
        start_ordinal=start_ordinal,
        lanes_per_member=lanes_per_member,
        sink=AtomisticMemoryGenerationSink("test-memory") if sink is None else sink,
        export=export,
    )


def _rights(
    payload: Any = b"original numerical fixture",
    *,
    training: Any = True,
    export: Any = True,
) -> Any:
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


def _fixture() -> Any:
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


def _data(support: Any, x: Any, rights: Any = None) -> Any:
    conditions = np.linspace(-1.0, 1.0, 12)
    positions = np.broadcast_to(x, (12, 4, 3)).copy()
    positions[:, 3, 2] += 0.25 * conditions
    manifest = _rights(positions.tobytes()) if rights is None else rights
    return prepare_coordinate_training_data(
        support,
        positions,
        {"analytic_deformation": conditions},
        conditioning=_spec(),
        record_ids=tuple(f"sample-{i}" for i in range(12)),
        source_manifest_ids=(manifest.manifest_id,) * 12,
        split_group_ids=tuple("heldout" if i % 4 == 0 else "train" for i in range(12)),
        validation_groups=("heldout",),
        rights=(manifest,),
        corpus_description="Independent held-out analytic parameter values; not a scientific protein corpus.",
    )


def _source(manifest: Any, name: Any = "raw") -> Any:
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


def test_coordinate_generation_scenario_1() -> None:
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
    _, _, _, support, x = _fixture()
    with pytest.raises(PermissionError):
        _data(support, x, _rights(training=False))
    rights = _rights()
    with pytest.raises(ValueError, match="leak"):
        prepare_coordinate_training_data(
            support,
            np.stack((x, x)),
            {"condition": np.array([0.0, 1.0])},
            conditioning=_spec("condition"),
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


@pytest.fixture(scope="module")
def fitted() -> Any:
    _, _, _, support, x = _fixture()
    return fit_coordinate_model(
        _data(support, x),
        key=jr.key(11),
        steps=120,
        width=24,
        depth=2,
        pairs_per_step=24,
        learning_rate=4e-3,
    )


def test_actual_native_training_sampling_and_weight_admission(
    tmp_path: Any, fitted: Any
) -> None:
    fit = fitted
    support = fit.support
    assert fit.final_training_loss < 0.85 * fit.initial_training_loss
    assert np.isfinite(fit.validation_loss)
    sink = AtomisticMemoryGenerationSink("training-memory")
    proposals = _sample(fit, [-0.5, 0.5], sink=sink)
    replay = _sample(fit, [-0.5, 0.5])
    assert jnp.array_equal(proposals.raw_positions, replay.raw_positions)
    assert jnp.all(proposals.solver_valid)
    assert proposals.raw_positions.shape == (2, 4, 3)
    assert proposals.ordinals == (0, 1)
    assert proposals.condition_names == ("analytic_deformation",)
    assert jnp.array_equal(proposals.conditions[:, 0], jnp.array([-0.5, 0.5]))
    # Identity is (campaign, generation support, sampler, ordinal), independently derived.
    assert proposals.sample_ids == tuple(
        atomistic_proposal_id(
            _CAMPAIGN, proposals.generation_support_id, proposals.sampler_id, ordinal
        )
        for ordinal in (0, 1)
    )
    assert tuple(record.proposal_id for record in sink.records) == proposals.sample_ids
    assert jnp.array_equal(
        proposals.qualification.accepted, proposals.proposal_status == 0
    )
    masses = support.template.masses[0]
    assert jnp.allclose(
        jnp.sum(proposals.raw_positions * masses[None, :, None], axis=1), 0.0, atol=1e-7
    )
    assert proposals.likelihood_capability.startswith("unavailable")
    assert "not molecular-dynamics time" in proposals.time_semantics
    assert "no permutation-equivariance claim" in proposals.symmetry_semantics
    sampler = prepare_coordinate_sampler(fit)
    assert sampler.idempotent and not sampler.stateful
    keys = jr.split(jr.key(7), 2)

    def energy(values: Any) -> Any:
        conditioning = AtomisticConditioning(
            fit.conditioning,
            {"analytic_deformation": values},
            case_count=2,
            atom_capacity=support.template.atom_capacity,
        )
        return jnp.sum(sampler.propose(keys, jnp.arange(2), conditioning).positions ** 2)

    derivative = eqx.filter_grad(energy)(jnp.array([-0.5, 0.5]))
    assert jnp.all(jnp.isfinite(derivative))
    assert jnp.any(jnp.abs(derivative) > 1e-7)
    destination = tmp_path / "coordinate-model.phxml"
    save_coordinate_model(destination, fit)
    weight_rights = _rights(destination.read_bytes())
    restored = load_coordinate_model(
        destination, support, weight_rights=weight_rights, conditioning=fit.conditioning
    )
    restored_samples = _sample(restored, [-0.5, 0.5])
    assert jnp.array_equal(proposals.raw_positions, restored_samples.raw_positions)
    assert restored_samples.sample_ids == proposals.sample_ids
    with pytest.raises(ValueError):
        load_coordinate_model(
            destination,
            support,
            weight_rights=_rights(b"different bytes"),
            conditioning=fit.conditioning,
        )
    with pytest.raises(ValueError, match="condition ports"):
        load_coordinate_model(
            destination,
            support,
            weight_rights=weight_rights,
            conditioning=_spec("other_deformation"),
        )


def test_ordinal_addressed_proposals_ignore_lane_and_chunk_layout(fitted: Any) -> None:
    values = np.linspace(-0.6, 0.6, 5)
    whole = _sample(fitted, values, lanes_per_member=5)
    paired = _sample(fitted, values, lanes_per_member=2)
    head = _sample(fitted, values[:2], lanes_per_member=3)
    tail = _sample(fitted, values[2:], start_ordinal=2, lanes_per_member=1)
    assert whole.ordinals == paired.ordinals == (0, 1, 2, 3, 4)
    assert head.ordinals + tail.ordinals == whole.ordinals
    assert paired.sample_ids == whole.sample_ids
    assert head.sample_ids + tail.sample_ids == whole.sample_ids
    chunked = jnp.concatenate((head.raw_coordinates, tail.raw_coordinates))
    # Batched solver arithmetic may differ by rounding across lane counts only.
    np.testing.assert_allclose(
        paired.raw_coordinates, whole.raw_coordinates, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(chunked, whole.raw_coordinates, atol=1e-6, rtol=0)
    # Independent control: another campaign addresses other noise and identities.
    other = _sample(fitted, values, lanes_per_member=5, campaign_id="other-campaign")
    assert set(other.sample_ids).isdisjoint(whole.sample_ids)
    assert float(jnp.max(jnp.abs(other.raw_coordinates - whole.raw_coordinates))) > 1e-3
    shifted = _sample(fitted, values, lanes_per_member=5, start_ordinal=1)
    assert shifted.sample_ids[:4] == whole.sample_ids[1:]
    assert (
        float(jnp.max(jnp.abs(shifted.raw_coordinates[0] - whole.raw_coordinates[0])))
        > 1e-3
    )


def test_geometry_filter_rejections_are_retained_with_reasons(fitted: Any) -> None:
    construct, _, atom_ids, support, x = _fixture()
    reflected = x * np.array([1.0, 1.0, -1.0])
    screen = CoordinateGeometryFilter(support)
    block = AtomisticProposalBlock(
        jnp.arange(2),
        jnp.stack((jnp.asarray(x), jnp.asarray(reflected))),
        jnp.broadcast_to(jnp.array([6, 7, 6, 6]), (2, 4)),
        jnp.ones((2, 4), dtype=bool),
        None,
        jnp.zeros((2,), dtype=jnp.int32),
        (),
        sampler_id="hand-built-control",
        payload_names=(),
    )
    evidence = screen.evaluate(block)
    assert np.array_equal(evidence.keep, [True, False])
    assert np.array_equal(evidence.value, [0.0, 8.0])
    report = screen.qualification(evidence.value, jnp.array([0, 1]))
    assert np.array_equal(report.chirality_valid, [True, False])
    assert np.array_equal(report.bond_valid, [True, True])
    assert np.array_equal(report.accepted, [True, False])
    impossible = prepare_protein_coordinate_support(
        construct,
        support.template,
        atom_ids,
        gauge_atom_ids=(101, 809, 405),
        geometry=replace(
            support.geometry,
            bond_bounds=((50.0, 60.0),) * 3,
            policy_id="unsatisfiable-bond-screen",
        ),
    )
    stream = AtomisticGenerationStreamPlan(
        prepare_coordinate_sampler(fitted),
        campaign_id="rejection-campaign",
        lanes_per_member=2,
        pending_capacity=4,
        conditions=AtomisticConditionSchedule(
            fitted.conditioning,
            {"analytic_deformation": jnp.array([-0.5, 0.5, 0.0])},
            policy="bounded",
        ),
        filters=(CoordinateGeometryFilter(impossible),),
    ).prepare()
    sink = AtomisticMemoryGenerationSink("rejections")
    state = stream.run(
        stream.start(jr.key(3), target=3, attempt_budget=3), sink, maximum_transitions=8
    )
    assert stream.finished(state)
    assert stream.shortfall(state) == 3
    records = sink.records
    assert tuple(record.ordinal for record in records) == (0, 1, 2)
    for record in records:
        assert bool(record.value("solver_valid"))
        assert record.filter_bits == 1
        assert record.status & AtomisticProposalStatus.GEOMETRY_INVALID
        assert record.status & AtomisticProposalStatus.FILTER_REJECTED
        assert int(record.filter_values[0]) & 4
        assert record.positions.shape == (4, 3)
        assert np.all(np.isfinite(record.positions))


def test_typed_conditions_refuse_missing_undeclared_and_invalid_fields(
    fitted: Any,
) -> None:
    sink = AtomisticMemoryGenerationSink("refusals")

    def sample(conditions: dict[str, Any]) -> Any:
        return sample_coordinate_proposals(
            fitted,
            jr.key(1),
            conditions,
            campaign_id=_CAMPAIGN,
            start_ordinal=0,
            lanes_per_member=1,
            sink=sink,
        )

    with pytest.raises(ValueError, match="missing"):
        sample({})
    with pytest.raises(ValueError, match="undeclared"):
        sample({"analytic_deformation": jnp.zeros(1), "temperature": jnp.zeros(1)})
    with pytest.raises(ValueError, match="finite"):
        sample({"analytic_deformation": jnp.array([jnp.nan])})
    with pytest.raises(ValueError, match="one scalar value per case"):
        sample({"analytic_deformation": jnp.zeros((1, 1))})
    assert sink.records == ()
    categorical = AtomisticConditioningSpec(
        (
            AtomisticConditionField(
                "construct_class",
                ValuePort(
                    "test.coordinate-generation.construct-class",
                    event_shape=(),
                    component_ids=("code",),
                    dimensions=(DIMENSIONLESS,),
                    representation="declared-category-code",
                    normalization_id="identity",
                ),
                kind=AtomisticConditionKind.CATEGORICAL,
                dtype=jnp.int32,
                categories=(0, 1),
            ),
        )
    )
    with pytest.raises(ValueError, match="continuous scalar case-level"):
        fitted.support.generation_support(categorical)


def test_generation_support_binds_fixed_template_chemistry(fitted: Any) -> None:
    support = fitted.support
    generation = support.generation_support(fitted.conditioning)
    assert generation.chemistry is AtomisticChemistryAdmission.FIXED
    assert generation.topology is AtomisticTopologyAdmission.FIXED
    assert generation.cell_admission is AtomisticCellAdmission.NONPERIODIC
    assert generation.fixed_species == (6, 7, 6, 6)
    assert generation.fixed_bonds == support.bond_indices
    assert generation.species_vocabulary == (6, 7)
    assert generation.vocabulary_masses == (12.0, 14.0)
    assert support.support_id in generation.mass_provenance
    assert generation.conditioning.spec_id == fitted.conditioning.spec_id
    assert prepare_coordinate_sampler(fitted).support.support_id == generation.support_id
    construct, _, atom_ids, _, x = _fixture()
    isotopic = AtomisticBatch(
        np.array([[6, 7, 6, 6]]),
        x[None],
        np.array([[12.0, 14.0, 13.0, 12.0]]),
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        particle_ids=np.array([(101, 809, 405, 222)]),
    )
    mixed = prepare_protein_coordinate_support(
        construct,
        isotopic,
        atom_ids,
        gauge_atom_ids=(101, 809, 405),
        geometry=support.geometry,
    )
    with pytest.raises(ValueError, match="one template mass per element"):
        mixed.generation_support(fitted.conditioning)


def test_rights_refusal_is_preserved_for_stream_sampling(fitted: Any) -> None:
    restricted = replace(fitted, rights=(_rights(export=False),))
    sink = AtomisticMemoryGenerationSink("rights")
    with pytest.raises(PermissionError):
        prepare_coordinate_sampler(restricted, export=True)
    with pytest.raises(PermissionError):
        _sample(restricted, [0.0], export=True, sink=sink)
    assert sink.records == ()
    assert _sample(restricted, [0.0]).sample_ids
