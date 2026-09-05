import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nucleic_acid_biophysics import (
    BaseInteraction,
    BaseInteractionGraph,
    normalize_nucleic_hypothesis,
    NucleicAcidConstruct,
    NucleicStructureHypothesis,
    NucleotideAtomMapping,
    NucleotideKey,
    prepare_nucleotide_binding,
)
from phydrax.applications.nucleic_acid_biophysics.structure import (
    base_frames,
    ERMSDCollectiveVariableProgram,
    geometric_contacts,
    GeometricContactCriteria,
    NucleotideGDescriptor,
    NucleotideTorsionEvaluation,
    NucleotideTorsionProgram,
    sugar_pseudorotation,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.atomistic import AtomisticSystemPlan, AtomisticUnitSystem
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.series import SampledSeries, SeriesPairView, SeriesSupport
from phydrax.units import ANGSTROM, METER


def fixture():
    construct = NucleicAcidConstruct(("rna",), ("AUC",), ("RNA",), (False,))
    ring = np.array([[1.0, 0.0, 0.0], [-0.5, 0.8, 0.0], [-0.5, -0.8, 0.0]])
    centers = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.4], [30.0, 0.0, 0.0]])
    keys = tuple(key for key in construct.nucleotide_keys for _ in range(3))
    names = ("C2", "C6", "C4", "C2", "C4", "C6", "C2", "C4", "C6")
    mapping = NucleotideAtomMapping(
        construct, (90, 22, 700, 11, 850, 31, 809, 27, 301), keys, names
    )
    positions = (ring[None] + centers[:, None]).reshape((-1, 3))
    return mapping, jnp.asarray(positions)


def rights():
    payload = b"independently authored geometric fixture"
    return ReferenceArtifactManifest(
        "synthetic geometry",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"length_angstrom": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic-geometry",),
    )


def test_construct_connectivity_chemistry_and_full_graph_refusal():
    construct = NucleicAcidConstruct(
        ("dna", "rna"), ("ACGT", "ACGU"), ("DNA", "RNA"), (True, False)
    )
    assert (NucleotideKey("dna", 3), NucleotideKey("dna", 0)) in construct.directed_edges
    assert (
        NucleotideKey("rna", 3),
        NucleotideKey("rna", 0),
    ) not in construct.directed_edges
    assert construct.nucleotide_count == 8
    with pytest.raises(ValueError):
        NucleicAcidConstruct(("rna",), ("ACT",), ("RNA",), (False,))
    linear = NucleicAcidConstruct(("r",), ("ACGU",), ("RNA",), (False,))
    k = linear.nucleotide_keys
    graph = BaseInteractionGraph(
        linear, (BaseInteraction(k[0], k[3], "pair", "canonical", "source"),)
    )
    assert graph.to_dot_bracket() == "(..)"
    crossing = BaseInteractionGraph(
        linear,
        (
            BaseInteraction(k[0], k[2], "pair", "canonical", "source"),
            BaseInteraction(k[1], k[3], "pair", "canonical", "source"),
        ),
    )
    with pytest.raises(ValueError):
        crossing.to_dot_bracket()
    multi = BaseInteractionGraph(
        linear,
        graph.interactions
        + (BaseInteraction(k[0], k[1], "pair", "Hoogsteen", "source"),),
    )
    with pytest.raises(ValueError):
        multi.to_dot_bracket()
    assert len(multi.interactions) == 2


def test_published_frame_order_and_proper_rigid_invariance():
    mapping, positions = fixture()
    binding = prepare_nucleotide_binding(mapping, mapping.atom_ids)
    descriptor = NucleotideGDescriptor(
        binding, length_unit=ANGSTROM, image_policy="nonperiodic"
    )
    reference = descriptor.evaluate(positions)
    np.testing.assert_allclose(
        base_frames(positions, binding, image_policy="nonperiodic").axes,
        np.tile(np.eye(3), (3, 1, 1)),
        atol=1e-12,
    )
    rotation = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = positions @ rotation.T + jnp.array([8.0, -3.0, 1.0])
    np.testing.assert_allclose(
        descriptor.evaluate(transformed).values, reference.values, atol=1e-12
    )
    permutation = np.array([8, 2, 4, 1, 5, 0, 7, 3, 6])
    permuted_binding = prepare_nucleotide_binding(
        mapping, tuple(mapping.atom_ids[i] for i in permutation)
    )
    permuted = NucleotideGDescriptor(
        permuted_binding, length_unit=ANGSTROM, image_policy="nonperiodic"
    )
    np.testing.assert_allclose(
        permuted.evaluate(positions[permutation]).values, reference.values, atol=1e-12
    )
    # Direction is a local source-frame vector; reversal is not symmetrization.
    np.testing.assert_allclose(
        reference.values[0, :3], -reference.values[2, :3], atol=1e-12
    )
    assert bool(jnp.any(reference.values[0, :3] != reference.values[2, :3]))


def test_sparse_dense_equivalence_and_native_cv_series_support():
    mapping, positions = fixture()
    binding = prepare_nucleotide_binding(mapping, mapping.atom_ids)
    k = mapping.construct.nucleotide_keys
    dense = NucleotideGDescriptor(
        binding, length_unit=ANGSTROM, image_policy="nonperiodic"
    )
    sparse = NucleotideGDescriptor(
        binding,
        length_unit=ANGSTROM,
        pairs=((k[0], k[1]), (k[1], k[0])),
        image_policy="nonperiodic",
    )
    moved = positions.at[3:6, 1].add(0.25)
    np.testing.assert_allclose(
        sparse.compare(moved, positions).value,
        dense.compare(moved, positions).value,
        atol=1e-12,
    )
    expected = jnp.sqrt(
        jnp.sum((sparse.evaluate(moved).values - sparse.evaluate(positions).values) ** 2)
        / 3
    )
    np.testing.assert_allclose(sparse.compare(moved, positions).value, expected)
    gradient = jax.jit(jax.grad(lambda x: sparse.compare(x, positions).squared_distance))(
        moved
    )
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_allclose(jnp.sum(gradient, axis=0), 0.0, atol=1e-12)
    support = SeriesSupport(
        jnp.array([0.0, 1.0, 0.0, 1.0]),
        edge_valid=jnp.array([True, False, True]),
        coordinate_name="time",
        coordinate_id="two-trajectories",
    )
    coordinates = SampledSeries(
        support,
        jnp.stack((positions, moved, positions, moved)),
        series_id="explicit-source-coordinates",
    )
    features = sparse.observe_series(coordinates)
    np.testing.assert_array_equal(
        SeriesPairView.from_lag(features, 1).valid, [True, False, True]
    )
    mask = jnp.ones(coordinates.values.shape, bool).at[1, 0, :].set(False)
    incomplete = sparse.observe_series(
        SampledSeries(
            support, coordinates.values, value_valid=mask, series_id="missing-marker"
        )
    )
    assert not bool(jnp.any(incomplete.value_valid[1]))


def test_ermsd_drives_native_harmonic_bias_energy_and_forces():
    from phydrax.atomistic import (
        AtomisticDynamicsPlan,
        AtomisticPotentialProgram,
        LennardJonesPotential,
        VelocityVerletPlan,
    )
    from phydrax.atomistic.sampling import (
        AtomisticBiasPlan,
        BiasKind,
        PreparedAtomisticBias,
    )
    from phydrax.discretization import DenseParticleNeighborhoodPlan

    mapping, reference = fixture()
    units = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    count = len(mapping.atom_ids)
    system = AtomisticSystemPlan(
        mapping.atom_ids, [6] * count, [12.0] * count, units, atom_type_ids=[0] * count
    ).prepare()
    neighborhood = DenseParticleNeighborhoodPlan(count * (count - 1) // 2).prepare(
        system.particles
    )
    potential = AtomisticPotentialProgram(
        [LennardJonesPotential([0.2], [1.0], 2.5)]
    ).prepare(system)
    dynamics = AtomisticDynamicsPlan(
        system, potential, neighborhood, VelocityVerletPlan(1e-3)
    ).prepare()
    descriptor = NucleotideGDescriptor(
        prepare_nucleotide_binding(mapping, system),
        length_unit=ANGSTROM,
        image_policy="nonperiodic",
    )
    program = ERMSDCollectiveVariableProgram(descriptor, reference)
    stiffness = 2.5
    bias = PreparedAtomisticBias(
        AtomisticBiasPlan(
            BiasKind.HARMONIC, program, center=[0.0], stiffness=[stiffness]
        ),
        dynamics,
    )
    state = bias.plan.initialize()
    positions = reference.at[3:6, 1].add(0.25)
    evaluate = eqx.filter_jit(bias.evaluate)
    result = evaluate(positions, state, jnp.asarray(0.0))
    assert bool(result.successful)
    expected_energy = (
        0.5 * stiffness * descriptor.compare(positions, reference).squared_distance
    )
    np.testing.assert_allclose(result.energy, expected_energy, atol=1e-12)
    # Independent directional finite difference through the actual bias consumer.
    direction = jnp.zeros_like(positions).at[3:6, 1].set(1.0)
    step = 1e-5
    plus = evaluate(positions + step * direction, state, jnp.asarray(0.0)).energy
    minus = evaluate(positions - step * direction, state, jnp.asarray(0.0)).energy
    np.testing.assert_allclose(
        jnp.sum(result.forces * direction),
        -(plus - minus) / (2 * step),
        rtol=1e-7,
        atol=1e-10,
    )
    np.testing.assert_allclose(jnp.sum(result.forces, axis=0), 0.0, atol=1e-12)
    # Loss of a required base frame propagates a failed bias, not stale forces.
    collapsed = positions.at[:3].set(positions[0])
    assert not bool(evaluate(collapsed, state, jnp.asarray(0.0)).successful)
    with pytest.raises(ValueError):
        program.evaluate(positions, cell_vectors=jnp.eye(3))


def test_cutoff_sides_and_distinct_c2_descriptor():
    mapping, positions = fixture()
    k = mapping.construct.nucleotide_keys
    binding = prepare_nucleotide_binding(mapping, mapping.atom_ids)
    hard = NucleotideGDescriptor(
        binding,
        length_unit=ANGSTROM,
        pairs=((k[0], k[1]),),
        cutoff=2.4,
        image_policy="nonperiodic",
    )
    smooth = NucleotideGDescriptor(
        binding,
        length_unit=ANGSTROM,
        pairs=((k[0], k[1]),),
        cutoff=2.4,
        smooth_width=0.4,
        image_policy="nonperiodic",
    )

    def place(distance):
        return positions.at[3:6].set(positions[:3] + jnp.array([distance, 0.0, 0.0]))

    left, right = hard.evaluate(place(12.0 - 1e-6)), hard.evaluate(place(12.0 + 1e-6))
    assert bool(left.within_cutoff[0]) and not bool(right.within_cutoff[0])
    np.testing.assert_allclose(left.values, right.values, atol=3e-7)
    assert bool(
        jnp.any(
            jnp.abs(
                hard.evaluate(place(11.0)).values - smooth.evaluate(place(11.0)).values
            )
            > 1e-4
        )
    )
    assert smooth.descriptor_id != hard.descriptor_id


def test_missing_and_degenerate_ring_never_shorten_construct():
    mapping, positions = fixture()
    mask = np.ones(9, bool)
    mask[1] = False
    missing = prepare_nucleotide_binding(mapping, mapping.atom_ids, coordinate_mask=mask)
    frame = base_frames(positions, missing, image_policy="nonperiodic")
    np.testing.assert_array_equal(frame.valid, [False, True, True])
    assert frame.centers.shape == (3, 3)
    full = prepare_nucleotide_binding(mapping, mapping.atom_ids)
    collapsed = positions.at[:3].set(jnp.array([1.0, 1.0, 1.0]))
    frame = base_frames(collapsed, full, image_policy="nonperiodic")
    assert not bool(frame.valid[0]) and bool(frame.covered[0])
    comparison = NucleotideGDescriptor(
        missing, length_unit=ANGSTROM, image_policy="nonperiodic"
    ).compare(positions, positions)
    assert not bool(comparison.successful)
    with pytest.raises(ValueError):
        prepare_nucleotide_binding(mapping, mapping.atom_ids[:-1])
    with pytest.raises(ValueError):
        base_frames(positions, full, image_policy="minimum-image")


def test_source_normalization_keeps_raw_and_restrictions():
    mapping, positions = fixture()
    manifest = rights()
    source = ScientificArtifactEnvelope(
        artifact_kind="raw-structure",
        content_digest="synthetic-positions",
        producer="independent-test",
        producer_version="native",
        build_id="fixture",
        license_id=manifest.license_id,
        resource_id=manifest.manifest_id,
        status="complete",
    )
    raw = NucleicStructureHypothesis(mapping, positions, ANGSTROM, source, manifest)
    normalized = normalize_nucleic_hypothesis(raw, length_unit=METER)
    assert normalized.raw is raw and normalized.normalized.parent is raw
    assert normalized.normalized.source.artifact_id != raw.source.artifact_id
    np.testing.assert_allclose(raw.positions, positions)
    descriptor = NucleotideGDescriptor(
        normalized.normalized.prepare_binding(),
        length_unit=METER,
        image_policy="nonperiodic",
    )
    expected = NucleotideGDescriptor(
        raw.prepare_binding(), length_unit=ANGSTROM, image_policy="nonperiodic"
    )
    np.testing.assert_allclose(
        descriptor.evaluate(normalized.normalized.positions).values,
        expected.evaluate(raw.positions).values,
        atol=1e-12,
    )
    with pytest.raises(PermissionError):
        normalize_nucleic_hypothesis(
            raw, length_unit=ANGSTROM, requested_use={"training_use": True}
        )


def test_native_torsions_keep_termini_and_pucker_phase_degeneracy():
    construct = NucleicAcidConstruct(("r",), ("A",), ("RNA",), (False,))
    names = ("P", "O5'", "C5'", "C4'", "C3'", "O3'", "O4'", "C1'", "C2'", "N9", "C4")
    ids = tuple(10 + 7 * i for i in range(len(names)))
    mapping = NucleotideAtomMapping(
        construct, ids, (construct.nucleotide_keys[0],) * len(names), names
    )
    units = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = AtomisticSystemPlan(
        ids, [6] * len(ids), [12.0] * len(ids), units, atom_type_ids=[0] * len(ids)
    ).prepare()
    t = jnp.arange(len(ids), dtype=float)
    positions = jnp.stack((jnp.cos(t), jnp.sin(t), 0.2 * t), axis=-1)
    program = NucleotideTorsionProgram(mapping, system)
    result = program.evaluate(positions)
    assert (
        not bool(result.valid[0, 0])
        and not bool(result.valid[0, 4])
        and not bool(result.valid[0, 5])
    )
    assert bool(result.valid[0, 6]) and bool(jnp.all(result.valid[0, 7:]))
    support = SeriesSupport(
        jnp.array([0.0, 1.0]), coordinate_name="time", coordinate_id="torsion-series"
    )
    coordinates = jnp.stack((positions, positions))
    coverage = jnp.ones(coordinates.shape, bool).at[1, 6, :].set(False)
    series = SampledSeries(
        support, coordinates, value_valid=coverage, series_id="partial-sugar"
    )
    observed = program.observe_series(series)
    assert bool(observed.value_valid[0, 0, 6]) and not bool(observed.value_valid[1, 0, 6])
    assert not bool(jnp.any(program.observe_pseudorotation_series(series).value_valid[1]))
    phase, amplitude = 0.73, 0.42
    nu = amplitude * jnp.cos(phase + 4 * jnp.pi * (jnp.arange(5) - 2) / 5)
    values = jnp.zeros((1, 12)).at[0, 7:].set(nu)
    pucker = sugar_pseudorotation(
        NucleotideTorsionEvaluation(values, jnp.ones((1, 12), bool), jnp.ones((1, 12)))
    )
    np.testing.assert_allclose(pucker.phase, [phase], atol=1e-12)
    np.testing.assert_allclose(pucker.amplitude, [amplitude], atol=1e-12)
    flat = sugar_pseudorotation(
        NucleotideTorsionEvaluation(
            jnp.zeros((1, 12)), jnp.ones((1, 12), bool), jnp.ones((1, 12))
        )
    )
    assert not bool(flat.valid[0])


def test_named_contacts_are_geometry_not_inferred_canonical_pairs():
    mapping, positions = fixture()
    binding = prepare_nucleotide_binding(mapping, mapping.atom_ids)
    descriptor = NucleotideGDescriptor(
        binding, length_unit=ANGSTROM, image_policy="nonperiodic"
    )
    criteria = GeometricContactCriteria(
        "declared-test-geometry", 6.0, 0.9, 0.8, (2.0, 4.0), 2.0
    )
    coplanar = geometric_contacts(positions, descriptor, criteria)
    assert bool(coplanar.coplanar[0]) and not bool(coplanar.stacked[0])
    stacked_positions = positions.at[3:6].set(positions[:3] + jnp.array([0.0, 0.0, 3.0]))
    stacked = geometric_contacts(stacked_positions, descriptor, criteria)
    assert bool(stacked.stacked[0]) and not bool(stacked.coplanar[0])


def test_native_source_records_to_descriptor_retains_author_identity():
    from phydrax.applications.nucleic_acid_biophysics import (
        nucleic_hypothesis_from_pdb_records,
    )
    from phydrax.atomistic.interchange._structure_records import (
        read_pdb_atom_records,
        select_pdb_model,
    )

    mapping, positions = fixture()
    lines = []
    for serial, (key, name, point) in enumerate(
        zip(
            mapping.nucleotide_keys,
            mapping.atom_names,
            np.asarray(positions),
            strict=True,
        ),
        1,
    ):
        base = mapping.construct.bases[key.position]
        x, y, z = point
        lines.append(
            f"ATOM  {serial:5d} {name:>4s} {base:>3s} X{100 + key.position:4d}A   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{12.0:6.2f}           C"
        )
    raw = read_pdb_atom_records("\n".join(lines), source_id="source-pdb-records")
    selected = select_pdb_model(raw, "1", alternate_locations={})
    assignments = {
        row.record_id: (key, atom)
        for row, key, atom in zip(
            selected, mapping.nucleotide_keys, mapping.atom_ids, strict=True
        )
    }
    manifest = rights()
    envelope = ScientificArtifactEnvelope(
        artifact_kind="source-PDB",
        content_digest=hashlib.sha256("\n".join(lines).encode()).hexdigest(),
        producer="independent-fixture",
        producer_version="native",
        build_id="fixture",
        license_id=manifest.license_id,
        resource_id=manifest.manifest_id,
        status="complete",
    )
    imported = nucleic_hypothesis_from_pdb_records(
        raw,
        tuple(reversed(selected)),
        construct=mapping.construct,
        record_assignments=assignments,
        source=envelope,
        rights=manifest,
        requested_use={},
        image_policy="nonperiodic",
    )
    descriptor = NucleotideGDescriptor(
        imported.hypothesis.prepare_binding(),
        length_unit=ANGSTROM,
        image_policy="nonperiodic",
    )
    direct = NucleotideGDescriptor(
        prepare_nucleotide_binding(mapping, mapping.atom_ids),
        length_unit=ANGSTROM,
        image_policy="nonperiodic",
    )
    np.testing.assert_allclose(
        descriptor.evaluate(imported.hypothesis.positions).values,
        direct.evaluate(positions).values,
        atol=1e-12,
    )
    assert imported.source_records[0].author_residue_number == "100"
    assert imported.source_records[0].insertion_code == "A"


def test_complete_frame_is_not_complete_chemical_geometry():
    from phydrax.applications.nucleic_acid_biophysics.structure import (
        NucleotideStructureQualifier,
    )

    mapping, positions = fixture()
    binding = prepare_nucleotide_binding(mapping, mapping.atom_ids)
    qualifier = NucleotideStructureQualifier(
        binding,
        maximum_ring_deviation=0.05,
        backbone_interval=(1.3, 1.9),
        image_policy="nonperiodic",
    )
    evidence = qualifier.evaluate(positions)
    assert bool(jnp.all(evidence.frame_valid))
    assert not bool(jnp.any(evidence.ring_covered))
    assert not bool(jnp.any(evidence.backbone_covered))
    assert not bool(evidence.successful)
