import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax import atomistic
from phydrax.applications.protein_folding import (
    bind_protein,
    PreparedProteinQualification,
    ProteinAtomKey,
    ProteinConstruct,
    ProteinSourceAtom,
    ProteinStructureHypothesis,
    ResolvedProteinChemistry,
)
from phydrax.applications.protein_folding.generation import (
    fit_coordinate_model,
    prepare_bound_protein_coordinate_generation,
    prepare_coordinate_training_data,
    sample_protein_coordinate_proposals,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM


def _cis_proline_fixture(*, unsupported_ring=False, glycine=False, aromatic=False):
    if glycine and aromatic:
        raise ValueError("Fixture residue choice must be unique.")
    letter = "G" if glycine else "F" if aromatic else "P"
    construct = ProteinConstruct(("A",), ("A" + letter,))
    first, second = construct.residue_keys
    first_names = ("N", "CA", "C", "O", "CB", "H")
    if glycine:
        second_names = ("N", "CA", "C", "O", "OXT", "H")
    elif aromatic:
        second_names = (
            "N",
            "CA",
            "C",
            "O",
            "OXT",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "H",
        )
    else:
        second_names = ("N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "H")
    keys = tuple(ProteinAtomKey(first, name) for name in first_names) + tuple(
        ProteinAtomKey(second, name) for name in second_names
    )
    first_coordinates = (
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.6, 1.3, 0.0),
        (0.1, 2.4, 0.0),
        (0.2, -0.7, -1.0),
        (-1.7, 0.2, 0.4),
    )
    if glycine:
        side_chain_coordinates = ()
    elif aromatic:
        side_chain_coordinates = (
            (3.5, 2.5, -0.7),
            (4.4, 3.3, -0.7),
            (5.6, 2.9, -0.7),
            (4.1, 4.6, -0.7),
            (6.5, 3.8, -0.7),
            (5.0, 5.5, -0.7),
            (6.2, 5.1, -0.7),
        )
    else:
        side_chain_coordinates = (
            (3.5, 2.5, -0.7),
            (3.7, 1.3, -1.0),
            (2.7, 0.7, -0.5),
        )
    second_coordinates = (
        (1.7, 1.3, 0.0),
        (2.3, 2.6, 0.0),
        (1.5, 3.7, 0.0),
        (0.3, 3.6, 0.0),
        (2.1, 4.8, 0.0),
        *side_chain_coordinates,
        (1.5, 0.5, 0.5),
    )
    coordinates = np.asarray((*first_coordinates, *second_coordinates), dtype=float)
    numbers = tuple(
        1 if key.atom_name == "H" else {"C": 6, "N": 7, "O": 8}[key.atom_name[0]]
        for key in keys
    )
    key_index = {key: index for index, key in enumerate(keys)}

    def atom(residue, name):
        return key_index[ProteinAtomKey(residue, name)]

    routes = [
        (atom(first, "N"), atom(first, "CA")),
        (atom(first, "CA"), atom(first, "C")),
        (atom(first, "C"), atom(first, "O")),
        (atom(first, "CA"), atom(first, "CB")),
        (atom(first, "N"), atom(first, "H")),
        (atom(first, "C"), atom(second, "N")),
        (atom(second, "N"), atom(second, "CA")),
        (atom(second, "CA"), atom(second, "C")),
        (atom(second, "C"), atom(second, "O")),
        (atom(second, "C"), atom(second, "OXT")),
        (atom(second, "N"), atom(second, "H")),
    ]
    if aromatic:
        routes.extend(
            (
                (atom(second, "CA"), atom(second, "CB")),
                (atom(second, "CB"), atom(second, "CG")),
                (atom(second, "CG"), atom(second, "CD1")),
                (atom(second, "CG"), atom(second, "CD2")),
                (atom(second, "CD1"), atom(second, "CE1")),
                (atom(second, "CD2"), atom(second, "CE2")),
                (atom(second, "CE1"), atom(second, "CZ")),
                (atom(second, "CE2"), atom(second, "CZ")),
            )
        )
    elif not glycine:
        routes.extend(
            (
                (atom(second, "CA"), atom(second, "CB")),
                (atom(second, "CB"), atom(second, "CG")),
                (atom(second, "CG"), atom(second, "CD")),
                (atom(second, "CD"), atom(second, "N")),
            )
        )
    if unsupported_ring:
        routes.append((atom(first, "N"), atom(first, "CB")))
    routes = np.asarray(routes, dtype=np.int32)
    payload = coordinates.tobytes() + routes.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    rights = ReferenceArtifactManifest(
        "original-cis-proline-internal-coordinate-fixture",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id="LicenseRef-Phydrax-OriginalSyntheticFixture",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted-original-synthetic-data",
        nondimensionalization={"coordinate_angstrom": 1.0},
        uncertainty=None,
        lineage_ids=("hand-authored-topology-fixture",),
    )
    source = ScientificArtifactEnvelope(
        artifact_kind="synthetic-protein-topology",
        content_digest=digest,
        producer="phydrax-test",
        producer_version="native",
        build_id="cis-proline",
        license_id=rights.license_id,
        resource_id="cis-proline",
        status="complete",
    )
    chemistry = ResolvedProteinChemistry(
        construct,
        keys,
        numbers,
        ("standard", "standard"),
        (1, 1),
        "NH2",
        "COO-",
        source.artifact_id,
    )
    rows = tuple(
        ProteinSourceAtom(
            str(index), key, "1", "A", str(key.residue.position + 1), "", "", 1.0, number
        )
        for index, (key, number) in enumerate(zip(keys, numbers, strict=True))
    )
    hypothesis = ProteinStructureHypothesis(
        construct, rows, coordinates, ANGSTROM, source, (rights,)
    )
    ids = np.arange(len(keys), dtype=np.int64) * 17 + 101
    topology = atomistic.MolecularTopologyPlan(
        bonds=ids[routes], bond_type_ids=np.arange(len(routes))
    )
    units = atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    masses = np.asarray(
        [1.0 if z == 1 else 14.0 if z == 7 else 16.0 if z == 8 else 12.0 for z in numbers]
    )
    system = atomistic.AtomisticSystemPlan(
        ids,
        numbers,
        masses,
        units,
        topology=topology,
        molecule_ids=np.zeros(len(keys), dtype=int),
    )
    lengths = np.sqrt(
        np.sum((coordinates[routes[:, 0]] - coordinates[routes[:, 1]]) ** 2, axis=-1)
    )
    potential = atomistic.AtomisticPotentialProgram(
        [atomistic.HarmonicBondPotential(np.ones(len(routes)), lengths)]
    )
    force_field = atomistic.AtomisticForceFieldPlan(
        system,
        potential,
        atomistic.AtomisticNonbondedPolicy(8.0, electrostatics="direct"),
        atomistic.AtomisticForceFieldProvenance(
            "synthetic", (digest,), "analytic-harmonic", "explicit-unit-fixture"
        ),
    ).prepare()
    binding = bind_protein(
        hypothesis,
        chemistry,
        force_field,
        dict(zip(keys, map(int, ids), strict=True)),
        parameter_energy_unit=units.scale.energy_unit,
        parameter_rights=(rights,),
    )
    qualifier = PreparedProteinQualification(
        binding,
        bond_bounds=np.stack((0.7 * lengths, 1.3 * lengths), axis=1),
        clash_distance=0.1,
        minimum_chiral_volume=0.01,
    )
    return binding, qualifier, rights


def test_periodic_decoder_preserves_chirality_rigid_proline_and_declared_cis():
    binding, qualification, _ = _cis_proline_fixture()
    with pytest.raises(ValueError, match="Cis peptide geometry"):
        prepare_bound_protein_coordinate_generation(binding, qualification)
    support, decoder = prepare_bound_protein_coordinate_generation(
        binding, qualification, cis_peptide_indices=(0,)
    )
    assert decoder.plan.peptide_modes == ("cis",)
    assert tuple(group.kind for group in decoder.plan.closure_groups) == ("proline-ring",)
    reference = decoder.reference_periodic_coordinates()
    pairs = reference.reshape((-1, 2))
    angles = jnp.arctan2(pairs[:, 0], pairs[:, 1]) + 0.37
    shifted = jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=1).reshape((-1,))
    decoded = eqx.filter_jit(decoder.decode)(shifted)
    full = qualification.evaluate(decoded.positions)
    assert decoded.valid
    assert jnp.all(decoded.closure.valid)
    assert jnp.allclose(decoded.bond_residuals, 0.0, atol=2e-6)
    assert jnp.allclose(decoded.angle_residuals, 0.0, atol=2e-6)
    assert jnp.all(full.chirality_valid)
    replay = decoder.decode(
        jnp.stack(
            (jnp.sin(angles + 2 * jnp.pi), jnp.cos(angles + 2 * jnp.pi)), axis=1
        ).reshape((-1,))
    )
    assert jnp.allclose(decoded.positions, replay.positions, atol=2e-6)
    gradient = jax.grad(lambda value: jnp.sum(decoder.decode(value).positions ** 2))(
        shifted
    )
    assert jnp.all(jnp.isfinite(gradient))
    assert support.support_id == decoder.support_id


def test_invalid_periodic_case_and_unsupported_standard_residue_ring_are_retained_or_refused():
    binding, qualification, _ = _cis_proline_fixture()
    _, decoder = prepare_bound_protein_coordinate_generation(
        binding, qualification, cis_peptide_indices=(0,)
    )
    invalid = decoder.reference_periodic_coordinates().at[:2].set(0.0)
    decoded = decoder.decode(invalid)
    assert not decoded.valid
    assert not decoded.periodic_valid[0]
    assert jnp.all(jnp.isfinite(decoded.positions))
    ring_binding, ring_qualification, _ = _cis_proline_fixture(unsupported_ring=True)
    with pytest.raises(ValueError, match="standard heavy-atom covalent graph"):
        prepare_bound_protein_coordinate_generation(
            ring_binding, ring_qualification, cis_peptide_indices=(0,)
        )


def test_glycine_is_achiral_without_suppressing_other_residue_evidence():
    binding, qualification, _ = _cis_proline_fixture(glycine=True)
    _, decoder = prepare_bound_protein_coordinate_generation(
        binding, qualification, cis_peptide_indices=(0,)
    )
    encoding, decoded = decoder.reconstruct(binding.realized_positions)
    full = qualification.evaluate(decoded.positions)
    assert encoding.valid
    assert decoded.valid
    assert decoder.plan.closure_groups == ()
    assert qualification.chirality_indices.shape == (1, 4)
    assert full.chiral_volumes.shape == (1,)
    assert jnp.all(full.chirality_valid)


def test_standard_aromatic_ring_is_a_rigid_evaluated_closure_group():
    binding, qualification, _ = _cis_proline_fixture(aromatic=True)
    _, decoder = prepare_bound_protein_coordinate_generation(
        binding, qualification, cis_peptide_indices=(0,)
    )
    decoded = decoder.decode(decoder.reference_periodic_coordinates())
    assert tuple(group.kind for group in decoder.plan.closure_groups) == (
        "aromatic-ring",
    )
    assert decoded.valid
    assert jnp.all(decoded.closure.valid)
    assert qualification.evaluate(decoded.positions).successful


def test_internal_flow_retains_every_raw_decoded_and_failed_proposal():
    binding, qualification, rights = _cis_proline_fixture()
    support, decoder = prepare_bound_protein_coordinate_generation(
        binding, qualification, cis_peptide_indices=(0,)
    )
    reference = decoder.reference_periodic_coordinates().reshape((-1, 2))
    offsets = np.linspace(-0.3, 0.3, 6)
    periodic = jnp.stack(
        tuple(
            jnp.stack(
                (
                    jnp.sin(jnp.arctan2(reference[:, 0], reference[:, 1]) + offset),
                    jnp.cos(jnp.arctan2(reference[:, 0], reference[:, 1]) + offset),
                ),
                axis=1,
            ).reshape((-1,))
            for offset in offsets
        )
    )
    coordinates = decoder.decode(periodic).positions
    groups = ("validation", "training", "training", "training", "training", "validation")
    data = prepare_coordinate_training_data(
        support,
        coordinates,
        np.asarray(offsets)[:, None],
        condition_names=("declared_torsion_offset_radian",),
        record_ids=tuple(f"synthetic-{index}" for index in range(6)),
        source_manifest_ids=(rights.manifest_id,) * 6,
        split_group_ids=groups,
        validation_groups=("validation",),
        rights=(rights,),
        corpus_description=(
            "Original fixed-construct kinematic perturbations for decoder integration; "
            "not independent proteins, equilibrium samples, or sequence generalization."
        ),
        decoder=decoder,
    )
    fit = fit_coordinate_model(
        data,
        key=jr.key(41),
        steps=2,
        pairs_per_step=4,
        width=8,
        depth=1,
    )
    batch = sample_protein_coordinate_proposals(
        fit, jr.key(42), jnp.asarray([[-0.2], [0.2]]), qualification
    )
    assert batch.internal_coordinates.shape == (2, decoder.coordinate_size)
    assert batch.raw_decoded_coordinates.shape == (2, support.template.atom_capacity, 3)
    assert batch.decoded_coordinates.shape == (2, support.template.atom_capacity, 3)
    assert batch.failures.successful.shape == (2,)
    assert len(batch.sample_ids) == 2
    assert batch.physical_energy_evidence is None
    assert "non-equilibrium" in batch.scientific_scope
    assert jnp.array_equal(
        batch.decoder_evidence.periodic_coordinates, batch.internal_coordinates
    )
    assert batch.protein_geometry.successful.shape == (2,)
    assert all(value.shape == (2,) for value in jax.tree.leaves(batch.failures))
    assert "not sequence-general" in batch.scientific_scope
