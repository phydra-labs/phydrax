import hashlib

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nucleic_acid_biophysics._binding import NucleotideAtomMapping
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.generation import (
    CoordinateGeometryPolicy,
    CoordinateProviderProvenance,
    import_nucleic_hypotheses,
    map_nucleic_hypothesis,
    prepare_nucleic_coordinate_support,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.atomistic import AtomisticBatch, AtomisticScaleContract
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM, ELECTRONVOLT, UnitDefinition


NANOMETER = UnitDefinition(
    "nm",
    ANGSTROM.dimension,
    ANGSTROM.reference_system_id,
    10 * ANGSTROM.scale_to_reference,
)


def _rights(*, training=True):
    payload = b"original-nucleotide-token-test-fixture"
    return ReferenceArtifactManifest(
        "original-nucleotide-fixture",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Phydrax-OriginalSyntheticFixture",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=training,
        export_permitted=True,
        export_classification="unrestricted-fixture",
        nondimensionalization={"coordinate_angstrom": 1.0},
        uncertainty=None,
        lineage_ids=("original-analytic-token-fixture",),
    )


def _source(rights):
    return ScientificArtifactEnvelope(
        artifact_kind="raw-nucleic-output",
        content_digest=rights.checksum,
        producer="caller",
        producer_version="native",
        build_id="test",
        license_id=rights.license_id,
        resource_id="original-analytic-fixture",
        status="complete",
    )


def _fixture(polymer="DNA"):
    construct = NucleicAcidConstruct(("strand",), ("A",), (polymer,), (False,))
    mapping = NucleotideAtomMapping(
        construct,
        (20, 501, 101, 67),
        (construct.nucleotide_keys[0],) * 4,
        ("C2", "C4", "C6", "O4'"),
    )
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    template = AtomisticBatch(
        np.array([[6, 6, 6, 8]]),
        x[None],
        np.array([[12.0, 12.0, 12.0, 16.0]]),
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        particle_ids=np.array([mapping.atom_ids]),
    )
    geometry = CoordinateGeometryPolicy(
        ((20, 501),),
        ((0.5, 1.5),),
        ((20, 501, 101, 67),),
        (1,),
        0.01,
        "analytic-nucleotide-token-screen",
    )
    support = prepare_nucleic_coordinate_support(
        mapping, template, gauge_atom_ids=(20, 501, 101), geometry=geometry
    )
    return mapping, support, x


def test_nucleic_provider_preserves_unit_equivalence_and_stable_atom_mapping():
    mapping, support, x = _fixture()
    rights = _rights()
    order = (3, 1, 0, 2)
    permuted = NucleotideAtomMapping(
        mapping.construct,
        tuple(mapping.atom_ids[i] for i in order),
        tuple(mapping.nucleotide_keys[i] for i in order),
        tuple(mapping.atom_names[i] for i in order),
    )
    result = import_nucleic_hypotheses(
        permuted,
        (x[np.array(order)] / 10)[None],
        NANOMETER,
        (_source(rights),),
        provenance=CoordinateProviderProvenance("caller-supplied", (rights,)),
        confidence=((("caller-score", 0.7),),),
    )
    mapped = map_nucleic_hypothesis(result.hypotheses[0], support)
    assert jnp.allclose(mapped, x, atol=1e-14)
    assert jnp.allclose(result.hypotheses[0].positions, x[np.array(order)] / 10)
    _, rna_support, _ = _fixture("RNA")
    with pytest.raises(ValueError):
        map_nucleic_hypothesis(result.hypotheses[0], rna_support)


def test_nucleic_provider_refuses_incomplete_training_and_inherited_weight_restrictions():
    mapping, support, x = _fixture()
    rights, restricted = _rights(), _rights(training=False)
    provenance = CoordinateProviderProvenance(
        "offline-learned-output",
        (rights,),
        weight_rights=(restricted,),
        input_rights=(rights,),
        input_artifact_ids=("prepared-input",),
        learned_model=True,
    )
    result = import_nucleic_hypotheses(
        mapping, x[None], ANGSTROM, (_source(rights),), provenance=provenance
    )
    with pytest.raises(PermissionError):
        map_nucleic_hypothesis(result.hypotheses[0], support, training_use=True)
    partial = import_nucleic_hypotheses(
        mapping,
        x[None],
        ANGSTROM,
        (_source(rights),),
        coordinate_mask=[True, True, False, True],
        provenance=CoordinateProviderProvenance("incomplete-caller-output", (rights,)),
    )
    with pytest.raises(ValueError):
        map_nucleic_hypothesis(partial.hypotheses[0], support)
    wrong = NucleotideAtomMapping(
        mapping.construct,
        mapping.atom_ids,
        mapping.nucleotide_keys,
        ("C4", "C2", "C6", "O4'"),
    )
    swapped = import_nucleic_hypotheses(
        wrong,
        x[None],
        ANGSTROM,
        (_source(rights),),
        provenance=CoordinateProviderProvenance("caller-output", (rights,)),
    )
    with pytest.raises(ValueError):
        map_nucleic_hypothesis(swapped.hypotheses[0], support)
