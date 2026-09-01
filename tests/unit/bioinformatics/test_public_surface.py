#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys

import phydrax
import phydrax.bioinformatics as bioinformatics
import phydrax.interchange as generic_interchange
import phydrax.velocimetry.io as velocimetry_io
from phydrax._integration_guardrails import CANONICAL_CORE_OWNERS


_DOMAINS = {
    "foundation",
    "genomics",
    "interchange",
    "metagenomics",
    "models",
    "omics",
    "phylogenetics",
    "population",
    "rna",
    "sequence",
    "spatial",
    "spectrometry",
    "structure",
    "systems",
}

_FOUNDATION_EXPORTS = {
    "BioinformaticsMethodContract",
    "BiologicalGrouping",
    "BiologicalSplit",
    "BiospecimenLineage",
    "DifferentiationKind",
    "ExchangeabilityPlan",
    "ExecutionKind",
    "ExperimentalUnitPlan",
    "FeatureDictionary",
    "FeatureMapping",
    "LeakageAudit",
    "MethodKind",
    "OntologyGraph",
    "OutputKind",
}

_IMPLICIT_MODULE_EXPORTS = {
    "phydrax.bioinformatics.foundation._features": {
        "FeatureDictionary",
        "FeatureMapping",
        "OntologyGraph",
    },
    "phydrax.bioinformatics.foundation._method": {
        "BioinformaticsMethodContract",
        "DifferentiationKind",
        "ExecutionKind",
        "MethodKind",
        "OutputKind",
    },
    "phydrax.bioinformatics.foundation._splits": {
        "BiologicalGrouping",
        "BiologicalSplit",
        "LeakageAudit",
    },
    "phydrax.bioinformatics.foundation._study": {
        "BiospecimenLineage",
        "ExchangeabilityPlan",
        "ExperimentalUnitPlan",
    },
    "phydrax.bioinformatics.foundation._validation": set(),
    "phydrax.bioinformatics.genomics._alignment_events": {
        "ALIGNMENT_EVENT_STATUS_INVALID_REFERENCE_START",
        "ALIGNMENT_EVENT_STATUS_QUERY_LENGTH_MISMATCH",
        "ALIGNMENT_EVENT_STATUS_VALID",
        "AlignmentEventBatch",
        "expand_alignment_events",
    },
    "phydrax.bioinformatics.genomics._cigar": {
        "CIGAR_OPERATION_CHARS",
        "CIGAR_STATUS_DIRTY_PADDING",
        "CIGAR_STATUS_EVENT_CAPACITY_EXCEEDED",
        "CIGAR_STATUS_INVALID_COUNT",
        "CIGAR_STATUS_INVALID_LENGTH",
        "CIGAR_STATUS_INVALID_OPERATION",
        "CIGAR_STATUS_OPERATION_CAPACITY_EXCEEDED",
        "CIGAR_STATUS_VALID",
        "CigarBatch",
        "CigarOp",
        "cigar_batch_from_strings",
        "cigar_batch_from_tuples",
        "cigar_consumption_for_operation",
        "pack_cigar",
    },
    "phydrax.bioinformatics.genomics._mapping": {
        "MAPPING_STATUS_CANDIDATE_TRUNCATED",
        "MAPPING_STATUS_INVALID_INPUT",
        "MAPPING_STATUS_MAPQ_UNCALIBRATED",
        "MAPPING_STATUS_NO_CANDIDATES",
        "MAPPING_STATUS_VALID",
        "PILEUP_STATUS_CIGAR_REFERENCE_MISMATCH",
        "PILEUP_STATUS_INVALID_EVENTS",
        "PILEUP_STATUS_REFERENCE_BOUNDS",
        "PILEUP_STATUS_REFERENCE_MISSING",
        "PILEUP_STATUS_VALID",
        "MappingCandidateBatch",
        "MappingEvidenceResult",
        "MappingExecutionPlan",
        "PileupLikelihoodResult",
        "candidate_mapping_evidence",
        "reference_aware_pileup_likelihood",
    },
    "phydrax.bioinformatics.genomics._reads": {
        "READ_STATUS_IDENTITY_MISMATCH",
        "READ_STATUS_INVALID_CIGAR",
        "READ_STATUS_INVALID_MAPPED_COORDINATE",
        "READ_STATUS_INVALID_MAPQ",
        "READ_STATUS_INVALID_PAIR",
        "READ_STATUS_INVALID_QUALITY",
        "READ_STATUS_INVALID_UMI",
        "READ_STATUS_INVALID_UNMAPPED_STATE",
        "READ_STATUS_SEQUENCE_CIGAR_LENGTH_MISMATCH",
        "READ_STATUS_VALID",
        "SAM_FLAG_DUPLICATE",
        "SAM_FLAG_FIRST_IN_PAIR",
        "SAM_FLAG_MATE_REVERSE",
        "SAM_FLAG_MATE_UNMAPPED",
        "SAM_FLAG_PAIRED",
        "SAM_FLAG_PROPER_PAIR",
        "SAM_FLAG_QC_FAIL",
        "SAM_FLAG_REVERSE",
        "SAM_FLAG_SECONDARY",
        "SAM_FLAG_SECOND_IN_PAIR",
        "SAM_FLAG_SUPPLEMENTARY",
        "SAM_FLAG_UNMAPPED",
        "ReadBatch",
        "ReadEvidenceProvenance",
        "ReadLayout",
        "ReadPairLayout",
        "read_evidence_provenance",
    },
}

_COLLISION_OWNERS = {
    ("sequence", "DELETE"): "_alignment",
    ("sequence", "INSERT"): "_alignment",
    ("sequence", "MATCH"): "_alignment",
}

_BIOINFORMATICS_CORE_OWNERS = {
    "biochemical_network": "phydrax.bioinformatics.systems.StoichiometricNetwork",
    "biological_assay": "phydrax.bioinformatics.omics.CountAssay",
    "biological_feature_dictionary": (
        "phydrax.bioinformatics.foundation.FeatureDictionary"
    ),
    "biological_sequence": "phydrax.bioinformatics.sequence.SequenceBatch",
    "biospecimen_lineage": "phydrax.bioinformatics.foundation.BiospecimenLineage",
    "genomic_coordinate": "phydrax.bioinformatics.genomics.IntervalSet",
    "macromolecular_structure": (
        "phydrax.bioinformatics.structure.MacromolecularStructure"
    ),
    "mass_spectrum": "phydrax.bioinformatics.spectrometry.SpectrumBatch",
    "phylogenetic_tree": "phydrax.bioinformatics.phylogenetics.TreeTopology",
}


def _module_exports(module) -> set[str]:
    if hasattr(module, "__all__"):
        exports = list(module.__all__)
        assert len(exports) == len(set(exports))
        return set(exports)
    assert module.__name__ in _IMPLICIT_MODULE_EXPORTS
    return _IMPLICIT_MODULE_EXPORTS[module.__name__]


def test_root_exposes_namespaces_and_shared_foundation_contracts_only():
    assert "bioinformatics" in phydrax.__all__
    assert phydrax.bioinformatics is bioinformatics
    assert set(bioinformatics.__all__) == _DOMAINS | _FOUNDATION_EXPORTS

    foundation = bioinformatics.foundation
    assert set(foundation.__all__) == _FOUNDATION_EXPORTS
    for name in _FOUNDATION_EXPORTS:
        assert getattr(bioinformatics, name) is getattr(foundation, name)
        assert name not in phydrax.__all__

    for domain in _DOMAINS:
        facade = importlib.import_module(f"phydrax.bioinformatics.{domain}")
        assert getattr(bioinformatics, domain) is facade


def test_every_domain_facade_has_one_deliberate_owner_per_public_name():
    for domain in sorted(_DOMAINS):
        facade = importlib.import_module(f"phydrax.bioinformatics.{domain}")
        owners: dict[str, str] = {}
        for module_info in pkgutil.iter_modules(facade.__path__):
            if not module_info.name.startswith("_"):
                continue
            module = importlib.import_module(f"{facade.__name__}.{module_info.name}")
            for name in _module_exports(module):
                collision_owner = _COLLISION_OWNERS.get((domain, name))
                if collision_owner is not None and module_info.name != collision_owner:
                    continue
                assert name not in owners, (domain, name, owners[name], module_info.name)
                owners[name] = module_info.name
                assert getattr(facade, name) is getattr(module, name)

        assert len(facade.__all__) == len(set(facade.__all__))
        assert set(facade.__all__) == set(owners)


def test_bioinformatics_canonical_owner_paths_resolve():
    assert {
        kind: CANONICAL_CORE_OWNERS[kind] for kind in _BIOINFORMATICS_CORE_OWNERS
    } == _BIOINFORMATICS_CORE_OWNERS

    for owner in _BIOINFORMATICS_CORE_OWNERS.values():
        module_name, _, symbol_name = owner.rpartition(".")
        module = importlib.import_module(module_name)
        assert getattr(module, symbol_name) is not None


def test_generic_adapter_reports_are_not_reexported_by_domain_adapters():
    bioinformatics_interchange = bioinformatics.interchange
    canonical = {
        "AdapterError",
        "AdapterLoss",
        "AdapterReport",
        "AdapterStatus",
        "require_lossless",
    }
    removed = canonical | {"AdapterDirection", "AdapterLossCategory"}

    assert canonical <= set(generic_interchange.__all__)
    assert generic_interchange.AdapterReport.__module__ == "phydrax.interchange._report"
    assert removed.isdisjoint(vars(bioinformatics_interchange))
    assert removed.isdisjoint(vars(velocimetry_io))


def test_root_import_is_lazy_and_does_not_import_optional_bioinformatics_packages():
    script = """
import sys
import phydrax
import phydrax.bioinformatics

domains = {
    "genomics", "interchange", "metagenomics", "models", "omics",
    "phylogenetics", "population", "rna", "sequence", "spatial",
    "spectrometry", "structure", "systems",
}
assert all(f"phydrax.bioinformatics.{name}" not in sys.modules for name in domains)
assert "pysam" not in sys.modules
assert "pyteomics" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
