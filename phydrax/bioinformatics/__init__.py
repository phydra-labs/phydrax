#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bioinformatics domains and their shared scientific contracts."""

from importlib import import_module as _import_module

from . import foundation
from .foundation import (
    BioinformaticsMethodContract,
    BiologicalGrouping,
    BiologicalSplit,
    BiospecimenLineage,
    DifferentiationKind,
    ExchangeabilityPlan,
    ExecutionKind,
    ExperimentalUnitPlan,
    FeatureDictionary,
    FeatureMapping,
    LeakageAudit,
    MethodKind,
    OntologyGraph,
    OutputKind,
)


_DOMAIN_NAMES = frozenset(
    {
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
)


def __getattr__(name: str, /):
    if name not in _DOMAIN_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | _DOMAIN_NAMES)


__all__ = [
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
]
