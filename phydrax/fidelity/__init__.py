#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Model-agnostic contracts for coupled multi-fidelity SciML workflows."""

from ._archive import read_fidelity_dataset, write_fidelity_dataset
from ._dataset import (
    FidelityCaseSpec,
    FidelityDataset,
    FidelityDatasetSplit,
    FidelitySplitRequirements,
    split_fidelity_dataset,
)
from ._execution import FidelityEvaluation, FidelityEvaluator
from ._hierarchy import (
    FidelityCoupling,
    FidelityHierarchy,
    FidelityLevelSpec,
    FidelityPath,
    FidelityRelation,
)


__all__ = [
    "FidelityCaseSpec",
    "FidelityCoupling",
    "FidelityDataset",
    "FidelityDatasetSplit",
    "FidelitySplitRequirements",
    "FidelityEvaluation",
    "FidelityEvaluator",
    "FidelityHierarchy",
    "FidelityLevelSpec",
    "FidelityPath",
    "FidelityRelation",
    "read_fidelity_dataset",
    "split_fidelity_dataset",
    "write_fidelity_dataset",
]
