# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Admitted external hypotheses and uncalibrated native conditional proposals."""

from ..._coordinate_generation._native import (
    ConditionalCoordinateVelocity,
    CoordinateFitResult,
    CoordinateProposalBatch,
    CoordinateTrainingData,
    fit_coordinate_model,
    load_coordinate_model,
    prepare_coordinate_sampler,
    prepare_coordinate_training_data,
    PreparedCoordinateSampler,
    sample_coordinate_proposals,
    save_coordinate_model,
)
from ..._coordinate_generation._providers import CoordinateProviderProvenance
from ..._coordinate_generation._support import (
    CoordinateGeometryPolicy,
    CoordinateProposalQualification,
    CoordinateResourcePolicy,
    PreparedCoordinateSupport,
    qualify_coordinate_proposals,
)
from ._providers import (
    import_protein_hypotheses,
    map_protein_hypothesis,
    prepare_protein_coordinate_support,
    ProteinProviderHypotheses,
)


__all__ = [
    "ConditionalCoordinateVelocity",
    "CoordinateFitResult",
    "CoordinateGeometryPolicy",
    "CoordinateProposalBatch",
    "CoordinateProposalQualification",
    "CoordinateProviderProvenance",
    "CoordinateResourcePolicy",
    "CoordinateTrainingData",
    "PreparedCoordinateSampler",
    "PreparedCoordinateSupport",
    "ProteinProviderHypotheses",
    "fit_coordinate_model",
    "import_protein_hypotheses",
    "load_coordinate_model",
    "map_protein_hypothesis",
    "prepare_coordinate_sampler",
    "prepare_coordinate_training_data",
    "prepare_protein_coordinate_support",
    "qualify_coordinate_proposals",
    "sample_coordinate_proposals",
    "save_coordinate_model",
]
