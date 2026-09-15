#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen-predictor compilation and audited mixed-integer proposals."""

from ._binding import bind_predictor_inputs
from ._linear import compile_linear_predictor_constraint
from ._proposal import (
    AdjacentLatticeMaterializer,
    materialize_mixed_integer_proposal,
    MixedIntegerProposalManifest,
    MixedIntegerProposalResult,
    ParametricMixedIntegerProposal,
)
from ._support import (
    augment_with_support,
    compile_convex_hull_support,
    PredictorSupportCompilation,
)
from ._tree import compile_tree_predictor_constraint
from ._types import (
    augment_linear_program,
    PredictorCompilationGuarantee,
    PredictorConstraintCompilation,
    PredictorConstraintSense,
    PredictorInputBinding,
    PredictorOutputConstraint,
    PredictorOutputSemantic,
)


__all__ = [
    "AdjacentLatticeMaterializer",
    "MixedIntegerProposalManifest",
    "MixedIntegerProposalResult",
    "ParametricMixedIntegerProposal",
    "PredictorCompilationGuarantee",
    "PredictorConstraintCompilation",
    "PredictorConstraintSense",
    "PredictorInputBinding",
    "PredictorOutputConstraint",
    "PredictorOutputSemantic",
    "PredictorSupportCompilation",
    "augment_linear_program",
    "augment_with_support",
    "bind_predictor_inputs",
    "compile_convex_hull_support",
    "compile_linear_predictor_constraint",
    "compile_tree_predictor_constraint",
    "materialize_mixed_integer_proposal",
]
