#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

from ._features import OperatorInferenceFeatureLibrary
from ._sindy import fit_sindy, SINDyResult
from ._sindy_design import SINDyProblem
from ._sparse_regression import DenseBlockRidgeRegression


def operator_inference_block_sizes(
    library: OperatorInferenceFeatureLibrary,
    /,
) -> tuple[int, ...]:
    """Return constant, state, optional input, and state-quadratic widths."""
    if not isinstance(library, OperatorInferenceFeatureLibrary):
        raise TypeError("library must be an OperatorInferenceFeatureLibrary.")
    sizes = [1, library.state_layout.size]
    if library.input_layout is not None:
        sizes.append(library.input_layout.size)
    sizes.append(len(library.quadratic_indices))
    return tuple(sizes)


def fit_operator_inference(
    problem: SINDyProblem,
    regularization: Sequence[float],
    /,
    *,
    scale_features: bool = True,
    scale_targets: bool = False,
    rcond: float | None = None,
) -> SINDyResult:
    """Fit the restricted dense c/A/B/H operator-inference dictionary."""
    if not isinstance(problem, SINDyProblem):
        raise TypeError("problem must be a SINDyProblem.")
    if not isinstance(problem.library, OperatorInferenceFeatureLibrary):
        raise TypeError("Operator inference requires OperatorInferenceFeatureLibrary.")
    regressor = DenseBlockRidgeRegression(
        operator_inference_block_sizes(problem.library),
        regularization,
        scale_features=scale_features,
        scale_targets=scale_targets,
        rcond=rcond,
    )
    return fit_sindy(problem, regressor)


__all__ = [
    "fit_operator_inference",
    "operator_inference_block_sizes",
]
