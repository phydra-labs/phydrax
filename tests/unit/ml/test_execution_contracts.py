#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax import (
    ComponentAuthority,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
)
from phydrax.kernels import Matern32Kernel, SquaredExponentialKernel
from phydrax.ml import FeatureSchema, MLBatch, TargetSchema
from phydrax.ml.kernel_methods import KernelRidgeRecipe
from phydrax.ml.neighbors import (
    KernelNeighborsRegressorRecipe,
    KNeighborsRegressorRecipe,
)
from phydrax.ml.tree import DecisionTreeRegressor


INPUT = DerivativeSurface.INPUT
PARAMETER = DerivativeSurface.MODEL_PARAMETER
_PIECEWISE_CONSTANT = DerivativeRegularity(
    continuity=-1, pieces="polynomial", degree_bound=0
)


def _regression_batch():
    x0 = jnp.linspace(-3.0, 3.0, 10)
    return MLBatch(
        jnp.stack((x0, jnp.square(x0) - 2.0), axis=-1),
        1.5 * x0 - 0.25,
        feature_schema=FeatureSchema(("position", "curvature")),
        target_schema=TargetSchema("continuous", names=("response",)),
    )


def test_hard_tree_is_discontinuous_piecewise_constant():
    result = DecisionTreeRegressor(max_depth=2).fit_batch(_regression_batch())
    contract = result.model.model_execution_contract()
    assert contract.regularity == DerivativeRegularity(
        continuity=-1, pieces="polynomial", degree_bound=0
    )
    assert contract.regularity == result.derivative_contract.regularity
    assert contract.derivative.route is DerivativeRoute.DIRECT
    assert contract.derivative.level(INPUT) is GradientLevel.NONE
    assert contract.derivative.level(PARAMETER) is GradientLevel.ALMOST_EVERYWHERE
    assert "split structure" in contract.derivative.nondifferentiable_outputs
    assert contract.randomness.mode == "deterministic"
    assert contract.ports == result.model.model_ports()
    gradient = contract.derivative.admit(DifferentiationRequest((INPUT,)))
    assert not gradient.supported
    assert "regularity-degenerate" in gradient.reasons


def _agrees_with_fit(result):
    contract = result.model.model_execution_contract()
    fit = result.derivative_contract
    assert contract.regularity == fit.regularity
    for surface in (INPUT, PARAMETER):
        assert contract.derivative.level(surface) is fit.level(surface)
    return contract


def test_kernel_expansion_regularity_follows_the_kernel():
    batch = _regression_batch()
    smooth = _agrees_with_fit(
        KernelRidgeRecipe(SquaredExponentialKernel(), alpha=0.1).fit_batch(batch)
    )
    assert smooth.regularity == DerivativeRegularity.smooth()
    physical = DifferentiationRequest(
        (INPUT,), order=3, authority=ComponentAuthority.MODEL
    )
    assert smooth.derivative.admit(physical).supported
    matern = _agrees_with_fit(
        KernelRidgeRecipe(Matern32Kernel(), alpha=0.1).fit_batch(batch)
    )
    assert matern.regularity.continuity == 1
    assert matern.derivative.level(INPUT) is GradientLevel.SMOOTH
    assert not matern.derivative.admit(physical).supported


@pytest.mark.parametrize("metric", ["squared-euclidean", "euclidean", "manhattan"])
def test_exact_neighbors_are_piecewise_constant(metric):
    contract = _agrees_with_fit(
        KNeighborsRegressorRecipe(3, metric=metric).fit_batch(_regression_batch())
    )
    assert contract.regularity == _PIECEWISE_CONSTANT
    assert contract.derivative.level(INPUT) is GradientLevel.NONE


def test_kernel_neighbors_are_smooth_only_for_a_smooth_metric():
    smooth = _agrees_with_fit(
        KernelNeighborsRegressorRecipe(
            temperature=0.4, metric="squared-euclidean"
        ).fit_batch(_regression_batch())
    )
    assert smooth.regularity == DerivativeRegularity.smooth()
    kinked = _agrees_with_fit(
        KernelNeighborsRegressorRecipe(temperature=0.4, metric="euclidean").fit_batch(
            _regression_batch()
        )
    )
    assert kinked.regularity == DerivativeRegularity.piecewise_smooth(continuity=0)
    assert kinked.derivative.level(INPUT) is GradientLevel.ALMOST_EVERYWHERE
