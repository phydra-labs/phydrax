"""Public native singular-subspace lifecycle and model inference contracts."""

from jax import Array
from typing_extensions import assert_type

from phydrax.linalg import DenseLinearOperator
from phydrax.linalg.svd import (
    plan_svd,
    prepare_svd,
    PreparedSVDSolve,
    projector_action,
    RandomizedSVD,
    refresh_svd,
    require_exact_svd_rank,
    SingularSubspaceResponse,
    svd,
    SVDProblem,
    SVDRankEvidence,
    SVDSolvePlan,
    SVDSolvePolicy,
    SVDSolveResult,
)
from phydrax.ml import MLBatch
from phydrax.ml.decomposition import PCA, SubspaceModel
from phydrax.typing import PRNGKey


def native_types(matrix: Array, key: PRNGKey, probe: Array) -> None:
    problem = SVDProblem(DenseLinearOperator(matrix))
    method = RandomizedSVD(oversampling=4, power_iterations=1)
    assert_type(method, RandomizedSVD)
    policy = SVDSolvePolicy(method, count=2, differentiation="projector")
    plan = plan_svd(problem, policy)
    assert_type(plan, SVDSolvePlan)
    prepared = prepare_svd(problem, plan, key=key)
    assert_type(prepared, PreparedSVDSolve)
    assert_type(refresh_svd(prepared, problem), PreparedSVDSolve)
    result = svd(prepared)
    assert_type(result, SVDSolveResult)
    assert_type(result.rank_evidence, SVDRankEvidence)
    assert_type(result.rank_evidence.lower_bound, Array)
    assert_type(result.rank_evidence.upper_bound, Array)
    assert_type(result.derivative_valid, Array)
    assert_type(result.right_response, SingularSubspaceResponse)
    assert_type(
        projector_action(result.right_coordinates, result.right_response, probe), Array
    )
    assert_type(require_exact_svd_rank(result), Array)
    RandomizedSVD(power_iterations=0.5)  # ty: ignore[invalid-argument-type]
    SVDSolvePolicy(differentiation="covariance")  # ty: ignore[invalid-argument-type]
    prepare_svd(problem, policy, key="seed")  # ty: ignore[invalid-argument-type]


def fitted_types(values: Array, key: PRNGKey, probe: Array) -> None:
    recipe = PCA(2, method=RandomizedSVD(), differentiate="projector")
    result = recipe.fit_batch(MLBatch(values), key=key)
    model = result.as_trainable()
    if not isinstance(model, SubspaceModel):
        raise TypeError("PCA must produce a SubspaceModel.")
    assert_type(model.components, Array)
    assert_type(model.weighted_components, Array)
    assert_type(model.project(probe), Array)
    assert_type(model.projector(), Array)
    assert_type(model.prediction_only(), SubspaceModel)
    PCA(2, method="randomized")  # ty: ignore[invalid-argument-type]
