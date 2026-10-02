"""Consumer view of an installed Phydrax wheel.

Checked by `tools/check_installed_typing.py` from outside the repository, so every
`phydrax` import resolves to the installed distribution. Deliberately invalid
lines carry `# ty: ignore[<rule>]`; unused suppressions are errors.
"""

from typing import assert_type, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import phydrax as phx
import phydrax.typing as pt
from phydrax.equations import ChemicalComponentCatalog
from phydrax.linalg.svd import (
    prepare_svd,
    PreparedSVDSolve,
    projector_action,
    RandomizedSVD,
    SingularSubspaceResponse,
    svd,
    SVDProblem,
    SVDRankEvidence,
    SVDSolvePolicy,
    SVDSolveResult,
)
from phydrax.precision import precision_dtype_name, ScalarPrecisionDType
from phydrax.solver import (
    analyze_projector_monte_carlo,
    initialize_projector_monte_carlo,
    PreparedProjectorMonteCarlo,
    ProjectorEstimatorPolicy,
    ProjectorMonteCarloAnalysis,
    ProjectorMonteCarloResult,
    ProjectorMonteCarloState,
    ProjectorMonteCarloStepResult,
    solve_projector_monte_carlo,
    step_projector_monte_carlo,
)
from phydrax.uq import CorrelatedRatioResult


class ComponentDim(pt.Dim, minimum=1):
    """Number of chemical components."""


def tensor_forms(
    masses: pt.Float64[ComponentDim],
    table: pt.HostFloat64[pt.AnyDim, ComponentDim],
    count: pt.Size[ComponentDim],
    names: pt.Identifiers[ComponentDim],
    key: pt.PRNGKey,
) -> None:
    assert_type(masses, jax.Array)
    assert_type(table, npt.NDArray[np.float64])
    assert_type(count, int)
    assert_type(names, tuple[str, ...])
    assert_type(key, jax.Array)


def boundaries() -> None:
    basis = pt.parse("dense", Literal["dense", "sparse"], "basis")
    assert_type(basis, Literal["dense", "sparse"])
    values = pt.as_array([1.0, 2.0], pt.Float64[ComponentDim], "values")
    assert_type(values, jax.Array)
    host = pt.as_host_array(values, pt.HostFloat64[ComponentDim], "host")
    assert_type(host, npt.NDArray[np.float64])
    assert_type(precision_dtype_name("float32"), ScalarPrecisionDType)


def constructors() -> None:
    catalog = ChemicalComponentCatalog(
        ("H2",), np.asarray((2.016,)), ("H",), np.asarray(((2,),))
    )
    assert_type(catalog, ChemicalComponentCatalog)
    assert_type(catalog.molar_masses, jax.Array)
    ChemicalComponentCatalog(("H2",))  # ty: ignore[missing-argument]
    ChemicalComponentCatalog(
        ("H2",),
        jnp.ones((1,)),
        ("H",),
        np.ones((1, 1), dtype=np.int32),
        weights=1,  # ty: ignore[unknown-argument]
    )
    phx.typing.validate(catalog)


@pt.checked
def scaled_masses(catalog: ChemicalComponentCatalog, scale: float, /) -> jax.Array:
    return scale * catalog.molar_masses


def checked_signatures(catalog: ChemicalComponentCatalog) -> None:
    assert_type(scaled_masses(catalog, 2.0), jax.Array)
    scaled_masses(catalog)  # ty: ignore[missing-argument]
    scaled_masses("catalog", 2.0)  # ty: ignore[invalid-argument-type]


def singular_subspaces(problem: SVDProblem, key: pt.PRNGKey, probe: jax.Array) -> None:
    policy = SVDSolvePolicy(RandomizedSVD(), count=2, differentiation="projector")
    prepared = prepare_svd(problem, policy, key=key)
    assert_type(prepared, PreparedSVDSolve)
    result = svd(prepared)
    assert_type(result, SVDSolveResult)
    assert_type(result.rank_evidence, SVDRankEvidence)
    assert_type(result.right_response, SingularSubspaceResponse)
    assert_type(result.derivative_valid, jax.Array)
    assert_type(
        projector_action(result.right_coordinates, result.right_response, probe),
        jax.Array,
    )


def projector_consumer(
    prepared: PreparedProjectorMonteCarlo,
    key: pt.PRNGKey,
    result: ProjectorMonteCarloResult,
) -> None:
    state = initialize_projector_monte_carlo(prepared, key)
    assert_type(state, ProjectorMonteCarloState)
    assert_type(state.support_keys, jax.Array)
    assert_type(state.root_key, jax.Array)
    assert_type(
        step_projector_monte_carlo(prepared, state), ProjectorMonteCarloStepResult
    )
    assert_type(
        solve_projector_monte_carlo(prepared, state, steps=1), ProjectorMonteCarloResult
    )
    analysis = analyze_projector_monte_carlo(
        prepared, result, policy=ProjectorEstimatorPolicy()
    )
    assert_type(analysis, ProjectorMonteCarloAnalysis)
    assert_type(analysis.projected, CorrelatedRatioResult)
    assert_type(analysis.projected.mean_covariance, jax.Array)
    initialize_projector_monte_carlo(prepared, 1)  # ty: ignore[invalid-argument-type]
