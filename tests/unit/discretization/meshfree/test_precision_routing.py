"""Declared meshfree precision roles through neighborhoods, fits and operators.

Oracles are independent: the Laplacian of ``x^2 + y^2`` is exactly 4, and the
float64 fit of the same stencil serves as the positive control of a refusal.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import qmc

from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    MeshfreeOperator,
    MeshfreePrecisionPolicy,
    MeshfreeRowStatus,
    prepare_local_stencils,
)


_LAPLACIAN = MeshfreeFunctional(((2, 0), (0, 2)), np.ones(2), name="laplacian")


def _cloud(count: int = 256) -> np.ndarray:
    return qmc.LatinHypercube(d=2, seed=4).random(count)


@pytest.mark.parametrize(
    ("policy", "roles"),
    (
        pytest.param(
            MeshfreePrecisionPolicy(
                geometry_dtype="float32", certification_dtype="float64"
            ),
            {
                "geometry": "float32",
                "coefficient": "float32",
                "fit": "float32",
                "certification": "float64",
            },
            id="float32-wide-certification",
        ),
        pytest.param(
            MeshfreePrecisionPolicy(
                geometry_dtype="float64", fit_dtype="float64", coefficient_dtype="float32"
            ),
            {
                "geometry": "float64",
                "coefficient": "float32",
                "fit": "float64",
                "certification": "float64",
            },
            id="float64-fit-float32-coefficients",
        ),
    ),
)
def test_declared_roles_reach_selection_fit_weights_and_operator(
    policy: MeshfreePrecisionPolicy, roles: dict[str, str]
) -> None:
    points = _cloud()
    neighborhood = MeshfreeNeighborhoodPlan(points, 12, precision=policy).prepare()
    # Selection and distances are decided in the certification role.
    assert neighborhood.distances.dtype == np.dtype(policy.certification_dtype)
    assert neighborhood.precision.policy_id == policy.policy_id

    stencils = prepare_local_stencils(
        neighborhood,
        points,
        points,
        (_LAPLACIAN,),
        LocalStencilPolicy(polynomial_degree=2),
    )
    assert stencils.report.refused_rows == 0
    assert dict(stencils.report.precision) == roles
    assert stencils.weights[0].dtype == np.dtype(policy.coefficient_dtype)

    operator = MeshfreeOperator(stencils)
    assert operator.operator.accumulation_dtype == np.dtype(policy.accumulation_dtype)
    quadratic = jnp.asarray(np.sum(points**2, axis=1), dtype=policy.compute_dtype)
    laplacian = operator.mv(quadratic)
    assert laplacian.dtype == np.dtype(policy.output_dtype)
    # Rounding of float32 data, weights and accumulation bounds the defect by
    # a modest multiple of eps32 times the row sum of absolute weights.
    bound = 64 * np.finfo(np.float32).eps * stencils.report.maximum_amplification * 2.0
    defect = np.max(np.abs(np.asarray(laplacian, dtype=np.float64) - 4.0))
    assert defect <= bound


def test_float32_phs_saddle_is_refused_not_silently_inaccurate() -> None:
    points = _cloud()
    single = MeshfreePrecisionPolicy(geometry_dtype="float32")
    phs = LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=2)
    neighborhood = MeshfreeNeighborhoodPlan(points, 12, precision=single).prepare()
    with pytest.raises(ValueError, match="ILL_CONDITIONED|MOMENT_FAILURE"):
        prepare_local_stencils(neighborhood, points, points, (_LAPLACIAN,), phs)

    masked = prepare_local_stencils(
        neighborhood,
        points,
        points,
        (_LAPLACIAN,),
        LocalStencilPolicy(
            approximation="phs-rbf-fd", polynomial_degree=2, acceptance="mask"
        ),
    )
    status = np.asarray(masked.evidence.status)
    refused = status != MeshfreeRowStatus.VALID
    assert refused.any()
    # Saddles beyond the float32 conditioning admission are ILL_CONDITIONED;
    # those just below it miss their rounding-level moment reproduction.
    assert set(status[refused].tolist()) <= {
        MeshfreeRowStatus.ILL_CONDITIONED,
        MeshfreeRowStatus.MOMENT_FAILURE,
    }
    assert (status == MeshfreeRowStatus.ILL_CONDITIONED).any()
    assert not np.asarray(masked.weights[0])[refused].any()

    # Positive control: the same saddles are admitted by a float64 fit.
    wide = MeshfreeNeighborhoodPlan(points, 12).prepare()
    admitted = prepare_local_stencils(wide, points, points, (_LAPLACIAN,), phs)
    assert admitted.report.refused_rows == 0
    assert admitted.report.maximum_condition_number * np.finfo(np.float32).eps > 1e-3
