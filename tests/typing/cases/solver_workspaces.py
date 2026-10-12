"""Operator curvature and explicit solver workspace policies stay statically typed."""

from collections.abc import Callable

from jax import Array
from typing_extensions import assert_type

from phydrax.control import ILQRLinearizationPolicy, ILQRLinearizationRoute
from phydrax.linalg import (
    AbstractLinearOperator,
    inverse_residual_certificate,
    InverseResidualCertificate,
)
from phydrax.nonlinear import CausalAdjointPolicy, CausalAdjointRoute
from phydrax.optim import (
    ResidualGraphLinearization,
    SchurComplementPlan,
    solve_schur_system,
)


def operator_curvature(
    model: ResidualGraphLinearization,
    gradient: Array,
    schur: SchurComplementPlan,
    inverse_action: Callable[[Array], tuple[Array, Array]],
) -> None:
    assert_type(model.curvature, AbstractLinearOperator)
    assert_type(model.curvature.target.flatten(model.curvature.mv(gradient)), Array)
    assert_type(solve_schur_system(model.curvature, gradient, schur), Array)
    certificate = inverse_residual_certificate(model.curvature, inverse_action)
    assert_type(certificate, InverseResidualCertificate)
    assert_type(certificate.condition_upper_bound, Array)
    solve_schur_system(gradient, gradient, schur)  # ty: ignore[invalid-argument-type]


def explicit_routes() -> None:
    causal = CausalAdjointPolicy("checkpointed-action-scan", checkpoint_block_size=4)
    assert_type(causal.route, CausalAdjointRoute)
    ilqr = ILQRLinearizationPolicy("reverse-scan", max_batched_bytes=1024)
    assert_type(ilqr.route, ILQRLinearizationRoute)
    CausalAdjointPolicy("scan")  # ty: ignore[invalid-argument-type]
    ILQRLinearizationPolicy("dense")  # ty: ignore[invalid-argument-type]
