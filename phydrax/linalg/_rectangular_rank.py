#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Operator-scoped rank certificates built on the native SVD lifecycle.

The rank evidence itself is the native `SVDRankEvidence` of one SVD solve; this
owner binds it to an operator identity and numeric revision and adds the
singular-value gap that a fixed-rank (derivative-admitting) claim needs.
"""

from __future__ import annotations

from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..typing import checked, Float, parse, PRNGKey, Scalar
from ._operators import AbstractLinearOperator, DenseLinearOperator
from ._rank import rank_gap_margin, singular_value_backward_error
from ._spaces import _coordinate_dtype
from ._svd import _cost, _svd_method_geometry, svd
from ._svd_contracts import (
    DenseSVD,
    RandomizedSVD,
    SVDCostEstimate,
    SVDProblem,
    SVDRankEvidence,
    SVDSolvePolicy,
)


RectangularRankRoute: TypeAlias = Literal["dense-svd", "randomized-svd", "unavailable"]


@final
class RectangularRankCertificate(StrictModule):
    """Native SVD rank evidence scoped to one revision of one operator.

    ``rank_evidence`` is the `SVDRankEvidence` of the executed native SVD in the
    operator's declared pairings (``None`` for ``route="unavailable"``, an explicit
    refusal whose ``reason`` and ``cost`` record the exceeded SVD budget).
    ``rank`` is the certified numerical rank when the evidence bounds coincide and
    ``-1`` otherwise; randomized certificates without full deterministic coverage
    keep their ``rank_evidence`` bounds and ``failure_probability`` label.

    ``smallest_retained_singular_value`` and ``largest_discarded_singular_value``
    bound the singular values adjacent to the rank decision (lower and upper
    bounds respectively, including the native spectrum allowance and omitted
    tail); ``singular_value_error_bound`` is the factorization backward error.
    ``fixed_rank`` holds only for deterministic exact evidence whose gap to the
    cutoff interval exceeds that error bound: by Weyl's inequality the rank is then
    invariant under every perturbation smaller than the remaining margin, which is
    what an implicit pseudoinverse derivative requires.

    ``svd_status`` is the `SVDSolveStatus` of the executed solve (``-1`` when no
    SVD ran). A solve whose driver did not converge reports ``NONFINITE_OUTPUT``,
    unavailable rank evidence (``rank == -1``), NaN singular-value bounds, and
    ``converged=False``; it never looks like a computed rank.

    ``revision_operator`` retains the complete certified operator PyTree,
    including its numeric pairing state. :meth:`matches` compares every leaf
    exactly; an action on a probe cannot establish numeric revision identity.
    """

    __strict_contract__ = True
    operator_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    route: RectangularRankRoute = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)
    cost: SVDCostEstimate
    rank_evidence: SVDRankEvidence | None
    converged: Array
    svd_status: Array
    smallest_retained_singular_value: Float[Scalar]
    largest_discarded_singular_value: Float[Scalar]
    singular_value_error_bound: Float[Scalar]
    revision_operator: AbstractLinearOperator | None

    @checked
    def __init__(
        self,
        *,
        operator_id: str,
        route: RectangularRankRoute,
        reason: str,
        rows: int,
        columns: int,
        cost: SVDCostEstimate,
        rank_evidence: SVDRankEvidence | None,
        converged: ArrayLike,
        svd_status: ArrayLike,
        smallest_retained_singular_value: ArrayLike,
        largest_discarded_singular_value: ArrayLike,
        singular_value_error_bound: ArrayLike,
        revision_operator: AbstractLinearOperator | None,
    ) -> None:
        route_ = parse(route, RectangularRankRoute, "route")
        if not operator_id or not reason:
            raise ValueError("Rank certificate operator_id and reason must be non-empty.")
        if rows < 1 or columns < 1:
            raise ValueError("Rank certificate dimensions must be positive.")
        match route_:
            case "dense-svd" | "randomized-svd":
                if rank_evidence is None or revision_operator is None:
                    raise ValueError(
                        "An executed rank certificate needs SVD evidence and an operator."
                    )
            case "unavailable":
                if rank_evidence is not None or revision_operator is not None:
                    raise ValueError("An unavailable certificate carries no evidence.")
            case unknown:
                assert_never(unknown)
        if revision_operator is not None and (
            revision_operator.operator_id != operator_id
            or revision_operator.source.size != columns
            or revision_operator.target.size != rows
            or revision_operator.batch_shape
        ):
            raise ValueError("revision_operator must match the certificate scope.")
        self.operator_id = operator_id
        self.route = route_
        self.reason = reason
        self.rows = rows
        self.columns = columns
        self.cost = cost
        self.rank_evidence = rank_evidence
        self.converged = jnp.asarray(converged, dtype=jnp.bool_)
        self.svd_status = jnp.asarray(svd_status, dtype=jnp.int32)
        self.smallest_retained_singular_value = jnp.asarray(
            smallest_retained_singular_value
        )
        self.largest_discarded_singular_value = jnp.asarray(
            largest_discarded_singular_value
        )
        self.singular_value_error_bound = jnp.asarray(singular_value_error_bound)
        self.revision_operator = revision_operator
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "rectangular-rank-certificate",
                "operator": operator_id,
                "route": route_,
                "reason": reason,
                "rows": rows,
                "columns": columns,
                "certificate_kind": (
                    None if rank_evidence is None else rank_evidence.certificate_kind
                ),
            }
        )

    @property
    def maximum_rank(self) -> int:
        return min(self.rows, self.columns)

    @property
    def rank(self) -> Array:
        """Certified numerical rank, or ``-1`` when the bounds do not coincide."""
        evidence = self.rank_evidence
        if evidence is None:
            return jnp.asarray(-1, dtype=jnp.int32)
        exact = evidence.available & (evidence.lower_bound == evidence.upper_bound)
        return jnp.where(exact, evidence.lower_bound, -1).astype(jnp.int32)

    @property
    def right_nullity(self) -> Array:
        return jnp.where(self.rank >= 0, self.columns - self.rank, -1).astype(jnp.int32)

    @property
    def left_nullity(self) -> Array:
        return jnp.where(self.rank >= 0, self.rows - self.rank, -1).astype(jnp.int32)

    @property
    def gap_margin(self) -> Array:
        """Distance of the rank decision from the evidence cutoff interval."""
        evidence = self.rank_evidence
        if evidence is None:
            return jnp.asarray(jnp.nan, dtype=self.singular_value_error_bound.dtype)
        return rank_gap_margin(
            self.rank,
            self.maximum_rank,
            self.smallest_retained_singular_value,
            self.largest_discarded_singular_value,
            evidence.threshold_lower,
            evidence.threshold_upper,
        )

    @property
    def fixed_rank(self) -> Array:
        """Whether deterministic evidence certifies a locally constant rank."""
        evidence = self.rank_evidence
        if evidence is None:
            return jnp.asarray(False)
        margin = self.gap_margin
        return (
            evidence.deterministic_exact
            & self.converged
            & (self.rank >= 0)
            & ~jnp.isnan(margin)
            & jnp.isfinite(self.singular_value_error_bound)
            & (margin > self.singular_value_error_bound)
        )

    @property
    def storage_bytes(self) -> int:
        evidence = self.rank_evidence
        arrays: tuple[Array, ...] = (
            self.converged,
            self.smallest_retained_singular_value,
            self.largest_discarded_singular_value,
            self.singular_value_error_bound,
        )
        if evidence is not None:
            arrays += (
                evidence.lower_bound,
                evidence.upper_bound,
                evidence.threshold_lower,
                evidence.threshold_upper,
                evidence.available,
                evidence.deterministic_exact,
                evidence.failure_probability,
            )
        if self.revision_operator is not None:
            arrays += tuple(
                leaf
                for leaf in jax.tree.leaves(self.revision_operator)
                if eqx.is_array(leaf)
            )
        return sum(jnp.asarray(array).nbytes for array in arrays)

    @checked
    def matches(self, operator: AbstractLinearOperator, /) -> Array:
        """Return whether ``operator`` is the certified numeric revision."""
        require_certificate_scope(self, operator)
        if self.revision_operator is None:
            return jnp.asarray(False)
        return jnp.asarray(eqx.tree_equal(operator, self.revision_operator))


@checked
def require_certificate_scope(
    certificate: RectangularRankCertificate,
    operator: AbstractLinearOperator,
    /,
) -> None:
    """Refuse a certificate whose operator identity or dimensions differ."""
    if (
        operator.batch_shape
        or certificate.operator_id != operator.operator_id
        or certificate.rows != operator.target.size
        or certificate.columns != operator.source.size
    ):
        raise ValueError(
            "Rank certificate scope does not match the unbatched operator identity "
            "and dimensions."
        )


def _route(policy: SVDSolvePolicy, /) -> RectangularRankRoute:
    match policy.method:
        case DenseSVD():
            return "dense-svd"
        case RandomizedSVD():
            return "randomized-svd"
        case method:
            raise TypeError(f"Unsupported SVD method {type(method).__name__}.")


@checked
def certify_rectangular_rank(
    operator: AbstractLinearOperator,
    policy: SVDSolvePolicy | None = None,
    /,
    *,
    key: PRNGKey | None = None,
) -> RectangularRankCertificate:
    """Certify the numerical rank of ``operator`` through one native SVD solve.

    ``policy`` is the native `SVDSolvePolicy` (default: `DenseSVD` with the QR
    driver, robust on exactly clustered spectra, and the default `RankPolicy`);
    its rank cutoff, materialization limits, and resource
    budgets apply unchanged. A `RandomizedSVD` policy needs ``key`` and yields
    deterministic evidence only with full coverage of a resident dense operator.
    When the planned SVD exceeds its budgets, no SVD runs and the certificate is
    ``route="unavailable"`` with the refusal reason and the refused cost estimate.
    """
    problem = SVDProblem(operator)
    selected = SVDSolvePolicy(DenseSVD(algorithm="qr")) if policy is None else policy
    rows, columns = operator.target.size, operator.source.size
    resident = isinstance(operator, DenseLinearOperator)
    width, _ = _svd_method_geometry(problem, selected, resident)
    cost = _cost(problem, selected, width, resident)
    real_dtype = jnp.finfo(_coordinate_dtype(operator.source)).dtype
    if not cost.accepted:
        unavailable = jnp.asarray(jnp.nan, dtype=real_dtype)
        return RectangularRankCertificate(
            operator_id=operator.operator_id,
            route="unavailable",
            reason=cost.reason,
            rows=rows,
            columns=columns,
            cost=cost,
            rank_evidence=None,
            converged=False,
            svd_status=-1,
            smallest_retained_singular_value=unavailable,
            largest_discarded_singular_value=unavailable,
            singular_value_error_bound=unavailable,
            revision_operator=None,
        )
    result = svd(problem, policy=selected, key=key)
    evidence = result.rank_evidence
    bounds = result.range_evidence
    lower_bounds = bounds.singular_value_lower_bounds
    upper_bounds = bounds.singular_value_upper_bounds
    computed = lower_bounds.shape[0]
    rank = evidence.lower_bound
    infinity = jnp.asarray(jnp.inf, dtype=lower_bounds.dtype)
    not_available = jnp.asarray(jnp.nan, dtype=lower_bounds.dtype)
    smallest_retained = jnp.where(
        evidence.available,
        jnp.where(rank > 0, lower_bounds[jnp.maximum(rank - 1, 0)], infinity),
        not_available,
    )
    largest_discarded = jnp.where(
        evidence.available,
        jnp.where(
            rank < computed,
            upper_bounds[jnp.minimum(rank, computed - 1)],
            jnp.where(
                rank < problem.maximum_rank, bounds.omitted_spectrum_upper_bound, 0.0
            ),
        ),
        not_available,
    )
    return RectangularRankCertificate(
        operator_id=operator.operator_id,
        route=_route(selected),
        reason=f"native {selected.method.name} rank evidence ({evidence.certificate_kind})",
        rows=rows,
        columns=columns,
        cost=cost,
        rank_evidence=evidence,
        converged=evidence.available & jnp.all(result.converged),
        svd_status=result.status,
        smallest_retained_singular_value=smallest_retained,
        largest_discarded_singular_value=largest_discarded,
        singular_value_error_bound=singular_value_backward_error(
            upper_bounds[0], rows, columns
        ),
        revision_operator=operator,
    )


__all__ = [
    "RectangularRankCertificate",
    "RectangularRankRoute",
    "certify_rectangular_rank",
]
