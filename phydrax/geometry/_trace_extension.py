#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from phydrax.enforcement._geometry_support import BoundaryCover
from phydrax.linalg import (
    AbstractLinearOperator,
    ConstraintOperatorPlan,
    materialize,
    PreparedConstraintOperator,
    RankPolicy,
    SolveResourcePolicy,
)

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class TraceExtensionExactness(str, Enum):
    """Scope on which a prepared extension is a certified right inverse."""

    ANALYTIC_REPRESENTED = "analytic_represented"
    DISCRETE_ALGEBRAIC = "discrete_algebraic"
    APPROXIMATE = "approximate"


class TraceExtensionEvidence(StrictModule, NonTrainableState):
    """Algebraic and geometric evidence for a trace right inverse."""

    right_inverse_error: Array
    preservation_error: Array
    numeric_version: Array
    exactness: TraceExtensionExactness = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    trace_operator_id: str = eqx.field(static=True)
    candidate_operator_id: str = eqx.field(static=True)
    response_prepared_id: str = eqx.field(static=True)
    construction_certificate_id: str = eqx.field(static=True)
    preservation_certificate_id: str | None = eqx.field(static=True)
    stability_owner_ids: tuple[str, ...] = eqx.field(static=True)
    represented_geometry_ids: tuple[str, ...] = eqx.field(static=True)
    physical_geometry_ids: tuple[str | None, ...] = eqx.field(static=True)
    physical_exact: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        right_inverse_error: Array,
        preservation_error: Array,
        numeric_version: int,
        exactness: TraceExtensionExactness | str,
        provider_id: str,
        cover: BoundaryCover,
        trace_operator_id: str,
        candidate_operator_id: str,
        response_prepared_id: str,
        construction_certificate_id: str,
        preservation_certificate_id: str | None,
        stability_owner_ids: tuple[str, ...],
    ):
        inverse_error = jnp.asarray(right_inverse_error, dtype=float).reshape(())
        invariant_error = jnp.asarray(preservation_error, dtype=float).reshape(())
        version_int = int(numeric_version)
        if version_int < 0:
            raise ValueError("numeric_version must be non-negative.")
        version = jnp.asarray(version_int, dtype=jnp.int32)
        exactness_ = TraceExtensionExactness(exactness)
        provider = str(provider_id)
        trace_id = str(trace_operator_id)
        candidate_id = str(candidate_operator_id)
        response_id = str(response_prepared_id)
        construction = str(construction_certificate_id)
        stability = tuple(str(value) for value in stability_owner_ids)
        preservation = (
            None
            if preservation_certificate_id is None
            else str(preservation_certificate_id)
        )
        if (
            not provider
            or not trace_id
            or not candidate_id
            or not response_id
            or not construction
            or any(not value for value in stability)
        ):
            raise ValueError("Trace-extension evidence IDs must be non-empty.")
        if preservation is not None and not preservation:
            raise ValueError("preservation_certificate_id must be non-empty.")
        if exactness_ is not TraceExtensionExactness.APPROXIMATE:
            if not cover.evidence.coverage_complete:
                raise ValueError(
                    "Exact trace extension requires complete boundary coverage."
                )
            if not cover.evidence.intersections_resolved:
                raise ValueError("Exact trace extension requires resolved intersections.")
            if not cover.evidence.orientation_valid:
                raise ValueError("Exact trace extension requires valid orientations.")
        self.right_inverse_error = inverse_error
        self.preservation_error = invariant_error
        self.numeric_version = version
        self.exactness = exactness_
        self.provider_id = provider
        self.support_id = cover.cover_id
        self.trace_operator_id = trace_id
        self.candidate_operator_id = candidate_id
        self.response_prepared_id = response_id
        self.construction_certificate_id = construction
        self.preservation_certificate_id = preservation
        self.stability_owner_ids = stability
        self.represented_geometry_ids = cover.evidence.represented_geometry_ids
        self.physical_geometry_ids = cover.evidence.physical_geometry_ids
        self.physical_exact = cover.evidence.physical_exact
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "trace-extension-evidence-v1",
                "provider": provider,
                "support": cover.cover_id,
                "trace": self.trace_operator_id,
                "candidate": self.candidate_operator_id,
                "response": self.response_prepared_id,
                "construction": construction,
                "preservation": preservation,
                "stability_owners": stability,
                "exactness": exactness_.value,
                "physical_exact": self.physical_exact,
                "numeric_version": int(numeric_version),
            }
        )


class PreparedTraceExtension(StrictModule, NonTrainableState):
    """Reusable action R = E (C E)⁺ with explicit adjoint action."""

    trace_operator: AbstractLinearOperator
    candidate_operator: AbstractLinearOperator
    response: PreparedConstraintOperator
    right_inverse_operator: AbstractLinearOperator
    preservation_operator: AbstractLinearOperator | None
    evidence: TraceExtensionEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        trace_operator: AbstractLinearOperator,
        candidate_operator: AbstractLinearOperator,
        response: PreparedConstraintOperator,
        right_inverse_operator: AbstractLinearOperator,
        /,
        *,
        preservation_operator: AbstractLinearOperator | None,
        evidence: TraceExtensionEvidence,
    ):
        if not isinstance(trace_operator, AbstractLinearOperator):
            raise TypeError("trace_operator must be an AbstractLinearOperator.")
        if not isinstance(candidate_operator, AbstractLinearOperator):
            raise TypeError("candidate_operator must be an AbstractLinearOperator.")
        if not isinstance(response, PreparedConstraintOperator):
            raise TypeError("response must be a PreparedConstraintOperator.")
        if not isinstance(right_inverse_operator, AbstractLinearOperator):
            raise TypeError("right_inverse_operator must be an AbstractLinearOperator.")
        if preservation_operator is not None and not isinstance(
            preservation_operator, AbstractLinearOperator
        ):
            raise TypeError("preservation_operator must be linear or None.")
        expected_response = trace_operator @ candidate_operator
        if response.operator.operator_id != expected_response.operator_id:
            raise ValueError("Prepared response must be the declared C E composition.")
        if not response.source_space.compatible(expected_response.source) or not (
            response.target_space.compatible(expected_response.target)
        ):
            raise ValueError("Prepared response spaces do not match C E.")
        expected_right_inverse = candidate_operator @ response.right_inverse_operator
        if right_inverse_operator.operator_id != expected_right_inverse.operator_id:
            raise ValueError(
                "Trace right inverse must be the declared E (C E)⁺ composition."
            )
        if not right_inverse_operator.source.compatible(trace_operator.target) or not (
            right_inverse_operator.target.compatible(trace_operator.source)
        ):
            raise ValueError(
                "Trace right inverse must map trace values into field space."
            )
        if (
            preservation_operator is not None
            and not preservation_operator.source.compatible(trace_operator.source)
        ):
            raise ValueError(
                "Preservation operator must act on the extension field space."
            )
        if not isinstance(evidence, TraceExtensionEvidence):
            raise TypeError("evidence must be TraceExtensionEvidence.")
        if (
            evidence.trace_operator_id != trace_operator.operator_id
            or evidence.candidate_operator_id != candidate_operator.operator_id
            or evidence.response_prepared_id != response.prepared_id
        ):
            raise ValueError("Trace-extension evidence does not match its operators.")
        self.trace_operator = trace_operator
        self.candidate_operator = candidate_operator
        self.response = response
        self.right_inverse_operator = right_inverse_operator
        self.preservation_operator = preservation_operator
        self.evidence = evidence
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-trace-extension-v1",
                "trace": trace_operator.operator_id,
                "candidate": candidate_operator.operator_id,
                "response": response.prepared_id,
                "right_inverse": right_inverse_operator.operator_id,
                "evidence": evidence.evidence_id,
            }
        )

    @property
    def source_space(self):
        return self.right_inverse_operator.source

    @property
    def target_space(self):
        return self.right_inverse_operator.target

    def lift(self, trace_residual: PyTree[Any], /) -> PyTree[Array]:
        """Apply the certified trace right inverse."""
        return self.right_inverse_operator.mv(
            self.right_inverse_operator.source.validate(trace_residual)
        )

    def adjoint_lift(self, field_value: PyTree[Any], /) -> PyTree[Array]:
        """Apply the pairing-aware adjoint of the trace lift."""
        return self.right_inverse_operator.adjoint_mv(
            self.right_inverse_operator.target.validate(field_value)
        )

    def trace_lift(self, trace_residual: PyTree[Any], /) -> PyTree[Array]:
        return self.trace_operator.mv(self.lift(trace_residual))

    def preservation_lift(self, trace_residual: PyTree[Any], /) -> PyTree[Array] | None:
        if self.preservation_operator is None:
            return None
        return self.preservation_operator.mv(self.lift(trace_residual))


class _TraceCorrectionProvider(StrictModule, NonTrainableState):
    __strict_abstract__ = True
    trace_operator: AbstractLinearOperator
    candidate_operator: AbstractLinearOperator
    cover: BoundaryCover
    preservation_operator: AbstractLinearOperator | None
    exactness: TraceExtensionExactness = eqx.field(static=True)
    construction_certificate_id: str = eqx.field(static=True)
    preservation_certificate_id: str | None = eqx.field(static=True)
    stability_owner_ids: tuple[str, ...] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        trace_operator: AbstractLinearOperator,
        candidate_operator: AbstractLinearOperator,
        cover: BoundaryCover,
        /,
        *,
        exactness: TraceExtensionExactness,
        construction_certificate_id: str,
        preservation_operator: AbstractLinearOperator | None = None,
        preservation_certificate_id: str | None = None,
        stability_owner_ids: tuple[str, ...] = (),
        provider_kind: str,
    ):
        if not isinstance(trace_operator, AbstractLinearOperator) or not isinstance(
            candidate_operator, AbstractLinearOperator
        ):
            raise TypeError("Trace and candidate maps must be native linear operators.")
        if not isinstance(cover, BoundaryCover):
            raise TypeError("cover must be a BoundaryCover.")
        if not trace_operator.source.compatible(candidate_operator.target):
            raise ValueError("Candidate extensions must map into the trace source space.")
        exactness_ = TraceExtensionExactness(exactness)
        preservation_id = (
            None
            if preservation_certificate_id is None
            else str(preservation_certificate_id)
        )
        if preservation_operator is not None:
            if not isinstance(preservation_operator, AbstractLinearOperator):
                raise TypeError("preservation_operator must be linear or None.")
            if not preservation_operator.source.compatible(trace_operator.source):
                raise ValueError("Preservation operator acts on the wrong field space.")
            if preservation_id is None or not preservation_id:
                raise ValueError(
                    "A preservation operator requires an explicit preservation certificate."
                )
        elif preservation_id is not None:
            raise ValueError(
                "preservation_certificate_id requires a preservation operator."
            )
        construction = str(construction_certificate_id)
        kind = str(provider_kind)
        if not construction or not kind:
            raise ValueError(
                "Provider kind and construction certificate must be non-empty."
            )
        stability = tuple(str(value) for value in stability_owner_ids)
        if any(not value for value in stability):
            raise ValueError("stability_owner_ids must be non-empty strings.")
        self.trace_operator = trace_operator
        self.candidate_operator = candidate_operator
        self.cover = cover
        self.preservation_operator = preservation_operator
        self.exactness = exactness_
        self.construction_certificate_id = construction
        self.preservation_certificate_id = preservation_id
        self.stability_owner_ids = stability
        self.provider_id = canonical_fingerprint(
            {
                "kind": kind,
                "trace": trace_operator.operator_id,
                "candidate": candidate_operator.operator_id,
                "cover": cover.cover_id,
                "construction": construction,
                "preservation": preservation_id,
                "stability_owners": stability,
                "exactness": exactness_.value,
            }
        )

    def prepare(
        self,
        /,
        *,
        rank: RankPolicy | None = None,
        resources: SolveResourcePolicy | None = None,
        numeric_version: int = 0,
    ) -> PreparedTraceExtension:
        response_operator = self.trace_operator @ self.candidate_operator
        response = ConstraintOperatorPlan(
            response_operator,
            require_full_row_rank=True,
            rank=rank,
            resources=resources,
        ).prepare()
        right_inverse = self.candidate_operator @ response.right_inverse_operator
        preservation_error = jnp.asarray(0.0)
        if self.preservation_operator is not None:
            preserved_lift = self.preservation_operator @ right_inverse
            matrix = materialize(preserved_lift, response.plan.materialization)
            preservation_error = jnp.linalg.norm(matrix)
        evidence = TraceExtensionEvidence(
            right_inverse_error=response.evidence.strict_right_inverse_residual_norm,
            preservation_error=preservation_error,
            numeric_version=numeric_version,
            exactness=self.exactness,
            provider_id=self.provider_id,
            cover=self.cover,
            trace_operator_id=self.trace_operator.operator_id,
            candidate_operator_id=self.candidate_operator.operator_id,
            response_prepared_id=response.prepared_id,
            construction_certificate_id=self.construction_certificate_id,
            preservation_certificate_id=self.preservation_certificate_id,
            stability_owner_ids=self.stability_owner_ids,
        )
        return PreparedTraceExtension(
            self.trace_operator,
            self.candidate_operator,
            response,
            right_inverse,
            preservation_operator=self.preservation_operator,
            evidence=evidence,
        )


class TransfiniteCorrectionProvider(_TraceCorrectionProvider):
    """Exact represented transfinite/Hermite candidate extension provider."""

    def __init__(
        self, trace_operator, candidate_operator, cover, /, *, cardinal_certificate_id
    ):
        super().__init__(
            trace_operator,
            candidate_operator,
            cover,
            exactness=TraceExtensionExactness.ANALYTIC_REPRESENTED,
            construction_certificate_id=cardinal_certificate_id,
            provider_kind="transfinite-trace-correction-v1",
        )


class ClosestPointCorrectionProvider(_TraceCorrectionProvider):
    """Exact represented collar extension over certified regular retractions."""

    def __init__(
        self, trace_operator, candidate_operator, cover, /, *, collar_certificate_id
    ):
        if any(
            patch.collar_provider is None or patch.collar_certificate_id is None
            for patch in cover.patches
        ):
            raise ValueError(
                "Closest-point correction requires a certified collar on every patch."
            )
        declared = str(collar_certificate_id)
        if not declared or any(
            patch.collar_certificate_id != declared for patch in cover.patches
        ):
            raise ValueError("Closest-point collar certificate IDs are inconsistent.")
        super().__init__(
            trace_operator,
            candidate_operator,
            cover,
            exactness=TraceExtensionExactness.ANALYTIC_REPRESENTED,
            construction_certificate_id=declared,
            provider_kind="closest-point-trace-correction-v1",
        )


class PartitionOfUnityCorrectionProvider(_TraceCorrectionProvider):
    """Certified subordinate-cover extension with a full overlap response solve."""

    def __init__(
        self,
        trace_operator,
        candidate_operator,
        cover,
        /,
        *,
        partition_certificate_id,
    ):
        super().__init__(
            trace_operator,
            candidate_operator,
            cover,
            exactness=TraceExtensionExactness.ANALYTIC_REPRESENTED,
            construction_certificate_id=partition_certificate_id,
            provider_kind="partition-of-unity-trace-correction-v1",
        )


class EllipticExtensionCorrectionProvider(_TraceCorrectionProvider):
    """Coercive minimum-energy extension; pure Neumann paths require a gauge."""

    pure_neumann: bool = eqx.field(static=True)
    gauge_certificate_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        trace_operator,
        candidate_operator,
        cover,
        /,
        *,
        coercivity_certificate_id,
        pure_neumann=False,
        gauge_certificate_id=None,
        preservation_operator=None,
        preservation_certificate_id=None,
    ):
        pure = bool(pure_neumann)
        gauge = None if gauge_certificate_id is None else str(gauge_certificate_id)
        if pure and (gauge is None or not gauge):
            raise ValueError(
                "Pure-Neumann elliptic extension requires a gauge certificate."
            )
        super().__init__(
            trace_operator,
            candidate_operator,
            cover,
            exactness=TraceExtensionExactness.ANALYTIC_REPRESENTED,
            construction_certificate_id=coercivity_certificate_id,
            preservation_operator=preservation_operator,
            preservation_certificate_id=preservation_certificate_id,
            provider_kind="elliptic-trace-correction-v1",
        )
        self.pure_neumann = pure
        self.gauge_certificate_id = gauge


class DiscreteTraceCorrectionProvider(_TraceCorrectionProvider):
    """Algebraically exact Cₕ right inverse owned by a declared representation."""

    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        trace_operator,
        candidate_operator,
        cover,
        /,
        *,
        representation_id,
        representation_certificate_id,
        preservation_operator=None,
        preservation_certificate_id=None,
        stability_owner_ids=(),
    ):
        representation = str(representation_id)
        if not representation:
            raise ValueError("representation_id must be non-empty.")
        super().__init__(
            trace_operator,
            candidate_operator,
            cover,
            exactness=TraceExtensionExactness.DISCRETE_ALGEBRAIC,
            construction_certificate_id=representation_certificate_id,
            preservation_operator=preservation_operator,
            preservation_certificate_id=preservation_certificate_id,
            stability_owner_ids=tuple(stability_owner_ids),
            provider_kind="discrete-trace-correction-v1",
        )
        self.representation_id = representation


__all__ = [
    "ClosestPointCorrectionProvider",
    "DiscreteTraceCorrectionProvider",
    "EllipticExtensionCorrectionProvider",
    "PartitionOfUnityCorrectionProvider",
    "PreparedTraceExtension",
    "TraceExtensionEvidence",
    "TraceExtensionExactness",
    "TransfiniteCorrectionProvider",
]
