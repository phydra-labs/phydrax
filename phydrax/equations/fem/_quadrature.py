#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


QuadratureAccuracyKind = Literal[
    "exact-polynomial",
    "collocated",
    "overintegrated",
    "explicit-rule",
]
QuadratureRole = Literal[
    "volume",
    "interior-facet",
    "exterior-facet",
    "projection",
    "observation",
]


class QuadratureAccuracyPolicy(StrictModule, NonTrainableState):
    kind: QuadratureAccuracyKind = eqx.field(static=True)
    overintegration_factor: float = eqx.field(static=True)
    explicit_degree: int | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: QuadratureAccuracyKind = "exact-polynomial",
        /,
        *,
        overintegration_factor: float = 1.5,
        explicit_degree: int | None = None,
    ):
        kind_ = str(kind)
        factor = float(overintegration_factor)
        degree = None if explicit_degree is None else int(explicit_degree)
        if kind_ not in (
            "exact-polynomial",
            "collocated",
            "overintegrated",
            "explicit-rule",
        ):
            raise ValueError("Unknown quadrature accuracy policy.")
        if factor < 1.0:
            raise ValueError("Quadrature overintegration factor must be at least one.")
        if kind_ == "explicit-rule" and (degree is None or degree < 0):
            raise ValueError("Explicit quadrature policy requires a nonnegative degree.")
        if kind_ != "explicit-rule" and degree is not None:
            raise ValueError("explicit_degree requires explicit-rule policy.")
        self.kind = kind_
        self.overintegration_factor = factor
        self.explicit_degree = degree
        self.policy_id = canonical_fingerprint(
            {
                "kind": "quadrature-accuracy-policy",
                "policy": kind_,
                "overintegration_factor": factor,
                "explicit_degree": degree,
            }
        )

    def resolve_degree(
        self,
        trial_order: int,
        test_order: int,
        /,
        *,
        coordinate_order: int = 1,
        coefficient_order: int | None = 0,
        kernel_polynomial_degree: int | None = 1,
    ) -> int:
        trial = int(trial_order)
        test = int(test_order)
        coordinate = int(coordinate_order)
        if min(trial, test, coordinate) < 0:
            raise ValueError("Quadrature polynomial orders must be nonnegative.")
        if self.kind == "explicit-rule":
            return int(self.explicit_degree)
        if self.kind == "collocated":
            return max(trial, test)
        if coefficient_order is None or kernel_polynomial_degree is None:
            if self.kind != "overintegrated":
                raise ValueError(
                    "Polynomial exactness requires declared coefficient and kernel "
                    "degrees."
                )
            base_degree = trial + test + max(coordinate - 1, 0)
            if coefficient_order is not None:
                base_degree += int(coefficient_order)
            if kernel_polynomial_degree is not None:
                base_degree += int(kernel_polynomial_degree)
            return int(self.overintegration_factor * base_degree + 0.999999999)
        coefficient = int(coefficient_order)
        kernel = int(kernel_polynomial_degree)
        if coefficient < 0 or kernel < 0:
            raise ValueError("Coefficient/kernel polynomial degrees must be nonnegative.")
        exact_degree = trial + test + coefficient + kernel + max(coordinate - 1, 0)
        if self.kind == "overintegrated":
            return int(self.overintegration_factor * exact_degree + 0.999999999)
        return exact_degree


class QuadratureEvidence(StrictModule, NonTrainableState):
    role: QuadratureRole = eqx.field(static=True)
    requested_policy: str = eqx.field(static=True)
    selected_degree: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    aliasing_status: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        role: QuadratureRole,
        policy: QuadratureAccuracyPolicy,
        selected_degree: int,
        /,
        *,
        exact: bool,
        aliasing_status: str,
    ):
        if role not in (
            "volume",
            "interior-facet",
            "exterior-facet",
            "projection",
            "observation",
        ) or not isinstance(policy, QuadratureAccuracyPolicy):
            raise ValueError("Quadrature evidence role/policy is invalid.")
        degree = int(selected_degree)
        aliasing = str(aliasing_status)
        if degree < 0 or not aliasing:
            raise ValueError("Quadrature evidence degree/status are invalid.")
        self.role = role
        self.requested_policy = policy.kind
        self.selected_degree = degree
        self.exact = bool(exact)
        self.aliasing_status = aliasing
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "quadrature-evidence",
                "role": role,
                "policy": policy.policy_id,
                "selected_degree": degree,
                "exact": bool(exact),
                "aliasing_status": aliasing,
            }
        )


__all__ = [
    "QuadratureAccuracyKind",
    "QuadratureAccuracyPolicy",
    "QuadratureEvidence",
    "QuadratureRole",
]
