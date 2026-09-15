#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral numerical choices for electronic calculations."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class IntegralRepresentationKind(StrEnum):
    PROVIDER_NATIVE = "provider-native"
    DENSE = "dense"
    DIRECT = "direct"
    DENSITY_FITTING = "density-fitting"
    CHOLESKY = "cholesky"
    TENSOR_HYPERCONTRACTION = "tensor-hypercontraction"
    FFT_DENSITY_FITTING = "fft-density-fitting"
    GAUSSIAN_DENSITY_FITTING = "gaussian-density-fitting"


class ElectronicStationarySolverKind(StrEnum):
    PROVIDER_NATIVE = "provider-native"
    FIXED_POINT = "fixed-point"
    DIIS = "diis"
    ENERGY_DIIS = "energy-diis"
    AUGMENTED_DIIS = "augmented-diis"
    NEWTON_KRYLOV = "newton-krylov"
    CIAH = "ciah"


class ElectronicDerivativeRoute(StrEnum):
    PROVIDER_NATIVE = "provider-native"
    ANALYTIC_LAGRANGIAN = "analytic-lagrangian"
    IMPLICIT_RESPONSE = "implicit-response"
    CENTRAL_FINITE_DIFFERENCE = "central-finite-difference"


class ElectronicNumericalPlan(StrictModule, NonTrainableState):
    """Exact numerical representation without changing physical model chemistry."""

    integrals: IntegralRepresentationKind = eqx.field(static=True)
    stationary_solver: ElectronicStationarySolverKind = eqx.field(static=True)
    derivative_route: ElectronicDerivativeRoute = eqx.field(static=True)
    response_solver: ElectronicStationarySolverKind = eqx.field(static=True)
    component_plan_ids: tuple[str, ...] = eqx.field(static=True)
    numerical_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        integrals: IntegralRepresentationKind = IntegralRepresentationKind.PROVIDER_NATIVE,
        stationary_solver: ElectronicStationarySolverKind = ElectronicStationarySolverKind.PROVIDER_NATIVE,
        derivative_route: ElectronicDerivativeRoute = ElectronicDerivativeRoute.PROVIDER_NATIVE,
        response_solver: ElectronicStationarySolverKind = ElectronicStationarySolverKind.PROVIDER_NATIVE,
        component_plan_ids: Sequence[str] = (),
    ):
        if not isinstance(integrals, IntegralRepresentationKind):
            raise TypeError("integrals must be IntegralRepresentationKind.")
        if not isinstance(stationary_solver, ElectronicStationarySolverKind):
            raise TypeError("stationary_solver must be ElectronicStationarySolverKind.")
        if not isinstance(derivative_route, ElectronicDerivativeRoute):
            raise TypeError("derivative_route must be ElectronicDerivativeRoute.")
        if not isinstance(response_solver, ElectronicStationarySolverKind):
            raise TypeError("response_solver must be ElectronicStationarySolverKind.")
        components = tuple(sorted(str(value).strip() for value in component_plan_ids))
        if any(not value for value in components) or len(set(components)) != len(
            components
        ):
            raise ValueError("component_plan_ids must be unique non-empty identifiers.")
        self.integrals = integrals
        self.stationary_solver = stationary_solver
        self.derivative_route = derivative_route
        self.response_solver = response_solver
        self.component_plan_ids = components
        self.numerical_id = canonical_fingerprint(
            {
                "kind": "electronic-numerical-plan",
                "integrals": integrals.value,
                "stationary_solver": stationary_solver.value,
                "derivative_route": derivative_route.value,
                "response_solver": response_solver.value,
                "components": list(components),
            }
        )


__all__ = [
    "ElectronicDerivativeRoute",
    "ElectronicNumericalPlan",
    "ElectronicStationarySolverKind",
    "IntegralRepresentationKind",
]
