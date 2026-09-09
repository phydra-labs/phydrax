#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np

from .._fingerprint import canonical_fingerprint
from ._operators import AbstractLinearOperator, FunctionLinearOperator
from ._policies import FailurePolicy, GMRES, LinearSolvePolicy, TolerancePolicy
from ._preconditioners import AbstractPreconditioner, OperatorPreconditioner
from ._problems import LinearSystem
from ._properties import OperatorProperties
from ._runtime import solve


def _sum(left, right):
    return jax.tree.map(lambda a, b: a + b, left, right)


def hcurl_auxiliary_space_preconditioner(
    edge_inverse: AbstractPreconditioner,
    gradient: AbstractLinearOperator,
    scalar_inverse: AbstractPreconditioner,
    /,
) -> OperatorPreconditioner:
    """Additive edge smoother plus scalar-gradient auxiliary correction."""
    if not isinstance(edge_inverse, AbstractPreconditioner) or not isinstance(
        scalar_inverse, AbstractPreconditioner
    ):
        raise TypeError("H(curl) auxiliary actions must be prepared preconditioners.")
    if not isinstance(gradient, AbstractLinearOperator):
        raise TypeError("H(curl) gradient must be a native linear operator.")
    if not gradient.target.compatible(
        edge_inverse.space
    ) or not gradient.source.compatible(scalar_inverse.space):
        raise ValueError("H(curl) edge/scalar spaces and gradient do not compose.")

    def action(residual):
        edge = edge_inverse.apply(residual)
        scalar_residual = gradient.transpose_mv(residual)
        scalar = scalar_inverse.apply(scalar_residual)
        return _sum(edge, gradient.mv(scalar))

    positive = (
        edge_inverse.properties.positive_definite
        and scalar_inverse.properties.positive_definite
    )
    operator = FunctionLinearOperator(
        action,
        source=edge_inverse.space,
        target=edge_inverse.space,
        properties=OperatorProperties(
            self_adjoint=positive,
            positive_definite=positive,
            evidence={
                "self_adjoint": "transformed",
                "positive_definite": "transformed",
            }
            if positive
            else {},
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "hcurl-auxiliary-space-inverse",
                "edge": edge_inverse.preconditioner_id,
                "gradient": gradient.operator_id,
                "scalar": scalar_inverse.preconditioner_id,
            }
        ),
    )
    return OperatorPreconditioner(operator, positive_definite=positive)


def shifted_helmholtz_preconditioner(
    stiffness: AbstractLinearOperator,
    mass: AbstractLinearOperator,
    angular_frequency: float,
    /,
    *,
    damping_shift: float = 0.5,
    policy: LinearSolvePolicy | None = None,
) -> OperatorPreconditioner:
    """Native solve of K - (1+i beta) omega² M for Helmholtz/Maxwell."""
    if not isinstance(stiffness, AbstractLinearOperator) or not isinstance(
        mass, AbstractLinearOperator
    ):
        raise TypeError("Shifted Helmholtz components must be native operators.")
    if not (
        stiffness.source.compatible(stiffness.target)
        and mass.source.compatible(mass.target)
        and stiffness.source.compatible(mass.source)
    ):
        raise ValueError("Shifted Helmholtz stiffness and mass spaces must agree.")
    omega, beta = float(angular_frequency), float(damping_shift)
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(beta) or beta <= 0:
        raise ValueError(
            "Shifted Helmholtz frequency and damping must be positive finite."
        )
    selected = (
        LinearSolvePolicy(
            GMRES(restart=40, stagnation_iterations=40),
            tolerance=TolerancePolicy(relative=1e-2, absolute=0.0, max_steps=120),
            failure=FailurePolicy("status"),
        )
        if policy is None
        else policy
    )
    if not isinstance(selected, LinearSolvePolicy):
        raise TypeError("Shifted Helmholtz policy must be LinearSolvePolicy.")
    coefficient = -(1.0 + 1j * beta) * omega**2
    shifted = FunctionLinearOperator(
        lambda value: _sum(
            stiffness.mv(value),
            jax.tree.map(lambda item: coefficient * item, mass.mv(value)),
        ),
        source=stiffness.source,
        target=stiffness.target,
        operator_id=canonical_fingerprint(
            {
                "kind": "shifted-helmholtz-operator",
                "stiffness": stiffness.operator_id,
                "mass": mass.operator_id,
                "angular_frequency": omega,
                "damping_shift": beta,
            }
        ),
    )

    def inverse(residual):
        result = solve(LinearSystem(shifted), residual, policy=selected)
        return eqx.error_if(
            result.value,
            ~result.successful,
            "Shifted Helmholtz preconditioner inner solve failed.",
        )

    inverse_operator = FunctionLinearOperator(
        inverse,
        source=shifted.target,
        target=shifted.source,
        operator_id=canonical_fingerprint(
            {
                "kind": "shifted-helmholtz-inverse",
                "operator": shifted.operator_id,
            }
        ),
    )
    return OperatorPreconditioner(inverse_operator)


def porous_cpr_preconditioner(
    system_operator: AbstractLinearOperator,
    local_inverse: AbstractPreconditioner,
    pressure_restriction: AbstractLinearOperator,
    pressure_inverse: AbstractPreconditioner,
    /,
) -> OperatorPreconditioner:
    """Two-stage constrained-pressure-residual preconditioner."""
    if not isinstance(system_operator, AbstractLinearOperator):
        raise TypeError("CPR system action must be a native linear operator.")
    if not isinstance(local_inverse, AbstractPreconditioner) or not isinstance(
        pressure_inverse, AbstractPreconditioner
    ):
        raise TypeError("CPR actions must be prepared preconditioners.")
    if not isinstance(pressure_restriction, AbstractLinearOperator):
        raise TypeError("CPR pressure restriction must be a native operator.")
    if not (
        system_operator.source.compatible(system_operator.target)
        and system_operator.source.compatible(local_inverse.space)
        and pressure_restriction.source.compatible(local_inverse.space)
        and pressure_restriction.target.compatible(pressure_inverse.space)
    ):
        raise ValueError(
            "CPR system, restriction, local, and pressure spaces do not compose."
        )

    def action(residual):
        local = local_inverse.apply(residual)
        remaining = jax.tree.map(
            lambda value, applied: value - applied,
            residual,
            system_operator.mv(local),
        )
        pressure_residual = pressure_restriction.mv(remaining)
        pressure = pressure_inverse.apply(pressure_residual)
        return _sum(local, pressure_restriction.transpose_mv(pressure))

    operator = FunctionLinearOperator(
        action,
        source=local_inverse.space,
        target=local_inverse.space,
        operator_id=canonical_fingerprint(
            {
                "kind": "porous-cpr-inverse",
                "system": system_operator.operator_id,
                "local": local_inverse.preconditioner_id,
                "restriction": pressure_restriction.operator_id,
                "pressure": pressure_inverse.preconditioner_id,
            }
        ),
    )
    return OperatorPreconditioner(operator)


__all__ = [
    "hcurl_auxiliary_space_preconditioner",
    "porous_cpr_preconditioner",
    "shifted_helmholtz_preconditioner",
]
