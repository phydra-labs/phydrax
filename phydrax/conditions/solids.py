#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

from jaxtyping import ArrayLike

from phydrax.domain import DomainComponent, DomainFunction

from ..operators.differential import cauchy_stress
from ._base import Residual
from ._field_ops import dot, matvec, outer_scalar_vector
from .boundary import _condition_value, ConditionValue


def Traction(
    displacement_field: str,
    on: DomainComponent,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    traction: ConditionValue | None = None,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Prescribed Cauchy traction ``stress(displacement) normal = traction``."""
    normal = on.normal(var=var)
    target = _condition_value(traction, on, 0.0)

    def residual(displacement: DomainFunction, /) -> DomainFunction:
        stress = cauchy_stress(
            displacement,
            lambda_=lambda_,
            mu=mu,
            var=var,
            mode=mode,
        )
        return matvec(stress, normal) - target

    return Residual(displacement_field, on, residual, label=label)


def NormalDisplacement(
    displacement_field: str,
    on: DomainComponent,
    /,
    *,
    target: ConditionValue | None = None,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Prescribed normal displacement ``displacement · normal = target``."""
    normal = on.normal(var=var)
    target_value = _condition_value(target, on, 0.0)
    return Residual(
        displacement_field,
        on,
        lambda displacement: dot(displacement, normal) - target_value,
        label=label,
    )


def ElasticFoundation(
    displacement_field: str,
    on: DomainComponent,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    stiffness: DomainFunction | ArrayLike,
    foundation_displacement: ConditionValue | None = None,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Spring foundation relation ``traction + stiffness(displacement-u₀) = 0``."""
    normal = on.normal(var=var)
    reference = _condition_value(foundation_displacement, on, 0.0)

    def residual(displacement: DomainFunction, /) -> DomainFunction:
        stress = cauchy_stress(
            displacement,
            lambda_=lambda_,
            mu=mu,
            var=var,
            mode=mode,
        )
        traction = matvec(stress, normal)
        return traction + stiffness * (displacement - reference)

    return Residual(displacement_field, on, residual, label=label)


def ElasticSymmetry(
    displacement_field: str,
    on: DomainComponent,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Zero normal displacement and zero tangential traction."""
    normal = on.normal(var=var)

    def residual(displacement: DomainFunction, /) -> DomainFunction:
        stress = cauchy_stress(
            displacement,
            lambda_=lambda_,
            mu=mu,
            var=var,
            mode=mode,
        )
        traction = matvec(stress, normal)
        normal_displacement = outer_scalar_vector(
            dot(displacement, normal),
            normal,
        )
        tangential_traction = traction - outer_scalar_vector(
            dot(traction, normal),
            normal,
        )
        return normal_displacement + tangential_traction

    return Residual(displacement_field, on, residual, label=label)


__all__ = [
    "ElasticFoundation",
    "ElasticSymmetry",
    "NormalDisplacement",
    "Traction",
]
