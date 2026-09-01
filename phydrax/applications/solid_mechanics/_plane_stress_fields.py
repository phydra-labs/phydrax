#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Callable, Literal

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._strict import StrictModule
from ...domain import DomainFunction
from ...equations import CellEnergyAction, FiniteElementForm
from ...operators import deformation_gradient
from ...operators.mechanics import HyperelasticLaw
from ._plane_stress import BlockDiagonalPlaneStressReductionPlan


class PlaneStressFieldResponse(StrictModule):
    """Field-valued plane-stress observables and pointwise closure evidence."""

    energy: DomainFunction
    first_piola: DomainFunction
    cauchy: DomainFunction
    thickness_stretch: DomainFunction
    residual: DomainFunction
    successful: DomainFunction
    failure: DomainFunction


def plane_stress_hyperelastic_response(
    u: DomainFunction,
    law: HyperelasticLaw,
    plan: BlockDiagonalPlaneStressReductionPlan,
    /,
    *,
    reference_thickness: ArrayLike = 1.0,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> PlaneStressFieldResponse:
    """Lift one block-diagonal plane-stress law to displacement fields."""
    if not isinstance(u, DomainFunction):
        raise TypeError("u must be DomainFunction.")
    if not isinstance(law, HyperelasticLaw):
        raise TypeError("law must be HyperelasticLaw.")
    if not isinstance(plan, BlockDiagonalPlaneStressReductionPlan):
        raise TypeError("plan must be BlockDiagonalPlaneStressReductionPlan.")
    deformation = deformation_gradient(u, var=var, mode=mode)

    def select(selector: Callable):
        def operation(*args, key=None, **kwargs):
            value = jnp.asarray(deformation.func(*args, key=key, **kwargs))
            response = plan.evaluate(
                value,
                law,
                reference_thickness=reference_thickness,
            )
            return selector(response)

        return DomainFunction(
            domain=deformation.domain,
            deps=deformation.deps,
            func=operation,
            metadata=u.metadata,
        )

    return PlaneStressFieldResponse(
        select(lambda response: response.reference_energy_density),
        select(lambda response: response.first_piola),
        select(lambda response: response.cauchy_stress),
        select(lambda response: response.kinematics.thickness_stretch),
        select(lambda response: response.residual),
        select(lambda response: response.successful),
        select(lambda response: response.failure),
    )


def plane_stress_hyperelastic_form(
    field_name: str,
    law: HyperelasticLaw,
    plan: BlockDiagonalPlaneStressReductionPlan,
    /,
    *,
    reference_thickness: ArrayLike = 1.0,
    form_id: str = "plane-stress-hyperelastic",
) -> FiniteElementForm:
    """Create a two-dimensional FE cell energy with local plane-stress closure."""
    if not isinstance(law, HyperelasticLaw):
        raise TypeError("law must be HyperelasticLaw.")
    if not isinstance(plan, BlockDiagonalPlaneStressReductionPlan):
        raise TypeError("plan must be BlockDiagonalPlaneStressReductionPlan.")

    def density(values, gradients, points, context):
        del values, points, context
        displacement_gradient = jnp.swapaxes(jnp.asarray(gradients), -1, -2)
        if displacement_gradient.shape[-2:] != (2, 2):
            raise ValueError("Plane-stress FE displacement gradients must end in 2x2.")
        deformation = (
            jnp.eye(2, dtype=displacement_gradient.dtype) + displacement_gradient
        )
        return plan.evaluate(
            deformation,
            law,
            reference_thickness=reference_thickness,
        ).reference_energy_density

    return FiniteElementForm(
        form_id,
        field_name,
        (
            CellEnergyAction(
                field_name,
                density,
                action_id="plane-stress-hyperelastic-energy",
            ),
        ),
    )


__all__ = [
    "PlaneStressFieldResponse",
    "plane_stress_hyperelastic_form",
    "plane_stress_hyperelastic_response",
]
