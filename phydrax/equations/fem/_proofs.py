#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...discretization.fem import HDGCondensationPlan
from ...linalg import ArraySpace, DenseLinearOperator, LinearSystem, solve
from .._finite_element_variational import (
    CellResidualTerm,
    InteriorFacetTerm,
    WeakForm,
)
from ._ir import FieldSlot, LocalActionIR, LocalActionTermIR, RegionIR
from ._operators import curl, symmetric_gradient


def linear_elasticity_form(
    field_name: str,
    lame_lambda: ArrayLike,
    shear_modulus: ArrayLike,
    /,
    *,
    form_id: str = "linear-elasticity",
) -> WeakForm:
    lambda_ = jnp.asarray(lame_lambda)
    mu = jnp.asarray(shear_modulus)

    def kernel(values, gradients, points, weights, test_basis, test_gradients, context):
        gradient = gradients[0]
        strain = symmetric_gradient(gradient)
        trace = jnp.trace(strain, axis1=-2, axis2=-1)
        identity = jnp.eye(strain.shape[-1], dtype=strain.dtype)
        stress = lambda_ * trace[..., None, None] * identity + 2.0 * mu * strain
        return oe.contract(
            "cq,cqid,cqad->cia",
            weights,
            test_gradients,
            stress,
        )

    return WeakForm(
        form_id,
        field_name,
        (
            CellResidualTerm(
                field_name,
                (field_name,),
                kernel,
                term_id="elasticity",
            ),
        ),
    )


def upwind_advection_form(
    field_name: str,
    velocity: ArrayLike,
    /,
    *,
    form_id: str = "upwind-advection",
) -> WeakForm:
    velocity_ = jnp.asarray(velocity)

    def flux(plus, minus, points, weights, normal, context):
        speed = jnp.sum(velocity_ * normal, axis=-1)
        speed = speed[:, None]
        selected = jnp.where(speed[..., None] >= 0.0, plus, minus)
        numerical = speed[..., None] * selected
        return numerical, -numerical

    return WeakForm(
        form_id,
        field_name,
        (
            InteriorFacetTerm(
                field_name,
                flux,
                term_id="upwind-flux",
            ),
        ),
    )


def interior_penalty_form(
    field_name: str,
    penalty: ArrayLike,
    /,
    *,
    form_id: str = "interior-penalty",
) -> WeakForm:
    penalty_ = jnp.asarray(penalty)

    def flux(plus, minus, points, weights, normal, context):
        jump = plus - minus
        return penalty_ * jump, -penalty_ * jump

    return WeakForm(
        form_id,
        field_name,
        (
            InteriorFacetTerm(
                field_name,
                flux,
                term_id="penalty-flux",
            ),
        ),
    )


def darcy_form(
    flux_field: str,
    pressure_field: str,
    inverse_permeability: ArrayLike = 1.0,
    /,
    *,
    form_id: str = "mixed-darcy",
) -> WeakForm:
    inverse = jnp.asarray(inverse_permeability)

    def flux_residual(
        values, gradients, points, weights, test_basis, test_gradients, context
    ):
        flux_value, pressure_value = values
        div_test = jnp.trace(test_gradients, axis1=-2, axis2=-1)
        return oe.contract(
            "cq,cqid,cqd->ci",
            weights,
            test_basis,
            inverse * flux_value,
        ) - oe.contract("cq,cqi,cq->ci", weights, div_test, pressure_value)

    def pressure_residual(
        values, gradients, points, weights, test_basis, test_gradients, context
    ):
        flux_gradient = gradients[0]
        div_flux = jnp.trace(flux_gradient, axis1=-2, axis2=-1)
        return oe.contract("cq,qi,cq->ci", weights, test_basis, div_flux)

    return WeakForm(
        form_id,
        (flux_field, pressure_field),
        (
            CellResidualTerm(
                flux_field,
                (flux_field, pressure_field),
                flux_residual,
                term_id="darcy-flux",
            ),
            CellResidualTerm(
                pressure_field,
                (flux_field, pressure_field),
                pressure_residual,
                term_id="darcy-mass-balance",
            ),
        ),
    )


def maxwell_form(
    field_name: str,
    mass_coefficient: ArrayLike = 1.0,
    curl_coefficient: ArrayLike = 1.0,
    /,
    *,
    form_id: str = "maxwell-curl-curl",
) -> WeakForm:
    mass = jnp.asarray(mass_coefficient)
    curl_weight = jnp.asarray(curl_coefficient)

    def residual(values, gradients, points, weights, test_basis, test_gradients, context):
        value = values[0]
        field_curl = curl(gradients[0])
        test_curl = curl(test_gradients)
        return mass * oe.contract(
            "cq,cqiv,cqv->ci", weights, test_basis, value
        ) + curl_weight * oe.contract("cq,cqi,cq->ci", weights, test_curl, field_curl)

    return WeakForm(
        form_id,
        field_name,
        (
            CellResidualTerm(
                field_name,
                (field_name,),
                residual,
                term_id="maxwell",
            ),
        ),
    )


def sipg_poisson_ir(
    field_name: str,
    space_id: str,
    cell_domain_id: str,
    facet_domain_id: str,
    cell_rule_ids,
    facet_rule_ids,
    /,
) -> LocalActionIR:
    """Typed SIPG cell and consistency/symmetry/penalty facet semantics."""

    slot = FieldSlot(field_name, "unknown", space_id)
    cell_region = RegionIR("cell", cell_domain_id, cell_rule_ids)
    facet_region = RegionIR("interior-facet", facet_domain_id, facet_rule_ids)
    cell = LocalActionTermIR(
        "bilinear",
        field_name,
        (field_name,),
        ((field_name, "grad"),),
        cell_region,
        "sipg-cell-diffusion",
    )
    facet = LocalActionTermIR(
        "bilinear",
        field_name,
        (field_name,),
        (
            (field_name, "jump"),
            (field_name, "average"),
            (field_name, "grad"),
            (field_name, "normal-trace"),
        ),
        facet_region,
        "sipg-consistency-symmetry-penalty",
    )
    return LocalActionIR((slot,), (cell, facet))


class HDGPoissonSolution(StrictModule):
    trace: Array
    local: Array
    successful: Array


def solve_hdg_poisson(
    plan: HDGCondensationPlan,
    local_matrix: ArrayLike,
    local_rhs: ArrayLike,
    boundary_mask: ArrayLike,
    boundary_values: ArrayLike,
    /,
) -> HDGPoissonSolution:
    """Condense, assemble, constrain, solve, and reconstruct an HDG local system."""

    if not isinstance(plan, HDGCondensationPlan):
        raise TypeError("plan must be HDGCondensationPlan.")
    condensed = plan.condense(local_matrix, local_rhs)
    routes = plan.trace_space.cell_trace_dofs
    valid = plan.trace_space.trace_valid
    count = plan.trace_space.trace_dof_count
    matrix = jnp.zeros((count, count), dtype=condensed.schur.dtype)
    right_hand_side = jnp.zeros((count,), dtype=condensed.right_hand_side.dtype)
    for cell in range(routes.shape[0]):
        safe = jnp.where(valid[cell], routes[cell], 0)
        pair_valid = valid[cell][:, None] & valid[cell][None, :]
        matrix = matrix.at[safe[:, None], safe[None, :]].add(
            jnp.where(pair_valid, condensed.schur[cell], 0.0)
        )
        right_hand_side = right_hand_side.at[safe].add(
            jnp.where(valid[cell], condensed.right_hand_side[cell], 0.0)
        )
    boundary = jnp.asarray(boundary_mask, dtype=bool)
    prescribed = jnp.asarray(boundary_values)
    if boundary.shape != (count,) or prescribed.shape != (count,):
        raise ValueError("HDG boundary masks/values must match trace coordinates.")
    free = jnp.flatnonzero(~boundary, size=count, fill_value=0)
    free_count = int(jnp.sum(~boundary))
    if free_count == 0:
        raise ValueError("HDG trace solve requires free coordinates.")
    free = free[:free_count]
    reduced_matrix = matrix[free[:, None], free[None, :]]
    reduced_rhs = right_hand_side[free] - matrix[free] @ prescribed
    space = ArraySpace((free_count,), dtype=reduced_matrix.dtype)
    operator = DenseLinearOperator(
        reduced_matrix,
        source=space,
        target=space,
        operator_id=f"hdg:{plan.plan_id}",
    )
    result = solve(LinearSystem(operator), reduced_rhs)
    trace = prescribed.at[free].set(result.value, unique_indices=True)
    local_trace = trace[routes]
    local_solution = plan.reconstruct(local_trace, condensed)
    return HDGPoissonSolution(
        trace=trace,
        local=local_solution,
        successful=result.successful,
    )


__all__ = [
    "HDGPoissonSolution",
    "solve_hdg_poisson",
    "sipg_poisson_ir",
    "darcy_form",
    "interior_penalty_form",
    "linear_elasticity_form",
    "maxwell_form",
    "upwind_advection_form",
]
