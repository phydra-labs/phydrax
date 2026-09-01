#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...discretization.fem import (
    FiniteElementDiscretization,
    HDGCondensationPlan,
    HDGTraceSpace,
    IntegrationDomain,
)
from ...discretization.fem.smoothing import SmoothedElasticityOperator
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    LinearSystem,
    OperatorProperties,
    solve,
)
from .._finite_element_variational import (
    _interval_rule,
    _reference_rule_data,
    CellResidualAction,
    coefficient,
    DiffusionAction,
    ExteriorFacetAction,
    FiniteElementForm,
    InteriorFacetAction,
    PreparedOperatorAction,
    SIPGBoundaryCondition,
    SIPGFacetAction,
    SIPGPenaltyPolicy,
    SourceAction,
)
from .._variational import VariationalCoefficient
from ._operators import curl, symmetric_gradient


def linear_elasticity_form(
    field_name: str,
    lame_lambda: ArrayLike,
    shear_modulus: ArrayLike,
    /,
    *,
    form_id: str = "linear-elasticity",
) -> FiniteElementForm:
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

    return FiniteElementForm(
        form_id,
        field_name,
        (
            CellResidualAction(
                field_name,
                (field_name,),
                kernel,
                action_id="elasticity",
            ),
        ),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )


def upwind_advection_form(
    field_name: str,
    velocity: ArrayLike,
    /,
    *,
    interior_domain: IntegrationDomain | None = None,
    boundary_domain: IntegrationDomain | None = None,
    inflow: VariationalCoefficient | ArrayLike | Callable = 0.0,
    inflow_coefficient_id: str | None = None,
    source: ArrayLike | Callable | None = None,
    form_id: str = "upwind-advection",
) -> FiniteElementForm:
    velocity_ = jnp.asarray(velocity)
    if velocity_.ndim != 1 or velocity_.size not in (2, 3):
        raise ValueError("Upwind velocity must be a two- or three-vector.")
    if interior_domain is not None and interior_domain.kind != "interior_facet":
        raise ValueError("interior_domain must select interior facets.")
    if boundary_domain is not None and boundary_domain.kind != "exterior_facet":
        raise ValueError("boundary_domain must select exterior facets.")
    inflow_ = (
        inflow
        if isinstance(inflow, VariationalCoefficient)
        else coefficient(inflow, coefficient_id=inflow_coefficient_id)
    )

    def volume(values, gradients, points, weights, test_basis, test_gradients, context):
        value = values[0]
        directional_test_gradient = oe.contract(
            "cqid,d->cqi",
            test_gradients,
            velocity_,
        )
        return -oe.contract(
            "cq,cqi,cq->ci",
            weights,
            directional_test_gradient,
            value,
        )

    def flux(plus_values, minus_values, points, weights, normal, context):
        plus = plus_values[0]
        minus = minus_values[0]
        speed = jnp.sum(velocity_ * normal, axis=-1)
        factor = speed.reshape(speed.shape + (1,) * (plus.ndim - speed.ndim))
        selected = jnp.where(factor >= 0.0, plus, minus)
        numerical = factor * selected
        return numerical, -numerical

    def boundary_flux(plus_values, points, weights, normal, context):
        plus = plus_values[0]
        speed = jnp.sum(velocity_ * normal, axis=-1)
        factor = speed.reshape(speed.shape + (1,) * (plus.ndim - speed.ndim))
        incoming = inflow_.evaluate(points, context)
        incoming = jnp.broadcast_to(incoming, plus.shape)
        trace = jnp.where(factor >= 0.0, plus, incoming)
        return factor * trace

    terms = [
        CellResidualAction(
            field_name,
            (field_name,),
            volume,
            action_id="upwind-volume",
        ),
        InteriorFacetAction(
            field_name,
            (field_name,),
            flux,
            domain=interior_domain,
            action_id="upwind-interior-flux",
        ),
    ]
    if boundary_domain is not None:
        terms.append(
            ExteriorFacetAction(
                field_name,
                (field_name,),
                boundary_flux,
                domain=boundary_domain,
                action_id="upwind-boundary-flux",
            )
        )
    if source is not None:
        terms.append(
            SourceAction(
                field_name,
                source,
                action_id="upwind-source",
            )
        )
    return FiniteElementForm(form_id, field_name, tuple(terms))


def darcy_form(
    flux_field: str,
    pressure_field: str,
    inverse_permeability: ArrayLike = 1.0,
    /,
    *,
    form_id: str = "mixed-darcy",
) -> FiniteElementForm:
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

    return FiniteElementForm(
        form_id,
        (flux_field, pressure_field),
        (
            CellResidualAction(
                flux_field,
                (flux_field, pressure_field),
                flux_residual,
                action_id="darcy-flux",
            ),
            CellResidualAction(
                pressure_field,
                (flux_field,),
                pressure_residual,
                action_id="darcy-mass-balance",
            ),
        ),
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
    )


def maxwell_form(
    field_name: str,
    mass_coefficient: ArrayLike = 1.0,
    curl_coefficient: ArrayLike = 1.0,
    /,
    *,
    form_id: str = "maxwell-curl-curl",
) -> FiniteElementForm:
    mass = jnp.asarray(mass_coefficient)
    curl_weight = jnp.asarray(curl_coefficient)

    def residual(values, gradients, points, weights, test_basis, test_gradients, context):
        value = values[0]
        field_curl = curl(gradients[0])
        test_curl = curl(test_gradients)
        return mass * oe.contract(
            "cq,cqiv,cqv->ci", weights, test_basis, value
        ) + curl_weight * oe.contract("cq,cqi,cq->ci", weights, test_curl, field_curl)

    return FiniteElementForm(
        form_id,
        field_name,
        (
            CellResidualAction(
                field_name,
                (field_name,),
                residual,
                action_id="maxwell",
            ),
        ),
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
    )


def stokes_form(
    velocity_field: str,
    pressure_field: str,
    viscosity: ArrayLike = 1.0,
    /,
    *,
    form_id: str = "taylor-hood-stokes",
) -> FiniteElementForm:
    viscosity_ = jnp.asarray(viscosity)

    def momentum(values, gradients, points, weights, test_basis, test_gradients, context):
        velocity_gradient, _ = gradients
        _, pressure = values
        strain = symmetric_gradient(velocity_gradient)
        viscous = (
            2.0
            * viscosity_
            * oe.contract(
                "cq,cqib,cqab->cia",
                weights,
                test_gradients,
                strain,
            )
        )
        pressure_term = oe.contract(
            "cq,cqia,cq->cia",
            weights,
            test_gradients,
            pressure,
        )
        return viscous - pressure_term

    def incompressibility(
        values, gradients, points, weights, test_basis, test_gradients, context
    ):
        velocity_gradient = gradients[0]
        divergence = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        return oe.contract("cq,qi,cq->ci", weights, test_basis, divergence)

    return FiniteElementForm(
        form_id,
        (velocity_field, pressure_field),
        (
            CellResidualAction(
                velocity_field,
                (velocity_field, pressure_field),
                momentum,
                action_id="stokes-momentum",
            ),
            CellResidualAction(
                pressure_field,
                (velocity_field,),
                incompressibility,
                action_id="stokes-incompressibility",
            ),
        ),
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
    )


def sipg_dirichlet(
    domain: IntegrationDomain,
    value: ArrayLike | Callable,
    /,
    *,
    penalty_policy: SIPGPenaltyPolicy | None = None,
) -> SIPGBoundaryCondition:
    return SIPGBoundaryCondition(
        "dirichlet",
        domain,
        value,
        penalty_policy=penalty_policy,
    )


def sipg_neumann(
    domain: IntegrationDomain,
    flux: ArrayLike | Callable,
    /,
) -> SIPGBoundaryCondition:
    return SIPGBoundaryCondition("neumann", domain, flux)


def sipg_robin(
    domain: IntegrationDomain,
    coefficient: ArrayLike | Callable,
    value: ArrayLike | Callable,
    /,
) -> SIPGBoundaryCondition:
    return SIPGBoundaryCondition(
        "robin",
        domain,
        value,
        robin_coefficient=coefficient,
    )


def sipg_poisson_form(
    field_name: str,
    diffusivity: ArrayLike | Callable,
    penalty_policy: SIPGPenaltyPolicy,
    cell_domain: IntegrationDomain | None,
    interior_domain: IntegrationDomain,
    boundary_terms: Sequence[SIPGBoundaryCondition],
    /,
    *,
    source: ArrayLike | Callable | None = None,
    form_id: str = "sipg-poisson",
) -> FiniteElementForm:
    """Build executable symmetric interior-penalty Poisson actions."""

    if cell_domain is not None and cell_domain.kind != "cell":
        raise ValueError("SIPG cell_domain must be a cell integration domain.")
    if (
        not isinstance(interior_domain, IntegrationDomain)
        or interior_domain.kind != "interior_facet"
    ):
        raise ValueError("SIPG interior_domain must be an interior-facet domain.")
    if not isinstance(penalty_policy, SIPGPenaltyPolicy):
        raise TypeError("penalty_policy must be SIPGPenaltyPolicy.")
    boundaries = tuple(boundary_terms)
    if any(not isinstance(value, SIPGBoundaryCondition) for value in boundaries):
        raise TypeError("boundary_terms must contain SIPGBoundaryCondition values.")
    domain_ids = tuple(value.domain.domain_id for value in boundaries)
    if len(set(domain_ids)) != len(domain_ids):
        raise ValueError("SIPG boundary domains must be unique.")
    terms = [
        DiffusionAction(
            field_name,
            diffusivity,
            domain=cell_domain,
            action_id="sipg-cell-diffusion",
        ),
        SIPGFacetAction(
            field_name,
            diffusivity,
            penalty_policy,
            interior_domain,
            action_id="sipg-interior",
        ),
    ]
    if source is not None:
        terms.append(
            SourceAction(
                field_name,
                source,
                domain=cell_domain,
                action_id="sipg-source",
            )
        )
    terms.extend(
        SIPGFacetAction(
            field_name,
            diffusivity,
            penalty_policy,
            boundary.domain,
            boundary=boundary,
            action_id=f"sipg-boundary:{index}:{boundary.kind}",
        )
        for index, boundary in enumerate(boundaries)
    )
    return FiniteElementForm(
        form_id,
        field_name,
        tuple(terms),
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
    )


def smoothed_elasticity_form(
    field_name: str,
    operator: SmoothedElasticityOperator,
    /,
    *,
    form_id: str = "smoothed-elasticity",
) -> FiniteElementForm:
    """Bind one prepared smoothed-elasticity action to the FEM program."""

    if not isinstance(operator, SmoothedElasticityOperator):
        raise TypeError("operator must be SmoothedElasticityOperator.")
    return FiniteElementForm(
        form_id,
        field_name,
        (
            PreparedOperatorAction(
                field_name,
                operator.as_linear_operator(),
                action_id="smoothed-elasticity-action",
            ),
        ),
    )


class HDGPoissonSolution(StrictModule):
    trace: Array
    local: Array
    successful: Array


def _solve_hdg_local_system(
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


def solve_hdg_poisson(
    discretization: FiniteElementDiscretization,
    field_name: str,
    source: ArrayLike,
    boundary_values: ArrayLike | Callable[[Array], ArrayLike],
    /,
    *,
    diffusivity: ArrayLike = 1.0,
    penalty_factor: float = 12.0,
) -> HDGPoissonSolution:
    """Assemble, condense, solve, and reconstruct lowest-order primal HDG."""

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    if len(discretization.mesh.blocks) != 1:
        raise ValueError("HDG Poisson currently requires one homogeneous block.")
    block = discretization.mesh.blocks[0]
    if block.cell_kind != "triangle":
        raise ValueError("HDG Poisson currently supports triangular cells.")
    field_index = discretization._field_index(field_name)
    element = discretization.elements[field_index][0]
    dof_map = discretization.dof_maps[field_index]
    if (
        element.conformity != "L2"
        or element.local_dof_count != 3
        or dof_map.association != "cell"
    ):
        raise ValueError("HDG Poisson requires a discontinuous triangle P1 field.")
    kappa = jnp.asarray(diffusivity)
    forcing = jnp.asarray(source)
    penalty = float(penalty_factor)
    if (
        kappa.shape != ()
        or forcing.shape != ()
        or not bool(jnp.isfinite(kappa))
        or not bool(jnp.isfinite(forcing))
        or float(kappa) <= 0.0
        or not np.isfinite(penalty)
        or penalty <= 0.0
    ):
        raise ValueError("HDG coefficients and penalty must be finite and admissible.")
    geometry = discretization.block_geometries[field_index][0]
    basis = geometry.basis_values
    gradients = geometry.physical_gradients
    weights = geometry.physical_weights
    cell_count = block.cell_count
    local_field_size = element.local_dof_count
    trace_space = HDGTraceSpace(discretization.mesh)
    local_trace_size = int(trace_space.cell_trace_dofs.shape[1])
    plan = HDGCondensationPlan(trace_space, local_field_size)
    local_size = local_field_size + local_trace_size
    matrices = jnp.zeros((cell_count, local_size, local_size), dtype=weights.dtype)
    right_hand_side = jnp.zeros((cell_count, local_size), dtype=weights.dtype)
    stiffness = kappa * oe.contract(
        "cq,cqid,cqjd->cij",
        weights,
        gradients,
        gradients,
    )
    load = forcing * oe.contract("cq,qi->ci", weights, basis)
    matrices = matrices.at[:, :local_field_size, :local_field_size].set(stiffness)
    right_hand_side = right_hand_side.at[:, :local_field_size].set(load)
    interval_data = _reference_rule_data(_interval_rule())
    parameter = interval_data.points[:, 0]
    cell_vertices = context_coordinates = discretization.default_runtime.coordinates[
        block.vertices
    ]
    cell_centers = jnp.mean(cell_vertices, axis=1)
    for local_facet in range(3):
        side_points = (
            jnp.stack((parameter, jnp.zeros_like(parameter)), axis=-1)
            if local_facet == 0
            else (
                jnp.stack((1.0 - parameter, parameter), axis=-1)
                if local_facet == 1
                else jnp.stack((jnp.zeros_like(parameter), 1.0 - parameter), axis=-1)
            )
        )
        side_geometry = discretization.evaluate_block_geometry(
            field_name,
            0,
            discretization.default_runtime.coordinates,
            side_points,
            jnp.ones_like(interval_data.weights),
        )
        side_basis = side_geometry.basis_values
        side_gradients = side_geometry.physical_gradients
        start = context_coordinates[:, local_facet]
        stop = context_coordinates[:, (local_facet + 1) % 3]
        tangent = stop - start
        length = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
        normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        normal = normal / length[:, None]
        midpoint = 0.5 * (start + stop)
        outward = jnp.sum(normal * (midpoint - cell_centers), axis=-1)
        normal = jnp.where((outward < 0.0)[:, None], -normal, normal)
        side_weights = length[:, None] * interval_data.weights[None, :]
        normal_gradient = jnp.sum(
            side_gradients * normal[:, None, None, :],
            axis=-1,
        )
        height = 2.0 * geometry.measure / length
        tau = penalty * element.degree**2 * kappa / height
        uu = (
            -kappa
            * oe.contract(
                "cq,qi,cqj->cij",
                side_weights,
                side_basis,
                normal_gradient,
            )
            - kappa
            * oe.contract(
                "cq,cqi,qj->cij",
                side_weights,
                normal_gradient,
                side_basis,
            )
            + oe.contract(
                "cq,c,qi,qj->cij",
                side_weights,
                tau,
                side_basis,
                side_basis,
            )
        )
        coupling = kappa * oe.contract(
            "cq,cqi->ci", side_weights, normal_gradient
        ) - oe.contract(
            "cq,c,qi->ci",
            side_weights,
            tau,
            side_basis,
        )
        trace_index = local_field_size + local_facet
        matrices = matrices.at[:, :local_field_size, :local_field_size].add(uu)
        matrices = matrices.at[:, :local_field_size, trace_index].add(coupling)
        matrices = matrices.at[:, trace_index, :local_field_size].add(coupling)
        matrices = matrices.at[:, trace_index, trace_index].add(
            tau * jnp.sum(side_weights, axis=1)
        )
    connectivity = discretization.mesh.connectivity
    boundary_mask = connectivity.boundary_edges
    edge_vertices = jnp.asarray(connectivity.edges, dtype=jnp.int32)
    edge_midpoints = jnp.mean(
        discretization.default_runtime.coordinates[edge_vertices],
        axis=1,
    )
    if callable(boundary_values):
        evaluator = cast(Callable[[Array], ArrayLike], boundary_values)
        prescribed = jnp.asarray(evaluator(edge_midpoints))
    else:
        prescribed = jnp.asarray(boundary_values)
    prescribed = jnp.broadcast_to(prescribed, (trace_space.trace_dof_count,))
    return _solve_hdg_local_system(
        plan,
        matrices,
        right_hand_side,
        boundary_mask,
        prescribed,
    )


__all__ = [
    "HDGPoissonSolution",
    "SIPGBoundaryCondition",
    "SIPGPenaltyPolicy",
    "darcy_form",
    "linear_elasticity_form",
    "maxwell_form",
    "sipg_dirichlet",
    "sipg_neumann",
    "sipg_poisson_form",
    "sipg_robin",
    "smoothed_elasticity_form",
    "solve_hdg_poisson",
    "stokes_form",
    "upwind_advection_form",
]
