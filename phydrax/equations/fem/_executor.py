#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._numerics._compensated import compensated_sum
from ...discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ...discretization.fem import FiniteElementDiscretization, IntegrationDomain
from ...linalg import DualSpace
from .._finite_element_variational import (
    _action_domain,
    _action_rule,
    _interval_rule,
    _reference_rule_data,
    _ResolvedCoefficient,
    _triangle_rule,
    BoundaryLoadAction,
    CellBilinearAction,
    CellEnergyAction,
    CellResidualAction,
    DiffusionAction,
    ExteriorFacetAction,
    FiniteElementExecutionContext,
    FiniteElementForm,
    InteriorFacetAction,
    MassAction,
    PreparedOperatorAction,
    SIPGFacetAction,
    SourceAction,
)
from ._ir import LocalActionIR
from ._kernels import KernelTable
from ._operators import FacetJet
from ._worksets import WorksetProgram


def _coefficient_values(
    coefficient_: _ResolvedCoefficient,
    points: Array,
    context: FiniteElementExecutionContext,
    /,
    *,
    value_shape: tuple[int, ...] = (),
    entity_indices: ArrayLike | None = None,
) -> Array:
    values = coefficient_.evaluate(
        points,
        context,
        entity_indices=entity_indices,
    )
    point_shape = points.shape[:-1]
    expected = point_shape + value_shape
    if values.shape == ():
        return jnp.broadcast_to(values, expected)
    if values.shape == point_shape and value_shape:
        return jnp.broadcast_to(
            values.reshape(point_shape + (1,) * len(value_shape)),
            expected,
        )
    if values.shape != expected:
        raise ValueError(
            f"Finite-element coefficient must return shape {expected}; "
            f"got {values.shape}."
        )
    return values


def _scatter_local(
    residual: Array,
    dofs: Array,
    local: Array,
    accumulation: str,
    /,
) -> Array:
    if accumulation == "fast":
        return residual.at[dofs].add(local)
    flat_dofs = dofs.reshape((-1,))
    component_shape = residual.shape[1:]
    component_count = int(np.prod(component_shape, dtype=int)) if component_shape else 1
    flat_local = local.reshape((flat_dofs.size, component_count))
    if accumulation == "deterministic":
        grouped = jax.ops.segment_sum(
            flat_local,
            flat_dofs,
            residual.shape[0],
            indices_are_sorted=False,
            unique_indices=False,
        )
    elif accumulation == "compensated":
        grouped_components = []
        for component in range(component_count):
            grouped_components.append(
                jnp.stack(
                    tuple(
                        compensated_sum(
                            jnp.where(
                                flat_dofs == index,
                                flat_local[:, component],
                                jnp.zeros((), dtype=flat_local.dtype),
                            )
                        )
                        for index in range(residual.shape[0])
                    )
                )
            )
        grouped = jnp.stack(tuple(grouped_components), axis=-1)
    else:
        raise ValueError("Unknown finite-element accumulation policy.")
    return residual + grouped.reshape(residual.shape)


def _workset_domain(
    action,
    discretization: FiniteElementDiscretization,
    entity_indices: Array,
    /,
) -> IntegrationDomain:
    domain = _action_domain(action, discretization)
    requested = np.asarray(entity_indices, dtype=np.int32)
    source = np.asarray(domain.entity_indices, dtype=np.int32)
    rows_by_entity = {int(entity): row for row, entity in enumerate(source)}
    rows = np.asarray(
        tuple(rows_by_entity[int(entity)] for entity in requested),
        dtype=np.int32,
    )
    return IntegrationDomain(
        domain.kind,
        requested,
        domain.support_id,
        domain.entity_set_id,
        owner_cells=np.asarray(domain.owner_cells)[rows],
        neighbour_cells=np.asarray(domain.neighbour_cells)[rows],
        owner_local_entities=np.asarray(domain.owner_local_entities)[rows],
        neighbour_local_entities=np.asarray(domain.neighbour_local_entities)[rows],
        selection_id=domain.selection_id,
    )


def _full_residual(
    form: FiniteElementForm,
    discretization: FiniteElementDiscretization,
    workset_program: WorksetProgram,
    state: Array | tuple[Array, ...],
    accumulation: str,
    context: FiniteElementExecutionContext,
    /,
) -> Array | tuple[Array, ...]:
    states = (state,) if len(form.field_names) == 1 else tuple(state)
    if len(states) != len(form.field_names):
        raise ValueError("Finite-element state blocks must match form fields.")
    state_by_field = dict(zip(form.field_names, states, strict=True))
    residual_by_field = {
        field_name: jnp.zeros_like(value) for field_name, value in state_by_field.items()
    }
    block_names = tuple(block.name for block in discretization.mesh.blocks)
    cell_offsets = np.cumsum(
        np.asarray(
            (0,) + tuple(block.cell_count for block in discretization.mesh.blocks),
            dtype=np.int32,
        )
    )
    for workset in workset_program.worksets:
        block_index = block_names.index(workset.signature.block_name)
        block = discretization.mesh.blocks[block_index]
        work_cells = jnp.asarray(workset.owner_cells, dtype=jnp.int32)
        local_cells = work_cells - int(cell_offsets[block_index])
        gathers = dict(workset.gathers)
        for raw_action_index in workset.action_index_values:
            action = form.actions[raw_action_index]
            output_field = action.field_name
            output_state = state_by_field[output_field]
            output_field_index = discretization._field_index(output_field)
            output_dof_map = discretization.dof_maps[output_field_index]
            output_residual = residual_by_field[output_field]
            domain = (
                _action_domain(action, discretization)
                if len(discretization.mesh.blocks) == 1
                else _workset_domain(
                    action,
                    discretization,
                    workset.entity_index_values,
                )
            )
            if isinstance(action, PreparedOperatorAction):
                if len(discretization.mesh.blocks) != 1:
                    raise ValueError(
                        "Prepared global operator actions require one mesh block."
                    )
                image = action.operator.mv(output_state.reshape((-1,))).reshape(
                    output_state.shape
                )
                residual_by_field[output_field] = output_residual + image
                continue
            if isinstance(action, SIPGFacetAction):
                residual_by_field[output_field] = output_residual + _sipg_facet_residual(
                    discretization,
                    output_field_index,
                    output_state,
                    action,
                    domain,
                    context,
                    accumulation,
                )
                continue
            if isinstance(action, BoundaryLoadAction):
                residual_by_field[output_field] = output_residual - _boundary_load(
                    discretization,
                    output_field_index,
                    action,
                    domain,
                    context,
                )
                continue
            if isinstance(action, ExteriorFacetAction):
                residual_by_field[output_field] = (
                    output_residual
                    + _exterior_facet_residual(
                        discretization,
                        output_field_index,
                        output_state,
                        action,
                        domain,
                        context,
                        accumulation,
                    )
                )
                continue
            if isinstance(action, InteriorFacetAction):
                residual_by_field[output_field] = (
                    output_residual
                    + _interior_facet_residual(
                        discretization,
                        output_field_index,
                        output_state,
                        action,
                        domain,
                        context,
                        accumulation,
                    )
                )
                continue
            rule = _action_rule(action, block.name, block.cell_kind)
            rule_data = _reference_rule_data(rule)
            output_geometry = discretization.evaluate_block_geometry(
                output_field,
                block_index,
                context.runtime.coordinates,
                rule_data.points,
                rule_data.weights,
            )
            dofs = gathers[output_field]
            local_state = output_state[dofs]
            output_orientation = output_dof_map.orientations[block_index][local_cells]
            local_state = local_state * output_orientation.reshape(
                output_orientation.shape
                + (1,) * (local_state.ndim - output_orientation.ndim)
            )
            physical_points = output_geometry.physical_points[local_cells]
            physical_gradients = output_geometry.physical_gradients[local_cells]
            physical_weights = output_geometry.physical_weights[local_cells]
            basis_values = output_geometry.basis_values
            if basis_values.ndim > 2:
                basis_values = basis_values[local_cells]
            if isinstance(action, DiffusionAction):
                field_gradient = oe.contract(
                    "cqid,ci...->cqd...",
                    physical_gradients,
                    local_state,
                )
                values = _coefficient_values(
                    action.diffusivity,
                    physical_points,
                    context,
                    entity_indices=work_cells,
                )
                local = oe.contract(
                    "cq,cq,cqid,cqd...->ci...",
                    physical_weights,
                    values,
                    physical_gradients,
                    field_gradient,
                )
            elif isinstance(action, MassAction):
                if basis_values.ndim != 2:
                    raise ValueError(
                        "Built-in mass terms require a scalar reference basis."
                    )
                field_value = oe.contract(
                    "qi,ci...->cq...",
                    basis_values,
                    local_state,
                )
                values = _coefficient_values(
                    action.coefficient,
                    physical_points,
                    context,
                    entity_indices=work_cells,
                )
                local = oe.contract(
                    "cq,cq,qi,cq...->ci...",
                    physical_weights,
                    values,
                    basis_values,
                    field_value,
                )
            elif isinstance(action, SourceAction):
                if basis_values.ndim != 2:
                    raise ValueError(
                        "Built-in source terms require a scalar reference basis."
                    )
                values = _coefficient_values(
                    action.source,
                    physical_points,
                    context,
                    entity_indices=work_cells,
                    value_shape=output_state.shape[1:],
                )
                local = -oe.contract(
                    "cq,cq...,qi->ci...",
                    physical_weights,
                    values,
                    basis_values,
                )
            elif isinstance(action, CellResidualAction):
                input_values = []
                input_gradients = []
                for input_field in action.input_fields:
                    input_field_index = discretization._field_index(input_field)
                    input_dof_map = discretization.dof_maps[input_field_index]
                    input_geometry = discretization.evaluate_block_geometry(
                        input_field,
                        block_index,
                        context.runtime.coordinates,
                        rule_data.points,
                        rule_data.weights,
                    )
                    input_dofs = gathers[input_field]
                    local_input = state_by_field[input_field][input_dofs]
                    input_orientation = input_dof_map.orientations[block_index][
                        local_cells
                    ]
                    local_input = local_input * input_orientation.reshape(
                        input_orientation.shape
                        + (1,) * (local_input.ndim - input_orientation.ndim)
                    )
                    input_basis = input_geometry.basis_values
                    input_physical_gradients = input_geometry.physical_gradients[
                        local_cells
                    ]
                    if input_basis.ndim == 2:
                        input_values.append(
                            oe.contract(
                                "qi,ci...->cq...",
                                input_basis,
                                local_input,
                            )
                        )
                        input_gradients.append(
                            oe.contract(
                                "cqid,ci...->cqd...",
                                input_physical_gradients,
                                local_input,
                            )
                        )
                    else:
                        input_values.append(
                            oe.contract(
                                "cqiv,ci->cqv",
                                input_basis[local_cells],
                                local_input,
                            )
                        )
                        input_gradients.append(
                            oe.contract(
                                "cqivd,ci->cqvd",
                                input_physical_gradients,
                                local_input,
                            )
                        )
                local = jnp.asarray(
                    action.kernel(
                        tuple(input_values),
                        tuple(input_gradients),
                        physical_points,
                        physical_weights,
                        basis_values,
                        physical_gradients,
                        context,
                    )
                )
                if local.shape != local_state.shape:
                    raise ValueError(
                        "Cell residual kernel must return one local test residual "
                        "per selected cell and output-field DOF."
                    )
            elif isinstance(action, CellEnergyAction):

                def energy(
                    local_coefficients,
                    basis_values_=basis_values,
                    physical_gradients_=physical_gradients,
                    physical_points_=physical_points,
                    physical_weights_=physical_weights,
                    term_=action,
                    context_=context,
                ):
                    if basis_values_.ndim == 2:
                        values_ = oe.contract(
                            "qi,ci...->cq...",
                            basis_values_,
                            local_coefficients,
                        )
                        gradients_ = oe.contract(
                            "cqid,ci...->cqd...",
                            physical_gradients_,
                            local_coefficients,
                        )
                    else:
                        values_ = oe.contract(
                            "cqiv,ci->cqv",
                            basis_values_,
                            local_coefficients,
                        )
                        gradients_ = oe.contract(
                            "cqivd,ci->cqvd",
                            physical_gradients_,
                            local_coefficients,
                        )
                    density = jnp.asarray(
                        term_.density(
                            values_,
                            gradients_,
                            physical_points_,
                            context_,
                        )
                    )
                    if density.shape != physical_weights_.shape:
                        raise ValueError(
                            "Cell energy density must return one scalar per "
                            "selected quadrature point."
                        )
                    return jnp.sum(density * physical_weights_)

                local = jax.grad(energy)(local_state)
            elif isinstance(action, CellBilinearAction):
                matrix = jnp.asarray(
                    action.kernel(
                        physical_points,
                        physical_weights,
                        basis_values,
                        physical_gradients,
                        context,
                    )
                )
                expected_prefix = (
                    local_state.shape[0],
                    local_state.shape[1],
                    local_state.shape[1],
                )
                if matrix.shape != expected_prefix:
                    raise ValueError(
                        "Cell bilinear kernel must return shape "
                        "(cells, local_dofs, local_dofs)."
                    )
                local = oe.contract(
                    "cij,cj...->ci...",
                    matrix,
                    local_state,
                )
            else:
                raise TypeError("Unsupported finite-element term.")
            local = local * output_orientation.reshape(
                output_orientation.shape + (1,) * (local.ndim - output_orientation.ndim)
            )
            residual_by_field[output_field] = _scatter_local(
                output_residual,
                dofs,
                local,
                accumulation,
            )
    residuals = tuple(
        DualSpace(
            discretization.field_spaces[discretization._field_index(name)].vector_space
        ).validate(residual_by_field[name])
        for name in form.field_names
    )
    return residuals[0] if len(residuals) == 1 else residuals


def _polygon_side_points(
    cell_kind: str,
    local_facet: int,
    orientation: float,
    parameter: Array,
    /,
) -> Array:
    local_parameter = parameter if orientation > 0.0 else 1.0 - parameter
    if cell_kind == "triangle":
        if local_facet == 0:
            return jnp.stack((local_parameter, jnp.zeros_like(local_parameter)), axis=-1)
        if local_facet == 1:
            return jnp.stack((1.0 - local_parameter, local_parameter), axis=-1)
        if local_facet == 2:
            return jnp.stack(
                (jnp.zeros_like(local_parameter), 1.0 - local_parameter), axis=-1
            )
    elif cell_kind == "quadrilateral":
        if local_facet == 0:
            return jnp.stack((local_parameter, jnp.zeros_like(local_parameter)), axis=-1)
        if local_facet == 1:
            return jnp.stack((jnp.ones_like(local_parameter), local_parameter), axis=-1)
        if local_facet == 2:
            return jnp.stack(
                (1.0 - local_parameter, jnp.ones_like(local_parameter)), axis=-1
            )
        if local_facet == 3:
            return jnp.stack(
                (jnp.zeros_like(local_parameter), 1.0 - local_parameter), axis=-1
            )
    raise ValueError("SIPG supports triangle Pk or quadrilateral Qk facets.")


def _sipg_facet_residual(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    action: SIPGFacetAction,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    connectivity = discretization.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Executable SIPG currently requires a polygonal mesh.")
    if len(discretization.mesh.blocks) != 1:
        raise ValueError(
            "Executable SIPG currently requires one homogeneous polygonal block."
        )
    dof_map = discretization.dof_maps[field_index]
    if dof_map.association != "cell":
        raise ValueError("SIPG requires a discontinuous cell-local field.")
    element = discretization.elements[field_index][0]
    if element.conformity != "L2":
        raise ValueError("SIPG requires an L2-conforming discontinuous element.")
    block = discretization.mesh.blocks[0]
    rule = _interval_rule() if not action.rules else action.rules[0][1]
    rule_data = _reference_rule_data(rule)
    if rule_data.cell != "interval":
        raise ValueError("SIPG polygon facets require an interval rule.")
    facets = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    owners = jnp.asarray(domain.owner_cells, dtype=jnp.int32)
    neighbours = jnp.asarray(domain.neighbour_cells, dtype=jnp.int32)
    owner_local = jnp.asarray(domain.owner_local_entities, dtype=jnp.int32)
    neighbour_local = jnp.asarray(domain.neighbour_local_entities, dtype=jnp.int32)
    edge_signs = jnp.asarray(connectivity.cell_edge_signs)
    owner_sign = edge_signs[owners, owner_local]
    safe_neighbours = jnp.maximum(neighbours, 0)
    neighbour_sign = edge_signs[safe_neighbours, jnp.maximum(neighbour_local, 0)]
    edge_vertices = jnp.asarray(connectivity.edges, dtype=jnp.int32)[facets]
    edge_points = context.runtime.coordinates[edge_vertices]
    parameter = rule_data.points[:, 0]
    physical_points = (1.0 - parameter)[None, :, None] * edge_points[
        :, None, 0, :
    ] + parameter[None, :, None] * edge_points[:, None, 1, :]
    tangent = edge_points[:, 1] - edge_points[:, 0]
    facet_measure = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
    facet_measure = eqx.error_if(
        facet_measure,
        jnp.any(~jnp.isfinite(facet_measure) | (facet_measure <= 0.0)),
        "SIPG facets require positive finite measure.",
    )
    normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
    normal = normal / facet_measure[:, None]
    cell_centers = jnp.mean(
        context.runtime.coordinates[block.vertices],
        axis=1,
    )
    midpoint = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
    outward = jnp.sum(normal * (midpoint - cell_centers[owners]), axis=-1)
    normal = jnp.where((outward < 0.0)[:, None], -normal, normal)
    facet_weights = facet_measure[:, None] * rule_data.weights[None, :]
    normal_at_points = jnp.broadcast_to(
        normal[:, None, :],
        physical_points.shape,
    )
    measure_at_points = jnp.broadcast_to(
        facet_measure[:, None],
        physical_points.shape[:-1],
    )
    cell_geometry = discretization.evaluate_block_geometry(
        action.field_name,
        0,
        context.runtime.coordinates,
        discretization.block_geometries[field_index][0].reference_points,
        discretization.block_geometries[field_index][0].reference_weights,
    )
    cell_measure = cell_geometry.measure
    plus_height = 2.0 * cell_measure[owners] / facet_measure
    plus_diffusivity = _coefficient_values(
        action.diffusivity,
        physical_points,
        context,
        entity_indices=owners,
    )
    plus_diffusivity = eqx.error_if(
        plus_diffusivity,
        jnp.any(~jnp.isfinite(plus_diffusivity) | (plus_diffusivity <= 0.0)),
        "SIPG diffusivity must be positive and finite.",
    )

    def side_data(local_facet: int, orientation: float, cells: Array):
        reference_points = _polygon_side_points(
            block.cell_kind,
            local_facet,
            orientation,
            parameter,
        )
        geometry = discretization.evaluate_block_geometry(
            action.field_name,
            0,
            context.runtime.coordinates,
            reference_points,
            jnp.ones_like(rule_data.weights),
        )
        if geometry.basis_values.ndim != 2:
            raise ValueError("SIPG currently requires scalar discontinuous elements.")
        basis = geometry.basis_values
        gradients = geometry.physical_gradients[cells]
        dofs = dof_map.cell_dofs[0][cells]
        orientation_values = dof_map.orientations[0][cells]
        local_state = state[dofs] * orientation_values
        value = oe.contract("qi,ei->eq", basis, local_state)
        gradient = oe.contract("eqid,ei->eqd", gradients, local_state)
        return basis, gradients, dofs, orientation_values, value, gradient

    result = jnp.zeros_like(state)
    orientations = (-1.0, 1.0)
    if domain.kind == "interior_facet":
        minus_height = 2.0 * cell_measure[safe_neighbours] / facet_measure
        minus_diffusivity = _coefficient_values(
            action.diffusivity,
            physical_points,
            context,
            entity_indices=safe_neighbours,
        )
        minus_diffusivity = eqx.error_if(
            minus_diffusivity,
            jnp.any(~jnp.isfinite(minus_diffusivity) | (minus_diffusivity <= 0.0)),
            "SIPG diffusivity must be positive and finite.",
        )
        denominator = plus_diffusivity + minus_diffusivity
        plus_weight = minus_diffusivity / denominator
        minus_weight = plus_diffusivity / denominator
        penalty = action.penalty_policy.evaluate(
            element.degree,
            element.degree,
            plus_height[:, None],
            minus_height[:, None],
            plus_diffusivity,
            minus_diffusivity,
        )
        for plus_local_facet in range(block.arity):
            for plus_orientation in orientations:
                plus_mask = (owner_local == plus_local_facet) & (
                    owner_sign * plus_orientation > 0.0
                )
                (
                    plus_basis,
                    plus_gradients,
                    plus_dofs,
                    plus_dof_orientation,
                    plus_value,
                    plus_gradient,
                ) = side_data(plus_local_facet, plus_orientation, owners)
                for minus_local_facet in range(block.arity):
                    for minus_orientation in orientations:
                        active = (
                            plus_mask
                            & (neighbour_local == minus_local_facet)
                            & (neighbour_sign * minus_orientation > 0.0)
                        )
                        (
                            minus_basis,
                            minus_gradients,
                            minus_dofs,
                            minus_dof_orientation,
                            minus_value,
                            minus_gradient,
                        ) = side_data(
                            minus_local_facet,
                            minus_orientation,
                            safe_neighbours,
                        )
                        jet = FacetJet(
                            plus_value,
                            minus_value,
                            plus_gradient,
                            minus_gradient,
                            normal_at_points,
                            measure_at_points,
                        )
                        average_flux = (
                            plus_weight * plus_diffusivity * jet.plus_normal_derivative
                            + minus_weight
                            * minus_diffusivity
                            * jet.minus_normal_derivative
                        )
                        plus_test_normal = jnp.sum(
                            plus_gradients * normal_at_points[:, :, None, :],
                            axis=-1,
                        )
                        minus_test_normal = jnp.sum(
                            minus_gradients * normal_at_points[:, :, None, :],
                            axis=-1,
                        )
                        plus_density = (-average_flux + penalty * jet.jump)[
                            :, :, None
                        ] * plus_basis[None] - (
                            plus_weight * plus_diffusivity * jet.jump
                        )[:, :, None] * plus_test_normal
                        minus_density = (average_flux - penalty * jet.jump)[
                            :, :, None
                        ] * minus_basis[None] - (
                            minus_weight * minus_diffusivity * jet.jump
                        )[:, :, None] * minus_test_normal
                        weights = facet_weights * active[:, None]
                        plus_residual = oe.contract("eq,eqi->ei", weights, plus_density)
                        minus_residual = oe.contract("eq,eqi->ei", weights, minus_density)
                        result = _scatter_local(
                            result,
                            plus_dofs,
                            plus_residual * plus_dof_orientation,
                            accumulation,
                        )
                        result = _scatter_local(
                            result,
                            minus_dofs,
                            minus_residual * minus_dof_orientation,
                            accumulation,
                        )
        return result
    boundary = action.boundary
    if boundary is None:
        raise ValueError("Exterior SIPG facets require boundary data.")
    boundary_value = _coefficient_values(
        boundary.value,
        physical_points,
        context,
        entity_indices=facets,
    )
    penalty = (
        action.penalty_policy
        if boundary.penalty_policy is None
        else boundary.penalty_policy
    ).evaluate(
        element.degree,
        element.degree,
        plus_height[:, None],
        plus_height[:, None],
        plus_diffusivity,
        plus_diffusivity,
    )
    robin = None
    if boundary.robin_coefficient is not None:
        robin = _coefficient_values(
            boundary.robin_coefficient,
            physical_points,
            context,
            entity_indices=facets,
        )
        robin = eqx.error_if(
            robin,
            jnp.any(~jnp.isfinite(robin) | (robin < 0.0)),
            "Robin coefficients must be nonnegative and finite.",
        )
    for plus_local_facet in range(block.arity):
        for plus_orientation in orientations:
            active = (owner_local == plus_local_facet) & (
                owner_sign * plus_orientation > 0.0
            )
            (
                plus_basis,
                plus_gradients,
                plus_dofs,
                plus_dof_orientation,
                plus_value,
                plus_gradient,
            ) = side_data(plus_local_facet, plus_orientation, owners)
            plus_normal_derivative = jnp.sum(
                plus_gradient * normal_at_points,
                axis=-1,
            )
            if boundary.kind == "dirichlet":
                difference = plus_value - boundary_value
                test_normal = jnp.sum(
                    plus_gradients * normal_at_points[:, :, None, :],
                    axis=-1,
                )
                density = (
                    -plus_diffusivity * plus_normal_derivative + penalty * difference
                )[:, :, None] * plus_basis[None] - (plus_diffusivity * difference)[
                    :, :, None
                ] * test_normal
            elif boundary.kind == "neumann":
                density = -boundary_value[:, :, None] * plus_basis[None]
            else:
                if robin is None:
                    raise ValueError("Robin SIPG data require a coefficient.")
                density = (robin * plus_value - boundary_value)[:, :, None] * plus_basis[
                    None
                ]
            weights = facet_weights * active[:, None]
            plus_residual = oe.contract("eq,eqi->ei", weights, density)
            result = _scatter_local(
                result,
                plus_dofs,
                plus_residual * plus_dof_orientation,
                accumulation,
            )
    return result


def _cell_local_interior_facet_residual(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    action: InteriorFacetAction,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    connectivity = discretization.mesh.connectivity
    if (
        not isinstance(connectivity, PolygonalConnectivity)
        or len(discretization.mesh.blocks) != 1
    ):
        raise ValueError(
            "Cell-local numerical fluxes require one homogeneous polygonal block."
        )
    block = discretization.mesh.blocks[0]
    dof_map = discretization.dof_maps[field_index]
    rule = _interval_rule() if not action.rules else action.rules[0][1]
    data = _reference_rule_data(rule)
    if data.cell != "interval":
        raise ValueError("Polygon numerical fluxes require an interval rule.")
    facets = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    owners = jnp.asarray(domain.owner_cells, dtype=jnp.int32)
    neighbours = jnp.asarray(domain.neighbour_cells, dtype=jnp.int32)
    owner_local = jnp.asarray(domain.owner_local_entities, dtype=jnp.int32)
    neighbour_local = jnp.asarray(domain.neighbour_local_entities, dtype=jnp.int32)
    signs = jnp.asarray(connectivity.cell_edge_signs)
    owner_sign = signs[owners, owner_local]
    neighbour_sign = signs[neighbours, neighbour_local]
    edge_vertices = jnp.asarray(connectivity.edges)[facets]
    edge_points = context.runtime.coordinates[edge_vertices]
    parameter = data.points[:, 0]
    physical_points = (1.0 - parameter)[None, :, None] * edge_points[
        :, None, 0, :
    ] + parameter[None, :, None] * edge_points[:, None, 1, :]
    tangent = edge_points[:, 1] - edge_points[:, 0]
    measure = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
    normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
    normal = normal / measure[:, None]
    centers = jnp.mean(context.runtime.coordinates[block.vertices], axis=1)
    midpoint = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
    outward = jnp.sum(normal * (midpoint - centers[owners]), axis=-1)
    normal = jnp.where((outward < 0.0)[:, None], -normal, normal)
    weights = measure[:, None] * data.weights[None, :]

    def side_data(local_facet: int, orientation: float, cells: Array):
        points = _polygon_side_points(
            block.cell_kind,
            local_facet,
            orientation,
            parameter,
        )
        geometry = discretization.evaluate_block_geometry(
            action.field_name,
            0,
            context.runtime.coordinates,
            points,
            jnp.ones_like(data.weights),
        )
        if geometry.basis_values.ndim != 2:
            raise ValueError("Cell-local numerical flux requires a scalar element.")
        basis = geometry.basis_values
        dofs = dof_map.cell_dofs[0][cells]
        dof_orientation = dof_map.orientations[0][cells]
        value = oe.contract("qi,ei->eq", basis, state[dofs] * dof_orientation)
        return basis, dofs, dof_orientation, value

    result = jnp.zeros_like(state)
    for plus_local_facet in range(block.arity):
        for plus_orientation in (-1.0, 1.0):
            plus_mask = (owner_local == plus_local_facet) & (
                owner_sign * plus_orientation > 0.0
            )
            plus_basis, plus_dofs, plus_dof_orientation, plus_value = side_data(
                plus_local_facet,
                plus_orientation,
                owners,
            )
            for minus_local_facet in range(block.arity):
                for minus_orientation in (-1.0, 1.0):
                    active = (
                        plus_mask
                        & (neighbour_local == minus_local_facet)
                        & (neighbour_sign * minus_orientation > 0.0)
                    )
                    (
                        minus_basis,
                        minus_dofs,
                        minus_dof_orientation,
                        minus_value,
                    ) = side_data(
                        minus_local_facet,
                        minus_orientation,
                        neighbours,
                    )
                    plus_flux, minus_flux = action.kernel(
                        plus_value,
                        minus_value,
                        physical_points,
                        weights,
                        normal,
                        context,
                    )
                    plus_flux = jnp.asarray(plus_flux)
                    minus_flux = jnp.asarray(minus_flux)
                    if (
                        plus_flux.shape != plus_value.shape
                        or minus_flux.shape != minus_value.shape
                    ):
                        raise ValueError(
                            "Interior flux kernel must return plus/minus trace shapes."
                        )
                    active_weights = weights * active[:, None]
                    plus_residual = oe.contract(
                        "eq,eq,qi->ei",
                        active_weights,
                        plus_flux,
                        plus_basis,
                    )
                    minus_residual = oe.contract(
                        "eq,eq,qi->ei",
                        active_weights,
                        minus_flux,
                        minus_basis,
                    )
                    result = _scatter_local(
                        result,
                        plus_dofs,
                        plus_residual * plus_dof_orientation,
                        accumulation,
                    )
                    result = _scatter_local(
                        result,
                        minus_dofs,
                        minus_residual * minus_dof_orientation,
                        accumulation,
                    )
    return result


def _exterior_facet_residual(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    action: ExteriorFacetAction,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    connectivity = discretization.mesh.connectivity
    dof_map = discretization.dof_maps[field_index]
    if (
        not isinstance(connectivity, PolygonalConnectivity)
        or len(discretization.mesh.blocks) != 1
        or dof_map.association != "cell"
    ):
        raise ValueError(
            "Exterior state fluxes currently require one polygonal L2 block."
        )
    block = discretization.mesh.blocks[0]
    rule = _interval_rule() if not action.rules else action.rules[0][1]
    data = _reference_rule_data(rule)
    facets = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    owners = jnp.asarray(domain.owner_cells, dtype=jnp.int32)
    owner_local = jnp.asarray(domain.owner_local_entities, dtype=jnp.int32)
    signs = jnp.asarray(connectivity.cell_edge_signs)
    owner_sign = signs[owners, owner_local]
    edge_points = context.runtime.coordinates[jnp.asarray(connectivity.edges)[facets]]
    parameter = data.points[:, 0]
    physical_points = (1.0 - parameter)[None, :, None] * edge_points[
        :, None, 0, :
    ] + parameter[None, :, None] * edge_points[:, None, 1, :]
    tangent = edge_points[:, 1] - edge_points[:, 0]
    measure = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
    normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
    normal = normal / measure[:, None]
    centers = jnp.mean(context.runtime.coordinates[block.vertices], axis=1)
    midpoint = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
    outward = jnp.sum(normal * (midpoint - centers[owners]), axis=-1)
    normal = jnp.where((outward < 0.0)[:, None], -normal, normal)
    weights = measure[:, None] * data.weights[None, :]
    result = jnp.zeros_like(state)
    for local_facet in range(block.arity):
        for orientation in (-1.0, 1.0):
            active = (owner_local == local_facet) & (owner_sign * orientation > 0.0)
            points = _polygon_side_points(
                block.cell_kind,
                local_facet,
                orientation,
                parameter,
            )
            geometry = discretization.evaluate_block_geometry(
                action.field_name,
                0,
                context.runtime.coordinates,
                points,
                jnp.ones_like(data.weights),
            )
            basis = geometry.basis_values
            if basis.ndim != 2:
                raise ValueError("Exterior state flux requires a scalar element.")
            dofs = dof_map.cell_dofs[0][owners]
            dof_orientation = dof_map.orientations[0][owners]
            value = oe.contract("qi,ei->eq", basis, state[dofs] * dof_orientation)
            flux = jnp.asarray(
                action.kernel(value, physical_points, weights, normal, context)
            )
            if flux.shape != value.shape:
                raise ValueError("Exterior flux kernel must return the trace shape.")
            local = oe.contract(
                "eq,eq,qi->ei",
                weights * active[:, None],
                flux,
                basis,
            )
            result = _scatter_local(
                result,
                dofs,
                local * dof_orientation,
                accumulation,
            )
    return result


def _interior_facet_residual(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    action: InteriorFacetAction,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    connectivity = discretization.mesh.connectivity
    facets = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    owners = jnp.asarray(domain.owner_cells, dtype=jnp.int32)
    neighbours = jnp.asarray(domain.neighbour_cells, dtype=jnp.int32)
    dof_map = discretization.dof_maps[field_index]
    if dof_map.association == "cell":
        return _cell_local_interior_facet_residual(
            discretization,
            field_index,
            state,
            action,
            domain,
            context,
            accumulation,
        )
    result = jnp.zeros_like(state)
    if isinstance(connectivity, PolygonalConnectivity):
        rule = _interval_rule()
        if action.rules:
            rule = action.rules[0][1]
        data = _reference_rule_data(rule)
        if data.cell != "interval":
            raise ValueError("Polygon interior facets require an interval rule.")
        edge_vertices = jnp.asarray(connectivity.edges)[facets]
        edge_points = context.runtime.coordinates[edge_vertices]
        parameter = data.points[:, 0]
        physical_points = (1.0 - parameter)[None, :, None] * edge_points[
            :, None, 0, :
        ] + parameter[None, :, None] * edge_points[:, None, 1, :]
        tangent = edge_points[:, 1] - edge_points[:, 0]
        measure = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
        normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        normal = normal / measure[:, None]
        cell_centers = jnp.concatenate(
            tuple(
                jnp.mean(
                    context.runtime.coordinates[block.vertices],
                    axis=1,
                )
                for block in discretization.mesh.blocks
            ),
            axis=0,
        )
        owner_centers = cell_centers[owners]
        midpoint = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
        outward = jnp.sum(normal * (midpoint - owner_centers), axis=-1)
        normal = jnp.where((outward < 0.0)[:, None], -normal, normal)
        weights = measure[:, None] * data.weights[None, :]
        if dof_map.association == "cell":
            plus_dofs = jnp.asarray(owners)[:, None]
            minus_dofs = jnp.asarray(neighbours)[:, None]
            trace_basis = jnp.ones((data.points.shape[0], 1))
        elif dof_map.association == "edge":
            plus_dofs = jnp.asarray(facets)[:, None]
            minus_dofs = plus_dofs
            trace_basis = jnp.ones((data.points.shape[0], 1))
        elif dof_map.association == "vertex_edge":
            edge_dofs = int(discretization.mesh.coordinates.shape[0]) + jnp.asarray(
                facets
            )
            plus_dofs = jnp.concatenate((edge_vertices, edge_dofs[:, None]), axis=1)
            minus_dofs = plus_dofs
            trace_basis = jnp.stack(
                (
                    (1.0 - parameter) * (1.0 - 2.0 * parameter),
                    parameter * (2.0 * parameter - 1.0),
                    4.0 * parameter * (1.0 - parameter),
                ),
                axis=-1,
            )
        else:
            plus_dofs = edge_vertices
            minus_dofs = edge_vertices
            trace_basis = jnp.stack((1.0 - parameter, parameter), axis=-1)
    elif isinstance(connectivity, TetrahedralConnectivity):
        data = _reference_rule_data(_triangle_rule())
        face_vertices = jnp.asarray(connectivity.faces)[facets]
        face_points = context.runtime.coordinates[face_vertices]
        first = data.points[:, 0]
        second = data.points[:, 1]
        trace_basis = jnp.stack((1.0 - first - second, first, second), axis=-1)
        physical_points = oe.contract("qi,eid->eqd", trace_basis, face_points)
        cross = jnp.cross(
            face_points[:, 1] - face_points[:, 0],
            face_points[:, 2] - face_points[:, 0],
        )
        measure = jnp.sqrt(jnp.sum(cross**2, axis=-1))
        normal = cross / measure[:, None]
        weights = measure[:, None] * data.weights[None, :]
        plus_dofs = face_vertices
        minus_dofs = face_vertices
    else:
        raise TypeError("Unsupported interior-facet connectivity.")
    plus_local = state[plus_dofs]
    minus_local = state[minus_dofs]
    plus_value = oe.contract("qi,ei...->eq...", trace_basis, plus_local)
    minus_value = oe.contract("qi,ei...->eq...", trace_basis, minus_local)
    plus_flux, minus_flux = action.kernel(
        plus_value,
        minus_value,
        physical_points,
        weights,
        normal,
        context,
    )
    plus_flux = jnp.asarray(plus_flux)
    minus_flux = jnp.asarray(minus_flux)
    expected = plus_value.shape
    if plus_flux.shape != expected or minus_flux.shape != expected:
        raise ValueError(
            "Interior facet kernel must return plus/minus quadrature flux densities."
        )
    plus_residual = oe.contract(
        "eq,eq...,qi->ei...",
        weights,
        plus_flux,
        trace_basis,
    )
    minus_residual = oe.contract(
        "eq,eq...,qi->ei...",
        weights,
        minus_flux,
        trace_basis,
    )
    result = _scatter_local(result, plus_dofs, plus_residual, accumulation)
    return _scatter_local(result, minus_dofs, minus_residual, accumulation)


def _boundary_load(
    discretization: FiniteElementDiscretization,
    field_index: int,
    action: BoundaryLoadAction,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    /,
) -> Array:
    connectivity = discretization.mesh.connectivity
    selected = jnp.asarray(domain.entity_indices, dtype=jnp.int32)
    owner_cells = jnp.asarray(domain.owner_cells, dtype=jnp.int32)
    field_shape = discretization.field_spaces[field_index].vector_space.structure().shape
    component_shape = field_shape[1:]
    result = jnp.zeros(field_shape, dtype=context.runtime.coordinates.dtype)
    rule_bindings = dict(action.rules)
    cell_start = 0
    for block in discretization.mesh.blocks:
        cell_end = cell_start + block.cell_count
        active = (owner_cells >= cell_start) & (owner_cells < cell_end)
        cell_start = cell_end
        facet_indices = selected
        rule = rule_bindings.get(
            block.name,
            _interval_rule()
            if isinstance(connectivity, PolygonalConnectivity)
            else _triangle_rule(),
        )
        data = _reference_rule_data(rule)
        if isinstance(connectivity, PolygonalConnectivity):
            if data.cell != "interval":
                raise ValueError("Polygon boundary terms require an interval rule.")
            edge_vertices = jnp.asarray(connectivity.edges)[facet_indices]
            edge_points = context.runtime.coordinates[edge_vertices]
            parameter = data.points[:, 0]
            physical_points = (1.0 - parameter)[None, :, None] * edge_points[
                :, None, 0, :
            ] + parameter[None, :, None] * edge_points[:, None, 1, :]
            measure = jnp.sqrt(
                jnp.sum((edge_points[:, 1] - edge_points[:, 0]) ** 2, axis=-1)
            )
            physical_weights = measure[:, None] * data.weights[None, :]
            if (
                discretization.dof_maps[field_index].global_dof_count
                > discretization.mesh.coordinates.shape[0]
            ):
                basis = jnp.stack(
                    (
                        (1.0 - parameter) * (1.0 - 2.0 * parameter),
                        parameter * (2.0 * parameter - 1.0),
                        4.0 * parameter * (1.0 - parameter),
                    ),
                    axis=-1,
                )
                edge_dofs = int(discretization.mesh.coordinates.shape[0]) + jnp.asarray(
                    facet_indices
                )
                dofs = jnp.concatenate((edge_vertices, edge_dofs[:, None]), axis=1)
            else:
                basis = jnp.stack((1.0 - parameter, parameter), axis=-1)
                dofs = edge_vertices
        elif isinstance(connectivity, TetrahedralConnectivity):
            if data.cell != "triangle":
                raise ValueError("Tetrahedron boundary terms require a triangle rule.")
            face_vertices = jnp.asarray(connectivity.faces)[facet_indices]
            face_points = context.runtime.coordinates[face_vertices]
            first = data.points[:, 0]
            second = data.points[:, 1]
            basis = jnp.stack((1.0 - first - second, first, second), axis=-1)
            physical_points = oe.contract("qi,eid->eqd", basis, face_points)
            cross = jnp.cross(
                face_points[:, 1] - face_points[:, 0],
                face_points[:, 2] - face_points[:, 0],
            )
            measure_factor = jnp.sqrt(jnp.sum(cross**2, axis=-1))
            physical_weights = measure_factor[:, None] * data.weights[None, :]
            dofs = face_vertices
        else:
            raise TypeError("Unsupported finite-element boundary connectivity.")
        physical_weights = physical_weights * active[:, None]
        values = _coefficient_values(
            action.load,
            physical_points,
            context,
            entity_indices=facet_indices,
            value_shape=component_shape,
        )
        local = oe.contract(
            "eq,eq...,qi->ei...",
            physical_weights,
            values,
            basis,
        )
        result = result.at[dofs].add(local)
    return result


def execute_finite_element_residual(
    action_ir: LocalActionIR,
    workset_program: WorksetProgram,
    form: FiniteElementForm,
    kernel_table: KernelTable,
    discretization: FiniteElementDiscretization,
    state: Array,
    accumulation: str,
    context: FiniteElementExecutionContext,
    /,
) -> Array:
    if (
        not isinstance(action_ir, LocalActionIR)
        or not isinstance(workset_program, WorksetProgram)
        or not isinstance(kernel_table, KernelTable)
        or workset_program.ir.ir_id != action_ir.ir_id
        or tuple(binding.kernel_id for binding in kernel_table.bindings)
        != tuple(action.kernel_id for action in action_ir.actions)
    ):
        raise ValueError(
            "Finite-element IR, workset program, and kernel table are incompatible."
        )
    return _full_residual(
        form,
        discretization,
        workset_program,
        state,
        accumulation,
        context,
    )


__all__ = ["execute_finite_element_residual"]
