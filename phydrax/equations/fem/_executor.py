#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization._local_variational import (
    AbstractPreparedLocalDiscretization,
)
from ...discretization.fem import FiniteElementDiscretization, IntegrationDomain
from ...discretization.fem._boundary import tensor_local_face
from ...discretization.fem._high_order import SumFactorizationPlan
from ...discretization.fem._sbp import MappedTensorMetrics
from ...linalg import DualSpace
from ...sparse import scatter_local as _scatter_local
from ...variational import FunctionalContext, LocalFieldJet, LocalGeometry
from .._finite_element_variational import (
    _action_domain,
    _action_output_fields,
    _action_rule,
    _interval_rule,
    _reference_rule_data,
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
    LocalFunctionalAction,
    MassAction,
    PairwiseVolumeFluxAction,
    PreparedOperatorAction,
    SIPGFacetAction,
    SourceAction,
    TensorDiffusionAction,
)
from .._variational import VariationalCoefficient
from ._ir import LocalActionIR
from ._kernels import KernelTable
from ._operators import (
    FacetJet,
    FiniteElementFacetMetricData,
    FiniteElementMetricData,
)
from ._worksets import WorksetProgram


def execute_finite_element_mortar_flux(
    workset,
    owner_trace: ArrayLike,
    neighbour_trace: ArrayLike,
    kernel,
    context: FiniteElementExecutionContext,
    /,
) -> tuple[Array, Array]:
    """Evaluate one asymmetric mortar flux and return conservative side lifts."""

    mortar = workset.mortar
    metric = workset.mortar_metric
    if mortar is None or metric is None:
        raise ValueError("Mortar execution requires reference and metric data.")
    owner = mortar.interpolate_left(owner_trace)
    neighbour = mortar.interpolate_right(neighbour_trace)
    normal_scale = jnp.linalg.norm(metric.owner_scaled_normals, axis=-1)
    normal = metric.owner_scaled_normals / normal_scale[:, None]
    result = kernel(
        owner,
        neighbour,
        metric.physical_coordinates,
        metric.physical_weights,
        normal,
        context,
    )
    if isinstance(result, tuple):
        owner_flux, neighbour_flux = result
        weights = metric.physical_weights.reshape(
            metric.physical_weights.shape + (1,) * (jnp.asarray(owner_flux).ndim - 1)
        )
        return (
            mortar.pullback_left_raw(weights * owner_flux),
            mortar.pullback_right_raw(weights * neighbour_flux),
        )
    return mortar.conservative_flux_contributions(result, metric)


def _coefficient_values(
    coefficient_: VariationalCoefficient,
    points: Array,
    context: FiniteElementExecutionContext,
    /,
    *,
    value_shape: tuple[int, ...] | None = (),
    entity_indices: ArrayLike | None = None,
    dof_indices: ArrayLike | None = None,
    dof_orientations: ArrayLike | None = None,
    basis_values: ArrayLike | None = None,
    support_id: str | None = None,
    entity_set_id: str | None = None,
    field_space_id: str | None = None,
    rule_id: str | None = None,
    side: str | None = None,
) -> Array:
    values = coefficient_.evaluate(
        points,
        context,
        entity_indices=entity_indices,
        dof_indices=dof_indices,
        basis_values=basis_values,
        dof_orientations=dof_orientations,
        support_id=support_id if coefficient_.support_id is not None else None,
        entity_set_id=(entity_set_id if coefficient_.entity_set_id is not None else None),
        field_space_id=(
            field_space_id if coefficient_.field_space_id is not None else None
        ),
        rule_id=rule_id if coefficient_.rule_id is not None else None,
        side=side if coefficient_.side != "none" else None,
    )
    point_shape = points.shape[:-1]
    if value_shape is None:
        if values.shape == ():
            return jnp.broadcast_to(values, point_shape)
        physical_dimension = points.shape[-1]
        if values.shape == (physical_dimension, physical_dimension):
            return jnp.broadcast_to(values, point_shape + values.shape)
        if values.shape[: len(point_shape)] == point_shape:
            return values
        raise ValueError(
            "Finite-element tensor coefficient must preserve point axes or return "
            f"one constant ({physical_dimension}, {physical_dimension}) matrix; "
            f"got point shape {point_shape} and coefficient shape {values.shape}."
        )
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


def _cell_metric(
    discretization: FiniteElementDiscretization,
    block_index: int,
    local_cells: Array,
    reference,
    coordinates: Array,
    /,
) -> FiniteElementMetricData:
    coordinate_element = discretization.coordinate_elements[block_index]
    coordinate_basis, coordinate_gradients = coordinate_element.tabulate(
        reference.volume_rule.points
    )
    coordinate_routes = discretization.coordinate_dofs[block_index][local_cells]
    cell_coordinates = coordinates[coordinate_routes]
    return FiniteElementMetricData(
        coordinate_basis,
        coordinate_gradients,
        cell_coordinates,
        reference.weights,
    )


def _tensor_forward(plan: SumFactorizationPlan, local: Array, /) -> Array:
    component_shape = local.shape[2:]
    component_count = int(np.prod(component_shape, dtype=int)) if component_shape else 1
    grid = local.reshape(
        (local.shape[0],) + plan.tabulation.nodal_shape + (component_count,)
    )
    packed = jnp.moveaxis(grid, -1, 1)
    evaluated = plan.interpolate(packed)
    unpacked = jnp.moveaxis(evaluated, 1, -1)
    return unpacked.reshape(
        (local.shape[0],) + plan.tabulation.evaluation_shape + component_shape
    )


def _tensor_transpose(
    plan: SumFactorizationPlan,
    values: Array,
    component_shape: tuple[int, ...],
    /,
) -> Array:
    component_count = int(np.prod(component_shape, dtype=int)) if component_shape else 1
    packed = values.reshape(
        (values.shape[0],) + plan.tabulation.evaluation_shape + (component_count,)
    )
    packed = jnp.moveaxis(packed, -1, 1)
    local = plan.interpolate_transpose(packed)
    local = jnp.moveaxis(local, 1, -1)
    return local.reshape(
        (values.shape[0], int(np.prod(plan.tabulation.nodal_shape, dtype=int)))
        + component_shape
    )


def _tensor_gradient(plan: SumFactorizationPlan, local: Array, /) -> Array:
    component_shape = local.shape[2:]
    component_count = int(np.prod(component_shape, dtype=int)) if component_shape else 1
    grid = local.reshape(
        (local.shape[0],) + plan.tabulation.nodal_shape + (component_count,)
    )
    packed = jnp.moveaxis(grid, -1, 1)
    evaluated = plan.gradient(packed)
    unpacked = jnp.moveaxis(evaluated, 1, -2)
    return unpacked.reshape(
        (local.shape[0],)
        + plan.tabulation.evaluation_shape
        + component_shape
        + (plan.tabulation.dimension,)
    )


def _tensor_gradient_transpose(
    plan: SumFactorizationPlan,
    values: Array,
    component_shape: tuple[int, ...],
    /,
) -> Array:
    component_count = int(np.prod(component_shape, dtype=int)) if component_shape else 1
    packed = values.reshape(
        (values.shape[0],)
        + plan.tabulation.evaluation_shape
        + (component_count, plan.tabulation.dimension)
    )
    packed = jnp.moveaxis(packed, -2, 1)
    local = plan.gradient_transpose(packed)
    local = jnp.moveaxis(local, 1, -1)
    return local.reshape(
        (values.shape[0], int(np.prod(plan.tabulation.nodal_shape, dtype=int)))
        + component_shape
    )


def _pairwise_volume_residual(
    action: PairwiseVolumeFluxAction,
    local_state: Array,
    local_cells: Array,
    reference,
    metric: FiniteElementMetricData,
    context: FiniteElementExecutionContext,
    /,
) -> Array:
    if reference.tensor_tabulation is None or reference.tensor_weights_by_axis is None:
        raise ValueError(
            "Pairwise volume flux requires collocated tensor reference data."
        )
    plan = SumFactorizationPlan(reference.tensor_tabulation)
    if plan.tabulation.nodal_shape != plan.tabulation.evaluation_shape:
        raise ValueError("Pairwise volume flux requires collocated nodal axes.")
    component_shape = local_state.shape[2:]
    nodal_shape = plan.tabulation.nodal_shape
    state_grid = local_state.reshape(
        (local_state.shape[0],) + nodal_shape + component_shape
    )
    if isinstance(context.metric_data, MappedTensorMetrics):
        points_grid = context.metric_data.coordinates[local_cells]
        cofactor_grid = context.metric_data.contravariant_cofactors[local_cells]
    else:
        points_grid = metric.physical_points.reshape(
            (local_state.shape[0]) + nodal_shape + (metric.physical_points.shape[-1],)
        )
        cofactor_grid = metric.cofactor.reshape(
            (local_state.shape[0])
            + nodal_shape
            + (plan.tabulation.dimension, metric.physical_points.shape[-1])
        )
    weight_grid = reference.tensor_weights_by_axis[0]
    for axis_weights in reference.tensor_weights_by_axis[1:]:
        weight_grid = jnp.multiply.outer(weight_grid, axis_weights)
    residual = jnp.zeros_like(state_grid)
    for axis in range(plan.tabulation.dimension):
        source_axis = 1 + axis
        pair_axis = plan.tabulation.dimension
        line_state = jnp.moveaxis(state_grid, source_axis, pair_axis)
        line_points = jnp.moveaxis(points_grid, source_axis, pair_axis)
        line_cofactor = jnp.moveaxis(cofactor_grid[..., axis, :], source_axis, pair_axis)
        line_weights = jnp.moveaxis(weight_grid, axis, -1)
        left = jnp.expand_dims(line_state, pair_axis + 1)
        right = jnp.expand_dims(line_state, pair_axis)
        left_points = jnp.expand_dims(line_points, pair_axis + 1)
        right_points = jnp.expand_dims(line_points, pair_axis)
        physical_flux = jnp.asarray(
            action.kernel(left, right, left_points, right_points, context)
        )
        expected = (
            left.shape[:pair_axis]
            + (
                nodal_shape[axis],
                nodal_shape[axis],
            )
            + component_shape
            + (metric.physical_points.shape[-1],)
        )
        if physical_flux.shape != expected:
            raise ValueError(
                "Pairwise volume flux must return paired values with a trailing "
                "physical-coordinate axis."
            )
        metric_pair = 0.5 * (
            jnp.expand_dims(line_cofactor, pair_axis + 1)
            + jnp.expand_dims(line_cofactor, pair_axis)
        )
        metric_pair = metric_pair.reshape(
            metric_pair.shape[:-1]
            + (1,) * len(component_shape)
            + (metric_pair.shape[-1],)
        )
        contravariant_flux = jnp.sum(metric_pair * physical_flux, axis=-1)
        derivative = plan.tabulation.gradient_factors[axis]
        derivative_shape = (
            (1,) * pair_axis + derivative.shape + (1,) * len(component_shape)
        )
        differentiated = jnp.sum(
            derivative.reshape(derivative_shape) * contravariant_flux,
            axis=pair_axis + 1,
        )
        weights_shape = (1,) + line_weights.shape + (1,) * len(component_shape)
        line_residual = 2.0 * line_weights.reshape(weights_shape) * differentiated
        residual = residual + jnp.moveaxis(line_residual, pair_axis, source_axis)
    return residual.reshape(local_state.shape)


def _cell_coefficient_values(
    coefficient_: VariationalCoefficient,
    discretization: FiniteElementDiscretization,
    block_index: int,
    workset,
    gathers: dict[str, Array],
    physical_points: Array,
    reference_points: Array,
    context: FiniteElementExecutionContext,
    entity_indices: Array,
    /,
    *,
    value_shape: tuple[int, ...] = (),
) -> Array:
    dof_indices = None
    coefficient_basis = None
    coefficient_orientations = None
    field_space_id = coefficient_.field_space_id
    if coefficient_.location == "dof":
        coefficient_field = None
        for field_name in gathers:
            field_index = discretization._field_index(field_name)
            candidate_id = discretization.field_spaces[field_index].field_space_id
            if candidate_id == field_space_id:
                coefficient_field = field_name
                coefficient_element = discretization.elements[field_index][block_index]
                coefficient_basis = coefficient_element.tabulate(reference_points)[0]
                break
        if coefficient_field is None:
            raise ValueError(
                "DOF coefficient field is not gathered by the compiled workset."
            )
        dof_indices = gathers[coefficient_field]
        coefficient_field_index = discretization._field_index(coefficient_field)
        cell_start = sum(
            block.cell_count for block in discretization.mesh.blocks[:block_index]
        )
        local_cells = jnp.asarray(entity_indices, dtype=jnp.int32) - cell_start
        coefficient_orientations = discretization.dof_maps[
            coefficient_field_index
        ].orientations[block_index][local_cells]
    return _coefficient_values(
        coefficient_,
        physical_points,
        context,
        value_shape=value_shape,
        entity_indices=entity_indices,
        dof_indices=dof_indices,
        basis_values=coefficient_basis,
        dof_orientations=coefficient_orientations,
        support_id=workset.signature.support_id,
        entity_set_id=workset.signature.entity_set_id,
        field_space_id=field_space_id,
        rule_id=workset.signature.rule_id,
        side="none",
    )


def _workset_domain(
    action,
    discretization: FiniteElementDiscretization,
    entity_indices: ArrayLike | tuple[int, ...],
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


def _mortar_facet_residual(
    state: Array,
    action: InteriorFacetAction,
    workset,
    context: FiniteElementExecutionContext,
    /,
) -> Array:
    if workset.entity_indices.shape[0] != 1:
        raise ValueError("Each compiled mortar workset currently owns one patch.")
    owner_routes = dict(workset.gathers)[_action_output_fields(action)[0]][0]
    neighbour_routes = dict(workset.neighbour_gathers)[_action_output_fields(action)[0]][
        0
    ]
    owner_trace = state[owner_routes]
    neighbour_trace = state[neighbour_routes]
    owner_lift, neighbour_lift = execute_finite_element_mortar_flux(
        workset,
        owner_trace,
        neighbour_trace,
        lambda plus, minus, points, weights, normal, execution_context: action.kernel(
            (plus,),
            (minus,),
            points,
            weights,
            normal,
            execution_context,
        ),
        context,
    )
    result = jnp.zeros_like(state)
    result = result.at[owner_routes].add(owner_lift)
    return result.at[neighbour_routes].add(neighbour_lift)


def _prepared_local_basis(
    reference, runtime, entity_count: int, /
) -> tuple[Array, Array]:
    identity = jnp.broadcast_to(
        jnp.eye(reference.local_width),
        (entity_count, reference.local_width, reference.local_width),
    )
    values = reference.interpolate(runtime, identity)
    gradients = reference.reference_gradient(runtime, identity)
    return values, gradients


def _prepared_local_coefficient_values(
    coefficient_: VariationalCoefficient,
    discretization: AbstractPreparedLocalDiscretization,
    workset,
    references: dict[str, object],
    metric,
    context: FiniteElementExecutionContext,
    /,
    *,
    value_shape: tuple[int, ...] | None = (),
) -> Array:
    gathers = dict(workset.gathers)
    dof_indices = None
    basis_values = None
    if coefficient_.location == "dof":
        coefficient_field = None
        for field_name in workset.local_region.field_names:
            binding = discretization.local_field_binding(field_name)
            if binding.field_space.field_space_id == coefficient_.field_space_id:
                coefficient_field = field_name
                break
        if coefficient_field is None:
            raise ValueError(
                "DOF coefficient field is not gathered by the local workset."
            )
        dof_indices = gathers[coefficient_field]
        basis_values = _prepared_local_basis(
            references[coefficient_field],
            context.runtime,
            dof_indices.shape[0],
        )[0]
    return _coefficient_values(
        coefficient_,
        metric.points,
        context,
        value_shape=value_shape,
        entity_indices=workset.entity_indices,
        dof_indices=dof_indices,
        basis_values=basis_values,
        support_id=workset.signature.support_id,
        entity_set_id=workset.signature.entity_set_id,
        field_space_id=coefficient_.field_space_id,
        rule_id=workset.signature.rule_id,
        side="none",
    )


def _prepared_local_volume_residual(
    action,
    discretization: AbstractPreparedLocalDiscretization,
    workset,
    state_by_field: dict[str, Array],
    context: FiniteElementExecutionContext,
    /,
) -> tuple[str, Array, Array]:
    region = workset.local_region
    if region is None:
        raise ValueError("Prepared-local volume execution requires a local region.")
    metric = region.geometry_actions.realize(context.runtime)
    references = {
        name: reference.realize_reference_actions(context.runtime)
        for name, reference in zip(
            region.field_names, region.reference_actions, strict=True
        )
    }
    output_field = _action_output_fields(action)[0]
    output_state = state_by_field[output_field]
    gathers = dict(workset.gathers)
    dofs = gathers[output_field]
    local_state = output_state[dofs]
    reference = references[output_field]
    physical_weights = metric.physical_weights

    def weighted(values, scalar_weight):
        return values * scalar_weight.reshape(
            scalar_weight.shape + (1,) * (values.ndim - scalar_weight.ndim)
        )

    if isinstance(action, TensorDiffusionAction):
        if local_state.ndim != 2:
            raise ValueError(
                "TensorDiffusionAction requires a scalar finite-element field."
            )
        coefficient_values = _prepared_local_coefficient_values(
            action.diffusivity,
            discretization,
            workset,
            references,
            metric,
            context,
            value_shape=None,
        )
        physical_gradient = metric.physical_gradient(
            reference.reference_gradient(context.runtime, local_state)
        )
        tensor = action.physical_tensor(
            coefficient_values,
            metric.physical_dimension,
            leading_shape=metric.points.shape[:-1],
        )
        physical_flux = ein.contract("cqde,cqe->cqd", tensor, physical_gradient)
        local = reference.reference_gradient_transpose(
            context.runtime,
            metric.reference_gradient_transpose(
                weighted(physical_flux, physical_weights)
            ),
        )
    elif isinstance(action, DiffusionAction):
        coefficient_values = _prepared_local_coefficient_values(
            action.diffusivity,
            discretization,
            workset,
            references,
            metric,
            context,
        )
        reference_gradient = reference.reference_gradient(context.runtime, local_state)
        physical_gradient = metric.physical_gradient(reference_gradient)
        physical_flux = weighted(
            physical_gradient,
            physical_weights * coefficient_values,
        )
        local = reference.reference_gradient_transpose(
            context.runtime,
            metric.reference_gradient_transpose(physical_flux),
        )
    elif isinstance(action, MassAction):
        coefficient_values = _prepared_local_coefficient_values(
            action.coefficient,
            discretization,
            workset,
            references,
            metric,
            context,
        )
        field_values = reference.interpolate(context.runtime, local_state)
        local = reference.interpolate_transpose(
            context.runtime,
            weighted(field_values, physical_weights * coefficient_values),
        )
    elif isinstance(action, SourceAction):
        component_shape = output_state.shape[1:]
        source_values = _prepared_local_coefficient_values(
            action.source,
            discretization,
            workset,
            references,
            metric,
            context,
            value_shape=component_shape,
        )
        local = -reference.interpolate_transpose(
            context.runtime,
            weighted(source_values, physical_weights),
        )
    elif isinstance(action, BoundaryLoadAction):
        component_shape = output_state.shape[1:]
        load_values = _prepared_local_coefficient_values(
            action.load,
            discretization,
            workset,
            references,
            metric,
            context,
            value_shape=component_shape,
        )
        local = -reference.trace_transpose(
            context.runtime,
            weighted(load_values, physical_weights),
        )
    elif isinstance(action, CellResidualAction):
        basis_values, reference_basis_gradients = _prepared_local_basis(
            reference, context.runtime, local_state.shape[0]
        )
        physical_basis_gradients = metric.physical_gradient(reference_basis_gradients)
        input_values = []
        input_gradients = []
        for input_field in action.input_fields:
            input_reference = references[input_field]
            local_input = state_by_field[input_field][gathers[input_field]]
            input_values.append(input_reference.interpolate(context.runtime, local_input))
            input_reference_gradient = input_reference.reference_gradient(
                context.runtime, local_input
            )
            input_physical_gradient = metric.physical_gradient(input_reference_gradient)
            input_gradients.append(jnp.moveaxis(input_physical_gradient, -1, 2))
        local = jnp.asarray(
            action.kernel(
                tuple(input_values),
                tuple(input_gradients),
                metric.points,
                physical_weights,
                basis_values,
                physical_basis_gradients,
                context,
            )
        )
        if local.shape != local_state.shape:
            raise ValueError(
                "Cell residual kernel must return one local test residual "
                "per selected entity and output-field DOF."
            )
    elif isinstance(action, CellEnergyAction):

        def energy(local_coefficients):
            values = reference.interpolate(context.runtime, local_coefficients)
            reference_gradient = reference.reference_gradient(
                context.runtime, local_coefficients
            )
            physical_gradient = jnp.moveaxis(
                metric.physical_gradient(reference_gradient), -1, 2
            )
            density = jnp.asarray(
                action.density(values, physical_gradient, metric.points, context)
            )
            if density.shape != physical_weights.shape:
                raise ValueError(
                    "Cell energy density must return one scalar per local point."
                )
            return jnp.sum(density * physical_weights)

        local = jax.grad(energy)(local_state)
    elif isinstance(action, CellBilinearAction):
        basis_values, reference_basis_gradients = _prepared_local_basis(
            reference, context.runtime, local_state.shape[0]
        )
        physical_basis_gradients = metric.physical_gradient(reference_basis_gradients)
        matrix = jnp.asarray(
            action.kernel(
                metric.points,
                physical_weights,
                basis_values,
                physical_basis_gradients,
                context,
            )
        )
        expected = (local_state.shape[0], local_state.shape[1], local_state.shape[1])
        if matrix.shape != expected:
            raise ValueError(
                "Cell bilinear kernel must return shape "
                "(entities, local_dofs, local_dofs)."
            )
        local = ein.contract("cij,cj...->ci...", matrix, local_state)
    else:
        raise TypeError("Unsupported prepared-local volume action.")
    valid = jnp.asarray(workset.valid) & jnp.asarray(metric.valid)
    local = jnp.where(
        valid.reshape((valid.shape[0],) + (1,) * (local.ndim - 1)),
        local,
        0.0,
    )
    return output_field, dofs, local


def _prepared_local_functional_value_and_residual(
    action: LocalFunctionalAction,
    discretization: AbstractPreparedLocalDiscretization,
    workset,
    state_by_field: dict[str, Array],
    context: FiniteElementExecutionContext,
    /,
    *,
    with_residual: bool,
):
    region = workset.local_region
    if region is None:
        raise ValueError("Prepared functional execution requires a local region.")
    metric = region.geometry_actions.realize(context.runtime)
    references = {
        name: reference.realize_reference_actions(context.runtime)
        for name, reference in zip(
            region.field_names,
            region.reference_actions,
            strict=True,
        )
    }
    gathers = dict(workset.gathers)
    local_inputs = tuple(
        state_by_field[field_name][gathers[field_name]]
        for field_name in action.input_fields
    )
    input_indices = {
        field_name: index for index, field_name in enumerate(action.input_fields)
    }
    active_indices = tuple(input_indices[field] for field in action.output_fields)
    semantic_to_field = action.semantic_to_field
    functional_context = FunctionalContext(
        time=context.time,
        user_args=context.user_args,
    )
    valid = jnp.asarray(workset.valid) & jnp.asarray(metric.valid)
    normal = None
    if action.term.normal:
        if metric.normals.size == 0:
            raise ValueError(
                "Prepared functional normal request requires trace geometry normals."
            )
        normal = metric.normals

    def energy(*coefficients):
        jets = {}
        for specification in action.term.fields:
            field_name = semantic_to_field[specification.field_name]
            index = input_indices[field_name]
            reference = references[field_name]
            coefficient = coefficients[index]
            value = (
                reference.interpolate(context.runtime, coefficient)
                if specification.value
                else None
            )
            gradient = None
            if specification.gradient:
                reference_gradient = reference.reference_gradient(
                    context.runtime,
                    coefficient,
                )
                gradient = metric.physical_gradient(reference_gradient)
            jets[specification.field_name] = LocalFieldJet(
                value=value,
                gradient=gradient,
            )
        density = jnp.asarray(
            action.term.density(
                jets,
                LocalGeometry(metric.points, normal=normal),
                functional_context,
            )
        )
        if density.shape != metric.physical_weights.shape:
            raise ValueError(
                "A prepared functional density must return one scalar per local point."
            )
        if jnp.iscomplexobj(density):
            raise TypeError("Prepared functional densities must be real.")
        return action.term.weight * jnp.sum(
            density * metric.physical_weights * valid[:, None]
        )

    if not with_residual:
        return energy(*local_inputs), ()
    value, local_gradients = jax.value_and_grad(
        energy,
        argnums=active_indices,
    )(*local_inputs)
    blocks = tuple(
        (field_name, gathers[field_name], local)
        for field_name, local in zip(
            action.output_fields,
            local_gradients,
            strict=True,
        )
    )
    return value, blocks


def _cell_functional_problem(
    discretization: FiniteElementDiscretization,
    state_by_field: dict[str, Array],
    action: LocalFunctionalAction,
    workset,
    block_index: int,
    local_cells: Array,
    context: FiniteElementExecutionContext,
    /,
):
    block = discretization.mesh.blocks[block_index]
    rule = _action_rule(action, block.name, block.cell_kind)
    rule_data = _reference_rule_data(rule)
    gathers = dict(workset.gathers)
    local_inputs = []
    local_dofs = []
    orientations = []
    bases = []
    gradients = []
    physical_points = None
    physical_weights = None
    for field_name in action.input_fields:
        field_index = discretization._field_index(field_name)
        dof_map = discretization.dof_maps[field_index]
        geometry = discretization.evaluate_block_geometry(
            field_name,
            block_index,
            context.runtime.coordinates,
            rule_data.points,
            rule_data.weights,
        )
        dofs = gathers[field_name]
        local = state_by_field[field_name][dofs]
        orientation = dof_map.orientations[block_index][local_cells]
        local = local * orientation.reshape(
            orientation.shape + (1,) * (local.ndim - orientation.ndim)
        )
        basis = geometry.basis_values
        physical_gradient = geometry.physical_gradients[local_cells]
        if basis.ndim > 2:
            basis = basis[local_cells]
        points = geometry.physical_points[local_cells]
        weights = geometry.physical_weights[local_cells]
        if physical_points is None:
            physical_points = points
            physical_weights = weights
        elif (
            points.shape != physical_points.shape
            or weights.shape != physical_weights.shape
        ):
            raise ValueError(
                "Functional input fields must share one physical quadrature layout."
            )
        local_inputs.append(local)
        local_dofs.append(dofs)
        orientations.append(orientation)
        bases.append(basis)
        gradients.append(physical_gradient)
    if physical_points is None or physical_weights is None:
        raise RuntimeError("Functional action has no input fields.")
    semantic_to_field = action.semantic_to_field
    input_indices = {
        field_name: index for index, field_name in enumerate(action.input_fields)
    }
    functional_context = FunctionalContext(
        time=context.time,
        user_args=context.user_args,
    )
    valid = jnp.asarray(workset.valid)

    def energy(*coefficients):
        jets = {}
        for specification in action.term.fields:
            field_name = semantic_to_field[specification.field_name]
            index = input_indices[field_name]
            coefficient = coefficients[index]
            basis = bases[index]
            physical_gradient = gradients[index]
            if basis.ndim == 2:
                value = (
                    ein.contract("qi,ci...->cq...", basis, coefficient)
                    if specification.value
                    else None
                )
                gradient = (
                    jnp.moveaxis(
                        ein.contract(
                            "cqid,ci...->cqd...",
                            physical_gradient,
                            coefficient,
                        ),
                        2,
                        -1,
                    )
                    if specification.gradient
                    else None
                )
            else:
                value = (
                    ein.contract("cqiv,ci->cqv", basis, coefficient)
                    if specification.value
                    else None
                )
                gradient = (
                    ein.contract(
                        "cqivd,ci->cqvd",
                        physical_gradient,
                        coefficient,
                    )
                    if specification.gradient
                    else None
                )
            jets[specification.field_name] = LocalFieldJet(
                value=value,
                gradient=gradient,
            )
        density = jnp.asarray(
            action.term.density(
                jets,
                LocalGeometry(physical_points),
                functional_context,
            )
        )
        if density.shape != physical_weights.shape:
            raise ValueError(
                "A cell functional density must return one scalar per quadrature point."
            )
        if jnp.iscomplexobj(density):
            raise TypeError("Finite-element functional densities must be real.")
        return action.term.weight * jnp.sum(density * physical_weights * valid[:, None])

    return (
        energy,
        tuple(local_inputs),
        tuple(local_dofs),
        tuple(orientations),
        input_indices,
    )


def _accumulate_cell_functional_residual(
    discretization: FiniteElementDiscretization,
    state_by_field: dict[str, Array],
    residual_by_field: dict[str, Array],
    action: LocalFunctionalAction,
    workset,
    block_index: int,
    local_cells: Array,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    energy, local_inputs, local_dofs, orientations, input_indices = (
        _cell_functional_problem(
            discretization,
            state_by_field,
            action,
            workset,
            block_index,
            local_cells,
            context,
        )
    )
    active_indices = tuple(input_indices[field] for field in action.output_fields)
    value, local_gradients = jax.value_and_grad(
        energy,
        argnums=active_indices,
    )(*local_inputs)
    valid = jnp.asarray(workset.valid)
    for output_field, input_index, local in zip(
        action.output_fields,
        active_indices,
        local_gradients,
        strict=True,
    ):
        local = jnp.where(
            valid.reshape((local.shape[0],) + (1,) * (local.ndim - 1)),
            local,
            0.0,
        )
        orientation = orientations[input_index]
        local = local * orientation.reshape(
            orientation.shape + (1,) * (local.ndim - orientation.ndim)
        )
        residual_by_field[output_field] = _scatter_local(
            residual_by_field[output_field],
            local_dofs[input_index],
            local,
            accumulation,
        )
    return value


def _exterior_functional_value(
    discretization: FiniteElementDiscretization,
    state_by_field: dict[str, Array],
    action: LocalFunctionalAction,
    workset,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    /,
    *,
    residual_by_field: dict[str, Array] | None = None,
    accumulation: str = "fast",
) -> Array:
    connectivity = discretization.mesh.connectivity
    if (
        not isinstance(connectivity, PolygonalConnectivity)
        or len(discretization.mesh.blocks) != 1
    ):
        raise ValueError(
            "Exterior functional terms currently require one two-dimensional "
            "polygonal mesh block."
        )
    block = discretization.mesh.blocks[0]
    rule = _interval_rule() if not action.rules else action.rules[0][1]
    data = _reference_rule_data(rule)
    if data.cell != "interval":
        raise ValueError("Polygon exterior functionals require an interval rule.")
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
    if action.term.normal:
        normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        normal = normal / measure[:, None]
        centers = jnp.mean(context.runtime.coordinates[block.vertices], axis=1)
        midpoint = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
        outward = jnp.sum(normal * (midpoint - centers[owners]), axis=-1)
        normal = jnp.where((outward < 0.0)[:, None], -normal, normal)
        normal = jnp.broadcast_to(normal[:, None, :], physical_points.shape)
    else:
        normal = None
    weights = measure[:, None] * data.weights[None, :]
    valid = jnp.asarray(workset.valid)
    semantic_to_field = action.semantic_to_field
    input_indices = {
        field_name: index for index, field_name in enumerate(action.input_fields)
    }
    active_indices = tuple(input_indices[field] for field in action.output_fields)
    functional_context = FunctionalContext(
        time=context.time,
        user_args=context.user_args,
    )
    local_inputs = []
    local_dofs = []
    orientations = []
    selected_bases = []
    for field_name in action.input_fields:
        field_index = discretization._field_index(field_name)
        dof_map = discretization.dof_maps[field_index]
        dofs = dof_map.cell_dofs[0][owners]
        orientation = dof_map.orientations[0][owners]
        local = state_by_field[field_name][dofs]
        local = local * orientation.reshape(
            orientation.shape + (1,) * (local.ndim - orientation.ndim)
        )
        selected_basis = None
        for local_facet in range(block.arity):
            for side_orientation in (-1.0, 1.0):
                active = (
                    valid
                    & (owner_local == local_facet)
                    & (owner_sign * side_orientation > 0.0)
                )
                reference_points = _polygon_side_points(
                    block.cell_kind,
                    local_facet,
                    side_orientation,
                    parameter,
                )
                geometry = discretization.evaluate_block_geometry(
                    field_name,
                    0,
                    context.runtime.coordinates,
                    reference_points,
                    jnp.ones_like(data.weights),
                )
                basis = geometry.basis_values
                if basis.ndim == 2:
                    basis = jnp.broadcast_to(
                        basis,
                        (owners.shape[0],) + basis.shape,
                    )
                else:
                    basis = basis[owners]
                if selected_basis is None:
                    selected_basis = jnp.zeros_like(basis)
                mask = active.reshape((active.shape[0],) + (1,) * (basis.ndim - 1))
                selected_basis = jnp.where(mask, basis, selected_basis)
        if selected_basis is None:
            raise RuntimeError("Exterior functional field has no facet basis.")
        local_inputs.append(local)
        local_dofs.append(dofs)
        orientations.append(orientation)
        selected_bases.append(selected_basis)

    def energy(*coefficients):
        jets = {}
        for specification in action.term.fields:
            field_name = semantic_to_field[specification.field_name]
            index = input_indices[field_name]
            basis = selected_bases[index]
            coefficient = coefficients[index]
            if basis.ndim == 3:
                value = ein.contract("eqi,ei...->eq...", basis, coefficient)
            else:
                value = ein.contract("eqiv,ei->eqv", basis, coefficient)
            jets[specification.field_name] = LocalFieldJet(value=value)
        density = jnp.asarray(
            action.term.density(
                jets,
                LocalGeometry(physical_points, normal=normal),
                functional_context,
            )
        )
        if density.shape != weights.shape:
            raise ValueError(
                "An exterior functional density must return one scalar "
                "per quadrature point."
            )
        if jnp.iscomplexobj(density):
            raise TypeError("Finite-element functional densities must be real.")
        return action.term.weight * jnp.sum(density * weights * valid[:, None])

    if residual_by_field is None:
        return energy(*local_inputs)
    result, local_gradients = jax.value_and_grad(
        energy,
        argnums=active_indices,
    )(*local_inputs)
    for output_field, input_index, local in zip(
        action.output_fields,
        active_indices,
        local_gradients,
        strict=True,
    ):
        orientation = orientations[input_index]
        local = local * orientation.reshape(
            orientation.shape + (1,) * (local.ndim - orientation.ndim)
        )
        residual_by_field[output_field] = _scatter_local(
            residual_by_field[output_field],
            local_dofs[input_index],
            local,
            accumulation,
        )
    return result


def _full_residual(
    form: FiniteElementForm,
    discretization: AbstractPreparedLocalDiscretization,
    workset_program: WorksetProgram,
    state: Array | tuple[Array, ...],
    accumulation: str,
    context: FiniteElementExecutionContext,
    /,
) -> Array | tuple[Array, ...]:
    states = state if isinstance(state, tuple) else (state,)
    if len(states) != len(form.field_names):
        raise ValueError("Finite-element state blocks must match form fields.")
    bindings = {
        name: discretization.local_field_binding(name) for name in form.field_names
    }
    state_by_field = {
        name: bindings[name].flatten(value)
        for name, value in zip(form.field_names, states, strict=True)
    }
    residual_by_field = {
        field_name: jnp.zeros_like(value) for field_name, value in state_by_field.items()
    }
    if isinstance(discretization, FiniteElementDiscretization):
        block_names = tuple(block.name for block in discretization.mesh.blocks)
        cell_offsets = np.cumsum(
            np.asarray(
                (0,) + tuple(block.cell_count for block in discretization.mesh.blocks),
                dtype=np.int32,
            )
        )
    else:
        block_names = ()
        cell_offsets = np.empty((0,), dtype=np.int32)
    for workset in workset_program.worksets:
        if workset.local_region is not None:
            for raw_action_index in workset.action_index_values:
                action = form.actions[raw_action_index]
                if isinstance(action, LocalFunctionalAction):
                    _value, blocks = _prepared_local_functional_value_and_residual(
                        action,
                        discretization,
                        workset,
                        state_by_field,
                        context,
                        with_residual=True,
                    )
                else:
                    output_field, dofs, local = _prepared_local_volume_residual(
                        action,
                        discretization,
                        workset,
                        state_by_field,
                        context,
                    )
                    blocks = ((output_field, dofs, local),)
                for output_field, dofs, local in blocks:
                    residual_by_field[output_field] = _scatter_local(
                        residual_by_field[output_field],
                        dofs,
                        local,
                        accumulation,
                    )
            continue
        if not isinstance(discretization, FiniteElementDiscretization):
            raise ValueError("Facet and legacy volume worksets require finite elements.")
        block_index = block_names.index(workset.signature.block_name)
        block = discretization.mesh.blocks[block_index]
        work_cells = jnp.asarray(workset.owner_cells, dtype=jnp.int32)
        local_cells = work_cells - int(cell_offsets[block_index])
        gathers = dict(workset.gathers)
        for raw_action_index in workset.action_index_values:
            action = form.actions[raw_action_index]
            if isinstance(action, LocalFunctionalAction):
                domain = (
                    _action_domain(action, discretization)
                    if len(discretization.mesh.blocks) == 1
                    else _workset_domain(
                        action,
                        discretization,
                        workset.entity_index_values,
                    )
                )
                if domain.kind == "cell":
                    _accumulate_cell_functional_residual(
                        discretization,
                        state_by_field,
                        residual_by_field,
                        action,
                        workset,
                        block_index,
                        local_cells,
                        context,
                        accumulation,
                    )
                elif domain.kind == "exterior_facet":
                    _exterior_functional_value(
                        discretization,
                        state_by_field,
                        action,
                        workset,
                        domain,
                        context,
                        residual_by_field=residual_by_field,
                        accumulation=accumulation,
                    )
                else:
                    raise ValueError("Unsupported functional integration domain.")
                continue
            output_field = _action_output_fields(action)[0]
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
                load = (
                    _prepared_tensor_boundary_load(
                        discretization,
                        output_field_index,
                        output_state,
                        action,
                        workset,
                        context,
                        accumulation,
                    )
                    if workset.reference is not None
                    else _boundary_load(
                        discretization,
                        output_field_index,
                        action,
                        domain,
                        context,
                    )
                )
                residual_by_field[output_field] = output_residual - load
                continue
            if isinstance(action, ExteriorFacetAction):
                facet_residual = (
                    _prepared_tensor_facet_residual(
                        discretization,
                        state_by_field,
                        output_field,
                        action,
                        workset,
                        context,
                        accumulation,
                    )
                    if workset.reference is not None
                    else _exterior_facet_residual(
                        discretization,
                        output_field_index,
                        output_state,
                        action,
                        workset,
                        domain,
                        context,
                        accumulation,
                    )
                )
                residual_by_field[output_field] = output_residual + facet_residual
                continue
            if isinstance(action, InteriorFacetAction):
                if workset.mortar is not None:
                    if action.input_field_names != (output_field,):
                        raise ValueError(
                            "Cross-field mortar facets are not yet supported."
                        )
                    facet_residual = _mortar_facet_residual(
                        output_state,
                        action,
                        workset,
                        context,
                    )
                elif workset.reference is not None:
                    facet_residual = _prepared_tensor_facet_residual(
                        discretization,
                        state_by_field,
                        output_field,
                        action,
                        workset,
                        context,
                        accumulation,
                    )
                else:
                    if action.input_field_names != (output_field,):
                        raise ValueError(
                            "Cross-field legacy facets require prepared references."
                        )
                    facet_residual = _interior_facet_residual(
                        discretization,
                        output_field_index,
                        output_state,
                        action,
                        workset,
                        domain,
                        context,
                        accumulation,
                    )
                residual_by_field[output_field] = output_residual + facet_residual
                continue
            rule = _action_rule(action, block.name, block.cell_kind)
            rule_data = _reference_rule_data(rule)
            reference = workset.reference
            if reference is None:
                output_geometry = discretization.evaluate_block_geometry(
                    output_field,
                    block_index,
                    context.runtime.coordinates,
                    rule_data.points,
                    rule_data.weights,
                )
                physical_points = output_geometry.physical_points[local_cells]
                physical_gradients = output_geometry.physical_gradients[local_cells]
                physical_weights = output_geometry.physical_weights[local_cells]
                basis_values = output_geometry.basis_values
                if basis_values.ndim > 2:
                    basis_values = basis_values[local_cells]
                metric = None
            else:
                metric = _cell_metric(
                    discretization,
                    block_index,
                    local_cells,
                    reference,
                    context.runtime.coordinates,
                )
                physical_points = metric.physical_points
                factorized_without_test_gradients = workset.signature.local_kernel in (
                    "sum_factorized",
                    "collocated",
                ) and isinstance(
                    action,
                    (
                        DiffusionAction,
                        TensorDiffusionAction,
                        MassAction,
                        SourceAction,
                        CellEnergyAction,
                        PairwiseVolumeFluxAction,
                    ),
                )
                physical_gradients = (
                    None
                    if factorized_without_test_gradients
                    else metric.physical_gradients(reference.basis_gradients)
                )
                physical_weights = metric.weighted_measure
                basis_values = reference.basis_values
            reference_points = (
                rule_data.points if reference is None else reference.volume_rule.points
            )
            dofs = gathers[output_field]
            local_state = output_state[dofs]
            output_orientation = output_dof_map.orientations[block_index][local_cells]
            local_state = local_state * output_orientation.reshape(
                output_orientation.shape
                + (1,) * (local_state.ndim - output_orientation.ndim)
            )
            if isinstance(action, PairwiseVolumeFluxAction):
                if reference is None or metric is None:
                    raise ValueError(
                        "Pairwise volume flux requires a prepared tensor reference."
                    )
                local = _pairwise_volume_residual(
                    action,
                    local_state,
                    local_cells,
                    reference,
                    metric,
                    context,
                )
            elif isinstance(action, TensorDiffusionAction):
                if local_state.ndim != 2:
                    raise ValueError(
                        "TensorDiffusionAction requires a scalar finite-element field."
                    )
                values = _cell_coefficient_values(
                    action.diffusivity,
                    discretization,
                    block_index,
                    workset,
                    gathers,
                    physical_points,
                    reference_points,
                    context,
                    work_cells,
                    value_shape=None,
                )
                dimension = physical_points.shape[-1]
                tensor = action.physical_tensor(
                    values,
                    dimension,
                    leading_shape=physical_points.shape[:-1],
                )
                if (
                    workset.signature.local_kernel in ("sum_factorized", "collocated")
                    and reference is not None
                    and reference.tensor_tabulation is not None
                    and metric is not None
                ):
                    plan = SumFactorizationPlan(reference.tensor_tabulation)
                    reference_gradient = _tensor_gradient(plan, local_state)
                    qshape = plan.tabulation.evaluation_shape
                    inverse_jacobian = metric.inverse_jacobian.reshape(
                        (local_state.shape[0])
                        + qshape
                        + (plan.tabulation.dimension, dimension)
                    )
                    tensor_grid = tensor.reshape(
                        (local_state.shape[0]) + qshape + (dimension, dimension)
                    )
                    weighted_measure = metric.weighted_measure.reshape(
                        (local_state.shape[0]) + qshape
                    )
                    reference_tensor = ein.contract("...rd,...de,...se,...->...rs",
                    inverse_jacobian,
                    tensor_grid,
                    inverse_jacobian,
                    weighted_measure,)
                    reference_flux = ein.contract("...rs,...s->...r", reference_tensor, reference_gradient)
                    local = _tensor_gradient_transpose(plan, reference_flux, ())
                else:
                    field_gradient = ein.contract("cqid,ci->cqd", physical_gradients, local_state)
                    local = ein.contract("cq,cqid,cqde,cqe->ci",
                    physical_weights,
                    physical_gradients,
                    tensor,
                    field_gradient,)
            elif isinstance(action, DiffusionAction):
                values = _cell_coefficient_values(
                    action.diffusivity,
                    discretization,
                    block_index,
                    workset,
                    gathers,
                    physical_points,
                    reference_points,
                    context,
                    work_cells,
                )
                if (
                    workset.signature.local_kernel in ("sum_factorized", "collocated")
                    and reference is not None
                    and reference.tensor_tabulation is not None
                    and metric is not None
                ):
                    plan = SumFactorizationPlan(reference.tensor_tabulation)
                    component_shape = local_state.shape[2:]
                    reference_gradient = _tensor_gradient(plan, local_state)
                    qshape = plan.tabulation.evaluation_shape
                    weighted_metric = metric.weighted_metric.reshape(
                        (local_state.shape[0],)
                        + qshape
                        + (1,) * len(component_shape)
                        + (plan.tabulation.dimension, plan.tabulation.dimension)
                    )
                    flux = ein.contract(
                        "...ab,...b->...a", weighted_metric, reference_gradient
                    )
                    coefficient_grid = values.reshape(
                        (local_state.shape[0],)
                        + qshape
                        + (1,) * (len(component_shape) + 1)
                    )
                    local = _tensor_gradient_transpose(
                        plan,
                        coefficient_grid * flux,
                        component_shape,
                    )
                else:
                    field_gradient = ein.contract(
                        "cqid,ci...->cqd...",
                        physical_gradients,
                        local_state,
                    )
                    local = ein.contract(
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
                values = _cell_coefficient_values(
                    action.coefficient,
                    discretization,
                    block_index,
                    workset,
                    gathers,
                    physical_points,
                    reference_points,
                    context,
                    work_cells,
                )
                if (
                    workset.signature.local_kernel in ("sum_factorized", "collocated")
                    and reference is not None
                    and reference.tensor_tabulation is not None
                    and metric is not None
                ):
                    plan = SumFactorizationPlan(reference.tensor_tabulation)
                    component_shape = local_state.shape[2:]
                    qshape = plan.tabulation.evaluation_shape
                    field_value = _tensor_forward(plan, local_state)
                    weight = (metric.weighted_measure * values).reshape(
                        (local_state.shape[0],) + qshape + (1,) * len(component_shape)
                    )
                    local = _tensor_transpose(
                        plan,
                        weight * field_value,
                        component_shape,
                    )
                else:
                    field_value = ein.contract(
                        "qi,ci...->cq...",
                        basis_values,
                        local_state,
                    )
                    local = ein.contract(
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
                component_shape = output_state.shape[1:]
                values = _cell_coefficient_values(
                    action.source,
                    discretization,
                    block_index,
                    workset,
                    gathers,
                    physical_points,
                    reference_points,
                    context,
                    work_cells,
                    value_shape=component_shape,
                )
                if (
                    workset.signature.local_kernel in ("sum_factorized", "collocated")
                    and reference is not None
                    and reference.tensor_tabulation is not None
                    and metric is not None
                ):
                    plan = SumFactorizationPlan(reference.tensor_tabulation)
                    qshape = plan.tabulation.evaluation_shape
                    weight = metric.weighted_measure.reshape(
                        (local_state.shape[0],) + qshape + (1,) * len(component_shape)
                    )
                    values_grid = values.reshape(
                        (local_state.shape[0],) + qshape + component_shape
                    )
                    local = -_tensor_transpose(
                        plan,
                        weight * values_grid,
                        component_shape,
                    )
                else:
                    local = -ein.contract(
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
                            ein.contract(
                                "qi,ci...->cq...",
                                input_basis,
                                local_input,
                            )
                        )
                        input_gradients.append(
                            ein.contract(
                                "cqid,ci...->cqd...",
                                input_physical_gradients,
                                local_input,
                            )
                        )
                    else:
                        input_values.append(
                            ein.contract(
                                "cqiv,ci->cqv",
                                input_basis[local_cells],
                                local_input,
                            )
                        )
                        input_gradients.append(
                            ein.contract(
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
                if (
                    workset.signature.local_kernel in ("sum_factorized", "collocated")
                    and reference is not None
                    and reference.tensor_tabulation is not None
                    and metric is not None
                ):
                    plan = SumFactorizationPlan(reference.tensor_tabulation)
                    qshape = plan.tabulation.evaluation_shape
                    component_shape = local_state.shape[2:]
                    inverse_jacobian = metric.inverse_jacobian.reshape(
                        (local_state.shape[0],)
                        + qshape
                        + (1,) * len(component_shape)
                        + (
                            plan.tabulation.dimension,
                            metric.physical_points.shape[-1],
                        )
                    )

                    def energy(local_coefficients):
                        values_grid = _tensor_forward(plan, local_coefficients)
                        reference_gradient = _tensor_gradient(plan, local_coefficients)
                        physical_gradient = ein.contract(
                            "...r,...rd->...d",
                            reference_gradient,
                            inverse_jacobian,
                        )
                        physical_gradient = jnp.moveaxis(
                            physical_gradient,
                            -1,
                            1 + plan.tabulation.dimension,
                        )
                        values_ = values_grid.reshape(
                            (local_state.shape[0], -1) + component_shape
                        )
                        gradients_ = physical_gradient.reshape(
                            (
                                local_state.shape[0],
                                -1,
                                metric.physical_points.shape[-1],
                            )
                            + component_shape
                        )
                        density = jnp.asarray(
                            action.density(
                                values_,
                                gradients_,
                                physical_points,
                                context,
                            )
                        )
                        if density.shape != physical_weights.shape:
                            raise ValueError(
                                "Cell energy density must return one scalar per "
                                "selected quadrature point."
                            )
                        return jnp.sum(density * physical_weights)

                else:

                    def energy(local_coefficients):
                        if basis_values.ndim == 2:
                            values_ = ein.contract(
                                "qi,ci...->cq...",
                                basis_values,
                                local_coefficients,
                            )
                            gradients_ = ein.contract(
                                "cqid,ci...->cqd...",
                                physical_gradients,
                                local_coefficients,
                            )
                        else:
                            values_ = ein.contract(
                                "cqiv,ci->cqv",
                                basis_values,
                                local_coefficients,
                            )
                            gradients_ = ein.contract(
                                "cqivd,ci->cqvd",
                                physical_gradients,
                                local_coefficients,
                            )
                        density = jnp.asarray(
                            action.density(
                                values_,
                                gradients_,
                                physical_points,
                                context,
                            )
                        )
                        if density.shape != physical_weights.shape:
                            raise ValueError(
                                "Cell energy density must return one scalar per "
                                "selected quadrature point."
                            )
                        return jnp.sum(density * physical_weights)

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
                local = ein.contract(
                    "cij,cj...->ci...",
                    matrix,
                    local_state,
                )
            else:
                raise TypeError("Unsupported finite-element term.")
            local = jnp.where(
                jnp.asarray(workset.valid).reshape(
                    (local.shape[0],) + (1,) * (local.ndim - 1)
                ),
                local,
                0.0,
            )
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
        ).validate(bindings[name].unflatten(residual_by_field[name]))
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


def _canonicalize_facet(values: Array, permutations: Array, /) -> Array:
    return jax.vmap(
        lambda value, permutation: jnp.zeros_like(value).at[permutation].set(value)
    )(values, permutations)


def _localize_facet(values: Array, permutations: Array, /) -> Array:
    return jax.vmap(lambda value, permutation: value[permutation])(values, permutations)


def _facet_point_permutations(
    connectivity,
    workset,
    cells: Array,
    local_facet: int,
    facet_reference,
    /,
    *,
    neighbour: bool,
) -> Array:
    count = cells.shape[0]
    if neighbour and workset.neighbour_trace_permutations.shape[1] != 0:
        supplied = workset.neighbour_trace_permutations
        if supplied.shape[1] != facet_reference.points.shape[0]:
            raise ValueError(
                "Compiled neighbour trace permutation width does not match the rule."
            )
        if isinstance(connectivity, HexahedralConnectivity):
            points = np.asarray(facet_reference.points)
            normal = np.asarray(facet_reference.normals[0])
            tangential_axes = tuple(
                axis for axis in range(points.shape[1]) if abs(normal[axis]) < 0.5
            )
            if len(tangential_axes) != 2:
                raise ValueError("Hexahedral facet must have two tangential axes.")
            widths = tuple(
                np.unique(np.round(points[:, axis], decimals=14)).size
                for axis in tangential_axes
            )
            owner_to_canonical = jnp.asarray(
                connectivity.cell_face_permutations(*widths)
            )[
                workset.owner_cells,
                workset.owner_local_entities,
            ]
        else:
            identity = jnp.arange(facet_reference.points.shape[0], dtype=jnp.int32)
            owner_to_canonical = jnp.where(
                (workset.owner_permutations > 0)[:, None],
                identity[None],
                identity[::-1][None],
            )
        inverse_supplied = jax.vmap(
            lambda permutation: (
                jnp.zeros_like(permutation)
                .at[permutation]
                .set(jnp.arange(permutation.shape[0], dtype=jnp.int32))
            )
        )(supplied)
        return jnp.take_along_axis(owner_to_canonical, inverse_supplied, axis=1)
    if isinstance(connectivity, HexahedralConnectivity):
        points = np.asarray(facet_reference.points)
        normal = np.asarray(facet_reference.normals[0])
        tangential_axes = tuple(
            axis for axis in range(points.shape[1]) if abs(normal[axis]) < 0.5
        )
        if len(tangential_axes) != 2:
            raise ValueError("Hexahedral facet must have two tangential axes.")
        widths = tuple(
            np.unique(np.round(points[:, axis], decimals=14)).size
            for axis in tangential_axes
        )
        routes = connectivity.cell_face_permutations(*widths)
        return jnp.asarray(routes)[cells, int(local_facet)]
    raw = workset.neighbour_permutations if neighbour else workset.owner_permutations
    if raw.ndim == 1:
        identity = jnp.arange(facet_reference.points.shape[0], dtype=jnp.int32)
        reverse = identity[::-1]
        return jnp.where((raw > 0)[:, None], identity[None], reverse[None])
    return jnp.broadcast_to(
        jnp.arange(facet_reference.points.shape[0], dtype=jnp.int32),
        (count, facet_reference.points.shape[0]),
    )


def _certified_facet_geometry(
    metrics: MappedTensorMetrics,
    reference,
    facet,
    cells: Array,
    local_facet: int,
    /,
) -> tuple[Array, Array]:
    reference_nodes = jnp.asarray(reference.element.reference_nodes)
    dimension = reference_nodes.shape[1]
    axis, side = tensor_local_face(reference.element.cell_kind, int(local_facet))
    node_count = int(round(reference.element.local_dof_count ** (1.0 / dimension)))
    nodal_shape = (node_count,) * dimension
    reference_grid = reference_nodes.reshape(nodal_shape + (dimension,))
    face_nodes = jnp.take(
        reference_grid,
        0 if side == 0 else -1,
        axis=axis,
    ).reshape((-1, dimension))
    facet_points = jnp.asarray(facet.points)
    distances = jnp.max(
        jnp.abs(facet_points[:, None, :] - face_nodes[None, :, :]), axis=-1
    )
    permutation = jnp.argmin(distances, axis=1).astype(jnp.int32)
    physical_points = metrics.face_coordinates[axis][cells, side].reshape(
        (cells.shape[0], -1, dimension)
    )
    scaled_normal = metrics.face_scaled_normals[axis][cells, side].reshape(
        (cells.shape[0], -1, dimension)
    )
    return (
        jnp.take(physical_points, permutation, axis=1),
        jnp.take(scaled_normal, permutation, axis=1),
    )


def _prepared_facet_side(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array | None,
    workset,
    cells: Array,
    local_facet: int,
    context: FiniteElementExecutionContext,
    /,
    *,
    neighbour: bool,
):
    block_names = tuple(block.name for block in discretization.mesh.blocks)
    block_index = block_names.index(workset.signature.block_name)
    offsets = np.cumsum(
        (0,) + tuple(block.cell_count for block in discretization.mesh.blocks)
    )
    local_cells = cells - int(offsets[block_index])
    reference = (
        workset.neighbour_reference
        if neighbour and workset.neighbour_reference is not None
        else workset.reference
    )
    if reference is None:
        raise ValueError("Prepared facet execution requires a prepared reference.")
    facet = reference.facets[int(local_facet)]
    if isinstance(context.metric_data, MappedTensorMetrics):
        physical_points, scaled_normal = _certified_facet_geometry(
            context.metric_data,
            reference,
            facet,
            cells,
            local_facet,
        )
        surface_jacobian = jnp.linalg.norm(scaled_normal, axis=-1)
        normal = scaled_normal / surface_jacobian[..., None]
        physical_weights = surface_jacobian * facet.weights[None, :]
    else:
        coordinate_element = discretization.coordinate_elements[block_index]
        coordinate_basis, coordinate_gradients = coordinate_element.tabulate(facet.points)
        coordinate_routes = discretization.coordinate_dofs[block_index][local_cells]
        metric = FiniteElementMetricData(
            coordinate_basis,
            coordinate_gradients,
            context.runtime.coordinates[coordinate_routes],
            facet.weights,
        )
        facet_metric = FiniteElementFacetMetricData(
            metric,
            facet.normals,
            facet.weights,
        )
        physical_points = facet_metric.physical_points
        physical_weights = facet_metric.physical_weights
        normal = facet_metric.normal
    dof_map = discretization.dof_maps[field_index]
    dofs = dof_map.cell_dofs[block_index][local_cells]
    orientation = dof_map.orientations[block_index][local_cells]
    if state is None:
        trace = None
    else:
        local_state = state[dofs] * orientation.reshape(
            orientation.shape + (1,) * (state[dofs].ndim - orientation.ndim)
        )
        trace = ein.contract("qi,ei...->eq...", facet.basis_values, local_state)
    permutations = _facet_point_permutations(
        discretization.mesh.connectivity,
        workset,
        cells,
        local_facet,
        facet,
        neighbour=neighbour,
    )
    return (
        facet,
        dofs,
        orientation,
        permutations,
        None if trace is None else _canonicalize_facet(trace, permutations),
        _canonicalize_facet(physical_points, permutations),
        _canonicalize_facet(physical_weights, permutations),
        _canonicalize_facet(normal, permutations),
    )


def _prepared_tensor_facet_residual(
    discretization: FiniteElementDiscretization,
    state_by_field: dict[str, Array],
    output_field: str,
    action: InteriorFacetAction | ExteriorFacetAction,
    workset,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    reference = workset.reference
    if reference is None:
        raise ValueError("Prepared tensor facet execution requires a reference.")
    output_index = discretization._field_index(output_field)
    output_state = state_by_field[output_field]
    output_element = discretization.elements[output_index][0]
    for input_field in action.input_field_names:
        input_index = discretization._field_index(input_field)
        if (
            discretization.elements[input_index][0].element_id
            != output_element.element_id
        ):
            raise ValueError(
                "Prepared cross-field facets require one shared reference element."
            )
    owners = jnp.asarray(workset.owner_cells, dtype=jnp.int32)
    neighbours = jnp.maximum(jnp.asarray(workset.neighbour_cells, dtype=jnp.int32), 0)
    owner_local = jnp.asarray(workset.owner_local_entities, dtype=jnp.int32)
    neighbour_local = jnp.asarray(workset.neighbour_local_entities, dtype=jnp.int32)
    valid = jnp.asarray(workset.valid)
    count = owners.shape[0]
    point_count = reference.facets[0].points.shape[0]
    output_component_shape = output_state.shape[1:]
    output_trace_shape = (count, point_count) + output_component_shape
    physical_points = jnp.zeros(
        (count, point_count, context.runtime.coordinates.shape[-1]),
        dtype=context.runtime.coordinates.dtype,
    )
    physical_weights = jnp.zeros(
        (count, point_count), dtype=context.runtime.coordinates.dtype
    )
    normal = jnp.zeros_like(physical_points)
    plus_sides = []
    minus_sides = []
    for local_facet in range(len(reference.facets)):
        plus = _prepared_facet_side(
            discretization,
            output_index,
            None,
            workset,
            owners,
            local_facet,
            context,
            neighbour=False,
        )
        active = valid & (owner_local == local_facet)
        point_mask = active[:, None, None]
        scalar_mask = active[:, None]
        physical_points = jnp.where(point_mask, plus[5], physical_points)
        physical_weights = jnp.where(scalar_mask, plus[6], physical_weights)
        normal = jnp.where(point_mask, plus[7], normal)
        plus_sides.append(plus)
        if isinstance(action, InteriorFacetAction):
            minus_sides.append(
                _prepared_facet_side(
                    discretization,
                    output_index,
                    None,
                    workset,
                    neighbours,
                    local_facet,
                    context,
                    neighbour=True,
                )
            )

    def traces(field_name: str, /, *, neighbour: bool = False) -> Array:
        field_index = discretization._field_index(field_name)
        state = state_by_field[field_name]
        component_shape = state.shape[1:]
        values = jnp.zeros((count, point_count) + component_shape, dtype=state.dtype)
        cells = neighbours if neighbour else owners
        local_entities = neighbour_local if neighbour else owner_local
        for local_facet in range(len(reference.facets)):
            side = _prepared_facet_side(
                discretization,
                field_index,
                state,
                workset,
                cells,
                local_facet,
                context,
                neighbour=neighbour,
            )
            active = valid & (local_entities == local_facet)
            mask = active.reshape((count, 1) + (1,) * len(component_shape))
            values = jnp.where(mask, side[4], values)
        return values

    plus_values = tuple(traces(field) for field in action.input_field_names)
    if isinstance(action, InteriorFacetAction):
        minus_values = tuple(
            traces(field, neighbour=True) for field in action.input_field_names
        )
        plus_flux, minus_flux = action.kernel(
            plus_values,
            minus_values,
            physical_points,
            physical_weights,
            normal,
            context,
        )
        plus_flux = jnp.asarray(plus_flux)
        minus_flux = jnp.asarray(minus_flux)
        if (
            plus_flux.shape != output_trace_shape
            or minus_flux.shape != output_trace_shape
        ):
            raise ValueError(
                "Interior facet kernel must return output-field trace shapes."
            )
    else:
        plus_flux = jnp.asarray(
            action.kernel(
                plus_values,
                physical_points,
                physical_weights,
                normal,
                context,
            )
        )
        if plus_flux.shape != output_trace_shape:
            raise ValueError(
                "Exterior facet kernel must return the output-field trace shape."
            )
        minus_flux = None
    weighted_plus = (
        physical_weights.reshape(
            (count, point_count) + (1,) * len(output_component_shape)
        )
        * plus_flux
    )
    result = jnp.zeros_like(output_state)
    for local_facet, plus in enumerate(plus_sides):
        active = valid & (owner_local == local_facet)
        local_flux = _localize_facet(weighted_plus, plus[3])
        local = ein.contract("qi,eq...->ei...", plus[0].basis_values, local_flux)
        local = jnp.where(
            active.reshape((count, 1) + (1,) * len(output_component_shape)),
            local,
            0.0,
        )
        local = local * plus[2].reshape(
            plus[2].shape + (1,) * (local.ndim - plus[2].ndim)
        )
        result = _scatter_local(result, plus[1], local, accumulation)
    if minus_flux is not None:
        weighted_minus = (
            physical_weights.reshape(
                (count, point_count) + (1,) * len(output_component_shape)
            )
            * minus_flux
        )
        for local_facet, minus in enumerate(minus_sides):
            active = valid & (neighbour_local == local_facet)
            local_flux = _localize_facet(weighted_minus, minus[3])
            local = ein.contract("qi,eq...->ei...", minus[0].basis_values, local_flux)
            local = jnp.where(
                active.reshape((count, 1) + (1,) * len(output_component_shape)),
                local,
                0.0,
            )
            local = local * minus[2].reshape(
                minus[2].shape + (1,) * (local.ndim - minus[2].ndim)
            )
            result = _scatter_local(result, minus[1], local, accumulation)
    return result


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
        _action_output_fields(action)[0],
        0,
        context.runtime.coordinates,
        discretization.block_geometries[field_index][0].reference_points,
        discretization.block_geometries[field_index][0].reference_weights,
    )
    cell_measure = cell_geometry.measure
    plus_height = 2.0 * cell_measure[owners] / facet_measure
    diffusivity_entities = (
        facets if action.diffusivity.entity_set_id == domain.entity_set_id else owners
    )
    diffusivity_entity_set = (
        domain.entity_set_id
        if action.diffusivity.entity_set_id == domain.entity_set_id
        else discretization.cell_domain.entity_set_id
    )
    plus_diffusivity = _coefficient_values(
        action.diffusivity,
        physical_points,
        context,
        entity_indices=diffusivity_entities,
        support_id=domain.support_id,
        entity_set_id=diffusivity_entity_set,
        rule_id=action.diffusivity.rule_id,
        side="plus",
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
            _action_output_fields(action)[0],
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
        value = ein.contract("qi,ei->eq", basis, local_state)
        gradient = ein.contract("eqid,ei->eqd", gradients, local_state)
        return basis, gradients, dofs, orientation_values, value, gradient

    result = jnp.zeros_like(state)
    orientations = (-1.0, 1.0)
    if domain.kind == "interior_facet":
        minus_height = 2.0 * cell_measure[safe_neighbours] / facet_measure
        minus_diffusivity = _coefficient_values(
            action.diffusivity,
            physical_points,
            context,
            entity_indices=(
                facets
                if action.diffusivity.entity_set_id == domain.entity_set_id
                else safe_neighbours
            ),
            support_id=domain.support_id,
            entity_set_id=diffusivity_entity_set,
            rule_id=action.diffusivity.rule_id,
            side="minus",
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
                        plus_residual = ein.contract("eq,eqi->ei", weights, plus_density)
                        minus_residual = ein.contract(
                            "eq,eqi->ei", weights, minus_density
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
    boundary = action.boundary
    if boundary is None:
        raise ValueError("Exterior SIPG facets require boundary data.")
    boundary_value = _coefficient_values(
        boundary.value,
        physical_points,
        context,
        entity_indices=facets,
        support_id=domain.support_id,
        entity_set_id=domain.entity_set_id,
        rule_id=boundary.value.rule_id,
        side="plus",
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
            support_id=domain.support_id,
            entity_set_id=domain.entity_set_id,
            rule_id=boundary.robin_coefficient.rule_id,
            side="plus",
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
            plus_residual = ein.contract("eq,eqi->ei", weights, density)
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
            _action_output_fields(action)[0],
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
        local_state = state[dofs]
        oriented = local_state * dof_orientation.reshape(
            dof_orientation.shape + (1,) * (local_state.ndim - dof_orientation.ndim)
        )
        value = ein.contract("qi,ei...->eq...", basis, oriented)
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
                        (plus_value,),
                        (minus_value,),
                        physical_points,
                        weights,
                        jnp.broadcast_to(normal[:, None, :], physical_points.shape),
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
                    plus_residual = ein.contract(
                        "eq,eq...,qi->ei...",
                        active_weights,
                        plus_flux,
                        plus_basis,
                    )
                    minus_residual = ein.contract(
                        "eq,eq...,qi->ei...",
                        active_weights,
                        minus_flux,
                        minus_basis,
                    )
                    result = _scatter_local(
                        result,
                        plus_dofs,
                        plus_residual
                        * plus_dof_orientation.reshape(
                            plus_dof_orientation.shape
                            + (1,) * (plus_residual.ndim - plus_dof_orientation.ndim)
                        ),
                        accumulation,
                    )
                    result = _scatter_local(
                        result,
                        minus_dofs,
                        minus_residual
                        * minus_dof_orientation.reshape(
                            minus_dof_orientation.shape
                            + (1,) * (minus_residual.ndim - minus_dof_orientation.ndim)
                        ),
                        accumulation,
                    )
    return result


def _exterior_facet_residual(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    action: ExteriorFacetAction,
    workset,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    if workset.reference is not None:
        return _prepared_tensor_facet_residual(
            discretization,
            field_index,
            state,
            action,
            workset,
            context,
            accumulation,
        )
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
                _action_output_fields(action)[0],
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
            local_state = state[dofs]
            oriented = local_state * dof_orientation.reshape(
                dof_orientation.shape + (1,) * (local_state.ndim - dof_orientation.ndim)
            )
            value = ein.contract("qi,ei...->eq...", basis, oriented)
            flux = jnp.asarray(
                action.kernel(
                    (value,),
                    physical_points,
                    weights,
                    jnp.broadcast_to(normal[:, None, :], physical_points.shape),
                    context,
                )
            )
            if flux.shape != value.shape:
                raise ValueError("Exterior flux kernel must return the trace shape.")
            local = ein.contract(
                "eq,eq...,qi->ei...",
                weights * active[:, None],
                flux,
                basis,
            )
            result = _scatter_local(
                result,
                dofs,
                local
                * dof_orientation.reshape(
                    dof_orientation.shape + (1,) * (local.ndim - dof_orientation.ndim)
                ),
                accumulation,
            )
    return result


def _interior_facet_residual(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    action: InteriorFacetAction,
    workset,
    domain: IntegrationDomain,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    if workset.reference is not None:
        return _prepared_tensor_facet_residual(
            discretization,
            field_index,
            state,
            action,
            workset,
            context,
            accumulation,
        )
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
        physical_points = ein.contract("qi,eid->eqd", trace_basis, face_points)
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
    plus_value = ein.contract("qi,ei...->eq...", trace_basis, plus_local)
    minus_value = ein.contract("qi,ei...->eq...", trace_basis, minus_local)
    plus_flux, minus_flux = action.kernel(
        (plus_value,),
        (minus_value,),
        physical_points,
        weights,
        jnp.broadcast_to(normal[:, None, :], physical_points.shape),
        context,
    )
    plus_flux = jnp.asarray(plus_flux)
    minus_flux = jnp.asarray(minus_flux)
    expected = plus_value.shape
    if plus_flux.shape != expected or minus_flux.shape != expected:
        raise ValueError(
            "Interior facet kernel must return plus/minus quadrature flux densities."
        )
    plus_residual = ein.contract(
        "eq,eq...,qi->ei...",
        weights,
        plus_flux,
        trace_basis,
    )
    minus_residual = ein.contract(
        "eq,eq...,qi->ei...",
        weights,
        minus_flux,
        trace_basis,
    )
    result = _scatter_local(result, plus_dofs, plus_residual, accumulation)
    return _scatter_local(result, minus_dofs, minus_residual, accumulation)


def _prepared_tensor_boundary_load(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    action: BoundaryLoadAction,
    workset,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    reference = workset.reference
    if reference is None:
        raise ValueError("Prepared tensor boundary load requires a reference.")
    owners = jnp.asarray(workset.owner_cells, dtype=jnp.int32)
    owner_local = jnp.asarray(workset.owner_local_entities, dtype=jnp.int32)
    valid = jnp.asarray(workset.valid)
    count = owners.shape[0]
    point_count = reference.facets[0].points.shape[0]
    component_shape = state.shape[1:]
    physical_points = jnp.zeros(
        (count, point_count, context.runtime.coordinates.shape[-1]),
        dtype=context.runtime.coordinates.dtype,
    )
    physical_weights = jnp.zeros(
        (count, point_count), dtype=context.runtime.coordinates.dtype
    )
    sides = []
    for local_facet in range(len(reference.facets)):
        side = _prepared_facet_side(
            discretization,
            field_index,
            None,
            workset,
            owners,
            local_facet,
            context,
            neighbour=False,
        )
        active = valid & (owner_local == local_facet)
        physical_points = jnp.where(active[:, None, None], side[5], physical_points)
        physical_weights = jnp.where(active[:, None], side[6], physical_weights)
        sides.append(side)
    values = _coefficient_values(
        action.load,
        physical_points,
        context,
        entity_indices=workset.entity_indices,
        value_shape=component_shape,
        support_id=workset.signature.support_id,
        entity_set_id=workset.signature.entity_set_id,
        rule_id=action.load.rule_id,
        side="plus",
    )
    weighted = (
        physical_weights.reshape((count, point_count) + (1,) * len(component_shape))
        * values
    )
    result = jnp.zeros_like(state)
    for local_facet, side in enumerate(sides):
        active = valid & (owner_local == local_facet)
        local_values = _localize_facet(weighted, side[3])
        local = ein.contract("qi,eq...->ei...", side[0].basis_values, local_values)
        local = jnp.where(
            active.reshape((count, 1) + (1,) * len(component_shape)),
            local,
            0.0,
        )
        local = local * side[2].reshape(
            side[2].shape + (1,) * (local.ndim - side[2].ndim)
        )
        result = _scatter_local(result, side[1], local, accumulation)
    return result


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
            physical_points = ein.contract("qi,eid->eqd", basis, face_points)
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
            support_id=domain.support_id,
            entity_set_id=domain.entity_set_id,
            rule_id=action.load.rule_id,
            side="plus",
        )
        local = ein.contract(
            "eq,eq...,qi->ei...",
            physical_weights,
            values,
            basis,
        )
        result = result.at[dofs].add(local)
    return result


def _execute_finite_element_functional(
    workset_program: WorksetProgram,
    form: FiniteElementForm,
    discretization: AbstractPreparedLocalDiscretization,
    state: Array | tuple[Array, ...],
    context: FiniteElementExecutionContext,
    /,
    *,
    accumulation: str | None,
):
    if form.functional is None or not all(
        isinstance(action, LocalFunctionalAction) for action in form.actions
    ):
        raise ValueError("Finite-element form is not generated by one functional.")
    states = state if isinstance(state, tuple) else (state,)
    if len(states) != len(form.field_names):
        raise ValueError("Finite-element state blocks must match form fields.")
    bindings = {
        name: discretization.local_field_binding(name) for name in form.field_names
    }
    state_by_field = {
        name: bindings[name].flatten(value)
        for name, value in zip(form.field_names, states, strict=True)
    }
    residual_by_field = (
        None
        if accumulation is None
        else {
            field_name: jnp.zeros_like(value)
            for field_name, value in state_by_field.items()
        }
    )
    accumulator_dtype = next(iter(state_by_field.values())).dtype
    term_values = [jnp.zeros((), dtype=accumulator_dtype) for _ in form.actions]
    if isinstance(discretization, FiniteElementDiscretization):
        block_names = tuple(block.name for block in discretization.mesh.blocks)
        cell_offsets = np.cumsum(
            np.asarray(
                (0,) + tuple(block.cell_count for block in discretization.mesh.blocks),
                dtype=np.int32,
            )
        )
    else:
        block_names = ()
        cell_offsets = np.empty((0,), dtype=np.int32)
    for workset in workset_program.worksets:
        if workset.local_region is not None:
            for raw_action_index in workset.action_index_values:
                action = form.actions[raw_action_index]
                contribution, blocks = _prepared_local_functional_value_and_residual(
                    action,
                    discretization,
                    workset,
                    state_by_field,
                    context,
                    with_residual=residual_by_field is not None,
                )
                if residual_by_field is not None:
                    for output_field, dofs, local in blocks:
                        residual_by_field[output_field] = _scatter_local(
                            residual_by_field[output_field],
                            dofs,
                            local,
                            accumulation,
                        )
                term_values[raw_action_index] = (
                    term_values[raw_action_index] + contribution
                )
            continue
        if not isinstance(discretization, FiniteElementDiscretization):
            raise ValueError(
                "Legacy functional worksets require finite-element discretization."
            )
        block_index = block_names.index(workset.signature.block_name)
        work_cells = jnp.asarray(workset.owner_cells, dtype=jnp.int32)
        local_cells = work_cells - int(cell_offsets[block_index])
        for raw_action_index in workset.action_index_values:
            action = form.actions[raw_action_index]
            domain = (
                _action_domain(action, discretization)
                if len(discretization.mesh.blocks) == 1
                else _workset_domain(
                    action,
                    discretization,
                    workset.entity_index_values,
                )
            )
            if domain.kind == "cell":
                if residual_by_field is None:
                    energy, local_inputs, _, _, _ = _cell_functional_problem(
                        discretization,
                        state_by_field,
                        action,
                        workset,
                        block_index,
                        local_cells,
                        context,
                    )
                    contribution = energy(*local_inputs)
                else:
                    contribution = _accumulate_cell_functional_residual(
                        discretization,
                        state_by_field,
                        residual_by_field,
                        action,
                        workset,
                        block_index,
                        local_cells,
                        context,
                        accumulation,
                    )
            elif domain.kind == "exterior_facet":
                contribution = _exterior_functional_value(
                    discretization,
                    state_by_field,
                    action,
                    workset,
                    domain,
                    context,
                    residual_by_field=residual_by_field,
                    accumulation="fast" if accumulation is None else accumulation,
                )
            else:
                raise ValueError("Unsupported functional integration domain.")
            term_values[raw_action_index] = term_values[raw_action_index] + contribution
    values = tuple(discretization.precision_policy.output(value) for value in term_values)
    total = discretization.precision_policy.output(jnp.sum(jnp.stack(values)))
    if residual_by_field is None:
        return total, values, None
    residuals = tuple(
        DualSpace(
            discretization.field_spaces[discretization._field_index(name)].vector_space
        ).validate(bindings[name].unflatten(residual_by_field[name]))
        for name in form.field_names
    )
    residual = residuals[0] if len(residuals) == 1 else residuals
    return total, values, residual


def execute_finite_element_potential(
    workset_program: WorksetProgram,
    form: FiniteElementForm,
    discretization: AbstractPreparedLocalDiscretization,
    state: Array | tuple[Array, ...],
    context: FiniteElementExecutionContext,
    /,
) -> tuple[Array, tuple[Array, ...]]:
    """Evaluate all declared functional terms on one full FE state."""
    value, term_values, _ = _execute_finite_element_functional(
        workset_program,
        form,
        discretization,
        state,
        context,
        accumulation=None,
    )
    return value, term_values


def execute_finite_element_value_and_residual(
    workset_program: WorksetProgram,
    form: FiniteElementForm,
    discretization: AbstractPreparedLocalDiscretization,
    state: Array | tuple[Array, ...],
    accumulation: str,
    context: FiniteElementExecutionContext,
    /,
):
    """Evaluate one functional and all full-space variations in one local pass."""
    return _execute_finite_element_functional(
        workset_program,
        form,
        discretization,
        state,
        context,
        accumulation=accumulation,
    )


def execute_finite_element_residual(
    action_ir: LocalActionIR,
    workset_program: WorksetProgram,
    form: FiniteElementForm,
    kernel_table: KernelTable,
    discretization: AbstractPreparedLocalDiscretization,
    state: Array | tuple[Array, ...],
    accumulation: str,
    context: FiniteElementExecutionContext,
    /,
) -> Array | tuple[Array, ...]:
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
    for workset in workset_program.worksets:
        signature = workset.signature
        for action_index in workset.action_index_values:
            binding = kernel_table.binding(action_ir.actions[action_index].kernel_id)
            strategy_matches = binding.local_kernel == signature.local_kernel or (
                binding.local_kernel.startswith("mixed[")
                and signature.local_kernel in binding.local_kernel[6:-1].split(",")
            )
            if (
                not strategy_matches
                or not set(signature.reference_action_ids).issubset(
                    binding.reference_action_ids
                )
                or not set(signature.field_layout_ids).issubset(binding.field_layout_ids)
                or signature.geometry_action_id not in binding.geometry_action_ids
                or signature.coefficient_layout_ids != binding.coefficient_layout_ids
                or signature.precision_id != binding.precision_id
                or signature.ir_semantics_id != binding.ir_semantics_id
            ):
                raise ValueError(
                    "A workset signature does not match its compiled kernel binding."
                )
    discretization.validate_local_runtime(context.runtime)
    return _full_residual(
        form,
        discretization,
        workset_program,
        state,
        accumulation,
        context,
    )


__all__ = [
    "execute_finite_element_potential",
    "execute_finite_element_value_and_residual",
    "execute_finite_element_residual",
]
