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
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization.fem import FiniteElementDiscretization, IntegrationDomain
from ...discretization.fem._high_order import SumFactorizationPlan
from ...discretization.fem._sbp import MappedTensorMetrics
from ...linalg import DualSpace
from .._finite_element_variational import (
    _action_domain,
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
    FiniteElementCoefficient,
    FiniteElementExecutionContext,
    FiniteElementForm,
    InteriorFacetAction,
    MassAction,
    PairwiseVolumeFluxAction,
    PreparedOperatorAction,
    SIPGFacetAction,
    SourceAction,
)
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
    coefficient_: FiniteElementCoefficient,
    points: Array,
    context: FiniteElementExecutionContext,
    /,
    *,
    value_shape: tuple[int, ...] = (),
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
    coefficient_: FiniteElementCoefficient,
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
    owner_routes = dict(workset.gathers)[action.field_name][0]
    neighbour_routes = dict(workset.neighbour_gathers)[action.field_name][0]
    owner_trace = state[owner_routes]
    neighbour_trace = state[neighbour_routes]
    owner_lift, neighbour_lift = execute_finite_element_mortar_flux(
        workset,
        owner_trace,
        neighbour_trace,
        action.kernel,
        context,
    )
    result = jnp.zeros_like(state)
    result = result.at[owner_routes].add(owner_lift)
    return result.at[neighbour_routes].add(neighbour_lift)


def _full_residual(
    form: FiniteElementForm,
    discretization: FiniteElementDiscretization,
    workset_program: WorksetProgram,
    state: Array | tuple[Array, ...],
    accumulation: str,
    context: FiniteElementExecutionContext,
    /,
) -> Array | tuple[Array, ...]:
    states = state if isinstance(state, tuple) else (state,)
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
                residual_by_field[output_field] = (
                    output_residual
                    + _exterior_facet_residual(
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
                continue
            if isinstance(action, InteriorFacetAction):
                facet_residual = (
                    _mortar_facet_residual(
                        output_state,
                        action,
                        workset,
                        context,
                    )
                    if workset.mortar is not None
                    else _interior_facet_residual(
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
                    flux = oe.contract(
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
                    field_gradient = oe.contract(
                        "cqid,ci...->cqd...",
                        physical_gradients,
                        local_state,
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
                    field_value = oe.contract(
                        "qi,ci...->cq...",
                        basis_values,
                        local_state,
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
                        physical_gradient = oe.contract(
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
                            (local_state.shape[0], -1, metric.physical_points.shape[-1])
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
                            values_ = oe.contract(
                                "qi,ci...->cq...",
                                basis_values,
                                local_coefficients,
                            )
                            gradients_ = oe.contract(
                                "cqid,ci...->cqd...",
                                physical_gradients,
                                local_coefficients,
                            )
                        else:
                            values_ = oe.contract(
                                "cqiv,ci->cqv",
                                basis_values,
                                local_coefficients,
                            )
                            gradients_ = oe.contract(
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
                local = oe.contract(
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
    /,
) -> tuple[Array, Array]:
    reference_nodes = np.asarray(reference.element.reference_nodes)
    dimension = reference_nodes.shape[1]
    normal = np.asarray(facet.normals[0])
    axis = int(np.argmax(np.abs(normal)))
    side = 0 if normal[axis] < 0.0 else 1
    nodal_shape = tuple(
        np.unique(reference_nodes[:, coordinate_axis]).size
        for coordinate_axis in range(dimension)
    )
    reference_grid = reference_nodes.reshape(nodal_shape + (dimension,))
    face_nodes = np.take(
        reference_grid,
        0 if side == 0 else -1,
        axis=axis,
    ).reshape((-1, dimension))
    facet_points = np.asarray(facet.points)
    if face_nodes.shape != facet_points.shape:
        raise ValueError(
            "Certified facet metrics require the collocated nodal trace rule."
        )
    distances = np.max(np.abs(facet_points[:, None, :] - face_nodes[None, :, :]), axis=-1)
    permutation = np.argmin(distances, axis=1).astype(np.int32)
    if (
        np.max(np.min(distances, axis=1), initial=0.0) > 1.0e-12
        or np.unique(permutation).size != permutation.size
    ):
        raise ValueError(
            "Certified facet metric nodes do not match the prepared trace nodes."
        )
    physical_points = metrics.face_coordinates[axis][cells, side].reshape(
        (cells.shape[0], -1, dimension)
    )
    scaled_normal = metrics.face_scaled_normals[axis][cells, side].reshape(
        (cells.shape[0], -1, dimension)
    )
    indices = jnp.asarray(permutation, dtype=jnp.int32)
    return (
        jnp.take(physical_points, indices, axis=1),
        jnp.take(scaled_normal, indices, axis=1),
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
    if len(discretization.mesh.blocks) != 1:
        raise ValueError(
            "Prepared tensor facets currently require one homogeneous block."
        )
    reference = workset.reference
    if reference is None:
        raise ValueError("Prepared facet execution requires a prepared reference.")
    facet = reference.facets[int(local_facet)]
    if isinstance(context.metric_data, MappedTensorMetrics):
        physical_points, scaled_normal = _certified_facet_geometry(
            context.metric_data,
            reference,
            facet,
            cells,
        )
        surface_jacobian = jnp.linalg.norm(scaled_normal, axis=-1)
        normal = scaled_normal / surface_jacobian[..., None]
        physical_weights = surface_jacobian * facet.weights[None, :]
    else:
        coordinate_element = discretization.coordinate_elements[0]
        coordinate_basis, coordinate_gradients = coordinate_element.tabulate(facet.points)
        coordinate_routes = discretization.coordinate_dofs[0][cells]
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
    dofs = dof_map.cell_dofs[0][cells]
    orientation = dof_map.orientations[0][cells]
    if state is None:
        trace = None
    else:
        local_state = state[dofs] * orientation.reshape(
            orientation.shape + (1,) * (state[dofs].ndim - orientation.ndim)
        )
        trace = oe.contract("qi,ei...->eq...", facet.basis_values, local_state)
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
    field_index: int,
    state: Array,
    action: InteriorFacetAction | ExteriorFacetAction,
    workset,
    context: FiniteElementExecutionContext,
    accumulation: str,
    /,
) -> Array:
    reference = workset.reference
    if reference is None:
        raise ValueError("Prepared tensor facet execution requires a reference.")
    owners = jnp.asarray(workset.owner_cells, dtype=jnp.int32)
    neighbours = jnp.maximum(jnp.asarray(workset.neighbour_cells, dtype=jnp.int32), 0)
    owner_local = jnp.asarray(workset.owner_local_entities, dtype=jnp.int32)
    neighbour_local = jnp.asarray(workset.neighbour_local_entities, dtype=jnp.int32)
    valid = jnp.asarray(workset.valid)
    count = owners.shape[0]
    point_count = reference.facets[0].points.shape[0]
    component_shape = state.shape[1:]
    trace_shape = (count, point_count) + component_shape
    plus_value = jnp.zeros(trace_shape, dtype=state.dtype)
    minus_value = jnp.zeros(trace_shape, dtype=state.dtype)
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
            field_index,
            state,
            workset,
            owners,
            local_facet,
            context,
            neighbour=False,
        )
        active = valid & (owner_local == local_facet)
        value_mask = active.reshape((count, 1) + (1,) * len(component_shape))
        point_mask = active[:, None, None]
        scalar_mask = active[:, None]
        plus_value = jnp.where(value_mask, plus[4], plus_value)
        physical_points = jnp.where(point_mask, plus[5], physical_points)
        physical_weights = jnp.where(scalar_mask, plus[6], physical_weights)
        normal = jnp.where(point_mask, plus[7], normal)
        plus_sides.append(plus)
        if isinstance(action, InteriorFacetAction):
            minus = _prepared_facet_side(
                discretization,
                field_index,
                state,
                workset,
                neighbours,
                local_facet,
                context,
                neighbour=True,
            )
            minus_active = valid & (neighbour_local == local_facet)
            minus_mask = minus_active.reshape((count, 1) + (1,) * len(component_shape))
            minus_value = jnp.where(minus_mask, minus[4], minus_value)
            minus_sides.append(minus)
    if isinstance(action, InteriorFacetAction):
        plus_flux, minus_flux = action.kernel(
            plus_value,
            minus_value,
            physical_points,
            physical_weights,
            normal,
            context,
        )
        plus_flux = jnp.asarray(plus_flux)
        minus_flux = jnp.asarray(minus_flux)
        if plus_flux.shape != trace_shape or minus_flux.shape != trace_shape:
            raise ValueError(
                "Interior facet kernel must return both canonical trace shapes."
            )
    else:
        plus_flux = jnp.asarray(
            action.kernel(
                plus_value,
                physical_points,
                physical_weights,
                normal,
                context,
            )
        )
        if plus_flux.shape != trace_shape:
            raise ValueError("Exterior facet kernel must return the trace shape.")
        minus_flux = None
    weighted_plus = (
        physical_weights.reshape((count, point_count) + (1,) * len(component_shape))
        * plus_flux
    )
    result = jnp.zeros_like(state)
    for local_facet, plus in enumerate(plus_sides):
        active = valid & (owner_local == local_facet)
        local_flux = _localize_facet(weighted_plus, plus[3])
        local = oe.contract("qi,eq...->ei...", plus[0].basis_values, local_flux)
        local = jnp.where(
            active.reshape((count, 1) + (1,) * len(component_shape)),
            local,
            0.0,
        )
        local = local * plus[2].reshape(
            plus[2].shape + (1,) * (local.ndim - plus[2].ndim)
        )
        result = _scatter_local(result, plus[1], local, accumulation)
    if minus_flux is not None:
        weighted_minus = (
            physical_weights.reshape((count, point_count) + (1,) * len(component_shape))
            * minus_flux
        )
        for local_facet, minus in enumerate(minus_sides):
            active = valid & (neighbour_local == local_facet)
            local_flux = _localize_facet(weighted_minus, minus[3])
            local = oe.contract("qi,eq...->ei...", minus[0].basis_values, local_flux)
            local = jnp.where(
                active.reshape((count, 1) + (1,) * len(component_shape)),
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
        action.field_name,
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
        local = oe.contract("qi,eq...->ei...", side[0].basis_values, local_values)
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
            support_id=domain.support_id,
            entity_set_id=domain.entity_set_id,
            rule_id=action.load.rule_id,
            side="plus",
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
                or signature.prepared_reference_id not in binding.reference_ids
                or signature.element_id not in binding.element_ids
                or signature.coordinate_element_id not in binding.coordinate_element_ids
                or signature.representation not in binding.representations
                or signature.mapping not in binding.mappings
                or signature.coefficient_layout_ids != binding.coefficient_layout_ids
                or signature.precision_id != binding.precision_id
                or signature.ir_semantics_id != binding.ir_semantics_id
            ):
                raise ValueError(
                    "A workset signature does not match its compiled kernel binding."
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
