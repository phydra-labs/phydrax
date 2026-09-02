#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._polynomial import ScaledMonomialBasis, total_degree_multiindices
from ..._strict import StrictModule
from ...linalg import prepare_local_block_factorization, solve_local_blocks
from .._polygon_geometry import PolygonCubature, PolygonGeometry
from ._spec import VirtualElementSpec


class VirtualElementProjectionEvidence(StrictModule):
    factorization_valid: Array
    g_bd_error: Array
    h1_reproduction_error: Array
    l2_reproduction_error: Array
    h1_idempotence_error: Array
    l2_idempotence_error: Array
    differential_reproduction_error: Array
    minimum_g_singular_value: Array
    minimum_h_singular_value: Array
    maximum_g_condition: Array
    maximum_h_condition: Array

    @property
    def passed(self) -> Array:
        return jnp.all(self.factorization_valid)


class VirtualElementProjectionData(StrictModule):
    family: str = eqx.field(static=True)
    projection_kind: str = eqx.field(static=True)
    polynomial_value_shape: tuple[int, ...] = eqx.field(static=True)
    differential_kind: str = eqx.field(static=True)
    differential_degree: int = eqx.field(static=True)
    basis: ScaledMonomialBasis
    dof_matrix: Array
    energy_functionals: Array
    augmented_gram: Array
    gradient_gram: Array
    mass_gram: Array
    h1_coefficients: Array
    h1_dof_projector: Array
    l2_coefficients: Array
    l2_dof_projector: Array
    differential_coefficients: Array
    local_points: Array
    local_point_valid: Array
    evidence: VirtualElementProjectionEvidence
    projection_id: str = eqx.field(static=True)


def _edge_trace_points(geometry: PolygonGeometry, degree: int, /):
    from ...integration import GaussLobattoLegendreRule, interval_rule_data

    data = interval_rule_data(GaussLobattoLegendreRule(degree + 1))
    nodes = 0.5 * (jnp.asarray(data.nodes) + 1.0)
    weights = 0.5 * jnp.asarray(data.weights)
    start = geometry.vertices
    stop = jnp.roll(geometry.vertices, -1, axis=1)
    points = (1.0 - nodes[None, None, :, None]) * start[:, :, None, :] + nodes[
        None, None, :, None
    ] * stop[:, :, None, :]
    return points, nodes, weights


def _prepare_h1_virtual_element_projections(
    geometry: PolygonGeometry,
    cubature: PolygonCubature,
    element: VirtualElementSpec,
    /,
) -> VirtualElementProjectionData:
    """Build computable H1 and enhanced L2 projectors for one arity bucket."""

    cells, arity, _ = geometry.vertices.shape
    degree = element.degree
    local_dofs = element.local_dof_count(arity)
    basis = ScaledMonomialBasis(2, degree)
    polynomial_count = basis.feature_count
    cell_points = cubature.points
    cell_weights = cubature.weights
    basis_values = basis.evaluate(
        cell_points, geometry.centroids, geometry.characteristic_lengths
    )
    basis_gradients = basis.gradient(
        cell_points, geometry.centroids, geometry.characteristic_lengths
    )
    edge_points, _, edge_weights = _edge_trace_points(geometry, degree)
    flat_edge_points = edge_points.reshape((cells, arity * (degree + 1), 2))
    edge_basis = basis.evaluate(
        flat_edge_points, geometry.centroids, geometry.characteristic_lengths
    ).reshape((cells, arity, degree + 1, polynomial_count))
    edge_gradient = basis.gradient(
        flat_edge_points, geometry.centroids, geometry.characteristic_lengths
    ).reshape((cells, arity, degree + 1, polynomial_count, 2))

    dof_matrix = jnp.zeros((cells, local_dofs, polynomial_count), dtype=cell_points.dtype)
    dof_matrix = dof_matrix.at[:, :arity, :].set(
        basis.evaluate(
            geometry.vertices, geometry.centroids, geometry.characteristic_lengths
        )
    )
    local_points = jnp.zeros((cells, local_dofs, 2), dtype=cell_points.dtype)
    local_point_valid = jnp.zeros((cells, local_dofs), dtype=bool)
    local_points = local_points.at[:, :arity].set(geometry.vertices)
    local_point_valid = local_point_valid.at[:, :arity].set(True)
    edge_width = element.edge_interior_dof_count
    cursor = arity
    for edge in range(arity):
        if edge_width:
            dof_matrix = dof_matrix.at[:, cursor : cursor + edge_width].set(
                edge_basis[:, edge, 1:-1]
            )
            local_points = local_points.at[:, cursor : cursor + edge_width].set(
                edge_points[:, edge, 1:-1]
            )
            local_point_valid = local_point_valid.at[:, cursor : cursor + edge_width].set(
                True
            )
            cursor += edge_width

    moment_indices = total_degree_multiindices(2, degree - 2) if degree >= 2 else ()
    exponent_to_polynomial = {
        tuple(int(value) for value in exponent): index
        for index, exponent in enumerate(np.asarray(basis.exponents))
    }
    if moment_indices:
        moment_columns = jnp.asarray(
            tuple(exponent_to_polynomial[index] for index in moment_indices),
            dtype=jnp.int32,
        )
        moments = ein.contract(
            "cq,cqa,cqb->cab",
            cell_weights / geometry.areas[:, None],
            basis_values[..., moment_columns],
            basis_values,
        )
        dof_matrix = dof_matrix.at[:, cursor : cursor + len(moment_indices)].set(moments)

    boundary = ein.contract(
        "cenad,ced,n,ce->cena",
        edge_gradient,
        geometry.outward_normals,
        edge_weights,
        geometry.edge_lengths,
    )
    functionals = jnp.zeros(
        (cells, polynomial_count, local_dofs), dtype=cell_points.dtype
    )
    edge_cursor = arity
    for edge in range(arity):
        local_indices = [edge]
        local_indices.extend(range(edge_cursor, edge_cursor + edge_width))
        local_indices.append((edge + 1) % arity)
        for node, local_index in enumerate(local_indices):
            functionals = functionals.at[:, :, local_index].add(
                boundary[:, edge, node, :]
            )
        edge_cursor += edge_width
    exponents = np.asarray(basis.exponents, dtype=np.int32)
    scale_squared = geometry.characteristic_lengths * geometry.characteristic_lengths
    moment_start = arity + arity * edge_width
    for alpha, exponent in enumerate(exponents):
        if alpha == 0:
            continue
        for axis in range(2):
            power = int(exponent[axis])
            if power < 2:
                continue
            reduced = list(int(value) for value in exponent)
            reduced[axis] -= 2
            beta = moment_indices.index(tuple(reduced))
            coefficient = power * (power - 1) / scale_squared
            functionals = functionals.at[:, alpha, moment_start + beta].add(
                -geometry.areas * coefficient
            )
    functionals = functionals.at[:, 0].set(0.0)
    if degree == 1:
        functionals = functionals.at[:, 0, :arity].set(1.0 / arity)
    else:
        functionals = functionals.at[:, 0, moment_start].set(1.0)

    augmented_gram = ein.contract("cai,cib->cab", functionals, dof_matrix)
    gradient_gram = ein.contract(
        "cq,cqad,cqbd->cab", cell_weights, basis_gradients, basis_gradients
    )
    energy_factorization = prepare_local_block_factorization(augmented_gram)
    h1_coefficients, h1_failed = solve_local_blocks(energy_factorization, functionals)
    h1_coefficients = eqx.error_if(
        h1_coefficients,
        jnp.any(h1_failed | ~geometry.evidence.valid),
        "Virtual-element H1 projector factorization failed.",
    )
    h1_dof = ein.contract("cia,caj->cij", dof_matrix, h1_coefficients)

    mass_gram = ein.contract("cq,cqa,cqb->cab", cell_weights, basis_values, basis_values)
    l2_rhs = ein.contract("cab,cbj->caj", mass_gram, h1_coefficients)
    if moment_indices:
        for beta, exponent in enumerate(moment_indices):
            alpha = exponent_to_polynomial[exponent]
            l2_rhs = l2_rhs.at[:, alpha].set(0.0)
            l2_rhs = l2_rhs.at[:, alpha, moment_start + beta].set(geometry.areas)
    mass_factorization = prepare_local_block_factorization(
        mass_gram, positive_definite=True
    )
    l2_coefficients, l2_failed = solve_local_blocks(mass_factorization, l2_rhs)
    l2_coefficients = eqx.error_if(
        l2_coefficients,
        jnp.any(l2_failed | ~geometry.evidence.valid),
        "Virtual-element L2 projector factorization failed.",
    )
    l2_dof = ein.contract("cia,caj->cij", dof_matrix, l2_coefficients)

    identity = jnp.eye(polynomial_count, dtype=cell_points.dtype)
    h1_reproduction = ein.contract("cai,cib->cab", h1_coefficients, dof_matrix)
    l2_reproduction = ein.contract("cai,cib->cab", l2_coefficients, dof_matrix)
    h1_idempotence = ein.contract("cij,cjk->cik", h1_dof, h1_dof) - h1_dof
    l2_idempotence = ein.contract("cij,cjk->cik", l2_dof, l2_dof) - l2_dof
    g_singular = jnp.linalg.svd(augmented_gram, compute_uv=False)
    h_singular = jnp.linalg.svd(mass_gram, compute_uv=False)
    gradient_comparison = augmented_gram.at[:, 0].set(gradient_gram[:, 0])
    independent_gradient = gradient_gram.at[:, 0].set(augmented_gram[:, 0])
    evidence = VirtualElementProjectionEvidence(
        factorization_valid=~(h1_failed | l2_failed) & geometry.evidence.valid,
        g_bd_error=jnp.max(
            jnp.abs(gradient_comparison - independent_gradient), axis=(-2, -1)
        ),
        h1_reproduction_error=jnp.max(
            jnp.abs(h1_reproduction - identity[None]), axis=(-2, -1)
        ),
        l2_reproduction_error=jnp.max(
            jnp.abs(l2_reproduction - identity[None]), axis=(-2, -1)
        ),
        h1_idempotence_error=jnp.max(jnp.abs(h1_idempotence), axis=(-2, -1)),
        l2_idempotence_error=jnp.max(jnp.abs(l2_idempotence), axis=(-2, -1)),
        differential_reproduction_error=jnp.zeros((cells,), dtype=cell_points.dtype),
        minimum_g_singular_value=jnp.min(g_singular, axis=-1),
        minimum_h_singular_value=jnp.min(h_singular, axis=-1),
        maximum_g_condition=jnp.max(g_singular, axis=-1)
        / jnp.maximum(jnp.min(g_singular, axis=-1), jnp.finfo(cell_points.dtype).tiny),
        maximum_h_condition=jnp.max(h_singular, axis=-1)
        / jnp.maximum(jnp.min(h_singular, axis=-1), jnp.finfo(cell_points.dtype).tiny),
    )
    return VirtualElementProjectionData(
        family=element.family,
        projection_kind="H1",
        polynomial_value_shape=(),
        differential_kind=element.differential_kind,
        differential_degree=degree - 1,
        basis=basis,
        dof_matrix=dof_matrix,
        energy_functionals=functionals,
        augmented_gram=augmented_gram,
        gradient_gram=gradient_gram,
        mass_gram=mass_gram,
        h1_coefficients=h1_coefficients,
        h1_dof_projector=h1_dof,
        l2_coefficients=l2_coefficients,
        l2_dof_projector=l2_dof,
        differential_coefficients=jnp.zeros(
            (cells, 0, local_dofs), dtype=cell_points.dtype
        ),
        local_points=local_points,
        local_point_valid=local_point_valid,
        evidence=evidence,
        projection_id=canonical_fingerprint(
            {
                "kind": "virtual-element-projection",
                "geometry": geometry.geometry_id,
                "element": element.element_id,
                "cubature": cubature.cubature_id,
                "local_dofs": local_dofs,
                "polynomials": polynomial_count,
            }
        ),
    )


def _legendre_trace_data(degree: int, dtype, /):
    from ...integration import GaussLegendreRule, interval_rule_data

    data = interval_rule_data(GaussLegendreRule(degree + 1))
    nodes = 0.5 * (jnp.asarray(data.nodes, dtype=dtype) + 1.0)
    weights = 0.5 * jnp.asarray(data.weights, dtype=dtype)
    coordinate = 2.0 * nodes - 1.0
    values = [jnp.ones_like(coordinate)]
    if degree:
        values.append(coordinate)
    for order in range(2, degree + 1):
        values.append(
            ((2 * order - 1) * coordinate * values[-1] - (order - 1) * values[-2]) / order
        )
    return nodes, weights, jnp.stack(values, axis=-1)


def _polynomial_differential(
    basis: ScaledMonomialBasis,
    differential_basis: ScaledMonomialBasis,
    geometry: PolygonGeometry,
    family: str,
    /,
) -> Array:
    cells = int(geometry.vertices.shape[0])
    polynomial_count = basis.feature_count
    differential_count = differential_basis.feature_count
    result = jnp.zeros(
        (cells, differential_count, 2 * polynomial_count),
        dtype=geometry.vertices.dtype,
    )
    target = {
        tuple(int(value) for value in exponent): index
        for index, exponent in enumerate(np.asarray(differential_basis.exponents))
    }
    for alpha, exponent_array in enumerate(np.asarray(basis.exponents)):
        exponent = tuple(int(value) for value in exponent_array)
        for component in range(2):
            if family == "ConformingHdiv":
                axis = component
                sign = 1.0
            elif component == 0:
                axis = 1
                sign = -1.0
            else:
                axis = 0
                sign = 1.0
            power = exponent[axis]
            if not power:
                continue
            reduced = list(exponent)
            reduced[axis] -= 1
            beta = target[tuple(reduced)]
            result = result.at[:, beta, component * polynomial_count + alpha].set(
                sign * power / geometry.characteristic_lengths
            )
    return result


def _prepare_vector_virtual_element_projections(
    geometry: PolygonGeometry,
    cubature: PolygonCubature,
    element: VirtualElementSpec,
    /,
) -> VirtualElementProjectionData:
    cells, arity, _ = geometry.vertices.shape
    degree = element.degree
    edge_width = element.edge_dofs_per_entity
    basis = ScaledMonomialBasis(2, degree)
    differential_basis = ScaledMonomialBasis(2, degree - 1)
    polynomial_count = basis.feature_count
    differential_count = differential_basis.feature_count
    vector_polynomial_count = 2 * polynomial_count
    local_dofs = element.local_dof_count(arity)
    cell_points = cubature.points
    cell_weights = cubature.weights
    basis_values = basis.evaluate(
        cell_points, geometry.centroids, geometry.characteristic_lengths
    )
    differential_values = differential_basis.evaluate(
        cell_points, geometry.centroids, geometry.characteristic_lengths
    )
    nodes, edge_weights, legendre = _legendre_trace_data(degree, cell_points.dtype)
    start = geometry.vertices
    stop = jnp.roll(geometry.vertices, -1, axis=1)
    edge_points = (1.0 - nodes[None, None, :, None]) * start[:, :, None, :] + nodes[
        None, None, :, None
    ] * stop[:, :, None, :]
    edge_basis = basis.evaluate(
        edge_points.reshape((cells, arity * (degree + 1), 2)),
        geometry.centroids,
        geometry.characteristic_lengths,
    ).reshape((cells, arity, degree + 1, polynomial_count))
    edge_differential_basis = differential_basis.evaluate(
        edge_points.reshape((cells, arity * (degree + 1), 2)),
        geometry.centroids,
        geometry.characteristic_lengths,
    ).reshape((cells, arity, degree + 1, differential_count))
    if element.trace_kind == "normal":
        trace_directions = geometry.outward_normals
    else:
        trace_directions = geometry.edge_vectors / geometry.edge_lengths[..., None]
    differential_kind = element.differential_kind

    dof_matrix = jnp.zeros(
        (cells, local_dofs, vector_polynomial_count), dtype=cell_points.dtype
    )
    edge_scalar_moments = ein.contract(
        "q,qm,ceqp->cemp", edge_weights, legendre, edge_basis
    )
    edge_vector_moments = (
        edge_scalar_moments[..., None, :] * trace_directions[:, :, None, :, None]
    )
    edge_dof_count = arity * edge_width
    dof_matrix = dof_matrix.at[:, :edge_dof_count].set(
        edge_vector_moments.reshape((cells, edge_dof_count, vector_polynomial_count))
    )
    cell_cross_mass = ein.contract(
        "cq,cqa,cqb->cab",
        cell_weights / geometry.areas[:, None],
        differential_values,
        basis_values,
    )
    for component in range(2):
        row_start = edge_dof_count + component * differential_count
        column_start = component * polynomial_count
        dof_matrix = dof_matrix.at[
            :,
            row_start : row_start + differential_count,
            column_start : column_start + polynomial_count,
        ].set(cell_cross_mass)

    energy_functionals = jnp.swapaxes(dof_matrix, -1, -2)
    augmented_gram = ein.contract("cia,cib->cab", dof_matrix, dof_matrix)
    projection_factorization = prepare_local_block_factorization(
        augmented_gram, positive_definite=True
    )
    preliminary_coefficients, preliminary_failed = solve_local_blocks(
        projection_factorization, energy_functionals
    )
    preliminary_coefficients = eqx.error_if(
        preliminary_coefficients,
        jnp.any(preliminary_failed | ~geometry.evidence.valid),
        f"Virtual-element {element.conformity} functional projector failed.",
    )

    scalar_mass = ein.contract(
        "cq,cqa,cqb->cab", cell_weights, basis_values, basis_values
    )
    mass_gram = jnp.zeros(
        (cells, vector_polynomial_count, vector_polynomial_count),
        dtype=cell_points.dtype,
    )
    for component in range(2):
        start_column = component * polynomial_count
        mass_gram = mass_gram.at[
            :,
            start_column : start_column + polynomial_count,
            start_column : start_column + polynomial_count,
        ].set(scalar_mass)
    l2_rhs = ein.contract("cab,cbj->caj", mass_gram, preliminary_coefficients)
    exponent_to_differential = {
        tuple(int(value) for value in exponent): index
        for index, exponent in enumerate(np.asarray(differential_basis.exponents))
    }
    for alpha, exponent_array in enumerate(np.asarray(basis.exponents)):
        exponent = tuple(int(value) for value in exponent_array)
        if exponent not in exponent_to_differential:
            continue
        beta = exponent_to_differential[exponent]
        for component in range(2):
            row = component * polynomial_count + alpha
            column = edge_dof_count + component * differential_count + beta
            l2_rhs = l2_rhs.at[:, row].set(0.0)
            l2_rhs = l2_rhs.at[:, row, column].set(geometry.areas)
    mass_factorization = prepare_local_block_factorization(
        mass_gram, positive_definite=True
    )
    l2_coefficients, l2_failed = solve_local_blocks(mass_factorization, l2_rhs)
    l2_coefficients = eqx.error_if(
        l2_coefficients,
        jnp.any(l2_failed | ~geometry.evidence.valid),
        f"Virtual-element {element.conformity} enhanced L2 projector failed.",
    )
    l2_dof = ein.contract("cia,caj->cij", dof_matrix, l2_coefficients)

    differential_rhs = jnp.zeros(
        (cells, differential_count, local_dofs), dtype=cell_points.dtype
    )
    edge_test_moments = ein.contract(
        "q,qm,ceqa->cema",
        edge_weights,
        legendre,
        edge_differential_basis,
    )
    dual_factors = 2 * jnp.arange(edge_width, dtype=cell_points.dtype) + 1
    edge_dual_coefficients = edge_test_moments * dual_factors[None, None, :, None]
    for edge in range(arity):
        edge_start = edge * edge_width
        boundary_functionals = geometry.edge_lengths[:, edge, None, None] * jnp.swapaxes(
            edge_dual_coefficients[:, edge], -1, -2
        )
        differential_rhs = differential_rhs.at[
            :, :, edge_start : edge_start + edge_width
        ].set(boundary_functionals)

    exponent_to_differential = {
        tuple(int(value) for value in exponent): index
        for index, exponent in enumerate(np.asarray(differential_basis.exponents))
    }
    for alpha, exponent_array in enumerate(np.asarray(differential_basis.exponents)):
        exponent = tuple(int(value) for value in exponent_array)
        for axis in range(2):
            power = exponent[axis]
            if not power:
                continue
            reduced = list(exponent)
            reduced[axis] -= 1
            beta = exponent_to_differential[tuple(reduced)]
            if element.family == "ConformingHdiv":
                component = axis
                sign = -1.0
            elif axis == 0:
                component = 1
                sign = -1.0
            else:
                component = 0
                sign = 1.0
            column = edge_dof_count + component * differential_count + beta
            differential_rhs = differential_rhs.at[:, alpha, column].add(
                sign * geometry.areas * power / geometry.characteristic_lengths
            )
    differential_mass = ein.contract(
        "cq,cqa,cqb->cab",
        cell_weights,
        differential_values,
        differential_values,
    )
    differential_factorization = prepare_local_block_factorization(
        differential_mass, positive_definite=True
    )
    differential_coefficients, differential_failed = solve_local_blocks(
        differential_factorization, differential_rhs
    )
    differential_coefficients = eqx.error_if(
        differential_coefficients,
        jnp.any(differential_failed | ~geometry.evidence.valid),
        f"Virtual-element {differential_kind} projector factorization failed.",
    )

    polynomial_differential = _polynomial_differential(
        basis, differential_basis, geometry, element.family
    )
    gradient_gram = ein.contract(
        "cai,cab,cbj->cij",
        polynomial_differential,
        differential_mass,
        polynomial_differential,
    )
    identity = jnp.eye(vector_polynomial_count, dtype=cell_points.dtype)
    reproduction = ein.contract("cai,cib->cab", l2_coefficients, dof_matrix)
    differential_reproduction = ein.contract(
        "cai,cib->cab", differential_coefficients, dof_matrix
    )
    idempotence = ein.contract("cij,cjk->cik", l2_dof, l2_dof) - l2_dof
    g_singular = jnp.linalg.svd(augmented_gram, compute_uv=False)
    h_singular = jnp.linalg.svd(mass_gram, compute_uv=False)
    tiny = jnp.finfo(cell_points.dtype).tiny
    differential_error = jnp.max(
        jnp.abs(differential_reproduction - polynomial_differential),
        axis=(-2, -1),
    )
    evidence = VirtualElementProjectionEvidence(
        factorization_valid=~(preliminary_failed | l2_failed | differential_failed)
        & geometry.evidence.valid,
        g_bd_error=differential_error,
        h1_reproduction_error=jnp.zeros((cells,), dtype=cell_points.dtype),
        l2_reproduction_error=jnp.max(
            jnp.abs(reproduction - identity[None]), axis=(-2, -1)
        ),
        h1_idempotence_error=jnp.zeros((cells,), dtype=cell_points.dtype),
        l2_idempotence_error=jnp.max(jnp.abs(idempotence), axis=(-2, -1)),
        differential_reproduction_error=differential_error,
        minimum_g_singular_value=jnp.min(g_singular, axis=-1),
        minimum_h_singular_value=jnp.min(h_singular, axis=-1),
        maximum_g_condition=jnp.max(g_singular, axis=-1)
        / jnp.maximum(jnp.min(g_singular, axis=-1), tiny),
        maximum_h_condition=jnp.max(h_singular, axis=-1)
        / jnp.maximum(jnp.min(h_singular, axis=-1), tiny),
    )
    return VirtualElementProjectionData(
        family=element.family,
        projection_kind="vector_L2",
        polynomial_value_shape=(2,),
        differential_kind=differential_kind,
        differential_degree=degree - 1,
        basis=basis,
        dof_matrix=dof_matrix,
        energy_functionals=energy_functionals,
        augmented_gram=augmented_gram,
        gradient_gram=gradient_gram,
        mass_gram=mass_gram,
        h1_coefficients=jnp.zeros((cells, 0, local_dofs), dtype=cell_points.dtype),
        h1_dof_projector=jnp.zeros((cells, 0, 0), dtype=cell_points.dtype),
        l2_coefficients=l2_coefficients,
        l2_dof_projector=l2_dof,
        differential_coefficients=differential_coefficients,
        local_points=jnp.zeros((cells, local_dofs, 2), dtype=cell_points.dtype),
        local_point_valid=jnp.zeros((cells, local_dofs), dtype=bool),
        evidence=evidence,
        projection_id=canonical_fingerprint(
            {
                "kind": "virtual-element-vector-projection",
                "geometry": geometry.geometry_id,
                "element": element.element_id,
                "cubature": cubature.cubature_id,
                "local_dofs": local_dofs,
                "polynomials": vector_polynomial_count,
                "differential": differential_kind,
            }
        ),
    )


def _prepare_discontinuous_l2_virtual_element_projections(
    geometry: PolygonGeometry,
    cubature: PolygonCubature,
    element: VirtualElementSpec,
    /,
) -> VirtualElementProjectionData:
    cells, arity, _ = geometry.vertices.shape
    basis = ScaledMonomialBasis(2, element.degree)
    polynomial_count = basis.feature_count
    local_dofs = element.local_dof_count(arity)
    cell_points = cubature.points
    basis_values = basis.evaluate(
        cell_points, geometry.centroids, geometry.characteristic_lengths
    )
    mass_gram = ein.contract(
        "cq,cqa,cqb->cab", cubature.weights, basis_values, basis_values
    )
    dof_matrix = mass_gram / geometry.areas[:, None, None]
    identity = jnp.broadcast_to(
        jnp.eye(polynomial_count, dtype=cell_points.dtype),
        (cells, polynomial_count, polynomial_count),
    )
    factorization = prepare_local_block_factorization(dof_matrix, positive_definite=True)
    l2_coefficients, failed = solve_local_blocks(factorization, identity)
    l2_coefficients = eqx.error_if(
        l2_coefficients,
        jnp.any(failed | ~geometry.evidence.valid),
        "Discontinuous virtual-element L2 projector factorization failed.",
    )
    l2_dof = ein.contract("cia,caj->cij", dof_matrix, l2_coefficients)
    reproduction = ein.contract("cai,cib->cab", l2_coefficients, dof_matrix)
    idempotence = ein.contract("cij,cjk->cik", l2_dof, l2_dof) - l2_dof
    singular = jnp.linalg.svd(dof_matrix, compute_uv=False)
    mass_singular = jnp.linalg.svd(mass_gram, compute_uv=False)
    tiny = jnp.finfo(cell_points.dtype).tiny
    evidence = VirtualElementProjectionEvidence(
        factorization_valid=~failed & geometry.evidence.valid,
        g_bd_error=jnp.zeros((cells,), dtype=cell_points.dtype),
        h1_reproduction_error=jnp.zeros((cells,), dtype=cell_points.dtype),
        l2_reproduction_error=jnp.max(jnp.abs(reproduction - identity), axis=(-2, -1)),
        h1_idempotence_error=jnp.zeros((cells,), dtype=cell_points.dtype),
        l2_idempotence_error=jnp.max(jnp.abs(idempotence), axis=(-2, -1)),
        differential_reproduction_error=jnp.zeros((cells,), dtype=cell_points.dtype),
        minimum_g_singular_value=jnp.min(singular, axis=-1),
        minimum_h_singular_value=jnp.min(mass_singular, axis=-1),
        maximum_g_condition=jnp.max(singular, axis=-1)
        / jnp.maximum(jnp.min(singular, axis=-1), tiny),
        maximum_h_condition=jnp.max(mass_singular, axis=-1)
        / jnp.maximum(jnp.min(mass_singular, axis=-1), tiny),
    )
    return VirtualElementProjectionData(
        family=element.family,
        projection_kind="L2",
        polynomial_value_shape=(),
        differential_kind="none",
        differential_degree=-1,
        basis=basis,
        dof_matrix=dof_matrix,
        energy_functionals=identity,
        augmented_gram=dof_matrix,
        gradient_gram=jnp.zeros_like(mass_gram),
        mass_gram=mass_gram,
        h1_coefficients=jnp.zeros((cells, 0, local_dofs), dtype=cell_points.dtype),
        h1_dof_projector=jnp.zeros((cells, 0, 0), dtype=cell_points.dtype),
        l2_coefficients=l2_coefficients,
        l2_dof_projector=l2_dof,
        differential_coefficients=jnp.zeros(
            (cells, 0, local_dofs), dtype=cell_points.dtype
        ),
        local_points=jnp.zeros((cells, local_dofs, 2), dtype=cell_points.dtype),
        local_point_valid=jnp.zeros((cells, local_dofs), dtype=bool),
        evidence=evidence,
        projection_id=canonical_fingerprint(
            {
                "kind": "virtual-element-discontinuous-l2-projection",
                "geometry": geometry.geometry_id,
                "element": element.element_id,
                "cubature": cubature.cubature_id,
                "local_dofs": local_dofs,
                "polynomials": polynomial_count,
            }
        ),
    )


def prepare_virtual_element_projections(
    geometry: PolygonGeometry,
    cubature: PolygonCubature,
    element: VirtualElementSpec,
    /,
) -> VirtualElementProjectionData:
    """Prepare the canonical projector for the requested virtual-element family."""

    if element.family == "ConformingH1":
        return _prepare_h1_virtual_element_projections(geometry, cubature, element)
    if element.family in ("ConformingHdiv", "ConformingHcurl"):
        return _prepare_vector_virtual_element_projections(geometry, cubature, element)
    return _prepare_discontinuous_l2_virtual_element_projections(
        geometry, cubature, element
    )


__all__ = [
    "VirtualElementProjectionData",
    "VirtualElementProjectionEvidence",
    "prepare_virtual_element_projections",
]
