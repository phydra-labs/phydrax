#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array

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
    minimum_g_singular_value: Array
    minimum_h_singular_value: Array
    maximum_g_condition: Array
    maximum_h_condition: Array

    @property
    def passed(self) -> Array:
        return jnp.all(self.factorization_valid)


class VirtualElementProjectionData(StrictModule):
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


def prepare_virtual_element_projections(
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
        moments = oe.contract(
            "cq,cqa,cqb->cab",
            cell_weights / geometry.areas[:, None],
            basis_values[..., moment_columns],
            basis_values,
        )
        dof_matrix = dof_matrix.at[:, cursor : cursor + len(moment_indices)].set(moments)

    boundary = oe.contract(
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

    augmented_gram = oe.contract("cai,cib->cab", functionals, dof_matrix)
    gradient_gram = oe.contract(
        "cq,cqad,cqbd->cab", cell_weights, basis_gradients, basis_gradients
    )
    energy_factorization = prepare_local_block_factorization(augmented_gram)
    h1_coefficients, h1_failed = solve_local_blocks(energy_factorization, functionals)
    h1_coefficients = eqx.error_if(
        h1_coefficients,
        jnp.any(h1_failed | ~geometry.evidence.valid),
        "Virtual-element H1 projector factorization failed.",
    )
    h1_dof = oe.contract("cia,caj->cij", dof_matrix, h1_coefficients)

    mass_gram = oe.contract("cq,cqa,cqb->cab", cell_weights, basis_values, basis_values)
    l2_rhs = oe.contract("cab,cbj->caj", mass_gram, h1_coefficients)
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
    l2_dof = oe.contract("cia,caj->cij", dof_matrix, l2_coefficients)

    identity = jnp.eye(polynomial_count, dtype=cell_points.dtype)
    h1_reproduction = oe.contract("cai,cib->cab", h1_coefficients, dof_matrix)
    l2_reproduction = oe.contract("cai,cib->cab", l2_coefficients, dof_matrix)
    h1_idempotence = oe.contract("cij,cjk->cik", h1_dof, h1_dof) - h1_dof
    l2_idempotence = oe.contract("cij,cjk->cik", l2_dof, l2_dof) - l2_dof
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
        minimum_g_singular_value=jnp.min(g_singular, axis=-1),
        minimum_h_singular_value=jnp.min(h_singular, axis=-1),
        maximum_g_condition=jnp.max(g_singular, axis=-1)
        / jnp.maximum(jnp.min(g_singular, axis=-1), jnp.finfo(cell_points.dtype).tiny),
        maximum_h_condition=jnp.max(h_singular, axis=-1)
        / jnp.maximum(jnp.min(h_singular, axis=-1), jnp.finfo(cell_points.dtype).tiny),
    )
    return VirtualElementProjectionData(
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


__all__ = [
    "VirtualElementProjectionData",
    "VirtualElementProjectionEvidence",
    "prepare_virtual_element_projections",
]
