#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Superconvergent patch recovery and goal-oriented indicators for Lagrange fields.

Gradient recovery follows Zienkiewicz and Zhu (1992): on every vertex patch a
polynomial of the field degree is least-squares fitted to the finite-element
gradient at the elementwise superconvergent sampling points (centroids for P1,
the degree-two Gauss points for P2), and evaluated at the degree-of-freedom
nodes; nodes on edges average the fits of their endpoint patches. The Hessian is
the recovery of the recovered gradient. Patch geometry, routes, and the batched
normal-equation factorization are prepared once and reused by every field.
"""

from __future__ import annotations

import itertools
import math
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DensePropertyVerificationPolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    LocalBlockFactorization,
    prepare_local_block_factorization,
    PreparedLinearSolve,
    solve_adjoint,
    solve_local_blocks_detailed,
    verify_dense_properties,
)
from ...sparse import gather_routes, RowRelation
from .._reference_cell import reference_cell_topology
from ._adaptivity import (
    FiniteElementDWRIndicators,
    FiniteElementErrorEstimate,
    local_dual_weighted_residual,
)
from ._generic import (
    _degree_aware_reference_rule,
    FiniteElementDiscretization,
    FiniteElementRuntimeData,
)
from ._reference import FiniteElementSpec, lagrange_element


_PATCH_PROPERTY_POLICY = DensePropertyVerificationPolicy(require_positive_definite=True)
_TETRAHEDRON_GAUSS_LOW = 0.1381966011250105
_TETRAHEDRON_GAUSS_HIGH = 0.5854101966249685
# Elementwise superconvergent gradient sampling points (reference coordinates).
_SAMPLE_POINTS = {
    ("triangle", 1): ((1.0 / 3.0, 1.0 / 3.0),),
    ("triangle", 2): (
        (1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0),
    ),
    ("tetrahedron", 1): ((0.25, 0.25, 0.25),),
    ("tetrahedron", 2): (
        (_TETRAHEDRON_GAUSS_LOW,) * 3,
        (_TETRAHEDRON_GAUSS_HIGH, _TETRAHEDRON_GAUSS_LOW, _TETRAHEDRON_GAUSS_LOW),
        (_TETRAHEDRON_GAUSS_LOW, _TETRAHEDRON_GAUSS_HIGH, _TETRAHEDRON_GAUSS_LOW),
        (_TETRAHEDRON_GAUSS_LOW, _TETRAHEDRON_GAUSS_LOW, _TETRAHEDRON_GAUSS_HIGH),
    ),
}


def _exponents(dimension: int, degree: int, /) -> np.ndarray:
    """Monomial exponents of total degree at most ``degree``, graded order."""
    exponents = [
        powers
        for total in range(degree + 1)
        for powers in itertools.product(range(total + 1), repeat=dimension)
        if sum(powers) == total
    ]
    return np.asarray(exponents, dtype=np.int32)


def _monomials(points: Array, exponents: np.ndarray, /) -> Array:
    return jnp.prod(points[..., None, :] ** jnp.asarray(exponents), axis=-1)


def _field_elements(
    discretization: FiniteElementDiscretization, field_name: str, /
) -> tuple[int, tuple[FiniteElementSpec, ...]]:
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field = discretization._field_index(field_name)
    if discretization.dof_maps[field].component_shape != ():
        raise ValueError("Recovery requires a scalar finite-element field.")
    return field, tuple(discretization.elements[field])


def _validate_lagrange(elements: tuple[FiniteElementSpec, ...], /) -> tuple[str, int]:
    kinds = {(element.cell_kind, element.degree) for element in elements}
    if len(kinds) != 1:
        raise ValueError("Recovery requires one simplex kind and degree on every block.")
    kind, degree = next(iter(kinds))
    if (kind, degree) not in _SAMPLE_POINTS or any(
        element.conformity != "H1"
        or element.mapping != "identity"
        or element.value_shape != ()
        or any(entity for entities in element.entity_dofs[2:] for entity in entities)
        for element in elements
    ):
        raise ValueError(
            "Recovery supports scalar P1/P2 Lagrange fields on triangles and tetrahedra."
        )
    return kind, degree


def _padded_routes(
    rows: np.ndarray, columns: np.ndarray, count: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic fixed-capacity routes from unique ``(row, column)`` pairs."""
    pairs = np.unique(np.stack((rows, columns), axis=1), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=count)
    capacity = max(1, int(np.max(counts, initial=0)))
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    positions = np.arange(pairs.shape[0]) - np.repeat(offsets, counts)
    routes = np.zeros((count, capacity), dtype=np.int32)
    valid = np.zeros((count, capacity), dtype=np.bool_)
    routes[pairs[:, 0], positions] = pairs[:, 1]
    valid[pairs[:, 0], positions] = True
    return routes, valid


@partial(jax.jit, static_argnames=("dimension", "degree"))
def _patch_normal_equations(
    sample_points: Array,
    centers: Array,
    routes: Array,
    valid: Array,
    *,
    dimension: int,
    degree: int,
) -> tuple[Array, Array, Array]:
    relation = RowRelation(routes, source_size=sample_points.shape[0], valid=valid)
    offsets = gather_routes(relation, sample_points) - centers[:, None, :]
    distance = jnp.sqrt(jnp.sum(offsets**2, axis=-1))
    radius = jnp.max(jnp.where(valid, distance, 0.0), axis=1)
    safe_radius = jnp.where(radius > 0.0, radius, 1.0)
    # Scaled local coordinates keep the normal equations well conditioned.
    design = jnp.where(
        valid[..., None],
        _monomials(offsets / safe_radius[:, None, None], _exponents(dimension, degree)),
        0.0,
    )
    normal = contract("vpm,vpn->vmn", design, design)
    return design, safe_radius, normal


class _RecoveryBlock(StrictModule):
    dof_relation: RowRelation
    sample_gradients: Array
    quadrature_basis: Array
    quadrature_gradients: Array
    quadrature_weights: Array


class PreparedGradientRecovery(StrictModule, NonTrainableState):
    """Prepared vertex patches, sampling geometry, and patch factorizations."""

    blocks: tuple[_RecoveryBlock, ...]
    patch_samples: RowRelation
    patch_design: Array
    patch_centers: Array
    patch_radii: Array
    factorization: LocalBlockFactorization
    node_owners: RowRelation
    node_design: Array
    node_weights: Array
    field_name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    dof_count: int = eqx.field(static=True)
    patch_count: int = eqx.field(static=True)
    extended_patch_count: int = eqx.field(static=True)
    minimum_patch_samples: int = eqx.field(static=True)
    maximum_patch_condition: float = eqx.field(static=True)
    recovery_id: str = eqx.field(static=True)


class FiniteElementRecoveryEvidence(StrictModule, NonTrainableState):
    """Patch structure and runtime failures of one recovery.

    ``extended_patch_count`` counts vertex patches enlarged to their two-ring
    because the one-ring fit was rank deficient. ``failed_patch_count`` counts
    patch solves that produced non-finite coefficients at runtime.
    ``maximum_asymmetry`` is the largest entry of ``|H - H^T|`` before the
    recovered Hessian is replaced by its symmetric part (zero for gradients).
    """

    recovery_id: str = eqx.field(static=True)
    patch_count: int = eqx.field(static=True)
    extended_patch_count: int = eqx.field(static=True)
    minimum_patch_samples: int = eqx.field(static=True)
    maximum_patch_condition: float = eqx.field(static=True)
    failed_patch_count: int = eqx.field(static=True)
    maximum_asymmetry: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedGradientRecovery,
        /,
        *,
        failed_patch_count: int,
        maximum_asymmetry: float,
    ) -> None:
        self.recovery_id = prepared.recovery_id
        self.patch_count = prepared.patch_count
        self.extended_patch_count = prepared.extended_patch_count
        self.minimum_patch_samples = prepared.minimum_patch_samples
        self.maximum_patch_condition = prepared.maximum_patch_condition
        self.failed_patch_count = int(failed_patch_count)
        self.maximum_asymmetry = float(maximum_asymmetry)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "finite-element-recovery-evidence",
                "recovery": prepared.recovery_id,
                "failed_patches": self.failed_patch_count,
                "maximum_asymmetry": self.maximum_asymmetry,
            }
        )

    @property
    def passed(self) -> bool:
        return self.failed_patch_count == 0


def _block_geometry(
    discretization: Any, field_name: Any, block_index: Any, coordinates: Any, points: Any
) -> Any:
    points_ = jnp.asarray(points)
    return discretization.evaluate_block_geometry(
        field_name,
        block_index,
        coordinates,
        points_,
        jnp.ones((points_.shape[0],), dtype=points_.dtype),
    )


def _node_ownership(
    discretization: Any, field: Any, elements: Any, coordinates: Any, field_name: Any, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Owner vertices and physical positions of every DOF node and vertex."""
    dof_map = discretization.dof_maps[field]
    mesh = discretization.mesh
    owners = np.full((dof_map.global_dof_count, 2), -1, dtype=np.int64)
    positions = np.full((dof_map.global_dof_count, mesh.coordinates.shape[1]), np.nan)
    vertex_positions = np.full(mesh.coordinates.shape, np.nan)
    for block_index, element in enumerate(elements):
        block = mesh.block(dof_map.block_names[block_index])
        vertices = np.asarray(block.vertices, dtype=np.int64)
        cell_dofs = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        nodes = np.asarray(
            _block_geometry(
                discretization,
                field_name,
                block_index,
                coordinates,
                element.reference_nodes,
            ).physical_points
        )
        topology = reference_cell_topology(element.cell_kind)
        for dimension in (0, 1):
            for entity, dofs in enumerate(element.entity_dofs[dimension]):
                local_vertices = topology.entities[dimension][entity]
                for dof in dofs:
                    rows = cell_dofs[:, dof]
                    owners[rows, 0] = vertices[:, local_vertices[0]]
                    if dimension == 1:
                        owners[rows, 1] = vertices[:, local_vertices[1]]
                    positions[rows] = nodes[:, dof]
                    if dimension == 0:
                        vertex_positions[vertices[:, local_vertices[0]]] = nodes[:, dof]
    if np.any(owners[:, 0] < 0) or not np.all(np.isfinite(positions)):
        raise ValueError("Every recovered DOF must lie on a vertex or an edge.")
    return owners, positions, vertex_positions


def _vertex_patches(
    local: np.ndarray, count: int, /
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """One-ring and two-ring vertex-to-cell pairs from patch-row cell vertices."""
    cells = np.repeat(np.arange(local.shape[0]), local.shape[1])
    one_ring = (local.reshape(-1), cells)
    routes, valid = _padded_routes(*one_ring, count)
    first, second = np.meshgrid(
        np.arange(local.shape[1]), np.arange(local.shape[1]), indexing="ij"
    )
    neighbors = np.unique(
        np.stack((local[:, first.ravel()].ravel(), local[:, second.ravel()].ravel()), 1),
        axis=0,
    )
    width = routes.shape[1]
    expanded_rows = np.repeat(neighbors[:, 0], width)
    expanded_cells = routes[neighbors[:, 1]].reshape(-1)
    expanded_valid = valid[neighbors[:, 1]].reshape(-1)
    two_ring = (expanded_rows[expanded_valid], expanded_cells[expanded_valid])
    return one_ring, two_ring


def _patch_system(
    pairs: Any,
    samples_per_cell: Any,
    sample_points: Any,
    centers: Any,
    dimension: Any,
    degree: Any,
    condition: Any,
) -> Any:
    rows, cells = pairs
    sample_rows = np.repeat(rows, samples_per_cell)
    sample_columns = (
        cells[:, None] * samples_per_cell + np.arange(samples_per_cell)[None, :]
    ).reshape(-1)
    routes, valid = _padded_routes(sample_rows, sample_columns, centers.shape[0])
    design, radii, normal = _patch_normal_equations(
        sample_points,
        centers,
        jnp.asarray(routes),
        jnp.asarray(valid),
        dimension=dimension,
        degree=degree,
    )
    # Rank and conditioning of every patch fit come from the native property
    # substrate; a Cholesky pivot test alone admits numerically singular fits.
    properties = verify_dense_properties(normal, policy=_PATCH_PROPERTY_POLICY)
    estimates = np.asarray(properties.condition_estimate)
    admissible = np.asarray(properties.successful) & (estimates <= condition)
    factorization = prepare_local_block_factorization(normal, positive_definite=True)
    return routes, valid, design, radii, factorization, admissible, estimates


def prepare_gradient_recovery(
    discretization: FiniteElementDiscretization,
    field_name: str,
    /,
    *,
    runtime: FiniteElementRuntimeData | None = None,
    maximum_patch_condition: float = 1.0e10,
) -> PreparedGradientRecovery:
    """Prepare fixed-capacity vertex patches for superconvergent patch recovery.

    A vertex patch is the one-ring of cells; patches whose scaled least-squares
    normal matrix is singular or has a condition estimate above
    ``maximum_patch_condition`` (typically boundary corners) are enlarged to the
    two-ring. A patch that remains inadmissible raises ``ValueError``.
    """
    condition = float(maximum_patch_condition)
    if not math.isfinite(condition) or condition <= 1.0:
        raise ValueError("maximum_patch_condition must be finite and above one.")
    field, elements = _field_elements(discretization, field_name)
    kind, degree = _validate_lagrange(elements)
    runtime_ = discretization.default_runtime if runtime is None else runtime
    if not isinstance(runtime_, FiniteElementRuntimeData):
        raise TypeError("runtime must be FiniteElementRuntimeData or None.")
    coordinates = runtime_.coordinates
    dof_map = discretization.dof_maps[field]
    mesh = discretization.mesh
    dimension = mesh.coordinates.shape[1]
    if elements[0].topological_dimension != dimension:
        raise ValueError("Recovery requires full-dimensional simplex cells.")
    samples = jnp.asarray(_SAMPLE_POINTS[(kind, degree)])
    samples_per_cell = samples.shape[0]
    quadrature_points, quadrature_weights = _degree_aware_reference_rule(kind, 2 * degree)
    blocks = []
    sample_points = []
    cell_vertices = []
    for block_index, element in enumerate(elements):
        block = mesh.block(dof_map.block_names[block_index])
        sampled = _block_geometry(
            discretization, field_name, block_index, coordinates, samples
        )
        quadrature = discretization.evaluate_block_geometry(
            field_name, block_index, coordinates, quadrature_points, quadrature_weights
        )
        blocks.append(
            _RecoveryBlock(
                dof_relation=dof_map.relations[block_index],
                sample_gradients=sampled.physical_gradients,
                quadrature_basis=quadrature.basis_values,
                quadrature_gradients=quadrature.physical_gradients,
                quadrature_weights=quadrature.physical_weights,
            )
        )
        sample_points.append(sampled.physical_points.reshape((-1, dimension)))
        cell_vertices.append(np.asarray(block.vertices, dtype=np.int64))
    owners, node_positions, vertex_positions = _node_ownership(
        discretization, field, elements, coordinates, field_name
    )
    active = np.unique(owners[owners >= 0])
    lookup = np.full((mesh.coordinates.shape[0],), -1, dtype=np.int64)
    lookup[active] = np.arange(active.size)
    one_ring, two_ring = _vertex_patches(
        lookup[np.concatenate(cell_vertices, axis=0)], active.size
    )
    points = jnp.concatenate(sample_points, axis=0)
    centers = jnp.asarray(vertex_positions[active])
    first = _patch_system(
        one_ring, samples_per_cell, points, centers, dimension, degree, condition
    )
    extended = ~first[5]
    if np.any(extended):
        chosen = np.isin(one_ring[0], np.flatnonzero(~extended))
        widened = np.isin(two_ring[0], np.flatnonzero(extended))
        pairs = (
            np.concatenate((one_ring[0][chosen], two_ring[0][widened])),
            np.concatenate((one_ring[1][chosen], two_ring[1][widened])),
        )
        final = _patch_system(
            pairs, samples_per_cell, points, centers, dimension, degree, condition
        )
    else:
        final = first
    routes, valid, design, radii, factorization, admissible, estimates = final
    if not np.all(admissible):
        raise ValueError(
            f"{int(np.count_nonzero(~admissible))} recovery patches remain singular or "
            "ill conditioned after two-ring enlargement."
        )
    owner_rows = lookup[np.maximum(owners, 0)]
    owner_valid = owners >= 0
    scaled = (
        node_positions[:, None, :] - vertex_positions[active][owner_rows]
    ) / np.asarray(radii)[owner_rows][..., None]
    node_design = np.where(
        owner_valid[..., None],
        np.asarray(_monomials(jnp.asarray(scaled), _exponents(dimension, degree))),
        0.0,
    )
    recovery_id = canonical_fingerprint(
        {
            "kind": "gradient-recovery",
            "discretization": discretization.prepared_id,
            "field": field_name,
            "coordinates": array_tree_fingerprint(np.asarray(coordinates)),
            "degree": degree,
            "routes": array_tree_fingerprint(np.where(valid, routes, -1)),
        }
    )
    return PreparedGradientRecovery(
        blocks=tuple(blocks),
        patch_samples=RowRelation(
            jnp.asarray(routes), source_size=points.shape[0], valid=jnp.asarray(valid)
        ),
        patch_design=design,
        patch_centers=centers,
        patch_radii=radii,
        factorization=factorization,
        node_owners=RowRelation(
            jnp.asarray(owner_rows, dtype=jnp.int32),
            source_size=active.size,
            valid=jnp.asarray(owner_valid),
        ),
        node_design=jnp.asarray(node_design),
        node_weights=jnp.asarray(
            owner_valid / np.sum(owner_valid, axis=1, keepdims=True)
        ),
        field_name=str(field_name),
        cell_kind=kind,
        degree=degree,
        dimension=dimension,
        dof_count=dof_map.global_dof_count,
        patch_count=active.size,
        extended_patch_count=int(np.count_nonzero(extended)),
        minimum_patch_samples=int(np.min(np.sum(valid, axis=1))),
        maximum_patch_condition=float(np.max(estimates)),
        recovery_id=recovery_id,
    )


@eqx.filter_jit
def _recover_nodes(prepared: PreparedGradientRecovery, values: Array, /) -> Any:
    """Recovered gradients ``(dofs, columns, d)`` of ``(dofs, columns)`` fields."""
    columns = values.shape[1]
    dimension = prepared.dimension
    samples = []
    for block in prepared.blocks:
        local = gather_routes(block.dof_relation, values)
        gradients = contract("csld,clq->csqd", block.sample_gradients, local)
        samples.append(gradients.reshape((-1, columns * dimension)))
    patch = gather_routes(prepared.patch_samples, jnp.concatenate(samples, axis=0))
    right = contract("vpm,vpk->vmk", prepared.patch_design, patch)
    solved = solve_local_blocks_detailed(prepared.factorization, right)
    coefficients = solved.value.reshape((solved.value.shape[0], -1))
    owners = gather_routes(prepared.node_owners, coefficients).reshape(
        prepared.node_design.shape + (columns * dimension,)
    )
    nodal = contract(
        "no,nom,nomk->nk", prepared.node_weights, prepared.node_design, owners
    )
    return nodal.reshape((-1, columns, dimension)), jnp.sum(
        solved.failed_blocks, dtype=jnp.int32
    )


def _coefficients(prepared: PreparedGradientRecovery, coefficients: ArrayLike, /) -> Any:
    if not isinstance(prepared, PreparedGradientRecovery):
        raise TypeError("prepared must be PreparedGradientRecovery.")
    values = jnp.asarray(coefficients)
    if not jnp.issubdtype(values.dtype, jnp.floating):
        values = values.astype(jnp.float64)
    if values.shape != (prepared.dof_count,):
        raise ValueError("coefficients must be one value per field DOF.")
    return values


def recover_gradient(
    prepared: PreparedGradientRecovery, coefficients: ArrayLike, /
) -> tuple[Array, FiniteElementRecoveryEvidence]:
    """Recovered gradient ``(dofs, d)`` in the field's own Lagrange space."""
    values = _coefficients(prepared, coefficients)
    gradient, failed = _recover_nodes(prepared, values[:, None])
    evidence = FiniteElementRecoveryEvidence(
        prepared, failed_patch_count=int(np.asarray(failed)), maximum_asymmetry=0.0
    )
    return gradient[:, 0], evidence


def recover_hessian(
    prepared: PreparedGradientRecovery, coefficients: ArrayLike, /
) -> tuple[Array, FiniteElementRecoveryEvidence]:
    """Recovered Hessian ``(dofs, d, d)``: recovery of the recovered gradient.

    Row ``k`` of the raw tensor is the recovered gradient of gradient component
    ``k``; the returned Hessian is its symmetric part and the evidence reports
    the removed asymmetry.
    """
    values = _coefficients(prepared, coefficients)
    gradient, first_failed = _recover_nodes(prepared, values[:, None])
    raw, second_failed = _recover_nodes(prepared, gradient[:, 0])
    asymmetry = jnp.max(jnp.abs(raw - jnp.swapaxes(raw, -1, -2)))
    evidence = FiniteElementRecoveryEvidence(
        prepared,
        failed_patch_count=int(np.asarray(first_failed + second_failed)),
        maximum_asymmetry=float(np.asarray(asymmetry)),
    )
    return 0.5 * (raw + jnp.swapaxes(raw, -1, -2)), evidence


@eqx.filter_jit
def _recovery_indicators(
    prepared: PreparedGradientRecovery, values: Array, recovered: Array, /
) -> Array:
    indicators = []
    for block in prepared.blocks:
        local = gather_routes(block.dof_relation, values)
        local_recovered = gather_routes(block.dof_relation, recovered)
        discrete = contract("cqld,cl->cqd", block.quadrature_gradients, local)
        smoothed = contract("ql,cld->cqd", block.quadrature_basis, local_recovered)
        difference = jnp.sum((smoothed - discrete) ** 2, axis=-1)
        indicators.append(
            jnp.sqrt(jnp.sum(block.quadrature_weights * difference, axis=1))
        )
    return jnp.concatenate(indicators)


def recovery_error_estimate(
    prepared: PreparedGradientRecovery, coefficients: ArrayLike, /
) -> tuple[FiniteElementErrorEstimate, FiniteElementRecoveryEvidence]:
    """Zienkiewicz-Zhu indicators ``||G_h - grad u_h||_{L2(K)}`` per cell."""
    values = _coefficients(prepared, coefficients)
    recovered, evidence = recover_gradient(prepared, values)
    estimate = FiniteElementErrorEstimate(
        _recovery_indicators(prepared, values, recovered),
        "zienkiewicz-zhu-patch-recovery",
    )
    return estimate, evidence


def _interpolation_defect(enriched: FiniteElementSpec, base_degree: int, /) -> Array:
    """Local matrix ``I - B E`` mapping enriched nodal values to ``z - I_h z``."""
    base = lagrange_element(enriched.cell_kind, base_degree)
    enriched_at_base, _ = enriched.tabulate(base.reference_nodes)
    base_at_enriched, _ = base.tabulate(enriched.reference_nodes)
    interpolation = base_at_enriched @ enriched_at_base
    return jnp.eye(enriched.local_dof_count) - interpolation


def dual_weighted_residual_indicators(
    discretization: FiniteElementDiscretization,
    field_name: str,
    operator: LinearSystem | PreparedLinearSolve,
    functional: ArrayLike,
    cell_residual: ArrayLike,
    /,
    *,
    base_degree: int,
    free_dofs: ArrayLike | None = None,
    policy: LinearSolvePolicy | None = None,
) -> tuple[FiniteElementDWRIndicators, LinearSolveResult]:
    """Goal-oriented indicators ``eta_K = r_K(u_h)(z - I_h z)`` (Becker-Rannacher).

    ``discretization`` is the enriched Lagrange space (degree above
    ``base_degree``) on the primal mesh and ``operator`` its primal system on the
    ``free_dofs`` (default: every DOF). The adjoint ``A^T z = functional`` is
    solved natively through :func:`phydrax.linalg.solve_adjoint`; constrained DOFs
    carry ``z = 0``. ``cell_residual`` holds the element-local residual of the
    base solution tested with every enriched local basis function, one row per
    cell in block order. The signed indicators sum to ``r(u_h)(z)``, the enriched
    error in the goal functional. Adjoint status reaches the caller unchanged.
    """
    field, elements = _field_elements(discretization, field_name)
    if not isinstance(operator, (LinearSystem, PreparedLinearSolve)):
        raise TypeError("operator must be LinearSystem or PreparedLinearSolve.")
    if isinstance(base_degree, bool) or not isinstance(base_degree, int):
        raise TypeError("base_degree must be an integer.")
    if any(
        base_degree < 1
        or base_degree >= element.degree
        or element.mapping != "identity"
        or element.value_shape != ()
        for element in elements
    ):
        raise ValueError("The enriched field must be scalar Lagrange above base_degree.")
    dof_map = discretization.dof_maps[field]
    free = (
        np.arange(dof_map.global_dof_count)
        if free_dofs is None
        else np.asarray(free_dofs, dtype=np.int64)
    )
    if free.ndim != 1 or np.any(free < 0) or np.any(free >= dof_map.global_dof_count):
        raise ValueError("free_dofs must index the enriched field DOFs.")
    residual = jnp.asarray(cell_residual)
    cell_count = sum(dof_map.cell_dofs[index].shape[0] for index in range(len(elements)))
    if residual.shape != (cell_count, elements[0].local_dof_count):
        raise ValueError("cell_residual must be (cells, enriched local DOFs).")
    adjoint = (
        solve_adjoint(operator, functional)
        if policy is None
        else solve_adjoint(operator, functional, policy=policy)
    )
    dual = (
        jnp.zeros((dof_map.global_dof_count,), dtype=residual.dtype)
        .at[jnp.asarray(free)]
        .set(jnp.asarray(adjoint.value).reshape((-1,)))
    )
    corrections = []
    for block_index, element in enumerate(elements):
        local = gather_routes(dof_map.relations[block_index], dual)
        defect = _interpolation_defect(element, base_degree)
        corrections.append(local @ defect.T)
    indicators = local_dual_weighted_residual(
        residual, jnp.concatenate(corrections, axis=0)
    )
    return indicators, adjoint


__all__ = [
    "FiniteElementRecoveryEvidence",
    "PreparedGradientRecovery",
    "dual_weighted_residual_indicators",
    "prepare_gradient_recovery",
    "recover_gradient",
    "recover_hessian",
    "recovery_error_estimate",
]
