#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Heat kernel of the lumped-mass P1 Laplacian on simplicial meshes.

Sites are mesh vertices with the row-summed (lumped) P1 mass ``m``. With the P1
stiffness ``K`` (the cotangent Laplacian on triangles) the discrete heat
semigroup is ``G(tau) = exp(-tau M^{-1} K)``, ``M = diag(m)``. ``M^{-1} K`` is
self-adjoint and positive semidefinite in the ``m``-weighted inner product, so
the Esedoglu–Otto dissipation argument applies on the discrete space.

``M^{-1} K`` is one native sparse coordinate operator: the finite-element
stiffness routes scaled by the inverse lumped mass of their target vertex. The
native Taylor exponential action is prepared once and applied to one label
column per call. Its per-action error is the native analytic truncation bound
when the policy keeps stored-coordinate norm evidence, otherwise the observed
Taylor tail. Both are absolute 1-norm quantities (hence max-norm bounds or
estimates) and exclude floating-point roundoff. The analytic bound grows like
``exp(tau ||M^{-1} K||_1)``, so stored-coordinate policies refuse (status
``RESOURCE_EXHAUSTED``) at the resolved kernel times ``tau >~ h^2`` that
threshold dynamics needs; the default policy therefore estimates the norm.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._doc import DOC_KEY0
from .._fingerprint import canonical_fingerprint
from ..discretization import (
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from ..linalg import (
    ArraySpace,
    matrix_exponential_action,
    MatrixFunctionResult,
    MatrixFunctionStatus,
    plan_taylor_exponential_action,
    prepare_taylor_exponential_action,
    PreparedTaylorExponentialAction,
    TaylorExponentialPolicy,
)
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ..typing import as_host_array, Dim, HostFloat64, HostInteger, parse, PRNGKey
from ._contracts import AbstractThresholdHeatKernel, HeatActionEvidence


_FIELD_NAME = "label-indicator"


class _VertexDim(Dim, minimum=1):
    """Mesh vertices (threshold-dynamics sites)."""


class _AmbientDim(Dim, minimum=2):
    """Ambient coordinates of the mesh embedding."""


class _SimplexDim(Dim, minimum=1):
    """Mesh simplices."""


class _CornerDim(Dim, minimum=3):
    """Vertices of one simplex."""


def _cell_mesh(vertices: np.ndarray, simplices: np.ndarray, /) -> tuple[CellMesh, str]:
    match simplices.shape[1], vertices.shape[1]:
        case 3, 2 | 3:
            return CellMesh.from_triangles(vertices, simplices), "triangle"
        case 4, 3:
            return CellMesh.from_tetrahedra(vertices, simplices), "tetrahedron"
        case _:
            raise ValueError(
                "MeshHeatKernel admits triangles in two or three dimensions and "
                f"tetrahedra in three dimensions; got {simplices.shape[1]}-vertex "
                f"simplices with {vertices.shape[1]} coordinates."
            )


def _maximum_edge_length(vertices: np.ndarray, simplices: np.ndarray, /) -> float:
    first, second = np.triu_indices(simplices.shape[1], k=1)
    edges = vertices[simplices[:, first]] - vertices[simplices[:, second]]
    return float(np.max(np.linalg.norm(edges, axis=-1)))


class MeshHeatKernel(AbstractThresholdHeatKernel):
    """Heat kernel ``exp(-tau M^{-1} K)`` of the lumped-mass P1 Laplacian.

    ``vertices`` (``(V, d)``) and ``simplices`` declare a planar triangle mesh
    (``d = 2``), a triangulated surface in ``R^3`` (``(S, 3)`` simplices) or a
    tetrahedral mesh (``(S, 4)`` simplices, ``d = 3``) with consistently oriented
    (positive for tetrahedra) simplices; every vertex must belong to a simplex.
    Sites are the vertices, weighted by the lumped P1 mass; boundaries carry the
    natural zero-flux condition.

    ``policy`` configures the native Taylor exponential action (default:
    estimated norm evidence with ``key``). A policy whose resource envelope
    refuses the operator is refused at construction; runtime Taylor refusals
    and tolerance failures reach ``HeatActionEvidence``. Label columns are
    applied sequentially; batching a step with ``jax.vmap`` turns the native
    bounded Taylor loops into full-capacity selects.
    """

    action: PreparedTaylorExponentialAction
    lumped_mass: Array
    site_shape: tuple[int, ...] = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    resolution_length: float = eqx.field(static=True)
    equal_site_measure: bool = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        simplices: ArrayLike,
        /,
        *,
        policy: TaylorExponentialPolicy | None = None,
        key: PRNGKey = DOC_KEY0,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        points = as_host_array(vertices, HostFloat64[_VertexDim, _AmbientDim], "vertices")
        cells = as_host_array(
            simplices, HostInteger[_SimplexDim, _CornerDim], "simplices"
        )
        norm_key = parse(key, PRNGKey, "key")
        working = jnp.dtype(dtype)
        if working not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise ValueError("dtype must be float32 or float64.")
        selected = (
            TaylorExponentialPolicy(norm_mode="estimate") if policy is None else policy
        )
        mesh, cell_kind = _cell_mesh(points, cells)
        discretization = FiniteElementPlan(
            mesh, FiniteElementFieldSpec(_FIELD_NAME, lagrange_element(cell_kind, 1))
        ).prepare()
        mass, stiffness = discretization.assemble_field_operators(
            _FIELD_NAME, discretization.default_runtime
        )
        vertex_count = points.shape[0]
        # Host-only immutable preparation: the element-local stiffness routes are
        # coalesced once (every action then touches each nonzero once), and exact
        # site-measure equality decides whether the capacitated (equal-measure)
        # volume constraint is admissible.
        lumped = np.asarray(
            jax.device_get(
                mass.mv(jnp.ones((vertex_count,), dtype=mass.coefficients.dtype))
            )
        )
        routes = stiffness.to_scipy().tocoo()
        space = ArraySpace((vertex_count,), dtype=working)
        operator = SparseCoordinateOperator(
            EdgeRelation(
                routes.col,
                routes.row,
                source_size=vertex_count,
                target_size=vertex_count,
            ),
            jnp.asarray(-routes.data / lumped[routes.row], dtype=working),
            source=space,
            target=space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "lumped-mass-inverted-p1-laplacian",
                    "discretization": discretization.prepared_id,
                    "dtype": working.name,
                }
            ),
        )
        plan = plan_taylor_exponential_action(operator, selected)
        if not plan.feasible:
            resources = plan.policy.resources
            raise ValueError(
                "The Taylor exponential resource policy refuses this mesh operator: "
                f"setup {plan.setup_matvec_count + plan.transpose_matvec_count} "
                f"actions (limit {resources.max_setup_matvec_count}), workspace "
                f"{plan.workspace_bytes} bytes (limit {resources.max_workspace_bytes}), "
                f"retained storage {plan.retained_storage_bytes} bytes (limit "
                f"{resources.max_retained_storage_bytes})."
            )
        action = prepare_taylor_exponential_action(operator, plan, key=norm_key)
        self.action = action
        self.lumped_mass = jnp.asarray(lumped, dtype=working)
        self.site_shape = (vertex_count,)
        self.cell_kind = cell_kind
        self.ambient_dimension = points.shape[1]
        self.resolution_length = _maximum_edge_length(points, cells)
        self.equal_site_measure = bool(np.all(lumped == lumped[0]))
        self.site_id = canonical_fingerprint(
            {
                "kind": "threshold-mesh-sites",
                "discretization": discretization.prepared_id,
            }
        )
        self.route_id = canonical_fingerprint(
            {
                "kind": "mesh-heat-kernel",
                "sites": self.site_id,
                "taylor_plan": plan.plan_id,
                "taylor_prepared": action.prepared_id,
                "dtype": working.name,
            }
        )

    @property
    def dtype(self) -> jnp.dtype:
        return self.lumped_mass.dtype

    def site_measures(self) -> Array:
        return self.lumped_mass

    def combine(
        self, fields: Array, times: Array, coefficients: Array, /
    ) -> tuple[Array, HeatActionEvidence]:
        values = coefficients.astype(self.dtype)
        if values.ndim not in (1, 3):
            raise ValueError(
                "coefficients must have shape (kernels,) or (kernels, labels, labels)."
            )
        base = fields.astype(self.dtype)
        weighted = (
            values[:, None, None]
            * (jnp.sum(base, axis=-1, keepdims=True) - base)[None]
            if values.ndim == 1
            else base[None] @ values
        )
        return self._apply_weighted(weighted, times)

    def smooth(self, fields: Array, time: Array, /) -> tuple[Array, HeatActionEvidence]:
        return self._apply_weighted(
            fields.astype(self.dtype)[None],
            time.astype(self.dtype).reshape((1,)),
        )

    def _apply_weighted(
        self, weighted: Array, times: Array, /
    ) -> tuple[Array, HeatActionEvidence]:
        """Apply one native action per kernel and label column."""
        kernel_count, vertex_count, label_count = weighted.shape
        columns = jnp.swapaxes(weighted, -1, -2).reshape((-1, vertex_count))
        column_times = jnp.repeat(times.astype(self.dtype), label_count)

        def act(item: tuple[Array, Array]) -> MatrixFunctionResult:
            time, column = item
            return matrix_exponential_action(self.action, column, time)

        results = jax.lax.map(act, (column_times, columns))
        values = results.value.reshape((kernel_count, label_count, vertex_count))
        potentials = jnp.swapaxes(jnp.sum(values, axis=0), 0, 1)
        diagnostics = results.diagnostics
        column_error = jnp.where(
            diagnostics.error_bound_available,
            diagnostics.error_bound,
            diagnostics.residual_estimate,
        ).reshape((kernel_count, label_count))
        status = results.status
        iterations = diagnostics.selected_degree * diagnostics.scaling_count
        evidence = HeatActionEvidence(
            successful=jnp.all(status == int(MatrixFunctionStatus.SUCCESS)),
            error_estimate=jnp.max(jnp.sum(column_error, axis=0)).astype(self.dtype),
            native_status=jnp.max(status).astype(jnp.int32),
            converged=jnp.all(diagnostics.converged),
            derivative_valid=jnp.all(diagnostics.derivative_valid),
            iterations=jnp.max(iterations).astype(jnp.int32),
            setup_matvec_count=jnp.max(diagnostics.setup_matvec_count).astype(jnp.int32),
            action_matvec_count=jnp.sum(diagnostics.action_matvec_count).astype(jnp.int32),
            transpose_matvec_count=jnp.max(
                diagnostics.transpose_matvec_count
            ).astype(jnp.int32),
            breakdown_status=jnp.max(diagnostics.breakdown_status).astype(jnp.int32),
            numeric_version=jnp.max(results.provenance.numeric_version).astype(jnp.int32),
            method="taylor-exponential-action",
            exact=False,
            retained_storage_bytes=diagnostics.retained_storage_bytes,
            workspace_bytes=diagnostics.workspace_bytes,
            operator_id=results.provenance.operator_id,
            prepared_id=results.provenance.prepared_id,
        )
        return potentials, evidence


__all__ = ["MeshHeatKernel"]
