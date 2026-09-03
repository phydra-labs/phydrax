#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from .._integration_domain import IntegrationDomain
from .._local_variational import (
    LocalGeometryActions,
    LocalMetricResult,
    LocalReferenceActions,
    LocalVariationalCapabilities,
    LocalVariationalOffer,
    PreparedLocalRegion,
)


_REFERENCE_REALIZATION = "explicit-polygon-h1-dense"


class ExplicitPolygonH1ReferenceActions(LocalReferenceActions):
    """Runtime explicit polygon interpolation and exact transpose actions."""

    block_index: int = eqx.field(static=True)
    cell_rows: Array
    trace_values: Array
    action_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    kernel_modes: tuple[str, ...] = eqx.field(static=True)
    is_trace: bool = eqx.field(static=True)

    def __init__(
        self,
        block_index: int,
        cell_rows: ArrayLike,
        trace_values: ArrayLike,
        /,
        *,
        local_width: int,
        point_count: int,
        maximum_derivative_order: int,
        structural_id: str,
        is_trace: bool,
    ):
        rows = jnp.asarray(cell_rows, dtype=jnp.int32)
        traces = jnp.asarray(trace_values)
        width = int(local_width)
        count = int(point_count)
        order = int(maximum_derivative_order)
        if rows.ndim != 1 or width <= 0 or count <= 0 or order not in (0, 1):
            raise ValueError("Explicit polygon reference action metadata is invalid.")
        if bool(is_trace):
            if traces.shape != (rows.size, count, width):
                raise ValueError("Explicit polygon trace tabulation has the wrong shape.")
        elif traces.size:
            raise ValueError("Cell reference actions cannot carry trace tabulation.")
        self.block_index = int(block_index)
        self.cell_rows = rows
        self.trace_values = traces
        self.local_width = width
        self.point_count = count
        self.maximum_derivative_order = order
        self.kernel_modes = ("dense",)
        self.realization_id = _REFERENCE_REALIZATION
        self.is_trace = bool(is_trace)
        self.action_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-reference-action",
                "structural": str(structural_id),
                "block": self.block_index,
                "rows": array_tree_fingerprint(rows),
                "trace": self.is_trace,
                "trace_values": None
                if not self.is_trace
                else array_tree_fingerprint(traces),
                "local_width": width,
                "point_count": count,
                "maximum_derivative_order": order,
            }
        )

    def _runtime(self, runtime: object, /):
        from ._space import ExplicitPolygonH1RuntimeData

        if not isinstance(runtime, ExplicitPolygonH1RuntimeData):
            raise TypeError("Explicit polygon actions require explicit polygon runtime.")
        if self.block_index < 0 or self.block_index >= len(runtime.bases):
            raise ValueError("Explicit polygon action block is out of range.")
        return runtime

    def _values(self, runtime: object, /) -> Array:
        runtime_ = self._runtime(runtime)
        if self.is_trace:
            return self.trace_values
        return runtime_.bases[self.block_index].basis_values[self.cell_rows]

    def _gradients(self, runtime: object, /) -> Array:
        runtime_ = self._runtime(runtime)
        if self.is_trace:
            raise ValueError(
                "Exterior explicit polygon actions do not provide gradients."
            )
        return runtime_.bases[self.block_index].reference_gradients[self.cell_rows]

    def realize_reference_actions(
        self, runtime: object, /
    ) -> "ExplicitPolygonH1ReferenceActions":
        self._runtime(runtime)
        return self

    def interpolate(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        coefficients = jnp.asarray(local_coefficients)
        if coefficients.shape[:2] != (self.cell_rows.size, self.local_width):
            raise ValueError("Explicit polygon local coefficients have the wrong shape.")
        return oe.contract("cql,cl...->cq...", self._values(runtime), coefficients)

    def interpolate_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        if values_.shape[:2] != (self.cell_rows.size, self.point_count):
            raise ValueError("Explicit polygon point values have the wrong shape.")
        return oe.contract("cql,cq...->cl...", self._values(runtime), values_)

    def reference_gradient(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        coefficients = jnp.asarray(local_coefficients)
        if coefficients.shape[:2] != (self.cell_rows.size, self.local_width):
            raise ValueError("Explicit polygon local coefficients have the wrong shape.")
        return oe.contract("cqld,cl...->cq...d", self._gradients(runtime), coefficients)

    def reference_gradient_transpose(
        self, runtime: object, gradients: ArrayLike, /
    ) -> Array:
        gradients_ = jnp.asarray(gradients)
        if (
            gradients_.shape[:2] != (self.cell_rows.size, self.point_count)
            or gradients_.shape[-1] != 2
        ):
            raise ValueError("Explicit polygon gradients have the wrong shape.")
        return oe.contract("cqld,cq...d->cl...", self._gradients(runtime), gradients_)

    def reference_hessian(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        del runtime, local_coefficients
        raise ValueError("Explicit polygon H1 does not provide Hessian actions.")

    def reference_hessian_transpose(
        self, runtime: object, hessians: ArrayLike, /
    ) -> Array:
        del runtime, hessians
        raise ValueError("Explicit polygon H1 does not provide Hessian actions.")

    def trace(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        if not self.is_trace:
            raise ValueError("Cell explicit polygon actions do not define a trace.")
        return self.interpolate(runtime, local_coefficients)

    def trace_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        if not self.is_trace:
            raise ValueError("Cell explicit polygon actions do not define a trace.")
        return self.interpolate_transpose(runtime, values)


class ExplicitPolygonH1GeometryActions(LocalGeometryActions):
    block_index: int = eqx.field(static=True)
    cell_rows: Array
    local_edges: Array
    reference_points: Array
    reference_weights: Array
    action_id: str = eqx.field(static=True)
    runtime_layout_id: str = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    domain_kind: str = eqx.field(static=True)

    def __init__(
        self,
        block_index: int,
        cell_rows: ArrayLike,
        local_edges: ArrayLike,
        reference_points: ArrayLike,
        reference_weights: ArrayLike,
        /,
        *,
        runtime_layout_id: str,
        domain_kind: str,
        structural_id: str,
    ):
        rows = jnp.asarray(cell_rows, dtype=jnp.int32)
        edges = jnp.asarray(local_edges, dtype=jnp.int32)
        points = jnp.asarray(reference_points)
        weights = jnp.asarray(reference_weights)
        kind = str(domain_kind)
        if rows.ndim != 1 or edges.shape != rows.shape:
            raise ValueError("Explicit polygon geometry routes are invalid.")
        if points.ndim != 2 or weights.shape != (points.shape[0],):
            raise ValueError("Explicit polygon reference quadrature is invalid.")
        if kind not in ("cell", "exterior_facet"):
            raise ValueError(
                "Explicit polygon geometry supports cells and exterior facets."
            )
        self.block_index = int(block_index)
        self.cell_rows = rows
        self.local_edges = edges
        self.reference_points = points
        self.reference_weights = weights
        self.runtime_layout_id = str(runtime_layout_id)
        self.entity_count = int(rows.size)
        self.domain_kind = kind
        self.action_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-geometry-action",
                "structural": str(structural_id),
                "block": self.block_index,
                "rows": array_tree_fingerprint(rows),
                "local_edges": array_tree_fingerprint(edges),
                "domain_kind": kind,
                "runtime_layout": self.runtime_layout_id,
            }
        )

    def realize(self, runtime: object, /) -> LocalMetricResult:
        from ._space import ExplicitPolygonH1RuntimeData

        if not isinstance(runtime, ExplicitPolygonH1RuntimeData):
            raise TypeError("Explicit polygon geometry requires its runtime data.")
        if runtime.geometry_layout_id != self.runtime_layout_id:
            raise ValueError("Explicit polygon geometry runtime layout is stale.")
        block = runtime.bases[self.block_index]
        if self.domain_kind == "cell":
            rows = self.cell_rows
            return LocalMetricResult(
                block.physical_points[rows],
                block.physical_weights[rows],
                block.jacobians[rows],
                block.inverse_jacobians[rows],
                valid=block.evidence.passed[rows],
            )
        geometry = runtime.geometries[self.block_index]
        vertices = geometry.vertices[self.cell_rows]
        rows = jnp.arange(self.cell_rows.size, dtype=jnp.int32)
        start = vertices[rows, self.local_edges]
        following = (self.local_edges + 1) % geometry.vertices.shape[1]
        stop = vertices[rows, following]
        tangent = stop - start
        parameter = self.reference_points[:, 0]
        points = (1.0 - parameter)[None, :, None] * start[:, None, :] + parameter[
            None, :, None
        ] * stop[:, None, :]
        length = jnp.sqrt(jnp.sum(tangent * tangent, axis=-1))
        weights = length[:, None] * self.reference_weights[None, :]
        jacobian = jnp.broadcast_to(
            tangent[:, None, :, None],
            (self.entity_count, parameter.size, 2, 1),
        )
        inverse_tangent = tangent / jnp.maximum(
            length[:, None] * length[:, None],
            jnp.finfo(length.dtype).tiny,
        )
        inverse = jnp.broadcast_to(
            inverse_tangent[:, None, None, :],
            (self.entity_count, parameter.size, 1, 2),
        )
        normals = jnp.broadcast_to(
            geometry.outward_normals[self.cell_rows, self.local_edges][:, None, :],
            points.shape,
        )
        valid = geometry.evidence.valid[self.cell_rows] & (length > 0.0)
        return LocalMetricResult(
            points,
            weights,
            jacobian,
            inverse,
            normals=normals,
            valid=valid,
        )


class ExplicitPolygonH1LocalProvider(StrictModule):
    discretization: object

    def __init__(self, discretization, /):
        from ._space import ExplicitPolygonH1Discretization

        if not isinstance(discretization, ExplicitPolygonH1Discretization):
            raise TypeError("discretization must be ExplicitPolygonH1Discretization.")
        self.discretization = discretization

    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:
        semantics = (
            "exact_interpolation_transpose",
            "exact_gradient_transpose",
            "exact_trace_transpose",
        )
        return LocalVariationalCapabilities(
            "explicit-polygon-h1-local-provider",
            (
                LocalVariationalOffer(
                    "prepared-local",
                    ("cell",),
                    (
                        "diffusion",
                        "tensor-diffusion",
                        "mass",
                        "source",
                        "cell-residual",
                        "cell-energy",
                        "functional",
                    ),
                    ("value", "grad"),
                    ("value", "gradient"),
                    ("dense",),
                    ("matrix_free",),
                    (_REFERENCE_REALIZATION,),
                    automatic_kernel_mode="dense",
                    automatic_operator_realization="matrix_free",
                    automatic_reference_realization_id=_REFERENCE_REALIZATION,
                    action_semantics=semantics,
                ),
                LocalVariationalOffer(
                    "prepared-local",
                    ("exterior_facet",),
                    ("boundary-load", "functional"),
                    ("value",),
                    ("value", "trace"),
                    ("dense",),
                    ("matrix_free",),
                    (_REFERENCE_REALIZATION,),
                    automatic_kernel_mode="dense",
                    automatic_operator_realization="matrix_free",
                    automatic_reference_realization_id=_REFERENCE_REALIZATION,
                    action_semantics=semantics,
                ),
            ),
        )

    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:
        discretization = self.discretization
        if not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be IntegrationDomain.")
        if domain.support_id != discretization.support.support_id:
            raise ValueError("Explicit polygon domain belongs to another support.")
        if tuple(field_names) != (discretization.field.name,):
            raise ValueError("Explicit polygon regions require their single field.")
        if str(kernel_mode) != "dense":
            raise ValueError("Explicit polygon H1 supports only dense local kernels.")
        order = int(maximum_derivative_order)
        if order < 0 or order > 1:
            raise ValueError("Explicit polygon H1 supports derivative order zero or one.")
        if domain.kind == "cell":
            return self._cell_regions(domain, order)
        if domain.kind == "exterior_facet":
            if order != 0:
                raise ValueError(
                    "Explicit polygon exterior regions support value jets only."
                )
            return self._facet_regions(domain)
        raise ValueError("Explicit polygon H1 does not offer interior-facet regions.")

    def _cell_regions(
        self, domain: IntegrationDomain, order: int, /
    ) -> tuple[PreparedLocalRegion, ...]:
        discretization = self.discretization
        selected_entities = np.asarray(domain.entity_indices, dtype=np.int32)
        rows_by_entity = {
            int(entity): row for row, entity in enumerate(selected_entities)
        }
        regions = []
        cell_offset = 0
        for block_index, block in enumerate(discretization.mesh.blocks):
            block_cells = np.arange(
                cell_offset, cell_offset + block.cell_count, dtype=np.int32
            )
            cell_offset += block.cell_count
            local_rows = np.asarray(
                [
                    local
                    for local, entity in enumerate(block_cells)
                    if int(entity) in rows_by_entity
                ],
                dtype=np.int32,
            )
            if local_rows.size == 0:
                continue
            entities = block_cells[local_rows]
            domain_rows = np.asarray(
                [rows_by_entity[int(entity)] for entity in entities], dtype=np.int32
            )
            selected_domain = IntegrationDomain(
                "cell",
                entities,
                domain.support_id,
                domain.entity_set_id,
                owner_cells=np.asarray(domain.owner_cells)[domain_rows],
                neighbour_cells=np.asarray(domain.neighbour_cells)[domain_rows],
                owner_local_entities=np.asarray(domain.owner_local_entities)[domain_rows],
                neighbour_local_entities=np.asarray(domain.neighbour_local_entities)[
                    domain_rows
                ],
                selection_id=domain.selection_id,
            )
            basis = discretization.default_runtime.bases[block_index]
            reference = ExplicitPolygonH1ReferenceActions(
                block_index,
                local_rows,
                jnp.empty((0,), dtype=basis.basis_values.dtype),
                local_width=discretization.dof_map.local_width,
                point_count=basis.point_count,
                maximum_derivative_order=order,
                structural_id=selected_domain.domain_id,
                is_trace=False,
            )
            geometry = ExplicitPolygonH1GeometryActions(
                block_index,
                local_rows,
                -jnp.ones((local_rows.size,), dtype=jnp.int32),
                basis.reference_points,
                basis.reference_weights,
                runtime_layout_id=discretization.default_runtime.geometry_layout_id,
                domain_kind="cell",
                structural_id=selected_domain.domain_id,
            )
            regions.append(
                PreparedLocalRegion(
                    selected_domain,
                    (discretization.field.name,),
                    (discretization.dof_map.cell_dofs[block_index][local_rows],),
                    (reference,),
                    geometry,
                    block_name=block.name,
                    cell_kind=block.cell_kind,
                    valid=basis.evidence.passed[local_rows],
                )
            )
        if not regions:
            raise ValueError("Explicit polygon cell domain selects no cells.")
        return tuple(regions)

    def _facet_regions(
        self, domain: IntegrationDomain, /
    ) -> tuple[PreparedLocalRegion, ...]:
        discretization = self.discretization
        from ...integration import (
            GaussLegendreRule,
            reference_rule_data,
            ReferenceIntervalRule,
        )

        edge_data = reference_rule_data(
            ReferenceIntervalRule(
                GaussLegendreRule(discretization.quadrature_policy.facet_order)
            )
        )
        reference_points = discretization.precision_policy.basis(edge_data.points)
        reference_weights = discretization.precision_policy.basis(edge_data.weights)
        owner_cells = np.asarray(domain.owner_cells, dtype=np.int32)
        local_edges_all = np.asarray(domain.owner_local_entities, dtype=np.int32)
        regions = []
        cell_offset = 0
        for block_index, block in enumerate(discretization.mesh.blocks):
            selected = np.flatnonzero(
                (owner_cells >= cell_offset)
                & (owner_cells < cell_offset + block.cell_count)
            ).astype(np.int32)
            if selected.size == 0:
                cell_offset += block.cell_count
                continue
            cell_rows = owner_cells[selected] - cell_offset
            local_edges = local_edges_all[selected]
            selected_domain = IntegrationDomain(
                "exterior_facet",
                np.asarray(domain.entity_indices)[selected],
                domain.support_id,
                domain.entity_set_id,
                owner_cells=owner_cells[selected],
                owner_local_entities=local_edges,
                selection_id=domain.selection_id,
            )
            width = discretization.dof_map.local_width
            trace_values = np.zeros(
                (selected.size, reference_points.shape[0], width), dtype=float
            )
            parameter = np.asarray(reference_points[:, 0])
            for row, local_edge in enumerate(local_edges):
                trace_values[row, :, int(local_edge)] = 1.0 - parameter
                trace_values[row, :, (int(local_edge) + 1) % block.arity] = parameter
            reference = ExplicitPolygonH1ReferenceActions(
                block_index,
                cell_rows,
                discretization.precision_policy.basis(trace_values),
                local_width=width,
                point_count=int(reference_points.shape[0]),
                maximum_derivative_order=0,
                structural_id=selected_domain.domain_id,
                is_trace=True,
            )
            geometry = ExplicitPolygonH1GeometryActions(
                block_index,
                cell_rows,
                local_edges,
                reference_points,
                reference_weights,
                runtime_layout_id=discretization.default_runtime.geometry_layout_id,
                domain_kind="exterior_facet",
                structural_id=selected_domain.domain_id,
            )
            regions.append(
                PreparedLocalRegion(
                    selected_domain,
                    (discretization.field.name,),
                    (discretization.dof_map.cell_dofs[block_index][cell_rows],),
                    (reference,),
                    geometry,
                    block_name=f"{block.name}:exterior",
                    cell_kind=block.cell_kind,
                    valid=discretization.default_runtime.geometries[
                        block_index
                    ].evidence.valid[cell_rows],
                )
            )
            cell_offset += block.cell_count
        if not regions:
            raise ValueError("Explicit polygon exterior domain selects no facets.")
        return tuple(regions)


__all__ = [
    "ExplicitPolygonH1GeometryActions",
    "ExplicitPolygonH1LocalProvider",
    "ExplicitPolygonH1ReferenceActions",
]
