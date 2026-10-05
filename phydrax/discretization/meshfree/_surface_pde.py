# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Tangential vector and typed tensor operators on prepared meshfree surfaces.

Tangent tensor fields use ambient components with every index tangential. The
Levi-Civita connection of the induced metric is the Gauss formula
``nabla_X V = P d_X V``: the projector couples Cartesian components through the
second fundamental form, so no chart Christoffel symbols are recomputed here.
Bochner and Hodge vector Laplacians are distinct discrete operators: the
Bochner form is one projected stencil plus the curvature-squared term, the
Hodge form composes ``grad div`` with the divergence of the exterior
derivative. Their difference approximates the Ricci action (Weitzenboeck).
Tangency and incompressibility enter as native saddle-point constraints.
"""

from __future__ import annotations

from typing import assert_never, final, Literal, TypeAlias

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...ein import contract
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    ComposedLinearOperator,
    FunctionLinearOperator,
    IdentityLinearOperator,
    LinearSystem,
    saddle_point_system,
    ScaledLinearOperator,
    SumLinearOperator,
)
from ...metrix._tensor import raise_index, TensorType
from ...sparse import RowRelation, SparseCoordinateOperator
from ...typing import Float, parse
from ._surface import PreparedSurfacePointCloud
from ._surface_geometry import (
    ChartSurfaceGeometry,
    SurfaceAmbientDim,
    SurfaceNodeDim,
)


SurfaceVectorLaplacianKind: TypeAlias = Literal["bochner", "hodge"]


def _neighbor_gradient(surface: PreparedSurfacePointCloud) -> Array:
    """Scalar surface-gradient stencil weights (rows, neighbors, ambient)."""
    return surface.surface_gradient.coefficients[..., 0]


def _self_slots(relation: RowRelation) -> Array:
    rows = jnp.arange(relation.source_indices.shape[0])[:, None]
    return relation.valid & (relation.source_indices == rows)


def _projector_power(projector: Array, rank: int) -> Array:
    """Tensor-product projector acting on ``rank`` ambient indices, flattened."""
    count, ambient = projector.shape[0], projector.shape[1]
    result = jnp.ones((count, 1, 1), dtype=projector.dtype)
    for _ in range(rank):
        result = contract("nij,nab->niajb", result, projector).reshape(
            (count, result.shape[1] * ambient, result.shape[2] * ambient)
        )
    return result


@final
class SurfaceTangentCalculus(StrictModule):
    """Covariant first/second-order operators for tangent fields on one surface.

    Spaces: vectors ``(N, A)``, rank-two tangent tensors ``(N, A, A)`` with the
    derivative index last. ``ricci[n]`` is the Gauss-equation Ricci tensor
    ``<H, II> - sum II II`` in ambient tangential components.
    """

    __strict_contract__ = True
    surface: PreparedSurfacePointCloud
    covariant_gradient: SparseCoordinateOperator
    exterior_derivative: SparseCoordinateOperator
    tensor_divergence: SparseCoordinateOperator
    bochner_laplacian: SparseCoordinateOperator
    normal_constraint: SparseCoordinateOperator
    ricci: Float[SurfaceNodeDim, SurfaceAmbientDim, SurfaceAmbientDim]

    def __init__(self, surface: PreparedSurfacePointCloud) -> None:
        if not isinstance(surface, PreparedSurfacePointCloud):
            raise TypeError("surface must be a PreparedSurfacePointCloud.")
        geometry = surface.geometry
        relation = surface.relation
        projector = geometry.projectors
        form = geometry.second_fundamental_form
        count, ambient = surface.points.shape
        codimension = geometry.codimension
        dtype = surface.points.dtype
        gradient = _neighbor_gradient(surface)
        vector = surface.surface_gradient.target
        tensor = surface.surface_hessian.target
        identifier = surface.surface_gradient.operator_id
        curvature_square = contract("nabe,nade->nbd", form, form)
        mean = geometry.mean_curvature_vector
        self.ricci = contract("na,nabc->nbc", mean, form) - curvature_square
        self.surface = surface
        self.covariant_gradient = SparseCoordinateOperator(
            relation,
            contract("nac,nkb->nkabc", projector, gradient).reshape(
                (count, -1, ambient * ambient, ambient)
            ),
            source=vector,
            target=tensor,
            block_shape=(ambient * ambient, ambient),
            operator_id=f"{identifier}:covariant-gradient",
        )
        # (d omega)[e, d] = nabla_e V_d - nabla_d V_e for omega = V^flat.
        exterior = contract("ndc,nke->nkedc", projector, gradient) - contract(
            "nec,nkd->nkedc", projector, gradient
        )
        self.exterior_derivative = SparseCoordinateOperator(
            relation,
            exterior.reshape((count, -1, ambient * ambient, ambient)),
            source=vector,
            target=tensor,
            block_shape=(ambient * ambient, ambient),
            operator_id=f"{identifier}:exterior-derivative",
        )
        # (Div F)_a = P_ad sum_e nabla_e F[e, d]; the derivative is tangential.
        self.tensor_divergence = SparseCoordinateOperator(
            relation,
            contract("nad,nke->nkaed", projector, gradient).reshape(
                (count, -1, ambient, ambient * ambient)
            ),
            source=tensor,
            target=vector,
            block_shape=(ambient, ambient * ambient),
            operator_id=f"{identifier}:tensor-divergence",
        )
        self_slot = _self_slots(relation).astype(dtype)
        if not bool(jnp.all(jnp.any(self_slot > 0, axis=-1))):
            raise ValueError(
                "Every surface row must contain its own node in the support."
            )
        # Delta_B V = P Delta(V_ambient) + (sum II II) V for tangent V.
        bochner = (
            surface.laplace_beltrami.coefficients[:, :, None, None] * projector[:, None]
            + self_slot[:, :, None, None] * curvature_square[:, None]
        )
        self.bochner_laplacian = SparseCoordinateOperator(
            relation,
            bochner,
            source=vector,
            target=vector,
            block_shape=(ambient, ambient),
            operator_id=f"{identifier}:bochner-laplacian",
        )
        identity_relation = RowRelation(
            jnp.arange(count, dtype=relation.source_indices.dtype)[:, None],
            source_size=count,
            valid=jnp.ones((count, 1), dtype=jnp.bool_),
        )
        self.normal_constraint = SparseCoordinateOperator(
            identity_relation,
            jnp.swapaxes(geometry.normal_frames, -1, -2)[:, None],
            source=vector,
            target=ArraySpace(
                (count, codimension), dtype=dtype, space_id=f"{identifier}:normal"
            ),
            block_shape=(codimension, ambient),
            operator_id=f"{identifier}:normal-components",
        )

    @property
    def vector_space(self) -> ArraySpace:
        space = self.surface.surface_gradient.target
        if not isinstance(space, ArraySpace):
            raise TypeError("Surface vector space must be an ArraySpace.")
        return space

    @property
    def divergence(self) -> SparseCoordinateOperator:
        return self.surface.strong_surface_divergence

    @property
    def gradient(self) -> SparseCoordinateOperator:
        return self.surface.surface_gradient

    @property
    def hodge_laplacian(self) -> AbstractLinearOperator:
        """``grad div V + Div(d V^flat)`` (negative semidefinite convention)."""
        return SumLinearOperator(
            ComposedLinearOperator(self.gradient, self.divergence),
            ComposedLinearOperator(self.tensor_divergence, self.exterior_derivative),
        )

    @property
    def scalar_curvature(self) -> Array:
        return jnp.trace(self.ricci, axis1=-2, axis2=-1)

    @property
    def gauss_curvature(self) -> Array:
        if self.surface.geometry.intrinsic_dimension != 2:
            raise ValueError("Gauss curvature is defined for two-dimensional sheets.")
        return self.scalar_curvature / 2

    def vector_laplacian(
        self, kind: SurfaceVectorLaplacianKind
    ) -> AbstractLinearOperator:
        kind_ = parse(kind, SurfaceVectorLaplacianKind, "kind")
        match kind_:
            case "bochner":
                return self.bochner_laplacian
            case "hodge":
                return self.hodge_laplacian
            case _:
                assert_never(kind_)

    def tangential(self, field: ArrayLike) -> Array:
        return contract(
            "nab,nb->na", self.surface.geometry.projectors, jnp.asarray(field)
        )

    def covariant_derivative(self, tensor_type: TensorType) -> SparseCoordinateOperator:
        """``nabla T`` of a tangential tensor; the derivative index is appended last.

        With a Euclidean ambient metric, covariant and contravariant ambient
        components coincide; ``tensor_type`` keeps their identity for chart
        re-expression and fixes the rank.
        """
        if not isinstance(tensor_type, TensorType) or tensor_type.density_weight != 0:
            raise TypeError("tensor_type must be an unweighted metrix TensorType.")
        rank = tensor_type.rank
        if rank == 0:
            return self.gradient
        surface = self.surface
        count, ambient = surface.points.shape
        dtype = surface.points.dtype
        power = _projector_power(surface.geometry.projectors, rank)
        coefficients = contract("nij,nkb->nkibj", power, _neighbor_gradient(surface))
        identifier = surface.surface_gradient.operator_id
        source = ArraySpace(
            (count,) + (ambient,) * rank,
            dtype=dtype,
            space_id=f"{identifier}:tensor-rank-{rank}",
        )
        target = ArraySpace(
            (count,) + (ambient,) * (rank + 1),
            dtype=dtype,
            space_id=f"{identifier}:tensor-rank-{rank + 1}",
        )
        return SparseCoordinateOperator(
            surface.relation,
            coefficients.reshape((count, -1, ambient ** (rank + 1), ambient**rank)),
            source=source,
            target=target,
            block_shape=(ambient ** (rank + 1), ambient**rank),
            operator_id=f"{identifier}:covariant-derivative-{'-'.join(tensor_type.variance)}",
        )

    def chart_components(self, field: ArrayLike, tensor_type: TensorType) -> Array:
        """Typed components of an ambient tangential tensor in the declared charts.

        Covariant axes pull back through the chart tangent basis; contravariant
        axes are then raised with the metrix induced metric of each node's chart.
        """
        geometry = self.surface.plan.geometry
        if not isinstance(geometry, ChartSurfaceGeometry):
            raise ValueError("Chart components require an authoritative declared chart.")
        values = jnp.asarray(field)
        rank = tensor_type.rank
        count, ambient = self.surface.points.shape
        if values.shape != (count,) + (ambient,) * rank:
            raise ValueError("Field shape must match the tensor rank and ambient space.")
        coordinates = geometry.chart_coordinates.astype(values.dtype)
        _, basis, _ = geometry.jets(geometry.chart_indices, coordinates)
        lowered = values
        for axis in range(rank):
            lowered = jnp.moveaxis(
                contract("n...a,nai->n...i", jnp.moveaxis(lowered, axis + 1, -1), basis),
                -1,
                axis + 1,
            )
        # raise_index checks only the raised axis, which is covariant until raised.
        lowered_type = TensorType(("covariant",) * rank)
        result = jnp.zeros_like(lowered)
        for chart_index, chart in enumerate(geometry.charts):
            raised = lowered
            for axis, variance in enumerate(tensor_type.variance):
                if variance == "contravariant":
                    raised = raise_index(
                        raised,
                        chart.induced_metric(),
                        coordinates,
                        axis=axis,
                        tensor_type=lowered_type,
                    )
            selected = geometry.chart_indices == chart_index
            result = jnp.where(selected.reshape((count,) + (1,) * rank), raised, result)
        return result

    def tangent_vector_system(
        self,
        kind: SurfaceVectorLaplacianKind,
        *,
        diffusivity: float,
        reaction: float,
    ) -> LinearSystem:
        """Saddle system ``[[r I - k Delta, N], [N^T, 0]]`` for tangent unknowns."""
        if diffusivity <= 0 or reaction < 0:
            raise ValueError("diffusivity must be positive and reaction nonnegative.")
        primal = SumLinearOperator(
            ScaledLinearOperator(IdentityLinearOperator(self.vector_space), reaction),
            ScaledLinearOperator(self.vector_laplacian(kind), -diffusivity),
        )
        return saddle_point_system(primal, self.normal_constraint)

    def tangent_rhs(self, forcing: ArrayLike) -> tuple[Array, Array]:
        values = jnp.asarray(forcing, dtype=self.surface.points.dtype)
        space = self.normal_constraint.target
        if not isinstance(space, ArraySpace):
            raise TypeError("Surface normal constraint space must be an ArraySpace.")
        return values, jnp.zeros(space.shape, dtype=values.dtype)

    def stokes_system(
        self,
        *,
        viscosity: float,
        reaction: float,
        kind: SurfaceVectorLaplacianKind = "bochner",
    ) -> LinearSystem:
        """Closed-surface tangential Stokes/Brinkman collocated block system.

        Unknowns ``(U, (p, lambda))`` on the native block space:
        ``r U - nu Delta U + grad p + N lambda = F``, ``div U - m (m.p) = 0`` and
        ``N^T U = 0``. The consistent collocated gradient/divergence pair is not
        transpose-structured, so this is a general native block operator rather
        than a symmetric saddle. ``grad 1 = 0`` exactly, so the rank-one gauge
        block selects the measure-mean-zero pressure; its value ``m.p`` (see
        ``stokes_gauge_residual``) measures the discrete divergence-theorem
        defect and must be inspected with the native solve status.
        """
        surface = self.surface
        if surface.plan.boundary is not None:
            raise ValueError(
                "Open-surface Stokes requires velocity boundary rows; only closed sources are admitted."
            )
        if viscosity <= 0 or reaction < 0:
            raise ValueError("viscosity must be positive and reaction nonnegative.")
        count = surface.points.shape[0]
        codimension = surface.geometry.codimension
        dtype = surface.points.dtype
        measures = surface.measures
        vector = self.vector_space
        primal = SumLinearOperator(
            ScaledLinearOperator(IdentityLinearOperator(vector), reaction),
            ScaledLinearOperator(self.vector_laplacian(kind), -viscosity),
        )
        constraint_space = ArraySpace(
            (count, 1 + codimension),
            dtype=dtype,
            space_id=f"{vector.space_id}:stokes-constraints",
        )
        gradient = surface.surface_gradient
        divergence = surface.strong_surface_divergence
        normals = surface.geometry.normal_frames

        def constrain(value: Array) -> Array:
            tangency = contract("nac,na->nc", normals, value)
            return jnp.concatenate((divergence.mv(value)[:, None], tangency), axis=1)

        def constrain_transpose(value: Array) -> Array:
            return divergence.transpose_mv(value[:, 0]) + contract(
                "nac,nc->na", normals, value[:, 1:]
            )

        def couple(value: Array) -> Array:
            return gradient.mv(value[:, 0]) + contract(
                "nac,nc->na", normals, value[:, 1:]
            )

        def couple_transpose(value: Array) -> Array:
            return jnp.concatenate(
                (
                    gradient.transpose_mv(value)[:, None],
                    contract("nac,na->nc", normals, value),
                ),
                axis=1,
            )

        def gauge(value: Array) -> Array:
            pressure = -measures * jnp.sum(measures * value[:, 0])
            return jnp.concatenate(
                (pressure[:, None], jnp.zeros_like(value[:, 1:])), axis=1
            )

        space = BlockSpace(
            (vector, constraint_space),
            names=("velocity", "constraints"),
            space_id=f"{vector.space_id}:stokes",
        )
        operator = BlockLinearOperator(
            (
                (
                    primal,
                    FunctionLinearOperator(
                        couple,
                        source=constraint_space,
                        target=vector,
                        transpose_action=couple_transpose,
                        operator_id=f"{vector.space_id}:pressure-gradient-normal",
                    ),
                ),
                (
                    FunctionLinearOperator(
                        constrain,
                        source=vector,
                        target=constraint_space,
                        transpose_action=constrain_transpose,
                        operator_id=f"{vector.space_id}:divergence-tangency",
                    ),
                    FunctionLinearOperator(
                        gauge,
                        source=constraint_space,
                        target=constraint_space,
                        transpose_action=gauge,
                        operator_id=f"{vector.space_id}:pressure-gauge",
                    ),
                ),
            ),
            source=space,
            target=space,
            operator_id=f"{vector.space_id}:collocated-stokes",
        )
        return LinearSystem(operator)

    def stokes_rhs(self, forcing: ArrayLike) -> tuple[Array, Array]:
        values = jnp.asarray(forcing, dtype=self.surface.points.dtype)
        constraints = jnp.zeros(
            (values.shape[0], 1 + self.surface.geometry.codimension), dtype=values.dtype
        )
        return values, constraints

    def stokes_gauge_residual(self, solution: tuple[Array, Array]) -> Array:
        """Measure-weighted pressure mean selected by the gauge block."""
        return jnp.sum(self.surface.measures * solution[1][:, 0])


__all__ = [
    "SurfaceTangentCalculus",
    "SurfaceVectorLaplacianKind",
]
