#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import ArraySpace, FunctionLinearOperator
from ..._cell_mesh import CellMesh
from ._cell import q4_cell_smoothing_layout
from ._common import (
    evaluate_smoothing_geometry,
    SmoothingPatchGeometry,
    SmoothingPatchLayout,
)
from ._edge import edge_smoothing_layout
from ._elasticity import SmoothedElasticityOperator, smoothing_local_stiffness
from ._moments import boundary_moment, primitive_volume_moment, shape_average
from ._node import node_smoothing_layout
from ._stabilization import SmoothingStabilizationPolicy


class SmoothedElasticityPlan(StrictModule, NonTrainableState):
    method: str = eqx.field(static=True)
    layout: SmoothingPatchLayout
    constitutive: Array
    global_node_count: int = eqx.field(static=True)
    stabilization: SmoothingStabilizationPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: Literal["ES", "NS"],
        mesh: CellMesh,
        constitutive: ArrayLike,
        /,
        *,
        stabilization: SmoothingStabilizationPolicy | None = None,
    ):
        if method == "ES":
            layout = edge_smoothing_layout(mesh)
        elif method == "NS":
            layout = node_smoothing_layout(mesh)
        else:
            raise ValueError("SmoothedElasticityPlan method must be ES or NS.")
        constitutive_ = jnp.asarray(constitutive)
        if constitutive_.shape != (3, 3):
            raise ValueError("2-D smoothed elasticity needs a 3x3 constitutive matrix.")
        stabilization_ = (
            SmoothingStabilizationPolicy() if stabilization is None else stabilization
        )
        self.method = method
        self.layout = layout
        self.constitutive = constitutive_
        self.global_node_count = int(mesh.coordinates.shape[0])
        self.stabilization = stabilization_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smoothed-elasticity-plan",
                "method": method,
                "layout": layout.layout_id,
                "constitutive_shape": list(constitutive_.shape),
                "stabilization": stabilization_.policy_id,
            }
        )

    def geometry(self, coordinates: ArrayLike, /) -> SmoothingPatchGeometry:
        return evaluate_smoothing_geometry(self.layout, coordinates)

    def local_stiffness(
        self,
        coordinates: ArrayLike,
        /,
        *,
        compatible_local_stiffness: ArrayLike | None = None,
    ) -> Array:
        return smoothing_local_stiffness(
            self.layout,
            self.geometry(coordinates),
            self.constitutive,
            compatible_local_stiffness=compatible_local_stiffness,
            stabilization=self.stabilization,
        )

    def operator(
        self,
        coordinates: ArrayLike,
        /,
        *,
        compatible_local_stiffness: ArrayLike | None = None,
    ) -> SmoothedElasticityOperator:
        return SmoothedElasticityOperator(
            self.layout,
            self.local_stiffness(
                coordinates,
                compatible_local_stiffness=compatible_local_stiffness,
            ),
            self.global_node_count,
        )

    def materialize_stiffness(
        self,
        coordinates: ArrayLike,
        /,
        *,
        compatible_local_stiffness: ArrayLike | None = None,
        max_entries: int = 4_000_000,
    ) -> Array:
        return self.operator(
            coordinates,
            compatible_local_stiffness=compatible_local_stiffness,
        ).materialize(max_entries=max_entries)


class SelectiveESNSPlan(StrictModule, NonTrainableState):
    edge_plan: SmoothedElasticityPlan
    node_plan: SmoothedElasticityPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        deviatoric_constitutive: ArrayLike,
        volumetric_constitutive: ArrayLike,
        /,
    ):
        edge = SmoothedElasticityPlan("ES", mesh, deviatoric_constitutive)
        node = SmoothedElasticityPlan("NS", mesh, volumetric_constitutive)
        self.edge_plan = edge
        self.node_plan = node
        self.plan_id = canonical_fingerprint(
            {
                "kind": "selective-es-ns-plan",
                "edge": edge.plan_id,
                "node": node.plan_id,
            }
        )

    def operator(self, coordinates: ArrayLike, /) -> FunctionLinearOperator:
        edge = self.edge_plan.operator(coordinates)
        node = self.node_plan.operator(coordinates)
        size = 2 * self.edge_plan.global_node_count
        space = ArraySpace((size,), dtype=edge.local_stiffness.dtype)
        return FunctionLinearOperator(
            lambda value: edge.mv(value) + node.mv(value),
            source=space,
            target=space,
            operator_id=self.plan_id,
        )

    def materialize_stiffness(
        self,
        coordinates: ArrayLike,
        /,
        *,
        max_entries: int = 4_000_000,
    ) -> Array:
        edge = self.edge_plan.operator(coordinates).materialize(max_entries=max_entries)
        node = self.node_plan.operator(coordinates).materialize(max_entries=max_entries)
        return edge + node


class Q4FSDTChannels(StrictModule):
    membrane_gradient: Array
    bending_gradient: Array
    shear_average: Array
    nonlinear_gradient: Array


class Q4FSDTSmoothingPlan(StrictModule, NonTrainableState):
    membrane: SmoothingPatchLayout
    bending: SmoothingPatchLayout
    shear: SmoothingPatchLayout
    nonlinear_membrane: SmoothingPatchLayout
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        membrane_cells: int = 3,
        bending_cells: int = 3,
        shear_cells: int = 1,
        nonlinear_cells: int = 3,
    ):
        membrane = q4_cell_smoothing_layout(mesh, membrane_cells)
        bending = q4_cell_smoothing_layout(mesh, bending_cells)
        shear = q4_cell_smoothing_layout(mesh, shear_cells)
        nonlinear = q4_cell_smoothing_layout(mesh, nonlinear_cells)
        self.membrane = membrane
        self.bending = bending
        self.shear = shear
        self.nonlinear_membrane = nonlinear
        self.plan_id = canonical_fingerprint(
            {
                "kind": "q4-fsdt-smoothing-plan",
                "membrane": membrane.layout_id,
                "bending": bending.layout_id,
                "shear": shear.layout_id,
                "nonlinear": nonlinear.layout_id,
            }
        )

    def channels(self, coordinates: ArrayLike, /) -> Q4FSDTChannels:
        membrane_geometry = evaluate_smoothing_geometry(self.membrane, coordinates)
        bending_geometry = evaluate_smoothing_geometry(self.bending, coordinates)
        shear_values = jnp.mean(self.shear.boundary_shape_values, axis=(1, 2))
        nonlinear_geometry = evaluate_smoothing_geometry(
            self.nonlinear_membrane, coordinates
        )
        return Q4FSDTChannels(
            membrane_gradient=boundary_moment(self.membrane, membrane_geometry),
            bending_gradient=boundary_moment(self.bending, bending_geometry),
            shear_average=shape_average(shear_values[:, None, :]),
            nonlinear_gradient=boundary_moment(
                self.nonlinear_membrane, nonlinear_geometry
            ),
        )


class FullySmoothedAxisymmetricPlan(StrictModule, NonTrainableState):
    variant: str = eqx.field(static=True)
    layout: SmoothingPatchLayout
    constitutive: Array
    density: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        variant: Literal["CS", "ES", "NS"],
        mesh: CellMesh,
        constitutive: ArrayLike,
        /,
        *,
        density: float = 1.0,
        cell_smoothing_count: int = 1,
    ):
        if variant == "CS":
            layout = q4_cell_smoothing_layout(mesh, cell_smoothing_count)
        elif variant == "ES":
            layout = edge_smoothing_layout(mesh)
        elif variant == "NS":
            layout = node_smoothing_layout(mesh)
        else:
            raise ValueError("Axisymmetric smoothing variant must be CS, ES, or NS.")
        constitutive_ = jnp.asarray(constitutive)
        density_ = float(density)
        if constitutive_.shape != (4, 4) or density_ <= 0.0:
            raise ValueError("Axisymmetric constitutive matrix or density is invalid.")
        self.variant = variant
        self.layout = layout
        self.constitutive = constitutive_
        self.density = density_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fully-smoothed-axisymmetric-plan",
                "variant": variant,
                "layout": layout.layout_id,
                "density": density_,
            }
        )

    def operators(
        self,
        coordinates: ArrayLike,
        primitive_values: ArrayLike,
        smoothed_shape_values: ArrayLike,
        /,
    ) -> tuple[Array, Array, SmoothingPatchGeometry]:
        geometry = evaluate_smoothing_geometry(self.layout, coordinates)
        if jnp.any(geometry.centroid[:, 0] <= 0.0):
            raise ValueError("Axisymmetric smoothing patches require positive radius.")
        gradient = boundary_moment(self.layout, geometry)
        hoop = (
            primitive_volume_moment(
                self.layout,
                geometry,
                primitive_values,
            )
            / geometry.centroid[:, 0, None]
        )
        shape_values = jnp.asarray(smoothed_shape_values)
        if shape_values.shape != self.layout.dof_routes.shape:
            raise ValueError("Smoothed axisymmetric shape values have invalid shape.")
        gx = gradient[..., 0]
        gz = gradient[..., 1]
        zeros = jnp.zeros_like(gx)
        radial = jnp.stack((gx, zeros), axis=-1)
        axial = jnp.stack((zeros, gz), axis=-1)
        shear = jnp.stack((gz, gx), axis=-1)
        hoop_row = jnp.stack((hoop, zeros), axis=-1)
        strain = jnp.stack((radial, axial, shear, hoop_row), axis=-2)
        strain = jnp.transpose(strain, (0, 2, 1, 3)).reshape(
            (strain.shape[0], 4, 2 * strain.shape[1])
        )
        shape_matrix = jnp.zeros(
            (shape_values.shape[0], 2, 2 * shape_values.shape[1]),
            dtype=shape_values.dtype,
        )
        shape_matrix = shape_matrix.at[:, 0, 0::2].set(shape_values)
        shape_matrix = shape_matrix.at[:, 1, 1::2].set(shape_values)
        factor = 2.0 * jnp.pi * geometry.centroid[:, 0] * geometry.area
        stiffness = oe.contract(
            "p,psi,st,ptj->pij",
            factor,
            strain,
            self.constitutive,
            strain,
        )
        mass = self.density * oe.contract(
            "p,pai,paj->pij",
            factor,
            shape_matrix,
            shape_matrix,
        )
        return stiffness, mass, geometry


__all__ = [
    "FullySmoothedAxisymmetricPlan",
    "Q4FSDTChannels",
    "Q4FSDTSmoothingPlan",
    "SelectiveESNSPlan",
    "SmoothedElasticityPlan",
]
