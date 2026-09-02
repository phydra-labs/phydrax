#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.fem._generic import (
    _degree_aware_reference_rule,
    FiniteElementDiscretization,
    FiniteElementRuntimeData,
)
from ...discretization.fem._geometry_quality import (
    finite_element_geometry_quality,
    FiniteElementGeometryQualityEvidence,
)
from ...discretization.finite_volume._riemann import NumericalFluxResult
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    inverse_small_linear,
    OperatorProperties,
    SmallLinearSolvePlan,
)
from ._trace_routes import PreparedDGTraceRoute


class FiniteElementGeometrySnapshot(StrictModule, NonTrainableState):
    coordinates: Array
    coordinate_velocity: Array
    time: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        coordinate_velocity: ArrayLike,
        time: ArrayLike,
        /,
        *,
        topology_id: str,
        geometry_layout_id: str,
    ):
        values = jnp.asarray(coordinates)
        velocity = jnp.asarray(coordinate_velocity)
        time_ = jnp.asarray(time)
        topology = str(topology_id)
        layout = str(geometry_layout_id)
        if (
            values.ndim != 2
            or velocity.shape != values.shape
            or time_.shape != ()
            or not topology
            or not layout
        ):
            raise ValueError("Geometry snapshot shape or identities are invalid.")
        self.coordinates = values
        self.coordinate_velocity = velocity
        self.time = time_
        self.topology_id = topology
        self.geometry_layout_id = layout
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "finite-element-geometry-snapshot",
                "coordinates": array_tree_fingerprint(np.asarray(values)),
                "velocity": array_tree_fingerprint(np.asarray(velocity)),
                "time": float(np.asarray(time_)),
                "topology": topology,
                "layout": layout,
            }
        )

    def advance(
        self, step_size: ArrayLike, acceleration: ArrayLike | None = None, /
    ) -> "FiniteElementGeometrySnapshot":
        step = jnp.asarray(step_size)
        acceleration_ = (
            jnp.zeros_like(self.coordinate_velocity)
            if acceleration is None
            else jnp.asarray(acceleration)
        )
        if step.shape != () or acceleration_.shape != self.coordinates.shape:
            raise ValueError("Geometry advance inputs are incompatible.")
        coordinates = (
            self.coordinates
            + step * self.coordinate_velocity
            + 0.5 * step**2 * acceleration_
        )
        velocity = self.coordinate_velocity + step * acceleration_
        return FiniteElementGeometrySnapshot(
            coordinates,
            velocity,
            self.time + step,
            topology_id=self.topology_id,
            geometry_layout_id=self.geometry_layout_id,
        )


class ALEMetricEvidence(StrictModule, NonTrainableState):
    jacobian_rate: tuple[Array, ...]
    predicted_jacobian_rate: tuple[Array, ...]
    gcl_residual: tuple[Array, ...]
    maximum_gcl_defect: Array
    current_quality: FiniteElementGeometryQualityEvidence
    next_quality: FiniteElementGeometryQualityEvidence
    passed: Array
    evidence_id: str = eqx.field(static=True)


def finite_element_ale_metric_evidence(
    discretization: FiniteElementDiscretization,
    current: FiniteElementGeometrySnapshot,
    next_snapshot: FiniteElementGeometrySnapshot,
    /,
    *,
    tolerance: float = 1.0e-9,
) -> ALEMetricEvidence:
    if current.topology_id != next_snapshot.topology_id:
        raise ValueError("ALE snapshots use different topologies.")
    step = next_snapshot.time - current.time
    if float(np.asarray(step)) <= 0.0:
        raise ValueError("ALE metric evidence requires increasing time.")
    current_runtime = FiniteElementRuntimeData(
        discretization.mesh,
        current.coordinates,
        numeric_version=current.snapshot_id,
        geometry_layout_id=current.geometry_layout_id,
    )
    next_runtime = FiniteElementRuntimeData(
        discretization.mesh,
        next_snapshot.coordinates,
        numeric_version=next_snapshot.snapshot_id,
        geometry_layout_id=next_snapshot.geometry_layout_id,
    )
    current_quality = finite_element_geometry_quality(discretization, current_runtime)
    next_quality = finite_element_geometry_quality(discretization, next_runtime)
    rates = []
    predicted = []
    residuals = []
    for block_index, block in enumerate(discretization.mesh.blocks):
        coordinate_element = discretization.coordinate_elements[block_index]
        points, _weights = _degree_aware_reference_rule(
            block.cell_kind, max(2, coordinate_element.degree + 2)
        )
        _basis, gradients = coordinate_element.tabulate(points)
        routes = discretization.coordinate_dofs[block_index]
        current_coordinates = current.coordinates[routes]
        next_coordinates = next_snapshot.coordinates[routes]
        velocity = current.coordinate_velocity[routes]
        current_jacobian = oe.contract(
            "qid,cia->cqad", gradients, current_coordinates, backend="jax"
        )
        next_jacobian = oe.contract(
            "qid,cia->cqad", gradients, next_coordinates, backend="jax"
        )
        velocity_gradient_reference = oe.contract(
            "qid,cia->cqad", gradients, velocity, backend="jax"
        )
        current_inverse = inverse_small_linear(
            SmallLinearSolvePlan(current_jacobian.shape[-1]), current_jacobian
        )
        next_inverse = inverse_small_linear(
            SmallLinearSolvePlan(next_jacobian.shape[-1]), next_jacobian
        )
        inverse = eqx.error_if(
            current_inverse.value,
            jnp.any(~current_inverse.successful),
            "Current ALE geometry Jacobian is singular.",
        )
        current_determinant = current_inverse.determinant
        next_determinant = next_inverse.determinant
        velocity_gradient = oe.contract(
            "cqad,cqdb->cqab",
            velocity_gradient_reference,
            inverse,
            backend="jax",
        )
        divergence = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        actual = (next_determinant - current_determinant) / step
        predicted_rate = current_determinant * divergence
        residual = actual - predicted_rate
        rates.append(actual)
        predicted.append(predicted_rate)
        residuals.append(residual)
    maximum = jnp.max(jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in residuals)))
    passed = (maximum <= float(tolerance)) & current_quality.passed & next_quality.passed
    evidence_id = canonical_fingerprint(
        {
            "kind": "finite-element-ale-metric-evidence",
            "current": current.snapshot_id,
            "next": next_snapshot.snapshot_id,
            "tolerance": float(tolerance),
        }
    )
    return ALEMetricEvidence(
        tuple(rates),
        tuple(predicted),
        tuple(residuals),
        maximum,
        current_quality,
        next_quality,
        passed,
        evidence_id,
    )


def ale_physical_normal_flux(
    system: Any,
    state: ArrayLike,
    normal: ArrayLike,
    mesh_velocity: ArrayLike,
    args: Any = None,
    /,
) -> Array:
    value = jnp.asarray(state)
    normal_ = jnp.asarray(normal)
    velocity = jnp.asarray(mesh_velocity)
    grid_speed = oe.contract("...d,...d->...", velocity, normal_, backend="jax")
    return (
        system.physical_normal_flux(value, normal_, args) - grid_speed[..., None] * value
    )


def ale_numerical_normal_flux(
    interface_flux: Any,
    system: Any,
    left: ArrayLike,
    right: ArrayLike,
    normal: ArrayLike,
    mesh_velocity: ArrayLike,
    args: Any = None,
    /,
) -> NumericalFluxResult:
    left_ = jnp.asarray(left)
    right_ = jnp.asarray(right)
    normal_ = jnp.asarray(normal)
    velocity = jnp.asarray(mesh_velocity)
    base = interface_flux.normal_face_flux(system, left_, right_, normal_, args)
    grid_speed = oe.contract("...d,...d->...", velocity, normal_, backend="jax")
    flux = base.normal_flux - 0.5 * grid_speed[..., None] * (left_ + right_)
    return NumericalFluxResult(flux, base.max_speed + jnp.abs(grid_speed))


class MovingTraceRoute(StrictModule, NonTrainableState):
    current: PreparedDGTraceRoute
    next: PreparedDGTraceRoute
    route_id: str = eqx.field(static=True)

    def __init__(self, current: PreparedDGTraceRoute, next: PreparedDGTraceRoute, /):
        if (
            current.route_kind != next.route_kind
            or current.owner_dofs.shape != next.owner_dofs.shape
            or current.neighbour_dofs.shape != next.neighbour_dofs.shape
            or current.physical_points.shape != next.physical_points.shape
        ):
            raise ValueError("Moving trace route endpoints are incompatible.")
        self.current = current
        self.next = next
        self.route_id = canonical_fingerprint(
            {
                "kind": "moving-trace-route",
                "current": current.route_id,
                "next": next.route_id,
            }
        )

    def at(self, fraction: ArrayLike, /) -> PreparedDGTraceRoute:
        value = jnp.asarray(fraction)
        points = (
            1.0 - value
        ) * self.current.physical_points + value * self.next.physical_points
        weights = (
            1.0 - value
        ) * self.current.physical_weights + value * self.next.physical_weights
        normal = (1.0 - value) * self.current.normal + value * self.next.normal
        normal = (
            normal
            / jnp.sqrt(oe.contract("...d,...d->...", normal, normal, backend="jax"))[
                ..., None
            ]
        )
        return PreparedDGTraceRoute(
            self.current.route_kind,
            self.current.owner_dofs,
            neighbour_dofs=self.current.neighbour_dofs,
            owner_basis=self.current.owner_basis,
            neighbour_basis=self.current.neighbour_basis,
            owner_gradients=self.current.owner_gradients,
            physical_points=points,
            physical_weights=weights,
            normal=normal,
            mortar=self.current.mortar,
            boundary=self.current.boundary,
            component_transform=self.current.component_transform,
            coordinate_transform=self.current.coordinate_transform,
            route_id=self.route_id,
        )


class ConservativeRemapPlan(StrictModule):
    source_mass: Array
    target_mass: Array
    cross_mass: Array
    factorization: Any
    constant_defect: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mass: ArrayLike,
        target_mass: ArrayLike,
        cross_mass: ArrayLike,
        /,
    ):
        source = jnp.asarray(source_mass)
        target = jnp.asarray(target_mass)
        cross = jnp.asarray(cross_mass)
        if (
            source.ndim != 2
            or target.ndim != 2
            or cross.shape != (target.shape[0], source.shape[0])
            or source.shape[0] != source.shape[1]
            or target.shape[0] != target.shape[1]
        ):
            raise ValueError("Conservative remap mass shapes are incompatible.")
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "verified",
            },
        )
        factorization = factorize(
            DenseLinearOperator(
                target,
                properties=properties,
                operator_id=canonical_fingerprint(
                    {
                        "kind": "conservative-remap-target-mass",
                        "matrix": array_tree_fingerprint(np.asarray(target)),
                    }
                ),
            ),
            FactorizationPolicy("cholesky"),
        )
        source_constant = jnp.ones((source.shape[0],))
        target_constant = jnp.ones((target.shape[0],))
        defect = jnp.max(jnp.abs(cross @ source_constant - target @ target_constant))
        self.source_mass = source
        self.target_mass = target
        self.cross_mass = cross
        self.factorization = factorization
        self.constant_defect = defect
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-remap-plan",
                "source": array_tree_fingerprint(np.asarray(source)),
                "target": array_tree_fingerprint(np.asarray(target)),
                "cross": array_tree_fingerprint(np.asarray(cross)),
            }
        )

    def apply(self, source_coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(source_coefficients)
        right = oe.contract("ij,j...->i...", self.cross_mass, values, backend="jax")
        flat = right.reshape((right.shape[0], -1))
        components = []
        for component in range(flat.shape[-1]):
            solved = self.factorization.solve(flat[:, component])
            components.append(
                eqx.error_if(
                    solved.value,
                    ~solved.successful,
                    "Conservative remap target mass solve failed.",
                )
            )
        return jnp.stack(tuple(components), axis=-1).reshape(
            (self.target_mass.shape[0],) + values.shape[1:]
        )

    def transpose_apply(self, target_cotangent: ArrayLike, /) -> Array:
        cotangent = jnp.asarray(target_cotangent)
        flat = cotangent.reshape((cotangent.shape[0], -1))
        solved_components = []
        for component in range(flat.shape[-1]):
            solved = self.factorization.solve(flat[:, component])
            solved_components.append(
                eqx.error_if(
                    solved.value,
                    ~solved.successful,
                    "Conservative remap transpose mass solve failed.",
                )
            )
        solved = jnp.stack(tuple(solved_components), axis=-1)
        source = oe.contract("ij,i...->j...", self.cross_mass, solved, backend="jax")
        return source.reshape((self.source_mass.shape[0],) + cotangent.shape[1:])


class GeometryRecoveryResult(StrictModule, NonTrainableState):
    snapshot: FiniteElementGeometrySnapshot
    accepted_fraction: Array
    quality: FiniteElementGeometryQualityEvidence
    accepted: Array


def recover_geometry_snapshot(
    discretization: FiniteElementDiscretization,
    accepted: FiniteElementGeometrySnapshot,
    candidate: FiniteElementGeometrySnapshot,
    /,
    *,
    iterations: int = 24,
) -> GeometryRecoveryResult:
    if accepted.topology_id != candidate.topology_id:
        raise ValueError("Geometry recovery cannot change topology.")
    lower = 0.0
    upper = 1.0
    selected = accepted
    selected_quality = finite_element_geometry_quality(
        discretization,
        FiniteElementRuntimeData(
            discretization.mesh,
            accepted.coordinates,
            numeric_version=accepted.snapshot_id,
            geometry_layout_id=accepted.geometry_layout_id,
        ),
    )
    for _iteration in range(int(iterations)):
        fraction = 0.5 * (lower + upper)
        coordinates = (
            1.0 - fraction
        ) * accepted.coordinates + fraction * candidate.coordinates
        velocity = (
            1.0 - fraction
        ) * accepted.coordinate_velocity + fraction * candidate.coordinate_velocity
        snapshot = FiniteElementGeometrySnapshot(
            coordinates,
            velocity,
            (1.0 - fraction) * accepted.time + fraction * candidate.time,
            topology_id=accepted.topology_id,
            geometry_layout_id=accepted.geometry_layout_id,
        )
        quality = finite_element_geometry_quality(
            discretization,
            FiniteElementRuntimeData(
                discretization.mesh,
                coordinates,
                numeric_version=snapshot.snapshot_id,
                geometry_layout_id=snapshot.geometry_layout_id,
            ),
        )
        if bool(np.asarray(quality.passed)):
            lower = fraction
            selected = snapshot
            selected_quality = quality
        else:
            upper = fraction
    return GeometryRecoveryResult(
        selected,
        jnp.asarray(lower),
        selected_quality,
        selected_quality.passed,
    )


__all__ = [
    "ALEMetricEvidence",
    "ConservativeRemapPlan",
    "FiniteElementGeometrySnapshot",
    "GeometryRecoveryResult",
    "MovingTraceRoute",
    "ale_numerical_normal_flux",
    "ale_physical_normal_flux",
    "finite_element_ale_metric_evidence",
    "recover_geometry_snapshot",
]
