#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ...discretization.mpm._dynamics import PreparedMPMDynamics
from ...discretization.mpm._storage import DenseMPMNodalStoragePlan
from ...sparse._ops import linear_apply, linear_transpose_apply
from ...sparse._relation import EdgeRelation
from ...typing import checked


@final
class MPMToMACFaceTransferEvidence(StrictModule):
    """Host-prepared support and fixed-capacity admission, exposed on device."""

    supported: Array
    capacity_satisfied: Array
    successful: Array
    face_supported: tuple[Array, ...]
    partition_unity_defect: Array
    required_routes: int = eqx.field(static=True)
    maximum_routes: int = eqx.field(static=True)


@final
class MPMToMACFaceTransferPlan(StrictModule, NonTrainableState):
    """Prepare multilinear sampling of the actual MPM background nodes.

    The transfer reproduces a constant nodal vector at each supported MAC face.
    Its transpose conserves total component force under ordinary, unweighted
    pairings. No particle positions or particle assignment weights enter this
    geometry-only map. Sparse, multifield, and axisymmetric MPM are refused.
    """

    solid: PreparedMPMDynamics
    operators: PreparedMACOperators
    maximum_routes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        solid: PreparedMPMDynamics,
        operators: PreparedMACOperators,
        /,
        *,
        maximum_routes: int,
    ) -> None:
        if not isinstance(maximum_routes, int) or isinstance(maximum_routes, bool):
            raise TypeError("maximum_routes must be an integer.")
        if maximum_routes < 0:
            raise ValueError("maximum_routes must be nonnegative.")
        if not isinstance(solid.nodal_storage, DenseMPMNodalStoragePlan):
            raise ValueError("MPM-to-MAC transfer requires dense nodal storage.")
        if solid.nodal_fields.field_count != 1:
            raise ValueError("MPM-to-MAC transfer requires a single nodal field.")
        if solid.geometry.kind != "cartesian":
            raise ValueError("MPM-to-MAC transfer requires Cartesian MPM geometry.")
        grid = operators.discretization.grid
        if solid.splat.layout.axis_names != grid.axis_names:
            raise ValueError(
                "MPM and MAC spatial axes must have the same identity/order."
            )
        for solid_axis, fluid_axis in zip(
            solid.splat.plan.target.structured_axes, grid.structured_axes, strict=True
        ):
            if solid_axis.periodic != fluid_axis.periodic:
                raise ValueError("MPM and MAC periodic-axis semantics must agree.")
            if solid_axis.periodic:
                solid_period = solid_axis.bounds[1] - solid_axis.bounds[0]
                fluid_period = fluid_axis.bounds[1] - fluid_axis.bounds[0]
                if solid_period != fluid_period:
                    raise ValueError("MPM and MAC periodic-axis periods must agree.")
        self.solid = solid
        self.operators = operators
        self.maximum_routes = maximum_routes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-to-mac-face-transfer-plan",
                "solid": solid.prepared_id,
                "operators": operators.prepared_id,
                "maximum_routes": maximum_routes,
            }
        )

    def prepare(self, /) -> PreparedMPMToMACFaceTransfer:
        return PreparedMPMToMACFaceTransfer(self)


def _face_routes(
    plan: MPMToMACFaceTransferPlan,
    axis: int,
    offset: int,
    /,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
]:
    """Host-only immutable tensor interpolation routes, never dense matrices."""
    solid = plan.solid
    positions = np.asarray(
        plan.operators.discretization.face_centers[axis], dtype=np.float64
    ).reshape((-1, solid.dimension))
    supported = np.all(np.isfinite(positions), axis=1)
    indices: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    fractions: list[NDArray[np.float64]] = []
    for direction, coordinates in enumerate(solid.splat.layout.coordinates_by_axis):
        nodes = np.asarray(coordinates, dtype=np.float64)
        if nodes.size < 2 or np.any(~np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0):
            raise ValueError(
                "MPM interpolation axes require ordered finite background nodes."
            )
        coordinate = positions[:, direction]
        structured_axis = solid.splat.plan.target.structured_axes[direction]
        if structured_axis.periodic:
            period = float(structured_axis.bounds[1] - structured_axis.bounds[0])
            extended = np.concatenate((nodes, nodes[:1] + period))
            if extended[-1] <= extended[-2]:
                raise ValueError("Periodic MPM nodes must omit the repeated endpoint.")
            coordinate = nodes[0] + np.mod(coordinate - nodes[0], period)
            lower = np.clip(
                np.searchsorted(extended, coordinate, side="right") - 1, 0, nodes.size - 1
            )
            fraction = (coordinate - extended[lower]) / (
                extended[lower + 1] - extended[lower]
            )
            upper = (lower + 1) % nodes.size
        else:
            supported &= (coordinate >= nodes[0]) & (coordinate <= nodes[-1])
            lower = np.clip(
                np.searchsorted(nodes, coordinate, side="right") - 1, 0, nodes.size - 2
            )
            upper = lower + 1
            fraction = (coordinate - nodes[lower]) / (nodes[upper] - nodes[lower])
        indices.append((lower, upper))
        fractions.append(fraction)
    sources: list[NDArray[np.int32]] = []
    targets: list[NDArray[np.int32]] = []
    weights: list[NDArray[np.float64]] = []
    weight_sum = np.zeros((positions.shape[0],), dtype=np.float64)
    face_indices = np.arange(positions.shape[0], dtype=np.int32) + offset
    for corner in product((0, 1), repeat=solid.dimension):
        node_indices = tuple(
            pair[side] for pair, side in zip(indices, corner, strict=True)
        )
        weight = np.ones((positions.shape[0],), dtype=np.float64)
        for fraction, side in zip(fractions, corner, strict=True):
            weight *= fraction if side else 1.0 - fraction
        retained = supported & (weight > 0.0)
        source = np.ravel_multi_index(node_indices, solid.nodal_shape)
        sources.append((source[retained] * solid.dimension + axis).astype(np.int32))
        targets.append(face_indices[retained])
        weights.append(weight[retained])
        weight_sum += np.where(supported, weight, 0.0)
    return (
        np.concatenate(sources),
        np.concatenate(targets),
        np.concatenate(weights),
        supported,
        weight_sum,
    )


@final
class PreparedMPMToMACFaceTransfer(StrictModule, NonTrainableState):
    """Fixed-capacity nodal/face map and its exact algebraic force transpose.

    Failed support or capacity admission leaves the entire route map inert;
    consumers must reject using ``evidence.successful``. A truncated transfer
    is never used as a physical interpolation.
    """

    solid: PreparedMPMDynamics
    operators: PreparedMACOperators
    relation: EdgeRelation
    weights: Array
    evidence: MPMToMACFaceTransferEvidence
    face_offsets: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: MPMToMACFaceTransferPlan, /) -> None:
        layouts = plan.operators.discretization.face_layouts
        offsets = [0]
        for layout in layouts:
            offsets.append(offsets[-1] + prod(layout.shape))
        routes = tuple(
            _face_routes(plan, axis, offsets[axis]) for axis in range(len(layouts))
        )
        source = np.concatenate(tuple(route[0] for route in routes))
        target = np.concatenate(tuple(route[1] for route in routes))
        weights = np.concatenate(tuple(route[2] for route in routes))
        supported = all(bool(np.all(route[3])) for route in routes)
        defect = max(float(np.max(np.abs(route[4] - 1.0))) for route in routes)
        supported = (
            supported and defect <= 8.0 * plan.solid.dimension * np.finfo(np.float64).eps
        )
        required = source.size
        capacity_ok = required <= plan.maximum_routes
        admitted = supported and capacity_ok
        padded_source = np.zeros((plan.maximum_routes,), dtype=np.int32)
        padded_target = np.zeros((plan.maximum_routes,), dtype=np.int32)
        padded_weights = np.zeros((plan.maximum_routes,), dtype=np.float64)
        valid = np.zeros((plan.maximum_routes,), dtype=np.bool_)
        if admitted:
            padded_source[:required] = source
            padded_target[:required] = target
            padded_weights[:required] = weights
            valid[:required] = True
        self.solid = plan.solid
        self.operators = plan.operators
        self.relation = EdgeRelation(
            padded_source,
            padded_target,
            source_size=prod(plan.solid.nodal_shape) * plan.solid.dimension,
            target_size=offsets[-1],
            valid=valid,
        )
        self.weights = jnp.asarray(
            padded_weights, dtype=plan.operators.pressure_space.dtype
        )
        self.face_offsets = tuple(offsets)
        self.evidence = MPMToMACFaceTransferEvidence(
            supported=jnp.asarray(supported, dtype=jnp.bool_),
            capacity_satisfied=jnp.asarray(capacity_ok, dtype=jnp.bool_),
            successful=jnp.asarray(admitted, dtype=jnp.bool_),
            face_supported=tuple(
                jnp.asarray(route[3].reshape(layout.shape), dtype=jnp.bool_)
                for route, layout in zip(routes, layouts, strict=True)
            ),
            partition_unity_defect=jnp.asarray(defect, dtype=self.weights.dtype),
            required_routes=required,
            maximum_routes=plan.maximum_routes,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mpm-to-mac-face-transfer",
                "plan": plan.plan_id,
                "source": array_tree_fingerprint(padded_source),
                "target": array_tree_fingerprint(padded_target),
                "weights": array_tree_fingerprint(padded_weights),
                "admitted": admitted,
            }
        )

    def validate_nodal_velocity(self, value: Array, /) -> Array:
        expected = self.solid.nodal_shape + (self.solid.dimension,)
        if value.shape != expected or value.dtype != self.weights.dtype:
            raise ValueError(
                f"Solid nodal vectors require shape {expected} and dtype {self.weights.dtype}."
            )
        return value

    def velocity(self, nodal_velocity: Array, /) -> FaceVelocity:
        value = self.validate_nodal_velocity(nodal_velocity)
        flat = linear_apply(self.relation, self.weights, value.reshape((-1,)))
        return tuple(
            flat[start:stop].reshape(layout.shape)
            for start, stop, layout in zip(
                self.face_offsets[:-1],
                self.face_offsets[1:],
                self.operators.discretization.face_layouts,
                strict=True,
            )
        )

    def force_adjoint(self, face_force: FaceVelocity, /) -> Array:
        values = self.operators.validate_velocity(face_force)
        flat = jnp.concatenate(tuple(value.reshape((-1,)) for value in values))
        result = linear_transpose_apply(self.relation, self.weights, flat)
        return result.reshape(self.solid.nodal_shape + (self.solid.dimension,))


@checked
def cell_to_faces(operators: PreparedMACOperators, cell_array: Array, /) -> FaceVelocity:
    """Arithmetic centered cell interpolation, with replicated boundary values.

    Periodic faces average the current and previous cells, matching the MAC
    face indexing. The two phase fractions are interpolated independently.
    """
    value = operators.validate_pressure(cell_array)
    faces = []
    for axis, structured_axis in enumerate(operators.discretization.grid.structured_axes):
        if structured_axis.periodic:
            face = 0.5 * (value + jnp.roll(value, 1, axis=axis))
        else:
            moved = jnp.moveaxis(value, axis, 0)
            centered = 0.5 * (moved[:-1] + moved[1:])
            face = jnp.moveaxis(
                jnp.concatenate((moved[:1], centered, moved[-1:]), axis=0), 0, axis
            )
        faces.append(face)
    return tuple(faces)


@final
class MixedMPMPressureEvidence(StrictModule):
    """Physical phase-mass admission; zero masses are excluded, never floored."""

    solid_zero_mass: Array
    fluid_zero_mass: tuple[Array, ...]
    solid_mass_valid: Array
    fluid_mass_valid: Array
    fractions_valid: Array
    successful: Array


def _fluid_divergence(
    operators: PreparedMACOperators,
    fraction: FaceVelocity,
    active: tuple[Array, ...],
    velocity: FaceVelocity,
    /,
) -> Array:
    values = operators.validate_velocity(velocity)
    return operators.divergence(
        tuple(
            weight * jnp.where(mask, component, jnp.zeros((), dtype=component.dtype))
            for weight, mask, component in zip(fraction, active, values, strict=True)
        )
    )


def _solid_divergence(
    transfer: PreparedMPMToMACFaceTransfer,
    fraction: FaceVelocity,
    active: Array,
    velocity: Array,
    /,
) -> Array:
    value = transfer.validate_nodal_velocity(velocity)
    faces = transfer.velocity(jnp.where(active, value, jnp.zeros((), dtype=value.dtype)))
    return transfer.operators.divergence(
        tuple(
            weight * component for weight, component in zip(fraction, faces, strict=True)
        )
    )


@final
class MixedMPMPressureOperator(StrictModule, NonTrainableState):
    """Matrix-free ``D M^-1 D^T`` for distinct solid and fluid velocity DOFs.

    ``D_s = div F_s T_s`` and ``D_f = div F_f`` include the admitted active
    DOF masks. Their transposes use the ordinary cell/velocity Euclidean
    pairing, not the weighted MAC Hilbert gradient. Phase masses are physical
    deposited solid nodal masses and physical fluid face masses: fractions or
    densities must not be applied to them a second time. There is no solve or
    pressure gauge here; those belong to the coupled projection owner.

    Solid activity may be nodal or componentwise (for wall constraints); the
    stored mask always has the full nodal-vector shape.
    """

    transfer: PreparedMPMToMACFaceTransfer
    solid_fraction: FaceVelocity
    fluid_fraction: FaceVelocity
    solid_mass: Array
    fluid_mass: FaceVelocity
    solid_active: Array
    fluid_active: tuple[Array, ...]
    evidence: MixedMPMPressureEvidence

    @checked
    def __init__(
        self,
        transfer: PreparedMPMToMACFaceTransfer,
        solid_fraction: FaceVelocity,
        fluid_fraction: FaceVelocity,
        solid_mass: Array,
        fluid_mass: FaceVelocity,
        /,
        *,
        solid_active: Array | None = None,
        fluid_active: tuple[Array, ...] | None = None,
    ) -> None:
        operators = transfer.operators
        sf = operators.validate_velocity(solid_fraction)
        ff = operators.validate_velocity(fluid_fraction)
        fm = operators.validate_velocity(fluid_mass)
        if (
            solid_mass.shape != transfer.solid.nodal_shape
            or solid_mass.dtype != transfer.weights.dtype
        ):
            raise ValueError(
                "Solid mass must have the MPM nodal shape and MAC numerical dtype."
            )
        vector_shape = solid_mass.shape + (transfer.solid.dimension,)
        sa = (
            jnp.ones(vector_shape, dtype=jnp.bool_)
            if solid_active is None
            else solid_active
        )
        if sa.dtype != jnp.bool_ or sa.shape not in (solid_mass.shape, vector_shape):
            raise ValueError(
                "solid_active must be a Boolean nodal or nodal-component array."
            )
        if sa.shape == solid_mass.shape:
            sa = jnp.broadcast_to(sa[..., None], vector_shape)
        fa = (
            tuple(jnp.ones(value.shape, dtype=jnp.bool_) for value in fm)
            if fluid_active is None
            else fluid_active
        )
        if len(fa) != len(fm) or any(
            mask.shape != value.shape or mask.dtype != jnp.bool_
            for mask, value in zip(fa, fm, strict=True)
        ):
            raise ValueError(
                "fluid_active must contain one Boolean array per MAC face layout."
            )
        solid_valid = jnp.all(jnp.isfinite(solid_mass) & (solid_mass >= 0.0))
        fluid_valid = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value) & (value >= 0.0)) for value in fm)
            )
        )
        fractions_valid = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value) & (value >= 0.0) & (value <= 1.0))
                    for value in (*sf, *ff)
                )
            )
        )
        self.transfer = transfer
        self.solid_fraction = sf
        self.fluid_fraction = ff
        self.solid_mass = solid_mass
        self.fluid_mass = fm
        self.solid_active = (
            sa & (jnp.isfinite(solid_mass) & (solid_mass > 0.0))[..., None]
        )
        self.fluid_active = tuple(
            mask & jnp.isfinite(value) & (value > 0.0)
            for mask, value in zip(fa, fm, strict=True)
        )
        self.evidence = MixedMPMPressureEvidence(
            solid_zero_mass=sa & (solid_mass == 0.0)[..., None],
            fluid_zero_mass=tuple(
                mask & (value == 0.0) for mask, value in zip(fa, fm, strict=True)
            ),
            solid_mass_valid=solid_valid,
            fluid_mass_valid=fluid_valid,
            fractions_valid=fractions_valid,
            successful=transfer.evidence.successful
            & solid_valid
            & fluid_valid
            & fractions_valid,
        )

    def solid_divergence(self, velocity: Array, /) -> Array:
        return _solid_divergence(
            self.transfer, self.solid_fraction, self.solid_active, velocity
        )

    def fluid_divergence(self, velocity: FaceVelocity, /) -> Array:
        return _fluid_divergence(
            self.transfer.operators, self.fluid_fraction, self.fluid_active, velocity
        )

    def divergence(self, solid_velocity: Array, fluid_velocity: FaceVelocity, /) -> Array:
        return self.solid_divergence(solid_velocity) + self.fluid_divergence(
            fluid_velocity
        )

    def solid_transpose(self, pressure: Array, /) -> Array:
        value = self.transfer.operators.validate_pressure(pressure)
        zero = jnp.zeros(
            self.transfer.solid.nodal_shape + (self.transfer.solid.dimension,),
            dtype=self.solid_mass.dtype,
        )
        # Partial carries all numerical leaves dynamically; no array-bearing
        # callback is static and no weighted MAC gradient is substituted for D^T.
        action = eqx.Partial(
            _solid_divergence, self.transfer, self.solid_fraction, self.solid_active
        )
        return jax.linear_transpose(action, zero)(value)[0]

    def fluid_transpose(self, pressure: Array, /) -> FaceVelocity:
        value = self.transfer.operators.validate_pressure(pressure)
        zero = tuple(jnp.zeros_like(mass) for mass in self.fluid_mass)
        action = eqx.Partial(
            _fluid_divergence,
            self.transfer.operators,
            self.fluid_fraction,
            self.fluid_active,
        )
        return jax.linear_transpose(action, zero)(value)[0]

    def inverse_mass_s(self, force: Array, /) -> Array:
        value = self.transfer.validate_nodal_velocity(force)
        # The unit denominator is evaluated only on excluded DOFs, whose force
        # and output are zero. No positive mass is replaced or bounded below.
        denominator = jnp.where(
            self.solid_active,
            self.solid_mass[..., None],
            jnp.ones((), dtype=self.solid_mass.dtype),
        )
        return (
            jnp.where(self.solid_active, value, jnp.zeros((), dtype=value.dtype))
            / denominator
        )

    def inverse_mass_f(self, force: FaceVelocity, /) -> FaceVelocity:
        values = self.transfer.operators.validate_velocity(force)
        return tuple(
            jnp.where(mask, value, jnp.zeros((), dtype=value.dtype))
            / jnp.where(mask, mass, jnp.ones((), dtype=mass.dtype))
            for value, mass, mask in zip(
                values, self.fluid_mass, self.fluid_active, strict=True
            )
        )

    def pressure_action(self, pressure: Array, /) -> Array:
        return self.divergence(
            self.inverse_mass_s(self.solid_transpose(pressure)),
            self.inverse_mass_f(self.fluid_transpose(pressure)),
        )


__all__ = [
    "cell_to_faces",
    "MixedMPMPressureEvidence",
    "MixedMPMPressureOperator",
    "MPMToMACFaceTransferEvidence",
    "MPMToMACFaceTransferPlan",
    "PreparedMPMToMACFaceTransfer",
]
