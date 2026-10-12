#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified bounded cut fractions and measure-adjoint mixture direct forcing.

A uniform Cartesian subdivision encloses each body's field on each subbox by
``value ± (L*r + error)``. Strictly negative upper bounds certify solid;
strictly positive lower bounds certify fluid. The union uses any certified
solid body and all certified fluid bodies, rather than summing body fractions.
Midpoints of the resulting volume bounds are estimates, not exact cut volumes.
The unresolved-fraction admission limit applies independently to every cell.

The Phydrax discrete mass-fraction formulation is rho = eps*rho_f + phi*rho_s,
psi = phi*rho_s/rho and f_IB = rho*psi*(u_s-u_predictor)/dt. Body load density
is alpha*(psi*b_sigma - (1-psi)*f_IB). Alpha is the normalized per-body fraction
at the same quadrature location. Face-dual quadrature approximates the continuum
cell-volume integral, but its rigid velocity/load reduction is exactly adjoint:
it preserves resultant force, moment, and virtual work. The direct body reaction
balances the EXTERNAL-fluid force (1-psi)*f_IB, not the internal mixture force.
Pressure/viscous load and internal-fluid acceleration are separate ledgers.

Direct forcing and internal-fluid correction are discussed by Uhlmann (2005),
JCP 209, doi:10.1016/j.jcp.2005.03.017 (Appendix B). Dave et al. (2023), VFIB,
distinguishes internal and external stresses; neither reference is claimed to
publish the Phydrax-owned mass-fraction discretization above.
"""

from __future__ import annotations

from enum import IntFlag
from operator import index
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..discretization.finite_volume._mac_variable_density import _face_interpolate
from ..discretization.particle._dem import PreparedCompliantDEMDynamics
from ..discretization.particle._rigid_body import RigidBodyKinematics
from ..discretization.particle._rigid_shape import AbstractPreparedRigidContactShape
from ..discretization.particle._rigid_sphere import (
    sphere_lever_torque,
    sphere_spin_velocity,
)
from ..discretization.particle._signed_field import PosedSignedFieldQueryResult
from ..typing import checked
from ._mac_variable_density import CompiledMACVariableDensityDynamics


if TYPE_CHECKING:
    from ..solver._mac_variable_density import MACVariableDensityProjectionPlan


@final
class MACSDFCutFractionEvaluation(StrictModule):
    """Certified cell enclosures; all numerical estimates remain within bounds.

    Body arrays have shape (body_capacity, *cell_shape). Union arrays have
    cell_shape. geometry_margin is maximum_unresolved_fraction minus the
    per-cell unresolved volume. query_reliable excludes inactive shape lanes.
    Unqueried and unreliable active subboxes remain [0, 1], never fabricated.
    """

    solid_fraction_by_body: Array
    solid_fraction: Array
    fluid_fraction: Array
    fraction_lower: Array
    fraction_upper: Array
    fraction_lower_by_body: Array
    fraction_upper_by_body: Array
    unresolved_fraction: Array
    geometry_margin: Array
    query_overflow: Array
    query_reliable: Array
    domain_supported: Array
    query_count: Array
    query_capacity: int = eqx.field(static=True)
    successful: Array
    plan_id: str = eqx.field(static=True)


@final
class MACSDFCutFractionPlan(StrictModule, NonTrainableState):
    """Fixed dense body × cell × subbox route, with explicit query admission.

    subdivisions is the number of equal divisions on EACH coordinate axis.
    Refining by an integer multiple nests the subboxes. The required route
    count is body_capacity * cell_count * subdivisions**dimension, including
    native inactive lanes. maximum_queries bounds the actually queried prefix;
    overflow returns failed enclosure evidence without reallocating capacity.
    Domain support requires every active bounding sphere to lie wholly inside
    the grid, even on periodic axes: this route has no periodic image queries.
    """

    operators: PreparedMACOperators
    shape_query: AbstractPreparedRigidContactShape
    query_body_indices: Array
    query_points: Array
    query_radii: Array
    subdivisions: int = eqx.field(static=True)
    maximum_unresolved_fraction: float = eqx.field(static=True)
    maximum_queries: int = eqx.field(static=True)
    required_queries: int = eqx.field(static=True)
    subboxes_per_cell: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        operators: PreparedMACOperators,
        shape_query: AbstractPreparedRigidContactShape,
        /,
        *,
        subdivisions: int = 4,
        maximum_unresolved_fraction: float = 0.02,
        maximum_queries: int | None = None,
    ) -> None:
        if isinstance(subdivisions, bool):
            raise TypeError("subdivisions must be an integer.")
        divisions = index(subdivisions)
        maximum = float(maximum_unresolved_fraction)
        if divisions <= 0 or not np.isfinite(maximum) or not 0.0 <= maximum <= 1.0:
            raise ValueError(
                "Cut-fraction subdivisions and unresolved limit are invalid."
            )
        if shape_query.shape_kind == "superquadric":
            raise ValueError(
                "Superquadric shapes have no admitted signed-field enclosure."
            )
        dimension = len(operators.discretization.cell_shape)
        if dimension != shape_query.bodies.ambient_dimension:
            raise ValueError("Cut-fraction grid and rigid-body dimensions differ.")
        boxes = divisions**dimension
        cell_count = operators.discretization.cell_volumes.size
        required = shape_query.bodies.capacity * cell_count * boxes
        if isinstance(maximum_queries, bool):
            raise TypeError("maximum_queries must be an integer or None.")
        capacity = required if maximum_queries is None else index(maximum_queries)
        if capacity < 0:
            raise ValueError("maximum_queries must be nonnegative.")
        capacity = min(capacity, required)
        # Preparation constructs only the admitted query prefix; route ordering
        # is body, C-order cell, C-order subbox. Geometry stays dynamic PyTree data.
        axes = operators.discretization.grid.structured_axes
        centers = np.asarray(
            operators.discretization.cell_centers, dtype=np.float64
        ).reshape(-1, dimension)
        width_mesh = np.meshgrid(
            *(np.asarray(axis.interval_widths, dtype=np.float64) for axis in axes),
            indexing="ij",
        )
        widths = np.stack(width_mesh, axis=-1).reshape(-1, dimension)
        offsets = np.stack(
            np.meshgrid(
                *(np.arange(divisions, dtype=np.float64) for _ in axes), indexing="ij"
            ),
            axis=-1,
        ).reshape(boxes, dimension)
        offsets = (offsets + 0.5) / divisions - 0.5
        route = np.arange(capacity, dtype=np.int64)
        body_indices = route // (cell_count * boxes)
        cells = (route // boxes) % cell_count
        box_indices = route % boxes
        points = centers[cells] + widths[cells] * offsets[box_indices]
        dtype = operators.pressure_space.dtype
        quantized_points = np.asarray(points, dtype=np.dtype(dtype))
        halfwidth = widths[cells] / (2.0 * divisions)
        halfwidth = halfwidth + np.abs(quantized_points.astype(np.float64) - points)
        radii = np.nextafter(np.sqrt(np.sum(halfwidth**2, axis=-1)), np.inf)
        canonical_radii = jnp.asarray(radii, dtype=dtype)
        self.operators = operators
        self.shape_query = shape_query
        self.query_body_indices = jnp.asarray(body_indices, dtype=jnp.int32)
        self.query_points = jnp.asarray(quantized_points, dtype=dtype)
        self.query_radii = jnp.nextafter(
            canonical_radii, jnp.asarray(jnp.inf, dtype=dtype)
        )
        self.subdivisions = divisions
        self.maximum_unresolved_fraction = maximum
        self.maximum_queries = capacity
        self.required_queries = required
        self.subboxes_per_cell = boxes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-sdf-cut-fraction-plan",
                "operators": operators.prepared_id,
                "shape": shape_query.shape_id,
                "bodies": shape_query.bodies.prepared_id,
                "subdivisions": divisions,
                "maximum_unresolved_fraction_per_cell": maximum,
                "maximum_queries": capacity,
                "required_queries": required,
            }
        )

    @checked
    def evaluate(
        self,
        kinematics: RigidBodyKinematics,
        time: ArrayLike,
        /,
        *,
        shape_scale: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
    ) -> MACSDFCutFractionEvaluation:
        bodies = self.shape_query.bodies
        _validate_kinematics(bodies.capacity, bodies.ambient_dimension, kinematics)
        dtype = self.operators.pressure_space.dtype
        instant = jnp.asarray(time, dtype=dtype)
        if instant.shape != ():
            raise ValueError("time must be scalar.")
        scale = None if shape_scale is None else jnp.asarray(shape_scale, dtype=dtype)
        if scale is not None and scale.shape != (bodies.capacity,):
            raise ValueError("shape_scale must match body capacity.")
        active = bodies.particles.active_mask
        if active_mask is not None:
            supplied = jnp.asarray(active_mask)
            if supplied.shape != active.shape or supplied.dtype != jnp.bool_:
                raise TypeError("active_mask must be a boolean body-capacity vector.")
            active = active & supplied
        axes = self.operators.discretization.grid.structured_axes
        lower_domain = jnp.asarray(tuple(axis.bounds[0] for axis in axes), dtype=dtype)
        upper_domain = jnp.asarray(tuple(axis.bounds[1] for axis in axes), dtype=dtype)
        body_scale = jnp.ones((bodies.capacity,), dtype=dtype) if scale is None else scale
        radius = self.shape_query.bounding_radii * body_scale
        contained = jnp.all(
            (kinematics.position - radius[:, None] >= lower_domain[None, :])
            & (kinematics.position + radius[:, None] <= upper_domain[None, :]),
            axis=-1,
        )
        domain_supported = jnp.all(
            ~active | (contained & jnp.isfinite(radius) & (radius > 0.0))
        )
        shape = (
            bodies.capacity,
            *self.operators.discretization.cell_shape,
            self.subboxes_per_cell,
        )
        active_boxes = active.reshape((bodies.capacity,) + (1,) * (len(shape) - 1))
        solid = jnp.zeros((self.required_queries,), dtype=jnp.bool_)
        fluid = jnp.zeros_like(solid)
        reliable = jnp.asarray(True)
        if self.maximum_queries:
            enabled = active[self.query_body_indices]
            query = _enabled_shape_query(
                self.shape_query,
                kinematics,
                self.query_body_indices,
                self.query_points,
                scale,
                enabled,
            )
            if query.shape_id != self.shape_query.shape_id:
                raise ValueError("Signed-field query evidence names a different shape.")
            expected = (self.maximum_queries,)
            if (
                query.successful.shape != ()
                or query.successful.dtype != jnp.bool_
                or query.supported.dtype != jnp.bool_
                or any(
                    value.shape != expected
                    for value in (
                        query.value,
                        query.lipschitz_bound,
                        query.evaluation_error,
                        query.supported,
                    )
                )
            ):
                raise ValueError(
                    "Signed-field query evidence must match the prepared route."
                )
            lane_valid = (
                query.supported
                & jnp.isfinite(query.value)
                & jnp.isfinite(query.lipschitz_bound)
                & (query.lipschitz_bound >= 0.0)
                & jnp.isfinite(query.evaluation_error)
                & (query.evaluation_error >= 0.0)
            )
            reliable = query.successful & jnp.all(~enabled | lane_valid)
            enclosure = query.lipschitz_bound * self.query_radii + query.evaluation_error
            solid = solid.at[: self.maximum_queries].set(
                enabled & lane_valid & (query.value + enclosure < 0.0)
            )
            fluid = fluid.at[: self.maximum_queries].set(
                enabled & lane_valid & (query.value - enclosure > 0.0)
            )
        solid = solid.reshape(shape) & active_boxes
        fluid = fluid.reshape(shape) | ~active_boxes
        lower_body = jnp.mean(solid.astype(dtype), axis=-1)
        upper_body = jnp.mean((~fluid).astype(dtype), axis=-1)
        lower = jnp.mean(jnp.any(solid, axis=0).astype(dtype), axis=-1)
        upper = jnp.mean((~jnp.all(fluid, axis=0)).astype(dtype), axis=-1)
        unresolved = upper - lower
        margin = self.maximum_unresolved_fraction - unresolved
        overflow = jnp.asarray(self.maximum_queries < self.required_queries)
        reliable = reliable & jnp.isfinite(instant)
        successful = reliable & domain_supported & ~overflow & jnp.all(margin >= 0.0)
        fraction = 0.5 * (lower + upper)
        return MACSDFCutFractionEvaluation(
            0.5 * (lower_body + upper_body),
            fraction,
            1.0 - fraction,
            lower,
            upper,
            lower_body,
            upper_body,
            unresolved,
            margin,
            overflow,
            reliable,
            domain_supported,
            jnp.asarray(self.maximum_queries, dtype=jnp.int32),
            self.maximum_queries,
            successful,
            self.plan_id,
        )


def _enabled_shape_query(
    shape: AbstractPreparedRigidContactShape,
    kinematics: RigidBodyKinematics,
    body_indices: Array,
    points: Array,
    shape_scale: Array | None,
    enabled: Array,
    /,
) -> PosedSignedFieldQueryResult:
    """Duplicate an enabled route into inactive lanes; preserve native success.

    The route remains fixed capacity. Disabled queries cannot poison aggregate
    native evidence, nor can they excuse an unexplained native batch failure.
    When no route is enabled, do not execute a shape query at runtime.
    """

    def evaluate(_: None) -> PosedSignedFieldQueryResult:
        first = jnp.argmax(enabled.astype(jnp.int32))
        indices = jnp.where(enabled, body_indices, body_indices[first])
        locations = jnp.where(enabled[:, None], points, points[first])
        return shape.evaluate_signed_field_query(
            kinematics,
            indices,
            locations,
            shape_scale=shape_scale,
        )

    def empty(_: None) -> PosedSignedFieldQueryResult:
        zeros = jnp.zeros((body_indices.shape[0],), dtype=points.dtype)
        return PosedSignedFieldQueryResult(
            zeros,
            jnp.ones_like(zeros),
            zeros,
            jnp.ones_like(enabled),
            jnp.asarray(True),
            shape.shape_id,
        )

    return jax.lax.cond(jnp.any(enabled), evaluate, empty, operand=None)


def _validate_kinematics(
    capacity: int, dimension: int, kinematics: RigidBodyKinematics, /
) -> None:
    if kinematics.position.shape != (
        capacity,
        dimension,
    ) or kinematics.velocity.shape != (capacity, dimension):
        raise ValueError(
            "Rigid-body position and velocity do not match prepared capacity."
        )
    if kinematics.angular_velocity.shape != (
        capacity,
        1 if dimension == 2 else 3,
    ) or kinematics.orientation.shape != (capacity, 1 if dimension == 2 else 4):
        raise ValueError(
            "Rigid-body orientation and spin do not match prepared capacity."
        )


@checked
def centered_face_values(
    operators: PreparedMACOperators, cell: ArrayLike, /
) -> FaceVelocity:
    """Native centered cell-to-face map; boundary faces inherit adjacent cells."""
    value = operators.validate_pressure(cell)
    faces = []
    for axis, structured in enumerate(operators.discretization.grid.structured_axes):
        faces.append(
            _face_interpolate(
                value,
                axis,
                structured.periodic,
                jnp.take(value, 0, axis=axis),
                jnp.take(value, value.shape[axis] - 1, axis=axis),
            )
        )
    return tuple(faces)


class MACVolumeFractionIBStatus(IntFlag):
    SUCCESS = 0
    INVALID_STEP_SIZE = 1
    GEOMETRY_FAILED = 2
    QUERY_OVERFLOW = 4
    NONFINITE = 8
    IMPULSE_IDENTITY_FAILED = 16
    TORQUE_IDENTITY_FAILED = 32
    WORK_IDENTITY_FAILED = 64
    PROJECTION_FAILED = 128
    DEM_FAILED = 256
    CANDIDATE_GEOMETRY_FAILED = 512
    DOMAIN_UNSUPPORTED = 1024
    BOUNDARY_INTERSECTION = 2048


@final
class MACVolumeFractionIBCFDEMCouplingPlan(StrictModule, NonTrainableState):
    """Immutable geometry, fluid, DEM and projection scientific binding."""

    fluid: CompiledMACVariableDensityDynamics
    dynamics: PreparedCompliantDEMDynamics
    cut_fraction: MACSDFCutFractionPlan
    projection: MACVariableDensityProjectionPlan
    fluid_density: Array
    solid_density: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid: CompiledMACVariableDensityDynamics,
        dynamics: PreparedCompliantDEMDynamics,
        cut_fraction: MACSDFCutFractionPlan,
        projection: MACVariableDensityProjectionPlan,
        /,
        *,
        fluid_density: ArrayLike,
        solid_density: ArrayLike,
    ) -> None:
        from ..solver._mac_variable_density import MACVariableDensityProjectionPlan

        if not isinstance(fluid, CompiledMACVariableDensityDynamics) or not isinstance(
            dynamics, PreparedCompliantDEMDynamics
        ):
            raise TypeError(
                "Coupling requires canonical variable-density fluid and prepared DEM dynamics."
            )
        if not isinstance(cut_fraction, MACSDFCutFractionPlan) or not isinstance(
            projection, MACVariableDensityProjectionPlan
        ):
            raise TypeError(
                "Coupling requires canonical cut-fraction and projection plans."
            )
        if np.iscomplexobj(fluid_density) or np.iscomplexobj(solid_density):
            raise TypeError("Intrinsic densities must be real.")
        rho_f = np.asarray(fluid_density, dtype=np.float64)
        rho_s = np.asarray(solid_density, dtype=np.float64)
        if (
            rho_f.shape != ()
            or rho_s.shape != ()
            or not np.isfinite(rho_f)
            or not np.isfinite(rho_s)
            or rho_f <= 0.0
            or rho_s <= 0.0
        ):
            raise ValueError(
                "Intrinsic fluid and solid densities must be positive finite scalars."
            )
        operators = fluid.variable_density.operators
        if any(
            owner.prepared_id != operators.prepared_id
            for owner in (
                cut_fraction.operators,
                projection.operators,
                fluid.projection.operators,
            )
        ):
            raise ValueError(
                "Fluid, cut fraction and projections must share prepared MAC operators."
            )
        if (
            cut_fraction.shape_query.shape_id != dynamics.shapes.shape_id
            or cut_fraction.shape_query.bodies.prepared_id != dynamics.bodies.prepared_id
        ):
            raise ValueError(
                "Cut-fraction shape and body identity must match DEM dynamics."
            )
        active = np.asarray(dynamics.bodies.particles.active_mask)
        volumes = np.asarray(dynamics.shapes.volumes)
        masses = np.asarray(dynamics.bodies.mass_properties.masses)
        if (
            volumes.shape != masses.shape
            or np.any(~np.isfinite(volumes[active]))
            or np.any(volumes[active] <= 0.0)
            or not np.allclose(
                masses[active],
                rho_s * volumes[active],
                rtol=128.0 * np.finfo(masses.dtype).eps,
                atol=0.0,
            )
        ):
            raise ValueError(
                "solid_density must match active DEM reference mass divided by shape volume."
            )
        dtype = operators.pressure_space.dtype
        self.fluid = fluid
        self.dynamics = dynamics
        self.cut_fraction = cut_fraction
        self.projection = projection
        self.fluid_density = jnp.asarray(rho_f, dtype=dtype)
        self.solid_density = jnp.asarray(rho_s, dtype=dtype)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-volume-fraction-ib-cfd-dem-coupling-plan",
                "fluid": fluid.compilation_id,
                "dynamics": dynamics.prepared_id,
                "cut_fraction": cut_fraction.plan_id,
                "projection": projection.plan_id,
                "fluid_density": rho_f,
                "solid_density": rho_s,
            }
        )


@final
class MACVolumeFractionIBEvaluation(StrictModule):
    """Mixture forcing, separate stress loads, and exact discrete balance evidence."""

    cut_fraction: MACSDFCutFractionEvaluation

    mixture_density: Array
    solid_mass_fraction: Array
    solid_velocity: Array
    fluid_ib_force_density: FaceVelocity
    body_force: Array
    body_torque: Array
    pressure_viscous_load: Array
    pressure_viscous_torque: Array
    fluid_external_impulse: Array
    body_direct_impulse: Array
    fluid_external_angular_impulse: Array
    body_direct_angular_impulse: Array
    fluid_internal_impulse: Array
    impulse_residual: Array
    torque_residual: Array
    work_adjoint_residual: Array
    face_mixture_density: FaceVelocity
    boundary_supported: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _body_shares(fractions: Array, /) -> Array:
    total = jnp.sum(fractions, axis=0)
    return fractions / jnp.where(total > 0.0, total, 1.0)[None, ...]


def _rigid_velocity(
    kinematics: RigidBodyKinematics, points: Array, active: Array, /
) -> tuple[Array, Array]:
    dimension = kinematics.position.shape[1]
    batch = (kinematics.position.shape[0],) + (1,) * (points.ndim - 1)
    center = jnp.where(active[:, None], kinematics.position, 0.0).reshape(
        batch + (dimension,)
    )
    velocity = jnp.where(active[:, None], kinematics.velocity, 0.0).reshape(
        batch + (dimension,)
    )
    spin = jnp.where(active[:, None], kinematics.angular_velocity, 0.0).reshape(
        batch + (kinematics.angular_velocity.shape[1],)
    )
    arm = points[None, ...] - center
    rigid = velocity + sphere_spin_velocity(spin, arm, dimension)
    return rigid, arm


def _component_load(
    density: Array, arm: Array, measure: Array, axis: int, dimension: int, /
) -> tuple[Array, Array]:
    weighted = density * measure[None, ...]
    direction = jax.nn.one_hot(axis, dimension, dtype=density.dtype).reshape(
        (1,) * weighted.ndim + (dimension,)
    )
    vector = weighted[..., None] * direction
    spatial = tuple(range(1, weighted.ndim))
    return jnp.sum(vector, axis=spatial), jnp.sum(
        sphere_lever_torque(arm, vector, dimension), axis=spatial
    )


def _world_component_moment(
    points: Array, density: Array, measure: Array, axis: int, /
) -> Array:
    dimension = points.shape[-1]
    direction = jax.nn.one_hot(axis, dimension, dtype=density.dtype).reshape(
        (1,) * density.ndim + (dimension,)
    )
    vector = (density * measure)[..., None] * direction
    return jnp.sum(
        sphere_lever_torque(points, vector, dimension), axis=tuple(range(density.ndim))
    )


@final
class _IBFaceLedgers(StrictModule):
    forcing: FaceVelocity
    density: FaceVelocity
    force: Array
    torque: Array
    stress_force: Array
    stress_torque: Array
    direct_force: Array
    direct_torque: Array
    external_force: Array
    external_moment: Array
    internal_force: Array
    face_work: Array
    absolute_work: Array
    absolute_force: Array
    absolute_moment: Array
    boundary_supported: Array


def _face_ledgers(
    coupling: MACVolumeFractionIBCFDEMCouplingPlan,
    kinematics: RigidBodyKinematics,
    fractions: MACSDFCutFractionEvaluation,
    predictor: FaceVelocity,
    stress: FaceVelocity,
    safe_step: Array,
    /,
) -> _IBFaceLedgers:
    operators = coupling.cut_fraction.operators
    bodies = coupling.dynamics.bodies
    dimension = bodies.ambient_dimension
    active = bodies.particles.active_mask
    phi_faces = centered_face_values(operators, fractions.solid_fraction)

    # Each axis has a distinct native face shape; vectorize only its body axis.
    def interpolate_body(value: Array) -> FaceVelocity:
        return centered_face_values(operators, value)

    per_body = jax.vmap(interpolate_body)(fractions.solid_fraction_by_body)
    force = jnp.zeros_like(kinematics.position)
    torque = jnp.zeros_like(kinematics.angular_velocity)
    stress_force = jnp.zeros_like(force)
    stress_torque = jnp.zeros_like(torque)
    direct_force = jnp.zeros_like(force)
    direct_torque = jnp.zeros_like(torque)
    external = jnp.zeros((dimension,), dtype=force.dtype)
    external_moment = jnp.zeros((bodies.angular_dimension,), dtype=force.dtype)
    internal = jnp.zeros_like(external)
    work = jnp.asarray(0.0, dtype=force.dtype)
    absolute_work = jnp.zeros_like(work)
    absolute_force = jnp.zeros_like(work)
    absolute_moment = jnp.zeros_like(work)
    forcing = []
    densities = []
    boundary_supported = jnp.asarray(True)
    for axis, (phi, body_phi, up, sigma, measure, points) in enumerate(
        zip(
            phi_faces,
            per_body,
            predictor,
            stress,
            operators.face_dual_measures,
            operators.discretization.face_centers,
            strict=True,
        )
    ):
        alpha = _body_shares(body_phi)
        rigid_vector, arm = _rigid_velocity(kinematics, points, active)
        rigid = rigid_vector[..., axis]
        if not operators.discretization.grid.structured_axes[axis].periodic:
            boundary_supported = (
                boundary_supported
                & jnp.all(jnp.take(phi, 0, axis=axis) == 0.0)
                & jnp.all(jnp.take(phi, phi.shape[axis] - 1, axis=axis) == 0.0)
            )
        solid_velocity = jnp.sum(alpha * rigid, axis=0)
        rho = (1.0 - phi) * coupling.fluid_density + phi * coupling.solid_density
        psi = phi * coupling.solid_density / rho
        ib = rho * psi * (solid_velocity - up) / safe_step
        stress_density = alpha * (psi * sigma)[None, ...]
        direct_density = -alpha * ((1.0 - psi) * ib)[None, ...]
        total_density = stress_density + direct_density
        stress_f, stress_t = _component_load(
            stress_density, arm, measure, axis, dimension
        )
        direct_f, direct_t = _component_load(
            direct_density, arm, measure, axis, dimension
        )
        stress_force = stress_force + stress_f
        stress_torque = stress_torque + stress_t
        direct_force = direct_force + direct_f
        direct_torque = direct_torque + direct_t
        force = force + stress_f + direct_f
        torque = torque + stress_t + direct_t
        external_density = (1.0 - psi) * ib
        component_external = jnp.sum(measure * external_density)
        external = external.at[axis].set(component_external)
        internal = internal.at[axis].set(jnp.sum(measure * psi * ib))
        moment = _world_component_moment(points, external_density, measure, axis)
        external_moment = external_moment + moment
        face_work = measure[None, ...] * total_density * rigid
        work = work + jnp.sum(face_work)
        absolute_work = absolute_work + jnp.sum(jnp.abs(face_work))
        absolute_force = absolute_force + jnp.sum(jnp.abs(measure * external_density))
        moment_density = sphere_lever_torque(
            points,
            (measure * external_density)[..., None]
            * jax.nn.one_hot(axis, dimension, dtype=force.dtype).reshape(
                (1,) * external_density.ndim + (dimension,)
            ),
            dimension,
        )
        absolute_moment = absolute_moment + jnp.sum(jnp.abs(moment_density))
        forcing.append(ib)
        densities.append(rho)
    return _IBFaceLedgers(
        tuple(forcing),
        tuple(densities),
        force,
        torque,
        stress_force,
        stress_torque,
        direct_force,
        direct_torque,
        external,
        external_moment,
        internal,
        work,
        absolute_work,
        absolute_force,
        absolute_moment,
        boundary_supported,
    )


@checked
def evaluate_mac_volume_fraction_ib_cfd_dem(
    coupling: MACVolumeFractionIBCFDEMCouplingPlan,
    kinematics: RigidBodyKinematics,
    cut_fraction_evaluation: MACSDFCutFractionEvaluation,
    fluid_predictor: FaceVelocity,
    stress_force_density: FaceVelocity,
    step_size: ArrayLike,
    /,
) -> MACVolumeFractionIBEvaluation:
    """Evaluate certified mixture forcing without projecting or advancing state."""
    geometry = cut_fraction_evaluation
    if geometry.plan_id != coupling.cut_fraction.plan_id:
        raise ValueError(
            "Cut-fraction evidence does not bind the coupling geometry plan."
        )
    operators = coupling.cut_fraction.operators
    bodies = coupling.dynamics.bodies
    dimension = bodies.ambient_dimension
    _validate_kinematics(bodies.capacity, dimension, kinematics)
    if geometry.solid_fraction_by_body.shape != (
        bodies.capacity,
        *operators.discretization.cell_shape,
    ):
        raise ValueError("Body cut fractions do not match the coupling capacity.")
    phi = operators.validate_pressure(geometry.solid_fraction)
    predictor = operators.validate_velocity(fluid_predictor)
    stress = operators.validate_velocity(stress_force_density)
    step = jnp.asarray(step_size, dtype=phi.dtype)
    if step.shape != ():
        raise ValueError("step_size must be scalar.")
    valid_step = jnp.isfinite(step) & (step > 0.0)
    safe_step = jnp.where(valid_step, step, 1.0)
    rho = (1.0 - phi) * coupling.fluid_density + phi * coupling.solid_density
    psi = phi * coupling.solid_density / rho
    active = bodies.particles.active_mask
    cell_alpha = _body_shares(geometry.solid_fraction_by_body)
    cell_rigid, _ = _rigid_velocity(
        kinematics, operators.discretization.cell_centers, active
    )
    solid_velocity = jnp.sum(cell_alpha[..., None] * cell_rigid, axis=0)
    ledgers = _face_ledgers(coupling, kinematics, geometry, predictor, stress, safe_step)
    accepted_step = jnp.where(valid_step, step, 0.0)
    fluid_impulse = accepted_step * ledgers.external_force
    direct_impulse = accepted_step * ledgers.direct_force
    fluid_angular = accepted_step * ledgers.external_moment
    direct_angular = accepted_step * ledgers.direct_torque
    impulse_residual = fluid_impulse + jnp.sum(direct_impulse, axis=0)
    safe_position = jnp.where(active[:, None], kinematics.position, 0.0)
    body_world_angular = direct_angular + sphere_lever_torque(
        safe_position, direct_impulse, dimension
    )
    torque_residual = fluid_angular + jnp.sum(body_world_angular, axis=0)
    safe_velocity = jnp.where(active[:, None], kinematics.velocity, 0.0)
    safe_spin = jnp.where(active[:, None], kinematics.angular_velocity, 0.0)
    rigid_work = jnp.sum(safe_velocity * ledgers.force) + jnp.sum(
        safe_spin * ledgers.torque
    )
    work_residual = rigid_work - ledgers.face_work
    finite = jnp.all(jnp.isfinite(rho) & (rho > 0.0)) & jnp.all(
        jnp.isfinite(solid_velocity)
    )
    finite = (
        finite
        & jnp.isfinite(coupling.fluid_density)
        & (coupling.fluid_density > 0.0)
        & jnp.isfinite(coupling.solid_density)
        & (coupling.solid_density > 0.0)
    )
    finite = finite & jnp.all(
        jnp.stack(
            tuple(
                jnp.all(jnp.isfinite(value))
                for value in (
                    *ledgers.forcing,
                    *ledgers.density,
                    ledgers.force,
                    ledgers.torque,
                    ledgers.stress_force,
                    ledgers.stress_torque,
                    impulse_residual,
                    torque_residual,
                    work_residual,
                )
            )
        )
    )
    epsilon = 512.0 * jnp.finfo(phi.dtype).eps
    impulse_ok = jnp.all(
        jnp.abs(impulse_residual)
        <= epsilon
        * accepted_step
        * (ledgers.absolute_force + jnp.sum(jnp.abs(ledgers.direct_force)))
    )
    torque_scale = (
        accepted_step * ledgers.absolute_moment
        + jnp.sum(jnp.abs(direct_angular))
        + jnp.sum(jnp.abs(sphere_lever_torque(safe_position, direct_impulse, dimension)))
    )
    torque_ok = jnp.all(jnp.abs(torque_residual) <= epsilon * torque_scale)
    work_scale = (
        ledgers.absolute_work
        + jnp.sum(jnp.abs(safe_velocity * ledgers.force))
        + jnp.sum(jnp.abs(safe_spin * ledgers.torque))
    )
    work_ok = jnp.abs(work_residual) <= epsilon * work_scale
    status = jnp.asarray(MACVolumeFractionIBStatus.SUCCESS, dtype=jnp.int32)
    for failed, flag in (
        (~valid_step, MACVolumeFractionIBStatus.INVALID_STEP_SIZE),
        (~geometry.successful, MACVolumeFractionIBStatus.GEOMETRY_FAILED),
        (geometry.query_overflow, MACVolumeFractionIBStatus.QUERY_OVERFLOW),
        (~geometry.domain_supported, MACVolumeFractionIBStatus.DOMAIN_UNSUPPORTED),
        (~ledgers.boundary_supported, MACVolumeFractionIBStatus.BOUNDARY_INTERSECTION),
        (~finite, MACVolumeFractionIBStatus.NONFINITE),
        (~impulse_ok, MACVolumeFractionIBStatus.IMPULSE_IDENTITY_FAILED),
        (~torque_ok, MACVolumeFractionIBStatus.TORQUE_IDENTITY_FAILED),
        (~work_ok, MACVolumeFractionIBStatus.WORK_IDENTITY_FAILED),
    ):
        status = status | jnp.where(
            failed, jnp.asarray(flag, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)
        )
    return MACVolumeFractionIBEvaluation(
        geometry,
        rho,
        psi,
        solid_velocity,
        ledgers.forcing,
        ledgers.force,
        ledgers.torque,
        ledgers.stress_force,
        ledgers.stress_torque,
        fluid_impulse,
        direct_impulse,
        fluid_angular,
        direct_angular,
        accepted_step * ledgers.internal_force,
        impulse_residual,
        torque_residual,
        work_residual,
        ledgers.density,
        ledgers.boundary_supported,
        status,
        status == MACVolumeFractionIBStatus.SUCCESS,
        coupling.plan_id,
    )


__all__ = [
    "MACSDFCutFractionEvaluation",
    "MACSDFCutFractionPlan",
    "MACVolumeFractionIBCFDEMCouplingPlan",
    "MACVolumeFractionIBEvaluation",
    "MACVolumeFractionIBStatus",
    "centered_face_values",
    "evaluate_mac_volume_fraction_ib_cfd_dem",
]
