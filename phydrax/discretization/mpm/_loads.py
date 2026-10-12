#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import gather_patches
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..particle._core import ParticleSetPlan
from ..splatting._cpdi import AffineCPDISplatAssignment, CPDI2SplatAssignment
from ..splatting._gimp import UniformGIMPSplatAssignment
from ..splatting._reduction import _scatter_route_payload, cast_stage, certified_sum
from ..splatting._structured import (
    ParticleGridSplatPlan,
    ParticleGridSplatState,
    PreparedParticleGridSplat,
)
from ._axisymmetric_routes import (
    build_axisymmetric_routes,
    validate_axisymmetric_assignment,
)
from ._geometry import MPMGeometryPlan


if TYPE_CHECKING:
    from ...applications.solid_mechanics._loads import (
        AbstractMechanicalLoad,
        MechanicalLoadEvaluation,
        MechanicalLoadSemantics,
        MechanicalLoadState,
    )
    from ...integration import DeformedMeasurePlan, DeformedMeasureState


@final
class MPMBoundaryState(StrictModule):
    """Committed boundary coordinates and full constitutive deformation tensors."""

    position: Array
    deformation_gradient: Array


@final
class MPMBoundaryLoadResult(StrictModule):
    """Nodal force, physical exchange ledger, and atomic geometry candidate.

    Force and impulse use the flattened route source layout, including compact
    storage when the caller maps routes. Work uses the supplied nodal velocity.
    Canonical density, component identity, potential certification, and Nanson
    evidence remain available without translating the mechanical-load vocabulary.
    """

    force: Array
    impulse: Array
    work: Array
    dissipation: Array
    potential: Array | None
    candidate_state: MPMBoundaryState
    successful: Array
    evaluation: MechanicalLoadEvaluation
    measure: DeformedMeasureState

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.evaluation.component_ids

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self.evaluation.semantics


@final
class MPMDashpotResult(StrictModule):
    """Resisting nodal force and its work/dissipation at the supplied velocity."""

    force: Array
    impulse: Array
    work: Array
    dissipation: Array
    successful: Array


def _field_slot(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("field_slot must be an integer.")
    if value < 0:
        raise ValueError("field_slot must be nonnegative.")
    return value


def _surface_splat(
    splat: PreparedParticleGridSplat, slots: np.ndarray, /
) -> PreparedParticleGridSplat:
    """Restrict source-domain geometry once, preserving assignment parameters."""
    assignment = splat.plan.assignment
    if isinstance(assignment, UniformGIMPSplatAssignment):
        assignment = UniformGIMPSplatAssignment(
            np.asarray(assignment.reference_half_widths)[slots],
            maximum_half_width_cells=assignment.maximum_half_width_cells,
            evolving=assignment.evolving,
        )
    elif isinstance(assignment, AffineCPDISplatAssignment):
        assignment = AffineCPDISplatAssignment(
            np.asarray(assignment.reference_edges)[slots],
            maximum_extent_cells=assignment.maximum_extent_cells,
            maximum_condition=assignment.maximum_condition,
        )
    elif isinstance(assignment, CPDI2SplatAssignment):
        assignment = CPDI2SplatAssignment(
            np.asarray(assignment.reference_corner_offsets)[slots],
            maximum_extent_cells=assignment.maximum_extent_cells,
            maximum_condition=assignment.maximum_condition,
        )
    particles = ParticleSetPlan(
        np.asarray(splat.particles.particle_ids)[slots],
        np.asarray(splat.particles.masses)[slots],
        ambient_dimension=splat.particles.ambient_dimension,
        active_mask=np.asarray(splat.particles.active_mask)[slots],
        coordinate_dtype=splat.particles.plan.coordinate_dtype,
        name="mpm-boundary-particles",
    ).prepare()
    plan = ParticleGridSplatPlan(
        splat.plan.target,
        location=splat.plan.location,
        assignment=assignment,
        boundary=splat.plan.boundary,
        execution=splat.plan.execution,
        precision=splat.plan.precision,
        budget=splat.plan.budget,
    )
    return plan.prepare(particles)


@final
class PreparedMPMBoundaryLoad(StrictModule, NonTrainableState):
    """Canonical mechanical load at bounded MPM boundary quadrature.

    Reference area vectors contain complete physical measures; axisymmetric
    vectors are already ring-integrated. No hidden circumference multiplier is
    applied. Cartesian 2-D/3-D surfaces use native Nanson evaluation. Axisymmetric
    surfaces use full 3-D Nanson geometry and canonical 3-D load coordinates;
    only force pullback restricts to (r,z). A 1-D endpoint has unchanged point
    measure and orientation.

    ``step_size=0`` evaluates committed geometry. Positive steps evaluate trial
    backward-Euler boundary geometry with velocity supplied at the endpoint:
    x_trial=x+dt*v, F_trial=(I+dt*L)*F. The supplied routes freeze the assignment
    during this evaluation; the caller rebuilds them for the next accepted state.
    Reflected radial parity applies to both gathering and its force transpose.
    Failed load/geometry evaluations retain the committed candidate state and
    expose their failed evidence; they never repair or silently drop a load.
    """

    splat: PreparedParticleGridSplat
    reference_coordinates: Array
    reference_area_vectors: Array
    reference_measure: Array
    reference_normal: Array
    load: AbstractMechanicalLoad
    geometry: MPMGeometryPlan
    measure_plan: DeformedMeasurePlan | None
    default_load_state: MechanicalLoadState
    field_slot: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        splat: PreparedParticleGridSplat,
        reference_coordinates: ArrayLike,
        reference_area_vectors: ArrayLike,
        load: AbstractMechanicalLoad,
        /,
        *,
        geometry: MPMGeometryPlan | None = None,
        field_slot: int = 0,
    ) -> None:
        # Local import prevents the equations/MPM/solid-mechanics facade cycle.
        from ...applications.solid_mechanics._loads import (
            AbstractMechanicalLoad,
            MechanicalLoadState,
        )
        from ...integration import DeformedMeasurePlan

        if not isinstance(splat, PreparedParticleGridSplat):
            raise TypeError("splat must be PreparedParticleGridSplat.")
        if not isinstance(load, AbstractMechanicalLoad):
            raise TypeError("load must be AbstractMechanicalLoad.")
        if load.semantics.support != "boundary":
            raise ValueError("MPM boundary quadrature requires boundary load support.")
        slot = _field_slot(field_slot)
        dimension = splat.particles.ambient_dimension
        selected_geometry = (
            MPMGeometryPlan(spatial_dimension=dimension) if geometry is None else geometry
        )
        if not isinstance(selected_geometry, MPMGeometryPlan):
            raise TypeError("geometry must be MPMGeometryPlan.")
        if selected_geometry.spatial_dimension != dimension:
            raise ValueError("Boundary geometry and splat dimensions differ.")
        dtype = np.dtype(splat.plan.precision.geometry_dtype)
        coordinates = np.asarray(reference_coordinates)
        areas = np.asarray(reference_area_vectors)
        if np.iscomplexobj(coordinates) or np.iscomplexobj(areas):
            raise TypeError("Boundary coordinates and areas must be real.")
        coordinates = np.asarray(coordinates, dtype=dtype)
        areas = np.asarray(areas, dtype=dtype)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != dimension
            or not coordinates.shape[0]
        ):
            raise ValueError("Boundary coordinates must have shape (points, dimension).")
        if areas.shape != coordinates.shape:
            raise ValueError("Reference area vectors must match boundary coordinates.")
        measure = np.sqrt(np.sum(areas * areas, axis=-1))
        if (
            not np.all(np.isfinite(coordinates))
            or not np.all(np.isfinite(areas))
            or np.any(measure <= 0.0)
            or not np.all(np.isfinite(measure))
        ):
            raise ValueError("Boundary coordinates and nonzero areas must be finite.")
        if selected_geometry.kind == "axisymmetric" and np.any(coordinates[:, 0] <= 0.0):
            raise ValueError(
                "Axisymmetric boundary points require strictly positive radii."
            )
        particles = ParticleSetPlan(
            np.arange(coordinates.shape[0], dtype=np.int64),
            measure,
            ambient_dimension=dimension,
            coordinate_dtype=splat.particles.plan.coordinate_dtype,
            name="mpm-boundary-quadrature",
        ).prepare()
        boundary_splat = splat.plan.prepare(particles)
        normal = areas / measure[:, None]
        normal_for_plan = normal
        if selected_geometry.kind == "axisymmetric":
            validate_axisymmetric_assignment(boundary_splat)
            normal_for_plan = np.concatenate(
                (normal, np.zeros((normal.shape[0], 1), dtype=dtype)), axis=-1
            )
        measure_plan = (
            None
            if dimension == 1
            else DeformedMeasurePlan("surface", measure, reference_normal=normal_for_plan)
        )
        initial_coordinates = jnp.asarray(coordinates)
        initial_deformation = jnp.broadcast_to(
            jnp.eye(
                selected_geometry.constitutive_dimension, dtype=initial_coordinates.dtype
            ),
            (
                coordinates.shape[0],
                selected_geometry.constitutive_dimension,
                selected_geometry.constitutive_dimension,
            ),
        )
        assignment_input = boundary_splat.plan.assignment.update_input(
            initial_coordinates, initial_deformation, None
        )
        boundary_splat.plan.assignment.validate_input(
            assignment_input, coordinates.shape[0], dimension
        )
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-mpm-boundary-load",
                "splat": boundary_splat.prepared_id,
                "coordinates": array_tree_fingerprint(coordinates),
                "area_vectors": array_tree_fingerprint(areas),
                "load": load.load_id,
                "geometry": selected_geometry.plan_id,
                "field_slot": slot,
            }
        )
        self.splat = boundary_splat
        self.reference_coordinates = initial_coordinates
        self.reference_area_vectors = jnp.asarray(areas)
        self.reference_measure = jnp.asarray(measure)
        self.reference_normal = jnp.asarray(normal)
        self.load = load
        self.geometry = selected_geometry
        self.measure_plan = measure_plan
        self.default_load_state = MechanicalLoadState()
        self.field_slot = slot
        self.prepared_id = identifier

    @classmethod
    def from_surface_particles(
        cls,
        splat: PreparedParticleGridSplat,
        particle_slots: ArrayLike,
        reference_coordinates: ArrayLike,
        reference_area_vectors: ArrayLike,
        load: AbstractMechanicalLoad,
        /,
        **kwargs: Any,
    ) -> PreparedMPMBoundaryLoad:
        """Convert selected particle coordinates and source domains at preparation."""
        slots = np.asarray(particle_slots)
        coordinates = np.asarray(reference_coordinates)
        if (
            slots.ndim != 1
            or not slots.size
            or not np.issubdtype(slots.dtype, np.integer)
        ):
            raise ValueError("particle_slots must be a nonempty integer vector.")
        if (
            np.any(slots < 0)
            or np.any(slots >= splat.particles.capacity)
            or np.unique(slots).size != slots.size
        ):
            raise ValueError("particle_slots must contain distinct valid source slots.")
        if coordinates.shape != (
            splat.particles.capacity,
            splat.particles.ambient_dimension,
        ):
            raise ValueError("reference_coordinates must contain every source particle.")
        if not np.all(np.asarray(splat.particles.active_mask)[slots]):
            raise ValueError("Surface particles must be active.")
        selected = _surface_splat(splat, slots)
        return cls(selected, coordinates[slots], reference_area_vectors, load, **kwargs)

    def initialize_state(self) -> MPMBoundaryState:
        dimension = self.geometry.constitutive_dimension
        return MPMBoundaryState(
            self.reference_coordinates,
            jnp.broadcast_to(
                jnp.eye(dimension, dtype=self.reference_coordinates.dtype),
                (self.reference_coordinates.shape[0], dimension, dimension),
            ),
        )

    def _require_state(self, state: MPMBoundaryState, /) -> None:
        if state.position.shape != self.reference_coordinates.shape:
            raise ValueError("Boundary position shape differs from prepared quadrature.")
        dimension = self.geometry.constitutive_dimension
        if state.deformation_gradient.shape != (
            state.position.shape[0],
            dimension,
            dimension,
        ):
            raise ValueError("Boundary deformation shape differs from prepared geometry.")

    def build_routes(self, state: MPMBoundaryState, /) -> ParticleGridSplatState:
        """Build fixed-capacity routes with the original assignment and precision."""
        self._require_state(state)
        assignment_input = self.splat.plan.assignment.update_input(
            state.position, state.deformation_gradient, None
        )
        if self.geometry.kind == "axisymmetric":
            return build_axisymmetric_routes(self.splat, state.position, assignment_input)
        return self.splat.build(state.position, assignment_input=assignment_input)

    def _measure(self, deformation: Array, /) -> DeformedMeasureState:
        from ...integration import DeformedMeasureState

        if self.measure_plan is not None:
            return self.measure_plan.evaluate(deformation)
        jacobian = deformation[:, 0, 0]
        admissible = jnp.isfinite(jacobian) & (jacobian > 0.0)
        return DeformedMeasureState(
            reference_measure=self.reference_measure,
            current_measure=self.reference_measure,
            measure_ratio=jnp.ones_like(self.reference_measure),
            jacobian=jacobian,
            reference_normal=self.reference_normal,
            current_normal=self.reference_normal,
            admissible=admissible,
            valid=jnp.all(admissible),
            plan_id=self.prepared_id,
            kind="surface",
        )

    def evaluate(
        self,
        state: MPMBoundaryState,
        routes: ParticleGridSplatState,
        grid_velocity: Array,
        step_size: ArrayLike,
        load_state: MechanicalLoadState,
        args: Any = None,
        /,
    ) -> MPMBoundaryLoadResult:
        """Evaluate canonical density and its native route transpose, without repair."""
        self._require_state(state)
        if routes.prepared_id != self.splat.prepared_id:
            raise ValueError("Boundary routes belong to a different prepared splat.")
        dimension = self.geometry.spatial_dimension
        if grid_velocity.shape != (routes.stencil.source_size, dimension):
            raise ValueError(
                "grid_velocity must use flattened boundary route source layout."
            )
        dt = jnp.asarray(step_size, dtype=state.position.dtype)
        if dt.shape != ():
            raise ValueError("step_size must be scalar.")
        precision = self.splat.plan.precision
        velocity = cast_stage(grid_velocity, precision.evaluation_dtype)
        patches, valid = gather_patches(velocity, routes.stencil)
        patches = jnp.where(valid[..., None], patches, 0.0)
        if routes.radial_parity is not None:
            patches = patches.at[..., 0].multiply(routes.radial_parity)
        weights = jnp.where(valid, routes.stencil.weights, 0.0)
        point_velocity = ein.contract("pr,pri->pi", weights, patches)
        gradient = ein.contract(
            "pri,prj->pij",
            patches,
            jnp.where(valid[..., None], routes.weight_gradients, 0.0),
        )
        if self.geometry.kind == "axisymmetric":
            # Full hoop increment uses the gathered point velocity at current radius.
            full = jnp.zeros((gradient.shape[0], 3, 3), dtype=gradient.dtype)
            gradient = (
                full.at[:, :2, :2]
                .set(gradient)
                .at[:, 2, 2]
                .set(point_velocity[:, 0] / state.position[:, 0])
            )
        trial_position = jnp.where(
            dt == 0.0, state.position, state.position + dt * point_velocity
        )
        trial_deformation = jnp.where(
            dt == 0.0,
            state.deformation_gradient,
            (
                jnp.broadcast_to(
                    jnp.eye(self.geometry.constitutive_dimension, dtype=gradient.dtype),
                    gradient.shape,
                )
                + dt * gradient
            )
            @ state.deformation_gradient,
        )
        measure = self._measure(trial_deformation)
        committed_valid = jax.lax.cond(
            dt == 0.0,
            lambda: measure.valid,
            lambda: self._measure(state.deformation_gradient).valid,
        )
        reference = self.reference_coordinates
        current = trial_position
        if self.geometry.kind == "axisymmetric":
            zeros = jnp.zeros((current.shape[0], 1), dtype=current.dtype)
            reference = jnp.concatenate((reference, zeros), axis=-1)
            current = jnp.concatenate((current, zeros), axis=-1)
        evaluation = self.load.evaluate(reference, current, measure, load_state, args)
        point_force = (
            evaluation.total_force_density[:, :dimension]
            * measure.measure(evaluation.semantics.measure_frame)[:, None]
        )
        payload = (
            weights[..., None]
            * cast_stage(point_force, precision.evaluation_dtype)[:, None, :]
        )
        if routes.radial_parity is not None:
            payload = payload.at[..., 0].multiply(routes.radial_parity)
        force = _scatter_route_payload(
            routes.stencil,
            cast_stage(payload, precision.accumulation_dtype),
            self.splat.stable_source_order,
            routes.stencil.source_size,
            self.splat.plan.execution.accumulation,
        )
        force = cast_stage(force, precision.output_dtype)
        work = dt * certified_sum(
            jnp.sum(force * grid_velocity, axis=-1), precision.certification_dtype, axis=0
        )
        potential = (
            None
            if evaluation.potential_density is None
            else certified_sum(
                evaluation.potential_density
                * measure.measure(evaluation.semantics.measure_frame),
                precision.certification_dtype,
                axis=0,
            )
        )
        successful = (
            routes.successful
            & measure.valid
            & jnp.all(evaluation.valid)
            & committed_valid
            & jnp.all(jnp.isfinite(state.position))
            & jnp.isfinite(dt)
            & (dt >= 0.0)
            & jnp.all(jnp.isfinite(grid_velocity))
            & jnp.all(jnp.isfinite(trial_position))
            & jnp.all(jnp.isfinite(force))
            & jnp.isfinite(work)
        )
        if self.geometry.kind == "axisymmetric":
            successful = (
                successful
                & jnp.all(trial_position[:, 0] > 0.0)
                & jnp.all(state.position[:, 0] > 0.0)
                & jnp.all(evaluation.total_force_density[:, 2] == 0.0)
            )
        candidate = MPMBoundaryState(
            jnp.where(successful, trial_position, state.position),
            jnp.where(successful, trial_deformation, state.deformation_gradient),
        )
        return MPMBoundaryLoadResult(
            force,
            dt * force,
            work,
            jnp.zeros_like(work),
            potential,
            candidate,
            successful,
            evaluation,
            measure,
        )


@final
class LysmerKuhlemeyerDashpot(StrictModule, NonTrainableState):
    """Stateless viscous absorber with no spring or hidden time integration.

    Lysmer & Kuhlemeyer (1969), *Finite Dynamic Model for Infinite Media*,
    Journal of the Engineering Mechanics Division 95(4):859--877,
    https://doi.org/10.1061/JMCEA3.0001144. Original p.860 equations (1),(2)
    give normal/shear stresses a*rho*Vp*w_dot and b*rho*Vs*u_dot; p.866
    equations (34),(35) specialize to the standard viscous boundary a=b=1.
    In resisting-on-interior traction convention this is
    t=-rho*(cp*(v.n)*n+cs*(v-(v.n)*n)). Nodal area vectors contain complete
    physical areas (including axisymmetric rings). Zero area means inactive.

    Work is dt*sum(force*supplied_velocity), and dissipation=-work. The caller
    supplies midpoint velocity for an explicit passive midpoint update, or
    endpoint velocity for backward Euler. ``damping_matrix`` is the same linear
    operator C used by the caller's native nodal solve; evaluate returns -C*v.
    """

    nodal_area_vectors: Array
    density: Array
    compressional_speed: Array
    shear_speed: Array
    nodal_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    field_slot: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodal_area_vectors: ArrayLike,
        density: ArrayLike,
        compressional_speed: ArrayLike,
        shear_speed: ArrayLike,
        /,
        *,
        field_slot: int = 0,
    ) -> None:
        slot = _field_slot(field_slot)
        raw_areas = np.asarray(nodal_area_vectors)
        if np.iscomplexobj(raw_areas):
            raise TypeError("Nodal area vectors must be real.")
        areas = np.asarray(raw_areas, dtype=np.float64)
        if (
            areas.ndim < 2
            or areas.shape[-1] not in (1, 2, 3)
            or not all(areas.shape[:-1])
        ):
            raise ValueError("Nodal area vectors require shape (grid..., dimension).")
        if not np.all(np.isfinite(areas)):
            raise ValueError("Nodal area vectors must be finite.")
        coefficients = []
        for name, value in (
            ("density", density),
            ("compressional_speed", compressional_speed),
            ("shear_speed", shear_speed),
        ):
            raw = np.asarray(value)
            if np.iscomplexobj(raw):
                raise TypeError(f"{name} must be real.")
            coefficient = np.broadcast_to(
                np.asarray(raw, dtype=np.float64), areas.shape[:-1]
            )
            if not np.all(np.isfinite(coefficient)) or np.any(coefficient <= 0.0):
                raise ValueError(f"{name} must be finite and strictly positive.")
            coefficients.append(coefficient)
        rho, cp, cs = coefficients
        magnitude = np.sqrt(np.sum(areas * areas, axis=-1))
        if not np.all(np.isfinite(magnitude)):
            raise ValueError("Nodal areas must have finite magnitudes.")
        normal = areas / np.where(magnitude > 0.0, magnitude, 1.0)[..., None]
        damping = (magnitude * rho)[..., None, None] * (
            cs[..., None, None] * np.eye(areas.shape[-1], dtype=np.float64)
            + (cp - cs)[..., None, None] * normal[..., :, None] * normal[..., None, :]
        )
        if not np.all(np.isfinite(damping)):
            raise ValueError("Nodal impedance must be finite.")
        self.nodal_area_vectors = jnp.asarray(areas)
        self.density = jnp.asarray(rho)
        self.compressional_speed = jnp.asarray(cp)
        self.shear_speed = jnp.asarray(cs)
        self.nodal_shape = areas.shape[:-1]
        self.dimension = areas.shape[-1]
        self.field_slot = slot
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "lysmer-kuhlemeyer-dashpot",
                "areas": array_tree_fingerprint(areas),
                "coefficients": array_tree_fingerprint((rho, cp, cs)),
                "field_slot": slot,
            }
        )

    def damping_matrix(
        self,
        grid_velocity: Array,
        /,
        *,
        logical_node_ids: Array | None = None,
        node_valid: Array | None = None,
    ) -> Array:
        """Return C=A*rho*(cs*I+(cp-cs)*n*n.T) on dense or compact nodes."""
        if grid_velocity.ndim < 2 or grid_velocity.shape[-1] != self.dimension:
            raise ValueError("Dashpot velocity must have shape (nodes..., dimension).")
        shape = grid_velocity.shape[:-1]
        if node_valid is not None and node_valid.shape != shape:
            raise ValueError("node_valid must match nodal shape.")
        ids_valid = jnp.ones(shape, dtype=jnp.bool_)
        if logical_node_ids is None:
            if shape != self.nodal_shape and shape != (prod(self.nodal_shape),):
                raise ValueError("Dashpot velocity differs from prepared nodal layout.")
            areas = self.nodal_area_vectors.reshape(shape + (self.dimension,))
            rho = self.density.reshape(shape)
            cp = self.compressional_speed.reshape(shape)
            cs = self.shear_speed.reshape(shape)
        else:
            if logical_node_ids.shape != shape or not jnp.issubdtype(
                logical_node_ids.dtype, jnp.integer
            ):
                raise ValueError(
                    "logical_node_ids must be an integer array matching nodal shape."
                )
            ids = (
                logical_node_ids
                if node_valid is None
                else jnp.where(node_valid, logical_node_ids, 0)
            )
            ids_valid = (ids >= 0) & (ids < prod(self.nodal_shape))
            areas = jnp.take(
                self.nodal_area_vectors.reshape((-1, self.dimension)),
                ids,
                axis=0,
                mode="fill",
                fill_value=jnp.nan,
            )
            rho = jnp.take(self.density.reshape(-1), ids, mode="fill", fill_value=jnp.nan)
            cp = jnp.take(
                self.compressional_speed.reshape(-1), ids, mode="fill", fill_value=jnp.nan
            )
            cs = jnp.take(
                self.shear_speed.reshape(-1), ids, mode="fill", fill_value=jnp.nan
            )
        magnitude = jnp.sqrt(jnp.sum(areas * areas, axis=-1))
        normal = areas / jnp.where(magnitude > 0.0, magnitude, 1.0)[..., None]
        damping = (magnitude * rho)[..., None, None] * (
            cs[..., None, None]
            * jnp.broadcast_to(
                jnp.eye(self.dimension, dtype=areas.dtype),
                shape + (self.dimension, self.dimension),
            )
            + (cp - cs)[..., None, None] * normal[..., :, None] * normal[..., None, :]
        )
        admissible = (
            ids_valid
            & jnp.isfinite(magnitude)
            & jnp.isfinite(rho)
            & (rho > 0.0)
            & jnp.isfinite(cp)
            & (cp > 0.0)
            & jnp.isfinite(cs)
            & (cs > 0.0)
        )
        damping = jnp.where(admissible[..., None, None], damping, jnp.nan)
        if node_valid is not None:
            if node_valid.shape != shape:
                raise ValueError("node_valid must match nodal shape.")
            damping = jnp.where(node_valid[..., None, None], damping, 0.0)
        return damping.astype(grid_velocity.dtype)

    def evaluate(
        self,
        grid_velocity: Array,
        step_size: ArrayLike,
        /,
        *,
        logical_node_ids: Array | None = None,
        node_valid: Array | None = None,
    ) -> MPMDashpotResult:
        """Evaluate force and exchange, rejecting invalid time/active-node values."""
        dt = jnp.asarray(step_size, dtype=grid_velocity.dtype)
        if dt.shape != ():
            raise ValueError("step_size must be scalar.")
        damping = self.damping_matrix(
            grid_velocity, logical_node_ids=logical_node_ids, node_valid=node_valid
        )
        velocity = (
            grid_velocity
            if node_valid is None
            else jnp.where(node_valid[..., None], grid_velocity, 0.0)
        )
        force = -(damping @ velocity[..., None])[..., 0]
        work = dt * jnp.sum(force * velocity)
        successful = (
            jnp.isfinite(dt)
            & (dt >= 0.0)
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(force))
            & jnp.isfinite(work)
        )
        return MPMDashpotResult(force, dt * force, work, -work, successful)


__all__ = [
    "MPMBoundaryState",
    "MPMBoundaryLoadResult",
    "PreparedMPMBoundaryLoad",
    "MPMDashpotResult",
    "LysmerKuhlemeyerDashpot",
]
