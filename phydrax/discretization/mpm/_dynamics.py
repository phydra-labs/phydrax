#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...typing import checked
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from ..particle import ParticleDiscretization
from ..spatial import SparseBlockTopologyState
from ..splatting import (
    ParticleGridSplatState,
    PreparedParticleGridSplat,
    SplatDepositResult,
    SplatRouteScatterResult,
)
from ._axisymmetric_routes import (
    build_axisymmetric_routes,
    validate_axisymmetric_assignment,
)
from ._axisymmetric_volumetric import (
    build_axisymmetric_cell_average_routes,
    PreparedAxisymmetricCellAverage,
)
from ._boundary import PrescribedGridVelocityPlan, PrescribedGridVelocityResult
from ._contact import MPMGridConstraintResult, RigidMPMContactPlan
from ._domain import MPMParticleDomainPlan
from ._fields import MPMNodalFieldPlan
from ._geometry import MPMGeometryPlan
from ._lifecycle_amr import MPMLifecycleState
from ._loads import (
    LysmerKuhlemeyerDashpot,
    MPMBoundaryLoadResult,
    MPMBoundaryState,
    MPMDashpotResult,
    PreparedMPMBoundaryLoad,
)
from ._method import ExplicitMPMMethodPlan, MPMResourcePolicy
from ._multifield_dynamics import multifield_step_detailed
from ._phases import advance_grid_velocity, normalize_grid_momentum, update_deformation
from ._schedule import (
    AffineMUSLMPMSchedule,
    PostAdvectionMUSLMPMSchedule,
    USFMPMSchedule,
    USLMPMSchedule,
)
from ._storage import (
    AbstractMPMNodalStoragePlan,
    BlockSparseMPMNodalStoragePlan,
    DenseMPMNodalStoragePlan,
)
from ._transfer import (
    apic_particle_angular_momentum,
    apic_particle_kinetic_energy,
    build_apic_route_payload,
    gather_apic,
    grid_angular_momentum,
)
from ._types import (
    _MPMGridTrial,
    MPMDiagnostics,
    MPMEnergyLedger,
    MPMGridState,
    MPMLimitingProcess,
    MPMParticleState,
    MPMPredictorResult,
    MPMPreparationEvidence,
    MPMRejectionReason,
    MPMRunStatus,
    MPMRuntimeState,
    MPMScheduleEvidence,
    MPMStepRestriction,
    MPMStepResult,
    MPMTransferEvidence,
)
from ._velocity_transfer import apply_velocity_transfer
from ._volumetric import (
    build_cell_average_routes,
    prepare_cell_average,
    PreparedMPMCellAverage,
    stabilized_deformation,
)


ExternalMPMAcceleration = Callable[[Array, Array, Array, Any], ArrayLike]


def _relative_defect(left: Array, right: Array, /) -> Array:
    scale = jnp.maximum(1.0, jnp.maximum(jnp.linalg.norm(left), jnp.linalg.norm(right)))
    return jnp.linalg.norm(left - right) / scale


def _grid_kinetic(mass: Array, velocity: Array, active: Array, /) -> Array:
    terms = 0.5 * mass * jnp.sum(velocity * velocity, axis=-1)
    return compensated_sum(jnp.where(active, terms, 0.0).reshape((-1,)))


def _route_digest(state: ParticleGridSplatState) -> Array:
    slots = jnp.arange(state.stencil.indices.shape[1], dtype=jnp.int64)[None, :]
    values = jnp.where(
        state.stencil.valid,
        state.stencil.indices.astype(jnp.int64) + 1,
        0,
    )
    return jnp.sum(values * (slots + 17))


class PreparedMPMDynamics(StrictModule, NonTrainableState):
    """Prepared fixed-capacity explicit USL/APIC material-point dynamics."""

    particles: ParticleDiscretization
    splat: PreparedParticleGridSplat
    method: ExplicitMPMMethodPlan
    geometry: MPMGeometryPlan
    material: Any
    particle_domain: MPMParticleDomainPlan
    boundary: PrescribedGridVelocityPlan | None
    contact: RigidMPMContactPlan | None
    boundary_loads: tuple[PreparedMPMBoundaryLoad, ...]
    dashpots: tuple[LysmerKuhlemeyerDashpot, ...]
    nodal_fields: MPMNodalFieldPlan
    nodal_storage: AbstractMPMNodalStoragePlan
    external_acceleration: ExternalMPMAcceleration | None
    external_acceleration_id: str | None = eqx.field(static=True)
    resource_policy: MPMResourcePolicy
    key: DiscretizationKey
    grid_coordinates: Array
    cell_average: PreparedMPMCellAverage | PreparedAxisymmetricCellAverage | None
    nodal_shape: tuple[int, ...] = eqx.field(static=True)
    grid_count: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    minimum_spacing: float = eqx.field(static=True)
    preparation: PreparationReport
    resource_evidence: MPMPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        particles: ParticleDiscretization,
        splat: PreparedParticleGridSplat,
        method: ExplicitMPMMethodPlan,
        material: Any,
        particle_domain: MPMParticleDomainPlan,
        /,
        *,
        boundary: PrescribedGridVelocityPlan | None = None,
        contact: RigidMPMContactPlan | None = None,
        boundary_loads: tuple[PreparedMPMBoundaryLoad, ...] = (),
        dashpots: tuple[LysmerKuhlemeyerDashpot, ...] = (),
        nodal_fields: MPMNodalFieldPlan | None = None,
        nodal_storage: AbstractMPMNodalStoragePlan | None = None,
        external_acceleration: ExternalMPMAcceleration | None = None,
        external_acceleration_id: str | None = None,
        resource_policy: MPMResourcePolicy | None = None,
    ) -> None:
        """Bind one identity-sensitive prepared MPM execution plan.

        Admission remains centralized because storage, geometry, stabilization,
        loads, and material dimensions jointly determine PyTree field order,
        resource evidence, and ``prepared_id``. Splitting these checks across
        helpers would duplicate cross-contract conditions or obscure their
        deterministic ordering.
        """
        if boundary is not None and not isinstance(boundary, PrescribedGridVelocityPlan):
            raise TypeError("boundary must be PrescribedGridVelocityPlan or None.")
        if contact is not None and not isinstance(contact, RigidMPMContactPlan):
            raise TypeError("contact must be RigidMPMContactPlan or None.")
        fields = (
            MPMNodalFieldPlan(
                ("material",),
                np.zeros((particles.capacity,), dtype=np.int32),
            )
            if nodal_fields is None
            else nodal_fields
        )
        if not isinstance(fields, MPMNodalFieldPlan):
            raise TypeError("nodal_fields must be MPMNodalFieldPlan or None.")
        if fields.initial_particle_field_slots.shape != (particles.capacity,):
            raise ValueError("Nodal field slots must match particle capacity.")
        storage = (
            DenseMPMNodalStoragePlan(splat.target_shape)
            if nodal_storage is None
            else nodal_storage
        )
        if not isinstance(storage, AbstractMPMNodalStoragePlan):
            raise TypeError("nodal_storage must be AbstractMPMNodalStoragePlan or None.")
        if isinstance(storage, DenseMPMNodalStoragePlan):
            if storage.grid_shape != splat.target_shape:
                raise ValueError("Dense nodal storage must match the splat target.")
            if not splat.materialized_target:
                raise ValueError(
                    "A virtual splat target requires block-sparse nodal storage."
                )
        elif isinstance(storage, BlockSparseMPMNodalStoragePlan):
            if storage.grid_shape != splat.target_shape:
                raise ValueError("Sparse nodal storage must match the splat target.")
            if storage.topology_plan.layout.layout_id != splat.layout.layout_id:
                raise ValueError("Sparse nodal storage and splat layouts differ.")
        if splat.physical_particles().prepared_id != particles.prepared_id:
            raise ValueError("MPM splat was prepared for a different particle support.")
        dimension = particles.ambient_dimension
        if dimension not in (1, 2, 3) or particle_domain.dimension != dimension:
            raise ValueError("MPM dimensions must agree and be one, two, or three.")
        geometry = (
            MPMGeometryPlan(spatial_dimension=dimension)
            if method.geometry is None
            else method.geometry
        )
        if geometry.spatial_dimension != dimension:
            raise ValueError(
                "MPM geometry spatial dimension must match particle support."
            )
        if geometry.kind == "axisymmetric":
            if particle_domain.periodic[0]:
                raise ValueError("Axisymmetric MPM radial axis cannot be periodic.")
            if material.kinematics != "three_dimensional":
                raise ValueError(
                    "Axisymmetric MPM requires a three_dimensional material."
                )
            validate_axisymmetric_assignment(splat)
            if np.asarray(particle_domain.bounds)[0, 0] < 0.0:
                raise ValueError("Axisymmetric MPM domain requires r >= 0.")
        if tuple(splat.layout.axis_entities) != ("point",) * dimension:
            raise ValueError("Explicit MPM requires a nodal tensor-grid target.")
        assignment_capabilities = splat.plan.assignment.capabilities
        if (
            not assignment_capabilities.partition_of_unity
            or assignment_capabilities.maximum_explicit_derivative_order < 1
            or assignment_capabilities.polynomial_reproduction_order < 1
            or not assignment_capabilities.apic_compatible
        ):
            raise ValueError(
                "Explicit APIC MPM requires conservative first-order assignment "
                "weights, physical gradients, and particle moments."
            )
        if splat.plan.boundary != "reject":
            raise ValueError("Closed-domain MPM requires splat boundary='reject'.")
        if material.dimension != geometry.constitutive_dimension:
            raise ValueError(
                "MPM material dimension does not match constitutive geometry."
            )
        expected_kinematics = (
            ("three_dimensional",)
            if geometry.kind == "axisymmetric"
            else (
                ("one_dimensional",)
                if dimension == 1
                else (
                    ("plane_strain", "plane_stress")
                    if dimension == 2
                    else ("three_dimensional",)
                )
            )
        )
        if material.kinematics not in expected_kinematics:
            raise ValueError(
                "MPM material kinematics do not match the spatial dimension."
            )
        if external_acceleration is not None and not callable(external_acceleration):
            raise TypeError("external_acceleration must be callable or None.")
        if external_acceleration is None and external_acceleration_id is not None:
            raise ValueError("external_acceleration_id requires external acceleration.")
        if external_acceleration is not None and not external_acceleration_id:
            raise ValueError("External acceleration requires a stable non-empty ID.")

        for load in boundary_loads:
            if not isinstance(load, PreparedMPMBoundaryLoad):
                raise TypeError(
                    "boundary_loads must contain prepared MPM boundary loads."
                )
            if load.geometry.plan_id != geometry.plan_id:
                raise ValueError("Boundary load geometry must match MPM geometry.")
            if load.splat.layout.layout_id != splat.layout.layout_id:
                raise ValueError("Boundary load grid must match MPM grid.")
            if load.field_slot >= fields.field_count:
                raise ValueError("Boundary load field slot is outside the nodal fields.")
        for dashpot in dashpots:
            if not isinstance(dashpot, LysmerKuhlemeyerDashpot):
                raise TypeError("dashpots must contain LysmerKuhlemeyerDashpot.")
            if dashpot.nodal_area_vectors.shape != splat.target_shape + (dimension,):
                raise ValueError("Dashpot layout must match the MPM nodal grid.")
            if dashpot.field_slot >= fields.field_count:
                raise ValueError("Dashpot field slot is outside the nodal fields.")
        if geometry.kind == "axisymmetric" and boundary is not None:
            radial_axis = np.asarray(splat.layout.coordinates_by_axis[0]) == 0.0
            axis_mask = np.zeros(splat.target_shape + (dimension,), dtype=np.bool_)
            axis_mask[..., 0] = radial_axis[:, None]
            if np.any(
                axis_mask
                & np.asarray(boundary.mask)
                & (np.asarray(boundary.values) != 0.0)
            ):
                raise ValueError(
                    "Axisymmetric MPM axis nodes require prescribed v_r = 0."
                )
        axes = splat.plan.target.structured_axes
        coordinates = splat.layout.coordinates_by_axis
        spacings = []
        for axis_index, (axis, coordinate) in enumerate(
            zip(axes, coordinates, strict=True)
        ):
            values = np.asarray(coordinate, dtype=np.float64)
            spacing = float(np.diff(values)[0])
            spacings.append(spacing)
            if bool(axis.periodic) != particle_domain.periodic[axis_index]:
                raise ValueError("MPM domain and grid periodic axes must match.")
            required_margin = (
                assignment_capabilities.maximum_support_radius_cells * spacing
            )
            if not axis.periodic:
                grid_bounds = np.asarray(axis.bounds, dtype=np.float64)
                material_bounds = np.asarray(particle_domain.bounds)[:, axis_index]
                lower_gap = float(material_bounds[0] - grid_bounds[0])
                upper_gap = float(grid_bounds[1] - material_bounds[1])
                declared = particle_domain.support_margin[axis_index]
                axis_touching = (
                    geometry.kind == "axisymmetric"
                    and axis_index == 0
                    and grid_bounds[0] == 0.0
                    and material_bounds[0] == 0.0
                )
                minimum_gap = upper_gap if axis_touching else min(lower_gap, upper_gap)
                # Subtracting independently rounded bounds must admit an exact
                # declared halo; use a coordinate-scale roundoff allowance.
                epsilon = max(
                    np.finfo(np.asarray(coordinate).dtype).eps,
                    np.finfo(np.asarray(particle_domain.bounds).dtype).eps,
                )
                halo_roundoff = (
                    32.0
                    * epsilon
                    * max(
                        spacing,
                        declared,
                        float(np.max(np.abs(grid_bounds))),
                        float(np.max(np.abs(material_bounds))),
                    )
                )
                if (
                    declared + halo_roundoff < required_margin
                    or minimum_gap + halo_roundoff < declared
                ):
                    raise ValueError(
                        "Nonperiodic MPM needs the assignment's complete declared support halo."
                    )
        if isinstance(storage, DenseMPMNodalStoragePlan):
            mesh = jnp.meshgrid(*coordinates, indexing="ij")
            grid_coordinates = jnp.stack(mesh, axis=-1).reshape((-1, dimension))
            nodal_shape = splat.target_shape
        else:
            grid_coordinates = jnp.empty((0, dimension), dtype=coordinates[0].dtype)
            nodal_shape = (storage.storage_capacity,)
        if boundary is not None and boundary.mask.shape != splat.target_shape + (
            dimension,
        ):
            raise ValueError("Prescribed boundary layout must match the MPM nodal grid.")

        if boundary is not None and contact is not None:
            overlap_coordinates = grid_coordinates
            if isinstance(storage, BlockSparseMPMNodalStoragePlan):
                overlap_mesh = jnp.meshgrid(*coordinates, indexing="ij")
                overlap_coordinates = jnp.stack(overlap_mesh, axis=-1).reshape(
                    (-1, dimension)
                )
            overlap = np.asarray(contact.prospective_mask(overlap_coordinates)).reshape(
                splat.target_shape
            )
            if np.any(overlap[..., None] & np.asarray(boundary.mask)):
                raise ValueError(
                    "Prescribed velocity and rigid contact regions must be disjoint."
                )
        resource = MPMResourcePolicy() if resource_policy is None else resource_policy
        if not isinstance(resource, MPMResourcePolicy):
            raise TypeError("resource_policy must be MPMResourcePolicy or None.")
        itemsize = np.dtype(splat.plan.precision.evaluation_dtype).itemsize
        particle_count = particles.capacity
        grid_count = storage.storage_capacity
        route_payload_width = 3 * dimension
        cell_average = (
            (
                PreparedAxisymmetricCellAverage(splat)
                if geometry.kind == "axisymmetric"
                else prepare_cell_average(splat)
            )
            if method.volumetric_stabilization.kind == "b_bar"
            else None
        )
        route_count = (
            splat.route_count
            if cell_average is None
            else particle_count * cell_average.route_width
        )
        step_values = (
            route_count * route_payload_width
            + grid_count * (1 + 6 * dimension)
            + particle_count * (3 * dimension * dimension + 5 * dimension + 8)
        )
        state_values = particle_count * (
            3 * dimension * dimension
            + 2 * dimension
            + 3
            + int(np.prod(material.state_shape))
        )
        if method.volumetric_stabilization.kind == "b_bar":
            step_values += route_count * (3 * dimension + 3)
            if isinstance(cell_average, PreparedAxisymmetricCellAverage):
                step_values += (
                    splat.route_count * cell_average.quadrature_count * 16
                    + route_count * 4
                )
            elif cell_average is not None and not cell_average.axes:
                step_values += (
                    splat.route_count * (1 + 4**dimension) * (4 * dimension + 4)
                )
        elif method.volumetric_stabilization.kind == "f_bar":
            step_values += particle_count * (4 * dimension * dimension + 6)
            step_values += grid_count * 3 * fields.field_count
            state_values += particle_count * dimension * dimension
        if geometry.kind == "axisymmetric":
            tensor_extra = geometry.constitutive_dimension**2 - dimension**2
            state_values += particle_count * 2 * tensor_extra
            step_values += particle_count * 3 * tensor_extra + route_count
            if method.volumetric_stabilization.kind == "f_bar":
                state_values += particle_count * tensor_extra
                step_values += particle_count * 4 * tensor_extra
        for load in boundary_loads:
            boundary_count = load.reference_coordinates.shape[0]
            boundary_tensor = geometry.constitutive_dimension**2
            state_values += boundary_count * (dimension + boundary_tensor)
            step_values += boundary_count * (5 * dimension + 4 * boundary_tensor + 8)
            step_values += load.splat.route_count * (3 * dimension + 2)
            step_values += grid_count * 2 * dimension
        if dashpots:
            step_values += (
                grid_count * fields.field_count * (3 * dimension**2 + 4 * dimension)
            )
        workspace_bytes = step_values * itemsize
        state_bytes = state_values * itemsize
        resource.admit(
            step_workspace_bytes=workspace_bytes,
            state_bytes=state_bytes,
        )
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.FIELD_TRANSFER,
                DiscretizationCapability.MATRIX_FREE,
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
            ),
            diagnostics=(
                "explicit updated-Lagrangian USL",
                "quadratic nodal B-spline assignment",
                "matched APIC momentum and affine reconstruction",
                "first-Piola reference-volume internal force",
                "fixed material population and temporal topology",
            ),
            resource_counts={
                "particle_capacity": particle_count,
                "logical_grid_node_count": splat.target_size,
                "storage_node_count": grid_count,
                "route_count": route_count,
                "route_payload_width": route_payload_width,
                "step_workspace_bytes": workspace_bytes,
                "state_bytes": state_bytes,
            },
        )
        resource_evidence = MPMPreparationEvidence(
            particle_count,
            grid_count,
            route_count,
            route_payload_width,
            workspace_bytes,
            state_bytes,
            preparation.report_id,
        )
        key = DiscretizationKey(
            "mpm",
            DiscretizationRole.AUXILIARY,
            domain_labels=("material_point", "background_grid"),
        )
        self.particles = particles
        self.splat = splat
        self.cell_average = cell_average
        self.method = method
        self.material = material
        self.geometry = geometry
        self.particle_domain = particle_domain
        self.boundary = boundary
        self.contact = contact
        self.boundary_loads = tuple(boundary_loads)
        self.dashpots = tuple(dashpots)
        self.nodal_fields = fields
        self.nodal_storage = storage
        self.external_acceleration = external_acceleration
        self.external_acceleration_id = external_acceleration_id
        self.resource_policy = resource
        self.key = key
        self.grid_coordinates = grid_coordinates
        self.nodal_shape = nodal_shape
        self.grid_count = grid_count
        self.route_count = route_count
        self.minimum_spacing = min(spacings)
        self.preparation = preparation
        self.resource_evidence = resource_evidence
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mpm-dynamics",
                "key": key.key_id,
                "particles": particles.prepared_id,
                "splat": splat.prepared_id,
                "method": method.method_id,
                "material": material.plan_id,
                "particle_domain": particle_domain.plan_id,
                "boundary": None if boundary is None else boundary.plan_id,
                "contact": None if contact is None else contact.plan_id,
                "boundary_loads": tuple(load.prepared_id for load in boundary_loads),
                "dashpots": tuple(dashpot.prepared_id for dashpot in dashpots),
                "nodal_fields": fields.plan_id,
                "nodal_storage": storage.storage_id,
                "external_acceleration": external_acceleration_id,
                "resource": resource.policy_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def dimension(self) -> int:
        return self.particles.ambient_dimension

    @property
    def constitutive_dimension(self) -> int:
        return self.geometry.constitutive_dimension

    def _axis_mask(self, storage_state: SparseBlockTopologyState | None) -> Array:
        mask = jnp.zeros(self.nodal_shape + (self.dimension,), dtype=jnp.bool_)
        if self.geometry.kind == "axisymmetric":
            radial = self._storage_coordinates(storage_state)[:, 0].reshape(
                self.nodal_shape
            )
            mask = mask.at[..., 0].set(radial == 0.0)
        return mask

    def _axis_velocity(
        self, velocity: Array, storage_state: SparseBlockTopologyState | None
    ) -> Array:
        if self.geometry.kind == "axisymmetric":
            return jnp.where(self._axis_mask(storage_state), 0.0, velocity)
        return velocity

    def _axis_boundary(
        self,
        result: PrescribedGridVelocityResult,
        mass: Array,
        storage_state: SparseBlockTopologyState | None,
    ) -> PrescribedGridVelocityResult:
        if self.geometry.kind == "cartesian":
            return result
        velocity = self._axis_velocity(result.velocity, storage_state)
        impulse = compensated_sum(
            (mass[..., None] * (velocity - result.velocity)).reshape(
                (-1, self.dimension)
            ),
            axis=0,
        )
        work = _grid_kinetic(mass, velocity, mass > 0.0) - _grid_kinetic(
            mass, result.velocity, mass > 0.0
        )
        return PrescribedGridVelocityResult(
            velocity, result.impulse + impulse, result.work + work, result.successful
        )

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.splat.precision_evidence

    @property
    def resource_evidence_id(self) -> str:
        return self.resource_evidence.evidence_id

    @property
    def compact_storage(self) -> bool:
        return isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan)

    def _build_routes(
        self,
        position: Array,
        assignment_input: object = None,
        active_mask: Array | None = None,
    ) -> ParticleGridSplatState:
        if self.geometry.kind == "axisymmetric":
            routes = build_axisymmetric_routes(
                self.splat, position, assignment_input, active_mask
            )
            if self.method.volumetric_stabilization.kind == "b_bar":
                if not isinstance(self.cell_average, PreparedAxisymmetricCellAverage):
                    raise RuntimeError(
                        "Axisymmetric B-bar requires cylindrical cell averages."
                    )
                routes = build_axisymmetric_cell_average_routes(
                    self.splat, self.cell_average, routes, position
                )
            return routes
        routes = self.splat.build(
            position, assignment_input=assignment_input, active_mask=active_mask
        )
        if self.method.volumetric_stabilization.kind == "b_bar":
            if not isinstance(self.cell_average, PreparedMPMCellAverage):
                raise RuntimeError(
                    "Cartesian B-bar requires Cartesian cell-average factors."
                )
            routes = build_cell_average_routes(
                self.splat, self.cell_average, routes, position, assignment_input
            )
        return routes

    def _update_deformation(
        self,
        routes: ParticleGridSplatState,
        deformation_gradient: Array,
        velocity_gradient: Array,
        step_size: Array,
        mass: Array,
        active: Array,
        /,
        *,
        field_slots: Array | None = None,
        reference_volume: Array | None = None,
        physical_deformation_gradient: Array | None = None,
    ) -> Array:
        del mass
        return stabilized_deformation(
            self.method.volumetric_stabilization,
            self.splat,
            routes,
            deformation_gradient,
            velocity_gradient,
            step_size,
            reference_volume,
            physical_deformation_gradient,
            active,
            field_slots,
            self.nodal_fields.field_count,
        )

    @staticmethod
    def _update_physical_deformation(
        deformation_gradient: Array | None,
        velocity_gradient: Array,
        step_size: Array,
        /,
    ) -> Array | None:
        if deformation_gradient is None:
            return None
        return update_deformation(deformation_gradient, velocity_gradient, step_size)

    @staticmethod
    def _physical_deformation_admissible(
        deformation_gradient: Array | None,
        active: Array,
        /,
    ) -> Array:
        if deformation_gradient is None:
            return jnp.asarray(True)
        dimension = deformation_gradient.shape[-1]
        identity = jnp.broadcast_to(
            jnp.eye(dimension, dtype=deformation_gradient.dtype),
            deformation_gradient.shape,
        )
        determinant = solve_small_linear(
            SmallLinearSolvePlan(dimension), deformation_gradient, identity
        ).determinant
        return jnp.all(~active | (jnp.isfinite(determinant) & (determinant > 0.0)))

    def _build_storage(
        self,
        routes: ParticleGridSplatState,
        previous: SparseBlockTopologyState | None = None,
        boundary_states: tuple[MPMBoundaryState, ...] = (),
    ) -> SparseBlockTopologyState | None:
        if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
            if not self.boundary_loads:
                return self.nodal_storage.build(routes, previous)
            logical = [routes.stencil.indices.reshape((-1,))]
            valid = [routes.stencil.valid.reshape((-1,))]
            for load, boundary_state in zip(
                self.boundary_loads, boundary_states, strict=True
            ):
                boundary_routes = load.build_routes(boundary_state)
                logical.append(boundary_routes.stencil.indices.reshape((-1,)))
                valid.append(boundary_routes.stencil.valid.reshape((-1,)))
            logical_nodes = jnp.concatenate(tuple(logical))
            valid_nodes = jnp.concatenate(tuple(valid))
            topology = self.nodal_storage.topology_plan
            if previous is None:
                return topology.build(logical_nodes, valid_nodes)
            return topology.refresh(previous, logical_nodes, valid_nodes).candidate
        return None

    def _mapped_routes(
        self,
        routes: ParticleGridSplatState,
        storage_state: SparseBlockTopologyState | None,
    ) -> ParticleGridSplatState:
        if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
            if not (storage_state is not None):
                raise RuntimeError(
                    "Internal invariant failed: storage_state is not None."
                )
            return self.nodal_storage.mapped_routes(routes, storage_state)
        return routes

    def _deposit_content(
        self,
        routes: ParticleGridSplatState,
        storage_state: SparseBlockTopologyState | None,
        content: ArrayLike,
    ) -> SplatDepositResult:
        if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
            if not (storage_state is not None):
                raise RuntimeError(
                    "Internal invariant failed: storage_state is not None."
                )
            return self.nodal_storage.deposit_content(
                self.splat, routes, storage_state, content
            )
        return self.splat.deposit_content(routes, content)

    def _scatter_route_payload(
        self,
        routes: ParticleGridSplatState,
        storage_state: SparseBlockTopologyState | None,
        payload: ArrayLike,
    ) -> SplatRouteScatterResult:
        if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
            if not (storage_state is not None):
                raise RuntimeError(
                    "Internal invariant failed: storage_state is not None."
                )
            return self.nodal_storage.scatter_route_payload(
                self.splat, routes, storage_state, payload
            )
        return self.splat.scatter_route_payload(routes, payload)

    def _storage_coordinates(
        self, storage_state: SparseBlockTopologyState | None
    ) -> Array:
        if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
            if not (storage_state is not None):
                raise RuntimeError(
                    "Internal invariant failed: storage_state is not None."
                )
            return self.nodal_storage.coordinates(storage_state)
        return self.grid_coordinates

    def _storage_node_valid(
        self, storage_state: SparseBlockTopologyState | None
    ) -> Array:
        if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
            if not (storage_state is not None):
                raise RuntimeError(
                    "Internal invariant failed: storage_state is not None."
                )
            return storage_state.node_valid.reshape((-1,))
        return jnp.ones((self.grid_count,), dtype=jnp.bool_)

    def _storage_boundary_data(
        self, storage_state: SparseBlockTopologyState | None
    ) -> tuple[Array, Array] | None:
        if self.boundary is None:
            return None
        if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
            if not (storage_state is not None):
                raise RuntimeError(
                    "Internal invariant failed: storage_state is not None."
                )
            logical = storage_state.logical_node_ids.reshape((-1,))
            valid = storage_state.node_valid.reshape((-1,))
            mask = self.boundary.mask.reshape((-1, self.dimension))[logical]
            values = self.boundary.values.reshape((-1, self.dimension))[logical]
            return mask & valid[:, None], values
        return self.boundary.mask, self.boundary.values

    def _vector(self, name: str, value: ArrayLike) -> Array:
        array = self.splat.plan.precision.evaluation(value)
        expected = (self.particles.capacity, self.dimension)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        return array

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        reference_volume: ArrayLike,
        arguments: Any,
        /,
        *,
        deformation_gradient: ArrayLike | None = None,
        affine_velocity: ArrayLike | None = None,
        material_state: ArrayLike | None = None,
        assignment_input: object = None,
        material_slots: ArrayLike | None = None,
        body_ids: ArrayLike | None = None,
        velocity_field_slots: ArrayLike | None = None,
        lifecycle_state: MPMLifecycleState | None = None,
        time: ArrayLike = 0.0,
    ) -> MPMRuntimeState:
        position_ = self._vector("position", position)
        velocity_ = self._vector("velocity", velocity)
        count = self.particles.capacity
        dimension = self.dimension
        dtype = position_.dtype
        volume = jnp.asarray(reference_volume, dtype=dtype)
        if volume.shape != (count,):
            raise ValueError("reference_volume must have particle-capacity shape.")
        identity = jnp.broadcast_to(
            jnp.eye(self.constitutive_dimension, dtype=dtype),
            (count, self.constitutive_dimension, self.constitutive_dimension),
        )
        affine_identity = jnp.broadcast_to(
            jnp.eye(dimension, dtype=dtype), (count, dimension, dimension)
        )
        deformation = (
            identity
            if deformation_gradient is None
            else jnp.asarray(deformation_gradient, dtype=dtype)
        )
        affine = (
            jnp.zeros_like(affine_identity)
            if affine_velocity is None
            else jnp.asarray(affine_velocity, dtype=dtype)
        )
        if deformation.shape != identity.shape or affine.shape != affine_identity.shape:
            raise ValueError(
                "Initial F must have constitutive and C spatial tensor shape."
            )
        if not self.method.transfer.requires_affine_state:
            affine = jnp.zeros_like(affine)
        history_shape = (count,) + tuple(self.material.state_shape)
        if material_state is None:
            history = self.material.initialize_state((count,), dtype)
        else:
            history = jnp.asarray(material_state, dtype=dtype)
        if history.shape != history_shape:
            raise ValueError(f"material_state must have shape {history_shape}.")
        if lifecycle_state is not None:
            if not isinstance(lifecycle_state, MPMLifecycleState):
                raise TypeError("lifecycle_state must be MPMLifecycleState or None.")
            if lifecycle_state.active.shape != (count,):
                raise ValueError("Lifecycle state must match particle capacity.")
        active = (
            self.particles.active_mask
            if lifecycle_state is None
            else lifecycle_state.active
        )
        if self.geometry.kind == "axisymmetric":
            position_ = eqx.error_if(
                position_,
                jnp.any(
                    active & (~jnp.isfinite(position_[:, 0]) | (position_[:, 0] <= 0.0))
                ),
                "Axisymmetric MPM active particles require r > 0.",
            )
            deformation = eqx.error_if(
                deformation,
                jnp.any(
                    active
                    & (
                        jnp.any(deformation[:, :2, 2] != 0.0, axis=-1)
                        | jnp.any(deformation[:, 2, :2] != 0.0, axis=-1)
                        | (deformation[:, 2, 2] <= 0.0)
                    )
                ),
                "Axisymmetric MPM F requires zero meridional-hoop coupling and F_theta_theta > 0.",
            )
        runtime_masses = (
            self.particles.safe_masses.astype(dtype)
            if lifecycle_state is None
            else lifecycle_state.masses.astype(dtype)
        )
        density = runtime_masses / jnp.where(active, volume, 1.0)
        response = self.material.evaluate(
            deformation,
            history,
            density,
            arguments.material_parameters,
            jnp.asarray(time, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        runtime_assignment = self.splat.plan.assignment.update_input(
            position_, deformation, assignment_input
        )
        routes = self._build_routes(
            position_,
            assignment_input=runtime_assignment,
            active_mask=active,
        )
        boundary_states = tuple(load.initialize_state() for load in self.boundary_loads)
        storage_state = self._build_storage(routes, boundary_states=boundary_states)
        valid = (
            jnp.all((~active) | self.particle_domain.contains(position_))
            & routes.successful
            & (
                jnp.asarray(True)
                if storage_state is None
                else storage_state.evidence.successful
            )
            & jnp.all((~active) | (jnp.isfinite(volume) & (volume > 0.0)))
            & jnp.all((~active) | response.successful)
            & jnp.all((~active) | response.admissible)
        )
        if self.method.volumetric_stabilization.kind == "f_bar":
            valid = valid & self._physical_deformation_admissible(deformation, active)
        particles = MPMParticleState(
            position_,
            velocity_,
            deformation,
            affine,
            volume,
            response.first_piola,
            response.reference_energy_density,
            response.maximum_wave_speed,
            response.trial_state,
            deformation if self.method.volumetric_stabilization.kind == "f_bar" else None,
        )
        checked = eqx.error_if(
            particles.position,
            ~valid | ~tree_allfinite(particles),
            "Initial MPM state is inadmissible.",
        )
        particles = eqx.tree_at(lambda value: value.position, particles, checked)
        resolved_field_slots = (
            self.nodal_fields.initial_particle_field_slots
            if velocity_field_slots is None
            else velocity_field_slots
        )
        return MPMRuntimeState(
            particles,
            jnp.asarray(time, dtype=dtype).reshape(()),
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(int(MPMRunStatus.SUCCESS), dtype=jnp.int32),
            0,
            runtime_assignment,
            material_slots,
            body_ids,
            resolved_field_slots,
            storage_state,
            lifecycle_state,
            boundary_states,
        )

    def _external(
        self, time: Array, state: MPMParticleState, arguments: Any
    ) -> tuple[Array, Array]:
        if self.external_acceleration is None:
            return jnp.zeros_like(state.position), jnp.asarray(True)
        value = jnp.asarray(
            self.external_acceleration(
                time,
                state.position,
                state.velocity,
                arguments.external_arguments,
            ),
            dtype=state.position.dtype,
        )
        if value.shape != state.position.shape:
            raise ValueError("External MPM acceleration must match particle positions.")
        active = self.particles.active_mask[:, None]
        finite = jnp.all(jnp.where(active, jnp.isfinite(value), True))
        return jnp.where(active, value, 0.0), finite

    def _boundary_load_results(
        self,
        state: MPMRuntimeState,
        velocity: Array,
        step_size: Array,
        arguments: Any,
        storage_state: SparseBlockTopologyState | None,
        /,
        *,
        implicit: bool = False,
        field_slot: int = 0,
    ) -> tuple[tuple[MPMBoundaryLoadResult, ...], tuple[MPMDashpotResult, ...]]:
        if not self.boundary_loads and not self.dashpots:
            return (), ()
        states = arguments.mechanical_load_states
        if states and len(states) != len(self.boundary_loads):
            raise ValueError("Mechanical load states must match prepared boundary loads.")
        if len(state.boundary_states) != len(self.boundary_loads):
            raise ValueError(
                "Runtime boundary states must match prepared boundary loads."
            )
        results = []
        for index, (load, boundary_state) in enumerate(
            zip(self.boundary_loads, state.boundary_states, strict=True)
        ):
            if load.field_slot != field_slot:
                continue
            load_state = states[index] if states else load.default_load_state
            load_state = eqx.tree_at(
                lambda value: value.time,
                load_state,
                state.time + step_size if implicit else state.time,
            )
            routes = self._mapped_routes(load.build_routes(boundary_state), storage_state)
            result = load.evaluate(
                boundary_state,
                routes,
                velocity.reshape((self.grid_count, self.dimension)),
                step_size if implicit else jnp.zeros_like(step_size),
                load_state,
                arguments.boundary_arguments,
            )
            force = result.force.reshape(velocity.shape)
            result = eqx.tree_at(
                lambda value: (value.force, value.impulse, value.work),
                result,
                (
                    force,
                    step_size * force,
                    step_size * compensated_sum(force * velocity),
                ),
            )
            results.append(result)
        damping = tuple(
            dashpot.evaluate(
                velocity,
                step_size,
                logical_node_ids=(
                    None
                    if storage_state is None
                    else storage_state.logical_node_ids.reshape((-1,))
                ),
                node_valid=(
                    None
                    if storage_state is None
                    else storage_state.node_valid.reshape((-1,))
                ),
            )
            for dashpot in self.dashpots
            if dashpot.field_slot == field_slot
        )
        return tuple(results), damping

    def _damped_velocity(
        self,
        before: Array,
        forced: Array,
        mass: Array,
        active: Array,
        step_size: Array,
        storage_state: SparseBlockTopologyState | None,
        /,
        *,
        field_slot: int = 0,
    ) -> tuple[Array, Array]:
        selected = tuple(
            dashpot for dashpot in self.dashpots if dashpot.field_slot == field_slot
        )
        if not selected:
            return forced, jnp.asarray(True)
        matrix = jnp.zeros(
            mass.shape + (self.dimension, self.dimension), dtype=forced.dtype
        )
        for dashpot in selected:
            matrix = matrix + dashpot.damping_matrix(
                forced,
                logical_node_ids=(
                    None
                    if storage_state is None
                    else storage_state.logical_node_ids.reshape((-1,))
                ),
                node_valid=(
                    None
                    if storage_state is None
                    else storage_state.node_valid.reshape((-1,))
                ),
            )
        identity = jnp.broadcast_to(
            jnp.eye(self.dimension, dtype=forced.dtype), matrix.shape
        )
        system = mass[..., None, None] * identity + 0.5 * step_size * matrix
        rhs = (
            mass[..., None] * forced
            - 0.5 * step_size * (matrix @ before[..., None])[..., 0]
        )
        system = jnp.where(active[..., None, None], system, identity)
        rhs = jnp.where(active[..., None], rhs, 0.0)
        solved = solve_small_linear(SmallLinearSolvePlan(self.dimension), system, rhs)
        return solved.value, jnp.all(solved.successful)

    def _final_boundary_load_results(
        self,
        state: MPMRuntimeState,
        before: Array,
        after: Array,
        step_size: Array,
        arguments: Any,
        storage_state: SparseBlockTopologyState | None,
        initial: tuple[MPMBoundaryLoadResult, ...],
        /,
        *,
        field_slot: int = 0,
        force_velocity: Array | None = None,
    ) -> tuple[tuple[MPMBoundaryLoadResult, ...], tuple[MPMDashpotResult, ...]]:
        trial, _ = self._boundary_load_results(
            state,
            after,
            step_size,
            arguments,
            storage_state,
            implicit=True,
            field_slot=field_slot,
        )
        average = 0.5 * (before + (after if force_velocity is None else force_velocity))
        results = tuple(
            eqx.tree_at(
                lambda value: (value.candidate_state, value.successful, value.work),
                original,
                (
                    candidate.candidate_state,
                    original.successful & candidate.successful,
                    step_size * compensated_sum(original.force * average),
                ),
            )
            for original, candidate in zip(initial, trial, strict=True)
        )
        damping = tuple(
            dashpot.evaluate(
                average,
                step_size,
                logical_node_ids=(
                    None
                    if storage_state is None
                    else storage_state.logical_node_ids.reshape((-1,))
                ),
                node_valid=(
                    None
                    if storage_state is None
                    else storage_state.node_valid.reshape((-1,))
                ),
            )
            for dashpot in self.dashpots
            if dashpot.field_slot == field_slot
        )
        return results, damping

    def _apply_contact(
        self,
        velocity: Array,
        mass: Array,
        time: Array,
        step_size: Array,
        arguments: Any,
        storage_state: SparseBlockTopologyState | None,
    ) -> MPMGridConstraintResult:
        if self.contact is None:
            return MPMGridConstraintResult(
                velocity,
                jnp.zeros((self.dimension,), dtype=velocity.dtype),
                jnp.zeros((), dtype=velocity.dtype),
                jnp.zeros((), dtype=velocity.dtype),
                jnp.asarray(jnp.inf, dtype=velocity.dtype),
                jnp.zeros(mass.shape, dtype=jnp.bool_),
                jnp.zeros(mass.shape, dtype=jnp.int32),
                jnp.asarray(True),
            )
        coordinates = self._storage_coordinates(storage_state).reshape(
            self.nodal_shape + (self.dimension,)
        )
        return self.contact.apply(
            coordinates,
            velocity,
            mass,
            time,
            step_size,
            arguments.external_arguments,
        )

    def _empty_grid(self, dtype: DTypeLike) -> MPMGridState:
        scalar = jnp.zeros(
            (self.nodal_fields.field_count,) + self.nodal_shape,
            dtype=dtype,
        )
        vector = jnp.zeros(
            (self.nodal_fields.field_count,) + self.nodal_shape + (self.dimension,),
            dtype=dtype,
        )
        return MPMGridState(
            scalar, vector, vector, vector, vector, vector, scalar.astype("bool")
        )

    def _empty_diagnostics(
        self, state: MPMRuntimeState, route_state: ParticleGridSplatState, arguments: Any
    ) -> MPMDiagnostics:
        dtype = state.particles.position.dtype
        vector_shape = (self.dimension,) if self.dimension == 3 else ()
        zero_vector = jnp.zeros((self.dimension,), dtype=dtype)
        zero_angular = jnp.zeros(vector_shape, dtype=dtype)
        transfer = MPMTransferEvidence(
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.asarray(jnp.inf, dtype=dtype),
            zero_vector,
            zero_vector,
            jnp.asarray(jnp.inf, dtype=dtype),
            zero_angular,
            zero_angular,
            jnp.asarray(False),
            jnp.asarray(jnp.inf, dtype=dtype),
            zero_vector,
            jnp.max(jnp.abs(route_state.partition_sums - 1.0), initial=0.0),
            jnp.max(jnp.abs(route_state.gradient_sums), initial=0.0),
            jnp.max(jnp.abs(route_state.first_moments), initial=0.0),
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=dtype),
            jnp.asarray(True),
            route_state.valid_route_count,
            _route_digest(route_state),
            jnp.asarray(False),
        )
        schedule = MPMScheduleEvidence(
            jnp.asarray(self.method.schedule.schedule_code, dtype=jnp.int32),
            jnp.asarray(isinstance(self.method.schedule, USFMPMSchedule)),
            jnp.asarray(self.method.schedule.second_momentum_extrapolation),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=jnp.int64),
            jnp.asarray(False),
        )
        energy = MPMEnergyLedger(*(jnp.zeros((), dtype=dtype) for _ in range(13)))
        load_results = []
        dashpot_results = []
        if self.boundary_loads or self.dashpots:
            zero_velocity = jnp.zeros(self.nodal_shape + (self.dimension,), dtype=dtype)
            storage_state = self._build_storage(
                route_state, state.storage_state, state.boundary_states
            )
            for field_slot in range(self.nodal_fields.field_count):
                loads, damping = self._boundary_load_results(
                    state,
                    zero_velocity,
                    jnp.zeros((), dtype=dtype),
                    arguments,
                    storage_state,
                    field_slot=field_slot,
                )
                load_results.extend(loads)
                dashpot_results.extend(damping)
        return MPMDiagnostics(
            transfer,
            schedule,
            energy,
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(False),
            jnp.asarray(False),
            tuple(load_results),
            tuple(dashpot_results),
        )

    def _rejected(
        self,
        state: MPMRuntimeState,
        route_state: ParticleGridSplatState,
        arguments: Any,
        step_size: Array,
        reason: Any,
        status: Any,
        /,
        *,
        grid: MPMGridState | None = None,
        restriction: MPMStepRestriction | None = None,
    ) -> MPMStepResult:
        dtype = state.particles.position.dtype
        rejected = MPMRuntimeState(
            state.particles,
            state.time,
            state.accepted_step,
            jnp.asarray(status, dtype=jnp.int32),
            state.topology_generation,
            state.assignment_input,
            state.material_slots,
            state.body_ids,
            state.velocity_field_slots,
            state.storage_state,
            state.lifecycle_state,
            state.boundary_states,
        )
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        restriction_ = (
            MPMStepRestriction(
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                jnp.asarray(int(MPMLimitingProcess.NONE), dtype=jnp.int32),
                nan,
            )
            if restriction is None
            else restriction
        )
        return MPMStepResult(
            rejected,
            rejected,
            self._empty_grid(dtype) if grid is None else grid,
            restriction_,
            self._empty_diagnostics(state, route_state, arguments),
            jnp.asarray(False),
            jnp.asarray(reason, dtype=jnp.int32),
            step_size,
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(False),
            restriction_.suggested_step,
        )

    def _require_split_layout(self) -> None:
        if self.geometry.kind != "cartesian":
            raise ValueError("The MPM grid split requires Cartesian geometry.")
        if self.nodal_fields.field_count != 1:
            raise ValueError("The MPM grid split requires a single nodal field.")
        if not isinstance(self.nodal_storage, DenseMPMNodalStoragePlan):
            raise ValueError("The MPM grid split requires dense nodal storage.")
        if not isinstance(self.method.schedule, USLMPMSchedule):
            raise ValueError("The MPM grid split requires the USL schedule.")

    @checked
    def predict_grid(
        self,
        state: MPMRuntimeState,
        step_size: ArrayLike,
        arguments: Any,
        /,
        *,
        first_piola_increment: ArrayLike | None = None,
        nodal_force_increment: ArrayLike | None = None,
    ) -> MPMPredictorResult:
        """Freeze P2G and grid forces without advancing particles or material history.

        The split supports a single dense Cartesian USL field. Its velocity
        carries the leading field axis used by ``MPMGridState``. Stress and force
        increments affect this attempt only, not the committed skeleton stress.
        """
        self._require_split_layout()
        dt = jnp.asarray(step_size, dtype=state.particles.position.dtype).reshape(())
        active = (
            self.particles.active_mask
            if state.lifecycle_state is None
            else state.lifecycle_state.active
        )
        routes = self._build_routes(
            state.particles.position,
            assignment_input=state.assignment_input,
            active_mask=active,
        )
        storage_state = self._build_storage(
            routes, state.storage_state, state.boundary_states
        )
        domain_ok = jnp.all(
            (~active) | self.particle_domain.contains(state.particles.position)
        )
        finite = tree_allfinite(state.particles) & jnp.isfinite(dt) & (dt > 0.0)
        finite = finite & self._physical_deformation_admissible(
            state.particles.physical_deformation_gradient, active
        )
        route_ok = (
            routes.successful
            & ~jnp.any(routes.truncated_support_mask)
            & (
                jnp.asarray(True)
                if storage_state is None
                else storage_state.evidence.successful
            )
        )
        stress_increment = None
        if first_piola_increment is not None:
            stress_increment = jnp.asarray(
                first_piola_increment, dtype=state.particles.first_piola.dtype
            )
            if stress_increment.shape != state.particles.first_piola.shape:
                raise ValueError(
                    "First-Piola increment must match particle stress shape."
                )
            finite = finite & jnp.all(jnp.isfinite(stress_increment))
        force_increment = None
        if nodal_force_increment is not None:
            force_increment = jnp.asarray(
                nodal_force_increment, dtype=state.particles.position.dtype
            )
            if force_increment.shape == (1,) + self.nodal_shape + (self.dimension,):
                force_increment = force_increment[0]
            if force_increment.shape != self.nodal_shape + (self.dimension,):
                raise ValueError(
                    "Nodal force increment must match the grid velocity shape."
                )
            finite = finite & jnp.all(jnp.isfinite(force_increment))
        initial_valid = finite & domain_ok & route_ok
        trial = self._predict_grid_attempt(
            state,
            dt,
            arguments,
            routes,
            storage_state,
            active,
            first_piola_increment=stress_increment,
            nodal_force_increment=force_increment,
        )
        successful = initial_valid & trial.stable

        def constrain(
            _: None,
        ) -> tuple[MPMGridConstraintResult, PrescribedGridVelocityResult]:
            return self._constrain_grid_attempt(state, dt, arguments, trial)

        def unconstrained(
            _: None,
        ) -> tuple[MPMGridConstraintResult, PrescribedGridVelocityResult]:
            velocity = trial.grid.velocity_before[0]
            zero = jnp.zeros((), dtype=velocity.dtype)
            impulse = jnp.zeros((self.dimension,), dtype=velocity.dtype)
            contact = MPMGridConstraintResult(
                velocity,
                impulse,
                zero,
                zero,
                jnp.asarray(jnp.inf, dtype=velocity.dtype),
                jnp.zeros(trial.grid.mass[0].shape, dtype=jnp.bool_),
                jnp.zeros(trial.grid.mass[0].shape, dtype=jnp.int32),
                jnp.asarray(True),
            )
            return contact, PrescribedGridVelocityResult(
                velocity, impulse, zero, jnp.asarray(True)
            )

        contact_result, boundary_result = jax.lax.cond(
            successful, constrain, unconstrained, operand=None
        )
        successful = successful & contact_result.successful & boundary_result.successful
        grid = eqx.tree_at(
            lambda value: value.velocity_after,
            trial.grid,
            boundary_result.velocity[None, ...],
        )
        grid = tree_where(
            initial_valid, grid, self._empty_grid(state.particles.position.dtype)
        )
        reasons = jnp.where(
            ~initial_valid,
            jnp.where(
                ~finite,
                int(MPMRejectionReason.NONFINITE),
                jnp.where(
                    ~domain_ok,
                    int(MPMRejectionReason.DOMAIN),
                    int(MPMRejectionReason.ROUTE),
                ),
            ),
            jnp.where(
                trial.external_ok,
                int(MPMRejectionReason.STABILITY),
                int(MPMRejectionReason.NONFINITE),
            ),
        ).astype(jnp.int32)
        status = jnp.where(
            ~initial_valid,
            jnp.where(
                ~finite,
                int(MPMRunStatus.NONFINITE_STATE),
                jnp.where(
                    ~domain_ok,
                    int(MPMRunStatus.DOMAIN_REJECTED),
                    int(MPMRunStatus.ROUTE_REJECTED),
                ),
            ),
            jnp.where(
                trial.external_ok,
                int(MPMRunStatus.STABILITY_LIMIT_EXCEEDED),
                int(MPMRunStatus.NONFINITE_STATE),
            ),
        ).astype(jnp.int32)
        constraints_attempted = initial_valid & trial.stable
        reasons = jnp.where(constraints_attempted, 0, reasons)
        reasons = reasons | jnp.where(
            contact_result.successful, 0, int(MPMRejectionReason.CONTACT)
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            boundary_result.successful, 0, int(MPMRejectionReason.NONFINITE)
        ).astype(jnp.int32)
        status = jnp.where(
            constraints_attempted,
            jnp.where(
                ~contact_result.successful,
                int(MPMRunStatus.CONTACT_REJECTED),
                int(MPMRunStatus.NONFINITE_STATE),
            ),
            status,
        )
        return MPMPredictorResult(
            state,
            dt,
            arguments,
            routes,
            grid,
            trial,
            contact_result,
            boundary_result,
            successful,
            jnp.where(successful, int(MPMRejectionReason.NONE), reasons),
            jnp.where(successful, int(MPMRunStatus.SUCCESS), status),
            initial_valid,
            self.prepared_id,
        )

    @checked
    def finalize_grid_velocity(
        self,
        predictor: MPMPredictorResult,
        corrected_velocity: ArrayLike,
        /,
    ) -> MPMStepResult:
        """Finalize the frozen attempt once using its original time step and history.

        No routes or force predictor are rebuilt. The supplied velocity is the
        complete post-correction grid field, not an increment; it must already
        respect the prepared grid constraints.
        """
        self._require_split_layout()
        if predictor.prepared_id != self.prepared_id:
            raise ValueError("The predictor belongs to different prepared MPM dynamics.")
        velocity = jnp.asarray(corrected_velocity, dtype=predictor.grid.velocity.dtype)
        if velocity.shape == predictor.grid.velocity.shape:
            velocity = velocity[0]
        if velocity.shape != self.nodal_shape + (self.dimension,):
            raise ValueError("Corrected velocity must match the predictor grid shape.")
        boundary_result = eqx.tree_at(
            lambda value: value.successful,
            predictor.boundary_result,
            predictor.boundary_result.successful & jnp.all(jnp.isfinite(velocity)),
        )

        def advance(_: None) -> MPMStepResult:
            return self._finalize_grid_attempt(
                predictor.state,
                predictor.step_size,
                predictor.arguments,
                predictor.routes,
                predictor.trial,
                predictor.contact_result,
                boundary_result,
                velocity,
            )

        def reject(_: None) -> MPMStepResult:
            def initial_failure(_: None) -> MPMStepResult:
                return self._rejected(
                    predictor.state,
                    predictor.routes,
                    predictor.arguments,
                    predictor.step_size,
                    predictor.rejection_reasons,
                    predictor.status,
                )

            def grid_failure(_: None) -> MPMStepResult:
                return self._rejected(
                    predictor.state,
                    predictor.routes,
                    predictor.arguments,
                    predictor.step_size,
                    predictor.rejection_reasons,
                    predictor.status,
                    grid=predictor.trial.grid,
                    restriction=predictor.trial.restriction,
                )

            return jax.lax.cond(
                predictor.initial_valid, grid_failure, initial_failure, operand=None
            )

        return jax.lax.cond(predictor.successful, advance, reject, operand=None)

    def _predict_grid_attempt(
        self,
        state: MPMRuntimeState,
        dt: Array,
        arguments: Any,
        routes: ParticleGridSplatState,
        storage_state: SparseBlockTopologyState | None,
        active: Array,
        /,
        *,
        first_piola_increment: Array | None = None,
        nodal_force_increment: Array | None = None,
    ) -> _MPMGridTrial:
        execution_routes = self._mapped_routes(routes, storage_state)
        particle = state.particles
        mass = (
            self.particles.safe_masses.astype(particle.position.dtype)
            if state.lifecycle_state is None
            else state.lifecycle_state.masses.astype(particle.position.dtype)
        )
        acceleration_external, external_ok = self._external(
            state.time, particle, arguments
        )
        mass_result = self._deposit_content(routes, storage_state, mass)
        p2g_affine = (
            particle.affine_velocity
            if self.method.transfer.requires_affine_state
            else jnp.zeros_like(particle.affine_velocity)
        )
        route_payload = build_apic_route_payload(
            execution_routes,
            mass,
            particle.velocity,
            p2g_affine,
            particle.reference_volume,
            particle.first_piola
            if first_piola_increment is None
            else particle.first_piola + first_piola_increment,
            particle.deformation_gradient,
            acceleration_external,
            active,
            physical_deformation_gradient=particle.physical_deformation_gradient,
        )
        scattered = self._scatter_route_payload(routes, storage_state, route_payload)
        dimension = self.dimension
        grid_mass = mass_result.content
        grid_momentum = scattered.values[..., :dimension]
        internal_force = scattered.values[..., dimension : 2 * dimension]
        external_force = scattered.values[..., 2 * dimension :]
        increment_force = jnp.zeros_like(external_force)
        if first_piola_increment is not None:
            increment_payload = build_apic_route_payload(
                execution_routes,
                mass,
                jnp.zeros_like(particle.velocity),
                jnp.zeros_like(particle.affine_velocity),
                particle.reference_volume,
                first_piola_increment,
                particle.deformation_gradient,
                jnp.zeros_like(acceleration_external),
                active,
                physical_deformation_gradient=particle.physical_deformation_gradient,
            )
            increment_scatter = self._scatter_route_payload(
                routes, storage_state, increment_payload
            )
            increment_force = increment_scatter.values[..., dimension : 2 * dimension]
        if nodal_force_increment is not None:
            external_force = external_force + nodal_force_increment
            increment_force = increment_force + nodal_force_increment
        normalized_grid = normalize_grid_momentum(
            grid_mass,
            grid_momentum,
            mass_tolerance_factor=self.method.mass_tolerance_factor,
        )
        grid_active = normalized_grid.active
        velocity_before = normalized_grid.velocity
        node_valid = self._storage_node_valid(storage_state).reshape(self.nodal_shape)
        grid_active = grid_active & node_valid
        velocity_before = jnp.where(grid_active[..., None], velocity_before, 0.0)
        normalized_grid = eqx.tree_at(
            lambda value: (value.active, value.velocity),
            normalized_grid,
            (grid_active, velocity_before),
        )
        density = mass / jnp.where(active, particle.reference_volume, 1.0)
        identity = jnp.broadcast_to(
            jnp.eye(self.constitutive_dimension, dtype=particle.position.dtype),
            particle.deformation_gradient.shape,
        )
        schedule_p2g_success = jnp.asarray(True)
        scheduled_deformation = None
        scheduled_material = None
        first_gather = None
        if isinstance(self.method.schedule, USFMPMSchedule):
            first_gather = gather_apic(
                execution_routes,
                self._axis_velocity(velocity_before, storage_state).reshape(
                    (self.grid_count, dimension)
                ),
                active,
                self.method.transfer.maximum_condition,
            )
            scheduled_deformation = self._update_deformation(
                execution_routes,
                particle.deformation_gradient,
                first_gather.velocity_gradient,
                dt,
                mass,
                active,
                reference_volume=particle.reference_volume,
                physical_deformation_gradient=particle.physical_deformation_gradient,
            )
            scheduled_material = self.material.evaluate(
                scheduled_deformation,
                particle.material_state,
                density,
                arguments.material_parameters,
                state.time + dt,
                dt,
            )
            scheduled_payload = build_apic_route_payload(
                execution_routes,
                mass,
                particle.velocity,
                p2g_affine,
                particle.reference_volume,
                scheduled_material.first_piola,
                scheduled_deformation,
                acceleration_external,
                active,
                physical_deformation_gradient=self._update_physical_deformation(
                    particle.physical_deformation_gradient,
                    first_gather.velocity_gradient,
                    dt,
                ),
            )
            scheduled_scatter = self._scatter_route_payload(
                routes, storage_state, scheduled_payload
            )
            internal_force = scheduled_scatter.values[..., dimension : 2 * dimension]
            external_force = scheduled_scatter.values[..., 2 * dimension :]
            schedule_p2g_success = first_gather.successful & scheduled_scatter.successful
        load_results, _unused_dashpots = self._boundary_load_results(
            state, velocity_before, dt, arguments, storage_state
        )
        for result in load_results:
            external_force = external_force + result.force
        load_ok = jnp.asarray(True)
        for result in load_results:
            load_ok = (
                load_ok
                & result.successful
                & jnp.all(grid_active[..., None] | (result.force == 0.0))
            )
        grid_update = advance_grid_velocity(
            normalized_grid,
            internal_force,
            external_force,
            dt,
        )
        damped_velocity, damping_ok = self._damped_velocity(
            velocity_before,
            grid_update.velocity,
            grid_mass,
            grid_active,
            dt,
            storage_state,
        )
        wave_speed = (
            scheduled_material.maximum_wave_speed
            if scheduled_material is not None
            else particle.maximum_wave_speed
        )
        maximum_wave = jnp.max(jnp.where(active, wave_speed, 0.0), initial=0.0)
        maximum_velocity = jnp.max(
            jnp.where(
                active,
                jnp.sqrt(jnp.sum(particle.velocity * particle.velocity, axis=-1)),
                0.0,
            ),
            initial=0.0,
        )
        maximum_acceleration = grid_update.maximum_acceleration
        tiny = jnp.finfo(grid_mass.dtype).tiny
        acoustic = (
            self.method.acoustic_cfl
            * self.minimum_spacing
            / jnp.maximum(maximum_wave, tiny)
        )
        advective = jnp.where(
            maximum_velocity > 0.0,
            self.method.advective_cfl
            * self.minimum_spacing
            / jnp.maximum(maximum_velocity, tiny),
            jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
        )
        force_limit = jnp.where(
            maximum_acceleration > 0.0,
            self.method.force_cfl
            * jnp.sqrt(self.minimum_spacing / jnp.maximum(maximum_acceleration, tiny)),
            jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
        )
        infinite = jnp.asarray(jnp.inf, dtype=grid_mass.dtype)
        material_limit = (
            jnp.min(
                jnp.where(
                    active,
                    scheduled_material.suggested_step,
                    jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
                )
            )
            if scheduled_material is not None
            else infinite
        )
        limits = jnp.stack(
            (
                acoustic,
                advective,
                force_limit,
                infinite,
                material_limit,
                infinite,
                infinite,
            )
        )
        selected = jnp.min(limits)
        limiting_process = jnp.argmin(limits).astype(jnp.int32) + 1
        restriction = MPMStepRestriction(
            acoustic,
            advective,
            force_limit,
            infinite,
            material_limit,
            infinite,
            infinite,
            selected,
            limiting_process,
            selected,
        )
        stable = (
            external_ok & load_ok & damping_ok & jnp.isfinite(selected) & (dt <= selected)
        )
        grid_before = MPMGridState(
            grid_mass[None, ...],
            grid_momentum[None, ...],
            velocity_before[None, ...],
            internal_force[None, ...],
            external_force[None, ...],
            velocity_before[None, ...],
            grid_active[None, ...],
        )
        return _MPMGridTrial(
            active,
            storage_state,
            execution_routes,
            mass,
            acceleration_external,
            increment_force,
            mass_result,
            scattered,
            route_payload,
            density,
            identity,
            schedule_p2g_success,
            scheduled_deformation,
            scheduled_material,
            first_gather,
            load_results,
            damped_velocity,
            grid_before,
            restriction,
            stable,
            external_ok,
        )

    def _constrain_grid_attempt(
        self,
        state: MPMRuntimeState,
        dt: Array,
        arguments: Any,
        trial: _MPMGridTrial,
        /,
    ) -> tuple[MPMGridConstraintResult, PrescribedGridVelocityResult]:
        velocity_trial = trial.damped_velocity
        grid_mass = trial.grid.mass[0]
        storage_state = trial.storage_state
        dimension = self.dimension
        contact_result = self._apply_contact(
            velocity_trial,
            grid_mass,
            state.time,
            dt,
            arguments,
            storage_state,
        )
        if self.boundary is None:
            boundary_result = PrescribedGridVelocityResult(
                contact_result.velocity,
                jnp.zeros((dimension,), dtype=velocity_trial.dtype),
                jnp.zeros((), dtype=velocity_trial.dtype),
                jnp.asarray(True),
            )
        else:
            if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
                if not (storage_state is not None):
                    raise RuntimeError(
                        "Internal invariant failed: storage_state is not None."
                    )
                boundary_result = self.boundary.apply_indexed(
                    contact_result.velocity,
                    grid_mass,
                    dt,
                    storage_state.logical_node_ids.reshape((-1,)),
                    storage_state.node_valid.reshape((-1,)),
                )
            else:
                boundary_result = self.boundary.apply(
                    contact_result.velocity, grid_mass, dt
                )
        boundary_result = self._axis_boundary(boundary_result, grid_mass, storage_state)
        return contact_result, boundary_result

    def _finalize_grid_attempt(
        self,
        state: MPMRuntimeState,
        dt: Array,
        arguments: Any,
        routes: ParticleGridSplatState,
        trial: _MPMGridTrial,
        contact_result: MPMGridConstraintResult,
        boundary_result: PrescribedGridVelocityResult,
        grid_after: Array,
        /,
    ) -> MPMStepResult:
        particle = state.particles
        dimension = self.dimension
        active = trial.active
        storage_state = trial.storage_state
        execution_routes = trial.execution_routes
        mass = trial.mass
        acceleration_external = trial.acceleration_external
        mass_result = trial.mass_result
        scattered = trial.scattered
        route_payload = trial.route_payload
        density = trial.density
        identity = trial.identity
        schedule_p2g_success = trial.schedule_p2g_success
        scheduled_deformation = trial.scheduled_deformation
        scheduled_material = trial.scheduled_material
        first_gather = trial.first_gather
        load_results = trial.load_results
        damped_velocity = trial.damped_velocity
        restriction = trial.restriction
        grid_mass = trial.grid.mass[0]
        grid_momentum = trial.grid.momentum[0]
        velocity_before = trial.grid.velocity_before[0]
        internal_force = trial.grid.internal_force[0]
        external_force = trial.grid.external_force[0]
        grid_active = trial.grid.active[0]
        gathered = apply_velocity_transfer(
            self.method.transfer,
            self.method.advection,
            execution_routes,
            velocity_before,
            grid_after,
            particle.velocity,
            active,
        )
        candidate_position = particle.position + dt * gathered.advection_velocity
        second_mass_defect = jnp.asarray(0.0, dtype=grid_mass.dtype)
        second_momentum_defect = jnp.asarray(0.0, dtype=grid_mass.dtype)
        second_constraint_work = jnp.asarray(0.0, dtype=grid_mass.dtype)
        second_contact_work = jnp.asarray(0.0, dtype=grid_mass.dtype)
        second_contact_dissipation = jnp.asarray(0.0, dtype=grid_mass.dtype)
        second_contact_limit = jnp.asarray(jnp.inf, dtype=grid_mass.dtype)
        second_successful = jnp.asarray(True)
        kinematic_gradient = gathered.velocity_gradient
        kinematic_routes = execution_routes
        if self.method.schedule.second_momentum_extrapolation:
            second_routes = routes
            if isinstance(self.method.schedule, PostAdvectionMUSLMPMSchedule):
                second_input = self.splat.plan.assignment.update_input(
                    candidate_position,
                    particle.deformation_gradient
                    if particle.physical_deformation_gradient is None
                    else particle.physical_deformation_gradient,
                    state.assignment_input,
                )
                second_routes = self._build_routes(
                    candidate_position,
                    assignment_input=second_input,
                    active_mask=active,
                )
            second_execution_routes = self._mapped_routes(second_routes, storage_state)
            second_mass = self._deposit_content(second_routes, storage_state, mass)
            affine_second = (
                isinstance(
                    self.method.schedule,
                    (AffineMUSLMPMSchedule, PostAdvectionMUSLMPMSchedule),
                )
                and self.method.schedule.second_transfer_mode == "apic-affine-momentum"
            )
            if self.geometry.kind == "axisymmetric" or affine_second:
                zero_tensor = jnp.zeros_like(particle.first_piola)
                zero_acceleration = jnp.zeros_like(particle.velocity)
                second_payload = build_apic_route_payload(
                    second_execution_routes,
                    mass,
                    gathered.velocity,
                    gathered.affine_velocity
                    if affine_second
                    else jnp.zeros_like(gathered.affine_velocity),
                    particle.reference_volume,
                    zero_tensor,
                    particle.deformation_gradient,
                    zero_acceleration,
                    active,
                )
                second_momentum_result = self._scatter_route_payload(
                    second_routes, storage_state, second_payload
                )
                second_particle_momentum = mass[:, None] * gathered.velocity
                second_momentum_content = second_momentum_result.values[..., :dimension]
                second_momentum_successful = second_momentum_result.successful
            else:
                second_particle_momentum = mass[:, None] * gathered.velocity
                second_momentum_result = self._deposit_content(
                    second_routes,
                    storage_state,
                    second_particle_momentum,
                )
                second_momentum_content = second_momentum_result.content
                second_momentum_successful = second_momentum_result.successful
            second_grid = normalize_grid_momentum(
                second_mass.content,
                second_momentum_content,
                mass_tolerance_factor=self.method.mass_tolerance_factor,
            )
            second_contact = self._apply_contact(
                second_grid.velocity,
                second_mass.content,
                state.time,
                dt,
                arguments,
                storage_state,
            )
            if self.boundary is None:
                second_boundary = PrescribedGridVelocityResult(
                    second_contact.velocity,
                    jnp.zeros((dimension,), dtype=grid_mass.dtype),
                    jnp.zeros((), dtype=grid_mass.dtype),
                    jnp.asarray(True),
                )
            else:
                if isinstance(self.nodal_storage, BlockSparseMPMNodalStoragePlan):
                    if not (storage_state is not None):
                        raise RuntimeError(
                            "Internal invariant failed: storage_state is not None."
                        )
                    second_boundary = self.boundary.apply_indexed(
                        second_contact.velocity,
                        second_mass.content,
                        dt,
                        storage_state.logical_node_ids.reshape((-1,)),
                        storage_state.node_valid.reshape((-1,)),
                    )
                else:
                    second_boundary = self.boundary.apply(
                        second_contact.velocity, second_mass.content, dt
                    )
            second_boundary = self._axis_boundary(
                second_boundary, second_mass.content, storage_state
            )
            second_contact_work = second_contact.work
            second_contact_dissipation = second_contact.dissipation
            second_contact_limit = second_contact.contact_step_limit
            second_gather = gather_apic(
                second_execution_routes,
                second_boundary.velocity.reshape((self.grid_count, dimension)),
                active,
                (
                    self.method.transfer.maximum_condition
                    if np.isfinite(self.method.transfer.maximum_condition)
                    else 1.0e30
                ),
            )
            kinematic_gradient = second_gather.velocity_gradient
            kinematic_routes = second_execution_routes
            second_mass_defect = (
                second_mass.balance.maximum_absolute_balance_defect
                / jnp.maximum(
                    1.0,
                    jnp.abs(second_mass.balance.active_source_total),
                )
            )
            second_source_momentum = compensated_sum(
                jnp.where(active[:, None], second_particle_momentum, 0.0),
                axis=0,
            )
            if self.geometry.kind == "axisymmetric":
                second_source_momentum = compensated_sum(
                    second_payload[..., :dimension], axis=(0, 1)
                )
            second_target_momentum = compensated_sum(
                second_momentum_content.reshape((-1, dimension)), axis=0
            )
            second_momentum_defect = _relative_defect(
                second_source_momentum, second_target_momentum
            )
            second_constraint_work = second_boundary.work
            second_successful = (
                second_execution_routes.successful
                & second_mass.successful
                & second_momentum_successful
                & second_contact.successful
                & second_boundary.successful
                & second_gather.successful
            )
        if isinstance(self.method.schedule, USFMPMSchedule):
            if (
                scheduled_deformation is None
                or scheduled_material is None
                or first_gather is None
            ):
                raise RuntimeError("USF finalization requires its frozen material trial.")
            candidate_deformation = scheduled_deformation
            material = scheduled_material
            physical_gradient = first_gather.velocity_gradient
        else:
            physical_gradient = kinematic_gradient
            candidate_deformation = self._update_deformation(
                kinematic_routes,
                particle.deformation_gradient,
                kinematic_gradient,
                dt,
                mass,
                active,
                reference_volume=particle.reference_volume,
                physical_deformation_gradient=particle.physical_deformation_gradient,
            )
            material = self.material.evaluate(
                candidate_deformation,
                particle.material_state,
                density,
                arguments.material_parameters,
                state.time + dt,
                dt,
            )
        determinant = solve_small_linear(
            SmallLinearSolvePlan(self.constitutive_dimension),
            candidate_deformation,
            identity,
        ).determinant
        jacobian_ok = jnp.all(
            (~active) | (jnp.isfinite(determinant) & (determinant > 0.0))
        )
        if self.geometry.kind == "axisymmetric":
            jacobian_ok = jacobian_ok & jnp.all(
                ~active | (candidate_position[:, 0] > 0.0)
            )
        material_ok = jnp.all((~active) | (material.successful & material.admissible))
        candidate_material_limit = jnp.min(
            jnp.where(
                active,
                material.suggested_step,
                jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
            )
        )
        candidate_material_limit = jnp.where(
            jnp.isfinite(candidate_material_limit) & (candidate_material_limit > 0.0),
            candidate_material_limit,
            jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
        )
        candidate_contact_limit = jnp.minimum(
            contact_result.contact_step_limit, second_contact_limit
        )
        final_selected = jnp.minimum(
            restriction.selected,
            jnp.minimum(candidate_material_limit, candidate_contact_limit),
        )
        final_limiting = jnp.where(
            candidate_contact_limit
            < jnp.minimum(restriction.selected, candidate_material_limit),
            int(MPMLimitingProcess.CONTACT),
            jnp.where(
                candidate_material_limit < restriction.selected,
                int(MPMLimitingProcess.MATERIAL),
                restriction.limiting_process,
            ),
        ).astype(jnp.int32)
        final_restriction = MPMStepRestriction(
            restriction.acoustic,
            restriction.advective,
            restriction.force,
            candidate_contact_limit,
            candidate_material_limit,
            restriction.source_domain_motion,
            restriction.nonlinear,
            final_selected,
            final_limiting,
            final_selected,
        )
        material_step_ok = dt <= candidate_material_limit
        contact_step_ok = dt <= candidate_contact_limit
        candidate_particle = MPMParticleState(
            candidate_position,
            gathered.velocity,
            candidate_deformation,
            gathered.affine_velocity,
            particle.reference_volume,
            material.first_piola,
            material.reference_energy_density,
            material.maximum_wave_speed,
            material.trial_state,
            self._update_physical_deformation(
                particle.physical_deformation_gradient,
                physical_gradient,
                dt,
            ),
        )
        jacobian_ok = jacobian_ok & self._physical_deformation_admissible(
            candidate_particle.physical_deformation_gradient, active
        )
        finite_candidate = tree_allfinite(candidate_particle)
        final_loads, final_dashpots = self._final_boundary_load_results(
            state,
            velocity_before,
            grid_after,
            dt,
            arguments,
            storage_state,
            load_results,
            force_velocity=damped_velocity,
        )
        boundary_load_ok = jnp.asarray(True)
        for result in (*final_loads, *final_dashpots):
            boundary_load_ok = boundary_load_ok & result.successful
        successful = (
            gathered.successful
            & schedule_p2g_success
            & second_successful
            & contact_result.successful
            & boundary_result.successful
            & material_ok
            & material_step_ok
            & contact_step_ok
            & jacobian_ok
            & finite_candidate
            & boundary_load_ok
            & mass_result.successful
            & scattered.successful
        )
        reasons = jnp.zeros((), dtype=jnp.int32)
        reasons = reasons | jnp.where(
            gathered.successful & schedule_p2g_success & second_successful,
            0,
            int(MPMRejectionReason.APIC_MOMENT),
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            material_ok & material_step_ok,
            0,
            int(MPMRejectionReason.MATERIAL),
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            contact_result.successful & contact_step_ok,
            0,
            int(MPMRejectionReason.CONTACT),
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            jacobian_ok,
            0,
            int(MPMRejectionReason.JACOBIAN),
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            finite_candidate & boundary_result.successful,
            0,
            int(MPMRejectionReason.NONFINITE),
        ).astype(jnp.int32)
        status = jnp.where(
            successful,
            int(MPMRunStatus.SUCCESS),
            jnp.where(
                ~(gathered.successful & schedule_p2g_success & second_successful),
                int(MPMRunStatus.APIC_MOMENT_FAILED),
                jnp.where(
                    ~contact_result.successful | ~contact_step_ok,
                    int(MPMRunStatus.CONTACT_REJECTED),
                    jnp.where(
                        ~material_ok | ~material_step_ok | ~jacobian_ok,
                        int(MPMRunStatus.MATERIAL_REJECTED),
                        int(MPMRunStatus.NONFINITE_STATE),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        candidate_assignment = self.splat.plan.assignment.update_input(
            candidate_position,
            candidate_deformation
            if candidate_particle.physical_deformation_gradient is None
            else candidate_particle.physical_deformation_gradient,
            state.assignment_input,
        )
        accepted_assignment = tree_where(
            successful, candidate_assignment, state.assignment_input
        )
        accepted_storage = tree_where(successful, storage_state, state.storage_state)
        accepted_particle = tree_where(successful, candidate_particle, particle)
        candidate_generation = (
            state.topology_generation
            if storage_state is None
            else storage_state.generation
        )
        accepted_generation = jnp.where(
            successful,
            candidate_generation,
            state.topology_generation,
        )
        candidate_boundaries = tuple(result.candidate_state for result in final_loads)
        accepted_boundaries = tree_where(
            successful, candidate_boundaries, state.boundary_states
        )
        accepted_state = MPMRuntimeState(
            accepted_particle,
            jnp.where(successful, state.time + dt, state.time),
            jnp.where(successful, state.accepted_step + 1, state.accepted_step),
            status,
            accepted_generation,
            accepted_assignment,
            state.material_slots,
            state.body_ids,
            state.velocity_field_slots,
            accepted_storage,
            state.lifecycle_state,
            accepted_boundaries,
        )
        candidate_state = MPMRuntimeState(
            candidate_particle,
            state.time + dt,
            state.accepted_step + 1,
            status,
            candidate_generation,
            candidate_assignment,
            state.material_slots,
            state.body_ids,
            state.velocity_field_slots,
            storage_state,
            state.lifecycle_state,
            candidate_boundaries,
        )
        particle_mass = compensated_sum(jnp.where(active, mass, 0.0))
        target_mass = compensated_sum(grid_mass.reshape((-1,)))
        particle_momentum = compensated_sum(
            jnp.where(active[:, None], mass[:, None] * particle.velocity, 0.0),
            axis=0,
        )
        if self.geometry.kind == "axisymmetric":
            particle_momentum = compensated_sum(
                route_payload[..., :dimension], axis=(0, 1)
            )
        target_momentum = compensated_sum(grid_momentum.reshape((-1, dimension)), axis=0)
        particle_angular = apic_particle_angular_momentum(
            particle.position,
            particle.velocity,
            particle.affine_velocity,
            mass,
            routes,
            active,
        )
        target_angular = grid_angular_momentum(
            self._storage_coordinates(storage_state),
            grid_momentum.reshape((-1, dimension)),
            grid_active.reshape((-1,)),
        )
        angular_valid = jnp.asarray(
            self.geometry.kind == "cartesian"
            and self.dimension > 1
            and not any(self.particle_domain.periodic)
        )
        angular_defect = jnp.where(
            angular_valid,
            _relative_defect(particle_angular, target_angular),
            0.0,
        )
        net_internal = compensated_sum(internal_force.reshape((-1, dimension)), axis=0)
        maximum_condition = jnp.max(
            jnp.where(active, gathered.condition_estimate, 0.0), initial=0.0
        )
        transfer_success = (
            mass_result.balance.closed_domain_conservation_valid
            & gathered.successful
            & (_relative_defect(particle_momentum, target_momentum) <= 1.0e-10)
            & ((~angular_valid) | (angular_defect <= 1.0e-10))
        )
        transfer = MPMTransferEvidence(
            particle_mass,
            target_mass,
            jnp.abs(particle_mass - target_mass)
            / jnp.maximum(1.0, jnp.abs(particle_mass)),
            particle_momentum,
            target_momentum,
            _relative_defect(particle_momentum, target_momentum),
            particle_angular,
            target_angular,
            angular_valid,
            angular_defect,
            net_internal,
            mass_result.balance.maximum_partition_defect,
            jnp.max(jnp.abs(routes.gradient_sums), initial=0.0),
            jnp.max(jnp.abs(routes.first_moments), initial=0.0),
            maximum_condition,
            jnp.sum(grid_active, dtype=jnp.int32),
            jnp.zeros((), dtype=grid_mass.dtype),
            jnp.asarray(True),
            routes.valid_route_count,
            _route_digest(routes),
            transfer_success,
        )
        particle_kinetic_before = apic_particle_kinetic_energy(
            mass,
            particle.velocity,
            particle.affine_velocity,
            routes.second_moments,
            active,
            first_moment=routes.first_moments
            if self.geometry.kind == "axisymmetric"
            else None,
        )
        particle_kinetic_after = apic_particle_kinetic_energy(
            mass,
            candidate_particle.velocity,
            candidate_particle.affine_velocity,
            routes.second_moments,
            active,
            first_moment=routes.first_moments
            if self.geometry.kind == "axisymmetric"
            else None,
        )
        grid_kinetic_before = _grid_kinetic(grid_mass, velocity_before, grid_active)
        grid_kinetic_after = _grid_kinetic(grid_mass, grid_after, grid_active)
        material_before = compensated_sum(
            jnp.where(
                active,
                particle.reference_volume * particle.reference_energy_density,
                0.0,
            )
        )
        material_after = compensated_sum(
            jnp.where(
                active,
                particle.reference_volume * material.reference_energy_density,
                0.0,
            )
        )
        average_velocity = 0.5 * (particle.velocity + candidate_particle.velocity)
        external_work = dt * compensated_sum(
            jnp.where(
                active,
                mass * jnp.sum(acceleration_external * average_velocity, axis=-1),
                0.0,
            )
        )
        external_work = external_work + dt * compensated_sum(
            jnp.where(
                grid_active,
                jnp.sum(
                    trial.increment_force * 0.5 * (velocity_before + grid_after),
                    axis=-1,
                ),
                0.0,
            )
        )
        total_before = particle_kinetic_before + material_before
        total_after = particle_kinetic_after + material_after
        boundary_work = boundary_result.work + second_constraint_work
        for result in (*final_loads, *final_dashpots):
            boundary_work = boundary_work + result.work
        contact_work = contact_result.work + second_contact_work
        contact_dissipation = contact_result.dissipation + second_contact_dissipation
        plastic_dissipation = compensated_sum(
            jnp.where(
                active,
                particle.reference_volume * material.dissipation_increment,
                0.0,
            )
        )
        balance_defect = (
            total_after
            + plastic_dissipation
            - total_before
            - external_work
            - boundary_work
            - contact_work
        )
        energy = MPMEnergyLedger(
            particle_kinetic_before,
            grid_kinetic_before,
            grid_kinetic_after,
            particle_kinetic_after,
            material_before,
            material_after,
            external_work,
            boundary_work,
            contact_work,
            contact_dissipation,
            plastic_dissipation,
            jnp.zeros((), dtype=grid_mass.dtype),
            balance_defect,
        )
        schedule = MPMScheduleEvidence(
            jnp.asarray(self.method.schedule.schedule_code, dtype=jnp.int32),
            jnp.asarray(isinstance(self.method.schedule, USFMPMSchedule)),
            jnp.asarray(self.method.schedule.second_momentum_extrapolation),
            second_mass_defect,
            second_momentum_defect,
            second_constraint_work,
            _route_digest(routes)
            + jnp.asarray(self.method.schedule.schedule_code * 104729, dtype=jnp.int64),
            schedule_p2g_success & second_successful,
        )
        diagnostics = MPMDiagnostics(
            transfer,
            schedule,
            energy,
            jnp.min(jnp.where(active, determinant, jnp.inf)),
            jnp.max(jnp.where(active, determinant, 0.0)),
            material_ok,
            finite_candidate,
            final_loads,
            final_dashpots,
        )
        total_external_force = external_force
        for result in final_dashpots:
            total_external_force = total_external_force + result.force
        grid = MPMGridState(
            grid_mass[None, ...],
            grid_momentum[None, ...],
            velocity_before[None, ...],
            internal_force[None, ...],
            total_external_force[None, ...],
            grid_after[None, ...],
            grid_active[None, ...],
        )
        return MPMStepResult(
            candidate_state,
            accepted_state,
            grid,
            final_restriction,
            diagnostics,
            successful,
            reasons,
            dt,
            final_selected - dt,
            ~material_ok | ~material_step_ok,
            jnp.where(
                material_ok & material_step_ok,
                final_selected,
                jnp.minimum(final_selected, 0.5 * dt),
            ),
        )

    @checked
    def step_detailed(
        self,
        state: MPMRuntimeState,
        step_size: ArrayLike,
        arguments: Any,
        /,
    ) -> MPMStepResult:
        dt = jnp.asarray(step_size, dtype=state.particles.position.dtype).reshape(())
        active = (
            self.particles.active_mask
            if state.lifecycle_state is None
            else state.lifecycle_state.active
        )
        routes = self._build_routes(
            state.particles.position,
            assignment_input=state.assignment_input,
            active_mask=active,
        )
        storage_state = self._build_storage(
            routes, state.storage_state, state.boundary_states
        )
        domain_ok = jnp.all(
            (~active) | self.particle_domain.contains(state.particles.position)
        )
        finite = tree_allfinite(state.particles) & jnp.isfinite(dt) & (dt > 0.0)
        finite = finite & self._physical_deformation_admissible(
            state.particles.physical_deformation_gradient, active
        )
        route_ok = (
            routes.successful
            & ~jnp.any(routes.truncated_support_mask)
            & (
                jnp.asarray(True)
                if storage_state is None
                else storage_state.evidence.successful
            )
        )

        def invalid(_: ParticleGridSplatState) -> MPMStepResult:
            reason = jnp.where(
                ~finite,
                int(MPMRejectionReason.NONFINITE),
                jnp.where(
                    ~domain_ok,
                    int(MPMRejectionReason.DOMAIN),
                    int(MPMRejectionReason.ROUTE),
                ),
            )
            status = jnp.where(
                ~finite,
                int(MPMRunStatus.NONFINITE_STATE),
                jnp.where(
                    ~domain_ok,
                    int(MPMRunStatus.DOMAIN_REJECTED),
                    int(MPMRunStatus.ROUTE_REJECTED),
                ),
            )
            return self._rejected(
                state,
                routes,
                arguments,
                dt,
                reason,
                status,
            )

        def execute(routes: ParticleGridSplatState) -> MPMStepResult:
            if self.nodal_fields.field_count > 1:
                return multifield_step_detailed(self, state, dt, arguments, routes)
            trial = self._predict_grid_attempt(
                state, dt, arguments, routes, storage_state, active
            )

            def unstable(_: None) -> MPMStepResult:
                return self._rejected(
                    state,
                    routes,
                    arguments,
                    dt,
                    jnp.where(
                        trial.external_ok,
                        int(MPMRejectionReason.STABILITY),
                        int(MPMRejectionReason.NONFINITE),
                    ),
                    jnp.where(
                        trial.external_ok,
                        int(MPMRunStatus.STABILITY_LIMIT_EXCEEDED),
                        int(MPMRunStatus.NONFINITE_STATE),
                    ),
                    grid=trial.grid,
                    restriction=trial.restriction,
                )

            def advance(_: None) -> MPMStepResult:
                contact_result, boundary_result = self._constrain_grid_attempt(
                    state, dt, arguments, trial
                )
                return self._finalize_grid_attempt(
                    state,
                    dt,
                    arguments,
                    routes,
                    trial,
                    contact_result,
                    boundary_result,
                    boundary_result.velocity,
                )

            return jax.lax.cond(trial.stable, advance, unstable, operand=None)

        return jax.lax.cond(
            finite & domain_ok & route_ok, execute, invalid, operand=routes
        )


__all__ = ["ExternalMPMAcceleration", "PreparedMPMDynamics"]
