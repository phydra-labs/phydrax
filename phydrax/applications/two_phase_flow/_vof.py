#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._sharp_measures import QualifiedSharpGeometry
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import (
    FiniteVolumeDiscretization,
    HeightFunctionCurvaturePlan,
    HeightFunctionCurvatureResult,
    LinearSurfaceTensionLaw,
    MACBalancedCapillaryOperator,
    MACBoundaryPlan,
    MACBoundarySide,
    MACMomentumPlan,
    MACOperatorPlan,
    PreparedMACBoundaryPlan,
    PreparedMACMomentumOperators,
    PreparedMACOperators,
    StructuredPLICPlan,
    StructuredPLICReconstruction,
    SurfaceTensionPolicy,
    VariableSurfaceTensionPolicy,
)
from ...solver import MACVariableDensityProjectionPlan, MACVariationalViscosityPlan
from ...solver._mac_sharp_interface import MACSharpInterfaceProjectionPlan
from ...typing import checked


FaceTuple = tuple[Array, ...]

_WALL_KINDS = ("no-slip", "free-slip")


class PLICGeometry(StrictModule):
    """Exact PLIC interface of the liquid volume fraction.

    ``normal`` points out of the liquid.  In each cell's scaled coordinates
    ``xi`` in ``[0, 1]^D`` the liquid occupies ``{(normal * h) . xi <=
    plane_offset}``.  ``facet_measure`` is the exact facet length/area and
    ``interface_point`` its exact centroid.
    """

    normal: Array
    plane_offset: Array
    mixed_cell: Array
    facet_measure: Array
    interface_point: Array
    reconstruction_residual: Array
    finite: Array
    valid: Array
    solid_interface_conflict: Array


class TwoPhaseTopologyEvidence(StrictModule):
    liquid_volume: Array
    gas_volume: Array
    mixed_cell_count: Array
    interface_measure: Array
    component_proxy: Array
    changed_cell_mask: Array
    finite: Array
    valid: Array


class TwoPhaseVOFState(StrictModule):
    liquid_content: Array
    momentum: FaceTuple
    phase_scalar_content: dict[str, Array]
    material_scalar_content: dict[str, Array]
    level_set: Array
    geometry_epoch: Array
    geometry_id: str = eqx.field(static=True)


class TwoPhaseVOFView(StrictModule):
    alpha: Array
    density: Array
    viscosity: Array
    velocity: FaceTuple
    pressure: Array
    absolute_pressure: Array
    plic: PLICGeometry
    topology: TwoPhaseTopologyEvidence
    view_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


class TwoPhaseInterfaceGeometry(StrictModule):
    """PLIC, height-function curvature and interface positions of one alpha."""

    reconstruction: StructuredPLICReconstruction
    plic: PLICGeometry
    curvature: HeightFunctionCurvatureResult | None


class TwoPhaseMaterialPlan(StrictModule, NonTrainableState):
    liquid_density: float = eqx.field(static=True)
    gas_density: float = eqx.field(static=True)
    liquid_viscosity: float = eqx.field(static=True)
    gas_viscosity: float = eqx.field(static=True)
    surface_tension: float = eqx.field(static=True)
    contact_angle: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        liquid_density: float = 1000.0,
        gas_density: float = 1.2,
        liquid_viscosity: float = 1.0e-3,
        gas_viscosity: float = 1.8e-5,
        surface_tension: float = 0.0,
        contact_angle: float = 0.5 * np.pi,
    ) -> None:
        values = tuple(
            float(v)
            for v in (
                liquid_density,
                gas_density,
                liquid_viscosity,
                gas_viscosity,
                surface_tension,
                contact_angle,
            )
        )
        if (
            any(not np.isfinite(v) for v in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] < 0.0
            or not 0.0 < values[5] < np.pi
        ):
            raise ValueError("Invalid two-phase material parameters.")
        (
            self.liquid_density,
            self.gas_density,
            self.liquid_viscosity,
            self.gas_viscosity,
            self.surface_tension,
            self.contact_angle,
        ) = values
        self.material_id = canonical_fingerprint(
            {"kind": "two-phase-material-plan", "values": list(values)}
        )


def _validated_point(
    value: tuple[float, ...] | None, dimension: int, name: str, /
) -> tuple[float, ...] | None:
    if value is None:
        return None
    point = tuple(float(entry) for entry in value)
    if len(point) != dimension or not all(np.isfinite(entry) for entry in point):
        raise ValueError(f"{name} must be a finite vector of the grid dimension.")
    return point


class IncompressibleTwoPhaseVOFPlan(StrictModule):
    """Compile fixed-grid conservative incompressible two-phase VOF flow.

    Nonperiodic axes are impermeable walls; ``wall_sides`` selects their
    tangential closure (``no-slip`` or ``free-slip``, optionally with a
    prescribed wall velocity) and defaults to no-slip walls.  ``gravity`` is
    folded into the balanced interfacial potential (reduced gravity, see the
    step method) and requires ``hydrostatic_reference``; ``reference_pressure``
    is the absolute pressure at that point.  Gravity components along periodic
    axes are refused because the hydrostatic pressure is not periodic there.
    """

    discretization: FiniteVolumeDiscretization
    material: TwoPhaseMaterialPlan
    surface_tension_law: LinearSurfaceTensionLaw | None
    geometry: QualifiedSharpGeometry | None
    wall_sides: tuple[MACBoundarySide, ...] | None
    gravity: tuple[float, ...] = eqx.field(static=True)
    hydrostatic_reference: tuple[float, ...] | None = eqx.field(static=True)
    reference_pressure: float = eqx.field(static=True)
    surface_tension_scalar: str | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: FiniteVolumeDiscretization,
        material: TwoPhaseMaterialPlan | None = None,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        geometry: QualifiedSharpGeometry | None = None,
        gravity: tuple[float, ...] | None = None,
        hydrostatic_reference: tuple[float, ...] | None = None,
        reference_pressure: float = 0.0,
        wall_sides: tuple[MACBoundarySide, ...] | None = None,
        surface_tension_law: LinearSurfaceTensionLaw | None = None,
        surface_tension_scalar: str | None = None,
    ) -> None:
        dimension = len(discretization.cell_shape)
        if dimension not in (2, 3):
            raise ValueError("Two-phase VOF supports two or three dimensions.")
        material_ = TwoPhaseMaterialPlan() if material is None else material
        if not isinstance(material_, TwoPhaseMaterialPlan):
            raise TypeError("material must be TwoPhaseMaterialPlan or None.")
        if surface_tension_law is not None and not isinstance(
            surface_tension_law, LinearSurfaceTensionLaw
        ):
            raise TypeError(
                "surface_tension_law must be LinearSurfaceTensionLaw or None."
            )
        scalar_name = (
            None if surface_tension_scalar is None else str(surface_tension_scalar)
        )
        if (surface_tension_law is None) != (scalar_name is None):
            raise ValueError(
                "surface_tension_law and surface_tension_scalar must be given together."
            )
        if scalar_name is not None and (
            not scalar_name or scalar_name != scalar_name.strip()
        ):
            raise ValueError("surface_tension_scalar must be a canonical non-empty name.")
        if surface_tension_law is not None and material_.surface_tension != 0.0:
            raise ValueError(
                "Variable surface tension requires zero constant surface_tension."
            )
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if tolerance_ <= 0.0 or iterations <= 0:
            raise ValueError("Invalid two-phase solve policy.")
        gravity_ = _validated_point(gravity, dimension, "gravity")
        reference = _validated_point(
            hydrostatic_reference, dimension, "hydrostatic_reference"
        )
        if (gravity_ is None) != (reference is None):
            raise ValueError("gravity and hydrostatic_reference must be given together.")
        gravity_vector = (0.0,) * dimension if gravity_ is None else gravity_
        periodic = tuple(axis.periodic for axis in discretization.grid.structured_axes)
        if any(
            value != 0.0 and flag
            for value, flag in zip(gravity_vector, periodic, strict=True)
        ):
            raise ValueError(
                "Reduced gravity requires zero gravity components along periodic axes."
            )
        pressure_reference = float(reference_pressure)
        if not np.isfinite(pressure_reference):
            raise ValueError("reference_pressure must be finite.")
        sides = None if wall_sides is None else tuple(wall_sides)
        if sides is not None and not all(
            isinstance(side, MACBoundarySide) and side.kind in _WALL_KINDS
            for side in sides
        ):
            raise ValueError(
                "wall_sides must be no-slip or free-slip MACBoundarySide values."
            )
        if geometry is not None:
            if not isinstance(geometry, QualifiedSharpGeometry):
                raise TypeError("geometry must be QualifiedSharpGeometry or None.")
            if (
                geometry.support_id != discretization.support.support_id
                or geometry.cell_field_id != discretization.cell_space.field_space_id
                or geometry.face_field_ids
                != tuple(space.field_space_id for space in discretization.face_spaces)
            ):
                raise ValueError("VOF solid geometry binds another finite-volume grid.")
            if not bool(np.asarray(geometry.accepted)):
                raise ValueError("VOF preparation rejects failed sharp geometry.")
            if np.any(np.asarray(geometry.swept_cell_measure_rate) != 0.0):
                raise ValueError(
                    "Structured VOF sharp composition currently requires static geometry."
                )
            if material_.liquid_viscosity != 0.0 or material_.gas_viscosity != 0.0:
                raise ValueError(
                    "Viscous VOF does not resolve cut-solid viscous stresses; "
                    "qualified sharp geometry requires zero viscosities."
                )
            if (
                any(value != 0.0 for value in gravity_vector)
                or material_.surface_tension != 0.0
                or surface_tension_law is not None
            ):
                raise ValueError(
                    "Qualified sharp geometry does not carry the balanced interfacial "
                    "potential; it requires zero gravity and surface tension."
                )
        self.discretization = discretization
        self.material = material_
        self.surface_tension_law = surface_tension_law
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.geometry = geometry
        self.wall_sides = sides
        self.gravity = gravity_vector
        self.hydrostatic_reference = reference
        self.reference_pressure = pressure_reference
        self.surface_tension_scalar = scalar_name
        self.plan_id = canonical_fingerprint(
            {
                "kind": "incompressible-two-phase-vof-plan",
                "discretization": discretization.prepared_id,
                "material": material_.material_id,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
                "geometry": None if geometry is None else geometry.realization_id,
                "gravity": list(gravity_vector),
                "hydrostatic_reference": None if reference is None else list(reference),
                "reference_pressure": pressure_reference,
                "surface_tension_law": (
                    None if surface_tension_law is None else surface_tension_law.law_id
                ),
                "surface_tension_scalar": scalar_name,
                "wall_sides": None if sides is None else [side.side_id for side in sides],
            }
        )

    @property
    def gravity_enabled(self) -> bool:
        return any(value != 0.0 for value in self.gravity)

    @property
    def viscous(self) -> bool:
        return self.material.liquid_viscosity != 0.0 or self.material.gas_viscosity != 0.0

    def prepare(self) -> "PreparedIncompressibleTwoPhaseVOF":
        operators = MACOperatorPlan(self.discretization).prepare()
        boundaries = MACBoundaryPlan(operators, self.wall_sides).prepare()
        momentum = MACMomentumPlan(operators, boundaries=boundaries).prepare()
        projection = MACVariableDensityProjectionPlan(
            operators,
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations,
        )
        sharp_projection = (
            None
            if self.geometry is None
            else MACSharpInterfaceProjectionPlan(
                operators,
                boundaries,
                self.geometry,
                tolerance=self.tolerance,
            )
        )
        if sharp_projection is not None and sharp_projection.component_count != 1:
            raise ValueError(
                "Static VOF sharp composition currently requires one connected fluid component."
            )
        return PreparedIncompressibleTwoPhaseVOF(
            self, operators, boundaries, momentum, projection, sharp_projection
        )


class PreparedIncompressibleTwoPhaseVOF(StrictModule):
    plan: IncompressibleTwoPhaseVOFPlan
    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    momentum: PreparedMACMomentumOperators
    projection: MACVariableDensityProjectionPlan
    sharp_projection: MACSharpInterfaceProjectionPlan | None
    geometry: QualifiedSharpGeometry | None
    plic_plan: StructuredPLICPlan
    curvature_plan: HeightFunctionCurvaturePlan | None
    capillarity: MACBalancedCapillaryOperator
    viscosity: MACVariationalViscosityPlan | None
    reference_cell: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: IncompressibleTwoPhaseVOFPlan,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        momentum: PreparedMACMomentumOperators,
        projection: MACVariableDensityProjectionPlan,
        sharp_projection: MACSharpInterfaceProjectionPlan | None,
        /,
    ) -> None:
        material = plan.material
        self.plan = plan
        self.operators = operators
        self.boundaries = boundaries
        self.momentum = momentum
        self.projection = projection
        self.sharp_projection = sharp_projection
        self.geometry = plan.geometry
        self.plic_plan = StructuredPLICPlan(plan.discretization)
        interfacial = (
            material.surface_tension != 0.0
            or plan.surface_tension_law is not None
            or plan.gravity_enabled
        )
        # Height functions are prepared only when an interfacial potential
        # exists; their column-support refusal then applies to the grid.
        self.curvature_plan = (
            HeightFunctionCurvaturePlan(self.plic_plan) if interfacial else None
        )
        # Brackbill et al. (1992): dt < sqrt((rho_l + rho_g) h^3 / (4 pi sigma)),
        # evaluated below with the mean density and cfl = 1 / sqrt(2 pi).
        capillary_cfl = float(1.0 / np.sqrt(2.0 * np.pi))
        capillary_policy = (
            SurfaceTensionPolicy(
                material.surface_tension,
                min(material.liquid_density, material.gas_density),
                capillary_cfl,
                "two-phase-vof-surface-tension",
            )
            if plan.surface_tension_law is None
            else VariableSurfaceTensionPolicy(
                plan.surface_tension_law,
                density_floor=min(material.liquid_density, material.gas_density),
                capillary_cfl=capillary_cfl,
                law_id=plan.surface_tension_law.law_id,
            )
        )
        self.capillarity = MACBalancedCapillaryOperator(operators, capillary_policy)
        self.viscosity = (
            MACVariationalViscosityPlan(
                momentum,
                tolerance=plan.tolerance,
                maximum_iterations=plan.maximum_iterations,
            )
            if plan.viscous
            else None
        )
        self.reference_cell = _reference_cell(plan)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-incompressible-two-phase-vof",
                "plan": plan.plan_id,
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
                "momentum": momentum.prepared_id,
                "projection": projection.plan_id,
                "sharp_projection": (
                    None if sharp_projection is None else sharp_projection.plan_id
                ),
                "geometry": (
                    None if self.geometry is None else self.geometry.realization_id
                ),
                "plic": self.plic_plan.plan_id,
                "curvature": (
                    None if self.curvature_plan is None else self.curvature_plan.plan_id
                ),
                "capillarity": self.capillarity.operator_id,
                "viscosity": None if self.viscosity is None else self.viscosity.plan_id,
            }
        )

    @property
    def cell_fluid_measure(self) -> Array:
        return (
            self.plan.discretization.cell_volumes
            if self.geometry is None
            else self.geometry.cell_fluid_measure
        )

    @property
    def face_open_measure(self) -> FaceTuple:
        return (
            self.plan.discretization.face_measures
            if self.geometry is None
            else self.geometry.face_open_measure
        )

    @property
    def face_open_dual_measure(self) -> FaceTuple:
        if self.geometry is None:
            return self.operators.face_dual_measures
        return tuple(
            dual * opened / full
            for dual, opened, full in zip(
                self.operators.face_dual_measures,
                self.geometry.face_open_measure,
                self.geometry.face_full_measure,
                strict=True,
            )
        )

    def initial_state(
        self,
        alpha: ArrayLike,
        velocity: FaceTuple | None = None,
        phase_scalars: dict[str, ArrayLike] | None = None,
        /,
        *,
        material_scalars: dict[str, ArrayLike] | None = None,
    ) -> TwoPhaseVOFState:
        alpha_ = jnp.asarray(alpha, dtype=self.plan.discretization.cell_volumes.dtype)
        if alpha_.shape != self.plan.discretization.cell_shape:
            raise ValueError("Initial VOF alpha shape is invalid.")
        if bool(jnp.any(~jnp.isfinite(alpha_))) or bool(
            jnp.any((alpha_ < 0.0) | (alpha_ > 1.0))
        ):
            raise ValueError("Initial VOF alpha must lie in [0, 1].")
        fluid_volume = self.cell_fluid_measure
        fluid_active = fluid_volume > 0.0
        if bool(jnp.any((~fluid_active) & (alpha_ != 0.0))):
            raise ValueError("Initial VOF alpha must be zero in solid cells.")
        solid_cut = (
            jnp.zeros_like(alpha_, dtype=jnp.bool_)
            if self.geometry is None
            else self.geometry.cell_fluid_measure < self.geometry.cell_full_measure
        )
        liquid_cut = (alpha_ > 0.0) & (alpha_ < 1.0)
        if bool(jnp.any(solid_cut & liquid_cut)):
            raise ValueError(
                "Initial VOF rejects cells cut by both solid and liquid-gas PLIC."
            )
        stage = self.boundaries.homogeneous_stage()
        velocity_ = (
            tuple(
                jnp.zeros(layout.shape, dtype=alpha_.dtype)
                for layout in self.plan.discretization.face_layouts
            )
            if velocity is None
            else self.boundaries.enforce(
                self.operators.validate_velocity(velocity), stage
            )
        )
        density = self.mixture_density(alpha_)
        face_density = self.face_density(density)
        momentum = tuple(
            rho * measure * component
            for rho, measure, component in zip(
                face_density,
                self.face_open_dual_measure,
                velocity_,
                strict=True,
            )
        )
        supplied = {} if phase_scalars is None else dict(phase_scalars)
        scalar_content = {}
        for name, value in supplied.items():
            concentration = jnp.asarray(value, dtype=alpha_.dtype)
            if concentration.shape == ():
                concentration = jnp.broadcast_to(concentration, alpha_.shape)
            if concentration.shape != alpha_.shape:
                raise ValueError(f"Two-phase scalar {name!r} shape is invalid.")
            scalar_content[name] = fluid_volume * alpha_ * concentration
        supplied_material = {} if material_scalars is None else dict(material_scalars)
        material_scalar_content = {}
        for name, value in supplied_material.items():
            concentration = jnp.asarray(value, dtype=alpha_.dtype)
            if concentration.shape == ():
                concentration = jnp.broadcast_to(concentration, alpha_.shape)
            if concentration.shape != alpha_.shape:
                raise ValueError(f"Two-phase material scalar {name!r} shape is invalid.")
            if bool(jnp.any(~jnp.isfinite(concentration))):
                raise ValueError(f"Two-phase material scalar {name!r} must be finite.")
            material_scalar_content[name] = fluid_volume * concentration
        required_scalar = self.plan.surface_tension_scalar
        if required_scalar is not None and required_scalar not in material_scalar_content:
            raise ValueError(
                f"Variable surface tension requires material scalar {required_scalar!r}."
            )
        level_set = self.level_set_from_alpha(alpha_)
        geometry_epoch = (
            jnp.asarray(-1, dtype=jnp.int32)
            if self.geometry is None
            else self.geometry.epoch
        )
        geometry_id = "" if self.geometry is None else self.geometry.realization_id
        return TwoPhaseVOFState(
            liquid_content=fluid_volume * alpha_,
            momentum=momentum,
            phase_scalar_content=scalar_content,
            material_scalar_content=material_scalar_content,
            level_set=level_set,
            geometry_epoch=geometry_epoch,
            geometry_id=geometry_id,
        )

    def alpha(self, state: TwoPhaseVOFState, /) -> Array:
        expected = "" if self.geometry is None else self.geometry.realization_id
        if state.geometry_id != expected:
            raise ValueError("VOF state belongs to another solid geometry identity.")
        return self.alpha_from_content(state.liquid_content)

    def alpha_from_content(self, liquid_content: ArrayLike, /) -> Array:
        active = self.cell_fluid_measure > 0.0
        return jnp.where(
            active,
            jnp.asarray(liquid_content) / jnp.where(active, self.cell_fluid_measure, 1.0),
            0.0,
        )

    def mixture_density(self, alpha: ArrayLike, /) -> Array:
        alpha_ = jnp.asarray(alpha)
        return (
            self.plan.material.gas_density
            + (self.plan.material.liquid_density - self.plan.material.gas_density)
            * alpha_
        )

    def mixture_viscosity(self, alpha: ArrayLike, /) -> Array:
        alpha_ = jnp.asarray(alpha)
        return (
            self.plan.material.gas_viscosity
            + (self.plan.material.liquid_viscosity - self.plan.material.gas_viscosity)
            * alpha_
        )

    def face_density(self, density: ArrayLike, /) -> FaceTuple:
        """Arithmetic face density: the mass of the staggered face volume.

        This is the mass the transported face momentum carries, so kinetic
        energy, momentum transport and the projection's ``1/rho_f`` use one
        consistent density.  (A harmonic mean would make interface faces
        nearly as light as the gas and amplify any interfacial force there.)
        """

        density_ = jnp.asarray(density)
        return tuple(
            _arithmetic_faces(density_, axis, grid_axis.periodic)
            for axis, grid_axis in enumerate(
                self.plan.discretization.grid.structured_axes
            )
        )

    def velocity(self, state: TwoPhaseVOFState, /) -> FaceTuple:
        density = self.mixture_density(self.alpha(state))
        face_density = self.face_density(density)
        return tuple(
            jnp.where(
                rho * measure > 0.0,
                momentum / jnp.where(rho * measure > 0.0, rho * measure, 1.0),
                0.0,
            )
            for rho, measure, momentum in zip(
                face_density,
                self.face_open_dual_measure,
                state.momentum,
                strict=True,
            )
        )

    def _contact_angle_normal(self, normal: Array, mixed: Array, /) -> Array:
        """Impose the material contact angle on mixed wall-adjacent cells.

        With ``n`` out of the liquid and ``n_w`` the outward wall normal, the
        contact angle measured through the liquid satisfies
        ``n . n_w = -cos(theta)``.
        """

        output = normal
        angle = self.plan.material.contact_angle
        dimension = normal.shape[-1]
        for axis, grid_axis in enumerate(self.plan.discretization.grid.structured_axes):
            if grid_axis.periodic:
                continue
            for index, sign in ((0, -1.0), (-1, 1.0)):
                location: list[slice | int] = [slice(None)] * mixed.ndim
                location[axis] = index
                boundary = output[tuple(location)]
                boundary_mixed = mixed[tuple(location)]
                tangent = boundary.at[..., axis].set(0.0)
                tangent_norm = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
                basis = (
                    jnp.zeros((dimension,), dtype=normal.dtype)
                    .at[(axis + 1) % dimension]
                    .set(1.0)
                )
                direction = jnp.where(
                    tangent_norm[..., None] > 1.0e-12,
                    tangent
                    / jnp.where(tangent_norm > 1.0e-12, tangent_norm, 1.0)[..., None],
                    basis,
                )
                adjusted = jnp.sin(angle) * direction
                adjusted = adjusted.at[..., axis].set(-sign * jnp.cos(angle))
                output = output.at[tuple(location)].set(
                    jnp.where(boundary_mixed[..., None], adjusted, boundary)
                )
        return output

    def reconstruct(self, alpha: ArrayLike, /) -> StructuredPLICReconstruction:
        """Exact PLIC reconstruction with wall contact-angle normals."""

        alpha_ = jnp.asarray(alpha)
        normal = self.plic_plan.interface_normal(alpha_)
        normal = self._contact_angle_normal(normal, self.plic_plan.mixed_mask(alpha_))
        return self.plic_plan.reconstruct(alpha_, normal)

    def plic_geometry(
        self, alpha: ArrayLike, reconstruction: StructuredPLICReconstruction, /
    ) -> PLICGeometry:
        alpha_ = jnp.asarray(alpha)
        solid_cut = (
            jnp.zeros(alpha_.shape, dtype=jnp.bool_)
            if self.geometry is None
            else self.geometry.cell_fluid_measure < self.geometry.cell_full_measure
        )
        conflict = reconstruction.mixed & solid_cut
        return PLICGeometry(
            normal=reconstruction.normal,
            plane_offset=reconstruction.offset,
            mixed_cell=reconstruction.mixed,
            facet_measure=reconstruction.facet_measure,
            interface_point=reconstruction.interface_point,
            reconstruction_residual=reconstruction.residual,
            finite=reconstruction.finite,
            valid=reconstruction.valid & ~jnp.any(conflict),
            solid_interface_conflict=conflict,
        )

    def plic(self, alpha: ArrayLike, /) -> PLICGeometry:
        return self.plic_geometry(alpha, self.reconstruct(alpha))

    def interface_geometry(self, alpha: ArrayLike, /) -> TwoPhaseInterfaceGeometry:
        """PLIC plus height-function curvature/positions when interfacial."""

        reconstruction = self.reconstruct(alpha)
        return TwoPhaseInterfaceGeometry(
            reconstruction=reconstruction,
            plic=self.plic_geometry(alpha, reconstruction),
            curvature=(
                None
                if self.curvature_plan is None
                else self.curvature_plan.evaluate(alpha, reconstruction)
            ),
        )

    def gravity_vector(self, dtype: jnp.dtype, /) -> Array:
        return jnp.asarray(self.plan.gravity, dtype=dtype)

    def gravity_potential(
        self, curvature: HeightFunctionCurvatureResult, /
    ) -> tuple[Array, Array]:
        """Reduced-gravity interfacial potential and its support.

        ``phi_g = -(rho_l - rho_g) g . (x_I - Z)`` at the local interface
        position ``x_I`` (height crossing, PLIC centroid, or fitted crossing).
        """

        material = self.plan.material
        reference = self.plan.hydrostatic_reference
        position = curvature.interface_position
        if reference is None:
            return jnp.zeros(position.shape[:-1], dtype=position.dtype), jnp.zeros(
                position.shape[:-1], dtype=jnp.bool_
            )
        gravity = self.gravity_vector(position.dtype)
        height = jnp.sum(
            (position - jnp.asarray(reference, dtype=position.dtype)) * gravity, axis=-1
        )
        potential = -(material.liquid_density - material.gas_density) * height
        return jnp.where(
            curvature.position_valid, potential, 0.0
        ), curvature.position_valid

    def hydrostatic_height(self, dtype: jnp.dtype, /) -> Array:
        """Cell-center ``g . (x - Z)`` (zero without gravity)."""

        centers = self.plan.discretization.cell_centers.astype(dtype)
        reference = self.plan.hydrostatic_reference
        if reference is None:
            return jnp.zeros(centers.shape[:-1], dtype=dtype)
        return jnp.sum(
            (centers - jnp.asarray(reference, dtype=dtype)) * self.gravity_vector(dtype),
            axis=-1,
        )

    def gravitational_energy(self, liquid_content: ArrayLike, /) -> Array:
        """Potential energy ``-sum rho V g . (x - Z)`` relative to the reference."""

        content = jnp.asarray(liquid_content)
        material = self.plan.material
        mass = (
            material.gas_density * self.cell_fluid_measure
            + (material.liquid_density - material.gas_density) * content
        )
        return -jnp.sum(mass * self.hydrostatic_height(content.dtype))

    def absolute_pressure(self, alpha: ArrayLike, pressure: ArrayLike, /) -> Array:
        """Absolute pressure ``p_ref + (p - p(c_Z)) + rho(alpha) g . (x - Z)``.

        ``p`` is the gauge dynamic pressure of the projection and ``c_Z`` the
        cell containing the hydrostatic reference point, so the absolute
        pressure there equals ``reference_pressure``.  Without gravity the
        reference cell is the first cell and the hydrostatic term vanishes.
        """

        alpha_ = jnp.asarray(alpha)
        dynamic = jnp.asarray(pressure, dtype=alpha_.dtype)
        reference_value = dynamic[self.reference_cell]
        return (
            self.plan.reference_pressure
            + dynamic
            - reference_value
            + self.mixture_density(alpha_) * self.hydrostatic_height(alpha_.dtype)
        )

    def level_set_from_alpha(self, alpha: ArrayLike, /) -> Array:
        """Auxiliary signed interface indicator in length units.

        ``(alpha - 1/2) h_min`` is positive in the liquid, is the exact signed
        distance of a grid-aligned plane through a cell, and is bounded by half
        a cell elsewhere; alpha remains the sole interface authority.
        """

        alpha_ = jnp.asarray(alpha)
        width = jnp.min(
            jnp.stack(
                tuple(
                    jnp.min(jnp.asarray(axis.interval_widths, dtype=alpha_.dtype))
                    for axis in self.plan.discretization.grid.structured_axes
                )
            )
        )
        return jnp.where(self.cell_fluid_measure > 0.0, (alpha_ - 0.5) * width, 0.0)

    def topology_evidence(
        self,
        state: TwoPhaseVOFState,
        previous_alpha: ArrayLike | None = None,
        plic: PLICGeometry | None = None,
        /,
    ) -> TwoPhaseTopologyEvidence:
        alpha = self.alpha(state)
        geometry = self.plic(alpha) if plic is None else plic
        previous = alpha if previous_alpha is None else jnp.asarray(previous_alpha)
        changed = (alpha >= 0.5) != (previous >= 0.5)
        liquid_volume = jnp.sum(state.liquid_content)
        total_volume = jnp.sum(self.cell_fluid_measure)
        interface_measure = jnp.sum(geometry.facet_measure)
        finite = (
            jnp.all(jnp.isfinite(alpha))
            & jnp.isfinite(liquid_volume)
            & jnp.isfinite(interface_measure)
        )
        return TwoPhaseTopologyEvidence(
            liquid_volume=liquid_volume,
            gas_volume=total_volume - liquid_volume,
            mixed_cell_count=jnp.sum(geometry.mixed_cell, dtype=jnp.int32),
            interface_measure=interface_measure,
            component_proxy=jnp.sum(changed, dtype=jnp.int32),
            changed_cell_mask=changed,
            finite=finite,
            valid=finite
            & geometry.valid
            & jnp.all(
                (alpha >= -alpha_bound_tolerance(alpha.dtype))
                & (alpha <= 1.0 + alpha_bound_tolerance(alpha.dtype))
            ),
        )

    def view(
        self, state: TwoPhaseVOFState, pressure: ArrayLike | None = None, /
    ) -> TwoPhaseVOFView:
        alpha = self.alpha(state)
        density = self.mixture_density(alpha)
        viscosity = self.mixture_viscosity(alpha)
        pressure_ = (
            jnp.zeros_like(alpha)
            if pressure is None
            else jnp.asarray(pressure, dtype=alpha.dtype)
        )
        plic = self.plic(alpha)
        return TwoPhaseVOFView(
            alpha=alpha,
            density=density,
            viscosity=viscosity,
            velocity=self.velocity(state),
            pressure=pressure_,
            absolute_pressure=self.absolute_pressure(alpha, pressure_),
            plic=plic,
            topology=self.topology_evidence(state, None, plic),
            view_id=self.prepared_id,
            geometry_id="" if self.geometry is None else self.geometry.realization_id,
        )


def alpha_bound_tolerance(dtype: jnp.dtype, /) -> float:
    """Admissible rounding excursion of alpha outside ``[0, 1]``.

    Geometric fluxes inherit the PLIC reconstruction residual (at most
    ``256 eps``); four times that bounds its accumulation over the sweeps of a
    step.  Larger excursions are boundedness failures, never clipped.
    """

    return float(1024.0 * np.finfo(np.dtype(dtype)).eps)


def _reference_cell(plan: IncompressibleTwoPhaseVOFPlan, /) -> tuple[int, ...]:
    """Host-prepared index of the cell containing the hydrostatic reference."""

    reference = plan.hydrostatic_reference
    if reference is None:
        return (0,) * len(plan.discretization.cell_shape)
    index = []
    for axis, grid_axis in enumerate(plan.discretization.grid.structured_axes):
        centers = np.asarray(grid_axis.interval_centers, dtype=np.float64)
        index.append(int(np.argmin(np.abs(centers - reference[axis]))))
    return tuple(index)


def _arithmetic_faces(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        left = jnp.roll(moved, 1, axis=0)
        right = moved
    else:
        left = jnp.concatenate((moved[:1], moved), axis=0)
        right = jnp.concatenate((moved, moved[-1:]), axis=0)
    return jnp.moveaxis(0.5 * (left + right), 0, axis)


__all__ = [
    "IncompressibleTwoPhaseVOFPlan",
    "PLICGeometry",
    "PreparedIncompressibleTwoPhaseVOF",
    "TwoPhaseInterfaceGeometry",
    "TwoPhaseMaterialPlan",
    "TwoPhaseTopologyEvidence",
    "TwoPhaseVOFState",
    "TwoPhaseVOFView",
]
