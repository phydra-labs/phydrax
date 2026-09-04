#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._geometry_precision import GeometryPrecisionPolicy
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_boundary import MACBoundaryStageData
from ..discretization.finite_volume._mac_momentum import PreparedMACMomentumOperators
from ..discretization.finite_volume._mac_ocean import (
    MACOceanForcingEvidence,
    PreparedMACOceanForcing,
)
from ..discretization.finite_volume._mac_scalar import (
    MACScalarDiagnostics,
    MACScalarFluxResult,
    MACScalarProblem,
    MACScalarSGSPlan,
    MACScalarStepRestriction,
    PreparedMACScalarSGS,
    PreparedMACScalarTransport,
)
from ..discretization.finite_volume._mac_variational_viscosity import (
    MACVariationalViscosityResult,
    PreparedMACVariationalViscosityAction,
)
from ._incompressible import IncompressibleFlowProblem
from ._ksgs import (
    AbstractKSGSPlan,
    BuoyancyKSGSInputs,
    BuoyancyKSGSPlan,
    DynamicKSGSInputs,
    DynamicKSGSPlan,
    KSGSInputs,
    KSGSResult,
    KSGSState,
    KSGSTransportResult,
    LowReKSGSInputs,
    LowReKSGSPlan,
    replace_ksgs_kinetic_energy,
    StaticKSGSPlan,
)
from ._les_closures import LESFilterScale
from ._mac_dynamic_les import (
    MACExplicitTestFilterPlan,
    PreparedMACExplicitTestFilter,
)
from ._mac_incompressible import (
    compile_mac_incompressible_flow,
    CompiledMACIncompressibleDynamics,
    MACIncompressibleRateComponents,
    MACLESStepRestriction,
)
from ._mac_les import (
    _axis_values,
    _cell_centered_component,
    _periodic_center_derivative,
    _wall_center_derivative,
    MACAlgebraicLESPlan,
)


if TYPE_CHECKING:
    from ..solver._structured_incompressible import MACPressureProjectionPlan


def _canonical_coefficients(
    coefficients: Mapping[str, ArrayLike],
    references: Mapping[str, ArrayLike] | None,
    /,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    supplied = {str(name): jnp.asarray(value) for name, value in coefficients.items()}
    names = tuple(sorted(supplied))
    if not names or any(not name for name in names):
        raise ValueError("MAC buoyancy coefficients require non-empty field names.")
    reference_values = (
        {name: jnp.asarray(0.0) for name in names}
        if references is None
        else {str(name): jnp.asarray(value) for name, value in references.items()}
    )
    if set(reference_values) != set(names):
        raise ValueError("MAC buoyancy references must exactly match coefficient fields.")
    coefficient_values = []
    references_ = []
    for name in names:
        coefficient = supplied[name]
        reference = reference_values[name]
        if (
            coefficient.shape != ()
            or reference.shape != ()
            or jnp.iscomplexobj(coefficient)
            or jnp.iscomplexobj(reference)
            or not bool(jnp.isfinite(coefficient) & jnp.isfinite(reference))
        ):
            raise ValueError(
                "MAC buoyancy coefficients and references must be finite real scalars."
            )
        coefficient_values.append(float(coefficient))
        references_.append(float(reference))
    return names, tuple(coefficient_values), tuple(references_)


class MACBuoyancyLedger(StrictModule):
    """Face-exact kinetic/potential exchange and diffusive mixing evidence."""

    force: FaceVelocity
    power_by_field: dict[str, Array]
    potential_energy_rate_by_field: dict[str, Array]
    molecular_potential_energy_mixing_by_field: dict[str, Array]
    sgs_potential_energy_mixing_by_field: dict[str, Array]
    boundary_potential_energy_rate_by_field: dict[str, Array]
    total_power: Array
    potential_energy_rate: Array
    molecular_potential_energy_mixing: Array
    sgs_potential_energy_mixing: Array
    boundary_potential_energy_rate: Array
    exchange_defect: Array
    exchange_scale: Array
    normalized_exchange_defect: Array
    tolerance: Array
    finite: Array
    success: Array
    potential_energy_mixing_available: bool = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)


class MACBuoyancyLaw(StrictModule, NonTrainableState):
    """Named Boussinesq acceleration from transported scalar anomalies."""

    gravity: Array
    active_gravity_axes: tuple[int, ...] = eqx.field(static=True)
    principal_gravity_axis: int | None = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    coefficients: tuple[float, ...] = eqx.field(static=True)
    references: tuple[float, ...] = eqx.field(static=True)
    enforce_exchange: bool = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravity: Sequence[float] | ArrayLike,
        coefficients: Mapping[str, ArrayLike],
        /,
        *,
        references: Mapping[str, ArrayLike] | None = None,
        enforce_exchange: bool = False,
        law_id: str | None = None,
    ):
        gravity_ = jnp.asarray(gravity, dtype=float)
        if (
            gravity_.shape not in ((2,), (3,))
            or jnp.iscomplexobj(gravity_)
            or not bool(jnp.all(jnp.isfinite(gravity_)))
        ):
            raise ValueError(
                "MAC buoyancy gravity must be a finite real 2D or 3D vector."
            )
        names, coefficient_values, reference_values = _canonical_coefficients(
            coefficients, references
        )
        gravity_host = np.asarray(gravity_)
        active_axes = tuple(
            axis for axis, value in enumerate(gravity_host) if value != 0.0
        )
        principal_axis = None if not active_axes else int(np.argmax(np.abs(gravity_host)))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mac-buoyancy-law",
                    "gravity": np.asarray(gravity_).tolist(),
                    "fields": list(names),
                    "coefficients": list(coefficient_values),
                    "references": list(reference_values),
                    "enforce_exchange": bool(enforce_exchange),
                }
            )
            if law_id is None
            else str(law_id)
        )
        if not identifier:
            raise ValueError("law_id must be non-empty.")
        self.gravity = gravity_
        self.active_gravity_axes = active_axes
        self.principal_gravity_axis = principal_axis
        self.field_names = names
        self.coefficients = coefficient_values
        self.references = reference_values
        self.enforce_exchange = bool(enforce_exchange)
        self.law_id = identifier

    def evaluate(
        self,
        velocity: FaceVelocity,
        scalar_fluxes: Mapping[str, MACScalarFluxResult],
        transport: PreparedMACScalarTransport,
        momentum: PreparedMACMomentumOperators,
        /,
        *,
        projection_id: str,
    ) -> MACBuoyancyLedger:
        if not isinstance(transport, PreparedMACScalarTransport):
            raise TypeError("transport must be PreparedMACScalarTransport.")
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        if transport.layout.operators.prepared_id != momentum.operators.prepared_id:
            raise ValueError("MAC buoyancy momentum and scalar transport grids differ.")
        if len(self.gravity) != momentum.dimension:
            raise ValueError("MAC buoyancy gravity and momentum dimensions differ.")
        if not set(self.field_names).issubset(transport.layout.field_names):
            raise ValueError("MAC buoyancy fields must belong to scalar transport.")
        projection_identifier = str(projection_id)
        if not projection_identifier:
            raise ValueError("projection_id must be non-empty.")
        velocity_ = momentum.operators.validate_velocity(velocity)
        fluxes = dict(scalar_fluxes)
        if set(fluxes) != set(transport.layout.field_names):
            raise ValueError("MAC buoyancy requires every named scalar flux result.")
        discretization = momentum.operators.discretization
        force = tuple(
            jnp.zeros(layout.shape, dtype=transport.layout.dtype)
            for layout in discretization.face_layouts
        )
        power_by_field: dict[str, Array] = {}
        potential_by_field: dict[str, Array] = {}
        molecular_mixing_by_field: dict[str, Array] = {}
        sgs_mixing_by_field: dict[str, Array] = {}
        boundary_potential_by_field: dict[str, Array] = {}
        mixing_available = all(
            not discretization.grid.structured_axes[axis].periodic
            for axis in self.active_gravity_axes
        )
        gravity_coordinate = jnp.sum(
            discretization.cell_centers * self.gravity.astype(transport.layout.dtype),
            axis=-1,
        )
        for name, coefficient, reference in zip(
            self.field_names,
            self.coefficients,
            self.references,
            strict=True,
        ):
            result = fluxes[name]
            if (
                not isinstance(result, MACScalarFluxResult)
                or result.field_name != name
                or result.transport_id != transport.prepared_id
                or result.grid_id != momentum.operators.discretization.grid.prepared_id
            ):
                raise ValueError("MAC buoyancy scalar flux provenance does not match.")
            field_force = tuple(
                jnp.asarray(coefficient, dtype=transport.layout.dtype)
                * self.gravity[axis].astype(transport.layout.dtype)
                * (face_value - jnp.asarray(reference, dtype=transport.layout.dtype))
                for axis, face_value in enumerate(result.face_values)
            )
            force = tuple(
                total + contribution
                for total, contribution in zip(force, field_force, strict=True)
            )
            power = jnp.real(
                momentum.operators.velocity_space.inner(velocity_, field_force)
            )
            potential_rate = -sum(
                jnp.sum(
                    dual_measure
                    * jnp.asarray(coefficient, dtype=transport.layout.dtype)
                    * self.gravity[axis].astype(transport.layout.dtype)
                    * (
                        result.advective_fluxes[axis]
                        - jnp.asarray(reference, dtype=transport.layout.dtype)
                        * velocity_[axis]
                    )
                )
                for axis, dual_measure in enumerate(momentum.operators.face_dual_measures)
            )
            potential_factor = (
                -jnp.asarray(coefficient, dtype=transport.layout.dtype)
                * gravity_coordinate
            )
            zero = jnp.asarray(0.0, dtype=transport.layout.dtype)
            molecular_mixing = (
                jnp.sum(
                    discretization.cell_volumes
                    * potential_factor
                    * result.molecular_diffusive_divergence
                )
                if mixing_available
                else zero
            )
            sgs_mixing = (
                jnp.sum(
                    discretization.cell_volumes
                    * potential_factor
                    * result.sgs_diffusive_divergence
                )
                if mixing_available
                else zero
            )
            boundary_potential = (
                jnp.sum(
                    discretization.cell_volumes
                    * potential_factor
                    * result.boundary_diffusive_divergence
                )
                if mixing_available
                else zero
            )
            power_by_field[name] = power
            potential_by_field[name] = potential_rate
            molecular_mixing_by_field[name] = molecular_mixing
            sgs_mixing_by_field[name] = sgs_mixing
            boundary_potential_by_field[name] = boundary_potential
        force = tuple(
            eqx.error_if(
                component,
                jnp.any(~jnp.isfinite(component)),
                "MAC buoyancy force must be finite.",
            )
            for component in force
        )
        total_power = sum(power_by_field.values())
        potential_energy_rate = sum(potential_by_field.values())
        molecular_mixing = sum(molecular_mixing_by_field.values())
        sgs_mixing = sum(sgs_mixing_by_field.values())
        boundary_potential = sum(boundary_potential_by_field.values())
        exchange_defect = total_power + potential_energy_rate
        dtype = velocity_[0].dtype
        exchange_scale = jnp.maximum(
            jnp.abs(total_power) + jnp.abs(potential_energy_rate),
            jnp.asarray(1.0, dtype=dtype),
        )
        normalized_exchange_defect = jnp.abs(exchange_defect) / exchange_scale
        tolerance = 128.0 * jnp.finfo(dtype).eps
        finite = (
            jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in force)))
            & jnp.isfinite(total_power)
            & jnp.isfinite(potential_energy_rate)
            & jnp.isfinite(molecular_mixing)
            & jnp.isfinite(sgs_mixing)
            & jnp.isfinite(boundary_potential)
            & jnp.isfinite(exchange_defect)
            & jnp.isfinite(normalized_exchange_defect)
        )
        return MACBuoyancyLedger(
            force=force,
            power_by_field=power_by_field,
            potential_energy_rate_by_field=potential_by_field,
            molecular_potential_energy_mixing_by_field=molecular_mixing_by_field,
            sgs_potential_energy_mixing_by_field=sgs_mixing_by_field,
            boundary_potential_energy_rate_by_field=boundary_potential_by_field,
            total_power=total_power,
            potential_energy_rate=potential_energy_rate,
            molecular_potential_energy_mixing=molecular_mixing,
            sgs_potential_energy_mixing=sgs_mixing,
            boundary_potential_energy_rate=boundary_potential,
            exchange_defect=exchange_defect,
            exchange_scale=exchange_scale,
            normalized_exchange_defect=normalized_exchange_defect,
            tolerance=tolerance,
            finite=finite,
            success=finite
            & (
                (normalized_exchange_defect <= tolerance)
                | jnp.asarray(not self.enforce_exchange)
            ),
            potential_energy_mixing_available=mixing_available,
            law_id=self.law_id,
            transport_id=transport.prepared_id,
            momentum_id=momentum.prepared_id,
            projection_id=projection_identifier,
            grid_id=momentum.operators.discretization.grid.prepared_id,
            ledger_id=canonical_fingerprint(
                {
                    "kind": "mac-buoyancy-ledger",
                    "law": self.law_id,
                    "transport": transport.prepared_id,
                    "momentum": momentum.prepared_id,
                    "projection": projection_identifier,
                    "mixing_available": mixing_available,
                }
            ),
        )


class MACKSGSStageResult(StrictModule):
    """One prognostic KSGS constitutive and MAC momentum stage."""

    state: KSGSState
    transport: KSGSTransportResult
    result: KSGSResult
    velocity_gradient: Array
    viscosity_result: MACVariationalViscosityResult
    finite: Array
    success: Array
    prepared_id: str = eqx.field(static=True)
    boundary_stage_id: str = eqx.field(static=True)


class PreparedMACKSGS(StrictModule, NonTrainableState):
    """Structured-MAC runtime for prognostic KSGS closure families."""

    plan: AbstractKSGSPlan
    momentum: PreparedMACMomentumOperators
    scalar_field_name: str = eqx.field(static=True)
    filter_scale: LESFilterScale
    test_filter: PreparedMACExplicitTestFilter | None
    wall_distance: Array | None
    viscosity_action: PreparedMACVariationalViscosityAction
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AbstractKSGSPlan,
        momentum: PreparedMACMomentumOperators,
        scalar_field_name: str,
        /,
    ):
        if not isinstance(
            plan,
            (StaticKSGSPlan, BuoyancyKSGSPlan, DynamicKSGSPlan, LowReKSGSPlan),
        ):
            raise TypeError("Unsupported MAC KSGS plan.")
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        field_name = str(scalar_field_name)
        if not field_name:
            raise ValueError("MAC KSGS requires a non-empty scalar field name.")
        if momentum.dimension != 3:
            raise ValueError("MAC KSGS requires a three-dimensional grid.")
        grid = momentum.operators.discretization.grid
        dynamic = isinstance(plan, DynamicKSGSPlan)
        low_re = isinstance(plan, LowReKSGSPlan)
        allowed_boundaries = (
            ("no-slip", "free-slip", "symmetry") if low_re else ("free-slip", "symmetry")
        )
        unsupported = tuple(
            side.kind
            for side in momentum.boundaries.sides
            if side.kind not in allowed_boundaries
        )
        if unsupported:
            raise ValueError(
                "MAC KSGS supports only its admitted impermeable momentum "
                "boundary subset."
            )
        provenance = plan.provenance
        resolved_filter = provenance.resolved_filter
        boundary_class = (
            "periodic"
            if all(axis.periodic for axis in grid.structured_axes)
            else "wall-bounded"
        )
        expected_commutation = "commuting" if dynamic else "unmodeled"
        if (
            resolved_filter.family != "implicit-grid-volume"
            or resolved_filter.axis_names != grid.axis_names
            or resolved_filter.topology != "tensor-product"
            or resolved_filter.boundary_class != boundary_class
            or resolved_filter.scale_rule != "volume-equivalent"
            or resolved_filter.commutation_status != expected_commutation
            or resolved_filter.repeated_filter_semantics != "unmodeled"
        ):
            raise ValueError(
                "KSGS filter semantics do not match the admitted structured MAC "
                "implicit grid-volume filter."
            )
        if provenance.discretization_id != momentum.operators.discretization.prepared_id:
            raise ValueError("KSGS provenance does not match the MAC discretization.")
        if provenance.regime != "incompressible-unit-density":
            raise ValueError(
                "MAC KSGS requires the 'incompressible-unit-density' regime."
            )
        widths = tuple(axis.interval_widths for axis in grid.structured_axes)
        directional = jnp.stack(
            tuple(
                jnp.broadcast_to(
                    _axis_values(width, 3, axis),
                    momentum.operators.discretization.cell_shape,
                )
                for axis, width in enumerate(widths)
            ),
            axis=-1,
        )
        filter_scale = LESFilterScale(directional)
        test_filter = (
            MACExplicitTestFilterPlan(plan.test_filter).prepare(momentum)
            if dynamic
            else None
        )
        if dynamic and (
            plan.test_filter_scale_ratio != 2.0
            or test_filter is None
            or test_filter.plan.test_filter.filter_id != plan.test_filter.filter_id
            or test_filter.test_filter_ratio != (2.0, 2.0, 2.0)
        ):
            raise ValueError(
                "Dynamic MAC KSGS requires the exact prepared periodic-uniform "
                "binomial test-filter identity and scale ratio two."
            )
        wall_distance = self._resolved_wall_distance(momentum) if low_re else None
        action = PreparedMACVariationalViscosityAction(momentum)
        self.plan = plan
        self.momentum = momentum
        self.scalar_field_name = field_name
        self.filter_scale = filter_scale
        self.test_filter = test_filter
        self.wall_distance = wall_distance
        self.viscosity_action = action
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-ksgs",
                "plan": plan.plan_id,
                "momentum": momentum.prepared_id,
                "field": field_name,
                "filter": resolved_filter.filter_id,
                "test_filter": (
                    "none" if test_filter is None else test_filter.prepared_id
                ),
                "wall_distance": (
                    "none"
                    if wall_distance is None
                    else "resolved-no-slip-cell-center-distance"
                ),
                "viscosity_action": action.action_id,
            }
        )

    @staticmethod
    def _resolved_wall_distance(
        momentum: PreparedMACMomentumOperators,
        /,
    ) -> Array:
        discretization = momentum.operators.discretization
        centers = discretization.cell_centers
        distance = jnp.full(
            discretization.cell_shape,
            jnp.inf,
            dtype=momentum.operators.pressure_space.dtype,
        )
        no_slip_walls = 0
        for side in momentum.boundaries.sides:
            if side.kind != "no-slip":
                continue
            axis_index = discretization.grid.axis_names.index(side.axis)
            bound_index = 0 if side.side == "lower" else 1
            bound = discretization.grid.structured_axes[axis_index].bounds[bound_index]
            distance = jnp.minimum(
                distance,
                jnp.abs(centers[..., axis_index] - bound),
            )
            no_slip_walls += 1
        if no_slip_walls == 0:
            raise ValueError(
                "Low-Re MAC KSGS requires at least one resolved no-slip wall "
                "for wall distance; free-slip and symmetry sides are not walls "
                "for low-Re damping."
            )
        return distance

    def velocity_gradient(self, velocity: FaceVelocity, /) -> Array:
        values = self.momentum.operators.validate_velocity(velocity)
        grid = self.momentum.operators.discretization.grid
        centered = tuple(
            _cell_centered_component(value, axis, grid.structured_axes[axis].periodic)
            for axis, value in enumerate(values)
        )
        rows = []
        for component_axis, (face_value, cell_value) in enumerate(
            zip(values, centered, strict=True)
        ):
            derivatives = []
            for derivative_axis, axis in enumerate(grid.structured_axes):
                if derivative_axis == component_axis:
                    moved = jnp.moveaxis(face_value, derivative_axis, 0)
                    difference = (
                        jnp.roll(moved, -1, axis=0) - moved
                        if axis.periodic
                        else moved[1:] - moved[:-1]
                    )
                    derivative = jnp.moveaxis(
                        difference / _axis_values(axis.interval_widths, moved.ndim, 0),
                        0,
                        derivative_axis,
                    )
                elif axis.periodic:
                    derivative = _periodic_center_derivative(
                        cell_value,
                        axis.interval_centers,
                        axis.bounds[1] - axis.bounds[0],
                        derivative_axis,
                    )
                else:
                    derivative = _wall_center_derivative(
                        cell_value,
                        axis.interval_centers,
                        axis.bounds[0],
                        axis.bounds[1],
                        derivative_axis,
                    )
                derivatives.append(derivative)
            rows.append(jnp.stack(tuple(derivatives), axis=-1))
        return jnp.stack(tuple(rows), axis=-2)

    def _conservative_cell_gradient(self, field: Array, /) -> Array:
        grid = self.momentum.operators.discretization.grid
        derivatives = []
        for axis_index, axis in enumerate(grid.structured_axes):
            moved = jnp.moveaxis(field, axis_index, 0)
            if axis.periodic:
                lower = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
                upper = jnp.roll(lower, -1, axis=0)
            else:
                internal = 0.5 * (moved[:-1] + moved[1:])
                lower = jnp.concatenate((moved[:1], internal), axis=0)
                upper = jnp.concatenate((internal, moved[-1:]), axis=0)
            derivative = (upper - lower) / _axis_values(
                axis.interval_widths, moved.ndim, 0
            )
            derivatives.append(jnp.moveaxis(derivative, 0, axis_index))
        return jnp.stack(tuple(derivatives), axis=-1)

    @staticmethod
    def _coefficient_free_ksgs_tensor(
        gradient: Array,
        filter_scale: LESFilterScale,
        kinetic_energy: Array,
        /,
    ) -> Array:
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        trace = jnp.trace(strain, axis1=-2, axis2=-1)
        deviatoric = (
            strain - trace[..., None, None] * jnp.eye(3, dtype=gradient.dtype) / 3.0
        )
        return (
            -2.0
            * filter_scale.equivalent_width[..., None, None]
            * jnp.sqrt(kinetic_energy)[..., None, None]
            * deviatoric
        )

    def prepare_transport(
        self,
        kinetic_energy: Array,
        molecular_kinematic_viscosity: Array,
        /,
        *,
        continuation_state: KSGSState | None = None,
    ) -> tuple[KSGSState, KSGSTransportResult]:
        state = (
            self.plan.initialize_state(kinetic_energy)
            if continuation_state is None
            else replace_ksgs_kinetic_energy(continuation_state, kinetic_energy)
        )
        transport = self.plan.transport(
            state,
            self.filter_scale,
            molecular_kinematic_viscosity,
            wall_distance=self.wall_distance,
        )
        return state, transport

    def evaluate(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        state: KSGSState,
        transport: KSGSTransportResult,
        diffusion_rate: Array,
        molecular_kinematic_viscosity: Array,
        buoyancy_frequency_squared: Array,
        /,
        *,
        averaging_weight: ArrayLike = 1.0,
        accept_update: ArrayLike = False,
    ) -> MACKSGSStageResult:
        gradient = self.velocity_gradient(velocity)
        base_inputs = KSGSInputs(
            gradient,
            self.filter_scale,
            molecular_kinematic_viscosity,
            diffusion_rate,
        )
        if isinstance(self.plan, BuoyancyKSGSPlan):
            inputs: object = BuoyancyKSGSInputs(base_inputs, buoyancy_frequency_squared)
        elif isinstance(self.plan, DynamicKSGSPlan):
            if self.test_filter is None:
                raise ValueError("Dynamic MAC KSGS has no prepared test filter.")
            cell_velocity = jnp.stack(
                tuple(
                    _cell_centered_component(component, axis, True)
                    for axis, component in enumerate(velocity)
                ),
                axis=-1,
            )
            test_velocity = self.test_filter.apply(cell_velocity)
            leonard = (
                self.test_filter.apply(
                    cell_velocity[..., :, None] * cell_velocity[..., None, :]
                )
                - test_velocity[..., :, None] * test_velocity[..., None, :]
            )
            test_gradient = self.test_filter.apply(gradient)
            resolved_tensor = self._coefficient_free_ksgs_tensor(
                gradient,
                self.filter_scale,
                state.kinetic_energy,
            )
            test_tensor = self._coefficient_free_ksgs_tensor(
                test_gradient,
                self.test_filter.test_filter_scale(),
                self.test_filter.apply(state.kinetic_energy),
            )
            modeled = test_tensor - self.test_filter.apply(resolved_tensor)
            shape = state.kinetic_energy.shape
            requested_update = jnp.broadcast_to(
                jnp.asarray(accept_update, dtype=bool), shape
            )
            sample_numerator = jnp.sum(leonard * modeled, axis=(-2, -1))
            sample_denominator = jnp.sum(modeled * modeled, axis=(-2, -1))
            accepted_update = (
                requested_update & (sample_numerator >= 0.0) & (sample_denominator > 0.0)
            )
            inputs = DynamicKSGSInputs(
                base_inputs,
                leonard,
                modeled,
                jnp.broadcast_to(jnp.asarray(averaging_weight), shape),
                accepted_update,
            )
        elif isinstance(self.plan, LowReKSGSPlan):
            if self.wall_distance is None:
                raise ValueError("Low-Re MAC KSGS has no resolved wall distance.")
            inputs = LowReKSGSInputs(
                base_inputs,
                self.wall_distance,
                self._conservative_cell_gradient(jnp.sqrt(state.kinetic_energy)),
            )
        else:
            inputs = base_inputs
        result = self.plan.evaluate(state, inputs)
        viscosity_result = self.viscosity_action.evaluate(
            velocity,
            result.eddy_viscosity,
            boundary_stage,
        )
        finite = (
            jnp.all(result.evidence.finite)
            & jnp.all(transport.finite)
            & jnp.all(jnp.isfinite(gradient))
            & viscosity_result.finite
        )
        success = (
            finite
            & jnp.all(result.evidence.kinetic_energy_nonnegative)
            & jnp.all(result.evidence.eddy_viscosity_nonnegative)
            & jnp.all(result.evidence.production_nonnegative)
            & jnp.all(result.evidence.dissipation_nonnegative)
            & viscosity_result.successful
        )
        return MACKSGSStageResult(
            state=state,
            transport=transport,
            result=result,
            velocity_gradient=gradient,
            viscosity_result=viscosity_result,
            finite=finite,
            success=success,
            prepared_id=self.prepared_id,
            boundary_stage_id=boundary_stage.stage_id,
        )


class MACScalarBuoyancyStage(StrictModule):
    """One provenance-complete coupled momentum, scalar, and buoyancy stage."""

    boundary_stage: MACBoundaryStageData
    velocity: FaceVelocity
    scalars: dict[str, Array]
    momentum_components: MACIncompressibleRateComponents
    scalar_sgs_diffusivities: dict[str, Array]
    scalar_fluxes: dict[str, MACScalarFluxResult]
    ksgs: MACKSGSStageResult | None
    buoyancy: MACBuoyancyLedger
    ocean_forcing: MACOceanForcingEvidence | None
    unconstrained_velocity_rate: FaceVelocity
    velocity_rate: FaceVelocity
    scalar_rates: dict[str, Array]
    pressure: Array
    pressure_residual: Array
    divergence_before: Array
    divergence_after: Array
    projection_converged: Array
    finite: Array
    success: Array
    compilation_id: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)


class MACScalarBuoyancyDiagnostics(StrictModule):
    """Coupled kinetic, scalar-content, variance, and buoyancy ledgers."""

    scalars: MACScalarDiagnostics
    buoyancy: MACBuoyancyLedger
    kinetic_energy: Array
    nonlinear_energy_rate: Array
    forcing_power: Array
    buoyancy_power: Array
    ocean_forcing_power: Array
    viscous_energy_rate: Array
    sgs_energy_rate: Array
    sgs_dissipation: Array
    sgs_boundary_power: Array
    sgs_energy_transfer: Array
    dissipation: Array
    wall_power: Array
    semidiscrete_energy_rate: Array
    energy_balance_defect: Array
    divergence_norm: Array
    pressure_residual_norm: Array
    pressure_gauge_residual: Array
    projection_converged: Array
    finite: Array
    success: Array
    compilation_id: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class MACScalarBuoyancyStepRestriction(StrictModule):
    """Combined explicit momentum and named scalar stage restriction."""

    momentum: MACLESStepRestriction
    scalars: MACScalarStepRestriction
    ksgs: Array
    ocean_forcing: Array
    stratification: Array
    selected: Array
    finite: Array
    success: Array
    compilation_id: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)


class CompiledMACScalarBuoyancyDynamics(StrictModule):
    """Flat velocity-plus-named-scalar Boussinesq MAC dynamics."""

    flow_problem: IncompressibleFlowProblem
    scalar_problem: MACScalarProblem
    momentum: PreparedMACMomentumOperators
    projection: MACPressureProjectionPlan
    transport: PreparedMACScalarTransport
    scalar_sgs: PreparedMACScalarSGS | None
    ksgs: PreparedMACKSGS | None
    buoyancy: MACBuoyancyLaw
    ocean_forcing: PreparedMACOceanForcing | None
    base_dynamics: CompiledMACIncompressibleDynamics
    discretization_bundle: DiscretizationBundle
    velocity_size: int = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        flow_problem: IncompressibleFlowProblem,
        scalar_problem: MACScalarProblem,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        transport: PreparedMACScalarTransport,
        buoyancy: MACBuoyancyLaw,
        base_dynamics: CompiledMACIncompressibleDynamics,
        scalar_sgs: PreparedMACScalarSGS | None,
        ksgs: PreparedMACKSGS | None,
        ocean_forcing: PreparedMACOceanForcing | None = None,
        /,
        *,
        compilation_id: str,
    ):
        discretization = momentum.operators.discretization
        residual_key = DiscretizationKey(
            "mac_scalar_buoyancy_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                ),
                DiscretizationRecord(
                    residual_key,
                    "compiled-mac-scalar-buoyancy-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        if ocean_forcing is not None and (
            not isinstance(ocean_forcing, PreparedMACOceanForcing)
            or ocean_forcing.operators.prepared_id != momentum.operators.prepared_id
        ):
            raise ValueError(
                "Ocean forcing must be prepared on the coupled MAC operators."
            )
        if scalar_sgs is not None and (
            not isinstance(scalar_sgs, PreparedMACScalarSGS)
            or scalar_sgs.transport.prepared_id != transport.prepared_id
        ):
            raise ValueError("Prepared scalar SGS must match the coupled transport.")
        if ksgs is not None and (
            not isinstance(ksgs, PreparedMACKSGS)
            or ksgs.momentum.prepared_id != momentum.prepared_id
            or ksgs.scalar_field_name not in transport.layout.field_names
        ):
            raise ValueError("Prepared MAC KSGS must match momentum and scalar layout.")
        momentum_les_active = base_dynamics.algebraic_les is not None
        ksgs_active = ksgs is not None
        if momentum_les_active and ksgs_active:
            raise ValueError("Algebraic MAC LES and prognostic KSGS are alternatives.")
        if (momentum_les_active or ksgs_active) != (scalar_sgs is not None):
            raise ValueError(
                "Active MAC LES requires a complete prepared scalar SGS contract, "
                "and scalar SGS cannot be active without momentum LES or KSGS."
            )
        self.flow_problem = flow_problem
        self.scalar_problem = scalar_problem
        self.momentum = momentum
        self.projection = projection
        self.transport = transport
        self.scalar_sgs = scalar_sgs
        self.ksgs = ksgs
        self.buoyancy = buoyancy
        self.ocean_forcing = ocean_forcing
        self.base_dynamics = base_dynamics
        self.discretization_bundle = bundle
        self.velocity_size = momentum.operators.velocity_space.size
        self.compilation_id = str(compilation_id)
        self.source_hash = canonical_fingerprint(
            {
                "flow": flow_problem.problem_id,
                "base_dynamics": base_dynamics.compilation_id,
                "scalars": scalar_problem.problem_id,
                "scalar_sgs": ("none" if scalar_sgs is None else scalar_sgs.prepared_id),
                "ksgs": "none" if ksgs is None else ksgs.prepared_id,
                "buoyancy": buoyancy.law_id,
                "ocean_forcing": (
                    "none" if ocean_forcing is None else ocean_forcing.plan_id
                ),
            }
        )
        self.resolved_method = (
            "mac-symmetry-preserving-projected-scalar-buoyancy"
            if scalar_sgs is None
            else (
                "mac-symmetry-preserving-ksgs-projected-scalar-buoyancy"
                if ksgs is not None
                else "mac-symmetry-preserving-les-projected-scalar-buoyancy"
            )
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.velocity_size + self.transport.layout.state_size,)

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Coupled MAC coordinates must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        dtype = self.momentum.operators.pressure_space.dtype
        if value.dtype != dtype:
            raise TypeError(f"Coupled MAC coordinates must have dtype {dtype}.")
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "Coupled MAC coordinates must be finite.",
        )

    def pack_state(
        self,
        velocity: FaceVelocity,
        scalars: Mapping[str, ArrayLike],
        /,
    ) -> Array:
        velocity_coordinates = self.base_dynamics.pack_velocity(velocity)
        scalar_coordinates = self.transport.layout.pack(scalars)
        return self.validate_state(
            jnp.concatenate((velocity_coordinates, scalar_coordinates))
        )

    def unpack_state(self, state: ArrayLike, /) -> tuple[FaceVelocity, dict[str, Array]]:
        value = self.validate_state(state)
        velocity = self.base_dynamics.unpack_velocity(value[: self.velocity_size])
        scalars = self.transport.layout.unpack(value[self.velocity_size :])
        return velocity, scalars

    def project_state(
        self,
        velocity: FaceVelocity,
        scalars: Mapping[str, ArrayLike],
        /,
    ) -> Array:
        velocity_coordinates = self.base_dynamics.project_state(velocity)
        scalar_coordinates = self.transport.layout.pack(scalars)
        return self.validate_state(
            jnp.concatenate((velocity_coordinates, scalar_coordinates))
        )

    def physical_state(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[FaceVelocity, dict[str, Array]]:
        del time, args
        return self.unpack_state(state)

    def _buoyancy_frequency_squared(
        self,
        scalars: Mapping[str, Array],
        /,
    ) -> Array:
        density_anomaly = jnp.zeros(
            self.momentum.operators.discretization.cell_shape,
            dtype=self.momentum.operators.pressure_space.dtype,
        )
        for name, coefficient, reference in zip(
            self.buoyancy.field_names,
            self.buoyancy.coefficients,
            self.buoyancy.references,
            strict=True,
        ):
            density_anomaly = density_anomaly + coefficient * (scalars[name] - reference)
        grid = self.momentum.operators.discretization.grid
        frequency_squared = jnp.zeros_like(density_anomaly)
        for axis_index in self.buoyancy.active_gravity_axes:
            axis = grid.structured_axes[axis_index]
            gravity_component = self.buoyancy.gravity[axis_index].astype(
                density_anomaly.dtype
            )
            derivative = (
                _periodic_center_derivative(
                    density_anomaly,
                    axis.interval_centers,
                    axis.bounds[1] - axis.bounds[0],
                    axis_index,
                )
                if axis.periodic
                else _wall_center_derivative(
                    density_anomaly,
                    axis.interval_centers,
                    axis.bounds[0],
                    axis.bounds[1],
                    axis_index,
                )
            )
            frequency_squared = frequency_squared + gravity_component * derivative
        return frequency_squared

    def stage(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        ksgs_state: KSGSState | None = None,
        accept_ksgs_update: ArrayLike = False,
    ) -> MACScalarBuoyancyStage:
        value = self.validate_state(state)
        time_ = jnp.asarray(time)
        velocity_coordinates = value[: self.velocity_size]
        raw_velocity, scalars = self.unpack_state(value)
        boundary_stage = self.base_dynamics.boundary_stage(time_, args)
        velocity = self.momentum.boundaries.enforce(
            raw_velocity,
            boundary_stage,
        )
        base_components = self.base_dynamics._rate_components(
            time_,
            velocity_coordinates,
            args,
            boundary_stage,
        )
        ksgs_stage: MACKSGSStageResult | None = None
        if self.scalar_sgs is None:
            scalar_sgs_diffusivities = self.transport._runtime_sgs_diffusivities(None)
            scalar_fluxes = self.transport.evaluate(time_, scalars, velocity, args)
        elif self.ksgs is None:
            les_stage = base_components.les_stage
            if les_stage is None:
                raise ValueError("Prepared scalar SGS requires a momentum LES stage.")
            scalar_sgs_diffusivities = self.scalar_sgs.diffusivities(
                les_stage.model_result.kinematic_viscosity
            )
            scalar_fluxes = self.transport.evaluate(
                time_,
                scalars,
                velocity,
                args,
                sgs_diffusivities=scalar_sgs_diffusivities,
            )
        else:
            molecular_viscosity = jnp.full(
                self.transport.layout.cell_shape,
                self.flow_problem.viscosity,
                dtype=self.transport.layout.dtype,
            )
            kinetic_name = self.ksgs.scalar_field_name
            ksgs_state, ksgs_transport = self.ksgs.prepare_transport(
                scalars[kinetic_name],
                molecular_viscosity,
                continuation_state=ksgs_state,
            )
            scalar_sgs_diffusivities = self.scalar_sgs.diffusivities(
                ksgs_transport.eddy_viscosity
            )
            scalar_sgs_diffusivities[kinetic_name] = (
                self.ksgs.plan.coefficients.diffusion * ksgs_transport.eddy_viscosity
            )
            scalar_fluxes = self.transport.evaluate(
                time_,
                scalars,
                velocity,
                args,
                sgs_diffusivities=scalar_sgs_diffusivities,
            )
            kinetic_flux = scalar_fluxes[kinetic_name]
            ksgs_stage = self.ksgs.evaluate(
                velocity,
                boundary_stage,
                ksgs_state,
                ksgs_transport,
                kinetic_flux.diffusive_divergence,
                molecular_viscosity,
                self._buoyancy_frequency_squared(scalars),
                accept_update=accept_ksgs_update,
            )
            local_source = (
                ksgs_stage.result.contributions.rhs
                - ksgs_stage.result.contributions.diffusion
            )
            kinetic_source = kinetic_flux.source + local_source
            kinetic_rate = kinetic_flux.rate + local_source
            kinetic_finite = (
                kinetic_flux.finite
                & ksgs_stage.finite
                & jnp.all(jnp.isfinite(kinetic_source))
                & jnp.all(jnp.isfinite(kinetic_rate))
            )
            scalar_fluxes[kinetic_name] = eqx.tree_at(
                lambda result: (
                    result.source,
                    result.rate,
                    result.finite,
                    result.success,
                ),
                kinetic_flux,
                (
                    kinetic_source,
                    kinetic_rate,
                    kinetic_finite,
                    kinetic_finite & ksgs_stage.success,
                ),
            )
            ksgs_rate = ksgs_stage.viscosity_result.physical_diffusive_rate
            ksgs_unconstrained = self.momentum.boundaries.enforce_rate(
                tuple(
                    base + sgs
                    for base, sgs in zip(
                        base_components.unconstrained,
                        ksgs_rate,
                        strict=True,
                    )
                ),
                boundary_stage,
            )
            base_components = MACIncompressibleRateComponents(
                convection=base_components.convection,
                molecular=base_components.molecular,
                sgs=ksgs_rate,
                forcing=base_components.forcing,
                unconstrained=ksgs_unconstrained,
                les_stage=None,
                dynamic_les_stage=base_components.dynamic_les_stage,
            )
        buoyancy = self.buoyancy.evaluate(
            velocity,
            scalar_fluxes,
            self.transport,
            self.momentum,
            projection_id=self.projection.plan_id,
        )
        ocean_forcing = (
            None
            if self.ocean_forcing is None
            else self.ocean_forcing.evaluate(time_, velocity, args)
        )
        ocean_force = (
            tuple(jnp.zeros_like(component) for component in velocity)
            if ocean_forcing is None
            else ocean_forcing.force
        )
        unconstrained_base = base_components.unconstrained
        unconstrained = self.momentum.boundaries.homogeneous_rate(
            tuple(
                base + buoyancy_force + ocean_force_component
                for base, buoyancy_force, ocean_force_component in zip(
                    unconstrained_base,
                    buoyancy.force,
                    ocean_force,
                    strict=True,
                )
            )
        )
        projected = self.projection.project_rate(
            unconstrained,
            boundary_stage=boundary_stage,
        )
        projected_rate_finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(component)) for component in projected.rate)
            )
        )
        projected_rate_valid = projected.converged & projected_rate_finite
        projected_rate = tuple(
            jnp.where(projected_rate_valid, component, jnp.zeros_like(component))
            for component in projected.rate
        )
        scalar_rates = {
            name: scalar_fluxes[name].rate for name in self.transport.layout.field_names
        }
        les_finite = (
            jnp.asarray(True)
            if base_components.les_stage is None
            else base_components.les_stage.finite
        )
        les_success = (
            jnp.asarray(True)
            if base_components.les_stage is None
            else base_components.les_stage.successful
        )
        ksgs_finite = jnp.asarray(True) if ksgs_stage is None else ksgs_stage.finite
        ksgs_success = jnp.asarray(True) if ksgs_stage is None else ksgs_stage.success
        finite = (
            boundary_stage.finite
            & les_finite
            & ksgs_finite
            & buoyancy.finite
            & (jnp.asarray(True) if ocean_forcing is None else ocean_forcing.finite)
            & jnp.all(
                jnp.stack(
                    tuple(
                        scalar_fluxes[name].finite
                        for name in self.transport.layout.field_names
                    )
                )
            )
            & projected_rate_finite
            & jnp.all(jnp.isfinite(projected.pressure))
            & jnp.all(jnp.isfinite(projected.pressure_residual))
            & jnp.all(jnp.isfinite(projected.divergence_after))
        )
        success = (
            finite
            & boundary_stage.successful
            & les_success
            & ksgs_success
            & projected.converged
            & buoyancy.success
            & (jnp.asarray(True) if ocean_forcing is None else ocean_forcing.success)
        )
        return MACScalarBuoyancyStage(
            boundary_stage=boundary_stage,
            velocity=velocity,
            scalars=scalars,
            momentum_components=base_components,
            scalar_sgs_diffusivities=scalar_sgs_diffusivities,
            scalar_fluxes=scalar_fluxes,
            ksgs=ksgs_stage,
            buoyancy=buoyancy,
            ocean_forcing=ocean_forcing,
            unconstrained_velocity_rate=unconstrained,
            velocity_rate=projected_rate,
            scalar_rates=scalar_rates,
            pressure=projected.pressure,
            pressure_residual=projected.pressure_residual,
            divergence_before=projected.divergence_before,
            divergence_after=projected.divergence_after,
            projection_converged=projected.converged,
            finite=finite,
            success=success,
            compilation_id=self.compilation_id,
            momentum_id=self.momentum.prepared_id,
            transport_id=self.transport.prepared_id,
            projection_id=self.projection.plan_id,
            stage_id=canonical_fingerprint(
                {
                    "kind": "mac-scalar-buoyancy-stage",
                    "compilation": self.compilation_id,
                }
            ),
        )

    def pressure_field(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        stage = self.stage(time, state, args)
        return eqx.error_if(
            stage.pressure,
            ~stage.success,
            "Coupled MAC pressure recovery failed.",
        )

    def _stratification_step(self, scalars: Mapping[str, Array], /) -> Array:
        vertical_axis = self.buoyancy.principal_gravity_axis
        if vertical_axis is None:
            return jnp.asarray(
                math.inf,
                dtype=self.momentum.operators.pressure_space.dtype,
            )
        centers = self.momentum.operators.discretization.grid.structured_axes[
            vertical_axis
        ].interval_centers
        if centers.size < 2:
            return jnp.asarray(
                math.inf,
                dtype=self.momentum.operators.pressure_space.dtype,
            )
        density_anomaly = jnp.zeros(
            self.momentum.operators.discretization.cell_shape,
            dtype=self.momentum.operators.pressure_space.dtype,
        )
        for name, coefficient, reference in zip(
            self.buoyancy.field_names,
            self.buoyancy.coefficients,
            self.buoyancy.references,
            strict=True,
        ):
            density_anomaly = density_anomaly + coefficient * (scalars[name] - reference)
        moved = jnp.moveaxis(density_anomaly, vertical_axis, 0)
        widths = centers[1:] - centers[:-1]
        shape = (widths.size,) + (1,) * (moved.ndim - 1)
        gradient = (moved[1:] - moved[:-1]) / widths.reshape(shape)
        frequency_squared = jnp.max(
            jnp.maximum(
                self.buoyancy.gravity[vertical_axis] * gradient,
                0.0,
            )
        )
        frequency = jnp.sqrt(frequency_squared)
        safe_frequency = jnp.where(frequency > 0.0, frequency, 1.0)
        return jnp.where(
            frequency > 0.0,
            math.sqrt(3.0) / safe_frequency,
            jnp.inf,
        )

    def _momentum_step_restriction_from_stage(
        self,
        stage: MACScalarBuoyancyStage,
        /,
    ) -> MACLESStepRestriction:
        grid = self.momentum.operators.discretization.grid
        reduction_dtype = jnp.dtype(self.momentum.precision.reduction_dtype)
        inverse_advective = jnp.zeros(
            self.momentum.operators.discretization.cell_shape,
            dtype=reduction_dtype,
        )
        inverse_diffusive = jnp.zeros_like(inverse_advective)
        for axis_index, axis in enumerate(grid.structured_axes):
            component = stage.velocity[axis_index]
            moved = jnp.moveaxis(component, axis_index, 0)
            cell_velocity = (
                0.5 * (moved + jnp.roll(moved, -1, axis=0))
                if axis.periodic
                else 0.5 * (moved[:-1] + moved[1:])
            )
            cell_velocity = jnp.moveaxis(cell_velocity, 0, axis_index)
            shape = [1] * inverse_advective.ndim
            shape[axis_index] = int(axis.interval_widths.size)
            widths = axis.interval_widths.reshape(tuple(shape))
            inverse_advective = inverse_advective + jnp.abs(cell_velocity) / widths
            inverse_diffusive = inverse_diffusive + 2.0 / widths**2
        advective_rate = jnp.max(inverse_advective)
        viscosity = self.flow_problem.viscosity.astype(reduction_dtype)
        molecular_rate = viscosity * jnp.max(inverse_diffusive)
        safe_advective = jnp.where(advective_rate > 0.0, advective_rate, 1.0)
        safe_molecular = jnp.where(molecular_rate > 0.0, molecular_rate, 1.0)
        advective = jnp.where(
            advective_rate > 0.0,
            1.0 / safe_advective,
            jnp.inf,
        )
        molecular = jnp.where(
            molecular_rate > 0.0,
            1.0 / safe_molecular,
            jnp.inf,
        )
        if stage.momentum_components.les_stage is not None:
            prepared = self.base_dynamics.algebraic_les
            if prepared is None:
                raise ValueError("Coupled LES stage has no prepared momentum closure.")
            sgs = prepared.viscosity_action.explicit_step_bound(
                stage.momentum_components.les_stage.model_result.kinematic_viscosity
            )
            sgs_supported = prepared.viscosity_action.restriction_supported
        elif stage.ksgs is not None:
            if self.ksgs is None:
                raise ValueError("Coupled KSGS stage has no prepared closure.")
            sgs = self.ksgs.viscosity_action.explicit_step_bound(
                stage.ksgs.result.eddy_viscosity
            )
            sgs_supported = self.ksgs.viscosity_action.restriction_supported
        else:
            sgs = jnp.asarray(jnp.inf, dtype=reduction_dtype)
            sgs_supported = True
        combined = jnp.minimum(jnp.minimum(advective, molecular), sgs)
        return MACLESStepRestriction(
            advective=self.momentum.precision.reduction(advective),
            molecular=self.momentum.precision.reduction(molecular),
            sgs=self.momentum.precision.reduction(sgs),
            combined=self.momentum.precision.reduction(combined),
            sgs_supported=sgs_supported,
        )

    def step_restriction(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACScalarBuoyancyStepRestriction:
        value = self.validate_state(state)
        if self.scalar_sgs is None:
            velocity, scalar_fields = self.unpack_state(value)
            momentum = self.base_dynamics.step_restriction(
                time,
                value[: self.velocity_size],
                args,
            )
            scalars = self.transport.step_restriction(velocity)
            ksgs = jnp.asarray(jnp.inf, dtype=value.dtype)
            closure_success = jnp.asarray(True)
        else:
            stage = self.stage(time, value, args)
            scalar_fields = stage.scalars
            momentum = self._momentum_step_restriction_from_stage(stage)
            scalars = self.transport.step_restriction(
                stage.velocity,
                sgs_diffusivities=stage.scalar_sgs_diffusivities,
            )
            if stage.ksgs is None:
                ksgs = jnp.asarray(jnp.inf, dtype=value.dtype)
            else:
                contributions = stage.ksgs.result.contributions
                sink = (
                    contributions.dissipation
                    + contributions.low_re_dissipation
                    + jnp.maximum(-contributions.buoyancy, 0.0)
                )
                safe_sink = jnp.where(sink > 0.0, sink, 1.0)
                positivity_step = jnp.where(
                    sink > 0.0,
                    stage.ksgs.state.kinetic_energy / safe_sink,
                    jnp.inf,
                )
                ksgs = jnp.min(positivity_step)
            closure_success = stage.success & jnp.asarray(momentum.sgs_supported)
        ocean_forcing = (
            jnp.asarray(jnp.inf, dtype=value.dtype)
            if self.ocean_forcing is None
            else self.ocean_forcing.step_restriction()
        )
        stratification = self._stratification_step(scalar_fields)
        selected = jnp.minimum(
            jnp.minimum(momentum.combined, scalars.selected),
            jnp.minimum(ksgs, jnp.minimum(ocean_forcing, stratification)),
        )
        finite = ~jnp.isnan(selected)
        return MACScalarBuoyancyStepRestriction(
            momentum=momentum,
            scalars=scalars,
            ksgs=ksgs,
            ocean_forcing=ocean_forcing,
            stratification=stratification,
            selected=selected,
            finite=finite,
            success=finite & closure_success & scalars.success,
            compilation_id=self.compilation_id,
            momentum_id=self.momentum.prepared_id,
            transport_id=self.transport.prepared_id,
            projection_id=self.projection.plan_id,
        )

    def diagnostics_from_stage(
        self,
        stage: MACScalarBuoyancyStage,
        /,
    ) -> MACScalarBuoyancyDiagnostics:
        if (
            not isinstance(stage, MACScalarBuoyancyStage)
            or stage.compilation_id != self.compilation_id
            or stage.momentum_id != self.momentum.prepared_id
            or stage.transport_id != self.transport.prepared_id
            or stage.projection_id != self.projection.plan_id
        ):
            raise ValueError("Coupled MAC diagnostics stage provenance does not match.")
        components = stage.momentum_components
        scalar_diagnostics = self.transport.diagnostics_from_fluxes(
            stage.scalars,
            stage.scalar_fluxes,
        )
        space = self.momentum.operators.velocity_space
        nonlinear_rate = tuple(-component for component in components.convection)
        kinetic_energy = 0.5 * jnp.real(space.inner(stage.velocity, stage.velocity))
        nonlinear_energy_rate = jnp.real(space.inner(stage.velocity, nonlinear_rate))
        forcing_power = jnp.real(space.inner(stage.velocity, components.forcing))
        ocean_forcing_power = (
            jnp.asarray(0.0, dtype=kinetic_energy.dtype)
            if stage.ocean_forcing is None
            else (
                stage.ocean_forcing.coriolis_power
                + stage.ocean_forcing.surface_stress_power
            )
        )
        viscous_energy_rate = jnp.real(space.inner(stage.velocity, components.molecular))
        sgs_energy_rate = jnp.real(space.inner(stage.velocity, components.sgs))
        semidiscrete_energy_rate = jnp.real(
            space.inner(stage.velocity, stage.velocity_rate)
        )
        momentum_diagnostics = self.momentum.diagnostics(
            stage.velocity,
            stage=stage.boundary_stage,
        )
        viscosity = self.flow_problem.viscosity.astype(
            self.momentum.operators.pressure_space.dtype
        )
        molecular_dissipation = viscosity * momentum_diagnostics.dissipation
        traction_power = self.momentum._boundary_traction_power(
            stage.velocity,
            stage.boundary_stage,
        )
        molecular_boundary_power = (
            viscosity * (momentum_diagnostics.boundary_power - traction_power)
            + traction_power
        )
        les_stage = components.les_stage
        if les_stage is not None:
            sgs_dissipation = les_stage.viscosity_result.integrated_dissipation
            sgs_boundary_power = les_stage.boundary_power
            sgs_energy_transfer = jnp.sum(
                self.momentum.operators.discretization.cell_volumes
                * les_stage.model_result.energy_transfer
            )
            les_finite = les_stage.finite
            les_success = les_stage.successful
        elif stage.ksgs is not None:
            sgs_dissipation = stage.ksgs.viscosity_result.integrated_dissipation
            sgs_boundary_power = stage.ksgs.viscosity_result.boundary_power
            sgs_energy_transfer = -sgs_dissipation
            les_finite = stage.ksgs.finite
            les_success = stage.ksgs.success
        else:
            zero = jnp.asarray(0.0, dtype=kinetic_energy.dtype)
            sgs_dissipation = zero
            sgs_boundary_power = zero
            sgs_energy_transfer = zero
            les_finite = jnp.asarray(True)
            les_success = jnp.asarray(True)
        dissipation = molecular_dissipation + sgs_dissipation
        wall_power = molecular_boundary_power + sgs_boundary_power
        expected = (
            forcing_power
            + ocean_forcing_power
            + stage.buoyancy.total_power
            - dissipation
            + wall_power
            - momentum_diagnostics.open_backflow_dissipation
        )
        volumes = self.momentum.operators.discretization.cell_volumes
        pressure_residual_norm = jnp.sqrt(jnp.sum(volumes * stage.pressure_residual**2))
        divergence_norm = GeometryPrecisionPolicy().norm(
            stage.divergence_after.reshape((-1,))
        )
        pressure_gauge_residual = jnp.abs(
            jnp.sum(volumes * stage.pressure) / jnp.sum(volumes)
        )
        energy_defect = semidiscrete_energy_rate - expected
        finite = (
            stage.finite
            & scalar_diagnostics.finite
            & momentum_diagnostics.finite
            & les_finite
            & jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (
                            kinetic_energy,
                            nonlinear_energy_rate,
                            forcing_power,
                            stage.buoyancy.total_power,
                            ocean_forcing_power,
                            viscous_energy_rate,
                            sgs_energy_rate,
                            sgs_dissipation,
                            sgs_boundary_power,
                            sgs_energy_transfer,
                            dissipation,
                            wall_power,
                            semidiscrete_energy_rate,
                            energy_defect,
                            divergence_norm,
                            pressure_residual_norm,
                            pressure_gauge_residual,
                        )
                    )
                )
            )
        )
        success = (
            finite
            & stage.success
            & scalar_diagnostics.success
            & momentum_diagnostics.successful
            & les_success
        )
        return MACScalarBuoyancyDiagnostics(
            scalars=scalar_diagnostics,
            buoyancy=stage.buoyancy,
            kinetic_energy=kinetic_energy,
            nonlinear_energy_rate=nonlinear_energy_rate,
            forcing_power=forcing_power,
            buoyancy_power=stage.buoyancy.total_power,
            ocean_forcing_power=ocean_forcing_power,
            viscous_energy_rate=viscous_energy_rate,
            sgs_energy_rate=sgs_energy_rate,
            sgs_dissipation=sgs_dissipation,
            sgs_boundary_power=sgs_boundary_power,
            sgs_energy_transfer=sgs_energy_transfer,
            dissipation=dissipation,
            wall_power=wall_power,
            semidiscrete_energy_rate=semidiscrete_energy_rate,
            energy_balance_defect=energy_defect,
            divergence_norm=divergence_norm,
            pressure_residual_norm=pressure_residual_norm,
            pressure_gauge_residual=pressure_gauge_residual,
            projection_converged=stage.projection_converged,
            finite=finite,
            success=success,
            compilation_id=self.compilation_id,
            momentum_id=self.momentum.prepared_id,
            transport_id=self.transport.prepared_id,
            projection_id=self.projection.plan_id,
            grid_id=self.momentum.operators.discretization.grid.prepared_id,
        )

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACScalarBuoyancyDiagnostics:
        return self.diagnostics_from_stage(self.stage(time, state, args))

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        stage = self.stage(time, state, args)
        velocity_rate = self.momentum.operators.velocity_space.flatten(
            stage.velocity_rate
        )
        scalar_rate = self.transport.layout.pack(stage.scalar_rates)
        coordinates = jnp.concatenate((velocity_rate, scalar_rate))
        return eqx.error_if(
            coordinates,
            ~stage.success | jnp.any(~jnp.isfinite(coordinates)),
            "Coupled MAC scalar-buoyancy stage failed.",
        )


def compile_mac_scalar_buoyancy(
    flow_problem: IncompressibleFlowProblem,
    momentum: PreparedMACMomentumOperators,
    projection: MACPressureProjectionPlan,
    scalar_problem: MACScalarProblem,
    transport: PreparedMACScalarTransport,
    buoyancy: MACBuoyancyLaw,
    /,
    *,
    algebraic_les: MACAlgebraicLESPlan | None = None,
    scalar_sgs: MACScalarSGSPlan | None = None,
    ksgs: AbstractKSGSPlan | None = None,
    ksgs_field_name: str | None = None,
    ocean_forcing: PreparedMACOceanForcing | None = None,
) -> CompiledMACScalarBuoyancyDynamics:
    """Compile projected unit-density MAC flow with named explicit scalars."""
    from ..solver._structured_incompressible import MACPressureProjectionPlan

    if not isinstance(flow_problem, IncompressibleFlowProblem):
        raise TypeError("flow_problem must be IncompressibleFlowProblem.")
    if not isinstance(momentum, PreparedMACMomentumOperators):
        raise TypeError("momentum must be PreparedMACMomentumOperators.")
    if not isinstance(projection, MACPressureProjectionPlan):
        raise TypeError("projection must be MACPressureProjectionPlan.")
    if not isinstance(scalar_problem, MACScalarProblem):
        raise TypeError("scalar_problem must be MACScalarProblem.")
    if not isinstance(transport, PreparedMACScalarTransport):
        raise TypeError("transport must be PreparedMACScalarTransport.")
    if not isinstance(buoyancy, MACBuoyancyLaw):
        raise TypeError("buoyancy must be MACBuoyancyLaw.")
    if flow_problem.spatial_dimension != momentum.dimension:
        raise ValueError("Incompressible problem and MAC momentum dimensions differ.")
    if projection.operators.prepared_id != momentum.operators.prepared_id:
        raise ValueError("MAC momentum and pressure projection must share operators.")
    if transport.layout.operators.prepared_id != momentum.operators.prepared_id:
        raise ValueError("MAC scalar transport and momentum must share operators.")
    if transport.problem.problem_id != scalar_problem.problem_id:
        raise ValueError("Prepared MAC scalar transport does not match scalar problem.")
    if tuple(transport.layout.field_names) != tuple(scalar_problem.field_names):
        raise ValueError("MAC scalar problem and prepared layout fields differ.")
    if not set(buoyancy.field_names).issubset(scalar_problem.field_names):
        raise ValueError("MAC buoyancy fields must belong to the scalar problem.")
    if len(buoyancy.gravity) != momentum.dimension:
        raise ValueError("MAC buoyancy gravity and flow dimensions differ.")
    if not np.isclose(projection.density, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("MAC Boussinesq dynamics require unit reference density.")
    if ocean_forcing is not None and (
        not isinstance(ocean_forcing, PreparedMACOceanForcing)
        or ocean_forcing.operators.prepared_id != momentum.operators.prepared_id
    ):
        raise ValueError("Ocean forcing must be prepared on the coupled MAC operators.")
    if algebraic_les is not None and not isinstance(algebraic_les, MACAlgebraicLESPlan):
        raise TypeError("algebraic_les must be MACAlgebraicLESPlan or None.")
    if ksgs is not None and not isinstance(ksgs, AbstractKSGSPlan):
        raise TypeError("ksgs must be AbstractKSGSPlan or None.")
    if algebraic_les is not None and ksgs is not None:
        raise ValueError("Algebraic MAC LES and prognostic KSGS are alternatives.")
    if scalar_sgs is not None and not isinstance(scalar_sgs, MACScalarSGSPlan):
        raise TypeError("scalar_sgs must be MACScalarSGSPlan or None.")
    les_active = algebraic_les is not None or ksgs is not None
    if les_active != (scalar_sgs is not None):
        raise ValueError(
            "MAC Boussinesq LES requires an explicit complete scalar SGS "
            "declaration; no turbulent Prandtl or Schmidt number is defaulted."
        )
    if ksgs is None and ksgs_field_name is not None:
        raise ValueError("ksgs_field_name is valid only with a KSGS plan.")
    kinetic_name = None if ksgs_field_name is None else str(ksgs_field_name)
    if ksgs is not None and (
        not kinetic_name or kinetic_name not in scalar_problem.field_names
    ):
        raise ValueError(
            "Prognostic MAC KSGS requires an explicit transported ksgs_field_name."
        )
    if kinetic_name is not None:
        declarations = {
            declaration.name: declaration for declaration in scalar_problem.transports
        }
        kinetic_declaration = declarations[kinetic_name]
        diffusivity = np.asarray(kinetic_declaration.diffusivity)
        if (
            diffusivity.shape != ()
            or not np.isclose(
                float(diffusivity),
                float(flow_problem.viscosity),
                rtol=0.0,
                atol=0.0,
            )
            or kinetic_declaration.advection != "upwind"
            or kinetic_declaration.source is not None
        ):
            raise ValueError(
                "The prognostic KSGS scalar must use upwind advection, molecular "
                "diffusivity equal to flow viscosity, and no independent source."
            )
        for lower, upper in transport.boundaries.field_conditions(kinetic_name):
            for condition in (lower, upper):
                no_flux = (
                    condition.kind == "neumann"
                    and condition.function is None
                    and bool(jnp.all(condition.value == 0.0))
                )
                if condition.kind != "periodic" and not no_flux:
                    raise ValueError(
                        "Prognostic MAC KSGS supports only periodic or impermeable "
                        "zero-flux scalar boundaries."
                    )
    base = compile_mac_incompressible_flow(
        flow_problem,
        momentum,
        projection,
        algebraic_les=algebraic_les,
    )
    scalar_sgs_fields = tuple(
        name for name in transport.layout.field_names if name != kinetic_name
    )
    prepared_scalar_sgs = (
        None
        if scalar_sgs is None
        else scalar_sgs.prepare(transport, field_names=scalar_sgs_fields)
    )
    prepared_ksgs = (
        None
        if ksgs is None or kinetic_name is None
        else PreparedMACKSGS(ksgs, momentum, kinetic_name)
    )
    identifier = canonical_fingerprint(
        {
            "kind": "compiled-mac-scalar-buoyancy",
            "flow_problem": flow_problem.problem_id,
            "scalar_problem": scalar_problem.problem_id,
            "momentum": momentum.prepared_id,
            "projection": projection.plan_id,
            "transport": transport.prepared_id,
            "algebraic_les": (
                "none" if base.algebraic_les is None else base.algebraic_les.prepared_id
            ),
            "scalar_sgs": (
                "none" if prepared_scalar_sgs is None else prepared_scalar_sgs.prepared_id
            ),
            "ksgs": "none" if prepared_ksgs is None else prepared_ksgs.prepared_id,
            "buoyancy": buoyancy.law_id,
            "ocean_forcing": ("none" if ocean_forcing is None else ocean_forcing.plan_id),
        }
    )
    return CompiledMACScalarBuoyancyDynamics(
        flow_problem,
        scalar_problem,
        momentum,
        projection,
        transport,
        buoyancy,
        base,
        prepared_scalar_sgs,
        prepared_ksgs,
        ocean_forcing,
        compilation_id=identifier,
    )


__all__ = [
    "CompiledMACScalarBuoyancyDynamics",
    "MACBuoyancyLaw",
    "MACBuoyancyLedger",
    "MACKSGSStageResult",
    "MACScalarBuoyancyDiagnostics",
    "MACScalarBuoyancyStage",
    "MACScalarBuoyancyStepRestriction",
    "PreparedMACKSGS",
    "compile_mac_scalar_buoyancy",
]
