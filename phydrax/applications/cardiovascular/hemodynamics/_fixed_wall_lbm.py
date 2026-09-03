#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.lattice_boltzmann import (
    compile_staged_lattice_boltzmann_boundary,
    FixedSDFLinkGeometry,
    LatticeBoltzmannBodyBoundary,
    LatticeBoltzmannBoundaryParameters,
    LatticeBoltzmannBoundaryState,
    LatticeBoltzmannCornerRule,
    LatticeBoltzmannDiscretization,
    LatticeBoltzmannFaceBoundary,
    LatticeBoltzmannLinkOwner,
    LatticeBoltzmannMethodPlan,
    LatticeBoltzmannRuntimeParameters,
    PreparedLatticeBoltzmannDynamics,
    PreparedLatticeBoltzmannMethodPlan,
    PreparedStagedLatticeBoltzmannBoundary,
    TRTCollisionPlan,
)
from ....discretization.lattice_boltzmann._collision import macroscopic_raw_moments
from ._domain import (
    FixedWallLumenRegion,
    FixedWallScope,
    HemodynamicsEvidence,
    HemodynamicsScaling,
    HemodynamicsStatus,
    HemodynamicsValidityLimits,
)
from ._ports import (
    FlowTerminalPort,
    prepare_terminal_measurements,
    PreparedTerminalMeasurements,
    PressureTerminalPort,
    terminal_balance_evidence,
    TerminalBalanceEvidence,
    TerminalDirection,
    TerminalMeasurements,
    TerminalPort,
    TerminalPortValues,
)
from ._rheology import CarreauYasudaRheology, NewtonianRheology, RheologyModel


class FixedWallMacroscopicState(StrictModule):
    """Physical fields reconstructed from one fixed-shape population state."""

    density_mg_per_mm3: Array
    velocity_mm_per_ms: Array
    gauge_pressure_kpa: Array
    shear_rate_per_ms: Array
    dynamic_viscosity_kpa_ms: Array


class FixedWallLBMState(StrictModule):
    """Accepted state; geometry and array shapes never change after preparation."""

    populations: Array
    boundary_state: LatticeBoltzmannBoundaryState
    step_index: Array
    time_ms: Array
    initial_mass_mg: Array
    total_mass_mg: Array
    total_momentum_mg_mm_per_ms: Array
    cumulative_outward_volume_mm3: Array
    prepared_id: str = eqx.field(static=True)


class FixedWallLBMCandidate(StrictModule):
    """Uncommitted state proposal with mandatory admissibility evidence."""

    state: FixedWallLBMState
    macroscopic: FixedWallMacroscopicState
    terminals: TerminalMeasurements
    terminal_balance: TerminalBalanceEvidence
    evidence: HemodynamicsEvidence
    prepared_id: str = eqx.field(static=True)


class FixedWallLBMCheckpoint(StrictModule):
    """In-memory checkpoint payload bound to one immutable prepared topology."""

    state: FixedWallLBMState
    prepared_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)


class FixedWallLBMAdvance(StrictModule):
    """Candidate plus the fail-closed committed state selected from it."""

    candidate: FixedWallLBMCandidate
    committed_state: FixedWallLBMState


def _rheology_bounds(model: RheologyModel, density: float, /) -> tuple[float, float]:
    return (
        model.minimum_dynamic_viscosity_kpa_ms / density,
        model.maximum_dynamic_viscosity_kpa_ms / density,
    )


def _safe_norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


def _masked_derivative(
    field: Array,
    fluid_mask: Array,
    axis: int,
    cell_size: Array,
    periodic: bool,
    /,
) -> Array:
    """Centered derivative with one-sided values beside voxel walls."""

    forward = jnp.roll(field, -1, axis=axis)
    backward = jnp.roll(field, 1, axis=axis)
    forward_fluid = jnp.roll(fluid_mask, -1, axis=axis)
    backward_fluid = jnp.roll(fluid_mask, 1, axis=axis)
    if not periodic:
        lower: list[object] = [slice(None)] * fluid_mask.ndim
        upper: list[object] = [slice(None)] * fluid_mask.ndim
        lower[axis] = 0
        upper[axis] = -1
        backward_fluid = backward_fluid.at[tuple(lower)].set(False)
        forward_fluid = forward_fluid.at[tuple(upper)].set(False)
    centered = (forward - backward) / (2.0 * cell_size)
    forward_only = (forward - field) / cell_size
    backward_only = (field - backward) / cell_size
    derivative = jnp.where(
        forward_fluid & backward_fluid,
        centered,
        jnp.where(
            forward_fluid,
            forward_only,
            jnp.where(backward_fluid, backward_only, 0.0),
        ),
    )
    return jnp.where(fluid_mask, derivative, 0.0)


def _shear_rate(
    velocity: Array,
    fluid_mask: Array,
    cell_size: Array,
    periodic: tuple[bool, bool, bool],
    /,
) -> Array:
    gradients = jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    _masked_derivative(
                        velocity[..., component],
                        fluid_mask,
                        axis,
                        cell_size,
                        periodic[axis],
                    )
                    for axis in range(3)
                ),
                axis=-1,
            )
            for component in range(3)
        ),
        axis=-2,
    )
    rate_of_deformation = 0.5 * (gradients + jnp.swapaxes(gradients, -1, -2))
    second_invariant = oe.contract(
        "...ij,...ij->...", rate_of_deformation, rate_of_deformation
    )
    return jnp.where(fluid_mask, jnp.sqrt(jnp.maximum(2.0 * second_invariant, 0.0)), 0.0)


def _status(
    finite: Array,
    collision: Array,
    port_iterate: Array,
    low_mach: Array,
    mass: Array,
    momentum: Array,
    populations: Array,
    density: Array,
    rheology: Array,
    terminal: Array,
    /,
) -> Array:
    value = jnp.asarray(HemodynamicsStatus.SUCCESS, dtype=jnp.int32)
    value = jnp.where(~collision, HemodynamicsStatus.COLLISION_FAILURE, value)
    value = jnp.where(~terminal, HemodynamicsStatus.TERMINAL_BALANCE_VIOLATION, value)
    value = jnp.where(~rheology, HemodynamicsStatus.RHEOLOGY_INVALID, value)
    value = jnp.where(~density, HemodynamicsStatus.DENSITY_INADMISSIBLE, value)
    value = jnp.where(~populations, HemodynamicsStatus.POPULATION_INADMISSIBLE, value)
    value = jnp.where(~momentum, HemodynamicsStatus.MOMENTUM_VIOLATION, value)
    value = jnp.where(~mass, HemodynamicsStatus.MASS_BALANCE_VIOLATION, value)
    value = jnp.where(~low_mach, HemodynamicsStatus.LOW_MACH_VIOLATION, value)
    value = jnp.where(~port_iterate, HemodynamicsStatus.PORT_ITERATE_INVALID, value)
    return jnp.where(~finite, HemodynamicsStatus.NONFINITE, value)


class FixedWallLBMPlan(StrictModule, NonTrainableState):
    """Plan a stationary-voxel D3Q19 blood-flow realization.

    The plan reuses PhydraX's lattice, collision, staged-boundary, precision,
    and circulation DAE ports.  It introduces no second mesh, solver, or 0D
    network implementation.
    """

    discretization: LatticeBoltzmannDiscretization
    scaling: HemodynamicsScaling
    lumen: FixedWallLumenRegion
    terminals: tuple[TerminalPort, ...]
    rheology: RheologyModel
    limits: HemodynamicsValidityLimits
    trt_magic_parameter: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        scaling: HemodynamicsScaling,
        lumen: FixedWallLumenRegion,
        terminals: tuple[TerminalPort, ...],
        rheology: RheologyModel,
        /,
        *,
        limits: HemodynamicsValidityLimits | None = None,
        trt_magic_parameter: float = 3.0 / 16.0,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be LatticeBoltzmannDiscretization.")
        if discretization.velocity_set.name != "D3Q19":
            raise ValueError(
                "Fixed-wall cardiovascular LBM requires the certified D3Q19 lattice."
            )
        if not isinstance(scaling, HemodynamicsScaling):
            raise TypeError("scaling must be HemodynamicsScaling.")
        if not isinstance(lumen, FixedWallLumenRegion):
            raise TypeError("lumen must be FixedWallLumenRegion.")
        if lumen.shape != discretization.grid.shape:
            raise ValueError("Lumen mask and lattice shapes do not match.")
        if not np.isclose(
            float(discretization.cell_size),
            float(scaling.cell_size_mm),
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError("LBM cell size and cardiovascular scaling cell size differ.")
        ports = tuple(terminals)
        if not ports or any(
            not isinstance(value, (PressureTerminalPort, FlowTerminalPort))
            for value in ports
        ):
            raise ValueError("Fixed-wall LBM requires typed circulation terminal ports.")
        if any(
            isinstance(value, PressureTerminalPort)
            and value.face.direction is TerminalDirection.INTO_LUMEN
            for value in ports
        ):
            raise ValueError(
                "Pressure-controlled inflow is unsupported; prescribe inlet flow instead."
            )
        if not isinstance(rheology, (NewtonianRheology, CarreauYasudaRheology)):
            raise TypeError("rheology must be a supported cardiovascular rheology model.")
        limits_ = HemodynamicsValidityLimits() if limits is None else limits
        if not isinstance(limits_, HemodynamicsValidityLimits):
            raise TypeError("limits must be HemodynamicsValidityLimits.")
        if scaling.maximum_lattice_mach > limits_.maximum_lattice_mach:
            raise ValueError(
                "Scaling Mach envelope exceeds the workflow acceptance limit."
            )
        magic = float(trt_magic_parameter)
        if not np.isfinite(magic) or magic <= 0.0:
            raise ValueError("trt_magic_parameter must be finite and positive.")

        density = float(scaling.reference_density_mg_per_mm3)
        viscosity_minimum, viscosity_maximum = _rheology_bounds(rheology, density)
        rate_maximum = float(scaling.lattice.relaxation_rate(viscosity_minimum))
        rate_minimum = float(scaling.lattice.relaxation_rate(viscosity_maximum))
        if (
            rate_minimum < limits_.minimum_relaxation_rate
            or rate_maximum > limits_.maximum_relaxation_rate
        ):
            raise ValueError(
                "Rheology and scaling produce relaxation rates outside the declared LBM envelope."
            )
        self.discretization = discretization
        self.scaling = scaling
        self.lumen = lumen
        self.terminals = ports
        self.rheology = rheology
        self.limits = limits_
        self.trt_magic_parameter = magic
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-fixed-wall-d3q19-lbm-plan",
                "discretization": discretization.prepared_id,
                "scaling": scaling.scaling_id,
                "lumen": lumen.lumen_id,
                "terminals": tuple(value.port_id for value in ports),
                "rheology": rheology.rheology_id,
                "limits": limits_.limits_id,
                "collision": "trt",
                "trt_magic_parameter": magic,
            }
        )

    def prepare(self) -> "PreparedFixedWallLBM":
        measurements = prepare_terminal_measurements(
            self.discretization, self.lumen, self.terminals
        )
        mask = np.asarray(self.lumen.fluid_mask, dtype=bool)
        cell_size = float(self.discretization.cell_size)
        signed_distance = np.where(mask, 0.5 * cell_size, -0.5 * cell_size)
        has_solid = bool(np.any(~mask))
        geometry = FixedSDFLinkGeometry(
            self.discretization,
            signed_distance,
            body_names=("fixed-lumen-wall",) if has_solid else (),
        )
        faces = tuple(
            LatticeBoltzmannFaceBoundary(
                terminal.face.axis,
                terminal.face.side,
                (
                    LatticeBoltzmannLinkOwner.PRESSURE
                    if isinstance(terminal, PressureTerminalPort)
                    else LatticeBoltzmannLinkOwner.VELOCITY
                ),
                parameter_id=terminal.terminal_id,
                flow_direction="any",
            )
            for terminal in self.terminals
        )
        open_faces = {
            (terminal.face.axis, terminal.face.side): terminal
            for terminal in self.terminals
        }
        axis_names = self.discretization.grid.axis_names
        corner_rules = []
        for first_axis in range(3):
            if self.discretization.periodic[first_axis]:
                continue
            for second_axis in range(first_axis + 1, 3):
                if self.discretization.periodic[second_axis]:
                    continue
                for first_side in ("lower", "upper"):
                    for second_side in ("lower", "upper"):
                        first_face = (axis_names[first_axis], first_side)
                        second_face = (axis_names[second_axis], second_side)
                        first_open = first_face in open_faces
                        second_open = second_face in open_faces
                        if not first_open and not second_open:
                            continue
                        if first_open and second_open:
                            first_terminal = open_faces[first_face]
                            second_terminal = open_faces[second_face]
                            selected = (
                                first_face
                                if first_terminal.terminal_id
                                < second_terminal.terminal_id
                                else second_face
                            )
                        else:
                            selected = first_face if first_open else second_face
                        corner_rules.append(
                            LatticeBoltzmannCornerRule(
                                (first_face, second_face), selected
                            )
                        )
        boundary_plan = compile_staged_lattice_boltzmann_boundary(
            self.discretization,
            faces=faces,
            geometry=geometry,
            body_boundaries=(
                (
                    LatticeBoltzmannBodyBoundary(
                        "fixed-lumen-wall", LatticeBoltzmannLinkOwner.HALFWAY
                    ),
                )
                if has_solid
                else ()
            ),
            corner_rules=tuple(corner_rules),
        )
        boundary = boundary_plan.prepare(self.discretization)
        method_plan = LatticeBoltzmannMethodPlan(
            TRTCollisionPlan(self.trt_magic_parameter)
        )
        method = method_plan.prepare(
            self.discretization.velocity_set, self.discretization.precision
        )
        initializer = PreparedLatticeBoltzmannDynamics(
            self.discretization,
            self.scaling.lattice,
            method,
            boundary,
        )
        terminal_index = {
            terminal.terminal_id: index for index, terminal in enumerate(self.terminals)
        }
        velocity_terminal_indices = tuple(
            terminal_index[value] for value in boundary.velocity_parameter_ids
        )
        pressure_terminal_indices = tuple(
            terminal_index[value] for value in boundary.pressure_parameter_ids
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-fixed-wall-d3q19-lbm",
                "plan": self.plan_id,
                "method": method.method_id,
                "boundary": boundary.boundary_id,
                "measurements": measurements.prepared_id,
                "scope": FixedWallScope().scope_id,
            }
        )
        return PreparedFixedWallLBM(
            self,
            method,
            boundary,
            initializer,
            measurements,
            velocity_terminal_indices,
            pressure_terminal_indices,
            FixedWallScope(),
            prepared_id,
        )


class PreparedFixedWallLBM(StrictModule, NonTrainableState):
    """Prepared fixed-topology local-rheology collide/route workflow."""

    plan: FixedWallLBMPlan
    method: PreparedLatticeBoltzmannMethodPlan
    boundary: PreparedStagedLatticeBoltzmannBoundary
    initializer: PreparedLatticeBoltzmannDynamics
    terminal_measurements: PreparedTerminalMeasurements
    velocity_terminal_indices: tuple[int, ...] = eqx.field(static=True)
    pressure_terminal_indices: tuple[int, ...] = eqx.field(static=True)
    scope: FixedWallScope
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FixedWallLBMPlan,
        method: PreparedLatticeBoltzmannMethodPlan,
        boundary: PreparedStagedLatticeBoltzmannBoundary,
        initializer: PreparedLatticeBoltzmannDynamics,
        terminal_measurements: PreparedTerminalMeasurements,
        velocity_terminal_indices: tuple[int, ...],
        pressure_terminal_indices: tuple[int, ...],
        scope: FixedWallScope,
        prepared_id: str,
        /,
    ):
        self.plan = plan
        self.method = method
        self.boundary = boundary
        self.initializer = initializer
        self.terminal_measurements = terminal_measurements
        self.velocity_terminal_indices = tuple(velocity_terminal_indices)
        self.pressure_terminal_indices = tuple(pressure_terminal_indices)
        self.scope = scope
        self.prepared_id = str(prepared_id)

    @property
    def discretization(self) -> LatticeBoltzmannDiscretization:
        return self.plan.discretization

    @property
    def scaling(self) -> HemodynamicsScaling:
        return self.plan.scaling

    @property
    def rheology(self) -> RheologyModel:
        return self.plan.rheology

    def _raw_fields(self, populations: Array, /) -> tuple[Array, Array]:
        density, momentum = macroscopic_raw_moments(
            populations,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        safe_density = jnp.where(density > 0.0, density, 1.0)
        velocity = momentum / safe_density[..., None]
        return density, velocity

    def _macroscopic_from_populations(
        self, populations: Array, /
    ) -> FixedWallMacroscopicState:
        density_lattice, velocity_lattice = self._raw_fields(populations)
        density = self.scaling.physical_density(density_lattice)
        velocity = self.scaling.physical_velocity(velocity_lattice)
        pressure = self.scaling.density_gauge_pressure(density_lattice)
        fluid = self.boundary.geometry.fluid_mask
        shear = _shear_rate(
            velocity,
            fluid,
            self.scaling.cell_size_mm.astype(velocity.dtype),
            self.discretization.periodic,
        )
        rheology = self.rheology.evaluate(shear)
        return FixedWallMacroscopicState(
            jnp.where(fluid, density, 0.0),
            jnp.where(fluid[..., None], velocity, 0.0),
            jnp.where(fluid, pressure, 0.0),
            shear,
            jnp.where(fluid, rheology.dynamic_viscosity_kpa_ms, 0.0),
        )

    def macroscopic_state(self, state: FixedWallLBMState, /) -> FixedWallMacroscopicState:
        self._validate_state(state)
        return self._macroscopic_from_populations(state.populations)

    def _mass_and_momentum(self, populations: Array, /) -> tuple[Array, Array]:
        density, velocity = self._raw_fields(populations)
        fluid = self.boundary.geometry.fluid_mask
        mass_lattice = jnp.sum(jnp.where(fluid, density, 0.0))
        momentum_lattice = jnp.sum(
            jnp.where(fluid[..., None], density[..., None] * velocity, 0.0),
            axis=(0, 1, 2),
        )
        return (
            self.scaling.physical_mass(mass_lattice),
            self.scaling.physical_momentum(momentum_lattice),
        )

    def initialize_state(
        self,
        /,
        *,
        density_mg_per_mm3: ArrayLike | None = None,
        velocity_mm_per_ms: ArrayLike = (0.0, 0.0, 0.0),
        time_ms: ArrayLike = 0.0,
    ) -> FixedWallLBMState:
        density = (
            self.scaling.reference_density_mg_per_mm3
            if density_mg_per_mm3 is None
            else jnp.asarray(density_mg_per_mm3)
        )
        representative_viscosity = (
            self.rheology.maximum_dynamic_viscosity_kpa_ms
            / self.scaling.reference_density_mg_per_mm3
        )
        parameters = LatticeBoltzmannRuntimeParameters(
            jnp.asarray(representative_viscosity, dtype=jnp.float64)
        )
        populations = self.initializer.initialize_state(
            density,
            velocity_mm_per_ms,
            parameters,
            time=time_ms,
        )
        mass, momentum = self._mass_and_momentum(populations)
        dtype = populations.dtype
        return FixedWallLBMState(
            populations,
            self.boundary.initial_state(populations),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(time_ms, dtype=dtype).reshape(()),
            mass,
            mass,
            momentum,
            jnp.zeros((self.terminal_measurements.terminal_count,), dtype=dtype),
            self.prepared_id,
        )

    def _validate_state(self, state: FixedWallLBMState, /) -> None:
        if not isinstance(state, FixedWallLBMState):
            raise TypeError("state must be FixedWallLBMState.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("State belongs to another prepared fixed-wall workflow.")
        self.discretization.validate_populations(state.populations)
        if state.step_index.shape != () or state.time_ms.shape != ():
            raise ValueError("State step and time must be scalar arrays.")
        if state.cumulative_outward_volume_mm3.shape != (
            self.terminal_measurements.terminal_count,
        ):
            raise ValueError("State terminal accumulator shape is invalid.")

    def _safe_port_values(
        self,
        values: TerminalPortValues,
        dtype,
        /,
    ) -> tuple[TerminalPortValues, Array]:
        """Replace invalid boundary controls while retaining rejection evidence."""

        pressure = values.pressure_kpa.astype(dtype)
        flow = values.directed_flow_mm3_per_ms.astype(dtype)
        admissible = jnp.isfinite(pressure) & jnp.isfinite(flow)
        safe_pressure = jnp.where(jnp.isfinite(pressure), pressure, 0.0)
        safe_flow = jnp.where(jnp.isfinite(flow), flow, 0.0)
        for terminal_index, terminal in enumerate(self.plan.terminals):
            if isinstance(terminal, PressureTerminalPort):
                gauge = safe_pressure[terminal_index] - jnp.asarray(
                    terminal.pressure_reference_kpa, dtype=dtype
                )
                boundary_density = self.scaling.pressure_density(gauge)
                boundary_valid = (
                    jnp.isfinite(boundary_density)
                    & (boundary_density > 0.0)
                    & (
                        jnp.abs(boundary_density - 1.0)
                        <= self.plan.limits.maximum_relative_density_deviation
                    )
                )
                admissible = admissible.at[terminal_index].set(
                    admissible[terminal_index] & boundary_valid
                )
                safe_pressure = safe_pressure.at[terminal_index].set(
                    jnp.where(
                        boundary_valid,
                        safe_pressure[terminal_index],
                        jnp.asarray(terminal.pressure_reference_kpa, dtype=dtype),
                    )
                )
            else:
                area = self.terminal_measurements.flow_definitions[
                    terminal_index
                ].total_area_mm2.astype(dtype)
                outward_normal = self.terminal_measurements.flow_definitions[
                    terminal_index
                ].outward_normal.astype(dtype)
                physical_velocity = (
                    terminal.face.direction.outward_sign
                    * safe_flow[terminal_index]
                    * outward_normal
                    / area
                )
                lattice_velocity = self.scaling.lattice_velocity(physical_velocity)
                speed_squared = jnp.sum(lattice_velocity * lattice_velocity)
                lattice_mach = jnp.sqrt(speed_squared) / jnp.sqrt(
                    self.discretization.velocity_set.sound_speed_squared.astype(dtype)
                )
                boundary_valid = (
                    jnp.all(jnp.isfinite(lattice_velocity))
                    & (speed_squared < 1.0)
                    & (lattice_mach <= self.plan.limits.maximum_lattice_mach)
                )
                admissible = admissible.at[terminal_index].set(
                    admissible[terminal_index] & boundary_valid
                )
                safe_flow = safe_flow.at[terminal_index].set(
                    jnp.where(boundary_valid, safe_flow[terminal_index], 0.0)
                )
        return TerminalPortValues(safe_pressure, safe_flow), jnp.all(admissible)

    def _boundary_parameters(
        self,
        values: TerminalPortValues,
        dtype,
        /,
    ) -> LatticeBoltzmannBoundaryParameters:
        values = self.terminal_measurements.validate_values(values)
        velocity_targets = []
        for terminal_index in self.velocity_terminal_indices:
            terminal = self.plan.terminals[terminal_index]
            if not isinstance(terminal, FlowTerminalPort):
                raise RuntimeError(
                    "Velocity boundary order contains a pressure terminal."
                )
            area = self.terminal_measurements.flow_definitions[
                terminal_index
            ].total_area_mm2.astype(dtype)
            outward_normal = self.terminal_measurements.flow_definitions[
                terminal_index
            ].outward_normal.astype(dtype)
            directed_flow = values.directed_flow_mm3_per_ms[terminal_index].astype(dtype)
            physical_velocity = (
                terminal.face.direction.outward_sign
                * directed_flow
                * outward_normal
                / area
            )
            velocity_targets.append(self.scaling.lattice_velocity(physical_velocity))
        pressure_densities = []
        for terminal_index in self.pressure_terminal_indices:
            terminal = self.plan.terminals[terminal_index]
            if not isinstance(terminal, PressureTerminalPort):
                raise RuntimeError("Pressure boundary order contains a flow terminal.")
            pressure = values.pressure_kpa[terminal_index].astype(dtype) - jnp.asarray(
                terminal.pressure_reference_kpa, dtype=dtype
            )
            pressure_densities.append(self.scaling.pressure_density(pressure))
        dimension = 3
        body_count = len(self.boundary.body_ids)
        grid_shape = self.discretization.grid.shape
        return LatticeBoltzmannBoundaryParameters(
            velocity_targets=(
                jnp.stack(tuple(velocity_targets))
                if velocity_targets
                else jnp.zeros((0, dimension), dtype=dtype)
            ),
            pressure_densities=(
                jnp.stack(tuple(pressure_densities))
                if pressure_densities
                else jnp.zeros((0,), dtype=dtype)
            ),
            pressure_tangential_velocities=jnp.zeros(
                (len(pressure_densities), dimension), dtype=dtype
            ),
            half_force_density=jnp.zeros(grid_shape + (dimension,), dtype=dtype),
            body_centers=jnp.zeros((body_count, dimension), dtype=dtype),
            body_linear_velocities=jnp.zeros((body_count, dimension), dtype=dtype),
            body_angular_velocities=jnp.zeros((body_count, dimension), dtype=dtype),
            time_step=self.scaling.time_step_ms.astype(dtype),
        )

    def candidate(
        self,
        state: FixedWallLBMState,
        circulation_values: TerminalPortValues,
        /,
    ) -> FixedWallLBMCandidate:
        """Execute or reject one candidate without exposing native boundary errors."""

        self._validate_state(state)
        values = self.terminal_measurements.validate_values(circulation_values)
        safe_values, port_iterate_admissible = self._safe_port_values(
            values, state.populations.dtype
        )
        operands = (state, values, safe_values)
        return jax.lax.cond(
            port_iterate_admissible,
            lambda items: self._candidate_admissible(*items),
            lambda items: self._rejected_port_candidate(items[0], items[1]),
            operands,
        )

    def _candidate_admissible(
        self,
        state: FixedWallLBMState,
        values: TerminalPortValues,
        safe_values: TerminalPortValues,
        /,
    ) -> FixedWallLBMCandidate:
        populations = state.populations
        dtype = populations.dtype
        port_iterate_admissible = jnp.asarray(True)
        fluid = self.boundary.geometry.fluid_mask
        density, velocity_lattice = self._raw_fields(populations)
        velocity_physical = self.scaling.physical_velocity(velocity_lattice)
        shear = _shear_rate(
            velocity_physical,
            fluid,
            self.scaling.cell_size_mm.astype(dtype),
            self.discretization.periodic,
        )
        rheology_evaluation = self.rheology.evaluate(shear)
        reference_density = self.scaling.reference_density_mg_per_mm3.astype(dtype)
        kinematic_viscosity = (
            rheology_evaluation.dynamic_viscosity_kpa_ms / reference_density
        )
        safe_kinematic_viscosity = jnp.where(
            jnp.isfinite(kinematic_viscosity) & (kinematic_viscosity > 0.0),
            kinematic_viscosity,
            (
                self.rheology.maximum_dynamic_viscosity_kpa_ms.astype(dtype)
                / reference_density
            ),
        )
        lattice_viscosity = self.scaling.lattice_kinematic_viscosity(
            safe_kinematic_viscosity
        )
        sound_speed_squared = self.discretization.velocity_set.sound_speed_squared.astype(
            dtype
        )
        relaxation_rate = 1.0 / (0.5 + lattice_viscosity / sound_speed_squared)
        zero_force = jnp.zeros_like(velocity_lattice)
        collision = self.method.collide(
            self.discretization.precision.compute(populations),
            density,
            velocity_lattice,
            zero_force,
            relaxation_rate,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        post_collision = jnp.where(
            fluid[..., None], collision.candidate_populations, populations
        )
        boundary = self.boundary.apply(
            self.discretization.precision.population(post_collision),
            density,
            state.boundary_state,
            self._boundary_parameters(safe_values, dtype),
        )
        candidate_populations = self.discretization.precision.population(
            boundary.populations
        )
        candidate_macroscopic = self._macroscopic_from_populations(candidate_populations)
        candidate_density, candidate_velocity_lattice = self._raw_fields(
            candidate_populations
        )
        candidate_mass, candidate_momentum = self._mass_and_momentum(
            candidate_populations
        )
        terminal_measurements = self.terminal_measurements.measure(
            candidate_macroscopic.gauge_pressure_kpa,
            candidate_macroscopic.velocity_mm_per_ms,
        )
        storage_volume_change = (candidate_mass - state.total_mass_mg) / reference_density
        terminal_balance = terminal_balance_evidence(
            self.terminal_measurements,
            terminal_measurements,
            values,
            storage_volume_change_mm3=storage_volume_change,
            time_step_ms=self.scaling.time_step_ms,
            flow_relative_tolerance=self.plan.limits.maximum_terminal_flow_relative_defect,
            pressure_absolute_tolerance_kpa=(
                self.plan.limits.maximum_terminal_pressure_absolute_defect_kpa
            ),
            volume_relative_tolerance=self.plan.limits.maximum_relative_mass_balance_defect,
            power_relative_tolerance=self.plan.limits.maximum_terminal_power_relative_defect,
        )
        speed_lattice = jnp.sqrt(
            oe.contract(
                "...d,...d->...", candidate_velocity_lattice, candidate_velocity_lattice
            )
        )
        lattice_sound_speed = jnp.sqrt(
            self.discretization.velocity_set.sound_speed_squared.astype(dtype)
        )
        maximum_mach = jnp.max(jnp.where(fluid, speed_lattice / lattice_sound_speed, 0.0))
        relative_density_deviation = jnp.max(
            jnp.where(fluid, jnp.abs(candidate_density - 1.0), 0.0)
        )
        minimum_density = jnp.min(jnp.where(fluid, candidate_density, jnp.inf))
        minimum_population = jnp.min(
            jnp.where(fluid[..., None], candidate_populations, jnp.inf)
        )
        previous_momentum_scale = jnp.maximum(
            _safe_norm(state.total_momentum_mg_mm_per_ms),
            state.total_mass_mg * self.scaling.reference_velocity_mm_per_ms.astype(dtype),
        )
        relative_momentum_change = _safe_norm(
            candidate_momentum - state.total_momentum_mg_mm_per_ms
        ) / jnp.maximum(previous_momentum_scale, jnp.finfo(dtype).tiny)
        finite = (
            jnp.all(jnp.isfinite(candidate_populations))
            & jnp.all(jnp.isfinite(candidate_density))
            & jnp.all(jnp.isfinite(candidate_velocity_lattice))
            & jnp.all(jnp.isfinite(relaxation_rate))
            & jnp.all(jnp.isfinite(values.pressure_kpa))
            & jnp.all(jnp.isfinite(values.directed_flow_mm3_per_ms))
            & terminal_balance.finite
        )
        low_mach = maximum_mach <= self.plan.limits.maximum_lattice_mach
        mass_conservative = terminal_balance.volume_balanced
        momentum_admissible = (
            relative_momentum_change <= self.plan.limits.maximum_relative_momentum_change
        )
        populations_admissible = minimum_population >= self.plan.limits.minimum_population
        density_admissible = (minimum_density > 0.0) & (
            relative_density_deviation
            <= self.plan.limits.maximum_relative_density_deviation
        )
        rheology_admissible = (
            jnp.all(jnp.where(fluid, rheology_evaluation.admissible, True))
            & (
                jnp.min(jnp.where(fluid, relaxation_rate, jnp.inf))
                >= self.plan.limits.minimum_relaxation_rate
            )
            & (
                jnp.max(jnp.where(fluid, relaxation_rate, -jnp.inf))
                <= self.plan.limits.maximum_relaxation_rate
            )
        )
        terminal_admissible = (
            terminal_balance.flow_balanced
            & jnp.all(terminal_balance.pressure_balanced)
            & terminal_balance.power_balanced
        )
        successful = (
            finite
            & port_iterate_admissible
            & low_mach
            & mass_conservative
            & momentum_admissible
            & populations_admissible
            & density_admissible
            & rheology_admissible
            & terminal_admissible
            & collision.successful
        )
        status = _status(
            finite,
            collision.successful,
            port_iterate_admissible,
            low_mach,
            mass_conservative,
            momentum_admissible,
            populations_admissible,
            density_admissible,
            rheology_admissible,
            terminal_admissible,
        )
        evidence = HemodynamicsEvidence(
            status=status,
            successful=successful,
            finite=finite,
            collision_successful=collision.successful,
            port_iterate_admissible=port_iterate_admissible,
            low_mach=low_mach,
            mass_conservative=mass_conservative,
            momentum_admissible=momentum_admissible,
            populations_admissible=populations_admissible,
            density_admissible=density_admissible,
            rheology_admissible=rheology_admissible,
            terminal_balance_admissible=terminal_admissible,
            maximum_lattice_mach=maximum_mach,
            relative_mass_balance_defect=terminal_balance.volume_relative_defect,
            relative_momentum_change=relative_momentum_change,
            minimum_population=minimum_population,
            minimum_density_lattice=minimum_density,
            maximum_relative_density_deviation=relative_density_deviation,
            minimum_relaxation_rate=jnp.min(jnp.where(fluid, relaxation_rate, jnp.inf)),
            maximum_relaxation_rate=jnp.max(jnp.where(fluid, relaxation_rate, -jnp.inf)),
            terminal_flow_relative_defect=terminal_balance.flow_relative_defect,
            terminal_pressure_maximum_absolute_defect_kpa=jnp.max(
                jnp.abs(terminal_balance.pressure_residual_kpa)
            ),
            terminal_power_relative_defect=terminal_balance.power_relative_defect,
            wall_impulse_lattice=boundary.ledger.fluid_impulse,
            scope_id=self.scope.scope_id,
        )
        candidate_state = FixedWallLBMState(
            candidate_populations,
            boundary.state,
            state.step_index + jnp.asarray(1, dtype=jnp.int32),
            state.time_ms + self.scaling.time_step_ms.astype(dtype),
            state.initial_mass_mg,
            candidate_mass,
            candidate_momentum,
            state.cumulative_outward_volume_mm3
            + terminal_measurements.outward_flow_mm3_per_ms
            * self.scaling.time_step_ms.astype(dtype),
            self.prepared_id,
        )
        return FixedWallLBMCandidate(
            candidate_state,
            candidate_macroscopic,
            terminal_measurements,
            terminal_balance,
            evidence,
            self.prepared_id,
        )

    def _rejected_port_candidate(
        self,
        state: FixedWallLBMState,
        values: TerminalPortValues,
        /,
    ) -> FixedWallLBMCandidate:
        """Produce evidence from accepted fields without executing a boundary kernel."""

        populations = state.populations
        dtype = populations.dtype
        fluid = self.boundary.geometry.fluid_mask
        macroscopic = self._macroscopic_from_populations(populations)
        density, velocity_lattice = self._raw_fields(populations)
        terminal_measurements = self.terminal_measurements.measure(
            macroscopic.gauge_pressure_kpa,
            macroscopic.velocity_mm_per_ms,
        )
        terminal_balance = terminal_balance_evidence(
            self.terminal_measurements,
            terminal_measurements,
            values,
            storage_volume_change_mm3=jnp.asarray(0.0, dtype=dtype),
            time_step_ms=self.scaling.time_step_ms,
            flow_relative_tolerance=(
                self.plan.limits.maximum_terminal_flow_relative_defect
            ),
            pressure_absolute_tolerance_kpa=(
                self.plan.limits.maximum_terminal_pressure_absolute_defect_kpa
            ),
            volume_relative_tolerance=(
                self.plan.limits.maximum_relative_mass_balance_defect
            ),
            power_relative_tolerance=(
                self.plan.limits.maximum_terminal_power_relative_defect
            ),
        )
        rheology_evaluation = self.rheology.evaluate(macroscopic.shear_rate_per_ms)
        reference_density = self.scaling.reference_density_mg_per_mm3.astype(dtype)
        kinematic_viscosity = (
            rheology_evaluation.dynamic_viscosity_kpa_ms / reference_density
        )
        safe_viscosity = jnp.where(
            jnp.isfinite(kinematic_viscosity) & (kinematic_viscosity > 0.0),
            kinematic_viscosity,
            self.rheology.maximum_dynamic_viscosity_kpa_ms.astype(dtype)
            / reference_density,
        )
        lattice_viscosity = self.scaling.lattice_kinematic_viscosity(safe_viscosity)
        relaxation_rate = 1.0 / (
            0.5
            + lattice_viscosity
            / self.discretization.velocity_set.sound_speed_squared.astype(dtype)
        )
        speed_lattice = jnp.sqrt(
            oe.contract("...d,...d->...", velocity_lattice, velocity_lattice)
        )
        maximum_mach = jnp.max(
            jnp.where(
                fluid,
                speed_lattice
                / jnp.sqrt(
                    self.discretization.velocity_set.sound_speed_squared.astype(dtype)
                ),
                0.0,
            )
        )
        minimum_population = jnp.min(jnp.where(fluid[..., None], populations, jnp.inf))
        minimum_density = jnp.min(jnp.where(fluid, density, jnp.inf))
        density_deviation = jnp.max(jnp.where(fluid, jnp.abs(density - 1.0), 0.0))
        finite = (
            jnp.all(jnp.isfinite(populations))
            & jnp.all(jnp.isfinite(values.pressure_kpa))
            & jnp.all(jnp.isfinite(values.directed_flow_mm3_per_ms))
            & terminal_balance.finite
        )
        low_mach = maximum_mach <= self.plan.limits.maximum_lattice_mach
        mass_conservative = terminal_balance.volume_balanced
        momentum_admissible = jnp.asarray(True)
        populations_admissible = minimum_population >= self.plan.limits.minimum_population
        density_admissible = (minimum_density > 0.0) & (
            density_deviation <= self.plan.limits.maximum_relative_density_deviation
        )
        minimum_relaxation = jnp.min(jnp.where(fluid, relaxation_rate, jnp.inf))
        maximum_relaxation = jnp.max(jnp.where(fluid, relaxation_rate, -jnp.inf))
        rheology_admissible = (
            jnp.all(jnp.where(fluid, rheology_evaluation.admissible, True))
            & (minimum_relaxation >= self.plan.limits.minimum_relaxation_rate)
            & (maximum_relaxation <= self.plan.limits.maximum_relaxation_rate)
        )
        terminal_admissible = (
            terminal_balance.flow_balanced
            & jnp.all(terminal_balance.pressure_balanced)
            & terminal_balance.power_balanced
        )
        port_iterate_admissible = jnp.asarray(False)
        collision_successful = jnp.asarray(False)
        evidence = HemodynamicsEvidence(
            status=_status(
                finite,
                collision_successful,
                port_iterate_admissible,
                low_mach,
                mass_conservative,
                momentum_admissible,
                populations_admissible,
                density_admissible,
                rheology_admissible,
                terminal_admissible,
            ),
            successful=jnp.asarray(False),
            finite=finite,
            collision_successful=collision_successful,
            port_iterate_admissible=port_iterate_admissible,
            low_mach=low_mach,
            mass_conservative=mass_conservative,
            momentum_admissible=momentum_admissible,
            populations_admissible=populations_admissible,
            density_admissible=density_admissible,
            rheology_admissible=rheology_admissible,
            terminal_balance_admissible=terminal_admissible,
            maximum_lattice_mach=maximum_mach,
            relative_mass_balance_defect=terminal_balance.volume_relative_defect,
            relative_momentum_change=jnp.asarray(0.0, dtype=dtype),
            minimum_population=minimum_population,
            minimum_density_lattice=minimum_density,
            maximum_relative_density_deviation=density_deviation,
            minimum_relaxation_rate=minimum_relaxation,
            maximum_relaxation_rate=maximum_relaxation,
            terminal_flow_relative_defect=terminal_balance.flow_relative_defect,
            terminal_pressure_maximum_absolute_defect_kpa=jnp.max(
                jnp.abs(terminal_balance.pressure_residual_kpa)
            ),
            terminal_power_relative_defect=terminal_balance.power_relative_defect,
            wall_impulse_lattice=jnp.zeros((len(self.boundary.body_ids), 3), dtype=dtype),
            scope_id=self.scope.scope_id,
        )
        return FixedWallLBMCandidate(
            state,
            macroscopic,
            terminal_measurements,
            terminal_balance,
            evidence,
            self.prepared_id,
        )

    def commit(
        self,
        accepted_state: FixedWallLBMState,
        candidate: FixedWallLBMCandidate,
        /,
    ) -> FixedWallLBMState:
        """Commit an admissible candidate; otherwise retain the accepted state."""

        self._validate_state(accepted_state)
        if not isinstance(candidate, FixedWallLBMCandidate):
            raise TypeError("candidate must be FixedWallLBMCandidate.")
        if candidate.prepared_id != self.prepared_id:
            raise ValueError("Candidate belongs to another prepared workflow.")
        selected = candidate.evidence.successful
        proposed = candidate.state
        boundary_state = LatticeBoltzmannBoundaryState(
            jnp.where(
                selected,
                proposed.boundary_state.convective_history,
                accepted_state.boundary_state.convective_history,
            ),
            jnp.where(
                selected,
                proposed.boundary_state.convective_initialized,
                accepted_state.boundary_state.convective_initialized,
            ),
        )
        return FixedWallLBMState(
            jnp.where(selected, proposed.populations, accepted_state.populations),
            boundary_state,
            jnp.where(selected, proposed.step_index, accepted_state.step_index),
            jnp.where(selected, proposed.time_ms, accepted_state.time_ms),
            accepted_state.initial_mass_mg,
            jnp.where(selected, proposed.total_mass_mg, accepted_state.total_mass_mg),
            jnp.where(
                selected,
                proposed.total_momentum_mg_mm_per_ms,
                accepted_state.total_momentum_mg_mm_per_ms,
            ),
            jnp.where(
                selected,
                proposed.cumulative_outward_volume_mm3,
                accepted_state.cumulative_outward_volume_mm3,
            ),
            self.prepared_id,
        )

    def advance(
        self,
        state: FixedWallLBMState,
        circulation_values: TerminalPortValues,
        /,
    ) -> FixedWallLBMAdvance:
        candidate = self.candidate(state, circulation_values)
        return FixedWallLBMAdvance(candidate, self.commit(state, candidate))

    def checkpoint(self, state: FixedWallLBMState, /) -> FixedWallLBMCheckpoint:
        self._validate_state(state)
        return FixedWallLBMCheckpoint(
            state,
            self.prepared_id,
            self.boundary.topology.topology_id,
        )

    def restore(self, checkpoint: FixedWallLBMCheckpoint, /) -> FixedWallLBMState:
        if not isinstance(checkpoint, FixedWallLBMCheckpoint):
            raise TypeError("checkpoint must be FixedWallLBMCheckpoint.")
        if (
            checkpoint.prepared_id != self.prepared_id
            or checkpoint.topology_id != self.boundary.topology.topology_id
        ):
            raise ValueError("Checkpoint topology does not match the prepared workflow.")
        self._validate_state(checkpoint.state)
        return checkpoint.state


__all__ = [
    "FixedWallLBMAdvance",
    "FixedWallLBMCandidate",
    "FixedWallLBMCheckpoint",
    "FixedWallLBMPlan",
    "FixedWallLBMState",
    "FixedWallMacroscopicState",
    "PreparedFixedWallLBM",
]
