#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.lattice_boltzmann._lattice import (
    LatticeBoltzmannVelocitySet,
)
from ..discretization.lattice_boltzmann._precision import (
    LatticeBoltzmannPrecisionPolicy,
)
from ..discretization.lattice_boltzmann._program import (
    KineticProgramManifest,
    transport_population_manifest,
)
from ..discretization.lattice_boltzmann._thermal import (
    apply_thermal_boundary,
    boussinesq_force,
    BoussinesqCouplingPlan,
    collide_thermal,
    initialize_thermal_ledger,
    sensible_energy_from_temperature,
    temperature_from_sensible_energy,
    thermal_equilibrium,
    thermal_raw_moments,
    ThermalBoundaryCondition,
    ThermalEnergyLedger,
    ThermalLatticeBoltzmannPlan,
    ThermalLatticeBoltzmannState,
)


ThermalPopulationStream = Callable[[Array], Array]


class ThermalLatticeBoltzmannProblemIR(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    transport: ThermalLatticeBoltzmannPlan
    boundaries: tuple[ThermalBoundaryCondition, ...]
    volumetric_source: Array
    boussinesq: BoussinesqCouplingPlan | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        transport: ThermalLatticeBoltzmannPlan,
        /,
        *,
        boundaries=(),
        volumetric_source: ArrayLike = 0.0,
        boussinesq: BoussinesqCouplingPlan | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        boundary_values = tuple(boundaries)
        source = np.asarray(volumetric_source, dtype=float)
        if not name_:
            raise ValueError("Thermal LBM problem name must be nonempty.")
        if not isinstance(transport, ThermalLatticeBoltzmannPlan):
            raise TypeError("transport must be a ThermalLatticeBoltzmannPlan.")
        if any(
            not isinstance(value, ThermalBoundaryCondition) for value in boundary_values
        ):
            raise TypeError("boundaries must contain ThermalBoundaryCondition values.")
        if np.any(~np.isfinite(source)):
            raise ValueError("volumetric_source must be finite.")
        if boussinesq is not None and not isinstance(boussinesq, BoussinesqCouplingPlan):
            raise TypeError("boussinesq must be a BoussinesqCouplingPlan or None.")
        generated = canonical_fingerprint(
            {
                "kind": "thermal-lattice-boltzmann-problem-ir",
                "name": name_,
                "transport": transport.plan_id,
                "boundaries": [value.boundary_id for value in boundary_values],
                "source": array_tree_fingerprint(source),
                "boussinesq": None if boussinesq is None else boussinesq.plan_id,
            }
        )
        self.name = name_
        self.transport = transport
        self.boundaries = boundary_values
        self.volumetric_source = jnp.asarray(source)
        self.boussinesq = boussinesq
        self.problem_id = generated if problem_id is None else str(problem_id)
        if not self.problem_id:
            raise ValueError("problem_id must be nonempty.")


class ThermalLatticeBoltzmannTransportResult(StrictModule):
    candidate_state: ThermalLatticeBoltzmannState
    accepted_state: ThermalLatticeBoltzmannState
    boundary_energy: Array
    source_energy: Array
    successful: Array


class CompiledThermalLatticeBoltzmannProblem(StrictModule, NonTrainableState):
    problem: ThermalLatticeBoltzmannProblemIR
    lattice: LatticeBoltzmannVelocitySet
    precision: LatticeBoltzmannPrecisionPolicy
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    cell_measure: Array
    program_manifest: KineticProgramManifest
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: ThermalLatticeBoltzmannProblemIR,
        lattice: LatticeBoltzmannVelocitySet,
        precision: LatticeBoltzmannPrecisionPolicy,
        spatial_shape,
        spacing: float,
        step_size: float,
        cell_measure: ArrayLike,
        /,
    ):
        shape = tuple(int(value) for value in spatial_shape)
        dx = float(spacing)
        dt = float(step_size)
        measure = np.asarray(cell_measure, dtype=float)
        if not isinstance(problem, ThermalLatticeBoltzmannProblemIR):
            raise TypeError("problem must be a ThermalLatticeBoltzmannProblemIR.")
        if not isinstance(lattice, LatticeBoltzmannVelocitySet):
            raise TypeError("lattice must be a LatticeBoltzmannVelocitySet.")
        if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
            raise TypeError("precision must be a LatticeBoltzmannPrecisionPolicy.")
        if len(shape) != lattice.dimension or any(value <= 0 for value in shape):
            raise ValueError(
                "spatial_shape must contain one positive size per dimension."
            )
        if not np.isfinite(dx) or dx <= 0.0 or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("spacing and step_size must be finite and positive.")
        if (
            measure.shape not in ((), shape)
            or np.any(~np.isfinite(measure))
            or np.any(measure <= 0.0)
        ):
            raise ValueError(
                "cell_measure must be positive scalar or match spatial_shape."
            )
        if problem.volumetric_source.shape not in ((), shape):
            raise ValueError("Problem heat source must be scalar or match spatial_shape.")
        for boundary in problem.boundaries:
            if boundary.node_mask.shape != shape:
                raise ValueError("Every thermal boundary mask must match spatial_shape.")
            if boundary.outward_normal.shape != shape + (lattice.dimension,):
                raise ValueError(
                    "Every thermal boundary normal must match lattice dimension."
                )
        program_manifest = transport_population_manifest(
            "thermal_lattice_boltzmann",
            lattice.lattice_id,
            precision.policy_id,
            "thermal_populations",
            (lattice.population_count,),
            ("sensible_energy",),
            dimension=lattice.dimension,
        )
        self.problem = problem
        self.lattice = lattice
        self.precision = precision
        self.spatial_shape = shape
        self.spacing = dx
        self.step_size = dt
        self.program_manifest = program_manifest
        self.cell_measure = jnp.asarray(measure)
        generated = canonical_fingerprint(
            {
                "kind": "compiled-thermal-lattice-boltzmann-problem",
                "problem": problem.problem_id,
                "lattice": lattice.lattice_id,
                "precision": precision.policy_id,
                "shape": shape,
                "spacing": dx,
                "step_size": dt,
                "cell_measure": array_tree_fingerprint(measure),
                "program_manifest": program_manifest.manifest_id,
            }
        )
        self.compilation_id = generated

    def initialize_state(
        self, temperature: ArrayLike, velocity: ArrayLike, /
    ) -> ThermalLatticeBoltzmannState:
        value = jnp.asarray(temperature)
        flow = jnp.asarray(velocity, dtype=value.dtype)
        if value.shape != self.spatial_shape:
            raise ValueError("temperature must match compiled spatial_shape.")
        if flow.shape != self.spatial_shape + (self.lattice.dimension,):
            raise ValueError("velocity must extend spatial_shape by lattice dimension.")
        energy = sensible_energy_from_temperature(value, self.problem.transport)
        populations = thermal_equilibrium(energy, flow, self.lattice, self.precision)
        ledger = initialize_thermal_ledger(energy, self.cell_measure)
        return ThermalLatticeBoltzmannState(
            populations,
            ledger,
            jnp.asarray(True),
            jnp.zeros((), dtype=jnp.int32),
            self.compilation_id,
        )

    def sensible_energy(self, state: ThermalLatticeBoltzmannState, /) -> Array:
        self._validate_state(state)
        energy, _ = thermal_raw_moments(state.populations, self.lattice, self.precision)
        return energy

    def temperature(self, state: ThermalLatticeBoltzmannState, /) -> Array:
        return temperature_from_sensible_energy(
            self.sensible_energy(state), self.problem.transport
        )

    def buoyancy_force(self, state: ThermalLatticeBoltzmannState, /) -> Array:
        if self.problem.boussinesq is None:
            raise ValueError("The compiled thermal problem has no Boussinesq coupling.")
        return boussinesq_force(self.temperature(state), self.problem.boussinesq)

    def _validate_state(self, state):
        if not isinstance(state, ThermalLatticeBoltzmannState):
            raise TypeError("state must be a ThermalLatticeBoltzmannState.")
        if state.state_id != self.compilation_id:
            raise ValueError("Thermal state does not match compiled problem.")
        if state.populations.shape != self.spatial_shape + (
            self.lattice.population_count,
        ):
            raise ValueError("Thermal population shape is invalid.")


def compile_thermal_lattice_boltzmann_problem(
    problem: ThermalLatticeBoltzmannProblemIR,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    spatial_shape,
    /,
    *,
    spacing: float,
    step_size: float,
    cell_measure: ArrayLike | None = None,
) -> CompiledThermalLatticeBoltzmannProblem:
    shape = tuple(int(value) for value in spatial_shape)
    measure = spacing ** len(shape) if cell_measure is None else cell_measure
    return CompiledThermalLatticeBoltzmannProblem(
        problem, lattice, precision, shape, spacing, step_size, measure
    )


def advance_thermal_lattice_boltzmann(
    compiled: CompiledThermalLatticeBoltzmannProblem,
    state: ThermalLatticeBoltzmannState,
    velocity: ArrayLike,
    stream: ThermalPopulationStream,
    /,
    *,
    volumetric_source: ArrayLike | None = None,
) -> ThermalLatticeBoltzmannTransportResult:
    """Execute collide/common-stream/boundary as one atomic thermal step."""
    if not isinstance(compiled, CompiledThermalLatticeBoltzmannProblem):
        raise TypeError("compiled must be a CompiledThermalLatticeBoltzmannProblem.")
    compiled._validate_state(state)
    if not callable(stream):
        raise TypeError("stream must be a callable common lattice streaming step.")
    source = (
        compiled.problem.volumetric_source
        if volumetric_source is None
        else jnp.asarray(volumetric_source)
    )
    source = jnp.broadcast_to(source, compiled.spatial_shape)
    collision = collide_thermal(
        state.populations,
        velocity,
        source,
        compiled.problem.transport,
        compiled.lattice,
        compiled.precision,
        compiled.step_size,
        compiled.spacing,
    )
    streamed = jnp.asarray(stream(collision.populations))
    if streamed.shape != state.populations.shape:
        raise ValueError("stream returned an invalid thermal population shape.")
    bounded = streamed
    for boundary in compiled.problem.boundaries:
        bounded = apply_thermal_boundary(
            bounded,
            boundary,
            compiled.problem.transport,
            compiled.lattice,
            compiled.precision,
        )
    measure = jnp.broadcast_to(compiled.cell_measure, compiled.spatial_shape)
    pre_boundary, _ = thermal_raw_moments(streamed, compiled.lattice, compiled.precision)
    post_boundary, _ = thermal_raw_moments(bounded, compiled.lattice, compiled.precision)
    boundary_energy = jnp.sum((post_boundary - pre_boundary) * measure)
    source_energy = compiled.step_size * jnp.sum(source * measure)
    total = jnp.sum(post_boundary * measure)
    ledger = state.ledger
    updated_ledger = ThermalEnergyLedger(
        ledger.initial_sensible_energy,
        ledger.boundary_energy + boundary_energy,
        ledger.source_energy + source_energy,
        ledger.reaction_energy,
        total
        - (
            ledger.initial_sensible_energy
            + ledger.boundary_energy
            + boundary_energy
            + ledger.source_energy
            + source_energy
            + ledger.reaction_energy
        ),
    )
    successful = (
        collision.successful
        & jnp.all(jnp.isfinite(bounded))
        & jnp.all(jnp.isfinite(boundary_energy))
        & jnp.all(jnp.isfinite(source_energy))
    )
    candidate = ThermalLatticeBoltzmannState(
        bounded,
        updated_ledger,
        successful,
        state.step_index + jnp.asarray(1, dtype=state.step_index.dtype),
        state.state_id,
    )
    accepted = eqx.tree_at(
        lambda value: (
            value.populations,
            value.ledger,
            value.successful,
            value.step_index,
        ),
        state,
        replace=(
            jnp.where(successful, candidate.populations, state.populations),
            _select_thermal_ledger(successful, candidate.ledger, state.ledger),
            jnp.where(successful, candidate.successful, state.successful),
            jnp.where(successful, candidate.step_index, state.step_index),
        ),
    )
    return ThermalLatticeBoltzmannTransportResult(
        candidate, accepted, boundary_energy, source_energy, successful
    )


def _select_thermal_ledger(condition, proposed, current):
    return ThermalEnergyLedger(
        jnp.where(
            condition, proposed.initial_sensible_energy, current.initial_sensible_energy
        ),
        jnp.where(condition, proposed.boundary_energy, current.boundary_energy),
        jnp.where(condition, proposed.source_energy, current.source_energy),
        jnp.where(condition, proposed.reaction_energy, current.reaction_energy),
        jnp.where(condition, proposed.energy_residual, current.energy_residual),
    )


__all__ = [
    "CompiledThermalLatticeBoltzmannProblem",
    "ThermalLatticeBoltzmannProblemIR",
    "ThermalLatticeBoltzmannTransportResult",
    "ThermalPopulationStream",
    "advance_thermal_lattice_boltzmann",
    "compile_thermal_lattice_boltzmann_problem",
]
