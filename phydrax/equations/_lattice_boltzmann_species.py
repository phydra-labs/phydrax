#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

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
from ..discretization.lattice_boltzmann._species import (
    apply_species_boundary,
    collide_species,
    initialize_species_ledger,
    species_equilibrium,
    species_raw_moments,
    SpeciesBoundaryCondition,
    SpeciesLatticeBoltzmannPlan,
    SpeciesLatticeBoltzmannState,
    SpeciesLedger,
)
from ._particle_thermochemistry import ParticleSpeciesSchema


SpeciesPopulationStream = Callable[[Array], Array]


class SpeciesLatticeBoltzmannProblemIR(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    schema: ParticleSpeciesSchema
    transport: SpeciesLatticeBoltzmannPlan
    boundaries: tuple[SpeciesBoundaryCondition, ...]
    volumetric_source: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        schema: ParticleSpeciesSchema,
        transport: SpeciesLatticeBoltzmannPlan,
        /,
        *,
        boundaries=(),
        volumetric_source: ArrayLike | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        boundary_values = tuple(boundaries)
        if not name_:
            raise ValueError("Species LBM problem name must be nonempty.")
        if not isinstance(schema, ParticleSpeciesSchema):
            raise TypeError("schema must be a ParticleSpeciesSchema.")
        if not isinstance(transport, SpeciesLatticeBoltzmannPlan):
            raise TypeError("transport must be a SpeciesLatticeBoltzmannPlan.")
        if transport.species_count != schema.species_count:
            raise ValueError("Species schema and transport species counts must match.")
        if any(
            not isinstance(value, SpeciesBoundaryCondition) for value in boundary_values
        ):
            raise TypeError("boundaries must contain SpeciesBoundaryCondition values.")
        source = (
            np.zeros((schema.species_count,), dtype=float)
            if volumetric_source is None
            else np.asarray(volumetric_source, dtype=float)
        )
        if (
            source.ndim == 0
            or source.shape[-1] != schema.species_count
            or np.any(~np.isfinite(source))
        ):
            raise ValueError("volumetric_source must end in the schema species axis.")
        generated = canonical_fingerprint(
            {
                "kind": "species-lattice-boltzmann-problem-ir",
                "name": name_,
                "schema": schema.schema_id,
                "transport": transport.plan_id,
                "boundaries": [value.boundary_id for value in boundary_values],
                "source": array_tree_fingerprint(source),
            }
        )
        self.name = name_
        self.schema = schema
        self.transport = transport
        self.boundaries = boundary_values
        self.volumetric_source = jnp.asarray(source)
        self.problem_id = generated if problem_id is None else str(problem_id)
        if not self.problem_id:
            raise ValueError("problem_id must be nonempty.")


class SpeciesLatticeBoltzmannTransportResult(StrictModule):
    candidate_state: SpeciesLatticeBoltzmannState
    accepted_state: SpeciesLatticeBoltzmannState
    boundary_species_amount: Array
    source_species_amount: Array
    successful: Array


class CompiledSpeciesLatticeBoltzmannProblem(StrictModule, NonTrainableState):
    problem: SpeciesLatticeBoltzmannProblemIR
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
        problem: SpeciesLatticeBoltzmannProblemIR,
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
        if not isinstance(problem, SpeciesLatticeBoltzmannProblemIR):
            raise TypeError("problem must be a SpeciesLatticeBoltzmannProblemIR.")
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
        source_shape = problem.volumetric_source.shape
        if source_shape not in (
            (problem.schema.species_count,),
            shape + (problem.schema.species_count,),
        ):
            raise ValueError(
                "Problem species source must be uniform or match spatial_shape."
            )
        for boundary in problem.boundaries:
            if boundary.node_mask.shape != shape:
                raise ValueError("Every species boundary mask must match spatial_shape.")
            if boundary.outward_normal.shape != shape + (lattice.dimension,):
                raise ValueError(
                    "Every species boundary normal must match lattice dimension."
                )
            if boundary.value.shape not in (
                (problem.schema.species_count,),
                shape + (problem.schema.species_count,),
            ):
                raise ValueError("Every species boundary value must end in species axis.")
        program_manifest = transport_population_manifest(
            "species_lattice_boltzmann",
            lattice.lattice_id,
            precision.policy_id,
            "species_populations",
            (problem.schema.species_count, lattice.population_count),
            ("species_amount", "element_amount"),
            dimension=lattice.dimension,
            source_component_shape=(problem.schema.species_count,),
        )
        generated = canonical_fingerprint(
            {
                "kind": "compiled-species-lattice-boltzmann-problem",
                "problem": problem.problem_id,
                "lattice": lattice.lattice_id,
                "precision": precision.policy_id,
                "shape": list(shape),
                "spacing": dx,
                "step_size": dt,
                "cell_measure": array_tree_fingerprint(measure),
                "program_manifest": program_manifest.manifest_id,
            }
        )
        self.problem = problem
        self.lattice = lattice
        self.precision = precision
        self.spatial_shape = shape
        self.spacing = dx
        self.step_size = dt
        self.cell_measure = jnp.asarray(measure)
        self.program_manifest = program_manifest
        self.compilation_id = generated

    def initialize_state(
        self, concentration: ArrayLike, velocity: ArrayLike, /
    ) -> SpeciesLatticeBoltzmannState:
        amount = jnp.asarray(concentration)
        flow = jnp.asarray(velocity, dtype=amount.dtype)
        expected = self.spatial_shape + (self.problem.schema.species_count,)
        if amount.shape != expected:
            raise ValueError("concentration must match spatial_shape and species count.")
        if flow.shape != self.spatial_shape + (self.lattice.dimension,):
            raise ValueError("velocity must extend spatial_shape by lattice dimension.")
        populations = species_equilibrium(amount, flow, self.lattice, self.precision)
        ledger = initialize_species_ledger(
            amount, self.cell_measure, self.problem.schema.element_composition
        )
        return SpeciesLatticeBoltzmannState(
            populations,
            ledger,
            jnp.asarray(True),
            jnp.zeros((), dtype=jnp.int32),
            self.compilation_id,
        )

    def concentration(self, state: SpeciesLatticeBoltzmannState, /) -> Array:
        self._validate_state(state)
        concentration, _ = species_raw_moments(
            state.populations, self.lattice, self.precision
        )
        return concentration

    def _validate_state(self, state):
        if not isinstance(state, SpeciesLatticeBoltzmannState):
            raise TypeError("state must be a SpeciesLatticeBoltzmannState.")
        if state.state_id != self.compilation_id:
            raise ValueError("Species state does not match compiled problem.")
        if state.populations.shape != self.spatial_shape + (
            self.problem.schema.species_count,
            self.lattice.population_count,
        ):
            raise ValueError("Species population shape is invalid.")


def compile_species_lattice_boltzmann_problem(
    problem: SpeciesLatticeBoltzmannProblemIR,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    spatial_shape,
    /,
    *,
    spacing: float,
    step_size: float,
    cell_measure: ArrayLike | None = None,
) -> CompiledSpeciesLatticeBoltzmannProblem:
    shape = tuple(int(value) for value in spatial_shape)
    measure = spacing ** len(shape) if cell_measure is None else cell_measure
    return CompiledSpeciesLatticeBoltzmannProblem(
        problem, lattice, precision, shape, spacing, step_size, measure
    )


def advance_species_lattice_boltzmann(
    compiled: CompiledSpeciesLatticeBoltzmannProblem,
    state: SpeciesLatticeBoltzmannState,
    velocity: ArrayLike,
    stream: SpeciesPopulationStream,
    /,
    *,
    volumetric_source: ArrayLike | None = None,
) -> SpeciesLatticeBoltzmannTransportResult:
    """Execute collide/common-stream/boundary as one atomic species step."""
    if not isinstance(compiled, CompiledSpeciesLatticeBoltzmannProblem):
        raise TypeError("compiled must be a CompiledSpeciesLatticeBoltzmannProblem.")
    compiled._validate_state(state)
    if not callable(stream):
        raise TypeError("stream must be a callable common lattice streaming step.")
    source = (
        compiled.problem.volumetric_source
        if volumetric_source is None
        else jnp.asarray(volumetric_source)
    )
    source = jnp.broadcast_to(
        source,
        compiled.spatial_shape + (compiled.problem.schema.species_count,),
    )
    collision = collide_species(
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
        raise ValueError("stream returned an invalid species population shape.")
    bounded = streamed
    for boundary in compiled.problem.boundaries:
        bounded = apply_species_boundary(
            bounded,
            boundary,
            compiled.problem.transport,
            compiled.lattice,
            compiled.precision,
        )
    measure = jnp.broadcast_to(compiled.cell_measure, compiled.spatial_shape)
    pre_boundary, _ = species_raw_moments(streamed, compiled.lattice, compiled.precision)
    post_boundary, _ = species_raw_moments(bounded, compiled.lattice, compiled.precision)
    spatial_axes = tuple(range(len(compiled.spatial_shape)))
    boundary_amount = jnp.sum(
        (post_boundary - pre_boundary) * measure[..., None], axis=spatial_axes
    )
    source_amount = compiled.step_size * jnp.sum(
        source * measure[..., None], axis=spatial_axes
    )
    current_amount = jnp.sum(post_boundary * measure[..., None], axis=spatial_axes)
    ledger = state.ledger
    composition = compiled.problem.schema.element_composition.astype(current_amount.dtype)
    boundary_element = contract("es,s->e", composition, boundary_amount)
    source_element = contract("es,s->e", composition, source_amount)
    species_residual = current_amount - (
        ledger.initial_species_amount
        + ledger.boundary_species_amount
        + boundary_amount
        + ledger.source_species_amount
        + source_amount
        + ledger.reaction_species_amount
    )
    element_residual = contract("es,s->e", composition, species_residual)
    updated_ledger = SpeciesLedger(
        ledger.initial_species_amount,
        ledger.boundary_species_amount + boundary_amount,
        ledger.source_species_amount + source_amount,
        ledger.reaction_species_amount,
        species_residual,
        ledger.initial_element_amount,
        ledger.boundary_element_amount + boundary_element,
        ledger.source_element_amount + source_element,
        element_residual,
    )
    successful = (
        collision.successful
        & jnp.all(jnp.isfinite(bounded))
        & jnp.all(jnp.isfinite(boundary_amount))
        & jnp.all(jnp.isfinite(source_amount))
        & jnp.all(post_boundary >= 0.0)
    )
    candidate = SpeciesLatticeBoltzmannState(
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
            _select_species_ledger(successful, candidate.ledger, state.ledger),
            jnp.where(successful, candidate.successful, state.successful),
            jnp.where(successful, candidate.step_index, state.step_index),
        ),
    )
    return SpeciesLatticeBoltzmannTransportResult(
        candidate, accepted, boundary_amount, source_amount, successful
    )


def _select_species_ledger(condition, proposed, current):
    return SpeciesLedger(
        jnp.where(
            condition, proposed.initial_species_amount, current.initial_species_amount
        ),
        jnp.where(
            condition, proposed.boundary_species_amount, current.boundary_species_amount
        ),
        jnp.where(
            condition, proposed.source_species_amount, current.source_species_amount
        ),
        jnp.where(
            condition, proposed.reaction_species_amount, current.reaction_species_amount
        ),
        jnp.where(condition, proposed.species_residual, current.species_residual),
        jnp.where(
            condition, proposed.initial_element_amount, current.initial_element_amount
        ),
        jnp.where(
            condition, proposed.boundary_element_amount, current.boundary_element_amount
        ),
        jnp.where(
            condition, proposed.source_element_amount, current.source_element_amount
        ),
        jnp.where(condition, proposed.element_residual, current.element_residual),
    )


__all__ = [
    "CompiledSpeciesLatticeBoltzmannProblem",
    "SpeciesLatticeBoltzmannProblemIR",
    "SpeciesLatticeBoltzmannTransportResult",
    "SpeciesPopulationStream",
    "advance_species_lattice_boltzmann",
    "compile_species_lattice_boltzmann_problem",
]
