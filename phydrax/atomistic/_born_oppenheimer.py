#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization import (
    AbstractParticleNeighborhoodPlan,
    ParticleVerletState,
    PeriodicCell,
    PreparedVerletParticleNeighborhood,
    VerletParticleNeighborhoodPlan,
)
from ..discretization.particle._image_neighborhood import (
    CellListParticleImageNeighborhoodPlan,
)
from ..discretization.particle._verlet import (
    ImageVerletParticleNeighborhoodPlan,
    ParticleImageVerletState,
    PreparedImageVerletParticleNeighborhood,
)
from ..typing import checked
from ._graph import AtomisticGraphExecutionPlan
from ._hybrid import AbstractExternalAtomisticProvider, ExternalAtomisticEvaluation
from ._potential import AbstractAtomisticPotential, atomistic_potential_revision
from ._potential_program import (
    AtomisticPotentialProgram,
    LearnedGraphPotentialTerm,
    PreparedAtomisticPotentialProgram,
)
from ._system import PreparedAtomisticSystem


ElectronicEvaluator = Callable[
    [PreparedAtomisticSystem, Array, Array | None], ExternalAtomisticEvaluation
]
NativeProviderNeighborhood: TypeAlias = (
    PreparedVerletParticleNeighborhood | PreparedImageVerletParticleNeighborhood
)
NativeProviderNeighborhoodState: TypeAlias = (
    ParticleVerletState | ParticleImageVerletState
)


class NativeAtomisticRequest(StrictModule):
    """One request prepared through the candidate lifecycle, before evaluation.

    ``positions`` are wrapped into the prepared cell and ``image_counts`` are
    the removed lattice images, so ``positions + image_counts @ cell_vectors``
    recovers ``unwrapped_positions``; aperiodic requests carry zero images and
    no cell. ``neighborhood`` is the accepted cache state whose current relation
    the program consumes; it is the host-prepared graph boundary of frozen
    exports.
    """

    positions: Array
    unwrapped_positions: Array
    cell_vectors: Array | None
    image_counts: Array
    neighborhood: NativeProviderNeighborhoodState


class NativeAtomisticEvaluation(StrictModule):
    """Native provider result with its per-atom energy and accepted request."""

    evaluation: ExternalAtomisticEvaluation
    atom_energy: Array
    request: NativeAtomisticRequest


class NativeAtomisticProvider(AbstractExternalAtomisticProvider):
    """Serve one prepared native potential program through the provider boundary.

    Energy, forces, and stress come from one ``program.evaluate`` pass of the
    same scalar energy; the provider adds no energy or force loop. Requests
    without a cache state rebuild candidates through the prepared Verlet
    lifecycle; callers owning a state pass it to ``evaluate_state`` and keep the
    returned state. Stress is available exactly when the system is a fully
    periodic 3D cell and every program term owns cell derivatives.
    """

    program: PreparedAtomisticPotentialProgram
    neighborhood: NativeProviderNeighborhood
    stress_available: bool = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        program: PreparedAtomisticPotentialProgram,
        neighborhood: NativeProviderNeighborhood,
        /,
    ) -> None:
        system = program.system
        if neighborhood.particle_discretization_id != system.particles.prepared_id:
            raise ValueError("Neighborhood belongs to another particle support.")
        cell = system.cell
        box = neighborhood.box
        if cell is None:
            if isinstance(box, PeriodicCell) and any(box.periodic_axes):
                raise ValueError(
                    "A finite atomistic system cannot use a periodic neighborhood box."
                )
        elif not isinstance(box, PeriodicCell) or box.cell_id != cell.cell_id:
            raise ValueError(
                "Neighborhood periodic cell must exactly match the atomistic system cell."
            )
        if (
            cell is not None
            and program.plan.requirements.directed_graph
            and not isinstance(neighborhood, PreparedImageVerletParticleNeighborhood)
        ):
            raise ValueError(
                "Periodic learned graphs require an image-aware Verlet neighborhood."
            )
        stress = (
            cell is not None
            and cell.rank == 3
            and cell.fully_periodic
            and program.plan.capabilities.cell_derivative
        )
        self.program = program
        self.neighborhood = neighborhood
        self.stress_available = stress
        self.conservative = program.plan.capabilities.conservative_energy
        self.differentiable = False
        self.provider_id = canonical_fingerprint(
            {
                "kind": "native-atomistic-provider",
                "program": program.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "stress": stress,
            }
        )

    def _request(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None, /
    ) -> tuple[Array, Array | None]:
        system = self.program.system
        position = jnp.asarray(positions, dtype=system.plan.coordinate_dtype)
        if position.shape != (system.capacity, 3):
            raise ValueError(f"positions must have shape {(system.capacity, 3)}.")
        cell = system.cell
        if cell is None:
            if cell_vectors is not None:
                raise ValueError("An aperiodic native provider accepts no cell vectors.")
            return position, None
        vectors = (
            cell.vectors.astype(position.dtype)
            if cell_vectors is None
            else jnp.asarray(cell_vectors, dtype=position.dtype)
        )
        if vectors.shape != cell.vectors.shape:
            raise ValueError("cell_vectors must match the prepared cell shape.")
        return position, vectors

    def prepare_request(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        previous: NativeProviderNeighborhoodState | None,
        /,
    ) -> NativeAtomisticRequest:
        """Wrap positions and advance ``previous`` without evaluating the model."""

        position, vectors = self._request(positions, cell_vectors)
        return _prepare_native_request(self.neighborhood, position, vectors, previous)

    def evaluate_request(
        self, request: NativeAtomisticRequest, /
    ) -> NativeAtomisticEvaluation:
        return _evaluate_native_request(self, request)

    def evaluate_state(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        previous: NativeProviderNeighborhoodState | None,
        /,
    ) -> NativeAtomisticEvaluation:
        """Evaluate E/F/S, updating ``previous`` through the canonical lifecycle."""

        position, vectors = self._request(positions, cell_vectors)
        return _evaluate_native(self, position, vectors, previous)

    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        if system.prepared_id != self.program.system.prepared_id:
            raise ValueError("Native provider belongs to another atomistic system.")
        return self.evaluate_state(positions, cell_vectors, None).evaluation


def _native_request(
    neighborhood: NativeProviderNeighborhood,
    positions: Array,
    cell_vectors: Array | None,
    previous: NativeProviderNeighborhoodState | None,
    /,
) -> NativeAtomisticRequest:
    cell = neighborhood.box
    if not isinstance(cell, PeriodicCell) or cell_vectors is None:
        wrapped = positions
        images = jnp.zeros(positions.shape, dtype=jnp.int32)
        lifecycle: dict[str, Array] = {}
    else:
        wrapped, images = cell.wrap_with_vectors(positions, cell_vectors)
        lifecycle = {"cell_vectors": cell_vectors}
        if isinstance(neighborhood, PreparedImageVerletParticleNeighborhood):
            lifecycle["image_counts"] = images
    if previous is None:
        state = neighborhood.initialize(wrapped, **lifecycle)
    elif isinstance(neighborhood, PreparedImageVerletParticleNeighborhood):
        if not isinstance(previous, ParticleImageVerletState):
            raise TypeError("Image-aware Verlet updates require an image Verlet state.")
        state = neighborhood.update(wrapped, previous, **lifecycle)
    else:
        if not isinstance(previous, ParticleVerletState):
            raise TypeError("Verlet updates require a ParticleVerletState.")
        state = neighborhood.update(wrapped, previous, **lifecycle)
    return NativeAtomisticRequest(wrapped, positions, cell_vectors, images, state)


def _native_evaluation(
    provider: NativeAtomisticProvider, request: NativeAtomisticRequest, /
) -> NativeAtomisticEvaluation:
    context = (
        {}
        if request.cell_vectors is None
        else {
            "cell_vectors": request.cell_vectors,
            "unwrapped_positions": request.unwrapped_positions,
        }
    )
    value = provider.program.evaluate(
        request.positions,
        request.neighborhood.neighborhood,
        compute_stress=provider.stress_available,
        **context,
    )
    return NativeAtomisticEvaluation(
        ExternalAtomisticEvaluation(
            value.energy,
            value.forces,
            value.stress,
            value.successful & request.neighborhood.successful,
            provider.provider_id,
        ),
        value.atom_energy,
        request,
    )


@eqx.filter_jit
def _prepare_native_request(
    neighborhood: NativeProviderNeighborhood,
    positions: Array,
    cell_vectors: Array | None,
    previous: NativeProviderNeighborhoodState | None,
    /,
) -> NativeAtomisticRequest:
    return _native_request(neighborhood, positions, cell_vectors, previous)


@eqx.filter_jit
def _evaluate_native_request(
    provider: NativeAtomisticProvider, request: NativeAtomisticRequest, /
) -> NativeAtomisticEvaluation:
    return _native_evaluation(provider, request)


@eqx.filter_jit
def _evaluate_native(
    provider: NativeAtomisticProvider,
    positions: Array,
    cell_vectors: Array | None,
    previous: NativeProviderNeighborhoodState | None,
    /,
) -> NativeAtomisticEvaluation:
    request = _native_request(provider.neighborhood, positions, cell_vectors, previous)
    return _native_evaluation(provider, request)


class NativeAtomisticProviderPlan(StrictModule, NonTrainableState):
    """Recipe preparing one learned model as a native provider for any system.

    Periodic systems prepare an image-aware Verlet cache over a cell-list image
    search of radius ``cutoff + skin`` charged by
    ``graph_execution.image_capacity``; finite systems prepare a Verlet cache
    over ``finite_neighborhood``. ``deformation_margin`` bounds cell motion
    within one prepared image stencil before a fresh preparation is required.
    ``model_revision_id`` is the model's numeric revision at construction.
    """

    model: AbstractAtomisticPotential
    graph_execution: AtomisticGraphExecutionPlan
    finite_neighborhood: AbstractParticleNeighborhoodPlan
    skin: float = eqx.field(static=True)
    deformation_margin: float = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: AbstractAtomisticPotential,
        graph_execution: AtomisticGraphExecutionPlan,
        /,
        *,
        finite_neighborhood: AbstractParticleNeighborhoodPlan,
        skin: float,
        deformation_margin: float = 0.0,
    ) -> None:
        if not np.isfinite(skin) or skin <= 0.0:
            raise ValueError("skin must be finite and positive.")
        if not np.isfinite(deformation_margin) or deformation_margin < 0.0:
            raise ValueError("deformation_margin must be finite and non-negative.")
        revision = atomistic_potential_revision(model).revision_id
        self.model = model
        self.graph_execution = graph_execution
        self.finite_neighborhood = finite_neighborhood
        self.skin = float(skin)
        self.deformation_margin = float(deformation_margin)
        self.model_revision_id = revision
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-atomistic-provider-plan",
                "model_revision": revision,
                "architecture": model.architecture_id,
                "graph_execution": graph_execution.plan_id,
                "finite_neighborhood": finite_neighborhood.plan_id,
                "skin": self.skin,
                "deformation_margin": self.deformation_margin,
            }
        )

    def prepare(self, system: PreparedAtomisticSystem, /) -> NativeAtomisticProvider:
        cell = system.cell
        term = LearnedGraphPotentialTerm(self.model, allow_periodic=cell is not None)
        cutoff = term.requirements.cutoff
        if cutoff is None:
            raise ValueError("Learned graph potentials must declare a cutoff.")
        program = AtomisticPotentialProgram([term]).prepare(
            system, graph_execution=self.graph_execution
        )
        if cell is None:
            neighborhood = VerletParticleNeighborhoodPlan(
                self.finite_neighborhood, cutoff, self.skin
            ).prepare(system.particles)
            return NativeAtomisticProvider(program, neighborhood)
        capacity = self.graph_execution.image_capacity
        if capacity is None:
            raise ValueError("Periodic systems require graph_execution.image_capacity.")
        image = ImageVerletParticleNeighborhoodPlan(
            CellListParticleImageNeighborhoodPlan(
                cutoff + self.skin,
                cell,
                capacity,
                maximum_candidate_slots=self.graph_execution.maximum_candidate_slots,
                deformation_margin=self.deformation_margin,
            ),
            cutoff,
            self.skin,
            streamed=self.graph_execution.streamed,
        ).prepare(system.particles)
        return NativeAtomisticProvider(program, image)


class CallableBornOppenheimerProvider(AbstractExternalAtomisticProvider):
    """Explicit host-provider boundary for one Born–Oppenheimer surface."""

    evaluator: ElectronicEvaluator
    provider_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    def __init__(
        self,
        evaluator: ElectronicEvaluator,
        provider_id: str,
        /,
        *,
        conservative: bool = True,
        differentiable: bool = False,
    ) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        identifier = str(provider_id).strip()
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = identifier
        self.conservative = bool(conservative)
        self.differentiable = bool(differentiable)

    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        position = jnp.asarray(positions, dtype=system.plan.coordinate_dtype)
        expected = (system.capacity, 3)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        vectors = None if cell_vectors is None else jnp.asarray(cell_vectors)
        result = self.evaluator(system, position, vectors)
        if not isinstance(result, ExternalAtomisticEvaluation):
            raise TypeError(
                "Born–Oppenheimer evaluator must return ExternalAtomisticEvaluation."
            )
        if result.provider_id != self.provider_id:
            raise ValueError("Born–Oppenheimer result provider identity changed.")
        if result.forces.shape != expected or result.energy.shape != ():
            raise ValueError("Born–Oppenheimer energy or force shape is invalid.")
        return result


class BornOppenheimerState(StrictModule):
    positions: Array
    momenta: Array
    forces: Array
    energy: Array
    time: Array
    step_index: Array
    successful: Array
    provider_id: str = eqx.field(static=True)


class BornOppenheimerStep(StrictModule):
    state: BornOppenheimerState
    initial_evaluation: ExternalAtomisticEvaluation
    final_evaluation: ExternalAtomisticEvaluation
    successful: Array


class BornOppenheimerVelocityVerletPlan(StrictModule, NonTrainableState):
    system: PreparedAtomisticSystem
    provider: AbstractExternalAtomisticProvider
    step_size: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        system: PreparedAtomisticSystem,
        provider: AbstractExternalAtomisticProvider,
        step_size: float,
        /,
    ) -> None:
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        self.system = system
        self.provider = provider
        self.step_size = step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "born-oppenheimer-velocity-verlet",
                "system": system.prepared_id,
                "provider": provider.provider_id,
                "step_size": step,
            }
        )

    def initialize(
        self,
        positions: ArrayLike,
        /,
        *,
        velocity: ArrayLike | None = None,
        momentum: ArrayLike | None = None,
        time: ArrayLike = 0.0,
    ) -> BornOppenheimerState:
        if (velocity is None) == (momentum is None):
            raise ValueError("Supply exactly one of velocity or momentum.")
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        masses = self.system.plan.masses.astype(position.dtype)
        momenta = (
            jnp.asarray(momentum, dtype=position.dtype)
            if momentum is not None
            else masses[:, None] * jnp.asarray(velocity, dtype=position.dtype)
        )
        cell = None if self.system.cell is None else self.system.cell.vectors
        evaluation = self.provider.evaluate(self.system, position, cell)
        successful = evaluation.successful & jnp.all(jnp.isfinite(momenta))
        return BornOppenheimerState(
            position,
            momenta,
            evaluation.forces,
            evaluation.energy,
            jnp.asarray(time, dtype=position.dtype),
            jnp.zeros((), dtype=jnp.int32),
            successful,
            self.provider.provider_id,
        )

    def step(self, state: BornOppenheimerState, /) -> BornOppenheimerStep:
        if state.provider_id != self.provider.provider_id:
            raise ValueError("Born–Oppenheimer state belongs to another provider.")
        initial = ExternalAtomisticEvaluation(
            state.energy,
            state.forces,
            None,
            state.successful,
            self.provider.provider_id,
        )
        dt = jnp.asarray(self.step_size, dtype=state.positions.dtype)
        force_scale = self.system.plan.units.force_to_momentum_rate
        half = state.momenta + 0.5 * dt * force_scale * state.forces
        position = state.positions + dt * half * self.system.inverse_masses[:, None]
        cell = None if self.system.cell is None else self.system.cell.vectors
        final = self.provider.evaluate(self.system, position, cell)
        momentum = half + 0.5 * dt * force_scale * final.forces
        successful = state.successful & final.successful & jnp.all(jnp.isfinite(momentum))
        successor = BornOppenheimerState(
            position,
            momentum,
            final.forces,
            final.energy,
            state.time + dt,
            state.step_index + 1,
            successful,
            self.provider.provider_id,
        )
        return BornOppenheimerStep(
            tree_where(successful, successor, state), initial, final, successful
        )


__all__ = [
    "BornOppenheimerState",
    "BornOppenheimerStep",
    "BornOppenheimerVelocityVerletPlan",
    "CallableBornOppenheimerProvider",
    "NativeAtomisticEvaluation",
    "NativeAtomisticRequest",
    "NativeAtomisticProvider",
    "NativeAtomisticProviderPlan",
    "NativeProviderNeighborhood",
    "NativeProviderNeighborhoodState",
]
