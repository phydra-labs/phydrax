#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Born–Oppenheimer provider boundary and the native learned-model provider.

``NativeAtomisticProvider`` serves one prepared potential program through the
external provider boundary.  Its resources are the program's single
``PreparedAtomisticNeighborhoodBundle``: a request binds the current geometry
as bundle geometry and initializes, refreshes (``prepare_request``) or binds
without any rebuild (``bind_request``) the accepted bundle state, and the
program evaluates energy, forces and stress at exactly that certified point.
Capacity recovery is an explicit host lifecycle over the declared bundle
ladders; scientific failures are never retried as capacity.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import ExplicitFreeze, NonTrainableState
from .._tree_math import tree_where
from ..typing import checked, PRNGKey
from ._conditioning import AtomisticConditioning
from ._graph import AtomisticGraphExecutionPlan
from ._hybrid import (
    _provider_id,
    _same_provider,
    AbstractExternalAtomisticProvider,
    ExternalAtomisticEvaluation,
)
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodBundleState,
    PreparedAtomisticNeighborhoodBundle,
)
from ._potential import AbstractAtomisticPotential, atomistic_model_revision
from ._potential_program import (
    AtomisticPotentialEvaluation,
    AtomisticPotentialProgram,
    LearnedGraphPotentialTerm,
    PreparedAtomisticPotentialProgram,
)
from ._relaxation import trial_geometry
from ._system import PreparedAtomisticSystem


ElectronicEvaluator = Callable[
    [PreparedAtomisticSystem, Array, Array | None], ExternalAtomisticEvaluation
]


class NativeAtomisticRequest(StrictModule):
    """One geometry bound to the provider's resources, before evaluation.

    ``positions`` are wrapped into the prepared cell under the current
    ``cell_vectors`` and ``image_counts`` are the removed integer lattice
    images, so ``positions + image_counts @ cell_vectors`` recovers
    ``unwrapped_positions``; finite requests carry no cell and no images.
    ``neighborhoods`` is the bundle state certified at exactly this geometry.
    """

    positions: Array
    unwrapped_positions: Array
    cell_vectors: Array | None
    image_counts: Array | None
    neighborhoods: AtomisticNeighborhoodBundleState


class NativeAtomisticEvaluation(StrictModule):
    """Provider-boundary result with the complete program evaluation evidence.

    ``program`` retains the per-term evidence, fields, stress convention and
    homogeneous-strain derivative of the same scalar energy at the request's
    current cell; ``evaluation`` is its provider-boundary projection, whose
    success also requires a successful bundle state.
    """

    evaluation: ExternalAtomisticEvaluation
    program: AtomisticPotentialEvaluation
    request: NativeAtomisticRequest

    @property
    def atom_energy(self) -> Array:
        return self.program.atom_energy


class NativeAtomisticProvider(AbstractExternalAtomisticProvider, ExplicitFreeze):
    """Serve one prepared native potential program through the provider boundary.

    Energy, forces, and stress come from one ``program.evaluate`` pass of the
    same scalar energy at the request's current positions and cell; the
    provider adds no energy or force loop.  ``evaluate`` initializes fresh
    resources for every call; callers owning an accepted bundle state pass it
    to ``evaluate_state`` and keep the returned state.  Stress is available
    exactly when the system is a fully periodic 3D cell and every program term
    owns cell derivatives.

    This conditioned deployment intentionally freezes its prepared program,
    including embedded learned potentials: every array below the provider is
    FIXED. Internal position and cell derivatives still produce forces and
    stress; the external provider exposes no trainable model coordinates.

    ``provider_identity`` is the dynamic exact digest of the served program
    (``provider_label`` is ``None``), so one compiled evaluation serves every
    material/parameter revision of the same layout; the host ``provider_id``
    decodes it.  The ``ExternalAtomisticEvaluation`` is assembled inside the
    compiled kernel and carries the same words, so ``evaluate``,
    ``evaluate_state`` and ``evaluate_request`` also run under an outer
    transformation; ``evaluate`` folds the binding of the supplied system
    into the evaluation's success instead of decoding identities.
    """

    program: PreparedAtomisticPotentialProgram
    provider_identity: IdentityWords
    provider_label: str | None = eqx.field(static=True)
    stress_available: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    @checked
    def __init__(self, program: PreparedAtomisticPotentialProgram, /) -> None:
        cell = program.system.cell
        stress = (
            cell is not None
            and cell.rank == 3
            and cell.fully_periodic
            and program.plan.capabilities.cell_derivative
        )
        self.program = program
        self.stress_available = stress
        self.conservative = program.plan.capabilities.conservative_energy
        self.differentiable = False
        self.provider_identity = IdentityWords.of(
            {
                "kind": "native-atomistic-provider",
                "program": program.prepared_id,
                "stress": stress,
            }
        )
        self.provider_label = None

    @property
    def bundle(self) -> PreparedAtomisticNeighborhoodBundle:
        return self.program.neighborhood_bundle

    def with_neighborhood_bundle(
        self, bundle: PreparedAtomisticNeighborhoodBundle, /
    ) -> NativeAtomisticProvider:
        """Rebind grown resources; numerical term preparations are retained."""

        return NativeAtomisticProvider(self.program.with_neighborhood_bundle(bundle))

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
        previous: AtomisticNeighborhoodBundleState | None,
        /,
    ) -> NativeAtomisticRequest:
        """Wrap positions and initialize or refresh ``previous`` without evaluating."""

        position, vectors = self._request(positions, cell_vectors)
        return _prepare_native_request(self, position, vectors, previous, False)

    def bind_request(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        frozen: AtomisticNeighborhoodBundleState,
        /,
    ) -> NativeAtomisticRequest:
        """Bind a geometry to the frozen epochs of ``frozen`` without any rebuild.

        Uncertified geometry or a changed declared selection yields a stale,
        unsuccessful request (``PreparedAtomisticNeighborhoodBundle.bind``).
        """

        position, vectors = self._request(positions, cell_vectors)
        return _prepare_native_request(self, position, vectors, frozen, True)

    def evaluate_request(
        self, request: NativeAtomisticRequest, /
    ) -> NativeAtomisticEvaluation:
        return _evaluate_native_request(self, request)

    def evaluate_state(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        previous: AtomisticNeighborhoodBundleState | None,
        /,
    ) -> NativeAtomisticEvaluation:
        """Evaluate E/F/S, initializing or refreshing ``previous`` first."""

        position, vectors = self._request(positions, cell_vectors)
        return _evaluate_native(self, position, vectors, previous)

    def evaluate_recovering(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        previous: AtomisticNeighborhoodBundleState | None,
        /,
    ) -> NativeAtomisticRecovery:
        """Host lifecycle: evaluate, replaying capacity-only failures on grown ladders.

        A capacity-only resource failure selects the next declared capacities
        (``grow_capacities``), rebinds the program, rebases ``previous`` and
        replays the same request.  Scientific, stale and domain failures are
        returned unchanged; an exhausted ladder is reported, never retried.
        """

        provider = self
        evaluation = provider.evaluate_state(positions, cell_vectors, previous)
        growths = 0
        while bool(np.asarray(evaluation.request.neighborhoods.capacity_only)[0]):
            try:
                grown = provider.bundle.grow_capacities(evaluation.request.neighborhoods)
            except RuntimeError:
                return NativeAtomisticRecovery(provider, evaluation, growths, True)
            provider = provider.with_neighborhood_bundle(grown)
            accepted = None if previous is None else grown.rebase(previous)
            evaluation = provider.evaluate_state(positions, cell_vectors, accepted)
            growths += 1
        return NativeAtomisticRecovery(provider, evaluation, growths, False)

    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        """Evaluate E/F/S of ``system`` with freshly initialized resources.

        ``system`` is compared with the served system by identity words; a
        foreign system yields an unsuccessful evaluation (traceable).
        """
        value = self.evaluate_state(positions, cell_vectors, None).evaluation
        bound = system.prepared_identity.equals(self.program.system.prepared_identity)
        return ExternalAtomisticEvaluation(
            value.energy,
            value.forces,
            value.stress,
            value.successful & bound,
            provider_identity=self.provider_identity,
        )


class NativeAtomisticRecovery(StrictModule):
    """Result of one host capacity-recovery evaluation.

    ``provider`` is bound to the final resource levels (the input provider when
    no growth occurred); ``capacity_growths`` counts replays and ``exhausted``
    reports a declared capacity ladder that could not hold the request.
    """

    provider: NativeAtomisticProvider
    evaluation: NativeAtomisticEvaluation
    capacity_growths: int = eqx.field(static=True)
    exhausted: bool = eqx.field(static=True)


def _native_request(
    provider: NativeAtomisticProvider,
    positions: Array,
    cell_vectors: Array | None,
    previous: AtomisticNeighborhoodBundleState | None,
    frozen: bool,
    /,
) -> NativeAtomisticRequest:
    program = provider.program
    bundle = program.neighborhood_bundle
    geometry, wrapped, images = trial_geometry(program, bundle, positions, cell_vectors)
    if previous is None:
        if frozen:
            raise ValueError("A frozen-epoch binding requires the accepted state.")
        state = bundle.initialize(geometry)
    elif frozen:
        state = bundle.bind(previous, geometry)
    else:
        state = bundle.refresh(previous, geometry)
    return NativeAtomisticRequest(wrapped, positions, cell_vectors, images, state)


def _native_evaluation(
    provider: NativeAtomisticProvider, request: NativeAtomisticRequest, /
) -> NativeAtomisticEvaluation:
    """Traced: one program evaluation and its provider-boundary projection."""
    context = (
        {}
        if request.cell_vectors is None
        else {
            "cell_vectors": request.cell_vectors,
            "image_counts": request.image_counts,
            "unwrapped_positions": request.unwrapped_positions,
        }
    )
    value = provider.program.evaluate(
        request.positions,
        request.neighborhoods,
        compute_stress=provider.stress_available,
        **context,
    )
    return NativeAtomisticEvaluation(
        ExternalAtomisticEvaluation(
            value.energy,
            value.forces,
            value.stress,
            value.successful & request.neighborhoods.successful[0],
            provider_identity=provider.provider_identity,
        ),
        value,
        request,
    )


@eqx.filter_jit
def _prepare_native_request(
    provider: NativeAtomisticProvider,
    positions: Array,
    cell_vectors: Array | None,
    previous: AtomisticNeighborhoodBundleState | None,
    frozen: bool,
    /,
) -> NativeAtomisticRequest:
    return _native_request(provider, positions, cell_vectors, previous, frozen)


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
    previous: AtomisticNeighborhoodBundleState | None,
    /,
) -> NativeAtomisticEvaluation:
    request = _native_request(provider, positions, cell_vectors, previous, False)
    return _native_evaluation(provider, request)


def _resource_plan(
    resources: AtomisticNeighborhoodBundlePlan | None,
    required: frozenset[str],
    label: str,
    /,
) -> AtomisticNeighborhoodBundlePlan | None:
    if resources is None:
        return None
    if {value.requirement_id for value in resources.requirements} != required:
        raise ValueError(
            f"{label} must cover exactly the learned graph term's neighborhood "
            "requirements."
        )
    if resources.case_count != 1:
        raise ValueError(f"{label} must bind exactly one system case.")
    return resources


class NativeAtomisticProviderPlan(StrictModule, ExplicitFreeze):
    """Recipe preparing one learned model as a native provider for any system.

    The model is bound as one ``LearnedGraphPotentialTerm`` named ``name``.
    Resources are explicit complete ``AtomisticNeighborhoodBundlePlan`` values
    covering exactly that term's requirements: ``finite_resources`` (an
    explicit nonperiodic search frame for image sources) serves finite
    systems and ``periodic_resources`` (searched in each system's own cell)
    serves periodic systems.  A system whose kind has no declared resources
    is refused; no dense or implicit fallback exists.  Cell deformation,
    skin, search and capacity ladders are owned by those plans.
    ``model_revision_id`` is the model's numeric revision at construction.

    This immutable deployment recipe intentionally freezes its embedded model:
    every array below the recipe is FIXED, while training uses ``model``
    separately. A new numeric model revision requires a new recipe, preserving
    the revision-bound provider and compiler identities.
    """

    model: AbstractAtomisticPotential
    graph_execution: AtomisticGraphExecutionPlan
    finite_resources: AtomisticNeighborhoodBundlePlan | None
    periodic_resources: AtomisticNeighborhoodBundlePlan | None
    name: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: AbstractAtomisticPotential,
        graph_execution: AtomisticGraphExecutionPlan,
        /,
        *,
        finite_resources: AtomisticNeighborhoodBundlePlan | None = None,
        periodic_resources: AtomisticNeighborhoodBundlePlan | None = None,
        name: str = "learned",
    ) -> None:
        if finite_resources is None and periodic_resources is None:
            raise ValueError(
                "Declare finite_resources, periodic_resources or both explicitly."
            )
        term = LearnedGraphPotentialTerm(model, name=name)
        required = frozenset(
            value.requirement_id for value in term.requirements.neighborhoods
        )
        finite = _resource_plan(finite_resources, required, "finite_resources")
        periodic = _resource_plan(periodic_resources, required, "periodic_resources")
        if periodic is not None:
            if periodic.frame is not None:
                raise ValueError(
                    "periodic_resources are searched in each system cell; a frame is "
                    "a finite-system search domain."
                )
            declared = model.capabilities
            if not (declared.orthorhombic_periodic or declared.triclinic_periodic):
                raise ValueError(
                    "periodic_resources require a model with periodic execution."
                )
        revision = atomistic_model_revision(model).revision_id
        self.model = model
        self.graph_execution = graph_execution
        self.finite_resources = finite
        self.periodic_resources = periodic
        self.name = term.name
        self.model_revision_id = revision
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-atomistic-provider-plan",
                "model_revision": revision,
                "name": term.name,
                "graph_execution": graph_execution.plan_id,
                "finite_resources": None if finite is None else finite.plan_id,
                "periodic_resources": None if periodic is None else periodic.plan_id,
            }
        )

    def prepare(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
    ) -> NativeAtomisticProvider:
        """Prepare the provider for ``system`` with its declared resource plan.

        ``conditioning`` and ``key`` are bound on the learned term, so they
        enter the term, program and provider identities.
        """

        periodic = system.cell is not None
        resources = self.periodic_resources if periodic else self.finite_resources
        if resources is None:
            raise ValueError(
                "This provider recipe declares no "
                f"{'periodic' if periodic else 'finite'} resources."
            )
        term = LearnedGraphPotentialTerm(
            self.model,
            name=self.name,
            allow_periodic=periodic,
            conditioning=conditioning,
            key=key,
        )
        program = AtomisticPotentialProgram([term]).prepare(
            system,
            neighborhood_bundle=resources,
            graph_execution=self.graph_execution,
        )
        return NativeAtomisticProvider(program)


class CallableBornOppenheimerProvider(AbstractExternalAtomisticProvider):
    """Explicit host-provider boundary for one Born–Oppenheimer surface.

    A labelled provider: ``provider_label`` is the static declared label
    (``provider_identity`` is ``None``) and every evaluator result must carry
    exactly that label.
    """

    evaluator: ElectronicEvaluator
    provider_identity: IdentityWords | None
    provider_label: str | None = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    def __init__(
        self,
        evaluator: ElectronicEvaluator,
        provider_label: str,
        /,
        *,
        conservative: bool = True,
        differentiable: bool = False,
    ) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        identifier = str(provider_label).strip()
        if not identifier:
            raise ValueError("provider_label must be non-empty.")
        self.evaluator = evaluator
        self.provider_identity = None
        self.provider_label = identifier
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
        if (
            result.provider_identity is not None
            or result.provider_label != self.provider_label
        ):
            raise ValueError("Born–Oppenheimer result provider binding changed.")
        if result.forces.shape != expected or result.energy.shape != ():
            raise ValueError("Born–Oppenheimer energy or force shape is invalid.")
        return result


class BornOppenheimerState(StrictModule):
    """Born–Oppenheimer phase point with its cached provider evaluation.

    PyTree: the provider binding is the provider's exclusive pair, dynamic
    ``provider_identity`` words of a digest provider or the static declared
    ``provider_label`` of a labelled provider (the other is ``None``), so
    states of every material/parameter revision of one layout share one
    treedef; the host ``provider_id`` returns the label or decoded digest.
    """

    positions: Array
    momenta: Array
    forces: Array
    energy: Array
    time: Array
    step_index: Array
    successful: Array
    provider_identity: IdentityWords | None
    provider_label: str | None = eqx.field(static=True)

    @property
    def provider_id(self) -> str:
        """Host string of the bound provider: its label or decoded digest."""
        return _provider_id(self.provider_identity, self.provider_label)


class BornOppenheimerStep(StrictModule):
    state: BornOppenheimerState
    initial_evaluation: ExternalAtomisticEvaluation
    final_evaluation: ExternalAtomisticEvaluation
    successful: Array


class BornOppenheimerVelocityVerletPlan(StrictModule, NonTrainableState):
    """Velocity Verlet over one provider boundary.

    PyTree: ``plan_identity`` is the dynamic exact plan digest (system,
    provider and step size; host ``plan_id``), so the plans of every
    material/parameter revision of one layout share one executable. Digest
    bindings are checked numerically by ``step``. A mismatched static provider
    label is refused before invoking its potentially external evaluator.
    """

    system: PreparedAtomisticSystem
    provider: AbstractExternalAtomisticProvider
    step_size: float = eqx.field(static=True)
    plan_identity: IdentityWords

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
        self.plan_identity = IdentityWords.of(
            {
                "kind": "born-oppenheimer-velocity-verlet",
                "system": system.prepared_id,
                "provider": provider.provider_id,
                "step_size": step,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of ``plan_identity``."""
        return self.plan_identity.fingerprint()

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
        successful = (
            evaluation.successful
            & evaluation.bound_to(self.provider)
            & jnp.all(jnp.isfinite(momenta))
        )
        return BornOppenheimerState(
            position,
            momenta,
            evaluation.forces,
            evaluation.energy,
            jnp.asarray(time, dtype=position.dtype),
            jnp.zeros((), dtype=jnp.int32),
            successful,
            self.provider.provider_identity,
            self.provider.provider_label,
        )

    def step(self, state: BornOppenheimerState, /) -> BornOppenheimerStep:
        """Advance a bound state; reject foreign labels before external evaluation."""
        if state.provider_label != self.provider.provider_label:
            raise ValueError("Born–Oppenheimer state belongs to another provider.")
        bound = _same_provider(
            state.provider_identity,
            state.provider_label,
            self.provider.provider_identity,
            self.provider.provider_label,
        )
        initial = ExternalAtomisticEvaluation(
            state.energy,
            state.forces,
            None,
            state.successful,
            provider_identity=state.provider_identity,
            provider_label=state.provider_label,
        )
        dt = jnp.asarray(self.step_size, dtype=state.positions.dtype)
        force_scale = self.system.plan.units.force_to_momentum_rate
        half = state.momenta + 0.5 * dt * force_scale * state.forces
        position = state.positions + dt * half * self.system.inverse_masses[:, None]
        cell = None if self.system.cell is None else self.system.cell.vectors
        final = self.provider.evaluate(self.system, position, cell)
        momentum = half + 0.5 * dt * force_scale * final.forces
        successful = (
            state.successful
            & bound
            & final.successful
            & final.bound_to(self.provider)
            & jnp.all(jnp.isfinite(momentum))
        )
        # The successor keeps the state's binding (equal to the provider's
        # whenever it is accepted), so both branches share one treedef.
        successor = BornOppenheimerState(
            position,
            momentum,
            final.forces,
            final.energy,
            state.time + dt,
            state.step_index + 1,
            successful,
            state.provider_identity,
            state.provider_label,
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
    "NativeAtomisticRecovery",
]
