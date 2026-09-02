#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity spatial decomposition and execution for atomistic programs.

The local-reference runtime represents every logical shard with fixed-shape JAX
arrays. It is both an executable single-device implementation and the
scientific reference for collective implementations. Collective runtimes have
no implicit communication fallback: callers must prepare them with explicit
JAX exchange and reduction callables.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleDomainDecompositionPlan, ParticleHaloState
from ._constraints import PreparedDistanceConstraints
from ._potential_program import (
    AtomisticPotentialEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._system import PreparedAtomisticSystem


DistributedExecutionMode: TypeAlias = Literal["local-reference", "collective"]
DistributedReductionMode: TypeAlias = Literal["fast", "deterministic", "compensated"]
DistributedPhase: TypeAlias = Literal[
    "direct", "sparse-correction", "reciprocal", "reduction"
]


class DistributedOutputMask(StrictModule, NonTrainableState):
    """Static selection of fixed-shape evaluation outputs."""

    energy: bool = eqx.field(static=True)
    forces: bool = eqx.field(static=True)
    virial: bool = eqx.field(static=True)
    atom_energy: bool = eqx.field(static=True)
    partition_energy: bool = eqx.field(static=True)
    mask_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        energy: bool = True,
        forces: bool = True,
        virial: bool = True,
        atom_energy: bool = False,
        partition_energy: bool = True,
    ):
        values = (energy, forces, virial, atom_energy, partition_energy)
        if any(not isinstance(value, (bool, np.bool_)) for value in values):
            raise TypeError("Distributed output requests must be booleans.")
        self.energy = bool(energy)
        self.forces = bool(forces)
        self.virial = bool(virial)
        self.atom_energy = bool(atom_energy)
        self.partition_energy = bool(partition_energy)
        self.mask_id = canonical_fingerprint(
            {
                "kind": "distributed-atomistic-output-mask",
                "energy": self.energy,
                "forces": self.forces,
                "virial": self.virial,
                "atom_energy": self.atom_energy,
                "partition_energy": self.partition_energy,
            }
        )


class DistributedReductionPolicy(StrictModule, NonTrainableState):
    """Reduction order used by local and collective execution."""

    mode: DistributedReductionMode = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, mode: DistributedReductionMode = "deterministic", /):
        if mode not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown distributed reduction mode.")
        self.mode = mode
        self.policy_id = canonical_fingerprint(
            {"kind": "distributed-atomistic-reduction", "mode": mode}
        )


class DistributedCollectiveOperations(StrictModule, NonTrainableState):
    """Explicit JAX communication callables for a collective runtime.

    ``exchange`` receives padded values in ``(source, destination, slot, ...)``
    order plus a boolean route mask and returns values in
    ``(destination, source, slot, ...)`` order. ``reverse_exchange`` performs
    the inverse communication for halo-force return. ``reduce_sum`` performs a
    global sum for an arbitrary fixed-shape JAX array.
    """

    exchange: Callable[[Array, Array], Array] = eqx.field(static=True)
    reverse_exchange: Callable[[Array, Array], Array] = eqx.field(static=True)
    reduce_sum: Callable[[Array], Array] = eqx.field(static=True)
    partition_index: int = eqx.field(static=True)
    collective_id: str = eqx.field(static=True)

    def __init__(
        self,
        exchange: Callable[[Array, Array], Array],
        reverse_exchange: Callable[[Array, Array], Array],
        reduce_sum: Callable[[Array], Array],
        /,
        *,
        partition_index: int,
        collective_id: str,
    ):
        if (
            not callable(exchange)
            or not callable(reverse_exchange)
            or not callable(reduce_sum)
        ):
            raise TypeError("Distributed collective operations must be callable.")
        if (
            isinstance(partition_index, (bool, np.bool_))
            or not isinstance(partition_index, (int, np.integer))
            or int(partition_index) < 0
        ):
            raise ValueError("partition_index must be a non-negative integer.")
        identifier = str(collective_id).strip()
        if not identifier:
            raise ValueError("collective_id must be non-empty.")
        self.exchange = exchange
        self.reverse_exchange = reverse_exchange
        self.reduce_sum = reduce_sum
        self.partition_index = int(partition_index)
        self.collective_id = identifier


class DistributedPMEPlan(StrictModule, NonTrainableState):
    """Static pencil/slab contract for a distributed reciprocal mesh."""

    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    interpolation_order: int = eqx.field(static=True)
    decomposition_axis: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid_shape: Sequence[int],
        /,
        *,
        interpolation_order: int = 4,
        decomposition_axis: int = 0,
    ):
        shape = tuple(grid_shape)
        if len(shape) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in shape
        ):
            raise ValueError("Distributed PME grid_shape must contain three positives.")
        if (
            isinstance(interpolation_order, (bool, np.bool_))
            or not isinstance(interpolation_order, (int, np.integer))
            or not 2 <= int(interpolation_order) <= 8
        ):
            raise ValueError("PME interpolation_order must be between two and eight.")
        if (
            isinstance(decomposition_axis, (bool, np.bool_))
            or not isinstance(decomposition_axis, (int, np.integer))
            or int(decomposition_axis) not in (0, 1, 2)
        ):
            raise ValueError("PME decomposition_axis must be zero, one, or two.")
        self.grid_shape = tuple(int(value) for value in shape)
        self.interpolation_order = int(interpolation_order)
        self.decomposition_axis = int(decomposition_axis)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-pme",
                "grid_shape": list(self.grid_shape),
                "interpolation_order": self.interpolation_order,
                "decomposition_axis": self.decomposition_axis,
            }
        )

    def prepare(
        self, runtime: "PreparedDistributedAtomisticRuntime", /
    ) -> "PreparedDistributedPME":
        if not isinstance(runtime, PreparedDistributedAtomisticRuntime):
            raise TypeError("Distributed PME requires a prepared distributed runtime.")
        partitions = runtime.plan.decomposition.partitions
        extent = self.grid_shape[self.decomposition_axis]
        if extent < partitions:
            raise ValueError(
                "Distributed PME decomposition axis must cover every partition."
            )
        quotient, remainder = divmod(extent, partitions)
        widths = np.asarray(
            [quotient + int(partition < remainder) for partition in range(partitions)],
            dtype=np.int32,
        )
        bounds = np.concatenate((np.zeros((1,), np.int32), np.cumsum(widths)))
        return PreparedDistributedPME(
            self,
            runtime.runtime_id,
            jnp.asarray(bounds),
            canonical_fingerprint(
                {
                    "kind": "prepared-distributed-pme",
                    "plan": self.plan_id,
                    "runtime": runtime.runtime_id,
                    "bounds": bounds.tolist(),
                }
            ),
        )


class PreparedDistributedPME(StrictModule, NonTrainableState):
    """Prepared fixed mesh ownership bounds for one distributed runtime."""

    plan: DistributedPMEPlan
    runtime_id: str = eqx.field(static=True)
    mesh_bounds: Array
    prepared_id: str = eqx.field(static=True)


class DistributedPolarizationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity convergence contract for distributed polarization."""

    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 100,
        tolerance: float = 1.0e-7,
    ):
        if (
            isinstance(maximum_iterations, (bool, np.bool_))
            or not isinstance(maximum_iterations, (int, np.integer))
            or int(maximum_iterations) <= 0
        ):
            raise ValueError("maximum_iterations must be a positive integer.")
        if isinstance(tolerance, (bool, np.bool_)) or not isinstance(
            tolerance, (int, float, np.integer, np.floating)
        ):
            raise TypeError("Distributed polarization tolerance must be real.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Distributed polarization tolerance must be positive.")
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-polarization",
                "maximum_iterations": self.maximum_iterations,
                "tolerance": self.tolerance,
            }
        )

    def prepare(
        self, runtime: "PreparedDistributedAtomisticRuntime", /
    ) -> "PreparedDistributedPolarization":
        if not isinstance(runtime, PreparedDistributedAtomisticRuntime):
            raise TypeError(
                "Distributed polarization requires a prepared distributed runtime."
            )
        return PreparedDistributedPolarization(
            self,
            runtime.runtime_id,
            runtime.plan.system.capacity,
            canonical_fingerprint(
                {
                    "kind": "prepared-distributed-polarization",
                    "plan": self.plan_id,
                    "runtime": runtime.runtime_id,
                    "capacity": runtime.plan.system.capacity,
                }
            ),
        )


class PreparedDistributedPolarization(StrictModule, NonTrainableState):
    """Prepared polarization warm-start and convergence identity."""

    plan: DistributedPolarizationPlan
    runtime_id: str = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class DistributedPolarizationEvidence(StrictModule):
    """State-bound residual and bounded-iteration polarization evidence."""

    iterations: Array
    residual: Array
    finite: Array
    converged: Array
    successful: Array
    source_positions: Array
    source_cell: Array
    source_step_index: Array
    source_decomposition_epoch: Array
    prepared_id: str = eqx.field(static=True)
    source_run_id: str = eqx.field(static=True)
    source_replica_id: str = eqx.field(static=True)
    source_epoch_id: str = eqx.field(static=True)


def certify_distributed_polarization(
    prepared: PreparedDistributedPolarization,
    state: "DistributedAtomisticState",
    dipoles: ArrayLike,
    residual: ArrayLike,
    iterations: ArrayLike,
    /,
) -> DistributedPolarizationEvidence:
    if not isinstance(prepared, PreparedDistributedPolarization):
        raise TypeError("prepared must be PreparedDistributedPolarization.")
    if not isinstance(state, DistributedAtomisticState):
        raise TypeError("state must be DistributedAtomisticState.")
    if state.runtime_id != prepared.runtime_id:
        raise ValueError("Polarization state belongs to another prepared runtime.")
    dipole = jnp.asarray(dipoles)
    if dipole.shape != (prepared.particle_capacity, 3):
        raise ValueError("Distributed dipoles changed prepared capacity.")
    residual_ = jnp.asarray(residual)
    iterations_input = jnp.asarray(iterations)
    if residual_.shape or iterations_input.shape:
        raise ValueError("Polarization residual and iterations must be scalars.")
    if not jnp.issubdtype(iterations_input.dtype, jnp.integer):
        raise TypeError("Polarization iterations must have an integral dtype.")
    iterations_ = iterations_input.astype(jnp.int32)
    finite = (
        jnp.all(jnp.isfinite(dipole))
        & jnp.isfinite(residual_)
        & (residual_ >= 0)
        & (iterations_ >= 0)
    )
    converged = (residual_ <= prepared.plan.tolerance) & (
        iterations_ <= prepared.plan.maximum_iterations
    )
    successful = finite & converged & state.successful
    return DistributedPolarizationEvidence(
        iterations_,
        residual_,
        finite,
        converged,
        successful,
        state.positions,
        state.cell,
        state.step_index,
        state.decomposition_epoch,
        prepared.prepared_id,
        state.run_id,
        state.replica_id,
        state.epoch_id,
    )


class DistributedSpatialDecomposition(StrictModule, NonTrainableState):
    """Prepared ownership, permutation, and padded communication routes."""

    owner: Array
    permutation: Array
    inverse_permutation: Array
    block_bounds: Array
    owned_indices: Array
    owned_mask: Array
    halo_send_indices: Array
    halo_send_mask: Array
    halo_receive_indices: Array
    halo_receive_mask: Array
    route_counts: Array
    local_indices: Array
    local_mask: Array
    full_owned_mask: Array
    full_halo_mask: Array
    owned_counts: Array
    halo_counts: Array
    ownership_overflow: Array
    halo_overflow: Array
    nonfinite: Array
    outside_domain: Array
    migration_count: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    execution_mode: DistributedExecutionMode = eqx.field(static=True)


class DistributedExecutionStatus(StrictModule):
    """Fail-closed numerical, capacity, convergence, and communication status."""

    finite: Array
    ownership_capacity_ok: Array
    halo_capacity_ok: Array
    migration_capacity_ok: Array
    reciprocal_converged: Array
    polarization_converged: Array
    collective_supported: Array
    successful: Array


class DistributedAtomisticPlan(StrictModule, NonTrainableState):
    """Fixed-capacity atomistic domain-decomposition plan."""

    system: PreparedAtomisticSystem
    decomposition: ParticleDomainDecompositionPlan
    output_mask: DistributedOutputMask
    reduction: DistributedReductionPolicy
    pme: DistributedPMEPlan | None
    polarization: DistributedPolarizationPlan | None
    partition_capacity: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    migration_capacity: int = eqx.field(static=True)
    thermostat_capacity: int = eqx.field(static=True)
    barostat_capacity: int = eqx.field(static=True)
    bias_capacity: int = eqx.field(static=True)
    execution_mode: DistributedExecutionMode = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        decomposition: ParticleDomainDecompositionPlan,
        /,
        *,
        partition_capacity: int | None = None,
        halo_capacity: int | None = None,
        migration_capacity: int | None = None,
        thermostat_capacity: int = 0,
        barostat_capacity: int = 0,
        bias_capacity: int = 0,
        output_mask: DistributedOutputMask | None = None,
        reduction: DistributedReductionPolicy | None = None,
        pme: DistributedPMEPlan | None = None,
        polarization: DistributedPolarizationPlan | None = None,
        execution_mode: DistributedExecutionMode = "local-reference",
    ):
        if not isinstance(system, PreparedAtomisticSystem) or not isinstance(
            decomposition, ParticleDomainDecompositionPlan
        ):
            raise TypeError(
                "Distributed atomistics requires prepared system and decomposition."
            )
        if decomposition.box.ambient_dimension != 3:
            raise ValueError("Distributed atomistic decomposition must be 3D.")
        if execution_mode not in ("local-reference", "collective"):
            raise ValueError("Unknown distributed execution mode.")
        if execution_mode == "collective" and decomposition.partitions < 2:
            raise ValueError("Collective execution requires at least two partitions.")

        def capacity(name: str, value: int | None, default: int, *, positive: bool):
            resolved = default if value is None else value
            if (
                isinstance(resolved, (bool, np.bool_))
                or not isinstance(resolved, (int, np.integer))
                or int(resolved) < int(positive)
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be a {qualifier} integer.")
            return int(resolved)

        partition_capacity_ = capacity(
            "partition_capacity", partition_capacity, system.capacity, positive=True
        )
        halo_capacity_ = capacity(
            "halo_capacity", halo_capacity, system.capacity, positive=False
        )
        migration_capacity_ = capacity(
            "migration_capacity", migration_capacity, system.capacity, positive=False
        )
        thermostat_capacity_ = capacity(
            "thermostat_capacity", thermostat_capacity, 0, positive=False
        )
        barostat_capacity_ = capacity(
            "barostat_capacity", barostat_capacity, 0, positive=False
        )
        bias_capacity_ = capacity("bias_capacity", bias_capacity, 0, positive=False)
        output_mask_ = DistributedOutputMask() if output_mask is None else output_mask
        reduction_ = DistributedReductionPolicy() if reduction is None else reduction
        if not isinstance(output_mask_, DistributedOutputMask):
            raise TypeError("output_mask must be DistributedOutputMask or None.")
        if not isinstance(reduction_, DistributedReductionPolicy):
            raise TypeError("reduction must be DistributedReductionPolicy or None.")
        if pme is not None and not isinstance(pme, DistributedPMEPlan):
            raise TypeError("pme must be DistributedPMEPlan or None.")
        if polarization is not None and not isinstance(
            polarization, DistributedPolarizationPlan
        ):
            raise TypeError("polarization must be DistributedPolarizationPlan or None.")
        self.system = system
        self.decomposition = decomposition
        self.partition_capacity = partition_capacity_
        self.halo_capacity = halo_capacity_
        self.migration_capacity = migration_capacity_
        self.thermostat_capacity = thermostat_capacity_
        self.barostat_capacity = barostat_capacity_
        self.bias_capacity = bias_capacity_
        self.output_mask = output_mask_
        self.reduction = reduction_
        self.pme = pme
        self.polarization = polarization
        self.execution_mode = execution_mode
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-atomistic",
                "system": system.prepared_id,
                "decomposition": decomposition.plan_id,
                "partition_capacity": partition_capacity_,
                "halo_capacity": halo_capacity_,
                "migration_capacity": migration_capacity_,
                "thermostat_capacity": thermostat_capacity_,
                "barostat_capacity": barostat_capacity_,
                "bias_capacity": bias_capacity_,
                "output_mask": output_mask_.mask_id,
                "reduction": reduction_.policy_id,
                "pme": None if pme is None else pme.plan_id,
                "polarization": None if polarization is None else polarization.plan_id,
                "execution_mode": execution_mode,
            }
        )

    def prepare_runtime(
        self,
        collectives: DistributedCollectiveOperations | None = None,
        /,
    ) -> "PreparedDistributedAtomisticRuntime":
        if self.execution_mode == "collective":
            if not isinstance(collectives, DistributedCollectiveOperations):
                raise ValueError(
                    "Collective distributed execution requires explicit JAX collectives."
                )
            if collectives.partition_index >= self.decomposition.partitions:
                raise ValueError(
                    "Collective partition_index exceeds decomposition partitions."
                )
        elif collectives is not None:
            raise ValueError("Local-reference execution does not accept collectives.")
        collective_id = (
            "local-reference" if collectives is None else collectives.collective_id
        )
        runtime = PreparedDistributedAtomisticRuntime(
            self,
            collectives,
            canonical_fingerprint(
                {
                    "kind": "prepared-distributed-atomistic",
                    "plan": self.plan_id,
                    "collectives": collective_id,
                    "partition_index": (
                        None if collectives is None else collectives.partition_index
                    ),
                }
            ),
        )
        if self.pme is not None:
            self.pme.prepare(runtime)
        return runtime

    def prepare(self, positions: ArrayLike, /) -> "DistributedAtomisticState":
        """Initialize a local-reference state directly from the plan."""
        return self.prepare_runtime().initialize(positions)


class PreparedDistributedAtomisticRuntime(StrictModule, NonTrainableState):
    """Prepared local-reference or explicit-collective execution runtime."""

    plan: DistributedAtomisticPlan
    collectives: DistributedCollectiveOperations | None
    runtime_id: str = eqx.field(static=True)

    def initialize(
        self,
        positions: ArrayLike,
        /,
        *,
        momenta: ArrayLike | None = None,
        cell: ArrayLike | None = None,
        thermostat_state: ArrayLike | None = None,
        barostat_state: ArrayLike | None = None,
        polarization_warm_start: ArrayLike | None = None,
        bias_state: ArrayLike | None = None,
        rng_key: ArrayLike | None = None,
        step_index: ArrayLike = 0,
        decomposition_epoch: ArrayLike = 0,
        run_id: str | None = None,
        replica_id: str = "replica-0",
        epoch_id: str = "epoch-0",
    ) -> "DistributedAtomisticState":
        return _initialize_distributed_state(
            self,
            positions,
            momenta=momenta,
            cell=cell,
            thermostat_state=thermostat_state,
            barostat_state=barostat_state,
            polarization_warm_start=polarization_warm_start,
            bias_state=bias_state,
            rng_key=rng_key,
            step_index=step_index,
            decomposition_epoch=decomposition_epoch,
            run_id=run_id,
            replica_id=replica_id,
            epoch_id=epoch_id,
        )

    def pme_runtime(self, /) -> PreparedDistributedPME:
        if self.plan.pme is None:
            raise ValueError("This distributed runtime has no PME plan.")
        return self.plan.pme.prepare(self)

    def polarization_runtime(self, /) -> PreparedDistributedPolarization:
        if self.plan.polarization is None:
            raise ValueError("This distributed runtime has no polarization plan.")
        return self.plan.polarization.prepare(self)


class DistributedAtomisticState(StrictModule):
    """Complete fixed-shape physical and extended checkpoint state."""

    positions: Array
    momenta: Array
    cell: Array
    decomposition: DistributedSpatialDecomposition
    halos: ParticleHaloState
    partition_momentum: Array
    partition_energy: Array
    thermostat_state: Array
    barostat_state: Array
    polarization_warm_start: Array
    bias_state: Array
    rng_key: Array
    step_index: Array
    decomposition_epoch: Array
    status: DistributedExecutionStatus
    successful: Array
    plan_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    replica_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)


class DistributedReciprocalEvidence(StrictModule):
    """PME- and source-state-bound reciprocal evaluation evidence."""

    evaluation: AtomisticPotentialEvaluation
    finite: Array
    successful: Array
    source_positions: Array
    source_cell: Array
    source_step_index: Array
    source_decomposition_epoch: Array
    prepared_id: str = eqx.field(static=True)
    source_run_id: str = eqx.field(static=True)
    source_replica_id: str = eqx.field(static=True)
    source_epoch_id: str = eqx.field(static=True)


def certify_distributed_reciprocal(
    prepared: PreparedDistributedPME,
    state: DistributedAtomisticState,
    evaluation: AtomisticPotentialEvaluation,
    /,
) -> DistributedReciprocalEvidence:
    if not isinstance(prepared, PreparedDistributedPME):
        raise TypeError("prepared must be PreparedDistributedPME.")
    if not isinstance(state, DistributedAtomisticState):
        raise TypeError("state must be DistributedAtomisticState.")
    if not isinstance(evaluation, AtomisticPotentialEvaluation):
        raise TypeError("evaluation must be AtomisticPotentialEvaluation.")
    if state.runtime_id != prepared.runtime_id:
        raise ValueError("Reciprocal state belongs to another prepared PME runtime.")
    if (
        evaluation.forces.shape != state.positions.shape
        or evaluation.atom_energy.shape != (state.positions.shape[0],)
    ):
        raise ValueError("Reciprocal evaluation changed particle capacity.")
    finite = (
        jnp.isfinite(evaluation.energy)
        & jnp.all(jnp.isfinite(evaluation.forces))
        & jnp.all(jnp.isfinite(evaluation.virial))
        & jnp.all(jnp.isfinite(evaluation.atom_energy))
    )
    successful = finite & evaluation.successful & state.successful
    return DistributedReciprocalEvidence(
        evaluation,
        finite,
        successful,
        state.positions,
        state.cell,
        state.step_index,
        state.decomposition_epoch,
        prepared.prepared_id,
        state.run_id,
        state.replica_id,
        state.epoch_id,
    )


def _source_evidence_matches(
    state: DistributedAtomisticState,
    source_positions: Array,
    source_cell: Array,
    source_step_index: Array,
    source_decomposition_epoch: Array,
    source_run_id: str,
    source_replica_id: str,
    source_epoch_id: str,
    /,
) -> Array:
    static_match = (
        source_run_id == state.run_id
        and source_replica_id == state.replica_id
        and source_epoch_id == state.epoch_id
    )
    return (
        jnp.asarray(static_match)
        & jnp.array_equal(source_positions, state.positions)
        & jnp.array_equal(source_cell, state.cell)
        & (source_step_index == state.step_index)
        & (source_decomposition_epoch == state.decomposition_epoch)
    )


class DistributedMigrationCandidate(StrictModule):
    """Candidate decomposition that can be atomically committed or rolled back."""

    positions: Array
    decomposition: DistributedSpatialDecomposition
    halos: ParticleHaloState
    source_state: DistributedAtomisticState
    migration_indices: Array
    migration_mask: Array
    migration_count: Array
    overflow: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)


class DistributedDomainEvidence(StrictModule):
    """Per-partition ownership, halo, load, and fail-closed capacity evidence."""

    owned_particles: Array
    halo_particles: Array
    pair_work: Array
    iterative_work: Array
    weighted_work: Array
    imbalance: Array
    finite: Array
    inside_domain: Array
    ownership_capacity_ok: Array
    halo_capacity_ok: Array
    migration_capacity_ok: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)


class DistributedPhaseEvidence(StrictModule):
    """Direct, sparse-correction, reciprocal, and reduction phase evidence."""

    phase_energy: Array
    phase_successful: Array
    finite: Array
    reduction_successful: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)


class DistributedAtomisticEvaluation(StrictModule):
    """Fixed-shape masked outputs and phase/status evidence."""

    energy: Array
    forces: Array
    virial: Array
    atom_energy: Array
    partition_energy: Array
    available: Array
    phases: DistributedPhaseEvidence
    status: DistributedExecutionStatus
    successful: Array
    output_mask_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)


class DistributedAtomisticCheckpointIdentity(StrictModule, NonTrainableState):
    """Content identity covering every continuation-relevant state component."""

    plan_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    replica_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    owner_digest: str = eqx.field(static=True)
    payload_digest: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(self, state: DistributedAtomisticState, /):
        if not isinstance(state, DistributedAtomisticState):
            raise TypeError("state must be DistributedAtomisticState.")
        owner_record = array_tree_fingerprint(
            {
                "owner": state.decomposition.owner,
                "permutation": state.decomposition.permutation,
                "block_bounds": state.decomposition.block_bounds,
            }
        )
        payload_record = array_tree_fingerprint({"state": state})
        owner_digest = str(owner_record["sha256"])
        payload_digest = str(payload_record["sha256"])
        self.plan_id = state.plan_id
        self.runtime_id = state.runtime_id
        self.run_id = state.run_id
        self.replica_id = state.replica_id
        self.epoch_id = state.epoch_id
        self.owner_digest = owner_digest
        self.payload_digest = payload_digest
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "distributed-atomistic-checkpoint",
                "plan": state.plan_id,
                "runtime": state.runtime_id,
                "run": state.run_id,
                "replica": state.replica_id,
                "epoch": state.epoch_id,
                "owner": owner_digest,
                "payload": payload_digest,
            }
        )


class DistributedAtomisticCheckpoint(StrictModule, NonTrainableState):
    """In-memory exact continuation checkpoint and its content identity."""

    state: DistributedAtomisticState
    identity: DistributedAtomisticCheckpointIdentity


def _fixed_indices(mask: Array, capacity: int) -> Array:
    if capacity == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    return jnp.nonzero(mask, size=capacity, fill_value=-1)[0].astype(jnp.int32)


def _prepare_spatial_decomposition(
    plan: DistributedAtomisticPlan,
    positions: Array,
    /,
    *,
    migration_count: Array | None = None,
) -> DistributedSpatialDecomposition:
    partitions = plan.decomposition.partitions
    particle_capacity = plan.system.capacity
    active = jnp.asarray(plan.system.active_mask, bool)
    box = plan.decomposition.box
    finite_particle = jnp.all(jnp.isfinite(positions), axis=1)
    finite = jnp.all(jnp.where(active, finite_particle, True))
    safe_position = jnp.where(jnp.isfinite(positions), positions, box.lower)
    inside_axis = (safe_position >= box.lower) & (safe_position <= box.upper)
    inside_axis = inside_axis | box.periodic_mask
    inside_particle = jnp.all(inside_axis, axis=1)
    inside_domain = jnp.all(jnp.where(active, inside_particle, True))

    coordinate = safe_position[:, 0]
    if box.periodic_axes[0]:
        coordinate = box.lower[0] + jnp.mod(coordinate - box.lower[0], box.lengths[0])
    relative = (coordinate - box.lower[0]) / box.lengths[0]
    owner = jnp.floor(relative * partitions).astype(jnp.int32)
    owner = jnp.clip(owner, 0, partitions - 1)
    owner = jnp.where(active, owner, -1)
    full_owned = (
        jnp.arange(partitions, dtype=jnp.int32)[:, None] == owner[None, :]
    ) & active[None, :]
    owned_counts = jnp.sum(full_owned, axis=1, dtype=jnp.int32)

    sorting_key = jnp.where(active, owner, partitions)
    permutation = jnp.argsort(sorting_key, stable=True).astype(jnp.int32)
    inverse_permutation = (
        jnp.zeros((particle_capacity,), jnp.int32)
        .at[permutation]
        .set(jnp.arange(particle_capacity, dtype=jnp.int32))
    )
    block_bounds = jnp.concatenate(
        (jnp.zeros((1,), jnp.int32), jnp.cumsum(owned_counts, dtype=jnp.int32))
    )
    owned_indices = jax.vmap(lambda mask: _fixed_indices(mask, plan.partition_capacity))(
        full_owned
    )
    owned_mask = owned_indices >= 0

    edges = (
        box.lower[0]
        + box.lengths[0] * jnp.arange(partitions + 1, dtype=positions.dtype) / partitions
    )

    def interval_distance(value: Array) -> Array:
        return jnp.maximum(
            jnp.maximum(
                edges[:-1, None] - value[None, :],
                value[None, :] - edges[1:, None],
            ),
            0.0,
        )

    distance = interval_distance(coordinate)
    if box.periodic_axes[0]:
        distance = jnp.minimum(
            distance,
            jnp.minimum(
                interval_distance(coordinate - box.lengths[0]),
                interval_distance(coordinate + box.lengths[0]),
            ),
        )
    destination_near = distance <= plan.decomposition.halo_radius
    off_diagonal = ~jnp.eye(partitions, dtype=bool)
    full_routes = (
        full_owned[:, None, :] & destination_near[None, :, :] & off_diagonal[:, :, None]
    )
    route_counts = jnp.sum(full_routes, axis=2, dtype=jnp.int32)
    flattened_routes = full_routes.reshape((partitions * partitions, particle_capacity))
    send_indices = jax.vmap(lambda mask: _fixed_indices(mask, plan.halo_capacity))(
        flattened_routes
    ).reshape((partitions, partitions, plan.halo_capacity))
    send_mask = send_indices >= 0
    receive_indices = jnp.swapaxes(send_indices, 0, 1)
    receive_mask = jnp.swapaxes(send_mask, 0, 1)
    full_halo = jnp.any(full_routes, axis=0)
    halo_counts = jnp.sum(full_halo, axis=1, dtype=jnp.int32)
    incoming_indices = receive_indices.reshape(
        (partitions, partitions * plan.halo_capacity)
    )
    incoming_mask = receive_mask.reshape((partitions, partitions * plan.halo_capacity))
    local_indices = jnp.concatenate((owned_indices, incoming_indices), axis=1)
    local_mask = jnp.concatenate((owned_mask, incoming_mask), axis=1)

    ownership_overflow = jnp.any(owned_counts > plan.partition_capacity)
    halo_overflow = jnp.any(route_counts > plan.halo_capacity)
    successful = finite & inside_domain & ~ownership_overflow & ~halo_overflow
    migration_count_ = (
        jnp.zeros((), jnp.int32)
        if migration_count is None
        else jnp.asarray(migration_count, jnp.int32)
    )
    return DistributedSpatialDecomposition(
        owner,
        permutation,
        inverse_permutation,
        block_bounds,
        owned_indices,
        owned_mask,
        send_indices,
        send_mask,
        receive_indices,
        receive_mask,
        route_counts,
        local_indices,
        local_mask,
        full_owned,
        full_halo,
        owned_counts,
        halo_counts,
        ownership_overflow,
        halo_overflow,
        ~finite,
        ~inside_domain,
        migration_count_,
        successful,
        plan.plan_id,
        plan.execution_mode,
    )


def _particle_halo_state(
    decomposition: DistributedSpatialDecomposition, /
) -> ParticleHaloState:
    return ParticleHaloState(
        decomposition.owner,
        decomposition.full_owned_mask,
        decomposition.full_halo_mask,
        decomposition.full_owned_mask | decomposition.full_halo_mask,
        decomposition.migration_count,
        jnp.sum(decomposition.full_halo_mask, dtype=jnp.int32),
        decomposition.successful,
    )


def _state_vector(
    name: str, value: ArrayLike | None, capacity: int, dtype: jnp.dtype
) -> Array:
    array = jnp.zeros((capacity,), dtype=dtype) if value is None else jnp.asarray(value)
    if array.shape != (capacity,):
        raise ValueError(f"{name} must match its fixed prepared capacity.")
    return array.astype(dtype)


def _scalar_int(name: str, value: ArrayLike) -> Array:
    array = jnp.asarray(value, jnp.int32)
    if array.shape:
        raise ValueError(f"{name} must be a scalar.")
    return array


def _nonempty_identity(name: str, value: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _ordered_sum(value: Array, policy: DistributedReductionPolicy) -> Array:
    array = jnp.asarray(value)
    if array.ndim == 0:
        return array
    if policy.mode == "fast":
        return jnp.sum(array, axis=0)
    initial = jnp.zeros(array.shape[1:], dtype=array.dtype)
    if policy.mode == "deterministic":
        return jax.lax.fori_loop(
            0, array.shape[0], lambda index, total: total + array[index], initial
        )

    def compensated_step(index, carry):
        total, correction = carry
        increment = array[index] - correction
        updated = total + increment
        correction = (updated - total) - increment
        return updated, correction

    total, _ = jax.lax.fori_loop(
        0,
        array.shape[0],
        compensated_step,
        (initial, jnp.zeros_like(initial)),
    )
    return total


def _partition_vector_sum(
    values: Array, owned: Array, policy: DistributedReductionPolicy
) -> Array:
    return jax.vmap(
        lambda mask: _ordered_sum(jnp.where(mask[:, None], values, 0), policy)
    )(owned)


def _initialize_distributed_state(
    runtime: PreparedDistributedAtomisticRuntime,
    positions: ArrayLike,
    /,
    *,
    momenta: ArrayLike | None,
    cell: ArrayLike | None,
    thermostat_state: ArrayLike | None,
    barostat_state: ArrayLike | None,
    polarization_warm_start: ArrayLike | None,
    bias_state: ArrayLike | None,
    rng_key: ArrayLike | None,
    step_index: ArrayLike,
    decomposition_epoch: ArrayLike,
    run_id: str | None,
    replica_id: str,
    epoch_id: str,
) -> DistributedAtomisticState:
    plan = runtime.plan
    coordinate = jnp.asarray(positions)
    expected = (plan.system.capacity, 3)
    if coordinate.shape != expected:
        raise ValueError("Distributed positions must match atomistic capacity.")
    momentum = jnp.zeros_like(coordinate) if momenta is None else jnp.asarray(momenta)
    if momentum.shape != expected:
        raise ValueError("Distributed momenta must match atomistic capacity.")
    cell_ = (
        jnp.diag(plan.decomposition.box.lengths.astype(coordinate.dtype))
        if cell is None
        else jnp.asarray(cell)
    )
    if cell_.shape != (3, 3):
        raise ValueError("Distributed physical cell must have shape (3, 3).")
    thermostat = _state_vector(
        "thermostat_state", thermostat_state, plan.thermostat_capacity, coordinate.dtype
    )
    barostat = _state_vector(
        "barostat_state", barostat_state, plan.barostat_capacity, coordinate.dtype
    )
    bias = _state_vector("bias_state", bias_state, plan.bias_capacity, coordinate.dtype)
    polarization = (
        jnp.zeros_like(coordinate)
        if polarization_warm_start is None
        else jnp.asarray(polarization_warm_start)
    )
    if polarization.shape != expected:
        raise ValueError("Polarization warm start must match atomistic capacity.")
    key = (
        jnp.zeros((2,), dtype=jnp.uint32)
        if rng_key is None
        else jax.random.key_data(jnp.asarray(rng_key)).astype(jnp.uint32)
    )
    if key.shape != (2,):
        raise ValueError("Distributed RNG key must have canonical uint32 shape (2,).")
    step = _scalar_int("step_index", step_index)
    epoch = _scalar_int("decomposition_epoch", decomposition_epoch)
    decomposition = _prepare_spatial_decomposition(plan, coordinate)
    halos = _particle_halo_state(decomposition)
    partition_momentum = _partition_vector_sum(
        momentum, decomposition.full_owned_mask, plan.reduction
    )
    finite_extended = (
        jnp.all(jnp.isfinite(momentum))
        & jnp.all(jnp.isfinite(cell_))
        & jnp.all(jnp.isfinite(thermostat))
        & jnp.all(jnp.isfinite(barostat))
        & jnp.all(jnp.isfinite(polarization))
        & jnp.all(jnp.isfinite(bias))
    )
    collective_supported = jnp.asarray(
        plan.execution_mode == "local-reference" or runtime.collectives is not None
    )
    successful = decomposition.successful & finite_extended & collective_supported
    status = DistributedExecutionStatus(
        ~decomposition.nonfinite & finite_extended,
        ~decomposition.ownership_overflow,
        ~decomposition.halo_overflow,
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
        collective_supported,
        successful,
    )
    run = (
        canonical_fingerprint(
            {"kind": "distributed-atomistic-run", "runtime": runtime.runtime_id}
        )
        if run_id is None
        else _nonempty_identity("run_id", run_id)
    )
    replica = _nonempty_identity("replica_id", replica_id)
    epoch_identity = _nonempty_identity("epoch_id", epoch_id)
    return DistributedAtomisticState(
        coordinate,
        momentum,
        cell_,
        decomposition,
        halos,
        partition_momentum,
        jnp.zeros((plan.decomposition.partitions,), coordinate.dtype),
        thermostat,
        barostat,
        polarization,
        bias,
        key,
        step,
        epoch,
        status,
        successful,
        plan.plan_id,
        runtime.runtime_id,
        run,
        replica,
        epoch_identity,
    )


def propose_distributed_migration(
    plan: DistributedAtomisticPlan,
    state: DistributedAtomisticState,
    positions: ArrayLike,
    /,
) -> DistributedMigrationCandidate:
    if not isinstance(plan, DistributedAtomisticPlan) or not isinstance(
        state, DistributedAtomisticState
    ):
        raise TypeError("Migration requires distributed plan and state.")
    if state.plan_id != plan.plan_id:
        raise ValueError("Distributed state belongs to another plan.")
    if plan.execution_mode == "collective":
        raise ValueError(
            "Collective migration requires continuation-payload communication "
            "and is not supported by this runtime."
        )
    coordinate = jnp.asarray(positions)
    if coordinate.shape != state.positions.shape:
        raise ValueError("Migrated distributed positions changed shape.")
    active = jnp.asarray(plan.system.active_mask, bool)
    provisional = _prepare_spatial_decomposition(plan, coordinate)
    changed = active & (provisional.owner != state.decomposition.owner)
    migration_count = jnp.sum(changed, dtype=jnp.int32)
    decomposition = _prepare_spatial_decomposition(
        plan, coordinate, migration_count=migration_count
    )
    indices = _fixed_indices(changed, plan.migration_capacity)
    migration_mask = indices >= 0
    overflow = migration_count > plan.migration_capacity
    finite = ~decomposition.nonfinite
    successful = decomposition.successful & ~overflow & finite
    return DistributedMigrationCandidate(
        coordinate,
        decomposition,
        _particle_halo_state(decomposition),
        state,
        indices,
        migration_mask,
        migration_count,
        overflow,
        finite,
        successful,
        plan.plan_id,
        state.runtime_id,
    )


def _select_decomposition(
    predicate: Array,
    candidate: DistributedSpatialDecomposition,
    previous: DistributedSpatialDecomposition,
    /,
) -> DistributedSpatialDecomposition:
    values = [
        jnp.where(predicate, new, old)
        for new, old in zip(
            jax.tree.leaves(candidate), jax.tree.leaves(previous), strict=True
        )
    ]
    return jax.tree.unflatten(jax.tree.structure(previous), values)


def _select_halos(
    predicate: Array, candidate: ParticleHaloState, previous: ParticleHaloState, /
) -> ParticleHaloState:
    values = [
        jnp.where(predicate, new, old)
        for new, old in zip(
            jax.tree.leaves(candidate), jax.tree.leaves(previous), strict=True
        )
    ]
    return jax.tree.unflatten(jax.tree.structure(previous), values)


def _states_match_exactly(
    left: DistributedAtomisticState, right: DistributedAtomisticState, /
) -> Array:
    static_match = (
        left.plan_id == right.plan_id
        and left.runtime_id == right.runtime_id
        and left.run_id == right.run_id
        and left.replica_id == right.replica_id
        and left.epoch_id == right.epoch_id
    )
    dynamic_match = jnp.asarray(True)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        dynamic_match = dynamic_match & jnp.array_equal(left_leaf, right_leaf)
    return jnp.asarray(static_match) & dynamic_match


def commit_distributed_migration(
    plan: DistributedAtomisticPlan,
    state: DistributedAtomisticState,
    candidate: DistributedMigrationCandidate,
    /,
) -> DistributedAtomisticState:
    if (
        not isinstance(plan, DistributedAtomisticPlan)
        or not isinstance(state, DistributedAtomisticState)
        or not isinstance(candidate, DistributedMigrationCandidate)
    ):
        raise TypeError("Migration commit requires plan, state, and candidate.")
    if plan.execution_mode == "collective":
        raise ValueError(
            "Collective migration commit is unsupported without explicit "
            "continuation-payload exchange."
        )
    if state.plan_id != plan.plan_id or candidate.plan_id != plan.plan_id:
        raise ValueError("Migration candidate or state belongs to another plan.")
    if candidate.runtime_id != state.runtime_id:
        raise ValueError("Migration candidate belongs to another prepared runtime.")
    source_matches = _states_match_exactly(candidate.source_state, state)
    commit = state.successful & candidate.successful & source_matches
    decomposition = _select_decomposition(
        commit, candidate.decomposition, state.decomposition
    )
    halos = _select_halos(commit, candidate.halos, state.halos)
    positions = jnp.where(commit, candidate.positions, state.positions)
    partition_momentum = _partition_vector_sum(
        state.momenta, decomposition.full_owned_mask, plan.reduction
    )
    successful = state.successful & candidate.successful & source_matches
    status = DistributedExecutionStatus(
        state.status.finite & candidate.finite,
        state.status.ownership_capacity_ok & ~candidate.decomposition.ownership_overflow,
        state.status.halo_capacity_ok & ~candidate.decomposition.halo_overflow,
        state.status.migration_capacity_ok & ~candidate.overflow,
        state.status.reciprocal_converged,
        state.status.polarization_converged,
        state.status.collective_supported,
        successful,
    )
    return DistributedAtomisticState(
        positions,
        state.momenta,
        state.cell,
        decomposition,
        halos,
        partition_momentum,
        state.partition_energy,
        state.thermostat_state,
        state.barostat_state,
        state.polarization_warm_start,
        state.bias_state,
        state.rng_key,
        state.step_index,
        state.decomposition_epoch + commit.astype(jnp.int32),
        status,
        successful,
        state.plan_id,
        state.runtime_id,
        state.run_id,
        state.replica_id,
        state.epoch_id,
    )


def migrate_distributed_atomistic(
    plan: DistributedAtomisticPlan,
    state: DistributedAtomisticState,
    positions: ArrayLike,
    /,
) -> DistributedAtomisticState:
    """Propose and atomically commit a migration, rolling back on failure."""
    return commit_distributed_migration(
        plan, state, propose_distributed_migration(plan, state, positions)
    )


def exchange_distributed_halos(
    runtime: PreparedDistributedAtomisticRuntime,
    state: DistributedAtomisticState,
    values: ArrayLike,
    /,
) -> Array:
    """Exchange a canonical particle payload through fixed padded halo routes."""
    if not isinstance(runtime, PreparedDistributedAtomisticRuntime) or not isinstance(
        state, DistributedAtomisticState
    ):
        raise TypeError("Halo exchange requires prepared runtime and state.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("Distributed state belongs to another prepared runtime.")
    value = jnp.asarray(values)
    if not value.shape or value.shape[0] != runtime.plan.system.capacity:
        raise ValueError("Halo payload must begin with atomistic particle capacity.")
    indices = state.decomposition.halo_send_indices
    safe_indices = jnp.maximum(indices, 0)
    mask = state.decomposition.halo_send_mask
    expanded_mask = mask.reshape(mask.shape + (1,) * (value.ndim - 1))
    send = jnp.where(expanded_mask, value[safe_indices], 0)
    if runtime.plan.execution_mode == "local-reference":
        return jnp.swapaxes(send, 0, 1)
    if runtime.collectives is None:
        raise ValueError("Collective halo exchange has no communication operations.")
    rank = runtime.collectives.partition_index
    rank_mask = jnp.arange(runtime.plan.decomposition.partitions)[:, None, None] == rank
    collective_mask = mask & rank_mask
    collective_send = jnp.where(
        collective_mask.reshape(collective_mask.shape + (1,) * (value.ndim - 1)),
        send,
        0,
    )
    received = jnp.asarray(runtime.collectives.exchange(collective_send, collective_mask))
    expected = (
        runtime.plan.decomposition.partitions,
        runtime.plan.decomposition.partitions,
        runtime.plan.halo_capacity,
        *value.shape[1:],
    )
    if received.shape != expected:
        raise ValueError("Collective halo exchange returned the wrong padded shape.")
    return received


def _accumulate_route_forces(
    indices: Array,
    mask: Array,
    forces: Array,
    capacity: int,
    policy: DistributedReductionPolicy,
    /,
) -> Array:
    flattened_indices = indices.reshape((-1,))
    flattened_mask = mask.reshape((-1,))
    flattened_forces = forces.reshape((-1, 3))

    def particle_force(particle_index):
        selected = flattened_mask & (flattened_indices == particle_index)
        contributions = jnp.where(selected[:, None], flattened_forces, 0)
        return _ordered_sum(contributions, policy)

    return jax.vmap(particle_force)(jnp.arange(capacity, dtype=flattened_indices.dtype))


def reverse_halo_force_return(
    decomposition: DistributedSpatialDecomposition,
    received_forces: ArrayLike,
    /,
    *,
    policy: DistributedReductionPolicy | None = None,
) -> Array:
    """Return local-reference halo forces in a declared accumulation order."""
    if not isinstance(decomposition, DistributedSpatialDecomposition):
        raise TypeError("decomposition must be DistributedSpatialDecomposition.")
    if decomposition.execution_mode != "local-reference":
        raise ValueError(
            "Collective force return requires reverse_distributed_halo_force_return."
        )
    force = jnp.asarray(received_forces)
    if force.shape != decomposition.halo_receive_indices.shape + (3,):
        raise ValueError("Received halo forces must match padded receive routes.")
    policy_ = DistributedReductionPolicy() if policy is None else policy
    if not isinstance(policy_, DistributedReductionPolicy):
        raise TypeError("policy must be DistributedReductionPolicy or None.")
    return _accumulate_route_forces(
        decomposition.halo_receive_indices,
        decomposition.halo_receive_mask,
        force,
        decomposition.owner.shape[0],
        policy_,
    )


def reverse_distributed_halo_force_return(
    runtime: PreparedDistributedAtomisticRuntime,
    state: DistributedAtomisticState,
    received_forces: ArrayLike,
    /,
) -> Array:
    """Communicate halo forces back to owner ranks and accumulate deterministically."""
    if not isinstance(runtime, PreparedDistributedAtomisticRuntime) or not isinstance(
        state, DistributedAtomisticState
    ):
        raise TypeError("Force return requires prepared runtime and state.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("Distributed state belongs to another prepared runtime.")
    force = jnp.asarray(received_forces)
    expected = state.decomposition.halo_receive_indices.shape + (3,)
    if force.shape != expected:
        raise ValueError("Received halo forces must match padded receive routes.")
    if runtime.plan.execution_mode == "local-reference":
        return reverse_halo_force_return(
            state.decomposition, force, policy=runtime.plan.reduction
        )
    if runtime.collectives is None:
        raise ValueError("Collective force return has no communication operations.")
    rank = runtime.collectives.partition_index
    destination_mask = (
        jnp.arange(runtime.plan.decomposition.partitions)[:, None, None] == rank
    )
    receive_mask = state.decomposition.halo_receive_mask & destination_mask
    local_force = jnp.where(receive_mask[..., None], force, 0)
    returned = jnp.asarray(
        runtime.collectives.reverse_exchange(local_force, receive_mask)
    )
    if returned.shape != expected:
        raise ValueError("Reverse collective exchange returned the wrong shape.")
    return _accumulate_route_forces(
        state.decomposition.halo_send_indices,
        state.decomposition.halo_send_mask,
        returned,
        runtime.plan.system.capacity,
        runtime.plan.reduction,
    )


def distributed_domain_evidence(
    plan: DistributedAtomisticPlan,
    state: DistributedAtomisticState,
    /,
    *,
    pair_work: ArrayLike | None = None,
    iterative_work: ArrayLike | None = None,
) -> DistributedDomainEvidence:
    if state.plan_id != plan.plan_id:
        raise ValueError("Distributed state belongs to another plan.")
    partitions = plan.decomposition.partitions
    pair = (
        jnp.zeros((partitions,), state.positions.dtype)
        if pair_work is None
        else jnp.asarray(pair_work)
    )
    iterative = (
        jnp.zeros((partitions,), state.positions.dtype)
        if iterative_work is None
        else jnp.asarray(iterative_work)
    )
    if pair.shape != (partitions,) or iterative.shape != (partitions,):
        raise ValueError("Distributed work evidence must have one value per partition.")
    owned = state.decomposition.owned_counts
    halo = state.decomposition.halo_counts
    work = owned + halo + pair + iterative
    imbalance = jnp.max(work) / jnp.maximum(jnp.mean(work), 1.0)
    finite = jnp.all(jnp.isfinite(work)) & ~state.decomposition.nonfinite
    inside = ~state.decomposition.outside_domain
    successful = finite & inside & state.status.successful
    return DistributedDomainEvidence(
        owned,
        halo,
        pair,
        iterative,
        work,
        imbalance,
        finite,
        inside,
        ~state.decomposition.ownership_overflow,
        ~state.decomposition.halo_overflow,
        state.status.migration_capacity_ok,
        successful,
        canonical_fingerprint(
            {"kind": "distributed-domain-evidence", "plan": plan.plan_id}
        ),
    )


def _partition_evaluation(
    runtime: PreparedDistributedAtomisticRuntime,
    state: DistributedAtomisticState,
    evaluation: AtomisticPotentialEvaluation,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    if not isinstance(evaluation, AtomisticPotentialEvaluation):
        raise TypeError("Distributed phases require AtomisticPotentialEvaluation values.")
    capacity = runtime.plan.system.capacity
    if evaluation.forces.shape != (capacity, 3) or evaluation.atom_energy.shape != (
        capacity,
    ):
        raise ValueError("Atomistic evaluation changed distributed particle capacity.")
    if evaluation.virial.shape != (3, 3):
        raise ValueError("Atomistic evaluation virial must have shape (3, 3).")
    owned = state.decomposition.full_owned_mask
    local_atom = jnp.where(owned, evaluation.atom_energy[None, :], 0)
    local_energy = jax.vmap(lambda value: _ordered_sum(value, runtime.plan.reduction))(
        local_atom
    )
    residual = evaluation.energy - _ordered_sum(local_energy, runtime.plan.reduction)
    local_energy = local_energy.at[0].add(residual)
    local_force = jnp.where(owned[:, :, None], evaluation.forces[None, :, :], 0)
    local_virial = (
        jnp.zeros((runtime.plan.decomposition.partitions, 3, 3), evaluation.virial.dtype)
        .at[0]
        .set(evaluation.virial)
    )
    finite = (
        jnp.isfinite(evaluation.energy)
        & jnp.all(jnp.isfinite(evaluation.forces))
        & jnp.all(jnp.isfinite(evaluation.virial))
        & jnp.all(jnp.isfinite(evaluation.atom_energy))
    )
    return (
        local_energy,
        local_force,
        local_virial,
        local_atom,
        finite & evaluation.successful,
    )


def evaluate_distributed_atomistic(
    runtime: PreparedDistributedAtomisticRuntime,
    state: DistributedAtomisticState,
    direct: AtomisticPotentialEvaluation,
    /,
    *,
    sparse_correction: AtomisticPotentialEvaluation | None = None,
    reciprocal: DistributedReciprocalEvidence | None = None,
    polarization: DistributedPolarizationEvidence | None = None,
) -> DistributedAtomisticEvaluation:
    """Execute and reduce direct, sparse, and reciprocal reference phases."""
    if not isinstance(runtime, PreparedDistributedAtomisticRuntime) or not isinstance(
        state, DistributedAtomisticState
    ):
        raise TypeError("Distributed evaluation requires prepared runtime and state.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("Distributed state belongs to another prepared runtime.")

    reciprocal_evaluation = None
    reciprocal_binding = jnp.asarray(runtime.plan.pme is None)
    if reciprocal is not None:
        if not isinstance(reciprocal, DistributedReciprocalEvidence):
            raise TypeError("reciprocal must be DistributedReciprocalEvidence or None.")
        if runtime.plan.pme is None:
            raise ValueError("A reciprocal phase requires a distributed PME plan.")
        expected_pme = runtime.pme_runtime().prepared_id
        if reciprocal.prepared_id != expected_pme:
            raise ValueError("Reciprocal evidence belongs to another PME runtime.")
        reciprocal_binding = reciprocal.successful & _source_evidence_matches(
            state,
            reciprocal.source_positions,
            reciprocal.source_cell,
            reciprocal.source_step_index,
            reciprocal.source_decomposition_epoch,
            reciprocal.source_run_id,
            reciprocal.source_replica_id,
            reciprocal.source_epoch_id,
        )
        reciprocal_evaluation = reciprocal.evaluation

    polarization_binding = jnp.asarray(runtime.plan.polarization is None)
    if polarization is not None:
        if not isinstance(polarization, DistributedPolarizationEvidence):
            raise TypeError(
                "polarization must be DistributedPolarizationEvidence or None."
            )
        if runtime.plan.polarization is None:
            raise ValueError("Polarization evidence requires a polarization plan.")
        expected_polarization = runtime.polarization_runtime().prepared_id
        if polarization.prepared_id != expected_polarization:
            raise ValueError("Polarization evidence belongs to another runtime.")
        polarization_binding = polarization.successful & _source_evidence_matches(
            state,
            polarization.source_positions,
            polarization.source_cell,
            polarization.source_step_index,
            polarization.source_decomposition_epoch,
            polarization.source_run_id,
            polarization.source_replica_id,
            polarization.source_epoch_id,
        )

    dtype = jnp.asarray(direct.energy).dtype
    capacity = runtime.plan.system.capacity
    partitions = runtime.plan.decomposition.partitions

    def empty_phase():
        return (
            jnp.zeros((partitions,), dtype),
            jnp.zeros((partitions, capacity, 3), dtype),
            jnp.zeros((partitions, 3, 3), dtype),
            jnp.zeros((partitions, capacity), dtype),
            jnp.asarray(True),
        )

    direct_phase = _partition_evaluation(runtime, state, direct)
    sparse_phase = (
        empty_phase()
        if sparse_correction is None
        else _partition_evaluation(runtime, state, sparse_correction)
    )
    reciprocal_phase = (
        empty_phase()
        if reciprocal_evaluation is None
        else _partition_evaluation(runtime, state, reciprocal_evaluation)
    )
    phases = (direct_phase, sparse_phase, reciprocal_phase)
    component_phase_successful = (
        jnp.stack(tuple(jnp.asarray(phase[4]) for phase in phases))
        .at[2]
        .set(reciprocal_phase[4] & reciprocal_binding)
    )

    if runtime.plan.execution_mode == "collective":
        if runtime.collectives is None:
            raise ValueError("Collective reduction has no communication operations.")
        rank = runtime.collectives.partition_index

        def collective_sum(value: Array) -> Array:
            result = jnp.asarray(runtime.collectives.reduce_sum(value))
            if result.shape != value.shape:
                raise ValueError("Collective reduction changed the contribution shape.")
            return result

        def collective_all(value: Array) -> Array:
            failures = collective_sum((~jnp.asarray(value, bool)).astype(jnp.int32))
            return failures == 0

        local_component_energy = jnp.stack(tuple(phase[0][rank] for phase in phases))
        local_energy = _ordered_sum(local_component_energy, runtime.plan.reduction)
        local_force = sum(
            (phase[1][rank] for phase in phases),
            jnp.zeros_like(direct_phase[1][rank]),
        )
        local_virial = sum(
            (phase[2][rank] for phase in phases),
            jnp.zeros_like(direct_phase[2][rank]),
        )
        local_atom = sum(
            (phase[3][rank] for phase in phases),
            jnp.zeros_like(direct_phase[3][rank]),
        )
        local_partition_energy = (
            jnp.zeros((partitions,), dtype).at[rank].set(local_energy)
        )
        energy = collective_sum(local_energy)
        forces = collective_sum(local_force)
        virial = collective_sum(local_virial)
        atom_energy = collective_sum(local_atom)
        partition_energy = collective_sum(local_partition_energy)
        component_phase_energy = collective_sum(local_component_energy)
        component_phase_successful = collective_all(component_phase_successful)
        reciprocal_binding = collective_all(reciprocal_binding)
        polarization_binding = collective_all(polarization_binding)
        state_successful = collective_all(state.successful)
        status_finite = collective_all(state.status.finite)
        ownership_capacity_ok = collective_all(state.status.ownership_capacity_ok)
        halo_capacity_ok = collective_all(state.status.halo_capacity_ok)
        migration_capacity_ok = collective_all(state.status.migration_capacity_ok)
        collective_supported = collective_all(state.status.collective_supported)
    else:
        partition_energy = sum(
            (phase[0] for phase in phases), jnp.zeros_like(direct_phase[0])
        )
        local_force = sum((phase[1] for phase in phases), jnp.zeros_like(direct_phase[1]))
        local_virial = sum(
            (phase[2] for phase in phases), jnp.zeros_like(direct_phase[2])
        )
        local_atom = sum((phase[3] for phase in phases), jnp.zeros_like(direct_phase[3]))
        energy = _ordered_sum(partition_energy, runtime.plan.reduction)
        forces = _ordered_sum(local_force, runtime.plan.reduction)
        virial = _ordered_sum(local_virial, runtime.plan.reduction)
        atom_energy = _ordered_sum(local_atom, runtime.plan.reduction)
        component_phase_energy = jnp.stack(
            tuple(_ordered_sum(phase[0], runtime.plan.reduction) for phase in phases)
        )
        state_successful = state.successful
        status_finite = state.status.finite
        ownership_capacity_ok = state.status.ownership_capacity_ok
        halo_capacity_ok = state.status.halo_capacity_ok
        migration_capacity_ok = state.status.migration_capacity_ok
        collective_supported = state.status.collective_supported

    reciprocal_converged = reciprocal_binding & component_phase_successful[2]
    polarization_converged = polarization_binding
    finite = (
        jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(forces))
        & jnp.all(jnp.isfinite(virial))
        & jnp.all(jnp.isfinite(atom_energy))
    )
    reduction_successful = finite
    if runtime.plan.execution_mode == "collective":
        reduction_successful = collective_all(reduction_successful)
    phase_energy = jnp.concatenate((component_phase_energy, energy[None]))
    phase_successful = jnp.concatenate(
        (component_phase_successful, reduction_successful[None])
    )
    phase_success = jnp.all(phase_successful) & reciprocal_converged
    successful = (
        state_successful & phase_success & polarization_converged & reduction_successful
    )
    phase_evidence = DistributedPhaseEvidence(
        phase_energy,
        phase_successful,
        finite,
        reduction_successful,
        successful,
        canonical_fingerprint(
            {"kind": "distributed-phase-evidence", "runtime": runtime.runtime_id}
        ),
    )
    status = DistributedExecutionStatus(
        status_finite & finite,
        ownership_capacity_ok,
        halo_capacity_ok,
        migration_capacity_ok,
        reciprocal_converged,
        polarization_converged,
        collective_supported,
        successful,
    )
    mask = runtime.plan.output_mask
    output_energy = energy if mask.energy else jnp.zeros_like(energy)
    output_forces = forces if mask.forces else jnp.zeros_like(forces)
    output_virial = virial if mask.virial else jnp.zeros_like(virial)
    output_atom = atom_energy if mask.atom_energy else jnp.zeros_like(atom_energy)
    output_partition = (
        partition_energy if mask.partition_energy else jnp.zeros_like(partition_energy)
    )
    available = jnp.asarray(
        (
            mask.energy,
            mask.forces,
            mask.virial,
            mask.atom_energy,
            mask.partition_energy,
        ),
        dtype=bool,
    )
    return DistributedAtomisticEvaluation(
        output_energy,
        output_forces,
        output_virial,
        output_atom,
        output_partition,
        available,
        phase_evidence,
        status,
        successful,
        mask.mask_id,
        runtime.runtime_id,
    )


def halo_short_range_evaluate(
    plan: DistributedAtomisticPlan,
    state: DistributedAtomisticState,
    potential: PreparedAtomisticPotentialProgram,
    neighborhood,
    /,
):
    """Evaluate the established short-range API through canonical ownership."""
    if (
        state.plan_id != plan.plan_id
        or potential.system.prepared_id != plan.system.prepared_id
    ):
        raise ValueError("Distributed state or potential belongs to another plan.")
    if plan.execution_mode == "collective":
        raise ValueError(
            "halo_short_range_evaluate is local-reference only; collective "
            "execution requires evaluate_distributed_atomistic."
        )
    cutoff = potential.plan.requirements.cutoff
    if cutoff is not None and cutoff > plan.decomposition.halo_radius:
        raise ValueError("Distributed halo radius is smaller than potential cutoff.")
    if potential.plan.requirements.reciprocal_grid:
        raise ValueError(
            "Reciprocal terms require distributed_particle_mesh_electrostatics."
        )
    evaluation = potential.evaluate(state.positions, neighborhood)
    owned = state.decomposition.full_owned_mask
    local_force = jnp.where(owned[:, :, None], evaluation.forces[None, :, :], 0)
    reverse_force = _ordered_sum(local_force, plan.reduction)
    local_atom = jnp.where(owned, evaluation.atom_energy[None, :], 0)
    partition_energy = jax.vmap(lambda value: _ordered_sum(value, plan.reduction))(
        local_atom
    )
    partition_energy = partition_energy.at[0].add(
        evaluation.energy - _ordered_sum(partition_energy, plan.reduction)
    )
    distributed = AtomisticPotentialEvaluation(
        evaluation.energy,
        evaluation.term_energies,
        evaluation.atom_energy,
        reverse_force,
        evaluation.virial,
        evaluation.successful & state.successful,
        evaluation.neighborhood_successful,
        evaluation.graph_overflow,
        evaluation.program_id,
    )
    return distributed, partition_energy


def distributed_constraint_projection(
    constraints: PreparedDistanceConstraints,
    previous_positions: ArrayLike,
    proposed_positions: ArrayLike,
    momenta: ArrayLike,
    /,
):
    return constraints.project_positions(previous_positions, proposed_positions, momenta)


def distributed_thermodynamic_reduction(
    local_energy: ArrayLike,
    local_momentum: ArrayLike,
    /,
    *,
    policy: DistributedReductionPolicy | None = None,
    collectives: DistributedCollectiveOperations | None = None,
):
    """Reduce thermodynamic values in a declared deterministic order."""
    energy = jnp.asarray(local_energy)
    momentum = jnp.asarray(local_momentum)
    if energy.ndim != 1 or momentum.shape != (energy.shape[0], 3):
        raise ValueError(
            "Thermodynamic inputs must have shapes (partition,) and (partition, 3)."
        )
    policy_ = DistributedReductionPolicy() if policy is None else policy
    if not isinstance(policy_, DistributedReductionPolicy):
        raise TypeError("policy must be DistributedReductionPolicy or None.")
    if collectives is None:
        reduced_energy = _ordered_sum(energy, policy_)
        reduced_momentum = _ordered_sum(momentum, policy_)
    else:
        if not isinstance(collectives, DistributedCollectiveOperations):
            raise TypeError(
                "collectives must be DistributedCollectiveOperations or None."
            )
        if collectives.partition_index >= energy.shape[0]:
            raise ValueError("Collective partition_index exceeds local inputs.")
        reduced_energy = jnp.asarray(
            collectives.reduce_sum(energy[collectives.partition_index])
        )
        reduced_momentum = jnp.asarray(
            collectives.reduce_sum(momentum[collectives.partition_index])
        )
    return reduced_energy, reduced_momentum


def distributed_particle_mesh_electrostatics(
    runtime: PreparedDistributedAtomisticRuntime,
    state: DistributedAtomisticState,
    reciprocal: DistributedReciprocalEvidence,
    /,
):
    """Reduce state-bound reciprocal work through the prepared runtime."""
    if not isinstance(runtime, PreparedDistributedAtomisticRuntime) or not isinstance(
        state, DistributedAtomisticState
    ):
        raise TypeError("Distributed PME requires prepared runtime and state.")
    if not isinstance(reciprocal, DistributedReciprocalEvidence):
        raise TypeError("reciprocal must be DistributedReciprocalEvidence.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("Distributed state belongs to another prepared runtime.")
    prepared = runtime.pme_runtime()
    if reciprocal.prepared_id != prepared.prepared_id:
        raise ValueError("Reciprocal evidence belongs to another PME runtime.")
    binding = reciprocal.successful & _source_evidence_matches(
        state,
        reciprocal.source_positions,
        reciprocal.source_cell,
        reciprocal.source_step_index,
        reciprocal.source_decomposition_epoch,
        reciprocal.source_run_id,
        reciprocal.source_replica_id,
        reciprocal.source_epoch_id,
    )
    phase = _partition_evaluation(runtime, state, reciprocal.evaluation)
    if runtime.plan.execution_mode == "collective":
        if runtime.collectives is None:
            raise ValueError("Collective PME has no communication operations.")
        rank = runtime.collectives.partition_index
        force = jnp.asarray(runtime.collectives.reduce_sum(phase[1][rank]))
        energy = jnp.asarray(runtime.collectives.reduce_sum(phase[0][rank]))
        failures = jnp.asarray(
            runtime.collectives.reduce_sum((~binding).astype(jnp.int32))
        )
        binding = failures == 0
    else:
        force = _ordered_sum(phase[1], runtime.plan.reduction)
        energy = _ordered_sum(phase[0], runtime.plan.reduction)
    return (
        jnp.where(binding, force, jnp.full_like(force, jnp.nan)),
        jnp.where(binding, energy, jnp.asarray(jnp.nan, energy.dtype)),
    )


def checkpoint_distributed_atomistic(
    state: DistributedAtomisticState, /
) -> DistributedAtomisticCheckpoint:
    if not isinstance(state, DistributedAtomisticState):
        raise TypeError("state must be DistributedAtomisticState.")
    identity = DistributedAtomisticCheckpointIdentity(state)
    return DistributedAtomisticCheckpoint(state, identity)


def restore_distributed_atomistic_checkpoint(
    runtime: PreparedDistributedAtomisticRuntime,
    checkpoint: DistributedAtomisticCheckpoint,
    /,
) -> DistributedAtomisticState:
    if not isinstance(runtime, PreparedDistributedAtomisticRuntime) or not isinstance(
        checkpoint, DistributedAtomisticCheckpoint
    ):
        raise TypeError("Checkpoint restore requires prepared runtime and checkpoint.")
    if checkpoint.state.runtime_id != runtime.runtime_id:
        raise ValueError("Distributed checkpoint belongs to another prepared runtime.")
    observed = DistributedAtomisticCheckpointIdentity(checkpoint.state)
    if observed.checkpoint_id != checkpoint.identity.checkpoint_id:
        raise ValueError("Distributed checkpoint content identity is corrupt.")
    return checkpoint.state


__all__ = [
    "DistributedAtomisticCheckpoint",
    "DistributedAtomisticCheckpointIdentity",
    "DistributedAtomisticEvaluation",
    "DistributedAtomisticPlan",
    "DistributedAtomisticState",
    "DistributedCollectiveOperations",
    "DistributedDomainEvidence",
    "DistributedExecutionMode",
    "DistributedExecutionStatus",
    "DistributedMigrationCandidate",
    "DistributedOutputMask",
    "DistributedPMEPlan",
    "DistributedPhase",
    "DistributedPhaseEvidence",
    "DistributedPolarizationEvidence",
    "DistributedPolarizationPlan",
    "DistributedReciprocalEvidence",
    "DistributedReductionMode",
    "DistributedReductionPolicy",
    "DistributedSpatialDecomposition",
    "PreparedDistributedAtomisticRuntime",
    "PreparedDistributedPME",
    "PreparedDistributedPolarization",
    "certify_distributed_polarization",
    "certify_distributed_reciprocal",
    "checkpoint_distributed_atomistic",
    "commit_distributed_migration",
    "distributed_constraint_projection",
    "distributed_domain_evidence",
    "distributed_particle_mesh_electrostatics",
    "distributed_thermodynamic_reduction",
    "evaluate_distributed_atomistic",
    "exchange_distributed_halos",
    "halo_short_range_evaluate",
    "migrate_distributed_atomistic",
    "propose_distributed_migration",
    "restore_distributed_atomistic_checkpoint",
    "reverse_distributed_halo_force_return",
    "reverse_halo_force_return",
]
