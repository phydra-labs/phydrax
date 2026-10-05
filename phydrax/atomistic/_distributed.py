#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity spatial decomposition and execution for atomistic programs.

Two execution families share this module.

The slab family (`DistributedAtomisticPlan`) represents every logical shard
with fixed-shape JAX arrays over the global particle capacity. Its
`halo_short_range_evaluate` evaluates the global prepared program and then
masks owner contributions: it is a global-evaluate-then-mask reference for
classical programs, not owner-local execution. Collective runtimes have no
implicit communication fallback: callers must prepare them with explicit JAX
exchange and reduction callables.

The owner-local family (`OwnerLocalAtomisticPlan`) executes layered learned
potentials partition-locally. Atoms are owned through a fractional owner grid
of the periodic cell and the canonical spatial point layout; every owner
builds its own image-aware receiver graph from exchanged periodic image
aliases, evaluates only its receivers with source features gathered through
the canonical halo plan at every interaction, and returns reverse cotangents
exactly once through the halo transpose. Shared cell (strain) cotangents and
parameter cotangents are per-owner partials reduced in declared owner order.
Lane-reference ownership runs the identical program as named vmap lanes on one
device; it is the numerical oracle, not distributed hardware evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._execution_runtime import ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import combine_parameters, NonTrainableState, partition_parameters
from ..discretization import (
    ParticleDomainDecompositionPlan,
    ParticleHaloState,
    ParticleNeighborhoodState,
)
from ..discretization._periodic_cell import (
    _complete_image_shifts,
    PeriodicCell,
    PeriodicImageStencil,
)
from ..discretization.particle._distributed import FractionalOwnerPartition
from ..discretization.spatial._distributed_relations import (
    DistributedHaloPlan,
    DistributedMigrationEvidence,
    DistributedOwnershipPlan,
    DistributedPointLayout,
)
from ..ein import contract
from ..sparse import EdgeRelation
from ..sparse._streamed import PreparedStreamedRelation, StreamedRelationPlan
from ..typing import Bool, Dim, Float, Int32, parse, PRNGKey, Scalar
from ._constraints import ConstraintProjection, PreparedDistanceConstraints
from ._feature_execution import (
    AtomisticLayeredModel,
    owner_atom_energies,
    owner_layer_gradients,
    OwnerLayerTopology,
)
from ._potential import AbstractAtomisticPotential, atomistic_potential_revision
from ._potential_program import (
    AtomisticPotentialEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._system import PreparedAtomisticSystem
from ._units import AtomisticUnitSystem


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
    ) -> None:
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

    def __init__(self, mode: DistributedReductionMode = "deterministic", /) -> None:
        mode = parse(mode, DistributedReductionMode, "mode")
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
    ) -> None:
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

    @classmethod
    def from_execution_group(
        cls,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str | None = None,
    ) -> DistributedCollectiveOperations:
        if execution_group.spec.device_count != len(execution_group.spec.process_indices):
            raise ValueError(
                "Atomistic rank-local collectives require one assigned device per JAX process."
            )
        axis = execution_group.mesh.axis_names[0] if axis_name is None else str(axis_name)
        if axis not in execution_group.mesh.axis_names:
            raise ValueError("axis_name is outside the execution-group mesh.")

        def exchange(values: Array, mask: Array) -> Array:
            del mask
            return jax.lax.psum(jnp.swapaxes(values, 0, 1), axis)

        def reverse_exchange(values: Array, mask: Array) -> Array:
            del mask
            return jax.lax.psum(jnp.swapaxes(values, 0, 1), axis)

        def reduce_sum(value: Array) -> Array:
            return jax.lax.psum(value, axis)

        return cls(
            exchange,
            reverse_exchange,
            reduce_sum,
            partition_index=jax.process_index(),
            collective_id=canonical_fingerprint(
                {
                    "kind": "atomistic-jax-collectives",
                    "execution_group_id": execution_group.spec.group_id,
                    "axis_name": axis,
                }
            ),
        )


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
    ) -> None:
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
        self.grid_shape = tuple(shape)
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
    ) -> None:
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
    ) -> None:
        if not isinstance(system, PreparedAtomisticSystem) or not isinstance(
            decomposition, ParticleDomainDecompositionPlan
        ):
            raise TypeError(
                "Distributed atomistics requires prepared system and decomposition."
            )
        if decomposition.box.ambient_dimension != 3:
            raise ValueError("Distributed atomistic decomposition must be 3D.")
        execution_mode = parse(execution_mode, DistributedExecutionMode, "execution_mode")
        if execution_mode == "collective" and decomposition.partitions < 2:
            raise ValueError("Collective execution requires at least two partitions.")

        def capacity(
            name: str, value: int | None, default: int, *, positive: bool
        ) -> int:
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
    unit_system_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    replica_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    owner_digest: str = eqx.field(static=True)
    payload_digest: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self, state: DistributedAtomisticState, units: AtomisticUnitSystem, /
    ) -> None:
        if not isinstance(state, DistributedAtomisticState) or not isinstance(
            units, AtomisticUnitSystem
        ):
            raise TypeError(
                "Distributed checkpoint identity requires state and complete units."
            )
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
        self.unit_system_id = units.unit_system_id
        self.run_id = state.run_id
        self.replica_id = state.replica_id
        self.epoch_id = state.epoch_id
        self.owner_digest = owner_digest
        self.payload_digest = payload_digest
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "distributed-atomistic-checkpoint",
                "plan": state.plan_id,
                "unit_system": units.unit_system_id,
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
    units: AtomisticUnitSystem
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
    off_diagonal = ~jnp.eye(partitions, dtype=jnp.bool_)
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

    def compensated_step(
        index: int | Array, carry: tuple[Array, Array]
    ) -> tuple[Array, Array]:
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
            "Collective migration requires continuation-payload communication and is not supported by this runtime."
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
            "Collective migration commit is unsupported without explicit continuation-payload exchange."
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

    def particle_force(particle_index: Array) -> Array:
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

    def empty_phase() -> tuple[Array, Array, Array, Array, Array]:
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
        collectives = runtime.collectives
        if collectives is None:
            raise ValueError("Collective reduction has no communication operations.")
        rank = collectives.partition_index

        def collective_sum(value: Array) -> Array:
            result = jnp.asarray(collectives.reduce_sum(value))
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
        dtype=jnp.bool_,
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
    neighborhood: ParticleNeighborhoodState,
    /,
) -> tuple[AtomisticPotentialEvaluation, Array]:
    """Global-evaluate-then-mask reference of a short-range prepared program.

    The complete program is evaluated on the global state and owner masks
    attribute its outputs to slab partitions. This is the reference route for
    classical programs, not owner-local execution; layered learned potentials
    execute partition-locally through `evaluate_owner_local_atomistic`.
    """
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
        energy=evaluation.energy,
        term_energies=evaluation.term_energies,
        atom_energy=evaluation.atom_energy,
        forces=reverse_force,
        virial=evaluation.virial,
        successful=evaluation.successful & state.successful,
        neighborhood_successful=evaluation.neighborhood_successful,
        graph_overflow=evaluation.graph_overflow,
        strain_derivative=evaluation.strain_derivative,
        stress=evaluation.stress,
        program_id=evaluation.program_id,
        stress_convention=evaluation.stress_convention,
    )
    return distributed, partition_energy


def distributed_constraint_projection(
    constraints: PreparedDistanceConstraints,
    previous_positions: ArrayLike,
    proposed_positions: ArrayLike,
    momenta: ArrayLike,
    /,
) -> ConstraintProjection:
    return constraints.project_positions(previous_positions, proposed_positions, momenta)


def distributed_thermodynamic_reduction(
    local_energy: ArrayLike,
    local_momentum: ArrayLike,
    /,
    *,
    policy: DistributedReductionPolicy | None = None,
    collectives: DistributedCollectiveOperations | None = None,
) -> tuple[Array, Array]:
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
) -> tuple[Array, Array]:
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
    runtime: PreparedDistributedAtomisticRuntime,
    state: DistributedAtomisticState,
    /,
) -> DistributedAtomisticCheckpoint:
    if not isinstance(runtime, PreparedDistributedAtomisticRuntime) or not isinstance(
        state, DistributedAtomisticState
    ):
        raise TypeError("Checkpointing requires prepared runtime and state.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("Distributed checkpoint state belongs to another runtime.")
    units = runtime.plan.system.plan.units
    identity = DistributedAtomisticCheckpointIdentity(state, units)
    return DistributedAtomisticCheckpoint(state, units, identity)


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
    if checkpoint.units.unit_system_id != runtime.plan.system.plan.units.unit_system_id:
        raise ValueError(
            "Distributed checkpoint complete unit descriptor is incompatible."
        )
    observed = DistributedAtomisticCheckpointIdentity(checkpoint.state, checkpoint.units)
    if observed.checkpoint_id != checkpoint.identity.checkpoint_id:
        raise ValueError("Distributed checkpoint content identity is corrupt.")
    return checkpoint.state


class _OwnerRowDim(Dim, minimum=1):
    """Owner-blocked atom rows: ``owner_count * local_capacity``."""


class _OwnerEdgeDim(Dim, minimum=1):
    """Owner-blocked edge routes: ``owner_count * edge_capacity``."""


class _OwnerCountDim(Dim, minimum=1):
    """Owners of one ownership plan."""


class _LatticeRankDim(Dim, minimum=1):
    """Lattice rank of the periodic cell."""


_AMBIENT_DIMENSION = 3


def _owner_stack(tree: PyTree[Array], /) -> PyTree[Array]:
    """Give every owner-region output a leading owner axis of one."""
    return jax.tree.map(lambda leaf: leaf[None], tree)


def _owner_unstack(tree: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda leaf: leaf[0], tree)


def _positive_capacity(name: str, value: int, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _finite_length(name: str, value: float, /, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be real.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


def _require_complete_stencil(
    cell: PeriodicCell, stencil: PeriodicImageStencil, radius: float, /
) -> None:
    """Refuse an image stencil that is not the complete translation box of ``cell``.

    Its shifts must be the canonical lexicographic box of its extents, its
    nonperiodic extents zero, and every periodic extent must reach the bound
    ``floor(1 + radius * ||H^+[:, i]||)`` of wrapped endpoints in this cell.
    """
    extents = stencil.extents
    if len(extents) != cell.rank or not np.array_equal(
        np.asarray(stencil.shifts), _complete_image_shifts(extents)
    ):
        raise ValueError(
            "The image stencil shifts are not the complete box of its extents."
        )
    if any(
        extent != 0
        for extent, periodic in zip(extents, cell.periodic_axes, strict=True)
        if not periodic
    ):
        raise ValueError("The image stencil translates along a nonperiodic axis.")
    try:
        required = cell.image_stencil(
            radius, maximum_image_count=stencil.image_count
        ).extents
    except ValueError as error:
        raise ValueError(
            "The image stencil does not cover cutoff + skin in its cell."
        ) from error
    if any(have < need for have, need in zip(extents, required, strict=True)):
        raise ValueError("The image stencil does not cover cutoff + skin in its cell.")


@final
class OwnerLocalAtomisticPlan(StrictModule, NonTrainableState):
    """Owner-local layered learned execution over a fractional owner grid.

    Static contract of one fixed-cell owner-local execution: the canonical
    point ownership (devices or reference lanes), the fractional owner
    partition of the periodic cell, the complete integer image stencil for
    ``cutoff + skin``, and every packet/message capacity. The ownership address
    is the unit fractional box of the cell, so layout points are wrapped
    fractional coordinates.

    Capacities: ``alias_capacity`` image aliases per (sender, destination)
    owner pair, ``edge_capacity`` receiver edges per owner,
    ``halo_capacity`` deduplicated source columns per (requester, source)
    owner pair, ``migration_capacity`` migrating atoms per owner pair, and
    ``message_capacity_bytes`` bytes of one interaction's halo message per
    owner. Exceeding any capacity is a refused topology or migration, never a
    truncated relation. ``streaming`` tiles every owner's receiver relation; it
    is prepared once per owner and topology epoch and reused by every layer.
    """

    ownership: DistributedOwnershipPlan
    partition: FractionalOwnerPartition
    stencil: PeriodicImageStencil
    streaming: StreamedRelationPlan
    reduction: DistributedReductionPolicy
    cutoff: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    alias_capacity: int = eqx.field(static=True)
    edge_capacity: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    migration_capacity: int = eqx.field(static=True)
    message_capacity_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ownership: DistributedOwnershipPlan,
        partition: FractionalOwnerPartition,
        stencil: PeriodicImageStencil,
        /,
        *,
        streaming: StreamedRelationPlan,
        cutoff: float,
        skin: float,
        alias_capacity: int,
        edge_capacity: int,
        halo_capacity: int,
        migration_capacity: int,
        message_capacity_bytes: int,
        reduction: DistributedReductionPolicy | None = None,
    ) -> None:
        if not isinstance(ownership, DistributedOwnershipPlan):
            raise TypeError("ownership must be a DistributedOwnershipPlan.")
        if not isinstance(partition, FractionalOwnerPartition):
            raise TypeError("partition must be a FractionalOwnerPartition.")
        if not isinstance(stencil, PeriodicImageStencil):
            raise TypeError("stencil must be a PeriodicImageStencil.")
        if not isinstance(streaming, StreamedRelationPlan):
            raise TypeError("streaming must be a StreamedRelationPlan.")
        reduction_ = DistributedReductionPolicy() if reduction is None else reduction
        if not isinstance(reduction_, DistributedReductionPolicy):
            raise TypeError("reduction must be DistributedReductionPolicy or None.")
        cell = partition.cell
        if cell.ambient_dimension != _AMBIENT_DIMENSION:
            raise ValueError("Owner-local atomistics requires a three-dimensional cell.")
        if partition.owner_count != ownership.owner_count:
            raise ValueError(
                "The fractional partition must declare one region per owner."
            )
        address = ownership.address_plan
        if (
            address.dimension != cell.rank
            or any(value != 0.0 for value in address.lower)
            or any(value != 1.0 for value in address.upper)
            or address.periodic_axes != cell.periodic_axes
        ):
            raise ValueError(
                "Owner-local ownership must address the unit fractional box of the "
                "cell with its periodic axes."
            )
        if stencil.cell_id != cell.cell_id:
            raise ValueError("The image stencil belongs to another PeriodicCell.")
        cutoff_ = _finite_length("cutoff", cutoff, positive=True)
        skin_ = _finite_length("skin", skin, positive=False)
        if stencil.radius < cutoff_ + skin_:
            raise ValueError("The image stencil radius must cover cutoff + skin.")
        if any(
            excursion < 1.0
            for excursion, periodic in zip(
                stencil.fractional_excursion, cell.periodic_axes, strict=True
            )
            if periodic
        ):
            raise ValueError(
                "The image stencil must admit wrapped endpoints (fractional_excursion >= 1)."
            )
        _require_complete_stencil(cell, stencil, cutoff_ + skin_)
        self.ownership = ownership
        self.partition = partition
        self.stencil = stencil
        self.streaming = streaming
        self.reduction = reduction_
        self.cutoff = cutoff_
        self.skin = skin_
        self.alias_capacity = _positive_capacity("alias_capacity", alias_capacity)
        self.edge_capacity = _positive_capacity("edge_capacity", edge_capacity)
        self.halo_capacity = _positive_capacity("halo_capacity", halo_capacity)
        self.migration_capacity = _positive_capacity(
            "migration_capacity", migration_capacity
        )
        self.message_capacity_bytes = _positive_capacity(
            "message_capacity_bytes", message_capacity_bytes
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "owner-local-atomistic",
                "ownership": ownership.plan_id,
                "partition": partition.partition_id,
                "stencil": stencil.stencil_id,
                "streaming": streaming.plan_id,
                "reduction": reduction_.policy_id,
                "cutoff": cutoff_,
                "skin": skin_,
                "alias_capacity": self.alias_capacity,
                "edge_capacity": self.edge_capacity,
                "halo_capacity": self.halo_capacity,
                "migration_capacity": self.migration_capacity,
                "message_capacity_bytes": self.message_capacity_bytes,
            }
        )

    @property
    def search_radius(self) -> float:
        return self.cutoff + self.skin

    @property
    def topology_owner_id(self) -> str:
        return f"owner-local-atomistic:{self.plan_id}"


class OwnerLocalTopologyEvidence(StrictModule):
    """Completeness, capacity, and identity evidence of one topology epoch."""

    __strict_contract__ = True

    successful: Bool[Scalar]
    owners_current: Bool[Scalar]
    positions_finite: Bool[Scalar]
    maximum_alias_load: Int32[Scalar]
    alias_capacity: Int32[Scalar]
    maximum_edge_count: Int32[Scalar]
    edge_capacity: Int32[Scalar]
    maximum_degree: Int32[Scalar]
    halo_successful: Bool[Scalar]
    refused_routes: Int32[Scalar]
    maximum_halo_load: Int32[Scalar]
    halo_capacity: Int32[Scalar]


@final
class OwnerLocalAtomisticTopology(StrictModule, NonTrainableState):
    """One accepted owner-local image graph epoch.

    Owner-blocked edge routes name their receiver row and the physical source
    atom by ``(route_owners, route_slots)`` with integer image ``shifts``;
    ``halo`` maps every route onto an owned or deduplicated halo column. The
    prepared streamed relation of every owner is retained as owner-stacked
    leaves of ``relation_tree`` and reused until the next epoch.
    ``reference_positions`` are the stored coordinates the candidate edges
    were certified for.
    """

    __strict_contract__ = True

    route_owners: Int32[_OwnerEdgeDim]
    route_slots: Int32[_OwnerEdgeDim]
    receivers: Int32[_OwnerEdgeDim]
    shifts: Int32[_OwnerEdgeDim, _LatticeRankDim]
    valid: Bool[_OwnerEdgeDim]
    halo: DistributedHaloPlan
    relation_leaves: tuple[Array, ...]
    reference_positions: Float[_OwnerRowDim, Literal[3]]
    owner_epochs: Int32[_OwnerCountDim]
    epoch: Int32[Scalar]
    evidence: OwnerLocalTopologyEvidence
    relation_tree: PyTreeDef = eqx.field(static=True)
    streaming_plan_id: str = eqx.field(static=True)

    @property
    def columns(self) -> Array:
        return self.halo.route_columns

    @property
    def edge_valid(self) -> Array:
        return self.valid & self.halo.route_valid


@final
class OwnerLocalAtomisticState(StrictModule):
    """Complete accepted owner-local continuation state.

    Per-atom rows are owner-blocked in the layout order: stored Cartesian
    positions (wrapped at the last accepted epoch), integer image counts that
    recover unwrapped trajectories, velocities, masses, model species
    indices, the force cache with the step it belongs to, and caller-declared
    per-atom continuation payload (constraint membership, per-atom thermostat
    or bias histories). Replicated state holds the typed RNG key, thermostat,
    bias, and constraint state, and the step index. ``model_revision_id`` binds
    the numeric model revision; ``topology`` binds the owner epoch.
    """

    __strict_contract__ = True

    layout: DistributedPointLayout
    positions: Float[_OwnerRowDim, Literal[3]]
    velocities: Float[_OwnerRowDim, Literal[3]]
    masses: Float[_OwnerRowDim]
    species: Int32[_OwnerRowDim]
    image_counts: Int32[_OwnerRowDim, _LatticeRankDim]
    force_cache: Float[_OwnerRowDim, Literal[3]]
    force_cache_step: Int32[Scalar]
    atom_payload: Mapping[str, Array]
    rng_key: PRNGKey
    thermostat_state: Array
    bias_state: Array
    constraint_state: Array
    step_index: Int32[Scalar]
    topology: OwnerLocalAtomisticTopology
    plan_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)

    @property
    def force_cache_valid(self) -> Array:
        return self.force_cache_step == self.step_index

    def with_dynamics(
        self,
        positions: ArrayLike,
        velocities: ArrayLike,
        /,
        *,
        step_index: ArrayLike,
        forces: ArrayLike | None = None,
        rng_key: PRNGKey | None = None,
        thermostat_state: ArrayLike | None = None,
        bias_state: ArrayLike | None = None,
        constraint_state: ArrayLike | None = None,
        atom_payload: Mapping[str, ArrayLike] | None = None,
    ) -> OwnerLocalAtomisticState:
        """Advance dynamical state within the current owner and topology epoch.

        ``forces`` (owner-blocked) become the force cache of ``step_index``.
        Inactive padding rows are kept exactly zero, so values an integrator
        computes there (for example from zero padding masses) never enter a
        later owner exchange. Ownership, layout, and topology change only
        through `rebuild_owner_local_atomistic`.
        """
        position = jnp.asarray(positions, dtype=self.positions.dtype)
        velocity = jnp.asarray(velocities, dtype=self.velocities.dtype)
        if (
            position.shape != self.positions.shape
            or velocity.shape != self.velocities.shape
        ):
            raise ValueError("Dynamics must preserve the owner-blocked row shape.")
        step = jnp.asarray(step_index, dtype=jnp.int32)
        if step.shape:
            raise ValueError("step_index must be a scalar.")
        cache = (
            self.force_cache
            if forces is None
            else jnp.asarray(forces, dtype=self.force_cache.dtype)
        )
        if cache.shape != self.force_cache.shape:
            raise ValueError("forces must match the owner-blocked row shape.")
        payload = (
            self.atom_payload
            if atom_payload is None
            else _validated_atom_payload(atom_payload, self.positions.shape[0])
        )
        if set(payload) != set(self.atom_payload):
            raise ValueError("atom_payload must keep its declared entries.")
        padding = ~self.layout.active[:, None]

        def rows(value: Array) -> Array:
            return self.layout.plan.place(jnp.where(padding, 0.0, value))

        return OwnerLocalAtomisticState(
            layout=self.layout,
            positions=rows(position),
            velocities=rows(velocity),
            masses=self.masses,
            species=self.species,
            image_counts=self.image_counts,
            force_cache=rows(cache),
            force_cache_step=step if forces is not None else self.force_cache_step,
            atom_payload=payload,
            rng_key=self.rng_key if rng_key is None else _typed_key(rng_key),
            thermostat_state=_same_shape(
                "thermostat_state", thermostat_state, self.thermostat_state
            ),
            bias_state=_same_shape("bias_state", bias_state, self.bias_state),
            constraint_state=_same_shape(
                "constraint_state", constraint_state, self.constraint_state
            ),
            step_index=step,
            topology=self.topology,
            plan_id=self.plan_id,
            model_revision_id=self.model_revision_id,
            run_id=self.run_id,
        )


def _same_shape(name: str, value: ArrayLike | None, previous: Array, /) -> Array:
    if value is None:
        return previous
    array = jnp.asarray(value, dtype=previous.dtype)
    if array.shape != previous.shape:
        raise ValueError(f"{name} must keep its declared shape.")
    return array


def _typed_key(value: PRNGKey, /) -> Array:
    key = jnp.asarray(value)
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key) or key.shape:
        raise TypeError("rng_key must be one typed key from jax.random.key.")
    return key


def _validated_atom_payload(
    payload: Mapping[str, ArrayLike], rows: int, /
) -> dict[str, Array]:
    result: dict[str, Array] = {}
    for name in sorted(payload):
        array = jnp.asarray(payload[name])
        if not array.shape or array.shape[0] != rows:
            raise ValueError(f"atom_payload[{name!r}] must lead with the atom rows.")
        result[name] = array
    return result


class OwnerLocalExecutionStatus(StrictModule):
    """Fail-closed topology, certificate, numerical, and identity status."""

    __strict_contract__ = True

    topology_successful: Bool[Scalar]
    owners_current: Bool[Scalar]
    displacement_certified: Bool[Scalar]
    maximum_displacement: Float[Scalar]
    relation_successful: Bool[Scalar]
    finite: Bool[Scalar]
    successful: Bool[Scalar]


@final
class OwnerLocalAtomisticEvaluation(StrictModule):
    """Owner-local energy, forces, strain derivative, and stress.

    ``atom_energies`` and ``forces`` are owner-blocked rows of the state
    layout. ``owner_energies`` are the per-owner sums; ``energy`` is their sum
    in owner order. ``strain_gradient`` is ``dE/d strain`` at fixed fractional
    coordinates (row cell ``H' = H @ F.T``), the owner-ordered sum of every
    owner's edge partial. ``stress = sym(strain_gradient) / volume`` exists
    only for fully periodic three-dimensional cells. Failed execution poisons
    every numerical output with NaN.
    """

    __strict_contract__ = True

    energy: Float[Scalar]
    owner_energies: Float[_OwnerCountDim]
    atom_energies: Float[_OwnerRowDim]
    forces: Float[_OwnerRowDim, Literal[3]]
    strain_gradient: Float[Literal[3], Literal[3]]
    stress: Float[Literal[3], Literal[3]]
    stress_available: bool = eqx.field(static=True)
    status: OwnerLocalExecutionStatus
    successful: Bool[Scalar]


@final
class OwnerLocalTransition(StrictModule):
    """Result of one migration plus topology epoch transaction."""

    state: OwnerLocalAtomisticState
    committed: Array
    migration: DistributedMigrationEvidence
    topology: OwnerLocalTopologyEvidence


@final
class OwnerLocalLossGradient(StrictModule):
    """Energy/force/stress loss and its PARAMETER-lane gradient."""

    loss: Array
    parameter_gradient: PyTree[Array]
    evaluation: OwnerLocalAtomisticEvaluation
    successful: Array


def _require_model(
    plan: OwnerLocalAtomisticPlan, model: AtomisticLayeredModel, /
) -> StreamedRelationPlan:
    """Admit a layered potential and return the plan's owner relation tiling."""
    if not isinstance(model, AbstractAtomisticPotential):
        raise TypeError("Owner-local execution requires an AbstractAtomisticPotential.")
    model_cutoff = model.requirements.cutoff
    if model_cutoff is None or model_cutoff > plan.cutoff:
        raise ValueError("The owner-local cutoff must cover the model cutoff.")
    return plan.streaming


def _model_revision(model: AtomisticLayeredModel, /) -> str:
    """Numeric PARAMETER-lane revision of an admitted layered potential (host)."""
    if not isinstance(model, AbstractAtomisticPotential):
        raise TypeError("Owner-local execution requires an AbstractAtomisticPotential.")
    return atomistic_potential_revision(model).revision_id


@eqx.filter_jit
def _owner_edges(
    plan: OwnerLocalAtomisticPlan, positions: Array, active: Array
) -> tuple[Array, ...]:
    """Alias exchange plus owner-dense candidate filtering of every owner.

    Candidate sources of one owner are its own rows (translation zero) and the
    received image aliases. The filter is an owner-local dense bound of
    ``local_capacity x (local_capacity + owner_count * alias_capacity)``
    candidates, charged by the plan capacities; it never forms global pairs.
    Edges are emitted receiver-major in candidate order.
    """
    ownership = plan.ownership
    axis = ownership.axis_name
    local = ownership.local_capacity
    edges = plan.edge_capacity
    radius = plan.search_radius
    cell = plan.partition.cell

    def body(
        rows: Array, alive: Array, vectors: Array, inverse: Array, shifts: Array
    ) -> tuple[Array, ...]:
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        aliases = plan.partition.local_alias_exchange(
            rows, alive, vectors, inverse, shifts, radius, axis, plan.alias_capacity
        )
        candidates = jnp.concatenate((rows, aliases.positions), axis=0)
        candidate_valid = jnp.concatenate((alive, aliases.valid), axis=0)
        count = candidates.shape[0]
        separation = rows[:, None, :] - candidates[None, :, :]
        squared = jnp.sum(separation * separation, axis=-1)
        same_row = jnp.arange(local)[:, None] == jnp.arange(count)[None, :]
        selected = (
            alive[:, None]
            & candidate_valid[None, :]
            & ~same_row
            & (squared <= radius * radius)
        )
        total = jnp.sum(selected, dtype=jnp.int32)
        flat = jnp.nonzero(selected.reshape((-1,)), size=edges, fill_value=0)[0]
        valid = jnp.arange(edges, dtype=jnp.int32) < total
        receiver = (flat // count).astype(jnp.int32)
        candidate = (flat % count).astype(jnp.int32)
        owned = candidate < local
        alias = jnp.clip(candidate - local, 0, aliases.slots.shape[0] - 1)
        route_owner = jnp.where(owned, me, aliases.owners[alias])
        route_slot = jnp.where(owned, candidate, aliases.slots[alias])
        translation = jnp.where(owned[:, None], 0, aliases.translations[alias])
        statistics = jnp.stack(
            (
                aliases.maximum_load,
                total,
                jnp.max(jnp.sum(selected, axis=1, dtype=jnp.int32)),
            )
        ).astype(jnp.int32)
        return (
            jnp.where(valid, receiver, 0),
            jnp.where(valid, route_owner, 0),
            jnp.where(valid, route_slot, 0),
            jnp.where(valid[:, None], -translation, 0),
            valid,
            statistics[None],
        )

    owner = PartitionSpec(axis)
    replicated = PartitionSpec()
    return ownership.map(
        body,
        (owner, owner, replicated, replicated, replicated),
        (owner, owner, owner, owner, owner, owner),
    )(
        positions,
        active,
        cell.vectors.astype(positions.dtype),
        cell.inverse_vectors.astype(positions.dtype),
        plan.stencil.shifts,
    )


def _relation_function(
    plan: OwnerLocalAtomisticPlan, streaming: StreamedRelationPlan, column_count: int
) -> Callable[[Array, Array, Array, Array, Array], PreparedStreamedRelation]:
    local = plan.ownership.local_capacity
    owner_id = plan.topology_owner_id

    def prepare(
        receivers: Array, columns: Array, valid: Array, alive: Array, epoch: Array
    ) -> PreparedStreamedRelation:
        relation = EdgeRelation(
            columns,
            receivers,
            source_size=column_count,
            target_size=local,
            valid=valid,
        )
        return streaming.prepare(
            relation, owner_id=owner_id, epoch=epoch, receiver_valid=alive
        )

    return prepare


@eqx.filter_jit
def _owner_relations(
    plan: OwnerLocalAtomisticPlan,
    streaming: StreamedRelationPlan,
    column_count: int,
    receivers: Array,
    columns: Array,
    valid: Array,
    active: Array,
    epoch: Array,
) -> tuple[Array, ...]:
    """Prepare every owner's streamed relation once for this topology epoch."""
    prepare = _relation_function(plan, streaming, column_count)
    axis = plan.ownership.axis_name

    def body(
        receiver: Array, column: Array, edge: Array, alive: Array, step: Array
    ) -> tuple[Array, ...]:
        return tuple(
            _owner_stack(jax.tree.leaves(prepare(receiver, column, edge, alive, step)))
        )

    owner = PartitionSpec(axis)
    leaves = jax.eval_shape(
        lambda: jax.tree.leaves(
            prepare(
                receivers[: plan.edge_capacity],
                columns[: plan.edge_capacity],
                valid[: plan.edge_capacity],
                active[: plan.ownership.local_capacity],
                epoch,
            )
        )
    )
    return plan.ownership.map(
        body,
        (owner, owner, owner, owner, PartitionSpec()),
        tuple(owner for _ in leaves),
    )(receivers, columns, valid, active, epoch)


def _relation_tree(
    plan: OwnerLocalAtomisticPlan,
    streaming: StreamedRelationPlan,
    column_count: int,
    dtype_epoch: Array,
) -> PyTreeDef:
    """Static structure of one owner's prepared relation (no execution)."""
    edges = plan.edge_capacity
    local = plan.ownership.local_capacity
    prepare = _relation_function(plan, streaming, column_count)
    shapes = jax.eval_shape(
        prepare,
        jax.ShapeDtypeStruct((edges,), jnp.int32),
        jax.ShapeDtypeStruct((edges,), jnp.int32),
        jax.ShapeDtypeStruct((edges,), jnp.bool_),
        jax.ShapeDtypeStruct((local,), jnp.bool_),
        jax.ShapeDtypeStruct((), dtype_epoch.dtype),
    )
    return jax.tree.structure(shapes)


def _build_topology(
    plan: OwnerLocalAtomisticPlan,
    streaming: StreamedRelationPlan,
    layout: DistributedPointLayout,
    positions: Array,
    epoch: Array,
    /,
) -> OwnerLocalAtomisticTopology:
    """Discover, route, and prepare one owner-local topology epoch."""
    receivers, route_owners, route_slots, shifts, valid, statistics = _owner_edges(
        plan, positions, layout.active
    )
    halo = DistributedHaloPlan(
        plan.ownership,
        route_owners,
        route_slots,
        valid,
        halo_capacity=plan.halo_capacity,
    )
    return _bind_topology(
        plan,
        streaming,
        layout,
        positions,
        epoch,
        layout.owner_epochs,
        (receivers, route_owners, route_slots, shifts, valid),
        halo,
        statistics,
    )


def _bind_topology(
    plan: OwnerLocalAtomisticPlan,
    streaming: StreamedRelationPlan,
    layout: DistributedPointLayout,
    reference_positions: Array,
    epoch: Array,
    owner_epochs: Array,
    routes: tuple[Array, Array, Array, Array, Array],
    halo: DistributedHaloPlan,
    statistics: Array,
    /,
) -> OwnerLocalAtomisticTopology:
    receivers, route_owners, route_slots, shifts, valid = routes
    epoch_ = jnp.asarray(epoch, dtype=jnp.int32)
    edge_valid = valid & halo.route_valid
    leaves = _owner_relations(
        plan,
        streaming,
        halo.column_count,
        receivers,
        halo.route_columns,
        edge_valid,
        layout.active,
        epoch_,
    )
    # ``owner_epochs`` is the layout epoch this topology was discovered for; a
    # restore passes the recorded witness instead of re-reading the layout.
    owner_epochs = jnp.asarray(owner_epochs).astype(jnp.int32)
    owners_current = jnp.all(owner_epochs == owner_epochs[0]) & jnp.all(
        owner_epochs == layout.owner_epochs.astype(jnp.int32)
    )
    finite = jnp.all(
        jnp.where(layout.active[:, None], jnp.isfinite(reference_positions), True)
    )
    maximum_alias = jnp.max(statistics[:, 0])
    maximum_edges = jnp.max(statistics[:, 1])
    alias_capacity = jnp.asarray(plan.alias_capacity, dtype=jnp.int32)
    edge_capacity = jnp.asarray(plan.edge_capacity, dtype=jnp.int32)
    successful = (
        owners_current
        & finite
        & (maximum_alias <= alias_capacity)
        & (maximum_edges <= edge_capacity)
        & halo.evidence.successful
    )
    evidence = OwnerLocalTopologyEvidence(
        successful=successful,
        owners_current=owners_current,
        positions_finite=finite,
        maximum_alias_load=maximum_alias,
        alias_capacity=alias_capacity,
        maximum_edge_count=maximum_edges,
        edge_capacity=edge_capacity,
        maximum_degree=jnp.max(statistics[:, 2]),
        halo_successful=halo.evidence.successful,
        refused_routes=halo.evidence.refused_routes,
        maximum_halo_load=halo.evidence.maximum_halo_load,
        halo_capacity=halo.evidence.halo_capacity,
    )
    return OwnerLocalAtomisticTopology(
        route_owners=route_owners,
        route_slots=route_slots,
        receivers=receivers,
        shifts=shifts,
        valid=valid,
        halo=halo,
        relation_leaves=tuple(leaves),
        reference_positions=reference_positions,
        owner_epochs=owner_epochs,
        epoch=epoch_,
        evidence=evidence,
        relation_tree=_relation_tree(plan, streaming, halo.column_count, epoch_),
        streaming_plan_id=streaming.plan_id,
    )


def _wrapped_fractional(
    plan: OwnerLocalAtomisticPlan, positions: Array, /
) -> tuple[Array, Array]:
    """Wrapped fractional coordinates and integer wraps of stored positions."""
    cell = plan.partition.cell
    fractional = cell.fractional(positions)
    images = jnp.where(cell.periodic_mask, jnp.floor(fractional), 0.0)
    return fractional - images, images.astype(jnp.int32)


def prepare_owner_local_atomistic(
    plan: OwnerLocalAtomisticPlan,
    model: AtomisticLayeredModel,
    positions: ArrayLike,
    species: ArrayLike,
    /,
    *,
    rng_key: PRNGKey,
    velocities: ArrayLike | None = None,
    masses: ArrayLike | None = None,
    stable_ids: ArrayLike | None = None,
    atom_payload: Mapping[str, ArrayLike] | None = None,
    thermostat_state: ArrayLike | None = None,
    bias_state: ArrayLike | None = None,
    constraint_state: ArrayLike | None = None,
    step_index: int = 0,
    run_id: str | None = None,
) -> OwnerLocalAtomisticState:
    """Host ingress: wrap, own, distribute, and build the first topology.

    ``species`` are the model's native species indices. Every per-atom input
    is in logical atom order. Ingress refuses a failed first topology with
    its evidence instead of returning an unusable state.
    """
    streaming = _require_model(plan, model)
    coordinates = np.asarray(positions)
    if coordinates.ndim != 2 or coordinates.shape[1] != _AMBIENT_DIMENSION:
        raise ValueError("positions must have shape (atoms, 3).")
    if not np.issubdtype(coordinates.dtype, np.floating):
        raise TypeError("positions must have a floating dtype.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("positions must be finite.")
    count = coordinates.shape[0]
    dtype = coordinates.dtype
    species_ = np.asarray(species)
    if species_.shape != (count,) or not np.issubdtype(species_.dtype, np.integer):
        raise ValueError("species must hold one integer species index per atom.")
    velocity = (
        np.zeros_like(coordinates) if velocities is None else np.asarray(velocities)
    )
    mass = np.ones((count,), dtype=dtype) if masses is None else np.asarray(masses)
    if velocity.shape != coordinates.shape or mass.shape != (count,):
        raise ValueError("velocities and masses must match the atom rows.")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("masses must be finite and positive.")
    wrapped, images = _wrapped_fractional(plan, jnp.asarray(coordinates))
    cell = plan.partition.cell
    stored = np.asarray(
        jnp.asarray(coordinates)
        - contract("na,ad->nd", images.astype(dtype), cell.vectors.astype(dtype))
    )
    owners = np.asarray(plan.partition.owners(wrapped))
    layout = DistributedPointLayout.from_global(
        plan.ownership, np.asarray(wrapped), owners, stable_ids=stable_ids, epoch=0
    )
    payload = _validated_atom_payload({} if atom_payload is None else atom_payload, count)
    blocked_positions = layout.distribute(stored)
    epoch = jnp.zeros((), dtype=jnp.int32)
    topology = _build_topology(plan, streaming, layout, blocked_positions, epoch)
    if not bool(topology.evidence.successful):
        raise ValueError(
            "The initial owner-local topology was refused: "
            f"{jax.device_get(topology.evidence)}."
        )
    revision = _model_revision(model)
    run = (
        canonical_fingerprint(
            {"kind": "owner-local-atomistic-run", "plan": plan.plan_id, "model": revision}
        )
        if run_id is None
        else _nonempty_identity("run_id", run_id)
    )
    step = jnp.asarray(step_index, dtype=jnp.int32)
    return OwnerLocalAtomisticState(
        layout=layout,
        positions=blocked_positions,
        velocities=layout.distribute(velocity.astype(dtype)),
        masses=layout.distribute(mass.astype(dtype)),
        species=layout.distribute(species_.astype(np.int32)),
        image_counts=layout.distribute(np.asarray(images)),
        force_cache=layout.distribute(np.zeros_like(coordinates)),
        force_cache_step=step - 1,
        atom_payload={name: layout.distribute(value) for name, value in payload.items()},
        rng_key=_typed_key(rng_key),
        thermostat_state=jnp.zeros((0,), dtype)
        if thermostat_state is None
        else jnp.asarray(thermostat_state),
        bias_state=jnp.zeros((0,), dtype)
        if bias_state is None
        else jnp.asarray(bias_state),
        constraint_state=jnp.zeros((0,), dtype)
        if constraint_state is None
        else jnp.asarray(constraint_state),
        step_index=step,
        topology=topology,
        plan_id=plan.plan_id,
        model_revision_id=revision,
        run_id=run,
    )


def _require_state(
    plan: OwnerLocalAtomisticPlan,
    streaming: StreamedRelationPlan,
    state: OwnerLocalAtomisticState,
    /,
) -> None:
    if not isinstance(state, OwnerLocalAtomisticState):
        raise TypeError("state must be an OwnerLocalAtomisticState.")
    if state.plan_id != plan.plan_id:
        raise ValueError("The owner-local state belongs to another plan.")
    if state.topology.streaming_plan_id != streaming.plan_id:
        raise ValueError("The topology was prepared for another streamed relation plan.")


def _select_tree[T](predicate: Array, candidate: T, previous: T, /) -> T:
    """Leafwise transactional selection of two equally structured trees."""

    def select(new: Array, old: Array) -> Array:
        if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
            return jax.random.wrap_key_data(
                jnp.where(predicate, jax.random.key_data(new), jax.random.key_data(old)),
                impl=jax.random.key_impl(old),
            )
        return jnp.where(predicate, new, old)

    return jax.tree.map(select, candidate, previous)


def rebuild_owner_local_atomistic(
    plan: OwnerLocalAtomisticPlan,
    model: AtomisticLayeredModel,
    state: OwnerLocalAtomisticState,
    /,
) -> OwnerLocalTransition:
    """Wrap, migrate, and rebuild one owner epoch as a single transaction.

    Every atom moves with its complete continuation payload to the owner of
    its wrapped fractional coordinate; stored positions are wrapped by whole
    lattice translations and the translations are added to the image counts.
    The new topology is then discovered for the migrated layout. Migration
    overflow, epoch inconsistency, alias/edge/halo overflow, or non-finite
    positions on any owner return the accepted state unchanged together with
    the failed attempt's evidence.
    """
    streaming = _require_model(plan, model)
    _require_state(plan, streaming, state)
    layout = state.layout
    cell = plan.partition.cell
    dtype = state.positions.dtype
    wrapped, wraps = _wrapped_fractional(plan, state.positions)
    destinations = jnp.where(layout.active, plan.partition.owners(wrapped), 0)
    payload = {
        "positions": state.positions,
        "velocities": state.velocities,
        "masses": state.masses,
        "species": state.species,
        "image_counts": state.image_counts,
        "force_cache": state.force_cache,
        "wraps": wraps,
        "atom_payload": dict(state.atom_payload),
    }
    migrated = layout.migrate(
        destinations,
        packet_capacity=plan.migration_capacity,
        points=wrapped.astype(layout.points.dtype),
        payload=payload,
    )
    moved = migrated.payload
    translation = contract(
        "na,ad->nd", moved["wraps"].astype(dtype), cell.vectors.astype(dtype)
    )
    new_layout = migrated.layout
    alive = new_layout.active[:, None]
    positions = jnp.where(alive, moved["positions"] - translation, 0.0)
    topology = _build_topology(
        plan, streaming, new_layout, positions, state.topology.epoch + 1
    )
    committed = migrated.evidence.committed & topology.evidence.successful
    candidate = OwnerLocalAtomisticState(
        layout=new_layout,
        positions=positions,
        velocities=moved["velocities"],
        masses=moved["masses"],
        species=moved["species"],
        image_counts=moved["image_counts"] + moved["wraps"],
        force_cache=moved["force_cache"],
        force_cache_step=state.force_cache_step,
        atom_payload=moved["atom_payload"],
        rng_key=state.rng_key,
        thermostat_state=state.thermostat_state,
        bias_state=state.bias_state,
        constraint_state=state.constraint_state,
        step_index=state.step_index,
        topology=topology,
        plan_id=state.plan_id,
        model_revision_id=state.model_revision_id,
        run_id=state.run_id,
    )
    return OwnerLocalTransition(
        state=_select_tree(committed, candidate, state),
        committed=committed,
        migration=migrated.evidence,
        topology=topology.evidence,
    )


class _OwnerInputs(NamedTuple):
    """Owner-blocked operands of one owner-region evaluation."""

    positions: Array
    species: Array
    active: Array
    receivers: Array
    columns: Array
    shifts: Array
    valid: Array
    send_slots: Array
    send_valid: Array
    reference: Array
    relation: tuple[Array, ...]


def _owner_inputs(state: OwnerLocalAtomisticState, /) -> _OwnerInputs:
    topology = state.topology
    return _OwnerInputs(
        positions=state.positions,
        species=state.species,
        active=state.layout.active,
        receivers=topology.receivers,
        columns=topology.columns,
        shifts=topology.shifts,
        valid=topology.edge_valid,
        send_slots=topology.halo.send_slots,
        send_valid=topology.halo.send_valid,
        reference=topology.reference_positions,
        relation=topology.relation_leaves,
    )


def _owner_view(
    topology: OwnerLocalAtomisticTopology, inputs: _OwnerInputs, /
) -> OwnerLayerTopology:
    return OwnerLayerTopology(
        relation=jax.tree.unflatten(
            topology.relation_tree, _owner_unstack(inputs.relation)
        ),
        receivers=inputs.receivers,
        columns=inputs.columns,
        shifts=inputs.shifts,
        valid=inputs.valid,
        send_slots=inputs.send_slots,
        send_valid=inputs.send_valid,
    )


def _owner_displacement(inputs: _OwnerInputs, /) -> Array:
    moved = inputs.positions - inputs.reference
    distance = jnp.sqrt(jnp.sum(moved * moved, axis=-1))
    return jnp.max(jnp.where(inputs.active, distance, 0.0))


@eqx.filter_jit
def _evaluate_owner_regions(
    plan: OwnerLocalAtomisticPlan,
    model: AtomisticLayeredModel,
    state: OwnerLocalAtomisticState,
    gradients: bool,
) -> tuple[Array, ...]:
    topology = state.topology
    halo = topology.halo
    axis = plan.ownership.axis_name
    arrays, structure = eqx.partition(model, eqx.is_array)
    dtype = state.positions.dtype
    vectors = plan.partition.cell.vectors.astype(dtype)

    def body(inputs: _OwnerInputs, parameters: PyTree[Array]) -> tuple[Array, ...]:
        local_model = eqx.combine(parameters, structure)
        view = _owner_view(topology, inputs)
        if gradients:
            result = owner_layer_gradients(
                local_model,
                halo,
                view,
                inputs.positions,
                inputs.species,
                inputs.active,
                vectors,
                plan.cutoff,
                message_capacity_bytes=plan.message_capacity_bytes,
                parameters=False,
            )
            atom_energy = result.atom_energies
            gradient = result.gradient
            strain = result.strain_gradient
            relation_successful = result.relation_successful
        else:
            atom_energy, relation_successful = owner_atom_energies(
                local_model,
                halo,
                view,
                inputs.positions,
                inputs.species,
                inputs.active,
                vectors,
                plan.cutoff,
                message_capacity_bytes=plan.message_capacity_bytes,
            )
            gradient = jnp.zeros_like(inputs.positions)
            strain = jnp.zeros((3, 3), dtype=dtype)
        atom_energy = jnp.where(inputs.active, atom_energy, 0.0)
        owner_energy = _ordered_sum(atom_energy, plan.reduction)
        finite = (
            jnp.all(jnp.isfinite(atom_energy))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(strain))
        )
        return (
            atom_energy,
            gradient,
            owner_energy[None],
            strain[None],
            _owner_displacement(inputs)[None],
            finite[None],
            relation_successful[None],
        )

    owner = PartitionSpec(axis)
    return plan.ownership.map(
        body,
        (owner, PartitionSpec()),
        (owner,) * 7,
    )(_owner_inputs(state), arrays)


def _execution_status(
    plan: OwnerLocalAtomisticPlan,
    state: OwnerLocalAtomisticState,
    displacement: Array,
    finite: Array,
    relation: Array,
    /,
) -> OwnerLocalExecutionStatus:
    topology = state.topology
    owners_current = jnp.all(
        topology.owner_epochs == state.layout.owner_epochs.astype(jnp.int32)
    )
    maximum = jnp.max(displacement)
    certified = 2.0 * maximum <= plan.skin
    finite_ = jnp.all(finite)
    relation_ = jnp.all(relation)
    successful = (
        topology.evidence.successful & owners_current & certified & finite_ & relation_
    )
    return OwnerLocalExecutionStatus(
        topology_successful=topology.evidence.successful,
        owners_current=owners_current,
        displacement_certified=certified,
        maximum_displacement=maximum,
        relation_successful=relation_,
        finite=finite_,
        successful=successful,
    )


def evaluate_owner_local_atomistic(
    plan: OwnerLocalAtomisticPlan,
    model: AtomisticLayeredModel,
    state: OwnerLocalAtomisticState,
    /,
    *,
    forces: bool = True,
) -> OwnerLocalAtomisticEvaluation:
    """Partition-local energy, forces, and strain derivative of a layered model.

    Every owner evaluates only its receivers. Before each interaction the
    model's source payload of owned rows crosses the halo once; reverse
    cotangents return through the halo transpose once per interaction in
    reverse order. Owner energies, strain partials, and statuses reduce in
    owner order. A stale owner epoch, an expired displacement certificate
    (``2 max|x - x_ref| > skin``), a refused topology, or non-finite output
    fails the evaluation and poisons every value. The model revision binding
    is checked at ingress, restore, and `rebind_owner_local_model`, not by
    hashing parameters on every call.
    """
    streaming = _require_model(plan, model)
    _require_state(plan, streaming, state)
    if not isinstance(forces, bool):
        raise TypeError("forces must be a bool.")
    atom_energy, gradient, owner_energy, strain, displacement, finite, relation = (
        _evaluate_owner_regions(plan, model, state, forces)
    )
    status = _execution_status(plan, state, displacement, finite, relation)
    energy = _ordered_sum(owner_energy, plan.reduction)
    strain_gradient = _ordered_sum(strain, plan.reduction)
    cell = plan.partition.cell
    stress_available = forces and cell.fully_periodic and cell.rank == 3
    stress = (
        0.5 * (strain_gradient + strain_gradient.T) / cell.volume
        if stress_available
        else jnp.full((3, 3), jnp.nan, dtype=strain_gradient.dtype)
    )
    accepted = status.successful
    nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
    force = jnp.where(state.layout.active[:, None], -gradient, 0.0)
    return OwnerLocalAtomisticEvaluation(
        energy=jnp.where(accepted, energy, nan),
        owner_energies=jnp.where(accepted, owner_energy, nan),
        atom_energies=jnp.where(accepted, atom_energy, nan),
        forces=jnp.where(accepted & forces, force, nan),
        strain_gradient=jnp.where(accepted & forces, strain_gradient, nan),
        stress=jnp.where(accepted, stress, nan),
        stress_available=stress_available,
        status=status,
        successful=accepted,
    )


@eqx.filter_jit
def _owner_parameter_partials(
    plan: OwnerLocalAtomisticPlan,
    model: AtomisticLayeredModel,
    state: OwnerLocalAtomisticState,
    force_reference: Array,
    energy_factor: Array,
    force_weight: Array,
    stress_factor: Array,
) -> PyTree[Array]:
    """Per-owner PARAMETER-lane partials of the linearized global loss.

    Each owner differentiates its local surrogate
    ``energy_factor * E_owner + force_weight * sum |F - F_ref|^2 +
    <stress_factor, strain_partial>`` through its own explicit reverse sweep,
    including every halo gather and transpose; the sum over owners of these
    partials is the gradient of the global loss.
    """
    topology = state.topology
    halo = topology.halo
    axis = plan.ownership.axis_name
    arrays, structure = eqx.partition(model, eqx.is_array)
    dtype = state.positions.dtype
    vectors = plan.partition.cell.vectors.astype(dtype)

    def body(
        inputs: _OwnerInputs,
        reference: Array,
        parameters: PyTree[Array],
        factors: tuple[Array, Array, Array],
    ) -> PyTree[Array]:
        energy_scale, weight, stress_scale = factors
        lanes = partition_parameters(eqx.combine(parameters, structure))
        view = _owner_view(topology, inputs)

        def surrogate(lane: PyTree[Array]) -> Array:
            local_model = combine_parameters(lane, lanes[1], lanes[2])
            result = owner_layer_gradients(
                local_model,
                halo,
                view,
                inputs.positions,
                inputs.species,
                inputs.active,
                vectors,
                plan.cutoff,
                message_capacity_bytes=plan.message_capacity_bytes,
                parameters=False,
            )
            owned = jnp.where(inputs.active, result.atom_energies, 0.0)
            residual = jnp.where(
                inputs.active[:, None], -result.gradient - reference, 0.0
            )
            return (
                energy_scale * _ordered_sum(owned, plan.reduction)
                + weight * jnp.sum(residual * residual)
                + jnp.sum(stress_scale * result.strain_gradient)
            )

        return _owner_stack(jax.grad(surrogate)(lanes[0]))

    owner = PartitionSpec(axis)
    replicated = PartitionSpec()
    return plan.ownership.map(
        body,
        (owner, owner, replicated, replicated),
        owner,
    )(
        _owner_inputs(state),
        force_reference,
        arrays,
        (energy_factor, force_weight, stress_factor),
    )


def owner_local_loss_gradient(
    plan: OwnerLocalAtomisticPlan,
    model: AtomisticLayeredModel,
    state: OwnerLocalAtomisticState,
    energy_reference: ArrayLike,
    force_reference: ArrayLike,
    /,
    *,
    energy_weight: float,
    force_weight: float,
    stress_reference: ArrayLike | None = None,
    stress_weight: float = 0.0,
) -> OwnerLocalLossGradient:
    """Owner-local E/F/S loss and its exact PARAMETER-lane gradient.

    ``loss = w_E (E - E_ref)^2 + w_F sum_atoms |F - F_ref|^2 +
    w_S |stress - stress_ref|^2``; ``force_reference`` is in logical atom
    order. The gradient differentiates the owner-local forces themselves
    (mixed coordinate/parameter derivatives through every halo exchange) and
    reduces the per-owner partials in owner order. A failed evaluation fails
    the gradient and poisons it with NaN.
    """
    evaluation = evaluate_owner_local_atomistic(plan, model, state, forces=True)
    dtype = state.positions.dtype
    reference_energy = jnp.asarray(energy_reference, dtype=dtype)
    if reference_energy.shape:
        raise ValueError("energy_reference must be a scalar.")
    references = jnp.asarray(force_reference, dtype=dtype)
    if references.shape != (state.layout.logical_count, 3):
        raise ValueError("force_reference must have shape (atoms, 3).")
    blocked_reference = state.layout.distribute(references)
    weights = tuple(
        _finite_length(name, value, positive=False)
        for name, value in (
            ("energy_weight", energy_weight),
            ("force_weight", force_weight),
            ("stress_weight", stress_weight),
        )
    )
    energy_residual = evaluation.energy - reference_energy
    force_residual = jnp.where(
        state.layout.active[:, None], evaluation.forces - blocked_reference, 0.0
    )
    loss = weights[0] * energy_residual**2 + weights[1] * jnp.sum(force_residual**2)
    stress_factor = jnp.zeros((3, 3), dtype=dtype)
    if stress_reference is not None or weights[2] != 0.0:
        if not evaluation.stress_available or stress_reference is None:
            raise ValueError(
                "Stress supervision requires a stress reference and a fully "
                "periodic three-dimensional cell."
            )
        stress_target = jnp.asarray(stress_reference, dtype=dtype)
        if stress_target.shape != (3, 3):
            raise ValueError("stress_reference must have shape (3, 3).")
        stress_residual = evaluation.stress - stress_target
        loss = loss + weights[2] * jnp.sum(stress_residual**2)
        symmetric = 0.5 * (stress_residual + stress_residual.T)
        stress_factor = 2.0 * weights[2] * symmetric / plan.partition.cell.volume
    partials = _owner_parameter_partials(
        plan,
        model,
        state,
        blocked_reference,
        2.0 * weights[0] * energy_residual,
        jnp.asarray(weights[1], dtype=dtype),
        stress_factor,
    )
    gradient = jax.tree.map(lambda value: _ordered_sum(value, plan.reduction), partials)
    accepted = evaluation.successful
    return OwnerLocalLossGradient(
        loss=jnp.where(accepted, loss, jnp.nan),
        parameter_gradient=jax.tree.map(
            lambda value: jnp.where(accepted, value, jnp.nan), gradient
        ),
        evaluation=evaluation,
        successful=accepted,
    )


def rebind_owner_local_model(
    plan: OwnerLocalAtomisticPlan,
    model: AtomisticLayeredModel,
    state: OwnerLocalAtomisticState,
    /,
) -> OwnerLocalAtomisticState:
    """Bind an updated model revision; the force cache becomes stale."""
    streaming = _require_model(plan, model)
    _require_state(plan, streaming, state)
    revision = _model_revision(model)
    if revision == state.model_revision_id:
        return state
    return OwnerLocalAtomisticState(
        layout=state.layout,
        positions=state.positions,
        velocities=state.velocities,
        masses=state.masses,
        species=state.species,
        image_counts=state.image_counts,
        force_cache=state.force_cache,
        force_cache_step=state.step_index - 1,
        atom_payload=state.atom_payload,
        rng_key=state.rng_key,
        thermostat_state=state.thermostat_state,
        bias_state=state.bias_state,
        constraint_state=state.constraint_state,
        step_index=state.step_index,
        topology=state.topology,
        plan_id=state.plan_id,
        model_revision_id=revision,
        run_id=state.run_id,
    )


@final
class OwnerLocalAtomisticCheckpoint(StrictModule, NonTrainableState):
    """Host continuation record of an owner-local state and its identity.

    ``arrays`` holds every owner-blocked and replicated continuation array,
    including the accepted topology routes and reference positions, so a
    restored run continues bitwise like the uninterrupted one. Derived halo
    columns and prepared relations are rebuilt through their constructors.
    """

    arrays: Mapping[str, np.ndarray]
    key_implementation: str = eqx.field(static=True)
    logical_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


def _checkpoint_arrays(state: OwnerLocalAtomisticState, /) -> dict[str, np.ndarray]:
    layout = state.layout
    topology = state.topology
    records: dict[str, Array] = {
        "layout/points": layout.points,
        "layout/stable_ids": layout.stable_ids,
        "layout/active": layout.active,
        "layout/logical_indices": layout.logical_indices,
        "layout/owner_epochs": layout.owner_epochs,
        "layout/stable_ids_unique": layout.stable_ids_unique,
        "atoms/positions": state.positions,
        "atoms/velocities": state.velocities,
        "atoms/masses": state.masses,
        "atoms/species": state.species,
        "atoms/image_counts": state.image_counts,
        "atoms/force_cache": state.force_cache,
        "state/force_cache_step": state.force_cache_step,
        "state/rng_key": jax.random.key_data(state.rng_key),
        "state/thermostat": state.thermostat_state,
        "state/bias": state.bias_state,
        "state/constraint": state.constraint_state,
        "state/step_index": state.step_index,
        "topology/route_owners": topology.route_owners,
        "topology/route_slots": topology.route_slots,
        "topology/receivers": topology.receivers,
        "topology/shifts": topology.shifts,
        "topology/valid": topology.valid,
        "topology/reference_positions": topology.reference_positions,
        "topology/epoch": topology.epoch,
        "topology/owner_epochs": topology.owner_epochs,
        "topology/maximum_alias_load": topology.evidence.maximum_alias_load,
        "topology/maximum_edge_count": topology.evidence.maximum_edge_count,
        "topology/maximum_degree": topology.evidence.maximum_degree,
    }
    for name, value in state.atom_payload.items():
        records[f"payload/{name}"] = value
    return {name: np.asarray(jax.device_get(value)) for name, value in records.items()}


def _checkpoint_identity(
    arrays: Mapping[str, np.ndarray],
    key_implementation: str,
    logical_count: int,
    plan_id: str,
    model_revision_id: str,
    run_id: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "owner-local-atomistic-checkpoint",
            "plan": plan_id,
            "model_revision": model_revision_id,
            "run": run_id,
            "key_implementation": key_implementation,
            "logical_count": logical_count,
            "arrays": array_tree_fingerprint(dict(arrays)),
        }
    )


def _require_coherent_epochs(
    layout_epochs: np.ndarray, topology_epochs: np.ndarray, /
) -> None:
    """Refuse stale owners or a topology discovered for another layout epoch."""
    layout_ = np.asarray(layout_epochs).astype(np.int64)
    topology_ = np.asarray(topology_epochs).astype(np.int64)
    if layout_.shape != topology_.shape or not np.all(layout_ == layout_[0]):
        raise ValueError("The owner-local layout has stale owner epochs.")
    if not np.array_equal(layout_, topology_):
        raise ValueError(
            "The owner-local topology was discovered for another layout epoch; "
            "rebuild it with rebuild_owner_local_atomistic."
        )


def checkpoint_owner_local_atomistic(
    plan: OwnerLocalAtomisticPlan, state: OwnerLocalAtomisticState, /
) -> OwnerLocalAtomisticCheckpoint:
    """Explicit host egress of the complete accepted continuation state.

    Only a coherent accepted state is published: its topology must be the
    successful epoch of this plan's streamed relation, discovered for the
    layout's current (uniform) owner epoch. The topology's owner-epoch
    witness is recorded, so restore never re-derives it from the layout.
    """
    if not isinstance(state, OwnerLocalAtomisticState):
        raise TypeError("state must be an OwnerLocalAtomisticState.")
    if state.plan_id != plan.plan_id:
        raise ValueError("The owner-local state belongs to another plan.")
    if state.topology.streaming_plan_id != plan.streaming.plan_id:
        raise ValueError("The topology was prepared for another streamed relation plan.")
    arrays = _checkpoint_arrays(state)
    _require_coherent_epochs(
        arrays["layout/owner_epochs"], arrays["topology/owner_epochs"]
    )
    if not bool(jax.device_get(state.topology.evidence.successful)):
        raise ValueError("The owner-local topology is not an accepted epoch.")
    implementation = str(jax.random.key_impl(state.rng_key))
    logical = state.layout.logical_count
    return OwnerLocalAtomisticCheckpoint(
        arrays=arrays,
        key_implementation=implementation,
        logical_count=logical,
        plan_id=state.plan_id,
        model_revision_id=state.model_revision_id,
        run_id=state.run_id,
        checkpoint_id=_checkpoint_identity(
            arrays,
            implementation,
            logical,
            state.plan_id,
            state.model_revision_id,
            state.run_id,
        ),
    )


def restore_owner_local_atomistic(
    plan: OwnerLocalAtomisticPlan,
    model: AtomisticLayeredModel,
    checkpoint: OwnerLocalAtomisticCheckpoint,
    /,
) -> OwnerLocalAtomisticState:
    """Reconstruct an accepted owner-local state for a matching plan and model.

    The plan identity, the model numeric revision, and the content identity
    must match. The layout, halo plan, and prepared relations are rebuilt
    through their constructors from the recorded routes, so a restored run
    reproduces the uninterrupted accepted evolution. The topology binds its
    recorded owner-epoch witness, never the restored layout's epochs; a
    record whose witness disagrees with the layout, or that lacks it, is
    refused.
    """
    streaming = _require_model(plan, model)
    if not isinstance(checkpoint, OwnerLocalAtomisticCheckpoint):
        raise TypeError("checkpoint must be an OwnerLocalAtomisticCheckpoint.")
    if checkpoint.plan_id != plan.plan_id:
        raise ValueError("The checkpoint belongs to another owner-local plan.")
    revision = _model_revision(model)
    if checkpoint.model_revision_id != revision:
        raise ValueError("The checkpoint belongs to another model revision.")
    arrays = checkpoint.arrays
    observed = _checkpoint_identity(
        arrays,
        checkpoint.key_implementation,
        checkpoint.logical_count,
        checkpoint.plan_id,
        checkpoint.model_revision_id,
        checkpoint.run_id,
    )
    if observed != checkpoint.checkpoint_id:
        raise ValueError("The owner-local checkpoint content identity is corrupt.")
    ownership = plan.ownership
    layout = DistributedPointLayout(
        ownership,
        arrays["layout/points"],
        arrays["layout/stable_ids"],
        arrays["layout/active"],
        arrays["layout/logical_indices"],
        arrays["layout/owner_epochs"],
        arrays["layout/stable_ids_unique"],
        checkpoint.logical_count,
    )
    routes = tuple(
        ownership.place(arrays[f"topology/{name}"])
        for name in ("receivers", "route_owners", "route_slots", "shifts", "valid")
    )
    if "topology/owner_epochs" not in arrays:
        raise ValueError("The checkpoint lacks the topology owner-epoch witness.")
    _require_coherent_epochs(
        arrays["layout/owner_epochs"], arrays["topology/owner_epochs"]
    )
    halo = DistributedHaloPlan(
        ownership, routes[1], routes[2], routes[4], halo_capacity=plan.halo_capacity
    )
    owners = ownership.owner_count
    statistics = jnp.broadcast_to(
        jnp.stack(
            tuple(
                jnp.asarray(arrays[f"topology/{name}"], dtype=jnp.int32)
                for name in ("maximum_alias_load", "maximum_edge_count", "maximum_degree")
            )
        ),
        (owners, 3),
    )
    topology = _bind_topology(
        plan,
        streaming,
        layout,
        ownership.place(arrays["topology/reference_positions"]),
        jnp.asarray(arrays["topology/epoch"]),
        ownership.place(arrays["topology/owner_epochs"]),
        (routes[0], routes[1], routes[2], routes[3], routes[4]),
        halo,
        statistics,
    )
    if not bool(topology.evidence.successful):
        raise ValueError("The restored owner-local topology does not validate.")
    payload = {
        name.removeprefix("payload/"): ownership.place(value)
        for name, value in arrays.items()
        if name.startswith("payload/")
    }
    key = jax.random.wrap_key_data(
        jnp.asarray(arrays["state/rng_key"]), impl=checkpoint.key_implementation
    )
    return OwnerLocalAtomisticState(
        layout=layout,
        positions=ownership.place(arrays["atoms/positions"]),
        velocities=ownership.place(arrays["atoms/velocities"]),
        masses=ownership.place(arrays["atoms/masses"]),
        species=ownership.place(arrays["atoms/species"]),
        image_counts=ownership.place(arrays["atoms/image_counts"]),
        force_cache=ownership.place(arrays["atoms/force_cache"]),
        force_cache_step=jnp.asarray(arrays["state/force_cache_step"]),
        atom_payload=payload,
        rng_key=key,
        thermostat_state=jnp.asarray(arrays["state/thermostat"]),
        bias_state=jnp.asarray(arrays["state/bias"]),
        constraint_state=jnp.asarray(arrays["state/constraint"]),
        step_index=jnp.asarray(arrays["state/step_index"]),
        topology=topology,
        plan_id=checkpoint.plan_id,
        model_revision_id=checkpoint.model_revision_id,
        run_id=checkpoint.run_id,
    )


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
    "OwnerLocalAtomisticCheckpoint",
    "OwnerLocalAtomisticEvaluation",
    "OwnerLocalAtomisticPlan",
    "OwnerLocalAtomisticState",
    "OwnerLocalAtomisticTopology",
    "OwnerLocalExecutionStatus",
    "OwnerLocalLossGradient",
    "OwnerLocalTopologyEvidence",
    "OwnerLocalTransition",
    "PreparedDistributedAtomisticRuntime",
    "PreparedDistributedPME",
    "PreparedDistributedPolarization",
    "certify_distributed_polarization",
    "certify_distributed_reciprocal",
    "checkpoint_distributed_atomistic",
    "checkpoint_owner_local_atomistic",
    "commit_distributed_migration",
    "distributed_constraint_projection",
    "distributed_domain_evidence",
    "distributed_particle_mesh_electrostatics",
    "distributed_thermodynamic_reduction",
    "evaluate_distributed_atomistic",
    "evaluate_owner_local_atomistic",
    "exchange_distributed_halos",
    "halo_short_range_evaluate",
    "migrate_distributed_atomistic",
    "owner_local_loss_gradient",
    "prepare_owner_local_atomistic",
    "propose_distributed_migration",
    "rebind_owner_local_model",
    "rebuild_owner_local_atomistic",
    "restore_distributed_atomistic_checkpoint",
    "restore_owner_local_atomistic",
    "reverse_distributed_halo_force_return",
    "reverse_halo_force_return",
]
