#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._periodic_cell import lattice_measure
from ..typing import Bool, checked, Dim, Float, Int32
from ._graph import (
    AtomisticGraphExecutionPlan,
    AtomisticGraphTopology,
    bind_atomistic_graph,
    prepare_atomistic_graph_topology,
)
from ._potential import (
    AbstractAtomisticPotential,
    atomistic_potential_revision,
    AtomisticSpeciesKind,
    AtomisticStressConvention,
)
from ._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticScaleContract,
    AtomisticStatus,
)


_AtomisticPotential = AbstractAtomisticPotential


class AtomisticCaseDim(Dim):
    pass


class AtomisticAtomDim(Dim):
    pass


@final
class AtomisticEnergyDerivatives(StrictModule, NonTrainableState):
    """Fixed-topology numerical energy derivatives, with no host identity work."""

    __strict_contract__ = True

    energy: Float[AtomisticCaseDim]
    atom_energy: Float[AtomisticCaseDim, AtomisticAtomDim]
    forces: Float[AtomisticCaseDim, AtomisticAtomDim, Literal[3]] | None
    stress: Float[AtomisticCaseDim, Literal[3], Literal[3]] | None
    stress_available: Bool[AtomisticCaseDim]
    overflow: Bool[AtomisticCaseDim]
    maximum_neighbor_count: Int32[AtomisticCaseDim]
    successful: Bool[AtomisticCaseDim]


class AtomisticProvenance(StrictModule, NonTrainableState):
    """Identity and mathematical guarantees of one energy/force evaluation."""

    architecture_id: str = eqx.field(static=True)
    potential_revision_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)
    atom_topology_id: str = eqx.field(static=True)
    graph_execution_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    conservative_forces: bool = eqx.field(static=True)
    frozen_candidate_topology: bool = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    stress_available: bool = eqx.field(static=True)
    graph_topology_id: str = eqx.field(static=True)
    stress_conventions: tuple[AtomisticStressConvention | None, ...] = eqx.field(
        static=True
    )
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        potential: _AtomisticPotential,
        batch: AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        topology: AtomisticGraphTopology,
        compute_stress: bool = False,
        stress_case_mask: ArrayLike | None = None,
    ) -> None:
        revision = atomistic_potential_revision(potential)
        self.architecture_id = potential.architecture_id
        self.potential_revision_id = revision.revision_id
        self.batch_id = batch.batch_id
        self.atom_topology_id = batch.atom_topology_id
        self.graph_execution_id = execution.plan_id
        self.scale_id = batch.scale.scale_id
        self.precision_id = potential.precision.policy_id
        self.method_id = potential.method_id
        self.conservative_forces = True
        self.frozen_candidate_topology = True
        self.periodic = batch.periodic_axes is not None and bool(
            np.any(np.asarray(batch.periodic_axes))
        )
        self.stress_available = compute_stress
        self.graph_topology_id = topology.topology_id
        requested = (
            np.full((batch.case_count,), compute_stress, dtype=np.bool_)
            if stress_case_mask is None
            else np.asarray(stress_case_mask, dtype=np.bool_)
        )
        periodic = (
            np.zeros((batch.case_count, 3), dtype=np.bool_)
            if batch.periodic_axes is None
            else np.asarray(batch.periodic_axes, dtype=np.bool_)
        )
        self.stress_conventions = tuple(
            None
            if not selected or not np.any(axes)
            else AtomisticStressConvention.CAUCHY_TENSION_POSITIVE
            if np.all(axes)
            else AtomisticStressConvention.CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME
            for selected, axes in zip(requested, periodic, strict=True)
        )
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "atomistic-prediction-provenance",
                "architecture": potential.architecture_id,
                "potential_revision": revision.revision_id,
                "batch": batch.batch_id,
                "atom_topology": batch.atom_topology_id,
                "graph_execution": execution.plan_id,
                "scale": batch.scale.scale_id,
                "precision": potential.precision.policy_id,
                "method": self.method_id,
                "graph_topology": self.graph_topology_id,
                "periodic": self.periodic,
                "stress_requested": compute_stress,
                "stress_conventions": self.stress_conventions,
            }
        )


class AtomisticPrediction(StrictModule, NonTrainableState):
    """Case-isolated energies, conservative forces, requested stress and evidence."""

    __strict_contract__ = True

    energy: Float[AtomisticCaseDim]
    forces: Float[AtomisticCaseDim, AtomisticAtomDim, Literal[3]]
    atom_energy: Float[AtomisticCaseDim, AtomisticAtomDim]
    stress: Float[AtomisticCaseDim, Literal[3], Literal[3]] | None
    stress_available: Bool[AtomisticCaseDim]
    valid: Bool[AtomisticCaseDim]
    status: Int32[AtomisticCaseDim]
    neighbor_overflow: Bool[AtomisticCaseDim]
    maximum_neighbor_count: Int32[AtomisticCaseDim]
    net_force: Float[AtomisticCaseDim, Literal[3]]
    net_torque: Float[AtomisticCaseDim, Literal[3]]
    scale: AtomisticScaleContract
    provenance: AtomisticProvenance
    energy_axes: tuple[str] = eqx.field(static=True)
    force_axes: tuple[str, str, str] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


def _graph_energy_components(
    potential: _AtomisticPotential,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    topology: AtomisticGraphTopology,
    positions: Array,
    cell_vectors: Array | None,
    /,
) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
    graph = bind_atomistic_graph(
        topology,
        execution,
        positions,
        cutoff=potential.configuration.cutoff,
        cell_vectors=cell_vectors,
        periodic_axes=(
            jnp.zeros((batch.case_count, 3), dtype=jnp.bool_)
            if batch.periodic_axes is None
            else batch.periodic_axes
        ),
        atomic_numbers=batch.atomic_numbers.reshape((-1,)),
        atom_type_ids=batch.atom_type_ids.reshape((-1,)),
        masses=batch.masses.reshape((-1,)),
    )
    match potential.capabilities.species_kind:
        case AtomisticSpeciesKind.ATOMIC_NUMBER:
            species = batch.atomic_numbers
        case AtomisticSpeciesKind.ATOM_TYPE_ID:
            species = batch.atom_type_ids
        case _:
            raise ValueError("The potential declares an invalid species identity.")
    energy, atom_energy = potential.graph_energy(
        species.reshape((-1,)),
        batch.atom_mask.reshape((-1,)),
        batch.atom_cases,
        batch.case_count,
        batch.atom_capacity,
        graph,
    )
    return jnp.sum(energy.astype(potential.precision.reduction_dtype)), (
        energy,
        atom_energy,
        graph.overflow,
        graph.maximum_neighbor_count,
        graph.valid,
    )


def _stress_geometry(
    positions: Array, cell_vectors: Array, strain: Array, /
) -> tuple[Array, Array]:
    deformation = (
        jnp.broadcast_to(jnp.eye(3, dtype=positions.dtype), cell_vectors.shape) + strain
    )
    transpose = deformation.transpose((0, 2, 1))
    return positions @ transpose, cell_vectors @ transpose


@checked
def atomistic_energy_derivatives(
    potential: _AtomisticPotential,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    positions: Array,
    /,
    *,
    topology: AtomisticGraphTopology,
    cell_vectors: ArrayLike | None = None,
    compute_forces: bool = True,
    compute_stress: bool = False,
    stress_case_mask: ArrayLike | None = None,
) -> AtomisticEnergyDerivatives:
    """Differentiate one scalar on a bound graph; never discover or hash topology.

    This numerical boundary admits ``jit``, JVP/VJP and mixed force/stress-loss
    derivatives. Stress is tensile-positive ``dE/dstrain / volume`` under
    ``H' = H @ (I + strain).T``. Discrete candidates/images stay fixed. The
    batch must satisfy the potential's scale and coordinate precision contract
    (the same static check as ``energy_and_forces``); separately supplied
    ``positions`` and ``cell_vectors`` are then canonicalized once into that
    coordinate dtype.
    """
    potential._validate_batch(batch)
    if topology.atom_topology_id != batch.atom_topology_id:
        raise ValueError("Graph topology belongs to another material atom batch.")
    if (
        topology.case_count != batch.case_count
        or topology.atom_capacity != batch.atom_capacity
    ):
        raise ValueError("Graph topology and batch capacities differ.")
    coordinate = jnp.asarray(positions, dtype=potential.precision.coordinate_dtype)
    if coordinate.shape != batch.positions.shape:
        raise ValueError("positions must have the batch shape (case, atom, 3).")
    coordinate = jnp.where(batch.atom_mask[:, :, None], coordinate, 0.0)
    supplied_vectors = batch.cells if cell_vectors is None else cell_vectors
    vectors = (
        None
        if supplied_vectors is None
        else jnp.asarray(supplied_vectors, dtype=coordinate.dtype)
    )
    if vectors is not None and vectors.shape != (batch.case_count, 3, 3):
        raise ValueError("cell_vectors must have shape (case, 3, 3).")
    forces = None
    stress = None
    requested_stress = (
        jnp.full((batch.case_count,), compute_stress, dtype=jnp.bool_)
        if stress_case_mask is None
        else jnp.asarray(stress_case_mask, dtype=jnp.bool_)
    )
    if requested_stress.shape != (batch.case_count,):
        raise ValueError("stress_case_mask must have shape (case,).")
    if stress_case_mask is not None and not compute_stress:
        raise ValueError("stress_case_mask requires compute_stress=True.")
    volume_valid = jnp.ones((batch.case_count,), dtype=jnp.bool_)
    if compute_stress:
        if vectors is None or batch.periodic_axes is None:
            raise ValueError(
                "Requested bulk stress requires explicit periodic cell data."
            )
        if not potential.capabilities.cell_derivative:
            raise ValueError("The potential does not admit cell derivatives.")
        # An unrequested finite case may have an absent/singular cell. It must
        # never feed a singular-volume derivative before its stress is masked.
        measure_vectors = jnp.where(
            requested_stress[:, None, None],
            vectors,
            jnp.broadcast_to(jnp.eye(3, dtype=vectors.dtype), vectors.shape),
        )
        volumes, measure_valid = lattice_measure(measure_vectors)
        volume_valid = ~requested_stress | (
            measure_valid & jnp.any(batch.periodic_axes, axis=1)
        )
        volume = jnp.where(volume_valid, volumes, 1.0)
        reference_vectors = vectors

        def strained(
            value: Array, strain: Array
        ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
            deformed_position, deformed_cell = _stress_geometry(
                value, reference_vectors, strain
            )
            return _graph_energy_components(
                potential, batch, execution, topology, deformed_position, deformed_cell
            )

        zero_strain = jnp.zeros(reference_vectors.shape, dtype=coordinate.dtype)
        if compute_forces:
            (_, auxiliary), (position_gradient, strain_gradient) = jax.value_and_grad(
                strained, argnums=(0, 1), has_aux=True
            )(coordinate, zero_strain)
            forces = -position_gradient
        else:
            (_, auxiliary), strain_gradient = jax.value_and_grad(
                strained, argnums=1, has_aux=True
            )(coordinate, zero_strain)
        stress = (
            0.5 * (strain_gradient + strain_gradient.transpose((0, 2, 1)))
        ) / volume[:, None, None]
    else:

        def scalar(
            value: Array,
        ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
            return _graph_energy_components(
                potential, batch, execution, topology, value, vectors
            )

        if compute_forces:
            (_, auxiliary), gradient = jax.value_and_grad(scalar, has_aux=True)(
                coordinate
            )
            forces = -gradient
        else:
            _, auxiliary = scalar(coordinate)
    energy, atom_energy, overflow, maximum_neighbor_count, graph_valid = auxiliary
    finite = (
        jnp.isfinite(energy)
        & volume_valid
        & jnp.all(jnp.isfinite(coordinate), axis=(1, 2))
    )
    if forces is not None:
        forces = jnp.where(batch.atom_mask[:, :, None], forces, 0.0)
        finite = finite & jnp.all(jnp.isfinite(forces), axis=(1, 2))
    if stress is not None:
        finite = finite & (~requested_stress | jnp.all(jnp.isfinite(stress), axis=(1, 2)))
    finite = finite & jnp.all(
        jnp.isfinite(jnp.where(batch.atom_mask, atom_energy, 0.0)), axis=1
    )
    successful = graph_valid & finite
    nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
    # A failed numerical state has no valid tangent. A NaN scale poisons both
    # values and derivatives; a where(value, NaN) would return zero cotangents.
    case_scale = jnp.where(successful, jnp.ones_like(energy), nan)
    return AtomisticEnergyDerivatives(
        energy=energy * case_scale,
        atom_energy=(
            jnp.where(batch.atom_mask, atom_energy, 0.0)
            * case_scale.astype(atom_energy.dtype)[:, None]
        ),
        forces=(
            None
            if forces is None
            else (forces * case_scale.astype(forces.dtype)[:, None, None]).astype(
                potential.precision.output_dtype
            )
        ),
        stress=(
            None
            if stress is None
            else jnp.where(
                requested_stress[:, None, None],
                stress * case_scale.astype(stress.dtype)[:, None, None],
                nan.astype(stress.dtype),
            ).astype(potential.precision.output_dtype)
        ),
        stress_available=successful & requested_stress,
        overflow=overflow,
        maximum_neighbor_count=maximum_neighbor_count,
        successful=successful,
    )


def energy_and_forces(
    potential: _AtomisticPotential,
    structure: AtomicStructure | AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    /,
    *,
    compute_stress: bool = False,
    topology: AtomisticGraphTopology | None = None,
    stress_case_mask: ArrayLike | None = None,
) -> AtomisticPrediction:
    """Evaluate energy once and derive forces as its negative position gradient.

    Provenance names the canonical numeric revision of the potential's current
    parameters (``phydrax.atomistic.atomistic_potential_revision``), so any
    parameter update is reflected without a separate checkpoint step. The
    revision is a host boundary: call this function with concrete parameters,
    not on a potential traced by ``jit``, ``vmap``, or ``grad``.
    """

    if not isinstance(potential, AbstractAtomisticPotential):
        raise TypeError("potential must implement AbstractAtomisticPotential.")
    if not isinstance(execution, AtomisticGraphExecutionPlan):
        raise TypeError("execution must be an AtomisticGraphExecutionPlan.")
    if isinstance(structure, AtomicStructure):
        batch = AtomisticBatch.from_structure(structure)
    elif isinstance(structure, AtomisticBatch):
        batch = structure
    else:
        raise TypeError("structure must be AtomicStructure or AtomisticBatch.")
    potential._validate_batch(batch)

    topology = (
        prepare_atomistic_graph_topology(
            batch, execution, cutoff=potential.configuration.cutoff
        )
        if topology is None
        else topology
    )
    derivative = atomistic_energy_derivatives(
        potential,
        batch,
        execution,
        batch.positions,
        topology=topology,
        compute_stress=compute_stress,
        stress_case_mask=stress_case_mask,
    )
    energy = derivative.energy
    atom_energy = derivative.atom_energy
    overflow = derivative.overflow
    maximum_neighbor_count = derivative.maximum_neighbor_count
    if derivative.forces is None:
        raise RuntimeError("The requested force derivative was not returned.")
    forces = derivative.forces
    mask = batch.atom_mask
    forces = jnp.where(mask[:, :, None], forces, 0.0)
    finite_energy = jnp.isfinite(energy)
    finite_forces = jnp.all(
        jnp.isfinite(jnp.where(mask[:, :, None], forces, 0.0)), axis=(1, 2)
    )
    valid = derivative.successful & finite_energy & finite_forces
    status = jnp.where(
        overflow,
        int(AtomisticStatus.NEIGHBOR_OVERFLOW),
        jnp.where(
            finite_energy & finite_forces,
            int(AtomisticStatus.SUCCESS),
            int(AtomisticStatus.NONFINITE),
        ),
    ).astype(jnp.int32)
    nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
    energy = jnp.where(valid, energy, nan)
    atom_energy = jnp.where(valid[:, None], jnp.where(mask, atom_energy, 0.0), nan)
    forces = jnp.where(
        valid[:, None, None],
        jnp.where(mask[:, :, None], forces, 0.0),
        nan,
    )
    diagnostic_forces = jnp.where(mask[:, :, None], forces, 0.0)
    net_force = jnp.sum(diagnostic_forces, axis=1)
    torque_dtype = jnp.dtype(potential.precision.reduction_dtype)
    mass = jnp.where(mask, batch.masses, 0.0).astype(torque_dtype)
    torque_positions = jnp.where(mask[:, :, None], batch.positions, 0.0).astype(
        torque_dtype
    )
    center = (
        contract("ba,bad->bd", mass, torque_positions) / jnp.sum(mass, axis=1)[:, None]
    )
    lever = torque_positions - center[:, None, :]
    torque_force = diagnostic_forces.astype(torque_dtype)
    net_torque = jnp.sum(jnp.cross(lever, torque_force), axis=1).astype(
        potential.precision.output_dtype
    )
    return AtomisticPrediction(
        energy=energy,
        forces=forces,
        atom_energy=atom_energy,
        stress=derivative.stress,
        stress_available=derivative.stress_available,
        valid=valid,
        status=status,
        neighbor_overflow=overflow,
        maximum_neighbor_count=maximum_neighbor_count,
        net_force=net_force,
        net_torque=net_torque,
        scale=batch.scale,
        provenance=AtomisticProvenance(
            potential,
            batch,
            execution,
            topology=topology,
            compute_stress=compute_stress,
            stress_case_mask=stress_case_mask,
        ),
        energy_axes=("case",),
        force_axes=("case", "atom", "cartesian"),
    )


__all__ = [
    "AtomisticPrediction",
    "AtomisticProvenance",
    "AtomisticEnergyDerivatives",
    "atomistic_energy_derivatives",
    "energy_and_forces",
]
