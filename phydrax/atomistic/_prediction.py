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

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._periodic_cell import lattice_measure
from ..typing import Bool, checked, Float, Int32, PRNGKey
from ._conditioning import AtomisticConditioning
from ._dimensions import AtomisticAtomDim, AtomisticCaseDim
from ._graph import (
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticGraphPayload,
)
from ._model_inputs import AtomisticGraphInputs
from ._neighborhood_bundle import AtomisticNeighborhoodRole
from ._potential import (
    AbstractAtomisticModel,
    AbstractPreparedAtomisticModel,
    atomistic_model_revision,
    AtomisticStressConvention,
)
from ._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticScaleContract,
    AtomisticStatus,
)


type _AtomisticModel = AbstractAtomisticModel | AbstractPreparedAtomisticModel


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
    """Identity and mathematical guarantees of one energy/force evaluation.

    Scientific source identity (``batch_id``, ``atom_topology_id``, the host
    graph content ``graph_topology_id``) is kept separate from the executable
    identity of the compiled evaluation (``graph_layout_id``).
    """

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
    graph_layout_id: str = eqx.field(static=True)
    conditioning_id: str | None = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)
    random_realization_id: str | None = eqx.field(static=True)
    image_counts_id: str | None = eqx.field(static=True)
    stress_conventions: tuple[AtomisticStressConvention | None, ...] = eqx.field(
        static=True
    )
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        potential: _AtomisticModel,
        batch: AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        payload: AtomisticGraphPayload,
        compute_stress: bool = False,
        stress_case_mask: ArrayLike | None = None,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
        image_counts: ArrayLike | None = None,
    ) -> None:
        if payload.execution.plan_id != execution.plan_id:
            raise ValueError(
                "The graph payload was prepared under another execution plan."
            )
        revision = atomistic_model_revision(potential)
        definition = (
            potential.model
            if isinstance(potential, AbstractPreparedAtomisticModel)
            else potential
        )
        realization_id = (
            potential.prepared_id
            if isinstance(potential, AbstractPreparedAtomisticModel)
            else None
        )
        random_id = (
            None
            if key is None
            else canonical_fingerprint(
                {
                    "kind": "atomistic-random-realization",
                    "implementation": str(jax.random.key_impl(key)),
                    "key": array_tree_fingerprint(jax.random.key_data(key)),
                }
            )
        )
        image_counts_id = (
            None
            if image_counts is None
            else canonical_fingerprint(array_tree_fingerprint(jnp.asarray(image_counts)))
        )
        condition_id = (
            None
            if conditioning is None
            else canonical_fingerprint(
                {
                    "kind": "atomistic-bound-conditioning",
                    "spec": conditioning.spec.spec_id,
                    "case_count": conditioning.case_count,
                    "atom_capacity": conditioning.atom_capacity,
                    "values": array_tree_fingerprint(conditioning),
                }
            )
        )
        self.architecture_id = definition.architecture_id
        self.potential_revision_id = revision.revision_id
        self.batch_id = batch.batch_id
        self.atom_topology_id = batch.atom_topology_id
        self.graph_execution_id = execution.plan_id
        self.scale_id = batch.scale.scale_id
        self.precision_id = potential.precision.policy_id
        self.method_id = definition.method_id
        self.conservative_forces = True
        self.frozen_candidate_topology = True
        self.periodic = batch.periodic_axes is not None and bool(
            np.any(np.asarray(batch.periodic_axes))
        )
        self.stress_available = compute_stress
        self.graph_topology_id = payload.topology.content_id()
        self.graph_layout_id = payload.layout_id
        self.conditioning_id = condition_id
        self.realization_id = realization_id
        self.random_realization_id = random_id
        self.image_counts_id = image_counts_id
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
                "architecture": definition.architecture_id,
                "potential_revision": revision.revision_id,
                "batch": batch.batch_id,
                "atom_topology": batch.atom_topology_id,
                "graph_execution": execution.plan_id,
                "scale": batch.scale.scale_id,
                "precision": potential.precision.policy_id,
                "method": self.method_id,
                "graph_topology": self.graph_topology_id,
                "graph_layout": self.graph_layout_id,
                "conditioning": condition_id,
                "realization": realization_id,
                "random_realization": random_id,
                "image_counts": image_counts_id,
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


def _require_payload_model(
    potential: _AtomisticModel, payload: AtomisticGraphPayload, /
) -> None:
    """Static admission of a payload layout by a model's scale and precision."""
    if not isinstance(payload, AtomisticGraphPayload):
        raise TypeError("payload must be an AtomisticGraphPayload.")
    if payload.scale.scale_id != potential.scale.scale_id:
        raise ValueError("Model and graph payload must share one exact scale contract.")
    if jnp.dtype(payload.layout.coordinate_dtype) != jnp.dtype(
        potential.precision.coordinate_dtype
    ):
        raise ValueError(
            "The graph payload coordinate dtype does not match the model precision "
            "contract."
        )


def _graph_energy_components(
    potential: _AtomisticModel,
    payload: AtomisticGraphPayload,
    positions: Array,
    cell_vectors: Array | None,
    conditioning: AtomisticConditioning | None,
    key: PRNGKey | None,
    image_counts: Array | None,
    /,
) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
    graph = payload.graph(
        cutoff=potential.configuration.cutoff,
        positions=positions,
        cell_vectors=cell_vectors,
        selection=potential.requirements.neighborhood(
            AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
        ).selection,
        image_counts=image_counts,
    )
    outputs = potential.evaluate_graph(
        AtomisticGraphInputs(
            graph,
            payload.scale,
            species_kind=potential.capabilities.species_kind,
            conditioning=conditioning,
            key=key,
        )
    )
    return jnp.sum(outputs.energy.astype(potential.precision.reduction_dtype)), (
        outputs.energy,
        outputs.atom_energy,
        graph.overflow,
        graph.maximum_neighbor_count,
        outputs.successful,
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
    potential: _AtomisticModel,
    payload: AtomisticGraphPayload,
    positions: Array,
    /,
    *,
    cell_vectors: ArrayLike | None = None,
    compute_forces: bool = True,
    compute_stress: bool = False,
    stress_case_mask: ArrayLike | None = None,
    conditioning: AtomisticConditioning | None = None,
    key: PRNGKey | None = None,
    image_counts: ArrayLike | None = None,
) -> AtomisticEnergyDerivatives:
    """Differentiate one scalar on a layout-bound graph; the canonical path.

    This numerical boundary admits ``jit``, JVP/VJP and mixed force/stress-loss
    derivatives; it never discovers routes or hashes content, and every payload
    of one ``AtomisticGraphLayout`` shares its static structure. Stress is
    tensile-positive ``dE/dstrain / volume`` under ``H' = H @ (I + strain).T``.
    Discrete candidates/images stay fixed. The payload layout must satisfy the
    model's scale and coordinate precision; separately supplied ``positions``
    and ``cell_vectors`` (default: the payload cells) are canonicalized once
    into that coordinate dtype.
    """
    _require_payload_model(potential, payload)
    if not potential.capabilities.conservative_energy:
        raise ValueError(
            "This atomistic model does not define a conservative scalar potential."
        )
    counts = None if image_counts is None else jnp.asarray(image_counts)
    cases = payload.case_count
    atom_mask = payload.atom_mask
    coordinate = jnp.asarray(positions, dtype=potential.precision.coordinate_dtype)
    if coordinate.shape != (cases, payload.atom_capacity, 3):
        raise ValueError("positions must have the payload shape (case, atom, 3).")
    coordinate = jnp.where(
        atom_mask[:, :, None], coordinate, jnp.zeros((), dtype=coordinate.dtype)
    )
    supplied_vectors = payload.cells if cell_vectors is None else cell_vectors
    vectors = (
        None
        if supplied_vectors is None
        else jnp.asarray(supplied_vectors, dtype=coordinate.dtype)
    )
    if vectors is not None and vectors.shape != (cases, 3, 3):
        raise ValueError("cell_vectors must have shape (case, 3, 3).")
    forces = None
    stress = None
    requested_stress = (
        jnp.full((cases,), compute_stress, dtype=jnp.bool_)
        if stress_case_mask is None
        else jnp.asarray(stress_case_mask, dtype=jnp.bool_)
    )
    if requested_stress.shape != (cases,):
        raise ValueError("stress_case_mask must have shape (case,).")
    if stress_case_mask is not None and not compute_stress:
        raise ValueError("stress_case_mask requires compute_stress=True.")
    volume_valid = jnp.ones((cases,), dtype=jnp.bool_)
    if compute_stress:
        periodic_axes = payload.periodic_axes
        if vectors is None or periodic_axes is None:
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
            measure_valid & jnp.any(periodic_axes, axis=1)
        )
        volume = jnp.where(volume_valid, volumes, jnp.ones((), dtype=volumes.dtype))
        reference_vectors = vectors

        def strained(
            value: Array, strain: Array
        ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
            deformed_position, deformed_cell = _stress_geometry(
                value, reference_vectors, strain
            )
            return _graph_energy_components(
                potential,
                payload,
                deformed_position,
                deformed_cell,
                conditioning,
                key,
                counts,
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
            jnp.asarray(0.5, dtype=strain_gradient.dtype)
            * (strain_gradient + strain_gradient.transpose((0, 2, 1)))
        ) / volume[:, None, None]
    else:

        def scalar(
            value: Array,
        ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
            return _graph_energy_components(
                potential,
                payload,
                value,
                vectors,
                conditioning,
                key,
                counts,
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
        forces = jnp.where(
            atom_mask[:, :, None], forces, jnp.zeros((), dtype=forces.dtype)
        )
        finite = finite & jnp.all(jnp.isfinite(forces), axis=(1, 2))
    if stress is not None:
        finite = finite & (~requested_stress | jnp.all(jnp.isfinite(stress), axis=(1, 2)))
    finite = finite & jnp.all(
        jnp.isfinite(
            jnp.where(atom_mask, atom_energy, jnp.zeros((), dtype=atom_energy.dtype)),
        ),
        axis=1,
    )
    successful = graph_valid & finite
    nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
    # A failed numerical state has no valid tangent. A NaN scale poisons both
    # values and derivatives; a where(value, NaN) would return zero cotangents.
    case_scale = jnp.where(successful, jnp.ones_like(energy), nan)
    return AtomisticEnergyDerivatives(
        energy=energy * case_scale,
        atom_energy=(
            jnp.where(atom_mask, atom_energy, jnp.zeros((), dtype=atom_energy.dtype))
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
    potential: _AtomisticModel,
    structure: AtomicStructure | AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    /,
    *,
    compute_stress: bool = False,
    payload: AtomisticGraphPayload | None = None,
    stress_case_mask: ArrayLike | None = None,
    conditioning: AtomisticConditioning | None = None,
    key: PRNGKey | None = None,
    image_counts: ArrayLike | None = None,
) -> AtomisticPrediction:
    """Evaluate energy once and derive forces as its negative position gradient.

    Host boundary over the canonical ``atomistic_energy_derivatives`` path.
    Without ``payload`` the one-batch ``AtomisticGraphLayout`` of the batch is
    prepared; a supplied payload is bound to the batch by
    ``AtomisticGraphPayload.admit``. Provenance names the canonical numeric
    revision of the potential's current parameters
    (``phydrax.atomistic.atomistic_model_revision``), the host batch and graph
    content identities and the executable layout. Call this function with
    concrete parameters, not on a potential traced by ``jit``, ``vmap``, or
    ``grad``.
    """

    if not isinstance(
        potential, (AbstractAtomisticModel, AbstractPreparedAtomisticModel)
    ):
        raise TypeError("potential must implement a native atomistic model contract.")
    if not potential.capabilities.conservative_energy:
        raise ValueError(
            "This atomistic model does not define a conservative scalar potential."
        )
    if not isinstance(execution, AtomisticGraphExecutionPlan):
        raise TypeError("execution must be an AtomisticGraphExecutionPlan.")
    if isinstance(structure, AtomicStructure):
        batch = AtomisticBatch.from_structure(structure)
    elif isinstance(structure, AtomisticBatch):
        batch = structure
    else:
        raise TypeError("structure must be AtomicStructure or AtomisticBatch.")
    potential._validate_batch(batch)
    bound = (
        AtomisticGraphLayout.for_batch(
            batch, execution, cutoff=potential.configuration.cutoff
        ).prepare(batch)
        if payload is None
        else payload.admit(batch)
    )
    if bound.execution.plan_id != execution.plan_id:
        raise ValueError("The graph payload was prepared under another execution plan.")
    derivative = atomistic_energy_derivatives(
        potential,
        bound,
        batch.positions,
        compute_stress=compute_stress,
        stress_case_mask=stress_case_mask,
        conditioning=conditioning,
        key=key,
        image_counts=image_counts,
    )
    energy = derivative.energy
    atom_energy = derivative.atom_energy
    overflow = derivative.overflow
    maximum_neighbor_count = derivative.maximum_neighbor_count
    if derivative.forces is None:
        raise RuntimeError("The requested force derivative was not returned.")
    forces = derivative.forces
    mask = batch.atom_mask
    forces = jnp.where(mask[:, :, None], forces, jnp.zeros((), dtype=forces.dtype))
    finite_energy = jnp.isfinite(energy)
    finite_forces = jnp.all(
        jnp.isfinite(
            jnp.where(mask[:, :, None], forces, jnp.zeros((), dtype=forces.dtype))
        ),
        axis=(1, 2),
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
    atom_energy = jnp.where(
        valid[:, None],
        jnp.where(mask, atom_energy, jnp.zeros((), dtype=atom_energy.dtype)),
        nan,
    )
    forces = jnp.where(
        valid[:, None, None],
        jnp.where(mask[:, :, None], forces, jnp.zeros((), dtype=forces.dtype)),
        nan,
    )
    diagnostic_forces = jnp.where(
        mask[:, :, None], forces, jnp.zeros((), dtype=forces.dtype)
    )
    net_force = jnp.sum(diagnostic_forces, axis=1)
    torque_dtype = jnp.dtype(potential.precision.reduction_dtype)
    mass = jnp.where(mask, batch.masses, jnp.zeros((), dtype=batch.masses.dtype)).astype(
        torque_dtype
    )
    torque_positions = jnp.where(
        mask[:, :, None], batch.positions, jnp.zeros((), dtype=batch.positions.dtype)
    ).astype(torque_dtype)
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
            payload=bound,
            compute_stress=compute_stress,
            stress_case_mask=stress_case_mask,
            conditioning=conditioning,
            key=key,
            image_counts=image_counts,
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
