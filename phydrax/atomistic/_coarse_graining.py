#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Disjoint center-of-mass coarse maps and projected-force matching.

`CoarseForceMatchingProblem` maps fine configurations and forces onto beads,
lowers the projected (optionally prior-corrected) forces into an immutable
canonical source of atom-type-ID beads with coarse masses, and binds a
force-only `AtomisticTrainingProblem`; `fit_coarse_potential` delegates to
`fit_atomistic_potential`.
"""

from __future__ import annotations

from typing import cast, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked, PRNGKey
from ._graph import AtomisticGraphExecutionPlan
from ._potential import AbstractAtomisticPotential
from ._structure_store import AtomisticStructureSource
from ._system import AtomisticSystemPlan, PreparedAtomisticSystem
from ._topology import MolecularTopologyPlan
from ._training import (
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticTrainingFields,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    AtomisticTrainingResult,
    fit_atomistic_potential,
)
from ._training_objectives import AtomisticForceObjective, AtomisticTrainingPrecision
from ._types import AtomisticBatch, AtomisticSpeciesKind


class MolecularCoarseMapPlan(StrictModule, NonTrainableState):
    bead_particle_ids: Array
    bead_type_ids: Array
    particle_to_bead: Array
    topology: MolecularTopologyPlan
    name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bead_particle_ids: ArrayLike,
        bead_type_ids: ArrayLike,
        particle_to_bead: ArrayLike,
        /,
        *,
        topology: MolecularTopologyPlan | None = None,
        name: str = "molecular-coarse-map",
    ) -> None:
        bead_ids = np.asarray(bead_particle_ids)
        bead_types = np.asarray(bead_type_ids)
        membership = np.asarray(particle_to_bead)
        if (
            bead_ids.ndim != 1
            or bead_ids.size == 0
            or not np.issubdtype(bead_ids.dtype, np.integer)
            or np.unique(bead_ids).size != bead_ids.size
        ):
            raise ValueError(
                "bead_particle_ids must be a non-empty unique integer vector."
            )
        if bead_types.shape != bead_ids.shape or not np.issubdtype(
            bead_types.dtype, np.integer
        ):
            raise TypeError("bead_type_ids must be an integer vector matching beads.")
        if np.any(bead_types < 0):
            raise ValueError("bead_type_ids must be nonnegative.")
        if membership.ndim != 1 or not np.issubdtype(membership.dtype, np.integer):
            raise TypeError("particle_to_bead must be an integer vector.")
        topology_ = MolecularTopologyPlan.empty() if topology is None else topology
        if not isinstance(topology_, MolecularTopologyPlan):
            raise TypeError("topology must be MolecularTopologyPlan or None.")
        identifier = str(name).strip()
        if not identifier:
            raise ValueError("name must be non-empty.")
        self.bead_particle_ids = jnp.asarray(bead_ids, dtype=jnp.int64)
        self.bead_type_ids = jnp.asarray(bead_types, dtype=jnp.int32)
        self.particle_to_bead = jnp.asarray(membership, dtype=jnp.int32)
        self.topology = topology_
        self.name = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-coarse-map-plan",
                "name": identifier,
                "topology": topology_.plan_id,
                "arrays": array_tree_fingerprint(
                    {
                        "bead_particle_ids": bead_ids,
                        "bead_type_ids": bead_types,
                        "particle_to_bead": membership,
                    }
                ),
                "mapping": "disjoint-center-of-mass",
            }
        )

    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedMolecularCoarseMap":
        return PreparedMolecularCoarseMap(self, system)


class MolecularCoarseMapEvaluation(StrictModule):
    positions: Array
    forces: Array | None
    momenta: Array | None
    image_counts: Array | None
    minimum_image_margin: Array
    mass_residual: Array
    charge_residual: Array
    force_residual: Array
    momentum_residual: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    kinetic_fidelity_claimed: bool = eqx.field(static=True, default=False)


class PreparedMolecularCoarseMap(StrictModule, NonTrainableState):
    plan: MolecularCoarseMapPlan
    fine_system: PreparedAtomisticSystem
    coarse_system: PreparedAtomisticSystem
    membership: Array
    center_weights: Array
    member_mask: Array
    anchor_indices: Array
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: MolecularCoarseMapPlan, system: PreparedAtomisticSystem, /
    ) -> None:
        membership = np.asarray(plan.particle_to_bead)
        if membership.shape != (system.capacity,):
            raise ValueError("particle_to_bead must match fine-system capacity.")
        active = np.asarray(system.active_mask, dtype=np.bool_)
        if np.any(membership[active] < 0) or np.any(
            membership[active] >= plan.bead_particle_ids.size
        ):
            raise ValueError(
                "Every active fine particle requires one valid bead assignment."
            )
        if np.any(membership[~active] != -1):
            raise ValueError("Inactive fine padding must use bead assignment -1.")
        bead_count = plan.bead_particle_ids.size
        member_mask = np.stack(
            tuple(active & (membership == index) for index in range(bead_count))
        )
        if np.any(np.sum(member_mask, axis=1) == 0):
            raise ValueError(
                "Every coarse bead requires at least one active fine member."
            )
        molecule_ids = np.asarray(system.plan.molecule_ids)
        region_ids = np.asarray(system.plan.region_ids)
        bead_molecules = []
        bead_regions = []
        anchors = []
        for mask in member_mask:
            indices = np.flatnonzero(mask)
            molecules = np.unique(molecule_ids[indices])
            regions = np.unique(region_ids[indices])
            if molecules.size != 1 or regions.size != 1:
                raise ValueError(
                    "A coarse bead cannot span molecule or region identities."
                )
            bead_molecules.append(int(molecules[0]))
            bead_regions.append(int(regions[0]))
            anchors.append(int(indices[0]))
        masses = np.asarray(system.plan.masses)
        charges = np.asarray(system.plan.charges)
        bead_masses = member_mask @ masses
        bead_charges = member_mask @ charges
        weights = member_mask * masses[None, :] / bead_masses[:, None]
        coarse_plan = AtomisticSystemPlan(
            plan.bead_particle_ids,
            np.zeros((bead_count,), dtype=np.int32),
            bead_masses,
            system.plan.units,
            atom_type_ids=plan.bead_type_ids,
            element_mask=np.zeros((bead_count,), dtype=np.bool_),
            charges=bead_charges,
            molecule_ids=np.asarray(bead_molecules, dtype=np.int32),
            region_ids=np.asarray(bead_regions, dtype=np.int32),
            topology=plan.topology,
            cell=system.cell,
            name=plan.name,
            coordinate_dtype=system.plan.coordinate_dtype,
        )
        self.plan = plan
        self.fine_system = system
        self.coarse_system = coarse_plan.prepare(numeric_version=system.numeric_version)
        self.membership = jnp.asarray(membership, dtype=jnp.int32)
        self.center_weights = jnp.asarray(weights, dtype=system.plan.masses.dtype)
        self.member_mask = jnp.asarray(member_mask, dtype=jnp.bool_)
        self.anchor_indices = jnp.asarray(anchors, dtype=jnp.int32)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-molecular-coarse-map",
                "plan": plan.plan_id,
                "fine_system": system.prepared_id,
                "coarse_system": self.coarse_system.prepared_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        /,
        *,
        forces: ArrayLike | None = None,
        momenta: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
    ) -> MolecularCoarseMapEvaluation:
        position = jnp.asarray(positions, dtype=self.fine_system.plan.coordinate_dtype)
        expected = (self.fine_system.capacity, 3)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        force = None if forces is None else jnp.asarray(forces, dtype=position.dtype)
        momentum = None if momenta is None else jnp.asarray(momenta, dtype=position.dtype)
        if force is not None and force.shape != expected:
            raise ValueError("forces must match fine positions.")
        if momentum is not None and momentum.shape != expected:
            raise ValueError("momenta must match fine positions.")
        cell = self.fine_system.cell
        images = (
            None if image_counts is None else jnp.asarray(image_counts, dtype=jnp.int32)
        )
        if images is not None and images.shape != expected:
            raise ValueError("image_counts must match fine positions.")
        margin = jnp.asarray(jnp.inf, dtype=position.dtype)
        unwrapped = position
        if cell is not None and images is not None:
            unwrapped = position + contract(
                "ni,ij->nj",
                images.astype(position.dtype),
                cell.vectors.astype(position.dtype),
            )
        elif cell is not None:
            active = self.fine_system.active_mask
            safe_membership = jnp.where(active, self.membership, 0)
            atom_anchors = self.anchor_indices[safe_membership]
            anchors = position[atom_anchors]
            relative = cell.minimum_image(position - anchors)
            unwrapped = jnp.where(active[:, None], anchors + relative, 0.0)
            distances = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
            margin = jnp.min(
                jnp.where(active, cell.unique_image_radius - distances, jnp.inf)
            )
        coarse_position = contract("bn,nd->bd", self.center_weights, unwrapped)
        coarse_images = None
        if cell is not None:
            coarse_position, coarse_images = cell.wrap(coarse_position)
        coarse_force = (
            None
            if force is None
            else contract("bn,nd->bd", self.member_mask.astype(force.dtype), force)
        )
        coarse_momentum = (
            None
            if momentum is None
            else contract("bn,nd->bd", self.member_mask.astype(momentum.dtype), momentum)
        )
        force_residual = (
            jnp.asarray(jnp.nan, dtype=position.dtype)
            if coarse_force is None or force is None
            else jnp.max(
                jnp.abs(
                    jnp.sum(coarse_force, axis=0)
                    - jnp.sum(
                        jnp.where(self.fine_system.active_mask[:, None], force, 0.0),
                        axis=0,
                    )
                )
            )
        )
        momentum_residual = (
            jnp.asarray(jnp.nan, dtype=position.dtype)
            if coarse_momentum is None or momentum is None
            else jnp.max(
                jnp.abs(
                    jnp.sum(coarse_momentum, axis=0)
                    - jnp.sum(
                        jnp.where(self.fine_system.active_mask[:, None], momentum, 0.0),
                        axis=0,
                    )
                )
            )
        )
        fine_mass = jnp.sum(
            jnp.where(self.fine_system.active_mask, self.fine_system.plan.masses, 0.0)
        )
        coarse_mass = jnp.sum(self.coarse_system.plan.masses)
        fine_charge = jnp.sum(
            jnp.where(self.fine_system.active_mask, self.fine_system.plan.charges, 0.0)
        )
        coarse_charge = jnp.sum(self.coarse_system.plan.charges)
        finite = jnp.all(jnp.isfinite(coarse_position))
        if coarse_force is not None:
            finite = finite & jnp.all(jnp.isfinite(coarse_force))
        if coarse_momentum is not None:
            finite = finite & jnp.all(jnp.isfinite(coarse_momentum))
        successful = finite & (margin > 0.0)
        return MolecularCoarseMapEvaluation(
            positions=coarse_position,
            forces=coarse_force,
            momenta=coarse_momentum,
            image_counts=coarse_images,
            minimum_image_margin=margin,
            mass_residual=coarse_mass - fine_mass,
            charge_residual=coarse_charge - fine_charge,
            force_residual=force_residual,
            momentum_residual=momentum_residual,
            successful=successful,
            prepared_id=self.prepared_id,
        )


@final
class CoarseForceMatchingProblem(StrictModule, NonTrainableState):
    """Projected-force matching of a coarse potential as a streaming problem.

    Every fine configuration is mapped onto beads; the projected forces (minus
    ``prior_forces`` when a ``prior_id`` declares a prior) become the
    ``forces`` labels of one canonical source record per configuration, whose
    species are bead atom-type IDs and whose masses are the coarse masses in
    the system mass unit. A record's independent-unit lineage is its fine
    structure ID, so validation configurations must carry structure IDs
    disjoint from training. ``training`` is the force-only problem over those
    sources (``objective``, default component-balanced squared force matching
    with a fitted scale) prepared for ``cutoff + skin``. ``projected_forces``
    and ``residual_forces`` keep the training force-scale evidence and
    ``validation_forces`` the validation residual targets.
    """

    mapping: PreparedMolecularCoarseMap
    training_source: AtomisticStructureSource
    validation_source: AtomisticStructureSource | None
    training: AtomisticTrainingProblem
    projected_forces: Array
    residual_forces: Array
    validation_forces: Array | None
    prior_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mapping: PreparedMolecularCoarseMap,
        fine_batch: AtomisticBatch,
        fine_forces: ArrayLike,
        graph_execution: AtomisticGraphExecutionPlan,
        /,
        *,
        cutoff: float,
        skin: float = 0.0,
        objective: AtomisticForceObjective | None = None,
        batch_size: int | None = None,
        seed: int = 0,
        precision: AtomisticTrainingPrecision | None = None,
        prior_forces: ArrayLike | None = None,
        prior_id: str | None = None,
        validation_batch: AtomisticBatch | None = None,
        validation_fine_forces: ArrayLike | None = None,
        validation_prior_forces: ArrayLike | None = None,
    ) -> None:
        if (
            not isinstance(graph_execution, AtomisticGraphExecutionPlan)
            or graph_execution.backend != "dense"
        ):
            raise TypeError("graph_execution must be a dense graph plan.")
        force_objective = AtomisticForceObjective() if objective is None else objective
        if not isinstance(force_objective, AtomisticForceObjective):
            raise TypeError("objective must be an AtomisticForceObjective or None.")
        force = jnp.asarray(fine_forces, dtype=fine_batch.positions.dtype)
        if force.shape != fine_batch.positions.shape:
            raise ValueError("fine_forces must match fine_batch positions.")
        coarse = mapping.coarse_system.plan
        capacity = mapping.coarse_system.capacity

        def mapped_batch(
            batch: AtomisticBatch, labels: Array
        ) -> tuple[AtomisticBatch, Array]:
            evaluations = tuple(
                mapping.evaluate(batch.positions[index], forces=labels[index])
                for index in range(batch.case_count)
            )
            if not all(bool(value.successful) for value in evaluations):
                raise ValueError(
                    "A fine configuration could not be mapped unambiguously."
                )
            positions = jnp.stack(tuple(value.positions for value in evaluations))
            # Mapping with supplied forces always returns projected forces.
            mapped_forces = cast(
                tuple[Array, ...], tuple(value.forces for value in evaluations)
            )
            projected = jnp.stack(mapped_forces)
            count = batch.case_count
            cells = None
            periodic_axes = None
            if coarse.cell is not None:
                cells = jnp.broadcast_to(coarse.cell.vectors, (count, 3, 3))
                periodic_axes = jnp.broadcast_to(coarse.cell.periodic_mask, (count, 3))
            result = AtomisticBatch(
                jnp.zeros((count, capacity), dtype=jnp.int32),
                positions,
                jnp.broadcast_to(coarse.masses, (count, capacity)),
                coarse.units.scale,
                particle_ids=jnp.broadcast_to(coarse.particle_ids, (count, capacity)),
                atom_type_ids=jnp.broadcast_to(coarse.atom_type_ids, (count, capacity)),
                element_mask=jnp.zeros((count, capacity), dtype=jnp.bool_),
                atom_mask=jnp.ones((count, capacity), dtype=jnp.bool_),
                cells=cells,
                periodic_axes=periodic_axes,
                structure_ids=tuple(
                    f"{identifier}:coarse:{mapping.prepared_id}"
                    for identifier in batch.structure_ids
                ),
            )
            return result, projected

        def source(
            batch: AtomisticBatch, fine: AtomisticBatch, labels: Array
        ) -> AtomisticStructureSource:
            return atomistic_training_source(
                batch,
                mass_unit=coarse.units.mass_unit,
                forces=np.asarray(labels, dtype=np.float64),
                lineage_ids=fine.structure_ids,
            )

        training_batch, projected = mapped_batch(fine_batch, force)
        prior = (
            None
            if prior_forces is None
            else jnp.asarray(prior_forces, dtype=projected.dtype)
        )
        if prior is not None and prior.shape != projected.shape:
            raise ValueError("prior_forces must match mapped projected forces.")
        if (prior is None) != (prior_id is None):
            raise ValueError("prior_forces and prior_id must be supplied together.")
        residual = projected if prior is None else projected - prior
        if validation_batch is None:
            if validation_fine_forces is not None or validation_prior_forces is not None:
                raise ValueError("Validation forces require validation_batch.")
            validation_source = None
            validation_residual = None
        else:
            if validation_fine_forces is None:
                raise ValueError("validation_batch requires validation_fine_forces.")
            coarse_validation, validation_projected = mapped_batch(
                validation_batch,
                jnp.asarray(
                    validation_fine_forces, dtype=validation_batch.positions.dtype
                ),
            )
            if prior_id is None:
                if validation_prior_forces is not None:
                    raise ValueError("validation_prior_forces require a prior_id.")
                validation_residual = validation_projected
            else:
                if validation_prior_forces is None:
                    raise ValueError(
                        "A prior-corrected validation set requires validation_prior_forces."
                    )
                validation_prior = jnp.asarray(
                    validation_prior_forces, dtype=validation_projected.dtype
                )
                if validation_prior.shape != validation_projected.shape:
                    raise ValueError(
                        "validation_prior_forces must match projected validation forces."
                    )
                validation_residual = validation_projected - validation_prior
            validation_source = source(
                coarse_validation, validation_batch, validation_residual
            )
        training_source = source(training_batch, fine_batch, residual)
        fields = (
            AtomisticTrainingFields(species_kind=AtomisticSpeciesKind.ATOM_TYPE_ID)
            if coarse.cell is None
            else AtomisticTrainingFields(
                cell="cell",
                periodic="periodic",
                species_kind=AtomisticSpeciesKind.ATOM_TYPE_ID,
            )
        )
        training = AtomisticTrainingProblem(
            atomistic_training_loader(
                training_source, batch_size=batch_size, stream="training", seed=seed
            ),
            graph_execution,
            (force_objective,),
            scale=coarse.units.scale,
            cutoff=cutoff,
            skin=skin,
            fields=fields,
            validation=(
                None
                if validation_source is None
                else atomistic_training_loader(validation_source, stream="validation")
            ),
            precision=precision,
        )
        self.mapping = mapping
        self.training_source = training_source
        self.validation_source = validation_source
        self.training = training
        self.projected_forces = projected
        self.residual_forces = residual
        self.validation_forces = validation_residual
        self.prior_id = prior_id
        self.problem_id = canonical_fingerprint(
            {
                "kind": "coarse-force-matching-problem",
                "mapping": mapping.prepared_id,
                "training": training.problem_id,
                "execution": training.execution_id,
                "prior": prior_id,
            }
        )


@final
class CoarseForceMatchingResult(StrictModule, NonTrainableState):
    """Fitted coarse potential with projected and residual force-scale evidence."""

    training: AtomisticTrainingResult
    projected_force_rms: Array
    residual_force_rms: Array
    valid: Array
    mapping_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def fit_coarse_potential(
    potential: AbstractAtomisticPotential,
    problem: CoarseForceMatchingProblem,
    policy: AtomisticTrainingPolicy,
    key: PRNGKey,
    /,
) -> CoarseForceMatchingResult:
    """Fit an atom-type-ID potential on ``problem.training``.

    ``projected_force_rms`` and ``residual_force_rms`` are the root mean square
    of the projected and prior-corrected training forces; ``valid`` requires a
    successful fit and finite force scales.
    """
    if not isinstance(potential, AbstractAtomisticPotential):
        raise TypeError("potential must implement AbstractAtomisticPotential.")
    if not isinstance(problem, CoarseForceMatchingProblem):
        raise TypeError("problem must be a CoarseForceMatchingProblem.")
    if potential.capabilities.species_kind is not AtomisticSpeciesKind.ATOM_TYPE_ID:
        raise ValueError("Coarse force matching requires an atom-type-ID potential.")
    result = fit_atomistic_potential(potential, problem.training, policy, key=key)
    projected_rms = jnp.sqrt(jnp.mean(problem.projected_forces**2))
    residual_rms = jnp.sqrt(jnp.mean(problem.residual_forces**2))
    result_id = canonical_fingerprint(
        {
            "kind": "coarse-force-matching-result",
            "problem": problem.problem_id,
            "training": result.result_id,
        }
    )
    return CoarseForceMatchingResult(
        training=result,
        projected_force_rms=projected_rms,
        residual_force_rms=residual_rms,
        valid=result.successful
        & jnp.isfinite(projected_rms)
        & jnp.isfinite(residual_rms),
        mapping_id=problem.mapping.prepared_id,
        problem_id=problem.problem_id,
        result_id=result_id,
    )


class MolecularCoarseQualificationResult(StrictModule, NonTrainableState):
    mapping_valid: Array
    force_match_valid: Array
    equilibrium_residual: Array
    equilibrium_valid: Array
    rollout_valid: Array
    claims_satisfied: Array
    mapping_id: str = eqx.field(static=True)
    fit_result_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def qualify_molecular_coarse_model(
    mapping: MolecularCoarseMapEvaluation,
    fit: CoarseForceMatchingResult,
    equilibrium_residual: ArrayLike,
    rollout_valid: ArrayLike,
    /,
    *,
    conservation_tolerance: float = 1.0e-10,
    equilibrium_tolerance: float = 5.0e-2,
) -> MolecularCoarseQualificationResult:
    if not isinstance(mapping, MolecularCoarseMapEvaluation):
        raise TypeError("mapping must be MolecularCoarseMapEvaluation.")
    if not isinstance(fit, CoarseForceMatchingResult):
        raise TypeError("fit must be CoarseForceMatchingResult.")
    conservation = float(conservation_tolerance)
    equilibrium = float(equilibrium_tolerance)
    if (
        not np.isfinite(conservation)
        or conservation <= 0.0
        or not np.isfinite(equilibrium)
        or equilibrium <= 0.0
    ):
        raise ValueError("Qualification tolerances must be finite and positive.")
    residual = jnp.asarray(equilibrium_residual).reshape(())
    rollout = jnp.asarray(rollout_valid, dtype=jnp.bool_).reshape(())
    mapping_valid = (
        mapping.successful
        & (jnp.abs(mapping.mass_residual) <= conservation)
        & (jnp.abs(mapping.charge_residual) <= conservation)
    )
    equilibrium_valid = jnp.isfinite(residual) & (residual <= equilibrium)
    claims = mapping_valid & fit.valid & equilibrium_valid & rollout
    result_id = canonical_fingerprint(
        {
            "kind": "molecular-coarse-qualification",
            "mapping": mapping.prepared_id,
            "fit": fit.result_id,
            "conservation_tolerance": conservation.hex(),
            "equilibrium_tolerance": equilibrium.hex(),
        }
    )
    return MolecularCoarseQualificationResult(
        mapping_valid=mapping_valid,
        force_match_valid=fit.valid,
        equilibrium_residual=residual,
        equilibrium_valid=equilibrium_valid,
        rollout_valid=rollout,
        claims_satisfied=claims,
        mapping_id=mapping.prepared_id,
        fit_result_id=fit.result_id,
        result_id=result_id,
    )


__all__ = [
    "CoarseForceMatchingProblem",
    "CoarseForceMatchingResult",
    "MolecularCoarseQualificationResult",
    "MolecularCoarseMapEvaluation",
    "MolecularCoarseMapPlan",
    "PreparedMolecularCoarseMap",
    "fit_coarse_potential",
    "qualify_molecular_coarse_model",
]
