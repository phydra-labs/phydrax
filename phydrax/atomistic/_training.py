#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from phydrax.optim._adam import adam

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._doc import DOC_KEY0
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._iteration import IterationSession
from .._strict import StrictModule
from .._trainable import (
    combine_parameters,
    ExplicitFreeze,
    NonTrainableState,
)
from .._training import (
    TrainingController,
    TrainingIterationKind,
    TrainingProgress,
)
from .._training_checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
)
from .._training_kernel import (
    build_training_checkpoint,
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    PreparedTrainingKernel,
    restore_training_checkpoint,
    run_training_attempt,
    TrainingKernelSpec,
    TrainingKernelState,
    TrainingKeys,
    TrainingRejectionBudgetError,
)
from .._training_objective import _ObjectiveContribution
from ..discretization._periodic_cell import lattice_measure
from ..typing import checked, parse, PRNGKey
from ._graph import (
    AtomisticGraphExecutionPlan,
    AtomisticGraphTopology,
    prepare_atomistic_graph_topology,
)
from ._potential import AbstractAtomisticPotential, atomistic_potential_revision
from ._prediction import atomistic_energy_derivatives
from ._types import AtomisticBatch, AtomisticStatus


_AtomisticPotential = AbstractAtomisticPotential
AtomisticSupervisionRole: TypeAlias = Literal["training", "validation"]


def _optional_host(value: Array | None, /) -> np.ndarray | None:
    return None if value is None else np.asarray(value)


def _cell_volume_cases(batch: AtomisticBatch, /) -> np.ndarray:
    """Cases whose stress has a declared cell volume.

    A case qualifies when at least one axis is periodic and its supplied
    ``(3, 3)`` cell is a finite, numerically nonsingular embedding whose
    volume comes from the owning lattice measure. Partial periodicity is
    admitted through that explicit embedding cell; its volume is never
    invented from a missing or singular cell.
    """
    if batch.cells is None or batch.periodic_axes is None:
        return np.zeros((batch.case_count,), dtype=np.bool_)
    _, embedded = lattice_measure(batch.cells)
    periodic = np.any(np.asarray(batch.periodic_axes, dtype=np.bool_), axis=1)
    return periodic & np.asarray(embedded, dtype=np.bool_)


def _energy_supervision(
    batch: AtomisticBatch,
    values: ArrayLike | None,
    mask: ArrayLike | None,
    /,
    *,
    prefix: AtomisticSupervisionRole,
) -> tuple[Array | None, Array | None]:
    if values is None:
        if mask is not None:
            raise ValueError(f"{prefix}_energy_mask requires energy labels.")
        return None, None
    energy = jnp.asarray(values, dtype=batch.positions.dtype)
    if energy.shape != (batch.case_count,):
        raise ValueError(f"{prefix}_energy must have shape (case,).")
    energy_mask = (
        jnp.ones((batch.case_count,), dtype=jnp.bool_)
        if mask is None
        else jnp.asarray(mask, dtype=jnp.bool_)
    )
    if energy_mask.shape != energy.shape:
        raise ValueError(f"{prefix}_energy_mask must have shape (case,).")
    if not np.any(np.asarray(energy_mask)):
        raise ValueError(f"{prefix}_energy_mask must select at least one label.")
    return energy, energy_mask


def _force_supervision(
    batch: AtomisticBatch,
    values: ArrayLike | None,
    mask: ArrayLike | None,
    /,
    *,
    prefix: AtomisticSupervisionRole,
) -> tuple[Array | None, Array | None]:
    if values is None:
        if mask is not None:
            raise ValueError(f"{prefix}_force_mask requires force labels.")
        return None, None
    forces = jnp.asarray(values, dtype=batch.positions.dtype)
    if forces.shape != batch.positions.shape:
        raise ValueError(f"{prefix}_forces must have shape (case, atom, 3).")
    active = jnp.broadcast_to(batch.atom_mask[:, :, None], forces.shape)
    force_mask = active if mask is None else jnp.asarray(mask, dtype=jnp.bool_)
    if force_mask.shape != forces.shape:
        raise ValueError(f"{prefix}_force_mask must have shape (case, atom, 3).")
    force_mask = force_mask & active
    if not np.any(np.asarray(force_mask)):
        raise ValueError(f"{prefix}_force_mask must select at least one component.")
    return forces, force_mask


def _stress_supervision(
    batch: AtomisticBatch,
    values: ArrayLike | None,
    mask: ArrayLike | None,
    /,
    *,
    prefix: AtomisticSupervisionRole,
) -> tuple[Array | None, Array | None]:
    if values is None:
        if mask is not None:
            raise ValueError(f"{prefix}_stress_mask requires stress labels.")
        return None, None
    stress = jnp.asarray(values, dtype=batch.positions.dtype)
    shape = (batch.case_count, 3, 3)
    if stress.shape != shape:
        raise ValueError(f"{prefix}_stress must have shape (case, 3, 3).")
    admitted = np.broadcast_to(_cell_volume_cases(batch)[:, None, None], shape)
    selected = admitted if mask is None else np.asarray(mask, dtype=np.bool_)
    if selected.shape != shape:
        raise ValueError(f"{prefix}_stress_mask must have shape (case, 3, 3).")
    if np.any(selected & ~admitted):
        raise ValueError(
            f"{prefix}_stress_mask selects a case without a periodic axis and a "
            "finite nonsingular embedding cell; cell-volume-normalized stress "
            "needs that declared volume."
        )
    if not np.any(selected):
        raise ValueError(
            f"{prefix}_stress must supervise at least one component of a periodic "
            "case with a finite nonsingular embedding cell."
        )
    return stress, jnp.asarray(selected, dtype=jnp.bool_)


class AtomisticSupervisionSplit(StrictModule, NonTrainableState):
    """One supervised split bound to its frozen candidate graph topology.

    Labels use the batch scale: energies in energy units, forces in energy per
    length, and stress as the tensile-positive Cartesian tensor
    ``(1/V) dE/d strain`` in energy per cubic length for a row cell deformed as
    ``H @ F.T``, where ``V`` is the volume of the supplied ``(3, 3)`` cell. For
    partially periodic cases that cell is the caller's explicit embedding, so
    the label is cell-volume-normalized stress, not an intrinsic surface or
    line stress. Stress applies only to cases with a periodic axis and a
    finite nonsingular cell. Masks select supervised entries. The topology is
    prepared on the host and never rebuilt during an optimizer step.
    """

    batch: AtomisticBatch
    topology: AtomisticGraphTopology
    energy: Array | None
    forces: Array | None
    stress: Array | None
    energy_mask: Array | None
    force_mask: Array | None
    stress_mask: Array | None
    role: AtomisticSupervisionRole = eqx.field(static=True)
    split_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        batch: AtomisticBatch,
        topology: AtomisticGraphTopology,
        /,
        *,
        role: AtomisticSupervisionRole,
        energy: ArrayLike | None = None,
        forces: ArrayLike | None = None,
        stress: ArrayLike | None = None,
        energy_mask: ArrayLike | None = None,
        force_mask: ArrayLike | None = None,
        stress_mask: ArrayLike | None = None,
    ) -> None:
        role = parse(role, AtomisticSupervisionRole, "role")
        if (
            topology.case_count != batch.case_count
            or topology.atom_capacity != batch.atom_capacity
        ):
            raise ValueError(
                f"{role} topology case/atom capacity does not match its batch."
            )
        if energy is None and forces is None and stress is None:
            raise ValueError(
                f"The {role} split requires energy, force, or stress labels."
            )
        energy_, energy_mask_ = _energy_supervision(
            batch, energy, energy_mask, prefix=role
        )
        forces_, force_mask_ = _force_supervision(batch, forces, force_mask, prefix=role)
        stress_, stress_mask_ = _stress_supervision(
            batch, stress, stress_mask, prefix=role
        )
        self.batch = batch
        self.topology = topology
        self.energy = energy_
        self.forces = forces_
        self.stress = stress_
        self.energy_mask = energy_mask_
        self.force_mask = force_mask_
        self.stress_mask = stress_mask_
        self.role = role
        self.split_id = canonical_fingerprint(
            {
                "kind": "atomistic-supervision-split",
                "role": role,
                "batch": batch.batch_id,
                "scale": batch.scale.scale_id,
                "topology": topology.topology_id,
                "labels": array_tree_fingerprint(
                    {
                        "energy": _optional_host(energy_),
                        "forces": _optional_host(forces_),
                        "stress": _optional_host(stress_),
                        "energy_mask": _optional_host(energy_mask_),
                        "force_mask": _optional_host(force_mask_),
                        "stress_mask": _optional_host(stress_mask_),
                    }
                ),
            }
        )

    @property
    def target_kinds(self) -> tuple[bool, bool, bool]:
        """Presence of energy, force, and stress labels, in that order."""
        return (
            self.energy is not None,
            self.forces is not None,
            self.stress is not None,
        )


def _split_topology(
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    supplied: AtomisticGraphTopology | None,
    cutoff: float | None,
    skin: float,
    /,
    *,
    role: AtomisticSupervisionRole,
) -> AtomisticGraphTopology:
    if supplied is None:
        if cutoff is None:
            raise ValueError(
                f"Preparing the {role} graph topology requires cutoff when no "
                f"{role}_topology is supplied."
            )
        return prepare_atomistic_graph_topology(
            batch, execution, cutoff=cutoff, skin=skin
        )
    if supplied.execution_id != execution.plan_id:
        raise ValueError(
            f"{role}_topology was prepared under another graph execution plan."
        )
    if cutoff is not None and supplied.search_radius != cutoff + skin:
        raise ValueError(
            f"{role}_topology search radius differs from the requested cutoff + skin."
        )
    return supplied


class AtomisticTrainingProblem(StrictModule, NonTrainableState):
    """Typed energy/force/stress supervision for one train and optional validation split.

    Each split freezes its candidate graph topology at construction, either
    prepared here from ``cutoff`` and ``skin`` or supplied (for example a
    spatial periodic-image topology prepared by the caller). Validation must
    supervise the same target kinds as training.
    """

    graph_execution: AtomisticGraphExecutionPlan
    training: AtomisticSupervisionSplit
    validation: AtomisticSupervisionSplit | None
    problem_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        training_batch: AtomisticBatch,
        /,
        graph_execution: AtomisticGraphExecutionPlan,
        *,
        cutoff: float | None = None,
        skin: float = 0.0,
        training_topology: AtomisticGraphTopology | None = None,
        training_energy: ArrayLike | None = None,
        training_forces: ArrayLike | None = None,
        training_stress: ArrayLike | None = None,
        training_energy_mask: ArrayLike | None = None,
        training_force_mask: ArrayLike | None = None,
        training_stress_mask: ArrayLike | None = None,
        validation_batch: AtomisticBatch | None = None,
        validation_topology: AtomisticGraphTopology | None = None,
        validation_energy: ArrayLike | None = None,
        validation_forces: ArrayLike | None = None,
        validation_stress: ArrayLike | None = None,
        validation_energy_mask: ArrayLike | None = None,
        validation_force_mask: ArrayLike | None = None,
        validation_stress_mask: ArrayLike | None = None,
    ) -> None:
        cutoff_ = None if cutoff is None else float(cutoff)
        skin_ = float(skin)
        if cutoff_ is not None and (not math.isfinite(cutoff_) or cutoff_ <= 0.0):
            raise ValueError("cutoff must be finite and positive when provided.")
        if not math.isfinite(skin_) or skin_ < 0.0:
            raise ValueError("skin must be finite and non-negative.")
        training = AtomisticSupervisionSplit(
            training_batch,
            _split_topology(
                training_batch,
                graph_execution,
                training_topology,
                cutoff_,
                skin_,
                role="training",
            ),
            role="training",
            energy=training_energy,
            forces=training_forces,
            stress=training_stress,
            energy_mask=training_energy_mask,
            force_mask=training_force_mask,
            stress_mask=training_stress_mask,
        )
        if validation_batch is None:
            if any(
                value is not None
                for value in (
                    validation_topology,
                    validation_energy,
                    validation_forces,
                    validation_stress,
                    validation_energy_mask,
                    validation_force_mask,
                    validation_stress_mask,
                )
            ):
                raise ValueError(
                    "Validation topology, labels, and masks require validation_batch."
                )
            validation = None
        else:
            if validation_batch.scale.scale_id != training_batch.scale.scale_id:
                raise ValueError("Training and validation batches must share one scale.")
            validation = AtomisticSupervisionSplit(
                validation_batch,
                _split_topology(
                    validation_batch,
                    graph_execution,
                    validation_topology,
                    cutoff_,
                    skin_,
                    role="validation",
                ),
                role="validation",
                energy=validation_energy,
                forces=validation_forces,
                stress=validation_stress,
                energy_mask=validation_energy_mask,
                force_mask=validation_force_mask,
                stress_mask=validation_stress_mask,
            )
            if validation.target_kinds != training.target_kinds:
                raise ValueError(
                    "Validation must provide the same energy/force/stress target kinds "
                    "as training."
                )
        self.graph_execution = graph_execution
        self.training = training
        self.validation = validation
        self.problem_id = canonical_fingerprint(
            {
                "kind": "atomistic-training-problem",
                "graph_execution": graph_execution.plan_id,
                "scale": training_batch.scale.scale_id,
                "training": training.split_id,
                "validation": None if validation is None else validation.split_id,
            }
        )

    @property
    def splits(self) -> tuple[AtomisticSupervisionSplit, ...]:
        if self.validation is None:
            return (self.training,)
        return (self.training, self.validation)


class AtomisticTrainingPolicy(StrictModule, NonTrainableState):
    """Domain-specific full-batch Adam, scaling, selection, and stopping policy."""

    maximum_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    energy_weight: float = eqx.field(static=True)
    force_weight: float = eqx.field(static=True)
    stress_weight: float = eqx.field(static=True)
    energy_scale: float | None = eqx.field(static=True)
    force_scale: float | None = eqx.field(static=True)
    stress_scale: float | None = eqx.field(static=True)
    normalization_floor: float = eqx.field(static=True)
    validation_every: int = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)
    min_delta: float = eqx.field(static=True)
    select_best: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    continuation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int,
        learning_rate: float = 1e-3,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 1.0,
        energy_scale: float | None = None,
        force_scale: float | None = None,
        stress_scale: float | None = None,
        normalization_floor: float = 1e-12,
        validation_every: int = 1,
        patience: int | None = None,
        min_delta: float = 0.0,
        select_best: bool = True,
    ) -> None:
        steps = int(maximum_steps)
        rate = float(learning_rate)
        weights = {
            "energy_weight": float(energy_weight),
            "force_weight": float(force_weight),
            "stress_weight": float(stress_weight),
        }
        scales = {
            name: None if value is None else float(value)
            for name, value in (
                ("energy_scale", energy_scale),
                ("force_scale", force_scale),
                ("stress_scale", stress_scale),
            )
        }
        floor = float(normalization_floor)
        every = int(validation_every)
        patience_ = None if patience is None else int(patience)
        delta = float(min_delta)
        if steps < 0:
            raise ValueError("maximum_steps must be non-negative.")
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("Loss weights must be finite and non-negative.")
        if sum(weights.values()) <= 0.0:
            raise ValueError("At least one loss weight must be positive.")
        for name, value in scales.items():
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when provided.")
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("normalization_floor must be finite and positive.")
        if every <= 0:
            raise ValueError("validation_every must be positive.")
        if patience_ is not None and patience_ <= 0:
            raise ValueError("patience must be positive when provided.")
        if not math.isfinite(delta) or delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative.")
        continuation_data = {
            "kind": "atomistic-training-continuation-policy",
            "optimizer": "adam",
            "learning_rate": rate,
            **weights,
            **scales,
            "normalization_floor": floor,
            "validation_every": every,
            "patience": patience_,
            "min_delta": delta,
            "select_best": bool(select_best),
        }
        self.maximum_steps = steps
        self.learning_rate = rate
        self.energy_weight = weights["energy_weight"]
        self.force_weight = weights["force_weight"]
        self.stress_weight = weights["stress_weight"]
        self.energy_scale = scales["energy_scale"]
        self.force_scale = scales["force_scale"]
        self.stress_scale = scales["stress_scale"]
        self.normalization_floor = floor
        self.validation_every = every
        self.patience = patience_
        self.min_delta = delta
        self.select_best = bool(select_best)
        self.continuation_id = canonical_fingerprint(continuation_data)
        self.policy_id = canonical_fingerprint(
            {**continuation_data, "maximum_steps": steps}
        )

    def active_targets(
        self, split: AtomisticSupervisionSplit, /
    ) -> tuple[bool, bool, bool]:
        """Energy, force, and stress targets that are labeled and positively weighted."""
        energy, forces, stress = split.target_kinds
        return (
            energy and self.energy_weight > 0.0,
            forces and self.force_weight > 0.0,
            stress and self.stress_weight > 0.0,
        )


class AtomisticTrainingNormalization(StrictModule, NonTrainableState):
    """Loss normalization fitted exclusively from the training split."""

    energy_per_atom_mean: Array
    energy_per_atom_scale: Array
    force_component_scale: Array
    stress_component_scale: Array
    fitted_from_problem_id: str = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)


class AtomisticTrainingResult(StrictModule, ExplicitFreeze):
    """Complete continuation, best/final model, histories, and terminal status.

    A trained-artifact holder: it freezes the embedded potentials on purpose
    (`ExplicitFreeze`); continue training from `potential` explicitly.
    `training_state` is the committed training-kernel state (parameters, Adam
    state, root key, cursors) a continuation resumes; `training_checkpoint_id`
    names the kernel configuration it belongs to. Component histories record the
    normalized energy, force, and stress losses of every accepted update.
    """

    potential: _AtomisticPotential
    best_potential: _AtomisticPotential
    training_state: TrainingKernelState
    normalization: AtomisticTrainingNormalization
    training_loss_history: Array
    energy_loss_history: Array
    force_loss_history: Array
    stress_loss_history: Array
    validation_loss_history: Array
    validation_steps: Array
    final_loss: Array
    best_loss: Array
    status: Array
    progress: TrainingProgress = eqx.field(static=True)
    termination: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    continuation_id: str = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)
    training_checkpoint_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def final_potential(self) -> _AtomisticPotential:
        return self.potential

    @property
    def successful(self) -> Array:
        return (self.status == int(AtomisticStatus.SUCCESS)) | (
            self.status == int(AtomisticStatus.STOPPED_EARLY)
        )


def _fitted_rms(values: Array | None, mask: Array | None, /) -> float:
    if values is None or mask is None:
        return 1.0
    selected = np.asarray(values)[np.asarray(mask, dtype=np.bool_)]
    if not np.all(np.isfinite(selected)):
        return 1.0
    return float(np.sqrt(np.mean(selected * selected)))


def _resolved_scale(fitted: float, override: float | None, floor: float, /) -> float:
    return max(fitted, floor) if override is None else override


def _normalization(
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
) -> AtomisticTrainingNormalization:
    split = problem.training
    energy_mean = 0.0
    fitted_energy_scale = 1.0
    if split.energy is not None:
        values = np.asarray(split.energy) / np.asarray(split.batch.atom_counts)
        selected = values[np.asarray(split.energy_mask, dtype=np.bool_)]
        if np.all(np.isfinite(selected)):
            energy_mean = float(np.mean(selected))
            fitted_energy_scale = float(np.std(selected))
    floor = policy.normalization_floor
    energy_scale = _resolved_scale(fitted_energy_scale, policy.energy_scale, floor)
    force_scale = _resolved_scale(
        _fitted_rms(split.forces, split.force_mask), policy.force_scale, floor
    )
    stress_scale = _resolved_scale(
        _fitted_rms(split.stress, split.stress_mask), policy.stress_scale, floor
    )
    normalization_id = canonical_fingerprint(
        {
            "kind": "atomistic-training-normalization",
            "problem": problem.problem_id,
            "energy_per_atom_mean": energy_mean,
            "energy_per_atom_scale": energy_scale,
            "force_component_scale": force_scale,
            "stress_component_scale": stress_scale,
        }
    )
    dtype = split.batch.positions.dtype
    return AtomisticTrainingNormalization(
        energy_per_atom_mean=jnp.asarray(energy_mean, dtype=dtype),
        energy_per_atom_scale=jnp.asarray(energy_scale, dtype=dtype),
        force_component_scale=jnp.asarray(force_scale, dtype=dtype),
        stress_component_scale=jnp.asarray(stress_scale, dtype=dtype),
        fitted_from_problem_id=problem.problem_id,
        normalization_id=normalization_id,
    )


class _SupervisionLoss(StrictModule):
    """Weighted total and normalized component losses of one split evaluation."""

    total: Array
    energy: Array
    force: Array
    stress: Array


def _normalized_mean_square(
    prediction: Array | None,
    target: Array | None,
    mask: Array | None,
    scale: Array,
    /,
    *,
    kind: str,
    dtype: jnp.dtype,
) -> Array:
    if prediction is None or target is None or mask is None:
        raise RuntimeError(f"Active {kind} supervision lacks a prediction or labels.")
    residual = jnp.where(
        mask, prediction.astype(dtype) - target.astype(dtype), 0.0
    ) / scale.astype(dtype)
    return jnp.sum(residual * residual, dtype=dtype) / jnp.sum(mask, dtype=dtype)


def _loss(
    potential: _AtomisticPotential,
    split: AtomisticSupervisionSplit,
    execution: AtomisticGraphExecutionPlan,
    normalization: AtomisticTrainingNormalization,
    policy: AtomisticTrainingPolicy,
    /,
) -> _SupervisionLoss:
    """Masked normalized E/F/S loss over the split's frozen candidate topology.

    Forces and stress are first derivatives of one scalar energy in positions
    and homogeneous strain, so the parameter gradient of this loss is a mixed
    coordinate/parameter (or strain/parameter) derivative; no Jacobian or
    Hessian is materialized. Overflow or an unsuccessful case poisons the total.
    """
    energy_active, force_active, stress_active = policy.active_targets(split)
    stress_cases = (
        jnp.any(split.stress_mask, axis=(1, 2))
        if stress_active and split.stress_mask is not None
        else None
    )
    derivatives = atomistic_energy_derivatives(
        potential,
        split.batch,
        execution,
        split.batch.positions,
        topology=split.topology,
        cell_vectors=split.batch.cells,
        compute_forces=force_active,
        compute_stress=stress_active,
        stress_case_mask=stress_cases,
    )
    dtype = jnp.dtype(potential.precision.reduction_dtype)
    zero = jnp.zeros((), dtype=dtype)
    energy_loss = zero
    force_loss = zero
    stress_loss = zero
    if energy_active:
        count = split.batch.atom_counts.astype(dtype)
        energy_loss = _normalized_mean_square(
            derivatives.energy.astype(dtype) / count,
            None if split.energy is None else split.energy.astype(dtype) / count,
            split.energy_mask,
            normalization.energy_per_atom_scale,
            kind="energy",
            dtype=dtype,
        )
    if force_active:
        force_loss = _normalized_mean_square(
            derivatives.forces,
            split.forces,
            split.force_mask,
            normalization.force_component_scale,
            kind="force",
            dtype=dtype,
        )
    if stress_active:
        stress_loss = _normalized_mean_square(
            derivatives.stress,
            split.stress,
            split.stress_mask,
            normalization.stress_component_scale,
            kind="stress",
            dtype=dtype,
        )
    total = (
        policy.energy_weight * energy_loss
        + policy.force_weight * force_loss
        + policy.stress_weight * stress_loss
    )
    failed = jnp.any(derivatives.overflow) | ~jnp.all(derivatives.successful)
    total = jnp.where(failed, jnp.asarray(jnp.nan, total.dtype), total)
    return _SupervisionLoss(
        total=total, energy=energy_loss, force=force_loss, stress=stress_loss
    )


_compiled_loss = eqx.filter_jit(_loss)


def _energy_overflow(
    potential: _AtomisticPotential,
    split: AtomisticSupervisionSplit,
    execution: AtomisticGraphExecutionPlan,
    /,
) -> Array:
    derivatives = atomistic_energy_derivatives(
        potential,
        split.batch,
        execution,
        split.batch.positions,
        topology=split.topology,
        cell_vectors=split.batch.cells,
        compute_forces=False,
        compute_stress=False,
    )
    return jnp.any(derivatives.overflow)


_compiled_energy_overflow = eqx.filter_jit(_energy_overflow)


def _split_overflow(
    potential: _AtomisticPotential,
    split: AtomisticSupervisionSplit,
    execution: AtomisticGraphExecutionPlan,
    /,
) -> bool:
    """Host check of candidate-topology and bound-neighborhood capacity overflow."""
    if bool(np.any(np.asarray(split.topology.overflow))):
        return True
    return bool(np.asarray(_compiled_energy_overflow(potential, split, execution)))


def _host_loss(
    potential: _AtomisticPotential,
    problem: AtomisticTrainingProblem,
    normalization: AtomisticTrainingNormalization,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    validation: bool,
) -> _SupervisionLoss:
    """Compiled training (or validation, when a split exists) loss."""
    split = (
        problem.validation
        if validation and problem.validation is not None
        else problem.training
    )
    return _compiled_loss(
        potential, split, problem.graph_execution, normalization, policy
    )


def _supervision_objective(
    parameters: PyTree[Any],
    model_state: PyTree[Any],
    fixed: PyTree[Any],
    payload: tuple[
        AtomisticTrainingProblem, AtomisticTrainingNormalization, AtomisticTrainingPolicy
    ],
    keys: TrainingKeys,
) -> tuple[_ObjectiveContribution, PyTree[Any], _SupervisionLoss]:
    """Full-batch E/F/S data fit; the payload is `(problem, normalization, policy)`."""
    del keys
    problem, normalization, policy = payload
    loss = _loss(
        combine_parameters(parameters, model_state, fixed),
        problem.training,
        problem.graph_execution,
        normalization,
        policy,
    )
    return _ObjectiveContribution(loss.total, 1.0), model_state, loss


def _training_kernel(
    potential: _AtomisticPotential, policy: AtomisticTrainingPolicy, /
) -> PreparedTrainingKernel:
    return prepare_training_kernel(
        potential,
        (
            KernelObjective(
                objective_id="atomistic-energy-force-stress-supervision",
                kind=ObjectiveKind.DATA_FIT,
                route=DerivativeRoute.DIRECT,
                fn=_supervision_objective,
            ),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(
                adam(policy.learning_rate),
                rule_id=f"atomistic-training:{policy.continuation_id}",
            ),
            context="fit_atomistic_potential",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )


def _fingerprint_history(values: Sequence[float], /) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def _validate_fit_inputs(
    potential: _AtomisticPotential,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
) -> None:
    """Refuse unsupported supervision before any optimizer state exists."""
    if not isinstance(potential, AbstractAtomisticPotential):
        raise TypeError("potential must implement AbstractAtomisticPotential.")
    if not isinstance(problem, AtomisticTrainingProblem):
        raise TypeError("problem must be an AtomisticTrainingProblem.")
    if not isinstance(policy, AtomisticTrainingPolicy):
        raise TypeError("policy must be an AtomisticTrainingPolicy.")
    if potential.scale.scale_id != problem.training.batch.scale.scale_id:
        raise ValueError("Potential and training problem must share one scale contract.")
    active = policy.active_targets(problem.training)
    if not any(active):
        raise ValueError(
            "The policy assigns zero weight to every target supervised by the problem."
        )
    if active[2] and not potential.capabilities.cell_derivative:
        raise ValueError(
            "Stress supervision requires a potential with cell-derivative capability."
        )
    cutoff = float(potential.configuration.cutoff)
    for split in problem.splits:
        potential._validate_batch(split.batch)
        if split.topology.search_radius < cutoff:
            raise ValueError(
                "A frozen split topology search radius is smaller than the potential "
                "cutoff."
            )


@dataclass(slots=True)
class _Histories:
    """Host-side accepted-update and selection histories of one training run."""

    training: list[float]
    energy: list[float]
    force: list[float]
    stress: list[float]
    validation: list[float]
    validation_steps: list[int]

    def record_update(self, loss: _SupervisionLoss, /) -> None:
        self.training.append(float(np.asarray(loss.total)))
        self.energy.append(float(np.asarray(loss.energy)))
        self.force.append(float(np.asarray(loss.force)))
        self.stress.append(float(np.asarray(loss.stress)))

    def record_rejection(self) -> None:
        for history in (self.training, self.energy, self.force, self.stress):
            history.append(float("nan"))


def _new_histories() -> _Histories:
    return _Histories(
        training=[], energy=[], force=[], stress=[], validation=[], validation_steps=[]
    )


def _continued_histories(continuation: AtomisticTrainingResult, /) -> _Histories:
    return _Histories(
        training=np.asarray(continuation.training_loss_history).tolist(),
        energy=np.asarray(continuation.energy_loss_history).tolist(),
        force=np.asarray(continuation.force_loss_history).tolist(),
        stress=np.asarray(continuation.stress_loss_history).tolist(),
        validation=np.asarray(continuation.validation_loss_history).tolist(),
        validation_steps=np.asarray(continuation.validation_steps).tolist(),
    )


def _resume(
    potential: _AtomisticPotential,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    normalization: AtomisticTrainingNormalization,
    continuation: AtomisticTrainingResult,
    /,
) -> tuple[PreparedTrainingKernel, TrainingKernelState]:
    if not isinstance(continuation, AtomisticTrainingResult):
        raise TypeError("continuation must be an AtomisticTrainingResult or None.")
    if (
        type(potential) is not type(continuation.potential)
        or potential.architecture_id != continuation.potential.architecture_id
        or potential.capabilities.capabilities_id != continuation.capabilities_id
    ):
        raise ValueError(
            "Continuation potential must have the same concrete family and configuration as the supplied potential."
        )
    if continuation.problem_id != problem.problem_id:
        raise ValueError("Continuation result belongs to a different training problem.")
    if continuation.continuation_id != policy.continuation_id:
        raise ValueError(
            "Continuation policy changed optimizer, loss, or selection semantics."
        )
    if continuation.normalization.normalization_id != normalization.normalization_id:
        raise ValueError(
            "Continuation normalization no longer matches the training split."
        )
    if continuation.progress.update_step > policy.maximum_steps:
        raise ValueError("Continuation step exceeds the requested training ceiling.")
    kernel = _training_kernel(continuation.potential, policy)
    if continuation.training_checkpoint_id != kernel.checkpoint_id:
        raise ValueError(
            "Continuation training state belongs to a different training kernel."
        )
    state = restore_training_checkpoint(
        kernel,
        build_training_checkpoint(
            kernel, continuation.training_state, allow_intermediate=True
        ),
    ).state
    return kernel, state


def _overflow_termination(
    potential: _AtomisticPotential, problem: AtomisticTrainingProblem, /
) -> str | None:
    training = _split_overflow(potential, problem.training, problem.graph_execution)
    validation = problem.validation is not None and _split_overflow(
        potential, problem.validation, problem.graph_execution
    )
    if training and validation:
        return "training_and_validation_neighbor_overflow"
    if training:
        return "training_neighbor_overflow"
    if validation:
        return "validation_neighbor_overflow"
    return None


def fit_atomistic_potential(
    potential: _AtomisticPotential,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    key: PRNGKey = DOC_KEY0,
    session: IterationSession | None = None,
    continuation: AtomisticTrainingResult | None = None,
) -> AtomisticTrainingResult:
    """Fit one atomistic energy potential with typed energy/force/stress supervision.

    Finite and periodic batches train over their frozen per-split candidate
    topologies. Every update is one full-batch Adam attempt of the shared
    training kernel (`MODEL` root authority, one data-fit objective). A
    nonfinite training loss or gradient, including capacity overflow, rolls the
    attempt back and terminates with `NONFINITE`; an update whose post-update
    training loss is nonfinite is discarded the same way, so `potential` is
    always the last finite accepted state. Unsupported periodic, cell, or
    target configurations are refused before any optimizer state exists. A
    continuation resumes its committed kernel state, including its root key.
    """

    _validate_fit_inputs(potential, problem, policy)
    normalization = _normalization(problem, policy)
    if continuation is None:
        kernel = _training_kernel(potential, policy)
        state = kernel.init(potential, key)
        progress = TrainingProgress()
        histories = _new_histories()
    else:
        kernel, state = _resume(potential, problem, policy, normalization, continuation)
        progress = continuation.progress
        histories = _continued_histories(continuation)
    current = kernel.tree(state)
    control = TrainingController(
        total_steps=policy.maximum_steps,
        algorithm_id="atomistic-training",
        progress=progress,
        session=session,
    )
    if continuation is not None:
        control.best_payload = continuation.best_potential
    control.emit(TrainingIterationKind.RUN_START)
    terminal_status = AtomisticStatus.SUCCESS
    termination = "maximum_steps"
    overflow = _overflow_termination(current, problem)
    if overflow is not None:
        terminal_status = AtomisticStatus.NEIGHBOR_OVERFLOW
        termination = overflow
    elif continuation is None:
        initial_loss = _host_loss(
            current, problem, normalization, policy, validation=True
        ).total
        initial_value = float(np.asarray(initial_loss))
        histories.validation.append(initial_value)
        histories.validation_steps.append(0)
        if math.isfinite(initial_value):
            control.select(
                initial_value,
                current,
                step=0,
                min_delta=policy.min_delta,
                patience=policy.patience,
            )
            control.emit(TrainingIterationKind.VALIDATION, metrics={"loss": initial_loss})
        else:
            terminal_status = AtomisticStatus.NONFINITE
            termination = "nonfinite_initial_loss"
    if terminal_status == AtomisticStatus.SUCCESS and control.stop_requested:
        terminal_status = AtomisticStatus.STOPPED_EARLY
        termination = "host_control_stop_before_first_update"

    payload = (problem, normalization, policy)
    for step in range(progress.update_step + 1, policy.maximum_steps + 1):
        if terminal_status != AtomisticStatus.SUCCESS or control.stop_requested:
            break
        try:
            candidate_state, _ = run_training_attempt(kernel, state, payload)
        except TrainingRejectionBudgetError:
            candidate_state = None
            termination = "nonfinite_training_loss_or_gradient"
        if candidate_state is not None:
            candidate = kernel.tree(candidate_state)
            post = _host_loss(candidate, problem, normalization, policy, validation=False)
            if not bool(np.asarray(jnp.isfinite(post.total))):
                candidate_state = None
                termination = "nonfinite_updated_loss"
        if candidate_state is None:
            # The update was rolled back; the potential is the last accepted one.
            histories.record_rejection()
            terminal_status = AtomisticStatus.NONFINITE
            break
        state = candidate_state
        current = candidate
        histories.record_update(post)
        control.complete_update(step)
        control.emit(
            TrainingIterationKind.UPDATE,
            metrics={
                "loss": post.total,
                "energy_loss": post.energy,
                "force_loss": post.force,
                "stress_loss": post.stress,
            },
        )
        validate = step % policy.validation_every == 0 or step == policy.maximum_steps
        if validate:
            selected_loss = _host_loss(
                current, problem, normalization, policy, validation=True
            ).total
            selected_value = float(np.asarray(selected_loss))
            histories.validation.append(selected_value)
            histories.validation_steps.append(step)
            if not math.isfinite(selected_value):
                terminal_status = AtomisticStatus.NONFINITE
                termination = "nonfinite_validation_loss"
                break
            control.select(
                selected_value,
                current,
                step=step,
                min_delta=policy.min_delta,
                patience=policy.patience,
            )
            control.emit(
                TrainingIterationKind.VALIDATION, metrics={"loss": selected_loss}
            )
        if control.stop_requested:
            terminal_status = AtomisticStatus.STOPPED_EARLY
            termination = "selection_or_host_control_stop"
            break

    if not histories.validation and terminal_status not in (
        AtomisticStatus.NEIGHBOR_OVERFLOW,
        AtomisticStatus.NONFINITE,
    ):
        selected_loss = _host_loss(
            current, problem, normalization, policy, validation=True
        ).total
        selected_value = float(np.asarray(selected_loss))
        histories.validation.append(selected_value)
        histories.validation_steps.append(control.progress.update_step)
        if math.isfinite(selected_value):
            control.select(selected_value, current, step=control.progress.update_step)
        else:
            terminal_status = AtomisticStatus.NONFINITE
            termination = "nonfinite_selection_loss"
    best_potential = control.selected(current) if policy.select_best else current
    result = _assemble_result(
        current,
        best_potential,
        state,
        normalization,
        histories,
        control.progress,
        problem_id=problem.problem_id,
        policy_id=policy.policy_id,
        continuation_id=policy.continuation_id,
        training_checkpoint_id=kernel.checkpoint_id,
        status=terminal_status,
        termination=termination,
        dtype=problem.training.batch.positions.dtype,
    )
    control.emit(
        TrainingIterationKind.RUN_TERMINAL,
        metrics={"final_loss": result.final_loss, "best_loss": result.best_loss},
    )
    return result


def _assemble_result(
    potential: _AtomisticPotential,
    best_potential: _AtomisticPotential,
    state: TrainingKernelState,
    normalization: AtomisticTrainingNormalization,
    histories: _Histories,
    progress: TrainingProgress,
    /,
    *,
    problem_id: str,
    policy_id: str,
    continuation_id: str,
    training_checkpoint_id: str,
    status: AtomisticStatus,
    termination: str,
    dtype: np.dtype,
) -> AtomisticTrainingResult:
    """Build a result and its content identity from committed host state."""
    if histories.training:
        final_loss = histories.training[-1]
    elif histories.validation:
        final_loss = histories.validation[-1]
    else:
        final_loss = float("nan")
    best_loss = float("nan") if progress.best_value is None else progress.best_value
    capabilities_id = potential.capabilities.capabilities_id
    result_id = canonical_fingerprint(
        {
            "kind": "atomistic-training-result",
            "problem": problem_id,
            "policy": policy_id,
            "capabilities": capabilities_id,
            "potential": atomistic_potential_revision(potential).revision_id,
            "best_potential": atomistic_potential_revision(best_potential).revision_id,
            "normalization": normalization.normalization_id,
            "training_checkpoint": training_checkpoint_id,
            "updates": progress.update_step,
            "iteration_session": progress.iteration_session_id,
            "iteration_control": progress.iteration_control_id,
            "status": int(status),
            "termination": termination,
            "training_history": _fingerprint_history(histories.training),
            "energy_history": _fingerprint_history(histories.energy),
            "force_history": _fingerprint_history(histories.force),
            "stress_history": _fingerprint_history(histories.stress),
            "validation_history": _fingerprint_history(histories.validation),
            "validation_steps": list(histories.validation_steps),
        }
    )
    return AtomisticTrainingResult(
        potential=potential,
        best_potential=best_potential,
        training_state=state,
        normalization=normalization,
        training_loss_history=jnp.asarray(histories.training, dtype=dtype),
        energy_loss_history=jnp.asarray(histories.energy, dtype=dtype),
        force_loss_history=jnp.asarray(histories.force, dtype=dtype),
        stress_loss_history=jnp.asarray(histories.stress, dtype=dtype),
        validation_loss_history=jnp.asarray(histories.validation, dtype=dtype),
        validation_steps=jnp.asarray(histories.validation_steps, dtype=jnp.int32),
        final_loss=jnp.asarray(final_loss, dtype=dtype),
        best_loss=jnp.asarray(best_loss, dtype=dtype),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        progress=progress,
        termination=termination,
        problem_id=problem_id,
        policy_id=policy_id,
        continuation_id=continuation_id,
        capabilities_id=capabilities_id,
        training_checkpoint_id=training_checkpoint_id,
        result_id=result_id,
    )


_CHECKPOINT_FORMAT = "atomistic-training-checkpoint"
_CHECKPOINT_FIELDS = frozenset(
    {
        "problem_id",
        "policy_id",
        "continuation_id",
        "capabilities_id",
        "training_checkpoint_id",
        "normalization_id",
        "status",
        "termination",
        "update_count",
        "validation_count",
        "result_id",
    }
)


def write_atomistic_training_checkpoint(
    path: str | Path,
    result: AtomisticTrainingResult,
    policy: AtomisticTrainingPolicy,
    /,
) -> Path:
    """Atomically write a pickle-free fresh-process continuation boundary.

    The directory holds the committed training-kernel payload (parameters,
    Adam state, root key, cursors, and selection progress), the best potential,
    and the component histories. Normalization is not stored: it is refitted
    from the training split on restore and must reproduce its identity.
    """
    if not isinstance(result, AtomisticTrainingResult):
        raise TypeError("result must be an AtomisticTrainingResult.")
    if not isinstance(policy, AtomisticTrainingPolicy):
        raise TypeError("policy must be an AtomisticTrainingPolicy.")
    if result.continuation_id != policy.continuation_id:
        raise ValueError("policy does not match the result's continuation semantics.")
    kernel = _training_kernel(result.potential, policy)
    if kernel.checkpoint_id != result.training_checkpoint_id:
        raise ValueError("Result training state belongs to a different training kernel.")
    destination = Path(path)
    save_training_checkpoint(
        destination,
        # Every attempt is atomic, so a post-rejection state is a boundary.
        build_training_checkpoint(
            kernel,
            result.training_state,
            selection=result.progress,
            allow_intermediate=True,
        ),
        (
            result.best_potential,
            result.training_loss_history,
            result.energy_loss_history,
            result.force_loss_history,
            result.stress_loss_history,
            result.validation_loss_history,
            result.validation_steps,
        ),
        format=_CHECKPOINT_FORMAT,
        metadata={
            "problem_id": result.problem_id,
            "policy_id": result.policy_id,
            "continuation_id": result.continuation_id,
            "capabilities_id": result.capabilities_id,
            "training_checkpoint_id": result.training_checkpoint_id,
            "normalization_id": result.normalization.normalization_id,
            "status": int(np.asarray(result.status)),
            "termination": result.termination,
            "update_count": result.training_loss_history.shape[0],
            "validation_count": result.validation_loss_history.shape[0],
            "result_id": result.result_id,
        },
    )
    return destination


def _checkpoint_count(metadata: dict[str, Any], name: str, /) -> int:
    value = metadata[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Atomistic training checkpoint field {name!r} is invalid.")
    return value


def read_atomistic_training_checkpoint(
    path: str | Path,
    potential: _AtomisticPotential,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
) -> AtomisticTrainingResult:
    """Restore a continuation result against a fresh potential, problem and policy.

    ``potential`` is a structural template of the trained family and
    configuration; its parameters are replaced by the checkpoint. The problem,
    policy continuation semantics, refitted normalization, capabilities, kernel
    and the restored content identity must all match exactly; serialized
    identities are verified, never trusted. ``policy`` may raise
    ``maximum_steps`` for a continued run.
    """
    _validate_fit_inputs(potential, problem, policy)
    metadata = read_training_checkpoint_metadata(path, format=_CHECKPOINT_FORMAT)
    if set(metadata) != _CHECKPOINT_FIELDS:
        raise ValueError("Atomistic training checkpoint manifest is not canonical.")
    normalization = _normalization(problem, policy)
    kernel = _training_kernel(potential, policy)
    expected = {
        "problem_id": problem.problem_id,
        "continuation_id": policy.continuation_id,
        "capabilities_id": potential.capabilities.capabilities_id,
        "training_checkpoint_id": kernel.checkpoint_id,
        "normalization_id": normalization.normalization_id,
    }
    if any(metadata[name] != value for name, value in expected.items()):
        raise ValueError(
            "Atomistic training checkpoint problem, policy, normalization, "
            "capability, or kernel identity mismatch."
        )
    if not isinstance(metadata["policy_id"], str) or not isinstance(
        metadata["termination"], str
    ):
        raise ValueError("Atomistic training checkpoint identities are invalid.")
    status = AtomisticStatus(_checkpoint_count(metadata, "status"))
    updates = _checkpoint_count(metadata, "update_count")
    validations = _checkpoint_count(metadata, "validation_count")
    dtype = problem.training.batch.positions.dtype
    history = jnp.zeros((updates,), dtype=dtype)
    loaded = load_training_checkpoint(
        path,
        kernel,
        kernel.init(potential, DOC_KEY0),
        (
            potential,
            history,
            history,
            history,
            history,
            jnp.zeros((validations,), dtype=dtype),
            jnp.zeros((validations,), dtype=jnp.int32),
        ),
        format=_CHECKPOINT_FORMAT,
    )
    progress = loaded.restored.selection
    if progress is None:
        raise ValueError("Atomistic training checkpoint lacks selection progress.")
    (
        best_potential,
        training_history,
        energy_history,
        force_history,
        stress_history,
        validation_history,
        validation_steps,
    ) = loaded.extra
    state = loaded.restored.state
    result = _assemble_result(
        kernel.tree(state),
        best_potential,
        state,
        normalization,
        _Histories(
            training=np.asarray(training_history).tolist(),
            energy=np.asarray(energy_history).tolist(),
            force=np.asarray(force_history).tolist(),
            stress=np.asarray(stress_history).tolist(),
            validation=np.asarray(validation_history).tolist(),
            validation_steps=np.asarray(validation_steps).tolist(),
        ),
        progress,
        problem_id=problem.problem_id,
        policy_id=metadata["policy_id"],
        continuation_id=policy.continuation_id,
        training_checkpoint_id=kernel.checkpoint_id,
        status=status,
        termination=metadata["termination"],
        dtype=dtype,
    )
    if result.result_id != metadata["result_id"]:
        raise ValueError("Atomistic training checkpoint content identity is corrupt.")
    return result


__all__ = [
    "AtomisticSupervisionRole",
    "AtomisticSupervisionSplit",
    "AtomisticTrainingNormalization",
    "AtomisticTrainingPolicy",
    "AtomisticTrainingProblem",
    "AtomisticTrainingResult",
    "fit_atomistic_potential",
    "read_atomistic_training_checkpoint",
    "write_atomistic_training_checkpoint",
]
