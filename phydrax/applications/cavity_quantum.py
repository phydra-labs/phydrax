#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_mesh import CellMesh
from ..discretization.fem._nedelec_tetrahedron import TetrahedralNedelecSpace
from ..linalg import (
    ArraySpace,
    ComplexCartesianCoordinates,
    prepare_real_coordinate_tree,
    PreparedRealCoordinateTree,
)
from ..solver._differential import DifferentialProblem
from ..solver._finite_element_adaptivity import FiniteElementTopologyTransaction


def _cavity_state_coordinates(state: Any, /) -> PreparedRealCoordinateTree:
    treedef = jax.tree.structure(state)
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(state))
    maps = tuple(
        ComplexCartesianCoordinates(ArraySpace(leaf.shape, dtype=leaf.dtype))
        if jnp.issubdtype(leaf.dtype, jnp.complexfloating)
        else None
        for leaf in leaves
    )
    return prepare_real_coordinate_tree(
        state,
        jax.tree.unflatten(treedef, maps),
    )


class CavityModeStatus(IntEnum):
    """Fail-closed status for one normalized Maxwell eigenmode."""

    SUCCESS = 0
    NONFINITE = 1
    NONPOSITIVE_ENERGY = 2
    EQUIPARTITION_FAILURE = 3
    NORMALIZATION_FAILURE = 4


class MaxwellEigenmodeNormalizationEvidence(StrictModule):
    electric_energies: Array
    magnetic_energies: Array
    total_energies: Array
    target_energies: Array
    electric_magnetic_residuals: Array
    normalization_residuals: Array
    status: Array
    successful: Array


class NormalizedMaxwellEigenmodes(StrictModule):
    """Complex phasors normalized to one declared energy per mode."""

    electric_modes: Array
    magnetic_modes: Array
    angular_frequencies: Array
    scale_factors: Array
    evidence: MaxwellEigenmodeNormalizationEvidence
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class MaxwellEigenmodeNormalizationPlan(StrictModule, NonTrainableState):
    """Energy inner products and single-quantum normalization for Maxwell modes.

    Electric and magnetic inputs are peak complex phasors. Their cycle-averaged
    energy is ``(E* M_e E + H* M_h H)/4``. The default target is ``hbar*omega``;
    callers may instead provide an explicit positive per-mode energy.
    """

    electric_energy_metric: Array
    magnetic_energy_metric: Array
    angular_frequencies: Array
    target_energies: Array
    hbar: Array
    equipartition_tolerance: float = eqx.field(static=True)
    normalization_tolerance: float = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    maximum_dofs: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        electric_energy_metric: ArrayLike,
        magnetic_energy_metric: ArrayLike,
        angular_frequencies: ArrayLike,
        /,
        *,
        hbar: ArrayLike = 1.054571817e-34,
        target_energies: ArrayLike | None = None,
        equipartition_tolerance: float = 1e-6,
        normalization_tolerance: float = 1e-10,
        maximum_modes: int = 4096,
        maximum_dofs: int = 2**18,
        maximum_workspace_bytes: int = 2**31,
    ):
        electric = np.asarray(electric_energy_metric)
        magnetic = np.asarray(magnetic_energy_metric)
        frequencies = np.asarray(angular_frequencies, dtype=np.float64)
        hbar_value = np.asarray(hbar, dtype=np.float64)
        if (
            electric.ndim != 2
            or electric.shape[0] != electric.shape[1]
            or electric.shape[0] < 1
        ):
            raise ValueError(
                "electric_energy_metric must be one non-empty square matrix."
            )
        if (
            magnetic.ndim != 2
            or magnetic.shape[0] != magnetic.shape[1]
            or magnetic.shape[0] < 1
        ):
            raise ValueError(
                "magnetic_energy_metric must be one non-empty square matrix."
            )
        if not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
            raise ValueError("Maxwell energy metrics must be finite.")
        if (
            np.max(np.abs(electric - electric.conj().T)) > 1e-10
            or np.max(np.abs(magnetic - magnetic.conj().T)) > 1e-10
        ):
            raise ValueError("Maxwell energy metrics must be Hermitian.")
        if (
            np.min(np.linalg.eigvalsh(electric)) <= 0.0
            or np.min(np.linalg.eigvalsh(magnetic)) <= 0.0
        ):
            raise ValueError("Maxwell energy metrics must be positive definite.")
        if (
            frequencies.ndim != 1
            or frequencies.size < 1
            or not np.all(np.isfinite(frequencies) & (frequencies > 0.0))
        ):
            raise ValueError("angular_frequencies must be one positive finite vector.")
        if (
            hbar_value.shape != ()
            or not np.isfinite(hbar_value)
            or float(hbar_value) <= 0.0
        ):
            raise ValueError("hbar must be one positive finite scalar in joule seconds.")
        targets = (
            float(hbar_value) * frequencies
            if target_energies is None
            else np.asarray(target_energies, dtype=np.float64)
        )
        if targets.shape != frequencies.shape or not np.all(
            np.isfinite(targets) & (targets > 0.0)
        ):
            raise ValueError("target_energies must be positive and mode-aligned.")
        equipartition = float(equipartition_tolerance)
        normalization = float(normalization_tolerance)
        if (
            not isfinite(equipartition)
            or equipartition < 0.0
            or not isfinite(normalization)
            or normalization < 0.0
        ):
            raise ValueError(
                "Mode normalization tolerances must be finite and non-negative."
            )
        for name, value in (
            ("maximum_modes", maximum_modes),
            ("maximum_dofs", maximum_dofs),
            ("maximum_workspace_bytes", maximum_workspace_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        mode_limit = maximum_modes
        dof_limit = maximum_dofs
        workspace_limit = maximum_workspace_bytes
        if mode_limit < 1 or dof_limit < 1 or workspace_limit < 1:
            raise ValueError("Mode normalization resources must be positive.")
        if frequencies.size > mode_limit:
            raise MemoryError("Maxwell mode count exceeds maximum_modes.")
        if max(electric.shape[0], magnetic.shape[0]) > dof_limit:
            raise MemoryError("Maxwell mode degrees of freedom exceed maximum_dofs.")
        itemsize = max(electric.dtype.itemsize, magnetic.dtype.itemsize)
        workspace = frequencies.size * (electric.size + magnetic.size) * itemsize
        if workspace > workspace_limit:
            raise MemoryError("Maxwell normalization exceeds maximum_workspace_bytes.")
        self.electric_energy_metric = jnp.asarray(electric)
        self.magnetic_energy_metric = jnp.asarray(magnetic)
        self.angular_frequencies = jnp.asarray(frequencies)
        self.target_energies = jnp.asarray(targets)
        self.hbar = jnp.asarray(hbar_value)
        self.equipartition_tolerance = equipartition
        self.normalization_tolerance = normalization
        self.maximum_modes = mode_limit
        self.maximum_dofs = dof_limit
        self.maximum_workspace_bytes = workspace_limit
        plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-eigenmode-energy-normalization",
                "electric_metric": array_tree_fingerprint(electric),
                "magnetic_metric": array_tree_fingerprint(magnetic),
                "angular_frequencies": array_tree_fingerprint(frequencies),
                "target_energies": array_tree_fingerprint(targets),
                "hbar": array_tree_fingerprint(hbar_value),
                "phasor_convention": "peak-complex-cycle-average-quarter",
                "equipartition_tolerance": equipartition,
                "normalization_tolerance": normalization,
                "maximum_modes": mode_limit,
                "maximum_dofs": dof_limit,
                "maximum_workspace_bytes": workspace_limit,
            }
        )
        self.plan_id = plan_id
        self.mode_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "normalized-maxwell-eigenmode",
                    "plan": plan_id,
                    "index": index,
                    "frequency": float(frequency),
                }
            )
            for index, frequency in enumerate(frequencies)
        )

    def prepare(self, /) -> PreparedMaxwellEigenmodeNormalization:
        return PreparedMaxwellEigenmodeNormalization(self)

    def _normalize(
        self,
        electric_modes: ArrayLike,
        magnetic_modes: ArrayLike,
        /,
    ) -> NormalizedMaxwellEigenmodes:
        electric = jnp.asarray(electric_modes)
        magnetic = jnp.asarray(magnetic_modes)
        mode_count = self.angular_frequencies.size
        if electric.shape != (mode_count, self.electric_energy_metric.shape[0]):
            raise ValueError("electric_modes must have shape (modes, electric_dofs).")
        if magnetic.shape != (mode_count, self.magnetic_energy_metric.shape[0]):
            raise ValueError("magnetic_modes must have shape (modes, magnetic_dofs).")
        electric_energy = 0.25 * jnp.real(
            ein.contract(
                "mi,ij,mj->m",
                jnp.conj(electric),
                self.electric_energy_metric,
                electric,
            )
        )
        magnetic_energy = 0.25 * jnp.real(
            ein.contract(
                "mi,ij,mj->m",
                jnp.conj(magnetic),
                self.magnetic_energy_metric,
                magnetic,
            )
        )
        total = electric_energy + magnetic_energy
        finite = (
            jnp.all(jnp.isfinite(electric), axis=-1)
            & jnp.all(jnp.isfinite(magnetic), axis=-1)
            & jnp.isfinite(total)
        )
        positive = total > 0.0
        safe_total = jnp.where(positive, total, 1.0)
        scales = jnp.sqrt(self.target_energies / safe_total)
        normalized_electric = scales[:, None] * electric
        normalized_magnetic = scales[:, None] * magnetic
        final_electric = scales**2 * electric_energy
        final_magnetic = scales**2 * magnetic_energy
        final_total = final_electric + final_magnetic
        equipartition_residual = jnp.abs(final_electric - final_magnetic) / jnp.maximum(
            final_total, jnp.finfo(final_total.dtype).tiny
        )
        normalization_residual = (
            jnp.abs(final_total - self.target_energies) / self.target_energies
        )
        status = jnp.where(
            ~finite,
            int(CavityModeStatus.NONFINITE),
            jnp.where(
                ~positive,
                int(CavityModeStatus.NONPOSITIVE_ENERGY),
                jnp.where(
                    equipartition_residual > self.equipartition_tolerance,
                    int(CavityModeStatus.EQUIPARTITION_FAILURE),
                    jnp.where(
                        normalization_residual > self.normalization_tolerance,
                        int(CavityModeStatus.NORMALIZATION_FAILURE),
                        int(CavityModeStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        successful = status == int(CavityModeStatus.SUCCESS)
        evidence = MaxwellEigenmodeNormalizationEvidence(
            final_electric,
            final_magnetic,
            final_total,
            self.target_energies,
            equipartition_residual,
            normalization_residual,
            status,
            successful,
        )
        result_id = canonical_fingerprint(
            {
                "kind": "normalized-maxwell-eigenmodes-result",
                "plan": self.plan_id,
                "input_shapes": [list(electric.shape), list(magnetic.shape)],
                "input_dtypes": [str(electric.dtype), str(magnetic.dtype)],
            }
        )
        return NormalizedMaxwellEigenmodes(
            normalized_electric,
            normalized_magnetic,
            self.angular_frequencies,
            scales,
            evidence,
            self.mode_ids,
            self.plan_id,
            result_id,
        )


class PreparedMaxwellEigenmodeNormalization(StrictModule, NonTrainableState):
    """Prepared fixed-metric normalization executable."""

    plan: MaxwellEigenmodeNormalizationPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MaxwellEigenmodeNormalizationPlan, /):
        if not isinstance(plan, MaxwellEigenmodeNormalizationPlan):
            raise TypeError("plan must be MaxwellEigenmodeNormalizationPlan.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-eigenmode-normalization",
                "plan": plan.plan_id,
            }
        )

    def normalize(
        self,
        electric_modes: ArrayLike,
        magnetic_modes: ArrayLike,
        /,
    ) -> NormalizedMaxwellEigenmodes:
        return self.plan._normalize(electric_modes, magnetic_modes)


class CavityParticipationResult(StrictModule):
    region_energies: Array
    participation_ratios: Array
    residual_participation: Array
    finite: Array
    successful: Array
    region_ids: tuple[str, ...] = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class CavityParticipationPlan(StrictModule, NonTrainableState):
    """Named positive-semidefinite electric-energy region metrics."""

    region_metrics: tuple[Array, ...]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    electric_dofs: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        region_metrics: Mapping[str, ArrayLike] | Sequence[tuple[str, ArrayLike]],
        /,
        *,
        tolerance: float = 1e-10,
    ):
        records = (
            tuple(region_metrics.items())
            if isinstance(region_metrics, Mapping)
            else tuple(region_metrics)
        )
        if not records:
            raise ValueError("At least one participation region is required.")
        names = tuple(str(name).strip() for name, _ in records)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Participation region IDs must be unique and non-empty.")
        matrices = tuple(np.asarray(matrix) for _, matrix in records)
        dimension = matrices[0].shape[0]
        if any(
            matrix.ndim != 2 or matrix.shape != (dimension, dimension)
            for matrix in matrices
        ):
            raise ValueError(
                "Participation metrics must be equally sized square matrices."
            )
        if any(not np.isfinite(matrix).all() for matrix in matrices):
            raise ValueError("Participation metrics must be finite.")
        if any(np.max(np.abs(matrix - matrix.conj().T)) > 1e-10 for matrix in matrices):
            raise ValueError("Participation metrics must be Hermitian.")
        if any(np.min(np.linalg.eigvalsh(matrix)) < -1e-10 for matrix in matrices):
            raise ValueError("Participation metrics must be positive semidefinite.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Participation tolerance must be finite and non-negative.")
        self.region_metrics = tuple(jnp.asarray(matrix) for matrix in matrices)
        self.region_ids = names
        self.electric_dofs = dimension
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cavity-electric-participation",
                "regions": [
                    {"id": name, "metric": array_tree_fingerprint(matrix)}
                    for name, matrix in zip(names, matrices, strict=True)
                ],
                "tolerance": tolerance_,
            }
        )

    def evaluate(
        self, modes: NormalizedMaxwellEigenmodes, /
    ) -> CavityParticipationResult:
        if not isinstance(modes, NormalizedMaxwellEigenmodes):
            raise TypeError("modes must be NormalizedMaxwellEigenmodes.")
        if modes.electric_modes.shape[1] != self.electric_dofs:
            raise ValueError("Participation metrics and electric modes disagree.")
        metrics = jnp.stack(self.region_metrics, axis=0)
        energies = 0.25 * jnp.real(
            ein.contract(
                "mi,rij,mj->mr",
                jnp.conj(modes.electric_modes),
                metrics,
                modes.electric_modes,
            )
        )
        ratios = energies / modes.evidence.total_energies[:, None]
        residual = 1.0 - jnp.sum(ratios, axis=-1)
        finite = jnp.all(jnp.isfinite(ratios), axis=-1)
        successful = (
            modes.evidence.successful
            & finite
            & jnp.all(ratios >= -self.tolerance, axis=-1)
            & (residual >= -self.tolerance)
        )
        return CavityParticipationResult(
            energies,
            ratios,
            residual,
            finite,
            successful,
            self.region_ids,
            modes.mode_ids,
            self.plan_id,
        )


class CavityDipoleCouplingResult(StrictModule):
    angular_couplings: Array
    interaction_energies: Array
    finite: Array
    successful: Array
    angular_coupling_unit: str = eqx.field(static=True)
    interaction_energy_unit: str = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class CavityDipoleCouplingPlan(StrictModule, NonTrainableState):
    """Evaluate SI dipole coupling ``g = -d·E_zpf / hbar`` at one emitter."""

    electric_field_evaluation: Array
    dipole_moment_coulomb_meter: Array
    hbar_joule_second: Array
    emitter_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_field_evaluation: ArrayLike,
        dipole_moment_coulomb_meter: ArrayLike,
        /,
        *,
        emitter_id: str,
        hbar_joule_second: ArrayLike = 1.054571817e-34,
    ):
        evaluation = np.asarray(electric_field_evaluation)
        dipole = np.asarray(dipole_moment_coulomb_meter)
        hbar = np.asarray(hbar_joule_second, dtype=np.float64)
        if evaluation.ndim != 2 or evaluation.shape[0] not in (1, 2, 3):
            raise ValueError(
                "electric_field_evaluation must map coefficients to 1-3 components."
            )
        if dipole.shape != (evaluation.shape[0],):
            raise ValueError("dipole moment must match evaluated field components.")
        if not np.isfinite(evaluation).all() or not np.isfinite(dipole).all():
            raise ValueError("Dipole coupling data must be finite.")
        if hbar.shape != () or not np.isfinite(hbar) or float(hbar) <= 0.0:
            raise ValueError("hbar_joule_second must be positive finite scalar.")
        identifier = str(emitter_id).strip()
        if not identifier:
            raise ValueError("emitter_id must be non-empty.")
        self.electric_field_evaluation = jnp.asarray(evaluation)
        self.dipole_moment_coulomb_meter = jnp.asarray(dipole)
        self.hbar_joule_second = jnp.asarray(hbar)
        self.emitter_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "si-cavity-dipole-coupling",
                "evaluation": array_tree_fingerprint(evaluation),
                "dipole_coulomb_meter": array_tree_fingerprint(dipole),
                "hbar_joule_second": array_tree_fingerprint(hbar),
                "emitter": identifier,
            }
        )

    def evaluate(
        self, modes: NormalizedMaxwellEigenmodes, /
    ) -> CavityDipoleCouplingResult:
        if not isinstance(modes, NormalizedMaxwellEigenmodes):
            raise TypeError("modes must be NormalizedMaxwellEigenmodes.")
        if modes.electric_modes.shape[1] != self.electric_field_evaluation.shape[1]:
            raise ValueError("Field evaluation and electric mode coordinates disagree.")
        local_fields = ein.contract(
            "ci,mi->mc", self.electric_field_evaluation, modes.electric_modes
        )
        interaction = -ein.contract(
            "c,mc->m", self.dipole_moment_coulomb_meter, local_fields
        )
        couplings = interaction / self.hbar_joule_second
        finite = jnp.isfinite(couplings)
        successful = finite & modes.evidence.successful
        return CavityDipoleCouplingResult(
            couplings,
            interaction,
            finite,
            successful,
            "rad s^-1",
            "J",
            modes.mode_ids,
            self.plan_id,
        )


class PurcellLoweringResult(StrictModule):
    induced_decay_rate: Array
    total_decay_rate: Array
    purcell_factor: Array
    lamb_shift_angular_frequency: Array
    effective_jump_operators: Array
    finite: Array
    successful: Array
    rate_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PurcellLoweringPlan(StrictModule, NonTrainableState):
    """Bad-cavity adiabatic lowering with detuning-resolved decay and Lamb shift."""

    cavity_angular_frequency: Array
    emitter_angular_frequency: Array
    cavity_energy_decay_rate: Array
    bare_emitter_decay_rate: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cavity_angular_frequency: ArrayLike,
        emitter_angular_frequency: ArrayLike,
        cavity_energy_decay_rate: ArrayLike,
        bare_emitter_decay_rate: ArrayLike,
        /,
    ):
        values = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                cavity_angular_frequency,
                emitter_angular_frequency,
                cavity_energy_decay_rate,
                bare_emitter_decay_rate,
            )
        )
        if any(value.shape != () or not np.isfinite(value) for value in values):
            raise ValueError("Purcell frequencies and rates must be finite scalars.")
        cavity, emitter, linewidth, decay = (float(value) for value in values)
        if cavity <= 0.0 or emitter <= 0.0 or linewidth <= 0.0 or decay < 0.0:
            raise ValueError("Purcell frequencies/rates must be physically non-negative.")
        self.cavity_angular_frequency = jnp.asarray(cavity)
        self.emitter_angular_frequency = jnp.asarray(emitter)
        self.cavity_energy_decay_rate = jnp.asarray(linewidth)
        self.bare_emitter_decay_rate = jnp.asarray(decay)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "purcell-bad-cavity-lowering",
                "cavity_angular_frequency": cavity,
                "emitter_angular_frequency": emitter,
                "cavity_energy_decay_rate": linewidth,
                "bare_emitter_decay_rate": decay,
            }
        )

    def lower(self, angular_coupling: ArrayLike, /) -> PurcellLoweringResult:
        coupling = jnp.asarray(angular_coupling)
        detuning = self.emitter_angular_frequency - self.cavity_angular_frequency
        denominator = detuning * detuning + 0.25 * self.cavity_energy_decay_rate**2
        strength = jnp.abs(coupling) ** 2
        induced = strength * self.cavity_energy_decay_rate / denominator
        shift = strength * detuning / denominator
        total = self.bare_emitter_decay_rate + induced
        factor = jnp.where(
            self.bare_emitter_decay_rate > 0.0,
            induced / self.bare_emitter_decay_rate,
            jnp.inf,
        )
        lowering = jnp.asarray(((0.0, 1.0), (0.0, 0.0)), dtype=jnp.complex128)
        jumps = jnp.sqrt(total)[..., None, None, None] * lowering[None, ...]
        finite = jnp.isfinite(induced) & jnp.isfinite(total) & jnp.isfinite(shift)
        successful = finite & (total >= 0.0)
        return PurcellLoweringResult(
            induced,
            total,
            factor,
            shift,
            jumps,
            finite,
            successful,
            "rad s^-1",
            self.plan_id,
        )


class ZeroCavityDrive(StrictModule, NonTrainableState):
    drive_id: str = eqx.field(static=True)

    def __init__(self):
        self.drive_id = canonical_fingerprint({"kind": "zero-cavity-drive"})

    def __call__(self, time: ArrayLike, /) -> Array:
        return jnp.zeros_like(jnp.asarray(time), dtype=jnp.complex128)


class MaxwellBlochState(StrictModule):
    cavity_amplitude: Array
    bloch_u: Array
    bloch_v: Array
    inversion: Array

    def __init__(
        self,
        cavity_amplitude: ArrayLike,
        bloch_u: ArrayLike,
        bloch_v: ArrayLike,
        inversion: ArrayLike,
        /,
    ):
        amplitude = jnp.asarray(cavity_amplitude, dtype=jnp.complex128)
        u = jnp.asarray(bloch_u)
        v = jnp.asarray(bloch_v)
        w = jnp.asarray(inversion)
        if amplitude.shape != () or u.shape != () or v.shape != () or w.shape != ():
            raise ValueError("Single-mode Maxwell-Bloch coordinates must be scalars.")
        if any(jnp.iscomplexobj(value) for value in (u, v, w)):
            raise TypeError("Bloch coordinates must be real.")
        self.cavity_amplitude = amplitude
        self.bloch_u = u
        self.bloch_v = v
        self.inversion = w


class MaxwellBlochEvidence(StrictModule):
    bloch_radius_squared: Array
    finite: Array
    physical: Array
    successful: Array


class MaxwellBlochPlan(StrictModule, NonTrainableState):
    """Single-mode driven dissipative Maxwell-Bloch coupling."""

    cavity_detuning: Array
    emitter_detuning: Array
    angular_coupling: Array
    cavity_energy_decay_rate: Array
    transverse_decay_rate: Array
    longitudinal_decay_rate: Array
    equilibrium_inversion: Array
    drive: Any
    drive_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        cavity_detuning: ArrayLike,
        emitter_detuning: ArrayLike,
        angular_coupling: ArrayLike,
        cavity_energy_decay_rate: ArrayLike,
        transverse_decay_rate: ArrayLike,
        longitudinal_decay_rate: ArrayLike,
        equilibrium_inversion: ArrayLike = -1.0,
        drive: Any | None = None,
        drive_id: str | None = None,
    ):
        scalars = tuple(
            np.asarray(value)
            for value in (
                cavity_detuning,
                emitter_detuning,
                angular_coupling,
                cavity_energy_decay_rate,
                transverse_decay_rate,
                longitudinal_decay_rate,
                equilibrium_inversion,
            )
        )
        if any(value.shape != () or not np.isfinite(value) for value in scalars):
            raise ValueError("Maxwell-Bloch parameters must be finite scalars.")
        if not all(np.isrealobj(scalars[index]) for index in (0, 1, 2, 3, 4, 5, 6)):
            raise TypeError(
                "Detunings, coupling, decay rates, and inversion must be real."
            )
        if any(float(value) < 0.0 for value in scalars[3:6]):
            raise ValueError("Maxwell-Bloch decay rates must be non-negative.")
        if abs(float(scalars[6])) > 1.0:
            raise ValueError("equilibrium_inversion must lie in [-1, 1].")
        if drive is None:
            source = ZeroCavityDrive()
            if drive_id is not None:
                raise ValueError("The zero drive has a canonical ID.")
            source_id = source.drive_id
        else:
            if not callable(drive):
                raise TypeError("drive must be callable or None.")
            if drive_id is None or not isinstance(drive_id, str) or not drive_id.strip():
                raise ValueError("Opaque cavity drives require drive_id.")
            source = drive
            source_id = drive_id.strip()
        self.cavity_detuning = jnp.asarray(scalars[0])
        self.emitter_detuning = jnp.asarray(scalars[1])
        self.angular_coupling = jnp.asarray(scalars[2])
        self.cavity_energy_decay_rate = jnp.asarray(scalars[3])
        self.transverse_decay_rate = jnp.asarray(scalars[4])
        self.longitudinal_decay_rate = jnp.asarray(scalars[5])
        self.equilibrium_inversion = jnp.asarray(scalars[6])
        self.drive = source
        self.drive_id = source_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "single-mode-maxwell-bloch",
                "parameters": [array_tree_fingerprint(value) for value in scalars],
                "drive": source_id,
            }
        )

    def evidence(self, state: MaxwellBlochState, /) -> MaxwellBlochEvidence:
        if not isinstance(state, MaxwellBlochState):
            raise TypeError("state must be MaxwellBlochState.")
        radius_squared = (
            state.bloch_u * state.bloch_u
            + state.bloch_v * state.bloch_v
            + state.inversion * state.inversion
        )
        finite = jnp.isfinite(state.cavity_amplitude) & jnp.isfinite(radius_squared)
        physical = radius_squared <= 1.0 + 1e-10
        return MaxwellBlochEvidence(
            radius_squared,
            finite,
            physical,
            finite & physical,
        )

    def prepare(
        self,
        initial_state: MaxwellBlochState,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> PreparedMaxwellBloch:
        if not isinstance(initial_state, MaxwellBlochState):
            raise TypeError("initial_state must be MaxwellBlochState.")
        if not bool(self.evidence(initial_state).successful):
            raise ValueError("Initial Bloch vector must lie in the physical unit ball.")
        state_coordinates = _cavity_state_coordinates(initial_state)
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-bloch",
                "plan": self.plan_id,
                "initial": array_tree_fingerprint(initial_state),
                "t0": array_tree_fingerprint(np.asarray(t0)),
                "t1": array_tree_fingerprint(np.asarray(t1)),
            }
        )
        problem = DifferentialProblem(
            MaxwellBlochVectorField(self),
            initial_state,
            t0=t0,
            t1=t1,
            problem_id=canonical_fingerprint(
                {"kind": "maxwell-bloch-problem", "prepared": prepared_id}
            ),
        )
        return PreparedMaxwellBloch(
            self, initial_state, problem, state_coordinates, prepared_id
        )


class MaxwellBlochVectorField(StrictModule):
    plan: MaxwellBlochPlan

    def __call__(
        self, time: Array, state: MaxwellBlochState, args: Any, /
    ) -> MaxwellBlochState:
        del args
        polarization = 0.5 * (state.bloch_u - 1j * state.bloch_v)
        amplitude_rate = (
            -(0.5 * self.plan.cavity_energy_decay_rate + 1j * self.plan.cavity_detuning)
            * state.cavity_amplitude
            - 1j * self.plan.angular_coupling * polarization
            + self.plan.drive(time)
        )
        rabi_frequency = 2.0 * jnp.real(
            self.plan.angular_coupling * state.cavity_amplitude
        )
        u_rate = (
            -self.plan.transverse_decay_rate * state.bloch_u
            + self.plan.emitter_detuning * state.bloch_v
        )
        v_rate = (
            -self.plan.transverse_decay_rate * state.bloch_v
            - self.plan.emitter_detuning * state.bloch_u
            + rabi_frequency * state.inversion
        )
        inversion_rate = (
            -self.plan.longitudinal_decay_rate
            * (state.inversion - self.plan.equilibrium_inversion)
            - rabi_frequency * state.bloch_v
        )
        return MaxwellBlochState(amplitude_rate, u_rate, v_rate, inversion_rate)


class PreparedMaxwellBloch(StrictModule, NonTrainableState):
    plan: MaxwellBlochPlan
    initial_state: MaxwellBlochState
    problem: DifferentialProblem
    state_coordinates: PreparedRealCoordinateTree
    prepared_id: str = eqx.field(static=True)


class MaxwellLindbladState(StrictModule):
    cavity_amplitude: Array
    density_matrix: Array

    def __init__(self, cavity_amplitude: ArrayLike, density_matrix: ArrayLike, /):
        amplitude = jnp.asarray(cavity_amplitude, dtype=jnp.complex128)
        density = jnp.asarray(density_matrix, dtype=jnp.complex128)
        if amplitude.shape != ():
            raise ValueError("Single-mode cavity amplitude must be scalar.")
        if density.ndim != 2 or density.shape[0] != density.shape[1]:
            raise ValueError("Lindblad density_matrix must be square.")
        self.cavity_amplitude = amplitude
        self.density_matrix = density


class MaxwellLindbladEvidence(StrictModule):
    trace_residual: Array
    hermiticity_residual: Array
    minimum_eigenvalue: Array
    finite: Array
    successful: Array


class MaxwellLindbladPlan(StrictModule, NonTrainableState):
    """Classical cavity amplitude coupled self-consistently to a finite Lindblad system."""

    bare_hamiltonian_angular_frequency: Array
    transition_operator: Array
    jump_operators: Array
    angular_coupling: Array
    cavity_detuning: Array
    cavity_energy_decay_rate: Array
    drive: Any
    dimension: int = eqx.field(static=True)
    drive_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bare_hamiltonian_angular_frequency: ArrayLike,
        transition_operator: ArrayLike,
        jump_operators: ArrayLike,
        /,
        *,
        angular_coupling: ArrayLike,
        cavity_detuning: ArrayLike,
        cavity_energy_decay_rate: ArrayLike,
        drive: Any | None = None,
        drive_id: str | None = None,
        maximum_dimension: int = 1024,
    ):
        hamiltonian = np.asarray(bare_hamiltonian_angular_frequency, dtype=np.complex128)
        transition = np.asarray(transition_operator, dtype=np.complex128)
        jumps = np.asarray(jump_operators, dtype=np.complex128)
        if (
            hamiltonian.ndim != 2
            or hamiltonian.shape[0] != hamiltonian.shape[1]
            or hamiltonian.shape[0] < 1
        ):
            raise ValueError("Bare Lindblad Hamiltonian must be non-empty square.")
        dimension = hamiltonian.shape[0]
        if dimension > int(maximum_dimension):
            raise MemoryError("Maxwell-Lindblad system exceeds maximum_dimension.")
        if (
            transition.shape != (dimension, dimension)
            or jumps.ndim != 3
            or jumps.shape[1:] != (dimension, dimension)
        ):
            raise ValueError("Transition and jump operators must match the Hamiltonian.")
        if (
            not np.isfinite(hamiltonian).all()
            or not np.isfinite(transition).all()
            or not np.isfinite(jumps).all()
        ):
            raise ValueError("Maxwell-Lindblad operators must be finite.")
        if np.max(np.abs(hamiltonian - hamiltonian.conj().T)) > 1e-10:
            raise ValueError("Bare Lindblad Hamiltonian must be Hermitian.")
        coupling = np.asarray(angular_coupling)
        detuning = np.asarray(cavity_detuning)
        decay = np.asarray(cavity_energy_decay_rate, dtype=np.float64)
        if any(
            value.shape != () or not np.isfinite(value)
            for value in (coupling, detuning, decay)
        ):
            raise ValueError("Maxwell-Lindblad scalar parameters must be finite.")
        if not np.isrealobj(detuning):
            raise TypeError("cavity_detuning must be real.")
        if float(decay) < 0.0:
            raise ValueError("cavity_energy_decay_rate must be non-negative.")
        if drive is None:
            source = ZeroCavityDrive()
            if drive_id is not None:
                raise ValueError("The zero drive has a canonical ID.")
            source_id = source.drive_id
        else:
            if not callable(drive):
                raise TypeError("drive must be callable or None.")
            if drive_id is None or not isinstance(drive_id, str) or not drive_id.strip():
                raise ValueError("Opaque cavity drives require drive_id.")
            source = drive
            source_id = drive_id.strip()
        self.bare_hamiltonian_angular_frequency = jnp.asarray(hamiltonian)
        self.transition_operator = jnp.asarray(transition)
        self.jump_operators = jnp.asarray(jumps)
        self.angular_coupling = jnp.asarray(coupling)
        self.cavity_detuning = jnp.asarray(detuning)
        self.cavity_energy_decay_rate = jnp.asarray(decay)
        self.drive = source
        self.dimension = dimension
        self.drive_id = source_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "single-mode-maxwell-lindblad",
                "hamiltonian": array_tree_fingerprint(hamiltonian),
                "transition": array_tree_fingerprint(transition),
                "jumps": array_tree_fingerprint(jumps),
                "coupling": array_tree_fingerprint(coupling),
                "detuning": array_tree_fingerprint(detuning),
                "decay": array_tree_fingerprint(decay),
                "drive": source_id,
            }
        )

    def evidence(self, state: MaxwellLindbladState, /) -> MaxwellLindbladEvidence:
        if not isinstance(state, MaxwellLindbladState) or state.density_matrix.shape != (
            self.dimension,
            self.dimension,
        ):
            raise ValueError("Maxwell-Lindblad state does not match the plan.")
        density = state.density_matrix
        trace_residual = jnp.abs(jnp.trace(density) - 1.0)
        hermiticity = jnp.max(jnp.abs(density - jnp.conj(density.T)))
        minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (density + jnp.conj(density.T))))
        finite = jnp.all(jnp.isfinite(density)) & jnp.isfinite(state.cavity_amplitude)
        successful = (
            finite & (trace_residual <= 1e-8) & (hermiticity <= 1e-8) & (minimum >= -1e-8)
        )
        return MaxwellLindbladEvidence(
            trace_residual, hermiticity, minimum, finite, successful
        )

    def prepare(
        self,
        initial_state: MaxwellLindbladState,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> PreparedMaxwellLindblad:
        evidence = self.evidence(initial_state)
        if not bool(evidence.successful):
            raise ValueError("Initial Maxwell-Lindblad density must be physical.")
        state_coordinates = _cavity_state_coordinates(initial_state)
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-lindblad",
                "plan": self.plan_id,
                "initial": array_tree_fingerprint(initial_state),
                "t0": array_tree_fingerprint(np.asarray(t0)),
                "t1": array_tree_fingerprint(np.asarray(t1)),
            }
        )
        problem = DifferentialProblem(
            MaxwellLindbladVectorField(self),
            initial_state,
            t0=t0,
            t1=t1,
            problem_id=canonical_fingerprint(
                {"kind": "maxwell-lindblad-problem", "prepared": prepared_id}
            ),
        )
        return PreparedMaxwellLindblad(
            self, initial_state, problem, state_coordinates, prepared_id
        )


class MaxwellLindbladVectorField(StrictModule):
    plan: MaxwellLindbladPlan

    def __call__(
        self, time: Array, state: MaxwellLindbladState, args: Any, /
    ) -> MaxwellLindbladState:
        del args
        transition_adjoint = jnp.conj(self.plan.transition_operator.T)
        hamiltonian = self.plan.bare_hamiltonian_angular_frequency + (
            self.plan.angular_coupling * state.cavity_amplitude * transition_adjoint
            + jnp.conj(self.plan.angular_coupling * state.cavity_amplitude)
            * self.plan.transition_operator
        )
        density = state.density_matrix
        density_rate = -1j * (hamiltonian @ density - density @ hamiltonian)
        for jump in self.plan.jump_operators:
            adjoint = jnp.conj(jump.T)
            product = adjoint @ jump
            density_rate = (
                density_rate
                + jump @ density @ adjoint
                - 0.5 * (product @ density + density @ product)
            )
        polarization = jnp.trace(self.plan.transition_operator @ density)
        amplitude_rate = (
            -(0.5 * self.plan.cavity_energy_decay_rate + 1j * self.plan.cavity_detuning)
            * state.cavity_amplitude
            - 1j * jnp.conj(self.plan.angular_coupling) * polarization
            + self.plan.drive(time)
        )
        return MaxwellLindbladState(amplitude_rate, density_rate)


class PreparedMaxwellLindblad(StrictModule, NonTrainableState):
    plan: MaxwellLindbladPlan
    initial_state: MaxwellLindbladState
    problem: DifferentialProblem
    state_coordinates: PreparedRealCoordinateTree
    prepared_id: str = eqx.field(static=True)


class HcurlCapabilityStatus(IntEnum):
    """Truthful disposition of an adaptive H(curl) request."""

    SUPPORTED = 0
    UNSUPPORTED_POLYNOMIAL_ORDER = 1
    ADAPTATION_TRANSACTION_REQUIRED = 2
    HCURL_TRANSFER_REQUIRED = 3
    RESOURCE_LIMIT_EXCEEDED = 4
    UNSUPPORTED_MESH = 5


class HcurlCapabilityEvidence(StrictModule, NonTrainableState):
    status: int = eqx.field(static=True)
    assembly_supported: bool = eqx.field(static=True)
    adaptation_supported: bool = eqx.field(static=True)
    requested_polynomial_order: int = eqx.field(static=True)
    native_polynomial_order: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)


class PreparedAdaptiveHcurlCapability(StrictModule, NonTrainableState):
    """Native lowest-order Nedelec space and explicit AMR transaction binding."""

    space: TetrahedralNedelecSpace
    transaction: FiniteElementTopologyTransaction | None
    evidence: HcurlCapabilityEvidence
    prepared_id: str = eqx.field(static=True)


class AdaptiveHcurlCapabilityPlan(StrictModule, NonTrainableState):
    """Capability gate over the existing tetrahedral FEM/AMR substrate.

    Phydrax currently owns a lowest-order first-family tetrahedral Nedelec kernel.
    Higher order is reported unsupported rather than relabeled. Adaptive transfer is
    admitted only when a finite-element topology transaction supplies an explicit
    edge-field transfer; its automatic P1 vertex transfer is not claimed for H(curl).
    """

    mesh: CellMesh
    transaction: FiniteElementTopologyTransaction | None
    evidence: HcurlCapabilityEvidence
    require_adaptation: bool = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        requested_polynomial_order: int = 1,
        require_adaptation: bool = False,
        transaction: FiniteElementTopologyTransaction | None = None,
        maximum_edges: int = 2**18,
        maximum_cells: int = 2**17,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        order = int(requested_polynomial_order)
        if order < 1:
            raise ValueError("requested_polynomial_order must be positive.")
        if not isinstance(require_adaptation, bool):
            raise TypeError("require_adaptation must be bool.")
        if transaction is not None and not isinstance(
            transaction, FiniteElementTopologyTransaction
        ):
            raise TypeError(
                "transaction must be FiniteElementTopologyTransaction or None."
            )
        edge_limit = int(maximum_edges)
        cell_limit = int(maximum_cells)
        if edge_limit < 1 or cell_limit < 1:
            raise ValueError("H(curl) resource limits must be positive.")
        edge_count = mesh.entity_set(1).entity_ids.size
        cell_count = sum(block.cell_count for block in mesh.blocks)
        mesh_supported = (
            mesh.topological_dimension == 3
            and mesh.ambient_dimension == 3
            and all(block.cell_kind == "tetrahedron" for block in mesh.blocks)
        )
        if not mesh_supported:
            status = HcurlCapabilityStatus.UNSUPPORTED_MESH
            assembly = False
            adaptation = False
            reason = "native H(curl) requires a full-dimensional tetrahedral CellMesh"
        elif edge_count > edge_limit or cell_count > cell_limit:
            status = HcurlCapabilityStatus.RESOURCE_LIMIT_EXCEEDED
            assembly = False
            adaptation = False
            reason = "tetrahedral H(curl) mesh exceeds the declared fixed resources"
        elif order != 1:
            status = HcurlCapabilityStatus.UNSUPPORTED_POLYNOMIAL_ORDER
            assembly = False
            adaptation = False
            reason = "native tetrahedral H(curl) is lowest-order first-family only"
        elif require_adaptation and transaction is None:
            status = HcurlCapabilityStatus.ADAPTATION_TRANSACTION_REQUIRED
            assembly = True
            adaptation = False
            reason = "adaptive H(curl) requires an explicit finite-element topology transaction"
        elif require_adaptation and transaction.field_transfer is None:
            status = HcurlCapabilityStatus.HCURL_TRANSFER_REQUIRED
            assembly = True
            adaptation = False
            reason = "automatic P1 vertex transfer is not an H(curl) edge transfer"
        else:
            status = HcurlCapabilityStatus.SUPPORTED
            assembly = True
            adaptation = require_adaptation
            reason = "native lowest-order tetrahedral Nedelec assembly admitted"
        capability_id = canonical_fingerprint(
            {
                "kind": "adaptive-hcurl-capability",
                "mesh": mesh.mesh_id,
                "requested_polynomial_order": order,
                "native_polynomial_order": 1,
                "require_adaptation": require_adaptation,
                "transaction": None
                if transaction is None
                else transaction.transaction_id,
                "edge_count": edge_count,
                "cell_count": cell_count,
                "maximum_edges": edge_limit,
                "maximum_cells": cell_limit,
                "status": int(status),
            }
        )
        evidence = HcurlCapabilityEvidence(
            int(status),
            assembly,
            adaptation,
            order,
            1,
            edge_count,
            cell_count,
            edge_limit,
            cell_limit,
            reason,
            capability_id,
        )
        self.mesh = mesh
        self.transaction = transaction
        self.evidence = evidence
        self.require_adaptation = require_adaptation
        self.maximum_edges = edge_limit
        self.maximum_cells = cell_limit
        self.plan_id = canonical_fingerprint(
            {"kind": "adaptive-hcurl-capability-plan", "capability": capability_id}
        )

    def prepare(self, /) -> PreparedAdaptiveHcurlCapability:
        if self.evidence.status != int(HcurlCapabilityStatus.SUPPORTED):
            raise NotImplementedError(self.evidence.reason)
        space = TetrahedralNedelecSpace(self.mesh)
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-adaptive-hcurl-capability",
                "plan": self.plan_id,
                "space": space.space_id,
                "transaction": None
                if self.transaction is None
                else self.transaction.transaction_id,
            }
        )
        return PreparedAdaptiveHcurlCapability(
            space, self.transaction, self.evidence, prepared_id
        )


__all__ = [
    "AdaptiveHcurlCapabilityPlan",
    "CavityDipoleCouplingPlan",
    "CavityDipoleCouplingResult",
    "CavityModeStatus",
    "CavityParticipationPlan",
    "CavityParticipationResult",
    "HcurlCapabilityEvidence",
    "HcurlCapabilityStatus",
    "MaxwellBlochEvidence",
    "MaxwellBlochPlan",
    "MaxwellBlochState",
    "MaxwellBlochVectorField",
    "MaxwellEigenmodeNormalizationEvidence",
    "MaxwellEigenmodeNormalizationPlan",
    "MaxwellLindbladEvidence",
    "MaxwellLindbladPlan",
    "MaxwellLindbladState",
    "MaxwellLindbladVectorField",
    "NormalizedMaxwellEigenmodes",
    "PreparedAdaptiveHcurlCapability",
    "PreparedMaxwellEigenmodeNormalization",
    "PreparedMaxwellBloch",
    "PreparedMaxwellLindblad",
    "PurcellLoweringPlan",
    "PurcellLoweringResult",
    "ZeroCavityDrive",
]
