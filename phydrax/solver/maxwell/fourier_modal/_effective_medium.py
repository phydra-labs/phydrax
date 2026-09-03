#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from fractions import Fraction
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....circuit import ModalWaveReference
from ....lifecycle import NumericRevision
from ._contracts import (
    ContinuousFourierModalLayer,
    FourierModalLayer,
    FourierModalSourcePlane,
)
from ._loss import FourierModalLossConvergenceEvidence, FourierModalLossEvidence
from ._numeric_revision import require_fourier_modal_numeric_revision
from ._runtime import _canonical_material_samples, PreparedFourierModalMaxwell
from ._scattering import HomogeneousPortModes


RetrievalAnchor: TypeAlias = Literal["low-frequency", "known-index", "cross-thickness"]


class EquivalentSlabRetrievalStatus(IntEnum):
    VALID = 0
    INELIGIBLE = 1
    AMBIGUOUS = 2


class LocalIsotropicQualificationStatus(IntEnum):
    QUALIFIED = 0
    INELIGIBLE = 1
    AMBIGUOUS = 2


class MaxwellModalSweep(StrictModule):
    """Host-revision-bound Maxwell modal data for one slab and polarization."""

    angular_frequencies: Array
    bloch_wavevectors: Array
    polar_angles: Array
    left_longitudinal_wavevectors: Array
    right_longitudinal_wavevectors: Array
    left_modal_admittances: Array
    right_modal_admittances: Array
    left_reflection: Array
    right_reflection: Array
    left_to_right_transmission: Array
    right_to_left_transmission: Array
    deembedded_left_reflection: Array
    deembedded_right_reflection: Array
    deembedded_left_to_right_transmission: Array
    deembedded_right_to_left_transmission: Array
    cross_polarization_conversion: Array
    additional_propagating_orders: Array
    grazing: Array
    symmetric_termination: Array
    physical_finite_layers: Array
    finite: Array
    thickness: Array
    left_references: tuple[ModalWaveReference, ...]
    right_references: tuple[ModalWaveReference, ...]
    polarization: str = eqx.field(static=True)
    harmonic_mode_id: str = eqx.field(static=True)
    preparation_ids: tuple[str, ...] = eqx.field(static=True)
    physical_state_digests: tuple[str, ...] = eqx.field(static=True)
    numeric_revision_ids: tuple[str, ...] = eqx.field(static=True)
    harmonic_discretization_ids: tuple[str, ...] = eqx.field(static=True)
    physical_stack_id: str = eqx.field(static=True)
    physical_stack_digest: str = eqx.field(static=True)
    sweep_id: str = eqx.field(static=True)


class EquivalentSlabRetrievalPlan(StrictModule, NonTrainableState):
    """Bounded propagation/root branches and an explicit branch anchor."""

    minimum_branch: int = eqx.field(static=True)
    maximum_branch: int = eqx.field(static=True)
    anchor: RetrievalAnchor = eqx.field(static=True)
    anchor_refractive_index: Array | None
    anchor_tolerance: float = eqx.field(static=True)
    reconstruction_tolerance: float = eqx.field(static=True)
    cross_polarization_tolerance: float = eqx.field(static=True)
    normal_incidence_tolerance: float = eqx.field(static=True)
    grazing_tolerance: float = eqx.field(static=True)
    transmission_tolerance: float = eqx.field(static=True)
    passive_claim: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        branch_window: tuple[int, int],
        /,
        *,
        anchor: RetrievalAnchor,
        anchor_refractive_index: ArrayLike | None = None,
        anchor_tolerance: float = 1.0e-4,
        reconstruction_tolerance: float = 1.0e-7,
        cross_polarization_tolerance: float = 1.0e-8,
        normal_incidence_tolerance: float = 1.0e-10,
        grazing_tolerance: float = 1.0e-10,
        transmission_tolerance: float = 1.0e-12,
        passive_claim: bool = False,
    ):
        minimum, maximum = (int(value) for value in branch_window)
        if minimum > maximum:
            raise ValueError("branch_window must be ordered.")
        if anchor not in ("low-frequency", "known-index", "cross-thickness"):
            raise ValueError("Unknown equivalent-slab branch anchor.")
        anchor_value = (
            None
            if anchor_refractive_index is None
            else jnp.asarray(anchor_refractive_index)
        )
        if anchor in ("known-index", "cross-thickness") and anchor_value is None:
            raise ValueError(f"{anchor} requires anchor_refractive_index.")
        tolerances = (
            anchor_tolerance,
            reconstruction_tolerance,
            cross_polarization_tolerance,
            normal_incidence_tolerance,
            grazing_tolerance,
            transmission_tolerance,
        )
        if any(float(value) < 0.0 for value in tolerances):
            raise ValueError("Equivalent-slab tolerances must be non-negative.")
        self.minimum_branch = minimum
        self.maximum_branch = maximum
        self.anchor = anchor
        self.anchor_refractive_index = anchor_value
        self.anchor_tolerance = float(anchor_tolerance)
        self.reconstruction_tolerance = float(reconstruction_tolerance)
        self.cross_polarization_tolerance = float(cross_polarization_tolerance)
        self.normal_incidence_tolerance = float(normal_incidence_tolerance)
        self.grazing_tolerance = float(grazing_tolerance)
        self.transmission_tolerance = float(transmission_tolerance)
        self.passive_claim = bool(passive_claim)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "equivalent-slab-retrieval-plan",
                "branch_window": [minimum, maximum],
                "anchor": anchor,
                "anchor_tolerance": self.anchor_tolerance,
                "reconstruction_tolerance": self.reconstruction_tolerance,
                "cross_polarization_tolerance": self.cross_polarization_tolerance,
                "normal_incidence_tolerance": self.normal_incidence_tolerance,
                "grazing_tolerance": self.grazing_tolerance,
                "transmission_tolerance": self.transmission_tolerance,
                "passive_claim": self.passive_claim,
            }
        )


class EquivalentSlabRetrieval(StrictModule):
    """All bounded equivalent-slab candidates plus one explicit modal chart."""

    sweep: MaxwellModalSweep
    candidate_branch_numbers: Array
    candidate_root_signs: Array
    candidate_impedance_signs: Array
    candidate_propagation_factors: Array
    candidate_wave_numbers: Array
    candidate_refractive_indices: Array
    candidate_relative_impedances: Array
    candidate_effective_impedances: Array
    candidate_permittivities: Array
    candidate_permeabilities: Array
    candidate_reconstruction_residuals: Array
    candidate_admissible: Array
    modal_chart_candidate_indices: Array
    wave_number: Array
    refractive_index: Array
    relative_impedance: Array
    effective_impedance: Array
    permittivity: Array
    permeability: Array
    reconstruction_residual: Array
    branch_margin: Array
    finite_band_kramers_kronig_residual: Array
    stable_fit_residual: Array
    eligible: Array
    ambiguous: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    retrieval_id: str = eqx.field(static=True)


class LocalIsotropicQualificationPolicy(StrictModule, NonTrainableState):
    """Fail-closed invariance, prediction, passivity, and evidence gates."""

    parameter_relative_tolerance: float = eqx.field(static=True)
    parameter_absolute_tolerance: float = eqx.field(static=True)
    reconstructed_scattering_tolerance: float = eqx.field(static=True)
    minimum_branch_margin: float = eqx.field(static=True)
    angle_prediction_tolerance: float = eqx.field(static=True)
    commensurate_denominator_limit: int = eqx.field(static=True)
    commensurate_tolerance: float = eqx.field(static=True)
    passive_claim: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        parameter_relative_tolerance: float = 1.0e-3,
        parameter_absolute_tolerance: float = 1.0e-6,
        reconstructed_scattering_tolerance: float = 1.0e-6,
        minimum_branch_margin: float = 1.0e-3,
        angle_prediction_tolerance: float = 1.0e-3,
        commensurate_denominator_limit: int = 16,
        commensurate_tolerance: float = 1.0e-8,
        passive_claim: bool = True,
    ):
        tolerances = (
            parameter_relative_tolerance,
            parameter_absolute_tolerance,
            reconstructed_scattering_tolerance,
            minimum_branch_margin,
            angle_prediction_tolerance,
            commensurate_tolerance,
        )
        if any(float(value) < 0.0 for value in tolerances):
            raise ValueError(
                "Local-isotropic qualification tolerances must be non-negative."
            )
        denominator = int(commensurate_denominator_limit)
        if denominator < 1:
            raise ValueError("commensurate_denominator_limit must be positive.")
        self.parameter_relative_tolerance = float(parameter_relative_tolerance)
        self.parameter_absolute_tolerance = float(parameter_absolute_tolerance)
        self.reconstructed_scattering_tolerance = float(
            reconstructed_scattering_tolerance
        )
        self.minimum_branch_margin = float(minimum_branch_margin)
        self.angle_prediction_tolerance = float(angle_prediction_tolerance)
        self.commensurate_denominator_limit = denominator
        self.commensurate_tolerance = float(commensurate_tolerance)
        self.passive_claim = bool(passive_claim)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "local-isotropic-qualification-policy",
                "parameter_relative_tolerance": self.parameter_relative_tolerance,
                "parameter_absolute_tolerance": self.parameter_absolute_tolerance,
                "reconstructed_scattering_tolerance": self.reconstructed_scattering_tolerance,
                "minimum_branch_margin": self.minimum_branch_margin,
                "angle_prediction_tolerance": self.angle_prediction_tolerance,
                "commensurate_denominator_limit": denominator,
                "commensurate_tolerance": self.commensurate_tolerance,
                "passive_claim": self.passive_claim,
            }
        )


class LocalIsotropicMediumQualification(StrictModule):
    """Typed evidence; never a constructed Maxwell material."""

    representative_permittivity: Array
    representative_permeability: Array
    thickness_invariance_residual: Array
    polarization_invariance_residual: Array
    angle_prediction_residual: Array
    reconstructed_scattering_residual: Array
    minimum_branch_margin: Array
    passive_consistent: Array
    loss_evidence_accepted: Array
    loss_convergence_accepted: Array
    qualified: Array
    status: Array
    reasons: tuple[str, ...] = eqx.field(static=True)
    retrieval_ids: tuple[str, ...] = eqx.field(static=True)
    angle_sweep_ids: tuple[str, ...] = eqx.field(static=True)
    loss_numeric_revision_ids: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)


def _modal_admittance(modes: HomogeneousPortModes, index: int, /) -> Array:
    electric = modes.electric_matrix[:, index]
    magnetic = modes.magnetic_matrix[:, index]
    count = len(modes.mode_ids) // 2
    ex = electric[:count]
    ey = electric[count:]
    hx = magnetic[:count]
    hy = magnetic[count:]
    denominator = jnp.sum(jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2)
    return jnp.sum(jnp.conj(ex) * hy - jnp.conj(ey) * hx) / denominator


def _same_material(left, right, problem, /) -> bool:
    left_values = _canonical_material_samples(left, problem)
    right_values = _canonical_material_samples(right, problem)
    return (
        left.material_role == "physical"
        and right.material_role == "physical"
        and all(
            np.array_equal(np.asarray(first), np.asarray(second), equal_nan=False)
            for first, second in zip(left_values, right_values, strict=True)
        )
    )


def _physical_stack_id(prepared: PreparedFourierModalMaxwell, /) -> str:
    elements = []
    for element in prepared.problem.elements:
        if isinstance(element, FourierModalSourcePlane):
            elements.append({"kind": "source", "source_id": element.source_id})
        elif isinstance(element, FourierModalLayer):
            elements.append(
                {
                    "kind": "layer",
                    "layer_id": element.layer_id,
                    "material_id": element.material.material_id,
                    "material_role": element.material.material_role,
                    "origin_evidence_id": element.material.origin_evidence_id,
                    "factorization_id": element.factorization.plan_id,
                }
            )
        elif isinstance(element, ContinuousFourierModalLayer):
            elements.append(
                {
                    "kind": "continuous",
                    "layer_id": element.layer_id,
                    "factorization_id": element.factorization.plan_id,
                }
            )
    return canonical_fingerprint(
        {"kind": "fourier-modal-physical-stack", "elements": elements}
    )


def prepare_maxwell_modal_sweep(
    prepared_cases: Sequence[PreparedFourierModalMaxwell],
    numeric_revisions: Sequence[NumericRevision],
    /,
    *,
    slab_thickness: ArrayLike,
    harmonic_mode_id: str,
    polarization: Literal["te", "tm"],
) -> MaxwellModalSweep:
    """Retain exact per-frequency modal references and de-embedded Maxwell S data."""
    cases = tuple(prepared_cases)
    revisions = tuple(numeric_revisions)
    if not cases or len(cases) != len(revisions):
        raise ValueError("One numeric revision is required for every prepared frequency.")
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'.")
    mode_id = str(harmonic_mode_id)
    if not mode_id:
        raise ValueError("harmonic_mode_id must be non-empty.")
    thickness = jnp.asarray(slab_thickness)
    if (
        thickness.ndim != 0
        or not bool(np.isfinite(np.asarray(thickness)))
        or float(np.asarray(thickness)) <= 0.0
    ):
        raise ValueError("slab_thickness must be one positive host scalar.")
    if not isinstance(cases[0], PreparedFourierModalMaxwell):
        raise TypeError("prepared_cases must contain PreparedFourierModalMaxwell.")
    reference_case = cases[0]
    angular_frequencies = []
    bloch_wavevectors = []
    polar_angles = []
    left_kz = []
    reference_stack_id = _physical_stack_id(reference_case)
    right_kz = []
    left_admittance = []
    right_admittance = []
    left_reflection = []
    right_reflection = []
    left_to_right = []
    right_to_left = []
    deembedded_left_reflection = []
    deembedded_right_reflection = []
    deembedded_left_to_right = []
    deembedded_right_to_left = []
    cross_conversion = []
    additional_orders = []
    grazing = []
    symmetric = []
    physical_layers = []
    finite = []
    left_references = []
    right_references = []
    preparation_ids = []
    physical_digests = []
    physical_stack_digests = []
    revision_ids = []
    harmonic_ids = []
    selected_mode_id = f"{mode_id}:{polarization}"
    for case, revision in zip(cases, revisions, strict=True):
        if not isinstance(case, PreparedFourierModalMaxwell):
            raise TypeError("prepared_cases must contain PreparedFourierModalMaxwell.")
        require_fourier_modal_numeric_revision(case, revision)
        revision_metadata = dict(revision.metadata)
        case_stack_digest = revision_metadata["physical_stack_digest"]
        if physical_stack_digests and case_stack_digest != physical_stack_digests[0]:
            raise ValueError("A modal sweep requires one content-bound physical stack.")
        physical_stack_digests.append(case_stack_digest)
        if (
            case.problem.harmonics.plan.layout.layout_id
            != reference_case.problem.harmonics.plan.layout.layout_id
            or case.left_modes.mode_ids != reference_case.left_modes.mode_ids
            or case.right_modes.mode_ids != reference_case.right_modes.mode_ids
            or not np.array_equal(
                np.asarray(case.problem.harmonics.primitive_vectors),
                np.asarray(reference_case.problem.harmonics.primitive_vectors),
                equal_nan=False,
            )
        ):
            raise ValueError(
                "A modal sweep requires one harmonic layout and primitive-vector state."
            )
        if not np.allclose(
            np.asarray(case.total_thickness),
            np.asarray(thickness),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "slab_thickness must equal the prepared finite-stack thickness."
            )
        if not isinstance(case.left_modes, HomogeneousPortModes) or not isinstance(
            case.right_modes, HomogeneousPortModes
        ):
            raise TypeError("Equivalent-slab sweeps require homogeneous Maxwell ports.")
        if _physical_stack_id(case) != reference_stack_id:
            raise ValueError("A modal sweep requires one semantic physical stack.")
        if (
            selected_mode_id not in case.left_modes.mode_ids
            or selected_mode_id not in case.right_modes.mode_ids
        ):
            raise ValueError(
                "The selected polarization channel is absent from a port basis."
            )
        left_index = case.left_modes.mode_ids.index(selected_mode_id)
        right_index = case.right_modes.mode_ids.index(selected_mode_id)
        left_mode_kz = case.left_modes.longitudinal_wavevector[left_index // 2]
        right_mode_kz = case.right_modes.longitudinal_wavevector[right_index // 2]
        left_y = _modal_admittance(case.left_modes, left_index)
        right_y = _modal_admittance(case.right_modes, right_index)
        scattering = case.scattering
        interface = case.interface_scattering
        r_left = scattering.s21.matrix[left_index, left_index]
        r_right = scattering.s12.matrix[right_index, right_index]
        t_lr = scattering.s11.matrix[right_index, left_index]
        t_rl = scattering.s22.matrix[left_index, right_index]
        ri_left = interface.s21.matrix[left_index, left_index]
        ri_right = interface.s12.matrix[right_index, right_index]
        ti_lr = interface.s11.matrix[right_index, left_index]
        ti_rl = interface.s22.matrix[left_index, right_index]
        left_propagating = np.flatnonzero(np.asarray(case.left_modes.propagating))
        right_propagating = np.flatnonzero(np.asarray(case.right_modes.propagating))
        left_other = left_propagating[left_propagating != left_index]
        right_other = right_propagating[right_propagating != right_index]
        converted = [jnp.asarray(0.0)]
        if left_other.size:
            converted.extend(
                (
                    jnp.max(jnp.abs(interface.s21.matrix[left_other, left_index])),
                    jnp.max(jnp.abs(interface.s11.matrix[right_other, left_index]))
                    if right_other.size
                    else jnp.asarray(0.0),
                )
            )
        if right_other.size:
            converted.extend(
                (
                    jnp.max(jnp.abs(interface.s12.matrix[right_other, right_index])),
                    jnp.max(jnp.abs(interface.s22.matrix[left_other, right_index]))
                    if left_other.size
                    else jnp.asarray(0.0),
                )
            )
        basis_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-retrieval-basis",
                "preparation_id": case.preparation_id,
                "numeric_revision_id": revision.revision_id,
            }
        )
        left_references.append(
            ModalWaveReference(
                basis_id,
                selected_mode_id,
                polarization=polarization,
                reference_plane=case.problem.superstrate.reference_plane,
            )
        )
        right_references.append(
            ModalWaveReference(
                basis_id,
                selected_mode_id,
                polarization=polarization,
                reference_plane=case.problem.substrate.reference_plane,
            )
        )
        frequency = case.problem.angular_frequency
        transverse = jnp.sqrt(jnp.sum(jnp.abs(case.problem.bloch_wavevector) ** 2))
        angular_frequencies.append(frequency)
        bloch_wavevectors.append(case.problem.bloch_wavevector)
        polar_angles.append(jnp.arctan2(transverse, jnp.abs(jnp.real(left_mode_kz))))
        left_kz.append(left_mode_kz)
        right_kz.append(right_mode_kz)
        left_admittance.append(left_y)
        right_admittance.append(right_y)
        left_reflection.append(r_left)
        right_reflection.append(r_right)
        left_to_right.append(t_lr)
        right_to_left.append(t_rl)
        deembedded_left_reflection.append(ri_left)
        deembedded_right_reflection.append(ri_right)
        deembedded_left_to_right.append(ti_lr)
        deembedded_right_to_left.append(ti_rl)
        cross_conversion.append(jnp.max(jnp.stack(tuple(converted))))
        additional_orders.append(
            (left_propagating.size > 2) or (right_propagating.size > 2)
        )
        grazing.append(
            case.left_modes.grazing[left_index] | case.right_modes.grazing[right_index]
        )
        same_termination = _same_material(
            case.problem.superstrate.material,
            case.problem.substrate.material,
            case.problem,
        ) and bool(
            np.allclose(
                np.asarray(left_y), np.asarray(right_y), rtol=1.0e-10, atol=1.0e-12
            )
        )
        symmetric.append(same_termination)
        physical_layers.append(
            case.problem.layer_count > 0
            and not case.problem.source_ids
            and all(
                isinstance(element, FourierModalLayer)
                and element.material.material_role == "physical"
                for element in case.problem.elements
            )
        )
        finite.append(
            jnp.all(
                jnp.isfinite(jnp.stack((r_left, r_right, t_lr, t_rl, left_y, right_y)))
            )
        )
        preparation_ids.append(case.preparation_id)
        revision_ids.append(revision.revision_id)
        physical_digests.append(revision_metadata["physical_state_digest"])
        harmonic_ids.append(case.problem.harmonics.preparation_id)
    frequency_values = np.asarray(jnp.stack(tuple(angular_frequencies)))
    if (
        np.any(~np.isfinite(frequency_values))
        or np.any(np.imag(frequency_values) != 0.0)
        or np.any(np.diff(np.real(frequency_values)) <= 0.0)
    ):
        raise ValueError("Modal sweep frequencies must be finite, real, and increasing.")
    sweep_id = canonical_fingerprint(
        {
            "kind": "maxwell-modal-sweep",
            "preparation_ids": preparation_ids,
            "numeric_revision_ids": revision_ids,
            "harmonic_discretization_ids": harmonic_ids,
            "physical_state_digests": physical_digests,
            "physical_stack_digest": physical_stack_digests[0],
            "polarization": polarization,
            "harmonic_mode_id": mode_id,
            "slab_thickness": array_tree_fingerprint(thickness),
        }
    )
    return MaxwellModalSweep(
        jnp.stack(tuple(angular_frequencies)),
        jnp.stack(tuple(bloch_wavevectors)),
        jnp.stack(tuple(polar_angles)),
        jnp.stack(tuple(left_kz)),
        jnp.stack(tuple(right_kz)),
        jnp.stack(tuple(left_admittance)),
        jnp.stack(tuple(right_admittance)),
        jnp.stack(tuple(left_reflection)),
        jnp.stack(tuple(right_reflection)),
        jnp.stack(tuple(left_to_right)),
        jnp.stack(tuple(right_to_left)),
        jnp.stack(tuple(deembedded_left_reflection)),
        jnp.stack(tuple(deembedded_right_reflection)),
        jnp.stack(tuple(deembedded_left_to_right)),
        jnp.stack(tuple(deembedded_right_to_left)),
        jnp.stack(tuple(cross_conversion)),
        jnp.asarray(additional_orders, dtype=bool),
        jnp.stack(tuple(grazing)),
        jnp.asarray(symmetric, dtype=bool),
        jnp.asarray(physical_layers, dtype=bool),
        jnp.stack(tuple(finite)),
        thickness,
        tuple(left_references),
        tuple(right_references),
        polarization=polarization,
        harmonic_mode_id=mode_id,
        preparation_ids=tuple(preparation_ids),
        numeric_revision_ids=tuple(revision_ids),
        harmonic_discretization_ids=tuple(harmonic_ids),
        physical_state_digests=tuple(physical_digests),
        physical_stack_id=reference_stack_id,
        physical_stack_digest=physical_stack_digests[0],
        sweep_id=sweep_id,
    )


def _slab_scattering(
    propagation: Array, relative_impedance: Array, /
) -> tuple[Array, Array]:
    interface_reflection = (relative_impedance - 1.0) / (relative_impedance + 1.0)
    denominator = 1.0 - interface_reflection**2 * propagation**2
    reflection = interface_reflection * (1.0 - propagation**2) / denominator
    transmission = propagation * (1.0 - interface_reflection**2) / denominator
    return reflection, transmission


def _finite_band_kk_residual(frequencies: Array, values: Array, /) -> Array:
    count = int(frequencies.shape[0])
    if count < 3:
        return jnp.asarray(jnp.inf, dtype=frequencies.real.dtype)
    omega = jnp.real(frequencies)
    delta = omega[1:] - omega[:-1]
    trapezoid = jnp.concatenate((delta[:1], 0.5 * (delta[:-1] + delta[1:]), delta[-1:]))
    denominator = omega[None, :] ** 2 - omega[:, None] ** 2
    mask = ~jnp.eye(count, dtype=bool)
    kernel = jnp.where(mask, omega[None, :] / denominator, 0.0)
    predicted = (2.0 / jnp.pi) * jnp.sum(
        kernel * trapezoid[None, :] * jnp.imag(values)[None, :], axis=1
    )
    offset = jnp.mean(jnp.real(values) - predicted)
    residual = jnp.real(values) - predicted - offset
    return jnp.sqrt(jnp.sum(residual**2)) / jnp.maximum(
        jnp.sqrt(jnp.sum(jnp.real(values) ** 2)), 1.0
    )


def retrieve_equivalent_slab(
    sweep: MaxwellModalSweep,
    plan: EquivalentSlabRetrievalPlan,
    /,
) -> EquivalentSlabRetrieval:
    """Retrieve bounded Maxwell-native slab candidates without circuit conversion."""
    if not isinstance(sweep, MaxwellModalSweep):
        raise TypeError("sweep must be MaxwellModalSweep.")
    if not isinstance(plan, EquivalentSlabRetrievalPlan):
        raise TypeError("plan must be EquivalentSlabRetrievalPlan.")
    frequencies = sweep.angular_frequencies
    if (
        plan.anchor_refractive_index is not None
        and plan.anchor_refractive_index.shape
        not in (
            (),
            frequencies.shape,
        )
    ):
        raise ValueError("anchor_refractive_index must be scalar or frequency-shaped.")
    if plan.anchor_refractive_index is not None and not np.all(
        np.isfinite(np.asarray(plan.anchor_refractive_index))
    ):
        raise ValueError("anchor_refractive_index must be host-materialized and finite.")
    r_left = sweep.deembedded_left_reflection
    r_right = sweep.deembedded_right_reflection
    t_lr = sweep.deembedded_left_to_right_transmission
    t_rl = sweep.deembedded_right_to_left_transmission
    reflection = 0.5 * (r_left + r_right)
    transmission = 0.5 * (t_lr + t_rl)
    safe_transmission = jnp.where(
        jnp.abs(transmission) > plan.transmission_tolerance,
        transmission,
        1.0 + 0.0j,
    )
    cosine = (1.0 - reflection**2 + transmission**2) / (2.0 * safe_transmission)
    radical = jnp.sqrt(cosine**2 - 1.0 + 0.0j)
    roots = jnp.stack((cosine + radical, cosine - radical), axis=1)
    impedance_base = jnp.sqrt(
        ((1.0 + reflection) ** 2 - transmission**2)
        / ((1.0 - reflection) ** 2 - transmission**2)
        + 0.0j
    )
    branch_numbers = []
    root_signs = []
    impedance_signs = []
    propagation = []
    wave_numbers = []
    impedances = []
    for branch in range(plan.minimum_branch, plan.maximum_branch + 1):
        for root_index in range(2):
            for impedance_sign in (1, -1):
                q = roots[:, root_index]
                wave_number = (
                    -1.0j * jnp.log(q) + 2.0 * jnp.pi * branch
                ) / sweep.thickness
                branch_numbers.append(branch)
                root_signs.append(1 if root_index == 0 else -1)
                impedance_signs.append(impedance_sign)
                propagation.append(q)
                wave_numbers.append(wave_number)
                impedances.append(impedance_sign * impedance_base)
    branch_array = jnp.asarray(branch_numbers, dtype=jnp.int32)
    root_array = jnp.asarray(root_signs, dtype=jnp.int32)
    impedance_sign_array = jnp.asarray(impedance_signs, dtype=jnp.int32)
    propagation_array = jnp.stack(tuple(propagation), axis=1)
    wave_number_array = jnp.stack(tuple(wave_numbers), axis=1)
    relative_impedance_array = jnp.stack(tuple(impedances), axis=1)
    effective_impedance_array = (
        relative_impedance_array / sweep.left_modal_admittances[:, None]
    )
    refractive_array = wave_number_array / frequencies[:, None]
    permittivity_array = refractive_array / effective_impedance_array
    permeability_array = refractive_array * effective_impedance_array
    reconstructed_reflection, reconstructed_transmission = _slab_scattering(
        propagation_array, relative_impedance_array
    )
    reconstruction_residuals = jnp.maximum(
        jnp.maximum(
            jnp.abs(reconstructed_reflection - r_left[:, None]),
            jnp.abs(reconstructed_reflection - r_right[:, None]),
        ),
        jnp.maximum(
            jnp.abs(reconstructed_transmission - t_lr[:, None]),
            jnp.abs(reconstructed_transmission - t_rl[:, None]),
        ),
    )
    finite_candidates = (
        jnp.isfinite(wave_number_array)
        & jnp.isfinite(relative_impedance_array)
        & jnp.isfinite(effective_impedance_array)
        & jnp.isfinite(reconstruction_residuals)
    )
    passive_candidates = (
        jnp.imag(wave_number_array) >= -plan.reconstruction_tolerance
    ) & (jnp.real(effective_impedance_array) >= -plan.reconstruction_tolerance)
    candidate_admissible = finite_candidates & (
        reconstruction_residuals <= plan.reconstruction_tolerance
    )
    if plan.passive_claim:
        candidate_admissible = candidate_admissible & passive_candidates
    normal = jnp.all(
        jnp.sqrt(jnp.sum(jnp.abs(sweep.bloch_wavevectors) ** 2, axis=-1))
        <= plan.normal_incidence_tolerance
    )
    sweep_eligible = (
        jnp.all(sweep.finite)
        & jnp.all(sweep.symmetric_termination)
        & jnp.all(sweep.physical_finite_layers)
        & ~jnp.any(sweep.additional_propagating_orders)
        & ~jnp.any(sweep.grazing)
        & (
            jnp.max(sweep.cross_polarization_conversion)
            <= plan.cross_polarization_tolerance
        )
        & jnp.all(jnp.abs(transmission) > plan.transmission_tolerance)
        & normal
        & jnp.all(jnp.real(frequencies) > 0.0)
        & jnp.all(jnp.imag(frequencies) == 0.0)
    )
    selected_indices = []
    margins = []
    ambiguous_rows = []
    previous_selected = None
    for frequency_index in range(int(frequencies.shape[0])):
        if frequency_index == 0:
            if plan.anchor == "low-frequency":
                target = jnp.asarray(1.0 + 0.0j)
            else:
                assert plan.anchor_refractive_index is not None
                target = (
                    plan.anchor_refractive_index
                    if plan.anchor_refractive_index.shape == ()
                    else plan.anchor_refractive_index[frequency_index]
                )
        elif (
            plan.anchor_refractive_index is not None
            and plan.anchor_refractive_index.shape == frequencies.shape
        ):
            target = plan.anchor_refractive_index[frequency_index]
        else:
            assert previous_selected is not None
            target = previous_selected
        scores = jnp.abs(refractive_array[frequency_index] - target)
        scores = jnp.where(candidate_admissible[frequency_index], scores, jnp.inf)
        ordering = np.argsort(np.asarray(scores))
        selected = int(ordering[0])
        best = float(np.asarray(scores[selected]))
        second = float(np.asarray(scores[ordering[1]])) if ordering.size > 1 else np.inf
        margin = second - best
        row_ambiguous = (not np.isfinite(best)) or margin <= plan.anchor_tolerance
        selected_indices.append(selected)
        margins.append(margin)
        ambiguous_rows.append(row_ambiguous)
        previous_selected = refractive_array[frequency_index, selected]
    selected_array = jnp.asarray(selected_indices, dtype=jnp.int32)
    frequency_indices = jnp.arange(frequencies.shape[0])
    selected_wave_number = wave_number_array[frequency_indices, selected_array]
    selected_refractive_index = refractive_array[frequency_indices, selected_array]
    selected_relative_impedance = relative_impedance_array[
        frequency_indices, selected_array
    ]
    selected_effective_impedance = effective_impedance_array[
        frequency_indices, selected_array
    ]
    selected_permittivity = permittivity_array[frequency_indices, selected_array]
    selected_permeability = permeability_array[frequency_indices, selected_array]
    selected_residual = reconstruction_residuals[frequency_indices, selected_array]
    branch_margin = jnp.asarray(margins, dtype=frequencies.real.dtype)
    ambiguous = jnp.asarray(any(ambiguous_rows))
    eligible = sweep_eligible & jnp.all(
        candidate_admissible[frequency_indices, selected_array]
    )
    status = jnp.where(
        ~eligible,
        int(EquivalentSlabRetrievalStatus.INELIGIBLE),
        jnp.where(
            ambiguous,
            int(EquivalentSlabRetrievalStatus.AMBIGUOUS),
            int(EquivalentSlabRetrievalStatus.VALID),
        ),
    )
    kk_residual = jnp.maximum(
        _finite_band_kk_residual(frequencies, selected_permittivity - 1.0),
        _finite_band_kk_residual(frequencies, selected_permeability - 1.0),
    )
    stable_fit = (
        jnp.sqrt(jnp.sum(jnp.abs(jnp.diff(selected_refractive_index, n=2)) ** 2))
        / jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(selected_refractive_index) ** 2)), 1.0)
        if frequencies.shape[0] >= 3
        else jnp.asarray(jnp.inf, dtype=frequencies.real.dtype)
    )
    retrieval_id = canonical_fingerprint(
        {
            "kind": "equivalent-slab-retrieval",
            "sweep_id": sweep.sweep_id,
            "plan_id": plan.plan_id,
            "anchor_refractive_index": array_tree_fingerprint(
                ()
                if plan.anchor_refractive_index is None
                else plan.anchor_refractive_index
            ),
            "numeric_revision_ids": list(sweep.numeric_revision_ids),
        }
    )
    return EquivalentSlabRetrieval(
        sweep,
        branch_array,
        root_array,
        impedance_sign_array,
        propagation_array,
        wave_number_array,
        refractive_array,
        relative_impedance_array,
        effective_impedance_array,
        permittivity_array,
        permeability_array,
        reconstruction_residuals,
        candidate_admissible,
        selected_array,
        selected_wave_number,
        selected_refractive_index,
        selected_relative_impedance,
        selected_effective_impedance,
        selected_permittivity,
        selected_permeability,
        selected_residual,
        branch_margin,
        kk_residual,
        stable_fit,
        eligible,
        ambiguous,
        status,
        plan_id=plan.plan_id,
        retrieval_id=retrieval_id,
    )


def _commensurate(
    values: Sequence[float], policy: LocalIsotropicQualificationPolicy, /
) -> bool:
    base = min(values)
    for value in values:
        ratio = value / base
        approximation = Fraction(ratio).limit_denominator(
            policy.commensurate_denominator_limit
        )
        if abs(float(approximation) - ratio) > policy.commensurate_tolerance:
            return False
    return True


def _maximum_relative_deviation(
    values: Array, reference: Array, absolute_tolerance: float, /
) -> Array:
    scale = jnp.maximum(jnp.abs(reference), absolute_tolerance)
    return jnp.max(jnp.abs(values - reference) / scale)


def _angle_prediction_residual(
    sweep: MaxwellModalSweep,
    permittivity: Array,
    permeability: Array,
    /,
) -> Array:
    omega = sweep.angular_frequencies
    transverse_squared = jnp.sum(jnp.abs(sweep.bloch_wavevectors) ** 2, axis=-1)
    kz = jnp.sqrt(omega**2 * permittivity * permeability - transverse_squared + 0.0j)
    kz = jnp.where(jnp.imag(kz) < 0.0, -kz, kz)
    kz = jnp.where((jnp.abs(jnp.imag(kz)) <= 1.0e-12) & (jnp.real(kz) < 0.0), -kz, kz)
    material_admittance = (
        kz / (omega * permeability)
        if sweep.polarization == "te"
        else omega * permittivity / kz
    )
    relative_impedance = sweep.left_modal_admittances / material_admittance
    propagation = jnp.exp(1.0j * kz * sweep.thickness)
    reflection, transmission = _slab_scattering(propagation, relative_impedance)
    return jnp.max(
        jnp.stack(
            (
                jnp.max(jnp.abs(reflection - sweep.deembedded_left_reflection)),
                jnp.max(jnp.abs(reflection - sweep.deembedded_right_reflection)),
                jnp.max(
                    jnp.abs(transmission - sweep.deembedded_left_to_right_transmission)
                ),
                jnp.max(
                    jnp.abs(transmission - sweep.deembedded_right_to_left_transmission)
                ),
            )
        )
    )


def qualify_local_isotropic_medium(
    retrievals: Sequence[EquivalentSlabRetrieval],
    angle_sweeps: Sequence[MaxwellModalSweep],
    loss_evidence: Sequence[FourierModalLossEvidence],
    loss_convergence: FourierModalLossConvergenceEvidence,
    policy: LocalIsotropicQualificationPolicy,
    /,
) -> LocalIsotropicMediumQualification:
    """Qualify locality/isotropy without creating a constitutive material."""
    retrieved = tuple(retrievals)
    angled = tuple(angle_sweeps)
    losses = tuple(loss_evidence)
    if len(retrieved) < 4:
        raise ValueError("Qualification requires both polarizations at two thicknesses.")
    if not angled:
        raise ValueError("Qualification requires declared nonzero-angle Maxwell sweeps.")
    if not isinstance(loss_convergence, FourierModalLossConvergenceEvidence):
        raise TypeError("loss_convergence must be FourierModalLossConvergenceEvidence.")
    if not isinstance(policy, LocalIsotropicQualificationPolicy):
        raise TypeError("policy must be LocalIsotropicQualificationPolicy.")
    ambiguous_input = any(
        int(np.asarray(value.status)) == int(EquivalentSlabRetrievalStatus.AMBIGUOUS)
        for value in retrieved
    )
    valid_retrievals = all(
        int(np.asarray(value.status)) == int(EquivalentSlabRetrievalStatus.VALID)
        for value in retrieved
    )
    polarizations = {value.sweep.polarization for value in retrieved}
    thicknesses = sorted(
        {float(np.asarray(value.sweep.thickness)) for value in retrieved}
    )
    thickness_polarizations = {
        (float(np.asarray(value.sweep.thickness)), value.sweep.polarization)
        for value in retrieved
    }
    required_pairs = {
        (thickness, polarization)
        for thickness in thicknesses
        for polarization in ("te", "tm")
    }
    frequency_shape = retrieved[0].sweep.angular_frequencies.shape
    frequencies_match = all(
        value.sweep.angular_frequencies.shape == frequency_shape
        and np.array_equal(
            np.asarray(value.sweep.angular_frequencies),
            np.asarray(retrieved[0].sweep.angular_frequencies),
            equal_nan=False,
        )
        for value in retrieved
    )
    structural = (
        polarizations == {"te", "tm"}
        and len(thicknesses) >= 2
        and required_pairs.issubset(thickness_polarizations)
        and _commensurate(thicknesses, policy)
        and frequencies_match
    )
    permittivities = jnp.stack(tuple(value.permittivity for value in retrieved))
    permeabilities = jnp.stack(tuple(value.permeability for value in retrieved))
    representative_epsilon = jnp.mean(permittivities, axis=0)
    representative_mu = jnp.mean(permeabilities, axis=0)
    thickness_residuals = []
    polarization_residuals = []
    for polarization in ("te", "tm"):
        values = jnp.stack(
            tuple(
                jnp.stack((value.permittivity, value.permeability), axis=0)
                for value in retrieved
                if value.sweep.polarization == polarization
            )
        )
        thickness_residuals.append(
            _maximum_relative_deviation(
                values, jnp.mean(values, axis=0), policy.parameter_absolute_tolerance
            )
        )
    for thickness in thicknesses:
        values = jnp.stack(
            tuple(
                jnp.stack((value.permittivity, value.permeability), axis=0)
                for value in retrieved
                if float(np.asarray(value.sweep.thickness)) == thickness
            )
        )
        polarization_residuals.append(
            _maximum_relative_deviation(
                values, jnp.mean(values, axis=0), policy.parameter_absolute_tolerance
            )
        )
    thickness_residual = jnp.max(jnp.stack(tuple(thickness_residuals)))
    polarization_residual = jnp.max(jnp.stack(tuple(polarization_residuals)))
    angle_polarizations = {
        value.polarization
        for value in angled
        if bool(np.any(np.asarray(value.polar_angles) > 0.0))
    }
    angle_structural = angle_polarizations == {"te", "tm"} and all(
        value.angular_frequencies.shape == frequency_shape
        and bool(value.numeric_revision_ids)
        and np.array_equal(
            np.asarray(value.angular_frequencies),
            np.asarray(retrieved[0].sweep.angular_frequencies),
            equal_nan=False,
        )
        for value in angled
    )
    angle_eligible = jnp.all(
        jnp.stack(
            tuple(
                jnp.all(value.finite)
                & jnp.all(value.symmetric_termination)
                & jnp.all(value.physical_finite_layers)
                & ~jnp.any(value.additional_propagating_orders)
                & ~jnp.any(value.grazing)
                & (
                    jnp.max(value.cross_polarization_conversion)
                    <= policy.reconstructed_scattering_tolerance
                )
                for value in angled
            )
        )
    )
    angle_residual = jnp.max(
        jnp.stack(
            tuple(
                _angle_prediction_residual(
                    value, representative_epsilon, representative_mu
                )
                for value in angled
            )
        )
    )
    reconstruction_residual = jnp.max(
        jnp.stack(tuple(jnp.max(value.reconstruction_residual) for value in retrieved))
    )
    branch_margin = jnp.min(
        jnp.stack(tuple(jnp.min(value.branch_margin) for value in retrieved))
    )
    passive_consistent = jnp.all(
        jnp.imag(
            retrieved[0].sweep.angular_frequencies
            * jnp.sqrt(representative_epsilon * representative_mu + 0.0j)
        )
        >= -policy.parameter_absolute_tolerance
    )
    if not policy.passive_claim:
        passive_consistent = jnp.asarray(True)
    loss_accepted = bool(losses) and jnp.all(
        jnp.stack(tuple(value.accepted for value in losses))
    )
    convergence_accepted = loss_convergence.accepted
    loss_revisions = tuple(
        value.numeric_revision_id
        for value in losses
        if value.numeric_revision_id is not None
    )
    stack_ids = (
        {value.sweep.physical_stack_id for value in retrieved}
        | {value.physical_stack_id for value in losses}
        | {value.physical_stack_id for value in angled}
    )
    stack_digests = (
        {value.sweep.physical_stack_digest for value in retrieved}
        | {value.physical_stack_digest for value in losses}
        | {value.physical_stack_digest for value in angled}
        | {loss_convergence.physical_stack_digest}
    )
    single_physical_stack = (
        None not in stack_digests and len(stack_ids) == 1 and len(stack_digests) == 1
    )
    retrieval_stack_ids = {value.sweep.physical_stack_id for value in retrieved}
    loss_relevant = all(
        value.physical_stack_id in retrieval_stack_ids
        and value.physical_stack_digest in stack_digests
        and any(
            np.array_equal(
                np.asarray(value.angular_frequency),
                np.asarray(frequency),
                equal_nan=False,
            )
            for frequency in retrieved[0].sweep.angular_frequencies
        )
        and any(
            np.array_equal(
                np.asarray(value.total_physical_thickness),
                np.asarray(thickness),
                equal_nan=False,
            )
            for thickness in thicknesses
        )
        for value in losses
    )
    loss_linked = (
        bool(losses)
        and set(loss_revisions).issubset(loss_convergence.numeric_revision_ids)
        and loss_relevant
    )
    parameter_limit = (
        policy.parameter_relative_tolerance + policy.parameter_absolute_tolerance
    )
    gates = (
        jnp.asarray(
            valid_retrievals
            and structural
            and angle_structural
            and loss_linked
            and single_physical_stack
        )
        & angle_eligible
        & (thickness_residual <= parameter_limit)
        & (polarization_residual <= parameter_limit)
        & (angle_residual <= policy.angle_prediction_tolerance)
        & (reconstruction_residual <= policy.reconstructed_scattering_tolerance)
        & (branch_margin >= policy.minimum_branch_margin)
        & passive_consistent
        & loss_accepted
        & convergence_accepted
        & jnp.all(
            jnp.stack(
                tuple(value.sweep.symmetric_termination.all() for value in retrieved)
            )
        )
        & ~jnp.any(
            jnp.stack(
                tuple(
                    jnp.any(value.sweep.additional_propagating_orders)
                    for value in retrieved
                )
            )
        )
    )
    reasons = []
    if ambiguous_input:
        reasons.append("retrieval_branch_ambiguous")
    if not valid_retrievals:
        reasons.append("retrieval_ineligible")
    if not structural:
        reasons.append("thickness_or_polarization_coverage")
    if not angle_structural:
        reasons.append("nonzero_angle_or_modal_revision_missing")
    if not bool(np.asarray(angle_eligible)):
        reasons.append("angle_termination_diffraction_or_conversion")
    if not all(
        bool(np.all(np.asarray(value.sweep.symmetric_termination))) for value in retrieved
    ):
        reasons.append("asymmetric_termination")
    if any(
        bool(np.any(np.asarray(value.sweep.additional_propagating_orders)))
        for value in retrieved
    ):
        reasons.append("higher_propagating_diffraction_order")
    if not single_physical_stack:
        reasons.append("unrelated_physical_stack")
    if not loss_linked:
        reasons.append("loss_revision_link")
    if float(np.asarray(thickness_residual)) > parameter_limit:
        reasons.append("thickness_disagreement")
    if float(np.asarray(polarization_residual)) > parameter_limit:
        reasons.append("polarization_disagreement")
    if float(np.asarray(angle_residual)) > policy.angle_prediction_tolerance:
        reasons.append("angle_or_spatial_dispersion_disagreement")
    if (
        float(np.asarray(reconstruction_residual))
        > policy.reconstructed_scattering_tolerance
    ):
        reasons.append("reconstructed_scattering_disagreement")
    if float(np.asarray(branch_margin)) < policy.minimum_branch_margin:
        reasons.append("branch_margin")
    if not bool(np.asarray(passive_consistent)):
        reasons.append("passive_decay_or_sign")
    if not bool(np.asarray(loss_accepted)):
        reasons.append("physical_loss_evidence")
    if not bool(np.asarray(convergence_accepted)):
        reasons.append("loss_convergence")
    qualified = gates & ~jnp.asarray(ambiguous_input)
    status = jnp.where(
        jnp.asarray(ambiguous_input),
        int(LocalIsotropicQualificationStatus.AMBIGUOUS),
        jnp.where(
            qualified,
            int(LocalIsotropicQualificationStatus.QUALIFIED),
            int(LocalIsotropicQualificationStatus.INELIGIBLE),
        ),
    )
    retrieval_ids = tuple(value.retrieval_id for value in retrieved)
    angle_sweep_ids = tuple(value.sweep_id for value in angled)
    qualification_id = canonical_fingerprint(
        {
            "kind": "local-isotropic-medium-qualification",
            "retrieval_ids": list(retrieval_ids),
            "angle_sweep_ids": list(angle_sweep_ids),
            "loss_numeric_revision_ids": list(loss_revisions),
            "loss_harmonic_discretization_ids": list(
                loss_convergence.harmonic_discretization_ids
            ),
            "loss_physical_state_digest": loss_convergence.physical_state_digest,
            "policy_id": policy.policy_id,
        }
    )
    return LocalIsotropicMediumQualification(
        representative_epsilon,
        representative_mu,
        thickness_residual,
        polarization_residual,
        angle_residual,
        reconstruction_residual,
        branch_margin,
        passive_consistent,
        jnp.asarray(loss_accepted),
        convergence_accepted,
        qualified,
        status,
        tuple(reasons),
        retrieval_ids,
        angle_sweep_ids,
        loss_revisions,
        policy.policy_id,
        qualification_id,
    )


__all__ = [
    "EquivalentSlabRetrieval",
    "EquivalentSlabRetrievalPlan",
    "EquivalentSlabRetrievalStatus",
    "LocalIsotropicMediumQualification",
    "LocalIsotropicQualificationPolicy",
    "LocalIsotropicQualificationStatus",
    "MaxwellModalSweep",
    "prepare_maxwell_modal_sweep",
    "qualify_local_isotropic_medium",
    "retrieve_equivalent_slab",
]
