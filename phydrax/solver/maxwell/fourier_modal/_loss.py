#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....lifecycle import NumericRevision
from ....linalg import HermitianSpectrum
from ._continuous import PreparedContinuousFourierModalLayer
from ._fields import fields_in_layer
from ._numeric_revision import require_fourier_modal_numeric_revision
from ._runtime import (
    _canonical_material_samples,
    FourierModalSolveResult,
    PreparedFourierModalLayer,
    PreparedFourierModalMaxwell,
)


class FourierModalLossStatus(IntEnum):
    SUCCESS = 0
    INELIGIBLE = 1
    NONFINITE = 2
    NUMERIC_REVISION_REQUIRED = 3
    CLOSURE_TOLERANCE_NOT_MET = 4
    PASSIVE_CLAIM_VIOLATED = 5


class FourierModalLossPolicy(StrictModule, NonTrainableState):
    """Opt-in physical-volume loss quadrature and evidence tolerances."""

    z_quadrature_order: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    passive_psd_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        z_quadrature_order: int = 8,
        relative_tolerance: float = 1.0e-6,
        absolute_tolerance: float = 1.0e-9,
        passive_psd_tolerance: float = 1.0e-10,
    ):
        order = int(z_quadrature_order)
        if order < 2:
            raise ValueError("z_quadrature_order must be at least two.")
        if min(relative_tolerance, absolute_tolerance, passive_psd_tolerance) < 0.0:
            raise ValueError("Fourier-modal loss tolerances must be non-negative.")
        self.z_quadrature_order = order
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.passive_psd_tolerance = float(passive_psd_tolerance)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-loss-policy",
                "z_quadrature_order": order,
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
                "passive_psd_tolerance": self.passive_psd_tolerance,
            }
        )


class FourierModalLossEvidence(StrictModule):
    """Independent port, face-flux, and physical-volume dissipation evidence."""

    left_incoming_power: Array
    right_incoming_power: Array
    left_outgoing_power: Array
    right_outgoing_power: Array
    net_port_power_into_stack: Array
    layer_face_fluxes: Array
    layer_face_poynting_drop: Array
    layer_volume_material_loss: Array
    total_volume_material_loss: Array
    unresolved_numerical_closure: Array
    z_quadrature_defect: Array
    passive_minimum_loss_eigenvalue: Array
    passive_claim_violation: Array
    eligible: Array
    finite: Array
    revision_bound: Array
    accepted: Array
    angular_frequency: Array
    total_physical_thickness: Array
    status: Array
    primitive_vectors: Array
    harmonic_discretization_id: str = eqx.field(static=True)
    physical_stack_id: str = eqx.field(static=True)
    physical_stack_digest: str | None = eqx.field(static=True)
    physical_state_digest: str | None = eqx.field(static=True)
    harmonic_mode_ids: tuple[str, ...] = eqx.field(static=True)
    numeric_revision_id: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)


class FourierModalLossConvergenceEvidence(StrictModule):
    """Convergence evidence across independently prepared harmonic discretizations."""

    port_power_differences: Array
    material_loss_differences: Array
    closure_differences: Array
    port_converged: Array
    material_converged: Array
    closure_converged: Array
    nested_refinement: Array
    accepted: Array
    harmonic_discretization_ids: tuple[str, ...] = eqx.field(static=True)
    numeric_revision_ids: tuple[str, ...] = eqx.field(static=True)
    physical_state_digest: str = eqx.field(static=True)
    physical_stack_digest: str = eqx.field(static=True)


def _adjoint_constitutive(value: Array, /) -> Array:
    return jnp.conj(jnp.transpose(value, (1, 0, 3, 2)))


def _hermitian_imaginary(value: Array, /) -> Array:
    return (value - _adjoint_constitutive(value)) / (2.0j)


def _loss_density(field, layer: PreparedFourierModalLayer, omega: Array, /) -> Array:
    epsilon_loss = _hermitian_imaginary(layer.operator.permittivity)
    mu_loss = _hermitian_imaginary(layer.operator.permeability)
    electric = field.electric_harmonics
    magnetic = field.magnetic_harmonics
    electric_work = contract(
        "hir,ijhk,kjr->r", jnp.conj(electric), epsilon_loss, electric
    )
    magnetic_work = contract("hir,ijhk,kjr->r", jnp.conj(magnetic), mu_loss, magnetic)
    return 0.5 * jnp.real(omega) * jnp.real(electric_work + magnetic_work)


def _integrated_layer_loss(
    prepared: PreparedFourierModalMaxwell,
    result: FourierModalSolveResult,
    layer_index: int,
    layer: PreparedFourierModalLayer,
    order: int,
    /,
) -> Array:
    nodes_host, weights_host = np.polynomial.legendre.leggauss(order)
    dtype = layer.layer.thickness.real.dtype
    nodes = jnp.asarray(nodes_host, dtype=dtype)
    weights = jnp.asarray(weights_host, dtype=dtype)
    thickness = jnp.real(layer.layer.thickness)
    offsets = 0.5 * thickness * (nodes + 1.0)
    values = jnp.stack(
        tuple(
            _loss_density(
                fields_in_layer(prepared, result, layer_index, offsets[index]),
                layer,
                prepared.problem.angular_frequency,
            )
            for index in range(order)
        ),
        axis=0,
    )
    return (
        0.5
        * thickness
        * prepared.problem.harmonics.cell_measure
        * contract("z,zr->r", weights, values)
    )


def _tangential_face_flux(
    electric: Array,
    magnetic: Array,
    harmonic_count: int,
    cell_measure: Array,
    /,
) -> Array:
    ex = electric[:harmonic_count]
    ey = electric[harmonic_count:]
    hx = magnetic[:harmonic_count]
    hy = magnetic[harmonic_count:]
    return (
        0.5
        * cell_measure
        * jnp.real(jnp.sum(ex * jnp.conj(hy) - ey * jnp.conj(hx), axis=0))
    )


def _minimum_loss_eigenvalue(layer: PreparedFourierModalLayer, /) -> Array:
    epsilon = _hermitian_imaginary(layer.operator.permittivity)
    mu = _hermitian_imaginary(layer.operator.permeability)
    count = layer.operator.harmonic_count
    epsilon_matrix = jax.lax.stop_gradient(
        jnp.transpose(epsilon, (0, 2, 1, 3)).reshape((3 * count, 3 * count))
    )
    mu_matrix = jax.lax.stop_gradient(
        jnp.transpose(mu, (0, 2, 1, 3)).reshape((3 * count, 3 * count))
    )
    epsilon_spectrum = HermitianSpectrum(epsilon_matrix)
    mu_spectrum = HermitianSpectrum(mu_matrix)
    return jnp.minimum(
        epsilon_spectrum.minimum_eigenvalue, mu_spectrum.minimum_eigenvalue
    )


def _port_is_lossless(prepared: PreparedFourierModalMaxwell, /) -> Array:
    checks = []
    for port in (prepared.problem.superstrate, prepared.problem.substrate):
        samples = _canonical_material_samples(port.material, prepared.problem)
        epsilon, permeability, xi, zeta = samples
        epsilon_loss = (epsilon - jnp.conj(jnp.swapaxes(epsilon, -1, -2))) / (2.0j)
        permeability_loss = (
            permeability - jnp.conj(jnp.swapaxes(permeability, -1, -2))
        ) / (2.0j)
        checks.append(
            jnp.all(epsilon_loss == 0.0)
            & jnp.all(permeability_loss == 0.0)
            & jnp.all(xi == 0.0)
            & jnp.all(zeta == 0.0)
        )
    return jnp.all(jnp.stack(tuple(checks)))


def _physical_stack_id(prepared: PreparedFourierModalMaxwell, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "fourier-modal-physical-stack",
            "elements": [
                {
                    "kind": "layer",
                    "layer_id": element.layer.layer_id,
                    "material_id": element.layer.material.material_id,
                    "material_role": element.layer.material.material_role,
                    "origin_evidence_id": element.layer.material.origin_evidence_id,
                    "factorization_id": element.layer.factorization.plan_id,
                }
                for element in prepared.elements
                if isinstance(element, PreparedFourierModalLayer)
            ],
        }
    )


def evaluate_fourier_modal_loss(
    prepared: PreparedFourierModalMaxwell,
    result: FourierModalSolveResult,
    policy: FourierModalLossPolicy,
    /,
    *,
    numeric_revision: NumericRevision | None = None,
) -> FourierModalLossEvidence:
    """Evaluate physical volume loss without changing the ordinary solve path."""
    if not isinstance(prepared, PreparedFourierModalMaxwell):
        raise TypeError("prepared must be PreparedFourierModalMaxwell.")
    if not isinstance(result, FourierModalSolveResult):
        raise TypeError("result must be FourierModalSolveResult.")
    if not isinstance(policy, FourierModalLossPolicy):
        raise TypeError("policy must be FourierModalLossPolicy.")
    if result.provenance.preparation_id != prepared.preparation_id:
        raise ValueError("result belongs to a different prepared Fourier-modal state.")
    physical_state_digest = None
    physical_stack_digest = None
    if numeric_revision is not None:
        require_fourier_modal_numeric_revision(prepared, numeric_revision)
        revision_metadata = dict(numeric_revision.metadata)
        physical_state_digest = revision_metadata["physical_state_digest"]
        physical_stack_digest = revision_metadata["physical_stack_digest"]
    revision_bound = jnp.asarray(numeric_revision is not None)
    revision_id = None if numeric_revision is None else numeric_revision.revision_id
    rhs_count = int(result.net_port_power_into_stack.shape[0])
    layer_count = prepared.problem.layer_count
    static_eligible = (
        layer_count > 0
        and bool(result.boundary_electric_fields)
        and not prepared.problem.source_ids
        and prepared.problem.superstrate.material.material_role == "physical"
        and prepared.problem.substrate.material.material_role == "physical"
        and not any(
            isinstance(element, PreparedContinuousFourierModalLayer)
            for element in prepared.elements
        )
        and all(
            not isinstance(element, PreparedFourierModalLayer)
            or element.layer.material.material_role == "physical"
            for element in prepared.elements
        )
    )
    layers = tuple(
        element
        for element in prepared.elements
        if isinstance(element, PreparedFourierModalLayer)
    )
    zero_coupling = jnp.all(
        jnp.stack(
            tuple(
                jnp.all(jnp.abs(layer.operator.magnetoelectric_xi) == 0.0)
                & jnp.all(jnp.abs(layer.operator.magnetoelectric_zeta) == 0.0)
                for layer in layers
            )
            or (jnp.asarray(True),)
        )
    )
    port_power = result.left_incoming_power + result.right_incoming_power
    port_driven = jnp.any(port_power > 0.0)
    positive_frequency = (
        jnp.isfinite(prepared.problem.angular_frequency)
        & (jnp.real(prepared.problem.angular_frequency) > 0.0)
        & (jnp.imag(prepared.problem.angular_frequency) == 0.0)
    )
    eligible = (
        jnp.asarray(static_eligible)
        & zero_coupling
        & _port_is_lossless(prepared)
        & port_driven
        & ~result.internal_source_excitation
        & positive_frequency
    )
    if static_eligible:
        face_fluxes = []
        face_drops = []
        volume_losses = []
        quadrature_defects = []
        passive_minima = []
        layer_index = 0
        count = prepared.problem.harmonics.harmonic_count
        measure = prepared.problem.harmonics.cell_measure
        for element_index, element in enumerate(prepared.elements):
            if not isinstance(element, PreparedFourierModalLayer):
                continue
            left_flux = _tangential_face_flux(
                result.boundary_electric_fields[element_index],
                result.boundary_magnetic_fields[element_index],
                count,
                measure,
            )
            right_flux = _tangential_face_flux(
                result.boundary_electric_fields[element_index + 1],
                result.boundary_magnetic_fields[element_index + 1],
                count,
                measure,
            )
            high = _integrated_layer_loss(
                prepared, result, layer_index, element, policy.z_quadrature_order
            )
            low = _integrated_layer_loss(
                prepared,
                result,
                layer_index,
                element,
                max(1, policy.z_quadrature_order // 2),
            )
            face_fluxes.append(jnp.stack((left_flux, right_flux), axis=0))
            face_drops.append(left_flux - right_flux)
            volume_losses.append(high)
            quadrature_defects.append(jnp.abs(high - low))
            passive_minima.append(_minimum_loss_eigenvalue(element))
            layer_index += 1
        layer_face_fluxes = jnp.stack(tuple(face_fluxes), axis=0)
        layer_face_drop = jnp.stack(tuple(face_drops), axis=0)
        layer_volume_loss = jnp.stack(tuple(volume_losses), axis=0)
        z_defect = jnp.stack(tuple(quadrature_defects), axis=0)
        passive_minimum = jnp.stack(tuple(passive_minima))
    else:
        real_dtype = result.net_port_power_into_stack.dtype
        layer_face_fluxes = jnp.full(
            (layer_count, 2, rhs_count), jnp.nan, dtype=real_dtype
        )
        layer_face_drop = jnp.full((layer_count, rhs_count), jnp.nan, dtype=real_dtype)
        layer_volume_loss = jnp.full((layer_count, rhs_count), jnp.nan, dtype=real_dtype)
        z_defect = jnp.full((layer_count, rhs_count), jnp.nan, dtype=real_dtype)
        passive_minimum = jnp.full((layer_count,), jnp.nan, dtype=real_dtype)
    total_volume_loss = jnp.sum(layer_volume_loss, axis=0)
    closure = result.net_port_power_into_stack - total_volume_loss
    passive_claims = (
        jnp.asarray(
            tuple(layer.layer.material.passive is True for layer in layers),
            dtype=bool,
        )
        if static_eligible
        else jnp.zeros((layer_count,), dtype=bool)
    )
    passive_violation = passive_claims & (passive_minimum < -policy.passive_psd_tolerance)
    finite = (
        jnp.all(jnp.isfinite(layer_face_fluxes))
        & jnp.all(jnp.isfinite(layer_volume_loss))
        & jnp.all(jnp.isfinite(closure))
    )
    scale = jnp.maximum(
        jnp.maximum(
            jnp.abs(result.net_port_power_into_stack), jnp.abs(total_volume_loss)
        ),
        1.0,
    )
    closure_limit = policy.absolute_tolerance + policy.relative_tolerance * scale
    quadrature_limit = (
        policy.absolute_tolerance
        + policy.relative_tolerance * jnp.maximum(jnp.abs(layer_volume_loss), 1.0)
    )
    closure_met = jnp.all(jnp.abs(closure) <= closure_limit) & jnp.all(
        z_defect <= quadrature_limit
    )
    status = jnp.where(
        ~eligible,
        int(FourierModalLossStatus.INELIGIBLE),
        jnp.where(
            ~finite,
            int(FourierModalLossStatus.NONFINITE),
            jnp.where(
                ~revision_bound,
                int(FourierModalLossStatus.NUMERIC_REVISION_REQUIRED),
                jnp.where(
                    jnp.any(passive_violation),
                    int(FourierModalLossStatus.PASSIVE_CLAIM_VIOLATED),
                    jnp.where(
                        ~closure_met,
                        int(FourierModalLossStatus.CLOSURE_TOLERANCE_NOT_MET),
                        int(FourierModalLossStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    )
    accepted = status == int(FourierModalLossStatus.SUCCESS)
    return FourierModalLossEvidence(
        result.left_incoming_power,
        result.right_incoming_power,
        result.left_outgoing_power,
        result.right_outgoing_power,
        result.net_port_power_into_stack,
        layer_face_fluxes,
        layer_face_drop,
        layer_volume_loss,
        total_volume_loss,
        closure,
        z_defect,
        passive_minimum,
        passive_violation,
        eligible,
        finite,
        revision_bound,
        accepted,
        prepared.problem.angular_frequency,
        prepared.total_thickness,
        status,
        prepared.problem.harmonics.primitive_vectors,
        harmonic_discretization_id=prepared.problem.harmonics.preparation_id,
        physical_stack_id=_physical_stack_id(prepared),
        physical_stack_digest=physical_stack_digest,
        physical_state_digest=physical_state_digest,
        harmonic_mode_ids=prepared.problem.harmonics.plan.layout.mode_ids,
        numeric_revision_id=revision_id,
        policy_id=policy.policy_id,
    )


def _relative_difference(previous: Array, current: Array, /) -> Array:
    scale = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(current) ** 2)), 1.0)
    return jnp.sqrt(jnp.sum(jnp.abs(current - previous) ** 2)) / scale


def assess_fourier_modal_loss_convergence(
    evidence: tuple[FourierModalLossEvidence, ...],
    /,
    *,
    relative_tolerance: float = 1.0e-4,
    absolute_tolerance: float = 1.0e-8,
) -> FourierModalLossConvergenceEvidence:
    """Assess nested-harmonic convergence across independent loss evaluations."""
    values = tuple(evidence)
    if len(values) < 2:
        raise ValueError("At least two loss evaluations are required.")
    if min(relative_tolerance, absolute_tolerance) < 0.0:
        raise ValueError("Convergence tolerances must be non-negative.")
    if any(
        value.numeric_revision_id is None
        or value.physical_state_digest is None
        or value.physical_stack_digest is None
        for value in values
    ):
        raise ValueError(
            "Every loss evaluation requires a bound NumericRevision and physical digest."
        )
    harmonic_ids = tuple(value.harmonic_discretization_id for value in values)
    if len(set(harmonic_ids)) != len(harmonic_ids):
        raise ValueError("Loss convergence requires distinct harmonic discretizations.")
    revision_ids = tuple(
        value.numeric_revision_id
        for value in values
        if value.numeric_revision_id is not None
    )
    if len(set(revision_ids)) != len(revision_ids):
        raise ValueError("Loss convergence requires distinct numeric revisions.")
    if any(
        isinstance(leaf, jax.core.Tracer)
        for value in values
        for leaf in jax.tree.leaves((value.primitive_vectors, value.accepted))
    ):
        raise ValueError("Loss convergence requires host-materialized evidence.")
    if len({value.physical_stack_id for value in values}) != 1:
        raise ValueError("Loss convergence requires one semantic physical stack.")
    if len({value.physical_stack_digest for value in values}) != 1:
        raise ValueError("Loss convergence requires one content-bound physical stack.")
    if len({value.physical_state_digest for value in values}) != 1:
        raise ValueError("Loss convergence requires equal content-bound physical states.")
    reference = values[0]
    if any(
        not np.array_equal(
            np.asarray(value.angular_frequency),
            np.asarray(reference.angular_frequency),
            equal_nan=False,
        )
        or not np.array_equal(
            np.asarray(value.total_physical_thickness),
            np.asarray(reference.total_physical_thickness),
            equal_nan=False,
        )
        for value in values[1:]
    ):
        raise ValueError("Loss convergence requires equal frequency and thickness.")
    nested = True
    for previous, current in zip(values[:-1], values[1:], strict=True):
        nested = (
            nested
            and len(previous.harmonic_mode_ids) < len(current.harmonic_mode_ids)
            and set(previous.harmonic_mode_ids).issubset(current.harmonic_mode_ids)
        )
        nested = nested and np.array_equal(
            np.asarray(previous.primitive_vectors),
            np.asarray(current.primitive_vectors),
            equal_nan=False,
        )
    if not nested:
        raise ValueError("Loss evidence does not form a nested harmonic refinement.")
    port_differences = jnp.stack(
        tuple(
            _relative_difference(
                previous.net_port_power_into_stack, current.net_port_power_into_stack
            )
            for previous, current in zip(values[:-1], values[1:], strict=True)
        )
    )
    material_differences = jnp.stack(
        tuple(
            _relative_difference(
                previous.total_volume_material_loss,
                current.total_volume_material_loss,
            )
            for previous, current in zip(values[:-1], values[1:], strict=True)
        )
    )
    closure_differences = jnp.stack(
        tuple(
            _relative_difference(
                previous.unresolved_numerical_closure,
                current.unresolved_numerical_closure,
            )
            for previous, current in zip(values[:-1], values[1:], strict=True)
        )
    )
    limit = relative_tolerance + absolute_tolerance
    port_converged = port_differences[-1] <= limit
    material_converged = material_differences[-1] <= limit
    closure_converged = closure_differences[-1] <= limit
    inputs_accepted = jnp.all(jnp.stack(tuple(value.accepted for value in values)))
    accepted = inputs_accepted & port_converged & material_converged & closure_converged
    return FourierModalLossConvergenceEvidence(
        port_differences,
        material_differences,
        closure_differences,
        port_converged,
        material_converged,
        closure_converged,
        jnp.asarray(True),
        accepted,
        harmonic_ids,
        revision_ids,
        next(
            value.physical_state_digest
            for value in values
            if value.physical_state_digest is not None
        ),
        next(
            value.physical_stack_digest
            for value in values
            if value.physical_stack_digest is not None
        ),
    )


__all__ = [
    "FourierModalLossConvergenceEvidence",
    "FourierModalLossEvidence",
    "FourierModalLossPolicy",
    "FourierModalLossStatus",
    "assess_fourier_modal_loss_convergence",
    "evaluate_fourier_modal_loss",
]
