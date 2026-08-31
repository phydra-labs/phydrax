#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...dynamics import LinearDescriptorSystem
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    FailurePolicy,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveStatus,
    solve,
)
from .._models import AbstractScatteringComponent, ScatteringResponse
from .._ports import WavePort


class RationalFitPolicy(StrictModule):
    pole_count: int = eqx.field(static=True)
    minimum_decay: float = eqx.field(static=True)
    maximum_decay: float = eqx.field(static=True)
    include_proportional: bool = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        pole_count: int = 8,
        minimum_decay: float = 1e-3,
        maximum_decay: float = 1e3,
        include_proportional: bool = False,
        residual_tolerance: float = 1e-4,
    ):
        count = int(pole_count)
        minimum, maximum = float(minimum_decay), float(maximum_decay)
        tolerance = float(residual_tolerance)
        if count <= 0 or minimum <= 0.0 or maximum < minimum or tolerance < 0.0:
            raise ValueError("Rational fit policy values are invalid.")
        self.pole_count = count
        self.minimum_decay = minimum
        self.maximum_decay = maximum
        self.include_proportional = bool(include_proportional)
        self.residual_tolerance = tolerance


class RationalMatrixModel(StrictModule):
    poles: Array
    residues: Array
    direct: Array
    proportional: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        poles: ArrayLike,
        residues: ArrayLike,
        direct: ArrayLike,
        proportional: ArrayLike,
        /,
        *,
        model_id: str | None = None,
    ):
        poles_ = jnp.asarray(poles, dtype=jnp.complex128)
        residues_ = jnp.asarray(residues, dtype=jnp.complex128)
        direct_ = jnp.asarray(direct, dtype=jnp.complex128)
        proportional_ = jnp.asarray(proportional, dtype=jnp.complex128)
        if poles_.ndim != 1 or poles_.size == 0:
            raise ValueError("poles must be a nonempty vector.")
        if (
            residues_.ndim != 3
            or residues_.shape[0] != poles_.size
            or residues_.shape[1] != residues_.shape[2]
            or direct_.shape != residues_.shape[1:]
            or proportional_.shape != direct_.shape
        ):
            raise ValueError("Rational model arrays have incompatible shapes.")
        if bool(jnp.any(~jnp.isfinite(poles_))) or bool(jnp.any(jnp.real(poles_) >= 0.0)):
            raise ValueError("Rational model poles must be finite and stable.")
        if any(
            bool(jnp.any(~jnp.isfinite(value)))
            for value in (residues_, direct_, proportional_)
        ):
            raise ValueError("Rational model coefficients must be finite.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "rational-matrix-model",
                    "poles": array_tree_fingerprint(poles_),
                    "residues": array_tree_fingerprint(residues_),
                    "direct": array_tree_fingerprint(direct_),
                    "proportional": array_tree_fingerprint(proportional_),
                }
            )
            if model_id is None
            else str(model_id)
        )
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.poles, self.residues = poles_, residues_
        self.direct, self.proportional = direct_, proportional_
        self.model_id = identifier

    @property
    def port_count(self) -> int:
        return int(self.direct.shape[0])

    def evaluate_s(self, points: ArrayLike, /) -> Array:
        values = jnp.asarray(points, dtype=jnp.complex128)
        kernel = 1.0 / (values[..., None] - self.poles)
        dynamic = jnp.sum(
            kernel[..., :, None, None] * self.residues,
            axis=-3,
        )
        return dynamic + self.direct + values[..., None, None] * self.proportional

    def evaluate_frequency(self, angular_frequency: ArrayLike, /) -> Array:
        return self.evaluate_s(-1j * jnp.asarray(angular_frequency))


class RationalFitEvidence(StrictModule):
    absolute_residual: Array
    relative_residual: Array
    maximum_error: Array
    linear_status: Array
    stable: Array
    finite: Array
    accepted: Array


class RationalFitResult(StrictModule):
    model: RationalMatrixModel
    evidence: RationalFitEvidence
    fitted: Array


class RationalReductionResult(StrictModule):
    model: RationalMatrixModel
    retained_indices: Array
    discarded_residue_norm: Array
    passivity_preserved: bool = eqx.field(static=True)


class RationalScatteringComponent(AbstractScatteringComponent):
    model: RationalMatrixModel
    _ports: tuple[WavePort, ...]
    numeric_version: Array
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: RationalMatrixModel,
        ports: Sequence[WavePort],
        /,
        *,
        numeric_version: ArrayLike = 0,
        component_id: str | None = None,
    ):
        if not isinstance(model, RationalMatrixModel):
            raise TypeError("model must be RationalMatrixModel.")
        port_tuple = tuple(ports)
        if sum(port.size for port in port_tuple) != model.port_count:
            raise ValueError("Rational scattering ports do not match model coordinates.")
        identifier = model.model_id if component_id is None else str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.model, self._ports = model, port_tuple
        self.numeric_version = jnp.asarray(numeric_version, dtype=jnp.int32)
        self.component_id = identifier

    @property
    def ports(self) -> tuple[WavePort, ...]:
        return self._ports

    def evaluate(self, angular_frequency: ArrayLike, /) -> ScatteringResponse:
        return ScatteringResponse(
            self.model.evaluate_frequency(angular_frequency),
            tuple(reference for port in self._ports for reference in port.references),
            self.numeric_version,
        )


def _default_poles(frequencies: Array, policy: RationalFitPolicy) -> Array:
    maximum_frequency = jnp.maximum(jnp.max(jnp.abs(frequencies)), 1.0)
    decay = jnp.geomspace(
        policy.minimum_decay * maximum_frequency,
        policy.maximum_decay * maximum_frequency,
        policy.pole_count,
    )
    return -decay.astype(jnp.complex128)


def fit_rational_matrix(
    angular_frequency: ArrayLike,
    samples: ArrayLike,
    /,
    *,
    policy: RationalFitPolicy | None = None,
    poles: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    model_id: str | None = None,
) -> RationalFitResult:
    omega = jnp.asarray(angular_frequency, dtype=float)
    data = jnp.asarray(samples, dtype=jnp.complex128)
    selected = RationalFitPolicy() if policy is None else policy
    if not isinstance(selected, RationalFitPolicy):
        raise TypeError("policy must be RationalFitPolicy or None.")
    if omega.ndim != 1 or omega.size < 2 or data.shape[0] != omega.size:
        raise ValueError("Frequency samples must have one nonempty leading data axis.")
    if data.ndim != 3 or data.shape[1] != data.shape[2]:
        raise ValueError("samples must have shape (frequency, output, input).")
    if bool(jnp.any(~jnp.isfinite(omega))) or bool(jnp.any(jnp.diff(omega) <= 0.0)):
        raise ValueError("angular_frequency must be finite and strictly increasing.")
    poles_ = (
        _default_poles(omega, selected)
        if poles is None
        else jnp.asarray(poles, dtype=jnp.complex128)
    )
    if poles_.shape != (selected.pole_count,) or bool(
        jnp.any(~jnp.isfinite(poles_)) | jnp.any(jnp.real(poles_) >= 0.0)
    ):
        raise ValueError("Fit poles must match pole_count and be finite and stable.")
    points = -1j * omega
    columns = [1.0 / (points - pole) for pole in poles_]
    columns.append(jnp.ones_like(points))
    if selected.include_proportional:
        columns.append(points)
    design = jnp.stack(columns, axis=-1)
    rhs = data.reshape((omega.size, -1))
    if weights is not None:
        weight = jnp.asarray(weights, dtype=float)
        if weight.shape != omega.shape or bool(
            jnp.any(~jnp.isfinite(weight)) | jnp.any(weight <= 0.0)
        ):
            raise ValueError("weights must be a finite positive frequency vector.")
        root = jnp.sqrt(weight)
        design = root[:, None] * design
        rhs = root[:, None] * rhs
    linear = solve(
        LeastSquaresProblem(
            DenseLinearOperator(design, operator_id="rational-fit-design"),
            problem_id="rational-fit",
        ),
        rhs,
        policy=LinearSolvePolicy(DenseSVD(), failure=FailurePolicy("status")),
    )
    coefficients = jnp.asarray(linear.value)
    count = data.shape[1]
    residues = coefficients[: poles_.size].reshape((poles_.size, count, count))
    direct = coefficients[poles_.size].reshape((count, count))
    proportional = (
        coefficients[poles_.size + 1].reshape((count, count))
        if selected.include_proportional
        else jnp.zeros_like(direct)
    )
    model = RationalMatrixModel(poles_, residues, direct, proportional, model_id=model_id)
    fitted = model.evaluate_frequency(omega)
    residual = fitted - data
    absolute = jnp.linalg.norm(residual)
    relative = absolute / jnp.maximum(jnp.linalg.norm(data), 1.0)
    maximum = jnp.max(jnp.abs(residual))
    finite = jnp.all(jnp.isfinite(fitted))
    success = jnp.all(linear.status == int(LinearSolveStatus.SUCCESS))
    evidence = RationalFitEvidence(
        absolute,
        relative,
        maximum,
        jnp.asarray(linear.status, dtype=jnp.int32),
        jnp.asarray(True),
        finite,
        success & finite & (relative <= selected.residual_tolerance),
    )
    return RationalFitResult(model, evidence, fitted)


def realize_rational_model(model: RationalMatrixModel, /) -> LinearDescriptorSystem:
    if not isinstance(model, RationalMatrixModel):
        raise TypeError("model must be RationalMatrixModel.")
    if bool(jnp.any(jnp.abs(model.proportional) > 0.0)):
        raise ValueError("Descriptor realization currently requires a proper model.")
    ports = model.port_count
    poles = model.poles.size
    state_size = poles * ports
    mass = jnp.eye(state_size, dtype=jnp.complex128)
    state = jnp.zeros((state_size, state_size), dtype=jnp.complex128)
    inputs = jnp.zeros((state_size, ports), dtype=jnp.complex128)
    outputs = jnp.zeros((ports, state_size), dtype=jnp.complex128)
    for index, pole in enumerate(model.poles):
        block = slice(index * ports, (index + 1) * ports)
        state = state.at[block, block].set(pole * jnp.eye(ports))
        inputs = inputs.at[block, :].set(jnp.eye(ports))
        outputs = outputs.at[:, block].set(model.residues[index])
    return LinearDescriptorSystem(
        mass,
        state,
        inputs,
        outputs,
        model.direct,
        system_id=f"{model.model_id}/descriptor",
    )


def reduce_rational_model(
    model: RationalMatrixModel,
    retained_poles: int,
    /,
) -> RationalReductionResult:
    if not isinstance(model, RationalMatrixModel):
        raise TypeError("model must be RationalMatrixModel.")
    count = int(retained_poles)
    if count <= 0 or count > model.poles.size:
        raise ValueError("retained_poles must lie within the model pole count.")
    norms = jnp.linalg.norm(model.residues, axis=(-2, -1))
    indices = jnp.sort(jnp.argsort(norms)[-count:])
    mask = jnp.ones((model.poles.size,), dtype=bool).at[indices].set(False)
    reduced = RationalMatrixModel(
        model.poles[indices],
        model.residues[indices],
        model.direct,
        model.proportional,
        model_id=f"{model.model_id}/reduced-{count}",
    )
    return RationalReductionResult(
        reduced,
        indices,
        jnp.sum(norms[mask]),
        False,
    )


class RationalPassivityAudit(StrictModule):
    maximum_singular_value: Array
    passivity_residual: Array
    finite: Array
    passive: Array
    certified: bool = eqx.field(static=True)


class PassiveDescriptorCertificate(StrictModule):
    minimum_energy_eigenvalue: Array
    interconnection_defect: Array
    minimum_dissipation_eigenvalue: Array
    minimum_feedthrough_eigenvalue: Array
    certified: Array
    certificate_id: str = eqx.field(static=True)


def audit_rational_scattering(
    model: RationalMatrixModel,
    angular_frequency: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
) -> RationalPassivityAudit:
    if not isinstance(model, RationalMatrixModel):
        raise TypeError("model must be RationalMatrixModel.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative.")
    matrix = model.evaluate_frequency(angular_frequency)
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    maximum = jnp.max(singular_values, axis=-1)
    finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    residual = jnp.maximum(maximum - 1.0, 0.0)
    return RationalPassivityAudit(
        maximum,
        residual,
        finite,
        finite & (residual <= tolerance),
        False,
    )


def passive_descriptor_system(
    energy_matrix: ArrayLike,
    interconnection_matrix: ArrayLike,
    dissipation_matrix: ArrayLike,
    input_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    system_id: str = "passive-descriptor",
) -> tuple[LinearDescriptorSystem, PassiveDescriptorCertificate]:
    energy = jnp.asarray(energy_matrix, dtype=jnp.complex128)
    interconnection = jnp.asarray(interconnection_matrix, dtype=jnp.complex128)
    dissipation = jnp.asarray(dissipation_matrix, dtype=jnp.complex128)
    inputs = jnp.asarray(input_matrix, dtype=jnp.complex128)
    feedthrough = jnp.asarray(feedthrough_matrix, dtype=jnp.complex128)
    if energy.ndim != 2 or energy.shape[0] != energy.shape[1]:
        raise ValueError("energy_matrix must be square.")
    size = energy.shape[0]
    input_count = inputs.shape[-1] if inputs.ndim == 2 else -1
    if (
        interconnection.shape != (size, size)
        or dissipation.shape != (size, size)
        or inputs.shape != (size, input_count)
        or feedthrough.shape != (input_count, input_count)
    ):
        raise ValueError("Passive descriptor matrices have incompatible shapes.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative.")
    energy_h = 0.5 * (energy + jnp.conj(energy.T))
    dissipation_h = 0.5 * (dissipation + jnp.conj(dissipation.T))
    feedthrough_h = 0.5 * (feedthrough + jnp.conj(feedthrough.T))
    energy_min = jnp.min(jnp.linalg.eigvalsh(energy_h))
    dissipation_min = jnp.min(jnp.linalg.eigvalsh(dissipation_h))
    feedthrough_min = jnp.min(jnp.linalg.eigvalsh(feedthrough_h))
    interconnection_defect = jnp.linalg.norm(
        interconnection + jnp.conj(interconnection.T)
    )
    scale = jnp.maximum(
        jnp.linalg.norm(energy_h)
        + jnp.linalg.norm(dissipation_h)
        + jnp.linalg.norm(feedthrough_h),
        1.0,
    )
    certified = (
        (energy_min > tolerance)
        & (dissipation_min >= -tolerance)
        & (feedthrough_min >= -tolerance)
        & (interconnection_defect <= tolerance * scale)
    )
    if not bool(certified):
        raise ValueError(
            "Passive descriptor construction requires positive energy and "
            "semidefinite dissipation/feedthrough with skew interconnection."
        )
    system = LinearDescriptorSystem(
        energy_h,
        interconnection - dissipation_h,
        inputs,
        jnp.conj(inputs.T),
        feedthrough_h,
        system_id=system_id,
    )
    certificate_id = canonical_fingerprint(
        {
            "kind": "passive-descriptor-certificate",
            "system": system.system_id,
        }
    )
    certificate = PassiveDescriptorCertificate(
        energy_min,
        interconnection_defect,
        dissipation_min,
        feedthrough_min,
        certified,
        certificate_id,
    )
    return system, certificate


__all__ = [
    "PassiveDescriptorCertificate",
    "RationalPassivityAudit",
    "RationalFitEvidence",
    "RationalFitPolicy",
    "RationalFitResult",
    "RationalMatrixModel",
    "RationalReductionResult",
    "RationalScatteringComponent",
    "audit_rational_scattering",
    "passive_descriptor_system",
    "fit_rational_matrix",
    "realize_rational_model",
    "reduce_rational_model",
]
