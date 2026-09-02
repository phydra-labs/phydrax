#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import LatticeHarmonicDiscretization
from ....linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ._contracts import AbstractFourierFactorizationPlan, FrequencyMaxwellMaterial


FrameDifferentiation: TypeAlias = Literal["mathematical", "frozen", "none"]


def _dense_solve(matrix: Array, right_hand_side: Array) -> Array:
    policy = LinearSolvePolicy(
        DenseLU(),
        differentiation=DifferentiationPolicy("mathematical"),
        failure=FailurePolicy("error"),
    )
    problem = LinearSystem(DenseLinearOperator(matrix))
    return solve(problem, right_hand_side, policy=policy).value


def _identity_tensor_samples(
    scalar: Array,
    sample_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    scalar_ = jnp.asarray(scalar, dtype=dtype)
    values = jnp.broadcast_to(scalar_, sample_shape)
    identity = jnp.eye(3, dtype=dtype)
    return values[..., None, None] * identity


def _tensor_samples(
    value: ArrayLike,
    lattice: LatticeHarmonicDiscretization,
    /,
) -> tuple[Array, bool]:
    dtype = jnp.dtype(lattice.plan.precision.transform_dtype)
    array = jnp.asarray(value, dtype=dtype)
    shape = lattice.sample_shape
    if array.ndim == 0:
        return _identity_tensor_samples(array, shape, dtype), True
    if array.shape == (3, 3):
        return jnp.broadcast_to(array, shape + (3, 3)), False
    if array.shape == shape:
        return _identity_tensor_samples(array, shape, dtype), True
    if array.shape == shape + (3, 3):
        return array, False
    raise ValueError(
        "Material data must be scalar, a sampled scalar field, one 3x3 tensor, "
        f"or sampled tensors with shape {shape + (3, 3)}; got {array.shape}."
    )


def _component_convolutions(
    tensor_samples: Array,
    lattice: LatticeHarmonicDiscretization,
    /,
) -> Array:
    matrices = lattice.convolution_matrix(tensor_samples)
    return jnp.transpose(matrices, (2, 3, 0, 1))


def _translate_tensor_convolutions(
    matrices: Array,
    lattice: LatticeHarmonicDiscretization,
    translation: ArrayLike,
    /,
) -> Array:
    values = jnp.transpose(matrices, (2, 3, 0, 1))
    translated = lattice.translate_convolution(values, translation)
    return jnp.transpose(translated, (2, 3, 0, 1))


def _inverse_scalar_convolution(
    scalar_samples: Array,
    lattice: LatticeHarmonicDiscretization,
    /,
) -> Array:
    inverse_samples = jnp.reciprocal(scalar_samples)
    inverse_convolution = lattice.convolution_matrix(inverse_samples)
    identity = jnp.eye(lattice.harmonic_count, dtype=inverse_convolution.dtype)
    return _dense_solve(inverse_convolution, identity)


def _scalar_from_tensor(tensor_samples: Array) -> Array:
    diagonal = jnp.stack(
        (tensor_samples[..., 0, 0], tensor_samples[..., 1, 1], tensor_samples[..., 2, 2]),
        axis=-1,
    )
    reference = diagonal[..., :1]
    off_diagonal = (
        tensor_samples - jnp.eye(3, dtype=tensor_samples.dtype) * reference[..., None]
    )
    scale = jnp.maximum(jnp.max(jnp.abs(tensor_samples)), jnp.asarray(1.0))
    return eqx.error_if(
        diagonal[..., 0],
        (
            jnp.max(jnp.abs(diagonal - reference))
            > 100 * jnp.finfo(diagonal.real.dtype).eps * scale
        )
        | (
            jnp.max(jnp.abs(off_diagonal))
            > 100 * jnp.finfo(diagonal.real.dtype).eps * scale
        ),
        "This Fourier factorization requires scalar isotropic material samples.",
    )


class DirectFourierFactorizationPlan(AbstractFourierFactorizationPlan):
    """Direct Laurent multiplication for every constitutive component."""

    def __init__(self):
        self.plan_id = canonical_fingerprint({"kind": "direct-fourier-factorization"})

    @property
    def kind(self) -> str:
        return "direct"


class InverseFourierFactorizationPlan(AbstractFourierFactorizationPlan):
    """Inverse-rule transverse factorization for scalar isotropic media."""

    def __init__(self):
        self.plan_id = canonical_fingerprint({"kind": "inverse-fourier-factorization"})

    @property
    def kind(self) -> str:
        return "inverse"


class AnalyticInterfaceFramePlan(StrictModule):
    """Caller-supplied periodic tangent field."""

    tangent_field: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, tangent_field: ArrayLike, /, *, frame_id: str | None = None):
        field = jnp.asarray(tangent_field)
        if field.shape[-1:] != (2,):
            raise ValueError("tangent_field must have trailing shape (2,).")
        identifier = (
            canonical_fingerprint(
                {"kind": "analytic-interface-frame", "shape": list(field.shape)}
            )
            if frame_id is None
            else str(frame_id)
        )
        self.tangent_field = field
        self.plan_id = identifier


class JonesDirectFramePlan(StrictModule, NonTrainableState):
    """Smooth Fourier least-squares Jones tangent-field plan."""

    regularization: float = eqx.field(static=True)
    gradient_regularization: float = eqx.field(static=True)
    differentiation: FrameDifferentiation = eqx.field(static=True)
    complex_jones: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        regularization: float = 1e-4,
        gradient_regularization: float = 1e-8,
        differentiation: FrameDifferentiation = "mathematical",
        complex_jones: bool = True,
    ):
        regularization_ = float(regularization)
        gradient_regularization_ = float(gradient_regularization)
        if regularization_ <= 0.0 or gradient_regularization_ <= 0.0:
            raise ValueError("Jones regularization values must be positive.")
        if differentiation not in ("mathematical", "frozen", "none"):
            raise ValueError("Unknown Jones frame differentiation policy.")
        self.regularization = regularization_
        self.gradient_regularization = gradient_regularization_
        self.differentiation = differentiation
        self.complex_jones = bool(complex_jones)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "jones-direct-frame",
                "regularization": regularization_,
                "gradient_regularization": gradient_regularization_,
                "differentiation": differentiation,
                "complex_jones": self.complex_jones,
            }
        )


class VectorFourierFactorizationPlan(AbstractFourierFactorizationPlan):
    """Local-frame inverse rule for scalar isotropic permittivity."""

    frame: AnalyticInterfaceFramePlan | JonesDirectFramePlan

    def __init__(
        self,
        frame: AnalyticInterfaceFramePlan | JonesDirectFramePlan | None = None,
        /,
    ):
        frame_ = JonesDirectFramePlan() if frame is None else frame
        if not isinstance(frame_, AnalyticInterfaceFramePlan | JonesDirectFramePlan):
            raise TypeError("frame must be an analytic or Jones-direct frame plan.")
        self.frame = frame_
        self.plan_id = canonical_fingerprint(
            {"kind": "vector-fourier-factorization", "frame": frame_.plan_id}
        )

    @property
    def kind(self) -> str:
        return "vector"


class FourierFactorizationDiagnostics(StrictModule):
    frame_fit_residual: Array
    frame_normalization_defect: Array
    frame_gradient_omitted: Array


class PreparedFourierMaterial(StrictModule):
    """Convolution matrices for all constitutive tensor components."""

    permittivity: Array
    permeability: Array
    magnetoelectric_xi: Array
    magnetoelectric_zeta: Array
    tangent_field: Array | None
    diagnostics: FourierFactorizationDiagnostics
    material_id: str = eqx.field(static=True)
    factorization_id: str = eqx.field(static=True)


def _jones_target(tangent: Array, regularization: float, complex_jones: bool) -> Array:
    magnitude = jnp.sqrt(
        jnp.sum(jnp.abs(tangent) ** 2, axis=-1, keepdims=True) + regularization**2
    )
    normalized = tangent / magnitude
    if not complex_jones:
        return normalized
    tx = normalized[..., 0:1]
    ty = normalized[..., 1:2]
    theta = jnp.arctan2(jnp.real(ty), jnp.real(tx))
    gradient_magnitude = jnp.clip(magnitude, 0.0, 1.0)
    phi = jnp.pi / 8.0 * (1.0 + jnp.cos(jnp.pi * gradient_magnitude))
    phase = jnp.exp(1j * theta)
    jx = phase * (tx * jnp.cos(phi) - 1j * ty * jnp.sin(phi))
    jy = phase * (ty * jnp.cos(phi) + 1j * tx * jnp.sin(phi))
    return jnp.concatenate((jx, jy), axis=-1)


def _prepare_jones_frame(
    scalar_samples: Array,
    lattice: LatticeHarmonicDiscretization,
    plan: JonesDirectFramePlan,
    /,
) -> tuple[Array, Array, Array]:
    source = jnp.real(scalar_samples)
    if plan.differentiation in ("frozen", "none"):
        source = jax.lax.stop_gradient(source)
    source = source - jnp.min(source)
    scale = jnp.maximum(jnp.max(source), plan.gradient_regularization)
    source = source / scale
    coefficients = lattice.analysis(source)
    wavevectors = lattice.harmonic_wavevectors.astype(coefficients.dtype)
    grad_x = lattice.synthesis(1j * wavevectors[:, 0] * coefficients)
    grad_y = lattice.synthesis(1j * wavevectors[:, 1] * coefficients)
    gradient_magnitude = jnp.sqrt(
        jnp.abs(grad_x) ** 2 + jnp.abs(grad_y) ** 2 + plan.gradient_regularization**2
    )
    tangent = jnp.stack((grad_y, -grad_x), axis=-1) / gradient_magnitude[..., None]
    target = _jones_target(tangent, plan.gradient_regularization, plan.complex_jones)
    weights = gradient_magnitude / jnp.maximum(
        jnp.max(gradient_magnitude), plan.gradient_regularization
    )

    identity = jnp.eye(lattice.harmonic_count, dtype=coefficients.dtype)
    synthesis = lattice.synthesis(identity).reshape((-1, lattice.harmonic_count))
    target_flat = target.reshape((-1, 2))
    weight_flat = weights.reshape((-1,))
    normalization = jnp.maximum(jnp.sum(weight_flat), plan.gradient_regularization)
    weighted_synthesis = synthesis * weight_flat[:, None]
    normal_matrix = jnp.conj(synthesis.T) @ weighted_synthesis / normalization
    wavevector_norm = jnp.sum(jnp.abs(wavevectors) ** 2, axis=-1)
    wavevector_norm = wavevector_norm / jnp.maximum(jnp.max(wavevector_norm), 1.0)
    normal_matrix = normal_matrix + plan.regularization * jnp.diag(
        wavevector_norm + plan.gradient_regularization
    )
    right_hand_side = (
        jnp.conj(synthesis.T) @ (weight_flat[:, None] * target_flat) / normalization
    )
    fourier_field = _dense_solve(normal_matrix, right_hand_side)
    field = lattice.synthesis(fourier_field)
    field_magnitude = jnp.sqrt(
        jnp.sum(jnp.abs(field) ** 2, axis=-1, keepdims=True)
        + plan.gradient_regularization**2
    )
    field = field / field_magnitude
    residual = jnp.sqrt(
        jnp.sum(weight_flat[:, None] * jnp.abs(field.reshape((-1, 2)) - target_flat) ** 2)
        / normalization
    )
    normalization_defect = jnp.max(jnp.abs(jnp.sum(jnp.abs(field) ** 2, axis=-1) - 1.0))
    return field, residual, normalization_defect


def _prepare_frame(
    scalar_samples: Array,
    lattice: LatticeHarmonicDiscretization,
    frame: AnalyticInterfaceFramePlan | JonesDirectFramePlan,
    /,
) -> tuple[Array, Array, Array, Array]:
    if isinstance(frame, AnalyticInterfaceFramePlan):
        field = jnp.asarray(
            frame.tangent_field,
            dtype=jnp.dtype(lattice.plan.precision.transform_dtype),
        )
        if field.shape != lattice.sample_shape + (2,):
            raise ValueError(
                "Analytic tangent field must have shape "
                f"{lattice.sample_shape + (2,)}; got {field.shape}."
            )
        magnitude = jnp.sqrt(jnp.sum(jnp.abs(field) ** 2, axis=-1, keepdims=True))
        field = field / eqx.error_if(
            magnitude,
            jnp.any(magnitude <= jnp.finfo(magnitude.dtype).eps),
            "Analytic tangent field must be nonzero everywhere.",
        )
        defect = jnp.max(jnp.abs(jnp.sum(jnp.abs(field) ** 2, axis=-1) - 1.0))
        return field, jnp.asarray(0.0), defect, jnp.asarray(False)
    field, residual, defect = _prepare_jones_frame(scalar_samples, lattice, frame)
    return field, residual, defect, jnp.asarray(frame.differentiation != "mathematical")


def _vector_transverse_blocks(
    scalar_samples: Array,
    direct: Array,
    inverse: Array,
    tangent_field: Array,
    lattice: LatticeHarmonicDiscretization,
    /,
) -> Array:
    count = lattice.harmonic_count
    tx = tangent_field[..., 0]
    ty = tangent_field[..., 1]
    projector = jnp.block(
        [
            [
                lattice.convolution_matrix(tx * jnp.conj(tx)),
                lattice.convolution_matrix(tx * jnp.conj(ty)),
            ],
            [
                lattice.convolution_matrix(ty * jnp.conj(tx)),
                lattice.convolution_matrix(ty * jnp.conj(ty)),
            ],
        ]
    )
    direct_block = jnp.block(
        [[direct, jnp.zeros_like(direct)], [jnp.zeros_like(direct), direct]]
    )
    inverse_block = jnp.block(
        [[inverse, jnp.zeros_like(inverse)], [jnp.zeros_like(inverse), inverse]]
    )
    effective = direct_block - (direct_block - inverse_block) @ projector
    return effective.reshape((2, count, 2, count)).transpose((0, 2, 1, 3))


def prepare_fourier_material(
    material: FrequencyMaxwellMaterial,
    lattice: LatticeHarmonicDiscretization,
    factorization: AbstractFourierFactorizationPlan,
    /,
    *,
    translation: ArrayLike = (0.0, 0.0),
) -> PreparedFourierMaterial:
    epsilon_samples, epsilon_scalar = _tensor_samples(material.permittivity, lattice)
    mu_samples, mu_scalar = _tensor_samples(material.permeability, lattice)
    xi_samples, _ = _tensor_samples(material.magnetoelectric_xi, lattice)
    zeta_samples, _ = _tensor_samples(material.magnetoelectric_zeta, lattice)
    epsilon = _component_convolutions(epsilon_samples, lattice)
    mu = _component_convolutions(mu_samples, lattice)
    xi = _component_convolutions(xi_samples, lattice)
    zeta = _component_convolutions(zeta_samples, lattice)
    tangent_field = None
    residual = jnp.asarray(0.0)
    defect = jnp.asarray(0.0)
    omitted = jnp.asarray(False)

    if isinstance(factorization, InverseFourierFactorizationPlan):
        if not epsilon_scalar or not mu_scalar:
            raise ValueError(
                "Inverse factorization currently requires scalar isotropic media."
            )
        epsilon_scalar_samples = _scalar_from_tensor(epsilon_samples)
        mu_scalar_samples = _scalar_from_tensor(mu_samples)
        epsilon_inverse = _inverse_scalar_convolution(epsilon_scalar_samples, lattice)
        mu_inverse = _inverse_scalar_convolution(mu_scalar_samples, lattice)
        epsilon = epsilon.at[0, 0].set(epsilon_inverse)
        epsilon = epsilon.at[1, 1].set(epsilon_inverse)
        mu = mu.at[0, 0].set(mu_inverse)
        mu = mu.at[1, 1].set(mu_inverse)
    elif isinstance(factorization, VectorFourierFactorizationPlan):
        if not epsilon_scalar:
            raise ValueError(
                "Vector factorization currently requires scalar permittivity."
            )
        epsilon_scalar_samples = _scalar_from_tensor(epsilon_samples)
        epsilon_direct = lattice.convolution_matrix(epsilon_scalar_samples)
        epsilon_inverse = _inverse_scalar_convolution(epsilon_scalar_samples, lattice)
        tangent_field, residual, defect, omitted = _prepare_frame(
            epsilon_scalar_samples,
            lattice,
            factorization.frame,
        )
        transverse = _vector_transverse_blocks(
            epsilon_scalar_samples,
            epsilon_direct,
            epsilon_inverse,
            tangent_field,
            lattice,
        )
        epsilon = epsilon.at[0, 0].set(transverse[0, 0])
        epsilon = epsilon.at[0, 1].set(transverse[0, 1])
        epsilon = epsilon.at[1, 0].set(transverse[1, 0])
        epsilon = epsilon.at[1, 1].set(transverse[1, 1])
    elif not isinstance(factorization, DirectFourierFactorizationPlan):
        raise TypeError("Unknown Fourier factorization plan.")

    diagnostics = FourierFactorizationDiagnostics(residual, defect, omitted)
    prepared = PreparedFourierMaterial(
        epsilon,
        mu,
        xi,
        zeta,
        tangent_field,
        diagnostics,
        material_id=material.material_id,
        factorization_id=factorization.plan_id,
    )
    return translate_prepared_fourier_material(prepared, lattice, translation)


def translate_prepared_fourier_material(
    material: PreparedFourierMaterial,
    lattice: LatticeHarmonicDiscretization,
    translation: ArrayLike,
    /,
) -> PreparedFourierMaterial:
    """Apply reciprocal-space translation without rebuilding material convolutions."""
    epsilon = _translate_tensor_convolutions(material.permittivity, lattice, translation)
    mu = _translate_tensor_convolutions(material.permeability, lattice, translation)
    xi = _translate_tensor_convolutions(material.magnetoelectric_xi, lattice, translation)
    zeta = _translate_tensor_convolutions(
        material.magnetoelectric_zeta, lattice, translation
    )
    tangent_field = material.tangent_field
    if tangent_field is not None:
        tangent_coefficients = lattice.analysis(tangent_field)
        tangent_field = lattice.synthesis(
            lattice.translate_coefficients(tangent_coefficients, translation)
        )
    return PreparedFourierMaterial(
        epsilon,
        mu,
        xi,
        zeta,
        tangent_field,
        material.diagnostics,
        material_id=material.material_id,
        factorization_id=material.factorization_id,
    )


__all__ = [
    "AnalyticInterfaceFramePlan",
    "DirectFourierFactorizationPlan",
    "FourierFactorizationDiagnostics",
    "FrameDifferentiation",
    "InverseFourierFactorizationPlan",
    "JonesDirectFramePlan",
    "PreparedFourierMaterial",
    "VectorFourierFactorizationPlan",
    "prepare_fourier_material",
    "translate_prepared_fourier_material",
]
