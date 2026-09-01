#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear


VolumetricConstraintKind = Literal["jacobian", "logarithmic"]
_FINITE_STRAIN_SOLVE_PLAN = SmallLinearSolvePlan(3)


def plane_strain_embedding(deformation_gradient: ArrayLike, /) -> Array:
    """Embed a 2-D deformation gradient as ``diag(F, 1)`` in three dimensions."""
    deformation = jnp.asarray(deformation_gradient)
    if deformation.shape[-2:] != (2, 2):
        raise ValueError("Plane-strain deformation gradients must end in 2x2.")
    embedded = jnp.zeros(deformation.shape[:-2] + (3, 3), dtype=deformation.dtype)
    embedded = embedded.at[..., :2, :2].set(deformation)
    return embedded.at[..., 2, 2].set(1.0)


class FiniteStrainKinematics(StrictModule):
    """Pointwise orientation-preserving finite-strain kinematic evidence.

    Two-dimensional inputs are interpreted only as three-dimensional plane strain
    and are stored after the explicit ``diag(F, 1)`` embedding. Three-dimensional
    inputs are unchanged. ``admissible`` requires finite entries, a successful
    native small inverse solve, and a strictly positive Jacobian.
    """

    deformation_gradient: Array
    inverse_deformation_gradient: Array
    jacobian: Array
    cofactor: Array
    right_cauchy_green: Array
    left_cauchy_green: Array
    inverse_condition_estimate: Array
    inverse_residual_norm: Array
    admissible: Array
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)

    def __init__(self, deformation_gradient: ArrayLike, /):
        deformation = jnp.asarray(deformation_gradient)
        trailing_shape = deformation.shape[-2:]
        if trailing_shape == (2, 2):
            dimension = 2
            embedded = plane_strain_embedding(deformation)
        elif trailing_shape == (3, 3):
            dimension = 3
            embedded = deformation
        else:
            raise ValueError(
                "Finite-strain deformation gradients must end in 2x2 or 3x3."
            )

        identity = jnp.eye(3, dtype=embedded.dtype)
        inverse = solve_small_linear(
            _FINITE_STRAIN_SOLVE_PLAN,
            embedded,
            jnp.broadcast_to(identity, embedded.shape),
        )
        jacobian = inverse.determinant
        inverse_deformation = inverse.value
        inverse_transpose = jnp.swapaxes(inverse_deformation, -1, -2)
        finite = jnp.all(jnp.isfinite(embedded), axis=(-2, -1))
        admissible = (
            finite & inverse.successful & jnp.isfinite(jacobian) & (jacobian > 0.0)
        )

        self.deformation_gradient = embedded
        self.inverse_deformation_gradient = inverse_deformation
        self.jacobian = jacobian
        self.cofactor = jacobian[..., None, None] * inverse_transpose
        self.right_cauchy_green = oe.contract("...ki,...kj->...ij", embedded, embedded)
        self.left_cauchy_green = oe.contract("...ik,...jk->...ij", embedded, embedded)
        self.inverse_condition_estimate = inverse.condition_estimate
        self.inverse_residual_norm = inverse.residual_norm
        self.admissible = admissible
        self.dimension = dimension
        self.kinematics = "plane_strain" if dimension == 2 else "three_dimensional"

    @property
    def determinant(self) -> Array:
        return self.jacobian

    @property
    def volume_ratio(self) -> Array:
        return self.jacobian

    @property
    def inverse_transpose(self) -> Array:
        return jnp.swapaxes(self.inverse_deformation_gradient, -1, -2)


def finite_strain_kinematics(
    deformation_gradient: FiniteStrainKinematics | ArrayLike, /
) -> FiniteStrainKinematics:
    """Return canonical finite-strain kinematics, preserving prepared evidence."""
    if isinstance(deformation_gradient, FiniteStrainKinematics):
        return deformation_gradient
    return FiniteStrainKinematics(deformation_gradient)


class VolumetricConstraint(StrictModule, NonTrainableState):
    """Immutable incompressibility constraint in Jacobian or logarithmic form.

    ``kind="jacobian"`` evaluates ``J - 1`` with derivative ``cof(F)``.
    ``kind="logarithmic"`` evaluates ``log(J)`` with derivative ``F^{-T}``.
    Values and derivatives are non-finite outside the orientation-preserving
    admissible set.
    """

    kind: VolumetricConstraintKind = eqx.field(static=True)

    def __init__(self, kind: VolumetricConstraintKind = "jacobian", /):
        if kind not in ("jacobian", "logarithmic"):
            raise ValueError(
                "Volumetric constraint kind must be 'jacobian' or 'logarithmic'."
            )
        self.kind = kind

    def value(self, deformation_gradient: FiniteStrainKinematics | ArrayLike, /) -> Array:
        kinematics = finite_strain_kinematics(deformation_gradient)
        if self.kind == "jacobian":
            value = kinematics.jacobian - 1.0
        else:
            value = jnp.log(jnp.where(kinematics.admissible, kinematics.jacobian, 1.0))
        return jnp.where(kinematics.admissible, value, jnp.nan)

    def derivative(
        self, deformation_gradient: FiniteStrainKinematics | ArrayLike, /
    ) -> Array:
        kinematics = finite_strain_kinematics(deformation_gradient)
        derivative = (
            kinematics.cofactor
            if self.kind == "jacobian"
            else kinematics.inverse_transpose
        )
        return jnp.where(
            kinematics.admissible[..., None, None],
            derivative,
            jnp.full_like(derivative, jnp.nan),
        )

    def evaluate(
        self, deformation_gradient: FiniteStrainKinematics | ArrayLike, /
    ) -> tuple[Array, Array]:
        """Evaluate the scalar constraint and its derivative with respect to ``F``."""
        kinematics = finite_strain_kinematics(deformation_gradient)
        return self.value(kinematics), self.derivative(kinematics)


def volumetric_constraint(
    deformation_gradient: FiniteStrainKinematics | ArrayLike, /
) -> Array:
    """Evaluate the orientation-preserving incompressibility residual ``J - 1``."""
    return VolumetricConstraint("jacobian").value(deformation_gradient)


def logarithmic_volumetric_constraint(
    deformation_gradient: FiniteStrainKinematics | ArrayLike, /
) -> Array:
    """Evaluate the orientation-preserving incompressibility residual ``log(J)``."""
    return VolumetricConstraint("logarithmic").value(deformation_gradient)


def _embedded_area_vector(
    vector: ArrayLike, kinematics: FiniteStrainKinematics, /
) -> tuple[Array, int]:
    area_vector = jnp.asarray(vector)
    if area_vector.shape[-1:] == (3,):
        return area_vector, 3
    if kinematics.dimension == 2 and area_vector.shape[-1:] == (2,):
        embedded = jnp.zeros(area_vector.shape[:-1] + (3,), dtype=area_vector.dtype)
        return embedded.at[..., :2].set(area_vector), 2
    raise ValueError(
        "Nanson area vectors must have three components, or two for plane strain."
    )


def _restore_area_vector(vector: Array, dimension: int, /) -> Array:
    return vector if dimension == 3 else vector[..., :2]


def nanson_transform(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    reference_area_vector: ArrayLike,
    /,
) -> Array:
    r"""Push an oriented reference-area vector into the current frame.

    This is Nanson's formula ``n da = J F^{-T} N dA``. Inputs and outputs are
    signed, oriented area vectors rather than unit normals; no normal sign is
    selected or changed. The map is reference-to-current and is non-finite when
    ``F`` is not orientation preserving.
    """
    kinematics = finite_strain_kinematics(deformation_gradient)
    reference, dimension = _embedded_area_vector(reference_area_vector, kinematics)
    current = oe.contract("...ij,...j->...i", kinematics.cofactor, reference)
    current = jnp.where(
        kinematics.admissible[..., None], current, jnp.full_like(current, jnp.nan)
    )
    return _restore_area_vector(current, dimension)


def inverse_nanson_transform(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    current_area_vector: ArrayLike,
    /,
) -> Array:
    r"""Pull an oriented current-area vector back to the reference frame.

    This is the inverse of Nanson's orientation-preserving map,
    ``N dA = J^{-1} F^T n da``. It preserves the caller's oriented-vector sign
    and performs a current-to-reference frame transformation.
    """
    kinematics = finite_strain_kinematics(deformation_gradient)
    current, dimension = _embedded_area_vector(current_area_vector, kinematics)
    reference = (
        oe.contract("...ji,...j->...i", kinematics.deformation_gradient, current)
        / jnp.where(kinematics.admissible, kinematics.jacobian, 1.0)[..., None]
    )
    reference = jnp.where(
        kinematics.admissible[..., None],
        reference,
        jnp.full_like(reference, jnp.nan),
    )
    return _restore_area_vector(reference, dimension)


class NansonResponse(StrictModule):
    """Reference-to-current oriented-area evidence with explicit frame fields."""

    reference_area_vector: Array
    current_area_vector: Array
    area_ratio: Array
    admissible: Array


def nanson_response(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    reference_area_vector: ArrayLike,
    /,
) -> NansonResponse:
    """Evaluate Nanson's signed reference-to-current map and area-scale evidence."""
    kinematics = finite_strain_kinematics(deformation_gradient)
    reference = jnp.asarray(reference_area_vector)
    current = nanson_transform(kinematics, reference)
    reference_norm = jnp.sqrt(jnp.sum(reference * reference, axis=-1))
    current_norm = jnp.sqrt(jnp.sum(current * current, axis=-1))
    valid_vector = (
        jnp.all(jnp.isfinite(reference), axis=-1)
        & jnp.isfinite(reference_norm)
        & (reference_norm > 0.0)
    )
    admissible = kinematics.admissible & valid_vector & jnp.isfinite(current_norm)
    area_ratio = current_norm / jnp.where(valid_vector, reference_norm, 1.0)
    area_ratio = jnp.where(admissible, area_ratio, jnp.nan)
    return NansonResponse(reference, current, area_ratio, admissible)


def first_piola_to_cauchy(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    first_piola: ArrayLike,
    /,
) -> Array:
    r"""Push reference-frame first Piola stress to current-frame Cauchy stress.

    The orientation-preserving Nanson relation is ``sigma = J^{-1} P F^T``.
    ``P`` is a reference-to-current two-point tensor and ``sigma`` acts in the
    current frame. No traction or normal sign convention is silently reversed.
    """
    kinematics = finite_strain_kinematics(deformation_gradient)
    stress = jnp.asarray(first_piola)
    if stress.shape[-2:] != (3, 3):
        raise ValueError("First Piola stress must end in 3x3 in the embedded frame.")
    cauchy = (
        oe.contract("...ij,...kj->...ik", stress, kinematics.deformation_gradient)
        / jnp.where(kinematics.admissible, kinematics.jacobian, 1.0)[..., None, None]
    )
    return jnp.where(
        kinematics.admissible[..., None, None],
        cauchy,
        jnp.full_like(cauchy, jnp.nan),
    )


def cauchy_to_first_piola(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    cauchy_stress: ArrayLike,
    /,
) -> Array:
    r"""Pull current-frame Cauchy stress to reference-frame first Piola stress.

    The orientation-preserving inverse Nanson relation is
    ``P = J sigma F^{-T}``; it preserves the traction/normal sign convention.
    """
    kinematics = finite_strain_kinematics(deformation_gradient)
    stress = jnp.asarray(cauchy_stress)
    if stress.shape[-2:] != (3, 3):
        raise ValueError("Cauchy stress must end in 3x3 in the embedded frame.")
    first_piola = kinematics.jacobian[..., None, None] * oe.contract(
        "...ij,...jk->...ik", stress, kinematics.inverse_transpose
    )
    return jnp.where(
        kinematics.admissible[..., None, None],
        first_piola,
        jnp.full_like(first_piola, jnp.nan),
    )


class HyperelasticResponse(StrictModule):
    """Reference-energy/stress/tangent response with numerical admissibility evidence."""

    kinematics: FiniteStrainKinematics
    reference_energy_density: Array
    first_piola: Array
    cauchy_stress: Array
    tangent: Array
    kinematic_admissible: Array
    material_admissible: Array
    admissible: Array

    @property
    def material_tangent(self) -> Array:
        return self.tangent


class HyperelasticLaw(StrictModule, NonTrainableState):
    """Abstract pure pointwise hyperelastic constitutive law."""

    __strict_abstract__ = True

    @abc.abstractmethod
    def evaluate(self, deformation_gradient: ArrayLike, /) -> HyperelasticResponse:
        raise NotImplementedError


class NeoHookeanParameters(StrictModule, NonTrainableState):
    """Logarithmic compressible Neo-Hookean Lamé parameters."""

    shear_modulus: Array
    lame_lambda: Array

    def __init__(self, shear_modulus: ArrayLike, lame_lambda: ArrayLike, /):
        shear = jnp.asarray(shear_modulus)
        lambda_ = jnp.asarray(lame_lambda)
        bulk = lambda_ + (2.0 / 3.0) * shear
        if (
            shear.shape != ()
            or lambda_.shape != ()
            or not bool(jnp.isfinite(shear))
            or not bool(jnp.isfinite(lambda_))
            or bool(shear <= 0.0)
            or bool(bulk <= 0.0)
        ):
            raise ValueError(
                "Neo-Hookean shear modulus and implied bulk modulus must be "
                "positive finite scalars."
            )
        self.shear_modulus = shear
        self.lame_lambda = lambda_

    @classmethod
    def from_shear_bulk(
        cls,
        shear_modulus: ArrayLike,
        bulk_modulus: ArrayLike,
        /,
    ) -> "NeoHookeanParameters":
        shear = jnp.asarray(shear_modulus)
        bulk = jnp.asarray(bulk_modulus)
        if (
            shear.shape != ()
            or bulk.shape != ()
            or not bool(jnp.isfinite(shear))
            or not bool(jnp.isfinite(bulk))
            or bool(shear <= 0.0)
            or bool(bulk <= 0.0)
        ):
            raise ValueError(
                "Neo-Hookean shear and bulk moduli must be positive finite scalars."
            )
        return cls(shear, bulk - (2.0 / 3.0) * shear)

    @property
    def bulk_modulus(self) -> Array:
        return self.lame_lambda + (2.0 / 3.0) * self.shear_modulus


def _neo_hookean_material_values(
    shear_modulus: ArrayLike,
    lame_lambda: ArrayLike,
    /,
    *,
    batch_shape: tuple[int, ...],
) -> tuple[Array, Array, Array]:
    shear = jnp.asarray(shear_modulus)
    lambda_ = jnp.asarray(lame_lambda)
    if jnp.iscomplexobj(shear) or jnp.iscomplexobj(lambda_):
        raise TypeError("Neo-Hookean material values must be real.")
    broadcast_shape = jnp.broadcast_shapes(
        shear.shape,
        lambda_.shape,
        batch_shape,
    )
    if broadcast_shape != batch_shape:
        raise ValueError(
            "Neo-Hookean material values must be scalar fields on the deformation batch."
        )
    shear = jnp.broadcast_to(shear, batch_shape)
    lambda_ = jnp.broadcast_to(lambda_, batch_shape)
    bulk = lambda_ + (2.0 / 3.0) * shear
    admissible = (
        jnp.isfinite(shear) & jnp.isfinite(lambda_) & (shear > 0.0) & (bulk > 0.0)
    )
    return shear, lambda_, admissible


def _neo_hookean_state(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    shear_modulus: ArrayLike,
    lame_lambda: ArrayLike,
    /,
) -> tuple[FiniteStrainKinematics, Array, Array, Array, Array]:
    kinematics = finite_strain_kinematics(deformation_gradient)
    shear, lambda_, material_admissible = _neo_hookean_material_values(
        shear_modulus,
        lame_lambda,
        batch_shape=kinematics.jacobian.shape,
    )
    admissible = kinematics.admissible & material_admissible
    logarithm = jnp.log(jnp.where(admissible, kinematics.jacobian, 1.0))
    return kinematics, shear, lambda_, material_admissible, logarithm


def neo_hookean_reference_energy_from_moduli(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    shear_modulus: ArrayLike,
    lame_lambda: ArrayLike,
    /,
) -> Array:
    """Evaluate the canonical reference-volume Neo-Hookean energy kernel."""
    kinematics, shear, lambda_, material_admissible, logarithm = _neo_hookean_state(
        deformation_gradient, shear_modulus, lame_lambda
    )
    first_invariant = oe.contract(
        "...ij,...ij->...",
        kinematics.deformation_gradient,
        kinematics.deformation_gradient,
    )
    energy = (
        0.5 * shear * (first_invariant - 3.0)
        - shear * logarithm
        + 0.5 * lambda_ * logarithm * logarithm
    )
    admissible = kinematics.admissible & material_admissible
    return jnp.where(admissible, energy, jnp.nan)


def neo_hookean_first_piola_from_moduli(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    shear_modulus: ArrayLike,
    lame_lambda: ArrayLike,
    /,
) -> Array:
    """Evaluate the canonical embedded-frame Neo-Hookean first Piola kernel."""
    kinematics, shear, lambda_, material_admissible, logarithm = _neo_hookean_state(
        deformation_gradient, shear_modulus, lame_lambda
    )
    inverse_transpose = kinematics.inverse_transpose
    stress = (
        shear[..., None, None] * (kinematics.deformation_gradient - inverse_transpose)
        + (lambda_ * logarithm)[..., None, None] * inverse_transpose
    )
    admissible = kinematics.admissible & material_admissible
    return jnp.where(admissible[..., None, None], stress, jnp.full_like(stress, jnp.nan))


def neo_hookean_cauchy_from_moduli(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    shear_modulus: ArrayLike,
    lame_lambda: ArrayLike,
    /,
) -> Array:
    """Evaluate Cauchy stress through the canonical Nanson stress transform."""
    kinematics = finite_strain_kinematics(deformation_gradient)
    first_piola = neo_hookean_first_piola_from_moduli(
        kinematics, shear_modulus, lame_lambda
    )
    return first_piola_to_cauchy(kinematics, first_piola)


def neo_hookean_tangent_from_moduli(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    shear_modulus: ArrayLike,
    lame_lambda: ArrayLike,
    /,
) -> Array:
    r"""Evaluate ``dP_iJ/dF_kL`` for the canonical first Piola kernel."""
    kinematics, shear, lambda_, material_admissible, logarithm = _neo_hookean_state(
        deformation_gradient, shear_modulus, lame_lambda
    )
    inverse_transpose = kinematics.inverse_transpose
    identity = jnp.eye(3, dtype=kinematics.deformation_gradient.dtype)
    direct = shear[..., None, None, None, None] * oe.contract(
        "ik,jl->ijkl",
        identity,
        identity,
    )
    volumetric = lambda_[..., None, None, None, None] * oe.contract(
        "...ij,...kl->...ijkl",
        inverse_transpose,
        inverse_transpose,
    )
    geometric = (shear - lambda_ * logarithm)[..., None, None, None, None] * oe.contract(
        "...il,...kj->...ijkl",
        inverse_transpose,
        inverse_transpose,
    )
    tangent = direct + volumetric + geometric
    admissible = kinematics.admissible & material_admissible
    return jnp.where(
        admissible[..., None, None, None, None],
        tangent,
        jnp.full_like(tangent, jnp.nan),
    )


def neo_hookean_response_from_moduli(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    shear_modulus: ArrayLike,
    lame_lambda: ArrayLike,
    /,
) -> HyperelasticResponse:
    """Evaluate all canonical Neo-Hookean observables from scalar Lamé moduli."""
    kinematics = finite_strain_kinematics(deformation_gradient)
    _, _, material_admissible = _neo_hookean_material_values(
        shear_modulus,
        lame_lambda,
        batch_shape=kinematics.jacobian.shape,
    )
    energy = neo_hookean_reference_energy_from_moduli(
        kinematics, shear_modulus, lame_lambda
    )
    first_piola = neo_hookean_first_piola_from_moduli(
        kinematics, shear_modulus, lame_lambda
    )
    cauchy = first_piola_to_cauchy(kinematics, first_piola)
    tangent = neo_hookean_tangent_from_moduli(kinematics, shear_modulus, lame_lambda)
    admissible = kinematics.admissible & material_admissible
    return HyperelasticResponse(
        kinematics,
        energy,
        first_piola,
        cauchy,
        tangent,
        kinematics.admissible,
        material_admissible,
        admissible,
    )


def neo_hookean_reference_energy(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    parameters: NeoHookeanParameters,
    /,
) -> Array:
    if not isinstance(parameters, NeoHookeanParameters):
        raise TypeError("parameters must be NeoHookeanParameters.")
    return neo_hookean_reference_energy_from_moduli(
        deformation_gradient, parameters.shear_modulus, parameters.lame_lambda
    )


def neo_hookean_first_piola(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    parameters: NeoHookeanParameters,
    /,
) -> Array:
    if not isinstance(parameters, NeoHookeanParameters):
        raise TypeError("parameters must be NeoHookeanParameters.")
    return neo_hookean_first_piola_from_moduli(
        deformation_gradient, parameters.shear_modulus, parameters.lame_lambda
    )


def neo_hookean_cauchy(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    parameters: NeoHookeanParameters,
    /,
) -> Array:
    if not isinstance(parameters, NeoHookeanParameters):
        raise TypeError("parameters must be NeoHookeanParameters.")
    return neo_hookean_cauchy_from_moduli(
        deformation_gradient, parameters.shear_modulus, parameters.lame_lambda
    )


def neo_hookean_tangent(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    parameters: NeoHookeanParameters,
    /,
) -> Array:
    if not isinstance(parameters, NeoHookeanParameters):
        raise TypeError("parameters must be NeoHookeanParameters.")
    return neo_hookean_tangent_from_moduli(
        deformation_gradient, parameters.shear_modulus, parameters.lame_lambda
    )


def neo_hookean_response(
    deformation_gradient: FiniteStrainKinematics | ArrayLike,
    parameters: NeoHookeanParameters,
    /,
) -> HyperelasticResponse:
    if not isinstance(parameters, NeoHookeanParameters):
        raise TypeError("parameters must be NeoHookeanParameters.")
    return neo_hookean_response_from_moduli(
        deformation_gradient, parameters.shear_modulus, parameters.lame_lambda
    )


class NeoHookeanLaw(HyperelasticLaw):
    """Canonical logarithmic compressible Neo-Hookean constitutive law."""

    parameters: NeoHookeanParameters

    def __init__(self, parameters: NeoHookeanParameters, /):
        if not isinstance(parameters, NeoHookeanParameters):
            raise TypeError("parameters must be NeoHookeanParameters.")
        self.parameters = parameters

    def evaluate(self, deformation_gradient: ArrayLike, /) -> HyperelasticResponse:
        return neo_hookean_response(deformation_gradient, self.parameters)


__all__ = [
    "FiniteStrainKinematics",
    "HyperelasticLaw",
    "HyperelasticResponse",
    "NansonResponse",
    "NeoHookeanLaw",
    "NeoHookeanParameters",
    "VolumetricConstraint",
    "VolumetricConstraintKind",
    "cauchy_to_first_piola",
    "finite_strain_kinematics",
    "first_piola_to_cauchy",
    "inverse_nanson_transform",
    "logarithmic_volumetric_constraint",
    "nanson_response",
    "nanson_transform",
    "neo_hookean_cauchy",
    "neo_hookean_cauchy_from_moduli",
    "neo_hookean_first_piola",
    "neo_hookean_first_piola_from_moduli",
    "neo_hookean_reference_energy",
    "neo_hookean_reference_energy_from_moduli",
    "neo_hookean_response",
    "neo_hookean_response_from_moduli",
    "neo_hookean_tangent",
    "neo_hookean_tangent_from_moduli",
    "plane_strain_embedding",
    "volumetric_constraint",
]
