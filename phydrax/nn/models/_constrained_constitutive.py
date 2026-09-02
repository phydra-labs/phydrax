# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from .._keys import EvalKey
from ._constitutive import DeformationGradientMinors


class PolyconvexMaterialConstraints(StrictModule):
    """Static hypotheses for a structurally constrained polyconvex material."""

    gradient_exponents: tuple[float, ...] = eqx.field(static=True)
    cofactor_exponents: tuple[float, ...] = eqx.field(static=True)
    minimum_determinant: float = eqx.field(static=True)
    orientation_barrier: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        gradient_exponents: tuple[float, ...] = (2.0, 4.0),
        cofactor_exponents: tuple[float, ...] = (2.0,),
        minimum_determinant: float = 0.0,
        orientation_barrier: bool = True,
    ):
        gradient = tuple(float(value) for value in gradient_exponents)
        cofactor = tuple(float(value) for value in cofactor_exponents)
        if (
            not gradient
            or not cofactor
            or any(
                not jnp.isfinite(value) or value < 2.0 for value in gradient + cofactor
            )
        ):
            raise ValueError(
                "Polyconvex coercive exponents must be finite and at least two."
            )
        minimum = float(minimum_determinant)
        if not jnp.isfinite(minimum) or minimum < 0.0 or minimum >= 1.0:
            raise ValueError("minimum_determinant must be finite and lie in [0, 1).")
        self.gradient_exponents = gradient
        self.cofactor_exponents = cofactor
        self.minimum_determinant = minimum
        self.orientation_barrier = bool(orientation_barrier)


class ReferenceConfiguration(StrictModule, NonTrainableState):
    """Positive-orientation constitutive reference with derived audited geometry."""

    _deformation_gradient: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(self, deformation_gradient: ArrayLike, /):
        gradient = jnp.asarray(deformation_gradient)
        if gradient.ndim != 2 or gradient.shape[0] != gradient.shape[1]:
            raise ValueError("Reference deformation gradient must be square.")
        dimension = int(gradient.shape[0])
        if dimension not in (2, 3):
            raise ValueError("ReferenceConfiguration supports dimension two or three.")
        if not jnp.issubdtype(gradient.dtype, jnp.floating):
            raise TypeError("Reference deformation gradient must be real floating point.")
        solve = solve_small_linear(
            SmallLinearSolvePlan(dimension),
            gradient,
            jnp.eye(dimension, dtype=gradient.dtype),
        )
        if not bool(solve.successful) or not bool(solve.determinant > 0.0):
            raise ValueError(
                "Reference deformation gradient must be finite, full rank, and "
                "positive orientation."
            )
        self._deformation_gradient = tuple(
            tuple(float(value) for value in row) for row in gradient.tolist()
        )
        self._dtype = gradient.dtype.name
        self.dimension = dimension

    @property
    def deformation_gradient(self) -> Array:
        """Return the immutable audited reference deformation gradient."""
        return jnp.asarray(self._deformation_gradient, dtype=jnp.dtype(self._dtype))

    def _derived_geometry(self) -> tuple[Array, Array]:
        gradient = self.deformation_gradient
        solve = solve_small_linear(
            SmallLinearSolvePlan(self.dimension),
            gradient,
            jnp.eye(self.dimension, dtype=gradient.dtype),
        )
        return solve.value, solve.determinant

    @property
    def inverse(self) -> Array:
        """Return the inverse derived from the audited reference gradient."""
        inverse, _ = self._derived_geometry()
        return inverse

    @property
    def determinant(self) -> Array:
        """Return the determinant derived from the audited reference gradient."""
        _, determinant = self._derived_geometry()
        return determinant


class CoercivePolyconvexEnvelope(StrictModule):
    """Positive-coefficient convex energy in lifted relative minors."""

    raw_gradient_coefficients: Array
    raw_cofactor_coefficients: Array
    raw_determinant_coefficient: Array
    raw_barrier_coefficient: Array
    constraints: PolyconvexMaterialConstraints

    def __init__(
        self,
        constraints: PolyconvexMaterialConstraints,
        /,
        *,
        gradient_coefficients: ArrayLike = (1.0, 0.1),
        cofactor_coefficients: ArrayLike = (0.25,),
        determinant_coefficient: float = 1.0,
        barrier_coefficient: float = 0.01,
    ):
        if not isinstance(constraints, PolyconvexMaterialConstraints):
            raise TypeError("constraints must be PolyconvexMaterialConstraints.")
        gradient = jnp.asarray(gradient_coefficients, dtype=float)
        cofactor = jnp.asarray(cofactor_coefficients, dtype=float)
        if gradient.shape != (len(constraints.gradient_exponents),) or cofactor.shape != (
            len(constraints.cofactor_exponents),
        ):
            raise ValueError("Envelope coefficient counts must match the exponent lists.")
        scalars = jnp.asarray((determinant_coefficient, barrier_coefficient), dtype=float)
        if bool(jnp.any(~jnp.isfinite(gradient))) or bool(jnp.any(gradient <= 0.0)):
            raise ValueError("Gradient coefficients must be finite and positive.")
        if bool(jnp.any(~jnp.isfinite(cofactor))) or bool(jnp.any(cofactor <= 0.0)):
            raise ValueError("Cofactor coefficients must be finite and positive.")
        if bool(jnp.any(~jnp.isfinite(scalars))) or bool(jnp.any(scalars <= 0.0)):
            raise ValueError("Determinant and barrier coefficients must be positive.")
        inverse_softplus = lambda value: jnp.log(jnp.expm1(value))
        self.raw_gradient_coefficients = inverse_softplus(gradient)
        self.raw_cofactor_coefficients = inverse_softplus(cofactor)
        self.raw_determinant_coefficient = inverse_softplus(scalars[0])
        self.raw_barrier_coefficient = inverse_softplus(scalars[1])
        self.constraints = constraints

    @property
    def gradient_coefficients(self) -> Array:
        return jax.nn.softplus(self.raw_gradient_coefficients)

    @property
    def cofactor_coefficients(self) -> Array:
        return jax.nn.softplus(self.raw_cofactor_coefficients)

    @property
    def determinant_coefficient(self) -> Array:
        return jax.nn.softplus(self.raw_determinant_coefficient)

    @property
    def barrier_coefficient(self) -> Array:
        return jax.nn.softplus(self.raw_barrier_coefficient)

    def __call__(self, gradient: Array, cofactor: Array, determinant: Array, /) -> Array:
        gradient_norm = jnp.sum(gradient * gradient, axis=(-2, -1))
        cofactor_norm = jnp.sum(cofactor * cofactor, axis=(-2, -1))
        energy = jnp.zeros_like(determinant)
        derivative_at_reference = jnp.zeros_like(determinant)
        dimension = int(gradient.shape[-1])
        for coefficient, exponent in zip(
            self.gradient_coefficients,
            self.constraints.gradient_exponents,
            strict=True,
        ):
            energy = energy + coefficient * gradient_norm ** (0.5 * exponent)
            derivative_at_reference = (
                derivative_at_reference
                + coefficient * exponent * dimension ** (0.5 * exponent - 1.0)
            )
        for coefficient, exponent in zip(
            self.cofactor_coefficients,
            self.constraints.cofactor_exponents,
            strict=True,
        ):
            energy = energy + coefficient * cofactor_norm ** (0.5 * exponent)
            derivative_at_reference = derivative_at_reference + coefficient * exponent * (
                dimension - 1
            ) * dimension ** (0.5 * exponent - 1.0)
        energy = energy + self.determinant_coefficient * (determinant - 1.0) ** 2
        if self.constraints.orientation_barrier:
            margin = determinant - self.constraints.minimum_determinant
            reference_margin = 1.0 - self.constraints.minimum_determinant
            energy = energy - self.barrier_coefficient * jnp.log(margin)
            derivative_at_reference = (
                derivative_at_reference - self.barrier_coefficient / reference_margin
            )
        # Affine functions of lifted determinant preserve convexity.
        return energy - derivative_at_reference * (determinant - 1.0)


class MaterialConstraintReport(StrictModule):
    """Structural and numerical evidence for a constrained material."""

    objective: bool = eqx.field(static=True)
    isotropic: bool = eqx.field(static=True)
    polyconvex: bool = eqx.field(static=True)
    coercive: bool = eqx.field(static=True)
    orientation_preserving: bool = eqx.field(static=True)
    reference_determinant: Array
    orientation_margin: Array
    reference_energy: Array
    reference_stress_norm: Array
    reference_tangent_diagonal_minimum: Array


class ConstrainedPolyconvexPotential(StrictModule):
    """Objective, isotropic, coercive polyconvex energy relative to a reference."""

    reference: ReferenceConfiguration
    envelope: CoercivePolyconvexEnvelope
    minors: DeformationGradientMinors

    def __init__(
        self,
        reference: ReferenceConfiguration,
        envelope: CoercivePolyconvexEnvelope,
        /,
    ):
        if not isinstance(reference, ReferenceConfiguration):
            raise TypeError("reference must be a ReferenceConfiguration.")
        if not isinstance(envelope, CoercivePolyconvexEnvelope):
            raise TypeError("envelope must be a CoercivePolyconvexEnvelope.")
        self.reference = reference
        self.envelope = envelope
        self.minors = DeformationGradientMinors(reference.dimension)

    @property
    def dimension(self) -> int:
        return self.reference.dimension

    @property
    def reference_energy_shift(self) -> Array:
        """Return the current envelope energy at the stored reference."""
        relative = self.relative_gradient(self.reference.deformation_gradient)
        return self.envelope(
            relative,
            self.minors.cofactor(relative),
            self.minors.determinant(relative),
        )

    def relative_gradient(self, deformation_gradient: ArrayLike, /) -> Array:
        gradient = self.minors._validate(jnp.asarray(deformation_gradient))
        return contract("...ij,jk->...ik", gradient, self.reference.inverse)

    def __call__(
        self,
        deformation_gradient: ArrayLike,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        relative = self.relative_gradient(deformation_gradient)
        cofactor = self.minors.cofactor(relative)
        determinant = self.minors.determinant(relative)
        invalid = ~jnp.isfinite(determinant) | (
            determinant <= self.envelope.constraints.minimum_determinant
        )
        determinant = eqx.error_if(
            determinant,
            jnp.any(invalid),
            "Constrained polyconvex energy is undefined outside its orientation domain.",
        )
        return (
            self.envelope(relative, cofactor, determinant) - self.reference_energy_shift
        )

    def first_piola_stress(
        self,
        deformation_gradient: ArrayLike,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        gradient = self.minors._validate(jnp.asarray(deformation_gradient))
        flat = gradient.reshape((-1, self.dimension, self.dimension))
        derivative = jax.grad(lambda value: self(value, key=key))
        return jax.vmap(derivative)(flat).reshape(gradient.shape)

    def material_tangent(
        self,
        deformation_gradient: ArrayLike,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        gradient = self.minors._validate(jnp.asarray(deformation_gradient))
        flat = gradient.reshape((-1, self.dimension, self.dimension))
        derivative = jax.hessian(lambda value: self(value, key=key))
        return jax.vmap(derivative)(flat).reshape(
            gradient.shape[:-2]
            + (self.dimension, self.dimension, self.dimension, self.dimension)
        )

    def constraint_report(self) -> MaterialConstraintReport:
        reference = self.reference.deformation_gradient
        stress = self.first_piola_stress(reference)
        tangent = self.material_tangent(reference).reshape(
            self.dimension**2,
            self.dimension**2,
        )
        return MaterialConstraintReport(
            True,
            True,
            True,
            True,
            self.envelope.constraints.orientation_barrier,
            self.reference.determinant,
            jnp.asarray(1.0 - self.envelope.constraints.minimum_determinant),
            self(reference),
            jnp.sqrt(jnp.sum(stress * stress)),
            jnp.min(jnp.diag(tangent)),
        )


__all__ = [
    "CoercivePolyconvexEnvelope",
    "ConstrainedPolyconvexPotential",
    "MaterialConstraintReport",
    "PolyconvexMaterialConstraints",
    "ReferenceConfiguration",
]
