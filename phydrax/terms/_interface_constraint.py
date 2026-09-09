#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractScalarTerm
from .._trainable import NonTrainableState
from ..domain import DomainFunction, LocalFieldRef, PairedSupport


FieldBinding = str | LocalFieldRef


def _name(value: FieldBinding, /) -> str:
    return value.solver_name if isinstance(value, LocalFieldRef) else str(value)


def _normal_contract(flux: Array, normal: Array, /) -> Array:
    flux_ = jnp.asarray(flux)
    normal_ = jnp.asarray(normal)
    if flux_.ndim == 0 or flux_.shape[-1:] != normal_.shape[-1:]:
        return flux_ * normal_
    return ein.contract("...i,...i->...", flux_, normal_)


class MortarInterfaceEvidence(StrictModule, NonTrainableState):
    minimum_gram_eigenvalue: float = eqx.field(static=True)
    gram_condition: float = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_gram_eigenvalue: float,
        gram_condition: float,
        verified: bool,
    ):
        self.minimum_gram_eigenvalue = float(minimum_gram_eigenvalue)
        self.gram_condition = float(gram_condition)
        self.verified = bool(verified)


class MortarInterfacePenalty(AbstractScalarTerm):
    """Riesz-weighted weak value-jump moments on one fixed paired support."""

    pairing: PairedSupport
    points: Any
    basis_values: Array
    weights: Array
    gram_inverse: Array
    target: Array
    evidence: MortarInterfaceEvidence
    scale: Array
    fields: tuple[str, ...] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        left_field: FieldBinding,
        right_field: FieldBinding,
        pairing: PairedSupport,
        points: Any,
        basis_values: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        target: ArrayLike = 0.0,
        scale: float = 1.0,
        label: str | None = None,
        gram_tolerance: float = 1.0e-12,
    ):
        basis = np.asarray(basis_values, dtype=float)
        if basis.ndim != 2 or basis.shape[0] <= 0 or basis.shape[1] <= 0:
            raise ValueError("basis_values must have shape (points, modes).")
        weights_ = (
            np.full((basis.shape[0],), 1.0 / basis.shape[0], dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if weights_.shape != (basis.shape[0],):
            raise ValueError("Mortar weights must have one value per point.")
        if np.any(weights_ <= 0.0) or not np.all(np.isfinite(weights_)):
            raise ValueError("Mortar weights must be finite and positive.")
        gram = basis.T @ (weights_[:, None] * basis)
        eigenvalues = np.linalg.eigvalsh(gram)
        minimum = float(eigenvalues[0])
        condition = float(eigenvalues[-1] / eigenvalues[0]) if minimum > 0.0 else np.inf
        verified = bool(minimum > float(gram_tolerance) and np.isfinite(condition))
        if not verified:
            raise ValueError("Mortar basis Gram matrix is singular or ill-conditioned.")
        gram_inverse = np.linalg.solve(gram, np.eye(gram.shape[0]))
        scale_ = jnp.asarray(scale, dtype=float).reshape(())
        if float(scale_) < 0.0:
            raise ValueError("scale must be non-negative.")
        self.fields = (_name(left_field), _name(right_field))
        self.pairing = pairing
        self.points = points
        self.basis_values = jnp.asarray(basis)
        self.weights = jnp.asarray(weights_)
        self.gram_inverse = jnp.asarray(gram_inverse)
        self.target = jnp.asarray(target)
        self.evidence = MortarInterfaceEvidence(
            minimum_gram_eigenvalue=minimum,
            gram_condition=condition,
            verified=verified,
        )
        self.scale = scale_
        self.label = None if label is None else str(label)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        left = self.pairing.trace(functions[self.fields[0]], side="left")(
            self.points, key=key
        ).data
        right = self.pairing.trace(functions[self.fields[1]], side="right")(
            self.points, key=key
        ).data
        jump = jnp.asarray(right) - jnp.asarray(left) - self.target
        moments = ein.contract(
            "nm,n...,n->m...",
            self.basis_values,
            jump,
            self.weights,
        )
        flat = moments.reshape((moments.shape[0], -1))
        riesz = ein.contract("mn,nk->mk", self.gram_inverse, flat)
        return self.scale * jnp.real(jnp.sum(jnp.conj(flat) * riesz))


class NitscheInterfaceFunctional(AbstractScalarTerm):
    """Symmetric scalar Nitsche consistency and stabilization functional."""

    pairing: PairedSupport
    points: Any
    left_flux: Callable[[DomainFunction], DomainFunction] = eqx.field(static=True)
    right_flux: Callable[[DomainFunction], DomainFunction] = eqx.field(static=True)
    penalty: Array
    fields: tuple[str, ...] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        left_field: FieldBinding,
        right_field: FieldBinding,
        pairing: PairedSupport,
        points: Any,
        left_flux: Callable[[DomainFunction], DomainFunction],
        right_flux: Callable[[DomainFunction], DomainFunction],
        /,
        *,
        penalty: float,
        label: str | None = None,
    ):
        if pairing.normal is None:
            raise ValueError("Nitsche coupling requires an oriented interface normal.")
        penalty_ = jnp.asarray(penalty, dtype=float).reshape(())
        if not bool(jnp.isfinite(penalty_)) or float(penalty_) <= 0.0:
            raise ValueError("Nitsche penalty must be finite and positive.")
        self.fields = (_name(left_field), _name(right_field))
        self.pairing = pairing
        self.points = points
        self.left_flux = left_flux
        self.right_flux = right_flux
        self.penalty = penalty_
        self.label = None if label is None else str(label)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        left_field = functions[self.fields[0]]
        right_field = functions[self.fields[1]]
        left = self.pairing.trace(left_field, side="left")(self.points, key=key).data
        right = self.pairing.trace(right_field, side="right")(self.points, key=key).data
        left_flux = self.pairing.trace(self.left_flux(left_field), side="left")(
            self.points, key=key
        ).data
        right_flux = self.pairing.trace(self.right_flux(right_field), side="right")(
            self.points, key=key
        ).data
        normal = self.pairing.normal
        if normal is None:
            raise RuntimeError("Validated Nitsche pairing lost its normal.")
        normal_values = normal(self.points, key=key).data
        average_flux = 0.5 * (
            _normal_contract(left_flux, normal_values)
            + _normal_contract(right_flux, normal_values)
        )
        jump = jnp.asarray(right) - jnp.asarray(left)
        integrand = -2.0 * jnp.real(
            jnp.conj(average_flux) * jump
        ) + self.penalty * jnp.real(jnp.conj(jump) * jump)
        return jnp.mean(integrand)


class AugmentedInterfaceEvidence(StrictModule):
    primal_residual: Array
    dual_residual: Array
    iteration: int = eqx.field(static=True)

    def __init__(
        self,
        primal_residual: Array,
        dual_residual: Array,
        /,
        *,
        iteration: int,
    ):
        self.primal_residual = jnp.asarray(primal_residual).reshape(())
        self.dual_residual = jnp.asarray(dual_residual).reshape(())
        self.iteration = int(iteration)


class AugmentedValueConstraint(AbstractScalarTerm):
    """Fixed-trace augmented-Lagrangian value continuity constraint."""

    pairing: PairedSupport
    points: Any
    target: Array
    multiplier: Array
    previous_residual: Array
    penalty: Array
    fields: tuple[str, ...] = eqx.field(static=True)
    iteration: int = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        left_field: FieldBinding,
        right_field: FieldBinding,
        pairing: PairedSupport,
        points: Any,
        multiplier: ArrayLike,
        /,
        *,
        target: ArrayLike = 0.0,
        previous_residual: ArrayLike | None = None,
        penalty: float = 1.0,
        iteration: int = 0,
        label: str | None = None,
    ):
        multiplier_ = jnp.asarray(multiplier)
        target_ = jnp.asarray(target)
        previous = (
            jnp.zeros_like(multiplier_)
            if previous_residual is None
            else jnp.asarray(previous_residual)
        )
        if previous.shape != multiplier_.shape:
            raise ValueError("previous_residual must match multiplier shape.")
        penalty_ = jnp.asarray(penalty, dtype=float).reshape(())
        if not bool(jnp.isfinite(penalty_)) or float(penalty_) <= 0.0:
            raise ValueError("Augmented constraint penalty must be finite and positive.")
        self.fields = (_name(left_field), _name(right_field))
        self.pairing = pairing
        self.points = points
        self.target = target_
        self.multiplier = multiplier_
        self.previous_residual = previous
        self.penalty = penalty_
        self.iteration = int(iteration)
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        left = self.pairing.trace(functions[self.fields[0]], side="left")(
            self.points, key=key
        ).data
        right = self.pairing.trace(functions[self.fields[1]], side="right")(
            self.points, key=key
        ).data
        value = jnp.asarray(right) - jnp.asarray(left) - self.target
        if value.shape != self.multiplier.shape:
            raise ValueError("Interface residual and multiplier shapes must match.")
        return value

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        residual = self.residual(functions, key=key)
        linear = jnp.real(jnp.conj(self.multiplier) * residual)
        quadratic = 0.5 * self.penalty * jnp.real(jnp.conj(residual) * residual)
        return jnp.mean(linear + quadratic)

    def update(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[AugmentedValueConstraint, AugmentedInterfaceEvidence]:
        residual = self.residual(functions, key=key)
        multiplier = self.multiplier + self.penalty * residual
        primal = jnp.sqrt(jnp.mean(jnp.real(jnp.conj(residual) * residual)))
        difference = residual - self.previous_residual
        dual = self.penalty * jnp.sqrt(
            jnp.mean(jnp.real(jnp.conj(difference) * difference))
        )
        updated = AugmentedValueConstraint(
            self.fields[0],
            self.fields[1],
            self.pairing,
            self.points,
            multiplier,
            target=self.target,
            previous_residual=residual,
            penalty=float(self.penalty),
            iteration=self.iteration + 1,
            label=self.label,
        )
        return updated, AugmentedInterfaceEvidence(
            primal,
            dual,
            iteration=updated.iteration,
        )

    @classmethod
    def initialize(
        cls,
        left_field: FieldBinding,
        right_field: FieldBinding,
        pairing: PairedSupport,
        points: Any,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        target: ArrayLike = 0.0,
        penalty: float = 1.0,
        label: str | None = None,
    ) -> AugmentedValueConstraint:
        left_name = _name(left_field)
        right_name = _name(right_field)
        left = pairing.trace(functions[left_name], side="left")(points).data
        right = pairing.trace(functions[right_name], side="right")(points).data
        residual = jnp.asarray(right) - jnp.asarray(left) - jnp.asarray(target)
        return cls(
            left_name,
            right_name,
            pairing,
            points,
            jnp.zeros_like(residual),
            target=target,
            previous_residual=jnp.zeros_like(residual),
            penalty=penalty,
            label=label,
        )


__all__ = [
    "AugmentedInterfaceEvidence",
    "AugmentedValueConstraint",
    "MortarInterfaceEvidence",
    "MortarInterfacePenalty",
    "NitscheInterfaceFunctional",
]
