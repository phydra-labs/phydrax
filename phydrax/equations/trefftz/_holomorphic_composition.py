#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._holomorphic import (
    HolomorphicJet,
    HolomorphicMapCertificate,
    HolomorphicPotentialProvider,
)
from ..._holomorphic_linear import (
    HolomorphicMultiIndexSet,
    HolomorphicMultiJet,
    MultivariableHolomorphicPotentialProvider,
)
from ..._holomorphic_taylor import (
    multijet_from_normalized,
    normalized_coefficients,
    taylor_multiply,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _certificates(
    providers: tuple[Any, ...],
    /,
) -> tuple[HolomorphicMapCertificate, ...]:
    if not providers:
        raise ValueError("Holomorphic composition requires at least one provider.")
    certificates = []
    for provider in providers:
        if not isinstance(provider, HolomorphicPotentialProvider):
            raise TypeError(
                "Holomorphic composition children must implement "
                "HolomorphicPotentialProvider."
            )
        certificates.append(provider.holomorphic_certificate())
    return tuple(certificates)


def _aggregate_operations(
    certificates: tuple[HolomorphicMapCertificate, ...],
    *additional: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            operation
            for certificate in certificates
            for operation in certificate.operations
        )
    ) + tuple(additional)


def _aggregate_normalization_id(
    kind: str,
    certificates: tuple[HolomorphicMapCertificate, ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "normalizations": [
                certificate.normalization_id for certificate in certificates
            ],
        }
    )


class HolomorphicFactorizationEvidence(StrictModule, NonTrainableState):
    """Static composition and gauge contract for one holomorphic factorization."""

    factorization_kind: str = eqx.field(static=True)
    factor_count: int = eqx.field(static=True)
    latent_rank: int | None = eqx.field(static=True)
    branch_count: int = eqx.field(static=True)
    coordinate_mode: str = eqx.field(static=True)
    child_certificate_ids: tuple[str, ...] = eqx.field(static=True)
    gauge_kind: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        factorization_kind: str,
        factor_count: int,
        latent_rank: int | None,
        branch_count: int,
        coordinate_mode: str,
        child_certificate_ids: Sequence[str],
        gauge_kind: str,
    ):
        factors = int(factor_count)
        branches = int(branch_count)
        rank = None if latent_rank is None else int(latent_rank)
        children = tuple(str(value) for value in child_certificate_ids)
        identifiers = (
            str(factorization_kind),
            str(coordinate_mode),
            str(gauge_kind),
        )
        if factors <= 0 or branches <= 0 or (rank is not None and rank <= 0):
            raise ValueError("Holomorphic factorization counts must be positive.")
        if len(children) != factors or any(not value for value in children):
            raise ValueError(
                "Holomorphic factorization requires one child certificate per factor."
            )
        if any(not value for value in identifiers):
            raise ValueError("Holomorphic factorization identifiers must be non-empty.")
        self.factorization_kind = identifiers[0]
        self.factor_count = factors
        self.latent_rank = rank
        self.branch_count = branches
        self.coordinate_mode = identifiers[1]
        self.child_certificate_ids = children
        self.gauge_kind = identifiers[2]
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "holomorphic-factorization-evidence",
                "factorization_kind": identifiers[0],
                "factor_count": factors,
                "latent_rank": rank,
                "branch_count": branches,
                "coordinate_mode": identifiers[1],
                "child_certificate_ids": list(children),
                "gauge_kind": identifiers[2],
            }
        )


class HolomorphicFactorGaugeReport(StrictModule, NonTrainableState):
    """Pointwise factor-norm imbalance evidence for a product potential."""

    finite: Array
    factor_norms: Array
    minimum_factor_norm: Array
    maximum_factor_norm: Array
    imbalance_ratio: Array

    def __init__(self, factor_norms: ArrayLike, /):
        norms = jnp.asarray(factor_norms)
        if norms.ndim != 1 or norms.size == 0:
            raise ValueError("Holomorphic factor norms must be one nonempty vector.")
        finite = jnp.all(jnp.isfinite(norms))
        minimum = jnp.min(norms)
        maximum = jnp.max(norms)
        tiny = jnp.finfo(norms.dtype).tiny
        self.finite = finite
        self.factor_norms = norms
        self.minimum_factor_norm = minimum
        self.maximum_factor_norm = maximum
        self.imbalance_ratio = jnp.where(
            maximum == 0,
            jnp.asarray(1.0, dtype=norms.dtype),
            maximum / jnp.maximum(minimum, tiny),
        )


class HolomorphicBranchBundle(StrictModule):
    """Ordered independent scalar-input holomorphic potential branches."""

    __hash__ = object.__hash__

    providers: tuple[Any, ...]
    branch_offsets: tuple[int, ...] = eqx.field(static=True)
    factorization: HolomorphicFactorizationEvidence
    _certificate: HolomorphicMapCertificate

    def __init__(self, providers: Sequence[HolomorphicPotentialProvider], /):
        resolved = tuple(providers)
        certificates = _certificates(resolved)
        if any(certificate.complex_input_size != 1 for certificate in certificates):
            raise ValueError(
                "Initial HolomorphicBranchBundle support requires one complex input."
            )
        offsets = [0]
        for certificate in certificates:
            offsets.append(offsets[-1] + certificate.complex_output_size)
        output_size = offsets[-1]
        factorization = HolomorphicFactorizationEvidence(
            factorization_kind="independent-branch-bundle",
            factor_count=len(resolved),
            latent_rank=None,
            branch_count=output_size,
            coordinate_mode="same-scalar-coordinate",
            child_certificate_ids=tuple(
                certificate.certificate_id for certificate in certificates
            ),
            gauge_kind="none",
        )
        linear = all(certificate.linear_in_parameters for certificate in certificates)
        finite_subspace = all(
            certificate.parameter_coverage == "finite-subspace"
            for certificate in certificates
        )
        self.providers = resolved
        self.branch_offsets = tuple(offsets)
        self.factorization = factorization
        self._certificate = HolomorphicMapCertificate(
            complex_input_size=1,
            complex_output_size=output_size,
            construction="independent-holomorphic-branch-bundle",
            normalization_id=_aggregate_normalization_id(
                "holomorphic-branch-normalizations", certificates
            ),
            maximum_derivative_order=min(
                certificate.maximum_derivative_order for certificate in certificates
            ),
            operations=_aggregate_operations(
                certificates, "holomorphic-branch-concatenation"
            ),
            parameter_coverage=(
                "finite-subspace" if finite_subspace else "finite-parametric-family"
            ),
            linear_in_parameters=linear,
            construction_dependencies=(factorization.evidence_id,),
        )

    def __call__(self, coordinate: Array, /) -> Array:
        values = tuple(
            jnp.asarray(provider(coordinate)).reshape((-1,))
            for provider in self.providers
        )
        return jnp.concatenate(values, axis=0)

    def jet(self, coordinate: Array, order: int, /) -> HolomorphicJet:
        order_ = int(order)
        if order_ < 0 or order_ > self._certificate.maximum_derivative_order:
            raise ValueError("Requested branch-bundle jet order is unavailable.")
        jets = tuple(provider.jet(coordinate, order_) for provider in self.providers)
        value = jnp.concatenate(tuple(jet.value.reshape((-1,)) for jet in jets))
        derivatives = tuple(
            jnp.concatenate(tuple(jet.derivative(current).reshape((-1,)) for jet in jets))
            for current in range(1, order_ + 1)
        )
        return HolomorphicJet(value, derivatives)

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._certificate


class HolomorphicProductPotential(StrictModule):
    """Finite sum of same-coordinate products of certified holomorphic factors."""

    __hash__ = object.__hash__

    factors: tuple[Any, ...]
    latent_rank: int = eqx.field(static=True)
    branches: int = eqx.field(static=True)
    factorization: HolomorphicFactorizationEvidence
    _certificate: HolomorphicMapCertificate

    def __init__(
        self,
        factors: Sequence[HolomorphicPotentialProvider],
        /,
        *,
        latent_rank: int,
        branches: int,
    ):
        resolved = tuple(factors)
        certificates = _certificates(resolved)
        rank = int(latent_rank)
        branches_ = int(branches)
        if rank <= 0 or branches_ <= 0:
            raise ValueError("Product potential rank and branches must be positive.")
        expected_output = rank * branches_
        input_sizes = {certificate.complex_input_size for certificate in certificates}
        if len(input_sizes) != 1:
            raise ValueError("Every product factor must use the same complex input size.")
        input_size = next(iter(input_sizes))
        for certificate in certificates:
            if certificate.complex_output_size != expected_output:
                raise ValueError(
                    "Every product factor must output latent_rank * branches values."
                )
        factorization = HolomorphicFactorizationEvidence(
            factorization_kind="same-coordinate-holomorphic-product",
            factor_count=len(resolved),
            latent_rank=rank,
            branch_count=branches_,
            coordinate_mode=(
                "same-scalar-coordinate"
                if input_size == 1
                else "same-complex-vector-coordinate"
            ),
            child_certificate_ids=tuple(
                certificate.certificate_id for certificate in certificates
            ),
            gauge_kind=("none" if len(resolved) == 1 else "multiplicative-factor-scale"),
        )
        if len(resolved) == 1:
            coverage = certificates[0].parameter_coverage
            linear = certificates[0].linear_in_parameters
        else:
            coverage = "finite-parametric-family"
            linear = False
        self.factors = resolved
        self.latent_rank = rank
        self.branches = branches_
        self.factorization = factorization
        self._certificate = HolomorphicMapCertificate(
            complex_input_size=input_size,
            complex_output_size=branches_,
            construction="same-coordinate-holomorphic-product-potential",
            normalization_id=_aggregate_normalization_id(
                "holomorphic-product-normalizations", certificates
            ),
            maximum_derivative_order=min(
                certificate.maximum_derivative_order for certificate in certificates
            ),
            operations=_aggregate_operations(
                certificates,
                "complex-multiplication",
                "finite-complex-summation",
            ),
            parameter_coverage=coverage,
            linear_in_parameters=linear,
            construction_dependencies=(factorization.evidence_id,),
        )

    def _factor_value(self, factor: Any, coordinate: Array, /) -> Array:
        values = jnp.asarray(factor(coordinate))
        expected = self.latent_rank * self.branches
        if values.size != expected:
            raise ValueError("Holomorphic product factor returned the wrong value count.")
        return values.reshape((self.latent_rank, self.branches))

    def __call__(self, coordinate: Array, /) -> Array:
        factor_values = tuple(
            self._factor_value(factor, coordinate) for factor in self.factors
        )
        product = jnp.ones(
            (self.latent_rank, self.branches),
            dtype=jnp.result_type(*(value.dtype for value in factor_values)),
        )
        for value in factor_values:
            product = product * value
        return jnp.sum(product, axis=0)

    def jet(self, coordinate: Array, order: int, /) -> HolomorphicJet:
        order_ = int(order)
        if order_ < 0 or order_ > self._certificate.maximum_derivative_order:
            raise ValueError("Requested product-potential jet order is unavailable.")
        factor_jets = tuple(factor.jet(coordinate, order_) for factor in self.factors)
        dtype = jnp.result_type(*(jet.value.dtype for jet in factor_jets))
        coefficients = [jnp.ones((self.latent_rank, self.branches), dtype=dtype)] + [
            jnp.zeros((self.latent_rank, self.branches), dtype=dtype)
            for _ in range(order_)
        ]
        for factor_jet in factor_jets:
            factor_coefficients = [
                factor_jet.derivative(current).reshape((self.latent_rank, self.branches))
                / math.factorial(current)
                for current in range(order_ + 1)
            ]
            next_coefficients = []
            for current in range(order_ + 1):
                value = jnp.zeros_like(coefficients[0])
                for factor_order in range(current + 1):
                    value = value + (
                        coefficients[current - factor_order]
                        * factor_coefficients[factor_order]
                    )
                next_coefficients.append(value)
            coefficients = next_coefficients
        values = tuple(
            math.factorial(current) * jnp.sum(coefficient, axis=0)
            for current, coefficient in enumerate(coefficients)
        )
        return HolomorphicJet(values[0], values[1:])

    def multi_jet(
        self,
        coordinates: Array,
        index_set: HolomorphicMultiIndexSet,
        /,
    ) -> HolomorphicMultiJet:
        if not isinstance(index_set, HolomorphicMultiIndexSet):
            raise TypeError("index_set must be HolomorphicMultiIndexSet.")
        if index_set.complex_dimension != self._certificate.complex_input_size:
            raise ValueError("Product potential and multijet dimensions differ.")
        if index_set.maximum_total_order > self._certificate.maximum_derivative_order:
            raise ValueError("Requested product-potential multijet is unavailable.")
        factor_coefficients = []
        for factor in self.factors:
            if not isinstance(factor, MultivariableHolomorphicPotentialProvider):
                raise TypeError(
                    "Every multivariable product factor must provide multijets."
                )
            coefficients = normalized_coefficients(
                factor.multi_jet(coordinates, index_set)
            )
            factor_coefficients.append(
                coefficients.reshape((index_set.count, self.latent_rank, self.branches))
            )
        product = jnp.zeros_like(factor_coefficients[0])
        zero_position = index_set.indices.index((0,) * index_set.complex_dimension)
        product = product.at[zero_position].set(jnp.ones_like(product[zero_position]))
        for coefficients in factor_coefficients:
            product = taylor_multiply(product, coefficients, index_set)
        summed = jnp.sum(product, axis=1)
        return multijet_from_normalized(summed, index_set)

    def gauge_report(self, coordinate: Array, /) -> HolomorphicFactorGaugeReport:
        norms = jnp.asarray(
            [
                jnp.sqrt(jnp.mean(jnp.abs(self._factor_value(factor, coordinate)) ** 2))
                for factor in self.factors
            ]
        )
        return HolomorphicFactorGaugeReport(norms)

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._certificate


__all__ = [
    "HolomorphicBranchBundle",
    "HolomorphicFactorGaugeReport",
    "HolomorphicFactorizationEvidence",
    "HolomorphicProductPotential",
]
