#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._holomorphic_linear import MultivariableHolomorphicPotentialProvider
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._kahler_potential import KahlerPotentialGeometry


class KahlerGaugeInvarianceReport(StrictModule, NonTrainableState):
    """Numerical validation of a construction-exact pluriharmonic Kähler gauge."""

    valid: Array
    maximum_complex_hessian_change: Array
    tolerance: Array
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_complex_hessian_change: ArrayLike,
        tolerance: ArrayLike,
        /,
        *,
        gauge_id: str,
    ):
        change = jnp.asarray(maximum_complex_hessian_change)
        tolerance_ = jnp.asarray(tolerance)
        if change.shape != () or tolerance_.shape != ():
            raise ValueError("Kähler gauge report values must be scalar.")
        if not str(gauge_id):
            raise ValueError("gauge_id must be nonempty.")
        self.valid = jnp.isfinite(change) & (change <= tolerance_)
        self.maximum_complex_hessian_change = change
        self.tolerance = tolerance_
        self.gauge_id = str(gauge_id)


class KahlerHolomorphicGauge(StrictModule):
    """Add the real part of a holomorphic map to one Kähler potential."""

    base: KahlerPotentialGeometry
    provider: MultivariableHolomorphicPotentialProvider
    branch: int = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: KahlerPotentialGeometry,
        provider: MultivariableHolomorphicPotentialProvider,
        /,
        *,
        branch: int = 0,
    ):
        if not isinstance(base, KahlerPotentialGeometry):
            raise TypeError("base must be KahlerPotentialGeometry.")
        if not isinstance(provider, MultivariableHolomorphicPotentialProvider):
            raise TypeError(
                "provider must implement MultivariableHolomorphicPotentialProvider."
            )
        certificate = provider.holomorphic_certificate()
        branch_ = int(branch)
        if certificate.complex_input_size != base.convention.complex_dimension:
            raise ValueError("Kähler gauge and complex convention dimensions differ.")
        if not 0 <= branch_ < certificate.complex_output_size:
            raise ValueError("Kähler gauge branch is invalid.")
        self.base = base
        self.provider = provider
        self.branch = branch_
        self.gauge_id = canonical_fingerprint(
            {
                "kind": "kahler-holomorphic-gauge",
                "holomorphic_certificate": certificate.certificate_id,
                "branch": branch_,
                "coordinate_pairs": [list(pair) for pair in base.convention.pairs],
            }
        )

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinates)
        base_value = jnp.asarray(self.base.potential_function(values))
        if base_value.shape != () or jnp.iscomplexobj(base_value):
            raise ValueError("Base Kähler potential must return one real scalar.")
        complex_coordinates = self.base.convention.to_complex(values)
        gauge = self.provider(complex_coordinates)[self.branch]
        return base_value + jnp.real(gauge)

    def geometry(self) -> KahlerPotentialGeometry:
        return KahlerPotentialGeometry(
            self.base.reference_metric,
            self.base.convention,
            self,
        )

    def invariance_report(
        self,
        coordinates: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> KahlerGaugeInvarianceReport:
        tolerance_ = float(tolerance)
        if tolerance_ < 0.0 or not jnp.isfinite(tolerance_):
            raise ValueError("Kähler gauge tolerance must be finite and nonnegative.")
        base_hessian = self.base.complex_hessian(coordinates)
        gauged_hessian = self.geometry().complex_hessian(coordinates)
        change = jnp.max(jnp.abs(gauged_hessian - base_hessian), initial=0.0)
        return KahlerGaugeInvarianceReport(
            change,
            jnp.asarray(tolerance_),
            gauge_id=self.gauge_id,
        )


__all__ = [
    "KahlerGaugeInvarianceReport",
    "KahlerHolomorphicGauge",
]
