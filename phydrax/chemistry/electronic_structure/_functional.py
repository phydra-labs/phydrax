#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable spin-density LDA and PBE exchange-correlation energies."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from .._method import DensityFunctionalPlan


_PW92_UNPOLARIZED = (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
_PW92_POLARIZED = (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)


def _safe_density(value: Array, /) -> Array:
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny ** 0.25)


def _pw92_component(radius: Array, parameters, /) -> Array:
    a, alpha, beta1, beta2, beta3, beta4 = parameters
    square_root = jnp.sqrt(radius)
    denominator = (
        2.0
        * a
        * (
            beta1 * square_root
            + beta2 * radius
            + beta3 * radius * square_root
            + beta4 * radius**2
        )
    )
    return -2.0 * a * (1.0 + alpha * radius) * jnp.log1p(jnp.reciprocal(denominator))


def pw92_correlation_per_particle(alpha_density: Array, beta_density: Array, /) -> Array:
    total = _safe_density(alpha_density + beta_density)
    polarization = jnp.clip((alpha_density - beta_density) / total, -1.0, 1.0)
    radius = (3.0 / (4.0 * jnp.pi * total)) ** (1.0 / 3.0)
    unpolarized = _pw92_component(radius, _PW92_UNPOLARIZED)
    polarized = _pw92_component(radius, _PW92_POLARIZED)
    interpolation = (
        (1.0 + polarization) ** (4.0 / 3.0) + (1.0 - polarization) ** (4.0 / 3.0) - 2.0
    ) / (2.0 ** (4.0 / 3.0) - 2.0)
    return unpolarized + interpolation * (polarized - unpolarized)


def lda_exchange_energy_density(alpha_density: Array, beta_density: Array, /) -> Array:
    coefficient = -0.75 * (6.0 / jnp.pi) ** (1.0 / 3.0)
    return coefficient * (
        _safe_density(alpha_density) ** (4.0 / 3.0)
        + _safe_density(beta_density) ** (4.0 / 3.0)
    )


def _pbe_exchange_spin(density: Array, gradient: Array, /) -> Array:
    doubled_density = 2.0 * _safe_density(density)
    doubled_gradient = 2.0 * gradient
    fermi = (3.0 * jnp.pi**2 * doubled_density) ** (1.0 / 3.0)
    reduced = jnp.sqrt(jnp.sum(doubled_gradient**2, axis=-1)) / (
        2.0 * fermi * doubled_density
    )
    kappa = 0.804
    mu = 0.2195149727645171
    enhancement = 1.0 + kappa - kappa / (1.0 + mu * reduced**2 / kappa)
    unpolarized = -0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0) * doubled_density ** (4.0 / 3.0)
    return 0.5 * unpolarized * enhancement


def pbe_exchange_energy_density(
    alpha_density: Array,
    beta_density: Array,
    alpha_gradient: Array,
    beta_gradient: Array,
    /,
) -> Array:
    return _pbe_exchange_spin(alpha_density, alpha_gradient) + _pbe_exchange_spin(
        beta_density, beta_gradient
    )


def pbe_correlation_energy_density(
    alpha_density: Array,
    beta_density: Array,
    total_gradient: Array,
    /,
) -> Array:
    total = _safe_density(alpha_density + beta_density)
    polarization = jnp.clip((alpha_density - beta_density) / total, -1.0, 1.0)
    phi = 0.5 * (
        (1.0 + polarization) ** (2.0 / 3.0) + (1.0 - polarization) ** (2.0 / 3.0)
    )
    fermi = (3.0 * jnp.pi**2 * total) ** (1.0 / 3.0)
    screening = jnp.sqrt(4.0 * fermi / jnp.pi)
    reduced = jnp.sqrt(jnp.sum(total_gradient**2, axis=-1)) / (
        2.0 * phi * screening * total
    )
    lda = pw92_correlation_per_particle(alpha_density, beta_density)
    beta = 0.06672455060314922
    gamma = (1.0 - jnp.log(2.0)) / jnp.pi**2
    exponential = jnp.expm1(-lda / (gamma * phi**3))
    a = beta / gamma / jnp.maximum(exponential, jnp.finfo(total.dtype).tiny)
    reduced_squared = reduced**2
    numerator = (beta / gamma) * reduced_squared * (1.0 + a * reduced_squared)
    denominator = 1.0 + a * reduced_squared + a**2 * reduced_squared**2
    correction = gamma * phi**3 * jnp.log1p(numerator / denominator)
    return total * (lda + correction)


class NativeXCFunctional(StrictModule, NonTrainableState):
    plan: DensityFunctionalPlan
    functional_id: str = eqx.field(static=True)

    def __init__(self, plan: DensityFunctionalPlan, /):
        if not isinstance(plan, DensityFunctionalPlan):
            raise TypeError("plan must be DensityFunctionalPlan.")
        supported = {
            "slater-x",
            "pw92-c",
            "pbe-x",
            "pbe-c",
            "regularized-meta",
        }
        unknown = {name for name, _ in plan.components} - supported
        if unknown:
            raise ValueError(
                "Native functional contains unsupported components: "
                + ", ".join(sorted(unknown))
            )
        self.plan = plan
        self.functional_id = canonical_fingerprint(
            {"kind": "native-xc-functional", "plan": plan.functional_id}
        )

    def energy_density(
        self,
        alpha_density: Array,
        beta_density: Array,
        alpha_gradient: Array,
        beta_gradient: Array,
        /,
        *,
        alpha_kinetic: Array | None = None,
        beta_kinetic: Array | None = None,
    ) -> Array:
        result = jnp.zeros_like(alpha_density)
        components = dict(self.plan.components)
        if "slater-x" in components:
            result = result + components["slater-x"] * lda_exchange_energy_density(
                alpha_density, beta_density
            )
        if "pw92-c" in components:
            total = alpha_density + beta_density
            result = result + components[
                "pw92-c"
            ] * total * pw92_correlation_per_particle(alpha_density, beta_density)
        if "pbe-x" in components:
            result = result + components["pbe-x"] * pbe_exchange_energy_density(
                alpha_density,
                beta_density,
                alpha_gradient,
                beta_gradient,
            )
        if "pbe-c" in components:
            result = result + components["pbe-c"] * pbe_correlation_energy_density(
                alpha_density,
                beta_density,
                alpha_gradient + beta_gradient,
            )
        if "regularized-meta" in components:
            if alpha_kinetic is None or beta_kinetic is None:
                raise ValueError(
                    "Meta-GGA functional evaluation requires spin kinetic densities."
                )
            total = _safe_density(alpha_density + beta_density)
            total_gradient = alpha_gradient + beta_gradient
            von_weizsaecker = jnp.sum(total_gradient**2, axis=-1) / (8.0 * total)
            kinetic = _safe_density(alpha_kinetic + beta_kinetic)
            indicator = jnp.clip(von_weizsaecker / kinetic, 0.0, 1.0)
            correction = (
                total ** (4.0 / 3.0) * (1.0 - indicator) ** 2 / (1.0 + indicator**2)
            )
            result = result + components["regularized-meta"] * correction
        return result

    def energy(
        self,
        alpha_density_matrix: ArrayLike,
        beta_density_matrix: ArrayLike,
        ao_values: ArrayLike,
        ao_gradients: ArrayLike,
        weights: ArrayLike,
        /,
    ) -> Array:
        alpha = jnp.asarray(alpha_density_matrix)
        beta = jnp.asarray(beta_density_matrix, dtype=alpha.dtype)
        ao = jnp.asarray(ao_values, dtype=alpha.dtype)
        gradient = jnp.asarray(ao_gradients, dtype=alpha.dtype)
        weights_ = jnp.asarray(weights, dtype=alpha.real.dtype)
        alpha_density = jnp.real(contract("ab,pa,pb->p", alpha, ao, ao))
        beta_density = jnp.real(contract("ab,pa,pb->p", beta, ao, ao))
        alpha_gradient = 2.0 * jnp.real(contract("ab,pax,pb->px", alpha, gradient, ao))
        beta_gradient = 2.0 * jnp.real(contract("ab,pax,pb->px", beta, gradient, ao))
        alpha_kinetic = 0.5 * jnp.real(
            contract("ab,pax,pbx->p", alpha, gradient, gradient)
        )
        beta_kinetic = 0.5 * jnp.real(contract("ab,pax,pbx->p", beta, gradient, gradient))
        density = self.energy_density(
            alpha_density,
            beta_density,
            alpha_gradient,
            beta_gradient,
            alpha_kinetic=alpha_kinetic,
            beta_kinetic=beta_kinetic,
        )
        return jnp.sum(weights_ * density)

    def potential_matrices(
        self,
        alpha_density_matrix: ArrayLike,
        beta_density_matrix: ArrayLike,
        ao_values: ArrayLike,
        ao_gradients: ArrayLike,
        weights: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        alpha = jnp.asarray(alpha_density_matrix)
        beta = jnp.asarray(beta_density_matrix, dtype=alpha.dtype)

        def energy(alpha_, beta_):
            return self.energy(alpha_, beta_, ao_values, ao_gradients, weights)

        value, gradients = jax.value_and_grad(energy, argnums=(0, 1))(alpha, beta)
        return value, gradients[0], gradients[1]


__all__ = [
    "NativeXCFunctional",
    "lda_exchange_energy_density",
    "pbe_correlation_energy_density",
    "pbe_exchange_energy_density",
    "pw92_correlation_per_particle",
]
