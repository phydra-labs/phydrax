#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...ein import contract
from ...linalg import DenseLinearOperator, matrix_exponential_action, matrix_phi1_action


class EnergyBalanceResult(StrictModule):
    temperature: Array
    surface_temperature_integral: Array
    input_energy: Array
    outgoing_energy: Array
    energy_residual: Array
    successful: Array


class MultilayerEnergyBalance(StrictModule):
    """Serial, conservative layer exchange and linear surface feedback.

    Temperature is a K anomaly, capacity is W model-year m^-2 K^-1,
    feedback/exchange are W m^-2 K^-1; duration uses 365.25-day years.
    Multiply integrated energies by MODEL_YEAR_SECONDS to obtain J m^-2.
    A zero surface feedback is allowed: its equilibrium exists only at zero
    net forcing. Positive layer capacities are always required.
    """

    capacities: Array
    exchanges: Array
    feedback: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacities: ArrayLike = (8.0, 100.0),
        exchanges: ArrayLike = (0.7,),
        *,
        feedback: float = 1.2,
    ):
        capacities_ = np.asarray(capacities, dtype=float)
        exchanges_ = np.asarray(exchanges, dtype=float)
        if (
            capacities_.ndim != 1
            or len(capacities_) < 1
            or exchanges_.shape != (len(capacities_) - 1,)
        ):
            raise ValueError(
                "Capacities require one or more layers and one exchange per adjacent pair."
            )
        if (
            not np.all(np.isfinite(capacities_))
            or np.any(capacities_ <= 0.0)
            or not np.all(np.isfinite(exchanges_))
            or np.any(exchanges_ < 0.0)
            or not np.isfinite(feedback)
            or feedback < 0.0
        ):
            raise ValueError(
                "Capacities must be positive; exchanges and feedback finite and nonnegative."
            )
        self.capacities = jnp.asarray(capacities_)
        self.exchanges = jnp.asarray(exchanges_)
        self.feedback = jnp.asarray(feedback, dtype=self.capacities.dtype)
        self.model_id = canonical_fingerprint(
            {
                "kind": "serial-multilayer-energy-balance",
                "coefficients": array_tree_fingerprint(
                    (self.capacities, self.exchanges, self.feedback)
                ),
                "capacity_unit": "W model-year m^-2 K^-1",
                "forcing_unit": "W m^-2",
            }
        )

    def heat_content(self, temperature: Array, /) -> Array:
        return contract("i,i->", self.capacities, temperature, backend="jax")

    def advance(
        self, temperature: Array, duration: Array, forcing: Array, /
    ) -> EnergyBalanceResult:
        """Exact affine subproblem for a constant interval forcing.

        An integral coordinate tracks surface temperature independently, so
        energy evidence compares storage with supplied minus radiated energy.
        Native dense matrix functions handle singular and zero generators;
        no inversion or assumption of diagonalizability is made.
        """
        layers = self.capacities.shape[0]
        if temperature.shape != (layers,):
            raise ValueError("Temperature must match the layer capacities.")
        generator = jnp.zeros((layers + 1, layers + 1), dtype=temperature.dtype)
        generator = generator.at[0, 0].set(-self.feedback / self.capacities[0])
        for index in range(layers - 1):
            exchange = self.exchanges[index]
            generator = generator.at[index, index].add(-exchange / self.capacities[index])
            generator = generator.at[index, index + 1].add(
                exchange / self.capacities[index]
            )
            generator = generator.at[index + 1, index].add(
                exchange / self.capacities[index + 1]
            )
            generator = generator.at[index + 1, index + 1].add(
                -exchange / self.capacities[index + 1]
            )
        generator = generator.at[-1, 0].set(1.0)
        initial = jnp.concatenate((temperature, jnp.zeros((1,), dtype=temperature.dtype)))
        source = jnp.zeros_like(initial).at[0].set(forcing / self.capacities[0])
        # An explicit singleton operator batch selects the existing exact dense
        # native path rather than a Krylov approximation for this tiny system.
        operator = DenseLinearOperator(generator[None, ...])
        homogeneous = matrix_exponential_action(operator, initial[None, ...], duration)
        driven = matrix_phi1_action(operator, source[None, ...], duration)
        evolved = homogeneous.value[0] + duration * driven.value[0]
        updated, integral = evolved[:-1], evolved[-1]
        input_energy = duration * forcing
        outgoing = self.feedback * integral
        residual = self.heat_content(updated - temperature) - input_energy + outgoing
        successful = (
            jnp.all(homogeneous.converged)
            & jnp.all(driven.converged)
            & jnp.all(jnp.isfinite(evolved))
            & (duration >= 0.0)
            & jnp.isfinite(duration)
        )
        return EnergyBalanceResult(
            updated, integral, input_energy, outgoing, residual, successful
        )


__all__ = ["EnergyBalanceResult", "MultilayerEnergyBalance"]
