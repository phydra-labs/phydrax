#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class EnergyChannelBalance(StrictModule):
    before: Array
    after: Array
    dissipation: Array
    external_work: Array
    channel: str = eqx.field(static=True)

    def __init__(
        self,
        channel: str,
        before: ArrayLike,
        after: ArrayLike,
        /,
        *,
        dissipation: ArrayLike = 0.0,
        external_work: ArrayLike = 0.0,
    ):
        name = str(channel)
        values = tuple(
            jnp.asarray(value) for value in (before, after, dissipation, external_work)
        )
        if not name or any(value.shape != () for value in values):
            raise ValueError("Energy-channel name must be nonempty and values scalar.")
        self.channel = name
        self.before, self.after, self.dissipation, self.external_work = values

    @property
    def residual_without_exchange(self) -> Array:
        return self.after - self.before + self.dissipation - self.external_work


class InternalEnergyExchange(StrictModule):
    source_amount: Array
    target_amount: Array
    exchange_id: str = eqx.field(static=True)
    source_channel: str = eqx.field(static=True)
    target_channel: str = eqx.field(static=True)

    def __init__(
        self,
        exchange_id: str,
        source_channel: str,
        target_channel: str,
        source_amount: ArrayLike,
        target_amount: ArrayLike,
        /,
    ):
        identifier = str(exchange_id)
        source = str(source_channel)
        target = str(target_channel)
        source_value = jnp.asarray(source_amount)
        target_value = jnp.asarray(target_amount, dtype=source_value.dtype)
        if (
            not identifier
            or not source
            or not target
            or source == target
            or source_value.shape != ()
            or target_value.shape != ()
        ):
            raise ValueError("Internal energy exchange is invalid.")
        self.exchange_id = identifier
        self.source_channel = source
        self.target_channel = target
        self.source_amount = source_value
        self.target_amount = target_value

    @property
    def cancellation_defect(self) -> Array:
        return self.source_amount + self.target_amount


class ConservationChannelBalance(StrictModule):
    before: Array
    after: Array
    volume_source: Array
    boundary_source: Array
    event_source: Array
    channel: str = eqx.field(static=True)

    def __init__(
        self,
        channel: str,
        before: ArrayLike,
        after: ArrayLike,
        /,
        *,
        volume_source: ArrayLike = 0.0,
        boundary_source: ArrayLike = 0.0,
        event_source: ArrayLike = 0.0,
    ):
        name = str(channel)
        values = tuple(
            jnp.asarray(value)
            for value in (
                before,
                after,
                volume_source,
                boundary_source,
                event_source,
            )
        )
        if not name or any(value.shape != () for value in values):
            raise ValueError(
                "Conservation-channel name must be nonempty and values scalar."
            )
        self.channel = name
        (
            self.before,
            self.after,
            self.volume_source,
            self.boundary_source,
            self.event_source,
        ) = values

    @property
    def residual(self) -> Array:
        return (
            self.after
            - self.before
            - self.volume_source
            - self.boundary_source
            - self.event_source
        )


class EntropyProductionBalance(StrictModule):
    before: Array
    after: Array
    entropy_flux: Array
    external_entropy: Array
    production: Array
    residual: Array
    nonnegative: Array

    def __init__(
        self,
        before: ArrayLike,
        after: ArrayLike,
        /,
        *,
        entropy_flux: ArrayLike,
        external_entropy: ArrayLike = 0.0,
        production: ArrayLike,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                before,
                after,
                entropy_flux,
                external_entropy,
                production,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Entropy-ledger values must be scalar.")
        (
            self.before,
            self.after,
            self.entropy_flux,
            self.external_entropy,
            self.production,
        ) = values
        self.residual = (
            self.after
            - self.before
            - self.entropy_flux
            - self.external_entropy
            - self.production
        )
        self.nonnegative = self.production >= 0.0


class CoupledPhaseFieldLedger(StrictModule):
    energy_channels: tuple[EnergyChannelBalance, ...]
    exchanges: tuple[InternalEnergyExchange, ...]
    conservation_channels: tuple[ConservationChannelBalance, ...]
    entropy: EntropyProductionBalance
    energy_residual: Array
    exchange_defect: Array
    maximum_conservation_defect: Array
    finite: Array
    successful: Array
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_channels: tuple[EnergyChannelBalance, ...],
        exchanges: tuple[InternalEnergyExchange, ...],
        conservation_channels: tuple[ConservationChannelBalance, ...],
        entropy: EntropyProductionBalance,
        /,
        *,
        energy_tolerance: ArrayLike,
        exchange_tolerance: ArrayLike,
        conservation_tolerance: ArrayLike,
        entropy_tolerance: ArrayLike,
        ledger_id: str,
    ):
        if not energy_channels:
            raise ValueError("Coupled ledger requires energy channels.")
        energy_names = tuple(channel.channel for channel in energy_channels)
        exchange_names = tuple(exchange.exchange_id for exchange in exchanges)
        conservation_names = tuple(channel.channel for channel in conservation_channels)
        if (
            len(set(energy_names)) != len(energy_names)
            or len(set(exchange_names)) != len(exchange_names)
            or len(set(conservation_names)) != len(conservation_names)
        ):
            raise ValueError("Coupled ledger channel identities must be unique.")
        if any(
            exchange.source_channel not in energy_names
            or exchange.target_channel not in energy_names
            for exchange in exchanges
        ):
            raise ValueError("Internal exchange references an unknown energy channel.")
        tolerances = tuple(
            jnp.asarray(value)
            for value in (
                energy_tolerance,
                exchange_tolerance,
                conservation_tolerance,
                entropy_tolerance,
            )
        )
        if any(value.shape != () for value in tolerances):
            raise ValueError("Coupled-ledger tolerances must be scalar.")
        self.energy_channels = energy_channels
        self.exchanges = exchanges
        self.conservation_channels = conservation_channels
        self.entropy = entropy
        self.energy_residual = sum(
            (channel.residual_without_exchange for channel in energy_channels[1:]),
            start=energy_channels[0].residual_without_exchange,
        )
        self.exchange_defect = (
            jnp.asarray(0.0, dtype=self.energy_residual.dtype)
            if not exchanges
            else jnp.max(
                jnp.abs(
                    jnp.stack(
                        tuple(exchange.cancellation_defect for exchange in exchanges)
                    )
                )
            )
        )
        self.maximum_conservation_defect = (
            jnp.asarray(0.0, dtype=self.energy_residual.dtype)
            if not conservation_channels
            else jnp.max(
                jnp.abs(
                    jnp.stack(
                        tuple(channel.residual for channel in conservation_channels)
                    )
                )
            )
        )
        values = (
            self.energy_residual,
            self.exchange_defect,
            self.maximum_conservation_defect,
            entropy.residual,
            entropy.production,
        )
        self.finite = jnp.all(jnp.isfinite(jnp.stack(values)))
        self.successful = (
            self.finite
            & (jnp.abs(self.energy_residual) <= tolerances[0])
            & (self.exchange_defect <= tolerances[1])
            & (self.maximum_conservation_defect <= tolerances[2])
            & entropy.nonnegative
            & (jnp.abs(entropy.residual) <= tolerances[3])
        )
        identifier = str(ledger_id)
        if not identifier:
            raise ValueError("Coupled ledger requires a stable ledger_id.")
        self.ledger_id = canonical_fingerprint(
            {
                "kind": "coupled-phase-field-ledger",
                "declared_id": identifier,
                "energy_channels": energy_names,
                "exchanges": exchange_names,
                "conservation_channels": conservation_names,
            }
        )


__all__ = [
    "ConservationChannelBalance",
    "CoupledPhaseFieldLedger",
    "EnergyChannelBalance",
    "EntropyProductionBalance",
    "InternalEnergyExchange",
]
