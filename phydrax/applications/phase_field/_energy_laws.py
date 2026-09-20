#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._phase_field import AbstractBulkFreeEnergy, DoubleWellFreeEnergy
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class AbstractBulkEvolutionLaw(StrictModule, NonTrainableState):
    """Energy-compatible current/previous bulk discretization."""

    law_id: eqx.AbstractVar[str]
    exact_identity: eqx.AbstractVar[bool]

    @abc.abstractmethod
    def derivative(
        self,
        potential: AbstractBulkFreeEnergy,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def incremental_density(
        self,
        potential: AbstractBulkFreeEnergy,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        raise NotImplementedError


class DiscreteGradientBulkLaw(AbstractBulkEvolutionLaw):
    """Exact potential-provided discrete-gradient evolution."""

    law_id: str = eqx.field(static=True)
    exact_identity: bool = eqx.field(static=True)

    def __init__(self):
        self.law_id = "phase-field-bulk-law/discrete-gradient"
        self.exact_identity = True

    def derivative(
        self,
        potential: AbstractBulkFreeEnergy,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        return potential.discrete_derivative(current, previous)

    def incremental_density(
        self,
        potential: AbstractBulkFreeEnergy,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        return potential.discrete_incremental_density(current, previous)


class ConvexSplitDoubleWellLaw(AbstractBulkEvolutionLaw):
    """First-order quartic convex-current/concave-previous specialization."""

    law_id: str = eqx.field(static=True)
    exact_identity: bool = eqx.field(static=True)

    def __init__(self):
        self.law_id = "phase-field-bulk-law/double-well-convex-split"
        self.exact_identity = False

    @staticmethod
    def _potential(potential: AbstractBulkFreeEnergy) -> DoubleWellFreeEnergy:
        if not isinstance(potential, DoubleWellFreeEnergy):
            raise TypeError("ConvexSplitDoubleWellLaw requires DoubleWellFreeEnergy.")
        return potential

    def derivative(
        self,
        potential: AbstractBulkFreeEnergy,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        selected = self._potential(potential)
        value = jnp.asarray(current)
        old = jnp.asarray(previous, dtype=value.dtype)
        return selected.scale.astype(value.dtype) * (value**3 - old)

    def incremental_density(
        self,
        potential: AbstractBulkFreeEnergy,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        selected = self._potential(potential)
        value = jnp.asarray(current)
        old = jnp.asarray(previous, dtype=value.dtype)
        return selected.scale.astype(value.dtype) * (0.25 * value**4 - old * value)


class PhaseFieldEnergyLedger(StrictModule):
    """Complete accepted-step storage, dissipation, and external-work accounting."""

    bulk_before: Array
    bulk_after: Array
    gradient_before: Array
    gradient_after: Array
    surface_before: Array
    surface_after: Array
    kinetic_dissipation: Array
    diffusion_dissipation: Array
    boundary_dissipation: Array
    boundary_work: Array
    source_work: Array
    stochastic_work: Array
    transfer_defect: Array
    total_residual: Array
    tolerance: Array
    finite: Array
    closed: Array
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        bulk_before: ArrayLike,
        bulk_after: ArrayLike,
        gradient_before: ArrayLike,
        gradient_after: ArrayLike,
        surface_before: ArrayLike = 0.0,
        surface_after: ArrayLike = 0.0,
        kinetic_dissipation: ArrayLike = 0.0,
        diffusion_dissipation: ArrayLike = 0.0,
        boundary_dissipation: ArrayLike = 0.0,
        boundary_work: ArrayLike = 0.0,
        source_work: ArrayLike = 0.0,
        stochastic_work: ArrayLike = 0.0,
        transfer_defect: ArrayLike = 0.0,
        tolerance: ArrayLike,
        ledger_id: str,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                bulk_before,
                bulk_after,
                gradient_before,
                gradient_after,
                surface_before,
                surface_after,
                kinetic_dissipation,
                diffusion_dissipation,
                boundary_dissipation,
                boundary_work,
                source_work,
                stochastic_work,
                transfer_defect,
                tolerance,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Phase-field energy-ledger entries must be scalar.")
        (
            self.bulk_before,
            self.bulk_after,
            self.gradient_before,
            self.gradient_after,
            self.surface_before,
            self.surface_after,
            self.kinetic_dissipation,
            self.diffusion_dissipation,
            self.boundary_dissipation,
            self.boundary_work,
            self.source_work,
            self.stochastic_work,
            self.transfer_defect,
            self.tolerance,
        ) = values
        energy_change = (
            self.bulk_after
            - self.bulk_before
            + self.gradient_after
            - self.gradient_before
            + self.surface_after
            - self.surface_before
        )
        self.total_residual = (
            energy_change
            + self.kinetic_dissipation
            + self.diffusion_dissipation
            + self.boundary_dissipation
            - self.boundary_work
            - self.source_work
            - self.stochastic_work
            + self.transfer_defect
        )
        self.finite = jnp.all(jnp.isfinite(jnp.stack(values + (self.total_residual,))))
        self.closed = self.finite & (self.total_residual <= self.tolerance)
        declared = str(ledger_id)
        if not declared:
            raise ValueError("Phase-field ledger_id must be nonempty.")
        self.ledger_id = canonical_fingerprint(
            {"kind": "phase-field-energy-ledger", "declared_id": declared}
        )

    @property
    def energy_before(self) -> Array:
        return self.bulk_before + self.gradient_before + self.surface_before

    @property
    def energy_after(self) -> Array:
        return self.bulk_after + self.gradient_after + self.surface_after


__all__ = [
    "AbstractBulkEvolutionLaw",
    "ConvexSplitDoubleWellLaw",
    "DiscreteGradientBulkLaw",
    "PhaseFieldEnergyLedger",
]
