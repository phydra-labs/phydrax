#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Governed multigroup particle sources and scalar fluxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..measurement import (
    IndexSampleSupport,
    PreparedQuantityField,
    QuantityField,
    SpatialSamplingKind,
)
from ..units import derived_unit, METER, SECOND
from ._energy import EnergyGroupStructure, PreparedEnergyGroupStructure
from ._identity import NuclearSpeciesKey


_SOURCE_DENSITY = derived_unit("1/m3/s-source", ((METER, -3), (SECOND, -1)))
_SCALAR_FLUX = derived_unit("1/m2/s-flux", ((METER, -2), (SECOND, -1)))


def _validate_field(
    field: QuantityField,
    groups: EnergyGroupStructure,
    /,
    *,
    quantity_kind: str,
    unit,
) -> None:
    if not isinstance(field, QuantityField):
        raise TypeError("field must be QuantityField.")
    if not isinstance(groups, EnergyGroupStructure):
        raise TypeError("groups must be EnergyGroupStructure.")
    if field.quantity.quantity_kind != quantity_kind:
        raise ValueError(f"Field must represent {quantity_kind}.")
    if not isinstance(field.support, IndexSampleSupport):
        raise TypeError("Multigroup fields require IndexSampleSupport.")
    if field.support.axis_labels[-1] != "energy_group":
        raise ValueError("The trailing multigroup support axis must be energy_group.")
    if field.support.sample_shape[-1] != groups.group_count:
        raise ValueError("Measurement support and energy-group count disagree.")
    if field.sampling.spatial_kind is not SpatialSamplingKind.CELL_AVERAGE:
        raise ValueError("Initial multigroup fields require cell-average sampling.")
    if field.quantity.unit.dimension != unit.dimension:
        raise ValueError("Multigroup quantity uses an incompatible physical unit.")
    selected = field.values[field.valid_mask]
    if np.any(selected < 0.0):
        raise ValueError("Valid multigroup values must be nonnegative.")


class PreparedMultigroupParticleSource(StrictModule, NonTrainableState):
    field: PreparedQuantityField
    energy_groups: PreparedEnergyGroupStructure
    particle_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class MultigroupParticleSource:
    """Cell-average particle birth rate density resolved by energy group."""

    field: QuantityField
    energy_groups: EnergyGroupStructure
    particle: NuclearSpeciesKey
    source_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_field(
            self.field,
            self.energy_groups,
            quantity_kind="particle_source_density",
            unit=_SOURCE_DENSITY,
        )
        if not isinstance(self.particle, NuclearSpeciesKey):
            raise TypeError("particle must be NuclearSpeciesKey.")
        object.__setattr__(
            self,
            "source_id",
            canonical_fingerprint(
                {
                    "kind": "multigroup-particle-source",
                    "field": self.field.content_id,
                    "groups": self.energy_groups.group_id,
                    "particle": self.particle.species_id,
                }
            ),
        )

    def prepare(self) -> PreparedMultigroupParticleSource:
        return PreparedMultigroupParticleSource(
            self.field.prepare(target_unit=_SOURCE_DENSITY),
            self.energy_groups.prepare(),
            self.particle.species_id,
            self.source_id,
        )


class PreparedMultigroupScalarFlux(StrictModule, NonTrainableState):
    field: PreparedQuantityField
    energy_groups: PreparedEnergyGroupStructure
    particle_id: str = eqx.field(static=True)
    realization_count: int | None = eqx.field(static=True)
    flux_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class MultigroupScalarFlux:
    """Cell-average group-integrated scalar flux and statistical evidence."""

    field: QuantityField
    energy_groups: EnergyGroupStructure
    particle: NuclearSpeciesKey
    realization_count: int | None = None
    flux_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_field(
            self.field,
            self.energy_groups,
            quantity_kind="scalar_flux",
            unit=_SCALAR_FLUX,
        )
        if not isinstance(self.particle, NuclearSpeciesKey):
            raise TypeError("particle must be NuclearSpeciesKey.")
        count = self.realization_count
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise TypeError("realization_count must be an integer or None.")
            count = int(count)
            if count < 1:
                raise ValueError("realization_count must be positive.")
        if self.field.uncertainty is not None and count is None:
            raise ValueError("Statistical flux uncertainty requires realization_count.")
        object.__setattr__(self, "realization_count", count)
        object.__setattr__(
            self,
            "flux_id",
            canonical_fingerprint(
                {
                    "kind": "multigroup-scalar-flux",
                    "field": self.field.content_id,
                    "groups": self.energy_groups.group_id,
                    "particle": self.particle.species_id,
                    "realizations": count,
                }
            ),
        )

    def prepare(self) -> PreparedMultigroupScalarFlux:
        return PreparedMultigroupScalarFlux(
            self.field.prepare(target_unit=_SCALAR_FLUX),
            self.energy_groups.prepare(),
            self.particle.species_id,
            self.realization_count,
            self.flux_id,
        )


def group_reaction_rate(
    microscopic_cross_section_m2,
    scalar_flux_m2_s,
    /,
):
    """Return per-target reaction rate from matched group-integrated quantities."""

    cross_section = jnp.asarray(microscopic_cross_section_m2)
    flux = jnp.asarray(scalar_flux_m2_s)
    if cross_section.shape != flux.shape:
        raise ValueError("Cross section and scalar flux must have identical shapes.")
    if cross_section.ndim < 1:
        raise ValueError("Grouped reaction data require a trailing group axis.")
    return jnp.sum(cross_section * flux, axis=-1)


__all__ = [
    "MultigroupParticleSource",
    "MultigroupScalarFlux",
    "PreparedMultigroupParticleSource",
    "PreparedMultigroupScalarFlux",
    "group_reaction_rate",
]
