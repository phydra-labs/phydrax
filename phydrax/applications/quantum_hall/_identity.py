#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical scientific identities for quantum Hall orbital manifolds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class HallComponentKey(StrictModule, NonTrainableState):
    """One explicit spin/valley/layer/subband component identity."""

    component_id: str = eqx.field(static=True)
    species_id: str = eqx.field(static=True)
    twice_spin_projection: int | None = eqx.field(static=True)
    valley_id: str | None = eqx.field(static=True)
    layer_id: str | None = eqx.field(static=True)
    subband_id: str | None = eqx.field(static=True)
    key_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_id: str,
        species_id: str,
        /,
        *,
        twice_spin_projection: int | None = None,
        valley_id: str | None = None,
        layer_id: str | None = None,
        subband_id: str | None = None,
    ):
        component = str(component_id).strip()
        species = str(species_id).strip()
        spin = None if twice_spin_projection is None else int(twice_spin_projection)
        valley = None if valley_id is None else str(valley_id).strip()
        layer = None if layer_id is None else str(layer_id).strip()
        subband = None if subband_id is None else str(subband_id).strip()
        if (
            not component
            or not species
            or any(value == "" for value in (valley, layer, subband) if value is not None)
        ):
            raise ValueError("Hall component identities must be explicit and nonempty.")
        self.component_id = component
        self.species_id = species
        self.twice_spin_projection = spin
        self.valley_id = valley
        self.layer_id = layer
        self.subband_id = subband
        self.key_id = canonical_fingerprint(
            {
                "kind": "hall-component-key",
                "component_id": component,
                "species_id": species,
                "twice_spin_projection": spin,
                "valley_id": valley,
                "layer_id": layer,
                "subband_id": subband,
            }
        )


SPIN_POLARIZED_ELECTRON = HallComponentKey(
    "electron-polarized",
    "electron",
    twice_spin_projection=1,
)


class MonopoleOrbitalKey(StrictModule, NonTrainableState):
    component: HallComponentKey
    landau_level: int = eqx.field(static=True)
    twice_orbital_projection: int = eqx.field(static=True)
    orbital_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: HallComponentKey,
        landau_level: int,
        twice_orbital_projection: int,
        /,
    ):
        if not isinstance(component, HallComponentKey):
            raise TypeError("component must be HallComponentKey.")
        level = int(landau_level)
        projection = int(twice_orbital_projection)
        if level < 0:
            raise ValueError("landau_level must be non-negative.")
        self.component = component
        self.landau_level = level
        self.twice_orbital_projection = projection
        self.orbital_id = canonical_fingerprint(
            {
                "kind": "monopole-orbital-key",
                "component": component.key_id,
                "landau_level": level,
                "twice_orbital_projection": projection,
            }
        )


class MonopoleLandauLevel(StrictModule, NonTrainableState):
    """One component-resolved Landau-level manifold at physical monopole Q."""

    twice_monopole_strength: int = eqx.field(static=True)
    landau_level: int = eqx.field(static=True)
    twice_orbital_spin: int = eqx.field(static=True)
    component: HallComponentKey
    one_body_energy: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)

    def __init__(
        self,
        twice_monopole_strength: int,
        landau_level: int,
        component: HallComponentKey,
        /,
        *,
        one_body_energy: float = 0.0,
    ):
        flux = int(twice_monopole_strength)
        level = int(landau_level)
        energy = float(one_body_energy)
        if not isinstance(component, HallComponentKey):
            raise TypeError("component must be HallComponentKey.")
        if flux < 1 or level < 0 or not isfinite(energy):
            raise ValueError("Monopole strength, Landau level, or energy is invalid.")
        self.twice_monopole_strength = flux
        self.landau_level = level
        self.twice_orbital_spin = flux + 2 * level
        self.component = component
        self.one_body_energy = energy
        self.manifold_id = canonical_fingerprint(
            {
                "kind": "monopole-landau-level",
                "twice_monopole_strength": flux,
                "landau_level": level,
                "component": component.key_id,
                "one_body_energy": energy,
            }
        )

    @property
    def orbital_count(self) -> int:
        return self.twice_orbital_spin + 1

    @property
    def orbital_keys(self) -> tuple[MonopoleOrbitalKey, ...]:
        return tuple(
            MonopoleOrbitalKey(self.component, self.landau_level, projection)
            for projection in range(
                -self.twice_orbital_spin,
                self.twice_orbital_spin + 1,
                2,
            )
        )


class HallChargeSector(StrictModule, NonTrainableState):
    """One deterministic roster of conserved integral or modular charges."""

    targets: tuple[tuple[str, int], ...] = eqx.field(static=True)
    moduli: tuple[tuple[str, int | None], ...] = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)

    def __init__(
        self,
        targets: Mapping[str, int],
        /,
        *,
        moduli: Mapping[str, int | None] | None = None,
    ):
        target_values = tuple(
            sorted((str(key), int(value)) for key, value in targets.items())
        )
        modulus_source = {} if moduli is None else dict(moduli)
        modulus_values = tuple(
            (key, None if modulus_source.get(key) is None else int(modulus_source[key]))
            for key, _ in target_values
        )
        if not target_values or any(not key for key, _ in target_values):
            raise ValueError("Hall charge sectors require nonempty unique labels.")
        if set(modulus_source) - {key for key, _ in target_values}:
            raise ValueError("Charge moduli reference undeclared targets.")
        if any(modulus is not None and modulus < 2 for _, modulus in modulus_values):
            raise ValueError("Modular charges require modulus at least two.")
        normalized_targets = tuple(
            (key, value if modulus is None else value % modulus)
            for (key, value), (_, modulus) in zip(
                target_values, modulus_values, strict=True
            )
        )
        self.targets = normalized_targets
        self.moduli = modulus_values
        self.sector_id = canonical_fingerprint(
            {
                "kind": "hall-charge-sector",
                "targets": normalized_targets,
                "moduli": modulus_values,
            }
        )


class HallComponentRoster(StrictModule, NonTrainableState):
    components: tuple[HallComponentKey, ...] = eqx.field(static=True)
    roster_id: str = eqx.field(static=True)

    def __init__(self, components: Sequence[HallComponentKey], /):
        values = tuple(components)
        if not values or any(not isinstance(value, HallComponentKey) for value in values):
            raise TypeError("components must contain HallComponentKey values.")
        identities = tuple(value.key_id for value in values)
        if len(set(identities)) != len(identities):
            raise ValueError("Hall component identities must be unique.")
        values = tuple(sorted(values, key=lambda value: value.component_id))
        self.components = values
        self.roster_id = canonical_fingerprint(
            {
                "kind": "hall-component-roster",
                "components": tuple(value.key_id for value in values),
            }
        )


__all__ = [
    "HallChargeSector",
    "HallComponentKey",
    "HallComponentRoster",
    "MonopoleLandauLevel",
    "MonopoleOrbitalKey",
    "SPIN_POLARIZED_ELECTRON",
]
