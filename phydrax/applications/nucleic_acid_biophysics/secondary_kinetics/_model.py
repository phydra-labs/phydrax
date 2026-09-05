#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from ...._fingerprint import canonical_fingerprint
from ....qualification import ReferenceArtifactManifest
from ....units import (
    conversion_factor,
    CUBIC_METER,
    KELVIN,
    MOLE_PER_CUBIC_METER,
    SECOND,
    UnitDefinition,
)
from ._state import SecondaryStructureState


_AVOGADRO = 6.02214076e23


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _table(value: Mapping[str, float], name: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a parameter mapping.")
    result = tuple(sorted((key, float(number)) for key, number in value.items()))
    if any(
        not isinstance(key, str) or not key or not math.isfinite(number)
        for key, number in result
    ):
        raise ValueError(f"{name} requires named finite parameters.")
    return result


@dataclass(frozen=True, slots=True, init=False)
class AssociationConvention:
    """Dimensionless standard-volume factor for labelled strand copies.

    Each independent association adds log(c_standard N_A V) to G/(RT).
    Standard-state mode sets that factor to one and makes no finite-volume or
    concentration-timescale claim. Number-to-molar conversion is explicit here,
    not an ordinary multiplicative energy conversion.
    """

    mode: str
    standard_concentration_mol_per_m3: float
    volume_m3: float | None
    log_standard_volume: float

    def __init__(
        self,
        *,
        mode: str,
        standard_concentration: float,
        concentration_unit: UnitDefinition = MOLE_PER_CUBIC_METER,
        volume: float | None = None,
        volume_unit: UnitDefinition = CUBIC_METER,
    ):
        concentration = _positive(
            standard_concentration
            * conversion_factor(concentration_unit, MOLE_PER_CUBIC_METER),
            "standard concentration",
        )
        if mode not in ("standard_state", "fixed_volume"):
            raise ValueError("Association mode must be standard_state or fixed_volume.")
        if mode == "fixed_volume":
            if volume is None:
                raise ValueError("Fixed-volume association requires an explicit volume.")
            volume_si = _positive(
                volume * conversion_factor(volume_unit, CUBIC_METER), "volume"
            )
            log_factor = (
                math.log(concentration) + math.log(_AVOGADRO) + math.log(volume_si)
            )
        else:
            if volume is not None:
                raise ValueError(
                    "Standard-state association does not accept a hidden finite volume."
                )
            volume_si, log_factor = None, 0.0
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "standard_concentration_mol_per_m3", concentration)
        object.__setattr__(self, "volume_m3", volume_si)
        object.__setattr__(self, "log_standard_volume", log_factor)

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            (self.mode, self.standard_concentration_mol_per_m3, self.volume_m3)
        )


@dataclass(frozen=True, slots=True, init=False)
class SecondaryEnergyModel:
    """Source-pinned additive G/(RT) at one explicitly declared temperature.

    Supported profiles are ``pair_loop`` and ``nearest_neighbor_loop``.
    Both include per-pair terms, hairpin/bulge/internal/multibranch loops and
    association initiation. Only the latter adds antiparallel stack terms.
    Nick-containing/exterior loops have zero loop penalty by profile definition;
    there are no hidden dangling-end, coaxial-stack or mismatch corrections.
    No published calibration is built in. Missing required table entries refuse
    compilation rather than extrapolating uncalibrated loop parameters.
    """

    profile: str
    chemistry: str
    pairing_rule: str
    temperature_kelvin: float
    minimum_hairpin_unpaired: int
    pair_energies: tuple[tuple[str, float], ...]
    stack_energies: tuple[tuple[str, float], ...]
    hairpin_energies: tuple[tuple[str, float], ...]
    bulge_energies: tuple[tuple[str, float], ...]
    internal_energies: tuple[tuple[str, float], ...]
    multibranch: tuple[float, float, float]
    association_initiation: float
    manifest: ReferenceArtifactManifest
    requested_use: tuple[tuple[str, bool], ...]
    model_id: str

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        manifest: ReferenceArtifactManifest,
        /,
        *,
        requested_use: Mapping[str, bool],
        temperature_unit: UnitDefinition = KELVIN,
    ) -> SecondaryEnergyModel:
        """Admit and parse exact caller-supplied UTF-8 JSON parameter bytes.

        Required JSON fields match the public model attributes except manifest,
        requested_use and model_id; ``temperature`` is converted to Kelvin.
        Energy values are explicitly dimensionless molar G/(RT), not kelvin or
        raw kJ/mol. The pinned manifest governs these exact bytes and all
        inherited lineage restrictions must already have been admitted.
        """
        allowed_uses = {"commercial_use", "redistribution", "training_use", "export"}
        if set(requested_use) != allowed_uses or any(
            type(value) is not bool for value in requested_use.values()
        ):
            raise ValueError(
                "requested_use must explicitly declare all four manifest use flags."
            )
        manifest.require_rights(**dict(requested_use))
        if (
            len(content) != manifest.size_bytes
            or hashlib.new(manifest.checksum_algorithm, content).hexdigest()
            != manifest.checksum
        ):
            raise ValueError(
                "Parameter bytes disagree with the source manifest checksum or size."
            )
        data = json.loads(content)
        required = {
            "profile",
            "chemistry",
            "pairing_rule",
            "temperature",
            "energy_convention",
            "minimum_hairpin_unpaired",
            "pair_energies",
            "stack_energies",
            "hairpin_energies",
            "bulge_energies",
            "internal_energies",
            "multibranch",
            "association_initiation",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError(
                "Parameter artifact must contain exactly the declared model fields."
            )
        if data["profile"] not in ("pair_loop", "nearest_neighbor_loop"):
            raise ValueError("Unsupported secondary free-energy model profile.")
        if data["chemistry"] not in ("DNA", "RNA", "DNA-RNA"):
            raise ValueError("Chemistry must explicitly select DNA, RNA or DNA-RNA.")
        if data["pairing_rule"] not in ("watson_crick", "watson_crick_wobble"):
            raise ValueError("Unsupported secondary pairing rule.")
        if data["energy_convention"] != "dimensionless_molar_G_over_RT":
            raise ValueError("Parameters must explicitly use dimensionless molar G/(RT).")
        minimum = data["minimum_hairpin_unpaired"]
        if type(minimum) is not int or minimum < 0:
            raise ValueError("Minimum hairpin size must be a nonnegative integer.")
        temperature = _positive(
            data["temperature"] * conversion_factor(temperature_unit, KELVIN),
            "temperature",
        )
        tables = {
            name: _table(data[name], name)
            for name in (
                "pair_energies",
                "stack_energies",
                "hairpin_energies",
                "bulge_energies",
                "internal_energies",
            )
        }
        if not tables["pair_energies"]:
            raise ValueError("An explicit pair-energy table is required.")
        if data["profile"] == "pair_loop" and tables["stack_energies"]:
            raise ValueError(
                "pair_loop does not consume a stack table; select nearest_neighbor_loop."
            )
        multibranch = tuple(float(value) for value in data["multibranch"])
        initiation = float(data["association_initiation"])
        if len(multibranch) != 3 or not all(
            math.isfinite(value) for value in (*multibranch, initiation)
        ):
            raise ValueError(
                "Multibranch initiation/branch/unpaired and association terms must be finite."
            )
        instance = object.__new__(cls)
        for name in ("profile", "chemistry", "pairing_rule", "minimum_hairpin_unpaired"):
            object.__setattr__(instance, name, data[name])
        for name, table in tables.items():
            object.__setattr__(instance, name, table)
        object.__setattr__(instance, "temperature_kelvin", temperature)
        object.__setattr__(instance, "multibranch", multibranch)
        object.__setattr__(instance, "association_initiation", initiation)
        object.__setattr__(instance, "manifest", manifest)
        object.__setattr__(
            instance, "requested_use", tuple(sorted(requested_use.items()))
        )
        object.__setattr__(
            instance,
            "model_id",
            canonical_fingerprint(
                (manifest.manifest_id, temperature, tuple(sorted(requested_use.items())))
            ),
        )
        return instance

    def require_calibrated_reference(self) -> tuple[tuple[str, float], ...]:
        """Require quantified source uncertainty before a calibration claim."""
        return self.manifest.require_uncertainty()

    def standard_free_energy(self, state: SecondaryStructureState) -> float:
        """Evaluate host G/(RT) for this named, restricted model profile."""
        keys, bases = state.construct.nucleotide_keys, state.construct.bases
        pairs = state.numeric_pairs
        expected = {"DNA", "RNA"} if self.chemistry == "DNA-RNA" else {self.chemistry}
        if set(state.construct.polymer_types) != expected or any(
            state.construct.circular
        ):
            raise ValueError("State chemistry/linearity is outside this model profile.")
        strand_types = dict(
            zip(state.construct.strand_ids, state.construct.polymer_types, strict=True)
        )
        for i, j in pairs:
            code = bases[i] + bases[j]
            canonical = code in ("AT", "TA", "CG", "GC", "AU", "UA")
            wobble = (
                self.pairing_rule == "watson_crick_wobble"
                and code in ("GU", "UG")
                and strand_types[keys[i].strand_id]
                == strand_types[keys[j].strand_id]
                == "RNA"
            )
            if not (canonical or wobble):
                raise ValueError(
                    "State contains a pairing partner forbidden by this model."
                )
            if (
                keys[i].strand_id == keys[j].strand_id
                and j - i - 1 < self.minimum_hairpin_unpaired
            ):
                raise ValueError("State violates the model minimum hairpin size.")
        pair_table, stack_table = dict(self.pair_energies), dict(self.stack_energies)
        hairpins, bulges, internals = (
            dict(self.hairpin_energies),
            dict(self.bulge_energies),
            dict(self.internal_energies),
        )

        def required(table, key, kind):
            if key not in table:
                raise ValueError(f"Missing source-pinned {kind} parameter for {key}.")
            return table[key]

        energy = sum(
            required(pair_table, bases[i] + bases[j], "base pair") for i, j in pairs
        )
        pair_set = set(pairs)
        for i, j in pairs:
            stacked = (
                (i + 1, j - 1) in pair_set
                and keys[i].strand_id == keys[i + 1].strand_id
                and keys[j].strand_id == keys[j - 1].strand_id
            )
            if stacked and self.profile == "nearest_neighbor_loop":
                stack_key = bases[i] + bases[j] + "/" + bases[i + 1] + bases[j - 1]
                energy += required(stack_table, stack_key, "nearest-neighbor stack")
            # Nicked loops are explicitly exterior in this profile.
            if keys[i].strand_id != keys[j].strand_id:
                continue
            children = [
                (k, l)
                for k, l in pairs
                if i < k < l < j and not any(i < a < k < l < b < j for a, b in pairs)
            ]
            if not children:
                energy += required(hairpins, str(j - i - 1), "hairpin")
            elif len(children) == 1:
                k, l = children[0]
                left, right = k - i - 1, j - l - 1
                if left == 0 and right == 0:
                    continue
                if left == 0 or right == 0:
                    energy += required(bulges, str(left + right), "bulge")
                else:
                    energy += required(internals, f"{left},{right}", "internal loop")
            else:
                unpaired = j - i - 1 - sum(l - k + 1 for k, l in children)
                initiation, branch, residue = self.multibranch
                energy += initiation + branch * (len(children) + 1) + residue * unpaired
        associations = len(state.construct.strand_ids) - state.partition.complex_count
        return energy + associations * self.association_initiation


@dataclass(frozen=True, slots=True)
class SecondaryRateLaw:
    """Named rates with independent equilibrium energies and kinetic prefactors.

    ``metropolis`` and ``symmetric_barrier`` use symmetric attempt frequencies.
    ``association_metropolis`` additionally gives joins their explicit 1/(c°N_AV)
    factor and keeps dissociation independent of volume. Only this last law has
    an elementary dilute bimolecular interpretation; multistep macrostates do not
    automatically inherit a second-order rate constant.
    """

    name: str
    unimolecular_prefactor: float
    association_prefactor: float
    time_unit: UnitDefinition = SECOND

    def __post_init__(self):
        if self.name not in ("metropolis", "symmetric_barrier", "association_metropolis"):
            raise ValueError("Unknown secondary kinetic rate law.")
        _positive(self.unimolecular_prefactor, "unimolecular prefactor")
        _positive(self.association_prefactor, "association prefactor")
        conversion_factor(self.time_unit, SECOND)

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            (
                self.name,
                self.unimolecular_prefactor,
                self.association_prefactor,
                self.time_unit.unit_id,
            )
        )


__all__ = ["AssociationConvention", "SecondaryEnergyModel", "SecondaryRateLaw"]
