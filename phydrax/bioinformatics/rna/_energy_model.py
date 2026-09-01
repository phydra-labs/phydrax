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
from ..._trainable import NonTrainableState


class RNAEnergyModel(StrictModule, NonTrainableState):
    """Declared additive scoring grammar, not a bundled thermodynamic parameter set."""

    pair_energies: Array
    unpaired_energies: Array
    allowed_pairs: Array
    temperature: Array
    gas_constant: Array
    alphabet_size: int = eqx.field(static=True)
    minimum_hairpin_length: int = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    alphabet_fingerprint: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        pair_energies: ArrayLike,
        /,
        *,
        allowed_pairs: ArrayLike | None = None,
        unpaired_energies: ArrayLike | float = 0.0,
        temperature: float = 310.15,
        gas_constant: float = 0.00198720425864083,
        minimum_hairpin_length: int = 3,
        energy_unit: str = "declared-energy-unit",
        alphabet_fingerprint: str = "rna-acgu",
    ):
        pair = np.asarray(pair_energies)
        if pair.ndim != 2 or pair.shape[0] == 0 or pair.shape[0] != pair.shape[1]:
            raise ValueError("pair_energies must be a non-empty square matrix.")
        if not np.issubdtype(pair.dtype, np.inexact):
            pair = pair.astype(np.float64)
        if np.any(~np.isfinite(pair)):
            raise ValueError(
                "pair_energies must be finite; allowed_pairs encodes exclusions."
            )
        if not np.allclose(pair, pair.T, atol=0.0, rtol=0.0):
            raise ValueError(
                "pair_energies must be symmetric for an unoriented pairing grammar."
            )
        alphabet_size = int(pair.shape[0])
        allowed = (
            np.ones(pair.shape, dtype=bool)
            if allowed_pairs is None
            else np.asarray(allowed_pairs, dtype=bool)
        )
        if allowed.shape != pair.shape or not np.array_equal(allowed, allowed.T):
            raise ValueError(
                "allowed_pairs must be a symmetric matrix matching pair_energies."
            )
        if np.any(np.diag(allowed)):
            raise ValueError(
                "Self-symbol pairing must be explicitly represented by distinct "
                "codes; allowed_pairs diagonal must be false."
            )
        unpaired = np.asarray(unpaired_energies, dtype=pair.dtype)
        if unpaired.ndim == 0:
            unpaired = np.full((alphabet_size,), float(unpaired), dtype=pair.dtype)
        if unpaired.shape != (alphabet_size,) or np.any(~np.isfinite(unpaired)):
            raise ValueError(
                "unpaired_energies must be finite and scalar or alphabet-sized."
            )
        temperature_ = float(temperature)
        gas_constant_ = float(gas_constant)
        minimum = int(minimum_hairpin_length)
        unit = str(energy_unit).strip()
        alphabet_id = str(alphabet_fingerprint).strip()
        if not np.isfinite(temperature_) or temperature_ <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        if not np.isfinite(gas_constant_) or gas_constant_ <= 0.0:
            raise ValueError(
                "gas_constant must be finite and positive in energy_unit/(mol*K)."
            )
        if minimum < 0 or not unit or not alphabet_id:
            raise ValueError(
                "minimum_hairpin_length must be non-negative and identity strings non-empty."
            )
        self.pair_energies = jnp.asarray(pair)
        self.unpaired_energies = jnp.asarray(unpaired)
        self.allowed_pairs = jnp.asarray(allowed)
        self.temperature = jnp.asarray(temperature_, dtype=pair.dtype)
        self.gas_constant = jnp.asarray(gas_constant_, dtype=pair.dtype)
        self.alphabet_size = alphabet_size
        self.minimum_hairpin_length = minimum
        self.energy_unit = unit
        self.alphabet_fingerprint = alphabet_id
        self.model_id = canonical_fingerprint(
            {
                "kind": "rna-additive-energy-model",
                "arrays": array_tree_fingerprint(
                    {
                        "pair_energies": pair,
                        "unpaired_energies": unpaired,
                        "allowed_pairs": allowed,
                        "temperature": np.asarray(temperature_, dtype=pair.dtype),
                        "gas_constant": np.asarray(gas_constant_, dtype=pair.dtype),
                    }
                ),
                "minimum_hairpin_length": minimum,
                "energy_unit": unit,
                "alphabet_fingerprint": alphabet_id,
            }
        )

    @property
    def thermal_energy(self) -> Array:
        return self.gas_constant * self.temperature

    def with_temperature(self, temperature: float) -> "RNAEnergyModel":
        """Return an independently fingerprinted model at a new temperature."""

        return RNAEnergyModel(
            np.asarray(self.pair_energies),
            allowed_pairs=np.asarray(self.allowed_pairs),
            unpaired_energies=np.asarray(self.unpaired_energies),
            temperature=temperature,
            gas_constant=float(np.asarray(self.gas_constant)),
            minimum_hairpin_length=self.minimum_hairpin_length,
            energy_unit=self.energy_unit,
            alphabet_fingerprint=self.alphabet_fingerprint,
        )


def nussinov_energy_model(
    *,
    pair_energy: float = -1.0,
    wobble_energy: float | None = None,
    unpaired_energy: float = 0.0,
    temperature: float = 310.15,
    minimum_hairpin_length: int = 3,
) -> RNAEnergyModel:
    """Create a unit-declared A,C,G,U pairing score without empirical data."""

    pair_value = float(pair_energy)
    wobble_value = pair_value if wobble_energy is None else float(wobble_energy)
    if not np.isfinite(pair_value) or not np.isfinite(wobble_value):
        raise ValueError("Pair scores must be finite.")
    energies = np.zeros((4, 4), dtype=np.float64)
    allowed = np.zeros((4, 4), dtype=bool)
    # Alphabet order A, C, G, U.
    for first, second, value in (
        (0, 3, pair_value),
        (1, 2, pair_value),
        (2, 3, wobble_value),
    ):
        energies[first, second] = value
        energies[second, first] = value
        allowed[first, second] = True
        allowed[second, first] = True
    return RNAEnergyModel(
        energies,
        allowed_pairs=allowed,
        unpaired_energies=unpaired_energy,
        temperature=temperature,
        minimum_hairpin_length=minimum_hairpin_length,
        energy_unit="declared-score-unit",
        alphabet_fingerprint="RNA:A,C,G,U",
    )


__all__ = ["RNAEnergyModel", "nussinov_energy_model"]
