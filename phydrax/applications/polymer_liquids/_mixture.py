#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


SequenceFormFactorKind: TypeAlias = Literal[
    "gaussian-chain", "freely-jointed-chain", "gaussian-ring"
]


class SiteMixturePlan(StrictModule, NonTrainableState):
    site_ids: tuple[str, ...] = eqx.field(static=True)
    number_densities: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, site_ids: tuple[str, ...], number_densities: ArrayLike, /):
        identifiers = tuple(str(value).strip() for value in site_ids)
        densities = np.asarray(number_densities, dtype=float)
        if (
            not identifiers
            or any(not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or densities.shape != (len(identifiers),)
            or np.any(~np.isfinite(densities))
            or np.any(densities <= 0.0)
        ):
            raise ValueError("PRISM site identities and positive densities are invalid.")
        self.site_ids = identifiers
        self.number_densities = jnp.asarray(densities)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prism-site-mixture-plan",
                "site_ids": list(identifiers),
                "number_densities": densities.tolist(),
            }
        )

    @property
    def site_count(self) -> int:
        return len(self.site_ids)


class SequenceFormFactorPlan(StrictModule, NonTrainableState):
    kind: SequenceFormFactorKind = eqx.field(static=True)
    sequence_site_types: Array
    statistical_segment_length: float = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SequenceFormFactorKind,
        sequence_site_types: ArrayLike,
        statistical_segment_length: float,
        /,
        *,
        site_count: int | None = None,
    ):
        sequence = np.asarray(sequence_site_types, dtype=np.int32)
        length = float(statistical_segment_length)
        count = (
            int(np.max(sequence)) + 1
            if site_count is None and sequence.size
            else int(site_count or 0)
        )
        if (
            kind not in ("gaussian-chain", "freely-jointed-chain", "gaussian-ring")
            or sequence.ndim != 1
            or sequence.size == 0
            or np.any(sequence < 0)
            or count <= 0
            or np.any(sequence >= count)
            or not math.isfinite(length)
            or length <= 0.0
        ):
            raise ValueError("Sequence form-factor configuration is invalid.")
        self.kind = kind
        self.sequence_site_types = jnp.asarray(sequence)
        self.statistical_segment_length = length
        self.site_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sequence-form-factor-plan",
                "model": kind,
                "sequence_site_types": sequence.tolist(),
                "statistical_segment_length": length,
                "site_count": count,
            }
        )

    def evaluate(self, wave_numbers: ArrayLike, /) -> Array:
        wave = jnp.asarray(wave_numbers, dtype=float)
        if wave.ndim != 1 or wave.shape[0] == 0:
            raise ValueError("wave_numbers must be a non-empty vector.")
        sequence = self.sequence_site_types
        contour = jnp.arange(sequence.size)
        separation = jnp.abs(contour[:, None] - contour[None, :])
        squared_wave = wave[:, None, None] ** 2
        length_squared = self.statistical_segment_length**2
        if self.kind == "gaussian-chain":
            pair = jnp.exp(-squared_wave * length_squared * separation / 6.0)
        elif self.kind == "freely-jointed-chain":
            characteristic = jnp.sinc(wave * self.statistical_segment_length / jnp.pi)
            pair = characteristic[:, None, None] ** separation[None, :, :]
        else:
            chain_length = sequence.size
            ring_distance = separation * (chain_length - separation) / chain_length
            pair = jnp.exp(-squared_wave * length_squared * ring_distance / 6.0)
        channels = jnp.arange(self.site_count)
        membership = sequence[:, None] == channels[None, :]
        counts = jnp.sum(membership, axis=0)
        aggregated = contract("ia,jb,kij->kab", membership, membership, pair)
        normalization = jnp.sqrt(counts[:, None] * counts[None, :])
        return aggregated / normalization[None, :, :]


class TabulatedFormFactorPlan(StrictModule, NonTrainableState):
    wave_numbers: Array
    values: Array
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wave_numbers: ArrayLike,
        values: ArrayLike,
        /,
        *,
        source_id: str,
    ):
        wave = np.asarray(wave_numbers, dtype=float)
        matrix = np.asarray(values, dtype=float)
        source = str(source_id).strip()
        if (
            wave.ndim != 1
            or wave.size == 0
            or np.any(~np.isfinite(wave))
            or np.any(wave <= 0.0)
            or np.any(np.diff(wave) <= 0.0)
            or matrix.ndim != 3
            or matrix.shape[0] != wave.size
            or matrix.shape[1] != matrix.shape[2]
            or np.any(~np.isfinite(matrix))
            or not np.allclose(matrix, np.swapaxes(matrix, -1, -2))
            or not source
        ):
            raise ValueError("Tabulated form-factor data are invalid.")
        eigenvalues = np.linalg.eigvalsh(matrix)
        tolerance = 100.0 * np.finfo(matrix.dtype).eps * max(1.0, np.max(np.abs(matrix)))
        if np.any(eigenvalues < -tolerance):
            raise ValueError("Form-factor matrices must be positive semidefinite.")
        self.wave_numbers = jnp.asarray(wave)
        self.values = jnp.asarray(matrix)
        self.source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "tabulated-form-factor-plan",
                "wave_numbers": array_tree_fingerprint(wave),
                "values": array_tree_fingerprint(matrix),
                "source_id": source,
            }
        )

    @property
    def site_count(self) -> int:
        return int(self.values.shape[-1])

    def evaluate(self, wave_numbers: ArrayLike, /) -> Array:
        wave = np.asarray(wave_numbers, dtype=float)
        reference = np.asarray(self.wave_numbers)
        if wave.shape != reference.shape or not np.array_equal(wave, reference):
            raise ValueError(
                "Tabulated form factors require the exact admitted wave-number grid."
            )
        return self.values


class SitePairPotentialPlan(StrictModule, NonTrainableState):
    radii: Array
    beta_potential: Array
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radii: ArrayLike,
        beta_potential: ArrayLike,
        /,
        *,
        source_id: str,
    ):
        radial = np.asarray(radii, dtype=float)
        potential = np.asarray(beta_potential, dtype=float)
        source = str(source_id).strip()
        if (
            radial.ndim != 1
            or radial.size == 0
            or np.any(~np.isfinite(radial))
            or np.any(radial <= 0.0)
            or np.any(np.diff(radial) <= 0.0)
            or potential.ndim != 3
            or potential.shape[-1] != radial.size
            or potential.shape[0] != potential.shape[1]
            or np.any(~np.isfinite(potential))
            or not np.allclose(potential, np.swapaxes(potential, 0, 1))
            or not source
        ):
            raise ValueError("Site-pair potential data are invalid.")
        self.radii = jnp.asarray(radial)
        self.beta_potential = jnp.asarray(potential)
        self.source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "site-pair-potential-plan",
                "radii": array_tree_fingerprint(radial),
                "beta_potential": array_tree_fingerprint(potential),
                "source_id": source,
            }
        )

    @property
    def site_count(self) -> int:
        return int(self.beta_potential.shape[0])


__all__ = [
    "SequenceFormFactorKind",
    "SequenceFormFactorPlan",
    "SiteMixturePlan",
    "SitePairPotentialPlan",
    "TabulatedFormFactorPlan",
]
