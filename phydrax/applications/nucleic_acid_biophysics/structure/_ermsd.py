# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent implementation of Bottaro et al., doi:10.1093/nar/gku972.

Published G uses ellipsoidal axes (5,5,3) angstrom and G=(sin(gamma*r)
*r_hat/gamma, (1+cos(gamma*r))/gamma) inside the cutoff, zero outside.
The published distance divides the directed-pair sum by nucleotide count,
NOT number of pairs. The cutoff is continuous but not globally differentiable.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....atomistic.sampling import (
    AbstractCollectiveVariableProgram,
    CollectiveVariableMetric,
)
from ....ein import contract
from ....series import SampledSeries
from ....units import ANGSTROM, conversion_factor, UnitDefinition
from .._binding import PreparedNucleotideBinding
from ._frames import base_frames


class GFeatureEvaluation(StrictModule):
    values: Array
    pair_valid: Array
    within_cutoff: Array
    cutoff_margin: Array
    frame_valid: Array
    descriptor_id: str = eqx.field(static=True)


class ERMSDEvaluation(StrictModule):
    value: Array
    squared_distance: Array
    successful: Array
    pair_valid: Array
    coverage_fraction: Array
    cutoff_margin: Array


class NucleotideGDescriptor(StrictModule, NonTrainableState):
    binding: PreparedNucleotideBinding
    pair_indices: Array
    length_to_angstrom: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    smooth_width: float = eqx.field(static=True)
    image_policy: str = eqx.field(static=True)
    descriptor_id: str = eqx.field(static=True)
    complete_pair_support: bool = eqx.field(static=True)

    def __init__(
        self,
        binding,
        *,
        length_unit: UnitDefinition,
        pairs=None,
        cutoff=2.4,
        image_policy: str,
        smooth_width=0.0,
    ):
        """Fixed directed pair support; omission changes the descriptor identity.

        pairs contains NucleotideKey pairs, never atom IDs. None admits all
        off-diagonal directed pairs. A sparse subset is exact for its selected
        support; it is full eRMSD only when omitted contributions vanish in BOTH
        conformations, a condition the caller must separately establish.
        smooth_width>0 selects a distinct C2-tapered G, not published eRMSD.
        """
        if not isinstance(binding, PreparedNucleotideBinding):
            raise TypeError("binding must be PreparedNucleotideBinding.")
        if not np.isfinite(cutoff) or cutoff <= 0 or not 0 <= smooth_width < cutoff:
            raise ValueError("Cutoff must be positive and taper width in [0, cutoff).")
        if image_policy not in ("nonperiodic", "unwrapped") or (
            binding.periodic and image_policy != "unwrapped"
        ):
            raise ValueError(
                "Periodic input must be explicitly unwrapped before preparation."
            )
        keys = binding.construct.nucleotide_keys
        order = {key: i for i, key in enumerate(keys)}
        selected = (
            tuple((a, b) for a in keys for b in keys if a != b)
            if pairs is None
            else tuple(pairs)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("At least one distinct directed pair is required.")
        if any(a not in order or b not in order or a == b for a, b in selected):
            raise ValueError("Pair support must use distinct construct nucleotide keys.")
        indices = [(order[a], order[b]) for a, b in selected]
        self.binding = binding
        self.pair_indices = jnp.asarray(indices, dtype=jnp.int64)
        self.length_to_angstrom = float(conversion_factor(length_unit, ANGSTROM))
        self.cutoff, self.smooth_width = float(cutoff), float(smooth_width)
        self.image_policy = image_policy
        self.complete_pair_support = len(selected) == len(keys) * (len(keys) - 1)
        self.descriptor_id = canonical_fingerprint(
            {
                "kind": "published-ermsd-G"
                if smooth_width == 0
                else "C2-tapered-ermsd-G",
                "binding": binding.binding_id,
                "pairs": indices,
                "length_factor": self.length_to_angstrom,
                "cutoff": self.cutoff,
                "width": self.smooth_width,
                "images": image_policy,
                "source": "10.1093/nar/gku972",
            }
        )

    def evaluate(self, positions) -> GFeatureEvaluation:
        frames = base_frames(positions, self.binding, image_policy=self.image_policy)
        left, right = self.pair_indices[:, 0], self.pair_indices[:, 1]
        relative = frames.centers[right] - frames.centers[left]
        local = contract("pi,pij->pj", relative, frames.axes[left])
        scaled = local * self.length_to_angstrom / jnp.asarray((5.0, 5.0, 3.0))
        squared = jnp.sum(scaled**2, axis=-1)
        radius = jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
        radius = jnp.where(squared > 0, radius, 0.0)
        phase = jnp.pi * radius / self.cutoff
        vector = scaled * jnp.sinc(radius / self.cutoff)[:, None]
        scalar = (1 + jnp.cos(phase)) * self.cutoff / jnp.pi
        values = jnp.concatenate((vector, scalar[:, None]), axis=-1)
        if self.smooth_width > 0:
            t = jnp.clip(
                (radius - self.cutoff + self.smooth_width) / self.smooth_width, 0.0, 1.0
            )
            values = values * (1 - 10 * t**3 + 15 * t**4 - 6 * t**5)[:, None]
        valid = frames.valid[left] & frames.valid[right]
        inside = radius < self.cutoff
        values = jnp.where((valid & inside)[:, None], values, 0.0)
        return GFeatureEvaluation(
            values,
            valid,
            inside & valid,
            jnp.abs(radius - self.cutoff),
            frames.valid,
            self.descriptor_id,
        )

    def compare(self, positions, reference) -> ERMSDEvaluation:
        left, right = self.evaluate(positions), self.evaluate(reference)
        valid = left.pair_valid & right.pair_valid
        squared = (
            jnp.sum(jnp.where(valid[:, None], (left.values - right.values) ** 2, 0.0))
            / self.binding.construct.nucleotide_count
        )
        value = jnp.where(
            squared > 0, jnp.sqrt(jnp.where(squared > 0, squared, 1.0)), 0.0
        )
        return ERMSDEvaluation(
            value,
            squared,
            jnp.all(valid),
            valid,
            jnp.mean(valid.astype(value.dtype)),
            jnp.min(jnp.minimum(left.cutoff_margin, right.cutoff_margin)),
        )

    def observe_series(self, coordinates: SampledSeries) -> SampledSeries:
        """Preserve sampled support/resets; coordinates are not assumed physical time."""
        values = jnp.asarray(coordinates.values)
        if values.shape[-2:] != (self.binding.support_size, 3):
            raise ValueError("Series coordinates must match the bound support.")
        flat = values.reshape((-1, self.binding.support_size, 3))
        evaluated = jax.vmap(self.evaluate)(flat)
        shape = values.shape[:-2] + evaluated.values.shape[1:]
        mask = evaluated.pair_valid.reshape(values.shape[:-2] + (-1, 1))
        if coordinates.value_valid is not None:
            # A masked atom may carry a finite placeholder; its affected rings
            # must remain unavailable rather than manufactured observations.
            source_mask = jnp.broadcast_to(coordinates.value_valid, values.shape)
            rings = source_mask[..., self.binding.ring_indices, :]
            frame_mask = jnp.all(rings, axis=(-2, -1))
            pair_mask = (
                frame_mask[..., self.pair_indices[:, 0]]
                & frame_mask[..., self.pair_indices[:, 1]]
            )
            mask = mask & pair_mask[..., None]
        return SampledSeries(
            coordinates.support,
            evaluated.values.reshape(shape),
            value_valid=jnp.broadcast_to(mask, shape),
            series_id=canonical_fingerprint(
                {"source": coordinates.series_id, "descriptor": self.descriptor_id}
            ),
        )


class ERMSDCollectiveVariableProgram(AbstractCollectiveVariableProgram):
    descriptor: NucleotideGDescriptor
    reference: Array
    output_size: int = eqx.field(static=True, default=1)
    names: tuple[str, ...] = eqx.field(static=True, default=("ermsd",))
    metrics: tuple[CollectiveVariableMetric, ...]
    program_id: str = eqx.field(static=True)

    def __init__(self, descriptor, reference):
        evaluated = descriptor.evaluate(reference)
        if not bool(jnp.all(evaluated.pair_valid)):
            raise ValueError(
                "A reference CV requires complete selected-pair coordinate coverage."
            )
        self.descriptor, self.reference = descriptor, jnp.asarray(reference)
        self.metrics = (CollectiveVariableMetric(),)
        self.program_id = canonical_fingerprint(
            {
                "descriptor": descriptor.descriptor_id,
                "reference": np.asarray(reference).tolist(),
            }
        )

    def evaluate(self, positions, /, *, cell=None, cell_vectors=None):
        """Native program ABI; geometry is already unwrapped before evaluation."""
        if cell is not None or cell_vectors is not None:
            raise ValueError(
                "eRMSD accepts already-unwrapped coordinates, not an implicit cell policy."
            )
        result = self.descriptor.compare(positions, self.reference)
        return result.value[None], result.successful


__all__ = [
    "GFeatureEvaluation",
    "ERMSDEvaluation",
    "NucleotideGDescriptor",
    "ERMSDCollectiveVariableProgram",
]
