#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Two-step split-field PML for PSATD (Shapoval, Vay, Vincenti 2019).

Absorbing layers occupy the outer ``thickness[a]`` cells at both ends of axis
``a`` inside the periodic spectral box, so the two layers meet across the
periodic seam. Every field component ``F_c`` is split by the derivative that
drives it, ``F_c = Σ_a F_{c,a}``: ``∂ₜE_{c,a} = c² ε_{cab} D_a⁻ B_b`` and
``∂ₜB_{c,a} = −ε_{cab} D_a⁺ E_b``; the diagonal part ``E_{c,c}`` carries the
direct sources. Step one advances the splits by the exact integrals of the
analytic PSATD solution (their sum is the exact PSATD update of the totals);
step two multiplies ``F_{c,a}`` by ``exp(−σ_a h)`` in real space with the
graded conductivity ``σ_a = σ_max (d/L)^m`` at the component's own location.
The normal-incidence continuum reflection is ``exp(−2σ_max L/(c(m + 1)))``;
``σ_max`` is chosen from the declared ``reflection``.

The split-field damping is not divergence-preserving. The solver books the
divergence each damping step creates as absorber charge supported in the
layers (dilated by the stencil half-width at finite order); an infinite-order
divergence is global, so it is confined to the layers by a static curl-free
correction, the content the derivative cannot see being placed on the two
outermost planes of the thickest layer (which must therefore span at least
two cells). The Gauss law then holds to roundoff outside the layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log

import equinox as eqx
import numpy as np

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class SpectralPMLPlan(StrictModule, NonTrainableState):
    """Declared PSATD PML layers, target reflection, and grading power."""

    thickness: tuple[int, int, int] = eqx.field(static=True)
    reflection: float = eqx.field(static=True)
    profile_power: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thickness: Sequence[int],
        /,
        *,
        reflection: float = 1.0e-6,
        profile_power: float = 3.0,
    ) -> None:
        cells = tuple(int(value) for value in thickness)
        target = float(reflection)
        power = float(profile_power)
        if len(cells) != 3 or any(value < 0 for value in cells) or not any(cells):
            raise ValueError(
                "PML thickness must give three nonnegative cell counts, one nonzero."
            )
        if not 0.0 < target < 1.0:
            raise ValueError("PML reflection must lie in (0, 1).")
        if not np.isfinite(power) or power < 1.0:
            raise ValueError("PML profile power must be at least one.")
        self.thickness = (cells[0], cells[1], cells[2])
        self.reflection = target
        self.profile_power = power
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-psatd-pml",
                "thickness": list(cells),
                "reflection": target,
                "profile_power": power,
            }
        )

    def conductivity(
        self,
        counts: tuple[int, int, int],
        spacing: tuple[float, float, float],
        offsets: np.ndarray,
        speed: float,
        /,
    ) -> np.ndarray:
        """Host ``σ[N₀, N₁, N₂, 3 components, 3 axes]`` at component locations.

        ``offsets[c, a]`` is the half-cell offset of component ``c`` along axis
        ``a``; the depth is measured from the interior boundary.
        """
        result = np.zeros((*counts, 3, 3), dtype=np.float64)
        for axis in range(3):
            layer = self.thickness[axis]
            if layer == 0:
                continue
            count = counts[axis]
            width = layer * spacing[axis]
            maximum = (
                -(self.profile_power + 1.0) * speed * log(self.reflection) / (2.0 * width)
            )
            for component in range(3):
                position = np.arange(count, dtype=np.float64) + offsets[component, axis]
                depth = np.maximum(
                    np.maximum(layer - position, position - (count - layer)), 0.0
                )
                profile = maximum * np.minimum(depth / layer, 1.0) ** self.profile_power
                shape = [1, 1, 1]
                shape[axis] = count
                result[..., component, axis] = np.broadcast_to(
                    profile.reshape(shape), counts
                )
        return result

    def interior(self, counts: tuple[int, int, int], /) -> np.ndarray:
        """Host mask of nodes outside every absorbing layer."""
        mask = np.ones(counts, dtype=np.bool_)
        for axis in range(3):
            layer = self.thickness[axis]
            if layer == 0:
                continue
            index = np.arange(counts[axis])
            inside = (index >= layer) & (index <= counts[axis] - layer)
            shape = [1, 1, 1]
            shape[axis] = counts[axis]
            mask &= inside.reshape(shape)
        return mask


__all__ = ["SpectralPMLPlan"]
