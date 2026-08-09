#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, ArrayLike

from ._algebra import AmplitudeKernel, SumKernel
from ._base import AbstractPositiveDefiniteKernel
from ._finite_feature import (
    AbstractFiniteFeatureKernel,
    kernel_feature_rank,
    kernel_features,
)
from ._spectral import AbstractSpectralMultiplier, SpectralFeatureKernel


class CochainHodgeSpectralKernel(AbstractFiniteFeatureKernel):
    """Finite covariance sum over selected harmonic, exact, and coexact sectors."""

    kernel: AbstractPositiveDefiniteKernel
    spectra: Any
    sector_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        spectra: Any,
        /,
        *,
        harmonic_multiplier: AbstractSpectralMultiplier | None = None,
        exact_multiplier: AbstractSpectralMultiplier | None = None,
        coexact_multiplier: AbstractSpectralMultiplier | None = None,
        harmonic_amplitude: ArrayLike = 1.0,
        exact_amplitude: ArrayLike = 1.0,
        coexact_amplitude: ArrayLike = 1.0,
        normalize_sectors: bool = True,
    ):
        from ..graph._cochain_spectrum import CochainHodgeSectorSpectra

        if not isinstance(spectra, CochainHodgeSectorSpectra):
            raise TypeError("spectra must be a CochainHodgeSectorSpectra.")
        declarations = (
            (
                "harmonic",
                spectra.harmonic,
                harmonic_multiplier,
                harmonic_amplitude,
            ),
            ("exact", spectra.exact, exact_multiplier, exact_amplitude),
            ("coexact", spectra.coexact, coexact_multiplier, coexact_amplitude),
        )
        children = []
        names = []
        for name, basis, multiplier, amplitude in declarations:
            if multiplier is None:
                continue
            if basis is None:
                raise ValueError(f"The {name} Hodge sector is empty.")
            if not isinstance(multiplier, AbstractSpectralMultiplier):
                raise TypeError(f"{name}_multiplier has an incompatible type.")
            children.append(
                AmplitudeKernel(
                    SpectralFeatureKernel(
                        basis,
                        multiplier,
                        normalize=normalize_sectors,
                    ),
                    amplitude,
                )
            )
            names.append(name)
        if not children:
            raise ValueError("At least one nonempty Hodge sector must be selected.")
        self.kernel = SumKernel(tuple(children))
        self.spectra = spectra
        self.sector_names = tuple(names)

    def features(self, points: ArrayLike, /) -> Array:
        return kernel_features(self.kernel, points)

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.kernel.pairwise(left, right)

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.kernel.matrix(left, right)

    def diagonal(self, points: ArrayLike, /) -> Array:
        return self.kernel.diagonal(points)

    @property
    def feature_rank(self) -> int:
        rank = kernel_feature_rank(self.kernel)
        if rank is None:
            raise RuntimeError("Hodge sector composition lost its feature capability.")
        return rank

    @property
    def max_derivative_order(self) -> int:
        return 0

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        sectors = "+".join(self.sector_names)
        return (
            f"CochainHodgeSpectralKernel[degree={self.spectra.degree};"
            f"sectors={sectors};boundary={self.spectra.boundary_policy}]"
        )


__all__ = ["CochainHodgeSpectralKernel"]
