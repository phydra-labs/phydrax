#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
from jax import Array
from jax.typing import ArrayLike

from ..exterior._spectra import HodgeSectorSpectra
from ._algebra import AmplitudeKernel, SumKernel
from ._base import AbstractPositiveDefiniteKernel
from ._finite_feature import (
    AbstractFiniteFeatureKernel,
    kernel_feature_rank,
    kernel_features,
)
from ._spectral import AbstractSpectralMultiplier, SpectralFeatureKernel


@final
class HodgeSpectralKernel(AbstractFiniteFeatureKernel):
    """Finite covariance over harmonic, exact and coexact coefficient sectors.

    Queries use full realization coefficient IDs: each sector's ``index_offset``
    plus its row index, not topology entity indices or physical point samples.
    """

    kernel: AbstractPositiveDefiniteKernel
    spectra: HodgeSectorSpectra
    sector_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        spectra: HodgeSectorSpectra,
        /,
        *,
        harmonic_multiplier: AbstractSpectralMultiplier | None = None,
        exact_multiplier: AbstractSpectralMultiplier | None = None,
        coexact_multiplier: AbstractSpectralMultiplier | None = None,
        harmonic_amplitude: ArrayLike = 1.0,
        exact_amplitude: ArrayLike = 1.0,
        coexact_amplitude: ArrayLike = 1.0,
        normalize_sectors: bool = True,
    ) -> None:
        if not isinstance(spectra, HodgeSectorSpectra):
            raise TypeError("spectra must be a HodgeSectorSpectra.")
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
            f"HodgeSpectralKernel[{self.spectra.realization_id};degree={self.spectra.degree};"
            f"sectors={sectors};boundary={self.spectra.boundary};{self.kernel.kernel_id}]"
        )


__all__ = ["HodgeSpectralKernel"]
