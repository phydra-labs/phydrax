#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...metrix import (
    AtlasCover,
    AtlasOverlap,
    ChartSupport,
    ChartTransition,
    ComplexCoordinateConvention,
    CoordinateAtlas,
    CoordinateChart,
    HermitianStructure,
    KahlerStructure,
    RiemannianMetric,
    standard_complex_structure,
)


class _ProjectiveTransition(StrictModule):
    source_convention: ComplexCoordinateConvention
    target_convention: ComplexCoordinateConvention
    complex_dimension: int = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    target_index: int = eqx.field(static=True)

    def __init__(
        self,
        complex_dimension: int,
        source_index: int,
        target_index: int,
        source_convention: ComplexCoordinateConvention,
        target_convention: ComplexCoordinateConvention,
        /,
    ):
        self.complex_dimension = int(complex_dimension)
        self.source_index = int(source_index)
        self.target_index = int(target_index)
        self.source_convention = source_convention
        self.target_convention = target_convention

    def __call__(self, coordinates: Array, /) -> Array:
        local = self.source_convention.to_complex(coordinates)
        homogeneous = jnp.ones((self.complex_dimension + 1,), dtype=local.dtype)
        source_axes = tuple(
            index
            for index in range(self.complex_dimension + 1)
            if index != self.source_index
        )
        homogeneous = homogeneous.at[jnp.asarray(source_axes)].set(local)
        denominator = homogeneous[self.target_index]
        target_axes = tuple(
            index
            for index in range(self.complex_dimension + 1)
            if index != self.target_index
        )
        target = homogeneous[jnp.asarray(target_axes)] / denominator
        return self.target_convention.to_real(target)


class _ProjectiveOverlapPredicate(StrictModule):
    convention: ComplexCoordinateConvention
    complex_dimension: int = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    target_index: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        convention: ComplexCoordinateConvention,
        complex_dimension: int,
        source_index: int,
        target_index: int,
        tolerance: float,
        /,
    ):
        self.convention = convention
        self.complex_dimension = int(complex_dimension)
        self.source_index = int(source_index)
        self.target_index = int(target_index)
        self.tolerance = float(tolerance)

    def __call__(self, coordinates: Array, /) -> Array:
        local = self.convention.to_complex(coordinates)
        source_axes = tuple(
            index
            for index in range(self.complex_dimension + 1)
            if index != self.source_index
        )
        target_position = source_axes.index(self.target_index)
        return jnp.abs(local[..., target_position]) > self.tolerance


class _FiniteSupport(StrictModule):
    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.all(jnp.isfinite(coordinates), axis=-1)


class _FubiniStudyMetricMap(StrictModule):
    convention: ComplexCoordinateConvention

    def __init__(self, convention: ComplexCoordinateConvention, /):
        self.convention = convention

    def __call__(self, coordinates: Array, /) -> Array:
        complex_coordinates = self.convention.to_complex(coordinates)
        dimension = self.convention.complex_dimension
        radius = 1.0 + jnp.real(jnp.vdot(complex_coordinates, complex_coordinates))
        identity = jnp.eye(dimension, dtype=complex_coordinates.dtype)
        hermitian = (
            radius * identity
            - jnp.conj(complex_coordinates)[:, None] * complex_coordinates[None, :]
        ) / radius**2
        real = jnp.real(hermitian)
        imaginary = jnp.imag(hermitian)
        return jnp.block([[real, -imaginary], [imaginary, real]])


class ComplexProjectiveAtlas(StrictModule):
    """Affine atlas and Fubini–Study references for CP^n."""

    cover: AtlasCover
    conventions: tuple[ComplexCoordinateConvention, ...]
    complex_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(self, complex_dimension: int, /, *, tolerance: float = 1e-8):
        dimension = int(complex_dimension)
        if dimension < 1:
            raise ValueError("Complex projective dimension must be positive.")
        charts = tuple(
            CoordinateChart(
                f"CP{dimension}:U{index}",
                tuple(
                    [f"x{axis}" for axis in range(dimension)]
                    + [f"y{axis}" for axis in range(dimension)]
                ),
            )
            for index in range(dimension + 1)
        )
        conventions = tuple(ComplexCoordinateConvention(chart) for chart in charts)
        transitions = []
        overlaps = []
        for source in range(dimension + 1):
            for target in range(dimension + 1):
                if source == target:
                    continue
                forward = _ProjectiveTransition(
                    dimension,
                    source,
                    target,
                    conventions[source],
                    conventions[target],
                )
                inverse = _ProjectiveTransition(
                    dimension,
                    target,
                    source,
                    conventions[target],
                    conventions[source],
                )
                transition = ChartTransition(
                    charts[source], charts[target], forward, inverse=inverse
                )
                transitions.append(transition)
                overlaps.append(
                    AtlasOverlap(
                        source,
                        target,
                        transition,
                        _ProjectiveOverlapPredicate(
                            conventions[source],
                            dimension,
                            source,
                            target,
                            tolerance,
                        ),
                        overlap_id=f"CP{dimension}:U{source}->U{target}",
                    )
                )
        atlas = CoordinateAtlas(charts, transitions)
        supports = tuple(
            ChartSupport(chart, _FiniteSupport(), support_id=f"{chart.name}:finite")
            for chart in charts
        )
        self.cover = AtlasCover(
            atlas,
            supports,
            overlaps,
            cover_id=f"complex-projective-cover:{dimension}",
        )
        self.conventions = conventions
        self.complex_dimension = dimension
        self.tolerance = float(tolerance)

    def potential(self, chart_index: int, coordinates: ArrayLike, /) -> Array:
        values = self.conventions[int(chart_index)].to_complex(coordinates)
        return jnp.log1p(jnp.sum(jnp.abs(values) ** 2, axis=-1))

    def metric(self, chart_index: int, /) -> RiemannianMetric:
        convention = self.conventions[int(chart_index)]
        return RiemannianMetric(_FubiniStudyMetricMap(convention), chart=convention.chart)

    def kahler_structure(self, chart_index: int, /) -> KahlerStructure:
        convention = self.conventions[int(chart_index)]
        return KahlerStructure(
            HermitianStructure(
                self.metric(chart_index), standard_complex_structure(convention)
            )
        )


__all__ = ["ComplexProjectiveAtlas"]
