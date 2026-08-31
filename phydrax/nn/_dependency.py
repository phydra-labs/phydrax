#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authored spatial-dependency evidence for configured neural operators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable, TypeAlias

import numpy as np


OperatorDependencyKind: TypeAlias = Literal["pointwise", "finite", "global", "unknown"]
OperatorDependencyEvidence: TypeAlias = Literal["exact", "conservative"]


@runtime_checkable
class OperatorDependencyProvider(Protocol):
    """Runtime protocol for instance-authored spatial dependency evidence."""

    def dependency_support(
        self,
        axes: Sequence[Any] | None = None,
        /,
    ) -> OperatorDependencySupport: ...


@dataclass(frozen=True, slots=True)
class AxisDependencyReach:
    """Directional lattice reach from one output site along one spatial axis."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("Dependency reach must be finite.")
        if lower < 0.0 or upper < 0.0:
            raise ValueError("Dependency reach must be non-negative.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def sequential(self, other: AxisDependencyReach, /) -> AxisDependencyReach:
        """Compose two sequential local maps by adding directional reach."""
        if not isinstance(other, AxisDependencyReach):
            raise TypeError("Sequential dependency reach requires AxisDependencyReach.")
        return AxisDependencyReach(
            self.lower + other.lower,
            self.upper + other.upper,
        )

    def parallel(self, other: AxisDependencyReach, /) -> AxisDependencyReach:
        """Merge parallel local maps by taking their directional envelope."""
        if not isinstance(other, AxisDependencyReach):
            raise TypeError("Parallel dependency reach requires AxisDependencyReach.")
        return AxisDependencyReach(
            max(self.lower, other.lower),
            max(self.upper, other.upper),
        )


@dataclass(frozen=True, slots=True)
class OperatorDependencySupport:
    """Spatial support classification and authored evidence for one operator.

    ``reach`` is measured in lattice sites. ``scale`` records the physical size
    represented by one site on each axis; changing it is always explicit through
    :meth:`rescaled`. Global and unknown supports intentionally carry no finite
    reach.
    """

    kind: OperatorDependencyKind
    dimension: int | None = None
    reach: tuple[AxisDependencyReach, ...] = ()
    scale: tuple[float, ...] = ()
    evidence: OperatorDependencyEvidence = "conservative"

    def __post_init__(self) -> None:
        if self.kind not in ("pointwise", "finite", "global", "unknown"):
            raise ValueError(
                "Dependency kind must be 'pointwise', 'finite', 'global', or 'unknown'."
            )
        if self.evidence not in ("exact", "conservative"):
            raise ValueError("Dependency evidence must be 'exact' or 'conservative'.")
        reach = tuple(self.reach)
        if any(not isinstance(axis, AxisDependencyReach) for axis in reach):
            raise TypeError("reach must contain AxisDependencyReach entries.")
        dimension = self.dimension
        if dimension is None and reach:
            dimension = len(reach)
        if dimension is not None:
            dimension = int(dimension)
            if dimension <= 0:
                raise ValueError("Dependency dimension must be positive when known.")
        if self.kind in ("pointwise", "finite"):
            if dimension is None:
                raise ValueError(
                    "Pointwise and finite dependency support needs a dimension."
                )
            if not reach:
                if self.kind == "finite":
                    raise ValueError(
                        "Finite dependency support requires directional reach."
                    )
                reach = (AxisDependencyReach(0.0, 0.0),) * dimension
            if len(reach) != dimension:
                raise ValueError("Dependency reach must contain one entry per dimension.")
        elif reach:
            raise ValueError(
                "Global and unknown dependency support cannot carry finite reach."
            )
        if self.kind == "finite" and all(
            axis.lower == 0.0 and axis.upper == 0.0 for axis in reach
        ):
            object.__setattr__(self, "kind", "pointwise")
        scale = tuple(float(value) for value in self.scale)
        if dimension is None:
            if scale:
                raise ValueError(
                    "Unknown-dimensional dependency support cannot carry scale."
                )
        else:
            if not scale:
                scale = (1.0,) * dimension
            if len(scale) != dimension:
                raise ValueError("Dependency scale must contain one entry per dimension.")
            if any(not math.isfinite(value) or value <= 0.0 for value in scale):
                raise ValueError("Dependency scales must be finite and positive.")
        if self.kind == "unknown" and self.evidence != "conservative":
            raise ValueError("Unknown dependency support is necessarily conservative.")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "reach", reach)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def pointwise(
        cls,
        dimension: int,
        /,
        *,
        scale: Sequence[float] = (),
        evidence: OperatorDependencyEvidence = "exact",
    ) -> OperatorDependencySupport:
        return cls("pointwise", dimension, scale=tuple(scale), evidence=evidence)

    @classmethod
    def finite(
        cls,
        reach: Sequence[AxisDependencyReach],
        /,
        *,
        scale: Sequence[float] = (),
        evidence: OperatorDependencyEvidence = "exact",
    ) -> OperatorDependencySupport:
        reach_value = tuple(reach)
        return cls(
            "finite",
            len(reach_value),
            reach_value,
            tuple(scale),
            evidence,
        )

    @classmethod
    def global_(
        cls,
        dimension: int,
        /,
        *,
        scale: Sequence[float] = (),
        evidence: OperatorDependencyEvidence = "exact",
    ) -> OperatorDependencySupport:
        return cls("global", dimension, scale=tuple(scale), evidence=evidence)

    @classmethod
    def unknown(
        cls,
        dimension: int | None = None,
        /,
        *,
        scale: Sequence[float] = (),
    ) -> OperatorDependencySupport:
        return cls("unknown", dimension, scale=tuple(scale), evidence="conservative")

    def _common_frame(
        self, other: OperatorDependencySupport, /
    ) -> tuple[int | None, tuple[float, ...], OperatorDependencyEvidence]:
        if not isinstance(other, OperatorDependencySupport):
            raise TypeError("Dependency composition requires OperatorDependencySupport.")
        if self.dimension is None:
            dimension = other.dimension
            scale = other.scale
        elif other.dimension is None:
            dimension = self.dimension
            scale = self.scale
        else:
            if self.dimension != other.dimension:
                raise ValueError("Dependency dimensions must agree before composition.")
            if self.scale != other.scale:
                raise ValueError(
                    "Dependency scales must be explicitly rescaled before composition."
                )
            dimension = self.dimension
            scale = self.scale
        evidence: OperatorDependencyEvidence = (
            "exact"
            if self.evidence == "exact" and other.evidence == "exact"
            else "conservative"
        )
        return dimension, scale, evidence

    def sequential(
        self, other: OperatorDependencySupport, /
    ) -> OperatorDependencySupport:
        """Compose maps in execution order, summing finite directional reach."""
        dimension, scale, evidence = self._common_frame(other)
        if self.kind == "unknown" or other.kind == "unknown":
            return OperatorDependencySupport.unknown(
                dimension,
                scale=scale,
            )
        if self.kind == "global" or other.kind == "global":
            assert dimension is not None
            return OperatorDependencySupport.global_(
                dimension,
                scale=scale,
                evidence=evidence,
            )
        return OperatorDependencySupport.finite(
            tuple(
                left.sequential(right)
                for left, right in zip(self.reach, other.reach, strict=True)
            ),
            scale=scale,
            evidence=evidence,
        )

    def parallel(self, other: OperatorDependencySupport, /) -> OperatorDependencySupport:
        """Merge branches by taking their finite directional envelope."""
        dimension, scale, evidence = self._common_frame(other)
        if self.kind == "unknown" or other.kind == "unknown":
            return OperatorDependencySupport.unknown(
                dimension,
                scale=scale,
            )
        if self.kind == "global" or other.kind == "global":
            assert dimension is not None
            return OperatorDependencySupport.global_(
                dimension,
                scale=scale,
                evidence=evidence,
            )
        return OperatorDependencySupport.finite(
            tuple(
                left.parallel(right)
                for left, right in zip(self.reach, other.reach, strict=True)
            ),
            scale=scale,
            evidence=evidence,
        )

    def rescaled(self, scale: float | Sequence[float], /) -> OperatorDependencySupport:
        """Attach an explicit physical scale to the lattice-unit reach."""
        if self.dimension is None:
            raise ValueError("Cannot rescale dependency support with unknown dimension.")
        scale_value = (
            (float(scale),) * self.dimension
            if isinstance(scale, (int, float))
            else tuple(float(value) for value in scale)
        )
        return OperatorDependencySupport(
            self.kind,
            self.dimension,
            self.reach,
            scale_value,
            self.evidence,
        )

    def saturated_periodic(
        self, axis_sizes: Sequence[int | None], /
    ) -> OperatorDependencySupport:
        """Cap finite reach on periodic axes and classify full-domain support."""
        if self.dimension is None:
            raise ValueError("Periodic saturation requires a known dependency dimension.")
        sizes = tuple(axis_sizes)
        if len(sizes) != self.dimension:
            raise ValueError("Periodic axis sizes must match dependency dimension.")
        if any(size is not None and int(size) <= 0 for size in sizes):
            raise ValueError("Periodic axis sizes must be positive.")
        if self.kind not in ("pointwise", "finite"):
            return self
        saturated = []
        reach = []
        for axis, size in zip(self.reach, sizes, strict=True):
            if size is None or axis.lower + axis.upper + 1.0 < int(size):
                saturated.append(False)
                reach.append(axis)
                continue
            maximum = float(int(size) - 1)
            lower = min(axis.lower, maximum)
            reach.append(AxisDependencyReach(lower, maximum - lower))
            saturated.append(True)
        if all(saturated):
            return OperatorDependencySupport.global_(
                self.dimension,
                scale=self.scale,
                evidence=self.evidence,
            )
        return OperatorDependencySupport.finite(
            reach,
            scale=self.scale,
            evidence=self.evidence,
        )

    def on_axes(self, axes: Sequence[Any], /) -> OperatorDependencySupport:
        """Bind lattice support to validated uniform axes and periodic extents."""
        axes_value = tuple(axes)
        if self.dimension is None or len(axes_value) != self.dimension:
            raise ValueError("Dependency axes must match the known support dimension.")
        scales = []
        periodic_sizes: list[int | None] = []
        for axis in axes_value:
            nodes = np.asarray(axis.nodes)
            if nodes.size < 2 or np.any(~np.isfinite(nodes)):
                raise ValueError("Dependency axes require at least two finite nodes.")
            spacing = np.diff(nodes)
            if np.any(spacing <= 0.0) or not np.allclose(
                spacing,
                np.mean(spacing),
                rtol=1e-5,
                atol=1e-8,
            ):
                raise ValueError("Dependency axes must be strictly ordered and uniform.")
            scales.append(float(np.mean(spacing)))
            periodic_sizes.append(int(nodes.size) if axis.periodic else None)
        return self.rescaled(scales).saturated_periodic(periodic_sizes)


def operator_dependency_support(
    model: Any,
    axes: Sequence[Any] | None = None,
    /,
) -> OperatorDependencySupport:
    """Return instance-authored evidence, defaulting to conservative unknown."""
    if isinstance(model, OperatorDependencyProvider):
        return model.dependency_support(axes)
    dimension = None if axes is None else len(tuple(axes))
    return OperatorDependencySupport.unknown(dimension)


__all__ = [
    "AxisDependencyReach",
    "OperatorDependencyEvidence",
    "OperatorDependencyKind",
    "OperatorDependencyProvider",
    "OperatorDependencySupport",
    "operator_dependency_support",
]
