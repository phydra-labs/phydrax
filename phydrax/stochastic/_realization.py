#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from math import prod
from typing import TypeAlias

import equinox as eqx
import numpy as np

from .._frozendict import frozendict
from .._strict import StrictModule
from ._fractional import FractionalGaussianRealization
from ._jump import PoissonClockRealization
from ._levy import LevyProcessRealization
from ._ou import OrnsteinUhlenbeckRealization
from ._wiener import WienerRealization


AtomicStochasticRealization: TypeAlias = (
    WienerRealization
    | OrnsteinUhlenbeckRealization
    | PoissonClockRealization
    | LevyProcessRealization
    | FractionalGaussianRealization
)


def _digest(parts: tuple[str, ...], /, *, prefix: bytes) -> str:
    digest = hashlib.sha256(prefix)
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_path_labels(realization: AtomicStochasticRealization, /) -> tuple[str, ...]:
    indices = np.asarray(realization.path_indices).reshape((-1,))
    if isinstance(realization, (WienerRealization, OrnsteinUhlenbeckRealization)):
        signs = np.asarray(realization.path_signs).reshape((-1,))
        return tuple(
            f"{realization.realization_id}:path:{int(index)}:sign:{float(sign):g}"
            for index, sign in zip(indices, signs, strict=True)
        )
    return tuple(f"{realization.realization_id}:path:{int(index)}" for index in indices)


def _atomic_independence_labels(
    realization: AtomicStochasticRealization,
    /,
) -> tuple[str, ...]:
    indices = np.asarray(realization.path_indices).reshape((-1,))
    return tuple(
        f"{realization.coupling_id}:independent-path:{int(index)}" for index in indices
    )


class CompositeStochasticRealization(StrictModule):
    """Named atomic realizations sharing one sample layout and support."""

    components: frozendict[str, AtomicStochasticRealization]
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    support: tuple[float, float] = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: Mapping[str, AtomicStochasticRealization],
        /,
    ):
        resolved = frozendict(components)
        if not resolved:
            raise ValueError("CompositeStochasticRealization requires components.")
        if any(not isinstance(name, str) or not name for name in resolved):
            raise ValueError("Composite realization names must be non-empty strings.")
        values = tuple(resolved.values())
        if any(
            not isinstance(
                value,
                (
                    WienerRealization,
                    OrnsteinUhlenbeckRealization,
                    PoissonClockRealization,
                    LevyProcessRealization,
                    FractionalGaussianRealization,
                ),
            )
            for value in values
        ):
            raise TypeError(
                "Composite components must be Wiener, OU, Poisson, Lévy, or "
                "fractional Gaussian realizations."
            )
        sample_shape = values[0].sample_shape
        support = values[0].support
        if any(value.sample_shape != sample_shape for value in values[1:]):
            raise ValueError("Composite realization sample shapes must match.")
        if any(value.support != support for value in values[1:]):
            raise ValueError("Composite realization supports must match.")
        names = tuple(resolved)
        realization_parts = tuple(
            f"{name}:{resolved[name].realization_id}" for name in names
        )
        coupling_parts = tuple(f"{name}:{resolved[name].coupling_id}" for name in names)
        self.components = resolved
        self.sample_shape = sample_shape
        self.support = support
        self.realization_id = _digest(
            realization_parts,
            prefix=b"phydrax-composite-realization\0",
        )
        self.coupling_id = _digest(
            coupling_parts,
            prefix=b"phydrax-composite-coupling\0",
        )

    @property
    def num_paths(self) -> int:
        return prod(self.sample_shape) if self.sample_shape else 1

    def component(self, name: str, /) -> AtomicStochasticRealization:
        if name not in self.components:
            raise KeyError(f"Unknown realization component {name!r}.")
        return self.components[name]

    @property
    def path_labels(self) -> tuple[str, ...]:
        component_labels = tuple(
            _atomic_path_labels(component) for component in self.components.values()
        )
        return tuple(
            _digest(
                tuple(labels[index] for labels in component_labels),
                prefix=b"phydrax-composite-path\0",
            )
            for index in range(self.num_paths)
        )

    @property
    def independence_labels(self) -> tuple[str, ...]:
        component_labels = tuple(
            _atomic_independence_labels(component)
            for component in self.components.values()
        )
        return tuple(
            _digest(
                tuple(labels[index] for labels in component_labels),
                prefix=b"phydrax-composite-independent-path\0",
            )
            for index in range(self.num_paths)
        )


StochasticRealization: TypeAlias = (
    WienerRealization
    | OrnsteinUhlenbeckRealization
    | PoissonClockRealization
    | LevyProcessRealization
    | FractionalGaussianRealization
    | CompositeStochasticRealization
)


def is_stochastic_realization(value: object, /) -> bool:
    return isinstance(
        value,
        (
            WienerRealization,
            OrnsteinUhlenbeckRealization,
            PoissonClockRealization,
            LevyProcessRealization,
            FractionalGaussianRealization,
            CompositeStochasticRealization,
        ),
    )


def realization_path_labels(
    case_id: str,
    realization: StochasticRealization | None,
    realization_shape: tuple[int, ...],
    /,
) -> tuple[str, ...]:
    """Return stable labels aligned with one trajectory realization layout."""
    count = prod(realization_shape) if realization_shape else 1
    if realization is None:
        return tuple(f"{case_id}:path:{index}" for index in range(count))
    if isinstance(realization, CompositeStochasticRealization):
        labels = realization.path_labels
    else:
        labels = _atomic_path_labels(realization)
    if len(labels) != count:
        raise ValueError("Realization path labels do not match trajectory shape.")
    return labels


def realization_independence_labels(
    realization: StochasticRealization | None,
    realization_shape: tuple[int, ...],
    /,
) -> tuple[str | None, ...]:
    """Return independent Monte Carlo cluster labels for one realization layout.

    Antithetic Wiener paths deliberately share a label. Unknown realization
    provenance remains unknown rather than being inferred from array position.
    """
    count = prod(realization_shape) if realization_shape else 1
    if realization is None:
        return (None,) * count
    if isinstance(realization, CompositeStochasticRealization):
        labels = realization.independence_labels
    else:
        labels = _atomic_independence_labels(realization)
    if len(labels) != count:
        raise ValueError("Independence labels do not match realization shape.")
    return labels


__all__ = [
    "AtomicStochasticRealization",
    "CompositeStochasticRealization",
    "StochasticRealization",
    "is_stochastic_realization",
    "realization_independence_labels",
    "realization_path_labels",
]
