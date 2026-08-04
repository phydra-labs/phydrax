#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx

from .._strict import StrictModule


DesignName = Literal[
    "uniform",
    "latin_hypercube",
    "halton",
    "halton_scrambled",
    "hammersley",
    "sobol",
    "sobol_scrambled",
]

SUPPORTED_DESIGNS: tuple[DesignName, ...] = (
    "uniform",
    "latin_hypercube",
    "halton",
    "halton_scrambled",
    "hammersley",
    "sobol",
    "sobol_scrambled",
)

DESIGN_ALGORITHM_VERSION = 1


class IIDDesign(StrictModule):
    """Independent random points in a unit cube."""

    def __init__(self):
        pass


class LatinHypercubeDesign(StrictModule):
    """Randomized Latin-hypercube stratification in a unit cube."""

    def __init__(self):
        pass


class HammersleyDesign(StrictModule):
    """Deterministic count-dependent Hammersley point set."""

    def __init__(self):
        pass


class HaltonDesign(StrictModule):
    """Halton sequence with optional Owen scrambling."""

    scrambled: bool = eqx.field(static=True)

    def __init__(self, *, scrambled: bool = False):
        self.scrambled = bool(scrambled)


class SobolDesign(StrictModule):
    """Sobol digital net with optional LMS+shift scrambling."""

    scrambled: bool = eqx.field(static=True)

    def __init__(self, *, scrambled: bool = False):
        self.scrambled = bool(scrambled)


class RandomizedQMCDesign(StrictModule):
    """Sobol or Halton design with integration replicate semantics."""

    sequence: Literal["sobol", "halton"] = eqx.field(static=True)
    scrambled: bool = eqx.field(static=True)
    num_replicates: int = eqx.field(static=True)
    allow_arbitrary_count: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        sequence: Literal["sobol", "halton"] = "sobol",
        scrambled: bool = True,
        num_replicates: int = 8,
        allow_arbitrary_count: bool = False,
    ):
        if sequence not in ("sobol", "halton"):
            raise ValueError("QMC sequence must be 'sobol' or 'halton'.")
        replicas = int(num_replicates)
        if replicas < 1:
            raise ValueError("num_replicates must be positive.")
        if not scrambled and replicas != 1:
            raise ValueError("Unscrambled QMC has one deterministic replicate.")
        self.sequence = sequence
        self.scrambled = bool(scrambled)
        self.num_replicates = replicas
        self.allow_arbitrary_count = bool(allow_arbitrary_count)


class AntitheticDesign(StrictModule):
    """Pair a base design through an explicit measure-preserving involution."""

    base: Any
    involution: Any

    def __init__(self, base: Any | None = None, *, involution: Callable | None = None):
        base_ = IIDDesign() if base is None else base
        if not isinstance(base_, (IIDDesign, LatinHypercubeDesign)):
            raise TypeError(
                "AntitheticDesign base must be IIDDesign or LatinHypercubeDesign."
            )
        self.base = base_
        self.involution = involution


UnitDesign: TypeAlias = (
    IIDDesign
    | LatinHypercubeDesign
    | HammersleyDesign
    | HaltonDesign
    | SobolDesign
    | RandomizedQMCDesign
)
DesignLike: TypeAlias = DesignName | str | UnitDesign


class DesignCapabilities(StrictModule):
    """Static execution and sequence properties of a reference design."""

    randomized: bool = eqx.field(static=True)
    count_dependent: bool = eqx.field(static=True)
    prefix_stable: bool = eqx.field(static=True)
    random_access: bool = eqx.field(static=True)
    factorwise_composable: bool = eqx.field(static=True)
    jax_native: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        randomized: bool,
        count_dependent: bool,
        prefix_stable: bool,
        random_access: bool,
        factorwise_composable: bool,
        jax_native: bool,
    ):
        self.randomized = bool(randomized)
        self.count_dependent = bool(count_dependent)
        self.prefix_stable = bool(prefix_stable)
        self.random_access = bool(random_access)
        self.factorwise_composable = bool(factorwise_composable)
        self.jax_native = bool(jax_native)


def normalize_design_name(name: str, /) -> DesignName:
    """Normalize and validate a reference-design name."""
    normalized = str(name).lower()
    if normalized not in SUPPORTED_DESIGNS:
        raise ValueError(f"design must be one of {SUPPORTED_DESIGNS}; got {name!r}.")
    return normalized


def resolve_design(design: DesignLike, /) -> UnitDesign:
    """Resolve a string shorthand to its canonical typed design."""
    if isinstance(
        design,
        (
            IIDDesign,
            LatinHypercubeDesign,
            HammersleyDesign,
            HaltonDesign,
            SobolDesign,
            RandomizedQMCDesign,
        ),
    ):
        return design
    name = normalize_design_name(design)
    if name == "uniform":
        return IIDDesign()
    if name == "latin_hypercube":
        return LatinHypercubeDesign()
    if name == "hammersley":
        return HammersleyDesign()
    if name.startswith("halton"):
        return HaltonDesign(scrambled=name.endswith("_scrambled"))
    return SobolDesign(scrambled=name.endswith("_scrambled"))


def design_name(design: DesignLike, /) -> DesignName:
    """Return the stable string identity of a canonical design."""
    resolved = resolve_design(design)
    if isinstance(resolved, IIDDesign):
        return "uniform"
    if isinstance(resolved, LatinHypercubeDesign):
        return "latin_hypercube"
    if isinstance(resolved, HammersleyDesign):
        return "hammersley"
    if isinstance(resolved, (HaltonDesign, SobolDesign)):
        family = "halton" if isinstance(resolved, HaltonDesign) else "sobol"
        return family + ("_scrambled" if resolved.scrambled else "")
    return resolved.sequence + ("_scrambled" if resolved.scrambled else "")


def design_signature(design: DesignLike, /) -> str:
    """Return the replay-relevant identity of a reference design."""
    return f"{design_name(design)}:v{DESIGN_ALGORITHM_VERSION}"


def design_capabilities(design: DesignLike, /) -> DesignCapabilities:
    """Return explicit execution capabilities for a reference design."""
    resolved = resolve_design(design)
    if isinstance(resolved, IIDDesign):
        return DesignCapabilities(
            randomized=True,
            count_dependent=False,
            prefix_stable=False,
            random_access=False,
            factorwise_composable=True,
            jax_native=True,
        )
    if isinstance(resolved, LatinHypercubeDesign):
        return DesignCapabilities(
            randomized=True,
            count_dependent=True,
            prefix_stable=False,
            random_access=False,
            factorwise_composable=True,
            jax_native=False,
        )
    if isinstance(resolved, HammersleyDesign):
        return DesignCapabilities(
            randomized=False,
            count_dependent=True,
            prefix_stable=False,
            random_access=False,
            factorwise_composable=False,
            jax_native=False,
        )
    scrambled = resolved.scrambled
    return DesignCapabilities(
        randomized=scrambled,
        count_dependent=False,
        prefix_stable=True,
        random_access=True,
        factorwise_composable=False,
        jax_native=False,
    )


__all__ = [
    "AntitheticDesign",
    "DESIGN_ALGORITHM_VERSION",
    "DesignCapabilities",
    "DesignLike",
    "DesignName",
    "HaltonDesign",
    "HammersleyDesign",
    "IIDDesign",
    "LatinHypercubeDesign",
    "RandomizedQMCDesign",
    "SUPPORTED_DESIGNS",
    "SobolDesign",
    "UnitDesign",
    "design_capabilities",
    "design_name",
    "design_signature",
    "normalize_design_name",
    "resolve_design",
]
