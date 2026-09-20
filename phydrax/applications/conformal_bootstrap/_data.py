#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed conformal data, exchanged sectors, and crossing channels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...tensor_network import RepresentationCategory


class ExternalScalarOperator(StrictModule):
    """One scalar primary in a declared global-symmetry representation."""

    label: str = eqx.field(static=True)
    scaling_dimension: float = eqx.field(static=True)
    representation_label: str = eqx.field(static=True)
    reality: Literal["real", "complex", "pseudoreal"] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        scaling_dimension: float,
        representation_label: str,
        /,
        *,
        reality: Literal["real", "complex", "pseudoreal"] = "real",
    ):
        name = str(label)
        dimension = float(scaling_dimension)
        representation = str(representation_label)
        reality_value = str(reality)
        if not name or not representation:
            raise ValueError("Operator and representation labels must be non-empty.")
        if not np.isfinite(dimension) or dimension <= 0.0:
            raise ValueError("External scaling dimensions must be finite and positive.")
        if reality_value not in {"real", "complex", "pseudoreal"}:
            raise ValueError("Unsupported representation reality.")
        self.label = name
        self.scaling_dimension = dimension
        self.representation_label = representation
        self.reality = reality_value
        self.operator_id = canonical_fingerprint(
            {
                "kind": "external-scalar-primary",
                "label": name,
                "scaling_dimension": dimension,
                "representation": representation,
                "reality": reality_value,
            }
        )


class ExchangedOperatorSector(StrictModule):
    """One exchanged representation/spin family and its declared lower gap."""

    label: str = eqx.field(static=True)
    representation_label: str = eqx.field(static=True)
    spins: tuple[int, ...] = eqx.field(static=True)
    parity: Literal["even", "odd", "mixed"] = eqx.field(static=True)
    minimum_dimension: float = eqx.field(static=True)
    includes_identity: bool = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        representation_label: str,
        spins: Sequence[int],
        /,
        *,
        parity: Literal["even", "odd", "mixed"] = "mixed",
        minimum_dimension: float,
        includes_identity: bool = False,
    ):
        name = str(label)
        representation = str(representation_label)
        spin_values = tuple(spins)
        parity_value = str(parity)
        minimum = float(minimum_dimension)
        identity = bool(includes_identity)
        if not name or not representation:
            raise ValueError("Sector and representation labels must be non-empty.")
        if not spin_values or any(value < 0 for value in spin_values):
            raise ValueError("Exchanged spins must be nonempty and non-negative.")
        if (
            len(set(spin_values)) != len(spin_values)
            or tuple(sorted(spin_values)) != spin_values
        ):
            raise ValueError("Exchanged spins must be unique and increasing.")
        if parity_value not in {"even", "odd", "mixed"}:
            raise ValueError("Unsupported exchanged-sector parity.")
        if parity_value == "even" and any(value % 2 for value in spin_values):
            raise ValueError("An even sector may contain only even spins.")
        if parity_value == "odd" and any(value % 2 == 0 for value in spin_values):
            raise ValueError("An odd sector may contain only odd spins.")
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_dimension must be finite and non-negative.")
        if identity and (0 not in spin_values or minimum != 0.0):
            raise ValueError(
                "An identity sector must include spin zero at dimension zero."
            )
        self.label = name
        self.representation_label = representation
        self.spins = spin_values
        self.parity = parity_value
        self.minimum_dimension = minimum
        self.includes_identity = identity
        self.sector_id = canonical_fingerprint(
            {
                "kind": "exchanged-operator-sector",
                "label": name,
                "representation": representation,
                "spins": spin_values,
                "parity": parity_value,
                "minimum_dimension": minimum,
                "includes_identity": identity,
            }
        )


class CrossingChannel(StrictModule):
    """One ordered four-point permutation and crossing-matrix label."""

    label: str = eqx.field(static=True)
    permutation: tuple[int, int, int, int] = eqx.field(static=True)
    involutive: bool = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        permutation: Sequence[int],
        /,
        *,
        involutive: bool,
    ):
        name = str(label)
        values = tuple(permutation)
        if not name:
            raise ValueError("A crossing-channel label must be non-empty.")
        if len(values) != 4 or tuple(sorted(values)) != (0, 1, 2, 3):
            raise ValueError("A crossing channel must be a permutation of four legs.")
        is_involutive = all(values[values[index]] == index for index in range(4))
        if bool(involutive) != is_involutive:
            raise ValueError("Crossing-channel involutive metadata is inconsistent.")
        self.label = name
        self.permutation = values
        self.involutive = is_involutive
        self.channel_id = canonical_fingerprint(
            {
                "kind": "crossing-channel",
                "label": name,
                "permutation": values,
                "involutive": is_involutive,
            }
        )


class ConformalDataPlan(StrictModule):
    """Finite semantic declaration for one scalar four-point bootstrap problem."""

    external_operators: tuple[ExternalScalarOperator, ...]
    exchanged_sectors: tuple[ExchangedOperatorSector, ...]
    channels: tuple[CrossingChannel, ...]
    category: RepresentationCategory | None
    spacetime_dimension: float = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    block_normalization: str = eqx.field(static=True)
    maximum_tensor_structures: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spacetime_dimension: float,
        external_operators: Sequence[ExternalScalarOperator],
        exchanged_sectors: Sequence[ExchangedOperatorSector],
        channels: Sequence[CrossingChannel],
        /,
        *,
        category: RepresentationCategory | None,
        coordinate_convention: str = "z-zbar-euclidean",
        block_normalization: str = "unit-leading-radial-primary",
        maximum_tensor_structures: int = 256,
    ):
        dimension = float(spacetime_dimension)
        externals = tuple(external_operators)
        sectors = tuple(exchanged_sectors)
        channel_values = tuple(channels)
        coordinate = str(coordinate_convention)
        normalization = str(block_normalization)
        maximum = int(maximum_tensor_structures)
        if not np.isfinite(dimension) or dimension <= 1.0:
            raise ValueError("Global scalar bootstrap dimensions must exceed one.")
        if len(externals) != 4 or any(
            not isinstance(value, ExternalScalarOperator) for value in externals
        ):
            raise TypeError("Exactly four ExternalScalarOperator values are required.")
        if not sectors or any(
            not isinstance(value, ExchangedOperatorSector) for value in sectors
        ):
            raise TypeError("At least one exchanged operator sector is required.")
        if not channel_values or any(
            not isinstance(value, CrossingChannel) for value in channel_values
        ):
            raise TypeError("At least one crossing channel is required.")
        if len({value.label for value in sectors}) != len(sectors):
            raise ValueError("Exchanged-sector labels must be unique.")
        if len({value.label for value in channel_values}) != len(channel_values):
            raise ValueError("Crossing-channel labels must be unique.")
        if not coordinate or not normalization or maximum < 1:
            raise ValueError("Conformal conventions and resource limits are required.")
        if category is not None and not isinstance(category, RepresentationCategory):
            raise TypeError("category must be RepresentationCategory or None.")
        if category is not None:
            labels = {value.label for value in category.irreps}
            if any(value.representation_label not in labels for value in externals):
                raise ValueError("An external representation is outside the category.")
            if any(value.representation_label not in labels for value in sectors):
                raise ValueError("An exchanged representation is outside the category.")
            for first, second in ((0, 1), (2, 3)):
                allowed = {
                    label
                    for label, _ in category.fusion(
                        externals[first].representation_label,
                        externals[second].representation_label,
                    )
                }
                declared = {value.representation_label for value in sectors}
                if not declared <= allowed:
                    raise ValueError(
                        "An exchanged sector is incompatible with external fusion."
                    )
        self.external_operators = externals
        self.exchanged_sectors = sectors
        self.channels = channel_values
        self.category = category
        self.spacetime_dimension = dimension
        self.coordinate_convention = coordinate
        self.block_normalization = normalization
        self.maximum_tensor_structures = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scalar-four-point-conformal-data-plan",
                "spacetime_dimension": dimension,
                "external_operators": tuple(value.operator_id for value in externals),
                "exchanged_sectors": tuple(value.sector_id for value in sectors),
                "channels": tuple(value.channel_id for value in channel_values),
                "category": None if category is None else category.category_id,
                "coordinate_convention": coordinate,
                "block_normalization": normalization,
                "maximum_tensor_structures": maximum,
            }
        )

    @property
    def external_dimension_differences(self) -> tuple[float, float]:
        return (
            self.external_operators[0].scaling_dimension
            - self.external_operators[1].scaling_dimension,
            self.external_operators[2].scaling_dimension
            - self.external_operators[3].scaling_dimension,
        )


__all__ = [
    "ConformalDataPlan",
    "CrossingChannel",
    "ExchangedOperatorSector",
    "ExternalScalarOperator",
]
