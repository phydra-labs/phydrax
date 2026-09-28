#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Externally prescribed electromagnetic fields gathered by particle pushes.

An external field source supplies ``E`` and ``B`` at particle positions and
lab times without owning a field solve: magnets, insertion devices, tabulated
field maps, and prescribed laser envelopes implement it, and particle pushes
consume it gather-only. Fields are expressed in the scale of the consumer that
binds the source; the source reports where its field is defined so that
unsupported samples are never mistaken for field-free space.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from jax import Array

from ..._strict import StrictModule
from ...typing import Bool, Dim, Float64


class _QueryDim(Dim, minimum=1):
    """Gathered particle positions."""


class ExternalFieldSample(StrictModule):
    """External fields gathered at ``[N, 3]`` positions.

    ``support[N]`` is false where the source has no valid field (outside a
    tabulated map, beyond a magnet aperture, outside an analytic model's domain
    of validity). Unsupported samples carry zero fields and MUST be treated as
    refused by consumers, not as field-free space.
    """

    __strict_contract__ = True

    electric: Float64[_QueryDim, Literal[3]]
    magnetic: Float64[_QueryDim, Literal[3]]
    support: Bool[_QueryDim]


@runtime_checkable
class ExternalFieldSource(Protocol):
    """Capability of supplying prescribed ``E`` and ``B`` at positions and times.

    ``external_fields(positions[N, 3], times[N])`` returns the fields at every
    position at its own lab time (per-lane times need not be equal). The
    method must be traceable: consumers call it inside compiled time loops.
    ``source_id`` is the canonical fingerprint of the source.
    """

    @property
    def source_id(self) -> str: ...

    def external_fields(
        self, positions: Array, times: Array, /
    ) -> ExternalFieldSample: ...


__all__ = ["ExternalFieldSample", "ExternalFieldSource"]
