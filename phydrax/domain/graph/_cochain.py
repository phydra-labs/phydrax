#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Degree-aware ``DomainFunction`` views over metric cochain graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp

from ..._strict import StrictModule
from .._domain import RelabeledDomain
from .._function import DomainFunction
from ._dataset import GraphDatasetDomain
from ._domain import GraphDomain


if TYPE_CHECKING:
    from ...graph import (
        CochainCellOrientation,
        CochainFieldSpec,
        CochainSampling,
        CochainSide,
    )


_COCHAIN_FIELD_SPEC_KEY = "_phydrax_cochain_field_spec"


class _CochainDegreeMask(StrictModule):
    degree: int

    def __init__(self, degree: int):
        self.degree = int(degree)

    def __call__(self, cell: Mapping[str, object]):
        if not isinstance(cell, Mapping) or "cell_dim" not in cell:
            raise ValueError(
                "Cochain fields require mapping-valued graph nodes with 'cell_dim'."
            )
        return jnp.asarray(cell["cell_dim"]) == self.degree


def _graph_label(field: DomainFunction, /) -> str:
    labels: list[str] = []
    for label in field.domain.labels:
        factor = field.domain.factor(label)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        if isinstance(factor, (GraphDomain, GraphDatasetDomain)):
            labels.append(label)
    if len(labels) != 1:
        raise ValueError(
            "Cochain fields require exactly one graph-domain label; "
            f"found {tuple(labels)!r}."
        )
    return labels[0]


def _spec_tuple(spec: CochainFieldSpec, /) -> tuple[int, str, str, str]:
    return (
        spec.degree,
        spec.complex_side,
        spec.cell_orientation,
        spec.sampling,
    )


def _attach_cochain_field_spec(
    field: DomainFunction,
    spec: CochainFieldSpec,
    /,
) -> DomainFunction:
    return field.with_metadata(**{_COCHAIN_FIELD_SPEC_KEY: _spec_tuple(spec)})


def cochain_field_spec(field: DomainFunction, /) -> CochainFieldSpec:
    """Return the declared cochain semantics of a domain field."""
    from ...graph import CochainFieldSpec

    if not isinstance(field, DomainFunction):
        raise TypeError("cochain_field_spec expects a DomainFunction.")
    encoded = field.metadata.get(_COCHAIN_FIELD_SPEC_KEY)
    if not isinstance(encoded, tuple) or len(encoded) != 4:
        raise ValueError("DomainFunction has no declared cochain field semantics.")
    degree, side, orientation, sampling = encoded
    return CochainFieldSpec(
        int(degree),
        complex_side=side,
        cell_orientation=orientation,
        sampling=sampling,
    )


def as_cochain_field(
    field: DomainFunction,
    spec: CochainFieldSpec | int,
    /,
    *,
    complex_side: CochainSide = "primal",
    cell_orientation: CochainCellOrientation | None = None,
    sampling: CochainSampling | None = None,
) -> DomainFunction:
    """Declare and degree-mask a graph-backed discrete differential form.

    Passing an integer degree requires explicit orientation and sampling semantics.
    Values outside the declared cell degree are identically zero, including when a
    downstream graph operator evaluates the field over the full cochain complex.
    """
    from ...graph import CochainFieldSpec

    if not isinstance(field, DomainFunction):
        raise TypeError("as_cochain_field expects a DomainFunction.")
    graph_label = _graph_label(field)
    if isinstance(spec, CochainFieldSpec):
        if cell_orientation is not None or sampling is not None or complex_side != "primal":
            raise ValueError(
                "Do not pass cochain semantic keywords with a CochainFieldSpec."
            )
        resolved = spec
    else:
        if cell_orientation is None or sampling is None:
            raise ValueError(
                "Integer cochain degrees require cell_orientation and sampling."
            )
        resolved = CochainFieldSpec(
            int(spec),
            complex_side=complex_side,
            cell_orientation=cell_orientation,
            sampling=sampling,
        )

    mask = DomainFunction(
        domain=field.domain,
        deps=(graph_label,),
        func=_CochainDegreeMask(resolved.degree),
        metadata={},
    )
    masked = field * mask
    return _attach_cochain_field_spec(masked, resolved)


__all__ = ["as_cochain_field", "cochain_field_spec"]
