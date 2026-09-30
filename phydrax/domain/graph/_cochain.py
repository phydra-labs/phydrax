#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed ``DomainFunction`` views over cochain graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import final, Literal

import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ...exterior._form_type import FormType
from ...typing import parse
from .._function import DomainFunction


CochainRepresentation = Literal["cochain"]
_FORM_TYPE_KEY = "_phydrax_form_type"
_REPRESENTATION_KEY = "_phydrax_form_representation"


@final
class _CochainDegreeMask(StrictModule):
    degree: int

    def __init__(self, degree: int) -> None:
        self.degree = degree

    def __call__(self, cell: Mapping[str, object]) -> Array:
        if not isinstance(cell, Mapping) or "cell_dim" not in cell:
            raise ValueError(
                "Cochain fields require mapping-valued graph nodes with 'cell_dim'."
            )
        return jnp.asarray(cell["cell_dim"]) == self.degree


def _graph_label(field: DomainFunction, /) -> str:
    labels = tuple(
        label
        for label in field.domain.labels
        if field.domain.coordinate(label).kind == "graph"
    )
    if len(labels) != 1:
        raise ValueError(
            f"Cochain fields require exactly one graph-domain label; found {labels!r}."
        )
    return labels[0]


def has_cochain_form_type(field: DomainFunction, /) -> bool:
    """Return whether a field declares canonical cochain semantics."""
    if not isinstance(field, DomainFunction):
        raise TypeError("has_cochain_form_type expects a DomainFunction.")
    return (
        isinstance(field.metadata.get(_FORM_TYPE_KEY), FormType)
        and field.metadata.get(_REPRESENTATION_KEY) == "cochain"
    )


def cochain_form_type(field: DomainFunction, /) -> FormType:
    """Return the declared scientific type of a graph cochain field."""
    if not isinstance(field, DomainFunction):
        raise TypeError("cochain_form_type expects a DomainFunction.")
    form_type = field.metadata.get(_FORM_TYPE_KEY)
    if not isinstance(form_type, FormType) or not has_cochain_form_type(field):
        raise ValueError("DomainFunction has no declared cochain form_type.")
    return form_type


def cochain_representation(field: DomainFunction, /) -> CochainRepresentation:
    """Return the admitted representation of a graph cochain field."""
    cochain_form_type(field)
    return parse(
        field.metadata[_REPRESENTATION_KEY], CochainRepresentation, "representation"
    )


def as_cochain_field(
    field: DomainFunction,
    form_type: FormType,
    /,
    *,
    representation: CochainRepresentation,
) -> DomainFunction:
    """Declare and degree-mask a graph-backed differential form.

    Cochain coordinates are cell integrals (point values in degree zero).
    Orientation and placement derive from the form type and realization, not
    independent orientation or sampling flags. Off-degree values are zero.
    """
    if not isinstance(field, DomainFunction):
        raise TypeError("as_cochain_field expects a DomainFunction.")
    if not isinstance(form_type, FormType):
        raise TypeError("form_type must be a FormType.")
    resolved = parse(representation, CochainRepresentation, "representation")
    graph_label = _graph_label(field)
    mask = field.domain.Function(graph_label)(_CochainDegreeMask(form_type.degree))
    return (field * mask).with_metadata(
        **{_FORM_TYPE_KEY: form_type, _REPRESENTATION_KEY: resolved}
    )


__all__ = [
    "CochainRepresentation",
    "as_cochain_field",
    "cochain_form_type",
    "cochain_representation",
    "has_cochain_form_type",
]
