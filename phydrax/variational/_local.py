#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


class FieldJetSpec(StrictModule):
    """Static local quantities required from one semantic field."""

    field_name: str = eqx.field(static=True)
    value: bool = eqx.field(static=True)
    gradient: bool = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        /,
        *,
        value: bool = False,
        gradient: bool = False,
    ):
        name = str(field_name)
        if not name:
            raise ValueError("field_name must be non-empty.")
        if not value and not gradient:
            raise ValueError("A field jet must request value, gradient, or both.")
        self.field_name = name
        self.value = bool(value)
        self.gradient = bool(gradient)


class LocalFieldJet(StrictModule):
    """Pointwise field value and spatial gradient in canonical axis order."""

    value: Array | None
    gradient: Array | None

    def __init__(
        self,
        *,
        value: Array | None = None,
        gradient: Array | None = None,
    ):
        if value is None and gradient is None:
            raise ValueError("A local field jet must contain value or gradient.")
        self.value = value
        self.gradient = gradient


class LocalGeometry(StrictModule):
    """Pointwise physical geometry available to a local density."""

    points: Array
    normal: Array | None

    def __init__(self, points: Array, /, *, normal: Array | None = None):
        points_ = jnp.asarray(points)
        if points_.ndim < 1:
            raise ValueError("Local geometry points must end in a coordinate axis.")
        if normal is not None:
            normal_ = jnp.asarray(normal)
            if normal_.shape != points_.shape:
                raise ValueError("Local geometry normals must match point shape.")
        else:
            normal_ = None
        self.points = points_
        self.normal = normal_


class FunctionalContext(StrictModule):
    """Backend-neutral dynamic context for local functional evaluation."""

    time: Array
    user_args: Any

    def __init__(self, *, time: Any = 0.0, user_args: Any = None):
        time_ = jnp.asarray(time)
        if time_.shape != ():
            raise ValueError("Functional context time must be scalar.")
        if jnp.iscomplexobj(time_):
            raise TypeError("Functional context time must be real.")
        self.time = time_
        self.user_args = user_args


LocalDensity = Callable[
    [Mapping[str, LocalFieldJet], LocalGeometry, FunctionalContext],
    Array,
]


class LocalIntegralTerm(StrictModule):
    """One signed local integral over a semantic region."""

    identifier: str = eqx.field(static=True)
    region: str = eqx.field(static=True)
    fields: tuple[FieldJetSpec, ...]
    density: LocalDensity
    density_id: str = eqx.field(static=True)
    weight: float = eqx.field(static=True)
    normal: bool = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        identifier: str,
        /,
        *,
        region: str,
        fields: Sequence[FieldJetSpec],
        density: LocalDensity,
        density_id: str,
        normal: bool = False,
        weight: float = 1.0,
    ):
        identifier_ = str(identifier)
        region_ = str(region)
        density_id_ = str(density_id)
        if not identifier_:
            raise ValueError("Integral term identifier must be non-empty.")
        if not region_:
            raise ValueError("Integral term region must be non-empty.")
        if not density_id_:
            raise ValueError("Integral term density_id must be non-empty.")
        if not callable(density):
            raise TypeError("Integral term density must be callable.")
        fields_ = tuple(fields)
        if not fields_ or any(not isinstance(field, FieldJetSpec) for field in fields_):
            raise TypeError("Integral term fields must contain FieldJetSpec values.")
        names = tuple(field.field_name for field in fields_)
        if len(set(names)) != len(names):
            raise ValueError("Integral term field jets must not contain duplicates.")
        weight_ = float(weight)
        if not math.isfinite(weight_):
            raise ValueError("Integral term weight must be finite.")
        self.identifier = identifier_
        self.region = region_
        self.fields = fields_
        self.density = density
        self.density_id = density_id_
        self.normal = bool(normal)
        self.weight = weight_
        self.term_id = canonical_fingerprint(
            {
                "identifier": identifier_,
                "region": region_,
                "density_id": density_id_,
                "normal": bool(normal),
                "weight": weight_,
                "fields": [
                    {
                        "name": field.field_name,
                        "value": field.value,
                        "gradient": field.gradient,
                    }
                    for field in fields_
                ],
            }
        )


__all__ = [
    "FieldJetSpec",
    "FunctionalContext",
    "LocalDensity",
    "LocalFieldJet",
    "LocalGeometry",
    "LocalIntegralTerm",
]
