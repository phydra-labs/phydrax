#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry


InputRole: TypeAlias = Literal["control", "forcing", "parameter"]


def _shape(value: Sequence[int], owner: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _axes(
    value: Sequence[str] | None,
    rank: int,
    prefix: str,
    /,
) -> tuple[str, ...]:
    axes = (
        (
            ()
            if rank == 0
            else (prefix,)
            if rank == 1
            else tuple(f"{prefix}_{i}" for i in range(rank))
        )
        if value is None
        else tuple(str(name) for name in value)
    )
    if len(axes) != rank or any(not name for name in axes) or len(set(axes)) != rank:
        raise ValueError("Axis names must uniquely name every physical array axis.")
    return axes


def _components(
    value: Sequence[str] | None,
    count: int,
    prefix: str,
    /,
) -> tuple[str, ...]:
    names = (
        (prefix,)
        if value is None and count == 1
        else tuple(f"{prefix}{i}" for i in range(count))
        if value is None
        else tuple(str(name) for name in value)
    )
    if len(names) != count or any(not name for name in names) or len(set(names)) != count:
        raise ValueError("Component names must uniquely name every flattened component.")
    return names


def _identifier(value: str | None, payload, prefix: str, /) -> str:
    if value is not None:
        if not isinstance(value, str) or not value:
            raise ValueError("layout_id must be a non-empty string or None.")
        return value
    return f"{prefix}:{canonical_fingerprint(payload)}"


class StateLayout(StrictModule):
    """Physical state shape, labels, and explicit array-state geometry."""

    geometry: AbstractStateGeometry
    shape: tuple[int, ...] = eqx.field(static=True)
    axes: tuple[str, ...] = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        axes: Sequence[str] | None = None,
        component_names: Sequence[str] | None = None,
        geometry: AbstractStateGeometry | None = None,
        layout_id: str | None = None,
    ):
        resolved_shape = _shape(shape, "StateLayout shape")
        resolved_axes = _axes(axes, len(resolved_shape), "state")
        count = prod(resolved_shape) if resolved_shape else 1
        resolved_components = _components(component_names, count, "x")
        resolved_geometry = EuclideanStateGeometry() if geometry is None else geometry
        if not isinstance(resolved_geometry, AbstractStateGeometry):
            raise TypeError("geometry must be an AbstractStateGeometry or None.")
        self.geometry = resolved_geometry
        self.shape = resolved_shape
        self.axes = resolved_axes
        self.component_names = resolved_components
        self.size = count
        self.layout_id = _identifier(
            layout_id,
            {
                "shape": list(resolved_shape),
                "axes": list(resolved_axes),
                "components": list(resolved_components),
                "geometry": resolved_geometry.geometry_id,
            },
            "state-layout",
        )


class InputLayout(StrictModule):
    """Physical exogenous-input shape, labels, roles, and stable identity."""

    shape: tuple[int, ...] = eqx.field(static=True)
    axes: tuple[str, ...] = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    roles: tuple[InputRole, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        axes: Sequence[str] | None = None,
        component_names: Sequence[str] | None = None,
        roles: Sequence[InputRole] | InputRole = "control",
        layout_id: str | None = None,
    ):
        resolved_shape = _shape(shape, "InputLayout shape")
        resolved_axes = _axes(axes, len(resolved_shape), "input")
        count = prod(resolved_shape) if resolved_shape else 1
        resolved_components = _components(component_names, count, "u")
        raw_roles = (roles,) * count if isinstance(roles, str) else tuple(roles)
        resolved_role_values: list[InputRole] = []
        for role in raw_roles:
            if role == "control":
                resolved_role_values.append("control")
            elif role == "forcing":
                resolved_role_values.append("forcing")
            elif role == "parameter":
                resolved_role_values.append("parameter")
            else:
                raise ValueError(
                    "roles must assign 'control', 'forcing', or 'parameter' "
                    "to every component."
                )
        if len(resolved_role_values) != count:
            raise ValueError(
                "roles must assign 'control', 'forcing', or 'parameter' "
                "to every component."
            )
        resolved_roles = tuple(resolved_role_values)
        self.shape = resolved_shape
        self.axes = resolved_axes
        self.component_names = resolved_components
        self.roles = resolved_roles
        self.size = count
        self.layout_id = _identifier(
            layout_id,
            {
                "shape": list(resolved_shape),
                "axes": list(resolved_axes),
                "components": list(resolved_components),
                "roles": list(resolved_roles),
            },
            "input-layout",
        )


__all__ = ["InputLayout", "InputRole", "StateLayout"]
