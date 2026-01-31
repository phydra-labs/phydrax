#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import interpax
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..domain._components import (
    DomainComponent,
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
)
from ..domain._function import DomainFunction
from ..domain._structure import NumPoints, ProductStructure
from ._functional import FunctionalConstraint
from ._interpolate import idw_interpolant
from ._pointset import PointSetConstraint


def _default_structure(
    component: DomainComponent | DomainComponentUnion, /
) -> ProductStructure:
    labels = component.domain.labels
    return ProductStructure((labels,)).canonicalize(component.domain.labels)


def _non_fixed_labels(component: DomainComponent, /) -> tuple[str, ...]:
    fixed = {
        lbl
        for lbl in component.domain.labels
        if isinstance(component.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed))
    }
    return tuple(lbl for lbl in component.domain.labels if lbl not in fixed)


def _as_label_points(
    points: Mapping[str, ArrayLike], labels: Sequence[str]
) -> dict[str, Array]:
    out: dict[str, Array] = {}
    n = None
    for lbl in labels:
        if lbl not in points:
            raise KeyError(f"Missing points for label {lbl!r}.")
        arr = jnp.asarray(points[lbl], dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 1))
        if n is None:
            n = arr.shape[0]
        elif arr.shape[0] != n:
            raise ValueError(
                "All label point arrays must share the same leading dimension."
            )
        out[lbl] = arr
    return out


def DiscreteInteriorDataConstraint(
    constraint_var: str,
    domain,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike | None = None,
    values: ArrayLike | None = None,
    sensors: Mapping[str, ArrayLike] | ArrayLike | None = None,
    times: ArrayLike | None = None,
    sensor_values: ArrayLike | None = None,
    num_points: NumPoints | tuple[Any, ...] | None = None,
    structure: ProductStructure | None = None,
    coord_separable: Mapping[str, Any] | None = None,
    dense_structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    reduction: Literal["mean", "sum"] = "mean",
    idw_exponent: float = 2.0,
    eps_snap: float = 1e-12,
    lengthscales: Mapping[str, float] | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
    label: str | None = None,
) -> FunctionalConstraint | PointSetConstraint:
    r"""Discrete interior data-fit constraint from anchors or sensor tracks.

    This enforces agreement between a field $u$ and discrete data in one of two forms:

    1) **Anchors** (scattered measurements):
       given points $\{z_i\}_{i=1}^N$ and values $\{y_i\}$, enforce $u(z_i)\approx y_i$,
       producing a `PointSetConstraint` with residual $r_i = u(z_i) - y_i$.

    2) **Sensor tracks** (fixed sensors over time):
       given sensor locations $x_m$, times $t_j$, and measurements $y_{m,j}$, construct
       a target function $g(x,t)$ (via spatial IDW + time interpolation) and enforce
       $u(x,t)\approx g(x,t)$ at sampled $(x,t)$ pairs, producing a `FunctionalConstraint`.

    For anchor data, a continuous target function can be constructed via inverse
    distance weighting (IDW) interpolation; see `idw_interpolant`.

    **Returns:**

    - A `PointSetConstraint` (anchors), or a `FunctionalConstraint` (sensor tracks).
    """
    component = domain.component(where=where, where_all=where_all)
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "DiscreteInteriorDataConstraint requires a DomainComponent, not a union."
        )

    if sensors is not None or times is not None or sensor_values is not None:
        if sensors is None or times is None or sensor_values is None:
            raise ValueError("Provide sensors, times, and sensor_values together.")
        if num_points is None:
            raise ValueError("num_points is required for sensor-track constraints.")
        structure = _default_structure(component) if structure is None else structure
        target = _sensor_track_target(
            component,
            sensors=sensors,
            times=times,
            values=sensor_values,
            time_var="t",
            idw_exponent=idw_exponent,
            eps_snap=eps_snap,
            lengthscales=lengthscales,
        )

        def operator(u: DomainFunction, /) -> DomainFunction:
            return u - target

        return FunctionalConstraint.from_operator(
            component=component,
            operator=operator,
            constraint_vars=constraint_var,
            num_points=num_points,
            structure=structure,
            coord_separable=coord_separable,
            dense_structure=dense_structure,
            sampler=sampler,
            weight=weight,
            label=label,
            reduction="mean" if reduction == "mean" else "integral",
        )

    if points is None or values is None:
        raise ValueError(
            "Provide either (points, values) or (sensors, times, sensor_values)."
        )

    labels = _non_fixed_labels(component)
    target = idw_interpolant(
        component.domain,
        anchors=points,
        values=values,
        labels=labels,
        lengthscales=lengthscales,
        idw_exponent=idw_exponent,
        eps_snap=eps_snap,
    )

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        return functions[constraint_var] - target

    pts_in = points if labels else {}
    return PointSetConstraint.from_points(
        component=component,
        points=pts_in,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def _sensor_track_target(
    component: DomainComponent,
    /,
    *,
    sensors: Mapping[str, ArrayLike] | ArrayLike,
    times: ArrayLike,
    values: ArrayLike,
    time_var: str,
    idw_exponent: float,
    eps_snap: float,
    lengthscales: Mapping[str, float] | None,
) -> DomainFunction:
    labels = tuple(lbl for lbl in component.domain.labels if lbl != time_var)
    if time_var not in component.domain.labels:
        raise KeyError(f"Label {time_var!r} not in domain {component.domain.labels}.")

    sensor_map: dict[str, ArrayLike]
    if isinstance(sensors, Mapping):
        sensor_map = {lbl: sensors[lbl] for lbl in labels}  # ty: ignore
    else:
        if "x" not in component.domain.labels:
            raise ValueError("sensors array requires a domain label 'x'.")
        sensor_map = {"x": sensors}
    anchors = _as_label_points(sensor_map, labels)

    t = jnp.asarray(times, dtype=float).reshape((-1,))
    y = jnp.asarray(values, dtype=float)
    if y.ndim == 1:
        y = y[None, :, None]
    elif y.ndim == 2:
        y = y[..., None]
    if y.shape[1] != t.shape[0]:
        raise ValueError("sensor_values must have shape (M, T, ...) matching times.")

    vals_t = jnp.moveaxis(y, 1, 0)
    single_time = int(t.shape[0]) < 2
    if single_time:
        vals0 = vals_t[0]
        spline = None
    else:
        slopes = interpax.approx_df(t, vals_t, axis=0)
        spline = interpax.CubicHermiteSpline(t, vals_t, slopes, axis=0, check=False)

    eps = float(eps_snap)
    exponent = float(idw_exponent)
    ls_map = lengthscales or {}

    def _weights(x_by_label: Mapping[str, Array], /) -> Array:
        d2_terms: list[Array] = []
        for lbl in labels:
            a = anchors[lbl]
            ls = float(ls_map.get(lbl, 1.0))
            xq = jnp.asarray(x_by_label[lbl], dtype=float)
            if a.ndim == 2:
                if xq.ndim == 0:
                    xq = xq.reshape((1, 1))
                elif xq.ndim == 1:
                    xq = xq.reshape((1, -1))
                diff = (a[:, None, :] - xq[None, :, :]) / ls
                d2_lbl = jnp.sum(diff * diff, axis=-1)
            else:
                xq = xq.reshape((-1,))
                diff = (a[:, None] - xq[None, :]) / ls
                d2_lbl = diff * diff

            d2_terms.append(d2_lbl)

        if not d2_terms:
            raise ValueError("sensor tracks require at least one spatial label.")

        d2 = d2_terms[0]
        for term in d2_terms[1:]:
            d2 = d2 + term

        min_idx = jnp.argmin(d2, axis=0)
        min_val = jnp.min(d2, axis=0)
        if eps > 0.0:
            is_snap = min_val < eps
        else:
            is_snap = min_val <= 0.0

        w = (1.0 / (d2 + eps) ** exponent).astype(float)
        w = w / jnp.sum(w, axis=0, keepdims=True)

        snap_w = jax.nn.one_hot(min_idx, w.shape[0], dtype=w.dtype)
        if snap_w.ndim == 2:
            snap_w = jnp.swapaxes(snap_w, 0, 1)
        return jnp.where(is_snap[None, :], snap_w, w)

    def _target(*args, key=None, **kwargs):
        del key, kwargs
        x_by_label = {
            lbl: jnp.asarray(args[i], dtype=float) for i, lbl in enumerate(labels)
        }
        w = _weights(x_by_label)
        t_eval = jnp.asarray(args[len(labels)], dtype=float)
        if single_time:
            y_eval = vals0
        else:
            assert spline is not None
            y_eval = spline(t_eval)
        return jnp.tensordot(w, y_eval, axes=([0], [0]))

    deps = labels + (time_var,)
    return DomainFunction(domain=component.domain, deps=deps, func=_target, metadata={})
