#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import interpax
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ..constraints._enforced import _enforced_constraint_weight_fn, enforce_initial
from ..domain._base import _AbstractGeometry
from ..domain._components import (
    Boundary,
    DomainComponent,
    Fixed,
    FixedEnd,
    FixedStart,
)
from ..domain._domain import _AbstractDomain, RelabeledDomain
from ..domain._function import DomainFunction
from ..domain._scalar import _AbstractScalarDomain


def _unwrap_factor(factor: object, /) -> object:
    if isinstance(factor, RelabeledDomain):
        return factor.base
    return factor


def _geometry_boundary_labels(component: DomainComponent, /) -> tuple[str, ...]:
    out: list[str] = []
    for lbl in component.domain.labels:
        if not isinstance(component.spec.component_for(lbl), Boundary):
            continue
        factor = _unwrap_factor(component.domain.factor(lbl))
        if isinstance(factor, _AbstractGeometry):
            out.append(lbl)
    return tuple(out)


def _initial_label(component: DomainComponent, evolution_var: str, /) -> str | None:
    if evolution_var not in component.domain.labels:
        return None
    comp = component.spec.component_for(evolution_var)
    if isinstance(comp, (FixedStart, FixedEnd, Fixed, Boundary)):
        return evolution_var
    return None


def _constraint_stage(
    component: DomainComponent, evolution_var: str, /
) -> Literal["boundary", "initial", "interior"]:
    if _geometry_boundary_labels(component):
        return "boundary"
    if _initial_label(component, evolution_var) is not None:
        return "initial"
    return "interior"


class SingleFieldEnforcedConstraint(StrictModule):
    r"""Enforced constraint term acting on a single field.

    An *enforced constraint* is a transformation that builds an ansatz
    $\tilde u$ from a base field $u$ such that some condition is satisfied
    **by construction**. Conceptually this is a map

    $$
    \mathcal{H}: u \mapsto \tilde u
    $$

    where $\tilde u$ is intended to satisfy, e.g.,
    Dirichlet boundary conditions $\tilde u|_{\partial\Omega}=g$, initial
    conditions $\tilde u(\cdot,t_0)=u_0$, or similar constraints.

    `SingleFieldEnforcedConstraint` stores the *metadata* needed to stage and compose
    these transformations inside `EnforcedConstraintPipeline`.
    """

    field: str
    component: DomainComponent
    apply: Callable[[DomainFunction], DomainFunction]
    max_derivative_order: int
    time_derivative_order: int
    initial_target: DomainFunction | ArrayLike | None

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        apply: Callable[[DomainFunction], DomainFunction],
        /,
        *,
        max_derivative_order: int = 0,
        time_derivative_order: int = 0,
        initial_target: DomainFunction | ArrayLike | None = None,
    ):
        r"""Create a single-field enforced constraint.

        **Arguments:**

        - `field`: Name of the field being modified (e.g. `"u"`).
        - `component`: The domain subset where the constraint is defined (boundary,
          initial, etc.).
        - `apply`: The enforcement map $\mathcal{H}$ implemented as a callable that
          returns a new `DomainFunction`.

        **Keyword arguments:**

        - `max_derivative_order`: Maximum spatial derivative order that the
          constraint construction expects to be well-defined (used for staging
          and gating in composite pipelines).
        - `time_derivative_order`: When the constraint represents an *initial*
          derivative target, this is the derivative order $k$ for
          $\partial_t^k u(\cdot,t_0)$.
        - `initial_target`: Optional target for the initial derivative order:
          provide $g_k$ for $\partial_t^k u(\cdot,t_0)=g_k$.

        Notes:

        - If `initial_target` is provided, `EnforcedConstraintPipeline` will group
          targets for orders $k=0,\dots,K$ and build a single initial overlay.
        """
        self.field = str(field)
        self.component = component
        self.apply = apply
        self.max_derivative_order = int(max_derivative_order)
        self.time_derivative_order = int(time_derivative_order)
        self.initial_target = initial_target

    @property
    def co_vars(self) -> tuple[str, ...]:
        """Names of co-dependent fields (always empty for single-field terms)."""
        return ()


class MultiFieldEnforcedConstraint(StrictModule):
    r"""Enforced constraint term that depends on other fields.

    Some enforced constraint constructions require access to *co-variables* (other
    fields) when enforcing a given field. For example, a boundary condition for
    a stress-like quantity might depend on both displacement and material
    parameters.

    This stores an `apply` function of the form

    $$
    \tilde u = \mathcal{H}\big(u,\ \{v\}_{v\in\texttt{co\_vars}}\big),
    $$

    where `get_field(name)` supplies the current (possibly already enforced)
    `DomainFunction` for each co-variable.
    """

    field: str
    component: DomainComponent
    co_vars: tuple[str, ...]
    apply: Callable[[DomainFunction, Callable[[str], DomainFunction]], DomainFunction]
    max_derivative_order: int
    time_derivative_order: int

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        co_vars: Sequence[str],
        apply: Callable[
            [DomainFunction, Callable[[str], DomainFunction]], DomainFunction
        ],
        /,
        *,
        max_derivative_order: int = 0,
        time_derivative_order: int = 0,
    ):
        r"""Create a multi-field enforced constraint.

        **Arguments:**

        - `field`: Name of the field being modified.
        - `component`: The domain subset where the constraint is defined.
        - `co_vars`: Names of fields that `apply` depends on.
        - `apply`: An enforcement map $\mathcal{H}$ implemented as
          `apply(u, get_field) -> DomainFunction`.

        **Keyword arguments:**

        - `max_derivative_order`: Maximum spatial derivative order expected by the
          constraint.
        - `time_derivative_order`: Optional initial-derivative order metadata.
        """
        self.field = str(field)
        self.component = component
        self.co_vars = tuple(str(v) for v in co_vars)
        self.apply = apply
        self.max_derivative_order = int(max_derivative_order)
        self.time_derivative_order = int(time_derivative_order)


class EnforcedInteriorData(StrictModule):
    r"""Interior data source for enforced data overlays.

    `EnforcedInteriorData` represents *measurements* (or anchor constraints) that you
    want to enforce by construction in the interior of a domain, without
    competing with boundary/initial enforced constraints.

    Two input modes are supported:

    1) **Anchor points**: a set of points $z_j$ with values $y_j$.

       - `points`: mapping from domain label to coordinate arrays.
       - `values`: an array of values with leading dimension $N$.

    2) **Sensor tracks**: a set of fixed sensors $x_m$ observed over times $t_n$.

       - `sensors`: array with shape $(M,d)$ (or $(d,)$ for a single sensor).
       - `times`: array with shape $(N,)$.
       - `sensor_values`: array with shape $(M,N)$ or $(M,N,C)$.

    The enforced overlay built from this data uses inverse-distance weights (IDW)
    to compute a correction term $\Delta u(z)$ from residuals
    $r_j = y_j - u(z_j)$, while multiplying by a gate $M(z)$ that vanishes on
    constrained sets (boundary / initial time) so the overlay does not destroy
    those conditions.

    See `EnforcedConstraintPipeline` for how this integrates with other enforced stages.
    """

    field: str
    points: frozendict[str, Array] | None
    values: Array | None

    sensors: Array | None
    times: Array | None
    sensor_values: Array | None

    idw_exponent: float
    eps_snap: float
    lengthscales: frozendict[str, float]

    use_envelope: bool
    envelope_scale: float

    space_label: str
    time_label: str
    time_interp: str

    def __init__(
        self,
        field: str,
        /,
        *,
        points: Mapping[str, ArrayLike] | None = None,
        values: ArrayLike | None = None,
        sensors: ArrayLike | None = None,
        times: ArrayLike | None = None,
        sensor_values: ArrayLike | None = None,
        idw_exponent: float = 2.0,
        eps_snap: float = 1e-12,
        lengthscales: Mapping[str, float] | None = None,
        use_envelope: bool = False,
        envelope_scale: float = 1.0,
        space_label: str = "x",
        time_label: str = "t",
        time_interp: Literal["idw", "hermite"] = "idw",
    ):
        r"""Create an enforced interior data source.

        **Arguments:**

        - `field`: Name of the field that this data applies to.

        **Anchor mode:**

        - `points`: Mapping `{label: coords}` giving anchor coordinates per domain
          label. Geometry labels should use shape `(N,d)` and scalar labels shape
          `(N,)`.
        - `values`: Anchor values with shape `(N,)` or `(N,C)`.

        **Sensor-track mode:**

        - `sensors`: Sensor locations, shape `(M,d)` (or `(d,)`).
        - `times`: Observation times, shape `(N,)`.
        - `sensor_values`: Observations, shape `(M,N)` or `(M,N,C)`.
        - `space_label`: Domain label corresponding to space.
        - `time_label`: Domain label corresponding to time.
        - `time_interp`:
            - `"idw"` flattens the $(x_m,t_n)$ grid into anchors in $(x,t)$ and uses
              IDW in the full domain.
            - `"hermite"` uses a cubic Hermite spline in time and IDW only in space
              (requires a 2-factor domain with `(space_label, time_label)`).

        **IDW details:**

        - `idw_exponent`: Power $p$ in weights
          $w_j(z)\propto (\|z-z_j\|^2+\varepsilon)^{-p/2}$.
        - `lengthscales`: Optional per-label lengthscales $\ell_\alpha$ used inside
          the distance metric:
          $\|z-z_j\|^2=\sum_\alpha \|(z_\alpha-z_{j,\alpha})/\ell_\alpha\|^2$.
        - `eps_snap`: Snap threshold: when $z$ is closer than `eps_snap` to an
          anchor, the overlay uses a one-hot weight so that $u(z)$ matches the
          anchor exactly.

        **Envelope (optional):**

        - `use_envelope`: If enabled, multiplies IDW weights by a source-local envelope.
        - `envelope_scale`: Envelope scale $s$ in $\psi(z)=\exp(-d(z)^2/s^2)$.
        """
        self.field = str(field)

        if points is None:
            self.points = None
        else:
            self.points = frozendict(
                {k: jnp.asarray(v, dtype=float) for k, v in points.items()}
            )

        if values is None:
            self.values = None
        else:
            self.values = jnp.asarray(values, dtype=float)

        self.sensors = None if sensors is None else jnp.asarray(sensors, dtype=float)
        self.times = (
            None if times is None else jnp.asarray(times, dtype=float).reshape((-1,))
        )
        self.sensor_values = (
            None if sensor_values is None else jnp.asarray(sensor_values, dtype=float)
        )

        self.idw_exponent = float(idw_exponent)
        self.eps_snap = float(eps_snap)
        self.lengthscales = frozendict(
            {}
            if lengthscales is None
            else {str(k): float(v) for k, v in lengthscales.items()}
        )

        self.use_envelope = bool(use_envelope)
        self.envelope_scale = float(envelope_scale)

        self.space_label = str(space_label)
        self.time_label = str(time_label)
        self.time_interp = str(time_interp)
        if self.time_interp not in ("idw", "hermite"):
            raise ValueError("time_interp must be 'idw' or 'hermite'.")

        if self.points is None:
            if self.sensors is None or self.times is None or self.sensor_values is None:
                raise ValueError(
                    "EnforcedInteriorData requires either (points, values) or (sensors, times, sensor_values)."
                )
        else:
            if self.values is None:
                raise ValueError("EnforcedInteriorData(points=...) requires values=...")

    def _as_anchors(
        self,
        *,
        labels: tuple[str, ...],
    ) -> tuple[frozendict[str, Array], Array]:
        if self.points is not None:
            missing = [lbl for lbl in labels if lbl not in self.points]
            if missing:
                raise KeyError(f"Interior anchors missing labels {tuple(missing)!r}.")
            anchors = {lbl: jnp.asarray(self.points[lbl], dtype=float) for lbl in labels}
            y = jnp.asarray(self.values, dtype=float)
            return frozendict(anchors), y

        if self.time_interp == "hermite":
            raise ValueError("Hermite sensor tracks cannot be flattened into anchors.")
        assert (
            self.sensors is not None
            and self.times is not None
            and self.sensor_values is not None
        )
        if self.space_label not in labels or self.time_label not in labels:
            raise KeyError(
                f"Sensor-track anchors require labels ({self.space_label!r}, {self.time_label!r}) in {labels!r}."
            )
        if len(labels) != 2:
            raise ValueError(
                "Sensor-track anchors require domain labels exactly (space_label, time_label)."
            )

        sensors = jnp.asarray(self.sensors, dtype=float)
        if sensors.ndim == 1:
            sensors = sensors.reshape((1, -1))
        times = jnp.asarray(self.times, dtype=float).reshape((-1,))
        values = jnp.asarray(self.sensor_values, dtype=float)
        if values.ndim == 1:
            values = values.reshape((1, -1, 1))
        elif values.ndim == 2:
            values = values.reshape((values.shape[0], values.shape[1], 1))

        m = int(sensors.shape[0])
        n = int(times.shape[0])
        if values.shape[0] != m or values.shape[1] != n:
            raise ValueError(
                "sensor_values must have shape (M, N) or (M, N, C) matching sensors/times."
            )

        xs = jnp.broadcast_to(sensors[:, None, :], (m, n, sensors.shape[1])).reshape(
            (-1, sensors.shape[1])
        )
        ts = jnp.broadcast_to(times[None, :], (m, n)).reshape((-1,))
        ys = values.reshape((m * n, -1))
        if ys.shape[1] == 1:
            ys = ys.reshape((-1,))

        return frozendict({self.space_label: xs, self.time_label: ts}), ys

    def _as_track(self) -> tuple[Array, Array, Array]:
        assert (
            self.sensors is not None
            and self.times is not None
            and self.sensor_values is not None
        )
        sensors = jnp.asarray(self.sensors, dtype=float)
        if sensors.ndim == 1:
            sensors = sensors.reshape((1, -1))
        times = jnp.asarray(self.times, dtype=float).reshape((-1,))
        values = jnp.asarray(self.sensor_values, dtype=float)
        if values.ndim == 1:
            values = values.reshape((1, -1, 1))
        elif values.ndim == 2:
            values = values.reshape((values.shape[0], values.shape[1], 1))

        m = int(sensors.shape[0])
        n = int(times.shape[0])
        if values.shape[0] != m or values.shape[1] != n:
            raise ValueError(
                "sensor_values must have shape (M, N) or (M, N, C) matching sensors/times."
            )

        return sensors, times, values


@dataclass(frozen=True)
class _TrackSource:
    source_index: int
    space_label: str
    time_label: str
    sensors: Array  # (M, d)
    times: Array  # (N,)
    values: Array  # (M, N, C)
    idw_exponent: float
    eps_snap: float
    lengthscales: frozendict[str, float]
    use_envelope: bool
    envelope_scale: float


@dataclass(frozen=True)
class _TrackOverlaySource:
    source_index: int
    space_label: str
    time_label: str
    sensors: Array  # (M, d)
    times: Array  # (N,)
    values_scaled_t: Array  # (N, M, C)
    slopes_values_t: Array  # (N, M, C)
    m_anchor: Array  # (M, N)
    idw_exponent: float
    eps_snap: float
    lengthscale: float
    use_envelope: bool
    envelope_scale: float


@dataclass(frozen=True)
class _UnifiedAnchorSet:
    labels: tuple[str, ...]
    anchors: frozendict[str, Array]  # label -> (N, ...) arrays
    values: Array  # (N,) or (N,C)
    source_index: Array  # (N,) int32
    idw_exponent: Array  # (N,) float
    eps_snap: Array  # (N,) float
    lengthscales: frozendict[str, Array]  # label -> (N,) float
    envelope_enabled: tuple[bool, ...]
    envelope_scale: Array  # (S,) float
    track_sources: tuple[_TrackSource, ...]


def _normalize_anchor_values(y: Array, /) -> Array:
    y = jnp.asarray(y, dtype=float)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        if y.shape[1] == 1:
            return y.reshape((-1,))
        return y
    raise ValueError(f"Anchor values must be scalar or rank-2, got shape {y.shape}.")


def _as_anchor_array(domain: _AbstractDomain, label: str, x: Array, /) -> Array:
    factor = _unwrap_factor(domain.factor(label))
    arr = jnp.asarray(x, dtype=float)

    if isinstance(factor, _AbstractGeometry):
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        if arr.ndim != 2:
            raise ValueError(
                f"Geometry anchor {label!r} must have shape (N,d), got {arr.shape}."
            )
        d = int(factor.var_dim)
        if arr.shape[1] != d:
            raise ValueError(
                f"Geometry anchor {label!r} must have d={d}, got {arr.shape[1]}."
            )
        return arr

    if isinstance(factor, _AbstractScalarDomain):
        if arr.ndim == 0:
            return arr.reshape((1,))
        if arr.ndim == 1:
            return arr.reshape((-1,))
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.reshape((-1,))
        raise ValueError(
            f"Scalar anchor {label!r} must have shape (N,), got {arr.shape}."
        )

    raise TypeError(
        f"Unsupported anchor domain factor {type(factor).__name__} for label {label!r}."
    )


def _build_anchor_set(
    sources: Sequence[EnforcedInteriorData],
    *,
    domain: _AbstractDomain,
) -> _UnifiedAnchorSet:
    if not sources:
        raise ValueError("Must provide at least one EnforcedInteriorData source.")

    labels = tuple(domain.labels)
    static_sources: list[
        tuple[int, frozendict[str, Array], Array, float, float, dict[str, float]]
    ] = []
    track_sources: list[_TrackSource] = []
    envelope_enabled: list[bool] = []
    envelope_scale: list[float] = []

    for si, src in enumerate(sources):
        envelope_enabled.append(bool(src.use_envelope))
        envelope_scale.append(float(src.envelope_scale))

        if src.points is not None or src.time_interp == "idw":
            a, y = src._as_anchors(labels=labels)
            anchors = frozendict(
                {lbl: _as_anchor_array(domain, lbl, a[lbl]) for lbl in labels}
            )
            values = _normalize_anchor_values(y)
            static_sources.append(
                (
                    int(si),
                    anchors,
                    values,
                    float(src.idw_exponent),
                    float(src.eps_snap),
                    dict(src.lengthscales),
                )
            )
            continue

        sensors, times, values = src._as_track()
        if src.space_label not in labels or src.time_label not in labels:
            raise KeyError(
                f"Sensor-track anchors require labels ({src.space_label!r}, {src.time_label!r}) in {labels!r}."
            )
        if len(labels) != 2:
            raise ValueError(
                "Sensor-track anchors require domain labels exactly (space_label, time_label)."
            )

        track_sources.append(
            _TrackSource(
                source_index=int(si),
                space_label=src.space_label,
                time_label=src.time_label,
                sensors=sensors,
                times=times,
                values=values,
                idw_exponent=float(src.idw_exponent),
                eps_snap=float(src.eps_snap),
                lengthscales=frozendict(
                    {str(k): float(v) for k, v in src.lengthscales.items()}
                ),
                use_envelope=bool(src.use_envelope),
                envelope_scale=float(src.envelope_scale),
            )
        )

    anchors: dict[str, list[Array]] = {lbl: [] for lbl in labels}
    values_list: list[Array] = []
    src_index_list: list[Array] = []
    idw_list: list[Array] = []
    eps_list: list[Array] = []
    ls_list: dict[str, list[Array]] = {lbl: [] for lbl in labels}

    for si, a, y, idw_exp_src, eps_src, ls_src in static_sources:
        n = int(next(iter(a.values())).shape[0])
        if y.shape[0] != n:
            raise ValueError(
                f"Interior data values must have leading dim N={n}, got {y.shape[0]}."
            )
        for lbl in labels:
            anchors[lbl].append(a[lbl])
            ls_val = float(ls_src.get(lbl, 1.0))
            ls_list[lbl].append(jnp.full((n,), ls_val, dtype=float))
        values_list.append(y)
        src_index_list.append(jnp.full((n,), int(si), dtype=jnp.int32))
        idw_list.append(jnp.full((n,), float(idw_exp_src), dtype=float))
        eps_list.append(jnp.full((n,), float(eps_src), dtype=float))

    if values_list:
        anchors_cat = {lbl: jnp.concatenate(anchors[lbl], axis=0) for lbl in labels}
        values_cat = jnp.concatenate(values_list, axis=0)
        source_index = jnp.concatenate(src_index_list, axis=0)
        idw_exp = jnp.concatenate(idw_list, axis=0)
        eps_snap = jnp.concatenate(eps_list, axis=0)
        lengthscales = {lbl: jnp.concatenate(ls_list[lbl], axis=0) for lbl in labels}
    else:
        anchors_cat = {}
        lengthscales = {}
        for lbl in labels:
            factor = _unwrap_factor(domain.factor(lbl))
            if isinstance(factor, _AbstractGeometry):
                anchors_cat[lbl] = jnp.zeros((0, int(factor.var_dim)), dtype=float)
            elif isinstance(factor, _AbstractScalarDomain):
                anchors_cat[lbl] = jnp.zeros((0,), dtype=float)
            else:
                raise TypeError(
                    f"Unsupported anchor domain factor {type(factor).__name__} for label {lbl!r}."
                )
            lengthscales[lbl] = jnp.zeros((0,), dtype=float)
        values_cat = jnp.zeros((0,), dtype=float)
        source_index = jnp.zeros((0,), dtype=jnp.int32)
        idw_exp = jnp.zeros((0,), dtype=float)
        eps_snap = jnp.zeros((0,), dtype=float)

    # Dedupe coincident anchors and raise errors on conflicts.
    # Coincidence uses the same directional snap metric as the runtime overlay:
    # anchor i considers j coincident if d2_i(z_j) < eps_snap_i (and vice-versa).
    import numpy as np

    n_total = int(eps_snap.shape[0])
    if n_total > 0:
        keep = np.ones((n_total,), dtype=bool)

        anchors_np = {lbl: np.asarray(anchors_cat[lbl]) for lbl in labels}
        values_np = np.asarray(values_cat)
        eps_np = np.asarray(eps_snap).reshape((-1,))
        ls_np = {lbl: np.asarray(lengthscales[lbl]).reshape((-1,)) for lbl in labels}

        def _d2(i: int, j: int) -> float:
            out = 0.0
            for lbl in labels:
                ai = anchors_np[lbl][i]
                aj = anchors_np[lbl][j]
                s = float(ls_np[lbl][i])
                if np.ndim(ai) == 1:
                    diff = (aj - ai) / s
                    out += float(np.sum(diff * diff))
                else:
                    diff = (float(aj) - float(ai)) / s
                    out += float(diff * diff)
            return out

        for i in range(n_total):
            if not keep[i]:
                continue
            for j in range(i + 1, n_total):
                if not keep[j]:
                    continue
                if (_d2(i, j) < eps_np[i]) or (_d2(j, i) < eps_np[j]):
                    if values_np.ndim == 1:
                        same = np.allclose(
                            values_np[i], values_np[j], rtol=0.0, atol=1e-12
                        )
                    else:
                        same = np.allclose(
                            values_np[i, :], values_np[j, :], rtol=0.0, atol=1e-12
                        )
                    if not same:
                        raise ValueError(
                            "Conflicting coincident interior anchors detected."
                        )
                    keep[j] = False

        if not np.all(keep):
            idx_keep = np.nonzero(keep)[0]
            anchors_cat = {
                lbl: jnp.asarray(anchors_np[lbl][idx_keep], dtype=float) for lbl in labels
            }
            values_cat = jnp.asarray(values_np[idx_keep], dtype=float)
            source_index = jnp.asarray(
                np.asarray(source_index)[idx_keep], dtype=jnp.int32
            )
            idw_exp = jnp.asarray(np.asarray(idw_exp)[idx_keep], dtype=float)
            eps_snap = jnp.asarray(eps_np[idx_keep], dtype=float)
            lengthscales = {
                lbl: jnp.asarray(ls_np[lbl][idx_keep], dtype=float) for lbl in labels
            }

    return _UnifiedAnchorSet(
        labels=labels,
        anchors=frozendict(anchors_cat),
        values=values_cat,
        source_index=source_index,
        idw_exponent=idw_exp,
        eps_snap=eps_snap,
        lengthscales=frozendict(lengthscales),
        envelope_enabled=tuple(envelope_enabled),
        envelope_scale=jnp.asarray(envelope_scale, dtype=float),
        track_sources=tuple(track_sources),
    )


def _gate_exponents_from_boundary_constraints(
    constraints: Sequence[SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint],
    /,
    *,
    evolution_var: str,
) -> dict[str, int]:
    exps: dict[str, int] = {}
    for c in constraints:
        if _constraint_stage(c.component, evolution_var) != "boundary":
            continue
        labels = _geometry_boundary_labels(c.component)
        if len(labels) != 1:
            raise ValueError(
                "Boundary enforced constraints must specify exactly one geometry Boundary() label."
            )
        lbl = labels[0]
        exps[lbl] = max(exps.get(lbl, 0), int(c.max_derivative_order) + 1)
    return exps


def _max_initial_order(
    constraints: Sequence[SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint],
    /,
    *,
    evolution_var: str,
) -> int:
    orders: list[int] = []
    for c in constraints:
        if _constraint_stage(c.component, evolution_var) != "initial":
            continue
        orders.append(int(c.time_derivative_order))
    return max(orders) if orders else -1


class _BoundaryBlendOverlay(StrictModule):
    var: str
    pieces: tuple[SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint, ...]
    include_identity_remainder: bool
    weights: tuple[DomainFunction, ...]
    remainder_weight: DomainFunction | None

    def __init__(
        self,
        u_base: DomainFunction,
        pieces: Sequence[SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint],
        /,
        *,
        var: str,
        include_identity_remainder: bool,
        num_reference: int,
        sampler: str,
        key: Key[Array, ""],
    ):
        if not pieces:
            raise ValueError("_BoundaryBlendOverlay requires at least one piece.")
        self.var = str(var)
        self.pieces = tuple(pieces)
        self.include_identity_remainder = bool(include_identity_remainder)

        if self.var not in u_base.domain.labels:
            raise KeyError(
                f"Label {self.var!r} not in base domain {u_base.domain.labels}."
            )

        base_factor = _unwrap_factor(u_base.domain.factor(self.var))
        if not isinstance(base_factor, _AbstractGeometry):
            raise TypeError("_BoundaryBlendOverlay requires a geometry label.")
        geom = base_factor

        for c in self.pieces:
            if self.var not in c.component.domain.labels:
                raise KeyError(
                    f"Label {self.var!r} not in piece domain {c.component.domain.labels}."
                )
            factor = _unwrap_factor(c.component.domain.factor(self.var))
            if not isinstance(factor, _AbstractGeometry):
                raise TypeError("Boundary pieces must use a geometry label for var.")
            if not geom.equivalent(factor):
                raise ValueError(
                    "Boundary blend requires all pieces to share an equivalent geometry."
                )
            comp = c.component.spec.component_for(self.var)
            if not isinstance(comp, Boundary):
                raise ValueError(
                    "Boundary blend pieces require component Boundary() for var."
                )

        weights: list[DomainFunction] = []
        wheres: list[Callable | None] = []
        remainder_weight: DomainFunction | None = None

        num_terms = len(self.pieces) + (1 if include_identity_remainder else 0)
        keys = jr.split(key, num_terms)
        key_iter = iter(keys)

        for c in self.pieces:
            where_fn = c.component.where.get(self.var)
            wheres.append(where_fn)
            w_fn = _enforced_constraint_weight_fn(
                geom,
                where_fn,
                num_reference=int(num_reference),
                sampler=str(sampler),
                key=next(key_iter),
                on_empty="error",
            )
            weights.append(
                DomainFunction(
                    domain=u_base.domain, deps=(self.var,), func=w_fn, metadata={}
                )
            )

        if include_identity_remainder:
            rem_where = _complement_where(wheres)
            if rem_where is not None:
                w_rem_fn = _enforced_constraint_weight_fn(
                    geom,
                    rem_where,
                    num_reference=int(num_reference),
                    sampler=str(sampler),
                    key=next(key_iter),
                    on_empty="zero",
                )
                remainder_weight = DomainFunction(
                    domain=u_base.domain, deps=(self.var,), func=w_rem_fn, metadata={}
                )

        self.weights = tuple(weights)
        self.remainder_weight = remainder_weight

    def apply(
        self, u: DomainFunction, /, *, get_field: Callable[[str], DomainFunction]
    ) -> DomainFunction:
        num = DomainFunction(domain=u.domain, deps=(), func=0.0, metadata=u.metadata)
        den = DomainFunction(domain=u.domain, deps=(), func=0.0, metadata={})

        for c, w in zip(self.pieces, self.weights, strict=True):
            if isinstance(c, SingleFieldEnforcedConstraint):
                u_piece = c.apply(u)
            else:
                u_piece = c.apply(u, get_field)
            num = num + w * u_piece
            den = den + w

        if self.remainder_weight is not None:
            num = num + self.remainder_weight * u
            den = den + self.remainder_weight

        return num / den


class _InitialEnforcedOverlay(StrictModule):
    component: DomainComponent
    var: str
    targets: frozendict[int, DomainFunction | ArrayLike]

    def __init__(
        self,
        component: DomainComponent,
        /,
        *,
        var: str,
        targets: Mapping[int, DomainFunction | ArrayLike],
    ):
        self.component = component
        self.var = str(var)
        self.targets = frozendict({int(k): v for k, v in targets.items()})

    def apply(self, u: DomainFunction, /) -> DomainFunction:
        return enforce_initial(u, self.component, var=self.var, targets=self.targets)


def _complement_where(wheres: Sequence[Callable | None], /) -> Callable | None:
    if any(w is None for w in wheres):
        return None
    if not wheres:
        return lambda x: jnp.asarray(True)

    def _union(x):
        fn = wheres[0]
        assert callable(fn)
        out = fn(x)
        for fn in wheres[1:]:
            assert callable(fn)
            out = jnp.logical_or(out, fn(x))
        return out

    def _comp(x):
        return jnp.logical_not(_union(x))

    return _comp


def _idw_weights(
    d2: Array,
    *,
    idw_exponent: Array,
    eps: float,
) -> Array:
    p = jnp.asarray(idw_exponent, dtype=float)
    d2 = jnp.asarray(d2, dtype=float)

    w = jnp.where(p == 2.0, 1.0 / (d2 + eps), 1.0 / ((d2 + eps) ** (p / 2.0)))
    wsum = jnp.sum(w) + eps
    return w / wsum


class _InteriorDataOverlay(StrictModule):
    anchor_set: _UnifiedAnchorSet
    gate_exponents: frozendict[str, int]
    evolution_var: str
    max_init_order: int
    t0: Array | None
    geometry_factors: frozendict[str, _AbstractGeometry]
    m_anchor: Array
    track_sources: tuple[_TrackOverlaySource, ...]

    def __init__(
        self,
        domain: _AbstractDomain,
        anchor_set: _UnifiedAnchorSet,
        /,
        *,
        gate_exponents: Mapping[str, int],
        evolution_var: str,
        max_init_order: int,
    ):
        self.anchor_set = anchor_set
        self.gate_exponents = frozendict(
            {str(k): int(v) for k, v in gate_exponents.items()}
        )
        self.evolution_var = str(evolution_var)
        self.max_init_order = int(max_init_order)

        t0: Array | None = None
        if self.evolution_var in domain.labels:
            factor = _unwrap_factor(domain.factor(self.evolution_var))
            if isinstance(factor, _AbstractScalarDomain):
                t0 = jnp.asarray(factor.fixed("start"), dtype=float).reshape(())
        self.t0 = t0

        geom: dict[str, _AbstractGeometry] = {}
        for lbl in self.gate_exponents:
            factor = _unwrap_factor(domain.factor(lbl))
            if not isinstance(factor, _AbstractGeometry):
                raise TypeError(
                    "Boundary gating exponents must refer to geometry labels."
                )
            geom[lbl] = factor
        self.geometry_factors = frozendict(geom)

        # Precompute M(anchor_i) to validate anchors and reuse inside the overlay.
        n = int(self.anchor_set.source_index.shape[0])
        m_anchor = jnp.ones((n,), dtype=float)
        for lbl, p in self.gate_exponents.items():
            x = jnp.asarray(self.anchor_set.anchors[lbl], dtype=float)
            sdf = jax.vmap(self.geometry_factors[lbl].adf)(x)
            m_anchor = m_anchor * (jnp.abs(sdf) ** int(p))

        q = (self.max_init_order + 1) if self.max_init_order >= 0 else 0
        if q > 0:
            if self.t0 is None:
                raise ValueError("Missing scalar time domain for evolution_var.")
            if self.evolution_var not in self.anchor_set.anchors:
                raise KeyError(
                    f"Missing evolution_var {self.evolution_var!r} in interior anchors."
                )
            t = jnp.asarray(
                self.anchor_set.anchors[self.evolution_var], dtype=float
            ).reshape((-1,))
            m_anchor = m_anchor * (jnp.maximum(t - self.t0, 0.0) ** int(q))

        if int(jnp.sum(m_anchor <= 0)) != 0:
            raise ValueError("Interior anchors include points where M(z_i)=0.")
        self.m_anchor = m_anchor

        track_sources: list[_TrackOverlaySource] = []
        for src in self.anchor_set.track_sources:
            if src.time_label != self.evolution_var:
                raise ValueError(
                    "Hermite sensor tracks require time_label to match evolution_var."
                )
            if src.space_label not in self.gate_exponents and self.gate_exponents:
                raise ValueError(
                    "Hermite sensor tracks must use the same space label as boundary gating."
                )

            sensors = jnp.asarray(src.sensors, dtype=float)
            times = jnp.asarray(src.times, dtype=float).reshape((-1,))
            values = jnp.asarray(src.values, dtype=float)
            m_count = int(sensors.shape[0])
            n_count = int(times.shape[0])
            if values.shape[0] != m_count or values.shape[1] != n_count:
                raise ValueError(
                    "sensor_values must have shape (M, N, C) matching sensors/times."
                )

            m_track = jnp.ones((m_count, n_count), dtype=float)
            for lbl, p in self.gate_exponents.items():
                if lbl != src.space_label:
                    raise ValueError(
                        "Hermite sensor tracks require boundary gating on the space label only."
                    )
                sdf = jax.vmap(self.geometry_factors[lbl].adf)(sensors)
                m_track = m_track * (jnp.abs(sdf) ** int(p))[:, None]

            if q > 0:
                if self.t0 is None:
                    raise ValueError("Missing scalar time domain for evolution_var.")
                m_track = m_track * (jnp.maximum(times - self.t0, 0.0) ** int(q))[None, :]

            if int(jnp.sum(m_track <= 0)) != 0:
                raise ValueError("Interior anchors include points where M(z_i)=0.")

            values_scaled = values / m_track[..., None]
            values_scaled_t = jnp.moveaxis(values_scaled, 1, 0)
            if n_count < 2:
                slopes_values_t = jnp.zeros_like(values_scaled_t)
            else:
                slopes_values_t = interpax.approx_df(times, values_scaled_t, axis=0)

            lengthscale = float(src.lengthscales.get(src.space_label, 1.0))
            track_sources.append(
                _TrackOverlaySource(
                    source_index=src.source_index,
                    space_label=src.space_label,
                    time_label=src.time_label,
                    sensors=sensors,
                    times=times,
                    values_scaled_t=values_scaled_t,
                    slopes_values_t=slopes_values_t,
                    m_anchor=m_track,
                    idw_exponent=float(src.idw_exponent),
                    eps_snap=float(src.eps_snap),
                    lengthscale=lengthscale,
                    use_envelope=bool(src.use_envelope),
                    envelope_scale=float(src.envelope_scale),
                )
            )
        self.track_sources = tuple(track_sources)

    def apply(self, u0: DomainFunction, /) -> DomainFunction:
        deps = tuple(u0.domain.labels)
        idx = {lbl: i for i, lbl in enumerate(deps)}

        anchors = self.anchor_set.anchors
        values = jnp.asarray(self.anchor_set.values, dtype=float)
        src_index = self.anchor_set.source_index
        idw_exp = self.anchor_set.idw_exponent
        eps_snap = self.anchor_set.eps_snap
        lengthscales = self.anchor_set.lengthscales
        env_enabled = self.anchor_set.envelope_enabled
        env_scale = self.anchor_set.envelope_scale
        m_anchor = self.m_anchor
        track_sources = self.track_sources

        gate_exps = dict(self.gate_exponents)
        geom_factors = dict(self.geometry_factors)
        q = (self.max_init_order + 1) if self.max_init_order >= 0 else 0
        t0 = self.t0 if self.t0 is not None else jnp.asarray(0.0, dtype=float)
        evolution_var = self.evolution_var

        if values.shape[0] == 0 and not track_sources:
            raise ValueError("Interior data overlay has no anchors or tracks.")

        def _M_query(z_by_label: Mapping[str, Array], /) -> Array:
            m = jnp.asarray(1.0, dtype=float)
            for lbl, p in gate_exps.items():
                sdf = jnp.asarray(
                    geom_factors[lbl].adf(z_by_label[lbl]), dtype=float
                ).reshape(())
                m = m * (jnp.abs(sdf) ** int(p))
            if q > 0:
                t = jnp.asarray(z_by_label[evolution_var], dtype=float).reshape(())
                m = m * (jnp.maximum(t - t0, 0.0) ** int(q))
            return m

        def _call(*args: Any, key=None, **kwargs: Any):
            z = {lbl: args[idx[lbl]] for lbl in deps}

            u_query = jnp.asarray(
                u0.func(*(z[lbl] for lbl in u0.deps), key=key, **kwargs), dtype=float
            )

            def _u0_at_anchor(*dep_vals):
                return jnp.asarray(u0.func(*dep_vals, key=key, **kwargs), dtype=float)

            coord_indices = [i for i, arg in enumerate(args) if isinstance(arg, tuple)]
            if coord_indices and track_sources:
                raise ValueError(
                    "Hermite sensor tracks are not supported for coord-separable batches."
                )

            if not coord_indices:
                r_parts: list[Array] = []
                d2_parts: list[Array] = []
                idw_parts: list[Array] = []
                eps_parts: list[Array] = []
                src_parts: list[Array] = []

                if values.shape[0] > 0:
                    if u0.deps:
                        u_anchor = jax.vmap(_u0_at_anchor)(
                            *(anchors[lbl] for lbl in u0.deps)
                        )
                    else:
                        base = jnp.asarray(u0.func(key=key, **kwargs), dtype=float)
                        u_anchor = jnp.broadcast_to(base, values.shape)

                    y = values
                    if y.ndim == 1 and u_anchor.ndim == 2 and u_anchor.shape[1] == 1:
                        u_anchor = u_anchor.reshape((-1,))
                    if y.ndim == 2 and u_anchor.ndim == 1:
                        u_anchor = u_anchor.reshape((-1, 1))

                    r_static = (
                        (y - u_anchor) / m_anchor[:, None]
                        if y.ndim == 2
                        else (y - u_anchor) / m_anchor
                    )

                    d2_static = jnp.asarray(0.0, dtype=float)
                    for lbl in deps:
                        a = anchors[lbl]
                        ls = lengthscales[lbl]
                        zq = z[lbl]
                        if a.ndim == 2:
                            diff = (a - zq[None, :]) / ls[:, None]
                            d2_static = d2_static + jnp.sum(diff * diff, axis=1)
                        else:
                            diff = (a - zq) / ls
                            d2_static = d2_static + diff * diff

                    r_parts.append(r_static)
                    d2_parts.append(d2_static)
                    idw_parts.append(idw_exp)
                    eps_parts.append(eps_snap)
                    src_parts.append(src_index)

                for track in track_sources:
                    sensors = track.sensors
                    times = track.times
                    m_count = int(sensors.shape[0])
                    n_count = int(times.shape[0])

                    if u0.deps:
                        xs = jnp.broadcast_to(
                            sensors[:, None, :], (m_count, n_count, sensors.shape[1])
                        )
                        xs_flat = xs.reshape((-1, sensors.shape[1]))
                        ts_flat = jnp.broadcast_to(
                            times[None, :], (m_count, n_count)
                        ).reshape((-1,))

                        dep_vals: list[Array] = []
                        for lbl in u0.deps:
                            if lbl == track.space_label:
                                dep_vals.append(xs_flat)
                            elif lbl == track.time_label:
                                dep_vals.append(ts_flat)
                            else:
                                raise ValueError(
                                    "Sensor tracks require field dependencies to match space/time labels."
                                )
                        u_anchor_flat = jax.vmap(_u0_at_anchor)(*dep_vals)
                    else:
                        base = jnp.asarray(u0.func(key=key, **kwargs), dtype=float)
                        u_anchor_flat = jnp.broadcast_to(base, (m_count * n_count,))

                    if u_anchor_flat.ndim == 1:
                        u_anchor_track = u_anchor_flat.reshape((m_count, n_count))
                    elif u_anchor_flat.ndim == 2:
                        u_anchor_track = u_anchor_flat.reshape(
                            (m_count, n_count, u_anchor_flat.shape[1])
                        )
                    else:
                        raise ValueError(
                            "Unsupported anchor evaluation shape for sensor tracks."
                        )

                    if u_anchor_track.ndim == 2 and track.values_scaled_t.ndim == 3:
                        u_anchor_track = u_anchor_track[..., None]

                    if u_anchor_track.ndim == 2:
                        u_scaled = u_anchor_track / track.m_anchor
                    else:
                        u_scaled = u_anchor_track / track.m_anchor[..., None]
                    u_scaled_t = jnp.moveaxis(u_scaled, 1, 0)
                    if u_scaled_t.ndim == 2 and track.values_scaled_t.ndim == 3:
                        u_scaled_t = u_scaled_t[..., None]

                    if n_count < 2:
                        slopes_u_scaled_t = jnp.zeros_like(u_scaled_t)
                    else:
                        slopes_u_scaled_t = interpax.approx_df(times, u_scaled_t, axis=0)

                    y_spline = interpax.CubicHermiteSpline(
                        times,
                        track.values_scaled_t,
                        track.slopes_values_t,
                        axis=0,
                        check=False,
                    )
                    u_spline = interpax.CubicHermiteSpline(
                        times, u_scaled_t, slopes_u_scaled_t, axis=0, check=False
                    )
                    t_query = jnp.asarray(z[track.time_label], dtype=float)
                    y_scaled = y_spline(t_query)
                    u_scaled_q = u_spline(t_query)
                    r_track = y_scaled - u_scaled_q
                    if r_track.ndim == 2 and r_track.shape[1] == 1:
                        r_track = r_track.reshape((-1,))

                    zq = jnp.asarray(z[track.space_label], dtype=float)
                    if zq.ndim == 0:
                        zq_vec = zq.reshape((1,))
                    else:
                        zq_vec = zq.reshape((-1,))
                    diff = (sensors - zq_vec[None, :]) / track.lengthscale
                    d2_track = jnp.sum(diff * diff, axis=1)

                    r_parts.append(r_track)
                    d2_parts.append(d2_track)
                    idw_parts.append(
                        jnp.full((m_count,), track.idw_exponent, dtype=float)
                    )
                    eps_parts.append(jnp.full((m_count,), track.eps_snap, dtype=float))
                    src_parts.append(
                        jnp.full((m_count,), track.source_index, dtype=jnp.int32)
                    )

                if not r_parts:
                    raise ValueError("Interior data overlay has no anchors or tracks.")

                scalar_output = all(r.ndim == 1 for r in r_parts)
                if scalar_output:
                    r_all = jnp.concatenate(r_parts, axis=0)
                else:
                    width: int | None = None
                    r_aligned: list[Array] = []
                    for r in r_parts:
                        if r.ndim == 1:
                            if width is None:
                                width = 1
                            if width != 1:
                                raise ValueError(
                                    "Inconsistent interior anchor value shapes."
                                )
                            r_aligned.append(r.reshape((-1, 1)))
                        else:
                            if width is None:
                                width = int(r.shape[1])
                            if int(r.shape[1]) != width:
                                raise ValueError(
                                    "Inconsistent interior anchor value shapes."
                                )
                            r_aligned.append(r)
                    r_all = jnp.concatenate(r_aligned, axis=0)

                d2 = jnp.concatenate(d2_parts, axis=0)
                idw_all = jnp.concatenate(idw_parts, axis=0)
                eps_all = jnp.concatenate(eps_parts, axis=0)
                src_all = jnp.concatenate(src_parts, axis=0)

                n = int(d2.shape[0])
                jstar = jnp.argmin(d2)
                is_snap = d2[jstar] < eps_all[jstar]

                w_idw = _idw_weights(d2, idw_exponent=idw_all, eps=1e-12)
                w_snap = jnp.eye(n, dtype=float)[jstar]
                w = jnp.where(is_snap, w_snap, w_idw)

                if any(env_enabled):
                    psi_src: list[Array] = []
                    for si, enabled in enumerate(env_enabled):
                        if not enabled:
                            psi_src.append(jnp.asarray(1.0, dtype=float))
                            continue
                        mask = src_all == si
                        d2_min = jnp.min(jnp.where(mask, d2, jnp.asarray(jnp.inf)))
                        s = env_scale[si]
                        psi_src.append(jnp.exp(-(d2_min / ((s * s) + 1e-12))))
                    psi_src_arr = jnp.stack(psi_src, axis=0)
                    psi = psi_src_arr[src_all]
                else:
                    psi = jnp.ones_like(w)

                wpsi = w * psi
                if r_all.ndim == 2:
                    corr = jnp.sum(wpsi[:, None] * r_all, axis=0)
                else:
                    corr = jnp.sum(wpsi * r_all)

                m_q = _M_query(z)
                return u_query + m_q * corr

            if u0.deps:
                u_anchor = jax.vmap(_u0_at_anchor)(*(anchors[lbl] for lbl in u0.deps))
            else:
                base = jnp.asarray(u0.func(key=key, **kwargs), dtype=float)
                u_anchor = jnp.broadcast_to(base, values.shape)

            y = values
            if y.ndim == 1 and u_anchor.ndim == 2 and u_anchor.shape[1] == 1:
                u_anchor = u_anchor.reshape((-1,))
            if y.ndim == 2 and u_anchor.ndim == 1:
                u_anchor = u_anchor.reshape((-1, 1))

            r = (
                (y - u_anchor) / m_anchor[:, None]
                if y.ndim == 2
                else (y - u_anchor) / m_anchor
            )

            axis_pos: dict[tuple[int, int], int] = {}
            total_axes = 0
            for i in coord_indices:
                coords = args[i]
                for j in range(len(coords)):
                    axis_pos[(i, j)] = total_axes
                    total_axes += 1

            coord_axes: dict[tuple[int, int], Array] = {}
            for i in coord_indices:
                coords = args[i]
                for j, coord in enumerate(coords):
                    arr = jnp.asarray(coord, dtype=float).reshape((-1,))
                    shape = [1] * total_axes
                    shape[axis_pos[(i, j)]] = int(arr.shape[0])
                    coord_axes[(i, j)] = jnp.reshape(arr, tuple(shape))

            def _geom_coords(label: str, /) -> Array:
                zq = z[label]
                if not isinstance(zq, tuple):
                    return jnp.asarray(zq, dtype=float)
                i = idx[label]
                coords = [coord_axes[(i, j)] for j in range(len(zq))]
                if len(coords) == 1:
                    return coords[0]
                return jnp.stack(coords, axis=-1)

            d2 = jnp.asarray(0.0, dtype=float)
            if values.shape[0] == 0:
                raise ValueError(
                    "coord-separable interior data requires explicit anchors."
                )
            for lbl in deps:
                a = anchors[lbl]
                ls = lengthscales[lbl]
                zq = z[lbl]
                if isinstance(zq, tuple):
                    if a.ndim != 2:
                        raise TypeError(
                            f"coord-separable distance expects geometry anchors for {lbl!r}."
                        )
                    if len(zq) != a.shape[1]:
                        raise ValueError(
                            f"coord-separable {lbl!r} expects {a.shape[1]} axes, got {len(zq)}."
                        )
                    diff2 = jnp.asarray(0.0, dtype=float)
                    for j in range(a.shape[1]):
                        coord = coord_axes[(idx[lbl], j)]
                        a_j = a[:, j].reshape((a.shape[0],) + (1,) * total_axes)
                        ls_j = ls.reshape((ls.shape[0],) + (1,) * total_axes)
                        diff = (a_j - coord) / ls_j
                        diff2 = diff2 + diff * diff
                    d2 = d2 + diff2
                else:
                    if a.ndim == 2:
                        diff = (a - zq[None, :]) / ls[:, None]
                        d2_add = jnp.sum(diff * diff, axis=1)
                    else:
                        diff = (a - zq) / ls
                        d2_add = diff * diff
                    if d2_add.ndim == 1:
                        d2_add = d2_add.reshape((d2_add.shape[0],) + (1,) * total_axes)
                    d2 = d2 + d2_add

            n = int(src_index.shape[0])
            p = idw_exp.reshape((n,) + (1,) * total_axes)
            w_raw = jnp.where(
                p == 2.0, 1.0 / (d2 + 1e-12), 1.0 / ((d2 + 1e-12) ** (p / 2.0))
            )
            wsum = jnp.sum(w_raw, axis=0) + 1e-12
            w_idw = w_raw / wsum

            jstar = jnp.argmin(d2, axis=0)
            d2_min = jnp.min(d2, axis=0)
            eps_here = jnp.take(eps_snap, jstar)
            is_snap = d2_min < eps_here
            w_snap = jax.nn.one_hot(jstar, n, axis=0, dtype=float)
            w = jnp.where(is_snap[None, ...], w_snap, w_idw)

            if any(env_enabled):
                psi_src: list[Array] = []
                mask_shape = (n,) + (1,) * total_axes
                for si, enabled in enumerate(env_enabled):
                    if not enabled:
                        psi_src.append(jnp.asarray(1.0, dtype=float))
                        continue
                    mask = (src_index == si).reshape(mask_shape)
                    d2_min = jnp.min(jnp.where(mask, d2, jnp.asarray(jnp.inf)), axis=0)
                    s = env_scale[si]
                    psi_src.append(jnp.exp(-(d2_min / ((s * s) + 1e-12))))
                psi_src_arr = jnp.stack(psi_src, axis=0)
                psi = psi_src_arr[src_index]
            else:
                psi = jnp.ones_like(w)

            wpsi = w * psi
            if r.ndim == 1:
                r_b = r.reshape((r.shape[0],) + (1,) * total_axes)
                corr = jnp.sum(wpsi * r_b, axis=0)
            else:
                r_b = r.reshape((r.shape[0],) + (1,) * total_axes + (r.shape[1],))
                corr = jnp.sum(wpsi[..., None] * r_b, axis=0)

            m_q = jnp.asarray(1.0, dtype=float)
            for lbl, pwr in gate_exps.items():
                sdf = jnp.asarray(geom_factors[lbl].adf(_geom_coords(lbl)), dtype=float)
                m_q = m_q * (jnp.abs(sdf) ** int(pwr))
            if q > 0:
                t = jnp.asarray(z[evolution_var], dtype=float).reshape(())
                m_q = m_q * (jnp.maximum(t - t0, 0.0) ** int(q))

            return u_query + m_q * corr

        return DomainFunction(
            domain=u0.domain, deps=deps, func=_call, metadata=u0.metadata
        )


class EnforcedConstraintPipeline(StrictModule):
    r"""Compose enforced overlays for a single field.

    A pipeline takes a base field $u$ and returns an enforced field $\tilde u$
    after applying a sequence of *enforced* transformations:

    1. **Boundary overlays** (if any), typically enforcing conditions on
       $\partial\Omega$ using smooth blend weights.
    2. **Initial overlays** (if any), enforcing values and/or time-derivative
       targets at $t=t_0$.
    3. **Interior data overlays** (optional), enforcing interior anchors/tracks
       while preserving the boundary/initial conditions via a multiplicative
       gate that vanishes on the constrained sets.

    Boundary/initial stages are blended using a *boundary gate* $\gamma(z)$,
    where $\gamma=0$ on the constrained boundary and $\gamma\approx 1$ away from it:

    $$
    u \leftarrow u + \gamma\,(u_{\text{next}}-u).
    $$

    This ensures that later stages do not re-violate boundary enforced constraints.
    """

    field: str
    evolution_var: str
    boundary: tuple[_BoundaryBlendOverlay, ...]
    initial_overlay: "_InitialEnforcedOverlay | None"
    initial: tuple[SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint, ...]
    interior_data: _InteriorDataOverlay | None
    boundary_gate: DomainFunction | None
    co_vars: tuple[str, ...]

    def __init__(
        self,
        u_base: DomainFunction,
        /,
        *,
        field: str,
        constraints: Sequence[
            SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint
        ] = (),
        interior_data: Sequence[EnforcedInteriorData] = (),
        evolution_var: str = "t",
        include_identity_remainder: bool = True,
        num_reference: int = 3_000_000,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""Build a pipeline for one field.

        **Arguments:**

        - `u_base`: Base `DomainFunction` for the field.

        **Keyword arguments:**

        - `field`: Field name (used to match constraints/data to the field).
        - `constraints`: Enforced constraint terms (single-field or multi-field).
        - `interior_data`: Interior data sources used to build an IDW-based enforced
          overlay.
        - `evolution_var`: Name of the time-like label used to detect initial
          constraints (default `"t"`).
        - `include_identity_remainder`: When blending multiple boundary pieces,
          include a remainder weight for the identity map (keeps $u$ unchanged
          away from all pieces).
        - `num_reference`: Reference sample count used to normalize boundary blend weights.
        - `sampler`: Sampler used to draw reference points.
        - `key`: PRNG key used to draw reference points.

        Notes:

        - Boundary staging currently requires boundary constraints to specify
          exactly one geometry boundary label, shared across all boundary pieces.
        """
        self.field = str(field)
        self.evolution_var = str(evolution_var)

        boundary_constraints: list[
            SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint
        ] = []
        initial_constraints: list[
            SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint
        ] = []
        initial_target_constraints: list[SingleFieldEnforcedConstraint] = []
        for c in constraints:
            stage = _constraint_stage(c.component, self.evolution_var)
            if stage == "boundary":
                boundary_constraints.append(c)
            elif stage == "initial":
                initial_constraints.append(c)
                if (
                    isinstance(c, SingleFieldEnforcedConstraint)
                    and c.initial_target is not None
                ):
                    initial_target_constraints.append(c)

        boundary_overlays: list[_BoundaryBlendOverlay] = []
        if boundary_constraints:
            # For now: single geometry boundary label for all boundary constraints.
            labels = [
                _geometry_boundary_labels(c.component) for c in boundary_constraints
            ]
            for ls in labels:
                if len(ls) != 1:
                    raise ValueError(
                        "Boundary enforced constraints must specify exactly one geometry Boundary() label."
                    )
            bvar = labels[0][0]
            if any(ls[0] != bvar for ls in labels[1:]):
                raise ValueError(
                    "Boundary enforced constraints must share exactly one geometry Boundary() label."
                )
            boundary_overlays.append(
                _BoundaryBlendOverlay(
                    u_base,
                    boundary_constraints,
                    var=bvar,
                    include_identity_remainder=include_identity_remainder,
                    num_reference=num_reference,
                    sampler=sampler,
                    key=key,
                )
            )
        self.boundary = tuple(boundary_overlays)

        initial_overlay: _InitialEnforcedOverlay | None = None
        if initial_target_constraints:
            base_component = initial_target_constraints[0].component
            var = _initial_label(base_component, self.evolution_var)
            if var is None:
                raise ValueError(
                    "Initial enforced targets require a scalar FixedStart/FixedEnd/Fixed component."
                )

            comp = base_component.spec.component_for(var)
            if not isinstance(comp, (FixedStart, FixedEnd, Fixed)):
                raise ValueError(
                    "Initial enforced targets require FixedStart/FixedEnd/Fixed for the evolution var."
                )

            factor = _unwrap_factor(base_component.domain.factor(var))
            if not isinstance(factor, _AbstractScalarDomain):
                raise TypeError(
                    "Initial enforced targets require a scalar evolution variable."
                )

            targets_by_order: dict[int, DomainFunction | ArrayLike] = {}
            for c in initial_target_constraints:
                if c.component is not base_component:
                    raise ValueError(
                        "Initial enforced targets must share the same component."
                    )
                order = int(c.time_derivative_order)
                if order < 0:
                    raise ValueError(
                        "Initial enforced targets require non-negative derivative orders."
                    )
                if order in targets_by_order:
                    raise ValueError(
                        f"Initial enforced targets include duplicate order {order}."
                    )
                if c.initial_target is None:
                    raise ValueError(
                        "Initial enforced targets require a non-None initial_target."
                    )
                targets_by_order[order] = c.initial_target

            max_order = max(targets_by_order)
            for order in range(max_order + 1):
                if order not in targets_by_order:
                    raise ValueError(
                        "Initial enforced targets must provide all derivative orders from 0..max_order."
                    )

            initial_overlay = _InitialEnforcedOverlay(
                base_component,
                var=var,
                targets=targets_by_order,
            )
            initial_constraints = [
                c
                for c in initial_constraints
                if not (
                    isinstance(c, SingleFieldEnforcedConstraint)
                    and c.initial_target is not None
                )
            ]

        self.initial_overlay = initial_overlay
        self.initial = tuple(initial_constraints)

        boundary_exps = _gate_exponents_from_boundary_constraints(
            constraints, evolution_var=self.evolution_var
        )
        needs_boundary_gate = (self.initial_overlay is not None) or bool(self.initial)
        if boundary_exps and needs_boundary_gate:
            gate_labels = tuple(boundary_exps.keys())
            gate_factors_raw = tuple(
                _unwrap_factor(u_base.domain.factor(lbl)) for lbl in gate_labels
            )
            gate_factors_list: list[_AbstractGeometry] = []
            for lbl, factor in zip(gate_labels, gate_factors_raw, strict=True):
                if not isinstance(factor, _AbstractGeometry):
                    raise TypeError(
                        f"Boundary gate label {lbl!r} must refer to a geometry factor."
                    )
                gate_factors_list.append(factor)
            gate_factors = tuple(gate_factors_list)
            gate_powers = tuple(int(boundary_exps[lbl]) for lbl in gate_labels)
            gate_scales = tuple(
                float(
                    jnp.max(
                        jnp.abs(
                            jnp.asarray(
                                f.adf(
                                    f.sample_interior(
                                        int(num_reference),
                                        sampler=sampler,
                                        key=jr.fold_in(key, i + 1),
                                    )
                                ),
                                dtype=float,
                            )
                        )
                    )
                    + 1e-12
                )
                for i, f in enumerate(gate_factors)
            )
            gate_widths = tuple(
                float(
                    jnp.linalg.norm(
                        jnp.asarray(f.mesh_bounds, dtype=float)[1]
                        - jnp.asarray(f.mesh_bounds, dtype=float)[0]
                    )
                    + 1e-12
                )
                * 0.05
                for f in gate_factors
            )

            def _gate(*args, key=None, **kwargs):
                del key, kwargs
                m = jnp.asarray(1.0, dtype=float)
                for arg, factor, power, width, scale in zip(
                    args,
                    gate_factors,
                    gate_powers,
                    gate_widths,
                    gate_scales,
                    strict=True,
                ):
                    sdf = jnp.asarray(factor.adf(arg), dtype=float)
                    w = jnp.asarray(width, dtype=float)
                    s = jnp.asarray(scale, dtype=float)
                    x = jnp.abs(sdf) / (w * s + 1e-12)
                    if int(power) != 1:
                        x = x ** int(power)
                    gamma = 1.0 - jnp.exp(-(x * x))
                    m = m * gamma
                return m

            self.boundary_gate = DomainFunction(
                domain=u_base.domain,
                deps=gate_labels,
                func=_gate,
                metadata={},
            )
        else:
            self.boundary_gate = None
        max_init_order = _max_initial_order(constraints, evolution_var=self.evolution_var)

        interior_data_overlay: _InteriorDataOverlay | None = None
        if interior_data:
            anchor_set = _build_anchor_set(interior_data, domain=u_base.domain)
            interior_data_overlay = _InteriorDataOverlay(
                u_base.domain,
                anchor_set,
                gate_exponents=boundary_exps,
                evolution_var=self.evolution_var,
                max_init_order=max_init_order,
            )
        self.interior_data = interior_data_overlay

        deps: set[str] = set()
        for c in constraints:
            deps.update(c.co_vars)
        self.co_vars = tuple(sorted(deps))

    def apply(
        self,
        u_base: DomainFunction,
        /,
        *,
        get_field: Callable[[str], DomainFunction],
    ) -> DomainFunction:
        u = u_base
        for overlay in self.boundary:
            u = overlay.apply(u, get_field=get_field)
        if self.initial_overlay is not None:
            u_next = self.initial_overlay.apply(u)
            if self.boundary_gate is not None:
                u = u + self.boundary_gate * (u_next - u)
            else:
                u = u_next
        for c in self.initial:
            if isinstance(c, SingleFieldEnforcedConstraint):
                u_next = c.apply(u)
            else:
                u_next = c.apply(u, get_field)
            if self.boundary_gate is not None:
                u = u + self.boundary_gate * (u_next - u)
            else:
                u = u_next
        if self.interior_data is not None:
            u = self.interior_data.apply(u)
        return u


class EnforcedConstraintPipelines(StrictModule):
    r"""Enforced constraint pipelines for multiple fields.

    When enforcing multiple fields $\{u^{(k)}\}$, some enforced constraints may
    require access to *co-variables* (other fields). This object builds a
    directed acyclic graph (DAG) from `MultiFieldEnforcedConstraint.co_vars` and
    applies per-field `EnforcedConstraintPipeline`s in a topological order so that
    dependencies are available when needed.
    """

    pipelines: frozendict[str, EnforcedConstraintPipeline]
    order: tuple[str, ...]

    def __init__(
        self,
        pipelines: Mapping[str, EnforcedConstraintPipeline],
        /,
        *,
        field_order: Sequence[str],
    ):
        r"""Create a multi-field pipeline orchestrator.

        **Arguments:**

        - `pipelines`: Mapping `{field: EnforcedConstraintPipeline}`.

        **Keyword arguments:**

        - `field_order`: Preferred ordering for tie-breaking in the toposort
          (typically `tuple(functions.keys())`).
        """
        self.pipelines = frozendict(pipelines)
        self.order = _toposort(self.pipelines, field_order=tuple(field_order))

    @classmethod
    def build(
        cls,
        *,
        functions: Mapping[str, DomainFunction],
        constraints: Sequence[
            SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint
        ] = (),
        interior_data: Sequence[EnforcedInteriorData] = (),
        evolution_var: str = "t",
        include_identity_remainder: bool = True,
        num_reference: int = 3_000_000,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> "EnforcedConstraintPipelines":
        field_order = tuple(functions.keys())

        by_field_constraints: dict[
            str, list[SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint]
        ] = {}
        for c in constraints:
            by_field_constraints.setdefault(c.field, []).append(c)

        by_field_data: dict[str, list[EnforcedInteriorData]] = {}
        for d in interior_data:
            by_field_data.setdefault(d.field, []).append(d)

        pipelines: dict[str, EnforcedConstraintPipeline] = {}
        for field, u_base in functions.items():
            cs = by_field_constraints.get(field, [])
            ds = by_field_data.get(field, [])
            if not cs and not ds:
                continue
            pipelines[field] = EnforcedConstraintPipeline(
                u_base,
                field=field,
                constraints=cs,
                interior_data=ds,
                evolution_var=evolution_var,
                include_identity_remainder=include_identity_remainder,
                num_reference=num_reference,
                sampler=sampler,
                key=key,
            )

        return cls(pipelines, field_order=field_order)

    def apply(
        self, functions: Mapping[str, DomainFunction], /
    ) -> frozendict[str, DomainFunction]:
        r"""Apply all pipelines and return an enforced field mapping.

        Pipelines are applied in a dependency-respecting order. If a pipeline
        for field $u$ requires co-variables $\{v\}$, then those $v$ are taken from
        the *current* enforced mapping as the iteration proceeds.
        """
        out: dict[str, DomainFunction] = dict(functions)

        def get_field(name: str) -> DomainFunction:
            if name in out:
                return out[name]
            raise KeyError(f"Unknown field {name!r}.")

        for field in self.order:
            pipe = self.pipelines[field]
            u_base = functions[field]
            out[field] = pipe.apply(u_base, get_field=get_field)
        return frozendict(out)


def _toposort(
    pipelines: Mapping[str, EnforcedConstraintPipeline],
    /,
    *,
    field_order: tuple[str, ...],
) -> tuple[str, ...]:
    # Kahn's algorithm with deterministic ordering: preserve provided field_order for ties.
    deps: dict[str, set[str]] = {}
    rev: dict[str, set[str]] = {}
    for field, pipe in pipelines.items():
        req = set(pipe.co_vars).intersection(pipelines.keys())
        deps[field] = set(req)
        for d in req:
            rev.setdefault(d, set()).add(field)

    remaining = set(pipelines.keys())
    order: list[str] = []

    def _ready() -> list[str]:
        ready = [f for f in field_order if f in remaining and not deps.get(f)]
        for f in remaining:
            if f not in ready and not deps.get(f) and f not in field_order:
                ready.append(f)
        return ready

    while remaining:
        ready = _ready()
        if not ready:
            cycle = tuple(sorted(remaining))
            raise ValueError(
                f"EnforcedConstraintPipelines dependency cycle detected among {cycle!r}."
            )
        for f in ready:
            remaining.remove(f)
            order.append(f)
            for nxt in rev.get(f, ()):
                deps[nxt].discard(f)
    return tuple(order)
