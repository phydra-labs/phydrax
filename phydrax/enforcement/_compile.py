#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import (
    AbstractGeometry,
    AbstractScalarDomain,
    Boundary,
    CallbackDerivativeRule,
    Domain,
    DomainComponent,
    DomainFunction,
    EnforcementGateMethod,
    Fixed,
    FixedEnd,
    FixedStart,
    PointSampling,
    SampleLayout,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._interpolation import (
    cubic_hermite_interpolate,
    inverse_distance_stencil,
    local_cubic_slopes,
)
from .._strict import StrictModule
from ..operators.differential._domain_ops import partial_n
from ..operators.differential._hooks import (
    blend_with_gate,
    nth_quotient_rule,
    with_derivative_rule,
)
from ._ansatz import _enforcement_weight_fn, enforce_initial
from ._spec import EnforcementSpec


def _unwrap_factor(factor: object, /) -> object:
    return factor


def _geometry_boundary_labels(component: DomainComponent, /) -> tuple[str, ...]:
    out: list[str] = []
    for lbl in component.domain.labels:
        if not isinstance(component.spec.selection_for(lbl), Boundary):
            continue
        factor = _unwrap_factor(component.domain.factor(lbl))
        if isinstance(factor, AbstractGeometry):
            out.append(lbl)
    return tuple(out)


def _initial_label(component: DomainComponent, evolution_var: str, /) -> str | None:
    if evolution_var not in component.domain.labels:
        return None
    comp = component.spec.selection_for(evolution_var)
    if isinstance(comp, (FixedStart, FixedEnd, Fixed, Boundary)):
        return evolution_var
    return None


class InteriorAnchors(StrictModule):
    r"""Interior data source for enforced data overlays.

    `InteriorAnchors` represents measurements to enforce by construction in the
    interior without competing with boundary or initial hard transforms.

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

    See `EnforcementProgram` for how this integrates with other enforcement stages.
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
                    "InteriorAnchors requires either (points, values) or "
                    "(sensors, times, sensor_values)."
                )
        else:
            if self.values is None:
                raise ValueError("InteriorAnchors(points=...) requires values=...")

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


def _as_anchor_array(domain: Domain, label: str, x: Array, /) -> Array:
    factor = _unwrap_factor(domain.factor(label))
    arr = jnp.asarray(x, dtype=float)

    if isinstance(factor, AbstractGeometry):
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        if arr.ndim != 2:
            raise ValueError(
                f"Geometry anchor {label!r} must have shape (N,d), got {arr.shape}."
            )
        d = int(factor.spatial_dim)
        if arr.shape[1] != d:
            raise ValueError(
                f"Geometry anchor {label!r} must have d={d}, got {arr.shape[1]}."
            )
        return arr

    if isinstance(factor, AbstractScalarDomain):
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
    sources: Sequence[InteriorAnchors],
    *,
    domain: Domain,
) -> _UnifiedAnchorSet:
    if not sources:
        raise ValueError("Must provide at least one InteriorAnchors source.")

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
            if isinstance(factor, AbstractGeometry):
                anchors_cat[lbl] = jnp.zeros((0, int(factor.spatial_dim)), dtype=float)
            elif isinstance(factor, AbstractScalarDomain):
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


def _gate_exponents_from_boundary_specs(
    specs: Sequence[EnforcementSpec],
    /,
    *,
    evolution_var: str,
) -> dict[str, int]:
    exps: dict[str, int] = {}
    for c in specs:
        if c.stage != "boundary":
            continue
        labels = _geometry_boundary_labels(c.component)
        if len(labels) != 1:
            raise ValueError(
                "Boundary enforcement specs must select exactly one geometry boundary."
            )
        lbl = labels[0]
        exps[lbl] = max(exps.get(lbl, 0), int(c.max_derivative_order) + 1)
    return exps


def _max_initial_order(
    specs: Sequence[EnforcementSpec],
    /,
    *,
    evolution_var: str,
) -> int:
    orders: list[int] = []
    for c in specs:
        if c.stage != "initial":
            continue
        orders.append(int(c.time_derivative_order))
    return max(orders) if orders else -1


def _initial_overlay_boundary_compatible(
    *,
    u_base: DomainFunction,
    boundary_overlays: Sequence["_BoundaryBlendOverlay"],
    initial_overlay: "_InitialEnforcedOverlay",
    key: Key[Array, ""],
    num_probe: int = 64,
    atol: float = 1e-8,
) -> bool:
    if not boundary_overlays:
        return False

    for overlay in boundary_overlays:
        for piece in overlay.pieces:
            if piece.dependencies:
                return False
            # A value-only probe cannot establish compatibility of derivative
            # boundary conditions. Conservatively retain the powered boundary
            # gate so later stages cannot alter the declared boundary jet.
            if int(piece.max_derivative_order) > 0:
                return False

    def _get_field_unavailable(name: str, /) -> DomainFunction:
        raise KeyError(
            f"Boundary-compatibility probe cannot resolve co-variable {name!r}."
        )

    u_boundary = u_base
    for overlay in boundary_overlays:
        u_boundary = overlay.apply(u_boundary, get_field=_get_field_unavailable)

    u_initial = initial_overlay.apply(u_boundary)
    u_reenforced = u_initial
    for overlay in boundary_overlays:
        u_reenforced = overlay.apply(u_reenforced, get_field=_get_field_unavailable)

    diff = u_initial - u_reenforced

    probe_components: list[DomainComponent] = []
    for overlay in boundary_overlays:
        for piece in overlay.pieces:
            probe_components.append(piece.component)

    unique_components: list[DomainComponent] = []
    seen_ids: set[int] = set()
    for comp in probe_components:
        comp_id = id(comp)
        if comp_id in seen_ids:
            continue
        seen_ids.add(comp_id)
        unique_components.append(comp)

    n_probe = max(int(num_probe), 2)
    for i, component in enumerate(unique_components):
        non_fixed_labels = tuple(
            lbl
            for lbl in component.domain.labels
            if not isinstance(
                component.spec.selection_for(lbl), (FixedStart, FixedEnd, Fixed)
            )
        )
        if not non_fixed_labels:
            vals = jnp.asarray(diff.func(key=jr.fold_in(key, i)), dtype=float)
        else:
            structure = SampleLayout((non_fixed_labels,))
            batch = component.sample(
                PointSampling(n_probe, layout=structure, design="uniform"),
                key=jr.fold_in(key, i),
            )
            vals = jnp.asarray(diff(batch, key=jr.fold_in(key, i + 10_000)).data)
        if bool(jnp.any(~jnp.isfinite(vals))):
            return False
        if float(jnp.max(jnp.abs(vals))) > float(atol):
            return False

    return True


class _BoundaryWeightedQuotientCallable(StrictModule):
    pieces: tuple[DomainFunction, ...]
    weights: tuple[DomainFunction, ...]
    remainder_weight: DomainFunction | None
    base: DomainFunction
    piece_pos: tuple[tuple[int, ...], ...]
    weight_pos: tuple[tuple[int, ...], ...]
    remainder_weight_pos: tuple[int, ...] | None
    base_pos: tuple[int, ...]

    def __init__(
        self,
        *,
        pieces: tuple[DomainFunction, ...],
        weights: tuple[DomainFunction, ...],
        remainder_weight: DomainFunction | None,
        base: DomainFunction,
        piece_pos: tuple[tuple[int, ...], ...],
        weight_pos: tuple[tuple[int, ...], ...],
        remainder_weight_pos: tuple[int, ...] | None,
        base_pos: tuple[int, ...],
    ):
        self.pieces = pieces
        self.weights = weights
        self.remainder_weight = remainder_weight
        self.base = base
        self.piece_pos = piece_pos
        self.weight_pos = weight_pos
        self.remainder_weight_pos = remainder_weight_pos
        self.base_pos = base_pos

    def __call__(self, *args, key=None, **kwargs):
        num = jnp.asarray(0.0, dtype=float)
        den = jnp.asarray(0.0, dtype=float)

        for w, p, w_pos, p_pos in zip(
            self.weights,
            self.pieces,
            self.weight_pos,
            self.piece_pos,
            strict=True,
        ):
            w_args = tuple(args[i] for i in w_pos)
            p_args = tuple(args[i] for i in p_pos)
            w_val = w.func(*w_args, key=key, **kwargs)
            p_val = p.func(*p_args, key=key, **kwargs)
            num = num + w_val * p_val
            den = den + w_val

        if self.remainder_weight is not None:
            rem_pos = self.remainder_weight_pos
            if rem_pos is None:
                raise ValueError("Missing remainder weight argument positions.")
            w_args = tuple(args[i] for i in rem_pos)
            u_args = tuple(args[i] for i in self.base_pos)
            w_val = self.remainder_weight.func(*w_args, key=key, **kwargs)
            u_val = self.base.func(*u_args, key=key, **kwargs)
            num = num + w_val * u_val
            den = den + w_val

        return num / den


class _BoundaryBlendOverlay(StrictModule):
    var: str
    pieces: tuple[EnforcementSpec, ...]
    include_identity_remainder: bool
    weights: tuple[DomainFunction, ...]
    remainder_weight: DomainFunction | None

    def __init__(
        self,
        u_base: DomainFunction,
        pieces: Sequence[EnforcementSpec],
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
        if not isinstance(base_factor, AbstractGeometry):
            raise TypeError("_BoundaryBlendOverlay requires a geometry label.")
        geom = base_factor

        for c in self.pieces:
            if self.var not in c.component.domain.labels:
                raise KeyError(
                    f"Label {self.var!r} not in piece domain {c.component.domain.labels}."
                )
            factor = _unwrap_factor(c.component.domain.factor(self.var))
            if not isinstance(factor, AbstractGeometry):
                raise TypeError("Boundary pieces must use a geometry label for var.")
            if not geom.same_support(factor):
                raise ValueError(
                    "Boundary blend requires all pieces to share an equivalent geometry."
                )
            comp = c.component.spec.selection_for(self.var)
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
            w_fn = _enforcement_weight_fn(
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
                w_rem_fn = _enforcement_weight_fn(
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
        piece_functions: list[DomainFunction] = []

        for c, w in zip(self.pieces, self.weights, strict=True):
            u_piece = c.apply(u, get_field)
            piece_functions.append(u_piece)
            num = num + w * u_piece
            den = den + w

        if self.remainder_weight is not None:
            num = num + self.remainder_weight * u
            den = den + self.remainder_weight

        blended_expr = num / den
        deps = blended_expr.deps
        dep_idx = {lbl: i for i, lbl in enumerate(deps)}
        piece_pos = tuple(
            tuple(dep_idx[lbl] for lbl in piece.deps) for piece in piece_functions
        )
        weight_pos = tuple(
            tuple(dep_idx[lbl] for lbl in weight.deps) for weight in self.weights
        )
        remainder_weight_pos: tuple[int, ...] | None = None
        base_pos: tuple[int, ...] = ()
        if self.remainder_weight is not None:
            remainder_weight_pos = tuple(
                dep_idx[lbl] for lbl in self.remainder_weight.deps
            )
            base_pos = tuple(dep_idx[lbl] for lbl in u.deps)

        blended = DomainFunction(
            domain=blended_expr.domain,
            deps=deps,
            func=_BoundaryWeightedQuotientCallable(
                pieces=tuple(piece_functions),
                weights=self.weights,
                remainder_weight=self.remainder_weight,
                base=u,
                piece_pos=piece_pos,
                weight_pos=weight_pos,
                remainder_weight_pos=remainder_weight_pos,
                base_pos=base_pos,
            ),
            metadata=blended_expr.metadata,
        )

        def _hook(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode: Literal["reverse", "forward"],
            backend: Literal["ad", "jet", "fd", "basis"],
            basis: Literal["poly", "fourier", "sine", "cosine"],
            periodic: bool,
        ) -> DomainFunction | None:
            if backend not in ("ad", "jet"):
                return None

            def _derive(fn: DomainFunction, k: int, /) -> DomainFunction:
                return partial_n(
                    fn,
                    var=var,
                    axis=axis,
                    order=int(k),
                    mode=mode,
                    backend=backend,
                    basis=basis,
                    periodic=periodic,
                )

            return nth_quotient_rule(
                num,
                den,
                var=var,
                order=int(order),
                derive=_derive,
            )

        return with_derivative_rule(blended, CallbackDerivativeRule(_hook))


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
    distances = jnp.asarray(d2, dtype=float)
    count = int(distances.shape[0])
    stencil = inverse_distance_stencil(
        jnp.arange(count, dtype=jnp.int32),
        distances,
        source_size=count,
        power=idw_exponent,
        regularization=eps,
    )
    return stencil.weights


class _InteriorAnchorOverlay(StrictModule):
    anchor_set: _UnifiedAnchorSet
    gate_exponents: frozendict[str, int]
    evolution_var: str
    max_init_order: int
    t0: Array | None
    geometry_gates: frozendict[str, Callable[[Array], Array]]
    m_anchor: Array
    track_sources: tuple[_TrackOverlaySource, ...]

    def __init__(
        self,
        domain: Domain,
        anchor_set: _UnifiedAnchorSet,
        /,
        *,
        gate_exponents: Mapping[str, int],
        evolution_var: str,
        max_init_order: int,
        gate_method: EnforcementGateMethod,
        gate_saturation_fraction: float,
        gate_linear_fraction: float,
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
            if isinstance(factor, AbstractScalarDomain):
                t0 = jnp.asarray(factor.fixed("start"), dtype=float).reshape(())
        self.t0 = t0

        gates: dict[str, Callable[[Array], Array]] = {}
        for label in self.gate_exponents:
            factor = _unwrap_factor(domain.factor(label))
            if not isinstance(factor, AbstractGeometry):
                raise TypeError(
                    "Boundary gating exponents must refer to geometry labels."
                )
            gates[label] = factor.make_enforcement_gate(
                method=gate_method,
                saturation_fraction=gate_saturation_fraction,
                linear_fraction=gate_linear_fraction,
            )
        self.geometry_gates = frozendict(gates)

        # Precompute M(anchor_i) to validate anchors and reuse inside the overlay.
        n = int(self.anchor_set.source_index.shape[0])
        m_anchor = jnp.ones((n,), dtype=float)
        for lbl, p in self.gate_exponents.items():
            x = jnp.asarray(self.anchor_set.anchors[lbl], dtype=float)
            gate = self.geometry_gates[lbl](x)
            m_anchor = m_anchor * (jnp.abs(gate) ** int(p))

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
                gate = self.geometry_gates[lbl](sensors)
                m_track = m_track * (jnp.abs(gate) ** int(p))[:, None]

            if q > 0:
                if self.t0 is None:
                    raise ValueError("Missing scalar time domain for evolution_var.")
                m_track = m_track * (jnp.maximum(times - self.t0, 0.0) ** int(q))[None, :]

            if int(jnp.sum(m_track <= 0)) != 0:
                raise ValueError("Interior anchors include points where M(z_i)=0.")

            values_scaled = values / m_track[..., None]
            values_scaled_t = jnp.moveaxis(values_scaled, 1, 0)
            slopes_values_t = local_cubic_slopes(times, values_scaled_t)

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
        geom_gates = dict(self.geometry_gates)
        q = (self.max_init_order + 1) if self.max_init_order >= 0 else 0
        t0 = self.t0 if self.t0 is not None else jnp.asarray(0.0, dtype=float)
        evolution_var = self.evolution_var

        if values.shape[0] == 0 and not track_sources:
            raise ValueError("Interior data overlay has no anchors or tracks.")

        def _M_query(z_by_label: Mapping[str, Array], /) -> Array:
            m = jnp.asarray(1.0, dtype=float)
            for lbl, p in gate_exps.items():
                gate = jnp.asarray(geom_gates[lbl](z_by_label[lbl]), dtype=float).reshape(
                    ()
                )
                m = m * (jnp.abs(gate) ** int(p))
            if q > 0:
                t = jnp.asarray(z_by_label[evolution_var], dtype=float).reshape(())
                m = m * (jnp.maximum(t - t0, 0.0) ** int(q))
            return m

        def _correction(*args: Any, key=None, **kwargs: Any):
            z = {lbl: args[idx[lbl]] for lbl in deps}

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

                    slopes_u_scaled_t = local_cubic_slopes(times, u_scaled_t)
                    t_query = jnp.asarray(z[track.time_label], dtype=float)
                    y_scaled = cubic_hermite_interpolate(
                        times,
                        track.values_scaled_t,
                        t_query,
                        slopes=track.slopes_values_t,
                        bounds="extrapolate",
                    ).values
                    u_scaled_q = cubic_hermite_interpolate(
                        times,
                        u_scaled_t,
                        t_query,
                        slopes=slopes_u_scaled_t,
                        bounds="extrapolate",
                    ).values
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
                return m_q * corr

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
            distance_squared = jnp.moveaxis(d2, 0, -1)
            candidate_indices = jnp.broadcast_to(
                jnp.arange(n, dtype=jnp.int32),
                distance_squared.shape,
            )
            stencil = inverse_distance_stencil(
                candidate_indices,
                distance_squared,
                source_size=n,
                power=idw_exp,
                regularization=1e-12,
                snap_tolerance_squared=eps_snap,
                snap_policy="first",
            )
            w = jnp.moveaxis(stencil.weights, -1, 0)

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
            for lbl, power in gate_exps.items():
                gate = jnp.asarray(geom_gates[lbl](_geom_coords(lbl)), dtype=float)
                m_q = m_q * (jnp.abs(gate) ** int(power))
            if q > 0:
                t = jnp.asarray(z[evolution_var], dtype=float).reshape(())
                m_q = m_q * (jnp.maximum(t - t0, 0.0) ** int(q))

            return m_q * corr

        correction = DomainFunction(
            domain=u0.domain,
            deps=deps,
            func=_correction,
            metadata={"interior_data_correction": True},
        )
        ansatz = u0 + correction

        def _hook(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode: Literal["reverse", "forward"],
            backend: Literal["ad", "jet", "fd", "basis"],
            basis: Literal["poly", "fourier", "sine", "cosine"],
            periodic: bool,
        ) -> DomainFunction | None:
            if backend not in ("ad", "jet"):
                return None
            return partial_n(
                u0,
                var=var,
                axis=axis,
                order=int(order),
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            ) + partial_n(
                correction,
                var=var,
                axis=axis,
                order=int(order),
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )

        return with_derivative_rule(ansatz, CallbackDerivativeRule(_hook))


class _FieldEnforcementPipeline(StrictModule):
    r"""Compose enforced overlays for a single field.

    A pipeline takes a base field $u$ and returns an enforced field $\tilde u$
    after applying three ordered stages:

    1. **Boundary overlays**, compiled from boundary-condition specs.
    2. **Initial overlays**, compiled from initial value and derivative specs.
    3. **Interior anchor overlays**, preserving earlier exact conditions with a
       multiplicative gate that vanishes on their supports.

    Boundary and initial stages use a dimensionless gate $\gamma(z)$:

    $$
    u \leftarrow u + \gamma\,(u_{\text{next}}-u).
    $$

    The gate prevents later stages from violating already-enforced conditions.
    """

    field: str
    evolution_var: str
    boundary: tuple[_BoundaryBlendOverlay, ...]
    initial_overlay: "_InitialEnforcedOverlay | None"
    initial: tuple[EnforcementSpec, ...]
    interior: _InteriorAnchorOverlay | None
    boundary_gate: DomainFunction | None
    initial_overlay_boundary_compatible: bool
    co_vars: tuple[str, ...]

    def __init__(
        self,
        u_base: DomainFunction,
        /,
        *,
        field: str,
        specs: Sequence[EnforcementSpec] = (),
        interior: Sequence[InteriorAnchors] = (),
        evolution_var: str = "t",
        include_identity_remainder: bool = True,
        gate_method: EnforcementGateMethod = "auto",
        gate_saturation_fraction: float = 0.5,
        gate_linear_fraction: float = 0.5,
        num_reference: int = 3_000_000,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""Build a pipeline for one field.

        **Arguments:**

        - `u_base`: Base `DomainFunction` for the field.

        **Keyword arguments:**

        - `field`: Field selected by these specifications and anchors.
        - `specs`: Typed hard-enforcement specifications for this field.
        - `interior`: Exact interior anchor sources.
        - `evolution_var`: Time-like label used to identify initial conditions.
        - `include_identity_remainder`: When blending multiple boundary pieces,
          include a remainder weight for the identity map (keeps $u$ unchanged
          away from all pieces).
        - `gate_method`: CAD gate implementation. ``"auto"`` selects the global
          R-equivalence gate; ``"compact"`` selects the compact fallback.
        - `gate_saturation_fraction`: Relative extent of compact CAD preservation
          gates. Used only when ``gate_method="compact"``.
        - `gate_linear_fraction`: Fraction of the compact gate extent retaining a
          linear boundary profile.
        - `num_reference`: Reference sample count used to normalize boundary blend weights.
        - `sampler`: Sampler used to draw reference points.
        - `key`: PRNG key used to draw reference points.

        Notes:

        - Boundary staging requires every specification to select one shared
          geometry boundary label.
        """
        self.field = str(field)
        self.evolution_var = str(evolution_var)

        boundary_specs: list[EnforcementSpec] = []
        initial_specs: list[EnforcementSpec] = []
        initial_target_specs: list[EnforcementSpec] = []
        for c in specs:
            stage = c.stage
            if stage == "boundary":
                boundary_specs.append(c)
            elif stage == "initial":
                initial_specs.append(c)
                if c.initial_target is not None:
                    initial_target_specs.append(c)
            else:
                initial_specs.append(c)

        boundary_overlays: list[_BoundaryBlendOverlay] = []
        if boundary_specs:
            # For now: single geometry boundary label for all boundary constraints.
            labels = [_geometry_boundary_labels(c.component) for c in boundary_specs]
            for ls in labels:
                if len(ls) != 1:
                    raise ValueError(
                        "Boundary enforcement specs must select exactly one geometry boundary."
                    )
            bvar = labels[0][0]
            if any(ls[0] != bvar for ls in labels[1:]):
                raise ValueError(
                    "Boundary enforcement specs must share one geometry boundary label."
                )
            boundary_overlays.append(
                _BoundaryBlendOverlay(
                    u_base,
                    boundary_specs,
                    var=bvar,
                    include_identity_remainder=include_identity_remainder,
                    num_reference=num_reference,
                    sampler=sampler,
                    key=key,
                )
            )
        self.boundary = tuple(boundary_overlays)

        initial_overlay: _InitialEnforcedOverlay | None = None
        if initial_target_specs:
            base_component = initial_target_specs[0].component
            var = _initial_label(base_component, self.evolution_var)
            if var is None:
                raise ValueError(
                    "Initial enforced targets require a scalar FixedStart/FixedEnd/Fixed component."
                )

            comp = base_component.spec.selection_for(var)
            if not isinstance(comp, (FixedStart, FixedEnd, Fixed)):
                raise ValueError(
                    "Initial enforced targets require FixedStart/FixedEnd/Fixed for the evolution var."
                )

            factor = _unwrap_factor(base_component.domain.factor(var))
            if not isinstance(factor, AbstractScalarDomain):
                raise TypeError(
                    "Initial enforced targets require a scalar evolution variable."
                )

            targets_by_order: dict[int, DomainFunction | ArrayLike] = {}
            for c in initial_target_specs:
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
            initial_specs = [c for c in initial_specs if c.initial_target is None]

        self.initial_overlay = initial_overlay
        self.initial = tuple(initial_specs)
        self.initial_overlay_boundary_compatible = bool(
            (self.initial_overlay is not None)
            and _initial_overlay_boundary_compatible(
                u_base=u_base,
                boundary_overlays=self.boundary,
                initial_overlay=self.initial_overlay,
                key=jr.fold_in(key, 42_791),
            )
        )

        boundary_exps = _gate_exponents_from_boundary_specs(
            specs, evolution_var=self.evolution_var
        )
        needs_boundary_gate = (self.initial_overlay is not None) or bool(self.initial)
        if boundary_exps and needs_boundary_gate:
            gate_labels = tuple(boundary_exps.keys())
            gate_factors_raw = tuple(
                _unwrap_factor(u_base.domain.factor(lbl)) for lbl in gate_labels
            )
            gate_factors_list: list[AbstractGeometry] = []
            for lbl, factor in zip(gate_labels, gate_factors_raw, strict=True):
                if not isinstance(factor, AbstractGeometry):
                    raise TypeError(
                        f"Boundary gate label {lbl!r} must refer to a geometry factor."
                    )
                gate_factors_list.append(factor)
            gate_factors = tuple(gate_factors_list)
            gate_powers = tuple(int(boundary_exps[lbl]) for lbl in gate_labels)
            gate_functions = tuple(
                factor.make_enforcement_gate(
                    method=gate_method,
                    saturation_fraction=gate_saturation_fraction,
                    linear_fraction=gate_linear_fraction,
                )
                for factor in gate_factors
            )

            def _gate(*args, key=None, **kwargs):
                del key, kwargs
                value = jnp.asarray(1.0, dtype=float)
                for arg, gate, power in zip(
                    args,
                    gate_functions,
                    gate_powers,
                    strict=True,
                ):
                    gate_value = jnp.clip(
                        jnp.abs(jnp.asarray(gate(arg), dtype=float)),
                        0.0,
                        1.0,
                    )
                    if int(power) != 1:
                        gate_value = gate_value ** int(power)
                    value = value * gate_value
                return value

            self.boundary_gate = DomainFunction(
                domain=u_base.domain,
                deps=gate_labels,
                func=_gate,
                metadata={},
            )
        else:
            self.boundary_gate = None
        max_init_order = _max_initial_order(specs, evolution_var=self.evolution_var)

        interior_overlay: _InteriorAnchorOverlay | None = None
        if interior:
            anchor_set = _build_anchor_set(interior, domain=u_base.domain)
            interior_overlay = _InteriorAnchorOverlay(
                u_base.domain,
                anchor_set,
                gate_exponents=boundary_exps,
                gate_method=gate_method,
                evolution_var=self.evolution_var,
                max_init_order=max_init_order,
                gate_saturation_fraction=gate_saturation_fraction,
                gate_linear_fraction=gate_linear_fraction,
            )
        self.interior = interior_overlay

        deps: set[str] = set()
        for c in specs:
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
            if (
                self.boundary_gate is not None
                and not self.initial_overlay_boundary_compatible
            ):
                u = blend_with_gate(u, u_next, self.boundary_gate)
            else:
                u = u_next
        for spec in self.initial:
            u_next = spec.apply(u, get_field)
            if self.boundary_gate is not None:
                u = blend_with_gate(u, u_next, self.boundary_gate)
            else:
                u = u_next
        if self.interior is not None:
            u = self.interior.apply(u)
        return u


class EnforcementProgram(StrictModule):
    """Compiled hard-enforcement program for one or more fields.

    Specification dependencies form a directed acyclic graph. Field pipelines
    run in topological order and consume already-enforced dependencies.
    """

    pipelines: frozendict[str, _FieldEnforcementPipeline]
    order: tuple[str, ...]

    def __init__(
        self,
        pipelines: Mapping[str, _FieldEnforcementPipeline],
        /,
        *,
        field_order: Sequence[str],
    ):
        """Create a multi-field enforcement program.

        `pipelines` maps field names to compiled field pipelines. `field_order`
        provides deterministic tie-breaking for topological sorting.
        """
        self.pipelines = frozendict(pipelines)
        self.order = _toposort(self.pipelines, field_order=tuple(field_order))

    @classmethod
    def build(
        cls,
        *,
        functions: Mapping[str, DomainFunction],
        specs: Sequence[EnforcementSpec] = (),
        interior: Sequence[InteriorAnchors] = (),
        evolution_var: str = "t",
        include_identity_remainder: bool = True,
        gate_method: EnforcementGateMethod = "auto",
        gate_saturation_fraction: float = 0.5,
        gate_linear_fraction: float = 0.5,
        num_reference: int = 3_000_000,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> "EnforcementProgram":
        field_order = tuple(functions.keys())

        by_field_specs: dict[str, list[EnforcementSpec]] = {}
        for spec in specs:
            by_field_specs.setdefault(spec.field, []).append(spec)

        by_field_interior: dict[str, list[InteriorAnchors]] = {}
        for anchors in interior:
            by_field_interior.setdefault(anchors.field, []).append(anchors)

        pipelines: dict[str, _FieldEnforcementPipeline] = {}
        for field, u_base in functions.items():
            field_specs = by_field_specs.get(field, [])
            field_anchors = by_field_interior.get(field, [])
            if not field_specs and not field_anchors:
                continue
            pipelines[field] = _FieldEnforcementPipeline(
                u_base,
                field=field,
                specs=field_specs,
                interior=field_anchors,
                evolution_var=evolution_var,
                gate_method=gate_method,
                gate_saturation_fraction=gate_saturation_fraction,
                gate_linear_fraction=gate_linear_fraction,
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
    pipelines: Mapping[str, _FieldEnforcementPipeline],
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
                f"EnforcementProgram dependency cycle detected among {cycle!r}."
            )
        for f in ready:
            remaining.remove(f)
            order.append(f)
            for nxt in rev.get(f, ()):
                deps[nxt].discard(f)
    return tuple(order)
