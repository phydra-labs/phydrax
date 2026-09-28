#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coherent synchrotron radiation (CSR) of accelerator bunches.

Conventions
-----------
- Bunch-frame coordinates are relative to the reference particle: ``z`` is the
  distance *ahead* along the path (``z = −zeta`` for ``"positive-late"`` and
  ``z = +zeta`` for ``"positive-early"``), ``x`` the horizontal and ``y`` the
  vertical offset. A :class:`CSRLattice` is a planar path of drifts and
  circular arcs; positive curvature ``h = 1/ρ`` bends the orbit toward ``−x``,
  so ``+x`` points away from the center of curvature. Before the lattice the
  path continues as a straight line along the entrance direction, after it
  along the exit direction.
- Densities are charge densities (C/m for one-dimensional models, C/m³ for
  three-dimensional models) of the macroparticle charge
  ``weight × reference_charge``. Wakes are the energy change per unit path
  length of one particle of charge ``reference_charge`` (scale energy per
  scale length); ``Δδ = ΔE·E₀/(p₀c)²`` and ``Δ(p⊥/p₀) = F⊥Δs/(β₀ p₀c)``.
- Retarded models write the energy change as ``dE/ds = −dΦ/ds|particle +
  W_rad`` with the Liénard–Wiechert potential ``Φ`` and the radiative wake
  ``W_rad = (q/4πε₀)∫(β² u·u′ − 1) ∂λ/∂z′ ds′/r`` (Mayes and Hoffstaetter,
  PRST-AB 12, 024401, 2009). The field of the same density moving on a
  straight line (space charge) is subtracted pairwise, which removes the
  line-charge singularity (Saldin, Schneidmiller, and Yurkov, NIM A 398, 373,
  1997). Tracking applies ``−ΔΦ`` per particle between kicks, so the potential
  term telescopes exactly along each particle.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from enum import IntFlag
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import cubic_hermite_segment
from ..._physical import ElectromagneticScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...discretization import ParticleSetPlan, PreparedTensorGrid
from ...discretization.splatting import ParticleGridSplatPlan, PreparedParticleGridSplat
from ...nonlinear import NonlinearTermination, scalar_root, ScalarRootProblem, TOMS748
from ...operators.integral._free_space_convolution import FreeSpaceConvolutionPlan
from ...signal import convolve
from ...special import ellipeinc, ellipkinc
from ...typing import (
    AnyShape,
    Bool,
    ConvertibleToArray,
    Dim,
    Float64,
    Identifier,
    Int32,
    parse,
    Scalar,
)
from ._beam import AcceleratorBunch


CSRModel: TypeAlias = Literal[
    "1d-steady", "1d-transient-shielded", "3d-steady-igf", "3d-retarded-mesh"
]

_CANONICAL_MOMENTUM_NORMALIZATION = "px-over-p0,py-over-p0,delta-p-over-p0"
# Near-diagonal source window (in cells) integrated with geometric nodes; beyond
# it the trapezoid rule on the grid is accurate to ~1/(6M³) for 1/Δ kernels.
_NEAR_CELLS = 4
_NODES_PER_OCTAVE = 4
_MAXIMUM_OCTAVES = 64
_GAUSSIAN_TRUNCATION = 4.0


class _ParticleDim(Dim, minimum=1):
    """Bunch capacity."""


class _StepDim(Dim, minimum=1):
    """Tracking steps."""


class _SnapshotDim(Dim, minimum=1):
    """Stored density snapshots."""


class CSRStatus(IntFlag):
    """CSR evaluation evidence bits.

    Refusal bits (the kick is not applied): ``NONFINITE``, ``UNSUPPORTED``
    (an active particle left the grid), ``ROOT_FAILURE`` (a retarded-time solve
    did not converge), ``HISTORY_INCOMPLETE`` (a retarded time fell into dropped
    history), ``SHIELDING_TRUNCATED`` (the last image pair exceeds the declared
    tolerance). Validity bits (reported, not refused): ``DERBENEV_VIOLATED``
    (one-dimensional model with ``σ_x ≥ (σ_z²R)^{1/3}`` times the declared
    ratio) and ``NOT_STEADY`` (steady model inside the overtaking length
    ``(24σ_zR²)^{1/3}`` of the current bend).
    """

    SUCCESS = 0
    NONFINITE = 1
    UNSUPPORTED = 2
    ROOT_FAILURE = 4
    HISTORY_INCOMPLETE = 8
    SHIELDING_TRUNCATED = 16
    DERBENEV_VIOLATED = 32
    NOT_STEADY = 64


_REFUSAL_BITS = int(
    CSRStatus.NONFINITE
    | CSRStatus.UNSUPPORTED
    | CSRStatus.ROOT_FAILURE
    | CSRStatus.HISTORY_INCOMPLETE
    | CSRStatus.SHIELDING_TRUNCATED
)


class CSRResourceError(ValueError):
    """A CSR plan exceeds its declared resource limits."""


class CSRResources(StrictModule, NonTrainableState):
    """Declared per-evaluation limits of a CSR plan.

    ``maximum_retarded_pairs`` bounds the retarded-time solves of one field
    evaluation, ``maximum_history_bytes`` the stored density history, and
    ``maximum_kernel_bytes`` the prepared Green-function tables.
    ``observer_chunk`` bounds the observers contracted at once.
    """

    maximum_retarded_pairs: int = eqx.field(static=True)
    maximum_history_bytes: int = eqx.field(static=True)
    maximum_kernel_bytes: int = eqx.field(static=True)
    observer_chunk: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_retarded_pairs: int = 2**24,
        maximum_history_bytes: int = 2**31,
        maximum_kernel_bytes: int = 2**31,
        observer_chunk: int = 64,
    ) -> None:
        self.maximum_retarded_pairs = positive_integer(
            maximum_retarded_pairs, "maximum_retarded_pairs"
        )
        self.maximum_history_bytes = positive_integer(
            maximum_history_bytes, "maximum_history_bytes"
        )
        self.maximum_kernel_bytes = positive_integer(
            maximum_kernel_bytes, "maximum_kernel_bytes"
        )
        self.observer_chunk = positive_integer(observer_chunk, "observer_chunk")


class CSRResourceEstimate(NamedTuple):
    """Work and memory of one CSR evaluation."""

    retarded_pairs: int
    history_bytes: int
    kernel_bytes: int


class CSRLattice(StrictModule, NonTrainableState):
    """Planar reference path of drifts and circular arcs with pole-face angles.

    ``curvatures`` are signed ``1/ρ`` (zero for a drift; positive bends toward
    ``−x``). ``entrance_edges`` and ``exit_edges`` are hard-edge pole-face
    rotation angles (radians) applied as thin edge-focusing kicks by
    :func:`track_csr`; rectangular bends of angle ``θ`` use ``θ/2`` on each face.
    The path is extended by straight lines before and after the lattice.
    """

    lengths: Array
    curvatures: Array
    entrance_edges: Array
    exit_edges: Array
    element_ids: tuple[str, ...] = eqx.field(static=True)
    element_count: int = eqx.field(static=True)
    total_length: float = eqx.field(static=True)
    maximum_curvature: float = eqx.field(static=True)
    # Extended segment table: [upstream line, elements..., downstream line].
    segment_starts: Array
    segment_curvatures: Array
    segment_headings: Array
    segment_origins: Array
    origin_differences: Array
    lattice_id: str = eqx.field(static=True)

    def __init__(
        self,
        lengths: ConvertibleToArray,
        curvatures: ConvertibleToArray,
        /,
        *,
        element_ids: Iterable[str],
        entrance_edges: ConvertibleToArray | None = None,
        exit_edges: ConvertibleToArray | None = None,
    ) -> None:
        lengths_ = np.asarray(lengths, dtype=np.float64)
        curvatures_ = np.asarray(curvatures, dtype=np.float64)
        if lengths_.ndim != 1 or lengths_.size < 1:
            raise ValueError("lengths must be a non-empty vector.")
        count = lengths_.shape[0]
        entrance = (
            np.zeros((count,), dtype=np.float64)
            if entrance_edges is None
            else np.asarray(entrance_edges, dtype=np.float64)
        )
        exit_ = (
            np.zeros((count,), dtype=np.float64)
            if exit_edges is None
            else np.asarray(exit_edges, dtype=np.float64)
        )
        if (
            curvatures_.shape != (count,)
            or entrance.shape != (count,)
            or exit_.shape != (count,)
        ):
            raise ValueError("Lattice arrays must align with the element count.")
        if (
            not np.all(np.isfinite(lengths_))
            or np.any(lengths_ <= 0.0)
            or not np.all(np.isfinite(curvatures_))
            or not np.all(np.isfinite(entrance))
            or not np.all(np.isfinite(exit_))
            or np.any(np.abs(entrance) >= 0.5 * np.pi)
            or np.any(np.abs(exit_) >= 0.5 * np.pi)
        ):
            raise ValueError(
                "Lattice lengths must be positive and curvatures and edge angles "
                "finite, with |edge| < π/2."
            )
        if np.any((curvatures_ == 0.0) & ((entrance != 0.0) | (exit_ != 0.0))):
            raise ValueError("Drifts carry no pole-face angles.")
        ids = tuple(str(value).strip() for value in element_ids)
        if len(ids) != count or any(not value for value in ids) or len(set(ids)) != count:
            raise ValueError("element_ids must be unique, non-empty, and aligned.")
        starts = np.concatenate(([0.0], np.cumsum(lengths_)))
        headings = np.concatenate(([0.0], np.cumsum(curvatures_ * lengths_)))
        origins = np.zeros((count + 1, 2), dtype=np.float64)
        chords = np.zeros((count, 2), dtype=np.float64)
        for index in range(count):
            chords[index] = _host_chord(
                lengths_[index], curvatures_[index], headings[index]
            )
            origins[index + 1] = origins[index] + chords[index]
        # Extended table: index 0 is the upstream line (origin at the lattice
        # entrance), 1..count the elements, count+1 the downstream line.
        segment_starts = np.concatenate(([-np.inf], starts[:-1], [starts[-1]]))
        segment_curvatures = np.concatenate(([0.0], curvatures_, [0.0]))
        segment_headings = np.concatenate(([0.0], headings[:-1], [headings[-1]]))
        segment_origins = np.concatenate((origins[:1], origins[:-1], origins[-1:]))
        # Origin differences summed from chords keep relative geometry exact
        # to rounding of the chords between two segments, not of |X|.
        cumulative = np.concatenate(([np.zeros(2)], np.cumsum(chords, axis=0)))
        extended_cumulative = np.concatenate(
            (cumulative[:1], cumulative[:-1], cumulative[-1:])
        )
        differences = extended_cumulative[:, None, :] - extended_cumulative[None, :, :]
        self.lengths = jnp.asarray(lengths_)
        self.curvatures = jnp.asarray(curvatures_)
        self.entrance_edges = jnp.asarray(entrance)
        self.exit_edges = jnp.asarray(exit_)
        self.element_ids = ids
        self.element_count = count
        self.total_length = float(starts[-1])
        self.maximum_curvature = float(np.max(np.abs(curvatures_)))
        self.segment_starts = jnp.asarray(segment_starts)
        self.segment_curvatures = jnp.asarray(segment_curvatures)
        self.segment_headings = jnp.asarray(segment_headings)
        self.segment_origins = jnp.asarray(segment_origins)
        self.origin_differences = jnp.asarray(differences)
        self.lattice_id = canonical_fingerprint(
            {
                "kind": "csr-lattice",
                "arrays": array_tree_fingerprint(
                    (lengths_, curvatures_, entrance, exit_)
                ),
                "element_ids": list(ids),
            }
        )

    def _segment(self, s: Array) -> Array:
        index = jnp.searchsorted(self.segment_starts, s, side="right") - 1
        return jnp.clip(index, 0, self.element_count + 1)

    def curvature(self, s: ArrayLike, /) -> Array:
        """Signed curvature at path position(s) ``s``."""
        s_ = jnp.asarray(s, dtype=jnp.float64)
        return self.segment_curvatures[self._segment(s_)]

    def heading(self, s: ArrayLike, /) -> Array:
        """Direction angle of the tangent at ``s`` (radians)."""
        s_ = jnp.asarray(s, dtype=jnp.float64)
        segment = self._segment(s_)
        local = s_ - jnp.where(segment == 0, 0.0, self.segment_starts[segment])
        return self.segment_headings[segment] + self.segment_curvatures[segment] * local

    def position(self, s: ArrayLike, /) -> Array:
        """Planar position ``X(s)`` (``(..., 2)``) with the entrance at the origin."""
        s_ = jnp.asarray(s, dtype=jnp.float64)
        segment = self._segment(s_)
        return self.segment_origins[segment] + self._local(segment, s_)

    def _local(self, segment: Array, s: Array) -> Array:
        """Position relative to the segment origin (exact chord form)."""
        local = s - jnp.where(segment == 0, 0.0, self.segment_starts[segment])
        curvature = self.segment_curvatures[segment]
        heading = self.segment_headings[segment] + 0.5 * curvature * local
        chord = local * jnp.sinc(0.5 * curvature * local / jnp.pi)
        return chord[..., None] * jnp.stack((jnp.cos(heading), jnp.sin(heading)), -1)

    def _forward_chord(
        self,
        later: Array,
        later_segment: Array,
        earlier: Array,
        earlier_segment: Array,
        separation: Array,
    ) -> Array:
        """``X(later) − X(earlier)`` for the path separation ``later − earlier ≥ 0``.

        Within one segment the arc chord is formed from the exact separation;
        across segments the chord is split at the end of the earlier segment, so
        no absolute coordinate enters and nearby points keep relative precision.
        """
        later_curvature = self.segment_curvatures[later_segment]
        direct_heading = self.heading(later) - 0.5 * later_curvature * separation
        direct = (separation * jnp.sinc(0.5 * later_curvature * separation / jnp.pi))[
            ..., None
        ] * jnp.stack((jnp.cos(direct_heading), jnp.sin(direct_heading)), -1)
        following = jnp.minimum(earlier_segment + 1, self.element_count + 1)
        remaining = self.segment_starts[following] - earlier
        earlier_curvature = self.segment_curvatures[earlier_segment]
        tail_heading = self.heading(earlier) + 0.5 * earlier_curvature * remaining
        tail = (remaining * jnp.sinc(0.5 * earlier_curvature * remaining / jnp.pi))[
            ..., None
        ] * jnp.stack((jnp.cos(tail_heading), jnp.sin(tail_heading)), -1)
        across = (
            self._local(later_segment, later)
            + self.origin_differences[later_segment, following]
            + tail
        )
        return jnp.where((later_segment == earlier_segment)[..., None], direct, across)

    def _chord(self, observer: Array, observer_segment: Array, lag: Array) -> Array:
        """``X(observer) − X(observer − lag)`` for either sign of the path lag."""
        source = observer - lag
        source_segment = self._segment(source)
        behind = lag >= 0.0
        later = jnp.where(behind, observer, source)
        earlier = jnp.where(behind, source, observer)
        later_segment = jnp.where(behind, observer_segment, source_segment)
        earlier_segment = jnp.where(behind, source_segment, observer_segment)
        chord = self._forward_chord(
            later, later_segment, earlier, earlier_segment, jnp.abs(lag)
        )
        return jnp.where(behind[..., None], chord, -chord)

    def bend_entry_distance(self, s: ArrayLike, /) -> Array:
        """Path length since the entrance of the current arc (``inf`` in drifts).

        Consecutive arcs of the same curvature count as one arc.
        """
        s_ = jnp.asarray(s, dtype=jnp.float64)
        segment = self._segment(s_)
        curvature = self.segment_curvatures
        same = (curvature[1:] == curvature[:-1]) & (curvature[1:] != 0.0)
        starts = self.segment_starts
        # Start of the maximal run of equal nonzero curvature ending at each segment.
        run_start = starts
        for _ in range(self.element_count):
            run_start = run_start.at[1:].set(
                jnp.where(same, run_start[:-1], run_start[1:])
            )
        return jnp.where(curvature[segment] != 0.0, s_ - run_start[segment], jnp.inf)


def _host_chord(length: float, curvature: float, heading: float) -> np.ndarray:
    chord = length * np.sinc(0.5 * curvature * length / np.pi)
    angle = heading + 0.5 * curvature * length
    return chord * np.asarray([np.cos(angle), np.sin(angle)])


class _LagNodes(NamedTuple):
    """Positive retarded-offset nodes ``Δ`` of one side and their Δ-space weights.

    Geometric nodes span ``[Δ_min, M h]`` (log-trapezoid weights, exact for the
    ``1/Δ`` straight-line kernel); grid nodes ``k h`` continue to the window
    extent (trapezoid weights).
    """

    delta: Array
    weights: Array


def _lag_nodes(spacing: float, count: int, minimum: float) -> _LagNodes:
    upper = _NEAR_CELLS * spacing
    octaves = min(math.log2(upper / minimum), float(_MAXIMUM_OCTAVES))
    graded_count = max(int(math.ceil(_NODES_PER_OCTAVE * octaves)) + 1, 3)
    logarithms = np.linspace(
        math.log(upper) - octaves * math.log(2.0), math.log(upper), graded_count
    )
    graded = np.exp(logarithms)
    graded_weights = graded * (logarithms[1] - logarithms[0])
    graded_weights[0] *= 0.5
    graded_weights[-1] *= 0.5
    grid = spacing * np.arange(_NEAR_CELLS + 1, max(count, _NEAR_CELLS + 2))
    grid_weights = np.full(grid.shape, spacing)
    graded_weights[-1] += 0.5 * spacing
    return _LagNodes(
        jnp.asarray(np.concatenate((graded, grid))),
        jnp.asarray(np.concatenate((graded_weights, grid_weights))),
    )


class _PairGeometry(NamedTuple):
    radius: Array
    delta: Array
    alignment: Array
    transverse_alignment: Array


def _pair_geometry(
    lattice: CSRLattice,
    beta: float,
    observer: Array,
    observer_segment: Array,
    observer_heading: Array,
    displacement: Array,
    vertical: Array,
    lag: Array,
) -> _PairGeometry:
    """Distance, retarded offset ``v − βR``, and tangent alignments at path lag ``v``."""
    chord = lattice._chord(observer, observer_segment, lag) + displacement
    radius = jnp.sqrt(jnp.sum(chord * chord, axis=-1) + vertical * vertical)
    source_heading = lattice.heading(observer - lag)
    return _PairGeometry(
        radius,
        lag - beta * radius,
        jnp.cos(observer_heading - source_heading),
        jnp.sin(observer_heading - source_heading),
    )


def _retarded_lags(
    lattice: CSRLattice,
    beta: float,
    gamma: float,
    observer: Array,
    delta: Array,
    horizontal: Array,
    vertical: Array,
    tolerance: float,
) -> tuple[Array, Array]:
    """Path lags ``v`` with ``v − βR(v) = Δ`` for flat arrays of pairs.

    ``g(v) = v − βR(v) − Δ`` is increasing (``1 − β ≤ g′ ≤ 1 + β``) and, with
    ``R ≤ |v| + d``, ``g(Δ) ≤ 0`` while the upper bounds below give ``g ≥ 0``
    with a relative margin. The bracket spans up to ``γ²`` in scale, so the
    native bracketed root works in ``u = asinh(v/s)`` with ``s = |Δ| + d``,
    on the residual normalized by ``s``, and reports per-pair convergence.
    """
    segment = lattice._segment(observer)
    heading = lattice.heading(observer)
    normal = jnp.stack((jnp.sin(heading), -jnp.cos(heading)), -1)
    displacement = horizontal[..., None] * normal
    distance = jnp.sqrt(horizontal * horizontal + vertical * vertical)
    one_minus_beta = 1.0 / (gamma * gamma * (1.0 + beta))
    shifted = delta + beta * distance
    scale = jnp.maximum(jnp.abs(delta) + distance, jnp.finfo(jnp.float64).tiny)
    lower = delta
    upper = jnp.where(
        shifted <= 0.0,
        shifted * (1.0 - 1.0e-9) / (1.0 + beta),
        shifted * (1.0 + 1.0e-9) / one_minus_beta,
    )
    upper = jnp.maximum(upper, lower + 1.0e-30 + 1.0e-12 * jnp.abs(lower))
    termination = NonlinearTermination(
        absolute_residual=tolerance,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=1.0e-15,
        maximum_steps=200,
    )

    def residual(mapped: Array, index: Array) -> Array:
        lag = scale[index] * jnp.sinh(mapped)
        chord = lattice._chord(observer[index], segment[index], lag) + displacement[index]
        radius = jnp.sqrt(jnp.sum(chord * chord) + vertical[index] ** 2)
        return (lag - beta * radius - delta[index]) / scale[index]

    mapped_lower = jnp.arcsinh(lower / scale)
    mapped_upper = jnp.maximum(jnp.arcsinh(upper / scale), mapped_lower + 1.0e-12)

    def solve(index: Array) -> tuple[Array, Array]:
        problem = ScalarRootProblem(
            residual,
            bracket=(mapped_lower[index], mapped_upper[index]),
            problem_id="csr-retarded-lag",
        )
        result = scalar_root(
            problem, method=TOMS748(), termination=termination, args=index
        )
        return scale[index] * jnp.sinh(result.root), result.successful

    return jax.vmap(solve)(jnp.arange(delta.shape[0]))


class _LineSamples(NamedTuple):
    """Quadrature samples of retarded line integrals for every observer.

    ``location`` is the source bunch offset and ``retarded`` the bunch-center
    path position at the retarded time of each sample. ``weights`` holds four
    channels applied to the density (0: potential, 2: transverse potential,
    3: horizontal vector potential) or to its ``z`` derivative (1: radiative
    wake). Shapes are ``(lines, observers, samples)``.
    """

    location: Array
    retarded: Array
    weights: Array
    failures: Array


def _lag_measure(lag: Array, distance: Array) -> Array:
    """Trapezoid weights for ``∫ f dv`` over sorted lags ``v`` of one line.

    The rule is the trapezoid in ``w = asinh(v/d)`` (``d`` the line offset),
    which is exact for ``1/√(v² + d²)``; on the axis (``d = 0``) it becomes the
    trapezoid in ``sign(v)·ln|v|`` and the interval across ``v = 0`` (inside the
    omitted ``|Δ| < Δ_min``) carries no weight.
    """
    on_axis = distance <= 0.0
    safe = jnp.where(on_axis, 1.0, distance)
    magnitude = jnp.maximum(jnp.abs(lag), jnp.finfo(jnp.float64).tiny)
    mapped = jnp.where(
        on_axis, jnp.sign(lag) * jnp.log(magnitude), jnp.arcsinh(lag / safe)
    )
    jacobian = jnp.where(on_axis, magnitude, jnp.sqrt(lag * lag + safe * safe))
    step = jnp.diff(mapped, axis=-1)
    crossing = (lag[..., :-1] < 0.0) & (lag[..., 1:] > 0.0)
    step = jnp.where(on_axis & crossing, 0.0, step)
    padded = jnp.concatenate(
        (jnp.zeros_like(step[..., :1]), step, jnp.zeros_like(step[..., :1])), axis=-1
    )
    return jacobian * 0.5 * (padded[..., :-1] + padded[..., 1:])


def _line_samples(
    lattice: CSRLattice,
    position: Array,
    beta: float,
    gamma: float,
    observer: Array,
    horizontal: Array,
    vertical: Array,
    nodes: _LagNodes,
    far_count: int,
    *,
    subtract: bool,
    tolerance: float,
) -> _LineSamples:
    """Samples of the retarded integrals ``∫ ds′ f(λ(z′, t_r))/R`` along lines.

    The full (curved-path) part is integrated over the path lag ``v`` with the
    Jefimenko measure ``ds′/R`` on merged nodes: the retarded lags of the
    offsets ``±Δ_k`` (resolving the density), ``far_count`` geometric lags
    between the smallest and largest behind lag (resolving the far-upstream
    horizon where ``dΔ/dv → 0``), and, off axis, ``far_count`` lags
    ``d·sinh(t)`` resolving ``R ≈ √(v² + d²)``. With ``subtract`` the field of the
    same density on a straight line, ``∫ λ dz′/√(Δ² + d²/γ²)``, is subtracted in
    ``Δ``-space at the same samples; both parts omit ``|Δ| < Δ_min``.
    """
    lines = horizontal.shape[0]
    count = observer.shape[0]
    delta_nodes = nodes.delta
    side = delta_nodes.shape[0]
    signed = jnp.concatenate((-delta_nodes, delta_nodes))
    shape = (lines, count, 2 * side)
    s = jnp.broadcast_to((position + observer)[None, :, None], shape)
    lag, successful = _retarded_lags(
        lattice,
        beta,
        gamma,
        s.reshape(-1),
        jnp.broadcast_to(signed[None, None, :], shape).reshape(-1),
        jnp.broadcast_to(horizontal[:, None, None], shape).reshape(-1),
        jnp.broadcast_to(vertical[:, None, None], shape).reshape(-1),
        tolerance,
    )
    lag = lag.reshape(shape)
    distance = jnp.sqrt(horizontal * horizontal + vertical * vertical)[:, None, None]
    fraction = (jnp.arange(far_count, dtype=jnp.float64) + 0.5) / far_count
    behind = lag[..., side:]
    first = jnp.maximum(behind[..., :1], jnp.finfo(jnp.float64).tiny)
    last = jnp.maximum(behind[..., -1:], first)
    far = first * jnp.exp(fraction * jnp.log(last / first))
    stretch = jnp.arcsinh(
        jnp.maximum(last, first) / jnp.where(distance > 0.0, distance, 1.0)
    )
    near = jnp.where(
        distance > 0.0,
        distance * jnp.sinh((2.0 * fraction - 1.0) * stretch),
        first,
    )
    inverse_gamma_squared = 1.0 / (gamma * gamma)
    straight = (
        jnp.concatenate((nodes.weights, nodes.weights))
        / jnp.sqrt(signed * signed + distance * distance * inverse_gamma_squared)
        if subtract
        else jnp.zeros((1, 1, 2 * side))
    )
    straight = jnp.concatenate(
        (
            jnp.broadcast_to(straight, shape),
            jnp.zeros((lines, count, 2 * far_count)),
        ),
        axis=-1,
    )
    merged = jnp.concatenate((lag, far, near), axis=-1)
    order = jnp.argsort(merged, axis=-1)
    merged = jnp.take_along_axis(merged, order, axis=-1)
    straight = jnp.take_along_axis(straight, order, axis=-1)
    observers = jnp.broadcast_to((position + observer)[None, :, None], merged.shape)
    segment = lattice._segment(observers)
    heading = lattice.heading(observers)
    normal = jnp.stack((jnp.sin(heading), -jnp.cos(heading)), -1)
    displacement = (
        jnp.broadcast_to(horizontal[:, None, None], merged.shape)[..., None] * normal
    )
    pair = _pair_geometry(
        lattice,
        beta,
        observers,
        segment,
        heading,
        displacement,
        jnp.broadcast_to(vertical[:, None, None], merged.shape),
        merged,
    )
    inverse = _lag_measure(merged, distance) / pair.radius
    beta_squared = beta * beta
    weights = jnp.stack(
        (
            inverse - straight,
            (beta_squared * pair.alignment - 1.0) * inverse
            + inverse_gamma_squared * straight,
            (1.0 - beta_squared * pair.alignment) * inverse
            - inverse_gamma_squared * straight,
            beta * pair.transverse_alignment * inverse,
        )
    )
    return _LineSamples(
        jnp.broadcast_to(observer[None, :, None], merged.shape) - pair.delta,
        position - beta * pair.radius,
        weights,
        jnp.sum(~successful, dtype=jnp.int32),
    )


class _Timeline(NamedTuple):
    """Chronological density snapshots ``[initial, ring…, current]``."""

    positions: Array
    densities: Array
    slopes: Array
    dropped: Array
    oldest_retained: Array


class CSRState(StrictModule, NonTrainableState):
    """Density history and per-particle potential memory of CSR tracking.

    ``densities``/``slopes`` form a bounded ring of deposited charge densities and
    their ``z`` derivatives recorded at bunch-center path ``positions``;
    ``initial_*`` keep the snapshot at the start of tracking, which also stands
    for the bunch on the straight line before it (the incoming-drift model).
    ``potential`` and ``vector_potential`` hold each particle's Liénard–Wiechert
    potential energy ``qΦ`` and ``q c A_x`` at its previous kick, so the
    total-derivative terms telescope exactly along every particle.
    """

    __strict_contract__ = True

    densities: Float64[AnyShape]
    slopes: Float64[AnyShape]
    positions: Float64[_SnapshotDim]
    initial_density: Float64[AnyShape]
    initial_slope: Float64[AnyShape]
    initial_position: Float64[Scalar]
    cursor: Int32[Scalar]
    count: Int32[Scalar]
    potential: Float64[_ParticleDim]
    vector_potential: Float64[_ParticleDim]
    capacity: int = eqx.field(static=True)

    def __init__(
        self,
        densities: ArrayLike,
        slopes: ArrayLike,
        positions: ArrayLike,
        initial_density: ArrayLike,
        initial_slope: ArrayLike,
        initial_position: ArrayLike,
        cursor: ArrayLike,
        count: ArrayLike,
        potential: ArrayLike,
        vector_potential: ArrayLike,
        /,
    ) -> None:
        densities_ = jnp.asarray(densities, dtype=jnp.float64)
        slopes_ = jnp.asarray(slopes, dtype=jnp.float64)
        positions_ = jnp.asarray(positions, dtype=jnp.float64)
        if (
            densities_.ndim < 2
            or slopes_.shape != densities_.shape
            or positions_.shape != densities_.shape[:1]
        ):
            raise ValueError("CSR history arrays must share (capacity, *grid) shapes.")
        self.densities = densities_
        self.slopes = slopes_
        self.positions = positions_
        self.initial_density = jnp.asarray(initial_density, dtype=jnp.float64)
        self.initial_slope = jnp.asarray(initial_slope, dtype=jnp.float64)
        self.initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
        self.cursor = jnp.asarray(cursor, dtype=jnp.int32)
        self.count = jnp.asarray(count, dtype=jnp.int32)
        self.potential = jnp.asarray(potential, dtype=jnp.float64)
        self.vector_potential = jnp.asarray(vector_potential, dtype=jnp.float64)
        self.capacity = positions_.shape[0]

    def record(self, density: Array, slope: Array, position: Array, /) -> CSRState:
        """Store one snapshot, overwriting the oldest once ``capacity`` is reached."""
        slot = self.cursor
        return CSRState(
            self.densities.at[slot].set(density),
            self.slopes.at[slot].set(slope),
            self.positions.at[slot].set(position),
            self.initial_density,
            self.initial_slope,
            self.initial_position,
            (slot + 1) % self.capacity,
            self.count + 1,
            self.potential,
            self.vector_potential,
        )

    def _timeline(self, density: Array, slope: Array, position: Array) -> _Timeline:
        capacity = self.capacity
        stored = jnp.minimum(self.count, capacity)
        order = (self.cursor - capacity + jnp.arange(capacity)) % capacity
        valid = jnp.arange(capacity) >= capacity - stored
        mask = valid.reshape((capacity,) + (1,) * (self.densities.ndim - 1))
        ring_positions = jnp.where(valid, self.positions[order], self.initial_position)
        ring_densities = jnp.where(mask, self.densities[order], self.initial_density)
        ring_slopes = jnp.where(mask, self.slopes[order], self.initial_slope)
        return _Timeline(
            jnp.concatenate(
                (self.initial_position[None], ring_positions, position[None])
            ),
            jnp.concatenate((self.initial_density[None], ring_densities, density[None])),
            jnp.concatenate((self.initial_slope[None], ring_slopes, slope[None])),
            self.count > capacity,
            ring_positions[0],
        )


def _frozen_timeline(density: Array, slope: Array, position: Array) -> _Timeline:
    return _Timeline(
        position[None],
        density[None],
        slope[None],
        jnp.asarray(False),
        position,
    )


class _Sampled(NamedTuple):
    values: Array
    derivatives: Array
    incomplete: Array


def _sample_timeline(
    timeline: _Timeline,
    column: Array,
    location: Array,
    retarded: Array,
    lower_center: float,
    spacing: float,
) -> _Sampled:
    """Density and ``∂λ/∂z`` at bunch offsets and retarded positions.

    Longitudinally the cubic Hermite segment through the grid values and
    slopes; in time linear between the bracketing snapshots (before the first
    snapshot the initial one holds). ``timeline`` arrays are
    ``(times, columns, z)``.
    """
    times = timeline.positions.shape[0]
    count = timeline.densities.shape[-1]
    index = jnp.searchsorted(timeline.positions, retarded, side="right")
    earlier = jnp.clip(index - 1, 0, times - 1)
    later = jnp.clip(index, 0, times - 1)
    width = timeline.positions[later] - timeline.positions[earlier]
    weight = jnp.where(
        width > 0.0,
        jnp.clip(
            (retarded - timeline.positions[earlier]) / jnp.where(width > 0.0, width, 1.0),
            0.0,
            1.0,
        ),
        0.0,
    )
    position = (location - lower_center) / spacing
    lower = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, count - 2)
    fraction = position - lower
    inside = (position >= 0.0) & (position <= count - 1)

    def hermite(time: Array, order: int) -> Array:
        return cubic_hermite_segment(
            timeline.densities[time, column, lower],
            timeline.densities[time, column, lower + 1],
            timeline.slopes[time, column, lower],
            timeline.slopes[time, column, lower + 1],
            fraction,
            spacing,
            derivative_order=order,
        )

    values = (1.0 - weight) * hermite(earlier, 0) + weight * hermite(later, 0)
    derivatives = (1.0 - weight) * hermite(earlier, 1) + weight * hermite(later, 1)
    incomplete = (
        timeline.dropped
        & (retarded > timeline.positions[0])
        & (retarded < timeline.oldest_retained)
    )
    return _Sampled(
        jnp.where(inside, values, 0.0),
        jnp.where(inside, derivatives, 0.0),
        jnp.any(incomplete & inside),
    )


def _cai_ding_alpha(
    chi: Array, zeta: Array, xi: Array, beta: float
) -> tuple[Array, Array]:
    """Half retarded angle α of Cai–Ding (2020) Eq. (5), exactly.

    ``ξ = α − (β/2)√(χ² + ζ² + 4(1+χ) sin²α)`` is increasing in α; with
    ``κ₀ = √(χ² + ζ²)`` and ``κ_max = √(κ₀² + 4(1+χ))`` the root lies in
    ``[ξ + βκ₀/2, ξ + βκ_max/2]``.
    """
    base = chi * chi + zeta * zeta
    lower = xi + 0.5 * beta * jnp.sqrt(base)
    upper = xi + 0.5 * beta * jnp.sqrt(base + 4.0 * (1.0 + chi))
    upper = jnp.maximum(upper, lower + 1.0e-300)
    termination = NonlinearTermination(
        absolute_residual=1.0e-13,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=1.0e-15,
        maximum_steps=200,
    )

    def residual(alpha: Array, index: Array) -> Array:
        sine = jnp.sin(alpha)
        return (
            alpha
            - 0.5 * beta * jnp.sqrt(base[index] + 4.0 * (1.0 + chi[index]) * sine * sine)
            - xi[index]
        )

    def solve(index: Array) -> tuple[Array, Array]:
        problem = ScalarRootProblem(
            residual,
            bracket=(lower[index], upper[index]),
            problem_id="csr-cai-ding-retarded-angle",
        )
        result = scalar_root(
            problem, method=TOMS748(), termination=termination, args=index
        )
        return result.root, result.successful

    return jax.vmap(solve)(jnp.arange(xi.shape[0]))


@eqx.filter_jit
def _steady_potentials(
    chi: Array, zeta: Array, xi: Array, gamma: float
) -> tuple[Array, Array, Array, Array]:
    """Steady Green functions ``ψ_s``, ``ψ̂_x``, ``ψ_y`` of a source on a circle.

    Dimensionless ``χ = x/ρ``, ``ζ = y/ρ``, ``ξ = z/(2ρ)`` of the observer
    relative to a source on the reference orbit; each force is
    ``(qe β²/(2ρ²)) ∂ψ/∂ξ`` (prefactor omitted). ``ψ_s`` is Cai and Ding's
    longitudinal potential with its Coulomb term (PRAB 23, 014402, 2020,
    Eq. 23 and Appendix B).

    The transverse potentials are ``ξ``-antiderivatives of the exact Lorentz
    force on an observer moving at the reference velocity ``βc`` along its
    tangent, plus the curvilinear ``−qφ/ρ`` term. The source field rotates
    rigidly, so with ``Ψ = φ − βA_s`` and the Liénard–Wiechert potentials
    ``φ = 1/D``, ``A_s = β cos 2α/D``, ``A_x = β sin 2α/D``
    (``D = κ − β(1+χ) sin 2α``, units ``e/ρ``)::

        F̂_x = −∂_χΨ + βA_s/(1+χ) − φ + βχ/(2(1+χ)) ∂_ξA_x,
        F_y = −∂_ζΨ.

    Since ``φ dξ = dα/κ``, the antiderivatives of ``φ`` and ``A_s`` are
    incomplete elliptic integrals and their transverse derivatives at fixed
    ``ξ`` follow in closed form. Cai and Ding's published transverse
    potentials instead set ``β_s ≈ β`` in the magnetic term, which moves the
    observer rigidly with the bunch at speed ``β(1+χ)c``; the extra ``−βχB``
    acting on the near Coulomb field adds ``2⟨x²/r²⟩`` to the residual
    centripetal coefficient (``Λ = 3`` instead of ``2`` for a round beam).
    """
    beta_squared = 1.0 - 1.0 / (gamma * gamma)
    beta = math.sqrt(beta_squared)
    alpha, successful = _cai_ding_alpha(chi, zeta, xi, beta)
    kappa = 2.0 * (alpha - xi) / beta
    sin2a = jnp.sin(2.0 * alpha)
    cos2a = jnp.cos(2.0 * alpha)
    xp = 1.0 + chi
    longitudinal_denominator = kappa - beta * xp * sin2a
    psi_s = (cos2a - 1.0 / xp) / longitudinal_denominator - 1.0 / (
        (gamma * gamma - 1.0) * xp * longitudinal_denominator
    )
    # Transverse terms are evaluated as functions of α alone, so that the
    # large cancelling Coulomb pieces stay consistent with each other.
    sine = jnp.sin(alpha)
    cosine = jnp.cos(alpha)
    sine_squared = sine * sine
    a = chi * chi + zeta * zeta
    b = 4.0 * xp
    root_a = jnp.sqrt(a)
    radius = jnp.sqrt(a + b * sine_squared)
    denominator = radius - beta * xp * sin2a
    parameter = -b / a
    first = ellipkinc(alpha, parameter)
    second = ellipeinc(alpha, parameter)
    # ∫dα/κ, ∫κ dα, ∫dα/κ³ with κ² = a + b sin²α.
    inverse = first / root_a
    direct = root_a * second
    cubic = second / (root_a * (a + b)) + b * sine * cosine / (a * (a + b) * radius)
    # ∫φ dξ and ∫cos 2α φ dξ; cos 2α = 1 + 2a/b − 2κ²/b.
    potential = inverse
    aligned = (1.0 + 2.0 * a / b) * inverse - (2.0 / b) * direct
    shape = 1.0 - beta_squared * cos2a
    # ∂_χ at fixed α: ∂_χ(1/κ) = −(χ + 2 sin²α)/κ³, expanded in powers of κ².
    potential_chi = (chi - 2.0 * a / b) * cubic + (2.0 / b) * inverse
    aligned_chi = (
        (chi - (2.0 - 2.0 * chi) * a / b - 4.0 * a * a / (b * b)) * cubic
        + ((2.0 - 2.0 * chi) / b + 8.0 * a / (b * b)) * inverse
        - 4.0 / (b * b) * direct
    )
    alpha_chi = beta * (chi + 2.0 * sine_squared) / (2.0 * denominator)
    alpha_zeta = beta * zeta / (2.0 * denominator)
    transverse_chi = shape * alpha_chi / radius - (
        potential_chi - beta_squared * aligned_chi
    )
    transverse_zeta = shape * alpha_zeta / radius - zeta * (
        cubic - beta_squared * ((1.0 + 2.0 * a / b) * cubic - (2.0 / b) * inverse)
    )
    horizontal = (
        -transverse_chi
        + beta_squared * aligned / xp
        - potential
        + beta_squared * chi * sin2a / (2.0 * xp * denominator)
    )
    scale = 2.0 / beta_squared
    return psi_s, scale * horizontal, -scale * transverse_zeta, successful


def _steady_table(
    spacing: np.ndarray,
    shape: tuple[int, int, int],
    curvature: float,
    gamma: float,
    quadrature: int,
    offsets: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Integrated Green table ``g = (1/h_z)⟨ψ(Δz + h_z/2) − ψ(Δz − h_z/2)⟩_⊥``.

    The longitudinal derivative of the Green function is integrated exactly
    over each source cell (piecewise-constant density) and the transverse cell
    average uses a ``quadrature × quadrature`` Gauss–Legendre rule. Offsets
    default to every on-grid displacement; the result carries the
    ``(ψ_s, ψ̂_x, ψ_y)`` components along the last axis with ``x`` measured away
    from the center of curvature.
    """
    radius = 1.0 / abs(curvature)
    orientation = math.copysign(1.0, curvature)
    if offsets is None:
        offsets = tuple(
            spacing[axis] * np.arange(-(count - 1), count)
            for axis, count in enumerate(shape)
        )
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    nodes = 0.5 * nodes
    weights = 0.25 * np.outer(weights, weights)
    x = (
        offsets[0][:, None, None, None, None]
        + spacing[0] * nodes[None, None, None, :, None]
    )
    y = (
        offsets[1][None, :, None, None, None]
        + spacing[1] * nodes[None, None, None, None, :]
    )
    z = offsets[2][None, None, :, None, None]
    target = np.broadcast_shapes(x.shape, y.shape, z.shape)
    faces = []
    failures = 0
    for side in (0.5, -0.5):
        chi = np.broadcast_to(orientation * x / radius, target).reshape(-1)
        zeta = np.broadcast_to(y / radius, target).reshape(-1)
        xi = np.broadcast_to((z + side * spacing[2]) / (2.0 * radius), target).reshape(-1)
        psi_s, psi_x, psi_y, successful = _steady_potentials(
            jnp.asarray(chi), jnp.asarray(zeta), jnp.asarray(xi), gamma
        )
        failures += int(np.sum(~np.asarray(successful)))
        stacked = np.stack(
            (np.asarray(psi_s), orientation * np.asarray(psi_x), np.asarray(psi_y)), -1
        ).reshape(target + (3,))
        faces.append(np.tensordot(stacked, weights, axes=((3, 4), (0, 1))))
    return (faces[0] - faces[1]) / spacing[2], failures


def _grid_axes(
    grid: PreparedTensorGrid, dimension: int
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Cell centers and uniform spacings of a bounded cell-centered grid."""
    if not isinstance(grid, PreparedTensorGrid):
        raise TypeError("grid must be a PreparedTensorGrid.")
    if len(grid.shape) != dimension:
        raise ValueError(f"This CSR model needs a {dimension}-dimensional grid.")
    centers = []
    spacing = []
    for axis in grid.structured_axes:
        if axis.periodic or axis.primary_entity != "interval":
            raise ValueError("CSR grids need bounded, cell-centered (interval) axes.")
        coordinates = np.asarray(axis.interval_centers, dtype=np.float64)
        if coordinates.size < 2:
            raise ValueError("CSR grids need at least two cells per axis.")
        steps = np.diff(coordinates)
        mean = float(np.mean(steps))
        if mean <= 0.0 or np.max(np.abs(steps - mean)) > 1.0e-9 * mean:
            raise ValueError("CSR grids need uniform spacing.")
        centers.append(coordinates)
        spacing.append(mean)
    return tuple(centers), np.asarray(spacing, dtype=np.float64)


def _gaussian_taps(width: float, spacing: float) -> np.ndarray | None:
    if width == 0.0:
        return None
    sigma = width / spacing
    half = max(int(math.ceil(_GAUSSIAN_TRUNCATION * sigma)), 1)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.exp(-0.5 * (offsets / sigma) ** 2)
    return taps / np.sum(taps)


class _Fields(NamedTuple):
    """Grid fields per unit ``q/(4πε₀)`` before the particle kick."""

    potential: Array
    wake: Array
    horizontal: Array
    vertical: Array
    vector_x: Array
    failures: Array
    incomplete: Array
    truncation: Array


class CSREvidence(StrictModule):
    """Validity, resolution, and integrity evidence of one CSR evaluation.

    ``status`` holds :class:`CSRStatus` bits. ``derbenev_ratio`` is
    ``σ_x/(σ_z²R)^{1/3}`` (one-dimensional models in a bend, else zero);
    ``overtaking_length`` is ``(24σ_zR²)^{1/3}`` and ``bend_entry_distance`` the
    path length since the current arc began (``inf`` in drifts).
    ``shielding_truncation`` is the largest field of the last image pair over
    the largest total field; ``retarded_failures`` counts retarded-time solves
    that did not converge; ``history_complete`` is false when a retarded time
    fell into dropped history. ``support_fraction`` is the share of active
    particles deposited inside the grid.
    """

    __strict_contract__ = True

    status: Int32[Scalar]
    derbenev_ratio: Float64[Scalar]
    overtaking_length: Float64[Scalar]
    bend_entry_distance: Float64[Scalar]
    shielding_truncation: Float64[Scalar]
    retarded_failures: Int32[Scalar]
    history_complete: Bool[Scalar]
    support_fraction: Float64[Scalar]
    rms_length: Float64[Scalar]
    rms_width: Float64[Scalar]
    accepted: Bool[Scalar]


class CSRWake(StrictModule):
    """CSR fields of one density on the plan grid.

    ``potential`` is ``qΦ`` (scale energy) of a reference-charge particle,
    ``radiative_wake`` the part of ``dE/ds`` that is not a total derivative, and
    ``wake = radiative_wake − dΦ/ds`` the instantaneous longitudinal wake
    (``dΦ/ds`` at fixed bunch offset by a symmetric difference). For steady
    models ``potential`` is zero and ``wake = radiative_wake``.
    ``horizontal_force``/``vertical_force`` are the transverse forces (scale
    energy per length; ``+x`` per the accelerator convention), zero for
    one-dimensional models.
    """

    __strict_contract__ = True

    potential: Float64[AnyShape]
    radiative_wake: Float64[AnyShape]
    wake: Float64[AnyShape]
    horizontal_force: Float64[AnyShape]
    vertical_force: Float64[AnyShape]
    evidence: CSREvidence


class CSRKickResult(StrictModule):
    """One CSR kick of a bunch over path length ``length`` at ``position``.

    ``energy_kick`` is each particle's energy change (``−ΔqΦ`` since its
    previous kick plus ``W·length``); ``momentum_kick`` holds the increments of
    ``(px/p₀, py/p₀, δ)``. ``energy_change`` and ``transverse_impulse`` are
    bunch ledgers summed over macroparticle weights. The bunch is kicked only
    when ``evidence.accepted``; the state always records the density.
    """

    __strict_contract__ = True

    bunch: AcceleratorBunch
    state: CSRState
    energy_kick: Float64[_ParticleDim]
    momentum_kick: Float64[_ParticleDim, Literal[3]]
    energy_change: Float64[Scalar]
    transverse_impulse: Float64[Literal[2]]
    density: Float64[AnyShape]
    wake: Float64[AnyShape]
    evidence: CSREvidence


class _Constants(NamedTuple):
    """Host constants of one bunch bound to a plan."""

    charge_constant: float
    particle_charge: float
    energy_factor: float
    momentum: float
    longitudinal_sign: float


class CSRPlan(StrictModule, NonTrainableState):
    """Coherent synchrotron radiation of a bunch on a :class:`CSRLattice`.

    ``model`` selects the physics:

    - ``"1d-steady"``: Saldin–Derbenev steady-state wake of the line density,
      ``W(z) = −(2q/4πε₀)(3R²)^{-1/3} ∫_{−∞}^{z} (z − z′)^{-1/3} λ′(z′) dz′`` in
      bends (zero in drifts), with the ``(z − z′)^{-1/3}`` kernel integrated
      exactly over each cell.
    - ``"1d-transient-shielded"``: the exact one-dimensional retarded
      line-charge model of Mayes and Hoffstaetter (2009) over the lattice,
      including the incoming straight line, entrance and exit transients,
      density history (bunch compression), and, with ``plate_gap``, the
      parallel-plate image series ``2Σ(−1)ⁿ`` over ``image_count`` pairs.
    - ``"3d-steady-igf"``: steady three-dimensional Green functions of a source
      on the circular orbit, integrated over grid cells and convolved with the
      deposited density on the doubled grid of
      :class:`phydrax.operators.FreeSpaceConvolutionPlan`. The energy wake uses
      the longitudinal potential of Cai and Ding (2020) with its Coulomb term;
      the effective horizontal force (with the curvilinear ``−qφ/ρ`` term) and
      the vertical force are closed-form antiderivatives of the exact Lorentz
      force of a reference-velocity particle.
    - ``"3d-retarded-mesh"``: the retarded Liénard–Wiechert potentials of the
      smooth deposited density over its recorded history on the lattice (the
      three-dimensional extension of the 1-D model, sources on the reference
      path displaced by the transverse grid offsets); forces follow from
      ``Ψ = Φ − βcA_s`` as ``F_x = −q(∂_xΨ + hΨ) − q c dA_x/dt`` (the exact
      Lorentz force of the reference-velocity particle plus the ``−qφ/ρ``
      curvilinear term) and ``F_y = −q∂_yΨ``.

    Both three-dimensional routes compute the same steady forces. The
    residual centripetal force of a Gaussian bunch in steady state is
    ``F_x = −2 q λ(z)/(4πε₀ρ)`` for any transverse aspect ratio (Derbenev and
    Shiltsev 1996; Stupakov, PRAB 25, 014401, 2022). Cai and Ding's published
    transverse potentials give ``2 ≤ Λ ≤ 4`` instead because their paraxial
    step ``β_s ≈ β`` moves the test particle at ``β(1+x/ρ)c``; the IGF kernels
    here do not make that substitution.

    Retarded models subtract the same density moving on a straight line (the
    space-charge field, owned by :class:`SpaceChargeIGFPlan`). ``grid`` is a
    uniform cell-centered grid in bunch coordinates relative to the reference
    particle: ``(z,)`` for one-dimensional models, ``(x, y, z)`` otherwise.
    ``smoothing`` is the Gaussian kernel width (length) applied to the
    deposited density per axis. ``reference_rest_energy`` and
    ``reference_momentum`` (``p₀c``) are in the scale energy unit and must match
    the bunch.
    """

    model: CSRModel = eqx.field(static=True)
    lattice: CSRLattice
    scale: ElectromagneticScaleContract = eqx.field(static=True)
    grid: PreparedTensorGrid
    splat: PreparedParticleGridSplat
    shape: tuple[int, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    lower_centers: tuple[float, ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    rest_energy: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    smoothing: tuple[float, ...] = eqx.field(static=True)
    smoothing_taps: tuple[Array | None, ...]
    plate_gap: float | None = eqx.field(static=True)
    image_count: int = eqx.field(static=True)
    shielding_tolerance: float = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    far_nodes: int = eqx.field(static=True)
    lag_nodes: Array
    lag_weights: Array
    line_horizontal: Array
    line_vertical: Array
    line_factors: Array
    line_table: Array
    steady_taps: Array | None
    igf: tuple[FreeSpaceConvolutionPlan, ...]
    igf_curvatures: tuple[float, ...] = eqx.field(static=True)
    kernel_quadrature: int = eqx.field(static=True)
    kernel_defect: float = eqx.field(static=True)
    derbenev_limit: float = eqx.field(static=True)
    root_tolerance: float = eqx.field(static=True)
    resources: CSRResources
    estimate: CSRResourceEstimate = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: CSRModel,
        lattice: CSRLattice,
        scale: ElectromagneticScaleContract,
        grid: PreparedTensorGrid,
        /,
        *,
        reference_rest_energy: float,
        reference_momentum: float,
        capacity: int,
        smoothing: float | tuple[float, ...] = 0.0,
        plate_gap: float | None = None,
        image_count: int = 0,
        shielding_tolerance: float = 1.0e-3,
        history_capacity: int = 256,
        far_nodes: int = 96,
        kernel_quadrature: int = 4,
        derbenev_limit: float = 1.0,
        resources: CSRResources | None = None,
    ) -> None:
        model_ = parse(model, CSRModel, "model")
        if not isinstance(lattice, CSRLattice):
            raise TypeError("lattice must be a CSRLattice.")
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
        resources_ = CSRResources() if resources is None else resources
        if not isinstance(resources_, CSRResources):
            raise TypeError("resources must be CSRResources or None.")
        rest = positive_finite_float(reference_rest_energy, "reference_rest_energy")
        momentum = positive_finite_float(reference_momentum, "reference_momentum")
        capacity_ = positive_integer(capacity, "capacity")
        gamma = math.sqrt(1.0 + (momentum / rest) ** 2)
        beta = momentum / (gamma * rest)
        match model_:
            case "1d-steady" | "1d-transient-shielded":
                dimension = 1
            case "3d-steady-igf" | "3d-retarded-mesh":
                dimension = 3
            case _:
                assert_never(model_)
        centers, spacing = _grid_axes(grid, dimension)
        shape = tuple(center.shape[0] for center in centers)
        if shape[-1] < _NEAR_CELLS + 2:
            raise ValueError(
                f"The longitudinal grid needs at least {_NEAR_CELLS + 2} cells."
            )
        widths = (
            (smoothing,) * dimension
            if isinstance(smoothing, (int, float))
            else tuple(smoothing)
        )
        if len(widths) != dimension or any(
            not math.isfinite(float(width)) or float(width) < 0.0 for width in widths
        ):
            raise ValueError("smoothing must be finite, nonnegative, one width per axis.")
        widths = tuple(float(width) for width in widths)
        taps = tuple(
            _gaussian_taps(width, float(step)) for width, step in zip(widths, spacing)
        )
        gap: float | None = None
        images = 0
        tolerance_ = positive_finite_float(shielding_tolerance, "shielding_tolerance")
        if model_ == "1d-transient-shielded":
            if plate_gap is not None:
                gap = positive_finite_float(plate_gap, "plate_gap")
                images = positive_integer(image_count, "image_count")
            elif image_count != 0:
                raise ValueError("image_count requires plate_gap.")
        elif plate_gap is not None or image_count != 0:
            raise ValueError("Parallel-plate shielding belongs to 1d-transient-shielded.")
        retarded = model_ in ("1d-transient-shielded", "3d-retarded-mesh")
        history = (
            positive_integer(history_capacity, "history_capacity") if retarded else 1
        )
        far = positive_integer(far_nodes, "far_nodes")
        quadrature = positive_integer(kernel_quadrature, "kernel_quadrature")
        if quadrature % 2:
            raise ValueError(
                "kernel_quadrature must be even so no node sits on the singular axis."
            )
        limit = positive_finite_float(derbenev_limit, "derbenev_limit")
        maximum_curvature = lattice.maximum_curvature
        if dimension == 3 and maximum_curvature > 0.0:
            transverse = 0.5 * max(
                float(centers[0][-1] - centers[0][0]) + spacing[0],
                float(centers[1][-1] - centers[1][0]) + spacing[1],
            )
            if transverse * maximum_curvature >= 0.1:
                raise ValueError(
                    "The transverse grid must be small against the bend radius."
                )
        longitudinal = spacing[-1]
        # Geometric nodes reach below the formation scale R/γ³ of the kernel.
        formation = (
            1.0e-3 / (maximum_curvature * gamma**3)
            if maximum_curvature > 0.0
            else longitudinal
        )
        minimum = min(1.0e-4 * longitudinal, formation)
        nodes = _lag_nodes(longitudinal, shape[-1], minimum)
        horizontal = np.zeros((1,), dtype=np.float64)
        vertical = np.zeros((1,), dtype=np.float64)
        factors = np.ones((1,), dtype=np.float64)
        table = np.zeros((1, 1), dtype=np.int32)
        if model_ == "1d-transient-shielded" and gap is not None:
            order = np.arange(1, images + 1, dtype=np.float64)
            horizontal = np.zeros((images + 1,), dtype=np.float64)
            vertical = np.concatenate(([0.0], order * gap))
            # Alternating image pairs 2Σ(−1)ⁿ; the last pair is half-weighted
            # (mean of the last two partial sums), and that half weight is the
            # reported truncation measure.
            factors = np.concatenate(([1.0], 2.0 * (-1.0) ** order))
            factors[-1] *= 0.5
        if model_ == "3d-retarded-mesh":
            nx, ny = shape[0], shape[1]
            dx = spacing[0] * np.arange(-(nx - 1), nx)
            dy = spacing[1] * np.arange(-(ny - 1), ny)
            horizontal = np.repeat(dx, 2 * ny - 1)
            vertical = np.tile(dy, 2 * nx - 1)
            factors = np.full(horizontal.shape, spacing[0] * spacing[1])
            a = np.arange(nx)[:, None, None, None]
            b = np.arange(ny)[None, :, None, None]
            a_ = np.arange(nx)[None, None, :, None]
            b_ = np.arange(ny)[None, None, None, :]
            table = (
                ((a - a_ + nx - 1) * (2 * ny - 1) + (b - b_ + ny - 1))
                .reshape(nx * ny, nx * ny)
                .astype(np.int32)
            )
        steady = None
        if model_ == "1d-steady":
            index = np.arange(shape[0], dtype=np.float64)
            steady = jnp.asarray(
                1.5
                * (
                    ((index + 0.5) * longitudinal) ** (2.0 / 3.0)
                    - (np.maximum(index - 0.5, 0.0) * longitudinal) ** (2.0 / 3.0)
                )
            )
        tolerance = max(1.0e-10, 100.0 * float(np.finfo(np.float64).eps) * gamma * gamma)
        side = nodes.delta.shape[0]
        grid_size = int(np.prod(shape))
        pairs = horizontal.shape[0] * shape[-1] * 2 * side if retarded else 0
        samples = (
            horizontal.shape[0] * shape[-1] * (2 * side + 2 * far) if retarded else 0
        )
        curvatures: tuple[float, ...] = ()
        if model_ == "3d-steady-igf":
            curvatures = tuple(
                sorted(
                    {
                        float(value)
                        for value in np.asarray(lattice.curvatures)
                        if value != 0.0
                    }
                )
            )
        kernel_bytes = (
            len(curvatures) * 8 * grid_size * 3 * 16
            if model_ == "3d-steady-igf"
            else samples * 6 * 8
        )
        estimate = CSRResourceEstimate(
            pairs, (history + 2) * grid_size * 2 * 8, kernel_bytes
        )
        if (
            estimate.retarded_pairs > resources_.maximum_retarded_pairs
            or estimate.history_bytes > resources_.maximum_history_bytes
            or estimate.kernel_bytes > resources_.maximum_kernel_bytes
        ):
            raise CSRResourceError(
                "CSR plan exceeds its declared resources: "
                f"{estimate} against {resources_}."
            )
        igf: tuple[FreeSpaceConvolutionPlan, ...] = ()
        defect = 0.0
        if model_ == "3d-steady-igf":
            plans = []
            for curvature in curvatures:
                values, failures = _steady_table(
                    spacing, (shape[0], shape[1], shape[2]), curvature, gamma, quadrature
                )
                if failures:
                    raise ValueError(
                        "Steady retarded angles did not converge on the kernel table."
                    )
                plans.append(
                    FreeSpaceConvolutionPlan("tabulated", grid, kernel_table=values)
                )
                defect = max(
                    defect,
                    _kernel_quadrature_defect(
                        spacing,
                        (shape[0], shape[1], shape[2]),
                        curvature,
                        gamma,
                        quadrature,
                    ),
                )
            igf = tuple(plans)
        particles = ParticleSetPlan(
            np.arange(capacity_, dtype=np.int64),
            np.ones((capacity_,), dtype=np.float64),
            ambient_dimension=dimension,
            name="csr-macroparticles",
        ).prepare()
        splat = ParticleGridSplatPlan(grid, boundary="drop").prepare(particles)
        self.model = model_
        self.lattice = lattice
        self.scale = scale
        self.grid = grid
        self.splat = splat
        self.shape = shape
        self.spacing = tuple(float(step) for step in spacing)
        self.lower_centers = tuple(float(center[0]) for center in centers)
        self.capacity = capacity_
        self.rest_energy = rest
        self.momentum = momentum
        self.gamma = gamma
        self.beta = beta
        self.smoothing = widths
        self.smoothing_taps = tuple(
            None if tap is None else jnp.asarray(tap) for tap in taps
        )
        self.plate_gap = gap
        self.image_count = images
        self.shielding_tolerance = tolerance_
        self.history_capacity = history
        self.far_nodes = far
        self.lag_nodes = jnp.asarray(nodes.delta)
        self.lag_weights = jnp.asarray(nodes.weights)
        self.line_horizontal = jnp.asarray(horizontal)
        self.line_vertical = jnp.asarray(vertical)
        self.line_factors = jnp.asarray(factors)
        self.line_table = jnp.asarray(table)
        self.steady_taps = steady
        self.igf = igf
        self.igf_curvatures = curvatures
        self.kernel_quadrature = quadrature
        self.kernel_defect = defect
        self.derbenev_limit = limit
        self.root_tolerance = tolerance
        self.resources = resources_
        self.estimate = estimate
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-csr-plan",
                "model": model_,
                "lattice": lattice.lattice_id,
                "scale": scale.scale_id,
                "grid": grid.prepared_id,
                "capacity": capacity_,
                "reference": [rest, momentum],
                "smoothing": list(widths),
                "plate_gap": gap,
                "image_count": images,
                "shielding_tolerance": tolerance_,
                "history_capacity": history,
                "far_nodes": far,
                "kernel_quadrature": quadrature,
                "derbenev_limit": limit,
            }
        )

    def _bind(self, bunch: AcceleratorBunch, /) -> _Constants:
        if not isinstance(bunch, AcceleratorBunch):
            raise TypeError("bunch must be an AcceleratorBunch.")
        if bunch.capacity != self.capacity:
            raise ValueError("bunch capacity must match the CSR plan capacity.")
        convention = bunch.convention
        if convention.momentum_normalization != _CANONICAL_MOMENTUM_NORMALIZATION:
            raise ValueError(
                "CSR kicks require the canonical momentum normalization "
                f"{_CANONICAL_MOMENTUM_NORMALIZATION!r}."
            )
        match convention.longitudinal_sign:
            case "positive-late":
                sign = -1.0
            case "positive-early":
                sign = 1.0
            case _:
                raise ValueError(
                    "CSR needs a 'positive-late' or 'positive-early' longitudinal sign."
                )
        rest = float(bunch.reference_rest_energy)
        momentum = float(bunch.reference_momentum)
        if not (
            math.isclose(rest, self.rest_energy, rel_tol=1.0e-12)
            and math.isclose(momentum, self.momentum, rel_tol=1.0e-12)
        ):
            raise ValueError("The bunch reference energy differs from the CSR plan.")
        charge = float(bunch.reference_charge) * float(self.scale.elementary_charge)
        energy = self.gamma * self.rest_energy
        return _Constants(
            charge / (4.0 * math.pi * float(self.scale.vacuum_permittivity)),
            charge,
            energy / (self.momentum * self.momentum),
            self.momentum,
            sign,
        )

    def _smooth(self, density: Array, /) -> Array:
        for axis, taps in enumerate(self.smoothing_taps):
            if taps is not None:
                density = convolve(density, taps, axis=axis, mode="same")
        return density

    def _positions(self, coordinates: Array, sign: float, /) -> Array:
        longitudinal = sign * coordinates[:, 4]
        if len(self.shape) == 1:
            return longitudinal[:, None]
        return jnp.stack((coordinates[:, 0], coordinates[:, 2], longitudinal), axis=-1)

    def _fields(self, timeline: _Timeline, position: Array, /) -> _Fields:
        density = timeline.densities[-1]
        zero = jnp.zeros(self.shape)
        none = jnp.asarray(0, dtype=jnp.int32)
        complete = jnp.asarray(False)
        truncation = jnp.asarray(0.0)
        curvature = self.lattice.curvature(position)
        match self.model:
            case "1d-steady":
                if self.steady_taps is None:
                    raise AssertionError("1d-steady lost its kernel taps.")
                slope = timeline.slopes[-1]
                convolved = convolve(slope, self.steady_taps, mode="full")[
                    : self.shape[0]
                ]
                wake = (
                    -2.0
                    * jnp.abs(curvature) ** (2.0 / 3.0)
                    / 3.0 ** (1.0 / 3.0)
                    * convolved
                )
                return _Fields(zero, wake, zero, zero, zero, none, complete, truncation)
            case "1d-transient-shielded":
                return self._line_fields(timeline, position)
            case "3d-steady-igf":
                return self._igf_fields(density, curvature)
            case "3d-retarded-mesh":
                return self._mesh_fields(timeline, position, curvature)
            case _:
                assert_never(self.model)

    def _samples(
        self, position: Array, horizontal: Array, vertical: Array, *, subtract: bool
    ) -> _LineSamples:
        count = self.shape[-1]
        spacing = self.spacing[-1]
        longitudinal = jnp.asarray(self.lower_centers[-1] + spacing * np.arange(count))
        # Off-axis image lines carry no singularity: grid offsets resolve the
        # density and the sinh/far lags resolve the geometry.
        nodes = (
            _LagNodes(self.lag_nodes, self.lag_weights)
            if subtract
            else _LagNodes(
                jnp.asarray(spacing * np.arange(1, count, dtype=np.float64)),
                jnp.full((count - 1,), spacing),
            )
        )
        return _line_samples(
            self.lattice,
            position,
            self.beta,
            self.gamma,
            longitudinal,
            horizontal,
            vertical,
            nodes,
            self.far_nodes,
            subtract=subtract,
            tolerance=self.root_tolerance,
        )

    def _line_fields(self, timeline: _Timeline, position: Array, /) -> _Fields:
        columns = _Timeline(
            timeline.positions,
            timeline.densities[:, None, :],
            timeline.slopes[:, None, :],
            timeline.dropped,
            timeline.oldest_retained,
        )
        groups = [
            (
                self._samples(
                    position,
                    self.line_horizontal[:1],
                    self.line_vertical[:1],
                    subtract=True,
                ),
                self.line_factors[:1],
            )
        ]
        if self.image_count:
            groups.append(
                (
                    self._samples(
                        position,
                        self.line_horizontal[1:],
                        self.line_vertical[1:],
                        subtract=False,
                    ),
                    self.line_factors[1:],
                )
            )
        potentials = []
        wakes = []
        failures = jnp.asarray(0, dtype=jnp.int32)
        incomplete = jnp.asarray(False)
        for samples, factors in groups:
            sampled = _sample_timeline(
                columns,
                jnp.zeros(samples.location.shape, dtype=jnp.int32),
                samples.location,
                samples.retarded,
                self.lower_centers[-1],
                self.spacing[-1],
            )
            potentials.append(
                factors[:, None] * jnp.sum(samples.weights[0] * sampled.values, axis=-1)
            )
            wakes.append(
                factors[:, None]
                * jnp.sum(samples.weights[1] * sampled.derivatives, axis=-1)
            )
            failures = failures + samples.failures
            incomplete = incomplete | sampled.incomplete
        potential_lines = jnp.concatenate(potentials, axis=0)
        wake_lines = jnp.concatenate(wakes, axis=0)
        potential = jnp.sum(potential_lines, axis=0)
        wake = jnp.sum(wake_lines, axis=0)
        tiny = jnp.finfo(jnp.float64).tiny
        truncation = (
            jnp.maximum(
                jnp.max(jnp.abs(wake_lines[-1]))
                / jnp.maximum(jnp.max(jnp.abs(wake)), tiny),
                jnp.max(jnp.abs(potential_lines[-1]))
                / jnp.maximum(jnp.max(jnp.abs(potential)), tiny),
            )
            if self.image_count
            else jnp.asarray(0.0)
        )
        zero = jnp.zeros(self.shape)
        return _Fields(
            potential, wake, zero, zero, zero, failures, incomplete, truncation
        )

    def _igf_fields(self, density: Array, curvature: Array, /) -> _Fields:
        def drift(values: Array) -> Array:
            return jnp.zeros(self.shape + (3,))

        def bend(plan: FreeSpaceConvolutionPlan) -> Callable[[Array], Array]:
            def convolved(values: Array) -> Array:
                return plan.convolve(values).field

            return convolved

        index = jnp.asarray(0, dtype=jnp.int32)
        for position, value in enumerate(self.igf_curvatures):
            index = jnp.where(curvature == value, position + 1, index)
        field = jax.lax.switch(
            index, [drift, *(bend(plan) for plan in self.igf)], density
        )
        factor = self.beta * self.beta * jnp.abs(curvature)
        zero = jnp.zeros(self.shape)
        return _Fields(
            zero,
            factor * field[..., 0],
            factor * field[..., 1],
            factor * field[..., 2],
            zero,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(0.0),
        )

    def _mesh_fields(
        self, timeline: _Timeline, position: Array, curvature: Array, /
    ) -> _Fields:
        nx, ny, nz = self.shape
        columns = nx * ny
        flat = _Timeline(
            timeline.positions,
            timeline.densities.reshape((-1, columns, nz)),
            timeline.slopes.reshape((-1, columns, nz)),
            timeline.dropped,
            timeline.oldest_retained,
        )
        samples = self._samples(
            position, self.line_horizontal, self.line_vertical, subtract=True
        )
        area = self.spacing[0] * self.spacing[1]
        source_columns = jnp.broadcast_to(
            jnp.arange(columns, dtype=jnp.int32)[:, None, None],
            (columns,) + samples.location.shape[1:],
        )

        def column(observer: Array) -> tuple[Array, Array]:
            lines = self.line_table[observer]
            sampled = _sample_timeline(
                flat,
                source_columns,
                samples.location[lines],
                samples.retarded[lines],
                self.lower_centers[-1],
                self.spacing[-1],
            )
            weights = samples.weights[:, lines]
            density_channels = jnp.sum(
                weights[jnp.asarray([0, 2, 3])] * sampled.values[None], axis=(1, 3)
            )
            wake = jnp.sum(weights[1] * sampled.derivatives, axis=(0, 2))
            return (
                area * jnp.concatenate((density_channels, wake[None]), axis=0),
                sampled.incomplete,
            )

        channels, incomplete = jax.lax.map(
            column,
            jnp.arange(columns, dtype=jnp.int32),
            batch_size=min(self.resources.observer_chunk, columns),
        )
        grids = jnp.moveaxis(channels, 0, 1).reshape((4, nx, ny, nz))
        potential, transverse, vector_x, wake = grids
        horizontal = -(
            _axis_gradient(transverse, self.spacing[0], 0) + curvature * transverse
        )
        vertical = -_axis_gradient(transverse, self.spacing[1], 1)
        return _Fields(
            potential,
            wake,
            horizontal,
            vertical,
            vector_x,
            samples.failures,
            jnp.any(incomplete),
            jnp.asarray(0.0),
        )

    def _evidence(
        self,
        fields: _Fields,
        length: Array,
        width: Array,
        support_fraction: Array,
        position: Array,
        finite: Array,
        /,
    ) -> CSREvidence:
        curvature = jnp.abs(self.lattice.curvature(position))
        bend = curvature > 0.0
        radius = 1.0 / jnp.where(bend, curvature, 1.0)
        overtaking = jnp.where(
            bend, (24.0 * length * radius * radius) ** (1.0 / 3.0), jnp.inf
        )
        entry = self.lattice.bend_entry_distance(position)
        one_dimensional = len(self.shape) == 1
        derbenev = (
            jnp.where(
                bend,
                width
                / jnp.maximum(length * length * radius, jnp.finfo(jnp.float64).tiny)
                ** (1.0 / 3.0),
                0.0,
            )
            if one_dimensional
            else jnp.asarray(0.0)
        )
        steady = self.model in ("1d-steady", "3d-steady-igf")
        status = (
            jnp.where(finite, 0, int(CSRStatus.NONFINITE))
            | jnp.where(support_fraction >= 1.0 - 1.0e-12, 0, int(CSRStatus.UNSUPPORTED))
            | jnp.where(fields.failures == 0, 0, int(CSRStatus.ROOT_FAILURE))
            | jnp.where(fields.incomplete, int(CSRStatus.HISTORY_INCOMPLETE), 0)
            | jnp.where(
                fields.truncation <= self.shielding_tolerance,
                0,
                int(CSRStatus.SHIELDING_TRUNCATED),
            )
            | jnp.where(
                derbenev >= self.derbenev_limit, int(CSRStatus.DERBENEV_VIOLATED), 0
            )
            | jnp.where(
                steady & bend & (entry < overtaking), int(CSRStatus.NOT_STEADY), 0
            )
        ).astype(jnp.int32)
        return CSREvidence(
            status,
            derbenev,
            overtaking,
            entry,
            fields.truncation,
            fields.failures,
            ~fields.incomplete,
            support_fraction,
            length,
            width,
            (status & _REFUSAL_BITS) == 0,
        )

    def _deposit(
        self, coordinates: Array, weights: Array, active: Array, constants: _Constants, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        positions = self._positions(coordinates, constants.longitudinal_sign)
        splat_state = self.splat.build(positions, active_mask=active)
        charges = jnp.where(active, weights * constants.particle_charge, 0.0)
        deposit = self.splat.deposit_content(splat_state, charges)
        density = self._smooth(deposit.density)
        slope = _axis_gradient(density, self.spacing[-1], -1)
        supported = splat_state.supported_mask & ~splat_state.truncated_support_mask
        fraction = jnp.sum(active & supported) / jnp.maximum(jnp.sum(active), 1)
        return density, slope, positions, fraction, deposit.successful

    def _gather(self, positions: Array, active: Array, grids: Array, /) -> Array:
        splat_state = self.splat.build(positions, active_mask=active)
        return self.splat.gather(splat_state, grids).values

    def _kick_arrays(
        self,
        coordinates: Array,
        weights: Array,
        active: Array,
        state: CSRState,
        position: Array,
        length: Array,
        constants: _Constants,
        /,
    ) -> tuple[Array, CSRState, Array, Array, Array, Array, Array, Array, CSREvidence]:
        density, slope, positions, fraction, deposited = self._deposit(
            coordinates, weights, active, constants
        )
        timeline = state._timeline(density, slope, position)
        fields = _evaluate_fields(self, timeline, position)
        c = constants.charge_constant
        grids = jnp.stack(
            (
                fields.potential,
                fields.wake,
                fields.horizontal,
                fields.vertical,
                fields.vector_x,
            ),
            axis=-1,
        )
        sampled = c * self._gather(positions, active, grids)
        potential, wake, horizontal, vertical, vector_x = (
            sampled[:, index] for index in range(5)
        )
        energy = -(potential - state.potential) + wake * length
        transverse = jnp.stack(
            (
                horizontal * length / self.beta - (vector_x - state.vector_potential),
                vertical * length / self.beta,
            ),
            axis=-1,
        )
        energy = jnp.where(active, energy, 0.0)
        transverse = jnp.where(active[:, None], transverse, 0.0)
        increments = jnp.concatenate(
            (
                transverse / constants.momentum,
                (energy * constants.energy_factor)[:, None],
            ),
            axis=-1,
        )
        finite = (
            deposited & jnp.all(jnp.isfinite(grids)) & jnp.all(jnp.isfinite(increments))
        )
        length_, width = _moments(
            constants.longitudinal_sign * coordinates[:, 4],
            coordinates[:, 0],
            jnp.where(active, jnp.abs(weights), 0.0),
        )
        evidence = self._evidence(fields, length_, width, fraction, position, finite)
        candidate = coordinates.at[:, 1].add(increments[:, 0])
        candidate = candidate.at[:, 3].add(increments[:, 1])
        candidate = candidate.at[:, 5].add(increments[:, 2])
        kicked = jnp.where(evidence.accepted, candidate, coordinates)
        recorded = state.record(density, slope, position)
        next_state = CSRState(
            recorded.densities,
            recorded.slopes,
            recorded.positions,
            recorded.initial_density,
            recorded.initial_slope,
            recorded.initial_position,
            recorded.cursor,
            recorded.count,
            jnp.where(active, potential, state.potential),
            jnp.where(active, vector_x, state.vector_potential),
        )
        return (
            kicked,
            next_state,
            energy,
            increments,
            jnp.sum(weights * energy),
            jnp.sum(weights[:, None] * transverse, axis=0),
            density,
            c * fields.wake,
            evidence,
        )

    def initial_state(
        self, bunch: AcceleratorBunch, /, *, position: ArrayLike
    ) -> CSRState:
        """History seeded with the bunch at ``position`` and its potential memory.

        The seeded snapshot also stands for the bunch on the straight line before
        ``position`` (the incoming-drift model of retarded times before tracking).
        """
        constants = self._bind(bunch)
        position_ = jnp.asarray(position, dtype=jnp.float64).reshape(())
        active = bunch.active & bunch.valid
        density, slope, positions, _, _ = self._deposit(
            bunch.coordinates, bunch.weights, active, constants
        )
        potential = jnp.zeros((self.capacity,))
        vector = jnp.zeros((self.capacity,))
        if self.model in ("1d-transient-shielded", "3d-retarded-mesh"):
            fields = _evaluate_fields(
                self, _frozen_timeline(density, slope, position_), position_
            )
            sampled = constants.charge_constant * self._gather(
                positions, active, jnp.stack((fields.potential, fields.vector_x), axis=-1)
            )
            potential = jnp.where(active, sampled[:, 0], 0.0)
            vector = jnp.where(active, sampled[:, 1], 0.0)
        capacity = self.history_capacity
        return CSRState(
            jnp.zeros((capacity,) + self.shape),
            jnp.zeros((capacity,) + self.shape),
            jnp.zeros((capacity,)),
            density,
            slope,
            position_,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            potential,
            vector,
        )

    def kick(
        self,
        bunch: AcceleratorBunch,
        state: CSRState,
        /,
        *,
        position: ArrayLike,
        length: ArrayLike,
    ) -> CSRKickResult:
        """Kick ``bunch`` with CSR over ``length`` centred at path ``position``."""
        constants = self._bind(bunch)
        if not isinstance(state, CSRState):
            raise TypeError("state must be a CSRState.")
        if state.capacity != self.history_capacity or state.potential.shape != (
            self.capacity,
        ):
            raise ValueError("state does not belong to this CSR plan.")
        position_ = jnp.asarray(position, dtype=jnp.float64).reshape(())
        length_ = jnp.asarray(length, dtype=jnp.float64).reshape(())
        (
            coordinates,
            next_state,
            energy,
            increments,
            energy_change,
            impulse,
            density,
            wake,
            evidence,
        ) = self._kick_arrays(
            bunch.coordinates.astype(jnp.float64),
            bunch.weights.astype(jnp.float64),
            bunch.active & bunch.valid,
            state,
            position_,
            length_,
            constants,
        )
        result_bunch = AcceleratorBunch(
            coordinates.astype(bunch.coordinates.dtype),
            bunch.weights,
            bunch.particle_ids,
            active=bunch.active,
            reference_rest_energy=float(bunch.reference_rest_energy),
            reference_momentum=float(bunch.reference_momentum),
            reference_charge=float(bunch.reference_charge),
            convention=bunch.convention,
            bunch_id=f"{bunch.bunch_id}:{self.plan_id}",
        )
        return CSRKickResult(
            result_bunch,
            next_state,
            energy,
            increments,
            energy_change,
            impulse,
            density,
            wake,
            evidence,
        )

    def wake(
        self,
        density: ArrayLike,
        /,
        *,
        position: ArrayLike,
        reference_charge: float,
        state: CSRState | None = None,
        derivative_step: float | None = None,
    ) -> CSRWake:
        """Fields of a given charge density (C/m or C/m³ on ``grid``) at ``position``.

        Without ``state`` the density is taken as frozen over its whole history.
        Retarded models need ``derivative_step`` (path length) for the symmetric
        difference of the potentials along a reference-speed particle: at fixed
        bunch offset plus, off axis in a bend, the path-length slip
        ``dz/ds = −h x`` of a particle at horizontal offset ``x``.
        """
        density_ = jnp.asarray(density, dtype=jnp.float64)
        if density_.shape != self.shape:
            raise ValueError(f"density must have the grid shape {self.shape}.")
        position_ = jnp.asarray(position, dtype=jnp.float64).reshape(())
        charge = float(reference_charge) * float(self.scale.elementary_charge)
        c = charge / (4.0 * math.pi * float(self.scale.vacuum_permittivity))
        slope = _axis_gradient(density_, self.spacing[-1], -1)
        retarded = self.model in ("1d-transient-shielded", "3d-retarded-mesh")
        if retarded != (derivative_step is not None):
            raise ValueError(
                "derivative_step is required by retarded models and refused by steady ones."
            )
        if state is not None and not isinstance(state, CSRState):
            raise TypeError("state must be a CSRState or None.")

        def timeline(at: Array) -> _Timeline:
            if state is None:
                return _frozen_timeline(density_, slope, at)
            return state._timeline(density_, slope, at)

        fields = _evaluate_fields(self, timeline(position_), position_)
        wake = fields.wake
        horizontal = fields.horizontal
        failures = fields.failures
        incomplete = fields.incomplete
        if derivative_step is not None:
            step = positive_finite_float(derivative_step, "derivative_step")
            ahead = _evaluate_fields(self, timeline(position_ + step), position_ + step)
            behind = _evaluate_fields(self, timeline(position_ - step), position_ - step)
            wake = wake - (ahead.potential - behind.potential) / (2.0 * step)
            horizontal = horizontal - self.beta * (ahead.vector_x - behind.vector_x) / (
                2.0 * step
            )
            failures = failures + ahead.failures + behind.failures
            incomplete = incomplete | ahead.incomplete | behind.incomplete
            if len(self.shape) == 3:
                horizontal_offsets = self.grid.points.reshape(self.shape + (3,))[..., 0]
                slip = self.lattice.curvature(position_) * horizontal_offsets
                wake = wake + slip * _axis_gradient(
                    fields.potential, self.spacing[-1], -1
                )
                horizontal = horizontal + self.beta * slip * _axis_gradient(
                    fields.vector_x, self.spacing[-1], -1
                )
        finite = jnp.all(jnp.isfinite(wake)) & jnp.all(jnp.isfinite(horizontal))
        points = self.grid.points.reshape((-1, len(self.shape)))
        length, width = _moments(
            points[:, -1],
            points[:, 0] if len(self.shape) == 3 else jnp.zeros(points.shape[:1]),
            jnp.abs(density_).reshape(-1),
        )
        evidence = self._evidence(
            _Fields(
                fields.potential,
                wake,
                horizontal,
                fields.vertical,
                fields.vector_x,
                failures,
                incomplete,
                fields.truncation,
            ),
            length,
            width,
            jnp.asarray(1.0),
            position_,
            finite,
        )
        return CSRWake(
            c * fields.potential,
            c * fields.wake,
            c * wake,
            c * horizontal,
            c * fields.vertical,
            evidence,
        )


@eqx.filter_jit
def _evaluate_fields(plan: CSRPlan, timeline: _Timeline, position: Array, /) -> _Fields:
    """Compiled field evaluation with stable identity.

    The retarded solves are bracketed ``while`` loops under ``vmap``; compiling
    them once per plan structure keeps eager kicks and wakes from retracing.
    """
    return plan._fields(timeline, position)


def _moments(
    longitudinal: Array, horizontal: Array, weights: Array
) -> tuple[Array, Array]:
    """Weighted rms bunch length and horizontal width."""
    total = jnp.maximum(jnp.sum(weights), jnp.finfo(jnp.float64).tiny)
    mean_z = jnp.sum(weights * longitudinal) / total
    mean_x = jnp.sum(weights * horizontal) / total
    return (
        jnp.sqrt(jnp.sum(weights * (longitudinal - mean_z) ** 2) / total),
        jnp.sqrt(jnp.sum(weights * (horizontal - mean_x) ** 2) / total),
    )


def _axis_gradient(values: Array, spacing: ArrayLike, axis: int) -> Array:
    """Second-order centered difference along one axis (one-sided at the ends)."""
    gradient = jnp.gradient(values, spacing, axis=axis)
    if isinstance(gradient, list):
        raise AssertionError("jnp.gradient returned per-axis gradients for one axis.")
    return gradient


def _kernel_quadrature_defect(
    spacing: np.ndarray,
    shape: tuple[int, int, int],
    curvature: float,
    gamma: float,
    quadrature: int,
) -> float:
    """Largest change of the near-origin kernel block under doubled quadrature."""
    offsets = tuple(
        spacing[axis] * np.arange(-min(2, count - 1), min(2, count - 1) + 1)
        for axis, count in enumerate(shape)
    )
    coarse, _ = _steady_table(spacing, shape, curvature, gamma, quadrature, offsets)
    fine, _ = _steady_table(spacing, shape, curvature, gamma, 2 * quadrature, offsets)
    scale = np.max(np.abs(fine), axis=(0, 1, 2))
    return float(
        np.max(
            np.max(np.abs(coarse - fine), axis=(0, 1, 2))
            / np.maximum(scale, np.finfo(np.float64).tiny)
        )
    )


def _transport(
    coordinates: Array, curvature: Array, length: Array, gamma: float, sign: float
) -> Array:
    """First-order hard-edge sector-bend (or drift) map over ``length``.

    Horizontal: ``x″ = −h²x + hδ``; vertical drift; the path-length excess
    ``h∫x ds − lδ/γ²`` delays the particle (``ζ`` grows for ``"positive-late"``).
    """
    x, px, y, py, zeta, delta = (coordinates[:, index] for index in range(6))
    angle = curvature * length
    sine = jnp.sin(angle)
    cosine = jnp.cos(angle)
    focal = length * jnp.sinc(angle / jnp.pi)
    dispersion = 0.5 * curvature * length * length * jnp.sinc(0.5 * angle / jnp.pi) ** 2
    small = jnp.abs(angle) < 1.0e-2
    safe = jnp.where(small, 1.0, curvature)
    squared = angle * angle
    excess = jnp.where(
        small,
        length * squared / 6.0 * (1.0 - squared / 20.0 + squared * squared / 840.0),
        (angle - sine) / safe,
    )
    path = sine * x + dispersion * px + excess * delta - length * delta / (gamma * gamma)
    return jnp.stack(
        (
            cosine * x + focal * px + dispersion * delta,
            -curvature * sine * x + cosine * px + sine * delta,
            y + length * py,
            py,
            zeta - sign * path,
            delta,
        ),
        axis=-1,
    )


def _edge(coordinates: Array, curvature: Array, angle: Array) -> Array:
    """Thin hard-edge pole-face kick ``Δpx = h tan(e) x``, ``Δpy = −h tan(e) y``."""
    strength = curvature * jnp.tan(angle)
    kicked = coordinates.at[:, 1].add(strength * coordinates[:, 0])
    return kicked.at[:, 3].add(-strength * coordinates[:, 2])


class CSRTrackingPlan(StrictModule, NonTrainableState):
    """Split-step tracking through a :class:`CSRPlan` lattice with CSR kicks.

    Each element is split into ``substeps`` equal steps; a step applies the
    entrance pole-face kick (first step of a bend), half the first-order
    transfer map, one CSR kick over the step length at its midpoint, the other
    half map, and the exit pole-face kick (last step).
    """

    plan: CSRPlan
    step_lengths: Array
    step_positions: Array
    step_curvatures: Array
    entrance_angles: Array
    exit_angles: Array
    substeps: tuple[int, ...] = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, plan: CSRPlan, /, *, substeps: int | Iterable[int]) -> None:
        if not isinstance(plan, CSRPlan):
            raise TypeError("plan must be a CSRPlan.")
        lattice = plan.lattice
        count = lattice.element_count
        splits = (
            (positive_integer(substeps, "substeps"),) * count
            if isinstance(substeps, int)
            else tuple(positive_integer(value, "substeps") for value in substeps)
        )
        if len(splits) != count:
            raise ValueError("substeps must give one count per lattice element.")
        lengths = np.asarray(lattice.lengths)
        curvatures = np.asarray(lattice.curvatures)
        entrance = np.asarray(lattice.entrance_edges)
        exit_ = np.asarray(lattice.exit_edges)
        starts = np.concatenate(([0.0], np.cumsum(lengths)))
        step_lengths = np.concatenate(
            [
                np.full((split,), lengths[index] / split)
                for index, split in enumerate(splits)
            ]
        )
        step_positions = np.concatenate(
            [
                starts[index] + (np.arange(split) + 0.5) * lengths[index] / split
                for index, split in enumerate(splits)
            ]
        )
        step_curvatures = np.concatenate(
            [np.full((split,), curvatures[index]) for index, split in enumerate(splits)]
        )
        first = np.concatenate([np.arange(split) == 0 for split in splits])
        last = np.concatenate([np.arange(split) == split - 1 for split in splits])
        element = np.concatenate(
            [np.full((split,), index) for index, split in enumerate(splits)]
        )
        self.plan = plan
        self.step_lengths = jnp.asarray(step_lengths)
        self.step_positions = jnp.asarray(step_positions)
        self.step_curvatures = jnp.asarray(step_curvatures)
        self.entrance_angles = jnp.asarray(np.where(first, entrance[element], 0.0))
        self.exit_angles = jnp.asarray(np.where(last, exit_[element], 0.0))
        self.substeps = splits
        self.step_count = int(step_lengths.shape[0])
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-csr-tracking-plan",
                "plan": plan.plan_id,
                "substeps": list(splits),
            }
        )


class CSRTrackingResult(StrictModule):
    """Tracked bunch, final CSR state, and per-step ledgers and evidence.

    Per step: kick ``positions``, bunch ``energy_change`` (scale energy),
    ``transverse_impulse`` (``Σ Δp⊥c``), :class:`CSRStatus` bits, acceptance,
    and the weighted mean and rms of ``δ`` after the step. ``accepted`` holds
    when every kick was accepted; a refused kick leaves that step unkicked.
    """

    __strict_contract__ = True

    bunch: AcceleratorBunch
    state: CSRState
    positions: Float64[_StepDim]
    energy_change: Float64[_StepDim]
    transverse_impulse: Float64[_StepDim, Literal[2]]
    status: Int32[_StepDim]
    step_accepted: Bool[_StepDim]
    mean_delta: Float64[_StepDim]
    rms_delta: Float64[_StepDim]
    accepted: Bool[Scalar]
    plan_id: Identifier = eqx.field(static=True)


def track_csr(plan: CSRTrackingPlan, bunch: AcceleratorBunch, /) -> CSRTrackingResult:
    """Track ``bunch`` from the lattice entrance to its exit with CSR as a collective process."""
    if not isinstance(plan, CSRTrackingPlan):
        raise TypeError("plan must be a CSRTrackingPlan.")
    csr = plan.plan
    constants = csr._bind(bunch)
    state = csr.initial_state(bunch, position=0.0)
    active = bunch.active & bunch.valid
    weights = bunch.weights.astype(jnp.float64)
    total = jnp.maximum(
        jnp.sum(jnp.where(active, weights, 0.0)), jnp.finfo(jnp.float64).tiny
    )
    gamma = csr.gamma
    sign = constants.longitudinal_sign

    def step(
        carry: tuple[Array, CSRState],
        data: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[tuple[Array, CSRState], tuple[Array, Array, Array, Array, Array, Array]]:
        coordinates, current = carry
        length, position, curvature, entrance, exit_ = data
        coordinates = _edge(coordinates, curvature, entrance)
        coordinates = _transport(coordinates, curvature, 0.5 * length, gamma, sign)
        (
            coordinates,
            current,
            _,
            _,
            energy_change,
            impulse,
            _,
            _,
            evidence,
        ) = csr._kick_arrays(
            coordinates, weights, active, current, position, length, constants
        )
        coordinates = _transport(coordinates, curvature, 0.5 * length, gamma, sign)
        coordinates = _edge(coordinates, curvature, exit_)
        delta = coordinates[:, 5]
        mean = jnp.sum(jnp.where(active, weights * delta, 0.0)) / total
        spread = jnp.sqrt(
            jnp.sum(jnp.where(active, weights * (delta - mean) ** 2, 0.0)) / total
        )
        return (coordinates, current), (
            energy_change,
            impulse,
            evidence.status,
            evidence.accepted,
            mean,
            spread,
        )

    (coordinates, final_state), history = jax.lax.scan(
        step,
        (
            jnp.where(
                active[:, None], bunch.coordinates.astype(jnp.float64), bunch.coordinates
            ),
            state,
        ),
        (
            plan.step_lengths,
            plan.step_positions,
            plan.step_curvatures,
            plan.entrance_angles,
            plan.exit_angles,
        ),
    )
    energy_change, impulse, status, accepted, mean, spread = history
    coordinates = jnp.where(active[:, None], coordinates, bunch.coordinates)
    result_bunch = AcceleratorBunch(
        coordinates.astype(bunch.coordinates.dtype),
        bunch.weights,
        bunch.particle_ids,
        active=bunch.active,
        reference_rest_energy=float(bunch.reference_rest_energy),
        reference_momentum=float(bunch.reference_momentum),
        reference_charge=float(bunch.reference_charge),
        convention=bunch.convention,
        bunch_id=f"{bunch.bunch_id}:{plan.plan_id}",
    )
    return CSRTrackingResult(
        result_bunch,
        final_state,
        plan.step_positions,
        energy_change,
        impulse,
        status,
        accepted,
        mean,
        spread,
        jnp.all(accepted),
        plan.plan_id,
    )


__all__ = [
    "CSREvidence",
    "CSRKickResult",
    "CSRLattice",
    "CSRModel",
    "CSRPlan",
    "CSRResourceError",
    "CSRResourceEstimate",
    "CSRResources",
    "CSRState",
    "CSRStatus",
    "CSRTrackingPlan",
    "CSRTrackingResult",
    "CSRWake",
    "track_csr",
]
