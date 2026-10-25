#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Term-owned reciprocal-space bindings for long-range atomistic methods.

Two prepared owners are provided, both instantiated per energy term (there is
no global cache and no FFT implementation here):

* :class:`PreparedReciprocalMesh` binds the native periodic tensor B-spline
  particle-grid transfer to the native real FFT on one fractional row-cell
  mesh. It owns the half-spectrum integer modes, their Hermitian weights, the
  declared assignment modulus (the exact discrete B-spline symbol or the
  continuous ``sinc`` window), the declared Nyquist convention, bounded channel
  execution and resource evidence.
* :class:`PreparedReciprocalModeSet` owns an explicit canonical half-space of
  integer reciprocal modes and evaluates channel structure factors in bounded
  mode chunks, never a full mode-by-source phase matrix.

Both evaluate the quadratic form

``Q = sum_c w_c sum_{k in support} h(k) G_c(k) |S_c(k)|^2``

where ``S_c(k) = sum_i v_ic exp(i k . r_i)`` (approximated on the mesh by the
deconvolved transform of the deposited channel), ``h`` the Hermitian
multiplicity, ``G_c`` the method's influence and ``w_c`` optional signed
channel weights. Influences are callables ``G(wavevectors, squared) -> values``
evaluated on the current-cell geometry (shape ``(...)`` or ``(C, ...)``) and
must be finite at every stored mode; the ``k = 0`` value belongs to the
scientific method and every call states ``zero_mode`` explicitly. Wavevectors
use the current row cell ``H`` with ``k = 2 pi m @ inv(H).T`` from the native
lattice solve, so coefficients follow the runtime cell and stay
differentiable in it. ``support_radius`` is the largest ``|k|`` below which
every mode is unambiguously represented under that cell.

Mesh Nyquist convention (``nyquist``), part of the plan identity:

* ``"symmetrized"`` keeps the complete real-FFT spectrum. On an even axis the
  index ``K/2`` represents both ``m = +-K/2``; for a triclinic cell these give
  different ``|k|``, so the influence there is the average over every sign
  choice of the Nyquist components. The result is conjugation-consistent, so
  the Hermitian half-spectrum sum equals the full-spectrum sum exactly.
* ``"excluded"`` removes Nyquist planes of even axes (a conjugation-closed
  truncation, not the unfiltered PME spectrum); the number of removed modes
  is reported in :class:`ReciprocalResourceEvidence`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._model import register_artifact_value
from .._model._fixed_binding import require_identical
from .._spectral._nonuniform_fourier import _smooth_even_size
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    cardinal_bspline_periodic_symbol,
    lattice_measure,
    lattice_right_inverse_with_status,
    ParticleDiscretization,
    ParticleGridSplatBudget,
    ParticleGridSplatPlan,
    ParticleGridSplatState,
    PeriodicCell,
    PreparedParticleGridSplat,
    TensorBSplineSplatAssignment,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from ..typing import parse


ReciprocalMeshModulus: TypeAlias = Literal[
    "discrete-bspline-modulus", "continuous-sinc-window"
]
ReciprocalMeshFrame: TypeAlias = Literal["cell-origin", "anchor-source"]
ReciprocalNyquistPolicy: TypeAlias = Literal["symmetrized", "excluded"]
ZeroModePolicy: TypeAlias = Literal["exclude", "include"]
ReciprocalInfluence: TypeAlias = Callable[[Array, Array], Array]
ReciprocalBindingScope: TypeAlias = Literal["concrete-cell", "lattice-family"]

_TWO_PI = 2.0 * math.pi
_COMPLEX_ITEMSIZE = 16
_REAL_ITEMSIZE = 8


def fft_friendly_mesh_size(minimum: float, /) -> int:
    """Smallest even 2-3-5-smooth mesh size of at least ``minimum`` (host sizing)."""
    value = float(minimum)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Mesh size lower bound must be finite and positive.")
    return _smooth_even_size(value)


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _axis_triple(values: Sequence[int], name: str, /) -> tuple[int, int, int]:
    items = tuple(_positive_int(value, name) for value in values)
    if len(items) != 3:
        raise ValueError(f"{name} must name three lattice axes.")
    return (items[0], items[1], items[2])


def _require_cell(cell: PeriodicCell, /) -> None:
    if not isinstance(cell, PeriodicCell):
        raise TypeError("cell must be a PeriodicCell.")
    if cell.rank != 3 or cell.ambient_dimension != 3:
        raise ValueError(
            "Reciprocal bindings require a full rank-3 row cell in three dimensions."
        )


class ReciprocalResourceEvidence(StrictModule, NonTrainableState):
    """Exact static resources and mode accounting of one prepared binding.

    ``stored_modes`` is the stored (half-spectrum or half-space) mode count,
    ``supported_modes`` those with nonzero Hermitian weight,
    ``excluded_modes`` stored modes removed by the declared Nyquist convention
    and ``symmetrized_modes`` those whose influence is a Nyquist sign average.
    """

    kind: str = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    channel_chunk: int = eqx.field(static=True)
    stored_modes: int = eqx.field(static=True)
    supported_modes: int = eqx.field(static=True)
    excluded_modes: int = eqx.field(static=True)
    symmetrized_modes: int = eqx.field(static=True)
    grid_points: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    chunk_workspace_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        /,
        *,
        source_capacity: int,
        channel_chunk: int,
        stored_modes: int,
        supported_modes: int,
        excluded_modes: int,
        symmetrized_modes: int,
        grid_points: int,
        route_count: int,
        chunk_workspace_bytes: int,
    ) -> None:
        values = {
            "source_capacity": int(source_capacity),
            "channel_chunk": int(channel_chunk),
            "stored_modes": int(stored_modes),
            "supported_modes": int(supported_modes),
            "excluded_modes": int(excluded_modes),
            "symmetrized_modes": int(symmetrized_modes),
            "grid_points": int(grid_points),
            "route_count": int(route_count),
            "chunk_workspace_bytes": int(chunk_workspace_bytes),
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("Reciprocal resource counts must be non-negative.")
        self.kind = str(kind)
        for name, value in values.items():
            setattr(self, name, value)
        self.evidence_id = canonical_fingerprint(
            {"kind": "reciprocal-resource-evidence", "binding": self.kind, **values}
        )


class ReciprocalBudget(StrictModule, NonTrainableState):
    """Static admission limits of one reciprocal binding.

    ``maximum_chunk_workspace_bytes`` bounds one channel/mode chunk's transient
    grids; ``maximum_retained_bytes`` bounds every array a call returns or keeps
    resident across chunks (for example retained channel densities).
    """

    maximum_grid_points: int = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    maximum_chunk_workspace_bytes: int = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_grid_points: int = 1 << 24,
        maximum_modes: int = 1 << 22,
        maximum_chunk_workspace_bytes: int = 1 << 30,
        maximum_retained_bytes: int = 1 << 31,
    ) -> None:
        self.maximum_grid_points = _positive_int(
            maximum_grid_points, "maximum_grid_points"
        )
        self.maximum_modes = _positive_int(maximum_modes, "maximum_modes")
        self.maximum_chunk_workspace_bytes = _positive_int(
            maximum_chunk_workspace_bytes, "maximum_chunk_workspace_bytes"
        )
        self.maximum_retained_bytes = _positive_int(
            maximum_retained_bytes, "maximum_retained_bytes"
        )
        self.budget_id = canonical_fingerprint(
            {
                "kind": "reciprocal-budget",
                "maximum_grid_points": self.maximum_grid_points,
                "maximum_modes": self.maximum_modes,
                "maximum_chunk_workspace_bytes": self.maximum_chunk_workspace_bytes,
                "maximum_retained_bytes": self.maximum_retained_bytes,
            }
        )

    def validate(self) -> None:
        """Replay the constructor on these limits; refuse a non-canonical budget.

        A restored budget bypasses the constructor, so its positive integer
        limits and ``budget_id`` are re-derived and must reproduce every field.
        """
        replay = ReciprocalBudget(
            maximum_grid_points=self.maximum_grid_points,
            maximum_modes=self.maximum_modes,
            maximum_chunk_workspace_bytes=self.maximum_chunk_workspace_bytes,
            maximum_retained_bytes=self.maximum_retained_bytes,
        )
        require_identical(self, replay, "The reciprocal budget", ValueError)

    def admit(
        self,
        *,
        grid_points: int = 0,
        modes: int = 0,
        workspace_bytes: int = 0,
        retained_bytes: int = 0,
    ) -> None:
        """Refuse a binding or call whose exact static resources exceed this budget."""
        for name, value, limit in (
            ("grid_points", grid_points, self.maximum_grid_points),
            ("modes", modes, self.maximum_modes),
            (
                "chunk_workspace_bytes",
                workspace_bytes,
                self.maximum_chunk_workspace_bytes,
            ),
            ("retained_bytes", retained_bytes, self.maximum_retained_bytes),
        ):
            if int(value) > limit:
                raise ValueError(
                    f"Reciprocal resource {name} requires {int(value)}, exceeding "
                    f"budget {limit}."
                )


class ReciprocalGeometry(StrictModule):
    """Current-cell reciprocal coefficients of one evaluation.

    ``wavevectors`` follow the stored mode layout; ``volume`` is ``|det H|``;
    ``support_radius`` bounds the unambiguously represented modes;
    ``successful`` carries the native lattice solve and measure status (never
    raised).
    """

    inverse: Array
    reciprocal_vectors: Array
    volume: Array
    wavevectors: Array
    squared_wavenumber: Array
    support_radius: Array
    successful: Array


class ReciprocalQuadratic(StrictModule):
    """Value of a reciprocal quadratic form and its numerical status."""

    value: Array
    influence_finite: Array
    sources_finite: Array
    routes_successful: Array
    successful: Array


class ReciprocalMeshDensity(StrictModule):
    """Raw real-FFT transforms ``D[C, *spectral]`` of deposited channels.

    ``D_c(m) ~ window(m) exp(i phi(m)) conj(S_c(k))`` with a channel-independent
    node phase, so ``modulus * Re(D_a conj(D_b))`` approximates
    ``Re(S_a conj(S_b))`` for every channel pair. ``window`` and ``modulus``
    are the prepared mesh's. ``mesh_identity`` is the producing mesh's
    dynamic ``prepared_identity``; a consuming mesh folds its equality into
    ``successful`` instead of raising.
    """

    density: Array
    sources_finite: Array
    routes_successful: Array
    successful: Array
    mesh_identity: IdentityWords


def _lattice_geometry(
    cell: PeriodicCell | None,
    vectors: Array,
    modes: Array,
    maximum_mode: tuple[int, int, int],
    /,
) -> ReciprocalGeometry:
    matrix = jnp.asarray(vectors)
    if matrix.shape != (3, 3):
        raise ValueError("Reciprocal geometry requires (3, 3) row cell vectors.")
    if cell is None:
        if not jnp.issubdtype(matrix.dtype, jnp.floating):
            raise TypeError(
                "Lattice-family geometry requires real floating cell vectors."
            )
        inverse, solved = lattice_right_inverse_with_status(matrix)
        volume, measured = lattice_measure(matrix)
    else:
        inverse, solved = cell.inverse_for_vectors_with_status(matrix)
        volume, measured = cell.measure_with_vectors(matrix)
    inverse = inverse.astype(matrix.dtype)
    reciprocal = _TWO_PI * inverse.T
    wave = contract(
        "...i,ij->...j", modes.astype(matrix.dtype), reciprocal, backend="jax"
    )
    squared = jnp.sum(wave * wave, axis=-1)
    lengths = jnp.sqrt(jnp.sum(matrix * matrix, axis=-1))
    bound = jnp.asarray(maximum_mode, dtype=matrix.dtype) + 1.0
    support = jnp.min(_TWO_PI * bound / lengths)
    successful = jnp.all(solved) & measured & jnp.all(jnp.isfinite(reciprocal))
    return ReciprocalGeometry(
        inverse=inverse,
        reciprocal_vectors=reciprocal,
        volume=volume,
        wavevectors=wave,
        squared_wavenumber=squared,
        support_radius=support,
        successful=successful,
    )


def _evaluate_influence(
    influence: ReciprocalInfluence,
    wave: Array,
    squared: Array,
    channels: int | None,
    /,
) -> Array:
    values = jnp.asarray(influence(wave, squared))
    shape = squared.shape
    if values.shape != shape and (channels is None or values.shape != (channels, *shape)):
        raise ValueError(
            f"Influence must return shape {shape} or (C, *{shape}); got {values.shape}."
        )
    return values


def _channel_chunks(values: Array, chunk: int, /) -> Array:
    """Return channel-major ``(chunks, chunk, ...)`` zero padding to whole chunks."""
    count = values.shape[0]
    chunks = -(-count // chunk)
    padding = chunks * chunk - count
    if padding:
        values = jnp.concatenate(
            (values, jnp.zeros((padding,) + values.shape[1:], dtype=values.dtype))
        )
    return values.reshape((chunks, chunk) + values.shape[1:])


def _source_values(values: ArrayLike, source_capacity: int, /) -> tuple[Array, Array]:
    source = jnp.asarray(values)
    if source.ndim != 2 or source.shape[0] != source_capacity or source.shape[1] == 0:
        raise ValueError(
            f"Reciprocal channel values must have shape ({source_capacity}, C>0)."
        )
    finite = jnp.all(jnp.isfinite(source))
    return jnp.where(jnp.isfinite(source), source, 0.0), finite


def _channel_weights(
    channel_weights: ArrayLike | None, channels: int, dtype: jnp.dtype, /
) -> tuple[Array, Array]:
    weights = (
        jnp.ones((channels,), dtype=dtype)
        if channel_weights is None
        else jnp.asarray(channel_weights, dtype=dtype)
    )
    if weights.shape != (channels,):
        raise ValueError("channel_weights must have one value per channel.")
    return weights, jnp.all(jnp.isfinite(weights))


def _masked_kernel(
    kernel: Array, counted: Array, dtype: jnp.dtype, /
) -> tuple[Array, Array]:
    mask = counted if kernel.ndim == counted.ndim else counted[None]
    kernel = jnp.where(mask, kernel, 0.0)
    finite = jnp.all(jnp.isfinite(kernel))
    return jnp.where(jnp.isfinite(kernel), kernel, 0.0).astype(dtype), finite


class ReciprocalMeshPlan(StrictModule, NonTrainableState):
    """Static particle-mesh reciprocal binding definition.

    ``spline_degree`` is the native cardinal B-spline degree (1 to 5; order is
    degree + 1). ``modulus`` declares how assignment smoothing is removed from
    ``|S|^2``: the exact discrete symbol of the native assignment
    (``cardinal_bspline_periodic_symbol``, Essmann's B-spline modulus) or the
    continuous ``sinc^(p+1)`` window. ``nyquist`` is the declared Nyquist
    convention (see module documentation). ``frame`` ``"anchor-source"``
    measures fractional coordinates from one declared source placed at the mesh
    center, making the mesh energy exactly translation invariant.
    ``channel_chunk`` bounds resident channel grids.
    """

    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    spline_degree: int = eqx.field(static=True)
    modulus: ReciprocalMeshModulus = eqx.field(static=True)
    nyquist: ReciprocalNyquistPolicy = eqx.field(static=True)
    frame: ReciprocalMeshFrame = eqx.field(static=True)
    channel_chunk: int = eqx.field(static=True)
    budget: ReciprocalBudget
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid_shape: Sequence[int],
        /,
        *,
        spline_degree: int = 4,
        modulus: ReciprocalMeshModulus = "discrete-bspline-modulus",
        nyquist: ReciprocalNyquistPolicy = "symmetrized",
        frame: ReciprocalMeshFrame = "cell-origin",
        channel_chunk: int = 8,
        budget: ReciprocalBudget | None = None,
    ) -> None:
        shape = _axis_triple(grid_shape, "grid_shape")
        degree = _positive_int(spline_degree, "spline_degree")
        if degree > 5:
            raise ValueError("Mesh spline_degree must be one to five.")
        if any(value < max(4, degree + 1) for value in shape):
            raise ValueError(
                "Every mesh axis needs at least four and at least degree + 1 points."
            )
        modulus = parse(modulus, ReciprocalMeshModulus, "modulus")
        nyquist = parse(nyquist, ReciprocalNyquistPolicy, "nyquist")
        frame = parse(frame, ReciprocalMeshFrame, "frame")
        chunk = _positive_int(channel_chunk, "channel_chunk")
        budget_ = ReciprocalBudget() if budget is None else budget
        if not isinstance(budget_, ReciprocalBudget):
            raise TypeError("budget must be a ReciprocalBudget.")
        self.grid_shape = shape
        self.spline_degree = degree
        self.modulus = modulus
        self.nyquist = nyquist
        self.frame = frame
        self.channel_chunk = chunk
        self.budget = budget_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reciprocal-mesh-plan",
                "grid_shape": list(shape),
                "spline_degree": degree,
                "modulus": modulus,
                "nyquist": nyquist,
                "frame": frame,
                "channel_chunk": chunk,
                "budget": budget_.budget_id,
            }
        )

    def prepare(
        self,
        cell: PeriodicCell,
        sources: ParticleDiscretization,
        /,
        *,
        anchor_index: int | None = None,
    ) -> "PreparedReciprocalMesh":
        return PreparedReciprocalMesh(self, cell, sources, anchor_index=anchor_index)


class _MeshTables(NamedTuple):
    """Host preparation tables of one mesh plan."""

    modes: np.ndarray
    weights: np.ndarray
    window: np.ndarray
    zero: np.ndarray
    nyquist_indices: np.ndarray
    nyquist_variants: np.ndarray
    excluded: int


def _mesh_tables(plan: ReciprocalMeshPlan, /) -> _MeshTables:
    shape = plan.grid_shape
    half = shape[2] // 2 + 1
    axes = [np.rint(np.fft.fftfreq(count) * count) for count in shape[:2]]
    axes.append(np.arange(half, dtype=np.float64))
    modes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    on_nyquist = np.stack(
        [
            (np.abs(modes[..., axis]) * 2 == count)
            if count % 2 == 0
            else np.zeros(modes.shape[:-1], dtype=np.bool_)
            for axis, count in enumerate(shape)
        ],
        axis=-1,
    )
    any_nyquist = np.any(on_nyquist, axis=-1)
    multiplicity = np.where(
        (modes[..., 2] == 0.0) | (2 * modes[..., 2] == shape[2]), 1.0, 2.0
    )
    if plan.nyquist == "excluded":
        weights = np.where(any_nyquist, 0.0, multiplicity)
        indices = np.zeros((0,), dtype=np.int64)
        variants = np.zeros((1, 0, 3))
        excluded = int(np.count_nonzero(any_nyquist))
    else:
        weights = multiplicity
        indices = np.flatnonzero(any_nyquist.reshape((-1,)))
        flat_modes = modes.reshape((-1, 3))[indices]
        flat_nyquist = on_nyquist.reshape((-1, 3))[indices]
        even_axes = [axis for axis, count in enumerate(shape) if count % 2 == 0]
        rows = []
        for flips in itertools.product((False, True), repeat=len(even_axes)):
            variant = flat_modes.copy()
            for axis, flip in zip(even_axes, flips, strict=True):
                if flip:
                    variant[:, axis] = np.where(
                        flat_nyquist[:, axis], -variant[:, axis], variant[:, axis]
                    )
            rows.append(variant)
        variants = np.stack(rows) if rows else np.zeros((1, 0, 3))
        excluded = 0
    if plan.modulus == "discrete-bspline-modulus":
        symbols = [
            cardinal_bspline_periodic_symbol(plan.spline_degree, count) for count in shape
        ]
        symbols[2] = symbols[2][:half]
    else:
        symbols = [
            np.sinc(axis_modes / count) ** (plan.spline_degree + 1)
            for axis_modes, count in zip(axes, shape, strict=True)
        ]
    window = (
        symbols[0][:, None, None] * symbols[1][None, :, None] * symbols[2][None, None, :]
    )
    counted = weights > 0.0
    if np.any(~np.isfinite(window[counted])) or np.any(window[counted] <= 0.0):
        raise ValueError("Mesh assignment window vanishes on a supported mode.")
    zero = np.all(modes == 0.0, axis=-1)
    return _MeshTables(
        modes=modes,
        weights=weights,
        window=np.where(counted, window, 1.0),
        zero=zero,
        nyquist_indices=indices,
        nyquist_variants=variants,
        excluded=excluded,
    )


class ReciprocalMeshRoutes(StrictModule):
    """Fixed-width assignment routes of one runtime configuration.

    ``state`` holds the splat routes of every source slot and ``fractional``
    the wrapped fractional coordinates. ``mesh_identity`` is the building
    mesh's dynamic ``prepared_identity``.
    """

    state: ParticleGridSplatState
    fractional: Array
    successful: Array
    mesh_identity: IdentityWords


class PreparedReciprocalMesh(StrictModule, NonTrainableState):
    """Per-term mesh binding: splat routes, half spectrum, modulus and resources.

    The PyTree is layout-only. ``splat`` is the assignment transfer prepared
    against the mesh-source indexed-support schema (source capacity, three
    dimensions, coordinate dtype; never a material identity): its statics are
    the splat plan, schema and capacity, while the material particle IDs'
    stable order and the active mask are its dynamic leaves, so deposits
    accumulate in exactly the order of a splat bound to the material support.
    Under the ``"anchor-source"`` frame the anchor slot is a dynamic int32
    ``anchor_index`` (``None`` otherwise, as selected by the static
    ``plan.frame``). The canonical digest of the material binding (plan,
    cell, material splat, anchor, resources) is the dynamic
    ``prepared_identity``; ``prepared_id`` decodes it on the host. Routes and
    densities carry that identity; a mismatch fails ``successful`` in
    compiled code.
    """

    plan: ReciprocalMeshPlan
    cell: PeriodicCell
    splat: PreparedParticleGridSplat
    modes: Array
    hermitian_weights: Array
    window: Array
    modulus: Array
    zero_mode: Array
    nyquist_indices: Array
    nyquist_variant_modes: Array
    anchor_index: Array | None
    maximum_mode: tuple[int, int, int] = eqx.field(static=True)
    spectral_shape: tuple[int, int, int] = eqx.field(static=True)
    resources: ReciprocalResourceEvidence
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: ReciprocalMeshPlan,
        cell: PeriodicCell,
        sources: ParticleDiscretization,
        /,
        *,
        anchor_index: int | None = None,
    ) -> None:
        _require_cell(cell)
        if sources.ambient_dimension != 3:
            raise ValueError("Mesh sources must be three-dimensional particles.")
        capacity = int(sources.capacity)
        anchor: int | None = None
        if plan.frame == "anchor-source":
            if anchor_index is None:
                raise ValueError("An anchor-source mesh frame requires anchor_index.")
            anchor = int(anchor_index)
            if not 0 <= anchor < capacity or not bool(
                np.asarray(sources.active_mask)[anchor]
            ):
                raise ValueError("The mesh anchor must be an active source index.")
        elif anchor_index is not None:
            raise ValueError("anchor_index is only valid for an anchor-source frame.")
        shape = plan.grid_shape
        grid_points = int(np.prod(shape))
        spectral_shape = (shape[0], shape[1], shape[2] // 2 + 1)
        spectral_points = int(np.prod(spectral_shape))
        chunk = plan.channel_chunk
        workspace = chunk * (
            2 * grid_points * _REAL_ITEMSIZE + 2 * spectral_points * _COMPLEX_ITEMSIZE
        )
        tables = _mesh_tables(plan)
        variant_count, nyquist_count = tables.nyquist_variants.shape[:2]
        retained = (
            spectral_points * (5 * _REAL_ITEMSIZE + 1)
            + variant_count * nyquist_count * 3 * _REAL_ITEMSIZE
        )
        plan.budget.admit(
            grid_points=grid_points,
            modes=spectral_points,
            workspace_bytes=workspace,
            retained_bytes=retained,
        )
        grid = TensorGridPlan(
            tuple(UniformCellAxisSpec(size, periodic=True) for size in shape),
            axis_names=("fractional_0", "fractional_1", "fractional_2"),
        ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        splat_plan = ParticleGridSplatPlan(
            grid,
            assignment=TensorBSplineSplatAssignment(plan.spline_degree),
            budget=ParticleGridSplatBudget(
                maximum_scalar_workspace_bytes=plan.budget.maximum_chunk_workspace_bytes
            ),
        )
        # The material-support splat is prepared on the host only for its exact
        # binding digest; it is not retained.
        material_splat = splat_plan.prepare(sources)
        coordinate_dtype = sources.plan.coordinate_dtype
        splat = PreparedParticleGridSplat(
            splat_plan,
            particle_ids=np.asarray(sources.particle_ids),
            active_mask=np.asarray(sources.active_mask, dtype=np.bool_),
            support_id=canonical_fingerprint(
                {
                    "kind": "reciprocal-mesh-source-schema",
                    "source_capacity": capacity,
                    "ambient_dimension": 3,
                    "coordinate_dtype": coordinate_dtype,
                }
            ),
            coordinate_dtype=coordinate_dtype,
        )
        self.plan = plan
        self.cell = cell
        self.splat = splat
        self.modes = jnp.asarray(tables.modes)
        self.hermitian_weights = jnp.asarray(tables.weights)
        self.window = jnp.asarray(tables.window)
        self.modulus = jnp.asarray(
            np.where(tables.weights > 0.0, 1.0 / tables.window**2, 0.0)
        )
        self.zero_mode = jnp.asarray(tables.zero)
        self.nyquist_indices = jnp.asarray(tables.nyquist_indices, dtype=jnp.int32)
        self.nyquist_variant_modes = jnp.asarray(tables.nyquist_variants)
        self.anchor_index = (
            None if anchor is None else jnp.asarray(anchor, dtype=jnp.int32)
        )
        self.maximum_mode = (
            (shape[0] - 1) // 2,
            (shape[1] - 1) // 2,
            (shape[2] - 1) // 2,
        )
        self.spectral_shape = spectral_shape
        self.resources = ReciprocalResourceEvidence(
            "particle-mesh",
            source_capacity=capacity,
            channel_chunk=chunk,
            stored_modes=spectral_points,
            supported_modes=int(np.count_nonzero(tables.weights)),
            excluded_modes=tables.excluded,
            symmetrized_modes=nyquist_count,
            grid_points=grid_points,
            route_count=splat.route_count,
            chunk_workspace_bytes=workspace,
        )
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-reciprocal-mesh",
                "plan": plan.plan_id,
                "cell": cell.cell_id,
                "splat": material_splat.prepared_id,
                "anchor_index": anchor,
                "resources": self.resources.evidence_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared-mesh digest."""
        return self.prepared_identity.fingerprint()

    @property
    def source_capacity(self) -> int:
        return self.splat.source_capacity

    def geometry(self, cell_vectors: ArrayLike, /) -> ReciprocalGeometry:
        """Current-cell wavevectors on the stored half spectrum."""
        return _lattice_geometry(
            self.cell, jnp.asarray(cell_vectors), self.modes, self.maximum_mode
        )

    def resolve_influence(
        self,
        geometry: ReciprocalGeometry,
        influence: ReciprocalInfluence,
        /,
        *,
        channels: int | None = None,
    ) -> Array:
        """Evaluate ``influence`` on the stored spectrum under the Nyquist convention.

        Returns ``(*spectral)`` or ``(channels, *spectral)``. Under
        ``"symmetrized"`` each Nyquist entry is the average over all sign
        choices of its Nyquist components on the current cell.
        """
        values = _evaluate_influence(
            influence, geometry.wavevectors, geometry.squared_wavenumber, channels
        )
        if self.resources.symmetrized_modes == 0:
            return values
        modes = self.nyquist_variant_modes.astype(geometry.reciprocal_vectors.dtype)
        wave = contract("vpi,ij->vpj", modes, geometry.reciprocal_vectors, backend="jax")
        variant = jnp.asarray(influence(wave, jnp.sum(wave * wave, axis=-1)))
        lead = values.shape[: values.ndim - 3]
        if variant.shape != lead + wave.shape[:2]:
            raise ValueError("Influence returned an inconsistent Nyquist variant shape.")
        averaged = jnp.mean(variant, axis=-2).astype(values.dtype)
        flat = values.reshape(lead + (-1,))
        flat = flat.at[..., self.nyquist_indices].set(averaged)
        return flat.reshape(values.shape)

    def spectral_weights(self, zero_mode: ZeroModePolicy, /) -> Array:
        """Hermitian multiplicity times modulus with the stated ``k = 0`` policy."""
        zero_mode = parse(zero_mode, ZeroModePolicy, "zero_mode")
        weights = self.hermitian_weights * self.modulus
        if zero_mode == "exclude":
            weights = jnp.where(self.zero_mode, 0.0, weights)
        return weights

    def build(
        self, positions: ArrayLike, geometry: ReciprocalGeometry, /
    ) -> ReciprocalMeshRoutes:
        """Assignment routes of Cartesian ``positions`` in the geometry's cell."""
        position = jnp.asarray(positions)
        if position.shape != (self.source_capacity, 3):
            raise ValueError("Mesh positions must match the source capacity.")
        origin = self.cell.origin.astype(position.dtype)
        fractional = contract(
            "ni,ij->nj",
            position - origin,
            geometry.inverse.astype(position.dtype),
            backend="jax",
        )
        if self.anchor_index is not None:
            fractional = fractional - fractional[self.anchor_index] + 0.5
        active = self.splat.source_active_mask
        finite = jnp.all(jnp.where(active[:, None], jnp.isfinite(fractional), True))
        fractional = jnp.where(jnp.isfinite(fractional), fractional, 0.0)
        wrapped = fractional - jax.lax.stop_gradient(jnp.floor(fractional))
        state = self.splat.build(wrapped)
        return ReciprocalMeshRoutes(
            state=state,
            fractional=wrapped,
            successful=state.successful & finite & geometry.successful,
            mesh_identity=self.prepared_identity,
        )

    def _routes_successful(self, routes: ReciprocalMeshRoutes, /) -> Array:
        """Route status, failed for routes built by another prepared mesh."""
        return routes.successful & routes.mesh_identity.equals(self.prepared_identity)

    def _transform(self, routes: ReciprocalMeshRoutes, values: Array, /) -> Array:
        deposited = self.splat.deposit_content(routes.state, values)
        return jnp.moveaxis(jnp.fft.rfftn(deposited.content, axes=(0, 1, 2)), -1, 0)

    def transform(
        self, routes: ReciprocalMeshRoutes, values: ArrayLike, /
    ) -> ReciprocalMeshDensity:
        """Retained channel densities ``D[C, *spectral]`` in bounded channel chunks.

        The complete retained output is admitted against
        ``budget.maximum_retained_bytes`` before execution, so each
        species/rank chunk is transformed once and can be reused for every
        pair-dependent influence through :meth:`cross_quadratic`.
        """
        routes_successful = self._routes_successful(routes)
        source, finite = _source_values(values, self.source_capacity)
        channels = source.shape[1]
        chunk = self.plan.channel_chunk
        padded = -(-channels // chunk) * chunk
        self.plan.budget.admit(
            retained_bytes=padded * int(np.prod(self.spectral_shape)) * _COMPLEX_ITEMSIZE,
            workspace_bytes=self.resources.chunk_workspace_bytes,
        )

        def body(carry: None, chunk_values: Array) -> tuple[None, Array]:
            return carry, self._transform(routes, chunk_values.T)

        _, density = jax.lax.scan(
            jax.checkpoint(body), None, _channel_chunks(source.T, chunk)
        )
        density = density.reshape((padded, *self.spectral_shape))[:channels]
        successful = routes_successful & finite & jnp.all(jnp.isfinite(density))
        return ReciprocalMeshDensity(
            density=density,
            sources_finite=finite,
            routes_successful=routes_successful,
            successful=successful,
            mesh_identity=self.prepared_identity,
        )

    def cross_quadratic(
        self,
        left: ReciprocalMeshDensity,
        right: ReciprocalMeshDensity,
        influence: ArrayLike,
        /,
        *,
        zero_mode: ZeroModePolicy,
        channel_weights: ArrayLike | None = None,
    ) -> ReciprocalQuadratic:
        """``sum_c w_c sum_k h modulus G_c Re(D^L_c conj(D^R_c))`` of retained densities.

        ``influence`` is a resolved array (:meth:`resolve_influence`) of shape
        ``(*spectral)`` or ``(C, *spectral)``; channel ``c`` of ``left`` pairs
        with channel ``c`` of ``right``.
        """
        bound = left.mesh_identity.equals(
            self.prepared_identity
        ) & right.mesh_identity.equals(self.prepared_identity)
        if left.density.shape != right.density.shape:
            raise ValueError("Paired mesh densities must have equal channel layouts.")
        channels = left.density.shape[0]
        dtype = left.density.real.dtype
        weights, weights_finite = _channel_weights(channel_weights, channels, dtype)
        base = self.spectral_weights(zero_mode).astype(dtype)
        kernel = jnp.asarray(influence)
        if kernel.shape not in (self.spectral_shape, (channels, *self.spectral_shape)):
            raise ValueError("Resolved influence does not match the mesh spectrum.")
        kernel, influence_finite = _masked_kernel(kernel, base > 0.0, dtype)
        product = jnp.real(left.density * jnp.conj(right.density))
        value = (
            contract("cabd,abd,cabd,c->", product, base, kernel, weights, backend="jax")
            if kernel.ndim == 4
            else contract("cabd,abd,c->", product, base * kernel, weights, backend="jax")
        )
        sources = left.sources_finite & right.sources_finite & weights_finite
        routes = left.routes_successful & right.routes_successful & bound
        successful = (
            left.successful
            & right.successful
            & bound
            & weights_finite
            & influence_finite
            & jnp.isfinite(value)
        )
        return ReciprocalQuadratic(
            value=value,
            influence_finite=influence_finite,
            sources_finite=sources,
            routes_successful=routes,
            successful=successful,
        )

    def quadratic_energy(
        self,
        routes: ReciprocalMeshRoutes,
        values: ArrayLike,
        geometry: ReciprocalGeometry,
        influence: ReciprocalInfluence,
        /,
        *,
        zero_mode: ZeroModePolicy,
        channel_weights: ArrayLike | None = None,
    ) -> ReciprocalQuadratic:
        """Evaluate ``sum_c w_c sum_k h modulus G_c |D_c|^2`` without retaining densities.

        ``values`` has shape ``(sources, C)``. Channels run in checkpointed
        chunks of ``channel_chunk``; only the power of each chunk is reduced.
        """
        routes_successful = self._routes_successful(routes)
        source, finite = _source_values(values, self.source_capacity)
        channels = source.shape[1]
        dtype = source.dtype
        weights, weights_finite = _channel_weights(channel_weights, channels, dtype)
        base = self.spectral_weights(zero_mode).astype(dtype)
        resolved = self.resolve_influence(geometry, influence, channels=channels)
        kernel, influence_finite = _masked_kernel(resolved, base > 0.0, dtype)
        per_channel = kernel.ndim == 4
        chunk = self.plan.channel_chunk
        value_chunks = _channel_chunks(source.T, chunk)
        weight_chunks = _channel_chunks(weights, chunk)
        kernel_chunks = _channel_chunks(kernel, chunk) if per_channel else None
        shared = None if per_channel else base * kernel

        def body(
            total: Array, chunk_inputs: tuple[Array, Array, Array | None]
        ) -> tuple[Array, None]:
            chunk_values, chunk_weights, chunk_kernel = chunk_inputs
            density = self._transform(routes, chunk_values.T)
            power = jnp.real(density) ** 2 + jnp.imag(density) ** 2
            if chunk_kernel is None:
                weighted = contract(
                    "qabd,abd,q->", power, shared, chunk_weights, backend="jax"
                )
            else:
                weighted = contract(
                    "qabd,abd,qabd,q->",
                    power,
                    base,
                    chunk_kernel,
                    chunk_weights,
                    backend="jax",
                )
            return total + weighted, None

        total, _ = jax.lax.scan(
            jax.checkpoint(body),
            jnp.zeros((), dtype=dtype),
            (value_chunks, weight_chunks, kernel_chunks),
        )
        successful = (
            routes_successful
            & finite
            & weights_finite
            & influence_finite
            & jnp.isfinite(total)
        )
        return ReciprocalQuadratic(
            value=total,
            influence_finite=influence_finite,
            sources_finite=finite & weights_finite,
            routes_successful=routes_successful,
            successful=successful,
        )


_MODE_BLOCK_ROWS = 1 << 15
"""Rows of one canonical mode block: fixed host scratch for any plan."""


class _ModeSetLayout(NamedTuple):
    """Exact host counts of one explicit mode set, derived without any array."""

    dims: tuple[int, int, int]
    origin: int
    zero_rows: int
    stored: int
    chunk: int
    chunks: int
    padded: int
    workspace: int
    retained: int


def _mode_set_layout(plan: "ReciprocalModeSetPlan", capacity: int, /) -> _ModeSetLayout:
    """Python-integer layout of ``plan`` for ``capacity`` sources.

    In the ``ij`` flattening of the full box ``|m_i| <= e_i`` the row order is
    lexicographic, the box size ``N`` is odd and ``k = 0`` is its center
    ``(N - 1) / 2``. A mode's first nonzero component is positive exactly when
    it follows the origin, so the canonical half space is the ``(N - 1) / 2``
    flat rows after the center, in order.
    """
    dims = (
        2 * plan.extents[0] + 1,
        2 * plan.extents[1] + 1,
        2 * plan.extents[2] + 1,
    )
    origin = (dims[0] * dims[1] * dims[2] - 1) // 2
    zero_rows = 1 if plan.include_zero_mode else 0
    stored = origin + zero_rows
    chunk = min(plan.mode_chunk, stored)
    chunks = -(-stored // chunk)
    padded = chunks * chunk
    return _ModeSetLayout(
        dims=dims,
        origin=origin,
        zero_rows=zero_rows,
        stored=stored,
        chunk=chunk,
        chunks=chunks,
        padded=padded,
        workspace=chunk * (3 * capacity + 8) * _REAL_ITEMSIZE,
        retained=padded * (5 * _REAL_ITEMSIZE + 1),
    )


def _admit_mode_set(plan: "ReciprocalModeSetPlan", layout: _ModeSetLayout, /) -> None:
    """Refuse a layout over budget or beyond host indexing, before any array."""
    plan.budget.admit(
        modes=layout.padded,
        workspace_bytes=layout.workspace,
        retained_bytes=layout.retained,
    )
    limit = int(np.iinfo(np.intp).max)
    if math.prod(layout.dims) > limit or 3 * layout.padded * _REAL_ITEMSIZE > limit:
        raise ValueError("The explicit mode set exceeds the host index range.")


def _canonical_mode_block(
    layout: _ModeSetLayout, extents: tuple[int, int, int], start: int, stop: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stored rows ``[start, stop)``: optional ``k = 0``, the half space, padding.

    Returns float64 integer modes, Hermitian weights (one for ``k = 0``, two
    for the half space, zero for padding) and the ``k = 0`` mask.
    """
    rows = np.arange(start, stop, dtype=np.intp)
    zero = rows < layout.zero_rows
    live = ~zero & (rows < layout.stored)
    modes = np.zeros((rows.shape[0], 3), dtype=np.float64)
    flat = rows[live] - layout.zero_rows + layout.origin + 1
    if flat.size:
        index = np.stack(np.unravel_index(flat, layout.dims), axis=-1)
        modes[live] = index - np.asarray(extents, dtype=np.intp)
    weights = np.where(zero, 1.0, np.where(live, 2.0, 0.0))
    return modes, weights, zero


def _mode_set_evidence(
    layout: _ModeSetLayout, capacity: int, /
) -> "ReciprocalResourceEvidence":
    return ReciprocalResourceEvidence(
        "explicit-modes",
        source_capacity=capacity,
        channel_chunk=layout.chunk,
        stored_modes=layout.padded,
        supported_modes=layout.stored,
        excluded_modes=0,
        symmetrized_modes=0,
        grid_points=0,
        route_count=0,
        chunk_workspace_bytes=layout.workspace,
    )


def _mode_set_identity(
    plan: "ReciprocalModeSetPlan",
    cell: PeriodicCell | None,
    capacity: int,
    evidence: ReciprocalResourceEvidence,
    /,
) -> str:
    scope_identity = (
        {"cell": cell.cell_id}
        if cell is not None
        else {"scope": "lattice-family-fully-periodic-rank-3"}
    )
    return canonical_fingerprint(
        {
            "kind": "prepared-reciprocal-mode-set",
            "plan": plan.plan_id,
            **scope_identity,
            "source_capacity": capacity,
            "resources": evidence.evidence_id,
        }
    )


class ReciprocalModeSetPlan(StrictModule, NonTrainableState):
    """Explicit integer reciprocal modes ``|m_i| <= extents[i]`` in a canonical half space.

    Each nonzero mode is stored once with Hermitian multiplicity two (its
    first nonzero component is positive). ``include_zero_mode`` adds ``k = 0``
    with multiplicity one for methods that own a finite ``k = 0`` term.
    ``mode_chunk`` bounds the resident ``sources x mode_chunk`` phase block.
    """

    extents: tuple[int, int, int] = eqx.field(static=True)
    include_zero_mode: bool = eqx.field(static=True)
    mode_chunk: int = eqx.field(static=True)
    budget: ReciprocalBudget
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        extents: tuple[int, int, int],
        /,
        *,
        include_zero_mode: bool = False,
        mode_chunk: int = 128,
        budget: ReciprocalBudget | None = None,
    ) -> None:
        bounds = _axis_triple(extents, "extents")
        chunk = _positive_int(mode_chunk, "mode_chunk")
        budget_ = ReciprocalBudget() if budget is None else budget
        if not isinstance(budget_, ReciprocalBudget):
            raise TypeError("budget must be a ReciprocalBudget.")
        self.extents = bounds
        self.include_zero_mode = bool(include_zero_mode)
        self.mode_chunk = chunk
        self.budget = budget_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reciprocal-mode-set-plan",
                "extents": list(bounds),
                "include_zero_mode": self.include_zero_mode,
                "mode_chunk": chunk,
                "budget": budget_.budget_id,
            }
        )

    def prepare(
        self, cell: PeriodicCell, source_capacity: int, /
    ) -> "PreparedReciprocalModeSet":
        """Bind to one concrete prepared rank-3 cell (its full 3D embedding lattice)."""
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell; use prepare_lattice_family.")
        return PreparedReciprocalModeSet(self, cell, source_capacity)

    def prepare_lattice_family(
        self, source_capacity: int, /
    ) -> "PreparedReciprocalModeSet":
        """Bind to the family of current fully periodic rank-3 lattices.

        For graph-batched evaluation whose cells exist only as runtime row
        vectors. Geometry uses the canonical native lattice solve and measure
        directly; their status reaches every result.
        """
        return PreparedReciprocalModeSet(self, None, source_capacity)


class PreparedReciprocalModeSet(StrictModule, NonTrainableState):
    """Per-term explicit mode binding with bounded structure-factor chunks.

    ``cell`` is the concrete prepared cell, or ``None`` for a lattice-family
    binding (:meth:`ReciprocalModeSetPlan.prepare_lattice_family`) that admits
    any current full-rank, fully periodic ``(3, 3)`` row lattice supplied at
    evaluation; no reference cell is claimed and mode coverage is evaluated
    only from the current lattice (``ReciprocalGeometry.support_radius``).
    The binding holds no material content: its static ``prepared_id`` digests
    only the plan, reference cell (or lattice-family scope), source capacity
    and resources, so it is part of the executable layout.
    """

    plan: ReciprocalModeSetPlan
    cell: PeriodicCell | None
    modes: Array
    hermitian_weights: Array
    zero_mode: Array
    source_capacity: int = eqx.field(static=True)
    chunk_count: int = eqx.field(static=True)
    maximum_mode: tuple[int, int, int] = eqx.field(static=True)
    resources: ReciprocalResourceEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ReciprocalModeSetPlan,
        cell: PeriodicCell | None,
        source_capacity: int,
        /,
    ) -> None:
        if cell is not None:
            _require_cell(cell)
        capacity = _positive_int(source_capacity, "source_capacity")
        layout = _mode_set_layout(plan, capacity)
        _admit_mode_set(plan, layout)
        modes = np.empty((layout.padded, 3), dtype=np.float64)
        weights = np.empty((layout.padded,), dtype=np.float64)
        zero = np.empty((layout.padded,), dtype=np.bool_)
        for start in range(0, layout.padded, _MODE_BLOCK_ROWS):
            stop = min(start + _MODE_BLOCK_ROWS, layout.padded)
            modes[start:stop], weights[start:stop], zero[start:stop] = (
                _canonical_mode_block(layout, plan.extents, start, stop)
            )
        self.plan = plan
        self.cell = cell
        self.modes = jnp.asarray(modes)
        self.hermitian_weights = jnp.asarray(weights)
        self.zero_mode = jnp.asarray(zero)
        self.source_capacity = capacity
        self.chunk_count = layout.chunks
        self.maximum_mode = plan.extents
        self.resources = _mode_set_evidence(layout, capacity)
        self.prepared_id = _mode_set_identity(plan, cell, capacity, self.resources)

    def validate(self) -> None:
        """Re-admit this (possibly restored) binding without preparing it again.

        This is a host boundary over concrete arrays. The plan replays its
        constructor with its validated budget. The integer layout of plan and
        capacity is admitted by that budget and then bound to the already
        decoded stored arrays (JAX arrays of exactly the padded row count in
        the canonical real and boolean dtypes) before any canonical row is
        generated, so inflated extents or budgets cannot allocate beyond what
        is stored. Resource evidence, chunk count, support box and
        ``prepared_id`` are recomputed, and every stored row (``k = 0``, the
        half space and padding) must equal the canonical row, compared in
        fixed-size blocks. A concrete ``cell`` is only required to be a full
        rank-3 `PeriodicCell`; its derived data is its owner's (it is not a
        portable artifact value).
        """
        plan = self.plan
        if not isinstance(plan, ReciprocalModeSetPlan):
            raise TypeError("plan must be a ReciprocalModeSetPlan.")
        budget = plan.budget
        if not isinstance(budget, ReciprocalBudget):
            raise TypeError("budget must be a ReciprocalBudget.")
        budget.validate()
        require_identical(
            plan,
            ReciprocalModeSetPlan(
                plan.extents,
                include_zero_mode=plan.include_zero_mode,
                mode_chunk=plan.mode_chunk,
                budget=budget,
            ),
            "The reciprocal mode-set plan",
            ValueError,
        )
        if self.cell is not None:
            _require_cell(self.cell)
        capacity = _positive_int(self.source_capacity, "source_capacity")
        layout = _mode_set_layout(plan, capacity)
        _admit_mode_set(plan, layout)
        real = np.dtype(jax.dtypes.canonicalize_dtype(np.float64))
        stored = (self.modes, self.hermitian_weights, self.zero_mode)
        if any(not isinstance(value, jax.Array) for value in stored):
            raise TypeError(
                "Stored reciprocal modes, weights, and masks must be JAX arrays."
            )
        expected = (
            ((layout.padded, 3), real),
            ((layout.padded,), real),
            ((layout.padded,), np.dtype(np.bool_)),
        )
        if any(
            value.shape != shape or value.dtype != dtype
            for value, (shape, dtype) in zip(stored, expected, strict=True)
        ):
            raise ValueError("The stored mode arrays do not match the mode-set layout.")
        evidence = _mode_set_evidence(layout, capacity)
        require_identical(
            self.resources, evidence, "The reciprocal mode-set resources", ValueError
        )
        if (
            self.chunk_count != layout.chunks
            or self.maximum_mode != plan.extents
            or self.prepared_id != _mode_set_identity(plan, self.cell, capacity, evidence)
        ):
            raise ValueError("The reciprocal mode set does not match its identity.")
        modes, weights, zero = (np.asarray(jax.device_get(value)) for value in stored)
        for start in range(0, layout.padded, _MODE_BLOCK_ROWS):
            stop = min(start + _MODE_BLOCK_ROWS, layout.padded)
            canonical_modes, canonical_weights, canonical_zero = _canonical_mode_block(
                layout, plan.extents, start, stop
            )
            if not (
                np.array_equal(modes[start:stop], canonical_modes.astype(real))
                and np.array_equal(weights[start:stop], canonical_weights.astype(real))
                and np.array_equal(zero[start:stop], canonical_zero)
            ):
                raise ValueError(
                    "The stored reciprocal modes are not the canonical half space."
                )

    @property
    def scope(self) -> ReciprocalBindingScope:
        return "concrete-cell" if self.cell is not None else "lattice-family"

    @property
    def mode_chunk(self) -> int:
        return self.resources.channel_chunk

    def geometry(self, cell_vectors: ArrayLike, /) -> ReciprocalGeometry:
        """Current-lattice wavevectors on the stored modes.

        A concrete binding solves through its prepared cell (whose dtype it
        adopts); a lattice-family binding solves the supplied ``(3, 3)`` rows
        directly in their own dtype. Map over cases for batched lattices.
        """
        return _lattice_geometry(
            self.cell, jnp.asarray(cell_vectors), self.modes, self.maximum_mode
        )

    def quadratic_energy(
        self,
        positions: ArrayLike,
        values: ArrayLike,
        geometry: ReciprocalGeometry,
        influence: ReciprocalInfluence,
        /,
        *,
        zero_mode: ZeroModePolicy,
        channel_weights: ArrayLike | None = None,
    ) -> ReciprocalQuadratic:
        """Evaluate ``sum_c w_c sum_k h G_c |S_c|^2`` over bounded mode chunks.

        ``positions`` may be any periodic representation (integer lattice
        translations leave every phase unchanged).
        """
        zero_mode = parse(zero_mode, ZeroModePolicy, "zero_mode")
        if zero_mode == "include" and not self.plan.include_zero_mode:
            raise ValueError("zero_mode='include' requires a mode set that stores k = 0.")
        position = jnp.asarray(positions)
        if position.shape != (self.source_capacity, 3):
            raise ValueError("Mode-set positions must match the source capacity.")
        source, finite = _source_values(values, self.source_capacity)
        channels = source.shape[1]
        dtype = source.dtype
        weights, weights_finite = _channel_weights(channel_weights, channels, dtype)
        base = self.hermitian_weights.astype(dtype)
        if zero_mode == "exclude":
            base = jnp.where(self.zero_mode, 0.0, base)
        resolved = _evaluate_influence(
            influence, geometry.wavevectors, geometry.squared_wavenumber, channels
        )
        kernel, influence_finite = _masked_kernel(resolved, base > 0.0, dtype)
        per_channel = kernel.ndim == 2
        position_finite = jnp.all(jnp.isfinite(position))
        position = jnp.where(jnp.isfinite(position), position, 0.0).astype(dtype)
        chunk = self.mode_chunk
        wave = geometry.wavevectors.astype(dtype).reshape((self.chunk_count, chunk, 3))
        factor = base * kernel
        factor_chunks = (
            jnp.moveaxis(factor.reshape((channels, self.chunk_count, chunk)), 1, 0)
            if per_channel
            else factor.reshape((self.chunk_count, chunk))
        )

        def body(total: Array, chunk_inputs: tuple[Array, Array]) -> tuple[Array, None]:
            chunk_wave, chunk_factor = chunk_inputs
            phase = contract("nd,md->nm", position, chunk_wave, backend="jax")
            real = contract("nc,nm->cm", source, jnp.cos(phase), backend="jax")
            imaginary = contract("nc,nm->cm", source, jnp.sin(phase), backend="jax")
            power = real * real + imaginary * imaginary
            weighted = contract(
                "cm,c,cm->" if per_channel else "cm,c,m->",
                power,
                weights,
                chunk_factor,
                backend="jax",
            )
            return total + weighted, None

        total, _ = jax.lax.scan(
            jax.checkpoint(body),
            jnp.zeros((), dtype=dtype),
            (wave, factor_chunks),
        )
        routes_successful = geometry.successful & position_finite
        successful = (
            routes_successful
            & finite
            & weights_finite
            & influence_finite
            & jnp.isfinite(total)
        )
        return ReciprocalQuadratic(
            value=total,
            influence_finite=influence_finite,
            sources_finite=finite & weights_finite,
            routes_successful=routes_successful,
            successful=successful,
        )


# Portable data of the explicit mode bindings stored by prepared model terms;
# the owning prepared term replays them through these constructors on restore.
for _artifact in (
    ReciprocalBudget,
    ReciprocalModeSetPlan,
    ReciprocalResourceEvidence,
    PreparedReciprocalModeSet,
):
    register_artifact_value(f"phydrax.atomistic:{_artifact.__name__}", _artifact)
del _artifact


__all__ = [
    "fft_friendly_mesh_size",
    "PreparedReciprocalMesh",
    "PreparedReciprocalModeSet",
    "ReciprocalBindingScope",
    "ReciprocalBudget",
    "ReciprocalGeometry",
    "ReciprocalInfluence",
    "ReciprocalMeshDensity",
    "ReciprocalMeshFrame",
    "ReciprocalMeshModulus",
    "ReciprocalMeshPlan",
    "ReciprocalMeshRoutes",
    "ReciprocalModeSetPlan",
    "ReciprocalNyquistPolicy",
    "ReciprocalQuadratic",
    "ReciprocalResourceEvidence",
    "ZeroModePolicy",
]
