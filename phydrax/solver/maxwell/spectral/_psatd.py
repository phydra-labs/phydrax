#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cartesian PSATD/Galilean spectral Maxwell field solver for explicit PIC.

`SpectralMaxwellPlan` declares one pseudo-spectral analytical time-domain
(PSATD) configuration on a periodic uniform 3-D `StructuredCochainBridge`;
``prepare(transfers, currents)`` binds the P1a particle transfers and returns
`PreparedSpectralMaxwell`, an `AbstractPreparedPICFieldSolver` that also
implements `PICSpectralSymbol`, `PICHuygensSampling`, `PICMultiDeposit`,
`PICRestartState`, `PICGaussProjection`, and `PICGalileanGrid`.

Fields ``E``/``B`` are stored in real space on the Yee (``"staggered"``) or
nodal (``"collocated"``) grid; the Gauss charge is the node density. Each
species deposits the spline-Whitney (Esirkepov) path current per current
sub-interval together with its node-charge change; the solver owns the spectral
treatment of that current (charge-conservation mode, staggering, Galilean lab
conversion ``J = J′ + v_gal ρ``). B5 plane antennas (``antennas=``) add
band-limited electric and magnetic sheet currents with declared sheet charges;
the split-field PML books the divergence its damping creates as layer-supported
absorber charge. The Gauss constraints hold over the whole grid against the
Gauss charge plus these declared charges.

Compatibility (refused at construction):

- ``"spectral-correction"``: constant-J, global FFT; standard or Galilean form.
- ``"vay-deposition"``: standard constant-J only; collocated grids need odd
  spectral extents (Nyquist-plane symbols vanish there).
- ``"update-with-rho"``: every variant and time dependency.
- ``"linear-j"``/``"multi-j"`` and the Galilean variants use ``"update-with-rho"``
  (Galilean constant-J may use the Galilean-form correction with global FFT).
- ``"local-guarded"`` requires finite-order stencils and refuses the spectral
  correction; ``"psatd-pml"`` and Huygens observers are standard only;
  ``"averaged-galilean"`` refuses ``"linear-j"`` (the averaged fields are
  defined for piecewise-constant currents); antennas need global FFT, constant-
  or multi-J, and no Huygens observers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from math import factorial, pi
from typing import Any, assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._dtype_names import RealPrecisionDType
from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import StructuredCochainBridge
from ....discretization.pic import (
    ChargeConservingCurrentPlan,
    PICSpeciesPlan,
    PreparedPICParticleCochainTransfer,
)
from ....discretization.spectral import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
)
from ....sparse import SparseLinearMap
from ....typing import parse
from ..._maxwell_antenna import SampledPlaneCurrentAntennaPlan
from ..._maxwell_far_field import (
    _gather_map,
    _payload,
    _phasors,
    _require_exterior,
    _require_huygens_acquisition,
    _SurfaceGeometry,
    HomogeneousMaxwellExterior,
    HuygensSurfacePhasors,
)
from ..._maxwell_observers import DFTObserverState, MaxwellSpectralAcquisition
from ..._pic_field_solver import (
    AbstractPreparedPICFieldSolver,
    PICFieldAdvance,
    PICFieldDeposit,
    PICGaussProjectionResult,
    PICRestartComponent,
    restart_component,
    restore_component,
)
from ._antenna import PreparedSpectralPlaneAntenna
from ._galilean import (
    exact_interval,
    galilean_charge_current,
    gauss_following_increment,
    gauss_following_window,
    phi_functions,
    SpectralOperators,
    window_integral,
)
from ._pml import SpectralPMLPlan


SpectralMaxwellVariant: TypeAlias = Literal["standard", "galilean", "averaged-galilean"]
SpectralTimeDependency: TypeAlias = Literal["constant-j", "linear-j", "multi-j"]
SpectralChargeConservation: TypeAlias = Literal[
    "spectral-correction", "vay-deposition", "update-with-rho"
]
SpectralStencil: TypeAlias = Literal["infinite-order", "finite-order"]
SpectralDecomposition: TypeAlias = Literal["global-fft", "local-guarded"]
SpectralGrid: TypeAlias = Literal["collocated", "staggered"]
SpectralAbsorber: TypeAlias = Literal["none", "psatd-pml"]

_NEUTRALITY_TOLERANCE = 1.0e-10
_LEVI_CIVITA = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _LEVI_CIVITA[_i, _j, _k] = 1.0
    _LEVI_CIVITA[_i, _k, _j] = -1.0


# -- stencils ---------------------------------------------------------------------


def stencil_coefficients(order: int, staggered: bool, /) -> np.ndarray:
    """Centered derivative weights ``w_m`` of an even-order stencil (Fornberg).

    Staggered: ``f′ ≈ Σ w_m [f(x + (m−½)Δ) − f(x − (m−½)Δ)]/Δ``; collocated:
    ``f′ ≈ Σ w_m [f(x + mΔ) − f(x − mΔ)]/Δ``.
    """
    half = order // 2
    if staggered:
        return np.asarray(
            [
                (-1) ** (m + 1)
                * factorial(2 * half) ** 2
                / (
                    (2 * m - 1) ** 2
                    * factorial(half + m - 1)
                    * factorial(half - m)
                    * factorial(half) ** 2
                    * 4 ** (2 * half - 1)
                )
                for m in range(1, half + 1)
            ]
        )
    return np.asarray(
        [
            (-1) ** (m + 1)
            * factorial(half) ** 2
            / (m * factorial(half - m) * factorial(half + m))
            for m in range(1, half + 1)
        ]
    )


def modified_wavenumber(
    wavenumber: np.ndarray, spacing: float, order: int | None, staggered: bool, /
) -> np.ndarray:
    """Real symbol ``[k]`` of the solver derivative (``k`` at infinite order)."""
    if order is None:
        return np.asarray(wavenumber, dtype=np.float64)
    weights = stencil_coefficients(order, staggered)
    shift = 0.5 if staggered else 0.0
    return sum(
        2.0 * weight * np.sin((m + 1 - shift) * wavenumber * spacing) / spacing
        for m, weight in enumerate(weights)
    )


class _AxisSymbols(NamedTuple):
    """Host 1-D symbols of one axis of one spectral extent."""

    plus: np.ndarray
    minus: np.ndarray
    modified: np.ndarray
    advection: np.ndarray
    to_grid: np.ndarray
    node_to_current: np.ndarray
    vay: np.ndarray


def _axis_symbols(
    count: int, spacing: float, order: int | None, staggered: bool, /
) -> _AxisSymbols:
    k = 2.0 * pi * np.fft.fftfreq(count, d=spacing)
    nyquist = np.zeros(count, dtype=np.bool_)
    if count % 2 == 0:
        nyquist[count // 2] = True
    collocated = np.where(nyquist, 0.0, modified_wavenumber(k, spacing, order, False))
    half = np.exp(0.5j * k * spacing)
    esirkepov = 2.0 * np.sin(0.5 * k * spacing) / spacing
    if staggered:
        modified = modified_wavenumber(k, spacing, order, True)
        plus = 1j * modified * half
        minus = 1j * modified * np.conj(half)
        to_grid = np.ones(count, dtype=np.complex128)
        node_to_current = np.where(nyquist, 0.0, half)
        vay = np.divide(
            esirkepov, modified, out=np.ones(count), where=np.abs(modified) > 0.0
        )
    else:
        modified = collocated
        plus = minus = 1j * collocated
        to_grid = np.where(nyquist, 0.0, np.conj(half))
        node_to_current = np.ones(count, dtype=np.complex128)
        vay = np.where(
            k == 0.0,
            1.0,
            np.divide(
                esirkepov,
                collocated,
                out=np.zeros(count),
                where=np.abs(collocated) > 0.0,
            ),
        )
    return _AxisSymbols(
        plus=plus.astype(np.complex128),
        minus=minus.astype(np.complex128),
        modified=np.asarray(modified, dtype=np.float64),
        advection=np.asarray(collocated, dtype=np.float64),
        to_grid=to_grid.astype(np.complex128),
        node_to_current=node_to_current.astype(np.complex128),
        vay=np.asarray(vay, dtype=np.float64),
    )


class _SpectralGrid(StrictModule, NonTrainableState):
    """Prepared operators and staggering/deposit symbols on one spectral extent."""

    operators: SpectralOperators
    to_grid: Array
    node_to_current: Array
    vay: Array


def _spectral_grid(
    counts: tuple[int, ...],
    spacing: tuple[float, ...],
    order: int | None,
    staggered: bool,
    velocity: tuple[float, float, float],
    speed: float,
    permittivity: float,
    /,
) -> _SpectralGrid:
    axes = tuple(
        _axis_symbols(count, width, order, staggered)
        for count, width in zip(counts, spacing, strict=True)
    )

    def vector(columns: tuple[np.ndarray, ...]) -> np.ndarray:
        expanded = []
        for axis, column in enumerate(columns):
            shape = [1, 1, 1]
            shape[axis] = counts[axis]
            expanded.append(np.broadcast_to(column.reshape(shape), counts))
        return np.stack(expanded, axis=-1)[:, :, :, None, :]

    modified = vector(tuple(value.modified for value in axes))
    advection = vector(tuple(value.advection for value in axes))
    squared = np.sum(modified**2, axis=-1)
    kappa = np.tensordot(advection, np.asarray(velocity), axes=([4], [0]))
    operators = SpectralOperators(
        plus=jnp.asarray(vector(tuple(value.plus for value in axes))),
        minus=jnp.asarray(vector(tuple(value.minus for value in axes))),
        squared=jnp.asarray(squared),
        advection=jnp.asarray(kappa),
        resolved=jnp.asarray(squared > 0.0),
        speed=speed,
        permittivity=permittivity,
    )
    return _SpectralGrid(
        operators=operators,
        to_grid=jnp.asarray(vector(tuple(value.to_grid for value in axes))),
        node_to_current=jnp.asarray(
            vector(tuple(value.node_to_current for value in axes))
        ),
        vay=jnp.asarray(vector(tuple(value.vay for value in axes))),
    )


# -- transforms ---------------------------------------------------------------------


class _GlobalTransform(StrictModule, NonTrainableState):
    """One distributed C2C transform of the whole periodic box."""

    plan: DistributedSpectralExecutionPlan

    def forward(self, values: Array, /) -> Array:
        return self.plan.to_modal_batched(values[:, :, :, None, :])

    def inverse(self, coefficients: Array, /) -> Array:
        return jnp.real(self.plan.to_physical_batched(coefficients))[:, :, :, 0, :]


class _GuardedTransform(StrictModule, NonTrainableState):
    """Guard-cell local transforms of the blocks of a (sub)domain.

    Each block is extended by ``guards`` cells on both sides, transformed as its
    own periodic box by a local unitary FFT, and only its interior is kept on
    the way back. ``indices[a]`` are the block rows along axis ``a`` of the
    input: a device window already padded by the guards (exchanged from the
    neighbors) or a whole periodic axis whose guards wrap.
    """

    indices: tuple[Array, Array, Array]
    blocks: tuple[int, int, int] = eqx.field(static=True)
    guards: tuple[int, int, int] = eqx.field(static=True)
    interior: tuple[int, int, int] = eqx.field(static=True)

    def forward(self, values: Array, /) -> Array:
        first, second, third = self.indices
        blocks = values[
            first[:, None, None, :, None, None],
            second[None, :, None, None, :, None],
            third[None, None, :, None, None, :],
        ]
        extent = blocks.shape[3:6]
        blocks = jnp.transpose(blocks, (3, 4, 5, 0, 1, 2, 6)).reshape(
            (*extent, -1, values.shape[-1])
        )
        return jnp.fft.fftn(blocks.astype(jnp.complex128), axes=(0, 1, 2), norm="ortho")

    def inverse(self, coefficients: Array, /) -> Array:
        blocks = jnp.real(jnp.fft.ifftn(coefficients, axes=(0, 1, 2), norm="ortho"))
        g0, g1, g2 = self.guards
        n0, n1, n2 = self.interior
        s0, s1, s2 = self.blocks
        inner = blocks[g0 : g0 + n0, g1 : g1 + n1, g2 : g2 + n2]
        inner = inner.reshape((n0, n1, n2, s0, s1, s2, blocks.shape[-1]))
        return jnp.transpose(inner, (3, 0, 4, 1, 5, 2, 6)).reshape(
            (s0 * n0, s1 * n1, s2 * n2, blocks.shape[-1])
        )


def _guarded_transform(
    sizes: tuple[int, ...],
    interior: tuple[int, int, int],
    guards: tuple[int, int, int],
    padded: tuple[bool, ...],
    /,
) -> _GuardedTransform:
    """Blocks of ``interior`` cells tiling owned extents ``sizes``.

    Along ``padded`` axes the input carries ``guards`` extra cells on both sides;
    along the others it is the whole periodic axis.
    """
    indices = []
    for size, width, guard, window in zip(sizes, interior, guards, padded, strict=True):
        if size % width:
            raise ValueError("Local-guarded blocks must tile the owned extent.")
        rows = (
            np.arange(size // width)[:, None] * width
            + np.arange(width + 2 * guard)[None, :]
        )
        indices.append(
            jnp.asarray(rows if window else (rows - guard) % size, dtype=jnp.int32)
        )
    return _GuardedTransform(
        indices=(indices[0], indices[1], indices[2]),
        blocks=(
            sizes[0] // interior[0],
            sizes[1] // interior[1],
            sizes[2] // interior[2],
        ),
        guards=guards,
        interior=interior,
    )


# -- state ---------------------------------------------------------------------------


class SpectralMaxwellSource(StrictModule):
    """Deposited source of one step in the solver's current layout.

    ``current[l]`` is the spline-Whitney (Esirkepov) convective current of
    sub-interval ``l`` on the grid's edges, averaged over the sub-interval;
    ``charge_change[l]`` its node-charge change. Both are additive over species
    and consecutive path segments.
    """

    current: Array
    charge_change: Array


class SpectralMaxwellState(StrictModule):
    """Real-space fields, node Gauss charge, and optional absorber/averaged state.

    ``electric``/``magnetic`` are ``[N₀, N₁, N₂, 3]`` components at their grid
    locations. With a PML the totals equal the sum of the split parts
    ``electric_split[..., c, a]``. ``averaged_*`` are the time-averaged fields the
    averaged Galilean variant gathers.

    ``absorber_charge``/``absorber_magnetic_charge`` are the bookkeeping
    divergences the PML damping creates (supported in
    `PreparedSpectralMaxwell.absorber_support`; ``None`` without a PML) and
    ``antenna_charge``/``antenna_magnetic_charge`` the declared sheet charges of
    the antennas (``None`` without antennas). The Gauss laws read
    ``∇⁻·E = (ρ + ρ_absorber + ρ_antenna)/ε`` and
    ``∇⁺·B = ρ_m,absorber + ρ_m,antenna`` on the resolved modes.
    """

    electric: Array
    magnetic: Array
    charge: Array
    averaged_electric: Array | None
    averaged_magnetic: Array | None
    electric_split: Array | None
    magnetic_split: Array | None
    absorber_charge: Array | None
    absorber_magnetic_charge: Array | None
    antenna_charge: Array | None
    antenna_magnetic_charge: Array | None
    observations: tuple[DFTObserverState, ...]


class SpectralMaxwellDiagnostics(StrictModule):
    """Constraint, energy, absorber, antenna, and surface evidence of one field step.

    ``unresolved_charge`` is the largest node-charge content in spectral modes
    the solver's derivative cannot see (``[k]² = 0`` other than ``k = 0``);
    ``mean_charge`` the ``k = 0`` content. The constraints are the largest
    Gauss residuals over the whole grid against the Gauss charge plus the
    declared absorber and antenna charges. ``absorbed_energy`` and
    ``absorbed_momentum`` (``ε∫E×B`` with ``B`` co-located with ``E``) are
    removed by the PML damping; ``antenna_work[a]`` is the work antenna ``a``
    did on the field over the step, ``−∫∫(J·E + M·H) dV dt`` of the analytic
    interval fields; ``surface_current`` is the largest current sampled on a
    Huygens surface inside its window.
    """

    electric_constraint: Array
    magnetic_constraint: Array
    energy: Array
    absorbed_energy: Array
    absorbed_momentum: Array
    antenna_work: Array
    unresolved_charge: Array
    mean_charge: Array
    surface_current: Array


class SpectralLocalUpdate(StrictModule):
    """Fields of one PSATD step before their evidence is formed.

    ``electric``/``magnetic`` are the damped end fields, ``charge`` the end node
    charge, ``undamped_*`` the PML fields before damping (``None`` without an
    absorber), ``absorber_*``/``antenna_*`` the end declared charges (as in
    `SpectralMaxwellState`), ``antenna_work[..., a]`` the per-cell step work of
    antenna ``a`` (``None`` without antennas), and
    ``electric_divergence``/``magnetic_divergence`` the discrete ``∇⁻·E``/
    ``∇⁺·B`` of the end fields. Every leaf leads with the grid axes:
    `PreparedSpectralMaxwell.guarded_update` returns one device's owned cells;
    `complete_advance` consumes the whole grid.
    """

    electric: Array
    magnetic: Array
    charge: Array
    averaged_electric: Array | None
    averaged_magnetic: Array | None
    electric_split: Array | None
    magnetic_split: Array | None
    undamped_electric: Array | None
    undamped_magnetic: Array | None
    absorber_charge: Array | None
    absorber_magnetic_charge: Array | None
    antenna_charge: Array | None
    antenna_magnetic_charge: Array | None
    antenna_work: Array | None
    electric_divergence: Array
    magnetic_divergence: Array


def _unchanged(values: Array, /) -> Array:
    return values


# -- Huygens ------------------------------------------------------------------------


class SpectralHuygensBoxPlan(StrictModule, NonTrainableState):
    """Closed axis-aligned Huygens box on node planes of the staggered grid.

    ``lower``/``upper`` are node indices. Each tangential ``E`` component is
    sampled at its own edge midpoints in the face (no interpolation) with the
    midpoint rule along its edge and the trapezoid rule across it; the paired
    tangential ``H = B/μ`` is interpolated to the same points by the
    fourth-order midpoint stencil ``(−1, 9, 9, −1)/16`` across the face, so the
    surface quadrature is exact for the stationary far-field component up to
    ``O((k_n h)⁴)``. The stencil reaches ``3h/2`` on both sides of each face.
    """

    lower: tuple[int, int, int] = eqx.field(static=True)
    upper: tuple[int, int, int] = eqx.field(static=True)
    acquisition: MaxwellSpectralAcquisition
    exterior: HomogeneousMaxwellExterior
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: Sequence[int],
        upper: Sequence[int],
        acquisition: MaxwellSpectralAcquisition,
        exterior: HomogeneousMaxwellExterior,
        /,
    ) -> None:
        low = tuple(int(value) for value in lower)
        high = tuple(int(value) for value in upper)
        if len(low) != 3 or len(high) != 3:
            raise ValueError("A spectral Huygens box is three-dimensional.")
        if any(a < 0 or b <= a for a, b in zip(low, high, strict=True)):
            raise ValueError("Huygens box node bounds must satisfy 0 <= lower < upper.")
        self.lower = (low[0], low[1], low[2])
        self.upper = (high[0], high[1], high[2])
        self.acquisition = _require_huygens_acquisition(acquisition)
        self.exterior = _require_exterior(exterior)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-huygens-box",
                "lower": list(low),
                "upper": list(high),
                "acquisition": self.acquisition.acquisition_id,
                "exterior": self.exterior.exterior_id,
            }
        )


class PreparedSpectralHuygensBox(StrictModule, NonTrainableState):
    """Prepared spectral Huygens box; implements `MaxwellHuygensSampler`.

    ``current_gather`` maps the solver's flattened current to the surface
    current entries the ``J = 0`` admissibility watches.
    """

    geometry: _SurfaceGeometry
    current_gather: SparseLinearMap
    acquisition: MaxwellSpectralAcquisition
    exterior: HomogeneousMaxwellExterior
    prepared_id: str = eqx.field(static=True)

    @property
    def surface_count(self) -> int:
        return self.geometry.measures.shape[0]

    def initialize(self, /) -> DFTObserverState:
        return self.acquisition.initialize((6 * self.surface_count,))

    def update(
        self, time: Array, electric: Array, magnetic: Array, state: Any, /
    ) -> DFTObserverState:
        """Fold one sample; ``electric``/``magnetic`` are flattened ``E`` and ``H``."""
        if not isinstance(state, DFTObserverState):
            raise TypeError("Spectral Huygens box requires DFTObserverState.")
        return self.acquisition.accumulate(
            state, time, _payload(self.geometry, electric, magnetic)
        )

    def surface_current(self, current: Array, /) -> Array:
        return jnp.max(jnp.abs(self.current_gather.mv(current)), initial=0.0)

    def surface_phasors(self, state: Any, /) -> HuygensSurfacePhasors:
        return _phasors(self.geometry, self.acquisition, self.exterior, state)


# Fourth-order midpoint interpolation across a face: offsets of the staggered
# node-plane index (positions ``(offset + ½)h`` from the face) and weights.
_NORMAL_STENCIL = ((-2, -1.0 / 16.0), (-1, 9.0 / 16.0), (0, 9.0 / 16.0), (1, -1.0 / 16.0))


def _box_geometry(
    box: SpectralHuygensBoxPlan,
    counts: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    /,
) -> tuple[_SurfaceGeometry, SparseLinearMap]:
    """Yee-native surface samples and the surface current entries.

    On the face ``x_a = node`` the tangential ``E_c`` lives at
    ``(node, j_c + ½, m_d)`` and its partner ``B_d`` at
    ``(node ± ½, j_c + ½, m_d)``; every sample carries one ``E``/``H`` pair
    (the other tangential components are zero), so the far-field sum is the
    product midpoint/trapezoid rule on each component's own lattice.
    """

    def flat(index: Sequence[int], component: int, /) -> int:
        wrapped = tuple(value % count for value, count in zip(index, counts, strict=True))
        return int(np.ravel_multi_index(wrapped, counts)) * 3 + component

    positions, normals, measures, patches = [], [], [], []
    electric, magnetic, current = [], [], []
    cell = 0
    for normal_axis in range(3):
        tangential = tuple(axis for axis in range(3) if axis != normal_axis)
        for side, node in enumerate((box.lower[normal_axis], box.upper[normal_axis])):
            normal = np.zeros(3)
            normal[normal_axis] = 1.0 if side else -1.0
            for edge_axis, across in (tangential, tangential[::-1]):
                for j in range(box.lower[edge_axis], box.upper[edge_axis]):
                    for m in range(box.lower[across], box.upper[across] + 1):
                        index = [0, 0, 0]
                        index[normal_axis], index[edge_axis], index[across] = node, j, m
                        point = np.asarray(origin) + np.asarray(index) * np.asarray(
                            spacing
                        )
                        point[edge_axis] += 0.5 * spacing[edge_axis]
                        end = m in (box.lower[across], box.upper[across])
                        positions.append(point)
                        normals.append(normal)
                        measures.append(
                            spacing[edge_axis] * spacing[across] * (0.5 if end else 1.0)
                        )
                        patches.append(2 * normal_axis + side)
                        electric.append(
                            (flat(index, edge_axis), 3 * cell + edge_axis, 1.0)
                        )
                        current.append(flat(index, edge_axis))
                        for offset, weight in _NORMAL_STENCIL:
                            shifted = list(index)
                            shifted[normal_axis] = node + offset
                            magnetic.append(
                                (flat(shifted, across), 3 * cell + across, weight)
                            )
                        cell += 1
            for jb in range(box.lower[tangential[0]], box.upper[tangential[0]] + 1):
                for jc in range(box.lower[tangential[1]], box.upper[tangential[1]] + 1):
                    for offset in (-1, 0):
                        index = [0, 0, 0]
                        index[normal_axis] = node + offset
                        index[tangential[0]], index[tangential[1]] = jb, jc
                        current.append(flat(index, normal_axis))
    size = 3 * int(np.prod(counts))
    geometry_id = canonical_fingerprint(
        {
            "kind": "spectral-huygens-geometry",
            "box": box.plan_id,
            "counts": list(counts),
            "spacing": list(spacing),
            "origin": list(origin),
        }
    )

    def routes(values: list[tuple[int, int, float]], name: str) -> Any:
        array = np.asarray(values)
        return _gather_map(
            array[:, 0].astype(np.int32),
            array[:, 1].astype(np.int32),
            array[:, 2],
            source_size=size,
            target_count=cell,
            operator_id=f"{geometry_id}:{name}",
        )

    geometry = _SurfaceGeometry(
        positions=jnp.asarray(np.stack(positions)),
        normals=jnp.asarray(np.stack(normals)),
        measures=jnp.asarray(np.asarray(measures)),
        patches=jnp.asarray(np.asarray(patches, dtype=np.int32)),
        electric_gather=routes(electric, "electric"),
        magnetic_gather=routes(magnetic, "magnetic"),
        electric_indices=jnp.asarray(
            np.unique(np.asarray([value[0] for value in electric], dtype=np.int32))
        ),
        magnetic_indices=jnp.asarray(
            np.unique(np.asarray([value[0] for value in magnetic], dtype=np.int32))
        ),
        geometry_id=geometry_id,
    )
    entries = np.unique(np.asarray(current, dtype=np.int32))
    # The current gather selects each watched entry once (targets past the
    # entry count stay zero).
    watched = _gather_map(
        entries,
        np.arange(entries.size, dtype=np.int32),
        np.ones(entries.size),
        source_size=size,
        target_count=-(-entries.size // 3),
        operator_id=f"{geometry_id}:current",
    )
    return geometry, watched


# -- plan ------------------------------------------------------------------------------


def _three(values: Sequence[int] | None, name: str, /) -> tuple[int, int, int] | None:
    if values is None:
        return None
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must give one value per axis.")
    return (result[0], result[1], result[2])


class _SpectralGridMetadata(NamedTuple):
    counts: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]


class _SpectralExecution(NamedTuple):
    stencil_order: int | None
    current_intervals: int
    galilean_velocity: tuple[float, float, float]
    subdomains: tuple[int, int, int] | None
    guard_cells: tuple[int, int, int] | None


def _spectral_grid_metadata(bridge: StructuredCochainBridge, /) -> _SpectralGridMetadata:
    if not isinstance(bridge, StructuredCochainBridge):
        raise TypeError("bridge must be a StructuredCochainBridge.")
    axes = bridge.grid.structured_axes
    if bridge.dimension != 3 or any(not axis.periodic for axis in axes):
        raise ValueError("Spectral Maxwell requires a periodic three-dimensional grid.")
    widths = tuple(np.asarray(axis.interval_widths, dtype=np.float64) for axis in axes)
    if any(not np.allclose(value, value[0], rtol=1e-12, atol=0.0) for value in widths):
        raise ValueError("Spectral Maxwell requires uniform axes.")
    counts = tuple(int(axis.interval_centers.size) for axis in axes)
    return _SpectralGridMetadata(
        (counts[0], counts[1], counts[2]),
        (float(widths[0][0]), float(widths[1][0]), float(widths[2][0])),
        (
            float(axes[0].bounds[0]),
            float(axes[1].bounds[0]),
            float(axes[2].bounds[0]),
        ),
    )


def _spectral_execution(
    variant: SpectralMaxwellVariant,
    time_dependency: SpectralTimeDependency,
    charge_conservation: SpectralChargeConservation,
    stencil: SpectralStencil,
    stencil_order: int | None,
    decomposition: SpectralDecomposition,
    grid: SpectralGrid,
    counts: tuple[int, int, int],
    speed: float,
    galilean_velocity: Sequence[float] | None,
    current_substeps: int | None,
    subdomains: Sequence[int] | None,
    guard_cells: Sequence[int] | None,
    /,
) -> _SpectralExecution:
    order = _stencil_order(stencil, stencil_order)
    intervals = _current_intervals(time_dependency, current_substeps)
    velocity = _galilean_velocity(variant, galilean_velocity, speed)
    blocks, guards = _local_blocks(
        decomposition,
        order,
        counts,
        _three(subdomains, "subdomains"),
        _three(guard_cells, "guard_cells"),
    )
    local_counts = (
        counts
        if blocks is None or guards is None
        else tuple(
            count // block + 2 * guard
            for count, block, guard in zip(counts, blocks, guards, strict=True)
        )
    )
    _require_charge_conservation(
        charge_conservation,
        variant,
        time_dependency,
        decomposition,
        grid,
        local_counts,
    )
    if variant == "averaged-galilean" and time_dependency == "linear-j":
        raise ValueError(
            "averaged-galilean fields are defined for piecewise-constant currents; "
            "use constant-j or multi-j."
        )
    return _SpectralExecution(order, intervals, velocity, blocks, guards)


def _validate_spectral_absorber(
    absorber: SpectralAbsorber,
    pml: SpectralPMLPlan | None,
    variant: SpectralMaxwellVariant,
    counts: tuple[int, int, int],
    stencil_order: int | None,
    /,
) -> None:
    match absorber:
        case "none":
            if pml is not None:
                raise ValueError("A PML plan requires absorber='psatd-pml'.")
        case "psatd-pml":
            if not isinstance(pml, SpectralPMLPlan):
                raise TypeError("absorber='psatd-pml' requires a SpectralPMLPlan.")
            if variant != "standard":
                raise ValueError(
                    "The PSATD PML is standard-only; Galilean coordinates with a "
                    "PML are refused."
                )
            if any(
                2 * layer >= count
                for layer, count in zip(pml.thickness, counts, strict=True)
            ):
                raise ValueError("PML layers leave no interior.")
            if stencil_order is None and max(pml.thickness) < 2:
                raise ValueError(
                    "An infinite-order PSATD PML confines its layer charge to "
                    "the two outermost planes of its thickest layer, which must "
                    "span at least two cells."
                )
        case _:
            assert_never(absorber)


def _validated_spectral_observers(
    observers: Sequence[SpectralHuygensBoxPlan],
    variant: SpectralMaxwellVariant,
    grid: SpectralGrid,
    pml: SpectralPMLPlan | None,
    counts: tuple[int, int, int],
    permittivity: float,
    permeability: float,
    /,
) -> tuple[SpectralHuygensBoxPlan, ...]:
    values = tuple(observers)
    if any(not isinstance(value, SpectralHuygensBoxPlan) for value in values):
        raise TypeError("observers must be SpectralHuygensBoxPlan instances.")
    if values and variant != "standard":
        raise ValueError("Huygens sampling requires the standard (lab-frame) variant.")
    if values and grid != "staggered":
        raise ValueError(
            "Huygens sampling requires grid='staggered': the collocated half-cell "
            "current centering leaves current on every Huygens surface."
        )
    for box in values:
        limit = (0, 0, 0) if pml is None else tuple(value + 2 for value in pml.thickness)
        if any(
            low < layer or high > count - layer
            for low, high, layer, count in zip(
                box.lower, box.upper, limit, counts, strict=True
            )
        ):
            raise ValueError(
                "Huygens box must lie inside the grid interior, two cells clear "
                "of any absorbing layer."
            )
        if (
            box.exterior.permittivity != permittivity
            or box.exterior.permeability != permeability
        ):
            raise ValueError("Huygens exterior must be the solver's vacuum medium.")
    return values


class SpectralMaxwellPlan(StrictModule, NonTrainableState):
    """Declared Cartesian PSATD configuration on a periodic uniform 3-D grid.

    ``antennas`` are B5 `SampledPlaneCurrentAntennaPlan` sheets planned on
    ``bridge``; the solver radiates them as band-limited continuum sheet
    currents (`PreparedSpectralPlaneAntenna`) and tracks their declared charges.
    """

    bridge: StructuredCochainBridge
    variant: SpectralMaxwellVariant = eqx.field(static=True)
    time_dependency: SpectralTimeDependency = eqx.field(static=True)
    charge_conservation: SpectralChargeConservation = eqx.field(static=True)
    stencil: SpectralStencil = eqx.field(static=True)
    stencil_order: int | None = eqx.field(static=True)
    decomposition: SpectralDecomposition = eqx.field(static=True)
    grid: SpectralGrid = eqx.field(static=True)
    absorber: SpectralAbsorber = eqx.field(static=True)
    galilean_velocity: tuple[float, float, float] = eqx.field(static=True)
    current_intervals: int = eqx.field(static=True)
    subdomains: tuple[int, int, int] | None = eqx.field(static=True)
    guard_cells: tuple[int, int, int] | None = eqx.field(static=True)
    pml: SpectralPMLPlan | None
    observers: tuple[SpectralHuygensBoxPlan, ...]
    antennas: tuple[SampledPlaneCurrentAntennaPlan, ...]
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    topology: SpectralMeshTopology
    counts: tuple[int, int, int] = eqx.field(static=True)
    spacing: tuple[float, float, float] = eqx.field(static=True)
    origin: tuple[float, float, float] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        variant: SpectralMaxwellVariant = "standard",
        time_dependency: SpectralTimeDependency = "constant-j",
        charge_conservation: SpectralChargeConservation = "spectral-correction",
        stencil: SpectralStencil = "infinite-order",
        stencil_order: int | None = None,
        decomposition: SpectralDecomposition = "global-fft",
        grid: SpectralGrid = "collocated",
        absorber: SpectralAbsorber = "none",
        galilean_velocity: Sequence[float] | None = None,
        current_substeps: int | None = None,
        subdomains: Sequence[int] | None = None,
        guard_cells: Sequence[int] | None = None,
        pml: SpectralPMLPlan | None = None,
        observers: Sequence[SpectralHuygensBoxPlan] = (),
        antennas: Sequence[SampledPlaneCurrentAntennaPlan] = (),
        permittivity: float = 1.0,
        permeability: float = 1.0,
        topology: SpectralMeshTopology | None = None,
    ) -> None:
        variant = parse(variant, SpectralMaxwellVariant, "variant")
        time_dependency = parse(
            time_dependency, SpectralTimeDependency, "time_dependency"
        )
        charge_conservation = parse(
            charge_conservation, SpectralChargeConservation, "charge_conservation"
        )
        stencil = parse(stencil, SpectralStencil, "stencil")
        decomposition = parse(decomposition, SpectralDecomposition, "decomposition")
        grid = parse(grid, SpectralGrid, "grid")
        absorber = parse(absorber, SpectralAbsorber, "absorber")
        metadata = _spectral_grid_metadata(bridge)
        epsilon, mu = float(permittivity), float(permeability)
        if not (np.isfinite(epsilon) and epsilon > 0.0 and np.isfinite(mu) and mu > 0.0):
            raise ValueError("permittivity and permeability must be finite and positive.")
        execution = _spectral_execution(
            variant,
            time_dependency,
            charge_conservation,
            stencil,
            stencil_order,
            decomposition,
            grid,
            metadata.counts,
            1.0 / np.sqrt(epsilon * mu),
            galilean_velocity,
            current_substeps,
            subdomains,
            guard_cells,
        )
        _validate_spectral_absorber(
            absorber,
            pml,
            variant,
            metadata.counts,
            execution.stencil_order,
        )
        observer_values = _validated_spectral_observers(
            observers,
            variant,
            grid,
            pml,
            metadata.counts,
            epsilon,
            mu,
        )
        antenna_values = _require_antennas(
            antennas,
            bridge,
            time_dependency,
            decomposition,
            observer_values,
            execution.galilean_velocity,
            epsilon,
            mu,
        )
        topology_ = SpectralMeshTopology.one_device() if topology is None else topology
        if not isinstance(topology_, SpectralMeshTopology):
            raise TypeError("topology must be SpectralMeshTopology or None.")
        self.bridge = bridge
        self.variant = variant
        self.time_dependency = time_dependency
        self.charge_conservation = charge_conservation
        self.stencil = stencil
        self.stencil_order = execution.stencil_order
        self.decomposition = decomposition
        self.grid = grid
        self.absorber = absorber
        self.galilean_velocity = execution.galilean_velocity
        self.current_intervals = execution.current_intervals
        self.subdomains = execution.subdomains
        self.guard_cells = execution.guard_cells
        self.pml = pml
        self.observers = observer_values
        self.antennas = antenna_values
        self.permittivity = epsilon
        self.permeability = mu
        self.topology = topology_
        self.counts = metadata.counts
        self.spacing = metadata.spacing
        self.origin = metadata.origin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-maxwell-plan",
                "bridge": bridge.bridge_id,
                "variant": variant,
                "time_dependency": time_dependency,
                "charge_conservation": charge_conservation,
                "stencil": stencil,
                "stencil_order": execution.stencil_order,
                "decomposition": decomposition,
                "grid": grid,
                "absorber": absorber,
                "galilean_velocity": list(execution.galilean_velocity),
                "current_intervals": execution.current_intervals,
                "subdomains": (
                    None if execution.subdomains is None else list(execution.subdomains)
                ),
                "guard_cells": (
                    None if execution.guard_cells is None else list(execution.guard_cells)
                ),
                "pml": None if pml is None else pml.plan_id,
                "observers": [value.plan_id for value in observer_values],
                "antennas": [value.source_id for value in antenna_values],
                "permittivity": epsilon,
                "permeability": mu,
                "topology": topology_.topology_id,
            }
        )

    @property
    def speed_of_light(self) -> float:
        return float(1.0 / np.sqrt(self.permittivity * self.permeability))

    def prepare(
        self,
        transfers: Sequence[PreparedPICParticleCochainTransfer] = (),
        currents: Sequence[ChargeConservingCurrentPlan] = (),
        /,
    ) -> PreparedSpectralMaxwell:
        return PreparedSpectralMaxwell(self, transfers, currents)


def _stencil_order(stencil: SpectralStencil, order: int | None, /) -> int | None:
    match stencil:
        case "infinite-order":
            if order is not None:
                raise ValueError("infinite-order stencils take no stencil_order.")
            return None
        case "finite-order":
            if order is None or int(order) < 2 or int(order) % 2:
                raise ValueError(
                    "finite-order stencils require an even stencil_order >= 2."
                )
            return int(order)
        case _:
            raise ValueError("stencil is invalid.")


def _current_intervals(
    dependency: SpectralTimeDependency, substeps: int | None, /
) -> int:
    match dependency:
        case "constant-j":
            if substeps is not None:
                raise ValueError("current_substeps is only defined for multi-j.")
            return 1
        case "linear-j":
            if substeps is not None:
                raise ValueError("current_substeps is only defined for multi-j.")
            return 2
        case "multi-j":
            if substeps is None or int(substeps) < 2:
                raise ValueError("multi-j requires current_substeps >= 2.")
            return int(substeps)
        case _:
            raise ValueError("time_dependency is invalid.")


def _galilean_velocity(
    variant: SpectralMaxwellVariant, velocity: Sequence[float] | None, speed: float, /
) -> tuple[float, float, float]:
    match variant:
        case "standard":
            if velocity is not None:
                raise ValueError("The standard variant takes no galilean_velocity.")
            return (0.0, 0.0, 0.0)
        case "galilean" | "averaged-galilean":
            if velocity is None:
                raise ValueError("Galilean variants require galilean_velocity.")
            value = np.asarray(velocity, dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("galilean_velocity must be a finite 3-vector.")
            magnitude = float(np.linalg.norm(value))
            if not 0.0 < magnitude < speed:
                raise ValueError("galilean_velocity must be nonzero and subluminal.")
            return (float(value[0]), float(value[1]), float(value[2]))
        case _:
            raise ValueError("variant is invalid.")


def _local_blocks(
    decomposition: SpectralDecomposition,
    order: int | None,
    counts: tuple[int, ...],
    blocks: tuple[int, int, int] | None,
    guards: tuple[int, int, int] | None,
    /,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    match decomposition:
        case "global-fft":
            if blocks is not None or guards is not None:
                raise ValueError("subdomains and guard_cells require local-guarded.")
            return None, None
        case "local-guarded":
            if order is None:
                raise ValueError(
                    "local-guarded decomposition requires finite-order stencils."
                )
            if blocks is None or guards is None:
                raise ValueError("local-guarded requires subdomains and guard_cells.")
            if any(b <= 0 or n % b for b, n in zip(blocks, counts, strict=True)):
                raise ValueError("subdomains must divide every axis count.")
            if any(g < order // 2 or g > n for g, n in zip(guards, counts, strict=True)):
                raise ValueError(
                    "guard_cells must cover the stencil half-width and fit the grid."
                )
            return blocks, guards
        case _:
            raise ValueError("decomposition is invalid.")


def _require_charge_conservation(
    mode: SpectralChargeConservation,
    variant: SpectralMaxwellVariant,
    dependency: SpectralTimeDependency,
    decomposition: SpectralDecomposition,
    grid: SpectralGrid,
    extents: tuple[int, ...],
    /,
) -> None:
    match mode:
        case "spectral-correction":
            if dependency != "constant-j":
                raise ValueError(
                    "spectral-correction requires constant-j; linear-j and multi-j "
                    "use update-with-rho."
                )
            if decomposition != "global-fft":
                raise ValueError(
                    "spectral-correction needs the global spectrum; local-guarded is "
                    "refused."
                )
        case "vay-deposition":
            if variant != "standard" or dependency != "constant-j":
                raise ValueError(
                    "vay-deposition is defined for the standard constant-j PSATD only."
                )
            if grid == "collocated" and any(count % 2 == 0 for count in extents):
                raise ValueError(
                    "vay-deposition on collocated grids needs odd spectral extents: "
                    "the Nyquist-plane derivative symbol vanishes."
                )
        case "update-with-rho":
            return
        case _:
            raise ValueError("charge_conservation is invalid.")


def _require_antennas(
    antennas: Sequence[SampledPlaneCurrentAntennaPlan],
    bridge: StructuredCochainBridge,
    time_dependency: SpectralTimeDependency,
    decomposition: SpectralDecomposition,
    observers: tuple[SpectralHuygensBoxPlan, ...],
    velocity: tuple[float, float, float],
    permittivity: float,
    permeability: float,
    /,
) -> tuple[SampledPlaneCurrentAntennaPlan, ...]:
    values = tuple(antennas)
    if any(not isinstance(value, SampledPlaneCurrentAntennaPlan) for value in values):
        raise TypeError("antennas must be SampledPlaneCurrentAntennaPlan instances.")
    if not values:
        return values
    if decomposition != "global-fft":
        raise ValueError(
            "Spectral antennas require the global-fft decomposition: a band-limited "
            "sheet extends over the whole normal axis."
        )
    if time_dependency == "linear-j":
        raise ValueError(
            "Spectral antenna currents are constant over each step; use constant-j "
            "or multi-j (linear-j has no in-step Gauss-following field for the "
            "antenna work ledger)."
        )
    if observers:
        raise ValueError(
            "A spectral antenna sheet has current at every normal node, so a "
            "Huygens surface cannot be current-free; Huygens sampling with antennas "
            "is refused."
        )
    for value in values:
        if value.bridge.bridge_id != bridge.bridge_id:
            raise ValueError("Every antenna must be planned on the spectral bridge.")
        if not (
            np.isclose(value.medium.permittivity, permittivity, rtol=1e-12, atol=0.0)
            and np.isclose(value.medium.permeability, permeability, rtol=1e-12, atol=0.0)
        ):
            raise ValueError("Antenna media must be the solver's vacuum medium.")
        if any(velocity[axis] != 0.0 for axis in value.tangential_axes):
            raise ValueError(
                "A Galilean grid carrying antennas must move along their normals."
            )
    return values


# -- prepared solver ----------------------------------------------------------------


def _component_offsets(grid: SpectralGrid, /) -> tuple[np.ndarray, np.ndarray]:
    """Half-cell offsets ``[component, axis]`` of ``E`` and ``B`` from the nodes."""
    match grid:
        case "collocated":
            return np.zeros((3, 3)), np.zeros((3, 3))
        case "staggered":
            identity = np.eye(3)
            return 0.5 * identity, 0.5 * (1.0 - identity)
        case _:
            raise ValueError("grid is invalid.")


def _absorber_support(
    pml: SpectralPMLPlan, counts: tuple[int, int, int], order: int | None, /
) -> tuple[np.ndarray, np.ndarray | None]:
    """Nodes that may carry absorber charge, and the confinement planes.

    A finite-order stencil of order ``p`` reaches ``p/2`` cells, so the
    divergence the damping creates stays within the layers dilated by ``p/2``
    along their axes. The infinite-order divergence is global; the solver
    confines it to the layers and places the content its derivative cannot see
    (the mean and, on collocated grids, the checkerboard modes) on the two
    outermost planes ``{N − 1, 0}`` of the thickest layer, on which every such
    pattern is orthogonal (returned as a float mask).
    """
    reach = 0 if order is None else order // 2
    inside = np.ones(counts, dtype=np.bool_)
    for axis, layer in enumerate(pml.thickness):
        if layer == 0:
            continue
        index = np.arange(counts[axis])
        keep = (index >= layer + reach) & (index <= counts[axis] - layer - reach)
        shape = [1, 1, 1]
        shape[axis] = counts[axis]
        inside &= keep.reshape(shape)
    if order is not None:
        return ~inside, None
    axis = int(np.argmax(pml.thickness))
    index = np.arange(counts[axis])
    shape = [1, 1, 1]
    shape[axis] = counts[axis]
    outermost = ((index == 0) | (index == counts[axis] - 1)).reshape(shape)
    return ~inside, np.broadcast_to(outermost, counts).astype(np.float64)


def _require_antenna_interior(
    antenna: SampledPlaneCurrentAntennaPlan,
    prepared: PreparedSpectralPlaneAntenna,
    pml: SpectralPMLPlan,
    plan: SpectralMaxwellPlan,
    /,
) -> None:
    """Refuse antenna sheets whose active support enters an absorbing layer."""
    normal = antenna.normal_axis
    count, layer = plan.counts[normal], pml.thickness[normal]
    low, high = (int(value) for value in np.asarray(prepared.evidence.node_plane_range))
    # The band-limited sheet's main lobe spans one cell on either side.
    nodes = np.arange(low - 1, high + 2) % count
    if layer and np.any((nodes < layer) | (nodes > count - layer)):
        raise ValueError(
            "The antenna sheet enters the PML layer along its normal over its "
            "active window."
        )
    for axis, coordinates in zip(
        antenna.tangential_axes,
        (antenna.first_coordinates, antenna.second_coordinates),
        strict=True,
    ):
        count, layer = plan.counts[axis], pml.thickness[axis]
        samples = np.asarray(coordinates)
        position = plan.origin[axis] + np.arange(count) * plan.spacing[axis]
        index = np.flatnonzero((position >= samples[0]) & (position <= samples[-1]))
        if layer and np.any((index < layer) | (index > count - layer)):
            raise ValueError("The antenna aperture reaches into the PML layer.")


class PreparedSpectralMaxwell(AbstractPreparedPICFieldSolver, NonTrainableState):
    """Prepared PSATD field solver bound to the species transfers of one PIC run.

    With a PML, ``absorber_support`` marks the nodes that may carry absorber
    charge: the layers dilated by the stencil half-width for finite-order
    stencils (whose divergence is local), the layers themselves for infinite
    order (whose divergence is not, so the solver confines it; see
    `SpectralPMLPlan`). The Gauss residual ``∇⁻·E − ρ/ε`` vanishes to roundoff
    outside it.
    """

    plan: SpectralMaxwellPlan
    transfers: tuple[PreparedPICParticleCochainTransfer, ...]
    currents: tuple[ChargeConservingCurrentPlan, ...]
    transform: _GlobalTransform
    guarded: _GuardedTransform | None
    spectral: _SpectralGrid
    local: _SpectralGrid | None
    huygens: tuple[PreparedSpectralHuygensBox, ...]
    antennas: tuple[PreparedSpectralPlaneAntenna, ...]
    electric_conductivity: Array | None
    magnetic_conductivity: Array | None
    interior: Array
    absorber_support: Array | None
    confinement_planes: Array | None
    edge_lengths: Array
    maximum_symbol: float = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    field_dtype: RealPrecisionDType = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralMaxwellPlan,
        transfers: Sequence[PreparedPICParticleCochainTransfer],
        currents: Sequence[ChargeConservingCurrentPlan],
        /,
    ) -> None:
        if not isinstance(plan, SpectralMaxwellPlan):
            raise TypeError("plan must be a SpectralMaxwellPlan.")
        transfer_values = tuple(transfers)
        current_values = tuple(currents)
        if len(transfer_values) != len(current_values):
            raise ValueError("One transfer and current plan is required per species.")
        if any(
            not isinstance(value, PreparedPICParticleCochainTransfer)
            for value in transfer_values
        ) or any(
            not isinstance(value, ChargeConservingCurrentPlan) for value in current_values
        ):
            raise TypeError(
                "Spectral PIC transfers and current plans have incompatible types."
            )
        bridge = plan.bridge
        if any(value.bridge.bridge_id != bridge.bridge_id for value in transfer_values):
            raise ValueError("Every PIC transfer must use the spectral plan's bridge.")
        if any(
            current.transfer.prepared_id != transfer.prepared_id
            for current, transfer in zip(current_values, transfer_values, strict=True)
        ):
            raise ValueError("Every current plan must use its matching PIC transfer.")
        counts, spacing = plan.counts, plan.spacing
        staggered = plan.grid == "staggered"
        speed = plan.speed_of_light
        spectral = _spectral_grid(
            counts,
            spacing,
            plan.stencil_order,
            staggered,
            plan.galilean_velocity,
            speed,
            plan.permittivity,
        )
        topology = plan.topology
        schedule = "slab" if len(topology.mesh_shape) == 1 else "pencil"
        lengths = tuple(n * h for n, h in zip(counts, spacing, strict=True))
        transform = _GlobalTransform(
            DistributedSpectralExecutionPlan(
                topology,
                counts,
                schedule=schedule,
                domain_lengths=lengths,
                coefficient_dtype=jnp.complex128,
            ).prepare()
        )
        guarded = None
        local = None
        if plan.subdomains is not None and plan.guard_cells is not None:
            interior = (
                counts[0] // plan.subdomains[0],
                counts[1] // plan.subdomains[1],
                counts[2] // plan.subdomains[2],
            )
            extent = tuple(
                n + 2 * g for n, g in zip(interior, plan.guard_cells, strict=True)
            )
            guarded = _guarded_transform(
                counts, interior, plan.guard_cells, (False, False, False)
            )
            local = _spectral_grid(
                extent,
                spacing,
                plan.stencil_order,
                staggered,
                plan.galilean_velocity,
                speed,
                plan.permittivity,
            )
        electric_offsets, magnetic_offsets = _component_offsets(plan.grid)
        if plan.pml is None:
            electric_sigma = magnetic_sigma = None
            interior_mask = np.ones(counts, dtype=np.bool_)
            support = planes = None
        else:
            electric_sigma = jnp.asarray(
                plan.pml.conductivity(counts, spacing, electric_offsets, speed)
            )
            magnetic_sigma = jnp.asarray(
                plan.pml.conductivity(counts, spacing, magnetic_offsets, speed)
            )
            interior_mask = plan.pml.interior(counts)
            support, planes = _absorber_support(plan.pml, counts, plan.stencil_order)
        plus = [
            _axis_symbols(n, h, plan.stencil_order, staggered).plus
            for n, h in zip(counts, spacing, strict=True)
        ]
        antennas = tuple(
            PreparedSpectralPlaneAntenna(
                value,
                counts,
                spacing,
                plan.origin,
                (electric_offsets, magnetic_offsets),
                (plus[value.tangential_axes[0]], plus[value.tangential_axes[1]]),
                plan.galilean_velocity[value.normal_axis],
            )
            for value in plan.antennas
        )
        if plan.pml is not None:
            for value, prepared in zip(plan.antennas, antennas, strict=True):
                _require_antenna_interior(value, prepared, plan.pml, plan)
        huygens = []
        for box in plan.observers:
            geometry, current = _box_geometry(box, counts, spacing, plan.origin)
            huygens.append(
                PreparedSpectralHuygensBox(
                    geometry=geometry,
                    current_gather=current,
                    acquisition=box.acquisition,
                    exterior=box.exterior,
                    prepared_id=canonical_fingerprint(
                        {
                            "kind": "prepared-spectral-huygens-box",
                            "plan": box.plan_id,
                            "geometry": geometry.geometry_id,
                        }
                    ),
                )
            )
        corner = np.asarray(
            [
                np.max(
                    np.abs(_axis_symbols(n, h, plan.stencil_order, staggered).modified)
                )
                for n, h in zip(counts, spacing, strict=True)
            ]
        )
        self.plan = plan
        self.transfers = transfer_values
        self.currents = current_values
        self.transform = transform
        self.guarded = guarded
        self.spectral = spectral
        self.local = local
        self.huygens = tuple(huygens)
        self.antennas = antennas
        self.electric_conductivity = electric_sigma
        self.magnetic_conductivity = magnetic_sigma
        self.interior = jnp.asarray(interior_mask)
        self.absorber_support = None if support is None else jnp.asarray(support)
        self.confinement_planes = None if planes is None else jnp.asarray(planes)
        self.edge_lengths = jnp.asarray(spacing)
        self.maximum_symbol = float(np.linalg.norm(corner))
        self.spatial_dimension = 3
        self.field_dtype = "float64"
        self.solver_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-maxwell",
                "plan": plan.plan_id,
                "transfers": [value.prepared_id for value in transfer_values],
                "currents": [value.plan_id for value in current_values],
            }
        )

    # -- geometry and core protocol ------------------------------------------------

    @property
    def grid_velocity(self) -> tuple[float, ...]:
        """Velocity of the Galilean grid; particle positions are grid coordinates."""
        return self.plan.galilean_velocity

    @property
    def stable_step(self) -> Array:
        """Largest step with unaliased vacuum dispersion, ``π/(c max|[k]|)``.

        PSATD integrates vacuum Maxwell exactly for any step; beyond this bound
        the highest grid modes alias in time.
        """
        return jnp.asarray(pi / (self.plan.speed_of_light * self.maximum_symbol))

    @property
    def displacement_widths(self) -> Array:
        return self.edge_lengths

    def validate_species(self, species: tuple[PICSpeciesPlan, ...], /) -> None:
        if len(species) != len(self.transfers):
            raise ValueError("Spectral PIC requires one prepared transfer per species.")
        for value, transfer in zip(species, self.transfers, strict=True):
            if (
                value.population.particles.prepared_id
                != transfer.species.particles.prepared_id
            ):
                raise ValueError(
                    "Species population and spectral transfer use different particles."
                )

    def pairing_probe(self, species: int, capacity: int, /) -> tuple[Array, Array]:
        del species
        layer = (0, 0, 0) if self.plan.pml is None else self.plan.pml.thickness
        slot = np.arange(capacity)
        start = np.stack(
            tuple(
                origin + (layer_ + slot % (count - 2 * layer_ - 1) + 0.7) * width
                for origin, layer_, count, width in zip(
                    self.plan.origin,
                    layer,
                    self.plan.counts,
                    self.plan.spacing,
                    strict=True,
                )
            ),
            axis=-1,
        )
        step = np.asarray(
            [
                fraction * width
                for fraction, width in zip(
                    (0.5, 0.2, -0.1), self.plan.spacing, strict=True
                )
            ]
        )
        return jnp.asarray(start), jnp.asarray(start + step)

    def _zeros(self) -> Array:
        return jnp.zeros((*self.plan.counts, 3), dtype=jnp.float64)

    def _state(
        self, electric: Array, magnetic: Array, charge: Array, /
    ) -> SpectralMaxwellState:
        averaged = self.plan.variant == "averaged-galilean"
        split = self.plan.pml is not None
        eye = jnp.eye(3)
        zero = jnp.zeros_like(charge)
        absorber = zero if split else None
        antenna = zero if self.antennas else None
        return SpectralMaxwellState(
            electric=electric,
            magnetic=magnetic,
            charge=charge,
            averaged_electric=electric if averaged else None,
            averaged_magnetic=magnetic if averaged else None,
            electric_split=electric[..., :, None] * eye if split else None,
            magnetic_split=magnetic[..., :, None] * eye if split else None,
            absorber_charge=absorber,
            absorber_magnetic_charge=absorber,
            antenna_charge=antenna,
            antenna_magnetic_charge=antenna,
            observations=tuple(box.initialize() for box in self.huygens),
        )

    def field_with_charge(self, charge: Array, /) -> SpectralMaxwellState:
        return self._state(
            self._zeros(), self._zeros(), jnp.asarray(charge, dtype=jnp.float64)
        )

    def _patterns(self, like: Array, /) -> tuple[Array, ...]:
        """``±1`` nonzero modes the derivative cannot see, shaped like ``like``.

        On collocated grids these are the checkerboard modes whose every axis
        index is zero or Nyquist; staggered grids resolve every nonzero mode.
        """
        if self.plan.grid == "staggered":
            return ()
        signs = []
        for axis, count in enumerate(self.plan.counts):
            shape = [1, 1, 1]
            shape[axis] = count
            signs.append(
                None
                if count % 2
                else jnp.asarray((-1.0) ** np.arange(count)).reshape(shape)
            )
        modes = []
        for pattern in product((0, 1), repeat=3):
            if not any(pattern) or any(
                bit and sign is None for bit, sign in zip(pattern, signs, strict=True)
            ):
                continue
            mode = jnp.ones_like(like)
            for bit, sign in zip(pattern, signs, strict=True):
                if bit and sign is not None:
                    mode = mode * sign
            modes.append(mode)
        return tuple(modes)

    def _unresolved(self, charge: Array, /) -> tuple[Array, Array]:
        """Charge content in the nonzero modes the derivative cannot see, and the mean."""
        mean = jnp.mean(charge)
        projection = jnp.zeros_like(charge)
        for mode in self._patterns(charge):
            projection = projection + jnp.mean(charge * mode) * mode
        return projection, mean

    def _confine(self, created: Array, /) -> Array:
        """Layer-confined copy ``[…, 2]`` of the damping's divergences ``created``.

        ``created`` (electric, magnetic) is restricted to the layers; its content
        along the mean and the unresolved patterns, which the derivative cannot
        carry, is moved onto the confinement planes, where those patterns are
        mutually orthogonal. The result differs from ``created`` by a resolved,
        zero-mean divergence, so a curl-free field realizes the difference.
        """
        support = self.absorber_support
        planes = self.confinement_planes
        if support is None or planes is None:
            raise ValueError("Layer confinement requires an infinite-order PSATD PML.")
        confined = jnp.where(support[..., None], created, 0.0)
        size = jnp.sum(planes)
        for mode in (jnp.ones_like(planes), *self._patterns(planes)):
            content = jnp.sum(confined * mode[..., None], axis=(0, 1, 2))
            confined = confined - content * (mode * planes / size)[..., None]
        return confined

    def _resolved_charge(self, charge: Array, /) -> Array:
        projection, _ = self._unresolved(charge)
        return charge - projection

    def initialize_field(
        self, charge: Array, /, *, magnetic: Any = None
    ) -> tuple[SpectralMaxwellState, Array]:
        """Coulomb field ``E = −∇⁺ρ/(ε[k]²)`` of a neutral periodic charge."""
        rho = jnp.asarray(charge, dtype=jnp.float64)
        coefficients = self.transform.forward(rho[..., None])[..., 0]
        electric = self.transform.inverse(
            self.spectral.operators.coulomb_field(coefficients)
        )
        field = (
            self._zeros()
            if magnetic is None
            else jnp.asarray(magnetic, dtype=jnp.float64)
        )
        if field.shape != (*self.plan.counts, 3):
            raise ValueError("magnetic must have shape (N0, N1, N2, 3).")
        unresolved, mean = self._unresolved(rho)
        # Absolute compatibility tolerance, as the cochain Poisson solve: exactly
        # neutral co-located species leave only roundoff with no relative scale.
        neutral = (jnp.abs(mean) <= _NEUTRALITY_TOLERANCE) & (
            jnp.max(jnp.abs(unresolved), initial=0.0) <= _NEUTRALITY_TOLERANCE
        )
        return self._state(electric, field, rho), neutral & jnp.all(
            jnp.isfinite(electric)
        )

    def field_charge(self, field: SpectralMaxwellState, /) -> Array:
        return field.charge

    def _energy(self, electric: Array, magnetic: Array, /) -> Array:
        volume = float(np.prod(self.plan.spacing))
        return (
            0.5
            * volume
            * (
                self.plan.permittivity * jnp.sum(electric**2)
                + jnp.sum(magnetic**2) / self.plan.permeability
            )
        )

    def field_energy(self, field: SpectralMaxwellState, /) -> Array:
        return self._energy(field.electric, field.magnetic)

    def _momentum(
        self, electric: tuple[Array, ...], magnetic: tuple[Array, ...], /
    ) -> tuple[Array, ...]:
        """``ε∫E×B dV`` of each field pair, ``B`` co-located with ``E``.

        On staggered grids ``(E×B)_a`` pairs ``E_b`` with ``B_c`` (and ``E_c``
        with ``B_b``), which sit half a cell further along ``a``; ``B`` is moved
        back by the band-limited spectral half-cell shift.
        """
        volume = float(np.prod(self.plan.spacing))
        count = len(magnetic)
        match self.plan.grid:
            case "collocated":
                shifted = tuple((value, value, value) for value in magnetic)
            case "staggered":
                coefficients = self.transform.forward(jnp.concatenate(magnetic, axis=-1))
                half = jnp.conj(self.spectral.node_to_current)
                physical = self.transform.inverse(
                    jnp.concatenate(
                        tuple(
                            coefficients * half[..., axis : axis + 1] for axis in range(3)
                        ),
                        axis=-1,
                    )
                )
                width = 3 * count
                shifted = tuple(
                    tuple(
                        physical[
                            ..., axis * width + 3 * index : axis * width + 3 * index + 3
                        ]
                        for axis in range(3)
                    )
                    for index in range(count)
                )
            case _:
                raise ValueError("grid is invalid.")
        return tuple(
            self.plan.permittivity
            * volume
            * jnp.stack(
                tuple(
                    jnp.sum(jnp.cross(field, moved[axis])[..., axis]) for axis in range(3)
                )
            )
            for field, moved in zip(electric, shifted, strict=True)
        )

    # -- deposition -------------------------------------------------------------------

    def _raw_source(
        self,
        species: int,
        start: Array,
        end: Array,
        macrocharge: Array,
        active: Array,
        step_size: Array,
        /,
    ) -> tuple[SpectralMaxwellSource, Array, Array, Array, Array]:
        """Per-sub-interval Esirkepov currents and charge changes of one species.

        Sub-intervals are the straight-path pieces ``[l/m, (l+1)/m]`` (a static,
        bounded ``m`` from the time dependency); returns the source, start
        density, continuity defect, and success.
        """
        count = self.plan.current_intervals
        bridge = self.plan.bridge
        plan = self.currents[species]
        dt = jnp.asarray(step_size, dtype=start.dtype).reshape(())
        currents, changes = [], []
        defect = jnp.zeros((), dtype=start.dtype)
        scale = jnp.zeros((), dtype=start.dtype)
        successful = jnp.asarray(True)
        first_charge = None
        for piece in range(count):
            lower = start + (piece / count) * (end - start)
            upper = start + ((piece + 1) / count) * (end - start)
            result = plan.deposit(
                lower, upper, dt / count, macrocharge=macrocharge, active_mask=active
            )
            components = bridge.unpack(1, result.current)
            currents.append(
                jnp.stack(
                    tuple(
                        value / length
                        for value, length in zip(
                            components, self.plan.spacing, strict=True
                        )
                    ),
                    axis=-1,
                )
            )
            begin = bridge.unpack(0, result.start_charge.cochain)[0]
            finish = bridge.unpack(0, result.end_charge.cochain)[0]
            if first_charge is None:
                first_charge = begin
            changes.append(finish - begin)
            defect = jnp.maximum(defect, result.maximum_continuity_defect)
            scale = jnp.maximum(scale, result.continuity_scale)
            successful = successful & result.successful
        if first_charge is None:
            raise ValueError("A spectral deposit requires at least one sub-interval.")
        return (
            SpectralMaxwellSource(jnp.stack(currents), jnp.stack(changes)),
            first_charge,
            defect,
            scale,
            successful,
        )

    def _finish_deposit(
        self,
        source: SpectralMaxwellSource,
        start_charge: Array,
        defect: Array,
        scale: Array,
        successful: Array,
        /,
    ) -> PICFieldDeposit:
        changes = jnp.stack(
            tuple(self._resolved_charge(value) for value in source.charge_change)
        )
        start = self._resolved_charge(start_charge)
        outside = ~self.interior
        absorber_clear = ~jnp.any(
            jnp.where(outside[None, ..., None], source.current != 0.0, False)
        )
        return PICFieldDeposit(
            SpectralMaxwellSource(source.current, changes),
            start,
            start + jnp.sum(changes, axis=0),
            defect,
            successful & absorber_clear,
            scale,
        )

    def deposit_charge(
        self,
        species: int,
        position: Array,
        macrocharge: Array,
        active: Array,
        /,
    ) -> tuple[Array, Array]:
        transfer = self.transfers[species]
        result = transfer.deposit_macrocharge(
            transfer.build(position, active_mask=active), macrocharge
        )
        density = self.plan.bridge.unpack(0, result.cochain)[0]
        return self._resolved_charge(density), result.successful

    def deposit(
        self,
        species: int,
        start: Array,
        end: Array,
        velocity: Array,
        macrocharge: Array,
        active: Array,
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        """Esirkepov current per sub-interval along ``start → end`` (grid frame).

        ``velocity`` is not needed: the lab current ``J′ + v_gal ρ`` is formed
        from the deposited convective current and charge in `advance`.
        """
        del velocity
        source, charge, defect, scale, successful = self._raw_source(
            species, start, end, macrocharge, active, step_size
        )
        return self._finish_deposit(source, charge, defect, scale, successful)

    def deposit_all(
        self,
        starts: tuple[Array, ...],
        ends: tuple[Array, ...],
        velocities: tuple[Array, ...],
        macrocharges: tuple[Array, ...],
        actives: tuple[Array, ...],
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        """Every species' path currents summed before one resolved-charge projection."""
        del velocities
        raw = tuple(
            self._raw_source(index, *arrays, step_size)
            for index, arrays in enumerate(
                zip(starts, ends, macrocharges, actives, strict=True)
            )
        )
        first, *rest = raw
        source, charge, defect, scale, successful = first
        for other, other_charge, other_defect, other_scale, other_success in rest:
            source = SpectralMaxwellSource(
                source.current + other.current, source.charge_change + other.charge_change
            )
            charge = charge + other_charge
            defect = jnp.maximum(defect, other_defect)
            scale = jnp.maximum(scale, other_scale)
            successful = successful & other_success
        return self._finish_deposit(source, charge, defect, scale, successful)

    # -- advance ----------------------------------------------------------------------

    def _lab_current(
        self,
        grid: _SpectralGrid,
        convective: Array,
        start_charge: Array,
        end_charge: Array,
        /,
    ) -> Array:
        """Solver-layout lab current ``J′ + v_gal ρ̄`` of one sub-interval."""
        current = grid.to_grid * convective
        if self.plan.charge_conservation == "vay-deposition":
            current = current * grid.vay
        if self.plan.variant != "standard":
            velocity = jnp.asarray(self.plan.galilean_velocity)
            mean = 0.5 * (start_charge + end_charge)
            current = current + grid.node_to_current * velocity * mean[..., None]
        return current

    def _propagate(
        self,
        grid: _SpectralGrid,
        electric: Array,
        magnetic: Array,
        convective: Array,
        charges: Array,
        antenna: tuple[Array, Array] | None,
        dt: Array,
        /,
    ) -> tuple[Array, Array, Array | None, Array | None, Array, Array]:
        """Advance spectral ``(E, B)`` over one step; returns averaged fields and ∫E, ∫B.

        ``antenna`` is the step's spectral antenna ``(J, M)``, added whole (the
        sheet's charge is declared, not deposited) to every current interval.
        ``∫E``/``∫B`` are the analytic in-step integrals (with the Gauss-following
        field of ``"update-with-rho"``), formed when a PML or antenna needs them.
        """
        operators = grid.operators
        mode = self.plan.charge_conservation
        count = self.plan.current_intervals
        averaged = self.plan.variant == "averaged-galilean"
        integrate = self.plan.pml is not None or antenna is not None
        sheet, sheet_m = (None, None) if antenna is None else antenna
        integral_e = jnp.zeros_like(electric)
        integral_b = jnp.zeros_like(magnetic)
        average_e = average_b = None
        if self.plan.time_dependency == "linear-j":
            first = self._lab_current(grid, convective[0], charges[0], charges[1])
            second = self._lab_current(grid, convective[1], charges[1], charges[2])
            current = operators.transverse_current(0.5 * (3.0 * first - second))
            slope = operators.transverse_current(2.0 * (second - first) / dt)
            end_e, end_b = exact_interval(
                operators, electric, magnetic, current, slope, dt
            )
            end_e = end_e + gauss_following_increment(
                operators, charges[0], charges[2], dt
            )
            if integrate:
                integral_e, integral_b = window_integral(
                    operators, electric, magnetic, current, slope, jnp.zeros_like(dt), dt
                )
            return end_e, end_b, None, None, integral_e, integral_b
        h = dt / count
        for piece in range(count):
            start_charge, end_charge = charges[piece], charges[piece + 1]
            current = self._lab_current(grid, convective[piece], start_charge, end_charge)
            match mode:
                case "spectral-correction":
                    driven = operators.transverse_current(
                        current
                    ) + galilean_charge_current(operators, start_charge, end_charge, h)
                case "vay-deposition":
                    driven = current
                case "update-with-rho":
                    driven = operators.transverse_current(current)
                case _:
                    raise ValueError("charge_conservation is invalid.")
            if sheet is not None:
                driven = driven + sheet
            next_e, next_b = exact_interval(
                operators, electric, magnetic, driven, None, h, magnetic_current=sheet_m
            )
            if mode == "update-with-rho":
                next_e = next_e + gauss_following_increment(
                    operators, start_charge, end_charge, h
                )
            if integrate:
                window_e, window_b = window_integral(
                    operators,
                    electric,
                    magnetic,
                    driven,
                    None,
                    jnp.zeros_like(h),
                    h,
                    magnetic_current=sheet_m,
                )
                if mode == "update-with-rho":
                    window_e = window_e + gauss_following_window(
                        operators, start_charge, end_charge, h, jnp.zeros_like(h), h
                    )
                integral_e = integral_e + window_e
                integral_b = integral_b + window_b
            if averaged and piece == count - 1:
                lower, upper = h - 0.5 * dt, h + 0.5 * dt
                window_e, window_b = window_integral(
                    operators,
                    electric,
                    magnetic,
                    driven,
                    None,
                    lower,
                    upper,
                    magnetic_current=sheet_m,
                )
                if mode == "update-with-rho":
                    window_e = window_e + gauss_following_window(
                        operators, start_charge, end_charge, h, lower, upper
                    )
                average_e, average_b = window_e / dt, window_b / dt
            electric, magnetic = next_e, next_b
        return electric, magnetic, average_e, average_b, integral_e, integral_b

    def _residuals(
        self, electric: Array, magnetic: Array, charge: Array, magnetic_charge: Array, /
    ) -> tuple[Array, Array]:
        """Spectral Gauss residuals against total electric and magnetic charges."""
        operators = self.spectral.operators
        return (
            operators.electric_divergence(electric)
            - jnp.where(operators.resolved, charge, 0.0) / self.plan.permittivity,
            operators.magnetic_divergence(magnetic)
            - jnp.where(operators.resolved, magnetic_charge, 0.0),
        )

    def _update(
        self,
        engine: _GlobalTransform | _GuardedTransform,
        grid: _SpectralGrid,
        divergence: tuple[_GlobalTransform | _GuardedTransform, _SpectralGrid],
        field: SpectralMaxwellState,
        current: SpectralMaxwellSource,
        dt: Array,
        conductivity: tuple[Array, Array] | None,
        extend: Callable[[Array], Array],
        antennas: tuple[tuple[Array, Array], ...],
        /,
    ) -> SpectralLocalUpdate:
        """One field step of owned fields; ``extend`` adds exchanged guards.

        ``antennas`` are the real-space mid-step ``(J, M)`` of every antenna
        (none on the local-guarded route, which refuses antennas).
        """
        count = self.plan.current_intervals
        shape = field.electric.shape[:3]
        charges_real = field.charge + jnp.concatenate(
            (jnp.zeros((1, *shape)), jnp.cumsum(current.charge_change, axis=0))
        )
        charge = charges_real[-1]
        blocks = [
            field.electric,
            field.magnetic,
            jnp.moveaxis(current.current, 0, -2).reshape((*shape, 3 * count)),
            jnp.moveaxis(charges_real, 0, -1),
        ]
        declared = None
        if antennas:
            if field.antenna_charge is None or field.antenna_magnetic_charge is None:
                raise ValueError("Antenna runs carry both declared sheet charges.")
            declared = jnp.stack(
                (field.antenna_charge, field.antenna_magnetic_charge), axis=-1
            )
            sheet_real, sheet_magnetic_real = antennas[0]
            for value in antennas[1:]:
                sheet_real = sheet_real + value[0]
                sheet_magnetic_real = sheet_magnetic_real + value[1]
            blocks += [sheet_real, sheet_magnetic_real, declared]
        coefficients = engine.forward(extend(jnp.concatenate(blocks, axis=-1)))
        electric = coefficients[..., 0:3]
        magnetic = coefficients[..., 3:6]
        convective = jnp.moveaxis(
            coefficients[..., 6 : 6 + 3 * count].reshape(
                (*coefficients.shape[:-1], count, 3)
            ),
            -2,
            0,
        )
        sources = 7 + 4 * count
        charges = jnp.moveaxis(coefficients[..., 6 + 3 * count : sources], -1, 0)
        sheet = (
            None
            if declared is None
            else (
                coefficients[..., sources : sources + 3],
                coefficients[..., sources + 3 : sources + 6],
            )
        )
        next_e, next_b, average_e, average_b, integral_e, integral_b = self._propagate(
            grid, electric, magnetic, convective, charges, sheet, dt
        )
        outputs = [next_e, next_b]
        if average_e is not None and average_b is not None:
            outputs += [average_e, average_b]
        if self.plan.pml is not None:
            outputs += list(
                _split_increments(
                    grid.operators,
                    electric,
                    magnetic,
                    next_e,
                    next_b,
                    integral_e,
                    integral_b,
                )
            )
        if sheet is not None:
            outputs += [
                _sheet_charges(
                    grid.operators,
                    coefficients[..., sources + 6 : sources + 8],
                    sheet[0],
                    sheet[1],
                    dt,
                ),
                integral_e,
                integral_b,
            ]
        physical = engine.inverse(jnp.concatenate(outputs, axis=-1))
        new_e, new_b = physical[..., 0:3], physical[..., 3:6]
        offset = 6
        averaged_e = averaged_b = None
        if average_e is not None:
            averaged_e, averaged_b = physical[..., 6:9], physical[..., 9:12]
            offset = 12
        electric_split = magnetic_split = undamped_e = undamped_b = None
        if (
            conductivity is not None
            and field.electric_split is not None
            and field.magnetic_split is not None
        ):
            increment_e = physical[..., offset : offset + 9].reshape((*shape, 3, 3))
            increment_b = physical[..., offset + 9 : offset + 18].reshape((*shape, 3, 3))
            offset += 18
            split_e = field.electric_split + increment_e
            split_b = field.magnetic_split + increment_b
            electric_split = split_e * jnp.exp(-conductivity[0] * dt)
            magnetic_split = split_b * jnp.exp(-conductivity[1] * dt)
            new_e = jnp.sum(electric_split, axis=-1)
            new_b = jnp.sum(magnetic_split, axis=-1)
            undamped_e = jnp.sum(split_e, axis=-1)
            undamped_b = jnp.sum(split_b, axis=-1)
        antenna_charge = antenna_magnetic_charge = work = None
        if sheet is not None:
            antenna_charge = physical[..., offset]
            antenna_magnetic_charge = physical[..., offset + 1]
            integral_real = physical[..., offset + 2 : offset + 8]
            volume = float(np.prod(self.plan.spacing))
            # Work on the field: −∫(J·E + M·H) dτ per cell.
            work = -volume * jnp.stack(
                tuple(
                    jnp.sum(value[0] * integral_real[..., 0:3], axis=-1)
                    + jnp.sum(value[1] * integral_real[..., 3:6], axis=-1)
                    / self.plan.permeability
                    for value in antennas
                ),
                axis=-1,
            )
        absorber = None
        if undamped_e is None or undamped_b is None:
            divergences = self._divergences(divergence, new_e, new_b, extend)
        else:
            new_e, new_b, correction, divergences, absorber = self._absorbed_divergences(
                divergence, new_e, new_b, undamped_e, undamped_b, extend
            )
            if (
                correction is not None
                and electric_split is not None
                and magnetic_split is not None
            ):
                eye = jnp.eye(3)
                electric_split = electric_split + correction[..., 0:3, None] * eye
                magnetic_split = magnetic_split + correction[..., 3:6, None] * eye
        absorber_charge = absorber_magnetic_charge = None
        if (
            absorber is not None
            and field.absorber_charge is not None
            and field.absorber_magnetic_charge is not None
        ):
            absorber_charge = (
                field.absorber_charge + self.plan.permittivity * absorber[..., 0]
            )
            absorber_magnetic_charge = field.absorber_magnetic_charge + absorber[..., 1]
        return SpectralLocalUpdate(
            electric=new_e,
            magnetic=new_b,
            charge=charge,
            averaged_electric=averaged_e,
            averaged_magnetic=averaged_b,
            electric_split=electric_split,
            magnetic_split=magnetic_split,
            undamped_electric=undamped_e,
            undamped_magnetic=undamped_b,
            absorber_charge=absorber_charge,
            absorber_magnetic_charge=absorber_magnetic_charge,
            antenna_charge=antenna_charge,
            antenna_magnetic_charge=antenna_magnetic_charge,
            antenna_work=work,
            electric_divergence=divergences[..., 0],
            magnetic_divergence=divergences[..., 1],
        )

    def _divergences(
        self,
        divergence: tuple[_GlobalTransform | _GuardedTransform, _SpectralGrid],
        electric: Array,
        magnetic: Array,
        extend: Callable[[Array], Array],
        /,
    ) -> Array:
        """Real-space ``(∇⁻·E, ∇⁺·B)`` ``[…, 2]`` of the end fields."""
        transform, operators = divergence[0], divergence[1].operators
        fields = transform.forward(extend(jnp.concatenate((electric, magnetic), axis=-1)))
        return transform.inverse(
            jnp.stack(
                (
                    operators.electric_divergence(fields[..., 0:3]),
                    operators.magnetic_divergence(fields[..., 3:6]),
                ),
                axis=-1,
            )
        )

    def _absorbed_divergences(
        self,
        divergence: tuple[_GlobalTransform | _GuardedTransform, _SpectralGrid],
        electric: Array,
        magnetic: Array,
        undamped_electric: Array,
        undamped_magnetic: Array,
        extend: Callable[[Array], Array],
        /,
    ) -> tuple[Array, Array, Array | None, Array, Array]:
        """End fields and divergences, and the PML's created divergences ``[…, 2]``.

        The damping changes ``(E, B)`` by layer-supported increments whose
        discrete divergences ``∇⁻·ΔE``/``∇⁺·ΔB`` are the created absorber
        charge (over ``ε``) and magnetic charge. A finite-order stencil keeps
        them within `absorber_support`. An infinite-order divergence is global:
        it is confined to the layers (`_confine`) and the difference realized
        by a curl-free, hence static and non-radiating, correction of
        ``(E, B)``, also returned ``[…, 6]`` for the diagonal splits.
        """
        transform, operators = divergence[0], divergence[1].operators
        spectral = transform.forward(
            extend(
                jnp.concatenate(
                    (
                        electric,
                        magnetic,
                        electric - undamped_electric,
                        magnetic - undamped_magnetic,
                    ),
                    axis=-1,
                )
            )
        )
        fields_e, fields_b = spectral[..., 0:3], spectral[..., 3:6]
        created = jnp.stack(
            (
                operators.electric_divergence(spectral[..., 6:9]),
                operators.magnetic_divergence(spectral[..., 9:12]),
            ),
            axis=-1,
        )
        if self.plan.stencil_order is not None:
            ends = jnp.stack(
                (
                    operators.electric_divergence(fields_e),
                    operators.magnetic_divergence(fields_b),
                ),
                axis=-1,
            )
            physical = transform.inverse(jnp.concatenate((ends, created), axis=-1))
            return electric, magnetic, None, physical[..., 0:2], physical[..., 2:4]
        created_real = transform.inverse(created)
        confined = self._confine(created_real)
        difference = transform.forward(extend(confined - created_real))
        correction_e = operators.coulomb_field(
            self.plan.permittivity * difference[..., 0]
        )
        correction_b = operators.magnetic_charge_field(difference[..., 1])
        fields_e = fields_e + correction_e
        fields_b = fields_b + correction_b
        ends = jnp.stack(
            (
                operators.electric_divergence(fields_e),
                operators.magnetic_divergence(fields_b),
            ),
            axis=-1,
        )
        physical = transform.inverse(
            jnp.concatenate((correction_e, correction_b, ends), axis=-1)
        )
        correction = physical[..., 0:6]
        return (
            electric + correction[..., 0:3],
            magnetic + correction[..., 3:6],
            correction,
            physical[..., 6:8],
            confined,
        )

    def _conductivity(self) -> tuple[Array, Array] | None:
        if self.electric_conductivity is None or self.magnetic_conductivity is None:
            return None
        return self.electric_conductivity, self.magnetic_conductivity

    def advance(
        self,
        time: Array,
        field: SpectralMaxwellState,
        current: SpectralMaxwellSource,
        step_size: Array,
        /,
    ) -> PICFieldAdvance:
        self._require_source(current)
        dt = jnp.asarray(step_size, dtype=jnp.float64).reshape(())
        engine, grid = (
            (self.transform, self.spectral)
            if self.guarded is None or self.local is None
            else (self.guarded, self.local)
        )
        middle = jnp.asarray(time, dtype=jnp.float64) + 0.5 * dt
        update = self._update(
            engine,
            grid,
            (self.transform, self.spectral),
            field,
            current,
            dt,
            self._conductivity(),
            _unchanged,
            tuple(antenna.sources(middle) for antenna in self.antennas),
        )
        return self.complete_advance(time, field, current, dt, update)

    def _require_source(self, current: SpectralMaxwellSource, /) -> None:
        if not isinstance(current, SpectralMaxwellSource):
            raise TypeError("Spectral Maxwell advances with a SpectralMaxwellSource.")
        if current.current.shape != (self.plan.current_intervals, *self.plan.counts, 3):
            raise ValueError("The deposited source does not match the current intervals.")

    def guarded_update(
        self,
        field: SpectralMaxwellState,
        current: SpectralMaxwellSource,
        step_size: Array,
        conductivity: tuple[Array, Array] | None,
        extend: Callable[[Array], Array],
        padded: tuple[bool, bool, bool],
        /,
    ) -> SpectralLocalUpdate:
        """Local-guarded field step of one device's owned block.

        ``field``/``current``/``conductivity`` hold the owned cells of the
        device's block (whole local-guarded subdomains). ``extend(values)`` returns
        ``values`` with ``guard_cells[a]`` cells exchanged from the neighbors
        added on both sides of every ``padded`` axis; unpadded axes are whole
        periodic axes. Every transform is local to the device. The result feeds
        `complete_advance` once reassembled over the devices.
        """
        guards = self.plan.guard_cells
        subdomains = self.plan.subdomains
        if self.local is None or guards is None or subdomains is None:
            raise ValueError("guarded_update requires local-guarded PSATD.")
        counts = self.plan.counts
        interior = (
            counts[0] // subdomains[0],
            counts[1] // subdomains[1],
            counts[2] // subdomains[2],
        )
        engine = _guarded_transform(field.electric.shape[:3], interior, guards, padded)
        dt = jnp.asarray(step_size, dtype=jnp.float64).reshape(())
        return self._update(
            engine,
            self.local,
            (engine, self.local),
            field,
            current,
            dt,
            conductivity,
            extend,
            (),
        )

    def complete_advance(
        self,
        time: Array,
        field: SpectralMaxwellState,
        current: SpectralMaxwellSource,
        step_size: Array,
        update: SpectralLocalUpdate,
        /,
    ) -> PICFieldAdvance:
        """Constraint, energy, absorber, antenna, and observer evidence of a step.

        The Gauss residuals ``∇⁻·E − (ρ_resolved + ρ_absorber + ρ_antenna)/ε``
        and ``∇⁺·B − ρ_m`` are taken over the whole grid; ``ρ_resolved`` removes
        the mean and the unresolved node-charge modes in real space, exactly as
        on the resolved spectral modes, and the declared charges carry none.
        """
        self._require_source(current)
        dt = jnp.asarray(step_size, dtype=jnp.float64).reshape(())
        new_e, new_b, charge = update.electric, update.magnetic, update.charge
        absorbed = jnp.zeros((), dtype=jnp.float64)
        absorbed_momentum = jnp.zeros((3,), dtype=jnp.float64)
        if update.undamped_electric is not None and update.undamped_magnetic is not None:
            absorbed = self._energy(
                update.undamped_electric, update.undamped_magnetic
            ) - self._energy(new_e, new_b)
            before, after = self._momentum(
                (update.undamped_electric, new_e), (update.undamped_magnetic, new_b)
            )
            absorbed_momentum = before - after
        unresolved, mean = self._unresolved(charge)
        declared, declared_magnetic = _declared_charges(
            charge,
            (update.absorber_charge, update.antenna_charge),
            (update.absorber_magnetic_charge, update.antenna_magnetic_charge),
        )
        residual_e = (
            update.electric_divergence
            - (charge - unresolved - mean + declared) / self.plan.permittivity
        )
        electric_constraint = jnp.max(jnp.abs(residual_e), initial=0.0)
        magnetic_constraint = jnp.max(
            jnp.abs(update.magnetic_divergence - declared_magnetic), initial=0.0
        )
        antenna_work = (
            jnp.zeros((0,), dtype=jnp.float64)
            if update.antenna_work is None
            else jnp.sum(update.antenna_work, axis=(0, 1, 2))
        )
        next_time = jnp.asarray(time, dtype=jnp.float64) + dt
        observations = []
        surface = jnp.zeros((), dtype=jnp.float64)
        for box, state in zip(self.huygens, field.observations, strict=True):
            observations.append(
                box.update(
                    next_time,
                    new_e.reshape((-1,)),
                    (new_b / self.plan.permeability).reshape((-1,)),
                    state,
                )
            )
            active = box.acquisition.active(next_time) | box.acquisition.active(
                jnp.asarray(time, dtype=jnp.float64)
            )
            for piece in current.current:
                surface = jnp.maximum(
                    surface,
                    jnp.where(active, box.surface_current(piece.reshape((-1,))), 0.0),
                )
        state = SpectralMaxwellState(
            electric=new_e,
            magnetic=new_b,
            charge=charge,
            averaged_electric=update.averaged_electric,
            averaged_magnetic=update.averaged_magnetic,
            electric_split=update.electric_split,
            magnetic_split=update.magnetic_split,
            absorber_charge=update.absorber_charge,
            absorber_magnetic_charge=update.absorber_magnetic_charge,
            antenna_charge=update.antenna_charge,
            antenna_magnetic_charge=update.antenna_magnetic_charge,
            observations=tuple(observations),
        )
        energy = self._energy(new_e, new_b)
        finite = jnp.all(jnp.isfinite(new_e)) & jnp.all(jnp.isfinite(new_b))
        diagnostics = SpectralMaxwellDiagnostics(
            electric_constraint=electric_constraint,
            magnetic_constraint=magnetic_constraint,
            energy=energy,
            absorbed_energy=absorbed,
            absorbed_momentum=absorbed_momentum,
            antenna_work=antenna_work,
            unresolved_charge=jnp.max(jnp.abs(unresolved), initial=0.0),
            mean_charge=mean,
            surface_current=surface,
        )
        return PICFieldAdvance(
            state,
            charge,
            electric_constraint,
            magnetic_constraint,
            energy,
            diagnostics,
            finite & (surface == 0.0) & (dt > 0.0),
        )

    def guard_truncation(self, step_size: float, /) -> float:
        """Relative kernel mass of one vacuum PSATD step outside the guard cells.

        Local-guarded PSATD reproduces the global-FFT step on a block interior
        up to the real-space kernel of the step beyond ``guard_cells``: for the
        multipliers ``cos(ωΔt)`` and ``c[k]_a sin(ωΔt)/ω`` (``ω = c|[k]|``, the
        finite-order modified wavenumbers of the whole periodic grid) this
        returns ``max Σ_{outside}|K| / Σ|K|``, the per-step relative stencil
        truncation of the local transforms.
        """
        guards = self.plan.guard_cells
        if guards is None:
            raise ValueError("guard_truncation requires local-guarded PSATD.")
        dt = float(step_size)
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError("step_size must be finite and positive.")
        staggered = self.plan.grid == "staggered"
        speed = self.plan.speed_of_light
        counts = self.plan.counts
        symbols = [
            _axis_symbols(count, width, self.plan.stencil_order, staggered).modified
            for count, width in zip(counts, self.plan.spacing, strict=True)
        ]
        grids = np.meshgrid(*symbols, indexing="ij")
        omega = speed * np.sqrt(sum(value**2 for value in grids))
        safe = np.where(omega > 0.0, omega, 1.0)
        multipliers = [np.cos(omega * dt)] + [
            np.where(omega > 0.0, speed * value * np.sin(omega * dt) / safe, 0.0)
            for value in grids
        ]
        offsets = np.meshgrid(
            *(np.fft.fftfreq(count, d=1.0 / count) for count in counts), indexing="ij"
        )
        outside = np.zeros(counts, dtype=np.bool_)
        for offset, guard in zip(offsets, guards, strict=True):
            outside |= np.abs(offset) > guard
        tail = 0.0
        for multiplier in multipliers:
            kernel = np.abs(np.fft.ifftn(multiplier))
            total = float(np.sum(kernel))
            if total > 0.0:
                tail = max(tail, float(np.sum(kernel[outside])) / total)
        return tail

    # -- gather -------------------------------------------------------------------------

    def gather_fields(
        self,
        species: int,
        position: Array,
        active: Array,
        field: SpectralMaxwellState,
        /,
    ) -> tuple[Array, Array, Array]:
        transfer = self.transfers[species]
        routes = transfer.build(position, active_mask=active)
        electric, magnetic = field.electric, field.magnetic
        if field.averaged_electric is not None and field.averaged_magnetic is not None:
            electric, magnetic = field.averaged_electric, field.averaged_magnetic
        match self.plan.grid:
            case "staggered":
                e = tuple(
                    splat.gather(route, electric[..., axis])
                    for axis, (splat, route) in enumerate(
                        zip(transfer.electric, routes.electric, strict=True)
                    )
                )
                b = tuple(
                    splat.gather(route, magnetic[..., axis])
                    for axis, (splat, route) in enumerate(
                        zip(transfer.magnetic, routes.magnetic, strict=True)
                    )
                )
            case "collocated":
                e = tuple(
                    transfer.charge.gather(routes.charge, electric[..., axis])
                    for axis in range(3)
                )
                b = tuple(
                    transfer.charge.gather(routes.charge, magnetic[..., axis])
                    for axis in range(3)
                )
            case _:
                raise ValueError("grid is invalid.")
        support = jnp.all(jnp.stack(tuple(value.support for value in (*e, *b))), axis=0)
        return (
            jnp.stack(tuple(value.values for value in e), axis=-1),
            jnp.stack(tuple(value.values for value in b), axis=-1),
            support,
        )

    # -- capabilities -------------------------------------------------------------------

    def dispersion_frequency(
        self, wavevector: ArrayLike, step_size: ArrayLike, /
    ) -> Array:
        """Vacuum ``ω = c|[k]|``: PSATD integrates each mode exactly in time.

        ``[k]`` is the solver's derivative symbol (``k`` at infinite order, the
        stencil's modified wavenumber at finite order). Galilean variants report
        the lab-frame frequency. Stepped modes alias for ``c|[k]|Δt > π``.
        """
        dt = float(jnp.asarray(step_size))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("step_size must be positive and finite.")
        k = np.asarray(wavevector, dtype=np.float64)
        if k.ndim != 2 or k.shape[1] != 3:
            raise ValueError("wavevector must have shape (K, 3).")
        staggered = self.plan.grid == "staggered"
        squared = sum(
            modified_wavenumber(
                k[:, axis], self.plan.spacing[axis], self.plan.stencil_order, staggered
            )
            ** 2
            for axis in range(3)
        )
        return jnp.asarray(self.plan.speed_of_light * np.sqrt(squared))

    def huygens_phasors(
        self, field: SpectralMaxwellState, /
    ) -> tuple[HuygensSurfacePhasors, ...]:
        return tuple(
            box.surface_phasors(state)
            for box, state in zip(self.huygens, field.observations, strict=True)
        )

    def restart_component(self, field: SpectralMaxwellState, /) -> PICRestartComponent:
        return restart_component("field", self.solver_id, field)

    def restore_component(
        self, component: PICRestartComponent, /
    ) -> SpectralMaxwellState:
        template = self.field_with_charge(jnp.zeros(self.plan.counts, dtype=jnp.float64))
        return restore_component(component, "field", self.solver_id, template)

    def project_gauss(
        self, field: SpectralMaxwellState, charge: Array, /
    ) -> PICGaussProjectionResult:
        """Spectral Poisson projection ``E ← E + E_c`` onto the Gauss charge ``charge``.

        ``E_c = −∇⁺(ρ + ρ_declared − ε∇⁻·E)/(ε[k]²)`` is curl free; the declared
        absorber and antenna charges, ``B``, absorber splits (``E_c`` is added
        to the diagonal part), and observers are unchanged.
        """
        rho = self._resolved_charge(jnp.asarray(charge, dtype=jnp.float64))
        declared, _ = _declared_charges(
            rho,
            (field.absorber_charge, field.antenna_charge),
            (field.absorber_magnetic_charge, field.antenna_magnetic_charge),
        )
        coefficients = self.transform.forward(
            jnp.concatenate(
                (field.electric, field.magnetic, (rho + declared)[..., None]), axis=-1
            )
        )
        operators = self.spectral.operators
        zero = jnp.zeros_like(coefficients[..., 6])
        before, _ = self._residuals(
            coefficients[..., 0:3], coefficients[..., 3:6], coefficients[..., 6], zero
        )
        correction_k = operators.coulomb_field(-self.plan.permittivity * before)
        after, _ = self._residuals(
            coefficients[..., 0:3] + correction_k,
            coefficients[..., 3:6],
            coefficients[..., 6],
            zero,
        )
        physical = self.transform.inverse(
            jnp.concatenate((correction_k, before[..., None], after[..., None]), axis=-1)
        )
        correction = physical[..., 0:3]
        electric = field.electric + correction
        split = (
            None
            if field.electric_split is None
            else field.electric_split + correction[..., :, None] * jnp.eye(3)
        )
        projected = SpectralMaxwellState(
            electric=electric,
            magnetic=field.magnetic,
            charge=rho,
            averaged_electric=None
            if field.averaged_electric is None
            else field.averaged_electric + correction,
            averaged_magnetic=field.averaged_magnetic,
            electric_split=split,
            magnetic_split=field.magnetic_split,
            absorber_charge=field.absorber_charge,
            absorber_magnetic_charge=field.absorber_magnetic_charge,
            antenna_charge=field.antenna_charge,
            antenna_magnetic_charge=field.antenna_magnetic_charge,
            observations=field.observations,
        )
        return PICGaussProjectionResult(
            projected,
            jnp.max(jnp.abs(physical[..., 3]), initial=0.0),
            jnp.max(jnp.abs(physical[..., 4]), initial=0.0),
            self.field_energy(projected) - self.field_energy(field),
            jnp.all(jnp.isfinite(electric)),
            "spectral-poisson",
        )


def _sheet_charges(
    operators: SpectralOperators,
    charges: Array,
    current: Array,
    magnetic_current: Array,
    duration: Array,
    /,
) -> Array:
    """Declared sheet charges ``[…, (ρ_a, ρ_m)]`` after one step, spectral.

    The exact longitudinal propagator of a constant source:
    ``ρ(h) = e^{iκh}ρ(0) − hφ₁(iκh)∇·S`` with ``∇⁻·J`` and ``∇⁺·M``, the
    divergences the field update puts into ``∇⁻·E`` and ``∇⁺·B``.
    """
    z = 1j * operators.advection * duration
    phase = jnp.exp(z)[..., None]
    first = phi_functions(z, 2)[1][..., None]
    divergence = jnp.stack(
        (
            operators.electric_divergence(current),
            operators.magnetic_divergence(magnetic_current),
        ),
        axis=-1,
    )
    return phase * charges - duration * first * divergence


def _declared_charges(
    like: Array,
    electric: tuple[Array | None, ...],
    magnetic: tuple[Array | None, ...],
    /,
) -> tuple[Array, Array]:
    """Sums of the present declared electric and magnetic charges."""
    total_e = jnp.zeros_like(like)
    total_b = jnp.zeros_like(like)
    for value in electric:
        if value is not None:
            total_e = total_e + value
    for value in magnetic:
        if value is not None:
            total_b = total_b + value
    return total_e, total_b


def _split_increments(
    operators: SpectralOperators,
    electric: Array,
    magnetic: Array,
    next_electric: Array,
    next_magnetic: Array,
    integral_e: Array,
    integral_b: Array,
    /,
) -> tuple[Array, Array]:
    """Split-field increments ``[…, 3·c + a]`` whose sums are the exact updates.

    ``ΔE_{c,a} = c² ε_{cab} D_a⁻ ∫B_b`` and ``ΔB_{c,a} = −ε_{cab} D_a⁺ ∫E_b`` for
    ``a ≠ c``; the diagonal parts carry the remainder (the direct sources).
    """
    speed2 = operators.speed**2

    def split(symbol: Array, integral: Array) -> Array:
        rows = []
        for component in range(3):
            columns = []
            for axis in range(3):
                term = jnp.zeros_like(integral[..., 0])
                for other in range(3):
                    sign = _LEVI_CIVITA[component, axis, other]
                    if sign:
                        term = term + sign * symbol[..., axis] * integral[..., other]
                columns.append(term)
            rows.append(jnp.stack(columns, axis=-1))
        return jnp.stack(rows, axis=-2)

    delta_e = speed2 * split(operators.minus, integral_b)
    delta_b = -split(operators.plus, integral_e)
    eye = jnp.eye(3)
    total_e = next_electric - electric
    total_b = next_magnetic - magnetic
    delta_e = delta_e + (total_e - jnp.sum(delta_e, axis=-1))[..., :, None] * eye
    delta_b = delta_b + (total_b - jnp.sum(delta_b, axis=-1))[..., :, None] * eye
    shape = (*delta_e.shape[:-2], 9)
    return delta_e.reshape(shape), delta_b.reshape(shape)


__all__ = [
    "PreparedSpectralHuygensBox",
    "PreparedSpectralMaxwell",
    "SpectralAbsorber",
    "SpectralChargeConservation",
    "SpectralDecomposition",
    "SpectralGrid",
    "SpectralHuygensBoxPlan",
    "SpectralLocalUpdate",
    "SpectralMaxwellDiagnostics",
    "SpectralMaxwellPlan",
    "SpectralMaxwellSource",
    "SpectralMaxwellState",
    "SpectralMaxwellVariant",
    "SpectralStencil",
    "SpectralTimeDependency",
    "modified_wavenumber",
    "stencil_coefficients",
]
