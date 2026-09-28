#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Solver-native one-way plane current antennas for compatible Maxwell runs.

A sampled plane antenna launches a prescribed forward wave from a sheet normal to
one structured axis ``â``. With emission direction ``s = ±1`` along ``â`` and the
tangential rest-frame fields ``(E', H')`` of the forward wave at the sheet, the
equivalence principle gives the electric and magnetic surface currents

    K = s â × H',    K_m = −s â × E',

which radiate ``(E', H')`` on the emission side and nothing behind the sheet.
On the cochain lattice the electric sheet lies on node planes (edge currents
``j = K ℓ / h``) and the magnetic sheet half a cell upstream on interval centers
(face currents ``m = K_m ℓ``). This is the total-field/scattered-field placement:
each sheet is driven by the incident field at the *other* sheet (``J`` by ``H'``
at the magnetic sheet, ``M`` by ``E'`` at the electric sheet), so the backward
leakage is limited to the mismatch between continuum and lattice plane waves.

Fields are sampled envelopes ``A(x_b, x_c, τ)`` with physical value
``Re[A(τ) e^{−iω₀τ}]`` (``exp(−iωt)`` phasors). A moving antenna translates along
its normal with ``β`` in the simulation frame. Surface currents are four-vector
densities and the boost is along the normal, so the simulation-frame currents
are ``K'(τ)/γ`` with ``τ = t' − s z'/c`` the rest-frame retarded time of each
sheet event, obtained through :func:`phydrax.boost_event`. The launched wave is
then the Lorentz transform of the rest-frame wave, Doppler-shifted by
``γ(1 + sβ)``; a moving antenna therefore requires the vacuum of a declared
:class:`~phydrax.ElectromagneticScaleContract`.
"""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation import (
    cubic_hermite_interpolate,
    linear_interpolate,
    local_cubic_slopes,
)
from .._lorentz import boost_event, boost_wavevector
from .._physical import ElectromagneticScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CochainDiscretization, StructuredCochainBridge
from ..typing import (
    as_array,
    Bool,
    Complex128,
    ConvertibleToArray,
    Dim,
    Float64,
    Int32,
    parse,
    Scalar,
    Scope,
)
from ._maxwell import MaxwellCochainLayout
from ._maxwell_far_field import _surface_material, HomogeneousMaxwellExterior
from ._maxwell_observers import (
    AbstractMaxwellObserverPlan,
    AbstractPreparedMaxwellObserver,
)
from ._maxwell_sources import AbstractMaxwellSourcePlan, MaxwellSourceForcing


if TYPE_CHECKING:
    from ._maxwell import PreparedCompatibleMaxwell


AntennaEmissionDirection: TypeAlias = Literal["positive", "negative"]


class _FirstDim(Dim, minimum=2):
    """Antenna samples along the first tangential grid axis."""


class _SecondDim(Dim, minimum=2):
    """Antenna samples along the second tangential grid axis."""


class _TimeDim(Dim, minimum=2):
    """Rest-frame antenna sample times."""


class _FirstPointDim(Dim, minimum=1):
    """Sheet points (b-center, c-node) carrying J_b and M_c."""


class _SecondPointDim(Dim, minimum=1):
    """Sheet points (b-node, c-center) carrying J_c and M_b."""


class _ElectricEntryDim(Dim, minimum=1):
    """Electric-sheet edge entries on one node plane."""


class _MagneticEntryDim(Dim, minimum=1):
    """Magnetic-sheet face entries on one interval plane."""


def _vacuum_medium(scale: ElectromagneticScaleContract, /) -> HomogeneousMaxwellExterior:
    return HomogeneousMaxwellExterior(
        permittivity=float(scale.vacuum_permittivity),
        permeability=float(scale.vacuum_permeability),
    )


def _resolve_medium(
    medium: HomogeneousMaxwellExterior | None,
    scale: ElectromagneticScaleContract | None,
    beta: float,
    /,
) -> HomogeneousMaxwellExterior:
    if scale is not None and not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract or None.")
    if medium is not None and not isinstance(medium, HomogeneousMaxwellExterior):
        raise TypeError("medium must be a HomogeneousMaxwellExterior or None.")
    if beta != 0.0 and scale is None:
        raise ValueError(
            "A moving antenna requires the vacuum of a declared "
            "ElectromagneticScaleContract; the boost is undefined in a medium."
        )
    if scale is None:
        return HomogeneousMaxwellExterior() if medium is None else medium
    vacuum = _vacuum_medium(scale)
    if medium is None:
        return vacuum
    if not (
        math.isclose(medium.permittivity, vacuum.permittivity, rel_tol=1e-12)
        and math.isclose(medium.permeability, vacuum.permeability, rel_tol=1e-12)
    ):
        raise ValueError(
            "An antenna bound to a scale contract must radiate into that scale's vacuum."
        )
    return medium


def _strictly_increasing(values: Array, name: str, /) -> Array:
    host = np.asarray(values)
    if not np.all(np.isfinite(host)) or not np.all(np.diff(host) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return values


def _finite_complex(values: Array, name: str, /) -> Array:
    host = np.asarray(values)
    if not (np.all(np.isfinite(host.real)) and np.all(np.isfinite(host.imag))):
        raise ValueError(f"{name} must contain finite values.")
    return values


class SampledPlaneCurrentAntennaPlan(AbstractMaxwellSourcePlan, NonTrainableState):
    """One-way sheet antenna driven by sampled tangential rest-frame fields.

    The antenna plane is normal to ``normal_axis`` at ``plane_coordinate`` (the
    rest-frame origin: the sheet passes it at ``t = 0``). The tangential axes are
    the two remaining grid axes in increasing order ``(b, c)``;
    ``first_coordinates``/``second_coordinates`` are the strictly increasing
    sample coordinates along them and ``times`` the rest-frame sample times.
    ``electric[b, c, t, component]`` holds the complex ``(E_b, E_c)`` envelope of
    the forward wave at the sheet; ``magnetic`` holds ``(H_b, H_c)`` and defaults
    to the plane-wave relation ``H' = s â × E' / η`` of ``medium``. Samples outside
    the sampled window are zero: the aperture is the sampled window.

    ``beta`` is the sheet velocity along ``+â`` in units of the vacuum speed of
    light of ``scale``; moving antennas require ``scale``, and ``medium`` then
    defaults to (and must equal) its vacuum. The normal axis must be nonperiodic
    and uniformly spaced; preparation refuses sheets whose active support leaves
    the interior node planes.
    """

    __strict_contract__ = True

    bridge: StructuredCochainBridge
    medium: HomogeneousMaxwellExterior
    first_coordinates: Float64[_FirstDim]
    second_coordinates: Float64[_SecondDim]
    times: Float64[_TimeDim]
    electric: Complex128[_FirstDim, _SecondDim, _TimeDim, Literal[2]]
    magnetic: Complex128[_FirstDim, _SecondDim, _TimeDim, Literal[2]]
    normal_axis: int = eqx.field(static=True)
    plane_coordinate: float = eqx.field(static=True)
    carrier_angular_frequency: float = eqx.field(static=True)
    direction: AntennaEmissionDirection = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    scale: ElectromagneticScaleContract | None = eqx.field(static=True)
    provenance_id: str | None = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        normal_axis: int,
        plane_coordinate: float,
        first_coordinates: ConvertibleToArray,
        second_coordinates: ConvertibleToArray,
        times: ConvertibleToArray,
        electric: ConvertibleToArray,
        /,
        *,
        magnetic: ConvertibleToArray | None = None,
        carrier_angular_frequency: float = 0.0,
        direction: AntennaEmissionDirection = "positive",
        medium: HomogeneousMaxwellExterior | None = None,
        beta: float = 0.0,
        scale: ElectromagneticScaleContract | None = None,
        provenance_id: str | None = None,
    ) -> None:
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be a StructuredCochainBridge.")
        if bridge.dimension != 3:
            raise ValueError("Plane current antennas require a three-dimensional bridge.")
        if isinstance(normal_axis, bool) or normal_axis not in (0, 1, 2):
            raise ValueError("normal_axis must be 0, 1, or 2.")
        axis = bridge.grid.structured_axes[normal_axis]
        if bool(axis.periodic):
            raise ValueError("The antenna normal axis must be nonperiodic.")
        widths = np.asarray(axis.interval_widths, dtype=np.float64)
        if not np.allclose(widths, widths[0], rtol=1e-12, atol=0.0):
            raise ValueError("The antenna normal axis must be uniformly spaced.")
        position = float(plane_coordinate)
        carrier = float(carrier_angular_frequency)
        velocity = float(beta)
        if not math.isfinite(position):
            raise ValueError("plane_coordinate must be finite.")
        if not math.isfinite(carrier) or carrier < 0.0:
            raise ValueError("carrier_angular_frequency must be finite and nonnegative.")
        if not math.isfinite(velocity) or abs(velocity) >= 1.0:
            raise ValueError("beta must be finite with |beta| < 1.")
        emission = parse(direction, AntennaEmissionDirection, "direction")
        resolved_medium = _resolve_medium(medium, scale, velocity)
        if provenance_id is not None and not str(provenance_id):
            raise ValueError("provenance_id must be nonempty when supplied.")
        scope = Scope()
        first = _strictly_increasing(
            as_array(
                first_coordinates, Float64[_FirstDim], "first_coordinates", scope=scope
            ),
            "first_coordinates",
        )
        second = _strictly_increasing(
            as_array(
                second_coordinates, Float64[_SecondDim], "second_coordinates", scope=scope
            ),
            "second_coordinates",
        )
        sample_times = _strictly_increasing(
            as_array(times, Float64[_TimeDim], "times", scope=scope), "times"
        )
        electric_ = _finite_complex(
            as_array(
                electric,
                Complex128[_FirstDim, _SecondDim, _TimeDim, Literal[2]],
                "electric",
                scope=scope,
            ),
            "electric",
        )
        sign = 1.0 if emission == "positive" else -1.0
        tangential_b, tangential_c = (a for a in range(3) if a != normal_axis)
        # ε = Levi-Civita(a, b, c): â × b̂ = ε ĉ and â × ĉ = −ε b̂.
        orientation = (
            1.0 if (normal_axis, tangential_b, tangential_c) != (1, 0, 2) else -1.0
        )
        if magnetic is None:
            # Forward plane wave: H' = s â × E' / η.
            magnetic_ = (
                sign
                * orientation
                * jnp.stack((-electric_[..., 1], electric_[..., 0]), axis=-1)
                / resolved_medium.impedance
            )
        else:
            magnetic_ = _finite_complex(
                as_array(
                    magnetic,
                    Complex128[_FirstDim, _SecondDim, _TimeDim, Literal[2]],
                    "magnetic",
                    scope=scope,
                ),
                "magnetic",
            )
        self.bridge = bridge
        self.medium = resolved_medium
        self.first_coordinates = first
        self.second_coordinates = second
        self.times = sample_times
        self.electric = electric_
        self.magnetic = magnetic_
        self.normal_axis = normal_axis
        self.plane_coordinate = position
        self.carrier_angular_frequency = carrier
        self.direction = emission
        self.beta = velocity
        self.scale = scale
        self.provenance_id = None if provenance_id is None else str(provenance_id)
        self.source_id = canonical_fingerprint(
            {
                "kind": "sampled-plane-current-antenna-plan",
                "bridge": bridge.bridge_id,
                "normal_axis": normal_axis,
                "plane_coordinate": position.hex(),
                "direction": emission,
                "carrier_angular_frequency": carrier.hex(),
                "beta": velocity.hex(),
                "medium": resolved_medium.exterior_id,
                "scale": None if scale is None else scale.scale_id,
                "provenance": self.provenance_id,
                "samples": array_tree_fingerprint(
                    (first, second, sample_times, electric_, magnetic_)
                ),
            }
        )

    @property
    def tangential_axes(self) -> tuple[int, int]:
        first, second = (a for a in range(3) if a != self.normal_axis)
        return (first, second)

    @property
    def emission_sign(self) -> float:
        match self.direction:
            case "positive":
                return 1.0
            case "negative":
                return -1.0

    @property
    def lorentz_factor(self) -> float:
        return 1.0 / math.sqrt(1.0 - self.beta * self.beta)

    def prepare(
        self, bridge: StructuredCochainBridge, layout: Any, /
    ) -> PreparedSampledPlaneCurrentAntenna:
        if not isinstance(layout, MaxwellCochainLayout):
            raise TypeError("Antenna preparation requires a MaxwellCochainLayout.")
        if bridge.bridge_id != self.bridge.bridge_id:
            raise ValueError("The antenna was planned on a different cochain bridge.")
        if layout.polarization != "full_3d":
            raise ValueError("Plane current antennas require full_3d Maxwell.")
        return PreparedSampledPlaneCurrentAntenna(self, layout)


class SampledPlaneAntennaEvidence(StrictModule):
    """Static discretization, support, and kinematic evidence of one antenna.

    ``aperture_support_fraction`` is the fraction of staggered sheet sample points
    inside the sampled window; ``cells_per_wavelength`` resolves the emitted
    carrier ``emitted_carrier_angular_frequency = γ(1 + sβ) ω₀`` along the normal
    (infinite for a zero carrier). ``active_window`` is the simulation-frame time
    interval in which the sheet currents can be nonzero and ``node_plane_range``/
    ``interval_range`` the inclusive node planes and intervals they touch.
    ``magnetic_closure_defect`` is the relative discrete divergence of the magnetic
    sheet: nonzero exactly when the launched field has a normal ``B`` component,
    whose jump across the sheet is the magnetic surface charge of the equivalent
    source. The runtime tracks that charge as declared magnetic charge rather than
    projecting it out.
    """

    __strict_contract__ = True

    aperture_support_fraction: Float64[Scalar]
    emitted_carrier_angular_frequency: Float64[Scalar]
    cells_per_wavelength: Float64[Scalar]
    lorentz_factor: Float64[Scalar]
    magnetic_closure_defect: Float64[Scalar]
    active_window: Float64[Literal[2]]
    node_plane_range: Int32[Literal[2]]
    interval_range: Int32[Literal[2]]
    aperture_supported: Bool[Scalar]


def _transverse_samples(
    plan: SampledPlaneCurrentAntennaPlan,
    first_query: np.ndarray,
    second_query: np.ndarray,
    component: int,
    field: Array,
    /,
) -> tuple[Array, Array]:
    """Bilinearly map ``field[..., component]`` onto a tensor of sheet points.

    Returns ``values[time, point]`` with points ordered first-query-major and the
    matching support mask; points outside the sampled window carry zero.
    """
    along_first = linear_interpolate(
        plan.first_coordinates,
        field[..., component],
        jnp.asarray(first_query),
        axis=0,
        bounds="fill",
    )
    along_second = linear_interpolate(
        plan.second_coordinates,
        jnp.moveaxis(along_first.values, 1, 0),
        jnp.asarray(second_query),
        axis=0,
        bounds="fill",
    )
    support = along_first.support[:, None] & along_second.support[None, :]
    values = jnp.transpose(along_second.values, (2, 1, 0)).reshape((field.shape[2], -1))
    return values, support.reshape(-1)


class _SheetGeometry(NamedTuple):
    """Host-prepared plane-0 cochain entries, coefficients, and sheet samples.

    Sheet points of the first kind (b-center, c-node) carry ``J_b`` and ``M_c`` and
    sample ``(E'_b, H'_c)``; points of the second kind (b-node, c-center) carry
    ``J_c`` and ``M_b`` and sample ``(E'_c, H'_b)``. Electric entries are ordered
    ``(J_b, J_c)`` and magnetic entries ``(M_b, M_c)``. Commutator coefficients
    multiply the lab-frame incident field of the paired event (``H`` for ``J``,
    ``E`` for ``M``); convective coefficients multiply ``∂Θ/∂t`` times the
    co-located incident field (``D`` for ``J``, ``B`` for ``M``).
    """

    first_samples: Array
    second_samples: Array
    electric_indices: np.ndarray
    electric_strides: np.ndarray
    electric_commutator: np.ndarray
    electric_convective: np.ndarray
    magnetic_indices: np.ndarray
    magnetic_strides: np.ndarray
    magnetic_commutator: np.ndarray
    magnetic_convective: np.ndarray
    support: Array
    closure_defect: float


def _sheet_geometry(plan: SampledPlaneCurrentAntennaPlan, /) -> _SheetGeometry:
    """Lower the sheet onto plane-0 cochain entries and staggered sheet samples."""
    bridge = plan.bridge
    a = plan.normal_axis
    b, c = plan.tangential_axes
    sign = plan.emission_sign
    orientation = 1.0 if (a, b, c) != (1, 0, 2) else -1.0
    permittivity, permeability = plan.medium.permittivity, plan.medium.permeability
    axes = bridge.grid.structured_axes
    height = float(np.asarray(axes[a].interval_widths)[0])
    nodes = {
        axis: np.asarray(axes[axis].point_coordinates, dtype=np.float64)[
            : bridge.orientation_shapes[0][0][axis]
        ]
        for axis in (b, c)
    }
    centers = {
        axis: np.asarray(axes[axis].interval_centers, dtype=np.float64) for axis in (b, c)
    }
    widths = {
        axis: np.asarray(axes[axis].interval_widths, dtype=np.float64) for axis in (b, c)
    }
    edge_shapes, edge_offsets = (
        bridge.orientation_shapes[1],
        bridge.orientation_offsets[1],
    )
    face_orientations = bridge.orientations[2]
    face_shapes, face_offsets = (
        bridge.orientation_shapes[2],
        bridge.orientation_offsets[2],
    )

    def packed(
        shape: tuple[int, ...], offset: int, cell_axis: int, node_axis: int
    ) -> tuple[np.ndarray, np.ndarray]:
        index: list[np.ndarray] = [np.zeros(1, dtype=np.int64)] * 3
        cells = np.arange(shape[cell_axis])
        points = np.arange(shape[node_axis])
        grid_cell, grid_node = np.meshgrid(cells, points, indexing="ij")
        index[cell_axis] = grid_cell.reshape(-1)
        index[node_axis] = grid_node.reshape(-1)
        index[a] = np.zeros(grid_cell.size, dtype=np.int64)
        flat = offset + np.ravel_multi_index(tuple(index), shape)
        stride = math.prod(shape[a + 1 :])
        return flat.astype(np.int32), np.full(flat.shape, stride, dtype=np.int32)

    def face(normal: int) -> tuple[tuple[int, ...], int, float]:
        orientation_key = tuple(sorted((a, ({b, c} - {normal}).pop())))
        position = face_orientations.index(orientation_key)
        # Packed (p, q) faces carry +B_r for cyclic (p, q, r), −B_r for (0, 2, 1).
        levi_civita = -1.0 if orientation_key == (0, 2) else 1.0
        return face_shapes[position], face_offsets[position], levi_civita

    # Components are (b, c) = (0, 1); query order matches the packed (cell, node)
    # meshgrid of the first-kind entries.
    e_b_first, support_first = _transverse_samples(
        plan, centers[b], nodes[c], 0, plan.electric
    )
    h_c_first, _ = _transverse_samples(plan, centers[b], nodes[c], 1, plan.magnetic)
    e_c_second, support_second = _transverse_samples(
        plan, nodes[b], centers[c], 1, plan.electric
    )
    h_b_second, _ = _transverse_samples(plan, nodes[b], centers[c], 0, plan.magnetic)
    length_b = np.repeat(widths[b], nodes[c].size)
    length_c = np.tile(widths[c], nodes[b].size)

    j_b_index, j_b_stride = packed(edge_shapes[b], edge_offsets[b], b, c)
    j_c_index, j_c_stride = packed(edge_shapes[c], edge_offsets[c], c, b)
    # Second-kind entries are raveled (c-cell, b-node); reorder to (b-node, c-cell).
    second_order = (
        np.arange(nodes[b].size * centers[c].size)
        .reshape(centers[c].size, nodes[b].size)
        .T.reshape(-1)
    )
    j_c_index, j_c_stride = j_c_index[second_order], j_c_stride[second_order]
    face_b_shape, face_b_offset, sigma_b = face(b)
    face_c_shape, face_c_offset, sigma_c = face(c)
    m_b_index, m_b_stride = packed(face_b_shape, face_b_offset, c, b)
    m_b_index, m_b_stride = m_b_index[second_order], m_b_stride[second_order]
    m_c_index, m_c_stride = packed(face_c_shape, face_c_offset, b, c)
    # K = s â × H (K_b = −sε H_c, K_c = sε H_b), edge cochain j = K ℓ / h; the
    # convective current −D ∂Θ/∂t has edge cochain −ε E ℓ ∂Θ/∂t.
    electric_commutator = np.concatenate(
        (-sign * orientation * length_b / height, sign * orientation * length_c / height)
    )
    electric_convective = -permittivity * np.concatenate((length_b, length_c))
    # K_m = −s â × E (K_m,b = sε E_c, K_m,c = −sε E_b), face cochain m = σ K_m ℓ;
    # the convective current −B ∂Θ/∂t has face cochain −σ μ H h ℓ ∂Θ/∂t.
    magnetic_commutator = np.concatenate(
        (
            sign * orientation * sigma_b * length_c,
            -sign * orientation * sigma_c * length_b,
        )
    )
    magnetic_convective = (
        -permeability * height * np.concatenate((sigma_b * length_c, sigma_c * length_b))
    )
    # Exact discrete divergence of the magnetic sheet on its interval layer: the
    # outward face fluxes K_m,b ℓ_c and K_m,c ℓ_b differenced across each cell.
    flux_b = (e_c_second * jnp.asarray(length_c)).reshape(
        (-1, nodes[b].size, centers[c].size)
    )
    flux_c = -(e_b_first * jnp.asarray(length_b)).reshape(
        (-1, centers[b].size, nodes[c].size)
    )

    def forward_difference(values: Array, axis: int, periodic: bool) -> Array:
        if periodic:
            return jnp.roll(values, -1, axis=axis) - values
        return jnp.diff(values, axis=axis)

    divergence = forward_difference(flux_b, 1, bool(axes[b].periodic)) + (
        forward_difference(flux_c, 2, bool(axes[c].periodic))
    )
    flux_scale = float(jnp.maximum(jnp.max(jnp.abs(flux_b)), jnp.max(jnp.abs(flux_c))))
    closure_defect = (
        0.0 if flux_scale == 0.0 else float(jnp.max(jnp.abs(divergence))) / flux_scale
    )
    return _SheetGeometry(
        jnp.stack((e_b_first, h_c_first), axis=-1),
        jnp.stack((e_c_second, h_b_second), axis=-1),
        np.concatenate((j_b_index, j_c_index)),
        np.concatenate((j_b_stride, j_c_stride)),
        electric_commutator,
        electric_convective,
        np.concatenate((m_b_index, m_c_index)),
        np.concatenate((m_b_stride, m_c_stride)),
        magnetic_commutator,
        magnetic_convective,
        jnp.concatenate((support_first, support_second)),
        closure_defect,
    )


class PreparedSampledPlaneCurrentAntenna(StrictModule, NonTrainableState):
    """Prepared antenna: a smoothed moving total-field/scattered-field boundary.

    Implements the prepared Maxwell source contract. The simulation-frame field is
    ``Θ·F_inc`` with ``F_inc`` the Lorentz transform of the rest-frame wave and
    ``Θ`` a step smoothed by the quadratic B-spline of the sheet position over its
    three nearest node planes. The lattice sources of that field are the
    commutator of the curl with ``Θ`` (one exact TFSF pair per plane: ``J`` on node
    ``k`` driven by ``H`` at the upstream interval, ``M`` on that interval by ``E``
    on node ``k``, weighted by the B-spline) plus the convective currents
    ``−D ∂Θ/∂t`` and ``−B ∂Θ/∂t`` of the moving boundary. Every incident value is
    evaluated at the rest-frame retarded time of its own event. A static sheet is a
    constant superposition of exact one-way pairs.
    """

    __strict_contract__ = True

    times: Float64[_TimeDim]
    first_samples: Complex128[_TimeDim, _FirstPointDim, Literal[2]]
    first_slopes: Complex128[_TimeDim, _FirstPointDim, Literal[2]]
    second_samples: Complex128[_TimeDim, _SecondPointDim, Literal[2]]
    second_slopes: Complex128[_TimeDim, _SecondPointDim, Literal[2]]
    electric_indices: Int32[_ElectricEntryDim]
    electric_strides: Int32[_ElectricEntryDim]
    electric_commutator: Float64[_ElectricEntryDim]
    electric_convective: Float64[_ElectricEntryDim]
    magnetic_indices: Int32[_MagneticEntryDim]
    magnetic_strides: Int32[_MagneticEntryDim]
    magnetic_commutator: Float64[_MagneticEntryDim]
    magnetic_convective: Float64[_MagneticEntryDim]
    evidence: SampledPlaneAntennaEvidence
    medium: HomogeneousMaxwellExterior
    orientation: float = eqx.field(static=True)
    normal_axis: int = eqx.field(static=True)
    emission_sign: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    lorentz_factor: float = eqx.field(static=True)
    boost_speed: float = eqx.field(static=True)
    carrier_angular_frequency: float = eqx.field(static=True)
    plane_coordinate: float = eqx.field(static=True)
    first_node: float = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    node_plane_range: tuple[int, int] = eqx.field(static=True)
    interval_range: tuple[int, int] = eqx.field(static=True)
    interval_shift: int = eqx.field(static=True)
    electric_count: int = eqx.field(static=True)
    magnetic_count: int = eqx.field(static=True)
    bridge_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: SampledPlaneCurrentAntennaPlan, layout: MaxwellCochainLayout, /
    ) -> None:
        geometry = _sheet_geometry(plan)
        closure_defect = geometry.closure_defect
        axis = plan.bridge.grid.structured_axes[plan.normal_axis]
        nodes = np.asarray(axis.point_coordinates, dtype=np.float64)
        spacing = float(np.asarray(axis.interval_widths)[0])
        sign = plan.emission_sign
        gamma = plan.lorentz_factor
        speed = (
            plan.medium.wave_speed
            if plan.scale is None
            else float(plan.scale.speed_of_light)
        )
        times = np.asarray(plan.times, dtype=np.float64)
        # τ = t/γ at the sheet; pair events sit within 1.5 h of it (at most
        # γ(1 + |β|)·1.5h/c in retarded time), bounding the active window.
        retardation = gamma * (1.0 + abs(plan.beta)) * 1.5 * spacing / speed
        window = (gamma * (times[0] - retardation), gamma * (times[-1] + retardation))
        coordinates = (
            plan.plane_coordinate + plan.beta * speed * np.asarray(window) - nodes[0]
        ) / spacing
        centers = np.round(coordinates).astype(np.int64)
        node_range = (int(centers.min()) - 1, int(centers.max()) + 1)
        # M pairs with the interval upstream of each node: k − 1 for s = +1, k for −1.
        interval_shift = -1 if sign > 0.0 else 0
        interval_range = (node_range[0] + interval_shift, node_range[1] + interval_shift)
        if node_range[0] < 1 or node_range[1] > nodes.size - 2:
            raise ValueError(
                "The antenna electric sheet must stay on interior node planes "
                f"1..{nodes.size - 2} over its active window; it touches {node_range}."
            )
        if interval_range[0] < 0 or interval_range[1] > nodes.size - 2:
            raise ValueError(
                "The antenna magnetic sheet leaves the grid intervals over its "
                "active window."
            )
        normal = np.zeros(3)
        normal[plan.normal_axis] = 1.0
        # The simulation frame moves with −β relative to the antenna rest frame.
        emitted = boost_wavevector(
            jnp.asarray(-plan.beta * normal),
            jnp.asarray(plan.carrier_angular_frequency),
            jnp.asarray(sign * normal),
        )[0]
        wavelength_cells = jnp.where(
            emitted > 0.0,
            2.0
            * jnp.pi
            * plan.medium.wave_speed
            / jnp.maximum(emitted, 1e-300)
            / spacing,
            jnp.inf,
        )
        support = geometry.support
        support_fraction = jnp.mean(support.astype(jnp.float64))
        evidence = SampledPlaneAntennaEvidence(
            support_fraction,
            jnp.asarray(emitted, dtype=jnp.float64),
            jnp.asarray(wavelength_cells, dtype=jnp.float64),
            jnp.asarray(gamma, dtype=jnp.float64),
            jnp.asarray(closure_defect, dtype=jnp.float64),
            jnp.asarray(window, dtype=jnp.float64),
            jnp.asarray(node_range, dtype=jnp.int32),
            jnp.asarray(interval_range, dtype=jnp.int32),
            jnp.all(support),
        )
        time_nodes = plan.times
        self.times = time_nodes
        self.first_samples = geometry.first_samples
        self.first_slopes = local_cubic_slopes(time_nodes, geometry.first_samples, axis=0)
        self.second_samples = geometry.second_samples
        self.second_slopes = local_cubic_slopes(
            time_nodes, geometry.second_samples, axis=0
        )
        self.electric_indices = jnp.asarray(geometry.electric_indices)
        self.electric_strides = jnp.asarray(geometry.electric_strides)
        self.electric_commutator = jnp.asarray(geometry.electric_commutator)
        self.electric_convective = jnp.asarray(geometry.electric_convective)
        self.magnetic_indices = jnp.asarray(geometry.magnetic_indices)
        self.magnetic_strides = jnp.asarray(geometry.magnetic_strides)
        self.magnetic_commutator = jnp.asarray(geometry.magnetic_commutator)
        self.magnetic_convective = jnp.asarray(geometry.magnetic_convective)
        self.evidence = evidence
        self.medium = plan.medium
        tangential_b, tangential_c = plan.tangential_axes
        self.orientation = (
            1.0 if (plan.normal_axis, tangential_b, tangential_c) != (1, 0, 2) else -1.0
        )
        self.normal_axis = plan.normal_axis
        self.emission_sign = sign
        self.beta = plan.beta
        self.lorentz_factor = gamma
        self.boost_speed = speed
        self.carrier_angular_frequency = plan.carrier_angular_frequency
        self.plane_coordinate = plan.plane_coordinate
        self.first_node = float(nodes[0])
        self.spacing = spacing
        self.node_plane_range = node_range
        self.interval_range = interval_range
        self.interval_shift = interval_shift
        self.electric_count = layout.electric_count
        self.magnetic_count = layout.magnetic_count
        self.bridge_id = plan.bridge.bridge_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-sampled-plane-current-antenna",
                "source": plan.source_id,
                "layout": layout.layout_id,
            }
        )

    def _support_indices(
        self, indices: Array, strides: Array, lower: int, upper: int, /
    ) -> np.ndarray:
        base = np.asarray(indices, dtype=np.int64)
        stride = np.asarray(strides, dtype=np.int64)
        planes = np.arange(lower, upper + 1, dtype=np.int64)
        return np.unique((base[None, :] + planes[:, None] * stride[None, :]).reshape(-1))

    @property
    def electric_support(self) -> np.ndarray:
        """Every electric cochain index the sheet can drive."""
        return self._support_indices(
            self.electric_indices, self.electric_strides, *self.node_plane_range
        )

    @property
    def magnetic_support(self) -> np.ndarray:
        """Every magnetic cochain index the sheet can drive."""
        return self._support_indices(
            self.magnetic_indices, self.magnetic_strides, *self.interval_range
        )

    def validate_runtime(self, prepared: PreparedCompatibleMaxwell, /) -> None:
        """Refuse runtimes whose antenna support is not the declared medium or
        intersects the CPML region."""
        if prepared.plan.bridge.bridge_id != self.bridge_id:
            raise ValueError("The antenna was prepared on a different cochain bridge.")
        electric, magnetic = self.electric_support, self.magnetic_support
        if prepared.pml is not None:
            absorbed_electric = np.concatenate(
                tuple(np.asarray(term.indices) for term in prepared.pml.electric_terms)
                or (np.zeros((0,), dtype=np.int32),)
            )
            absorbed_magnetic = np.concatenate(
                tuple(np.asarray(term.indices) for term in prepared.pml.magnetic_terms)
                or (np.zeros((0,), dtype=np.int32),)
            )
            if np.any(np.isin(electric, absorbed_electric)) or np.any(
                np.isin(magnetic, absorbed_magnetic)
            ):
                raise ValueError("The antenna sheet intersects the CPML region.")
        _surface_material(
            prepared.constitutive,
            electric,
            magnetic,
            self.medium.permittivity,
            self.medium.permeability,
            owner="Antenna sheet",
        )

    def _stencil(self, time: Array, /) -> tuple[Array, Array, Array, Array]:
        """Quadratic B-spline sheet stencil on its three nearest node planes.

        Returns the planes, their weights (the jumps of the smoothed step ``Θ``
        across each plane's TFSF pair), and ``∂Θ/∂t`` on each node and on each
        node's paired upstream interval.
        """
        coordinate = (
            self.plane_coordinate + self.beta * self.boost_speed * time - self.first_node
        ) / self.spacing
        center = jnp.clip(
            jnp.round(coordinate),
            self.node_plane_range[0] + 1,
            self.node_plane_range[1] - 1,
        )
        offset = jnp.clip(coordinate - center, -0.5, 0.5)
        weights = jnp.stack(
            (0.5 * (0.5 - offset) ** 2, 0.75 - offset**2, 0.5 * (0.5 + offset) ** 2)
        )
        # Θ accumulates the weights from the scattered (upstream) side; its node
        # derivatives in the sheet coordinate u are (−(½−x), −(½+x), 0) for s = +1
        # and (0, ½−x, ½+x) for s = −1, and each upstream interval takes the Θ of
        # its upstream node.
        zero = jnp.zeros_like(offset)
        rate = self.beta * self.boost_speed / self.spacing
        if self.emission_sign > 0.0:
            node_rate = jnp.stack((-(0.5 - offset), -(0.5 + offset), zero))
            interval_rate = jnp.stack((zero, -(0.5 - offset), -(0.5 + offset)))
        else:
            node_rate = jnp.stack((zero, 0.5 - offset, 0.5 + offset))
            interval_rate = jnp.stack((0.5 - offset, 0.5 + offset, zero))
        planes = (center + jnp.asarray([-1.0, 0.0, 1.0])).astype(jnp.int32)
        return planes, weights, rate * node_rate, rate * interval_rate

    def _retarded_times(self, time: Array, positions: Array, /) -> Array:
        """Rest-frame retarded times ``τ = t' − s z'/c`` of events on ``positions``."""
        speed = self.boost_speed
        events = jnp.zeros(positions.shape + (4,), dtype=jnp.float64)
        events = (
            events.at[..., 0]
            .set(speed * time)
            .at[..., 1 + self.normal_axis]
            .set(positions - self.plane_coordinate)
        )
        velocity = jnp.zeros((3,), dtype=jnp.float64).at[self.normal_axis].set(self.beta)
        rest = boost_event(velocity, events)
        return (
            rest[..., 0] / speed
            - self.emission_sign * rest[..., 1 + self.normal_axis] / speed
        )

    def _incident(self, retarded: Array, /) -> tuple[Array, Array]:
        """Simulation-frame tangential incident fields at the given events.

        Returns ``(E, H)`` ordered like the electric entries ``(J_b, J_c)``: ``E_b``
        and ``H_c`` on first-kind points, ``E_c`` and ``H_b`` on second-kind points.
        With ``v = βc â`` the tangential fields transform as
        ``E = γ(E' − v × B')`` and ``B = γ(B' + v × E'/c²)``.
        """
        carrier = jnp.exp(-1j * self.carrier_angular_frequency * retarded)

        def rest(samples: Array, slopes: Array) -> Array:
            envelope = cubic_hermite_interpolate(
                self.times, samples, retarded, slopes=slopes, bounds="fill"
            ).values
            return jnp.real(envelope * carrier[:, None, None])

        first = rest(self.first_samples, self.first_slopes)
        second = rest(self.second_samples, self.second_slopes)
        impedance = self.medium.impedance
        tilt = self.beta * self.orientation
        gamma = self.lorentz_factor
        electric = gamma * jnp.concatenate(
            (
                first[..., 0] + tilt * impedance * first[..., 1],
                second[..., 0] - tilt * impedance * second[..., 1],
            ),
            axis=-1,
        )
        magnetic = gamma * jnp.concatenate(
            (
                first[..., 1] + tilt * first[..., 0] / impedance,
                second[..., 1] - tilt * second[..., 0] / impedance,
            ),
            axis=-1,
        )
        return electric, magnetic

    def sample(self, time: ArrayLike, args: object = None, /) -> MaxwellSourceForcing:
        del args
        time_ = jnp.asarray(time, dtype=jnp.float64)
        if time_.shape != ():
            raise ValueError("Antenna sample time must be scalar.")
        planes, weights, node_rate, interval_rate = self._stencil(time_)
        nodes = self.first_node + self.spacing * planes.astype(jnp.float64)
        upstream = nodes - self.emission_sign * 0.5 * self.spacing
        node_electric, _ = self._incident(self._retarded_times(time_, nodes))
        _, interval_magnetic = self._incident(self._retarded_times(time_, upstream))
        first = self.first_samples.shape[1]
        # Magnetic entries are ordered (M_b, M_c): second-kind points first.
        node_electric_faces = jnp.concatenate(
            (node_electric[:, first:], node_electric[:, :first]), axis=-1
        )
        interval_magnetic_faces = jnp.concatenate(
            (interval_magnetic[:, first:], interval_magnetic[:, :first]), axis=-1
        )
        electric_values = (
            weights[:, None] * self.electric_commutator * interval_magnetic
            + node_rate[:, None] * self.electric_convective * node_electric
        )
        magnetic_values = (
            weights[:, None] * self.magnetic_commutator * node_electric_faces
            + interval_rate[:, None] * self.magnetic_convective * interval_magnetic_faces
        )
        intervals = planes + self.interval_shift
        electric = (
            jnp.zeros((self.electric_count,), dtype=jnp.float64)
            .at[self.electric_indices[None, :] + planes[:, None] * self.electric_strides]
            .add(electric_values)
        )
        magnetic = (
            jnp.zeros((self.magnetic_count,), dtype=jnp.float64)
            .at[
                self.magnetic_indices[None, :]
                + intervals[:, None] * self.magnetic_strides
            ]
            .add(magnetic_values)
        )
        return MaxwellSourceForcing(electric, magnetic)


class MaxwellAntennaWorkState(StrictModule):
    """Streaming trapezoidal work ledger of one antenna."""

    __strict_contract__ = True

    previous_time: Float64[Scalar]
    previous_electric_power: Float64[Scalar]
    previous_magnetic_power: Float64[Scalar]
    electric_work: Float64[Scalar]
    magnetic_work: Float64[Scalar]
    first_power: Float64[Scalar]
    samples: Int32[Scalar]


class MaxwellAntennaWorkEvidence(StrictModule):
    """Work done on the field by the electric and magnetic antenna sheets.

    ``electric_work = −∫ E·J dV dt`` and ``magnetic_work = −∫ H·M dV dt`` are
    trapezoidal integrals over the synchronized step-end fields; the first
    observed step has no left endpoint, so ``first_power`` (the power delivered
    at the first observed time) must be negligible for a complete ledger.
    """

    __strict_contract__ = True

    electric_work: Float64[Scalar]
    magnetic_work: Float64[Scalar]
    total_work: Float64[Scalar]
    first_power: Float64[Scalar]
    samples: Int32[Scalar]


class MaxwellAntennaWorkObserverPlan(AbstractMaxwellObserverPlan):
    """Streaming ledger of the work an antenna in the same run does on the field."""

    antenna: SampledPlaneCurrentAntennaPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, antenna: SampledPlaneCurrentAntennaPlan, /) -> None:
        if not isinstance(antenna, SampledPlaneCurrentAntennaPlan):
            raise TypeError("antenna must be a SampledPlaneCurrentAntennaPlan.")
        self.antenna = antenna
        self.plan_id = canonical_fingerprint(
            {"kind": "maxwell-antenna-work-observer-plan", "antenna": antenna.source_id}
        )

    def prepare(self, layout: Any, /) -> PreparedMaxwellAntennaWorkObserver:
        return PreparedMaxwellAntennaWorkObserver(
            self, self.antenna.prepare(self.antenna.bridge, layout)
        )


class PreparedMaxwellAntennaWorkObserver(AbstractPreparedMaxwellObserver):
    """Prepared antenna work ledger over the runtime's Hodge metrics."""

    antenna: PreparedSampledPlaneCurrentAntenna
    cochain: CochainDiscretization
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MaxwellAntennaWorkObserverPlan,
        antenna: PreparedSampledPlaneCurrentAntenna,
        /,
    ) -> None:
        self.antenna = antenna
        self.cochain = plan.antenna.bridge.cochain
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-antenna-work-observer",
                "plan": plan.plan_id,
                "antenna": antenna.prepared_id,
            }
        )

    def validate_runtime(self, prepared: PreparedCompatibleMaxwell, /) -> None:
        if not any(
            isinstance(source, PreparedSampledPlaneCurrentAntenna)
            and source.prepared_id == self.antenna.prepared_id
            for source in prepared.sources
        ):
            raise ValueError("The observed antenna does not drive this Maxwell run.")

    def initialize(self, /) -> MaxwellAntennaWorkState:
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        return MaxwellAntennaWorkState(
            zero, zero, zero, zero, zero, zero, jnp.asarray(0, dtype=jnp.int32)
        )

    def _powers(self, time: Array, electric: Array, magnetic: Array, /) -> Array:
        forcing = self.antenna.sample(time)
        # Power delivered to the field is −(E·⋆J + H·⋆M).
        return -jnp.stack(
            (
                jnp.real(
                    jnp.vdot(
                        electric, self.cochain.apply_hodge(1, forcing.electric_current)
                    )
                ),
                jnp.real(
                    jnp.vdot(
                        magnetic, self.cochain.apply_hodge(2, forcing.magnetic_current)
                    )
                ),
            )
        ).astype(jnp.float64)

    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
    ) -> MaxwellAntennaWorkState:
        if not isinstance(state, MaxwellAntennaWorkState):
            raise TypeError("Antenna work observer requires MaxwellAntennaWorkState.")
        time_ = jnp.asarray(time, dtype=jnp.float64)
        power = self._powers(time_, electric, magnetic)
        started = state.samples > 0
        half_step = 0.5 * (time_ - state.previous_time)
        electric_increment = jnp.where(
            started, half_step * (state.previous_electric_power + power[0]), 0.0
        )
        magnetic_increment = jnp.where(
            started, half_step * (state.previous_magnetic_power + power[1]), 0.0
        )
        return MaxwellAntennaWorkState(
            time_,
            power[0],
            power[1],
            state.electric_work + electric_increment,
            state.magnetic_work + magnetic_increment,
            jnp.where(started, state.first_power, power[0] + power[1]),
            state.samples + 1,
        )

    def value(self, state: Any, /) -> Array:
        """Return the total work done on the field."""
        return self.evidence(state).total_work

    def evidence(self, state: Any, /) -> MaxwellAntennaWorkEvidence:
        if not isinstance(state, MaxwellAntennaWorkState):
            raise TypeError("Antenna work observer requires MaxwellAntennaWorkState.")
        return MaxwellAntennaWorkEvidence(
            state.electric_work,
            state.magnetic_work,
            state.electric_work + state.magnetic_work,
            state.first_power,
            state.samples,
        )


__all__ = [
    "AntennaEmissionDirection",
    "MaxwellAntennaWorkEvidence",
    "MaxwellAntennaWorkObserverPlan",
    "MaxwellAntennaWorkState",
    "PreparedMaxwellAntennaWorkObserver",
    "PreparedSampledPlaneCurrentAntenna",
    "SampledPlaneAntennaEvidence",
    "SampledPlaneCurrentAntennaPlan",
]
