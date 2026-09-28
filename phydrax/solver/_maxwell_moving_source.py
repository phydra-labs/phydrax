#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frequency-domain Maxwell fields of a charge in uniform rectilinear motion.

The transformed current of one charge on the path ``r(s) = r₀ + s d̂`` (time
``t = s/v``) is ``J̃(r, ω) = q d̂ δ_⊥ exp(i ω s / v)``. Its Galerkin load on the
lowest-order Whitney edge forms, ``b_e = q ∫ W_e(r(s))·d̂ exp(i ω s/v) ds``, is
integrated exactly: on each cell the path crosses, ``W_e·d̂`` is a polynomial of
degree ``dimension − 1`` in the path parameter, so the segment integral reduces
to closed-form moments ``∫₀¹ τᵐ exp(iθτ) dτ``. The same construction on the
Whitney node forms gives the transformed charge ``b₀``; the pair satisfies the
discrete continuity law ``d₀ᵀ b + i ω b₀ = 0`` at every node the path does not
end on, to roundoff. The primal current cochain is ``⋆₁⁻¹ b``.

A path along a periodic axis of length ``L`` is one closed pass; it requires
``ω L / v ∈ 2πℤ`` and then equals the transform of a single charge on an
infinite line. Any other path is clipped to the domain and its ends are
reported as open endpoints (charge created or absorbed there).

``"total-field"`` solves ``A E = iω J̃``. ``"scattered-field"`` takes the
analytic uniform-motion field ``E_inc`` of a declared homogeneous background and
solves ``A E_s = P(A_b E_inc − A P E_inc)`` with ``E_s = −E_inc`` on perfect
conductors (``P`` removes conductor rows), so only material contrast and
conductors radiate; the path itself never enters the discrete problem.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._polynomial._orthogonal import legendre_rule_data
from .._strict import StrictModule
from ..discretization import StructuredCochainBridge
from ..ein import contract
from ..electromagnetics._uniform_motion_field import (
    UniformMotionFieldPlan,
    UniformMotionGeometry,
    UniformMotionMedium,
)
from ..linalg import LinearSolvePolicy
from ..sparse import EdgeRelation, SparseColoring, SparseLinearMap
from ..typing import (
    Bool,
    Complex128,
    ConvertibleToArray,
    Dim,
    Float64,
    parse,
    Scalar,
    Scope,
)
from ._maxwell import AbstractPreparedMaxwellConstitutive, MaxwellCochainLayout
from ._maxwell_boundaries import MaxwellBoundaryPlan
from ._maxwell_frequency import (
    FrequencyMaxwellOperator,
    FrequencyMaxwellPowerLedger,
    FrequencyMaxwellSolveMethod,
    FrequencyMaxwellSolveResult,
)
from ._maxwell_pml import MaxwellCPMLPlan


SourceFormulation: TypeAlias = Literal["total-field", "scattered-field"]

# Series/recurrence switch for the moments ∫₀¹ τᵐ exp(iθτ) dτ: below it the
# upward recurrence would amplify rounding by up to m!/|θ|ᵐ.
_MOMENT_SERIES_LIMIT = 1.0
_MOMENT_SERIES_TERMS = 26
_MOMENT_COUNT = 4
# Edge quadrature for the incident field and the closest admissible approach of
# a scattered-field source entity to the charge path, in local edge lengths.
_EDGE_QUADRATURE_NODES = 16
_MINIMUM_SOURCE_DISTANCE = 0.25
_COMMENSURABILITY_TOLERANCE = 1e-8
_PARALLEL_TOLERANCE = 1e-12


class _AxisDim(Dim, minimum=2):
    """Ambient coordinates of the structured complex."""


class _EdgeContributionDim(Dim, minimum=1):
    """Path-segment × Whitney-edge contributions."""


class _NodeContributionDim(Dim, minimum=1):
    """Path-segment × Whitney-node contributions."""


class _EdgeDim(Dim, minimum=1):
    """Electric cochain entries."""


class _NodeDim(Dim, minimum=1):
    """Node cochain entries."""


class _FaceDim(Dim, minimum=1):
    """Magnetic cochain entries."""


class _MomentDim(Dim, minimum=1):
    """Polynomial moment orders."""


def _real(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real.")
    result = array.astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def _moments(theta: Array, /) -> Array:
    """``M_m(θ) = ∫₀¹ τᵐ exp(iθτ) dτ`` for ``m < 4``, stable at every ``θ``."""
    z = 1j * theta.astype(jnp.complex128)
    small = jnp.abs(theta) < _MOMENT_SERIES_LIMIT
    safe = jnp.where(small, 1.0, z)
    exponential = jnp.exp(z)
    recurrence = [(exponential - 1.0) / safe]
    for order in range(1, _MOMENT_COUNT):
        recurrence.append((exponential - order * recurrence[-1]) / safe)
    # Σₙ (iθ)ⁿ / (n! (n + m + 1)).
    powers = jnp.ones_like(z)
    series = [jnp.zeros_like(z) for _ in range(_MOMENT_COUNT)]
    factorial = 1.0
    for term in range(_MOMENT_SERIES_TERMS):
        if term:
            powers = powers * z
            factorial = factorial * term
        for order in range(_MOMENT_COUNT):
            series[order] = series[order] + powers / (factorial * (term + order + 1))
    return jnp.stack(
        [
            jnp.where(small, series[order], recurrence[order])
            for order in range(_MOMENT_COUNT)
        ],
        axis=-1,
    )


class _GridGeometry:
    """Host view of a structured bridge: nodes, widths, periodicity, indexing."""

    def __init__(self, bridge: StructuredCochainBridge, /) -> None:
        axes = bridge.grid.structured_axes
        self.dimension = bridge.dimension
        self.periodic = tuple(bool(axis.periodic) for axis in axes)
        self.widths = tuple(
            np.asarray(axis.interval_widths, dtype=np.float64) for axis in axes
        )
        points = tuple(
            np.asarray(axis.point_coordinates, dtype=np.float64) for axis in axes
        )
        # Cell boundaries including the closing node of periodic axes.
        self.boundaries = tuple(
            np.concatenate((point[:1], point[0] + np.cumsum(width)))
            for point, width in zip(points, self.widths, strict=True)
        )
        self.lower = np.asarray([edges[0] for edges in self.boundaries])
        self.upper = np.asarray([edges[-1] for edges in self.boundaries])
        self.length = self.upper - self.lower
        self.node_shape = bridge.orientation_shapes[0][0]
        self.edge_shapes = bridge.orientation_shapes[1]
        self.edge_offsets = bridge.orientation_offsets[1]

    def cell(self, axis: int, coordinate: float, /) -> tuple[int, float]:
        """Cell index and lower corner containing ``coordinate`` (wrapped if periodic)."""
        value = coordinate
        if self.periodic[axis]:
            value = self.lower[axis] + (coordinate - self.lower[axis]) % self.length[axis]
        edges = self.boundaries[axis]
        index = int(
            np.clip(np.searchsorted(edges, value, side="right") - 1, 0, edges.size - 2)
        )
        return index, value - edges[index]

    def node(self, axis: int, index: int, /) -> int:
        return index % self.node_shape[axis] if self.periodic[axis] else index

    def edge_index(self, axis: int, index: tuple[int, ...], /) -> int:
        return self.edge_offsets[axis] + int(
            np.ravel_multi_index(index, self.edge_shapes[axis])
        )

    def node_index(self, index: tuple[int, ...], /) -> int:
        return int(np.ravel_multi_index(index, self.node_shape))


def _path_breakpoints(
    grid: _GridGeometry, origin: np.ndarray, direction: np.ndarray, /
) -> tuple[np.ndarray, int | None, int]:
    """Sorted path parameters of every cell crossing, periodic axis, open ends."""
    moving = tuple(
        axis
        for axis in range(grid.dimension)
        if abs(direction[axis]) > _PARALLEL_TOLERANCE
    )
    periodic_moving = tuple(axis for axis in moving if grid.periodic[axis])
    for axis in range(grid.dimension):
        if axis in moving or grid.periodic[axis]:
            continue
        if not grid.lower[axis] <= origin[axis] <= grid.upper[axis]:
            raise ValueError("The charge path lies outside the structured domain.")
    if periodic_moving:
        if len(moving) != 1:
            raise ValueError(
                "A path with a component along a periodic axis must be parallel to it."
            )
        axis = periodic_moving[0]
        values = (grid.boundaries[axis] - origin[axis]) / direction[axis]
        return np.sort(values), axis, 0
    entries = np.asarray(
        [(grid.lower[axis] - origin[axis]) / direction[axis] for axis in moving]
    )
    exits = np.asarray(
        [(grid.upper[axis] - origin[axis]) / direction[axis] for axis in moving]
    )
    start = float(np.max(np.minimum(entries, exits)))
    stop = float(np.min(np.maximum(entries, exits)))
    if not stop > start:
        raise ValueError("The charge path does not cross the structured domain.")
    crossings = [np.asarray([start, stop])]
    for axis in moving:
        values = (grid.boundaries[axis] - origin[axis]) / direction[axis]
        crossings.append(values[(values > start) & (values < stop)])
    return np.unique(np.concatenate(crossings)), None, 2


def _linear_factor(alpha: float, beta: float, high: bool, /) -> np.ndarray:
    """Coefficients (ascending in τ) of ``ξ`` or ``1 − ξ`` for ``ξ = α + βτ``."""
    return np.asarray([alpha, beta]) if high else np.asarray([1.0 - alpha, -beta])


def _padded(polynomial: np.ndarray, /) -> np.ndarray:
    return np.pad(polynomial, (0, _MOMENT_COUNT - polynomial.size))


class _WhitneyPath:
    """Host Whitney edge/node contributions of the straight path."""

    def __init__(
        self,
        grid: _GridGeometry,
        origin: np.ndarray,
        direction: np.ndarray,
        breakpoints: np.ndarray,
        /,
    ) -> None:
        dimension = grid.dimension
        edge_rows: list[int] = []
        edge_coefficients: list[np.ndarray] = []
        edge_segments: list[int] = []
        node_rows: list[int] = []
        node_coefficients: list[np.ndarray] = []
        node_segments: list[int] = []
        starts: list[float] = []
        lengths: list[float] = []
        cells: list[tuple[int, ...]] = []
        scale = max(float(np.max(np.abs(breakpoints))), 1.0)
        for start, stop in zip(breakpoints[:-1], breakpoints[1:], strict=True):
            length = float(stop - start)
            if length <= 64.0 * np.finfo(np.float64).eps * scale:
                continue
            middle = origin + 0.5 * (start + stop) * direction
            index: list[int] = []
            alpha: list[float] = []
            beta: list[float] = []
            for axis in range(dimension):
                cell, offset = grid.cell(axis, float(middle[axis]))
                width = float(grid.widths[axis][cell])
                index.append(cell)
                slope = direction[axis] * length / width
                alpha.append(offset / width - 0.5 * slope)
                beta.append(slope)
            segment = len(starts)
            starts.append(float(start))
            lengths.append(length)
            cells.append(tuple(index))
            for axis in range(dimension):
                if abs(direction[axis]) <= _PARALLEL_TOLERANCE:
                    continue
                others = tuple(other for other in range(dimension) if other != axis)
                weight = direction[axis] / float(grid.widths[axis][index[axis]])
                for sides in np.ndindex(*((2,) * len(others))):
                    polynomial = np.asarray([weight])
                    node = list(index)
                    for other, side in zip(others, sides, strict=True):
                        polynomial = np.polynomial.polynomial.polymul(
                            polynomial,
                            _linear_factor(alpha[other], beta[other], bool(side)),
                        )
                        node[other] = grid.node(other, index[other] + side)
                    edge_rows.append(grid.edge_index(axis, tuple(node)))
                    edge_coefficients.append(_padded(polynomial))
                    edge_segments.append(segment)
            for sides in np.ndindex(*((2,) * dimension)):
                polynomial = np.asarray([1.0])
                node = []
                for axis, side in enumerate(sides):
                    polynomial = np.polynomial.polynomial.polymul(
                        polynomial, _linear_factor(alpha[axis], beta[axis], bool(side))
                    )
                    node.append(grid.node(axis, index[axis] + side))
                node_rows.append(grid.node_index(tuple(node)))
                node_coefficients.append(_padded(polynomial))
                node_segments.append(segment)
        if not starts:
            raise ValueError("The charge path has no segment inside the domain.")
        self.edge_rows = np.asarray(edge_rows, dtype=np.int32)
        self.edge_coefficients = np.stack(edge_coefficients)
        self.edge_segments = np.asarray(edge_segments, dtype=np.int32)
        self.node_rows = np.asarray(node_rows, dtype=np.int32)
        self.node_coefficients = np.stack(node_coefficients)
        self.node_segments = np.asarray(node_segments, dtype=np.int32)
        self.starts = np.asarray(starts)
        self.lengths = np.asarray(lengths)
        self.first_cell = cells[0]
        self.last_cell = cells[-1]


class MaxwellMovingChargePlan(StrictModule):
    """Charge in uniform rectilinear motion on a structured compatible complex.

    ``full_3d`` layouts carry a point charge ``q``; ``tez`` layouts a line charge
    ``λ`` per unit length along the invariant ``z`` axis moving in the plane.
    ``origin`` is the position at ``t = 0`` and ``direction`` the motion.
    """

    __strict_contract__ = True

    bridge: StructuredCochainBridge
    layout: MaxwellCochainLayout
    charge: Float64[Scalar]
    speed: Float64[Scalar]
    origin: Float64[_AxisDim]
    direction: Float64[_AxisDim]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        layout: MaxwellCochainLayout,
        /,
        *,
        charge: ConvertibleToArray,
        speed: ConvertibleToArray,
        origin: ConvertibleToArray,
        direction: ConvertibleToArray,
    ) -> None:
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be a StructuredCochainBridge.")
        if not isinstance(layout, MaxwellCochainLayout):
            raise TypeError("layout must be a MaxwellCochainLayout.")
        if layout.polarization == "tmz":
            raise ValueError(
                "A moving charge carries in-plane current; use the tez or full_3d layout."
            )
        if layout.electric_count != bridge.cochain.cell_counts[1]:
            raise ValueError("Maxwell layout does not belong to the bridge.")
        charge_ = _real(charge, "charge")
        speed_ = _real(speed, "speed")
        origin_ = _real(origin, "origin")
        direction_ = _real(direction, "direction")
        if charge_.shape != () or speed_.shape != ():
            raise ValueError("charge and speed must be scalars.")
        if not speed_ > 0.0:
            raise ValueError("speed must be positive.")
        if origin_.shape != (bridge.dimension,) or direction_.shape != (
            bridge.dimension,
        ):
            raise ValueError("origin and direction need one entry per structured axis.")
        norm = float(np.linalg.norm(direction_))
        if norm == 0.0:
            raise ValueError("direction must be nonzero.")
        scope = Scope()
        self.bridge = bridge
        self.layout = layout
        self.charge = parse(jnp.asarray(charge_), Float64[Scalar], "charge", scope=scope)
        self.speed = parse(jnp.asarray(speed_), Float64[Scalar], "speed", scope=scope)
        self.origin = parse(
            jnp.asarray(origin_), Float64[_AxisDim], "origin", scope=scope
        )
        self.direction = parse(
            jnp.asarray(direction_ / norm), Float64[_AxisDim], "direction", scope=scope
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-moving-charge-plan",
                "bridge": bridge.bridge_id,
                "layout": layout.layout_id,
                "values": array_tree_fingerprint(
                    (charge_, speed_, origin_, direction_ / norm)
                ),
            }
        )

    def prepare(self) -> PreparedMaxwellMovingCharge:
        return PreparedMaxwellMovingCharge(self)


class PreparedMaxwellMovingCharge(StrictModule):
    """Exact Whitney current and charge loads of one uniformly moving charge.

    ``periodic_length`` is the closed-pass length along a periodic axis (``None``
    for a clipped path); ``open_endpoints`` counts path ends inside the domain
    trace, where charge is created or absorbed; ``endpoint_nodes`` marks the
    nodes of the cells holding those ends.
    """

    __strict_contract__ = True

    plan: MaxwellMovingChargePlan
    edge_coefficients: Float64[_EdgeContributionDim, _MomentDim]
    edge_starts: Float64[_EdgeContributionDim]
    edge_lengths: Float64[_EdgeContributionDim]
    node_coefficients: Float64[_NodeContributionDim, _MomentDim]
    node_starts: Float64[_NodeContributionDim]
    node_lengths: Float64[_NodeContributionDim]
    edge_scatter: SparseLinearMap
    node_scatter: SparseLinearMap
    endpoint_nodes: Bool[_NodeDim]
    path_edges: Bool[_EdgeDim]
    periodic_length: Float64[Scalar] | None
    open_endpoints: int = eqx.field(static=True)
    periodic_axis: int | None = eqx.field(static=True)
    path_length: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MaxwellMovingChargePlan, /) -> None:
        grid = _GridGeometry(plan.bridge)
        origin = np.asarray(plan.origin)
        direction = np.asarray(plan.direction)
        breakpoints, periodic_axis, open_endpoints = _path_breakpoints(
            grid, origin, direction
        )
        path = _WhitneyPath(grid, origin, direction, breakpoints)
        edge_count = plan.layout.electric_count
        node_count = plan.bridge.cochain.cell_counts[0]
        endpoint_nodes = np.zeros((node_count,), dtype=np.bool_)
        if open_endpoints:
            for cell in (path.first_cell, path.last_cell):
                for sides in np.ndindex(*((2,) * grid.dimension)):
                    node = tuple(
                        grid.node(axis, cell[axis] + side)
                        for axis, side in enumerate(sides)
                    )
                    endpoint_nodes[grid.node_index(node)] = True
        path_edges = np.zeros((edge_count,), dtype=np.bool_)
        path_edges[path.edge_rows] = True
        scope = Scope()
        self.plan = plan
        self.edge_coefficients = parse(
            jnp.asarray(path.edge_coefficients),
            Float64[_EdgeContributionDim, _MomentDim],
            "edge_coefficients",
            scope=scope,
        )
        self.edge_starts = parse(
            jnp.asarray(path.starts[path.edge_segments]),
            Float64[_EdgeContributionDim],
            "edge_starts",
            scope=scope,
        )
        self.edge_lengths = parse(
            jnp.asarray(path.lengths[path.edge_segments]),
            Float64[_EdgeContributionDim],
            "edge_lengths",
            scope=scope,
        )
        self.node_coefficients = parse(
            jnp.asarray(path.node_coefficients),
            Float64[_NodeContributionDim, _MomentDim],
            "node_coefficients",
            scope=scope,
        )
        self.node_starts = parse(
            jnp.asarray(path.starts[path.node_segments]),
            Float64[_NodeContributionDim],
            "node_starts",
            scope=scope,
        )
        self.node_lengths = parse(
            jnp.asarray(path.lengths[path.node_segments]),
            Float64[_NodeContributionDim],
            "node_lengths",
            scope=scope,
        )
        self.edge_scatter = _scatter(path.edge_rows, edge_count, f"{plan.plan_id}:edges")
        self.node_scatter = _scatter(path.node_rows, node_count, f"{plan.plan_id}:nodes")
        self.endpoint_nodes = parse(
            jnp.asarray(endpoint_nodes), Bool[_NodeDim], "endpoint_nodes", scope=scope
        )
        self.path_edges = parse(
            jnp.asarray(path_edges), Bool[_EdgeDim], "path_edges", scope=scope
        )
        self.periodic_axis = periodic_axis
        self.periodic_length = (
            None
            if periodic_axis is None
            else parse(
                jnp.asarray(grid.length[periodic_axis]),
                Float64[Scalar],
                "periodic_length",
                scope=scope,
            )
        )
        self.open_endpoints = open_endpoints
        self.path_length = float(np.sum(path.lengths))
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-maxwell-moving-charge", "plan": plan.plan_id}
        )

    def _commensurate(self, angular_frequency: Array, /) -> Array:
        omega = jnp.asarray(angular_frequency, dtype=jnp.float64)
        if self.periodic_length is None:
            return omega
        # A closed pass is the single-charge transform only when the path phase
        # exp(iωL/v) returns to one.
        cycles = omega * self.periodic_length / (2.0 * jnp.pi * self.plan.speed)
        return eqx.error_if(
            omega,
            jnp.abs(cycles - jnp.round(cycles))
            > _COMMENSURABILITY_TOLERANCE * jnp.maximum(cycles, 1.0),
            "A periodic path requires ω L / v to be an integer multiple of 2π.",
        )

    def _load(
        self,
        omega: Array,
        coefficients: Array,
        starts: Array,
        lengths: Array,
        scatter: SparseLinearMap,
        /,
    ) -> Array:
        wavenumber = omega / self.plan.speed
        moments = _moments(wavenumber * lengths)
        polynomial = jnp.sum(coefficients * moments, axis=-1)
        values = lengths * jnp.exp(1j * wavenumber * starts) * polynomial
        return scatter.mv(values)

    def edge_load(self, angular_frequency: ConvertibleToArray, /) -> Array:
        """Galerkin load ``b_e = q ∫ W_e·d̂ exp(iωs/v) ds`` on electric entries."""
        omega = self._commensurate(jnp.asarray(angular_frequency))
        return self.plan.charge * self._load(
            omega,
            self.edge_coefficients,
            self.edge_starts,
            self.edge_lengths,
            self.edge_scatter,
        )

    def node_load(self, angular_frequency: ConvertibleToArray, /) -> Array:
        """Galerkin charge ``b₀ = (q/v) ∫ W_n exp(iωs/v) ds`` on node entries."""
        omega = self._commensurate(jnp.asarray(angular_frequency))
        return (
            self.plan.charge
            / self.plan.speed
            * self._load(
                omega,
                self.node_coefficients,
                self.node_starts,
                self.node_lengths,
                self.node_scatter,
            )
        )

    def current(self, angular_frequency: ConvertibleToArray, /) -> Array:
        """Primal electric-current cochain ``J̃ = ⋆₁⁻¹ b``; the source is ``iω J̃``."""
        cochain = self.plan.bridge.cochain
        return cochain.solve_hodge(
            self.plan.layout.electric_degree, self.edge_load(angular_frequency)
        )

    def continuity_defect(self, angular_frequency: ConvertibleToArray, /) -> Array:
        """Largest ``|d₀ᵀ b + iω b₀|`` off the endpoint nodes, relative to ``|ω b₀|``."""
        omega = jnp.asarray(angular_frequency, dtype=jnp.float64)
        incidence = self.plan.bridge.cochain.topology.incidences[0].exterior_derivative()
        charge = self.node_load(omega)
        divergence = incidence.transpose_mv(self.edge_load(omega))
        defect = jnp.where(
            self.endpoint_nodes, 0.0, jnp.abs(divergence + 1j * omega * charge)
        )
        return jnp.max(defect) / jnp.maximum(
            jnp.max(jnp.abs(omega * charge)), jnp.finfo(jnp.float64).tiny
        )


def _scatter(rows: np.ndarray, size: int, identifier: str, /) -> SparseLinearMap:
    return SparseLinearMap(
        EdgeRelation(
            np.arange(rows.size, dtype=np.int32),
            rows.astype(np.int32),
            source_size=rows.size,
            target_size=size,
        ),
        np.ones((rows.size,), dtype=np.float64),
        operator_id=identifier,
    )


class FrequencyMovingChargeEvidence(StrictModule):
    """Branch, domain-size, source, and solve evidence of one frequency.

    ``bound_extent = 2π/Im k_ρ`` is the transverse reach ``γβλ`` of the charge's
    bound field in the medium seen by the path (infinite when radiating
    losslessly); ``transverse_clearance`` is the distance from the path to the
    nearest absorbing layer or truncating wall across the motion, and
    ``bound_field_contained`` requires it to be at least ``bound_extent`` for
    bound (non-radiating) frequencies. ``continuity_defect`` is the Whitney
    continuity residual (total field) and ``source_distance`` the closest
    approach of a scattered-field source entity to the path in local edge
    lengths (scattered field).
    """

    __strict_contract__ = True

    angular_frequency: Float64[Scalar]
    transverse_wavenumber: Complex128[Scalar]
    radiating: Bool[Scalar]
    bound_extent: Float64[Scalar]
    transverse_clearance: Float64[Scalar]
    bound_field_contained: Bool[Scalar]
    continuity_defect: Float64[Scalar]
    source_distance: Float64[Scalar]
    converged: Bool[Scalar]
    residual_norm: Float64[Scalar]
    iterations: Array
    open_endpoints: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    formulation: SourceFormulation = eqx.field(static=True)
    method: FrequencyMaxwellSolveMethod = eqx.field(static=True)


class FrequencyMovingChargeResult(StrictModule):
    """Total field, optional scattered/incident split, ledger, and evidence.

    ``electric`` is the total edge-circulation phasor ``Ẽ(ω) = ∫ E e^{iωt} dt``
    of the single charge and ``magnetic_flux = dẼ/(iω)`` its face-flux phasor.
    In the scattered-field formulation the total is ``E_s + E_inc`` off
    conductors and zero on them; edges within a quarter edge of the path carry
    no supported incident value and are NaN in ``incident_electric`` and the
    total (and in the adjacent magnetic fluxes). ``ledger`` balances the solved
    unknown: the total field for ``"total-field"`` (``(2/π)·source_power`` is
    the one-sided ``dW/dω`` the field extracts from the charge) and the
    scattered field for ``"scattered-field"`` (``(2/π)·absorbed_power`` is the
    one-sided ``dW/dω`` radiated into the absorbing layers).
    """

    __strict_contract__ = True

    electric: Complex128[_EdgeDim]
    magnetic_flux: Complex128[_FaceDim]
    scattered_electric: Complex128[_EdgeDim] | None
    incident_electric: Complex128[_EdgeDim] | None
    source: Complex128[_EdgeDim]
    solve: FrequencyMaxwellSolveResult
    ledger: FrequencyMaxwellPowerLedger
    evidence: FrequencyMovingChargeEvidence


def _homogeneous_scalar(values: Array, probe: Array, name: str, /) -> complex:
    ratio = np.asarray(values / probe)
    reference = complex(ratio[0])
    if not np.allclose(ratio, reference, rtol=1e-10, atol=0.0):
        raise ValueError(
            f"The scattered-field background must be homogeneous and isotropic in {name}."
        )
    return reference


def _edge_segments(
    grid: _GridGeometry, size: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start point, axis, and length of every electric edge."""
    starts = np.zeros((size, grid.dimension))
    axes = np.zeros((size,), dtype=np.int32)
    lengths = np.zeros((size,))
    for axis in range(grid.dimension):
        shape = grid.edge_shapes[axis]
        count = int(np.prod(shape))
        offset = grid.edge_offsets[axis]
        indices = np.unravel_index(np.arange(count), shape)
        for other in range(grid.dimension):
            edges = grid.boundaries[other]
            starts[offset : offset + count, other] = edges[indices[other]]
        axes[offset : offset + count] = axis
        lengths[offset : offset + count] = grid.widths[axis][indices[axis]]
    return starts, axes, lengths


def _segment_line_distance(
    starts: np.ndarray,
    ends: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    /,
) -> np.ndarray:
    """Distance from each segment to the infinite path line."""

    def perpendicular(points: np.ndarray, /) -> np.ndarray:
        relative = points - origin
        return relative - (relative @ direction)[:, None] * direction

    first = perpendicular(starts)
    change = perpendicular(ends) - first
    denominator = np.sum(change * change, axis=1)
    parameter = np.clip(
        -np.sum(first * change, axis=1) / np.where(denominator > 0.0, denominator, 1.0),
        0.0,
        1.0,
    )
    return np.linalg.norm(first + parameter[:, None] * change, axis=1)


class FrequencyMovingChargePlan(StrictModule):
    """Frequency-domain Maxwell solve of a uniformly moving charge.

    ``background`` (scattered field only) is the prepared homogeneous isotropic
    medium the analytic incident field lives in; the domain's material contrast
    against it and every perfect conductor act as sources. ``stretching`` and
    ``boundaries`` are the B2a coordinate stretching and the shared
    `MaxwellBoundaryPlan` vocabulary. ``method`` selects native GMRES or sparse
    LU; ``policy`` overrides its default linear-solve policy.
    """

    source: PreparedMaxwellMovingCharge
    constitutive: AbstractPreparedMaxwellConstitutive
    background: AbstractPreparedMaxwellConstitutive | None
    stretching: MaxwellCPMLPlan | None
    boundaries: tuple[MaxwellBoundaryPlan, ...]
    policy: LinearSolvePolicy | None
    tolerance: float = eqx.field(static=True)
    restart: int = eqx.field(static=True)
    maxiter: int = eqx.field(static=True)
    formulation: SourceFormulation = eqx.field(static=True)
    method: FrequencyMaxwellSolveMethod = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MaxwellMovingChargePlan | PreparedMaxwellMovingCharge,
        constitutive: AbstractPreparedMaxwellConstitutive,
        /,
        *,
        formulation: SourceFormulation = "total-field",
        background: AbstractPreparedMaxwellConstitutive | None = None,
        stretching: MaxwellCPMLPlan | None = None,
        boundaries: Sequence[MaxwellBoundaryPlan] = (),
        method: FrequencyMaxwellSolveMethod = "direct",
        tolerance: float = 1e-9,
        restart: int = 80,
        maxiter: int = 2000,
        policy: LinearSolvePolicy | None = None,
    ) -> None:
        formulation = parse(formulation, SourceFormulation, "formulation")
        method = parse(method, FrequencyMaxwellSolveMethod, "method")
        prepared = (
            source.prepare() if isinstance(source, MaxwellMovingChargePlan) else source
        )
        if not isinstance(prepared, PreparedMaxwellMovingCharge):
            raise TypeError(
                "source must be a MaxwellMovingChargePlan or its preparation."
            )
        if not isinstance(constitutive, AbstractPreparedMaxwellConstitutive):
            raise TypeError("constitutive must be prepared Maxwell material data.")
        if constitutive.layout_id != prepared.plan.layout.layout_id:
            raise ValueError("Constitutive law and moving-charge layout do not match.")
        match formulation:
            case "total-field":
                if background is not None:
                    raise ValueError(
                        "Only the scattered-field formulation takes a background."
                    )
            case "scattered-field":
                if not isinstance(background, AbstractPreparedMaxwellConstitutive):
                    raise TypeError(
                        "The scattered-field formulation requires a prepared background."
                    )
                if background.layout_id != prepared.plan.layout.layout_id:
                    raise ValueError("Background and moving-charge layout do not match.")
            case _:
                assert_never(formulation)
        if policy is not None and not isinstance(policy, LinearSolvePolicy):
            raise TypeError("policy must be a LinearSolvePolicy.")
        boundary_plans = tuple(boundaries)
        self.source = prepared
        self.constitutive = constitutive
        self.background = background
        self.stretching = stretching
        self.boundaries = boundary_plans
        self.policy = policy
        self.tolerance = float(tolerance)
        self.restart = int(restart)
        self.maxiter = int(maxiter)
        self.formulation = formulation
        self.method = method
        self.plan_id = canonical_fingerprint(
            {
                "kind": "frequency-moving-charge-plan",
                "source": prepared.prepared_id,
                "constitutive": constitutive.prepared_id,
                "background": None if background is None else background.prepared_id,
                "stretching": None if stretching is None else stretching.plan_id,
                "boundaries": [plan.plan_id for plan in boundary_plans],
                "formulation": formulation,
                "method": method,
            }
        )

    def prepare(self) -> PreparedFrequencyMovingCharge:
        return PreparedFrequencyMovingCharge(self)


class PreparedFrequencyMovingCharge(StrictModule):
    """Prepared moving-charge solve: reusable sparse pattern and host geometry."""

    plan: FrequencyMovingChargePlan
    coloring: SparseColoring | None
    transverse_clearance: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FrequencyMovingChargePlan, /) -> None:
        # The sparsity pattern is frequency independent; any ω > 0 exposes it.
        operator = self._operator(plan, jnp.asarray(1.0))
        self.plan = plan
        self.coloring = operator.sparse_coloring() if plan.method == "direct" else None
        self.transverse_clearance = _transverse_clearance(plan)
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-frequency-moving-charge", "plan": plan.plan_id}
        )

    @staticmethod
    def _operator(
        plan: FrequencyMovingChargePlan, omega: Array, /
    ) -> FrequencyMaxwellOperator:
        return FrequencyMaxwellOperator(
            plan.source.plan.bridge,
            plan.source.plan.layout,
            plan.constitutive,
            omega,
            stretching=plan.stretching,
            boundaries=plan.boundaries,
        )

    def _path_medium(
        self, operator: FrequencyMaxwellOperator, /
    ) -> tuple[complex, complex]:
        """``ε`` and ``μ`` seen by the path: the background, else the path cells."""
        plan = self.plan
        layout = plan.source.plan.layout
        if plan.background is not None:
            response = plan.background.frequency_response(operator.angular_frequency)
            electric_probe = 1.5 + jnp.cos(
                jnp.arange(layout.electric_count, dtype=jnp.float64)
            )
            magnetic_probe = 1.5 + jnp.cos(
                jnp.arange(layout.magnetic_count, dtype=jnp.float64)
            )
            permittivity = _homogeneous_scalar(
                response.electric_displacement(electric_probe.astype(jnp.complex128)),
                electric_probe,
                "permittivity",
            )
            inverse = _homogeneous_scalar(
                response.magnetic_field(magnetic_probe.astype(jnp.complex128)),
                magnetic_probe,
                "permeability",
            )
            return permittivity, 1.0 / inverse
        path = plan.source.path_edges
        ones = jnp.ones((layout.electric_count,), dtype=jnp.complex128)
        permittivity = complex(
            np.mean(
                np.asarray(operator.response.electric_displacement(ones))[
                    np.asarray(path)
                ]
            )
        )
        faces = np.asarray(
            plan.source.plan.bridge.cochain.exterior_derivative(
                layout.electric_degree, path.astype(jnp.float64)
            )
            != 0.0
        )
        inverse = np.asarray(
            operator.response.magnetic_field(
                jnp.ones((layout.magnetic_count,), dtype=jnp.complex128)
            )
        )[faces]
        return permittivity, 1.0 / complex(np.mean(inverse))

    def _incident(
        self, omega: Array, permittivity: complex, permeability: complex, /
    ) -> UniformMotionFieldPlan:
        source = self.plan.source.plan
        geometry: UniformMotionGeometry = (
            "point" if source.layout.polarization == "full_3d" else "line"
        )
        return UniformMotionFieldPlan(
            geometry,
            UniformMotionMedium(omega[None], permittivity, permeability),
            charge=source.charge,
            speed=source.speed,
            origin=source.origin,
            direction=source.direction,
        )

    def _scattered_source(
        self,
        operator: FrequencyMaxwellOperator,
        permittivity: complex,
        permeability: complex,
        /,
    ) -> tuple[Array, Array, float]:
        """Scattered-field right-hand side, incident circulations, source distance.

        Free rows carry ``A_b E_inc − A P E_inc``; conductor rows coupled to free
        rows carry ``−E_inc`` and the conductor interior carries zero, so thick
        conductors may enclose the path. The incident field is integrated on every
        free and coupled conductor edge; edges closer to the path than a quarter
        edge are unsupported (NaN) and may not act as sources.
        """
        plan = self.plan
        if plan.background is None:
            raise RuntimeError("Scattered-field preparation lost its background.")
        omega = operator.angular_frequency
        background = FrequencyMaxwellOperator(
            plan.source.plan.bridge,
            plan.source.plan.layout,
            plan.background,
            omega,
            stretching=plan.stretching,
        )
        conductor = operator.conductor

        def free_rows(electric: Array) -> Array:
            free = jnp.where(conductor, 0, electric)
            return jnp.where(conductor, 0, background.mv(electric) - operator.mv(free))

        size = operator.size
        # Random positive row weights expose every structurally coupled column.
        weights = jnp.asarray(
            np.random.default_rng(0).uniform(0.5, 1.5, size), dtype=jnp.complex128
        )
        _, pullback = jax.vjp(free_rows, jnp.zeros((size,), dtype=jnp.complex128))
        (influence,) = pullback(weights)
        host_conductor = np.asarray(conductor)
        needed = np.abs(np.asarray(influence)) > 0.0
        coupled = needed & host_conductor
        evaluated = needed | ~host_conductor
        grid = _GridGeometry(plan.source.plan.bridge)
        starts, axes, lengths = _edge_segments(grid, size)
        tangents = np.eye(grid.dimension)[axes]
        distance = (
            _segment_line_distance(
                starts,
                starts + lengths[:, None] * tangents,
                np.asarray(plan.source.plan.origin),
                np.asarray(plan.source.plan.direction),
            )
            / lengths
        )
        supported = distance >= _MINIMUM_SOURCE_DISTANCE
        closest = float(np.min(distance[needed], initial=np.inf))
        if closest < _MINIMUM_SOURCE_DISTANCE:
            raise ValueError(
                "Scattered-field sources (material contrast or conductor surfaces) must "
                "stay at least a quarter edge from the charge path; use the total-field "
                "formulation."
            )
        indices = np.flatnonzero(evaluated & supported)
        field = self._incident(omega, permittivity, permeability)
        rule = legendre_rule_data(_EDGE_QUADRATURE_NODES)
        nodes = 0.5 * (np.asarray(rule.nodes) + 1.0)
        node_weights = 0.5 * np.asarray(rule.weights)
        points = starts[indices][:, None, :] + (
            nodes[None, :, None]
            * lengths[indices][:, None, None]
            * tangents[indices][:, None, :]
        )
        values = field.evaluate(
            jnp.asarray(points.reshape((-1, grid.dimension)))
        ).electric[0]
        circulation = contract(
            "enc,ec,n->e",
            values.reshape((indices.size, nodes.size, grid.dimension)),
            jnp.asarray(tangents[indices]),
            jnp.asarray(node_weights),
        ) * jnp.asarray(lengths[indices])
        incident = (
            jnp.full((size,), jnp.nan, dtype=jnp.complex128)
            .at[jnp.asarray(indices)]
            .set(circulation)
        )
        sources = jnp.where(jnp.asarray(needed), incident, 0)
        right_hand_side = jnp.where(
            conductor,
            jnp.where(jnp.asarray(coupled), -sources, 0),
            free_rows(sources),
        )
        return right_hand_side, incident, closest

    def solve(
        self, angular_frequency: ConvertibleToArray, /
    ) -> FrequencyMovingChargeResult:
        plan = self.plan
        omega = jnp.asarray(angular_frequency, dtype=jnp.float64)
        operator = self._operator(plan, omega)
        permittivity, permeability = self._path_medium(operator)
        medium = UniformMotionMedium(omega[None], permittivity, permeability)
        wavenumber = medium.transverse_wavenumber(plan.source.plan.speed)[0]
        match plan.formulation:
            case "total-field":
                right_hand_side = (
                    1j
                    * omega
                    * jnp.where(operator.conductor, 0, plan.source.current(omega))
                )
                continuity = plan.source.continuity_defect(omega)
                distance = jnp.asarray(jnp.nan)
                incident = None
            case "scattered-field":
                right_hand_side, incident, closest = self._scattered_source(
                    operator, permittivity, permeability
                )
                continuity = jnp.asarray(jnp.nan)
                distance = jnp.asarray(closest)
            case _:
                assert_never(plan.formulation)
        solved = operator.solve(
            right_hand_side,
            method=plan.method,
            tolerance=plan.tolerance,
            restart=plan.restart,
            maxiter=plan.maxiter,
            policy=plan.policy,
            coloring=self.coloring,
        )
        unknown = solved.electric
        total = (
            unknown
            if incident is None
            else jnp.where(operator.conductor, 0, unknown + incident)
        )
        layout = plan.source.plan.layout
        magnetic_flux = plan.source.plan.bridge.cochain.exterior_derivative(
            layout.electric_degree, total
        ) / (1j * omega)
        radiating = (
            jnp.real(jnp.asarray(permittivity * permeability)) * plan.source.plan.speed**2
            > 1.0
        )
        extent = 2.0 * jnp.pi / jnp.imag(wavenumber)
        clearance = jnp.asarray(self.transverse_clearance)
        evidence = FrequencyMovingChargeEvidence(
            angular_frequency=omega,
            transverse_wavenumber=wavenumber,
            radiating=radiating,
            bound_extent=extent,
            transverse_clearance=clearance,
            bound_field_contained=radiating | (clearance >= extent),
            continuity_defect=continuity,
            source_distance=distance,
            converged=solved.converged,
            residual_norm=solved.residual_norm,
            iterations=solved.iterations,
            open_endpoints=plan.source.open_endpoints,
            periodic=plan.source.periodic_axis is not None,
            formulation=plan.formulation,
            method=plan.method,
        )
        return FrequencyMovingChargeResult(
            electric=total,
            magnetic_flux=magnetic_flux,
            scattered_electric=None if incident is None else unknown,
            incident_electric=incident,
            source=right_hand_side,
            solve=solved,
            ledger=operator.power_ledger(unknown, right_hand_side),
            evidence=evidence,
        )


def _transverse_clearance(plan: FrequencyMovingChargePlan, /) -> float:
    """Distance from the path to the nearest absorbing layer or wall across it."""
    source = plan.source.plan
    grid = _GridGeometry(source.bridge)
    origin = np.asarray(source.origin)
    direction = np.asarray(source.direction)
    widths = (0,) * grid.dimension if plan.stretching is None else plan.stretching.widths
    clearance = np.inf
    for axis in range(grid.dimension):
        if grid.periodic[axis] or abs(direction[axis]) > _PARALLEL_TOLERANCE:
            continue
        layers = widths[axis] if len(widths) > axis else 0
        lower = grid.boundaries[axis][layers]
        upper = grid.boundaries[axis][grid.boundaries[axis].size - 1 - layers]
        clearance = min(clearance, origin[axis] - lower, upper - origin[axis])
    return float(clearance)


__all__ = [
    "FrequencyMovingChargeEvidence",
    "FrequencyMovingChargePlan",
    "FrequencyMovingChargeResult",
    "MaxwellMovingChargePlan",
    "PreparedFrequencyMovingCharge",
    "PreparedMaxwellMovingCharge",
    "SourceFormulation",
]
