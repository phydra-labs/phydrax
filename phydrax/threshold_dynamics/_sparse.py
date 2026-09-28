#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse candidate-label threshold dynamics on periodic sparse voxel grids.

Sites are the active voxels of a `SparseVoxelGridPlan` over a fully periodic
Morton box, stored in aligned bricks. Each step gathers the labels in every
brick halo (the brick plus the stencil radius), groups them with a case-local
`KeyGroupPlan` into at most ``candidate_capacity`` candidate labels per brick,
and accumulates the box-stencil heat convolution only into those candidate slots.
No ``site x global-label`` array is formed: storage and work scale with sites,
stencil size and candidate capacity, independent of the declared label count.

The stencil is the tensor product of the exact 1D periodic heat kernels of the
box, ``w(o) = prod_a g_a(o_a)`` with ``g_a = exp(tau d^2/dx_a^2) delta_0`` from the
canonical periodic Fourier axis, truncated to ``|o_a| <= stencil_radius``. When
the radius covers the period the route reproduces the exact periodic route.
Evidence reports the truncated kernel mass and the smallest eigenvalue factor of
the truncated periodic stencil; energy dissipation is admitted only when that
factor is nonnegative (positive semidefinite kernel). Neighbours outside the
active voxel set do not interact: the kernel is the box kernel restricted to the
active sites, which remains positive semidefinite.
"""

from __future__ import annotations

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import positive_integer
from ..discretization import AxisDomain
from ..discretization.spatial import (
    MortonAddressPlan,
    PreparedSparseVoxelGrid,
    SparseVoxelGridPlan,
)
from ..discretization.spectral import (
    FourierBasisPlan,
    PreparedSpectralAxis,
    SpectralPrecisionPolicy,
)
from ..sparse import KeyGroupPlan
from ._contracts import (
    HeatActionEvidence,
    SparseCandidateEvidence,
    ThresholdKernelDecomposition,
    ThresholdPotentials,
)


def _axis_offsets(resolution: int, radius: int, /) -> tuple[int, ...]:
    """Stencil offsets along one periodic axis, each residue at most once."""
    if 2 * radius + 1 <= resolution:
        return tuple(range(-radius, radius + 1))
    return tuple(range(-((resolution - 1) // 2), resolution // 2 + 1))


def _covers_period(resolution: int, radius: int, /) -> bool:
    """Whether the stencil covers every periodic residue exactly once."""
    return len({offset % resolution for offset in _axis_offsets(resolution, radius)}) == (
        resolution
    )


def _row_major(coordinates: np.ndarray, width: int, /) -> np.ndarray:
    dimension = coordinates.shape[-1]
    strides = np.asarray(
        [width ** (dimension - axis - 1) for axis in range(dimension)], dtype=np.int64
    )
    return np.sum(coordinates * strides, axis=-1)


class SparseLabelGrid(StrictModule):
    """Active voxels of a periodic Morton box with brick halos and a heat stencil.

    Sites follow the canonical lexicographic order of the unique active integer
    coordinates (``site_coordinates``); label arrays of the sparse route use that
    order. Every site carries the voxel measure ``prod(extent_a / resolution)``.
    """

    voxels: PreparedSparseVoxelGrid
    axes: tuple[PreparedSpectralAxis, ...]
    site_coordinates: Array
    site_brick: Array
    site_local: Array
    storage_to_site: Array
    halo_storage: Array
    stencil_halo: Array
    own_halo: Array
    axis_offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    site_shape: tuple[int, ...] = eqx.field(static=True)
    brick_count: int = eqx.field(static=True)
    halo_size: int = eqx.field(static=True)
    stencil_radius: int = eqx.field(static=True)
    full_period: bool = eqx.field(static=True)
    site_measure: float = eqx.field(static=True)
    resolution_length: float = eqx.field(static=True)
    equal_site_measure: bool = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        active_coordinates: ArrayLike,
        /,
        *,
        brick_size: int,
        brick_capacity: int,
        stencil_radius: int,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        if not isinstance(address_plan, MortonAddressPlan):
            raise TypeError("address_plan must be a MortonAddressPlan.")
        if not all(address_plan.periodic_axes):
            raise ValueError("The sparse threshold route needs a fully periodic box.")
        radius = positive_integer(stencil_radius, "stencil_radius")
        working = jnp.dtype(dtype)
        if working not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise ValueError("dtype must be float32 or float64.")
        voxels = SparseVoxelGridPlan(
            address_plan, brick_size=brick_size, brick_capacity=brick_capacity
        ).prepare(active_coordinates)
        coordinates = np.unique(np.asarray(active_coordinates, dtype=np.int64), axis=0)
        dimension = address_plan.dimension
        resolution = address_plan.resolution
        size = voxels.brick_size
        offsets = tuple(_axis_offsets(resolution, radius) for _ in range(dimension))
        halo_radius = max(abs(value) for value in offsets[0])
        width = size + 2 * halo_radius
        lookup = voxels.lookup_integer(jnp.asarray(coordinates))
        site_brick = np.asarray(lookup.brick_slots, dtype=np.int64)
        site_local = np.asarray(lookup.local_slots, dtype=np.int64)
        per_brick = voxels.voxels_per_brick
        capacity = voxels.brick_capacity
        storage_to_site = np.full((capacity * per_brick,), -1, dtype=np.int64)
        storage_to_site[site_brick * per_brick + site_local] = np.arange(
            coordinates.shape[0]
        )
        origins = np.zeros((capacity, dimension), dtype=np.int64)
        origins[site_brick] = (coordinates // size) * size
        brick_active = np.zeros((capacity,), dtype=np.bool_)
        brick_active[site_brick] = True
        halo_local = np.asarray(
            tuple(itertools.product(range(width), repeat=dimension)), dtype=np.int64
        )
        halo_lookup = voxels.lookup_integer(
            jnp.asarray(origins[:, None, :] + halo_local[None, :, :] - halo_radius)
        )
        halo_storage = np.where(
            np.asarray(halo_lookup.supported) & brick_active[:, None],
            np.asarray(halo_lookup.brick_slots, dtype=np.int64) * per_brick
            + np.asarray(halo_lookup.local_slots, dtype=np.int64),
            -1,
        )
        voxel_local = np.asarray(
            tuple(itertools.product(range(size), repeat=dimension)), dtype=np.int64
        )
        stencil = np.asarray(tuple(itertools.product(*offsets)), dtype=np.int64)
        stencil_halo = _row_major(
            voxel_local[None, :, :] + stencil[:, None, :] + halo_radius, width
        )
        extents = tuple(
            upper - lower
            for lower, upper in zip(address_plan.lower, address_plan.upper, strict=True)
        )
        precision = SpectralPrecisionPolicy(working)
        self.voxels = voxels
        self.axes = tuple(
            FourierBasisPlan(resolution).prepare(
                AxisDomain.periodic(lower, upper), precision=precision
            )
            for lower, upper in zip(address_plan.lower, address_plan.upper, strict=True)
        )
        self.site_coordinates = jnp.asarray(coordinates, dtype=jnp.int32)
        self.site_brick = jnp.asarray(site_brick, dtype=jnp.int32)
        self.site_local = jnp.asarray(site_local, dtype=jnp.int32)
        self.storage_to_site = jnp.asarray(storage_to_site, dtype=jnp.int32)
        self.halo_storage = jnp.asarray(halo_storage, dtype=jnp.int32)
        self.stencil_halo = jnp.asarray(stencil_halo, dtype=jnp.int32)
        self.own_halo = jnp.asarray(
            _row_major(voxel_local + halo_radius, width), jnp.int32
        )
        self.axis_offsets = offsets
        self.site_shape = (coordinates.shape[0],)
        self.brick_count = capacity
        self.halo_size = halo_local.shape[0]
        self.stencil_radius = radius
        self.full_period = _covers_period(resolution, radius)
        self.site_measure = math.prod(extent / resolution for extent in extents)
        self.resolution_length = max(extent / resolution for extent in extents)
        self.equal_site_measure = True
        self.site_id = canonical_fingerprint(
            {
                "kind": "threshold-sparse-grid-sites",
                "voxels": voxels.grid_id,
            }
        )
        self.route_id = canonical_fingerprint(
            {
                "kind": "sparse-label-grid",
                "sites": self.site_id,
                "stencil_radius": radius,
                "dtype": working.name,
            }
        )

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.dtype(self.axes[0].precision.physical_dtype)

    @property
    def site_count(self) -> int:
        return self.site_shape[0]

    @property
    def stencil_size(self) -> int:
        return self.stencil_halo.shape[0]

    @property
    def storage_sites(self) -> int:
        return self.storage_to_site.shape[0]

    def site_measures(self) -> Array:
        return jnp.full(self.site_shape, self.site_measure, dtype=self.dtype)

    def stencil_weights(self, times: Array, /) -> tuple[Array, Array, Array]:
        """Box-stencil weights ``(K, S)``, truncated mass ``(K,)`` and symbol minimum.

        The symbol minimum is the smallest eigenvalue factor over axes and times of
        the truncated 1D periodic stencils, normalized by the factor's maximum.
        """
        axis_weights = []
        mass = jnp.ones(times.shape, self.dtype)
        symbol_minimum = jnp.asarray(jnp.inf, self.dtype)
        for axis, offsets in zip(self.axes, self.axis_offsets, strict=True):
            count = axis.physical_count
            residues = np.mod(np.asarray(offsets), count)
            delta = jnp.zeros((count,), self.dtype).at[0].set(1.0)
            modal = axis.analyze(delta)
            eigenvalues = axis.laplacian_eigenvalues().astype(self.dtype)
            kernels = jnp.real(
                jax.vmap(
                    lambda tau: axis.synthesize(jnp.exp(-tau * eigenvalues) * modal)
                )(times.astype(self.dtype))
            )
            local = kernels[:, residues]
            truncated = jnp.zeros_like(kernels).at[:, residues].set(local)
            symbol = jnp.real(jax.vmap(axis.analyze)(truncated))
            scale = jnp.max(jnp.abs(symbol), axis=-1)
            symbol_minimum = jnp.minimum(
                symbol_minimum, jnp.min(jnp.min(symbol, axis=-1) / scale)
            )
            mass = mass * jnp.sum(local, axis=-1)
            axis_weights.append(local)
        weights = axis_weights[0]
        for local in axis_weights[1:]:
            weights = (weights[:, :, None] * local[:, None, :]).reshape(
                (times.shape[0], -1)
            )
        truncated_mass = jnp.zeros_like(mass) if self.full_period else 1.0 - mass
        return weights, truncated_mass, symbol_minimum


def sparse_potentials(
    grid: SparseLabelGrid,
    groups: KeyGroupPlan,
    labels: Array,
    active_labels: Array,
    decomposition: ThresholdKernelDecomposition,
    /,
) -> ThresholdPotentials:
    """Candidate-slot potentials of one sparse label state (no site x label array)."""
    dtype = grid.dtype
    capacity = groups.group_capacity
    storage_site = jnp.clip(grid.storage_to_site, 0, None)
    storage_labels = jnp.where(grid.storage_to_site >= 0, labels[storage_site], -1)
    halo_labels = jnp.where(
        grid.halo_storage >= 0, storage_labels[jnp.clip(grid.halo_storage, 0, None)], -1
    )
    grouped = groups.build(jnp.maximum(halo_labels, 0), halo_labels >= 0)
    slots = grouped.item_group_slots
    weights, truncated_mass, symbol_minimum = grid.stencil_weights(decomposition.times)
    # The single-Gaussian form has a vanishing second kernel: accumulate only one.
    kernels = 1 if decomposition.form == "single-gaussian" else 2
    per_brick = grid.voxels.voxels_per_brick
    cells = grid.brick_count * per_brick * capacity
    # Flat (brick, voxel) row offsets into the (brick, voxel, slot) accumulator.
    rows = (jnp.arange(grid.brick_count * per_brick, dtype=jnp.int32) * capacity).reshape(
        (grid.brick_count, per_brick)
    )

    def accumulate(total: Array, route: tuple[Array, Array]) -> tuple[Array, None]:
        halo_items, offset_weights = route
        slot = slots[:, halo_items]
        # Each site receives exactly one neighbour per offset, so the scatter is
        # collision-free and deterministic; invalid or overflowing slots drop out.
        target = jnp.where((slot >= 0) & (slot < capacity), rows + slot, cells)
        updated = total.at[:, target.reshape(-1)].add(
            jnp.broadcast_to(offset_weights[:, None], (kernels, target.size)),
            mode="drop",
        )
        return updated, None

    accumulated, _ = jax.lax.scan(
        accumulate,
        jnp.zeros((kernels, cells), dtype),
        (grid.stencil_halo, weights[:kernels].T.astype(dtype)),
    )
    accumulated = accumulated.reshape((kernels, grid.brick_count, per_brick, capacity))
    keys = jnp.where(grouped.group_active, grouped.group_keys, 0).astype(jnp.int32)
    if decomposition.uniform:
        coefficients = decomposition.coefficients[:kernels].astype(dtype)
        potentials = coefficients[0] * (
            jnp.sum(accumulated[0], axis=-1, keepdims=True) - accumulated[0]
        )
        if kernels == 2:
            potentials = potentials + coefficients[1] * (
                jnp.sum(accumulated[1], axis=-1, keepdims=True) - accumulated[1]
            )
        coefficient_scale = jnp.abs(coefficients)
    else:
        coefficients = decomposition.pair_coefficients(
            keys[:, :, None], keys[:, None, :]
        ).astype(dtype)
        potentials = accumulated[0] @ coefficients[0]
        if kernels == 2:
            potentials = potentials + accumulated[1] @ coefficients[1]
        coefficient_scale = jnp.max(jnp.abs(coefficients), axis=(1, 2, 3))
    own_slots = slots[:, grid.own_halo]
    site_values = potentials[grid.site_brick, grid.site_local]
    site_valid = grouped.group_active[grid.site_brick]
    own = jnp.take_along_axis(
        site_values, own_slots[grid.site_brick, grid.site_local][:, None], axis=-1
    )[:, 0]
    finite = jnp.all(jnp.where(site_valid, jnp.isfinite(site_values), True))
    error = jnp.sum(coefficient_scale * jnp.abs(truncated_mass[:kernels]))
    overflow_bricks = grouped.evidence.group_overflow
    active_counts = jnp.bincount(grid.site_brick, length=grid.brick_count)
    zero = jnp.zeros((), jnp.int32)
    heat = HeatActionEvidence(
        successful=finite,
        error_estimate=error.astype(dtype),
        native_status=zero,
        converged=finite,
        derivative_valid=finite,
        iterations=zero,
        setup_matvec_count=zero,
        action_matvec_count=zero,
        transpose_matvec_count=zero,
        breakdown_status=zero,
        numeric_version=zero,
        method="periodic-box-heat-stencil",
        exact=grid.full_period,
        retained_storage_bytes=0,
        workspace_bytes=0,
        operator_id=None,
        prepared_id=None,
    )
    sparse = SparseCandidateEvidence(
        required_candidates=jnp.max(grouped.evidence.required_groups),
        candidate_overflow=jnp.any(overflow_bricks),
        overflowed_sites=jnp.sum(jnp.where(overflow_bricks, active_counts, 0)).astype(
            jnp.int32
        ),
        active_sites=jnp.asarray(grid.site_count, jnp.int32),
        active_label_count=jnp.sum(active_labels, dtype=jnp.int32),
        stencil_routes=jnp.asarray(grid.stencil_size * grid.storage_sites, jnp.int64),
        truncated_kernel_mass=jnp.max(truncated_mass),
        kernel_symbol_minimum=symbol_minimum,
        candidate_capacity=capacity,
    )
    tolerance = 64.0 * jnp.finfo(dtype).eps
    return ThresholdPotentials(
        values=jnp.where(site_valid, site_values, jnp.inf),
        keys=keys[grid.site_brick],
        valid=site_valid,
        own=own,
        heat=heat,
        sparse=sparse,
        kernel_admitted=symbol_minimum >= -tolerance,
    )


__all__ = ["SparseLabelGrid", "sparse_potentials"]
