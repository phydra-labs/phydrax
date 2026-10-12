#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Periodic structure functions in the orthonormal Fourier convention.

The correlation identities implement the increment statistics in Lanthaler,
Mishra & Parés-Pulido (arXiv:1909.06615, structure-function definitions).
Execution chunk sizes and byte limits are resource policies, not physical constants.
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy import special

from ..._execution_resources import MemoryBudgetError
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import finite_real_scalar, positive_integer, real_number_array
from ...ein import contract
from ...typing import Bool, checked, Dim, Float, Int32, parse, VariadicDim
from ._dealias import PaddingDealiasingPlan, PreparedDealiasingPlan
from ._fourier_shells import _FourierShellBinGeometry
from ._space import TensorSpectralDiscretization


IncrementAverage: TypeAlias = Literal["sphere", "ball"]
ThirdIncrementStatistic: TypeAlias = Literal["longitudinal_cube", "energy_flux"]
IncrementProjection: TypeAlias = Literal[
    "magnitude", "longitudinal", "absolute_longitudinal"
]


class IncrementMemberDim(Dim):
    """Explicit ensemble member axis."""


class IncrementValueDims(VariadicDim):
    """Prepared output or geometry axes."""


class IncrementGridDims(VariadicDim):
    """Native Fourier grid axes."""


class IncrementSpatialDim(Dim):
    """Spatial vector components."""


class IncrementRadiusDim(Dim):
    """Prepared radial queries."""


class IncrementBinDim(Dim):
    """Lattice radial bins."""


class IncrementEdgeDim(Dim):
    """Lattice radial bin edges."""


class IncrementShellDim(Dim):
    """Exact integer squared-radius groups."""


class IncrementQueryDim(Dim):
    """Explicit integer displacement queries."""


@final
class PeriodicIncrementMomentResult(StrictModule):
    """Member-major increments and retained-band evidence.

    ``values`` is (members, *native_shape), (members, radii), or
    (members, displacements). ``finite`` and ``forbidden_coefficient_energy``
    are (members,). Forbidden energy is Σ|c|², not divided by volume.
    ``radii`` is empty for unbinned lattice/estimator results; ``bin_counts``
    and ``valid_bins`` are empty except for radial-binned third moments.
    Exactness concerns the retained real Fourier field, not discarded Nyquist
    content. The estimator is always quadrature, even at polynomial orders.
    """

    __strict_contract__ = True
    values: Float[IncrementMemberDim, IncrementValueDims]
    forbidden_coefficient_energy: Float[IncrementMemberDim]
    finite: Bool[IncrementMemberDim]
    radii: Float[IncrementRadiusDim]
    bin_counts: Int32[IncrementBinDim]
    valid_bins: Bool[IncrementBinDim]
    exact: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


@final
class _IncrementGeometry(StrictModule, NonTrainableState):
    __strict_contract__ = True
    discretization: TensorSpectralDiscretization
    mask: Bool[IncrementGridDims]
    shape: tuple[int, ...] = eqx.field(static=True)
    lengths: tuple[float, ...] = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    member_chunk_size: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        member_chunk_size: int,
        maximum_workspace_bytes: int,
        workspace_bytes: int,
    ) -> None:
        dimension = len(discretization.axes)
        if dimension not in (1, 2, 3) or any(
            axis.family != "fourier" for axis in discretization.axes
        ):
            raise ValueError("Increment moments require one to three Fourier-only axes.")
        member_chunk_size = positive_integer(member_chunk_size, "member_chunk_size")
        maximum_workspace_bytes = positive_integer(
            maximum_workspace_bytes, "maximum_workspace_bytes"
        )
        shape = discretization.modal_shape
        lengths = tuple(float(axis.length) for axis in discretization.axes)
        _admit_workspace(workspace_bytes, maximum_workspace_bytes)
        mask = np.ones(shape, dtype=np.bool_)
        for index, axis in enumerate(discretization.axes):
            axis_shape = (1,) * index + (shape[index],) + (1,) * (dimension - index - 1)
            mask &= ~np.asarray(axis.modes.nyquist_mask).reshape(axis_shape)
        self.discretization = discretization
        self.mask = jnp.asarray(mask)
        self.shape = shape
        self.lengths = lengths
        self.volume = prod(lengths)
        self.dimension = dimension
        self.member_chunk_size = member_chunk_size
        self.maximum_workspace_bytes = maximum_workspace_bytes
        self.workspace_bytes = workspace_bytes

    def inputs(
        self,
        coefficients: ArrayLike,
        component_shape: tuple[int, ...],
        /,
        *,
        workspace_bytes: int,
    ) -> Array:
        values = jnp.asarray(coefficients)
        expected = self.shape + component_shape
        if (
            values.ndim != len(expected) + 1
            or values.shape[0] < 1
            or values.shape[1:] != expected
            or not jnp.issubdtype(values.dtype, jnp.complexfloating)
        ):
            raise ValueError(f"Coefficients require complex shape (members, {expected}).")
        _admit_workspace(workspace_bytes, self.maximum_workspace_bytes)
        return values

    def prepare_member(self, value: Array, /) -> tuple[Array, Array]:
        """Report support and validate reality within a single bounded lane."""
        power = jnp.sum(jnp.abs(value.reshape(self.shape + (-1,))) ** 2, axis=-1)
        forbidden = jnp.sum(jnp.where(self.mask, jnp.zeros_like(power), power))
        retained = value * self.mask.astype(value.dtype).reshape(
            self.shape + (1,) * (value.ndim - self.dimension)
        )
        # Reality is an input contract; do not silently take a real part of a
        # non-Hermitian realization. Nyquist energy remains visible separately.
        opposite = retained
        for axis, n in enumerate(self.shape):
            opposite = jnp.take(opposite, (-jnp.arange(n)) % n, axis=axis)
        tolerance = 64 * jnp.finfo(value.real.dtype).eps
        defect = jnp.max(jnp.abs(retained - jnp.conj(opposite)))
        scale = jnp.maximum(
            jnp.max(jnp.abs(retained)), jnp.asarray(1, dtype=value.real.dtype)
        )
        retained = eqx.error_if(
            retained,
            jnp.isfinite(defect) & (defect > tolerance * scale),
            "Increment coefficients must represent a real (Hermitian) field.",
        )
        return retained, forbidden


def _admit_workspace(required_bytes: int, limit_bytes: int, /) -> None:
    if required_bytes > limit_bytes:
        raise MemoryBudgetError("maximum_workspace_bytes", required_bytes, limit_bytes)


def _output_bytes(members: int, chunk: int, points: int, real_bytes: int, /) -> int:
    # lax.map materializes all fixed-width blocks before removing virtual lanes.
    padded = members + (-members) % chunk
    return padded * (points * real_bytes + real_bytes + 1)


def _lattice_workspace_bytes(
    points: int, components: int, chunk: int, members: int, real_bytes: int = 8, /
) -> int:
    # Gather, retained coefficients and Hermitian comparison: five complex
    # component buffers. Scalar powers, correlation and FFT scratch: seven reals.
    lanes = chunk * points * real_bytes * (10 * components + 7)
    return lanes + _output_bytes(members, chunk, points, real_bytes)


def _radial_workspace_bytes(
    points: int,
    components: int,
    shells: int,
    radii: int,
    chunk: int,
    members: int,
    real_bytes: int = 8,
    /,
) -> int:
    # The support/reality lane uses five complex component buffers; powers use
    # three scalar reals. Two shell reductions coexist with the radial output.
    lanes = chunk * real_bytes * (points * (10 * components + 3) + 2 * shells)
    return lanes + _output_bytes(members, chunk, radii, real_bytes)


def _third_workspace_bytes(
    points: int,
    evaluation_points: int,
    dimension: int,
    bins: int,
    chunk: int,
    members: int,
    real_bytes: int = 8,
    /,
) -> int:
    # Native support/reality plus cross spectra, packed correlations and output;
    # padded reconstruction, product/projection and complex FFT scratch. Pair
    # products execute sequentially, so no d*d padded tensor is resident.
    lanes = (
        chunk
        * real_bytes
        * (
            points * (18 * dimension + 8)
            + evaluation_points * (4 * dimension + 8)
            + 2 * bins
        )
    )
    geometry = points * dimension * 8 + (bins + 1) * 8 if bins else points * dimension * 8
    return lanes + geometry + _output_bytes(members, chunk, bins or points, real_bytes)


def _estimator_workspace_bytes(
    points: int,
    components: int,
    dimension: int,
    queries: int,
    chunk: int,
    members: int,
    real_bytes: int = 8,
    /,
) -> int:
    # Support/reality/reconstruction and roll/delta reuse the same sequential
    # displacement lane. Count each component once, not once per query.
    lanes = chunk * points * real_bytes * (10 * components + 3)
    padded_queries = queries + (-queries) % chunk
    metadata = 2 * padded_queries * dimension * (4 + 8)
    return lanes + metadata + _output_bytes(members, chunk, queries, real_bytes)


def _component_shape(value: tuple[int, ...], /) -> tuple[int, ...]:
    return tuple(positive_integer(size, "component_shape entries") for size in value)


def _map_members(
    function: Callable[[Array], Array], values: Array, geometry: _IncrementGeometry, /
) -> tuple[Array, Array]:
    count, chunk = values.shape[0], geometry.member_chunk_size
    block_count = (count + chunk - 1) // chunk

    def lane(value: Array) -> tuple[Array, Array]:
        retained, forbidden = geometry.prepare_member(value)
        return function(retained), forbidden

    def block(index: Array) -> tuple[Array, Array]:
        positions = index * chunk + jnp.arange(chunk, dtype=jnp.int32)
        valid = positions < count
        # Virtual zero padding avoids copying the entire ensemble solely to
        # append lanes. Each gathered block has one fixed compiled shape.
        members = jnp.take(values, jnp.minimum(positions, count - 1), axis=0)
        members = jnp.where(
            valid.reshape((chunk,) + (1,) * (members.ndim - 1)),
            members,
            jnp.zeros_like(members),
        )
        result, forbidden = jax.vmap(lane)(members)
        return (
            jnp.where(
                valid.reshape((chunk,) + (1,) * (result.ndim - 1)),
                result,
                jnp.zeros_like(result),
            ),
            jnp.where(valid, forbidden, jnp.zeros_like(forbidden)),
        )

    result, forbidden = jax.lax.map(block, jnp.arange(block_count, dtype=jnp.int32))
    return result.reshape((-1,) + result.shape[2:])[:count], forbidden.reshape((-1,))[
        :count
    ]


def _result(
    values: Array,
    forbidden: Array,
    plan_id: str,
    /,
    *,
    exact: bool,
    radii: Array | None = None,
    bin_counts: Array | None = None,
) -> PeriodicIncrementMomentResult:
    counts = jnp.zeros((0,), dtype=jnp.int32) if bin_counts is None else bin_counts
    finite = jnp.all(
        jnp.isfinite(values), axis=tuple(range(1, values.ndim))
    ) & jnp.isfinite(forbidden)
    return PeriodicIncrementMomentResult(
        values=values,
        forbidden_coefficient_energy=forbidden,
        finite=finite,
        radii=jnp.zeros((0,), dtype=values.dtype) if radii is None else radii,
        bin_counts=counts,
        valid_bins=counts > 0,
        exact=exact,
        plan_id=plan_id,
    )


@final
class PeriodicLatticeIncrementMomentPlan(StrictModule, NonTrainableState):
    """Exact M₂ on all native periodic displacements, including h=0.

    Directions for lattice statistics use minimum images; an even-grid
    half-box tie is positive. Lattice M₂ itself does not depend on direction.
    The unnormalized inverse DFT gives M₂=(2/V)(ΣP-Re IDFT(P)).
    """

    geometry: _IncrementGeometry
    component_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        component_shape: tuple[int, ...],
        member_chunk_size: int = 8,
        maximum_workspace_bytes: int = 2**30,
    ) -> None:
        shape = _component_shape(component_shape)
        geometry = _IncrementGeometry(
            discretization,
            member_chunk_size=member_chunk_size,
            maximum_workspace_bytes=maximum_workspace_bytes,
            workspace_bytes=_lattice_workspace_bytes(
                prod(discretization.modal_shape),
                prod(shape),
                positive_integer(member_chunk_size, "member_chunk_size"),
                positive_integer(member_chunk_size, "member_chunk_size"),
            ),
        )
        identifier = canonical_fingerprint(
            {
                "kind": "periodic-lattice-increment-moment",
                "discretization": discretization.prepared_id,
                "component_shape": shape,
                "chunk": member_chunk_size,
                "workspace": maximum_workspace_bytes,
            }
        )
        self.geometry, self.component_shape, self.plan_id = geometry, shape, identifier

    def __call__(self, coefficients: ArrayLike, /) -> PeriodicIncrementMomentResult:
        inputs = jnp.asarray(coefficients)
        inputs = self.geometry.inputs(
            inputs,
            self.component_shape,
            workspace_bytes=_lattice_workspace_bytes(
                prod(self.geometry.shape),
                prod(self.component_shape),
                self.geometry.member_chunk_size,
                inputs.shape[0] if inputs.ndim else 0,
                inputs.real.dtype.itemsize,
            ),
        )
        return _execute_lattice(self, inputs)


def _radial_kernel_defect(
    x: np.ndarray, dimension: int, averaging: IncrementAverage, /
) -> np.ndarray:
    """Prepare 1-K directly: Taylor series avoid small-r cancellation."""
    square = x * x
    match (averaging, dimension):
        case ("sphere", 1):
            return 2 * np.sin(x / 2) ** 2
        case ("sphere", 2):
            return np.where(
                np.abs(x) < 0.1,
                square / 4 - square**2 / 64 + square**3 / 2304 - square**4 / 147456,
                1 - special.j0(x),
            )
        case ("sphere", 3) | ("ball", 1):
            return np.where(
                np.abs(x) < 0.1,
                square / 6 - square**2 / 120 + square**3 / 5040 - square**4 / 362880,
                1 - np.sinc(x / np.pi),
            )
        case ("ball", 2):
            safe = np.where(x == 0, 1.0, x)
            return np.where(
                np.abs(x) < 0.1,
                square / 8 - square**2 / 192 + square**3 / 9216 - square**4 / 737280,
                1 - 2 * special.j1(x) / safe,
            )
        case ("ball", 3):
            safe = np.where(x == 0, 1.0, x)
            return np.where(
                np.abs(x) < 0.1,
                square / 10 - square**2 / 280 + square**3 / 15120 - square**4 / 1330560,
                1 - 3 * (np.sin(x) - x * np.cos(x)) / safe**3,
            )
        case _:
            raise ValueError("Unsupported radial averaging dimension.")


@final
class PeriodicRadialIncrementMomentPlan(StrictModule, NonTrainableState):
    """Exact sphere/ball averages of M₂; radii are immutable prepared metadata.

    Host SciPy Bessel kernels are grouped by exact integer squared mode norm.
    Runtime performs one canonical shell reduction and one matrix product.
    Isotropic here means equal box lengths and equal Fourier point counts.
    ``maximum_workspace_bytes`` bounds member-lane support/reality buffers,
    native powers, shell reductions and padded output. It is independent of
    the prepared matrix's ``maximum_kernel_bytes`` admission.
    """

    __strict_contract__ = True
    geometry: _IncrementGeometry
    shells: _FourierShellBinGeometry
    kernel: Float[IncrementRadiusDim, IncrementShellDim]
    radii: tuple[float, ...] = eqx.field(static=True)
    averaging: IncrementAverage = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    maximum_kernel_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        radii: ArrayLike,
        /,
        *,
        averaging: IncrementAverage,
        component_shape: tuple[int, ...],
        member_chunk_size: int = 8,
        maximum_kernel_bytes: int = 2**28,
        maximum_workspace_bytes: int = 2**30,
    ) -> None:
        average = parse(averaging, IncrementAverage, "averaging")
        shape = _component_shape(component_shape)
        maximum_kernel_bytes = positive_integer(
            maximum_kernel_bytes, "maximum_kernel_bytes"
        )
        member_chunk_size = positive_integer(member_chunk_size, "member_chunk_size")
        lengths = tuple(float(axis.length) for axis in discretization.axes)
        if len(set(discretization.modal_shape)) != 1 or not np.allclose(
            lengths, lengths[0], rtol=1e-12, atol=0
        ):
            raise ValueError(
                "Radial increment moments require an isotropic periodic box."
            )
        r = real_number_array(radii, "radii")
        if r.ndim != 1 or r.size == 0 or np.any(~np.isfinite(r)) or np.any(r < 0):
            raise ValueError("radii must be a nonempty finite nonnegative vector.")
        numbers = np.meshgrid(
            *(
                np.asarray(axis.modes.mode_numbers, dtype=np.int64)
                for axis in discretization.axes
            ),
            indexing="ij",
        )
        squared = sum(number**2 for number in numbers)
        unique, indices = np.unique(squared, return_inverse=True)
        geometry = _IncrementGeometry(
            discretization,
            member_chunk_size=member_chunk_size,
            maximum_workspace_bytes=maximum_workspace_bytes,
            workspace_bytes=_radial_workspace_bytes(
                prod(discretization.modal_shape),
                prod(shape),
                unique.size,
                r.size,
                member_chunk_size,
                member_chunk_size,
            ),
        )
        kernel_bytes = r.size * unique.size * np.dtype(np.float64).itemsize
        if kernel_bytes > maximum_kernel_bytes:
            raise MemoryBudgetError(
                "maximum_kernel_bytes", kernel_bytes, maximum_kernel_bytes
            )
        shell_geometry = _FourierShellBinGeometry(
            indices.reshape(geometry.shape),
            np.arange(unique.size + 1, dtype=np.float64) - 0.5,
            mode_mask=geometry.mask,
            source_id=discretization.prepared_id,
        )
        x = r[:, None] * (2 * np.pi / geometry.lengths[0]) * np.sqrt(unique)[None, :]
        kernel = (2 / geometry.volume) * _radial_kernel_defect(
            x, geometry.dimension, average
        )
        identifier = canonical_fingerprint(
            {
                "kind": "periodic-radial-increment-moment",
                "discretization": discretization.prepared_id,
                "radii": r.tolist(),
                "averaging": average,
                "component_shape": shape,
                "chunk": member_chunk_size,
                "kernel_bytes": maximum_kernel_bytes,
                "workspace_bytes": maximum_workspace_bytes,
            }
        )
        self.geometry, self.shells = geometry, shell_geometry
        self.kernel = jnp.asarray(kernel)
        self.radii = tuple(float(radius) for radius in r)
        self.averaging, self.component_shape = average, shape
        self.maximum_kernel_bytes, self.plan_id = maximum_kernel_bytes, identifier

    def __call__(self, coefficients: ArrayLike, /) -> PeriodicIncrementMomentResult:
        inputs = jnp.asarray(coefficients)
        inputs = self.geometry.inputs(
            inputs,
            self.component_shape,
            workspace_bytes=_radial_workspace_bytes(
                prod(self.geometry.shape),
                prod(self.component_shape),
                self.kernel.shape[1],
                len(self.radii),
                self.geometry.member_chunk_size,
                inputs.shape[0] if inputs.ndim else 0,
                inputs.real.dtype.itemsize,
            ),
        )
        return _execute_radial(self, inputs)


@final
class PeriodicThirdIncrementMomentPlan(StrictModule, NonTrainableState):
    """Exact longitudinal cube or energy-flux increment on the native lattice.

    Retained quadratic products use degree-two padding, never a cubic grid.
    Only Im(Q_ab conj(c_k)) contributes. Pairs of real antisymmetric
    correlations are packed in a single complex inverse FFT. Minimum-image
    directions use the positive half-box tie; h=0 returns zero.
    Optional bins average lattice values (not continuous radial averages).

    For A_ab,k(h) = Re IDFT(i Im(Q_ab conj(c_k)))(h)/V,
    G_ab,k(h)-G_ab,k(-h)=2A_ab,k(h). Homogeneity gives
    δ_ijk = -2(A_ij,k + A_ik,j + A_jk,i). Thus the cube is
    -6 Σ_ijk ĥ_i ĥ_j ĥ_k A_ij,k, and energy flux is
    -2 Σ_ik ĥ_k A_ii,k - 4 Σ_ik ĥ_k A_ik,i.
    """

    __strict_contract__ = True
    geometry: _IncrementGeometry
    dealiasing: PreparedDealiasingPlan
    bins: _FourierShellBinGeometry | None
    directions: Float[IncrementGridDims, IncrementSpatialDim]
    radii: Float[IncrementBinDim]
    radial_bin_edges: Float[IncrementEdgeDim]
    statistic: ThirdIncrementStatistic = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        statistic: ThirdIncrementStatistic,
        dealiasing: PaddingDealiasingPlan,
        radial_bin_edges: ArrayLike | None = None,
        member_chunk_size: int,
        maximum_workspace_bytes: int,
    ) -> None:
        selector = parse(statistic, ThirdIncrementStatistic, "statistic")
        if dealiasing.maximum_polynomial_degree != 2:
            raise ValueError("Third moments require degree-2 exact padding.")
        member_chunk_size = positive_integer(member_chunk_size, "member_chunk_size")
        prepared = dealiasing.prepare(discretization, required_polynomial_degree=2)
        if not prepared.report.exact:
            raise ValueError("Third moments require exact quadratic products.")
        dimension = len(discretization.axes)
        edges = (
            None
            if radial_bin_edges is None
            else real_number_array(radial_bin_edges, "radial_bin_edges")
        )
        if edges is not None:
            if (
                edges.ndim != 1
                or edges.size < 2
                or np.any(~np.isfinite(edges))
                or np.any(np.diff(edges) <= 0)
                or edges[0] < 0
            ):
                raise ValueError(
                    "radial_bin_edges must be finite, increasing, and nonnegative."
                )
        bin_count = 0 if edges is None else edges.size - 1
        geometry = _IncrementGeometry(
            discretization,
            member_chunk_size=member_chunk_size,
            maximum_workspace_bytes=maximum_workspace_bytes,
            workspace_bytes=_third_workspace_bytes(
                prod(discretization.modal_shape),
                prod(prepared.report.evaluation_shape),
                dimension,
                bin_count,
                member_chunk_size,
                member_chunk_size,
            ),
        )
        # Only third moments need the full minimum-image vector grid. Its norm
        # serves both direction preparation and optional bins, without a second mesh.
        offsets = np.meshgrid(
            *(
                np.where(np.arange(n) <= n // 2, np.arange(n), np.arange(n) - n)
                * length
                / n
                for n, length in zip(geometry.shape, geometry.lengths, strict=True)
            ),
            indexing="ij",
        )
        h = np.stack(offsets, axis=-1)
        norm = np.sqrt(np.sum(h * h, axis=-1, keepdims=True))
        directions = np.divide(h, norm, out=np.zeros_like(h), where=norm != 0)
        bins = None
        centers = np.zeros((0,), dtype=np.float64)
        if edges is not None:
            bins = _FourierShellBinGeometry(
                norm[..., 0], edges, source_id=discretization.prepared_id
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
        identifier = canonical_fingerprint(
            {
                "kind": "periodic-third-increment-moment",
                "discretization": discretization.prepared_id,
                "statistic": selector,
                "dealiasing": prepared.prepared_id,
                "edges": None if edges is None else edges.tolist(),
                "chunk": member_chunk_size,
                "workspace": maximum_workspace_bytes,
            }
        )
        self.geometry, self.dealiasing, self.bins = geometry, prepared, bins
        self.radii = jnp.asarray(centers)
        self.radial_bin_edges = jnp.asarray(
            np.zeros((0,), dtype=np.float64) if edges is None else edges
        )
        self.statistic, self.plan_id = selector, identifier
        self.directions = jnp.asarray(directions)

    def _one(self, value: Array) -> Array:
        g = self.geometry
        physical = self.dealiasing.reconstruct(value).real
        direction = self.directions.astype(value.real.dtype)
        output = jnp.zeros(g.shape, dtype=value.real.dtype)
        # These loops have bounded static d<=3, never member/grid loop sizes.
        for a in range(g.dimension):
            for b in range(a, g.dimension):
                q = self.dealiasing.project(physical[..., a] * physical[..., b])
                q = q * g.mask.astype(q.dtype)
                cross = q[..., None] * jnp.conj(value)
                odd = jax.lax.complex(jnp.zeros_like(cross.real), cross.imag)
                correlations = []
                for k in range(0, g.dimension, 2):
                    second = odd[..., k + 1] if k + 1 < g.dimension else jnp.zeros_like(q)
                    packed = jnp.fft.ifftn(odd[..., k] + 1j * second) * (
                        prod(g.shape) / g.volume
                    )
                    correlations.append(packed.real)
                    if k + 1 < g.dimension:
                        correlations.append(packed.imag)
                anti = jnp.stack(correlations, axis=-1)
                match self.statistic:
                    case "longitudinal_cube":
                        multiplier = 1 if a == b else 2
                        output = output - 6 * multiplier * direction[..., a] * direction[
                            ..., b
                        ] * contract("...c,...c->...", direction, anti)
                    case "energy_flux":
                        if a == b:
                            output = (
                                output
                                - 2 * contract("...c,...c->...", direction, anti)
                                - 4 * direction[..., a] * anti[..., a]
                            )
                        else:
                            output = output - 4 * (
                                direction[..., b] * anti[..., a]
                                + direction[..., a] * anti[..., b]
                            )
                    case _:
                        assert_never(self.statistic)
        if self.bins is not None:
            total = self.bins.reduce_integral(output)
            count = self.bins.stored_mode_count
            return total / jnp.maximum(count, jnp.ones_like(count))
        return output

    def __call__(self, coefficients: ArrayLike, /) -> PeriodicIncrementMomentResult:
        inputs = jnp.asarray(coefficients)
        inputs = self.geometry.inputs(
            inputs,
            (self.geometry.dimension,),
            workspace_bytes=_third_workspace_bytes(
                prod(self.geometry.shape),
                prod(self.dealiasing.report.evaluation_shape),
                self.geometry.dimension,
                0 if self.bins is None else self.bins.bin_count,
                self.geometry.member_chunk_size,
                inputs.shape[0] if inputs.ndim else 0,
                inputs.real.dtype.itemsize,
            ),
        )
        return _execute_third(self, inputs)


def _increment_projection_policy(
    projection: IncrementProjection, order: float, /
) -> tuple[int | float, bool]:
    """Own projection-dependent order, exponent and spatial-component rules."""
    match projection:
        case "magnitude":
            return order, False
        case "longitudinal":
            if not order.is_integer():
                raise ValueError("Signed longitudinal order must be integer.")
            return int(order), True
        case "absolute_longitudinal":
            return order, True
        case _:
            assert_never(projection)


@final
class PeriodicIncrementMomentEstimatorPlan(StrictModule, NonTrainableState):
    """Bounded native-grid quadrature for explicitly listed integer shifts.

    Longitudinal projections require a d-component field. Magnitude accepts
    any trailing component shape, including a scalar field. Shifts are reduced
    to minimum images (positive even-grid half-box tie); periodic zero shifts
    are refused. Displacement chunks have the same fixed width as member chunks.
    ``maximum_workspace_bytes`` admits one sequential displacement's working
    set, counting component buffers once, plus query metadata and padded output.
    """

    __strict_contract__ = True
    geometry: _IncrementGeometry
    displacements: Int32[IncrementQueryDim, IncrementSpatialDim]
    directions: Float[IncrementQueryDim, IncrementSpatialDim]
    order: float = eqx.field(static=True)
    projection: IncrementProjection = eqx.field(static=True)
    maximum_displacements: int = eqx.field(static=True)
    exponent: int | float = eqx.field(static=True)
    requires_spatial_vector: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        displacements: ArrayLike,
        /,
        *,
        order: float,
        projection: IncrementProjection,
        member_chunk_size: int,
        maximum_displacements: int,
        maximum_workspace_bytes: int = 2**30,
    ) -> None:
        selector = parse(projection, IncrementProjection, "projection")
        maximum_displacements = positive_integer(
            maximum_displacements, "maximum_displacements"
        )
        member_chunk_size = positive_integer(member_chunk_size, "member_chunk_size")
        p = finite_real_scalar(order, "order")
        if p <= 0:
            raise ValueError("order must be positive.")
        exponent, requires_spatial_vector = _increment_projection_policy(selector, p)
        shifts = np.asarray(displacements)
        if not np.issubdtype(shifts.dtype, np.integer):
            raise TypeError("displacements must contain integers.")
        dimension = len(discretization.axes)
        if (
            shifts.ndim != 2
            or shifts.shape[1] != dimension
            or shifts.shape[0] == 0
            or shifts.shape[0] > maximum_displacements
        ):
            raise ValueError(
                "displacements require a bounded nonempty integer (shifts, dimension) matrix."
            )
        geometry = _IncrementGeometry(
            discretization,
            member_chunk_size=member_chunk_size,
            maximum_workspace_bytes=maximum_workspace_bytes,
            # The minimum component count follows the selected projection;
            # magnitude's actual trailing component shape is known at execution.
            workspace_bytes=_estimator_workspace_bytes(
                prod(discretization.modal_shape),
                dimension if requires_spatial_vector else 1,
                dimension,
                shifts.shape[0],
                member_chunk_size,
                member_chunk_size,
            ),
        )
        reduced = shifts % np.asarray(geometry.shape, dtype=np.int64)
        reduced = np.where(
            reduced <= np.asarray(geometry.shape) // 2,
            reduced,
            reduced - np.asarray(geometry.shape),
        )
        if np.any(np.all(reduced == 0, axis=-1)):
            raise ValueError("Zero periodic displacement is not an estimator query.")
        h = reduced * (np.asarray(geometry.lengths) / np.asarray(geometry.shape))
        direction = h / np.sqrt(np.sum(h * h, axis=-1, keepdims=True))
        identifier = canonical_fingerprint(
            {
                "kind": "periodic-increment-moment-estimator",
                "discretization": discretization.prepared_id,
                "displacements": reduced.tolist(),
                "order": p,
                "projection": selector,
                "chunk": member_chunk_size,
                "maximum_displacements": maximum_displacements,
                "workspace_bytes": maximum_workspace_bytes,
            }
        )
        self.geometry = geometry
        self.displacements, self.directions = (
            jnp.asarray(reduced, dtype=jnp.int32),
            jnp.asarray(direction),
        )
        self.order, self.projection = p, selector
        self.exponent, self.requires_spatial_vector = exponent, requires_spatial_vector
        self.maximum_displacements, self.plan_id = maximum_displacements, identifier

    def _one(self, value: Array) -> Array:
        g = self.geometry
        physical = g.discretization.reconstruct(value).real.reshape(g.shape + (-1,))
        count, chunk = self.displacements.shape[0], g.member_chunk_size
        padding = (-count) % chunk
        shifts = jnp.pad(self.displacements, ((0, padding), (0, 0))).reshape(
            (-1, chunk, g.dimension)
        )
        directions = jnp.pad(self.directions, ((0, padding), (0, 0))).reshape(
            (-1, chunk, g.dimension)
        )
        valid = (jnp.arange(count + padding) < count).reshape((-1, chunk))

        def displacement(data: tuple[Array, Array]) -> Array:
            shift, direction = data
            delta = (
                jnp.roll(
                    physical,
                    -shift,
                    axis=tuple(range(g.dimension)),
                )
                - physical
            )
            match self.projection:
                case "magnitude":
                    projected = jnp.sqrt(jnp.sum(delta**2, axis=-1))
                case "longitudinal":
                    projected = contract(
                        "...c,c->...", delta, direction.astype(delta.dtype)
                    )
                case "absolute_longitudinal":
                    projected = jnp.abs(
                        contract("...c,c->...", delta, direction.astype(delta.dtype))
                    )
                case _:
                    assert_never(self.projection)
            return jnp.mean(projected**self.exponent)

        def block(data: tuple[Array, Array, Array]) -> Array:
            shift, direction, lanes = data
            return jnp.where(
                lanes,
                jax.lax.map(displacement, (shift, direction)),
                jnp.zeros((chunk,), dtype=value.real.dtype),
            )

        return jax.lax.map(block, (shifts, directions, valid)).reshape((-1,))[:count]

    def __call__(self, coefficients: ArrayLike, /) -> PeriodicIncrementMomentResult:
        values = jnp.asarray(coefficients)
        if values.ndim < self.geometry.dimension + 1:
            raise ValueError("Estimator coefficients require a leading member axis.")
        component_shape = _component_shape(values.shape[1 + self.geometry.dimension :])
        if self.requires_spatial_vector and component_shape != (self.geometry.dimension,):
            raise ValueError("Longitudinal estimation requires a spatial vector field.")
        inputs = self.geometry.inputs(
            values,
            component_shape,
            workspace_bytes=_estimator_workspace_bytes(
                prod(self.geometry.shape),
                prod(component_shape),
                self.geometry.dimension,
                self.displacements.shape[0],
                self.geometry.member_chunk_size,
                values.shape[0],
                values.real.dtype.itemsize,
            ),
        )
        return _execute_estimator(self, inputs)


@eqx.filter_jit
def _execute_lattice(
    plan: PeriodicLatticeIncrementMomentPlan, coefficients: Array, /
) -> PeriodicIncrementMomentResult:
    def one(value: Array) -> Array:
        power = jnp.sum(jnp.abs(value.reshape(plan.geometry.shape + (-1,))) ** 2, axis=-1)
        correlation = jnp.fft.ifftn(power).real * prod(plan.geometry.shape)
        return (2 / plan.geometry.volume) * (jnp.sum(power) - correlation)

    values, forbidden = _map_members(one, coefficients, plan.geometry)
    return _result(values, forbidden, plan.plan_id, exact=True)


@eqx.filter_jit
def _execute_radial(
    plan: PeriodicRadialIncrementMomentPlan, coefficients: Array, /
) -> PeriodicIncrementMomentResult:
    def one(value: Array) -> Array:
        power = jnp.sum(jnp.abs(value.reshape(plan.geometry.shape + (-1,))) ** 2, axis=-1)
        return plan.shells.reduce_integral(power) @ plan.kernel.astype(power.dtype).T

    values, forbidden = _map_members(one, coefficients, plan.geometry)
    return _result(
        values,
        forbidden,
        plan.plan_id,
        exact=True,
        radii=jnp.asarray(plan.radii, dtype=jnp.float64),
    )


@eqx.filter_jit
def _execute_third(
    plan: PeriodicThirdIncrementMomentPlan, coefficients: Array, /
) -> PeriodicIncrementMomentResult:
    values, forbidden = _map_members(plan._one, coefficients, plan.geometry)
    counts = None if plan.bins is None else plan.bins.stored_mode_count.astype(jnp.int32)
    return _result(
        values,
        forbidden,
        plan.plan_id,
        exact=True,
        radii=plan.radii,
        bin_counts=counts,
    )


@eqx.filter_jit
def _execute_estimator(
    plan: PeriodicIncrementMomentEstimatorPlan, coefficients: Array, /
) -> PeriodicIncrementMomentResult:
    values, forbidden = _map_members(plan._one, coefficients, plan.geometry)
    return _result(values, forbidden, plan.plan_id, exact=False)


__all__ = [
    "IncrementAverage",
    "IncrementProjection",
    "ThirdIncrementStatistic",
    "PeriodicIncrementMomentResult",
    "PeriodicLatticeIncrementMomentPlan",
    "PeriodicRadialIncrementMomentPlan",
    "PeriodicThirdIncrementMomentPlan",
    "PeriodicIncrementMomentEstimatorPlan",
]
