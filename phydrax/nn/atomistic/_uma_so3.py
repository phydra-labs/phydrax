#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""UMA/eSCN-MD coefficient layouts, source basis, edge frames and sphere grids.

Source basis. The pinned fairchem UMA source (``bb0a062``) stores every degree
in the e3nn real basis: orders ``m = -l..l`` of real orthonormal harmonics
without Condon--Shortley phase whose polar axis is ``y`` and whose azimuth is
measured from ``z`` toward ``x``, so degree one is ``(x, y, z)``. This is the
native real basis of :mod:`phydrax.special` (polar axis ``z``, azimuth from
``x`` toward ``y``, degree one ``(y, z, x)``) evaluated in relabelled
coordinates: with the proper rotation ``P`` mapping source ``(x, y, z)`` to
native ``(z, x, y)``,

``Y_source_l(v) = Y_native_l(P v)``  and  ``D_source_l(R) = D_native_l(P R P^T)``

for every degree. The identity follows from the shared closed form (e3nn's
``Legendre`` is the Rodrigues associated Legendre function without the
``(-1)^m`` phase, times ``sqrt(2) cos(m alpha)`` / ``sqrt(2) sin(|m| alpha)``);
it is checked against each source checkpoint's stored e3nn ``J`` matrices by
`uma_source_basis_residual`. Every Wigner matrix here comes from the native
arbitrary-degree O(3) recurrence; no source kernel is imported.

Edge frames. The source rotates each edge vector ``v = r_source - r_target``
(plus image shift) onto ``+y`` with a two-chart quaternion blended by NLERP
under a C-infinity smooth step, then composes a roll drawn uniformly in
``[0, 2 pi)`` per edge per forward pass. The native realization uses the same
two-chart quaternion with zero roll (`UMAEdgeRollGauge`
``"two-chart-nlerp-zero-roll"``) and never draws random numbers. The SO(2)
convolutions, ``m`` truncation, envelope and gate activation commute exactly
with rolls, so gate-activated networks realize the source function. The
separable S2 activation is roll invariant only up to grid quadrature, so the
fixed gauge is a distinct deterministic realization, named in the architecture
(``roll_realization``) and method identities. Explicit rolls can be composed
with ``uma_edge_rotation(..., roll=...)``.

Grids. `UMASphericalGrid` reproduces e3nn ``ToS2Grid``/``FromS2Grid`` with
``"integral"`` normalization exactly as the source's ``SO3_Grid`` uses them:
``beta_i = pi (i + 1/2) / B``, ``alpha_j = 2 pi j / A``, source point
``(sin b sin a, cos b, sin b cos a)``, Kostelec--Rockmore quadrature weights
times ``B^2 / A`` and ``4 pi`` for the inverse, the ``sqrt((2l+1)/(2m_max+1))``
rescale of degrees above ``m_max`` and the reduced coefficient selection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.scipy.linalg import block_diag
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._model import register_artifact_value
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import nonnegative_integer, positive_integer
from ...ein import contract
from ...special import RealCartesianHarmonics
from ...typing import Dim, Float, Identifier, Size
from ..operator.representations._irreps import _host_proper_actions, _proper_actions


# "two-chart-nlerp-zero-roll": deterministic zero roll. "keyed-route-uniform-roll":
# the source's uniform roll on [0, 2 pi), drawn from an explicitly supplied key
# addressed by each route's full-width identity (`uma_keyed_edge_rolls`).
UMAEdgeRollGauge: TypeAlias = Literal[
    "two-chart-nlerp-zero-roll", "keyed-route-uniform-roll"
]
# fold_in domain separator of UMA edge rolls ("UMAR").
_ROLL_DOMAIN = 0x554D4152

# Native coordinates of a source (e3nn) vector: native = P @ source.
UMA_SOURCE_FRAME = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
UMA_SOURCE_FRAME.setflags(write=False)

# Source quaternion chart blend: ``e_y`` in [-0.9, 0.9] (quaternion_utils).
_BLEND_START = -0.9
_BLEND_WIDTH = 1.8
# torch.nn.functional.normalize's default epsilon.
_NORMALIZE_EPSILON = 1.0e-12


@lru_cache(maxsize=64)
def _coefficient_tables(
    lmax: int, mmax: int, /
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Exact source ``CoefficientMapping`` tables.

    Returns ``(reduced, m_major, m_sizes, degrees)``: the full l-major index of
    each reduced coefficient (``|m| <= mmax``, l-major, ``m`` ascending), the
    reduced index at each m-major position (for ``m = 0..mmax``: the ``+m``
    coefficients by ascending degree, then the ``-m`` ones), the number of
    degrees carrying each order and the degree of each m-major position.
    """
    reduced: list[int] = []
    orders: list[int] = []
    degrees_reduced: list[int] = []
    for degree in range(lmax + 1):
        bound = min(mmax, degree)
        for order in range(-bound, bound + 1):
            reduced.append(degree * degree + degree + order)
            orders.append(order)
            degrees_reduced.append(degree)
    m_major: list[int] = []
    m_sizes: list[int] = []
    for order in range(mmax + 1):
        real = [i for i, value in enumerate(orders) if value == order]
        m_major.extend(real)
        m_sizes.append(len(real))
        if order:
            m_major.extend(i for i, value in enumerate(orders) if value == -order)
    degrees = tuple(degrees_reduced[index] for index in m_major)
    return tuple(reduced), tuple(m_major), tuple(m_sizes), degrees


class UMAGridLatitudeDim(Dim):
    pass


class UMAGridLongitudeDim(Dim):
    pass


class UMAReducedCoefficientDim(Dim):
    pass


@final
class UMACoefficientLayout(StrictModule, NonTrainableState):
    """Exact source coefficient bookkeeping for one ``(lmax, mmax)``.

    Node features are full l-major ``(lmax + 1)**2`` coefficients; edge-frame
    features are the reduced ``|m| <= mmax`` coefficients in m-major order.
    """

    __strict_contract__ = True

    lmax: int = eqx.field(static=True)
    mmax: int = eqx.field(static=True)
    layout_id: Identifier = eqx.field(static=True)

    def __init__(self, lmax: int, mmax: int, /) -> None:
        lmax_ = nonnegative_integer(lmax, "lmax")
        mmax_ = nonnegative_integer(mmax, "mmax")
        if mmax_ > lmax_:
            raise ValueError("UMA mmax must not exceed lmax.")
        self.lmax = lmax_
        self.mmax = mmax_
        self.layout_id = canonical_fingerprint(
            {"kind": "uma-coefficient-layout", "lmax": lmax_, "mmax": mmax_}
        )

    @property
    def full_size(self) -> int:
        return (self.lmax + 1) ** 2

    @property
    def reduced_indices(self) -> tuple[int, ...]:
        """Full l-major index of each reduced coefficient (source ``coefficient_idx``)."""
        return _coefficient_tables(self.lmax, self.mmax)[0]

    @property
    def reduced_size(self) -> int:
        return len(self.reduced_indices)

    @property
    def m_major(self) -> tuple[int, ...]:
        """Reduced index at each m-major position (source ``to_m``)."""
        return _coefficient_tables(self.lmax, self.mmax)[1]

    @property
    def m_sizes(self) -> tuple[int, ...]:
        """Degrees per order ``m = 0..mmax``, ``lmax - m + 1``."""
        return _coefficient_tables(self.lmax, self.mmax)[2]

    @property
    def m_major_degrees(self) -> tuple[int, ...]:
        return _coefficient_tables(self.lmax, self.mmax)[3]

    @property
    def edge_rows(self) -> tuple[int, ...]:
        """Full l-major row of each m-major edge coefficient."""
        reduced = self.reduced_indices
        return tuple(reduced[index] for index in self.m_major)

    @property
    def m_split_sizes(self) -> tuple[int, ...]:
        """m-major row counts per order (``m > 0`` holds both signs)."""
        sizes = self.m_sizes
        return (sizes[0], *(2 * size for size in sizes[1:]))


def uma_edge_rotation(vector: ArrayLike, /, *, roll: ArrayLike | None = None) -> Array:
    """Source edge-to-``+y`` rotation, ``(..., 3) -> (..., 3, 3)``.

    ``vector`` is the source edge vector ``r_source - r_target (+ shift)``. The
    two charts ``q1 = (1 + e_y, -e_z, 0, e_x)`` and ``q2 = (-e_z, 1 - e_y, e_x,
    0)`` are each normalized with a squared-norm floor of the dtype epsilon,
    NLERP-blended with the C-infinity step of ``(e_y + 0.9) / 1.8`` and converted
    to a rotation matrix ``R`` with ``R e = +y``. ``roll`` (radians, shape
    ``(...)``) is an explicitly supplied rotation ``gamma`` about ``+y``
    composed as the source does (``q = q_y(gamma) q_edge``); ``None`` is the
    zero roll. Nothing here draws random numbers. A zero vector (an inactive
    padded edge) yields a finite rotation whose derivatives stay finite.
    """
    values = jnp.asarray(vector)
    norm = jnp.sqrt(jnp.sum(values * values, axis=-1, keepdims=True))
    unit = values / jnp.maximum(norm, _NORMALIZE_EPSILON)
    ex, ey, ez = unit[..., 0], unit[..., 1], unit[..., 2]
    epsilon = jnp.finfo(unit.dtype).eps
    zero = jnp.zeros_like(ex)

    def normalized(q: Array, /) -> Array:
        squared = jnp.sum(q * q, axis=-1, keepdims=True)
        return q / jnp.sqrt(jnp.maximum(squared, epsilon))

    first = normalized(jnp.stack([1.0 + ey, -ez, zero, ex], axis=-1))
    second = normalized(jnp.stack([-ez, 1.0 - ey, ex, zero], axis=-1))
    t = jnp.clip((ey - _BLEND_START) / _BLEND_WIDTH, 0.0, 1.0)
    denominator = jnp.maximum(t * (1.0 - t), epsilon)
    step = jax.nn.sigmoid((2.0 * t - 1.0) / denominator)
    step = jnp.where(t < epsilon, 0.0, jnp.where(t > 1.0 - epsilon, 1.0, step))
    dot = jnp.sum(second * first, axis=-1, keepdims=True)
    aligned = jnp.where(dot < 0.0, -second, second)
    blended = (1.0 - step[..., None]) * aligned + step[..., None] * first
    blended_norm = jnp.sqrt(jnp.sum(blended * blended, axis=-1, keepdims=True))
    q = blended / jnp.maximum(blended_norm, _NORMALIZE_EPSILON)
    if roll is not None:
        half = 0.5 * jnp.asarray(roll, dtype=q.dtype)
        cosine, sine = jnp.cos(half), jnp.sin(half)
        w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        # Hamilton product (cos, 0, sin, 0) * (w0, x0, y0, z0).
        q = jnp.stack(
            [
                cosine * w0 - sine * y0,
                cosine * x0 + sine * z0,
                cosine * y0 + sine * w0,
                cosine * z0 - sine * x0,
            ],
            axis=-1,
        )
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                axis=-1,
            ),
            jnp.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                axis=-1,
            ),
            jnp.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def uma_source_wigner(rotation: ArrayLike, lmax: int, /) -> Array:
    """Block-diagonal source-basis Wigner matrix ``(F, F)`` of one proper rotation.

    Each degree block is the native real Wigner matrix of ``P R P^T``.
    """
    matrix = jnp.asarray(rotation)
    frame = jnp.asarray(UMA_SOURCE_FRAME, dtype=matrix.dtype)
    actions = _proper_actions(frame @ matrix @ frame.T, lmax)
    return block_diag(*actions)


def uma_edge_wigner(
    vector: ArrayLike, layout: UMACoefficientLayout, /, *, roll: ArrayLike | None = None
) -> tuple[Array, Array]:
    """Source ``prepare_wigner`` pair of one edge: ``(R x F, F x R)`` m-major maps.

    ``wigner`` rotates full l-major node coefficients into the reduced m-major
    edge frame (source ``to_m @ D[coefficient_index]``); ``wigner_inv`` is its
    transpose (the source's ``D^T[:, coefficient_index] @ to_m^T``). ``roll`` is
    an explicit edge roll (`uma_edge_rotation`).
    """
    full = uma_source_wigner(uma_edge_rotation(vector, roll=roll), layout.lmax)
    rows = full[jnp.asarray(layout.edge_rows, dtype=jnp.int32)]
    return rows, rows.T


def _words(values: Array, count: int, /) -> list[Array]:
    """Full-width uint32 words of one integer route field (two's complement)."""
    flat = values.reshape(count, -1)
    if not jnp.issubdtype(flat.dtype, jnp.integer):
        raise TypeError("Route identity fields must be integers.")
    if flat.dtype.itemsize == 8:
        bits = jax.lax.bitcast_convert_type(flat, jnp.uint64)
        return [
            jax.lax.convert_element_type(bits >> 32, jnp.uint32),
            jax.lax.convert_element_type(bits & 0xFFFFFFFF, jnp.uint32),
        ]
    wide = flat.astype(jnp.int32) if flat.dtype.itemsize < 4 else flat
    return [jax.lax.bitcast_convert_type(wide, jnp.uint32)]


def uma_keyed_edge_rolls(
    key: ArrayLike, fields: Sequence[ArrayLike], /, *, dtype: DTypeLike
) -> Array:
    """Source-distributed edge rolls ``2 pi U[0, 1)`` addressed by route identity.

    Each route's key is ``fold_in`` of the explicit ``key`` with a UMA domain
    tag and then with every 32-bit word of its identity fields, in order
    (64-bit fields contribute both words, so identities are never truncated).
    The rolls depend only on the key and the route identities, never on route
    storage order, and are constants of the geometry and the parameters.
    """
    arrays = [jnp.asarray(field) for field in fields]
    if not arrays:
        raise ValueError("Route rolls need at least one identity field.")
    count = arrays[0].shape[0]
    if any(array.ndim < 1 or array.shape[0] != count for array in arrays):
        raise ValueError("Route identity fields must share their route axis.")
    words = jnp.concatenate(
        [word for array in arrays for word in _words(array, count)], axis=1
    )
    base = jr.fold_in(key, _ROLL_DOMAIN)

    def roll(row: Array, /) -> Array:
        route = base
        for index in range(row.shape[0]):
            route = jr.fold_in(route, row[index])
        return jr.uniform(route, (), dtype=dtype)

    return (2.0 * math.pi) * jax.vmap(roll)(words)


def uma_source_basis_residual(blocks: Sequence[ArrayLike], /) -> float:
    """Largest deviation of stored e3nn ``J_l`` from ``D_native_l(P J_1 P^T)``.

    The source registers e3nn's per-degree ``J`` matrices (``Jd_l``), the source
    representation of one fixed proper rotation whose degree-one block is that
    rotation in ``(x, y, z)`` order. Agreement for every stored degree proves
    the source basis is the relabelled native basis (``J`` is only defined up
    to that representation; a different basis convention cannot pass).
    """
    arrays = [np.asarray(block, dtype=np.float64) for block in blocks]
    if len(arrays) < 2:
        raise ValueError("Basis evidence needs the stored J blocks of degrees 0 and 1.")
    for degree, array in enumerate(arrays):
        size = 2 * degree + 1
        if array.shape != (size, size) or not np.all(np.isfinite(array)):
            raise ValueError(f"Stored J block {degree} must be finite ({size}, {size}).")
    rotation = arrays[1]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6) or (
        np.linalg.det(rotation) < 0.0
    ):
        raise ValueError("The stored degree-one J block is not a proper rotation.")
    actions = _host_proper_actions(
        UMA_SOURCE_FRAME @ rotation @ UMA_SOURCE_FRAME.T, len(arrays) - 1
    )
    return float(
        max(np.max(np.abs(array - action)) for array, action in zip(arrays, actions))
    )


def _quadrature_weights(half: int, /) -> np.ndarray:
    """e3nn ``_quadrature_weights(b)`` (Kostelec--Rockmore), float64."""
    k = np.arange(half, dtype=np.float64)
    weights = np.empty(2 * half, dtype=np.float64)
    for j in range(2 * half):
        weights[j] = (
            (2.0 / half)
            * math.sin(math.pi * (2.0 * j + 1.0) / (4.0 * half))
            * float(
                np.sum(
                    (1.0 / (2.0 * k + 1.0))
                    * np.sin((2.0 * j + 1.0) * (2.0 * k + 1.0) * math.pi / (4.0 * half))
                )
            )
        )
    return weights / (2.0 * (2 * half) ** 2)


def uma_grid_resolution(
    lmax: int, mmax: int, resolution: int | None, /
) -> tuple[int, int]:
    """Source ``SO3_Grid`` latitude/longitude counts, validated as e3nn does."""
    if resolution is None:
        latitude = 2 * (lmax + 1)
        longitude = 2 * (mmax + 1) + 1 if lmax == mmax else 2 * mmax + 1
    else:
        latitude = longitude = positive_integer(resolution, "grid resolution")
    if latitude % 2 or lmax + 1 > latitude // 2:
        raise ValueError(
            "The UMA grid latitude count must be even and at least 2 (lmax + 1)."
        )
    return latitude, longitude


@lru_cache(maxsize=16)
def _grid_tables(
    lmax: int, mmax: int, latitude: int, longitude: int, rescale: bool, /
) -> tuple[np.ndarray, np.ndarray]:
    betas = (np.arange(latitude, dtype=np.float64) + 0.5) / latitude * math.pi
    alphas = np.arange(longitude, dtype=np.float64) / longitude * 2.0 * math.pi
    beta, alpha = np.meshgrid(betas, alphas, indexing="ij")
    # Source point (sin b sin a, cos b, sin b cos a) in native coordinates.
    native = np.stack(
        [np.sin(beta) * np.cos(alpha), np.sin(beta) * np.sin(alpha), np.cos(beta)],
        axis=-1,
    )
    harmonics = np.asarray(
        RealCartesianHarmonics(lmax, normalization="orthonormal")(native)
    )
    if harmonics.dtype != np.float64:
        raise ValueError(
            "UMA grid preparation evaluates harmonics in float64; enable JAX x64."
        )
    weights = _quadrature_weights(latitude // 2) * latitude**2 / longitude
    to_grid = harmonics.copy()
    from_grid = 4.0 * math.pi * weights[:, None, None] * harmonics
    if rescale and lmax != mmax:
        for degree in range(mmax + 1, lmax + 1):
            span = slice(degree * degree, (degree + 1) * (degree + 1))
            factor = math.sqrt((2 * degree + 1) / (2 * mmax + 1))
            to_grid[..., span] *= factor
            from_grid[..., span] *= factor
    reduced = np.asarray(_coefficient_tables(lmax, mmax)[0], dtype=np.int64)
    to_table = np.ascontiguousarray(to_grid[..., reduced])
    from_table = np.ascontiguousarray(from_grid[..., reduced])
    to_table.setflags(write=False)
    from_table.setflags(write=False)
    return to_table, from_table


@final
class UMASphericalGrid(StrictModule, NonTrainableState):
    """Source ``SO3_Grid`` projection tables ``(B, A, R)`` in reduced l-major order."""

    __strict_contract__ = True

    to_grid: Float[UMAGridLatitudeDim, UMAGridLongitudeDim, UMAReducedCoefficientDim]
    from_grid: Float[UMAGridLatitudeDim, UMAGridLongitudeDim, UMAReducedCoefficientDim]
    lmax: int = eqx.field(static=True)
    mmax: int = eqx.field(static=True)
    latitude: Size[UMAGridLatitudeDim] = eqx.field(static=True)
    longitude: Size[UMAGridLongitudeDim] = eqx.field(static=True)
    rescale: bool = eqx.field(static=True)
    grid_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        lmax: int,
        mmax: int,
        /,
        *,
        resolution: int | None,
        dtype: DTypeLike,
        rescale: bool = True,
    ) -> None:
        layout = UMACoefficientLayout(lmax, mmax)
        latitude, longitude = uma_grid_resolution(layout.lmax, layout.mmax, resolution)
        if not isinstance(rescale, bool):
            raise TypeError("rescale must be a boolean.")
        to_table, from_table = _grid_tables(
            layout.lmax, layout.mmax, latitude, longitude, rescale
        )
        self.to_grid = jnp.asarray(to_table, dtype=dtype)
        self.from_grid = jnp.asarray(from_table, dtype=dtype)
        self.lmax = layout.lmax
        self.mmax = layout.mmax
        self.latitude = latitude
        self.longitude = longitude
        self.rescale = rescale
        self.grid_id = canonical_fingerprint(
            {
                "kind": "uma-e3nn-integral-s2-grid",
                "lmax": layout.lmax,
                "mmax": layout.mmax,
                "latitude": latitude,
                "longitude": longitude,
                "rescale": rescale,
            }
        )

    def validate(self) -> None:
        """Recompute the tables and compare them with the stored arrays."""
        to_table, from_table = _grid_tables(
            self.lmax, self.mmax, self.latitude, self.longitude, self.rescale
        )
        dtype = np.dtype(self.to_grid.dtype)
        for stored, expected in ((self.to_grid, to_table), (self.from_grid, from_table)):
            if not np.array_equal(np.asarray(stored), expected.astype(dtype)):
                raise ValueError("UMA grid tables do not match their definition.")

    def project(self, values: Array, /) -> Array:
        """Reduced l-major coefficients ``(..., R, C)`` to grid values ``(..., B, A, C)``."""
        return contract("bai,...ic->...bac", self.to_grid, values)

    def integrate(self, values: Array, /) -> Array:
        """Grid values ``(..., B, A, C)`` to reduced l-major coefficients ``(..., R, C)``."""
        return contract("bai,...bac->...ic", self.from_grid, values)

    def project_m_major(self, values: Array, layout: UMACoefficientLayout, /) -> Array:
        """m-major edge coefficients ``(..., R, C)`` to grid values (source ``to_grid_mat_m``)."""
        columns = self.to_grid[..., jnp.asarray(layout.m_major, dtype=jnp.int32)]
        return contract("bai,...ic->...bac", columns, values)

    def integrate_m_major(self, values: Array, layout: UMACoefficientLayout, /) -> Array:
        """Grid values to m-major edge coefficients (source ``from_grid_mat_m``)."""
        columns = self.from_grid[..., jnp.asarray(layout.m_major, dtype=jnp.int32)]
        return contract("bai,...bac->...ic", columns, values)


register_artifact_value("phydrax.nn.atomistic:UMACoefficientLayout", UMACoefficientLayout)
register_artifact_value("phydrax.nn.atomistic:UMASphericalGrid", UMASphericalGrid)


__all__ = [
    "UMA_SOURCE_FRAME",
    "UMACoefficientLayout",
    "UMAEdgeRollGauge",
    "UMASphericalGrid",
    "uma_edge_rotation",
    "uma_edge_wigner",
    "uma_keyed_edge_rolls",
    "uma_grid_resolution",
    "uma_source_basis_residual",
    "uma_source_wigner",
]
