#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Point and Gaussian multipole electrostatics through public degree 2.

Physical definitions (:class:`MultipoleElectrostaticsDefinition`)
----------------------------------------------------------------
Sites carry :class:`~phydrax.atomistic._multipole_basis.CartesianMultipoles`
(site-centred primitive coefficients, ``1/2 Theta : grad grad`` convention).
The energy is ``E = 1/2 B(M, M)`` of the symmetric bilinear interaction form

``B(L, R) = k sum'_{i, j, n} O_{L_i}(-grad) O_{R_j}(grad) phi_ij(R),
R = r_i - r_j + n H``

with ``O_M(grad) = q - mu . grad + 1/2 Theta : grad grad``. ``"point"`` sites
use ``phi = 1/R`` and exclude every co-centred term (``n = 0, i = j``).
``"gaussian"`` sites (unit-normalised Gaussians of per-site standard deviation
``sigma``) use ``phi_ij = erf(R / sqrt(2 (sigma_i^2 + sigma_j^2))) / R`` and
declare their intrasite energy: ``"exclude"`` (the pinned source default) or
``"include"`` (the complete smeared-density energy, adding
``k/2 O(-grad) O(grad) erf(r/(2 sigma))/r |_0`` per site). Cartesian
derivatives of ``phi`` use ``B_0 = phi``, ``B_{n+1} = -(1/R) dB_n/dR``.

Periodic sums exclude ``k = 0`` (tin-foil boundary). ``"require-neutral"``
refuses a charged cell; ``"uniform-background"`` adds
``-pi k Q^2 / (2 alpha^2 V)``, the ``k -> 0`` limit of the Ewald split, which
is identical for point and Gaussian charges. The alpha-independent Gaussian
form-factor constant ``-2 pi k (sigma_i^2 + sigma_j^2) q_i q_j / V`` of the
physical kernel's ``k -> 0`` expansion belongs to the excluded ``k = 0`` term
by this convention (as in the pinned source). Only the charge-charge
``k = 0`` limit is regularised: dipole and second-moment couplings at
``k = 0`` are excluded by the tin-foil convention.

Numerical routes (separate identities)
--------------------------------------
* :class:`FiniteMultipoleRoute` -- the complete nonperiodic pair sum over a
  declared bounded all-pairs reference relation (``O(N^2)`` work, bounded
  ``row_block x N`` workspace).
* :class:`EwaldMultipoleRoute` -- Ewald split with explicit integer modes from
  the shared :class:`~phydrax.atomistic._reciprocal_mesh.ReciprocalModeSetPlan`;
  the multipole structure factor
  ``S(k) = sum_j (q_j + i k.mu_j - 1/2 k.Theta_j.k) exp(-sigma_j^2 k^2/2) e^{i k.r_j}``
  is evaluated per bounded mode chunk (direct k).
* :class:`ParticleMeshMultipoleRoute` -- smooth PME on the shared
  :class:`~phydrax.atomistic._reciprocal_mesh.PreparedReciprocalMesh`: every
  Cartesian component of every Gaussian width class is one B-spline channel;
  the ``i k`` and ``-k k/2`` multipole factors and the width factors are
  applied exactly in reciprocal space on the current-cell wavevectors and the
  shared ``cross_quadratic`` removes the declared assignment modulus. Under the
  ``"symmetrized"`` Nyquist convention the complete k-dependent quadratic is
  averaged over every Nyquist sign variant of the current cell; ``"excluded"``
  drops those planes and is a different, separately identified truncation.

Real space uses either the declared explicit-image reference relation (every
lattice translation of :meth:`PeriodicCell.image_stencil`, with a traced
completeness margin) or a bound
:class:`~phydrax.atomistic._neighborhood_bundle.AtomisticNeighborhoodView` of
the ``image-pair-sum`` role. Positions, cell vectors and coefficients are all
differentiated through the same scalar; no current-cell coefficient is cached.
"""

from __future__ import annotations

import abc
import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._jax_config import x64_enabled
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleDiscretization, PeriodicCell, PeriodicImageStencil
from ..typing import parse
from ._conditioning import (
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
)
from ._multipole_basis import (
    CartesianMultipoles,
    co_centered_interaction,
    component_count,
    erf_kernel_origin_coefficients,
    GaussianMultipoleWidths,
    MultipoleBasis,
    second_moment_matrix,
    unique_second_moments,
    unpack_multipoles,
)
from ._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodView,
    AtomisticPairExclusions,
)
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from ._reciprocal_mesh import (
    PreparedReciprocalMesh,
    PreparedReciprocalModeSet,
    ReciprocalGeometry,
    ReciprocalMeshDensity,
    ReciprocalMeshPlan,
    ReciprocalModeSetPlan,
)
from ._sites import AtomisticSiteDomain
from ._system import PreparedAtomisticSystem


MultipoleSiteKind: TypeAlias = Literal["point", "gaussian"]
MultipoleIntrasitePolicy: TypeAlias = Literal["exclude", "include"]
MultipoleNeutralityPolicy: TypeAlias = Literal["require-neutral", "uniform-background"]
MultipoleRealSpace: TypeAlias = Literal["explicit-image-reference", "neighborhood"]

_SQRT_PI = math.sqrt(math.pi)
_SERIES_SWITCH = 1.0
_SERIES_TERMS = 20
_REAL_ITEMSIZE = 8
_COMPLEX_ITEMSIZE = 16
# Per-pair transient arrays of one bilinear evaluation (displacement, kernel
# coefficients, contractions): a conservative byte estimate per pair route.
_PAIR_WORKSPACE_BYTES = 64 * _REAL_ITEMSIZE


def _finite_positive(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _identity(kind: str, values: dict, /) -> str:
    return canonical_fingerprint({"kind": kind, **values})


# --- site data -----------------------------------------------------------------


class MultipoleSiteSet(StrictModule):
    """Bound coefficients of one evaluation with their site parameters.

    ``widths`` is ``None`` for point sites. ``damping`` is an optional
    per-site parameter consumed only by kernels that declare it (for example
    the polarization damping kernel). Inactive sites contribute nothing.
    """

    moments: CartesianMultipoles
    widths: GaussianMultipoleWidths | None
    active: Array
    damping: Array | None

    def __init__(
        self,
        moments: CartesianMultipoles,
        /,
        *,
        active: ArrayLike,
        widths: GaussianMultipoleWidths | None = None,
        damping: ArrayLike | None = None,
    ) -> None:
        if not isinstance(moments, CartesianMultipoles):
            raise TypeError("moments must be CartesianMultipoles.")
        mask = jnp.asarray(active)
        if mask.shape != (moments.site_capacity,) or mask.dtype != jnp.bool_:
            raise ValueError("active must be a boolean array with shape (N,).")
        if widths is not None:
            if not isinstance(widths, GaussianMultipoleWidths):
                raise TypeError("widths must be GaussianMultipoleWidths or None.")
            if widths.site_capacity != moments.site_capacity:
                raise ValueError("Widths and moments must share their site capacity.")
        damp = None
        if damping is not None:
            damp = jnp.asarray(damping, dtype=moments.charges.dtype)
            if damp.shape != (moments.site_capacity,):
                raise ValueError("damping must have shape (N,).")
        self.moments = moments
        self.widths = widths
        self.active = mask
        self.damping = damp

    @property
    def site_capacity(self) -> int:
        return self.moments.site_capacity

    @property
    def maximum_degree(self) -> int:
        return self.moments.maximum_degree

    def masked_moments(self, /) -> CartesianMultipoles:
        return self.moments.masked(self.active)

    def width_squared(self, /) -> Array:
        dtype = self.moments.charges.dtype
        if self.widths is None:
            return jnp.zeros((self.site_capacity,), dtype=dtype)
        return self.widths.site_widths().astype(dtype) ** 2

    def damping_values(self, /) -> Array:
        dtype = self.moments.charges.dtype
        if self.damping is None:
            return jnp.ones((self.site_capacity,), dtype=dtype)
        return self.damping


class _PairSites(StrictModule):
    """Per-route gathered coefficients and site parameters."""

    charges: Array
    dipoles: Array
    second: Array
    width_squared: Array
    damping: Array


def _gather(
    site: MultipoleSiteSet, moments: CartesianMultipoles, index: Array, /
) -> _PairSites:
    return _PairSites(
        charges=moments.charges[index],
        dipoles=moments.dipoles[index],
        second=moments.second_moments[index],
        width_squared=site.width_squared()[index],
        damping=site.damping_values()[index],
    )


# --- radial kernels ------------------------------------------------------------


def _inverse_power(distance: Array, order: int, /) -> tuple[Array, ...]:
    inverse = 1.0 / distance
    inverse2 = inverse * inverse
    values = [inverse]
    for n in range(1, order + 1):
        values.append((2 * n - 1) * values[-1] * inverse2)
    return tuple(values)


def _erfc_coefficients(alpha: Array, distance: Array, order: int, /) -> tuple[Array, ...]:
    """``B_n`` of ``erfc(alpha r)/r`` by the stable upward recursion."""
    gauss = jnp.exp(-((alpha * distance) ** 2))
    inverse2 = 1.0 / (distance * distance)
    values = [jsp.erfc(alpha * distance) / distance]
    lead = 1.0 / (alpha * _SQRT_PI)
    for n in range(1, order + 1):
        lead = lead * 2.0 * alpha * alpha
        values.append(((2 * n - 1) * values[-1] + lead * gauss) * inverse2)
    return tuple(values)


def _erf_coefficients(beta: Array, distance: Array, order: int, /) -> tuple[Array, ...]:
    """``B_n`` of ``erf(beta r)/r`` with a justified small-argument series.

    For ``beta r < 1`` the alternating series
    ``B_n = (2 beta/sqrt(pi)) (2 beta^2)^n sum_j (-x^2)^j / (j! (2j+2n+1))``
    (20 terms, truncation below ``1/20!``) avoids the cancellation of
    ``(2n-1)!!/r^(2n+1) - B_n[erfc]``; otherwise that closed form is used.
    Both branches are evaluated on safe arguments so derivatives stay finite.
    """
    x = beta * distance
    small = x < _SERIES_SWITCH
    xs = jnp.where(small, x, 0.0)
    x2 = xs * xs
    terms = [jnp.ones_like(x2)]
    for j in range(1, _SERIES_TERMS):
        terms.append(terms[-1] * (-x2) / j)
    lead = 2.0 * beta / _SQRT_PI
    two_beta2 = 2.0 * beta * beta
    safe = jnp.where(small, _SERIES_SWITCH / beta, distance)
    closed_inverse = _inverse_power(safe, order)
    closed_erfc = _erfc_coefficients(beta, safe, order)
    values = []
    scale = lead
    for n in range(order + 1):
        series = scale * sum(term / (2 * j + 2 * n + 1) for j, term in enumerate(terms))
        values.append(jnp.where(small, series, closed_inverse[n] - closed_erfc[n]))
        scale = scale * two_beta2
    return tuple(values)


class AbstractMultipoleRadialKernel(StrictModule):
    """Radial pair kernel ``phi`` supplying ``B_0..B_order`` per route."""

    kernel_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def coefficients(
        self, distance: Array, left: _PairSites, right: _PairSites, order: int, /
    ) -> tuple[Array, ...]:
        raise NotImplementedError


class CoulombKernel(AbstractMultipoleRadialKernel):
    """``phi = 1/R`` (point sites, nonperiodic)."""

    kernel_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.kernel_id = "coulomb-inverse-distance"

    def coefficients(
        self, distance: Array, left: _PairSites, right: _PairSites, order: int, /
    ) -> tuple[Array, ...]:
        del left, right
        return _inverse_power(distance, order)


class ErfcKernel(AbstractMultipoleRadialKernel):
    """``phi = erfc(alpha R)/R`` (point Ewald real space)."""

    alpha: float = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(self, alpha: float, /) -> None:
        self.alpha = _finite_positive(alpha, "alpha")
        self.kernel_id = f"erfc-{float(self.alpha).hex()}"

    def coefficients(
        self, distance: Array, left: _PairSites, right: _PairSites, order: int, /
    ) -> tuple[Array, ...]:
        del left, right
        alpha = jnp.asarray(self.alpha, dtype=distance.dtype)
        return _erfc_coefficients(alpha, distance, order)


class GaussianKernel(AbstractMultipoleRadialKernel):
    """``phi = erf(R / sqrt(2 (s_i^2 + s_j^2))) / R`` (Gaussian, nonperiodic)."""

    kernel_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.kernel_id = "gaussian-erf"

    def coefficients(
        self, distance: Array, left: _PairSites, right: _PairSites, order: int, /
    ) -> tuple[Array, ...]:
        beta = jax.lax.rsqrt(2.0 * (left.width_squared + right.width_squared))
        return _erf_coefficients(beta, distance, order)


class GaussianEwaldRealKernel(AbstractMultipoleRadialKernel):
    """Gaussian Ewald real space ``[erf(b R) - erf(b' R)] / R``.

    ``b = 1/sqrt(2 g^2)``, ``b' = 1/sqrt(2 g^2 + 1/alpha^2)``,
    ``g^2 = s_i^2 + s_j^2``; the point limit is ``erfc(alpha R)/R``.
    """

    alpha: float = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(self, alpha: float, /) -> None:
        self.alpha = _finite_positive(alpha, "alpha")
        self.kernel_id = f"gaussian-ewald-real-{float(self.alpha).hex()}"

    def coefficients(
        self, distance: Array, left: _PairSites, right: _PairSites, order: int, /
    ) -> tuple[Array, ...]:
        spread = 2.0 * (left.width_squared + right.width_squared)
        physical = _erf_coefficients(jax.lax.rsqrt(spread), distance, order)
        smooth = _erf_coefficients(
            jax.lax.rsqrt(spread + 1.0 / self.alpha**2), distance, order
        )
        return tuple(a - b for a, b in zip(physical, smooth, strict=True))


def _pair_interaction(
    coefficients: tuple[Array, ...],
    displacement: Array,
    left: _PairSites,
    right: _PairSites,
    degree: int,
    /,
) -> Array:
    """``O_L(-grad) O_R(grad) phi(R)`` from ``B_0..B_{2 degree}`` (broadcasting)."""
    b = coefficients
    R = displacement
    value = left.charges * right.charges * b[0]
    if degree >= 1:
        a_left = jnp.sum(left.dipoles * R, axis=-1)
        a_right = jnp.sum(right.dipoles * R, axis=-1)
        dot = jnp.sum(left.dipoles * right.dipoles, axis=-1)
        value = value + b[1] * (left.charges * a_right - right.charges * a_left)
        value = value - (a_left * a_right * b[2] - dot * b[1])
    if degree == 2:
        t_left = jnp.trace(left.second, axis1=-2, axis2=-1)
        t_right = jnp.trace(right.second, axis1=-2, axis2=-1)
        theta_left_r = jnp.sum(left.second * R[..., None, :], axis=-1)
        theta_right_r = jnp.sum(right.second * R[..., None, :], axis=-1)
        r_left = jnp.sum(R * theta_left_r, axis=-1)
        r_right = jnp.sum(R * theta_right_r, axis=-1)
        value = value + 0.5 * left.charges * (r_right * b[2] - t_right * b[1])
        value = value + 0.5 * right.charges * (r_left * b[2] - t_left * b[1])
        value = value + 0.5 * (
            -a_left * r_right * b[3]
            + (2.0 * jnp.sum(left.dipoles * theta_right_r, axis=-1) + a_left * t_right)
            * b[2]
        )
        value = value - 0.5 * (
            -a_right * r_left * b[3]
            + (2.0 * jnp.sum(right.dipoles * theta_left_r, axis=-1) + a_right * t_left)
            * b[2]
        )
        value = value + 0.25 * (
            r_left * r_right * b[4]
            - (
                t_left * r_right
                + t_right * r_left
                + 4.0 * jnp.sum(theta_left_r * theta_right_r, axis=-1)
            )
            * b[3]
            + (
                t_left * t_right
                + 2.0 * jnp.sum(left.second * right.second, axis=(-2, -1))
            )
            * b[2]
        )
    return value


# --- pair scale tables ------------------------------------------------------------


class MultipolePairScales(StrictModule, NonTrainableState):
    """Sparse per-pair interaction scales with named columns (unlisted = 1).

    Each unordered pair ``(i, j)``, ``i != j``, is listed once. Lookups use the
    sorted directed keys ``i N + j`` of both orientations, never a dense
    ``N x N`` array.

    PyTree layout: ``columns`` and ``site_capacity`` are static; the listed
    pair count is the declared shape of ``pairs``/``keys``. The exact canonical
    digest of the listed pairs and values is the dynamic ``scales_identity``
    (``scales_id`` decodes it on the host).
    """

    pairs: Array
    values: Array
    keys: Array
    key_values: Array
    columns: tuple[str, ...] = eqx.field(static=True)
    site_capacity: int = eqx.field(static=True)
    scales_identity: IdentityWords

    def __init__(
        self,
        pairs: ArrayLike,
        values: ArrayLike,
        /,
        *,
        columns: tuple[str, ...],
        site_capacity: int,
    ) -> None:
        capacity = _positive_int(site_capacity, "site_capacity")
        names = tuple(str(item) for item in columns)
        if not names or len(set(names)) != len(names):
            raise ValueError("columns must be distinct non-empty names.")
        host_pairs = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
        host_values = np.asarray(values, dtype=np.float64).reshape(
            (host_pairs.shape[0], len(names))
        )
        if np.any(host_pairs < 0) or np.any(host_pairs >= capacity):
            raise ValueError("Scaled pairs must index the site capacity.")
        if np.any(host_pairs[:, 0] == host_pairs[:, 1]):
            raise ValueError("Scaled pairs must join distinct sites.")
        ordered = np.sort(host_pairs, axis=1)
        unique_keys = ordered[:, 0] * capacity + ordered[:, 1]
        if np.unique(unique_keys).size != unique_keys.size:
            raise ValueError("Each unordered scaled pair may be listed once.")
        if np.any(~np.isfinite(host_values)) or np.any(host_values < 0.0):
            raise ValueError("Pair scales must be finite and non-negative.")
        if capacity * capacity >= 2**31 and not x64_enabled():
            raise ValueError("Pair keys of this capacity require 64-bit JAX integers.")
        directed = np.concatenate(
            (
                host_pairs[:, 0] * capacity + host_pairs[:, 1],
                host_pairs[:, 1] * capacity + host_pairs[:, 0],
            )
        )
        order = np.argsort(directed)
        key_dtype = jnp.int64 if capacity * capacity >= 2**31 else jnp.int32
        self.pairs = jnp.asarray(host_pairs, dtype=jnp.int32)
        self.values = jnp.asarray(host_values)
        self.keys = jnp.asarray(directed[order], dtype=key_dtype)
        self.key_values = jnp.asarray(np.concatenate((host_values, host_values))[order])
        self.columns = names
        self.site_capacity = capacity
        self.scales_identity = IdentityWords.of(
            {
                "kind": "multipole-pair-scales",
                "columns": list(names),
                "site_capacity": capacity,
                "pairs": host_pairs.tolist(),
                "values": [[float(item).hex() for item in row] for row in host_values],
            }
        )

    @property
    def scales_id(self) -> str:
        """Host decoding of the canonical pair-scale digest."""
        return self.scales_identity.fingerprint()

    @property
    def pair_count(self) -> int:
        return int(self.pairs.shape[0])

    def column(self, name: str, /) -> int:
        if name not in self.columns:
            raise ValueError(f"Unknown pair-scale column {name!r}.")
        return self.columns.index(name)

    def lookup(self, receiver: Array, source: Array, column: int, /) -> Array:
        """Scale of every directed route (``1`` for unlisted pairs)."""
        if self.pair_count == 0:
            return jnp.ones(jnp.broadcast_shapes(receiver.shape, source.shape))
        key = receiver.astype(self.keys.dtype) * self.site_capacity + source.astype(
            self.keys.dtype
        )
        position = jnp.clip(jnp.searchsorted(self.keys, key), 0, self.keys.shape[0] - 1)
        found = self.keys[position] == key
        return jnp.where(found, self.key_values[position, column], 1.0)


class _ScaleColumn(StrictModule):
    table: MultipolePairScales
    column: int = eqx.field(static=True)


# --- real-space relations ---------------------------------------------------------


class _PairSum(StrictModule):
    value: Array
    geometry_successful: Array
    coverage_successful: Array


def _admissible_cell_vectors(
    cell: PeriodicCell, vectors: Array, /
) -> tuple[Array, Array]:
    """Return consumable cell vectors and the native lattice-solve status.

    Inadmissible (singular or nonfinite) vectors are replaced by the prepared
    cell before any wrap, minimum-image or reach computation consumes them, so
    the native solve failure becomes this relation's reported geometry status
    instead of a runtime abort; callers fold the status into evidence and the
    fail-closed NaN result.
    """
    _, solved = cell.inverse_for_vectors_with_status(vectors)
    admissible = jnp.all(solved) & jnp.all(jnp.isfinite(vectors))
    prepared = cell.vectors.astype(vectors.dtype)
    return jnp.where(admissible, vectors, prepared), admissible


class AbstractMultipolePairs(StrictModule):
    """A declared real-space route set evaluating one bilinear pair sum."""

    relation_id: eqx.AbstractVar[str]
    site_capacity: eqx.AbstractVar[int]

    @abc.abstractmethod
    def pair_sum(
        self,
        positions: Array,
        cell_vectors: Array | None,
        left: MultipoleSiteSet,
        right: MultipoleSiteSet,
        kernel: AbstractMultipoleRadialKernel,
        /,
        *,
        cutoff: float | None,
        minimum_distance: float,
        scale: _ScaleColumn | None,
        symmetric: bool,
    ) -> _PairSum:
        raise NotImplementedError


class ExplicitImagePairs(AbstractMultipolePairs):
    """Declared bounded reference relation over every pair and lattice image.

    Nonperiodic: every ordered pair ``i != j``. Periodic: every
    ``(i, j, n)`` except the co-centred ``i = j, n = 0`` route, over the
    complete :meth:`PeriodicCell.image_stencil` of the cutoff for wrapped
    coordinates; a traced margin certifies completeness under the current
    cell. Work is ``O(N^2 S)``; resident workspace is bounded by
    ``row_block x N x S`` routes, admitted against
    ``maximum_workspace_bytes`` at preparation.
    """

    cell: PeriodicCell | None
    stencil: PeriodicImageStencil | None
    site_capacity: int = eqx.field(static=True)
    row_block: int = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_capacity: int,
        /,
        *,
        cell: PeriodicCell | None = None,
        cutoff: float | None = None,
        row_block: int = 32,
        maximum_image_count: int = 4096,
        maximum_workspace_bytes: int = 1 << 28,
    ) -> None:
        capacity = _positive_int(site_capacity, "site_capacity")
        block = min(_positive_int(row_block, "row_block"), capacity)
        stencil = None
        if cell is not None:
            if not isinstance(cell, PeriodicCell) or cell.rank != 3:
                raise ValueError("Explicit image pairs require a rank-3 PeriodicCell.")
            if cutoff is None:
                raise ValueError("A periodic explicit-image relation requires a cutoff.")
            stencil = cell.image_stencil(
                _finite_positive(cutoff, "cutoff"),
                maximum_image_count=_positive_int(
                    maximum_image_count, "maximum_image_count"
                ),
                fractional_excursion=1.0,
            )
        elif cutoff is not None:
            raise ValueError("A nonperiodic all-pairs relation is complete; omit cutoff.")
        images = 1 if stencil is None else stencil.image_count
        workspace = block * capacity * images * _PAIR_WORKSPACE_BYTES
        if workspace > _positive_int(maximum_workspace_bytes, "maximum_workspace_bytes"):
            raise ValueError(
                f"Explicit-image reference workspace {workspace} bytes exceeds "
                f"maximum_workspace_bytes={maximum_workspace_bytes}; reduce row_block "
                "or use the neighborhood relation."
            )
        self.cell = cell
        self.stencil = stencil
        self.site_capacity = capacity
        self.row_block = block
        self.relation_id = _identity(
            "multipole-explicit-image-pairs",
            {
                "site_capacity": capacity,
                "row_block": block,
                "stencil": None if stencil is None else stencil.stencil_id,
            },
        )

    def pair_sum(
        self,
        positions: Array,
        cell_vectors: Array | None,
        left: MultipoleSiteSet,
        right: MultipoleSiteSet,
        kernel: AbstractMultipoleRadialKernel,
        /,
        *,
        cutoff: float | None,
        minimum_distance: float,
        scale: _ScaleColumn | None,
        symmetric: bool,
    ) -> _PairSum:
        del symmetric
        count = self.site_capacity
        dtype = positions.dtype
        degree = max(left.maximum_degree, right.maximum_degree)
        order = 2 * degree
        left_moments, right_moments = left.masked_moments(), right.masked_moments()
        if self.stencil is None or self.cell is None:
            if cell_vectors is not None:
                raise ValueError("A nonperiodic relation received cell vectors.")
            wrapped = positions
            translations = jnp.zeros((1, 3), dtype=dtype)
            coverage = jnp.asarray(True)
            zero_shift = jnp.ones((1,), dtype=jnp.bool_)
            admissible = jnp.asarray(True)
        else:
            if cell_vectors is None or cutoff is None:
                raise ValueError(
                    "A periodic relation requires cell vectors and a cutoff."
                )
            if scale is not None:
                raise ValueError(
                    "Periodic pair scales apply to one image; use WeightedPairCorrections."
                )
            vectors, admissible = _admissible_cell_vectors(
                self.cell, cell_vectors.astype(dtype)
            )
            wrapped, _ = self.cell.wrap_with_vectors(positions, vectors)
            shifts = self.stencil.shifts.astype(dtype)
            translations = shifts @ vectors
            zero_shift = jnp.all(self.stencil.shifts == 0, axis=-1)
            margin = self.cell.image_stencil_margin(self.stencil, vectors, cutoff, 1.0)
            coverage = (
                admissible & jnp.all(margin > 0.0) & (cutoff <= self.stencil.radius)
            )
        blocks = -(-count // self.row_block)
        rows = jnp.arange(blocks * self.row_block, dtype=jnp.int32).reshape(
            (blocks, self.row_block)
        )
        columns = jnp.arange(count, dtype=jnp.int32)
        right_sites = _gather(right, right_moments, columns)
        right_sites = jax.tree_util.tree_map(
            lambda value: value[None, :, None], right_sites
        )
        active_columns = right.active[None, :, None]

        def body(
            carry: tuple[Array, Array], row: Array
        ) -> tuple[tuple[Array, Array], None]:
            total, distinct = carry
            valid_row = row < count
            safe_row = jnp.where(valid_row, row, 0)
            left_sites = jax.tree_util.tree_map(
                lambda value: value[:, None, None],
                _gather(left, left_moments, safe_row),
            )
            displacement = (
                wrapped[safe_row][:, None, None, :]
                - wrapped[None, :, None, :]
                + translations[None, None, :, :]
            )
            squared = jnp.sum(displacement * displacement, axis=-1)
            same = safe_row[:, None, None] == columns[None, :, None]
            routed = (
                valid_row[:, None, None]
                & left.active[safe_row][:, None, None]
                & active_columns
                & ~(same & zero_shift[None, None, :])
            )
            if scale is not None:
                factor = scale.table.lookup(
                    safe_row[:, None, None], columns[None, :, None], scale.column
                )
                factor = jnp.where(zero_shift[None, None, :], factor, 1.0)
                routed = routed & (factor != 0.0)
            else:
                factor = None
            if cutoff is not None:
                routed = routed & (squared < cutoff * cutoff)
            positive = squared > minimum_distance * minimum_distance
            safe = jnp.sqrt(jnp.where(routed & positive, squared, 1.0))
            coefficients = kernel.coefficients(safe, left_sites, right_sites, order)
            value = _pair_interaction(
                coefficients, displacement, left_sites, right_sites, degree
            )
            if factor is not None:
                value = factor * value
            total = total + jnp.sum(jnp.where(routed & positive, value, 0.0))
            distinct = distinct & jnp.all(~routed | positive)
            return (total, distinct), None

        (total, distinct), _ = jax.lax.scan(
            jax.checkpoint(body), (jnp.zeros((), dtype=dtype), jnp.asarray(True)), rows
        )
        return _PairSum(total, distinct & admissible, coverage)

    def row_tensor_norm_sums(
        self,
        positions: Array,
        sites: MultipoleSiteSet,
        weights: Array,
        kernel: AbstractMultipoleRadialKernel,
        /,
        *,
        minimum_distance: float,
        scale: _ScaleColumn | None,
    ) -> tuple[Array, Array]:
        """Rows ``sum_{j != i} |w_i w_j s_ij| ||grad grad phi_ij||_2`` (nonperiodic).

        ``grad grad phi = B_2 R R^T - B_1 I`` has eigenvalues ``B_2 r^2 - B_1``
        and ``-B_1`` (twice), so its spectral norm is
        ``max(|B_2 r^2 - B_1|, |B_1|)`` for the kernel's actual coefficients
        (damping and Gaussian widths included). A contributing coincident pair
        makes its row infinite. Work is ``O(N^2)``; resident memory is one
        ``row_block x N`` block. Returns the row sums and whether every
        contributing pair is distinct.
        """
        if self.stencil is not None or self.cell is not None:
            raise ValueError(
                "Row tensor-norm bounds cover the nonperiodic relation only; a periodic "
                "operator also has reciprocal and intrasite parts."
            )
        count = self.site_capacity
        dtype = positions.dtype
        moments = sites.masked_moments()
        blocks = -(-count // self.row_block)
        rows = jnp.arange(blocks * self.row_block, dtype=jnp.int32).reshape(
            (blocks, self.row_block)
        )
        columns = jnp.arange(count, dtype=jnp.int32)
        right_sites = jax.tree_util.tree_map(
            lambda value: value[None, :], _gather(sites, moments, columns)
        )
        weight = jnp.abs(weights.astype(dtype))

        def body(distinct: Array, row: Array) -> tuple[Array, Array]:
            valid_row = row < count
            safe_row = jnp.where(valid_row, row, 0)
            left_sites = jax.tree_util.tree_map(
                lambda value: value[:, None], _gather(sites, moments, safe_row)
            )
            displacement = positions[safe_row][:, None, :] - positions[None, :, :]
            squared = jnp.sum(displacement * displacement, axis=-1)
            factor = weight[safe_row][:, None] * weight[None, :]
            if scale is not None:
                factor = factor * jnp.abs(
                    scale.table.lookup(safe_row[:, None], columns[None, :], scale.column)
                )
            routed = (
                valid_row[:, None]
                & (safe_row[:, None] != columns[None, :])
                & (factor != 0.0)
            )
            positive = squared > minimum_distance * minimum_distance
            safe = jnp.sqrt(jnp.where(routed & positive, squared, 1.0))
            coefficients = kernel.coefficients(safe, left_sites, right_sites, 2)
            first, second = coefficients[1], coefficients[2]
            norm = jnp.maximum(jnp.abs(second * safe * safe - first), jnp.abs(first))
            contribution = jnp.where(routed & positive, factor * norm, 0.0)
            contribution = jnp.where(routed & ~positive, jnp.inf, contribution)
            return distinct & jnp.all(~routed | positive), jnp.sum(contribution, axis=1)

        distinct, sums = jax.lax.scan(body, jnp.asarray(True), rows)
        return sums.reshape((-1,))[:count], distinct


class NeighborhoodPairs(AbstractMultipolePairs):
    """Scalable real space over a bound ``image-pair-sum`` neighborhood view.

    Displacements are recomputed from current positions and cell vectors by
    the view. A half view (``directed_multiplicity == 2``) contributes
    ``U(L_r, R_s; R) + U(L_s, R_r; -R)`` per stored route (including nonzero
    self images); a directed view contributes half of that. The view's own
    evidence (stale, capacity, scientific) gates the result.
    """

    view: AtomisticNeighborhoodView
    site_capacity: int = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)

    def __init__(self, view: AtomisticNeighborhoodView, site_capacity: int, /) -> None:
        if not isinstance(view, AtomisticNeighborhoodView):
            raise TypeError("view must be an AtomisticNeighborhoodView.")
        if (
            view.role
            not in (
                AtomisticNeighborhoodRole.IMAGE_PAIR_SUM,
                AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH,
            )
            or view.selected
        ):
            raise ValueError(
                "Multipole real space requires the complete image-pair-sum role or an "
                "unselected directed image graph."
            )
        if view.entity not in (
            AtomisticSiteDomain.PHYSICAL_ATOMS,
            AtomisticSiteDomain.DOF_ATOMS,
        ):
            raise ValueError("Multipole image routes must address the system atoms.")
        self.view = view
        self.site_capacity = _positive_int(site_capacity, "site_capacity")
        self.relation_id = _identity(
            "multipole-neighborhood-pairs",
            {"requirement": view.requirement_id, "site_capacity": self.site_capacity},
        )

    def pair_sum(
        self,
        positions: Array,
        cell_vectors: Array | None,
        left: MultipoleSiteSet,
        right: MultipoleSiteSet,
        kernel: AbstractMultipoleRadialKernel,
        /,
        *,
        cutoff: float | None,
        minimum_distance: float,
        scale: _ScaleColumn | None,
        symmetric: bool,
    ) -> _PairSum:
        if cell_vectors is None or cutoff is None:
            raise ValueError(
                "Neighborhood real space requires cell vectors and a cutoff."
            )
        # A view without a cutoff routes every valid pair and covers any radius.
        covered = self.view.cutoff
        if covered is not None and covered < cutoff:
            raise ValueError("The neighborhood cutoff is below the real-space cutoff.")
        if scale is not None:
            raise ValueError(
                "Periodic pair scales apply to one image; use WeightedPairCorrections."
            )
        dtype = positions.dtype
        vectors = cell_vectors.astype(dtype)
        degree = max(left.maximum_degree, right.maximum_degree)
        order = 2 * degree
        displacement = self.view.displacement(positions, vectors)
        squared = jnp.sum(displacement * displacement, axis=-1)
        receiver = self.view.relation.receiver_indices
        source = self.view.relation.source_indices
        routed = (
            self.view.cutoff_mask(positions, vectors)
            & (squared < cutoff * cutoff)
            & left.active[receiver]
            & right.active[source]
        )
        if not symmetric:
            routed = routed | (
                self.view.cutoff_mask(positions, vectors)
                & (squared < cutoff * cutoff)
                & left.active[source]
                & right.active[receiver]
            )
        positive = squared > minimum_distance * minimum_distance
        safe = jnp.sqrt(jnp.where(routed & positive, squared, 1.0))
        left_moments, right_moments = left.masked_moments(), right.masked_moments()
        lr = _gather(left, left_moments, receiver)
        rs = _gather(right, right_moments, source)
        forward = _pair_interaction(
            kernel.coefficients(safe, lr, rs, order), displacement, lr, rs, degree
        )
        if symmetric:
            value = 2.0 * forward
        else:
            ls = _gather(left, left_moments, source)
            rr = _gather(right, right_moments, receiver)
            value = forward + _pair_interaction(
                kernel.coefficients(safe, ls, rr, order), -displacement, ls, rr, degree
            )
        weight = 0.5 * self.view.directed_multiplicity
        total = weight * jnp.sum(jnp.where(routed & positive, value, 0.0))
        distinct = jnp.all(~routed | positive)
        evidence = jnp.all(self.view.evidence.successful)
        return _PairSum(total, distinct & evidence, jnp.asarray(True))


class WeightedPairCorrections(StrictModule):
    """Explicit per-pair corrections ``sum_p w_p [U(L_i,R_j;d) + U(L_j,R_i;-d)]``.

    ``d`` is the plain difference (nonperiodic) or the current-cell minimum
    image, valid while ``|d|`` stays below half the smallest lattice plane
    spacing (a traced status, never a silent wrap choice).

    PyTree layout: ``column`` and the method ``offset`` are static selectors;
    the exact canonical relation digest (which includes the pair-scale
    content) is the dynamic ``relation_identity`` (``relation_id`` decodes it
    on the host).
    """

    table: MultipolePairScales
    cell: PeriodicCell | None
    column: int = eqx.field(static=True)
    offset: float = eqx.field(static=True)
    relation_identity: IdentityWords

    def __init__(
        self,
        table: MultipolePairScales,
        column: str,
        /,
        *,
        cell: PeriodicCell | None = None,
        offset: float = -1.0,
    ) -> None:
        if not isinstance(table, MultipolePairScales):
            raise TypeError("table must be MultipolePairScales.")
        self.table = table
        self.cell = cell
        self.column = table.column(column)
        self.offset = float(offset)
        self.relation_identity = IdentityWords.of(
            {
                "kind": "multipole-weighted-pair-corrections",
                "scales": table.scales_id,
                "column": column,
                "offset": self.offset.hex(),
                "cell": None if cell is None else cell.cell_id,
            }
        )

    @property
    def relation_id(self) -> str:
        """Host decoding of the canonical correction-relation digest."""
        return self.relation_identity.fingerprint()

    def pair_sum(
        self,
        positions: Array,
        cell_vectors: Array | None,
        left: MultipoleSiteSet,
        right: MultipoleSiteSet,
        kernel: AbstractMultipoleRadialKernel,
        /,
        *,
        minimum_distance: float,
    ) -> _PairSum:
        dtype = positions.dtype
        if self.table.pair_count == 0:
            return _PairSum(
                jnp.zeros((), dtype=dtype), jnp.asarray(True), jnp.asarray(True)
            )
        degree = max(left.maximum_degree, right.maximum_degree)
        order = 2 * degree
        first, second = self.table.pairs[:, 0], self.table.pairs[:, 1]
        difference = positions[first] - positions[second]
        unique = jnp.ones(first.shape, dtype=jnp.bool_)
        admissible = jnp.asarray(True)
        if self.cell is not None:
            if cell_vectors is None:
                raise ValueError("Periodic pair corrections require cell vectors.")
            vectors, admissible = _admissible_cell_vectors(
                self.cell, cell_vectors.astype(dtype)
            )
            difference = self.cell.minimum_image_with_vectors(difference, vectors)
            reach = self.cell.axis_reach_with_vectors(vectors)
            length = jnp.sqrt(jnp.sum(difference * difference, axis=-1))
            unique = jnp.all(length[:, None] * reach[None, :] < 0.5, axis=-1)
        weight = self.table.values[:, self.column] + self.offset
        routed = (
            (left.active[first] & right.active[second])
            | (left.active[second] & right.active[first])
        ) & (weight != 0.0)
        squared = jnp.sum(difference * difference, axis=-1)
        positive = squared > minimum_distance * minimum_distance
        safe = jnp.sqrt(jnp.where(routed & positive, squared, 1.0))
        left_moments, right_moments = left.masked_moments(), right.masked_moments()
        lf, rs = _gather(left, left_moments, first), _gather(right, right_moments, second)
        ls, rf = _gather(left, left_moments, second), _gather(right, right_moments, first)
        value = _pair_interaction(
            kernel.coefficients(safe, lf, rs, order), difference, lf, rs, degree
        ) + _pair_interaction(
            kernel.coefficients(safe, ls, rf, order), -difference, ls, rf, degree
        )
        total = jnp.sum(jnp.where(routed & positive, weight * value, 0.0))
        # Only contributing corrections need a unique minimum image and a
        # nonzero distance; inactive or exactly unit-scale rows are no-ops.
        geometry = admissible & jnp.all(~routed | (positive & unique))
        return _PairSum(total, geometry, jnp.asarray(True))


# --- reciprocal space -------------------------------------------------------------


def _ewald_kernel(
    squared: Array, volume: Array, alpha: float, constant: float, /
) -> Array:
    """``4 pi k exp(-k^2/(4 alpha^2)) / (V k^2)`` for ``k != 0``, else zero."""
    safe = jnp.where(squared > 0.0, squared, 1.0)
    return jnp.where(
        squared > 0.0,
        4.0 * jnp.pi * constant / volume * jnp.exp(-safe / (4.0 * alpha**2)) / safe,
        0.0,
    )


def _mode_coefficients(
    moments: CartesianMultipoles, wave: Array, degree: int, /
) -> tuple[Array, Array]:
    """Real and imaginary parts of ``q + i k.mu - 1/2 k.Theta.k`` (sites x modes)."""
    real = jnp.broadcast_to(
        moments.charges[:, None], (moments.site_capacity, wave.shape[0])
    )
    imaginary = jnp.zeros_like(real)
    if degree >= 1:
        imaginary = moments.dipoles @ wave.T
    if degree == 2:
        projected = contract("nab,ma,mb->nm", moments.second_moments, wave, wave)
        real = real - 0.5 * projected
    return real, imaginary


def _mode_set_bilinear(
    modes: PreparedReciprocalModeSet,
    geometry: ReciprocalGeometry,
    positions: Array,
    left: MultipoleSiteSet,
    right: MultipoleSiteSet,
    alpha: float,
    constant: float,
    /,
    *,
    symmetric: bool,
) -> tuple[Array, Array]:
    dtype = positions.dtype
    degree = max(left.maximum_degree, right.maximum_degree)
    base = jnp.where(modes.zero_mode, 0.0, modes.hermitian_weights).astype(dtype)
    squared = geometry.squared_wavenumber.astype(dtype)
    factor = base * _ewald_kernel(squared, geometry.volume.astype(dtype), alpha, constant)
    chunk = modes.mode_chunk
    wave = geometry.wavevectors.astype(dtype).reshape((modes.chunk_count, chunk, 3))
    factor = factor.reshape((modes.chunk_count, chunk))
    left_moments, right_moments = left.masked_moments(), right.masked_moments()
    left_width, right_width = left.width_squared(), right.width_squared()
    finite_positions = jnp.all(jnp.isfinite(positions))
    position = jnp.where(jnp.isfinite(positions), positions, 0.0)

    def structure(
        moments: CartesianMultipoles, width: Array, chunk_wave: Array, phase: Array
    ) -> tuple[Array, Array]:
        real, imaginary = _mode_coefficients(moments, chunk_wave, degree)
        k2 = jnp.sum(chunk_wave * chunk_wave, axis=-1)
        envelope = jnp.exp(-0.5 * width[:, None] * k2[None, :])
        cosine, sine = jnp.cos(phase), jnp.sin(phase)
        s_real = jnp.sum(envelope * (real * cosine - imaginary * sine), axis=0)
        s_imaginary = jnp.sum(envelope * (real * sine + imaginary * cosine), axis=0)
        return s_real, s_imaginary

    def body(total: Array, inputs: tuple[Array, Array]) -> tuple[Array, None]:
        chunk_wave, chunk_factor = inputs
        phase = position @ chunk_wave.T
        lr, li = structure(left_moments, left_width, chunk_wave, phase)
        if symmetric:
            rr, ri = lr, li
        else:
            rr, ri = structure(right_moments, right_width, chunk_wave, phase)
        return total + jnp.sum(chunk_factor * (lr * rr + li * ri)), None

    total, _ = jax.lax.scan(
        jax.checkpoint(body), jnp.zeros((), dtype=dtype), (wave, factor)
    )
    successful = geometry.successful & finite_positions & jnp.isfinite(total)
    return total, successful


def _channel_values(site: MultipoleSiteSet, degree: int, class_count: int, /) -> Array:
    """``(N, W K)`` B-spline channel values: Cartesian components per width class."""
    moments = site.masked_moments()
    parts = [moments.charges[:, None]]
    if degree >= 1:
        parts.append(moments.dipoles)
    if degree == 2:
        parts.append(unique_second_moments(moments.second_moments))
    values = jnp.concatenate(parts, axis=-1)
    if site.widths is None:
        return values
    classes = jax.nn.one_hot(site.widths.site_classes, class_count, dtype=values.dtype)
    return (classes[:, :, None] * values[:, None, :]).reshape(
        (site.site_capacity, class_count * values.shape[1])
    )


def _combined_density(
    density: Array,
    wave: Array,
    class_widths: Array | None,
    degree: int,
    /,
) -> Array:
    """``sum_w g_w(k) [D_q - i k.D_mu - 1/2 k k : D_Theta]`` (conjugate structure)."""
    components = component_count("cartesian-primitive", degree)
    classes = 1 if class_widths is None else class_widths.shape[0]
    blocks = density.reshape((classes, components) + density.shape[1:])
    kx, ky, kz = wave[..., 0], wave[..., 1], wave[..., 2]
    combined = blocks[:, 0]
    if degree >= 1:
        combined = combined - 1j * (
            kx * blocks[:, 1] + ky * blocks[:, 2] + kz * blocks[:, 3]
        )
    if degree == 2:
        combined = combined - 0.5 * (
            kx * kx * blocks[:, 4]
            + 2.0 * kx * ky * blocks[:, 5]
            + 2.0 * kx * kz * blocks[:, 6]
            + ky * ky * blocks[:, 7]
            + 2.0 * ky * kz * blocks[:, 8]
            + kz * kz * blocks[:, 9]
        )
    if class_widths is not None:
        squared = jnp.sum(wave * wave, axis=-1)
        envelope = jnp.exp(
            -0.5
            * class_widths.astype(squared.dtype).reshape((classes,) + (1,) * squared.ndim)
            ** 2
            * squared[None]
        )
        combined = combined * envelope
    return jnp.sum(combined, axis=0)[None]


def _mesh_bilinear(
    mesh: PreparedReciprocalMesh,
    geometry: ReciprocalGeometry,
    positions: Array,
    left: MultipoleSiteSet,
    right: MultipoleSiteSet,
    alpha: float,
    constant: float,
    /,
    *,
    symmetric: bool,
) -> tuple[Array, Array]:
    dtype = positions.dtype
    degree = max(left.maximum_degree, right.maximum_degree)
    spectral = int(np.prod(mesh.spectral_shape))
    mesh.plan.budget.admit(retained_bytes=2 * spectral * _COMPLEX_ITEMSIZE)
    routes = mesh.build(positions, geometry)
    left_classes = 1 if left.widths is None else left.widths.class_count
    right_classes = 1 if right.widths is None else right.widths.class_count
    left_density = mesh.transform(routes, _channel_values(left, degree, left_classes))
    right_density = (
        left_density
        if symmetric
        else mesh.transform(routes, _channel_values(right, degree, right_classes))
    )
    base = geometry.wavevectors.astype(dtype)
    flat = base.reshape((-1, 3))
    variants = mesh.nyquist_variant_modes.astype(dtype)
    reciprocal = geometry.reciprocal_vectors.astype(dtype)
    volume = geometry.volume.astype(dtype)
    left_widths = None if left.widths is None else left.widths.class_widths
    right_widths = None if right.widths is None else right.widths.class_widths
    total = jnp.zeros((), dtype=dtype)
    successful = jnp.asarray(True)
    count = variants.shape[0]
    for variant in range(count):
        if variants.shape[1]:
            wave = flat.at[mesh.nyquist_indices].set(variants[variant] @ reciprocal)
            wave = wave.reshape(base.shape)
        else:
            wave = base
        squared = jnp.sum(wave * wave, axis=-1)
        kernel = _ewald_kernel(squared, volume, alpha, constant)
        combined_left = ReciprocalMeshDensity(
            density=_combined_density(left_density.density, wave, left_widths, degree),
            sources_finite=left_density.sources_finite,
            routes_successful=left_density.routes_successful,
            successful=left_density.successful,
            mesh_identity=left_density.mesh_identity,
        )
        combined_right = (
            combined_left
            if symmetric
            else ReciprocalMeshDensity(
                density=_combined_density(
                    right_density.density, wave, right_widths, degree
                ),
                sources_finite=right_density.sources_finite,
                routes_successful=right_density.routes_successful,
                successful=right_density.successful,
                mesh_identity=right_density.mesh_identity,
            )
        )
        quadratic = mesh.cross_quadratic(
            combined_left, combined_right, kernel, zero_mode="exclude"
        )
        total = total + quadratic.value
        successful = successful & quadratic.successful
    return total / count, successful & geometry.successful


# --- definitions and routes ------------------------------------------------------


class MultipoleElectrostaticsDefinition(StrictModule, NonTrainableState):
    """Physical multipole electrostatics definition, independent of numerics."""

    site_kind: MultipoleSiteKind = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    intrasite: MultipoleIntrasitePolicy = eqx.field(static=True)
    neutrality: MultipoleNeutralityPolicy = eqx.field(static=True)
    charge_tolerance: float = eqx.field(static=True)
    coulomb_constant: float = eqx.field(static=True)
    minimum_distance: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_kind: MultipoleSiteKind,
        maximum_degree: int,
        /,
        *,
        coulomb_constant: float,
        intrasite: MultipoleIntrasitePolicy = "exclude",
        neutrality: MultipoleNeutralityPolicy = "require-neutral",
        charge_tolerance: float = 1.0e-10,
        minimum_distance: float = 1.0e-8,
    ) -> None:
        kind = parse(site_kind, MultipoleSiteKind, "site_kind")
        degree = int(maximum_degree)
        if isinstance(maximum_degree, bool) or not 0 <= degree <= 2:
            raise ValueError("maximum_degree must be 0, 1 or 2 (no public octupole).")
        intrasite_ = parse(intrasite, MultipoleIntrasitePolicy, "intrasite")
        if kind == "point" and intrasite_ != "exclude":
            raise ValueError("Point sites have no finite intrasite energy to include.")
        self.site_kind = kind
        self.maximum_degree = degree
        self.intrasite = intrasite_
        self.neutrality = parse(neutrality, MultipoleNeutralityPolicy, "neutrality")
        self.charge_tolerance = _finite_positive(charge_tolerance, "charge_tolerance")
        self.coulomb_constant = _finite_positive(coulomb_constant, "coulomb_constant")
        self.minimum_distance = _finite_positive(minimum_distance, "minimum_distance")
        self.definition_id = _identity(
            "multipole-electrostatics-definition",
            {
                "site_kind": kind,
                "maximum_degree": degree,
                "intrasite": intrasite_,
                "neutrality": self.neutrality,
                "charge_tolerance": float(self.charge_tolerance).hex(),
                "coulomb_constant": float(self.coulomb_constant).hex(),
                "minimum_distance": float(self.minimum_distance).hex(),
                "moments": "site-centred-primitive-half-theta-grad-grad",
                "gaussian": "unit-normalised-exp-r2-over-2-sigma2",
                "boundary": "tin-foil-k-zero-excluded",
            },
        )


class FiniteMultipoleRoute(StrictModule, NonTrainableState):
    """Complete nonperiodic pair sum over the bounded all-pairs reference relation."""

    row_block: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self, /, *, row_block: int = 32, maximum_workspace_bytes: int = 1 << 28
    ) -> None:
        self.row_block = _positive_int(row_block, "row_block")
        self.maximum_workspace_bytes = _positive_int(
            maximum_workspace_bytes, "maximum_workspace_bytes"
        )
        self.route_id = _identity(
            "finite-multipole-route",
            {
                "relation": "all-pairs-reference",
                "row_block": self.row_block,
                "maximum_workspace_bytes": self.maximum_workspace_bytes,
            },
        )

    @property
    def periodic(self) -> bool:
        return False

    @property
    def method(self) -> str:
        return "finite-all-pairs"


class _PeriodicRouteSettings(StrictModule, NonTrainableState):
    alpha: float = eqx.field(static=True)
    real_cutoff: float = eqx.field(static=True)
    real_space: MultipoleRealSpace = eqx.field(static=True)
    reciprocal_cutoff: float | None = eqx.field(static=True)
    row_block: int = eqx.field(static=True)
    maximum_image_count: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        alpha: float,
        real_cutoff: float,
        /,
        *,
        real_space: MultipoleRealSpace,
        reciprocal_cutoff: float | None,
        row_block: int,
        maximum_image_count: int,
        maximum_workspace_bytes: int,
    ) -> None:
        self.alpha = _finite_positive(alpha, "alpha")
        self.real_cutoff = _finite_positive(real_cutoff, "real_cutoff")
        self.real_space = parse(real_space, MultipoleRealSpace, "real_space")
        self.reciprocal_cutoff = (
            None
            if reciprocal_cutoff is None
            else _finite_positive(reciprocal_cutoff, "reciprocal_cutoff")
        )
        self.row_block = _positive_int(row_block, "row_block")
        self.maximum_image_count = _positive_int(
            maximum_image_count, "maximum_image_count"
        )
        self.maximum_workspace_bytes = _positive_int(
            maximum_workspace_bytes, "maximum_workspace_bytes"
        )

    def identity(self) -> dict:
        return {
            "alpha": float(self.alpha).hex(),
            "real_cutoff": float(self.real_cutoff).hex(),
            "real_space": self.real_space,
            "reciprocal_cutoff": None
            if self.reciprocal_cutoff is None
            else float(self.reciprocal_cutoff).hex(),
            "row_block": self.row_block,
            "maximum_image_count": self.maximum_image_count,
            "maximum_workspace_bytes": self.maximum_workspace_bytes,
        }


class EwaldMultipoleRoute(StrictModule, NonTrainableState):
    """Ewald split with explicit canonical half-space modes (direct k)."""

    settings: _PeriodicRouteSettings
    modes: ReciprocalModeSetPlan
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        alpha: float,
        real_cutoff: float,
        modes: ReciprocalModeSetPlan,
        /,
        *,
        real_space: MultipoleRealSpace = "explicit-image-reference",
        reciprocal_cutoff: float | None = None,
        row_block: int = 32,
        maximum_image_count: int = 4096,
        maximum_workspace_bytes: int = 1 << 28,
    ) -> None:
        if not isinstance(modes, ReciprocalModeSetPlan):
            raise TypeError("modes must be a ReciprocalModeSetPlan.")
        if modes.include_zero_mode:
            raise ValueError(
                "Tin-foil multipole Ewald excludes k = 0; use a set without it."
            )
        self.settings = _PeriodicRouteSettings(
            alpha,
            real_cutoff,
            real_space=real_space,
            reciprocal_cutoff=reciprocal_cutoff,
            row_block=row_block,
            maximum_image_count=maximum_image_count,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )
        self.modes = modes
        self.route_id = _identity(
            "ewald-multipole-route",
            {**self.settings.identity(), "modes": modes.plan_id},
        )

    @property
    def periodic(self) -> bool:
        return True

    @property
    def method(self) -> str:
        return "ewald-explicit-modes"


class ParticleMeshMultipoleRoute(StrictModule, NonTrainableState):
    """Smooth particle-mesh Ewald on the shared term-owned reciprocal mesh."""

    settings: _PeriodicRouteSettings
    mesh: ReciprocalMeshPlan
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        alpha: float,
        real_cutoff: float,
        mesh: ReciprocalMeshPlan,
        /,
        *,
        real_space: MultipoleRealSpace = "explicit-image-reference",
        reciprocal_cutoff: float | None = None,
        row_block: int = 32,
        maximum_image_count: int = 4096,
        maximum_workspace_bytes: int = 1 << 28,
    ) -> None:
        if not isinstance(mesh, ReciprocalMeshPlan):
            raise TypeError("mesh must be a ReciprocalMeshPlan.")
        self.settings = _PeriodicRouteSettings(
            alpha,
            real_cutoff,
            real_space=real_space,
            reciprocal_cutoff=reciprocal_cutoff,
            row_block=row_block,
            maximum_image_count=maximum_image_count,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )
        self.mesh = mesh
        self.route_id = _identity(
            "particle-mesh-multipole-route",
            {**self.settings.identity(), "mesh": mesh.plan_id},
        )

    @property
    def periodic(self) -> bool:
        return True

    @property
    def method(self) -> str:
        return (
            f"particle-mesh-degree-{self.mesh.spline_degree}-{self.mesh.modulus}-"
            f"nyquist-{self.mesh.nyquist}"
        )


MultipoleRoute: TypeAlias = (
    FiniteMultipoleRoute | EwaldMultipoleRoute | ParticleMeshMultipoleRoute
)


class MultipoleBilinearParts(StrictModule):
    """Components and numerical status of one bilinear form ``B(L, R)``."""

    value: Array
    real: Array
    reciprocal: Array
    intrasite: Array
    background: Array
    left_charge: Array
    right_charge: Array
    neutral: Array
    cell_successful: Array
    real_space_successful: Array
    real_space_coverage: Array
    mode_coverage_successful: Array
    reciprocal_successful: Array
    support_radius: Array
    volume: Array
    successful: Array


class MultipoleElectrostaticsEvidence(StrictModule):
    """Compact evidence of one multipole electrostatic energy.

    ``prepared_identity`` is the producing preparation's exact digest as dynamic
    words (``prepared_id`` decodes it on the host).
    """

    method: str = eqx.field(static=True)
    site_kind: str = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    intrasite: str = eqx.field(static=True)
    neutrality: str = eqx.field(static=True)
    real_space: str = eqx.field(static=True)
    declared_reciprocal_cutoff: float | None = eqx.field(static=True)
    moment_source: str = eqx.field(static=True)
    prepared_identity: IdentityWords
    moments_successful: Array
    total_charge: Array
    neutral: Array
    cell_successful: Array
    real_space_successful: Array
    real_space_coverage: Array
    mode_coverage_successful: Array
    support_radius: Array
    reciprocal_successful: Array
    volume: Array
    real_energy: Array
    reciprocal_energy: Array
    intrasite_energy: Array
    background_energy: Array
    successful: Array

    @property
    def prepared_id(self) -> str:
        """Host decoding of the producing preparation digest."""
        return self.prepared_identity.fingerprint()


class MultipoleElectrostaticsResult(StrictModule):
    """Fail-closed energy and its evidence."""

    energy: Array
    successful: Array
    evidence: MultipoleElectrostaticsEvidence


class MultipoleFieldFeatures(StrictModule):
    """Gaussian-smeared potential, field and field gradient per receiver width.

    ``potential[i, r] = k <phi>``, ``field[i, r] = -k <grad phi>`` and
    ``field_gradient[i, r] = k <grad grad phi>`` (``k`` the Coulomb constant) at
    site ``i`` smeared by a
    unit-normalised receiver Gaussian of width ``receiver_widths[r]`` (points
    for point sites). These Cartesian moments are not the source's receiver-GTO
    spherical projections; no source-feature parity is implied.
    """

    potential: Array
    field: Array
    field_gradient: Array
    own_site: str = eqx.field(static=True)
    successful: Array


class MultipoleElectrostaticsPlan(StrictModule, NonTrainableState):
    """Composition of one physical definition and one numerical route."""

    definition: MultipoleElectrostaticsDefinition
    route: MultipoleRoute
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, definition: MultipoleElectrostaticsDefinition, route: MultipoleRoute, /
    ) -> None:
        if not isinstance(definition, MultipoleElectrostaticsDefinition):
            raise TypeError("definition must be MultipoleElectrostaticsDefinition.")
        if not isinstance(
            route, (FiniteMultipoleRoute, EwaldMultipoleRoute, ParticleMeshMultipoleRoute)
        ):
            raise TypeError("route must be a finite, Ewald or particle-mesh route.")
        self.definition = definition
        self.route = route
        self.plan_id = _identity(
            "multipole-electrostatics-plan",
            {"definition": definition.definition_id, "route": route.route_id},
        )

    def prepare(
        self,
        site_capacity: int,
        /,
        *,
        cell: PeriodicCell | None = None,
        particles: ParticleDiscretization | None = None,
    ) -> "PreparedMultipoleElectrostatics":
        return PreparedMultipoleElectrostatics(
            self, site_capacity, cell=cell, particles=particles
        )


class PreparedMultipoleElectrostatics(StrictModule, NonTrainableState):
    """Prepared definition, route resources and declared reference relation.

    PyTree layout: ``site_capacity`` and the content-free ``layout_id`` (plan,
    capacity, reference cell, mode-set/mesh layouts and reference relation)
    are static. The exact canonical preparation digest, which includes the
    material-bound reciprocal mesh, is the dynamic ``prepared_identity``
    (``prepared_id`` decodes it on the host).
    """

    plan: MultipoleElectrostaticsPlan
    cell: PeriodicCell | None
    modes: PreparedReciprocalModeSet | None
    mesh: PreparedReciprocalMesh | None
    reference_pairs: ExplicitImagePairs | None
    site_capacity: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: MultipoleElectrostaticsPlan,
        site_capacity: int,
        /,
        *,
        cell: PeriodicCell | None = None,
        particles: ParticleDiscretization | None = None,
    ) -> None:
        capacity = _positive_int(site_capacity, "site_capacity")
        route = plan.route
        modes = mesh = reference = None
        if isinstance(route, FiniteMultipoleRoute):
            if cell is not None or particles is not None:
                raise ValueError("A finite multipole route takes no cell or particles.")
            reference = ExplicitImagePairs(
                capacity,
                row_block=route.row_block,
                maximum_workspace_bytes=route.maximum_workspace_bytes,
            )
        else:
            if not isinstance(cell, PeriodicCell) or not cell.fully_periodic:
                raise ValueError(
                    "Periodic multipole routes require a fully periodic rank-3 cell "
                    "(only the tin-foil boundary is defined)."
                )
            settings = route.settings
            if isinstance(route, EwaldMultipoleRoute):
                if particles is not None:
                    raise ValueError("Explicit-mode Ewald takes no particle support.")
                modes = route.modes.prepare(cell, capacity)
            else:
                if not isinstance(particles, ParticleDiscretization):
                    raise ValueError("Particle-mesh multipoles require particle support.")
                if int(particles.capacity) != capacity:
                    raise ValueError("Particle support and site capacity differ.")
                mesh = route.mesh.prepare(cell, particles)
            if settings.real_space == "explicit-image-reference":
                reference = ExplicitImagePairs(
                    capacity,
                    cell=cell,
                    cutoff=settings.real_cutoff,
                    row_block=settings.row_block,
                    maximum_image_count=settings.maximum_image_count,
                    maximum_workspace_bytes=settings.maximum_workspace_bytes,
                )
        self.plan = plan
        self.cell = cell
        self.modes = modes
        self.mesh = mesh
        self.reference_pairs = reference
        self.site_capacity = capacity
        self.layout_id = _identity(
            "prepared-multipole-electrostatics-layout",
            {
                "plan": plan.plan_id,
                "site_capacity": capacity,
                "cell": None if cell is None else cell.cell_id,
                "modes": None if modes is None else modes.prepared_id,
                "mesh": None if mesh is None else mesh.plan.plan_id,
                "reference": None if reference is None else reference.relation_id,
            },
        )
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-multipole-electrostatics",
                "plan": plan.plan_id,
                "site_capacity": capacity,
                "cell": None if cell is None else cell.cell_id,
                "modes": None if modes is None else modes.prepared_id,
                "mesh": None if mesh is None else mesh.prepared_id,
                "reference": None if reference is None else reference.relation_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical preparation digest."""
        return self.prepared_identity.fingerprint()

    @property
    def definition(self) -> MultipoleElectrostaticsDefinition:
        return self.plan.definition

    @property
    def method(self) -> str:
        return f"{self.definition.site_kind}-multipole-{self.plan.route.method}"

    def _check_sites(self, site: MultipoleSiteSet, name: str, /, *, source: bool) -> None:
        if not isinstance(site, MultipoleSiteSet):
            raise TypeError(f"{name} must be a MultipoleSiteSet.")
        if site.site_capacity != self.site_capacity:
            raise ValueError(f"{name} does not match the prepared site capacity.")
        if source and site.maximum_degree > self.definition.maximum_degree:
            raise ValueError(f"{name} exceeds the declared maximum source degree.")
        gaussian = self.definition.site_kind == "gaussian"
        if gaussian != (site.widths is not None):
            raise ValueError(
                f"{name} widths do not match the declared {self.definition.site_kind} sites."
            )

    def resolve_pairs(
        self, pairs: AbstractMultipolePairs | None, /
    ) -> AbstractMultipolePairs:
        """The declared reference relation or the supplied neighborhood relation."""
        if self.reference_pairs is not None:
            if pairs is not None:
                raise ValueError(
                    "This route uses its declared reference relation; omit pairs."
                )
            return self.reference_pairs
        if not isinstance(pairs, NeighborhoodPairs):
            raise ValueError(
                "The neighborhood real-space route requires NeighborhoodPairs."
            )
        if pairs.site_capacity != self.site_capacity:
            raise ValueError("Neighborhood pairs belong to another site capacity.")
        return pairs

    def requirement(self, owner: str, /) -> AtomisticNeighborhoodRequirement | None:
        """The image-pair-sum requirement of a neighborhood real-space route."""
        route = self.plan.route
        if isinstance(route, FiniteMultipoleRoute) or route.settings.real_space != (
            "neighborhood"
        ):
            return None
        return AtomisticNeighborhoodRequirement(
            owner,
            AtomisticNeighborhoodRole.IMAGE_PAIR_SUM,
            route.settings.real_cutoff,
            exclusions=AtomisticPairExclusions.NONE,
        )

    # Keep real, reciprocal, self, intrasite and boundary contributions in their
    # declared bilinear order. The explicit route branches preserve source degree,
    # symmetry and derivative evidence; they are not an alternate solver framework.
    def bilinear(
        self,
        positions: ArrayLike,
        left: MultipoleSiteSet,
        right: MultipoleSiteSet,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
        pairs: AbstractMultipolePairs | None = None,
        intrasite: MultipoleIntrasitePolicy | None = None,
        symmetric: bool = False,
    ) -> MultipoleBilinearParts:
        """``B(L, R)`` with its components; ``symmetric`` asserts ``L is R``.

        The declared maximum degree bounds the source set ``R``; the receiver
        ``L`` may carry any public degree (field probes are degree-2 receivers).
        """
        position = jnp.asarray(positions)
        if position.shape != (self.site_capacity, 3):
            raise ValueError("positions must have shape (N, 3).")
        if not jnp.issubdtype(position.dtype, jnp.floating):
            raise TypeError("positions must be floating point.")
        self._check_sites(left, "left", source=False)
        self._check_sites(right, "right", source=True)
        if symmetric and left is not right:
            raise ValueError("symmetric=True requires the same site set on both sides.")
        policy = parse(
            self.definition.intrasite if intrasite is None else intrasite,
            MultipoleIntrasitePolicy,
            "intrasite",
        )
        definition = self.definition
        constant = definition.coulomb_constant
        dtype = position.dtype
        relation = self.resolve_pairs(pairs)
        route = self.plan.route
        left_moments, right_moments = left.masked_moments(), right.masked_moments()
        left_charge = jnp.sum(left_moments.charges)
        right_charge = jnp.sum(right_moments.charges)
        gaussian = definition.site_kind == "gaussian"
        spread = left.width_squared() + right.width_squared()
        if isinstance(route, FiniteMultipoleRoute):
            if cell_vectors is not None:
                raise ValueError("A finite multipole route takes no cell vectors.")
            kernel: AbstractMultipoleRadialKernel = (
                GaussianKernel() if gaussian else CoulombKernel()
            )
            real = relation.pair_sum(
                position,
                None,
                left,
                right,
                kernel,
                cutoff=None,
                minimum_distance=definition.minimum_distance,
                scale=None,
                symmetric=symmetric,
            )
            intrasite_value = jnp.zeros((), dtype=dtype)
            if policy == "include":
                if not gaussian:
                    raise ValueError("Point sites have no finite intrasite term.")
                beta = jax.lax.rsqrt(2.0 * spread)
                intrasite_value = jnp.sum(
                    co_centered_interaction(
                        erf_kernel_origin_coefficients(beta, 2),
                        left_moments,
                        right_moments,
                    )
                )
            reciprocal = jnp.zeros((), dtype=dtype)
            background = jnp.zeros((), dtype=dtype)
            cell_ok = jnp.asarray(True)
            reciprocal_ok = jnp.asarray(True)
            coverage = jnp.asarray(True)
            support = jnp.asarray(jnp.inf, dtype=dtype)
            volume = jnp.asarray(jnp.inf, dtype=dtype)
        else:
            if cell_vectors is None:
                raise ValueError("Periodic multipole routes require cell vectors.")
            vectors = jnp.asarray(cell_vectors, dtype=dtype)
            settings = route.settings
            alpha = settings.alpha
            binding = self.modes if self.modes is not None else self.mesh
            if binding is None:
                raise ValueError(
                    "A periodic multipole route needs its prepared modes or mesh."
                )
            geometry = binding.geometry(vectors)
            kernel = GaussianEwaldRealKernel(alpha) if gaussian else ErfcKernel(alpha)
            real = relation.pair_sum(
                position,
                vectors,
                left,
                right,
                kernel,
                cutoff=settings.real_cutoff,
                minimum_distance=definition.minimum_distance,
                scale=None,
                symmetric=symmetric,
            )
            if isinstance(binding, PreparedReciprocalModeSet):
                reciprocal, reciprocal_ok = _mode_set_bilinear(
                    binding,
                    geometry,
                    position,
                    left,
                    right,
                    alpha,
                    constant,
                    symmetric=symmetric,
                )
            else:
                reciprocal, reciprocal_ok = _mesh_bilinear(
                    binding,
                    geometry,
                    position,
                    left,
                    right,
                    alpha,
                    constant,
                    symmetric=symmetric,
                )
            smooth = jax.lax.rsqrt(2.0 * spread + 1.0 / alpha**2)
            intrasite_value = -jnp.sum(
                co_centered_interaction(
                    erf_kernel_origin_coefficients(smooth, 2), left_moments, right_moments
                )
            )
            if policy == "include":
                if not gaussian:
                    raise ValueError("Point sites have no finite intrasite term.")
                physical = jax.lax.rsqrt(2.0 * spread)
                intrasite_value = intrasite_value + jnp.sum(
                    co_centered_interaction(
                        erf_kernel_origin_coefficients(physical, 2),
                        left_moments,
                        right_moments,
                    )
                )
            volume = geometry.volume.astype(dtype)
            background = (
                -jnp.pi * left_charge * right_charge / (alpha**2 * volume)
                if definition.neutrality == "uniform-background"
                else jnp.zeros((), dtype=dtype)
            )
            cell_ok = geometry.successful
            support = geometry.support_radius.astype(dtype)
            coverage = (
                jnp.asarray(True)
                if settings.reciprocal_cutoff is None
                else settings.reciprocal_cutoff <= support
            )
        real_value = constant * real.value
        intrasite_value = constant * intrasite_value
        background = constant * background
        value = real_value + reciprocal + intrasite_value + background
        neutral = (jnp.abs(left_charge) <= definition.charge_tolerance) & (
            jnp.abs(right_charge) <= definition.charge_tolerance
        )
        neutral_ok = (
            neutral
            if route.periodic and definition.neutrality == "require-neutral"
            else jnp.asarray(True)
        )
        finite_inputs = (
            jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(left_moments.second_moments))
            & jnp.all(jnp.isfinite(right_moments.second_moments))
            & jnp.all(jnp.isfinite(left_moments.dipoles))
            & jnp.all(jnp.isfinite(right_moments.dipoles))
            & jnp.isfinite(left_charge)
            & jnp.isfinite(right_charge)
        )
        widths_ok = jnp.asarray(True)
        for site in (left, right):
            if site.widths is not None:
                widths_ok = widths_ok & site.widths.valid()
        successful = (
            finite_inputs
            & widths_ok
            & neutral_ok
            & cell_ok
            & real.geometry_successful
            & real.coverage_successful
            & coverage
            & reciprocal_ok
            & jnp.isfinite(value)
        )
        return MultipoleBilinearParts(
            value=value,
            real=real_value,
            reciprocal=reciprocal,
            intrasite=intrasite_value,
            background=background,
            left_charge=left_charge,
            right_charge=right_charge,
            neutral=neutral,
            cell_successful=cell_ok,
            real_space_successful=real.geometry_successful,
            real_space_coverage=real.coverage_successful,
            mode_coverage_successful=coverage,
            reciprocal_successful=reciprocal_ok,
            support_radius=support,
            volume=volume,
            successful=successful,
        )

    def energy(
        self,
        positions: ArrayLike,
        sites: MultipoleSiteSet,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
        pairs: AbstractMultipolePairs | None = None,
        moments_successful: ArrayLike = True,
        moment_source: str = "explicit-site-data",
    ) -> MultipoleElectrostaticsResult:
        """Fail-closed ``E = B(M, M)/2`` with compact evidence."""
        parts = self.bilinear(
            positions,
            sites,
            sites,
            cell_vectors=cell_vectors,
            pairs=pairs,
            symmetric=True,
        )
        status = jnp.asarray(moments_successful, dtype=jnp.bool_)
        successful = parts.successful & status
        energy = 0.5 * parts.value
        route = self.plan.route
        evidence = MultipoleElectrostaticsEvidence(
            method=self.method,
            site_kind=self.definition.site_kind,
            maximum_degree=self.definition.maximum_degree,
            intrasite=self.definition.intrasite,
            neutrality=self.definition.neutrality,
            real_space=(
                "all-pairs-reference"
                if isinstance(route, FiniteMultipoleRoute)
                else route.settings.real_space
            ),
            declared_reciprocal_cutoff=(
                None
                if isinstance(route, FiniteMultipoleRoute)
                else route.settings.reciprocal_cutoff
            ),
            moment_source=moment_source,
            prepared_identity=self.prepared_identity,
            moments_successful=status,
            total_charge=parts.left_charge,
            neutral=parts.neutral,
            cell_successful=parts.cell_successful,
            real_space_successful=parts.real_space_successful,
            real_space_coverage=parts.real_space_coverage,
            mode_coverage_successful=parts.mode_coverage_successful,
            support_radius=parts.support_radius,
            reciprocal_successful=parts.reciprocal_successful,
            volume=parts.volume,
            real_energy=0.5 * parts.real,
            reciprocal_energy=0.5 * parts.reciprocal,
            intrasite_energy=0.5 * parts.intrasite,
            background_energy=0.5 * parts.background,
            successful=successful,
        )
        return MultipoleElectrostaticsResult(
            jnp.where(successful, energy, jnp.nan), successful, evidence
        )

    def features(
        self,
        positions: ArrayLike,
        sites: MultipoleSiteSet,
        /,
        *,
        receiver_widths: ArrayLike | None = None,
        own_site: MultipoleIntrasitePolicy = "exclude",
        cell_vectors: ArrayLike | None = None,
        pairs: AbstractMultipolePairs | None = None,
    ) -> MultipoleFieldFeatures:
        """Receiver-smeared potential, field and field gradient of ``sites``.

        Computed as exact derivatives of ``B(P, M)`` with respect to unit probe
        coefficients ``P`` at ``P = 0`` (the form is linear in ``P``).
        ``own_site="exclude"`` removes the co-centred source density of each
        site; ``"include"`` (Gaussian only) keeps it.
        """
        own = parse(own_site, MultipoleIntrasitePolicy, "own_site")
        position = jnp.asarray(positions)
        dtype = position.dtype
        count = self.site_capacity
        gaussian = self.definition.site_kind == "gaussian"
        if gaussian:
            if receiver_widths is None:
                raise ValueError("Gaussian features require receiver_widths.")
            widths = jnp.asarray(receiver_widths, dtype=dtype)
            if widths.ndim != 1 or widths.shape[0] == 0:
                raise ValueError("receiver_widths must have shape (R,) with R > 0.")
            if not isinstance(widths, jax_core.Tracer):
                host = np.asarray(widths, dtype=np.float64)
                if np.any(~np.isfinite(host)) or np.any(host <= 0.0):
                    raise ValueError("Receiver widths must be finite and positive.")
        else:
            if receiver_widths is not None:
                raise ValueError("Point-site features use point receivers.")
            if own != "exclude":
                raise ValueError("Point sites have no finite own-site field.")
            widths = jnp.zeros((1,), dtype=dtype)
        degree = 2
        zero = jnp.zeros((count,), dtype=dtype)

        def receiver(width: Array) -> tuple[Array, Array, Array, Array]:
            def form(
                charges: Array, dipoles: Array, unique: Array
            ) -> tuple[Array, Array]:
                probe = CartesianMultipoles(
                    charges,
                    dipoles if degree >= 1 else None,
                    unique if degree == 2 else None,
                    maximum_degree=degree,
                )
                probe_widths = (
                    GaussianMultipoleWidths(
                        width[None], np.zeros((count,), dtype=np.int32)
                    )
                    if gaussian
                    else None
                )
                probe_sites = MultipoleSiteSet(
                    probe, active=sites.active, widths=probe_widths
                )
                parts = self.bilinear(
                    position,
                    probe_sites,
                    sites,
                    cell_vectors=cell_vectors,
                    pairs=pairs,
                    intrasite=own,
                )
                return parts.value, parts.successful

            gradients, successful = jax.grad(form, argnums=(0, 1, 2), has_aux=True)(
                zero,
                jnp.zeros((count, 3), dtype=dtype),
                jnp.zeros((count, 6), dtype=dtype),
            )
            potential, dipole_gradient, unique_gradient = gradients
            diagonal = jnp.asarray((1, 0, 0, 1, 0, 1), dtype=dtype)
            gradient_tensor = second_moment_matrix(unique_gradient * (1.0 + diagonal))
            return potential, -dipole_gradient, gradient_tensor, successful

        potential, field, gradient, successful = jax.vmap(
            receiver, out_axes=(1, 1, 1, 0)
        )(widths)
        return MultipoleFieldFeatures(
            potential=potential,
            field=field,
            field_gradient=gradient,
            own_site=own,
            successful=jnp.all(successful),
        )


# --- typed moment sources and the energy term -----------------------------------


class MultipoleMomentFields(StrictModule, NonTrainableState):
    """Typed predicted per-site moments in a declared basis and unit.

    Every reference must be atom-level, carry an explicit unit dimensionally
    equal to charge x length^degree, and its port ``normalization_id`` must
    name the declared basis exactly (the basis witness; event shapes never
    select a basis). A traceless real-spherical degree-two field requires an
    explicit ``trace`` declaration.
    """

    charges: AtomisticFieldReference | None
    dipoles: AtomisticFieldReference | None
    second_moments: AtomisticFieldReference | None
    basis: MultipoleBasis = eqx.field(static=True)
    trace: Literal["zero"] | None = eqx.field(static=True)
    fields_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: MultipoleBasis,
        /,
        *,
        charges: AtomisticFieldReference | None = None,
        dipoles: AtomisticFieldReference | None = None,
        second_moments: AtomisticFieldReference | None = None,
        trace: Literal["zero"] | None = None,
    ) -> None:
        basis_ = parse(basis, MultipoleBasis, "basis")
        spherical = basis_ != "cartesian-primitive"
        expected = (
            (charges, AtomisticObservable.CHARGES, ()),
            (dipoles, AtomisticObservable.DIPOLE, (3,)),
            (second_moments, AtomisticObservable.QUADRUPOLE, (5,) if spherical else (6,)),
        )
        for reference, observable, shape in expected:
            if reference is None:
                continue
            if not isinstance(reference, AtomisticFieldReference):
                raise TypeError("Moment fields must be AtomisticFieldReference values.")
            if (
                reference.observable is not observable
                or reference.level is not AtomisticConditionLevel.ATOM
                or tuple(reference.port.event_shape) != shape
                or reference.unit is None
            ):
                raise ValueError(
                    f"{observable.value} must be an atom-level field of shape {shape} "
                    "with an explicit unit."
                )
            if observable is not AtomisticObservable.CHARGES and (
                reference.port.normalization_id != basis_
            ):
                raise ValueError(
                    f"{observable.value} port normalization_id must name the declared "
                    f"basis {basis_!r} exactly."
                )
        if second_moments is not None and spherical and trace != "zero":
            raise ValueError(
                "A traceless real-spherical second-moment field needs trace='zero' "
                "declared explicitly."
            )
        if trace is not None and not (second_moments is not None and spherical):
            raise ValueError("trace only applies to real-spherical second moments.")
        self.charges = charges
        self.dipoles = dipoles
        self.second_moments = second_moments
        self.basis = basis_
        self.trace = trace
        self.fields_id = _identity(
            "multipole-moment-fields",
            {
                "basis": basis_,
                "trace": trace,
                "references": [
                    None if item is None else item.reference_id
                    for item in (charges, dipoles, second_moments)
                ],
            },
        )

    @property
    def references(self) -> tuple[AtomisticFieldReference, ...]:
        return tuple(
            item
            for item in (self.charges, self.dipoles, self.second_moments)
            if item is not None
        )


def _unit_factor(
    reference: AtomisticFieldReference | None,
    system: PreparedAtomisticSystem,
    degree: int,
    /,
) -> float:
    """Exact field-to-system unit factor of one moment degree (one when absent)."""
    if reference is None:
        return 1.0
    from ..units import derived_unit

    units = system.plan.units
    charge = units.charge_unit
    length = units.scale.length_unit
    expected = (
        charge
        if degree == 0
        else derived_unit(
            f"{charge.symbol}*{length.symbol}^{degree}", ((charge, 1), (length, degree))
        )
    )
    unit = reference.unit
    if unit is None or unit.dimension != expected.dimension:
        raise ValueError("A moment field unit has the wrong physical dimension.")
    if unit.reference_system_id != expected.reference_system_id:
        raise ValueError("A moment field unit belongs to another reference system.")
    return float(unit.scale_to_reference / expected.scale_to_reference)


class MultipoleElectrostaticPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Multipole electrostatic energy term of a potential program.

    Moments are explicit fixed site data (``moments``) or typed predictions of a
    preceding native model (``moment_fields``) evaluated inside the same scalar
    energy, so coordinate, cell and parameter derivatives include their
    response. Global reciprocal/intrasite terms are not split into atoms:
    results report ``atom_energy_complete=False``.

    The exact canonical term digest is the dynamic ``term_identity``
    (``term_id`` decodes it on the host); explicit moments and width values are
    dynamic leaves, so moment/width revisions share one executable.
    """

    plan: MultipoleElectrostaticsPlan
    moments: CartesianMultipoles | None
    widths: GaussianMultipoleWidths | None
    moment_fields: MultipoleMomentFields | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: MultipoleElectrostaticsPlan,
        /,
        *,
        moments: CartesianMultipoles | None = None,
        moment_fields: MultipoleMomentFields | None = None,
        widths: GaussianMultipoleWidths | None = None,
        name: str = "multipole-electrostatics",
        force_group: int = 0,
    ) -> None:
        if not isinstance(plan, MultipoleElectrostaticsPlan):
            raise TypeError("plan must be a MultipoleElectrostaticsPlan.")
        if (moments is None) == (moment_fields is None):
            raise ValueError("Supply exactly one of explicit moments or moment_fields.")
        if (
            moments is not None
            and moments.maximum_degree != plan.definition.maximum_degree
        ):
            raise ValueError("Explicit moments must have the declared maximum degree.")
        if (plan.definition.site_kind == "gaussian") != (widths is not None):
            raise ValueError("Gaussian sites require widths; point sites refuse them.")
        identifier = str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError("Term name must be non-empty and force_group non-negative.")
        route = plan.route
        periodic = route.periodic
        neighborhood = (
            not isinstance(route, FiniteMultipoleRoute)
            and route.settings.real_space == "neighborhood"
        )
        dependencies = () if moment_fields is None else moment_fields.references
        self.plan = plan
        self.moments = moments
        self.widths = widths
        self.moment_fields = moment_fields
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=not periodic,
            orthorhombic_periodic=periodic,
            triclinic_periodic=periodic,
            cell_derivative=periodic,
            local_energy=False,
        )
        self.requirements = (
            AtomisticPotentialRequirements(
                reciprocal_grid=periodic,
                neighborhoods=(
                    AtomisticNeighborhoodRequirement(
                        identifier,
                        AtomisticNeighborhoodRole.IMAGE_PAIR_SUM,
                        route.settings.real_cutoff,
                        exclusions=AtomisticPairExclusions.NONE,
                    ),
                ),
                field_dependencies=dependencies,
            )
            if neighborhood
            else AtomisticPotentialRequirements(
                reciprocal_grid=periodic, field_dependencies=dependencies
            )
        )
        self.term_identity = IdentityWords.of(
            {
                "kind": "multipole-electrostatic-potential",
                "plan": plan.plan_id,
                "moments": "explicit"
                if moment_fields is None
                else moment_fields.fields_id,
                "widths": None if widths is None else widths.widths_id,
                "name": identifier,
                "force_group": group,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedMultipoleElectrostaticPotential":
        return PreparedMultipoleElectrostaticPotential(self, system)


class PreparedMultipoleElectrostaticPotential(AbstractPreparedAtomisticEnergyTerm):
    """Prepared multipole electrostatic term bound to one atomistic system.

    PyTree layout: static ``name``, ``force_group`` and the declared moment-field
    ``unit_factors`` (fixed by the declared field units and the system unit
    system, never by material content). ``term_identity`` and
    ``prepared_identity`` are exact canonical digests as dynamic words; a
    context of another prepared system yields a NaN, unsuccessful evaluation
    instead of a traced identity decode. Particle-mesh support is the host
    ``system.particle_discretization()``.
    """

    plan: MultipoleElectrostaticPotential
    system: PreparedAtomisticSystem
    electrostatics: PreparedMultipoleElectrostatics
    unit_factors: tuple[float, float, float] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: MultipoleElectrostaticPotential, system: PreparedAtomisticSystem, /
    ) -> None:
        capacity = system.capacity
        route = plan.plan.route
        if isinstance(route, FiniteMultipoleRoute):
            if system.cell is not None:
                raise ValueError(
                    "A finite multipole route requires a nonperiodic system."
                )
            electrostatics = plan.plan.prepare(capacity)
        elif isinstance(route, EwaldMultipoleRoute):
            electrostatics = plan.plan.prepare(capacity, cell=system.cell)
        else:
            electrostatics = plan.plan.prepare(
                capacity, cell=system.cell, particles=system.particle_discretization()
            )
        if plan.moments is not None and plan.moments.site_capacity != capacity:
            raise ValueError("Explicit moments do not match the system capacity.")
        if plan.widths is not None and plan.widths.site_capacity != capacity:
            raise ValueError("Gaussian widths do not match the system capacity.")
        factors = (1.0, 1.0, 1.0)
        fields = plan.moment_fields
        if fields is not None:
            factors = (
                _unit_factor(fields.charges, system, 0),
                _unit_factor(fields.dipoles, system, 1),
                _unit_factor(fields.second_moments, system, 2),
            )
        self.plan = plan
        self.system = system
        self.electrostatics = electrostatics
        self.unit_factors = factors
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-multipole-electrostatic-potential",
                "term": plan.term_id,
                "system": system.prepared_id,
                "electrostatics": electrostatics.prepared_id,
            }
        )

    def _moments(
        self, context: AtomisticPotentialContext, /
    ) -> tuple[CartesianMultipoles, Array, str]:
        dtype = context.positions.dtype
        degree = self.electrostatics.definition.maximum_degree
        explicit = self.plan.moments
        if explicit is not None:
            moments = explicit
            return (
                CartesianMultipoles(
                    moments.charges.astype(dtype),
                    moments.dipoles.astype(dtype) if degree >= 1 else None,
                    unique_second_moments(moments.second_moments.astype(dtype))
                    if degree == 2
                    else None,
                    maximum_degree=degree,
                ),
                jnp.asarray(True),
                "explicit-site-data",
            )
        fields = self.plan.moment_fields
        if fields is None:
            raise ValueError(
                "The multipole term declares neither moments nor moment fields."
            )
        capacity = self.system.capacity
        successful = jnp.asarray(True)
        parts: list[Array] = []
        for index, (reference, width) in enumerate(
            (
                (fields.charges, 1),
                (fields.dipoles, 3),
                (
                    fields.second_moments,
                    5 if fields.basis != "cartesian-primitive" else 6,
                ),
            )
        ):
            if index > degree:
                if reference is not None:
                    raise ValueError("A moment field exceeds the declared degree.")
                continue
            if reference is None:
                parts.append(jnp.zeros((capacity, width), dtype=dtype))
                continue
            value = context.field(reference)
            if value.available.shape != (1,) or value.values.shape[0] != capacity:
                raise ValueError("Predicted moments must be one atom-level case.")
            values = value.values.astype(dtype).reshape((capacity, width))
            usable = jnp.isfinite(values) & value.available[0] & value.successful[0]
            parts.append(jnp.where(usable, self.unit_factors[index] * values, 0.0))
            successful = successful & value.available[0] & value.successful[0]
        moments = unpack_multipoles(
            jnp.concatenate(parts, axis=-1),
            fields.basis,
            degree,
            trace=fields.trace,
        )
        return moments, successful, fields.fields_id

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        bound = context.system.prepared_identity.equals(self.system.prepared_identity)
        moments, moments_ok, source = self._moments(context)
        sites = MultipoleSiteSet(
            moments, active=self.system.active_mask, widths=self.plan.widths
        )
        requirement = self.electrostatics.requirement(self.name)
        pairs = (
            None
            if requirement is None
            else NeighborhoodPairs(
                context.neighborhood(requirement), self.system.capacity
            )
        )
        result = self.electrostatics.energy(
            context.positions,
            sites,
            cell_vectors=None
            if not self.plan.plan.route.periodic
            else context.cell_vectors,
            pairs=pairs,
            moments_successful=moments_ok,
            moment_source=source,
        )
        successful = result.successful & bound
        return AtomisticTermEvaluation(
            jnp.where(bound, result.energy, jnp.nan),
            jnp.zeros((self.system.capacity,), dtype=result.energy.dtype),
            successful,
            atom_energy_complete=False,
            evidence=(result.evidence,),
        )


__all__ = [
    "AbstractMultipolePairs",
    "AbstractMultipoleRadialKernel",
    "CoulombKernel",
    "ErfcKernel",
    "EwaldMultipoleRoute",
    "ExplicitImagePairs",
    "FiniteMultipoleRoute",
    "GaussianEwaldRealKernel",
    "GaussianKernel",
    "MultipoleBilinearParts",
    "MultipoleElectrostaticPotential",
    "MultipoleElectrostaticsDefinition",
    "MultipoleElectrostaticsEvidence",
    "MultipoleElectrostaticsPlan",
    "MultipoleElectrostaticsResult",
    "MultipoleFieldFeatures",
    "MultipoleIntrasitePolicy",
    "MultipoleMomentFields",
    "MultipoleNeutralityPolicy",
    "MultipolePairScales",
    "MultipoleRealSpace",
    "MultipoleSiteKind",
    "MultipoleSiteSet",
    "NeighborhoodPairs",
    "ParticleMeshMultipoleRoute",
    "PreparedMultipoleElectrostaticPotential",
    "PreparedMultipoleElectrostatics",
    "WeightedPairCorrections",
]
