#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact public degree-0/1/2 multipole definitions and basis maps.

Canonical physical definition
-----------------------------
Every interaction owner consumes :class:`CartesianMultipoles`: site-centred
*primitive* Cartesian coefficients ``(q, mu, Theta)`` in the system Cartesian
frame. A point site carries the distribution

``rho(r) = (q - mu . grad + 1/2 Theta : grad grad) delta(r - r_i)``

so ``q`` is the charge, ``mu = int rho r`` the dipole and ``Theta`` the raw,
*traceful*, symmetric second moment ``int rho r r`` (no Buckingham ``3/2``
factor, no implied trace removal). Its far potential is
``q/r + mu.r/r^3 + Theta:(3 r r - r^2 I)/(2 r^5)``. For point sites the trace
of ``Theta`` only produces a contact term; for Gaussian sites it is a real
radial density and changes energies, so trace handling is never implicit.

A Gaussian site replaces ``delta`` by the unit-normalised isotropic Gaussian
``g_sigma(r) = (2 pi sigma^2)^(-3/2) exp(-r^2 / (2 sigma^2))`` with standard
deviation ``sigma`` (the pinned NVIDIA nvalchemi-toolkit-ops source width
convention ``exp(-r^2/(2 sigma^2))``). The coefficients are then derivative-
operator coefficients: the density has charge ``q``, dipole ``mu``, traceless
second moment ``traceless(Theta)`` and raw second-moment trace
``tr(Theta) + 3 q sigma^2`` (:func:`gaussian_density_moments`).

Packed bases
------------
``"cartesian-primitive"``
    ``[q, mu_x, mu_y, mu_z, T_xx, T_xy, T_xz, T_yy, T_yz, T_zz]`` truncated to
    the declared degree; the second moment keeps its trace.
``"native-real-orthonormal"``
    Degree-major native real O(3) layout of
    :class:`phydrax.nn.operator.representations.O3Representation`:
    ``[q, mu_y, mu_z, mu_x, c_-2 .. c_2]`` where ``c`` are the coordinates of
    ``traceless(Theta)`` in the Frobenius-orthonormal native symmetric-
    traceless basis ordered ``(xy, yz, 3 z^2 - r^2, xz, x^2 - y^2)`` (z-polar).
``"nvalchemi-ops-e3nn-component"``
    The pinned source packed layout (nvalchemi-toolkit-ops commit
    ``10e12340c39fd12a2375d6301768e72d2cd12868``,
    ``_multipole_moments.py``): ``[q, mu_y, mu_z, mu_x, c_-2 .. c_2]`` with the
    source's exact component-normalised, y-polar degree-two matrix
    (polynomials ordered ``(xz, xy, 3 y^2 - r^2, yz, z^2 - x^2)``). The source
    Cartesian ``Q`` equals ``traceless(Theta)`` of this module: its GTO-Ewald
    self coefficient ``F |Q|_F^2 / (320 pi^(3/2) sigma_c^5)`` is reproduced by
    the ``1/2 Theta : grad grad`` convention.

Both real-spherical layouts are traceless. Packing a degree-two site into
them requires ``trace="project"``; the removed trace is returned. Unpacking
requires an explicit trace array (or ``"zero"``). Shapes alone never select a
basis. Octupole and higher degrees are not part of the public definition.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from ..typing import parse


MultipoleBasis: TypeAlias = Literal[
    "cartesian-primitive", "native-real-orthonormal", "nvalchemi-ops-e3nn-component"
]
MultipoleTraceRequest: TypeAlias = Literal["retain", "project"]
GaussianWidthConvention: TypeAlias = Literal["standard-deviation", "gaussian-exponent"]

PUBLIC_MAXIMUM_DEGREE = 2
_SQRT_PI = math.sqrt(math.pi)
# Unique symmetric components in packed Cartesian order.
_UNIQUE_ROWS = (0, 0, 0, 1, 1, 2)
_UNIQUE_COLUMNS = (0, 1, 2, 1, 2, 2)
_UNIQUE_OF_MATRIX = ((0, 1, 2), (1, 3, 4), (2, 4, 5))

# Pinned source (nvalchemi-toolkit-ops, Apache-2.0) degree-two maps:
# q6 = [xx, yy, zz, xy, xz, yz] = _SOURCE_TO_CARTESIAN @ c and its exact
# pseudo-inverse on traceless tensors.
_S5 = math.sqrt(5.0)
_S15 = math.sqrt(15.0)
_SOURCE_TO_CARTESIAN = np.asarray(
    (
        (0.0, 0.0, -_S5 / 2.0, 0.0, -_S15 / 2.0),
        (0.0, 0.0, _S5, 0.0, 0.0),
        (0.0, 0.0, -_S5 / 2.0, 0.0, _S15 / 2.0),
        (0.0, _S15 / 2.0, 0.0, 0.0, 0.0),
        (_S15 / 2.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, _S15 / 2.0, 0.0),
    )
)
_CARTESIAN_TO_SOURCE = np.asarray(
    (
        (0.0, 0.0, 0.0, 0.0, 2.0 / _S15, 0.0),
        (0.0, 0.0, 0.0, 2.0 / _S15, 0.0, 0.0),
        (-_S5 / 15.0, 2.0 * _S5 / 15.0, -_S5 / 15.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 2.0 / _S15),
        (-1.0 / _S15, 0.0, 1.0 / _S15, 0.0, 0.0, 0.0),
    )
)
_SOURCE_DIPOLE_ORDER = (1, 2, 0)
_SOURCE_DIPOLE_INVERSE = (2, 0, 1)


def _degree(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("maximum_degree must be an integer.")
    degree = int(value)
    if not 0 <= degree <= PUBLIC_MAXIMUM_DEGREE:
        raise ValueError(
            "Public multipoles support degrees 0, 1 and 2 only; no octupole or "
            "higher-degree definition is claimed."
        )
    return degree


def _is_traced(value: Array, /) -> bool:
    return isinstance(value, jax_core.Tracer)


def component_count(basis: MultipoleBasis, maximum_degree: int, /) -> int:
    """Packed component count of ``basis`` through ``maximum_degree``."""
    basis = parse(basis, MultipoleBasis, "basis")
    degree = _degree(maximum_degree)
    second = 6 if basis == "cartesian-primitive" else 5
    return (1, 4, 4 + second)[degree]


def unique_second_moments(matrix: Array, /) -> Array:
    """``(..., 3, 3)`` symmetric tensors to ``(..., 6)`` ``xx, xy, xz, yy, yz, zz``."""
    return matrix[..., _UNIQUE_ROWS, _UNIQUE_COLUMNS]


def second_moment_matrix(unique: Array, /) -> Array:
    """``(..., 6)`` ``xx, xy, xz, yy, yz, zz`` to symmetric ``(..., 3, 3)``."""
    index = np.asarray(_UNIQUE_OF_MATRIX)
    return unique[..., index]


class CartesianMultipoles(StrictModule):
    """Site-centred primitive Cartesian multipoles through a declared degree.

    ``second_moments`` is the symmetric raw (traceful) tensor in the
    ``1/2 Theta : grad grad`` convention. Components above
    ``maximum_degree`` are stored as exact zeros so that the static degree,
    not a runtime value test, selects interaction terms.
    """

    charges: Array
    dipoles: Array
    second_moments: Array
    maximum_degree: int = eqx.field(static=True)

    def __init__(
        self,
        charges: ArrayLike,
        dipoles: ArrayLike | None = None,
        second_moments: ArrayLike | None = None,
        /,
        *,
        maximum_degree: int,
    ) -> None:
        degree = _degree(maximum_degree)
        charge = jnp.asarray(charges)
        if charge.ndim != 1 or charge.shape[0] == 0:
            raise ValueError("charges must have shape (N,) with N > 0.")
        if not jnp.issubdtype(charge.dtype, jnp.floating):
            raise TypeError("Multipole coefficients must be floating point.")
        count, dtype = charge.shape[0], charge.dtype
        if degree < 1 and dipoles is not None:
            raise ValueError("Dipoles were supplied for a degree-0 multipole set.")
        if degree < 2 and second_moments is not None:
            raise ValueError("Second moments were supplied below degree 2.")
        if (degree >= 1 and dipoles is None) or (degree == 2 and second_moments is None):
            raise ValueError(
                "Every coefficient through the declared degree must be explicit data."
            )
        dipole = (
            jnp.zeros((count, 3), dtype=dtype)
            if dipoles is None
            else jnp.asarray(dipoles, dtype=dtype)
        )
        if dipole.shape != (count, 3):
            raise ValueError("dipoles must have shape (N, 3).")
        if second_moments is None:
            second = jnp.zeros((count, 3, 3), dtype=dtype)
        else:
            raw = jnp.asarray(second_moments, dtype=dtype)
            if raw.shape == (count, 6):
                second = second_moment_matrix(raw)
            elif raw.shape == (count, 3, 3):
                if _is_traced(raw):
                    raise TypeError(
                        "Traced second moments must use the six unique components; "
                        "a traced 3x3 tensor cannot be certified symmetric."
                    )
                host = np.asarray(raw, dtype=np.float64)
                scale = max(float(np.max(np.abs(host), initial=0.0)), 1.0)
                if np.max(np.abs(host - np.swapaxes(host, -1, -2)), initial=0.0) > (
                    64.0 * np.finfo(np.dtype(dtype)).eps * scale
                ):
                    raise ValueError("Second-moment tensors must be symmetric.")
                second = raw
            else:
                raise ValueError("second_moments must have shape (N, 3, 3) or (N, 6).")
        self.charges = charge
        self.dipoles = dipole
        self.second_moments = second
        self.maximum_degree = degree

    @property
    def site_capacity(self) -> int:
        return self.charges.shape[0]

    def trace(self, /) -> Array:
        """Trace of the primitive second moment (zero below degree 2)."""
        return jnp.trace(self.second_moments, axis1=-2, axis2=-1)

    def project_traceless(self, /) -> tuple["CartesianMultipoles", Array]:
        """Explicit trace projection: ``(traceless moments, removed trace)``."""
        trace = self.trace()
        identity = jnp.eye(3, dtype=self.charges.dtype)
        projected = self.second_moments - trace[:, None, None] * identity / 3.0
        return (
            CartesianMultipoles(
                self.charges,
                self.dipoles if self.maximum_degree >= 1 else None,
                unique_second_moments(projected) if self.maximum_degree == 2 else None,
                maximum_degree=self.maximum_degree,
            ),
            trace,
        )

    def rotated(self, rotation: ArrayLike, /) -> "CartesianMultipoles":
        """Exact O(3) action ``mu -> R mu``, ``Theta -> R Theta R^T``."""
        matrix = jnp.asarray(rotation, dtype=self.charges.dtype)
        if matrix.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3).")
        dipoles = contract("ab,nb->na", matrix, self.dipoles)
        second = contract("ac,ncd,bd->nab", matrix, self.second_moments, matrix)
        return CartesianMultipoles(
            self.charges,
            dipoles if self.maximum_degree >= 1 else None,
            unique_second_moments(second) if self.maximum_degree == 2 else None,
            maximum_degree=self.maximum_degree,
        )

    def masked(self, mask: ArrayLike, /) -> "CartesianMultipoles":
        """Zero every coefficient of sites where ``mask`` is false."""
        keep = jnp.asarray(mask, dtype=jnp.bool_)
        if keep.shape != self.charges.shape:
            raise ValueError("mask must have shape (N,).")
        zero = jnp.zeros((), dtype=self.charges.dtype)
        return CartesianMultipoles(
            jnp.where(keep, self.charges, zero),
            jnp.where(keep[:, None], self.dipoles, zero)
            if self.maximum_degree >= 1
            else None,
            unique_second_moments(
                jnp.where(keep[:, None, None], self.second_moments, zero)
            )
            if self.maximum_degree == 2
            else None,
            maximum_degree=self.maximum_degree,
        )

    def scaled(self, factor: ArrayLike, /) -> "CartesianMultipoles":
        """Multiply all coefficients by a scalar or per-site factor."""
        value = jnp.asarray(factor, dtype=self.charges.dtype)
        if value.ndim == 1:
            value_vector, value_matrix = value[:, None], value[:, None, None]
        elif value.ndim == 0:
            value_vector = value_matrix = value
        else:
            raise ValueError("factor must be scalar or have shape (N,).")
        return CartesianMultipoles(
            value * self.charges,
            value_vector * self.dipoles if self.maximum_degree >= 1 else None,
            unique_second_moments(value_matrix * self.second_moments)
            if self.maximum_degree == 2
            else None,
            maximum_degree=self.maximum_degree,
        )


def dipole_multipoles(dipoles: ArrayLike, /) -> CartesianMultipoles:
    """A pure-dipole degree-1 set (zero charges), e.g. induced dipoles."""
    value = jnp.asarray(dipoles)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("dipoles must have shape (N, 3).")
    return CartesianMultipoles(
        jnp.zeros((value.shape[0],), dtype=value.dtype), value, maximum_degree=1
    )


def add_multipoles(
    left: CartesianMultipoles, right: CartesianMultipoles, /
) -> CartesianMultipoles:
    """Coefficient-wise sum of two site sets on the same capacity."""
    if left.site_capacity != right.site_capacity:
        raise ValueError("Multipole sets must share their site capacity.")
    degree = max(left.maximum_degree, right.maximum_degree)
    return CartesianMultipoles(
        left.charges + right.charges,
        left.dipoles + right.dipoles if degree >= 1 else None,
        unique_second_moments(left.second_moments + right.second_moments)
        if degree == 2
        else None,
        maximum_degree=degree,
    )


def shift_multipole_origin(
    moments: CartesianMultipoles, displacement: ArrayLike, /
) -> CartesianMultipoles:
    """Exact moment map to an origin displaced by ``d`` (new = old + d).

    ``q' = q``, ``mu' = mu - q d`` and
    ``Theta' = Theta - mu d - d mu + q d d`` are the exact raw moments about
    the new origin. The degree-2 truncation of the re-expanded interaction is
    not exact (shifting generates higher moments), so interaction owners
    always require site-centred moments.
    """
    if moments.maximum_degree < 2:
        raise ValueError(
            "An exact origin shift of a dipole or quadrupole requires the "
            "degree-2 primitive set (a shifted charge generates a dipole and "
            "second moment)."
        )
    delta = jnp.asarray(displacement, dtype=moments.charges.dtype)
    if delta.shape != (moments.site_capacity, 3):
        raise ValueError("displacement must have shape (N, 3).")
    q = moments.charges
    mu = moments.dipoles
    second = (
        moments.second_moments
        - mu[:, :, None] * delta[:, None, :]
        - delta[:, :, None] * mu[:, None, :]
        + q[:, None, None] * delta[:, :, None] * delta[:, None, :]
    )
    return CartesianMultipoles(
        q,
        mu - q[:, None] * delta,
        unique_second_moments(second),
        maximum_degree=2,
    )


class MultipolePackedValues(StrictModule):
    """Packed components, the declared basis and any explicitly removed trace."""

    values: Array
    removed_trace: Array | None
    basis: str = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)


def multipole_convention_id(basis: MultipoleBasis, maximum_degree: int, /) -> str:
    basis = parse(basis, MultipoleBasis, "basis")
    return canonical_fingerprint(
        {
            "kind": "multipole-basis-convention",
            "basis": basis,
            "maximum_degree": _degree(maximum_degree),
            "moments": "site-centred-primitive-half-theta-grad-grad",
            "source_pin": (
                "nvalchemi-toolkit-ops@10e12340c39fd12a2375d6301768e72d2cd12868"
                if basis == "nvalchemi-ops-e3nn-component"
                else None
            ),
        }
    )


def _native_tensor_components(traceless: Array, /) -> Array:
    from ..nn.operator.representations import O3Features, O3Representation

    count, dtype = traceless.shape[0], traceless.dtype
    representation = O3Representation(tensors=1)
    features = O3Features(
        scalars=jnp.zeros((count, 0), dtype=dtype),
        pseudoscalars=jnp.zeros((count, 0), dtype=dtype),
        vectors=jnp.zeros((count, 0, 3), dtype=dtype),
        pseudovectors=jnp.zeros((count, 0, 3), dtype=dtype),
        tensors=traceless[:, None, :, :],
        pseudotensors=jnp.zeros((count, 0, 3, 3), dtype=dtype),
    )
    coefficients = representation.join(features)
    basis = jnp.asarray(O3Representation.real_harmonic_basis(2), dtype=dtype)
    return contract("mk,nk->nm", basis, coefficients)


def _native_tensor_matrix(components: Array, /) -> Array:
    from ..nn.operator.representations import O3Representation

    dtype = components.dtype
    basis = jnp.asarray(O3Representation.real_harmonic_basis(2), dtype=dtype)
    coefficients = contract("mk,nm->nk", basis, components)
    return O3Representation(tensors=1).split(coefficients).tensors[:, 0]


def _native_vector_components(vectors: Array, /) -> Array:
    from ..nn.operator.representations import O3Representation

    basis = jnp.asarray(O3Representation.real_harmonic_basis(1), dtype=vectors.dtype)
    return contract("ab,nb->na", basis, vectors)


def _native_vector_matrix(components: Array, /) -> Array:
    from ..nn.operator.representations import O3Representation

    basis = jnp.asarray(O3Representation.real_harmonic_basis(1), dtype=components.dtype)
    return contract("ab,na->nb", basis, components)


def pack_multipoles(
    moments: CartesianMultipoles,
    basis: MultipoleBasis,
    /,
    *,
    trace: MultipoleTraceRequest,
) -> MultipolePackedValues:
    """Pack site-centred primitive moments into a declared basis.

    Real-spherical degree-two layouts are traceless: they require
    ``trace="project"`` and return the removed trace. Every other packing
    requires ``trace="retain"``.
    """
    basis = parse(basis, MultipoleBasis, "basis")
    request = parse(trace, MultipoleTraceRequest, "trace")
    degree = moments.maximum_degree
    spherical_projection = basis != "cartesian-primitive" and degree == 2
    if spherical_projection and request != "project":
        raise ValueError(
            f"Basis {basis!r} is traceless at degree 2; request trace='project' "
            "explicitly to drop and report the trace."
        )
    if not spherical_projection and request != "retain":
        raise ValueError("trace='project' only applies to real-spherical degree 2.")
    parts = [moments.charges[:, None]]
    removed: Array | None = None
    if basis == "cartesian-primitive":
        if degree >= 1:
            parts.append(moments.dipoles)
        if degree == 2:
            parts.append(unique_second_moments(moments.second_moments))
    else:
        if degree >= 1:
            parts.append(
                _native_vector_components(moments.dipoles)
                if basis == "native-real-orthonormal"
                else moments.dipoles[:, _SOURCE_DIPOLE_ORDER]
            )
        if degree == 2:
            projected, removed = moments.project_traceless()
            traceless = projected.second_moments
            if basis == "native-real-orthonormal":
                parts.append(_native_tensor_components(traceless))
            else:
                q6 = jnp.stack(
                    (
                        traceless[:, 0, 0],
                        traceless[:, 1, 1],
                        traceless[:, 2, 2],
                        traceless[:, 0, 1],
                        traceless[:, 0, 2],
                        traceless[:, 1, 2],
                    ),
                    axis=-1,
                )
                matrix = jnp.asarray(_CARTESIAN_TO_SOURCE, dtype=q6.dtype)
                parts.append(contract("mk,nk->nm", matrix, q6))
    return MultipolePackedValues(
        values=jnp.concatenate(parts, axis=-1),
        removed_trace=removed,
        basis=basis,
        maximum_degree=degree,
        convention_id=multipole_convention_id(basis, degree),
    )


def unpack_multipoles(
    values: ArrayLike,
    basis: MultipoleBasis,
    maximum_degree: int,
    /,
    *,
    trace: ArrayLike | Literal["zero"] | None = None,
) -> CartesianMultipoles:
    """Inverse of :func:`pack_multipoles` with an explicit trace.

    A real-spherical degree-two layout carries no trace, so ``trace`` must be
    the primitive trace per site or ``"zero"``; other layouts refuse it.
    """
    basis = parse(basis, MultipoleBasis, "basis")
    degree = _degree(maximum_degree)
    packed = jnp.asarray(values)
    count = packed.shape[0] if packed.ndim == 2 else 0
    if packed.ndim != 2 or packed.shape[1] != component_count(basis, degree):
        raise ValueError(
            f"Packed {basis!r} degree-{degree} values need shape "
            f"(N, {component_count(basis, degree)})."
        )
    needs_trace = basis != "cartesian-primitive" and degree == 2
    if needs_trace and trace is None:
        raise ValueError(
            "A traceless real-spherical degree-2 layout requires an explicit "
            "primitive trace array or trace='zero'."
        )
    if not needs_trace and trace is not None:
        raise ValueError("trace only applies to real-spherical degree-2 layouts.")
    charges = packed[:, 0]
    dipoles: Array | None = None
    second: Array | None = None
    if basis == "cartesian-primitive":
        if degree >= 1:
            dipoles = packed[:, 1:4]
        if degree == 2:
            second = packed[:, 4:10]
    else:
        if degree >= 1:
            components = packed[:, 1:4]
            dipoles = (
                _native_vector_matrix(components)
                if basis == "native-real-orthonormal"
                else components[:, _SOURCE_DIPOLE_INVERSE]
            )
        if degree == 2:
            components = packed[:, 4:9]
            if basis == "native-real-orthonormal":
                traceless = _native_tensor_matrix(components)
            else:
                matrix = jnp.asarray(_SOURCE_TO_CARTESIAN, dtype=packed.dtype)
                q6 = contract("km,nm->nk", matrix, components)
                xx, yy, zz, xy, xz, yz = (q6[:, index] for index in range(6))
                traceless = second_moment_matrix(
                    jnp.stack((xx, xy, xz, yy, yz, zz), axis=-1)
                )
            if isinstance(trace, str):
                if trace != "zero":
                    raise ValueError("trace must be an array or 'zero'.")
                trace_values = jnp.zeros((count,), dtype=packed.dtype)
            else:
                trace_values = jnp.asarray(trace, dtype=packed.dtype)
                if trace_values.shape != (count,):
                    raise ValueError("trace must have shape (N,).")
            identity = jnp.eye(3, dtype=packed.dtype)
            second = unique_second_moments(
                traceless + trace_values[:, None, None] * identity / 3.0
            )
    return CartesianMultipoles(charges, dipoles, second, maximum_degree=degree)


class GaussianMultipoleWidths(StrictModule):
    """Per-site Gaussian standard deviations through a small width-class table.

    ``class_widths`` (``W``) are dynamic numerical leaves (standard deviations
    of ``exp(-r^2/(2 sigma^2))``) and ``site_classes`` (``N``) fixed integer
    data. The class table makes reciprocal channels separable by width; the
    pinned source public Ewald/PME energy wrappers admit exactly one scalar
    width, i.e. ``W = 1`` with every site in class 0. Concrete widths are
    admitted eagerly; traced widths cannot be, so every consumer gates its
    result on :meth:`valid` (finite and strictly positive), never on the
    squared width alone.

    PyTree layout: ``class_count`` is the static width-class table shape; the
    exact canonical digest of the site-class assignment is the dynamic
    ``widths_identity`` (``widths_id`` decodes it on the host), so width
    tables that differ only in content share one executable.
    """

    class_widths: Array
    site_classes: Array
    class_count: int = eqx.field(static=True)
    widths_identity: IdentityWords

    def __init__(
        self,
        class_widths: ArrayLike,
        site_classes: ArrayLike,
        /,
        *,
        convention: GaussianWidthConvention = "standard-deviation",
    ) -> None:
        convention = parse(convention, GaussianWidthConvention, "convention")
        widths = jnp.asarray(class_widths)
        classes = np.asarray(site_classes)
        if widths.ndim != 1 or widths.shape[0] == 0:
            raise ValueError("class_widths must have shape (W,) with W > 0.")
        if not jnp.issubdtype(widths.dtype, jnp.floating):
            raise TypeError("class_widths must be floating point.")
        if classes.ndim != 1 or classes.size == 0 or classes.dtype.kind not in "iu":
            raise ValueError("site_classes must be a non-empty integer array (N,).")
        if np.any(classes < 0) or np.any(classes >= widths.shape[0]):
            raise ValueError("site_classes must index the width-class table.")
        if not _is_traced(widths):
            host = np.asarray(widths, dtype=np.float64)
            if np.any(~np.isfinite(host)) or np.any(host <= 0.0):
                raise ValueError("Gaussian widths must be finite and positive.")
        if convention == "gaussian-exponent":
            widths = 1.0 / jnp.sqrt(2.0 * widths)
        self.class_widths = widths
        self.site_classes = jnp.asarray(classes, dtype=jnp.int32)
        self.class_count = int(widths.shape[0])
        self.widths_identity = IdentityWords.of(
            {
                "kind": "gaussian-multipole-widths",
                "convention": "standard-deviation-exp-r2-over-2-sigma2",
                "class_count": self.class_count,
                "site_classes": classes.astype(np.int64).tolist(),
            }
        )

    @property
    def widths_id(self) -> str:
        """Host decoding of the canonical width-class assignment digest."""
        return self.widths_identity.fingerprint()

    @property
    def site_capacity(self) -> int:
        return int(self.site_classes.shape[0])

    def site_widths(self, /) -> Array:
        return self.class_widths[self.site_classes]

    def valid(self, /) -> Array:
        """Traced admission: every class width finite and strictly positive."""
        return jnp.all(jnp.isfinite(self.class_widths) & (self.class_widths > 0.0))


def gaussian_density_moments(
    coefficients: CartesianMultipoles, widths: GaussianMultipoleWidths, /
) -> CartesianMultipoles:
    """Raw density moments of Gaussian multipole coefficients.

    Charge and dipole are unchanged; the raw second moment gains
    ``q sigma^2 I`` from the charge Gaussian's own spread.
    """
    if widths.site_capacity != coefficients.site_capacity:
        raise ValueError("Widths and multipoles must share their site capacity.")
    if coefficients.maximum_degree < 2:
        raise ValueError(
            "Raw Gaussian density second moments require degree-2 coefficients."
        )
    sigma = widths.site_widths().astype(coefficients.charges.dtype)
    identity = jnp.eye(3, dtype=coefficients.charges.dtype)
    second = (
        coefficients.second_moments
        + (coefficients.charges * sigma**2)[:, None, None] * identity
    )
    return CartesianMultipoles(
        coefficients.charges,
        coefficients.dipoles,
        unique_second_moments(second),
        maximum_degree=2,
    )


def gaussian_coefficients(
    density_moments: CartesianMultipoles, widths: GaussianMultipoleWidths, /
) -> CartesianMultipoles:
    """Exact inverse of :func:`gaussian_density_moments`."""
    if widths.site_capacity != density_moments.site_capacity:
        raise ValueError("Widths and multipoles must share their site capacity.")
    if density_moments.maximum_degree < 2:
        raise ValueError("Gaussian coefficients from raw moments require degree 2.")
    sigma = widths.site_widths().astype(density_moments.charges.dtype)
    identity = jnp.eye(3, dtype=density_moments.charges.dtype)
    second = (
        density_moments.second_moments
        - (density_moments.charges * sigma**2)[:, None, None] * identity
    )
    return CartesianMultipoles(
        density_moments.charges,
        density_moments.dipoles,
        unique_second_moments(second),
        maximum_degree=2,
    )


def erf_kernel_origin_coefficients(beta: ArrayLike, order: int, /) -> tuple[Array, ...]:
    """``B_n(0)`` of ``erf(beta r)/r`` for ``n = 0..order``.

    With ``B_0 = f`` and ``B_{n+1} = -(1/r) dB_n/dr``,
    ``B_n(0) = (2 beta / sqrt(pi)) (2 beta^2)^n / (2 n + 1)``. These are the
    Gaussian self/overlap coefficients: a pair of co-centred Gaussians with
    standard deviations ``s_a, s_b`` uses ``beta = 1/sqrt(2 (s_a^2 + s_b^2))``.
    """
    value = jnp.asarray(beta)
    lead = 2.0 * value / _SQRT_PI
    return tuple(
        lead * (2.0 * value * value) ** n / (2 * n + 1) for n in range(int(order) + 1)
    )


def co_centered_interaction(
    coefficients: tuple[Array, ...],
    left: CartesianMultipoles,
    right: CartesianMultipoles,
    /,
) -> Array:
    """Per-site ``O_L(-grad) O_R(grad) f`` at zero separation.

    ``coefficients`` are ``(B_0, B_1, B_2)`` of a radial kernel at the origin
    (per site or scalar). The result is
    ``q_L q_R B_0 - (q_L t_R + q_R t_L) B_1 / 2 + mu_L . mu_R B_1
    + (t_L t_R + 2 Theta_L : Theta_R) B_2 / 4``.
    """
    b0, b1, b2 = coefficients[:3]
    value = left.charges * right.charges * b0
    degree = max(left.maximum_degree, right.maximum_degree)
    if degree >= 1:
        value = value + contract("na,na->n", left.dipoles, right.dipoles) * b1
    if degree == 2:
        trace_left, trace_right = left.trace(), right.trace()
        value = (
            value - 0.5 * (left.charges * trace_right + right.charges * trace_left) * b1
        )
        value = (
            value
            + 0.25
            * (
                trace_left * trace_right
                + 2.0 * contract("nab,nab->n", left.second_moments, right.second_moments)
            )
            * b2
        )
    return value


def gaussian_self_energy(
    coefficients: CartesianMultipoles,
    widths: GaussianMultipoleWidths,
    coulomb_constant: float,
    /,
) -> Array:
    """Per-site intrasite energy ``(k/2) O(-grad) O(grad) erf(r/(2 sigma))/r |_0``.

    Invalid (nonfinite or nonpositive) widths give NaN, never an energy of
    ``|sigma|``.
    """
    if widths.site_capacity != coefficients.site_capacity:
        raise ValueError("Widths and multipoles must share their site capacity.")
    sigma = widths.site_widths().astype(coefficients.charges.dtype)
    valid = jnp.isfinite(sigma) & (sigma > 0.0)
    beta = 1.0 / (2.0 * jnp.where(valid, sigma, 1.0))
    value = (
        0.5
        * coulomb_constant
        * co_centered_interaction(
            erf_kernel_origin_coefficients(beta, 2), coefficients, coefficients
        )
    )
    return jnp.where(valid, value, jnp.nan)


__all__ = [
    "CartesianMultipoles",
    "GaussianMultipoleWidths",
    "GaussianWidthConvention",
    "MultipoleBasis",
    "MultipolePackedValues",
    "MultipoleTraceRequest",
    "PUBLIC_MAXIMUM_DEGREE",
    "add_multipoles",
    "co_centered_interaction",
    "component_count",
    "dipole_multipoles",
    "erf_kernel_origin_coefficients",
    "gaussian_coefficients",
    "gaussian_density_moments",
    "gaussian_self_energy",
    "multipole_convention_id",
    "pack_multipoles",
    "second_moment_matrix",
    "shift_multipole_origin",
    "unique_second_moments",
    "unpack_multipoles",
]
