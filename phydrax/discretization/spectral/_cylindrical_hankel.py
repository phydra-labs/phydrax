#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-radius Bessel-zero Hankel transforms with explicit certification.

`CylindricalHankelPlan` is the self-reciprocal fixed-order quadrature (QDHT).
`SharedGridHankelPlan` is the quasi-cylindrical transform family: one uniform
cell-centered radial grid shared by every azimuthal mode and by the Bessel
orders ``m − 1``, ``m``, ``m + 1`` that carry ``(F_r ∓ iF_θ)/2`` and ``F_z``.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.special import jn_zeros

from ... import ein
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import LinearSolveStatus, pseudoinverse
from ...special import jv
from ...typing import parse


SharedGridHankelOffset: TypeAlias = Literal[-1, 0, 1]


class CylindricalHankelStatus(IntEnum):
    """Preparation status for one finite-radius transform."""

    SUCCESS = 0
    ROOT_RESIDUAL = 1
    ORTHOGONALITY = 2
    INVERSE = 3
    PARSEVAL = 4


class CylindricalHankelEvidence(StrictModule, NonTrainableState):
    """Root, orthogonality, inverse, metric, and resource certification."""

    root_residual: Array
    orthogonality_defect: Array
    inverse_defect: Array
    parseval_defect: Array
    finite: Array
    successful: Array
    status: Array
    matrix_elements: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class CylindricalHankelPlan(StrictModule, NonTrainableState):
    """Static finite-radius, fixed-order Bessel-zero transform policy.

    The transform samples radius at ``r_n = alpha_n R / alpha_(N+1)`` and
    transverse angular wavenumber at ``k_n = alpha_n / R``. It approximates
    the self-reciprocal Hankel pair using the Bessel-zero quadrature. The first
    release admits one fixed non-negative integer azimuthal order; nonlinear
    optical propagation is separately restricted to order zero.
    """

    radius: float = eqx.field(static=True)
    radial_count: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    root_tolerance: float = eqx.field(static=True)
    orthogonality_tolerance: float = eqx.field(static=True)
    inverse_tolerance: float = eqx.field(static=True)
    parseval_tolerance: float = eqx.field(static=True)
    maximum_matrix_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        radial_count: int,
        /,
        *,
        order: int = 0,
        root_tolerance: float = 1.0e-8,
        orthogonality_tolerance: float = 2.0e-6,
        inverse_tolerance: float = 2.0e-6,
        parseval_tolerance: float = 2.0e-6,
        maximum_matrix_elements: int = 4_194_304,
    ) -> None:
        radius_value = float(radius)
        count = int(radial_count)
        azimuthal_order = int(order)
        tolerances = (
            float(root_tolerance),
            float(orthogonality_tolerance),
            float(inverse_tolerance),
            float(parseval_tolerance),
        )
        capacity = int(maximum_matrix_elements)
        if not math.isfinite(radius_value) or radius_value <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if count < 2:
            raise ValueError("radial_count must be at least two.")
        if azimuthal_order < 0:
            raise ValueError("order must be a non-negative integer.")
        if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError(
                "Hankel certification tolerances must be finite and positive."
            )
        if capacity <= 0 or count * count > capacity:
            raise ValueError("The Hankel matrix exceeds maximum_matrix_elements.")
        self.radius = radius_value
        self.radial_count = count
        self.order = azimuthal_order
        (
            self.root_tolerance,
            self.orthogonality_tolerance,
            self.inverse_tolerance,
            self.parseval_tolerance,
        ) = tolerances
        self.maximum_matrix_elements = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cylindrical-hankel-plan",
                "radius": radius_value,
                "radial_count": count,
                "order": azimuthal_order,
                "root_tolerance": tolerances[0],
                "orthogonality_tolerance": tolerances[1],
                "inverse_tolerance": tolerances[2],
                "parseval_tolerance": tolerances[3],
                "maximum_matrix_elements": capacity,
            }
        )

    def prepare(self, /) -> PreparedCylindricalHankel:
        roots_host = jn_zeros(self.order, self.radial_count + 1)
        roots = jnp.asarray(roots_host[:-1], dtype=jnp.float64)
        boundary_root = jnp.asarray(roots_host[-1], dtype=jnp.float64)
        radius = jnp.asarray(self.radius, dtype=roots.dtype)
        radial_coordinates = radius * roots / boundary_root
        transverse_angular_wavenumbers = roots / radius

        next_order_values = jv(float(self.order + 1), roots)
        arguments = roots[:, None] * roots[None, :] / boundary_root
        kernel = jv(float(self.order), arguments)
        radial_weights = (
            2.0
            * radius
            * radius
            / (boundary_root * boundary_root * next_order_values * next_order_values)
        )
        maximum_wavenumber = boundary_root / radius
        spectral_weights = (
            2.0
            * maximum_wavenumber
            * maximum_wavenumber
            / (boundary_root * boundary_root * next_order_values * next_order_values)
        )
        forward_matrix = kernel * radial_weights[None, :]
        inverse_matrix = kernel.T * spectral_weights[None, :]
        normalized_matrix = (
            2.0
            * kernel
            / (boundary_root * next_order_values[:, None] * next_order_values[None, :])
        )

        identity = jnp.eye(self.radial_count, dtype=roots.dtype)
        root_residual = jnp.max(jnp.abs(jv(float(self.order), roots)))
        orthogonality_defect = jnp.linalg.norm(
            ein.contract("mn,nk->mk", normalized_matrix, normalized_matrix) - identity,
            ord="fro",
        ) / jnp.sqrt(jnp.asarray(self.radial_count, dtype=roots.dtype))
        inverse_defect = jnp.linalg.norm(
            ein.contract("mn,nk->mk", inverse_matrix, forward_matrix) - identity,
            ord="fro",
        ) / jnp.sqrt(jnp.asarray(self.radial_count, dtype=roots.dtype))
        weighted_metric = ein.contract(
            "mn,m,mk->nk",
            jnp.conj(forward_matrix),
            spectral_weights,
            forward_matrix,
        )
        radial_metric = jnp.diag(radial_weights)
        metric_scale = jnp.maximum(jnp.linalg.norm(radial_metric, ord="fro"), 1.0e-30)
        parseval_defect = (
            jnp.linalg.norm(
                weighted_metric - radial_metric,
                ord="fro",
            )
            / metric_scale
        )
        finite = jnp.asarray(
            jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (
                            root_residual,
                            orthogonality_defect,
                            inverse_defect,
                            parseval_defect,
                        )
                    )
                )
            )
        )
        root_ok = root_residual <= self.root_tolerance
        orthogonality_ok = orthogonality_defect <= self.orthogonality_tolerance
        inverse_ok = inverse_defect <= self.inverse_tolerance
        parseval_ok = parseval_defect <= self.parseval_tolerance
        successful = finite & root_ok & orthogonality_ok & inverse_ok & parseval_ok
        status = jnp.where(
            ~finite | ~root_ok,
            int(CylindricalHankelStatus.ROOT_RESIDUAL),
            jnp.where(
                ~orthogonality_ok,
                int(CylindricalHankelStatus.ORTHOGONALITY),
                jnp.where(
                    ~inverse_ok,
                    int(CylindricalHankelStatus.INVERSE),
                    jnp.where(
                        ~parseval_ok,
                        int(CylindricalHankelStatus.PARSEVAL),
                        int(CylindricalHankelStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        itemsize = np.dtype(np.asarray(roots).dtype).itemsize
        retained_bytes = itemsize * (
            3 * self.radial_count * self.radial_count + 6 * self.radial_count + 1
        )
        fingerprint = array_tree_fingerprint(
            {
                "roots": roots,
                "boundary_root": boundary_root,
                "radial_coordinates": radial_coordinates,
                "transverse_angular_wavenumbers": transverse_angular_wavenumbers,
                "radial_weights": radial_weights,
                "spectral_weights": spectral_weights,
                "forward_matrix": forward_matrix,
                "inverse_matrix": inverse_matrix,
            }
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cylindrical-hankel",
                "plan_id": self.plan_id,
                "arrays": fingerprint,
            }
        )
        evidence = CylindricalHankelEvidence(
            root_residual,
            orthogonality_defect,
            inverse_defect,
            parseval_defect,
            finite,
            successful,
            status,
            matrix_elements=self.radial_count * self.radial_count,
            retained_bytes=retained_bytes,
            plan_id=self.plan_id,
            prepared_id=prepared_id,
        )
        return PreparedCylindricalHankel(
            self,
            roots,
            boundary_root,
            radial_coordinates,
            transverse_angular_wavenumbers,
            radial_weights,
            spectral_weights,
            forward_matrix,
            inverse_matrix,
            normalized_matrix,
            evidence,
            prepared_id=prepared_id,
        )


class PreparedCylindricalHankel(StrictModule, NonTrainableState):
    """Prepared physical Hankel pair and its certified quadrature."""

    plan: CylindricalHankelPlan
    roots: Array
    boundary_root: Array
    radial_coordinates: Array
    transverse_angular_wavenumbers: Array
    radial_weights: Array
    spectral_weights: Array
    forward_matrix: Array
    inverse_matrix: Array
    normalized_matrix: Array
    evidence: CylindricalHankelEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def area_weights(self) -> Array:
        """Axisymmetric physical-area quadrature weights ``2*pi*r*dr``."""
        return 2.0 * math.pi * self.radial_weights

    def forward(self, values: ArrayLike, /, *, axis: int = -1) -> Array:
        """Approximate ``integral f(r) J_m(k r) r dr`` on the prepared grid."""
        return _apply_matrix(self.forward_matrix, values, axis, self.plan.radial_count)

    def inverse(self, values: ArrayLike, /, *, axis: int = -1) -> Array:
        """Approximate ``integral F(k) J_m(k r) k dk`` on the prepared grid."""
        return _apply_matrix(self.inverse_matrix, values, axis, self.plan.radial_count)

    def normalized(self, values: ArrayLike, /, *, axis: int = -1) -> Array:
        """Apply the symmetric dimensionless self-inverse QDHT matrix."""
        return _apply_matrix(self.normalized_matrix, values, axis, self.plan.radial_count)


def prepare_cylindrical_hankel(
    plan: CylindricalHankelPlan,
    /,
) -> PreparedCylindricalHankel:
    """Prepare one finite-radius Bessel-zero transform."""
    if not isinstance(plan, CylindricalHankelPlan):
        raise TypeError("plan must be a CylindricalHankelPlan.")
    return plan.prepare()


def _apply_matrix(
    matrix: Array,
    values: ArrayLike,
    axis: int,
    expected_count: int,
) -> Array:
    array = jnp.asarray(values)
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += array.ndim
    if normalized_axis < 0 or normalized_axis >= array.ndim:
        raise ValueError("axis is outside the input rank.")
    if array.shape[normalized_axis] != expected_count:
        raise ValueError("The selected axis does not match the prepared radial count.")
    moved = jnp.moveaxis(array, normalized_axis, -1)
    transformed = ein.contract("mn,...n->...m", matrix, moved)
    return jnp.moveaxis(transformed, -1, normalized_axis)


class SharedGridHankelEvidence(StrictModule, NonTrainableState):
    """Rank, conditioning, and pseudoinverse evidence of every mode and order.

    Arrays are ``[mode, 3]`` with order index ``0, 1, 2`` for the Bessel orders
    ``m − 1, m, m + 1``. ``expected_rank`` is the synthesis rank the shared
    k-grid admits: full for ``m = 0`` and for order ``m − 1``, one less for
    orders ``m`` and ``m + 1`` at ``m ≥ 1`` (their ``k = 0`` column vanishes, so
    one radial direction per such pair has no spectral representation).
    ``condition`` is the retained-spectrum condition estimate and
    ``pseudoinverse_residual`` the relative Penrose residual ``‖AA⁺A − A‖/‖A‖``;
    ``status`` holds the native `phydrax.linalg.LinearSolveStatus` codes.
    """

    rank: Array
    expected_rank: Array
    condition: Array
    pseudoinverse_residual: Array
    status: Array
    successful: Array
    matrix_elements: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class SharedGridHankelPlan(StrictModule, NonTrainableState):
    """Quasi-cylindrical Hankel transforms of orders ``m − 1, m, m + 1``.

    Every azimuthal mode ``m = 0, …, mode_count − 1`` samples the same radii
    ``r_j = (j + ½)R/N``. Mode ``m`` uses one k-grid ``k_{m,n} = α_{m,n}/R``
    from the zeros of ``J_m``: the ``N`` positive zeros for ``m = 0``, and
    ``k = 0`` followed by the first ``N − 1`` positive zeros for ``m ≥ 1``.
    Synthesis is ``f(r_j) = Σ_n c_n J_p(k_{m,n} r_j)`` for
    ``p ∈ {m − 1, m, m + 1}`` with ``J_{−1} = −J_1`` at ``m = 0``; the ``k = 0``
    column of order ``m − 1`` is its ``k → 0`` limit shape ``(r/R)^{m−1}`` (the
    harmonic polynomial ``(x + iy)^{m−1}``) and vanishes for orders ``m`` and
    ``m + 1``. Analysis is the Moore–Penrose pseudoinverse of each synthesis
    matrix, prepared once through `phydrax.linalg.pseudoinverse` with its rank
    and conditioning evidence.

    Because ``J_{m−1}(α) = −J_{m+1}(α)`` at zeros of ``J_m``, the three orders of
    one mode share the Fourier–Bessel normalization of each k-node: the order
    ``m ∓ 1`` coefficients of ``(F_r ∓ iF_θ)/2`` and the order ``m``
    coefficients of ``F_z`` are, up to one common factor per node, the Cartesian
    spectrum at transverse wavevector ``(k_{m,n}, 0)`` (Lehe et al., Comput.
    Phys. Commun. 203, 66, 2016).
    """

    radius: float = eqx.field(static=True)
    radial_count: int = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    maximum_matrix_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        radial_count: int,
        mode_count: int,
        /,
        *,
        maximum_matrix_elements: int = 16_777_216,
    ) -> None:
        radius_value = float(radius)
        count = int(radial_count)
        modes = int(mode_count)
        capacity = int(maximum_matrix_elements)
        if not math.isfinite(radius_value) or radius_value <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if count < 2:
            raise ValueError("radial_count must be at least two.")
        if modes < 1:
            raise ValueError("mode_count must be at least one.")
        if capacity <= 0 or 6 * modes * count * count > capacity:
            raise ValueError(
                "The shared-grid Hankel matrices exceed maximum_matrix_elements."
            )
        self.radius = radius_value
        self.radial_count = count
        self.mode_count = modes
        self.maximum_matrix_elements = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shared-grid-hankel-plan",
                "radius": radius_value,
                "radial_count": count,
                "mode_count": modes,
                "maximum_matrix_elements": capacity,
            }
        )

    def prepare(self, /) -> PreparedSharedGridHankel:
        count, modes = self.radial_count, self.mode_count
        radii = self.radius * (np.arange(count, dtype=np.float64) + 0.5) / count
        wavenumbers = np.stack(
            tuple(
                (
                    jn_zeros(0, count)
                    if mode == 0
                    else np.concatenate(([0.0], jn_zeros(mode, count - 1)))
                )
                / self.radius
                for mode in range(modes)
            )
        )
        synthesis = np.zeros((3, modes, count, count), dtype=np.float64)
        analysis = np.zeros((3, modes, count, count), dtype=np.float64)
        rank = np.zeros((modes, 3), dtype=np.int32)
        expected = np.zeros((modes, 3), dtype=np.int32)
        condition = np.zeros((modes, 3), dtype=np.float64)
        residual = np.zeros((modes, 3), dtype=np.float64)
        status = np.zeros((modes, 3), dtype=np.int32)
        for mode in range(modes):
            argument = jnp.asarray(radii[:, None] * wavenumbers[mode][None, :])
            for index, offset in enumerate((-1, 0, 1)):
                order = mode + offset
                # J_{-1} = -J_1; the native Bessel owner takes nonnegative orders.
                matrix = np.array(
                    math.copysign(1.0, order) * jv(float(abs(order)), argument),
                    dtype=np.float64,
                )
                if mode > 0:
                    matrix[:, 0] = (
                        (radii / self.radius) ** (mode - 1) if offset == -1 else 0.0
                    )
                inverse = pseudoinverse(jnp.asarray(matrix))
                synthesis[index, mode] = matrix
                analysis[index, mode] = np.asarray(inverse.value)
                rank[mode, index] = int(inverse.diagnostics.rank)
                expected[mode, index] = count if mode == 0 or offset == -1 else count - 1
                condition[mode, index] = float(inverse.diagnostics.condition_estimate)
                residual[mode, index] = float(inverse.diagnostics.relative_residual)
                status[mode, index] = int(inverse.status)
        finite = bool(
            np.all(np.isfinite(synthesis))
            and np.all(np.isfinite(analysis))
            and np.all(np.isfinite(condition))
        )
        successful = (
            finite
            and bool(np.all(status == int(LinearSolveStatus.SUCCESS)))
            and bool(np.all(rank == expected))
        )
        fingerprint = array_tree_fingerprint(
            {
                "radii": radii,
                "wavenumbers": wavenumbers,
                "synthesis": synthesis,
                "analysis": analysis,
            }
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-shared-grid-hankel",
                "plan_id": self.plan_id,
                "arrays": fingerprint,
            }
        )
        evidence = SharedGridHankelEvidence(
            rank=jnp.asarray(rank),
            expected_rank=jnp.asarray(expected),
            condition=jnp.asarray(condition),
            pseudoinverse_residual=jnp.asarray(residual),
            status=jnp.asarray(status),
            successful=jnp.asarray(successful),
            matrix_elements=6 * modes * count * count,
            retained_bytes=8 * (6 * modes * count * count + modes * count + count),
            plan_id=self.plan_id,
            prepared_id=prepared_id,
        )
        return PreparedSharedGridHankel(
            self,
            jnp.asarray(radii),
            jnp.asarray(wavenumbers),
            jnp.asarray(synthesis),
            jnp.asarray(analysis),
            evidence,
            prepared_id=prepared_id,
        )


class PreparedSharedGridHankel(StrictModule, NonTrainableState):
    """Prepared quasi-cylindrical synthesis matrices and their pseudoinverses.

    ``synthesis[o, m]`` is the ``[radius, k]`` matrix of order ``m + o − 1``
    and ``analysis[o, m]`` its pseudoinverse; values are laid out
    ``[mode, radial, …]`` with the radial axis second.
    """

    plan: SharedGridHankelPlan
    radial_coordinates: Array
    wavenumbers: Array
    synthesis: Array
    analysis: Array
    evidence: SharedGridHankelEvidence
    prepared_id: str = eqx.field(static=True)

    def _matrix(self, stack: Array, offset: SharedGridHankelOffset, /) -> Array:
        match parse(offset, SharedGridHankelOffset, "offset"):
            case -1:
                return stack[0]
            case 0:
                return stack[1]
            case 1:
                return stack[2]
            case _:
                raise ValueError("offset is invalid.")

    def _check(self, values: Array, /) -> None:
        expected = (self.plan.mode_count, self.plan.radial_count)
        if values.ndim < 2 or values.shape[:2] != expected:
            raise ValueError(f"Hankel values must begin with shape {expected}.")

    def forward(self, values: ArrayLike, offset: SharedGridHankelOffset, /) -> Array:
        """Coefficients ``c[m, n, …]`` of order ``m + offset`` from ``f[m, j, …]``."""
        array = jnp.asarray(values)
        self._check(array)
        return ein.contract(
            "mkr,mr...->mk...", self._matrix(self.analysis, offset), array
        )

    def inverse(
        self, coefficients: ArrayLike, offset: SharedGridHankelOffset, /
    ) -> Array:
        """Samples ``f[m, j, …] = Σ_n c[m, n, …] J_{m+offset}(k_{m,n} r_j)``."""
        array = jnp.asarray(coefficients)
        self._check(array)
        return ein.contract(
            "mrk,mk...->mr...", self._matrix(self.synthesis, offset), array
        )


__all__ = [
    "CylindricalHankelEvidence",
    "CylindricalHankelPlan",
    "CylindricalHankelStatus",
    "PreparedCylindricalHankel",
    "PreparedSharedGridHankel",
    "SharedGridHankelEvidence",
    "SharedGridHankelOffset",
    "SharedGridHankelPlan",
    "prepare_cylindrical_hankel",
]
