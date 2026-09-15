#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-radius Bessel-zero Hankel transforms with explicit certification."""

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.special import jn_zeros

from ... import ein
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...special import jv


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
    ):
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
        roots = jnp.asarray(roots_host[:-1], dtype=float)
        boundary_root = jnp.asarray(roots_host[-1], dtype=float)
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


__all__ = [
    "CylindricalHankelEvidence",
    "CylindricalHankelPlan",
    "CylindricalHankelStatus",
    "PreparedCylindricalHankel",
    "prepare_cylindrical_hankel",
]
