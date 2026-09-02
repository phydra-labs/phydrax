#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from functools import lru_cache
from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike
from s2fft.recursions.risbo_jax import compute_full as _wigner_small_d

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._spherical import SphericalSpectralDiscretization
from ._spherical_layout import SphericalModeLayout


@lru_cache(maxsize=None)
def _coupled_angular_momentum_block(
    degree_left: int,
    degree_right: int,
    total_order: int,
    /,
) -> tuple[tuple[int, ...], np.ndarray, tuple[int, ...], float]:
    """Stable bounded CG coefficients from the J² three-term coupling."""
    left_orders = tuple(
        order
        for order in range(-degree_left, degree_left + 1)
        if -degree_right <= total_order - order <= degree_right
    )
    count = len(left_orders)
    matrix = np.zeros((count, count), dtype=float)
    for index_, left_order in enumerate(left_orders):
        right_order = total_order - left_order
        matrix[index_, index_] = (
            degree_left * (degree_left + 1)
            + degree_right * (degree_right + 1)
            + 2 * left_order * right_order
        )
        if index_ + 1 < count:
            coupling = math.sqrt(
                (degree_left - left_order)
                * (degree_left + left_order + 1)
                * (degree_right + right_order)
                * (degree_right - right_order + 1)
            )
            matrix[index_, index_ + 1] = coupling
            matrix[index_ + 1, index_] = coupling
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    degrees = tuple(
        degree
        for degree in range(
            max(abs(degree_left - degree_right), abs(total_order)),
            degree_left + degree_right + 1,
        )
    )
    ordered = np.empty((count, len(degrees)), dtype=float)
    for column, degree in enumerate(degrees):
        eigen_index = int(np.argmin(np.abs(eigenvalues - degree * (degree + 1))))
        vector = eigenvectors[:, eigen_index]
        anchor = int(np.flatnonzero(np.abs(vector) > 64 * np.finfo(float).eps)[-1])
        ordered[:, column] = vector * (1.0 if vector[anchor] >= 0 else -1.0)
    residual = np.max(
        np.abs(
            matrix @ ordered
            - ordered * np.asarray([degree * (degree + 1) for degree in degrees])[None, :]
        ),
        initial=0.0,
    )
    if residual > 1e-10 * max(1, degree_left + degree_right) ** 2:
        raise ValueError("Clebsch-Gordan three-term recurrence certification failed.")
    return left_orders, ordered, degrees, float(residual)


def _stable_clebsch_gordan(
    degree_left: int,
    degree_right: int,
    output_degree: int,
    left_order: int,
    right_order: int,
    /,
) -> float:
    total_order = left_order + right_order
    orders, coefficients, degrees, _ = _coupled_angular_momentum_block(
        degree_left, degree_right, total_order
    )
    if output_degree not in degrees or left_order not in orders:
        return 0.0
    return float(coefficients[orders.index(left_order), degrees.index(output_degree)])


def _layout(value, name: str, /) -> SphericalModeLayout:
    if isinstance(value, SphericalModeLayout):
        return value
    if isinstance(value, SphericalSpectralDiscretization):
        return value.layout
    raise TypeError(f"{name} must be a spherical layout or discretization.")


class SphericalRotationPlan(StrictModule, NonTrainableState):
    """Active ZYZ rotations of one fixed spherical coefficient layout."""

    layout: SphericalModeLayout
    maximum_matrix_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: SphericalModeLayout | SphericalSpectralDiscretization,
        /,
        *,
        maximum_matrix_bytes: int = 512 * 1024**2,
    ):
        layout_ = _layout(layout, "layout")
        if isinstance(maximum_matrix_bytes, bool):
            raise TypeError("maximum_matrix_bytes must be an integer.")
        maximum = index(maximum_matrix_bytes)
        if maximum <= 0:
            raise ValueError("maximum_matrix_bytes must be positive.")
        required = (
            layout_.bandlimit
            * (2 * layout_.bandlimit - 1) ** 2
            * np.dtype(np.complex128).itemsize
        )
        if required > maximum:
            raise ValueError("Spherical rotation blocks exceed maximum_matrix_bytes.")
        self.layout = layout_
        self.maximum_matrix_bytes = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spherical-rotation-plan",
                "layout": layout_.layout_id,
                "maximum_matrix_bytes": maximum,
            }
        )

    def prepare(self, /) -> "PreparedSphericalRotation":
        return PreparedSphericalRotation(self)


class PreparedSphericalRotation(StrictModule, NonTrainableState):
    plan: SphericalRotationPlan
    prepared_id: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)

    def __init__(self, plan: SphericalRotationPlan, /):
        if not isinstance(plan, SphericalRotationPlan):
            raise TypeError("plan must be a SphericalRotationPlan.")
        self.plan = plan
        self.convention = "active-ZYZ"
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spherical-rotation",
                "plan": plan.plan_id,
                "convention": self.convention,
            }
        )

    def wigner_d(self, euler_angles: ArrayLike, /) -> Array:
        """Return padded full Wigner-D degree blocks for active ZYZ angles."""
        angles = jnp.asarray(euler_angles)
        if angles.ndim < 1 or angles.shape[-1] != 3:
            raise ValueError("Euler angles must end with shape (3,).")
        if jnp.iscomplexobj(angles):
            raise TypeError("Euler angles must be real.")
        if not jnp.issubdtype(angles.dtype, jnp.inexact):
            angles = angles.astype(float)
        angles = angles.astype(jnp.result_type(angles.dtype, jnp.float64))
        angles = eqx.error_if(
            angles,
            jnp.any(~jnp.isfinite(angles)),
            "Euler angles must be finite.",
        )
        batch_shape = angles.shape[:-1]
        flattened = angles.reshape((-1, 3))
        limit = self.plan.layout.bandlimit
        orders = jnp.arange(-(limit - 1), limit, dtype=angles.dtype)

        def one_rotation(angle):
            alpha, beta, gamma = angle
            plane = jnp.zeros((2 * limit - 1, 2 * limit - 1), dtype=angles.dtype)
            blocks = []
            for degree in range(limit):
                plane = _wigner_small_d(plane, beta, limit, degree)
                left = jnp.exp(-1j * orders * alpha)
                right = jnp.exp(-1j * orders * gamma)
                blocks.append(
                    plane.astype(jnp.result_type(angles.dtype, 1j))
                    * left[:, None]
                    * right[None, :]
                )
            return jnp.stack(tuple(blocks), axis=0)

        matrices = jax.vmap(one_rotation)(flattened)
        return matrices.reshape(batch_shape + (limit, 2 * limit - 1, 2 * limit - 1))

    def apply(self, coefficients: ArrayLike, euler_angles: ArrayLike, /) -> Array:
        layout = self.plan.layout
        modal = layout.mask_invalid(coefficients)
        _, _, channel_last = layout._coefficient_axes(modal)
        if modal.ndim not in (2, 3) or (modal.ndim == 3 and not channel_last):
            raise ValueError(
                "Spherical rotation currently accepts one field with optional channels."
            )
        payload = modal[..., None] if modal.ndim == 2 else modal
        matrices = self.wigner_d(euler_angles)
        angle_shape = matrices.shape[:-3]
        flattened_matrices = matrices.reshape(
            (-1, layout.bandlimit, 2 * layout.bandlimit - 1, 2 * layout.bandlimit - 1)
        )
        output = jnp.zeros(
            (
                flattened_matrices.shape[0],
                layout.bandlimit,
                2 * layout.bandlimit - 1,
                payload.shape[-1],
            ),
            dtype=jnp.result_type(modal.dtype, matrices.dtype),
        )
        offset = layout.bandlimit - 1
        for degree in range(layout.bandlimit):
            active = slice(offset - degree, offset + degree + 1)
            block = flattened_matrices[:, degree, active, active]
            values = payload[degree, active, :]
            rotated = oe.contract("bmn,nc->bmc", block, values, backend="jax")
            output = output.at[:, degree, active, :].set(rotated)
        output = output.reshape(
            angle_shape + layout.coefficient_shape + (payload.shape[-1],)
        )
        if modal.ndim == 2:
            output = output[..., 0]
        output = layout.mask_invalid(output)
        if layout.reality:
            output = layout.canonicalize_reality(output)
        return output


class SphericalClebschGordanPlan(StrictModule, NonTrainableState):
    """Fixed-capacity coefficient product under spherical mode coupling."""

    left_layout: SphericalModeLayout
    right_layout: SphericalModeLayout
    output_layout: SphericalModeLayout
    maximum_couplings: int = eqx.field(static=True)
    maximum_coefficient_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_layout: SphericalModeLayout | SphericalSpectralDiscretization,
        right_layout: SphericalModeLayout | SphericalSpectralDiscretization,
        /,
        *,
        output_bandlimit: int | None = None,
        maximum_couplings: int = 10_000_000,
        maximum_coefficient_bytes: int = 1024 * 1024**2,
    ):
        left = _layout(left_layout, "left_layout")
        right = _layout(right_layout, "right_layout")
        output_limit = (
            left.bandlimit + right.bandlimit - 1
            if output_bandlimit is None
            else int(output_bandlimit)
        )
        output_spin = left.spin + right.spin
        if output_limit <= abs(output_spin):
            raise ValueError("output_bandlimit must exceed the product spin magnitude.")
        output = SphericalModeLayout(
            output_limit,
            spin=output_spin,
            reality=left.reality and right.reality and output_spin == 0,
        )
        couplings = index(maximum_couplings)
        coefficient_bytes = index(maximum_coefficient_bytes)
        if couplings <= 0 or coefficient_bytes <= 0:
            raise ValueError("CG coupling and byte limits must be positive.")
        self.left_layout = left
        self.right_layout = right
        self.output_layout = output
        self.maximum_couplings = couplings
        self.maximum_coefficient_bytes = coefficient_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spherical-clebsch-gordan-plan",
                "left": left.layout_id,
                "right": right.layout_id,
                "output": output.layout_id,
                "maximum_couplings": couplings,
                "maximum_coefficient_bytes": coefficient_bytes,
            }
        )

    def prepare(self, /) -> "PreparedSphericalClebschGordan":
        return PreparedSphericalClebschGordan(self)


class SphericalClebschGordanReport(StrictModule, NonTrainableState):
    coupling_count: int = eqx.field(static=True)
    coefficient_bytes: int = eqx.field(static=True)
    recurrence_residual: float = eqx.field(static=True)
    output_layout_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class PreparedSphericalClebschGordan(StrictModule, NonTrainableState):
    plan: SphericalClebschGordanPlan
    left_indices: Array
    right_indices: Array
    output_indices: Array
    weights: Array
    report: SphericalClebschGordanReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SphericalClebschGordanPlan, /):
        if not isinstance(plan, SphericalClebschGordanPlan):
            raise TypeError("plan must be a SphericalClebschGordanPlan.")
        left = plan.left_layout
        right = plan.right_layout
        output = plan.output_layout
        left_indices = []
        right_indices = []
        output_indices = []
        weights = []
        left_offset = left.bandlimit - 1
        right_offset = right.bandlimit - 1
        output_offset = output.bandlimit - 1
        maximum_recurrence_residual = 0.0
        for l1 in range(abs(left.spin), left.bandlimit):
            for m1 in range(-l1, l1 + 1):
                for l2 in range(abs(right.spin), right.bandlimit):
                    for m2 in range(-l2, l2 + 1):
                        order = m1 + m2
                        lower = max(abs(output.spin), abs(l1 - l2), abs(order))
                        upper = min(output.bandlimit - 1, l1 + l2)
                        for degree in range(lower, upper + 1):
                            spin_cg = _stable_clebsch_gordan(
                                l1,
                                l2,
                                degree,
                                -left.spin,
                                -right.spin,
                            )
                            mode_cg = _stable_clebsch_gordan(l1, l2, degree, m1, m2)
                            maximum_recurrence_residual = max(
                                maximum_recurrence_residual,
                                _coupled_angular_momentum_block(l1, l2, -output.spin)[3],
                                _coupled_angular_momentum_block(l1, l2, order)[3],
                            )
                            coefficient = (
                                spin_cg
                                * mode_cg
                                * math.sqrt(
                                    (2 * l1 + 1)
                                    * (2 * l2 + 1)
                                    / (4.0 * math.pi * (2 * degree + 1))
                                )
                            )
                            if coefficient == 0.0:
                                continue
                            left_indices.append(
                                l1 * (2 * left.bandlimit - 1) + m1 + left_offset
                            )
                            right_indices.append(
                                l2 * (2 * right.bandlimit - 1) + m2 + right_offset
                            )
                            output_indices.append(
                                degree * (2 * output.bandlimit - 1)
                                + order
                                + output_offset
                            )
                            weights.append(coefficient)
                            if len(weights) > plan.maximum_couplings:
                                raise ValueError(
                                    "Spherical CG action exceeds maximum_couplings."
                                )
        coefficient_bytes = len(weights) * (
            3 * np.dtype(np.int32).itemsize + np.dtype(np.float64).itemsize
        )
        if coefficient_bytes > plan.maximum_coefficient_bytes:
            raise ValueError(
                "Spherical CG coefficients exceed maximum_coefficient_bytes."
            )
        report_id = canonical_fingerprint(
            {
                "kind": "spherical-clebsch-gordan-report",
                "plan": plan.plan_id,
                "coupling_count": len(weights),
                "coefficient_bytes": coefficient_bytes,
                "recurrence_residual": maximum_recurrence_residual,
                "output_layout": output.layout_id,
            }
        )
        self.plan = plan
        self.left_indices = jnp.asarray(left_indices, dtype=jnp.int32)
        self.right_indices = jnp.asarray(right_indices, dtype=jnp.int32)
        self.output_indices = jnp.asarray(output_indices, dtype=jnp.int32)
        self.weights = jnp.asarray(weights, dtype=jnp.float64)
        self.report = SphericalClebschGordanReport(
            coupling_count=len(weights),
            coefficient_bytes=coefficient_bytes,
            recurrence_residual=maximum_recurrence_residual,
            output_layout_id=output.layout_id,
            report_id=report_id,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spherical-clebsch-gordan",
                "plan": plan.plan_id,
                "report": report_id,
            }
        )

    def apply(
        self, left_coefficients: ArrayLike, right_coefficients: ArrayLike, /
    ) -> Array:
        left_layout = self.plan.left_layout
        right_layout = self.plan.right_layout
        output_layout = self.plan.output_layout
        left = left_layout.mask_invalid(left_coefficients)
        right = right_layout.mask_invalid(right_coefficients)
        if (
            left.shape != left_layout.coefficient_shape
            or right.shape != right_layout.coefficient_shape
        ):
            raise ValueError("Spherical CG action currently accepts scalar modal fields.")
        left_flat = left.reshape((-1,))
        right_flat = right.reshape((-1,))
        coupled = oe.contract(
            "k,k,k->k",
            self.weights.astype(jnp.result_type(left.dtype, right.dtype)),
            left_flat[self.left_indices],
            right_flat[self.right_indices],
            backend="jax",
        )
        output = jnp.zeros(
            (math.prod(output_layout.coefficient_shape),),
            dtype=jnp.result_type(left.dtype, right.dtype),
        )
        output = output.at[self.output_indices].add(coupled)
        output = output_layout.mask_invalid(
            output.reshape(output_layout.coefficient_shape)
        )
        if output_layout.reality:
            output = output_layout.canonicalize_reality(output)
        return output


__all__ = [
    "PreparedSphericalClebschGordan",
    "PreparedSphericalRotation",
    "SphericalClebschGordanPlan",
    "SphericalClebschGordanReport",
    "SphericalRotationPlan",
]
