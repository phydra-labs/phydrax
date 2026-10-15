#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native orthonormal Laplace translations and Cartesian displacement actions.

Write R_lm = r**l Y_lm and I_lm = R_lm/r**(2*l+1). Differentiating
our solid-harmonic recurrence gives

    d_z R_lm = sqrt((2*l+1)/(2*l-1) * (l*l-m*m)) R_(l-1,m).

Consequently R_lm(x+t*z) has coefficient

    C(l,k,m) = sqrt((2*l+1)/(2*k+1)
                     * (l+m)!(l-m)!/((k+m)!(k-m)!))/(l-k)!

on t**(l-k) R_km(x). Moments carry an additional 1/(2*l+1), so
M2M uses C(l,k,m)*(2*k+1)/(2*l+1); L2L uses C(k,l,m).
Expanding the leading transverse monomial of I_km(x+t*z), for t>0,
gives its R_lm coefficient

    (-1)**(l-|m|) sqrt((2*k+1)/(2*l+1)) (k+l)!
        / sqrt((k+m)!(k-m)!(l+m)!(l-m)!) / t**(k+l+1).

These formulas use only the package's complex Condon--Shortley convention.
No projection grid, fitted normalization, or finite-band derivative correction
is used. The active-ZYZ rotation owner supplies the degree blocks. Cartesian
JVPs use the exact translation generators for the two polynomial maps and
rotational covariance plus homogeneous radial scaling for M2L. Those rules
remain differentiable themselves, including at either pole and zero shift.
"""

from __future__ import annotations

import math
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .... import ein
from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral._spherical_algebra import (
    PreparedSphericalRotation,
    SphericalRotationPlan,
)
from ....discretization.spectral._spherical_layout import SphericalModeLayout


type _TranslationKind = Literal["m2m", "m2l", "l2l"]

# Qualified complex128 envelope. Bound both the degree program and inverse
# radius powers (up to 17), rather than silently overflowing at arbitrary order.
_MAXIMUM_ORDER = 8
_MINIMUM_M2L_RADIUS = 1.0e-8
_MAXIMUM_RADIUS = 1.0e8
_POLYNOMIAL_SHIFT_SCALE = 1.0e-6


@final
class _PreparedLaplaceRotation(StrictModule, NonTrainableState):
    rotation: PreparedSphericalRotation
    axial: Array
    degrees: Array
    orders: Array
    raising: Array
    lowering: Array
    moment_z: Array
    local_z: Array
    chart_frames: Array
    chart_rotations: Array
    table_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, layout: SphericalModeLayout, maximum_bytes: int, /) -> None:
        order = layout.bandlimit - 1
        if order > _MAXIMUM_ORDER:
            raise ValueError(
                "Rotation Laplace translations support expansion_order at most 8."
            )
        if layout.spin != 0 or layout.reality:
            raise ValueError(
                "Rotation Laplace translations require complex spin-zero coefficients."
            )
        limit, width = layout.coefficient_shape
        axial = np.zeros((3, limit, limit, width), dtype=np.float64)
        moment_z = np.zeros((limit, width), dtype=np.float64)
        local_z = np.zeros((limit, width), dtype=np.float64)
        degrees = np.arange(limit, dtype=np.float64)[:, None]
        orders = np.arange(-order, order + 1, dtype=np.float64)[None, :]
        raising = np.sqrt(np.maximum(0.0, (degrees + orders) * (degrees - orders + 1.0)))
        lowering = np.sqrt(np.maximum(0.0, (degrees - orders) * (degrees + orders + 1.0)))
        for ell in range(limit):
            for m in range(-ell, ell + 1):
                slot = order + m
                if ell > 0:
                    moment_z[ell, slot] = math.sqrt(
                        (2 * ell - 1) / (2 * ell + 1) * (ell * ell - m * m)
                    )
                if ell < order:
                    local_z[ell, slot] = math.sqrt(
                        (2 * ell + 3) / (2 * ell + 1) * ((ell + 1) ** 2 - m * m)
                    )
                for k in range(abs(m), limit):
                    if k <= ell:
                        ratio = (
                            math.factorial(ell + m)
                            * math.factorial(ell - m)
                            / (math.factorial(k + m) * math.factorial(k - m))
                        )
                        axial[0, ell, k, slot] = math.sqrt(
                            (2 * k + 1) / (2 * ell + 1) * ratio
                        ) / math.factorial(ell - k)
                    if k >= ell:
                        ratio = (
                            math.factorial(k + m)
                            * math.factorial(k - m)
                            / (math.factorial(ell + m) * math.factorial(ell - m))
                        )
                        axial[2, ell, k, slot] = math.sqrt(
                            (2 * k + 1) / (2 * ell + 1) * ratio
                        ) / math.factorial(k - ell)
                    denominator = math.sqrt(
                        math.factorial(k + m)
                        * math.factorial(k - m)
                        * math.factorial(ell + m)
                        * math.factorial(ell - m)
                    )
                    axial[1, ell, k, slot] = (
                        (-1.0) ** (ell - abs(m))
                        * math.sqrt((2 * k + 1) / (2 * ell + 1))
                        * math.factorial(k + ell)
                        / denominator
                    )
        tables = (axial, degrees, orders, raising, lowering, moment_z, local_z)
        chart_frames = np.asarray(
            (
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
                ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            ),
            dtype=np.float64,
        )
        table_bytes = (
            sum(table.nbytes for table in tables)
            + chart_frames.nbytes
            + 3 * limit * width * width * np.dtype(np.complex128).itemsize
        )
        # Bound two vector translations (primal and tangent), their Wigner
        # degree blocks and axial tables, and scalar-channel vector actions.
        # No degree-pair coefficient matrix is formed, in primal or AD.
        workspace_bytes = (
            2 * limit * width * width * 16
            + 2 * limit * limit * width * 8
            + 8 * limit * width * 16
        )
        if table_bytes + workspace_bytes > maximum_bytes:
            raise ValueError(
                "Rotation Laplace translation tables exceed maximum_coefficient_bytes."
            )
        self.rotation = SphericalRotationPlan(
            layout, maximum_matrix_bytes=maximum_bytes
        ).prepare()
        self.axial = jnp.asarray(axial, dtype=jnp.float64)
        self.degrees = jnp.asarray(degrees, dtype=jnp.float64)
        self.orders = jnp.asarray(orders, dtype=jnp.float64)
        self.raising = jnp.asarray(raising, dtype=jnp.float64)
        self.lowering = jnp.asarray(lowering, dtype=jnp.float64)
        self.moment_z = jnp.asarray(moment_z, dtype=jnp.float64)
        self.local_z = jnp.asarray(local_z, dtype=jnp.float64)
        self.chart_frames = jnp.asarray(chart_frames, dtype=jnp.float64)
        chart_angles = jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.0, np.pi / 2.0, 0.0),
                (-np.pi / 2.0, np.pi / 2.0, np.pi / 2.0),
            ),
            dtype=jnp.float64,
        )
        chart_rotations = self.rotation.wigner_d(chart_angles)
        valid = layout.valid_mask
        self.chart_rotations = jnp.where(
            valid[None, :, :, None] & valid[None, :, None, :],
            chart_rotations,
            0.0,
        )
        self.table_bytes = table_bytes
        self.workspace_bytes = workspace_bytes
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-laplace-rotation",
                "rotation": self.rotation.prepared_id,
                "maximum_order": _MAXIMUM_ORDER,
                "minimum_m2l_radius": _MINIMUM_M2L_RADIUS,
                "maximum_radius": _MAXIMUM_RADIUS,
                "table_bytes": table_bytes,
                "workspace_bytes": workspace_bytes,
                "normalization": "orthonormal-complex-condon-shortley",
                "rotation_chart": "least-polar-alignment",
                "polynomial_shift_scale": _POLYNOMIAL_SHIFT_SCALE,
            }
        )

    def angular_action(self, modal: Array, vector: Array, /) -> Array:
        """Apply the active-rotation generator -i vector dot J, degreewise."""
        zero = jnp.zeros_like(modal[:, :1])
        raised = self.raising[..., None] * jnp.concatenate((zero, modal[:, :-1]), axis=1)
        lowered = self.lowering[..., None] * jnp.concatenate((modal[:, 1:], zero), axis=1)
        return (
            -0.5j * vector[0] * (raised + lowered)
            + 0.5 * vector[1] * (lowered - raised)
            - 1j * vector[2] * self.orders[..., None] * modal
        )

    def z_action(self, modal: Array, kind: _TranslationKind, /) -> Array:
        zero = jnp.zeros_like(modal[:1])
        match kind:
            case "m2m":
                return self.moment_z[..., None] * jnp.concatenate(
                    (zero, modal[:-1]), axis=0
                )
            case "l2l":
                return self.local_z[..., None] * jnp.concatenate(
                    (modal[1:], zero), axis=0
                )
            case "m2l":
                raise ValueError(
                    "M2L does not have a finite polynomial translation generator."
                )
            case _:
                assert_never(kind)

    def polynomial_action(
        self, modal: Array, vector: Array, kind: _TranslationKind, /
    ) -> Array:
        """Analytic displacement derivative, including the zero-shift limit."""
        z = self.z_action(modal, kind)
        angular_vector = jnp.stack(
            (-vector[1], vector[0], jnp.zeros((), dtype=vector.dtype))
        )
        return (
            vector[2] * z
            + self.angular_action(z, angular_vector)
            - self.z_action(self.angular_action(modal, angular_vector), kind)
        )

    def rotated_apply(
        self, displacement: Array, modal: Array, kind: _TranslationKind, /
    ) -> Array:
        """Apply D (A (D^H x)) in a nonsingular active-ZYZ coordinate chart."""
        # Choose a fixed frame whose polar axis is least aligned with the shift.
        # Its Euler sine is at least sqrt(2/3) at every nonzero shift. This exact
        # coordinate atlas also keeps automatic coefficient/geometry mixed
        # transpose actions away from the Euler pole; it changes no operator.
        chart = jnp.argmin(jnp.abs(displacement[jnp.asarray((2, 0, 1), dtype=jnp.int32)]))
        framed = self.chart_frames[chart] @ displacement
        scale = jnp.max(jnp.abs(framed))
        direction = framed / scale
        log_radius = jnp.log(scale) + 0.5 * jnp.log(jnp.sum(direction * direction))
        transverse = jnp.hypot(direction[0], direction[1])
        alpha = jnp.arctan2(direction[1], direction[0])
        beta = jnp.arctan2(transverse, direction[2])
        angles = jnp.stack((alpha, beta, jnp.zeros((), dtype=displacement.dtype)))
        blocks = self.rotation.wigner_d(angles)
        valid = self.rotation.plan.layout.valid_mask
        blocks = jnp.where(valid[:, :, None] & valid[:, None, :], blocks, 0.0)
        frame_blocks = self.chart_rotations[chart]
        framed_modal = ein.contract("lmn,lnc->lmc", frame_blocks, modal, backend="jax")
        aligned = ein.contract(
            "lmn,lmc->lnc", jnp.conj(blocks), framed_modal, backend="jax"
        )
        ell = self.degrees[:, 0]
        match kind:
            case "m2m":
                table = self.axial[0]
                power = ell[:, None] - ell[None, :]
            case "m2l":
                table = self.axial[1]
                power = -ell[:, None] - ell[None, :] - 1.0
            case "l2l":
                table = self.axial[2]
                power = ell[None, :] - ell[:, None]
            case _:
                assert_never(kind)
        powers = jnp.where(jnp.any(table != 0.0, axis=-1), power, 0.0)
        axial = table * jnp.exp(log_radius * powers)[..., None]
        translated = ein.contract("lkm,kmc->lmc", axial, aligned, backend="jax")
        rotated = ein.contract("lmn,lnc->lmc", blocks, translated, backend="jax")
        return ein.contract(
            "lmn,lmc->lnc", jnp.conj(frame_blocks), rotated, backend="jax"
        )

    def polynomial_translate(
        self, displacement: Array, modal: Array, kind: _TranslationKind, /
    ) -> Array:
        """Exact finite exponential of the commuting nilpotent generators."""
        output = modal
        term = modal
        for degree in range(1, self.rotation.plan.layout.bandlimit):
            term = self.polynomial_action(term, displacement, kind) / degree
            output = output + term
        return output

    def raw_apply(
        self, displacement: Array, modal: Array, kind: _TranslationKind, /
    ) -> Array:
        match kind:
            case "m2l":
                return self.rotated_apply(displacement, modal, kind)
            case "m2m" | "l2l":
                # Both branches are the same retained polynomial operator.
                # The polynomial chart owns all zero-shift jets, including
                # geometry/coefficient mixed actions produced by transposition.
                # Keep the unused rotation branch finite under batched cond.
                safe = jnp.where(
                    jnp.any(displacement != 0.0),
                    displacement,
                    jnp.asarray((0.0, 0.0, 1.0), dtype=displacement.dtype),
                )
                return jax.lax.cond(
                    jnp.max(jnp.abs(displacement)) <= _POLYNOMIAL_SHIFT_SCALE,
                    lambda: self.polynomial_translate(displacement, modal, kind),
                    lambda: self.rotated_apply(safe, modal, kind),
                )
            case _:
                assert_never(kind)

    def apply(
        self, modal: Array, displacement: Array, kind: _TranslationKind, /
    ) -> Array:
        if jnp.iscomplexobj(displacement):
            raise TypeError("Rotation Laplace centers must be real.")
        shift = displacement.astype(jnp.float64)
        radius_squared = jnp.sum(shift * shift)
        shift = eqx.error_if(
            shift,
            jnp.any(~jnp.isfinite(shift)) | (radius_squared > _MAXIMUM_RADIUS**2),
            "Rotation Laplace displacement exceeds the finite radius envelope [0, 1e8].",
        )
        if kind == "m2l":
            shift = eqx.error_if(
                shift,
                radius_squared < _MINIMUM_M2L_RADIUS**2,
                "Rotation M2L requires noncoincident centers and radius at least 1e-8.",
            )
        payload_shape = modal.shape[2:]
        flattened = modal.reshape(modal.shape[:2] + (math.prod(payload_shape),))
        result = _translate(self, shift, flattened, kind)
        if kind != "m2l":
            # A zero shift is the identity operator. Complex GEMM realizations
            # (for example three-multiplication CUDA kernels) round even an
            # identity product, so return the input exactly while keeping the
            # contraction's tangent, i.e. the derivative of the same operator.
            identity = jnp.all(shift == 0.0)
            exact = jax.lax.stop_gradient(flattened) + (
                result - jax.lax.stop_gradient(result)
            )
            result = jnp.where(identity, exact, result)
        return result.reshape(modal.shape)


@eqx.filter_custom_jvp
def _translate(
    prepared: _PreparedLaplaceRotation,
    displacement: Array,
    modal: Array,
    kind: _TranslationKind,
    /,
) -> Array:
    return prepared.raw_apply(displacement, modal, kind)


@_translate.def_jvp
def _translate_jvp(
    primals: tuple[_PreparedLaplaceRotation, Array, Array, _TranslationKind],
    tangents: tuple[object, Array | None, Array | None, None],
    /,
) -> tuple[Array, Array]:
    prepared, shift, modal, kind = primals
    _, shift_tangent, modal_tangent, _ = tangents
    output = _translate(prepared, shift, modal, kind)
    tangent = jnp.zeros_like(output)
    if shift_tangent is None:
        if modal_tangent is not None:
            tangent = _translate(prepared, shift, modal_tangent, kind)
    else:
        match kind:
            case "m2m" | "l2l":
                tangent = prepared.polynomial_action(output, shift_tangent, kind)
                if modal_tangent is not None:
                    tangent = tangent + _translate(prepared, shift, modal_tangent, kind)
            case "m2l":
                radius_squared = jnp.sum(shift * shift)
                radial = jnp.sum(shift * shift_tangent) / radius_squared
                angular = jnp.cross(shift, shift_tangent) / radius_squared
                degree = prepared.degrees[..., None]
                incoming = -radial * degree * modal - prepared.angular_action(
                    modal, angular
                )
                if modal_tangent is not None:
                    incoming = incoming + modal_tangent
                tangent = (
                    -radial * (degree + 1.0) * output
                    + prepared.angular_action(output, angular)
                    + _translate(prepared, shift, incoming, kind)
                )
            case _:
                assert_never(kind)
    return output, tangent
