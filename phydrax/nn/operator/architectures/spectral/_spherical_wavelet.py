# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ....._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....._strict import StrictModule
from ....._trainable import NonTrainableState
from .....discretization.spectral import (
    prepare_spectral_modal_transfer,
    SphericalRotationPlan,
    SphericalSpectralDiscretization,
)


class DirectionalSphericalWaveletPlan(StrictModule, NonTrainableState):
    """Finite directional spherical scales, orientations, and resource envelope.

    Recursive orders use the declared weighted Wigner ``n=0`` analysis frame to
    form each scalar S2 modulus intermediate.  This is a bounded finite-frame
    projection, not a claim of an exact continuous SO(3) inverse.
    """

    discretization: SphericalSpectralDiscretization
    orientations: Array
    orientation_weights: Array
    wigner_matrices: Array
    orientation_analysis: Array
    path_indices: Array
    path_scales: Array
    path_mask: Array
    analysis_frame_lower_bound: Array
    analysis_frame_upper_bound: Array
    rotation: object
    transfer: object
    scales: tuple[int, ...] = eqx.field(static=True)
    azimuthal_bandlimit: int = eqx.field(static=True)
    scattering_order: int = eqx.field(static=True)
    path_capacity: int = eqx.field(static=True)
    path_admissibility: str = eqx.field(static=True)
    recursive_projection: str = eqx.field(static=True)
    maximum_materialization_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: SphericalSpectralDiscretization,
        orientations: ArrayLike,
        /,
        *,
        scales: Sequence[int],
        azimuthal_bandlimit: int,
        scattering_order: int = 1,
        orientation_weights: ArrayLike | None = None,
        maximum_materialization_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(discretization, SphericalSpectralDiscretization):
            raise TypeError("discretization must be SphericalSpectralDiscretization.")
        angles = jnp.asarray(orientations, dtype=float)
        if angles.ndim != 2 or angles.shape[-1] != 3 or int(angles.shape[0]) == 0:
            raise ValueError("orientations must have shape (orientation, 3).")
        if bool(jnp.any(~jnp.isfinite(angles))):
            raise ValueError("orientations must be finite.")
        scale_values = tuple(int(value) for value in scales)
        if not scale_values or any(value < 0 for value in scale_values):
            raise ValueError("scales must be a nonempty tuple of nonnegative indices.")
        azimuthal = int(azimuthal_bandlimit)
        order = int(scattering_order)
        if azimuthal <= 0 or azimuthal > discretization.layout.bandlimit:
            raise ValueError("azimuthal_bandlimit exceeds the finite spherical layout.")
        if order < 1:
            raise ValueError("scattering_order must be positive.")
        if order > 1 and discretization.layout.spin != 0:
            raise ValueError("Recursive spherical modulus scattering requires spin zero.")
        if orientation_weights is None:
            weights = jnp.full((angles.shape[0],), 1.0 / angles.shape[0])
        else:
            weights = jnp.asarray(orientation_weights, dtype=float)
            if weights.shape != (angles.shape[0],):
                raise ValueError("orientation_weights must match the orientation count.")
            if (
                bool(jnp.any(~jnp.isfinite(weights)))
                or bool(jnp.any(weights < 0.0))
                or not bool(jnp.sum(weights) > 0.0)
            ):
                raise ValueError("orientation_weights must be finite and nonnegative.")
            weights = weights / jnp.sum(weights)

        scale_count = len(scale_values)
        capacity = scale_count**order
        maximum = int(maximum_materialization_bytes)
        limit = discretization.layout.bandlimit
        order_count = 2 * limit - 1
        wigner_bytes = (
            int(angles.shape[0])
            * limit
            * order_count**2
            * jnp.dtype(jnp.complex128).itemsize
        )
        analysis_bytes = (
            int(angles.shape[0])
            * limit
            * order_count
            * jnp.dtype(jnp.complex128).itemsize
        )
        path_bytes = (
            order
            * capacity
            * (2 * order * jnp.dtype(jnp.int32).itemsize + jnp.dtype(jnp.bool_).itemsize)
        )
        recursive_bytes = capacity * (
            int(angles.shape[0]) * jnp.dtype(jnp.complex128).itemsize
            + limit * order_count * jnp.dtype(jnp.complex128).itemsize
            + order * jnp.dtype(jnp.float64).itemsize
        )
        required = wigner_bytes + analysis_bytes + path_bytes + recursive_bytes
        if maximum <= 0 or required > maximum:
            raise ValueError(
                "Directional scattering materialization exceeds the byte cap."
            )

        index_rows = []
        scale_rows = []
        mask_rows = []
        for depth in range(1, order + 1):
            candidates = tuple(product(range(scale_count), repeat=depth))
            padding = capacity - len(candidates)
            index_rows.append(
                tuple(candidate + (-1,) * (order - depth) for candidate in candidates)
                + ((-1,) * order,) * padding
            )
            scale_rows.append(
                tuple(
                    tuple(scale_values[index] for index in candidate)
                    + (-1,) * (order - depth)
                    for candidate in candidates
                )
                + ((-1,) * order,) * padding
            )
            mask_rows.append(
                tuple(
                    all(
                        scale_values[left] < scale_values[right]
                        for left, right in zip(candidate[:-1], candidate[1:], strict=True)
                    )
                    for candidate in candidates
                )
                + (False,) * padding
            )
        path_indices = jnp.asarray(index_rows, dtype=jnp.int32)
        path_scales = jnp.asarray(scale_rows, dtype=jnp.int32)
        path_mask = jnp.asarray(mask_rows, dtype=bool)

        rotation_plan = SphericalRotationPlan(
            discretization.layout,
            maximum_matrix_bytes=maximum,
        )
        rotation = rotation_plan.prepare()
        transfer = prepare_spectral_modal_transfer(discretization, discretization)
        wigner_matrices = rotation.wigner_d(angles)
        scalar_column = wigner_matrices[..., discretization.layout.bandlimit - 1]
        degree = jnp.arange(limit, dtype=angles.dtype)
        evaluation_normalization = jnp.sqrt((2.0 * degree + 1.0) / (4.0 * jnp.pi))
        evaluation = jnp.conj(scalar_column) * evaluation_normalization[None, :, None]
        analysis_normalization = jnp.sqrt(4.0 * jnp.pi * (2.0 * degree + 1.0))
        orientation_analysis = (
            scalar_column * weights[:, None, None] * analysis_normalization[None, :, None]
        )
        valid_evaluation = evaluation.reshape((angles.shape[0], -1))[
            :, discretization.layout.valid_mask.reshape((-1,))
        ]
        weighted_evaluation = jnp.sqrt(weights)[:, None] * valid_evaluation
        singular_values = jnp.linalg.svd(weighted_evaluation, compute_uv=False)
        upper_bound = singular_values[0] ** 2
        full_column_count = discretization.layout.logical_mode_count
        lower_bound = (
            singular_values[-1] ** 2
            if int(angles.shape[0]) >= full_column_count
            else jnp.zeros((), dtype=upper_bound.dtype)
        )
        rank_tolerance = (
            jnp.finfo(singular_values.dtype).eps
            * max(weighted_evaluation.shape)
            * singular_values[0]
        )
        if order > 1 and (
            int(angles.shape[0]) < full_column_count
            or not bool(singular_values[-1] > rank_tolerance)
        ):
            raise ValueError(
                "Recursive orientation samples and weights must form a full-rank "
                "weighted Wigner n=0 analysis frame."
            )
        self.discretization = discretization
        self.orientations = angles
        self.orientation_weights = weights
        self.wigner_matrices = wigner_matrices
        self.orientation_analysis = orientation_analysis
        self.path_indices = path_indices
        self.path_scales = path_scales
        self.path_mask = path_mask
        self.analysis_frame_lower_bound = lower_bound
        self.analysis_frame_upper_bound = upper_bound
        self.rotation = rotation
        self.transfer = transfer
        self.scales = scale_values
        self.azimuthal_bandlimit = azimuthal
        self.scattering_order = order
        self.path_capacity = capacity
        self.path_admissibility = "strictly-increasing-scale"
        self.recursive_projection = "weighted-Wigner-n0-scalar-S2"
        self.maximum_materialization_bytes = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "directional-spherical-wavelet-plan",
                "discretization": discretization.discretization_id,
                "rotation": rotation.prepared_id,
                "transfer": transfer.prepared_id,
                "orientations": array_tree_fingerprint(angles),
                "orientation_weights": array_tree_fingerprint(weights),
                "scales": list(scale_values),
                "azimuthal_bandlimit": azimuthal,
                "scattering_order": order,
                "path_admissibility": self.path_admissibility,
                "recursive_projection": self.recursive_projection,
                "path_capacity": capacity,
                "maximum_materialization_bytes": maximum,
            }
        )


class DirectionalSphericalWaveletLayer(StrictModule):
    """Learned finite directional wavelet contraction over prepared Wigner blocks."""

    plan: DirectionalSphericalWaveletPlan
    wavelets: Array

    def __init__(
        self,
        plan: DirectionalSphericalWaveletPlan,
        wavelets: ArrayLike,
        /,
    ):
        if not isinstance(plan, DirectionalSphericalWaveletPlan):
            raise TypeError("plan must be DirectionalSphericalWaveletPlan.")
        filters = jnp.asarray(wavelets)
        expected = (len(plan.scales),) + plan.discretization.layout.coefficient_shape
        if filters.shape != expected:
            raise ValueError(f"wavelets must have shape {expected}.")
        if not jnp.issubdtype(filters.dtype, jnp.inexact):
            raise TypeError("wavelets must have an inexact dtype.")
        order = plan.discretization.layout.orders
        filters = jnp.where(
            jnp.abs(order)[None, ...] < plan.azimuthal_bandlimit, filters, 0.0
        )
        filters = plan.discretization.layout.mask_invalid(filters)
        if bool(jnp.any(~jnp.isfinite(filters))):
            raise ValueError("wavelets must be finite on active modes.")
        self.plan = plan
        self.wavelets = filters

    def __call__(self, coefficients: ArrayLike, /) -> Array:
        modal = self.plan.transfer(coefficients)
        layout = self.plan.discretization.layout
        payload_shape = modal.shape[2:]
        payload = modal.reshape(layout.coefficient_shape + (-1,))
        payload = layout.mask_invalid(payload)
        payload = eqx.error_if(
            payload,
            jnp.any(~jnp.isfinite(payload)),
            "Spherical wavelet coefficients must be finite on active modes.",
        )
        result = contract(
            "lmc,jln,olmn->joc",
            payload,
            jnp.conj(self.wavelets),
            self.plan.wigner_matrices,
        )
        if not payload_shape:
            return result[..., 0]
        return result.reshape(
            (
                len(self.plan.scales),
                int(self.plan.orientations.shape[0]),
            )
            + payload_shape
        )

    def _project_orientation_payload(self, payload: Array, /) -> Array:
        """Apply the prepared weighted Wigner n=0 scalar S2 analysis."""
        orientation_count = int(self.plan.orientations.shape[0])
        if payload.ndim < 2 or payload.shape[-2] != orientation_count:
            raise ValueError(
                "Directional samples must end in (orientation, channel) with "
                f"orientation count {orientation_count}."
            )
        values = jnp.asarray(payload)
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Directional samples must be finite.",
        )
        prefix_shape = values.shape[:-2]
        flattened = values.reshape((-1, orientation_count, values.shape[-1]))
        layout = self.plan.discretization.layout
        projected = contract(
            "qoc,olm->lmqc",
            flattened,
            self.plan.orientation_analysis,
        )
        projected = projected.reshape(
            layout.coefficient_shape + prefix_shape + (values.shape[-1],)
        )
        flattened_projected = projected.reshape(layout.coefficient_shape + (-1,))
        flattened_projected = layout.mask_invalid(flattened_projected)
        if layout.reality:
            flattened_projected = layout.canonicalize_reality(flattened_projected)
        return flattened_projected.reshape(projected.shape)

    def project_orientation_samples(self, samples: ArrayLike, /) -> Array:
        """Project one declared orientation sample field to spherical coefficients."""
        values = jnp.asarray(samples)
        orientation_count = int(self.plan.orientations.shape[0])
        if values.ndim >= 1 and values.shape[-1] == orientation_count:
            return self._project_orientation_payload(values[..., None])[..., 0]
        if values.ndim >= 2 and values.shape[-2] == orientation_count:
            return self._project_orientation_payload(values)
        raise ValueError(
            "Directional samples must end in (orientation,) or "
            f"(orientation, channels), with orientation count {orientation_count}."
        )


class SphericalWaveletScattering(StrictModule):
    """Recursive finite-frame modulus scattering with invariant path reduction.

    The bounded path axis has ``len(scales) ** scattering_order`` slots.  Row
    ``k`` uses the first ``len(scales) ** (k + 1)`` slots in lexicographic path
    order; :attr:`DirectionalSphericalWaveletPlan.path_mask` identifies the
    strictly increasing scale paths and all other slots remain exactly zero.
    """

    layer: DirectionalSphericalWaveletLayer

    def __init__(self, layer: DirectionalSphericalWaveletLayer, /):
        if not isinstance(layer, DirectionalSphericalWaveletLayer):
            raise TypeError("layer must be DirectionalSphericalWaveletLayer.")
        self.layer = layer

    def __call__(self, coefficients: ArrayLike, /) -> Array:
        layout = self.layer.plan.discretization.layout
        modal = jnp.asarray(coefficients)
        _, _, channel_last = layout._coefficient_axes(modal)
        directional = jnp.abs(self.layer(coefficients))
        current = directional if channel_last else directional[..., None]
        scale_count = len(self.layer.plan.scales)
        orientation_count = int(self.layer.plan.orientations.shape[0])
        capacity = self.layer.plan.path_capacity
        weights = self.layer.plan.orientation_weights
        features = []

        for depth in range(self.layer.plan.scattering_order):
            candidate_count = scale_count ** (depth + 1)
            active = self.layer.plan.path_mask[depth, :candidate_count]
            current = jnp.where(active[:, None, None], current, 0.0)
            reduced = contract("poc,o->pc", current, weights)
            reduced = jnp.pad(reduced, ((0, capacity - candidate_count), (0, 0)))
            features.append(reduced if channel_last else reduced[..., 0])

            if depth + 1 < self.layer.plan.scattering_order:
                projected = self.layer._project_orientation_payload(current)
                projected = projected.reshape(layout.coefficient_shape + (-1,))
                transformed = jnp.abs(self.layer(projected))
                transformed = transformed.reshape(
                    (
                        scale_count,
                        orientation_count,
                        candidate_count,
                        current.shape[-1],
                    )
                )
                transformed = jnp.moveaxis(transformed, 2, 0)
                current = transformed.reshape(
                    (candidate_count * scale_count, orientation_count, current.shape[-1])
                )

        return jnp.stack(tuple(features), axis=-2)


__all__ = [
    "DirectionalSphericalWaveletLayer",
    "DirectionalSphericalWaveletPlan",
    "SphericalWaveletScattering",
]
