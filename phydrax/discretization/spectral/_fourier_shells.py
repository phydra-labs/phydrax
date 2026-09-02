#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


DCPolicy = Literal["exclude", "include"]
NyquistPolicy = Literal["include", "exclude"]
FinalEdgePolicy = Literal["include", "exclude"]


class ModeTransferCorrection(StrictModule, NonTrainableState):
    multiplier: Array
    source_operator_id: str = eqx.field(static=True)
    minimum_transfer_magnitude: float = eqx.field(static=True)
    correction_id: str = eqx.field(static=True)

    def __init__(
        self,
        multiplier: ArrayLike,
        source_operator_id: str,
        /,
        *,
        minimum_transfer_magnitude: float,
    ):
        values = jax.lax.stop_gradient(jnp.asarray(multiplier))
        source = str(source_operator_id).strip()
        minimum = float(minimum_transfer_magnitude)
        if (
            values.ndim < 1
            or jnp.iscomplexobj(values)
            or not source
            or not np.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("Mode-transfer correction is invalid.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(values < 0.0),
            "Mode-transfer multiplier must be finite and non-negative.",
        )
        self.multiplier = values
        self.source_operator_id = source
        self.minimum_transfer_magnitude = minimum
        self.correction_id = canonical_fingerprint(
            {
                "kind": "mode-transfer-correction",
                "source_operator_id": source,
                "minimum_transfer_magnitude": minimum,
                "multiplier": array_tree_fingerprint(values),
            }
        )


class PeriodicFourierField(StrictModule):
    coefficients: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class FourierShellStatisticResult(StrictModule):
    representative_wavenumbers: Array
    bin_edges: Array
    shell_values: Array
    weighted_mode_count: Array
    stored_mode_count: Array
    valid_shells: Array
    imaginary_residual: Array
    total_weighted_value: Array
    excluded_mode_count: Array
    finite: Array
    successful: Array
    statistic_kind: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    correction_id: str = eqx.field(static=True)


class _FourierShellBinGeometry(StrictModule, NonTrainableState):
    """Shared prepared shell bins with explicit per-mode reduction weights."""

    source_shape: tuple[int, ...] = eqx.field(static=True)
    bin_count: int = eqx.field(static=True)
    bin_edges: Array
    bin_widths: Array
    wavenumber_magnitude: Array
    shell_indices: Array
    mode_weights: Array
    weighted_mode_count: Array
    stored_mode_count: Array
    representative_wavenumbers: Array
    valid_shells: Array
    valid_modes: Array
    excluded_mode_count: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavenumber_magnitude: ArrayLike,
        bin_edges: ArrayLike,
        /,
        *,
        mode_mask: ArrayLike | None = None,
        mode_weights: ArrayLike | None = None,
        final_edge_policy: FinalEdgePolicy = "include",
        source_id: str,
    ):
        magnitude = np.asarray(wavenumber_magnitude, dtype=float)
        edges = np.asarray(bin_edges, dtype=float).reshape((-1,))
        mask = (
            np.ones(magnitude.shape, dtype=bool)
            if mode_mask is None
            else np.asarray(mode_mask, dtype=bool)
        )
        weights = (
            np.ones(magnitude.shape, dtype=float)
            if mode_weights is None
            else np.asarray(mode_weights, dtype=float)
        )
        source = str(source_id).strip()
        if (
            magnitude.ndim < 1
            or mask.shape != magnitude.shape
            or weights.shape != magnitude.shape
            or np.any(~np.isfinite(magnitude))
            or np.any(magnitude < 0.0)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
            or final_edge_policy not in ("include", "exclude")
            or not source
        ):
            raise ValueError("Fourier shell-bin geometry is invalid.")
        indices = np.searchsorted(edges, magnitude, side="right") - 1
        if final_edge_policy == "include":
            indices[np.isclose(magnitude, edges[-1], rtol=1.0e-12, atol=1.0e-14)] = (
                edges.size - 2
            )
        valid = mask & (indices >= 0) & (indices < edges.size - 1)
        safe_indices = np.where(valid, indices, 0)
        effective_weights = np.where(valid, weights, 0.0)
        weighted_count = np.bincount(
            safe_indices.reshape((-1,)),
            weights=effective_weights.reshape((-1,)),
            minlength=edges.size - 1,
        )
        stored_count = np.bincount(
            safe_indices.reshape((-1,)),
            weights=valid.reshape((-1,)).astype(float),
            minlength=edges.size - 1,
        )
        weighted_k = np.bincount(
            safe_indices.reshape((-1,)),
            weights=(effective_weights * magnitude).reshape((-1,)),
            minlength=edges.size - 1,
        )
        representative = np.divide(
            weighted_k,
            weighted_count,
            out=0.5 * (edges[:-1] + edges[1:]),
            where=weighted_count > 0.0,
        )
        self.source_shape = magnitude.shape
        self.bin_count = edges.size - 1
        self.bin_edges = jnp.asarray(edges)
        self.bin_widths = jnp.asarray(np.diff(edges))
        self.wavenumber_magnitude = jnp.asarray(magnitude)
        self.shell_indices = jnp.asarray(safe_indices, dtype=jnp.int32)
        self.mode_weights = jnp.asarray(effective_weights)
        self.weighted_mode_count = jnp.asarray(weighted_count)
        self.stored_mode_count = jnp.asarray(stored_count)
        self.representative_wavenumbers = jnp.asarray(representative)
        self.valid_shells = jnp.asarray(weighted_count > 0.0)
        self.valid_modes = jnp.asarray(valid)
        self.excluded_mode_count = int(np.size(valid) - np.count_nonzero(valid))
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "fourier-shell-bin-geometry",
                "source_id": source,
                "wavenumber_magnitude": array_tree_fingerprint(magnitude),
                "bin_edges": edges.tolist(),
                "mode_mask": array_tree_fingerprint(valid),
                "mode_weights": array_tree_fingerprint(effective_weights),
                "final_edge_policy": final_edge_policy,
            }
        )

    def reduce_integral(self, mode_values: ArrayLike, /) -> Array:
        values = jnp.asarray(mode_values)
        if values.shape != self.source_shape:
            raise ValueError("Mode statistic does not match shell-bin geometry.")
        weighted = values * self.mode_weights.astype(values.real.dtype)
        return (
            jnp.zeros((self.bin_count,), dtype=values.dtype)
            .at[self.shell_indices.reshape((-1,))]
            .add(weighted.reshape((-1,)))
        )

    def total_integral(self, mode_values: ArrayLike, /) -> Array:
        values = jnp.asarray(mode_values)
        if values.shape != self.source_shape:
            raise ValueError("Mode statistic does not match shell-bin geometry.")
        return jnp.sum(values * self.mode_weights.astype(values.real.dtype))


class PeriodicFourierShellPlan(StrictModule, NonTrainableState):
    """Prepared isotropic shell reduction for real periodic cell fields."""

    geometry: _FourierShellBinGeometry

    source_shape: tuple[int, ...] = eqx.field(static=True)
    transformed_shape: tuple[int, ...] = eqx.field(static=True)
    box_lengths: tuple[float, ...] = eqx.field(static=True)
    bin_count: int = eqx.field(static=True)
    dc_policy: DCPolicy = eqx.field(static=True)
    nyquist_policy: NyquistPolicy = eqx.field(static=True)
    final_edge_policy: FinalEdgePolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    cell_volume: float = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    bin_edges: Array
    wavenumber_magnitude: Array
    shell_indices: Array
    hermitian_multiplicity: Array
    weighted_mode_count: Array
    stored_mode_count: Array
    representative_wavenumbers: Array
    valid_shells: Array
    excluded_mode_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_shape: tuple[int, ...],
        box_lengths: tuple[float, ...],
        bin_edges: ArrayLike,
        /,
        *,
        dc_policy: DCPolicy = "exclude",
        nyquist_policy: NyquistPolicy = "include",
        final_edge_policy: FinalEdgePolicy = "include",
        source_id: str = "periodic-cell-field",
    ):
        shape = tuple(int(value) for value in source_shape)
        lengths = tuple(float(value) for value in box_lengths)
        edges = np.asarray(bin_edges, dtype=float).reshape((-1,))
        source = str(source_id).strip()
        if (
            len(shape) not in (1, 2, 3)
            or len(shape) != len(lengths)
            or any(value < 2 for value in shape)
            or any(not np.isfinite(value) or value <= 0.0 for value in lengths)
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
            or not source
            or dc_policy not in ("exclude", "include")
            or nyquist_policy not in ("include", "exclude")
            or final_edge_policy not in ("include", "exclude")
        ):
            raise ValueError("Periodic Fourier-shell geometry or policy is invalid.")
        frequencies = tuple(
            2.0
            * np.pi
            * (
                np.fft.rfftfreq(count, d=length / count)
                if axis == len(shape) - 1
                else np.fft.fftfreq(count, d=length / count)
            )
            for axis, (count, length) in enumerate(zip(shape, lengths, strict=True))
        )
        grids = np.meshgrid(*frequencies, indexing="ij")
        magnitude = np.sqrt(sum(component**2 for component in grids))
        transformed_shape = magnitude.shape
        mode_mask = np.ones(transformed_shape, dtype=bool)
        if dc_policy == "exclude":
            mode_mask &= magnitude > 0.0
        last_indices = np.arange(transformed_shape[-1])
        multiplicity_last = np.full(last_indices.shape, 2.0)
        multiplicity_last[0] = 1.0
        if shape[-1] % 2 == 0:
            if nyquist_policy == "include":
                multiplicity_last[-1] = 1.0
            else:
                mode_mask[..., -1] = False
                multiplicity_last[-1] = 0.0
        multiplicity = np.broadcast_to(
            multiplicity_last.reshape((1,) * (len(shape) - 1) + (-1,)),
            transformed_shape,
        ).copy()
        geometry = _FourierShellBinGeometry(
            magnitude,
            edges,
            mode_mask=mode_mask,
            mode_weights=multiplicity,
            final_edge_policy=final_edge_policy,
            source_id=f"rfft:{source}",
        )
        volume = float(np.prod(lengths))
        cell_volume = volume / int(np.prod(shape))
        self.source_shape = shape
        self.transformed_shape = transformed_shape
        self.box_lengths = lengths
        self.bin_count = edges.size - 1
        self.dc_policy = dc_policy
        self.nyquist_policy = nyquist_policy
        self.final_edge_policy = final_edge_policy
        self.source_id = source
        self.cell_volume = cell_volume
        self.volume = volume
        self.geometry = geometry
        self.bin_edges = geometry.bin_edges
        self.wavenumber_magnitude = geometry.wavenumber_magnitude
        self.shell_indices = geometry.shell_indices
        self.hermitian_multiplicity = geometry.mode_weights
        self.weighted_mode_count = geometry.weighted_mode_count
        self.stored_mode_count = geometry.stored_mode_count
        self.representative_wavenumbers = geometry.representative_wavenumbers
        self.valid_shells = geometry.valid_shells
        self.excluded_mode_count = geometry.excluded_mode_count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-fourier-shell-plan",
                "source_shape": list(shape),
                "box_lengths": list(lengths),
                "bin_edges": edges.tolist(),
                "dc_policy": dc_policy,
                "nyquist_policy": nyquist_policy,
                "final_edge_policy": final_edge_policy,
                "source_id": source,
            }
        )

    def transform(self, field: ArrayLike, /) -> PeriodicFourierField:
        values = jnp.asarray(field)
        if values.shape != self.source_shape:
            raise ValueError(
                f"Periodic Fourier field must have shape {self.source_shape}."
            )
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Periodic Fourier field must be finite.",
        )
        coefficients = self.cell_volume * jnp.fft.rfftn(values)
        finite = jnp.all(jnp.isfinite(coefficients))
        return PeriodicFourierField(coefficients, finite, self.plan_id)

    def _reduce(
        self,
        mode_values: Array,
        /,
        *,
        statistic_kind: str,
        correction: ModeTransferCorrection | None,
        shot_noise: ArrayLike = 0.0,
    ) -> FourierShellStatisticResult:
        if mode_values.shape != self.transformed_shape:
            raise ValueError("Mode statistic does not match shell-plan transform shape.")
        if (
            correction is not None
            and correction.multiplier.shape != self.transformed_shape
        ):
            raise ValueError("Mode-transfer correction does not match transform shape.")
        corrected = mode_values * (1.0 if correction is None else correction.multiplier)
        shell_integrals = self.geometry.reduce_integral(corrected)
        real_sum = jnp.real(shell_integrals)
        imaginary_sum = jnp.imag(shell_integrals)
        safe_count = jnp.where(
            self.weighted_mode_count > 0.0, self.weighted_mode_count, 1.0
        )
        shot = jnp.asarray(shot_noise, dtype=corrected.real.dtype)
        if shot.shape != ():
            raise ValueError("Shot-noise subtraction must be scalar.")
        shell_values = real_sum / safe_count - shot
        shell_values = jnp.where(self.valid_shells, shell_values, 0.0)
        imaginary = imaginary_sum / safe_count
        imaginary = jnp.where(self.valid_shells, imaginary, 0.0)
        finite = (
            jnp.all(jnp.isfinite(shell_values))
            & jnp.all(jnp.isfinite(imaginary))
            & jnp.isfinite(shot)
        )
        successful = finite & jnp.all(
            jnp.where(self.valid_shells, self.weighted_mode_count > 0.0, True)
        )
        return FourierShellStatisticResult(
            self.representative_wavenumbers,
            self.bin_edges,
            shell_values,
            self.weighted_mode_count,
            self.stored_mode_count,
            self.valid_shells,
            jnp.max(jnp.abs(imaginary)),
            jnp.real(self.geometry.total_integral(corrected)),
            jnp.asarray(self.excluded_mode_count, dtype=jnp.int32),
            finite,
            successful,
            statistic_kind,
            self.plan_id,
            "none" if correction is None else correction.correction_id,
        )

    def auto_power(
        self,
        field: PeriodicFourierField,
        /,
        *,
        correction: ModeTransferCorrection | None = None,
        shot_noise: ArrayLike = 0.0,
    ) -> FourierShellStatisticResult:
        if field.plan_id != self.plan_id:
            raise ValueError("Fourier field and shell plan disagree.")
        mode_values = jnp.abs(field.coefficients) ** 2 / self.volume
        return self._reduce(
            mode_values.astype(jnp.complex128),
            statistic_kind="auto-power",
            correction=correction,
            shot_noise=shot_noise,
        )

    def cross_power(
        self,
        left: PeriodicFourierField,
        right: PeriodicFourierField,
        /,
        *,
        correction: ModeTransferCorrection | None = None,
    ) -> FourierShellStatisticResult:
        if left.plan_id != self.plan_id or right.plan_id != self.plan_id:
            raise ValueError("Fourier fields and shell plan disagree.")
        mode_values = left.coefficients * jnp.conj(right.coefficients) / self.volume
        return self._reduce(
            mode_values,
            statistic_kind="cross-power",
            correction=correction,
        )

    def discrepancy(
        self,
        predicted: PeriodicFourierField,
        target: PeriodicFourierField,
        /,
        *,
        correction: ModeTransferCorrection | None = None,
    ) -> FourierShellStatisticResult:
        if predicted.plan_id != self.plan_id or target.plan_id != self.plan_id:
            raise ValueError("Fourier fields and shell plan disagree.")
        difference = predicted.coefficients - target.coefficients
        mode_values = jnp.abs(difference) ** 2 / self.volume
        return self._reduce(
            mode_values.astype(jnp.complex128),
            statistic_kind="field-discrepancy",
            correction=correction,
        )


__all__ = [
    "DCPolicy",
    "FinalEdgePolicy",
    "FourierShellStatisticResult",
    "ModeTransferCorrection",
    "NyquistPolicy",
    "PeriodicFourierField",
    "PeriodicFourierShellPlan",
]
