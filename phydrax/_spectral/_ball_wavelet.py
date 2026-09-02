#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, log, prod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract
from scipy.integrate import quad
from scipy.special import gammaln

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._fourier_laguerre import FourierLaguerrePlan
from ._laguerre import RadialLaguerrePlan
from ._spherical import SphericalExecution
from ._wigner import WignerTransformPlan
from ._wigner_laguerre import WignerLaguerrePlan


_DEFAULT_RESOURCE_BYTES = 512 * 1024**2
_TILING_TOLERANCE = 1.0e-11


def _array_bytes(tree: object, /) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, (np.ndarray, jax.Array))
    )


def _tiling_integrand(value: float, dilation: float, /) -> float:
    mapped = (value - 1.0 / dilation) * (2.0 * dilation / (dilation - 1.0)) - 1.0
    if abs(mapped) >= 1.0:
        return 0.0
    return float(np.exp(-2.0 / (1.0 - mapped**2)) / value)


def _scale_windows(
    bandlimit: int,
    dilation: float,
    minimum_scale: int,
    /,
) -> tuple[tuple[int, ...], np.ndarray]:
    maximum_scale = int(ceil(log(bandlimit - 1) / log(dilation)))
    if minimum_scale < 0 or minimum_scale > maximum_scale:
        raise ValueError(
            "minimum wavelet scale must lie between zero and its maximum scale."
        )
    normalization = quad(
        _tiling_integrand,
        1.0 / dilation,
        1.0,
        args=(dilation,),
        epsabs=1.0e-13,
        epsrel=1.0e-13,
        limit=100,
    )[0]
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise RuntimeError("wavelet generating-function normalization failed.")
    cumulative = np.zeros((maximum_scale + 2, bandlimit), dtype=float)
    modes = np.arange(bandlimit, dtype=float)
    for scale in range(maximum_scale + 2):
        normalized_modes = modes / dilation**scale
        for mode, value in enumerate(normalized_modes):
            if value <= 1.0 / dilation:
                cumulative[scale, mode] = 1.0
            elif value >= 1.0:
                cumulative[scale, mode] = 0.0
            else:
                integral = quad(
                    _tiling_integrand,
                    float(value),
                    1.0,
                    args=(dilation,),
                    epsabs=1.0e-13,
                    epsrel=1.0e-13,
                    limit=100,
                )[0]
                cumulative[scale, mode] = integral / normalization
    differences = cumulative[1:] - cumulative[:-1]
    minimum = float(np.min(differences))
    if minimum < -_TILING_TOLERANCE:
        raise RuntimeError("wavelet generating windows have negative energy.")
    differences = np.maximum(differences, 0.0)
    all_windows = np.sqrt(differences)
    scale_indices: list[int] = []
    windows: list[np.ndarray] = []
    for scale in range(minimum_scale, maximum_scale + 1):
        window = all_windows[scale]
        if np.any(window > 0.0):
            scale_indices.append(scale)
            windows.append(window)
    if not windows:
        raise ValueError("wavelet configuration has no active detail scales.")
    return tuple(scale_indices), np.stack(windows, axis=0)


def _directionality(bandlimit: int, directional_bandlimit: int, /) -> np.ndarray:
    phase = 1.0 if directional_bandlimit % 2 else 1.0j
    result = np.zeros((bandlimit, 2 * directional_bandlimit - 1), dtype=np.complex128)
    n_values = np.arange(-(directional_bandlimit - 1), directional_bandlimit, dtype=int)
    for degree in range(1, bandlimit):
        if (directional_bandlimit + degree) % 2:
            gamma = min(directional_bandlimit - 1, degree)
        else:
            gamma = min(directional_bandlimit - 1, degree - 1)
        for index, order in enumerate(n_values):
            if abs(order) > degree or (directional_bandlimit + order) % 2 == 0:
                continue
            choose = (gamma - order) // 2
            if choose < 0 or choose > gamma or gamma - order != 2 * choose:
                continue
            log_binomial = (
                gammaln(gamma + 1.0)
                - gammaln(choose + 1.0)
                - gammaln(gamma - choose + 1.0)
            )
            result[degree, index] = phase * np.sqrt(
                np.exp(log_binomial - gamma * np.log(2.0))
            )
    active_norms = np.sum(np.abs(result[1:]) ** 2, axis=1)
    if not np.allclose(active_norms, 1.0, rtol=0.0, atol=_TILING_TOLERANCE):
        raise RuntimeError("wavelet directionality is not degree-wise normalized.")
    return result


class _BallWaveletScale(NamedTuple):
    radial_scale: int
    angular_scale: int
    radial_window: int
    angular_window: int
    radial_bandlimit: int
    angular_bandlimit: int
    directional_bandlimit: int
    lower_angular_bandlimit: int
    radial_plan: int
    wigner_plan: int
    full_m_start: int
    full_n_start: int
    sample_shape: tuple[int, int, int, int]


class BallWaveletCoefficients(StrictModule):
    """Full-resolution scaling field and ordered ragged directional details."""

    scaling: Array
    details: tuple[Array, ...]
    scale_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        scaling: ArrayLike,
        details: Sequence[ArrayLike],
        /,
        *,
        scale_pairs: Sequence[tuple[int, int]],
        transform_id: str,
    ):
        scaling_array = jnp.asarray(scaling)
        detail_arrays = tuple(jnp.asarray(detail) for detail in details)
        pairs = tuple((int(radial), int(angular)) for radial, angular in scale_pairs)
        fingerprint = str(transform_id).strip()
        if not detail_arrays:
            raise ValueError("ball wavelet coefficients require detail leaves.")
        if len(detail_arrays) != len(pairs):
            raise ValueError("detail leaves and scale pairs must have equal lengths.")
        if not fingerprint:
            raise ValueError("transform_id must be non-empty.")
        self.scaling = scaling_array
        self.details = detail_arrays
        self.scale_pairs = pairs
        self.transform_id = fingerprint

    def with_coefficients(
        self,
        *,
        scaling: ArrayLike | None = None,
        details: Sequence[ArrayLike] | None = None,
    ) -> BallWaveletCoefficients:
        """Replace coefficient leaves while preserving transform metadata."""
        return BallWaveletCoefficients(
            self.scaling if scaling is None else scaling,
            self.details if details is None else details,
            scale_pairs=self.scale_pairs,
            transform_id=self.transform_id,
        )


class DirectionalBallWaveletPlan(StrictModule, NonTrainableState):
    """Exact scale-discretized directional wavelets on the radial-spherical ball."""

    fourier_laguerre: FourierLaguerrePlan
    angular_windows: Array
    radial_windows: Array
    directionality: Array
    scaling_window: Array
    radial_plans: tuple[RadialLaguerrePlan, ...]
    wigner_plans: tuple[WignerTransformPlan, ...]
    _scales: tuple[_BallWaveletScale, ...]
    angular_scale_indices: tuple[int, ...]
    radial_scale_indices: tuple[int, ...]
    directional_bandlimit: int
    angular_dilation: float
    radial_dilation: float
    angular_minimum_scale: int
    radial_minimum_scale: int
    wigner_execution: SphericalExecution
    max_scale_pairs: int
    max_precompute_bytes: int
    max_runtime_bytes: int
    admissibility_defect: float
    fingerprint: str

    def __init__(
        self,
        fourier_laguerre: FourierLaguerrePlan,
        /,
        *,
        directional_bandlimit: int,
        angular_dilation: float = 2.0,
        radial_dilation: float = 2.0,
        angular_minimum_scale: int = 0,
        radial_minimum_scale: int = 0,
        wigner_execution: SphericalExecution = "recursive",
        max_scale_pairs: int = 128,
        max_precompute_bytes: int = _DEFAULT_RESOURCE_BYTES,
        max_runtime_bytes: int = _DEFAULT_RESOURCE_BYTES,
    ):
        if not isinstance(fourier_laguerre, FourierLaguerrePlan):
            raise TypeError("fourier_laguerre must be a FourierLaguerrePlan.")
        angular = fourier_laguerre.angular
        if angular.spin != 0:
            raise ValueError("directional ball wavelets currently require spin zero.")
        angular_bandlimit = angular.bandlimit
        radial_bandlimit = fourier_laguerre.radial.radial_bandlimit
        selected_directional = int(directional_bandlimit)
        selected_angular_dilation = float(angular_dilation)
        selected_radial_dilation = float(radial_dilation)
        selected_angular_minimum = int(angular_minimum_scale)
        selected_radial_minimum = int(radial_minimum_scale)
        selected_execution = str(wigner_execution).lower()
        selected_scale_limit = int(max_scale_pairs)
        selected_precompute_limit = int(max_precompute_bytes)
        selected_runtime_limit = int(max_runtime_bytes)
        if angular_bandlimit < 2 or radial_bandlimit < 2:
            raise ValueError("directional ball wavelets require L >= 2 and P >= 2.")
        if selected_directional < 1 or selected_directional > angular_bandlimit:
            raise ValueError("directional_bandlimit must satisfy 1 <= N <= L.")
        if (
            not np.isfinite(selected_angular_dilation)
            or selected_angular_dilation <= 1.0
            or not np.isfinite(selected_radial_dilation)
            or selected_radial_dilation <= 1.0
        ):
            raise ValueError("wavelet dilation factors must be finite and exceed one.")
        if selected_execution not in ("recursive", "precomputed"):
            raise ValueError("wigner_execution must be 'recursive' or 'precomputed'.")
        if selected_scale_limit <= 0:
            raise ValueError("max_scale_pairs must be positive.")
        if selected_precompute_limit <= 0 or selected_runtime_limit <= 0:
            raise ValueError("wavelet resource limits must be positive.")

        angular_indices, angular_windows = _scale_windows(
            angular_bandlimit,
            selected_angular_dilation,
            selected_angular_minimum,
        )
        radial_indices, radial_windows = _scale_windows(
            radial_bandlimit,
            selected_radial_dilation,
            selected_radial_minimum,
        )
        pair_count = len(angular_indices) * len(radial_indices)
        if pair_count > selected_scale_limit:
            raise ValueError(
                f"wavelet configuration has {pair_count} scale pairs, exceeding "
                "max_scale_pairs."
            )
        directionality = _directionality(
            angular_bandlimit,
            selected_directional,
        )
        angular_energy = np.sum(angular_windows**2, axis=0)
        radial_energy = np.sum(radial_windows**2, axis=0)
        detail_energy = radial_energy[:, None] * angular_energy[None, :]
        scaling_energy = 1.0 - detail_energy
        if (
            float(np.min(scaling_energy)) < -_TILING_TOLERANCE
            or float(np.max(scaling_energy)) > 1.0 + _TILING_TOLERANCE
        ):
            raise RuntimeError("wavelet scaling complement is outside [0, 1].")
        scaling_window = np.sqrt(np.clip(scaling_energy, 0.0, 1.0))
        admissibility_defect = float(
            np.max(np.abs(scaling_window**2 + detail_energy - 1.0))
        )
        if admissibility_defect > _TILING_TOLERANCE:
            raise RuntimeError("wavelet filters fail pointwise admissibility.")

        filter_bytes = sum(
            int(array.size) * int(array.dtype.itemsize)
            for array in (
                angular_windows,
                radial_windows,
                directionality,
                scaling_window,
            )
        )
        remaining = (
            selected_precompute_limit - fourier_laguerre.precompute_bytes - filter_bytes
        )
        if remaining <= 0:
            raise ValueError(
                "base Fourier-Laguerre plan and wavelet filters exceed "
                "max_precompute_bytes."
            )

        radial_plans: list[RadialLaguerrePlan] = [fourier_laguerre.radial]
        radial_lookup = {radial_bandlimit: 0}
        wigner_plans: list[WignerTransformPlan] = []
        wigner_lookup: dict[tuple[int, int, int], int] = {}
        scales: list[_BallWaveletScale] = []
        full_directional_center = selected_directional - 1
        for radial_window_index, radial_scale in enumerate(radial_indices):
            radial_nonzero = np.flatnonzero(radial_windows[radial_window_index] > 0.0)
            local_radial_bandlimit = int(radial_nonzero[-1]) + 1
            if local_radial_bandlimit not in radial_lookup:
                radial_plan = RadialLaguerrePlan(
                    local_radial_bandlimit,
                    tau=fourier_laguerre.radial.tau,
                    max_precompute_bytes=remaining,
                )
                radial_lookup[local_radial_bandlimit] = len(radial_plans)
                radial_plans.append(radial_plan)
                remaining -= radial_plan.precompute_bytes
            radial_plan_index = radial_lookup[local_radial_bandlimit]
            for angular_window_index, angular_scale in enumerate(angular_indices):
                angular_nonzero = np.flatnonzero(
                    angular_windows[angular_window_index] > 0.0
                )
                local_lower = int(angular_nonzero[0])
                local_angular_bandlimit = int(angular_nonzero[-1]) + 1
                active_directionality = directionality[
                    local_lower:local_angular_bandlimit
                ]
                active_n = np.flatnonzero(
                    np.any(np.abs(active_directionality) > 0.0, axis=0)
                )
                if not active_n.size:
                    continue
                maximum_n = int(np.max(np.abs(active_n - full_directional_center)))
                local_directional_bandlimit = maximum_n + 1
                key = (
                    local_angular_bandlimit,
                    local_directional_bandlimit,
                    local_lower,
                )
                if key not in wigner_lookup:
                    wigner_plan = WignerTransformPlan(
                        local_angular_bandlimit,
                        local_directional_bandlimit,
                        sampling=angular.sampling,
                        execution=selected_execution,
                        lower_bandlimit=local_lower,
                        max_precompute_bytes=remaining,
                    )
                    wigner_lookup[key] = len(wigner_plans)
                    wigner_plans.append(wigner_plan)
                    remaining -= wigner_plan.precompute_bytes
                wigner_plan_index = wigner_lookup[key]
                wigner_plan = wigner_plans[wigner_plan_index]
                scales.append(
                    _BallWaveletScale(
                        radial_scale=radial_scale,
                        angular_scale=angular_scale,
                        radial_window=radial_window_index,
                        angular_window=angular_window_index,
                        radial_bandlimit=local_radial_bandlimit,
                        angular_bandlimit=local_angular_bandlimit,
                        directional_bandlimit=local_directional_bandlimit,
                        lower_angular_bandlimit=local_lower,
                        radial_plan=radial_plan_index,
                        wigner_plan=wigner_plan_index,
                        full_m_start=angular_bandlimit - local_angular_bandlimit,
                        full_n_start=(selected_directional - local_directional_bandlimit),
                        sample_shape=(
                            local_radial_bandlimit,
                            *wigner_plan.sample_shape,
                        ),
                    )
                )
        if not scales:
            raise ValueError("wavelet configuration has no active scale pairs.")

        self.fourier_laguerre = fourier_laguerre
        self.angular_windows = jnp.asarray(angular_windows)
        self.radial_windows = jnp.asarray(radial_windows)
        self.directionality = jnp.asarray(directionality)
        self.scaling_window = jnp.asarray(scaling_window)
        self.radial_plans = tuple(radial_plans)
        self.wigner_plans = tuple(wigner_plans)
        self._scales = tuple(scales)
        self.angular_scale_indices = angular_indices
        self.radial_scale_indices = radial_indices
        self.directional_bandlimit = selected_directional
        self.angular_dilation = selected_angular_dilation
        self.radial_dilation = selected_radial_dilation
        self.angular_minimum_scale = selected_angular_minimum
        self.radial_minimum_scale = selected_radial_minimum
        self.wigner_execution = selected_execution
        self.max_scale_pairs = selected_scale_limit
        self.max_precompute_bytes = selected_precompute_limit
        self.max_runtime_bytes = selected_runtime_limit
        self.admissibility_defect = admissibility_defect
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "directional-ball-wavelet-plan-v1",
                "fourier_laguerre": fourier_laguerre.transform_id,
                "directional_bandlimit": selected_directional,
                "angular_dilation": selected_angular_dilation,
                "radial_dilation": selected_radial_dilation,
                "angular_minimum_scale": selected_angular_minimum,
                "radial_minimum_scale": selected_radial_minimum,
                "scale_pairs": self.scale_pairs,
                "angular_windows": array_tree_fingerprint(angular_windows),
                "radial_windows": array_tree_fingerprint(radial_windows),
                "directionality": array_tree_fingerprint(directionality),
                "scaling_window": array_tree_fingerprint(scaling_window),
            }
        )
        if self.persistent_bytes > selected_precompute_limit:
            raise ValueError(
                "wavelet materialization exceeds max_precompute_bytes; "
                f"materialized {self.persistent_bytes} bytes."
            )

    @property
    def transform_id(self) -> str:
        """Identity of the exact directional wavelet frame."""
        return self.fingerprint

    @property
    def execution_id(self) -> str:
        """Identity of the concrete multiresolution execution."""
        return canonical_fingerprint(
            {
                "kind": "directional-ball-wavelet-execution-v1",
                "transform": self.transform_id,
                "fourier_laguerre": self.fourier_laguerre.execution_id,
                "wigner_execution": self.wigner_execution,
                "radial_plans": tuple(plan.execution_id for plan in self.radial_plans),
                "wigner_plans": tuple(plan.execution_id for plan in self.wigner_plans),
                "persistent_bytes": self.persistent_bytes,
            }
        )

    @property
    def scale_pairs(self) -> tuple[tuple[int, int], ...]:
        """Canonical radial-major ordering of active detail scales."""
        return tuple((scale.radial_scale, scale.angular_scale) for scale in self._scales)

    @property
    def detail_shapes(self) -> tuple[tuple[int, int, int, int], ...]:
        """Core radial-SO(3) sample shape of each ragged detail leaf."""
        return tuple(scale.sample_shape for scale in self._scales)

    @property
    def scale_count(self) -> int:
        """Number of active radial/angular detail pairs."""
        return len(self._scales)

    @property
    def persistent_bytes(self) -> int:
        """Bytes retained by unique transforms and compact wavelet filters."""
        reduced_radial = sum(plan.precompute_bytes for plan in self.radial_plans[1:])
        wigner = sum(plan.precompute_bytes for plan in self.wigner_plans)
        filters = _array_bytes(
            (
                self.angular_windows,
                self.radial_windows,
                self.directionality,
                self.scaling_window,
            )
        )
        return self.fourier_laguerre.precompute_bytes + reduced_radial + wigner + filters

    def output_bytes(self, fields: int = 1, /) -> int:
        """Conservative complex128 size of returned coefficient leaves."""
        field_count = int(fields)
        if field_count <= 0:
            raise ValueError("fields must be positive.")
        elements = prod(self.fourier_laguerre.sample_shape) + sum(
            prod(shape) for shape in self.detail_shapes
        )
        return field_count * elements * np.dtype(np.complex128).itemsize

    def workspace_bytes(self, fields: int = 1, /) -> int:
        """Structural modal workspace, excluding backend FFT allocator overhead."""
        field_count = int(fields)
        if field_count <= 0:
            raise ValueError("fields must be positive.")
        full_modal = prod(self.fourier_laguerre.coefficient_shape)
        largest_detail = max(
            scale.radial_bandlimit
            * (2 * scale.directional_bandlimit - 1)
            * scale.angular_bandlimit
            * (2 * scale.angular_bandlimit - 1)
            for scale in self._scales
        )
        return (
            field_count
            * (2 * full_modal + largest_detail)
            * np.dtype(np.complex128).itemsize
        )

    def estimated_peak_bytes(self, fields: int = 1, /) -> int:
        """Returned leaves plus structurally explicit runtime workspace."""
        return self.output_bytes(fields) + self.workspace_bytes(fields)

    def _input_context(
        self,
        shape: tuple[int, ...],
        core_shape: tuple[int, ...],
        name: str,
        /,
    ) -> tuple[tuple[int, ...], int | None]:
        core_rank = len(core_shape)
        if len(shape) >= core_rank and tuple(shape[-core_rank:]) == core_shape:
            return tuple(shape[:-core_rank]), None
        if (
            len(shape) >= core_rank + 1
            and tuple(shape[-core_rank - 1 : -1]) == core_shape
        ):
            return tuple(shape[: -core_rank - 1]), int(shape[-1])
        raise ValueError(f"{name} has an invalid core transform shape.")

    def _field_count(self, leading: tuple[int, ...], channels: int | None, /) -> int:
        return prod(leading) * (1 if channels is None else channels)

    def _detail_plan(self, scale: _BallWaveletScale, /) -> WignerLaguerrePlan:
        return WignerLaguerrePlan(
            self.radial_plans[scale.radial_plan],
            self.wigner_plans[scale.wigner_plan],
        )

    def analysis(self, values: ArrayLike, /) -> BallWaveletCoefficients:
        """Analyze radial-spherical samples into scaling and directional details."""
        array = jnp.asarray(values)
        shape = tuple(int(size) for size in array.shape)
        leading, channels = self._input_context(
            shape,
            self.fourier_laguerre.sample_shape,
            "directional ball wavelet input",
        )
        fields = self._field_count(leading, channels)
        estimate = self.estimated_peak_bytes(fields)
        if estimate > self.max_runtime_bytes:
            raise ValueError(
                "directional ball wavelet analysis exceeds max_runtime_bytes; "
                f"estimated {estimate} bytes."
            )
        coefficients = self.fourier_laguerre.analysis(array)
        if channels is None:
            scaling_modes = contract(
                "...plm,pl->...plm",
                coefficients,
                self.scaling_window,
            )
        else:
            scaling_modes = contract(
                "...plmc,pl->...plmc",
                coefficients,
                self.scaling_window,
            )
        scaling = self.fourier_laguerre.synthesis(scaling_modes)
        details: list[Array] = []
        for scale in self._scales:
            m_stop = scale.full_m_start + 2 * scale.angular_bandlimit - 1
            radial_window = self.radial_windows[
                scale.radial_window, : scale.radial_bandlimit
            ]
            angular_window = self.angular_windows[
                scale.angular_window, : scale.angular_bandlimit
            ]
            n_stop = scale.full_n_start + 2 * scale.directional_bandlimit - 1
            zeta = self.directionality[
                : scale.angular_bandlimit,
                scale.full_n_start : n_stop,
            ]
            degree_factor = jnp.sqrt(
                8.0 * jnp.pi**2 / (2.0 * jnp.arange(scale.angular_bandlimit) + 1.0)
            )
            analysis_directionality = jnp.conj(zeta) * degree_factor[:, None]
            if channels is None:
                subset = coefficients[
                    ...,
                    : scale.radial_bandlimit,
                    : scale.angular_bandlimit,
                    scale.full_m_start : m_stop,
                ]
                detail_modes = contract(
                    "...plm,p,l,ln->...pnlm",
                    subset,
                    radial_window,
                    angular_window,
                    analysis_directionality,
                )
            else:
                subset = coefficients[
                    ...,
                    : scale.radial_bandlimit,
                    : scale.angular_bandlimit,
                    scale.full_m_start : m_stop,
                    :,
                ]
                detail_modes = contract(
                    "...plmc,p,l,ln->...pnlmc",
                    subset,
                    radial_window,
                    angular_window,
                    analysis_directionality,
                )
            details.append(self._detail_plan(scale).synthesis(detail_modes))
        return BallWaveletCoefficients(
            scaling,
            tuple(details),
            scale_pairs=self.scale_pairs,
            transform_id=self.transform_id,
        )

    def synthesis(self, coefficients: BallWaveletCoefficients, /) -> Array:
        """Synthesize radial-spherical samples from scaling and detail leaves."""
        if not isinstance(coefficients, BallWaveletCoefficients):
            raise TypeError("coefficients must be BallWaveletCoefficients.")
        if coefficients.transform_id != self.transform_id:
            raise ValueError("ball wavelet coefficients belong to another transform.")
        if coefficients.scale_pairs != self.scale_pairs:
            raise ValueError("ball wavelet coefficient scale ordering is incompatible.")
        if len(coefficients.details) != len(self._scales):
            raise ValueError("ball wavelet detail count is incompatible.")
        scaling_shape = tuple(int(size) for size in coefficients.scaling.shape)
        leading, channels = self._input_context(
            scaling_shape,
            self.fourier_laguerre.sample_shape,
            "ball wavelet scaling coefficients",
        )
        expected_context = (leading, channels)
        for detail, scale in zip(coefficients.details, self._scales, strict=True):
            detail_shape = tuple(int(size) for size in detail.shape)
            if (
                self._input_context(
                    detail_shape,
                    scale.sample_shape,
                    "ball wavelet detail coefficients",
                )
                != expected_context
            ):
                raise ValueError(
                    "ball wavelet detail batch/channel shape is incompatible."
                )
        fields = self._field_count(leading, channels)
        estimate = self.estimated_peak_bytes(fields)
        if estimate > self.max_runtime_bytes:
            raise ValueError(
                "directional ball wavelet synthesis exceeds max_runtime_bytes; "
                f"estimated {estimate} bytes."
            )

        scaling_modes = self.fourier_laguerre.analysis(coefficients.scaling)
        if channels is None:
            reconstructed_modes = contract(
                "...plm,pl->...plm",
                scaling_modes,
                self.scaling_window,
            )
        else:
            reconstructed_modes = contract(
                "...plmc,pl->...plmc",
                scaling_modes,
                self.scaling_window,
            )
        for detail, scale in zip(coefficients.details, self._scales, strict=True):
            detail_modes = self._detail_plan(scale).analysis(detail)
            radial_window = self.radial_windows[
                scale.radial_window, : scale.radial_bandlimit
            ]
            angular_window = self.angular_windows[
                scale.angular_window, : scale.angular_bandlimit
            ]
            n_stop = scale.full_n_start + 2 * scale.directional_bandlimit - 1
            zeta = self.directionality[
                : scale.angular_bandlimit,
                scale.full_n_start : n_stop,
            ]
            degree_factor = jnp.sqrt(
                (2.0 * jnp.arange(scale.angular_bandlimit) + 1.0) / (8.0 * jnp.pi**2)
            )
            synthesis_directionality = zeta * degree_factor[:, None]
            if channels is None:
                contribution = contract(
                    "...pnlm,p,l,ln->...plm",
                    detail_modes,
                    radial_window,
                    angular_window,
                    synthesis_directionality,
                )
            else:
                contribution = contract(
                    "...pnlmc,p,l,ln->...plmc",
                    detail_modes,
                    radial_window,
                    angular_window,
                    synthesis_directionality,
                )
            m_stop = scale.full_m_start + 2 * scale.angular_bandlimit - 1
            if channels is None:
                reconstructed_modes = reconstructed_modes.at[
                    ...,
                    : scale.radial_bandlimit,
                    : scale.angular_bandlimit,
                    scale.full_m_start : m_stop,
                ].add(contribution)
            else:
                reconstructed_modes = reconstructed_modes.at[
                    ...,
                    : scale.radial_bandlimit,
                    : scale.angular_bandlimit,
                    scale.full_m_start : m_stop,
                    :,
                ].add(contribution)
        return self.fourier_laguerre.synthesis(reconstructed_modes)


__all__ = [
    "BallWaveletCoefficients",
    "DirectionalBallWaveletPlan",
]
