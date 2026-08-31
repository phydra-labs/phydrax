#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._products import CosmologyProductProvenance, CosmologyRealizationSignature


CMB_FIELDS = ("T", "E", "B", "phi")
CMB_MODES = ("scalar", "vector", "tensor", "total")


class PrimordialPowerLaw(StrictModule):
    """Differentiable scalar/tensor primordial power laws at fixed pivots."""

    scalar_amplitude: Array
    scalar_tilt: Array
    scalar_running: Array
    scalar_pivot: Array
    tensor_amplitude: Array
    tensor_tilt: Array
    tensor_pivot: Array
    model_form_id: str = eqx.field(static=True)

    def __init__(
        self,
        scalar_amplitude: ArrayLike,
        scalar_tilt: ArrayLike,
        scalar_pivot: ArrayLike,
        /,
        *,
        scalar_running: ArrayLike = 0.0,
        tensor_amplitude: ArrayLike = 0.0,
        tensor_tilt: ArrayLike = 0.0,
        tensor_pivot: ArrayLike | None = None,
    ):
        dtype = jnp.result_type(
            scalar_amplitude,
            scalar_tilt,
            scalar_pivot,
            scalar_running,
            tensor_amplitude,
            tensor_tilt,
        )
        scalar_a = jnp.asarray(scalar_amplitude, dtype=dtype)
        scalar_n = jnp.asarray(scalar_tilt, dtype=dtype)
        scalar_k = jnp.asarray(scalar_pivot, dtype=dtype)
        running = jnp.asarray(scalar_running, dtype=dtype)
        tensor_a = jnp.asarray(tensor_amplitude, dtype=dtype)
        tensor_n = jnp.asarray(tensor_tilt, dtype=dtype)
        tensor_k = jnp.asarray(
            scalar_pivot if tensor_pivot is None else tensor_pivot, dtype=dtype
        )
        values = (scalar_a, scalar_n, running, scalar_k, tensor_a, tensor_n, tensor_k)
        if any(value.shape != () for value in values):
            raise ValueError("Primordial power-law parameters must be scalar.")
        scalar_a = eqx.error_if(
            scalar_a,
            jnp.any(~jnp.isfinite(jnp.stack(values)))
            | (scalar_a < 0.0)
            | (tensor_a < 0.0)
            | (scalar_k <= 0.0)
            | (tensor_k <= 0.0),
            "Primordial power-law parameters are invalid.",
        )
        self.scalar_amplitude = scalar_a
        self.scalar_tilt = scalar_n
        self.scalar_running = running
        self.scalar_pivot = scalar_k
        self.tensor_amplitude = tensor_a
        self.tensor_tilt = tensor_n
        self.tensor_pivot = tensor_k
        self.model_form_id = canonical_fingerprint(
            {"kind": "scalar-tensor-primordial-power-law"}
        )

    def scalar_power(self, wavenumber: ArrayLike, /) -> Array:
        k = jnp.asarray(wavenumber, dtype=self.scalar_amplitude.dtype)
        k = eqx.error_if(
            k,
            jnp.any(~jnp.isfinite(k)) | jnp.any(k <= 0.0),
            "Primordial wavenumber must be finite and positive.",
        )
        logarithm = jnp.log(k / self.scalar_pivot)
        exponent = self.scalar_tilt - 1.0 + 0.5 * self.scalar_running * logarithm
        return self.scalar_amplitude * jnp.exp(exponent * logarithm)

    def tensor_power(self, wavenumber: ArrayLike, /) -> Array:
        k = jnp.asarray(wavenumber, dtype=self.scalar_amplitude.dtype)
        k = eqx.error_if(
            k,
            jnp.any(~jnp.isfinite(k)) | jnp.any(k <= 0.0),
            "Primordial wavenumber must be finite and positive.",
        )
        return self.tensor_amplitude * (k / self.tensor_pivot) ** self.tensor_tilt


class CmbSpectrumTable(StrictModule):
    """Canonical raw C_ell covariance over (T,E,B,phi)."""

    multipoles: Array
    spectra: Array
    modes: tuple[str, ...] = eqx.field(static=True)
    lensing_state: str = eqx.field(static=True)
    nonlinear_source_id: str = eqx.field(static=True)
    temperature_unit: str = eqx.field(static=True)
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature

    def __init__(
        self,
        multipoles: ArrayLike,
        spectra: ArrayLike,
        modes: tuple[str, ...],
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        /,
        *,
        lensing_state: str = "unlensed",
        nonlinear_source_id: str = "none",
        temperature_unit: str = "dimensionless-thermodynamic",
    ):
        ell_host = np.asarray(multipoles, dtype=int).reshape((-1,))
        modes_ = tuple(str(mode).strip() for mode in modes)
        lensing = str(lensing_state).strip()
        nonlinear = str(nonlinear_source_id).strip()
        unit = str(temperature_unit).strip()
        if (
            ell_host.size < 1
            or np.any(ell_host < 0)
            or np.any(np.diff(ell_host) <= 0)
            or not modes_
            or any(mode not in CMB_MODES for mode in modes_)
            or len(set(modes_)) != len(modes_)
            or lensing not in ("unlensed", "lensed", "delensed")
            or not nonlinear
            or not unit
        ):
            raise ValueError("CMB spectrum coordinates or conventions are invalid.")
        ell = jnp.asarray(ell_host)
        values = jnp.asarray(spectra)
        expected = (len(modes_), ell.size, len(CMB_FIELDS), len(CMB_FIELDS))
        if values.shape != expected:
            raise ValueError(f"CMB spectra must have shape {expected}.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values))
            | jnp.any(jnp.abs(values - jnp.swapaxes(values, -1, -2)) > 1.0e-10),
            "CMB spectra must be finite and symmetric.",
        )
        if provenance.differentiability != "native-parameter":
            ell = jax.lax.stop_gradient(ell)
            values = jax.lax.stop_gradient(values)
        self.multipoles = ell
        self.spectra = values
        self.modes = modes_
        self.lensing_state = lensing
        self.nonlinear_source_id = nonlinear
        self.temperature_unit = unit
        self.provenance = provenance
        self.realization = realization

    def d_ell(self, /) -> Array:
        ell = self.multipoles.astype(self.spectra.dtype)
        factor = ell * (ell + 1.0) / (2.0 * jnp.pi)
        field_factor = jnp.asarray((1.0, 1.0, 1.0, 0.0), dtype=self.spectra.dtype)
        block = field_factor[:, None] * field_factor[None, :]
        scaled = self.spectra * factor[None, :, None, None]
        return jnp.where(block[None, None, :, :] > 0.0, scaled, self.spectra)

    def temperature_scaled(self, temperature: ArrayLike, /) -> Array:
        scale = jnp.asarray(temperature, dtype=self.spectra.dtype)
        if scale.shape != ():
            raise ValueError("CMB thermodynamic temperature scale must be scalar.")
        field_scale = jnp.asarray((scale, scale, scale, 1.0), dtype=self.spectra.dtype)
        return (
            self.spectra
            * field_scale[None, None, :, None]
            * field_scale[None, None, None, :]
        )


class CmbSpectrumTransformPlan(StrictModule, NonTrainableState):
    """Static field/mode selection and canonical theory-vector packing."""

    mode_indices: tuple[int, ...] = eqx.field(static=True)
    field_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    use_d_ell: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_indices: tuple[int, ...],
        field_pairs: tuple[tuple[int, int], ...],
        /,
        *,
        use_d_ell: bool = False,
    ):
        modes = tuple(int(index) for index in mode_indices)
        pairs = tuple((int(left), int(right)) for left, right in field_pairs)
        if not modes or any(index < 0 for index in modes):
            raise ValueError("CMB mode selection is invalid.")
        if not pairs or any(
            left < 0
            or left >= len(CMB_FIELDS)
            or right < left
            or right >= len(CMB_FIELDS)
            for left, right in pairs
        ):
            raise ValueError("CMB field-pair selection is invalid.")
        self.mode_indices = modes
        self.field_pairs = pairs
        self.use_d_ell = bool(use_d_ell)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cmb-spectrum-transform",
                "mode_indices": list(modes),
                "field_pairs": [list(pair) for pair in pairs],
                "use_d_ell": bool(use_d_ell),
            }
        )

    def pack(self, table: CmbSpectrumTable, /) -> Array:
        if any(index >= len(table.modes) for index in self.mode_indices):
            raise ValueError("CMB mode selection exceeds table modes.")
        values = table.d_ell() if self.use_d_ell else table.spectra
        rows = tuple(
            values[mode, :, left, right]
            for mode in self.mode_indices
            for left, right in self.field_pairs
        )
        return jnp.stack(rows)


class CmbBandpowerResponseResult(StrictModule):
    predicted_bandpowers: Array
    residual: Array
    whitened_residual: Array
    log_likelihood: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class CmbBandpowerResponsePlan(StrictModule, NonTrainableState):
    """Fixed theory-vector windows and Cholesky-whitened bandpower likelihood."""

    transform: CmbSpectrumTransformPlan
    windows: Array
    observed_bandpowers: Array
    covariance_cholesky: Array
    expected_temperature_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transform: CmbSpectrumTransformPlan,
        windows: ArrayLike,
        observed_bandpowers: ArrayLike,
        covariance_cholesky: ArrayLike,
        /,
        *,
        expected_temperature_unit: str,
        response_id: str,
    ):
        if not isinstance(transform, CmbSpectrumTransformPlan):
            raise TypeError("transform must be a CmbSpectrumTransformPlan.")
        windows_host = np.asarray(windows, dtype=float)
        observed_host = np.asarray(observed_bandpowers, dtype=float)
        cholesky_host = np.asarray(covariance_cholesky, dtype=float)
        bands = windows_host.shape[0] if windows_host.ndim == 3 else 0
        if (
            windows_host.ndim != 3
            or observed_host.shape != (bands,)
            or cholesky_host.shape != (bands, bands)
            or np.any(~np.isfinite(windows_host))
            or np.any(~np.isfinite(observed_host))
            or np.any(~np.isfinite(cholesky_host))
            or np.any(np.diag(cholesky_host) <= 0.0)
        ):
            raise ValueError("CMB response arrays are invalid.")
        unit = str(expected_temperature_unit).strip()
        identifier = str(response_id).strip()
        if not unit or not identifier:
            raise ValueError("CMB response unit and ID must be non-empty.")
        self.transform = transform
        self.windows = jnp.asarray(windows_host)
        self.observed_bandpowers = jnp.asarray(observed_host)
        self.covariance_cholesky = jnp.asarray(cholesky_host)
        self.expected_temperature_unit = unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cmb-bandpower-response",
                "response_id": identifier,
                "transform": transform.plan_id,
                "shape": list(windows_host.shape),
                "temperature_unit": unit,
            }
        )

    def evaluate(self, table: CmbSpectrumTable, /) -> CmbBandpowerResponseResult:
        if not isinstance(table, CmbSpectrumTable):
            raise TypeError("table must be a CmbSpectrumTable.")
        if table.temperature_unit != self.expected_temperature_unit:
            raise ValueError("CMB table temperature unit does not match response.")
        packed = self.transform.pack(table)
        if packed.shape != self.windows.shape[1:]:
            raise ValueError("CMB packed theory grid does not match response windows.")
        predicted = contract("brl,rl->b", self.windows, packed)
        residual = self.observed_bandpowers - predicted
        whitened = jsp.linalg.solve_triangular(
            self.covariance_cholesky, residual, lower=True
        )
        valid = jnp.all(jnp.isfinite(predicted)) & jnp.all(jnp.isfinite(whitened))
        return CmbBandpowerResponseResult(
            predicted,
            residual,
            whitened,
            jnp.where(valid, -0.5 * jnp.sum(whitened * whitened), -jnp.inf),
            valid,
            self.plan_id,
            table.provenance.provenance_id,
        )


__all__ = [
    "CmbBandpowerResponsePlan",
    "CmbBandpowerResponseResult",
    "CMB_FIELDS",
    "CMB_MODES",
    "CmbSpectrumTable",
    "CmbSpectrumTransformPlan",
    "PrimordialPowerLaw",
]
