#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..control._frequency import (
    frequency_response,
    FrequencyResponseResult,
    FrequencySystemType,
)


class LinearGaussianTransferResult(StrictModule):
    """Input and process-noise transfers sharing one diagnosed resolvent."""

    diagnostics: FrequencyResponseResult
    input_to_state: Array
    input_to_output: Array
    process_to_state: Array
    process_to_output: Array
    method_id: str = eqx.field(static=True)


class LinearGaussianSpectra(StrictModule):
    r"""Stationary spectra with cross orientation :math:`S_{ab}=E[a b^H]`."""

    transfer: LinearGaussianTransferResult
    input_spectrum: Array
    process_spectrum: Array
    measurement_spectrum: Array
    state_spectrum: Array
    output_spectrum: Array
    state_output_cross_spectrum: Array
    state_input_cross_spectrum: Array
    output_input_cross_spectrum: Array
    valid: Array
    method_id: str = eqx.field(static=True)


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _matrix(value: ArrayLike, /, *, owner: str) -> Array:
    array = _inexact(value)
    if array.ndim < 2:
        raise ValueError(f"{owner} must have at least two dimensions.")
    return array


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _system_shapes(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    /,
) -> tuple[Array, tuple[int, ...], int, int, int]:
    state = _matrix(state_matrix, owner="state_matrix")
    inputs = _matrix(input_matrix, owner="input_matrix")
    outputs = _matrix(output_matrix, owner="output_matrix")
    feedthrough = _matrix(feedthrough_matrix, owner="feedthrough_matrix")
    state_size = int(state.shape[-1])
    input_size = int(inputs.shape[-1])
    output_size = int(outputs.shape[-2])
    if state.shape[-2:] != (state_size, state_size):
        raise ValueError("state_matrix must end in a square matrix.")
    if inputs.shape[-2] != state_size:
        raise ValueError("input_matrix row count must equal state size.")
    if outputs.shape[-1] != state_size:
        raise ValueError("output_matrix column count must equal state size.")
    if feedthrough.shape[-2:] != (output_size, input_size):
        raise ValueError("feedthrough_matrix must end in (output_size, input_size).")
    batch_shape = jnp.broadcast_shapes(
        state.shape[:-2],
        inputs.shape[:-2],
        outputs.shape[:-2],
        feedthrough.shape[:-2],
    )
    outputs = jnp.broadcast_to(outputs, batch_shape + (output_size, state_size))
    return outputs, batch_shape, state_size, input_size, output_size


def linear_gaussian_transfer_function(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
    *,
    system_type: FrequencySystemType = "continuous",
    sample_time: float = 1.0,
    singular_rtol: float | None = None,
    singular_atol: float = 0.0,
    stability_margin: float = 0.0,
) -> LinearGaussianTransferResult:
    """Build stochastic input/process transfers from one control resolvent."""

    outputs, batch_shape, _, _, _ = _system_shapes(
        state_matrix, input_matrix, output_matrix, feedthrough_matrix
    )
    diagnosed = frequency_response(
        state_matrix,
        input_matrix,
        output_matrix,
        feedthrough_matrix,
        angular_frequencies,
        system_type=system_type,
        sample_time=sample_time,
        singular_rtol=singular_rtol,
        singular_atol=singular_atol,
        stability_margin=stability_margin,
    )
    point_rank = diagnosed.resolvent.ndim - len(batch_shape) - 2
    expanded_outputs = outputs.reshape(
        batch_shape + (1,) * point_rank + outputs.shape[-2:]
    )
    process_to_output = expanded_outputs @ diagnosed.resolvent
    return LinearGaussianTransferResult(
        diagnostics=diagnosed,
        input_to_state=diagnosed.state_response,
        input_to_output=diagnosed.response,
        process_to_state=diagnosed.resolvent,
        process_to_output=process_to_output,
        method_id="control-resolvent/linear-gaussian-transfer",
    )


def _validated_spectrum(
    value: ArrayLike,
    size: int,
    batch_shape: tuple[int, ...],
    point_shape: tuple[int, ...],
    /,
    *,
    owner: str,
) -> Array:
    spectrum = _matrix(value, owner=owner)
    if spectrum.shape[-2:] != (size, size):
        raise ValueError(f"{owner} must end in shape ({size}, {size}).")
    spectrum_prefix = tuple(spectrum.shape[:-2])
    if spectrum_prefix == batch_shape:
        spectrum = spectrum.reshape(batch_shape + (1,) * len(point_shape) + (size, size))
    spectrum = jnp.broadcast_to(spectrum, batch_shape + point_shape + (size, size))
    if bool(jnp.any(~jnp.isfinite(spectrum))):
        raise ValueError(f"{owner} must be finite.")
    adjoint = _adjoint(spectrum)
    real_dtype = jnp.real(spectrum).dtype
    tolerance = float(jnp.finfo(real_dtype).eps * max(8, 8 * size))
    scale = jnp.maximum(jnp.max(jnp.abs(spectrum), axis=(-2, -1)), 1.0)
    hermitian_error = jnp.max(jnp.abs(spectrum - adjoint), axis=(-2, -1))
    if bool(jnp.any(hermitian_error > tolerance * scale)):
        raise ValueError(f"{owner} must be Hermitian.")
    eigenvalues = jnp.linalg.eigvalsh(spectrum)
    spectral_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues), axis=-1), 1.0)
    if bool(jnp.any(eigenvalues[..., 0] < -tolerance * spectral_scale)):
        raise ValueError(f"{owner} must be positive semidefinite.")
    return spectrum


def linear_gaussian_spectral_densities(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
    *,
    input_spectrum: ArrayLike,
    process_spectrum: ArrayLike,
    measurement_spectrum: ArrayLike,
    system_type: FrequencySystemType = "continuous",
    sample_time: float = 1.0,
    singular_rtol: float | None = None,
    singular_atol: float = 0.0,
    stability_margin: float = 0.0,
) -> LinearGaussianSpectra:
    """Compute stationary state/output/input spectra for a stable LTI model.

    ``process_spectrum`` acts directly in state coordinates and
    ``measurement_spectrum`` acts directly in output coordinates. No spectral
    factor is squared implicitly and no covariance is clipped or repaired.
    """

    _, batch_shape, state_size, input_size, output_size = _system_shapes(
        state_matrix, input_matrix, output_matrix, feedthrough_matrix
    )
    transfer = linear_gaussian_transfer_function(
        state_matrix,
        input_matrix,
        output_matrix,
        feedthrough_matrix,
        angular_frequencies,
        system_type=system_type,
        sample_time=sample_time,
        singular_rtol=singular_rtol,
        singular_atol=singular_atol,
        stability_margin=stability_margin,
    )
    if bool(jnp.any(~transfer.diagnostics.stable)):
        raise ValueError("Stationary spectra require a stable system.")
    if bool(jnp.any(transfer.diagnostics.singular)):
        raise ValueError("Stationary spectra are undefined at singular resolvents.")
    prefix_shape = tuple(transfer.input_to_state.shape[:-2])
    point_shape = prefix_shape[len(batch_shape) :]
    input_psd = _validated_spectrum(
        input_spectrum,
        input_size,
        batch_shape,
        point_shape,
        owner="input_spectrum",
    )
    process_psd = _validated_spectrum(
        process_spectrum,
        state_size,
        batch_shape,
        point_shape,
        owner="process_spectrum",
    )
    measurement_psd = _validated_spectrum(
        measurement_spectrum,
        output_size,
        batch_shape,
        point_shape,
        owner="measurement_spectrum",
    )

    h_xu = transfer.input_to_state
    h_yu = transfer.input_to_output
    h_xw = transfer.process_to_state
    h_yw = transfer.process_to_output
    state_from_input = h_xu @ input_psd @ _adjoint(h_xu)
    state_from_process = h_xw @ process_psd @ _adjoint(h_xw)
    output_from_input = h_yu @ input_psd @ _adjoint(h_yu)
    output_from_process = h_yw @ process_psd @ _adjoint(h_yw)
    state_spectrum_value = state_from_input + state_from_process
    output_spectrum_value = output_from_input + output_from_process + measurement_psd
    state_output_cross = h_xu @ input_psd @ _adjoint(
        h_yu
    ) + h_xw @ process_psd @ _adjoint(h_yw)
    state_input_cross = h_xu @ input_psd
    output_input_cross = h_yu @ input_psd
    finite = (
        jnp.all(jnp.isfinite(state_spectrum_value), axis=(-2, -1))
        & jnp.all(jnp.isfinite(output_spectrum_value), axis=(-2, -1))
        & jnp.all(jnp.isfinite(state_output_cross), axis=(-2, -1))
        & jnp.all(jnp.isfinite(state_input_cross), axis=(-2, -1))
        & jnp.all(jnp.isfinite(output_input_cross), axis=(-2, -1))
    )
    valid = transfer.diagnostics.valid & finite
    return LinearGaussianSpectra(
        transfer=transfer,
        input_spectrum=input_psd,
        process_spectrum=process_psd,
        measurement_spectrum=measurement_psd,
        state_spectrum=state_spectrum_value,
        output_spectrum=output_spectrum_value,
        state_output_cross_spectrum=state_output_cross,
        state_input_cross_spectrum=state_input_cross,
        output_input_cross_spectrum=output_input_cross,
        valid=valid,
        method_id="control-resolvent/conjugate-spectral-propagation",
    )


def state_spectral_density(*args, **kwargs) -> Array:
    """Return the state auto-spectrum from ``linear_gaussian_spectral_densities``."""

    return linear_gaussian_spectral_densities(*args, **kwargs).state_spectrum


def output_spectral_density(*args, **kwargs) -> Array:
    """Return the output auto-spectrum from ``linear_gaussian_spectral_densities``."""

    return linear_gaussian_spectral_densities(*args, **kwargs).output_spectrum


def state_output_cross_spectral_density(*args, **kwargs) -> Array:
    """Return :math:`S_{xy}=E[x y^H]`."""

    return linear_gaussian_spectral_densities(*args, **kwargs).state_output_cross_spectrum


def state_input_cross_spectral_density(*args, **kwargs) -> Array:
    """Return :math:`S_{xu}=E[x u^H]`."""

    return linear_gaussian_spectral_densities(*args, **kwargs).state_input_cross_spectrum


def output_input_cross_spectral_density(*args, **kwargs) -> Array:
    """Return :math:`S_{yu}=E[y u^H]`."""

    return linear_gaussian_spectral_densities(*args, **kwargs).output_input_cross_spectrum
