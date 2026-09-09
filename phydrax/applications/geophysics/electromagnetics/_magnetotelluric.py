#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....interchange import MTImpedanceData
from ._layered import LayeredEarthModel


def _inverse_2x2(matrix: Array, name: str) -> Array:
    if matrix.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have trailing 2x2 shape.")
    determinant = (
        matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
    )
    scale = jnp.max(jnp.abs(matrix), axis=(-2, -1))
    determinant = eqx.error_if(
        determinant,
        jnp.any(~jnp.isfinite(matrix))
        | jnp.any(
            jnp.abs(determinant) <= 64 * jnp.finfo(jnp.real(matrix).dtype).eps * scale**2
        ),
        f"{name} must be finite and nonsingular.",
    )
    inverse = jnp.stack(
        (
            jnp.stack((matrix[..., 1, 1], -matrix[..., 0, 1]), axis=-1),
            jnp.stack((-matrix[..., 1, 0], matrix[..., 0, 0]), axis=-1),
        ),
        axis=-2,
    )
    return inverse / determinant[..., None, None]


class GalvanicDistortion(StrictModule):
    matrix: Array

    def __init__(self, matrix: ArrayLike, /):
        value = jnp.asarray(matrix)
        if value.shape != (2, 2) or jnp.iscomplexobj(value):
            raise ValueError("Galvanic distortion must be one real 2x2 matrix.")
        determinant = value[0, 0] * value[1, 1] - value[0, 1] * value[1, 0]
        self.matrix = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)) | (determinant <= 0),
            "Galvanic distortion must be finite and orientation preserving.",
        )

    def apply(self, impedance: ArrayLike, /) -> Array:
        value = jnp.asarray(impedance)
        if value.shape[-2:] != (2, 2):
            raise ValueError("MT impedance must have trailing 2x2 shape.")
        return self.matrix @ value


class MagnetotelluricResponse(StrictModule):
    impedance_ohm: Array
    tipper: Array | None
    phase_tensor: Array
    apparent_resistivity_ohm_m: Array
    phase_radians: Array
    finite: Array


class MagnetotelluricResponsePlan(StrictModule, NonTrainableState):
    frequencies_Hz: Array

    def __init__(self, frequencies_Hz: ArrayLike, /):
        frequency = jnp.asarray(frequencies_Hz)
        if frequency.ndim != 1 or frequency.size == 0:
            raise ValueError("MT frequencies must be a nonempty vector.")
        self.frequencies_Hz = eqx.error_if(
            frequency,
            jnp.any(~jnp.isfinite(frequency)) | jnp.any(frequency <= 0),
            "MT frequencies must be finite and positive.",
        )

    @classmethod
    def from_layered(
        cls, model: LayeredEarthModel, frequencies_Hz: ArrayLike, /
    ) -> tuple[MagnetotelluricResponsePlan, Array]:
        if not isinstance(model, LayeredEarthModel):
            raise TypeError("Layered MT response requires LayeredEarthModel.")
        plan = cls(frequencies_Hz)
        scalar = model.magnetotelluric_impedance(plan.frequencies_Hz)
        tensor = jnp.zeros((scalar.size, 2, 2), dtype=scalar.dtype)
        tensor = tensor.at[:, 0, 1].set(scalar)
        tensor = tensor.at[:, 1, 0].set(-scalar)
        return plan, tensor

    def evaluate(
        self,
        electric_horizontal: ArrayLike,
        magnetic_horizontal: ArrayLike,
        /,
        *,
        magnetic_vertical: ArrayLike | None = None,
        distortion: GalvanicDistortion | None = None,
    ) -> MagnetotelluricResponse:
        electric = jnp.asarray(electric_horizontal)
        magnetic = jnp.asarray(magnetic_horizontal)
        expected = (self.frequencies_Hz.size, 2, 2)
        if electric.shape != expected or magnetic.shape != expected:
            raise ValueError(
                "MT electric/magnetic calibration fields must be (frequency,2,2)."
            )
        impedance = electric @ _inverse_2x2(magnetic, "MT horizontal magnetic field")
        if distortion is not None:
            if not isinstance(distortion, GalvanicDistortion):
                raise TypeError("MT distortion must be GalvanicDistortion or None.")
            impedance = distortion.apply(impedance)
        real = jnp.real(impedance)
        phase_tensor = _inverse_2x2(real, "MT real impedance") @ jnp.imag(impedance)
        omega = 2 * jnp.pi * self.frequencies_Hz
        permeability = 1.25663706212e-6
        apparent = jnp.abs(impedance) ** 2 / (permeability * omega[:, None, None])
        phase = jnp.angle(impedance)
        tipper = None
        if magnetic_vertical is not None:
            vertical = jnp.asarray(magnetic_vertical)
            if vertical.shape != (self.frequencies_Hz.size, 2):
                raise ValueError("MT vertical magnetic field must be (frequency,2).")
            tipper = ein.contract(
                "fi,fij->fj", vertical, _inverse_2x2(magnetic, "MT magnetic field")
            )
        finite = (
            jnp.all(jnp.isfinite(impedance))
            & jnp.all(jnp.isfinite(phase_tensor))
            & jnp.all(jnp.isfinite(apparent))
            & (True if tipper is None else jnp.all(jnp.isfinite(tipper)))
        )
        return MagnetotelluricResponse(
            impedance, tipper, phase_tensor, apparent, phase, finite
        )

    def log_likelihood(
        self,
        prediction_ohm: ArrayLike,
        data: MTImpedanceData,
        /,
    ) -> Array:
        if not isinstance(data, MTImpedanceData):
            raise TypeError("MT likelihood requires qualified MTImpedanceData.")
        prediction = jnp.asarray(prediction_ohm)
        if (
            prediction.shape != data.impedance_ohm.shape
            or prediction.shape[0] != self.frequencies_Hz.size
        ):
            raise ValueError("MT prediction and data shapes disagree.")
        if not np.allclose(
            np.asarray(self.frequencies_Hz), np.asarray(data.frequencies_Hz)
        ):
            raise ValueError("MT prediction and data frequencies disagree.")
        residual = prediction - data.impedance_ohm
        variance = data.standard_deviation_ohm**2
        return jnp.sum(-(jnp.abs(residual) ** 2) / variance - jnp.log(jnp.pi * variance))


class RemoteReferenceMTPlan(StrictModule, NonTrainableState):
    """Cross-spectral remote-reference transfer estimator."""

    regularization: float = eqx.field(static=True)

    def __init__(self, *, regularization: float = 0.0):
        value = float(regularization)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                "Remote-reference regularization must be nonnegative finite."
            )
        self.regularization = value

    def estimate(
        self,
        electric_local: ArrayLike,
        magnetic_local: ArrayLike,
        magnetic_remote: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        electric = jnp.asarray(electric_local)
        local = jnp.asarray(magnetic_local)
        remote = jnp.asarray(magnetic_remote)
        if (
            electric.ndim != 3
            or electric.shape[-1] != 2
            or local.shape != electric.shape
            or remote.shape != electric.shape
            or electric.shape[0] < 2
        ):
            raise ValueError("Remote-reference arrays must be (windows,frequency,2).")
        electric = eqx.error_if(
            electric,
            jnp.any(~jnp.isfinite(electric))
            | jnp.any(~jnp.isfinite(local))
            | jnp.any(~jnp.isfinite(remote)),
            "Remote-reference electric and magnetic windows must be finite.",
        )
        cross_e = (
            ein.contract("wfi,wfj->fij", electric, jnp.conj(remote)) / electric.shape[0]
        )
        cross_h = (
            ein.contract("wfi,wfj->fij", local, jnp.conj(remote)) / electric.shape[0]
        )
        diagonal_cross = jnp.diagonal(cross_h, axis1=-2, axis2=-1)
        local_power = jnp.mean(jnp.abs(local) ** 2, axis=0)
        remote_power = jnp.mean(jnp.abs(remote) ** 2, axis=0)
        coherence_denominator = local_power * remote_power
        coherence = jnp.abs(diagonal_cross) ** 2 / jnp.where(
            coherence_denominator > 0, coherence_denominator, 1.0
        )
        tolerance = 100 * jnp.finfo(coherence.dtype).eps
        coherence = eqx.error_if(
            coherence,
            jnp.any(~jnp.isfinite(coherence))
            | jnp.any(coherence_denominator <= 0)
            | jnp.any(coherence < -tolerance)
            | jnp.any(coherence > 1.0 + tolerance),
            "Remote-reference coherence is undefined or outside its physical range.",
        )
        coherence = jnp.minimum(jnp.maximum(coherence, 0.0), 1.0)
        regularized_cross_h = cross_h + self.regularization * jnp.eye(2)
        impedance = cross_e @ _inverse_2x2(
            regularized_cross_h, "remote-reference magnetic spectrum"
        )
        return impedance, coherence


__all__ = [
    "GalvanicDistortion",
    "MagnetotelluricResponse",
    "MagnetotelluricResponsePlan",
    "RemoteReferenceMTPlan",
]
