#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Petersen--Rostalski 2019 planar anisotropic surface-EMG conductor."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


PETERSEN_ROSTALSKI_2019_DOI = "10.3389/fphys.2019.00176"
PETERSEN_ROSTALSKI_2019_DRYAD_DOI = "10.5061/dryad.326qs26"


class PlanarConductorParameters(StrictModule):
    muscle_longitudinal_conductivity_S_per_m: Array
    muscle_transverse_conductivity_S_per_m: Array
    fat_conductivity_S_per_m: Array
    skin_conductivity_S_per_m: Array
    fat_thickness_m: Array
    skin_thickness_m: Array
    source_depth_m: Array

    def __init__(
        self,
        muscle_longitudinal_conductivity_S_per_m: ArrayLike,
        muscle_transverse_conductivity_S_per_m: ArrayLike,
        fat_conductivity_S_per_m: ArrayLike,
        skin_conductivity_S_per_m: ArrayLike,
        fat_thickness_m: ArrayLike,
        skin_thickness_m: ArrayLike,
        source_depth_m: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                muscle_longitudinal_conductivity_S_per_m,
                muscle_transverse_conductivity_S_per_m,
                fat_conductivity_S_per_m,
                skin_conductivity_S_per_m,
                fat_thickness_m,
                skin_thickness_m,
                source_depth_m,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Planar conductor parameters must be scalar.")
        values = tuple(
            value if jnp.issubdtype(value.dtype, jnp.inexact) else value.astype(float)
            for value in values
        )
        (
            self.muscle_longitudinal_conductivity_S_per_m,
            self.muscle_transverse_conductivity_S_per_m,
            self.fat_conductivity_S_per_m,
            self.skin_conductivity_S_per_m,
            self.fat_thickness_m,
            self.skin_thickness_m,
            self.source_depth_m,
        ) = values


class PlanarConductorEvidence(StrictModule, NonTrainableState):
    parameters_valid: Array
    montage_charge_neutral: Array
    source_charge_neutral: Array
    finite: Array
    real_signal_residual: Array
    zero_mode_removed: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    claim_scope: str = eqx.field(
        static=True,
        default="infinite planar muscle/fat/skin surface conductor; not intramuscular or limb geometry",
    )


class PlanarConductorResult(StrictModule, NonTrainableState):
    potential_spectrum_V: Array
    surface_voltage_V: Array
    evidence: PlanarConductorEvidence
    plan_id: str = eqx.field(static=True)


class PetersenRostalski2019PlanarConductorPlan(StrictModule):
    """Prepared Eq. 25--31 transfer on one discrete spatial-frequency grid."""

    frequency_x_rad_per_m: Array
    frequency_z_rad_per_m: Array
    electrode_transfer: Array
    electrode_positions_m: Array
    electrode_weights: Array
    parameters: PlanarConductorParameters
    zero_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency_x_rad_per_m: ArrayLike,
        frequency_z_rad_per_m: ArrayLike,
        electrode_transfer: ArrayLike,
        electrode_positions_m: ArrayLike,
        electrode_weights: ArrayLike,
        parameters: PlanarConductorParameters,
        /,
        *,
        zero_tolerance: float = 1.0e-10,
    ):
        fx = jnp.asarray(frequency_x_rad_per_m)
        fz = jnp.asarray(frequency_z_rad_per_m)
        transfer = jnp.asarray(electrode_transfer)
        positions = jnp.asarray(electrode_positions_m)
        weights = jnp.asarray(electrode_weights)
        if fx.ndim != 1 or fz.ndim != 1 or not fx.size or not fz.size:
            raise ValueError("Spatial-frequency axes must be nonempty vectors.")
        if transfer.shape != (fx.size, fz.size):
            raise ValueError("electrode_transfer must match the frequency grid.")
        if not (
            np.isclose(float(np.asarray(fx[0])), 0.0)
            and np.isclose(float(np.asarray(fz[0])), 0.0)
        ):
            raise ValueError("The first frequency on each axis must be the zero mode.")
        if positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError("electrode_positions_m must have shape (electrode, 2).")
        if weights.shape != (positions.shape[0],):
            raise ValueError("electrode_weights must match electrode positions.")
        if not isinstance(parameters, PlanarConductorParameters):
            raise TypeError("parameters must be PlanarConductorParameters.")
        tolerance = float(zero_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("zero_tolerance must be positive and finite.")
        if not (
            np.all(np.isfinite(np.asarray(fx)))
            and np.all(np.isfinite(np.asarray(fz)))
            and np.all(np.isfinite(np.asarray(transfer)))
            and np.all(np.isfinite(np.asarray(positions)))
            and np.all(np.isfinite(np.asarray(weights)))
        ):
            raise ValueError("Conductor grid and electrode data must be finite.")
        self.frequency_x_rad_per_m = fx
        self.frequency_z_rad_per_m = fz
        self.electrode_transfer = transfer
        self.electrode_positions_m = positions
        self.electrode_weights = weights
        self.parameters = parameters
        self.zero_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "petersen-rostalski-2019-planar-surface-conductor",
                "source_doi": PETERSEN_ROSTALSKI_2019_DOI,
                "dryad_doi": PETERSEN_ROSTALSKI_2019_DRYAD_DOI,
                "frequency_x": array_tree_fingerprint(fx),
                "frequency_z": array_tree_fingerprint(fz),
                "electrode_transfer": array_tree_fingerprint(transfer),
                "electrode_positions": array_tree_fingerprint(positions),
                "electrode_weights": array_tree_fingerprint(weights),
                "parameters": array_tree_fingerprint(parameters),
                "zero_tolerance": tolerance.hex(),
            }
        )

    def transfer_function_V_per_A(self, /) -> Array:
        p = self.parameters
        wx, wz = jnp.meshgrid(
            self.frequency_x_rad_per_m,
            self.frequency_z_rad_per_m,
            indexing="ij",
        )
        omega_y = jnp.sqrt(wx * wx + wz * wz)
        anisotropy = (
            p.muscle_longitudinal_conductivity_S_per_m
            / p.muscle_transverse_conductivity_S_per_m
        )
        omega_ya = jnp.sqrt(wx * wx + anisotropy * wz * wz)
        skin_fat_ratio = p.skin_conductivity_S_per_m / p.fat_conductivity_S_per_m
        fat_muscle_ratio = (
            p.fat_conductivity_S_per_m
            / p.muscle_transverse_conductivity_S_per_m
        )
        plus = omega_y * (p.fat_thickness_m + p.skin_thickness_m)
        minus = omega_y * (p.fat_thickness_m - p.skin_thickness_m)

        def nu(value):
            return omega_ya + value * fat_muscle_ratio * jnp.tanh(value)

        denominator = (
            (1.0 + skin_fat_ratio) * jnp.cosh(plus) * nu(plus)
            + (1.0 - skin_fat_ratio) * jnp.cosh(minus) * nu(minus)
        )
        nonzero = omega_y > self.zero_tolerance
        safe_denominator = jnp.where(nonzero, denominator, 1.0)
        volume = (
            2.0
            / p.muscle_transverse_conductivity_S_per_m
            * jnp.exp(-omega_ya * jnp.abs(p.source_depth_m))
            / safe_denominator
        )
        volume = jnp.where(nonzero, volume, 0.0)
        phase = -1j * (
            wx[..., None] * self.electrode_positions_m[:, 0]
            + wz[..., None] * self.electrode_positions_m[:, 1]
        )
        montage = jnp.sum(
            self.electrode_weights[None, None, :] * jnp.exp(phase), axis=-1
        )
        return volume * self.electrode_transfer * montage

    def evaluate(
        self, source_current_spectrum_A: ArrayLike, /
    ) -> PlanarConductorResult:
        source = jnp.asarray(source_current_spectrum_A)
        expected = (
            self.frequency_x_rad_per_m.size,
            self.frequency_z_rad_per_m.size,
        )
        if source.shape != expected:
            raise ValueError(f"source_current_spectrum_A must have shape {expected}.")
        transfer = self.transfer_function_V_per_A()
        potential = source * transfer
        spatial_potential = jnp.fft.ifft2(potential)
        voltage = jnp.real(spatial_potential)
        p = self.parameters
        parameter_values = jnp.stack(
            (
                p.muscle_longitudinal_conductivity_S_per_m,
                p.muscle_transverse_conductivity_S_per_m,
                p.fat_conductivity_S_per_m,
                p.skin_conductivity_S_per_m,
                p.fat_thickness_m,
                p.skin_thickness_m,
                p.source_depth_m,
            )
        )
        parameters_valid = (
            jnp.all(jnp.isfinite(parameter_values))
            & jnp.all(parameter_values[:6] > 0.0)
            & (p.source_depth_m <= 0.0)
        )
        montage_neutral = (
            jnp.abs(jnp.sum(self.electrode_weights)) <= self.zero_tolerance
        )
        source_neutral = jnp.abs(source[0, 0]) <= self.zero_tolerance
        finite = jnp.all(jnp.isfinite(potential)) & jnp.all(
            jnp.isfinite(voltage)
        )
        zero_removed = jnp.abs(transfer[0, 0]) <= self.zero_tolerance
        real_signal_residual = jnp.max(jnp.abs(jnp.imag(spatial_potential)))
        successful = (
            parameters_valid
            & montage_neutral
            & source_neutral
            & finite
            & zero_removed
            & (real_signal_residual <= self.zero_tolerance)
        )
        evidence = PlanarConductorEvidence(
            parameters_valid,
            montage_neutral,
            source_neutral,
            finite,
            real_signal_residual,
            zero_removed,
            successful,
            self.plan_id,
        )
        return PlanarConductorResult(potential, voltage, evidence, self.plan_id)


__all__ = [
    "PETERSEN_ROSTALSKI_2019_DOI",
    "PETERSEN_ROSTALSKI_2019_DRYAD_DOI",
    "PetersenRostalski2019PlanarConductorPlan",
    "PlanarConductorEvidence",
    "PlanarConductorParameters",
    "PlanarConductorResult",
]
