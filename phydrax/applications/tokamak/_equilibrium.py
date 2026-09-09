#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical axisymmetric tokamak equilibrium state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._conventions import AxisymmetricMachineFrame, TokamakMagneticConvention


def _real_array(value, name: str, shape: tuple[int, ...] | None = None):
    array = np.array(value, dtype=np.float64, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


class PreparedAxisymmetricEquilibrium(StrictModule, NonTrainableState):
    r_m: Array
    z_m: Array
    poloidal_flux_wb_per_rad: Array
    f_rb_t_m: Array
    pressure_pa: Array
    ffprime_t2_m2_per_wb_rad: Array
    pprime_pa_per_wb_rad: Array
    safety_factor: Array
    boundary_rz_m: Array
    limiter_rz_m: Array
    magnetic_axis_rz_m: Array
    magnetic_axis_flux_wb_per_rad: Array
    boundary_flux_wb_per_rad: Array
    reference_major_radius_m: Array
    reference_toroidal_field_t: Array
    plasma_current_a: Array
    frame_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    equilibrium_id: str = eqx.field(static=True)

    def normalized_flux(self) -> Array:
        denominator = self.boundary_flux_wb_per_rad - self.magnetic_axis_flux_wb_per_rad
        return (
            self.poloidal_flux_wb_per_rad - self.magnetic_axis_flux_wb_per_rad
        ) / denominator


@dataclass(frozen=True, slots=True)
class AxisymmetricEquilibrium:
    """Canonical R-Z equilibrium with source and convention identities."""

    machine_frame: AxisymmetricMachineFrame
    convention: TokamakMagneticConvention
    r_m: np.ndarray
    z_m: np.ndarray
    poloidal_flux_wb_per_rad: np.ndarray
    f_rb_t_m: np.ndarray
    pressure_pa: np.ndarray
    ffprime_t2_m2_per_wb_rad: np.ndarray
    pprime_pa_per_wb_rad: np.ndarray
    safety_factor: np.ndarray
    boundary_rz_m: np.ndarray
    limiter_rz_m: np.ndarray
    magnetic_axis_rz_m: np.ndarray
    magnetic_axis_flux_wb_per_rad: float
    boundary_flux_wb_per_rad: float
    reference_major_radius_m: float
    reference_toroidal_field_t: float
    plasma_current_a: float
    source_id: str
    equilibrium_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.machine_frame, AxisymmetricMachineFrame):
            raise TypeError("machine_frame must be AxisymmetricMachineFrame.")
        if not isinstance(self.convention, TokamakMagneticConvention):
            raise TypeError("convention must be TokamakMagneticConvention.")
        r = _real_array(self.r_m, "r_m")
        z = _real_array(self.z_m, "z_m")
        if r.ndim != 1 or z.ndim != 1 or r.size < 2 or z.size < 2:
            raise ValueError("r_m and z_m must be nontrivial rank-one coordinate arrays.")
        if np.any(r <= 0.0) or np.any(np.diff(r) <= 0.0) or np.any(np.diff(z) <= 0.0):
            raise ValueError("Equilibrium coordinates require R > 0 and strict ordering.")
        psi = _real_array(
            self.poloidal_flux_wb_per_rad,
            "poloidal_flux_wb_per_rad",
            (z.size, r.size),
        )
        profiles = tuple(
            _real_array(value, name, (r.size,))
            for value, name in (
                (self.f_rb_t_m, "f_rb_t_m"),
                (self.pressure_pa, "pressure_pa"),
                (self.ffprime_t2_m2_per_wb_rad, "ffprime_t2_m2_per_wb_rad"),
                (self.pprime_pa_per_wb_rad, "pprime_pa_per_wb_rad"),
                (self.safety_factor, "safety_factor"),
            )
        )
        boundary = _real_array(self.boundary_rz_m, "boundary_rz_m")
        limiter = _real_array(self.limiter_rz_m, "limiter_rz_m")
        if boundary.ndim != 2 or boundary.shape[1:] != (2,):
            raise ValueError("boundary_rz_m must have shape (point_count, 2).")
        if limiter.ndim != 2 or limiter.shape[1:] != (2,):
            raise ValueError("limiter_rz_m must have shape (point_count, 2).")
        axis = _real_array(self.magnetic_axis_rz_m, "magnetic_axis_rz_m", (2,))
        if not (r[0] <= axis[0] <= r[-1] and z[0] <= axis[1] <= z[-1]):
            raise ValueError("Magnetic axis must lie inside the R-Z grid.")
        if boundary.size and (
            np.any(boundary[:, 0] < r[0])
            or np.any(boundary[:, 0] > r[-1])
            or np.any(boundary[:, 1] < z[0])
            or np.any(boundary[:, 1] > z[-1])
        ):
            raise ValueError("Plasma boundary must lie inside the R-Z grid.")
        scalars = tuple(
            float(value)
            for value in (
                self.magnetic_axis_flux_wb_per_rad,
                self.boundary_flux_wb_per_rad,
                self.reference_major_radius_m,
                self.reference_toroidal_field_t,
                self.plasma_current_a,
            )
        )
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("Equilibrium scalar values must be finite.")
        if scalars[2] <= 0.0:
            raise ValueError("reference_major_radius_m must be positive.")
        if scalars[0] == scalars[1]:
            raise ValueError("Magnetic-axis and boundary flux must differ.")
        source = str(self.source_id).strip()
        if not source or source != self.source_id:
            raise ValueError("source_id must be non-empty canonical text.")
        for name, value in (
            ("r_m", r),
            ("z_m", z),
            ("poloidal_flux_wb_per_rad", psi),
            ("f_rb_t_m", profiles[0]),
            ("pressure_pa", profiles[1]),
            ("ffprime_t2_m2_per_wb_rad", profiles[2]),
            ("pprime_pa_per_wb_rad", profiles[3]),
            ("safety_factor", profiles[4]),
            ("boundary_rz_m", boundary),
            ("limiter_rz_m", limiter),
            ("magnetic_axis_rz_m", axis),
        ):
            object.__setattr__(self, name, value)
        (
            axis_flux,
            boundary_flux,
            reference_radius,
            reference_field,
            plasma_current,
        ) = scalars
        object.__setattr__(self, "magnetic_axis_flux_wb_per_rad", axis_flux)
        object.__setattr__(self, "boundary_flux_wb_per_rad", boundary_flux)
        object.__setattr__(self, "reference_major_radius_m", reference_radius)
        object.__setattr__(self, "reference_toroidal_field_t", reference_field)
        object.__setattr__(self, "plasma_current_a", plasma_current)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(
            self,
            "equilibrium_id",
            canonical_fingerprint(
                {
                    "kind": "axisymmetric-equilibrium",
                    "frame": self.machine_frame.frame_id,
                    "convention": self.convention.convention_id,
                    "r_m": array_tree_fingerprint(r),
                    "z_m": array_tree_fingerprint(z),
                    "poloidal_flux": array_tree_fingerprint(psi),
                    "profiles": [array_tree_fingerprint(value) for value in profiles],
                    "boundary": array_tree_fingerprint(boundary),
                    "limiter": array_tree_fingerprint(limiter),
                    "axis": array_tree_fingerprint(axis),
                    "scalars": list(scalars),
                    "source": source,
                }
            ),
        )

    @property
    def normalized_flux(self) -> np.ndarray:
        values = (self.poloidal_flux_wb_per_rad - self.magnetic_axis_flux_wb_per_rad) / (
            self.boundary_flux_wb_per_rad - self.magnetic_axis_flux_wb_per_rad
        )
        values = np.asarray(values)
        values.setflags(write=False)
        return values

    def prepare(self) -> PreparedAxisymmetricEquilibrium:
        return PreparedAxisymmetricEquilibrium(
            jnp.asarray(self.r_m),
            jnp.asarray(self.z_m),
            jnp.asarray(self.poloidal_flux_wb_per_rad),
            jnp.asarray(self.f_rb_t_m),
            jnp.asarray(self.pressure_pa),
            jnp.asarray(self.ffprime_t2_m2_per_wb_rad),
            jnp.asarray(self.pprime_pa_per_wb_rad),
            jnp.asarray(self.safety_factor),
            jnp.asarray(self.boundary_rz_m),
            jnp.asarray(self.limiter_rz_m),
            jnp.asarray(self.magnetic_axis_rz_m),
            jnp.asarray(self.magnetic_axis_flux_wb_per_rad),
            jnp.asarray(self.boundary_flux_wb_per_rad),
            jnp.asarray(self.reference_major_radius_m),
            jnp.asarray(self.reference_toroidal_field_t),
            jnp.asarray(self.plasma_current_a),
            self.machine_frame.frame_id,
            self.convention.convention_id,
            self.equilibrium_id,
        )


__all__ = ["AxisymmetricEquilibrium", "PreparedAxisymmetricEquilibrium"]
