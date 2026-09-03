#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import FaceVelocity, PreparedMACOperators
from ...discretization.spectral._fourier_shells import _FourierShellBinGeometry
from ...discretization.spectral._incompressible import PeriodicLerayProjector
from ...discretization.spectral._space import TensorSpectralDiscretization
from ._forcing import _hermitian_defect, _periodic_modal_geometry


class ModalShellStatistic(StrictModule):
    representative_wavenumbers: Array
    bin_edges: Array
    bin_widths: Array
    integral: Array
    density: Array
    valid_shells: Array
    total: Array
    finite: Array
    statistic_kind: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


def _modal_shell_statistic(
    geometry: _FourierShellBinGeometry,
    mode_values: Array,
    statistic_kind: str,
    /,
) -> ModalShellStatistic:
    integral = jnp.real(geometry.reduce_integral(mode_values))
    integral = jnp.where(geometry.valid_shells, integral, 0.0)
    density = jnp.where(
        geometry.valid_shells,
        integral / geometry.bin_widths.astype(integral.dtype),
        0.0,
    )
    total = jnp.real(geometry.total_integral(mode_values))
    finite = (
        jnp.all(jnp.isfinite(integral))
        & jnp.all(jnp.isfinite(density))
        & jnp.isfinite(total)
    )
    return ModalShellStatistic(
        representative_wavenumbers=geometry.representative_wavenumbers,
        bin_edges=geometry.bin_edges,
        bin_widths=geometry.bin_widths,
        integral=integral,
        density=density,
        valid_shells=geometry.valid_shells,
        total=total,
        finite=finite,
        statistic_kind=statistic_kind,
        geometry_id=geometry.geometry_id,
    )


class PeriodicModalTurbulenceStatistics(StrictModule):
    energy_shells: ModalShellStatistic
    dissipation_shells: ModalShellStatistic
    nonlinear_transfer_shells: ModalShellStatistic
    forcing_injection_shells: ModalShellStatistic
    kinetic_energy: Array
    mean_kinetic_energy: Array
    dissipation: Array
    mean_dissipation: Array
    nonlinear_energy_rate: Array
    mean_nonlinear_energy_rate: Array
    forcing_power: Array
    mean_forcing_power: Array
    enstrophy: Array
    mean_enstrophy: Array
    helicity: Array
    mean_helicity: Array
    taylor_microscale: Array
    kolmogorov_scale: Array
    kmax_kolmogorov: Array
    integral_scale: Array
    energy_tail_fraction: Array
    dissipation_tail_fraction: Array
    divergence_norm: Array
    velocity_reality_defect: Array
    transfer_available: Array
    forcing_available: Array
    helicity_valid: Array
    taylor_microscale_valid: Array
    kolmogorov_scale_valid: Array
    integral_scale_valid: Array
    energy_tail_valid: Array
    dissipation_tail_valid: Array
    finite: Array
    successful: Array
    tail_start_wavenumber: float = eqx.field(static=True)
    spectrum_convention: str = eqx.field(static=True)
    integral_scale_convention: str = eqx.field(static=True)
    tail_convention: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PeriodicModalTurbulenceStatisticsPlan(StrictModule, NonTrainableState):
    """Conservative full-complex periodic turbulence statistics.

    Shell ``integral`` values are native domain integrals and ``density`` is
    the integral divided by shell wavenumber width. In three dimensions the
    reported integral scale is ``3 pi sum(E_k / |k|) / (4 sum(E_k))`` and is
    valid only for a negligible zero mode. Tail fractions use modes at or
    above the prepared ``tail_start_wavenumber``.
    """

    projector: PeriodicLerayProjector
    geometry: _FourierShellBinGeometry
    conjugate_indices: Array
    viscosity: float = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    maximum_admissible_wavenumber: float = eqx.field(static=True)
    tail_start_wavenumber: float = eqx.field(static=True)
    reality_tolerance: float = eqx.field(static=True)
    solenoidal_tolerance: float = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        projector: PeriodicLerayProjector,
        bin_edges: ArrayLike,
        /,
        *,
        viscosity: float,
        tail_start_wavenumber: float | None = None,
        reality_tolerance: float = 1.0e-10,
        solenoidal_tolerance: float = 1.0e-10,
    ):
        if not isinstance(projector, PeriodicLerayProjector):
            raise TypeError("projector must be a PeriodicLerayProjector.")
        viscosity_ = float(viscosity)
        reality = float(reality_tolerance)
        solenoidal = float(solenoidal_tolerance)
        edges = np.asarray(bin_edges, dtype=float).reshape((-1,))
        if (
            not np.isfinite(viscosity_)
            or viscosity_ < 0.0
            or not np.isfinite(reality)
            or reality < 0.0
            or not np.isfinite(solenoidal)
            or solenoidal < 0.0
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("Periodic turbulence-statistics parameters are invalid.")
        magnitude, admissible, conjugates, volume = _periodic_modal_geometry(projector)
        maximum_wave = float(np.max(magnitude[admissible]))
        if edges[0] > 0.0 or edges[-1] < maximum_wave:
            raise ValueError(
                "Shell edges must cover zero through every admissible full-complex mode."
            )
        tail_start = (
            (2.0 / 3.0) * maximum_wave
            if tail_start_wavenumber is None
            else float(tail_start_wavenumber)
        )
        if not np.isfinite(tail_start) or tail_start < 0.0 or tail_start > maximum_wave:
            raise ValueError("tail_start_wavenumber must lie in the resolved range.")
        geometry = _FourierShellBinGeometry(
            magnitude,
            edges,
            mode_mask=admissible,
            mode_weights=np.ones(magnitude.shape, dtype=float),
            final_edge_policy="include",
            source_id=f"full-complex:{projector.projector_id}",
        )
        self.projector = projector
        self.geometry = geometry
        self.conjugate_indices = jnp.asarray(conjugates, dtype=jnp.int32)
        self.viscosity = viscosity_
        self.volume = volume
        self.maximum_admissible_wavenumber = maximum_wave
        self.tail_start_wavenumber = tail_start
        self.reality_tolerance = reality
        self.solenoidal_tolerance = solenoidal
        self.discretization_id = projector.discretization.prepared_id
        self.projector_id = projector.projector_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-modal-turbulence-statistics",
                "discretization": self.discretization_id,
                "projector": self.projector_id,
                "shell_geometry": geometry.geometry_id,
                "viscosity": viscosity_,
                "tail_start_wavenumber": tail_start,
                "tail_policy": (
                    "upper-admissible-third"
                    if tail_start_wavenumber is None
                    else "declared-wavenumber"
                ),
                "reality_tolerance": reality,
                "solenoidal_tolerance": solenoidal,
                "storage": "full-complex-no-hermitian-multiplicity",
            }
        )

    def evaluate(
        self,
        velocity: ArrayLike,
        /,
        *,
        nonlinear_rate: ArrayLike | None = None,
        forcing: ArrayLike | None = None,
    ) -> PeriodicModalTurbulenceStatistics:
        value = self.projector.validate_state(velocity)
        finite_velocity = jnp.all(jnp.isfinite(value))
        clean_velocity = jnp.where(finite_velocity, value, jnp.zeros_like(value))
        velocity_ = self.projector.zero_forbidden_modes(clean_velocity)
        velocity_reality_defect = _hermitian_defect(velocity_, self.conjugate_indices)
        divergence_norm = self.projector.divergence_norm(velocity_)
        transfer_available = jnp.asarray(nonlinear_rate is not None)
        if nonlinear_rate is None:
            nonlinear = jnp.zeros_like(velocity_)
            finite_nonlinear = jnp.asarray(True)
        else:
            nonlinear_value = self.projector.validate_state(
                nonlinear_rate, owner="Nonlinear rate"
            )
            finite_nonlinear = jnp.all(jnp.isfinite(nonlinear_value))
            nonlinear = self.projector.zero_forbidden_modes(
                jnp.where(
                    finite_nonlinear, nonlinear_value, jnp.zeros_like(nonlinear_value)
                )
            )
        forcing_available = jnp.asarray(forcing is not None)
        if forcing is None:
            forcing_ = jnp.zeros_like(velocity_)
            finite_forcing = jnp.asarray(True)
        else:
            forcing_value = self.projector.validate_state(forcing, owner="Forcing")
            finite_forcing = jnp.all(jnp.isfinite(forcing_value))
            forcing_ = self.projector.zero_forbidden_modes(
                jnp.where(finite_forcing, forcing_value, jnp.zeros_like(forcing_value))
            )
        modal_energy = 0.5 * jnp.sum(jnp.abs(velocity_) ** 2, axis=-1)
        modal_dissipation = (
            2.0
            * self.viscosity
            * self.projector.wavenumber_squared.astype(modal_energy.dtype)
            * modal_energy
        )
        modal_transfer = jnp.real(
            ein.contract("...i,...i->...", jnp.conj(velocity_), nonlinear)
        )
        modal_injection = jnp.real(
            ein.contract("...i,...i->...", jnp.conj(velocity_), forcing_)
        )
        energy_shells = _modal_shell_statistic(
            self.geometry, modal_energy, "kinetic-energy"
        )
        dissipation_shells = _modal_shell_statistic(
            self.geometry, modal_dissipation, "dissipation"
        )
        nonlinear_shells = _modal_shell_statistic(
            self.geometry, modal_transfer, "nonlinear-transfer"
        )
        forcing_shells = _modal_shell_statistic(
            self.geometry, modal_injection, "forcing-injection"
        )
        kinetic_energy = energy_shells.total
        dissipation = dissipation_shells.total
        nonlinear_energy_rate = nonlinear_shells.total
        forcing_power = forcing_shells.total
        waves = tuple(
            wave.astype(velocity_.real.dtype) for wave in self.projector.wavenumbers
        )
        if self.projector.spatial_dimension == 3:
            vorticity = 1j * jnp.stack(
                (
                    waves[1] * velocity_[..., 2] - waves[2] * velocity_[..., 1],
                    waves[2] * velocity_[..., 0] - waves[0] * velocity_[..., 2],
                    waves[0] * velocity_[..., 1] - waves[1] * velocity_[..., 0],
                ),
                axis=-1,
            )
            helicity = jnp.real(
                ein.contract("...i,...i->", jnp.conj(velocity_), vorticity)
            )
            helicity_valid = jnp.asarray(True)
        else:
            vorticity = 1j * (waves[0] * velocity_[..., 1] - waves[1] * velocity_[..., 0])
            helicity = jnp.asarray(0.0, dtype=velocity_.real.dtype)
            helicity_valid = jnp.asarray(False)
        enstrophy = 0.5 * jnp.sum(jnp.abs(vorticity) ** 2)
        mean_energy = kinetic_energy / self.volume
        mean_dissipation = dissipation / self.volume
        mean_enstrophy = enstrophy / self.volume
        mean_helicity = helicity / self.volume
        scale_valid = (
            (self.projector.spatial_dimension == 3)
            & (self.viscosity > 0.0)
            & (mean_energy > 0.0)
            & (mean_dissipation > 0.0)
        )
        safe_mean_dissipation = jnp.where(mean_dissipation > 0.0, mean_dissipation, 1.0)
        taylor = jnp.where(
            scale_valid,
            jnp.sqrt(10.0 * self.viscosity * mean_energy / safe_mean_dissipation),
            0.0,
        )
        kolmogorov = jnp.where(
            scale_valid,
            (self.viscosity**3 / safe_mean_dissipation) ** 0.25,
            0.0,
        )
        magnitude = self.geometry.wavenumber_magnitude.astype(modal_energy.dtype)
        nonzero = magnitude > 0.0
        inverse_wave = jnp.where(nonzero, 1.0 / magnitude, 0.0)
        zero_energy = jnp.sum(jnp.where(nonzero, 0.0, modal_energy))
        integral_valid = (
            (self.projector.spatial_dimension == 3)
            & (kinetic_energy > 0.0)
            & (zero_energy <= self.reality_tolerance * jnp.maximum(kinetic_energy, 1.0))
        )
        safe_energy = jnp.where(kinetic_energy > 0.0, kinetic_energy, 1.0)
        integral_scale = jnp.where(
            integral_valid,
            3.0 * jnp.pi * jnp.sum(modal_energy * inverse_wave) / (4.0 * safe_energy),
            0.0,
        )
        tail = magnitude >= self.tail_start_wavenumber
        tail_energy = jnp.sum(jnp.where(tail, modal_energy, 0.0))
        tail_dissipation = jnp.sum(jnp.where(tail, modal_dissipation, 0.0))
        energy_tail_valid = kinetic_energy > 0.0
        dissipation_tail_valid = dissipation > 0.0
        energy_tail_fraction = jnp.where(
            energy_tail_valid, tail_energy / safe_energy, 0.0
        )
        safe_dissipation = jnp.where(dissipation > 0.0, dissipation, 1.0)
        dissipation_tail_fraction = jnp.where(
            dissipation_tail_valid, tail_dissipation / safe_dissipation, 0.0
        )
        finite = (
            finite_velocity
            & finite_nonlinear
            & finite_forcing
            & energy_shells.finite
            & dissipation_shells.finite
            & nonlinear_shells.finite
            & forcing_shells.finite
            & jnp.isfinite(enstrophy)
            & jnp.isfinite(helicity)
            & jnp.isfinite(taylor)
            & jnp.isfinite(kolmogorov)
            & jnp.isfinite(integral_scale)
            & jnp.isfinite(energy_tail_fraction)
            & jnp.isfinite(dissipation_tail_fraction)
        )
        successful = (
            finite
            & (velocity_reality_defect <= self.reality_tolerance)
            & (divergence_norm <= self.solenoidal_tolerance)
        )
        return PeriodicModalTurbulenceStatistics(
            energy_shells=energy_shells,
            dissipation_shells=dissipation_shells,
            nonlinear_transfer_shells=nonlinear_shells,
            forcing_injection_shells=forcing_shells,
            kinetic_energy=kinetic_energy,
            mean_kinetic_energy=mean_energy,
            dissipation=dissipation,
            mean_dissipation=mean_dissipation,
            nonlinear_energy_rate=nonlinear_energy_rate,
            mean_nonlinear_energy_rate=nonlinear_energy_rate / self.volume,
            forcing_power=forcing_power,
            mean_forcing_power=forcing_power / self.volume,
            enstrophy=enstrophy,
            mean_enstrophy=mean_enstrophy,
            helicity=helicity,
            mean_helicity=mean_helicity,
            taylor_microscale=taylor,
            kolmogorov_scale=kolmogorov,
            kmax_kolmogorov=self.maximum_admissible_wavenumber * kolmogorov,
            integral_scale=integral_scale,
            energy_tail_fraction=energy_tail_fraction,
            dissipation_tail_fraction=dissipation_tail_fraction,
            divergence_norm=divergence_norm,
            velocity_reality_defect=velocity_reality_defect,
            transfer_available=transfer_available,
            forcing_available=forcing_available,
            helicity_valid=helicity_valid,
            taylor_microscale_valid=scale_valid,
            kolmogorov_scale_valid=scale_valid,
            integral_scale_valid=integral_valid,
            energy_tail_valid=energy_tail_valid,
            dissipation_tail_valid=dissipation_tail_valid,
            finite=finite,
            successful=successful,
            tail_start_wavenumber=self.tail_start_wavenumber,
            spectrum_convention=(
                "full-complex native domain integral; density=integral/bin-width"
            ),
            integral_scale_convention="3*pi*sum(E_k/|k|)/(4*sum(E_k))",
            tail_convention="modes with |k| >= tail_start_wavenumber",
            discretization_id=self.discretization_id,
            projector_id=self.projector_id,
            plan_id=self.plan_id,
        )


class SpectralChannelStatistics(StrictModule):
    wall_normal_coordinates: Array
    mean_streamwise_velocity: Array
    mean_wall_normal_velocity: Array
    mean_spanwise_velocity: Array
    raw_uu: Array
    raw_vv: Array
    raw_ww: Array
    raw_uv: Array
    raw_uw: Array
    raw_vw: Array
    reynolds_uu: Array
    reynolds_vv: Array
    reynolds_ww: Array
    reynolds_uv: Array
    reynolds_uw: Array
    reynolds_vw: Array
    lower_wall_shear: Array
    upper_wall_shear: Array
    bulk_velocity: Array
    lower_friction_velocity: Array
    upper_friction_velocity: Array
    lower_friction_reynolds: Array
    upper_friction_reynolds: Array
    lower_wall_coordinates: Array
    upper_wall_coordinates: Array
    imaginary_leakage: Array
    finite: Array
    successful: Array
    wall_shear_convention: str = eqx.field(static=True)
    wall_length_convention: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class SpectralChannelStatisticsPlan(StrictModule, NonTrainableState):
    """Instantaneous homogeneous-plane and separate-wall channel statistics."""

    discretization: TensorSpectralDiscretization
    wall_normal_coordinates: Array
    wall_quadrature_weights: Array
    wall_normal_axis: int = eqx.field(static=True)
    homogeneous_axes: tuple[int, int] = eqx.field(static=True)
    lower_wall_index: int = eqx.field(static=True)
    upper_wall_index: int = eqx.field(static=True)
    density: float = eqx.field(static=True)
    kinematic_viscosity: float = eqx.field(static=True)
    plane_area: float = eqx.field(static=True)
    channel_height: float = eqx.field(static=True)
    half_height: float = eqx.field(static=True)
    reality_tolerance: float = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        density: float,
        kinematic_viscosity: float,
        wall_normal_axis: int = 1,
        reality_tolerance: float = 1.0e-10,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        if len(discretization.axes) != 3:
            raise ValueError("Spectral channel statistics require three dimensions.")
        wall_axis = int(wall_normal_axis)
        if wall_axis < 0 or wall_axis >= 3:
            raise ValueError("wall_normal_axis must identify one channel axis.")
        homogeneous = tuple(axis for axis in range(3) if axis != wall_axis)
        wall = discretization.axes[wall_axis]
        density_ = float(density)
        viscosity = float(kinematic_viscosity)
        tolerance = float(reality_tolerance)
        if (
            wall.family == "fourier"
            or any(discretization.axes[axis].family != "fourier" for axis in homogeneous)
            or not wall.lower_endpoint_included
            or not wall.upper_endpoint_included
            or not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(viscosity)
            or viscosity <= 0.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Spectral channel geometry or material data is invalid.")
        nodes = np.asarray(wall.nodes, dtype=float)
        lower_index = int(np.argmin(nodes))
        upper_index = int(np.argmax(nodes))
        height = float(nodes[upper_index] - nodes[lower_index])
        plane_area = float(
            np.prod([float(discretization.axes[axis].length) for axis in homogeneous])
        )
        if height <= 0.0 or not np.isfinite(plane_area) or plane_area <= 0.0:
            raise ValueError("Spectral channel measures must be finite and positive.")
        self.discretization = discretization
        self.wall_normal_coordinates = wall.nodes
        self.wall_quadrature_weights = wall.quadrature_weights
        self.wall_normal_axis = wall_axis
        self.homogeneous_axes = homogeneous
        self.lower_wall_index = lower_index
        self.upper_wall_index = upper_index
        self.density = density_
        self.kinematic_viscosity = viscosity
        self.plane_area = plane_area
        self.channel_height = height
        self.half_height = 0.5 * height
        self.reality_tolerance = tolerance
        self.discretization_id = discretization.prepared_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-channel-statistics",
                "discretization": self.discretization_id,
                "wall_normal_axis": wall_axis,
                "homogeneous_axes": list(homogeneous),
                "velocity_components": {
                    "streamwise": homogeneous[0],
                    "wall_normal": wall_axis,
                    "spanwise": homogeneous[1],
                },
                "density": density_,
                "kinematic_viscosity": viscosity,
                "wall_shear": "rho*nu*d<streamwise-velocity>/dy",
                "wall_length": "half-height-separate-walls",
                "reality_tolerance": tolerance,
            }
        )

    def _plane_mean(self, values: Array, /) -> Array:
        return (
            self.discretization.integral(values, axes=self.homogeneous_axes)
            / self.plane_area
        )

    def evaluate(self, modal_velocity: ArrayLike, /) -> SpectralChannelStatistics:
        value = jnp.asarray(modal_velocity)
        expected = self.discretization.modal_shape + (3,)
        if value.shape != expected:
            raise ValueError(f"Channel velocity must have modal shape {expected}.")
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Channel modal velocity must be complex-valued.")
        finite_input = jnp.all(jnp.isfinite(value))
        clean = jnp.where(finite_input, value, jnp.zeros_like(value))
        complex_velocity = self.discretization.reconstruct(clean, real_output=False)
        imaginary_leakage = jnp.max(jnp.abs(jnp.imag(complex_velocity)), initial=0.0)
        velocity = jnp.real(complex_velocity)
        mean = self._plane_mean(velocity)
        streamwise, spanwise = self.homogeneous_axes
        wall_normal = self.wall_normal_axis
        u = velocity[..., streamwise]
        v = velocity[..., wall_normal]
        w = velocity[..., spanwise]
        raw_uu = self._plane_mean(u * u)
        raw_vv = self._plane_mean(v * v)
        raw_ww = self._plane_mean(w * w)
        raw_uv = self._plane_mean(u * v)
        raw_uw = self._plane_mean(u * w)
        raw_vw = self._plane_mean(v * w)
        mean_u = mean[..., streamwise]
        mean_v = mean[..., wall_normal]
        mean_w = mean[..., spanwise]
        derivative = self.discretization.partial_derivative(u, axis=self.wall_normal_axis)
        mean_derivative = self._plane_mean(derivative)
        dynamic_viscosity = self.density * self.kinematic_viscosity
        lower_shear = dynamic_viscosity * mean_derivative[self.lower_wall_index]
        upper_shear = dynamic_viscosity * mean_derivative[self.upper_wall_index]
        bulk = jnp.sum(self.wall_quadrature_weights * mean_u) / self.channel_height
        lower_friction = jnp.sqrt(jnp.abs(lower_shear) / self.density)
        upper_friction = jnp.sqrt(jnp.abs(upper_shear) / self.density)
        lower_reynolds = lower_friction * self.half_height / self.kinematic_viscosity
        upper_reynolds = upper_friction * self.half_height / self.kinematic_viscosity
        lower_position = self.wall_normal_coordinates[self.lower_wall_index]
        upper_position = self.wall_normal_coordinates[self.upper_wall_index]
        lower_coordinates = (
            (self.wall_normal_coordinates - lower_position)
            * lower_friction
            / self.kinematic_viscosity
        )
        upper_coordinates = (
            (upper_position - self.wall_normal_coordinates)
            * upper_friction
            / self.kinematic_viscosity
        )
        finite = (
            finite_input
            & jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(raw_uu))
            & jnp.all(jnp.isfinite(raw_vv))
            & jnp.all(jnp.isfinite(raw_ww))
            & jnp.all(jnp.isfinite(raw_uv))
            & jnp.all(jnp.isfinite(raw_uw))
            & jnp.all(jnp.isfinite(raw_vw))
            & jnp.isfinite(lower_shear)
            & jnp.isfinite(upper_shear)
            & jnp.isfinite(bulk)
            & jnp.all(jnp.isfinite(lower_coordinates))
            & jnp.all(jnp.isfinite(upper_coordinates))
        )
        successful = finite & (imaginary_leakage <= self.reality_tolerance)
        return SpectralChannelStatistics(
            wall_normal_coordinates=self.wall_normal_coordinates,
            mean_streamwise_velocity=mean_u,
            mean_wall_normal_velocity=mean_v,
            mean_spanwise_velocity=mean_w,
            raw_uu=raw_uu,
            raw_vv=raw_vv,
            raw_ww=raw_ww,
            raw_uv=raw_uv,
            raw_uw=raw_uw,
            raw_vw=raw_vw,
            reynolds_uu=raw_uu - mean_u * mean_u,
            reynolds_vv=raw_vv - mean_v * mean_v,
            reynolds_ww=raw_ww - mean_w * mean_w,
            reynolds_uv=raw_uv - mean_u * mean_v,
            reynolds_uw=raw_uw - mean_u * mean_w,
            reynolds_vw=raw_vw - mean_v * mean_w,
            lower_wall_shear=lower_shear,
            upper_wall_shear=upper_shear,
            bulk_velocity=bulk,
            lower_friction_velocity=lower_friction,
            upper_friction_velocity=upper_friction,
            lower_friction_reynolds=lower_reynolds,
            upper_friction_reynolds=upper_reynolds,
            lower_wall_coordinates=lower_coordinates,
            upper_wall_coordinates=upper_coordinates,
            imaginary_leakage=imaginary_leakage,
            finite=finite,
            successful=successful,
            wall_shear_convention=(
                "tau_xy=rho*nu*d<streamwise-velocity>/dy in increasing y"
            ),
            wall_length_convention="half-height, evaluated separately at each wall",
            discretization_id=self.discretization_id,
            plan_id=self.plan_id,
        )


class MACPlaneWallStatistics(StrictModule):
    """Raw volume-weighted plane and separate-wall statistics on a MAC grid."""

    wall_normal_coordinates: Array
    plane_weights: Array
    mean_velocity: Array
    raw_second_moment: Array
    reynolds_stress: Array
    lower_wall_shear: Array
    upper_wall_shear: Array
    bulk_velocity: Array
    lower_wall_normal_velocity: Array
    upper_wall_normal_velocity: Array
    kinetic_energy: Array
    mean_kinetic_energy: Array
    forcing_power: Array
    mean_forcing_power: Array
    divergence_norm: Array
    finite: Array
    successful: Array
    wall_normal_axis: int = eqx.field(static=True)
    streamwise_axis: int = eqx.field(static=True)
    face_to_cell_convention: str = eqx.field(static=True)
    plane_weight_convention: str = eqx.field(static=True)
    wall_shear_convention: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    operators_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _mac_face_to_cell(
    component: Array,
    axis: int,
    periodic: bool,
    /,
) -> Array:
    moved = jnp.moveaxis(component, axis, 0)
    centered = (
        0.5 * (moved + jnp.roll(moved, -1, axis=0))
        if periodic
        else 0.5 * (moved[:-1] + moved[1:])
    )
    return jnp.moveaxis(centered, 0, axis)


class MACPlaneWallStatisticsPlan(StrictModule, NonTrainableState):
    """Staggering-native MAC plane profiles and raw wall evidence.

    Face-normal velocity components are arithmetically centered to their
    adjacent cells. Homogeneous-plane reductions use the exact cell volumes,
    and wall-normal velocities retain their native boundary-face values.
    """

    operators: PreparedMACOperators
    wall_normal_coordinates: Array
    cell_volumes: Array
    lower_wall_velocity: Array
    upper_wall_velocity: Array
    wall_normal_axis: int = eqx.field(static=True)
    streamwise_axis: int = eqx.field(static=True)
    homogeneous_axes: tuple[int, ...] = eqx.field(static=True)
    density: float = eqx.field(static=True)
    kinematic_viscosity: float = eqx.field(static=True)
    channel_height: float = eqx.field(static=True)
    total_volume: float = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    operators_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        density: float,
        kinematic_viscosity: float,
        wall_normal_axis: int = 1,
        streamwise_axis: int = 0,
        lower_wall_velocity: ArrayLike | None = None,
        upper_wall_velocity: ArrayLike | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        dimension = len(operators.discretization.cell_shape)
        wall_axis = int(wall_normal_axis)
        stream_axis = int(streamwise_axis)
        density_ = float(density)
        viscosity = float(kinematic_viscosity)
        lower_velocity = np.zeros((dimension,), dtype=float)
        if lower_wall_velocity is not None:
            lower_velocity = np.asarray(lower_wall_velocity, dtype=float)
        upper_velocity = np.zeros((dimension,), dtype=float)
        if upper_wall_velocity is not None:
            upper_velocity = np.asarray(upper_wall_velocity, dtype=float)
        if (
            dimension not in (2, 3)
            or wall_axis < 0
            or wall_axis >= dimension
            or stream_axis < 0
            or stream_axis >= dimension
            or stream_axis == wall_axis
            or lower_velocity.shape != (dimension,)
            or upper_velocity.shape != (dimension,)
            or np.any(~np.isfinite(lower_velocity))
            or np.any(~np.isfinite(upper_velocity))
            or not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(viscosity)
            or viscosity <= 0.0
        ):
            raise ValueError("MAC plane/wall statistic parameters are invalid.")
        axes = operators.discretization.grid.structured_axes
        wall = axes[wall_axis]
        homogeneous = tuple(axis for axis in range(dimension) if axis != wall_axis)
        if wall.periodic or any(not axes[axis].periodic for axis in homogeneous):
            raise ValueError(
                "MAC plane statistics require one nonperiodic wall axis and "
                "periodic homogeneous axes."
            )
        if wall.bounds is None:
            raise ValueError("The MAC wall-normal axis must have finite bounds.")
        coordinates = np.asarray(wall.interval_centers, dtype=float)
        lower, upper = (float(value) for value in wall.bounds)
        height = upper - lower
        volumes = np.asarray(operators.discretization.cell_volumes)
        total_volume = float(np.sum(volumes))
        if (
            coordinates.size == 0
            or np.any(~np.isfinite(coordinates))
            or height <= 0.0
            or not np.isfinite(total_volume)
            or total_volume <= 0.0
        ):
            raise ValueError("MAC plane/wall geometry is invalid.")
        self.operators = operators
        self.wall_normal_coordinates = wall.interval_centers
        self.cell_volumes = operators.discretization.cell_volumes
        self.lower_wall_velocity = jnp.asarray(lower_velocity)
        self.upper_wall_velocity = jnp.asarray(upper_velocity)
        self.wall_normal_axis = wall_axis
        self.streamwise_axis = stream_axis
        self.homogeneous_axes = homogeneous
        self.density = density_
        self.kinematic_viscosity = viscosity
        self.channel_height = height
        self.total_volume = total_volume
        self.discretization_id = operators.discretization.prepared_id
        self.operators_id = operators.prepared_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-plane-wall-statistics-v1",
                "operators": operators.prepared_id,
                "wall_normal_axis": wall_axis,
                "streamwise_axis": stream_axis,
                "homogeneous_axes": homogeneous,
                "density": density_,
                "kinematic_viscosity": viscosity,
                "lower_wall_velocity": tuple(float(value) for value in lower_velocity),
                "upper_wall_velocity": tuple(float(value) for value in upper_velocity),
                "face_to_cell": "adjacent-face-arithmetic-average",
                "plane_weight": "exact-cell-volume",
                "wall_normal_value": "native-boundary-face",
                "wall_shear": "no-slip-one-sided-cell-center",
            }
        )

    def _cell_velocity(self, velocity: FaceVelocity, /) -> Array:
        values = self.operators.validate_velocity(velocity)
        axes = self.operators.discretization.grid.structured_axes
        return jnp.stack(
            tuple(
                _mac_face_to_cell(value, axis, axes[axis].periodic)
                for axis, value in enumerate(values)
            ),
            axis=-1,
        )

    def evaluate(
        self,
        velocity: FaceVelocity,
        /,
        *,
        forcing: FaceVelocity | None = None,
    ) -> MACPlaneWallStatistics:
        values = self.operators.validate_velocity(velocity)
        cell_velocity = self._cell_velocity(values)
        volumes = self.cell_volumes.astype(cell_velocity.dtype)
        profile_weights = jnp.sum(volumes, axis=self.homogeneous_axes)
        weighted_velocity = volumes[..., None] * cell_velocity
        mean = (
            jnp.sum(weighted_velocity, axis=self.homogeneous_axes)
            / profile_weights[..., None]
        )
        products = cell_velocity[..., :, None] * cell_velocity[..., None, :]
        raw_second = (
            jnp.sum(
                volumes[..., None, None] * products,
                axis=self.homogeneous_axes,
            )
            / profile_weights[..., None, None]
        )
        reynolds = raw_second - mean[..., :, None] * mean[..., None, :]
        total_volume = jnp.asarray(self.total_volume, dtype=cell_velocity.dtype)
        bulk = jnp.sum(weighted_velocity, axis=tuple(range(volumes.ndim))) / total_volume
        wall_axis = self.wall_normal_axis
        wall = self.operators.discretization.grid.structured_axes[wall_axis]
        lower, upper = (float(value) for value in wall.bounds)
        lower_distance = self.wall_normal_coordinates[0] - lower
        upper_distance = upper - self.wall_normal_coordinates[-1]
        dynamic_viscosity = self.density * self.kinematic_viscosity
        lower_wall_velocity = self.lower_wall_velocity.astype(mean.dtype)
        upper_wall_velocity = self.upper_wall_velocity.astype(mean.dtype)
        lower_shear = dynamic_viscosity * (mean[0] - lower_wall_velocity) / lower_distance
        upper_shear = (
            dynamic_viscosity * (upper_wall_velocity - mean[-1]) / upper_distance
        )
        lower_shear = lower_shear.at[wall_axis].set(0.0)
        upper_shear = upper_shear.at[wall_axis].set(0.0)
        normal = values[wall_axis]
        normal_measures = self.operators.discretization.face_measures[wall_axis]
        lower_selector = [slice(None)] * normal.ndim
        upper_selector = [slice(None)] * normal.ndim
        lower_selector[wall_axis] = 0
        upper_selector[wall_axis] = normal.shape[wall_axis] - 1
        lower_values = normal[tuple(lower_selector)]
        upper_values = normal[tuple(upper_selector)]
        lower_measures = normal_measures[tuple(lower_selector)]
        upper_measures = normal_measures[tuple(upper_selector)]
        lower_normal = jnp.sum(lower_measures * lower_values) / jnp.sum(lower_measures)
        upper_normal = jnp.sum(upper_measures * upper_values) / jnp.sum(upper_measures)
        kinetic_energy = 0.5 * sum(
            jnp.sum(measure.astype(value.dtype) * value**2)
            for measure, value in zip(
                self.operators.face_dual_measures, values, strict=True
            )
        )
        if forcing is None:
            force_values = tuple(jnp.zeros_like(value) for value in values)
        else:
            force_values = self.operators.validate_velocity(forcing)
        forcing_power = sum(
            jnp.sum(measure.astype(value.dtype) * value * force)
            for measure, value, force in zip(
                self.operators.face_dual_measures,
                values,
                force_values,
                strict=True,
            )
        )
        divergence = self.operators.divergence(values)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence**2))
        finite = (
            jnp.all(jnp.isfinite(cell_velocity))
            & jnp.all(jnp.isfinite(raw_second))
            & jnp.all(jnp.isfinite(lower_shear))
            & jnp.all(jnp.isfinite(upper_shear))
            & jnp.isfinite(lower_normal)
            & jnp.isfinite(upper_normal)
            & jnp.isfinite(kinetic_energy)
            & jnp.isfinite(forcing_power)
            & jnp.isfinite(divergence_norm)
        )
        return MACPlaneWallStatistics(
            wall_normal_coordinates=self.wall_normal_coordinates,
            plane_weights=profile_weights,
            mean_velocity=mean,
            raw_second_moment=raw_second,
            reynolds_stress=reynolds,
            lower_wall_shear=lower_shear,
            upper_wall_shear=upper_shear,
            bulk_velocity=bulk,
            lower_wall_normal_velocity=lower_normal,
            upper_wall_normal_velocity=upper_normal,
            kinetic_energy=kinetic_energy,
            mean_kinetic_energy=kinetic_energy / total_volume,
            forcing_power=forcing_power,
            mean_forcing_power=forcing_power / total_volume,
            divergence_norm=divergence_norm,
            finite=finite,
            successful=finite,
            wall_normal_axis=self.wall_normal_axis,
            streamwise_axis=self.streamwise_axis,
            face_to_cell_convention="adjacent-face arithmetic average",
            plane_weight_convention="exact cell-volume weighted homogeneous plane",
            wall_shear_convention=(
                "rho*nu*d<cell-centered tangential velocity>/dy; "
                "one-sided derivative from each declared no-slip wall velocity"
            ),
            discretization_id=self.discretization_id,
            operators_id=self.operators_id,
            plan_id=self.plan_id,
        )


__all__ = [
    "ModalShellStatistic",
    "MACPlaneWallStatistics",
    "MACPlaneWallStatisticsPlan",
    "PeriodicModalTurbulenceStatistics",
    "PeriodicModalTurbulenceStatisticsPlan",
    "SpectralChannelStatistics",
    "SpectralChannelStatisticsPlan",
]
