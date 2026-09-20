#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded flat-periodic, wave-only cosmological Schrödinger--Poisson dynamics."""

from __future__ import annotations

from math import isfinite, pi, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spectral._dealias import (
    AbstractDealiasingPlan,
    DealiasingReport,
    OversamplingDealiasingPlan,
    PreparedDealiasingPlan,
)
from ...discretization.spectral._space import TensorSpectralDiscretization
from ._background import FLRWBackground
from ._scales import CODE_COSMOLOGY_SCALE, CosmologyScaleContract


WaveDarkMatterDifferentiability: TypeAlias = Literal["smooth_fixed_grid", "none"]


class WaveDarkMatterState(StrictModule):
    """Complex physical-grid wavefunction and its dimensionless scale factor.

    The normalization convention is comoving number density ``n_c = |psi|^2``;
    consequently, comoving mass density is ``rho_c = m |psi|^2``. Spatial
    coordinates are the comoving Cartesian coordinates declared by the plan's
    :class:`CosmologyScaleContract`.
    """

    psi: Array
    scale_factor: Array


class WaveDarkMatterStepPolicy(StrictModule, NonTrainableState):
    """Accuracy, resolution, and differentiation gates for a fixed schedule."""

    maximum_phase_radians: float = eqx.field(static=True)
    minimum_de_broglie_cells: float = eqx.field(static=True)
    norm_relative_tolerance: float = eqx.field(static=True)
    poisson_relative_tolerance: float = eqx.field(static=True)
    zero_mode_absolute_tolerance: float = eqx.field(static=True)
    relative_amplitude_floor: float = eqx.field(static=True)
    differentiability: WaveDarkMatterDifferentiability = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_phase_radians: float = 0.75,
        minimum_de_broglie_cells: float = 4.0,
        norm_relative_tolerance: float = 1.0e-8,
        poisson_relative_tolerance: float = 1.0e-9,
        zero_mode_absolute_tolerance: float = 1.0e-10,
        relative_amplitude_floor: float = 1.0e-10,
        differentiability: WaveDarkMatterDifferentiability = "smooth_fixed_grid",
    ):
        values = (
            float(maximum_phase_radians),
            float(minimum_de_broglie_cells),
            float(norm_relative_tolerance),
            float(poisson_relative_tolerance),
            float(zero_mode_absolute_tolerance),
            float(relative_amplitude_floor),
        )
        if (
            not all(isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] < 2.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] < 0.0
            or not 0.0 < values[5] < 1.0
        ):
            raise ValueError("Wave-dark-matter step-policy values are invalid.")
        if differentiability not in ("smooth_fixed_grid", "none"):
            raise ValueError("Unknown wave-dark-matter differentiation policy.")
        (
            self.maximum_phase_radians,
            self.minimum_de_broglie_cells,
            self.norm_relative_tolerance,
            self.poisson_relative_tolerance,
            self.zero_mode_absolute_tolerance,
            self.relative_amplitude_floor,
        ) = values
        self.differentiability = differentiability
        self.policy_id = canonical_fingerprint(
            {
                "kind": "wave-dark-matter-step-policy",
                "maximum_phase_radians": values[0],
                "minimum_de_broglie_cells": values[1],
                "norm_relative_tolerance": values[2],
                "poisson_relative_tolerance": values[3],
                "zero_mode_absolute_tolerance": values[4],
                "relative_amplitude_floor": values[5],
                "differentiability": differentiability,
            }
        )


class WaveDarkMatterPoissonResult(StrictModule):
    """Mean-zero spectral Poisson action at one wave state.

    ``potential`` is the scale-weighted peculiar potential ``phi = a Phi``.
    With comoving Cartesian derivatives the solved convention is
    ``laplacian(phi) = 4 pi G (rho_c - mean(rho_c))``. The physical peculiar
    potential is therefore ``Phi = potential / scale_factor``. All numeric
    quantities use the plan's code length, mass, and time units.
    """

    density: Array
    density_contrast: Array
    source: Array
    potential: Array
    physical_potential: Array
    laplacian: Array
    relative_residual: Array
    zero_mode_absolute: Array
    mean_density: Array
    finite: Array
    successful: Array


class WaveDarkMatterDiagnostics(StrictModule):
    """Fixed-shape rollout diagnostics and transactional acceptance evidence.

    State histories have the fixed schedule length. Step quantities have one
    entry per scheduled interval; ``attempted`` masks entries after the first
    rejected interval. Status codes are documented by ``step_status_meanings``.
    """

    state_scale_factors: Array
    norm: Array
    mass: Array
    poisson_relative_residual: Array
    potential_zero_mode_absolute: Array
    kinetic_energy: Array
    potential_energy: Array
    total_energy: Array
    de_broglie_nyquist_fraction: Array
    maximum_kinetic_phase: Array
    maximum_potential_phase: Array
    norm_relative_error: Array
    phase_resolved: Array
    de_broglie_resolved: Array
    poisson_closed: Array
    zero_mode_removed: Array
    finite: Array
    attempted: Array
    accepted: Array
    step_status: Array
    initial_finite: Array
    initial_accepted: Array
    completed: Array
    accepted_steps: Array
    first_failed_step: Array
    dealiasing: DealiasingReport
    step_status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "not_attempted",
            "non_finite",
            "phase_unresolved",
            "de_broglie_unresolved",
            "poisson_unclosed",
            "norm_drift",
        ),
    )


class WaveDarkMatterResult(StrictModule):
    state: WaveDarkMatterState
    diagnostics: WaveDarkMatterDiagnostics
    successful: Array
    prepared_id: str = eqx.field(static=True)


class WaveDarkMatterPlan(StrictModule, NonTrainableState):
    """Host-side configuration for one fixed-grid wave-only cosmology rollout.

    ``boson_mass``, ``gravitational_constant``, and ``reduced_planck_constant``
    are expressed in the supplied cosmological mass/length/time code units.
    The default oversampling policy is deliberately approximate for the
    nonpolynomial potential exponential and reports that fact explicitly.
    """

    boson_mass: float = eqx.field(static=True)
    scale_factors: Array
    gravitational_constant: float = eqx.field(static=True)
    reduced_planck_constant: float = eqx.field(static=True)
    step_policy: WaveDarkMatterStepPolicy
    dealiasing: AbstractDealiasingPlan
    scale: CosmologyScaleContract
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boson_mass: float,
        scale_factors: ArrayLike,
        /,
        *,
        gravitational_constant: float = 1.0,
        reduced_planck_constant: float = 1.0,
        step_policy: WaveDarkMatterStepPolicy | None = None,
        dealiasing: AbstractDealiasingPlan | None = None,
        scale: CosmologyScaleContract = CODE_COSMOLOGY_SCALE,
    ):
        mass = float(boson_mass)
        gravity = float(gravitational_constant)
        hbar = float(reduced_planck_constant)
        if not all(isfinite(value) and value > 0.0 for value in (mass, gravity, hbar)):
            raise ValueError("Wave-dark-matter physical coefficients must be positive.")
        schedule_host = np.asarray(scale_factors, dtype=np.float64).reshape((-1,))
        if (
            schedule_host.size < 2
            or np.any(~np.isfinite(schedule_host))
            or np.any(schedule_host <= 0.0)
            or np.any(np.diff(schedule_host) <= 0.0)
        ):
            raise ValueError(
                "Wave-dark-matter scale factors must be finite, positive, and increasing."
            )
        policy = WaveDarkMatterStepPolicy() if step_policy is None else step_policy
        nonlinear = OversamplingDealiasingPlan(1.5) if dealiasing is None else dealiasing
        if not isinstance(policy, WaveDarkMatterStepPolicy):
            raise TypeError("step_policy must be WaveDarkMatterStepPolicy or None.")
        if not isinstance(nonlinear, AbstractDealiasingPlan):
            raise TypeError("dealiasing must be an AbstractDealiasingPlan or None.")
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be a CosmologyScaleContract.")
        if scale.length_coordinate_kind != "comoving":
            raise ValueError(
                "Wave dark matter requires a comoving length scale contract."
            )
        self.boson_mass = mass
        self.scale_factors = jax.lax.stop_gradient(jnp.asarray(schedule_host))
        self.gravitational_constant = gravity
        self.reduced_planck_constant = hbar
        self.step_policy = policy
        self.dealiasing = nonlinear
        self.scale = scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flat-periodic-wave-dark-matter-plan",
                "boson_mass": mass,
                "scale_factors": array_tree_fingerprint(schedule_host),
                "gravitational_constant": gravity,
                "reduced_planck_constant": hbar,
                "step_policy": policy.policy_id,
                "dealiasing": nonlinear.plan_id,
                "scale": scale.scale_id,
                "profile": "wave-only-fixed-schedule-single-device",
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        background: FLRWBackground,
        /,
    ) -> PreparedPeriodicWaveDarkMatter:
        """Bind the plan to one local tensor Fourier grid and flat FLRW model."""
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be TensorSpectralDiscretization.")
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if background.scale.scale_id != self.scale.scale_id:
            raise ValueError("Background and wave-dark-matter scale contracts disagree.")
        if float(np.asarray(background.curvature_density)) != 0.0:
            raise ValueError("Periodic wave dark matter requires zero spatial curvature.")
        prepared_dealiasing = self.dealiasing.prepare(
            discretization,
            required_polynomial_degree=None,
        )
        return PreparedPeriodicWaveDarkMatter(
            self,
            discretization,
            prepared_dealiasing,
            background,
        )


class PreparedPeriodicWaveDarkMatter(StrictModule, NonTrainableState):
    """Prepared Strang split for bounded periodic cosmological wave dynamics.

    The comoving wavefunction obeys

    ``i hbar dpsi/da = -hbar^2 laplacian(psi)/(2 m a^3 H)``
    ``+ m phi psi/(a^2 H)``,

    where ``phi = a Phi`` and ``laplacian(phi) = 4 pi G (rho_c-rho_bar)``.
    Each scheduled interval uses a potential-half / kinetic-full /
    potential-half split. The drift and kick midpoint quadratures are exactly
    :meth:`FLRWBackground.drift_factor` and :meth:`FLRWBackground.kick_factor`.
    """

    plan: WaveDarkMatterPlan
    discretization: TensorSpectralDiscretization
    dealiasing: PreparedDealiasingPlan
    background: FLRWBackground
    wavenumber_squared: Array
    inverse_wavenumber_squared: Array
    zero_mode_mask: Array
    scale_factors: Array
    boson_mass: float = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    reduced_planck_constant: float = eqx.field(static=True)
    cell_volume: float = eqx.field(static=True)
    maximum_grid_spacing: float = eqx.field(static=True)
    nyquist_wavenumber: float = eqx.field(static=True)
    cosmology_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    potential_convention: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: WaveDarkMatterPlan,
        discretization: TensorSpectralDiscretization,
        dealiasing: PreparedDealiasingPlan,
        background: FLRWBackground,
        /,
    ):
        if not isinstance(plan, WaveDarkMatterPlan):
            raise TypeError("plan must be WaveDarkMatterPlan.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be TensorSpectralDiscretization.")
        if not isinstance(dealiasing, PreparedDealiasingPlan):
            raise TypeError("dealiasing must be PreparedDealiasingPlan.")
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        rank = len(discretization.axes)
        if rank not in (1, 2, 3) or any(
            axis.family != "fourier" or not axis.periodic for axis in discretization.axes
        ):
            raise ValueError(
                "Periodic wave dark matter requires one to three tensor Fourier axes."
            )
        coefficient_dtype = jnp.dtype(discretization.plan.precision.coefficient_dtype)
        if not jnp.issubdtype(coefficient_dtype, jnp.complexfloating):
            raise TypeError("Wave dark matter requires complex spectral coefficients.")
        if dealiasing.retained.prepared_id != discretization.prepared_id:
            raise ValueError("Dealiasing must retain the wave-dark-matter grid.")
        if dealiasing.report.kind == "none":
            raise ValueError(
                "The nonpolynomial potential action requires explicit dealiasing."
            )
        real_dtype = jnp.empty((), dtype=coefficient_dtype).real.dtype
        squared = jnp.zeros(discretization.modal_shape, dtype=real_dtype)
        for axis_index, axis in enumerate(discretization.axes):
            values = axis.laplacian_eigenvalues().astype(real_dtype)
            shape = [1] * rank
            shape[axis_index] = axis.mode_count
            squared = squared + values.reshape(tuple(shape))
        zero = squared == 0.0
        safe = jnp.where(zero, jnp.ones_like(squared), squared)
        inverse = jnp.where(zero, jnp.zeros_like(squared), 1.0 / safe)
        lengths = tuple(float(np.asarray(axis.length)) for axis in discretization.axes)
        spacings = tuple(
            length / count
            for length, count in zip(lengths, discretization.physical_shape, strict=True)
        )
        volume = float(prod(lengths))
        maximum_spacing = float(max(spacings))
        nyquist = float(min(pi / spacing for spacing in spacings))
        if not all(
            isfinite(value) and value > 0.0
            for value in (volume, maximum_spacing, nyquist)
        ):
            raise ValueError("Periodic wave-dark-matter grid geometry is invalid.")
        cosmology_id = background.physical_state.content_id()
        self.plan = plan
        self.discretization = discretization
        self.dealiasing = dealiasing
        self.background = background
        self.wavenumber_squared = squared
        self.inverse_wavenumber_squared = inverse
        self.zero_mode_mask = zero
        self.scale_factors = plan.scale_factors.astype(real_dtype)
        self.boson_mass = plan.boson_mass
        self.gravitational_constant = plan.gravitational_constant
        self.reduced_planck_constant = plan.reduced_planck_constant
        self.cell_volume = volume
        self.maximum_grid_spacing = maximum_spacing
        self.nyquist_wavenumber = nyquist
        self.cosmology_id = cosmology_id
        self.scale_id = plan.scale.scale_id
        self.coordinate_convention = "flat-periodic-comoving-cartesian"
        self.potential_convention = "phi=a*Phi; laplacian(phi)=4*pi*G*(rho_c-mean)"
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-wave-dark-matter",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
                "dealiasing": dealiasing.prepared_id,
                "cosmology": cosmology_id,
                "scale": plan.scale.scale_id,
                "modal_geometry": array_tree_fingerprint(np.asarray(squared)),
                "coordinate_convention": self.coordinate_convention,
                "potential_convention": self.potential_convention,
            }
        )

    @property
    def step_policy(self) -> WaveDarkMatterStepPolicy:
        return self.plan.step_policy

    def _weighted_sum(self, values: Array, weights: Array, /) -> Array:
        return ein.contract("i,i->", weights.reshape((-1,)), values.reshape((-1,)))

    def _norm(self, psi: Array, /) -> Array:
        return jnp.real(
            self._weighted_sum(
                jnp.abs(psi) ** 2,
                self.discretization.quadrature_weights,
            )
        )

    def _validate_state(
        self,
        state: WaveDarkMatterState,
        /,
        *,
        require_schedule_start: bool,
    ) -> WaveDarkMatterState:
        if not isinstance(state, WaveDarkMatterState):
            raise TypeError("state must be WaveDarkMatterState.")
        psi = jnp.asarray(state.psi)
        if psi.shape != self.discretization.physical_shape:
            raise ValueError(
                f"Wavefunction must have physical grid shape {self.discretization.physical_shape}; got {psi.shape}."
            )
        if not jnp.issubdtype(psi.dtype, jnp.complexfloating):
            raise TypeError("Wavefunction psi must have a complex dtype.")
        scale = jnp.asarray(state.scale_factor, dtype=psi.real.dtype)
        if scale.shape != ():
            raise ValueError("Wave-dark-matter scale factor must be scalar.")
        invalid = (
            jnp.any(~jnp.isfinite(psi))
            | ~jnp.isfinite(scale)
            | (scale <= 0.0)
            | (self._norm(psi) <= 0.0)
        )
        if require_schedule_start:
            invalid = invalid | (
                jnp.abs(scale - self.scale_factors[0]) > 32.0 * jnp.finfo(scale.dtype).eps
            )
        psi = eqx.error_if(
            psi,
            invalid,
            (
                "Wave-dark-matter state must be finite, nonzero, positive in scale, "
                "and start at the first scheduled scale factor."
                if require_schedule_start
                else "Wave-dark-matter state must be finite, nonzero, and positive in scale."
            ),
        )
        return WaveDarkMatterState(psi, scale)

    def initialize(
        self,
        psi: ArrayLike,
        scale_factor: ArrayLike | None = None,
        /,
    ) -> WaveDarkMatterState:
        """Create a validated physical-grid state at the first scheduled scale."""
        value = jnp.asarray(psi)
        scale = self.scale_factors[0] if scale_factor is None else scale_factor
        return self._validate_state(
            WaveDarkMatterState(value, jnp.asarray(scale)),
            require_schedule_start=True,
        )

    def _evaluation_wavefunction(self, psi: Array, /) -> tuple[Array, Array]:
        coefficients = self.discretization.project(psi)
        embedded = self.dealiasing.embed(coefficients)
        values = self.dealiasing.evaluation.reconstruct(embedded, real_output=False)
        return coefficients, values

    def _poisson_modal(self, psi: Array, /) -> tuple[Array, Array, Array, Array, Array]:
        coefficients, evaluation_psi = self._evaluation_wavefunction(psi)
        density = self.boson_mass * jnp.abs(evaluation_psi) ** 2
        density_coefficients = self.dealiasing.project(density)
        contrast_coefficients = jnp.where(
            self.zero_mode_mask,
            jnp.zeros_like(density_coefficients),
            density_coefficients,
        )
        source_coefficients = (
            4.0 * pi * self.gravitational_constant * contrast_coefficients
        )
        potential_coefficients = -source_coefficients * self.inverse_wavenumber_squared
        return (
            coefficients,
            evaluation_psi,
            density,
            source_coefficients,
            potential_coefficients,
        )

    def _poisson_result(
        self,
        psi: Array,
        scale_factor: Array,
        /,
    ) -> WaveDarkMatterPoissonResult:
        _, _, density_evaluation, source_coefficients, potential_coefficients = (
            self._poisson_modal(psi)
        )
        density = self.boson_mass * jnp.abs(psi) ** 2
        mean_density = (
            self._weighted_sum(
                density_evaluation,
                self.dealiasing.evaluation.quadrature_weights,
            )
            / self.cell_volume
        )
        contrast = self.discretization.reconstruct(
            source_coefficients / (4.0 * pi * self.gravitational_constant),
            real_output=True,
        )
        source = self.discretization.reconstruct(source_coefficients, real_output=True)
        potential = self.discretization.reconstruct(
            potential_coefficients,
            real_output=True,
        )
        laplacian = self.discretization.reconstruct(
            -self.wavenumber_squared * potential_coefficients,
            real_output=True,
        )
        residual = laplacian - source
        residual_norm = jnp.sqrt(
            jnp.maximum(
                self._weighted_sum(
                    residual**2,
                    self.discretization.quadrature_weights,
                ),
                0.0,
            )
        )
        source_norm = jnp.sqrt(
            jnp.maximum(
                self._weighted_sum(
                    source**2,
                    self.discretization.quadrature_weights,
                ),
                0.0,
            )
        )
        safe_source_norm = jnp.where(
            source_norm > 0.0,
            source_norm,
            jnp.ones_like(source_norm),
        )
        relative_residual = jnp.where(
            source_norm > 0.0,
            residual_norm / safe_source_norm,
            residual_norm,
        )
        potential_mean = (
            self._weighted_sum(
                potential,
                self.discretization.quadrature_weights,
            )
            / self.cell_volume
        )
        zero_mode = jnp.abs(potential_mean)
        physical_potential = potential / scale_factor
        finite = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(contrast))
            & jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(physical_potential))
            & jnp.all(jnp.isfinite(laplacian))
            & jnp.isfinite(mean_density)
            & jnp.isfinite(relative_residual)
            & jnp.isfinite(zero_mode)
        )
        successful = (
            finite
            & (relative_residual <= self.step_policy.poisson_relative_tolerance)
            & (zero_mode <= self.step_policy.zero_mode_absolute_tolerance)
        )
        return WaveDarkMatterPoissonResult(
            density,
            contrast,
            source,
            potential,
            physical_potential,
            laplacian,
            relative_residual,
            zero_mode,
            mean_density,
            finite,
            successful,
        )

    def poisson(self, state: WaveDarkMatterState, /) -> WaveDarkMatterPoissonResult:
        """Apply the mean-zero spectral Poisson solve at ``state.scale_factor``."""
        checked = self._validate_state(state, require_schedule_start=False)
        return self._poisson_result(checked.psi, checked.scale_factor)

    def density(self, state: WaveDarkMatterState, /) -> Array:
        """Return comoving mass density on the prepared physical grid."""
        checked = self._validate_state(state, require_schedule_start=False)
        return self.boson_mass * jnp.abs(checked.psi) ** 2

    def _validate_action_interval(
        self,
        state: WaveDarkMatterState,
        start_scale_factor: ArrayLike,
        end_scale_factor: ArrayLike,
        /,
    ) -> tuple[WaveDarkMatterState, Array, Array]:
        checked = self._validate_state(state, require_schedule_start=False)
        start = jnp.asarray(start_scale_factor, dtype=checked.psi.real.dtype)
        end = jnp.asarray(end_scale_factor, dtype=checked.psi.real.dtype)
        if start.shape != () or end.shape != ():
            raise ValueError("Wave action scale factors must be scalar.")
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start) | ~jnp.isfinite(end) | (start <= 0.0) | (end <= start),
            "Wave action scale factors must be finite, positive, and increasing.",
        )
        return checked, start, end

    def kinetic_drift(
        self,
        state: WaveDarkMatterState,
        start_scale_factor: ArrayLike,
        end_scale_factor: ArrayLike,
        /,
    ) -> WaveDarkMatterState:
        """Apply the exact fixed-grid kinetic action and advance its time level."""
        checked, start, end = self._validate_action_interval(
            state,
            start_scale_factor,
            end_scale_factor,
        )
        tolerance = 32.0 * jnp.finfo(checked.psi.real.dtype).eps
        psi = eqx.error_if(
            checked.psi,
            jnp.abs(checked.scale_factor - start) > tolerance,
            "Kinetic drift state must be at start_scale_factor.",
        )
        drift = self.background.drift_factor(start, end).astype(psi.real.dtype)
        updated, _ = self._kinetic_step(psi, drift)
        return WaveDarkMatterState(updated, end)

    def potential_kick(
        self,
        state: WaveDarkMatterState,
        potential: ArrayLike,
        start_scale_factor: ArrayLike,
        end_scale_factor: ArrayLike,
        fraction: ArrayLike,
        /,
    ) -> WaveDarkMatterState:
        """Apply stop-gradient external ``phi=a*Phi`` without changing time level."""
        checked, start, end = self._validate_action_interval(
            state,
            start_scale_factor,
            end_scale_factor,
        )
        values = jnp.asarray(potential)
        if values.shape != self.discretization.physical_shape:
            raise ValueError(
                "External potential must have physical grid shape "
                f"{self.discretization.physical_shape}; got {values.shape}."
            )
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("External potential must have a real floating dtype.")
        values = values.astype(checked.psi.real.dtype)
        amount = jnp.asarray(fraction, dtype=checked.psi.real.dtype)
        if amount.shape != ():
            raise ValueError("Potential-kick fraction must be scalar.")
        tolerance = 32.0 * jnp.finfo(checked.psi.real.dtype).eps
        at_endpoint = (jnp.abs(checked.scale_factor - start) <= tolerance) | (
            jnp.abs(checked.scale_factor - end) <= tolerance
        )
        values = eqx.error_if(
            values,
            ~jnp.all(jnp.isfinite(values))
            | ~jnp.isfinite(amount)
            | (amount < 0.0)
            | (amount > 1.0)
            | ~at_endpoint,
            "Potential kick requires finite data, a fraction in [0, 1], and state at an interval endpoint.",
        )
        values = jax.lax.stop_gradient(values)
        kick = self.background.kick_factor(start, end).astype(checked.psi.real.dtype)
        phase = amount * self.boson_mass * kick * values / self.reduced_planck_constant
        updated = checked.psi * jnp.exp(-1j * phase)
        return WaveDarkMatterState(updated, checked.scale_factor)

    def _de_broglie_fraction(self, psi: Array, /) -> Array:
        coefficients = self.discretization.project(psi)
        amplitude_squared = jnp.abs(psi) ** 2
        maximum = jnp.max(amplitude_squared)
        occupied = (
            amplitude_squared >= self.step_policy.relative_amplitude_floor * maximum
        )
        safe_density = jnp.where(
            occupied, amplitude_squared, jnp.ones_like(amplitude_squared)
        )
        wave_squared = jnp.zeros_like(amplitude_squared)
        for axis in range(len(self.discretization.axes)):
            derivative = self.discretization.reconstruct(
                self.discretization.modal_derivative(coefficients, axis=axis),
                real_output=False,
            )
            component = jnp.imag(jnp.conj(psi) * derivative) / safe_density
            wave_squared = wave_squared + jnp.where(occupied, component**2, 0.0)
        maximum_wave = jnp.sqrt(jnp.max(wave_squared))
        return maximum_wave / self.nyquist_wavenumber

    def _snapshot(
        self,
        psi: Array,
        scale_factor: Array,
        /,
    ) -> tuple[Array, ...]:
        poisson = self._poisson_result(psi, scale_factor)
        norm = self._norm(psi)
        mass = self.boson_mass * norm
        coefficients = self.discretization.project(psi)
        kinetic_integral = jnp.real(
            ein.contract(
                "i,i->",
                self.wavenumber_squared.reshape((-1,)),
                (jnp.abs(coefficients) ** 2).reshape((-1,)),
            )
        )
        kinetic = (
            self.reduced_planck_constant**2
            * kinetic_integral
            / (2.0 * self.boson_mass * scale_factor**2)
        )
        potential = (
            0.5
            * self._weighted_sum(
                poisson.potential * poisson.density,
                self.discretization.quadrature_weights,
            )
            / scale_factor
        )
        total = kinetic + potential
        de_broglie = self._de_broglie_fraction(psi)
        finite = (
            poisson.finite
            & jnp.isfinite(norm)
            & jnp.isfinite(mass)
            & jnp.isfinite(kinetic)
            & jnp.isfinite(potential)
            & jnp.isfinite(total)
            & jnp.isfinite(de_broglie)
        )
        return (
            scale_factor,
            norm,
            mass,
            poisson.relative_residual,
            poisson.zero_mode_absolute,
            kinetic,
            potential,
            total,
            de_broglie,
            finite,
            poisson.successful,
        )

    def _potential_half_step(
        self,
        psi: Array,
        kick_factor: Array,
        /,
    ) -> tuple[Array, Array]:
        _, evaluation_psi, _, _, potential_coefficients = self._poisson_modal(psi)
        potential = self.dealiasing.evaluation.reconstruct(
            self.dealiasing.embed(potential_coefficients),
            real_output=True,
        )
        coefficient = 0.5 * self.boson_mass * kick_factor / self.reduced_planck_constant
        phase = coefficient * potential
        updated_evaluation = evaluation_psi * jnp.exp(-1j * phase)
        updated_coefficients = self.dealiasing.project(updated_evaluation)
        updated = self.discretization.reconstruct(
            updated_coefficients,
            real_output=False,
        )
        return updated, jnp.max(jnp.abs(phase))

    def _kinetic_step(
        self,
        psi: Array,
        drift_factor: Array,
        /,
    ) -> tuple[Array, Array]:
        coefficients = self.discretization.project(psi)
        coefficient = 0.5 * self.reduced_planck_constant * drift_factor / self.boson_mass
        phase = coefficient * self.wavenumber_squared
        power = jnp.abs(coefficients) ** 2
        occupied = power >= self.step_policy.relative_amplitude_floor * jnp.max(power)
        maximum_phase = jnp.max(jnp.where(occupied, jnp.abs(phase), 0.0))
        updated = self.discretization.reconstruct(
            coefficients * jnp.exp(-1j * phase),
            real_output=False,
        )
        return updated, maximum_phase

    def _candidate(
        self,
        psi: Array,
        start_scale: Array,
        end_scale: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        drift = self.background.drift_factor(start_scale, end_scale).astype(
            psi.real.dtype
        )
        kick = self.background.kick_factor(start_scale, end_scale).astype(psi.real.dtype)
        first, first_phase = self._potential_half_step(psi, kick)
        kinetic, kinetic_phase = self._kinetic_step(first, drift)
        candidate, second_phase = self._potential_half_step(kinetic, kick)
        return candidate, kinetic_phase, jnp.maximum(first_phase, second_phase)

    def _smooth_final_psi(self, initial_psi: Array, /) -> Array:
        intervals = jnp.stack((self.scale_factors[:-1], self.scale_factors[1:]), axis=-1)

        def step(psi, interval):
            candidate, _, _ = self._candidate(psi, interval[0], interval[1])
            return candidate, None

        final, _ = jax.lax.scan(step, initial_psi, intervals)
        return final

    def solve(self, state: WaveDarkMatterState, /) -> WaveDarkMatterResult:
        """Advance transactionally over the fixed scale schedule.

        A rejected interval leaves the accepted state unchanged and masks all
        later intervals as not attempted. Numerical rejection is reported in
        ``step_status`` rather than being presented as a physical outcome.
        """
        initial_state = self._validate_state(state, require_schedule_start=True)
        initial_snapshot = self._snapshot(
            initial_state.psi,
            initial_state.scale_factor,
        )
        de_broglie_limit = 2.0 / self.step_policy.minimum_de_broglie_cells
        initial_finite = initial_snapshot[9]
        initial_poisson = initial_snapshot[10]
        initial_resolved = initial_snapshot[8] <= de_broglie_limit
        initial_accepted = initial_finite & initial_poisson & initial_resolved
        initial_norm = initial_snapshot[1]
        intervals = jnp.stack((self.scale_factors[:-1], self.scale_factors[1:]), axis=-1)

        def step(carry, interval):
            current_psi, current_scale, active, accepted_count, current_snapshot = carry
            candidate_psi, kinetic_phase, potential_phase = self._candidate(
                current_psi,
                interval[0],
                interval[1],
            )
            candidate_snapshot = self._snapshot(candidate_psi, interval[1])
            norm_error = jnp.abs(candidate_snapshot[1] - initial_norm) / initial_norm
            finite = candidate_snapshot[9]
            phase_resolved = (kinetic_phase <= self.step_policy.maximum_phase_radians) & (
                potential_phase <= self.step_policy.maximum_phase_radians
            )
            de_broglie_resolved = candidate_snapshot[8] <= de_broglie_limit
            poisson_closed = candidate_snapshot[10]
            zero_mode_removed = (
                candidate_snapshot[4] <= self.step_policy.zero_mode_absolute_tolerance
            )
            norm_conserved = norm_error <= self.step_policy.norm_relative_tolerance
            attempted = active
            accepted = (
                attempted
                & finite
                & phase_resolved
                & de_broglie_resolved
                & poisson_closed
                & zero_mode_removed
                & norm_conserved
            )
            next_psi = jnp.where(accepted, candidate_psi, current_psi)
            next_scale = jnp.where(accepted, interval[1], current_scale)
            next_snapshot = tuple(
                jnp.where(accepted, candidate_value, current_value)
                for candidate_value, current_value in zip(
                    candidate_snapshot,
                    current_snapshot,
                    strict=True,
                )
            )
            status = jnp.where(
                ~attempted,
                1,
                jnp.where(
                    ~finite,
                    2,
                    jnp.where(
                        ~phase_resolved,
                        3,
                        jnp.where(
                            ~de_broglie_resolved,
                            4,
                            jnp.where(
                                ~(poisson_closed & zero_mode_removed),
                                5,
                                jnp.where(~norm_conserved, 6, 0),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            output = (
                next_snapshot,
                jnp.where(attempted, kinetic_phase, 0.0),
                jnp.where(attempted, potential_phase, 0.0),
                jnp.where(attempted, norm_error, 0.0),
                attempted & phase_resolved,
                attempted & de_broglie_resolved,
                attempted & poisson_closed,
                attempted & zero_mode_removed,
                attempted & finite,
                attempted,
                accepted,
                status,
            )
            next_carry = (
                next_psi,
                next_scale,
                accepted,
                accepted_count + accepted.astype(jnp.int32),
                next_snapshot,
            )
            return next_carry, output

        carry = (
            initial_state.psi,
            initial_state.scale_factor,
            initial_accepted,
            jnp.asarray(0, dtype=jnp.int32),
            initial_snapshot,
        )
        final_carry, recorded = jax.lax.scan(step, carry, intervals)
        final_psi, final_scale, active, accepted_steps, _ = final_carry
        (
            snapshots,
            kinetic_phase,
            potential_phase,
            norm_error,
            phase_resolved,
            de_broglie_resolved,
            poisson_closed,
            zero_mode_removed,
            finite,
            attempted,
            accepted,
            status,
        ) = recorded
        histories = tuple(
            jnp.concatenate((initial_value[None], values), axis=0)
            for initial_value, values in zip(initial_snapshot, snapshots, strict=True)
        )
        failed = attempted & ~accepted
        first_failed = jnp.where(
            ~initial_accepted,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                jnp.any(failed),
                jnp.argmax(failed).astype(jnp.int32),
                jnp.asarray(-1, dtype=jnp.int32),
            ),
        )
        completed = (
            active
            & (accepted_steps == self.scale_factors.size - 1)
            & (
                jnp.abs(final_scale - self.scale_factors[-1])
                <= 32.0 * jnp.finfo(final_scale.dtype).eps
            )
        )
        diagnostics = WaveDarkMatterDiagnostics(
            state_scale_factors=histories[0],
            norm=histories[1],
            mass=histories[2],
            poisson_relative_residual=histories[3],
            potential_zero_mode_absolute=histories[4],
            kinetic_energy=histories[5],
            potential_energy=histories[6],
            total_energy=histories[7],
            de_broglie_nyquist_fraction=histories[8],
            maximum_kinetic_phase=kinetic_phase,
            maximum_potential_phase=potential_phase,
            norm_relative_error=norm_error,
            phase_resolved=phase_resolved,
            de_broglie_resolved=de_broglie_resolved,
            poisson_closed=poisson_closed,
            zero_mode_removed=zero_mode_removed,
            finite=finite,
            attempted=attempted,
            accepted=accepted,
            step_status=status,
            initial_finite=initial_finite,
            initial_accepted=initial_accepted,
            completed=completed,
            accepted_steps=accepted_steps,
            first_failed_step=first_failed,
            dealiasing=self.dealiasing.report,
        )
        return WaveDarkMatterResult(
            WaveDarkMatterState(final_psi, final_scale),
            diagnostics,
            completed,
            self.prepared_id,
        )

    def jvp(
        self,
        state: WaveDarkMatterState,
        tangent_psi: ArrayLike,
        /,
    ) -> Array:
        """Return the final-wavefunction JVP on an accepted smooth fixed-grid branch.

        The derivative excludes schedule, grid, cosmology, policy, and acceptance
        changes. It fails closed when differentiation is disabled or when the
        primal trajectory touches a rejected/nonsmooth policy boundary.
        """
        if self.step_policy.differentiability != "smooth_fixed_grid":
            raise ValueError("Wave-dark-matter JVP is disabled by the step policy.")
        checked = self._validate_state(state, require_schedule_start=True)
        tangent = jnp.asarray(tangent_psi, dtype=checked.psi.dtype)
        if tangent.shape != checked.psi.shape:
            raise ValueError("Wave-dark-matter tangent must match psi shape.")
        primal = self.solve(checked)
        _, action = jax.jvp(
            self._smooth_final_psi,
            (checked.psi,),
            (tangent,),
        )
        phase_interior = jnp.all(
            jnp.maximum(
                primal.diagnostics.maximum_kinetic_phase,
                primal.diagnostics.maximum_potential_phase,
            )
            < self.step_policy.maximum_phase_radians
        )
        resolution_interior = jnp.all(
            primal.diagnostics.de_broglie_nyquist_fraction
            < 2.0 / self.step_policy.minimum_de_broglie_cells
        )
        valid = primal.successful & phase_interior & resolution_interior
        return eqx.error_if(
            action,
            ~valid,
            "Wave-dark-matter JVP requires an accepted interior fixed-grid trajectory.",
        )


__all__ = [
    "PreparedPeriodicWaveDarkMatter",
    "WaveDarkMatterDiagnostics",
    "WaveDarkMatterDifferentiability",
    "WaveDarkMatterPlan",
    "WaveDarkMatterPoissonResult",
    "WaveDarkMatterResult",
    "WaveDarkMatterState",
    "WaveDarkMatterStepPolicy",
]
