#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

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
from ...equations import (
    CompiledIncompressibleSpectralDynamics,
    PeriodicIncompressibleStage,
    PeriodicLESStepRestriction,
)
from ...equations._dynamic_les import LagrangianDynamicLESState
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
    molecular_dissipation_shells: ModalShellStatistic
    advective_transfer_shells: ModalShellStatistic
    sgs_transfer_shells: ModalShellStatistic
    forcing_injection_shells: ModalShellStatistic
    resolved_spectral_flux: Array
    kinetic_energy: Array
    mean_kinetic_energy: Array
    molecular_dissipation: Array
    mean_molecular_dissipation: Array
    advective_energy_rate: Array
    mean_advective_energy_rate: Array
    sgs_energy_rate: Array
    mean_sgs_energy_rate: Array
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
    molecular_dissipation_tail_fraction: Array
    divergence_norm: Array
    velocity_reality_defect: Array
    sgs_modeled_dissipation: Array
    sgs_energy_identity_defect: Array
    sgs_projection_energy_defect: Array
    sgs_regularization_activity_count: Array
    sgs_dynamic_coefficient_minimum: Array
    sgs_dynamic_coefficient_mean: Array
    sgs_dynamic_coefficient_maximum: Array
    sgs_backscatter_activity_count: Array
    sgs_backscatter_limit_count: Array
    sgs_accepted_update_count: Array
    sgs_rejected_update_count: Array
    sgs_maximum_kinematic_viscosity: Array
    sgs_advective_step_limit: Array
    sgs_diffusive_step_limit: Array
    sgs_combined_step_limit: Array
    sgs_etdrk_step_limit: Array
    sgs_fully_explicit_step_limit: Array
    sgs_available: Array
    sgs_regularization_available: Array
    sgs_stability_available: Array
    forcing_available: Array
    helicity_valid: Array
    taylor_microscale_valid: Array
    kolmogorov_scale_valid: Array
    integral_scale_valid: Array
    energy_tail_valid: Array
    molecular_dissipation_tail_valid: Array
    finite: Array
    successful: Array
    tail_start_wavenumber: float = eqx.field(static=True)
    spectrum_convention: str = eqx.field(static=True)
    resolved_flux_convention: str = eqx.field(static=True)
    integral_scale_convention: str = eqx.field(static=True)
    tail_convention: str = eqx.field(static=True)
    source_problem_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    sgs_filter_id: str | None = eqx.field(static=True)
    sgs_model_id: str | None = eqx.field(static=True)
    sgs_prepared_model_id: str | None = eqx.field(static=True)
    sgs_prepared_action_id: str | None = eqx.field(static=True)
    sgs_regularization_id: str | None = eqx.field(static=True)
    sgs_dynamic_provenance_id: str | None = eqx.field(static=True)
    sgs_averaging_id: str | None = eqx.field(static=True)
    sgs_backscatter_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PeriodicModalTurbulenceStatisticsPlan(StrictModule, NonTrainableState):
    """Term-resolved full-complex statistics for one compiled periodic equation."""

    dynamics: CompiledIncompressibleSpectralDynamics
    projector: PeriodicLerayProjector
    geometry: _FourierShellBinGeometry
    conjugate_indices: Array
    viscosity: float = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    maximum_admissible_wavenumber: float = eqx.field(static=True)
    tail_start_wavenumber: float = eqx.field(static=True)
    reality_tolerance: float = eqx.field(static=True)
    solenoidal_tolerance: float = eqx.field(static=True)
    source_problem_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    sgs_filter_id: str | None = eqx.field(static=True)
    sgs_model_id: str | None = eqx.field(static=True)
    sgs_prepared_model_id: str | None = eqx.field(static=True)
    sgs_prepared_action_id: str | None = eqx.field(static=True)
    sgs_regularization_id: str | None = eqx.field(static=True)
    sgs_dynamic_provenance_id: str | None = eqx.field(static=True)
    sgs_averaging_id: str | None = eqx.field(static=True)
    sgs_backscatter_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledIncompressibleSpectralDynamics,
        bin_edges: ArrayLike,
        /,
        *,
        tail_start_wavenumber: float | None = None,
        reality_tolerance: float = 1.0e-10,
        solenoidal_tolerance: float = 1.0e-10,
    ):
        if not isinstance(dynamics, CompiledIncompressibleSpectralDynamics):
            raise TypeError("dynamics must be CompiledIncompressibleSpectralDynamics.")
        projector = dynamics.projector
        viscosity = float(np.asarray(dynamics.problem.viscosity))
        reality = float(reality_tolerance)
        solenoidal = float(solenoidal_tolerance)
        edges = np.asarray(bin_edges, dtype=float).reshape((-1,))
        if (
            not np.isfinite(viscosity)
            or viscosity < 0.0
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
        algebraic_les = dynamics.algebraic_les
        dynamic_les = dynamics.dynamic_les
        if algebraic_les is None and dynamic_les is None:
            filter_id = None
            model_id = None
            prepared_model_id = None
            prepared_action_id = None
            regularization_id = None
            dynamic_provenance_id = None
            averaging_id = None
            backscatter_id = None
        elif algebraic_les is not None:
            filter_id = algebraic_les.grid_filter.plan.resolved_filter.filter_id
            model_id = algebraic_les.model.model_id
            prepared_model_id = algebraic_les.model.prepared_id
            prepared_action_id = algebraic_les.prepared_id
            regularization_id = canonical_fingerprint(
                {
                    "kind": "periodic-algebraic-les-regularization",
                    "policy": "not-applicable-static-coefficient",
                }
            )
            dynamic_provenance_id = None
            averaging_id = None
            backscatter_id = None
        else:
            dynamic_model = dynamic_les.dynamic_model
            filter_id = dynamic_les.grid_filter.plan.resolved_filter.filter_id
            model_id = dynamic_model.model_id
            prepared_model_id = dynamic_model.prepared_id
            prepared_action_id = dynamic_les.prepared_id
            regularization_id = dynamic_model.regularization.regularization_id
            dynamic_provenance_id = dynamic_model.provenance.provenance_id
            averaging_id = dynamic_model.averaging.averaging_id
            backscatter_id = dynamic_model.backscatter.backscatter_id
        self.dynamics = dynamics
        self.projector = projector
        self.geometry = geometry
        self.conjugate_indices = jnp.asarray(conjugates, dtype=jnp.int32)
        self.viscosity = viscosity
        self.volume = volume
        self.maximum_admissible_wavenumber = maximum_wave
        self.tail_start_wavenumber = tail_start
        self.reality_tolerance = reality
        self.solenoidal_tolerance = solenoidal
        self.source_problem_id = dynamics.problem.problem_id
        self.compilation_id = dynamics.compilation_id
        self.discretization_id = projector.discretization.prepared_id
        self.projector_id = projector.projector_id
        self.sgs_filter_id = filter_id
        self.sgs_model_id = model_id
        self.sgs_prepared_model_id = prepared_model_id
        self.sgs_prepared_action_id = prepared_action_id
        self.sgs_regularization_id = regularization_id
        self.sgs_dynamic_provenance_id = dynamic_provenance_id
        self.sgs_averaging_id = averaging_id
        self.sgs_backscatter_id = backscatter_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-modal-turbulence-statistics",
                "source_problem": self.source_problem_id,
                "compilation": self.compilation_id,
                "discretization": self.discretization_id,
                "projector": self.projector_id,
                "shell_geometry": geometry.geometry_id,
                "viscosity": viscosity,
                "tail_start_wavenumber": tail_start,
                "tail_policy": (
                    "upper-admissible-third"
                    if tail_start_wavenumber is None
                    else "declared-wavenumber"
                ),
                "reality_tolerance": reality,
                "solenoidal_tolerance": solenoidal,
                "sgs_filter": filter_id,
                "sgs_model": model_id,
                "sgs_prepared_model": prepared_model_id,
                "sgs_prepared_action": prepared_action_id,
                "sgs_regularization": regularization_id,
                "sgs_dynamic_provenance": dynamic_provenance_id,
                "sgs_averaging": averaging_id,
                "sgs_backscatter": backscatter_id,
                "terms": (
                    "molecular-dissipation",
                    "advective-transfer",
                    "sgs-transfer",
                    "forcing-injection",
                    "resolved-advective-flux",
                ),
                "storage": "full-complex-no-hermitian-multiplicity",
            }
        )

    def evaluate(
        self,
        time: ArrayLike,
        velocity: ArrayLike,
        args: Any = None,
        /,
        *,
        stage: PeriodicIncompressibleStage | None = None,
        additive_forcing_rate: ArrayLike | None = None,
        step_restriction: PeriodicLESStepRestriction | None = None,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> PeriodicModalTurbulenceStatistics:
        value = self.projector.validate_state(velocity)
        finite_velocity = jnp.all(jnp.isfinite(value))
        clean_velocity = jnp.where(finite_velocity, value, jnp.zeros_like(value))
        velocity_ = self.projector.zero_forbidden_modes(clean_velocity)
        stage_ = (
            self.dynamics.stage(
                jnp.asarray(time),
                velocity_,
                args,
                continuation_state=continuation_state,
                accepted_update_mask=accepted_update_mask,
            )
            if stage is None
            else stage
        )
        if not isinstance(stage_, PeriodicIncompressibleStage):
            raise TypeError("stage must be PeriodicIncompressibleStage or None.")
        algebraic_stage = stage_.algebraic_les
        dynamic_stage = stage_.dynamic_les
        if self.dynamics.algebraic_les is not None:
            if algebraic_stage is None or dynamic_stage is not None:
                raise ValueError("Static LES statistics require a static LES stage.")
            if algebraic_stage.prepared_id != self.dynamics.algebraic_les.prepared_id:
                raise ValueError("LES stage belongs to another prepared LES action.")
            if step_restriction is None:
                step_restriction = self.dynamics.step_restriction(
                    velocity_,
                    algebraic_les_stage=algebraic_stage,
                )
        elif self.dynamics.dynamic_les is not None:
            if dynamic_stage is None or algebraic_stage is not None:
                raise ValueError("Dynamic LES statistics require a dynamic LES stage.")
            if dynamic_stage.prepared_id != self.dynamics.dynamic_les.prepared_id:
                raise ValueError("Dynamic stage belongs to another prepared LES action.")
            if step_restriction is None:
                step_restriction = self.dynamics.step_restriction(
                    velocity_,
                    dynamic_les_stage=dynamic_stage,
                )
        elif (
            algebraic_stage is not None
            or dynamic_stage is not None
            or step_restriction is not None
        ):
            raise ValueError("A no-LES statistics plan cannot consume LES evidence.")
        if step_restriction is not None:
            if not isinstance(step_restriction, PeriodicLESStepRestriction):
                raise TypeError(
                    "step_restriction must be PeriodicLESStepRestriction or None."
                )
            if step_restriction.prepared_id != self.sgs_prepared_action_id:
                raise ValueError(
                    "LES step restriction belongs to another prepared action."
                )
        rates = stage_.rates
        advective = self.projector.validate_state(
            rates.advective_rate, owner="Advective rate"
        )
        molecular = self.projector.validate_state(
            rates.molecular_rate, owner="Molecular rate"
        )
        sgs = self.projector.validate_state(rates.sgs_rate, owner="SGS rate")
        compiled_forcing = self.projector.validate_state(
            rates.forcing_rate, owner="Compiled forcing rate"
        )
        forcing_available = self.dynamics.problem.forcing is not None
        if additive_forcing_rate is None:
            forcing = compiled_forcing
            finite_additive_forcing = jnp.asarray(True)
        else:
            if forcing_available:
                raise ValueError(
                    "additive_forcing_rate cannot supplement compiled forcing."
                )
            additive = self.projector.validate_state(
                additive_forcing_rate, owner="Additive forcing rate"
            )
            finite_additive_forcing = jnp.all(jnp.isfinite(additive))
            forcing = compiled_forcing + additive
            forcing_available = True
        finite_rates = (
            jnp.all(jnp.isfinite(advective))
            & jnp.all(jnp.isfinite(molecular))
            & jnp.all(jnp.isfinite(sgs))
            & jnp.all(jnp.isfinite(compiled_forcing))
            & finite_additive_forcing
        )
        advective = self.projector.zero_forbidden_modes(
            jnp.where(finite_rates, advective, jnp.zeros_like(advective))
        )
        molecular = self.projector.zero_forbidden_modes(
            jnp.where(finite_rates, molecular, jnp.zeros_like(molecular))
        )
        sgs = self.projector.zero_forbidden_modes(
            jnp.where(finite_rates, sgs, jnp.zeros_like(sgs))
        )
        forcing = self.projector.zero_forbidden_modes(
            jnp.where(finite_rates, forcing, jnp.zeros_like(forcing))
        )
        modal_energy = 0.5 * jnp.sum(jnp.abs(velocity_) ** 2, axis=-1)
        modal_molecular_dissipation = -jnp.real(
            ein.contract("...i,...i->...", jnp.conj(velocity_), molecular)
        )
        modal_advective_transfer = jnp.real(
            ein.contract("...i,...i->...", jnp.conj(velocity_), advective)
        )
        modal_sgs_transfer = jnp.real(
            ein.contract("...i,...i->...", jnp.conj(velocity_), sgs)
        )
        modal_forcing_injection = jnp.real(
            ein.contract("...i,...i->...", jnp.conj(velocity_), forcing)
        )
        energy_shells = _modal_shell_statistic(
            self.geometry, modal_energy, "kinetic-energy"
        )
        molecular_shells = _modal_shell_statistic(
            self.geometry,
            modal_molecular_dissipation,
            "molecular-dissipation",
        )
        advective_shells = _modal_shell_statistic(
            self.geometry,
            modal_advective_transfer,
            "advective-transfer",
        )
        sgs_shells = _modal_shell_statistic(
            self.geometry,
            modal_sgs_transfer,
            "sgs-transfer",
        )
        forcing_shells = _modal_shell_statistic(
            self.geometry,
            modal_forcing_injection,
            "forcing-injection",
        )
        resolved_flux = -jnp.cumsum(advective_shells.integral)
        kinetic_energy = energy_shells.total
        molecular_dissipation = molecular_shells.total
        advective_energy_rate = advective_shells.total
        sgs_energy_rate = sgs_shells.total
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
        mean_molecular_dissipation = molecular_dissipation / self.volume
        mean_enstrophy = enstrophy / self.volume
        mean_helicity = helicity / self.volume
        scale_valid = (
            (self.projector.spatial_dimension == 3)
            & (self.viscosity > 0.0)
            & (mean_energy > 0.0)
            & (mean_molecular_dissipation > 0.0)
        )
        safe_mean_dissipation = jnp.where(
            mean_molecular_dissipation > 0.0,
            mean_molecular_dissipation,
            1.0,
        )
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
        tail_molecular_dissipation = jnp.sum(
            jnp.where(tail, modal_molecular_dissipation, 0.0)
        )
        energy_tail_valid = kinetic_energy > 0.0
        molecular_tail_valid = molecular_dissipation > 0.0
        energy_tail_fraction = jnp.where(
            energy_tail_valid, tail_energy / safe_energy, 0.0
        )
        safe_molecular_dissipation = jnp.where(
            molecular_dissipation > 0.0,
            molecular_dissipation,
            1.0,
        )
        molecular_tail_fraction = jnp.where(
            molecular_tail_valid,
            tail_molecular_dissipation / safe_molecular_dissipation,
            0.0,
        )
        scalar_dtype = velocity_.real.dtype
        zero = jnp.zeros((), dtype=scalar_dtype)
        zero_count = jnp.asarray(0, dtype=jnp.int32)
        selected_algebraic = (
            algebraic_stage
            if algebraic_stage is not None
            else None
            if dynamic_stage is None
            else dynamic_stage.algebraic_stage
        )
        if selected_algebraic is None:
            sgs_modeled_dissipation = zero
            sgs_identity_defect = zero
            sgs_projection_defect = zero
            sgs_regularization_count = zero_count
            coefficient_minimum = zero
            coefficient_mean = zero
            coefficient_maximum = zero
            backscatter_count = zero_count
            backscatter_limit_count = zero_count
            accepted_count = zero_count
            rejected_count = zero_count
            sgs_maximum_viscosity = zero
            advective_limit = zero
            diffusive_limit = zero
            combined_limit = zero
            etdrk_limit = zero
            explicit_limit = zero
            sgs_available = jnp.asarray(False)
            regularization_available = jnp.asarray(False)
            stability_available = jnp.asarray(False)
            finite_les = jnp.asarray(True)
            successful_les = jnp.asarray(True)
        else:
            sgs_modeled_dissipation = selected_algebraic.modeled_dissipation
            sgs_identity_defect = selected_algebraic.energy_identity_defect
            sgs_projection_defect = selected_algebraic.projection_energy_defect
            sgs_maximum_viscosity = selected_algebraic.maximum_kinematic_viscosity
            advective_limit = step_restriction.advective
            diffusive_limit = step_restriction.algebraic_les_diffusive
            combined_limit = step_restriction.combined_diffusive
            etdrk_limit = step_restriction.etdrk_selected
            explicit_limit = step_restriction.fully_explicit_selected
            sgs_available = jnp.asarray(True)
            stability_available = jnp.asarray(True)
            if dynamic_stage is None:
                sgs_regularization_count = zero_count
                coefficient_minimum = zero
                coefficient_mean = zero
                coefficient_maximum = zero
                backscatter_count = zero_count
                backscatter_limit_count = zero_count
                accepted_count = zero_count
                rejected_count = zero_count
                regularization_available = jnp.asarray(False)
                dynamic_finite = jnp.asarray(True)
                successful_policy = (
                    selected_algebraic.dissipative & selected_algebraic.energy_consistent
                )
            else:
                dynamic_result = dynamic_stage.dynamic_result
                evidence = dynamic_result.evidence
                coefficient = dynamic_result.coefficient
                sgs_regularization_count = evidence.regularization_activity_count
                coefficient_minimum = jnp.min(coefficient)
                coefficient_mean = jnp.mean(coefficient)
                coefficient_maximum = jnp.max(coefficient)
                backscatter_count = evidence.backscatter_activity_count
                backscatter_limit_count = evidence.backscatter_limit_count
                accepted_count = (
                    evidence.accepted_update_count
                    if continuation_state is None
                    else continuation_state.accepted_updates
                )
                rejected_count = (
                    evidence.rejected_update_count
                    if continuation_state is None
                    else continuation_state.rejected_updates
                )
                regularization_available = jnp.asarray(True)
                dynamic_finite = evidence.finite
                successful_policy = evidence.finite
            finite_les = (
                selected_algebraic.finite
                & dynamic_finite
                & jnp.isfinite(sgs_modeled_dissipation)
                & jnp.isfinite(sgs_identity_defect)
                & jnp.isfinite(sgs_projection_defect)
                & jnp.isfinite(sgs_maximum_viscosity)
                & step_restriction.finite
            )
            successful_les = successful_policy & step_restriction.finite
        velocity_reality_defect = _hermitian_defect(velocity_, self.conjugate_indices)
        divergence_norm = self.projector.divergence_norm(velocity_)
        finite = (
            finite_velocity
            & finite_rates
            & finite_les
            & energy_shells.finite
            & molecular_shells.finite
            & advective_shells.finite
            & sgs_shells.finite
            & forcing_shells.finite
            & jnp.all(jnp.isfinite(resolved_flux))
            & jnp.isfinite(enstrophy)
            & jnp.isfinite(helicity)
            & jnp.isfinite(taylor)
            & jnp.isfinite(kolmogorov)
            & jnp.isfinite(integral_scale)
            & jnp.isfinite(energy_tail_fraction)
            & jnp.isfinite(molecular_tail_fraction)
        )
        successful = (
            finite
            & successful_les
            & (velocity_reality_defect <= self.reality_tolerance)
            & (divergence_norm <= self.solenoidal_tolerance)
        )
        return PeriodicModalTurbulenceStatistics(
            energy_shells=energy_shells,
            molecular_dissipation_shells=molecular_shells,
            advective_transfer_shells=advective_shells,
            sgs_transfer_shells=sgs_shells,
            forcing_injection_shells=forcing_shells,
            resolved_spectral_flux=resolved_flux,
            kinetic_energy=kinetic_energy,
            mean_kinetic_energy=mean_energy,
            molecular_dissipation=molecular_dissipation,
            mean_molecular_dissipation=mean_molecular_dissipation,
            advective_energy_rate=advective_energy_rate,
            mean_advective_energy_rate=advective_energy_rate / self.volume,
            sgs_energy_rate=sgs_energy_rate,
            mean_sgs_energy_rate=sgs_energy_rate / self.volume,
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
            molecular_dissipation_tail_fraction=molecular_tail_fraction,
            divergence_norm=divergence_norm,
            velocity_reality_defect=velocity_reality_defect,
            sgs_modeled_dissipation=sgs_modeled_dissipation,
            sgs_energy_identity_defect=sgs_identity_defect,
            sgs_projection_energy_defect=sgs_projection_defect,
            sgs_regularization_activity_count=sgs_regularization_count,
            sgs_dynamic_coefficient_minimum=coefficient_minimum,
            sgs_dynamic_coefficient_mean=coefficient_mean,
            sgs_dynamic_coefficient_maximum=coefficient_maximum,
            sgs_backscatter_activity_count=backscatter_count,
            sgs_backscatter_limit_count=backscatter_limit_count,
            sgs_accepted_update_count=accepted_count,
            sgs_rejected_update_count=rejected_count,
            sgs_maximum_kinematic_viscosity=sgs_maximum_viscosity,
            sgs_advective_step_limit=advective_limit,
            sgs_diffusive_step_limit=diffusive_limit,
            sgs_combined_step_limit=combined_limit,
            sgs_etdrk_step_limit=etdrk_limit,
            sgs_fully_explicit_step_limit=explicit_limit,
            sgs_available=sgs_available,
            sgs_regularization_available=regularization_available,
            sgs_stability_available=stability_available,
            forcing_available=jnp.asarray(forcing_available),
            helicity_valid=helicity_valid,
            taylor_microscale_valid=scale_valid,
            kolmogorov_scale_valid=scale_valid,
            integral_scale_valid=integral_valid,
            energy_tail_valid=energy_tail_valid,
            molecular_dissipation_tail_valid=molecular_tail_valid,
            finite=finite,
            successful=successful,
            tail_start_wavenumber=self.tail_start_wavenumber,
            spectrum_convention=(
                "full-complex native domain integral; density=integral/bin-width"
            ),
            resolved_flux_convention=(
                "-cumulative-sum of ascending-shell advective transfer"
            ),
            integral_scale_convention="3*pi*sum(E_k/|k|)/(4*sum(E_k))",
            tail_convention="modes with |k| >= tail_start_wavenumber",
            source_problem_id=self.source_problem_id,
            compilation_id=self.compilation_id,
            discretization_id=self.discretization_id,
            projector_id=self.projector_id,
            sgs_filter_id=self.sgs_filter_id,
            sgs_model_id=self.sgs_model_id,
            sgs_prepared_model_id=self.sgs_prepared_model_id,
            sgs_prepared_action_id=self.sgs_prepared_action_id,
            sgs_regularization_id=self.sgs_regularization_id,
            sgs_dynamic_provenance_id=self.sgs_dynamic_provenance_id,
            sgs_averaging_id=self.sgs_averaging_id,
            sgs_backscatter_id=self.sgs_backscatter_id,
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
