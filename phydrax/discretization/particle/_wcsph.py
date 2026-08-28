#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
    resolved_identifier,
)
from ._core import ParticleDiscretization
from ._free_surface import (
    detect_free_surface,
    FreeSurfaceDetectionPlan,
    FreeSurfaceOperatorCorrectionPlan,
    FreeSurfacePressurePlan,
    FreeSurfaceState,
)
from ._graph import particle_graph_view
from ._neighborhood import (
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pairwise import particle_pair_geometry, ParticlePairGeometry
from ._precision import ParticleExecutionPolicy, ParticlePrecisionPolicy
from ._smoothing import AbstractSPHSmoothingKernel
from ._sph_density import AbstractSPHDensityPlan, ContinuityDensityPlan
from ._sph_operators import (
    sph_continuity_density_rate,
    sph_morris_viscous_force,
    sph_summation_density,
    sph_symmetric_pressure_gradient,
    SPHViscousForceResult,
)
from ._sph_state import WeaklyCompressibleSPHStateLayout
from ._sph_viscosity import MorrisViscosityPlan
from ._stabilization import (
    AbstractSPHDensityDiffusionPlan,
    ArtificialViscosityResult,
    MonaghanArtificialViscosityPlan,
    sph_artificial_viscosity_force,
    sph_density_diffusion_rate,
    SPHDensityDiffusionResult,
)


ExternalParticleAcceleration = Callable[[Array, Array, Array, Array, Any], ArrayLike]


class WeaklyCompressibleSPHMethodPlan(StrictModule, NonTrainableState):
    """First-order fixed-h weakly compressible SPH method."""

    kernel: AbstractSPHSmoothingKernel
    density: AbstractSPHDensityPlan
    physical_viscosity: MorrisViscosityPlan | None
    artificial_viscosity: MonaghanArtificialViscosityPlan | None
    density_diffusion: AbstractSPHDensityDiffusionPlan | None
    free_surface_detection: FreeSurfaceDetectionPlan | None
    free_surface_pressure: FreeSurfacePressurePlan | None
    free_surface_correction: FreeSurfaceOperatorCorrectionPlan | None
    smoothing_length: float = eqx.field(static=True)
    acoustic_cfl: float = eqx.field(static=True)
    force_cfl: float = eqx.field(static=True)
    viscous_cfl: float = eqx.field(static=True)
    key: DiscretizationKey
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractSPHSmoothingKernel,
        smoothing_length: float,
        /,
        *,
        density: AbstractSPHDensityPlan,
        physical_viscosity: MorrisViscosityPlan | None = None,
        artificial_viscosity: MonaghanArtificialViscosityPlan | None = None,
        density_diffusion: AbstractSPHDensityDiffusionPlan | None = None,
        free_surface_detection: FreeSurfaceDetectionPlan | None = None,
        free_surface_pressure: FreeSurfacePressurePlan | None = None,
        free_surface_correction: FreeSurfaceOperatorCorrectionPlan | None = None,
        acoustic_cfl: float = 0.25,
        force_cfl: float = 0.25,
        viscous_cfl: float = 0.125,
        name: str = "weakly-compressible-sph",
        method_id: str | None = None,
    ):
        if not isinstance(kernel, AbstractSPHSmoothingKernel):
            raise TypeError("kernel must be an AbstractSPHSmoothingKernel.")
        if not isinstance(density, AbstractSPHDensityPlan):
            raise TypeError("density must be an AbstractSPHDensityPlan.")
        if physical_viscosity is not None and not isinstance(
            physical_viscosity, MorrisViscosityPlan
        ):
            raise TypeError("physical_viscosity must be a MorrisViscosityPlan or None.")
        if artificial_viscosity is not None and not isinstance(
            artificial_viscosity, MonaghanArtificialViscosityPlan
        ):
            raise TypeError(
                "artificial_viscosity must be a MonaghanArtificialViscosityPlan or None."
            )
        if density_diffusion is not None and not isinstance(
            density_diffusion, AbstractSPHDensityDiffusionPlan
        ):
            raise TypeError(
                "density_diffusion must be an SPH density-diffusion plan or None."
            )
        if density_diffusion is not None and not isinstance(
            density, ContinuityDensityPlan
        ):
            raise ValueError("Density diffusion requires ContinuityDensityPlan.")
        if free_surface_pressure is not None and free_surface_detection is None:
            raise ValueError("free_surface_pressure requires free_surface_detection.")
        smoothing = float(smoothing_length)
        coefficients = tuple(
            float(value) for value in (acoustic_cfl, force_cfl, viscous_cfl)
        )
        if not np.isfinite(smoothing) or smoothing <= 0.0:
            raise ValueError("smoothing_length must be finite and positive.")
        if any(not np.isfinite(value) or value <= 0.0 for value in coefficients):
            raise ValueError("WCSPH CFL coefficients must be finite and positive.")
        key = DiscretizationKey(
            name,
            DiscretizationRole.RESIDUAL,
            domain_labels=("material_point", "weakly_compressible_fluid"),
        )
        self.kernel = kernel
        self.density = density
        self.physical_viscosity = physical_viscosity
        self.artificial_viscosity = artificial_viscosity
        self.density_diffusion = density_diffusion
        self.free_surface_detection = free_surface_detection
        self.free_surface_pressure = free_surface_pressure
        self.free_surface_correction = free_surface_correction
        self.smoothing_length = smoothing
        self.acoustic_cfl, self.force_cfl, self.viscous_cfl = coefficients
        self.key = key
        self.method_id = resolved_identifier(
            "method_id",
            method_id,
            {
                "kind": "weakly-compressible-sph-method-plan",
                "kernel": kernel.kernel_id,
                "smoothing_length": smoothing,
                "density": density.plan_id,
                "physical_viscosity": None
                if physical_viscosity is None
                else physical_viscosity.plan_id,
                "artificial_viscosity": None
                if artificial_viscosity is None
                else artificial_viscosity.plan_id,
                "density_diffusion": None
                if density_diffusion is None
                else density_diffusion.plan_id,
                "free_surface_detection": None
                if free_surface_detection is None
                else free_surface_detection.plan_id,
                "free_surface_pressure": None
                if free_surface_pressure is None
                else free_surface_pressure.plan_id,
                "acoustic_cfl": self.acoustic_cfl,
                "force_cfl": self.force_cfl,
                "viscous_cfl": self.viscous_cfl,
                "key": key.key_id,
            },
        )


class WeaklyCompressibleSPHStepRestriction(StrictModule):
    acoustic: Array
    force: Array
    viscous: Array
    selected: Array


class WeaklyCompressibleSPHDiagnostics(StrictModule):
    density_minimum: Array
    density_maximum: Array
    density_mean: Array
    reference_density_error: Array
    neighbor_count_minimum: Array
    neighbor_count_maximum: Array
    total_mass: Array
    linear_momentum: Array
    angular_momentum: Array
    kinetic_energy: Array
    internal_energy: Array
    total_energy: Array
    pressure_kinetic_power: Array
    internal_energy_rate: Array
    pressure_energy_balance_defect: Array
    viscous_power: Array
    viscous_dissipation_rate: Array
    viscous_positive_power_defect: Array
    artificial_viscosity_power: Array
    artificial_viscosity_dissipation: Array
    artificial_positive_power_defect: Array
    density_diffusion_energy_rate: Array
    density_variance_rate: Array
    net_artificial_force: Array
    free_surface_count: Array
    free_surface_ambiguity_count: Array
    external_force: Array
    external_power: Array
    total_energy_rate: Array
    net_pressure_force: Array
    net_viscous_force: Array
    net_viscous_torque: Array
    active_pairs: Array
    pair_count: Array
    maximum_cell_occupancy: Array
    neighborhood_successful: Array
    admissible: Array


class _WeaklyCompressibleSPHEvaluation(StrictModule):
    position: Array
    velocity: Array
    density: Array
    pressure: Array
    neighborhood: ParticleNeighborhoodState
    geometry: ParticlePairGeometry
    physical_pairs: Array
    pressure_gradient: Array
    pressure_force: Array
    viscous: SPHViscousForceResult
    artificial: ArtificialViscosityResult
    density_diffusion: SPHDensityDiffusionResult | None
    free_surface: FreeSurfaceState | None
    external_acceleration: Array
    density_rate: Array | None


class PreparedWeaklyCompressibleSPHDynamics(StrictModule, NonTrainableState):
    """Prepared first-order WCSPH drift with complete balance evidence."""

    particles: ParticleDiscretization
    neighborhood: AbstractPreparedParticleNeighborhood
    method: WeaklyCompressibleSPHMethodPlan
    material: Any
    external_acceleration: ExternalParticleAcceleration | None
    external_acceleration_id: str | None = eqx.field(static=True)
    execution: ParticleExecutionPolicy
    precision: ParticlePrecisionPolicy
    state_layout: WeaklyCompressibleSPHStateLayout
    key: DiscretizationKey
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        neighborhood: AbstractPreparedParticleNeighborhood,
        method: WeaklyCompressibleSPHMethodPlan,
        material: Any,
        /,
        *,
        execution: ParticleExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
        external_acceleration: ExternalParticleAcceleration | None = None,
        external_acceleration_id: str | None = None,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError(
                "neighborhood must be an AbstractPreparedParticleNeighborhood."
            )
        if not isinstance(method, WeaklyCompressibleSPHMethodPlan):
            raise TypeError("method must be a WeaklyCompressibleSPHMethodPlan.")
        if neighborhood.particle_discretization_id != particles.prepared_id:
            raise ValueError(
                "Particle neighborhood was prepared for a different support."
            )
        if method.kernel.dimension != particles.ambient_dimension:
            raise ValueError("SPH kernel dimension does not match particle support.")
        if external_acceleration is not None and not callable(external_acceleration):
            raise TypeError("external_acceleration must be callable or None.")
        if external_acceleration is None and external_acceleration_id is not None:
            raise ValueError("external_acceleration_id requires external acceleration.")
        if external_acceleration is not None and not external_acceleration_id:
            raise ValueError("External acceleration requires a stable non-empty ID.")
        execution_ = (
            ParticleExecutionPolicy(realization=neighborhood.backend)
            if execution is None
            else execution
        )
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        if execution_.realization != neighborhood.backend:
            raise ValueError(
                "Particle execution realization does not match the prepared neighborhood backend."
            )
        state_layout = WeaklyCompressibleSPHStateLayout(
            particles, density_evolved=method.density.density_evolved
        )
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.MATRIX_FREE,
            ),
            diagnostics=(
                "fixed smoothing length",
                "first-order weakly compressible SPH",
                "one neighborhood build per drift evaluation",
                "pairwise pressure and viscosity exchange",
                "branchwise fixed-capacity neighborhood decisions",
            ),
            resource_counts={
                "particle_capacity": particles.capacity,
                "active_particles": particles.active_count,
                "pair_capacity": neighborhood.pair_capacity,
                "state_width": state_layout.width,
                "ambient_dimension": particles.ambient_dimension,
            },
        )
        self.particles = particles
        self.neighborhood = neighborhood
        self.method = method
        self.material = material
        self.external_acceleration = external_acceleration
        self.external_acceleration_id = external_acceleration_id
        self.execution = execution_
        self.precision = precision_
        self.state_layout = state_layout
        self.key = method.key
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-weakly-compressible-sph-dynamics",
                "particles": particles.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "method": method.method_id,
                "material": material.material_id,
                "external_acceleration": external_acceleration_id,
                "execution": execution_.policy_id,
                "precision": precision_.policy_id,
                "state_layout": state_layout.layout_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.precision.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def _physical_pair_mask(self, geometry: ParticlePairGeometry, /) -> Array:
        support = self.method.kernel.support_radius(self.method.smoothing_length)
        return geometry.valid & (geometry.distance < support)

    def _summation_density(
        self,
        neighborhood: ParticleNeighborhoodState,
        geometry: ParticlePairGeometry,
        physical_pairs: Array,
        /,
    ) -> Array:
        return sph_summation_density(
            self.particles.safe_masses,
            self.particles.active_mask,
            neighborhood.pair_relation,
            geometry,
            physical_pairs,
            self.method.kernel,
            self.method.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )

    def _external_acceleration(
        self,
        time: ArrayLike,
        position: Array,
        velocity: Array,
        density: Array,
        args: Any,
        /,
    ) -> Array:
        if self.external_acceleration is None:
            return jnp.zeros_like(position)
        acceleration = self.precision.evaluation(
            self.external_acceleration(
                jnp.asarray(time), position, velocity, density, args
            )
        )
        if acceleration.shape != position.shape:
            raise ValueError("external_acceleration must match particle position shape.")
        active = self.particles.active_mask[:, None]
        acceleration = eqx.error_if(
            acceleration,
            jnp.any(jnp.where(active, ~jnp.isfinite(acceleration), False)),
            "Active external particle accelerations must be finite.",
        )
        return jnp.where(active, acceleration, 0.0)

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        density: ArrayLike | None = None,
        /,
    ) -> Array:
        position_ = self.state_layout._vector("position", position)
        velocity_ = self.state_layout._vector("velocity", velocity)
        if self.method.density.density_evolved and density is None:
            neighborhood = self.neighborhood.build(position_)
            position_ = neighborhood.require_success(position_)
            geometry = particle_pair_geometry(
                position_, neighborhood.pair_relation, box=self.neighborhood.box
            )
            density = self._summation_density(
                neighborhood,
                geometry,
                self._physical_pair_mask(geometry),
            )
        state = self.state_layout.pack(position_, velocity_, density)
        if self.method.density.density_evolved:
            density_ = self.state_layout.density(state)
            density_ = eqx.error_if(
                density_,
                jnp.any(
                    jnp.where(
                        self.particles.active_mask,
                        ~self.material.admissible(density_),
                        False,
                    )
                ),
                "Initial WCSPH density is not material-admissible.",
            )
        return state

    def _evaluate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> _WeaklyCompressibleSPHEvaluation:
        state_ = self.state_layout.validate(state)
        position, velocity, state_density = self.state_layout.unpack(state_)
        neighborhood = self.neighborhood.build(position)
        position = neighborhood.require_success(position)
        geometry = particle_pair_geometry(
            position, neighborhood.pair_relation, box=self.neighborhood.box
        )
        physical_pairs = self._physical_pair_mask(geometry)
        density = (
            self._summation_density(neighborhood, geometry, physical_pairs)
            if state_density is None
            else self.precision.evaluation(state_density)
        )
        density = eqx.error_if(
            density,
            jnp.any(
                jnp.where(
                    self.particles.active_mask,
                    ~self.material.admissible(density),
                    False,
                )
            ),
            "WCSPH density is not material-admissible.",
        )
        safe_density = jnp.where(
            self.particles.active_mask,
            density,
            jnp.asarray(self.material.reference_density, dtype=density.dtype),
        )
        pressure = self.precision.evaluation(self.material.pressure(safe_density))
        pressure = jnp.where(self.particles.active_mask, pressure, 0.0)
        surface = None
        if self.method.free_surface_detection is not None:
            surface = detect_free_surface(
                self.method.free_surface_detection,
                self.particles,
                density,
                neighborhood.pair_relation,
                geometry,
                physical_pairs,
                self.method.kernel,
                self.method.smoothing_length,
                self.execution,
            )
            if self.method.free_surface_correction is not None:
                pressure = self.method.free_surface_correction.normalize(
                    pressure, surface
                )
            if self.method.free_surface_pressure is not None:
                pressure = self.method.free_surface_pressure.apply(pressure, surface)
        pressure_gradient = sph_symmetric_pressure_gradient(
            self.particles.safe_masses,
            density,
            pressure,
            neighborhood.pair_relation,
            geometry,
            physical_pairs,
            self.method.kernel,
            self.method.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )
        pair_shape = (neighborhood.pair_relation.capacity,)
        if self.method.physical_viscosity is None:
            viscous = SPHViscousForceResult(
                force=jnp.zeros_like(position),
                pair_force=jnp.zeros(
                    pair_shape + (self.particles.ambient_dimension,),
                    dtype=position.dtype,
                ),
                pair_power=jnp.zeros(pair_shape, dtype=position.dtype),
                dissipation_rate=jnp.zeros((), dtype=position.dtype),
                positive_power_defect=jnp.zeros((), dtype=position.dtype),
            )
        else:
            viscous = sph_morris_viscous_force(
                self.particles.safe_masses,
                density,
                velocity,
                neighborhood.pair_relation,
                geometry,
                physical_pairs,
                self.method.kernel,
                self.method.smoothing_length,
                self.method.physical_viscosity.kinematic_viscosity,
                self.method.physical_viscosity.regularization,
                particle_count=self.particles.capacity,
                execution=self.execution,
                precision=self.precision,
            )
        sound_speed = self.material.sound_speed(density)
        if self.method.artificial_viscosity is None:
            artificial = ArtificialViscosityResult(
                force=jnp.zeros_like(position),
                pair_power=jnp.zeros(pair_shape, dtype=position.dtype),
                dissipation_rate=jnp.zeros((), dtype=position.dtype),
                positive_power_defect=jnp.zeros((), dtype=position.dtype),
                active_pairs=jnp.zeros((), dtype=jnp.int32),
            )
        else:
            artificial = sph_artificial_viscosity_force(
                self.method.artificial_viscosity,
                self.particles,
                density,
                sound_speed,
                velocity,
                neighborhood.pair_relation,
                geometry,
                physical_pairs,
                self.method.kernel,
                self.method.smoothing_length,
                self.execution,
            )
        external = self._external_acceleration(time, position, velocity, density, args)
        density_rate = None
        density_diffusion = None
        if isinstance(self.method.density, ContinuityDensityPlan):
            density_rate = sph_continuity_density_rate(
                self.particles.safe_masses,
                velocity,
                neighborhood.pair_relation,
                geometry,
                physical_pairs,
                self.method.kernel,
                self.method.smoothing_length,
                particle_count=self.particles.capacity,
                execution=self.execution,
                precision=self.precision,
            )
            if self.method.density_diffusion is not None:
                density_diffusion = sph_density_diffusion_rate(
                    self.method.density_diffusion,
                    self.particles,
                    density,
                    sound_speed,
                    neighborhood.pair_relation,
                    geometry,
                    physical_pairs,
                    self.method.kernel,
                    self.method.smoothing_length,
                    self.execution,
                    free_surface_weight=None
                    if surface is None
                    else surface.smooth_weight,
                )
                density_rate = density_rate + density_diffusion.rate
        return _WeaklyCompressibleSPHEvaluation(
            position,
            velocity,
            density,
            pressure,
            neighborhood,
            geometry,
            physical_pairs,
            pressure_gradient,
            -pressure_gradient,
            viscous,
            artificial,
            density_diffusion,
            surface,
            external,
            density_rate,
        )

    def __call__(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        evaluation = self._evaluate(time, state, args)
        masses = self.precision.evaluation(self.particles.safe_masses)
        acceleration = (
            evaluation.pressure_force
            + evaluation.viscous.force
            + evaluation.artificial.force
        ) / masses[:, None] + evaluation.external_acceleration
        acceleration = jnp.where(self.particles.active_mask[:, None], acceleration, 0.0)
        return self.state_layout.pack_rate(
            evaluation.velocity,
            acceleration,
            evaluation.density_rate,
        )

    def graph_view(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        directed: bool = True,
    ):
        evaluation = self._evaluate(time, state, args)
        return particle_graph_view(
            self.particles,
            evaluation.neighborhood,
            evaluation.position,
            directed=directed,
            edge_mask=evaluation.physical_pairs,
            geometry=evaluation.geometry,
        )

    @staticmethod
    def _angular_sum(position: Array, vector: Array, /) -> Array:
        dimension = int(position.shape[-1])
        if dimension == 1:
            return jnp.zeros((), dtype=position.dtype)
        if dimension == 2:
            return compensated_sum(
                position[:, 0] * vector[:, 1] - position[:, 1] * vector[:, 0]
            )
        if dimension == 3:
            return compensated_sum(jnp.cross(position, vector), axis=0)
        raise ValueError("Angular diagnostics support dimensions 1, 2, and 3.")

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> WeaklyCompressibleSPHDiagnostics:
        evaluation = self._evaluate(time, state, args)
        active = self.particles.active_mask
        masses = self.precision.certification(self.particles.safe_masses)
        position = self.precision.certification(evaluation.position)
        velocity = self.precision.certification(evaluation.velocity)
        density = self.precision.certification(evaluation.density)
        momentum = masses[:, None] * velocity
        pressure_force = self.precision.certification(evaluation.pressure_force)
        viscous_force = self.precision.certification(evaluation.viscous.force)
        artificial_force = self.precision.certification(evaluation.artificial.force)
        external_force = masses[:, None] * self.precision.certification(
            evaluation.external_acceleration
        )
        kinetic_terms = jnp.where(
            active,
            0.5 * masses * jnp.sum(velocity * velocity, axis=-1),
            0.0,
        )
        internal_terms = jnp.where(
            active,
            masses * self.material.specific_internal_energy(density),
            0.0,
        )
        pressure_kinetic_power = compensated_sum(
            jnp.sum(velocity * pressure_force, axis=-1)
        )
        if evaluation.density_rate is None:
            internal_energy_rate = compensated_sum(
                jnp.sum(velocity * evaluation.pressure_gradient, axis=-1)
            )
        else:
            internal_energy_rate = compensated_sum(
                jnp.where(
                    active,
                    masses * evaluation.pressure / density**2 * evaluation.density_rate,
                    0.0,
                )
            )
        diffusion_energy_rate = (
            jnp.zeros((), dtype=density.dtype)
            if evaluation.density_diffusion is None
            else compensated_sum(
                jnp.where(
                    active,
                    masses
                    * evaluation.pressure
                    / density**2
                    * evaluation.density_diffusion.rate,
                    0.0,
                )
            )
        )
        pressure_internal_rate = internal_energy_rate - diffusion_energy_rate
        external_power = compensated_sum(jnp.sum(velocity * external_force, axis=-1))
        counts = jnp.zeros((self.particles.capacity,), dtype=jnp.int32)
        increments = evaluation.physical_pairs.astype(jnp.int32)
        pairs = evaluation.neighborhood.pair_relation
        counts = counts.at[pairs.left_indices].add(increments)
        counts = counts.at[pairs.right_indices].add(increments)
        density_mean = (
            compensated_sum(jnp.where(active, density, 0.0)) / self.particles.active_count
        )
        reference = jnp.asarray(self.material.reference_density, dtype=density.dtype)
        reference_error = jnp.max(jnp.where(active, jnp.abs(density - reference), 0.0))
        pressure_balance = pressure_kinetic_power + pressure_internal_rate
        total_energy_rate = (
            pressure_balance
            - evaluation.viscous.dissipation_rate
            - evaluation.artificial.dissipation_rate
            + diffusion_energy_rate
            + external_power
        )
        return WeaklyCompressibleSPHDiagnostics(
            density_minimum=jnp.min(jnp.where(active, density, jnp.inf)),
            density_maximum=jnp.max(jnp.where(active, density, -jnp.inf)),
            density_mean=density_mean,
            reference_density_error=reference_error,
            neighbor_count_minimum=jnp.min(
                jnp.where(active, counts, jnp.iinfo(jnp.int32).max)
            ),
            neighbor_count_maximum=jnp.max(jnp.where(active, counts, 0)),
            total_mass=compensated_sum(jnp.where(active, masses, 0.0)),
            linear_momentum=compensated_sum(
                jnp.where(active[:, None], momentum, 0.0), axis=0
            ),
            angular_momentum=self._angular_sum(position, momentum),
            kinetic_energy=compensated_sum(kinetic_terms),
            internal_energy=compensated_sum(internal_terms),
            total_energy=compensated_sum(kinetic_terms) + compensated_sum(internal_terms),
            pressure_kinetic_power=pressure_kinetic_power,
            internal_energy_rate=internal_energy_rate,
            pressure_energy_balance_defect=pressure_balance,
            viscous_power=-evaluation.viscous.dissipation_rate,
            viscous_dissipation_rate=evaluation.viscous.dissipation_rate,
            viscous_positive_power_defect=evaluation.viscous.positive_power_defect,
            artificial_viscosity_power=-evaluation.artificial.dissipation_rate,
            artificial_viscosity_dissipation=evaluation.artificial.dissipation_rate,
            artificial_positive_power_defect=evaluation.artificial.positive_power_defect,
            density_diffusion_energy_rate=diffusion_energy_rate,
            density_variance_rate=jnp.zeros((), dtype=density.dtype)
            if evaluation.density_diffusion is None
            else evaluation.density_diffusion.variance_rate,
            free_surface_count=jnp.zeros((), dtype=jnp.int32)
            if evaluation.free_surface is None
            else jnp.sum(evaluation.free_surface.hard_mask.astype(jnp.int32)),
            free_surface_ambiguity_count=jnp.zeros((), dtype=jnp.int32)
            if evaluation.free_surface is None
            else jnp.sum(evaluation.free_surface.ambiguous_mask.astype(jnp.int32)),
            external_force=compensated_sum(external_force, axis=0),
            external_power=external_power,
            total_energy_rate=total_energy_rate,
            net_pressure_force=compensated_sum(pressure_force, axis=0),
            net_viscous_force=compensated_sum(viscous_force, axis=0),
            net_artificial_force=compensated_sum(artificial_force, axis=0),
            net_viscous_torque=self._angular_sum(position, viscous_force),
            active_pairs=jnp.sum(evaluation.physical_pairs.astype(jnp.int32)),
            pair_count=evaluation.neighborhood.pair_count,
            maximum_cell_occupancy=evaluation.neighborhood.maximum_cell_occupancy,
            neighborhood_successful=evaluation.neighborhood.successful,
            admissible=jnp.all(
                jnp.where(active, self.material.admissible(density), True)
            ),
        )

    def stable_step(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> WeaklyCompressibleSPHStepRestriction:
        evaluation = self._evaluate(time, state, args)
        sound_speed = self.material.sound_speed(evaluation.density)
        maximum_sound_speed = jnp.max(
            jnp.where(self.particles.active_mask, sound_speed, 0.0)
        )
        acoustic = (
            self.method.acoustic_cfl
            * self.method.smoothing_length
            / jnp.maximum(maximum_sound_speed, jnp.finfo(sound_speed.dtype).tiny)
        )
        masses = self.precision.evaluation(self.particles.safe_masses)
        acceleration = (
            evaluation.pressure_force
            + evaluation.viscous.force
            + evaluation.artificial.force
        ) / masses[:, None] + evaluation.external_acceleration
        maximum_acceleration = jnp.max(
            jnp.where(
                self.particles.active_mask,
                jnp.sqrt(jnp.sum(acceleration * acceleration, axis=-1)),
                0.0,
            )
        )
        force = jnp.where(
            maximum_acceleration > 0.0,
            self.method.force_cfl
            * jnp.sqrt(self.method.smoothing_length / maximum_acceleration),
            jnp.asarray(jnp.inf, dtype=acoustic.dtype),
        )
        viscosity = jnp.asarray(
            0.0
            if self.method.physical_viscosity is None
            else self.method.physical_viscosity.kinematic_viscosity,
            dtype=acoustic.dtype,
        )
        positive_viscosity = viscosity > 0.0
        viscous = jnp.where(
            positive_viscosity,
            self.method.viscous_cfl
            * self.method.smoothing_length**2
            / jnp.where(positive_viscosity, viscosity, 1.0),
            jnp.asarray(jnp.inf, dtype=acoustic.dtype),
        )
        return WeaklyCompressibleSPHStepRestriction(
            acoustic=acoustic,
            force=force,
            viscous=viscous,
            selected=jnp.minimum(acoustic, jnp.minimum(force, viscous)),
        )

    def linearize(
        self,
        time: Array,
        state: Array,
        args: Any = None,
        /,
    ):
        value, jvp = jax.linearize(lambda current: self(time, current, args), state)
        _, vjp = jax.vjp(lambda current: self(time, current, args), state)
        return value, jvp, vjp


__all__ = [
    "ExternalParticleAcceleration",
    "PreparedWeaklyCompressibleSPHDynamics",
    "WeaklyCompressibleSPHDiagnostics",
    "WeaklyCompressibleSPHMethodPlan",
    "WeaklyCompressibleSPHStepRestriction",
]
