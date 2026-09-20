#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matched relativistic particle-to-grid stress-energy transfer."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from ...metrix import StressEnergyProjection
from ..splatting import ParticleGridSplatState, PreparedParticleGridSplat


def _identity(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _real(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a real floating-point dtype.")
    return array


def _integer(value: ArrayLike, name: str, /, *, dtype) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    return array.astype(dtype)


def _frame_token(frame: LocalRelativisticFramePlan, /) -> Array:
    return jnp.asarray(frame.frame_token, dtype=jnp.int32).reshape(())


class RelativisticParticleState(StrictModule):
    """Fixed-capacity particles with exact stage, lineage, and frame identity.

    Positions are comoving coordinate positions. ``local_momenta`` are physical
    spatial momenta in the supplied orthonormal frame; ``covariant_momenta`` are
    spatial coordinate covectors in the declared momentum unit. ``weights`` are
    dimensionless particle multiplicities, never inferred masses.
    """

    particle_ids: Array
    weights: Array
    positions: Array
    local_momenta: Array
    covariant_momenta: Array
    species_ids: Array
    active_mask: Array
    incarnations: Array
    lineage_ids: Array
    time: Array
    scale_factor: Array
    frame_token: Array
    frame_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    frame_lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_ids: ArrayLike,
        weights: ArrayLike,
        positions: ArrayLike,
        local_momenta: ArrayLike,
        covariant_momenta: ArrayLike,
        species_ids: ArrayLike,
        active_mask: ArrayLike,
        incarnations: ArrayLike,
        lineage_ids: ArrayLike,
        time: ArrayLike,
        scale_factor: ArrayLike,
        frame_token: ArrayLike,
        /,
        *,
        frame_id: str,
        topology_id: str,
        frame_lineage_id: str,
    ):
        particle_ids_ = _integer(particle_ids, "particle_ids", dtype=jnp.int64)
        if particle_ids_.ndim != 1 or particle_ids_.size == 0:
            raise ValueError("particle_ids must be a non-empty rank-one array.")
        capacity = particle_ids_.size
        weights_ = _real(weights, "weights")
        positions_ = _real(positions, "positions")
        local_ = _real(local_momenta, "local_momenta")
        covariant_ = _real(covariant_momenta, "covariant_momenta")
        species_ = _integer(species_ids, "species_ids", dtype=jnp.int32)
        active_ = jnp.asarray(active_mask, dtype=jnp.bool_)
        incarnations_ = _integer(incarnations, "incarnations", dtype=jnp.int32)
        lineage_ = _integer(lineage_ids, "lineage_ids", dtype=jnp.int64)
        expected_vector = (capacity, 3)
        for value, name in (
            (positions_, "positions"),
            (local_, "local_momenta"),
            (covariant_, "covariant_momenta"),
        ):
            if value.shape != expected_vector:
                raise ValueError(f"{name} must have shape {expected_vector}.")
        for value, name in (
            (weights_, "weights"),
            (species_, "species_ids"),
            (active_, "active_mask"),
            (incarnations_, "incarnations"),
            (lineage_, "lineage_ids"),
        ):
            if value.shape != (capacity,):
                raise ValueError(f"{name} must have particle-capacity shape.")
        if local_.dtype != positions_.dtype or covariant_.dtype != positions_.dtype:
            raise TypeError("Particle position and momentum arrays must share one dtype.")
        if weights_.dtype != positions_.dtype:
            weights_ = weights_.astype(positions_.dtype)
        time_ = _real(time, "time").reshape(())
        scale_ = _real(scale_factor, "scale_factor").reshape(())
        token = _integer(frame_token, "frame_token", dtype=jnp.int32).reshape(())

        self.particle_ids = particle_ids_
        self.weights = weights_
        self.positions = positions_
        self.local_momenta = local_
        self.covariant_momenta = covariant_
        self.species_ids = species_
        self.active_mask = active_
        self.incarnations = incarnations_
        self.lineage_ids = lineage_
        self.time = time_
        self.scale_factor = scale_
        self.frame_token = token
        self.frame_id = _identity(frame_id, "frame_id")
        self.topology_id = _identity(topology_id, "topology_id")
        self.frame_lineage_id = _identity(frame_lineage_id, "frame_lineage_id")

    @property
    def capacity(self) -> int:
        return self.particle_ids.size

    @property
    def finite(self) -> Array:
        active = self.active_mask
        return (
            jnp.all(jnp.where(active, jnp.isfinite(self.weights), True))
            & jnp.all(jnp.where(active[:, None], jnp.isfinite(self.positions), True))
            & jnp.all(jnp.where(active[:, None], jnp.isfinite(self.local_momenta), True))
            & jnp.all(
                jnp.where(active[:, None], jnp.isfinite(self.covariant_momenta), True)
            )
            & jnp.isfinite(self.time)
            & jnp.isfinite(self.scale_factor)
        )

    @property
    def physically_valid(self) -> Array:
        active = self.active_mask
        return (
            self.finite
            & jnp.all(jnp.where(active, self.weights > 0.0, self.weights >= 0.0))
            & jnp.all(self.incarnations >= 0)
            & (self.scale_factor > 0.0)
        )

    def replace_dynamics(
        self,
        positions: ArrayLike,
        local_momenta: ArrayLike,
        covariant_momenta: ArrayLike,
        time: ArrayLike,
        scale_factor: ArrayLike,
        frame_token: ArrayLike,
        /,
    ) -> "RelativisticParticleState":
        """Replace only continuous stage data while preserving stable identity."""
        return RelativisticParticleState(
            self.particle_ids,
            self.weights,
            positions,
            local_momenta,
            covariant_momenta,
            self.species_ids,
            self.active_mask,
            self.incarnations,
            self.lineage_ids,
            time,
            scale_factor,
            frame_token,
            frame_id=self.frame_id,
            topology_id=self.topology_id,
            frame_lineage_id=self.frame_lineage_id,
        )


class RelativisticStressSourceIntegrals(StrictModule):
    """Proper-volume integrals of the Eulerian stress-energy projections."""

    energy: Array
    momentum_covector: Array
    stress_covariant: Array
    trace_stress: Array
    anisotropic_stress: Array


class RelativisticStressGatherResult(StrictModule):
    values: Array
    support: Array
    support_complete: Array
    finite: Array
    successful: Array
    source_id: str = eqx.field(static=True)


class RelativisticStressAdjointEvidence(StrictModule):
    particle_pairing: Array
    grid_pairing: Array
    absolute_defect: Array
    relative_defect: Array
    support_complete: Array
    finite: Array
    successful: Array
    source_id: str = eqx.field(static=True)


class RelativisticStressDepositResult(StrictModule):
    """One matched, proper-measure stress-energy source evaluation."""

    projection: StressEnergyProjection
    routes: ParticleGridSplatState
    particle_energy: Array
    local_velocity: Array
    source_integrals: RelativisticStressSourceIntegrals
    anisotropic_stress: Array
    mass_shell_defect: Array
    frame_momentum_defect: Array
    support_complete: Array
    route_valid: Array
    mass_shell_valid: Array
    frame_consistent: Array
    source_conserved: Array
    finite: Array
    successful: Array
    energy_density_unit_id: str = eqx.field(static=True)
    momentum_density_unit_id: str = eqx.field(static=True)
    stress_unit_id: str = eqx.field(static=True)
    volume_measure_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


class RelativisticStressDepositPlan(StrictModule, NonTrainableState):
    """Compile one conservative particle/grid route for relativistic ``T_mu_nu``.

    The plan is snapshot-independent. Every operation receives the exact
    :class:`LocalRelativisticFramePlan` for its stage, so no compiled call can
    silently reuse stale geometry.
    """

    transfer: PreparedParticleGridSplat
    units: RelativisticUnitContract
    species_ids: Array
    rest_masses: Array
    target_coordinates: Array
    topology_id: str = eqx.field(static=True)
    transfer_topology_id: str = eqx.field(static=True)
    mass_shell_relative_tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    frame_momentum_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedParticleGridSplat,
        units: RelativisticUnitContract,
        species_ids: ArrayLike,
        rest_masses: ArrayLike,
        /,
        *,
        topology_id: str | None = None,
        mass_shell_relative_tolerance: float = 1.0e-8,
        conservation_tolerance: float = 1.0e-8,
        frame_momentum_relative_tolerance: float = 1.0e-8,
    ):
        if not isinstance(transfer, PreparedParticleGridSplat):
            raise TypeError("transfer must be a PreparedParticleGridSplat.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be a RelativisticUnitContract.")
        if not transfer.materialized_target:
            raise ValueError(
                "Relativistic stress transfer requires a materialized tensor grid."
            )
        if transfer.particles.ambient_dimension != 3:
            raise ValueError("Relativistic stress transfer requires three dimensions.")
        species_host = np.asarray(species_ids)
        masses_host = np.asarray(rest_masses, dtype=np.float64)
        if (
            species_host.ndim != 1
            or species_host.size == 0
            or not np.issubdtype(species_host.dtype, np.integer)
            or masses_host.shape != species_host.shape
            or len(set(int(value) for value in species_host)) != species_host.size
            or np.any(~np.isfinite(masses_host))
            or np.any(masses_host < 0.0)
        ):
            raise ValueError(
                "species_ids/rest_masses must define unique species and finite nonnegative masses."
            )
        tolerances = (
            float(mass_shell_relative_tolerance),
            float(conservation_tolerance),
            float(frame_momentum_relative_tolerance),
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "Relativistic stress tolerances must be finite and nonnegative."
            )
        target = transfer.plan.target
        transfer_topology = target.topology.topology_id
        declared_topology = (
            transfer_topology
            if topology_id is None
            else _identity(topology_id, "topology_id")
        )
        measure_id = transfer.target_measure.measure_id
        target_coordinates = jnp.stack(
            jnp.meshgrid(
                *transfer.layout.coordinates_by_axis,
                indexing="ij",
            ),
            axis=-1,
        )
        self.transfer = transfer
        self.units = units
        self.species_ids = jnp.asarray(species_host, dtype=jnp.int32)
        self.rest_masses = jnp.asarray(masses_host)
        self.target_coordinates = target_coordinates
        self.topology_id = declared_topology
        self.transfer_topology_id = transfer_topology
        self.mass_shell_relative_tolerance = tolerances[0]
        self.conservation_tolerance = tolerances[1]
        self.frame_momentum_relative_tolerance = tolerances[2]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-particle-stress-transfer",
                "transfer": transfer.prepared_id,
                "topology": {
                    "declared_geometry": declared_topology,
                    "transfer": transfer_topology,
                    "transfer_prepared_id": transfer.prepared_id,
                },
                "measure": measure_id,
                "units": units.contract_id,
                "species": array_tree_fingerprint(species_host.astype(np.int64)),
                "rest_masses": array_tree_fingerprint(masses_host),
                "mass_shell_relative_tolerance": tolerances[0],
                "conservation_tolerance": tolerances[1],
                "frame_momentum_relative_tolerance": tolerances[2],
            }
        )

    @property
    def volume_measure_id(self) -> str:
        return self.transfer.target_measure.measure_id

    def frame_token(self, frame: LocalRelativisticFramePlan, /) -> Array:
        """Return the authoritative dynamic ADM stage token."""
        self._require_frame(frame)
        return _frame_token(frame)

    def _require_frame(self, frame: LocalRelativisticFramePlan, /) -> None:
        if not isinstance(frame, LocalRelativisticFramePlan):
            raise TypeError("frame must be a LocalRelativisticFramePlan.")
        if frame.units.contract_id != self.units.contract_id:
            raise ValueError("Frame and stress transfer use different unit contracts.")
        if frame.geometry.leading_shape != self.transfer.target_shape:
            raise ValueError("Frame geometry and transfer target shapes differ.")
        if frame.geometry.topology_id != self.topology_id:
            raise ValueError("Frame geometry and transfer topologies differ.")
        if not frame.frame_id:
            raise ValueError("Frame identity must be non-empty.")

    def _require_state(
        self, state: RelativisticParticleState, frame: LocalRelativisticFramePlan, /
    ) -> None:
        if not isinstance(state, RelativisticParticleState):
            raise TypeError("state must be a RelativisticParticleState.")
        self._require_frame(frame)
        particles = self.transfer.particles
        if state.capacity != particles.capacity:
            raise ValueError("Particle state and transfer capacities differ.")
        if state.topology_id != self.topology_id:
            raise ValueError("Particle state and transfer topologies differ.")
        if state.frame_lineage_id != frame.geometry.geometry_lineage_id:
            raise ValueError("Particle state and frame geometry lineages differ.")
        if state.frame_id != frame.frame_id:
            raise ValueError("Particle state and local-frame families differ.")

    def _species_rest_mass(
        self, state: RelativisticParticleState, /
    ) -> tuple[Array, Array]:
        matches = state.species_ids[:, None] == self.species_ids[None, :]
        count = jnp.sum(matches, axis=-1)
        masses = jnp.sum(
            jnp.where(matches, self.rest_masses[None, :], 0.0), axis=-1
        ).astype(state.positions.dtype)
        return masses, count == 1

    def _frame_time(
        self, frame: LocalRelativisticFramePlan, /
    ) -> tuple[Array, Array, Array]:
        active = frame.geometry.active
        first = jnp.argmax(active.reshape((-1,))).astype(jnp.int32)
        time = frame.time.reshape((-1,))[first]
        scale = frame.scale_factor.reshape((-1,))[first]
        uniform = jnp.all(jnp.where(active, frame.time == time, True)) & jnp.all(
            jnp.where(active, frame.scale_factor == scale, True)
        )
        return time, scale, uniform

    def _frame_coordinates_valid(self, frame: LocalRelativisticFramePlan, /) -> Array:
        target = self.target_coordinates
        observed = frame.observer_coordinates[..., 1:]
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(target), jnp.abs(observed)),
            1.0,
        )
        tolerance = 128.0 * jnp.finfo(observed.dtype).eps * scale
        return jnp.all(
            jnp.where(
                frame.geometry.active[..., None],
                jnp.abs(observed - target) <= tolerance,
                True,
            )
        )

    def _routes(
        self, state: RelativisticParticleState, positions: ArrayLike | None = None, /
    ) -> ParticleGridSplatState:
        values = state.positions if positions is None else positions
        return self.transfer.build(values, active_mask=state.active_mask)

    def local_from_covariant(
        self,
        state: RelativisticParticleState,
        frame: LocalRelativisticFramePlan,
        covariant_momenta: ArrayLike,
        /,
        *,
        positions: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Map slice-covariant momenta into the exact gathered spatial tetrad."""
        self._require_state(state, frame)
        covariant = _real(covariant_momenta, "covariant_momenta")
        if covariant.shape != (state.capacity, 3):
            raise ValueError("covariant_momenta must have shape (capacity, 3).")
        routes = self._routes(state, positions)
        spatial_basis = frame.tetrad.spatial_vectors[..., 1:]
        gathered = self.transfer.gather(routes, spatial_basis)
        local = ein.contract("pai,pi->pa", gathered.values, covariant, backend="jax")
        local = jnp.where(state.active_mask[:, None], local, 0.0)
        support = jnp.all(gathered.support | ~state.active_mask)
        tangent = jnp.max(
            jnp.where(
                frame.geometry.active[..., None],
                jnp.abs(frame.tetrad.spatial_vectors[..., 0]),
                0.0,
            ),
            initial=0.0,
        )
        tolerance = 128.0 * jnp.finfo(local.dtype).eps
        valid = (
            support
            & (tangent <= tolerance)
            & self._frame_coordinates_valid(frame)
            & jnp.all(jnp.isfinite(local))
        )
        return local, valid

    def initialize(
        self,
        weights: ArrayLike,
        positions: ArrayLike,
        covariant_momenta: ArrayLike,
        species_ids: ArrayLike,
        frame: LocalRelativisticFramePlan,
        /,
        *,
        active_mask: ArrayLike | None = None,
        incarnations: ArrayLike | None = None,
        lineage_ids: ArrayLike | None = None,
    ) -> RelativisticParticleState:
        self._require_frame(frame)
        particles = self.transfer.particles
        capacity = particles.capacity
        active = (
            particles.active_mask
            if active_mask is None
            else particles.active_mask & jnp.asarray(active_mask, dtype=jnp.bool_)
        )
        incarnation = (
            jnp.zeros((capacity,), dtype=jnp.int32)
            if incarnations is None
            else incarnations
        )
        lineage = particles.particle_ids if lineage_ids is None else lineage_ids
        time, scale, uniform = self._frame_time(frame)
        seed = RelativisticParticleState(
            particles.particle_ids,
            weights,
            positions,
            jnp.zeros((capacity, 3), dtype=jnp.asarray(positions).dtype),
            covariant_momenta,
            species_ids,
            active,
            incarnation,
            lineage,
            time,
            scale,
            _frame_token(frame),
            frame_id=frame.frame_id,
            topology_id=self.topology_id,
            frame_lineage_id=frame.geometry.geometry_lineage_id,
        )
        local, conversion_valid = self.local_from_covariant(
            seed, frame, covariant_momenta, positions=positions
        )
        local = eqx.error_if(
            local,
            ~(uniform & self._frame_coordinates_valid(frame) & conversion_valid),
            "Relativistic particle initialization requires one valid co-temporal frame.",
        )
        return seed.replace_dynamics(
            positions,
            local,
            covariant_momenta,
            time,
            scale,
            _frame_token(frame),
        )

    def replace_covariant_dynamics(
        self,
        state: RelativisticParticleState,
        frame: LocalRelativisticFramePlan,
        positions: ArrayLike,
        covariant_momenta: ArrayLike,
        time: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> RelativisticParticleState:
        local, valid = self.local_from_covariant(
            state,
            frame,
            covariant_momenta,
            positions=positions,
        )
        local = eqx.error_if(
            local,
            ~valid,
            "Covariant/local momentum conversion lacks complete frame support.",
        )
        return state.replace_dynamics(
            positions,
            local,
            covariant_momenta,
            time,
            scale_factor,
            _frame_token(frame),
        )

    def gather(
        self,
        state: RelativisticParticleState,
        frame: LocalRelativisticFramePlan,
        grid_values: ArrayLike,
        /,
    ) -> RelativisticStressGatherResult:
        self._require_state(state, frame)
        routes = self._routes(state)
        gathered = self.transfer.gather(routes, grid_values)
        frame_match = state.frame_token == _frame_token(frame)
        support = gathered.support | ~state.active_mask
        finite = jnp.all(
            jnp.where(
                state.active_mask.reshape(
                    state.active_mask.shape
                    + (1,) * (gathered.values.ndim - state.active_mask.ndim)
                ),
                jnp.isfinite(gathered.values),
                True,
            )
        )
        support_complete = jnp.all(support)
        successful = (
            routes.successful
            & frame_match
            & self._frame_coordinates_valid(frame)
            & support_complete
            & finite
        )
        values = jnp.where(
            state.active_mask.reshape(
                state.active_mask.shape
                + (1,) * (gathered.values.ndim - state.active_mask.ndim)
            ),
            gathered.values,
            0.0,
        )
        return RelativisticStressGatherResult(
            values,
            gathered.support,
            support_complete,
            finite,
            successful,
            self.plan_id,
        )

    def adjoint_evidence(
        self,
        state: RelativisticParticleState,
        frame: LocalRelativisticFramePlan,
        particle_values: ArrayLike,
        grid_values: ArrayLike,
        /,
    ) -> RelativisticStressAdjointEvidence:
        self._require_state(state, frame)
        particle = jnp.asarray(particle_values)
        if particle.ndim < 1 or particle.shape[0] != state.capacity:
            raise ValueError("particle_values must begin with particle capacity.")
        grid = jnp.asarray(grid_values, dtype=particle.dtype)
        expected = self.transfer.target_shape + particle.shape[1:]
        if grid.shape != expected:
            raise ValueError(f"grid_values must have shape {expected}.")
        routes = self._routes(state)
        deposited = self.transfer.deposit_content(routes, particle)
        gathered = self.transfer.gather(routes, grid)
        axes = tuple(range(particle.ndim))
        particle_pairing = jnp.sum(particle * gathered.values, axis=axes)
        grid_pairing = jnp.sum(deposited.content * grid)
        absolute = jnp.abs(particle_pairing - grid_pairing)
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(particle_pairing), jnp.abs(grid_pairing)), 1.0
        )
        relative = absolute / scale
        finite = (
            jnp.isfinite(particle_pairing)
            & jnp.isfinite(grid_pairing)
            & jnp.isfinite(relative)
        )
        support_complete = jnp.all(gathered.support | ~state.active_mask)
        frame_match = state.frame_token == _frame_token(frame)
        successful = (
            deposited.successful
            & routes.successful
            & frame_match
            & self._frame_coordinates_valid(frame)
            & support_complete
            & finite
        )
        return RelativisticStressAdjointEvidence(
            particle_pairing,
            grid_pairing,
            absolute,
            relative,
            support_complete,
            finite,
            successful,
            self.plan_id,
        )

    def deposit(
        self,
        state: RelativisticParticleState,
        frame: LocalRelativisticFramePlan,
        /,
    ) -> RelativisticStressDepositResult:
        self._require_state(state, frame)
        particles = self.transfer.particles
        ids_match = jnp.all(state.particle_ids == particles.particle_ids)
        active_compatible = jnp.all(~state.active_mask | particles.active_mask)
        frame_match = state.frame_token == _frame_token(frame)
        time, scale_factor, frame_uniform = self._frame_time(frame)
        time_match = (state.time == time) & (state.scale_factor == scale_factor)
        rest_mass, species_known = self._species_rest_mass(state)
        active_species = jnp.all(species_known | ~state.active_mask)
        light_speed = jnp.asarray(
            float(self.units.speed_of_light), dtype=state.positions.dtype
        )
        rest_energy = self.units.mass_to_rest_energy(rest_mass)
        momentum_squared = ein.contract(
            "pi,pi->p", state.local_momenta, state.local_momenta, backend="jax"
        )
        energy = jnp.sqrt(rest_energy**2 + light_speed**2 * momentum_squared)
        positive_energy = energy > 0.0
        safe_energy = jnp.where(state.active_mask & positive_energy, energy, 1.0)
        local_velocity = light_speed**2 * state.local_momenta / safe_energy[:, None]
        local_velocity = jnp.where(state.active_mask[:, None], local_velocity, 0.0)
        four_momentum = self.units.assemble_four_momentum(energy, state.local_momenta)
        shell_defect = self.units.mass_shell_residual(four_momentum, rest_mass)
        shell_valid = self.units.mass_shell_admissible(
            four_momentum,
            rest_mass,
            relative_tolerance=self.mass_shell_relative_tolerance,
        )
        mass_shell_valid = jnp.all((shell_valid & positive_energy) | ~state.active_mask)
        derived_local, local_map_valid = self.local_from_covariant(
            state, frame, state.covariant_momenta
        )
        momentum_difference = jnp.sqrt(
            ein.contract(
                "pi,pi->p",
                state.local_momenta - derived_local,
                state.local_momenta - derived_local,
                backend="jax",
            )
        )
        momentum_scale = jnp.maximum(
            jnp.sqrt(
                ein.contract(
                    "pi,pi->p",
                    state.local_momenta,
                    state.local_momenta,
                    backend="jax",
                )
            ),
            jnp.sqrt(
                ein.contract("pi,pi->p", derived_local, derived_local, backend="jax")
            ),
        )
        frame_momentum_defect = jnp.where(
            state.active_mask,
            momentum_difference
            / jnp.maximum(momentum_scale, jnp.finfo(energy.dtype).tiny),
            0.0,
        )
        frame_consistent = local_map_valid & jnp.all(
            (frame_momentum_defect <= self.frame_momentum_relative_tolerance)
            | ~state.active_mask
        )

        routes = self._routes(state)
        geometry_coverage = self.transfer.gather(
            routes, frame.admissible.astype(state.positions.dtype)
        )
        support_complete = jnp.all(geometry_coverage.support | ~state.active_mask)
        route_valid = jnp.all(
            jnp.where(
                state.active_mask,
                geometry_coverage.values >= 1.0 - 128.0 * jnp.finfo(energy.dtype).eps,
                True,
            )
        )
        weight = jnp.where(state.active_mask, state.weights, 0.0)
        energy_content = weight * energy
        momentum_content = weight[:, None] * light_speed * state.covariant_momenta
        stress_content = (
            weight[:, None, None]
            * light_speed**2
            * ein.contract(
                "pi,pj->pij",
                state.covariant_momenta,
                state.covariant_momenta,
                backend="jax",
            )
            / safe_energy[:, None, None]
        )
        shell_content = weight * jnp.abs(shell_defect)
        energy_deposit = self.transfer.deposit_content(routes, energy_content)
        momentum_deposit = self.transfer.deposit_content(routes, momentum_content)
        stress_deposit = self.transfer.deposit_content(routes, stress_content)
        shell_deposit = self.transfer.deposit_content(routes, shell_content)

        sqrt_det = frame.geometry.sqrt_det_spatial_metric
        energy_density = energy_deposit.density / sqrt_det
        momentum_density = momentum_deposit.density / sqrt_det[..., None]
        stress_density = stress_deposit.density / sqrt_det[..., None, None]
        shell_density = shell_deposit.density / sqrt_det
        stress_density = 0.5 * (stress_density + jnp.swapaxes(stress_density, -1, -2))
        trace = ein.contract(
            "...ij,...ij->...",
            frame.geometry.inverse_spatial_metric,
            stress_density,
            backend="jax",
        )
        anisotropic = (
            stress_density - trace[..., None, None] * frame.geometry.spatial_metric / 3.0
        )
        balance = jnp.maximum(
            jnp.maximum(
                energy_deposit.balance.maximum_absolute_balance_defect,
                momentum_deposit.balance.maximum_absolute_balance_defect,
            ),
            stress_deposit.balance.maximum_absolute_balance_defect,
        )
        source_scale = jnp.maximum(
            jnp.maximum(jnp.max(jnp.abs(energy_content), initial=0.0), 1.0),
            jnp.max(jnp.abs(stress_content), initial=0.0),
        )
        source_conserved = balance <= self.conservation_tolerance * source_scale
        finite = (
            state.finite
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(local_velocity))
            & jnp.all(jnp.isfinite(energy_density))
            & jnp.all(jnp.isfinite(momentum_density))
            & jnp.all(jnp.isfinite(stress_density))
            & jnp.all(jnp.isfinite(shell_density))
        )
        successful = (
            state.physically_valid
            & ids_match
            & active_compatible
            & frame_match
            & frame_uniform
            & self._frame_coordinates_valid(frame)
            & time_match
            & active_species
            & mass_shell_valid
            & frame_consistent
            & routes.successful
            & support_complete
            & route_valid
            & energy_deposit.successful
            & momentum_deposit.successful
            & stress_deposit.successful
            & shell_deposit.successful
            & source_conserved
            & finite
        )
        conservation_field = jnp.broadcast_to(balance, self.transfer.target_shape)
        valid = frame.geometry.physically_valid & jnp.isfinite(energy_density)
        projection_id = canonical_fingerprint(
            {
                "kind": "relativistic-stress-projection",
                "source": self.plan_id,
                "frame": frame.frame_id,
            }
        )
        projection = StressEnergyProjection(
            energy_density,
            momentum_density,
            stress_density,
            frame.geometry.active,
            valid,
            shell_density,
            conservation_field,
            snapshot_token=frame.geometry.snapshot_token,
            geometry_lineage_id=frame.geometry.geometry_lineage_id,
            convention_id=frame.geometry.convention_id,
            scale_id=frame.geometry.scale_id,
            topology_id=frame.geometry.topology_id,
            projection_id=projection_id,
        )
        coordinate_measure = self.transfer.target_measure.weights.reshape(
            self.transfer.target_shape
        ).astype(energy.dtype)
        proper_measure = coordinate_measure * sqrt_det
        energy_integral = ein.contract(
            "...,...->", energy_density, proper_measure, backend="jax"
        )
        momentum_integral = ein.contract(
            "...i,...->i", momentum_density, proper_measure, backend="jax"
        )
        stress_integral = ein.contract(
            "...ij,...->ij", stress_density, proper_measure, backend="jax"
        )
        anisotropic_integral = ein.contract(
            "...ij,...->ij", anisotropic, proper_measure, backend="jax"
        )
        trace_integral = ein.contract("...,...->", trace, proper_measure, backend="jax")
        integrals = RelativisticStressSourceIntegrals(
            energy_integral,
            momentum_integral,
            stress_integral,
            trace_integral,
            anisotropic_integral,
        )
        energy_density_unit = canonical_fingerprint(
            {
                "numerator": self.units.energy_unit.unit_id,
                "volume_measure": self.volume_measure_id,
                "proper_volume": True,
            }
        )
        return RelativisticStressDepositResult(
            projection,
            routes,
            energy,
            local_velocity,
            integrals,
            anisotropic,
            shell_defect,
            frame_momentum_defect,
            support_complete,
            route_valid,
            mass_shell_valid,
            frame_consistent,
            source_conserved,
            finite,
            successful,
            energy_density_unit,
            energy_density_unit,
            energy_density_unit,
            self.volume_measure_id,
            self.plan_id,
        )


__all__ = [
    "RelativisticParticleState",
    "RelativisticStressAdjointEvidence",
    "RelativisticStressDepositPlan",
    "RelativisticStressDepositResult",
    "RelativisticStressGatherResult",
    "RelativisticStressSourceIntegrals",
]
