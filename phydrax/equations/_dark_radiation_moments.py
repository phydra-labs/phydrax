#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Relativistic dark-radiation moment, hierarchy, and VET closures."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
)
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ._radiation_moments import MultigroupM1RadiationSystem


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string.")
    return value


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _content_id_words(value: str, name: str, /) -> Array:
    identity = _identifier(value, name)
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return jnp.asarray(
        [int(identity[index : index + 8], 16) for index in range(0, 64, 8)],
        dtype=jnp.uint32,
    )


def _content_id_from_words(value: ArrayLike, name: str, /) -> str:
    words = np.asarray(value, dtype=np.uint32)
    if words.shape != (8,):
        raise ValueError(f"{name} words must have shape (8,).")
    return "".join(f"{int(word):08x}" for word in words)


def _vector_norm(value: Array, /) -> Array:
    scale = jnp.max(jnp.abs(value), axis=-1)
    safe = jnp.where(scale > 0.0, scale, 1.0)
    norm = safe * jnp.sqrt(jnp.sum((value / safe[..., None]) ** 2, axis=-1))
    return jnp.where(scale > 0.0, norm, 0.0)


def _m1_tensor(
    energy: Array,
    flux: Array,
    physical_light_speed: float,
    energy_floor: float,
    /,
) -> tuple[Array, Array, Array]:
    """Return Euclidean M1 pressure, raw flux factor, and closure factor."""

    norm = _vector_norm(flux)
    safe_energy = jnp.maximum(energy, jnp.asarray(energy_floor, dtype=energy.dtype))
    speed = jnp.asarray(physical_light_speed, dtype=energy.dtype)
    raw = norm / (speed * safe_energy)
    reduced = jnp.minimum(raw, 1.0)
    chi = (3.0 + 4.0 * reduced**2) / (
        5.0 + 2.0 * jnp.sqrt(jnp.maximum(4.0 - 3.0 * reduced**2, 0.0))
    )
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    direction = flux / safe_norm[..., None]
    identity = jnp.eye(flux.shape[-1], dtype=energy.dtype)
    tensor = (
        0.5 * (1.0 - chi)[..., None, None] * identity
        + 0.5
        * (3.0 * chi - 1.0)[..., None, None]
        * direction[..., :, None]
        * direction[..., None, :]
    )
    return energy[..., None, None] * tensor, raw, chi


class DarkRadiationFourForce(StrictModule, NonTrainableState):
    """Exactly paired radiation and matter four-force in one local frame.

    Components are ``(energy rate, momentum rate...)``.  Transport may use a
    reduced signal speed, but the momentum component is always formed with the
    declared physical light speed.
    """

    radiation_four_force: Array
    matter_four_force: Array
    balance_residual: Array
    exact_opposite: Array
    endpoint_time: Array
    frame_token: Array
    source_state_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    exchange_id: str = eqx.field(static=True)

    def __init__(
        self,
        radiation_four_force: ArrayLike,
        matter_four_force: ArrayLike,
        endpoint_time: ArrayLike,
        /,
        *,
        frame_token: ArrayLike,
        source_state_id: str,
        frame_id: str,
        unit_contract_id: str,
        frame_realization_id: str,
    ):
        radiation = jnp.asarray(radiation_four_force)
        matter = jnp.asarray(matter_four_force, dtype=radiation.dtype)
        time = jnp.asarray(endpoint_time, dtype=radiation.dtype)
        token = jnp.asarray(frame_token)
        if radiation.ndim < 1 or radiation.shape[-1] != 4:
            raise ValueError("Dark-radiation four-force must end in four components.")
        if matter.shape != radiation.shape:
            raise ValueError("Radiation and matter four-force shapes must match.")
        if time.shape != () or token.shape != ():
            raise ValueError(
                "Dark-radiation four-force endpoint_time and frame_token must be scalar."
            )
        source = _identifier(source_state_id, "source_state_id")
        frame = _identifier(frame_id, "frame_id")
        units = _identifier(unit_contract_id, "unit_contract_id")
        realization = _identifier(frame_realization_id, "frame_realization_id")
        residual = radiation + matter
        exact = jnp.all(residual == 0.0, axis=-1)
        self.radiation_four_force = radiation
        self.matter_four_force = matter
        self.balance_residual = residual
        self.exact_opposite = exact
        self.endpoint_time = time
        self.frame_token = token
        self.source_state_id = source
        self.frame_id = frame
        self.frame_realization_id = realization
        self.unit_contract_id = units
        self.exchange_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-four-force",
                "source_state": source,
                "frame": frame,
                "frame_realization": realization,
                "units": units,
            }
        )

    @classmethod
    def paired(
        cls,
        radiation_four_force: ArrayLike,
        endpoint_time: ArrayLike,
        frame_token: ArrayLike,
        /,
        *,
        source_state_id: str,
        frame_id: str,
        unit_contract_id: str,
        frame_realization_id: str,
    ) -> DarkRadiationFourForce:
        radiation = jnp.asarray(radiation_four_force)
        return cls(
            radiation,
            -radiation,
            endpoint_time,
            frame_token=frame_token,
            source_state_id=source_state_id,
            frame_id=frame_id,
            frame_realization_id=frame_realization_id,
            unit_contract_id=unit_contract_id,
        )


class DarkRadiationM1RealizabilityEvidence(StrictModule, NonTrainableState):
    energy_positive_before: Array
    realizable_before: Array
    finite_before: Array
    correction_norm: Array
    correction_applied: Array
    energy_positive_after: Array
    realizable_after: Array
    accepted: Array


class DarkRadiationM1RealizabilityResult(StrictModule, NonTrainableState):
    source_state: Array
    candidate_state: Array
    accepted_state: Array
    evidence: DarkRadiationM1RealizabilityEvidence


class DarkRadiationM1Qualification(StrictModule, NonTrainableState):
    opposing_beam_fraction: Array
    beam_risk: Array
    accepted: Array
    refusal_reason: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)


class DarkRadiationGroupRedshiftResult(StrictModule, NonTrainableState):
    interface_flux: Array
    conservative_rate: Array
    boundary_loss_rate: Array
    conservation_residual: Array
    accepted: Array
    system_id: str = eqx.field(static=True)


class DarkRadiationM1ExchangeResult(StrictModule, NonTrainableState):
    source_state: Array
    accepted_state: Array
    exchange: DarkRadiationFourForce
    minimum_energy: Array
    realizable: Array
    accepted: Array
    system_id: str = eqx.field(static=True)


class DarkRadiationM1RefluxResult(StrictModule, NonTrainableState):
    source_state: Array
    candidate_state: Array
    accepted_state: Array
    coarse_change: Array
    face_flux_change: Array
    conservation_residual: Array
    realizable: Array
    rolled_back: Array
    accepted: Array
    topology_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)


class CosmologicalMultigroupM1System(StrictModule, NonTrainableState):
    """Physical-c multigroup M1 with reduced-c transport and FLRW redshift.

    The stored state remains the existing flattened ``(E_g, F_g^i)`` layout.
    Realizability uses physical ``c``. Hyperbolic signal bounds use reduced
    ``c``. The distinction is immutable and included in ``system_id``.
    """

    transport_system: MultigroupM1RadiationSystem
    group_edges: Array
    physical_light_speed: float = eqx.field(static=True)
    beam_risk_limit: float = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        group_edges: ArrayLike,
        dimension: int = 3,
        /,
        *,
        physical_light_speed: float = 1.0,
        reduced_light_speed: float | None = None,
        energy_floor: float = 1.0e-12,
        beam_risk_limit: float = 0.25,
    ):
        edges = jnp.asarray(group_edges)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Dark-radiation group_edges must be one dimensional.")
        if not jnp.issubdtype(edges.dtype, jnp.floating):
            raise TypeError("Dark-radiation group_edges must have floating dtype.")
        if not bool(np.all(np.isfinite(np.asarray(edges)))) or not bool(
            np.all(np.diff(np.asarray(edges)) > 0.0)
        ):
            raise ValueError("Dark-radiation group_edges must be finite and increasing.")
        physical = _positive(physical_light_speed, "physical_light_speed")
        reduced = (
            physical
            if reduced_light_speed is None
            else _positive(reduced_light_speed, "reduced_light_speed")
        )
        if reduced > physical:
            raise ValueError("Reduced light speed cannot exceed physical light speed.")
        beam_limit = _nonnegative(beam_risk_limit, "beam_risk_limit")
        if beam_limit > 1.0:
            raise ValueError("beam_risk_limit cannot exceed one.")
        transport = MultigroupM1RadiationSystem(
            edges.size - 1,
            int(dimension),
            reduced_light_speed=reduced,
            energy_floor=energy_floor,
        )
        self.transport_system = transport
        self.group_edges = edges
        self.physical_light_speed = physical
        self.beam_risk_limit = beam_limit
        self.component_names = transport.component_names
        self.system_id = canonical_fingerprint(
            {
                "kind": "cosmological-dark-radiation-multigroup-m1",
                "group_edges": np.asarray(edges).tolist(),
                "dimension": self.dimension,
                "physical_light_speed": physical,
                "reduced_light_speed": reduced,
                "energy_floor": self.energy_floor,
                "beam_risk_limit": beam_limit,
            }
        )

    @property
    def group_count(self) -> int:
        return self.transport_system.group_count

    @property
    def dimension(self) -> int:
        return self.transport_system.dimension

    @property
    def reduced_light_speed(self) -> float:
        return self.transport_system.reduced_light_speed

    @property
    def energy_floor(self) -> float:
        return self.transport_system.energy_floor

    @property
    def group_width(self) -> int:
        return self.transport_system.group_width

    def _groups(self, state: Array, /) -> Array:
        return self.transport_system._groups(state)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self.transport_system.conserved_to_primitive(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self.transport_system.primitive_to_conserved(primitive)

    def physical_flux(self, state: Array, axis: int, args=None, /) -> Array:
        del args
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("M1 flux axis is outside the configured dimension.")
        groups = self._groups(state)
        energy = groups[..., 0]
        flux = groups[..., 1:]
        tensor = self._eddington_tensor(groups)
        output = jnp.zeros_like(groups)
        output = output.at[..., 0].set(flux[..., axis_])
        output = output.at[..., 1:].set(
            self.reduced_light_speed**2 * energy[..., None] * tensor[..., :, axis_]
        )
        return output.reshape(jnp.asarray(state).shape)

    def max_wave_speed(self, left: Array, right: Array, axis: int, args=None, /) -> Array:
        return self.transport_system.max_wave_speed(left, right, axis, args)

    def signal_bounds(
        self, left: Array, right: Array, axis: int, args=None, /
    ) -> tuple[Array, Array]:
        return self.transport_system.signal_bounds(left, right, axis, args)

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        unit_normal: Array,
        args=None,
        /,
    ) -> tuple[Array, Array]:
        return self.transport_system.normal_signal_bounds(left, right, unit_normal, args)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.transport_system.reflect_state(state, axis)

    @property
    def physical_c_reduced_c_identity(self) -> str:
        return canonical_fingerprint(
            {
                "physical_light_speed": self.physical_light_speed,
                "reduced_light_speed": self.reduced_light_speed,
                "role": "physical-state-versus-transport-characteristics",
            }
        )

    def _eddington_tensor(self, group_state: Array, /) -> Array:
        pressure, _, _ = _m1_tensor(
            group_state[..., 0],
            group_state[..., 1:],
            self.physical_light_speed,
            self.energy_floor,
        )
        return pressure / jnp.maximum(
            group_state[..., 0, None, None],
            jnp.asarray(self.energy_floor, dtype=group_state.dtype),
        )

    def admissible(self, state: Array, /) -> Array:
        groups = self._groups(state)
        energy = groups[..., 0]
        norm = _vector_norm(groups[..., 1:])
        return jnp.all(jnp.isfinite(groups), axis=(-2, -1)) & jnp.all(
            (energy >= self.energy_floor) & (norm <= self.physical_light_speed * energy),
            axis=-1,
        )

    def enforce_realizability(
        self, state: ArrayLike, /
    ) -> DarkRadiationM1RealizabilityResult:
        source = jnp.asarray(state)
        groups = self._groups(source)
        energy = groups[..., 0]
        flux = groups[..., 1:]
        finite = jnp.all(jnp.isfinite(groups), axis=(-2, -1))
        positive = jnp.all(energy >= self.energy_floor, axis=-1)
        norm = _vector_norm(flux)
        realizable = jnp.all(
            norm <= self.physical_light_speed * jnp.maximum(energy, 0.0), axis=-1
        )
        repaired_energy = jnp.maximum(energy, self.energy_floor)
        maximum_norm = self.physical_light_speed * repaired_energy
        scale = jnp.minimum(1.0, maximum_norm / jnp.where(norm > 0.0, norm, 1.0))
        repaired_flux = flux * scale[..., None]
        repaired = groups.at[..., 0].set(repaired_energy).at[..., 1:].set(repaired_flux)
        candidate = repaired.reshape(source.shape)
        correction = candidate - source
        correction_norm = jnp.sqrt(jnp.sum(correction**2, axis=-1))
        candidate_admissible = self.admissible(candidate)
        accepted = finite & candidate_admissible
        accepted_state = jnp.where(accepted[..., None], candidate, source)
        evidence = DarkRadiationM1RealizabilityEvidence(
            positive,
            realizable,
            finite,
            correction_norm,
            correction_norm > 0.0,
            jnp.all(repaired_energy >= self.energy_floor, axis=-1),
            candidate_admissible,
            accepted,
        )
        return DarkRadiationM1RealizabilityResult(
            source, candidate, accepted_state, evidence
        )

    def qualify_beam_superposition(
        self, opposing_beam_fraction: ArrayLike, /
    ) -> DarkRadiationM1Qualification:
        fraction = jnp.asarray(opposing_beam_fraction)
        beam_risk = (
            (~jnp.isfinite(fraction))
            | (fraction < 0.0)
            | (fraction > self.beam_risk_limit)
        )
        accepted = ~beam_risk
        return DarkRadiationM1Qualification(
            fraction,
            beam_risk,
            accepted,
            "m1-crossing-beam-risk-when-beam_risk-is-true",
            self.system_id,
        )

    def group_redshift_flux(
        self,
        state: ArrayLike,
        hubble_rate: ArrayLike,
        /,
    ) -> DarkRadiationGroupRedshiftResult:
        """Conservative upwind flux in log-frequency under FLRW redshift.

        Positive expansion advects radiation toward lower group index. The
        returned boundary loss makes the finite represented band conservative:
        ``sum(rate) + boundary_loss == 0`` component by component.
        """

        source = jnp.asarray(state)
        groups = self._groups(source)
        hubble = jnp.asarray(hubble_rate, dtype=source.dtype)
        if hubble.shape != source.shape[:-1]:
            if hubble.shape != ():
                raise ValueError("hubble_rate must be scalar or match state lanes.")
        widths = jnp.log(self.group_edges[1:] / self.group_edges[:-1]).astype(
            source.dtype
        )
        density = groups / widths.reshape((1,) * (groups.ndim - 2) + (-1, 1))
        # Interface orientation is from group g-1 into g. Expansion is a
        # negative log-frequency velocity, hence uses the higher-frequency cell.
        zero = jnp.zeros_like(groups[..., :1, :])
        expansion_interfaces = jnp.concatenate(
            (-hubble[..., None, None] * density, zero), axis=-2
        )
        contraction_interfaces = jnp.concatenate(
            (zero, -hubble[..., None, None] * density), axis=-2
        )
        interfaces = jnp.where(
            (hubble >= 0.0)[..., None, None],
            expansion_interfaces,
            contraction_interfaces,
        )
        rate_groups = -(interfaces[..., 1:, :] - interfaces[..., :-1, :])
        boundary_loss = interfaces[..., -1, :] - interfaces[..., 0, :]
        residual = jnp.sum(rate_groups, axis=-2) + boundary_loss
        accepted = jnp.all(jnp.isfinite(interfaces), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(groups), axis=(-2, -1)
        )
        return DarkRadiationGroupRedshiftResult(
            interfaces,
            rate_groups.reshape(source.shape),
            boundary_loss,
            residual,
            accepted,
            self.system_id,
        )

    def imex_matter_exchange(
        self,
        state: ArrayLike,
        absorption_rate: ArrayLike,
        equilibrium_energy: ArrayLike,
        step_size: ArrayLike,
        endpoint_time: ArrayLike,
        /,
        *,
        frame_token: ArrayLike,
        source_state_id: str,
        frame_id: str,
        unit_contract_id: str,
        frame_realization_id: str,
    ) -> DarkRadiationM1ExchangeResult:
        """Backward-Euler local absorption/emission with exact paired force."""

        source = jnp.asarray(state)
        groups = self._groups(source)
        rate = jnp.asarray(absorption_rate, dtype=source.dtype)
        equilibrium = jnp.asarray(equilibrium_energy, dtype=source.dtype)
        dt = jnp.asarray(step_size, dtype=source.dtype)
        expected = groups.shape[:-1]
        if rate.shape != expected or equilibrium.shape != expected:
            raise ValueError("Absorption rate and equilibrium energy must match groups.")
        if dt.shape != ():
            raise ValueError("Dark-radiation M1 step_size must be scalar.")
        valid_coefficients = (
            jnp.all(jnp.isfinite(rate))
            & jnp.all(rate >= 0.0)
            & jnp.all(jnp.isfinite(equilibrium))
            & jnp.all(equilibrium >= self.energy_floor)
            & jnp.isfinite(dt)
            & (dt > 0.0)
        )
        denominator = 1.0 + dt * rate
        energy = (groups[..., 0] + dt * rate * equilibrium) / denominator
        flux = groups[..., 1:] / denominator[..., None]
        candidate_groups = groups.at[..., 0].set(energy).at[..., 1:].set(flux)
        candidate = candidate_groups.reshape(source.shape)
        realizable = self.admissible(candidate)
        accepted = valid_coefficients & realizable
        accepted_state = jnp.where(accepted[..., None], candidate, source)
        change = self._groups(accepted_state) - groups
        radiation_rate = jnp.concatenate(
            (
                jnp.sum(change[..., 0], axis=-1, keepdims=True) / dt,
                jnp.sum(change[..., 1:], axis=-2) / (dt * self.physical_light_speed**2),
            ),
            axis=-1,
        )
        exchange = DarkRadiationFourForce.paired(
            radiation_rate,
            endpoint_time,
            frame_token,
            source_state_id=source_state_id,
            frame_id=frame_id,
            frame_realization_id=frame_realization_id,
            unit_contract_id=unit_contract_id,
        )
        return DarkRadiationM1ExchangeResult(
            source,
            accepted_state,
            exchange,
            jnp.min(energy, axis=-1),
            realizable,
            accepted,
            self.system_id,
        )

    def reflux(
        self,
        coarse_state: ArrayLike,
        face_flux_correction: ArrayLike,
        cell_volume: ArrayLike,
        /,
        *,
        topology_id: str,
    ) -> DarkRadiationM1RefluxResult:
        source = jnp.asarray(coarse_state)
        correction = jnp.asarray(face_flux_correction, dtype=source.dtype)
        volume = jnp.asarray(cell_volume, dtype=source.dtype)
        if correction.shape != source.shape:
            raise ValueError("M1 face_flux_correction must match coarse_state.")
        if volume.shape not in ((), source.shape[:-1]):
            raise ValueError("M1 cell_volume must be scalar or match state lanes.")
        candidate = source + correction / volume[..., None]
        realizable = self.admissible(candidate)
        finite = jnp.all(jnp.isfinite(candidate), axis=-1) & jnp.all(volume > 0.0)
        accepted = realizable & finite
        accepted_state = jnp.where(accepted[..., None], candidate, source)
        coarse_change = accepted_state - source
        residual = coarse_change * volume[..., None] - jnp.where(
            accepted[..., None], correction, 0.0
        )
        return DarkRadiationM1RefluxResult(
            source,
            candidate,
            accepted_state,
            coarse_change,
            correction,
            residual,
            realizable,
            ~accepted,
            accepted,
            _identifier(topology_id, "topology_id"),
            self.system_id,
        )

    def stress_energy_projection(
        self,
        state: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        source_state_id: str,
    ) -> StressEnergyProjection:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        source = jnp.asarray(state)
        groups = self._groups(source)
        if source.shape[:-1] != geometry.leading_shape:
            raise ValueError("M1 state lanes must match the ADM geometry lanes.")
        energy_groups = groups[..., 0]
        flux_groups = groups[..., 1:]
        inverse_metric = geometry.inverse_spatial_metric.astype(source.dtype)
        metric = geometry.spatial_metric.astype(source.dtype)
        norm_squared = contract(
            "...gi,...ij,...gj->...g",
            flux_groups,
            inverse_metric,
            flux_groups,
            backend="jax",
        )
        norm = jnp.sqrt(jnp.maximum(norm_squared, 0.0))
        safe_energy = jnp.maximum(energy_groups, self.energy_floor)
        raw = norm / (self.physical_light_speed * safe_energy)
        reduced = jnp.minimum(raw, 1.0)
        chi = (3.0 + 4.0 * reduced**2) / (
            5.0 + 2.0 * jnp.sqrt(jnp.maximum(4.0 - 3.0 * reduced**2, 0.0))
        )
        direction = (
            contract("...ij,...gj->...gi", inverse_metric, flux_groups, backend="jax")
            / jnp.where(norm > 0.0, norm, 1.0)[..., None]
        )
        pressure_contravariant = energy_groups[..., None, None] * (
            0.5 * (1.0 - chi)[..., None, None] * inverse_metric[..., None, :, :]
            + 0.5
            * (3.0 * chi - 1.0)[..., None, None]
            * direction[..., :, None]
            * direction[..., None, :]
        )
        pressure_covariant = contract(
            "...ik,...jl,...gkl->...gij",
            metric,
            metric,
            pressure_contravariant,
            backend="jax",
        )
        energy = jnp.sum(energy_groups, axis=-1)
        momentum = jnp.sum(flux_groups, axis=-2) / self.physical_light_speed**2
        stress = jnp.sum(pressure_covariant, axis=-3)
        defect = jnp.max(jnp.abs(stress - jnp.swapaxes(stress, -1, -2)), axis=(-2, -1))
        valid = (
            jnp.all(jnp.isfinite(groups), axis=(-2, -1))
            & jnp.all(energy_groups >= self.energy_floor, axis=-1)
            & jnp.all(raw <= 1.0, axis=-1)
        )
        projection_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-m1-stress-energy",
                "system": self.system_id,
                "source_state": _identifier(source_state_id, "source_state_id"),
                "geometry_lineage": geometry.geometry_lineage_id,
            }
        )
        return StressEnergyProjection(
            energy,
            momentum,
            stress,
            geometry.active,
            valid & geometry.physically_valid,
            defect,
            jnp.zeros_like(energy),
            snapshot_token=geometry.snapshot_token,
            geometry_lineage_id=geometry.geometry_lineage_id,
            convention_id=geometry.convention_id,
            scale_id=geometry.scale_id,
            topology_id=geometry.topology_id,
            projection_id=projection_id,
        )


class DarkRadiationHierarchyState(StrictModule, NonTrainableState):
    """Fixed scale/k/momentum/multipole/polarization hierarchy state."""

    intensity: Array
    polarization_e: Array
    polarization_b: Array
    scale_factor: Array
    conformal_time: Array
    k_active: Array
    momentum_active: Array
    frame_token: Array
    observer_coordinates: Array
    frame_realization_words: Array
    valid: Array
    state_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)

    @property
    def frame_realization_id(self) -> str:
        return _content_id_from_words(
            self.frame_realization_words, "frame_realization_id"
        )


class DarkRadiationHierarchyClosureEvidence(StrictModule, NonTrainableState):
    last_multipole_ratio: Array
    truncation_bound: Array
    closure_qualified: Array
    tight_coupling: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class DarkRadiationLineOfSightOutput(StrictModule, NonTrainableState):
    density_contrast: Array
    velocity_divergence: Array
    anisotropic_stress: Array
    polarization_e: Array
    polarization_b: Array
    source_state_id: str = eqx.field(static=True)
    output_id: str = eqx.field(static=True)


class DarkRadiationBoltzmannHierarchyPlan(StrictModule, NonTrainableState):
    """Fixed-capacity linear dark-radiation Boltzmann hierarchy."""

    initial_frame: LocalRelativisticFramePlan
    wave_numbers: Array
    momentum_nodes: Array
    momentum_weights: Array
    multipole_count: int = eqx.field(static=True)
    polarization_count: int = eqx.field(static=True)
    closure_tolerance: float = eqx.field(static=True)
    self_interaction_rate: float = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    initial_frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wave_numbers: ArrayLike,
        momentum_nodes: ArrayLike,
        momentum_weights: ArrayLike,
        multipole_count: int,
        /,
        *,
        polarization_count: int = 2,
        closure_tolerance: float = 1.0e-3,
        self_interaction_rate: float = 0.0,
        frame: LocalRelativisticFramePlan,
    ):
        wave = jnp.asarray(wave_numbers)
        momentum = jnp.asarray(momentum_nodes, dtype=wave.dtype)
        weights = jnp.asarray(momentum_weights, dtype=wave.dtype)
        multipoles = int(multipole_count)
        polarizations = int(polarization_count)
        if wave.ndim != 1 or momentum.ndim != 1 or weights.shape != momentum.shape:
            raise ValueError("Hierarchy wave and momentum quadratures must be vectors.")
        if multipoles < 3 or polarizations not in (0, 2):
            raise ValueError(
                "Hierarchy needs at least three multipoles and zero or two polarizations."
            )
        if (
            not bool(np.all(np.asarray(wave) >= 0.0))
            or not bool(np.all(np.asarray(momentum) > 0.0))
            or not bool(np.all(np.asarray(weights) > 0.0))
        ):
            raise ValueError("Hierarchy quadrature nodes and weights are invalid.")
        tolerance = _nonnegative(closure_tolerance, "closure_tolerance")
        self_rate = _nonnegative(self_interaction_rate, "self_interaction_rate")
        if not isinstance(frame, LocalRelativisticFramePlan):
            raise TypeError("frame must be LocalRelativisticFramePlan.")
        frame_id = frame.frame_id
        units = frame.units.contract_id
        self.initial_frame = frame
        self.initial_frame_realization_id = frame.realization_id()
        self.wave_numbers = wave
        self.momentum_nodes = momentum
        self.momentum_weights = weights
        self.multipole_count = multipoles
        self.polarization_count = polarizations
        self.closure_tolerance = tolerance
        self.self_interaction_rate = self_rate
        self.frame_id = frame_id
        self.unit_contract_id = units
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-boltzmann-hierarchy",
                "wave_numbers": np.asarray(wave).tolist(),
                "momentum_nodes": np.asarray(momentum).tolist(),
                "momentum_weights": np.asarray(weights).tolist(),
                "multipoles": multipoles,
                "polarizations": polarizations,
                "closure_tolerance": tolerance,
                "self_interaction_rate": self_rate,
                "frame": frame_id,
                "frame_realization": self.initial_frame_realization_id,
                "units": units,
            }
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.wave_numbers.size, self.momentum_nodes.size, self.multipole_count)

    def initialize(
        self,
        intensity: ArrayLike,
        /,
        *,
        polarization_e: ArrayLike | None = None,
        polarization_b: ArrayLike | None = None,
        scale_factor: ArrayLike | None = None,
        conformal_time: ArrayLike | None = None,
        frame: LocalRelativisticFramePlan | None = None,
        frame_realization_id: str | None = None,
        k_active: ArrayLike | None = None,
        momentum_active: ArrayLike | None = None,
        state_id: str,
    ) -> DarkRadiationHierarchyState:
        values = jnp.asarray(intensity)
        if values.shape != self.shape:
            raise ValueError(f"Hierarchy intensity must have shape {self.shape}.")
        if polarization_e is None:
            e_mode = jnp.zeros_like(values)
        else:
            e_mode = jnp.asarray(polarization_e, dtype=values.dtype)
        if polarization_b is None:
            b_mode = jnp.zeros_like(values)
        else:
            b_mode = jnp.asarray(polarization_b, dtype=values.dtype)
        if e_mode.shape != values.shape or b_mode.shape != values.shape:
            raise ValueError("Hierarchy polarization arrays must match intensity.")
        stage_frame = self.initial_frame if frame is None else frame
        if not isinstance(stage_frame, LocalRelativisticFramePlan):
            raise TypeError("frame must be LocalRelativisticFramePlan.")
        if (
            stage_frame.frame_id != self.frame_id
            or stage_frame.units.contract_id != self.unit_contract_id
        ):
            raise ValueError("Hierarchy frame lineage or unit contract mismatch.")
        scale = jnp.asarray(
            stage_frame.scale_factor if scale_factor is None else scale_factor,
            dtype=values.dtype,
        )
        time = jnp.asarray(
            stage_frame.time if conformal_time is None else conformal_time,
            dtype=values.dtype,
        )
        realization = (
            self.initial_frame_realization_id
            if frame_realization_id is None
            else _identifier(frame_realization_id, "frame_realization_id")
        )
        if scale.shape != () or time.shape != ():
            raise ValueError("Hierarchy scale factor and conformal time must be scalar.")
        active_k = (
            jnp.ones((self.wave_numbers.size,), dtype=jnp.bool_)
            if k_active is None
            else jnp.asarray(k_active, dtype=jnp.bool_)
        )
        active_q = (
            jnp.ones((self.momentum_nodes.size,), dtype=jnp.bool_)
            if momentum_active is None
            else jnp.asarray(momentum_active, dtype=jnp.bool_)
        )
        if active_k.shape != (self.wave_numbers.size,) or active_q.shape != (
            self.momentum_nodes.size,
        ):
            raise ValueError("Hierarchy active masks do not match quadrature capacity.")
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(e_mode))
            & jnp.all(jnp.isfinite(b_mode))
            & jnp.isfinite(scale)
            & jnp.isfinite(time)
        )
        valid = (
            finite
            & (scale > 0.0)
            & jnp.all(stage_frame.admissible)
            & (scale == jnp.asarray(stage_frame.scale_factor).reshape(()))
            & (time == jnp.asarray(stage_frame.time).reshape(()))
        )
        return DarkRadiationHierarchyState(
            values,
            e_mode,
            b_mode,
            scale,
            time,
            active_k,
            active_q,
            jnp.asarray(stage_frame.frame_token),
            jnp.asarray(stage_frame.observer_coordinates),
            _content_id_words(realization, "frame_realization_id"),
            valid,
            _identifier(state_id, "state_id"),
            self.frame_id,
            self.unit_contract_id,
        )

    def _streaming_rhs(self, values: Array, /) -> Array:
        lower = jnp.concatenate(
            (jnp.zeros_like(values[..., :1]), values[..., :-1]), axis=-1
        )
        upper = jnp.concatenate(
            (values[..., 1:], jnp.zeros_like(values[..., :1])), axis=-1
        )
        ell = jnp.arange(self.multipole_count, dtype=values.dtype)
        k = self.wave_numbers.astype(values.dtype)[:, None, None]
        rhs = k * (ell * lower - (ell + 1.0) * upper) / (2.0 * ell + 1.0)
        # Asymptotic terminal closure F_{L+1}=((2L+1)/(k eta))F_L-F_{L-1}
        # is not silently imposed here; the terminal outgoing term is zero and
        # its adequacy is reported by closure_evidence.
        return rhs

    def rhs(
        self,
        state: DarkRadiationHierarchyState,
        metric_source: ArrayLike,
        collision_rate: ArrayLike,
        /,
    ) -> DarkRadiationHierarchyState:
        if not isinstance(state, DarkRadiationHierarchyState):
            raise TypeError("state must be DarkRadiationHierarchyState.")
        metric = jnp.asarray(metric_source, dtype=state.intensity.dtype)
        collision = jnp.asarray(collision_rate, dtype=state.intensity.dtype)
        if metric.shape != (self.wave_numbers.size, 2):
            raise ValueError("metric_source must have one monopole/dipole pair per k.")
        if collision.shape not in ((), (self.momentum_nodes.size,)):
            raise ValueError("collision_rate must be scalar or one value per momentum.")
        source = jnp.zeros_like(state.intensity)
        source = source.at[..., 0].add(metric[:, None, 0])
        source = source.at[..., 1].add(metric[:, None, 1])
        rate = collision + self.self_interaction_rate
        damping_selector = (jnp.arange(self.multipole_count) >= 2).astype(
            state.intensity.dtype
        )
        damping = rate[..., None] * damping_selector
        intensity_rhs = (
            self._streaming_rhs(state.intensity) + source - damping * state.intensity
        )
        e_rhs = self._streaming_rhs(state.polarization_e) - damping * state.polarization_e
        b_rhs = self._streaming_rhs(state.polarization_b) - damping * state.polarization_b
        active = state.k_active[:, None, None] & state.momentum_active[None, :, None]
        intensity_rhs = jnp.where(active, intensity_rhs, 0.0)
        e_rhs = jnp.where(active, e_rhs, 0.0)
        b_rhs = jnp.where(active, b_rhs, 0.0)
        return DarkRadiationHierarchyState(
            intensity_rhs,
            e_rhs,
            b_rhs,
            jnp.zeros_like(state.scale_factor),
            jnp.ones_like(state.conformal_time),
            state.k_active,
            state.momentum_active,
            state.frame_token,
            state.observer_coordinates,
            state.frame_realization_words,
            state.valid & jnp.all(jnp.isfinite(intensity_rhs)),
            state.state_id,
            state.frame_id,
            state.unit_contract_id,
        )

    def advance(
        self,
        state: DarkRadiationHierarchyState,
        step_size: ArrayLike,
        metric_source: ArrayLike,
        collision_rate: ArrayLike,
        /,
        *,
        end_frame: LocalRelativisticFramePlan,
        state_id: str,
        end_frame_realization_id: str,
    ) -> tuple[DarkRadiationHierarchyState, DarkRadiationHierarchyClosureEvidence]:
        dt = jnp.asarray(step_size, dtype=state.intensity.dtype)
        if dt.shape != ():
            raise ValueError("Hierarchy step size must be scalar.")
        if not isinstance(end_frame, LocalRelativisticFramePlan):
            raise TypeError("end_frame must be LocalRelativisticFramePlan.")
        if (
            end_frame.frame_id != state.frame_id
            or end_frame.units.contract_id != state.unit_contract_id
        ):
            raise ValueError("Hierarchy endpoint frame lineage or units mismatch.")
        derivative = self.rhs(state, metric_source, collision_rate)
        candidate = self.initialize(
            state.intensity + dt * derivative.intensity,
            polarization_e=state.polarization_e + dt * derivative.polarization_e,
            polarization_b=state.polarization_b + dt * derivative.polarization_b,
            scale_factor=end_frame.scale_factor,
            conformal_time=end_frame.time,
            frame=end_frame,
            frame_realization_id=end_frame_realization_id,
            k_active=state.k_active,
            momentum_active=state.momentum_active,
            state_id=state_id,
        )
        evidence = self.closure_evidence(candidate, collision_rate)
        accepted = (
            state.valid
            & candidate.valid
            & evidence.accepted
            & (dt > 0.0)
            & (candidate.conformal_time == state.conformal_time + dt)
        )
        selected = DarkRadiationHierarchyState(
            jnp.where(accepted, candidate.intensity, state.intensity),
            jnp.where(accepted, candidate.polarization_e, state.polarization_e),
            jnp.where(accepted, candidate.polarization_b, state.polarization_b),
            jnp.where(accepted, candidate.scale_factor, state.scale_factor),
            jnp.where(accepted, candidate.conformal_time, state.conformal_time),
            state.k_active,
            state.momentum_active,
            jnp.where(accepted, candidate.frame_token, state.frame_token),
            jnp.where(
                accepted,
                candidate.observer_coordinates,
                state.observer_coordinates,
            ),
            jnp.where(
                accepted,
                candidate.frame_realization_words,
                state.frame_realization_words,
            ),
            accepted,
            candidate.state_id,
            state.frame_id,
            state.unit_contract_id,
        )
        return selected, evidence

    def closure_evidence(
        self,
        state: DarkRadiationHierarchyState,
        collision_rate: ArrayLike = 0.0,
        /,
    ) -> DarkRadiationHierarchyClosureEvidence:
        penultimate = jnp.abs(state.intensity[..., -2])
        terminal = jnp.abs(state.intensity[..., -1])
        scale = jnp.maximum(jnp.max(jnp.abs(state.intensity), axis=-1), 1.0e-30)
        ratio = terminal / jnp.maximum(penultimate, 1.0e-30)
        bound = terminal / scale
        active = state.k_active[:, None] & state.momentum_active[None, :]
        qualified = jnp.all(~active | (bound <= self.closure_tolerance))
        collision = jnp.asarray(collision_rate, dtype=state.intensity.dtype)
        tight = jnp.all(collision * state.scale_factor > self.wave_numbers[:, None])
        finite = jnp.all(jnp.isfinite(state.intensity))
        accepted = state.valid & finite & qualified
        return DarkRadiationHierarchyClosureEvidence(
            ratio,
            bound,
            qualified,
            tight,
            finite,
            accepted,
            self.plan_id,
        )

    def line_of_sight(
        self, state: DarkRadiationHierarchyState, /
    ) -> DarkRadiationLineOfSightOutput:
        weight = self.momentum_weights.astype(state.intensity.dtype)
        normalization = jnp.sum(weight)
        average = lambda value: (
            contract("q,kql->kl", weight, value, backend="jax") / normalization
        )
        moments = average(state.intensity)
        e_mode = average(state.polarization_e)
        b_mode = average(state.polarization_b)
        output_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-line-of-sight",
                "plan": self.plan_id,
                "source_state": state.state_id,
            }
        )
        return DarkRadiationLineOfSightOutput(
            moments[:, 0],
            self.wave_numbers.astype(moments.dtype) * moments[:, 1],
            moments[:, 2],
            e_mode[:, 2],
            b_mode[:, 2],
            state.state_id,
            output_id,
        )


class DarkRadiationVETEvidence(StrictModule, NonTrainableState):
    residual_history: Array
    iteration_count: Array
    converged: Array
    lagged: Array
    shadow_contrast: Array
    finite: Array
    accepted: Array
    source_state_id: str = eqx.field(static=True)
    tensor_id: str = eqx.field(static=True)


class DarkRadiationVETResult(StrictModule, NonTrainableState):
    intensity: Array
    eddington_tensor: Array
    evidence: DarkRadiationVETEvidence


class DarkRadiationVETPlan(StrictModule, NonTrainableState):
    """Fixed angular-quadrature formal solve for research VET closures."""

    directions: Array
    weights: Array
    maximum_iterations: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        directions: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        maximum_iterations: int = 8,
        residual_tolerance: float = 1.0e-6,
    ):
        direction = jnp.asarray(directions)
        weight = jnp.asarray(weights, dtype=direction.dtype)
        iterations = int(maximum_iterations)
        tolerance = _positive(residual_tolerance, "residual_tolerance")
        if direction.ndim != 2 or direction.shape[-1] != 3:
            raise ValueError("VET directions must have shape (angle, 3).")
        if weight.shape != direction.shape[:1] or iterations <= 0:
            raise ValueError("VET quadrature weights or iteration capacity are invalid.")
        norms = np.linalg.norm(np.asarray(direction), axis=-1)
        if (
            not np.all(np.isfinite(norms))
            or not np.allclose(norms, 1.0)
            or not np.all(np.asarray(weight) > 0.0)
        ):
            raise ValueError("VET directions must be unit vectors with positive weights.")
        self.directions = direction
        self.weights = weight
        self.maximum_iterations = iterations
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-vet",
                "directions": np.asarray(direction).tolist(),
                "weights": np.asarray(weight).tolist(),
                "maximum_iterations": iterations,
                "residual_tolerance": tolerance,
            }
        )

    def formal_solve(
        self,
        incident_intensity: ArrayLike,
        source_function: ArrayLike,
        extinction: ArrayLike,
        path_length: ArrayLike,
        /,
        *,
        source_state_id: str,
        lagged: bool = False,
    ) -> DarkRadiationVETResult:
        incident = jnp.asarray(incident_intensity)
        source = jnp.asarray(source_function, dtype=incident.dtype)
        opacity = jnp.asarray(extinction, dtype=incident.dtype)
        distance = jnp.asarray(path_length, dtype=incident.dtype)
        if incident.shape[-1] != self.directions.shape[0]:
            raise ValueError("VET incident intensity angular capacity is invalid.")
        lane_shape = incident.shape[:-1]
        if (
            source.shape != lane_shape
            or opacity.shape != lane_shape
            or distance.shape != lane_shape
        ):
            raise ValueError(
                "VET source, extinction, and distance must match intensity lanes."
            )
        valid_input = (
            jnp.all(jnp.isfinite(incident), axis=-1)
            & jnp.isfinite(source)
            & jnp.isfinite(opacity)
            & jnp.isfinite(distance)
            & (incident >= 0.0).all(axis=-1)
            & (source >= 0.0)
            & (opacity >= 0.0)
            & (distance >= 0.0)
        )
        attenuation = jnp.exp(-opacity * distance)
        target = incident * attenuation[..., None] + source[..., None] * (
            1.0 - attenuation[..., None]
        )

        def iteration(_, carry):
            current, residuals = carry
            updated = 0.5 * (current + target)
            residual = jnp.max(jnp.abs(updated - current), axis=-1)
            residuals = residuals.at[..., _].set(residual)
            return updated, residuals

        residual_shape = lane_shape + (self.maximum_iterations,)
        initial_residuals = jnp.zeros(residual_shape, dtype=incident.dtype)
        intensity, residuals = jax.lax.fori_loop(
            0, self.maximum_iterations, iteration, (incident, initial_residuals)
        )
        total = contract("a,...a->...", self.weights, intensity, backend="jax")
        numerator = contract(
            "a,ai,aj,...a->...ij",
            self.weights,
            self.directions,
            self.directions,
            intensity,
            backend="jax",
        )
        tensor = numerator / jnp.maximum(total[..., None, None], 1.0e-30)
        converged = residuals[..., -1] <= self.residual_tolerance
        finite = jnp.all(jnp.isfinite(tensor), axis=(-2, -1))
        projection = contract(
            "...ij,ai,aj->...a", tensor, self.directions, self.directions, backend="jax"
        )
        shadow = jnp.max(projection, axis=-1) - jnp.min(projection, axis=-1)
        accepted = valid_input & converged & finite & (total > 0.0)
        tensor_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-vet-tensor",
                "plan": self.plan_id,
                "source_state": _identifier(source_state_id, "source_state_id"),
                "lagged": bool(lagged),
            }
        )
        evidence = DarkRadiationVETEvidence(
            residuals,
            jnp.asarray(self.maximum_iterations, dtype=jnp.int32),
            converged,
            jnp.asarray(lagged),
            shadow,
            finite,
            accepted,
            source_state_id,
            tensor_id,
        )
        return DarkRadiationVETResult(intensity, tensor, evidence)


class DarkRadiationConversionReceipt(StrictModule, NonTrainableState):
    """Explicit averaging or linearization authority between representations."""

    source_integral: Array
    target_integral: Array
    conservation_defect: Array
    accepted: Array
    source_state_id: str = eqx.field(static=True)
    target_state_id: str = eqx.field(static=True)
    source_representation: str = eqx.field(static=True)
    target_representation: str = eqx.field(static=True)
    operation: str = eqx.field(static=True)
    differentiation_policy: str = eqx.field(static=True)
    receipt_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_integral: ArrayLike,
        target_integral: ArrayLike,
        /,
        *,
        source_state_id: str,
        target_state_id: str,
        source_representation: str,
        target_representation: str,
        operation: str,
        differentiation_policy: str,
    ):
        source = jnp.asarray(source_integral)
        target = jnp.asarray(target_integral, dtype=source.dtype)
        if source.shape != target.shape:
            raise ValueError("Conversion receipt integral shapes must match.")
        source_id = _identifier(source_state_id, "source_state_id")
        target_id = _identifier(target_state_id, "target_state_id")
        source_kind = _identifier(source_representation, "source_representation")
        target_kind = _identifier(target_representation, "target_representation")
        operation_ = _identifier(operation, "operation")
        policy = _identifier(differentiation_policy, "differentiation_policy")
        defect = target - source
        scale = jnp.maximum(jnp.abs(source), 1.0)
        tolerance = 128.0 * jnp.finfo(source.dtype).eps * scale
        accepted = jnp.all(jnp.isfinite(defect)) & jnp.all(jnp.abs(defect) <= tolerance)
        self.source_integral = source
        self.target_integral = target
        self.conservation_defect = defect
        self.accepted = accepted
        self.source_state_id = source_id
        self.target_state_id = target_id
        self.source_representation = source_kind
        self.target_representation = target_kind
        self.operation = operation_
        self.differentiation_policy = policy
        self.receipt_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-conversion-receipt",
                "source_state": source_id,
                "target_state": target_id,
                "source_representation": source_kind,
                "target_representation": target_kind,
                "operation": operation_,
                "differentiation_policy": policy,
            }
        )


__all__ = [
    "CosmologicalMultigroupM1System",
    "DarkRadiationBoltzmannHierarchyPlan",
    "DarkRadiationConversionReceipt",
    "DarkRadiationFourForce",
    "DarkRadiationGroupRedshiftResult",
    "DarkRadiationHierarchyClosureEvidence",
    "DarkRadiationHierarchyState",
    "DarkRadiationLineOfSightOutput",
    "DarkRadiationM1ExchangeResult",
    "DarkRadiationM1Qualification",
    "DarkRadiationM1RealizabilityEvidence",
    "DarkRadiationM1RealizabilityResult",
    "DarkRadiationM1RefluxResult",
    "DarkRadiationVETEvidence",
    "DarkRadiationVETPlan",
    "DarkRadiationVETResult",
]
