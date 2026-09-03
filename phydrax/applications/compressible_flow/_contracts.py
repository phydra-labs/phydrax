#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._riemann import HLLFluxPlan
from ...equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
    HomogeneousMixtureEulerSystem,
)
from ...equations._homogeneous_thermodynamics import HomogeneousHelmholtzPlan
from ...equations._transport_closures import AbstractTransportClosure
from ...qualification._evidence import QualificationEvidence, SupportDependency


CompressibleEquation: TypeAlias = Literal["euler", "navier_stokes"]
CompressibleRoute: TypeAlias = Literal[
    "tensor-dgsem",
    "nodal-dg-ldg",
    "structured-fv",
    "mapped-fv",
]
CompressibleFidelity: TypeAlias = Literal["unqualified", "dns-candidate"]
ShockReconstruction: TypeAlias = Literal["weno_z", "teno", "mp5"]


class FiniteXBoundaryLayerInflowPlan(StrictModule, NonTrainableState):
    """Finite-x boundary-layer inflow in canonical mixture primitive variables."""

    free_stream_density: float = eqx.field(static=True)
    free_stream_mass_fractions: tuple[float, ...] = eqx.field(static=True)
    free_stream_velocity: float = eqx.field(static=True)
    free_stream_temperature: float = eqx.field(static=True)
    boundary_layer_thickness: float = eqx.field(static=True)
    velocity_exponent: float = eqx.field(static=True)
    wall_temperature: float | None = eqx.field(static=True)
    inflow_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        free_stream_density: float,
        free_stream_mass_fractions: Sequence[float],
        free_stream_velocity: float,
        free_stream_temperature: float,
        boundary_layer_thickness: float,
        velocity_exponent: float = 1.0,
        wall_temperature: float | None = None,
    ):
        density = float(free_stream_density)
        mass_fractions = tuple(float(value) for value in free_stream_mass_fractions)
        velocity = float(free_stream_velocity)
        temperature = float(free_stream_temperature)
        thickness = float(boundary_layer_thickness)
        exponent = float(velocity_exponent)
        wall_temperature_ = None if wall_temperature is None else float(wall_temperature)
        scalars = (density, velocity, temperature, thickness, exponent)
        if (
            not mass_fractions
            or any(not np.isfinite(value) or value < 0.0 for value in mass_fractions)
            or not np.isclose(sum(mass_fractions), 1.0)
            or any(not np.isfinite(value) for value in scalars)
            or density <= 0.0
            or velocity <= 0.0
            or temperature <= 0.0
            or thickness <= 0.0
            or exponent <= 0.0
            or (
                wall_temperature_ is not None
                and (not np.isfinite(wall_temperature_) or wall_temperature_ <= 0.0)
            )
        ):
            raise ValueError("Finite-x boundary-layer inflow parameters are invalid.")
        self.free_stream_density = density
        self.free_stream_mass_fractions = mass_fractions
        self.free_stream_velocity = velocity
        self.free_stream_temperature = temperature
        self.boundary_layer_thickness = thickness
        self.velocity_exponent = exponent
        self.wall_temperature = wall_temperature_
        self.inflow_id = canonical_fingerprint(
            {
                "kind": "finite-x-compressible-boundary-layer-inflow",
                "free_stream_density": density,
                "free_stream_mass_fractions": mass_fractions,
                "free_stream_velocity": velocity,
                "free_stream_temperature": temperature,
                "boundary_layer_thickness": thickness,
                "velocity_exponent": exponent,
                "wall_temperature": wall_temperature_,
            }
        )

    def primitive(
        self,
        wall_distance: ArrayLike,
        system: HomogeneousMixtureEulerSystem
        | HomogeneousMixtureCompressibleNavierStokesSystem,
        /,
    ) -> Array:
        if not isinstance(
            system,
            (
                HomogeneousMixtureEulerSystem,
                HomogeneousMixtureCompressibleNavierStokesSystem,
            ),
        ):
            raise TypeError("system must be a canonical homogeneous-mixture gas system.")
        if system.dimension not in (2, 3):
            raise ValueError(
                "Finite-x boundary-layer inflow requires two or three dimensions."
            )
        if len(self.free_stream_mass_fractions) != system.species_count:
            raise ValueError("Inflow composition must contain one value per species.")
        distance = jnp.asarray(wall_distance)
        distance = eqx.error_if(
            distance,
            jnp.any(~jnp.isfinite(distance) | (distance < 0.0)),
            "Wall distance must be finite and nonnegative.",
        )
        eta = distance / self.boundary_layer_thickness
        profile = jnp.tanh(eta) ** self.velocity_exponent
        velocity = (
            jnp.zeros(distance.shape + (system.dimension,), dtype=distance.dtype)
            .at[..., 0]
            .set(self.free_stream_velocity * profile)
        )
        composition = jnp.asarray(self.free_stream_mass_fractions, dtype=distance.dtype)
        species_density = (
            jnp.full_like(distance, self.free_stream_density)[..., None] * composition
        )
        temperature = jnp.full_like(distance, self.free_stream_temperature)
        if self.wall_temperature is not None:
            temperature = self.wall_temperature + profile * (
                self.free_stream_temperature - self.wall_temperature
            )
        primitive = jnp.concatenate(
            (species_density, velocity, temperature[..., None]), axis=-1
        )
        thermodynamic_state = system.thermodynamics.evaluate_density_temperature(
            species_density, temperature
        )
        return eqx.error_if(
            primitive,
            jnp.any(~thermodynamic_state.evidence.successful),
            "Boundary-layer inflow lies outside canonical thermodynamic evidence.",
        )


class FiniteXBoundaryLayerCaseSpec(StrictModule, NonTrainableState):
    """Streamwise-finite boundary-layer domain and exact inflow ownership."""

    x_bounds: tuple[float, float] = eqx.field(static=True)
    wall_normal_bounds: tuple[float, float] = eqx.field(static=True)
    spanwise_bounds: tuple[float, float] | None = eqx.field(static=True)
    inflow: FiniteXBoundaryLayerInflowPlan
    outflow_kind: str = eqx.field(static=True)
    wall_kind: str = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        x_bounds: Sequence[float],
        wall_normal_bounds: Sequence[float],
        inflow: FiniteXBoundaryLayerInflowPlan,
        /,
        *,
        spanwise_bounds: Sequence[float] | None = None,
        outflow_kind: str = "characteristic-nonreflecting",
        wall_kind: str = "no-slip-thermal",
    ):
        x = tuple(float(value) for value in x_bounds)
        wall_normal = tuple(float(value) for value in wall_normal_bounds)
        spanwise = (
            None
            if spanwise_bounds is None
            else tuple(float(value) for value in spanwise_bounds)
        )
        outflow = str(outflow_kind)
        wall = str(wall_kind)
        if (
            len(x) != 2
            or len(wall_normal) != 2
            or any(not np.isfinite(value) for value in (*x, *wall_normal))
            or x[1] <= x[0]
            or wall_normal[0] != 0.0
            or wall_normal[1] <= wall_normal[0]
            or (
                spanwise is not None
                and (
                    len(spanwise) != 2
                    or any(not np.isfinite(value) for value in spanwise)
                    or spanwise[1] <= spanwise[0]
                )
            )
            or not isinstance(inflow, FiniteXBoundaryLayerInflowPlan)
            or outflow != "characteristic-nonreflecting"
            or wall not in ("no-slip-adiabatic", "no-slip-thermal")
        ):
            raise ValueError("Finite-x boundary-layer case definition is invalid.")
        self.x_bounds = (x[0], x[1])
        self.wall_normal_bounds = (wall_normal[0], wall_normal[1])
        self.spanwise_bounds = None if spanwise is None else (spanwise[0], spanwise[1])
        self.inflow = inflow
        self.outflow_kind = outflow
        self.wall_kind = wall
        self.case_id = canonical_fingerprint(
            {
                "kind": "finite-x-compressible-boundary-layer-case",
                "x_bounds": x,
                "wall_normal_bounds": wall_normal,
                "spanwise_bounds": spanwise,
                "inflow": inflow.inflow_id,
                "outflow": outflow,
                "wall": wall,
            }
        )

    @property
    def dimension(self) -> int:
        return 2 if self.spanwise_bounds is None else 3


class CompressibleFlowCaseSpec(StrictModule, NonTrainableState):
    """Physical application case over one canonical homogeneous gas model."""

    thermodynamics: HomogeneousHelmholtzPlan
    boundary_layer: FiniteXBoundaryLayerCaseSpec | None
    name: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    equation: CompressibleEquation = eqx.field(static=True)
    route: CompressibleRoute = eqx.field(static=True)
    characteristic_length: float = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    reference_velocity: float = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    maximum_thermal_iterations: int = eqx.field(static=True)
    fidelity: CompressibleFidelity = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        dimension: int,
        equation: CompressibleEquation,
        route: CompressibleRoute,
        thermodynamics: HomogeneousHelmholtzPlan,
        /,
        *,
        characteristic_length: float = 1.0,
        reference_density: float = 1.0,
        reference_velocity: float = 1.0,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
        maximum_thermal_iterations: int = 80,
        fidelity: CompressibleFidelity = "unqualified",
        boundary_layer: FiniteXBoundaryLayerCaseSpec | None = None,
    ):
        name_ = str(name)
        dimension_ = int(dimension)
        length = float(characteristic_length)
        density = float(reference_density)
        velocity = float(reference_velocity)
        density_floor_ = float(density_floor)
        pressure_floor_ = float(pressure_floor)
        iterations = int(maximum_thermal_iterations)
        if (
            not name_
            or dimension_ not in (1, 2, 3)
            or equation not in ("euler", "navier_stokes")
            or route
            not in (
                "tensor-dgsem",
                "nodal-dg-ldg",
                "structured-fv",
                "mapped-fv",
            )
            or not isinstance(thermodynamics, HomogeneousHelmholtzPlan)
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in (
                    length,
                    density,
                    velocity,
                    density_floor_,
                    pressure_floor_,
                )
            )
            or iterations <= 0
            or fidelity not in ("unqualified", "dns-candidate")
            or (
                boundary_layer is not None
                and (
                    not isinstance(boundary_layer, FiniteXBoundaryLayerCaseSpec)
                    or boundary_layer.dimension != dimension_
                )
            )
        ):
            raise ValueError("Compressible-flow case specification is invalid.")
        if boundary_layer is not None and route not in (
            "tensor-dgsem",
            "structured-fv",
            "mapped-fv",
        ):
            raise ValueError("Finite-x boundary layers require a boundary-capable route.")
        self.name = name_
        self.dimension = dimension_
        self.equation = equation
        self.route = route
        self.thermodynamics = thermodynamics
        self.characteristic_length = length
        self.reference_density = density
        self.reference_velocity = velocity
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.maximum_thermal_iterations = iterations
        self.fidelity = fidelity
        self.boundary_layer = boundary_layer
        self.case_id = canonical_fingerprint(
            {
                "kind": "compressible-flow-case",
                "name": name_,
                "dimension": dimension_,
                "equation": equation,
                "route": route,
                "thermodynamics": thermodynamics.model_id,
                "characteristic_length": length,
                "reference_density": density,
                "reference_velocity": velocity,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
                "maximum_thermal_iterations": iterations,
                "fidelity": fidelity,
                "boundary_layer": None
                if boundary_layer is None
                else boundary_layer.case_id,
            }
        )

    @property
    def species_count(self) -> int:
        return self.thermodynamics.schema.species_count

    @property
    def component_count(self) -> int:
        return self.species_count + self.dimension + 1

    @property
    def claims_dns(self) -> bool:
        return False

    def prepare_inviscid_system(self) -> HomogeneousMixtureEulerSystem:
        return HomogeneousMixtureEulerSystem(
            self.thermodynamics,
            self.dimension,
            density_floor=self.density_floor,
            pressure_floor=self.pressure_floor,
            maximum_thermal_iterations=self.maximum_thermal_iterations,
        )

    def primitive_to_conserved(self, primitive: ArrayLike, /) -> Array:
        return self.prepare_inviscid_system().primitive_to_conserved(
            jnp.asarray(primitive)
        )

    def prepare_system(
        self,
        transport: AbstractTransportClosure | None = None,
        /,
        *,
        species_diffusivities: ArrayLike | None = None,
    ) -> HomogeneousMixtureEulerSystem | HomogeneousMixtureCompressibleNavierStokesSystem:
        """Prepare the canonical physical system supported by the declared route."""
        if self.equation == "euler":
            if transport is not None or species_diffusivities is not None:
                raise ValueError("Euler case preparation does not accept transport.")
            return self.prepare_inviscid_system()
        if not isinstance(transport, AbstractTransportClosure):
            raise TypeError(
                "Navier-Stokes case preparation requires an AbstractTransportClosure."
            )
        return HomogeneousMixtureCompressibleNavierStokesSystem(
            self.thermodynamics,
            transport,
            self.dimension,
            species_diffusivities=species_diffusivities,
            density_floor=self.density_floor,
            pressure_floor=self.pressure_floor,
            maximum_thermal_iterations=self.maximum_thermal_iterations,
        )

    def conserved_to_primitive(self, conserved: ArrayLike, /) -> Array:
        return self.prepare_inviscid_system().conserved_to_primitive(
            jnp.asarray(conserved)
        )


class AllSpeedCompressiblePolicy(StrictModule, NonTrainableState):
    """Declared O(M) acoustic dissipation scaling for low-Mach smooth flow."""

    reference_mach: float = eqx.field(static=True)
    minimum_mach: float = eqx.field(static=True)
    scaling: str = eqx.field(static=True)
    asymptotic_order: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_mach: float = 1.0,
        *,
        minimum_mach: float = 0.0,
        scaling: str = "linear-local-mach",
    ):
        reference = float(reference_mach)
        minimum = float(minimum_mach)
        if (
            not np.isfinite(reference)
            or reference <= 0.0
            or not np.isfinite(minimum)
            or not 0.0 <= minimum <= 1.0
            or scaling != "linear-local-mach"
        ):
            raise ValueError("All-speed policy values are invalid.")
        self.reference_mach = reference
        self.minimum_mach = minimum
        self.scaling = scaling
        self.asymptotic_order = 1
        self.policy_id = canonical_fingerprint(
            {
                "kind": "all-speed-compressible-policy",
                "reference_mach": reference,
                "minimum_mach": minimum,
                "scaling": scaling,
                "asymptotic_order": 1,
            }
        )

    def pressure_dissipation_scale(self, local_mach: ArrayLike, /) -> Array:
        mach = jnp.asarray(local_mach)
        mach = eqx.error_if(
            mach,
            jnp.any(~jnp.isfinite(mach) | (mach < 0.0)),
            "Local Mach number must be finite and nonnegative.",
        )
        return jnp.minimum(
            1.0,
            jnp.maximum(self.minimum_mach, mach / self.reference_mach),
        )

    def scaled_acoustic_speed(
        self, velocity_magnitude: ArrayLike, sound_speed: ArrayLike, /
    ) -> Array:
        velocity = jnp.asarray(velocity_magnitude)
        sound = jnp.asarray(sound_speed)
        local_mach = velocity / sound
        return self.pressure_dissipation_scale(local_mach) * sound


class ShockRouteLedger(StrictModule):
    route_label: str = eqx.field(static=True)
    shock_detected: Array
    primary_admissible: Array
    fallback_used: Array
    fallback_count: Array
    total_count: Array
    fallback_fraction: Array
    ledger_id: str = eqx.field(static=True)


class ShockResolvingPolicy(StrictModule, NonTrainableState):
    """Exact shock-route label and generic admissibility-preserving HLL fallback."""

    all_speed: AllSpeedCompressiblePolicy
    fallback_flux: HLLFluxPlan
    reconstruction: ShockReconstruction = eqx.field(static=True)
    sensor_threshold: float = eqx.field(static=True)
    route_label: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: ShockReconstruction = "weno_z",
        /,
        *,
        sensor_threshold: float = 0.05,
        all_speed: AllSpeedCompressiblePolicy | None = None,
        fallback_flux: HLLFluxPlan | None = None,
    ):
        threshold = float(sensor_threshold)
        all_speed_ = AllSpeedCompressiblePolicy() if all_speed is None else all_speed
        fallback = HLLFluxPlan() if fallback_flux is None else fallback_flux
        if (
            reconstruction not in ("weno_z", "teno", "mp5")
            or not np.isfinite(threshold)
            or threshold <= 0.0
            or not isinstance(all_speed_, AllSpeedCompressiblePolicy)
            or not isinstance(fallback, HLLFluxPlan)
        ):
            raise ValueError("Shock-resolving policy is invalid.")
        label = f"shock-resolving:{reconstruction}:all-speed->generic-hll"
        self.reconstruction = reconstruction
        self.sensor_threshold = threshold
        self.all_speed = all_speed_
        self.fallback_flux = fallback
        self.route_label = label
        self.policy_id = canonical_fingerprint(
            {
                "kind": "compressible-shock-resolving-policy",
                "reconstruction": reconstruction,
                "sensor_threshold": threshold,
                "all_speed": all_speed_.policy_id,
                "fallback": fallback.flux_id,
                "route_label": label,
            }
        )

    def ledger(
        self,
        sensor: ArrayLike,
        primary_admissible: ArrayLike,
        primary_successful: ArrayLike = True,
        /,
    ) -> ShockRouteLedger:
        sensor_ = jnp.asarray(sensor)
        admissible = jnp.asarray(primary_admissible, dtype=bool)
        successful = jnp.asarray(primary_successful, dtype=bool)
        shape = jnp.broadcast_shapes(sensor_.shape, admissible.shape, successful.shape)
        sensor_ = jnp.broadcast_to(sensor_, shape)
        admissible = jnp.broadcast_to(admissible, shape)
        successful = jnp.broadcast_to(successful, shape)
        sensor_ = eqx.error_if(
            sensor_,
            jnp.any(~jnp.isfinite(sensor_) | (sensor_ < 0.0)),
            "Shock sensor must be finite and nonnegative.",
        )
        shock = sensor_ >= self.sensor_threshold
        fallback = shock | ~admissible | ~successful
        count = jnp.sum(fallback.astype(jnp.int32))
        total = jnp.asarray(fallback.size, dtype=jnp.int32)
        fraction = count.astype(sensor_.dtype) / total.astype(sensor_.dtype)
        return ShockRouteLedger(
            self.route_label,
            shock,
            admissible,
            fallback,
            count,
            total,
            fraction,
            canonical_fingerprint(
                {"kind": "compressible-shock-route-ledger", "policy": self.policy_id}
            ),
        )

    def select_flux(
        self,
        primary_flux: ArrayLike,
        fallback_flux: ArrayLike,
        ledger: ShockRouteLedger,
        /,
    ) -> Array:
        if not isinstance(ledger, ShockRouteLedger):
            raise TypeError("ledger must be ShockRouteLedger.")
        primary = jnp.asarray(primary_flux)
        fallback = jnp.asarray(fallback_flux)
        if (
            primary.shape != fallback.shape
            or primary.shape[:-1] != ledger.fallback_used.shape
        ):
            raise ValueError("Shock fluxes and fallback ledger have incompatible shapes.")
        return jnp.where(ledger.fallback_used[..., None], fallback, primary)


class CompressibleQualificationEvidence(StrictModule, NonTrainableState):
    """Route-exact, unsigned application evidence ready for qualification binding."""

    case_id: str = eqx.field(static=True)
    route_label: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    checks: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    qualification_ready: bool = eqx.field(static=True)
    dns_claimed: bool = eqx.field(static=True)
    signed: bool = eqx.field(static=True)
    released: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        case_id: str,
        route_label: str,
        method_id: str,
        checks: Sequence[tuple[str, bool]],
        /,
    ):
        case = str(case_id)
        route = str(route_label)
        method = str(method_id)
        checks_ = tuple((str(name), bool(passed)) for name, passed in checks)
        if (
            not case
            or not route
            or not method
            or not checks_
            or any(not name for name, _ in checks_)
            or len({name for name, _ in checks_}) != len(checks_)
        ):
            raise ValueError("Compressible qualification evidence is invalid.")
        support_tuple_id = canonical_fingerprint(
            {"application": "compressible-flow", "route": route, "method": method}
        )
        ready = all(passed for _, passed in checks_)
        self.case_id = case
        self.route_label = route
        self.method_id = method
        self.support_tuple_id = support_tuple_id
        self.checks = checks_
        self.qualification_ready = ready
        self.dns_claimed = False
        self.signed = False
        self.released = False
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "compressible-qualification-evidence",
                "case": case,
                "route": route,
                "method": method,
                "support_tuple": support_tuple_id,
                "checks": checks_,
                "qualification_ready": ready,
                "dns_claimed": False,
                "signed": False,
                "released": False,
            }
        )

    def support_dependency(self, profile_id: str, /) -> SupportDependency:
        return SupportDependency(profile_id, self.support_tuple_id)

    def bind_qualification_evidence(
        self,
        /,
        *,
        evidence_kind: str,
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        reduction: str,
        replay_id: str,
        raw_artifact_ids: Sequence[str],
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
        reason: str,
        requalification_triggers: Sequence[str] = (),
        observed_resource_record_ids: Sequence[str] = (),
        forecast_resource_record_ids: Sequence[str] = (),
    ) -> QualificationEvidence:
        """Bind route evidence to the governed, expiring qualification spine."""

        return QualificationEvidence(
            evidence_kind,
            "passed" if self.qualification_ready else "inconclusive",
            (self.case_id, self.method_id, self.support_tuple_id),
            build_id=build_id,
            environment_id=environment_id,
            backend=backend,
            topology=topology,
            precision=precision,
            reduction=reduction,
            replay_id=replay_id,
            criteria_ids=tuple(name for name, _ in self.checks),
            raw_artifact_ids=raw_artifact_ids,
            reviewer_id=reviewer_id,
            issued_at=issued_at,
            expires_at=expires_at,
            reason=reason,
            requalification_triggers=requalification_triggers,
            observed_resource_record_ids=observed_resource_record_ids,
            forecast_resource_record_ids=forecast_resource_record_ids,
        )


__all__ = [
    "AllSpeedCompressiblePolicy",
    "CompressibleFlowCaseSpec",
    "CompressibleQualificationEvidence",
    "FiniteXBoundaryLayerCaseSpec",
    "FiniteXBoundaryLayerInflowPlan",
    "ShockResolvingPolicy",
    "ShockRouteLedger",
]
