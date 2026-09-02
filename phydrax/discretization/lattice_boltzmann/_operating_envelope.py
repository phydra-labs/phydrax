#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._collision import (
    BGKCollisionPlan,
    CentralMomentCollisionPlan,
    CumulantCollisionPlan,
    EntropicCollisionPlan,
    KBCCollisionPlan,
    LatticeBoltzmannCollisionPlan,
    MRTCollisionPlan,
    RegularizedCollisionPlan,
    SmagorinskyCollisionPlan,
    TRTCollisionPlan,
)
from ._forcing import GuoForcingPlan
from ._lattice import LatticeBoltzmannVelocitySet
from ._precision import LatticeBoltzmannPrecisionPolicy


_COLLISION_TYPES = (
    BGKCollisionPlan,
    TRTCollisionPlan,
    MRTCollisionPlan,
    RegularizedCollisionPlan,
    SmagorinskyCollisionPlan,
    CentralMomentCollisionPlan,
    CumulantCollisionPlan,
    KBCCollisionPlan,
    EntropicCollisionPlan,
)
_ENVELOPE_CHECKS = (
    "finite",
    "mach-number",
    "knudsen-number",
    "relaxation-rate",
    "density",
    "density-ratio",
    "force-number",
    "interface-width",
    "wall-resolution",
    "viscosity-ratio",
    "cahn-number",
    "capillary-number",
    "mass-drift",
    "spurious-current",
)


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier or identifier != identifier.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return identifier


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


class LatticeBoltzmannEnvelopeError(ValueError):
    """Raised when an exact LBM support tuple is used outside its envelope."""


class LatticeBoltzmannHardwareTarget(StrictModule, NonTrainableState):
    """Exact JAX hardware allocation and per-device memory ceiling."""

    platform: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    architecture: str = eqx.field(static=True)
    host_count: int = eqx.field(static=True)
    devices_per_host: int = eqx.field(static=True)
    maximum_device_bytes: int = eqx.field(static=True)
    hardware_id: str = eqx.field(static=True)

    def __init__(
        self,
        platform: str,
        provider: str,
        architecture: str,
        /,
        *,
        host_count: int = 1,
        devices_per_host: int = 1,
        maximum_device_bytes: int,
    ):
        platform_ = _identifier(platform, "platform")
        if platform_ not in ("cpu", "gpu", "tpu"):
            raise ValueError("LBM hardware platform must be 'cpu', 'gpu', or 'tpu'.")
        provider_ = _identifier(provider, "provider")
        architecture_ = _identifier(architecture, "architecture")
        hosts = int(host_count)
        devices = int(devices_per_host)
        budget = int(maximum_device_bytes)
        if hosts <= 0 or devices <= 0 or budget <= 0:
            raise ValueError("LBM hardware counts and memory ceiling must be positive.")
        self.platform = platform_
        self.provider = provider_
        self.architecture = architecture_
        self.host_count = hosts
        self.devices_per_host = devices
        self.maximum_device_bytes = budget
        self.hardware_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-hardware-target",
                "platform": platform_,
                "provider": provider_,
                "architecture": architecture_,
                "host_count": hosts,
                "devices_per_host": devices,
                "maximum_device_bytes": budget,
            }
        )

    @property
    def device_count(self) -> int:
        return self.host_count * self.devices_per_host

    @property
    def multi_host(self) -> bool:
        return self.host_count > 1


class LatticeBoltzmannOperatingPoint(StrictModule):
    """Observed nondimensional coordinates evaluated by an operating envelope."""

    mach_number: Array
    knudsen_number: Array
    relaxation_rate: Array
    minimum_density: Array
    maximum_density: Array
    force_number: Array
    interface_width_cells: Array
    wall_resolution_cells: Array
    viscosity_ratio: Array
    cahn_number: Array
    capillary_number: Array
    relative_mass_drift: Array
    spurious_current_ratio: Array

    def __init__(
        self,
        *,
        mach_number: ArrayLike,
        knudsen_number: ArrayLike,
        relaxation_rate: ArrayLike,
        minimum_density: ArrayLike,
        maximum_density: ArrayLike,
        force_number: ArrayLike,
        interface_width_cells: ArrayLike = 0.0,
        wall_resolution_cells: ArrayLike = 0.0,
        viscosity_ratio: ArrayLike = 1.0,
        cahn_number: ArrayLike = 0.0,
        capillary_number: ArrayLike = 0.0,
        relative_mass_drift: ArrayLike = 0.0,
        spurious_current_ratio: ArrayLike = 0.0,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                mach_number,
                knudsen_number,
                relaxation_rate,
                minimum_density,
                maximum_density,
                force_number,
                interface_width_cells,
                wall_resolution_cells,
                viscosity_ratio,
                cahn_number,
                capillary_number,
                relative_mass_drift,
                spurious_current_ratio,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Every LBM operating coordinate must be scalar.")
        (
            self.mach_number,
            self.knudsen_number,
            self.relaxation_rate,
            self.minimum_density,
            self.maximum_density,
            self.force_number,
            self.interface_width_cells,
            self.wall_resolution_cells,
            self.viscosity_ratio,
            self.cahn_number,
            self.capillary_number,
            self.relative_mass_drift,
            self.spurious_current_ratio,
        ) = values


class LatticeBoltzmannEnvelopeAdmission(StrictModule, NonTrainableState):
    """Fail-closed, named predicate results for one operating point."""

    checks: Array
    margins: Array
    admitted: Array
    check_names: tuple[str, ...] = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)

    def __init__(
        self,
        checks: ArrayLike,
        margins: ArrayLike,
        envelope_id: str,
        /,
    ):
        checks_ = jnp.asarray(checks, dtype=bool)
        margins_ = jnp.asarray(margins)
        if checks_.shape != (len(_ENVELOPE_CHECKS),):
            raise ValueError("LBM envelope checks have an invalid shape.")
        if margins_.shape != checks_.shape:
            raise ValueError("LBM envelope margins must match the named checks.")
        self.checks = checks_
        self.margins = margins_
        self.admitted = jnp.all(checks_)
        self.check_names = _ENVELOPE_CHECKS
        self.envelope_id = _identifier(envelope_id, "envelope_id")

    def failed_checks(self, /) -> tuple[str, ...]:
        host = np.asarray(self.checks, dtype=bool)
        return tuple(
            name
            for name, passed in zip(self.check_names, host, strict=True)
            if not passed
        )


class LatticeBoltzmannResourceEstimate(StrictModule, NonTrainableState):
    """Static per-device allocation evidence produced before compilation."""

    state_bytes: int = eqx.field(static=True)
    temporary_bytes: int = eqx.field(static=True)
    halo_bytes: int = eqx.field(static=True)
    checkpoint_bytes: int = eqx.field(static=True)
    output_bytes: int = eqx.field(static=True)
    total_bytes: int = eqx.field(static=True)
    maximum_device_bytes: int = eqx.field(static=True)
    fits_budget: bool = eqx.field(static=True)
    precision_resource_assumptions_id: str = eqx.field(static=True)
    estimate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_bytes: int,
        temporary_bytes: int,
        halo_bytes: int,
        checkpoint_bytes: int,
        output_bytes: int,
        maximum_device_bytes: int,
        precision_resource_assumptions_id: str,
    ):
        components = tuple(
            int(value)
            for value in (
                state_bytes,
                temporary_bytes,
                halo_bytes,
                checkpoint_bytes,
                output_bytes,
            )
        )
        budget = int(maximum_device_bytes)
        if any(value < 0 for value in components) or budget <= 0:
            raise ValueError("LBM resource byte counts are invalid.")
        total = sum(components)
        assumptions = _identifier(
            precision_resource_assumptions_id,
            "precision_resource_assumptions_id",
        )
        (
            self.state_bytes,
            self.temporary_bytes,
            self.halo_bytes,
            self.checkpoint_bytes,
            self.output_bytes,
        ) = components
        self.total_bytes = total
        self.maximum_device_bytes = budget
        self.fits_budget = total <= budget
        self.precision_resource_assumptions_id = assumptions
        self.estimate_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-resource-estimate",
                "components": components,
                "total_bytes": total,
                "maximum_device_bytes": budget,
                "precision_resource_assumptions": assumptions,
            }
        )


class LatticeBoltzmannOperatingEnvelopePlan(StrictModule, NonTrainableState):
    """Exact LBM support tuple and immutable nondimensional validity limits.

    Admission never changes a collision, forcing route, or observed coordinate.
    Nonfinite and out-of-range observations are rejected by named predicates.
    """

    lattice: LatticeBoltzmannVelocitySet
    collision: LatticeBoltzmannCollisionPlan
    forcing: GuoForcingPlan | None
    precision: LatticeBoltzmannPrecisionPolicy
    hardware: LatticeBoltzmannHardwareTarget
    physics_model: str = eqx.field(static=True)
    boundary_model: str = eqx.field(static=True)
    minimum_relaxation_rate: float = eqx.field(static=True)
    maximum_relaxation_rate: float = eqx.field(static=True)
    maximum_mach_number: float = eqx.field(static=True)
    maximum_knudsen_number: float = eqx.field(static=True)
    minimum_density: float = eqx.field(static=True)
    maximum_density: float = eqx.field(static=True)
    maximum_density_ratio: float = eqx.field(static=True)
    maximum_force_number: float = eqx.field(static=True)
    minimum_interface_width_cells: float = eqx.field(static=True)
    minimum_wall_resolution_cells: float = eqx.field(static=True)
    maximum_viscosity_ratio: float = eqx.field(static=True)
    maximum_cahn_number: float = eqx.field(static=True)
    maximum_capillary_number: float = eqx.field(static=True)
    maximum_relative_mass_drift: float = eqx.field(static=True)
    maximum_spurious_current_ratio: float = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice: LatticeBoltzmannVelocitySet,
        collision: LatticeBoltzmannCollisionPlan,
        forcing: GuoForcingPlan | None,
        precision: LatticeBoltzmannPrecisionPolicy,
        hardware: LatticeBoltzmannHardwareTarget,
        /,
        *,
        physics_model: str,
        boundary_model: str,
        relaxation_rate_limits: Sequence[float],
        maximum_mach_number: float,
        maximum_knudsen_number: float,
        density_limits: Sequence[float],
        maximum_density_ratio: float,
        maximum_force_number: float,
        minimum_interface_width_cells: float = 0.0,
        minimum_wall_resolution_cells: float = 0.0,
        maximum_viscosity_ratio: float = 1.0,
        maximum_cahn_number: float = 0.0,
        maximum_capillary_number: float = 0.0,
        maximum_relative_mass_drift: float = 0.0,
        maximum_spurious_current_ratio: float = 0.0,
    ):
        if not isinstance(lattice, LatticeBoltzmannVelocitySet):
            raise TypeError("lattice must be a LatticeBoltzmannVelocitySet.")
        if not isinstance(collision, _COLLISION_TYPES):
            raise TypeError("collision must be a supported LBM collision plan.")
        if forcing is not None and not isinstance(forcing, GuoForcingPlan):
            raise TypeError("forcing must be GuoForcingPlan or None.")
        if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
            raise TypeError("precision must be a LatticeBoltzmannPrecisionPolicy.")
        if not isinstance(hardware, LatticeBoltzmannHardwareTarget):
            raise TypeError("hardware must be a LatticeBoltzmannHardwareTarget.")
        if not lattice.supports(collision.family):
            raise ValueError(
                f"Lattice {lattice.name!r} is not certified for collision {collision.family!r}."
            )
        if forcing is not None:
            if not lattice.supports("guo-forcing") or not forcing.supports(
                collision.family
            ):
                raise ValueError(
                    "The exact lattice/collision tuple does not support Guo forcing."
                )
        rates = tuple(float(value) for value in relaxation_rate_limits)
        if len(rates) != 2:
            raise ValueError("relaxation_rate_limits must contain exactly two bounds.")
        minimum_rate, maximum_rate = rates
        if (
            not np.isfinite(minimum_rate)
            or not np.isfinite(maximum_rate)
            or minimum_rate <= 0.0
            or minimum_rate >= maximum_rate
            or maximum_rate >= 2.0
        ):
            raise ValueError(
                "Relaxation-rate limits must form a finite interval in (0, 2)."
            )
        density_bounds = tuple(float(value) for value in density_limits)
        if len(density_bounds) != 2:
            raise ValueError("density_limits must contain exactly two bounds.")
        minimum_density, maximum_density = density_bounds
        if (
            not np.isfinite(minimum_density)
            or not np.isfinite(maximum_density)
            or minimum_density <= 0.0
            or minimum_density >= maximum_density
        ):
            raise ValueError("Density limits must form a finite positive interval.")
        mach = _positive(maximum_mach_number, "maximum_mach_number")
        knudsen = _positive(maximum_knudsen_number, "maximum_knudsen_number")
        density = _positive(maximum_density_ratio, "maximum_density_ratio")
        force = _nonnegative(maximum_force_number, "maximum_force_number")
        interface = _nonnegative(
            minimum_interface_width_cells, "minimum_interface_width_cells"
        )
        wall = _nonnegative(
            minimum_wall_resolution_cells, "minimum_wall_resolution_cells"
        )
        viscosity = _positive(maximum_viscosity_ratio, "maximum_viscosity_ratio")
        cahn = _nonnegative(maximum_cahn_number, "maximum_cahn_number")
        capillary = _nonnegative(maximum_capillary_number, "maximum_capillary_number")
        mass = _nonnegative(maximum_relative_mass_drift, "maximum_relative_mass_drift")
        spurious = _nonnegative(
            maximum_spurious_current_ratio, "maximum_spurious_current_ratio"
        )
        if mach >= 1.0:
            raise ValueError("maximum_mach_number must be smaller than one.")
        if density < 1.0 or viscosity < 1.0:
            raise ValueError("Density and viscosity ratio limits must be at least one.")
        if cahn >= 1.0:
            raise ValueError("maximum_cahn_number must be smaller than one.")
        physics = _identifier(physics_model, "physics_model")
        boundary = _identifier(boundary_model, "boundary_model")
        self.lattice = lattice
        self.collision = collision
        self.forcing = forcing
        self.precision = precision
        self.hardware = hardware
        self.physics_model = physics
        self.boundary_model = boundary
        self.minimum_relaxation_rate = minimum_rate
        self.maximum_relaxation_rate = maximum_rate
        self.maximum_mach_number = mach
        self.maximum_knudsen_number = knudsen
        self.minimum_density = minimum_density
        self.maximum_density = maximum_density
        self.maximum_density_ratio = density
        self.maximum_force_number = force
        self.minimum_interface_width_cells = interface
        self.minimum_wall_resolution_cells = wall
        self.maximum_viscosity_ratio = viscosity
        self.maximum_cahn_number = cahn
        self.maximum_capillary_number = capillary
        self.maximum_relative_mass_drift = mass
        self.maximum_spurious_current_ratio = spurious
        self.envelope_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-operating-envelope",
                "lattice": lattice.lattice_id,
                "collision": collision.collision_id,
                "forcing": None if forcing is None else forcing.forcing_id,
                "physics_model": physics,
                "boundary_model": boundary,
                "precision": precision.policy_id,
                "hardware": hardware.hardware_id,
                "limits": {
                    "relaxation_rate": rates,
                    "mach_number": mach,
                    "knudsen_number": knudsen,
                    "density": density_bounds,
                    "density_ratio": density,
                    "force_number": force,
                    "interface_width_cells": interface,
                    "wall_resolution_cells": wall,
                    "viscosity_ratio": viscosity,
                    "cahn_number": cahn,
                    "capillary_number": capillary,
                    "relative_mass_drift": mass,
                    "spurious_current_ratio": spurious,
                },
            }
        )

    @property
    def collision_model(self) -> str:
        return self.collision.family

    @property
    def forcing_model(self) -> str:
        return "none" if self.forcing is None else "guo"

    @property
    def support_coordinates(self) -> tuple[tuple[str, str | int | bool], ...]:
        return (
            ("boundary", self.boundary_model),
            ("collision", self.collision_model),
            ("forcing", self.forcing_model),
            ("hardware", self.hardware.hardware_id),
            ("host_count", self.hardware.host_count),
            ("lattice", self.lattice.name),
            ("physics", self.physics_model),
            ("precision", self.precision.policy_id),
        )

    def evaluate(
        self, point: LatticeBoltzmannOperatingPoint, /
    ) -> LatticeBoltzmannEnvelopeAdmission:
        if not isinstance(point, LatticeBoltzmannOperatingPoint):
            raise TypeError("point must be a LatticeBoltzmannOperatingPoint.")
        values = jnp.stack(
            (
                point.mach_number,
                point.knudsen_number,
                point.relaxation_rate,
                point.minimum_density,
                point.maximum_density,
                point.force_number,
                point.interface_width_cells,
                point.wall_resolution_cells,
                point.viscosity_ratio,
                point.cahn_number,
                point.capillary_number,
                point.relative_mass_drift,
                point.spurious_current_ratio,
            )
        )
        finite = jnp.all(jnp.isfinite(values))
        density_ratio = point.maximum_density / point.minimum_density
        margins = jnp.stack(
            (
                jnp.where(finite, 0.0, -jnp.inf),
                self.maximum_mach_number - point.mach_number,
                self.maximum_knudsen_number - point.knudsen_number,
                jnp.minimum(
                    point.relaxation_rate - self.minimum_relaxation_rate,
                    self.maximum_relaxation_rate - point.relaxation_rate,
                ),
                jnp.minimum(
                    point.minimum_density - self.minimum_density,
                    self.maximum_density - point.maximum_density,
                ),
                self.maximum_density_ratio - density_ratio,
                self.maximum_force_number - jnp.abs(point.force_number),
                point.interface_width_cells - self.minimum_interface_width_cells,
                point.wall_resolution_cells - self.minimum_wall_resolution_cells,
                self.maximum_viscosity_ratio - point.viscosity_ratio,
                self.maximum_cahn_number - point.cahn_number,
                self.maximum_capillary_number - point.capillary_number,
                self.maximum_relative_mass_drift - jnp.abs(point.relative_mass_drift),
                self.maximum_spurious_current_ratio
                - jnp.abs(point.spurious_current_ratio),
            )
        )
        checks = jnp.stack(
            (
                finite,
                (point.mach_number >= 0.0)
                & (point.mach_number <= self.maximum_mach_number),
                (point.knudsen_number >= 0.0)
                & (point.knudsen_number <= self.maximum_knudsen_number),
                (point.relaxation_rate >= self.minimum_relaxation_rate)
                & (point.relaxation_rate <= self.maximum_relaxation_rate),
                (point.minimum_density >= self.minimum_density)
                & (point.maximum_density <= self.maximum_density)
                & (point.maximum_density >= point.minimum_density),
                density_ratio <= self.maximum_density_ratio,
                (point.force_number >= 0.0)
                & (point.force_number <= self.maximum_force_number),
                point.interface_width_cells >= self.minimum_interface_width_cells,
                point.wall_resolution_cells >= self.minimum_wall_resolution_cells,
                (point.viscosity_ratio >= 1.0)
                & (point.viscosity_ratio <= self.maximum_viscosity_ratio),
                (point.cahn_number >= 0.0)
                & (point.cahn_number <= self.maximum_cahn_number),
                (point.capillary_number >= 0.0)
                & (point.capillary_number <= self.maximum_capillary_number),
                (point.relative_mass_drift >= 0.0)
                & (point.relative_mass_drift <= self.maximum_relative_mass_drift),
                (point.spurious_current_ratio >= 0.0)
                & (point.spurious_current_ratio <= self.maximum_spurious_current_ratio),
            )
        )
        checks = checks & finite
        return LatticeBoltzmannEnvelopeAdmission(checks, margins, self.envelope_id)

    def require(self, point: LatticeBoltzmannOperatingPoint, /) -> None:
        admission = self.evaluate(point)
        if not bool(np.asarray(admission.admitted)):
            failed = ", ".join(admission.failed_checks())
            raise LatticeBoltzmannEnvelopeError(
                f"LBM operating point is outside envelope {self.envelope_id}: {failed}."
            )

    def preflight(
        self,
        *,
        local_cell_count: int,
        population_field_count: int = 1,
        scalar_field_count: int = 0,
        temporary_population_field_count: int = 1,
        halo_cell_count: int = 0,
        checkpoint_copies: int = 1,
        output_copies: int = 0,
    ) -> LatticeBoltzmannResourceEstimate:
        counts = tuple(
            int(value)
            for value in (
                local_cell_count,
                population_field_count,
                scalar_field_count,
                temporary_population_field_count,
                halo_cell_count,
                checkpoint_copies,
                output_copies,
            )
        )
        if counts[0] <= 0 or counts[1] <= 0 or any(value < 0 for value in counts[2:]):
            raise ValueError("LBM resource preflight counts are invalid.")
        (
            cells,
            population_fields,
            scalar_fields,
            temporaries,
            halo,
            checkpoints,
            outputs,
        ) = counts
        q = self.lattice.population_count
        assumptions = self.precision.resource_assumptions
        storage_itemsize = assumptions.itemsize("storage")
        checkpoint_itemsize = assumptions.itemsize("checkpoint")
        output_itemsize = assumptions.itemsize("output")
        state_scalars = cells * (population_fields * q + scalar_fields)
        return LatticeBoltzmannResourceEstimate(
            state_bytes=state_scalars * storage_itemsize,
            temporary_bytes=cells * temporaries * q * storage_itemsize,
            halo_bytes=halo * population_fields * q * storage_itemsize,
            checkpoint_bytes=state_scalars * checkpoints * checkpoint_itemsize,
            output_bytes=state_scalars * outputs * output_itemsize,
            maximum_device_bytes=self.hardware.maximum_device_bytes,
            precision_resource_assumptions_id=assumptions.assumptions_id,
        )

    def prepare(
        self, **resource_counts: int
    ) -> "PreparedLatticeBoltzmannOperatingEnvelope":
        estimate = self.preflight(**resource_counts)
        if not estimate.fits_budget:
            raise LatticeBoltzmannEnvelopeError(
                f"LBM allocation needs {estimate.total_bytes} bytes per device, exceeding "
                f"the bound {estimate.maximum_device_bytes}."
            )
        return PreparedLatticeBoltzmannOperatingEnvelope(self, estimate)


class PreparedLatticeBoltzmannOperatingEnvelope(StrictModule, NonTrainableState):
    """Resource-qualified envelope ready to execute point admission."""

    plan: LatticeBoltzmannOperatingEnvelopePlan
    resources: LatticeBoltzmannResourceEstimate
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LatticeBoltzmannOperatingEnvelopePlan,
        resources: LatticeBoltzmannResourceEstimate,
        /,
    ):
        if not isinstance(plan, LatticeBoltzmannOperatingEnvelopePlan):
            raise TypeError("plan must be a LatticeBoltzmannOperatingEnvelopePlan.")
        if not isinstance(resources, LatticeBoltzmannResourceEstimate):
            raise TypeError("resources must be LatticeBoltzmannResourceEstimate.")
        if not resources.fits_budget:
            raise LatticeBoltzmannEnvelopeError(
                "Prepared LBM resources exceed the hardware budget."
            )
        self.plan = plan
        self.resources = resources
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lattice-boltzmann-operating-envelope",
                "envelope": plan.envelope_id,
                "resources": resources.estimate_id,
            }
        )

    def execute(
        self, point: LatticeBoltzmannOperatingPoint, /
    ) -> LatticeBoltzmannEnvelopeAdmission:
        return self.plan.evaluate(point)

    def require(self, point: LatticeBoltzmannOperatingPoint, /) -> None:
        self.plan.require(point)


__all__ = [
    "LatticeBoltzmannEnvelopeAdmission",
    "LatticeBoltzmannEnvelopeError",
    "LatticeBoltzmannHardwareTarget",
    "LatticeBoltzmannOperatingEnvelopePlan",
    "LatticeBoltzmannOperatingPoint",
    "LatticeBoltzmannResourceEstimate",
    "PreparedLatticeBoltzmannOperatingEnvelope",
]
