#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Per-site helium kinetics of UO2 fuel.

Helium follows Cognini's coupled diffusion-trapping equation with a Henry
surface condition (``HeliumGasModel``) in shells of the equivalent-sphere grain.
"""

from __future__ import annotations

import math
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field
from ...linalg import SmallLinearSolvePlan, solve_small_linear, solve_tridiagonal_lines
from ...materials import PropertyValidityRange, TabulatedProperty
from ...nonlinear._domain_cond import domain_cond
from ...typing import (
    Bool,
    checked,
    ConvertibleToArray,
    Dim,
    Float64,
    Int32,
)
from ._constants import (
    BOLTZMANN_J_PER_K,
    JOULE_PER_ELECTRON_VOLT,
)
from ._gas_math import (
    broadcast_to_sites,
    carnahan_starling_numerator,
    host_parameters,
    NONNEGATIVITY_ROUNDOFF,
    positive_grain_radius,
    require_site_shape,
)
from ._gas_state import SiteDims


# Bit OR-ed into the helium reaction-solve status when the shell-interface
# tridiagonal solve fails; the reaction solve itself emits 0, 1 or 2.
_HELIUM_INTERFACE_FAILURE_STATUS = 4


@final
class HeliumTransportTable(StrictModule):
    """Measured helium diffusivity and Henry constant tables with their source.

    Diffusivity [m^2/s] and Henry constant [atoms/m^3/Pa] are tabulated in
    temperature [K]; ``validity`` must lie inside the table.
    """

    diffusivity_table: TabulatedProperty
    henry_table: TabulatedProperty
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        temperature: ConvertibleToArray,
        diffusivity: ConvertibleToArray,
        henry_constant: ConvertibleToArray,
        validity: PropertyValidityRange,
        source_id: str,
    ) -> None:
        temperatures = np.asarray(temperature, dtype=np.float64)
        diffusivities = np.asarray(diffusivity, dtype=np.float64)
        henry_constants = np.asarray(henry_constant, dtype=np.float64)
        if (
            temperatures.ndim != 1
            or temperatures.size < 2
            or diffusivities.shape != temperatures.shape
            or henry_constants.shape != temperatures.shape
            or not np.all(
                np.isfinite(temperatures)
                & np.isfinite(diffusivities)
                & np.isfinite(henry_constants)
            )
            or not np.all(temperatures > 0.0)
            or not np.all(np.diff(temperatures) > 0.0)
            or not np.all(diffusivities >= 0.0)
            or not np.all(henry_constants >= 0.0)
        ):
            raise ValueError(
                "Helium tables require ordered positive temperatures and finite nonnegative SI properties."
            )
        if not isinstance(validity, PropertyValidityRange):
            raise TypeError("validity must be a PropertyValidityRange.")
        if validity.variables != ("temperature",):
            raise ValueError("Helium transport validity must declare temperature only.")
        if (
            validity.lower[0] < temperatures[0]
            or validity.upper[0] > temperatures[-1]
            or validity.units != ("K",)
        ):
            raise ValueError(
                "Helium validity must lie within its measured temperature table in K."
            )
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Helium transport data require source_id.")
        self.diffusivity_table = TabulatedProperty(temperatures, diffusivities, "m^2/s")
        self.henry_table = TabulatedProperty(
            temperatures, henry_constants, "atoms/m^3/Pa"
        )
        self.validity = validity
        self.source_id = source_id
        self.law_id = canonical_fingerprint(
            {
                "kind": "helium-transport-table",
                "temperature": temperatures.tolist(),
                "diffusivity": diffusivities.tolist(),
                "henry_constant": henry_constants.tolist(),
                "source_id": source_id,
                "validity": validity.range_id,
            }
        )


class HeliumRadialCellDim(Dim):
    """Reference-volume shell averages of one equivalent-sphere helium grain."""


class HeliumRadialFaceDim(Dim):
    """Ordered shell traces, including the regular center and Henry surface."""


class HeliumRadialSampleDim(Dim):
    """Center, shell midpoints and surface of the reconstructed physical field."""


# Exact Maclaurin coefficients in ``eta^2``, highest power first for
# ``jnp.polyval``: the ``sinh(eta) / eta`` denominator and the numerators of
# the shell weights of ``_HeliumShellFactors``. For ``eta`` below
# ``_HELIUM_SERIES_ETA_LIMIT`` the truncation is below double roundoff.
_HELIUM_SERIES_DEGREE = 12
_HELIUM_SERIES_ETA_LIMIT = 1.0
_HELIUM_SINHC_SERIES = tuple(
    1.0 / math.factorial(2 * order + 1) for order in range(_HELIUM_SERIES_DEGREE, -1, -1)
)
_HELIUM_SAME_FACE_SERIES = tuple(
    2.0 * (order + 1) / math.factorial(2 * order + 3)
    for order in range(_HELIUM_SERIES_DEGREE, -1, -1)
)
_HELIUM_OPPOSITE_FACE_SERIES = tuple(
    1.0 / math.factorial(2 * order + 3) for order in range(_HELIUM_SERIES_DEGREE, -1, -1)
)
_HELIUM_SAME_FACE_DEFICIT_SERIES = tuple(
    4.0 * order * (order + 1) / (3.0 * math.factorial(2 * order + 3))
    for order in range(_HELIUM_SERIES_DEGREE, -1, -1)
)
_HELIUM_OPPOSITE_FACE_DEFICIT_SERIES = tuple(
    order * (2 * order + 5) / (3.0 * math.factorial(2 * order + 3))
    for order in range(_HELIUM_SERIES_DEGREE, -1, -1)
)


class _HeliumShellFactors(NamedTuple):
    """Positive spherical-shell Helmholtz weights at ``eta = kappa (r_u - r_l)``.

    ``same_face = eta coth(eta) - 1`` couples a face to itself and
    ``opposite_face = 1 - eta csch(eta)`` to the other face of the shell; the
    ``_over_square`` weights divide them by ``eta^2`` and the deficits are
    ``1/3`` and ``1/6`` minus those.
    """

    same_face: Array
    opposite_face: Array
    same_face_over_square: Array
    opposite_face_over_square: Array
    same_face_deficit: Array
    opposite_face_deficit: Array
    eta_csch: Array


def _helium_shell_factors(eta: Array) -> _HeliumShellFactors:
    """Shell weights, stable at both zero and infinite ``eta``."""

    def small(_: None) -> _HeliumShellFactors:
        square = eta**2
        denominator = jnp.polyval(jnp.asarray(_HELIUM_SINHC_SERIES), square)
        same_face_over_square = (
            jnp.polyval(jnp.asarray(_HELIUM_SAME_FACE_SERIES), square) / denominator
        )
        opposite_face_over_square = (
            jnp.polyval(jnp.asarray(_HELIUM_OPPOSITE_FACE_SERIES), square) / denominator
        )
        return _HeliumShellFactors(
            same_face=square * same_face_over_square,
            opposite_face=square * opposite_face_over_square,
            same_face_over_square=same_face_over_square,
            opposite_face_over_square=opposite_face_over_square,
            same_face_deficit=jnp.polyval(
                jnp.asarray(_HELIUM_SAME_FACE_DEFICIT_SERIES), square
            )
            / denominator,
            opposite_face_deficit=jnp.polyval(
                jnp.asarray(_HELIUM_OPPOSITE_FACE_DEFICIT_SERIES), square
            )
            / denominator,
            eta_csch=1.0 / denominator,
        )

    def large(_: None) -> _HeliumShellFactors:
        decay = jnp.exp(-eta)
        denominator = -jnp.expm1(-2.0 * eta)
        coth = (1.0 + decay**2) / denominator
        eta_csch = 2.0 * eta * decay / denominator
        inverse = 1.0 / eta
        same_face_over_square = inverse * coth - inverse**2
        opposite_face_over_square = inverse**2 * (1.0 - eta_csch)
        return _HeliumShellFactors(
            same_face=eta * coth - 1.0,
            opposite_face=1.0 - eta_csch,
            same_face_over_square=same_face_over_square,
            opposite_face_over_square=opposite_face_over_square,
            same_face_deficit=1.0 / 3.0 - same_face_over_square,
            opposite_face_deficit=1.0 / 6.0 - opposite_face_over_square,
            eta_csch=eta_csch,
        )

    return domain_cond(jnp.max(eta) < _HELIUM_SERIES_ETA_LIMIT, small, large)


def _helium_hard_sphere_diameter(temperature: Array, /) -> Array:
    """Helium hard-sphere diameter [m] at temperature [K]."""

    return 2.973e-10 * (0.8414 - 0.05 * jnp.log(temperature / 10.985))


@final
class HeliumGasSiteState(StrictModule):
    """Positive shell means per reference fuel volume; exported gas is signed."""

    __strict_contract__ = True
    dissolved_cells: Float64[SiteDims, HeliumRadialCellDim]
    trapped_cells: Float64[SiteDims, HeliumRadialCellDim]
    grain_radius: Float64[SiteDims]
    produced: Float64[SiteDims]
    net_released: Float64[SiteDims]


@final
class HeliumGasStepResult(StrictModule):
    """Helium candidate state, signed transfers, sampled fields and solve status."""

    __strict_contract__ = True
    state: HeliumGasSiteState
    released_increment: Float64[SiteDims]
    diffusion_to_boundary: Float64[SiteDims]
    grain_growth_sweeping: Float64[SiteDims]
    retained_total: Float64[SiteDims]
    intragranular_swelling: Float64[SiteDims]
    radial_cell_width: Float64[SiteDims]
    dissolved_concentration: Float64[SiteDims, HeliumRadialSampleDim]
    trapped_concentration: Float64[SiteDims, HeliumRadialSampleDim]
    linear_solve_successful: Bool[SiteDims]
    linear_solve_status: Int32[SiteDims]
    linear_solve_condition_estimate: Float64[SiteDims]
    linear_solve_conditioning_available: Bool[SiteDims]
    linear_solve_residual_norm: Float64[SiteDims]
    admissible: Bool[SiteDims]
    in_range: Bool[SiteDims]


@final
class _HeliumShellUpdate(StrictModule):
    dissolved: Array
    trapped: Array
    dissolved_concentration: Array
    trapped_concentration: Array
    boundary_transfer: Array
    physical_support: Array
    successful: Array
    status: Array
    condition_estimate: Array
    conditioning_available: Array
    residual_norm: Array


class _HeliumShellSolution(NamedTuple):
    dissolved: Array  # shell means [atoms/m^3]
    trapped: Array  # shell means [atoms/m^3]
    dissolved_concentration: Array  # center, shell midpoints, surface
    trapped_concentration: Array  # center, shell midpoints, surface
    boundary_transfer: Array  # signed outward release [atoms/m^3]
    physical_support: Array
    interface_successful: Array
    interface_residual: Array


class _HeliumCoefficients(NamedTuple):
    diffusivity: Array  # [m^2/s]
    surface_concentration: Array  # Henry trace [atoms/m^3]
    trapping_rate: Array  # [1/s]
    resolution_rate: Array  # thermal plus irradiation re-solution [1/s]
    packing: Array  # hard-sphere packing fraction of the bubble gas


@final
class HeliumGasModel(StrictModule):
    """Cognini's coupled helium PDE with an exact spherical-shell BE resolvent.

    Accepted shell means are a positive, conservative source projection.
    Each shell's modified Helmholtz equation is solved analytically, with
    continuous face fluxes and the exact Henry Dirichlet surface trace.
    Uniform source/initial data recover the full-sphere resolvent, including
    an arbitrarily thin absorption layer, at any shell count.
    """

    __strict_contract__ = True
    diffusivity_prefactor: Array
    diffusivity_activation_energy: Array
    henry_prefactor: Array
    henry_activation_energy: Array
    bubble_atomic_volume: Array
    fission_fragment_range: Array
    fission_fragment_influence_radius: Array
    validity: PropertyValidityRange
    transport: HeliumTransportTable | None
    reaction_solve: SmallLinearSolvePlan
    cell_faces: Float64[HeliumRadialFaceDim] = fixed_field()
    cell_volume_weights: Float64[HeliumRadialCellDim] = fixed_field()
    radial_cell_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        /,
        *,
        radial_cell_count: int = 40,
        diffusivity_prefactor: ArrayLike = 3.3e-10,
        diffusivity_activation_energy: ArrayLike = 1.64 * JOULE_PER_ELECTRON_VOLT,
        henry_prefactor: ArrayLike = 4.1e18,
        henry_activation_energy: ArrayLike = 0.65 * JOULE_PER_ELECTRON_VOLT,
        bubble_atomic_volume: ArrayLike = 7.8e-30,
        fission_fragment_range: ArrayLike = 6.0e-6,
        fission_fragment_influence_radius: ArrayLike = 1.0e-9,
        transport: HeliumTransportTable | None = None,
    ) -> None:
        if isinstance(radial_cell_count, bool) or not isinstance(radial_cell_count, int):
            raise TypeError("radial_cell_count must be an int.")
        if radial_cell_count < 1:
            raise ValueError("radial_cell_count must be positive.")
        parameters = host_parameters(
            {
                "diffusivity_prefactor": diffusivity_prefactor,
                "diffusivity_activation_energy": diffusivity_activation_energy,
                "henry_prefactor": henry_prefactor,
                "henry_activation_energy": henry_activation_energy,
                "bubble_atomic_volume": bubble_atomic_volume,
                "fission_fragment_range": fission_fragment_range,
                "fission_fragment_influence_radius": fission_fragment_influence_radius,
            },
            positive=("bubble_atomic_volume",),
        )
        self.diffusivity_prefactor = jnp.asarray(parameters["diffusivity_prefactor"])
        self.diffusivity_activation_energy = jnp.asarray(
            parameters["diffusivity_activation_energy"]
        )
        self.henry_prefactor = jnp.asarray(parameters["henry_prefactor"])
        self.henry_activation_energy = jnp.asarray(parameters["henry_activation_energy"])
        self.bubble_atomic_volume = jnp.asarray(parameters["bubble_atomic_volume"])
        self.fission_fragment_range = jnp.asarray(parameters["fission_fragment_range"])
        self.fission_fragment_influence_radius = jnp.asarray(
            parameters["fission_fragment_influence_radius"]
        )
        self.transport = transport
        self.radial_cell_count = radial_cell_count
        faces = np.linspace(0.0, 1.0, radial_cell_count + 1, dtype=np.float64)
        self.cell_faces = jnp.asarray(faces)
        self.cell_volume_weights = jnp.asarray(np.diff(faces**3))
        if transport is None:
            literature_source_id = (
                "cognini-2021-net-53-562:eq2-11;luzzi-2018-ned-330-265:eq2"
            )
            self.source_id = literature_source_id
            self.validity = PropertyValidityRange(
                {"temperature": (1073.0, 1773.0, "K")}, source_id=literature_source_id
            )
        else:
            self.source_id = transport.source_id
            self.validity = transport.validity
        self.reaction_solve = SmallLinearSolvePlan(2)
        self.law_id = canonical_fingerprint(
            {
                "kind": "helium-transport",
                "spatial_closure": "exact-shell-spherical-helmholtz",
                "parameters": parameters,
                "radial_cell_count": radial_cell_count,
                "cell_faces": faces.tolist(),
                "transport": None if transport is None else transport.law_id,
                "reaction_solve": self.reaction_solve.plan_id,
            }
        )

    @checked
    def initial_state(
        self, shape: tuple[int, ...], /, *, grain_radius: ArrayLike
    ) -> HeliumGasSiteState:
        require_site_shape(shape)
        radius = positive_grain_radius(grain_radius)
        zeros = jnp.zeros(shape, dtype=jnp.float64)
        cells = jnp.zeros((*shape, self.radial_cell_count), dtype=jnp.float64)
        return HeliumGasSiteState(
            dissolved_cells=cells,
            trapped_cells=cells,
            grain_radius=jnp.broadcast_to(jnp.asarray(radius), shape),
            produced=zeros,
            net_released=zeros,
        )

    def diffusivity(self, temperature: ArrayLike, /) -> Array:
        """Helium diffusivity [m^2/s] at temperature [K]."""

        if self.transport is not None:
            return self.transport.diffusivity_table.evaluate(temperature)
        return self.diffusivity_prefactor * jnp.exp(
            -self.diffusivity_activation_energy
            / (BOLTZMANN_J_PER_K * jnp.asarray(temperature))
        )

    def henry_constant(self, temperature: ArrayLike, /) -> Array:
        """Henry constant [atoms/m^3/Pa] at temperature [K]."""

        if self.transport is not None:
            return self.transport.henry_table.evaluate(temperature)
        return self.henry_prefactor * jnp.exp(
            -self.henry_activation_energy / (BOLTZMANN_J_PER_K * jnp.asarray(temperature))
        )

    def _reaction_coefficients(
        self,
        temperature: Array,
        fission_rate_density: Array,
        bubble_radius: Array,
        bubble_number_density: Array,
        boundary_pressure: Array,
    ) -> _HeliumCoefficients:
        """Transport, trapping and re-solution coefficients at the endpoint inputs.

        Trapping is ``4 pi D R_b N_b``; re-solution is the thermal
        ``3 D H k T Z / R_b^2`` with the Carnahan-Starling bubble gas ``Z``
        plus the irradiation ``2 pi mu_f F (R_b + Z_0)^2``.
        """

        diffusivity = self.diffusivity(temperature)
        henry = self.henry_constant(temperature)
        packing = (
            jnp.pi
            * _helium_hard_sphere_diameter(temperature) ** 3
            / (6.0 * self.bubble_atomic_volume)
        )
        compressibility = carnahan_starling_numerator(packing) / (1.0 - packing) ** 3
        occupied = bubble_radius > 0.0
        thermal_resolution = jnp.where(
            occupied,
            3.0
            * diffusivity
            * henry
            * BOLTZMANN_J_PER_K
            * temperature
            * compressibility
            / jnp.where(occupied, bubble_radius**2, 1.0),
            0.0,
        )
        return _HeliumCoefficients(
            diffusivity=diffusivity,
            surface_concentration=henry * boundary_pressure,
            trapping_rate=4.0
            * jnp.pi
            * diffusivity
            * bubble_radius
            * bubble_number_density,
            resolution_rate=thermal_resolution
            + 2.0
            * jnp.pi
            * self.fission_fragment_range
            * fission_rate_density
            * (bubble_radius + self.fission_fragment_influence_radius) ** 2,
            packing=packing,
        )

    def _interface_traces(
        self,
        particular: Array,
        surface_concentration: Array,
        transmission: Array,
        source_lower: Array,
        source_upper: Array,
        grain_radius: Array,
    ) -> tuple[Array, Array, Array]:
        """Interior shell-face traces from flux continuity, with solve status."""

        if self.radial_cell_count > 1:
            diagonal = (
                transmission[:-1]
                + transmission[1:]
                + source_upper[:-1]
                + source_lower[1:]
            )
            below = jnp.concatenate(
                [jnp.zeros((1,), dtype=grain_radius.dtype), -transmission[1:-1]]
            )
            above = jnp.concatenate(
                [-transmission[1:-1], jnp.zeros((1,), dtype=grain_radius.dtype)]
            )
            right = (
                source_upper[:-1] * particular[:-1] + source_lower[1:] * particular[1:]
            )
            right = right.at[-1].add(transmission[-1] * surface_concentration)
            interfaces = solve_tridiagonal_lines(below, diagonal, above, right)
            return interfaces.value, interfaces.successful, interfaces.residual_norm
        return (
            jnp.empty((0,), dtype=grain_radius.dtype),
            jnp.asarray(True, dtype=jnp.bool_),
            jnp.zeros_like(grain_radius),
        )

    def _diffusive_shells(
        self,
        particular: Array,
        previous_trapped: Array,
        reaction_factor: Array,
        diffusion_dt: Array,
        dt_trapping: Array,
        dt_resolution: Array,
        surface_concentration: Array,
        grain_radius: Array,
    ) -> _HeliumShellSolution:
        """Exact shell resolvent about the uniform reaction solution ``particular``.

        ``reaction_factor = 1 + dt g / (1 + dt b)`` and ``diffusion_dt = D dt``
        [m^2] set the shell Helmholtz number ``eta``.
        """

        lower, upper = self.cell_faces[:-1], self.cell_faces[1:]
        width = upper - lower
        volume = self.cell_volume_weights / 3.0
        kappa = grain_radius * jnp.sqrt(reaction_factor) / jnp.sqrt(diffusion_dt)
        eta = kappa * width
        factors = _helium_shell_factors(eta)
        transmission = lower * upper / width * factors.eta_csch
        source_lower = (
            lower**2 / width * factors.same_face
            + lower * upper / width * factors.opposite_face
        )
        source_upper = (
            upper**2 / width * factors.same_face
            + lower * upper / width * factors.opposite_face
        )
        interior, interface_successful, interface_residual = self._interface_traces(
            particular,
            surface_concentration,
            transmission,
            source_lower,
            source_upper,
            grain_radius,
        )
        traces = jnp.concatenate([particular[:1], interior, surface_concentration[None]])
        left_weight = (
            lower
            * width
            / volume
            * (
                lower * factors.same_face_over_square
                + upper * factors.opposite_face_over_square
            )
        )
        right_weight = (
            upper
            * width
            / volume
            * (
                upper * factors.same_face_over_square
                + lower * factors.opposite_face_over_square
            )
        )
        source_weight = (
            width
            / volume
            * (
                (lower**2 + upper**2) * factors.same_face_deficit
                + 2.0 * lower * upper * factors.opposite_face_deficit
            )
        )
        dissolved = (
            source_weight * particular
            + left_weight * traces[:-1]
            + right_weight * traces[1:]
        )
        half_decay = jnp.exp(-0.5 * eta)
        face_weight = 2.0 * half_decay / (1.0 + half_decay**2)
        center_source_weight = jnp.expm1(-0.5 * eta) ** 2 / (1.0 + half_decay**2)
        midpoint = center_source_weight * particular + face_weight * (
            lower * traces[:-1] + upper * traces[1:]
        ) / (lower + upper)
        origin = (
            factors.opposite_face[0] * particular[0] + factors.eta_csch[0] * traces[1]
        )
        dissolved_samples = jnp.concatenate(
            [origin[None], midpoint, surface_concentration[None]]
        )
        trapped = (previous_trapped + dt_trapping * dissolved) / (1.0 + dt_resolution)
        trapped_samples = (
            jnp.concatenate(
                [previous_trapped[:1], previous_trapped, previous_trapped[-1:]]
            )
            + dt_trapping * dissolved_samples
        ) / (1.0 + dt_resolution)
        # Integrated outward flux, without large boundary-lifting cancellation.
        transfer = 3.0 * (
            diffusion_dt
            / grain_radius**2
            * transmission[-1]
            * (traces[-2] - surface_concentration)
            + reaction_factor
            * volume[-1]
            * right_weight[-1]
            * (particular[-1] - surface_concentration)
        )
        return _HeliumShellSolution(
            dissolved=dissolved,
            trapped=trapped,
            dissolved_concentration=dissolved_samples,
            trapped_concentration=trapped_samples,
            boundary_transfer=transfer,
            physical_support=jnp.all(particular >= 0.0)
            & jnp.all(traces >= 0.0)
            & jnp.all(source_weight >= 0.0)
            & jnp.all(left_weight >= 0.0)
            & jnp.all(right_weight >= 0.0),
            interface_successful=interface_successful,
            interface_residual=interface_residual,
        )

    def _site_update(
        self,
        dissolved: Array,
        trapped: Array,
        production_rate: Array,
        dt: Array,
        diffusivity: Array,
        trapping_rate: Array,
        resolution_rate: Array,
        surface_concentration: Array,
        grain_radius: Array,
    ) -> _HeliumShellUpdate:
        dt_trapping, dt_resolution = dt * trapping_rate, dt * resolution_rate
        matrix = jnp.stack(
            [
                jnp.stack([1.0 + dt_trapping, -dt_resolution]),
                jnp.stack([-dt_trapping, 1.0 + dt_resolution]),
            ]
        )
        reaction = solve_small_linear(
            self.reaction_solve,
            matrix,
            jnp.stack([dissolved + production_rate * dt, trapped]),
        )
        particular = reaction.value[0]
        reaction_factor = 1.0 + dt_trapping / (1.0 + dt_resolution)
        diffusion_dt = dt * diffusivity

        def diffusion(_: None) -> _HeliumShellUpdate:
            shells = self._diffusive_shells(
                particular,
                trapped,
                reaction_factor,
                diffusion_dt,
                dt_trapping,
                dt_resolution,
                surface_concentration,
                grain_radius,
            )
            interface_status = jnp.where(
                shells.interface_successful, 0, _HELIUM_INTERFACE_FAILURE_STATUS
            ).astype(jnp.int32)
            return _HeliumShellUpdate(
                dissolved=shells.dissolved,
                trapped=shells.trapped,
                dissolved_concentration=shells.dissolved_concentration,
                trapped_concentration=shells.trapped_concentration,
                boundary_transfer=shells.boundary_transfer,
                physical_support=shells.physical_support,
                successful=reaction.successful & shells.interface_successful,
                status=jnp.bitwise_or(reaction.status, interface_status),
                # The shell-interface tridiagonal solve has no condition estimate.
                condition_estimate=jnp.where(
                    self.radial_cell_count > 1,
                    jnp.full_like(grain_radius, jnp.nan),
                    reaction.condition_estimate,
                ),
                conditioning_available=jnp.asarray(
                    self.radial_cell_count == 1, dtype=jnp.bool_
                ),
                residual_norm=jnp.maximum(
                    reaction.residual_norm, shells.interface_residual
                ),
            )

        def immobile(_: None) -> _HeliumShellUpdate:
            dissolved_cells, trapped_cells = reaction.value
            surface_trapped = (trapped[-1] + dt_trapping * surface_concentration) / (
                1.0 + dt_resolution
            )
            return _HeliumShellUpdate(
                dissolved=dissolved_cells,
                trapped=trapped_cells,
                dissolved_concentration=jnp.concatenate(
                    [dissolved_cells[:1], dissolved_cells, surface_concentration[None]]
                ),
                trapped_concentration=jnp.concatenate(
                    [trapped_cells[:1], trapped_cells, surface_trapped[None]]
                ),
                boundary_transfer=jnp.zeros_like(grain_radius),
                physical_support=jnp.all(dissolved_cells >= 0.0)
                & jnp.all(trapped_cells >= 0.0)
                & (surface_concentration >= 0.0),
                successful=reaction.successful,
                status=reaction.status,
                condition_estimate=reaction.condition_estimate,
                conditioning_available=jnp.asarray(True, dtype=jnp.bool_),
                residual_norm=reaction.residual_norm,
            )

        return domain_cond(diffusion_dt > 0.0, diffusion, immobile)

    @checked
    def advance(
        self,
        state: HeliumGasSiteState,
        /,
        *,
        temperature: ArrayLike,
        fission_rate_density: ArrayLike,
        gas_production_rate: ArrayLike,
        time_step: ArrayLike,
        grain_radius: ArrayLike,
        bubble_radius: ArrayLike,
        bubble_number_density: ArrayLike,
        boundary_helium_pressure: ArrayLike,
    ) -> HeliumGasStepResult:
        if state.dissolved_cells.shape[-1] != self.radial_cell_count:
            raise ValueError("Helium state and model carry different radial cell counts.")
        shape = state.grain_radius.shape
        (
            temperature_,
            fission_rate_density_,
            production_rate,
            dt,
            radius,
            bubble_radius_,
            bubble_number_density_,
            pressure,
        ) = (
            broadcast_to_sites(value, shape)
            for value in (
                temperature,
                fission_rate_density,
                gas_production_rate,
                time_step,
                grain_radius,
                bubble_radius,
                bubble_number_density,
                boundary_helium_pressure,
            )
        )
        retained_start = (
            state.dissolved_cells + state.trapped_cells
        ) @ self.cell_volume_weights
        remap = (state.grain_radius / radius) ** 3
        swept = (1.0 - remap) * retained_start
        coefficients = self._reaction_coefficients(
            temperature_,
            fission_rate_density_,
            bubble_radius_,
            bubble_number_density_,
            pressure,
        )
        site_inputs = tuple(
            value.reshape((-1,))
            for value in (
                production_rate,
                dt,
                coefficients.diffusivity,
                coefficients.trapping_rate,
                coefficients.resolution_rate,
                coefficients.surface_concentration,
                radius,
            )
        )
        flat_update = jax.vmap(self._site_update)(
            (state.dissolved_cells * remap[..., None]).reshape(
                (-1, self.radial_cell_count)
            ),
            (state.trapped_cells * remap[..., None]).reshape(
                (-1, self.radial_cell_count)
            ),
            *site_inputs,
        )
        updated = jax.tree.map(
            lambda value: value.reshape((*shape, *value.shape[1:])), flat_update
        )
        retained = (updated.dissolved + updated.trapped) @ self.cell_volume_weights
        released = swept + updated.boundary_transfer
        endpoint = jnp.stack(
            [
                retained,
                temperature_,
                fission_rate_density_,
                production_rate,
                dt,
                radius,
                bubble_radius_,
                bubble_number_density_,
                pressure,
                coefficients.packing,
            ],
            axis=-1,
        )
        return HeliumGasStepResult(
            state=HeliumGasSiteState(
                dissolved_cells=updated.dissolved,
                trapped_cells=updated.trapped,
                grain_radius=radius,
                produced=state.produced + production_rate * dt,
                net_released=state.net_released + released,
            ),
            released_increment=released,
            diffusion_to_boundary=updated.boundary_transfer,
            grain_growth_sweeping=swept,
            retained_total=retained,
            intragranular_swelling=self.bubble_atomic_volume
            * (updated.trapped @ self.cell_volume_weights),
            radial_cell_width=radius / self.radial_cell_count,
            dissolved_concentration=updated.dissolved_concentration,
            trapped_concentration=updated.trapped_concentration,
            linear_solve_successful=updated.successful,
            linear_solve_status=updated.status,
            linear_solve_condition_estimate=updated.condition_estimate,
            linear_solve_conditioning_available=updated.conditioning_available,
            linear_solve_residual_norm=updated.residual_norm,
            admissible=_helium_candidate_admissible(
                updated,
                endpoint,
                previous_grain_radius=state.grain_radius,
                grain_radius=radius,
                temperature=temperature_,
                packing=coefficients.packing,
            ),
            in_range=self.validity.contains(temperature=temperature_),
        )


def _helium_candidate_admissible(
    updated: _HeliumShellUpdate,
    endpoint: Array,
    *,
    previous_grain_radius: Array,
    grain_radius: Array,
    temperature: Array,
    packing: Array,
) -> Array:
    """Finite nonnegative endpoint inputs and shells within roundoff, solves successful.

    ``endpoint`` stacks the retained helium and every endpoint input along its
    last axis.
    """

    tolerance = NONNEGATIVITY_ROUNDOFF * jnp.maximum(
        jnp.max(
            jnp.abs(updated.dissolved_concentration)
            + jnp.abs(updated.trapped_concentration),
            axis=-1,
        ),
        1.0,
    )
    return (
        jnp.all(jnp.isfinite(endpoint), axis=-1)
        & jnp.all(endpoint >= 0.0, axis=-1)
        & (grain_radius >= previous_grain_radius)
        & (grain_radius > 0.0)
        & (temperature > 0.0)
        & (packing < 1.0)
        & jnp.all(
            jnp.isfinite(updated.dissolved) & jnp.isfinite(updated.trapped), axis=-1
        )
        & jnp.all(updated.dissolved >= -tolerance[..., None], axis=-1)
        & jnp.all(updated.trapped >= -tolerance[..., None], axis=-1)
        & jnp.all(updated.dissolved_concentration >= -tolerance[..., None], axis=-1)
        & jnp.all(updated.trapped_concentration >= -tolerance[..., None], axis=-1)
        & updated.physical_support
        & updated.successful
    )


__all__ = [
    "HeliumGasModel",
    "HeliumGasSiteState",
    "HeliumGasStepResult",
    "HeliumRadialCellDim",
    "HeliumRadialFaceDim",
    "HeliumRadialSampleDim",
    "HeliumTransportTable",
]
