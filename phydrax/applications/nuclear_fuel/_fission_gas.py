#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Per-site Xe + Kr fission gas kinetics of UO2 fuel with a passive Kr/Xe ledger.

The nonlinear total-gas model lumps Xe and Kr into one species. It is fully
implicit in time with every coefficient evaluated at the end-of-step inputs:

* Intragranular diffusion in the Booth (AECL-496, 1957) equivalent sphere with
  the Speight (Nucl. Sci. Eng. 37 (1969) 180) effective diffusivity
  ``D b / (b + g)``: single-atom diffusivity of Turnbull et al. (J. Nucl. Mater.
  107 (1982) 168), trapping ``g = 4 pi D R N`` and re-solution
  ``b = 3.03 pi mu_f F (R + Z_0)^2`` with the White and Tucker (J. Nucl. Mater.
  118 (1983) 1) intragranular bubble radius and density. The trapped fraction
  ``g / (b + g)`` gives the intragranular bubble swelling; no explicit
  intragranular bubble population exists (Speight excludes one).
* Grain-face bubbles of Pastore et al. (Nucl. Eng. Des. 256 (2013) 75): Speight
  and Beere vacancy absorption, van der Waals gas, lenticular geometry, White
  coalescence and venting of the bubbles that exceed the saturation coverage;
  grain-face re-solution is neglected by this combination.
* Ainscough et al. (J. Nucl. Mater. 49 (1973) 117) grain growth with grain
  boundary sweeping of the swept volume fraction.
* Barani et al. (J. Nucl. Mater. 486 (2017) 96) micro-cracking burst release
  on temperature variations with burnup healing.
* Barani et al. (J. Nucl. Mater. 539 (2020) 152296) high-burnup-structure
  formation on the effective burnup with intragranular gas depletion into the
  HBS pores.
* Athermal recoil and knockout release (Lewis, J. Nucl. Mater. 148 (1987) 28).

Inventories and production are atoms per reference fuel volume (the fabricated
solid volume); integrate with reference quadrature weights, never current ones.
Every step closes ``retained(t + dt) - retained(t) = beta dt - released_increment``.

A passive Kr/Xe ledger (``FissionGasModel.advance_species``) traces the two
identities through an evaluated total-gas step without recomputing its
geometry or roots.

Modal representation. ``c(r) = sum_m c_m psi_m(r / a)`` with
``psi_m(x) = (-1)^(m+1) sin(m pi x) / (m pi x)``. Its center limit is
``(-1)^(m+1)`` and it vanishes on the perfect-sink boundary. The uniform source
projects to ``s_m = 2`` for every mode and the volume mean is
``sum_m 3 c_m / (m pi)^2``. For coefficients frozen at the end of the step the
update ``c_m' = c_m E_m + beta s_m (1 - E_m) / k_m``, ``k_m = D (m pi / a)^2``,
``E_m = exp(-k_m dt)`` is exact for every resolved mode. The unresolved uniform
source weight ``1 - Z_N``, ``Z_N = (6/pi^2) sum_{m<=N} m^-2``, is held in a
separate positive subgrid inventory decaying at ``k_{N+1}``. It is not released
artificially at zero diffusivity. Its retained inventory bounds the missing
Booth tail, hence bounds the release error of this slowest-unresolved-mode
closure; the bound is reported as ``modal_truncation_bound``.

Grain radius changes are an explicit conservative remap before diffusion: the
swept fraction ``1 - (a / a')^3`` of the intragranular gas moves to the grain
faces and the modal coefficients are scaled by ``(a / a')^3`` in the normalized
coordinate ``r / a``. Grain-face gas and bubble volume per unit fuel volume are
conserved by growth: the face area per unit volume falls by ``a / a'``, so
gas atoms and vacancies per bubble scale by ``a' / a`` at fixed bubbles per
face area.
"""

from __future__ import annotations

from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...materials import PropertyValidityRange
from ...nonlinear import LocalRootPlan
from ...nonlinear._domain_cond import domain_cond
from ._constants import BOLTZMANN_J_PER_K
from ._gas_math import (
    broadcast_to_sites,
    host_parameters,
    NONNEGATIVITY_ROUNDOFF,
    positive_grain_radius,
    positive_quadratic_root,
    require_site_shape,
)
from ._gas_state import (
    FISSION_GAS_TRACER_SPECIES,
    FissionGasDiagnostics,
    FissionGasSiteState,
    FissionGasSpeciesState,
    FissionGasSpeciesStepResult,
    FissionGasSpeciesTransfers,
    FissionGasStepResult,
    FissionGasTransfers,
)
from ._grain_boundary import (
    AinscoughGrainGrowth,
    GrainBoundaryMicrocracking,
    GrainFaceBubbles,
    speight_beere_sink_strength,
)
from ._hbs import (
    DIMER_ATOMS,
    HbsPoreUpdate,
    HighBurnupStructureFormation,
    HighBurnupStructurePores,
)
from ._intragranular import modal_constants, SpeightTrapping, TurnbullDiffusivity


# Floor of ``3 - 2F`` in White's coalescence invariant ``N / (3 - 2F)^3``; it
# only binds if roundoff pushes the coverage ``F`` past 1.5.
_COALESCENCE_DENOMINATOR_FLOOR = 1.0e-12
# Below this rate-time product ``z`` the four-term Taylor series of
# ``(1 - exp(-z)) / z`` is exact to double precision.
_DECAY_SERIES_THRESHOLD = 1.0e-4
# Acceptance threshold, relative to the magnitude of the checked quantity.
# Allowed mismatch between the Kr/Xe sums and the total-gas ledger; also the
# passive-species concentration undershoot.
_SPECIES_LEDGER_ROUNDOFF = 128.0 * float(np.finfo(np.float64).eps)


def _decay_series(rate_time: Array, /) -> tuple[Array, Array, Array]:
    """Small-``z`` mask, the guarded ``z`` and the series of ``1 - (1 - exp(-z)) / z``."""

    small = rate_time < _DECAY_SERIES_THRESHOLD
    safe = jnp.where(small, 1.0, rate_time)
    loss = rate_time * (
        0.5 - rate_time * (1.0 / 6.0 - rate_time * (1.0 / 24.0 - rate_time / 120.0))
    )
    return small, safe, loss


def _stable_decay_mean(rate_time: Array) -> Array:
    """``(1 - exp(-z)) / z`` for ``z >= 0``, equal to one at ``z = 0``."""

    small, safe, loss = _decay_series(rate_time)
    return jnp.where(small, 1.0 - loss, -jnp.expm1(-safe) / safe)


def _stable_decay_loss(rate_time: Array) -> Array:
    """``1 - (1 - exp(-z)) / z``, accurate without subtracting near unity."""

    small, safe, loss = _decay_series(rate_time)
    return jnp.where(small, loss, 1.0 + jnp.expm1(-safe) / safe)


def _ratio_where_positive(numerator: Array, denominator: Array, /) -> Array:
    """``numerator / denominator`` where the denominator is positive, else zero."""

    positive = denominator > 0.0
    return jnp.where(positive, numerator / jnp.where(positive, denominator, 1.0), 0.0)


@final
class AthermalRelease(StrictModule):
    """Recoil and knockout release from fuel surfaces (Lewis, 1987).

    INL/CON-13-28389 Eq. (15) and BISON Theory Manual Eq. (10.21):
    ``R_a = (y F / 4)(S_g/V mu_f + 2 S_t/V mu_Uko)`` with fragment range
    ``mu_f = 6e-6`` m and uranium knock-on range ``mu_Uko = 5e-9`` m. With
    ``y F`` the gas production rate this is a fixed fraction of production.
    ``S_g/V`` (geometric) and ``S_t/V`` (total, including cracks) belong to the
    pellet geometry and have no source default.
    """

    fission_fragment_range: Array
    knockout_range: Array
    geometric_surface_to_volume: Array
    total_surface_to_volume: Array
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        geometric_surface_to_volume: ArrayLike,
        total_surface_to_volume: ArrayLike,
        fission_fragment_range: ArrayLike = 6.0e-6,
        knockout_range: ArrayLike = 5.0e-9,
    ) -> None:
        parameters = host_parameters(
            {
                "geometric_surface_to_volume": geometric_surface_to_volume,
                "total_surface_to_volume": total_surface_to_volume,
                "fission_fragment_range": fission_fragment_range,
                "knockout_range": knockout_range,
            }
        )
        if (
            parameters["total_surface_to_volume"]
            < parameters["geometric_surface_to_volume"]
        ):
            raise ValueError(
                "total_surface_to_volume must include the geometric surface."
            )
        fraction = 0.25 * (
            parameters["geometric_surface_to_volume"]
            * parameters["fission_fragment_range"]
            + 2.0 * parameters["total_surface_to_volume"] * parameters["knockout_range"]
        )
        if not fraction < 1.0:
            raise ValueError("Athermal release must release less than the production.")
        self.fission_fragment_range = jnp.asarray(parameters["fission_fragment_range"])
        self.knockout_range = jnp.asarray(parameters["knockout_range"])
        self.geometric_surface_to_volume = jnp.asarray(
            parameters["geometric_surface_to_volume"]
        )
        self.total_surface_to_volume = jnp.asarray(parameters["total_surface_to_volume"])
        self.source_id = "lewis-1987-jnm-148-28;inl-con-13-28389:eq15"
        self.law_id = canonical_fingerprint(
            {"kind": "athermal-recoil-knockout", "parameters": parameters}
        )

    def release_fraction(self) -> Array:
        """Fraction of the gas production released athermally."""

        return 0.25 * (
            self.geometric_surface_to_volume * self.fission_fragment_range
            + 2.0 * self.total_surface_to_volume * self.knockout_range
        )


@final
class _HbsSpeciesUpdate(StrictModule):
    pore_gas: Array
    boundary_gas: Array
    nucleation: Array
    trapping: Array
    resolution: Array
    successful: Array
    status: Array
    condition_estimate: Array
    residual_norm: Array


@final
class _GrainFaceUpdate(StrictModule):
    bubble_number_density: Array
    bubble_vacancies: Array
    bubble_gas_atoms: Array
    boundary_gas: Array
    venting_release: Array
    coverage: Array
    bubble_volume: Array
    connected_volume: Array
    converged: Array


class _MicrocrackingUpdate(NamedTuple):
    intactness: Array
    burst_release: Array  # [atoms/m^3]
    connected_volume: Array  # cavity volume connected by cracking [m^3/m^3]
    boundary_gas: Array  # grain-face gas left after the burst [atoms/m^3]
    bubble_number_density: Array  # surviving face bubbles [1/m^2]
    bubble_vacancies: Array  # vacancies per bubble after the growth remap
    saturation_coverage: Array


def _fission_gas_validity() -> PropertyValidityRange:
    # Solid UO2 up to its melting point (3120 K; Fink, J. Nucl. Mater. 279
    # (2000) 1); HBS formation and porosity data agree to 130 MWd/kgU (Barani
    # thesis 2020, Section 3.5.4).
    return PropertyValidityRange(
        {
            "temperature": (273.15, 3120.0, "K"),
            "burnup_mwd_per_kghm": (0.0, 130.0, "MWd/kgHM"),
        },
        source_id="fission-gas:solid-uo2-fink-2000;barani-2020-thesis-3.5.4",
    )


def _species_boundary_release(
    boundary_donor: Array,
    diffusion: Array,
    total_state: FissionGasSiteState,
    total_result: FissionGasStepResult,
) -> tuple[Array, Array, Array]:
    """Kr/Xe burst, venting and remaining grain-face gas at the total step's fractions.

    Burst draws on the swept boundary donor; venting draws on the gas present
    after the burst and diffusion arrival.
    """

    transfers = total_result.transfers
    total_boundary_donor = total_state.boundary_gas + transfers.grain_growth_sweeping
    burst_fraction = transfers.burst_release / jnp.where(
        total_boundary_donor > 0.0, total_boundary_donor, 1.0
    )
    burst = boundary_donor * burst_fraction[..., None]
    arrived = boundary_donor - burst + diffusion
    total_arrived = (
        total_boundary_donor - transfers.burst_release + transfers.diffusion_to_boundary
    )
    vent_fraction = transfers.venting_release / jnp.where(
        total_arrived > 0.0, total_arrived, 1.0
    )
    venting = arrived * vent_fraction[..., None]
    return burst, venting, arrived - venting


def _species_sum_matches(actual: Array, expected: Array) -> Array:
    """Whether the Kr/Xe sum of ``actual`` equals ``expected`` to roundoff."""

    return jnp.abs(
        jnp.sum(actual, axis=-1) - expected
    ) <= _SPECIES_LEDGER_ROUNDOFF * jnp.maximum(
        jnp.abs(expected) + jnp.sum(jnp.abs(actual), axis=-1), 1.0
    )


def _species_ledger_closes(
    state: FissionGasSpeciesState,
    next_state: FissionGasSpeciesState,
    total_state: FissionGasSiteState,
    total_result: FissionGasStepResult,
    *,
    released: Array,
    retained: Array,
    produced: Array,
) -> Array:
    """Whether the Kr/Xe inventories and transfers sum to the total-gas ledger."""

    total_next = total_result.state
    return (
        jnp.all(
            _species_sum_matches(
                state.intragranular_modes, total_state.intragranular_modes
            ),
            axis=-1,
        )
        & jnp.all(
            _species_sum_matches(
                next_state.intragranular_modes, total_next.intragranular_modes
            ),
            axis=-1,
        )
        & _species_sum_matches(
            state.intragranular_unresolved, total_state.intragranular_unresolved
        )
        & _species_sum_matches(state.boundary_gas, total_state.boundary_gas)
        & _species_sum_matches(state.hbs_pore_gas, total_state.hbs_pore_gas)
        & _species_sum_matches(state.hbs_boundary_gas, total_state.hbs_boundary_gas)
        & _species_sum_matches(state.produced, total_state.produced)
        & _species_sum_matches(state.released, total_state.released)
        & _species_sum_matches(state.boundary_released, total_state.boundary_released)
        & _species_sum_matches(next_state.produced, total_next.produced)
        & _species_sum_matches(next_state.released, total_next.released)
        & _species_sum_matches(next_state.boundary_released, total_next.boundary_released)
        & _species_sum_matches(
            next_state.intragranular_unresolved, total_next.intragranular_unresolved
        )
        & _species_sum_matches(next_state.boundary_gas, total_next.boundary_gas)
        & _species_sum_matches(next_state.hbs_pore_gas, total_next.hbs_pore_gas)
        & _species_sum_matches(next_state.hbs_boundary_gas, total_next.hbs_boundary_gas)
        & _species_sum_matches(released, total_result.released_increment)
        & _species_sum_matches(retained, total_result.retained_total)
        & _species_sum_matches(
            produced,
            total_result.transfers.intragranular_production
            + total_result.transfers.hbs_production
            + total_result.transfers.athermal_release,
        )
    )


@final
class FissionGasModel(StrictModule):
    """Per-site UO2 fission gas release and swelling (module docstring)."""

    diffusivity: TurnbullDiffusivity
    trapping: SpeightTrapping
    grain_faces: GrainFaceBubbles
    grain_growth: AinscoughGrainGrowth
    microcracking: GrainBoundaryMicrocracking
    high_burnup_structure: HighBurnupStructureFormation
    athermal: AthermalRelease
    validity: PropertyValidityRange
    vacancy_root: LocalRootPlan
    hbs_pores: HighBurnupStructurePores
    species_exchange: SmallLinearSolvePlan
    mode_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        athermal: AthermalRelease,
        mode_count: int = 40,
        diffusivity: TurnbullDiffusivity | None = None,
        trapping: SpeightTrapping | None = None,
        grain_faces: GrainFaceBubbles | None = None,
        grain_growth: AinscoughGrainGrowth | None = None,
        microcracking: GrainBoundaryMicrocracking | None = None,
        high_burnup_structure: HighBurnupStructureFormation | None = None,
        hbs_pores: HighBurnupStructurePores | None = None,
    ) -> None:
        if isinstance(mode_count, bool) or not isinstance(mode_count, int):
            raise TypeError("mode_count must be an int.")
        if mode_count < 1:
            raise ValueError("mode_count must be positive.")
        if not isinstance(athermal, AthermalRelease):
            raise TypeError("athermal must be an AthermalRelease.")
        diffusivity_ = TurnbullDiffusivity() if diffusivity is None else diffusivity
        trapping_ = SpeightTrapping() if trapping is None else trapping
        faces = GrainFaceBubbles() if grain_faces is None else grain_faces
        growth = AinscoughGrainGrowth() if grain_growth is None else grain_growth
        cracking = (
            GrainBoundaryMicrocracking() if microcracking is None else microcracking
        )
        restructuring = (
            HighBurnupStructureFormation()
            if high_burnup_structure is None
            else high_burnup_structure
        )
        pores = HighBurnupStructurePores() if hbs_pores is None else hbs_pores
        for value, kind, name in (
            (diffusivity_, TurnbullDiffusivity, "diffusivity"),
            (trapping_, SpeightTrapping, "trapping"),
            (faces, GrainFaceBubbles, "grain_faces"),
            (growth, AinscoughGrainGrowth, "grain_growth"),
            (cracking, GrainBoundaryMicrocracking, "microcracking"),
            (restructuring, HighBurnupStructureFormation, "high_burnup_structure"),
            (pores, HighBurnupStructurePores, "hbs_pores"),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be a {kind.__name__}.")
        self.diffusivity = diffusivity_
        self.trapping = trapping_
        self.grain_faces = faces
        self.grain_growth = growth
        self.microcracking = cracking
        self.high_burnup_structure = restructuring
        self.hbs_pores = pores
        self.athermal = athermal
        self.validity = _fission_gas_validity()
        self.vacancy_root = LocalRootPlan(
            maximum_steps=40,
            tolerance=1.0e-11,
            plan_id="fission-gas:grain-face-vacancies",
        )
        self.species_exchange = SmallLinearSolvePlan(2)
        self.mode_count = mode_count
        self.source_id = "pastore-2013-ned-256-75:sifgrs+barani-2017+barani-2020"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-fission-gas",
                "mode_count": mode_count,
                "cavity_topology": "conservative-connected-grain-boundary",
                "passive_species": FISSION_GAS_TRACER_SPECIES,
                "species_exchange": self.species_exchange.plan_id,
                "components": [
                    diffusivity_.law_id,
                    trapping_.law_id,
                    faces.law_id,
                    growth.law_id,
                    cracking.law_id,
                    restructuring.law_id,
                    pores.law_id,
                    athermal.law_id,
                ],
                "vacancy_root": self.vacancy_root.plan_id,
            }
        )

    def initial_state(
        self, shape: tuple[int, ...], /, *, grain_radius: ArrayLike
    ) -> FissionGasSiteState:
        """Fabricated state: gas-free, intact faces, ``N_0`` empty face bubbles."""

        require_site_shape(shape)
        radius = jnp.broadcast_to(
            jnp.asarray(positive_grain_radius(grain_radius), dtype=jnp.float64), shape
        )
        zeros = jnp.zeros(shape, dtype=jnp.float64)
        return FissionGasSiteState(
            intragranular_modes=jnp.zeros((*shape, self.mode_count), dtype=jnp.float64),
            intragranular_unresolved=zeros,
            grain_radius=radius,
            boundary_gas=zeros,
            bubble_number_density=jnp.broadcast_to(
                self.grain_faces.initial_bubble_number_density.astype(jnp.float64), shape
            ),
            bubble_vacancies=zeros,
            bubble_gas_atoms=zeros,
            grain_face_intactness=jnp.ones(shape, dtype=jnp.float64),
            hbs_fraction=zeros,
            hbs_pore_gas=zeros,
            hbs_boundary_gas=zeros,
            hbs_pore_number_density=zeros,
            hbs_pore_variance=zeros,
            hbs_pore_vacancies=zeros,
            hbs_porosity=zeros,
            effective_burnup_mwd_per_kghm=zeros,
            burnup_mwd_per_kghm=zeros,
            produced=zeros,
            released=zeros,
            boundary_released=zeros,
            connected_boundary_volume=zeros,
        )

    def initial_diagnostics(self, shape: tuple[int, ...], /) -> FissionGasDiagnostics:
        """State before the first step; no root or linear solve has been attempted.

        Status and rank -1 mean not attempted, the root flags are false and
        conditioning is unavailable. Numerical placeholders are finite zeros so
        carried state never injects NaN into derivatives (``NaN * 0 = NaN``).
        """

        require_site_shape(shape)
        zero = jnp.zeros(shape, dtype=jnp.float64)
        not_attempted = jnp.full(shape, -1, dtype=jnp.int32)
        false = jnp.zeros(shape, dtype=jnp.bool_)
        return FissionGasDiagnostics(
            effective_diffusivity=zero,
            fractional_coverage=zero,
            saturation_coverage=jnp.broadcast_to(
                self.grain_faces.saturation_coverage, shape
            ),
            bubble_pressure=zero,
            modal_truncation_bound=zero,
            vacancy_root_converged=false,
            hbs_root_converged=false,
            hbs_root_residual_norm=zero,
            hbs_root_condition_estimate=zero,
            hbs_root_linear_status=not_attempted,
            hbs_root_linear_rank=not_attempted,
            hbs_root_conditioning_available=false,
        )

    def initial_species_state(self, shape: tuple[int, ...], /) -> FissionGasSpeciesState:
        require_site_shape(shape)
        zeros = jnp.zeros((*shape, len(FISSION_GAS_TRACER_SPECIES)), dtype=jnp.float64)
        return FissionGasSpeciesState(
            intragranular_modes=jnp.zeros(
                (*shape, self.mode_count, len(FISSION_GAS_TRACER_SPECIES)),
                dtype=jnp.float64,
            ),
            intragranular_unresolved=zeros,
            boundary_gas=zeros,
            hbs_pore_gas=zeros,
            hbs_boundary_gas=zeros,
            produced=zeros,
            released=zeros,
            boundary_released=zeros,
        )

    def _hbs_species_step(
        self,
        pore_gas: Array,
        boundary_supply: Array,
        total_result: FissionGasStepResult,
    ) -> _HbsSpeciesUpdate:
        """Implicit Kr/Xe exchange between HBS pores and HBS boundary gas.

        Capture and re-solution use the total step's endpoint gross transfers
        divided by their endpoint donor inventories.
        """

        end_pore = total_result.state.hbs_pore_gas
        end_boundary = total_result.state.hbs_boundary_gas
        nucleation = total_result.transfers.hbs_nucleation
        capture = nucleation + total_result.transfers.hbs_trapping
        resolution = total_result.transfers.hbs_resolution
        capture_ratio = capture / jnp.where(end_boundary > 0.0, end_boundary, 1.0)
        resolution_ratio = resolution / jnp.where(end_pore > 0.0, end_pore, 1.0)
        matrix = jnp.stack(
            [
                jnp.stack([1.0 + resolution_ratio, -capture_ratio], axis=-1),
                jnp.stack([-resolution_ratio, 1.0 + capture_ratio], axis=-1),
            ],
            axis=-2,
        )
        # Exhausted donor pool (zero endpoint boundary gas with positive capture):
        # solve the exact limit A' + C' = available, C' = 0 as its own
        # conservation rows instead of an infinite capture coefficient.
        exhausted = (end_boundary == 0.0) & (capture > 0.0)
        right = jnp.stack([pore_gas, boundary_supply], axis=-2)
        closed_matrix = jnp.broadcast_to(
            jnp.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float64), matrix.shape
        )
        closed_right = jnp.stack(
            [pore_gas + boundary_supply, jnp.zeros_like(boundary_supply)], axis=-2
        )
        solved = solve_small_linear(
            self.species_exchange,
            jnp.where(exhausted[..., None, None], closed_matrix, matrix),
            jnp.where(exhausted[..., None, None], closed_right, right),
        )
        new_pore = solved.value[..., 0, :]
        new_boundary = solved.value[..., 1, :]
        resolved = new_pore * resolution_ratio[..., None]
        captured = jnp.where(
            exhausted[..., None],
            boundary_supply + resolved,
            capture_ratio[..., None] * new_boundary,
        )
        nucleated = (
            captured * (nucleation / jnp.where(capture > 0.0, capture, 1.0))[..., None]
        )
        return _HbsSpeciesUpdate(
            pore_gas=new_pore,
            boundary_gas=new_boundary,
            nucleation=nucleated,
            trapping=captured - nucleated,
            resolution=resolved,
            successful=solved.successful & ((end_pore > 0.0) | (resolution == 0.0)),
            status=solved.status,
            condition_estimate=solved.condition_estimate,
            residual_norm=solved.residual_norm,
        )

    def advance_species(
        self,
        state: FissionGasSpeciesState,
        total_state: FissionGasSiteState,
        total_result: FissionGasStepResult,
        /,
        *,
        species_production_rate: ArrayLike,
        time_step: ArrayLike,
    ) -> FissionGasSpeciesStepResult:
        """Trace the Kr/Xe identities through an evaluated total-gas step.

        ``species_production_rate`` has a trailing Kr/Xe axis. No geometry or
        root is recomputed: every transfer reuses the fractions and endpoint
        coefficients of ``total_result``. HBS exchange uses its endpoint gross
        transfers with implicit donor compositions; burst and venting draw on
        their own staged boundary donors, never on the whole retained mixture.
        """

        self._validate_species_step(state, total_state, total_result)
        shape = total_state.grain_radius.shape
        production_rate = jnp.broadcast_to(
            jnp.asarray(species_production_rate, dtype=jnp.float64),
            (*shape, len(FISSION_GAS_TRACER_SPECIES)),
        )
        dt = broadcast_to_sites(time_step, shape)
        _, weights_host, basis_host, unresolved_weight = modal_constants(self.mode_count)
        weights = jnp.asarray(weights_host)

        # Grain-growth sweeping and HBS depletion at the total step's fractions.
        intragranular_start = (
            jnp.swapaxes(state.intragranular_modes, -1, -2) @ weights
            + state.intragranular_unresolved
        )
        radius = total_result.state.grain_radius
        remap = (total_state.grain_radius / radius) ** 3
        sweeping = (1.0 - remap)[..., None] * intragranular_start
        depletion_fraction = _ratio_where_positive(
            total_result.state.hbs_fraction - total_state.hbs_fraction,
            1.0 - total_state.hbs_fraction,
        )
        depletion = depletion_fraction[..., None] * (intragranular_start - sweeping)
        remap = remap * (1.0 - depletion_fraction)

        produced = production_rate * dt[..., None]
        athermal = self.athermal.release_fraction() * produced
        hbs_production = (produced - athermal) * total_result.state.hbs_fraction[
            ..., None
        ]
        intragranular_production = produced - athermal - hbs_production
        modes, unresolved, diffusion = self._species_modal_step(
            state,
            remap=remap,
            production=intragranular_production,
            effective_diffusivity=total_result.diagnostics.effective_diffusivity,
            grain_radius=radius,
            dt=dt,
            weights=weights,
            unresolved_weight=unresolved_weight,
        )
        intragranular = jnp.swapaxes(modes, -1, -2) @ weights + unresolved
        burst, venting, boundary = _species_boundary_release(
            state.boundary_gas + sweeping, diffusion, total_state, total_result
        )
        hbs = self._hbs_species_step(
            state.hbs_pore_gas,
            state.hbs_boundary_gas + hbs_production + depletion,
            total_result,
        )
        released = athermal + burst + venting
        retained = intragranular + boundary + hbs.pore_gas + hbs.boundary_gas
        next_state = FissionGasSpeciesState(
            intragranular_modes=modes,
            intragranular_unresolved=unresolved,
            boundary_gas=boundary,
            hbs_pore_gas=hbs.pore_gas,
            hbs_boundary_gas=hbs.boundary_gas,
            produced=state.produced + produced,
            released=state.released + released,
            boundary_released=state.boundary_released + burst + venting,
        )
        consistent = _species_ledger_closes(
            state,
            next_state,
            total_state,
            total_result,
            released=released,
            retained=retained,
            produced=produced,
        )
        inventories = jnp.stack(
            [
                unresolved,
                boundary,
                hbs.pore_gas,
                hbs.boundary_gas,
                retained,
                produced,
                released,
                hbs.nucleation,
                hbs.trapping,
                hbs.resolution,
            ],
            axis=-2,
        )
        concentration = jnp.swapaxes(modes, -1, -2) @ jnp.asarray(basis_host).T
        scale = jnp.maximum(jnp.sum(jnp.abs(modes), axis=(-2, -1)), 1.0)
        admissible = (
            total_result.admissible
            & consistent
            & hbs.successful
            & jnp.all(jnp.isfinite(modes), axis=(-2, -1))
            & jnp.all(jnp.isfinite(inventories) & (inventories >= 0.0), axis=(-2, -1))
            & (jnp.min(concentration, axis=(-2, -1)) >= -_SPECIES_LEDGER_ROUNDOFF * scale)
        )
        return FissionGasSpeciesStepResult(
            state=next_state,
            transfers=FissionGasSpeciesTransfers(
                intragranular_production=intragranular_production,
                hbs_production=hbs_production,
                athermal_release=athermal,
                grain_growth_sweeping=sweeping,
                hbs_depletion=depletion,
                diffusion_to_boundary=diffusion,
                burst_release=burst,
                venting_release=venting,
                hbs_nucleation=hbs.nucleation,
                hbs_trapping=hbs.trapping,
                hbs_resolution=hbs.resolution,
            ),
            released_increment=released,
            retained_total=retained,
            linear_solve_successful=hbs.successful,
            linear_solve_status=hbs.status,
            linear_solve_condition_estimate=hbs.condition_estimate,
            linear_solve_residual_norm=hbs.residual_norm,
            admissible=admissible,
            in_range=total_result.in_range,
        )

    def _validate_species_step(
        self,
        state: FissionGasSpeciesState,
        total_state: FissionGasSiteState,
        total_result: FissionGasStepResult,
    ) -> None:
        if not isinstance(state, FissionGasSpeciesState):
            raise TypeError("state must be a FissionGasSpeciesState.")
        if not isinstance(total_state, FissionGasSiteState) or not isinstance(
            total_result, FissionGasStepResult
        ):
            raise TypeError("Total gas accepted state and candidate result are required.")
        if state.species != FISSION_GAS_TRACER_SPECIES:
            raise ValueError("Passive gas species must be in canonical Kr/Xe order.")
        if state.intragranular_modes.shape != (
            *total_state.grain_radius.shape,
            self.mode_count,
            len(FISSION_GAS_TRACER_SPECIES),
        ):
            raise ValueError(
                "Passive species site, mode and identity extents do not match the total model."
            )

    def _species_modal_step(
        self,
        state: FissionGasSpeciesState,
        *,
        remap: Array,
        production: Array,
        effective_diffusivity: Array,
        grain_radius: Array,
        dt: Array,
        weights: Array,
        unresolved_weight: float,
    ) -> tuple[Array, Array, Array]:
        """Kr/Xe modes, subgrid tail and diffusion release at the total step's diffusivity."""

        mode_numbers = jnp.arange(1, self.mode_count + 1, dtype=jnp.float64)
        rate_times = (
            effective_diffusivity[..., None]
            * (mode_numbers * jnp.pi / grain_radius[..., None]) ** 2
            * dt[..., None]
        )
        tail_rate_time = (
            effective_diffusivity
            * ((self.mode_count + 1) * jnp.pi / grain_radius) ** 2
            * dt
        )
        remapped_modes = state.intragranular_modes * remap[..., None, None]
        remapped_unresolved = state.intragranular_unresolved * remap[..., None]
        modes = (
            remapped_modes * jnp.exp(-rate_times)[..., None]
            + 2.0 * production[..., None, :] * _stable_decay_mean(rate_times)[..., None]
        )
        unresolved = (
            remapped_unresolved * jnp.exp(-tail_rate_time)[..., None]
            + unresolved_weight
            * production
            * _stable_decay_mean(tail_rate_time)[..., None]
        )
        diffusion_modes = (
            remapped_modes * (-jnp.expm1(-rate_times))[..., None]
            + 2.0 * production[..., None, :] * _stable_decay_loss(rate_times)[..., None]
        )
        diffusion = jnp.maximum(
            jnp.swapaxes(diffusion_modes, -1, -2) @ weights
            + remapped_unresolved * (-jnp.expm1(-tail_rate_time))[..., None]
            + unresolved_weight
            * production
            * _stable_decay_loss(tail_rate_time)[..., None],
            0.0,
        )
        return modes, unresolved, diffusion

    def effective_diffusivity(
        self, temperature: ArrayLike, fission_rate_density: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Speight ``D b / (b + g)`` [m^2/s] and the trapped fraction ``g / (b + g)``."""

        single = self.diffusivity.diffusivity(temperature, fission_rate_density)
        trapping, resolution = self.trapping.rates(
            temperature, fission_rate_density, single
        )
        total = trapping + resolution
        positive = total > 0.0
        safe_total = jnp.where(positive, total, 1.0)
        effective = jnp.where(positive, single * resolution / safe_total, single)
        trapped = jnp.where(positive, trapping / safe_total, 0.0)
        return effective, trapped

    def advance(
        self,
        state: FissionGasSiteState,
        /,
        *,
        temperature: ArrayLike,
        grain_radius: ArrayLike,
        fission_rate_density: ArrayLike,
        gas_production_rate: ArrayLike,
        burnup_mwd_per_kghm: ArrayLike,
        hydrostatic_stress: ArrayLike,
        time_step: ArrayLike,
        previous_temperature: ArrayLike,
    ) -> FissionGasStepResult:
        """One fully implicit step from ``state`` to the end-of-step inputs.

        Inputs broadcast to the site shape: temperature [K], fission rate
        density [1/m^3/s], gas (Xe + Kr) production rate [atoms/m^3/s],
        end-of-step burnup [MWd/kgHM], hydrostatic stress [Pa, compression
        negative], step length [s] and the temperature of ``state`` [K].
        ``grain_radius`` is the end-of-step radius prepared once by the caller
        through ``grain_growth.advance_radius`` and shared with helium.
        Stage order, which fixes every donor inventory and the ledger closure:
        grain-growth sweeping, HBS restructuring and depletion, production
        split and intragranular diffusion, micro-cracking burst, grain-face
        bubble growth, coalescence and venting, then HBS pore kinetics.
        """

        if not isinstance(state, FissionGasSiteState):
            raise TypeError("state must be a FissionGasSiteState.")
        if state.intragranular_modes.shape[-1] != self.mode_count:
            raise ValueError(
                f"state carries {state.intragranular_modes.shape[-1]} modes; the model "
                f"declares {self.mode_count}."
            )
        shape = state.grain_radius.shape
        temperature_ = broadcast_to_sites(temperature, shape)
        fission_rate_density_ = broadcast_to_sites(fission_rate_density, shape)
        production_rate = broadcast_to_sites(gas_production_rate, shape)
        # Endpoint burnup and accepted burnup are separate rounded epochs.
        # Fusing source-to-burnup conversion with their difference can invent
        # a negative increment at zero fission through an unrounded FMA.
        burnup = jax.lax.optimization_barrier(
            broadcast_to_sites(burnup_mwd_per_kghm, shape)
        )
        stress = broadcast_to_sites(hydrostatic_stress, shape)
        dt = broadcast_to_sites(time_step, shape)
        start_temperature = broadcast_to_sites(previous_temperature, shape)
        burnup_increment = burnup - state.burnup_mwd_per_kghm
        _, weights_host, basis_host, unresolved_weight = modal_constants(self.mode_count)
        weights = jnp.asarray(weights_host)
        radius = broadcast_to_sites(grain_radius, shape)

        effective_burnup, hbs_fraction = self._restructure(
            state, temperature_, burnup_increment
        )
        sweeping, hbs_depletion, modes, unresolved = self._remap_intragranular(
            state, radius, hbs_fraction, weights
        )
        produced_increment = production_rate * dt
        athermal_fraction = self.athermal.release_fraction()
        athermal_release = athermal_fraction * produced_increment
        hbs_production = (1.0 - athermal_fraction) * hbs_fraction * produced_increment
        intragranular_production = produced_increment - athermal_release - hbs_production
        effective, trapped_fraction = self.effective_diffusivity(
            temperature_, fission_rate_density_
        )
        diffusion, modes, unresolved = self._diffuse_intragranular(
            modes,
            unresolved,
            production=intragranular_production,
            effective_diffusivity=effective,
            grain_radius=radius,
            dt=dt,
            weights=weights,
            unresolved_weight=unresolved_weight,
        )
        intragranular = modes @ weights + unresolved
        cracking = self._crack_grain_faces(
            state,
            sweeping=sweeping,
            grain_radius=radius,
            start_temperature=start_temperature,
            temperature=temperature_,
            burnup=burnup,
            burnup_increment=burnup_increment,
        )
        faces = self._grain_face_step(
            cracking.boundary_gas,
            diffusion,
            cracking.bubble_number_density,
            cracking.bubble_vacancies,
            radius,
            temperature_,
            stress,
            dt,
            cracking.saturation_coverage,
        )
        hbs = self.hbs_pores.advance(
            state,
            arrival=hbs_production + hbs_depletion,
            fraction=hbs_fraction,
            temperature=temperature_,
            fission_rate_density=fission_rate_density_,
            stress=stress,
            time_step=dt,
        )

        released_increment = (
            athermal_release + cracking.burst_release + faces.venting_release
        )
        connected_increment = cracking.connected_volume + faces.connected_volume
        connected_volume = state.connected_boundary_volume + connected_increment
        candidate = FissionGasSiteState(
            intragranular_modes=modes,
            intragranular_unresolved=unresolved,
            grain_radius=radius,
            boundary_gas=faces.boundary_gas,
            bubble_number_density=faces.bubble_number_density,
            bubble_vacancies=faces.bubble_vacancies,
            bubble_gas_atoms=faces.bubble_gas_atoms,
            grain_face_intactness=cracking.intactness,
            connected_boundary_volume=connected_volume,
            hbs_fraction=hbs_fraction,
            hbs_pore_gas=hbs.gas,
            hbs_boundary_gas=hbs.boundary_gas,
            hbs_pore_number_density=hbs.density,
            hbs_pore_variance=hbs.variance,
            hbs_pore_vacancies=hbs.vacancies,
            hbs_porosity=hbs.porosity,
            effective_burnup_mwd_per_kghm=effective_burnup,
            burnup_mwd_per_kghm=burnup,
            produced=state.produced + produced_increment,
            released=state.released + released_increment,
            boundary_released=state.boundary_released
            + cracking.burst_release
            + faces.venting_release,
        )
        admissible = self._candidate_admissible(
            state,
            candidate,
            faces,
            hbs,
            connected_increment=connected_increment,
            burnup_increment=burnup_increment,
            temperature=temperature_,
            start_temperature=start_temperature,
            stress=stress,
            dt=dt,
            production_rate=production_rate,
            fission_rate_density=fission_rate_density_,
            basis=jnp.asarray(basis_host),
        )
        in_range = self.validity.contains(
            temperature=temperature_, burnup_mwd_per_kghm=burnup
        ) & self.grain_growth.drag_validity.contains(burnup_mwd_per_kghm=burnup)
        return FissionGasStepResult(
            state=candidate,
            transfers=FissionGasTransfers(
                intragranular_production=intragranular_production,
                hbs_production=hbs_production,
                athermal_release=athermal_release,
                grain_growth_sweeping=sweeping,
                hbs_depletion=hbs_depletion,
                diffusion_to_boundary=diffusion,
                burst_release=cracking.burst_release,
                venting_release=faces.venting_release,
                hbs_nucleation=hbs.nucleation,
                hbs_trapping=hbs.trapping,
                hbs_resolution=hbs.resolution,
            ),
            released_increment=released_increment,
            retained_total=intragranular
            + faces.boundary_gas
            + hbs.gas
            + hbs.boundary_gas,
            intragranular_swelling=self.trapping.bubble_atomic_volume
            * trapped_fraction
            * intragranular,
            intergranular_swelling=1.5
            / radius
            * faces.bubble_number_density
            * faces.bubble_volume
            + connected_volume,
            hbs_swelling=hbs.porosity,
            connected_boundary_volume_increment=connected_increment,
            cumulative_fgr=_ratio_where_positive(
                candidate.boundary_released, candidate.produced
            ),
            admissible=admissible,
            in_range=in_range,
            diagnostics=FissionGasDiagnostics(
                effective_diffusivity=effective,
                fractional_coverage=faces.coverage,
                saturation_coverage=cracking.saturation_coverage,
                bubble_pressure=self._bubble_pressure(faces, temperature_),
                modal_truncation_bound=unresolved,
                vacancy_root_converged=faces.converged,
                hbs_root_converged=hbs.converged,
                hbs_root_residual_norm=hbs.residual_norm,
                hbs_root_condition_estimate=hbs.condition_estimate,
                hbs_root_linear_status=hbs.linear_status,
                hbs_root_linear_rank=hbs.linear_rank,
                hbs_root_conditioning_available=hbs.conditioning_available,
            ),
        )

    def _restructure(
        self, state: FissionGasSiteState, temperature: Array, burnup_increment: Array
    ) -> tuple[Array, Array]:
        """Effective burnup, accumulated only below the HBS threshold, and HBS fraction."""

        effective_burnup = state.effective_burnup_mwd_per_kghm + jnp.where(
            temperature <= self.high_burnup_structure.threshold_temperature,
            burnup_increment,
            0.0,
        )
        return effective_burnup, self.high_burnup_structure.restructured_fraction(
            effective_burnup
        )

    def _remap_intragranular(
        self,
        state: FissionGasSiteState,
        grain_radius: Array,
        hbs_fraction: Array,
        weights: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Grain-growth sweeping and HBS depletion of the intragranular inventory.

        Returns the swept and depleted gas and the remapped modes and subgrid
        tail.
        """

        retained_volume_fraction = (state.grain_radius / grain_radius) ** 3
        intragranular_start = (
            state.intragranular_modes @ weights + state.intragranular_unresolved
        )
        sweeping = (1.0 - retained_volume_fraction) * intragranular_start
        modes = state.intragranular_modes * retained_volume_fraction[..., None]
        unresolved = state.intragranular_unresolved * retained_volume_fraction
        depleted_fraction = _ratio_where_positive(
            hbs_fraction - state.hbs_fraction, 1.0 - state.hbs_fraction
        )
        hbs_depletion = depleted_fraction * (intragranular_start - sweeping)
        modes = modes * (1.0 - depleted_fraction)[..., None]
        unresolved = unresolved * (1.0 - depleted_fraction)
        return sweeping, hbs_depletion, modes, unresolved

    def _diffuse_intragranular(
        self,
        modes: Array,
        unresolved: Array,
        *,
        production: Array,
        effective_diffusivity: Array,
        grain_radius: Array,
        dt: Array,
        weights: Array,
        unresolved_weight: float,
    ) -> tuple[Array, Array, Array]:
        """Exact frozen-coefficient modal diffusion.

        Returns the release to the grain faces and the updated modes and
        subgrid tail.
        """

        mode_numbers = jnp.reshape(
            jnp.arange(1, self.mode_count + 1, dtype=jnp.float64),
            (1,) * grain_radius.ndim + (self.mode_count,),
        )
        wavenumbers = mode_numbers * jnp.pi / grain_radius[..., None]
        rate_times = effective_diffusivity[..., None] * wavenumbers**2 * dt[..., None]
        tail_rate_time = (
            effective_diffusivity
            * ((self.mode_count + 1) * jnp.pi / grain_radius) ** 2
            * dt
        )
        diffusion = jnp.maximum(
            (
                modes * (-jnp.expm1(-rate_times))
                + 2.0 * production[..., None] * _stable_decay_loss(rate_times)
            )
            @ weights
            + unresolved * (-jnp.expm1(-tail_rate_time))
            + unresolved_weight * production * _stable_decay_loss(tail_rate_time),
            0.0,
        )
        # beta s_m (1 - E_m) / k_m with s_m = 2 and beta dt = production.
        modes = modes * jnp.exp(-rate_times) + 2.0 * production[
            ..., None
        ] * _stable_decay_mean(rate_times)
        unresolved = unresolved * jnp.exp(
            -tail_rate_time
        ) + unresolved_weight * production * _stable_decay_mean(tail_rate_time)
        return diffusion, modes, unresolved

    def _crack_grain_faces(
        self,
        state: FissionGasSiteState,
        *,
        sweeping: Array,
        grain_radius: Array,
        start_temperature: Array,
        temperature: Array,
        burnup: Array,
        burnup_increment: Array,
    ) -> _MicrocrackingUpdate:
        """Micro-cracking burst on the temperature variation, then healing.

        Cracking connects existing cavities without annihilating their gas or
        vacancy volume; the grain-growth remap already conserves that volume.
        """

        cracked, intact = self.microcracking.intactness(
            state.grain_face_intactness,
            start_temperature,
            temperature,
            burnup,
            burnup_increment,
        )
        surviving_faces = jnp.exp(cracked - state.grain_face_intactness)
        boundary_after_sweeping = state.boundary_gas + sweeping
        burst = (1.0 - surviving_faces) * boundary_after_sweeping
        vacancies = state.bubble_vacancies * (grain_radius / state.grain_radius)
        volume_before_burst = (
            self.grain_faces.van_der_waals_volume * boundary_after_sweeping
            + 1.5
            / grain_radius
            * state.bubble_number_density
            * vacancies
            * self.grain_faces.vacancy_volume
        )
        return _MicrocrackingUpdate(
            intactness=intact,
            burst_release=burst,
            connected_volume=(1.0 - surviving_faces) * volume_before_burst,
            boundary_gas=boundary_after_sweeping - burst,
            bubble_number_density=state.bubble_number_density * surviving_faces,
            bubble_vacancies=vacancies,
            saturation_coverage=self.grain_faces.saturation_coverage
            * jnp.exp(intact - 1.0),
        )

    def _bubble_pressure(self, faces: _GrainFaceUpdate, temperature: Array) -> Array:
        """Ideal-gas pressure [Pa] of the grain-face bubbles, zero without vacancies."""

        occupied = faces.bubble_vacancies > 0.0
        return jnp.where(
            occupied,
            BOLTZMANN_J_PER_K
            * temperature
            * faces.bubble_gas_atoms
            / (
                self.grain_faces.vacancy_volume
                * jnp.where(occupied, faces.bubble_vacancies, 1.0)
            ),
            0.0,
        )

    def _candidate_admissible(
        self,
        state: FissionGasSiteState,
        candidate: FissionGasSiteState,
        faces: _GrainFaceUpdate,
        hbs: HbsPoreUpdate,
        *,
        connected_increment: Array,
        burnup_increment: Array,
        temperature: Array,
        start_temperature: Array,
        stress: Array,
        dt: Array,
        production_rate: Array,
        fission_rate_density: Array,
        basis: Array,
    ) -> Array:
        """Finite nonnegative candidate inside its physical domain, both roots converged."""

        modes = candidate.intragranular_modes
        intact = candidate.grain_face_intactness
        radius = candidate.grain_radius
        concentration = modes @ basis.T
        scale = jnp.sum(jnp.abs(modes), axis=-1)
        inventories = jnp.stack(
            [
                faces.boundary_gas,
                faces.bubble_vacancies,
                faces.bubble_gas_atoms,
                faces.bubble_number_density,
                hbs.gas,
                hbs.boundary_gas,
                hbs.density,
                hbs.variance,
                hbs.vacancies,
                hbs.porosity,
                candidate.intragranular_unresolved,
                candidate.connected_boundary_volume,
                connected_increment,
                hbs.nucleation,
                hbs.trapping,
                hbs.resolution,
                candidate.released,
                candidate.produced,
                dt,
                production_rate,
                fission_rate_density,
            ],
            axis=-1,
        )
        return (
            jnp.all(jnp.isfinite(modes), axis=-1)
            & jnp.all(jnp.isfinite(inventories), axis=-1)
            & jnp.isfinite(intact)
            & jnp.isfinite(radius)
            & (jnp.min(concentration, axis=-1) >= -NONNEGATIVITY_ROUNDOFF * scale)
            & jnp.all(inventories >= 0.0, axis=-1)
            & (faces.coverage >= 0.0)
            & (faces.coverage <= 1.0)
            & (burnup_increment >= 0.0)
            & (intact >= 0.0)
            & (intact <= 1.0)
            & (radius > 0.0)
            & (radius >= state.grain_radius)
            & (temperature > 0.0)
            & jnp.isfinite(stress)
            & jnp.isfinite(start_temperature)
            & (start_temperature > 0.0)
            & (candidate.hbs_fraction >= state.hbs_fraction)
            & (candidate.hbs_fraction <= 1.0)
            & (hbs.porosity < 1.0)
            & (hbs.gas >= DIMER_ATOMS * hbs.density)
            & jnp.isfinite(hbs.residual_norm)
            & (~hbs.conditioning_available | jnp.isfinite(hbs.condition_estimate))
            & hbs.converged
            & faces.converged
        )

    def _grain_face_step(
        self,
        boundary_gas: Array,
        arrival: Array,
        bubble_number_density: Array,
        bubble_vacancies: Array,
        grain_radius: Array,
        temperature: Array,
        hydrostatic_stress: Array,
        time_step: Array,
        saturation_coverage: Array,
    ) -> _GrainFaceUpdate:
        """Gas arrival, implicit vacancy absorption, coalescence and venting."""

        faces = self.grain_faces
        # Grain-face bubbles per unit fuel volume: (3 / (2a)) N.
        bubbles_per_volume = 1.5 / grain_radius * bubble_number_density
        safe_bubbles = jnp.where(bubbles_per_volume > 0.0, bubbles_per_volume, 1.0)
        arrived = boundary_gas + arrival
        gas_start = jnp.where(bubbles_per_volume > 0.0, boundary_gas / safe_bubbles, 0.0)
        gas_end = jnp.where(bubbles_per_volume > 0.0, arrived / safe_bubbles, 0.0)

        flat_shape = (-1,)
        vacancies, converged = jax.vmap(self._vacancy_update)(
            jnp.reshape(gas_end, flat_shape),
            jnp.reshape(bubble_vacancies, flat_shape),
            jnp.reshape(bubble_number_density, flat_shape),
            jnp.reshape(temperature, flat_shape),
            jnp.reshape(hydrostatic_stress, flat_shape),
            jnp.reshape(time_step, flat_shape),
            jnp.reshape(saturation_coverage, flat_shape),
        )
        vacancies = jnp.reshape(vacancies, gas_end.shape)
        converged = jnp.reshape(converged, gas_end.shape)

        # White coalescence: N / (3 - 2F)^3 is invariant while bubbles grow and
        # bubble volume per face area N V is conserved, so
        # F = 3q / (1 + 2q), q = (K (N V)^2 / c^2)^(1/3), N = c^2 F^3 / (N V)^2
        # with V = c A^(3/2).
        volume_start = gas_start * faces.van_der_waals_volume + (
            bubble_vacancies * faces.vacancy_volume
        )
        volume_end = (
            gas_end * faces.van_der_waals_volume + vacancies * faces.vacancy_volume
        )

        def area(volume: Array, /) -> Array:
            # An empty bubble has zero area; the guard keeps the cube-root
            # derivative finite at zero volume.
            occupied = volume > 0.0
            safe = jnp.where(occupied, volume, 1.0)
            return jnp.where(occupied, faces.projected_area(safe), 0.0)

        coverage_start = bubble_number_density * area(volume_start)
        shape_constant = (
            4.0
            * jnp.pi
            * faces.lenticular_factor()
            / (3.0 * (jnp.pi * jnp.sin(faces.semidihedral_angle) ** 2) ** 1.5)
        )
        volume_per_area = bubble_number_density * volume_end
        invariant = (
            bubble_number_density
            / jnp.maximum(3.0 - 2.0 * coverage_start, _COALESCENCE_DENOMINATOR_FLOOR) ** 3
        )
        grows = (volume_end > volume_start) & (volume_per_area > 0.0)
        safe_volume_per_area = jnp.where(grows, volume_per_area, 1.0)
        ratio = jnp.cbrt(invariant * safe_volume_per_area**2 / shape_constant**2)
        coalesced_coverage = 3.0 * ratio / (1.0 + 2.0 * ratio)
        coalesced_density = (
            shape_constant**2 * coalesced_coverage**3 / safe_volume_per_area**2
        )
        density = jnp.where(grows, coalesced_density, bubble_number_density)
        coverage = jnp.where(
            grows,
            coalesced_coverage,
            bubble_number_density * area(volume_end),
        )

        # Venting: bubbles beyond the saturation coverage connect to the free
        # volume and release their gas; per-bubble contents are unchanged.
        vents = coverage > saturation_coverage
        kept = jnp.where(
            vents, saturation_coverage / jnp.where(vents, coverage, 1.0), 1.0
        )
        venting = arrived * (1.0 - kept)
        per_bubble = bubble_number_density / jnp.where(density > 0.0, density, 1.0)
        return _GrainFaceUpdate(
            bubble_number_density=density * kept,
            bubble_vacancies=vacancies * per_bubble,
            bubble_gas_atoms=gas_end * per_bubble,
            boundary_gas=arrived - venting,
            venting_release=venting,
            coverage=coverage * kept,
            bubble_volume=volume_end * per_bubble,
            connected_volume=1.5 / grain_radius * volume_per_area * (1.0 - kept),
            converged=converged,
        )

    def _vacancy_update(
        self,
        gas_atoms: Array,
        vacancies: Array,
        bubble_number_density: Array,
        temperature: Array,
        hydrostatic_stress: Array,
        time_step: Array,
        saturation_coverage: Array,
    ) -> tuple[Array, Array]:
        """Branch-local physical root; empty/zero-time faces never evaluate it."""

        def unchanged(_: None) -> tuple[Array, Array]:
            return vacancies, jnp.asarray(True, dtype=jnp.bool_)

        def occupied(_: None) -> tuple[Array, Array]:
            return self._occupied_vacancy_update(
                gas_atoms,
                vacancies,
                bubble_number_density,
                temperature,
                hydrostatic_stress,
                time_step,
                saturation_coverage,
            )

        def empty(_: None) -> tuple[Array, Array]:
            def sinter(_: None) -> tuple[Array, Array]:
                faces = self.grain_faces
                volume = vacancies * faces.vacancy_volume
                radius = faces.radius_of_curvature(volume)
                sink = speight_beere_sink_strength(
                    faces, bubble_number_density, volume, saturation_coverage
                )
                mobility = (
                    2.0
                    * jnp.pi
                    * faces.vacancy_diffusivity(temperature)
                    * faces.boundary_layer_thickness
                    / (BOLTZMANN_J_PER_K * temperature * sink)
                )
                equilibrium = 2.0 * faces.surface_energy / radius - hydrostatic_stress
                count = jnp.maximum(vacancies - time_step * mobility * equilibrium, 0.0)
                return count, jnp.asarray(True, dtype=jnp.bool_)

            return domain_cond(vacancies > 0.0, sinter, unchanged)

        def evolving(_: None) -> tuple[Array, Array]:
            return domain_cond(gas_atoms > 0.0, occupied, empty)

        return domain_cond(time_step > 0.0, evolving, unchanged)

    def _occupied_vacancy_update(
        self,
        gas_atoms: Array,
        vacancies: Array,
        bubble_number_density: Array,
        temperature: Array,
        hydrostatic_stress: Array,
        time_step: Array,
        saturation_coverage: Array,
    ) -> tuple[Array, Array]:
        """Backward-Euler Speight-Beere vacancy absorption of one site's bubbles.

        ``n' - n = dt 2 pi D_v delta / (k T S(F')) (k T n_g / (Omega n') - p_eq(R'))``
        (INL/CON-13-28389 Eqs. (5)-(9)), with the sink strength
        ``S(F) = -(3 - F)(1 - F)/4 - ln(F)/2`` evaluated at the end-of-step
        coverage bounded by the saturation coverage. Multiplied by ``n'`` it is
        a quadratic for frozen ``S`` and ``p_eq``; its root seeds a Newton
        solve in ``z = ln(n' / n_ref)``.
        """

        faces = self.grain_faces
        thermal_energy = BOLTZMANN_J_PER_K * temperature
        mobility_constant = (
            2.0
            * jnp.pi
            * faces.vacancy_diffusivity(temperature)
            * faces.boundary_layer_thickness
            / thermal_energy
        )
        gas_term = thermal_energy * gas_atoms / faces.vacancy_volume

        def coefficients(count: Array) -> tuple[Array, Array]:
            volume = gas_atoms * faces.van_der_waals_volume + count * faces.vacancy_volume
            radius = faces.radius_of_curvature(volume)
            sink = speight_beere_sink_strength(
                faces, bubble_number_density, volume, saturation_coverage
            )
            equilibrium = 2.0 * faces.surface_energy / radius - hydrostatic_stress
            return time_step * mobility_constant / sink, equilibrium

        start_mobility, start_equilibrium = coefficients(vacancies)
        constant = start_mobility * gas_term
        reference = positive_quadratic_root(
            vacancies - start_mobility * start_equilibrium, constant
        )
        scale = (
            reference * (reference + vacancies)
            + constant
            + start_mobility * jnp.abs(start_equilibrium) * reference
        )

        def residual(log_ratio: Array) -> Array:
            count = reference * jnp.exp(log_ratio)
            mobility, equilibrium = coefficients(count)
            return (
                count * (count - vacancies) - mobility * (gas_term - equilibrium * count)
            ) / scale

        log_ratio, diagnostics = self.vacancy_root.solve_with_diagnostics(
            residual, jnp.zeros_like(reference)
        )
        return reference * jnp.exp(log_ratio), diagnostics.converged


__all__ = [
    "AthermalRelease",
    "FissionGasModel",
]
