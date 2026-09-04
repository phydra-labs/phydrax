#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Backend-neutral prognostic subgrid kinetic-energy (KSGS) physics.

The transported scalar is specific SGS kinetic energy ``k`` [m² s⁻²]. This
module supplies only the local constitutive quantities and source part of

    ∂k/∂t + u·∇k = P - ε + ∇·(D ∇k) + B - ε_low-Re.

A spatial backend owns advection and evaluates the conservative diffusion
operator with the returned ``diffusivity`` ``D`` [m² s⁻¹]. It passes that
already-evaluated diffusion rate [m² s⁻³] back in :class:`KSGSInputs`; no
spatial stencil, boundary condition, or time integrator is invented here.

With volume-equivalent filter width ``Δ`` [m], deviatoric resolved strain
``Sᵈ`` [s⁻¹], and molecular kinematic viscosity ``ν`` [m² s⁻¹], the static
closure is

    ν_t = C_ν Δ √k,
    P_raw = 2 ν_t Sᵈ:Sᵈ,
    ε = C_ε k^(3/2)/Δ,
    D = ν + C_D ν_t,
    P = min(P_raw, C_lim ε).

Buoyancy coupling adds ``B = -C_B ν_t N²`` for a supplied squared buoyancy
frequency ``N²`` [s⁻²], so stable stratification is a sink and unstable
stratification is a source. The low-Re variant multiplies ``ν_t`` by
``1-exp(-C_f y√k/ν)`` for resolved wall distance ``y`` and adds
``ε_low-Re = C_L ν |∇√k|²``. Dynamic plans obtain ``C_ν`` from explicitly
supplied test-filter contractions and restartable exponential histories.
Every coefficient is required: published implementation values (including
Nalu-family choices) are reference/calibration inputs, never defaults.
"""

from __future__ import annotations

import abc
import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._les_closures import (
    LESFilterScale,
    LESParameterProvenance,
    ResolvedLESFilter,
)


class KSGSCoefficients(StrictModule, NonTrainableState):
    """Required dimensionless coefficients for the complete KSGS source."""

    eddy_viscosity: float = eqx.field(static=True)
    dissipation: float = eqx.field(static=True)
    diffusion: float = eqx.field(static=True)
    buoyancy: float = eqx.field(static=True)
    production_limit: float = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)

    def __init__(
        self,
        eddy_viscosity: float,
        dissipation: float,
        diffusion: float,
        buoyancy: float,
        production_limit: float,
        /,
    ):
        values = tuple(
            float(value)
            for value in (
                eddy_viscosity,
                dissipation,
                diffusion,
                buoyancy,
                production_limit,
            )
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Every KSGS coefficient must be finite and positive.")
        (
            self.eddy_viscosity,
            self.dissipation,
            self.diffusion,
            self.buoyancy,
            self.production_limit,
        ) = values
        self.coefficient_id = canonical_fingerprint(
            {
                "kind": "ksgs-coefficients",
                "eddy_viscosity": values[0],
                "dissipation": values[1],
                "diffusion": values[2],
                "buoyancy": values[3],
                "production_limit": values[4],
            }
        )


class LowReKSGSCoefficients(StrictModule, NonTrainableState):
    """Required low-Re damping and viscous-dissipation coefficients."""

    damping: float = eqx.field(static=True)
    viscous_dissipation: float = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)

    def __init__(self, damping: float, viscous_dissipation: float, /):
        damping_ = float(damping)
        dissipation_ = float(viscous_dissipation)
        if any(
            not math.isfinite(value) or value <= 0.0 for value in (damping_, dissipation_)
        ):
            raise ValueError("Low-Re KSGS coefficients must be finite and positive.")
        self.damping = damping_
        self.viscous_dissipation = dissipation_
        self.coefficient_id = canonical_fingerprint(
            {
                "kind": "low-re-ksgs-coefficients",
                "damping": damping_,
                "viscous_dissipation": dissipation_,
            }
        )


class KSGSState(StrictModule):
    """Restart-complete KSGS state; every field has the shape of ``k``.

    ``kinetic_energy`` is prognostic. The remaining fields are dynamic-model
    continuation data and remain bitwise unchanged for non-dynamic plans.
    ``dynamic_updates`` counts accepted averaging samples per local state.
    """

    kinetic_energy: Array
    dynamic_numerator: Array
    dynamic_denominator: Array
    eddy_viscosity_coefficient: Array
    dynamic_updates: Array


class KSGSInputs(StrictModule):
    """Backend data common to every KSGS rate evaluation.

    ``molecular_kinematic_viscosity`` has units m² s⁻¹. A backend first calls
    :meth:`AbstractKSGSPlan.transport` with the same state, filter scale, and
    molecular viscosity, evaluates ``∇·(D∇k)`` with its returned diffusivity,
    and supplies that rate [m² s⁻³] as ``diffusion_rate`` here.
    """

    velocity_gradient: Array
    filter_scale: LESFilterScale
    molecular_kinematic_viscosity: Array
    diffusion_rate: Array

    def __init__(
        self,
        velocity_gradient: ArrayLike,
        filter_scale: LESFilterScale,
        molecular_kinematic_viscosity: ArrayLike,
        diffusion_rate: ArrayLike,
        /,
    ):
        if not isinstance(filter_scale, LESFilterScale):
            raise TypeError("filter_scale must be LESFilterScale.")
        gradient = _inexact(velocity_gradient)
        if gradient.ndim < 2 or gradient.shape[-2:] != (3, 3):
            raise ValueError("KSGS velocity_gradient must have trailing shape (3, 3).")
        self.velocity_gradient = gradient
        self.filter_scale = filter_scale
        self.molecular_kinematic_viscosity = _inexact(molecular_kinematic_viscosity)
        self.diffusion_rate = _inexact(diffusion_rate)


class BuoyancyKSGSInputs(StrictModule):
    """Base KSGS inputs plus ``N²`` [s⁻²], positive for stable stratification."""

    base: KSGSInputs
    buoyancy_frequency_squared: Array

    def __init__(self, base: KSGSInputs, buoyancy_frequency_squared: ArrayLike, /):
        if not isinstance(base, KSGSInputs):
            raise TypeError("base must be KSGSInputs.")
        self.base = base
        self.buoyancy_frequency_squared = _inexact(buoyancy_frequency_squared)


class DynamicKSGSInputs(StrictModule):
    """Explicit test-filter and averaging data for one dynamic update.

    ``leonard_stress`` and ``modeled_stress`` are symmetric specific-stress
    tensors [m² s⁻²], with trailing shape ``(3, 3)``. The backend performs
    filtering; this module forms their double contractions. Their
    normalization must make ``(L:M)/(M:M)`` the dimensionless ``C_ν``.
    ``averaging_weight`` is the exponential weight in ``[0, 1]`` and
    ``accept_update`` explicitly records accepted-step continuation cadence.
    """

    base: KSGSInputs
    leonard_stress: Array
    modeled_stress: Array
    averaging_weight: Array
    accept_update: Array

    def __init__(
        self,
        base: KSGSInputs,
        leonard_stress: ArrayLike,
        modeled_stress: ArrayLike,
        averaging_weight: ArrayLike,
        accept_update: ArrayLike,
        /,
    ):
        if not isinstance(base, KSGSInputs):
            raise TypeError("base must be KSGSInputs.")
        leonard = _inexact(leonard_stress)
        modeled = _inexact(modeled_stress)
        if leonard.shape != modeled.shape or leonard.shape[-2:] != (3, 3):
            raise ValueError(
                "Dynamic KSGS stresses must have matching (..., 3, 3) shapes."
            )
        self.base = base
        self.leonard_stress = leonard
        self.modeled_stress = modeled
        self.averaging_weight = _inexact(averaging_weight)
        self.accept_update = jnp.asarray(accept_update, dtype=bool)


class LowReKSGSInputs(StrictModule):
    """Base inputs plus resolved wall distance and backend ``∇√k``."""

    base: KSGSInputs
    wall_distance: Array
    sqrt_kinetic_energy_gradient: Array

    def __init__(
        self,
        base: KSGSInputs,
        wall_distance: ArrayLike,
        sqrt_kinetic_energy_gradient: ArrayLike,
        /,
    ):
        if not isinstance(base, KSGSInputs):
            raise TypeError("base must be KSGSInputs.")
        distance = _inexact(wall_distance)
        gradient = _inexact(sqrt_kinetic_energy_gradient)
        if gradient.ndim < 1 or gradient.shape[-1] != 3:
            raise ValueError("Low-Re ∇sqrt(k) must have trailing shape (3,).")
        self.base = base
        self.wall_distance = distance
        self.sqrt_kinetic_energy_gradient = gradient


class KSGSTransportResult(StrictModule):
    """Pre-operator eddy viscosity and total ``k`` diffusivity [m² s⁻¹]."""

    eddy_viscosity: Array
    diffusivity: Array
    kinetic_energy_nonnegative: Array
    eddy_viscosity_nonnegative: Array
    finite: Array


class KSGSContributions(StrictModule):
    """Signed and unsigned local terms, all with units m² s⁻³."""

    raw_production: Array
    production: Array
    production_limit_reduction: Array
    dissipation: Array
    diffusion: Array
    buoyancy: Array
    low_re_dissipation: Array
    rhs: Array


class KSGSEvidence(StrictModule):
    """Pointwise audit evidence for one pure KSGS transition."""

    production_limited: Array
    kinetic_energy_nonnegative: Array
    eddy_viscosity_nonnegative: Array
    production_nonnegative: Array
    dissipation_nonnegative: Array
    finite: Array
    dynamic_update_accepted: Array
    dynamic_numerator: Array
    dynamic_denominator: Array


class KSGSResult(StrictModule):
    """Constitutive outputs, source RHS, evidence, and restart-complete state."""

    state: KSGSState
    eddy_viscosity: Array
    diffusivity: Array
    contributions: KSGSContributions
    evidence: KSGSEvidence
    plan_id: str = eqx.field(static=True)


class AbstractKSGSPlan(StrictModule, NonTrainableState):
    """Abstract provenance-bound prognostic KSGS equation family."""

    coefficients: KSGSCoefficients
    provenance: LESParameterProvenance
    plan_id: str = eqx.field(static=True)

    def initialize_state(self, kinetic_energy: ArrayLike, /) -> KSGSState:
        """Create fixed-shape state with explicit dynamic initial coefficient."""
        kinetic = _inexact(kinetic_energy)
        if kinetic.size == 0:
            raise ValueError("KSGS kinetic energy must be non-empty.")
        kinetic = eqx.error_if(
            kinetic,
            jnp.any(kinetic < 0.0),
            "KSGS kinetic energy is negative; no flooring policy is enabled.",
        )
        zeros = jnp.zeros_like(kinetic)
        return KSGSState(
            kinetic_energy=kinetic,
            dynamic_numerator=zeros,
            dynamic_denominator=zeros,
            eddy_viscosity_coefficient=jnp.full_like(
                kinetic, self.coefficients.eddy_viscosity
            ),
            dynamic_updates=jnp.zeros_like(kinetic, dtype=jnp.int32),
        )

    def transport(
        self,
        state: KSGSState,
        filter_scale: LESFilterScale,
        molecular_kinematic_viscosity: ArrayLike,
        /,
        *,
        wall_distance: ArrayLike | None = None,
    ) -> KSGSTransportResult:
        """Evaluate coefficients before a backend applies its diffusion operator."""
        return _transport(
            self,
            state,
            filter_scale,
            molecular_kinematic_viscosity,
            wall_distance=wall_distance,
        )

    @abc.abstractmethod
    def evaluate(self, state: KSGSState, inputs: object, /) -> KSGSResult:
        """Return one pure local RHS/continuation transition."""
        raise NotImplementedError


class StaticKSGSPlan(AbstractKSGSPlan):
    """Static-coefficient KSGS equation with no buoyancy or low-Re term."""

    def __init__(
        self, coefficients: KSGSCoefficients, provenance: LESParameterProvenance, /
    ):
        _assign_plan(self, coefficients, provenance, "static-ksgs")

    def evaluate(self, state: KSGSState, inputs: object, /) -> KSGSResult:
        if not isinstance(inputs, KSGSInputs):
            raise TypeError("Static KSGS requires KSGSInputs.")
        return _evaluate(
            self, state, inputs, state, jnp.zeros_like(state.kinetic_energy), None
        )


class BuoyancyKSGSPlan(AbstractKSGSPlan):
    """Static KSGS equation coupled to resolved stable/unstable buoyancy."""

    def __init__(
        self, coefficients: KSGSCoefficients, provenance: LESParameterProvenance, /
    ):
        _assign_plan(self, coefficients, provenance, "buoyancy-ksgs")

    def evaluate(self, state: KSGSState, inputs: object, /) -> KSGSResult:
        if not isinstance(inputs, BuoyancyKSGSInputs):
            raise TypeError("Buoyancy KSGS requires BuoyancyKSGSInputs.")
        return _evaluate(
            self,
            state,
            inputs.base,
            state,
            inputs.buoyancy_frequency_squared,
            None,
        )


class DynamicKSGSPlan(AbstractKSGSPlan):
    """Dissipative dynamic ``C_ν`` KSGS equation with restartable histories.

    The test filter has a distinct semantic identity and required scale ratio;
    neither can be confused with spectral dealiasing or inferred by a backend.
    Physics uses the committed coefficient in ``state``; an accepted sample
    updates the returned continuation state for the next evaluation.
    """

    test_filter: ResolvedLESFilter
    test_filter_scale_ratio: float = eqx.field(static=True)

    def __init__(
        self,
        coefficients: KSGSCoefficients,
        provenance: LESParameterProvenance,
        test_filter: ResolvedLESFilter,
        test_filter_scale_ratio: float,
        /,
    ):
        if not isinstance(test_filter, ResolvedLESFilter):
            raise TypeError("test_filter must be ResolvedLESFilter.")
        ratio = float(test_filter_scale_ratio)
        if not math.isfinite(ratio) or ratio <= 1.0:
            raise ValueError("Dynamic KSGS test-filter scale ratio must exceed one.")
        _assign_plan(self, coefficients, provenance, "dynamic-ksgs")
        resolved_filter = provenance.resolved_filter
        if test_filter.filter_id == resolved_filter.filter_id:
            raise ValueError(
                "Dynamic KSGS test filter must differ from the resolved LES filter."
            )
        if (
            test_filter.axis_names != resolved_filter.axis_names
            or test_filter.topology != resolved_filter.topology
            or test_filter.boundary_class != resolved_filter.boundary_class
        ):
            raise ValueError(
                "Dynamic KSGS filters require matching axes, topology, and boundary."
            )
        if test_filter.family not in (
            "explicit-filter",
            "sharp-fourier-projection",
        ):
            raise ValueError(
                "Dynamic KSGS requires an explicit or sharp test-filter family."
            )
        if (
            test_filter.commutation_status == "unmodeled"
            or resolved_filter.commutation_status == "unmodeled"
            or test_filter.commutation_status != resolved_filter.commutation_status
        ):
            raise ValueError(
                "Dynamic KSGS filters require matching modeled or commuting "
                "derivative-commutation semantics."
            )
        if test_filter.repeated_filter_semantics == "unmodeled" or (
            test_filter.family == "explicit-filter"
            and test_filter.repeated_filter_semantics != "composed"
        ):
            raise ValueError(
                "Dynamic KSGS requires declared compatible repeated filtering."
            )
        self.test_filter = test_filter
        self.test_filter_scale_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dynamic-ksgs",
                "coefficients": coefficients.coefficient_id,
                "provenance": provenance.provenance_id,
                "test_filter": test_filter.filter_id,
                "test_filter_scale_ratio": ratio,
            }
        )

    def evaluate(self, state: KSGSState, inputs: object, /) -> KSGSResult:
        if not isinstance(inputs, DynamicKSGSInputs):
            raise TypeError("Dynamic KSGS requires DynamicKSGSInputs.")
        next_state = _dynamic_transition(state, inputs)
        return _evaluate(
            self,
            state,
            inputs.base,
            next_state,
            jnp.zeros_like(state.kinetic_energy),
            inputs.accept_update,
        )


class LowReKSGSPlan(AbstractKSGSPlan):
    """Low-Re damped KSGS equation with explicit viscous dissipation."""

    low_re_coefficients: LowReKSGSCoefficients

    def __init__(
        self,
        coefficients: KSGSCoefficients,
        low_re_coefficients: LowReKSGSCoefficients,
        provenance: LESParameterProvenance,
        /,
    ):
        if not isinstance(low_re_coefficients, LowReKSGSCoefficients):
            raise TypeError("low_re_coefficients must be LowReKSGSCoefficients.")
        _assign_plan(self, coefficients, provenance, "low-re-ksgs", low_re_coefficients)
        self.low_re_coefficients = low_re_coefficients

    def evaluate(self, state: KSGSState, inputs: object, /) -> KSGSResult:
        if not isinstance(inputs, LowReKSGSInputs):
            raise TypeError("Low-Re KSGS requires LowReKSGSInputs.")
        return _evaluate(
            self,
            state,
            inputs.base,
            state,
            jnp.zeros_like(state.kinetic_energy),
            None,
            low_re=(
                self.low_re_coefficients,
                inputs.wall_distance,
                inputs.sqrt_kinetic_energy_gradient,
            ),
        )


def replace_ksgs_kinetic_energy(
    state: KSGSState, kinetic_energy: ArrayLike, /
) -> KSGSState:
    """Replace backend-integrated ``k`` while preserving continuation exactly."""
    if not isinstance(state, KSGSState):
        raise TypeError("state must be KSGSState.")
    kinetic = _inexact(kinetic_energy)
    if kinetic.shape != state.kinetic_energy.shape:
        raise ValueError("Replacement KSGS kinetic energy must preserve state shape.")
    return KSGSState(
        kinetic,
        state.dynamic_numerator,
        state.dynamic_denominator,
        state.eddy_viscosity_coefficient,
        state.dynamic_updates,
    )


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.result_type(array, float))
    return array


def _assign_plan(
    plan: AbstractKSGSPlan,
    coefficients: KSGSCoefficients,
    provenance: LESParameterProvenance,
    kind: str,
    low_re: LowReKSGSCoefficients | None = None,
    /,
) -> None:
    if not isinstance(coefficients, KSGSCoefficients):
        raise TypeError("coefficients must be KSGSCoefficients.")
    if not isinstance(provenance, LESParameterProvenance):
        raise TypeError("provenance must be LESParameterProvenance.")
    plan.coefficients = coefficients
    plan.provenance = provenance
    payload: dict[str, str] = {
        "kind": kind,
        "coefficients": coefficients.coefficient_id,
        "provenance": provenance.provenance_id,
        "filter": provenance.resolved_filter.filter_id,
        "discretization": provenance.discretization_id,
        "regime": provenance.regime,
    }
    if low_re is not None:
        payload["low_re_coefficients"] = low_re.coefficient_id
    plan.plan_id = canonical_fingerprint(payload)


def _validated_transport_fields(
    state: KSGSState,
    filter_scale: LESFilterScale,
    molecular_kinematic_viscosity: ArrayLike,
    /,
) -> tuple[Array, Array, Array]:
    if not isinstance(state, KSGSState):
        raise TypeError("state must be KSGSState.")
    if not isinstance(filter_scale, LESFilterScale):
        raise TypeError("filter_scale must be LESFilterScale.")
    kinetic = _inexact(state.kinetic_energy)
    shape = kinetic.shape
    continuation = (
        state.dynamic_numerator,
        state.dynamic_denominator,
        state.eddy_viscosity_coefficient,
        state.dynamic_updates,
    )
    if any(jnp.asarray(value).shape != shape for value in continuation):
        raise ValueError("Every KSGS continuation field must have the shape of k.")
    width = _inexact(filter_scale.equivalent_width)
    viscosity = _inexact(molecular_kinematic_viscosity)
    if width.shape != shape or viscosity.shape != shape:
        raise ValueError("KSGS filter width and viscosity must have the shape of k.")
    kinetic = eqx.error_if(
        kinetic,
        jnp.any(kinetic < 0.0),
        "KSGS kinetic energy is negative; no flooring policy is enabled.",
    )
    width = eqx.error_if(
        width, jnp.any(width <= 0.0), "KSGS filter width must be positive."
    )
    viscosity = eqx.error_if(
        viscosity,
        jnp.any(viscosity < 0.0),
        "KSGS molecular kinematic viscosity must be nonnegative.",
    )
    return kinetic, width, viscosity


def _validated_fields(
    state: KSGSState, inputs: KSGSInputs, /
) -> tuple[Array, Array, Array, Array, Array]:
    kinetic, width, viscosity = _validated_transport_fields(
        state, inputs.filter_scale, inputs.molecular_kinematic_viscosity
    )
    shape = kinetic.shape
    if inputs.velocity_gradient.shape != shape + (3, 3):
        raise ValueError("KSGS velocity-gradient leading shape must match k.")
    diffusion = _inexact(inputs.diffusion_rate)
    if diffusion.shape != shape:
        raise ValueError("KSGS diffusion rate must have the shape of k.")
    return kinetic, inputs.velocity_gradient, width, viscosity, diffusion


def _dynamic_transition(state: KSGSState, inputs: DynamicKSGSInputs, /) -> KSGSState:
    kinetic, _, _, _, _ = _validated_fields(state, inputs.base)
    shape = kinetic.shape
    if inputs.leonard_stress.shape != shape + (3, 3):
        raise ValueError("Dynamic KSGS stress leading shape must match k.")
    weight = _inexact(inputs.averaging_weight)
    accepted = jnp.asarray(inputs.accept_update, dtype=bool)
    if weight.shape != shape or accepted.shape != shape:
        raise ValueError("Dynamic KSGS averaging fields must have the shape of k.")
    weight = eqx.error_if(
        weight,
        jnp.any((weight < 0.0) | (weight > 1.0)),
        "Dynamic KSGS averaging_weight must lie in [0, 1].",
    )
    sample_numerator = ein.contract(
        "...ij,...ij->...", inputs.leonard_stress, inputs.modeled_stress
    )
    sample_denominator = ein.contract(
        "...ij,...ij->...", inputs.modeled_stress, inputs.modeled_stress
    )
    averaged_numerator = (
        1.0 - weight
    ) * state.dynamic_numerator + weight * sample_numerator
    averaged_denominator = (
        1.0 - weight
    ) * state.dynamic_denominator + weight * sample_denominator
    numerator = jnp.where(accepted, averaged_numerator, state.dynamic_numerator)
    denominator = jnp.where(accepted, averaged_denominator, state.dynamic_denominator)
    has_information = denominator > 0.0
    ratio = numerator / jnp.where(
        has_information, denominator, jnp.ones_like(denominator)
    )
    coefficient = jnp.where(
        accepted & has_information, ratio, state.eddy_viscosity_coefficient
    )
    coefficient = eqx.error_if(
        coefficient,
        jnp.any(coefficient < 0.0),
        "Dynamic KSGS produced a negative dissipative coefficient.",
    )
    return KSGSState(
        kinetic_energy=state.kinetic_energy,
        dynamic_numerator=numerator,
        dynamic_denominator=denominator,
        eddy_viscosity_coefficient=coefficient,
        dynamic_updates=state.dynamic_updates + accepted.astype(jnp.int32),
    )


def _transport(
    plan: AbstractKSGSPlan,
    state: KSGSState,
    filter_scale: LESFilterScale,
    molecular_kinematic_viscosity: ArrayLike,
    /,
    *,
    wall_distance: ArrayLike | None = None,
) -> KSGSTransportResult:
    kinetic, width, molecular_viscosity = _validated_transport_fields(
        state, filter_scale, molecular_kinematic_viscosity
    )
    sqrt_kinetic = jnp.sqrt(kinetic)
    coefficient = (
        state.eddy_viscosity_coefficient
        if isinstance(plan, DynamicKSGSPlan)
        else jnp.full_like(kinetic, plan.coefficients.eddy_viscosity)
    )
    damping = jnp.ones_like(kinetic)
    distance_finite = jnp.ones_like(kinetic, dtype=bool)
    if isinstance(plan, LowReKSGSPlan):
        if wall_distance is None:
            raise ValueError("Low-Re KSGS transport requires resolved wall distance.")
        molecular_viscosity = eqx.error_if(
            molecular_viscosity,
            jnp.any(molecular_viscosity <= 0.0),
            "Low-Re KSGS requires positive molecular kinematic viscosity.",
        )
        distance = _inexact(wall_distance)
        if distance.shape != kinetic.shape:
            raise ValueError("Low-Re wall distance must have the shape of k.")
        distance = eqx.error_if(
            distance,
            jnp.any(distance < 0.0),
            "Low-Re wall distance must be nonnegative.",
        )
        wall_reynolds = distance * sqrt_kinetic / molecular_viscosity
        damping = 1.0 - jnp.exp(-plan.low_re_coefficients.damping * wall_reynolds)
        distance_finite = jnp.isfinite(distance)
    elif wall_distance is not None:
        raise ValueError("wall_distance is valid only for LowReKSGSPlan.")
    eddy_viscosity = coefficient * width * sqrt_kinetic * damping
    diffusivity = molecular_viscosity + plan.coefficients.diffusion * eddy_viscosity
    return KSGSTransportResult(
        eddy_viscosity=eddy_viscosity,
        diffusivity=diffusivity,
        kinetic_energy_nonnegative=kinetic >= 0.0,
        eddy_viscosity_nonnegative=eddy_viscosity >= 0.0,
        finite=(
            jnp.isfinite(kinetic)
            & jnp.isfinite(width)
            & jnp.isfinite(molecular_viscosity)
            & jnp.isfinite(coefficient)
            & distance_finite
            & jnp.isfinite(eddy_viscosity)
            & jnp.isfinite(diffusivity)
        ),
    )


def _evaluate(
    plan: AbstractKSGSPlan,
    state: KSGSState,
    inputs: KSGSInputs,
    next_state: KSGSState,
    buoyancy_frequency_squared: Array,
    dynamic_update_accepted: Array | None,
    *,
    low_re: tuple[LowReKSGSCoefficients, Array, Array] | None = None,
) -> KSGSResult:
    kinetic, gradient, width, molecular_viscosity, diffusion = _validated_fields(
        state, inputs
    )
    shape = kinetic.shape
    buoyancy_frequency = _inexact(buoyancy_frequency_squared)
    if buoyancy_frequency.shape != shape:
        raise ValueError("KSGS buoyancy-frequency leading shape must match k.")
    symmetric = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    trace = jnp.trace(symmetric, axis1=-2, axis2=-1)
    deviatoric = (
        symmetric - trace[..., None, None] * jnp.eye(3, dtype=gradient.dtype) / 3.0
    )
    strain_squared = ein.contract("...ij,...ij->...", deviatoric, deviatoric)
    sqrt_kinetic = jnp.sqrt(kinetic)
    transport = plan.transport(
        state,
        inputs.filter_scale,
        inputs.molecular_kinematic_viscosity,
        wall_distance=None if low_re is None else low_re[1],
    )
    coefficient = (
        state.eddy_viscosity_coefficient
        if isinstance(plan, DynamicKSGSPlan)
        else jnp.full_like(kinetic, plan.coefficients.eddy_viscosity)
    )
    low_re_dissipation = jnp.zeros_like(kinetic)
    if low_re is not None:
        low_re_coefficients, wall_distance, sqrt_gradient = low_re
        if wall_distance.shape != shape:
            raise ValueError("Low-Re wall-distance leading shape must match k.")
        if sqrt_gradient.shape != shape + (3,):
            raise ValueError("Low-Re gradient leading shape must match k.")
        gradient_squared = ein.contract("...i,...i->...", sqrt_gradient, sqrt_gradient)
        low_re_dissipation = (
            low_re_coefficients.viscous_dissipation
            * molecular_viscosity
            * gradient_squared
        )
    wall_distance_finite = (
        jnp.ones_like(kinetic, dtype=bool) if low_re is None else jnp.isfinite(low_re[1])
    )
    eddy_viscosity = transport.eddy_viscosity
    diffusivity = transport.diffusivity
    raw_production = 2.0 * eddy_viscosity * strain_squared
    dissipation = plan.coefficients.dissipation * kinetic * sqrt_kinetic / width
    production_ceiling = plan.coefficients.production_limit * dissipation
    production = jnp.minimum(raw_production, production_ceiling)
    buoyancy = -plan.coefficients.buoyancy * eddy_viscosity * buoyancy_frequency
    # The supplied rate is the backend operator applied with ``diffusivity``.
    rhs = production - dissipation + diffusion + buoyancy - low_re_dissipation
    contributions = KSGSContributions(
        raw_production=raw_production,
        production=production,
        production_limit_reduction=raw_production - production,
        dissipation=dissipation,
        diffusion=diffusion,
        buoyancy=buoyancy,
        low_re_dissipation=low_re_dissipation,
        rhs=rhs,
    )
    accepted = (
        jnp.zeros_like(kinetic, dtype=bool)
        if dynamic_update_accepted is None
        else jnp.asarray(dynamic_update_accepted, dtype=bool)
    )
    finite = jnp.ones_like(kinetic, dtype=bool)
    for value in (
        kinetic,
        gradient,
        width,
        molecular_viscosity,
        diffusion,
        buoyancy_frequency,
        next_state.dynamic_numerator,
        next_state.dynamic_denominator,
        coefficient,
        eddy_viscosity,
        diffusivity,
        raw_production,
        production,
        dissipation,
        buoyancy,
        low_re_dissipation,
        rhs,
    ):
        axes = tuple(range(len(shape), value.ndim))
        value_finite = (
            jnp.all(jnp.isfinite(value), axis=axes) if axes else jnp.isfinite(value)
        )
        finite = finite & value_finite
    finite = finite & wall_distance_finite
    evidence = KSGSEvidence(
        production_limited=raw_production > production,
        kinetic_energy_nonnegative=kinetic >= 0.0,
        eddy_viscosity_nonnegative=eddy_viscosity >= 0.0,
        production_nonnegative=production >= 0.0,
        dissipation_nonnegative=(dissipation >= 0.0) & (low_re_dissipation >= 0.0),
        finite=finite,
        dynamic_update_accepted=accepted,
        dynamic_numerator=next_state.dynamic_numerator,
        dynamic_denominator=next_state.dynamic_denominator,
    )
    return KSGSResult(
        state=next_state,
        eddy_viscosity=eddy_viscosity,
        diffusivity=diffusivity,
        contributions=contributions,
        evidence=evidence,
        plan_id=plan.plan_id,
    )


__all__ = [
    "AbstractKSGSPlan",
    "BuoyancyKSGSInputs",
    "BuoyancyKSGSPlan",
    "DynamicKSGSInputs",
    "DynamicKSGSPlan",
    "KSGSCoefficients",
    "KSGSContributions",
    "KSGSEvidence",
    "KSGSInputs",
    "KSGSResult",
    "KSGSState",
    "KSGSTransportResult",
    "LowReKSGSCoefficients",
    "LowReKSGSInputs",
    "LowReKSGSPlan",
    "StaticKSGSPlan",
    "replace_ksgs_kinetic_energy",
]
