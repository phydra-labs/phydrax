#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    matrix_phi1_action,
    matrix_phi2_action,
    MatrixFunctionPolicy,
    MatrixFunctionResult,
)
from ._chemical_mechanism import PreparedChemicalMechanism
from ._chemical_rates import (
    AbstractChemicalRatePlan,
    ArrheniusRatePlan,
    ButlerVolmerRatePlan,
    ChebyshevRatePlan,
    ChemicalRateRuntime,
    LindemannRatePlan,
    PhotolysisRatePlan,
    PLogRatePlan,
    StickingRatePlan,
    SurfaceCoverageRatePlan,
    ThirdBodyRatePlan,
    TroeRatePlan,
)


class ChemicalReactionDirection(StrEnum):
    FORWARD = "forward"
    REVERSE = "reverse"


class ChemicalConditionalAffineStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    INVALID_RATES = 2
    MATRIX_ACTION_FAILED = 3
    NONFINITE_STATE = 4
    NEGATIVE_STATE = 5
    INVARIANT_FAILURE = 6


class ChemicalAffinePivot(StrictModule):
    reaction_index: int = eqx.field(static=True)
    direction: ChemicalReactionDirection = eqx.field(static=True)
    species: str = eqx.field(static=True)

    def __init__(
        self,
        reaction_index: int,
        direction: ChemicalReactionDirection | str,
        species: str,
        /,
    ):
        reaction = int(reaction_index)
        direction_ = ChemicalReactionDirection(direction)
        species_ = str(species)
        if reaction < 0:
            raise ValueError("reaction_index must be non-negative.")
        if not species_:
            raise ValueError("species must be non-empty.")
        self.reaction_index = reaction
        self.direction = direction_
        self.species = species_


class ChemicalConditionalAffinePlan(StrictModule):
    affine_species: tuple[str, ...] = eqx.field(static=True)
    driver_species: tuple[str, ...] = eqx.field(static=True)
    pivots: tuple[ChemicalAffinePivot, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        affine_species,
        driver_species=(),
        /,
        *,
        pivots=(),
        plan_id: str | None = None,
    ):
        affine = tuple(str(value) for value in affine_species)
        drivers = tuple(str(value) for value in driver_species)
        pivots_ = tuple(pivots)
        if not affine or any(not value for value in affine):
            raise ValueError("affine_species must contain non-empty names.")
        if any(not value for value in drivers):
            raise ValueError("driver_species names must be non-empty.")
        if len(set(affine)) != len(affine) or len(set(drivers)) != len(drivers):
            raise ValueError("Affine and driver species must each be unique.")
        if any(not isinstance(value, ChemicalAffinePivot) for value in pivots_):
            raise TypeError("pivots must contain ChemicalAffinePivot values.")
        keys = tuple((value.reaction_index, value.direction) for value in pivots_)
        if len(set(keys)) != len(keys):
            raise ValueError("Only one explicit pivot may be declared per direction.")
        generated = canonical_fingerprint(
            {
                "kind": "chemical-conditional-affine-plan",
                "affine_species": affine,
                "driver_species": drivers,
                "pivots": [
                    (value.reaction_index, value.direction.value, value.species)
                    for value in pivots_
                ],
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.affine_species = affine
        self.driver_species = drivers
        self.pivots = pivots_
        self.plan_id = identifier

    def analyze(
        self,
        mechanism: PreparedChemicalMechanism,
        /,
    ) -> ChemicalConditionalAffineCertificate:
        return _analyze_conditional_affinity(mechanism, self)

    def prepare(
        self,
        mechanism: PreparedChemicalMechanism,
        /,
    ) -> PreparedChemicalConditionalAffine:
        certificate = self.analyze(mechanism)
        if not bool(np.asarray(certificate.certified)):
            details = "; ".join(certificate.rejection_reasons)
            raise ValueError(
                f"Chemical conditional-affinity certification failed: {details}"
            )
        return PreparedChemicalConditionalAffine(mechanism, self, certificate)


class ChemicalConditionalAffineCertificate(StrictModule):
    channel_reaction_indices: Array
    channel_is_reverse: Array
    pivot_affine_indices: Array
    eligible_channels: Array
    affine_species: tuple[str, ...] = eqx.field(static=True)
    driver_species: tuple[str, ...] = eqx.field(static=True)
    pivot_species: tuple[str | None, ...] = eqx.field(static=True)
    rate_dependency_species: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    rejection_reasons: tuple[str, ...] = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    certified: Array


class ChemicalConditionalAffineDrivers(StrictModule):
    species_concentrations: Array
    temperature: Array
    pressure: Array
    runtime: ChemicalRateRuntime

    def __init__(
        self,
        species_concentrations: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        runtime: ChemicalRateRuntime | None = None,
    ):
        species = jnp.asarray(species_concentrations)
        if species.ndim < 1:
            raise ValueError("species_concentrations must have a trailing driver axis.")
        if not jnp.issubdtype(species.dtype, jnp.inexact):
            species = species.astype(float)
        temperature_ = jnp.asarray(temperature, dtype=species.dtype)
        pressure_ = jnp.asarray(pressure, dtype=species.dtype)
        if (
            temperature_.shape != species.shape[:-1]
            or pressure_.shape != species.shape[:-1]
        ):
            raise ValueError(
                "temperature and pressure must match driver leading dimensions."
            )
        runtime_ = (
            ChemicalRateRuntime(
                jnp.zeros((0,), dtype=species.dtype),
                jnp.asarray(0.0, dtype=species.dtype),
            )
            if runtime is None
            else runtime
        )
        if not isinstance(runtime_, ChemicalRateRuntime):
            raise TypeError("runtime must be ChemicalRateRuntime.")
        if runtime_.overpotential.shape not in ((), temperature_.shape):
            raise ValueError(
                "Runtime overpotential must be scalar or match driver shape."
            )
        self.species_concentrations = species
        self.temperature = temperature_
        self.pressure = pressure_
        self.runtime = runtime_


class ChemicalConditionalAffineAssembly(StrictModule):
    directional_coefficients: Array
    reaction_multiplier: Array
    operator: Array
    forcing: Array
    forward_rate_constants: Array
    reverse_rate_constants: Array
    input_valid: Array
    rates_valid: Array
    successful: Array
    certificate_id: str = eqx.field(static=True)


class ChemicalConditionalAffineResult(StrictModule):
    candidate_state: Array
    directional_extent: Array
    affine_candidate: Array
    status: Array
    successful: Array
    assembly: ChemicalConditionalAffineAssembly
    phi1_state_action: MatrixFunctionResult
    phi2_forcing_action: MatrixFunctionResult
    affine_consistency_residual: Array
    element_residual: Array
    charge_residual: Array
    minimum_species: Array
    certificate_id: str = eqx.field(static=True)


class PreparedChemicalConditionalAffine(StrictModule):
    mechanism: PreparedChemicalMechanism
    plan: ChemicalConditionalAffinePlan
    certificate: ChemicalConditionalAffineCertificate
    affine_indices: Array
    driver_indices: Array
    directional_stoichiometry: Array
    directional_orders: Array
    channel_reaction_indices: Array
    channel_is_reverse: Array
    pivot_affine_indices: Array
    pivot_incidence: Array
    source_mask: Array
    coefficient_orders: Array
    has_forcing: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        plan: ChemicalConditionalAffinePlan,
        certificate: ChemicalConditionalAffineCertificate,
        /,
    ):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        if not isinstance(plan, ChemicalConditionalAffinePlan):
            raise TypeError("plan must be ChemicalConditionalAffinePlan.")
        if not isinstance(certificate, ChemicalConditionalAffineCertificate):
            raise TypeError("certificate must be ChemicalConditionalAffineCertificate.")
        if certificate.mechanism_id != mechanism.mechanism_id:
            raise ValueError("Certificate and mechanism identities differ.")
        if certificate.plan_id != plan.plan_id:
            raise ValueError("Certificate and plan identities differ.")
        if not bool(np.asarray(certificate.certified)):
            raise ValueError(
                "A prepared conditional-affine mechanism requires certification."
            )
        species_index = {
            name: index for index, name in enumerate(mechanism.schema.species_names)
        }
        affine_indices = np.asarray(
            [species_index[name] for name in plan.affine_species], dtype=np.int32
        )
        driver_indices = np.asarray(
            [species_index[name] for name in plan.driver_species], dtype=np.int32
        )
        reaction_indices, reverse_flags, stoichiometry, orders = _directional_arrays(
            mechanism
        )
        pivot_affine = np.asarray(certificate.pivot_affine_indices, dtype=np.int32)
        pivot_incidence = np.zeros((pivot_affine.size, affine_indices.size), dtype=float)
        valid_pivot = pivot_affine >= 0
        pivot_incidence[np.nonzero(valid_pivot)[0], pivot_affine[valid_pivot]] = 1.0
        source_mask = (~valid_pivot).astype(float)
        coefficient_orders = orders.copy()
        for channel, local_index in enumerate(pivot_affine):
            if local_index >= 0:
                coefficient_orders[channel, affine_indices[local_index]] = 0.0
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-chemical-conditional-affine",
                "mechanism": mechanism.mechanism_id,
                "certificate": certificate.certificate_id,
                "affine_indices": affine_indices.tolist(),
                "driver_indices": driver_indices.tolist(),
                "directional_stoichiometry": array_tree_fingerprint(stoichiometry),
                "coefficient_orders": array_tree_fingerprint(coefficient_orders),
            }
        )
        dtype = mechanism.net_stoichiometry.dtype
        self.mechanism = mechanism
        self.plan = plan
        self.certificate = certificate
        self.affine_indices = jnp.asarray(affine_indices)
        self.driver_indices = jnp.asarray(driver_indices)
        self.directional_stoichiometry = jnp.asarray(stoichiometry, dtype=dtype)
        self.directional_orders = jnp.asarray(orders, dtype=dtype)
        self.channel_reaction_indices = jnp.asarray(reaction_indices)
        self.channel_is_reverse = jnp.asarray(reverse_flags)
        self.pivot_affine_indices = jnp.asarray(pivot_affine)
        self.pivot_incidence = jnp.asarray(pivot_incidence, dtype=dtype)
        self.source_mask = jnp.asarray(source_mask, dtype=dtype)
        self.coefficient_orders = jnp.asarray(coefficient_orders, dtype=dtype)
        self.has_forcing = bool(np.any(source_mask))
        self.prepared_id = identifier

    @property
    def affine_size(self) -> int:
        return len(self.plan.affine_species)

    @property
    def driver_size(self) -> int:
        return len(self.plan.driver_species)

    @property
    def channel_count(self) -> int:
        return int(self.channel_reaction_indices.shape[0])

    def assemble(
        self,
        drivers: ChemicalConditionalAffineDrivers,
        /,
        *,
        reaction_multiplier: ArrayLike | None = None,
    ) -> ChemicalConditionalAffineAssembly:
        if not isinstance(drivers, ChemicalConditionalAffineDrivers):
            raise TypeError("drivers must be ChemicalConditionalAffineDrivers.")
        driver_values = drivers.species_concentrations
        if driver_values.shape[-1] != self.driver_size:
            raise ValueError(
                f"Driver species axis must have size {self.driver_size}; "
                f"got {driver_values.shape[-1]}."
            )
        batch_shape = driver_values.shape[:-1]
        multiplier = (
            jnp.ones(
                batch_shape + (self.mechanism.reaction_count,), dtype=driver_values.dtype
            )
            if reaction_multiplier is None
            else jnp.asarray(reaction_multiplier, dtype=driver_values.dtype)
        )
        expected_multiplier = batch_shape + (self.mechanism.reaction_count,)
        if multiplier.shape != expected_multiplier:
            raise ValueError(
                f"reaction_multiplier must have shape {expected_multiplier}; "
                f"got {multiplier.shape}."
            )
        full_driver = jnp.zeros(
            batch_shape + (self.mechanism.schema.species_count,),
            dtype=driver_values.dtype,
        )
        full_driver = full_driver.at[..., self.driver_indices].set(driver_values)
        base = self.mechanism.evaluate(
            full_driver,
            drivers.temperature,
            drivers.pressure,
            runtime=drivers.runtime,
        )
        reaction_indices = self.channel_reaction_indices
        forward = jnp.take(base.forward_rate_constants, reaction_indices, axis=-1)
        reverse = jnp.take(base.reverse_rate_constants, reaction_indices, axis=-1)
        rate_constants = jnp.where(self.channel_is_reverse, reverse, forward)
        powered = jnp.where(
            self.coefficient_orders > 0.0,
            full_driver[..., None, :] ** self.coefficient_orders,
            1.0,
        )
        mass_action = jnp.prod(powered, axis=-1)
        channel_multiplier = jnp.take(multiplier, reaction_indices, axis=-1)
        coefficients = rate_constants * mass_action * channel_multiplier
        affine_stoichiometry = self.directional_stoichiometry[:, self.affine_indices]
        operator = contract(
            "ci,...c,cj->...ij",
            affine_stoichiometry,
            coefficients,
            self.pivot_incidence,
        )
        forcing = contract(
            "ci,...c,c->...i",
            affine_stoichiometry,
            coefficients,
            self.source_mask,
        )
        input_valid = (
            jnp.all(jnp.isfinite(driver_values), axis=-1)
            & jnp.all(driver_values >= 0.0, axis=-1)
            & jnp.isfinite(drivers.temperature)
            & (drivers.temperature > 0.0)
            & jnp.isfinite(drivers.pressure)
            & (drivers.pressure > 0.0)
            & jnp.all(jnp.isfinite(multiplier) & (multiplier > 0.0), axis=-1)
        )
        rates_valid = (
            jnp.all(jnp.isfinite(rate_constants) & (rate_constants >= 0.0), axis=-1)
            & jnp.all(jnp.isfinite(coefficients) & (coefficients >= 0.0), axis=-1)
            & jnp.all(jnp.isfinite(operator), axis=(-2, -1))
            & jnp.all(jnp.isfinite(forcing), axis=-1)
        )
        return ChemicalConditionalAffineAssembly(
            coefficients,
            multiplier,
            operator,
            forcing,
            base.forward_rate_constants,
            base.reverse_rate_constants,
            input_valid,
            rates_valid,
            base.successful & input_valid & rates_valid,
            self.certificate.certificate_id,
        )

    def advance(
        self,
        state: ArrayLike,
        drivers: ChemicalConditionalAffineDrivers,
        duration: ArrayLike,
        /,
        *,
        reaction_multiplier: ArrayLike | None = None,
        policy: MatrixFunctionPolicy | None = None,
    ) -> ChemicalConditionalAffineResult:
        state_ = jnp.asarray(state)
        if state_.ndim < 1 or state_.shape[-1] != self.mechanism.schema.species_count:
            raise ValueError("state must end in the complete mechanism species axis.")
        batch_shape = state_.shape[:-1]
        if drivers.species_concentrations.shape[:-1] != batch_shape:
            raise ValueError("state and driver leading dimensions must match exactly.")
        duration_ = jnp.asarray(duration, dtype=state_.dtype)
        if duration_.shape == () and batch_shape:
            duration_ = jnp.broadcast_to(duration_, batch_shape)
        if duration_.shape != batch_shape:
            raise ValueError(
                f"duration must be scalar or have shape {batch_shape}; got {duration_.shape}."
            )
        assembly = self.assemble(
            drivers,
            reaction_multiplier=reaction_multiplier,
        )
        operator = DenseLinearOperator(assembly.operator)
        affine_initial = state_[..., self.affine_indices]
        phi1_state = matrix_phi1_action(
            operator,
            affine_initial,
            duration_,
            policy=policy,
        )
        if self.has_forcing:
            phi2_forcing = matrix_phi2_action(
                operator,
                assembly.forcing,
                duration_,
                policy=policy,
            )
        else:
            zero = jnp.zeros(batch_shape, dtype=state_.real.dtype)
            phi2_forcing = MatrixFunctionResult(
                value=jnp.zeros_like(assembly.forcing),
                error_estimate=zero,
                residual_estimate=zero,
                converged=jnp.ones(batch_shape, dtype=bool),
                effective_dimension=jnp.zeros(batch_shape, dtype=jnp.int32),
                matvec_count=jnp.zeros(batch_shape, dtype=jnp.int32),
                breakdown_status=jnp.zeros(batch_shape, dtype=jnp.int32),
                method="not-required",
                kind="phi2",
                provenance="forcing is structurally zero",
            )
        safe_pivots = jnp.maximum(self.pivot_affine_indices, 0)
        pivot_mask = (self.pivot_affine_indices >= 0).astype(state_.dtype)
        phi1_at_pivot = jnp.take(phi1_state.value, safe_pivots, axis=-1)
        phi2_at_pivot = jnp.take(phi2_forcing.value, safe_pivots, axis=-1)
        h = duration_[..., None]
        extent = (
            h * assembly.directional_coefficients * pivot_mask * phi1_at_pivot
            + h**2 * assembly.directional_coefficients * pivot_mask * phi2_at_pivot
            + h
            * assembly.directional_coefficients
            * self.source_mask.astype(state_.dtype)
        )
        increment = contract("...c,cs->...s", extent, self.directional_stoichiometry)
        candidate = state_ + increment
        candidate = jnp.where((duration_ == 0)[..., None], state_, candidate)
        state_integral = (
            duration_[..., None] * phi1_state.value
            + duration_[..., None] ** 2 * phi2_forcing.value
        )
        affine_candidate = (
            affine_initial
            + contract("...ij,...j->...i", assembly.operator, state_integral)
            + duration_[..., None] * assembly.forcing
        )
        affine_candidate = jnp.where(
            (duration_ == 0)[..., None], affine_initial, affine_candidate
        )
        affine_difference = candidate[..., self.affine_indices] - affine_candidate
        affine_residual = jnp.max(jnp.abs(affine_difference), axis=-1)
        element_residual = contract(
            "es,...s->...e", self.mechanism.schema.element_composition, increment
        )
        charge_residual = contract(
            "s,...s->...", self.mechanism.schema.charges, increment
        )
        minimum_species = jnp.min(candidate, axis=-1)
        scale = jnp.maximum(jnp.max(jnp.abs(state_), axis=-1), 1.0)
        tolerance = 4096.0 * jnp.finfo(state_.dtype).eps * scale
        input_valid = (
            jnp.all(jnp.isfinite(state_), axis=-1)
            & jnp.all(state_ >= 0.0, axis=-1)
            & jnp.isfinite(duration_)
            & (duration_ >= 0.0)
            & assembly.input_valid
        )
        actions_valid = phi1_state.converged & phi2_forcing.converged
        finite_state = jnp.all(jnp.isfinite(candidate), axis=-1)
        nonnegative = jnp.all(candidate >= 0.0, axis=-1)
        invariant_valid = (
            jnp.all(jnp.abs(element_residual) <= tolerance[..., None], axis=-1)
            & (jnp.abs(charge_residual) <= tolerance)
            & (affine_residual <= tolerance)
        )
        status = jnp.full(
            batch_shape, int(ChemicalConditionalAffineStatus.SUCCESS), dtype=jnp.int32
        )
        status = jnp.where(
            ~input_valid,
            int(ChemicalConditionalAffineStatus.INVALID_INPUT),
            status,
        )
        status = jnp.where(
            input_valid & ~assembly.rates_valid,
            int(ChemicalConditionalAffineStatus.INVALID_RATES),
            status,
        )
        status = jnp.where(
            input_valid & assembly.rates_valid & ~actions_valid,
            int(ChemicalConditionalAffineStatus.MATRIX_ACTION_FAILED),
            status,
        )
        status = jnp.where(
            input_valid & assembly.rates_valid & actions_valid & ~finite_state,
            int(ChemicalConditionalAffineStatus.NONFINITE_STATE),
            status,
        )
        status = jnp.where(
            input_valid
            & assembly.rates_valid
            & actions_valid
            & finite_state
            & ~nonnegative,
            int(ChemicalConditionalAffineStatus.NEGATIVE_STATE),
            status,
        )
        status = jnp.where(
            input_valid
            & assembly.rates_valid
            & actions_valid
            & finite_state
            & nonnegative
            & ~invariant_valid,
            int(ChemicalConditionalAffineStatus.INVARIANT_FAILURE),
            status,
        )
        successful = status == int(ChemicalConditionalAffineStatus.SUCCESS)
        return ChemicalConditionalAffineResult(
            candidate,
            extent,
            affine_candidate,
            status,
            successful,
            assembly,
            phi1_state,
            phi2_forcing,
            affine_residual,
            element_residual,
            charge_residual,
            minimum_species,
            self.certificate.certificate_id,
        )


def _rate_concentration_dependencies(
    rate: AbstractChemicalRatePlan,
    /,
) -> tuple[int, ...]:
    if isinstance(
        rate,
        (
            ArrheniusRatePlan,
            ButlerVolmerRatePlan,
            ChebyshevRatePlan,
            PhotolysisRatePlan,
            PLogRatePlan,
            StickingRatePlan,
        ),
    ):
        return ()
    if isinstance(rate, (ThirdBodyRatePlan, LindemannRatePlan, TroeRatePlan)):
        efficiencies = np.asarray(rate.efficiencies)
        return tuple(int(value) for value in np.flatnonzero(efficiencies != 0.0))
    if isinstance(rate, SurfaceCoverageRatePlan):
        return (rate.species_index,)
    raise TypeError(
        f"Unsupported chemical rate plan for dependency analysis: {type(rate).__name__}."
    )


def _directional_arrays(
    mechanism: PreparedChemicalMechanism,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reaction_indices = []
    reverse_flags = []
    stoichiometry = []
    orders = []
    net = np.asarray(mechanism.net_stoichiometry)
    forward_orders = np.asarray(mechanism.forward_orders)
    reverse_orders = np.asarray(mechanism.product_stoichiometry)
    for reaction_index, reaction in enumerate(mechanism.reactions):
        reaction_indices.append(reaction_index)
        reverse_flags.append(False)
        stoichiometry.append(net[reaction_index])
        orders.append(forward_orders[reaction_index])
        if reaction.reverse_rate is not None or reaction.thermodynamic_reversible:
            reaction_indices.append(reaction_index)
            reverse_flags.append(True)
            stoichiometry.append(-net[reaction_index])
            orders.append(reverse_orders[reaction_index])
    return (
        np.asarray(reaction_indices, dtype=np.int32),
        np.asarray(reverse_flags, dtype=bool),
        np.asarray(stoichiometry),
        np.asarray(orders),
    )


def _analyze_conditional_affinity(
    mechanism: PreparedChemicalMechanism,
    plan: ChemicalConditionalAffinePlan,
    /,
) -> ChemicalConditionalAffineCertificate:
    if not isinstance(mechanism, PreparedChemicalMechanism):
        raise TypeError("mechanism must be PreparedChemicalMechanism.")
    species_names = mechanism.schema.species_names
    species_index = {name: index for index, name in enumerate(species_names)}
    unknown_affine = tuple(
        name for name in plan.affine_species if name not in species_index
    )
    unknown_drivers = tuple(
        name for name in plan.driver_species if name not in species_index
    )
    if unknown_affine or unknown_drivers:
        raise ValueError(
            f"Unknown species in conditional-affine plan: {unknown_affine + unknown_drivers}."
        )
    for pivot in plan.pivots:
        if pivot.reaction_index >= mechanism.reaction_count:
            raise ValueError("Explicit pivot reaction index exceeds mechanism reactions.")
        if pivot.species not in species_index:
            raise ValueError(
                f"Explicit pivot references unknown species {pivot.species!r}."
            )
    affine_indices = tuple(species_index[name] for name in plan.affine_species)
    affine_local = {index: local for local, index in enumerate(affine_indices)}
    driver_indices = {species_index[name] for name in plan.driver_species}
    explicit = {
        (
            value.reaction_index,
            value.direction is ChemicalReactionDirection.REVERSE,
        ): value
        for value in plan.pivots
    }
    reaction_indices, reverse_flags, _, orders = _directional_arrays(mechanism)
    pivots = []
    pivot_species = []
    dependencies = []
    eligible = []
    reasons = []
    seen_explicit = set()
    for channel, (reaction_index, reverse) in enumerate(
        zip(reaction_indices, reverse_flags, strict=True)
    ):
        reaction = mechanism.reactions[int(reaction_index)]
        rate = (
            reaction.reverse_rate
            if bool(reverse) and reaction.reverse_rate is not None
            else reaction.forward_rate
        )
        assert rate is not None
        dependency_indices = _rate_concentration_dependencies(rate)
        dependencies.append(tuple(species_names[index] for index in dependency_indices))
        channel_reasons = []
        missing_dependencies = tuple(
            species_names[index]
            for index in dependency_indices
            if index not in driver_indices
        )
        if missing_dependencies:
            channel_reasons.append(
                "rate-plan concentration dependencies are not drivers: "
                + ", ".join(missing_dependencies)
            )
        key = (int(reaction_index), bool(reverse))
        explicit_pivot = explicit.get(key)
        if explicit_pivot is not None:
            seen_explicit.add(key)
            pivot_index = species_index[explicit_pivot.species]
            candidates = (pivot_index,)
        else:
            candidates = tuple(
                index
                for index in affine_indices
                if orders[channel, index] == 1.0
                and all(
                    value == 0.0 or other == index or other in driver_indices
                    for other, value in enumerate(orders[channel])
                )
            )
        selected: int | None = None
        for candidate in candidates:
            if candidate not in affine_local:
                continue
            if orders[channel, candidate] != 1.0:
                continue
            if all(
                value == 0.0 or index == candidate or index in driver_indices
                for index, value in enumerate(orders[channel])
            ):
                selected = candidate
                break
        if explicit_pivot is not None and selected is None:
            channel_reasons.append(
                f"explicit pivot {explicit_pivot.species!r} is not an affine unit-order factor"
            )
        if selected is None:
            missing_factors = tuple(
                species_names[index]
                for index, value in enumerate(orders[channel])
                if value > 0.0 and index not in driver_indices
            )
            if missing_factors:
                channel_reasons.append(
                    "mass-action factors are neither a unit affine pivot nor drivers: "
                    + ", ".join(missing_factors)
                )
        local_pivot = -1 if selected is None else affine_local[selected]
        pivots.append(local_pivot)
        pivot_species.append(None if selected is None else species_names[selected])
        eligible.append(not channel_reasons)
        if channel_reasons:
            direction = "reverse" if reverse else "forward"
            reasons.append(
                f"reaction {int(reaction_index)} {direction}: "
                + "; ".join(channel_reasons)
            )
    unused_explicit = tuple(key for key in explicit if key not in seen_explicit)
    if unused_explicit:
        raise ValueError(
            f"Explicit pivots target inactive directions: {unused_explicit}."
        )
    eligible_array = np.asarray(eligible, dtype=bool)
    certificate_id = canonical_fingerprint(
        {
            "kind": "chemical-conditional-affinity-certificate",
            "mechanism": mechanism.mechanism_id,
            "plan": plan.plan_id,
            "reaction_indices": reaction_indices.tolist(),
            "reverse": reverse_flags.tolist(),
            "pivots": pivots,
            "dependencies": dependencies,
            "eligible": eligible,
        }
    )
    return ChemicalConditionalAffineCertificate(
        jnp.asarray(reaction_indices),
        jnp.asarray(reverse_flags),
        jnp.asarray(pivots, dtype=jnp.int32),
        jnp.asarray(eligible_array),
        plan.affine_species,
        plan.driver_species,
        tuple(pivot_species),
        tuple(dependencies),
        tuple(reasons),
        mechanism.mechanism_id,
        plan.plan_id,
        certificate_id,
        jnp.asarray(bool(np.all(eligible_array))),
    )


__all__ = [
    "ChemicalAffinePivot",
    "ChemicalConditionalAffineAssembly",
    "ChemicalConditionalAffineCertificate",
    "ChemicalConditionalAffineDrivers",
    "ChemicalConditionalAffinePlan",
    "ChemicalConditionalAffineResult",
    "ChemicalConditionalAffineStatus",
    "ChemicalReactionDirection",
    "PreparedChemicalConditionalAffine",
]
