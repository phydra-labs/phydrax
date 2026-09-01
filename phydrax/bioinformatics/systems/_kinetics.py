#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics import ContinuousSystem, EvolutionStep, StateLayout
from ...solver import DiffraxEvolution
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._network import StoichiometricNetwork, UnitDimension
from ._stoichiometry import conservation_analysis


class RateLawKind(StrEnum):
    """Closed, JAX-native kinetic laws supported without expression runtimes."""

    MASS_ACTION = "mass_action"
    REVERSIBLE_MASS_ACTION = "reversible_mass_action"
    MICHAELIS_MENTEN = "michaelis_menten"
    HILL = "hill"
    CONSTANT = "constant"


class KineticStatus(IntEnum):
    """Portable kinetic simulation and objective status."""

    SUCCESS = 0
    NONFINITE = 1
    NEGATIVE_CONCENTRATION = 2
    DYNAMICS_FAILURE = 3
    SHAPE_MISMATCH = 4


class KineticReaction(StrictModule):
    """One parameterized reaction rate linked to a compiled network reaction."""

    reaction_index: Array
    reactant_indices: Array
    reactant_orders: Array
    product_indices: Array
    product_orders: Array
    parameters: Array
    rate_scale: Array
    rate_law: RateLawKind = eqx.field(static=True)
    rate_unit: UnitDimension = eqx.field(static=True)
    kinetic_id: str = eqx.field(static=True)

    def __init__(
        self,
        reaction_index: int | ArrayLike,
        reactant_indices: ArrayLike,
        reactant_orders: ArrayLike,
        parameters: ArrayLike,
        /,
        *,
        rate_law: RateLawKind = RateLawKind.MASS_ACTION,
        product_indices: ArrayLike | None = None,
        product_orders: ArrayLike | None = None,
        rate_unit: UnitDimension,
        rate_scale: ArrayLike = 1.0,
        kinetic_id: str,
    ):
        reaction_index_ = jnp.asarray(reaction_index, dtype=jnp.int32)
        reactants = jnp.asarray(reactant_indices, dtype=jnp.int32).reshape((-1,))
        orders = jnp.asarray(reactant_orders)
        products = (
            jnp.zeros((0,), dtype=jnp.int32)
            if product_indices is None
            else jnp.asarray(product_indices, dtype=jnp.int32).reshape((-1,))
        )
        reverse_orders = (
            jnp.zeros((0,), dtype=jnp.float32)
            if product_orders is None
            else jnp.asarray(product_orders)
        )
        parameters_ = jnp.asarray(parameters)
        if reaction_index_.shape != ():
            raise ValueError("reaction_index must be scalar.")
        if orders.shape != reactants.shape:
            raise ValueError("reactant_orders must match reactant_indices.")
        if reverse_orders.shape != products.shape:
            raise ValueError("product_orders must match product_indices.")
        if parameters_.ndim != 1:
            raise ValueError("parameters must be one-dimensional.")
        dtype = jnp.result_type(
            orders.dtype, reverse_orders.dtype, parameters_.dtype, jnp.float32
        )
        orders = orders.astype(dtype)
        reverse_orders = reverse_orders.astype(dtype)
        parameters_ = parameters_.astype(dtype)
        orders = eqx.error_if(
            orders,
            jnp.any(~jnp.isfinite(orders) | (orders < 0.0)),
            "Kinetic orders must be finite and non-negative.",
        )
        reverse_orders = eqx.error_if(
            reverse_orders,
            jnp.any(~jnp.isfinite(reverse_orders) | (reverse_orders < 0.0)),
            "Reverse kinetic orders must be finite and non-negative.",
        )
        parameters_ = eqx.error_if(
            parameters_,
            jnp.any(~jnp.isfinite(parameters_) | (parameters_ < 0.0)),
            "Kinetic parameters must be finite and non-negative.",
        )
        rate_scale_ = jnp.asarray(rate_scale, dtype=dtype)
        if rate_scale_.shape != ():
            raise ValueError("rate_scale must be scalar.")
        rate_scale_ = eqx.error_if(
            rate_scale_,
            ~jnp.isfinite(rate_scale_) | (rate_scale_ <= 0.0),
            "rate_scale must be finite and positive.",
        )
        kind = RateLawKind(rate_law)
        expected_parameters = {
            RateLawKind.MASS_ACTION: 1,
            RateLawKind.REVERSIBLE_MASS_ACTION: 2,
            RateLawKind.MICHAELIS_MENTEN: 2,
            RateLawKind.HILL: 3,
            RateLawKind.CONSTANT: 1,
        }[kind]
        if parameters_.shape != (expected_parameters,):
            raise ValueError(
                f"{kind.value} requires {expected_parameters} kinetic parameters."
            )
        if kind in (
            RateLawKind.MICHAELIS_MENTEN,
            RateLawKind.HILL,
        ) and reactants.shape != (1,):
            raise ValueError(f"{kind.value} requires exactly one substrate.")
        if kind is RateLawKind.REVERSIBLE_MASS_ACTION and not products.size:
            raise ValueError(
                "reversible_mass_action requires at least one product order."
            )
        if kind is RateLawKind.MICHAELIS_MENTEN:
            parameters_ = eqx.error_if(
                parameters_,
                parameters_[1] <= 0.0,
                "Michaelis-Menten half-saturation must be positive.",
            )
        if kind is RateLawKind.HILL:
            parameters_ = eqx.error_if(
                parameters_,
                (parameters_[1] <= 0.0) | (parameters_[2] <= 0.0),
                "Hill half-saturation and exponent must be positive.",
            )
        if not isinstance(rate_unit, UnitDimension):
            raise TypeError("rate_unit must be a UnitDimension.")
        if not isinstance(kinetic_id, str) or not kinetic_id.strip():
            raise ValueError("kinetic_id must be a non-empty string.")
        self.reaction_index = reaction_index_
        self.reactant_indices = reactants
        self.reactant_orders = orders
        self.product_indices = products
        self.product_orders = reverse_orders
        self.parameters = parameters_
        self.rate_scale = rate_scale_
        self.rate_law = kind
        self.rate_unit = rate_unit
        self.kinetic_id = kinetic_id.strip()

    def evaluate(self, concentrations: ArrayLike, /) -> Array:
        state = jnp.asarray(concentrations)
        positive = jnp.maximum(state, 0.0)
        if self.rate_law is RateLawKind.CONSTANT:
            rate = self.parameters[0]
        else:
            substrates = positive[self.reactant_indices]
            if self.rate_law is RateLawKind.MASS_ACTION:
                rate = self.parameters[0] * jnp.prod(substrates**self.reactant_orders)
            elif self.rate_law is RateLawKind.REVERSIBLE_MASS_ACTION:
                products = positive[self.product_indices]
                forward = self.parameters[0] * jnp.prod(substrates**self.reactant_orders)
                reverse = self.parameters[1] * jnp.prod(products**self.product_orders)
                rate = forward - reverse
            else:
                substrate = substrates[0]
                if self.rate_law is RateLawKind.MICHAELIS_MENTEN:
                    vmax, michaelis = self.parameters
                    rate = vmax * substrate / (michaelis + substrate)
                else:
                    vmax, half_saturation, hill = self.parameters
                    powered = substrate**hill
                    rate = vmax * powered / (half_saturation**hill + powered)
        return self.rate_scale * rate


class KineticReactionSystem(StrictModule):
    """Concentration ODE assembled from native reaction laws and stoichiometry."""

    network: StoichiometricNetwork
    reactions: tuple[KineticReaction, ...]
    reaction_order: Array
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        network: StoichiometricNetwork,
        reactions: tuple[KineticReaction, ...] | list[KineticReaction],
        /,
        *,
        system_id: str | None = None,
    ):
        if not isinstance(network, StoichiometricNetwork):
            raise TypeError("network must be a StoichiometricNetwork.")
        reactions_ = tuple(reactions)
        if len(reactions_) != network.num_reactions or any(
            not isinstance(item, KineticReaction) for item in reactions_
        ):
            raise ValueError(
                "Exactly one KineticReaction is required per network reaction."
            )
        indices = np.asarray(
            [int(np.asarray(item.reaction_index)) for item in reactions_], dtype=np.int32
        )
        if sorted(indices.tolist()) != list(range(network.num_reactions)):
            raise ValueError(
                "Kinetic reaction indices must be a permutation of network reactions."
            )
        for item in reactions_:
            if item.reactant_indices.size and (
                int(np.asarray(item.reactant_indices).min()) < 0
                or int(np.asarray(item.reactant_indices).max()) >= network.num_species
            ):
                raise ValueError(
                    "Kinetic reactant index is outside the species capacity."
                )
            if item.product_indices.size and (
                int(np.asarray(item.product_indices).min()) < 0
                or int(np.asarray(item.product_indices).max()) >= network.num_species
            ):
                raise ValueError("Kinetic product index is outside the species capacity.")
            network_reaction = network.reactions[int(np.asarray(item.reaction_index))]
            if item.rate_unit.exponents != network_reaction.flux_unit.exponents:
                raise ValueError(
                    f"Kinetic rate unit for {item.kinetic_id!r} does not match its reaction flux unit."
                )
        identifier = (
            f"{network.network_id}:kinetics"
            if system_id is None
            else str(system_id).strip()
        )
        if not identifier:
            raise ValueError("system_id must be a non-empty string or None.")
        self.network = network
        self.reactions = reactions_
        self.reaction_order = jnp.asarray(np.argsort(indices), dtype=jnp.int32)
        self.system_id = identifier

    def rates(self, concentrations: ArrayLike, /) -> Array:
        state = jnp.asarray(concentrations)
        if state.shape != (self.network.num_species,):
            raise ValueError("concentrations must have one value per network species.")
        unordered = jnp.stack(tuple(item.evaluate(state) for item in self.reactions))
        return unordered[self.reaction_order]

    def derivative(
        self,
        time: ArrayLike,
        concentrations: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        del time, args
        rates = self.rates(concentrations)
        amount_rate = oe.contract("sr,r->s", self.network.stoichiometric_matrix, rates)
        concentration_rate = amount_rate / self.network.compartment_volumes
        return jnp.where(self.network.internal_species_mask, concentration_rate, 0.0)

    def to_continuous_system(self, /) -> ContinuousSystem:
        """Lower exactly to the canonical native continuous-system interface."""

        layout = StateLayout(
            (self.network.num_species,),
            axes=("species",),
            component_names=self.network.species_ids,
            layout_id=f"{self.system_id}:state-layout",
        )
        return ContinuousSystem(
            self.derivative,
            state_layout=layout,
            system_id=self.system_id,
        )

    def to_dynamics_evolution(self, /, **options: Any) -> DiffraxEvolution:
        """Bind the system to Phydrax's differentiable continuous evolution adapter."""

        return DiffraxEvolution(self.to_continuous_system(), **options)


class KineticSimulationEvidence(StrictModule):
    """Native dynamics steps plus positivity and conservation residual evidence."""

    evolution_steps: tuple[EvolutionStep, ...]
    finite_mask: Array
    nonnegative_mask: Array
    conserved_pool_drift: Array
    backend: str = eqx.field(static=True)
    positivity_claim: str = eqx.field(static=True)


class KineticSimulationResult(StrictModule):
    """Saved concentration trajectory with native dynamics status and evidence."""

    valid: Array
    status: Array
    times: Array
    concentrations: Array
    evidence: KineticSimulationEvidence
    method_contract: BioinformaticsMethodContract
    system_id: str = eqx.field(static=True)

    @property
    def final_concentrations(self) -> Array:
        return self.concentrations[-1]


class KineticObjectiveEvidence(StrictModule):
    """Pointwise residual and effective weighted-observation evidence."""

    residuals: Array
    weighted_residuals: Array
    effective_observations: Array
    finite: Array


class KineticObjectiveResult(StrictModule):
    """Finite weighted trajectory objective with an explicit method contract."""

    valid: Array
    status: Array
    value: Array
    evidence: KineticObjectiveEvidence
    method_contract: BioinformaticsMethodContract


class KineticTrajectoryObjective(StrictModule):
    """Weighted least-squares adapter over saved native dynamics states."""

    observed_concentrations: Array
    weights: Array
    observation_mask: Array

    def __init__(
        self,
        observed_concentrations: ArrayLike,
        /,
        *,
        weights: ArrayLike = 1.0,
        observation_mask: ArrayLike | None = None,
    ):
        observations = jnp.asarray(observed_concentrations)
        if observations.ndim != 2:
            raise ValueError("observed_concentrations must have shape (time, species).")
        observations = (
            observations
            if jnp.issubdtype(observations.dtype, jnp.inexact)
            else observations.astype(float)
        )
        weights_ = jnp.broadcast_to(
            jnp.asarray(weights, dtype=observations.dtype), observations.shape
        )
        mask = (
            jnp.isfinite(observations)
            if observation_mask is None
            else jnp.broadcast_to(
                jnp.asarray(observation_mask, dtype=bool), observations.shape
            )
        )
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_) | (weights_ < 0.0)),
            "Objective weights must be finite and non-negative.",
        )
        self.observed_concentrations = jnp.where(mask, observations, 0.0)
        self.weights = weights_
        self.observation_mask = mask

    def evaluate(self, simulated_concentrations: ArrayLike, /) -> KineticObjectiveResult:
        simulated = jnp.asarray(simulated_concentrations)
        if simulated.shape != self.observed_concentrations.shape:
            raise ValueError("Simulated and observed concentration shapes must match.")
        residual = jnp.where(
            self.observation_mask,
            simulated - self.observed_concentrations,
            0.0,
        )
        weighted = jnp.sqrt(self.weights) * residual
        finite = jnp.all(jnp.isfinite(weighted))
        value = 0.5 * jnp.sum(weighted * weighted)
        status = jnp.where(
            finite, int(KineticStatus.SUCCESS), int(KineticStatus.NONFINITE)
        ).astype(jnp.int32)
        evidence = KineticObjectiveEvidence(
            residuals=residual,
            weighted_residuals=weighted,
            effective_observations=jnp.sum(self.observation_mask, dtype=jnp.int32),
            finite=finite,
        )
        contract = BioinformaticsMethodContract(
            "kinetic-trajectory-weighted-least-squares",
            MethodKind.EXACT_MODEL,
            ExecutionKind.FLOATING_POINT_DIRECT,
            DifferentiationKind.EXACT_AD,
            OutputKind.SCALAR,
            conditioning_statement="Residual scaling is exactly the supplied square-root weight scaling.",
            truncation_statement="Every masked-in trajectory observation contributes.",
            capacity_semantics="Storage is exactly the supplied dense time-by-species observation array.",
            assumptions=(
                "Observation weights are non-negative inverse-variance-like scales.",
            ),
            nondifferentiable_outputs=("status", "valid", "effective_observations"),
        )
        return KineticObjectiveResult(
            valid=finite,
            status=status,
            value=jnp.where(finite, value, jnp.nan),
            evidence=evidence,
            method_contract=contract,
        )

    def __call__(self, simulated_concentrations: ArrayLike, /) -> Array:
        return self.evaluate(simulated_concentrations).value


def simulate_kinetics(
    system: KineticReactionSystem,
    initial_concentrations: ArrayLike,
    times: ArrayLike,
    /,
    **evolution_options: Any,
) -> KineticSimulationResult:
    """Simulate saved times through the native dynamics evolution lifecycle."""

    if not isinstance(system, KineticReactionSystem):
        raise TypeError("system must be a KineticReactionSystem.")
    times_ = jnp.asarray(times)
    state = jnp.asarray(initial_concentrations)
    if times_.ndim != 1 or times_.shape[0] < 1:
        raise ValueError("times must be a non-empty one-dimensional array.")
    if state.shape != (system.network.num_species,):
        raise ValueError("initial_concentrations must have one value per species.")
    if not bool(np.all(np.diff(np.asarray(times_)) > 0.0)):
        raise ValueError("times must be strictly increasing.")
    if not bool(np.all(np.asarray(state) >= 0.0)):
        raise ValueError("initial_concentrations must be non-negative.")
    evolution = system.to_dynamics_evolution(**evolution_options)
    states = [state]
    steps = []
    for index in range(times_.shape[0] - 1):
        step = evolution.advance(state, times_[index], times_[index + 1])
        steps.append(step)
        state = step.final_state
        states.append(state)
    concentrations = jnp.stack(tuple(states))
    finite = jnp.all(jnp.isfinite(concentrations), axis=1)
    nonnegative = jnp.all(concentrations >= -1.0e-10, axis=1)
    step_valid = (
        jnp.stack(tuple(step.valid for step in steps))
        if steps
        else jnp.ones((0,), dtype=bool)
    )
    valid = jnp.all(finite & nonnegative) & jnp.all(step_valid)
    status = jnp.where(
        ~jnp.all(finite),
        int(KineticStatus.NONFINITE),
        jnp.where(
            ~jnp.all(nonnegative),
            int(KineticStatus.NEGATIVE_CONCENTRATION),
            jnp.where(
                ~jnp.all(step_valid),
                int(KineticStatus.DYNAMICS_FAILURE),
                int(KineticStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    amount_states = concentrations * system.network.compartment_volumes[None, :]
    conservation = conservation_analysis(system.network)
    laws = conservation.conservation_laws
    if laws.shape[0]:
        internal_amount = amount_states[:, system.network.internal_species_mask]
        pools = oe.contract("cs,ts->tc", laws, internal_amount)
        drift = pools - pools[0]
    else:
        drift = jnp.zeros((times_.shape[0], 0), dtype=concentrations.dtype)
    evidence = KineticSimulationEvidence(
        evolution_steps=tuple(steps),
        finite_mask=finite,
        nonnegative_mask=nonnegative,
        conserved_pool_drift=drift,
        backend="phydrax.solver.DiffraxEvolution",
        positivity_claim="validated nonnegative saved states; no clipping of reported states",
    )
    contract = BioinformaticsMethodContract(
        "kinetic-reaction-system-simulation",
        MethodKind.EXACT_MODEL,
        ExecutionKind.ITERATIVE_TOLERANCE,
        DifferentiationKind.EXACT_AD,
        OutputKind.ARRAY,
        conditioning_statement=(
            "Integration conditioning depends on kinetic time-scale separation and the "
            "selected native dynamics method."
        ),
        truncation_statement="All requested save times are advanced in order.",
        capacity_semantics="The selected native dynamics evolution preflights its maximum step count.",
        assumptions=(
            "Compartments are well mixed.",
            "Kinetic rate laws use concentrations.",
        ),
        nondifferentiable_outputs=("status", "valid", "positivity masks"),
        absolute_tolerance=evolution.atol,
        relative_tolerance=evolution.rtol,
    )
    return KineticSimulationResult(
        valid=valid,
        status=status,
        times=times_,
        concentrations=concentrations,
        evidence=evidence,
        method_contract=contract,
        system_id=system.system_id,
    )


__all__ = [
    "simulate_kinetics",
    "KineticObjectiveEvidence",
    "KineticObjectiveResult",
    "KineticReaction",
    "KineticReactionSystem",
    "KineticSimulationEvidence",
    "KineticSimulationResult",
    "KineticStatus",
    "KineticTrajectoryObjective",
    "RateLawKind",
]
