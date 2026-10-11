"""Fixed-step adapter: a candidate step commits as a whole or rolls back."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ..._fingerprint import canonical_fingerprint
from ..._trainable import fixed_field
from ...solver import AbstractFixedStepMethod, FixedStepResult
from ._acceptance import (
    fuel_step_evidence,
    FuelAcceptancePolicy,
    FuelBalanceEvidence,
    FuelCouplingEvidence,
    FuelLinearEvidence,
    FuelNonlinearEvidence,
    FuelPhysicalEvidence,
    FuelSolverEvidence,
    FuelSourceSupport,
    FuelStepEvidence,
    source_ids,
)
from ._bundles import FuelModelBundle
from ._execution import FuelNumericalBinding, FuelNumericalExecution
from ._layout import SOURCE_SLOTS
from ._problem import PreparedFuelRod
from ._state import FuelRodState


def evidence_prototype(
    prepared: PreparedFuelRod, policy: FuelAcceptancePolicy, /
) -> FuelStepEvidence:
    """Evidence with the shapes, dtypes and static identities of every attempt.

    Numerical values are placeholders; unavailable linear and coupling
    diagnostics use the solver convention of -1 and NaN.
    """
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    count = jnp.asarray(0, dtype=jnp.int32)
    flag = jnp.asarray(False, dtype=jnp.bool_)
    unavailable_integer = jnp.asarray(-1, dtype=jnp.int32)
    unavailable = jnp.asarray(jnp.nan, dtype=jnp.float64)
    coupling_budget = prepared.coupling_termination.maximum_evaluations
    if coupling_budget is None:
        raise RuntimeError("Fuel coupling requires an explicit evaluation budget.")
    return FuelStepEvidence(
        step_index=jnp.asarray(0, dtype=jnp.int64),
        time=zero,
        time_step=zero,
        accepted=flag,
        terminal_domain_exit=flag,
        status=count,
        solver=FuelSolverEvidence(
            nonlinear=FuelNonlinearEvidence(
                status=count, residual=zero, iterations=count, counts_complete=flag
            ),
            linear=FuelLinearEvidence(
                status=unavailable_integer,
                rank=unavailable_integer,
                condition_estimate=unavailable,
                residual=unavailable,
                converged=flag,
                rank_available=flag,
                condition_available=flag,
            ),
            coupling=FuelCouplingEvidence(
                status=unavailable_integer,
                iterations=count,
                evaluations=count,
                joint_residual=unavailable,
                gas_admissible=flag,
                thermomechanical_residual_evaluations=count,
                thermomechanical_iterations=count,
                gas_block_evaluations=count,
            ),
            local_roots_converged=flag,
        ),
        physics=FuelPhysicalEvidence(
            physical=flag,
            in_range=flag,
            minimum_gap=zero,
            complementarity_error=zero,
            friction_error=zero,
            increments=jnp.zeros((4,), dtype=jnp.float64),
            accommodation_extrapolated=flag,
            accommodation_source_comparison=flag,
        ),
        balances=FuelBalanceEvidence(energy_error=zero, gas_error=zero),
        source_support=FuelSourceSupport(
            supported=jnp.zeros((len(SOURCE_SLOTS),), dtype=jnp.bool_),
            slots=SOURCE_SLOTS,
            source_ids=source_ids(prepared.admission.bundle, prepared.admission.coolant),
        ),
        prepared_id=prepared.prepared_id,
        bundle_id=prepared.admission.bundle.bundle_id,
        history_id=prepared.admission.history.history_id,
        policy_id=policy.policy_id,
        coupling_evaluation_budget=coupling_budget,
        coupling_tolerance=prepared.coupling_tolerance,
    )


class FuelStepMethod(AbstractFixedStepMethod):
    prepared: PreparedFuelRod
    policy: FuelAcceptancePolicy
    prototype: FuelStepEvidence
    method_id: str = eqx.field(static=True)
    numerical_binding: FuelNumericalBinding | None = fixed_field(default=None)

    def __init__(
        self,
        prepared: PreparedFuelRod,
        /,
        *,
        policy: FuelAcceptancePolicy | None = None,
        numerical_execution: FuelNumericalExecution | None = None,
    ) -> None:
        if not isinstance(prepared, PreparedFuelRod):
            raise TypeError("FuelStepMethod requires a prepared rod.")
        policy = FuelAcceptancePolicy() if policy is None else policy
        self.prepared = prepared
        self.policy = policy
        self.prototype = evidence_prototype(prepared, policy)
        self.method_id = canonical_fingerprint(
            {
                "kind": "fuel-partitioned-implicit-backward-euler",
                "prepared": prepared.prepared_id,
                "acceptance": policy.policy_id,
            }
        )
        self.numerical_binding = (
            None
            if numerical_execution is None
            else numerical_execution.bind(prepared, policy=policy)
        )

    @property
    def evidence_template(self) -> FuelStepEvidence:
        return self.prototype

    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: FuelModelBundle | None,
        /,
    ) -> FixedStepResult:
        if not isinstance(state, FuelRodState):
            raise TypeError(
                "Fuel fixed steps require a complete FuelRodState checkpoint."
            )
        if self.numerical_binding is not None:
            return self.numerical_binding.step(
                step_index, time, state, step_size, parameters=args
            )
        bundle = self.prepared.admission.bundle if args is None else args
        result = self.prepared.solve_candidate(state, time, step_size, parameters=bundle)
        evaluation = self.prepared.candidate_evaluation(result)
        evidence = fuel_step_evidence(
            self.prepared,
            result,
            state,
            time,
            step_size,
            step_index,
            parameters=bundle,
            policy=self.policy,
        )
        candidate = evaluation.candidate
        # The accepted state is the candidate or, on refusal, the unchanged
        # previous state; the candidate is kept either way for diagnostics.
        accepted = jax.tree.map(
            lambda candidate_leaf, previous_leaf: jnp.where(
                evidence.accepted, candidate_leaf, previous_leaf
            ),
            candidate,
            state,
        )
        return FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=evidence.accepted,
            residual=evidence.solver.nonlinear.residual,
            iterations=evidence.solver.coupling.thermomechanical_iterations
            + evidence.solver.coupling.iterations,
            work=evidence.solver.coupling.thermomechanical_residual_evaluations
            + evidence.solver.coupling.gas_block_evaluations,
            # The fuel step applies no learned accepted-step transform.
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.asarray(0.0),
            terminal_failure=evidence.terminal_domain_exit,
            evidence=evidence,
        )
