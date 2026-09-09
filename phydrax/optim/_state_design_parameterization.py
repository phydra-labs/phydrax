#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._iterative._types import Bounds
from ._pde_constrained import (
    AbstractStateSolver,
    StateDesignConstraint,
    StateDesignProblem,
)
from ._state_design_linearization import (
    prepare_state_design_linearization,
    state_design_response_vjp,
)


def _schema(tree, name):
    leaves, structure = jax.tree.flatten(tree)
    if not leaves or any(not eqx.is_array(leaf) for leaf in leaves):
        raise TypeError(f"{name} must be a nonempty array-only PyTree.")
    if any(not jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves):
        raise TypeError(f"{name} leaves must have real floating dtypes.")
    return structure, tuple((leaf.shape, str(leaf.dtype)) for leaf in leaves)


class _FrozenDecoder(StrictModule, NonTrainableState):
    function: Callable
    latent_schema: Any = eqx.field(static=True)
    physical_schema: Any = eqx.field(static=True)
    decoder_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    design_admissibility: Callable | None = eqx.field(static=True)

    def __call__(self, latent):
        if _schema(latent, "latent design") != self.latent_schema:
            raise ValueError("Latent design PyTree, shape, or dtype changed.")
        physical = self.function(latent)
        if _schema(physical, "decoded physical design") != self.physical_schema:
            raise ValueError(
                "Decoder changed the physical PyTree, shape, dtype, or schema."
            )
        finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(physical))
            )
        )
        if self.design_admissibility is not None:
            admissible = jnp.asarray(self.design_admissibility(physical))
            if admissible.shape != () or admissible.dtype != jnp.bool_:
                raise TypeError("design_admissibility must return one boolean scalar.")
            finite = finite & admissible
        # Invalid geometry must never be substituted by another geometry or sent to FE.
        return eqx.error_if(
            physical, ~finite, "Decoder produced an invalid physical design."
        )


class _DecodedStateSolver(AbstractStateSolver):
    physical_problem: StateDesignProblem
    decoder: _FrozenDecoder

    def __init__(self, physical_problem, decoder, /):
        self.physical_problem = physical_problem
        self.decoder = decoder

    @property
    def method_id(self):
        return f"decoded/{self.physical_problem.state_solver.method_id}"

    def solve(self, problem, design, initial_state, /, *, args):
        del problem
        # Some native solvers use physical material/geometry coordinates directly.
        # Merely reusing that solver with a latent design would violate its contract.
        return self.physical_problem.state_solver.solve(
            self.physical_problem, self.decoder(design), initial_state, args=args
        )


class StateDesignParameterization(StrictModule, NonTrainableState):
    """Frozen decoder lowering, not a claim of full-design optimality.

    ``decoder_id`` names fixed weights/code; ``realization_id`` names the fixed
    geometry/mesh/schema realization. Callables must be pure and must not close
    over mutable weights, random keys, or evolving geometry. Array parameters in
    an Equinox decoder are stopped at preparation. Only latent leaves vary.
    """

    physical_problem: StateDesignProblem
    problem: StateDesignProblem
    decoder: _FrozenDecoder

    @property
    def decoder_id(self):
        return self.decoder.decoder_id

    @property
    def realization_id(self):
        return self.decoder.realization_id

    def decode(self, latent, /):
        return self.decoder(latent)

    def response_vjp(
        self,
        latent,
        initial_state,
        /,
        *,
        response=None,
        cotangent=None,
        depends_on_state=True,
        args=None,
        linear_policy=None,
    ) -> LatentStateDesignVJP:
        """Pull one accepted physical response back without an outer-optimizer AD."""
        physical, pullback = jax.vjp(self.decode, latent)
        linearization = prepare_state_design_linearization(
            self.physical_problem,
            physical,
            initial_state,
            args=args,
            linear_policy=linear_policy,
        )
        physical_vjp = state_design_response_vjp(
            linearization,
            response=response,
            cotangent=cotangent,
            depends_on_state=depends_on_state,
        )
        latent_cotangent = pullback(physical_vjp.design_cotangent)[0]
        return LatentStateDesignVJP(
            physical_vjp.values,
            latent_cotangent,
            physical,
            physical_vjp,
            linearization.state_result,
            physical_vjp.accepted,
        )


class LatentStateDesignVJP(StrictModule):
    values: PyTree[Array]
    latent_cotangent: PyTree[Array]
    physical_design: PyTree[Array]
    physical_vjp: Any
    state_result: Any
    accepted: Array


class _DecodedCoordinate(StrictModule):
    decoder: _FrozenDecoder
    index: int = eqx.field(static=True)

    def __call__(self, state, latent, args):
        del state, args
        values, _ = ravel_pytree(self.decoder(latent))
        return values[self.index]


def reparameterize_state_design(
    problem: StateDesignProblem,
    decode: Callable,
    latent_template: PyTree[Array],
    physical_template: PyTree[Array],
    /,
    *,
    decoder_id: str,
    realization_id: str,
    latent_bounds: Bounds | None = None,
    design_admissibility: Callable | None = None,
) -> StateDesignParameterization:
    """Compose a fixed decoder through residual, objective, gates and constraints.

    Physical box bounds are retained as state-independent composed constraints;
    latent bounds never replace physical bounds. No assertion that an arbitrary
    callable is a bounded map is trusted. Templates declare the exact PyTrees,
    leaf shapes and dtypes (including static ``DesignState`` parameter schemas).
    ``design_admissibility`` optionally rejects invalid geometry before FE runs.
    The decoder is deterministic, takes only a latent tree, and is not trained by
    this API. A stationary latent solution need not be a full-design optimum.
    """
    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be a StateDesignProblem.")
    if not callable(decode):
        raise TypeError("decode must be callable.")
    if not isinstance(decoder_id, str) or not decoder_id:
        raise ValueError("decoder_id must be a nonempty frozen decoder identity.")
    if not isinstance(realization_id, str) or not realization_id:
        raise ValueError("realization_id must be a nonempty realization identity.")
    if design_admissibility is not None and not callable(design_admissibility):
        raise TypeError("design_admissibility must be callable or None.")
    frozen = jax.tree.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        decode,
    )
    decoder = _FrozenDecoder(
        frozen,
        _schema(latent_template, "latent template"),
        _schema(physical_template, "physical template"),
        decoder_id,
        realization_id,
        design_admissibility,
    )
    decoder(latent_template)

    def residual(state, latent, args):
        return problem.residual(state, decoder(latent), args)

    def objective(state, latent, args):
        return problem.objective(state, decoder(latent), args)

    def compose(function):
        return lambda state, latent, args: function(state, decoder(latent), args)

    def certification(
        state, latent, residual, status, *, reference_norm, args, solver_acceptance=None
    ):
        return problem.state_evidence(
            state,
            decoder(latent),
            residual,
            status,
            reference_norm=reference_norm,
            args=args,
            solver_acceptance=solver_acceptance,
        )

    constraints = tuple(
        StateDesignConstraint(
            compose(constraint.function),
            lower=constraint.lower,
            upper=constraint.upper,
            constraint_id=constraint.constraint_id,
            depends_on_state=constraint.depends_on_state,
        )
        for constraint in problem.constraints
    )
    identifier = f"{problem.problem_id}/latent/{decoder_id}/{realization_id}"
    if problem.design_bounds is not None:
        lower, upper = problem.design_bounds.materialize(physical_template)
        flat_lower, _ = ravel_pytree(lower)
        flat_upper, _ = ravel_pytree(upper)
        host_lower = np.asarray(flat_lower)
        host_upper = np.asarray(flat_upper)
        bound_constraints = tuple(
            StateDesignConstraint(
                _DecodedCoordinate(decoder, index),
                lower=flat_lower[index],
                upper=flat_upper[index],
                constraint_id=f"{identifier}/physical-bound:{index}",
                depends_on_state=False,
            )
            for index in range(flat_lower.size)
            if np.isfinite(host_lower[index]) or np.isfinite(host_upper[index])
        )
        constraints += bound_constraints
    lowered = StateDesignProblem(
        residual,
        objective,
        state_solver=_DecodedStateSolver(problem, decoder),
        acceptance_policy=problem.acceptance_policy,
        state_certification=certification,
        design_bounds=latent_bounds,
        constraints=constraints,
        has_aux=problem.has_aux,
        problem_id=identifier,
    )
    return StateDesignParameterization(problem, lowered, decoder)


__all__ = [
    "LatentStateDesignVJP",
    "StateDesignParameterization",
    "reparameterize_state_design",
]
