#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..dynamics import (
    AbstractDifferentiableEvolution,
    AbstractInputPolicy,
    ContinuousSystem,
    EVOLUTION_BACKEND_FAILED,
    EVOLUTION_NONFINITE,
    EVOLUTION_OUTSIDE_GEOMETRY,
    EVOLUTION_SUCCESS,
    EvolutionStep,
    EvolutionTangentStep,
)
from ..linalg import (
    ArraySpace,
    LinearizationPolicy,
    PreparedLinearization,
    PyTreeSpace,
)
from ._differential import DifferentialProblem
from ._diffrax_backend import solve_diffrax
from ._temporal_method import (
    configuration_id,
    diffrax_differentiation_evidence,
    TemporalDifferentiationEvidence,
)


_DIFFRAX_SUCCESS = jax.tree.leaves(dfx.RESULTS.successful)[0]


def _identifier(value: str | None, default: str, owner: str, /) -> str:
    resolved = default if value is None else value
    if not isinstance(resolved, str) or not resolved:
        raise ValueError(f"{owner} must be a non-empty string or None.")
    return resolved


def _split_linearization(
    forward_endpoint,
    reverse_endpoint,
    point,
    source_space,
    target_space,
    linearization_id: str,
    /,
) -> PreparedLinearization:
    point_ = source_space.validate(point)
    _, pushforward = jax.linearize(forward_endpoint, point_)
    primal, reverse_pullback = jax.vjp(reverse_endpoint, point_)

    def pullback(cotangent):
        return reverse_pullback(cotangent)[0]

    return PreparedLinearization(
        source=source_space,
        target=target_space,
        point=point_,
        primal=target_space.validate(primal),
        pushforward=pushforward,
        pullback=pullback,
        policy=LinearizationPolicy(),
        linearization_id=linearization_id,
    )


class DiffraxEvolution(AbstractDifferentiableEvolution):
    """Deterministic continuous-system evolution through the canonical Diffrax backend."""

    system: ContinuousSystem
    input_policy: AbstractInputPolicy | None
    solver: Any
    stepsize_controller: Any
    reverse_adjoint: Any
    forward_adjoint: Any
    composite_adjoint: Any
    reverse_differentiation: TemporalDifferentiationEvidence
    forward_differentiation: TemporalDifferentiationEvidence
    composite_differentiation: TemporalDifferentiationEvidence
    dt0: Array | None
    event: Any
    evolution_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    tangent_method_id: str = eqx.field(static=True)
    eventful: bool = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int | None = eqx.field(static=True)

    def __init__(
        self,
        system: ContinuousSystem,
        /,
        *,
        input_policy: AbstractInputPolicy | None = None,
        solver: Any = None,
        stepsize_controller: Any = None,
        reverse_adjoint: Any = None,
        forward_adjoint: Any = None,
        composite_adjoint: Any = None,
        dt0: ArrayLike | None = None,
        event: Any = None,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-8,
        max_steps: int | None = 4096,
        evolution_id: str | None = None,
    ):
        if not isinstance(system, ContinuousSystem):
            raise TypeError("DiffraxEvolution system must be a ContinuousSystem.")
        if input_policy is not None and not isinstance(input_policy, AbstractInputPolicy):
            raise TypeError("input_policy must be an AbstractInputPolicy or None.")
        if (system.input_layout is None) != (input_policy is None):
            raise ValueError(
                "DiffraxEvolution requires exactly one input policy for an input-driven system."
            )
        if input_policy is not None:
            system_input_layout = system.input_layout
            if system_input_layout is None:
                raise RuntimeError("Input-driven system is missing its input layout.")
            if input_policy.input_layout.layout_id != system_input_layout.layout_id:
                raise ValueError(
                    "Input policy and system input layouts must match exactly."
                )
        relative_tolerance = float(rtol)
        absolute_tolerance = float(atol)
        if (
            not np.isfinite((relative_tolerance, absolute_tolerance)).all()
            or relative_tolerance <= 0.0
            or absolute_tolerance <= 0.0
        ):
            raise ValueError("rtol and atol must be finite and positive.")
        step_limit = None if max_steps is None else int(max_steps)
        if step_limit is not None and step_limit < 1:
            raise ValueError("max_steps must be positive or None.")
        initial_step = None if dt0 is None else jnp.asarray(dt0)
        if initial_step is not None and (
            initial_step.shape != () or not bool(jnp.isfinite(initial_step))
        ):
            raise ValueError("dt0 must be a finite scalar or None.")

        selected_reverse = (
            dfx.RecursiveCheckpointAdjoint()
            if reverse_adjoint is None
            else reverse_adjoint
        )
        selected_forward = (
            dfx.ForwardMode() if forward_adjoint is None else forward_adjoint
        )
        selected_composite = (
            dfx.DirectAdjoint() if composite_adjoint is None else composite_adjoint
        )
        adaptive = stepsize_controller is None or isinstance(
            stepsize_controller,
            dfx.AbstractAdaptiveStepSizeController,
        )
        reverse_evidence = diffrax_differentiation_evidence(
            selected_reverse,
            adaptive=adaptive,
            has_event=event is not None,
            stochastic=False,
            maximum_steps=step_limit,
        )
        forward_evidence = diffrax_differentiation_evidence(
            selected_forward,
            adaptive=adaptive,
            has_event=event is not None,
            stochastic=False,
            maximum_steps=step_limit,
        )
        composite_evidence = diffrax_differentiation_evidence(
            selected_composite,
            adaptive=adaptive,
            has_event=event is not None,
            stochastic=False,
            maximum_steps=step_limit,
        )
        if (
            reverse_evidence.orientations
            and "reverse" not in reverse_evidence.orientations
        ):
            raise ValueError("reverse_adjoint must support reverse-mode differentiation.")
        if (
            forward_evidence.orientations
            and "forward" not in forward_evidence.orientations
        ):
            raise ValueError("forward_adjoint must support forward-mode differentiation.")
        if composite_evidence.orientations and not {
            "forward",
            "reverse",
        }.issubset(composite_evidence.orientations):
            raise ValueError(
                "composite_adjoint must support forward- and reverse-mode "
                "differentiation."
            )
        if (
            not reverse_evidence.orientations
            or not forward_evidence.orientations
            or not composite_evidence.orientations
        ) and evolution_id is None:
            raise ValueError(
                "Unclassified Diffrax adjoints require an explicit evolution_id."
            )
        self.system = system
        self.input_policy = input_policy
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.reverse_adjoint = selected_reverse
        self.forward_adjoint = selected_forward
        self.composite_adjoint = selected_composite
        self.reverse_differentiation = reverse_evidence
        self.forward_differentiation = forward_evidence
        self.composite_differentiation = composite_evidence
        self.dt0 = initial_step
        self.event = event
        self.eventful = event is not None
        self.stochastic = False
        self.evolution_id = _identifier(
            evolution_id,
            f"{system.system_id}:diffrax-evolution",
            "DiffraxEvolution evolution_id",
        )
        self.method_id = configuration_id(
            (
                solver,
                stepsize_controller,
                self.reverse_adjoint,
                self.forward_adjoint,
                self.composite_adjoint,
                None if dt0 is None else repr(dt0),
                event,
                relative_tolerance,
                absolute_tolerance,
                step_limit,
            ),
            prefix="diffrax-evolution",
        )
        self.backend_id = "backend:diffrax"
        self.discretization_id = "diffrax-selected-step-sequence"
        self.approximation_id = "numerical-differential-flow"
        self.tangent_method_id = "jax-jvp:numerical-differential-flow"
        self.rtol = relative_tolerance
        self.atol = absolute_tolerance
        self.max_steps = step_limit

    def _vector_field(self, time: Array, state: Array, args: Any, /) -> Array:
        inputs = (
            None
            if self.input_policy is None
            else self.input_policy.evaluate(time, state, args)
        )
        return self.system.evaluate(time, state, args, inputs=inputs)

    def _solve(
        self,
        state: Array,
        source: Array,
        target: Array,
        args: Any,
        /,
        *,
        adjoint: Any,
    ):
        problem = DifferentialProblem(
            self._vector_field,
            state,
            t0=source,
            t1=target,
            args=args,
            state_geometry=self.state_layout.geometry,
        )
        return solve_diffrax(
            problem,
            save_times=jnp.asarray([target]),
            solver=self.solver,
            stepsize_controller=self.stepsize_controller,
            adjoint=adjoint,
            dt0=self.dt0,
            event=self.event,
            rtol=self.rtol,
            atol=self.atol,
            dense=False,
            max_steps=self.max_steps,
            throw=False,
        )

    def _solve_data(
        self,
        state: Array,
        source: Array,
        target: Array,
        args: Any,
        /,
        *,
        adjoint: Any,
    ) -> tuple[Array, Array]:
        solution = self._solve(
            state,
            source,
            target,
            args,
            adjoint=adjoint,
        )
        backend_status = jax.tree.leaves(solution.backend_result)[0]
        return solution.states[-1], backend_status

    def _step(
        self,
        final_state: Array,
        backend_status: Array,
        source: Array,
        target: Array,
        /,
    ) -> EvolutionStep:
        finite = jnp.all(jnp.isfinite(final_state))
        membership = jnp.asarray(
            self.state_layout.geometry.contains(final_state), dtype=bool
        )
        if membership.shape != ():
            raise ValueError("State geometry contains() must return one scalar boolean.")
        backend_valid = backend_status == _DIFFRAX_SUCCESS
        valid = backend_valid & finite & membership
        status = jnp.where(
            ~backend_valid,
            EVOLUTION_BACKEND_FAILED,
            jnp.where(
                ~finite,
                EVOLUTION_NONFINITE,
                jnp.where(
                    ~membership,
                    EVOLUTION_OUTSIDE_GEOMETRY,
                    EVOLUTION_SUCCESS,
                ),
            ),
        ).astype(jnp.int32)
        return EvolutionStep(
            source_coordinate=source,
            target_coordinate=target,
            final_state=final_state,
            valid=valid,
            status=status,
            backend_status=backend_status,
            system_id=self.system.system_id,
            evolution_id=self.evolution_id,
            method_id=self.method_id,
            backend_id=self.backend_id,
            discretization_id=self.discretization_id,
            approximation_id=self.approximation_id,
        )

    def advance(
        self,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> EvolutionStep:
        state_array = jnp.asarray(state)
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        if state_array.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_array.shape}."
            )
        if source.shape != () or target.shape != ():
            raise ValueError("Evolution segment coordinates must be scalar.")
        final_state, backend_status = self._solve_data(
            state_array,
            source,
            target,
            args,
            adjoint=self.composite_adjoint,
        )
        return self._step(final_state, backend_status, source, target)

    def tangent_action(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> EvolutionTangentStep:
        state_array = jnp.asarray(state)
        vector = jnp.asarray(tangent)
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        if state_array.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_array.shape}."
            )
        if vector.shape != self.state_layout.shape:
            raise ValueError(
                f"tangent must have shape {self.state_layout.shape}; got {vector.shape}."
            )
        if source.shape != () or target.shape != ():
            raise ValueError("Evolution segment coordinates must be scalar.")

        geometry = self.state_layout.geometry
        if geometry.trivial:
            (final_state, backend_status), (propagated, _) = jax.jvp(
                lambda point: self._solve_data(
                    point,
                    source,
                    target,
                    args,
                    adjoint=self.forward_adjoint,
                ),
                (state_array,),
                (vector,),
            )
        else:
            final_state, backend_status = self._solve_data(
                state_array,
                source,
                target,
                args,
                adjoint=self.forward_adjoint,
            )
            zero = jnp.zeros_like(state_array)

            def local_flow(local):
                perturbed = geometry.retract(state_array, local)
                endpoint, _ = self._solve_data(
                    perturbed,
                    source,
                    target,
                    args,
                    adjoint=self.forward_adjoint,
                )
                return geometry.inverse_retract(final_state, endpoint)

            _, propagated = jax.jvp(local_flow, (zero,), (vector,))
        primal = self._step(final_state, backend_status, source, target)
        tangent_finite = jnp.all(jnp.isfinite(propagated))
        valid = primal.valid & tangent_finite
        status = jnp.where(
            ~primal.valid,
            primal.status,
            jnp.where(tangent_finite, EVOLUTION_SUCCESS, EVOLUTION_NONFINITE),
        ).astype(jnp.int32)
        return EvolutionTangentStep(
            primal=primal,
            tangent=propagated,
            valid=valid,
            status=status,
            tangent_method_id=self.tangent_method_id,
        )

    def state_linearization(
        self,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> PreparedLinearization:
        state_array = jnp.asarray(state)
        if state_array.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_array.shape}."
            )
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        if source.shape != () or target.shape != ():
            raise ValueError("Evolution segment coordinates must be scalar.")
        space = ArraySpace(self.state_layout.shape, dtype=state_array.dtype)
        geometry = self.state_layout.geometry
        if geometry.trivial:
            point = state_array

            def forward_endpoint(current_state):
                return self._solve_data(
                    current_state,
                    source,
                    target,
                    args,
                    adjoint=self.forward_adjoint,
                )[0]

            def reverse_endpoint(current_state):
                return self._solve_data(
                    current_state,
                    source,
                    target,
                    args,
                    adjoint=self.reverse_adjoint,
                )[0]

        else:
            point = jnp.zeros_like(state_array)
            base, _ = self._solve_data(
                state_array,
                source,
                target,
                args,
                adjoint=self.reverse_adjoint,
            )

            def forward_endpoint(local):
                endpoint, _ = self._solve_data(
                    geometry.retract(state_array, local),
                    source,
                    target,
                    args,
                    adjoint=self.forward_adjoint,
                )
                return geometry.inverse_retract(base, endpoint)

            def reverse_endpoint(local):
                endpoint, _ = self._solve_data(
                    geometry.retract(state_array, local),
                    source,
                    target,
                    args,
                    adjoint=self.reverse_adjoint,
                )
                return geometry.inverse_retract(base, endpoint)

        return _split_linearization(
            forward_endpoint,
            reverse_endpoint,
            point,
            space,
            space,
            (f"{self.evolution_id}:state-linearization:{self.state_layout.layout_id}"),
        )

    def argument_linearization(
        self,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any,
        /,
    ) -> PreparedLinearization:
        state_array = jnp.asarray(state)
        if state_array.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_array.shape}."
            )
        leaves = jax.tree.leaves(args)
        if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
            raise TypeError(
                "Evolution argument linearization requires a nonempty PyTree "
                "of inexact arrays."
            )
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        if source.shape != () or target.shape != ():
            raise ValueError("Evolution segment coordinates must be scalar.")

        def forward_endpoint(current_args):
            return self._solve_data(
                state_array,
                source,
                target,
                current_args,
                adjoint=self.forward_adjoint,
            )[0]

        def reverse_endpoint(current_args):
            return self._solve_data(
                state_array,
                source,
                target,
                current_args,
                adjoint=self.reverse_adjoint,
            )[0]

        return _split_linearization(
            forward_endpoint,
            reverse_endpoint,
            args,
            PyTreeSpace(args),
            ArraySpace(self.state_layout.shape, dtype=state_array.dtype),
            (f"{self.evolution_id}:argument-linearization:{self.state_layout.layout_id}"),
        )


__all__ = ["DiffraxEvolution"]
