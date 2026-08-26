#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import ArraySpace
from ...nonlinear import NonlinearSystemProblem
from .._evolution import AbstractDifferentiableEvolution


class RelativeEquilibriumProblem(StrictModule):
    """Comoving equilibrium residual with one phase condition per generator."""

    vector_field: Callable[[Array, Any], ArrayLike]
    generators: tuple[Callable[[Array], ArrayLike], ...]
    phase_conditions: tuple[Callable[[Array, Any], ArrayLike], ...]
    state_space: ArraySpace
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: Callable[[Array, Any], ArrayLike],
        generators: Sequence[Callable[[Array], ArrayLike]],
        phase_conditions: Sequence[Callable[[Array, Any], ArrayLike]],
        state_space: ArraySpace,
        /,
        *,
        problem_id: str | None = None,
    ):
        generators_ = tuple(generators)
        phases = tuple(phase_conditions)
        if not callable(vector_field):
            raise TypeError("vector_field must be callable.")
        if not generators_ or len(generators_) != len(phases):
            raise ValueError(
                "Relative equilibria require matching nonempty generators and phases."
            )
        if not all(callable(value) for value in generators_ + phases):
            raise TypeError("Generators and phase conditions must be callable.")
        if not isinstance(state_space, ArraySpace) or len(state_space.shape) != 1:
            raise TypeError("state_space must be one rank-one ArraySpace.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "relative-equilibrium-problem-v1",
                    "state_space": state_space.space_id,
                    "generator_count": len(generators_),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.vector_field = vector_field
        self.generators = generators_
        self.phase_conditions = phases
        self.state_space = state_space
        self.problem_id = identifier

    @property
    def unknown_size(self) -> int:
        return self.state_space.size + len(self.generators)

    def pack(self, state: ArrayLike, speeds: ArrayLike, /) -> Array:
        value = self.state_space.validate(state)
        rates = jnp.asarray(speeds, dtype=value.dtype)
        if rates.shape != (len(self.generators),):
            raise ValueError("speeds must contain one scalar per symmetry generator.")
        return jnp.concatenate((value, rates))

    def unpack(self, values: ArrayLike, /) -> tuple[Array, Array]:
        array = jnp.asarray(values)
        if array.shape != (self.unknown_size,):
            raise ValueError(
                f"Relative equilibrium unknowns must have shape {(self.unknown_size,)}."
            )
        return array[: self.state_space.size], array[self.state_space.size :]

    def residual(self, values: Array, args: Any = None, /) -> Array:
        state, speeds = self.unpack(values)
        rate = self.state_space.validate(self.vector_field(state, args))
        for speed, generator in zip(speeds, self.generators, strict=True):
            rate = rate - speed * self.state_space.validate(generator(state))
        phases = jnp.stack(
            tuple(
                jnp.asarray(phase(state, args)).reshape(())
                for phase in self.phase_conditions
            )
        ).astype(rate.dtype)
        return jnp.concatenate((rate, phases))

    def as_nonlinear_problem(self, /) -> NonlinearSystemProblem:
        space = ArraySpace((self.unknown_size,), dtype=self.state_space.dtype)
        return NonlinearSystemProblem(
            self.residual,
            state_space=space,
            residual_space=space,
            problem_id=self.problem_id,
        )


class RelativePeriodicOrbitProblem(StrictModule):
    """Multiple-shooting closure modulo a declared continuous group action."""

    evolution: AbstractDifferentiableEvolution
    group_action: Callable[[Array, Array], ArrayLike]
    temporal_phase: Callable[[Array, Array, Any], ArrayLike]
    spatial_phases: tuple[Callable[[Array, Any], ArrayLike], ...]
    num_segments: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        evolution: AbstractDifferentiableEvolution,
        group_action: Callable[[Array, Array], ArrayLike],
        temporal_phase: Callable[[Array, Array, Any], ArrayLike],
        spatial_phases: Sequence[Callable[[Array, Any], ArrayLike]],
        /,
        *,
        num_segments: int = 1,
        problem_id: str | None = None,
    ):
        if not isinstance(evolution, AbstractDifferentiableEvolution):
            raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
        phases = tuple(spatial_phases)
        if not callable(group_action) or not callable(temporal_phase):
            raise TypeError("group_action and temporal_phase must be callable.")
        if not phases or not all(callable(value) for value in phases):
            raise ValueError("spatial_phases must contain one or more callables.")
        segments = int(num_segments)
        if segments < 1:
            raise ValueError("num_segments must be positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "relative-periodic-orbit-problem-v1",
                    "evolution": evolution.evolution_id,
                    "segments": segments,
                    "group_dimension": len(phases),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.evolution = evolution
        self.group_action = group_action
        self.temporal_phase = temporal_phase
        self.spatial_phases = phases
        self.num_segments = segments
        self.problem_id = identifier

    @property
    def group_dimension(self) -> int:
        return len(self.spatial_phases)

    @property
    def unknown_size(self) -> int:
        return (
            self.num_segments * self.evolution.state_layout.size
            + 1
            + self.group_dimension
        )

    def pack(
        self,
        nodes: ArrayLike,
        period: ArrayLike,
        shifts: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(nodes)
        expected = (self.num_segments,) + self.evolution.state_layout.shape
        if values.shape != expected:
            raise ValueError(f"nodes must have shape {expected}.")
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Relative orbit nodes must use independent real coordinates.")
        period_ = jnp.asarray(period, dtype=values.dtype)
        shifts_ = jnp.asarray(shifts, dtype=values.dtype)
        if period_.shape != () or shifts_.shape != (self.group_dimension,):
            raise ValueError("period and shifts have incompatible shapes.")
        period_ = eqx.error_if(
            period_,
            ~(jnp.isfinite(period_) & (period_ > 0.0)),
            "Relative orbit period must be finite and positive.",
        )
        return jnp.concatenate((values.reshape((-1,)), jnp.log(period_)[None], shifts_))

    def unpack(self, values: ArrayLike, /) -> tuple[Array, Array, Array]:
        array = jnp.asarray(values)
        if array.shape != (self.unknown_size,):
            raise ValueError(
                f"Relative orbit unknowns must have shape {(self.unknown_size,)}."
            )
        node_size = self.num_segments * self.evolution.state_layout.size
        nodes = array[:node_size].reshape(
            (self.num_segments,) + self.evolution.state_layout.shape
        )
        period = jnp.exp(array[node_size])
        shifts = array[node_size + 1 :]
        return nodes, period, shifts

    def residual(self, values: Array, args: Any = None, /) -> Array:
        nodes, period, shifts = self.unpack(values)
        pieces = []
        for segment in range(self.num_segments):
            source = period * segment / self.num_segments
            target = period * (segment + 1) / self.num_segments
            advanced = self.evolution.advance(nodes[segment], source, target, args)
            target_state = (
                self.group_action(shifts, nodes[0])
                if segment + 1 == self.num_segments
                else nodes[segment + 1]
            )
            pieces.append((advanced.final_state - target_state).reshape((-1,)))
        temporal = jnp.asarray(self.temporal_phase(nodes[0], period, args)).reshape((1,))
        spatial = jnp.stack(
            tuple(
                jnp.asarray(phase(nodes[0], args)).reshape(())
                for phase in self.spatial_phases
            )
        )
        return jnp.concatenate(tuple(pieces) + (temporal, spatial))

    def as_nonlinear_problem(self, dtype: Any, /) -> NonlinearSystemProblem:
        space = ArraySpace((self.unknown_size,), dtype=dtype)
        return NonlinearSystemProblem(
            self.residual,
            state_space=space,
            residual_space=space,
            problem_id=self.problem_id,
        )


__all__ = ["RelativeEquilibriumProblem", "RelativePeriodicOrbitProblem"]
