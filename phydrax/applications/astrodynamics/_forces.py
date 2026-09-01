#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ...dynamics import ContinuousSystem
from ._context import AstrodynamicsContext
from ._state import CARTESIAN_ORBIT_STATE_LAYOUT
from ._status import AstrodynamicsStatus


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


class AstrodynamicsForceEvaluation(StrictModule):
    acceleration: Array
    potential: Array
    component_status: Array
    valid: Array
    status: Array
    force_id: str = eqx.field(static=True)


class AbstractAstrodynamicsForce(StrictModule):
    """Pure acceleration and potential contribution for one Cartesian state."""

    __strict_abstract__ = True
    force_id: AbstractAttribute[str]
    context: AbstractAttribute[AstrodynamicsContext]

    @abstractmethod
    def evaluate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> AstrodynamicsForceEvaluation:
        raise NotImplementedError


class PointMassGravity(AbstractAstrodynamicsForce):
    mu: Array
    context: AstrodynamicsContext
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        mu: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        force_id: str | None = None,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        coupling = jnp.asarray(mu).reshape(())
        self.mu = coupling
        self.context = context
        generated = canonical_fingerprint(
            {
                "kind": "point-mass-gravity",
                "context": context.context_id,
            }
        )
        self.force_id = generated if force_id is None else str(force_id)
        if not self.force_id:
            raise ValueError("force_id must be non-empty.")

    def evaluate(self, time, state, args=None, /) -> AstrodynamicsForceEvaluation:
        del time, args
        packed = jnp.asarray(state)
        if packed.shape != (6,):
            raise ValueError("Astrodynamics force state must have shape (6,).")
        position = packed[:3]
        radius = _norm(position)
        finite = jnp.isfinite(self.mu) & jnp.all(jnp.isfinite(packed))
        domain = finite & (self.mu > 0.0) & (radius > 0.0)
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        acceleration = -self.mu * position / safe_radius**3
        potential = -self.mu / safe_radius
        acceleration = jnp.where(domain, acceleration, jnp.zeros_like(acceleration))
        potential = jnp.where(domain, potential, jnp.asarray(jnp.nan, dtype=packed.dtype))
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                radius <= 0.0,
                int(AstrodynamicsStatus.COLLISION),
                jnp.where(
                    self.mu > 0.0,
                    int(AstrodynamicsStatus.SUCCESS),
                    int(AstrodynamicsStatus.INVALID_DOMAIN),
                ),
            ),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            acceleration,
            potential,
            status[None],
            domain,
            status,
            self.force_id,
        )


class ConstantAcceleration(AbstractAstrodynamicsForce):
    acceleration: Array
    context: AstrodynamicsContext
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        acceleration: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        force_id: str,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        value = jnp.asarray(acceleration)
        if value.shape != (3,):
            raise ValueError("Constant acceleration must have shape (3,).")
        identifier = str(force_id)
        if not identifier:
            raise ValueError("force_id must be non-empty.")
        self.acceleration = value
        self.context = context
        self.force_id = identifier

    def evaluate(self, time, state, args=None, /) -> AstrodynamicsForceEvaluation:
        del time, args
        packed = jnp.asarray(state)
        if packed.shape != (6,):
            raise ValueError("Astrodynamics force state must have shape (6,).")
        finite = jnp.all(jnp.isfinite(packed)) & jnp.all(jnp.isfinite(self.acceleration))
        status = jnp.where(
            finite,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            jnp.where(finite, self.acceleration, jnp.zeros_like(self.acceleration)),
            jnp.asarray(jnp.nan, dtype=packed.dtype),
            status[None],
            finite,
            status,
            self.force_id,
        )


class CompositeAstrodynamicsForce(AbstractAstrodynamicsForce):
    terms: tuple[AbstractAstrodynamicsForce, ...]
    context: AstrodynamicsContext
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: tuple[AbstractAstrodynamicsForce, ...],
        context: AstrodynamicsContext,
        /,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        terms_ = tuple(terms)
        if not terms_ or any(
            not isinstance(term, AbstractAstrodynamicsForce) for term in terms_
        ):
            raise ValueError("Composite force requires astrodynamics force terms.")
        for term in terms_:
            context.require_compatible(term.context)
        self.terms = terms_
        self.context = context
        self.force_id = canonical_fingerprint(
            {
                "kind": "composite-astrodynamics-force",
                "context": context.context_id,
                "terms": [term.force_id for term in terms_],
            }
        )

    def evaluate(self, time, state, args=None, /) -> AstrodynamicsForceEvaluation:
        evaluations = tuple(term.evaluate(time, state, args) for term in self.terms)
        statuses = jnp.stack(tuple(value.status for value in evaluations))
        valid = jnp.all(jnp.stack(tuple(value.valid for value in evaluations)))
        acceleration = jnp.sum(
            jnp.stack(tuple(value.acceleration for value in evaluations)), axis=0
        )
        finite_potentials = jnp.stack(
            tuple(jnp.isfinite(value.potential) for value in evaluations)
        )
        potentials = jnp.stack(tuple(value.potential for value in evaluations))
        potential = jnp.where(
            jnp.any(finite_potentials),
            jnp.sum(jnp.where(finite_potentials, potentials, 0.0)),
            jnp.asarray(jnp.nan, dtype=acceleration.dtype),
        )
        first_failure = jnp.argmax(statuses != int(AstrodynamicsStatus.SUCCESS))
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            statuses[first_failure],
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            jnp.where(valid, acceleration, jnp.zeros_like(acceleration)),
            jnp.where(valid, potential, jnp.asarray(jnp.nan, dtype=potential.dtype)),
            statuses,
            valid,
            status,
            self.force_id,
        )


class _AstrodynamicsVectorField(StrictModule):
    force: AbstractAstrodynamicsForce

    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        evaluation = self.force.evaluate(time, state, args)
        return jnp.concatenate((state[3:], evaluation.acceleration))


def astrodynamics_continuous_system(
    force: AbstractAstrodynamicsForce,
    /,
) -> ContinuousSystem:
    if not isinstance(force, AbstractAstrodynamicsForce):
        raise TypeError("force must be an AbstractAstrodynamicsForce.")
    return ContinuousSystem(
        _AstrodynamicsVectorField(force),
        state_layout=CARTESIAN_ORBIT_STATE_LAYOUT,
        system_id=f"astrodynamics-system:{force.force_id}",
    )


__all__ = [
    "AbstractAstrodynamicsForce",
    "AstrodynamicsForceEvaluation",
    "CompositeAstrodynamicsForce",
    "ConstantAcceleration",
    "PointMassGravity",
    "astrodynamics_continuous_system",
]
