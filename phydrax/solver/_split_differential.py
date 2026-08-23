#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from ..discretization import DiscretizationBundle
from ..metrix import AbstractStateGeometry
from ._differential import DifferentialVectorField
from ._semilinear_drift import SemilinearDrift


class SplitDifferentialProblem(StrictModule):
    """Deterministic additive ODE ``y' = f_explicit + f_implicit``."""

    explicit_drift: DifferentialVectorField
    implicit_drift: DifferentialVectorField
    initial_state: Array
    t0: Array
    t1: Array
    args: Any
    state_geometry: AbstractStateGeometry | None
    discretization_bundle: DiscretizationBundle | None
    wiener_term_slices: frozendict[str, tuple[int, int]] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    interpretation: str = eqx.field(static=True)
    state_geometry_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        explicit_drift: DifferentialVectorField,
        implicit_drift: DifferentialVectorField,
        initial_state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
        state_geometry: AbstractStateGeometry | None = None,
        discretization_bundle: DiscretizationBundle | None = None,
        problem_id: str | None = None,
    ):
        if not callable(explicit_drift) or not callable(implicit_drift):
            raise TypeError("Split differential drifts must be callable.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if start.shape != () or end.shape != ():
            raise ValueError("Split differential time bounds must be scalar.")
        start = eqx.error_if(
            start,
            ~(jnp.isfinite(start) & jnp.isfinite(end)),
            "Split differential time bounds must be finite.",
        )
        end = eqx.error_if(
            end,
            ~(end > start),
            "SplitDifferentialProblem requires t1 > t0.",
        )
        state = jnp.asarray(initial_state)
        geometry_id = None
        if state_geometry is not None:
            if not isinstance(state_geometry, AbstractStateGeometry):
                raise TypeError(
                    "state_geometry must be an AbstractStateGeometry or None."
                )
            if not state_geometry.trivial:
                raise ValueError(
                    "SplitDifferentialProblem currently requires Euclidean geometry."
                )
            membership = jnp.asarray(state_geometry.contains(state), dtype=bool)
            if membership.shape != ():
                raise ValueError("State geometry contains() must return one scalar.")
            state = eqx.error_if(
                state,
                ~membership,
                "Split differential initial state is outside state_geometry.",
            )
            geometry_id = state_geometry.geometry_id
        if discretization_bundle is not None and not isinstance(
            discretization_bundle, DiscretizationBundle
        ):
            raise TypeError(
                "discretization_bundle must be a DiscretizationBundle or None."
            )
        for function, owner in (
            (explicit_drift, "explicit_drift"),
            (implicit_drift, "implicit_drift"),
        ):
            value = jnp.asarray(function(start, state, args))
            if value.shape != state.shape:
                raise ValueError(
                    f"{owner} must preserve state shape {state.shape}; got {value.shape}."
                )
        bundle_id = (
            None if discretization_bundle is None else discretization_bundle.bundle_id
        )
        if problem_id is None:
            explicit_type = type(explicit_drift)
            implicit_type = type(implicit_drift)
            payload = {
                "explicit": (f"{explicit_type.__module__}.{explicit_type.__qualname__}"),
                "implicit": (f"{implicit_type.__module__}.{implicit_type.__qualname__}"),
                "state_shape": list(state.shape),
                "state_dtype": str(state.dtype),
                "geometry_id": geometry_id,
                "discretization_bundle_id": bundle_id,
            }
            identifier = f"split-differential-problem:{canonical_fingerprint(payload)}"
        else:
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError("problem_id must be non-empty or None.")
            identifier = problem_id
        self.explicit_drift = explicit_drift
        self.implicit_drift = implicit_drift
        self.initial_state = state
        self.t0 = start
        self.t1 = end
        self.args = args
        self.state_geometry = state_geometry
        self.discretization_bundle = discretization_bundle
        self.wiener_term_slices = frozendict()
        self.noise_shape = ()
        self.noise_id = None
        self.interpretation = "ito"
        self.state_geometry_id = geometry_id
        self.problem_id = identifier
        self.discretization_bundle_id = bundle_id

    @property
    def stochastic(self) -> bool:
        return False

    @property
    def additive_noise(self) -> bool:
        return False


class _SemilinearExplicitDrift(eqx.Module):
    drift: SemilinearDrift

    def __call__(self, time, state, args):
        return self.drift.nonlinear(time, state, args)


class _SemilinearImplicitDrift(eqx.Module):
    drift: SemilinearDrift

    def __call__(self, time, state, args):
        del time, args
        return self.drift.linear(state)


def split_differential_problem(
    compiled: Any,
    initial_state: ArrayLike,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    args: Any = None,
    problem_id: str | None = None,
) -> SplitDifferentialProblem:
    """Bind a certified semilinear PDE compilation to additive ODE integration."""
    from ..equations import CompiledDiscreteDynamics

    if not isinstance(compiled, CompiledDiscreteDynamics):
        raise TypeError("compiled must be CompiledDiscreteDynamics.")
    drift = compiled.semilinear_drift
    if not isinstance(drift, SemilinearDrift):
        raise ValueError("Compiled dynamics has no certified semilinear split.")
    return SplitDifferentialProblem(
        _SemilinearExplicitDrift(drift),
        _SemilinearImplicitDrift(drift),
        initial_state,
        t0=t0,
        t1=t1,
        args=args,
        discretization_bundle=compiled.discretization_bundle,
        problem_id=problem_id,
    )


__all__ = ["SplitDifferentialProblem", "split_differential_problem"]
