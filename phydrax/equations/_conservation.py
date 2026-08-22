#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    FiniteVolumeDiscretization,
    FirstOrderFiniteVolumeDynamics,
)


class ConservationProblemIR(StrictModule):
    """Scalar conservation law with explicit flux, source, and exterior state."""

    flux: Callable[[Array, Array, Array, Any], ArrayLike]
    wave_speed: Callable[[Array, Array, Array, Any], ArrayLike]
    exterior_state: Callable[[Array, Array, Array, Array, Any], ArrayLike] | None
    source: Callable[[Array, Array, Array, Any], ArrayLike] | None
    name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        field_name: str,
        flux: Callable[[Array, Array, Array, Any], ArrayLike],
        wave_speed: Callable[[Array, Array, Array, Any], ArrayLike],
        /,
        *,
        exterior_state: Callable[[Array, Array, Array, Array, Any], ArrayLike]
        | None = None,
        source: Callable[[Array, Array, Array, Any], ArrayLike] | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        field = str(field_name)
        if not name_ or not field:
            raise ValueError("Conservation problem and field names must be non-empty.")
        if not callable(flux) or not callable(wave_speed):
            raise TypeError("flux and wave_speed must be callable.")
        if exterior_state is not None and not callable(exterior_state):
            raise TypeError("exterior_state must be callable or None.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        self.flux = flux
        self.wave_speed = wave_speed
        self.exterior_state = exterior_state
        self.source = source
        self.name = name_
        self.field_name = field
        self.problem_id = (
            canonical_fingerprint(
                {
                    "kind": "scalar-conservation-problem",
                    "name": name_,
                    "field": field,
                    "flux": repr(flux),
                    "wave_speed": repr(wave_speed),
                    "exterior_state": None
                    if exterior_state is None
                    else repr(exterior_state),
                    "source": None if source is None else repr(source),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty.")


class CompiledConservationProblem(StrictModule):
    """Executable conservative residual with complete discretization provenance."""

    problem: ConservationProblemIR
    discretization: FiniteVolumeDiscretization
    dynamics: FirstOrderFiniteVolumeDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: ConservationProblemIR,
        discretization: FiniteVolumeDiscretization,
        dynamics: FirstOrderFiniteVolumeDynamics,
        /,
    ):
        if problem.field_name != discretization.field_spaces[0].name:
            raise ValueError("Conserved field name must match the finite-volume space.")
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-conservation-problem",
                "problem": problem.problem_id,
                "discretization": discretization.prepared_id,
            }
        )
        form_key = DiscretizationKey(
            "conservation_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        self.problem = problem
        self.discretization = discretization
        self.dynamics = dynamics
        self.discretization_bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                ),
                DiscretizationRecord(
                    form_key,
                    "compiled-conservation-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.compilation_id = compilation_id

    def face_flux(self, time: Array, state: Array, args: Any = None, /) -> Array:
        return self.dynamics.face_flux(time, state, args)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        return self.dynamics(time, state, args)


def compile_conservation_problem(
    problem: ConservationProblemIR,
    discretization: FiniteVolumeDiscretization,
    /,
) -> CompiledConservationProblem:
    """Lower one scalar conservation law onto prepared first-order finite volumes."""
    if not isinstance(problem, ConservationProblemIR):
        raise TypeError("problem must be a ConservationProblemIR.")
    if not isinstance(discretization, FiniteVolumeDiscretization):
        raise TypeError(
            "No conservation lowering is registered for this discretization type."
        )
    dynamics = discretization.first_order_dynamics(
        problem.flux,
        problem.wave_speed,
        exterior_state=problem.exterior_state,
        source=problem.source,
    )
    return CompiledConservationProblem(problem, discretization, dynamics)


__all__ = [
    "CompiledConservationProblem",
    "ConservationProblemIR",
    "compile_conservation_problem",
]
