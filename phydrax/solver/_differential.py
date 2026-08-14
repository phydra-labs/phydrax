#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from .._uncertainty import UncertaintySource, validate_uncertainty_source
from ..metrix import AbstractStateGeometry
from ..stochastic import WienerRealization
from ._solution_validation import validate_solution_arrays


DifferentialInterpretation: TypeAlias = Literal["ito", "stratonovich"]
NoiseStructure: TypeAlias = Literal["additive", "commutative", "general"]
DifferentialVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]


class WienerTerm(StrictModule):
    """One named independent Wiener source in a differential problem."""

    name: str = eqx.field(static=True)
    coefficient: DifferentialVectorField
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    structure: NoiseStructure = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        coefficient: DifferentialVectorField,
        noise_shape: Sequence[int],
        /,
        *,
        structure: NoiseStructure = "general",
        basis_id: str | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("WienerTerm name must be a non-empty string.")
        if not callable(coefficient):
            raise TypeError("WienerTerm coefficient must be callable.")
        shape = tuple(int(size) for size in noise_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("WienerTerm noise dimensions must be positive.")
        if structure not in ("additive", "commutative", "general"):
            raise ValueError(
                "WienerTerm structure must be 'additive', 'commutative', or 'general'."
            )
        if basis_id is not None and (not isinstance(basis_id, str) or not basis_id):
            raise ValueError("WienerTerm basis_id must be non-empty or None.")
        self.name = name
        self.coefficient = coefficient
        self.noise_shape = shape
        self.structure = structure
        self.basis_id = basis_id

    @property
    def noise_size(self) -> int:
        """Flattened control dimension contributed by this term."""
        return prod(self.noise_shape) if self.noise_shape else 1


def _noise_identity(terms: tuple[WienerTerm, ...], /) -> str | None:
    if not terms or all(term.basis_id is None for term in terms):
        return None
    if len(terms) == 1:
        return terms[0].basis_id
    digest = hashlib.sha256()
    digest.update(b"phydrax-wiener-terms\0")
    for term in terms:
        digest.update(
            repr((term.name, term.noise_shape, term.structure, term.basis_id)).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


class DifferentialProblem(StrictModule):
    """Finite-dimensional initial-value problem with named stochastic forcing."""

    drift: DifferentialVectorField
    initial_state: Array
    t0: Array
    t1: Array
    args: Any
    wiener_terms: tuple[WienerTerm, ...]
    wiener_term_slices: frozendict[str, tuple[int, int]] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    interpretation: DifferentialInterpretation = eqx.field(static=True)
    state_geometry_id: str | None = eqx.field(static=True)
    state_geometry: AbstractStateGeometry | None

    def __init__(
        self,
        drift: DifferentialVectorField,
        initial_state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
        wiener_terms: Sequence[WienerTerm] = (),
        interpretation: DifferentialInterpretation = "ito",
        state_geometry: AbstractStateGeometry | None = None,
    ):
        if not callable(drift):
            raise TypeError("DifferentialProblem drift must be callable.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if start.shape != () or end.shape != ():
            raise ValueError("DifferentialProblem t0 and t1 must be scalar.")
        start = eqx.error_if(
            start,
            ~(jnp.isfinite(start) & jnp.isfinite(end)),
            "DifferentialProblem time bounds must be finite.",
        )
        end = eqx.error_if(
            end,
            ~(end > start),
            "DifferentialProblem requires t1 > t0.",
        )
        if interpretation not in ("ito", "stratonovich"):
            raise ValueError("interpretation must be 'ito' or 'stratonovich'.")

        state = jnp.asarray(initial_state)
        if state_geometry is not None:
            if not isinstance(state_geometry, AbstractStateGeometry):
                raise TypeError(
                    "state_geometry must be an AbstractStateGeometry or None."
                )
            membership = jnp.asarray(state_geometry.contains(state), dtype=bool)
            if membership.shape != ():
                raise ValueError(
                    "State geometry contains() must return a scalar boolean."
                )
            state = eqx.error_if(
                state,
                ~membership,
                "DifferentialProblem initial_state is outside state_geometry.",
            )
        terms = tuple(wiener_terms)
        if any(not isinstance(term, WienerTerm) for term in terms):
            raise TypeError("wiener_terms must contain only WienerTerm objects.")
        names = tuple(term.name for term in terms)
        if len(set(names)) != len(names):
            raise ValueError("WienerTerm names must be unique within a problem.")

        offset = 0
        slices: dict[str, tuple[int, int]] = {}
        for term in terms:
            expected_shape = tuple(state.shape) + term.noise_shape
            coefficient = jnp.asarray(term.coefficient(start, state, args))
            if tuple(coefficient.shape) != expected_shape:
                raise ValueError(
                    f"WienerTerm {term.name!r} coefficient must return shape "
                    f"{expected_shape}; got {coefficient.shape}."
                )
            slices[term.name] = (offset, offset + term.noise_size)
            offset += term.noise_size

        self.drift = drift
        self.initial_state = state
        self.t0 = start
        self.t1 = end
        self.args = args
        self.wiener_terms = terms
        self.wiener_term_slices = frozendict(slices)
        self.noise_shape = (offset,) if terms else ()
        self.noise_id = _noise_identity(terms)
        self.interpretation = interpretation
        self.state_geometry_id = (
            None if state_geometry is None else state_geometry.geometry_id
        )
        self.state_geometry = state_geometry

    @property
    def stochastic(self) -> bool:
        return bool(self.wiener_terms)

    @property
    def additive_noise(self) -> bool:
        """Whether every stochastic term declares state-independent noise."""
        return self.stochastic and all(
            term.structure == "additive" for term in self.wiener_terms
        )


class DifferentialSolution(StrictModule):
    """Saved trajectory values plus solver and stochastic-realization provenance."""

    times: Array
    states: Array
    valid: Array
    sample_shape: tuple[int, ...]
    interpolation: Any | None
    backend_result: Any
    stats: frozendict[str, Any]
    event_mask: Any
    realization: WienerRealization | None
    wiener_term_slices: frozendict[str, tuple[int, int]] = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    interpretation: DifferentialInterpretation = eqx.field(static=True)
    state_geometry_id: str | None = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        sample_shape: Sequence[int] = (),
        interpolation: Any | None = None,
        backend_result: Any,
        stats: dict[str, Any] | frozendict[str, Any],
        event_mask: Any = None,
        realization: WienerRealization | None = None,
        wiener_term_slices: dict[str, tuple[int, int]]
        | frozendict[str, tuple[int, int]]
        | None = None,
        solver_name: str,
        interpretation: DifferentialInterpretation,
        state_geometry_id: str | None = None,
        solver_id: str | None = None,
        resolved_method: str | None = None,
    ):
        arrays = validate_solution_arrays(
            times,
            states,
            valid,
            sample_shape=sample_shape,
            state_shape=None,
            time_layout="per_path",
            owner="DifferentialSolution",
        )
        samples = arrays.sample_shape
        times_array = arrays.times
        states_array = arrays.states
        valid_array = arrays.valid
        if not isinstance(solver_name, str) or not solver_name:
            raise ValueError("DifferentialSolution solver_name must be non-empty.")
        resolved_solver_id = f"solver:{solver_name}" if solver_id is None else solver_id
        resolved_solver_method = (
            solver_name if resolved_method is None else resolved_method
        )
        if not isinstance(resolved_solver_id, str) or not resolved_solver_id:
            raise ValueError("DifferentialSolution solver_id must be non-empty.")
        if not isinstance(resolved_solver_method, str) or not resolved_solver_method:
            raise ValueError("DifferentialSolution resolved_method must be non-empty.")
        if interpretation not in ("ito", "stratonovich"):
            raise ValueError("interpretation must be 'ito' or 'stratonovich'.")
        if state_geometry_id is not None and (
            not isinstance(state_geometry_id, str) or not state_geometry_id
        ):
            raise ValueError("state_geometry_id must be a non-empty string or None.")
        if realization is not None:
            if not isinstance(realization, WienerRealization):
                raise TypeError("realization must be a WienerRealization or None.")
            if realization.sample_shape != samples:
                raise ValueError(
                    "DifferentialSolution realization sample_shape must match the "
                    f"solution; got {realization.sample_shape} and {samples}."
                )
        if interpolation is not None and not callable(
            getattr(interpolation, "evaluate", None)
        ):
            raise TypeError("DifferentialSolution interpolation must define evaluate().")
        self.times = times_array
        self.states = states_array
        self.valid = valid_array
        self.sample_shape = samples
        self.interpolation = interpolation
        self.backend_result = backend_result
        self.stats = frozendict(dict(stats))
        self.event_mask = event_mask
        self.realization = realization
        self.wiener_term_slices = frozendict(
            {} if wiener_term_slices is None else dict(wiener_term_slices)
        )
        self.solver_name = solver_name
        self.interpretation = interpretation
        self.state_geometry_id = state_geometry_id
        self.solver_id = resolved_solver_id
        self.resolved_method = resolved_solver_method

    @property
    def num_times(self) -> int:
        return int(self.times.shape[-1])

    @property
    def successful(self) -> Array:
        """Whether every requested saved value is finite for each realization."""
        return jnp.all(self.valid, axis=-1)

    @property
    def has_dense_interpolation(self) -> bool:
        """Whether dense evaluation is available between saved times."""
        return self.interpolation is not None

    def evaluate(
        self,
        query_times: ArrayLike,
        /,
        *,
        left: bool = True,
    ) -> Array:
        """Evaluate dense output with shape sample_shape + query_shape + state_shape."""
        if self.interpolation is None:
            raise ValueError(
                "DifferentialSolution has no dense interpolation; "
                "call solve_diffrax or solve_diffrax_ensemble with dense=True."
            )
        return self.interpolation.evaluate(query_times, left=left)

    def to_stochastic_trajectory(
        self,
        /,
        *,
        initial_state: ArrayLike | None = None,
        initial_time: ArrayLike | None = None,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] = ("state",),
        case_id: str = "case:0",
        parameter_id: str | None = None,
        discretization_id: str | None = None,
        basis_id: str | None = None,
        approximation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Convert to an axis-explicit stochastic trajectory without flattening paths."""
        from ..stochastic._trajectory import StochasticTrajectory

        return StochasticTrajectory.from_solution(
            self,
            initial_state=initial_state,
            initial_time=initial_time,
            realization_axes=realization_axes,
            state_axes=state_axes,
            case_id=case_id,
            parameter_id=parameter_id,
            discretization_id=discretization_id,
            basis_id=basis_id,
            approximation_id=approximation_id,
            metadata=metadata,
        )

    def to_predictive(
        self,
        /,
        *,
        sample_dim: str = "__phydra_uq_process",
        time_dim: str = "t",
        state_dims: Sequence[str | None] | None = None,
        source: UncertaintySource = "process",
    ) -> Any:
        """Convert an ensemble trajectory to a coordinate-aware predictive field."""
        if len(self.sample_shape) != 1:
            raise ValueError(
                "DifferentialSolution.to_predictive requires one ensemble sample axis."
            )
        if not isinstance(sample_dim, str) or not sample_dim:
            raise ValueError("sample_dim must be a non-empty string.")
        if not isinstance(time_dim, str) or not time_dim or time_dim == sample_dim:
            raise ValueError("time_dim must be non-empty and distinct from sample_dim.")
        source_value = validate_uncertainty_source(source)
        state_ndim = self.states.ndim - 2
        if state_dims is None:
            resolved_state_dims = (None,) * state_ndim
        else:
            resolved_state_dims = tuple(state_dims)
            if len(resolved_state_dims) != state_ndim:
                raise ValueError(
                    f"state_dims must contain {state_ndim} entries; "
                    f"got {len(resolved_state_dims)}."
                )
        from ..uq._predictive import PredictiveField, SampleAxis

        samples = cx.Field(
            self.states,
            dims=(sample_dim, time_dim) + resolved_state_dims,
        )
        sample_valid = cx.Field(self.successful, dims=(sample_dim,))
        return PredictiveField(
            samples,
            (SampleAxis(sample_dim, source_value),),
            valid=sample_valid,
        )


__all__ = [
    "DifferentialInterpretation",
    "DifferentialProblem",
    "DifferentialSolution",
    "DifferentialVectorField",
    "NoiseStructure",
    "WienerTerm",
]
