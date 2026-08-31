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
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from .._uncertainty import UncertaintySource, validate_uncertainty_source
from ..discretization import DiscretizationBundle
from ..metrix import AbstractStateGeometry
from ..stochastic import WienerRealization
from ._solution_validation import validate_solution_arrays
from ._temporal_method import TemporalSolveEvidence
from ._wiener_operator import WienerNoiseLayout


DifferentialInterpretation: TypeAlias = Literal["ito", "stratonovich"]
NoiseStructure: TypeAlias = Literal["additive", "commutative", "general"]
WienerCoefficientRepresentation: TypeAlias = Literal["dense", "diagonal", "operator"]
DifferentialVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
WienerCoefficient: TypeAlias = Callable[[Array, Array, Any], Any]


class WienerTerm(StrictModule):
    """One named independent Wiener source in a differential problem."""

    name: str = eqx.field(static=True)
    coefficient: WienerCoefficient
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    structure: NoiseStructure = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)
    representation: WienerCoefficientRepresentation = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        coefficient: WienerCoefficient,
        noise_shape: Sequence[int],
        /,
        *,
        structure: NoiseStructure = "general",
        basis_id: str | None = None,
        representation: WienerCoefficientRepresentation = "dense",
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
        if representation not in ("dense", "diagonal", "operator"):
            raise ValueError(
                "WienerTerm representation must be 'dense', 'diagonal', or 'operator'."
            )
        if basis_id is not None and (not isinstance(basis_id, str) or not basis_id):
            raise ValueError("WienerTerm basis_id must be non-empty or None.")
        self.name = name
        self.coefficient = coefficient
        self.noise_shape = shape
        self.structure = structure
        self.basis_id = basis_id
        self.representation = representation

    @property
    def noise_size(self) -> int:
        """Flattened control dimension contributed by this term."""
        return prod(self.noise_shape) if self.noise_shape else 1

    def coefficient_array(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Evaluate a coefficient in its declared array representation."""
        if self.representation == "operator":
            raise ValueError(
                "An operator Wiener coefficient has no implicit array representation."
            )
        time_array = jnp.asarray(time)
        state_array = jnp.asarray(state)
        expected_shape = (
            tuple(state_array.shape) + self.noise_shape
            if self.representation == "dense"
            else tuple(state_array.shape)
        )
        coefficient = jnp.asarray(self.coefficient(time_array, state_array, args))
        if tuple(coefficient.shape) != expected_shape:
            raise ValueError(
                f"WienerTerm {self.name!r} coefficient must return shape "
                f"{expected_shape}; got {coefficient.shape}."
            )
        return coefficient

    def coefficient_operator(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> lx.AbstractLinearOperator:
        """Evaluate and validate one explicitly operator-valued coefficient."""
        if self.representation != "operator":
            raise ValueError("coefficient_operator requires representation='operator'.")
        state_array = jnp.asarray(state)
        operator = self.coefficient(jnp.asarray(time), state_array, args)
        if not isinstance(operator, lx.AbstractLinearOperator):
            raise TypeError("Operator Wiener coefficients must return a Lineax operator.")
        input_structure = operator.in_structure()
        output_structure = operator.out_structure()
        if not isinstance(input_structure, jax.ShapeDtypeStruct) or not isinstance(
            output_structure, jax.ShapeDtypeStruct
        ):
            raise TypeError("Operator Wiener coefficients initially require array spaces.")
        if tuple(input_structure.shape) != self.noise_shape:
            raise ValueError(
                "Operator Wiener input structure must match the declared noise shape."
            )
        if tuple(output_structure.shape) != tuple(state_array.shape):
            raise ValueError(
                "Operator Wiener output structure must match the complete state shape."
            )
        return operator

    def coefficient_matrix(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Evaluate a declared dense coefficient as ``(state_size, noise_size)``."""
        if self.representation != "dense":
            raise ValueError(
                "A structured Wiener coefficient has no implicit dense matrix; "
                "use a backend with structured-operator support."
            )
        state_array = jnp.asarray(state)
        coefficient = self.coefficient_array(time, state_array, args)
        state_size = prod(state_array.shape) if state_array.shape else 1
        return coefficient.reshape((state_size, self.noise_size))


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


def _problem_identifier(
    value: str | None,
    drift: DifferentialVectorField,
    state: Array,
    terms: tuple[WienerTerm, ...],
    geometry_id: str | None,
    bundle_id: str | None,
    /,
) -> str:
    if value is not None:
        if not isinstance(value, str) or not value:
            raise ValueError("DifferentialProblem problem_id must be non-empty or None.")
        return value
    drift_type = type(drift)
    payload = {
        "kind": "differential-problem",
        "drift": f"{drift_type.__module__}.{drift_type.__qualname__}",
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype),
        "wiener_terms": [
            {
                "name": term.name,
                "noise_shape": list(term.noise_shape),
                "structure": term.structure,
                "basis_id": term.basis_id,
                "representation": term.representation,
            }
            for term in terms
        ],
        "geometry_id": geometry_id,
        "discretization_bundle_id": bundle_id,
    }
    return f"differential-problem:{canonical_fingerprint(payload)}"


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
    noise_layout: WienerNoiseLayout | None
    interpretation: DifferentialInterpretation = eqx.field(static=True)
    state_geometry_id: str | None = eqx.field(static=True)
    state_geometry: AbstractStateGeometry | None
    problem_id: str = eqx.field(static=True)
    discretization_bundle: DiscretizationBundle | None
    discretization_bundle_id: str | None = eqx.field(static=True)

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
        discretization_bundle: DiscretizationBundle | None = None,
        problem_id: str | None = None,
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
        structured = tuple(
            term for term in terms if term.representation in ("diagonal", "operator")
        )
        for term in structured:
            if term.representation == "diagonal" and term.noise_shape != tuple(state.shape):
                raise ValueError(
                    "A diagonal WienerTerm requires matching state and noise shapes."
                )

        offset = 0
        slices: dict[str, tuple[int, int]] = {}
        for term in terms:
            if term.representation == "operator":
                term.coefficient_operator(start, state, args)
            else:
                term.coefficient_array(start, state, args)
            slices[term.name] = (offset, offset + term.noise_size)
            offset += term.noise_size
        noise_layout = (
            None
            if not terms
            else WienerNoiseLayout(
                tuple((term.name, term.noise_shape, term.basis_id) for term in terms)
            )
        )

        if discretization_bundle is not None and not isinstance(
            discretization_bundle, DiscretizationBundle
        ):
            raise TypeError(
                "discretization_bundle must be a DiscretizationBundle or None."
            )
        geometry_id = None if state_geometry is None else state_geometry.geometry_id
        bundle_id = (
            None if discretization_bundle is None else discretization_bundle.bundle_id
        )
        self.drift = drift
        self.initial_state = state
        self.t0 = start
        self.t1 = end
        self.args = args
        self.wiener_terms = terms
        self.wiener_term_slices = frozendict(slices)
        self.noise_shape = (offset,) if terms else ()
        self.noise_layout = noise_layout
        self.noise_id = _noise_identity(terms)
        self.interpretation = interpretation
        self.state_geometry_id = geometry_id
        self.state_geometry = state_geometry
        self.discretization_bundle = discretization_bundle
        self.discretization_bundle_id = bundle_id
        self.problem_id = _problem_identifier(
            problem_id,
            drift,
            state,
            terms,
            geometry_id,
            bundle_id,
        )

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
    discretization_bundle: DiscretizationBundle | None
    backend_successful: Array
    event_terminated: Array
    temporal_evidence: TemporalSolveEvidence | None
    wiener_term_slices: frozendict[str, tuple[int, int]] = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    interpretation: DifferentialInterpretation = eqx.field(static=True)
    state_geometry_id: str | None = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)
    problem_id: str | None = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

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
        discretization_bundle: DiscretizationBundle | None = None,
        backend_successful: ArrayLike = True,
        event_terminated: ArrayLike = False,
        temporal_evidence: TemporalSolveEvidence | None = None,
        problem_id: str | None = None,
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
        if discretization_bundle is not None and not isinstance(
            discretization_bundle,
            DiscretizationBundle,
        ):
            raise TypeError(
                "discretization_bundle must be a DiscretizationBundle or None."
            )
        backend_ok = jnp.asarray(backend_successful, dtype=bool)
        event_stop = jnp.asarray(event_terminated, dtype=bool)
        if backend_ok.shape not in ((), samples) or event_stop.shape not in ((), samples):
            raise ValueError(
                "backend_successful and event_terminated must be scalar or have "
                f"sample shape {samples}."
            )
        backend_ok = jnp.broadcast_to(backend_ok, samples)
        event_stop = jnp.broadcast_to(event_stop, samples)
        if temporal_evidence is not None and not isinstance(
            temporal_evidence, TemporalSolveEvidence
        ):
            raise TypeError("temporal_evidence must be TemporalSolveEvidence or None.")
        if problem_id is not None and (not isinstance(problem_id, str) or not problem_id):
            raise ValueError("problem_id must be non-empty or None.")
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
        self.discretization_bundle = discretization_bundle
        self.backend_successful = backend_ok
        self.event_terminated = event_stop
        self.temporal_evidence = temporal_evidence
        self.problem_id = problem_id
        self.discretization_bundle_id = (
            None if discretization_bundle is None else discretization_bundle.bundle_id
        )

    @property
    def num_times(self) -> int:
        return int(self.times.shape[-1])

    @property
    def successful(self) -> Array:
        """Whether the backend succeeded and every requested value is finite."""
        return self.backend_successful & jnp.all(self.valid, axis=-1)

    @property
    def completed(self) -> Array:
        """Whether the solve succeeded without an intentional event termination."""
        return self.successful & ~self.event_terminated

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
    "WienerCoefficientRepresentation",
    "WienerTerm",
]
