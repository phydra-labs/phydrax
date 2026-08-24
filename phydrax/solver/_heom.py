#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._temporal_precision import TemporalPrecisionPolicy
from ..linalg import HermitianPrecisionPolicy, HermitianSpectrum
from ..operators.quantum import (
    ApproximationAxis,
    ApproximationQuantity,
    BathCorrelationExpansion,
    OpenSystemApproximationEvidence,
    OpenSystemPhysicalityEvidence,
)


class HEOMHierarchy(StrictModule):
    multi_indices: Array
    levels: Array
    upward: Array
    downward: Array
    depth: int = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    auxiliary_count: int = eqx.field(static=True)

    def __init__(self, term_count: int, depth: int, /):
        terms = int(term_count)
        depth_ = int(depth)
        if terms < 1 or depth_ < 0:
            raise ValueError("HEOM term count/depth are invalid.")
        indices = tuple(
            values
            for values in product(range(depth_ + 1), repeat=terms)
            if sum(values) <= depth_
        )
        lookup = {values: index for index, values in enumerate(indices)}
        upward = []
        downward = []
        for values in indices:
            up_row = []
            down_row = []
            for term in range(terms):
                up = list(values)
                up[term] += 1
                down = list(values)
                down[term] -= 1
                up_row.append(lookup.get(tuple(up), -1))
                down_row.append(lookup.get(tuple(down), -1) if down[term] >= 0 else -1)
            upward.append(up_row)
            downward.append(down_row)
        self.multi_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.levels = jnp.sum(self.multi_indices, axis=-1)
        self.upward = jnp.asarray(upward, dtype=jnp.int32)
        self.downward = jnp.asarray(downward, dtype=jnp.int32)
        self.depth = depth_
        self.term_count = terms
        self.auxiliary_count = len(indices)


class HEOMProblem(StrictModule):
    hamiltonian: Array
    coupling_operator: Array
    expansion: BathCorrelationExpansion
    hierarchy: HEOMHierarchy
    initial_state: Array
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: ArrayLike,
        coupling_operator: ArrayLike,
        expansion: BathCorrelationExpansion,
        hierarchy: HEOMHierarchy,
        initial_density: ArrayLike,
        /,
        *,
        geometry_precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
        problem_id: str = "heom",
    ):
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be HermitianPrecisionPolicy or None."
            )
        hamiltonian_ = jnp.asarray(hamiltonian)
        if geometry_.coordinate_dtype is not None:
            hamiltonian_ = hamiltonian_.astype(geometry_.coordinate_dtype)
        coupling = jnp.asarray(coupling_operator, dtype=hamiltonian_.dtype)
        density = jnp.asarray(initial_density, dtype=hamiltonian_.dtype)
        geometry_.validate_coordinates(density)
        if hamiltonian_.ndim != 2 or hamiltonian_.shape[0] != hamiltonian_.shape[1]:
            raise ValueError("HEOM Hamiltonian must be square.")
        if coupling.shape != hamiltonian_.shape or density.shape != hamiltonian_.shape:
            raise ValueError("HEOM operator/state shapes are incompatible.")
        if hierarchy.term_count != expansion.rank:
            raise ValueError("HEOM hierarchy and bath expansion ranks differ.")
        auxiliaries = (
            jnp.zeros((hierarchy.auxiliary_count,) + density.shape, dtype=density.dtype)
            .at[0]
            .set(density)
        )
        root_spectrum = HermitianSpectrum(density, precision=hermitian_)
        self.hamiltonian = hamiltonian_
        self.coupling_operator = coupling
        self.expansion = expansion
        self.hierarchy = hierarchy
        self.initial_state = auxiliaries
        self.geometry_precision = geometry_
        self.hermitian_precision = hermitian_
        self.precision_evidence = geometry_.evidence_for(
            density,
            children={"root-spectrum": root_spectrum.precision_evidence},
        )
        self.problem_id = str(problem_id)

    def rhs(
        self,
        auxiliaries: ArrayLike,
        /,
        *,
        precision: TemporalPrecisionPolicy | None = None,
    ) -> Array:
        precision_ = TemporalPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TemporalPrecisionPolicy):
            raise TypeError("precision must be TemporalPrecisionPolicy or None.")
        values = precision_.stage(auxiliaries)
        result = jnp.zeros_like(values)
        coupling = self.coupling_operator
        exponents = precision_.accumulation(self.expansion.exponents)
        for auxiliary in range(self.hierarchy.auxiliary_count):
            rho = values[auxiliary]
            derivative = -1j * (self.hamiltonian @ rho - rho @ self.hamiltonian)
            occupation = self.hierarchy.multi_indices[auxiliary]
            decay = jnp.sum(occupation * exponents)
            derivative = derivative - decay * rho
            for term in range(self.hierarchy.term_count):
                up = self.hierarchy.upward[auxiliary, term]
                down = self.hierarchy.downward[auxiliary, term]
                upper = values[jnp.maximum(up, 0)]
                upper_term = -1j * (coupling @ upper - upper @ coupling)
                derivative = derivative + jnp.where(up >= 0, upper_term, 0.0)
                lower = values[jnp.maximum(down, 0)]
                coefficient = precision_.accumulation(self.expansion.coefficients[term])
                lower_term = (
                    -1j
                    * occupation[term]
                    * (
                        coefficient * coupling @ lower
                        - jnp.conj(coefficient) * lower @ coupling
                    )
                )
                derivative = derivative + jnp.where(down >= 0, lower_term, 0.0)
            result = result.at[auxiliary].set(precision_.residual(derivative))
        return result


class HEOMSolution(StrictModule):
    root_states: Array
    final_auxiliaries: Array
    times: Array
    maximum_auxiliary_norm_by_level: Array
    approximation: OpenSystemApproximationEvidence
    physicality: OpenSystemPhysicalityEvidence
    valid: Array
    temporal_precision: TemporalPrecisionPolicy
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: HEOMProblem,
        root_states: ArrayLike,
        final_auxiliaries: ArrayLike,
        times: ArrayLike,
        /,
        *,
        step_size: ArrayLike,
        temporal_precision: TemporalPrecisionPolicy,
        geometry_precision: GeometryPrecisionPolicy,
        hermitian_precision: HermitianPrecisionPolicy,
        maximum_time_step: float = 0.1,
    ):
        roots = jnp.asarray(root_states)
        auxiliaries = jnp.asarray(final_auxiliaries)
        times_ = jnp.asarray(times)
        temporal_precision.validate_state(auxiliaries)
        geometry_precision.validate_coordinates(roots[0])
        norms = geometry_precision.norm(auxiliaries, axis=(-2, -1))
        level_norms = jnp.stack(
            [
                jnp.max(jnp.where(problem.hierarchy.levels == level, norms, 0.0))
                for level in range(problem.hierarchy.depth + 1)
            ]
        )
        trace = geometry_precision.accumulation(jnp.trace(roots, axis1=-2, axis2=-1))
        trace_residual = geometry_precision.decision(jnp.max(jnp.abs(trace - 1.0)))
        hermiticity = geometry_precision.decision(
            jnp.max(
                jnp.abs(
                    geometry_precision.accumulation(
                        roots - jnp.swapaxes(jnp.conj(roots), -1, -2)
                    )
                )
            )
        )
        root_spectrum = HermitianSpectrum(
            roots,
            precision=hermitian_precision,
        )
        positivity_margin = geometry_precision.decision(
            jnp.min(root_spectrum.minimum_eigenvalue)
        )
        valid = (
            jnp.all(jnp.isfinite(roots))
            & (trace_residual <= 1e-6)
            & (hermiticity <= 1e-6)
            & (positivity_margin >= -1e-6)
            & jnp.all(root_spectrum.valid)
        )
        self.root_states = temporal_precision.output(roots)
        self.final_auxiliaries = temporal_precision.output(auxiliaries)
        self.times = times_
        self.maximum_auxiliary_norm_by_level = geometry_precision.decision(level_norms)
        self.temporal_precision = temporal_precision
        self.geometry_precision = geometry_precision
        self.hermitian_precision = hermitian_precision
        self.precision_evidence = temporal_precision.evidence_for(
            auxiliaries,
            times_[-1],
            children={
                "hierarchy-reduction": geometry_precision.evidence_for(roots[0]),
                "root-spectrum": root_spectrum.precision_evidence,
            },
        )
        self.approximation = OpenSystemApproximationEvidence(
            "heom",
            (
                ApproximationAxis("hierarchy-depth", problem.hierarchy.depth),
                ApproximationAxis(
                    "bath-expansion-rank",
                    problem.hierarchy.term_count,
                ),
                ApproximationAxis("time-step", step_size, units="time"),
            ),
            (
                ApproximationQuantity(
                    "time-step",
                    temporal_precision.decision(step_size),
                    maximum_time_step,
                    units="time",
                    norm_id="absolute",
                    estimate_kind="estimate",
                ),
            ),
            execution_valid=valid,
            precision_evidence=self.precision_evidence,
            precision_policy_ids=(
                temporal_precision.policy_id,
                geometry_precision.policy_id,
                hermitian_precision.policy_id,
            ),
        )
        self.physicality = OpenSystemPhysicalityEvidence(
            trace_residual=trace_residual,
            hermiticity_residual=hermiticity,
            positivity_margin=positivity_margin,
            certified_properties=("trace", "hermiticity", "positivity"),
            precision_evidence=self.precision_evidence,
        )
        self.valid = valid
        self.problem_id = problem.problem_id


def solve_heom(
    problem: HEOMProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    temporal_precision: TemporalPrecisionPolicy | None = None,
    geometry_precision: GeometryPrecisionPolicy | None = None,
    hermitian_precision: HermitianPrecisionPolicy | None = None,
) -> HEOMSolution:
    if not isinstance(problem, HEOMProblem):
        raise TypeError("problem must be HEOMProblem.")
    temporal_ = (
        TemporalPrecisionPolicy() if temporal_precision is None else temporal_precision
    )
    geometry_ = (
        problem.geometry_precision if geometry_precision is None else geometry_precision
    )
    hermitian_ = (
        problem.hermitian_precision
        if hermitian_precision is None
        else hermitian_precision
    )
    if not isinstance(temporal_, TemporalPrecisionPolicy):
        raise TypeError("temporal_precision must be TemporalPrecisionPolicy or None.")
    if not isinstance(geometry_, GeometryPrecisionPolicy):
        raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
    if not isinstance(hermitian_, HermitianPrecisionPolicy):
        raise TypeError("hermitian_precision must be HermitianPrecisionPolicy or None.")
    temporal_.validate_state(problem.initial_state)
    step = temporal_.coefficient(
        jnp.asarray(step_size, dtype=problem.initial_state.real.dtype)
    ).reshape(())
    count = int(steps)
    if count < 0 or float(step) <= 0.0:
        raise ValueError("HEOM steps and step_size must be positive.")

    def advance(state, _):
        k1 = temporal_.stage(problem.rhs(state, precision=temporal_))
        k2 = temporal_.stage(
            problem.rhs(
                state + 0.5 * step * k1,
                precision=temporal_,
            )
        )
        k3 = temporal_.stage(
            problem.rhs(
                state + 0.5 * step * k2,
                precision=temporal_,
            )
        )
        k4 = temporal_.stage(
            problem.rhs(
                state + step * k3,
                precision=temporal_,
            )
        )
        increment = temporal_.accumulation(k1 + 2 * k2 + 2 * k3 + k4)
        next_state = jnp.asarray(
            state + step * increment / 6,
            dtype=state.dtype,
        )
        return next_state, next_state[0]

    final, roots = jax.lax.scan(advance, problem.initial_state, xs=None, length=count)
    roots = jnp.concatenate((problem.initial_state[0][None, ...], roots), axis=0)
    return HEOMSolution(
        problem,
        roots,
        final,
        step * jnp.arange(count + 1),
        step_size=step,
        temporal_precision=temporal_,
        geometry_precision=geometry_,
        hermitian_precision=hermitian_,
    )


def drude_lorentz_qubit_heom(
    coupling_strength: float,
    decay_rate: float,
    initial_density: ArrayLike,
    /,
    *,
    depth: int = 2,
) -> HEOMProblem:
    expansion = BathCorrelationExpansion(
        jnp.asarray([float(coupling_strength)], dtype=complex),
        jnp.asarray([float(decay_rate)], dtype=complex),
        expansion_id="drude-lorentz-one-term",
    )
    hierarchy = HEOMHierarchy(1, depth)
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=complex)
    return HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        sigma_z,
        expansion,
        hierarchy,
        initial_density,
        problem_id="drude-lorentz-qubit-heom",
    )


def thermal_drude_lorentz_qubit_heom(
    coupling_strength: float,
    decay_rate: float,
    temperature: float,
    initial_density: ArrayLike,
    /,
    *,
    depth: int = 2,
) -> HEOMProblem:
    """One-term finite-temperature Drude approximation with explicit truncation."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    thermal_factor = 1.0 / jnp.tanh(float(decay_rate) / (2.0 * float(temperature)))
    return drude_lorentz_qubit_heom(
        float(coupling_strength) * float(thermal_factor),
        decay_rate,
        initial_density,
        depth=depth,
    )


__all__ = [
    "HEOMHierarchy",
    "HEOMProblem",
    "HEOMSolution",
    "drude_lorentz_qubit_heom",
    "solve_heom",
    "thermal_drude_lorentz_qubit_heom",
]
