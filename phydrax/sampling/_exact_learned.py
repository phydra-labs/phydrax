#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact samplers whose learned objects are frozen proposal artifacts only."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._sampling._addressing import derive_key, SampleAddress
from .._sampling._chain import AbstractChainSampleResult
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    HermitianSpectrum,
    OperatorProperties,
    PreparedFactorization,
)


_DA_MOMENTUM = SampleAddress(
    "markov", "delayed-acceptance-hamiltonian", target="momentum", role="transition"
)
_DA_SURROGATE_ACCEPT = SampleAddress(
    "markov",
    "delayed-acceptance-hamiltonian",
    target="surrogate-acceptance",
    role="transition",
)
_DA_EXACT_ACCEPT = SampleAddress(
    "markov",
    "delayed-acceptance-hamiltonian",
    target="exact-correction",
    role="transition",
)
_FLOW_BASE = SampleAddress(
    "markov", "gauge-equivariant-flow", target="base", role="proposal"
)
_FLOW_ACCEPT = SampleAddress(
    "markov", "gauge-equivariant-flow", target="acceptance", role="transition"
)


LearnedSupportStatus = Literal[
    "qualified",
    "target-mismatch",
    "geometry-mismatch",
    "shape-mismatch",
    "dtype-mismatch",
    "out-of-domain",
]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _positive_shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    shape = tuple(value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{name} must contain positive dimensions.")
    return shape


class LearnedSupportTuple(StrictModule, NonTrainableState):
    """Exact target coordinates and a closed parameter box for one learned artifact."""

    parameter_lower: Array
    parameter_upper: Array
    target_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        target_id: str,
        geometry_id: str,
        configuration_shape: Sequence[int],
        coordinate_dtype: str,
        parameter_names: Sequence[str] = (),
        parameter_lower: ArrayLike = (),
        parameter_upper: ArrayLike = (),
    ):
        target = _identifier(target_id, "target_id")
        geometry = _identifier(geometry_id, "geometry_id")
        shape = _positive_shape(configuration_shape, "configuration_shape")
        dtype = np.dtype(coordinate_dtype)
        if dtype.kind != "f":
            raise TypeError("Learned sampler coordinates must use a real floating dtype.")
        names = tuple(
            _identifier(str(name), "parameter name") for name in parameter_names
        )
        if len(set(names)) != len(names):
            raise ValueError("parameter_names must be unique.")
        lower = np.asarray(parameter_lower, dtype=np.float64)
        upper = np.asarray(parameter_upper, dtype=np.float64)
        if lower.shape != (len(names),) or upper.shape != lower.shape:
            raise ValueError("Parameter bounds must have one entry per parameter name.")
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("Learned-support parameter bounds must be finite.")
        if np.any(lower > upper):
            raise ValueError("Learned-support lower bounds cannot exceed upper bounds.")
        self.parameter_lower = jnp.asarray(lower)
        self.parameter_upper = jnp.asarray(upper)
        self.target_id = target
        self.geometry_id = geometry
        self.configuration_shape = shape
        self.coordinate_dtype = dtype.name
        self.parameter_names = names
        self.support_id = canonical_fingerprint(
            {
                "kind": "learned-artifact-support",
                "target": target,
                "geometry": geometry,
                "configuration_shape": list(shape),
                "coordinate_dtype": dtype.name,
                "parameter_names": list(names),
                "parameter_lower": array_tree_fingerprint(lower),
                "parameter_upper": array_tree_fingerprint(upper),
            }
        )

    @property
    def dimension(self) -> int:
        return prod(self.configuration_shape)


class LearnedSupportQualification(StrictModule, NonTrainableState):
    """Fail-closed evidence for one exact learned-artifact support query."""

    parameter_values: Array
    admitted: Array
    support_id: str = eqx.field(static=True)
    requested_target_id: str = eqx.field(static=True)
    requested_geometry_id: str = eqx.field(static=True)
    requested_shape: tuple[int, ...] = eqx.field(static=True)
    requested_dtype: str = eqx.field(static=True)
    status: LearnedSupportStatus = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)


def assess_learned_support(
    support: LearnedSupportTuple,
    /,
    *,
    target_id: str,
    geometry_id: str,
    configuration_shape: Sequence[int],
    coordinate_dtype: str,
    parameter_values: ArrayLike = (),
) -> LearnedSupportQualification:
    """Assess an exact tuple and parameter point without weakening a mismatch."""
    if not isinstance(support, LearnedSupportTuple):
        raise TypeError("support must be LearnedSupportTuple.")
    target = _identifier(target_id, "target_id")
    geometry = _identifier(geometry_id, "geometry_id")
    shape = _positive_shape(configuration_shape, "configuration_shape")
    dtype = np.dtype(coordinate_dtype).name
    parameters = np.asarray(parameter_values, dtype=np.float64)
    if parameters.shape != support.parameter_lower.shape or np.any(
        ~np.isfinite(parameters)
    ):
        raise ValueError(
            "parameter_values must be finite with the declared support shape."
        )
    target_match = target == support.target_id
    geometry_match = geometry == support.geometry_id
    shape_match = shape == support.configuration_shape
    dtype_match = dtype == support.coordinate_dtype
    in_domain = bool(
        np.all(parameters >= np.asarray(support.parameter_lower))
        and np.all(parameters <= np.asarray(support.parameter_upper))
    )
    status: LearnedSupportStatus = (
        "target-mismatch"
        if not target_match
        else "geometry-mismatch"
        if not geometry_match
        else "shape-mismatch"
        if not shape_match
        else "dtype-mismatch"
        if not dtype_match
        else "out-of-domain"
        if not in_domain
        else "qualified"
    )
    admitted = status == "qualified"
    qualification_id = canonical_fingerprint(
        {
            "kind": "learned-support-qualification",
            "support": support.support_id,
            "requested_target": target,
            "requested_geometry": geometry,
            "requested_shape": list(shape),
            "requested_dtype": dtype,
            "parameter_values": array_tree_fingerprint(parameters),
            "status": status,
        }
    )
    return LearnedSupportQualification(
        parameter_values=jnp.asarray(parameters),
        admitted=jnp.asarray(admitted),
        support_id=support.support_id,
        requested_target_id=target,
        requested_geometry_id=geometry,
        requested_shape=shape,
        requested_dtype=dtype,
        status=status,
        qualification_id=qualification_id,
    )


def require_learned_support(
    support: LearnedSupportTuple,
    /,
    *,
    target_id: str,
    geometry_id: str,
    configuration_shape: Sequence[int],
    coordinate_dtype: str,
    parameter_values: ArrayLike = (),
) -> LearnedSupportQualification:
    """Return qualification evidence or refuse an unqualified artifact domain."""
    qualification = assess_learned_support(
        support,
        target_id=target_id,
        geometry_id=geometry_id,
        configuration_shape=configuration_shape,
        coordinate_dtype=coordinate_dtype,
        parameter_values=parameter_values,
    )
    if qualification.status != "qualified":
        raise ValueError(
            f"Learned artifact refused support query with status {qualification.status!r}."
        )
    return qualification


class FrozenLearnedMetric(StrictModule, NonTrainableState):
    """Content-addressed positive mass matrix frozen before production sampling."""

    mass_matrix: Array
    spectrum: HermitianSpectrum
    factorization: PreparedFactorization
    valid: Array
    support_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)

    def velocity(self, momentum: ArrayLike, /) -> Array:
        value = jnp.asarray(momentum, dtype=self.mass_matrix.dtype)
        if value.shape != (self.mass_matrix.shape[0],):
            raise ValueError(
                "Metric momentum must match the flattened support dimension."
            )
        return jnp.asarray(self.factorization.solve(value).value)

    def kinetic_energy(self, momentum: ArrayLike, /) -> Array:
        value = jnp.asarray(momentum, dtype=self.mass_matrix.dtype)
        return 0.5 * jnp.vdot(value, self.velocity(value)).real


class FrozenLearnedCoarseSpace(StrictModule, NonTrainableState):
    """Content-addressed full-rank coarse basis frozen before production use."""

    basis: Array
    gram_spectrum: HermitianSpectrum
    gram_factorization: PreparedFactorization
    orthogonality_residual: Array
    valid: Array
    support_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.basis.shape[0]

    @property
    def coarse_rank(self) -> int:
        return self.basis.shape[1]

    def restrict(self, fine_coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(fine_coordinates, dtype=self.basis.dtype)
        if value.shape != (self.dimension,):
            raise ValueError("Fine coordinates must match the learned support dimension.")
        right = self.basis.T @ value
        return jnp.asarray(self.gram_factorization.solve(right).value)

    def prolong(self, coarse_coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coarse_coordinates, dtype=self.basis.dtype)
        if value.shape != (self.coarse_rank,):
            raise ValueError("Coarse coordinates must match the learned coarse rank.")
        return self.basis @ value


def freeze_learned_metric(
    mass_matrix: ArrayLike,
    support: LearnedSupportTuple,
    /,
    *,
    maximum_dimension: int = 4096,
) -> FrozenLearnedMetric:
    """Validate and freeze a learned metric without retaining a training object."""
    if not isinstance(support, LearnedSupportTuple):
        raise TypeError("support must be LearnedSupportTuple.")
    maximum = int(maximum_dimension)
    if maximum <= 0 or support.dimension > maximum:
        raise ValueError(
            "Learned metric exceeds maximum_dimension; no matrix was allocated."
        )
    matrix_host = np.asarray(mass_matrix)
    expected = (support.dimension, support.dimension)
    if matrix_host.shape != expected:
        raise ValueError(f"mass_matrix must have shape {expected}.")
    if matrix_host.dtype.name != support.coordinate_dtype:
        raise TypeError("mass_matrix dtype does not match its exact support tuple.")
    if np.iscomplexobj(matrix_host):
        raise TypeError("Learned HMC metrics must be real.")
    symmetric = 0.5 * (matrix_host + matrix_host.T)
    tolerance = (
        128.0
        * np.finfo(matrix_host.dtype).eps
        * max(float(np.max(np.abs(matrix_host), initial=0.0)), 1.0)
    )
    if (
        np.any(~np.isfinite(matrix_host))
        or np.max(np.abs(matrix_host - matrix_host.T)) > tolerance
    ):
        raise ValueError("mass_matrix must be finite and symmetric.")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if eigenvalues[0] <= tolerance:
        raise ValueError("mass_matrix must be numerically positive definite.")
    matrix = jnp.asarray(symmetric)
    spectrum = HermitianSpectrum(matrix)
    factorization = factorize(
        DenseLinearOperator(
            matrix,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"self_adjoint": "verified", "positive_definite": "verified"},
            ),
        ),
        FactorizationPolicy("cholesky"),
    )
    artifact_id = canonical_fingerprint(
        {
            "kind": "frozen-learned-metric",
            "support": support.support_id,
            "mass_matrix": array_tree_fingerprint(matrix_host),
        }
    )
    return FrozenLearnedMetric(
        mass_matrix=matrix,
        spectrum=spectrum,
        factorization=factorization,
        valid=spectrum.valid & jnp.all(jnp.isfinite(matrix)),
        support_id=support.support_id,
        artifact_id=artifact_id,
        frozen=True,
    )


def freeze_learned_coarse_space(
    basis: ArrayLike,
    support: LearnedSupportTuple,
    /,
    *,
    maximum_dimension: int = 4096,
    maximum_coarse_rank: int = 256,
) -> FrozenLearnedCoarseSpace:
    """Validate and freeze a finite full-rank learned coarse subspace."""
    if not isinstance(support, LearnedSupportTuple):
        raise TypeError("support must be LearnedSupportTuple.")
    maximum = int(maximum_dimension)
    maximum_rank = int(maximum_coarse_rank)
    if maximum <= 0 or support.dimension > maximum or maximum_rank <= 0:
        raise ValueError("Coarse-space resource limits reject the support dimension.")
    basis_host = np.asarray(basis)
    if basis_host.ndim != 2 or basis_host.shape[0] != support.dimension:
        raise ValueError("basis must have shape (support.dimension, coarse_rank).")
    rank = basis_host.shape[1]
    if rank <= 0 or rank > maximum_rank:
        raise ValueError("Coarse-space rank exceeds maximum_coarse_rank.")
    if basis_host.dtype.name != support.coordinate_dtype:
        raise TypeError("basis dtype does not match its exact support tuple.")
    if np.iscomplexobj(basis_host) or np.any(~np.isfinite(basis_host)):
        raise ValueError("Learned coarse bases must be finite and real.")
    singular_values = np.linalg.svd(basis_host, compute_uv=False)
    tolerance = (
        128.0
        * np.finfo(basis_host.dtype).eps
        * max(float(singular_values[0]), 1.0)
        * max(basis_host.shape)
    )
    if singular_values[-1] <= tolerance:
        raise ValueError("Learned coarse basis must have full column rank.")
    basis_array = jnp.asarray(basis_host)
    gram = basis_array.T @ basis_array
    spectrum = HermitianSpectrum(gram)
    factorization = factorize(
        DenseLinearOperator(
            gram,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"self_adjoint": "verified", "positive_definite": "verified"},
            ),
        ),
        FactorizationPolicy("cholesky"),
    )
    residual = jnp.max(jnp.abs(gram - jnp.eye(rank, dtype=gram.dtype)))
    artifact_id = canonical_fingerprint(
        {
            "kind": "frozen-learned-coarse-space",
            "support": support.support_id,
            "basis": array_tree_fingerprint(basis_host),
        }
    )
    return FrozenLearnedCoarseSpace(
        basis=basis_array,
        gram_spectrum=spectrum,
        gram_factorization=factorization,
        orthogonality_residual=residual,
        valid=spectrum.valid & jnp.all(jnp.isfinite(basis_array)),
        support_id=support.support_id,
        artifact_id=artifact_id,
        frozen=True,
    )


class DelayedAcceptanceHMCPlan(StrictModule, NonTrainableState):
    """Static trajectory and support contract for exact delayed-acceptance HMC."""

    support: LearnedSupportTuple
    step_size: float = eqx.field(static=True)
    leapfrog_steps: int = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: LearnedSupportTuple,
        /,
        *,
        step_size: float,
        leapfrog_steps: int,
        divergence_threshold: float = 1000.0,
        maximum_dimension: int = 4096,
    ):
        if not isinstance(support, LearnedSupportTuple):
            raise TypeError("support must be LearnedSupportTuple.")
        size = float(step_size)
        steps = int(leapfrog_steps)
        threshold = float(divergence_threshold)
        maximum = int(maximum_dimension)
        if not np.isfinite(size) or size <= 0.0 or steps <= 0:
            raise ValueError("Delayed-acceptance HMC step size/steps are invalid.")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("divergence_threshold must be finite and positive.")
        if maximum <= 0 or support.dimension > maximum:
            raise ValueError("HMC support exceeds maximum_dimension.")
        self.support = support
        self.step_size = size
        self.leapfrog_steps = steps
        self.divergence_threshold = threshold
        self.maximum_dimension = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "delayed-acceptance-hmc-plan",
                "support": support.support_id,
                "step_size": size,
                "leapfrog_steps": steps,
                "divergence_threshold": threshold,
                "maximum_dimension": maximum,
            }
        )


class PreparedDelayedAcceptanceHMC(StrictModule, NonTrainableState):
    """Frozen exact target, surrogate dynamics, and learned proposal metric."""

    exact_log_target: Callable[[Array], Array] = eqx.field(static=True)
    surrogate_log_target: Callable[[Array], Array] = eqx.field(static=True)
    plan: DelayedAcceptanceHMCPlan
    metric: FrozenLearnedMetric
    qualification: LearnedSupportQualification
    target_id: str = eqx.field(static=True)
    surrogate_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)


class DelayedAcceptanceHMCState(StrictModule):
    position: Array
    exact_log_target: Array
    surrogate_log_target: Array
    surrogate_gradient: Array
    step_index: Array
    valid: Array
    kernel_id: str = eqx.field(static=True)


class DelayedAcceptanceHMCResult(AbstractChainSampleResult):
    samples: Array
    exact_log_target: Array
    surrogate_log_target: Array
    accepted: Array
    surrogate_accepted: Array
    surrogate_acceptance_probability: Array
    correction_acceptance_probability: Array
    energy_error: Array
    correction_log_ratio: Array
    divergent: Array
    exact_evaluated: Array
    final_state: DelayedAcceptanceHMCState
    root_key: Array
    target_id: str = eqx.field(static=True)
    surrogate_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    @property
    def num_chains(self) -> int:
        return self.exact_log_target.shape[0]

    @property
    def num_draws(self) -> int:
        return self.exact_log_target.shape[1]

    @property
    def chain_provenance(self) -> str:
        return f"delayed-acceptance-hamiltonian:{self.kernel_id}:{self.target_id}"

    @property
    def acceptance_rate(self) -> Array:
        return jnp.mean(self.accepted.astype("float64"), axis=1)


def prepare_delayed_acceptance_hmc(
    plan: DelayedAcceptanceHMCPlan,
    exact_log_target: Callable[[Array], Array],
    surrogate_log_target: Callable[[Array], Array],
    metric: FrozenLearnedMetric,
    /,
    *,
    target_id: str,
    surrogate_id: str,
    geometry_id: str,
    parameter_values: ArrayLike = (),
) -> PreparedDelayedAcceptanceHMC:
    """Bind immutable learned artifacts to an exact production correction."""
    if not isinstance(plan, DelayedAcceptanceHMCPlan):
        raise TypeError("plan must be DelayedAcceptanceHMCPlan.")
    if not callable(exact_log_target) or not callable(surrogate_log_target):
        raise TypeError("Target and surrogate log densities must be callable.")
    if not isinstance(metric, FrozenLearnedMetric) or not metric.frozen:
        raise TypeError("metric must be a frozen learned metric.")
    target = _identifier(target_id, "target_id")
    surrogate = _identifier(surrogate_id, "surrogate_id")
    if metric.support_id != plan.support.support_id:
        raise ValueError("Learned metric support does not match the HMC plan.")
    qualification = require_learned_support(
        plan.support,
        target_id=target,
        geometry_id=geometry_id,
        configuration_shape=plan.support.configuration_shape,
        coordinate_dtype=metric.mass_matrix.dtype.name,
        parameter_values=parameter_values,
    )
    kernel_id = canonical_fingerprint(
        {
            "kind": "prepared-delayed-acceptance-hmc",
            "plan": plan.plan_id,
            "target": target,
            "surrogate": surrogate,
            "metric": metric.artifact_id,
            "qualification": qualification.qualification_id,
        }
    )
    return PreparedDelayedAcceptanceHMC(
        exact_log_target=exact_log_target,
        surrogate_log_target=surrogate_log_target,
        plan=plan,
        metric=metric,
        qualification=qualification,
        target_id=target,
        surrogate_id=surrogate,
        kernel_id=kernel_id,
    )


def _evaluated_log_density(
    function: Callable[[Array], Array], position: Array, /
) -> Array:
    value = jnp.asarray(function(position))
    if value.shape != () or jnp.iscomplexobj(value):
        raise ValueError("Sampling log densities must return one real scalar.")
    return value


def _surrogate_value_gradient(
    kernel: PreparedDelayedAcceptanceHMC, position: Array, /
) -> tuple[Array, Array]:
    return jax.value_and_grad(
        lambda value: _evaluated_log_density(kernel.surrogate_log_target, value)
    )(position)


def initialize_delayed_acceptance_hmc(
    kernel: PreparedDelayedAcceptanceHMC,
    initial_positions: ArrayLike,
    /,
) -> DelayedAcceptanceHMCState:
    """Evaluate both densities once for a leading batch of persistent chains."""
    if not isinstance(kernel, PreparedDelayedAcceptanceHMC):
        raise TypeError("kernel must be PreparedDelayedAcceptanceHMC.")
    positions = jnp.asarray(initial_positions, dtype=kernel.metric.mass_matrix.dtype)
    shape = kernel.plan.support.configuration_shape
    if (
        positions.ndim != len(shape) + 1
        or positions.shape[1:] != shape
        or positions.shape[0] < 1
    ):
        raise ValueError(
            "initial_positions must have a leading chain axis and the supported shape."
        )
    exact = jax.vmap(
        lambda value: _evaluated_log_density(kernel.exact_log_target, value)
    )(positions)
    surrogate, gradients = jax.vmap(
        lambda value: _surrogate_value_gradient(kernel, value)
    )(positions)
    valid = (
        kernel.metric.valid
        & kernel.qualification.admitted
        & jnp.all(jnp.isfinite(positions).reshape((positions.shape[0], -1)), axis=1)
        & jnp.isfinite(exact)
        & jnp.isfinite(surrogate)
        & jnp.all(jnp.isfinite(gradients).reshape((positions.shape[0], -1)), axis=1)
    )
    exact = eqx.error_if(
        exact, ~jnp.all(valid), "Initial delayed-acceptance HMC states must be finite."
    )
    return DelayedAcceptanceHMCState(
        position=positions,
        exact_log_target=exact,
        surrogate_log_target=surrogate,
        surrogate_gradient=gradients,
        step_index=jnp.asarray(0, dtype=jnp.uint32),
        valid=valid,
        kernel_id=kernel.kernel_id,
    )


def _metric_solve(kernel: PreparedDelayedAcceptanceHMC, vector: Array, /) -> Array:
    return kernel.metric.velocity(vector)


def _metric_kinetic(kernel: PreparedDelayedAcceptanceHMC, momentum: Array, /) -> Array:
    return kernel.metric.kinetic_energy(momentum.reshape((-1,)))


def _sample_metric_momentum(
    kernel: PreparedDelayedAcceptanceHMC, key: Key[Array, ""], /
) -> Array:
    normal = jr.normal(
        key,
        (kernel.plan.support.dimension,),
        dtype=kernel.metric.mass_matrix.dtype,
    )
    flat = kernel.metric.spectrum.eigenvectors @ (
        jnp.sqrt(kernel.metric.spectrum.eigenvalues) * normal
    )
    return flat.reshape(kernel.plan.support.configuration_shape)


def _surrogate_trajectory(
    kernel: PreparedDelayedAcceptanceHMC,
    position: Array,
    gradient: Array,
    momentum: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    shape = kernel.plan.support.configuration_shape

    def step(carry, _):
        q, p, g, active, nonfinite = carry
        half = p + 0.5 * kernel.plan.step_size * g
        velocity = _metric_solve(kernel, half.reshape((-1,))).reshape(shape)
        candidate_q = q + kernel.plan.step_size * velocity
        candidate_value, candidate_gradient = _surrogate_value_gradient(
            kernel, candidate_q
        )
        candidate_p = half + 0.5 * kernel.plan.step_size * candidate_gradient
        finite = (
            jnp.all(jnp.isfinite(candidate_q))
            & jnp.isfinite(candidate_value)
            & jnp.all(jnp.isfinite(candidate_gradient))
            & jnp.all(jnp.isfinite(candidate_p))
        )
        commit = active & finite
        return (
            jnp.where(commit, candidate_q, q),
            jnp.where(commit, candidate_p, p),
            jnp.where(commit, candidate_gradient, g),
            commit,
            nonfinite | (active & ~finite),
        ), None

    initial = (position, momentum, gradient, jnp.asarray(True), jnp.asarray(False))
    (q, p, g, _, nonfinite), _ = jax.lax.scan(
        step, initial, None, length=kernel.plan.leapfrog_steps
    )
    return (
        q,
        -p,
        g,
        _evaluated_log_density(kernel.surrogate_log_target, q),
        nonfinite,
        jnp.asarray(kernel.plan.leapfrog_steps, dtype=jnp.int32),
    )


def _delayed_acceptance_transition(
    kernel: PreparedDelayedAcceptanceHMC,
    position: Array,
    exact_value: Array,
    surrogate_value: Array,
    surrogate_gradient: Array,
    state_valid: Array,
    root_key: Key[Array, ""],
    chain_index: Array,
    step_index: Array,
    /,
):
    momentum_key = derive_key(root_key, _DA_MOMENTUM, chain_index, step_index)
    stage_one_key = derive_key(root_key, _DA_SURROGATE_ACCEPT, chain_index, step_index)
    stage_two_key = derive_key(root_key, _DA_EXACT_ACCEPT, chain_index, step_index)
    momentum = _sample_metric_momentum(kernel, momentum_key)
    proposed_q, proposed_p, proposed_gradient, proposed_surrogate, nonfinite, used = (
        _surrogate_trajectory(kernel, position, surrogate_gradient, momentum)
    )
    energy_error = (-proposed_surrogate + _metric_kinetic(kernel, proposed_p)) - (
        -surrogate_value + _metric_kinetic(kernel, momentum)
    )
    divergent = (
        nonfinite
        | ~jnp.isfinite(energy_error)
        | (jnp.abs(energy_error) > kernel.plan.divergence_threshold)
    )
    stage_one_log = jnp.minimum(-energy_error, 0.0)
    surrogate_accepted = (
        state_valid & ~divergent & (jnp.log(jr.uniform(stage_one_key)) < stage_one_log)
    )
    evaluate_exact = surrogate_accepted & state_valid & ~divergent
    proposed_exact = jax.lax.cond(
        evaluate_exact,
        lambda candidate: _evaluated_log_density(kernel.exact_log_target, candidate),
        lambda _: exact_value,
        proposed_q,
    )
    correction = proposed_exact - proposed_surrogate - exact_value + surrogate_value
    exact_finite = jnp.isfinite(proposed_exact) & jnp.isfinite(correction)
    stage_two_log = jnp.minimum(correction, 0.0)
    accepted = (
        evaluate_exact
        & exact_finite
        & (jnp.log(jr.uniform(stage_two_key)) < stage_two_log)
    )
    return (
        jnp.where(accepted, proposed_q, position),
        jnp.where(accepted, proposed_exact, exact_value),
        jnp.where(accepted, proposed_surrogate, surrogate_value),
        jnp.where(accepted, proposed_gradient, surrogate_gradient),
        accepted,
        surrogate_accepted,
        jnp.where(state_valid & ~divergent, jnp.exp(stage_one_log), 0.0),
        jnp.where(evaluate_exact & exact_finite, jnp.exp(stage_two_log), 0.0),
        energy_error,
        jnp.where(evaluate_exact, correction, 0.0),
        divergent | (evaluate_exact & ~exact_finite),
        evaluate_exact,
        used,
    )


def sample_delayed_acceptance_hmc(
    kernel: PreparedDelayedAcceptanceHMC,
    state: DelayedAcceptanceHMCState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
) -> DelayedAcceptanceHMCResult:
    """Advance exact chains; the surrogate affects efficiency, never target density."""
    if not isinstance(kernel, PreparedDelayedAcceptanceHMC) or not isinstance(
        state, DelayedAcceptanceHMCState
    ):
        raise TypeError("kernel/state types are invalid.")
    if state.kernel_id != kernel.kernel_id:
        raise ValueError(
            "Delayed-acceptance HMC state belongs to another prepared kernel."
        )
    draws = int(num_draws)
    if draws <= 0:
        raise ValueError("num_draws must be positive.")
    chain_indices = jnp.arange(state.position.shape[0], dtype=jnp.uint32)

    def draw(carry, _):
        positions, exact, surrogate, gradients, valid, index = carry
        transition = jax.lax.map(
            lambda inputs: _delayed_acceptance_transition(
                kernel,
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                key,
                inputs[5],
                index,
            ),
            (positions, exact, surrogate, gradients, valid, chain_indices),
        )
        next_positions, next_exact, next_surrogate, next_gradients = transition[:4]
        return (
            next_positions,
            next_exact,
            next_surrogate,
            next_gradients,
            valid,
            index + 1,
        ), (next_positions, next_exact, next_surrogate) + transition[4:]

    (positions, exact, surrogate, gradients, valid, index), outputs = jax.lax.scan(
        draw,
        (
            state.position,
            state.exact_log_target,
            state.surrogate_log_target,
            state.surrogate_gradient,
            state.valid,
            state.step_index,
        ),
        None,
        length=draws,
    )
    (
        samples,
        exact_history,
        surrogate_history,
        accepted,
        surrogate_accepted,
        surrogate_probability,
        correction_probability,
        energy_error,
        correction,
        divergent,
        exact_evaluated,
        _,
    ) = (jnp.swapaxes(value, 0, 1) for value in outputs)
    final_valid = (
        valid
        & kernel.metric.valid
        & kernel.qualification.admitted
        & jnp.all(jnp.isfinite(positions).reshape((positions.shape[0], -1)), axis=1)
        & jnp.isfinite(exact)
        & jnp.isfinite(surrogate)
        & jnp.all(jnp.isfinite(gradients).reshape((gradients.shape[0], -1)), axis=1)
    )
    final = DelayedAcceptanceHMCState(
        position=positions,
        exact_log_target=exact,
        surrogate_log_target=surrogate,
        surrogate_gradient=gradients,
        step_index=index,
        valid=final_valid,
        kernel_id=kernel.kernel_id,
    )
    return DelayedAcceptanceHMCResult(
        samples=samples,
        exact_log_target=exact_history,
        surrogate_log_target=surrogate_history,
        accepted=accepted,
        surrogate_accepted=surrogate_accepted,
        surrogate_acceptance_probability=surrogate_probability,
        correction_acceptance_probability=correction_probability,
        energy_error=energy_error,
        correction_log_ratio=correction,
        divergent=divergent,
        exact_evaluated=exact_evaluated,
        final_state=final,
        root_key=jnp.asarray(key),
        target_id=kernel.target_id,
        surrogate_id=kernel.surrogate_id,
        kernel_id=kernel.kernel_id,
        claim="exact-target-delayed-acceptance-with-frozen-surrogate",
    )


class GaugeFlowEvaluation(StrictModule):
    """One direction of an invertible gauge-equivariant proposal map."""

    value: Array
    log_abs_det_jacobian: Array
    equivariance_residual: Array
    valid: Array


class AbstractGaugeEquivariantFlow(StrictModule, NonTrainableState):
    """Invertible flow protocol with explicit Jacobian and equivariance evidence."""

    flow_id: eqx.AbstractVar[str]
    support_id: eqx.AbstractVar[str]
    configuration_shape: eqx.AbstractVar[tuple[int, ...]]

    @abc.abstractmethod
    def forward(self, base_value: Array, /) -> GaugeFlowEvaluation:
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self, position: Array, /) -> GaugeFlowEvaluation:
        raise NotImplementedError


class ScalarGaugeEquivariantFlow(AbstractGaugeEquivariantFlow):
    """Nontrivial scalar map commuting with every linear gauge representation."""

    scale: Array
    log_abs_det: Array
    flow_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        support: LearnedSupportTuple,
        /,
        *,
        scale: float,
    ):
        if not isinstance(support, LearnedSupportTuple):
            raise TypeError("support must be LearnedSupportTuple.")
        scale_ = float(scale)
        if not np.isfinite(scale_) or scale_ <= 0.0:
            raise ValueError("Gauge-equivariant scalar flow scale must be positive.")
        self.scale = jnp.asarray(scale_, dtype=np.dtype(support.coordinate_dtype))
        self.log_abs_det = support.dimension * jnp.log(self.scale)
        self.support_id = support.support_id
        self.configuration_shape = support.configuration_shape
        self.flow_id = canonical_fingerprint(
            {
                "kind": "scalar-gauge-equivariant-flow",
                "support": support.support_id,
                "scale": scale_,
            }
        )

    def forward(self, base_value: Array, /) -> GaugeFlowEvaluation:
        value = jnp.asarray(base_value)
        if value.shape != self.configuration_shape:
            raise ValueError("Flow base value does not match configuration_shape.")
        mapped = self.scale * value
        finite = jnp.all(jnp.isfinite(mapped))
        return GaugeFlowEvaluation(
            value=mapped,
            log_abs_det_jacobian=self.log_abs_det,
            equivariance_residual=jnp.asarray(0.0, dtype=self.scale.dtype),
            valid=finite,
        )

    def inverse(self, position: Array, /) -> GaugeFlowEvaluation:
        value = jnp.asarray(position)
        if value.shape != self.configuration_shape:
            raise ValueError("Flow position does not match configuration_shape.")
        mapped = value / self.scale
        finite = jnp.all(jnp.isfinite(mapped))
        return GaugeFlowEvaluation(
            value=mapped,
            log_abs_det_jacobian=-self.log_abs_det,
            equivariance_residual=jnp.asarray(0.0, dtype=self.scale.dtype),
            valid=finite,
        )


class GaugeFlowProposalPlan(StrictModule, NonTrainableState):
    """Static exact-support and equivariance gate for independence proposals."""

    support: LearnedSupportTuple
    maximum_dimension: int = eqx.field(static=True)
    equivariance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: LearnedSupportTuple,
        /,
        *,
        maximum_dimension: int = 4096,
        equivariance_tolerance: float = 1e-6,
    ):
        if not isinstance(support, LearnedSupportTuple):
            raise TypeError("support must be LearnedSupportTuple.")
        maximum = int(maximum_dimension)
        tolerance = float(equivariance_tolerance)
        if maximum <= 0 or support.dimension > maximum:
            raise ValueError("Gauge flow support exceeds maximum_dimension.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("equivariance_tolerance must be finite and non-negative.")
        self.support = support
        self.maximum_dimension = maximum
        self.equivariance_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gauge-flow-proposal-plan",
                "support": support.support_id,
                "maximum_dimension": maximum,
                "equivariance_tolerance": tolerance,
            }
        )


class PreparedGaugeFlowProposal(StrictModule, NonTrainableState):
    target_log_density: Callable[[Array], Array] = eqx.field(static=True)
    base_sample: Callable[[Key[Array, ""]], Array] = eqx.field(static=True)
    base_log_density: Callable[[Array], Array] = eqx.field(static=True)
    flow: AbstractGaugeEquivariantFlow
    plan: GaugeFlowProposalPlan
    qualification: LearnedSupportQualification
    target_id: str = eqx.field(static=True)
    base_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)


class GaugeFlowChainState(StrictModule):
    position: Array
    log_target: Array
    log_proposal: Array
    step_index: Array
    valid: Array
    proposal_id: str = eqx.field(static=True)


class GaugeFlowProposalResult(AbstractChainSampleResult):
    samples: Array
    proposed_samples: Array
    log_target: Array
    log_proposal: Array
    accepted: Array
    acceptance_probability: Array
    log_acceptance_ratio: Array
    forward_log_abs_det_jacobian: Array
    inverse_log_abs_det_jacobian: Array
    equivariance_residual: Array
    proposal_valid: Array
    final_state: GaugeFlowChainState
    root_key: Array
    target_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    @property
    def num_chains(self) -> int:
        return self.log_target.shape[0]

    @property
    def num_draws(self) -> int:
        return self.log_target.shape[1]

    @property
    def chain_provenance(self) -> str:
        return f"gauge-equivariant-flow-mh:{self.proposal_id}:{self.target_id}"


def _flow_log_density_from_inverse(
    proposal: PreparedGaugeFlowProposal, position: Array, /
) -> tuple[Array, GaugeFlowEvaluation]:
    inverse = proposal.flow.inverse(position)
    if not isinstance(inverse, GaugeFlowEvaluation):
        raise TypeError("flow.inverse must return GaugeFlowEvaluation.")
    if inverse.value.shape != proposal.plan.support.configuration_shape:
        raise ValueError("flow.inverse must return the supported configuration shape.")
    if inverse.value.dtype.name != proposal.plan.support.coordinate_dtype:
        raise TypeError("flow.inverse dtype does not match the proposal support tuple.")
    base_log = jnp.asarray(proposal.base_log_density(inverse.value))
    if base_log.shape != () or jnp.iscomplexobj(base_log):
        raise ValueError("base_log_density must return one real scalar.")
    return base_log + inverse.log_abs_det_jacobian, inverse


def prepare_gauge_flow_proposal(
    plan: GaugeFlowProposalPlan,
    flow: AbstractGaugeEquivariantFlow,
    target_log_density: Callable[[Array], Array],
    base_sample: Callable[[Key[Array, ""]], Array],
    base_log_density: Callable[[Array], Array],
    /,
    *,
    target_id: str,
    geometry_id: str,
    base_id: str,
    parameter_values: ArrayLike = (),
) -> PreparedGaugeFlowProposal:
    """Bind an invertible flow to exact independence-MH correction."""
    if not isinstance(plan, GaugeFlowProposalPlan):
        raise TypeError("plan must be GaugeFlowProposalPlan.")
    if not isinstance(flow, AbstractGaugeEquivariantFlow):
        raise TypeError("flow must implement AbstractGaugeEquivariantFlow.")
    if not all(
        callable(value) for value in (target_log_density, base_sample, base_log_density)
    ):
        raise TypeError("Target/base proposal functions must be callable.")
    target = _identifier(target_id, "target_id")
    base = _identifier(base_id, "base_id")
    if (
        flow.support_id != plan.support.support_id
        or flow.configuration_shape != plan.support.configuration_shape
    ):
        raise ValueError("Gauge flow does not match the exact proposal support tuple.")
    qualification = require_learned_support(
        plan.support,
        target_id=target,
        geometry_id=geometry_id,
        configuration_shape=plan.support.configuration_shape,
        coordinate_dtype=plan.support.coordinate_dtype,
        parameter_values=parameter_values,
    )
    proposal_id = canonical_fingerprint(
        {
            "kind": "prepared-gauge-flow-proposal",
            "plan": plan.plan_id,
            "flow": flow.flow_id,
            "target": target,
            "base": base,
            "qualification": qualification.qualification_id,
        }
    )
    return PreparedGaugeFlowProposal(
        target_log_density=target_log_density,
        base_sample=base_sample,
        base_log_density=base_log_density,
        flow=flow,
        plan=plan,
        qualification=qualification,
        target_id=target,
        base_id=base,
        proposal_id=proposal_id,
    )


def initialize_gauge_flow_chain(
    proposal: PreparedGaugeFlowProposal,
    initial_positions: ArrayLike,
    /,
) -> GaugeFlowChainState:
    if not isinstance(proposal, PreparedGaugeFlowProposal):
        raise TypeError("proposal must be PreparedGaugeFlowProposal.")
    positions = jnp.asarray(initial_positions)
    shape = proposal.plan.support.configuration_shape
    if (
        positions.ndim != len(shape) + 1
        or positions.shape[1:] != shape
        or positions.shape[0] < 1
    ):
        raise ValueError(
            "initial_positions must have a leading chain axis and supported shape."
        )
    if positions.dtype.name != proposal.plan.support.coordinate_dtype:
        raise TypeError(
            "initial_positions dtype does not match the proposal support tuple."
        )
    target = jax.vmap(
        lambda value: _evaluated_log_density(proposal.target_log_density, value)
    )(positions)
    proposal_values, inverse = jax.vmap(
        lambda value: _flow_log_density_from_inverse(proposal, value)
    )(positions)
    roundtrip = jax.vmap(proposal.flow.forward)(inverse.value)
    roundtrip_position_residual = jnp.max(
        jnp.abs(roundtrip.value - positions).reshape((positions.shape[0], -1)),
        axis=1,
    )
    roundtrip_jacobian_residual = jnp.abs(
        roundtrip.log_abs_det_jacobian + inverse.log_abs_det_jacobian
    )
    roundtrip_residual = jnp.maximum(
        jnp.maximum(inverse.equivariance_residual, roundtrip.equivariance_residual),
        jnp.maximum(roundtrip_position_residual, roundtrip_jacobian_residual),
    )
    valid = (
        proposal.qualification.admitted
        & inverse.valid
        & roundtrip.valid
        & jnp.isfinite(roundtrip_residual)
        & (roundtrip_residual <= proposal.plan.equivariance_tolerance)
        & jnp.all(jnp.isfinite(positions).reshape((positions.shape[0], -1)), axis=1)
        & jnp.isfinite(target)
        & jnp.isfinite(proposal_values)
    )
    target = eqx.error_if(
        target, ~jnp.all(valid), "Initial gauge-flow states are outside proposal support."
    )
    return GaugeFlowChainState(
        position=positions,
        log_target=target,
        log_proposal=proposal_values,
        step_index=jnp.asarray(0, dtype=jnp.uint32),
        valid=valid,
        proposal_id=proposal.proposal_id,
    )


def _gauge_flow_transition(
    proposal: PreparedGaugeFlowProposal,
    position: Array,
    log_target: Array,
    log_proposal: Array,
    state_valid: Array,
    root_key: Key[Array, ""],
    chain_index: Array,
    step_index: Array,
    /,
):
    base_key = derive_key(root_key, _FLOW_BASE, chain_index, step_index)
    accept_key = derive_key(root_key, _FLOW_ACCEPT, chain_index, step_index)
    base = jnp.asarray(proposal.base_sample(base_key))
    if base.shape != proposal.plan.support.configuration_shape:
        raise ValueError("base_sample must return the supported configuration shape.")
    if base.dtype.name != proposal.plan.support.coordinate_dtype:
        raise TypeError("base_sample dtype does not match the proposal support tuple.")
    forward = proposal.flow.forward(base)
    if not isinstance(forward, GaugeFlowEvaluation):
        raise TypeError("flow.forward must return GaugeFlowEvaluation.")
    if forward.value.shape != proposal.plan.support.configuration_shape:
        raise ValueError("flow.forward must return the supported configuration shape.")
    if forward.value.dtype.name != proposal.plan.support.coordinate_dtype:
        raise TypeError("flow.forward dtype does not match the proposal support tuple.")
    proposed_target = _evaluated_log_density(proposal.target_log_density, forward.value)
    base_log = jnp.asarray(proposal.base_log_density(base))
    if base_log.shape != () or jnp.iscomplexobj(base_log):
        raise ValueError("base_log_density must return one real scalar.")
    proposed_log_proposal = base_log - forward.log_abs_det_jacobian
    inverse_log, inverse = _flow_log_density_from_inverse(proposal, forward.value)
    density_consistency = jnp.abs(proposed_log_proposal - inverse_log)
    inverse_consistency = jnp.max(jnp.abs(inverse.value - base))
    jacobian_consistency = jnp.abs(
        forward.log_abs_det_jacobian + inverse.log_abs_det_jacobian
    )
    equivariance_residual = jnp.maximum(
        jnp.maximum(forward.equivariance_residual, inverse.equivariance_residual),
        jnp.maximum(
            density_consistency,
            jnp.maximum(inverse_consistency, jacobian_consistency),
        ),
    )
    finite = (
        jnp.all(jnp.isfinite(base))
        & jnp.all(jnp.isfinite(forward.value))
        & jnp.isfinite(proposed_target)
        & jnp.isfinite(proposed_log_proposal)
        & jnp.isfinite(forward.log_abs_det_jacobian)
        & jnp.isfinite(inverse.log_abs_det_jacobian)
        & jnp.isfinite(equivariance_residual)
        & jnp.isfinite(jacobian_consistency)
    )
    valid = (
        state_valid
        & forward.valid
        & inverse.valid
        & finite
        & (equivariance_residual <= proposal.plan.equivariance_tolerance)
        & (jacobian_consistency <= proposal.plan.equivariance_tolerance)
    )
    log_ratio = proposed_target + log_proposal - log_target - proposed_log_proposal
    log_accept = jnp.minimum(log_ratio, 0.0)
    accepted = valid & (jnp.log(jr.uniform(accept_key)) < log_accept)
    return (
        jnp.where(accepted, forward.value, position),
        jnp.where(accepted, proposed_target, log_target),
        jnp.where(accepted, proposed_log_proposal, log_proposal),
        accepted,
        jnp.where(valid, jnp.exp(log_accept), 0.0),
        jnp.where(valid, log_ratio, -jnp.inf),
        forward.log_abs_det_jacobian,
        inverse.log_abs_det_jacobian,
        equivariance_residual,
        valid,
        forward.value,
    )


def sample_gauge_flow_proposal(
    proposal: PreparedGaugeFlowProposal,
    state: GaugeFlowChainState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
) -> GaugeFlowProposalResult:
    """Advance exact-target chains with an independence flow and full MH ratio."""
    if not isinstance(proposal, PreparedGaugeFlowProposal) or not isinstance(
        state, GaugeFlowChainState
    ):
        raise TypeError("proposal/state types are invalid.")
    if state.proposal_id != proposal.proposal_id:
        raise ValueError("Gauge-flow state belongs to another prepared proposal.")
    draws = int(num_draws)
    if draws <= 0:
        raise ValueError("num_draws must be positive.")
    chain_indices = jnp.arange(state.position.shape[0], dtype=jnp.uint32)

    def draw(carry, _):
        positions, targets, proposals, valid, index = carry
        transition = jax.vmap(
            lambda q, target, proposal_log, state_valid, chain: _gauge_flow_transition(
                proposal, q, target, proposal_log, state_valid, key, chain, index
            )
        )(positions, targets, proposals, valid, chain_indices)
        return (
            transition[0],
            transition[1],
            transition[2],
            valid,
            index + 1,
        ), transition

    (positions, targets, proposals, valid, index), outputs = jax.lax.scan(
        draw,
        (
            state.position,
            state.log_target,
            state.log_proposal,
            state.valid,
            state.step_index,
        ),
        None,
        length=draws,
    )
    (
        samples,
        target_history,
        proposal_history,
        accepted,
        probability,
        log_ratio,
        forward_logdet,
        inverse_logdet,
        residual,
        proposal_valid,
        proposed_samples,
    ) = (jnp.swapaxes(value, 0, 1) for value in outputs)
    final = GaugeFlowChainState(
        position=positions,
        log_target=targets,
        log_proposal=proposals,
        step_index=index,
        valid=(
            valid
            & proposal.qualification.admitted
            & jnp.all(jnp.isfinite(positions).reshape((positions.shape[0], -1)), axis=1)
            & jnp.isfinite(targets)
            & jnp.isfinite(proposals)
        ),
        proposal_id=proposal.proposal_id,
    )
    return GaugeFlowProposalResult(
        samples=samples,
        proposed_samples=proposed_samples,
        log_target=target_history,
        log_proposal=proposal_history,
        accepted=accepted,
        acceptance_probability=probability,
        log_acceptance_ratio=log_ratio,
        forward_log_abs_det_jacobian=forward_logdet,
        inverse_log_abs_det_jacobian=inverse_logdet,
        equivariance_residual=residual,
        proposal_valid=proposal_valid,
        final_state=final,
        root_key=jnp.asarray(key),
        target_id=proposal.target_id,
        proposal_id=proposal.proposal_id,
        claim="gauge-equivariant-flow-with-exact-jacobian-mh-correction",
    )


__all__ = [
    "AbstractGaugeEquivariantFlow",
    "DelayedAcceptanceHMCPlan",
    "DelayedAcceptanceHMCResult",
    "DelayedAcceptanceHMCState",
    "FrozenLearnedCoarseSpace",
    "FrozenLearnedMetric",
    "GaugeFlowChainState",
    "GaugeFlowEvaluation",
    "GaugeFlowProposalPlan",
    "GaugeFlowProposalResult",
    "LearnedSupportQualification",
    "LearnedSupportStatus",
    "LearnedSupportTuple",
    "PreparedDelayedAcceptanceHMC",
    "PreparedGaugeFlowProposal",
    "ScalarGaugeEquivariantFlow",
    "assess_learned_support",
    "freeze_learned_coarse_space",
    "freeze_learned_metric",
    "initialize_delayed_acceptance_hmc",
    "initialize_gauge_flow_chain",
    "prepare_delayed_acceptance_hmc",
    "prepare_gauge_flow_proposal",
    "require_learned_support",
    "sample_delayed_acceptance_hmc",
    "sample_gauge_flow_proposal",
]
