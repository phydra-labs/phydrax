# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .._sampling._addressing import derive_key, SampleAddress
from ..typing import parse, PRNGKey
from ._operators import DenseLinearOperator
from ._randomized import random_probes
from ._spaces import _coordinate_dtype
from ._svd_contracts import (
    RandomizedSVD,
    RandomizedSVDState,
    SVDProblem,
    SVDSolvePlan,
    SVDSolveStatus,
)
from ._svd_dense import diagonal_scales
from ._svd_evidence import (
    gaussian_range_bound,
    resident_range_residual,
    scaled_column_norms,
    version_failure_probability,
)


def forward(problem: SVDProblem, source: Array, target: Array, block: Array, /) -> Array:
    return target[:, None] * problem.operator.mv_block(block / source[:, None])


def backward(problem: SVDProblem, source: Array, target: Array, block: Array, /) -> Array:
    return source[:, None] * problem.operator.adjoint_mv_block(block / target[:, None])


def finite_qr(block: Array, /) -> tuple[Array, Array, Array]:
    finite = jnp.all(jnp.isfinite(block))
    width = block.shape[1]

    def factor(value: Array) -> tuple[Array, Array, Array]:
        basis, triangular = jnp.linalg.qr(value, mode="reduced")
        singular = jnp.linalg.svd(jax.lax.stop_gradient(triangular), compute_uv=False)
        scale = jnp.maximum(singular[0], jnp.finfo(singular.dtype).tiny)
        margin = singular[-1] / scale
        full_rank = (
            singular[-1] > 64 * max(value.shape) * jnp.finfo(singular.dtype).eps * scale
        )
        return basis, margin, full_rank

    def unavailable(value: Array) -> tuple[Array, Array, Array]:
        return (
            jnp.full((value.shape[0], width), jnp.nan, value.dtype),
            jnp.asarray(jnp.nan, value.real.dtype),
            jnp.asarray(False),
        )

    return jax.lax.cond(finite, factor, unavailable, block)


def prepare_randomized(
    problem: SVDProblem, plan: SVDSolvePlan, key: PRNGKey, version: Array, /
) -> RandomizedSVDState:
    root = parse(key, PRNGKey, "key")
    if root.shape != ():
        raise ValueError("Randomized SVD requires a scalar typed key.")
    method = plan.policy.method
    if not isinstance(method, RandomizedSVD):
        raise TypeError("Randomized worker requires RandomizedSVD.")
    source, source_valid = diagonal_scales(problem.operator.source)
    target, target_valid = diagonal_scales(problem.operator.target)
    rows, columns = problem.operator.target.size, problem.operator.source.size
    width = plan.sketch_size
    dtype = _coordinate_dtype(problem.operator.source)
    sketch_version = (
        version if method.probe_refresh == "redraw" else jnp.zeros_like(version)
    )
    sketch_address = SampleAddress(
        "phydrax.linalg.svd",
        "range",
        target=(plan.plan_id, problem.problem_id),
        role="construction-sketch",
    )
    audit_address = SampleAddress(
        "phydrax.linalg.svd",
        "range",
        target=(plan.plan_id, problem.problem_id),
        role="independent-audit",
    )
    sketch_key = derive_key(root, sketch_address, sketch_version)
    audit_key = derive_key(root, audit_address, version)
    resident = isinstance(problem.operator, DenseLinearOperator)
    initial_valid = source_valid & target_valid
    if resident:
        initial_valid = initial_valid & jnp.all(jnp.isfinite(problem.operator.matrix))

    def construct(_: Array) -> tuple[Array, Array, Array, Array]:
        probes = jax.lax.stop_gradient(
            random_probes(sketch_key, columns, width, jnp.dtype(dtype))
        )
        basis, margin, full_rank = finite_qr(forward(problem, source, target, probes))

        def refine(
            carry: tuple[Array, Array, Array], unused: None
        ) -> tuple[tuple[Array, Array, Array], None]:
            del unused

            def step(payload: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
                q, minimum_margin, rank_ok = payload
                z, backward_margin, backward_rank = finite_qr(
                    backward(problem, source, target, q)
                )

                def next_forward(block: Array) -> tuple[Array, Array, Array]:
                    return finite_qr(forward(problem, source, target, block))

                def unavailable_forward(block: Array) -> tuple[Array, Array, Array]:
                    return (
                        jnp.full((rows, width), jnp.nan, block.dtype),
                        jnp.asarray(jnp.nan, block.real.dtype),
                        jnp.asarray(False),
                    )

                q, forward_margin, forward_rank = jax.lax.cond(
                    jnp.all(jnp.isfinite(z)), next_forward, unavailable_forward, z
                )
                return (
                    q,
                    jnp.minimum(
                        minimum_margin, jnp.minimum(backward_margin, forward_margin)
                    ),
                    rank_ok & backward_rank & forward_rank,
                )

            result = jax.lax.cond(
                jnp.all(jnp.isfinite(carry[0])), step, lambda payload: payload, carry
            )
            return result, None

        (basis, margin, full_rank), _trajectory = jax.lax.scan(
            refine, (basis, margin, full_rank), None, length=method.power_iterations
        )
        compressed = jax.lax.cond(
            jnp.all(jnp.isfinite(basis)),
            lambda q: backward(problem, source, target, q).conj().T,
            lambda q: jnp.full((width, columns), jnp.nan, q.dtype),
            basis,
        )
        return basis, compressed, margin, full_rank

    def unavailable(_: Array) -> tuple[Array, Array, Array, Array]:
        return (
            jnp.full((rows, width), jnp.nan, dtype),
            jnp.full((width, columns), jnp.nan, dtype),
            jnp.asarray(jnp.nan, source.real.dtype),
            jnp.asarray(False),
        )

    basis, compressed, margin, rank_ok = jax.lax.cond(
        initial_valid, construct, unavailable, jnp.asarray(0)
    )
    valid = (
        initial_valid & jnp.all(jnp.isfinite(basis)) & jnp.all(jnp.isfinite(compressed))
    )
    real_dtype = source.real.dtype
    probability = (
        jnp.asarray(0, real_dtype)
        if resident
        else version_failure_probability(
            plan.policy.approximation.failure_probability, version, real_dtype
        )
    )

    def measure(_: Array) -> tuple[Array, Array, Array, Array]:
        if isinstance(problem.operator, DenseLinearOperator):
            residual, energy = resident_range_residual(
                problem.operator, jax.lax.stop_gradient(basis), source, target
            )
            allowance = (
                plan.policy.approximation.numerical_allowance
                * jnp.finfo(real_dtype).eps
                * max(rows, columns)
                * jnp.sqrt(energy)
            )
            return (
                residual + allowance,
                allowance,
                jnp.asarray(jnp.nan, real_dtype),
                energy,
            )
        probes = random_probes(
            audit_key, columns, plan.policy.approximation.audit_probes, jnp.dtype(dtype)
        )
        images = forward(problem, source, target, probes)
        residual = images - basis @ (basis.conj().T @ images)
        maximum = jnp.max(scaled_column_norms(residual))
        allowance = (
            plan.policy.approximation.numerical_allowance
            * jnp.finfo(real_dtype).eps
            * max(rows, columns)
            * jnp.max(scaled_column_norms(images))
        )
        eta = gaussian_range_bound(
            maximum + allowance,
            probability,
            plan.policy.approximation.audit_probes,
            jnp.issubdtype(dtype, jnp.complexfloating),
        )
        return eta, allowance, maximum, jnp.asarray(jnp.nan, real_dtype)

    def no_measure(_: Array) -> tuple[Array, Array, Array, Array]:
        nan = jnp.asarray(jnp.nan, real_dtype)
        return nan, nan, nan, nan

    eta, allowance, maximum, energy = jax.lax.cond(
        valid, measure, no_measure, jnp.asarray(0)
    )
    valid = valid & jnp.isfinite(eta)
    status = jnp.where(
        valid, int(SVDSolveStatus.SUCCESS), int(SVDSolveStatus.PREPARATION_FAILED)
    ).astype(jnp.int32)
    return RandomizedSVDState(
        basis,
        compressed,
        source,
        target,
        status,
        margin,
        rank_ok,
        jax.lax.stop_gradient(eta),
        jax.lax.stop_gradient(allowance),
        probability,
        jax.lax.stop_gradient(maximum),
        jax.lax.stop_gradient(energy) if resident else None,
        root,
        sketch_version,
        version,
    )
