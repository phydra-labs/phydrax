#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.fem import FiniteElementHPTransferPlan
from ..linalg import ArraySpace, DenseLinearOperator, LinearSystem, solve


class HPNewtonKrylovResult(StrictModule):
    value: Array
    residual_norm: Array
    iterations: Array
    converged: Array


class HPNewtonKrylovBuilder(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)

    def __init__(
        self,
        maximum_iterations: int = 20,
        tolerance: float = 1.0e-10,
        damping: float = 1.0,
        /,
    ):
        if maximum_iterations <= 0 or tolerance <= 0.0 or not 0.0 < damping <= 1.0:
            raise ValueError("Newton-Krylov controls are invalid.")
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.damping = float(damping)

    def solve(
        self, residual: Callable[[Array], Array], initial: ArrayLike, /
    ) -> HPNewtonKrylovResult:
        value = jnp.asarray(initial)
        iteration = 0
        norm = jnp.linalg.norm(residual(value))
        while iteration < self.maximum_iterations and float(norm) > self.tolerance:
            current = residual(value)
            jacobian = jax.jacfwd(residual)(value)
            space = ArraySpace(value.shape, dtype=value.dtype)
            operator = DenseLinearOperator(
                jacobian.reshape((value.size, value.size)), source=space, target=space
            )
            correction = solve(
                LinearSystem(operator),
                -current.reshape(value.shape),
            ).value
            value = value + self.damping * correction
            iteration += 1
            norm = jnp.linalg.norm(residual(value))
        return HPNewtonKrylovResult(
            value, norm, jnp.asarray(iteration), norm <= self.tolerance
        )


class NonlinearLocalCondensation(StrictModule, NonTrainableState):
    retained_dofs: Array
    interior_dofs: Array
    newton: HPNewtonKrylovBuilder

    def __init__(
        self,
        local_size: int,
        retained_dofs: ArrayLike,
        /,
        *,
        newton: HPNewtonKrylovBuilder | None = None,
    ):
        size = int(local_size)
        retained = np.asarray(retained_dofs, dtype=np.int32)
        interior = np.setdiff1d(np.arange(size, dtype=np.int32), retained)
        if size <= 1 or retained.size == 0 or interior.size == 0:
            raise ValueError("Nonlinear condensation requires trace and interior DOFs.")
        self.retained_dofs = jnp.asarray(retained)
        self.interior_dofs = jnp.asarray(interior)
        self.newton = HPNewtonKrylovBuilder() if newton is None else newton

    def eliminate(
        self,
        local_residual: Callable[[Array], Array],
        retained_values: ArrayLike,
        interior_initial: ArrayLike,
        /,
    ) -> HPNewtonKrylovResult:
        retained = jnp.asarray(retained_values)

        def interior_residual(interior):
            full = jnp.zeros(
                (self.retained_dofs.size + self.interior_dofs.size,), dtype=interior.dtype
            )
            full = full.at[self.retained_dofs].set(retained)
            full = full.at[self.interior_dofs].set(interior)
            return local_residual(full)[self.interior_dofs]

        return self.newton.solve(interior_residual, interior_initial)


class HPFASMultigrid(StrictModule, NonTrainableState):
    level_count: int = eqx.field(static=True)
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)

    def __init__(
        self, level_count: int, /, *, pre_smoothing: int = 2, post_smoothing: int = 2
    ):
        if level_count < 2 or pre_smoothing < 0 or post_smoothing < 0:
            raise ValueError("FAS hierarchy or smoothing counts are invalid.")
        self.level_count = int(level_count)
        self.pre_smoothing = int(pre_smoothing)
        self.post_smoothing = int(post_smoothing)

    def cycle(
        self,
        level: int,
        value: Array,
        right_hand_side: Array,
        residuals: Sequence[Callable],
        smoothers: Sequence[Callable],
        restrict: Sequence[Callable],
        prolong: Sequence[Callable],
        /,
    ) -> Array:
        index = int(level)
        result = value
        for _ in range(self.pre_smoothing):
            result = smoothers[index](result, right_hand_side)
        if index == self.level_count - 1:
            return smoothers[index](result, right_hand_side)
        fine_residual = right_hand_side - residuals[index](result)
        coarse_value = restrict[index](result)
        coarse_rhs = residuals[index + 1](coarse_value) + restrict[index](fine_residual)
        corrected = self.cycle(
            index + 1, coarse_value, coarse_rhs, residuals, smoothers, restrict, prolong
        )
        result = result + prolong[index](corrected - coarse_value)
        for _ in range(self.post_smoothing):
            result = smoothers[index](result, right_hand_side)
        return result


class HPRestrictedSchwarz(StrictModule, NonTrainableState):
    restrictions: tuple[Array, ...]
    local_inverses: tuple[Array, ...]
    weights: tuple[Array, ...]
    multiplicative: bool = eqx.field(static=True)

    def __init__(
        self,
        restrictions: Sequence[ArrayLike],
        local_matrices: Sequence[ArrayLike],
        /,
        *,
        multiplicative: bool = False,
    ):
        restriction = tuple(jnp.asarray(value) for value in restrictions)
        matrices = tuple(np.asarray(value) for value in local_matrices)
        if not restriction or len(restriction) != len(matrices):
            raise ValueError("Schwarz restrictions and local matrices disagree.")
        inverses = tuple(jnp.asarray(np.linalg.inv(value)) for value in matrices)
        multiplicity = sum(
            np.asarray(value).T @ np.ones(value.shape[0]) for value in restriction
        )
        multiplicity = np.maximum(multiplicity, 1.0)
        weights = tuple(
            jnp.asarray(
                np.ones(value.shape[0])
                / multiplicity[np.argmax(np.asarray(value), axis=1)]
            )
            for value in restriction
        )
        self.restrictions = restriction
        self.local_inverses = inverses
        self.weights = weights
        self.multiplicative = bool(multiplicative)

    def apply(
        self, residual: ArrayLike, operator: Callable[[Array], Array] | None = None, /
    ) -> Array:
        residual_ = jnp.asarray(residual)
        correction = jnp.zeros_like(residual_)
        running = residual_
        for restriction, inverse, weight in zip(
            self.restrictions, self.local_inverses, self.weights, strict=True
        ):
            local = restriction @ running
            update = restriction.T @ (weight * (inverse @ local))
            correction = correction + update
            if self.multiplicative:
                if operator is None:
                    raise ValueError(
                        "Multiplicative Schwarz requires an operator action."
                    )
                running = residual_ - operator(correction)
        return correction


class BDDCFETIDPTracePlan(StrictModule, NonTrainableState):
    primal_constraints: Array
    scaling: Array
    coarse_matrix: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, primal_constraints: ArrayLike, local_schur: ArrayLike, /):
        constraints = jnp.asarray(primal_constraints)
        schur = jnp.asarray(local_schur)
        if (
            constraints.ndim != 2
            or schur.ndim != 2
            or schur.shape[0] != schur.shape[1]
            or constraints.shape[1] != schur.shape[0]
        ):
            raise ValueError("BDDC/FETI-DP trace data are incompatible.")
        multiplicity = jnp.maximum(jnp.sum(jnp.abs(constraints), axis=0), 1.0)
        scaling = 1.0 / multiplicity
        coarse = constraints @ schur @ constraints.T
        self.primal_constraints = constraints
        self.scaling = scaling
        self.coarse_matrix = coarse
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bddc-feti-dp",
                "constraints": list(constraints.shape),
                "schur": list(schur.shape),
            }
        )


class HPEigenspaceTransfer(StrictModule, NonTrainableState):
    transfer: FiniteElementHPTransferPlan

    def transfer_modes(self, modes: ArrayLike, /) -> Array:
        values = jnp.asarray(modes)
        return jax.vmap(self.transfer.apply_mass_projection, in_axes=-1, out_axes=-1)(
            values
        )

    def rayleigh_ritz(
        self,
        operator: Callable[[Array], Array],
        metric: Callable[[Array], Array],
        modes: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        vectors = jnp.asarray(modes)
        av = jax.vmap(operator, in_axes=-1, out_axes=-1)(vectors)
        mv = jax.vmap(metric, in_axes=-1, out_axes=-1)(vectors)
        stiffness = jnp.conj(vectors).T @ av
        mass = jnp.conj(vectors).T @ mv
        space = ArraySpace((mass.shape[0],), dtype=mass.dtype)
        metric_operator = DenseLinearOperator(mass, source=space, target=space)
        transformed = jnp.stack(
            tuple(
                solve(LinearSystem(metric_operator), stiffness[:, column]).value
                for column in range(stiffness.shape[1])
            ),
            axis=1,
        )
        eigenvalues, eigenvectors = jnp.linalg.eig(transformed)
        return eigenvalues, vectors @ eigenvectors


def goal_oriented_eigen_indicators(
    residual_by_cell: ArrayLike, dual_correction_by_cell: ArrayLike, /
) -> Array:
    residual = jnp.asarray(residual_by_cell)
    dual = jnp.asarray(dual_correction_by_cell)
    if residual.shape != dual.shape or residual.ndim < 2:
        raise ValueError(
            "Eigen residual and dual correction must share cell-leading shape."
        )
    axes = tuple(range(1, residual.ndim))
    return jnp.abs(jnp.sum(jnp.conj(dual) * residual, axis=axes))


class FrozenHPAdjointSchedule(StrictModule, NonTrainableState):
    transfers: tuple[FiniteElementHPTransferPlan, ...]
    schedule_id: str = eqx.field(static=True)

    def __init__(self, transfers: Sequence[FiniteElementHPTransferPlan], /):
        transfers_ = tuple(transfers)
        if any(
            not isinstance(value, FiniteElementHPTransferPlan) for value in transfers_
        ):
            raise TypeError("Frozen hp schedules require hp transfer plans.")
        self.transfers = transfers_
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "frozen-hp-adjoint",
                "transfers": [value.transfer_id for value in transfers_],
            }
        )

    def forward(self, value: ArrayLike, /) -> tuple[Array, ...]:
        states = [jnp.asarray(value)]
        for transfer in self.transfers:
            states.append(transfer.apply_mass_projection(states[-1]))
        return tuple(states)

    def reverse(self, terminal_dual: ArrayLike, /) -> Array:
        value = jnp.asarray(terminal_dual)
        for transfer in reversed(self.transfers):
            value = transfer.pullback_raw(value)
        return value


class RelaxedHPMarking(StrictModule, NonTrainableState):
    temperature: float = eqx.field(static=True)
    budget: int = eqx.field(static=True)

    def __init__(self, budget: int, temperature: float = 0.1, /):
        if budget <= 0 or temperature <= 0.0:
            raise ValueError("Relaxed marking budget and temperature must be positive.")
        self.temperature = float(temperature)
        self.budget = int(budget)

    def weights(self, indicators: ArrayLike, valid: ArrayLike, /) -> Array:
        values = jnp.asarray(indicators)
        valid_ = jnp.asarray(valid, dtype=bool)
        logits = jnp.where(valid_, values / self.temperature, -jnp.inf)
        probabilities = jax.nn.softmax(logits)
        return jnp.minimum(1.0, self.budget * probabilities)

    def safe_project(
        self, indicators: ArrayLike, valid: ArrayLike, stable_ids: ArrayLike, /
    ) -> Array:
        values = np.asarray(indicators)
        valid_ = np.asarray(valid, dtype=bool)
        ids = np.asarray(stable_ids)
        candidates = np.flatnonzero(valid_)
        ordered = sorted(
            candidates.tolist(),
            key=lambda index: (-float(values[index]), tuple(ids[index].tolist())),
        )
        selected = np.zeros(valid_.shape, dtype=bool)
        selected[ordered[: self.budget]] = True
        return jnp.asarray(selected)


class MeshVaryingUQAggregator(StrictModule, NonTrainableState):
    reference_size: int = eqx.field(static=True)

    def __init__(self, reference_size: int, /):
        if reference_size <= 0:
            raise ValueError("Reference UQ field size must be positive.")
        self.reference_size = int(reference_size)

    def aggregate(
        self,
        fields: Sequence[ArrayLike],
        projections: Sequence[ArrayLike],
        weights: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        fields_ = tuple(jnp.asarray(value) for value in fields)
        projections_ = tuple(jnp.asarray(value) for value in projections)
        weights_ = jnp.asarray(weights)
        if len(fields_) != len(projections_) or weights_.shape != (len(fields_),):
            raise ValueError("Mesh-varying fields, projections, and weights disagree.")
        projected = jnp.stack(
            [
                projection @ field
                for projection, field in zip(projections_, fields_, strict=True)
            ]
        )
        normalized = weights_ / jnp.sum(weights_)
        mean = jnp.sum(
            normalized.reshape((-1,) + (1,) * (projected.ndim - 1)) * projected, axis=0
        )
        centered = projected - mean
        variance = jnp.sum(
            normalized.reshape((-1,) + (1,) * (projected.ndim - 1))
            * jnp.abs(centered) ** 2,
            axis=0,
        )
        return mean, variance


__all__ = [
    "BDDCFETIDPTracePlan",
    "FrozenHPAdjointSchedule",
    "HPEigenspaceTransfer",
    "HPFASMultigrid",
    "HPNewtonKrylovBuilder",
    "HPNewtonKrylovResult",
    "HPRestrictedSchwarz",
    "MeshVaryingUQAggregator",
    "NonlinearLocalCondensation",
    "RelaxedHPMarking",
    "goal_oriented_eigen_indicators",
]
