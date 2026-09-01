#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _symmetric(value: np.ndarray, /) -> np.ndarray:
    return 0.5 * (value + np.conjugate(value.T))


def _require_spd(value: np.ndarray, name: str, /) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be one nonempty square matrix.")
    if not np.issubdtype(matrix.dtype, np.number) or np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite and numeric.")
    hermitian = _symmetric(matrix)
    scale = max(float(np.linalg.norm(hermitian, ord=2)), 1.0)
    tolerance = 128.0 * np.finfo(np.asarray(hermitian.real).dtype).eps * scale
    if np.linalg.norm(matrix - np.conjugate(matrix.T), ord=2) > tolerance:
        raise ValueError(f"{name} must be self-adjoint.")
    if float(np.min(np.linalg.eigvalsh(hermitian))) <= tolerance:
        raise ValueError(f"{name} must be positive definite.")
    return hermitian


def _independent_rows(rows: np.ndarray, tolerance: float, /) -> np.ndarray:
    if rows.shape[0] == 0:
        return rows
    retained: list[np.ndarray] = []
    rank = 0
    for row in rows:
        candidate = np.stack((*retained, row), axis=0)
        next_rank = int(np.linalg.matrix_rank(candidate, tol=tolerance))
        if next_rank > rank:
            retained.append(row)
            rank = next_rank
    return np.stack(retained, axis=0) if retained else rows[:0]


def _interface_schur(
    matrix: np.ndarray,
    interface_indices: np.ndarray,
    /,
) -> np.ndarray:
    boundary = np.asarray(interface_indices, dtype=np.int64)
    interior = np.setdiff1d(
        np.arange(matrix.shape[0], dtype=np.int64), boundary, assume_unique=True
    )
    block = matrix[np.ix_(boundary, boundary)]
    if interior.size:
        coupling = matrix[np.ix_(boundary, interior)]
        block = block - coupling @ np.linalg.solve(
            matrix[np.ix_(interior, interior)], np.conjugate(coupling.T)
        )
    return _require_spd(_symmetric(block), "interface Schur complement")


class SubstructuredSPDSystem(StrictModule, NonTrainableState):
    """Broken SPD operators with deterministic local-to-global assembly."""

    local_matrices: tuple[Array, ...]
    local_to_global: tuple[Array, ...]
    global_dof_count: int = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_matrices: Sequence[ArrayLike],
        local_to_global: Sequence[ArrayLike],
        /,
        *,
        polynomial_degree: int = 1,
    ):
        matrices_host = tuple(
            _require_spd(np.asarray(value), f"local matrix {index}")
            for index, value in enumerate(local_matrices)
        )
        maps_host = tuple(np.asarray(value, dtype=np.int64) for value in local_to_global)
        degree = int(polynomial_degree)
        if not matrices_host or len(matrices_host) != len(maps_host):
            raise ValueError(
                "Substructured matrices and maps must be nonempty and aligned."
            )
        if degree < 1:
            raise ValueError("polynomial_degree must be positive.")
        for index, (matrix, mapping) in enumerate(
            zip(matrices_host, maps_host, strict=True)
        ):
            if (
                mapping.shape != (matrix.shape[0],)
                or np.any(mapping < 0)
                or np.unique(mapping).size != mapping.size
            ):
                raise ValueError(f"local-to-global map {index} is invalid.")
        global_count = int(max(int(np.max(value)) for value in maps_host)) + 1
        covered = np.unique(np.concatenate(maps_host))
        if not np.array_equal(covered, np.arange(global_count)):
            raise ValueError(
                "Local maps must cover every global coordinate exactly by ID."
            )
        dtype = np.result_type(*(matrix.dtype for matrix in matrices_host))
        matrices_host = tuple(
            matrix.astype(dtype, copy=False) for matrix in matrices_host
        )
        self.local_matrices = tuple(jnp.asarray(value) for value in matrices_host)
        self.local_to_global = tuple(jnp.asarray(value) for value in maps_host)
        self.global_dof_count = global_count
        self.polynomial_degree = degree
        self.system_id = canonical_fingerprint(
            {
                "kind": "substructured-spd-system",
                "matrices": [array_tree_fingerprint(value) for value in matrices_host],
                "maps": [array_tree_fingerprint(value) for value in maps_host],
                "degree": degree,
            }
        )

    @property
    def subdomain_count(self) -> int:
        return len(self.local_matrices)

    @property
    def broken_dof_count(self) -> int:
        return sum(int(matrix.shape[0]) for matrix in self.local_matrices)

    def assemble_matrix(self, /) -> Array:
        dtype = self.local_matrices[0].dtype
        result = jnp.zeros((self.global_dof_count, self.global_dof_count), dtype=dtype)
        for matrix, mapping in zip(
            self.local_matrices, self.local_to_global, strict=True
        ):
            result = result.at[mapping[:, None], mapping[None, :]].add(matrix)
        return result

    def restrict(self, value: ArrayLike, /) -> tuple[Array, ...]:
        vector = jnp.asarray(value)
        if vector.ndim == 0 or vector.shape[0] != self.global_dof_count:
            raise ValueError("Global vector has an invalid leading dimension.")
        return tuple(vector[mapping] for mapping in self.local_to_global)

    def assemble_local(
        self,
        values: Sequence[ArrayLike],
        /,
        *,
        average: bool = False,
    ) -> Array:
        local = tuple(jnp.asarray(value) for value in values)
        if len(local) != self.subdomain_count:
            raise ValueError("Local values must provide one array per subdomain.")
        trailing = local[0].shape[1:]
        dtype = jnp.result_type(*(value.dtype for value in local))
        result = jnp.zeros((self.global_dof_count,) + trailing, dtype=dtype)
        count = jnp.zeros((self.global_dof_count,) + (1,) * len(trailing), dtype=dtype)
        for index, (value, mapping, matrix) in enumerate(
            zip(local, self.local_to_global, self.local_matrices, strict=True)
        ):
            if value.shape != (matrix.shape[0],) + trailing:
                raise ValueError(f"Local value {index} has an invalid shape.")
            result = result.at[mapping].add(value)
            count = count.at[mapping].add(
                jnp.ones((mapping.shape[0],) + (1,) * len(trailing))
            )
        return result / count if average else result


class PrimalConstraintPlan(StrictModule, NonTrainableState):
    """Global corner/edge/face averages promoted to primal continuity."""

    global_dof_ids: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, global_dof_ids: ArrayLike = (), /):
        identifiers = np.asarray(global_dof_ids, dtype=np.int64)
        if identifiers.ndim != 1 or np.any(identifiers < 0):
            raise ValueError("Primal global DOF IDs must be one nonnegative vector.")
        if np.unique(identifiers).size != identifiers.size:
            raise ValueError("Primal global DOF IDs must be unique.")
        identifiers = np.sort(identifiers, kind="stable")
        self.global_dof_ids = jnp.asarray(identifiers)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "primal-constraint-plan",
                "global_dof_ids": array_tree_fingerprint(identifiers),
            }
        )


class InterfaceDeluxeScaling(StrictModule, NonTrainableState):
    """Operator-valued deluxe partition of unity on one pair interface."""

    left_subdomain: int = eqx.field(static=True)
    right_subdomain: int = eqx.field(static=True)
    global_dof_ids: Array
    left_local_indices: Array
    right_local_indices: Array
    left_weight: Array
    right_weight: Array
    parallel_sum: Array
    partition_unity_error: Array
    interface_id: str = eqx.field(static=True)


class DeluxeScalingPlan(StrictModule, NonTrainableState):
    """Schur-complement deluxe scaling for every two-subdomain interface."""

    interfaces: tuple[InterfaceDeluxeScaling, ...]
    multiplicity: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, system: SubstructuredSPDSystem, /):
        if not isinstance(system, SubstructuredSPDSystem):
            raise TypeError("system must be SubstructuredSPDSystem.")
        maps = tuple(np.asarray(value) for value in system.local_to_global)
        matrices = tuple(np.asarray(value) for value in system.local_matrices)
        multiplicity = np.zeros((system.global_dof_count,), dtype=np.int32)
        for mapping in maps:
            multiplicity[mapping] += 1
        interfaces: list[InterfaceDeluxeScaling] = []
        for left in range(system.subdomain_count):
            for right in range(left + 1, system.subdomain_count):
                shared = np.intersect1d(maps[left], maps[right], assume_unique=True)
                shared = shared[multiplicity[shared] == 2]
                if shared.size == 0:
                    continue
                left_indices = np.searchsorted(maps[left], shared)
                right_indices = np.searchsorted(maps[right], shared)
                if not np.array_equal(maps[left][left_indices], shared):
                    lookup = {int(value): index for index, value in enumerate(maps[left])}
                    left_indices = np.asarray([lookup[int(value)] for value in shared])
                if not np.array_equal(maps[right][right_indices], shared):
                    lookup = {
                        int(value): index for index, value in enumerate(maps[right])
                    }
                    right_indices = np.asarray([lookup[int(value)] for value in shared])
                left_schur = _interface_schur(matrices[left], left_indices)
                right_schur = _interface_schur(matrices[right], right_indices)
                total = left_schur + right_schur
                # The opposite-side convention is the energy-minimizing deluxe average.
                left_weight = np.linalg.solve(total, right_schur)
                right_weight = np.linalg.solve(total, left_schur)
                identity = np.eye(shared.size, dtype=total.dtype)
                unity_error = np.linalg.norm(left_weight + right_weight - identity, ord=2)
                parallel = _symmetric(left_schur @ np.linalg.solve(total, right_schur))
                interface_id = canonical_fingerprint(
                    {
                        "kind": "deluxe-interface",
                        "system": system.system_id,
                        "left": left,
                        "right": right,
                        "global_dof_ids": array_tree_fingerprint(shared),
                    }
                )
                interfaces.append(
                    InterfaceDeluxeScaling(
                        left,
                        right,
                        jnp.asarray(shared),
                        jnp.asarray(left_indices),
                        jnp.asarray(right_indices),
                        jnp.asarray(left_weight),
                        jnp.asarray(right_weight),
                        jnp.asarray(parallel),
                        jnp.asarray(unity_error),
                        interface_id,
                    )
                )
        self.interfaces = tuple(interfaces)
        self.multiplicity = jnp.asarray(multiplicity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "deluxe-scaling-plan",
                "system": system.system_id,
                "interfaces": [value.interface_id for value in interfaces],
                "multiplicity": array_tree_fingerprint(multiplicity),
            }
        )


class AdaptiveInterfaceModes(StrictModule, NonTrainableState):
    """Required generalized eigenmodes for one deluxe interface."""

    interface_id: str = eqx.field(static=True)
    global_dof_ids: Array
    eigenvalues: Array
    eigenvectors: Array
    required_mask: Array
    required_count: int = eqx.field(static=True)


class AdaptiveSpectralCoarseSpace(StrictModule, NonTrainableState):
    """Adaptive primal modes selected by the deluxe generalized eigenproblem."""

    modes: tuple[AdaptiveInterfaceModes, ...]
    threshold: float = eqx.field(static=True)
    required_mode_count: int = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: SubstructuredSPDSystem,
        deluxe: DeluxeScalingPlan,
        /,
        *,
        threshold: float,
        maximum_modes: int,
        primal: PrimalConstraintPlan | None = None,
    ):
        if not isinstance(system, SubstructuredSPDSystem) or not isinstance(
            deluxe, DeluxeScalingPlan
        ):
            raise TypeError("Adaptive coarse setup requires system and deluxe scaling.")
        if primal is not None and not isinstance(primal, PrimalConstraintPlan):
            raise TypeError("primal must be PrimalConstraintPlan or None.")
        threshold_ = float(threshold)
        cap = int(maximum_modes)
        if not isfinite(threshold_) or threshold_ <= 1.0 or cap < 0:
            raise ValueError(
                "Adaptive threshold must exceed one and mode cap be nonnegative."
            )
        primal_ids = (
            set() if primal is None else set(np.asarray(primal.global_dof_ids).tolist())
        )
        matrices = tuple(np.asarray(value) for value in system.local_matrices)
        interface_modes: list[AdaptiveInterfaceModes] = []
        required_total = 0
        for interface in deluxe.interfaces:
            ids_all = np.asarray(interface.global_dof_ids)
            keep = np.asarray([int(value) not in primal_ids for value in ids_all])
            ids = ids_all[keep]
            if ids.size == 0:
                continue
            left_all = np.asarray(interface.left_local_indices)
            right_all = np.asarray(interface.right_local_indices)
            left_indices = left_all[keep]
            right_indices = right_all[keep]
            left_schur = _interface_schur(
                matrices[interface.left_subdomain], left_indices
            )
            right_schur = _interface_schur(
                matrices[interface.right_subdomain], right_indices
            )
            total = left_schur + right_schur
            left_weight = np.linalg.solve(total, right_schur)
            right_weight = np.linalg.solve(total, left_schur)
            jump_energy = _symmetric(
                np.conjugate(right_weight.T) @ left_schur @ right_weight
                + np.conjugate(left_weight.T) @ right_schur @ left_weight
            )
            parallel = _require_spd(
                _symmetric(left_schur @ np.linalg.solve(total, right_schur)),
                "deluxe parallel sum",
            )
            factor = np.linalg.cholesky(parallel)
            transformed = np.linalg.solve(
                factor,
                np.linalg.solve(factor, jump_energy).T,
            ).T
            transformed = _symmetric(transformed)
            eigenvalues, transformed_vectors = np.linalg.eigh(transformed)
            eigenvectors = np.linalg.solve(np.conjugate(factor.T), transformed_vectors)
            order = np.argsort(eigenvalues, kind="stable")[::-1]
            eigenvalues = np.real(eigenvalues[order])
            eigenvectors = eigenvectors[:, order]
            required = eigenvalues > threshold_
            count = int(np.count_nonzero(required))
            required_total += count
            interface_modes.append(
                AdaptiveInterfaceModes(
                    interface.interface_id,
                    jnp.asarray(ids),
                    jnp.asarray(eigenvalues),
                    jnp.asarray(eigenvectors),
                    jnp.asarray(required),
                    count,
                )
            )
        if required_total > cap:
            raise ValueError(
                "Adaptive spectral coarse space requires "
                f"{required_total} modes, exceeding the declared cap {cap}; "
                "required modes are never truncated."
            )
        self.modes = tuple(interface_modes)
        self.threshold = threshold_
        self.required_mode_count = required_total
        self.maximum_modes = cap
        self.space_id = canonical_fingerprint(
            {
                "kind": "adaptive-spectral-coarse-space",
                "system": system.system_id,
                "deluxe": deluxe.plan_id,
                "threshold": threshold_.hex(),
                "maximum_modes": cap,
                "interfaces": [
                    {
                        "id": value.interface_id,
                        "eigenvalues": array_tree_fingerprint(value.eigenvalues),
                        "required": array_tree_fingerprint(value.required_mask),
                    }
                    for value in interface_modes
                ],
            }
        )


class PRobustnessEvidence(StrictModule, NonTrainableState):
    """Measured condition evidence for one polynomial degree and coarse space."""

    polynomial_degree: int = eqx.field(static=True)
    condition_estimate: Array
    logarithmic_degree_growth: Array
    adaptive_mode_count: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    growth_limit: float = eqx.field(static=True)
    passed: Array
    evidence_id: str = eqx.field(static=True)


class SubstructuringEvidence(StrictModule, NonTrainableState):
    method: str = eqx.field(static=True)
    condition_estimate: Array
    primal_constraint_count: int = eqx.field(static=True)
    adaptive_mode_count: int = eqx.field(static=True)
    deluxe_partition_unity_error: Array
    symmetric_error: Array
    positive_minimum: Array
    evidence_id: str = eqx.field(static=True)


class PreparedSPDSubstructuring(StrictModule, NonTrainableState):
    """Prepared BDDC/IETI-DP/FETI-DP primal and dual operators."""

    system: SubstructuredSPDSystem
    primal: PrimalConstraintPlan
    deluxe: DeluxeScalingPlan
    adaptive: AdaptiveSpectralCoarseSpace | None
    broken_matrix: Array
    averaging: Array
    constraint_matrix: Array
    jump_matrix: Array
    kkt_matrix: Array
    global_matrix: Array
    preconditioner_matrix: Array
    evidence: SubstructuringEvidence
    method: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def apply(self, right_hand_side: ArrayLike, /) -> Array:
        value = jnp.asarray(right_hand_side)
        if value.ndim == 0 or value.shape[0] != self.system.global_dof_count:
            raise ValueError("Substructuring right-hand side has an invalid shape.")
        return self.preconditioner_matrix @ value

    def transpose_apply(self, right_hand_side: ArrayLike, /) -> Array:
        value = jnp.asarray(right_hand_side)
        if value.ndim == 0 or value.shape[0] != self.system.global_dof_count:
            raise ValueError("Substructuring right-hand side has an invalid shape.")
        return jnp.conjugate(self.preconditioner_matrix.T) @ value

    def dual_operator(self, multiplier: ArrayLike, /) -> Array:
        value = jnp.asarray(multiplier)
        if value.ndim == 0 or value.shape[0] != self.jump_matrix.shape[0]:
            raise ValueError("FETI-DP multiplier has an invalid leading dimension.")
        if self.jump_matrix.shape[0] == 0:
            return jnp.zeros_like(value)
        broken_rhs = jnp.conjugate(self.jump_matrix.T) @ value
        broken = self._constrained_broken_solve(broken_rhs)
        return self.jump_matrix @ broken

    def dual_precondition(self, multiplier: ArrayLike, /) -> Array:
        value = jnp.asarray(multiplier)
        if value.ndim == 0 or value.shape[0] != self.jump_matrix.shape[0]:
            raise ValueError("FETI-DP multiplier has an invalid leading dimension.")
        if self.jump_matrix.shape[0] == 0:
            return jnp.zeros_like(value)
        diagonal = jnp.real(jnp.diag(self.broken_matrix))
        scaled = self.jump_matrix * diagonal[None, :]
        operator = scaled @ jnp.conjugate(self.jump_matrix.T)
        return jnp.linalg.solve(operator, value)

    def recover_primal(
        self,
        right_hand_side: ArrayLike,
        multiplier: ArrayLike,
        /,
    ) -> Array:
        rhs = jnp.asarray(right_hand_side)
        dual = jnp.asarray(multiplier)
        if rhs.ndim == 0 or rhs.shape[0] != self.system.global_dof_count:
            raise ValueError("Primal recovery right-hand side has an invalid shape.")
        if dual.ndim == 0 or dual.shape[0] != self.jump_matrix.shape[0]:
            raise ValueError("Primal recovery multiplier has an invalid shape.")
        broken_rhs = jnp.conjugate(self.averaging.T) @ rhs
        if dual.shape[0]:
            broken_rhs = broken_rhs - jnp.conjugate(self.jump_matrix.T) @ dual
        return self.averaging @ self._constrained_broken_solve(broken_rhs)

    def _constrained_broken_solve(self, right_hand_side: Array, /) -> Array:
        constraints = self.constraint_matrix.shape[0]
        padded = jnp.concatenate(
            (
                right_hand_side,
                jnp.zeros(
                    (constraints,) + right_hand_side.shape[1:],
                    dtype=right_hand_side.dtype,
                ),
            ),
            axis=0,
        )
        return jnp.linalg.solve(self.kkt_matrix, padded)[: self.broken_matrix.shape[0]]

    def p_robust_evidence(
        self,
        /,
        *,
        condition_limit: float,
        growth_limit: float = 2.0,
    ) -> PRobustnessEvidence:
        condition = float(condition_limit)
        growth = float(growth_limit)
        if not isfinite(condition) or condition <= 0.0:
            raise ValueError("condition_limit must be finite and positive.")
        if not isfinite(growth) or growth < 0.0:
            raise ValueError("growth_limit must be finite and nonnegative.")
        estimate = self.evidence.condition_estimate
        degree = self.system.polynomial_degree
        logarithmic_growth = jnp.log(jnp.maximum(estimate, 1.0)) / jnp.log(
            jnp.asarray(degree + 1, dtype=estimate.dtype)
        )
        passed = (
            jnp.isfinite(estimate)
            & (estimate <= condition)
            & (logarithmic_growth <= growth)
        )
        mode_count = 0 if self.adaptive is None else self.adaptive.required_mode_count
        return PRobustnessEvidence(
            degree,
            estimate,
            logarithmic_growth,
            mode_count,
            condition,
            growth,
            passed,
            canonical_fingerprint(
                {
                    "kind": "p-robust-substructuring-evidence",
                    "prepared": self.prepared_id,
                    "condition_limit": condition.hex(),
                    "growth_limit": growth.hex(),
                }
            ),
        )


class SPDSubstructuringPlan(StrictModule, NonTrainableState):
    """Prepare SPD H1 BDDC, IETI-DP, or FETI-DP without hidden fallbacks."""

    method: str = eqx.field(static=True)
    primal: PrimalConstraintPlan
    adaptive_threshold: float | None = eqx.field(static=True)
    maximum_adaptive_modes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: str,
        /,
        *,
        primal: PrimalConstraintPlan | None = None,
        adaptive_threshold: float | None = None,
        maximum_adaptive_modes: int = 0,
    ):
        method_ = str(method).lower().replace("_", "-")
        if method_ not in ("bddc", "ieti-dp", "feti-dp"):
            raise ValueError("method must be 'bddc', 'ieti-dp', or 'feti-dp'.")
        primal_ = PrimalConstraintPlan() if primal is None else primal
        if not isinstance(primal_, PrimalConstraintPlan):
            raise TypeError("primal must be PrimalConstraintPlan or None.")
        threshold = None if adaptive_threshold is None else float(adaptive_threshold)
        cap = int(maximum_adaptive_modes)
        if threshold is not None and (not isfinite(threshold) or threshold <= 1.0):
            raise ValueError("adaptive_threshold must exceed one or be None.")
        if cap < 0 or (threshold is None and cap != 0):
            raise ValueError("Adaptive mode capacity requires an adaptive threshold.")
        self.method = method_
        self.primal = primal_
        self.adaptive_threshold = threshold
        self.maximum_adaptive_modes = cap
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spd-substructuring-plan",
                "method": method_,
                "primal": primal_.plan_id,
                "adaptive_threshold": None if threshold is None else threshold.hex(),
                "maximum_adaptive_modes": cap,
            }
        )

    def prepare(self, system: SubstructuredSPDSystem, /) -> PreparedSPDSubstructuring:
        if not isinstance(system, SubstructuredSPDSystem):
            raise TypeError("system must be SubstructuredSPDSystem.")
        primal_ids = np.asarray(self.primal.global_dof_ids)
        if np.any(primal_ids >= system.global_dof_count):
            raise ValueError("Primal constraint references an absent global DOF.")
        deluxe = DeluxeScalingPlan(system)
        adaptive = (
            None
            if self.adaptive_threshold is None
            else AdaptiveSpectralCoarseSpace(
                system,
                deluxe,
                threshold=self.adaptive_threshold,
                maximum_modes=self.maximum_adaptive_modes,
                primal=self.primal,
            )
        )
        return _prepare_substructuring(system, self, deluxe, adaptive)


class InexactNewtonEvidence(StrictModule, NonTrainableState):
    iterations: Array
    requested_relative_residual: Array
    achieved_relative_residual: Array
    converged: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


class InexactNewtonTangentPreconditioner(StrictModule, NonTrainableState):
    """Substructured tangent inverse with Eisenstat--Walker forcing evidence."""

    prepared: PreparedSPDSubstructuring
    minimum_forcing: float = eqx.field(static=True)
    maximum_forcing: float = eqx.field(static=True)
    forcing_power: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    preconditioner_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedSPDSubstructuring,
        /,
        *,
        minimum_forcing: float = 1.0e-4,
        maximum_forcing: float = 0.9,
        forcing_power: float = 1.5,
        maximum_iterations: int = 32,
    ):
        if not isinstance(prepared, PreparedSPDSubstructuring):
            raise TypeError("prepared must be PreparedSPDSubstructuring.")
        minimum = float(minimum_forcing)
        maximum = float(maximum_forcing)
        power = float(forcing_power)
        iterations = int(maximum_iterations)
        if (
            not all(isfinite(value) for value in (minimum, maximum, power))
            or not 0.0 < minimum <= maximum < 1.0
            or power <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Inexact Newton forcing or iteration policy is invalid.")
        self.prepared = prepared
        self.minimum_forcing = minimum
        self.maximum_forcing = maximum
        self.forcing_power = power
        self.maximum_iterations = iterations
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "inexact-newton-substructure-preconditioner",
                "prepared": prepared.prepared_id,
                "minimum": minimum.hex(),
                "maximum": maximum.hex(),
                "power": power.hex(),
                "iterations": iterations,
            }
        )

    def forcing(
        self,
        residual_norm: ArrayLike,
        previous_residual_norm: ArrayLike,
        /,
    ) -> Array:
        current = jnp.asarray(residual_norm)
        previous = jnp.asarray(previous_residual_norm, dtype=current.dtype)
        ratio = current / jnp.maximum(previous, jnp.finfo(current.dtype).tiny)
        return jnp.clip(
            ratio**self.forcing_power,
            self.minimum_forcing,
            self.maximum_forcing,
        )

    def solve(
        self,
        tangent_action: Callable[[Array], Array],
        right_hand_side: ArrayLike,
        /,
        *,
        residual_norm: ArrayLike,
        previous_residual_norm: ArrayLike,
    ) -> tuple[Array, InexactNewtonEvidence]:
        if not callable(tangent_action):
            raise TypeError("tangent_action must be callable.")
        rhs = jnp.asarray(right_hand_side)
        if rhs.shape != (self.prepared.system.global_dof_count,):
            raise ValueError("Inexact tangent right-hand side has an invalid shape.")
        target = self.forcing(residual_norm, previous_residual_norm)
        norm = jnp.linalg.norm(rhs)
        solution = jnp.zeros_like(rhs)
        relative = jnp.asarray(jnp.inf, dtype=norm.dtype)
        iterations = jnp.asarray(0, dtype=jnp.int32)
        active = jnp.asarray(True)
        for index in range(self.maximum_iterations):
            residual = rhs - tangent_action(solution)
            relative = jnp.linalg.norm(residual) / jnp.maximum(
                norm, jnp.finfo(norm.dtype).tiny
            )
            update = self.prepared.apply(residual)
            take = active & jnp.isfinite(relative) & (relative > target)
            solution = jnp.where(take, solution + update, solution)
            iterations = jnp.where(take, index + 1, iterations)
            active = take
        residual = rhs - tangent_action(solution)
        relative = jnp.linalg.norm(residual) / jnp.maximum(
            norm, jnp.finfo(norm.dtype).tiny
        )
        finite = jnp.all(jnp.isfinite(solution)) & jnp.isfinite(relative)
        converged = finite & (relative <= target)
        evidence = InexactNewtonEvidence(
            iterations,
            target,
            relative,
            converged,
            finite,
            self.preconditioner_id,
        )
        return solution, evidence

    def aspin_tangent_hook(self, /) -> Callable[[Array], Array]:
        """Return the additive substructured inverse used by existing ASPIN."""

        return self.prepared.apply

    def raspen_tangent_hook(self, /) -> Callable[[Array], Array]:
        """Return the restricted additive inverse used by a RASPEN residual."""

        system = self.prepared.system
        owners = _global_replica_owners(system)

        def action(value: Array) -> Array:
            local = system.restrict(value)
            solved = tuple(
                jnp.linalg.solve(matrix, rhs)
                for matrix, rhs in zip(system.local_matrices, local, strict=True)
            )
            result = jnp.zeros_like(jnp.asarray(value))
            for subdomain, (mapping, correction) in enumerate(
                zip(system.local_to_global, solved, strict=True)
            ):
                mask = owners[mapping] == subdomain
                result = result.at[mapping].add(jnp.where(mask, correction, 0.0))
            return result

        return action


def _broken_layout(
    system: SubstructuredSPDSystem,
    /,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...]]:
    offsets = [0]
    for matrix in system.local_matrices:
        offsets.append(offsets[-1] + matrix.shape[0])
    broken_maps = tuple(
        np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
        for index in range(system.subdomain_count)
    )
    global_by_broken = np.concatenate(
        tuple(np.asarray(value) for value in system.local_to_global)
    )
    return global_by_broken, broken_maps, tuple(offsets)


def _global_replica_owners(system: SubstructuredSPDSystem, /) -> Array:
    owners = np.full((system.global_dof_count,), system.subdomain_count, dtype=np.int32)
    for subdomain, mapping in enumerate(system.local_to_global):
        owners[np.asarray(mapping)] = np.minimum(owners[np.asarray(mapping)], subdomain)
    return jnp.asarray(owners)


def _constraint_rows(
    system: SubstructuredSPDSystem,
    primal: PrimalConstraintPlan,
    adaptive: AdaptiveSpectralCoarseSpace | None,
    broken_maps: tuple[np.ndarray, ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.asarray(system.local_matrices[0]).dtype
    maps = tuple(np.asarray(value) for value in system.local_to_global)
    rows: list[np.ndarray] = []
    elementary: list[np.ndarray] = []
    primal_ids = set(np.asarray(primal.global_dof_ids).tolist())
    for global_id in range(system.global_dof_count):
        replicas = [
            int(broken_maps[subdomain][np.flatnonzero(mapping == global_id)[0]])
            for subdomain, mapping in enumerate(maps)
            if np.any(mapping == global_id)
        ]
        for replica in replicas[1:]:
            row = np.zeros((system.broken_dof_count,), dtype=dtype)
            row[replicas[0]] = 1.0
            row[replica] = -1.0
            elementary.append(row)
            if global_id in primal_ids:
                rows.append(row)
    if adaptive is not None:
        interface_by_id = {
            value.interface_id: value for value in DeluxeScalingPlan(system).interfaces
        }
        for modes in adaptive.modes:
            interface = interface_by_id[modes.interface_id]
            ids = np.asarray(modes.global_dof_ids)
            left_map = maps[interface.left_subdomain]
            right_map = maps[interface.right_subdomain]
            left_lookup = {int(value): index for index, value in enumerate(left_map)}
            right_lookup = {int(value): index for index, value in enumerate(right_map)}
            for mode_index in np.flatnonzero(np.asarray(modes.required_mask)):
                vector = np.asarray(modes.eigenvectors)[:, mode_index]
                row = np.zeros((system.broken_dof_count,), dtype=dtype)
                for coefficient, global_id in zip(vector, ids, strict=True):
                    row[
                        broken_maps[interface.left_subdomain][left_lookup[int(global_id)]]
                    ] += coefficient
                    row[
                        broken_maps[interface.right_subdomain][
                            right_lookup[int(global_id)]
                        ]
                    ] -= coefficient
                rows.append(row)
    row_array = (
        np.stack(rows, axis=0)
        if rows
        else np.zeros((0, system.broken_dof_count), dtype=dtype)
    )
    jump_array = (
        np.stack(elementary, axis=0)
        if elementary
        else np.zeros((0, system.broken_dof_count), dtype=dtype)
    )
    tolerance = 256.0 * np.finfo(np.asarray(row_array.real).dtype).eps
    row_array = _independent_rows(row_array, tolerance)
    if row_array.shape[0] and jump_array.shape[0]:
        gram = row_array @ np.conjugate(row_array.T)
        jump_array = jump_array - (
            jump_array @ np.conjugate(row_array.T)
        ) @ np.linalg.solve(gram, row_array)
    jump_array = _independent_rows(jump_array, tolerance)
    return row_array, jump_array


def _averaging_matrix(
    system: SubstructuredSPDSystem,
    deluxe: DeluxeScalingPlan,
    broken_maps: tuple[np.ndarray, ...],
    /,
) -> np.ndarray:
    dtype = np.asarray(system.local_matrices[0]).dtype
    maps = tuple(np.asarray(value) for value in system.local_to_global)
    averaging = np.zeros((system.global_dof_count, system.broken_dof_count), dtype=dtype)
    multiplicity = np.asarray(deluxe.multiplicity)
    covered = np.zeros((system.global_dof_count,), dtype=bool)
    for interface in deluxe.interfaces:
        ids = np.asarray(interface.global_dof_ids)
        left_broken = broken_maps[interface.left_subdomain][
            np.asarray(interface.left_local_indices)
        ]
        right_broken = broken_maps[interface.right_subdomain][
            np.asarray(interface.right_local_indices)
        ]
        averaging[np.ix_(ids, left_broken)] = np.asarray(interface.left_weight)
        averaging[np.ix_(ids, right_broken)] = np.asarray(interface.right_weight)
        covered[ids] = True
    diagonals = tuple(
        np.real(np.diag(np.asarray(value))) for value in system.local_matrices
    )
    for global_id in range(system.global_dof_count):
        if covered[global_id]:
            continue
        replicas: list[tuple[int, int, int]] = []
        energies: list[float] = []
        for subdomain, mapping in enumerate(maps):
            local = np.flatnonzero(mapping == global_id)
            if local.size:
                local_index = int(local[0])
                replicas.append(
                    (subdomain, local_index, int(broken_maps[subdomain][local_index]))
                )
                energies.append(float(diagonals[subdomain][local_index]))
        if multiplicity[global_id] == 1:
            averaging[global_id, replicas[0][2]] = 1.0
        else:
            total = sum(energies)
            for replica, energy in zip(replicas, energies, strict=True):
                averaging[global_id, replica[2]] = energy / total
    defect = averaging @ np.concatenate(
        tuple(np.eye(system.global_dof_count, dtype=dtype)[mapping] for mapping in maps),
        axis=0,
    )
    tolerance = 512.0 * np.finfo(np.asarray(defect.real).dtype).eps
    if np.linalg.norm(defect - np.eye(system.global_dof_count), ord=2) > tolerance:
        raise ValueError("Deluxe averaging does not form a partition of unity.")
    return averaging


def _prepare_substructuring(
    system: SubstructuredSPDSystem,
    plan: SPDSubstructuringPlan,
    deluxe: DeluxeScalingPlan,
    adaptive: AdaptiveSpectralCoarseSpace | None,
    /,
) -> PreparedSPDSubstructuring:
    global_by_broken, broken_maps, offsets = _broken_layout(system)
    del global_by_broken
    dtype = np.asarray(system.local_matrices[0]).dtype
    broken = np.zeros((system.broken_dof_count, system.broken_dof_count), dtype=dtype)
    for index, matrix in enumerate(system.local_matrices):
        broken[
            offsets[index] : offsets[index + 1], offsets[index] : offsets[index + 1]
        ] = np.asarray(matrix)
    averaging = _averaging_matrix(system, deluxe, broken_maps)
    constraints, jumps = _constraint_rows(system, plan.primal, adaptive, broken_maps)
    count = constraints.shape[0]
    kkt = np.block(
        [
            [broken, np.conjugate(constraints.T)],
            [constraints, np.zeros((count, count), dtype=dtype)],
        ]
    )
    if np.linalg.matrix_rank(kkt) != kkt.shape[0]:
        raise ValueError("Primal constraints do not define a nonsingular coarse problem.")
    inverse_top = np.linalg.solve(
        kkt,
        np.concatenate(
            (
                np.eye(system.broken_dof_count, dtype=dtype),
                np.zeros((count, system.broken_dof_count), dtype=dtype),
            ),
            axis=0,
        ),
    )[: system.broken_dof_count]
    preconditioner = _symmetric(averaging @ inverse_top @ np.conjugate(averaging.T))
    global_matrix = np.asarray(system.assemble_matrix())
    preconditioned = preconditioner @ global_matrix
    eigenvalues = np.linalg.eigvals(preconditioned)
    if np.max(np.abs(np.imag(eigenvalues))) > 1.0e-8 * max(
        float(np.max(np.abs(eigenvalues))), 1.0
    ):
        raise ValueError("Prepared substructuring operator has a nonreal spectrum.")
    real_eigenvalues = np.real(eigenvalues)
    minimum = float(np.min(real_eigenvalues))
    if minimum <= 0.0:
        raise ValueError("Prepared substructuring operator is not positive definite.")
    condition = float(np.max(real_eigenvalues) / minimum)
    symmetric_error = float(
        np.linalg.norm(preconditioner - np.conjugate(preconditioner.T), ord=2)
    )
    unity_error = max(
        (float(np.asarray(value.partition_unity_error)) for value in deluxe.interfaces),
        default=0.0,
    )
    adaptive_count = 0 if adaptive is None else adaptive.required_mode_count
    evidence_id = canonical_fingerprint(
        {
            "kind": "substructuring-evidence",
            "system": system.system_id,
            "plan": plan.plan_id,
            "deluxe": deluxe.plan_id,
            "adaptive": None if adaptive is None else adaptive.space_id,
            "constraints": array_tree_fingerprint(constraints),
            "jumps": array_tree_fingerprint(jumps),
        }
    )
    evidence = SubstructuringEvidence(
        plan.method,
        jnp.asarray(condition, dtype=jnp.asarray(global_matrix).real.dtype),
        constraints.shape[0],
        adaptive_count,
        jnp.asarray(unity_error, dtype=jnp.asarray(global_matrix).real.dtype),
        jnp.asarray(symmetric_error, dtype=jnp.asarray(global_matrix).real.dtype),
        jnp.asarray(minimum, dtype=jnp.asarray(global_matrix).real.dtype),
        evidence_id,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-spd-substructuring",
            "system": system.system_id,
            "plan": plan.plan_id,
            "evidence": evidence_id,
        }
    )
    return PreparedSPDSubstructuring(
        system,
        plan.primal,
        deluxe,
        adaptive,
        jnp.asarray(broken),
        jnp.asarray(averaging),
        jnp.asarray(constraints),
        jnp.asarray(jumps),
        jnp.asarray(kkt),
        jnp.asarray(global_matrix),
        jnp.asarray(preconditioner),
        evidence,
        plan.method,
        prepared_id,
    )


__all__ = [
    "AdaptiveInterfaceModes",
    "AdaptiveSpectralCoarseSpace",
    "DeluxeScalingPlan",
    "InexactNewtonEvidence",
    "InexactNewtonTangentPreconditioner",
    "InterfaceDeluxeScaling",
    "PRobustnessEvidence",
    "PreparedSPDSubstructuring",
    "PrimalConstraintPlan",
    "SPDSubstructuringPlan",
    "SubstructuredSPDSystem",
    "SubstructuringEvidence",
]
