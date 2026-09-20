#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed source/target fixed-sector quantum response contracts."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import MatrixFunctionPolicy, ShiftedSolvePolicy
from .lattice import QuantumSectorOperator


class QuantumSectorProbe(StrictModule):
    """One explicit source-to-target operator and its certified charge map."""

    operator: QuantumSectorOperator
    probe_id: str = eqx.field(static=True)
    source_basis_id: str = eqx.field(static=True)
    target_basis_id: str = eqx.field(static=True)
    charge_delta: int = eqx.field(static=True)

    def __init__(self, operator: QuantumSectorOperator, /, *, probe_id: str):
        if not isinstance(operator, QuantumSectorOperator):
            raise TypeError("operator must be QuantumSectorOperator.")
        identifier = str(probe_id)
        if not identifier:
            raise ValueError("probe_id must be non-empty.")
        self.operator = operator
        self.probe_id = identifier
        self.source_basis_id = operator.charge_map.source.basis_id
        self.target_basis_id = operator.charge_map.target.basis_id
        self.charge_delta = operator.charge_map.charge_delta


class ZeroTemperatureResponsePlan(StrictModule):
    """Positive-broadening shifted-Lanczos grid and moment contract."""

    frequencies: Array
    shifted_solve: ShiftedSolvePolicy
    broadening: float = eqx.field(static=True)
    moment_count: int = eqx.field(static=True)
    maximum_frequency_points: int = eqx.field(static=True)
    maximum_result_bytes: int = eqx.field(static=True)
    positivity_tolerance: float = eqx.field(static=True)
    source_tolerance: float = eqx.field(static=True)
    fixed_result_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequencies: ArrayLike,
        broadening: float,
        /,
        *,
        moment_count: int,
        shifted_solve: ShiftedSolvePolicy,
        maximum_frequency_points: int,
        maximum_result_bytes: int,
        positivity_tolerance: float = 1e-10,
        source_tolerance: float = 1e-8,
        required_window: tuple[float, float] | None = None,
    ):
        values = np.asarray(frequencies, dtype=np.float64)
        eta = float(broadening)
        moments = int(moment_count)
        point_limit = int(maximum_frequency_points)
        byte_limit = int(maximum_result_bytes)
        tolerance = float(positivity_tolerance)
        source_tolerance_ = float(source_tolerance)
        if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
            raise ValueError("frequencies must be one finite rank-one grid.")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("frequencies must be strictly increasing.")
        if (
            not isfinite(eta)
            or eta <= 0.0
            or moments < 1
            or point_limit < 1
            or byte_limit < 1
            or not isfinite(tolerance)
            or tolerance < 0.0
            or not isfinite(source_tolerance_)
            or source_tolerance_ < 0.0
        ):
            raise ValueError("Zero-temperature response controls are invalid.")
        if not isinstance(shifted_solve, ShiftedSolvePolicy):
            raise TypeError("shifted_solve must be ShiftedSolvePolicy.")
        if shifted_solve.method != "lanczos":
            raise ValueError("Zero-temperature response requires shifted Lanczos.")
        if shifted_solve.differentiation != "none":
            raise ValueError(
                "Zero-temperature response is a non-differentiable Krylov workflow."
            )
        if any(
            value is None
            for value in (
                shifted_solve.resources.max_matvec_count,
                shifted_solve.resources.max_storage_bytes,
                shifted_solve.resources.max_workspace_bytes,
            )
        ):
            raise ValueError(
                "Zero-temperature response requires explicit shifted-solve resource limits."
            )
        if values.size > point_limit:
            raise ValueError("Frequency grid exceeds maximum_frequency_points.")
        required_bytes = (
            values.size
            * (3 * np.dtype(np.complex128).itemsize + 5 * np.dtype(np.float64).itemsize)
            + moments * np.dtype(np.complex128).itemsize
        )
        if required_bytes > byte_limit:
            raise ValueError("Response result exceeds maximum_result_bytes.")
        if required_window is not None:
            lower, upper = map(float, required_window)
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError("required_window must be finite and increasing.")
            if values[0] > lower or values[-1] < upper:
                raise ValueError("Frequency grid does not cover required_window.")
        self.frequencies = jnp.asarray(values)
        self.shifted_solve = shifted_solve
        self.broadening = eta
        self.moment_count = moments
        self.maximum_frequency_points = point_limit
        self.maximum_result_bytes = byte_limit
        self.positivity_tolerance = tolerance
        self.source_tolerance = source_tolerance_
        self.fixed_result_bytes = required_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "zero-temperature-quantum-response-plan",
                "frequencies": array_tree_fingerprint(values),
                "broadening": eta,
                "moment_count": moments,
                "maximum_frequency_points": point_limit,
                "maximum_result_bytes": byte_limit,
                "positivity_tolerance": tolerance,
                "source_tolerance": source_tolerance_,
                "fixed_result_bytes": required_bytes,
                "shifted": {
                    "method": shifted_solve.method,
                    "execution": shifted_solve.execution,
                    "max_dimension": shifted_solve.max_dimension,
                    "orthogonalization": shifted_solve.orthogonalization,
                    "breakdown_tolerance": shifted_solve.breakdown_tolerance,
                    "relative_tolerance": shifted_solve.relative_tolerance,
                    "absolute_tolerance": shifted_solve.absolute_tolerance,
                    "differentiation": shifted_solve.differentiation,
                    "resources": {
                        "max_matvec_count": shifted_solve.resources.max_matvec_count,
                        "max_storage_bytes": shifted_solve.resources.max_storage_bytes,
                        "max_workspace_bytes": shifted_solve.resources.max_workspace_bytes,
                    },
                },
            }
        )


class FiniteTemperatureResponsePlan(StrictModule):
    """Symmetric time/frequency grids for TPQ correlation and KMS evidence."""

    times: Array
    frequencies: Array
    window: Array
    matrix_function: MatrixFunctionPolicy
    moment_count: int = eqx.field(static=True)
    maximum_result_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    positivity_tolerance: float = eqx.field(static=True)
    kms_tolerance: float = eqx.field(static=True)
    fixed_result_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        frequencies: ArrayLike,
        window: ArrayLike,
        /,
        *,
        moment_count: int,
        matrix_function: MatrixFunctionPolicy,
        maximum_result_bytes: int,
        maximum_workspace_bytes: int,
        positivity_tolerance: float = 1e-8,
        kms_tolerance: float = 1e-2,
    ):
        times_ = np.asarray(times, dtype=np.float64)
        frequencies_ = np.asarray(frequencies, dtype=np.float64)
        window_ = np.asarray(window, dtype=np.float64)
        moments = int(moment_count)
        byte_limit = int(maximum_result_bytes)
        workspace_limit = int(maximum_workspace_bytes)
        positivity = float(positivity_tolerance)
        kms = float(kms_tolerance)
        if (
            times_.ndim != 1
            or times_.size < 3
            or times_.size % 2 == 0
            or frequencies_.ndim != 1
            or frequencies_.size < 3
            or frequencies_.size % 2 == 0
            or window_.shape != times_.shape
            or np.any(~np.isfinite(times_))
            or np.any(~np.isfinite(frequencies_))
            or np.any(~np.isfinite(window_))
        ):
            raise ValueError(
                "Finite-temperature grids/window have invalid shapes or values."
            )
        if np.any(np.diff(times_) <= 0.0) or np.any(np.diff(frequencies_) <= 0.0):
            raise ValueError("Time and frequency grids must be strictly increasing.")
        if not np.allclose(times_, -times_[::-1], rtol=0.0, atol=1e-13):
            raise ValueError("times must be exactly symmetric about zero.")
        if not np.allclose(frequencies_, -frequencies_[::-1], rtol=0.0, atol=1e-13):
            raise ValueError("frequencies must be exactly symmetric about zero.")
        if np.any(window_ < 0.0) or not np.allclose(
            window_, window_[::-1], rtol=0.0, atol=1e-13
        ):
            raise ValueError("window must be non-negative and symmetric.")
        if (
            moments < 1
            or byte_limit < 1
            or workspace_limit < 1
            or not isfinite(positivity)
            or positivity < 0.0
            or not isfinite(kms)
            or kms < 0.0
        ):
            raise ValueError("Finite-temperature response controls are invalid.")
        if not isinstance(matrix_function, MatrixFunctionPolicy):
            raise TypeError("matrix_function must be MatrixFunctionPolicy.")
        if matrix_function.method != "lanczos":
            raise ValueError("Finite-temperature response requires Lanczos propagation.")
        if matrix_function.differentiation.mode != "none":
            raise ValueError(
                "Finite-temperature response requires a stopped Krylov AD boundary."
            )
        required = (
            times_.size * 5 * np.dtype(np.complex128).itemsize
            + frequencies_.size * 4 * np.dtype(np.float64).itemsize
            + moments * 2 * np.dtype(np.complex128).itemsize
        )
        if required > byte_limit:
            raise ValueError("Finite-temperature response exceeds maximum_result_bytes.")
        self.times = jnp.asarray(times_)
        self.frequencies = jnp.asarray(frequencies_)
        self.window = jnp.asarray(window_)
        self.matrix_function = matrix_function
        self.moment_count = moments
        self.maximum_result_bytes = byte_limit
        self.maximum_workspace_bytes = workspace_limit
        self.fixed_result_bytes = required
        self.positivity_tolerance = positivity
        self.kms_tolerance = kms
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-temperature-quantum-response-plan",
                "times": array_tree_fingerprint(times_),
                "frequencies": array_tree_fingerprint(frequencies_),
                "window": array_tree_fingerprint(window_),
                "moment_count": moments,
                "maximum_workspace_bytes": workspace_limit,
                "maximum_result_bytes": byte_limit,
                "positivity_tolerance": positivity,
                "kms_tolerance": kms,
                "matrix_function": {
                    "method": matrix_function.method,
                    "max_dimension": matrix_function.max_dimension,
                    "error_tolerance": matrix_function.error_tolerance,
                    "orthogonalization": matrix_function.orthogonalization,
                    "differentiation": matrix_function.differentiation.mode,
                },
                "fixed_result_bytes": required,
            }
        )


__all__ = [
    "FiniteTemperatureResponsePlan",
    "QuantumSectorProbe",
    "ZeroTemperatureResponsePlan",
]
