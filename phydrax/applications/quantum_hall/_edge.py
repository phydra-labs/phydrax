#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite Hall ribbons with explicit edge localization evidence."""

from __future__ import annotations

from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    DenseEigh,
    eigensolve,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
)
from ...operators.periodic import (
    PeriodicFiniteBoundaryPlan,
    PeriodicFiniteOrbitalPlan,
    PreparedPeriodicOrbitalPencil,
)


class HallRibbonSpectrumResult(StrictModule, NonTrainableState):
    twists: Array
    energies: Array
    eigenvectors: Array
    left_edge_weights: Array
    right_edge_weights: Array
    bulk_weights: Array
    eigenpair_residuals: Array
    successful: Array
    open_axis: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    pencil_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class HallRibbonPlan(StrictModule, NonTrainableState):
    pencil: PreparedPeriodicOrbitalPencil
    width: int = eqx.field(static=True)
    open_axis: int = eqx.field(static=True)
    twist_count: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pencil: PreparedPeriodicOrbitalPencil,
        width: int,
        /,
        *,
        open_axis: int = 0,
        twist_count: int = 81,
        residual_tolerance: float = 1.0e-9,
    ):
        if not isinstance(pencil, PreparedPeriodicOrbitalPencil):
            raise TypeError("pencil must be PreparedPeriodicOrbitalPencil.")
        width_ = int(width)
        axis = int(open_axis)
        count = int(twist_count)
        tolerance = float(residual_tolerance)
        if (
            pencil.plan.basis.cell.rank != 2
            or width_ < 2
            or axis not in (0, 1)
            or count < 3
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Hall ribbon width, axis, twist count, or tolerance is invalid."
            )
        self.pencil = pencil
        self.width = width_
        self.open_axis = axis
        self.twist_count = count
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hall-ribbon-plan",
                "pencil": pencil.prepared_id,
                "width": width_,
                "open_axis": axis,
                "twist_count": count,
                "residual_tolerance": tolerance,
            }
        )

    def evaluate(self, /) -> HallRibbonSpectrumResult:
        periodic_axis = 1 - self.open_axis
        shape = [1, 1]
        shape[self.open_axis] = self.width
        twists = np.linspace(-pi, pi, self.twist_count, endpoint=False)
        energies = []
        vectors = []
        left_weights = []
        right_weights = []
        residuals = []
        successes = []
        properties = OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        )
        metric_properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        for twist in twists:
            axes = tuple(index != self.open_axis for index in range(2))
            twist_values = [0.0, 0.0]
            twist_values[periodic_axis] = float(twist)
            kind = "slab" if twist == 0.0 else "twisted"
            boundary = PeriodicFiniteBoundaryPlan(
                tuple(shape),
                axes,
                tuple(twist_values),
                kind,
            )
            realization = PeriodicFiniteOrbitalPlan(self.pencil, boundary).realize()
            hamiltonian = realization.hamiltonian.to_dense()
            overlap = realization.overlap.to_dense()
            dimension = hamiltonian.shape[0]
            solved = eigensolve(
                GeneralizedEigenproblem(
                    DenseLinearOperator(hamiltonian, properties=properties),
                    DenseLinearOperator(overlap, properties=metric_properties),
                    problem_id=f"{self.plan_id}:{twist:.17g}",
                ),
                policy=EigenSolvePolicy(
                    DenseEigh(),
                    count=dimension,
                    which="smallest-algebraic",
                ),
            )
            value = solved.eigenvalues.real
            vector = solved.eigenvectors
            action = hamiltonian @ vector - overlap @ (vector * value[None, :])
            residual = jnp.sqrt(jnp.sum(jnp.abs(action) ** 2, axis=0))
            coordinates = np.asarray(realization.cell_coordinates)
            orbital_count = self.pencil.plan.basis.orbital_count
            layers = np.repeat(coordinates[:, self.open_axis], orbital_count)
            probabilities = jnp.abs(vector) ** 2
            left = jnp.sum(probabilities[jnp.asarray(layers == 0)], axis=0)
            right = jnp.sum(probabilities[jnp.asarray(layers == self.width - 1)], axis=0)
            energies.append(value)
            vectors.append(vector)
            left_weights.append(left)
            right_weights.append(right)
            residuals.append(residual)
            successes.append(
                jnp.all(solved.successful) & jnp.all(residual <= self.residual_tolerance)
            )
        energy_array = jnp.stack(energies)
        vector_array = jnp.stack(vectors)
        left_array = jnp.stack(left_weights)
        right_array = jnp.stack(right_weights)
        residual_array = jnp.stack(residuals)
        bulk_array = jnp.maximum(0.0, 1.0 - left_array - right_array)
        successful = jnp.all(jnp.stack(successes))
        result_id = canonical_fingerprint(
            {
                "kind": "hall-ribbon-spectrum-result",
                "plan": self.plan_id,
                "arrays": array_tree_fingerprint(
                    {
                        "twists": twists,
                        "energies": np.asarray(energy_array),
                        "left": np.asarray(left_array),
                        "right": np.asarray(right_array),
                    }
                ),
            }
        )
        return HallRibbonSpectrumResult(
            jnp.asarray(twists),
            energy_array,
            vector_array,
            left_array,
            right_array,
            bulk_array,
            residual_array,
            successful,
            self.open_axis,
            self.width,
            self.pencil.prepared_id,
            result_id,
        )


__all__ = ["HallRibbonPlan", "HallRibbonSpectrumResult"]
