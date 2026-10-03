# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Learned metric correction trained through the implicit conservation adjoint.

A log-linear edge-feature model proposes candidate metric weights
``w_c = w0 exp(F theta)``. ``MeshfreeMetricCorrectionPlan.project`` maps the
candidate onto the full moment set (the weighted minimum-norm projection), and
the projected weights are the runtime metric of a ``MonotoneEdgeConductance``
conservation solve. ``phydrax.solver.train_components`` fits ``theta`` through
the native implicit derivative of that solve and the native ``rhs-only``
derivative of the projection.

The reference is the discrete state of the same solve at projected truth
coefficients: a discrete identification problem on one cloud, not a continuum
convergence claim. A case is accepted only when the projection succeeds with
the declared sign margin and the conservation root is accepted; failed cases
are rejected by the native training kernel, never stepped on.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, final, Literal, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax import Array

import phydrax as phx
from phydrax.discretization.meshfree import (
    MeshfreeConservationProblem,
    MeshfreeExteriorCalculusPlan,
    MeshfreeMetricCorrectionPlan,
    MeshfreeMetricCorrectionPolicy,
    MonotoneEdgeConductance,
    prepare_meshfree_conservation_solve,
    PreparedMeshfreeConservationSolve,
    PreparedMeshfreeExteriorCalculus,
)
from phydrax.nn.models import InputConvexNetwork
from phydrax.nonlinear import NonlinearTermination
from phydrax.solver.coupling import ParameterBinding, RuntimeInput
from phydrax.typing import Dim, Float64


class TrainingReport(TypedDict):
    learned_training_loss_before: float
    learned_training_loss_after: float
    learned_training_accepted_updates: int
    learned_training_coefficients: str
    learned_training_status: str
    learned_training_max_moment_residual: float
    learned_training_primal_successful: bool
    learned_training_adjoint_successful: bool
    learned_gradient: float
    learned_gradient_fd: float
    learned_gradient_fd_relative_error: float
    learned_failed_case_loss_finite: bool
    learned_failed_step_rejected: bool


_PARAMETER = "corrected-metric"
_TERMINATION = NonlinearTermination(
    absolute_residual=1e-11, relative_residual=1e-11, maximum_steps=64
)


class _EdgeDim(Dim):
    """Compact canonical edges of the training cloud."""


class _FeatureDim(Dim):
    """Edge features of the learned candidate model."""


class _NodeDim(Dim):
    """Compact vertices of the training cloud."""


def _candidate_port(edges: int) -> phx.ValuePort:
    return phx.ValuePort(
        "meshfree-learned-metric-candidate",
        event_shape=(edges,),
        component_ids=tuple(f"edge-{index}" for index in range(edges)),
        representation="edge-metric-candidate",
    )


def _metric_port(edges: int) -> phx.ValuePort:
    return phx.ValuePort(
        "meshfree-corrected-metric",
        event_shape=(edges,),
        component_ids=tuple(f"edge-{index}" for index in range(edges)),
        representation="positive-edge-metric",
    )


@final
class EdgeMetricCandidate(phx.AbstractArrayModel):
    """Learned log-linear edge-feature model of candidate metric weights.

    Only ``coefficients`` is trained; the base metric and the edge features are
    fixed external data of the cloud.
    """

    __strict_contract__ = True
    coefficients: Float64[_FeatureDim]
    features: Float64[_EdgeDim, _FeatureDim] = phx.fixed_field()
    base: Float64[_EdgeDim] = phx.fixed_field()
    in_size: Literal["scalar"] = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, coefficients: Array, features: Array, base: Array, /) -> None:
        coefficients_ = jnp.asarray(coefficients, dtype=jnp.float64)
        features_ = jnp.asarray(features, dtype=jnp.float64)
        base_ = jnp.asarray(base, dtype=jnp.float64)
        if features_.ndim != 2 or features_.shape != (base_.size, coefficients_.size):
            raise ValueError("Edge features must hold one row per edge and coefficient.")
        self.coefficients = coefficients_
        self.features = features_
        self.base = base_
        self.in_size = "scalar"
        self.out_size = base_.size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        return self.base * jnp.exp(self.features @ self.coefficients)

    def model_ports(self) -> phx.ModelPorts:
        return phx.ModelPorts(inputs=(), outputs=(_candidate_port(self.out_size),))

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("deterministic"),
            ports=self.model_ports(),
        )


@final
class LearnedMetricTraining(phx.StrictModule):
    """Prepared projection, conservation solve, model data and reference state."""

    __strict_contract__ = True
    plan: MeshfreeMetricCorrectionPlan
    prepared: PreparedMeshfreeConservationSolve
    features: Float64[_EdgeDim, _FeatureDim]
    base: Float64[_EdgeDim]
    reference: Float64[_NodeDim]
    ports: phx.ModelPorts

    def model(self, coefficients: Array, /) -> EdgeMetricCandidate:
        return EdgeMetricCandidate(coefficients, self.features, self.base)


@final
class BoundLearnedMetricTraining(phx.StrictModule):
    training: LearnedMetricTraining
    component: phx.ComponentBinding


def bind(
    training: LearnedMetricTraining, component: phx.ComponentBinding
) -> BoundLearnedMetricTraining:
    return BoundLearnedMetricTraining(training, component)


def measure(owner: BoundLearnedMetricTraining, case: None) -> phx.solver.SolverCaseResult:
    """Candidate -> full-moment projection -> conservation solve -> state misfit."""
    del case
    training = owner.training
    candidate = owner.component.model(jnp.zeros((), dtype=jnp.float64))
    projection = training.plan.project(candidate)
    solved = training.prepared.solve(parameters={_PARAMETER: projection.weights})
    free = training.prepared.residual.free_indices
    return phx.solver.SolverCaseResult(
        residual=(solved.state - training.reference)[free],
        accepted=projection.admissible & solved.accepted,
        aux=(projection.linear.status, solved.primal_status),
    )


def candidate_component(
    model: EdgeMetricCandidate, training: LearnedMetricTraining
) -> phx.ComponentBinding:
    model_port = model.model_ports().outputs[0]
    owner_port = training.ports.outputs[0]
    return phx.bind_component(
        model,
        phx.ComponentAuthority.MODEL,
        owner_ports=training.ports,
        port_mapping=phx.PortMapping(outputs=((model_port.port_id, owner_port.port_id),)),
    )


def objective(training: LearnedMetricTraining) -> phx.solver.SolverObjective:
    return phx.solver.SolverObjective(
        training,
        bind,
        measure,
        objective_id="meshfree-learned-metric-correction",
        accepted_results="reject-attempt",
    )


def training_lattice(size: int = 6) -> PreparedMeshfreeExteriorCalculus:
    """Regular lattice with axial and diagonal edges and a positive admitted metric."""
    spacing = 1.0 / (size - 1)
    points = (
        np.stack(
            np.meshgrid(np.arange(size), np.arange(size), indexing="ij"), axis=-1
        ).reshape((-1, 2))
        * spacing
    )
    interior = np.all((points > 1e-9) & (points < 1.0 - 1e-9), axis=1)
    exterior = MeshfreeExteriorCalculusPlan(
        points,
        1.5 * spacing,
        8 * size * size,
        node_volumes=np.full(size * size, spacing * spacing),
        dirichlet=~interior,
    ).prepare()
    if not bool(exterior.metric_result.hilbert_admitted):
        raise RuntimeError("The training lattice must carry a positive admitted metric.")
    return exterior


def edge_features(exterior: PreparedMeshfreeExteriorCalculus) -> np.ndarray:
    """Diagonal-versus-axial indicator and a smooth midpoint modulation."""
    points = np.asarray(exterior.points)
    pairs = np.asarray(exterior.pairs)
    offsets = points[pairs[:, 1]] - points[pairs[:, 0]]
    diagonal = np.all(np.abs(offsets) > 1e-9, axis=1).astype(np.float64) - 0.5
    midpoint = 0.5 * (points[pairs[:, 0]] + points[pairs[:, 1]])
    modulation = np.sin(np.pi * midpoint[:, 0]) * np.cos(np.pi * midpoint[:, 1])
    return np.stack((diagonal, modulation), axis=1)


def prepare_training(
    *,
    size: int = 6,
    truth: tuple[float, float] = (0.6, 0.3),
    margin_fraction: float = 0.2,
) -> LearnedMetricTraining:
    """Prepare the actual consumer problem separately from training and execution."""
    exterior = training_lattice(size)
    base = exterior.metric_result.weights
    plan = MeshfreeMetricCorrectionPlan(
        exterior,
        policy=MeshfreeMetricCorrectionPolicy(
            margin=margin_fraction * float(jnp.min(base)), tolerance=1e-9
        ),
    )
    features = jnp.asarray(edge_features(exterior))
    edges = base.size
    ports = phx.ModelPorts(inputs=(), outputs=(_candidate_port(edges),))
    binding = ParameterBinding(
        _PARAMETER,
        _metric_port(edges),
        targets=(RuntimeInput("meshfree", "metric_weights"),),
        role="coefficient",
        derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
    )
    points = np.asarray(exterior.points)
    law = MonotoneEdgeConductance(
        InputConvexNetwork(in_size="scalar", width_size=2, depth=1, key=jr.key(3)),
        background_conductance=0.5,
    )
    problem = MeshfreeConservationProblem(
        exterior,
        law,
        source=1.0 + points[:, 0],
        boundary_values=np.sin(np.pi * points[:, 1]) + points[:, 0],
        parameter_bindings=(binding,),
        problem_id="learned-metric-correction",
    )
    prepared = prepare_meshfree_conservation_solve(
        problem, parameters={_PARAMETER: base}, termination=_TERMINATION
    )
    truth_projection = plan.project(
        EdgeMetricCandidate(jnp.asarray(truth, dtype=jnp.float64), features, base)(0.0)
    )
    if not bool(truth_projection.admissible):
        raise RuntimeError("The truth coefficients must project to an admissible metric.")
    solved = prepared.solve(parameters={_PARAMETER: truth_projection.weights})
    if not bool(solved.accepted):
        raise RuntimeError("The reference conservation solve did not converge.")
    return LearnedMetricTraining(plan, prepared, features, base, solved.state, ports)


def training_loss(
    training: LearnedMetricTraining,
    target: phx.solver.SolverObjective,
    coefficients: Array,
    /,
) -> Array:
    """Native objective value; nonfinite whenever a case is not accepted."""
    return target.evaluate(
        candidate_component(training.model(coefficients), training)
    ).value


def train(
    training: LearnedMetricTraining,
    target: phx.solver.SolverObjective,
    coefficients: Array,
    /,
    *,
    optimizer: optax.GradientTransformation,
    steps: int,
    seed: int = 0,
    rejection_budget: int = 0,
) -> tuple[phx.solver.ComponentTrainingResult, Array]:
    """Native component training; returns the result and committed coefficients."""
    result = phx.solver.train_components(
        candidate_component(training.model(coefficients), training),
        (target,),
        optimizer=optimizer,
        steps=steps,
        key=jr.key(seed),
        rejection_budget=rejection_budget,
    )
    tree = result.tree
    if not isinstance(tree, phx.ComponentBinding) or not isinstance(
        tree.model, EdgeMetricCandidate
    ):
        raise TypeError("The native component training result changed its model family.")
    return result, tree.model.coefficients


def run_training(*, size: int = 6, seed: int = 0, steps: int = 6) -> TrainingReport:
    """Gradient check, native training, and rejection of a failed attempt."""
    with jax.enable_x64(True):
        training = prepare_training(size=size)
        target = objective(training)
        point = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
        direction = jnp.asarray([0.8, 0.6], dtype=jnp.float64)
        gradient = float(
            jnp.vdot(
                jax.grad(lambda c: training_loss(training, target, c))(point), direction
            )
        )
        step = 1e-5
        central = float(
            (
                training_loss(training, target, point + step * direction)
                - training_loss(training, target, point - step * direction)
            )
            / (2 * step)
        )
        initial = jnp.zeros((2,), dtype=jnp.float64)
        before = float(training_loss(training, target, initial))
        trained, coefficients = train(
            training, target, initial, optimizer=optax.lbfgs(), steps=steps, seed=seed
        )
        after = float(training_loss(training, target, coefficients))
        candidate = training.model(coefficients)(0.0)
        corrected = training.plan.correct(candidate)
        weights = training.plan.project(candidate).weights
        solution = training.prepared.solve(parameters={_PARAMETER: weights})
        adjoint = training.prepared.adjoint(
            solution,
            (solution.state - training.reference)[
                training.prepared.residual.free_indices
            ],
            parameters={_PARAMETER: weights},
        )
        # A candidate whose projection breaks the sign margin: the native kernel
        # must reject the attempt and leave the committed coefficients unchanged.
        failing = jnp.asarray([6.0, 0.0], dtype=jnp.float64)
        failed_value = float(training_loss(training, target, failing))
        rejected, kept = train(
            training,
            target,
            failing,
            optimizer=optax.sgd(1e-3),
            steps=1,
            seed=seed,
            rejection_budget=1,
        )
        return {
            "learned_training_loss_before": before,
            "learned_training_loss_after": after,
            "learned_training_accepted_updates": trained.accepted_updates,
            "learned_training_coefficients": str(np.asarray(coefficients).tolist()),
            "learned_training_status": corrected.status.name,
            "learned_training_max_moment_residual": (
                corrected.evidence.maximum_moment_residual
            ),
            "learned_training_primal_successful": bool(solution.accepted),
            "learned_training_adjoint_successful": bool(adjoint.accepted),
            "learned_gradient": gradient,
            "learned_gradient_fd": central,
            "learned_gradient_fd_relative_error": abs(gradient - central)
            / max(abs(central), 1e-14),
            "learned_failed_case_loss_finite": bool(np.isfinite(failed_value)),
            "learned_failed_step_rejected": rejected.accepted_updates == 0
            and rejected.nonfinite_rejections == 1
            and bool(jnp.all(kept == failing)),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6)
    arguments = parser.parse_args()
    metrics = run_training(
        size=arguments.size, seed=arguments.seed, steps=arguments.steps
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
