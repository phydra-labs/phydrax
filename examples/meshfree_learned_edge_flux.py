# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Recover a conditioned constitutive edge law through an implicit conservative PDE.

The reference is a manufactured *discrete* conservation solution on an actual
1D/2D/3D cloud, not a claim of continuum convergence. Host endpoint algebra,
analytical sigmoid derivatives, and an analytical nodal field supply independent
reference data. No detached solve or differentiation through Newton iterations is
used. Every model value enters native ParameterBinding with MODEL authority.
"""

from __future__ import annotations

import argparse
from itertools import product
from math import isfinite
from typing import Any, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax import Array

import phydrax as phx
from benchmarks._runtime import logical_array_bytes
from phydrax.discretization.meshfree._conservation_solve import (
    MeshfreeConservationProblem,
    prepare_meshfree_conservation_solve,
    PreparedMeshfreeConservationSolve,
)
from phydrax.discretization.meshfree._constitutive import (
    EdgeFeatureField,
    EdgeFrameFeatures,
    MonotoneEdgeConductance,
)
from phydrax.discretization.meshfree._coverage import EdgeFeatureCoverage
from phydrax.discretization.meshfree._exterior import MeshfreeExteriorCalculusPlan
from phydrax.discretization.meshfree._exterior_metric import MeshfreeMetricPolicy
from phydrax.nn.models import PartiallyInputConvexNetwork
from phydrax.solver.coupling import ParameterBinding, RuntimeInput
from phydrax.typing import Dim, Float64, Scalar
from phydrax.units import DIMENSIONLESS


WorkflowMetric: TypeAlias = float | int | bool | str


class _RecoveryNodeDim(Dim):
    """Physical vertices of one independent recovery cloud."""


def _strength_port() -> phx.ValuePort:
    return phx.ValuePort(
        "meshfree-constitutive-strength",
        event_shape=(),
        component_ids=("strength",),
        representation="positive-scalar",
        dimensions=(DIMENSIONLESS,),
    )


@final
class ConstitutiveScale(phx.AbstractArrayModel):
    """Positive trainable strength of the same certified edge law at every node."""

    __strict_contract__ = True
    log_scale: Float64[Scalar]
    in_size: Literal["scalar"] = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(self, scale: float) -> None:
        if not isfinite(scale) or scale <= 0:
            raise ValueError("Constitutive scale must be finite and positive.")
        value = jnp.log(jnp.asarray(scale, dtype=jnp.float64))
        self.log_scale = value
        self.in_size = "scalar"
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        return jnp.exp(self.log_scale)

    def model_ports(self) -> phx.ModelPorts:
        return phx.ModelPorts(inputs=(), outputs=(_strength_port(),))

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
class LearnedFluxRecovery(phx.StrictModule):
    __strict_contract__ = True
    prepared: PreparedMeshfreeConservationSolve
    reference: Float64[_RecoveryNodeDim]
    truth_scale: float = eqx.field(static=True)
    ports: phx.ModelPorts


@final
class BoundLearnedFluxRecovery(phx.StrictModule):
    recovery: LearnedFluxRecovery
    component: phx.ComponentBinding


def bind(
    recovery: LearnedFluxRecovery, component: phx.ComponentBinding
) -> BoundLearnedFluxRecovery:
    return BoundLearnedFluxRecovery(recovery, component)


def measure(owner: BoundLearnedFluxRecovery, case: None) -> phx.solver.SolverCaseResult:
    del case
    solved = owner.recovery.prepared.solve(
        parameters={"constitutive-strength": owner.component}
    )
    indices = owner.recovery.prepared.residual.free_indices
    return phx.solver.SolverCaseResult(
        residual=(solved.state - owner.recovery.reference)[indices],
        accepted=solved.accepted,
        aux=(solved.primal_status, solved.ledger.balance_defect),
    )


def _lattice(
    size: int, dimension: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exactly N lattice nodes; incomplete axial neighborhoods are prescribed."""
    if dimension not in (1, 2, 3):
        raise ValueError(
            "Learned edge recovery supports actual ambient dimensions 1, 2, and 3."
        )
    if isinstance(size, bool) or not isinstance(size, int) or size < 2 * dimension + 1:
        raise ValueError(
            "The total cloud capacity must contain a center and all +/- coordinate neighbors."
        )
    radius = 1
    while (2 * radius + 1) ** dimension < size:
        radius += 1
    offsets = sorted(
        product(range(-radius, radius + 1), repeat=dimension),
        key=lambda coordinate: (sum(value * value for value in coordinate), coordinate),
    )[:size]
    nodes = set(offsets)
    axes = np.eye(dimension, dtype=int)
    boundary = np.asarray(
        [
            any(
                tuple(np.asarray(node) + sign * axis) not in nodes
                for axis in axes
                for sign in (-1, 1)
            )
            for node in offsets
        ]
    )
    spacing = 1.0 / (2 * radius + 2)
    coordinates = np.asarray(offsets, dtype=np.float64) * spacing
    # Translation is harmless but makes an unseen cloud a distinct geometry.
    coordinates += np.random.default_rng(seed).uniform(-0.05, 0.05, size=(dimension,))
    return coordinates, boundary, spacing


def _potential(seed: int) -> PartiallyInputConvexNetwork:
    """Canonical certificate, exactly phi(D;c)=softplus(D+tanh(c))."""
    potential = PartiallyInputConvexNetwork(
        context_size=1, convex_size="scalar", width_size=1, depth=1, key=jr.key(seed)
    )
    positive_raw = jnp.log(jnp.expm1(jnp.asarray(1.0, dtype=jnp.float64)))
    return eqx.tree_at(
        lambda network: (
            network.context_lift.weight,
            network.context_lift.bias,
            network.convex_input_layers[0].weight,
            network.convex_input_layers[0].bias,
            network.convex_input_layers[1].weight,
            network.convex_input_layers[1].bias,
            network.context_layers[0].weight,
            network.context_layers[1].weight,
            network.state_layers[0].weight,
        ),
        potential,
        (
            jnp.ones((1, 1)),
            jnp.zeros((1,)),
            jnp.ones((1, 1)),
            jnp.zeros((1,)),
            jnp.zeros((1, 1)),
            jnp.zeros((1,)),
            jnp.ones((1, 1)),
            jnp.zeros((1, 1)),
            positive_raw.reshape((1, 1)),
        ),
    )


def _manufactured(points: np.ndarray) -> np.ndarray:
    weights = np.arange(1, points.shape[1] + 1, dtype=np.float64)
    return np.sum(weights[None, :] * (points * points + 0.2 * np.sin(points)), axis=1)


def prepare_recovery(
    *, size: int = 12, dimension: int = 2, seed: int = 0, truth_scale: float = 1.7
) -> LearnedFluxRecovery:
    """Prepare the actual consumer problem separately from training and execution."""
    if not isfinite(truth_scale) or truth_scale <= 0:
        raise ValueError("Truth scale must be finite and positive.")
    points, boundary, spacing = _lattice(size, dimension, seed)
    capacity = dimension * size
    volumes = np.full(size, 1.0 / size)
    exterior = MeshfreeExteriorCalculusPlan(
        points,
        1.01 * spacing,
        capacity,
        node_volumes=volumes,
        dirichlet=boundary,
        edge_prior=np.full(capacity, 1.0 / (size * spacing * spacing)),
        metric_policy=MeshfreeMetricPolicy(sign="signed"),
    ).prepare()
    if not bool(exterior.metric_result.hilbert_admitted):
        raise RuntimeError(
            "The manufactured lattice did not produce a positive admitted exterior metric."
        )
    compact = np.asarray(exterior.points)
    pairs = np.asarray(exterior.pairs)
    material = np.sin(compact @ np.arange(1, dimension + 1, dtype=np.float64))
    features = EdgeFrameFeatures(
        exterior.points,
        exterior.pairs,
        (EdgeFeatureField("external-material", "scalar", dimension=DIMENSIONLESS),),
        (material,),
        source_id=exterior.incidence.source.space_id,
    )
    law = MonotoneEdgeConductance(_potential(seed), odd_size=1)
    # This sweep is declared training material data, fixed before unseen geometry.
    coverage = EdgeFeatureCoverage.fit(
        np.linspace(-1.0, 1.0, 65)[:, None],
        training_domain="train-only-external-material-sweep[-1,1]",
        feature_names=features.even_names,
    )
    reference = _manufactured(compact)
    difference = reference[pairs[:, 1]] - reference[pairs[:, 0]]
    context = np.tanh(0.5 * (material[pairs[:, 0]] + material[pairs[:, 1]]))
    sigmoid_forward = 1.0 / (1.0 + np.exp(-(difference + context)))
    sigmoid_reverse = 1.0 / (1.0 + np.exp(-(-difference + context)))
    flux = (
        truth_scale
        * np.asarray(exterior.metric_result.weights)
        * (difference + 0.5 * (sigmoid_forward - sigmoid_reverse))
    )
    integrated = np.zeros(size)
    np.add.at(integrated, pairs[:, 0], -flux)
    np.add.at(integrated, pairs[:, 1], flux)
    source = integrated / np.asarray(exterior.node_volumes)
    port = _strength_port()
    ports = phx.ModelPorts(inputs=(), outputs=(port,))
    parameter = ParameterBinding(
        "constitutive-strength",
        port,
        targets=(RuntimeInput("meshfree", "conductance"),),
        role="coefficient",
        derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
    )
    problem = MeshfreeConservationProblem(
        exterior,
        law,
        features=features,
        source=source,
        boundary_values=reference,
        coverage=coverage,
        parameter_bindings=(parameter,),
        problem_id=f"learned-edge-recovery:{dimension}D:{seed}",
    )
    prepared = prepare_meshfree_conservation_solve(
        problem,
        parameters={"constitutive-strength": jnp.asarray(1.0)},
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-11, relative_residual=1e-11, maximum_steps=64
        ),
    )
    return LearnedFluxRecovery(prepared, jnp.asarray(reference), truth_scale, ports)


def _component(
    model: ConstitutiveScale, recovery: LearnedFluxRecovery
) -> phx.ComponentBinding:
    model_port = model.model_ports().outputs[0]
    owner_port = recovery.ports.outputs[0]
    mapping = phx.PortMapping(outputs=((model_port.port_id, owner_port.port_id),))
    return phx.bind_component(
        model,
        phx.ComponentAuthority.MODEL,
        owner_ports=recovery.ports,
        port_mapping=mapping,
    )


def run_workflow(
    *, size: int = 12, dimension: int = 2, seed: int = 0, steps: int = 80
) -> dict[str, WorkflowMetric]:
    """Train, check the implicit derivative, and predict an independent unseen cloud."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Training steps must be a positive integer.")
    with jax.enable_x64(True):
        recovery = prepare_recovery(size=size, dimension=dimension, seed=seed)
        objective = phx.solver.SolverObjective(
            recovery,
            bind,
            measure,
            objective_id="meshfree-learned-constitutive-PDE",
            accepted_results="reject-attempt",
        )
        initial = _component(ConstitutiveScale(0.8), recovery)
        trained = phx.solver.train_components(
            initial, (objective,), optimizer=optax.lbfgs(), steps=steps, key=jr.key(seed)
        )
        if not isinstance(trained.tree, phx.ComponentBinding) or not isinstance(
            trained.tree.model, ConstitutiveScale
        ):
            raise TypeError(
                "The native component training result changed its model family."
            )
        scale = float(jnp.exp(trained.tree.model.log_scale))

        def loss(log_scale: Array) -> Array:
            model = eqx.tree_at(
                lambda component: component.log_scale, ConstitutiveScale(1.0), log_scale
            )
            return objective.evaluate(_component(model, recovery)).value

        point = jnp.log(jnp.asarray(1.15))
        gradient = float(jax.jit(jax.grad(loss))(point))
        step = 1e-5
        central = float((loss(point + step) - loss(point - step)) / (2 * step))
        relative_error = abs(gradient - central) / max(abs(central), 1e-12)
        solution = recovery.prepared.solve(
            parameters={"constitutive-strength": trained.tree}
        )
        adjoint = recovery.prepared.adjoint(
            solution,
            solution.state - recovery.reference,
            parameters={"constitutive-strength": trained.tree},
        )
        unseen = prepare_recovery(
            size=size + 2 * dimension,
            dimension=dimension,
            seed=seed + 1,
            truth_scale=recovery.truth_scale,
        )
        unseen_component = _component(trained.tree.model, unseen)
        prediction = unseen.prepared.solve(
            parameters={"constitutive-strength": unseen_component}
        )
        unseen_error = float(jnp.max(jnp.abs(prediction.state - unseen.reference)))
        # Deliberate unsuccessful forward: the accepted-result owner must reject
        # this value, not report a plausible finite loss from the initial guess.
        refused = recovery.prepared.solve(
            parameters={"constitutive-strength": jnp.asarray(-1.0)}, implicit=False
        )
        failed_case = phx.solver.SolverCaseResult(
            residual=refused.state - recovery.reference, accepted=refused.accepted
        )
        failure_rejected = (not bool(refused.accepted)) and not bool(
            jnp.isfinite(failed_case.loss)
        )
        coverage = recovery.prepared.problem.coverage
        if coverage is None or solution.coverage is None:
            raise RuntimeError("This workflow requires actual fitted coverage evidence.")
        outside = coverage.assess(jnp.asarray([[2.0]]))
        metrics: dict[str, WorkflowMetric] = {
            "dimension": dimension,
            "node_count": size,
            "equation_count": int(recovery.prepared.residual.free_indices.size),
            "learned_coefficient": scale,
            "truth_coefficient": recovery.truth_scale,
            "parameter_error": abs(scale - recovery.truth_scale),
            "implicit_gradient": gradient,
            "finite_difference_gradient": central,
            "gradient_relative_error": relative_error,
            "unseen_cloud_error": unseen_error,
            "primal_successful": bool(solution.accepted),
            "adjoint_status": int(adjoint.adjoint_status),
            "adjoint_successful": bool(adjoint.accepted),
            "unseen_primal_successful": bool(prediction.accepted),
            "conservation_defect": float(jnp.abs(solution.ledger.balance_defect)),
            "unseen_node_count": int(unseen.reference.size),
            "precision": "float64",
            "failure_rejected": failure_rejected,
            "failed_primal_status": int(refused.primal_status),
            "coverage_refused": not bool(outside.admitted),
            "coverage_status": int(jnp.max(solution.coverage.status)),
            "outside_coverage_status": int(outside.status[0]),
            "training_domain": coverage.training_domain,
            "coercivity_certified": bool(
                solution.constitutive_evidence.coercivity_certified
            ),
            "contraction_status": int(solution.constitutive_evidence.contraction_status),
            "accepted_updates": int(trained.accepted_updates),
            "reference": "independent-host-manufactured-discrete-conservation",
            "domain": f"Cartesian-lattice-cloud:{dimension}D:N={size}",
            "oracle_provenance": "analytical-nodal-field-and-host-endpoint-sigmoid-conservation",
            "retained_bytes": logical_array_bytes(
                (recovery, unseen, trained, solution, adjoint, prediction, refused)
            ),
        }
        if (
            not metrics["primal_successful"]
            or not metrics["unseen_primal_successful"]
            or not metrics["failure_rejected"]
        ):
            raise RuntimeError(
                "The learned PDE workflow did not satisfy its primal/failure admission contract."
            )
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", type=int, default=12, help="Total point count, not points per side."
    )
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    for key, value in run_workflow(**vars(parser.parse_args())).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
