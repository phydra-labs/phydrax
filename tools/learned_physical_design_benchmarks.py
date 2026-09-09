#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Deterministic no-checkpoint density, geometry and guided-proposal FEM workflows.

Run: python tools/learned_physical_design_benchmarks.py --segments 3 --steps 100
All derivatives are accepted-point matrix-free VJPs. Thresholding and reference
FE reanalysis are outside differentiation. A latent certificate is not a claim
of full-design optimality; heuristic denoised guidance is not a likelihood score.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._score_field import StateTimeScoreField
from phydrax.applications.solid_mechanics._learned_design import (
    MechanicsPotentialGuidance,
    prepare_learned_shape_design,
    prepare_learned_topology_design,
    solve_learned_topology_design,
)
from phydrax.optim._state_design_linearization import (
    prepare_state_design_linearization,
    state_design_response_vjp,
)


sm = phx.applications.solid_mechanics


def _linear_policy():
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1.0e-10,
            absolute=1.0e-12,
            max_steps=300,
        ),
    )


def _fe_solver():
    def execute(problem, design, initial, args):
        zero = jax.tree.map(jnp.zeros_like, initial)
        offset, action = jax.linearize(
            lambda state: problem.residual(state, design, args), zero
        )
        operator = phx.linalg.FunctionLinearOperator(
            action,
            source=phx.linalg.PyTreeSpace(zero),
            target=phx.linalg.PyTreeSpace(offset),
            transpose_action=action,
            operator_id="learned-design-native-fe-stiffness",
            closure_convert=False,
        )
        solved = phx.linalg.solve(
            phx.linalg.LinearSystem(operator),
            jax.tree.map(jnp.negative, offset),
            policy=_linear_policy(),
        )
        return sm.MechanicsStateCandidate(
            solved.value,
            status=jnp.where(
                solved.successful,
                int(phx.optim.OptimizationStatus.SUCCESS),
                int(phx.optim.OptimizationStatus.LINEAR_SOLVE_FAILED),
            ),
        )

    return sm.FiniteElementStateSolver(execute, solver_id="native-triangle-fe-gmres")


def _mesh(segments):
    if segments < 2:
        raise ValueError("segments must be at least two.")
    coordinates = jnp.asarray(
        tuple((float(i), float(j)) for i in range(segments + 1) for j in range(2))
    )
    cells = jnp.asarray(
        tuple(
            triangle
            for i in range(segments)
            for triangle in (
                (2 * i, 2 * i + 2, 2 * i + 1),
                (2 * i + 2, 2 * i + 3, 2 * i + 1),
            )
        ),
        dtype=jnp.int32,
    )
    mesh = phx.discretization.CellMesh.from_triangles(coordinates, cells)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("triangle", 1),
            component_shape=(2,),
        ),
    ).prepare()

    def elasticity(
        values, gradients, points, weights, test_basis, test_gradients, context
    ):
        del values, points, test_basis
        gradient = gradients[0]
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        trace = jnp.trace(strain, axis1=-2, axis2=-1)
        # Plane-strain isotropic elasticity, lambda = mu = cell modulus.
        modulus = context.user_args[:, None, None, None]
        stress = modulus * (trace[..., None, None] * jnp.eye(2) + 2.0 * strain)
        return phx.ein.contract("cq,cqid,cqad->cia", weights, test_gradients, stress)

    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "learned-native-elasticity",
            "u",
            (
                phx.equations.CellResidualAction(
                    "u", ("u",), elasticity, action_id="elasticity"
                ),
            ),
        ),
        discretization,
    )
    load = jnp.zeros((2 * segments, 2)).at[-1, 1].set(-0.01)

    def residual(state, modulus, geometry):
        runtime = discretization.prepare_runtime(
            geometry, numeric_version="learned-active-geometry"
        )
        context = phx.equations.FiniteElementExecutionContext(runtime, user_args=modulus)
        full_state = jnp.zeros_like(coordinates).at[2:].set(state)
        return compiled.full_residual(full_state, context)[2:]

    return coordinates, cells, load, residual


def _acceptance():
    return phx.optim.StateAcceptancePolicy(
        state_relative_tolerance=1.0e-7,
        state_absolute_tolerance=1.0e-10,
        adjoint_relative_tolerance=1.0e-7,
        adjoint_absolute_tolerance=1.0e-10,
    )


def _reference_solve(problem, density, initial, args):
    point = prepare_state_design_linearization(
        problem, density, initial, args=args, linear_policy=_linear_policy()
    )
    response = state_design_response_vjp(point)
    return sm.FiniteElementReanalysisCandidate(
        point.state,
        response.adjoint,
        state_status=point.state_result.status,
        adjoint_status=response.linear_result.status,
        solver_id="independent-native-reference-fe",
    )


def build_density_case(segments=3):
    coordinates, cells, load, residual = _mesh(segments)
    centers = jnp.mean(coordinates[cells], axis=1)
    count = cells.shape[0]
    mask = jnp.ones((count,), dtype=bool).at[0].set(False)
    fixed = jnp.zeros((count,)).at[0].set(1.0)
    prepared = phx.optim.DensityTransformPlan(
        phx.optim.ConicDensityFilterPlan(
            centers, 0.0, mask, fixed, jnp.full((count,), 0.5)
        ),
        phx.optim.TanhDensityProjectionPlan(jnp.asarray(0.5)),
    ).prepare()
    branch = sm.MechanicsBranchGate(("linear-elastic",))
    topology = sm.TopologyMechanicsProblem(
        lambda state, modulus, case, args: (
            residual(state, modulus, coordinates) - case.load
        ),
        (sm.LoadCase(load, case_id="cantilever-tip"),),
        sm.DensityTransform(prepared, beta=1.0),
        sm.MaterialInterpolation(1.0, minimum=0.05, penalty=2.0),
        0.65,
        _fe_solver(),
        acceptance_policy=_acceptance(),
        branch_evaluator=lambda state, physical, case, args: branch.evaluate(
            "linear-elastic"
        ),
        problem_id="learned-cantilever-density",
    )
    basis = jnp.stack((jnp.ones((count,)), centers[:, 0] / segments - 0.5), axis=-1)

    def decode(latent):
        return jax.nn.sigmoid(phx.ein.contract("ci,i->c", basis, latent))

    learned = prepare_learned_topology_design(
        topology,
        decode,
        jnp.zeros((2,)),
        latent_bounds=phx.optim.Bounds(-3.0, 3.0),
        decoder_id="deterministic-two-mode-logistic",
        realization_id=topology.density_transform.transform_id,
    )

    # Invert this same-mesh finite-beta projection exactly, not by clipping.
    def transfer(physical):
        beta = topology.density_transform.beta
        eta = prepared.plan.projection.eta
        lower = jnp.tanh(beta * eta)
        denominator = lower + jnp.tanh(beta * (1.0 - eta))
        raw = eta + jnp.arctanh(physical * denominator - lower) / beta
        return sm.DensityTransferCandidate(jnp.where(mask, raw, fixed))

    plan = sm.TopologyReanalysisPlan(
        topology, transfer, _reference_solve, plan_id="smooth-native-reference"
    )
    return learned, (jnp.zeros_like(load),), plan, mask, fixed


def build_shape_case(segments=3):
    coordinates, cells, load, residual = _mesh(segments)
    geometry = phx.geometry.design
    schema = geometry.ParameterSchema(
        (
            geometry.ParameterSpec(
                geometry.ParameterId("cantilever", "coordinates"),
                coordinates.shape,
                str(coordinates.dtype),
                "mesh-geometry",
                bounds=(-1.0, float(segments + 1)),
            ),
        )
    )
    template = geometry.DesignState(schema, (coordinates,))
    x = coordinates[:, 0] / segments
    basis = jnp.stack((x, x * (1.0 - x)), axis=-1)

    def decode(latent):
        height = 1.0 + 0.25 * jnp.tanh(phx.ein.contract("ni,i->n", basis, latent))
        points = coordinates.at[:, 1].set(coordinates[:, 1] * height)
        return geometry.DesignState(schema, (points,))

    def signed_area(design):
        triangle = design.values[0][cells]
        a, b = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    physical = phx.optim.StateDesignProblem(
        lambda state, design, args: (
            residual(state, jnp.ones((cells.shape[0],)), design.values[0]) - load
        ),
        lambda state, design, args: phx.ein.contract("nd,nd->", load, state),
        state_solver=_fe_solver(),
        acceptance_policy=_acceptance(),
        constraints=(
            phx.optim.StateDesignConstraint(
                lambda state, design, args: jnp.sum(signed_area(design)),
                upper=float(segments),
                constraint_id="shape-area",
                depends_on_state=False,
            ),
        ),
        problem_id="learned-cantilever-shape",
    )
    learned = prepare_learned_shape_design(
        physical,
        decode,
        jnp.zeros((2,)),
        template,
        latent_bounds=phx.optim.Bounds(-2.0, 2.0),
        decoder_id="deterministic-two-mode-boundary",
        realization_id="cantilever-fixed-triangle-connectivity",
        design_admissibility=lambda design: jnp.all(signed_area(design) > 1.0e-8),
    )
    return learned, jnp.zeros_like(load)


def run(segments=3, steps=100):
    started = time.perf_counter()
    termination = phx.optim.OptimizationTermination(
        maximum_steps=steps,
        absolute_optimality=1.0e-5,
        relative_optimality=0.0,
    )
    design, initial, plan, mask, fixed = build_density_case(segments)
    solved = solve_learned_topology_design(
        design,
        initial,
        jnp.zeros((2,)),
        plan,
        initial,
        termination=termination,
        method=phx.optim.ReducedMMA(linear_policy=_linear_policy()),
    )
    # Binary extraction preserves prescribed cells and runs outside every VJP.
    hard_plan = sm.TopologyReanalysisPlan(
        plan.reference_problem,
        lambda physical: sm.DensityTransferCandidate(
            jnp.where(
                mask,
                phx.optim.threshold_density(physical, 0.5),
                fixed,
            )
        ),
        _reference_solve,
        plan_id="hard-extraction-reference-fe",
    )
    hard = sm.reanalyse_topology_design(solved.topology_result, hard_plan, initial)
    shape, shape_initial = build_shape_case(segments)
    shape_solved = phx.optim.solve_state_design(
        shape.problem,
        shape_initial,
        jnp.zeros((2,)),
        method=phx.optim.ReducedMMA(linear_policy=_linear_policy()),
        termination=termination,
    )
    # New FE root and transpose at the final physical geometry, no optimizer AD.
    shape_final = shape.response_vjp(
        shape_solved.design, shape_initial, linear_policy=_linear_policy()
    )
    shape_constraints_accepted = jnp.asarray(True)
    for constraint in shape.physical_problem.constraints:
        values = constraint.value(
            shape_final.state_result.state, shape_final.physical_design
        )
        lower, upper = constraint.bounds(values)
        for value, lo, hi in zip(
            jax.tree.leaves(values),
            jax.tree.leaves(lower),
            jax.tree.leaves(upper),
            strict=True,
        ):
            shape_constraints_accepted = shape_constraints_accepted & jnp.all(
                (value >= lo - termination.absolute_optimality)
                & (value <= hi + termination.absolute_optimality)
            )

    domain = phx.domain.HyperRectangle(
        jnp.full((2,), -3.0), jnp.full((2,), 3.0), label="x"
    ) @ phx.domain.TimeInterval(0.0, 1.0)
    base = StateTimeScoreField(
        domain.Function("x", "t")(lambda state, time: -state),
        state_label="x",
        time_label="t",
    )
    guidance = MechanicsPotentialGuidance(
        design.parameterization,
        initial,
        scale=0.1,
        linear_policy=_linear_policy(),
        denoise=lambda latent, time, context: latent / (1.0 + time),
    )
    guided = phx.transport.GuidedScoreField(base, (guidance,))
    context = phx.transport.ScoreContext({})
    latent = jnp.asarray([-0.25, 0.1])
    score, evaluations, valid = guided.evaluate(latent, 0.25, context)
    proposal = latent + 0.05 * score
    proposal_result = None
    if bool(valid):
        proposal_result = solve_learned_topology_design(
            design,
            initial,
            proposal,
            plan,
            initial,
            termination=termination,
            method=phx.optim.ReducedMMA(linear_policy=_linear_policy()),
        )
    jax.block_until_ready(shape_final.values)
    return {
        "segments": segments,
        "cells": 2 * segments,
        "wall_seconds": time.perf_counter() - started,
        "density": {
            "latent_status": int(solved.latent_result.status),
            "accepted": bool(solved.accepted),
            "compliance": float(solved.latent_result.objective),
            "volume_ratio": float(solved.topology_result.volume_ratio),
            "fixed_cells_exact": bool(
                jnp.all(solved.topology_result.physical_density[~mask] == fixed[~mask])
            ),
            "reference_fe_accepted": bool(solved.reanalysis.evidence.mechanics.accepted),
            "reference_volume_ratio": float(solved.reference_volume_ratio),
            "reference_constraints_accepted": bool(solved.reference_feasible),
        },
        "hard_extraction": {
            "accepted": bool(hard.accepted),
            "transfer_accepted": bool(hard.evidence.transfer.accepted),
            "material_measure_error": float(
                hard.evidence.transfer.relative_measure_error
            ),
            "fe_accepted": bool(hard.evidence.mechanics.accepted),
            "rejection_is_final": not bool(hard.accepted),
        },
        "shape": {
            "latent_status": int(shape_solved.status),
            "physical_fe_vjp_accepted": bool(shape_final.accepted),
            "physical_constraints_accepted": bool(shape_constraints_accepted),
            "accepted": bool(
                shape_solved.successful
                & shape_final.accepted
                & shape_constraints_accepted
            ),
            "compliance": float(shape_final.values),
            "schema_preserved": shape_final.physical_design.schema
            == shape.decode(jnp.zeros((2,))).schema,
        },
        "guidance": {
            "valid": bool(valid),
            "exactness": evaluations[0].exactness,
            "correction": evaluations[0].correction.tolist(),
            "proposal_accepted_after_fe_reanalysis": False
            if proposal_result is None
            else bool(proposal_result.accepted),
            "proposal_rejected": proposal_result is None
            or not bool(proposal_result.accepted),
        },
        "optimality_scope": "latent decoder image only; not full physical-design stationarity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    options = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    print(json.dumps(run(options.segments, options.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
