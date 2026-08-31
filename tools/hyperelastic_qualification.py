#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import phydrax as phx


@dataclass(frozen=True)
class CaseConfiguration:
    shear_modulus: float
    bulk_modulus: float
    lame_lambda: float
    traction: float
    plane_strain: bool
    neural_width: int
    neural_depth: int
    neural_iterations: int
    training_interior_samples: int
    training_boundary_samples: int
    held_out_interior_samples: int
    held_out_boundary_samples: int
    neural_seeds: tuple[int, ...]
    fe_refinements: tuple[int, ...]


@dataclass(frozen=True)
class FiniteElementEvidence:
    refinement: int
    successful: bool
    status: int
    iterations: int
    residual_evaluations: int
    final_residual_norm: float
    minimum_jacobian: float
    total_potential: float
    tip_displacement: tuple[float, float]
    relative_force_balance: float


@dataclass(frozen=True)
class NeuralEvidence:
    seed: int
    final_training_potential: float
    held_out_potentials: tuple[float, ...]
    held_out_relative_spread: float
    minimum_jacobian: float
    nonpositive_jacobian_count: int
    clamp_linf: float
    relative_force_balance: float
    equilibrium_residual_l2: float
    displacement_relative_l2: float
    displacement_linf: float
    first_piola_relative_l2: float
    relative_potential_gap: float


@dataclass(frozen=True)
class AffineEvidence:
    energy_error: float
    first_piola_error: float
    cauchy_error: float
    invalid_jacobian_detected: bool


@dataclass(frozen=True)
class QualificationReport:
    maturity: str
    passed: bool
    smoke: bool
    configuration: CaseConfiguration
    finite_element: tuple[FiniteElementEvidence, ...]
    neural: tuple[NeuralEvidence, ...]
    affine: AffineEvidence
    environment: dict[str, str]
    gates: dict[str, float]


def _configuration(smoke: bool) -> CaseConfiguration:
    shear = 1.0
    bulk = 4.0
    return CaseConfiguration(
        shear_modulus=shear,
        bulk_modulus=bulk,
        lame_lambda=bulk - (2.0 / 3.0) * shear,
        traction=0.1,
        plane_strain=True,
        neural_width=8 if smoke else 24,
        neural_depth=2 if smoke else 3,
        neural_iterations=8 if smoke else 60,
        training_interior_samples=256 if smoke else 4096,
        training_boundary_samples=128 if smoke else 1024,
        held_out_interior_samples=512 if smoke else 8192,
        held_out_boundary_samples=256 if smoke else 8192,
        neural_seeds=(17,) if smoke else (17, 29),
        fe_refinements=(4, 8) if smoke else (8, 16, 24),
    )


def _structured_triangle_mesh(refinement: int):
    nodes = np.linspace(0.0, 1.0, refinement + 1)
    xx, yy = np.meshgrid(nodes, nodes, indexing="ij")
    vertices = np.stack((xx, yy), axis=-1).reshape((-1, 2))
    cells: list[tuple[int, int, int]] = []
    stride = refinement + 1
    for i in range(refinement):
        for j in range(refinement):
            lower_left = i * stride + j
            lower_right = (i + 1) * stride + j
            upper_left = i * stride + j + 1
            upper_right = (i + 1) * stride + j + 1
            cells.append((lower_left, lower_right, upper_left))
            cells.append((lower_right, upper_right, upper_left))
    return (
        jnp.asarray(vertices),
        jnp.asarray(cells, dtype=jnp.int32),
    )


def _material(configuration: CaseConfiguration):
    return phx.applications.solid_mechanics.NeoHookeanParameters(
        configuration.shear_modulus,
        configuration.lame_lambda,
    )


def _traction_coefficient(configuration: CaseConfiguration):
    traction = jnp.asarray((configuration.traction, 0.0))

    def load(points, context):
        del context
        right = points[..., 0] > 1.0 - 1e-10
        return jnp.where(right[..., None], traction, jnp.zeros_like(traction))

    return phx.equations.coefficient(
        load,
        coefficient_id="right-reference-traction",
    )


def _finite_element_problem(configuration: CaseConfiguration, refinement: int):
    vertices, cells = _structured_triangle_mesh(refinement)
    mesh = phx.discretization.CellMesh.from_triangles(vertices, cells)
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
        component_shape=(2,),
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    internal = phx.applications.solid_mechanics.neo_hookean_form(
        "u", _material(configuration)
    )
    form = phx.equations.FiniteElementForm(
        "plane-strain-traction",
        "u",
        internal.actions
        + (
            phx.equations.BoundaryLoadAction(
                "u",
                _traction_coefficient(configuration),
                domain=discretization.exterior_facet_domain,
                action_id="right-reference-traction",
            ),
        ),
    )
    left = jnp.isclose(vertices[:, 0], 0.0)
    constraint = phx.discretization.dirichlet_constraint(
        discretization,
        "u",
        boundary_mask=left,
        components=(0, 1),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=0.0,
    )
    return vertices, cells, left, discretization, compiled


def _cell_kinematics(vertices, cells, displacement, material):
    cell_points = vertices[cells]
    cell_displacement = displacement[cells]
    reference_edges = jnp.stack(
        (cell_points[:, 1] - cell_points[:, 0], cell_points[:, 2] - cell_points[:, 0]),
        axis=1,
    )
    displacement_edges = jnp.stack(
        (
            cell_displacement[:, 1] - cell_displacement[:, 0],
            cell_displacement[:, 2] - cell_displacement[:, 0],
        ),
        axis=1,
    )
    gradient_transpose = jax.vmap(jnp.linalg.solve)(reference_edges, displacement_edges)
    gradient = jnp.swapaxes(gradient_transpose, -1, -2)
    deformation_2d = jnp.eye(2) + gradient
    deformation_3d = (
        jnp.broadcast_to(jnp.eye(3), deformation_2d.shape[:-2] + (3, 3))
        .at[..., :2, :2]
        .set(deformation_2d)
    )
    first_piola = phx.applications.solid_mechanics.neo_hookean_first_piola(
        deformation_3d, material
    )[..., :2, :2]
    energy = phx.applications.solid_mechanics.neo_hookean_reference_energy(
        deformation_3d, material
    )
    area = 0.5 * jnp.abs(jnp.linalg.det(reference_edges))
    centroids = jnp.mean(cell_points, axis=1)
    return deformation_2d, first_piola, energy, area, centroids


def _finite_element_potential(
    configuration: CaseConfiguration,
    vertices,
    cells,
    displacement,
):
    deformation, first_piola, energy, area, centroids = _cell_kinematics(
        vertices, cells, displacement, _material(configuration)
    )
    internal = jnp.sum(energy * area)
    right = jnp.isclose(vertices[:, 0], 1.0)
    order = jnp.argsort(vertices[right, 1])
    right_values = displacement[right][order]
    right_y = vertices[right, 1][order]
    external = configuration.traction * jnp.trapezoid(right_values[:, 0], right_y)
    return (
        internal - external,
        deformation,
        first_piola,
        energy,
        area,
        centroids,
    )


def _solve_finite_element(configuration: CaseConfiguration, refinement: int):
    vertices, cells, left, _discretization, compiled = _finite_element_problem(
        configuration, refinement
    )
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=2e-7,
        relative_residual=2e-7,
        maximum_steps=40,
    )
    result = phx.nonlinear.NewtonTrustRegion().solve(
        compiled.as_nonlinear_problem(),
        compiled.state_space.zeros(),
        termination=termination,
    )
    jax.block_until_ready(result.state)
    full_state = jnp.asarray(compiled.expand(result.state))
    full_residual = jnp.asarray(compiled.full_residual(full_state))
    potential, deformation, first_piola, _, area, centroids = _finite_element_potential(
        configuration, vertices, cells, full_state
    )
    reaction = jnp.sum(full_residual[left], axis=0)
    applied = jnp.asarray((configuration.traction, 0.0))
    balance = jnp.linalg.norm(reaction + applied) / jnp.linalg.norm(applied)
    tip_index = refinement * (refinement + 1) + refinement // 2
    evidence = FiniteElementEvidence(
        refinement=refinement,
        successful=bool(result.successful),
        status=int(result.status),
        iterations=int(result.diagnostics.iterations),
        residual_evaluations=int(result.diagnostics.residual_evaluations),
        final_residual_norm=float(result.diagnostics.final_residual_norm),
        minimum_jacobian=float(jnp.min(jnp.linalg.det(deformation))),
        total_potential=float(potential),
        tip_displacement=tuple(float(value) for value in full_state[tip_index]),
        relative_force_balance=float(balance),
    )
    reference = {
        "vertices": vertices,
        "cells": cells,
        "displacement": full_state,
        "first_piola": first_piola,
        "area": area,
        "centroids": centroids,
        "potential": potential,
    }
    return evidence, reference


def _neural_problem(configuration: CaseConfiguration, seed: int):
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.5, 0.5), side=1.0).compile()
    )
    model = phx.nn.models.MLP(
        in_size=2,
        out_size=2,
        width_size=configuration.neural_width,
        depth=configuration.neural_depth,
        activation=jax.nn.tanh,
        key=jr.key(seed),
    )
    raw = domain.Model("x")(model)
    x_coordinate = domain.Function("x")(lambda x: x[0])
    displacement = 0.05 * x_coordinate * raw

    @domain.Function("x")
    def traction(x):
        return jnp.where(
            x[0] > 1.0 - 1e-10,
            jnp.asarray((configuration.traction, 0.0)),
            jnp.zeros(2),
        )

    def internal_density(functions):
        return phx.operators.neo_hookean_reference_energy(
            functions["u"],
            mu=configuration.shear_modulus,
            lambda_=configuration.lame_lambda,
        )

    def traction_work(functions):
        return phx.operators.einsum("...i,...i->...", traction, functions["u"])

    interior = phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.MonteCarloPlan(configuration.training_interior_samples),
        integrand=internal_density,
        materialization_policy="fixed",
        fixed_key=jr.key(seed + 1000),
        nonfinite_integrand="propagate",
        label="stored_energy",
    )
    boundary = phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component({"x": phx.domain.Boundary()})),
        plan=phx.integration.MonteCarloPlan(configuration.training_boundary_samples),
        integrand=traction_work,
        weight=-1.0,
        materialization_policy="fixed",
        fixed_key=jr.key(seed + 2000),
        nonfinite_integrand="propagate",
        label="traction_work",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": displacement},
        terms=(interior, boundary),
    )
    return domain, solver


def _held_out_potential(configuration, domain, displacement, key):
    interior_key, boundary_key = jr.split(key)

    @domain.Function("x")
    def traction(x):
        return jnp.where(
            x[0] > 1.0 - 1e-10,
            jnp.asarray((configuration.traction, 0.0)),
            jnp.zeros(2),
        )

    internal = phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.MonteCarloPlan(configuration.held_out_interior_samples),
        integrand=lambda functions: phx.operators.neo_hookean_reference_energy(
            functions["u"],
            mu=configuration.shear_modulus,
            lambda_=configuration.lame_lambda,
        ),
        materialization_policy="fixed",
        fixed_key=interior_key,
    )
    external = phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component({"x": phx.domain.Boundary()})),
        plan=phx.integration.MonteCarloPlan(configuration.held_out_boundary_samples),
        integrand=lambda functions: phx.operators.einsum(
            "...i,...i->...", traction, functions["u"]
        ),
        weight=-1.0,
        materialization_policy="fixed",
        fixed_key=boundary_key,
    )
    return phx.solver.FunctionalSolver(
        functions={"u": displacement}, terms=(internal, external)
    ).loss()


def _probe_grid(count: int):
    axis = jnp.linspace(0.025, 0.975, count)
    xx, yy = jnp.meshgrid(axis, axis, indexing="ij")
    return jnp.stack((xx, yy), axis=-1).reshape((-1, 2))


def _evaluate_field(field, points):
    return jax.vmap(field.func)(points)


def _neural_evidence(
    configuration: CaseConfiguration,
    seed: int,
    domain,
    trained,
    reference,
):
    displacement = trained["u"]
    training_potential = trained.loss()
    held_out_count = 1 if len(configuration.neural_seeds) == 1 else 3
    held_out = tuple(
        float(
            _held_out_potential(
                configuration,
                domain,
                displacement,
                jr.key(seed + 3000 + index),
            )
        )
        for index in range(held_out_count)
    )
    held_out_array = jnp.asarray(held_out)
    held_out_scale = jnp.maximum(jnp.abs(jnp.mean(held_out_array)), 1e-12)
    held_out_spread = (jnp.max(held_out_array) - jnp.min(held_out_array)) / held_out_scale

    probes = _probe_grid(7 if len(configuration.neural_seeds) == 1 else 15)
    deformation_field = phx.operators.deformation_gradient(displacement)
    deformation = _evaluate_field(deformation_field, probes)
    jacobian = jnp.linalg.det(deformation)
    first_piola_field = phx.operators.neo_hookean_pk1(
        displacement,
        mu=configuration.shear_modulus,
        lambda_=configuration.lame_lambda,
    )
    equilibrium = phx.operators.div_tensor(first_piola_field, var="x")
    equilibrium_values = _evaluate_field(equilibrium, probes)
    equilibrium_l2 = jnp.sqrt(jnp.mean(equilibrium_values * equilibrium_values))

    boundary_axis = jnp.linspace(0.0, 1.0, 101)
    left_points = jnp.stack((jnp.zeros_like(boundary_axis), boundary_axis), axis=-1)
    left_values = _evaluate_field(displacement, left_points)
    left_piola = _evaluate_field(first_piola_field, left_points)
    left_normal = jnp.asarray((-1.0, 0.0))
    reaction_density = left_piola @ left_normal
    reaction = jnp.trapezoid(reaction_density, boundary_axis, axis=0)
    applied = jnp.asarray((configuration.traction, 0.0))
    force_balance = jnp.linalg.norm(reaction + applied) / jnp.linalg.norm(applied)

    neural_at_vertices = _evaluate_field(displacement, reference["vertices"])
    displacement_error = neural_at_vertices - reference["displacement"]
    displacement_relative_l2 = jnp.linalg.norm(displacement_error) / jnp.maximum(
        jnp.linalg.norm(reference["displacement"]), 1e-12
    )
    displacement_linf = jnp.max(jnp.abs(displacement_error))

    neural_piola = _evaluate_field(first_piola_field, reference["centroids"])
    piola_error_sq = jnp.sum(
        reference["area"][:, None, None] * (neural_piola - reference["first_piola"]) ** 2
    )
    piola_reference_sq = jnp.sum(
        reference["area"][:, None, None] * reference["first_piola"] ** 2
    )
    piola_relative_l2 = jnp.sqrt(piola_error_sq / jnp.maximum(piola_reference_sq, 1e-24))
    potential_gap = jnp.abs(
        jnp.mean(held_out_array) - reference["potential"]
    ) / jnp.maximum(jnp.abs(reference["potential"]), 1e-12)

    return NeuralEvidence(
        seed=seed,
        final_training_potential=float(training_potential),
        held_out_potentials=held_out,
        held_out_relative_spread=float(held_out_spread),
        minimum_jacobian=float(jnp.min(jacobian)),
        nonpositive_jacobian_count=int(jnp.sum(jacobian <= 0.0)),
        clamp_linf=float(jnp.max(jnp.abs(left_values))),
        relative_force_balance=float(force_balance),
        equilibrium_residual_l2=float(equilibrium_l2),
        displacement_relative_l2=float(displacement_relative_l2),
        displacement_linf=float(displacement_linf),
        first_piola_relative_l2=float(piola_relative_l2),
        relative_potential_gap=float(potential_gap),
    )


def _solve_neural(configuration, seed, reference):
    domain, solver = _neural_problem(configuration, seed)
    trained = solver.solve(
        num_iter=configuration.neural_iterations,
        optim=optax.lbfgs(learning_rate=1.0),
        seed=seed + 4000,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    jax.block_until_ready(trained.loss())
    return _neural_evidence(configuration, seed, domain, trained, reference)


def _affine_evidence(configuration: CaseConfiguration):
    domain = phx.domain.GeometryDomain(
        phx.geometry.Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)).compile()
    )
    gradient = jnp.asarray([[0.08, 0.02, 0.01], [0.03, -0.04, 0.02], [0.0, 0.01, 0.05]])

    @domain.Function("x")
    def displacement(x):
        return gradient @ x

    point = jnp.asarray((0.2, -0.1, 0.3))
    energy = phx.operators.neo_hookean_reference_energy(
        displacement,
        mu=configuration.shear_modulus,
        lambda_=configuration.lame_lambda,
    ).func(point)
    first_piola = phx.operators.neo_hookean_pk1(
        displacement,
        mu=configuration.shear_modulus,
        lambda_=configuration.lame_lambda,
    ).func(point)
    cauchy = phx.operators.neo_hookean_cauchy(
        displacement,
        mu=configuration.shear_modulus,
        lambda_=configuration.lame_lambda,
    ).func(point)
    deformation = jnp.eye(3) + gradient
    material = _material(configuration)
    expected_energy = phx.applications.solid_mechanics.neo_hookean_reference_energy(
        deformation, material
    )
    expected_piola = phx.applications.solid_mechanics.neo_hookean_first_piola(
        deformation, material
    )
    expected_cauchy = expected_piola @ deformation.T / jnp.linalg.det(deformation)

    @domain.Function("x")
    def inverted(x):
        return jnp.asarray((-2.0 * x[0], 0.0, 0.0))

    invalid = phx.operators.neo_hookean_reference_energy(
        inverted,
        mu=configuration.shear_modulus,
        lambda_=configuration.lame_lambda,
    ).func(point)
    return AffineEvidence(
        energy_error=float(jnp.abs(energy - expected_energy)),
        first_piola_error=float(jnp.max(jnp.abs(first_piola - expected_piola))),
        cauchy_error=float(jnp.max(jnp.abs(cauchy - expected_cauchy))),
        invalid_jacobian_detected=not bool(jnp.isfinite(invalid)),
    )


def _passes(
    configuration: CaseConfiguration,
    finite_element: tuple[FiniteElementEvidence, ...],
    neural: tuple[NeuralEvidence, ...],
    affine: AffineEvidence,
    smoke: bool,
):
    gates = {
        "minimum_jacobian": 0.25,
        "clamp_linf": 1e-10,
        "fe_relative_force_balance": 5e-5,
        "neural_relative_force_balance": 0.35 if smoke else 0.2,
        "displacement_relative_l2": 0.45 if smoke else 0.2,
        "first_piola_relative_l2": 0.65 if smoke else 0.35,
        "relative_potential_gap": 0.35 if smoke else 0.15,
        "held_out_relative_spread": 0.3 if smoke else 0.12,
        "affine_error": 2e-10,
    }
    fe_tip = jnp.asarray([case.tip_displacement for case in finite_element])
    fe_converged = len(finite_element) < 2 or bool(
        jnp.linalg.norm(fe_tip[-1] - fe_tip[-2])
        / jnp.maximum(jnp.linalg.norm(fe_tip[-1]), 1e-12)
        < (0.08 if smoke else 0.02)
    )
    passed = (
        all(case.successful for case in finite_element)
        and all(
            case.minimum_jacobian > gates["minimum_jacobian"] for case in finite_element
        )
        and all(
            case.relative_force_balance < gates["fe_relative_force_balance"]
            for case in finite_element
        )
        and fe_converged
        and all(
            case.minimum_jacobian > gates["minimum_jacobian"]
            and case.nonpositive_jacobian_count == 0
            and case.clamp_linf < gates["clamp_linf"]
            and case.relative_force_balance < gates["neural_relative_force_balance"]
            and case.displacement_relative_l2 < gates["displacement_relative_l2"]
            and case.first_piola_relative_l2 < gates["first_piola_relative_l2"]
            and case.relative_potential_gap < gates["relative_potential_gap"]
            and case.held_out_relative_spread < gates["held_out_relative_spread"]
            for case in neural
        )
        and affine.invalid_jacobian_detected
        and affine.energy_error < gates["affine_error"]
        and affine.first_piola_error < gates["affine_error"]
        and affine.cauchy_error < gates["affine_error"]
    )
    return bool(passed), gates


def qualify(smoke: bool) -> QualificationReport:
    configuration = _configuration(smoke)
    finite_element_results = tuple(
        _solve_finite_element(configuration, refinement)
        for refinement in configuration.fe_refinements
    )
    finite_element = tuple(result[0] for result in finite_element_results)
    reference = finite_element_results[-1][1]
    neural = tuple(
        _solve_neural(configuration, seed, reference)
        for seed in configuration.neural_seeds
    )
    affine = _affine_evidence(configuration)
    passed, gates = _passes(configuration, finite_element, neural, affine, smoke)
    return QualificationReport(
        maturity="experimental",
        passed=passed,
        smoke=smoke,
        configuration=configuration,
        finite_element=finite_element,
        neural=neural,
        affine=affine,
        environment={
            "platform": platform.platform(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "dtype": str(jnp.asarray(0.0).dtype),
        },
        gates=gates,
    )


def _write_report(report: QualificationReport, output: Path) -> None:
    payload = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "hyperelastic_qualification.json",
    )
    arguments = parser.parse_args()
    report = qualify(arguments.smoke)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    _write_report(report, arguments.output)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
