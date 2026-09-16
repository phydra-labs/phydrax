#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exercise the native polynomial, geometry, optimization, tensor, and power paths."""

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax import algebraic
from phydrax.algebraic._positive_dimensional import WitnessSet
from phydrax.algebraic._quotient import solve_quotient_roots
from phydrax.applications import power
from phydrax.applications.power._polynomial_power_flow import (
    compile_fixed_mode_power_flow_polynomial,
)
from phydrax.geometry import polynomial_image
from phydrax.geometry.surface._g1_multipatch import (
    BiquinticGluingConstraints,
    GluingContinuity,
    PreparedHalfEdgePatchTopology,
)
from phydrax.optim import polynomial as polynomial_optim
from phydrax.tensor_decomposition import (
    normalize_waring_components,
    reconstruct_symmetric_tensor,
    solve_symmetric_waring,
    SymmetricWaringProblem,
    SymmetricWaringRefinement,
)


def _system(labels, equations, rows, exponents, coefficients):
    return algebraic.SparsePolynomialSystem.from_coo(
        labels,
        equations,
        rows,
        exponents,
        jnp.asarray(coefficients),
    )


def _core_case():
    system = _system(("x",), ("f",), (0, 0), ((0,), (2,)), (-1.0, 1.0))
    values = system.evaluate(jnp.asarray([[-1.0], [1.0]]))
    jacobian = system.jacobian(jnp.asarray([[0.0], [1.0]]))
    forecast = algebraic.total_degree_bezout_forecast(system.support)
    symmetry = algebraic.analyze_exponent_lattice_scaling(system)
    quotient = solve_quotient_roots(system, polish=True)
    roots = np.sort(np.asarray(quotient.roots[:, 0]).real)
    passed = (
        np.allclose(values, 0.0)
        and np.all(np.isfinite(np.asarray(jacobian)))
        and forecast.path_count == 2
        and symmetry.torsion_orders == (2,)
        and bool(quotient.successful)
        and np.allclose(roots, (-1.0, 1.0), atol=2.0e-6)
    )
    return {
        "passed": bool(passed),
        "support_id": system.support.support_id,
        "system_id": system.system_id,
        "total_degree_paths": forecast.path_count,
        "free_scaling_rank": symmetry.free_rank,
        "torsion_orders": symmetry.torsion_orders,
        "quotient_status": int(quotient.status),
        "quotient_roots": roots.tolist(),
        "maximum_residual": float(
            np.max(np.abs(np.asarray(quotient.original_residuals)))
        ),
    }


def _image_case():
    numeric = _system(
        ("t",),
        ("x", "y", "z"),
        (0, 1, 2),
        ((1,), (2,), (3,)),
        (1.0, 1.0, 1.0),
    )
    mapping = polynomial_image.SparsePolynomialMap(numeric)
    support = polynomial_image.TargetMonomialSupport(
        ("x", "y", "z"),
        tuple(
            (first, second, third)
            for first in range(3)
            for second in range(3 - first)
            for third in range(3 - first - second)
        ),
    )
    result = (
        polynomial_image.plan_polynomial_image_analysis(
            mapping,
            support,
            discovery_sample_count=24,
            heldout_sample_count=8,
            key=jr.PRNGKey(2026),
            policy=polynomial_image.PolynomialImageAnalysisPolicy(
                rank_relative_tolerance=1.0e-5,
                heldout_absolute_tolerance=2.0e-5,
                heldout_relative_tolerance=2.0e-5,
            ),
        )
        .prepare()
        .result
    )
    passed = (
        result.accepted
        and result.jacobian_rank.resolved
        and result.relations.relation_count == 3
    )
    return {
        "passed": bool(passed),
        "status": result.status.value,
        "dimension": result.jacobian_rank.dimension_lower_bound,
        "relations": result.relations.relation_count,
        "heldout_accepted": bool(result.relations.validation_accepted),
    }


def _optimization_case():
    objective = _system(("x",), ("objective",), (0,), ((2,),), (1.0,))
    inequalities = _system(("x",), ("unit interval",), (0, 0), ((0,), (2,)), (1.0, -1.0))
    problem = polynomial_optim.PolynomialOptimizationProblem(
        objective, inequalities=inequalities
    )
    prepared = polynomial_optim.prepare_polynomial_relaxation(problem, 1)
    candidate = polynomial_optim.audit_polynomial_candidate(problem, jnp.asarray([0.0]))
    passed = (
        prepared.plan.ready
        and bool(candidate.feasible)
        and float(candidate.objective) == 0.0
    )
    return {
        "passed": bool(passed),
        "moment_count": prepared.template.moment_basis.moment_count,
        "moment_matrix_size": prepared.template.moment_basis.matrix_size,
        "conic_rows": prepared.plan.estimate.conic_rows,
        "candidate_feasible": bool(candidate.feasible),
    }


def _witness_case():
    witness = WitnessSet(
        "parabola",
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        np.asarray([[-1.0, 1.0], [1.0, 1.0]], dtype=complex),
        np.asarray([0.0, 0.0]),
    )
    passed = witness.dimension == 1 and witness.degree == 2
    return {
        "passed": bool(passed),
        "dimension": witness.dimension,
        "degree": witness.degree,
        "witness_id": witness.witness_id,
    }


def _tensor_case():
    weights, factors = normalize_waring_components(
        jnp.asarray([1.7, -0.65]),
        jnp.asarray([[1.0, 0.35], [0.55, 1.0]]),
        3,
    )
    tensor = reconstruct_symmetric_tensor(weights, factors, 3)
    result = solve_symmetric_waring(
        SymmetricWaringProblem(tensor, 2),
        refinement=SymmetricWaringRefinement(enabled=False),
    )
    passed = bool(result.successful) and float(result.relative_residual) < 2.0e-5
    return {
        "passed": bool(passed),
        "status": int(result.status),
        "relative_residual": float(result.relative_residual),
    }


def _g1_case():
    topology = PreparedHalfEdgePatchTopology(
        ("left", "right"),
        (("a", "b", "e", "d"), ("b", "c", "f", "e")),
    )
    constraints = BiquinticGluingConstraints(topology, GluingContinuity.G1)
    parameter = jnp.linspace(0.0, 1.0, 6)
    u, v = jnp.meshgrid(parameter, parameter, indexing="ij")
    zero = jnp.zeros_like(u)
    coefficients = jnp.stack(
        (
            jnp.stack((u, v, zero), axis=-1),
            jnp.stack((1.0 + u, v, zero), axis=-1),
        )
    )
    residual = constraints.residual(coefficients)
    passed = (
        constraints.status.value == "supported_parametric_c1"
        and float(jnp.max(jnp.abs(residual))) < 2.0e-6
    )
    return {
        "passed": bool(passed),
        "status": constraints.status.value,
        "constraint_count": constraints.constraint_count,
        "maximum_residual": float(jnp.max(jnp.abs(residual))),
    }


def _power_case():
    network = power.PowerNetwork(
        (power.Bus("source", 110), power.Bus("load", 110)),
        (power.Branch("line", "source", "load", 0.0, 0.1),),
        (power.Generator("g", "source"),),
        (power.Load("d", "load", 0.5, 0.0),),
    )
    study = power.PowerStudy(
        (power.BusControl("source", "reference"), power.BusControl("load", "pq"))
    )
    compiled = power.compile_network(network, study)
    polynomial = compile_fixed_mode_power_flow_polynomial(compiled)
    result = power.fixed_mode_power_flow(compiled)
    residual = polynomial.system.evaluate(polynomial.coordinates(result.voltage))
    native = polynomial.native_residual(result.voltage)
    defect = float(jnp.max(jnp.abs(residual - native)))
    passed = bool(result.converged) and defect < 2.0e-8
    return {
        "passed": bool(passed),
        "support_terms": polynomial.system.support.term_count,
        "native_polynomial_defect": defect,
        "local_residual": float(result.residual_norm),
    }


def main():
    cases = {
        "core": _core_case(),
        "polynomial_image": _image_case(),
        "polynomial_optimization": _optimization_case(),
        "witness": _witness_case(),
        "symmetric_tensor": _tensor_case(),
        "g1_multipatch": _g1_case(),
        "power_flow": _power_case(),
    }
    output = {
        "claim": "bounded-native-algebraic-qualification-not-global-completeness-or-certification",
        "passed": all(case["passed"] for case in cases.values()),
        "cases": cases,
    }
    print(json.dumps(output, allow_nan=False, indent=2, sort_keys=True))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
