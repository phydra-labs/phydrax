# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Q12 higher-form and Q14 sensitivity workloads (P16 closure group).

Geometry-authorized k-form realizations, the abstract radius-clique research
route, fixed-support/smooth-envelope coordinate sensitivities, strict-active
nonnegative metric tangents and frozen multi-history remap derivatives are
measured phase by phase through the shared ``benchmarks.meshfree_scaling``
recorder. Every oracle is independent of the implementation under test: exact
polynomial/cubature integrals, exact integer topology, NumPy normal-equation
GMLS, central differences of the published maps, and inner-product duality.
Refusal workloads report the documented typed status or exception; they never
stand in for an accepted measurement.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations_with_replacement
from math import comb, pi
from typing import Any, assert_never, Literal, TYPE_CHECKING, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.spatial import cKDTree

from benchmarks._runtime import logical_array_bytes
from benchmarks.meshfree_scaling import (
    cloud_points,
    declare_reservation,
    DeclaredCapacityRefusal,
    fill_distance,
    MeshfreeConfig,
    PhaseRecorder,
    unit_cube_probes,
)
from phydrax.discretization.meshfree import LocalStencilEvidence


if TYPE_CHECKING:
    from phydrax.discretization import CellMesh, PreparedPointCloudDiscretization
    from phydrax.discretization.meshfree import (
        ComplexDomainIdentity,
        PreparedLocalStencils,
        PreparedMeshfreeCellComplex,
        PreparedMeshfreeExteriorCalculus,
        RadiusCliqueComplex,
    )
    from phydrax.metrix import CoordinateChart, DifferentialForm


HigherFormGeometry: TypeAlias = Literal["holed-square", "tunneled-slab", "icosphere"]

# Owner-published derivative contract identity of a bound nonnegative metric
# (PreparedMeshfreeMetric, MetricP2c); a different string is a different claim.
STRICT_ACTIVE_CONTRACT = (
    "strict-active fixed-set projection-KKT derivative after bind_active_set"
)
_MAXWELL_STEPS = 20
_TINY = float(np.finfo(np.float64).tiny)


def _relative(error: float, scale: float, /) -> float:
    return float(error) / max(float(scale), _TINY)


def _refused(
    operation: Callable[[], object], error: type[Exception], fragment: str, /
) -> bool:
    """True iff ``operation`` raises the documented typed refusal naming ``fragment``.

    Only the one documented exception type is caught; any other exception, or
    the documented type with a different cause, propagates to the row boundary.
    """
    try:
        operation()
    except error as raised:
        if fragment not in str(raised):
            raise
        return True
    return False


# ---------------------------------------------------------------------------
# Q12 geometry-authorized higher forms


def higher_form_top_cells(geometry: HigherFormGeometry, level: int, /) -> int:
    """Top-simplex count of one authoritative refinement level."""
    match geometry:
        case "holed-square":
            return 16 * level**2
        case "tunneled-slab":
            return 96 * level**3
        case "icosphere":
            return 20 * 4**level
        case _:
            assert_never(geometry)


def higher_form_level(geometry: HigherFormGeometry, capacity: int, /) -> int:
    """Finest refinement level whose top-cell count fits the requested capacity."""
    if higher_form_top_cells(geometry, 1) > capacity:
        raise ValueError(
            f"{geometry} needs a top-cell capacity of at least "
            f"{higher_form_top_cells(geometry, 1)}; requested {capacity}."
        )
    level = 1
    while higher_form_top_cells(geometry, level + 1) <= capacity:
        level += 1
    return level


def higher_form_dimension(geometry: HigherFormGeometry, /) -> int:
    """Ambient dimension of the authoritative complex."""
    match geometry:
        case "holed-square":
            return 2
        case "tunneled-slab" | "icosphere":
            return 3
        case _:
            assert_never(geometry)


def _higher_form_complex(
    geometry: HigherFormGeometry, level: int, /
) -> tuple[CellMesh, ComplexDomainIdentity]:
    import examples.meshfree_higher_forms as consumer
    from phydrax.discretization.meshfree import ComplexDomainIdentity

    match geometry:
        case "holed-square":
            return consumer.planar_grid_mesh(level, hole=True), ComplexDomainIdentity(
                "square-with-hole", "domain", measure=8.0, betti=(1, 1, 0)
            )
        case "tunneled-slab":
            return consumer.box_tetrahedral_mesh(
                level, tunnel=True
            ), ComplexDomainIdentity(
                "tunneled-slab", "domain", measure=8.0, betti=(1, 1, 0, 0)
            )
        case "icosphere":
            # Inscribed flat facets lose 7.2% (level 1) of the sphere area; the
            # declared tolerance states that chordal deficit for every level.
            return consumer.sphere_mesh(level), ComplexDomainIdentity(
                "unit-sphere",
                "closed_surface",
                measure=4.0 * pi,
                betti=(1, 0, 1),
                measure_tolerance=0.08,
            )
        case _:
            assert_never(geometry)


def _cubic_form(chart: CoordinateChart, degree: int, /) -> DifferentialForm:
    """Cubic coefficients: the declared order-6 simplex cubature is exact."""
    from phydrax.metrix import DifferentialForm

    components = comb(chart.dimension, degree)

    def coefficients(point: Array) -> Array:
        base = point[0] ** 3 - point[1] * point[-1] ** 2 + 0.5 * point[1]
        return base * jnp.arange(1, components + 1, dtype=jnp.float64)

    return DifferentialForm(coefficients, chart=chart, degree=degree)


def _linear_coefficients(dimension: int, degree: int, /) -> Callable[[Array], Array]:
    slopes = jnp.arange(1.0, comb(dimension, degree) + 1.0, dtype=jnp.float64)

    def coefficients(point: Array) -> Array:
        return slopes * (1.0 + point[0] - 0.5 * point[-1]) + 0.25 * point[1]

    return coefficients


def _commutation_and_stokes(complex_: PreparedMeshfreeCellComplex, /) -> dict[str, Any]:
    """De Rham commutation R d = d R and the discrete Stokes/Green identity."""
    from phydrax.exterior import integrate_form, trace_map, validate_de_rham_commutation

    cochain, bridge = complex_.cochain, complex_.bridge
    top = cochain.dimension
    commutation = max(
        float(
            validate_de_rham_commutation(
                _cubic_form(bridge.chart, degree), bridge, tolerance=1e-11
            ).relative_residual
        )
        for degree in range(top)
    )
    form = integrate_form(_cubic_form(bridge.chart, top - 1), bridge).values
    derivative = cochain.exterior_derivative(top - 1, form)
    interior = jnp.sum(derivative)
    has_boundary = bool(np.any(np.asarray(cochain.boundary_masks[top - 1])))
    boundary = (
        jnp.sum(trace_map(cochain).maps[top - 1].mv(form))
        if has_boundary
        else jnp.zeros((), dtype=jnp.float64)
    )
    return {
        "commutation_relative_residual": commutation,
        "stokes_relative_defect": _relative(
            float(jnp.abs(interior - boundary)), float(jnp.sum(jnp.abs(derivative)))
        ),
        "stokes_has_boundary_trace": has_boundary,
    }


def _complex_identities(
    complex_: PreparedMeshfreeCellComplex, seed: int, /
) -> dict[str, Any]:
    """d∘d = 0 on random cochains and the codifferential as metric adjoint of d."""
    cochain = complex_.cochain
    counts = cochain.cell_counts
    top = cochain.dimension
    generator = np.random.default_rng(seed)
    nilpotency = 0.0
    for degree in range(top - 1):
        first = cochain.exterior_derivative(
            degree, jnp.asarray(generator.standard_normal(counts[degree]))
        )
        second = cochain.exterior_derivative(degree + 1, first)
        nilpotency = max(
            nilpotency,
            _relative(float(jnp.max(jnp.abs(second))), float(jnp.max(jnp.abs(first)))),
        )
    adjoint = 0.0
    for degree in range(top):
        u = jnp.asarray(generator.standard_normal(counts[degree]))
        v = jnp.asarray(generator.standard_normal(counts[degree + 1]))
        left = jnp.vdot(
            cochain.exterior_derivative(degree, u), cochain.hodge_star(degree + 1, v)
        )
        right = jnp.vdot(
            u, cochain.hodge_star(degree, cochain.codifferential(degree + 1, v))
        )
        adjoint = max(
            adjoint,
            _relative(
                float(jnp.abs(left - right)),
                max(float(jnp.abs(left)), float(jnp.abs(right))),
            ),
        )
    evidence = complex_.evidence
    return {
        "d_squared_relative": nilpotency,
        "codifferential_adjoint_relative_defect": adjoint,
        "hodge_admitted": all(bool(item.hodge_admitted) for item in evidence.degrees),
        "reproduction_residual": max(
            float(item.reproduction_residual) for item in evidence.degrees
        ),
        "maximum_patch_condition": max(
            float(item.maximum_condition) for item in evidence.degrees
        ),
        "rank_deficient_patches": sum(
            int(np.sum(np.asarray(item.rank_deficient_patches)))
            for item in evidence.degrees
        ),
        "admitted": bool(evidence.admitted),
        "geometry_authorized": evidence.fidelity == "geometry-authorized",
    }


def _hodge_patch_test(complex_: PreparedMeshfreeCellComplex, /) -> dict[str, Any]:
    """Linear forms: exact reconstruction and M_k integrating |ω|² (flat domains)."""
    from phydrax.exterior import integrate_form
    from phydrax.metrix import DifferentialForm

    cochain, bridge = complex_.cochain, complex_.bridge
    ambient = bridge.chart.dimension
    hodge, reconstruction = 0.0, 0.0
    for degree in range(cochain.dimension + 1):
        coefficients = _linear_coefficients(ambient, degree)
        values = integrate_form(
            DifferentialForm(coefficients, chart=bridge.chart, degree=degree), bridge
        )
        exact = jax.vmap(coefficients)(complex_.nodes)
        norm = float(jnp.sum(complex_.node_weights[:, None] * exact * exact))
        discrete = float(
            jnp.vdot(values.values, cochain.hodge_star(degree, values.values))
        )
        hodge = max(hodge, _relative(abs(discrete - norm), norm))
        reconstruction = max(
            reconstruction,
            _relative(
                float(jnp.max(jnp.abs(complex_.reconstruct(values) - exact))),
                float(jnp.max(jnp.abs(exact))),
            ),
        )
    return {
        "hodge_linear_norm_relative_defect": hodge,
        "linear_reconstruction_relative_error": reconstruction,
    }


def _harmonic_ranks(
    complex_: PreparedMeshfreeCellComplex, boundary: Literal["absolute", "relative"], /
) -> tuple[tuple[int, ...], bool, dict[str, Any]]:
    """Harmonic ranks, owner completeness, and the measured certificate residuals."""
    from phydrax.exterior import validate_harmonic_cohomology

    reports = tuple(
        validate_harmonic_cohomology(
            complex_.cochain, degree, boundary=boundary, tolerance=1e-7
        )[1]
        for degree in range(complex_.cochain.dimension + 1)
    )
    return (
        tuple(int(report.harmonic_rank) for report in reports),
        all(bool(report.complete) for report in reports),
        {
            f"harmonic_{boundary}_kernel_residual": max(
                float(jnp.max(report.kernel_residuals, initial=0.0)) for report in reports
            ),
            f"harmonic_{boundary}_orthonormality_residual": max(
                float(report.orthonormality_residual) for report in reports
            ),
            f"harmonic_{boundary}_incomplete_degrees": sum(
                not bool(report.complete) for report in reports
            ),
        },
    )


def _sampled_commutation(complex_: PreparedMeshfreeCellComplex, /) -> dict[str, Any]:
    """GMLS sample maps: max |d S₀ f − S₁ df| for a nonpolynomial potential."""
    nodes = complex_.nodes
    x, y = nodes[:, 0], nodes[:, 1]
    potential = jnp.sin(x) * jnp.cos(y)
    gradient = [jnp.cos(x) * jnp.cos(y), -jnp.sin(x) * jnp.sin(y)]
    if nodes.shape[1] == 3:
        z = nodes[:, 2]
        potential = potential + 0.3 * z * z
        gradient.append(0.6 * z)
    derived = complex_.cochain.exterior_derivative(
        0, complex_.sample(0, potential[:, None]).values
    )
    sampled = complex_.sample(1, jnp.stack(gradient, axis=1)).values
    error = float(jnp.max(jnp.abs(derived - sampled)))
    return {
        "sampled_commutation_error": error,
        "sampled_commutation_relative_error": _relative(
            error, float(jnp.max(jnp.abs(sampled)))
        ),
    }


def _hodge_laplace_consumer(
    complex_: PreparedMeshfreeCellComplex,
    betti: tuple[int, ...],
    recorder: PhaseRecorder,
    /,
) -> dict[str, Any]:
    """Mixed Hodge–Laplace at the top degree; original-equation residual."""
    from phydrax.linalg import harmonic_subspace, hodge_laplacian
    from phydrax.solver import HodgeLaplacePlan

    cochain = complex_.cochain
    top = cochain.dimension
    hilbert = cochain.hilbert_complex(boundary="absolute")
    harmonic = recorder.run(
        "rank-certificate",
        lambda: harmonic_subspace(hilbert, top, expected_dimension=betti[top]),
        scope="top-degree-harmonic-subspace",
    )
    plan = recorder.run(
        "assembly",
        lambda: HodgeLaplacePlan(
            cochain, top, boundary="absolute", formulation="mixed", harmonic=harmonic
        ),
        scope="mixed-hodge-laplace-plan",
    )
    operator = hodge_laplacian(hilbert, top)
    expected = jnp.sin(jnp.arange(hilbert.space(top).size, dtype=jnp.float64))
    source = operator.mv(expected)
    result = recorder.run(
        "solve", lambda: plan.solve(source), scope="mixed-hodge-laplace-solve"
    )
    residual = operator.mv(result.u) - source
    space = hilbert.space(top)
    return {
        "hodge_laplace_successful": bool(result.successful),
        "hodge_laplace_relative_residual": _relative(
            float(jnp.sqrt(space.inner(residual, residual))),
            float(jnp.sqrt(space.inner(source, source))),
        ),
    }


def _measure_higher_forms(
    geometry: HigherFormGeometry, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    import examples.meshfree_higher_forms as consumer
    from phydrax.discretization.meshfree import (
        ComplexGeometryAuthority,
        MeshfreeCellComplexPlan,
        MeshfreeComplexPolicy,
    )

    level = higher_form_level(geometry, capacity)
    top_cells = higher_form_top_cells(geometry, level)
    if top_cells > config.max_points:
        raise DeclaredCapacityRefusal(
            f"{top_cells} top cells exceed the declared max_points={config.max_points}."
        )
    policy = MeshfreeComplexPolicy()
    # Declared before work: the owner-admitted local patch workset bound
    # (float64 entries) that patch growth refuses to exceed.
    reserved = declare_reservation(
        8 * policy.maximum_workset_entries,
        config,
        scope="higher-form local patch workset",
    )
    recorder = PhaseRecorder()
    mesh, identity = recorder.run(
        "geometry",
        lambda: _higher_form_complex(geometry, level),
        scope="authoritative-mesh",
    )
    authority = recorder.run(
        "geometry",
        lambda: ComplexGeometryAuthority(mesh, identity),
        scope="orientation-measure-betti-audit",
    )
    ambient = mesh.ambient_dimension
    complex_ = recorder.run(
        "local-fit",
        lambda: MeshfreeCellComplexPlan(
            authority, chart=consumer.chart_for(ambient), policy=policy
        ).prepare(),
        scope="patch-growth-gmls-fit-sparse-hodge-admission",
    )
    recorder.unavailable(
        "search",
        "Facet-adjacency patch growth is fused into PreparedMeshfreeCellComplex "
        "preparation (recorded as local-fit)",
    )
    cochain = complex_.cochain
    counts = cochain.cell_counts
    generator = np.random.default_rng(seed)
    _, compiled = recorder.compiled_action(
        lambda values: cochain.hodge_star(1, values),
        jnp.asarray(generator.standard_normal(counts[1])),
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
        scope="hodge-star-degree-1",
    )
    absolute, absolute_complete, absolute_residuals = recorder.run(
        "rank-certificate",
        lambda: _harmonic_ranks(complex_, "absolute"),
        scope="absolute-harmonic-cohomology",
    )
    detail: dict[str, Any] = {
        "geometry": geometry,
        "mesh_size_provenance": "maximum edge length of the authoritative mesh",
        "betti_declared": list(identity.betti),
        "harmonic_absolute_ranks": list(absolute),
    }
    metrics: dict[str, Any] = {
        "level": level,
        "top_cells": counts[-1],
        **{f"cells_degree_{degree}": count for degree, count in enumerate(counts)},
        "sample_nodes": int(complex_.nodes.shape[0]),
        "mesh_size_h": float(jnp.max(authority.measures[1])),
        "harmonic_matches_betti": absolute_complete and absolute == identity.betti,
        **absolute_residuals,
        "maximum_patch_cells": policy.maximum_patch_cells,
        "maximum_patch_entities": policy.maximum_patch_entities,
        "maximum_workset_entries": policy.maximum_workset_entries,
        **_commutation_and_stokes(complex_),
        **_complex_identities(complex_, seed),
    }
    if identity.role == "domain":
        relative, relative_complete, relative_residuals = recorder.run(
            "rank-certificate",
            lambda: _harmonic_ranks(complex_, "relative"),
            scope="relative-harmonic-cohomology",
        )
        # Poincaré–Lefschetz: relative harmonic dimension k equals b_{n-k}.
        detail["harmonic_relative_ranks"] = list(relative)
        metrics["harmonic_relative_matches_lefschetz"] = (
            relative_complete and relative == tuple(reversed(identity.betti))
        )
        metrics.update(relative_residuals)
        metrics.update(_hodge_patch_test(complex_))
    metrics.update(
        recorder.run(
            "transfer", lambda: _sampled_commutation(complex_), scope="gmls-sample-maps"
        )
    )
    metrics.update(_hodge_laplace_consumer(complex_, identity.betti, recorder))
    if geometry == "tunneled-slab":
        report = recorder.run(
            "solve",
            lambda: consumer.maxwell_flux_evolution(complex_, steps=_MAXWELL_STEPS),
            scope="compatible-maxwell-leapfrog-degree-2-3",
        )
        metrics["maxwell_steps"] = report.steps
        metrics["maxwell_magnetic_divergence"] = report.magnetic_divergence
        metrics["maxwell_energy_drift"] = report.energy_drift
    return {
        "workload": f"higher-forms-{geometry}",
        "capacity": counts[-1],
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": ambient,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "detail": detail,
        "compiler": compiled,
        "reserved_working_set_bytes": reserved,
        "retained_bytes": logical_array_bytes(complex_),
        "oracle_provenance": "exact order-6 simplex cubature of cubic/linear forms, exact "
        "rational Betti numbers, integer incidence, discrete Green identity, mixed "
        "Hodge-Laplace original-equation residual",
        "consumer": "phydrax.discretization.meshfree.MeshfreeCellComplexPlan -> "
        "PreparedMeshfreeCellComplex.cochain/.bridge",
    }


def measure_higher_forms_holed_square(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _measure_higher_forms("holed-square", capacity, seed, config)


def measure_higher_forms_tunneled_slab(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _measure_higher_forms("tunneled-slab", capacity, seed, config)


def measure_higher_forms_icosphere(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _measure_higher_forms("icosphere", capacity, seed, config)


def _noisy_circle(count: int, seed: int, /) -> tuple[np.ndarray, float]:
    """Seeded circle samples: gaps stay in [0.8, 1.2] of the mean arc spacing."""
    if count < 8:
        raise ValueError("The abstract clique circle needs at least eight samples.")
    generator = np.random.default_rng(seed)
    spacing = 2.0 * pi / count
    angles = spacing * (np.arange(count) + generator.uniform(-0.1, 0.1, count))
    radii = 1.0 + generator.uniform(-0.002, 0.002, count) * spacing
    return np.stack((radii * np.cos(angles), radii * np.sin(angles)), axis=1), spacing


def _clique_policy(count: int, /) -> Any:
    from phydrax.discretization.meshfree import RadiusCliquePolicy

    # With radius 1.8 s and gaps in [0.8 s, 1.2 s], every vertex has at most
    # two neighbors per side: <= 2N edges, <= N triangles, no 4-cliques.
    return RadiusCliquePolicy(
        maximum_pairs=4 * count,
        maximum_dimension=2,
        maximum_simplices=8 * count,
        maximum_work=64 * count,
    )


def _boundary_composition_nonzeros(complex_: RadiusCliqueComplex, /) -> int:
    boundaries = complex_.chain.boundaries
    return sum(
        lower.compose(upper).nonzero_count
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def measure_abstract_clique_forms(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Research-only radius-clique complex of a sampled circle (no geometry authority)."""
    from phydrax.discretization.meshfree import radius_clique_complex

    if capacity > config.max_points:
        raise DeclaredCapacityRefusal(
            f"{capacity} clique vertices exceed the declared max_points={config.max_points}."
        )
    points, spacing = _noisy_circle(capacity, seed)
    policy = _clique_policy(capacity)
    recorder = PhaseRecorder()
    complex_ = recorder.run(
        "search",
        lambda: radius_clique_complex(points, 1.8 * spacing, policy=policy),
        scope="radius-relation-clique-enumeration-exact-betti",
    )
    recorder.unavailable(
        "rank-certificate",
        "Exact rational Betti numbers are fused into radius_clique_complex (search)",
    )
    top = complex_.topology.dimension
    betti = tuple(complex_.betti.dimension(degree) for degree in range(top + 1))
    # A densely sampled circle: b0 = b1 = 1 and every higher Betti number zero.
    expected = (1, 1) + (0,) * (top - 1)
    return {
        "workload": "abstract-clique-forms",
        "capacity": capacity,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "betti_matches_circle": betti == expected,
            "boundary_composition_nonzeros": _boundary_composition_nonzeros(complex_),
            "abstract_research_fidelity": complex_.fidelity == "abstract-research",
            "simplex_count_total": sum(complex_.simplex_counts),
            "clique_work": complex_.work,
            "maximum_pairs": policy.maximum_pairs,
            "maximum_simplices": policy.maximum_simplices,
            "maximum_work": policy.maximum_work,
            "work_within_declared": complex_.work <= policy.maximum_work,
        },
        "detail": {"betti": list(betti), "simplex_counts": list(complex_.simplex_counts)},
        "retained_bytes": logical_array_bytes(complex_),
        "oracle_provenance": "homotopy type of a 1.8-spacing Rips complex of a circle "
        "sample (b = (1, 1, 0)); exact integer boundary composition",
        "consumer": "phydrax.discretization.meshfree.radius_clique_complex",
    }


def measure_higher_forms_refusals(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Abstract clique, measure/Betti/orientation audits and patch capacity refusals."""
    import examples.meshfree_higher_forms as consumer
    from phydrax.discretization.meshfree import (
        ComplexDomainIdentity,
        ComplexGeometryAuthority,
        MeshfreeCellComplexPlan,
        MeshfreeComplexAdmissionError,
        MeshfreeComplexPolicy,
        radius_clique_complex,
    )

    level = higher_form_level("holed-square", capacity)
    holed = consumer.planar_grid_mesh(level, hole=True)
    filled = consumer.planar_grid_mesh(level, hole=False)
    declared = ComplexDomainIdentity(
        "square-with-hole", "domain", measure=8.0, betti=(1, 1, 0)
    )
    points, spacing = _noisy_circle(32, seed)
    clique = radius_clique_complex(points, 1.8 * spacing, policy=_clique_policy(32))
    chart = consumer.chart_for(2)
    counts = holed.topology.entity_sets
    flips = np.ones(counts[2].count)
    flips[0] = -1.0
    orientation = (np.ones(counts[0].count), np.ones(counts[1].count), flips)
    recorder = PhaseRecorder()

    def check(scope: str, operation: Callable[[], bool]) -> bool:
        return recorder.run("geometry", operation, scope=scope)

    metrics = {
        "abstract_clique_plan_refused": check(
            "abstract-clique-as-plan-authority",
            lambda: _refused(
                lambda: MeshfreeCellComplexPlan(clique, chart=chart),  # ty: ignore[invalid-argument-type]
                TypeError,
                "research-only",
            ),
        ),
        "abstract_clique_authority_refused": check(
            "abstract-topology-as-geometry-authority",
            lambda: _refused(
                lambda: ComplexGeometryAuthority(clique.topology, declared),  # ty: ignore[invalid-argument-type]
                TypeError,
                "requires a CellMesh",
            ),
        ),
        "measure_audit_refused": check(
            "filled-square-claims-holed-measure",
            lambda: _refused(
                lambda: ComplexGeometryAuthority(
                    filled,
                    ComplexDomainIdentity(
                        "square-with-hole", "domain", measure=8.0, betti=(1, 0, 0)
                    ),
                ),
                MeshfreeComplexAdmissionError,
                "does not represent declared domain",
            ),
        ),
        "betti_audit_refused": check(
            "filled-square-claims-holed-betti",
            lambda: _refused(
                lambda: ComplexGeometryAuthority(
                    filled,
                    ComplexDomainIdentity(
                        "square-with-hole", "domain", measure=9.0, betti=(1, 1, 0)
                    ),
                ),
                MeshfreeComplexAdmissionError,
                "Betti numbers",
            ),
        ),
        "orientation_audit_refused": check(
            "single-reversed-top-cell",
            lambda: _refused(
                lambda: ComplexGeometryAuthority(
                    holed, declared, orientation=orientation
                ),
                MeshfreeComplexAdmissionError,
                "incoherent",
            ),
        ),
    }
    authority = ComplexGeometryAuthority(holed, declared)
    metrics["patch_capacity_refused"] = recorder.run(
        "local-fit",
        lambda: _refused(
            lambda: MeshfreeCellComplexPlan(
                authority,
                chart=chart,
                policy=MeshfreeComplexPolicy(maximum_patch_entities=4),
            ).prepare(),
            MeshfreeComplexAdmissionError,
            "maximum_patch_entities",
        ),
        scope="patch-entity-capacity",
    )
    return {
        "workload": "higher-forms-refusals",
        "capacity": higher_form_top_cells("holed-square", level),
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "oracle_provenance": "declared domain identity (measure 8, Betti (1,1,0)) versus a "
        "hole-filling mesh; one reversed top cell; owner-documented refusal types",
        "consumer": "phydrax.discretization.meshfree.ComplexGeometryAuthority / "
        "MeshfreeCellComplexPlan",
    }


def measure_abstract_clique_refusals(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Simplex and work capacities refuse before enumeration exceeds them."""
    from phydrax.discretization.meshfree import (
        MeshfreeComplexAdmissionError,
        radius_clique_complex,
        RadiusCliquePolicy,
    )

    points, spacing = _noisy_circle(capacity, seed)
    recorder = PhaseRecorder()
    metrics = {
        "simplex_capacity_refused": recorder.run(
            "search",
            lambda: _refused(
                lambda: radius_clique_complex(
                    points,
                    1.8 * spacing,
                    policy=RadiusCliquePolicy(
                        maximum_pairs=4 * capacity, maximum_simplices=capacity + 1
                    ),
                ),
                MeshfreeComplexAdmissionError,
                "maximum_simplices",
            ),
            scope="simplex-capacity",
        ),
        "work_capacity_refused": recorder.run(
            "search",
            lambda: _refused(
                lambda: radius_clique_complex(
                    points,
                    1.8 * spacing,
                    policy=RadiusCliquePolicy(
                        maximum_pairs=4 * capacity, maximum_work=capacity
                    ),
                ),
                MeshfreeComplexAdmissionError,
                "maximum_work",
            ),
            scope="work-capacity",
        ),
    }
    return {
        "workload": "abstract-clique-refusals",
        "capacity": capacity,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "oracle_provenance": "every circle vertex carries at least one edge, so N+1 "
        "simplices and N work units are exceeded",
        "consumer": "phydrax.discretization.meshfree.radius_clique_complex",
    }


# ---------------------------------------------------------------------------
# Q14 sensitivities


_SUPPORT_RADIUS = 3.0  # Smooth support radius in lattice spacings.
_ENVELOPE = 0.1  # Admitted displacement envelope in lattice spacings.


def _jittered_lattice(
    capacity: int, dimension: int, seed: int, /
) -> tuple[np.ndarray, float]:
    width = max(4, round(capacity ** (1.0 / dimension)))
    axis = np.linspace(0.0, 1.0, width)
    grid = np.stack(np.meshgrid(*(axis,) * dimension, indexing="ij"), axis=-1)
    points = grid.reshape((-1, dimension))
    spacing = 1.0 / (width - 1)
    jitter = np.random.default_rng(seed).uniform(-0.15, 0.15, points.shape)
    return points + jitter * spacing, spacing


def _candidate_capacity(points: np.ndarray, radius: float, /) -> int:
    """Exact host count of the largest closed candidate ball (independent KD-tree)."""
    counts = cKDTree(points).query_ball_point(points, radius, return_length=True)
    return int(np.max(counts))


def _smooth_point_cloud(
    capacity: int, seed: int, config: MeshfreeConfig, recorder: PhaseRecorder, /
) -> tuple[PreparedPointCloudDiscretization, np.ndarray, np.ndarray, float, int]:
    from phydrax.discretization import PointCloudPlan
    from phydrax.discretization.meshfree import LocalStencilPolicy, SmoothSupportEnvelope

    points, spacing = _jittered_lattice(capacity, config.dimension, seed)
    support = SmoothSupportEnvelope(_SUPPORT_RADIUS * spacing, _ENVELOPE * spacing)
    candidates = _candidate_capacity(points, support.candidate_radius)
    config.check_capacity(points.shape[0], degree=config.degree, neighbors=candidates)
    cloud = recorder.run(
        "local-fit",
        lambda: PointCloudPlan(
            points,
            np.full(points.shape[0], 1.0 / points.shape[0]),
            stencil=LocalStencilPolicy(support=support, polynomial_degree=config.degree),
            neighbors=candidates,
        ).prepare(),
        scope="smooth-envelope-neighborhood-and-gmls-fit",
    )
    recorder.unavailable(
        "search",
        "Envelope candidate search is fused into PointCloudPlan.prepare (local-fit)",
    )
    direction = np.random.default_rng(seed + 1).normal(size=points.shape)
    direction *= 0.5 * support.displacement / np.max(np.linalg.norm(direction, axis=1))
    return cloud, points, direction, spacing, candidates


def _stencil_weights(
    cloud: PreparedPointCloudDiscretization, points: np.ndarray, /
) -> tuple[Array, ...]:
    return tuple(item for _, item in cloud.refresh(points).discretization.mixed_weights)


def measure_fixed_support_sensitivity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Smooth fixed-support coordinate JVP vs central differences and VJP duality."""
    from phydrax.discretization.meshfree import LocalStencilRefreshStatus

    recorder = PhaseRecorder()
    cloud, points, direction, spacing, candidates = _smooth_point_cloud(
        capacity, seed, config, recorder
    )
    moved = points + 0.5 * direction
    refreshed = recorder.run(
        "numeric-refresh",
        lambda: _stencil_weights(cloud, moved),
        scope="fixed-support-refresh",
    )
    sensitivity = recorder.run(
        "numeric-refresh",
        lambda: cloud.coordinate_sensitivity(moved),
        scope="coordinate-linearization",
    )
    tangent = recorder.run_repeated(
        "jvp",
        lambda: sensitivity.linearization.jvp(jnp.asarray(direction)),
        repeats=config.repeats,
        scope="coordinate-jvp",
    )
    generator = np.random.default_rng(seed + 2)
    cotangent = tuple(jnp.asarray(generator.normal(size=item.shape)) for item in tangent)
    pulled = recorder.run_repeated(
        "vjp",
        lambda: sensitivity.linearization.vjp(cotangent),
        repeats=config.repeats,
        scope="coordinate-vjp",
    )
    # Central difference of the published refresh map; step 1e-3 of a 0.5-envelope
    # motion stays strictly inside the admitted envelope.
    step = 1e-3
    ahead = _stencil_weights(cloud, points + (0.5 + step) * direction)
    behind = _stencil_weights(cloud, points - (step - 0.5) * direction)
    scale = max(float(jnp.max(jnp.abs(item))) for item in tangent)
    difference = max(
        float(jnp.max(jnp.abs(derivative - (plus - minus) / (2.0 * step))))
        for derivative, plus, minus in zip(tangent, ahead, behind, strict=True)
    )
    forward = sum(float(jnp.vdot(c, t)) for c, t in zip(cotangent, tangent, strict=True))
    reverse = float(jnp.vdot(pulled, jnp.asarray(direction)))
    primal = sensitivity.linearization.primal
    identity = max(
        _relative(float(jnp.max(jnp.abs(a - b))), float(jnp.max(jnp.abs(b))))
        for a, b in zip(primal, refreshed, strict=True)
    )
    evidence = sensitivity.evidence
    if not isinstance(evidence, LocalStencilEvidence):
        raise TypeError("Fixed-support sensitivity requires LocalStencilEvidence.")
    basis = comb(config.dimension + config.degree, config.degree)
    probes = unit_cube_probes(config.dimension, 4 * points.shape[0], seed)
    return {
        "workload": "fixed-support-sensitivity",
        "capacity": points.shape[0],
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "accepted": bool(sensitivity.accepted),
            "status_accepted": int(sensitivity.status)
            == int(LocalStencilRefreshStatus.ACCEPTED),
            "rank_full": bool(np.all(np.asarray(evidence.rank) == basis)),
            "maximum_condition": float(np.max(np.asarray(evidence.condition))),
            "jvp_fd_relative_error": _relative(difference, scale),
            "vjp_duality_relative_error": _relative(
                abs(forward - reverse), max(abs(forward), abs(reverse))
            ),
            "linearization_primal_identity_error": identity,
            "candidate_capacity": candidates,
            "support_radius": _SUPPORT_RADIUS * spacing,
            "envelope_displacement": _ENVELOPE * spacing,
        },
        "fill": fill_distance(
            points,
            probes,
            provenance="max nearest-sample distance over scrambled Sobol probes of "
            "[0,1]^d (a lower estimate of the supremum)",
        ),
        "retained_bytes": logical_array_bytes(cloud),
        "oracle_provenance": "central differences of the published fixed-support refresh "
        "map and inner-product duality <c, J t> = <J^T c, t>",
        "consumer": "phydrax.discretization.PreparedPointCloudDiscretization."
        "coordinate_sensitivity",
    }


def measure_fixed_support_envelope_exit(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Motion beyond the admitted envelope: SUPPORT_EXCEEDED with NaN derivatives."""
    from phydrax.discretization.meshfree import LocalStencilRefreshStatus

    recorder = PhaseRecorder()
    cloud, points, direction, _, _ = _smooth_point_cloud(capacity, seed, config, recorder)
    beyond = recorder.run(
        "numeric-refresh",
        lambda: cloud.coordinate_sensitivity(points + 3.0 * direction),
        scope="linearization-beyond-envelope",
    )
    tangent = recorder.run(
        "jvp",
        lambda: beyond.linearization.jvp(jnp.asarray(direction)),
        scope="refused-jvp",
    )
    cotangent = tuple(jnp.ones_like(item) for item in tangent)
    pulled = recorder.run(
        "vjp", lambda: beyond.linearization.vjp(cotangent), scope="refused-vjp"
    )
    return {
        "workload": "fixed-support-envelope-exit",
        "capacity": points.shape[0],
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "support_exceeded": int(beyond.status)
            == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED),
            "not_accepted": not bool(beyond.accepted),
            "primal_nan": all(
                bool(jnp.all(jnp.isnan(item))) for item in beyond.linearization.primal
            ),
            "jvp_nan": all(bool(jnp.all(jnp.isnan(item))) for item in tangent),
            "vjp_nan": bool(jnp.all(jnp.isnan(pulled))),
        },
        "oracle_provenance": "maximum displacement 1.5x the declared envelope",
        "consumer": "phydrax.discretization.PreparedPointCloudDiscretization."
        "coordinate_sensitivity",
    }


def _quadratic_basis(points: np.ndarray, /) -> np.ndarray:
    dimension = points.shape[1]
    columns = [np.ones(points.shape[0])]
    columns.extend(points[:, axis] for axis in range(dimension))
    columns.extend(
        points[:, first] * points[:, second]
        for first, second in combinations_with_replacement(range(dimension), 2)
    )
    return np.stack(columns, axis=1)


def _crossing_oracle(sources: np.ndarray, radius: float, /) -> np.ndarray:
    """d/dx₀ GMLS weights at the origin: Wendland-C2 profile, NumPy normal equations."""
    scaled = sources / radius
    ratio = np.linalg.norm(scaled, axis=1)
    inside = ratio < 1.0
    r = np.where(inside, ratio, 1.0)
    weight = np.where(inside, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)
    basis = _quadratic_basis(scaled)
    moments = np.zeros(basis.shape[1])
    moments[1] = 1.0 / radius
    coefficients = np.linalg.solve(basis.T @ (weight[:, None] * basis), moments)
    return weight * (basis @ coefficients)


def _crossing_sources(
    capacity: int, dimension: int, seed: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed sources inside 0.85 R plus one source crossing |x| = R obliquely at t = 0."""
    basis = comb(dimension + 2, 2)
    if capacity - 1 < 2 * basis:
        raise ValueError(
            f"The crossing workload needs at least {2 * basis + 1} sources in {dimension}-D."
        )
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(capacity - 1, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = 0.85 * generator.uniform(0.05, 1.0, capacity - 1) ** (1.0 / dimension)
    anchor = np.zeros(dimension)
    anchor[0] = 1.0
    velocity = np.zeros(dimension)
    velocity[0] = -0.8
    if dimension > 1:
        velocity[1] = 0.6
    else:
        velocity[0] = -1.0
    return directions * radii[:, None], anchor, velocity


def _crossing_field(points: Array, /) -> Array:
    return jnp.exp(0.7 * points[:, 0]) * jnp.cos(points[:, -1] + 0.3 * points[:, 0])


def _crossing_quadratic(points: Array, /) -> Array:
    x, y = points[:, 0], points[:, -1]
    if points.shape[1] == 1:
        return 1.0 - 2.0 * x + 3.0 * x * x
    return 1.0 - 2.0 * x + 0.5 * y + 3.0 * x * x - x * y + 0.25 * y * y


def _prepare_crossing(
    sources: np.ndarray,
    target: np.ndarray,
    displacement: float,
    recorder: PhaseRecorder,
    /,
) -> PreparedLocalStencils:
    from phydrax.discretization.meshfree import (
        LocalStencilPolicy,
        MeshfreeFunctional,
        MeshfreeNeighborhoodPlan,
        prepare_local_stencils,
        SmoothSupportEnvelope,
    )

    support = SmoothSupportEnvelope(1.0, displacement)
    dimension = sources.shape[1]
    neighborhood = recorder.run(
        "search",
        lambda: MeshfreeNeighborhoodPlan(
            sources, sources.shape[0], targets=target, envelope=support
        ).prepare(),
        scope="smooth-envelope-candidates",
    )
    multi_index = tuple(1 if axis == 0 else 0 for axis in range(dimension))
    return recorder.run(
        "local-fit",
        lambda: prepare_local_stencils(
            neighborhood,
            sources,
            target,
            (MeshfreeFunctional((multi_index,), (1.0,)),),
            LocalStencilPolicy(
                weight_kernel="wendland-c2",
                support=support,
                coordinate_order=1,
                polynomial_degree=2,
            ),
        ),
        scope="wendland-c2-gmls-fit",
    )


def _by_source(stencils: PreparedLocalStencils, row: Array, /) -> Array:
    relation = stencils.neighborhood.relation
    return (
        jnp.zeros((relation.source_size,), dtype=row.dtype)
        .at[relation.source_indices[0]]
        .add(jnp.where(relation.valid[0], row, 0.0))
    )


def measure_smooth_support_crossing(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A neighbor crossing the compact cutoff inside the envelope: two-sided derivatives."""
    from phydrax.discretization.meshfree import refresh_local_stencils

    dimension = config.dimension
    fixed, anchor, velocity = _crossing_sources(capacity, dimension, seed)
    target = np.zeros((1, dimension))
    initial = np.concatenate((fixed, anchor[None]))
    recorder = PhaseRecorder()
    stencils = _prepare_crossing(initial, target, 0.2, recorder)

    def sources(t: Array | float) -> Array:
        return jnp.concatenate(
            (jnp.asarray(fixed), (jnp.asarray(anchor) + t * jnp.asarray(velocity))[None])
        )

    def weights(t: Array) -> Array:
        refreshed = refresh_local_stencils(stencils, sources(t), target)
        return _by_source(stencils, refreshed.stencils.weights[0][0])

    def derivative(t: Array) -> Array:
        return weights(t) @ _crossing_field(sources(t))

    def oracle(t: float) -> np.ndarray:
        return _crossing_oracle(np.asarray(sources(t)), 1.0)

    def oracle_derivative(t: float) -> float:
        moved = sources(t)
        return float(oracle(t) @ np.asarray(_crossing_field(moved)))

    recorder.run(
        "numeric-refresh",
        lambda: refresh_local_stencils(stencils, sources(0.01), target),
        scope="crossing-refresh",
    )
    _, tangent_at_cutoff = recorder.run_repeated(
        "jvp",
        lambda: jax.jvp(weights, (jnp.asarray(0.0),), (jnp.asarray(1.0),)),
        repeats=config.repeats,
        scope="cutoff-jvp",
    )
    value = weights(jnp.asarray(0.0))
    # Duality is measured strictly inside the support, where the tangent is nonzero.
    _, inside_tangent = jax.jvp(weights, (jnp.asarray(0.01),), (jnp.asarray(1.0),))
    _, pullback = jax.vjp(weights, jnp.asarray(0.01))
    cotangent = jnp.asarray(np.random.default_rng(seed + 3).normal(size=value.shape))
    (pulled,) = recorder.run_repeated(
        "vjp",
        lambda: pullback(cotangent),
        repeats=config.repeats,
        scope="inside-support-vjp",
    )
    # Outside the cutoff the map is locally constant (zero derivative), so
    # derivative errors are relative to the largest oracle derivative on the
    # trajectory, not to a vanishing local value.
    value_error, tangent_error, tangent_scale, rank_full, accepted = (
        0.0,
        0.0,
        0.0,
        True,
        True,
    )
    basis = comb(dimension + 2, 2)
    step = 1e-5
    for t in (-0.15, -0.01, -1e-3, 1e-3, 0.01, 0.15):
        refreshed = refresh_local_stencils(stencils, sources(t), target)
        accepted = accepted and bool(refreshed.accepted)
        rank_full = rank_full and bool(
            np.all(np.asarray(refreshed.stencils.evidence.rank) == basis)
        )
        current, tangent = jax.jvp(weights, (jnp.asarray(t),), (jnp.asarray(1.0),))
        exact = oracle(t)
        central = (oracle(t + step) - oracle(t - step)) / (2.0 * step)
        value_error = max(
            value_error,
            _relative(
                float(np.max(np.abs(current - exact))), float(np.max(np.abs(exact)))
            ),
        )
        tangent_error = max(tangent_error, float(np.max(np.abs(tangent - central))))
        tangent_scale = max(tangent_scale, float(np.max(np.abs(central))))
    # Second-order one-sided oracle differences from inside and outside at the cutoff.
    h = 1e-4
    left = (3.0 * oracle(0.0) - 4.0 * oracle(-h) + oracle(-2.0 * h)) / (2.0 * h)
    right = (-3.0 * oracle(0.0) + 4.0 * oracle(h) - oracle(2.0 * h)) / (2.0 * h)
    two_sided = max(
        float(np.max(np.abs(tangent_at_cutoff - left))),
        float(np.max(np.abs(tangent_at_cutoff - right))),
    )
    field_error, field_scale = 0.0, 0.0
    for t in (-0.12, -2e-3, 2e-3, 0.12):
        _, field_tangent = jax.jvp(derivative, (jnp.asarray(t),), (jnp.asarray(1.0),))
        central_field = (oracle_derivative(t + step) - oracle_derivative(t - step)) / (
            2.0 * step
        )
        field_error = max(field_error, abs(float(field_tangent) - central_field))
        field_scale = max(field_scale, abs(central_field))
    quadratic_error = max(
        abs(float(weights(jnp.asarray(t)) @ _crossing_quadratic(sources(t))) + 2.0) / 2.0
        for t in (-0.12, -2e-3, 0.0, 2e-3, 0.12)
    )
    forward = float(jnp.vdot(cotangent, inside_tangent))
    return {
        "workload": "smooth-support-crossing",
        "capacity": initial.shape[0],
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "accepted_along_trajectory": accepted,
            "rank_full_along_trajectory": rank_full,
            "cutoff_weight_vanishes": float(value[-1]) == 0.0,
            "weight_oracle_relative_error": value_error,
            "weight_tangent_relative_error": _relative(tangent_error, tangent_scale),
            "cutoff_two_sided_relative_error": _relative(two_sided, tangent_scale),
            "field_derivative_tangent_relative_error": _relative(
                field_error, field_scale
            ),
            "quadratic_exactness_relative_error": quadratic_error,
            "vjp_duality_relative_error": _relative(
                abs(forward - float(pulled)), max(abs(forward), abs(float(pulled)))
            ),
        },
        "retained_bytes": logical_array_bytes(stencils),
        "oracle_provenance": "NumPy normal-equation GMLS with the closed-form Wendland-C2 "
        "profile; central and one-sided second-order differences of that oracle",
        "consumer": "phydrax.discretization.meshfree.refresh_local_stencils with "
        "LocalStencilPolicy(support=SmoothSupportEnvelope)",
    }


def measure_smooth_support_envelope_exit(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """An outsider entering the support beyond the declared envelope is refused."""
    from phydrax.discretization.meshfree import (
        LocalStencilRefreshStatus,
        refresh_local_stencils,
    )

    dimension = config.dimension
    fixed, _, _ = _crossing_sources(capacity, dimension, seed)
    outsider = np.zeros(dimension)
    outsider[-1] = -1.15  # Outside the 1.1 candidate radius of a 0.05 envelope.
    sources = np.concatenate((fixed, outsider[None]))
    target = np.zeros((1, dimension))
    recorder = PhaseRecorder()
    stencils = _prepare_crossing(sources, target, 0.05, recorder)
    direction = np.zeros_like(sources)
    direction[-1, -1] = 1.0

    def weights(t: Array) -> Array:
        moved = jnp.asarray(sources) + t * jnp.asarray(direction)
        return refresh_local_stencils(stencils, moved, target).stencils.weights[0]

    refreshed = recorder.run(
        "numeric-refresh",
        lambda: refresh_local_stencils(stencils, sources + 0.2 * direction, target),
        scope="motion-beyond-envelope",
    )
    value, tangent = recorder.run(
        "jvp",
        lambda: jax.jvp(weights, (jnp.asarray(0.2),), (jnp.asarray(1.0),)),
        scope="refused-jvp",
    )
    _, pullback = jax.vjp(weights, jnp.asarray(0.2))
    (pulled,) = recorder.run(
        "vjp", lambda: pullback(jnp.ones_like(value)), scope="refused-vjp"
    )
    inside, inside_tangent = jax.jvp(weights, (jnp.asarray(0.04),), (jnp.asarray(1.0),))
    return {
        "workload": "smooth-support-envelope-exit",
        "capacity": sources.shape[0],
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "outsider_not_candidate": not bool(
                np.asarray(stencils.neighborhood.relation.valid)[0, -1]
            ),
            "support_exceeded": int(refreshed.status)
            == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED),
            "value_nan": bool(jnp.all(jnp.isnan(value))),
            "jvp_nan": bool(jnp.all(jnp.isnan(tangent))),
            "vjp_nan": bool(jnp.isnan(pulled)),
            "inside_envelope_finite": bool(
                jnp.all(jnp.isfinite(inside)) and jnp.all(jnp.isfinite(inside_tangent))
            ),
        },
        "oracle_provenance": "an outsider at 1.15 R enters the R-support after a 0.2 "
        "motion that exceeds the 0.05 envelope",
        "consumer": "phydrax.discretization.meshfree.refresh_local_stencils",
    }


def measure_knn_tie_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A kNN anchor on a selection tie: NaN tangent; crossing it refuses with NaN."""
    from phydrax.discretization.meshfree import (
        LocalStencilPolicy,
        LocalStencilRefreshStatus,
        MeshfreeFunctional,
        MeshfreeNeighborhoodPlan,
        prepare_local_stencils,
        refresh_local_stencils,
    )

    dimension = config.dimension
    # Distinct distances 0.9.., 1.0, 1.1, then an exact tie at 2 (+/- e0):
    # the k-th and (k+1)-th sources tie, so the anchor has no selection gap.
    near = [0.9 + 0.02 * axis for axis in range(1, dimension)]
    rows = [np.eye(dimension)[axis] * near[axis - 1] for axis in range(1, dimension)]
    rows += [np.eye(dimension)[0], -1.1 * np.eye(dimension)[0]]
    neighbors = len(rows) + 1
    rows += [2.0 * np.eye(dimension)[0], -2.0 * np.eye(dimension)[0]]
    if capacity < len(rows):
        raise ValueError(f"The kNN tie workload needs at least {len(rows)} sources.")
    rows += [
        (3.5 + 0.5 * index) * np.eye(dimension)[0]
        for index in range(capacity - len(rows))
    ]
    sources = np.asarray(rows, dtype=np.float64)
    targets = np.zeros((1, dimension))
    recorder = PhaseRecorder()
    neighborhood = recorder.run(
        "search",
        lambda: MeshfreeNeighborhoodPlan(sources, neighbors, targets=targets).prepare(),
        scope="knn-selection",
    )
    multi_index = tuple(1 if axis == 0 else 0 for axis in range(dimension))
    stencils = recorder.run(
        "local-fit",
        lambda: prepare_local_stencils(
            neighborhood,
            sources,
            targets,
            (MeshfreeFunctional((multi_index,), (1.0,)),),
            LocalStencilPolicy(polynomial_degree=1),
        ),
        scope="degree-1-fit",
    )
    shift = np.zeros(dimension)
    shift[0] = 1.0

    def weights(t: Array) -> Array:
        moved = jnp.asarray(targets) + t * jnp.asarray(shift)
        return refresh_local_stencils(stencils, sources, moved).stencils.weights[0]

    anchored, anchor_tangent = recorder.run(
        "jvp",
        lambda: jax.jvp(weights, (jnp.asarray(0.0),), (jnp.asarray(1.0),)),
        scope="tie-anchor",
    )
    crossed = recorder.run(
        "numeric-refresh",
        lambda: refresh_local_stencils(stencils, sources, targets + 1e-9 * shift),
        scope="tie-crossing",
    )
    value, tangent = jax.jvp(weights, (jnp.asarray(1e-9),), (jnp.asarray(1.0),))
    _, pullback = jax.vjp(weights, jnp.asarray(1e-9))
    (pulled,) = recorder.run(
        "vjp", lambda: pullback(jnp.ones_like(value)), scope="tie-crossing"
    )
    return {
        "workload": "knn-tie-refusal",
        "capacity": sources.shape[0],
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "anchor_value_finite": bool(jnp.all(jnp.isfinite(anchored))),
            "anchor_tangent_nan": bool(jnp.all(jnp.isnan(anchor_tangent))),
            "crossing_support_exceeded": int(crossed.status)
            == int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED),
            "crossing_value_nan": bool(jnp.all(jnp.isnan(value))),
            "crossing_jvp_nan": bool(jnp.all(jnp.isnan(tangent))),
            "crossing_vjp_nan": bool(jnp.isnan(pulled)),
        },
        "oracle_provenance": "exact distance tie at the k-th/(k+1)-th neighbor",
        "consumer": "phydrax.discretization.meshfree.refresh_local_stencils",
    }


def _metric_lattice(
    capacity: int, dimension: int, seed: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Full Cartesian lattice; boundary nodes are Dirichlet; tensor control volumes."""
    width = max(3, round(capacity ** (1.0 / dimension)))
    integers = np.stack(
        np.meshgrid(*(np.arange(width) for _ in range(dimension)), indexing="ij"), axis=-1
    ).reshape((-1, dimension))
    spacing = 1.0 / (width - 1)
    edge = (integers == 0) | (integers == width - 1)
    volume = spacing**dimension * np.prod(np.where(edge, 0.5, 1.0), axis=1)
    # A seeded translation changes no moment, rank or metric.
    shift = np.random.default_rng(seed).normal(0.0, 1e-3, dimension)
    return integers * spacing + shift, volume, np.any(edge, axis=1), spacing


def _lattice_edges(points: np.ndarray, boundary: np.ndarray, /) -> int:
    """Axial radius-1.1h edges with at least one free endpoint (host count).

    Edges joining two Dirichlet nodes carry no moment row and are not metric
    unknowns; the count is verified against the prepared metric.
    """
    dimension = points.shape[1]
    width = round(points.shape[0] ** (1.0 / dimension))
    grid = boundary.reshape((width,) * dimension)
    return sum(
        int(
            np.sum(
                ~(
                    np.take(grid, np.arange(width - 1), axis=axis)
                    & np.take(grid, np.arange(1, width), axis=axis)
                )
            )
        )
        for axis in range(dimension)
    )


def _prepare_metric(
    capacity: int, seed: int, config: MeshfreeConfig, recorder: PhaseRecorder, /
) -> PreparedMeshfreeExteriorCalculus:
    from phydrax.discretization.meshfree import (
        MeshfreeExteriorCalculusPlan,
        MeshfreeMetricPolicy,
    )
    from phydrax.linalg import MaterializationPolicy, RankPolicy
    from phydrax.linalg.svd import DenseSVD, SVDSolvePolicy

    points, volume, boundary, spacing = _metric_lattice(capacity, config.dimension, seed)
    if points.shape[0] > config.max_points:
        raise DeclaredCapacityRefusal(
            f"{points.shape[0]} metric nodes exceed the declared max_points={config.max_points}."
        )
    # The strict-active binding certifies KKT nonsingularity by a dense SVD of
    # the (edges + equalities + bounds) <= 3*edges square projection-KKT
    # Jacobian. That certificate budget is declared here, before execution,
    # from the lattice: matrix, factors and copy (3 dense float64 copies) plus
    # the symbolic moment budget. The rank cutoff and QR (gesvd) SVD route are
    # the metric defaults; divide-and-conquer fails on clustered lattice spectra.
    edges = _lattice_edges(points, boundary)
    entries = (3 * edges) ** 2
    policy = MeshfreeMetricPolicy(
        "nonnegative",
        tolerance=1e-9,
        rank=SVDSolvePolicy(
            DenseSVD(algorithm="qr"),
            rank=RankPolicy(relative_cutoff=1e-12),
            materialization=MaterializationPolicy(
                max_entries=entries, max_bytes=8 * entries
            ),
        ),
    )
    declare_reservation(
        3 * 8 * entries + 8 * policy.maximum_symbolic_entries,
        config,
        scope="dense KKT rank certificate and metric symbolic entries",
    )
    prepared = recorder.run(
        "conic",
        lambda: MeshfreeExteriorCalculusPlan(
            points,
            1.1 * spacing,
            2 * config.dimension * points.shape[0],
            node_volumes=volume,
            dirichlet=boundary,
            metric_policy=policy,
        ).prepare(),
        scope="edge-relation-moments-nonnegative-conic-solve",
    )
    recorder.unavailable(
        "search",
        "The radius edge relation is fused into MeshfreeExteriorCalculusPlan.prepare (conic)",
    )
    recorder.unavailable(
        "assembly",
        "Moment assembly is fused into MeshfreeExteriorCalculusPlan.prepare (conic)",
    )
    if prepared.metric_system.prior.shape[0] != edges:
        raise ValueError(
            f"Prepared metric has {prepared.metric_system.prior.shape[0]} edges; the "
            f"declared certificate budget assumed {edges}."
        )
    return prepared


def measure_strict_active_metric(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Strict-active nonnegative metric JVP against central differences of the solve."""
    from phydrax.optim import ConicSensitivityStatus

    recorder = PhaseRecorder()
    prepared = _prepare_metric(capacity, seed, config, recorder)
    system, result = prepared.metric_system, prepared.metric_result
    bound = recorder.run(
        "rank-certificate",
        lambda: system.bind_active_set(result),
        scope="strict-active-kkt-binding",
    )
    generator = np.random.default_rng(seed + 4)
    # Compatible tangents: an RHS change in the range of the moment map plus a
    # prior change; arbitrary RHS directions would leave the exact feasible set.
    rhs = system.constraint.mv(jnp.asarray(generator.normal(size=system.prior.shape)))
    prior = 0.1 * jnp.asarray(generator.normal(size=system.prior.shape)) * system.prior
    tangent, evidence = recorder.run_repeated(
        "jvp",
        lambda: system.strict_active_jvp(bound, rhs=rhs, prior=prior),
        repeats=config.repeats,
        scope="strict-active-jvp",
    )
    recorder.unavailable(
        "vjp", "PreparedMeshfreeMetric publishes the strict-active JVP only"
    )
    # Step 1e-4: the 1e-9 conic tolerance contributes <= 1e-5 FD noise; the map is
    # smooth on the fixed active set, so truncation is O(step^2).
    step = 1e-4
    ahead = recorder.run(
        "numeric-refresh",
        lambda: system.solve(
            rhs=system.rhs + step * rhs, prior=system.prior + step * prior
        ),
        scope="fd-forward-solve",
    )
    behind = recorder.run(
        "numeric-refresh",
        lambda: system.solve(
            rhs=system.rhs - step * rhs, prior=system.prior - step * prior
        ),
        scope="fd-backward-solve",
    )
    central = (ahead.weights - behind.weights) / (2.0 * step)
    active = bound.active_set
    status = -1 if active is None else int(active.status)
    regular = status == int(ConicSensitivityStatus.REGULAR_FIXED_ACTIVE)
    # A withheld (NaN) tangent leaves the FD comparison unexecuted, not finite.
    comparison = (
        {
            "strict_active_fd_relative_error": _relative(
                float(jnp.max(jnp.abs(tangent - central))),
                float(jnp.max(jnp.abs(central))),
            )
        }
        if bool(jnp.all(jnp.isfinite(tangent)))
        else {}
    )
    certificate = bound.kkt_rank_certificate
    bounds = (
        {}
        if certificate is None
        else {
            "kkt_smallest_retained_singular_value": float(
                certificate.smallest_retained_singular_value
            ),
            "kkt_singular_value_error_bound": float(
                certificate.singular_value_error_bound
            ),
        }
    )
    # Unavailable certificates carry non-finite bounds; those stay absent.
    kkt: dict[str, float | int] = {
        key: value for key, value in bounds.items() if np.isfinite(value)
    }
    if certificate is not None:
        kkt["kkt_certificate_rows"] = certificate.rows
        kkt["kkt_certificate_columns"] = certificate.columns
    detail = (
        {}
        if certificate is None
        else {
            "kkt_certificate_route": certificate.route,
            "kkt_certificate_reason": certificate.reason,
        }
    )
    return {
        "workload": "strict-active-metric-sensitivity",
        "capacity": int(prepared.points.shape[0]),
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "metric_accepted": bool(result.accepted),
            "fd_solves_accepted": bool(ahead.accepted) and bool(behind.accepted),
            "active_set_regular": regular,
            "active_set_status": status,
            **(
                {}
                if active is None
                else {
                    "strict_complementarity_margin": float(
                        active.strict_complementarity_margin
                    ),
                    "kkt_primal_residual": float(active.primal_residual_norm),
                    "kkt_dual_residual": float(active.dual_residual_norm),
                    "kkt_complementarity_residual": float(
                        active.complementarity_residual_norm
                    ),
                    "kkt_projection_residual": float(active.projection_residual_norm),
                    "kkt_tolerance": float(active.kkt_tolerance),
                }
            ),
            "derivative_available": bool(bound.derivative_available),
            "tangent_available": bool(evidence.available),
            "derivative_contract_matches": result.derivative_contract
            == STRICT_ACTIVE_CONTRACT,
            **comparison,
            **kkt,
            "edges": int(system.prior.shape[0]),
            "moments": int(system.rhs.shape[0]),
            "minimum_weight": float(jnp.min(result.weights)),
        },
        "detail": detail,
        "retained_bytes": logical_array_bytes(prepared),
        "oracle_provenance": "central differences of PreparedMeshfreeMetric.solve on the "
        "same nonnegative program",
        "consumer": "phydrax.discretization.meshfree.PreparedMeshfreeMetric."
        "bind_active_set/strict_active_jvp",
    }


def measure_strict_active_change_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """An edge driven to its bound under frozen roles: the derivative is withheld."""
    from phydrax.optim import ConicSensitivityStatus

    recorder = PhaseRecorder()
    prepared = _prepare_metric(capacity, seed, config, recorder)
    system, result = prepared.metric_system, prepared.metric_result
    bound = recorder.run(
        "rank-certificate",
        lambda: system.bind_active_set(result),
        scope="strict-active-kkt-binding",
    )
    # Compatible data whose solution puts one edge exactly on its bound.
    zeroed = result.weights.at[result.weights.shape[0] // 2].set(0.0)
    rhs = system.constraint.mv(zeroed)
    changed = recorder.run(
        "conic", lambda: system.solve(rhs=rhs), scope="zero-edge-solve"
    )
    rebound = recorder.run(
        "rank-certificate",
        lambda: system.bind_active_set(
            changed, rhs=rhs, fixed_active_set=bound.active_set
        ),
        scope="frozen-role-rebinding",
    )
    direction = system.constraint.mv(jnp.ones_like(result.weights))
    tangent, evidence = recorder.run(
        "jvp",
        lambda: system.strict_active_jvp(rebound, rhs=direction),
        scope="refused-jvp",
    )
    status = -1 if rebound.active_set is None else int(rebound.active_set.status)
    return {
        "workload": "strict-active-change-refusal",
        "capacity": int(prepared.points.shape[0]),
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "baseline_regular": bound.active_set is not None
            and int(bound.active_set.status)
            == int(ConicSensitivityStatus.REGULAR_FIXED_ACTIVE),
            "active_set_status": status,
            "active_set_change_named": status
            in (
                int(ConicSensitivityStatus.ACTIVE_SET_CHANGED),
                int(ConicSensitivityStatus.AMBIGUOUS_ACTIVE_SET),
            ),
            "derivative_withheld": not bool(rebound.derivative_available),
            "tangent_unavailable": not bool(evidence.available),
            "tangent_nan": bool(jnp.all(jnp.isnan(tangent))),
        },
        "oracle_provenance": "RHS = A w' with one edge weight of w' exactly zero",
        "consumer": "phydrax.discretization.meshfree.PreparedMeshfreeMetric."
        "bind_active_set(fixed_active_set=...)",
    }


_REMAP_HISTORIES = ("field/current", "history/0", "history/1", "predictor/rate")


def _remap_routes(
    capacity: int, dimension: int, seed: int, recorder: PhaseRecorder, /
) -> tuple[list[Any], list[tuple[np.ndarray, np.ndarray]]]:
    """Per-history conservative-positive frozen routes between two seeded supports."""
    from phydrax.discretization import TopologyEpoch
    from phydrax.discretization.meshfree import PointTransferPlan, PointTransferRequest
    from phydrax.sparse import EdgeRelation

    source_count, target_count = capacity, max(4, (3 * capacity) // 4)
    old_points = cloud_points(source_count, dimension, seed)
    new_points = cloud_points(target_count, dimension, seed + 1)
    bandwidth = 1.5 * source_count ** (-1.0 / dimension)
    pairs = cKDTree(new_points).query_ball_point(old_points, 3.0 * bandwidth)
    if any(len(item) == 0 for item in pairs):
        raise ValueError("Every source point needs a target within the transfer window.")
    columns = np.concatenate(
        [np.full(len(item), index) for index, item in enumerate(pairs)]
    )
    rows = np.concatenate([np.asarray(item, dtype=np.int64) for item in pairs])
    distance = np.linalg.norm(new_points[rows] - old_points[columns], axis=1)
    affinity = np.exp(-((distance / bandwidth) ** 2))
    source = TopologyEpoch(0, f"q14-cloud-{source_count}-{seed}", "bulk", "serial")
    target = TopologyEpoch(1, f"q14-cloud-{target_count}-{seed + 1}", "bulk", "serial")
    generator = np.random.default_rng(seed + 5)
    routes: list[Any] = []
    measures: list[tuple[np.ndarray, np.ndarray]] = []
    for name in _REMAP_HISTORIES:
        old = generator.uniform(0.5, 1.5, source_count) / source_count
        new = generator.uniform(0.5, 1.5, target_count) / target_count
        # Exactly normalized nonnegative allocation: sum_i new_i c_ij = old_j.
        normalizer = np.zeros(source_count)
        np.add.at(normalizer, columns, new[rows] * affinity)
        coefficients = affinity * old[columns] / normalizer[columns]
        prepared = recorder.run(
            "transfer",
            lambda: PointTransferPlan(
                EdgeRelation(
                    columns.astype(np.int32),
                    rows.astype(np.int32),
                    source_size=source_count,
                    target_size=target_count,
                ),
                coefficients,
                old,
                new,
                source_id=f"q14-remap/{name}/source/{source.epoch_id}",
                target_id=f"q14-remap/{name}/target/{target.epoch_id}",
                request=PointTransferRequest("conservative-positive"),
            ).prepare(),
            scope=f"route-preparation:{name}",
        )
        if not prepared.admitted:
            raise ValueError(f"The {name} conservative-positive route was not admitted.")
        routes.append(prepared.epoch_transition(source, target))
        measures.append((old, new))
    return routes, measures


def measure_frozen_remap_sensitivity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Frozen multi-history remap: JVP vs FD, published pullback, duality, adjoint."""
    from phydrax.discretization.meshfree import remap_live_histories

    recorder = PhaseRecorder()
    routes, measures = _remap_routes(capacity, config.dimension, seed, recorder)
    generator = np.random.default_rng(seed + 6)
    fields = tuple(jnp.asarray(generator.uniform(0.5, 2.0, capacity)) for _ in routes)
    tangents = tuple(jnp.asarray(generator.normal(size=capacity)) for _ in routes)
    target_count = measures[0][1].size
    cotangents = tuple(jnp.asarray(generator.normal(size=target_count)) for _ in routes)

    def remap(*histories: Array) -> tuple[Array, ...]:
        return remap_live_histories(routes, histories).values

    remapped = recorder.run(
        "transfer",
        lambda: remap_live_histories(routes, fields),
        scope="live-history-remap",
    )
    _, pushed = recorder.run_repeated(
        "jvp",
        lambda: jax.jvp(remap, fields, tangents),
        repeats=config.repeats,
        scope="remap-jvp",
    )
    pulled = recorder.run_repeated(
        "vjp",
        lambda: jax.vjp(remap, *fields)[1](cotangents),
        repeats=config.repeats,
        scope="remap-ad-vjp",
    )
    published = recorder.run(
        "vjp", lambda: remapped.pullback(cotangents), scope="published-pullback"
    )
    adjoint = recorder.run(
        "vjp", lambda: remapped.adjoint(cotangents), scope="hilbert-adjoint"
    )
    step = 1e-3
    ahead = remap(*(f + step * t for f, t in zip(fields, tangents, strict=True)))
    behind = remap(*(f - step * t for f, t in zip(fields, tangents, strict=True)))
    fd, identity, pullback_error, hilbert, conservation = 0.0, 0.0, 0.0, 0.0, 0.0
    for index, route in enumerate(routes):
        scale = float(jnp.max(jnp.abs(pushed[index])))
        central = (ahead[index] - behind[index]) / (2.0 * step)
        fd = max(fd, _relative(float(jnp.max(jnp.abs(pushed[index] - central))), scale))
        identity = max(
            identity,
            _relative(
                float(
                    jnp.max(jnp.abs(pushed[index] - route.apply(tangents[index]).values))
                ),
                scale,
            ),
        )
        pullback_error = max(
            pullback_error,
            _relative(
                float(jnp.max(jnp.abs(pulled[index] - published[index]))),
                float(jnp.max(jnp.abs(pulled[index]))),
            ),
        )
        old, new = measures[index]
        left = float(jnp.vdot(new * remapped.values[index], cotangents[index]))
        right = float(jnp.vdot(old * fields[index], adjoint[index]))
        hilbert = max(hilbert, _relative(abs(left - right), max(abs(left), abs(right))))
        content = float(jnp.vdot(old, fields[index]))
        conservation = max(
            conservation,
            _relative(
                abs(float(jnp.vdot(new, remapped.values[index])) - content), content
            ),
        )
    forward = sum(float(jnp.vdot(c, p)) for c, p in zip(cotangents, pushed, strict=True))
    reverse = sum(float(jnp.vdot(q, t)) for q, t in zip(pulled, tangents, strict=True))
    return {
        "workload": "frozen-remap-sensitivity",
        "capacity": capacity,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "histories": len(routes),
            "target_points": target_count,
            "remap_successful": bool(remapped.successful),
            "value_derivative_available": bool(remapped.value_derivative_available),
            "jvp_fd_relative_error": fd,
            "jvp_route_action_identity_error": identity,
            "published_pullback_relative_error": pullback_error,
            "vjp_duality_relative_error": _relative(
                abs(forward - reverse), max(abs(forward), abs(reverse))
            ),
            "hilbert_adjoint_relative_error": hilbert,
            "content_conservation_relative_error": conservation,
        },
        "retained_bytes": logical_array_bytes(routes),
        "oracle_provenance": "central differences of remap_live_histories, each route's own "
        "primal action, AD reverse mode, and measure-weighted inner products",
        "consumer": "phydrax.discretization.meshfree.remap_live_histories / "
        "MeshfreeHistoryRemap.pullback/adjoint",
    }


def measure_frozen_remap_failure_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """One failed history refuses the epoch: NaN values/tangents, no reverse maps."""
    from phydrax.discretization.meshfree import remap_live_histories

    recorder = PhaseRecorder()
    routes, _ = _remap_routes(capacity, config.dimension, seed, recorder)
    fields = [jnp.ones(capacity) for _ in routes]
    fields[1] = fields[1].at[capacity // 2].set(jnp.inf)
    remapped = recorder.run(
        "transfer", lambda: remap_live_histories(routes, fields), scope="refused-remap"
    )
    _, pushed = recorder.run(
        "jvp",
        lambda: jax.jvp(
            lambda first: remap_live_histories(routes, [first, *fields[1:]]).values[0],
            (fields[0],),
            (jnp.ones(capacity),),
        ),
        scope="refused-jvp",
    )
    target_count = remapped.values[0].shape[0]
    refused = recorder.run(
        "vjp",
        lambda: _refused(
            lambda: remapped.pullback([jnp.ones(target_count)] * len(routes)),
            ValueError,
            "no value derivative crosses a refused epoch",
        ),
        scope="refused-pullback",
    )
    adjoint_refused = _refused(
        lambda: remapped.adjoint([jnp.ones(target_count)] * len(routes)),
        ValueError,
        "no value derivative crosses a refused epoch",
    )
    return {
        "workload": "frozen-remap-failure-refusal",
        "capacity": capacity,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {
            "remap_refused": not bool(remapped.successful),
            "failed_histories_named": list(remapped.failed) == [1],
            "value_derivative_unavailable": not bool(remapped.value_derivative_available),
            "values_nan": all(bool(jnp.all(jnp.isnan(item))) for item in remapped.values),
            "jvp_nan": bool(jnp.all(jnp.isnan(pushed))),
            "pullback_refused": refused,
            "adjoint_refused": adjoint_refused,
        },
        "oracle_provenance": "one nonfinite live-history value fails its route",
        "consumer": "phydrax.discretization.meshfree.remap_live_histories",
    }


type _Workload = Callable[[int, int, MeshfreeConfig], dict[str, Any]]


def _float64_only(workload: _Workload, /) -> _Workload:
    """Every Q12/Q14 support tuple declares float64; other requests are refused."""

    def measured(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
        if config.precision != "float64":
            raise ValueError(
                f"Q12/Q14 workloads declare float64 only; requested {config.precision}."
            )
        return workload(capacity, seed, config)

    return measured


FORMS_SENSITIVITY_WORKLOADS: dict[str, _Workload] = {
    name: _float64_only(workload)
    for name, workload in (
        ("higher-forms-holed-square", measure_higher_forms_holed_square),
        ("higher-forms-tunneled-slab", measure_higher_forms_tunneled_slab),
        ("higher-forms-icosphere", measure_higher_forms_icosphere),
        ("abstract-clique-forms", measure_abstract_clique_forms),
        ("higher-forms-refusals", measure_higher_forms_refusals),
        ("abstract-clique-refusals", measure_abstract_clique_refusals),
        ("fixed-support-sensitivity", measure_fixed_support_sensitivity),
        ("fixed-support-envelope-exit", measure_fixed_support_envelope_exit),
        ("smooth-support-crossing", measure_smooth_support_crossing),
        ("smooth-support-envelope-exit", measure_smooth_support_envelope_exit),
        ("knn-tie-refusal", measure_knn_tie_refusal),
        ("strict-active-metric-sensitivity", measure_strict_active_metric),
        ("strict-active-change-refusal", measure_strict_active_change_refusal),
        ("frozen-remap-sensitivity", measure_frozen_remap_sensitivity),
        ("frozen-remap-failure-refusal", measure_frozen_remap_failure_refusal),
    )
}


__all__ = [
    "FORMS_SENSITIVITY_WORKLOADS",
    "HigherFormGeometry",
    "STRICT_ACTIVE_CONTRACT",
    "higher_form_dimension",
    "higher_form_level",
    "higher_form_top_cells",
]
