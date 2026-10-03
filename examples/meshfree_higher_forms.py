"""Geometry-authorized meshfree k-forms on 2-D, 3-D, holed and curved complexes.

Authoritative oriented simplicial meshes (a square with a square hole, a cube,
a solid torus, and a closed icosphere sheet) are realized by local GMLS k-form
reconstruction with natively admitted sparse Hodges. The workflow measures
de Rham commutation, harmonic spaces against exact Betti numbers, Stokes
identities, and drives a compatible Maxwell leapfrog that evolves degree-2
fluxes under the degree-3 constraint.
"""

from __future__ import annotations

from itertools import permutations
from math import comb
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import CellMesh
from phydrax.discretization.meshfree import (
    ComplexDomainIdentity,
    ComplexGeometryAuthority,
    MeshfreeCellComplexPlan,
    MeshfreeComplexAdmissionError,
    MeshfreeComplexPolicy,
    PreparedMeshfreeCellComplex,
    radius_clique_complex,
    RadiusCliquePolicy,
)
from phydrax.exterior import (
    integrate_form,
    trace_map,
    validate_de_rham_commutation,
    validate_harmonic_cohomology,
)
from phydrax.geometry.multiregion_surface import seed_sphere
from phydrax.geometry.simplicial import TriangleMesh
from phydrax.metrix import CoordinateChart, DifferentialForm
from phydrax.solver.maxwell import (
    CompatibleMaxwellState,
    DiagonalMaxwellConstitutivePlan,
    UnstructuredMaxwellPlan,
)


def planar_grid_mesh(cells: int, *, hole: bool) -> CellMesh:
    """Triangulate [0, 3]^2 on a 3n grid, optionally without [1, 2]^2."""
    count = 3 * cells
    points = np.asarray(
        [
            (3.0 * i / count, 3.0 * j / count)
            for j in range(count + 1)
            for i in range(count + 1)
        ],
        dtype=np.float64,
    )
    triangles: list[tuple[int, int, int]] = []
    for j in range(count):
        for i in range(count):
            if hole and cells <= i < 2 * cells and cells <= j < 2 * cells:
                continue
            a = j * (count + 1) + i
            b, c, d = a + 1, a + count + 1, a + count + 2
            triangles.extend(((a, b, d), (a, d, c)))
    used = np.unique(np.asarray(triangles, dtype=np.int32))
    remap = np.full((points.shape[0],), -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return CellMesh.from_triangles(
        points[used], remap[np.asarray(triangles, dtype=np.int32)]
    )


def box_tetrahedral_mesh(cells: int, *, tunnel: bool) -> CellMesh:
    """Kuhn-split [0, 3]^2 x [0, 1] on a 3n x 3n x 2n grid, optionally tunneled.

    Two vertex layers alone would not determine linear 1-forms: d(z(z - 1))
    has zero integral on every edge. The local GMLS admission refuses that
    mesh, so the slab carries at least three vertex layers.
    """
    shape = (3 * cells, 3 * cells, 2 * cells)
    spacing = np.asarray((3.0 / shape[0], 3.0 / shape[1], 1.0 / shape[2]))

    def index(i: int, j: int, k: int) -> int:
        return (k * (shape[1] + 1) + j) * (shape[0] + 1) + i

    points = np.asarray(
        [
            (i * spacing[0], j * spacing[1], k * spacing[2])
            for k in range(shape[2] + 1)
            for j in range(shape[1] + 1)
            for i in range(shape[0] + 1)
        ],
        dtype=np.float64,
    )
    tetrahedra: list[tuple[int, ...]] = []
    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                if tunnel and cells <= i < 2 * cells and cells <= j < 2 * cells:
                    continue
                for order in permutations(range(3)):
                    corner = np.asarray((i, j, k))
                    path = [index(*corner)]
                    for axis in order:
                        corner = corner + np.eye(3, dtype=np.int64)[axis]
                        path.append(index(*corner))
                    edges = points[path[1:]] - points[path[0]]
                    if np.linalg.det(edges) < 0.0:
                        path[2], path[3] = path[3], path[2]
                    tetrahedra.append(tuple(path))
    used = np.unique(np.asarray(tetrahedra, dtype=np.int32))
    remap = np.full((points.shape[0],), -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return CellMesh.from_tetrahedra(
        points[used], remap[np.asarray(tetrahedra, dtype=np.int32)]
    )


def sphere_mesh(subdivisions: int) -> CellMesh:
    """Closed unit icosphere sheet with every vertex on the sphere."""
    seed = seed_sphere(1.0, subdivisions=subdivisions)
    return TriangleMesh(
        np.asarray(seed.positions), np.asarray(seed.faces), source_id=seed.seed_id
    ).as_cell_mesh()


def chart_for(dimension: int) -> CoordinateChart:
    return CoordinateChart(
        f"cartesian-{dimension}d", tuple(f"x{axis}" for axis in range(dimension))
    )


def realize(
    mesh: CellMesh,
    identity: ComplexDomainIdentity,
    *,
    policy: MeshfreeComplexPolicy | None = None,
) -> PreparedMeshfreeCellComplex:
    authority = ComplexGeometryAuthority(mesh, identity)
    return MeshfreeCellComplexPlan(
        authority, chart=chart_for(mesh.ambient_dimension), policy=policy
    ).prepare()


def holed_square(cells: int = 1) -> PreparedMeshfreeCellComplex:
    return realize(
        planar_grid_mesh(cells, hole=True),
        ComplexDomainIdentity("square-with-hole", "domain", measure=8.0, betti=(1, 1, 0)),
    )


def solid_torus(cells: int = 1) -> PreparedMeshfreeCellComplex:
    return realize(
        box_tetrahedral_mesh(cells, tunnel=True),
        ComplexDomainIdentity("tunneled-slab", "domain", measure=8.0, betti=(1, 1, 0, 0)),
    )


def unit_sphere(subdivisions: int = 1) -> PreparedMeshfreeCellComplex:
    # Flat facets inscribed in the sphere lose 7.2% (one subdivision) or 1.9%
    # (two) of the area; the declared tolerance states that chordal deficit.
    return realize(
        sphere_mesh(subdivisions),
        ComplexDomainIdentity(
            "unit-sphere",
            "closed_surface",
            measure=4.0 * np.pi,
            betti=(1, 0, 1),
            measure_tolerance=0.08,
        ),
    )


def _cubic_form(complex_: PreparedMeshfreeCellComplex, degree: int) -> DifferentialForm:
    """Cubic coefficients: the declared order-6 cell quadrature is exact."""
    chart = complex_.bridge.chart
    components = comb(chart.dimension, degree)

    def coefficients(point: Array) -> Array:
        base = point[0] ** 3 - point[1] * point[-1] ** 2 + 0.5 * point[1]
        return base * jnp.arange(1, components + 1, dtype=jnp.float64)

    return DifferentialForm(coefficients, chart=chart, degree=degree)


class HigherFormReport(NamedTuple):
    commutation: tuple[float, ...]
    harmonic_ranks: tuple[int, ...]
    betti: tuple[int, ...]
    stokes_defect: float
    admitted: bool


def audit_complex(complex_: PreparedMeshfreeCellComplex) -> HigherFormReport:
    """Commutation, harmonic dimension and Stokes evidence for one realization."""
    dimension = complex_.cochain.dimension
    commutation = tuple(
        float(
            validate_de_rham_commutation(
                _cubic_form(complex_, degree), complex_.bridge, tolerance=1e-11
            ).relative_residual
        )
        for degree in range(dimension)
    )
    ranks = []
    for degree in range(dimension + 1):
        _, report = validate_harmonic_cohomology(complex_.cochain, degree, tolerance=1e-7)
        ranks.append(report.harmonic_rank if bool(report.complete) else -1)
    form = integrate_form(_cubic_form(complex_, dimension - 1), complex_.bridge)
    volume = jnp.sum(complex_.cochain.exterior_derivative(dimension - 1, form.values))
    if bool(np.any(np.asarray(complex_.cochain.boundary_masks[dimension - 1]))):
        trace = trace_map(complex_.cochain)
        boundary = jnp.sum(trace.maps[dimension - 1].mv(form.values))
    else:
        boundary = jnp.asarray(0.0)
    return HigherFormReport(
        commutation,
        tuple(ranks),
        complex_.evidence.betti,
        float(jnp.abs(volume - boundary)),
        bool(complex_.evidence.admitted),
    )


class MaxwellReport(NamedTuple):
    steps: int
    stable_dt: float
    energy_drift: float
    magnetic_divergence: float
    electric_activity: float


def maxwell_flux_evolution(
    complex_: PreparedMeshfreeCellComplex, *, steps: int = 20
) -> MaxwellReport:
    """Leapfrog D on edges and B on faces; dB lives on degree-3 cells."""
    runtime = UnstructuredMaxwellPlan(
        complex_.cochain, DiagonalMaxwellConstitutivePlan(), courant_factor=0.9
    ).prepare()
    chart = complex_.bridge.chart
    potential = DifferentialForm(
        lambda point: jnp.asarray(
            [jnp.sin(point[1]) * point[2], point[0] * point[2] ** 2, jnp.cos(point[0])]
        ),
        chart=chart,
        degree=1,
    )
    flux = complex_.cochain.exterior_derivative(
        1, integrate_form(potential, complex_.bridge).values
    )
    state = runtime.initialize()
    state = type(state)(
        type(state.primary)(
            state.primary.electric_displacement, flux, state.primary.charge
        ),
        state.auxiliary,
        state.observations,
    )
    dt = 0.5 * runtime.stable_dt

    def energy(value: CompatibleMaxwellState) -> Array:
        electric = runtime.electric_field(value)
        magnetic = runtime.magnetic_field(value)
        return 0.5 * (
            jnp.vdot(electric, complex_.cochain.hodge_star(1, electric))
            + jnp.vdot(magnetic, complex_.cochain.hodge_star(2, magnetic))
        )

    initial = energy(state)
    time = 0.0
    for _ in range(steps):
        state = runtime.step(time, state, dt)
        time += float(dt)
    divergence = complex_.cochain.exterior_derivative(2, state.primary.magnetic_flux)
    return MaxwellReport(
        steps,
        float(runtime.stable_dt),
        float(jnp.abs(energy(state) - initial) / initial),
        float(jnp.max(jnp.abs(divergence))),
        float(jnp.max(jnp.abs(state.primary.electric_displacement))),
    )


def research_clique_circle(points: int = 24) -> tuple[tuple[int, ...], int]:
    angles = 2.0 * np.pi * np.arange(points) / points
    cloud = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    complex_ = radius_clique_complex(
        cloud, 0.3, policy=RadiusCliquePolicy(maximum_pairs=4 * points)
    )
    betti = tuple(
        complex_.betti.dimension(degree)
        for degree in range(complex_.topology.dimension + 1)
    )
    return betti, complex_.work


def sampled_commutation_errors(levels: tuple[int, ...]) -> tuple[float, ...]:
    """Max |d S₀(f) − S₁(df)| of GMLS node-sample moments on refined squares."""
    errors = []
    for cells in levels:
        complex_ = realize(
            planar_grid_mesh(cells, hole=False),
            ComplexDomainIdentity("square", "domain", measure=9.0, betti=(1, 0, 0)),
        )
        x, y = complex_.nodes[:, 0], complex_.nodes[:, 1]
        potential = (jnp.sin(x) * jnp.cos(y))[:, None]
        gradient = jnp.stack((jnp.cos(x) * jnp.cos(y), -jnp.sin(x) * jnp.sin(y)), axis=1)
        derived = complex_.cochain.exterior_derivative(
            0, complex_.sample(0, potential).values
        )
        errors.append(
            float(jnp.max(jnp.abs(derived - complex_.sample(1, gradient).values)))
        )
    return tuple(errors)


def refused_hole_filling() -> str:
    """A triangulation that fills the hole cannot claim the holed domain."""
    try:
        ComplexGeometryAuthority(
            planar_grid_mesh(1, hole=False),
            ComplexDomainIdentity(
                "square-with-hole", "domain", measure=8.0, betti=(1, 1, 0)
            ),
        )
    except MeshfreeComplexAdmissionError as error:
        return str(error)
    raise RuntimeError("A hole-filling triangulation was admitted.")


def main() -> None:
    for name, build in (
        ("square with hole", holed_square),
        ("solid torus", solid_torus),
        ("unit sphere", unit_sphere),
    ):
        complex_ = build()
        report = audit_complex(complex_)
        print(
            f"{name}: admitted={report.admitted} betti={report.betti} "
            f"harmonic={report.harmonic_ranks} commutation={max(report.commutation):.2e} "
            f"stokes={report.stokes_defect:.2e}"
        )
    maxwell = maxwell_flux_evolution(solid_torus())
    print(
        f"maxwell: steps={maxwell.steps} dt={maxwell.stable_dt:.3e} "
        f"energy_drift={maxwell.energy_drift:.2e} max|dB|={maxwell.magnetic_divergence:.2e}"
    )
    errors = sampled_commutation_errors((1, 2))
    print(f"sampled GMLS commutation: {errors[0]:.2e} -> {errors[1]:.2e}")
    print(f"refused: {refused_hole_filling()}")
    betti, work = research_clique_circle()
    print(f"abstract clique (research only): betti={betti} work={work}")


if __name__ == "__main__":
    main()
