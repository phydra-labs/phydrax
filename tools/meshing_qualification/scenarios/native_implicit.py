from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature, RegionSeed
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing._tetra_metric import TetraMetricEvidence
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativeImplicitSource

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


_CONTRACT = SpatialCoordinateContract.si()


def _exact_value(points: Any) -> Any:
    return jnp.sum(points * points, axis=-1)


def _exact_source(points: Any, context: Any) -> Any:
    del context
    return jnp.full(points.shape[:-1], -6.0, dtype=points.dtype)


def _solve_physical_epoch(
    result: phx.meshing.CellMeshingResult,
    carried: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Solve the donor P1 diffusion problem and independently integrate its L2 error."""
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("tetrahedron", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    if constraint.free_dofs.size == 0:
        raise RuntimeError("Implicit P1 diffusion requires interior degrees of freedom.")
    form = phx.equations.FiniteElementForm(
        "native-implicit-manufactured-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _exact_source, coefficient_id="quadratic-exact-laplacian"
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=_exact_value
    )
    operator, rhs = problem.linear_system()
    initial_guess = None
    if carried is not None:
        coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
        if carried.shape != (coordinates.shape[0],):
            raise ValueError("Transferred P1 values must bind every target vertex.")
        by_coordinate = {
            tuple(
                float(coordinates[index, axis]) for axis in range(coordinates.shape[1])
            ): value
            for index, value in enumerate(carried)
        }
        dof_coordinates = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
        try:
            dof_values = np.asarray(
                [
                    by_coordinate[
                        tuple(
                            float(dof_coordinates[index, axis])
                            for axis in range(dof_coordinates.shape[1])
                        )
                    ]
                    for index in range(dof_coordinates.shape[0])
                ]
            )
        except KeyError as error:
            raise ValueError(
                "Target P1 degrees of freedom do not coincide with mesh vertices."
            ) from error
        initial_guess = jnp.asarray(
            dof_values[np.asarray(constraint.free_dofs, dtype=np.int64)]
        )
    solved = phx.linalg.solve(operator, rhs, initial_guess=initial_guess)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    residual = float(jnp.linalg.norm(problem.residual(solved.value))) / max(
        1.0, float(jnp.linalg.norm(rhs))
    )
    if not bool(jnp.all(solved.successful)) or residual > 1.0e-8:
        raise RuntimeError(
            f"Implicit P1 diffusion did not converge: residual={residual:g}."
        )
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    vertex_values = np.full(points.shape[0], np.nan)
    error_squared = 0.0
    denominator = math.factorial(7) / math.factorial(3)
    for block, element, route, transform in zip(
        result.mesh.blocks,
        space.elements[0],
        space.dof_maps[0].cell_dofs,
        space.dof_maps[0].cell_transforms,
        strict=True,
    ):
        cells = np.asarray(block.vertices, dtype=np.int64)
        corners = points[cells]
        local = np.einsum("cij,cj->ci", np.asarray(transform), values[route])
        nodal = local[:, [dofs[0] for dofs in element.entity_dofs[0]]]
        vertex_values[cells] = nodal
        coefficient = -corners @ np.swapaxes(corners, -1, -2)
        coefficient += 0.5 * (nodal[:, :, None] + nodal[:, None, :])
        integral = np.zeros(cells.shape[0], dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for ell in range(4):
                        exponents = np.bincount((i, j, k, ell), minlength=4)
                        moment = (
                            math.prod(math.factorial(int(e)) for e in exponents)
                            / denominator
                        )
                        integral += coefficient[:, i, j] * coefficient[:, k, ell] * moment
        edges = corners[:, 1:] - corners[:, :1]
        measures = np.abs(np.linalg.det(edges)) / 6.0
        error_squared += float(np.sum(measures * integral))
    if not np.all(np.isfinite(vertex_values)):
        raise RuntimeError("Implicit P1 solution did not cover all carrier vertices.")
    error = math.sqrt(max(0.0, error_squared))
    return (
        error,
        vertex_values,
        {
            "form_id": form.form_id,
            "dofs": int(values.size),
            "normalized_residual": residual,
            "l2_error": error,
        },
    )


def _ball_enclosure(radius: float, /) -> Callable[[float], tuple[float, float]]:
    """Exact volumes of a ball's inner and outer parallel bodies at offset epsilon.

    A closed carrier within epsilon of the sphere encloses a volume between
    them while epsilon stays below the reach, the radius (donor
    tools/meshing_qualification/scenarios/implicit.py lines 522-553).
    """

    def enclosure(epsilon: float) -> tuple[float, float]:
        if not 0.0 <= epsilon < radius:
            raise ValueError(
                f"The ball enclosure requires 0 <= epsilon < {radius}; got {epsilon}."
            )
        return (
            (4.0 / 3.0) * math.pi * (radius - epsilon) ** 3,
            (4.0 / 3.0) * math.pi * (radius + epsilon) ** 3,
        )

    return enclosure


def _ring_torus_enclosure(
    minor: float, major: float, /
) -> Callable[[float], tuple[float, float]]:
    """Exact volumes of a ring torus's inner and outer parallel bodies.

    The tube radius moves by epsilon below the reach min(minor, major - minor)
    (Pappus: 2 pi^2 major (minor -/+ epsilon)^2).
    """
    reach = min(minor, major - minor)

    def enclosure(epsilon: float) -> tuple[float, float]:
        if not 0.0 <= epsilon < reach:
            raise ValueError(
                f"The torus enclosure requires 0 <= epsilon < {reach}; got {epsilon}."
            )
        return (
            2.0 * math.pi**2 * major * (minor - epsilon) ** 2,
            2.0 * math.pi**2 * major * (minor + epsilon) ** 2,
        )

    return enclosure


def _volume_spec(
    source: AnalyticImplicitProfile | NativeImplicitSource,
    /,
    size: float,
    tolerance: float,
    witness: np.ndarray | None = None,
) -> VolumeMeshingSpec:
    scope = MeshingScope.whole(source, 2)
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(
                scope, size, maximum_size=1.5 * size, strength=SizeControlStrength.HARD
            ),
        ),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
        protected_features=(
            ProtectedFeature(
                scope, FeatureKind.SURFACE, maximum_deviation=tolerance, hard=True
            ),
        ),
        region_seeds=()
        if witness is None
        else (RegionSeed(witness, "interior", "uniform-material", RegionRole.USER),),
        maximum_deviation=tolerance,
    )


def _volume_case(
    name: str,
    profile: AnalyticImplicitProfile | NativeImplicitSource,
    exact_volume: float,
    /,
    *,
    size: float,
    tolerance: float,
    required_components: int,
    discovery: AdaptiveImplicitSurface | None = None,
    adaptation_rounds: int = 0,
    anisotropy: float = 1.0,
    fidelity_levels: int = 1,
    required_genus: int = 0,
    witness: np.ndarray | None = None,
    enclosure: Callable[[float], tuple[float, float]] | None = None,
    policy: AdaptiveImplicitSurfacePolicy | None = None,
) -> dict[str, object]:
    """Generate one implicit profile, its fidelity ladder, and metric epochs.

    ``enclosure`` maps a certified mesh-to-source bound epsilon to the exact
    volumes of the inner and outer parallel bodies at that offset; it is
    required for a ladder and checked at every level. ``policy`` is the
    route's discovery policy (root tolerance 1e-10 when omitted).
    """
    if fidelity_levels < 1:
        raise ValueError("The fidelity ladder must contain at least one level.")
    if fidelity_levels > 1 and enclosure is None:
        raise ValueError("A fidelity ladder requires the certified volume enclosure.")
    ladder: list[dict[str, object]] = []
    ladder_errors: list[float] = []
    fidelity_uppers: list[float] = []
    enclosure_widths: list[float] = []
    for level in range(fidelity_levels):
        level_tolerance = tolerance * 2.0**-level
        plan = NativeMeshingProvider(
            NativeMeshingOptions(
                "implicit_adaptive_tetrahedral",
                implicit_policy=AdaptiveImplicitSurfacePolicy(root_tolerance=1.0e-10)
                if policy is None
                else policy,
            )
        ).plan(
            profile,
            _volume_spec(profile, size, level_tolerance, witness),
            coordinate_contract=_CONTRACT,
        )
        result = plan.execute()
        volume = float(
            np.sum(np.asarray(result.quality.evaluation.measures, dtype=np.float64))
        )
        level_error = abs(volume - exact_volume) / exact_volume
        ladder_errors.append(level_error)
        if level_error > 0.15:
            raise RuntimeError(
                f"{name} fidelity level {level} exceeded the volume tolerance: {level_error}."
            )
        level_record: dict[str, object] = {
            "fidelity_m": level_tolerance,
            "measured_volume_m3": volume,
            "relative_volume_error": level_error,
        }
        if enclosure is not None:
            certificate = (
                None if result.certification is None else result.certification.fidelity
            )
            if certificate is None:
                raise RuntimeError(
                    f"{name} fidelity level {level} lacks source-fidelity certification."
                )
            # The certified mesh-to-source bound plays the donor's projection
            # bound: the exact volume must lie in V(r - epsilon) .. V(r + epsilon).
            epsilon = float(certificate.mesh_to_source_upper)
            lower, upper = enclosure(epsilon)
            if not lower <= volume <= upper:
                raise RuntimeError(
                    f"{name} fidelity level {level} volume {volume} lies outside its "
                    f"certified enclosure [{lower}, {upper}] at epsilon {epsilon}."
                )
            fidelity_uppers.append(epsilon)
            enclosure_widths.append(upper - lower)
            level_record.update(
                certified_fidelity_upper_m=epsilon,
                certified_volume_enclosure_m3=[lower, upper],
                certified_enclosure_width_m3=upper - lower,
                volume_inside_certified_enclosure=True,
            )
        level_record["accepted"] = accepted_result_record(result)
        ladder.append(level_record)
        if level + 1 < fidelity_levels:
            # The accepted record retains the completed level's evidence; its
            # mesh and discovery are not inputs to the next cold fidelity run.
            del result, plan
    # The sagitta-centered carrier's volume error is a near cancellation of
    # outward and inward caps and is not monotone in the tolerance; the ladder
    # instead requires the certified fidelity bound and the certified volume
    # enclosure it implies to shrink strictly (docs/plans/native-cad-meshing.md
    # section 10, "Approved certified implicit volume ladder").
    fidelity_decreasing = all(b < a for a, b in zip(fidelity_uppers, fidelity_uppers[1:]))
    enclosure_shrinking = all(
        b < a for a, b in zip(enclosure_widths, enclosure_widths[1:])
    )
    if not (fidelity_decreasing and enclosure_shrinking):
        raise RuntimeError(
            f"{name} fidelity ladder did not strictly shrink its certified fidelity "
            f"bounds {fidelity_uppers} and volume enclosures {enclosure_widths}."
        )
    accepted = accepted_result_record(result)
    measured_volume = float(
        np.sum(np.asarray(result.quality.evaluation.measures, dtype=np.float64))
    )
    relative_volume_error = abs(measured_volume - exact_volume) / exact_volume
    if relative_volume_error > 0.15:
        raise RuntimeError(
            f"{name} native implicit volume has {relative_volume_error:.6g} relative error; tolerance is 0.15."
        )
    blocks = tuple(
        np.asarray(block.vertices, dtype=np.int64) for block in result.mesh.blocks
    )
    cells = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
    facets = np.sort(
        cells[:, ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))].reshape(-1, 3),
        axis=1,
    )
    unique_facets, counts = np.unique(facets, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError(f"{name} tetrahedral carrier has nonmanifold faces.")
    faces = unique_facets[counts == 1]
    if not faces.size:
        raise RuntimeError(f"{name} tetrahedral carrier has no boundary.")
    edges = np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(edge_counts != 2):
        raise RuntimeError(f"{name} boundary is not a closed two-manifold.")
    parents = np.arange(faces.shape[0], dtype=np.int64)

    def find(parent_rows: np.ndarray, value: int) -> int:
        while int(parent_rows[value]) != value:
            parent_rows[value] = parent_rows[int(parent_rows[value])]
            value = int(parent_rows[value])
        return value

    first_by_vertex: dict[int, int] = {}
    for face_row, face in enumerate(faces):
        for vertex in face:
            first = first_by_vertex.setdefault(int(vertex), face_row)
            root_a, root_b = find(parents, face_row), find(parents, first)
            if root_a != root_b:
                parents[root_b] = root_a
    component_count = len({find(parents, row) for row in range(faces.shape[0])})
    euler = int(np.unique(faces).size - len(unique_edges) + len(faces))
    if (
        component_count != required_components
        or euler != 2 * required_components - 2 * required_genus
    ):
        raise RuntimeError(
            f"{name} boundary topology has {component_count} components and Euler characteristic "
            f"{euler}; expected {required_components} components and genus {required_genus}."
        )
    del (
        blocks,
        cells,
        facets,
        unique_facets,
        counts,
        faces,
        edges,
        unique_edges,
        edge_counts,
    )
    del first_by_vertex, find, parents, face
    if discovery is not None and discovery.mesh is None:
        raise RuntimeError(f"{name} adaptive discovery omitted its triangle carrier.")
    record: dict[str, object] = {
        "status": "passed",
        "case": name,
        "source_id": profile.source_id,
        "source_revision": profile.source_revision,
        "route": plan.route,
        "analytic_volume_m3": exact_volume,
        "measured_volume_m3": measured_volume,
        "relative_volume_error": relative_volume_error,
        "volume_relative_tolerance": 0.15,
        "discovered_components": component_count,
        "boundary_euler_characteristic": euler,
        "required_genus": required_genus,
        "required_components": required_components,
        "accepted": accepted,
        "fidelity_ladder": ladder,
        "fidelity_ladder_convergence": {
            "relative_volume_errors": ladder_errors,
            "certified_fidelity_uppers_m": fidelity_uppers,
            "certified_enclosure_widths_m3": enclosure_widths,
            "certified_fidelity_strictly_decreasing": fidelity_decreasing,
            "certified_enclosure_strictly_shrinking": enclosure_shrinking,
            "volume_inside_certified_enclosure_every_level": enclosure is not None,
        },
    }
    del plan
    if adaptation_rounds:
        if not isinstance(profile, AnalyticImplicitProfile):
            raise ValueError(
                "Metric adaptation requires the retained analytic authoring profile."
            )
        epochs = []
        current = result
        del result
        adaptation_errors = [relative_volume_error]
        physical_error, vertex_values, initial_physics = _solve_physical_epoch(current)
        physical_errors = [physical_error]
        carrier_sizes = [
            (measured_volume / sum(block.cell_count for block in current.mesh.blocks))
            ** (1.0 / 3.0)
        ]
        physical_epochs = [initial_physics]
        for epoch in range(adaptation_rounds):
            previous_certification = current.certification
            if previous_certification is None or previous_certification.fidelity is None:
                raise RuntimeError(
                    f"{name} metric source lacks source-fidelity certification."
                )
            h = size / 2.0 ** ((epoch + 1) / 3.0)
            sizes = np.asarray((h, h, h / anisotropy), dtype=np.float64)
            vertices = current.mesh.entity_set(0)
            scope = phx.meshing.MeshingScope(
                current.mesh.mesh_id,
                current.mesh.numeric_version,
                phx.meshing.MeshingEntityKind.MESH,
                0,
                vertices.entity_set_id,
                vertices.entity_ids,
            )
            tensors = np.broadcast_to(
                np.diag(1.0 / sizes**2), (current.mesh.coordinates.shape[0], 3, 3)
            ).copy()
            metric = phx.meshing.MeshMetricField(
                scope,
                tensors,
                minimum_size=float(np.min(sizes)),
                maximum_size=h,
                maximum_anisotropy=anisotropy,
            )
            adaptation = phx.meshing.execute_mesh_adaptation(
                phx.meshing.prepare_mesh_adaptation(
                    current,
                    phx.meshing.MetricMeshAdaptation(metric),
                    policy=phx.meshing.MeshAdaptationPolicy(
                        phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_3D,
                        association_transfer=phx.meshing.ImplicitAssociationTransfer(
                            profile
                        ),
                    ),
                )
            )
            if not adaptation.status.converged or adaptation.target is current:
                raise RuntimeError(
                    f"{name} metric epoch {epoch} did not complete: {adaptation.status.value}."
                )
            if not isinstance(adaptation.evidence, TetraMetricEvidence):
                raise RuntimeError(
                    f"{name} metric epoch {epoch} omitted tetrahedral metric evidence."
                )
            if adaptation.transfer is None or adaptation.metric is None:
                raise RuntimeError(
                    f"{name} metric epoch {epoch} omitted field transfer or target metric."
                )
            carried = np.asarray(
                adaptation.transfer.apply(vertex_values), dtype=np.float64
            )
            constant = np.asarray(
                adaptation.transfer.apply(np.ones_like(vertex_values)), dtype=np.float64
            )
            if not np.all(np.isfinite(carried)) or not np.allclose(
                constant, 1.0, rtol=0.0, atol=1.0e-12
            ):
                raise RuntimeError(
                    f"{name} metric epoch {epoch} failed P1 field transfer."
                )
            current = adaptation.target
            if current.certification is None or current.certification.fidelity is None:
                raise RuntimeError(
                    f"{name} metric adaptation dropped source-fidelity certification."
                )
            if current.certification.inputs.fidelity_tolerance != level_tolerance:
                raise RuntimeError(
                    f"{name} metric adaptation changed the requested fidelity."
                )
            if (
                current.certification.fidelity.certificate_id
                == previous_certification.fidelity.certificate_id
            ):
                raise RuntimeError(
                    f"{name} metric adaptation reused a stale fidelity certificate."
                )
            volume = float(
                np.sum(np.asarray(current.quality.evaluation.measures, dtype=np.float64))
            )
            error = abs(volume - exact_volume) / exact_volume
            adaptation_errors.append(error)
            if error > 0.15:
                raise RuntimeError(
                    f"{name} metric adaptation exceeded the volume tolerance: {error}."
                )
            # Every tensor is the same broadcast diagonal constructed above.
            eigenvalues = np.linalg.eigvalsh(tensors[0])
            epochs.append(
                {
                    "status": adaptation.status.value,
                    "metric_id": metric.metric_id,
                    "axis_sizes": sizes.tolist(),
                    "tensor_anisotropy": float(np.sqrt(eigenvalues[-1] / eigenvalues[0])),
                    "minimum_metric_quality": adaptation.evidence.minimum_metric_quality,
                    "minimum_metric_length": adaptation.evidence.minimum_metric_length,
                    "maximum_metric_length": adaptation.evidence.maximum_metric_length,
                    "out_of_range_edges": adaptation.evidence.out_of_range_edges,
                    "target_metric_id": adaptation.metric.metric_id,
                    "result": accepted_result_record(current),
                    "measured_volume_m3": volume,
                    "relative_volume_error": error,
                    "source_fidelity_certificate_id": current.certification.fidelity.certificate_id,
                }
            )
            # The completed transfer and source field are not inputs to the
            # target solve; retain the target, carried field, and scalar evidence.
            del adaptation, metric, tensors, vertices, scope, eigenvalues
            del vertex_values, constant, previous_certification
            physical_error, vertex_values, physical_record = _solve_physical_epoch(
                current, carried
            )
            del carried
            physical_errors.append(physical_error)
            carrier_sizes.append(
                (volume / sum(block.cell_count for block in current.mesh.blocks))
                ** (1.0 / 3.0)
            )
            physical_epochs.append(physical_record)
        if any(b >= a for a, b in zip(physical_errors, physical_errors[1:])):
            raise RuntimeError(
                f"{name} manufactured diffusion error did not strictly decrease."
            )
        orders = [
            math.log(a / b) / math.log(coarse / fine)
            if a > 0.0 and b > 0.0 and coarse > fine
            else None
            for a, b, coarse, fine in zip(
                physical_errors,
                physical_errors[1:],
                carrier_sizes,
                carrier_sizes[1:],
                strict=False,
            )
        ]
        if physical_errors[-1] > 0.05 or (
            adaptation_rounds > 1
            and any(order is None or order < 1.0 for order in orders)
        ):
            raise RuntimeError(
                f"{name} manufactured diffusion failed its error/order target: {orders}."
            )
        record["physical_error_convergence"] = {
            "epochs": physical_epochs,
            "errors": physical_errors,
            "realized_carrier_sizes": carrier_sizes,
            "observed_orders": orders,
            "target_error": 0.05,
            "minimum_order": 1.0 if adaptation_rounds > 1 else None,
        }
        record["adaptation_volume_errors"] = adaptation_errors
        record["adaptation"] = epochs
    return record


def native_implicit_adaptation() -> dict[str, object]:
    meshcore = require_meshcore()
    cases: list[dict[str, object]] = []
    failures: list[str] = []

    sphere_radius = 0.371
    sphere_model = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), sphere_radius, feature_id="native-qualification:implicit-sphere"
    ).compile()
    sphere = AnalyticImplicitProfile(
        sphere_model, _CONTRACT, source_id="native-qualification:implicit-sphere"
    )
    try:
        cases.append(
            _volume_case(
                "sphere",
                sphere,
                (4.0 / 3.0) * np.pi * sphere_radius**3,
                size=sphere_radius / 2.0,
                tolerance=0.02,
                required_components=1,
                adaptation_rounds=1,
                witness=np.zeros(3),
                enclosure=_ball_enclosure(sphere_radius),
            )
        )
    except Exception as error:
        failures.append(f"sphere: {type(error).__name__}: {error}")
    try:
        cases.append(
            _volume_case(
                "sphere-adaptive-refinement",
                sphere,
                (4.0 / 3.0) * np.pi * sphere_radius**3,
                size=sphere_radius / 2.0,
                tolerance=0.02,
                required_components=1,
                adaptation_rounds=2,
                fidelity_levels=3,
                witness=np.zeros(3),
                enclosure=_ball_enclosure(sphere_radius),
            )
        )
    except Exception as error:
        failures.append(f"sphere-adaptive-refinement: {type(error).__name__}: {error}")

    torus_minor = 0.371
    torus_major = 2.0 * torus_minor
    torus_inner, torus_outer = torus_major - torus_minor, torus_major + torus_minor
    torus_center = torus_minor * np.asarray((0.0331, -0.0191, 0.0466), dtype=np.float64)
    torus_model = phx.geometry.Torus(
        torus_center,
        torus_inner,
        torus_outer,
        feature_id="native-qualification:offset-ring-torus",
    ).compile()
    torus = AnalyticImplicitProfile(
        torus_model, _CONTRACT, source_id="native-qualification:offset-ring-torus"
    )
    try:
        cases.append(
            _volume_case(
                "offset-ring-torus",
                torus,
                2.0 * np.pi**2 * torus_major * torus_minor**2,
                size=torus_minor / 2.0,
                tolerance=0.02,
                required_components=1,
                adaptation_rounds=2,
                anisotropy=2.0,
                required_genus=1,
                fidelity_levels=2,
                witness=torus_center + np.asarray((torus_major, 0.0, 0.0)),
                enclosure=_ring_torus_enclosure(torus_minor, torus_major),
            )
        )
    except Exception as error:
        failures.append(f"offset-ring-torus: {type(error).__name__}: {error}")

    hidden_source_id = "native-qualification:implicit-hidden-component"
    hidden_geometry = phx.geometry.Union(
        (
            phx.geometry.Sphere((-0.6, 0.0, 0.0), 0.3, feature_id="hidden-left"),
            phx.geometry.Sphere((0.6, 0.0, 0.0), 0.3, feature_id="hidden-right"),
        ),
        feature_id="hidden-two-component-union",
    ).compile()
    domain = np.asarray(((-1.2, -0.8, -0.8), (1.2, 0.8, 0.8)), dtype=np.float64)
    # A sharp union is a NONSMOOTH field certificate, which adaptive discovery
    # admits only on explicit request (AdaptiveImplicitSurfacePolicy
    # .allow_nonsmooth_field); the route reuses this same policy.
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.01 / float(np.linalg.norm(domain[1] - domain[0])),
        root_tolerance=1.0e-10,
        allow_nonsmooth_field=True,
    )
    try:
        discovery = discover_adaptive_implicit_surface(
            hidden_geometry,
            domain=domain,
            policy=policy,
            source_id=hidden_source_id,
        )
        source = NativeImplicitSource(
            hidden_geometry,
            hidden_source_id,
            "revision-1",
            discovery=discovery,
        )
        cases.append(
            _volume_case(
                "hidden-disconnected-components",
                source,
                2.0 * (4.0 / 3.0) * np.pi * 0.3**3,
                size=0.14,
                tolerance=0.02,
                required_components=2,
                discovery=discovery,
                policy=policy,
            )
        )
    except Exception as error:
        failures.append(
            f"hidden-disconnected-components: {type(error).__name__}: {error}"
        )

    return {
        "status": "failed" if failures else "passed",
        "workflow": "native adaptive implicit source -> tetrahedral volume mesh -> source-faithful refinement",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#implicit-domain",
        "cases": cases,
        "failures": failures,
        "required_contracts": [
            "analytic sphere refinement with independent exact volume",
            "offset ring torus with independent exact volume and handle topology",
            "disconnected hidden components with independent component and total-volume oracles",
            "source-faithful adaptation with revalidated geometry fidelity",
        ],
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
