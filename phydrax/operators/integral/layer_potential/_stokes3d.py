#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._model import AbstractArrayModel
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import EntitySet
from ....equations.trefftz._core import (
    TRIAL_SPACE_CERTIFICATE_KEY,
    TrialSpaceCertificate,
)
from ....geometry import MeshRegion
from ....integration import IntegrationPrecisionPolicy
from ....linalg import AbstractLinearOperator, DenseLinearOperator, LinearCapabilityError
from ._core import LayerDiscretizationReport
from ._galerkin_quadrature3d import (
    _duffy_rule,
    _map_triangle,
    _regular_rule,
    _remap_edge,
    _remap_vertex,
    _surface_jacobian,
)
from ._surface3d import SurfacePanelization3D, SurfaceTargetReport3D
from ._surface_fem3d import _SurfaceFEMBinding3D


_STOKES_NON_GOALS = (
    "unsteady, inertial, Oseen, or Navier-Stokes flow",
    "variable viscosity, non-Newtonian rheology, or body forces",
    "open surfaces, contact, or free-surface evolution",
    "continuum discretization certification",
)


class StokesBoundaryContract3D(StrictModule, NonTrainableState):
    """Exact declared envelope for one steady incompressible Stokes route."""

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    velocity_convention: str = eqx.field(static=True)
    traction_pressure_convention: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)


class StokesLayerKernel3D(eqx.Module):
    """Stokeslet, pressure vector, and outward-source stresslet in 3D."""

    viscosity: Array
    contract: StokesBoundaryContract3D = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(self, viscosity: ArrayLike, /):
        mu = jnp.asarray(viscosity, dtype=float)
        if mu.shape != () or not bool(jnp.isfinite(mu) & (mu > 0.0)):
            raise ValueError("viscosity must be one finite positive scalar.")
        contract = StokesBoundaryContract3D(
            ambient_dimension=3,
            pde="steady incompressible constant-viscosity Stokes equations without body force",
            geometry="off-source points in three-dimensional Euclidean space",
            formulation="Stokeslet velocity, pressure vector, and source stresslet kernels",
            provider="closed-form free-space Stokes fundamental solution",
            precision=str(mu.dtype),
            velocity_convention="Cartesian velocity u_i from force density component f_j",
            traction_pressure_convention=(
                "sigma=-p I+mu(grad u+grad u^T); outward source normal; r=target-source"
            ),
            resource_evidence="fixed 3x3 velocity/stresslet block and 3-vector pressure per pair",
            error_evidence="closed-form arithmetic only; singular support is rejected",
            non_goals=_STOKES_NON_GOALS,
        )
        self.viscosity = mu
        self.contract = contract
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "steady-stokes-free-space-kernel-3d-v1",
                "viscosity": float(mu),
                "r": "target-source",
                "traction": "outward-source-stresslet",
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    def value(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        """Return G_ij: velocity i from a point force in direction j."""
        difference = jnp.asarray(target) - jnp.asarray(source)
        if difference.shape != (3,):
            raise ValueError("Stokes kernel points must both have shape (3,).")
        radius_squared = jnp.sum(difference * difference)
        radius = jnp.sqrt(radius_squared)
        safe = eqx.error_if(
            radius,
            radius_squared == 0.0,
            "Stokeslet is undefined on its point singularity.",
        )
        identity = jnp.eye(3, dtype=difference.dtype)
        return (
            identity / safe + jnp.outer(difference, difference) / (safe * radius_squared)
        ) / (8.0 * jnp.pi * self.viscosity)

    def pressure_vector(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        """Return P_j such that a point force f produces pressure P_j f_j."""
        difference = jnp.asarray(target) - jnp.asarray(source)
        if difference.shape != (3,):
            raise ValueError("Stokes pressure kernel points must have shape (3,).")
        radius_squared = jnp.sum(difference * difference)
        radius = jnp.sqrt(radius_squared)
        safe = eqx.error_if(
            radius,
            radius_squared == 0.0,
            "Stokes pressure kernel is undefined on its point singularity.",
        )
        return difference / (4.0 * jnp.pi * safe * radius_squared)

    def source_traction(
        self,
        target: ArrayLike,
        source: ArrayLike,
        source_normal: ArrayLike,
        /,
    ) -> Array:
        """Return source stresslet T_ij=-3 r_i r_j(r.n)/(4 pi r^5)."""
        difference = jnp.asarray(target) - jnp.asarray(source)
        normal = jnp.asarray(source_normal, dtype=difference.dtype)
        if difference.shape != (3,) or normal.shape != (3,):
            raise ValueError("Stokes stresslet points and normal must have shape (3,).")
        radius_squared = jnp.sum(difference * difference)
        radius = jnp.sqrt(radius_squared)
        safe = eqx.error_if(
            radius,
            radius_squared == 0.0,
            "Stokes stresslet is undefined on its point singularity.",
        )
        projection = jnp.dot(difference, normal)
        return (
            -3.0
            * jnp.outer(difference, difference)
            * projection
            / (4.0 * jnp.pi * safe * radius_squared * radius_squared)
        )

    def apply_point_force(
        self,
        target: ArrayLike,
        source: ArrayLike,
        force: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        value = jnp.asarray(force)
        if value.shape != (3,):
            raise ValueError("Point force must have shape (3,).")
        return self.value(target, source) @ value, jnp.dot(
            self.pressure_vector(target, source), value
        )


class StokesLayerPotential3D(AbstractArrayModel):
    """Finite steady-Stokes layer sum off its discrete source support."""

    panelization: SurfacePanelization3D
    kernel: StokesLayerKernel3D
    density: Array
    kind: Literal["single", "double"] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    contract: StokesBoundaryContract3D = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _discretization: LayerDiscretizationReport
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: SurfacePanelization3D,
        /,
        *,
        viscosity: ArrayLike,
        kind: Literal["single", "double"] = "single",
        density: ArrayLike | None = None,
    ):
        if not isinstance(panelization, SurfacePanelization3D):
            raise TypeError("panelization must be SurfacePanelization3D.")
        if kind not in ("single", "double"):
            raise ValueError("Stokes layer kind must be 'single' or 'double'.")
        values = (
            jnp.zeros((panelization.node_count, 3), dtype=panelization.points.dtype)
            if density is None
            else jnp.asarray(density, dtype=panelization.points.dtype)
        )
        if values.shape != (panelization.node_count, 3):
            raise ValueError("Stokes density must have shape (source_node_count, 3).")
        kernel = StokesLayerKernel3D(viscosity)
        representation_id = canonical_fingerprint(
            {
                "kind": "discrete-steady-stokes-layer-potential-3d-v1",
                "kernel": kernel.kernel_id,
                "panelization": panelization.panelization_id,
                "layer": kind,
            }
        )
        contract = StokesBoundaryContract3D(
            ambient_dimension=3,
            pde=kernel.contract.pde,
            geometry="oriented triangular quadrature nodes; targets strictly off source support",
            formulation=f"finite {kind}-layer quadrature sum",
            provider="PHYDRAX SurfacePanelization3D direct evaluator",
            precision=kernel.contract.precision,
            velocity_convention=kernel.contract.velocity_convention,
            traction_pressure_convention=kernel.contract.traction_pressure_convention,
            resource_evidence="one direct kernel action per target-source quadrature-node pair",
            error_evidence=(
                "discrete sum only; no near-surface quadrature or continuum error certification"
            ),
            non_goals=_STOKES_NON_GOALS,
        )
        self.panelization = panelization
        self.kernel = kernel
        self.density = values
        self.kind = kind
        self.in_size = 3
        self.out_size = 3
        self.contract = contract
        self.representation_id = representation_id
        self._certificate = TrialSpaceCertificate(
            equation_family="stokes",
            ambient_dimension=3,
            construction=f"finite-{kind}-stokes-layer-kernel-sum-3d",
            normalization_id="physical-euclidean-coordinates",
            basis_id=representation_id,
            rank=3 * panelization.node_count,
            assumptions=(
                "steady homogeneous incompressible constant-viscosity Stokes equation",
                "targets lie outside the discrete source singular support",
                f"layer-kernel:{kernel.kernel_id}",
            ),
            construction_residual=0.0,
            construction_tolerance=0.0,
            validity_region="off-singular-support",
            singular_support_id=panelization.source_support_id,
        )
        self._discretization = LayerDiscretizationReport(
            panelization=panelization,
            kernel_id=kernel.kernel_id,
            density_space="three-component-surface-quadrature-node-values",
            trace_policy="off-surface-reference-triangle",
        )

    def with_density(self, density: ArrayLike, /) -> "StokesLayerPotential3D":
        values = jnp.asarray(density, dtype=self.density.dtype)
        if values.shape != self.density.shape:
            raise ValueError("Replacement density must preserve (source_node_count, 3).")
        return eqx.tree_at(lambda potential: potential.density, self, values)

    def __call__(self, target: ArrayLike, /, *, key=None) -> Array:
        del key
        point = jnp.asarray(target, dtype=self.panelization.points.dtype)
        if point.shape != (3,):
            raise ValueError("Stokes layer target must have shape (3,).")
        differences = point[None, :] - self.panelization.points
        point = eqx.error_if(
            point,
            jnp.any(jnp.sum(differences * differences, axis=1) == 0.0),
            "Stokes layer target intersects its discrete singular support.",
        )
        if self.kind == "single":
            blocks = jax.vmap(self.kernel.value, in_axes=(None, 0))(
                point, self.panelization.points
            )
        else:
            blocks = jax.vmap(self.kernel.source_traction, in_axes=(None, 0, 0))(
                point, self.panelization.points, self.panelization.normals
            )
        return ein.contract(
            "nij,n,nj->i",
            blocks,
            self.panelization.weights,
            self.density,
            backend="jax",
        )

    def pressure(self, target: ArrayLike, /) -> Array:
        """Evaluate single-layer pressure; double-layer pressure is unsupported."""
        if self.kind != "single":
            raise LinearCapabilityError(
                "Pressure evaluation for the Stokes double layer is outside this capability."
            )
        point = jnp.asarray(target, dtype=self.panelization.points.dtype)
        if point.shape != (3,):
            raise ValueError("Stokes pressure target must have shape (3,).")
        differences = point[None, :] - self.panelization.points
        point = eqx.error_if(
            point,
            jnp.any(jnp.sum(differences * differences, axis=1) == 0.0),
            "Stokes pressure target intersects its discrete singular support.",
        )
        vectors = jax.vmap(self.kernel.pressure_vector, in_axes=(None, 0))(
            point, self.panelization.points
        )
        return ein.contract(
            "nj,n,nj->",
            vectors,
            self.panelization.weights,
            self.density,
            backend="jax",
        )

    def _evaluate_direct(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets, dtype=self.panelization.points.dtype)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("Stokes targets must have shape (target_count, 3).")
        return jax.vmap(self)(values)

    def discretization_report(self) -> LayerDiscretizationReport:
        return self._discretization

    def model_metadata(self) -> Mapping[str, Any]:
        return {TRIAL_SPACE_CERTIFICATE_KEY: self._certificate}


def evaluate_stokes_layer_3d(
    potential: StokesLayerPotential3D,
    targets: ArrayLike,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
    accuracy_clearance: float = 0.0,
) -> tuple[Array, SurfaceTargetReport3D]:
    """Evaluate velocity with continuous target-admissibility evidence."""
    if not isinstance(potential, StokesLayerPotential3D):
        raise TypeError("potential must be StokesLayerPotential3D.")
    values = jnp.asarray(targets, dtype=potential.panelization.points.dtype)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    report = SurfaceTargetReport3D(
        values,
        potential.panelization,
        target_side=target_side,
        accuracy_clearance=accuracy_clearance,
    )
    if not bool(report.pde_membership_valid):
        raise ValueError("Stokes layer evaluation requires off-surface targets.")
    output = potential._evaluate_direct(values)
    return (output[0] if single else output), report


def evaluate_stokes_pressure_3d(
    potential: StokesLayerPotential3D,
    targets: ArrayLike,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
    accuracy_clearance: float = 0.0,
) -> tuple[Array, SurfaceTargetReport3D]:
    """Evaluate single-layer pressure with target-admissibility evidence."""
    if not isinstance(potential, StokesLayerPotential3D):
        raise TypeError("potential must be StokesLayerPotential3D.")
    if potential.kind != "single":
        raise LinearCapabilityError(
            "Pressure evaluation for the Stokes double layer is outside this capability."
        )
    values = jnp.asarray(targets, dtype=potential.panelization.points.dtype)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    report = SurfaceTargetReport3D(
        values,
        potential.panelization,
        target_side=target_side,
        accuracy_clearance=accuracy_clearance,
    )
    if not bool(report.pde_membership_valid):
        raise ValueError("Stokes pressure evaluation requires off-surface targets.")
    output = jax.vmap(potential.pressure)(values)
    return (output[0] if single else output), report


class StokesSingleLayerDP0Policy3D(StrictModule, NonTrainableState):
    """Bounded dense constant-triangle Stokeslet Galerkin preparation policy."""

    regular_order: int = eqx.field(static=True)
    singular_order: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    minimum_disjoint_centroid_ratio: float = eqx.field(static=True)
    max_face_count: int = eqx.field(static=True)
    max_matrix_bytes: int = eqx.field(static=True)
    max_preparation_workspace_bytes: int = eqx.field(static=True)
    precision: IntegrationPrecisionPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        regular_order: int = 4,
        singular_order: int = 4,
        absolute_tolerance: float = 1.0e-5,
        relative_tolerance: float = 1.0e-3,
        minimum_disjoint_centroid_ratio: float = 2.0e-2,
        max_face_count: int = 256,
        max_matrix_bytes: int = 64 * 1024 * 1024,
        max_preparation_workspace_bytes: int = 64 * 1024 * 1024,
        precision: IntegrationPrecisionPolicy | None = None,
    ):
        regular, singular = int(regular_order), int(singular_order)
        if regular < 2 or singular < 2:
            raise ValueError("Stokes Galerkin quadrature orders must be at least two.")
        absolute, relative = float(absolute_tolerance), float(relative_tolerance)
        separation = float(minimum_disjoint_centroid_ratio)
        limits = (
            int(max_face_count),
            int(max_matrix_bytes),
            int(max_preparation_workspace_bytes),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in (absolute, relative)):
            raise ValueError(
                "Stokes quadrature tolerances must be finite and nonnegative."
            )
        if not math.isfinite(separation) or separation <= 0.0:
            raise ValueError(
                "minimum_disjoint_centroid_ratio must be finite and positive."
            )
        if any(value <= 0 for value in limits):
            raise ValueError("Stokes Galerkin resource limits must be positive.")
        precision_ = IntegrationPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, IntegrationPrecisionPolicy):
            raise TypeError("precision must be IntegrationPrecisionPolicy or None.")
        self.regular_order = regular
        self.singular_order = singular
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.minimum_disjoint_centroid_ratio = separation
        (
            self.max_face_count,
            self.max_matrix_bytes,
            self.max_preparation_workspace_bytes,
        ) = limits
        self.precision = precision_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "stokes-single-layer-dp0-policy-3d-v1",
                "orders": (regular, singular),
                "tolerances": (absolute, relative),
                "minimum_disjoint_centroid_ratio": separation,
                "limits": limits,
                "precision": precision_.policy_id,
            }
        )


class StokesNullspaceMetadata3D(StrictModule, NonTrainableState):
    """Rigid velocities, force/torque moments, and pressure/density gauges."""

    contract: StokesBoundaryContract3D = eqx.field(static=True)
    rigid_velocity_modes: Array
    force_torque_functionals: Array
    single_layer_density_null_vector: Array
    boundary_flux_functional: Array
    reference_origin: Array
    rigid_mode_dimension: int = eqx.field(static=True)
    pressure_nullspace_dimension: int = eqx.field(static=True)
    convention: str = eqx.field(static=True)


class StokesSingleLayerDP0AssemblyReport3D(StrictModule, NonTrainableState):
    """Quadrature/resource evidence; it does not certify continuum error."""

    contract: StokesBoundaryContract3D = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    pair_counts: tuple[int, int, int, int] = eqx.field(static=True)
    quadrature_evaluations: int = eqx.field(static=True)
    maximum_quadrature_error: Array
    preparation_workspace_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    continuum_discretization_error_estimated: bool = eqx.field(static=True)
    finite: Array
    accuracy_supported: Array
    report_id: str = eqx.field(static=True)


class StokesSingleLayerDP0Galerkin3D(StrictModule, NonTrainableState):
    """Bounded dense 3D steady-Stokes DP0 single-layer preparation."""

    weak_operator: AbstractLinearOperator
    strong_operator: AbstractLinearOperator
    panelization: SurfacePanelization3D
    surface_entities: EntitySet
    face_areas: Array
    face_centroids: Array
    face_normals: Array
    kernel: StokesLayerKernel3D
    nullspace: StokesNullspaceMetadata3D
    assembly_report: StokesSingleLayerDP0AssemblyReport3D
    contract: StokesBoundaryContract3D = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)

    def potential(self, coefficients: ArrayLike, /) -> StokesLayerPotential3D:
        values = jnp.asarray(coefficients, dtype=self.face_areas.dtype)
        if values.shape == (3 * self.face_count,):
            values = values.reshape((self.face_count, 3))
        if values.shape != (self.face_count, 3):
            raise ValueError("Stokes DP0 coefficients must have shape (face_count, 3).")
        node_density = jnp.repeat(values, self.panelization.nodes_per_panel, axis=0)
        return StokesLayerPotential3D(
            self.panelization,
            viscosity=self.kernel.viscosity,
            kind="single",
            density=node_density,
        )


def _stokes_pair_rule(
    faces: np.ndarray,
    target: int,
    source: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    shared = sorted(set(map(int, faces[target])) & set(map(int, faces[source])))
    if target == source:
        test_reference, source_reference, weights = _duffy_rule(order, "coincident")
        return test_reference, source_reference, weights, 0
    if len(shared) == 2:
        test_reference, source_reference, weights = _duffy_rule(order, "shared-edge")
        test_local = tuple(
            int(np.flatnonzero(faces[target] == value)[0]) for value in shared
        )
        source_local = tuple(
            int(np.flatnonzero(faces[source] == value)[0]) for value in shared
        )
        return (
            _remap_edge(test_reference, *test_local),
            _remap_edge(source_reference, *source_local),
            weights,
            1,
        )
    if len(shared) == 1:
        test_reference, source_reference, weights = _duffy_rule(order, "shared-vertex")
        test_local = int(np.flatnonzero(faces[target] == shared[0])[0])
        source_local = int(np.flatnonzero(faces[source] == shared[0])[0])
        return (
            _remap_vertex(test_reference, test_local),
            _remap_vertex(source_reference, source_local),
            weights,
            2,
        )
    points, one_weights = _regular_rule(order)
    count = points.shape[0]
    return (
        np.repeat(points, count, axis=0),
        np.tile(points, (count, 1)),
        (one_weights[:, None] * one_weights[None, :]).reshape(-1),
        3,
    )


def _stokeslet_pair_block(
    test_triangle: np.ndarray,
    source_triangle: np.ndarray,
    test_reference: np.ndarray,
    source_reference: np.ndarray,
    weights: np.ndarray,
    *,
    viscosity: float,
) -> np.ndarray:
    test_points = _map_triangle(test_triangle, test_reference)
    source_points = _map_triangle(source_triangle, source_reference)
    difference = test_points - source_points
    radius_squared = np.sum(difference * difference, axis=1)
    if np.any(~np.isfinite(radius_squared)) or np.any(radius_squared <= 0.0):
        raise ValueError("Stokes transformed quadrature encountered a singular point.")
    radius = np.sqrt(radius_squared)
    identity = np.eye(3)
    blocks = (
        identity[None, :, :] / radius[:, None, None]
        + difference[:, :, None]
        * difference[:, None, :]
        / (radius_squared * radius)[:, None, None]
    ) / (8.0 * np.pi * viscosity)
    physical_weights = (
        weights * _surface_jacobian(test_triangle) * _surface_jacobian(source_triangle)
    )
    return np.sum(physical_weights[:, None, None] * blocks, axis=0)


def _stokes_nullspace_metadata(
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    contract: StokesBoundaryContract3D,
) -> StokesNullspaceMetadata3D:
    origin = np.sum(areas[:, None] * centroids, axis=0) / np.sum(areas)
    relative = centroids - origin
    face_count = centroids.shape[0]
    rigid = np.zeros((3 * face_count, 6), dtype=float)
    functionals = np.zeros((6, 3 * face_count), dtype=float)
    for face in range(face_count):
        block = slice(3 * face, 3 * face + 3)
        rigid[block, :3] = np.eye(3)
        rigid[block, 3:] = np.stack(
            tuple(np.cross(np.eye(3)[axis], relative[face]) for axis in range(3)),
            axis=1,
        )
        x, y, z = relative[face]
        cross_matrix = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        functionals[:3, block] = areas[face] * np.eye(3)
        functionals[3:, block] = areas[face] * cross_matrix
    density_null = normals.reshape(-1)
    flux = (areas[:, None] * normals).reshape(-1)
    return StokesNullspaceMetadata3D(
        contract=contract,
        rigid_velocity_modes=jnp.asarray(rigid),
        force_torque_functionals=jnp.asarray(functionals),
        single_layer_density_null_vector=jnp.asarray(density_null),
        boundary_flux_functional=jnp.asarray(flux),
        reference_origin=jnp.asarray(origin),
        rigid_mode_dimension=6,
        pressure_nullspace_dimension=1,
        convention=(
            "translations followed by omega cross (x-origin); density moments are total "
            "force/torque; outward-normal density spans the interior pressure gauge; "
            "admissible Dirichlet velocity has zero outward volume flux"
        ),
    )


def prepare_stokes_single_layer_dp0_3d(
    region: MeshRegion,
    /,
    *,
    viscosity: ArrayLike,
    policy: StokesSingleLayerDP0Policy3D | None = None,
    numeric_version: str = "0",
) -> StokesSingleLayerDP0Galerkin3D:
    """Prepare a bounded closed-triangle Stokeslet DP0 Galerkin operator."""
    selected = StokesSingleLayerDP0Policy3D() if policy is None else policy
    if not isinstance(selected, StokesSingleLayerDP0Policy3D):
        raise TypeError("policy must be StokesSingleLayerDP0Policy3D or None.")
    kernel = StokesLayerKernel3D(viscosity)
    panel_order = max(selected.regular_order, selected.singular_order)
    binding = _SurfaceFEMBinding3D(
        region,
        quadrature_order=panel_order,
        numeric_version=numeric_version,
    )
    if binding.component_count != 1:
        raise ValueError(
            "Stokes DP0 pressure/nullspace metadata requires one connected surface."
        )
    face_count = int(binding.face_areas.shape[0])
    if face_count > selected.max_face_count:
        raise ValueError("Stokes DP0 face count exceeds policy max_face_count.")
    matrix_bytes = 2 * (3 * face_count) ** 2 * np.dtype(np.float64).itemsize
    if matrix_bytes > selected.max_matrix_bytes:
        raise ValueError("Stokes DP0 matrix exceeds policy max_matrix_bytes.")
    maximum_points = max(selected.regular_order**4, 6 * selected.singular_order**4)
    workspace_bytes = (maximum_points * 32 + 3 * (3 * face_count) ** 2) * np.dtype(
        np.float64
    ).itemsize
    if workspace_bytes > selected.max_preparation_workspace_bytes:
        raise ValueError("Stokes DP0 quadrature exceeds policy workspace capacity.")

    vertices = np.asarray(region.triangle_mesh.vertices, dtype=float)
    faces = np.asarray(region.triangle_mesh.faces, dtype=np.int32)
    triangles = vertices[faces]
    centroids = np.mean(triangles, axis=1)
    diameters = np.max(
        np.stack(
            (
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            ),
            axis=1,
        ),
        axis=1,
    )
    weak = np.zeros((3 * face_count, 3 * face_count), dtype=float)
    counts = [0, 0, 0, 0]
    maximum_error = 0.0
    maximum_scale = 0.0
    evaluations = 0
    mu = float(kernel.viscosity)
    for target in range(face_count):
        for source in range(target, face_count):
            high_order = (
                selected.singular_order
                if target == source or np.intersect1d(faces[target], faces[source]).size
                else selected.regular_order
            )
            high_rule = _stokes_pair_rule(faces, target, source, high_order)
            pair_class = high_rule[3]
            if pair_class == 3:
                ratio = np.linalg.norm(centroids[target] - centroids[source]) / max(
                    diameters[target], diameters[source]
                )
                if ratio < selected.minimum_disjoint_centroid_ratio:
                    raise ValueError(
                        "Stokes disjoint triangles lie outside the declared separation envelope."
                    )
            high = _stokeslet_pair_block(
                triangles[target],
                triangles[source],
                high_rule[0],
                high_rule[1],
                high_rule[2],
                viscosity=mu,
            )
            low_order = max(2, high_order - 1)
            low_rule = _stokes_pair_rule(faces, target, source, low_order)
            low = _stokeslet_pair_block(
                triangles[target],
                triangles[source],
                low_rule[0],
                low_rule[1],
                low_rule[2],
                viscosity=mu,
            )
            error = float(np.max(np.abs(high - low)))
            scale = float(np.max(np.abs(high)))
            maximum_error = max(maximum_error, error)
            maximum_scale = max(maximum_scale, scale)
            pair_multiplicity = 1 if target == source else 2
            counts[pair_class] += pair_multiplicity
            evaluations += pair_multiplicity * (high_rule[2].size + low_rule[2].size)
            target_block = slice(3 * target, 3 * target + 3)
            source_block = slice(3 * source, 3 * source + 3)
            weak[target_block, source_block] = high
            weak[source_block, target_block] = high.T
    oriented_area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = oriented_area_vectors / np.linalg.norm(
        oriented_area_vectors, axis=1, keepdims=True
    )
    normal_null = normals.reshape(-1)
    normal_norm_squared = np.dot(normal_null, normal_null)
    left_moment = normal_null @ weak
    right_moment = weak @ normal_null
    projected_weak = (
        weak
        - np.outer(normal_null, left_moment) / normal_norm_squared
        - np.outer(right_moment, normal_null) / normal_norm_squared
        + np.dot(left_moment, normal_null)
        * np.outer(normal_null, normal_null)
        / normal_norm_squared**2
    )
    maximum_error = max(maximum_error, float(np.max(np.abs(projected_weak - weak))))
    maximum_scale = max(maximum_scale, float(np.max(np.abs(projected_weak))))
    weak = projected_weak

    weak_array = selected.precision.accumulation(jnp.asarray(weak))
    repeated_areas = jnp.repeat(binding.face_areas, 3)
    strong_array = weak_array / repeated_areas[:, None]
    weak_operator = DenseLinearOperator(
        weak_array,
        operator_id=canonical_fingerprint(
            {
                "kind": "stokes-single-layer-dp0-weak-3d-v1",
                "binding": binding.binding_id,
                "policy": selected.policy_id,
                "kernel": kernel.kernel_id,
                "matrix": array_tree_fingerprint(weak_array),
            }
        ),
    )
    strong_operator = DenseLinearOperator(
        strong_array,
        operator_id=canonical_fingerprint(
            {
                "kind": "stokes-single-layer-dp0-strong-3d-v1",
                "weak": weak_operator.operator_id,
                "areas": array_tree_fingerprint(binding.face_areas),
            }
        ),
    )
    panelization = binding.panelization
    areas = np.asarray(binding.face_areas)
    contract = StokesBoundaryContract3D(
        ambient_dimension=3,
        pde=kernel.contract.pde,
        geometry=(
            "one connected positively oriented watertight nondegenerate triangle MeshRegion; "
            f"nonadjacent centroid ratio >= {selected.minimum_disjoint_centroid_ratio}"
        ),
        formulation=(
            "constant-vector-density weak and mass-inverted strong DP0 Stokes single "
            "layer with the analytic outward-normal density nullspace projected exactly"
        ),
        provider="PHYDRAX surface FEM binding plus direct product/Duffy pair quadrature",
        precision=selected.precision.policy_id,
        velocity_convention=kernel.contract.velocity_convention,
        traction_pressure_convention=kernel.contract.traction_pressure_convention,
        resource_evidence=(
            f"faces={face_count}; resident_matrix_bytes={int(weak_array.nbytes + strong_array.nbytes)}; "
            f"numeric_workspace_estimate_bytes={workspace_bytes}"
        ),
        error_evidence=(
            "maximum componentwise consecutive-order discrepancy or analytic "
            "normal-nullspace correction; continuum discretization error is not estimated"
        ),
        non_goals=_STOKES_NON_GOALS,
    )
    nullspace = _stokes_nullspace_metadata(centroids, normals, areas, contract)
    finite = jnp.all(jnp.isfinite(weak_array)) & jnp.all(jnp.isfinite(strong_array))
    accuracy = finite & (
        jnp.asarray(maximum_error)
        <= selected.absolute_tolerance + selected.relative_tolerance * maximum_scale
    )
    report = StokesSingleLayerDP0AssemblyReport3D(
        contract=contract,
        binding_id=binding.binding_id,
        policy_id=selected.policy_id,
        kernel_id=kernel.kernel_id,
        numeric_version=binding.numeric_version,
        face_count=face_count,
        component_count=binding.component_count,
        pair_counts=tuple(counts),
        quadrature_evaluations=evaluations,
        maximum_quadrature_error=selected.precision.decision(jnp.asarray(maximum_error)),
        preparation_workspace_bytes=workspace_bytes,
        resident_bytes=int(weak_array.nbytes + strong_array.nbytes),
        continuum_discretization_error_estimated=False,
        finite=finite,
        accuracy_supported=accuracy,
        report_id=canonical_fingerprint(
            {
                "kind": "stokes-single-layer-dp0-report-3d-v1",
                "binding": binding.binding_id,
                "policy": selected.policy_id,
                "kernel": kernel.kernel_id,
                "counts": tuple(counts),
                "maximum_error": maximum_error,
            }
        ),
    )
    return StokesSingleLayerDP0Galerkin3D(
        weak_operator=weak_operator,
        strong_operator=strong_operator,
        panelization=panelization,
        surface_entities=binding.surface_entities,
        face_areas=binding.face_areas,
        face_centroids=jnp.asarray(centroids),
        face_normals=jnp.asarray(normals),
        kernel=kernel,
        nullspace=nullspace,
        assembly_report=report,
        contract=contract,
        face_count=face_count,
        component_count=binding.component_count,
    )


__all__ = [
    "StokesBoundaryContract3D",
    "StokesLayerKernel3D",
    "StokesLayerPotential3D",
    "StokesNullspaceMetadata3D",
    "StokesSingleLayerDP0AssemblyReport3D",
    "StokesSingleLayerDP0Galerkin3D",
    "StokesSingleLayerDP0Policy3D",
    "evaluate_stokes_layer_3d",
    "evaluate_stokes_pressure_3d",
    "prepare_stokes_single_layer_dp0_3d",
]
