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
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

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
from ....linalg import AbstractLinearOperator, DenseLinearOperator
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


_ELASTICITY_NON_GOALS = (
    "dynamic or frequency-domain elasticity",
    "anisotropy, heterogeneity, nonlinear materials, or body forces",
    "contact, fracture propagation, or displacement-discontinuity elements",
    "continuum discretization certification",
)


class ElasticityBoundaryContract3D(StrictModule, NonTrainableState):
    """Exact declared envelope for one static-isotropic 3D boundary route."""

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    displacement_convention: str = eqx.field(static=True)
    traction_convention: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)


class ElasticityLayerKernel3D(eqx.Module):
    """Kelvin displacement and outward-source-traction kernels in 3D."""

    shear_modulus: Array
    poisson_ratio: Array
    contract: ElasticityBoundaryContract3D = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(self, shear_modulus: ArrayLike, poisson_ratio: ArrayLike, /):
        mu = jnp.asarray(shear_modulus, dtype=float)
        nu = jnp.asarray(poisson_ratio, dtype=float)
        if mu.shape != () or not bool(jnp.isfinite(mu) & (mu > 0.0)):
            raise ValueError("shear_modulus must be one finite positive scalar.")
        if nu.shape != () or not bool(jnp.isfinite(nu) & (nu > -1.0) & (nu < 0.5)):
            raise ValueError(
                "poisson_ratio must be finite and lie strictly in (-1, 0.5)."
            )
        contract = ElasticityBoundaryContract3D(
            ambient_dimension=3,
            pde="static isotropic Navier-Cauchy elasticity without body force",
            geometry="off-source points in three-dimensional Euclidean space",
            formulation="Kelvin displacement and source Cauchy-traction fundamental kernels",
            provider="closed-form Kelvin tensor",
            precision=str(jnp.result_type(mu, nu)),
            displacement_convention="Cartesian physical displacement u_i",
            traction_convention=(
                "t_j=sigma_jk n_k with outward source normal; r=target-source"
            ),
            resource_evidence="one fixed 3x3 kernel block per source-target pair",
            error_evidence="closed-form arithmetic only; singular support is rejected",
            non_goals=_ELASTICITY_NON_GOALS,
        )
        self.shear_modulus = mu
        self.poisson_ratio = nu
        self.contract = contract
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "static-isotropic-elasticity-kelvin-3d-v1",
                "mu": float(mu),
                "nu": float(nu),
                "r": "target-source",
                "traction": "outward-source-cauchy",
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    def value(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        """Return U_ij: displacement i from a point force in direction j."""
        difference = jnp.asarray(target) - jnp.asarray(source)
        if difference.shape != (3,):
            raise ValueError("Elasticity kernel points must both have shape (3,).")
        radius_squared = jnp.sum(difference * difference)
        radius = jnp.sqrt(radius_squared)
        safe = eqx.error_if(
            radius,
            radius_squared == 0.0,
            "Kelvin kernel is undefined on its point singularity.",
        )
        identity = jnp.eye(3, dtype=difference.dtype)
        prefactor = 1.0 / (
            16.0 * jnp.pi * self.shear_modulus * (1.0 - self.poisson_ratio)
        )
        return prefactor * (
            (3.0 - 4.0 * self.poisson_ratio) * identity / safe
            + jnp.outer(difference, difference) / (safe * radius_squared)
        )

    def source_traction(
        self,
        target: ArrayLike,
        source: ArrayLike,
        source_normal: ArrayLike,
        /,
    ) -> Array:
        """Return T_ij obtained by applying outward source traction to U_ij."""
        difference = jnp.asarray(target) - jnp.asarray(source)
        normal = jnp.asarray(source_normal, dtype=difference.dtype)
        if difference.shape != (3,) or normal.shape != (3,):
            raise ValueError(
                "Elasticity traction points and normal must have shape (3,)."
            )
        radius_squared = jnp.sum(difference * difference)
        radius = jnp.sqrt(radius_squared)
        safe = eqx.error_if(
            radius,
            radius_squared == 0.0,
            "Kelvin traction kernel is undefined on its point singularity.",
        )
        normal_projection = jnp.dot(difference, normal)
        identity = jnp.eye(3, dtype=difference.dtype)
        skew_source = jnp.outer(normal, difference) - jnp.outer(difference, normal)
        numerator = (1.0 - 2.0 * self.poisson_ratio) * (
            identity * normal_projection + skew_source
        ) + 3.0 * jnp.outer(difference, difference) * normal_projection / radius_squared
        return numerator / (8.0 * jnp.pi * (1.0 - self.poisson_ratio) * safe**3)

    def apply_point_force(
        self,
        target: ArrayLike,
        source: ArrayLike,
        force: ArrayLike,
        /,
    ) -> Array:
        value = jnp.asarray(force)
        if value.shape != (3,):
            raise ValueError("Point force must have shape (3,).")
        return self.value(target, source) @ value


class ElasticityLayerPotential3D(AbstractArrayModel):
    """Finite Kelvin layer sum; exact for the discrete sources off their support."""

    panelization: SurfacePanelization3D
    kernel: ElasticityLayerKernel3D
    density: Array
    kind: Literal["single", "double"] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    contract: ElasticityBoundaryContract3D = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _discretization: LayerDiscretizationReport
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: SurfacePanelization3D,
        /,
        *,
        shear_modulus: ArrayLike,
        poisson_ratio: ArrayLike,
        kind: Literal["single", "double"] = "single",
        density: ArrayLike | None = None,
    ):
        if not isinstance(panelization, SurfacePanelization3D):
            raise TypeError("panelization must be SurfacePanelization3D.")
        if kind not in ("single", "double"):
            raise ValueError("Elasticity layer kind must be 'single' or 'double'.")
        values = (
            jnp.zeros((panelization.node_count, 3), dtype=panelization.points.dtype)
            if density is None
            else jnp.asarray(density, dtype=panelization.points.dtype)
        )
        if values.shape != (panelization.node_count, 3):
            raise ValueError("Elasticity density must have shape (source_node_count, 3).")
        kernel = ElasticityLayerKernel3D(shear_modulus, poisson_ratio)
        representation_id = canonical_fingerprint(
            {
                "kind": "discrete-static-elasticity-layer-potential-3d-v1",
                "kernel": kernel.kernel_id,
                "panelization": panelization.panelization_id,
                "layer": kind,
            }
        )
        self.panelization = panelization
        self.kernel = kernel
        self.density = values
        self.kind = kind
        self.in_size = 3
        self.out_size = 3
        self.contract = ElasticityBoundaryContract3D(
            ambient_dimension=3,
            pde=kernel.contract.pde,
            geometry="oriented triangular quadrature nodes; targets strictly off source support",
            formulation=f"finite {kind}-layer quadrature sum",
            provider="PHYDRAX SurfacePanelization3D direct evaluator",
            precision=kernel.contract.precision,
            displacement_convention=kernel.contract.displacement_convention,
            traction_convention=kernel.contract.traction_convention,
            resource_evidence="one direct kernel action per target-source quadrature-node pair",
            error_evidence=(
                "discrete sum only; no near-surface quadrature or continuum error certification"
            ),
            non_goals=_ELASTICITY_NON_GOALS,
        )
        self.representation_id = representation_id
        self._certificate = TrialSpaceCertificate(
            equation_family="linear-elasticity",
            ambient_dimension=3,
            construction=f"finite-{kind}-kelvin-layer-kernel-sum-3d",
            normalization_id="physical-euclidean-coordinates",
            basis_id=representation_id,
            rank=3 * panelization.node_count,
            assumptions=(
                "static homogeneous isotropic Navier-Cauchy equation",
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

    def with_density(self, density: ArrayLike, /) -> "ElasticityLayerPotential3D":
        values = jnp.asarray(density, dtype=self.density.dtype)
        if values.shape != self.density.shape:
            raise ValueError("Replacement density must preserve (source_node_count, 3).")
        return eqx.tree_at(lambda potential: potential.density, self, values)

    def __call__(self, target: ArrayLike, /, *, key=None) -> Array:
        del key
        point = jnp.asarray(target, dtype=self.panelization.points.dtype)
        if point.shape != (3,):
            raise ValueError("Elasticity layer target must have shape (3,).")
        differences = point[None, :] - self.panelization.points
        point = eqx.error_if(
            point,
            jnp.any(jnp.sum(differences * differences, axis=1) == 0.0),
            "Elasticity layer target intersects its discrete singular support.",
        )
        if self.kind == "single":
            blocks = jax.vmap(self.kernel.value, in_axes=(None, 0))(
                point, self.panelization.points
            )
        else:
            blocks = jax.vmap(self.kernel.source_traction, in_axes=(None, 0, 0))(
                point, self.panelization.points, self.panelization.normals
            )
        return oe.contract(
            "nij,n,nj->i",
            blocks,
            self.panelization.weights,
            self.density,
            backend="jax",
        )

    def _evaluate_direct(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets, dtype=self.panelization.points.dtype)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("Elasticity targets must have shape (target_count, 3).")
        return jax.vmap(self)(values)

    def discretization_report(self) -> LayerDiscretizationReport:
        return self._discretization

    def model_metadata(self) -> Mapping[str, Any]:
        return {TRIAL_SPACE_CERTIFICATE_KEY: self._certificate}


def evaluate_elasticity_layer_3d(
    potential: ElasticityLayerPotential3D,
    targets: ArrayLike,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
    accuracy_clearance: float = 0.0,
) -> tuple[Array, SurfaceTargetReport3D]:
    """Evaluate one Kelvin layer with continuous target-admissibility evidence."""
    if not isinstance(potential, ElasticityLayerPotential3D):
        raise TypeError("potential must be ElasticityLayerPotential3D.")
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
        raise ValueError("Elasticity layer evaluation requires off-surface targets.")
    output = potential._evaluate_direct(values)
    return (output[0] if single else output), report


class ElasticitySingleLayerDP0Policy3D(StrictModule, NonTrainableState):
    """Bounded dense constant-triangle Kelvin Galerkin preparation policy."""

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
            raise ValueError(
                "Elasticity Galerkin quadrature orders must be at least two."
            )
        absolute, relative = float(absolute_tolerance), float(relative_tolerance)
        separation = float(minimum_disjoint_centroid_ratio)
        limits = (
            int(max_face_count),
            int(max_matrix_bytes),
            int(max_preparation_workspace_bytes),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in (absolute, relative)):
            raise ValueError(
                "Elasticity quadrature tolerances must be finite and nonnegative."
            )
        if not math.isfinite(separation) or separation <= 0.0:
            raise ValueError(
                "minimum_disjoint_centroid_ratio must be finite and positive."
            )
        if any(value <= 0 for value in limits):
            raise ValueError("Elasticity Galerkin resource limits must be positive.")
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
                "kind": "elasticity-single-layer-dp0-policy-3d-v1",
                "orders": (regular, singular),
                "tolerances": (absolute, relative),
                "minimum_disjoint_centroid_ratio": separation,
                "limits": limits,
                "precision": precision_.policy_id,
            }
        )


class ElasticityNullspaceMetadata3D(StrictModule, NonTrainableState):
    """DP0 rigid displacements and Neumann force/torque compatibility duals."""

    contract: ElasticityBoundaryContract3D = eqx.field(static=True)
    rigid_displacement_modes: Array
    force_torque_functionals: Array
    reference_origin: Array
    rigid_mode_dimension: int = eqx.field(static=True)
    pressure_nullspace_dimension: int = eqx.field(static=True)
    convention: str = eqx.field(static=True)


class ElasticitySingleLayerDP0AssemblyReport3D(StrictModule, NonTrainableState):
    """Quadrature/resource evidence; it does not certify continuum error."""

    contract: ElasticityBoundaryContract3D = eqx.field(static=True)
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


class ElasticitySingleLayerDP0Galerkin3D(StrictModule, NonTrainableState):
    """Bounded dense 3D static-elasticity DP0 single-layer preparation."""

    weak_operator: AbstractLinearOperator
    strong_operator: AbstractLinearOperator
    panelization: SurfacePanelization3D
    surface_entities: EntitySet
    face_areas: Array
    face_centroids: Array
    face_normals: Array
    kernel: ElasticityLayerKernel3D
    nullspace: ElasticityNullspaceMetadata3D
    assembly_report: ElasticitySingleLayerDP0AssemblyReport3D
    contract: ElasticityBoundaryContract3D = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)

    def potential(self, coefficients: ArrayLike, /) -> ElasticityLayerPotential3D:
        values = jnp.asarray(coefficients, dtype=self.face_areas.dtype)
        if values.shape == (3 * self.face_count,):
            values = values.reshape((self.face_count, 3))
        if values.shape != (self.face_count, 3):
            raise ValueError(
                "Elasticity DP0 coefficients must have shape (face_count, 3)."
            )
        node_density = jnp.repeat(values, self.panelization.nodes_per_panel, axis=0)
        return ElasticityLayerPotential3D(
            self.panelization,
            shear_modulus=self.kernel.shear_modulus,
            poisson_ratio=self.kernel.poisson_ratio,
            kind="single",
            density=node_density,
        )


def _pair_rule(
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


def _kelvin_pair_block(
    test_triangle: np.ndarray,
    source_triangle: np.ndarray,
    test_reference: np.ndarray,
    source_reference: np.ndarray,
    weights: np.ndarray,
    *,
    shear_modulus: float,
    poisson_ratio: float,
) -> np.ndarray:
    test_points = _map_triangle(test_triangle, test_reference)
    source_points = _map_triangle(source_triangle, source_reference)
    difference = test_points - source_points
    radius_squared = np.sum(difference * difference, axis=1)
    if np.any(~np.isfinite(radius_squared)) or np.any(radius_squared <= 0.0):
        raise ValueError(
            "Elasticity transformed quadrature encountered a singular point."
        )
    radius = np.sqrt(radius_squared)
    identity = np.eye(3)
    prefactor = 1.0 / (16.0 * np.pi * shear_modulus * (1.0 - poisson_ratio))
    blocks = prefactor * (
        (3.0 - 4.0 * poisson_ratio) * identity[None, :, :] / radius[:, None, None]
        + difference[:, :, None]
        * difference[:, None, :]
        / (radius_squared * radius)[:, None, None]
    )
    physical_weights = (
        weights * _surface_jacobian(test_triangle) * _surface_jacobian(source_triangle)
    )
    return np.sum(physical_weights[:, None, None] * blocks, axis=0)


def _rigid_and_equilibrium_metadata(
    centroids: np.ndarray,
    areas: np.ndarray,
    contract: ElasticityBoundaryContract3D,
) -> ElasticityNullspaceMetadata3D:
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
    return ElasticityNullspaceMetadata3D(
        contract=contract,
        rigid_displacement_modes=jnp.asarray(rigid),
        force_torque_functionals=jnp.asarray(functionals),
        reference_origin=jnp.asarray(origin),
        rigid_mode_dimension=6,
        pressure_nullspace_dimension=0,
        convention=(
            "translations followed by omega cross (x-origin); compatible Neumann "
            "tractions have zero total force and torque about origin"
        ),
    )


def prepare_elasticity_single_layer_dp0_3d(
    region: MeshRegion,
    /,
    *,
    shear_modulus: ArrayLike,
    poisson_ratio: ArrayLike,
    policy: ElasticitySingleLayerDP0Policy3D | None = None,
    numeric_version: str = "0",
) -> ElasticitySingleLayerDP0Galerkin3D:
    """Prepare a bounded closed-triangle Kelvin DP0 Galerkin operator."""
    selected = ElasticitySingleLayerDP0Policy3D() if policy is None else policy
    if not isinstance(selected, ElasticitySingleLayerDP0Policy3D):
        raise TypeError("policy must be ElasticitySingleLayerDP0Policy3D or None.")
    kernel = ElasticityLayerKernel3D(shear_modulus, poisson_ratio)
    panel_order = max(selected.regular_order, selected.singular_order)
    binding = _SurfaceFEMBinding3D(
        region,
        quadrature_order=panel_order,
        numeric_version=numeric_version,
    )
    if binding.component_count != 1:
        raise ValueError(
            "Elasticity DP0 rigid/force/torque metadata requires one connected surface."
        )
    face_count = int(binding.face_areas.shape[0])
    if face_count > selected.max_face_count:
        raise ValueError("Elasticity DP0 face count exceeds policy max_face_count.")
    matrix_bytes = 2 * (3 * face_count) ** 2 * np.dtype(np.float64).itemsize
    if matrix_bytes > selected.max_matrix_bytes:
        raise ValueError("Elasticity DP0 matrix exceeds policy max_matrix_bytes.")
    maximum_points = max(
        selected.regular_order**4,
        6 * selected.singular_order**4,
    )
    workspace_bytes = maximum_points * 32 * np.dtype(np.float64).itemsize
    if workspace_bytes > selected.max_preparation_workspace_bytes:
        raise ValueError(
            "Elasticity DP0 quadrature exceeds policy max_preparation_workspace_bytes."
        )

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
    mu = float(kernel.shear_modulus)
    nu = float(kernel.poisson_ratio)
    for target in range(face_count):
        for source in range(target, face_count):
            high_order = (
                selected.singular_order
                if target == source or np.intersect1d(faces[target], faces[source]).size
                else selected.regular_order
            )
            high_rule = _pair_rule(faces, target, source, high_order)
            pair_class = high_rule[3]
            if pair_class == 3:
                ratio = np.linalg.norm(centroids[target] - centroids[source]) / max(
                    diameters[target], diameters[source]
                )
                if ratio < selected.minimum_disjoint_centroid_ratio:
                    raise ValueError(
                        "Elasticity disjoint triangles lie outside the declared separation envelope."
                    )
            high = _kelvin_pair_block(
                triangles[target],
                triangles[source],
                high_rule[0],
                high_rule[1],
                high_rule[2],
                shear_modulus=mu,
                poisson_ratio=nu,
            )
            low_order = max(2, high_order - 1)
            low_rule = _pair_rule(faces, target, source, low_order)
            low = _kelvin_pair_block(
                triangles[target],
                triangles[source],
                low_rule[0],
                low_rule[1],
                low_rule[2],
                shear_modulus=mu,
                poisson_ratio=nu,
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

    weak_array = selected.precision.accumulation(jnp.asarray(weak))
    repeated_areas = jnp.repeat(binding.face_areas, 3)
    strong_array = weak_array / repeated_areas[:, None]
    weak_operator = DenseLinearOperator(
        weak_array,
        operator_id=canonical_fingerprint(
            {
                "kind": "elasticity-single-layer-dp0-weak-3d-v1",
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
                "kind": "elasticity-single-layer-dp0-strong-3d-v1",
                "weak": weak_operator.operator_id,
                "areas": array_tree_fingerprint(binding.face_areas),
            }
        ),
    )
    panelization = binding.panelization
    nodes_per_panel = panelization.nodes_per_panel
    normals = np.asarray(panelization.normals)[::nodes_per_panel]
    areas = np.asarray(binding.face_areas)
    contract = ElasticityBoundaryContract3D(
        ambient_dimension=3,
        pde=kernel.contract.pde,
        geometry=(
            "one connected positively oriented watertight nondegenerate triangle MeshRegion; "
            f"nonadjacent centroid ratio >= {selected.minimum_disjoint_centroid_ratio}"
        ),
        formulation=(
            "constant-vector-density weak and mass-inverted strong DP0 Kelvin single layer"
        ),
        provider="PHYDRAX surface FEM binding plus direct product/Duffy pair quadrature",
        precision=selected.precision.policy_id,
        displacement_convention=kernel.contract.displacement_convention,
        traction_convention=kernel.contract.traction_convention,
        resource_evidence=(
            f"faces={face_count}; "
            f"resident_matrix_bytes={int(weak_array.nbytes + strong_array.nbytes)}; "
            f"numeric_workspace_estimate_bytes={workspace_bytes}"
        ),
        error_evidence=(
            "maximum componentwise consecutive-order quadrature discrepancy; "
            "continuum discretization error is not estimated"
        ),
        non_goals=_ELASTICITY_NON_GOALS,
    )
    nullspace = _rigid_and_equilibrium_metadata(centroids, areas, contract)
    finite = jnp.all(jnp.isfinite(weak_array)) & jnp.all(jnp.isfinite(strong_array))
    accuracy = finite & (
        jnp.asarray(maximum_error)
        <= selected.absolute_tolerance + selected.relative_tolerance * maximum_scale
    )
    report = ElasticitySingleLayerDP0AssemblyReport3D(
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
                "kind": "elasticity-single-layer-dp0-report-3d-v1",
                "binding": binding.binding_id,
                "policy": selected.policy_id,
                "kernel": kernel.kernel_id,
                "counts": tuple(counts),
                "maximum_error": maximum_error,
            }
        ),
    )
    return ElasticitySingleLayerDP0Galerkin3D(
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
    "ElasticityBoundaryContract3D",
    "ElasticityLayerKernel3D",
    "ElasticityLayerPotential3D",
    "ElasticityNullspaceMetadata3D",
    "ElasticitySingleLayerDP0AssemblyReport3D",
    "ElasticitySingleLayerDP0Galerkin3D",
    "ElasticitySingleLayerDP0Policy3D",
    "evaluate_elasticity_layer_3d",
    "prepare_elasticity_single_layer_dp0_3d",
]
