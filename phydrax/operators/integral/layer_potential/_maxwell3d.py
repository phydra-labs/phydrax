#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.bem._rwg import RWGSurfaceCurrentSpace3D
from ....geometry import MeshRegion
from ....linalg import DenseLinearOperator, MaterializationPolicy
from ._galerkin3d import (
    LaplaceSingleLayerDP0Galerkin3D,
    LaplaceSingleLayerDP0GalerkinPolicy3D,
    prepare_laplace_single_layer_dp0_3d,
)


class MaxwellEFIEPolicy3D(StrictModule, NonTrainableState):
    """Finite dense RWG-EFIE preparation envelope, excluding low-frequency use."""

    regular_order: int = eqx.field(static=True)
    singular_order: int = eqx.field(static=True)
    near_order: int = eqx.field(static=True)
    near_ratio: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    min_ka: float = eqx.field(static=True)
    max_ka: float = eqx.field(static=True)
    max_kh: float = eqx.field(static=True)
    max_edges: int = eqx.field(static=True)
    max_dense_bytes: int = eqx.field(static=True)
    max_condition_number: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        regular_order: int = 4,
        singular_order: int = 5,
        near_order: int = 4,
        near_ratio: float = 2.0,
        absolute_tolerance: float = 1.0e-6,
        relative_tolerance: float = 1.0e-4,
        min_ka: float = 0.15,
        max_ka: float = 8.0,
        max_kh: float = 1.25,
        max_edges: int = 1024,
        max_dense_bytes: int = 256 * 1024 * 1024,
        max_condition_number: float = 1.0e10,
    ):
        orders = (int(regular_order), int(singular_order), int(near_order))
        values = (
            float(near_ratio),
            float(absolute_tolerance),
            float(relative_tolerance),
            float(min_ka),
            float(max_ka),
            float(max_kh),
            float(max_condition_number),
        )
        if any(order < 2 for order in orders):
            raise ValueError("Maxwell Galerkin quadrature orders must be at least two.")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Maxwell EFIE policy scales and limits must be finite and positive."
            )
        if values[4] <= values[3]:
            raise ValueError("max_ka must exceed min_ka.")
        if int(max_edges) < 1 or int(max_dense_bytes) < 1:
            raise ValueError("Maxwell dense resource limits must be positive.")
        self.regular_order, self.singular_order, self.near_order = orders
        (
            self.near_ratio,
            self.absolute_tolerance,
            self.relative_tolerance,
            self.min_ka,
            self.max_ka,
            self.max_kh,
            self.max_condition_number,
        ) = values
        self.max_edges = int(max_edges)
        self.max_dense_bytes = int(max_dense_bytes)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "maxwell-efie-policy-3d-v1",
                "orders": orders,
                "near_ratio": self.near_ratio,
                "tolerances": (self.absolute_tolerance, self.relative_tolerance),
                "electrical_envelope": (self.min_ka, self.max_ka, self.max_kh),
                "resources": (self.max_edges, self.max_dense_bytes),
                "max_condition_number": self.max_condition_number,
            }
        )


class MaxwellEFIEAssemblyReport3D(StrictModule, NonTrainableState):
    """Discrete EFIE provenance, risk decisions, and numerical evidence."""

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    laplace_report_id: str = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    genus: int = eqx.field(static=True)
    harmonic_dimension: int = eqx.field(static=True)
    characteristic_radius: float = eqx.field(static=True)
    maximum_edge_length: float = eqx.field(static=True)
    electrical_size_ka: float = eqx.field(static=True)
    mesh_phase_kh: float = eqx.field(static=True)
    dense_bytes: int = eqx.field(static=True)
    condition_number: Array
    finite: Array
    discrete_accuracy_supported: Array
    continuum_discretization_error_estimated: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class PreparedMaxwellEFIE3D(StrictModule, NonTrainableState):
    """Prepared dense mass-lumped RWG PEC EFIE for one bounded lossless case."""

    current_space: RWGSurfaceCurrentSpace3D
    operator: DenseLinearOperator
    scalar_layer_matrix: Array
    wavenumber: Array
    wave_impedance: Array
    laplace_substrate: LaplaceSingleLayerDP0Galerkin3D
    assembly_report: MaxwellEFIEAssemblyReport3D
    prepared_id: str = eqx.field(static=True)

    def incident_rhs(self, incident_electric: ArrayLike, /) -> Array:
        """Weak PEC right-hand side -<RWG,E_inc> using centroid mass lumping."""
        return _incident_rhs(self, incident_electric)


class MaxwellElectricFieldReport3D(StrictModule, NonTrainableState):
    """Off-surface dyadic electric-field action evidence without boundary limits."""

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    dense_bytes: int = eqx.field(static=True)
    minimum_distance: float = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class MaxwellElectricFieldAction3D(StrictModule, NonTrainableState):
    """Dense off-surface Maxwell Green-dyadic map from RWG coefficients to E."""

    targets: Array
    operator: DenseLinearOperator
    report: MaxwellElectricFieldReport3D
    action_id: str = eqx.field(static=True)

    def electric_field(self, coefficients: ArrayLike, /) -> Array:
        return self.operator.mv(coefficients).reshape((self.targets.shape[0], 3))

    def transpose_mv(self, field: ArrayLike, /) -> Array:
        values = jnp.asarray(field, dtype=self.operator.matrix.dtype)
        if values.shape != (self.targets.shape[0], 3):
            raise ValueError("field must have shape (target_count, 3).")
        return self.operator.transpose_mv(values.reshape(-1))

    def adjoint_mv(self, field: ArrayLike, /) -> Array:
        values = jnp.asarray(field, dtype=self.operator.matrix.dtype)
        if values.shape != (self.targets.shape[0], 3):
            raise ValueError("field must have shape (target_count, 3).")
        return self.operator.adjoint_mv(values.reshape(-1))


def _incident_rhs(
    prepared: PreparedMaxwellEFIE3D, incident_electric: ArrayLike, /
) -> Array:
    values = jnp.asarray(incident_electric, dtype=prepared.operator.matrix.dtype)
    surface = prepared.current_space.surface
    if values.shape != (surface.face_count, 3):
        raise ValueError(
            f"incident_electric must have shape {(surface.face_count, 3)}; got {values.shape}."
        )
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("incident_electric must be finite.")
    normals = surface.face_normals.astype(values.dtype)
    tangential = values - jnp.sum(values * normals, axis=1)[:, None] * normals
    local_load = jnp.sum(
        prepared.current_space.centroid_basis.astype(values.dtype)
        * (surface.face_areas[:, None, None] * tangential[:, None, :]),
        axis=2,
    )
    rhs = jnp.zeros((surface.edge_count,), dtype=values.dtype)
    return -rhs.at[surface.face_edges.reshape(-1)].add(local_load.reshape(-1))


def prepare_maxwell_efie_3d(
    current_space: RWGSurfaceCurrentSpace3D,
    wavenumber: ArrayLike,
    /,
    *,
    wave_impedance: ArrayLike = 1.0,
    policy: MaxwellEFIEPolicy3D | None = None,
) -> PreparedMaxwellEFIE3D:
    """Prepare the bounded mass-lumped RWG PEC EFIE over a genus-zero surface."""
    if not isinstance(current_space, RWGSurfaceCurrentSpace3D):
        raise TypeError(
            "current_space must be RWGSurfaceCurrentSpace3D; scalar spaces are not accepted."
        )
    selected = MaxwellEFIEPolicy3D() if policy is None else policy
    if not isinstance(selected, MaxwellEFIEPolicy3D):
        raise TypeError("policy must be MaxwellEFIEPolicy3D or None.")
    surface = current_space.surface
    topology = surface.topology_report
    if (
        topology.component_count != 1
        or topology.genus != 0
        or topology.harmonic_dimension != 0
    ):
        raise ValueError(
            "PEC EFIE foundation accepts one connected genus-zero surface only; "
            f"got components={topology.component_count}, genus={topology.genus}, "
            f"harmonic_dimension={topology.harmonic_dimension}."
        )
    if surface.edge_count > selected.max_edges:
        raise ValueError(
            "RWG edge count exceeds the bounded dense EFIE max_edges policy."
        )
    real_dtype = surface.vertices.dtype
    k = jnp.asarray(wavenumber, dtype=real_dtype)
    eta = jnp.asarray(wave_impedance, dtype=real_dtype)
    if k.shape != () or not bool(jnp.isfinite(k) & (k > 0.0)):
        raise ValueError("wavenumber must be one finite positive real scalar.")
    if eta.shape != () or not bool(jnp.isfinite(eta) & (eta > 0.0)):
        raise ValueError("wave_impedance must be one finite positive real scalar.")
    center = jnp.mean(surface.vertices, axis=0)
    radius = float(jnp.max(jnp.linalg.norm(surface.vertices - center, axis=1)))
    h = float(jnp.max(surface.edge_lengths))
    ka = float(k) * radius
    kh = float(k) * h
    if ka < selected.min_ka:
        raise ValueError(
            f"Low-frequency EFIE rejected: ka={ka:.6g} is below min_ka={selected.min_ka:.6g}."
        )
    if ka > selected.max_ka:
        raise ValueError("Electrical size exceeds the bounded dense EFIE max_ka policy.")
    if kh > selected.max_kh:
        raise ValueError("Mesh phase risk rejected: k*h exceeds the max_kh policy.")
    face_entries = surface.face_count * surface.face_count
    edge_entries = surface.edge_count * surface.edge_count
    dense_bytes = 16 * (face_entries + edge_entries)
    if dense_bytes > selected.max_dense_bytes:
        raise ValueError("Maxwell dense resident estimate exceeds max_dense_bytes.")

    laplace_policy = LaplaceSingleLayerDP0GalerkinPolicy3D(
        regular_order=selected.regular_order,
        singular_order=selected.singular_order,
        near_order=selected.near_order,
        near_ratio=selected.near_ratio,
        absolute_tolerance=selected.absolute_tolerance,
        relative_tolerance=selected.relative_tolerance,
        dense_oracle=MaterializationPolicy(
            max_entries=face_entries,
            max_bytes=max(8 * face_entries, 1),
        ),
    )
    laplace = prepare_laplace_single_layer_dp0_3d(
        MeshRegion(surface.vertices, surface.triangles), policy=laplace_policy
    )
    if laplace.dense_oracle is None or not bool(
        laplace.assembly_report.accuracy_supported
    ):
        raise ValueError(
            "Laplace DP0 Galerkin substrate did not provide finite quadrature evidence."
        )
    weak_laplace = surface.face_areas[:, None] * laplace.dense_oracle.matrix
    differences = surface.face_centroids[:, None, :] - surface.face_centroids[None, :, :]
    distances = jnp.linalg.norm(differences, axis=-1)
    safe_distances = jnp.where(distances > 0.0, distances, 1.0)
    correction = (jnp.exp(1j * k * safe_distances) - 1.0) / (
        4.0 * jnp.pi * safe_distances
    )
    correction = jnp.where(distances > 0.0, correction, 1j * k / (4.0 * jnp.pi))
    scalar_layer = weak_laplace.astype(correction.dtype) + (
        surface.face_areas[:, None] * surface.face_areas[None, :] * correction
    )
    basis = current_space.centroid_basis.astype(scalar_layer.dtype)
    vector_term = jnp.zeros(
        (surface.edge_count, surface.edge_count), dtype=scalar_layer.dtype
    )
    for component in range(3):
        component_basis = (
            jnp.zeros((surface.face_count, surface.edge_count), dtype=scalar_layer.dtype)
            .at[
                jnp.repeat(jnp.arange(surface.face_count), 3),
                surface.face_edges.reshape(-1),
            ]
            .set(basis[:, :, component].reshape(-1))
        )
        vector_term = vector_term + component_basis.T @ scalar_layer @ component_basis
    divergence = current_space.divergence_matrix.astype(scalar_layer.dtype)
    scalar_term = divergence.T @ scalar_layer @ divergence
    matrix = 1j * eta * (k * vector_term - scalar_term / k)
    condition = jnp.linalg.cond(matrix)
    if not bool(jnp.all(jnp.isfinite(matrix)) & jnp.isfinite(condition)):
        raise ValueError(
            "EFIE assembly produced non-finite matrix or condition evidence."
        )
    if float(condition) > selected.max_condition_number:
        raise ValueError(
            "Dense EFIE breakdown risk rejected: matrix condition number exceeds policy."
        )
    operator = DenseLinearOperator(
        matrix,
        source=current_space.vector_space,
        target=current_space.vector_space,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-maxwell-efie-3d-v1",
            "space": current_space.space_id,
            "policy": selected.policy_id,
            "wavenumber": array_tree_fingerprint(k),
            "wave_impedance": array_tree_fingerprint(eta),
        }
    )
    report = MaxwellEFIEAssemblyReport3D(
        ambient_dimension=3,
        pde="source-free time-harmonic Maxwell equations with exp(-i omega t) convention",
        geometry="one connected oriented closed genus-zero piecewise-planar triangular PEC boundary",
        formulation="mass-lumped RWG weak EFIE: i*k*eta*<f,S[J]> - i*eta/k*<div f,S[div J]>",
        provider="Phydra Laplace DP0 singular Galerkin plus analytic smooth Helmholtz correction",
        precision=str(matrix.dtype),
        resource_evidence=(
            f"dense {surface.edge_count}x{surface.edge_count} complex matrix; "
            f"estimated {dense_bytes} resident bytes"
        ),
        error_evidence=(
            "Laplace singular quadrature report retained; Helmholtz smooth correction "
            "and RWG products use centroid mass lumping without continuum error bound"
        ),
        non_goals=(
            "continuum certification",
            "low-frequency stabilization",
            "multiply connected harmonic currents",
            "BC/RBC or Calderon preconditioning",
            "CFIE or interior-resonance removal",
            "fast or matrix-free application",
        ),
        policy_id=selected.policy_id,
        laplace_report_id=laplace.assembly_report.report_id,
        edge_count=surface.edge_count,
        face_count=surface.face_count,
        component_count=topology.component_count,
        genus=topology.genus,
        harmonic_dimension=topology.harmonic_dimension,
        characteristic_radius=radius,
        maximum_edge_length=h,
        electrical_size_ka=ka,
        mesh_phase_kh=kh,
        dense_bytes=dense_bytes,
        condition_number=condition,
        finite=jnp.asarray(True),
        discrete_accuracy_supported=jnp.asarray(True),
        continuum_discretization_error_estimated=False,
        report_id=canonical_fingerprint(
            {"kind": "maxwell-efie-assembly-report-3d-v1", "prepared": prepared_id}
        ),
    )
    prepared = PreparedMaxwellEFIE3D(
        current_space=current_space,
        operator=operator,
        scalar_layer_matrix=scalar_layer,
        wavenumber=k,
        wave_impedance=eta,
        laplace_substrate=laplace,
        assembly_report=report,
        prepared_id=prepared_id,
    )
    return prepared


def _point_triangle_distance(point: np.ndarray, triangle: np.ndarray, /) -> float:
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return float(np.linalg.norm(ap - (d1 / (d1 - d3)) * ab))
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return float(np.linalg.norm(ap - (d2 / (d2 - d6)) * ac))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        weight = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(bp - weight * (c - b)))
    normal = np.cross(ab, ac)
    return abs(float(ap @ normal)) / float(np.linalg.norm(normal))


def prepare_maxwell_electric_field_action_3d(
    prepared: PreparedMaxwellEFIE3D,
    targets: ArrayLike,
    /,
    *,
    minimum_clearance_h: float = 0.25,
    max_targets: int = 4096,
    max_dense_bytes: int = 256 * 1024 * 1024,
) -> MaxwellElectricFieldAction3D:
    """Prepare the off-surface electric Green-dyadic action at fixed targets."""
    if not isinstance(prepared, PreparedMaxwellEFIE3D):
        raise TypeError("prepared must be PreparedMaxwellEFIE3D.")
    points = np.asarray(
        targets, dtype=np.asarray(prepared.current_space.surface.vertices).dtype
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] == 0
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "targets must be one nonempty finite array of shape (target_count, 3)."
        )
    target_limit = int(max_targets)
    byte_limit = int(max_dense_bytes)
    if target_limit < 1 or byte_limit < 1:
        raise ValueError("Electric-field target and dense-byte limits must be positive.")
    if points.shape[0] > target_limit:
        raise ValueError("Electric-field target count exceeds max_targets.")
    dense_bytes = 16 * points.shape[0] * 3 * prepared.current_space.size
    if dense_bytes > byte_limit:
        raise ValueError("Electric-field action exceeds max_dense_bytes.")
    clearance = float(minimum_clearance_h)
    if not math.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("minimum_clearance_h must be finite and positive.")
    surface = prepared.current_space.surface
    triangles = np.asarray(surface.vertices)[np.asarray(surface.triangles)]
    minimum_distance = min(
        _point_triangle_distance(point, triangle)
        for point in points
        for triangle in triangles
    )
    h = float(jnp.max(surface.edge_lengths))
    if minimum_distance < clearance * h:
        raise ValueError(
            "Electric-field targets are too close to the singular surface for the centroid action."
        )
    target_array = jnp.asarray(points)
    difference = target_array[:, None, :] - surface.face_centroids[None, :, :]
    radius = jnp.linalg.norm(difference, axis=-1)
    direction = difference / radius[:, :, None]
    k = prepared.wavenumber
    green = jnp.exp(1j * k * radius) / (4.0 * jnp.pi * radius)
    radial_first = green * (1j * k * radius - 1.0) / (radius**2)
    radial_second = green * (-(k**2) - 2j * k / radius + 2.0 / radius**2)
    identity_coefficient = green + radial_first / (k**2)
    radial_coefficient = (radial_second - radial_first) / (k**2)
    identity = jnp.eye(3, dtype=green.dtype)
    outer = direction[:, :, :, None] * direction[:, :, None, :]
    dyadic = (
        identity_coefficient[:, :, None, None] * identity
        + radial_coefficient[:, :, None, None] * outer
    )
    local_basis = prepared.current_space.centroid_basis.astype(green.dtype)
    face_basis = jnp.zeros((surface.face_count, 3, surface.edge_count), dtype=green.dtype)
    face_ids = jnp.repeat(jnp.arange(surface.face_count), 3)
    for component in range(3):
        face_basis = face_basis.at[
            face_ids,
            component,
            surface.face_edges.reshape(-1),
        ].set(local_basis[:, :, component].reshape(-1))
    matrix = (
        1j
        * k
        * prepared.wave_impedance
        * jnp.sum(
            (dyadic @ face_basis[None, :, :, :])
            * surface.face_areas[None, :, None, None],
            axis=1,
        )
    )
    flat_matrix = matrix.reshape((points.shape[0] * 3, surface.edge_count))
    if not bool(jnp.all(jnp.isfinite(flat_matrix))):
        raise ValueError("Electric Green-dyadic action produced non-finite coefficients.")
    operator = DenseLinearOperator(
        flat_matrix,
        source=prepared.current_space.vector_space,
    )
    action_id = canonical_fingerprint(
        {
            "kind": "maxwell-electric-field-action-3d-v1",
            "prepared": prepared.prepared_id,
            "targets": array_tree_fingerprint(points),
            "clearance_h": clearance,
            "resource_limits": (target_limit, byte_limit),
        }
    )
    report = MaxwellElectricFieldReport3D(
        ambient_dimension=3,
        pde="outgoing source-free time-harmonic Maxwell electric field away from the current support",
        geometry="fixed targets separated from a closed piecewise-planar RWG surface",
        formulation="electric Green dyadic (I + grad grad/k^2) exp(ikr)/(4*pi*r), centroid surface quadrature",
        provider="phydrax Maxwell dyadic direct dense action",
        precision=str(flat_matrix.dtype),
        resource_evidence=(
            f"dense {(points.shape[0] * 3)}x{surface.edge_count} complex action; "
            f"estimated {dense_bytes} bytes"
        ),
        error_evidence=(
            "exact target-to-triangle clearance; centroid quadrature has no continuum "
            "error certificate"
        ),
        non_goals=(
            "on-surface traces",
            "jump relations",
            "near-singular quadrature",
            "far-field acceleration",
        ),
        target_count=int(points.shape[0]),
        dense_bytes=dense_bytes,
        minimum_distance=minimum_distance,
        report_id=canonical_fingerprint(
            {"kind": "maxwell-electric-field-report-3d-v1", "action": action_id}
        ),
    )
    return MaxwellElectricFieldAction3D(
        targets=target_array,
        operator=operator,
        report=report,
        action_id=action_id,
    )


__all__ = [
    "MaxwellEFIEAssemblyReport3D",
    "MaxwellEFIEPolicy3D",
    "MaxwellElectricFieldAction3D",
    "MaxwellElectricFieldReport3D",
    "PreparedMaxwellEFIE3D",
    "prepare_maxwell_efie_3d",
    "prepare_maxwell_electric_field_action_3d",
]
