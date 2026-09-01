#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....geometry import MeshRegion
from ....linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DualSpace,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ....linalg._operators import _AbstractCostedLinearOperator
from ._galerkin3d import LaplaceSingleLayerDP0GalerkinPolicy3D
from ._galerkin_quadrature3d import (
    _diameter,
    _duffy_rule,
    _map_triangle,
    _prepare_surface_pairs_3d,
    _regular_rule,
    _remap_edge,
    _remap_vertex,
    _subdivide_triangle,
    _surface_jacobian,
    _SurfacePairData3D,
)
from ._helmholtz3d import HelmholtzCombinedField3D, HelmholtzLayerPotential3D
from ._laplace3d import LaplaceLayerPotential3D
from ._scalar_trace import (
    SCALAR_TRACE_CONVENTION_3D,
    ScalarTraceConvention3D,
    UnsupportedScalarBoundarySpaceError,
)
from ._surface3d import SurfacePanelization3D
from ._surface_fem3d import _SurfaceFEMBinding3D


ScalarKernelName3D = Literal["laplace", "modified-helmholtz", "outgoing-helmholtz"]


class ScalarKernelFamily3D(StrictModule, NonTrainableState):
    """One bounded scalar fundamental-solution family in three dimensions.

    The represented homogeneous PDE is stated in ``pde``. Helmholtz uses the
    outgoing Sommerfeld branch. Preparation accepts the family only when its
    dimensionless panel frequency and measured quadrature errors fit the
    requested policy. No continuum discretization certification is made.
    """

    family: ScalarKernelName3D = eqx.field(static=True)
    parameter: float = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    fundamental_solution: str = eqx.field(static=True)
    radiation_condition: str = eqx.field(static=True)
    scalar_dtype: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: ScalarKernelName3D = "laplace",
        /,
        *,
        parameter: float = 0.0,
    ):
        if family not in (
            "laplace",
            "modified-helmholtz",
            "outgoing-helmholtz",
        ):
            raise ValueError("Unsupported three-dimensional scalar kernel family.")
        value = float(parameter)
        if family == "laplace":
            if value != 0.0:
                raise ValueError("Laplace kernel parameter must be zero.")
            pde = "-Delta(u)=0"
            fundamental = "1/(4*pi*r)"
            radiation = "not-applicable"
            scalar_dtype = "real"
        else:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("Kernel parameter must be finite and positive.")
            if family == "modified-helmholtz":
                pde = "(-Delta+kappa^2)u=0"
                fundamental = "exp(-kappa*r)/(4*pi*r)"
                radiation = "decay-at-infinity"
                scalar_dtype = "real"
            else:
                pde = "(-Delta-k^2)u=0"
                fundamental = "exp(i*k*r)/(4*pi*r)"
                radiation = "outgoing-Sommerfeld"
                scalar_dtype = "complex"
        self.family = family
        self.parameter = value
        self.pde = pde
        self.fundamental_solution = fundamental
        self.radiation_condition = radiation
        self.scalar_dtype = scalar_dtype
        self.ambient_dimension = 3
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "scalar-layer-kernel-family-3d-v1",
                "family": family,
                "parameter": value,
                "pde": pde,
                "fundamental_solution": fundamental,
                "normal": "outward-source",
                "radiation": radiation,
            }
        )

    @classmethod
    def laplace(cls) -> "ScalarKernelFamily3D":
        return cls("laplace")

    @classmethod
    def modified_helmholtz(cls, decay: float, /) -> "ScalarKernelFamily3D":
        return cls("modified-helmholtz", parameter=decay)

    @classmethod
    def outgoing_helmholtz(cls, wavenumber: float, /) -> "ScalarKernelFamily3D":
        return cls("outgoing-helmholtz", parameter=wavenumber)


class ScalarCalderonAssemblyReport3D(StrictModule, NonTrainableState):
    """Discrete accuracy, precision, resources, and non-goals for V/K/K' DP0."""

    ambient_dimension: int = eqx.field(static=True)
    boundary_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    trial_space: str = eqx.field(static=True)
    test_space: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    pair_counts: tuple[int, int, int, int, int] = eqx.field(static=True)
    exception_count: int = eqx.field(static=True)
    quadrature_maximum_errors: Array
    quadrature_evaluations: Array
    dimensionless_panel_parameter: float = eqx.field(static=True)
    dimensionless_panel_parameter_limit: float = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    action_workspace_bytes_per_rhs: int = eqx.field(static=True)
    finite: Array
    accuracy_supported: Array
    materializable: bool = eqx.field(static=True)
    continuum_discretization_error_estimated: bool = eqx.field(static=True)
    hypersingular_supported: bool = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class _BlockedScalarWeakOperator3D(_AbstractCostedLinearOperator):
    pair_data: _SurfacePairData3D
    exception_values: Array
    source_normals: Array
    kernel: ScalarKernelFamily3D
    layer_kind: Literal["single", "double"] = eqx.field(static=True)
    target_block_size: int = eqx.field(static=True)
    source_block_size: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    action_workspace_bytes: int = eqx.field(static=True)
    diagonal: Array

    def __init__(
        self,
        pair_data: _SurfacePairData3D,
        exception_values: Array,
        source_normals: Array,
        kernel: ScalarKernelFamily3D,
        layer_kind: Literal["single", "double"],
        space: ArraySpace,
        /,
        *,
        target_block_size: int,
        source_block_size: int,
        operator_id: str,
    ):
        values = jnp.asarray(exception_values)
        if values.shape != pair_data.values.shape:
            raise ValueError("Exception values must match classified surface pairs.")
        normals = jnp.asarray(source_normals)
        if normals.shape != (space.size, 3):
            raise ValueError("Source normals must have shape (face_count, 3).")
        if layer_kind not in ("single", "double"):
            raise ValueError("Scalar layer kind must be 'single' or 'double'.")
        self.pair_data = pair_data
        self.exception_values = values
        self.source_normals = normals
        self.kernel = kernel
        self.layer_kind = layer_kind
        self.target_block_size = int(target_block_size)
        self.source_block_size = int(source_block_size)
        self.face_count = int(space.size)
        quadrature_count = int(pair_data.regular_points.shape[1])
        itemsize = np.dtype(values.dtype).itemsize
        self.action_workspace_bytes = int(
            itemsize
            * (
                self.target_block_size * quadrature_count * 4
                + self.source_block_size * quadrature_count * 7
                + self.target_block_size
                * self.source_block_size
                * quadrature_count
                * quadrature_count
                * 6
                + 3 * self.target_block_size * self.source_block_size
            )
        )
        diagonal = jnp.zeros((space.size,), dtype=values.dtype)
        self_mask = pair_data.targets == pair_data.sources
        self.diagonal = diagonal.at[pair_data.targets[self_mask]].set(values[self_mask])
        self.source = space
        self.target = DualSpace(space)
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
            diagonal_assembly=True,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def _kernel_values(self, differences: Array, normals: Array, /) -> Array:
        radius = jnp.linalg.norm(differences, axis=-1)
        if self.layer_kind == "single":
            if self.kernel.family == "laplace":
                factor = jnp.ones_like(radius)
            elif self.kernel.family == "modified-helmholtz":
                factor = jnp.exp(-self.kernel.parameter * radius)
            else:
                factor = jnp.exp(1j * self.kernel.parameter * radius)
            return factor / (4.0 * jnp.pi * radius)
        normal_difference = oe.contract(
            "tsqrc,sc->tsqr", differences, normals, backend="jax"
        )
        if self.kernel.family == "laplace":
            factor = jnp.ones_like(radius)
        elif self.kernel.family == "modified-helmholtz":
            scaled = self.kernel.parameter * radius
            factor = jnp.exp(-scaled) * (1.0 + scaled)
        else:
            scaled = self.kernel.parameter * radius
            factor = jnp.exp(1j * scaled) * (1.0 - 1j * scaled)
        return factor * normal_difference / (4.0 * jnp.pi * radius**3)

    def _regular_action(self, vector: Array, /, *, transpose: bool) -> Array:
        face_count = self.face_count
        target_block = self.target_block_size
        source_block = self.source_block_size
        points = self.pair_data.regular_points
        weights = self.pair_data.regular_weights
        keys = self.pair_data.exception_keys
        output = jnp.zeros((face_count,), dtype=vector.dtype)
        for target_start in range(0, face_count, target_block):
            target_ids = jnp.arange(target_start, target_start + target_block)
            target_valid = target_ids < face_count
            safe_targets = jnp.minimum(target_ids, face_count - 1)
            target_points = points[safe_targets]
            target_weights = weights[safe_targets]
            if not transpose:
                accumulator = jnp.zeros((target_block,), dtype=vector.dtype)
            for source_start in range(0, face_count, source_block):
                source_ids = jnp.arange(source_start, source_start + source_block)
                source_valid = source_ids < face_count
                safe_sources = jnp.minimum(source_ids, face_count - 1)
                source_points = points[safe_sources]
                source_weights = weights[safe_sources]
                source_normals = self.source_normals[safe_sources]
                pair_keys = target_ids[:, None] * face_count + source_ids[None, :]
                positions = jnp.searchsorted(keys, pair_keys)
                safe_positions = jnp.minimum(positions, keys.shape[0] - 1)
                is_exception = (positions < keys.shape[0]) & (
                    keys[safe_positions] == pair_keys
                )
                active = target_valid[:, None] & source_valid[None, :] & ~is_exception
                differences = (
                    target_points[:, None, :, None, :]
                    - source_points[None, :, None, :, :]
                )
                safe_differences = jnp.where(
                    active[:, :, None, None, None],
                    differences,
                    jnp.ones_like(differences),
                )
                kernel_values = self._kernel_values(safe_differences, source_normals)
                pair_matrix = oe.contract(
                    "tq,sr,tsqr->ts",
                    target_weights,
                    source_weights,
                    kernel_values,
                    backend="jax",
                )
                pair_matrix = jnp.where(active, pair_matrix, 0.0)
                if transpose:
                    contribution = oe.contract(
                        "ts,t->s",
                        pair_matrix,
                        vector[safe_targets] * target_valid,
                        backend="jax",
                    )
                    output = output.at[safe_sources].add(contribution * source_valid)
                else:
                    accumulator = accumulator + oe.contract(
                        "ts,s->t",
                        pair_matrix,
                        vector[safe_sources] * source_valid,
                        backend="jax",
                    )
            if not transpose:
                output = output.at[safe_targets].add(accumulator * target_valid)
        return output

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        output = self._regular_action(value, transpose=False)
        correction = self.exception_values * value[self.pair_data.sources]
        return self.target.validate(output.at[self.pair_data.targets].add(correction))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        output = self._regular_action(value, transpose=True)
        correction = self.exception_values * value[self.pair_data.targets]
        return self.source.validate(output.at[self.pair_data.sources].add(correction))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(value)))

    def _assemble_diagonal(self, /) -> Array:
        return self.diagonal

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Blocked scalar Galerkin operators cannot materialize."
        )

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        return self.action_workspace_bytes, "blocked-scalar-surface-galerkin-action"


class _WeakTransposeScalarOperator3D(_AbstractCostedLinearOperator):
    operator: _BlockedScalarWeakOperator3D

    def __init__(
        self,
        operator: _BlockedScalarWeakOperator3D,
        /,
        *,
        operator_id: str,
    ):
        self.operator = operator
        self.source = operator.source
        self.target = operator.target
        self.properties = OperatorProperties()
        self.capabilities = operator.capabilities
        self.batch_shape = ()
        self.operator_id = operator_id

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.target.validate(self.operator.transpose_mv(vector))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        return self.source.validate(self.operator.mv(vector))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(value)))

    def _assemble_diagonal(self, /) -> Array:
        return self.operator.diagonal

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Blocked scalar Galerkin operators cannot materialize."
        )

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        return self.operator._action_workspace_cost()


class _StrongScalarOperator3D(_AbstractCostedLinearOperator):
    weak: AbstractLinearOperator
    inverse_areas: Array
    diagonal: Array
    transposed_weak_action: bool = eqx.field(static=True)
    action_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        weak: AbstractLinearOperator,
        areas: Array,
        diagonal: Array,
        /,
        *,
        transposed_weak_action: bool,
        action_workspace_bytes: int,
        operator_id: str,
    ):
        self.weak = weak
        self.inverse_areas = jnp.reciprocal(jnp.asarray(areas))
        self.diagonal = jnp.asarray(diagonal)
        self.transposed_weak_action = bool(transposed_weak_action)
        self.action_workspace_bytes = int(action_workspace_bytes)
        self.source = weak.source
        self.target = weak.source
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
            diagonal_assembly=True,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        weak_value = (
            self.weak.transpose_mv(value)
            if self.transposed_weak_action
            else self.weak.mv(value)
        )
        return self.target.validate(weak_value * self.inverse_areas)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector) * self.inverse_areas
        weak_value = (
            self.weak.mv(value)
            if self.transposed_weak_action
            else self.weak.transpose_mv(value)
        )
        return self.source.validate(weak_value)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(value)))

    def _assemble_diagonal(self, /) -> Array:
        return self.diagonal

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Blocked scalar Galerkin operators cannot materialize."
        )

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        return (
            self.action_workspace_bytes,
            "strong-blocked-scalar-surface-galerkin-action",
        )


class ScalarCalderonDP0Galerkin3D(StrictModule, NonTrainableState):
    """Prepared closed-triangle DP0 V/K/K' calculus for one scalar PDE.

    Weak operators map discontinuous face constants to their coordinate duals;
    strong operators apply the diagonal DP0 mass inverse. K' is the exact
    algebraic weak transpose of K. W is deliberately absent because DP0 is
    not an H^{1/2}-conforming trial space for the hypersingular map.
    """

    single_layer_weak: AbstractLinearOperator
    double_layer_weak: AbstractLinearOperator
    adjoint_double_layer_weak: AbstractLinearOperator
    single_layer: AbstractLinearOperator
    double_layer: AbstractLinearOperator
    adjoint_double_layer: AbstractLinearOperator
    space: ArraySpace
    panelization: SurfacePanelization3D
    face_areas: Array
    face_component_ids: Array
    kernel: ScalarKernelFamily3D
    trace_convention: ScalarTraceConvention3D
    assembly_report: ScalarCalderonAssemblyReport3D
    _binding: _SurfaceFEMBinding3D

    @property
    def face_count(self) -> int:
        return self._binding.face_count

    @property
    def component_count(self) -> int:
        return self._binding.component_count

    def single_layer_potential(self, coefficients: ArrayLike, /):
        values = self.space.validate(coefficients)
        density = jnp.repeat(values, self.panelization.nodes_per_panel)
        if self.kernel.family == "laplace":
            return LaplaceLayerPotential3D(
                self.panelization, kind="single", density=density
            )
        if self.kernel.family == "outgoing-helmholtz":
            return HelmholtzLayerPotential3D(
                self.panelization,
                self.kernel.parameter,
                kind="single",
                density=density,
            )
        raise UnsupportedScalarBoundarySpaceError(
            "Modified Helmholtz off-surface reconstruction is not provided by "
            "the current layer-potential substrate."
        )

    def double_layer_potential(self, coefficients: ArrayLike, /):
        values = self.space.validate(coefficients)
        density = jnp.repeat(values, self.panelization.nodes_per_panel)
        if self.kernel.family == "laplace":
            return LaplaceLayerPotential3D(
                self.panelization, kind="double", density=density
            )
        if self.kernel.family == "outgoing-helmholtz":
            return HelmholtzLayerPotential3D(
                self.panelization,
                self.kernel.parameter,
                kind="double",
                density=density,
            )
        raise UnsupportedScalarBoundarySpaceError(
            "Modified Helmholtz off-surface reconstruction is not provided by "
            "the current layer-potential substrate."
        )

    def combined_field_potential(
        self,
        coefficients: ArrayLike,
        /,
        *,
        eta: float,
    ) -> HelmholtzCombinedField3D:
        if self.kernel.family != "outgoing-helmholtz":
            raise ValueError("Combined-field reconstruction requires outgoing Helmholtz.")
        values = self.space.validate(coefficients)
        density = jnp.repeat(values, self.panelization.nodes_per_panel)
        return HelmholtzCombinedField3D(
            self.panelization,
            self.kernel.parameter,
            density,
            eta=eta,
        )


def _triangle_normal(triangle: np.ndarray, /) -> np.ndarray:
    cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    return cross / np.linalg.norm(cross)


def _numpy_kernel(
    differences: np.ndarray,
    source_normal: np.ndarray,
    kernel: ScalarKernelFamily3D,
    layer_kind: Literal["single", "double"],
    /,
) -> np.ndarray:
    radius = np.linalg.norm(differences, axis=-1)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("Scalar Galerkin quadrature encountered a singular point.")
    if kernel.family == "laplace":
        factor = np.ones_like(radius)
    elif kernel.family == "modified-helmholtz":
        scaled = kernel.parameter * radius
        factor = np.exp(-scaled)
        if layer_kind == "double":
            factor = factor * (1.0 + scaled)
    else:
        scaled = kernel.parameter * radius
        factor = np.exp(1j * scaled)
        if layer_kind == "double":
            factor = factor * (1.0 - 1j * scaled)
    if layer_kind == "single":
        return factor / (4.0 * np.pi * radius)
    normal_difference = differences @ source_normal
    return factor * normal_difference / (4.0 * np.pi * radius**3)


def _regular_scalar_pair(
    test_triangle: np.ndarray,
    source_triangle: np.ndarray,
    kernel: ScalarKernelFamily3D,
    layer_kind: Literal["single", "double"],
    order: int,
    /,
):
    points, weights = _regular_rule(order)
    target_points = _map_triangle(test_triangle, points)
    source_points = _map_triangle(source_triangle, points)
    differences = target_points[:, None, :] - source_points[None, :, :]
    values = _numpy_kernel(
        differences,
        _triangle_normal(source_triangle),
        kernel,
        layer_kind,
    )
    return (
        np.sum(weights[:, None] * weights[None, :] * values)
        * _surface_jacobian(test_triangle)
        * _surface_jacobian(source_triangle)
    )


def _singular_scalar_pair(
    triangles: np.ndarray,
    faces: np.ndarray,
    target: int,
    source: int,
    adjacency: str,
    kernel: ScalarKernelFamily3D,
    layer_kind: Literal["single", "double"],
    order: int,
    /,
):
    if target == source and layer_kind == "double":
        return 0.0
    target_reference, source_reference, weights = _duffy_rule(order, adjacency)
    if adjacency == "shared-edge":
        shared = sorted(set(map(int, faces[target])) & set(map(int, faces[source])))
        target_local = tuple(
            int(np.flatnonzero(faces[target] == value)[0]) for value in shared
        )
        source_local = tuple(
            int(np.flatnonzero(faces[source] == value)[0]) for value in shared
        )
        target_reference = _remap_edge(target_reference, *target_local)
        source_reference = _remap_edge(source_reference, *source_local)
    elif adjacency == "shared-vertex":
        shared = tuple(set(map(int, faces[target])) & set(map(int, faces[source])))
        if len(shared) != 1:
            raise ValueError("Vertex-adjacent faces must share exactly one vertex.")
        target_local = int(np.flatnonzero(faces[target] == shared[0])[0])
        source_local = int(np.flatnonzero(faces[source] == shared[0])[0])
        target_reference = _remap_vertex(target_reference, target_local)
        source_reference = _remap_vertex(source_reference, source_local)
    target_triangle = triangles[target]
    source_triangle = triangles[source]
    differences = _map_triangle(target_triangle, target_reference) - _map_triangle(
        source_triangle, source_reference
    )
    values = _numpy_kernel(
        differences,
        _triangle_normal(source_triangle),
        kernel,
        layer_kind,
    )
    return (
        np.sum(weights * values)
        * _surface_jacobian(target_triangle)
        * _surface_jacobian(source_triangle)
    )


def _near_scalar_pair(
    test_triangle: np.ndarray,
    source_triangle: np.ndarray,
    kernel: ScalarKernelFamily3D,
    layer_kind: Literal["single", "double"],
    *,
    low_order: int,
    high_order: int,
    max_depth: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    depth: int = 0,
):
    low = _regular_scalar_pair(
        test_triangle, source_triangle, kernel, layer_kind, low_order
    )
    high = _regular_scalar_pair(
        test_triangle, source_triangle, kernel, layer_kind, high_order
    )
    error = abs(high - low)
    evaluations = low_order**4 + high_order**4
    threshold = absolute_tolerance + relative_tolerance * abs(high)
    if error <= threshold:
        return high, error, evaluations
    if depth >= max_depth:
        raise ValueError(
            f"{layer_kind} near-pair quadrature exhausted its subdivision capacity."
        )
    if _diameter(test_triangle) >= _diameter(source_triangle):
        children = tuple(
            (child, source_triangle) for child in _subdivide_triangle(test_triangle)
        )
    else:
        children = tuple(
            (test_triangle, child) for child in _subdivide_triangle(source_triangle)
        )
    values = tuple(
        _near_scalar_pair(
            left,
            right,
            kernel,
            layer_kind,
            low_order=low_order,
            high_order=high_order,
            max_depth=max_depth,
            absolute_tolerance=absolute_tolerance / 4.0,
            relative_tolerance=relative_tolerance,
            depth=depth + 1,
        )
        for left, right in children
    )
    return (
        sum(value[0] for value in values),
        sum(value[1] for value in values),
        evaluations + sum(value[2] for value in values),
    )


def _scalar_exception_values(
    vertices: np.ndarray,
    faces: np.ndarray,
    pair_data: _SurfacePairData3D,
    kernel: ScalarKernelFamily3D,
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    targets = np.asarray(pair_data.targets, dtype=np.int32)
    sources = np.asarray(pair_data.sources, dtype=np.int32)
    classes = np.asarray(pair_data.classes, dtype=np.int32)
    exception_indices = {
        int(target) * faces.shape[0] + int(source): index
        for index, (target, source) in enumerate(zip(targets, sources, strict=True))
    }
    dtype = complex if kernel.scalar_dtype == "complex" else float
    single_values = np.empty((targets.size,), dtype=dtype)
    double_values = np.empty((targets.size,), dtype=dtype)
    maximum_errors = np.zeros((2,), dtype=float)
    evaluations = np.zeros((2,), dtype=np.int64)
    high_regular = policy.regular_order + 2
    high_singular = policy.singular_order + 2
    high_near = policy.near_order + 2
    names = ("single", "double")
    for target in range(faces.shape[0]):
        for source in range(faces.shape[0]):
            key = target * faces.shape[0] + source
            exception_index = exception_indices.get(key)
            for layer_index, layer_kind in enumerate(names):
                if exception_index is None:
                    low = _regular_scalar_pair(
                        triangles[target],
                        triangles[source],
                        kernel,
                        layer_kind,
                        policy.regular_order,
                    )
                    high = _regular_scalar_pair(
                        triangles[target],
                        triangles[source],
                        kernel,
                        layer_kind,
                        high_regular,
                    )
                    error = abs(high - low)
                    threshold = (
                        policy.absolute_tolerance + policy.relative_tolerance * abs(high)
                    )
                    evaluations[layer_index] += policy.regular_order**4 + high_regular**4
                    if error > threshold:
                        raise ValueError(
                            f"{layer_kind} regular-pair quadrature exceeds its tolerance."
                        )
                else:
                    pair_class = int(classes[exception_index])
                    if pair_class < 3:
                        adjacency = (
                            "coincident",
                            "shared-edge",
                            "shared-vertex",
                        )[pair_class]
                        low = _singular_scalar_pair(
                            triangles,
                            faces,
                            target,
                            source,
                            adjacency,
                            kernel,
                            layer_kind,
                            policy.singular_order,
                        )
                        high = _singular_scalar_pair(
                            triangles,
                            faces,
                            target,
                            source,
                            adjacency,
                            kernel,
                            layer_kind,
                            high_singular,
                        )
                        error = abs(high - low)
                        threshold = (
                            policy.absolute_tolerance
                            + policy.relative_tolerance * abs(high)
                        )
                        evaluations[layer_index] += (
                            policy.singular_order**4 + high_singular**4
                        ) * (6, 5, 2)[pair_class]
                        if error > threshold:
                            raise ValueError(
                                f"{layer_kind} {adjacency} quadrature exceeds its tolerance."
                            )
                    else:
                        high, error, count = _near_scalar_pair(
                            triangles[target],
                            triangles[source],
                            kernel,
                            layer_kind,
                            low_order=policy.near_order,
                            high_order=high_near,
                            max_depth=policy.near_max_depth,
                            absolute_tolerance=policy.absolute_tolerance,
                            relative_tolerance=policy.relative_tolerance,
                        )
                        evaluations[layer_index] += count
                    if layer_kind == "single":
                        single_values[exception_index] = high
                    else:
                        double_values[exception_index] = high
                maximum_errors[layer_index] = max(
                    maximum_errors[layer_index], float(error)
                )
    return single_values, double_values, maximum_errors, evaluations


def prepare_scalar_calderon_dp0_3d(
    region: MeshRegion,
    /,
    *,
    kernel: ScalarKernelFamily3D | None = None,
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D | None = None,
    numeric_version: str = "0",
) -> ScalarCalderonDP0Galerkin3D:
    """Prepare bounded matrix-free DP0 V/K/K' maps on a closed triangle mesh."""
    family = ScalarKernelFamily3D.laplace() if kernel is None else kernel
    if not isinstance(family, ScalarKernelFamily3D):
        raise TypeError("kernel must be ScalarKernelFamily3D or None.")
    selected = LaplaceSingleLayerDP0GalerkinPolicy3D() if policy is None else policy
    if not isinstance(selected, LaplaceSingleLayerDP0GalerkinPolicy3D):
        raise TypeError("policy must be LaplaceSingleLayerDP0GalerkinPolicy3D or None.")
    if selected.dense_oracle is not None:
        raise LinearCapabilityError(
            "Scalar Calderon preparation does not permit dense materialization."
        )
    panel_order = max(
        selected.regular_order,
        selected.singular_order,
        selected.near_order,
    )
    binding = _SurfaceFEMBinding3D(
        region,
        quadrature_order=panel_order,
        numeric_version=numeric_version,
    )
    vertices = np.asarray(region.triangle_mesh.vertices, dtype=float)
    faces = np.asarray(region.triangle_mesh.faces, dtype=np.int32)
    triangles = vertices[faces]
    max_panel_diameter = max(_diameter(triangle) for triangle in triangles)
    dimensionless_parameter = family.parameter * max_panel_diameter
    dimensionless_limit = float(selected.regular_order)
    if dimensionless_parameter > dimensionless_limit:
        raise ValueError(
            "Kernel parameter exceeds the bounded panel-frequency envelope; "
            "increase regular_order or refine the surface."
        )
    pair_data = _prepare_surface_pairs_3d(
        region.triangle_mesh.vertices,
        region.triangle_mesh.faces,
        regular_order=selected.regular_order,
        singular_order=selected.singular_order,
        near_order=selected.near_order,
        near_ratio=selected.near_ratio,
        near_max_depth=selected.near_max_depth,
        absolute_tolerance=selected.absolute_tolerance,
        relative_tolerance=selected.relative_tolerance,
        max_exception_pairs=selected.max_exception_pairs,
        max_preparation_workspace_bytes=selected.max_preparation_workspace_bytes,
        max_resident_bytes=selected.max_resident_bytes,
    )
    single_host, double_host, maximum_errors, evaluations = _scalar_exception_values(
        vertices,
        faces,
        pair_data,
        family,
        selected,
    )
    scalar_dtype = (
        jnp.result_type(binding.face_areas.dtype, jnp.complex64)
        if family.scalar_dtype == "complex"
        else binding.face_areas.dtype
    )
    pair_data = eqx.tree_at(
        lambda data: (data.regular_points, data.regular_weights, data.maximum_errors),
        pair_data,
        (
            selected.precision.evaluation(pair_data.regular_points),
            selected.precision.accumulation(pair_data.regular_weights),
            selected.precision.decision(pair_data.maximum_errors),
        ),
    )
    single_values = selected.precision.accumulation(
        jnp.asarray(single_host, dtype=scalar_dtype)
    )
    double_values = selected.precision.accumulation(
        jnp.asarray(double_host, dtype=scalar_dtype)
    )
    normals_host = np.stack(tuple(_triangle_normal(triangle) for triangle in triangles))
    normals = selected.precision.evaluation(jnp.asarray(normals_host))
    if family.scalar_dtype == "complex":
        space = ArraySpace(
            (binding.face_count,),
            dtype=scalar_dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "complex-surface-dp0-space-3d-v1",
                    "binding": binding.binding_id,
                    "dtype": np.dtype(scalar_dtype).str,
                }
            ),
        )
    else:
        space = binding.discretization.field_spaces[0].vector_space
        if not isinstance(space, ArraySpace):
            raise TypeError("Surface DP0 FEM binding must provide an ArraySpace.")
    common_id = {
        "binding": binding.binding_id,
        "policy": selected.policy_id,
        "kernel": family.kernel_id,
    }
    single_weak = _BlockedScalarWeakOperator3D(
        pair_data,
        single_values,
        normals,
        family,
        "single",
        space,
        target_block_size=selected.target_block_size,
        source_block_size=selected.source_block_size,
        operator_id=canonical_fingerprint(
            {"kind": "scalar-single-layer-weak-dp0-3d-v1", **common_id}
        ),
    )
    double_weak_base = _BlockedScalarWeakOperator3D(
        pair_data,
        double_values,
        normals,
        family,
        "double",
        space,
        target_block_size=selected.target_block_size,
        source_block_size=selected.source_block_size,
        operator_id=canonical_fingerprint(
            {"kind": "scalar-double-layer-weak-dp0-3d-v1", **common_id}
        ),
    )
    adjoint_weak = _WeakTransposeScalarOperator3D(
        double_weak_base,
        operator_id=canonical_fingerprint(
            {"kind": "scalar-adjoint-double-layer-weak-dp0-3d-v1", **common_id}
        ),
    )
    vector_bytes = binding.face_count * np.dtype(scalar_dtype).itemsize
    single_diagonal = single_weak.diagonal / binding.face_areas
    double_diagonal = double_weak_base.diagonal / binding.face_areas
    single = _StrongScalarOperator3D(
        single_weak,
        binding.face_areas,
        single_diagonal,
        transposed_weak_action=False,
        action_workspace_bytes=single_weak.action_workspace_bytes + vector_bytes,
        operator_id=canonical_fingerprint(
            {"kind": "scalar-single-layer-strong-dp0-3d-v1", **common_id}
        ),
    )
    double = _StrongScalarOperator3D(
        double_weak_base,
        binding.face_areas,
        double_diagonal,
        transposed_weak_action=False,
        action_workspace_bytes=double_weak_base.action_workspace_bytes + vector_bytes,
        operator_id=canonical_fingerprint(
            {"kind": "scalar-double-layer-strong-dp0-3d-v1", **common_id}
        ),
    )
    adjoint_double = _StrongScalarOperator3D(
        double_weak_base,
        binding.face_areas,
        double_diagonal,
        transposed_weak_action=True,
        action_workspace_bytes=double_weak_base.action_workspace_bytes + vector_bytes,
        operator_id=canonical_fingerprint(
            {"kind": "scalar-adjoint-double-layer-strong-dp0-3d-v1", **common_id}
        ),
    )
    resident_bytes = int(
        pair_data.resident_bytes
        + np.asarray(single_values).nbytes
        + np.asarray(double_values).nbytes
        + np.asarray(normals).nbytes
    )
    if resident_bytes > selected.max_resident_bytes:
        raise ValueError("Scalar Calderon state exceeds max_resident_bytes.")
    finite = (
        jnp.all(jnp.isfinite(single_values))
        & jnp.all(jnp.isfinite(double_values))
        & jnp.all(jnp.isfinite(normals))
        & jnp.all(jnp.isfinite(binding.face_areas))
        & jnp.all(binding.face_areas > 0.0)
    )
    errors = selected.precision.decision(jnp.asarray(maximum_errors))
    counts = pair_data.counts
    non_goals = (
        "no-continuum-discretization-error-certificate",
        "no-hypersingular-W-on-DP0",
        "no-H^1/2-conformity-for-discontinuous-double-layer-density",
        "no-open-or-nonorientable-surfaces",
        "no-curved-or-higher-order-panels",
        "no-automatic-resonance-detection",
        "no-dense-materialization",
    )
    report_id = canonical_fingerprint(
        {
            "kind": "scalar-calderon-dp0-assembly-report-3d-v1",
            **common_id,
            "pair_counts": counts,
            "errors": array_tree_fingerprint(errors),
            "resident_bytes": resident_bytes,
        }
    )
    report = ScalarCalderonAssemblyReport3D(
        ambient_dimension=3,
        boundary_dimension=2,
        pde=family.pde,
        geometry=(
            "closed-oriented-watertight-piecewise-planar-triangle-mesh-with-"
            "strictly-separated-component-bounding-boxes"
        ),
        formulation="DP0-Galerkin-V-K-Kprime-with-diagonal-mass-strong-form",
        trial_space=(
            "discontinuous-face-constant-DP0-in-L2-subset-H^-1/2-with-K-and-"
            "Kprime-used-on-their-polyhedral-L2-bounded-route"
        ),
        test_space="discontinuous-face-constant-DP0-L2-dual-testing",
        provider="jax-blocked-actions-host-Duffy-and-adaptive-pair-classification",
        precision_policy_id=selected.precision.policy_id,
        kernel_id=family.kernel_id,
        binding_id=binding.binding_id,
        numeric_version=binding.numeric_version,
        face_count=binding.face_count,
        component_count=binding.component_count,
        pair_counts=counts,
        exception_count=int(pair_data.targets.shape[0]),
        quadrature_maximum_errors=errors,
        quadrature_evaluations=jnp.asarray(evaluations, dtype=jnp.int64),
        dimensionless_panel_parameter=dimensionless_parameter,
        dimensionless_panel_parameter_limit=dimensionless_limit,
        preparation_workspace_bytes=pair_data.preparation_workspace_bytes,
        resident_bytes=resident_bytes,
        action_workspace_bytes_per_rhs=max(
            single.action_workspace_bytes,
            double.action_workspace_bytes,
        ),
        finite=finite,
        accuracy_supported=finite,
        materializable=False,
        continuum_discretization_error_estimated=False,
        hypersingular_supported=False,
        non_goals=non_goals,
        report_id=report_id,
    )
    return ScalarCalderonDP0Galerkin3D(
        single_layer_weak=single_weak,
        double_layer_weak=double_weak_base,
        adjoint_double_layer_weak=adjoint_weak,
        single_layer=single,
        double_layer=double,
        adjoint_double_layer=adjoint_double,
        space=space,
        panelization=binding.panelization,
        face_areas=binding.face_areas,
        face_component_ids=binding.face_component_ids,
        kernel=family,
        trace_convention=SCALAR_TRACE_CONVENTION_3D,
        assembly_report=report,
        _binding=binding,
    )


def prepare_scalar_hypersingular_dp0_3d(
    region: MeshRegion,
    /,
    *,
    numeric_version: str = "0",
):
    """Reject W before geometry/FEM preparation: DP0 is not H1/2 conforming."""
    del region, numeric_version
    raise UnsupportedScalarBoundarySpaceError(
        "Hypersingular W requires an H^1/2-conforming trial space; closed-surface "
        "DP0 preparation is rejected before geometry or quadrature preparation."
    )


__all__ = [
    "ScalarCalderonAssemblyReport3D",
    "ScalarCalderonDP0Galerkin3D",
    "ScalarKernelFamily3D",
    "ScalarKernelName3D",
    "prepare_scalar_calderon_dp0_3d",
    "prepare_scalar_hypersingular_dp0_3d",
]
