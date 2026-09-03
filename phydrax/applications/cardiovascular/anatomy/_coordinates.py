#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import (
    CellMesh,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from ....linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    solve,
)
from ._roles import CardiacBoundaryRoles


def _nonempty(value: str, description: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{description} must be non-empty.")
    return result


def _mesh_components(mesh: CellMesh, /) -> tuple[np.ndarray, ...]:
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks), axis=0
    )
    neighbours = [set() for _ in range(mesh.coordinates.shape[0])]
    for cell in cells:
        for first in cell:
            neighbours[int(first)].update(
                int(second) for second in cell if second != first
            )
    unseen = set(range(mesh.coordinates.shape[0]))
    components: list[np.ndarray] = []
    while unseen:
        pending = [unseen.pop()]
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            attached = unseen.intersection(neighbours[current])
            unseen.difference_update(attached)
            pending.extend(attached)
        components.append(np.asarray(sorted(component), dtype=np.int32))
    return tuple(components)


class HarmonicCoordinateSpec(StrictModule, NonTrainableState):
    """One scalar harmonic coordinate with two semantic Dirichlet roles."""

    name: str = eqx.field(static=True)
    lower_role: str = eqx.field(static=True)
    upper_role: str = eqx.field(static=True)
    lower_value: float = eqx.field(static=True)
    upper_value: float = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        lower_role: str,
        upper_role: str,
        /,
        *,
        lower_value: float = 0.0,
        upper_value: float = 1.0,
    ):
        name_ = _nonempty(name, "Harmonic coordinate name")
        lower_role_ = _nonempty(lower_role, "Lower boundary role")
        upper_role_ = _nonempty(upper_role, "Upper boundary role")
        lower = float(lower_value)
        upper = float(upper_value)
        if lower_role_ == upper_role_:
            raise ValueError("Harmonic coordinate boundary roles must be distinct.")
        if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
            raise ValueError(
                "Harmonic coordinate endpoint values must be finite and distinct."
            )
        self.name = name_
        self.lower_role = lower_role_
        self.upper_role = upper_role_
        self.lower_value = lower
        self.upper_value = upper
        self.spec_id = canonical_fingerprint(
            {
                "kind": "cardiac-harmonic-coordinate-spec",
                "name": name_,
                "lower_role": lower_role_,
                "upper_role": upper_role_,
                "lower_value": lower,
                "upper_value": upper,
            }
        )


class HarmonicCoordinateEvidence(StrictModule, NonTrainableState):
    """Per-coordinate linear-solve, boundary, and maximum-principle evidence."""

    solver_status: Array
    solver_residual_norm: Array
    free_residual_norm: Array
    maximum_boundary_error: Array
    maximum_principle_violation: Array
    finite: Array
    successful: Array

    def __init__(
        self,
        solver_status: ArrayLike,
        solver_residual_norm: ArrayLike,
        free_residual_norm: ArrayLike,
        maximum_boundary_error: ArrayLike,
        maximum_principle_violation: ArrayLike,
        finite: ArrayLike,
        successful: ArrayLike,
        /,
    ):
        arrays = (
            np.shape(solver_status),
            np.shape(solver_residual_norm),
            np.shape(free_residual_norm),
            np.shape(maximum_boundary_error),
            np.shape(maximum_principle_violation),
            np.shape(finite),
            np.shape(successful),
        )
        if len(set(arrays)) != 1 or len(arrays[0]) != 1:
            raise ValueError(
                "Harmonic coordinate evidence must use one shared vector shape."
            )
        self.solver_status = jnp.asarray(solver_status, dtype=jnp.int32)
        self.solver_residual_norm = jnp.asarray(solver_residual_norm)
        self.free_residual_norm = jnp.asarray(free_residual_norm)
        self.maximum_boundary_error = jnp.asarray(maximum_boundary_error)
        self.maximum_principle_violation = jnp.asarray(maximum_principle_violation)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.successful = jnp.asarray(successful, dtype=bool)

    @property
    def all_successful(self) -> Array:
        return jnp.all(self.successful)


class HarmonicCoordinateFields(StrictModule):
    """Committed nodal and cellwise affine harmonic coordinate fields."""

    names: tuple[str, ...] = eqx.field(static=True)
    nodal_values: Array
    cell_values: Array
    cell_gradients: Array
    dirichlet_masks: Array
    evidence: HarmonicCoordinateEvidence
    fields_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: Sequence[str],
        nodal_values: ArrayLike,
        cell_values: ArrayLike,
        cell_gradients: ArrayLike,
        dirichlet_masks: ArrayLike,
        evidence: HarmonicCoordinateEvidence,
        /,
        *,
        fields_id: str,
    ):
        names_ = tuple(str(name) for name in names)
        nodal = jnp.asarray(nodal_values)
        cells = jnp.asarray(cell_values)
        gradients = jnp.asarray(cell_gradients)
        masks = jnp.asarray(dirichlet_masks, dtype=bool)
        if not names_ or len(set(names_)) != len(names_):
            raise ValueError("Committed coordinate names must be unique and non-empty.")
        count = len(names_)
        if nodal.ndim != 2 or nodal.shape[0] != count:
            raise ValueError(
                "nodal_values must have shape (coordinate_count, node_count)."
            )
        if cells.ndim != 2 or cells.shape[0] != count:
            raise ValueError(
                "cell_values must have shape (coordinate_count, cell_count)."
            )
        if gradients.shape != cells.shape + (3,):
            raise ValueError(
                "cell_gradients must have shape (coordinate_count, cell_count, 3)."
            )
        if masks.shape != nodal.shape:
            raise ValueError("dirichlet_masks must match nodal_values.")
        if not isinstance(evidence, HarmonicCoordinateEvidence):
            raise TypeError("evidence must be HarmonicCoordinateEvidence.")
        identifier = _nonempty(fields_id, "fields_id")
        self.names = names_
        self.nodal_values = nodal
        self.cell_values = cells
        self.cell_gradients = gradients
        self.dirichlet_masks = masks
        self.evidence = evidence
        self.fields_id = identifier

    def coordinate_index(self, name: str, /) -> int:
        name_ = str(name)
        if name_ not in self.names:
            raise KeyError(f"Unknown harmonic coordinate {name_!r}.")
        return self.names.index(name_)

    def nodal(self, name: str, /) -> Array:
        return self.nodal_values[self.coordinate_index(name)]

    def cell(self, name: str, /) -> Array:
        return self.cell_values[self.coordinate_index(name)]

    def gradient(self, name: str, /) -> Array:
        return self.cell_gradients[self.coordinate_index(name)]


class HarmonicCoordinateCandidate(StrictModule):
    """Uncommitted fixed-shape harmonic fields with fail-closed evidence."""

    names: tuple[str, ...] = eqx.field(static=True)
    nodal_values: Array
    cell_values: Array
    cell_gradients: Array
    dirichlet_masks: Array
    evidence: HarmonicCoordinateEvidence
    candidate_id: str = eqx.field(static=True)

    def commit(self, /) -> HarmonicCoordinateFields:
        """Commit only if every coordinate passed all numerical checks."""
        checked = eqx.error_if(
            self.nodal_values,
            ~self.evidence.all_successful,
            "Cannot commit unsuccessful harmonic cardiac coordinates.",
        )
        return HarmonicCoordinateFields(
            self.names,
            checked,
            self.cell_values,
            self.cell_gradients,
            self.dirichlet_masks,
            self.evidence,
            fields_id=canonical_fingerprint(
                {
                    "kind": "committed-cardiac-harmonic-fields",
                    "candidate": self.candidate_id,
                }
            ),
        )


class HarmonicCoordinatePlan(StrictModule, NonTrainableState):
    """Static plan for multiple affine-P1 cardiac harmonic coordinates."""

    mesh: CellMesh
    roles: CardiacBoundaryRoles
    specs: tuple[HarmonicCoordinateSpec, ...]
    coordinate_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        roles: CardiacBoundaryRoles,
        specs: HarmonicCoordinateSpec | Sequence[HarmonicCoordinateSpec],
        /,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if not isinstance(roles, CardiacBoundaryRoles):
            raise TypeError("roles must be CardiacBoundaryRoles.")
        if roles.mesh.mesh_id != mesh.mesh_id:
            raise ValueError("Boundary roles and harmonic plan must use the same mesh.")
        normalized = (
            (specs,) if isinstance(specs, HarmonicCoordinateSpec) else tuple(specs)
        )
        if not normalized or not all(
            isinstance(spec, HarmonicCoordinateSpec) for spec in normalized
        ):
            raise TypeError("specs must contain HarmonicCoordinateSpec values.")
        names = tuple(spec.name for spec in normalized)
        if len(set(names)) != len(names):
            raise ValueError("Harmonic coordinate names must be unique.")
        components = _mesh_components(mesh)
        for spec in normalized:
            lower = np.asarray(roles.vertex_mask(spec.lower_role), dtype=bool)
            upper = np.asarray(roles.vertex_mask(spec.upper_role), dtype=bool)
            if np.any(lower & upper):
                raise ValueError(
                    f"Coordinate {spec.name!r} Dirichlet role closures must be disjoint."
                )
            for component in components:
                if not np.any(lower[component]) or not np.any(upper[component]):
                    raise ValueError(
                        f"Coordinate {spec.name!r} must span both endpoints on every "
                        "mesh component."
                    )
        self.mesh = mesh
        self.roles = roles
        self.specs = normalized
        self.coordinate_names = names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiac-harmonic-coordinate-plan",
                "mesh": mesh.mesh_id,
                "roles": roles.roles_id,
                "specs": [spec.spec_id for spec in normalized],
            }
        )

    def prepare(self, /, *, numeric_version: str = "0") -> PreparedHarmonicCoordinates:
        return PreparedHarmonicCoordinates(self, numeric_version=numeric_version)


class PreparedHarmonicCoordinates(StrictModule, NonTrainableState):
    """Prepared FEM stiffness and fixed Dirichlet routes for cardiac coordinates."""

    plan: HarmonicCoordinatePlan
    finite_element: FiniteElementDiscretization
    stiffness: Array
    dirichlet_masks: Array
    dirichlet_values: Array
    boundary_indices: tuple[Array, ...]
    free_indices: tuple[Array, ...]
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    def __init__(self, plan: HarmonicCoordinatePlan, /, *, numeric_version: str = "0"):
        if not isinstance(plan, HarmonicCoordinatePlan):
            raise TypeError("plan must be a HarmonicCoordinatePlan.")
        version = _nonempty(numeric_version, "numeric_version")
        field = FiniteElementFieldSpec(
            "cardiac-harmonic-coordinate", lagrange_element("tetrahedron", 1)
        )
        finite_element = FiniteElementPlan(plan.mesh, field).prepare(
            numeric_version=version
        )
        stiffness = finite_element.stiffness.as_dense()
        masks: list[np.ndarray] = []
        values: list[np.ndarray] = []
        boundaries: list[np.ndarray] = []
        free_routes: list[np.ndarray] = []
        node_count = plan.mesh.coordinates.shape[0]
        for spec in plan.specs:
            lower = np.asarray(plan.roles.vertex_mask(spec.lower_role), dtype=bool)
            upper = np.asarray(plan.roles.vertex_mask(spec.upper_role), dtype=bool)
            mask = lower | upper
            prescribed = np.zeros((node_count,), dtype=float)
            prescribed[lower] = spec.lower_value
            prescribed[upper] = spec.upper_value
            masks.append(mask)
            values.append(prescribed)
            boundaries.append(np.flatnonzero(mask).astype(np.int32))
            free_routes.append(np.flatnonzero(~mask).astype(np.int32))
        self.plan = plan
        self.finite_element = finite_element
        self.stiffness = stiffness
        self.dirichlet_masks = jnp.asarray(np.stack(masks))
        self.dirichlet_values = jnp.asarray(
            np.stack(values), dtype=finite_element.vertices.dtype
        )
        self.boundary_indices = tuple(jnp.asarray(route) for route in boundaries)
        self.free_indices = tuple(jnp.asarray(route) for route in free_routes)
        self.numeric_version = version
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiac-harmonic-coordinates",
                "plan": plan.plan_id,
                "finite_element": finite_element.prepared_id,
                "numeric_version": version,
            }
        )

    def solve(self, /) -> HarmonicCoordinateCandidate:
        """Solve all coordinates and return an evidence-bearing candidate."""
        values: list[Array] = []
        statuses: list[Array] = []
        solver_residuals: list[Array] = []
        free_residuals: list[Array] = []
        boundary_errors: list[Array] = []
        principle_violations: list[Array] = []
        finite_flags: list[Array] = []
        success_flags: list[Array] = []
        node_count = self.plan.mesh.coordinates.shape[0]
        for coordinate_index, (spec, boundary, free) in enumerate(
            zip(
                self.plan.specs,
                self.boundary_indices,
                self.free_indices,
                strict=True,
            )
        ):
            prescribed = self.dirichlet_values[coordinate_index]
            if free.size:
                free_matrix = self.stiffness[jnp.ix_(free, free)]
                coupling = self.stiffness[jnp.ix_(free, boundary)]
                right_hand_side = -(coupling @ prescribed[boundary])
                result = solve(
                    LinearSystem(
                        DenseLinearOperator(free_matrix),
                        problem_id=f"{self.prepared_id}:{spec.name}",
                    ),
                    right_hand_side,
                    policy=LinearSolvePolicy(DenseLU()),
                )
                coordinate = prescribed.at[free].set(result.value)
                status = result.status
                solver_residual = result.diagnostics.residual_norm
                solver_success = result.successful
            else:
                coordinate = prescribed
                status = jnp.asarray(int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)
                solver_residual = jnp.asarray(0.0, dtype=coordinate.dtype)
                solver_success = jnp.asarray(True)
            residual = self.stiffness @ coordinate
            free_residual = (
                jnp.sqrt(jnp.sum(residual[free] ** 2))
                if free.size
                else jnp.asarray(0.0, dtype=coordinate.dtype)
            )
            boundary_error = jnp.max(jnp.abs(coordinate[boundary] - prescribed[boundary]))
            low = min(spec.lower_value, spec.upper_value)
            high = max(spec.lower_value, spec.upper_value)
            principle_violation = jnp.maximum(
                jnp.maximum(low - jnp.min(coordinate), jnp.max(coordinate) - high),
                0.0,
            )
            finite = jnp.all(jnp.isfinite(coordinate)) & jnp.isfinite(free_residual)
            tolerance = (
                128.0
                * jnp.finfo(coordinate.dtype).eps
                * jnp.maximum(
                    1.0, jnp.sqrt(jnp.asarray(node_count, dtype=coordinate.dtype))
                )
            )
            successful = (
                solver_success
                & finite
                & (free_residual <= tolerance)
                & (boundary_error <= tolerance)
                & (principle_violation <= tolerance)
            )
            values.append(coordinate)
            statuses.append(status)
            solver_residuals.append(solver_residual)
            free_residuals.append(free_residual)
            boundary_errors.append(boundary_error)
            principle_violations.append(principle_violation)
            finite_flags.append(finite)
            success_flags.append(successful)

        nodal_values = jnp.stack(values)
        cell_values, cell_gradients = _expand_affine_fields(
            self.finite_element, nodal_values
        )
        evidence = HarmonicCoordinateEvidence(
            jnp.stack(statuses),
            jnp.stack(solver_residuals),
            jnp.stack(free_residuals),
            jnp.stack(boundary_errors),
            jnp.stack(principle_violations),
            jnp.stack(finite_flags),
            jnp.stack(success_flags),
        )
        return HarmonicCoordinateCandidate(
            self.plan.coordinate_names,
            nodal_values,
            cell_values,
            cell_gradients,
            self.dirichlet_masks,
            evidence,
            candidate_id=canonical_fingerprint(
                {
                    "kind": "cardiac-harmonic-coordinate-candidate",
                    "prepared": self.prepared_id,
                }
            ),
        )


def _expand_affine_fields(finite_element, nodal_values: Array, /) -> tuple[Array, Array]:
    cell_values: list[Array] = []
    cell_gradients: list[Array] = []
    for block, geometry in zip(
        finite_element.mesh.blocks,
        finite_element.block_geometries[0],
        strict=True,
    ):
        cells = block.vertices
        local = nodal_values[:, cells]
        cell_values.append(jnp.mean(local, axis=-1))
        basis_gradients = geometry.physical_gradients[:, 0, :, :]
        cell_gradients.append(oe.contract("qci,cid->qcd", local, basis_gradients))
    return (
        jnp.concatenate(tuple(cell_values), axis=1),
        jnp.concatenate(tuple(cell_gradients), axis=1),
    )


def _validate_coordinate_recipe(
    role_names: Sequence[str],
    coordinate_names: Sequence[str],
    roles: CardiacBoundaryRoles | None,
    /,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    semantic_roles = tuple(
        _nonempty(name, "Coordinate recipe role") for name in role_names
    )
    field_names = tuple(
        _nonempty(name, "Coordinate recipe field") for name in coordinate_names
    )
    if len(set(semantic_roles)) != len(semantic_roles):
        raise ValueError("Coordinate recipe roles must have distinct names.")
    if len(set(field_names)) != len(field_names):
        raise ValueError("Coordinate recipe field names must be distinct.")
    if roles is not None:
        if not isinstance(roles, CardiacBoundaryRoles):
            raise TypeError("roles must be CardiacBoundaryRoles or None.")
        missing = sorted(set(semantic_roles).difference(roles.role_names))
        if missing:
            raise ValueError(
                f"Coordinate recipe roles are absent from the profile: {missing}."
            )
    return semantic_roles, field_names


def left_ventricular_coordinate_specs(
    *,
    endocardium: str = "lv-endocardium",
    epicardium: str = "epicardium",
    apex: str = "apex",
    base: str = "base",
    transmural_name: str = "lv-transmural",
    apicobasal_name: str = "lv-apicobasal",
    roles: CardiacBoundaryRoles | None = None,
) -> tuple[HarmonicCoordinateSpec, HarmonicCoordinateSpec]:
    """Return an LV transmural/apicobasal recipe for explicit cap roles."""

    (endo, epi, apex_, base_), (transmural, apicobasal) = _validate_coordinate_recipe(
        (endocardium, epicardium, apex, base),
        (transmural_name, apicobasal_name),
        roles,
    )
    return (
        HarmonicCoordinateSpec(transmural, endo, epi),
        HarmonicCoordinateSpec(apicobasal, apex_, base_),
    )


def biventricular_coordinate_specs(
    *,
    lv_endocardium: str = "lv-endocardium",
    rv_endocardium: str = "rv-endocardium",
    epicardium: str = "epicardium",
    apex: str = "apex",
    base: str = "base",
    lv_transmural_name: str = "lv-transmural",
    rv_transmural_name: str = "rv-transmural",
    apicobasal_name: str = "biventricular-apicobasal",
    ventricular_separation_name: str = "lv-rv-separation",
    roles: CardiacBoundaryRoles | None = None,
) -> tuple[
    HarmonicCoordinateSpec,
    HarmonicCoordinateSpec,
    HarmonicCoordinateSpec,
    HarmonicCoordinateSpec,
]:
    """Return separate LV/RV transmural, apicobasal, and cavity-separation fields."""

    (lv, rv, epi, apex_, base_), names = _validate_coordinate_recipe(
        (lv_endocardium, rv_endocardium, epicardium, apex, base),
        (
            lv_transmural_name,
            rv_transmural_name,
            apicobasal_name,
            ventricular_separation_name,
        ),
        roles,
    )
    lv_transmural, rv_transmural, apicobasal, separation = names
    return (
        HarmonicCoordinateSpec(lv_transmural, lv, epi),
        HarmonicCoordinateSpec(rv_transmural, rv, epi),
        HarmonicCoordinateSpec(apicobasal, apex_, base_),
        HarmonicCoordinateSpec(separation, lv, rv),
    )


def atrial_coordinate_specs(
    *,
    left_endocardium: str = "la-endocardium",
    right_endocardium: str = "ra-endocardium",
    epicardium: str = "atrial-epicardium",
    left_transmural_name: str = "la-transmural",
    right_transmural_name: str = "ra-transmural",
    roles: CardiacBoundaryRoles | None = None,
) -> tuple[HarmonicCoordinateSpec, HarmonicCoordinateSpec]:
    """Return chamber-specific atrial transmural fields.

    Atrial longitudinal coordinates require case-specific landmark patches and
    are intentionally not guessed by this foundation recipe.
    """

    (left, right, epi), (left_name, right_name) = _validate_coordinate_recipe(
        (left_endocardium, right_endocardium, epicardium),
        (left_transmural_name, right_transmural_name),
        roles,
    )
    return (
        HarmonicCoordinateSpec(left_name, left, epi),
        HarmonicCoordinateSpec(right_name, right, epi),
    )


def prepare_harmonic_coordinates(
    plan: HarmonicCoordinatePlan, /, *, numeric_version: str = "0"
) -> PreparedHarmonicCoordinates:
    """Prepare an affine-P1 harmonic coordinate plan."""
    return plan.prepare(numeric_version=numeric_version)


def solve_harmonic_coordinates(
    prepared: PreparedHarmonicCoordinates, /
) -> HarmonicCoordinateCandidate:
    """Solve a prepared coordinate plan without silently committing failures."""
    if not isinstance(prepared, PreparedHarmonicCoordinates):
        raise TypeError("prepared must be PreparedHarmonicCoordinates.")
    return prepared.solve()


__all__ = [
    "HarmonicCoordinateCandidate",
    "HarmonicCoordinateEvidence",
    "HarmonicCoordinateFields",
    "HarmonicCoordinatePlan",
    "HarmonicCoordinateSpec",
    "PreparedHarmonicCoordinates",
    "atrial_coordinate_specs",
    "biventricular_coordinate_specs",
    "left_ventricular_coordinate_specs",
    "prepare_harmonic_coordinates",
    "solve_harmonic_coordinates",
]
