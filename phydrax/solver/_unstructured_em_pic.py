#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tetrahedral compatible Maxwell with Whitney transfer as a PIC field solver."""

from __future__ import annotations

import itertools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.ein import contract

from .._dtype_names import RealPrecisionDType
from .._fingerprint import canonical_fingerprint
from .._trainable import NonTrainableState
from ..discretization import tetrahedral_cell_complex, tetrahedral_connectivity
from ..discretization.pic import PICSpeciesPlan, UnstructuredWhitneyCurrentPlan
from ..linalg import (
    ConjugateGradient,
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    solve,
    TolerancePolicy,
)
from ._maxwell import CompatibleMaxwellState, MaxwellPrimaryState
from ._maxwell_unstructured import PreparedUnstructuredMaxwell
from ._pic_field_solver import (
    AbstractPreparedPICFieldSolver,
    PICFieldAdvance,
    PICFieldDeposit,
    PICGaussProjectionResult,
    PICRestartComponent,
    restart_component,
    restore_component,
)


class UnstructuredMaxwellPICFieldSolver(
    AbstractPreparedPICFieldSolver, NonTrainableState
):
    """Whitney-0/1 trajectory current coupled to tetrahedral compatible Maxwell.

    The Whitney deposit returns nodal charge content and integrated edge flow
    on its own edge orientation. They enter Maxwell's Gauss charge density and
    electric current through the inverse degree-0/degree-1 Hodge stars on the
    Maxwell edge order, so the Maxwell charge rate ``δJ`` equals the deposited
    content rate. Boundary vertices are absolutely constrained by Maxwell and
    their charge is carried by the conductor, so charges live on interior
    vertices.
    """

    maxwell: PreparedUnstructuredMaxwell
    current: UnstructuredWhitneyCurrentPlan
    gradients: Array
    face_reconstruction: Array
    cell_faces: Array
    cell_face_signs: Array
    current_to_maxwell_edges: Array
    current_to_maxwell_signs: Array
    interior: Array
    minimum_edge_length: float = eqx.field(static=True)
    electrostatic: PreparedLinearSolve
    solver_id: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    field_dtype: RealPrecisionDType = eqx.field(static=True)

    def __init__(
        self,
        maxwell: PreparedUnstructuredMaxwell,
        current: UnstructuredWhitneyCurrentPlan,
        /,
        *,
        electrostatic_tolerance: float = 1.0e-12,
        maximum_electrostatic_iterations: int = 1000,
    ) -> None:
        if not isinstance(maxwell, PreparedUnstructuredMaxwell):
            raise TypeError("maxwell must be PreparedUnstructuredMaxwell.")
        if not isinstance(current, UnstructuredWhitneyCurrentPlan):
            raise TypeError("current must be UnstructuredWhitneyCurrentPlan.")
        if current.locator.dimension != 3:
            raise ValueError("Unstructured electromagnetic PIC requires tetrahedra.")
        cells = np.asarray(current.locator.cells, dtype=np.int32)
        connectivity = tetrahedral_connectivity(cells, current.locator.coordinate_count)
        expected_topology = tetrahedral_cell_complex(
            cells, current.locator.coordinate_count
        )
        cochain = maxwell.plan.cochain
        if cochain.topology.topology_id != expected_topology.topology_id:
            raise ValueError(
                "Whitney current and Maxwell cochains must share exact topology."
            )
        maxwell_edges = np.asarray(connectivity.edges, dtype=np.int32)
        if cochain.cell_counts[1] != maxwell_edges.shape[0]:
            raise ValueError("Maxwell degree-one capacity does not match topology.")
        if cochain.cell_counts[2] != connectivity.faces.shape[0]:
            raise ValueError("Maxwell degree-two capacity does not match topology.")
        edge_lookup = {
            (int(maxwell_edges[index, 0]), int(maxwell_edges[index, 1])): index
            for index in range(maxwell_edges.shape[0])
        }
        current_to_maxwell = []
        current_to_maxwell_signs = []
        current_edges = np.asarray(current.edges, dtype=np.int32)
        for edge_index in range(current_edges.shape[0]):
            first = int(current_edges[edge_index, 0])
            second = int(current_edges[edge_index, 1])
            canonical = (first, second) if first <= second else (second, first)
            if canonical not in edge_lookup:
                raise ValueError("Whitney edge is absent from the Maxwell cochain.")
            current_to_maxwell.append(edge_lookup[canonical])
            current_to_maxwell_signs.append(1.0 if (first, second) == canonical else -1.0)
        coordinates = np.asarray(current.locator.coordinates, dtype=np.float64)
        gradients = []
        reconstruction = []
        local_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
        for cell in cells:
            vertices = coordinates[cell]
            jacobian = (vertices[1:] - vertices[0]).T
            inverse = np.linalg.solve(jacobian, np.eye(3, dtype=jacobian.dtype))
            gradients.append(
                np.concatenate((-np.sum(inverse, axis=0, keepdims=True), inverse), axis=0)
            )
            normal_matrix = np.asarray(
                [
                    0.5
                    * np.cross(
                        coordinates[cell[face[1]]] - coordinates[cell[face[0]]],
                        coordinates[cell[face[2]]] - coordinates[cell[face[0]]],
                    )
                    for face in local_faces
                ]
            )
            reconstruction.append(
                np.linalg.lstsq(
                    normal_matrix,
                    np.eye(normal_matrix.shape[0], dtype=normal_matrix.dtype),
                    rcond=1.0e-15,
                )[0]
            )
        edge_lengths = np.linalg.norm(
            coordinates[maxwell_edges[:, 1]] - coordinates[maxwell_edges[:, 0]], axis=-1
        )
        interior = np.asarray(cochain.active_mask(0, "absolute"), dtype=np.bool_)
        if not np.any(interior):
            raise ValueError("Unstructured PIC requires at least one interior vertex.")
        material = maxwell.constitutive.initialize_state()

        def gauss_content(potential: Array) -> Array:
            # Content form ⋆0(-δD) of D = ε(-dφ); symmetric positive semidefinite.
            displacement = maxwell.constitutive.electric_displacement(
                -cochain.exterior_derivative(0, potential), material
            )
            return cochain.apply_hodge(0, -cochain.codifferential(1, displacement))

        count = cochain.cell_counts[0]
        stiffness = np.asarray(
            jax.vmap(gauss_content)(jnp.eye(count, dtype=jnp.float64)), dtype=np.float64
        ).T
        stiffness = interior[:, None] * stiffness * interior[None, :] + np.diag(
            (~interior).astype(np.float64)
        )
        operator = DenseLinearOperator(
            jnp.asarray(stiffness),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "unstructured-maxwell-pic-gauss",
                    "maxwell": maxwell.prepared_id,
                }
            ),
        )
        electrostatic = prepare(
            LinearSystem(operator),
            LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=float(electrostatic_tolerance),
                    absolute=float(electrostatic_tolerance),
                    max_steps=int(maximum_electrostatic_iterations),
                ),
            ),
        )
        self.maxwell = maxwell
        self.current = current
        self.gradients = jnp.asarray(np.asarray(gradients))
        self.face_reconstruction = jnp.asarray(np.asarray(reconstruction))
        self.cell_faces = connectivity.cell_faces
        self.cell_face_signs = connectivity.cell_face_signs
        self.current_to_maxwell_edges = jnp.asarray(current_to_maxwell, dtype=jnp.int32)
        self.current_to_maxwell_signs = jnp.asarray(current_to_maxwell_signs)
        self.interior = jnp.asarray(interior)
        self.minimum_edge_length = float(np.min(edge_lengths))
        self.electrostatic = electrostatic
        self.spatial_dimension = 3
        self.field_dtype = "float64"
        self.solver_id = canonical_fingerprint(
            {
                "kind": "unstructured-maxwell-pic-field-solver",
                "maxwell": maxwell.prepared_id,
                "current": current.plan_id,
                "topology": expected_topology.topology_id,
                "edge_order": tuple(current_to_maxwell),
                "edge_signs": tuple(current_to_maxwell_signs),
                "electrostatic": electrostatic.plan.plan_id,
            }
        )

    @property
    def stable_step(self) -> Array:
        return self.maxwell.stable_dt

    @property
    def displacement_widths(self) -> Array:
        return jnp.full((3,), self.minimum_edge_length)

    def validate_species(self, species: tuple[PICSpeciesPlan, ...], /) -> None:
        if any(value.population.particles.ambient_dimension != 3 for value in species):
            raise ValueError("Unstructured PIC species must be three-dimensional.")

    def pairing_probe(self, species: int, capacity: int, /) -> tuple[Array, Array]:
        del species
        cells = np.asarray(self.current.locator.cells)
        coordinates = np.asarray(self.current.locator.coordinates, dtype=np.float64)
        chosen = cells[np.arange(capacity) % cells.shape[0]]
        vertices = coordinates[chosen]
        centroid = np.mean(vertices, axis=1)
        # Move a quarter of the way toward the first vertex: stays in the cell.
        end = centroid + 0.25 * (vertices[:, 0] - centroid)
        return jnp.asarray(centroid), jnp.asarray(end)

    def _density(self, content: Array, /) -> Array:
        cochain = self.maxwell.plan.cochain
        return jnp.where(self.interior, cochain.solve_hodge(0, content), 0.0)

    def field_with_charge(self, charge: Array, /) -> CompatibleMaxwellState:
        initial = self.maxwell.initialize()
        return CompatibleMaxwellState(
            MaxwellPrimaryState(
                jnp.zeros_like(initial.primary.electric_displacement, dtype=charge.dtype),
                jnp.zeros_like(initial.primary.magnetic_flux, dtype=charge.dtype),
                jnp.where(self.interior, charge, 0.0),
            ),
            initial.auxiliary,
            initial.observations,
        )

    def initialize_field(
        self, charge: Array, /, *, magnetic: Any = None
    ) -> tuple[CompatibleMaxwellState, Array]:
        cochain = self.maxwell.plan.cochain
        density = jnp.where(self.interior, charge, 0.0)
        rhs = jnp.where(self.interior, cochain.apply_hodge(0, density), 0.0)
        result = solve(self.electrostatic, rhs, initial_guess=jnp.zeros_like(rhs))
        potential = jnp.where(self.interior, result.value, 0.0)
        initial = self.maxwell.initialize()
        displacement = self.maxwell.constitutive.electric_displacement(
            -cochain.exterior_derivative(0, potential), initial.auxiliary.material
        )
        flux = (
            jnp.zeros_like(initial.primary.magnetic_flux, dtype=density.dtype)
            if magnetic is None
            else jnp.asarray(magnetic, dtype=density.dtype)
        )
        field = CompatibleMaxwellState(
            MaxwellPrimaryState(displacement, flux, density),
            initial.auxiliary,
            initial.observations,
        )
        return field, result.successful

    def project_gauss(
        self, field: CompatibleMaxwellState, charge: Array, /
    ) -> PICGaussProjectionResult:
        """Cochain Poisson projection onto the interior-vertex density ``charge``.

        The interior content residual ``⋆0(ρ + δD)`` is solved with the prepared
        Gauss stiffness; ``D ← D + ε(-dφ)`` with ``φ = 0`` on the conductor, so
        ``B``, material memory, and observers are unchanged.
        """
        cochain = self.maxwell.plan.cochain
        density = jnp.where(self.interior, charge, 0.0)
        target = CompatibleMaxwellState(
            MaxwellPrimaryState(
                field.primary.electric_displacement, field.primary.magnetic_flux, density
            ),
            field.auxiliary,
            field.observations,
        )
        residual, _ = self.maxwell.constraints(target)
        rhs = jnp.where(self.interior, cochain.apply_hodge(0, -residual), 0.0)
        result = solve(self.electrostatic, rhs, initial_guess=jnp.zeros_like(rhs))
        potential = jnp.where(self.interior, result.value, 0.0)
        displacement = self.maxwell.constitutive.electric_displacement(
            -cochain.exterior_derivative(0, potential), field.auxiliary.material
        )
        projected = CompatibleMaxwellState(
            MaxwellPrimaryState(
                field.primary.electric_displacement + displacement,
                field.primary.magnetic_flux,
                density,
            ),
            field.auxiliary,
            field.observations,
        )
        after, _ = self.maxwell.constraints(projected)
        return PICGaussProjectionResult(
            projected,
            jnp.max(jnp.abs(residual), initial=0.0),
            jnp.max(jnp.abs(after), initial=0.0),
            self.field_energy(projected) - self.field_energy(field),
            result.successful & jnp.all(jnp.isfinite(after)),
            "cochain-poisson",
        )

    def field_charge(self, field: CompatibleMaxwellState, /) -> Array:
        return field.primary.charge

    def field_energy(self, field: CompatibleMaxwellState, /) -> Array:
        cochain = self.maxwell.plan.cochain
        return self.maxwell.constitutive.energy(
            field.primary.electric_displacement,
            field.primary.magnetic_flux,
            field.auxiliary.material,
            cochain.hodge_metric(1),
            cochain.hodge_metric(2),
        )

    def _nodal_content(
        self, position: Array, macrocharge: Array, active: Array, /
    ) -> tuple[Array, Array]:
        locator = self.current.locator
        location = locator.locate(position)
        valid = active & location.inside
        safe = jnp.maximum(location.cell_ids, 0)
        content = jnp.zeros((locator.coordinate_count,), dtype=position.dtype)
        for local in range(locator.cells.shape[1]):
            content = content.at[locator.cells[safe, local]].add(
                jnp.where(valid, macrocharge * location.barycentric[:, local], 0.0)
            )
        return content, jnp.all(location.successful | ~active)

    def deposit_charge(
        self,
        species: int,
        position: Array,
        macrocharge: Array,
        active: Array,
        /,
    ) -> tuple[Array, Array]:
        del species
        content, successful = self._nodal_content(position, macrocharge, active)
        return self._density(content), successful

    def deposit(
        self,
        species: int,
        start: Array,
        end: Array,
        velocity: Array,
        macrocharge: Array,
        active: Array,
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        del species, velocity
        result = self.current.deposit(start, end, macrocharge, active, step_size)
        flow = (
            jnp.zeros((self.maxwell.plan.cochain.cell_counts[1],), dtype=start.dtype)
            .at[self.current_to_maxwell_edges]
            .add(self.current_to_maxwell_signs * result.edge_current)
        )
        # The Whitney flow is the negative integrated flux ⋆1 J on Maxwell edges.
        current = -self.maxwell.plan.cochain.solve_hodge(1, flow)
        # The Whitney defect is in charge content, so its scale is too.
        magnitude, _ = self._nodal_content(end, jnp.abs(macrocharge), active)
        return PICFieldDeposit(
            current,
            self._density(result.start_charge),
            self._density(result.end_charge),
            result.maximum_continuity_defect,
            result.successful,
            jnp.max(
                jnp.abs(result.end_charge - result.start_charge) + 2.0 * magnitude,
                initial=0.0,
            )
            / step_size,
        )

    def advance(
        self,
        time: Array,
        field: CompatibleMaxwellState,
        current: Array,
        step_size: Array,
        /,
    ) -> PICFieldAdvance:
        advanced = self.maxwell.step(time, field, step_size, electric_current=current)
        electric, magnetic = self.maxwell.constraints(advanced)
        finite = jnp.all(jnp.isfinite(advanced.primary.electric_displacement)) & jnp.all(
            jnp.isfinite(advanced.primary.magnetic_flux)
        )
        return PICFieldAdvance(
            advanced,
            advanced.primary.charge,
            jnp.max(jnp.abs(electric), initial=0.0),
            jnp.max(jnp.abs(magnetic), initial=0.0),
            self.field_energy(advanced),
            None,
            finite & (step_size <= self.maxwell.stable_dt),
        )

    def gather_fields(
        self,
        species: int,
        position: Array,
        active: Array,
        field: CompatibleMaxwellState,
        /,
    ) -> tuple[Array, Array, Array]:
        del species
        location = self.current.locator.locate(position)
        cell = jnp.maximum(location.cell_ids, 0)
        electric_cochain = self.maxwell.electric_field(field)
        local_edges = self.current.cell_edges[cell]
        local_signs = self.current.cell_edge_signs[cell]
        electric = jnp.zeros((position.shape[0], 3), dtype=position.dtype)
        for local_index, (a, b) in enumerate(itertools.combinations(range(4), 2)):
            whitney = (
                location.barycentric[:, a, None] * self.gradients[cell, b]
                - location.barycentric[:, b, None] * self.gradients[cell, a]
            )
            coefficient = (
                electric_cochain[
                    self.current_to_maxwell_edges[local_edges[:, local_index]]
                ]
                * self.current_to_maxwell_signs[local_edges[:, local_index]]
                * local_signs[:, local_index]
            )
            electric = electric + coefficient[:, None] * whitney
        magnetic_flux = (
            field.primary.magnetic_flux[self.cell_faces[cell]]
            * self.cell_face_signs[cell]
        )
        magnetic = contract("pij,pj->pi", self.face_reconstruction[cell], magnetic_flux)
        support = location.successful
        return (
            jnp.where(active[:, None], electric, 0.0),
            jnp.where(active[:, None], magnetic, 0.0),
            support,
        )

    def restart_component(self, field: CompatibleMaxwellState, /) -> PICRestartComponent:
        return restart_component("field", self.solver_id, field)

    def restore_component(
        self, component: PICRestartComponent, /
    ) -> CompatibleMaxwellState:
        template = self.field_with_charge(
            jnp.zeros((self.maxwell.plan.cochain.cell_counts[0],), dtype=jnp.float64)
        )
        return restore_component(component, "field", self.solver_id, template)


__all__ = ["UnstructuredMaxwellPICFieldSolver"]
