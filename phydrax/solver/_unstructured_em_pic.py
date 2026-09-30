#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tetrahedral compatible Maxwell with Whitney transfer as a PIC field solver."""

from __future__ import annotations

from typing import Any, assert_never, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._dtype_names import RealPrecisionDType
from .._fingerprint import canonical_fingerprint
from .._trainable import NonTrainableState
from ..discretization import tetrahedral_cell_complex, tetrahedral_connectivity
from ..discretization.fem._simplicial_whitney_chains import SimplicialWhitneyKernel
from ..discretization.pic import PICSpeciesPlan, UnstructuredWhitneyCurrentPlan
from ..linalg import (
    AbstractVectorSpace,
    ArraySpace,
    ConjugateGradient,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    solve,
    TolerancePolicy,
)
from ..typing import Bool, Dim, Int32
from ._maxwell import CompatibleMaxwellState, MaxwellPrimaryState
from ._maxwell_unstructured import PreparedUnstructuredMaxwell
from ._pic_field_solver import (
    AbstractPreparedPICFieldSolver,
    PICCapabilityRecord,
    PICFieldAdvance,
    PICFieldDeposit,
    PICFieldSolverCapability,
    PICGaussProjectionResult,
    PICRestartComponent,
    restart_component,
    restore_component,
)


class MaxwellPICEdgeDim(Dim):
    pass


class MaxwellPICVertexDim(Dim):
    pass


class MaxwellPICActiveVertexDim(Dim):
    pass


class MaxwellPICActiveEdgeDim(Dim):
    pass


@final
class UnstructuredMaxwellPICFieldSolver(
    AbstractPreparedPICFieldSolver, NonTrainableState
):
    """Whitney-0/1 trajectory current coupled to tetrahedral compatible Maxwell.

    The Whitney deposit returns nodal charge content and integrated edge flow
    on its own edge orientation. They enter Maxwell's Gauss charge density and
    electric current through the inverse degree-0/degree-1 Hodge stars on the
    Maxwell edge order, so the Maxwell charge rate ``δJ`` equals the deposited
    content rate. The explicit relative Maxwell realization constrains conductor
    traces; its restricted metric inverse carries charges on interior vertices.
    Its Poisson stiffness acts on compact relative scalar coordinates with the
    restricted electric Gram; only physical field states are zero extended.
    """

    __strict_contract__ = True

    maxwell: PreparedUnstructuredMaxwell
    current: UnstructuredWhitneyCurrentPlan
    kernel: SimplicialWhitneyKernel
    current_to_maxwell_edges: Int32[MaxwellPICEdgeDim]
    interior: Bool[MaxwellPICVertexDim]
    charge_indices: Int32[MaxwellPICActiveVertexDim]
    current_indices: Int32[MaxwellPICActiveEdgeDim]
    charge_space: AbstractVectorSpace
    current_space: AbstractVectorSpace
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
        if maxwell.plan.boundary != "relative":
            raise ValueError(
                "Conductor PIC requires an explicitly relative Maxwell plan."
            )
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
        current_edges = np.asarray(current.edges, dtype=np.int32)
        for edge_index in range(current_edges.shape[0]):
            first = int(current_edges[edge_index, 0])
            second = int(current_edges[edge_index, 1])
            canonical = (first, second) if first <= second else (second, first)
            if canonical not in edge_lookup:
                raise ValueError("Whitney edge is absent from the Maxwell cochain.")
            current_to_maxwell.append(edge_lookup[canonical])
            if (first, second) != canonical:
                raise ValueError("Whitney current edges must use canonical orientation.")
        kernel = SimplicialWhitneyKernel(cochain, current.locator)
        coordinates = np.asarray(current.locator.coordinates, dtype=np.float64)
        edge_lengths = np.linalg.norm(
            coordinates[maxwell_edges[:, 1]] - coordinates[maxwell_edges[:, 0]], axis=-1
        )
        interior = ~np.asarray(connectivity.boundary_vertices, dtype=np.bool_)
        if not np.any(interior):
            raise ValueError("Unstructured PIC requires at least one interior vertex.")
        material = maxwell.constitutive.initialize_state()
        relative = cochain.hilbert_complex(boundary="relative")
        charge_indices = cochain.active_indices(0, boundary="relative")
        current_indices = cochain.active_indices(1, boundary="relative")
        derivative = relative.differential(0)
        current_space = relative.space(1)

        def gauss_action(potential: Array) -> Array:
            gradient = (
                jnp.zeros((cochain.cell_counts[1],), dtype=potential.dtype)
                .at[current_indices]
                .set(derivative.mv(potential))
            )
            displacement = maxwell.constitutive.electric_displacement(gradient, material)
            return derivative.transpose_mv(
                current_space.riesz(displacement[current_indices])
            )

        space = ArraySpace((charge_indices.size,), dtype=jnp.float64)
        operator = FunctionLinearOperator(
            gauss_action,
            source=space,
            target=space,
            transpose_action=gauss_action,
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
        self.kernel = kernel
        self.current_to_maxwell_edges = jnp.asarray(current_to_maxwell, dtype=jnp.int32)
        self.interior = jnp.asarray(interior)
        self.charge_indices = charge_indices
        self.current_indices = current_indices
        self.charge_space = relative.space(0)
        self.current_space = relative.space(1)
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
                "electrostatic": electrostatic.plan.plan_id,
            }
        )

    @property
    def pic_configuration(self) -> str:
        return "unstructured-whitney"

    def pic_capability(
        self, capability: PICFieldSolverCapability, /
    ) -> PICCapabilityRecord:
        refusal = PICCapabilityRecord.refusal
        match capability:
            case "restart-state":
                return PICCapabilityRecord.route(
                    capability, "Tetrahedral field cochains, admitted by solver identity."
                )
            case "gauss-projection":
                return PICCapabilityRecord.route(
                    capability,
                    "Cochain Poisson projection on interior vertices (cochain-poisson).",
                )
            case "tensor-layout" | "window-shift" | "open-domain":
                return refusal(
                    capability,
                    "Tetrahedral cochains have no structured axes, mirror parities, "
                    "or integer-cell translations.",
                )
            case "spectral-symbol":
                return refusal(
                    capability, "An unstructured mesh has no single vacuum symbol."
                )
            case "huygens-sampling" | "energy-accounting":
                return refusal(
                    capability,
                    "Unstructured Maxwell publishes no Huygens observers or energy split.",
                )
            case "multi-deposit":
                return refusal(
                    capability,
                    "Each species deposits its own Whitney trajectory current.",
                )
            case "galilean-grid":
                return refusal(capability, "The tetrahedral mesh is lab-fixed.")
            case "relativistic-self-fields":
                return refusal(
                    capability,
                    "Boosted-Coulomb fields are a structured cochain capability.",
                )
            case _:
                assert_never(capability)

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

    def _potential_gradient(self, potential: Array, /) -> Array:
        cochain = self.maxwell.plan.cochain
        gradient = (
            cochain.hilbert_complex(boundary="relative").differential(0).mv(potential)
        )
        return (
            jnp.zeros((cochain.cell_counts[1],), dtype=gradient.dtype)
            .at[self.current_indices]
            .set(gradient)
        )

    def _density(self, content: Array, /) -> Array:
        values = self.charge_space.inverse_riesz(content[self.charge_indices])
        return jnp.zeros_like(content).at[self.charge_indices].set(values)

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
        density = jnp.where(self.interior, charge, 0.0)
        rhs = self.charge_space.riesz(density[self.charge_indices])
        result = solve(self.electrostatic, rhs, initial_guess=jnp.zeros_like(rhs))
        initial = self.maxwell.initialize()
        displacement = self.maxwell.constitutive.electric_displacement(
            -self._potential_gradient(result.value),
            initial.auxiliary.material,
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
        density = jnp.where(self.interior, charge, 0.0)
        target = CompatibleMaxwellState(
            MaxwellPrimaryState(
                field.primary.electric_displacement, field.primary.magnetic_flux, density
            ),
            field.auxiliary,
            field.observations,
        )
        residual, _ = self.maxwell.constraints(target)
        rhs = self.charge_space.riesz(-residual[self.charge_indices])
        result = solve(self.electrostatic, rhs, initial_guess=jnp.zeros_like(rhs))
        displacement = self.maxwell.constitutive.electric_displacement(
            -self._potential_gradient(result.value),
            field.auxiliary.material,
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
            cochain.hilbert_complex().space(1),
            cochain.hilbert_complex().space(2),
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
            .add(result.edge_current)
        )
        values = self.current_space.inverse_riesz(flow[self.current_indices])
        current = jnp.zeros_like(flow).at[self.current_indices].set(values)
        # The Whitney defect is in charge content, so its scale is too.
        magnitude = result.end_charge_magnitude
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
        electric_query = self.kernel.evaluate(position, 1, location=location)
        magnetic_query = self.kernel.evaluate(
            position, 2, proxy="flux", location=location
        )
        electric = electric_query.gather(self.maxwell.electric_field(field))
        magnetic = magnetic_query.gather(field.primary.magnetic_flux)
        support = electric_query.successful & magnetic_query.successful
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
