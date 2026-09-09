#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la
from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....artifacts import DifferentiationContract
from ....discretization import (
    CellMesh,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    TetrahedralConnectivity,
)
from ....equations import (
    coefficient,
    CompiledFiniteElementProblem,
    FiniteElementExecutionPolicy,
    FiniteElementForm,
    TensorDiffusionAction,
)
from ....units import convert_value, derived_unit, METER, SIEMENS, UnitDefinition
from .._evidence import GeophysicalCapabilityEvidence, GeophysicalResourceEstimate
from ._survey import ElectricalSurvey


DC_CONDUCTIVITY_UNIT = derived_unit("S/m", ((SIEMENS, 1), (METER, -1)))


def _validate_connected_tetrahedra(mesh: CellMesh) -> None:
    if not isinstance(mesh, CellMesh):
        raise TypeError("FinitePatchDCPlan requires a CellMesh.")
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("DC FEM requires full three-dimensional Cartesian geometry.")
    if any(block.cell_kind != "tetrahedron" for block in mesh.blocks):
        raise ValueError("DC FEM supports affine P1 tetrahedra only.")
    # Face-connected domains exclude disconnected bodies and zero-area contacts.
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("DC tetrahedral blocks require tetrahedral connectivity.")
    cell_faces = np.asarray(connectivity.cell_faces)
    counts = np.asarray(connectivity.face_cell_counts)
    if np.any((counts < 1) | (counts > 2)):
        raise ValueError("DC FEM requires manifold tetrahedral face incidence.")
    parent = np.arange(cell_faces.shape[0])

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first_owner = {}
    for cell, faces in enumerate(cell_faces):
        for face in faces:
            face = int(face)
            if face in first_owner:
                parent[root(cell)] = root(first_owner[face])
            else:
                first_owner[face] = cell
    if len({root(cell) for cell in range(cell_faces.shape[0])}) != 1:
        raise ValueError(
            "Disconnected or only edge/vertex-connected DC domains are unsupported."
        )


def _conductivity_tensor(
    value: ArrayLike, cell_count: int, dtype, unit: UnitDefinition
) -> Array:
    raw = jnp.asarray(value)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError("DC conductivity must be real, not complex impedance.")
    values = jnp.asarray(
        convert_value(raw, source=unit, target=DC_CONDUCTIVITY_UNIT), dtype=dtype
    )
    if values.shape in ((), (cell_count,)):
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(values <= 0),
            "Scalar DC conductivity must be finite and strictly positive.",
        )
        return jnp.broadcast_to(values, (cell_count,))[:, None, None] * jnp.eye(
            3, dtype=dtype
        )
    if values.shape not in ((3, 3), (cell_count, 3, 3)):
        raise ValueError("Conductivity must be scalar, (cells,), (3,3), or (cells,3,3).")
    tensors = jnp.broadcast_to(values, (cell_count, 3, 3))
    scale = jnp.max(jnp.abs(tensors), axis=(-2, -1))
    normalized = tensors / jnp.where(scale > 0, scale, 1)[:, None, None]
    symmetric = jnp.all(
        jnp.abs(normalized - jnp.swapaxes(normalized, -1, -2))
        <= 64 * jnp.finfo(dtype).eps
    )
    normalized = 0.5 * normalized + 0.5 * jnp.swapaxes(normalized, -1, -2)
    # Sylvester's criterion on a scaled 3x3 matrix avoids eigensolves and overflow.
    a, b, c = normalized[:, 0, 0], normalized[:, 0, 1], normalized[:, 0, 2]
    d, e, f = normalized[:, 1, 1], normalized[:, 1, 2], normalized[:, 2, 2]
    determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    valid = (a > 0) & (a * d - b * b > 0) & (determinant > 0)
    tensors = eqx.error_if(
        tensors,
        jnp.any(~jnp.isfinite(tensors)) | ~symmetric | jnp.any(~valid),
        "Tensor DC conductivity must be finite, symmetric, and positive definite.",
    )
    # Remove only roundoff-level antisymmetry accepted above; the physical law is SPD.
    return 0.5 * tensors + 0.5 * jnp.swapaxes(tensors, -1, -2)


def _primal_operator(compiled: CompiledFiniteElementProblem) -> la.FunctionLinearOperator:
    raw = compiled.affine_operator()
    space = compiled.state_space
    return la.FunctionLinearOperator(
        lambda potential: space.inverse_riesz(raw.mv(potential)),
        source=space,
        target=space,
        properties=compiled.form.declared_properties,
        operator_id=canonical_fingerprint(
            {"kind": "dc-conductivity-operator", "compilation": compiled.compilation_id}
        ),
    )


class FinitePatchDCPlan(StrictModule, NonTrainableState):
    """Fixed-mesh 3D steady conduction with uniformly loaded boundary patches.

    Solves -div(sigma grad(phi)) = 0 with inward integrated patch currents,
    insulating remainder boundary, and the Euclidean minimum-norm nodal gauge.
    The numerical kernel uses metres, S/m, amperes, and volts. Positive cellwise
    scalar or SPD tensor conductivity is supported. This is NOT the complete
    electrode model: no contact impedance, equipotential metal, point sources,
    infinite exterior, geometric factors, or apparent-resistivity convention.
    Disconnected domains and edge/vertex-only contacts are rejected, not grounded.
    """

    mesh: CellMesh
    survey: ElectricalSurvey
    geospatial_contract: object = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        survey: ElectricalSurvey,
        /,
        *,
        batch_size: int = 8,
        length_unit: UnitDefinition = METER,
        geospatial_contract=None,
        relative_tolerance: float = 1e-9,
        absolute_tolerance: float = 1e-11,
        max_steps: int = 2000,
    ):
        _validate_connected_tetrahedra(mesh)
        if not isinstance(survey, ElectricalSurvey):
            raise TypeError("survey must be ElectricalSurvey.")
        if (
            int(batch_size) != batch_size
            or batch_size < 1
            or int(max_steps) != max_steps
            or max_steps < 1
        ):
            raise ValueError("batch_size and max_steps must be positive integers.")
        if (
            not np.isfinite(relative_tolerance)
            or not np.isfinite(absolute_tolerance)
            or relative_tolerance <= 0
            or absolute_tolerance < 0
        ):
            raise ValueError(
                "DC solve tolerances must be finite, relative positive, absolute nonnegative."
            )
        if geospatial_contract is not None:
            from ....interchange._geospatial import GeospatialContract

            if not isinstance(geospatial_contract, GeospatialContract):
                raise TypeError("geospatial_contract must be GeospatialContract or None.")
            spatial = geospatial_contract.require_cartesian(dimensions=3)
            if spatial.length_unit != length_unit:
                raise ValueError("Geospatial and mesh coordinate units disagree.")
        coordinates = convert_value(mesh.coordinates, source=length_unit, target=METER)
        if length_unit != METER:
            mesh = CellMesh(
                coordinates,
                mesh.blocks,
                vertex_global_ids=mesh.vertex_global_ids,
                entity_global_ids={
                    d: entities.entity_ids
                    for d, entities in enumerate(mesh.topology.entity_sets)
                },
            )
        connectivity = mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("DC tetrahedral blocks require tetrahedral connectivity.")
        exterior = np.asarray(connectivity.boundary_faces, dtype=bool)
        occupied = set()
        for patch in survey.patches:
            facets = np.asarray(patch.facet_indices)
            if np.any(facets >= exterior.size) or not np.all(exterior[facets]):
                raise ValueError(
                    f"Electrode {patch.name!r} contains absent or non-exterior facets."
                )
            if occupied.intersection(map(int, facets)):
                raise ValueError("Distinct electrode patches cannot overlap in area.")
            occupied.update(map(int, facets))
        self.mesh = mesh
        self.survey = survey
        self.geospatial_contract = geospatial_contract
        self.batch_size = min(int(batch_size), survey.source_count)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.max_steps = int(max_steps)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dc-fem-plan",
                "mesh": mesh.mesh_id,
                "survey": survey.survey_id,
                "batch_size": self.batch_size,
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
                "max_steps": self.max_steps,
            }
        )

    def prepare(self) -> PreparedDC:
        return PreparedDC(self)

    @property
    def capability_evidence(self) -> GeophysicalCapabilityEvidence:
        return GeophysicalCapabilityEvidence(
            "finite-patch-dc-resistivity",
            (3,),
            field_equations=("steady-conduction",),
            source_models=("uniform-boundary-current-density",),
            receiver_models=("area-average-potential-difference",),
            boundary_models=("insulating", "prescribed-current-patch"),
            material_models=("positive-scalar-conductivity", "spd-tensor-conductivity"),
            limitations=(
                "connected-domain",
                "affine-p1-tetrahedra",
                "no-point-electrodes",
                "no-contact-impedance",
                "finite-domain",
            ),
            differentiation=DifferentiationContract(
                upstream_physical_parameters=True,
                stored_values=True,
                query_coordinates=False,
                local_parameters=True,
                stochastic_realization=False,
                higher_order=True,
            ),
        )

    def resource_estimate(
        self, *, maximum_bytes: int | None = None
    ) -> GeophysicalResourceEstimate:
        coordinates = np.asarray(self.mesh.coordinates)
        retained = coordinates.nbytes
        retained += sum(np.asarray(block.vertices).nbytes for block in self.mesh.blocks)
        retained += np.asarray(self.survey.currents).nbytes
        retained += np.asarray(self.survey.receiver_weights).nbytes
        retained += np.asarray(self.survey.source_indices).nbytes
        cells = sum(block.cell_count for block in self.mesh.blocks)
        nodes = coordinates.shape[0]
        scalar_bytes = coordinates.dtype.itemsize
        retained += cells * 16 * scalar_bytes
        workspace = self.batch_size * (6 * nodes + cells * 16) * scalar_bytes
        observations = self.survey.measurement_count * scalar_bytes
        return GeophysicalResourceEstimate(
            retained_bytes=retained,
            workspace_bytes=workspace,
            checkpoint_bytes=0,
            observation_bytes=observations,
            source_batch_size=self.batch_size,
            maximum_bytes=maximum_bytes,
        )


class DCSolveResult(StrictModule):
    """Checked nodal fields and signed voltages; all source fields are retained."""

    potentials: Array
    electrode_potentials: Array
    voltages: Array
    residual_norms: Array
    gauge_residuals: Array
    compatibility_residuals: Array
    successful: Array


class PreparedDC(StrictModule, NonTrainableState):
    """Reusable P1 geometry, sparse patch rows, compiled form and solve template.

    Preparation is host-side. Binding and predictions are differentiable in
    conductivity on the fixed mesh. Geometry, patch supports and survey topology
    are not differentiation variables. Prediction workspace is bounded by
    ``batch_size * node_count``; it does not retain all source fields.
    """

    plan: FinitePatchDCPlan
    discretization: FiniteElementDiscretization
    compiled: CompiledFiniteElementProblem
    electrode_nodes: Array
    electrode_indices: Array
    electrode_weights: Array
    electrode_areas: Array
    nullspace_policy: la.NullspacePolicy
    template: la.LinearSolveTemplate
    check_policy: la.LinearSolveCheckPolicy

    def __init__(self, plan: FinitePatchDCPlan, /):
        if not isinstance(plan, FinitePatchDCPlan):
            raise TypeError("plan must be FinitePatchDCPlan.")
        discretization = FiniteElementPlan(
            plan.mesh,
            FiniteElementFieldSpec("potential", lagrange_element("tetrahedron", 1)),
        ).prepare()
        count = sum(block.cell_count for block in plan.mesh.blocks)
        properties = la.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        )
        diffusion = TensorDiffusionAction(
            "potential",
            coefficient(
                jnp.broadcast_to(jnp.eye(3), (count, 3, 3)),
                location="cell",
                support_id=discretization.support.support_id,
                entity_set_id=discretization.cell_domain.entity_set_id,
            ),
            properties=properties,
            action_id="dc-cell-conductivity",
        )
        compiled = CompiledFiniteElementProblem(
            FiniteElementForm(
                "dc-conduction", "potential", (diffusion,), properties=properties
            ),
            discretization,
            execution_policy=FiniteElementExecutionPolicy(realization="sparse"),
        )
        nodes, electrode_indices, weights, areas = [], [], [], []
        zero = compiled.state_space.zeros()
        connectivity = plan.mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("Prepared DC requires tetrahedral connectivity.")
        face_vertices = np.asarray(connectivity.faces, dtype=np.int32)
        coordinates = np.asarray(plan.mesh.coordinates)
        for electrode, patch in enumerate(plan.survey.patches):
            integrated_row = np.zeros(
                (compiled.state_space.size,), dtype=coordinates.dtype
            )
            for facet in np.asarray(patch.facet_indices):
                triangle = face_vertices[facet]
                points = coordinates[triangle]
                area = 0.5 * np.linalg.norm(
                    np.cross(points[1] - points[0], points[2] - points[0])
                )
                integrated_row[triangle] += area / 3.0
            area = float(np.sum(integrated_row))
            if not np.isfinite(area) or area <= 0 or np.any(~np.isfinite(integrated_row)):
                raise ValueError("Electrode patches must have positive finite area.")
            active = np.flatnonzero(integrated_row != 0)
            nodes.append(active)
            electrode_indices.append(np.full(active.size, electrode, dtype=np.int32))
            weights.append(integrated_row[active] / area)
            areas.append(area)
        operator = _primal_operator(compiled)
        modes = la.LinearSubspace(
            operator.source, jnp.ones((operator.source.size, 1), dtype=zero.dtype)
        )
        kernel_tolerance = max(
            plan.absolute_tolerance, 256 * np.finfo(zero.dtype).eps * operator.source.size
        )
        certificate = la.KernelCertificate(
            operator,
            modes,
            left=modes,
            complete=True,
            scope="structural",
            evidence="construction",
            tolerance=kernel_tolerance,
        )
        nullspace = la.NullspacePolicy(
            certificate=certificate, compatibility="error", gauge="minimum-norm"
        )
        policy = la.LinearSolvePolicy(
            la.ProjectedPCG(),
            tolerance=la.TolerancePolicy(
                relative=plan.relative_tolerance,
                absolute=plan.absolute_tolerance,
                max_steps=plan.max_steps,
            ),
            differentiation=la.DifferentiationPolicy("mathematical"),
            failure=la.FailurePolicy("error"),
        )
        self.plan = plan
        self.discretization = discretization
        self.compiled = compiled
        self.electrode_nodes = jnp.asarray(np.concatenate(nodes), dtype=jnp.int32)
        self.electrode_indices = jnp.asarray(
            np.concatenate(electrode_indices), dtype=jnp.int32
        )
        self.electrode_weights = jnp.asarray(np.concatenate(weights), dtype=zero.dtype)
        self.electrode_areas = jnp.asarray(areas, dtype=zero.dtype)
        self.nullspace_policy = nullspace
        self.template = la.prepare_template(
            la.LinearSystem(operator, nullspace_policy=nullspace), policy
        )
        self.check_policy = la.LinearSolveCheckPolicy(
            relative_tolerance=plan.relative_tolerance,
            absolute_tolerance=plan.absolute_tolerance,
            nullspace_tolerance=kernel_tolerance,
            require_nullspace=True,
        )

    @property
    def cell_count(self) -> int:
        return sum(block.cell_count for block in self.plan.mesh.blocks)

    def current_load(self, currents: ArrayLike, /) -> Array:
        """Assemble amperes into the weak Neumann load using normalized patch rows."""
        current = jnp.asarray(currents, dtype=self.electrode_weights.dtype)
        if current.shape != (len(self.plan.survey.patches),):
            raise ValueError("One current pattern must have one entry per electrode.")
        magnitude = jnp.sum(jnp.abs(current))
        current = eqx.error_if(
            current,
            jnp.any(~jnp.isfinite(current))
            | (jnp.abs(jnp.sum(current)) > 64 * jnp.finfo(current.dtype).eps * magnitude),
            "Current pattern must be finite and balanced.",
        )
        return (
            jnp.zeros((self.compiled.state_space.size,), dtype=current.dtype)
            .at[self.electrode_nodes]
            .add(self.electrode_weights * current[self.electrode_indices])
        )

    def electrode_potentials(self, potential: ArrayLike, /) -> Array:
        """Area-average the same supports used for current injection, in volts."""
        potential = self.compiled.state_space.validate(potential)
        return (
            jnp.zeros((len(self.plan.survey.patches),), dtype=potential.dtype)
            .at[self.electrode_indices]
            .add(self.electrode_weights * potential[self.electrode_nodes])
        )

    def voltages(self, electrode_potentials: ArrayLike, /) -> Array:
        values = jnp.asarray(electrode_potentials)
        survey = self.plan.survey
        if values.shape != (survey.source_count, len(survey.patches)):
            raise ValueError(
                "Electrode potentials must cover all sources and electrodes."
            )
        return ein.contract(
            "me,me->m", survey.receiver_weights, values[survey.source_indices]
        )

    def bind_conductivity(
        self, conductivity: ArrayLike, /, *, unit: UnitDefinition = DC_CONDUCTIVITY_UNIT
    ) -> PreparedDCConductivity:
        return PreparedDCConductivity(self, conductivity, unit=unit)

    def predict(
        self, conductivity: ArrayLike, /, *, unit: UnitDefinition = DC_CONDUCTIVITY_UNIT
    ) -> Array:
        return self.bind_conductivity(conductivity, unit=unit).predict()


class PreparedDCConductivity(StrictModule, NonTrainableState):
    """One conductivity operator reused for every bounded source batch."""

    dc: PreparedDC
    conductivity: Array
    linear_solve: la.PreparedLinearSolve

    def __init__(
        self,
        dc: PreparedDC,
        conductivity: ArrayLike,
        /,
        *,
        unit: UnitDefinition = DC_CONDUCTIVITY_UNIT,
    ):
        if not isinstance(dc, PreparedDC):
            raise TypeError("dc must be PreparedDC.")
        tensor = _conductivity_tensor(
            conductivity, dc.cell_count, dc.electrode_weights.dtype, unit
        )
        # Numerical rebinding preserves the native compiled field/coefficient layout.
        compiled = eqx.tree_at(
            lambda problem: problem.form.actions[0].diffusivity.value, dc.compiled, tensor
        )
        operator = _primal_operator(compiled)
        self.dc = dc
        self.conductivity = tensor
        self.linear_solve = la.bind_numeric(
            dc.template, la.LinearSystem(operator, nullspace_policy=dc.nullspace_policy)
        )

    def _solve_current(self, current):
        result, evidence = la.solve_checked(
            self.linear_solve,
            self.dc.current_load(current),
            check_policy=self.dc.check_policy,
        )
        potential = eqx.error_if(
            result.value,
            ~evidence.valid,
            "DC voltage predictions require a converged compatible gauge-fixed solve.",
        )
        return potential, evidence

    def predict(self) -> Array:
        """Return signed voltage channels without storing all source nodal fields.

        JAX JVP/VJP use phydrax.linalg's mathematical implicit solve derivative,
        not a dense Jacobian or differentiation through Krylov iteration history.
        """

        def one(current):
            potential, _ = self._solve_current(current)
            return self.dc.electrode_potentials(potential)

        electrodes = jax.lax.map(
            one, self.dc.plan.survey.currents, batch_size=self.dc.plan.batch_size
        )
        return self.dc.voltages(electrodes)

    def solve(self) -> DCSolveResult:
        """Explicitly retain all nodal fields, with independently checked evidence."""

        def one(current):
            potential, evidence = self._solve_current(current)
            return (
                potential,
                self.dc.electrode_potentials(potential),
                evidence.true_residual_norm,
                evidence.gauge_residual,
                evidence.compatibility_residual,
                evidence.valid,
            )

        potentials, electrodes, residuals, gauges, compatibility, valid = jax.lax.map(
            one, self.dc.plan.survey.currents, batch_size=self.dc.plan.batch_size
        )
        return DCSolveResult(
            potentials=potentials,
            electrode_potentials=electrodes,
            voltages=self.dc.voltages(electrodes),
            residual_norms=residuals,
            gauge_residuals=gauges,
            compatibility_residuals=compatibility,
            successful=jnp.all(valid),
        )


__all__ = [
    "DC_CONDUCTIVITY_UNIT",
    "FinitePatchDCPlan",
    "DCSolveResult",
    "PreparedDC",
    "PreparedDCConductivity",
]
