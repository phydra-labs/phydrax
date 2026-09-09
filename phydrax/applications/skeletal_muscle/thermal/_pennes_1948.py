#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-bound scalar Pennes field; no physiological property defaults.

Numerical identity: continuous affine tetrahedral P1 Galerkin space, consistent
capacity, backward Euler, native matrix-free PCG over prepared sparse actions.
All prescribed coefficients/boundaries are constant in time; the retained source
is interval-constant. Tensor transport and temperature feedback are not inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._identity import NumericRevision, SemanticProvenance
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import (
    CellMesh,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    IntegrationDomain,
    lagrange_element,
)
from ....discretization.fem import dirichlet_constraint
from ....equations import (
    BoundaryLoadAction,
    coefficient,
    compile_finite_element_problem,
    ExteriorFacetAction,
    FiniteElementExecutionPolicy,
    FiniteElementForm,
    MassAction,
    SourceAction,
    TensorDiffusionAction,
)
from ....linalg import (
    AbstractLinearOperator,
    ArraySpace,
    FailurePolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    PCG,
    solve,
    TolerancePolicy,
)
from ....sparse import EdgeRelation, SparseLinearMap
from ..energetics import RETAINED_HEAT_CATEGORIES, RetainedHeatLedger


PENNES_1948_DOI = "10.1152/jappl.1948.1.2.93"
_EVIDENCE_ROLES = (
    "equations",
    "geometry",
    "regions",
    "properties",
    "perfusion",
    "initial",
    "boundary",
    "source-projection",
    "retention",
    "validation",
)


class Pennes1948EvidenceBundle(StrictModule, NonTrainableState):
    """Caller-owned raw-asset attestations, not built-in human-case evidence.

    Each record is (role, source URL/URI, raw SHA256, reuse rights, units,
    calibration/held-out/manufactured/equation purpose). Every role is required;
    one raw manifest may attest several roles. No source values are inferred.
    """

    records: tuple[tuple[str, str, str, str, str, str], ...] = eqx.field(static=True)
    coordinate_frame: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        records: Sequence[tuple[str, str, str, str, str, str]],
        /,
        *,
        coordinate_frame: str,
        length_unit: str,
        perfusion_unit: str,
    ):
        rows = tuple(tuple(row) for row in records)
        if any(
            len(row) != 6 or any(not isinstance(x, str) or not x.strip() for x in row)
            for row in rows
        ):
            raise ValueError(
                "Thermal asset records need role, URI, hash, rights, units, purpose."
            )
        rows = tuple(sorted(rows))
        if len(rows) != len(_EVIDENCE_ROLES) or {r[0] for r in rows} != set(
            _EVIDENCE_ROLES
        ):
            raise ValueError(
                "Thermal evidence must cover every required case-asset role once."
            )
        for row in rows:
            if len(row[2]) != 64 or any(x not in "0123456789abcdef" for x in row[2]):
                raise ValueError(
                    "Raw thermal asset SHA256 must be lowercase hexadecimal."
                )
        if not coordinate_frame.strip() or length_unit != "m" or perfusion_unit != "1/s":
            raise ValueError(
                "Thermal geometry requires an explicit frame in m and volumetric perfusion in 1/s."
            )
        self.records = rows
        self.coordinate_frame = coordinate_frame
        self.evidence_id = SemanticProvenance(
            {
                "model": PENNES_1948_DOI,
                "records": rows,
                "coordinate_frame": coordinate_frame,
                "length_unit": length_unit,
                "perfusion_unit": perfusion_unit,
            }
        ).semantic_id


class Pennes1948Parameters(StrictModule):
    """Regionwise trainable scalar SI coefficients; blood perfusion is s⁻¹."""

    conductivity_W_per_m_K: Array
    capacity_J_per_m3_K: Array
    perfusion_per_s: Array
    blood_capacity_J_per_m3_K: Array
    arterial_temperature_K: Array

    def __init__(
        self,
        conductivity_W_per_m_K: ArrayLike,
        capacity_J_per_m3_K: ArrayLike,
        perfusion_per_s: ArrayLike,
        blood_capacity_J_per_m3_K: ArrayLike,
        arterial_temperature_K: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(x, dtype=float)
            for x in (
                conductivity_W_per_m_K,
                capacity_J_per_m3_K,
                perfusion_per_s,
                blood_capacity_J_per_m3_K,
                arterial_temperature_K,
            )
        )
        if (
            values[0].ndim != 1
            or not values[0].size
            or any(x.shape != values[0].shape for x in values)
        ):
            raise ValueError(
                "Pennes parameters must be equally sized scalar region vectors; tensors are not L1 inputs."
            )
        (
            self.conductivity_W_per_m_K,
            self.capacity_J_per_m3_K,
            self.perfusion_per_s,
            self.blood_capacity_J_per_m3_K,
            self.arterial_temperature_K,
        ) = values

    def admissible(self) -> Array:
        return (
            jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (
                            self.conductivity_W_per_m_K,
                            self.capacity_J_per_m3_K,
                            self.perfusion_per_s,
                            self.blood_capacity_J_per_m3_K,
                            self.arterial_temperature_K,
                        )
                    )
                )
            )
            & jnp.all(self.conductivity_W_per_m_K > 0)
            & jnp.all(self.capacity_J_per_m3_K > 0)
            & jnp.all(self.perfusion_per_s >= 0)
            & jnp.all(self.blood_capacity_J_per_m3_K > 0)
            & jnp.all(self.arterial_temperature_K > 0)
        )


class RetainedHeatProjection(StrictModule, NonTrainableState):
    """Sparse conservative accepted-source W → cell W/m³, reference volume."""

    operator: SparseLinearMap
    source_ids: tuple[str, ...] = eqx.field(static=True)
    source_model_id: str = eqx.field(static=True)
    retention_evidence_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_ids: tuple[str, ...],
        source_indices: ArrayLike,
        cell_indices: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        cell_count: int,
        source_model_id: str,
        retention_evidence_id: str,
        asset_id: str,
    ):
        ids = tuple(source_ids)
        source = np.asarray(source_indices)
        target = np.asarray(cell_indices)
        weight = np.asarray(weights, dtype=float)
        if not ids or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
            raise ValueError("Projection source IDs must be nonempty and unique.")
        if any(not x.strip() for x in (source_model_id, retention_evidence_id, asset_id)):
            raise ValueError(
                "Projection requires model, retention and mapping asset IDs."
            )
        if (
            source.ndim != 1
            or source.size == 0
            or source.shape != target.shape
            or source.shape != weight.shape
            or not np.issubdtype(source.dtype, np.integer)
            or not np.issubdtype(target.dtype, np.integer)
            or cell_count < 1
            or np.any(source < 0)
            or np.any(source >= len(ids))
            or np.any(target < 0)
            or np.any(target >= cell_count)
            or np.any(~np.isfinite(weight))
            or np.any(weight < 0)
        ):
            raise ValueError(
                "Heat projection needs finite nonnegative sparse weights and valid routes."
            )
        coverage = np.bincount(source, weights=weight, minlength=len(ids))
        if not np.allclose(coverage, 1, rtol=0, atol=1e-12):
            raise ValueError(
                "Every retained heat source must have conservative weight sum one."
            )
        self.operator = SparseLinearMap(
            EdgeRelation(
                source,
                target,
                source_size=len(ids),
                target_size=cell_count,
            ),
            weight,
        )
        self.source_ids = ids
        self.source_model_id = source_model_id
        self.retention_evidence_id = retention_evidence_id
        self.projection_id = canonical_fingerprint(
            {
                "source_ids": ids,
                "source_model": source_model_id,
                "retention": retention_evidence_id,
                "asset": asset_id,
                "source": source.tolist(),
                "cell": target.tolist(),
                "weight": weight.tolist(),
                "cell_count": cell_count,
                "volume": "fixed-reference-m3",
                "temporal": "interval-constant",
            }
        )

    def project(self, ledger: RetainedHeatLedger, cell_volume_m3: Array, /) -> Array:
        if (
            ledger.source_ids != self.source_ids
            or ledger.source_model_id != self.source_model_id
        ):
            raise ValueError("Retained heat ledger belongs to another source owner.")
        if ledger.evidence_id != self.retention_evidence_id:
            raise ValueError("Retained heat ledger has foreign retention evidence.")
        return self.operator.mv(ledger.retained_power_W) / cell_volume_m3


class Pennes1948Boundary(StrictModule):
    """One exterior patch; positive flux is outward, as thermal.HeatFlux.

    For convection, outward flux is h(T − ambient), as thermal.Convection.
    ``value`` is W/m² for flux, K for convection/Dirichlet, zero for insulation.
    """

    value: Array
    heat_transfer_W_per_m2_K: Array
    facet_ids: tuple[int, ...] = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)

    def __init__(
        self,
        facet_ids: Sequence[int],
        kind: str,
        value: ArrayLike,
        /,
        *,
        heat_transfer_W_per_m2_K: ArrayLike,
        asset_id: str,
    ):
        if not np.issubdtype(np.asarray(facet_ids).dtype, np.integer):
            raise ValueError(
                "Boundary facet IDs must be integers, not rounded coordinates."
            )
        ids = tuple(int(x) for x in facet_ids)
        if not ids or len(set(ids)) != len(ids) or any(x < 0 for x in ids):
            raise ValueError("Boundary facets must be nonempty unique IDs.")
        if (
            kind not in ("insulated", "flux", "convection", "dirichlet")
            or not asset_id.strip()
        ):
            raise ValueError(
                "A sourced insulated/flux/convection/Dirichlet patch is required."
            )
        value_ = jnp.asarray(value, dtype=float)
        transfer = jnp.asarray(heat_transfer_W_per_m2_K, dtype=float)
        if value_.shape != () or transfer.shape != ():
            raise ValueError("Each boundary patch has scalar constant values.")
        self.value = value_
        self.heat_transfer_W_per_m2_K = transfer
        self.facet_ids = ids
        self.kind = kind
        self.asset_id = asset_id

    def admissible(self) -> Array:
        valid = jnp.isfinite(self.value) & jnp.isfinite(self.heat_transfer_W_per_m2_K)
        valid &= self.heat_transfer_W_per_m2_K >= 0
        if self.kind != "convection":
            valid &= self.heat_transfer_W_per_m2_K == 0
        if self.kind in ("convection", "dirichlet"):
            valid &= self.value > 0
        if self.kind == "insulated":
            valid &= self.value == 0
        return valid


def _compiled(discretization, action):
    return compile_finite_element_problem(
        FiniteElementForm(action.action_id, "temperature", (action,)),
        discretization,
        execution_policy=FiniteElementExecutionPolicy(realization="sparse"),
    )


def _facet_mass(values, points, weights, normal, context):
    del points, weights, normal, context
    return values[0]


def _subdomain(base, ids):
    positions = np.asarray([list(np.asarray(base.entity_indices)).index(x) for x in ids])
    return IntegrationDomain(
        base.kind,
        np.asarray(base.entity_indices)[positions],
        base.support_id,
        base.entity_set_id,
        owner_cells=np.asarray(base.owner_cells)[positions],
        neighbour_cells=np.asarray(base.neighbour_cells)[positions],
        owner_local_entities=np.asarray(base.owner_local_entities)[positions],
        neighbour_local_entities=np.asarray(base.neighbour_local_entities)[positions],
    )


class Pennes1948Plan(StrictModule):
    """Caller-evidenced heterogeneous scalar Pennes, not a human tissue preset."""

    mesh: CellMesh
    parameters: Pennes1948Parameters
    projection: RetainedHeatProjection
    evidence: Pennes1948EvidenceBundle
    boundaries: tuple[Pennes1948Boundary, ...]
    initial_temperature_K: Array
    region_indices: tuple[int, ...] = eqx.field(static=True)
    region_ids: tuple[str, ...] = eqx.field(static=True)
    linear_relative_tolerance: float = eqx.field(static=True)
    linear_absolute_tolerance: float = eqx.field(static=True)
    balance_relative_tolerance: float = eqx.field(static=True)
    balance_absolute_tolerance_J: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        parameters: Pennes1948Parameters,
        region_ids: tuple[str, ...],
        region_indices: Sequence[int],
        projection: RetainedHeatProjection,
        boundaries: Sequence[Pennes1948Boundary],
        initial_temperature_K: ArrayLike,
        evidence: Pennes1948EvidenceBundle,
        /,
        *,
        linear_relative_tolerance: float,
        linear_absolute_tolerance: float,
        balance_relative_tolerance: float,
        balance_absolute_tolerance_J: float,
        maximum_iterations: int,
    ):
        if not isinstance(mesh, CellMesh) or mesh.ambient_dimension != 3:
            raise ValueError(
                "Thermal volume mesh must be a three-dimensional CellMesh in m."
            )
        if any(block.cell_kind != "tetrahedron" for block in mesh.blocks):
            raise ValueError(
                "This numerical identity requires affine tetrahedral P1 cells."
            )
        if not np.issubdtype(np.asarray(region_indices).dtype, np.integer):
            raise ValueError("Cell region indices must be integers.")
        regions = tuple(int(x) for x in region_indices)
        ids = tuple(region_ids)
        count = sum(block.cell_count for block in mesh.blocks)
        if (
            len(regions) != count
            or set(regions) != set(range(len(ids)))
            or len(set(ids)) != len(ids)
            or any(not x.strip() for x in ids)
            or parameters.conductivity_W_per_m_K.shape != (len(ids),)
        ):
            raise ValueError(
                "Every cell and every named material region must be covered exactly once."
            )
        if projection.operator.target.size != count:
            raise ValueError("Retained projection targets another cell count.")
        if not isinstance(evidence, Pennes1948EvidenceBundle):
            raise TypeError("A complete caller evidence bundle is required.")
        tolerances = tuple(
            float(x)
            for x in (
                linear_relative_tolerance,
                linear_absolute_tolerance,
                balance_relative_tolerance,
                balance_absolute_tolerance_J,
            )
        )
        if (
            any(not math.isfinite(x) or x <= 0 for x in tolerances)
            or maximum_iterations < 1
        ):
            raise ValueError(
                "Numerical tolerances and iteration capacity must be positive."
            )
        initial = jnp.asarray(
            initial_temperature_K, dtype=parameters.capacity_J_per_m3_K.dtype
        )
        if initial.shape != (mesh.coordinates.shape[0],):
            raise ValueError("Initial temperature must cover all P1 mesh vertices in K.")
        self.mesh = mesh
        self.parameters = parameters
        self.projection = projection
        self.evidence = evidence
        self.boundaries = tuple(boundaries)
        self.initial_temperature_K = initial
        self.region_indices = regions
        self.region_ids = ids
        (
            self.linear_relative_tolerance,
            self.linear_absolute_tolerance,
            self.balance_relative_tolerance,
            self.balance_absolute_tolerance_J,
        ) = tolerances
        self.maximum_iterations = int(maximum_iterations)
        semantic = SemanticProvenance(
            {
                "equation": PENNES_1948_DOI,
                "scheme": "backward-euler-consistent-P1-tetrahedron",
                "region_ids": ids,
                "regions": regions,
                "boundaries": [
                    (b.facet_ids, b.kind, b.asset_id) for b in self.boundaries
                ],
                "perfusion": "volumetric-1/s-constant-region",
                "source_scheme": "interval-constant",
                "solver": "native-PCG",
                "tolerances": tolerances,
                "maximum_iterations": self.maximum_iterations,
            },
            resource_ids={
                "mesh": mesh.mesh_id,
                "source_map": projection.projection_id,
                "evidence": evidence.evidence_id,
            },
        )
        self.plan_id = semantic.semantic_id

    def prepare(self) -> PreparedPennes1948:
        if not bool(self.parameters.admissible()) or any(
            not bool(b.admissible()) for b in self.boundaries
        ):
            raise ValueError(
                "Thermal parameters and boundary values must be finite and admissible."
            )
        if not bool(
            jnp.all(
                jnp.isfinite(self.initial_temperature_K)
                & (self.initial_temperature_K > 0)
            )
        ):
            raise ValueError("Initial absolute temperatures must be finite and positive.")
        disc = FiniteElementPlan(
            self.mesh,
            FiniteElementFieldSpec(
                "temperature",
                lagrange_element("tetrahedron", 1),
            ),
        ).prepare()
        exterior = set(np.asarray(disc.exterior_facet_domain.entity_indices).tolist())
        covered = [x for boundary in self.boundaries for x in boundary.facet_ids]
        if len(covered) != len(set(covered)) or set(covered) != exterior:
            raise ValueError(
                "Thermal exterior facets require a complete nonoverlapping boundary partition."
            )
        zero = jnp.zeros_like(self.initial_temperature_K)
        mass, diffusion, region_load = [], [], []
        for index in range(len(self.region_ids)):
            support = coefficient(
                jnp.asarray(np.asarray(self.region_indices) == index, dtype=zero.dtype),
                location="cell",
                support_id=disc.support.support_id,
                entity_set_id=disc.cell_domain.entity_set_id,
            )
            mass.append(
                _compiled(
                    disc,
                    MassAction(
                        "temperature", support, action_id=f"pennes-unit-mass-{index}"
                    ),
                ).affine_operator()
            )
            diffusion.append(
                _compiled(
                    disc,
                    TensorDiffusionAction(
                        "temperature", support, action_id=f"pennes-unit-diffusion-{index}"
                    ),
                ).affine_operator()
            )
            region_load.append(
                -_compiled(
                    disc,
                    SourceAction(
                        "temperature", support, action_id=f"pennes-unit-source-{index}"
                    ),
                ).full_residual(zero, None)
            )
        # The source projection uses the same native reference interpolation and
        # physical quadrature as SourceAction; no dense cell × node matrix.
        volumes = np.zeros((len(self.region_indices),), dtype=np.asarray(zero).dtype)
        sources, targets, weights = [], [], []
        for local in disc.prepare_local_regions(
            disc.cell_domain,
            field_names=("temperature",),
            maximum_derivative_order=0,
            kernel_mode="dense",
        ):
            metric = local.geometry_actions.realize(disc.default_runtime)
            load = local.reference_actions[0].interpolate_transpose(
                disc.default_runtime,
                metric.physical_weights,
            )
            entities = np.asarray(local.entity_indices)
            volumes[entities] = np.asarray(jnp.sum(metric.physical_weights, axis=1))
            sources.extend(np.repeat(entities, load.shape[1]).tolist())
            targets.extend(np.asarray(local.field_gathers[0]).reshape(-1).tolist())
            weights.extend(np.asarray(load).reshape(-1).tolist())
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ValueError("Thermal cells require positive finite physical volumes.")
        source_load = SparseLinearMap(
            EdgeRelation(
                np.asarray(sources),
                np.asarray(targets),
                source_size=volumes.size,
                target_size=zero.size,
            ),
            jnp.asarray(weights, dtype=zero.dtype),
        )
        boundary_loads, boundary_mass = [], []
        prescribed_masks, assigned_values = [], np.full(zero.shape, np.nan)
        faces = np.asarray(self.mesh.connectivity.faces)
        for index, boundary in enumerate(self.boundaries):
            domain = _subdomain(disc.exterior_facet_domain, boundary.facet_ids)
            boundary_loads.append(
                -_compiled(
                    disc,
                    BoundaryLoadAction(
                        "temperature",
                        1.0,
                        domain=domain,
                        action_id=f"pennes-boundary-load-{index}",
                    ),
                ).full_residual(zero, None)
            )
            boundary_mass.append(
                _compiled(
                    disc,
                    ExteriorFacetAction(
                        "temperature",
                        ("temperature",),
                        _facet_mass,
                        domain=domain,
                        action_id=f"pennes-boundary-mass-{index}",
                    ),
                ).affine_operator()
                if boundary.kind == "convection"
                else None
            )
            mask = np.zeros(zero.shape, dtype=bool)
            if boundary.kind == "dirichlet":
                mask[np.unique(faces[np.asarray(boundary.facet_ids)].reshape(-1))] = True
                conflict = (
                    mask
                    & np.isfinite(assigned_values)
                    & (assigned_values != float(boundary.value))
                )
                if np.any(conflict):
                    raise ValueError(
                        "Dirichlet patches prescribe conflicting temperatures at a shared vertex."
                    )
                assigned_values[mask] = float(boundary.value)
            prescribed_masks.append(jnp.asarray(mask))
        prescribed = np.isfinite(assigned_values)
        constraint = (
            dirichlet_constraint(disc, "temperature", boundary_mask=prescribed)
            if np.any(prescribed) and not np.all(prescribed)
            else None
        )
        if np.any(prescribed) and not np.array_equal(
            np.asarray(self.initial_temperature_K)[prescribed],
            assigned_values[prescribed],
        ):
            raise ValueError(
                "Initial temperature must satisfy the declared Dirichlet partition."
            )
        numeric = NumericRevision(
            self.plan_id,
            {
                "parameters": self.parameters,
                "boundaries": self.boundaries,
                "initial": self.initial_temperature_K,
            },
        )
        return PreparedPennes1948(
            self,
            _Pennes1948Geometry(
                disc,
                tuple(mass),
                tuple(diffusion),
                jnp.stack(region_load),
                source_load,
                jnp.asarray(volumes),
                tuple(boundary_loads),
                tuple(boundary_mass),
                tuple(prescribed_masks),
                constraint,
                tuple(np.flatnonzero(~prescribed).tolist()),
            ),
            canonical_fingerprint(
                {
                    "plan": self.plan_id,
                    "numeric": numeric.revision_id,
                    "field": disc.field_spaces[0].field_space_id,
                }
            ),
        )


class Pennes1948EnergyLedger(StrictModule, NonTrainableState):
    """J: storage + perfusion out + boundary out − retained source = residual."""

    storage_change_J: Array
    retained_category_J: Array
    perfusion_out_J: Array
    prescribed_flux_out_J: Array
    convection_out_J: Array
    dirichlet_out_J: Array
    balance_residual_J: Array


class Pennes1948State(StrictModule, NonTrainableState):
    temperature_K: Array
    time_s: Array
    accepted_steps: Array
    cumulative_energy: Pennes1948EnergyLedger
    prepared_id: str = eqx.field(static=True)


class Pennes1948SolveEvidence(StrictModule, NonTrainableState):
    finite: Array
    source_valid: Array
    source_conservation_error_W: Array
    linear_residual_norm: Array
    linear_iterations: Array
    linear_successful: Array
    balance_valid: Array
    successful: Array


class Pennes1948Candidate(StrictModule, NonTrainableState):
    previous_state: Pennes1948State
    proposed_state: Pennes1948State
    ledger: Pennes1948EnergyLedger
    volumetric_retained_heat_W_per_m3: Array
    volumetric_perfusion_exchange_W_per_m3: Array
    evidence: Pennes1948SolveEvidence
    physical_parameters: Pennes1948Parameters
    physical_boundaries: tuple[Pennes1948Boundary, ...]
    source_state_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def commit(
        self,
        current: Pennes1948State,
        /,
        *,
        prepared: PreparedPennes1948,
        source_state_id: str,
        accept: ArrayLike = True,
    ) -> Pennes1948State:
        if (
            prepared.prepared_id != self.prepared_id
            or current.prepared_id != self.prepared_id
            or source_state_id != self.source_state_id
        ):
            raise ValueError(
                "Cannot commit a foreign prepared or source-state thermal candidate."
            )
        accepted = jnp.asarray(accept, dtype=bool)
        if accepted.shape != ():
            raise ValueError("Thermal commit acceptance must be scalar.")
        valid = (
            self.evidence.successful
            & accepted
            & eqx.tree_equal(current, self.previous_state)
            & eqx.tree_equal(prepared.plan.parameters, self.physical_parameters)
            & eqx.tree_equal(prepared.plan.boundaries, self.physical_boundaries)
        )
        return jax.tree.map(
            lambda proposed, previous: jnp.where(valid, proposed, previous),
            self.proposed_state,
            current,
        )


class _Pennes1948Geometry(StrictModule, NonTrainableState):
    """Fixed native FEM actions and sparse registration; never trainable."""

    discretization: FiniteElementDiscretization
    mass_operators: tuple[AbstractLinearOperator, ...]
    diffusion_operators: tuple[AbstractLinearOperator, ...]
    region_source_load: Array
    source_load: SparseLinearMap
    cell_volume_m3: Array
    boundary_loads: tuple[Array, ...]
    boundary_mass: tuple[AbstractLinearOperator | None, ...]
    prescribed_masks: tuple[Array, ...]
    constraint: object
    free_dofs: tuple[int, ...] = eqx.field(static=True)


class PreparedPennes1948(StrictModule):
    plan: Pennes1948Plan
    geometry: _Pennes1948Geometry
    prepared_id: str = eqx.field(static=True)

    def initial_state(self, time_s: ArrayLike = 0.0) -> Pennes1948State:
        time = jnp.asarray(time_s, dtype=self.plan.initial_temperature_K.dtype)
        if time.shape != ():
            raise ValueError("Initial thermal time must be scalar.")
        zero = jnp.zeros_like(time)
        ledger = Pennes1948EnergyLedger(
            zero,
            jnp.zeros((len(RETAINED_HEAT_CATEGORIES),), dtype=time.dtype),
            zero,
            zero,
            zero,
            zero,
            zero,
        )
        return Pennes1948State(
            self.plan.initial_temperature_K,
            time,
            jnp.asarray(0),
            ledger,
            self.prepared_id,
        )

    def _mass(self, temperature):
        return sum(
            (
                c * op.mv(temperature)
                for c, op in zip(
                    self.plan.parameters.capacity_J_per_m3_K,
                    self.geometry.mass_operators,
                    strict=True,
                )
            ),
            jnp.zeros_like(temperature),
        )

    def _transport(self, temperature):
        p = self.plan.parameters
        image = jnp.zeros_like(temperature)
        for i, (mass, diffusion) in enumerate(
            zip(
                self.geometry.mass_operators,
                self.geometry.diffusion_operators,
                strict=True,
            )
        ):
            image += p.conductivity_W_per_m_K[i] * diffusion.mv(temperature)
            image += (
                p.perfusion_per_s[i]
                * p.blood_capacity_J_per_m3_K[i]
                * mass.mv(temperature)
            )
        for boundary, mass in zip(
            self.plan.boundaries, self.geometry.boundary_mass, strict=True
        ):
            if mass is not None:
                image += boundary.heat_transfer_W_per_m2_K * mass.mv(temperature)
        return image

    def propose(
        self, state: Pennes1948State, source: RetainedHeatLedger, /
    ) -> Pennes1948Candidate:
        if state.prepared_id != self.prepared_id:
            raise ValueError("Thermal state belongs to another prepared field.")
        p = self.plan.parameters
        dtype = state.temperature_K.dtype
        if any(
            value.dtype != dtype
            for value in (
                source.category_power_W,
                source.time_start_s,
                source.time_end_s,
                *jax.tree.leaves(p),
                *(
                    value
                    for boundary in self.plan.boundaries
                    for value in jax.tree.leaves(boundary)
                ),
            )
        ):
            raise ValueError(
                "Thermal source, physical parameters, boundaries and state must share one prepared precision."
            )
        q = self.plan.projection.project(source, self.geometry.cell_volume_m3)
        dt = source.time_end_s - source.time_start_s
        safe_dt = jnp.where(jnp.isfinite(dt) & (dt > 0), dt, 1.0)
        heat_load = self.geometry.source_load.mv(q)
        perfusion_load = jnp.sum(
            (p.perfusion_per_s * p.blood_capacity_J_per_m3_K * p.arterial_temperature_K)[
                :, None
            ]
            * self.geometry.region_source_load,
            axis=0,
        )
        boundary_supply = jnp.zeros_like(state.temperature_K)
        lift = jnp.zeros_like(state.temperature_K)
        boundary_valid = jnp.asarray(True)
        for boundary, load, mask in zip(
            self.plan.boundaries,
            self.geometry.boundary_loads,
            self.geometry.prescribed_masks,
            strict=True,
        ):
            boundary_valid &= boundary.admissible()
            if boundary.kind == "flux":
                boundary_supply -= boundary.value * load
            elif boundary.kind == "convection":
                boundary_supply += (
                    boundary.heat_transfer_W_per_m2_K * boundary.value * load
                )
            elif boundary.kind == "dirichlet":
                lift = jnp.where(mask, boundary.value, lift)
        rhs = self._mass(state.temperature_K) + safe_dt * (
            heat_load + perfusion_load + boundary_supply
        )

        def full_action(value):
            return self._mass(value) + safe_dt * self._transport(value)

        free = jnp.asarray(self.geometry.free_dofs, dtype=jnp.int32)

        def expand(value):
            if self.geometry.constraint is not None:
                return self.geometry.constraint.constraint_map.prolongation.mv(value)
            return jnp.zeros_like(state.temperature_K).at[free].set(value)

        if self.geometry.free_dofs:

            def action(value):
                return full_action(expand(value))[free]

            space = ArraySpace((len(self.geometry.free_dofs),), dtype=rhs.dtype)
            operator = FunctionLinearOperator(
                action,
                source=space,
                target=space,
                transpose_action=action,
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={
                        "self_adjoint": "construction",
                        "positive_definite": "construction",
                    },
                ),
                operator_id=self.prepared_id + ":backward-euler",
            )
            result = solve(
                LinearSystem(operator, problem_id=self.prepared_id),
                (rhs - full_action(lift))[free],
                policy=LinearSolvePolicy(
                    PCG(),
                    tolerance=TolerancePolicy(
                        relative=self.plan.linear_relative_tolerance,
                        absolute=self.plan.linear_absolute_tolerance,
                        max_steps=self.plan.maximum_iterations,
                    ),
                    failure=FailurePolicy("status"),
                ),
            )
            temperature = lift + expand(result.value)
            linear_success = result.successful
            residual_norm = result.diagnostics.residual_norm
            iterations = result.diagnostics.iterations
        else:
            temperature = lift
            linear_success = jnp.asarray(True)
            residual_norm = jnp.asarray(0, dtype=rhs.dtype)
            iterations = jnp.asarray(0)
        storage = jnp.sum(self._mass(temperature - state.temperature_K))
        perfusion_out = safe_dt * sum(
            (
                p.perfusion_per_s[i]
                * p.blood_capacity_J_per_m3_K[i]
                * jnp.sum(
                    mass.mv(temperature)
                    - p.arterial_temperature_K[i] * self.geometry.region_source_load[i]
                )
                for i, mass in enumerate(self.geometry.mass_operators)
            ),
            jnp.asarray(0, dtype=rhs.dtype),
        )
        flux_out, convection_out = (
            jnp.asarray(0, dtype=rhs.dtype),
            jnp.asarray(0, dtype=rhs.dtype),
        )
        for boundary, load, mass in zip(
            self.plan.boundaries,
            self.geometry.boundary_loads,
            self.geometry.boundary_mass,
            strict=True,
        ):
            if boundary.kind == "flux":
                flux_out += safe_dt * boundary.value * jnp.sum(load)
            elif mass is not None:
                convection_out += (
                    safe_dt
                    * boundary.heat_transfer_W_per_m2_K
                    * jnp.sum(mass.mv(temperature) - boundary.value * load)
                )
        full_residual = full_action(temperature) - rhs
        constrained = jnp.ones(state.temperature_K.shape, dtype=bool).at[free].set(False)
        dirichlet_out = -jnp.sum(jnp.where(constrained, full_residual, 0))
        category = safe_dt * jnp.sum(source.category_power_W, axis=1)
        balance = (
            storage
            + perfusion_out
            + flux_out
            + convection_out
            + dirichlet_out
            - jnp.sum(category)
        )
        scale = (
            jnp.abs(storage)
            + jnp.abs(perfusion_out)
            + jnp.abs(flux_out)
            + jnp.abs(convection_out)
            + jnp.abs(dirichlet_out)
            + jnp.sum(jnp.abs(category))
        )
        balance_valid = (
            jnp.abs(balance)
            <= self.plan.balance_absolute_tolerance_J
            + self.plan.balance_relative_tolerance * scale
        )
        conservation_error = jnp.sum(q * self.geometry.cell_volume_m3) - jnp.sum(
            source.retained_power_W
        )
        source_valid = (
            source.successful
            & (source.time_start_s == state.time_s)
            & (
                jnp.abs(conservation_error) * safe_dt
                <= self.plan.balance_absolute_tolerance_J
                + self.plan.balance_relative_tolerance * jnp.sum(jnp.abs(category))
            )
        )
        ledger = Pennes1948EnergyLedger(
            storage,
            category,
            perfusion_out,
            flux_out,
            convection_out,
            dirichlet_out,
            balance,
        )
        finite = (
            jnp.all(jnp.isfinite(temperature))
            & jnp.all(temperature > 0)
            & jnp.all(state.temperature_K > 0)
            & (state.accepted_steps >= 0)
            & jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(state)])
            )
            & jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(ledger)])
            )
        )
        successful = (
            finite
            & source_valid
            & linear_success
            & balance_valid
            & p.admissible()
            & boundary_valid
        )
        cumulative = jax.tree.map(
            lambda old, new: old + new, state.cumulative_energy, ledger
        )
        proposed = Pennes1948State(
            temperature,
            source.time_end_s,
            state.accepted_steps + 1,
            cumulative,
            self.prepared_id,
        )
        evidence = Pennes1948SolveEvidence(
            finite,
            source_valid,
            conservation_error,
            residual_norm,
            iterations,
            linear_success,
            balance_valid,
            successful,
        )
        cell_temperature = (
            self.geometry.source_load.transpose_mv(temperature)
            / self.geometry.cell_volume_m3
        )
        regions = jnp.asarray(self.plan.region_indices, dtype=jnp.int32)
        perfusion_exchange = (
            p.perfusion_per_s[regions]
            * p.blood_capacity_J_per_m3_K[regions]
            * (p.arterial_temperature_K[regions] - cell_temperature)
        )
        return Pennes1948Candidate(
            state,
            proposed,
            ledger,
            q,
            perfusion_exchange,
            evidence,
            p,
            self.plan.boundaries,
            source.source_state_id,
            self.prepared_id,
        )


__all__ = [
    "PENNES_1948_DOI",
    "Pennes1948Boundary",
    "Pennes1948Candidate",
    "Pennes1948EnergyLedger",
    "Pennes1948EvidenceBundle",
    "Pennes1948Parameters",
    "Pennes1948Plan",
    "Pennes1948SolveEvidence",
    "Pennes1948State",
    "PreparedPennes1948",
    "RetainedHeatProjection",
]
