#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import EntitySelection
from ..linalg import (
    AbstractPreconditioner,
    DifferentiationPolicy,
    FailurePolicy,
    FGMRES,
    JacobiPreconditionerBuilder,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    PreconditionerProperties,
    PreconditioningPolicy,
    prepare as prepare_linear,
    PreparedLinearSolve,
    solve,
)
from ..operators.integral.layer_potential import (
    BoundaryMeshEpoch,
    BoundaryRefinementResult,
    DP0BoundaryTransfer,
    LaplaceLayerPotential3D,
    LaplaceSingleLayerDP0AssemblyReport3D,
    LaplaceSingleLayerDP0Galerkin3D,
)


class LaplaceCapacitanceSensitivityEvidence3D(StrictModule, NonTrainableState):
    """Derivative envelope for one fixed topology/pair-class capacitance epoch."""

    maximum_quadrature_error: Array
    minimum_face_area: float = eqx.field(static=True)
    fixed_epoch: bool = eqx.field(static=True)
    permittivity_differentiable: bool = eqx.field(static=True)
    rhs_differentiable: bool = eqx.field(static=True)
    coordinates_differentiable: bool = eqx.field(static=True)
    topology_differentiable: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class LaplaceCapacitanceResult3D(StrictModule):
    """Unit-voltage conductor responses and their Maxwell capacitance matrix."""

    layer_density: Array
    capacitance: Array
    linear_results: tuple[LinearSolveResult, ...]
    potentials: tuple[LaplaceLayerPotential3D, ...]
    assembly_report: LaplaceSingleLayerDP0AssemblyReport3D
    sensitivity: LaplaceCapacitanceSensitivityEvidence3D
    permittivity: Array
    capacitance_reciprocity_defect: Array
    valid: Array
    conductor_names: tuple[str, ...] = eqx.field(static=True)
    conductor_selection_ids: tuple[str, ...] = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    @property
    def surface_charge_density(self) -> Array:
        return self.permittivity * self.layer_density


class LaplaceCapacitancePlan3D(StrictModule, NonTrainableState):
    """Static conductor, epoch, quadrature, solve, and differentiation contract."""

    epoch: BoundaryMeshEpoch
    galerkin: LaplaceSingleLayerDP0Galerkin3D
    masks: Array
    selections: tuple[EntitySelection, ...]
    linear_policy: LinearSolvePolicy
    conductor_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        epoch: BoundaryMeshEpoch,
        galerkin: LaplaceSingleLayerDP0Galerkin3D,
        conductors: Mapping[str, EntitySelection],
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(epoch, BoundaryMeshEpoch):
            raise TypeError("epoch must be BoundaryMeshEpoch.")
        if not isinstance(galerkin, LaplaceSingleLayerDP0Galerkin3D):
            raise TypeError("galerkin must be LaplaceSingleLayerDP0Galerkin3D.")
        if epoch.mesh.geometry_id != galerkin._binding.mesh.geometry_id:
            raise ValueError("Capacitance epoch and Galerkin geometry differ.")
        if not bool(galerkin.assembly_report.accuracy_supported):
            raise ValueError(
                "Galerkin quadrature does not support the requested accuracy."
            )
        names, selections, masks = _conductors(galerkin, conductors)
        policy = _default_policy() if linear_policy is None else linear_policy
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        if policy.differentiation.mode not in ("mathematical", "rhs-only"):
            raise ValueError(
                "Prepared capacitance requires mathematical or rhs-only differentiation."
            )
        self.epoch = epoch
        self.galerkin = galerkin
        self.masks = jnp.asarray(masks, dtype=galerkin.face_areas.dtype)
        self.selections = selections
        self.linear_policy = policy
        self.conductor_names = names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "laplace-capacitance-plan-3d",
                "epoch": epoch.epoch_id,
                "galerkin": galerkin.assembly_report.report_id,
                "conductors": [selection.selection_id for selection in selections],
                "masks": array_tree_fingerprint(masks),
                "linear": {
                    "method": type(policy.method).__qualname__,
                    "differentiation": policy.differentiation.mode,
                    "preconditioning": (
                        None
                        if policy.preconditioning is None
                        else {
                            "source": type(
                                policy.preconditioning.preconditioner
                                if policy.preconditioning.preconditioner is not None
                                else policy.preconditioning.builder
                            ).__qualname__,
                            "side": policy.preconditioning.side,
                            "refresh": policy.preconditioning.refresh_policy,
                        }
                    ),
                },
            }
        )

    def prepare(self, /) -> "PreparedLaplaceCapacitance3D":
        return PreparedLaplaceCapacitance3D(self)


class PreparedLaplaceCapacitance3D(StrictModule, NonTrainableState):
    """Prepared block-capable fixed-epoch capacitance solve."""

    epoch: BoundaryMeshEpoch
    galerkin: LaplaceSingleLayerDP0Galerkin3D
    masks: Array
    selections: tuple[EntitySelection, ...]
    prepared_linear: PreparedLinearSolve
    sensitivity: LaplaceCapacitanceSensitivityEvidence3D
    conductor_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: LaplaceCapacitancePlan3D, /):
        if not isinstance(plan, LaplaceCapacitancePlan3D):
            raise TypeError("plan must be LaplaceCapacitancePlan3D.")
        problem = LinearSystem(
            plan.galerkin.strong_operator,
            problem_id=f"laplace-capacitance:{plan.epoch.epoch_id}",
        )
        prepared_linear = prepare_linear(problem, plan.linear_policy)
        maximum_error = jnp.max(plan.galerkin.assembly_report.maximum_errors)
        minimum_area = float(np.min(np.asarray(plan.galerkin.face_areas)))
        sensitivity_id = canonical_fingerprint(
            {
                "kind": "laplace-capacitance-sensitivity-evidence-3d",
                "epoch": plan.epoch.epoch_id,
                "quadrature": plan.galerkin.assembly_report.report_id,
                "maximum_error": float(maximum_error),
                "minimum_area": minimum_area,
                "permittivity_differentiable": True,
                "rhs_differentiable": True,
                "coordinates_differentiable": True,
                "topology_differentiable": False,
            }
        )
        sensitivity = LaplaceCapacitanceSensitivityEvidence3D(
            maximum_quadrature_error=maximum_error,
            minimum_face_area=minimum_area,
            fixed_epoch=True,
            permittivity_differentiable=True,
            rhs_differentiable=True,
            coordinates_differentiable=True,
            topology_differentiable=False,
            evidence_id=sensitivity_id,
        )
        self.epoch = plan.epoch
        self.galerkin = plan.galerkin
        self.masks = plan.masks
        self.selections = plan.selections
        self.prepared_linear = prepared_linear
        self.sensitivity = sensitivity
        self.conductor_names = plan.conductor_names
        self.plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-laplace-capacitance-3d",
                "plan": plan.plan_id,
                "linear": prepared_linear.plan.plan_id,
                "sensitivity": sensitivity_id,
            }
        )

    def solve(
        self,
        /,
        *,
        permittivity: ArrayLike = 1.0,
        conductor_potentials: ArrayLike | None = None,
    ) -> LaplaceCapacitanceResult3D:
        self.epoch.validate_mesh(self.galerkin._binding.mesh)
        epsilon = jnp.asarray(permittivity, dtype=self.galerkin.face_areas.dtype)
        if epsilon.shape != ():
            raise ValueError("permittivity must be one scalar.")
        epsilon = eqx.error_if(
            epsilon,
            ~jnp.isfinite(epsilon) | (epsilon <= 0.0),
            "permittivity must be finite and positive.",
        )
        right_hand_sides = self.masks
        if conductor_potentials is not None:
            potentials_ = jnp.asarray(
                conductor_potentials, dtype=self.galerkin.face_areas.dtype
            )
            if potentials_.shape != (len(self.conductor_names),):
                raise ValueError(
                    "conductor_potentials must have one value per conductor."
                )
            right_hand_sides = potentials_[:, None] * right_hand_sides
        linear_results = tuple(
            solve(self.prepared_linear, right_hand_sides[index])
            for index in range(len(self.conductor_names))
        )
        layer_density = jnp.stack(
            tuple(jnp.asarray(result.value) for result in linear_results), axis=1
        )
        potentials = tuple(
            self.galerkin.potential(layer_density[:, index])
            for index in range(len(self.conductor_names))
        )
        surface_charge = epsilon * layer_density
        capacitance = ein.contract(
            "if,f,fj->ij",
            self.masks,
            self.galerkin.face_areas,
            surface_charge,
            backend="jax",
        )
        scale = jnp.maximum(
            jnp.max(jnp.abs(capacitance)), jnp.finfo(capacitance.dtype).tiny
        )
        reciprocity = jnp.max(jnp.abs(capacitance - capacitance.T)) / scale
        linear_valid = jnp.all(
            jnp.stack(
                tuple(
                    result.successful & result.diagnostics.finite
                    for result in linear_results
                )
            )
        )
        finite = (
            jnp.all(jnp.isfinite(layer_density))
            & jnp.all(jnp.isfinite(surface_charge))
            & jnp.all(jnp.isfinite(capacitance))
            & jnp.isfinite(reciprocity)
        )
        valid = self.galerkin.assembly_report.accuracy_supported & linear_valid & finite
        return LaplaceCapacitanceResult3D(
            layer_density=layer_density,
            capacitance=capacitance,
            linear_results=linear_results,
            potentials=potentials,
            assembly_report=self.galerkin.assembly_report,
            sensitivity=self.sensitivity,
            permittivity=epsilon,
            capacitance_reciprocity_defect=reciprocity,
            valid=valid,
            conductor_names=self.conductor_names,
            conductor_selection_ids=tuple(
                selection.selection_id for selection in self.selections
            ),
            epoch_id=self.epoch.epoch_id,
        )


class LaplaceCapacitanceCoordinateJVP3D(StrictModule):
    """Exact discrete fixed-epoch JVP from Galerkin geometry tangents."""

    layer_density_tangent: Array
    capacitance_tangent: Array
    operator_tangent_action: Array
    valid: Array
    epoch_id: str = eqx.field(static=True)


class PreparedLaplaceStableDualCalderon3D(AbstractPreconditioner):
    """Prepared stable-dual first-kind Calderón action for a declared mesh family."""

    matrix: Array
    shape_regularity_margin: float = eqx.field(static=True)
    dual_mass_condition_number: float = eqx.field(static=True)

    def apply(self, residual, /, *, iteration=None):
        del iteration
        value = self.space.validate(residual)
        return self.matrix @ value


def prepare_laplace_stable_dual_calderon_3d(
    galerkin: LaplaceSingleLayerDP0Galerkin3D,
    dual_cross_mass: ArrayLike,
    dual_hypersingular: ArrayLike,
    /,
    *,
    shape_regularity_margin: float,
    gauge_weight: float = 1.0,
) -> PreparedLaplaceStableDualCalderon3D:
    """Prepare M_d^-T (W_tilde + R) M_d^-1 with rank/shape evidence."""

    if not isinstance(galerkin, LaplaceSingleLayerDP0Galerkin3D):
        raise TypeError("galerkin must be LaplaceSingleLayerDP0Galerkin3D.")
    mass = np.asarray(dual_cross_mass, dtype=float)
    hypersingular = np.asarray(dual_hypersingular, dtype=float)
    count = galerkin.face_count
    if mass.shape != (count, count) or hypersingular.shape != (count, count):
        raise ValueError("Stable-dual matrices must be square DP0 maps.")
    margin = float(shape_regularity_margin)
    gauge = float(gauge_weight)
    if not np.isfinite(margin) or margin <= 0.0 or not np.isfinite(gauge) or gauge <= 0.0:
        raise ValueError("Shape margin and gauge weight must be finite and positive.")
    singular_values = np.linalg.svd(mass, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1])
    if singular_values[-1] <= np.finfo(float).eps * singular_values[0] * count:
        raise ValueError("Stable-dual cross mass is rank deficient.")
    inverse_mass = np.linalg.inv(mass)
    constant = np.ones((count,), dtype=float)
    rank_one = gauge * np.outer(constant, constant) / float(count * count)
    matrix = inverse_mass.T @ (hypersingular + rank_one) @ inverse_mass
    symmetric = 0.5 * (matrix + matrix.T)
    if np.min(np.linalg.eigvalsh(symmetric)) <= 0.0:
        raise ValueError("Stable-dual Calderón action is not positive definite.")
    matrix = symmetric
    return PreparedLaplaceStableDualCalderon3D(
        space=galerkin.strong_operator.source,
        properties=PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "linear": "construction",
                "stationary": "construction",
                "self_adjoint": "construction",
                "positive_definite": "verified",
            },
        ),
        preconditioner_id=canonical_fingerprint(
            {
                "kind": "laplace-stable-dual-calderon-3d",
                "galerkin": galerkin.assembly_report.report_id,
                "mass": array_tree_fingerprint(mass),
                "hypersingular": array_tree_fingerprint(hypersingular),
                "shape_margin": margin,
            }
        ),
        matrix=jnp.asarray(matrix),
        shape_regularity_margin=margin,
        dual_mass_condition_number=condition,
    )


def differentiate_laplace_capacitance_coordinates_3d(
    prepared: PreparedLaplaceCapacitance3D,
    result: LaplaceCapacitanceResult3D,
    operator_tangents: ArrayLike,
    area_tangents: ArrayLike,
    /,
) -> LaplaceCapacitanceCoordinateJVP3D:
    """Apply the exact implicit discrete JVP for frozen topology/pair classes."""

    if result.epoch_id != prepared.epoch.epoch_id:
        raise ValueError("Coordinate JVP received a stale capacitance result.")
    d_operator = jnp.asarray(operator_tangents)
    d_area = jnp.asarray(area_tangents, dtype=prepared.galerkin.face_areas.dtype)
    face_count = prepared.galerkin.face_count
    if d_operator.ndim != 3 or d_operator.shape[1:] != (face_count, face_count):
        raise ValueError("operator_tangents must have shape (parameter, face, face).")
    if d_area.shape != (d_operator.shape[0], face_count):
        raise ValueError("area_tangents must have shape (parameter, face).")
    density_tangents = []
    capacitance_tangents = []
    actions = []
    for parameter in range(d_operator.shape[0]):
        action = d_operator[parameter] @ result.layer_density
        actions.append(action)
        d_density = jnp.stack(
            tuple(
                solve(prepared.prepared_linear, -action[:, column]).value
                for column in range(len(prepared.conductor_names))
            ),
            axis=1,
        )
        density_tangents.append(d_density)
        capacitance_tangents.append(
            result.permittivity
            * (
                ein.contract(
                    "if,f,fj->ij",
                    prepared.masks,
                    d_area[parameter],
                    result.layer_density,
                    backend="jax",
                )
                + ein.contract(
                    "if,f,fj->ij",
                    prepared.masks,
                    prepared.galerkin.face_areas,
                    d_density,
                    backend="jax",
                )
            )
        )
    density = jnp.stack(density_tangents)
    capacitance = jnp.stack(capacitance_tangents)
    valid = (
        result.valid & jnp.all(jnp.isfinite(density)) & jnp.all(jnp.isfinite(capacitance))
    )
    return LaplaceCapacitanceCoordinateJVP3D(
        density,
        capacitance,
        jnp.stack(actions),
        valid,
        prepared.epoch.epoch_id,
    )


class LaplaceCapacitanceEpochTransition3D(StrictModule, NonTrainableState):
    """Candidate topology transition with explicit conductor correspondence."""

    source: PreparedLaplaceCapacitance3D
    target: PreparedLaplaceCapacitance3D
    transfer: DP0BoundaryTransfer
    refinement: BoundaryRefinementResult
    conductor_correspondence: tuple[tuple[str, str], ...] = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)


def _conductors(
    galerkin: LaplaceSingleLayerDP0Galerkin3D,
    conductors: Mapping[str, EntitySelection],
    /,
) -> tuple[tuple[str, ...], tuple[EntitySelection, ...], np.ndarray]:
    if not isinstance(conductors, Mapping):
        raise TypeError("conductors must map names to EntitySelection values.")
    items = tuple(
        sorted((str(name), selection) for name, selection in conductors.items())
    )
    if not items or any(not name for name, _ in items):
        raise ValueError("conductors must contain non-empty names.")
    names = tuple(name for name, _ in items)
    selections = tuple(selection for _, selection in items)
    if not all(isinstance(selection, EntitySelection) for selection in selections):
        raise TypeError("Every conductor must be an EntitySelection.")
    masks = []
    for selection in selections:
        if selection.entity_set_id != galerkin.surface_entities.entity_set_id:
            raise ValueError("Conductor selection does not match the prepared surface.")
        mask = np.asarray(selection.mask, dtype=bool)
        if mask.shape != (galerkin.face_count,) or not np.any(mask):
            raise ValueError("Every conductor must select at least one surface face.")
        masks.append(mask)
    matrix = np.stack(masks)
    if np.any(np.sum(matrix, axis=0) != 1):
        raise ValueError("Conductor selections must disjointly cover every surface face.")
    components = np.asarray(galerkin.face_component_ids, dtype=np.int32)
    for component in range(galerkin.component_count):
        component_faces = components == component
        owners = np.flatnonzero(np.any(matrix[:, component_faces], axis=1))
        if owners.size != 1 or not np.all(matrix[owners[0], component_faces]):
            raise ValueError(
                "Each surface component must belong to exactly one conductor."
            )
    return names, selections, matrix


def _default_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
        differentiation=DifferentiationPolicy("mathematical"),
        failure=FailurePolicy("status"),
    )


def advance_laplace_capacitance_3d(
    source: PreparedLaplaceCapacitance3D,
    refinement: BoundaryRefinementResult,
    target_galerkin: LaplaceSingleLayerDP0Galerkin3D,
    target_conductors: Mapping[str, EntitySelection],
    /,
    *,
    conductor_correspondence: Mapping[str, str],
    linear_policy: LinearSolvePolicy | None = None,
) -> LaplaceCapacitanceEpochTransition3D:
    """Prepare an atomic candidate epoch; topology transitions are nondifferentiable."""

    if not isinstance(source, PreparedLaplaceCapacitance3D):
        raise TypeError("source must be PreparedLaplaceCapacitance3D.")
    if not isinstance(refinement, BoundaryRefinementResult):
        raise TypeError("refinement must be BoundaryRefinementResult.")
    if refinement.source_epoch.epoch_id != source.epoch.epoch_id:
        raise ValueError("Boundary refinement does not originate at the prepared source.")
    if not isinstance(conductor_correspondence, Mapping):
        raise TypeError("conductor_correspondence must be an explicit mapping.")
    pairs = tuple(sorted((str(a), str(b)) for a, b in conductor_correspondence.items()))
    if tuple(value[0] for value in pairs) != tuple(sorted(source.conductor_names)):
        raise ValueError(
            "Every source conductor requires explicit target correspondence."
        )
    target_names = tuple(sorted(str(name) for name in target_conductors))
    if tuple(sorted(value[1] for value in pairs)) != target_names:
        raise ValueError(
            "Target conductor correspondence must be one-to-one and complete."
        )
    target_plan = LaplaceCapacitancePlan3D(
        refinement.target_epoch,
        target_galerkin,
        target_conductors,
        linear_policy=(
            source.prepared_linear.plan.policy if linear_policy is None else linear_policy
        ),
    )
    target = target_plan.prepare()
    transition_id = canonical_fingerprint(
        {
            "kind": "laplace-capacitance-epoch-transition-3d",
            "source": source.prepared_id,
            "target": target.prepared_id,
            "refinement": refinement.result_id,
            "transfer": refinement.transfer.transfer_id,
            "conductor_correspondence": pairs,
            "differentiable": False,
        }
    )
    return LaplaceCapacitanceEpochTransition3D(
        source=source,
        target=target,
        transfer=refinement.transfer,
        refinement=refinement,
        conductor_correspondence=pairs,
        differentiable=False,
        transition_id=transition_id,
    )


__all__ = [
    "LaplaceCapacitanceCoordinateJVP3D",
    "LaplaceCapacitanceEpochTransition3D",
    "LaplaceCapacitancePlan3D",
    "LaplaceCapacitanceResult3D",
    "LaplaceCapacitanceSensitivityEvidence3D",
    "PreparedLaplaceStableDualCalderon3D",
    "PreparedLaplaceCapacitance3D",
    "advance_laplace_capacitance_3d",
    "differentiate_laplace_capacitance_coordinates_3d",
    "prepare_laplace_stable_dual_calderon_3d",
]
