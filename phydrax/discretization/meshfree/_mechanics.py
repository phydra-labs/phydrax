#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Meshfree solid mechanics and generalized Stokes on prepared point clouds.

Kinematics use the cloud's admitted strong derivatives: ``G[n, a, j] = d_j u_a``.
Constitutive semantics are owned by ``operators.mechanics``
(``LinearElasticityTensor``, ``NeoHookeanLaw``); coupled block assembly and
boundary-row ownership by ``PointBlockSystemPlan``; solves by native
``phydrax.linalg``/``phydrax.nonlinear`` owners. Boundary integrals (traction
resultants, external work) use the cloud's declared boundary quadrature and are
NaN, never a plausible zero, when none is declared.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    assemble_sparse,
    BlockFactorizationPreconditionerBuilder,
    BlockLinearOperator,
    BlockSpace,
    DiagonalLinearOperator,
    FailurePolicy,
    GMRES,
    ILUPreconditionerBuilder,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    PreconditioningPolicy,
    prepare,
    prepare_linearization,
    PreparedLinearization,
    PreparedLinearSolve,
    solve,
    SparseAssemblyPolicy,
    TolerancePolicy,
)
from ...nonlinear import (
    NewtonForcingPolicy,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    root,
)
from ...operators.mechanics import LinearElasticityTensor, NeoHookeanLaw
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import parse
from .._point_cloud import PreparedPointCloudDiscretization
from .._point_cloud_pde import (
    _unit,
    PointBoundaryPlan,
    PointDiffusionForm,
    PointStabilityPolicy,
)
from ._boundary import (
    discretization_family,
    PointDerivativeFamily,
    PointGhostLayerPlan,
    PreparedPointGhostLayer,
)
from ._multilevel import (
    MeshfreeCoarseningPolicy,
    MeshfreeNearNullspace,
)
from ._systems import (
    PointBlockSystemPlan,
    PointBlockSystemResult,
    PointCouplingEvidence,
    PreparedPointBlockSystem,
)


_AXES = ("x", "y", "z")


class MechanicsStatus(IntEnum):
    """Fail-closed meshfree mechanics acceptance in reporting precedence."""

    ACCEPTED = 0
    NONFINITE = 1
    INADMISSIBLE_DEFORMATION = 2
    SOLVE_REFUSED = 3
    RESIDUAL_TOO_LARGE = 4


def displacement_gradient(
    discretization: PreparedPointCloudDiscretization, field: ArrayLike, /
) -> Array:
    """``G[n, a, j] = d_j u_a`` from the cloud's admitted first-derivative rows."""
    value = jnp.asarray(field, dtype=jnp.float64)
    if value.shape != (discretization.state_shape[0], discretization.spatial_dimension):
        raise ValueError("Mechanics fields must have shape (points, spatial dimension).")
    return discretization.gradient(value)


def _symmetric(tensor: Array, /) -> Array:
    return 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))


def _components(dimension: int, /) -> tuple[str, ...]:
    return _AXES[:dimension]


def _rigid_nullspace(dimension: int, /) -> MeshfreeNearNullspace:
    # Rigid-body candidates are defined for planar and spatial vector fields.
    return MeshfreeNearNullspace("rigid-body" if dimension > 1 else "constant")


def _boundary_rows(
    discretization: PreparedPointCloudDiscretization, /
) -> tuple[Array, Array, Array | None]:
    plan = discretization.plan
    weights = plan.boundary_quadrature_weights
    return plan.boundary_mask, plan.boundary_normals, weights


def _boundary_integral(weights: Array | None, values: Array, /) -> Array:
    """Boundary quadrature of point values; NaN when no quadrature is declared."""
    if weights is None:
        return jnp.full(values.shape[1:], jnp.nan, dtype=values.dtype)
    return ein.contract("n,n...->...", weights, values)


def _status(
    finite: Array, admissible: Array, solved: Array, residual_ok: Array, /
) -> Array:
    return jnp.where(
        ~finite,
        int(MechanicsStatus.NONFINITE),
        jnp.where(
            ~admissible,
            int(MechanicsStatus.INADMISSIBLE_DEFORMATION),
            jnp.where(
                ~solved,
                int(MechanicsStatus.SOLVE_REFUSED),
                jnp.where(
                    ~residual_ok,
                    int(MechanicsStatus.RESIDUAL_TOO_LARGE),
                    int(MechanicsStatus.ACCEPTED),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _validated_boundary(
    discretization: PreparedPointCloudDiscretization, boundary: PointBoundaryPlan, /
) -> None:
    if not isinstance(discretization, PreparedPointCloudDiscretization):
        raise TypeError("discretization must be a PreparedPointCloudDiscretization.")
    if not isinstance(boundary, PointBoundaryPlan):
        raise TypeError("boundary must be a PointBoundaryPlan.")
    if boundary.components != discretization.spatial_dimension:
        raise ValueError("Mechanics boundary plans declare one component per axis.")


# --------------------------------------------------------------------------
# Linear elasticity
# --------------------------------------------------------------------------


@final
class MeshfreeElasticityResult(StrictModule):
    """Committed small-strain equilibrium and its physical evidence.

    ``displacement`` holds the candidate only when ``status`` is ACCEPTED and
    NaN otherwise. ``strain_energy`` is ``1/2 sum_n w_n sigma:eps``;
    ``external_work`` is ``sum w f.u + sum_b w_b (sigma n).u`` with the computed
    boundary traction (Clapeyron: ``2 U = W`` up to integration-by-parts
    consistency, reported as ``clapeyron_defect``). ``force_balance_defect`` is
    ``|int f + oint sigma n|`` relative to the load scale. Strain, stress and
    traction use the solved stencils (ghost-extended on the ghost route, where
    ``block.ghost_values`` holds the ghost displacements).
    """

    displacement: Array
    candidate_displacement: Array
    strain: Array
    stress: Array
    boundary_traction: Array
    strain_energy: Array
    external_work: Array
    clapeyron_defect: Array
    body_force_resultant: Array
    boundary_force_resultant: Array
    force_balance_defect: Array
    block: PointBlockSystemResult
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(MechanicsStatus.ACCEPTED)


MeshfreeElasticityPreconditioner: TypeAlias = Literal["auxiliary", "multigrid", "ilu"]
MeshfreeTractionRoute: TypeAlias = Literal["ghost", "square"]


def _ghost_gradient(ghosts: PreparedPointGhostLayer, unknowns: Array, /) -> Array:
    """``G[n, a, k] = D_k u_a`` at the cloud rows by ghost-extended stencils."""
    d = ghosts.points.shape[1]
    return jnp.stack(
        [
            ghosts.family.apply(unknowns, _unit(d, k))[: ghosts.cloud_count]
            for k in range(d)
        ],
        axis=2,
    )


def _traction_ghosts(
    discretization: PreparedPointCloudDiscretization,
    boundary: PointBoundaryPlan,
    route: MeshfreeTractionRoute,
    /,
) -> PreparedPointGhostLayer | None:
    """The ghost layer of the traction/Robin rows on the ghost route."""
    selected = parse(route, MeshfreeTractionRoute, "traction_route")
    match selected:
        case "ghost":
            flux = any(c.kind in ("neumann", "robin") for c in boundary.conditions)
            return PointGhostLayerPlan(boundary).prepare(discretization) if flux else None
        case "square":
            return None
        case unreachable:
            assert_never(unreachable)


@final
class MeshfreeElasticityPlan(StrictModule):
    """Small-strain elasticity ``-div(C : eps(u)) = f`` with traction/displacement rows.

    ``tensor`` is one verified homogeneous ``LinearElasticityTensor`` of the
    cloud dimension. Neumann rows of ``boundary`` are tractions ``sigma n = t``
    (Robin rows add ``k u``); Dirichlet rows prescribe displacement.

    ``preconditioner`` selects the block owner's GMRES preconditioning:

    - ``"auxiliary"`` (default in 2-D and 3-D): a symmetric Gauss--Seidel
      meshfree hierarchy, on rigid-body candidates, of the SPD Delaunay P1
      elasticity stiffness on the same points, with traction/Robin rows eliminated exactly
      (``meshfree_auxiliary_builder``). Its iteration count stays nearly
      constant under refinement. It needs Dirichlet rows in every component,
      no periodic rows, and a 2-D or 3-D cloud; other boundaries are refused
      with a pointer to ``"multigrid"``.
    - ``"multigrid"`` (default in 1-D): block meshfree multigrid of the
      collocated operator (rigid-body transfers, eliminated Dirichlet
      coordinates). No stationary smoother contracts collocated elasticity:
      measured rollers tension is refused from about a thousand points.
    - ``"ilu"``: ILU of the assembled block operator, refused on rollers
      tension already at 289 points.

    ``traction_route="ghost"`` (default) keeps the PDE at every traction/Robin
    boundary point and moves the condition to the row of one ghost unknown
    outside the domain per such point (``PointGhostLayerPlan``; ghost rows are
    divided by the ghost offset). Square collocation of traction rows
    (``"square"``) leaves one-sided stencils whose spurious modes the stability
    assessment refuses (2-D and 3-D jittered clouds measured nonpositive real
    parts). ``stability`` is passed to the block owner's square-collocation spectral
    assessment (``"require-assessment"`` refuses unstable operators;
    ``"diagnostic"`` records the evidence on ``block.stability``). An explicit
    ``linear_policy`` owns its preconditioning and excludes
    ``preconditioner``. ``assembly_policy`` defaults to the block owner's
    capacity-scaled limits, which also budget the hierarchy's coarse products.
    Floating bodies (no anchoring rows) are refused by the block owner.
    """

    system: PointBlockSystemPlan
    tensor: LinearElasticityTensor
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        boundary: PointBoundaryPlan,
        tensor: LinearElasticityTensor,
        /,
        *,
        form: PointDiffusionForm = "collocated",
        linear_policy: LinearSolvePolicy | None = None,
        preconditioner: MeshfreeElasticityPreconditioner | None = None,
        coarsening: MeshfreeCoarseningPolicy | None = None,
        assembly_policy: SparseAssemblyPolicy | None = None,
        stability: PointStabilityPolicy = "require-assessment",
        traction_route: MeshfreeTractionRoute = "ghost",
    ) -> None:
        _validated_boundary(discretization, boundary)
        if not isinstance(tensor, LinearElasticityTensor):
            raise TypeError("tensor must be a LinearElasticityTensor.")
        dimension = discretization.spatial_dimension
        if tensor.dimension != dimension:
            raise ValueError("Elasticity tensor dimension must match the cloud.")
        if not bool(tensor.successful):
            raise ValueError(
                "Elasticity tensor failed symmetry or coercivity verification."
            )
        if linear_policy is not None and preconditioner is not None:
            raise ValueError("An explicit linear_policy owns its preconditioning.")
        default = "auxiliary" if dimension > 1 else "multigrid"
        selected = parse(
            default
            if preconditioner is None and linear_policy is None
            else preconditioner,
            MeshfreeElasticityPreconditioner | None,
            "preconditioner",
        )
        if coarsening is not None and selected != "multigrid":
            raise ValueError("coarsening configures only the multigrid preconditioner.")
        candidates = _rigid_nullspace(dimension)
        ghosts = _traction_ghosts(discretization, boundary, traction_route)
        system = PointBlockSystemPlan(
            discretization,
            boundary,
            components=_components(dimension),
            form=form,
            ghosts=ghosts,
            linear_policy=linear_policy,
            assembly_policy=assembly_policy,
            multigrid=candidates if selected == "multigrid" else None,
            coarsening=coarsening,
            auxiliary=candidates if selected == "auxiliary" else None,
            stability=stability,
        )
        self.system = system
        self.tensor = tensor
        self.plan_id = canonical_fingerprint(
            {"kind": "meshfree-elasticity", "system": system.plan_id}
        )

    @property
    def coefficients(self) -> Array:
        """Block coupling ``C[a, b, i, j] = stiffness[a, i, b, j]``."""
        return jnp.transpose(self.tensor.stiffness, (0, 2, 1, 3))

    def prepare(self, /) -> PreparedMeshfreeElasticity:
        return PreparedMeshfreeElasticity(self)


@final
class PreparedMeshfreeElasticity(StrictModule, NonTrainableState):
    """Prepared block assembly, multigrid solve, and mechanics actions."""

    plan: MeshfreeElasticityPlan
    block: PreparedPointBlockSystem

    def __init__(self, plan: MeshfreeElasticityPlan, /) -> None:
        if not isinstance(plan, MeshfreeElasticityPlan):
            raise TypeError("plan must be a MeshfreeElasticityPlan.")
        self.plan = plan
        self.block = plan.system.prepare(plan.coefficients)

    @property
    def discretization(self) -> PreparedPointCloudDiscretization:
        return self.plan.system.discretization

    @property
    def coupling_evidence(self) -> PointCouplingEvidence:
        return self.block.evidence

    def strain(self, displacement: ArrayLike, /) -> Array:
        return _symmetric(displacement_gradient(self.discretization, displacement))

    def stress(self, displacement: ArrayLike, /) -> Array:
        return self.plan.tensor.stress(self.strain(displacement))

    def traction(self, displacement: ArrayLike, normals: ArrayLike, /) -> Array:
        """Pointwise ``sigma(u) n`` for per-point normals ``(points, d)``."""
        normal = jnp.asarray(normals, dtype=jnp.float64)
        stress = self.stress(displacement)
        if normal.shape != stress.shape[:2]:
            raise ValueError("normals must have shape (points, spatial dimension).")
        return ein.contract("nij,nj->ni", stress, normal)

    def residual(
        self,
        displacement: ArrayLike,
        body_force: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Physical rows: bulk ``-div sigma - f``, traction ``sigma n - t``, ``u - g``."""
        return self.block.apply(displacement) - self.block.physical_rhs(
            body_force, boundary_values=boundary_values
        )

    def strain_energy(self, displacement: ArrayLike, /) -> Array:
        strain = self.strain(displacement)
        density = 0.5 * ein.contract(
            "nij,nij->n", self.plan.tensor.stress(strain), strain
        )
        return jnp.sum(self.discretization.quadrature_weights * density)

    def solve(
        self,
        body_force: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> MeshfreeElasticityResult:
        force = jnp.asarray(body_force, dtype=jnp.float64)
        block = self.block.solve(force, boundary_values=boundary_values)
        candidate = block.values
        cloud = self.discretization
        weights = cloud.quadrature_weights
        ghosts = self.plan.system.ghosts
        # Strain by the solved stencils: on the ghost route the ghost-extended
        # ones, so the reported tractions are those the equations enforce.
        strain = (
            self.strain(candidate)
            if ghosts is None or block.ghost_values is None
            else _symmetric(
                _ghost_gradient(ghosts, jnp.concatenate((candidate, block.ghost_values)))
            )
        )
        stress = self.plan.tensor.stress(strain)
        mask, normals, boundary_weights = _boundary_rows(cloud)
        traction = jnp.where(
            mask[:, None], ein.contract("nij,nj->ni", stress, normals), 0.0
        )
        energy = 0.5 * jnp.sum(weights * ein.contract("nij,nij->n", stress, strain))
        work = jnp.sum(weights * jnp.sum(force * candidate, axis=1)) + _boundary_integral(
            boundary_weights, jnp.sum(traction * candidate, axis=1)
        )
        body = ein.contract("n,na->a", weights, force)
        surface = _boundary_integral(boundary_weights, traction)
        scale = jnp.linalg.norm(body) + _boundary_integral(
            boundary_weights, jnp.linalg.norm(traction, axis=1)
        )
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(block.residual_norm)
        status = _status(
            finite,
            jnp.asarray(True),
            block.linear_result.successful & block.coupling_evidence.successful,
            block.residual_norm <= block.residual_tolerance,
        )
        accepted = status == int(MechanicsStatus.ACCEPTED)
        return MeshfreeElasticityResult(
            displacement=jnp.where(accepted, candidate, jnp.nan),
            candidate_displacement=candidate,
            strain=strain,
            stress=stress,
            boundary_traction=traction,
            strain_energy=energy,
            external_work=work,
            clapeyron_defect=jnp.abs(2.0 * energy - work)
            / jnp.maximum(jnp.abs(work), jnp.finfo(jnp.float64).tiny),
            body_force_resultant=body,
            boundary_force_resultant=surface,
            force_balance_defect=jnp.linalg.norm(body + surface)
            / jnp.maximum(scale, jnp.finfo(jnp.float64).tiny),
            block=block,
            status=status,
        )


# --------------------------------------------------------------------------
# Finite-strain hyperelasticity
# --------------------------------------------------------------------------


@final
class MeshfreeHyperelasticResult(StrictModule):
    """Incrementally loaded hyperelastic equilibrium with energy/work evidence.

    Each load factor ``k / load_steps`` scales body force, traction and
    displacement data. A refused increment keeps the last accepted state:
    ``displacement`` is the committed state, ``candidate_displacement`` the last
    attempted Newton state, and ``accepted_load_factor`` the committed factor.
    ``stored_energy`` is ``sum w W(F)``; ``external_work`` integrates the body
    force and computed boundary tractions along the load path (trapezoid);
    for a hyperelastic equilibrium path the two agree up to collocation,
    quadrature and load-step consistency, reported as ``energy_work_defect``.
    On the ghost route ``ghost_displacement`` holds the committed ghost
    unknowns and ``ghost_extension_defect`` their extension evidence
    (``PreparedPointGhostLayer.extension_defect``, max over components).
    """

    displacement: Array
    candidate_displacement: Array
    deformation_gradient: Array
    first_piola: Array
    jacobian: Array
    stored_energy: Array
    external_work: Array
    energy_work_defect: Array
    residual_norm: Array
    accepted_load_factor: Array
    step_status: Array
    step_iterations: Array
    nonlinear: NonlinearResult
    ghost_displacement: Array | None
    ghost_extension_defect: Array | None
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(MechanicsStatus.ACCEPTED)


@final
class _HyperelasticLoad(StrictModule):
    body_force: Array
    boundary: Array
    factor: Array


@final
class MeshfreeHyperelasticPlan(StrictModule):
    """Finite-strain equilibrium ``-Div P(F) = f`` with ``F = I + grad u``.

    ``P`` is the law's first Piola stress (two-dimensional clouds are plane
    strain through the law's ``diag(F, 1)`` embedding). Neumann rows are
    reference tractions ``P N = t`` (Robin adds ``k u``); Dirichlet rows
    prescribe displacement; periodic rows are refused. Newton–Krylov uses the
    native prepared linearization, a deformation-admissibility trial guard
    (``J > 0``), and GMRES right-preconditioned by the block owner's prepared
    preconditioner of the reference tangent, frozen along the load path. The
    law's linearization at ``F = I`` is the Hooke tensor of the same Lamé
    parameters, so ``preconditioner`` selects the ``MeshfreeElasticityPlan``
    routes: ``"auxiliary"`` (default; nearly resolution-independent),
    ``"multigrid"`` or ``"ilu"`` (both stall under refinement, the ILU from a
    few hundred points). ``traction_route`` is the ``MeshfreeElasticityPlan``
    choice: on the default ghost route the equation unknowns are the cloud
    displacements then one ghost displacement per traction/Robin row; cloud
    rows carry ``-A(F) : D² u - f`` (``A = dP/dF``, the non-conservative form
    of ``-Div P`` for a homogeneous law) with ghost-extended stencils, and
    ghost rows carry ``(P N + k u - t) / δ_b`` (or the PDE at ``x_b`` for a
    component Dirichlet there). Its linearization at ``u = 0`` is the
    elasticity ghost system, whose prepared preconditioner is frozen.
    """

    system: PointBlockSystemPlan
    law: NeoHookeanLaw
    termination: NonlinearTermination
    load_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        boundary: PointBoundaryPlan,
        law: NeoHookeanLaw,
        /,
        *,
        load_steps: int = 1,
        termination: NonlinearTermination | None = None,
        preconditioner: MeshfreeElasticityPreconditioner = "auxiliary",
        stability: PointStabilityPolicy = "require-assessment",
        traction_route: MeshfreeTractionRoute = "ghost",
    ) -> None:
        _validated_boundary(discretization, boundary)
        if not isinstance(law, NeoHookeanLaw):
            raise TypeError("law must be a NeoHookeanLaw.")
        if (
            isinstance(load_steps, bool)
            or not isinstance(load_steps, int)
            or load_steps < 1
        ):
            raise ValueError("load_steps must be a positive integer.")
        if any(condition.kind == "periodic" for condition in boundary.conditions):
            raise ValueError("Finite-strain rows do not admit periodic identification.")
        dimension = discretization.spatial_dimension
        if dimension not in (2, 3):
            raise ValueError("Finite strain needs plane-strain 2-D or 3-D clouds.")
        termination_ = (
            NonlinearTermination(
                absolute_residual=1e-9, relative_residual=1e-10, maximum_steps=40
            )
            if termination is None
            else termination
        )
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination or None.")
        selected = parse(
            preconditioner, MeshfreeElasticityPreconditioner, "preconditioner"
        )
        candidates = _rigid_nullspace(dimension)
        system = PointBlockSystemPlan(
            discretization,
            boundary,
            components=_components(dimension),
            multigrid=candidates if selected == "multigrid" else None,
            auxiliary=candidates if selected == "auxiliary" else None,
            stability=stability,
            ghosts=_traction_ghosts(discretization, boundary, traction_route),
        )
        self.system = system
        self.law = law
        self.termination = termination_
        self.load_steps = load_steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-hyperelastic",
                "system": system.plan_id,
                "law": type(law).__name__,
                "shear": float(law.parameters.shear_modulus),
                "lambda": float(law.parameters.lame_lambda),
                "load_steps": load_steps,
            }
        )

    @property
    def reference_coefficients(self) -> Array:
        """Block coupling of the law's tangent at ``F = I``."""
        d = self.system.discretization.spatial_dimension
        tangent = self.law.evaluate(jnp.eye(d, dtype=jnp.float64)).tangent
        return jnp.transpose(tangent[:d, :d, :d, :d], (0, 2, 1, 3))

    def prepare(self, /) -> PreparedMeshfreeHyperelastic:
        return PreparedMeshfreeHyperelastic(self)


@final
class PreparedMeshfreeHyperelastic(StrictModule, NonTrainableState):
    """Prepared reference-tangent preconditioner, residual rows, and Newton policy."""

    plan: MeshfreeHyperelasticPlan
    method: NewtonKrylov

    def __init__(self, plan: MeshfreeHyperelasticPlan, /) -> None:
        if not isinstance(plan, MeshfreeHyperelasticPlan):
            raise TypeError("plan must be a MeshfreeHyperelasticPlan.")
        # The block owner's preparation of the reference tangent supplies the
        # frozen right preconditioner of every Newton system on the load path.
        reference = plan.system.prepare(plan.reference_coefficients)
        if not bool(reference.evidence.successful):
            raise ValueError(
                "The law's reference tangent failed symmetry or semidefiniteness verification."
            )
        prepared = reference.linear_solve.preconditioning_state
        if prepared is None:
            raise RuntimeError("The block owner prepared no reference preconditioner.")
        policy = reference.linear_solve.plan.policy
        self.plan = plan
        # Constant forcing: each Newton system is solved to 1e-6 relative.
        # With a resolution-independent preconditioner a decade costs a few
        # GMRES iterations, so Newton stays quadratic (2-3 steps per
        # increment) for the same work as Eisenstat–Walker forcing, whose
        # safeguarded decay from 0.5 alone takes 6-7 linearly convergent steps.
        self.method = NewtonKrylov(
            linear_policy=LinearSolvePolicy(
                policy.method,
                tolerance=TolerancePolicy(relative=1e-6, absolute=1e-12, max_steps=2000),
                preconditioning=PreconditioningPolicy(prepared.action),
                failure=FailurePolicy("status"),
                resources=policy.resources,
            ),
            forcing_policy=NewtonForcingPolicy("constant"),
        )

    @property
    def discretization(self) -> PreparedPointCloudDiscretization:
        return self.plan.system.discretization

    @property
    def unknown_count(self) -> int:
        """Equation unknowns per component: cloud points, then any ghosts."""
        ghosts = self.plan.system.ghosts
        return self.discretization.state_shape[0] if ghosts is None else ghosts.row_count

    def _unflatten(self, state: Array, /) -> Array:
        d = self.discretization.spatial_dimension
        return state.reshape((d, -1)).T

    def _cloud(self, state: Array, /) -> Array:
        """Cloud displacements ``(points, d)`` of flat equation unknowns."""
        return self._unflatten(state)[: self.discretization.state_shape[0]]

    def deformation_gradient(self, displacement: ArrayLike, /) -> Array:
        d = self.discretization.spatial_dimension
        return jnp.eye(d, dtype=jnp.float64) + displacement_gradient(
            self.discretization, displacement
        )

    def first_piola(self, displacement: ArrayLike, /) -> Array:
        """In-plane (or full) first Piola stress ``(points, d, d)``."""
        d = self.discretization.spatial_dimension
        response = self.plan.law.evaluate(self.deformation_gradient(displacement))
        return response.first_piola[:, :d, :d]

    def stored_energy(self, displacement: ArrayLike, /) -> Array:
        response = self.plan.law.evaluate(self.deformation_gradient(displacement))
        return jnp.sum(
            self.discretization.quadrature_weights * response.reference_energy_density
        )

    def boundary_data(
        self, *, boundary_values: Mapping[str, ArrayLike] | None = None
    ) -> Array:
        """Per-component boundary row data ``(points, d)`` (zero on bulk rows)."""
        boundary = self.plan.system.boundary
        return jnp.stack(
            [
                boundary.row_values(component, overrides=boundary_values)
                for component in range(len(self.plan.system.components))
            ],
            axis=1,
        )

    def _rows(self, unknowns: Array, load: _HyperelasticLoad, /) -> Array:
        ghosts = self.plan.system.ghosts
        if ghosts is None:
            return self._square_rows(unknowns, load)
        return self._ghost_rows(ghosts, unknowns, load)

    def _square_rows(self, displacement: Array, load: _HyperelasticLoad, /) -> Array:
        cloud = self.discretization
        d = cloud.spatial_dimension
        stress = self.first_piola(displacement)
        divergence = sum(
            cloud.partial_derivative(stress[:, :, j], axis=j) for j in range(d)
        )
        layouts = self.plan.system.layouts
        dirichlet = jnp.stack([layout.dirichlet for layout in layouts], axis=1)
        flux = jnp.stack([layout.flux for layout in layouts], axis=1)
        robin = jnp.stack([layout.robin for layout in layouts], axis=1)
        flux_normals = jnp.stack([layout.flux_normals[0] for layout in layouts], axis=1)
        conormal = ein.contract("naj,naj->na", stress, flux_normals)
        scaled = load.factor * load.boundary
        return jnp.where(
            dirichlet,
            displacement - scaled,
            jnp.where(
                flux,
                conormal + robin * displacement - scaled,
                -divergence - load.factor * load.body_force,
            ),
        )

    def _ghost_rows(
        self, ghosts: PreparedPointGhostLayer, unknowns: Array, load: _HyperelasticLoad, /
    ) -> Array:
        """Cloud PDE/Dirichlet rows, then one condition row per ghost and component."""
        d = self.discretization.spatial_dimension
        count = ghosts.cloud_count
        family = ghosts.family
        displacement = unknowns[:count]
        # Ghost-extended derivatives at the cloud rows: grad[n, b, k] = D_k u_b
        # and hessian[n, b, j, k] = D_jk u_b.
        gradient = _ghost_gradient(ghosts, unknowns)
        hessian = jnp.stack(
            [
                jnp.stack(
                    [family.apply(unknowns, _unit(d, j, k))[:count] for k in range(d)],
                    axis=2,
                )
                for j in range(d)
            ],
            axis=2,
        )
        identity = jnp.eye(d, dtype=jnp.float64)

        def piola(deformation: Array, /) -> Array:
            return self.plan.law.evaluate(deformation).first_piola[:, :d, :d]

        # Div P = sum_j dP/dF[d_j F], with d_j F_bk = D_jk u_b (homogeneous law).
        deformation = identity + gradient
        stress = piola(deformation)
        divergence = sum(
            jax.jvp(piola, (deformation,), (hessian[:, :, j, :],))[1][:, :, j]
            for j in range(d)
        )
        layouts = self.plan.system.layouts
        dirichlet = jnp.stack([layout.dirichlet for layout in layouts], axis=1)
        flux = jnp.stack([layout.flux for layout in layouts], axis=1)
        robin = jnp.stack([layout.robin for layout in layouts], axis=1)
        scaled = load.factor * load.boundary
        pde = -divergence - load.factor * load.body_force
        cloud = jnp.where(dirichlet, displacement - scaled, pde)
        rows = jnp.asarray(ghosts.plan.rows)
        # Each component's condition uses its own declared normal.
        flux_normals = jnp.stack([layout.flux_normals[0] for layout in layouts], axis=1)
        conormal = ein.contract("gaj,gaj->ga", stress[rows], flux_normals[rows])
        condition = (conormal + robin[rows] * displacement[rows] - scaled[rows]) / (
            ghosts.offsets[:, None]
        )
        return jnp.concatenate((cloud, jnp.where(flux[rows], condition, pde[rows])))

    def residual(
        self,
        unknowns: ArrayLike,
        body_force: ArrayLike,
        /,
        *,
        load_factor: ArrayLike = 1.0,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Solved equation rows ``(unknown_count, d)`` of ``(unknown_count, d)`` unknowns.

        Square route: ``-Div P - f``, ``P N (+k u) - t``, ``u - g`` on cloud
        values. Ghost route: cloud PDE/Dirichlet rows then the ghost rows.
        """
        load = self._load(body_force, load_factor, boundary_values)
        return self._rows(self._unknowns(unknowns), load)

    def _unknowns(self, unknowns: ArrayLike, /) -> Array:
        value = jnp.asarray(unknowns, dtype=jnp.float64)
        if value.shape != (self.unknown_count, self.discretization.spatial_dimension):
            raise ValueError(
                "unknowns must have shape (unknown_count, spatial dimension)."
            )
        return value

    def extend(self, displacement: ArrayLike, /) -> Array:
        """Equation unknowns of cloud displacements: ghosts from the extension law.

        Each ghost value is the one-sided reconstruction ``(E u)(x_g)`` from
        cloud points (``PreparedPointGhostLayer.extension``).
        """
        cloud = self.discretization
        value = jnp.asarray(displacement, dtype=jnp.float64)
        if value.shape != (cloud.state_shape[0], cloud.spatial_dimension):
            raise ValueError("displacement must have shape (points, spatial dimension).")
        ghosts = self.plan.system.ghosts
        if ghosts is None:
            return value
        # The extension family has one row per ghost.
        reconstructed = ghosts.extension.apply(value, (0,) * cloud.spatial_dimension)
        return jnp.concatenate((value, reconstructed))

    def _load(
        self,
        body_force: ArrayLike,
        load_factor: ArrayLike,
        boundary_values: Mapping[str, ArrayLike] | None,
        /,
    ) -> _HyperelasticLoad:
        cloud = self.discretization
        force = jnp.asarray(body_force, dtype=jnp.float64)
        if force.shape != (cloud.state_shape[0], cloud.spatial_dimension):
            raise ValueError("body_force must have shape (points, spatial dimension).")
        return _HyperelasticLoad(
            force,
            self.boundary_data(boundary_values=boundary_values),
            jnp.asarray(load_factor, dtype=jnp.float64),
        )

    def _flat_residual(self, state: Array, load: _HyperelasticLoad, /) -> Array:
        return self._rows(self._unflatten(state), load).T.reshape((-1,))

    def _admissible(self, state: Array, load: _HyperelasticLoad, /) -> Array:
        del load
        response = self.plan.law.evaluate(self.deformation_gradient(self._cloud(state)))
        return jnp.all(response.admissible)

    def problem(self) -> NonlinearSystemProblem:
        """Native residual problem on the flat component-major state space."""
        space = self.plan.system.solve_space
        return NonlinearSystemProblem(
            self._flat_residual,
            state_space=space,
            residual_space=space,
            trial_validity=self._admissible,
            trial_validity_id=f"{self.plan.plan_id}:positive-jacobian",
            problem_id=f"{self.plan.plan_id}:equilibrium",
        )

    def linearize(
        self,
        unknowns: ArrayLike,
        body_force: ArrayLike,
        /,
        *,
        load_factor: ArrayLike = 1.0,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> PreparedLinearization:
        """Native tangent (JVP) and adjoint (VJP) of the flat residual rows."""
        load = self._load(body_force, load_factor, boundary_values)
        state = self._unknowns(unknowns).T.reshape((-1,))
        space = self.plan.system.solve_space
        return prepare_linearization(
            lambda flat: self._flat_residual(flat, load),
            state,
            source=space,
            target=space,
            linearization_id=f"{self.plan.plan_id}:tangent",
        )

    def _solved_deformation(self, state: Array, /) -> Array:
        """``F = I + grad u`` at the cloud from flat unknowns, by the solved stencils.

        The ghost route differentiates with its ghost-extended stencils, so the
        reported stress and tractions are those its equations enforce.
        """
        d = self.discretization.spatial_dimension
        ghosts = self.plan.system.ghosts
        if ghosts is None:
            return self.deformation_gradient(self._cloud(state))
        return jnp.eye(d, dtype=jnp.float64) + _ghost_gradient(
            ghosts, self._unflatten(state)
        )

    def _boundary_traction(self, state: Array, /) -> Array:
        cloud = self.discretization
        d = cloud.spatial_dimension
        response = self.plan.law.evaluate(self._solved_deformation(state))
        stress = response.first_piola[:, :d, :d]
        mask, normals, _ = _boundary_rows(cloud)
        return jnp.where(mask[:, None], ein.contract("naj,nj->na", stress, normals), 0.0)

    def solve(
        self,
        body_force: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
        initial: ArrayLike | None = None,
    ) -> MeshfreeHyperelasticResult:
        cloud = self.discretization
        d = cloud.spatial_dimension
        count = cloud.state_shape[0]
        load = self._load(body_force, 1.0, boundary_values)
        start = (
            jnp.zeros((count, d), dtype=jnp.float64)
            if initial is None
            else jnp.asarray(initial, dtype=jnp.float64)
        )
        if start.shape != (count, d):
            raise ValueError("initial must have shape (points, spatial dimension).")
        problem = self.problem()
        weights = cloud.quadrature_weights
        _, _, boundary_weights = _boundary_rows(cloud)
        steps = self.plan.load_steps

        def power(state: Array, factor: Array, /) -> tuple[Array, Array]:
            # Body-force and boundary-traction resultants paired later with
            # increments; both are linear in the displacement increment.
            return factor * load.body_force, self._boundary_traction(state)

        def increment(
            carry: tuple[Array, Array, Array, Array, Array], factor: Array
        ) -> tuple[tuple[Array, Array, Array, Array, Array], tuple[Array, Array]]:
            state, committed, work, live, _ = carry
            step_load = _HyperelasticLoad(load.body_force, load.boundary, factor)
            result = root(
                problem,
                state,
                method=self.method,
                termination=self.plan.termination,
                args=step_load,
            )
            accepted = live & result.successful & jnp.all(jnp.isfinite(result.state))
            previous = self._cloud(state)
            current = self._cloud(result.state)
            force_old, traction_old = power(state, committed)
            force_new, traction_new = power(result.state, factor)
            delta = current - previous
            body_work = 0.5 * jnp.sum(weights[:, None] * (force_old + force_new) * delta)
            surface_work = 0.5 * _boundary_integral(
                boundary_weights,
                jnp.sum((traction_old + traction_new) * delta, axis=1),
            )
            next_carry = (
                jnp.where(accepted, result.state, state),
                jnp.where(accepted, factor, committed),
                jnp.where(accepted, work + body_work + surface_work, work),
                accepted,
                result.state,
            )
            return next_carry, (result.status, result.diagnostics.iterations)

        flat = self.extend(start).T.reshape((-1,))
        factors = jnp.arange(1, steps + 1, dtype=jnp.float64) / steps
        initial_energy = jnp.sum(
            weights
            * self.plan.law.evaluate(
                self._solved_deformation(flat)
            ).reference_energy_density
        )
        (state, committed, work, live, candidate), (statuses, iterations) = jax.lax.scan(
            increment,
            (flat, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(True), flat),
            factors,
        )
        final = root(
            problem,
            state,
            method=self.method,
            termination=self.plan.termination,
            args=_HyperelasticLoad(load.body_force, load.boundary, committed),
        )
        displacement = self._cloud(state)
        response = self.plan.law.evaluate(self._solved_deformation(state))
        energy = jnp.sum(weights * response.reference_energy_density) - initial_energy
        residual_norm = jnp.linalg.norm(
            self._flat_residual(
                state, _HyperelasticLoad(load.body_force, load.boundary, committed)
            )
        )
        finite = jnp.all(jnp.isfinite(displacement)) & jnp.isfinite(residual_norm)
        status = _status(
            finite,
            jnp.all(response.admissible),
            live & (committed == 1.0),
            final.successful,
        )
        ghosts = self.plan.system.ghosts
        unknowns = self._unflatten(state)
        ghost_displacement = None if ghosts is None else unknowns[count:]
        ghost_defect = (
            None
            if ghosts is None
            else jnp.max(
                jnp.stack([ghosts.extension_defect(unknowns[:, a]) for a in range(d)])
            )
        )
        return MeshfreeHyperelasticResult(
            displacement=displacement,
            candidate_displacement=self._cloud(candidate),
            deformation_gradient=response.kinematics.deformation_gradient[:, :d, :d],
            first_piola=response.first_piola[:, :d, :d],
            jacobian=response.kinematics.jacobian,
            stored_energy=energy,
            external_work=work,
            energy_work_defect=jnp.abs(energy - work)
            / jnp.maximum(jnp.abs(work), jnp.finfo(jnp.float64).tiny),
            residual_norm=residual_norm,
            accepted_load_factor=committed,
            step_status=statuses.astype(jnp.int32),
            step_iterations=iterations.astype(jnp.int32),
            nonlinear=final,
            ghost_displacement=ghost_displacement,
            ghost_extension_defect=ghost_defect,
            status=status,
        )


# --------------------------------------------------------------------------
# Generalized Stokes / Herrmann mixed displacement–pressure
# --------------------------------------------------------------------------


@final
class MeshfreeMixedResult(StrictModule):
    """Mixed vector/pressure solution and inf-sup/consistency evidence.

    ``field`` is displacement (near-incompressible elasticity) or velocity
    (Stokes). ``pressure`` is the physical pressure; for incompressible gauged
    plans it is reported mean-zero (quadrature weights). Gauged (all-Dirichlet)
    plans close the mean pressure by the discrete volume balance
    ``kappa sum_i V_i p_i = -sum_b w_b u_b . n_b``, and
    ``compatibility_residual`` is the uniform continuity source
    ``|c - kappa p_bar|`` the collocated rows leave beyond it (``|c|`` for
    ``kappa = 0``; zero with traction rows). ``volumetric_defect`` is
    ``||div u + kappa p||`` (weighted RMS); ``pressure_oscillation`` is the
    checkerboard index ``h^2 ||lap_h p|| / ||p - mean p||``, O(h^2)-small for a
    smooth pressure and O(1) for spurious modes.
    """

    field: Array
    pressure: Array
    strain: Array
    stress: Array
    boundary_traction: Array
    divergence: Array
    volumetric_defect: Array
    momentum_residual_norm: Array
    pressure_residual_norm: Array
    compatibility_residual: Array
    pressure_oscillation: Array
    residual_tolerance: Array
    linear: LinearSolveResult
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(MechanicsStatus.ACCEPTED)


@final
class MeshfreeGeneralizedStokesPlan(StrictModule):
    """Generalized Stokes / Herrmann mixed form on one collocated cloud.

    ``-div(2 mu eps(u)) + grad p = f`` with ``div u + kappa p = 0``:
    incompressible Stokes flow for ``compressibility = 0`` (``mu`` = viscosity)
    and near-incompressible elasticity for ``kappa = 1 / lambda``. Equal-order
    collocation is stabilized by two terms that both vanish on the exact
    solution. Pressure rows read
    ``div u + kappa p + V^-1 G^T V tau R - tau ((1 + 2 mu kappa) L p - div f) = 0``
    with ``R = -div(2 mu eps(u)) + grad p - f`` the nodal momentum residual at
    bulk and Dirichlet nodes, ``G`` the nodal gradient, ``L`` the compact
    Laplacian, ``V`` the point volumes, and ``tau = stabilization h^2 / mu``
    (``h`` from point volumes; ``tau = 0`` at traction nodes, whose pressure
    the traction rows carry). The weak-form term's pressure part
    ``V^-1 G^T V tau G`` is volume-self-adjoint positive semidefinite and
    controls boundary and corner pressure, where the compact term is
    anti-dissipative; the compact term (consistent through
    ``div R = (1 + 2 mu kappa) lap p - div f``) damps the high-frequency
    pressure that ``G^T G`` barely sees. Neumann
    rows are tractions ``(2 mu eps - p I) n = t``. Only a traction component
    with a nonzero owned normal component anchors pressure. Otherwise constant
    pressure leaves momentum unchanged and enters the pressure rows only as
    ``kappa p_bar``. The solve then splits ``p = p_tilde + p_bar``
    with ``p_tilde`` pinned at the first bulk row and a bordered uniform
    continuity source ``c``. The collocated rows fix ``c`` only up to their
    O(h^q) compatibility defect, which ``1 / kappa`` would amplify, so the
    mean pressure is closed by the consistent discrete volume balance
    ``kappa sum_i V_i p_i = -sum_b w_b u_b . n_b`` (boundary quadrature
    required), and the rest of ``c`` is reported as the compatibility
    residual. For ``kappa = 0`` pressure is reported mean-zero. The saddle
    solve is native GMRES with an upper block-factorization preconditioner:
    ILU on the viscous block and ILU on the Schur model
    ``(1/mu + kappa) I + V^-1 G^T V tau G - (1 + 2 mu kappa) tau L``.
    """

    system: PointBlockSystemPlan
    shear_modulus: float = eqx.field(static=True)
    compressibility: float = eqx.field(static=True)
    stabilization: float = eqx.field(static=True)
    gauge_row: int | None = eqx.field(static=True)
    tolerance: TolerancePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        boundary: PointBoundaryPlan,
        /,
        *,
        shear_modulus: float,
        compressibility: float = 0.0,
        stabilization: float = 0.1,
        tolerance: TolerancePolicy | None = None,
    ) -> None:
        _validated_boundary(discretization, boundary)
        mu, kappa, tau = (
            float(shear_modulus),
            float(compressibility),
            float(stabilization),
        )
        if not np.isfinite(mu) or mu <= 0.0:
            raise ValueError("shear_modulus must be finite and positive.")
        if not np.isfinite(kappa) or kappa < 0.0:
            raise ValueError("compressibility must be finite and nonnegative.")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError(
                "stabilization must be positive; equal-order collocation is not inf-sup stable."
            )
        if any(condition.kind == "periodic" for condition in boundary.conditions):
            raise ValueError("Mixed rows do not admit periodic identification.")
        dimension = discretization.spatial_dimension
        system = PointBlockSystemPlan(
            discretization, boundary, components=_components(dimension)
        )
        pressure_anchored = any(
            np.any(
                np.where(
                    np.asarray(layout.flux),
                    np.asarray(layout.flux_normals[0])[:, axis],
                    0.0,
                )
                != 0.0
            )
            for axis, layout in enumerate(system.layouts)
        )
        bulk = np.all(
            np.stack([np.asarray(layout.bulk) for layout in system.layouts]), axis=0
        )
        gauge = None if pressure_anchored else int(np.flatnonzero(bulk)[0])
        if (
            gauge is not None
            and kappa > 0.0
            and discretization.plan.boundary_quadrature_weights is None
        ):
            raise ValueError(
                "A clamped compressible mixed body closes its mean pressure by the "
                "boundary volume flux, which needs declared boundary quadrature weights."
            )
        tolerance_ = (
            TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=1000)
            if tolerance is None
            else tolerance
        )
        if not isinstance(tolerance_, TolerancePolicy):
            raise TypeError("tolerance must be a TolerancePolicy or None.")
        self.system = system
        self.shear_modulus = mu
        self.compressibility = kappa
        self.stabilization = tau
        self.gauge_row = gauge
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-generalized-stokes",
                "system": system.plan_id,
                "shear_modulus": mu,
                "compressibility": kappa,
                "stabilization": tau,
                "gauge_row": gauge,
            }
        )

    @property
    def pressure_space(self) -> ArraySpace:
        count = self.system.discretization.state_shape[0]
        return ArraySpace(
            (count,), dtype=jnp.float64, space_id=f"{self.plan_id}:pressure"
        )

    @property
    def space(self) -> BlockSpace:
        """Physical ``(field, pressure)`` space of ``physical_operator``."""
        return BlockSpace(
            (self.system.solve_space, self.pressure_space),
            names=("field", "pressure"),
            space_id=f"{self.plan_id}:mixed",
        )

    @property
    def solve_space(self) -> BlockSpace:
        """Gauged solve space: pinned pressure plus the bordered uniform source ``c``."""
        count = self.system.discretization.state_shape[0]
        border = 0 if self.gauge_row is None else 1
        return BlockSpace(
            (
                self.system.solve_space,
                ArraySpace(
                    (count + border,),
                    dtype=jnp.float64,
                    space_id=f"{self.plan_id}:gauged-pressure",
                ),
            ),
            names=("field", "pressure"),
            space_id=f"{self.plan_id}:gauged-mixed",
        )

    def prepare(self, /) -> PreparedMeshfreeGeneralizedStokes:
        return PreparedMeshfreeGeneralizedStokes(self)


def _stencil_relation(
    family: PointDerivativeFamily, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host valid routes ``(row, source, route position)`` of the cloud stencils."""
    valid = np.asarray(family.relation.valid)
    rows, routes = np.nonzero(valid)
    sources = np.asarray(family.relation.source_indices)[rows, routes]
    return rows.astype(np.int32), sources.astype(np.int32), routes.astype(np.int32)


def _coupling_operator(
    rows: np.ndarray,
    sources: np.ndarray,
    values: Array,
    source: ArraySpace,
    target: ArraySpace,
    operator_id: str,
    /,
) -> SparseCoordinateOperator:
    return SparseCoordinateOperator(
        EdgeRelation(sources, rows, source_size=source.size, target_size=target.size),
        values,
        source=source,
        target=target,
        operator_id=operator_id,
    )


def _gradient_coupling(
    system: PointBlockSystemPlan,
    rows: np.ndarray,
    sources: np.ndarray,
    first: list[Array],
    pressure: ArraySpace,
    operator_id: str,
    /,
) -> SparseCoordinateOperator:
    """``grad p`` on bulk rows, ``-p n`` on traction rows, nothing on Dirichlet rows."""
    count = system.discretization.state_shape[0]
    nodes = np.arange(count, dtype=np.int32)
    targets, columns, values = [], [], []
    for a, layout in enumerate(system.layouts):
        targets += [a * count + rows, a * count + nodes]
        columns += [sources, nodes]
        values += [
            jnp.asarray(layout.bulk, dtype=jnp.float64)[rows] * first[a],
            -jnp.where(layout.flux, layout.flux_normals[0][:, a], 0.0),
        ]
    return _coupling_operator(
        np.concatenate(targets).astype(np.int32),
        np.concatenate(columns).astype(np.int32),
        jnp.concatenate(values),
        pressure,
        system.solve_space,
        operator_id,
    )


def _residual_stabilization(
    family: PointDerivativeFamily,
    relation: tuple[np.ndarray, np.ndarray, np.ndarray],
    first: list[Array],
    plan: MeshfreeGeneralizedStokesPlan,
    velocity: ArraySpace,
    pressure: ArraySpace,
    /,
) -> tuple[
    SparseCoordinateOperator, Array, AbstractLinearOperator, AbstractLinearOperator
]:
    """``(T, tau, T A, T G - (1 + 2 mu kappa) tau L)`` of the pressure stabilization.

    Pressure rows carry ``T R - tau ((1 + 2 mu kappa) L p - div f)``. The first
    term is weak-form: ``R = A u + G p - f`` is the nodal momentum residual
    (``A = -div(2 mu eps)``, ``G`` the nodal gradient) at bulk and Dirichlet
    nodes, corners included, and ``T = V^-1 G^T V tau`` is the volume adjoint of
    ``G``. It vanishes with the residual, and ``T G`` is volume-self-adjoint
    positive semidefinite with constants in its kernel, so it controls
    boundary and corner pressure. The second term is the collocated divergence
    of the momentum residual with the compact Laplacian ``L``, consistent by
    ``div R = (1 + 2 mu kappa) lap p - div f`` for constant ``mu``. It damps the
    high-frequency pressure that the wide ``G^T G`` barely sees on collocated
    stencils (alone, ``T R`` left the bounded-Stokes oscillation index at 2.3).
    The compact term alone is anti-dissipative on one-sided corner stencils
    (positive diagonal): ``kappa I - tau L`` was singular at ``kappa ~ 1e-2``
    for every ``h``. With ``T G`` it has no positive critical ``kappa``.
    ``tau = stabilization h^2 / mu`` is zero at traction nodes, whose pressure
    the ``-p n`` rows already carry; a one-sided residual there made the
    operator singular at ``stabilization ~ 0.1``.
    """
    rows, sources, routes = relation
    cloud = plan.system.discretization
    d, count = cloud.spatial_dimension, cloud.state_shape[0]
    mu = plan.shear_modulus
    volumes = cloud.quadrature_weights
    traction = np.any(
        np.stack([np.asarray(layout.flux) for layout in plan.system.layouts]), axis=0
    )
    tau = jnp.where(traction, 0.0, plan.stabilization * volumes ** (2.0 / d) / mu)
    stacked = np.concatenate([a * count + rows for a in range(d)]).astype(np.int32)
    repeated = np.concatenate([sources] * d).astype(np.int32)
    gradient = _coupling_operator(
        stacked,
        repeated,
        jnp.concatenate(first),
        pressure,
        velocity,
        f"{plan.plan_id}:nodal-gradient",
    )
    laplace = sum(family.weights_for(_unit(d, c, c))[rows, routes] for c in range(d))
    pairs = [(a, b) for a in range(d) for b in range(d)]
    momentum = _coupling_operator(
        np.concatenate([a * count + rows for a, _ in pairs]).astype(np.int32),
        np.concatenate([b * count + sources for _, b in pairs]).astype(np.int32),
        jnp.concatenate(
            [
                -mu
                * (
                    family.weights_for(_unit(d, a, b))[rows, routes]
                    + (laplace if a == b else 0.0)
                )
                for a, b in pairs
            ]
        ),
        velocity,
        velocity,
        f"{plan.plan_id}:nodal-momentum",
    )
    weight = (volumes * tau)[rows] / volumes[sources]
    adjoint = _coupling_operator(
        repeated,
        stacked,
        jnp.concatenate([weight * w for w in first]),
        velocity,
        pressure,
        f"{plan.plan_id}:residual-stabilization",
    )
    compact = _coupling_operator(
        rows,
        sources,
        -(tau * (1.0 + 2.0 * mu * plan.compressibility))[rows] * laplace,
        pressure,
        pressure,
        f"{plan.plan_id}:compact-stabilization",
    )
    return adjoint, tau, adjoint @ momentum, adjoint @ gradient + compact


def _gauge_maps(
    plan: MeshfreeGeneralizedStokesPlan, pressure: ArraySpace, gauged: ArraySpace, /
) -> tuple[
    SparseCoordinateOperator, SparseCoordinateOperator, SparseCoordinateOperator | None
]:
    """Row map, column restriction, and border of the gauged pressure rows.

    The pinned row's continuity equation moves to the border row and the
    bordered uniform source ``c`` enters every continuity row with unit
    weight, so every diagonal (also of the Schur model) is nonzero and
    ILU-admissible; the pinned row reads ``p_tilde[gauge] = 0``.
    """
    count = pressure.size
    gauge = plan.gauge_row
    nodes = np.arange(count, dtype=np.int32)
    moved = (
        nodes
        if gauge is None
        else np.where(nodes == gauge, count, nodes).astype(np.int32)
    )
    ones = jnp.ones((count,), dtype=jnp.float64)
    row_map = _coupling_operator(
        moved, nodes, ones, pressure, gauged, f"{plan.plan_id}:gauge-rows"
    )
    restriction = _coupling_operator(
        nodes, nodes, ones, gauged, pressure, f"{plan.plan_id}:gauge-columns"
    )
    if gauge is None:
        return row_map, restriction, None
    continuity = np.arange(count + 1, dtype=np.int32)
    continuity = continuity[continuity != gauge]
    border = _coupling_operator(
        np.concatenate((continuity, np.asarray([gauge], dtype=np.int32))),
        np.concatenate(
            (
                np.full(continuity.shape, count, dtype=np.int32),
                np.asarray([gauge], dtype=np.int32),
            )
        ),
        jnp.ones((count + 1,), dtype=jnp.float64),
        gauged,
        gauged,
        f"{plan.plan_id}:gauge-border",
    )
    return row_map, restriction, border


@final
class PreparedMeshfreeGeneralizedStokes(StrictModule, NonTrainableState):
    """Prepared mixed operator, block preconditioner, and native saddle solve."""

    plan: MeshfreeGeneralizedStokesPlan
    physical_operator: BlockLinearOperator
    operator: BlockLinearOperator
    coupling_evidence: PointCouplingEvidence
    residual_stabilization: SparseCoordinateOperator
    stabilization_weights: Array
    linear_solve: PreparedLinearSolve

    def __init__(self, plan: MeshfreeGeneralizedStokesPlan, /) -> None:
        if not isinstance(plan, MeshfreeGeneralizedStokesPlan):
            raise TypeError("plan must be a MeshfreeGeneralizedStokesPlan.")
        system = plan.system
        cloud = system.discretization
        policy = system.assembly_policy
        d, count = cloud.spatial_dimension, cloud.state_shape[0]
        mu, kappa = plan.shear_modulus, plan.compressibility
        eye = jnp.eye(d, dtype=jnp.float64)
        coefficients = mu * (
            eye[:, :, None, None] * eye[None, None, :, :]
            + eye[:, None, None, :] * eye[None, :, :, None]
        )
        physical, evidence = system.physical_operator(coefficients)
        viscous = system.flat_operator(assemble_sparse(physical, policy))
        family = discretization_family(cloud)
        rows, sources, routes = _stencil_relation(family)
        velocity, pressure = system.solve_space, plan.pressure_space
        gauged = plan.solve_space.spaces[1]
        if not isinstance(gauged, ArraySpace):
            raise TypeError("The Stokes pressure gauge space must be an ArraySpace.")
        first = [family.weights_for(_unit(d, a))[rows, routes] for a in range(d)]
        adjoint, tau, field_stabilization, pressure_stabilization = (
            _residual_stabilization(
                family, (rows, sources, routes), first, plan, velocity, pressure
            )
        )
        divergence = _coupling_operator(
            np.concatenate([rows] * d),
            np.concatenate([a * count + sources for a in range(d)]),
            jnp.concatenate(first),
            velocity,
            pressure,
            f"{plan.plan_id}:divergence",
        )
        constraint_field = assemble_sparse(divergence + field_stabilization, policy)
        stabilization = assemble_sparse(pressure_stabilization, policy)

        def diagonal(value: float, /) -> DiagonalLinearOperator:
            return DiagonalLinearOperator(
                jnp.full((count,), value, dtype=jnp.float64), space=pressure
            )

        constraint_pressure = assemble_sparse(stabilization + diagonal(kappa), policy)
        self.plan = plan
        self.physical_operator = BlockLinearOperator(
            (
                (
                    viscous,
                    _gradient_coupling(
                        system, rows, sources, first, pressure, f"{plan.plan_id}:gradient"
                    ),
                ),
                (constraint_field, constraint_pressure),
            ),
            source=plan.space,
            target=plan.space,
            operator_id=f"{plan.plan_id}:physical",
        )
        row_map, restriction, border = _gauge_maps(plan, pressure, gauged)

        def gauged_pressure(block: AbstractLinearOperator, /) -> AbstractLinearOperator:
            operator = row_map @ block @ restriction
            return assemble_sparse(
                operator if border is None else operator + border, policy
            )

        mixed = BlockLinearOperator(
            (
                (
                    viscous,
                    _gradient_coupling(
                        system,
                        rows,
                        sources,
                        first,
                        gauged,
                        f"{plan.plan_id}:gauged-gradient",
                    ),
                ),
                (
                    assemble_sparse(row_map @ constraint_field, policy),
                    gauged_pressure(constraint_pressure),
                ),
            ),
            source=plan.solve_space,
            target=plan.solve_space,
            operator_id=f"{plan.plan_id}:operator",
        )
        schur = gauged_pressure(stabilization + diagonal(1.0 / mu + kappa))
        # Near-incompressible bodies keep a few Schur modes that the
        # (1/mu + kappa) model misses; a short restart discards them and
        # stagnates, so the Krylov space is kept.
        solve_policy = LinearSolvePolicy(
            GMRES(restart=min(400, (d + 1) * count + 1)),
            tolerance=plan.tolerance,
            preconditioning=PreconditioningPolicy(
                BlockFactorizationPreconditionerBuilder(
                    ILUPreconditionerBuilder(),
                    ILUPreconditionerBuilder(),
                    "upper",
                    schur_setup_operator=schur,
                )
            ),
            failure=FailurePolicy("status"),
        )
        self.operator = mixed
        self.coupling_evidence = evidence
        self.residual_stabilization = adjoint
        self.stabilization_weights = tau
        self.linear_solve = prepare(
            LinearSystem(mixed, problem_id=f"{plan.plan_id}:saddle"), solve_policy
        )

    @property
    def discretization(self) -> PreparedPointCloudDiscretization:
        return self.plan.system.discretization

    def _flatten(self, field: Array, /) -> Array:
        return field.T.reshape((-1,))

    def _unflatten(self, flat: Array, /) -> Array:
        return flat.reshape((self.discretization.spatial_dimension, -1)).T

    def _divergence(self, field: Array, /) -> Array:
        return jnp.trace(
            displacement_gradient(self.discretization, field), axis1=1, axis2=2
        )

    def physical_rhs(
        self,
        body_force: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> tuple[Array, Array]:
        """``(field rows (d*N,), pressure rows (N,))`` of the un-gauged equations."""
        plan = self.plan
        cloud = self.discretization
        force = jnp.asarray(body_force, dtype=jnp.float64)
        if force.shape != (cloud.state_shape[0], cloud.spatial_dimension):
            raise ValueError("body_force must have shape (points, spatial dimension).")
        boundary = plan.system.boundary
        columns = [
            jnp.where(
                layout.bulk,
                force[:, a],
                boundary.row_values(a, overrides=boundary_values),
            )
            for a, layout in enumerate(plan.system.layouts)
        ]
        field = jnp.stack(columns, axis=1)
        # Stabilization rows T R - tau ((1 + 2 mu kappa) L p - div f) carry
        # T f - tau div f on the right-hand side.
        pressure = self.residual_stabilization.mv(
            self._flatten(force)
        ) - self.stabilization_weights * self._divergence(force)
        return self._flatten(field), pressure

    def residual(
        self,
        field: ArrayLike,
        pressure: ArrayLike,
        body_force: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> tuple[Array, Array]:
        """Physical ``(momentum (N,d), pressure (N,))`` residual rows (traction rows pointwise)."""
        value = jnp.asarray(field, dtype=jnp.float64)
        p = jnp.asarray(pressure, dtype=jnp.float64)
        momentum, constraint = self.physical_operator.mv((self._flatten(value), p))
        rhs_field, rhs_pressure = self.physical_rhs(
            body_force, boundary_values=boundary_values
        )
        return self._unflatten(momentum - rhs_field), constraint - rhs_pressure

    def strain(self, field: ArrayLike, /) -> Array:
        return _symmetric(displacement_gradient(self.discretization, field))

    def stress(self, field: ArrayLike, pressure: ArrayLike, /) -> Array:
        d = self.discretization.spatial_dimension
        p = jnp.asarray(pressure, dtype=jnp.float64)
        return 2.0 * self.plan.shear_modulus * self.strain(field) - p[
            :, None, None
        ] * jnp.eye(d, dtype=jnp.float64)

    def solve(
        self,
        body_force: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> MeshfreeMixedResult:
        plan = self.plan
        cloud = self.discretization
        rhs_field, rhs_pressure = self.physical_rhs(
            body_force, boundary_values=boundary_values
        )
        weights = cloud.quadrature_weights
        volume = jnp.sum(weights)
        count = rhs_pressure.shape[0]
        kappa = plan.compressibility
        gauge = plan.gauge_row
        mask, normals, boundary_weights = _boundary_rows(cloud)
        if gauge is None:
            linear = solve(self.linear_solve, (rhs_field, rhs_pressure))
            flat, p = linear.value
            pinned, border, source = p, jnp.asarray(0.0), jnp.asarray(0.0)
        else:
            gauged_rhs = jnp.concatenate(
                (rhs_pressure.at[gauge].set(0.0), rhs_pressure[gauge][None])
            )
            linear = solve(self.linear_solve, (rhs_field, gauged_rhs))
            flat, solved = linear.value
            pinned, border = solved[:count], solved[count]
            if kappa > 0.0:
                # The collocated rows fix the uniform source only up to their
                # O(h^q) compatibility defect, which 1/kappa would amplify. The
                # mean pressure follows the discrete volume balance
                # kappa sum V p = -sum_b w_b u_b . n_b instead (Gauss with the
                # declared boundary quadrature), and c = kappa p_bar.
                outflow = _boundary_integral(
                    boundary_weights,
                    jnp.where(
                        mask, jnp.sum(self._unflatten(flat) * normals, axis=1), 0.0
                    ),
                )
                source = -(outflow + kappa * jnp.sum(weights * pinned)) / volume
                p = pinned + source / kappa
            else:
                source = jnp.asarray(0.0)
                p = pinned - jnp.sum(weights * pinned) / volume
        field = self._unflatten(flat)
        # Residuals use the pinned representative: constants leave momentum
        # unchanged (no pressure-anchoring traction component) and enter continuity as
        # c, so p_bar never cancels in floating point. The part of c beyond
        # kappa p_bar is the reported compatibility residual.
        momentum, constraint = self.physical_operator.mv((flat, pinned))
        momentum_residual = momentum - rhs_field
        pressure_residual = constraint + border - rhs_pressure
        compatibility = jnp.abs(border - source)
        momentum_norm = jnp.linalg.norm(momentum_residual)
        pressure_norm = jnp.linalg.norm(pressure_residual)
        rhs_norm = jnp.sqrt(jnp.sum(rhs_field**2) + jnp.sum(rhs_pressure**2))
        threshold = plan.tolerance.absolute + plan.tolerance.relative * rhs_norm
        divergence = self._divergence(field)
        mean = jnp.sum(weights * p) / jnp.sum(weights)
        spread = jnp.sqrt(jnp.sum(weights * (p - mean) ** 2))
        laplace = cloud.laplacian(p)
        oscillation = (
            jnp.mean(cloud.quadrature_weights ** (2.0 / cloud.spatial_dimension))
            * jnp.sqrt(jnp.sum(weights * laplace**2))
            / jnp.maximum(spread, jnp.finfo(jnp.float64).tiny)
        )
        stress = self.stress(field, p)
        finite = (
            jnp.all(jnp.isfinite(flat))
            & jnp.all(jnp.isfinite(p))
            & jnp.isfinite(momentum_norm)
            & jnp.isfinite(pressure_norm)
        )
        status = _status(
            finite,
            jnp.asarray(True),
            linear.successful & self.coupling_evidence.successful,
            (momentum_norm <= 10.0 * threshold) & (pressure_norm <= 10.0 * threshold),
        )
        return MeshfreeMixedResult(
            field=field,
            pressure=p,
            strain=self.strain(field),
            stress=stress,
            boundary_traction=jnp.where(
                mask[:, None], ein.contract("nij,nj->ni", stress, normals), 0.0
            ),
            divergence=divergence,
            volumetric_defect=jnp.sqrt(
                jnp.sum(weights * (divergence + plan.compressibility * p) ** 2)
                / jnp.sum(weights)
            ),
            momentum_residual_norm=momentum_norm,
            pressure_residual_norm=pressure_norm,
            compatibility_residual=compatibility,
            pressure_oscillation=oscillation,
            residual_tolerance=threshold,
            linear=linear,
            status=status,
        )


__all__ = [
    "displacement_gradient",
    "MechanicsStatus",
    "MeshfreeElasticityPlan",
    "MeshfreeElasticityPreconditioner",
    "MeshfreeTractionRoute",
    "MeshfreeElasticityResult",
    "MeshfreeGeneralizedStokesPlan",
    "MeshfreeHyperelasticPlan",
    "MeshfreeHyperelasticResult",
    "MeshfreeMixedResult",
    "PreparedMeshfreeElasticity",
    "PreparedMeshfreeGeneralizedStokes",
    "PreparedMeshfreeHyperelastic",
]
