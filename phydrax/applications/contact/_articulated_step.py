#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native fixed-base articulated hard-contact step with coupled joint stops.

One requested interval ``[t0, t0 + H]`` is advanced by the declared
first-order, position-level, inelastic hard-contact discretization (zero
restitution, zero compliance, isotropic static/dynamic Coulomb friction) with
unilateral hinge/prismatic stops in the same generalized response.

For one attempted segment of duration ``h`` at its provisional source
``(q0, v0)``:

* ``M0 = M(q0)`` is prepared once by the native smooth predictor; ``a0`` is the
  native ABA acceleration for the held source effort; ``v_free = v0 + h a0``.
* The binding lifted path is ``q(s) = q0 + s h v1`` with
  ``v1 = v_free + M0⁻¹ G(q1)* p`` and ``q1 = q0 + h v1``.
* Each contact/stop row obeys ``p_n >= 0``, ``g(q1)/h >= 0`` and
  ``p_n g(q1)/h = 0`` within dimensional tolerances; tangential friction acts
  on the physical tangential velocity ``G_t(q1) v1``.

The coupled root is solved by the native
:class:`~phydrax.nonlinear.FixedPointIteration` on the physical velocity
``v1``: each evaluation solves the frozen-route cone program relinearized at
``q_hat = q0 + h v1`` (normal bias ``b_n = (g(q_hat) - G_n(q_hat)(q_hat -
q0)) / h``) from the same history warm start and returns
``v_free + M0⁻¹ G(q_hat)* p``. Its fixed point therefore satisfies the
declared equations with the actual geometry ``G(q1)``, while non-unique
impulse directions never enter the convergence measure. The route
set is rebuilt from the actual path enclosure under a bounded outer epoch, and
the final result is independently recertified: kinematic update, momentum
equation, actual gaps, cone law, material completeness, physical work, energy
ledger and conservative full-path admissibility of the identical trajectory.

The contact-free identity is exact: when the native smooth Euler trajectory is
completely admissible and zero impulse satisfies every law, the native result
is returned unchanged. Unresolved segments subdivide time on a dyadic schedule
within fixed budgets; accepted segments are committed only when they cover the
entire request, otherwise every history, mode and warm start rolls back.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntEnum
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core

from ... import linalg as la
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ...discretization.contact._articulated_geometry import (
    ArticulatedCollisionGeometry,
    ArticulatedContactTrajectory,
)
from ...discretization.contact._ccd import (
    AdmissiblePathEvidence,
    AdmissiblePathPlan,
    AdmissiblePathStatus,
    certify_admissible_path,
)
from ...discretization.contact._compiled_search import (
    CompiledContactSearchPlan,
    LBVHContactSearchPlan,
)
from ...discretization.contact._distance import (
    contact_tangent_basis,
    offset_surface_witnesses,
)
from ...discretization.contact._participant import ContactParticipantScene
from ...discretization.contact._search import (
    contact_route_capacity_bounds,
    ContactCandidateEpoch,
)
from ...discretization.contact._stencils import (
    ContactPrimitiveProfile,
    ContactStencilKind,
    evaluate_contact_stencils,
)
from ...discretization.particle._articulated_mass import PreparedArticulatedMass
from ...discretization.particle._reduced_articulation import (
    PreparedReducedArticulation,
    ReducedArticulationState,
)
from ...discretization.particle._reduced_dynamics import (
    reduced_energy,
    reduced_semi_implicit_velocity_euler_step,
    ReducedSemiImplicitVelocityEulerStepPolicy,
    ReducedSemiImplicitVelocityEulerStepResult,
)
from ...discretization.particle._rigid_joint_coordinates import (
    LiftedJointCoordinateState,
    PreparedLiftedJointCoordinates,
)
from ...discretization.particle._rigid_joint_laws import (
    PreparedScalarJointDamping,
    PreparedScalarJointMotors,
)
from ...discretization.particle._rigid_unilateral import PreparedReducedJointStops
from ...linalg import (
    dual_transpose,
    FunctionLinearOperator,
)
from ...linalg.svd import SVDSolvePolicy
from ...nonlinear import (
    FixedPointIteration,
    FixedPointProblem,
    NonlinearStatus,
    NonlinearTermination,
)
from ...typing import ConvertibleToArray, parse
from ._articulated import build_delassus_operator
from ._closure_sensitivity import (
    contact_cone_fixed_branch_admitted,
    contact_cone_fixed_branch_residual,
    fix_contact_cone_branch,
)
from ._cone import (
    build_contact_cone_result,
    certify_contact_cone_impulse,
    classify_contact_cone_branches,
    contact_cone_space,
    ContactBlockLayout,
    ContactCertificatePolicy,
    ContactConeBranch,
    ContactConeCertificate,
    ContactConeProgram,
    ContactConeResult,
    ContactConeSolverPlan,
    solve_contact_cone,
)
from ._manifold import (
    ContactManifoldObservation,
    ContactManifoldPolicy,
    ContactManifoldState,
    record_contact_response,
    update_contact_manifold,
)
from ._participant_adapters import (
    ArticulatedPrimitiveContactParticipant,
    prepare_articulated_contact_scene,
)
from ._route_state import match_route_keys


CERTIFICATE_FAMILIES: tuple[str, str, str] = ("contact", "hinge-stop", "prismatic-stop")
_CONTACT_FAMILY = 0
_HINGE_FAMILY = 1
_PRISMATIC_FAMILY = 2
_NOT_ATTEMPTED = -1
_MAXIMUM_SUBDIVISION_DEPTH = 30
_ACTIVE_BRANCHES = (
    int(ContactConeBranch.STICKING),
    int(ContactConeBranch.SLIDING),
    int(ContactConeBranch.NORMAL_ACTIVE),
    int(ContactConeBranch.FRICTIONLESS_BOUNDARY),
)

type ArticulatedContactSearchPlan = CompiledContactSearchPlan | LBVHContactSearchPlan
type ArticulatedContactSearchBackend = Literal["exhaustive", "lbvh"]


class ArticulatedActuationMode(IntEnum):
    """Explicitly declared generalized-effort input of the native step.

    ``AUTONOMOUS`` applies no input. ``MAPPED_MOTORS`` maps one actuator command
    per declared stateless scalar joint motor through its gear/sign/clamp law.
    ``GENERALIZED_EFFORT`` adds one generalized effort covector (N m on hinge,
    N on prismatic coordinates, articulation velocity order) directly. Nothing
    is inferred from source effort or velocity bounds.
    """

    AUTONOMOUS = 0
    MAPPED_MOTORS = 1
    GENERALIZED_EFFORT = 2


class ArticulatedContactStepStatus(IntEnum):
    """First failing stage of one requested articulated contact interval."""

    SUCCESS = 0
    INVALID_INPUT = 1
    FREE_DYNAMICS_FAILED = 2
    SEARCH_INCOMPLETE = 3
    NARROW_PHASE_FAILED = 4
    INITIALLY_INADMISSIBLE = 5
    INVALID_GEOMETRY = 6
    WINDING_RANGE_EXCEEDED = 7
    HISTORY_FAILED = 8
    MASS_RESPONSE_FAILED = 9
    ROOT_NOT_CONVERGED = 10
    ROUTE_REBUILD_EXHAUSTED = 11
    PATH_NOT_ADMISSIBLE = 12
    CONE_FAILED = 13
    MOMENTUM_INVALID = 14
    GAP_INVALID = 15
    CONTACT_WORK_INVALID = 16
    ENERGY_INVALID = 17
    REQUEST_BUDGET_EXCEEDED = 18
    SUBSTEP_BUDGET_EXHAUSTED = 19
    MINIMUM_SEGMENT_REACHED = 20


_RETRYABLE = (
    ArticulatedContactStepStatus.ROOT_NOT_CONVERGED,
    ArticulatedContactStepStatus.ROUTE_REBUILD_EXHAUSTED,
    ArticulatedContactStepStatus.PATH_NOT_ADMISSIBLE,
    ArticulatedContactStepStatus.CONE_FAILED,
    ArticulatedContactStepStatus.MOMENTUM_INVALID,
    ArticulatedContactStepStatus.GAP_INVALID,
    ArticulatedContactStepStatus.CONTACT_WORK_INVALID,
    ArticulatedContactStepStatus.ENERGY_INVALID,
)


def _positive(value: ConvertibleToArray, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.float64)
    if array.shape != ():
        raise ValueError(f"{name} must be one scalar.")
    if not isinstance(array, jax_core.Tracer):
        concrete = float(np.asarray(array))
        if not (math.isfinite(concrete) and concrete > 0.0):
            raise ValueError(f"{name} must be finite and positive.")
    return array


class ArticulatedContactCertificatePolicy(StrictModule, NonTrainableState):
    """Dimensional, timestep-independent physical certificate of the step.

    Reference velocities define the congruent cone scaling per unit family
    (contacts m/s, hinge stops rad/s, prismatic stops m/s). Impulse tolerances
    carry the family impulse units (N s, N m s, N s); gap tolerances are the
    declared geometric floors (m, rad, m) and enter law velocities only as
    ``gap / h``. Slip speed is a physical sticking tolerance (m/s). Every
    energy quantity is in joules: complementarity work ``|p_n g / h|``,
    tangential friction work ``p_t · v_t``, physical contact work
    ``p · G(q1) v1``, frozen-metric contact kinetic change, ledger identity
    residual, positive mechanical surplus ``E1 - E0 - W_s``, the kinetic energy
    ``0.5 r · M0⁻¹ r`` of the momentum residual, and the request budget charged
    by measured geometry/work discretization terms. Impulse and energy
    tolerances are full-request budgets allocated to accepted segments in
    proportion to their duration; nothing is multiplied by ``1/h``.
    """

    contact_reference_velocity: Array
    hinge_reference_velocity: Array
    prismatic_reference_velocity: Array
    contact_impulse_tolerance: Array
    hinge_impulse_tolerance: Array
    prismatic_impulse_tolerance: Array
    contact_gap_tolerance: Array
    hinge_gap_tolerance: Array
    prismatic_gap_tolerance: Array
    slip_speed_tolerance: Array
    complementarity_work_tolerance: Array
    friction_work_tolerance: Array
    contact_work_tolerance: Array
    contact_energy_tolerance: Array
    ledger_tolerance: Array
    energy_surplus_tolerance: Array
    momentum_energy_tolerance: Array
    discretization_energy_budget: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        contact_reference_velocity: ConvertibleToArray,
        hinge_reference_velocity: ConvertibleToArray,
        prismatic_reference_velocity: ConvertibleToArray,
        contact_impulse_tolerance: ConvertibleToArray,
        hinge_impulse_tolerance: ConvertibleToArray,
        prismatic_impulse_tolerance: ConvertibleToArray,
        contact_gap_tolerance: ConvertibleToArray,
        hinge_gap_tolerance: ConvertibleToArray,
        prismatic_gap_tolerance: ConvertibleToArray,
        slip_speed_tolerance: ConvertibleToArray,
        complementarity_work_tolerance: ConvertibleToArray,
        friction_work_tolerance: ConvertibleToArray,
        contact_work_tolerance: ConvertibleToArray,
        contact_energy_tolerance: ConvertibleToArray,
        ledger_tolerance: ConvertibleToArray,
        energy_surplus_tolerance: ConvertibleToArray,
        momentum_energy_tolerance: ConvertibleToArray,
        discretization_energy_budget: ConvertibleToArray,
    ) -> None:
        self.contact_reference_velocity = _positive(
            contact_reference_velocity, "contact_reference_velocity"
        )
        self.hinge_reference_velocity = _positive(
            hinge_reference_velocity, "hinge_reference_velocity"
        )
        self.prismatic_reference_velocity = _positive(
            prismatic_reference_velocity, "prismatic_reference_velocity"
        )
        self.contact_impulse_tolerance = _positive(
            contact_impulse_tolerance, "contact_impulse_tolerance"
        )
        self.hinge_impulse_tolerance = _positive(
            hinge_impulse_tolerance, "hinge_impulse_tolerance"
        )
        self.prismatic_impulse_tolerance = _positive(
            prismatic_impulse_tolerance, "prismatic_impulse_tolerance"
        )
        self.contact_gap_tolerance = _positive(
            contact_gap_tolerance, "contact_gap_tolerance"
        )
        self.hinge_gap_tolerance = _positive(hinge_gap_tolerance, "hinge_gap_tolerance")
        self.prismatic_gap_tolerance = _positive(
            prismatic_gap_tolerance, "prismatic_gap_tolerance"
        )
        self.slip_speed_tolerance = _positive(
            slip_speed_tolerance, "slip_speed_tolerance"
        )
        self.complementarity_work_tolerance = _positive(
            complementarity_work_tolerance, "complementarity_work_tolerance"
        )
        self.friction_work_tolerance = _positive(
            friction_work_tolerance, "friction_work_tolerance"
        )
        self.contact_work_tolerance = _positive(
            contact_work_tolerance, "contact_work_tolerance"
        )
        self.contact_energy_tolerance = _positive(
            contact_energy_tolerance, "contact_energy_tolerance"
        )
        self.ledger_tolerance = _positive(ledger_tolerance, "ledger_tolerance")
        self.energy_surplus_tolerance = _positive(
            energy_surplus_tolerance, "energy_surplus_tolerance"
        )
        self.momentum_energy_tolerance = _positive(
            momentum_energy_tolerance, "momentum_energy_tolerance"
        )
        self.discretization_energy_budget = _positive(
            discretization_energy_budget, "discretization_energy_budget"
        )
        self.policy_id = canonical_fingerprint(
            {
                "kind": "articulated-contact-certificate-policy",
                "families": list(CERTIFICATE_FAMILIES),
                "allocation": "duration-proportional-request-budget",
            }
        )

    def segment_cone_certificate(
        self, step_size: Array, fraction: Array, /
    ) -> ContactCertificatePolicy:
        """Cone certificate of one segment: allocated impulse/work, gap/h laws."""

        dtype = step_size.dtype
        reference = jnp.stack(
            (
                self.contact_reference_velocity,
                self.hinge_reference_velocity,
                self.prismatic_reference_velocity,
            )
        ).astype(dtype)
        impulse = fraction * jnp.stack(
            (
                self.contact_impulse_tolerance,
                self.hinge_impulse_tolerance,
                self.prismatic_impulse_tolerance,
            )
        ).astype(dtype)
        law = (
            jnp.stack(
                (
                    self.contact_gap_tolerance,
                    self.hinge_gap_tolerance,
                    self.prismatic_gap_tolerance,
                )
            ).astype(dtype)
            / step_size
        )
        # Stop blocks are normal-only; their slip tolerance is their own
        # velocity-unit law tolerance and never enters a friction decision.
        slip = jnp.stack((self.slip_speed_tolerance.astype(dtype), law[1], law[2]))
        return ContactCertificatePolicy(
            CERTIFICATE_FAMILIES,
            reference_velocity=reference,
            impulse_tolerance=impulse,
            law_velocity_tolerance=law,
            slip_speed_tolerance=slip,
            complementarity_work_tolerance=fraction
            * self.complementarity_work_tolerance.astype(dtype),
            friction_work_tolerance=fraction * self.friction_work_tolerance.astype(dtype),
        )


class ArticulatedContactStepPolicy(StrictModule, NonTrainableState):
    """Physical certificate plus every bounded numerical budget of the step.

    ``maximum_root_iterations`` and the root tolerances control the native
    fixed-point iteration only. ``maximum_route_rebuilds`` bounds outer route
    epochs per segment, ``maximum_subdivision_depth`` the dyadic time
    refinement (segment durations ``H / 2**k``), ``maximum_segment_attempts``
    every attempted segment of one request and ``minimum_segment_duration``
    (seconds) the smallest attempted segment. The admissible-path plan's
    ``gap_floor`` must equal the certificate's contact gap tolerance.

    ``derivative_regularity`` is the explicit bounded native SVD policy that
    certifies nonsingularity of the fixed-branch cone Jacobian and of the
    coupled velocity Schur complement before a step derivative is admitted;
    ``None`` admits no contact derivative at all.
    """

    certificate: ArticulatedContactCertificatePolicy
    path: AdmissiblePathPlan
    cone_solver: ContactConeSolverPlan
    manifold: ContactManifoldPolicy
    smooth: ReducedSemiImplicitVelocityEulerStepPolicy
    derivative_regularity: SVDSolvePolicy | None
    maximum_root_iterations: int = eqx.field(static=True)
    root_absolute_tolerance: float = eqx.field(static=True)
    root_relative_tolerance: float = eqx.field(static=True)
    root_damping: float = eqx.field(static=True)
    maximum_route_rebuilds: int = eqx.field(static=True)
    maximum_subdivision_depth: int = eqx.field(static=True)
    maximum_segment_attempts: int = eqx.field(static=True)
    minimum_segment_duration: float = eqx.field(static=True)
    derivative_margin_factor: float = eqx.field(static=True)
    derivative_linear_tolerance: float = eqx.field(static=True)
    derivative_linear_iterations: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        certificate: ArticulatedContactCertificatePolicy,
        path: AdmissiblePathPlan,
        /,
        *,
        cone_solver: ContactConeSolverPlan | None = None,
        manifold: ContactManifoldPolicy | None = None,
        smooth: ReducedSemiImplicitVelocityEulerStepPolicy | None = None,
        maximum_root_iterations: int = 64,
        root_absolute_tolerance: float = 1.0e-12,
        root_relative_tolerance: float = 1.0e-10,
        root_damping: float = 1.0,
        maximum_route_rebuilds: int = 4,
        maximum_subdivision_depth: int = 6,
        maximum_segment_attempts: int = 32,
        minimum_segment_duration: float = 1.0e-6,
        derivative_margin_factor: float = 1.0,
        derivative_linear_tolerance: float = 1.0e-12,
        derivative_linear_iterations: int = 400,
        derivative_regularity: SVDSolvePolicy | None = None,
    ) -> None:
        if not isinstance(certificate, ArticulatedContactCertificatePolicy):
            raise TypeError("certificate must be ArticulatedContactCertificatePolicy.")
        if not isinstance(path, AdmissiblePathPlan):
            raise TypeError("path must be AdmissiblePathPlan.")
        if not isinstance(certificate.contact_gap_tolerance, jax_core.Tracer) and (
            float(np.asarray(certificate.contact_gap_tolerance)) != path.gap_floor
        ):
            raise ValueError(
                "The admissible-path gap floor must equal the contact gap tolerance."
            )
        solver = ContactConeSolverPlan() if cone_solver is None else cone_solver
        history = ContactManifoldPolicy() if manifold is None else manifold
        smooth_ = (
            ReducedSemiImplicitVelocityEulerStepPolicy() if smooth is None else smooth
        )
        if not isinstance(solver, ContactConeSolverPlan):
            raise TypeError("cone_solver must be ContactConeSolverPlan or None.")
        if not isinstance(history, ContactManifoldPolicy):
            raise TypeError("manifold must be ContactManifoldPolicy or None.")
        if not isinstance(smooth_, ReducedSemiImplicitVelocityEulerStepPolicy):
            raise TypeError("smooth must be the native semi-implicit Euler policy.")
        integers = {
            "maximum_root_iterations": int(maximum_root_iterations),
            "maximum_route_rebuilds": int(maximum_route_rebuilds),
            "maximum_subdivision_depth": int(maximum_subdivision_depth),
            "maximum_segment_attempts": int(maximum_segment_attempts),
            "derivative_linear_iterations": int(derivative_linear_iterations),
        }
        if integers["maximum_root_iterations"] < 1:
            raise ValueError("maximum_root_iterations must be positive.")
        if integers["maximum_route_rebuilds"] < 0:
            raise ValueError("maximum_route_rebuilds must be nonnegative.")
        if not 0 <= integers["maximum_subdivision_depth"] <= _MAXIMUM_SUBDIVISION_DEPTH:
            raise ValueError(
                f"maximum_subdivision_depth must lie in [0, {_MAXIMUM_SUBDIVISION_DEPTH}]."
            )
        if integers["maximum_segment_attempts"] < 1:
            raise ValueError("maximum_segment_attempts must be positive.")
        if integers["derivative_linear_iterations"] < 1:
            raise ValueError("derivative_linear_iterations must be positive.")
        reals = {
            "root_absolute_tolerance": float(root_absolute_tolerance),
            "root_relative_tolerance": float(root_relative_tolerance),
            "minimum_segment_duration": float(minimum_segment_duration),
            "derivative_margin_factor": float(derivative_margin_factor),
            "derivative_linear_tolerance": float(derivative_linear_tolerance),
        }
        if any(not math.isfinite(value) or value < 0.0 for value in reals.values()):
            raise ValueError("Step tolerances must be finite and nonnegative.")
        if reals["minimum_segment_duration"] <= 0.0:
            raise ValueError("minimum_segment_duration must be positive.")
        damping = float(root_damping)
        if not math.isfinite(damping) or not 0.0 < damping <= 1.0:
            raise ValueError("root_damping must lie in (0, 1].")
        self.certificate = certificate
        self.path = path
        self.cone_solver = solver
        self.manifold = history
        self.smooth = smooth_
        if derivative_regularity is not None and not isinstance(
            derivative_regularity, SVDSolvePolicy
        ):
            raise TypeError("derivative_regularity must be SVDSolvePolicy or None.")
        self.derivative_regularity = derivative_regularity
        self.maximum_root_iterations = integers["maximum_root_iterations"]
        self.root_absolute_tolerance = reals["root_absolute_tolerance"]
        self.root_relative_tolerance = reals["root_relative_tolerance"]
        self.root_damping = damping
        self.maximum_route_rebuilds = integers["maximum_route_rebuilds"]
        self.maximum_subdivision_depth = integers["maximum_subdivision_depth"]
        self.maximum_segment_attempts = integers["maximum_segment_attempts"]
        self.minimum_segment_duration = reals["minimum_segment_duration"]
        self.derivative_margin_factor = reals["derivative_margin_factor"]
        self.derivative_linear_tolerance = reals["derivative_linear_tolerance"]
        self.derivative_linear_iterations = integers["derivative_linear_iterations"]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "articulated-contact-step-policy",
                "certificate": certificate.policy_id,
                "path": path.plan_id,
                "cone_solver": solver.plan_id,
                "manifold": {
                    "retention_steps": history.retention_steps,
                    "frame_transport": int(history.frame_transport),
                },
                "smooth": {
                    "maximum_step_size": float(smooth_.maximum_step_size).hex(),
                    "absolute_energy_tolerance": float(
                        smooth_.absolute_energy_tolerance
                    ).hex(),
                    "relative_energy_tolerance": float(
                        smooth_.relative_energy_tolerance
                    ).hex(),
                    "inverse_forward_tolerance": float(
                        smooth_.inverse_forward_tolerance
                    ).hex(),
                },
                "integers": integers,
                "reals": {name: float(value).hex() for name, value in reals.items()},
                "root_damping": damping.hex(),
                "root": "native-fixed-point-relinearized-frozen-route-cone",
                # The SVD policy itself is a PyTree field of this policy, so
                # its configuration is part of every step binding and treedef.
                "derivative_regularity": derivative_regularity is not None,
            }
        )


class ArticulatedContactSource(StrictModule):
    """Complete mechanical source of one case.

    ``configuration`` is the articulation's principal chart point and
    ``winding`` its integer hinge turn history; the lifted coordinate is their
    sum. ``contact_history`` is the mechanical manifold (``None`` without
    collision geometry); ``stop_impulse``/``stop_branch`` are the last accepted
    stop reactions and fixed-branch labels in stop-row order.
    """

    configuration: Array
    winding: Array
    velocity: Array
    contact_history: ContactManifoldState | None
    stop_impulse: Array
    stop_branch: Array


class ArticulatedRouteGeometry(StrictModule):
    """Narrow-phase rows of one route epoch evaluated at one lifted configuration."""

    route_keys: Array
    valid: Array
    failed: Array
    gap: Array
    normal: Array
    tangent_basis: Array
    left_point: Array
    right_point: Array
    left_body: Array
    right_body: Array
    left_material: Array
    right_material: Array
    feature_margin: Array
    finite: Array


class ArticulatedContactEnergyLedger(StrictModule):
    """Mechanical-energy ledger of a segment or a full request (joules).

    ``E1 - E0 - source_work - contact_work + correction_loss =
    predictor_defect + geometry_residual + work_residual + identity_residual``.
    ``correction_loss`` is ``D_contact = 0.5 (v1 - v_free)·M0(v1 - v_free)``;
    ``contact_work`` is the physical ``p · G(q1) v1``. Positive parts and the
    discretization charge are accumulated against the request budgets.
    """

    initial_energy: Array
    final_energy: Array
    source_work: Array
    contact_work: Array
    correction_loss: Array
    contact_kinetic_change: Array
    predictor_defect: Array
    geometry_residual: Array
    work_residual: Array
    identity_residual: Array
    friction_work: Array
    positive_contact_work: Array
    positive_contact_kinetic_change: Array
    positive_friction_work: Array
    positive_energy_surplus: Array
    discretization_charge: Array
    roundoff_allowance: Array

    def accumulate(
        self, segment: ArticulatedContactEnergyLedger, accepted: Array, /
    ) -> ArticulatedContactEnergyLedger:
        """Add an accepted segment; the request's final energy is the segment's."""

        def add(total: Array, value: Array) -> Array:
            return total + jnp.where(accepted, value, 0.0)

        return ArticulatedContactEnergyLedger(
            self.initial_energy,
            jnp.where(accepted, segment.final_energy, self.final_energy),
            add(self.source_work, segment.source_work),
            add(self.contact_work, segment.contact_work),
            add(self.correction_loss, segment.correction_loss),
            add(self.contact_kinetic_change, segment.contact_kinetic_change),
            add(self.predictor_defect, segment.predictor_defect),
            add(self.geometry_residual, segment.geometry_residual),
            add(self.work_residual, segment.work_residual),
            add(self.identity_residual, segment.identity_residual),
            add(self.friction_work, segment.friction_work),
            add(self.positive_contact_work, segment.positive_contact_work),
            add(
                self.positive_contact_kinetic_change,
                segment.positive_contact_kinetic_change,
            ),
            add(self.positive_friction_work, segment.positive_friction_work),
            add(self.positive_energy_surplus, segment.positive_energy_surplus),
            add(self.discretization_charge, segment.discretization_charge),
            add(self.roundoff_allowance, segment.roundoff_allowance),
        )


class ArticulatedContactRootEvidence(StrictModule):
    """Coupled-root and route-epoch evidence of one segment.

    ``status`` is the native nonlinear status of the last root solve, or ``-1``
    when the exact contact-free bypass made no root attempt.
    """

    attempted: Array
    status: Array
    iterations: Array
    residual_norm: Array
    route_rebuilds: Array
    routes_stable: Array
    search_successful: Array
    narrow_phase_successful: Array
    warm_history_successful: Array


class ArticulatedContactSegmentEvidence(StrictModule):
    """Every staged result of one attempted segment, accepted or not.

    Blocks are ordered contact routes first, then stop rows. ``law_velocity``
    is the position-level law velocity (``g(q1)/h`` on normals) and is distinct
    from the measured ``physical_velocity = G(q1) v1``. ``cone_certificate``
    evaluates the law at the accepted endpoint velocity ``v1``;
    ``response_certificate`` evaluates it at the response map's own velocity
    ``v_free + M0⁻¹G(q1)* p``. Both must hold, together with the separate
    momentum residual budget.
    """

    duration: Array
    fraction: Array
    free_bypass: Array
    free_status: Array
    free_successful: Array
    mass_status: Array
    root: ArticulatedContactRootEvidence
    route_epoch: ContactCandidateEpoch | None
    coverage_epoch: ContactCandidateEpoch | None
    path: AdmissiblePathEvidence | None
    stop_path_minimum_gap: Array
    winding_increment: Array
    lift_residual: Array
    block_keys: Array
    block_valid: Array
    impulse: Array
    branch: Array
    branch_margin: Array
    activation_event: Array
    gap: Array
    physical_velocity: Array
    law_velocity: Array
    cone_certificate: ContactConeCertificate | None
    response_certificate: ContactConeCertificate | None
    momentum_residual_energy: Array
    momentum_successful: Array
    gaps_successful: Array
    cone_successful: Array
    contact_work_successful: Array
    energy_successful: Array
    history_successful: Array
    feature_margin: Array
    fixed_branch_candidate: Array
    ledger: ArticulatedContactEnergyLedger
    active_contact_count: Array
    active_stop_count: Array
    finite: Array
    retryable: Array
    successful: Array
    status: Array


class ArticulatedContactStepBinding(StrictModule):
    """Exact numerical inputs of one request, retained for derivative binding.

    ``step_arrays`` is every array leaf of the prepared step (geometry,
    inertia, friction tables, layouts and solver/certificate numerics) with
    non-array leaves removed; ``step_id`` its static identity. Derivatives
    compare the current step, actuation and gravity against this binding
    exactly and refuse any mismatch rather than rebuilding a fresh revision.
    """

    step_arrays: PreparedArticulatedContactStep
    actuation: Array
    gravity: Array
    geometry_revision: Array
    material_revision: Array
    step_id: str = eqx.field(static=True)


class ArticulatedContactStepResult(StrictModule):
    """Atomic result of one requested interval.

    ``accepted`` equals ``candidate`` only when the concatenated accepted
    segments cover the full request and every request budget holds; otherwise
    it is the unchanged source. A certified prefix never counts as success.

    ``fixed_branch_candidate`` is only the primal discrete-branch condition for
    differentiation (one accepted segment, classified branches with margin, no
    activation, feature or winding transition). It is not derivative
    admission: :meth:`PreparedArticulatedContactStep.step_jvp`/``step_vjp``
    additionally require the exact input binding and certified regularity.
    """

    source: ArticulatedContactSource
    candidate: ArticulatedContactSource
    accepted: ArticulatedContactSource
    requested_duration: Array
    covered_duration: Array
    covered_fraction: Array
    accepted_segments: Array
    attempted_segments: Array
    route_rebuilds: Array
    root_iterations: Array
    attempt_status: Array
    attempt_fraction: Array
    attempt_free_bypass: Array
    attempt_accepted: Array
    last_segment: ArticulatedContactSegmentEvidence
    ledger: ArticulatedContactEnergyLedger
    request_budgets_successful: Array
    free_bypass: Array
    fixed_branch_candidate: Array
    binding: ArticulatedContactStepBinding
    successful: Array
    status: Array


class ArticulatedContactAdmissibility(StrictModule):
    """Static admissibility of one accepted configuration (reset/refresh)."""

    epoch: ContactCandidateEpoch | None
    path: AdmissiblePathEvidence | None
    minimum_contact_gap: Array
    minimum_stop_gap: Array
    winding_valid: Array
    finite: Array
    successful: Array
    status: Array


class ArticulatedContactStepJVP(StrictModule):
    """Fixed-branch tangent of a single-segment step, or a refusal.

    With contact or stop blocks the tangent solves ONE coupled fixed-branch
    linearization in the unknowns ``(v1, p)``: the momentum residual
    ``v1 - v_free - M0⁻¹G(q1)* p`` together with the cone owner's fixed-branch
    law residual at the accepted endpoint law velocity (``g(q1)/h`` normal,
    ``G_t(q1) v1`` tangential). ``successful`` requires the primal
    fixed-branch candidate, the exact input binding, the owner's branch
    admission (``branch_admitted``), a native rank certificate proving that
    coupled Jacobian nonsingular with fixed rank (``coupled_regular``), a
    successful linear solve on that same certified operator, and finite
    output. The certificate is computed before any solve; a refused or
    over-budget certificate (route ``"unavailable"``) skips the solve.
    ``coupled_regularity`` is ``None`` without contact/stop blocks (explicit
    Euler map, nothing implicit to certify) or without a regularity policy,
    which refuses. Refusals return zero tangents.
    """

    configuration_tangent: Array
    velocity_tangent: Array
    impulse_tangent: Array
    fixed_branch_candidate: Array
    binding_current: Array
    branch_admitted: Array
    coupled_regular: Array
    coupled_regularity: la.RectangularRankCertificate | None
    linear_successful: Array
    successful: Array


class ArticulatedContactStepVJP(StrictModule):
    """Fixed-branch cotangents of a single-segment step, or a refusal.

    Admission is identical to :class:`ArticulatedContactStepJVP`; the adjoint
    solves the transpose of the same certified coupled operator.
    ``step_cotangent`` holds the cotangent of every inexact step leaf; when
    ``friction_cotangent`` is false its friction tables are ``None`` (not
    formed), never zero.
    """

    step_cotangent: PreparedArticulatedContactStep
    configuration_cotangent: Array
    velocity_cotangent: Array
    actuation_cotangent: Array
    gravity_cotangent: Array
    fixed_branch_candidate: Array
    binding_current: Array
    branch_admitted: Array
    coupled_regular: Array
    coupled_regularity: la.RectangularRankCertificate | None
    linear_successful: Array
    successful: Array
    friction_cotangent: bool = eqx.field(static=True)


def _first_failure(
    stages: tuple[tuple[Array, ArticulatedContactStepStatus], ...], /
) -> Array:
    status = jnp.asarray(int(ArticulatedContactStepStatus.SUCCESS), dtype=jnp.int32)
    for passed, failure in stages:
        status = jnp.where(
            (status == int(ArticulatedContactStepStatus.SUCCESS)) & ~passed,
            jnp.asarray(int(failure), dtype=jnp.int32),
            status,
        )
    return status


def _retryable(status: Array, /) -> Array:
    flags = jnp.asarray(False)
    for value in _RETRYABLE:
        flags = flags | (status == int(value))
    return flags


def _material_tables(
    scene: ContactParticipantScene,
    material_pairs: ConvertibleToArray,
    static_friction: ConvertibleToArray,
    dynamic_friction: ConvertibleToArray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host lookup ``feature -> material rank`` and rank-pair friction tables."""

    features = np.asarray(scene.feature_material_ids, dtype=np.int64)
    materials = np.unique(features)
    rank = {int(value): index for index, value in enumerate(materials)}
    feature_rank = np.asarray([rank[int(value)] for value in features], dtype=np.int32)
    pairs = np.asarray(material_pairs)
    if (
        pairs.ndim != 2
        or pairs.shape[1:] != (2,)
        or not np.issubdtype(pairs.dtype, np.integer)
    ):
        raise TypeError("material_pairs must be an integer (pairs, 2) array.")
    static = np.asarray(static_friction, dtype=np.float64).reshape((-1,))
    dynamic = np.asarray(dynamic_friction, dtype=np.float64).reshape((-1,))
    if static.shape != (pairs.shape[0],) or dynamic.shape != static.shape:
        raise ValueError("Friction coefficients need one value per material pair.")
    if np.any(~np.isfinite(static)) or np.any(dynamic < 0.0) or np.any(dynamic > static):
        raise ValueError("Friction requires finite 0 <= dynamic <= static.")
    count = materials.size
    static_table = np.zeros((count, count), dtype=np.float64)
    dynamic_table = np.zeros((count, count), dtype=np.float64)
    available = np.zeros((count, count), dtype=np.bool_)
    for (left, right), mu_s, mu_d in zip(pairs.tolist(), static, dynamic, strict=True):
        if int(left) not in rank or int(right) not in rank:
            continue
        i, j = rank[int(left)], rank[int(right)]
        if available[i, j]:
            raise ValueError("material_pairs must not repeat an unordered pair.")
        for a, b in ((i, j), (j, i)):
            static_table[a, b] = mu_s
            dynamic_table[a, b] = mu_d
            available[a, b] = True
    return feature_rank, static_table, dynamic_table, available


def _right_vertex_column(kind: ContactStencilKind, /) -> int:
    match kind:
        case (
            ContactStencilKind.VERTEX_VERTEX
            | ContactStencilKind.EDGE_VERTEX
            | ContactStencilKind.FACE_VERTEX
        ):
            return 1
        case ContactStencilKind.EDGE_EDGE:
            return 2
        case _:
            return -1


def _epoch_keys(epoch: ContactCandidateEpoch, /) -> tuple[Array, Array]:
    batches = epoch.active_batches
    return (
        jnp.concatenate(tuple(batch.route_keys for batch in batches)),
        jnp.concatenate(tuple(batch.valid for batch in batches)),
    )


def _routes_cover(
    reference: ContactCandidateEpoch, query: ContactCandidateEpoch, /
) -> Array:
    """Whether every valid route of ``query`` is a valid route of ``reference``."""

    reference_keys, reference_valid = _epoch_keys(reference)
    query_keys, query_valid = _epoch_keys(query)
    found, _, _, _ = match_route_keys(
        query_keys, query_valid, reference_keys, reference_valid
    )
    return jnp.all(~query_valid | found)


class _RouteCarry(StrictModule):
    """Outer route-epoch loop state of one segment's coupled root."""

    rebuild: Array
    epoch: ContactCandidateEpoch
    solved: ContactCandidateEpoch
    coverage: ContactCandidateEpoch
    velocity: Array
    impulse: Array
    status: Array
    iterations: Array
    residual: Array
    stable: Array
    search: Array
    narrow: Array
    history: Array
    done: Array
    revision: Array


class _RequestCarry(StrictModule):
    """Dyadic segment-schedule state of one requested interval."""

    source: ArticulatedContactSource
    covered: Array
    level: Array
    count: Array
    accepted: Array
    rebuilds: Array
    iterations: Array
    ledger: ArticulatedContactEnergyLedger
    last: ArticulatedContactSegmentEvidence
    attempt_status: Array
    attempt_fraction: Array
    attempt_bypass: Array
    attempt_accepted: Array
    status: Array
    done: Array


# Differentiated inputs: inexact step leaves, lifted q0, v0, actuation, gravity.
type _StepParameters = tuple[PreparedArticulatedContactStep, Array, Array, Array, Array]

type _Predictor = Callable[
    [PreparedArticulatedContactStep, Array, Array, Array, Array],
    tuple[Array, PreparedArticulatedMass, ContactCertificatePolicy, Array],
]


class _CoupledFixedBranch(StrictModule):
    """One prepared coupled ``(v1, p)`` fixed-branch linearization and its proof.

    The unknown vector is ``[v1, p / scale]`` (blockwise congruent impulse
    scaling of the cone owner); ``jacobian`` acts on it through JVP/VJP of
    ``linearization`` only, with no inner derivative solves.
    """

    linearization: la.PreparedLinearization
    jacobian: la.AbstractLinearOperator
    certificate: la.RectangularRankCertificate
    regular: Array
    branch_admitted: Array
    scale: Array
    zero_parameters: _StepParameters
    zero_unknown: Array


class _ContactPreparation(NamedTuple):
    """Host result of binding optional collision geometry to the step."""

    geometry: ArticulatedCollisionGeometry | None
    scene: ContactParticipantScene | None
    search: ArticulatedContactSearchPlan | None
    contact_capacity: int
    catalogue_id: str
    feature_rank: np.ndarray
    static_table: np.ndarray
    dynamic_table: np.ndarray
    available: np.ndarray
    contact_free_geometry: str | None


def _validate_step_owners(
    articulation: PreparedReducedArticulation,
    coordinates: PreparedLiftedJointCoordinates,
    policy: ArticulatedContactStepPolicy,
    /,
) -> None:
    """Require the scalar-joint articulation, its coordinates and a step policy."""

    if not isinstance(articulation, PreparedReducedArticulation):
        raise TypeError("articulation must be PreparedReducedArticulation.")
    if not isinstance(coordinates, PreparedLiftedJointCoordinates):
        raise TypeError("coordinates must be PreparedLiftedJointCoordinates.")
    if coordinates.articulation_id != articulation.prepared_id:
        raise ValueError("Lifted coordinates belong to another articulation.")
    if not isinstance(policy, ArticulatedContactStepPolicy):
        raise TypeError("policy must be ArticulatedContactStepPolicy.")
    if articulation.nq != articulation.nv:
        raise ValueError("The native contact step requires scalar joint coordinates.")


def _validate_joint_laws(
    articulation: PreparedReducedArticulation,
    mode: ArticulatedActuationMode,
    motors: PreparedScalarJointMotors | None,
    stops: PreparedReducedJointStops | None,
    /,
) -> None:
    """Require motors exactly in MAPPED_MOTORS mode and stops of this articulation."""

    if mode == ArticulatedActuationMode.MAPPED_MOTORS and motors is None:
        raise ValueError("MAPPED_MOTORS actuation requires prepared joint motors.")
    if mode != ArticulatedActuationMode.MAPPED_MOTORS and motors is not None:
        raise ValueError("Prepared motors are executed only in MAPPED_MOTORS mode.")
    if stops is not None and stops.nv != articulation.nv:
        raise ValueError("Joint stops belong to another articulation.")


def _contact_free(dtype: np.dtype, geometry_id: str | None, /) -> _ContactPreparation:
    """Preparation without contact routes (no geometry, or provably none)."""

    return _ContactPreparation(
        None,
        None,
        None,
        0,
        "no-contact-geometry",
        np.zeros((0,), dtype=np.int32),
        np.zeros((0, 0), dtype=dtype),
        np.zeros((0, 0), dtype=dtype),
        np.zeros((0, 0), dtype=np.bool_),
        geometry_id,
    )


def _scene_participant(
    scene: ContactParticipantScene, /
) -> ArticulatedPrimitiveContactParticipant:
    """The single articulated primitive participant owned by a step's scene."""

    match scene.participants:
        case (ArticulatedPrimitiveContactParticipant() as participant,):
            return participant
        case _:
            raise TypeError(
                "The articulated contact scene must hold one articulated primitive "
                "participant."
            )


def _route_capacities(
    scene: ContactParticipantScene,
    route_capacities: tuple[int, int, int, int, int, int] | None,
    /,
) -> tuple[int, int, int, int, int, int]:
    """Explicit witness-row budget checked against O(features) legal bounds."""

    bounds = contact_route_capacity_bounds(scene, ContactPrimitiveProfile.SPHERE_CAPSULE)
    if route_capacities is None and any(bounds):
        raise ValueError(
            "Contact geometry requires explicit route_capacities "
            "(vertex-vertex, edge-vertex, edge-edge, face-vertex, "
            "vertex-plane, edge-plane witness rows); legal-route bounds are "
            f"{bounds}."
        )
    capacities = (
        (0, 0, 0, 0, 0, 0)
        if route_capacities is None
        else tuple(int(value) for value in route_capacities)
    )
    if len(capacities) != 6 or any(value < 0 for value in capacities):
        raise ValueError("route_capacities must be six nonnegative counts.")
    if sum(capacities) == 0 and any(bounds):
        raise ValueError(
            "Zero route capacity is admitted only for geometry that can "
            f"form no contact route; this scene's bounds are {bounds}."
        )
    return (
        capacities[0],
        capacities[1],
        capacities[2],
        capacities[3],
        capacities[4],
        capacities[5],
    )


def _contact_search(
    scene: ContactParticipantScene,
    capacities: tuple[int, int, int, int, int, int],
    backend: ArticulatedContactSearchBackend,
    activation_distance: float,
    envelope_radius: float,
    maximum_traversal_visits: int | None,
    /,
) -> ArticulatedContactSearchPlan:
    """The explicitly selected canonical route search with explicit budgets."""

    vertex_vertex, edge_vertex, edge_edge, face_vertex, vertex_plane, edge_plane = (
        capacities
    )
    if backend == "exhaustive":
        if maximum_traversal_visits is not None:
            raise ValueError("maximum_traversal_visits applies only to the LBVH backend.")
        return CompiledContactSearchPlan(
            scene,
            vertex_vertex_capacity=vertex_vertex,
            edge_vertex_capacity=edge_vertex,
            edge_edge_capacity=edge_edge,
            face_vertex_capacity=face_vertex,
            vertex_plane_capacity=vertex_plane,
            edge_plane_capacity=edge_plane,
            activation_distance=float(activation_distance),
            envelope_radius=float(envelope_radius),
            primitive_profile=ContactPrimitiveProfile.SPHERE_CAPSULE,
        )
    if maximum_traversal_visits is None:
        raise ValueError(
            "The LBVH backend requires an explicit maximum_traversal_visits budget."
        )
    return LBVHContactSearchPlan(
        scene,
        vertex_vertex_capacity=vertex_vertex,
        edge_vertex_capacity=edge_vertex,
        edge_edge_capacity=edge_edge,
        face_vertex_capacity=face_vertex,
        vertex_plane_capacity=vertex_plane,
        edge_plane_capacity=edge_plane,
        activation_distance=float(activation_distance),
        envelope_radius=float(envelope_radius),
        primitive_profile=ContactPrimitiveProfile.SPHERE_CAPSULE,
        maximum_traversal_visits=int(maximum_traversal_visits),
    )


def _prepare_contact(
    articulation: PreparedReducedArticulation,
    geometry: ArticulatedCollisionGeometry | None,
    /,
    *,
    material_pairs: ConvertibleToArray | None,
    static_friction: ConvertibleToArray | None,
    dynamic_friction: ConvertibleToArray | None,
    backend: ArticulatedContactSearchBackend,
    route_capacities: tuple[int, int, int, int, int, int] | None,
    activation_distance: float | None,
    envelope_radius: float,
    maximum_traversal_visits: int | None,
    dtype: np.dtype,
) -> _ContactPreparation:
    """Bind geometry, participant, route budget, materials and search on the host.

    Geometry whose legal-route bounds are all zero is recorded by identity
    only and the step runs contact-free.
    """

    if geometry is None:
        if material_pairs is not None or route_capacities is not None:
            raise ValueError("Friction tables and route capacities require geometry.")
        return _contact_free(dtype, None)
    if not isinstance(geometry, ArticulatedCollisionGeometry):
        raise TypeError("geometry must be ArticulatedCollisionGeometry or None.")
    if geometry.articulation_id != articulation.prepared_id:
        raise ValueError("Collision geometry belongs to another articulation.")
    if material_pairs is None or static_friction is None or dynamic_friction is None:
        raise ValueError("Contact geometry requires an explicit material friction table.")
    if activation_distance is None:
        raise ValueError("Contact geometry requires an explicit activation_distance (m).")
    scene = prepare_articulated_contact_scene(articulation, geometry)
    capacities = _route_capacities(scene, route_capacities)
    _scene_participant(scene)
    feature_rank, static_table, dynamic_table, available = _material_tables(
        scene, material_pairs, static_friction, dynamic_friction
    )
    if sum(capacities) == 0:
        # No pair of this geometry can ever form a route: the step is
        # contact-free and keeps only the geometry identity.
        return _contact_free(dtype, geometry.geometry_id)
    search = _contact_search(
        scene,
        capacities,
        backend,
        activation_distance,
        envelope_radius,
        maximum_traversal_visits,
    )
    return _ContactPreparation(
        geometry,
        scene,
        search,
        sum(capacities),
        scene.feature_catalogue.catalogue_id,
        feature_rank,
        static_table,
        dynamic_table,
        available,
        None,
    )


def _stop_block_layout(
    contact_capacity: int, stops: PreparedReducedJointStops | None, /
) -> tuple[int, np.ndarray, ContactBlockLayout | None]:
    """Stop rows, their hinge flags and the mixed Coulomb/normal-only layout."""

    stop_capacity = 0 if stops is None else stops.capacity
    stop_units = () if stops is None else stops.row_units
    stop_hinge = np.asarray([unit == "rad" for unit in stop_units], dtype=np.bool_)
    if contact_capacity + stop_capacity == 0:
        return stop_capacity, stop_hinge, None
    layout = ContactBlockLayout.mixed(
        contact_capacity,
        tuple(_HINGE_FAMILY if hinge else _PRISMATIC_FAMILY for hinge in stop_hinge),
        contact_unit_family=_CONTACT_FAMILY,
        family_count=len(CERTIFICATE_FAMILIES),
        width=3,
    )
    return stop_capacity, stop_hinge, layout


class PreparedArticulatedContactStep(StrictModule, NonTrainableState):
    """Prepared native articulated contact/limit step of one robot model.

    Owners are composed, not duplicated: native ABA prediction and prepared
    articulated mass, lifted joint coordinates, analytic articulated geometry
    and its participant scene, canonical device route search, conservative
    admissible-path certification, the operator-valued mixed cone program and
    the mechanical contact manifold. All loop bounds and capacities are static;
    every numerical leaf stays dynamic. Whenever contact routes exist the
    participant scene is the sole owner of the articulation and collision
    geometry; `participant`, `articulation` and `geometry` read through it.
    """

    _articulation: PreparedReducedArticulation | None
    coordinates: PreparedLiftedJointCoordinates
    scene: ContactParticipantScene | None
    search: ArticulatedContactSearchPlan | None
    stops: PreparedReducedJointStops | None
    motors: PreparedScalarJointMotors | None
    damping: PreparedScalarJointDamping | None
    policy: ArticulatedContactStepPolicy
    layout: ContactBlockLayout | None
    feature_material_rank: Array
    static_friction_table: Array
    dynamic_friction_table: Array
    material_available: Array
    stop_hinge: Array
    actuation_mode: ArticulatedActuationMode = eqx.field(static=True)
    contact_capacity: int = eqx.field(static=True)
    stop_capacity: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    catalogue_id: str = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    step_id: str = eqx.field(static=True)

    def __init__(
        self,
        articulation: PreparedReducedArticulation,
        coordinates: PreparedLiftedJointCoordinates,
        policy: ArticulatedContactStepPolicy,
        /,
        *,
        actuation_mode: ArticulatedActuationMode = ArticulatedActuationMode.AUTONOMOUS,
        geometry: ArticulatedCollisionGeometry | None = None,
        material_pairs: ConvertibleToArray | None = None,
        static_friction: ConvertibleToArray | None = None,
        dynamic_friction: ConvertibleToArray | None = None,
        stops: PreparedReducedJointStops | None = None,
        motors: PreparedScalarJointMotors | None = None,
        damping: PreparedScalarJointDamping | None = None,
        search_backend: ArticulatedContactSearchBackend = "exhaustive",
        route_capacities: tuple[int, int, int, int, int, int] | None = None,
        activation_distance: float | None = None,
        envelope_radius: float = 0.0,
        maximum_traversal_visits: int | None = None,
        history_capacity: int | None = None,
    ) -> None:
        _validate_step_owners(articulation, coordinates, policy)
        mode = ArticulatedActuationMode(actuation_mode)
        backend = parse(search_backend, ArticulatedContactSearchBackend, "search_backend")
        _validate_joint_laws(articulation, mode, motors, stops)
        dtype = np.dtype(articulation.reference_position.dtype)
        contact = _prepare_contact(
            articulation,
            geometry,
            material_pairs=material_pairs,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            backend=backend,
            route_capacities=route_capacities,
            activation_distance=activation_distance,
            envelope_radius=envelope_radius,
            maximum_traversal_visits=maximum_traversal_visits,
            dtype=dtype,
        )
        geometry = contact.geometry
        search = contact.search
        contact_capacity = contact.contact_capacity
        stop_capacity, stop_hinge, layout = _stop_block_layout(contact_capacity, stops)
        history = (
            0
            if geometry is None
            else (contact_capacity if history_capacity is None else int(history_capacity))
        )
        if geometry is not None and history < 1:
            raise ValueError("history_capacity must be positive.")
        # With a scene, its participant holds this articulation and geometry.
        self._articulation = articulation if contact.scene is None else None
        self.coordinates = coordinates
        self.scene = contact.scene
        self.search = search
        self.stops = stops
        self.motors = motors
        self.damping = damping
        self.policy = policy
        self.layout = layout
        self.feature_material_rank = jnp.asarray(contact.feature_rank)
        self.static_friction_table = jnp.asarray(contact.static_table, dtype=dtype)
        self.dynamic_friction_table = jnp.asarray(contact.dynamic_table, dtype=dtype)
        self.material_available = jnp.asarray(contact.available)
        self.stop_hinge = jnp.asarray(stop_hinge)
        self.actuation_mode = mode
        self.contact_capacity = contact_capacity
        self.stop_capacity = stop_capacity
        self.history_capacity = history
        self.catalogue_id = contact.catalogue_id
        self.step_id = canonical_fingerprint(
            {
                "kind": "prepared-articulated-contact-step",
                "articulation": articulation.prepared_id,
                "coordinates": coordinates.prepared_id,
                "geometry": None if geometry is None else geometry.geometry_id,
                "contact_free_geometry": contact.contact_free_geometry,
                "search": None if search is None else search.plan_id,
                "stops": None if stops is None else stops.prepared_id,
                "motors": None if motors is None else motors.prepared_id,
                "damping": None if damping is None else damping.prepared_id,
                "policy": policy.policy_id,
                "layout": None if layout is None else layout.layout_id,
                "actuation_mode": int(mode),
                "history_capacity": history,
                "law": "position-level-inelastic-coulomb-hard-contact-coupled-stops",
            }
        )
        self.program_id = canonical_fingerprint(
            {"kind": "articulated-contact-step-program", "step": self.step_id}
        )

    # ------------------------------------------------------------------ owners

    @property
    def participant(self) -> ArticulatedPrimitiveContactParticipant | None:
        """The scene's articulated participant; ``None`` without contact routes."""

        return None if self.scene is None else _scene_participant(self.scene)

    @property
    def articulation(self) -> PreparedReducedArticulation:
        """The prepared articulation; the scene participant's when routes exist.

        A step with a scene stores no second articulation, so an update made
        through ``scene`` reaches prediction, mass, routes and trajectories alike.
        """

        participant = self.participant
        if participant is None:
            if self._articulation is None:
                raise RuntimeError("A contact-free step must own its articulation.")
            return self._articulation
        if self._articulation is not None:
            raise RuntimeError(
                "A step with a contact scene must not store a second articulation."
            )
        return participant.articulation

    @property
    def geometry(self) -> ArticulatedCollisionGeometry | None:
        """Collision geometry of the scene participant; ``None`` without routes."""

        participant = self.participant
        return None if participant is None else participant.geometry

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.articulation.reference_position.dtype)

    @property
    def block_count(self) -> int:
        return self.contact_capacity + self.stop_capacity

    @property
    def actuation_size(self) -> int:
        match self.actuation_mode:
            case ArticulatedActuationMode.MAPPED_MOTORS:
                if self.motors is None:
                    raise RuntimeError("Mapped actuation has no prepared motor law.")
                return int(self.motors.plan.count)
            case ArticulatedActuationMode.GENERALIZED_EFFORT:
                return self.articulation.nv
            case _:
                return 0

    def initial_source(
        self,
        configuration: ConvertibleToArray,
        winding: ConvertibleToArray,
        velocity: ConvertibleToArray,
        /,
    ) -> ArticulatedContactSource:
        """Source with empty contact history and zero stop reactions."""

        dtype = self.dtype
        history = (
            None
            if self.geometry is None
            else ContactManifoldState.empty(
                self.history_capacity,
                catalogue_id=self.catalogue_id,
                ambient_dimension=3,
                witness_count=2,
                dtype=dtype,
            )
        )
        return ArticulatedContactSource(
            jnp.asarray(configuration, dtype=dtype),
            jnp.asarray(winding, dtype=jnp.int32),
            jnp.asarray(velocity, dtype=dtype),
            history,
            jnp.zeros((self.stop_capacity,), dtype=dtype),
            jnp.full(
                (self.stop_capacity,), int(ContactConeBranch.UNCLASSIFIED), dtype=jnp.int8
            ),
        )

    def effort(self, actuation: Array, velocity: Array, /) -> tuple[Array, Array]:
        """Held generalized source effort and its success (motors/direct + damping)."""

        nv = self.articulation.nv
        effort = jnp.zeros((nv,), dtype=velocity.dtype)
        successful = jnp.asarray(True)
        match self.actuation_mode:
            case ArticulatedActuationMode.MAPPED_MOTORS:
                if self.motors is None:
                    raise RuntimeError("Mapped actuation has no prepared motor law.")
                motor = self.motors.evaluate(actuation, velocity)
                effort = effort + motor.generalized_effort.astype(velocity.dtype)
                successful = successful & motor.successful
            case ArticulatedActuationMode.GENERALIZED_EFFORT:
                effort = effort + actuation.astype(velocity.dtype)
                successful = successful & jnp.all(jnp.isfinite(actuation))
            case _:
                pass
        if self.damping is not None:
            damping = self.damping.evaluate(velocity)
            effort = effort + damping.generalized_effort.astype(velocity.dtype)
            successful = successful & damping.successful
        return effort, successful

    # --------------------------------------------------------------- geometry

    def _trajectory(
        self, start: Array, velocity: Array, step_size: Array, revision: Array, /
    ) -> ArticulatedContactTrajectory:
        participant = self.participant
        if participant is None:
            raise RuntimeError("Contact trajectories require a prepared participant.")
        return participant.trajectory(start, velocity, step_size, revision=revision)

    def _search(
        self, trajectory: ArticulatedContactTrajectory, /
    ) -> ContactCandidateEpoch:
        if self.search is None or self.scene is None:
            raise RuntimeError("Contact search preparation is incomplete.")
        return self.search.evaluate(
            self.scene,
            trajectory.evaluate(0.0),
            path_enclosure=trajectory.enclosure(0.0, 1.0),
        ).epoch

    def route_geometry(
        self, epoch: ContactCandidateEpoch, configuration: Array, /
    ) -> ArticulatedRouteGeometry:
        """Evaluate every route row of ``epoch`` at one lifted configuration."""

        if self.geometry is None or self.scene is None:
            raise RuntimeError("Contact route geometry preparation is incomplete.")
        geometry = self.geometry
        scene = self.scene
        positions = geometry.vertex_positions(self.articulation, configuration)
        radius = scene.feature_physical_radius.astype(positions.dtype)
        vertex_bodies = geometry.vertex_body_indices
        parts: list[ArticulatedRouteGeometry] = []
        for batch in epoch.active_batches:
            evaluation = evaluate_contact_stencils(
                batch,
                positions,
                positions,
                scene=scene,
                tolerance=self.policy.path.distance_tolerance,
            )
            left = jnp.maximum(batch.left_feature_indices, 0)
            right = jnp.maximum(batch.right_feature_indices, 0)
            gap, left_point, right_point = offset_surface_witnesses(
                evaluation.distance, radius[left], radius[right]
            )
            valid = evaluation.valid
            normal = jnp.where(valid[:, None], evaluation.distance.normal, 0.0)
            basis = jnp.where(valid[:, None, None], contact_tangent_basis(normal), 0.0)
            vertices = jnp.clip(batch.vertex_indices, 0, positions.shape[0] - 1)
            column = _right_vertex_column(batch.kind)
            right_body = (
                jnp.full((batch.capacity,), -1, dtype=jnp.int32)
                if column < 0
                else vertex_bodies[vertices[:, column]]
            )
            parts.append(
                ArticulatedRouteGeometry(
                    route_keys=batch.route_keys,
                    valid=valid,
                    failed=evaluation.failed,
                    gap=jnp.where(valid, gap, 0.0),
                    normal=normal,
                    tangent_basis=basis,
                    left_point=jnp.where(valid[:, None], left_point, 0.0),
                    right_point=jnp.where(valid[:, None], right_point, 0.0),
                    left_body=vertex_bodies[vertices[:, 0]].astype(jnp.int32),
                    right_body=right_body.astype(jnp.int32),
                    left_material=self.feature_material_rank[left],
                    right_material=self.feature_material_rank[right],
                    feature_margin=jnp.where(
                        valid, evaluation.distance.feature_margin, 0.0
                    ),
                    finite=jnp.asarray(evaluation.finite),
                )
            )

        def concatenate(*leaves: Array) -> Array:
            # Row arrays concatenate over batches; the scalar finiteness flag
            # is the conjunction of every batch.
            if leaves[0].ndim:
                return jnp.concatenate(leaves)
            return jnp.all(jnp.stack(leaves))

        return jax.tree.map(concatenate, *parts)

    def _contact_rates(
        self, rows: ArticulatedRouteGeometry, configuration: Array, velocity: Array, /
    ) -> Array:
        if self.geometry is None:
            raise RuntimeError("Contact rates require prepared geometry.")
        left = self.geometry.point_velocities(
            self.articulation, configuration, velocity, rows.left_body, rows.left_point
        )
        right = self.geometry.point_velocities(
            self.articulation, configuration, velocity, rows.right_body, rows.right_point
        )
        relative = left - right
        normal = jnp.sum(rows.normal * relative, axis=-1)
        tangent = (jnp.swapaxes(rows.tangent_basis, -1, -2) @ relative[..., None])[..., 0]
        local = jnp.concatenate((normal[:, None], tangent), axis=-1)
        return jnp.where(rows.valid[:, None], local, 0.0)

    def _contact_pullback(
        self, rows: ArticulatedRouteGeometry, configuration: Array, impulse: Array, /
    ) -> Array:
        if self.geometry is None:
            raise RuntimeError("Contact pullback requires prepared geometry.")
        local = jnp.where(rows.valid[:, None], impulse, 0.0)
        force = (
            rows.normal * local[:, :1] + (rows.tangent_basis @ local[:, 1:, None])[..., 0]
        )
        return self.geometry.point_force_pullback(
            self.articulation,
            configuration,
            jnp.concatenate((rows.left_body, rows.right_body)),
            jnp.concatenate((rows.left_point, rows.right_point)),
            jnp.concatenate((force, -force)),
        )

    def row_operator(
        self, rows: ArticulatedRouteGeometry | None, configuration: Array, /
    ) -> FunctionLinearOperator:
        """Mixed ``G(q)``: generalized velocity to contact blocks then stop rows."""

        dtype = self.dtype
        nv = self.articulation.nv
        contacts = self.contact_capacity
        stops = self.stop_capacity
        space = contact_cone_space(self.block_count, 3, dtype)

        def action(velocity: Array) -> Array:
            blocks: list[Array] = []
            if rows is not None:
                blocks.append(self._contact_rates(rows, configuration, velocity))
            if self.stops is not None:
                rate = self.stops.row_velocity(velocity)
                blocks.append(
                    jnp.zeros((stops, 3), dtype=velocity.dtype).at[:, 0].set(rate)
                )
            return jnp.concatenate(blocks, axis=0)

        def transpose_action(impulse: Array) -> Array:
            effort = jnp.zeros((nv,), dtype=impulse.dtype)
            if rows is not None:
                effort = effort + self._contact_pullback(
                    rows, configuration, impulse[:contacts]
                )
            if self.stops is not None:
                effort = effort + self.stops.generalized_impulse(impulse[contacts:, 0])
            return effort

        return FunctionLinearOperator(
            action,
            source=self.articulation.generalized_velocity_space,
            target=space,
            transpose_action=transpose_action,
            operator_id=f"{self.program_id}:rows",
        )

    def _block_arrays(
        self, rows: ArticulatedRouteGeometry | None, configuration: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Keys, validity, gaps, friction and material availability per block."""

        dtype = self.dtype
        keys: list[Array] = []
        valid: list[Array] = []
        gaps: list[Array] = []
        static: list[Array] = []
        dynamic: list[Array] = []
        available: list[Array] = []
        if rows is not None:
            keys.append(rows.route_keys.astype(jnp.int64))
            valid.append(rows.valid)
            gaps.append(rows.gap.astype(dtype))
            static.append(
                self.static_friction_table[rows.left_material, rows.right_material]
            )
            dynamic.append(
                self.dynamic_friction_table[rows.left_material, rows.right_material]
            )
            available.append(
                self.material_available[rows.left_material, rows.right_material]
            )
        if self.stops is not None:
            count = self.stop_capacity
            keys.append(jnp.arange(count, dtype=jnp.int64))
            valid.append(self.stops.row_valid)
            gaps.append(self.stops.gaps(configuration).astype(dtype))
            static.append(jnp.zeros((count,), dtype=dtype))
            dynamic.append(jnp.zeros((count,), dtype=dtype))
            available.append(jnp.ones((count,), dtype=jnp.bool_))
        return (
            jnp.concatenate(keys),
            jnp.concatenate(valid),
            jnp.concatenate(gaps),
            jnp.concatenate(static),
            jnp.concatenate(dynamic),
            jnp.concatenate(available),
        )

    def block_gap_tolerance(self, /) -> Array:
        """Per-block geometric gap floor (m for contacts, rad/m for stops)."""

        certificate = self.policy.certificate
        dtype = self.dtype
        parts: list[Array] = []
        if self.geometry is not None:
            parts.append(
                jnp.full(
                    (self.contact_capacity,),
                    certificate.contact_gap_tolerance,
                    dtype=dtype,
                )
            )
        if self.stops is not None:
            parts.append(
                jnp.where(
                    self.stop_hinge,
                    certificate.hinge_gap_tolerance,
                    certificate.prismatic_gap_tolerance,
                ).astype(dtype)
            )
        return jnp.concatenate(parts)

    def program(
        self,
        rows: ArticulatedRouteGeometry | None,
        relinearization: Array,
        start: Array,
        step_size: Array,
        free_velocity: Array,
        mass: PreparedArticulatedMass,
        certificate: ContactCertificatePolicy,
        /,
    ) -> tuple[ContactConeProgram, FunctionLinearOperator]:
        """Frozen-route mixed cone program relinearized at ``q_hat``.

        ``b_n = (g(q_hat) - G_n(q_hat)(q_hat - q0)) / h``; at ``q_hat = q1``
        the law normal velocity is exactly ``g(q1)/h``.
        """

        if self.layout is None:
            raise RuntimeError("A mixed contact program requires its block layout.")
        operator = self.row_operator(rows, relinearization)
        response = build_delassus_operator(operator, mass.inverse_mass_operator)
        keys, valid, gaps, static, dynamic, available = self._block_arrays(
            rows, relinearization
        )
        free = operator.mv(free_velocity)
        displacement = operator.mv(relinearization - start)[:, 0]
        bias = jnp.where(valid, (gaps - displacement) / step_size, 0.0)
        program = ContactConeProgram(
            response,
            free,
            0.0,
            dynamic,
            keys,
            valid,
            self.layout,
            certificate,
            self.program_id,
            static_friction=static,
            normal_bias=bias,
            mechanical_available=available,
            response_binding=mass.binding,
        )
        return program, operator

    def _observation(
        self,
        rows: ArticulatedRouteGeometry,
        geometry_revision: Array,
        material_revision: Array,
        /,
    ) -> ContactManifoldObservation:
        return ContactManifoldObservation(
            rows.route_keys.astype(jnp.int64),
            rows.valid,
            rows.normal,
            rows.tangent_basis,
            jnp.stack((rows.left_point, rows.right_point), axis=1),
            material_revision,
            geometry_revision,
            catalogue_id=self.catalogue_id,
        )

    # -------------------------------------------------------------- admission

    def admissibility(
        self, source: ArticulatedContactSource, /
    ) -> ArticulatedContactAdmissibility:
        """Certify one accepted configuration as a legal resting state."""

        dtype = self.dtype
        principal = source.configuration.astype(dtype)
        winding = source.winding
        lifted = self.coordinates.lift(principal, LiftedJointCoordinateState(winding))
        _, _, in_range = self.coordinates.split(lifted)
        principal_chart = jnp.all(
            ~self.coordinates.hinge_mask
            | (jnp.abs(principal) <= jnp.pi + self.coordinates.lift_resolution)
        )
        winding_valid = (
            in_range
            & principal_chart
            & jnp.all(jnp.abs(winding) <= self.coordinates.maximum_turns)
        )
        finite = jnp.all(jnp.isfinite(principal)) & jnp.all(jnp.isfinite(source.velocity))
        epoch: ContactCandidateEpoch | None = None
        path: AdmissiblePathEvidence | None = None
        contact_gap = jnp.asarray(jnp.inf, dtype=dtype)
        search_ok = jnp.asarray(True)
        narrow_ok = jnp.asarray(True)
        path_ok = jnp.asarray(True)
        invalid_geometry = jnp.asarray(False)
        if self.geometry is not None:
            trajectory = self._trajectory(
                lifted,
                jnp.zeros_like(source.velocity, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
                jnp.asarray(0, dtype=jnp.int64),
            )
            epoch = self._search(trajectory)
            if self.scene is None:
                raise RuntimeError("Contact geometry has no prepared scene.")
            path = certify_admissible_path(
                self.policy.path, self.scene, epoch, trajectory
            )
            rows = self.route_geometry(epoch, lifted)
            search_ok = epoch.successful
            narrow_ok = ~jnp.any(rows.failed) & rows.finite
            path_ok = path.successful
            invalid_geometry = path.status == int(AdmissiblePathStatus.INVALID_GEOMETRY)
            contact_gap = jnp.min(
                jnp.where(rows.valid, rows.gap, jnp.inf), initial=jnp.inf
            )
        stop_gap = jnp.asarray(jnp.inf, dtype=dtype)
        stop_ok = jnp.asarray(True)
        if self.stops is not None:
            gaps = self.stops.gaps(lifted)
            tolerance = self.block_gap_tolerance()[self.contact_capacity :]
            stop_ok = jnp.all(~self.stops.row_valid | (gaps >= -tolerance))
            stop_gap = jnp.min(
                jnp.where(self.stops.row_valid, gaps, jnp.inf), initial=jnp.inf
            )
        status = _first_failure(
            (
                (finite, ArticulatedContactStepStatus.INVALID_INPUT),
                (winding_valid, ArticulatedContactStepStatus.WINDING_RANGE_EXCEEDED),
                (search_ok, ArticulatedContactStepStatus.SEARCH_INCOMPLETE),
                (narrow_ok, ArticulatedContactStepStatus.NARROW_PHASE_FAILED),
                (~invalid_geometry, ArticulatedContactStepStatus.INVALID_GEOMETRY),
                (path_ok & stop_ok, ArticulatedContactStepStatus.INITIALLY_INADMISSIBLE),
            )
        )
        return ArticulatedContactAdmissibility(
            epoch,
            path,
            contact_gap,
            stop_gap,
            winding_valid,
            finite,
            status == int(ArticulatedContactStepStatus.SUCCESS),
            status,
        )

    # ---------------------------------------------------------------- segment

    def _root(
        self,
        source: ArticulatedContactSource,
        start: Array,
        step_size: Array,
        free_velocity: Array,
        mass: PreparedArticulatedMass,
        certificate: ContactCertificatePolicy,
        free_epoch: ContactCandidateEpoch | None,
        geometry_revision: Array,
        material_revision: Array,
        /,
    ) -> tuple[
        Array,
        Array,
        ContactCandidateEpoch | None,
        ContactCandidateEpoch | None,
        ArticulatedContactRootEvidence,
        Array,
    ]:
        """Bounded route epochs around the native fixed-point coupled root.

        Kept as one method on purpose: ``respond``, ``solve`` and ``warm_start``
        close over the same segment source, mass factor, certificate and
        revisions, and the outer ``while_loop`` body must see exactly those
        values so that every route epoch solves the identical frozen law.
        Splitting them into methods would only re-thread that shared numerical
        state through argument lists without separating any invariant.
        """

        dtype = self.dtype
        blocks = self.block_count
        policy = self.policy
        termination = NonlinearTermination(
            absolute_residual=policy.root_absolute_tolerance,
            relative_residual=policy.root_relative_tolerance,
            maximum_steps=policy.maximum_root_iterations,
        )
        method = FixedPointIteration(damping=policy.root_damping)

        def respond(
            epoch: ContactCandidateEpoch | None, trial_velocity: Array, warm: Array
        ) -> tuple[Array, Array]:
            """``(v_free + M0⁻¹ G(q_hat)* p, p)`` of the cone frozen at ``q_hat``.

            Every evaluation starts the cone iteration from the same history
            warm start, so the response is a deterministic function of the
            trial velocity and the outer fixed point is not polluted by the
            non-unique or ill-conditioned impulse directions of the cone.
            """

            relinearization = start + step_size * trial_velocity
            rows = None if epoch is None else self.route_geometry(epoch, relinearization)
            program, operator = self.program(
                rows,
                relinearization,
                start,
                step_size,
                free_velocity,
                mass,
                certificate,
            )
            result = solve_contact_cone(
                program, solver=policy.cone_solver, initial_impulse=warm
            )
            impulse = result.candidate_impulse
            velocity_update = mass.inverse_mass_operator.mv(
                dual_transpose(operator).mv(impulse)
            )
            return free_velocity + velocity_update, impulse

        def solve(
            epoch: ContactCandidateEpoch | None, velocity: Array, warm: Array
        ) -> tuple[Array, Array, Array, Array, Array]:
            def mapping(trial_velocity: Array, args: Any) -> Array:
                del args
                return respond(epoch, trial_velocity, warm)[0]

            problem = FixedPointProblem(
                mapping, problem_id=f"{self.program_id}:coupled-root"
            )
            result = method.solve(problem, velocity, termination=termination)
            _, impulse = respond(epoch, result.state, warm)
            return (
                result.state,
                impulse,
                result.status,
                result.diagnostics.iterations,
                result.diagnostics.final_residual_norm,
            )

        stop_warm = (
            jnp.zeros((self.stop_capacity, 3), dtype=dtype)
            .at[:, 0]
            .set(source.stop_impulse)
        )

        def warm_start(epoch: ContactCandidateEpoch | None) -> tuple[Array, Array, Array]:
            if epoch is None:
                return stop_warm, jnp.asarray(True), jnp.asarray(True)
            if source.contact_history is None:
                raise ValueError("A contact source requires its complete history.")
            rows = self.route_geometry(epoch, start)
            transition = update_contact_manifold(
                source.contact_history,
                self._observation(rows, geometry_revision, material_revision),
                policy=policy.manifold,
            )
            warm = jnp.concatenate((transition.warm_start.astype(dtype), stop_warm))
            narrow = ~jnp.any(rows.failed) & rows.finite
            return warm, narrow, transition.successful

        if free_epoch is None:
            warm, narrow, history = warm_start(None)
            velocity, impulse, status, iterations, residual = solve(
                None, free_velocity, warm
            )
            evidence = ArticulatedContactRootEvidence(
                jnp.asarray(True),
                status.astype(jnp.int32),
                iterations.astype(jnp.int32),
                residual.astype(dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(True),
                jnp.asarray(True),
                narrow,
                history,
            )
            return (
                velocity,
                impulse,
                None,
                None,
                evidence,
                jnp.asarray(0, dtype=jnp.int64),
            )

        initial = _RouteCarry(
            jnp.asarray(0, dtype=jnp.int32),
            free_epoch,
            free_epoch,
            free_epoch,
            free_velocity,
            jnp.zeros((blocks, 3), dtype=dtype),
            jnp.asarray(int(NonlinearStatus.ITERATING), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(False),
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int64),
        )

        def condition(carry: _RouteCarry) -> Array:
            return ~carry.done & (carry.rebuild <= policy.maximum_route_rebuilds)

        def body(carry: _RouteCarry) -> _RouteCarry:
            warm, narrow, history = warm_start(carry.epoch)
            velocity, impulse, status, iterations, residual = solve(
                carry.epoch, carry.velocity, warm
            )
            revision = (carry.rebuild + 1).astype(jnp.int64)
            coverage = self._search(
                self._trajectory(start, velocity, step_size, revision)
            )
            converged = status == int(NonlinearStatus.SUCCESS)
            stable = _routes_cover(carry.epoch, coverage)
            search = coverage.successful
            done = stable | ~converged | ~search | ~narrow | ~history
            epoch = tree_where(stable | ~converged, carry.epoch, coverage)
            return _RouteCarry(
                carry.rebuild + 1,
                epoch,
                carry.epoch,
                coverage,
                velocity,
                impulse,
                status.astype(jnp.int32),
                carry.iterations + iterations.astype(jnp.int32),
                residual.astype(dtype),
                stable,
                search,
                narrow,
                history,
                done,
                revision,
            )

        final = jax.lax.while_loop(condition, body, initial)
        evidence = ArticulatedContactRootEvidence(
            jnp.asarray(True),
            final.status,
            final.iterations,
            final.residual,
            final.rebuild - 1,
            final.stable,
            final.search,
            final.narrow,
            final.history,
        )
        return (
            final.velocity,
            final.impulse,
            final.solved,
            final.coverage,
            evidence,
            final.revision,
        )

    def segment(
        self,
        source: ArticulatedContactSource,
        step_size: Array,
        fraction: Array,
        actuation: Array,
        gravity: Array,
        geometry_revision: Array,
        material_revision: Array,
        /,
    ) -> tuple[ArticulatedContactSource, ArticulatedContactSegmentEvidence]:
        """Attempt one segment from a provisional source; return target and evidence.

        Deliberately straight-line: the stages (native predictor, exact
        contact-free attempt, coupled root, endpoint recertification, energy
        ledger, manifold history, first-failure status, evidence record) share
        one set of segment locals, and the evidence record must report exactly
        the values each certificate used. The physical acceptance order is the
        ``_first_failure`` table near the end; extracting stages would require
        passing those intermediate arrays back and forth and would hide that
        one-to-one correspondence between certificates and evidence fields.
        """

        dtype = self.dtype
        policy = self.policy
        articulation = self.articulation
        principal = source.configuration
        velocity0 = source.velocity
        winding = LiftedJointCoordinateState(source.winding)
        start = self.coordinates.lift(principal, winding)
        effort, effort_ok = self.effort(actuation, velocity0)
        free: ReducedSemiImplicitVelocityEulerStepResult = (
            reduced_semi_implicit_velocity_euler_step(
                articulation,
                ReducedArticulationState(principal, velocity0),
                effort,
                gravity,
                step_size,
                policy=policy.smooth,
            )
        )
        mass = free.dynamics.mass
        free_velocity = free.candidate_state.velocity
        free_principal = free.candidate_state.configuration
        input_ok = (
            effort_ok
            & jnp.all(jnp.isfinite(principal))
            & jnp.all(jnp.isfinite(velocity0))
            & jnp.all(jnp.isfinite(gravity))
            & jnp.isfinite(step_size)
            & (step_size > 0.0)
        )
        dynamics_ok = free.dynamics.successful & mass.successful
        certificate = policy.certificate.segment_cone_certificate(step_size, fraction)
        block_tolerance = (
            self.block_gap_tolerance() if self.block_count else jnp.zeros((0,), dtype)
        )
        free_end = start + step_size * free_velocity
        free_advance = self.coordinates.advance(
            principal, winding, free_velocity, step_size
        )

        # Exact contact-free attempt on the native smooth trajectory.
        free_epoch: ContactCandidateEpoch | None = None
        free_path: AdmissiblePathEvidence | None = None
        search_ok = jnp.asarray(True)
        narrow_ok = jnp.asarray(True)
        free_path_ok = jnp.asarray(True)
        initially_inadmissible = jnp.asarray(False)
        invalid_geometry = jnp.asarray(False)
        free_rows: ArticulatedRouteGeometry | None = None
        if self.geometry is not None:
            if self.scene is None:
                raise RuntimeError("Contact geometry has no prepared scene.")
            trajectory = self._trajectory(
                start, free_velocity, step_size, jnp.asarray(0, dtype=jnp.int64)
            )
            free_epoch = self._search(trajectory)
            free_path = certify_admissible_path(
                policy.path, self.scene, free_epoch, trajectory
            )
            source_rows = self.route_geometry(free_epoch, start)
            free_rows = self.route_geometry(free_epoch, free_end)
            search_ok = free_epoch.successful
            narrow_ok = ~jnp.any(source_rows.failed) & source_rows.finite
            free_path_ok = (
                free_path.successful & ~jnp.any(free_rows.failed) & free_rows.finite
            )
            initially_inadmissible = free_path.status == int(
                AdmissiblePathStatus.INITIALLY_INADMISSIBLE
            )
            invalid_geometry = (
                free_path.status == int(AdmissiblePathStatus.INVALID_GEOMETRY)
            ) | (free_path.status == int(AdmissiblePathStatus.NONFINITE_INPUT))
        stop_initial_ok = jnp.asarray(True)
        stop_free_ok = jnp.asarray(True)
        if self.stops is not None:
            stop_tolerance = block_tolerance[self.contact_capacity :]
            stop_initial_ok = jnp.all(
                ~self.stops.row_valid | (self.stops.gaps(start) >= -stop_tolerance)
            )
            stop_free_ok = jnp.all(
                ~self.stops.row_valid
                | (self.stops.path_minimum_gaps(start, free_end) >= -stop_tolerance)
            )
        zero_impulse = jnp.zeros((self.block_count, 3), dtype=dtype)
        if self.block_count:
            free_program, _ = self.program(
                free_rows,
                free_end,
                start,
                step_size,
                free_velocity,
                mass,
                certificate,
            )
            free_law_ok = certify_contact_cone_impulse(
                free_program, zero_impulse
            ).successful
        else:
            free_law_ok = jnp.asarray(True)
        terminal = (
            ~input_ok
            | ~dynamics_ok
            | ~search_ok
            | ~narrow_ok
            | initially_inadmissible
            | invalid_geometry
            | ~stop_initial_ok
        )
        bypass = (
            ~terminal
            & free.successful
            & free_path_ok
            & stop_free_ok
            & free_law_ok
            & free_advance.successful
        )

        def contact_branch(
            _: None,
        ) -> tuple[
            Array,
            Array,
            ContactCandidateEpoch | None,
            ContactCandidateEpoch | None,
            AdmissiblePathEvidence | None,
            ArticulatedContactRootEvidence,
        ]:
            velocity, impulse, route_epoch, coverage_epoch, root, revision = self._root(
                source,
                start,
                step_size,
                free_velocity,
                mass,
                certificate,
                free_epoch,
                geometry_revision,
                material_revision,
            )
            path: AdmissiblePathEvidence | None = None
            if coverage_epoch is not None:
                if self.scene is None:
                    raise RuntimeError("Contact coverage has no prepared scene.")
                path = certify_admissible_path(
                    policy.path,
                    self.scene,
                    coverage_epoch,
                    self._trajectory(start, velocity, step_size, revision),
                )
            return velocity, impulse, route_epoch, coverage_epoch, path, root

        def bypass_branch(
            _: None,
        ) -> tuple[
            Array,
            Array,
            ContactCandidateEpoch | None,
            ContactCandidateEpoch | None,
            AdmissiblePathEvidence | None,
            ArticulatedContactRootEvidence,
        ]:
            root = ArticulatedContactRootEvidence(
                jnp.asarray(False),
                jnp.asarray(_NOT_ATTEMPTED, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(True),
                search_ok,
                narrow_ok,
                jnp.asarray(True),
            )
            return free_velocity, zero_impulse, free_epoch, free_epoch, free_path, root

        if self.block_count:
            velocity1, impulse, route_epoch, coverage_epoch, path, root = jax.lax.cond(
                bypass | terminal, bypass_branch, contact_branch, None
            )
        else:
            velocity1, impulse, route_epoch, coverage_epoch, path, root = bypass_branch(
                None
            )

        # Independent recertification of the actual endpoint and trajectory.
        end = start + step_size * velocity1
        advance = self.coordinates.advance(principal, winding, velocity1, step_size)
        end_principal = advance.configuration
        rows: ArticulatedRouteGeometry | None = (
            None if route_epoch is None else self.route_geometry(route_epoch, end)
        )
        if self.block_count:
            program, operator = self.program(
                rows, end, start, step_size, free_velocity, mass, certificate
            )
            pullback = dual_transpose(operator).mv(impulse)
            response = mass.inverse_mass_operator.mv(pullback)
            # Actual endpoint law: with the generalized free velocity
            # ``v1 - M0⁻¹G(q1)* p`` the program's physical velocity is exactly
            # ``G(q1) v1`` and its normal law velocity ``g(q1)/h``, so this
            # certificate measures the accepted state rather than the
            # response map's own prediction ``v_free + M0⁻¹G* p``.
            endpoint_program, _ = self.program(
                rows, end, start, step_size, velocity1 - response, mass, certificate
            )
            response_cone = certify_contact_cone_impulse(program, impulse)
            cone = certify_contact_cone_impulse(endpoint_program, impulse)
            branch, margin = classify_contact_cone_branches(
                endpoint_program, impulse, cone
            )
            physical = operator.mv(velocity1)
            law = cone.law_velocity
            keys, block_valid, gaps, _, _, _ = self._block_arrays(rows, end)
            cone_ok = cone.successful & response_cone.successful
            cone_certificate: ContactConeCertificate | None = cone
            response_certificate: ContactConeCertificate | None = response_cone
        else:
            branch = jnp.zeros((0,), dtype=jnp.int8)
            margin = jnp.zeros((0,), dtype=dtype)
            physical = jnp.zeros((0, 3), dtype=dtype)
            law = jnp.zeros((0, 3), dtype=dtype)
            pullback = jnp.zeros_like(velocity1)
            keys = jnp.zeros((0,), dtype=jnp.int64)
            block_valid = jnp.zeros((0,), dtype=jnp.bool_)
            gaps = jnp.zeros((0,), dtype=dtype)
            cone_ok = jnp.asarray(True)
            cone_certificate = None
            response_certificate = None
        correction = velocity1 - free_velocity
        mass_correction = mass.apply_mass(articulation, principal, correction)
        residual_effort = mass_correction.candidate - pullback
        residual_velocity = mass.apply_inverse(articulation, principal, residual_effort)
        momentum_energy = 0.5 * jnp.abs(
            jnp.vdot(residual_effort, residual_velocity.candidate)
        )
        mass_actions_ok = mass_correction.successful & residual_velocity.current
        momentum_ok = (
            momentum_energy <= fraction * policy.certificate.momentum_energy_tolerance
        )

        gaps_ok = jnp.all(~block_valid | (gaps >= -block_tolerance))
        stop_path_gap = jnp.asarray(jnp.inf, dtype=dtype)
        stop_path_ok = jnp.asarray(True)
        if self.stops is not None:
            path_gaps = self.stops.path_minimum_gaps(start, end)
            stop_path_ok = jnp.all(
                ~self.stops.row_valid
                | (path_gaps >= -block_tolerance[self.contact_capacity :])
            )
            stop_path_gap = jnp.min(
                jnp.where(self.stops.row_valid, path_gaps, jnp.inf), initial=jnp.inf
            )
        path_ok = stop_path_ok & (jnp.asarray(True) if path is None else path.successful)

        # Mechanical-energy ledger (plan section 3.4) with native energy owners.
        energy0 = reduced_energy(articulation, principal, velocity0, gravity)
        energy1 = reduced_energy(articulation, end_principal, velocity1, gravity)
        energy_free = reduced_energy(articulation, free_principal, free_velocity, gravity)

        def frozen_kinetic(value: Array) -> Array:
            return 0.5 * jnp.vdot(
                value, mass.apply_mass(articulation, principal, value).candidate
            )

        kinetic_free_m0 = frozen_kinetic(free_velocity)
        kinetic_one_m0 = frozen_kinetic(velocity1)
        source_work = step_size * jnp.vdot(effort, velocity1)
        free_work = free.diagnostics.applied_work
        contact_work = jnp.sum(impulse * physical)
        correction_loss = 0.5 * jnp.vdot(correction, mass_correction.candidate)
        contact_kinetic_change = kinetic_one_m0 - kinetic_free_m0
        predictor_defect = free.diagnostics.energy_defect
        geometry_residual = (
            (energy1.potential - energy_free.potential)
            + (energy1.kinetic - kinetic_one_m0)
            - (energy_free.kinetic - kinetic_free_m0)
        )
        work_residual = free_work - source_work
        identity_residual = (
            energy1.total - energy0.total - source_work - contact_work + correction_loss
        ) - (predictor_defect + geometry_residual + work_residual)
        friction_work = (
            jnp.asarray(0.0, dtype=dtype)
            if cone_certificate is None
            else cone_certificate.physical_tangential_work
        )
        surplus = energy1.total - energy0.total - source_work
        contact_segment = ~bypass
        eps = jnp.finfo(dtype).eps
        roundoff = (
            64.0
            * eps
            * (
                jnp.abs(energy0.total)
                + jnp.abs(energy1.total)
                + jnp.abs(energy_free.total)
                + jnp.abs(source_work)
                + jnp.abs(free_work)
                + jnp.abs(contact_work)
                + jnp.abs(correction_loss)
                + jnp.abs(kinetic_one_m0)
                + jnp.abs(kinetic_free_m0)
            )
        )
        certificate_policy = policy.certificate
        contact_work_ok = (
            contact_work <= fraction * certificate_policy.contact_work_tolerance
        )
        contact_energy_ok = (
            contact_kinetic_change
            <= fraction * certificate_policy.contact_energy_tolerance
        )
        loss_ok = correction_loss >= -roundoff
        ledger_ok = jnp.abs(identity_residual) <= (
            fraction * certificate_policy.ledger_tolerance + roundoff
        )
        predictor_ok = free.diagnostics.step_size_within_bound & (
            jnp.abs(predictor_defect) <= free.diagnostics.allowed_energy_defect
        )
        surplus_ok = ~contact_segment | (
            surplus <= fraction * certificate_policy.energy_surplus_tolerance + roundoff
        )
        discretization = jnp.where(
            contact_segment, jnp.abs(geometry_residual) + jnp.abs(work_residual), 0.0
        )
        discretization_ok = (
            discretization <= fraction * certificate_policy.discretization_energy_budget
        )
        energy_ok = (
            contact_energy_ok
            & loss_ok
            & ledger_ok
            & predictor_ok
            & surplus_ok
            & discretization_ok
            & energy0.successful
            & energy1.successful
            & energy_free.successful
        )
        ledger = ArticulatedContactEnergyLedger(
            energy0.total,
            energy1.total,
            source_work,
            contact_work,
            correction_loss,
            contact_kinetic_change,
            predictor_defect,
            geometry_residual,
            work_residual,
            identity_residual,
            friction_work,
            jnp.maximum(contact_work, 0.0),
            jnp.maximum(contact_kinetic_change, 0.0),
            jnp.maximum(friction_work, 0.0),
            jnp.where(contact_segment, jnp.maximum(surplus, 0.0), 0.0),
            discretization,
            roundoff,
        )

        # Mechanical history: observe the accepted endpoint, record responses.
        contacts = self.contact_capacity
        history_ok = root.warm_history_successful
        target_history = source.contact_history
        activation = jnp.zeros((self.block_count,), dtype=jnp.bool_)
        feature_margin = jnp.zeros((contacts,), dtype=dtype)
        if rows is not None:
            if source.contact_history is None:
                raise ValueError("A contact source requires its complete history.")
            transition = update_contact_manifold(
                source.contact_history,
                self._observation(rows, geometry_revision, material_revision),
                policy=policy.manifold,
            )
            found, origin, _, _ = match_route_keys(
                rows.route_keys.astype(jnp.int64),
                rows.valid,
                source.contact_history.route_keys,
                source.contact_history.occupied,
            )
            inherited = jnp.where(
                found,
                source.contact_history.branch[origin],
                int(ContactConeBranch.UNCLASSIFIED),
            )
            contact_branch_ = branch[:contacts]
            new_active = _is_active(contact_branch_)
            old_active = _is_active(inherited)
            activation = activation.at[:contacts].set(
                rows.valid & (new_active | old_active) & (contact_branch_ != inherited)
            )
            target_history = record_contact_response(
                transition.candidate,
                rows.route_keys.astype(jnp.int64),
                impulse[:contacts],
                contact_branch_,
                physical[:contacts, 1:],
                step_size=step_size,
                valid=rows.valid,
            )
            history_ok = history_ok & transition.successful
            feature_margin = rows.feature_margin
        if self.stops is not None:
            stop_branch = branch[contacts:]
            activation = activation.at[contacts:].set(
                self.stops.row_valid
                & (_is_active(stop_branch) | _is_active(source.stop_branch))
                & (stop_branch != source.stop_branch)
            )
            target_stop_impulse = impulse[contacts:, 0]
            target_stop_branch = stop_branch.astype(jnp.int8)
        else:
            target_stop_impulse = source.stop_impulse
            target_stop_branch = source.stop_branch

        finite = (
            jnp.all(jnp.isfinite(velocity1))
            & jnp.all(jnp.isfinite(impulse))
            & jnp.all(jnp.isfinite(end))
            & jnp.isfinite(identity_residual)
        )
        route_feature_ok = jnp.asarray(True)
        if rows is not None:
            route_feature_ok = ~jnp.any(rows.failed) & rows.finite
        search_final_ok = root.search_successful & (
            jnp.asarray(True) if coverage_epoch is None else coverage_epoch.successful
        )
        root_attempted = root.attempted
        root_ok = ~root_attempted | (root.status == int(NonlinearStatus.SUCCESS))
        status = _first_failure(
            (
                (input_ok, ArticulatedContactStepStatus.INVALID_INPUT),
                (dynamics_ok, ArticulatedContactStepStatus.FREE_DYNAMICS_FAILED),
                (
                    search_ok & search_final_ok,
                    ArticulatedContactStepStatus.SEARCH_INCOMPLETE,
                ),
                (
                    narrow_ok & root.narrow_phase_successful,
                    ArticulatedContactStepStatus.NARROW_PHASE_FAILED,
                ),
                (
                    ~initially_inadmissible & stop_initial_ok,
                    ArticulatedContactStepStatus.INITIALLY_INADMISSIBLE,
                ),
                (~invalid_geometry, ArticulatedContactStepStatus.INVALID_GEOMETRY),
                (
                    advance.within_range,
                    ArticulatedContactStepStatus.WINDING_RANGE_EXCEEDED,
                ),
                (history_ok, ArticulatedContactStepStatus.HISTORY_FAILED),
                (mass_actions_ok, ArticulatedContactStepStatus.MASS_RESPONSE_FAILED),
                (root_ok, ArticulatedContactStepStatus.ROOT_NOT_CONVERGED),
                (
                    root.routes_stable,
                    ArticulatedContactStepStatus.ROUTE_REBUILD_EXHAUSTED,
                ),
                (
                    path_ok & route_feature_ok,
                    ArticulatedContactStepStatus.PATH_NOT_ADMISSIBLE,
                ),
                (cone_ok, ArticulatedContactStepStatus.CONE_FAILED),
                (
                    momentum_ok & advance.successful,
                    ArticulatedContactStepStatus.MOMENTUM_INVALID,
                ),
                (gaps_ok, ArticulatedContactStepStatus.GAP_INVALID),
                (contact_work_ok, ArticulatedContactStepStatus.CONTACT_WORK_INVALID),
                (energy_ok & finite, ArticulatedContactStepStatus.ENERGY_INVALID),
            )
        )
        successful = status == int(ArticulatedContactStepStatus.SUCCESS)
        block_margin = jnp.min(jnp.where(block_valid, margin, jnp.inf), initial=jnp.inf)
        branch_qualified = jnp.all(
            ~block_valid | (branch != int(ContactConeBranch.UNCLASSIFIED))
        ) & (block_margin > policy.derivative_margin_factor)
        feature_qualified = (
            jnp.asarray(True)
            if rows is None
            else jnp.all(~rows.valid | (rows.feature_margin > 0.0))
        )
        fixed_branch_candidate = (
            successful
            & branch_qualified
            & feature_qualified
            & ~jnp.any(activation)
            & jnp.all(advance.winding_increment == 0)
        )
        impulse_tolerance = (
            cone_certificate.impulse_tolerance
            if cone_certificate is not None
            else jnp.zeros((0,), dtype)
        )
        active = block_valid & (impulse[:, 0] > impulse_tolerance)
        target = ArticulatedContactSource(
            end_principal,
            advance.candidate_state.winding,
            velocity1,
            target_history,
            target_stop_impulse.astype(dtype),
            target_stop_branch,
        )
        accepted_target = tree_where(successful, target, source)
        evidence = ArticulatedContactSegmentEvidence(
            step_size,
            fraction,
            bypass,
            free.status,
            free.successful,
            mass.status,
            root,
            route_epoch,
            coverage_epoch,
            path,
            stop_path_gap,
            advance.winding_increment,
            advance.lift_residual,
            keys,
            block_valid,
            impulse,
            branch,
            margin,
            activation,
            gaps,
            physical,
            law,
            cone_certificate,
            response_certificate,
            momentum_energy,
            momentum_ok,
            gaps_ok,
            cone_ok,
            contact_work_ok,
            energy_ok,
            history_ok,
            feature_margin,
            fixed_branch_candidate,
            ledger,
            jnp.sum(active[:contacts], dtype=jnp.int32),
            jnp.sum(active[contacts:], dtype=jnp.int32),
            finite,
            ~successful & _retryable(status),
            successful,
            status,
        )
        return accepted_target, evidence

    # ---------------------------------------------------------------- request

    def step(
        self,
        source: ArticulatedContactSource,
        duration: ConvertibleToArray,
        actuation: ConvertibleToArray,
        gravity: ConvertibleToArray,
        /,
        *,
        geometry_revision: ConvertibleToArray = 0,
        material_revision: ConvertibleToArray = 0,
    ) -> ArticulatedContactStepResult:
        """Advance one case over the full requested interval or roll back.

        Inputs are normalized at this boundary and the request executes in one
        module-level compiled body, so repeated requests with the same step
        structure and shapes reuse one executable with every input dynamic.
        """

        dtype = self.dtype
        return _compiled_contact_step(
            self,
            source,
            jnp.asarray(duration, dtype=dtype).reshape(()),
            jnp.asarray(actuation, dtype=dtype).reshape((self.actuation_size,)),
            jnp.asarray(gravity, dtype=dtype).reshape((3,)),
            jnp.asarray(geometry_revision, dtype=jnp.int64).reshape(()),
            jnp.asarray(material_revision, dtype=jnp.int64).reshape(()),
        )

    def _execute(
        self,
        source: ArticulatedContactSource,
        requested: Array,
        command: Array,
        gravity_: Array,
        geometry_revision_: Array,
        material_revision_: Array,
        /,
    ) -> ArticulatedContactStepResult:
        """Single-case request body; called only through the compiled entry.

        This is the atomic request state machine (dyadic segment schedule,
        retry/terminal/minimum/exhaustion transitions, full-coverage commit and
        request budgets). Its ``_RequestCarry`` transitions and the final
        record are kept together so the commit decision and every reported
        budget are visibly derived from one carry.
        """

        dtype = self.dtype
        policy = self.policy
        depth = policy.maximum_subdivision_depth
        units = 1 << depth
        attempts = policy.maximum_segment_attempts

        def attempt(
            provisional: ArticulatedContactSource, covered: Array, level: Array
        ) -> tuple[ArticulatedContactSource, ArticulatedContactSegmentEvidence, Array]:
            span = jnp.minimum(
                jnp.left_shift(jnp.asarray(1, dtype=jnp.int32), depth - level),
                units - covered,
            )
            fraction = span.astype(dtype) / units
            target, evidence = self.segment(
                provisional,
                requested * fraction,
                fraction,
                command,
                gravity_,
                geometry_revision_,
                material_revision_,
            )
            return target, evidence, span

        def advance(
            carry: _RequestCarry,
            target: ArticulatedContactSource,
            evidence: ArticulatedContactSegmentEvidence,
            span: Array,
        ) -> _RequestCarry:
            accepted = evidence.successful
            retry = evidence.retryable
            next_level = jnp.where(
                accepted,
                jnp.maximum(carry.level - 1, 0),
                jnp.where(retry, carry.level + 1, carry.level),
            )
            finer = requested * (
                jnp.left_shift(
                    jnp.asarray(1, dtype=jnp.int32), jnp.maximum(depth - next_level, 0)
                ).astype(dtype)
                / units
            )
            minimum = retry & (
                (next_level > depth) | (finer < policy.minimum_segment_duration)
            )
            terminal = ~accepted & ~retry
            covered = carry.covered + jnp.where(accepted, span, 0)
            count = carry.count + 1
            complete = covered == units
            exhausted = ~complete & ~terminal & ~minimum & (count >= attempts)
            status = jnp.where(
                terminal,
                evidence.status,
                jnp.where(
                    minimum,
                    int(ArticulatedContactStepStatus.MINIMUM_SEGMENT_REACHED),
                    jnp.where(
                        exhausted,
                        int(ArticulatedContactStepStatus.SUBSTEP_BUDGET_EXHAUSTED),
                        int(ArticulatedContactStepStatus.SUCCESS),
                    ),
                ),
            ).astype(jnp.int32)
            index = carry.count
            return _RequestCarry(
                target,
                covered,
                jnp.minimum(next_level, depth),
                count,
                carry.accepted + accepted.astype(jnp.int32),
                carry.rebuilds + evidence.root.route_rebuilds,
                carry.iterations + evidence.root.iterations,
                carry.ledger.accumulate(evidence.ledger, accepted),
                evidence,
                carry.attempt_status.at[index].set(evidence.status),
                carry.attempt_fraction.at[index].set(span.astype(dtype) / units),
                carry.attempt_bypass.at[index].set(evidence.free_bypass),
                carry.attempt_accepted.at[index].set(accepted),
                status,
                complete | terminal | minimum | exhausted,
            )

        level0 = jnp.asarray(0, dtype=jnp.int32)
        covered0 = jnp.asarray(0, dtype=jnp.int32)
        target, evidence, span = attempt(source, covered0, level0)
        empty = ArticulatedContactEnergyLedger(
            evidence.ledger.initial_energy,
            evidence.ledger.initial_energy,
            *(jnp.zeros((), dtype=dtype) for _ in range(15)),
        )
        initial = _RequestCarry(
            source,
            covered0,
            level0,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            empty,
            evidence,
            jnp.full((attempts,), _NOT_ATTEMPTED, dtype=jnp.int32),
            jnp.zeros((attempts,), dtype=dtype),
            jnp.zeros((attempts,), dtype=jnp.bool_),
            jnp.zeros((attempts,), dtype=jnp.bool_),
            jnp.asarray(int(ArticulatedContactStepStatus.SUCCESS), dtype=jnp.int32),
            jnp.asarray(False),
        )
        first = advance(initial, target, evidence, span)

        def condition(carry: _RequestCarry) -> Array:
            return ~carry.done

        def body(carry: _RequestCarry) -> _RequestCarry:
            target, evidence, span = attempt(carry.source, carry.covered, carry.level)
            return advance(carry, target, evidence, span)

        final = jax.lax.while_loop(condition, body, first)
        ledger = final.ledger
        certificate = policy.certificate
        request_identity = (
            ledger.final_energy
            - ledger.initial_energy
            - ledger.source_work
            - ledger.contact_work
            + ledger.correction_loss
            - (ledger.predictor_defect + ledger.geometry_residual + ledger.work_residual)
        )
        budgets_ok = (
            (ledger.positive_contact_work <= certificate.contact_work_tolerance)
            & (
                ledger.positive_contact_kinetic_change
                <= certificate.contact_energy_tolerance
            )
            & (ledger.positive_friction_work <= certificate.friction_work_tolerance)
            & (
                ledger.positive_energy_surplus
                <= certificate.energy_surplus_tolerance + ledger.roundoff_allowance
            )
            & (ledger.discretization_charge <= certificate.discretization_energy_budget)
            & (
                jnp.abs(request_identity)
                <= certificate.ledger_tolerance + ledger.roundoff_allowance
            )
        )
        complete = final.covered == units
        status = jnp.where(
            (final.status == int(ArticulatedContactStepStatus.SUCCESS))
            & complete
            & ~budgets_ok,
            int(ArticulatedContactStepStatus.REQUEST_BUDGET_EXCEEDED),
            final.status,
        ).astype(jnp.int32)
        successful = complete & (status == int(ArticulatedContactStepStatus.SUCCESS))
        accepted = tree_where(successful, final.source, source)
        single = (final.count == 1) & successful
        free_bypass = successful & jnp.all(
            jnp.where(final.attempt_accepted, final.attempt_bypass, True)
        )
        return ArticulatedContactStepResult(
            source,
            final.source,
            accepted,
            requested,
            requested * final.covered.astype(dtype) / units,
            final.covered.astype(dtype) / units,
            final.accepted,
            final.count,
            final.rebuilds,
            final.iterations,
            final.attempt_status,
            final.attempt_fraction,
            final.attempt_bypass,
            final.attempt_accepted,
            final.last,
            ledger,
            budgets_ok,
            free_bypass,
            single & final.last.fixed_branch_candidate,
            ArticulatedContactStepBinding(
                eqx.filter(self, eqx.is_array),
                command,
                gravity_,
                geometry_revision_,
                material_revision_,
                self.step_id,
            ),
            successful,
            status,
        )

    # ------------------------------------------------------------ derivatives

    def _fixed_branch_predictor(
        self,
        result: ArticulatedContactStepResult,
        /,
    ) -> _Predictor:
        """Native smooth predictor of the result's segment as a function of inputs."""

        source = result.source

        def predictor(
            step: PreparedArticulatedContactStep,
            start: Array,
            velocity0: Array,
            actuation: Array,
            gravity: Array,
        ) -> tuple[Array, PreparedArticulatedMass, ContactCertificatePolicy, Array]:
            step_size = result.requested_duration
            principal = (
                start - 2.0 * jnp.pi * step.coordinates.hinge_mask * source.winding
            )
            effort, _ = step.effort(actuation, velocity0)
            free = reduced_semi_implicit_velocity_euler_step(
                step.articulation,
                ReducedArticulationState(principal, velocity0),
                effort,
                gravity,
                step_size,
                policy=step.policy.smooth,
            )
            certificate = step.policy.certificate.segment_cone_certificate(
                step_size, jnp.asarray(1.0, dtype=step_size.dtype)
            )
            return (
                free.candidate_state.velocity,
                free.dynamics.mass,
                certificate,
                principal,
            )

        return predictor

    def _binding_current(
        self,
        result: ArticulatedContactStepResult,
        command: Array,
        gravity: Array,
        /,
    ) -> Array:
        """Exact equality of the current step and force inputs with the result."""

        binding = result.binding
        current = eqx.filter(self, eqx.is_array)
        if binding.step_id != self.step_id or jax.tree.structure(
            current
        ) != jax.tree.structure(binding.step_arrays):
            return jnp.asarray(False)
        pairs = list(
            zip(
                jax.tree.leaves(current),
                jax.tree.leaves(binding.step_arrays),
                strict=True,
            )
        )
        pairs.append((command, binding.actuation))
        pairs.append((gravity, binding.gravity))
        return jnp.all(
            jnp.stack(
                [jnp.asarray(jnp.array_equal(left, right)) for left, right in pairs]
            )
        )

    def _derivative_inputs(
        self,
        result: ArticulatedContactStepResult,
        actuation: ConvertibleToArray,
        gravity: ConvertibleToArray,
        /,
    ) -> tuple[
        Array,
        Array,
        Array,
        Array,
        PreparedArticulatedContactStep,
        PreparedArticulatedContactStep,
    ]:
        """Normalized command, gravity, binding flag, lifted q0 and step partition."""

        dtype = self.dtype
        command = jnp.asarray(actuation, dtype=dtype).reshape((self.actuation_size,))
        gravity_ = jnp.asarray(gravity, dtype=dtype).reshape((3,))
        source = result.source
        start = self.coordinates.lift(
            source.configuration, LiftedJointCoordinateState(source.winding)
        )
        dynamic, static = eqx.partition(self, eqx.is_inexact_array)
        return (
            command,
            gravity_,
            self._binding_current(result, command, gravity_),
            start,
            dynamic,
            static,
        )

    def _coupled_fixed_branch(
        self,
        result: ArticulatedContactStepResult,
        parameters: _StepParameters,
        static: PreparedArticulatedContactStep,
        regularity: SVDSolvePolicy,
        /,
        *,
        friction_derivative: bool,
    ) -> _CoupledFixedBranch:
        """Prepare and rank-certify the coupled fixed-branch residual.

        Unknowns are ``z = [v1, p / scale]``. The residual stacks the momentum
        equation ``v1 - v_free - M0⁻¹G(q1)* p`` with the cone owner's
        fixed-branch law residual evaluated at the accepted endpoint law
        velocity: normal ``g(q1)/h`` and tangential ``G_t(q1) v1`` with
        ``q1 = q0 + h v1``. Its Jacobian acts only through one prepared JVP/VJP
        pair, and the native rank owner certifies it under the caller's
        bounded SVD policy (refusing over-budget work before materializing).
        """

        segment = result.last_segment
        epoch = segment.route_epoch
        step_size = result.requested_duration
        nv = self.articulation.nv
        dtype = self.dtype
        predictor = self._fixed_branch_predictor(result)
        dynamic, start, velocity0, command, gravity = parameters
        velocity_star = result.candidate.velocity
        impulse_star = segment.impulse
        shape = impulse_star.shape

        # Fix the branch from the certificate of the accepted endpoint law.
        step_star = eqx.combine(dynamic, static)
        free_star, mass_star, certificate_star, _ = predictor(
            step_star, start, velocity0, command, gravity
        )
        end_star = start + step_size * velocity_star
        rows_star = None if epoch is None else self.route_geometry(epoch, end_star)
        operator_star = self.row_operator(rows_star, end_star)
        response_star = mass_star.inverse_mass_operator.mv(
            dual_transpose(operator_star).mv(impulse_star)
        )
        endpoint_program, _ = self.program(
            rows_star,
            end_star,
            start,
            step_size,
            velocity_star - response_star,
            mass_star,
            certificate_star,
        )
        endpoint_result = _rebound_cone_result(
            endpoint_program, impulse_star, self.policy
        )
        fixed = fix_contact_cone_branch(endpoint_program, endpoint_result)
        branch_admitted = contact_cone_fixed_branch_admitted(
            endpoint_program,
            fixed,
            margin_factor=self.policy.derivative_margin_factor,
            friction_derivative=friction_derivative,
        )
        scale = fixed.scale

        def residual(point: tuple[Array, _StepParameters]) -> Array:
            unknown, (dynamic_, start_, velocity0_, command_, gravity_) = point
            velocity = unknown[:nv]
            scaled_impulse = unknown[nv:].reshape(shape)
            step = eqx.combine(dynamic_, static)
            free_velocity, mass, _, _ = predictor(
                step, start_, velocity0_, command_, gravity_
            )
            end = start_ + step_size * velocity
            rows = None if epoch is None else step.route_geometry(epoch, end)
            operator = step.row_operator(rows, end)
            _, valid, gaps, _, friction, _ = step._block_arrays(rows, end)
            impulse = scale[:, None] * scaled_impulse
            momentum = (
                velocity
                - free_velocity
                - mass.inverse_mass_operator.mv(dual_transpose(operator).mv(impulse))
            )
            law = operator.mv(velocity)
            law = law.at[:, 0].set(jnp.where(valid, gaps / step_size, law[:, 0]))
            cone = contact_cone_fixed_branch_residual(
                fixed, scaled_impulse, scale[:, None] * law, friction
            )
            return jnp.concatenate((momentum, cone.reshape((-1,))))

        unknown = jnp.concatenate(
            (velocity_star, (impulse_star / scale[:, None]).reshape((-1,)))
        )
        linearization = la.prepare_linearization(residual, (unknown, parameters))
        zero_parameters = jax.tree.map(jnp.zeros_like, parameters)
        zero_unknown = jnp.zeros_like(unknown)
        space = la.ArraySpace(unknown.shape, dtype=dtype)

        def action(direction: Array) -> Array:
            return linearization.jvp((direction, zero_parameters))

        def transpose_action(direction: Array) -> Array:
            return linearization.vjp(direction)[0]

        jacobian = la.FunctionLinearOperator(
            action,
            source=space,
            target=space,
            transpose_action=transpose_action,
            operator_id=f"{self.program_id}:coupled-fixed-branch-jacobian",
        )
        certificate = la.certify_rectangular_rank(jacobian, regularity)
        regular = certificate.fixed_rank & (certificate.rank == unknown.size)
        return _CoupledFixedBranch(
            linearization,
            jacobian,
            certificate,
            regular,
            branch_admitted,
            scale,
            zero_parameters,
            zero_unknown,
        )

    def _gated_solve(
        self,
        operator: la.AbstractLinearOperator,
        right: Array,
        admitted: Array,
        /,
    ) -> tuple[Array, Array]:
        """Solve only when admission and the rank certificate already hold."""

        policy = _derivative_policy(self.policy)

        def run(_: None) -> tuple[Array, Array]:
            solution = la.solve(la.LinearSystem(operator), right, policy=policy)
            return solution.value, jnp.asarray(solution.successful)

        def skip(_: None) -> tuple[Array, Array]:
            return jnp.zeros_like(right), jnp.asarray(False)

        return jax.lax.cond(admitted, run, skip, None)

    def step_jvp(
        self,
        result: ArticulatedContactStepResult,
        actuation: ConvertibleToArray,
        gravity: ConvertibleToArray,
        tangents: tuple[
            PreparedArticulatedContactStep | None,
            ConvertibleToArray,
            ConvertibleToArray,
            ConvertibleToArray,
            ConvertibleToArray,
        ],
        /,
    ) -> ArticulatedContactStepJVP:
        """Fixed-branch tangent of a single-segment step, or an explicit refusal.

        ``tangents`` is ``(step_tangent, lifted_configuration, velocity,
        actuation, gravity)``; ``step_tangent`` is a tangent PyTree of this
        prepared step's inexact leaves (geometry, inertia, friction), or
        ``None`` for zero. The result must have been produced by this exact
        step with these actuation and gravity values. With contact or stop
        blocks the coupled ``(v1, p)`` fixed-branch Jacobian must be certified
        nonsingular by ``policy.derivative_regularity`` before the single
        linear solve; a supplied ``step_tangent`` additionally requires that
        no frictionless contact sits on its friction-parameter boundary.
        Subdivided results and feature, branch, activation and winding
        transitions are refused with zero tangents.
        """

        dtype = self.dtype
        command, gravity_, binding_current, start, dynamic, static = (
            self._derivative_inputs(result, actuation, gravity)
        )
        source = result.source
        step_size = result.requested_duration
        candidate = result.fixed_branch_candidate
        nv = self.articulation.nv
        (
            step_tangent,
            start_tangent,
            velocity_tangent,
            actuation_tangent,
            gravity_tangent,
        ) = tangents
        step_tangent_ = (
            jax.tree.map(jnp.zeros_like, dynamic)
            if step_tangent is None
            else step_tangent
        )
        parameters: _StepParameters = (dynamic, start, source.velocity, command, gravity_)
        parameter_tangent: _StepParameters = (
            step_tangent_,
            jnp.asarray(start_tangent, dtype=dtype),
            jnp.asarray(velocity_tangent, dtype=dtype),
            jnp.asarray(actuation_tangent, dtype=dtype).reshape(command.shape),
            jnp.asarray(gravity_tangent, dtype=dtype).reshape((3,)),
        )
        zero_impulse = jnp.zeros_like(result.last_segment.impulse)
        zero_velocity = jnp.zeros_like(result.candidate.velocity)

        if self.block_count == 0:
            # Without contact or stop blocks the step is the explicit native
            # Euler map: no implicit system, so no regularity proof is needed.
            predictor = self._fixed_branch_predictor(result)

            def free_map(arguments: _StepParameters) -> Array:
                dynamic_, start_, velocity0, command_, gravity__ = arguments
                step = eqx.combine(dynamic_, static)
                return predictor(step, start_, velocity0, command_, gravity__)[0]

            _, velocity_dot = jax.jvp(free_map, (parameters,), (parameter_tangent,))
            configuration_dot = parameter_tangent[1] + step_size * velocity_dot
            finite = jnp.all(jnp.isfinite(velocity_dot))
            successful = candidate & binding_current & finite
            true = jnp.asarray(True)
            return ArticulatedContactStepJVP(
                jnp.where(successful, configuration_dot, 0.0),
                jnp.where(successful, velocity_dot, 0.0),
                zero_impulse,
                candidate,
                binding_current,
                true,
                true,
                None,
                true,
                successful,
            )
        regularity = self.policy.derivative_regularity
        if regularity is None:
            refused = jnp.asarray(False)
            return ArticulatedContactStepJVP(
                zero_velocity,
                zero_velocity,
                zero_impulse,
                candidate,
                binding_current,
                refused,
                refused,
                None,
                refused,
                refused,
            )
        coupled = self._coupled_fixed_branch(
            result,
            parameters,
            static,
            regularity,
            friction_derivative=step_tangent is not None,
        )
        admitted = candidate & binding_current & coupled.branch_admitted & coupled.regular
        right = -coupled.linearization.jvp((coupled.zero_unknown, parameter_tangent))
        unknown_dot, linear_successful = self._gated_solve(
            coupled.jacobian, right, admitted
        )
        velocity_dot = unknown_dot[:nv]
        impulse_dot = coupled.scale[:, None] * unknown_dot[nv:].reshape(
            zero_impulse.shape
        )
        configuration_dot = parameter_tangent[1] + step_size * velocity_dot
        finite = jnp.all(jnp.isfinite(unknown_dot)) & jnp.all(
            jnp.isfinite(configuration_dot)
        )
        successful = admitted & linear_successful & finite
        return ArticulatedContactStepJVP(
            jnp.where(successful, configuration_dot, 0.0),
            jnp.where(successful, velocity_dot, 0.0),
            jnp.where(successful, impulse_dot, 0.0),
            candidate,
            binding_current,
            coupled.branch_admitted,
            coupled.regular,
            coupled.certificate,
            linear_successful,
            successful,
        )

    def step_vjp(
        self,
        result: ArticulatedContactStepResult,
        actuation: ConvertibleToArray,
        gravity: ConvertibleToArray,
        cotangents: tuple[ConvertibleToArray, ConvertibleToArray, ConvertibleToArray],
        /,
        *,
        friction_cotangent: bool = True,
    ) -> ArticulatedContactStepVJP:
        """Transpose of :meth:`step_jvp` on the same certified coupled operator.

        ``cotangents`` is ``(lifted_configuration_bar, velocity_bar,
        impulse_bar)`` on the accepted endpoint and block impulses. With
        ``friction_cotangent=True`` the friction-table cotangents are returned
        and frictionless contacts on their friction-parameter boundary refuse
        the VJP. With ``False`` no friction derivative is formed: the step
        cotangent's ``static_friction_table`` and ``dynamic_friction_table``
        are ``None`` rather than a fabricated zero.
        """

        if not isinstance(friction_cotangent, bool):
            raise TypeError("friction_cotangent must be a bool.")
        dtype = self.dtype
        command, gravity_, binding_current, start, dynamic, static = (
            self._derivative_inputs(result, actuation, gravity)
        )
        source = result.source
        step_size = result.requested_duration
        candidate = result.fixed_branch_candidate
        parameters: _StepParameters = (dynamic, start, source.velocity, command, gravity_)
        configuration_bar, velocity_bar, impulse_bar = (
            jnp.asarray(value, dtype=dtype) for value in cotangents
        )
        state_bar = velocity_bar + step_size * configuration_bar

        def masked_result(
            parameter_bar: _StepParameters,
            successful: Array,
            branch_admitted: Array,
            regular: Array,
            certificate: la.RectangularRankCertificate | None,
            linear_successful: Array,
        ) -> ArticulatedContactStepVJP:
            masked = jax.tree.map(
                lambda leaf: jnp.where(successful, leaf, 0.0), parameter_bar
            )
            step_cotangent = masked[0]
            if not friction_cotangent:
                step_cotangent = eqx.tree_at(
                    lambda step: (
                        step.static_friction_table,
                        step.dynamic_friction_table,
                    ),
                    step_cotangent,
                    (None, None),
                    is_leaf=lambda leaf: leaf is None,
                )
            return ArticulatedContactStepVJP(
                step_cotangent,
                masked[1] + jnp.where(successful, configuration_bar, 0.0),
                masked[2],
                masked[3],
                masked[4],
                candidate,
                binding_current,
                branch_admitted,
                regular,
                certificate,
                linear_successful,
                successful,
                friction_cotangent,
            )

        def all_finite(tree: _StepParameters) -> Array:
            return jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)])
            )

        if self.block_count == 0:
            # Explicit native Euler map: no implicit system to certify.
            predictor = self._fixed_branch_predictor(result)

            def free_map(arguments: _StepParameters) -> Array:
                dynamic_, start_, velocity0, command_, gravity__ = arguments
                step = eqx.combine(dynamic_, static)
                return predictor(step, start_, velocity0, command_, gravity__)[0]

            _, pullback = jax.vjp(free_map, parameters)
            (parameter_bar,) = pullback(state_bar)
            true = jnp.asarray(True)
            return masked_result(
                parameter_bar,
                candidate & binding_current & all_finite(parameter_bar),
                true,
                true,
                None,
                true,
            )
        regularity = self.policy.derivative_regularity
        if regularity is None:
            refused = jnp.asarray(False)
            zeros = jax.tree.map(jnp.zeros_like, parameters)
            return masked_result(zeros, refused, refused, refused, None, refused)
        coupled = self._coupled_fixed_branch(
            result,
            parameters,
            static,
            regularity,
            friction_derivative=friction_cotangent,
        )
        admitted = candidate & binding_current & coupled.branch_admitted & coupled.regular
        unknown_bar = jnp.concatenate(
            (state_bar, (coupled.scale[:, None] * impulse_bar).reshape((-1,)))
        )
        multiplier, linear_successful = self._gated_solve(
            la.transpose(coupled.jacobian), unknown_bar, admitted
        )
        _, residual_parameter_bar = coupled.linearization.vjp(multiplier)
        parameter_bar = jax.tree.map(lambda leaf: -leaf, residual_parameter_bar)
        successful = admitted & linear_successful & all_finite(parameter_bar)
        return masked_result(
            parameter_bar,
            successful,
            coupled.branch_admitted,
            coupled.regular,
            coupled.certificate,
            linear_successful,
        )


def _is_active(branch: Array, /) -> Array:
    flags = jnp.zeros(branch.shape, dtype=jnp.bool_)
    for value in _ACTIVE_BRANCHES:
        flags = flags | (branch == value)
    return flags


def _derivative_policy(policy: ArticulatedContactStepPolicy, /) -> la.LinearSolvePolicy:
    return la.LinearSolvePolicy(
        la.GMRES(),
        tolerance=la.TolerancePolicy(
            relative=policy.derivative_linear_tolerance,
            absolute=0.0,
            max_steps=policy.derivative_linear_iterations,
        ),
    )


def _rebound_cone_result(
    program: ContactConeProgram, impulse: Array, policy: ArticulatedContactStepPolicy, /
) -> ContactConeResult:
    """Cone result of the accepted impulse on its rebuilt response program.

    Called only by derivatives whose exact input binding is checked: identical
    step, source, actuation and gravity reproduce the primal response program,
    so this result's numeric revision is the original one. A stale binding
    makes the derivative refuse regardless of this reconstruction.
    """

    certificate = certify_contact_cone_impulse(program, impulse)
    return build_contact_cone_result(
        program,
        policy.cone_solver,
        impulse,
        converged=certificate.successful,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        fixed_point_residual=jnp.asarray(0.0, dtype=impulse.dtype),
        iteration_tolerance=jnp.asarray(0.0, dtype=impulse.dtype),
        step_size=jnp.asarray(1.0, dtype=impulse.dtype),
        step_reductions=jnp.asarray(0, dtype=jnp.int32),
        step_budget_exhausted=jnp.asarray(False),
    )


@eqx.filter_jit
def _compiled_contact_step(
    step: PreparedArticulatedContactStep,
    source: ArticulatedContactSource,
    duration: Array,
    actuation: Array,
    gravity: Array,
    geometry_revision: Array,
    material_revision: Array,
    /,
) -> ArticulatedContactStepResult:
    """Compiler boundary of :meth:`PreparedArticulatedContactStep.step`."""

    return step._execute(
        source, duration, actuation, gravity, geometry_revision, material_revision
    )


__all__ = [
    "CERTIFICATE_FAMILIES",
    "ArticulatedActuationMode",
    "ArticulatedContactAdmissibility",
    "ArticulatedContactCertificatePolicy",
    "ArticulatedContactEnergyLedger",
    "ArticulatedContactRootEvidence",
    "ArticulatedContactSearchBackend",
    "ArticulatedContactSearchPlan",
    "ArticulatedContactSegmentEvidence",
    "ArticulatedContactSource",
    "ArticulatedContactStepBinding",
    "ArticulatedContactStepJVP",
    "ArticulatedContactStepPolicy",
    "ArticulatedContactStepResult",
    "ArticulatedContactStepStatus",
    "ArticulatedContactStepVJP",
    "ArticulatedRouteGeometry",
    "PreparedArticulatedContactStep",
]
