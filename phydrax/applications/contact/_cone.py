#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Operator-valued fixed-route Signorini--Coulomb impulse programs.

The effective response ``W`` is a native linear operator from the contact
impulse dual to the contact velocity space.  Mixed programs combine Coulomb
contact blocks with normal-only unilateral blocks (for example joint stops) in
one fixed-width layout with explicit component masks.  Every block belongs to
one physical unit family whose reference velocity defines the congruent
scaling ``S W S`` used by the iteration and whose impulse, law-velocity and
slip-speed tolerances define the physical certificate.  Energy-like
certificates (complementarity work and tangential work) share one unit family
by construction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    DualSpace,
    FunctionLinearOperator,
    OperatorProperties,
)
from ...typing import checked, ConvertibleToArray
from ._materials import ContactMaterialPairTable


class ContactBlockLaw(IntEnum):
    """Unilateral law carried by one fixed-width impulse block."""

    COULOMB = 0
    NORMAL_ONLY = 1


class ContactBlockIdentity(IntEnum):
    """Typed owner of one block key; keys of different owners never compare."""

    CONTACT_ROUTE = 0
    JOINT_STOP = 1


class ContactConeBranch(IntEnum):
    """Fixed-branch classification of a certified block.

    ``NORMAL_ACTIVE`` is a normal-only block (for example a joint stop) in
    contact.  ``FRICTIONLESS_BOUNDARY`` is a Coulomb block with zero friction
    coefficients in contact: its tangential impulse is zero, but a friction
    coefficient increase changes it one-sidedly, so friction-parameter
    derivatives are refused there.
    """

    UNCLASSIFIED = 0
    SEPARATING = 1
    STICKING = 2
    SLIDING = 3
    NORMAL_ACTIVE = 4
    FRICTIONLESS_BOUNDARY = 5


def _concrete(value: ConvertibleToArray, /) -> np.ndarray | None:
    try:
        return np.asarray(value)
    except jax.errors.TracerArrayConversionError:
        return None


class ContactCertificatePolicy(StrictModule, NonTrainableState):
    """Dimensionally separate physical certificate of a cone program.

    Each unit family (for example translational contact in m/s and N s, hinge
    stops in rad/s and N m s) has a timestep-independent ``reference_velocity``
    defining the congruent scaling ``S W S``, an impulse tolerance in the
    family's impulse units, and law-velocity and slip-speed tolerances in its
    velocity units.  Complementarity and tangential work tolerances are
    energies shared by all families.  All tolerance values are dynamic leaves
    bound into every numeric revision; family names are static semantics.
    Iteration tolerances in solver plans remain convergence controls only.
    """

    reference_velocity: Array
    impulse_tolerance: Array
    law_velocity_tolerance: Array
    slip_speed_tolerance: Array
    complementarity_work_tolerance: Array
    friction_work_tolerance: Array
    family_names: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        family_names: Sequence[str],
        /,
        *,
        reference_velocity: ConvertibleToArray,
        impulse_tolerance: ConvertibleToArray,
        law_velocity_tolerance: ConvertibleToArray,
        slip_speed_tolerance: ConvertibleToArray,
        complementarity_work_tolerance: ConvertibleToArray,
        friction_work_tolerance: ConvertibleToArray,
    ) -> None:
        names = tuple(str(name) for name in family_names)
        if not names or any(not name for name in names):
            raise ValueError("Certificate unit families need nonempty names.")
        if len(set(names)) != len(names):
            raise ValueError("Certificate unit family names must be unique.")
        count = len(names)

        def family(value: ConvertibleToArray, name: str) -> Array:
            array = jnp.asarray(value)
            if not jnp.issubdtype(array.dtype, jnp.floating):
                array = array.astype(jnp.float64)
            if array.shape == ():
                array = jnp.full((count,), array)
            if array.shape != (count,):
                raise ValueError(f"{name} must be scalar or one value per family.")
            return array

        def energy(value: ConvertibleToArray, name: str) -> Array:
            array = jnp.asarray(value)
            if not jnp.issubdtype(array.dtype, jnp.floating):
                array = array.astype(jnp.float64)
            if array.shape != ():
                raise ValueError(f"{name} must be one scalar energy.")
            return array

        values = (
            family(reference_velocity, "reference_velocity"),
            family(impulse_tolerance, "impulse_tolerance"),
            family(law_velocity_tolerance, "law_velocity_tolerance"),
            family(slip_speed_tolerance, "slip_speed_tolerance"),
            energy(complementarity_work_tolerance, "complementarity_work_tolerance"),
            energy(friction_work_tolerance, "friction_work_tolerance"),
        )
        for value in values:
            concrete = _concrete(value)
            if concrete is not None and not np.all(
                np.isfinite(concrete) & (concrete > 0.0)
            ):
                raise ValueError(
                    "Certificate scales and tolerances must be finite and positive."
                )
        (
            self.reference_velocity,
            self.impulse_tolerance,
            self.law_velocity_tolerance,
            self.slip_speed_tolerance,
            self.complementarity_work_tolerance,
            self.friction_work_tolerance,
        ) = values
        self.family_names = names
        self.policy_id = canonical_fingerprint(
            {"kind": "contact-certificate-policy", "families": list(names)}
        )

    @property
    def family_count(self) -> int:
        return len(self.family_names)

    def family_values(self, dtype: DTypeLike, /) -> tuple[Array, Array, Array, Array]:
        """Return reference velocity and impulse/law/slip tolerances per family."""

        return (
            self.reference_velocity.astype(dtype),
            self.impulse_tolerance.astype(dtype),
            self.law_velocity_tolerance.astype(dtype),
            self.slip_speed_tolerance.astype(dtype),
        )


class ContactBlockLayout(StrictModule, NonTrainableState):
    """Fixed-width mixed block layout prepared on the host.

    Component zero is the normal component of every block.  Coulomb blocks
    activate all tangent components; normal-only blocks pad them with the
    explicit equation ``p_component = 0``.
    """

    law: Array
    identity: Array
    component_mask: Array
    unit_family: Array
    width: int = eqx.field(static=True)
    family_count: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        law: ConvertibleToArray,
        identity: ConvertibleToArray,
        component_mask: ConvertibleToArray,
        unit_family: ConvertibleToArray,
        /,
        *,
        family_count: int,
    ) -> None:
        law_ = np.asarray(law, dtype=np.int8)
        identity_ = np.asarray(identity, dtype=np.int8)
        mask = np.asarray(component_mask, dtype=np.bool_)
        family = np.asarray(unit_family, dtype=np.int32)
        families = int(family_count)
        if mask.ndim != 2 or mask.shape[1] < 1:
            raise ValueError("component_mask must have shape (blocks, width).")
        count, width = mask.shape
        if law_.shape != (count,) or identity_.shape != (count,):
            raise ValueError("Block law and identity must have block shape.")
        if family.shape != (count,):
            raise ValueError("unit_family must have block shape.")
        if families <= 0 or np.any(family < 0) or np.any(family >= families):
            raise ValueError("unit_family indices must address declared families.")
        known_laws = {int(value) for value in ContactBlockLaw}
        known_identities = {int(value) for value in ContactBlockIdentity}
        if not set(np.unique(law_).tolist()) <= known_laws:
            raise ValueError("Unknown contact block law.")
        if not set(np.unique(identity_).tolist()) <= known_identities:
            raise ValueError("Unknown contact block identity.")
        if count and not np.all(mask[:, 0]):
            raise ValueError("Every block must activate its normal component.")
        coulomb = law_ == int(ContactBlockLaw.COULOMB)
        if np.any(~mask[coulomb]):
            raise ValueError("Coulomb blocks must activate every tangent component.")
        if np.any(mask[~coulomb, 1:]):
            raise ValueError("Normal-only blocks must pad tangent components.")
        if np.any((identity_ == int(ContactBlockIdentity.JOINT_STOP)) & coulomb):
            raise ValueError("Joint-stop blocks are normal-only.")
        self.law = jnp.asarray(law_)
        self.identity = jnp.asarray(identity_)
        self.component_mask = jnp.asarray(mask)
        self.unit_family = jnp.asarray(family)
        self.width = int(width)
        self.family_count = families
        self.layout_id = canonical_fingerprint(
            {
                "kind": "contact-block-layout",
                "width": int(width),
                "family_count": families,
                "law": law_.tolist(),
                "identity": identity_.tolist(),
                "unit_family": family.tolist(),
            }
        )

    @classmethod
    def coulomb(
        cls,
        count: int,
        width: int,
        /,
        *,
        unit_family: int = 0,
        family_count: int = 1,
    ) -> ContactBlockLayout:
        """Homogeneous Coulomb contact layout of one unit family."""

        blocks = int(count)
        width_ = int(width)
        if blocks < 0 or width_ < 1:
            raise ValueError("Coulomb layout needs nonnegative count and width.")
        return cls(
            np.full((blocks,), int(ContactBlockLaw.COULOMB), dtype=np.int8),
            np.full((blocks,), int(ContactBlockIdentity.CONTACT_ROUTE), dtype=np.int8),
            np.ones((blocks, width_), dtype=np.bool_),
            np.full((blocks,), int(unit_family), dtype=np.int32),
            family_count=family_count,
        )

    @classmethod
    def mixed(
        cls,
        contact_count: int,
        stop_unit_families: Sequence[int],
        /,
        *,
        contact_unit_family: int = 0,
        family_count: int,
        width: int = 3,
    ) -> ContactBlockLayout:
        """Coulomb contact blocks followed by normal-only joint-stop blocks."""

        contacts = int(contact_count)
        stops = np.asarray(tuple(int(value) for value in stop_unit_families))
        width_ = int(width)
        if contacts < 0 or width_ < 1:
            raise ValueError("Mixed layout needs nonnegative contacts and width.")
        stop_count = int(stops.size)
        mask = np.zeros((contacts + stop_count, width_), dtype=np.bool_)
        mask[:contacts] = True
        mask[:, 0] = True
        return cls(
            np.concatenate(
                (
                    np.full((contacts,), int(ContactBlockLaw.COULOMB), dtype=np.int8),
                    np.full(
                        (stop_count,), int(ContactBlockLaw.NORMAL_ONLY), dtype=np.int8
                    ),
                )
            ),
            np.concatenate(
                (
                    np.full(
                        (contacts,),
                        int(ContactBlockIdentity.CONTACT_ROUTE),
                        dtype=np.int8,
                    ),
                    np.full(
                        (stop_count,),
                        int(ContactBlockIdentity.JOINT_STOP),
                        dtype=np.int8,
                    ),
                )
            ),
            mask,
            np.concatenate(
                (
                    np.full((contacts,), int(contact_unit_family), dtype=np.int32),
                    stops.astype(np.int32).reshape((stop_count,)),
                )
            ),
            family_count=family_count,
        )

    @property
    def block_count(self) -> int:
        return int(self.law.shape[0])


def contact_cone_space(count: int, width: int, dtype: DTypeLike, /) -> ArraySpace:
    """Contact velocity space of a fixed-width block program."""

    return ArraySpace((int(count), int(width)), dtype=dtype)


def dense_contact_response(
    matrix: ConvertibleToArray,
    space: ArraySpace,
    /,
    *,
    operator_id: str | None = None,
) -> AbstractLinearOperator:
    """Explicit dense response ``W`` over flattened block coordinates."""

    if not isinstance(space, ArraySpace) or len(space.shape) != 2:
        raise TypeError("space must be a two-dimensional contact ArraySpace.")
    return DenseLinearOperator(
        jnp.asarray(matrix, dtype=space.dtype),
        source=DualSpace(space),
        target=space,
        operator_id=operator_id,
    )


def local_block_contact_response(
    blocks: ConvertibleToArray,
    /,
    *,
    operator_id: str | None = None,
) -> AbstractLinearOperator:
    """Block-diagonal response with one ``(width, width)`` block per contact."""

    values = jnp.asarray(blocks)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("blocks must have shape (count, width, width).")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(jnp.float64)
    space = contact_cone_space(values.shape[0], values.shape[1], values.dtype)

    def action(impulse: Array) -> Array:
        return (values @ impulse[..., None])[..., 0]

    def transpose_action(velocity: Array) -> Array:
        return (jnp.swapaxes(values, -1, -2) @ velocity[..., None])[..., 0]

    return FunctionLinearOperator(
        action,
        source=DualSpace(space),
        target=space,
        transpose_action=transpose_action,
        properties=OperatorProperties(
            block_diagonal=True,
            evidence={"block_diagonal": "construction"},
        ),
        operator_id=operator_id
        if operator_id is not None
        else canonical_fingerprint(
            {
                "kind": "local-block-contact-response",
                "count": int(values.shape[0]),
                "width": int(values.shape[1]),
                "dtype": str(values.dtype),
            }
        ),
    )


def _per_block(
    value: ConvertibleToArray, count: int, dtype: DTypeLike, name: str, /
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if array.shape == ():
        array = jnp.full((count,), array)
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or have block shape.")
    return array


def _per_component(
    value: ConvertibleToArray, count: int, width: int, dtype: DTypeLike, name: str, /
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if array.shape == ():
        array = jnp.full((count, width), array)
    elif array.shape == (count,):
        array = jnp.broadcast_to(array[:, None], (count, width))
    elif array.shape == (count * width,):
        array = array.reshape((count, width))
    if array.shape != (count, width):
        raise ValueError(f"{name} must be scalar, per block, or per component.")
    return array


class ContactConeProgram(StrictModule, NonTrainableState):
    """Fixed-route Signorini--Coulomb program over a native response operator.

    ``free_velocity`` is the physical contact velocity ``G v_free``.  The law
    velocity is ``W p + C p + free_velocity + normal_bias e_n``; the physical
    post-impulse contact velocity is ``W p + free_velocity``.  ``normal_bias``
    carries restitution targets or a position-level gap law ``g/h``.
    ``response_binding`` holds the response owner's dynamic numeric binding
    (for example a prepared articulated-mass binding); it is compared exactly
    together with every visible array leaf of ``response``.
    """

    response: AbstractLinearOperator
    free_velocity: Array
    normal_bias: Array
    compliance: Array
    dynamic_friction: Array
    static_friction: Array
    restitution: Array
    route_keys: Array
    valid: Array
    mechanical_available: Array
    response_binding: PyTree[Array]
    layout: ContactBlockLayout
    certificate: ContactCertificatePolicy
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        response: AbstractLinearOperator,
        free_velocity: ConvertibleToArray,
        compliance: ConvertibleToArray,
        dynamic_friction: ConvertibleToArray,
        route_keys: ConvertibleToArray,
        valid: ConvertibleToArray,
        layout: ContactBlockLayout,
        certificate: ContactCertificatePolicy,
        program_id: str,
        /,
        *,
        static_friction: ConvertibleToArray | None = None,
        normal_bias: ConvertibleToArray = 0.0,
        restitution: ConvertibleToArray = 0.0,
        mechanical_available: ConvertibleToArray = True,
        response_binding: PyTree[Array] = (),
    ) -> None:
        if not isinstance(response, AbstractLinearOperator):
            raise TypeError("response must be AbstractLinearOperator.")
        if not isinstance(layout, ContactBlockLayout):
            raise TypeError("layout must be ContactBlockLayout.")
        if not isinstance(certificate, ContactCertificatePolicy):
            raise TypeError("certificate must be ContactCertificatePolicy.")
        if layout.family_count != certificate.family_count:
            raise ValueError("Layout and certificate unit families disagree.")
        free = jnp.asarray(free_velocity)
        if not jnp.issubdtype(free.dtype, jnp.floating):
            free = free.astype(jnp.float64)
        count = layout.block_count
        width = layout.width
        if free.shape != (count, width):
            raise ValueError("free_velocity must have shape (blocks, width).")
        space = contact_cone_space(count, width, free.dtype)
        if response.batch_shape:
            raise ValueError("Contact response cannot be operator-batched.")
        if not response.target.compatible(space) or not response.source.compatible(
            DualSpace(space)
        ):
            raise ValueError(
                "response must map DualSpace(contact space) to the contact space."
            )
        dynamic = _per_block(dynamic_friction, count, free.dtype, "dynamic_friction")
        static = (
            dynamic
            if static_friction is None
            else _per_block(static_friction, count, free.dtype, "static_friction")
        )
        keys = jnp.asarray(route_keys)
        if keys.shape != (count,) or not jnp.issubdtype(keys.dtype, jnp.integer):
            raise ValueError("route_keys must be one integer key per block.")
        route_mask = jnp.asarray(valid, dtype=jnp.bool_)
        if route_mask.shape != (count,):
            raise ValueError("valid must have block shape.")
        binding_leaves = jax.tree.leaves(response_binding)
        if not all(isinstance(leaf, jax.Array) for leaf in binding_leaves):
            raise TypeError("response_binding leaves must be arrays.")
        identifier = str(program_id)
        if not identifier:
            raise ValueError("program_id must be non-empty.")
        self.response = response
        self.free_velocity = free
        self.normal_bias = _per_block(normal_bias, count, free.dtype, "normal_bias")
        self.compliance = _per_component(
            compliance, count, width, free.dtype, "compliance"
        )
        self.dynamic_friction = dynamic
        self.static_friction = static
        self.restitution = _per_block(restitution, count, free.dtype, "restitution")
        self.route_keys = keys
        self.valid = route_mask
        self.mechanical_available = jnp.broadcast_to(
            jnp.asarray(mechanical_available, dtype=jnp.bool_), (count,)
        )
        self.response_binding = response_binding
        self.layout = layout
        self.certificate = certificate
        self.program_id = identifier

    @property
    def contact_count(self) -> int:
        return self.layout.block_count

    @property
    def local_dimension(self) -> int:
        return self.layout.width

    @property
    def tangent_dimension(self) -> int:
        return self.layout.width - 1

    @property
    def contact_space(self) -> ArraySpace:
        return contact_cone_space(
            self.contact_count, self.local_dimension, self.free_velocity.dtype
        )

    @property
    def law_free_velocity(self) -> Array:
        """Law velocity at zero impulse: ``free_velocity + normal_bias e_n``."""

        return self.free_velocity.at[:, 0].add(self.normal_bias)

    def response_action(self, impulse: ConvertibleToArray, /) -> Array:
        """Apply the native response ``W`` to a contact impulse."""

        return self.response.mv(jnp.asarray(impulse, dtype=self.free_velocity.dtype))


class _ContactConeSolverControls(Protocol):
    @property
    def maximum_iterations(self) -> int: ...

    @property
    def absolute_tolerance(self) -> float: ...

    @property
    def relative_tolerance(self) -> float: ...

    @property
    def relaxation(self) -> float: ...

    @property
    def plan_id(self) -> str: ...


class ContactConeSolverPlan(StrictModule, NonTrainableState):
    """Projected-gradient controls with descent-lemma step backtracking.

    No spectral bound is assumed.  Every accepted step satisfies the quadratic
    descent inequality ``d·(S W S + S C S) d <= (1 + sqrt(eps)) |d|^2 / t`` on the
    actual iterate, which is the qualified convergence condition for the
    projected iteration on the fixed-friction cone program.  The total number
    of step halvings is bounded by ``maximum_step_reductions``; exhausting it is
    a failure, not convergence.
    """

    maximum_iterations: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    maximum_step_reductions: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 200,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        relaxation: float = 1.0,
        maximum_step_reductions: int = 64,
    ) -> None:
        iterations = int(maximum_iterations)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        relaxation_ = float(relaxation)
        reductions = int(maximum_step_reductions)
        if iterations <= 0 or reductions < 0:
            raise ValueError("Cone iteration budgets are invalid.")
        if absolute <= 0.0 or relative < 0.0:
            raise ValueError("Cone solver tolerances are invalid.")
        if not 0.0 < relaxation_ <= 1.0:
            raise ValueError("relaxation must lie in (0, 1].")
        self.maximum_iterations = iterations
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.relaxation = relaxation_
        self.maximum_step_reductions = reductions
        self.plan_id = canonical_fingerprint(
            {
                "kind": "contact-cone-solver-plan",
                "maximum_iterations": iterations,
                "absolute_tolerance": absolute.hex(),
                "relative_tolerance": relative.hex(),
                "relaxation": relaxation_.hex(),
                "maximum_step_reductions": reductions,
                "step": "descent-lemma-backtracking",
            }
        )


class ContactConeNumericRevision(StrictModule):
    """Exact operator, binding, law, certificate and solver numerics of a result.

    ``response`` keeps every visible array leaf of the response operator
    (closure-converted actions expose their captured arrays as leaves).
    """

    response: AbstractLinearOperator
    response_binding: PyTree[Array]
    certificate: ContactCertificatePolicy
    free_velocity: Array
    normal_bias: Array
    compliance: Array
    static_friction: Array
    dynamic_friction: Array
    restitution: Array
    route_keys: Array
    route_mask: Array
    mechanical_available: Array
    block_law: Array
    block_identity: Array
    component_mask: Array
    unit_family: Array
    solver_parameters: Array
    program_id: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)


class ContactConeCertificate(StrictModule):
    """Per-block physical residuals of one impulse against its program law.

    Impulse residuals carry each block's family impulse units, velocity
    residuals its family velocity units, and work residuals energy.
    """

    post_velocity: Array
    law_velocity: Array
    normal_impulse_defect: Array
    law_normal_defect: Array
    complementarity_work: Array
    cone_defect: Array
    law_slip_speed: Array
    sliding_impulse_defect: Array
    padding_defect: Array
    friction_law_satisfied: Array
    physical_tangential_work: Array
    impulse_tolerance: Array
    law_velocity_tolerance: Array
    slip_speed_tolerance: Array
    normal_impulse_valid: Array
    law_normal_valid: Array
    complementarity_valid: Array
    cone_valid: Array
    friction_law_valid: Array
    padding_valid: Array
    friction_work_valid: Array
    numeric_inputs_valid: Array
    material_law_complete: Array
    finite: Array
    successful: Array


class ContactConeEvidence(StrictModule):
    """Iteration controls, physical certificate and numeric binding of a solve.

    ``converged``, ``fixed_point_residual`` and ``iteration_tolerance`` describe
    the scaled iteration; the selected-friction stage of
    :func:`solve_contact_cone` stops only once its residual is within tolerance
    and the physical certificate holds, so non-convergence within the budget is
    a bounded refusal.  ``step_size``, ``step_reductions`` and
    ``step_budget_exhausted`` report the descent-lemma step search.
    ``certificate`` holds the dimensional physical certificates; ``successful``
    requires convergence, an unexhausted step budget, finiteness and every
    certificate.  ``numeric_revision`` binds the exact inputs.
    """

    converged: Array
    iterations: Array
    fixed_point_residual: Array
    iteration_tolerance: Array
    step_size: Array
    step_reductions: Array
    step_budget_exhausted: Array
    certificate: ContactConeCertificate
    finite: Array
    successful: Array
    numeric_revision: ContactConeNumericRevision
    program_id: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)


class ContactConeResult(StrictModule):
    """Accepted impulse plus the candidate retained for failure diagnosis.

    ``post_relative_velocity`` is the physical contact velocity ``W p + free``;
    ``contact_law_velocity`` additionally includes compliance and the law bias.
    """

    impulse: Array
    post_relative_velocity: Array
    evidence: ContactConeEvidence
    candidate_impulse: Array
    candidate_post_relative_velocity: Array
    contact_law_velocity: Array
    candidate_contact_law_velocity: Array
    branch: Array

    @checked
    def __init__(
        self,
        impulse: ConvertibleToArray,
        post_relative_velocity: ConvertibleToArray,
        evidence: ContactConeEvidence,
        /,
        *,
        candidate_impulse: ConvertibleToArray,
        candidate_post_relative_velocity: ConvertibleToArray,
        contact_law_velocity: ConvertibleToArray,
        candidate_contact_law_velocity: ConvertibleToArray,
        branch: ConvertibleToArray,
    ) -> None:
        accepted = jnp.asarray(impulse)
        values = tuple(
            jnp.asarray(value, dtype=accepted.dtype)
            for value in (
                post_relative_velocity,
                candidate_impulse,
                candidate_post_relative_velocity,
                contact_law_velocity,
                candidate_contact_law_velocity,
            )
        )
        if any(value.shape != accepted.shape for value in values):
            raise ValueError("Cone result arrays must have identical shapes.")
        branch_ = jnp.asarray(branch, dtype=jnp.int8)
        if branch_.shape != accepted.shape[:1]:
            raise ValueError("Cone branch labels must have block shape.")
        self.impulse = accepted
        (
            self.post_relative_velocity,
            self.candidate_impulse,
            self.candidate_post_relative_velocity,
            self.contact_law_velocity,
            self.candidate_contact_law_velocity,
        ) = values
        self.evidence = evidence
        self.branch = branch_

    @property
    def successful(self) -> Array:
        return self.evidence.successful


def project_signorini_coulomb_product(
    impulse: ConvertibleToArray,
    friction: ConvertibleToArray,
    /,
) -> Array:
    """Project the Signorini normal and Coulomb friction ball product.

    Branches are selected explicitly so that derivatives are finite elements of
    the generalized Jacobian: the identity strictly inside a positive-radius
    friction ball, the radial boundary map outside it (where the tangent norm
    is positive), and the constant zero tangent for a zero radius at a zero
    tangent.
    """

    value = jnp.asarray(impulse)
    coefficient = jnp.asarray(friction, dtype=value.dtype)
    normal = jnp.maximum(value[..., 0], 0.0)
    tangent = value[..., 1:]
    squared = jnp.sum(tangent * tangent, axis=-1)
    positive = squared > 0.0
    tangent_norm = jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)
    limit = coefficient * normal
    interior = tangent_norm < limit
    radial = (~interior) & positive
    scale = jnp.where(
        interior,
        1.0,
        jnp.where(radial, limit / jnp.where(radial, tangent_norm, 1.0), 0.0),
    )
    projected_tangent = scale[..., None] * tangent
    return jnp.concatenate((normal[..., None], projected_tangent), axis=-1)


def project_contact_blocks(
    impulse: ConvertibleToArray,
    friction: ConvertibleToArray,
    layout: ContactBlockLayout,
    valid: ConvertibleToArray,
    /,
) -> Array:
    """Project every block on its declared law; padded/invalid components vanish."""

    value = jnp.asarray(impulse)
    coulomb = layout.law == int(ContactBlockLaw.COULOMB)
    coefficient = jnp.where(coulomb, jnp.asarray(friction, dtype=value.dtype), 0.0)
    mask = layout.component_mask & jnp.asarray(valid, dtype=jnp.bool_)[:, None]
    masked = jnp.where(mask, value, 0.0)
    return jnp.where(mask, project_signorini_coulomb_product(masked, coefficient), 0.0)


def _family_values(program: ContactConeProgram, /) -> tuple[Array, Array, Array, Array]:
    reference, impulse, law, slip = program.certificate.family_values(
        program.free_velocity.dtype
    )
    family = program.layout.unit_family
    return reference[family], impulse[family], law[family], slip[family]


def block_scale(program: ContactConeProgram, /) -> Array:
    """Per-block congruent scale ``s = 1 / reference_velocity``."""

    return 1.0 / _family_values(program)[0]


def _numeric_inputs_valid(program: ContactConeProgram, /) -> Array:
    return (
        jnp.all(program.compliance >= 0.0)
        & jnp.all(program.static_friction >= 0.0)
        & jnp.all(program.dynamic_friction >= 0.0)
        & jnp.all(program.dynamic_friction <= program.static_friction)
        & jnp.all((program.restitution >= 0.0) & (program.restitution <= 1.0))
        & jnp.all(jnp.isfinite(program.free_velocity))
        & jnp.all(jnp.isfinite(program.normal_bias))
        & jnp.all(jnp.isfinite(program.compliance))
        & jnp.all(jnp.isfinite(program.static_friction))
        & jnp.all(jnp.isfinite(program.dynamic_friction))
        & jnp.all(jnp.isfinite(program.restitution))
    )


def _certify(
    program: ContactConeProgram, impulse: Array, response_impulse: Array, /
) -> ContactConeCertificate:
    dtype = program.free_velocity.dtype
    valid = program.valid
    mask = program.layout.component_mask & valid[:, None]
    coulomb = (program.layout.law == int(ContactBlockLaw.COULOMB)) & valid
    _, impulse_tolerance, law_tolerance, slip_tolerance = _family_values(program)
    post = response_impulse + program.free_velocity
    law = post + program.compliance * impulse
    law = law.at[:, 0].add(program.normal_bias)
    normal_impulse = jnp.where(valid, impulse[:, 0], 0.0)
    normal_law = jnp.where(valid, law[:, 0], 0.0)
    tangent_mask = mask[:, 1:]
    tangent_impulse = jnp.where(tangent_mask, impulse[:, 1:], 0.0)
    tangent_law = jnp.where(tangent_mask, law[:, 1:], 0.0)
    tangent_post = jnp.where(tangent_mask, post[:, 1:], 0.0)
    tangent_impulse_norm = jnp.sqrt(jnp.sum(tangent_impulse**2, axis=-1))
    slip = jnp.sqrt(jnp.sum(tangent_law**2, axis=-1))
    positive_normal = jnp.maximum(normal_impulse, 0.0)
    normal_impulse_defect = jnp.maximum(-normal_impulse, 0.0)
    law_normal_defect = jnp.maximum(-normal_law, 0.0)
    complementarity_work = jnp.abs(normal_impulse * normal_law)
    cone_defect = jnp.where(
        coulomb,
        jnp.maximum(
            tangent_impulse_norm - program.static_friction * positive_normal, 0.0
        ),
        0.0,
    )
    safe_slip = jnp.maximum(slip, jnp.finfo(dtype).tiny)
    sliding_impulse = (-program.dynamic_friction * positive_normal / safe_slip)[
        :, None
    ] * tangent_law
    sliding_impulse_defect = jnp.where(
        coulomb,
        jnp.sqrt(jnp.sum((tangent_impulse - sliding_impulse) ** 2, axis=-1)),
        0.0,
    )
    sticking = (slip <= slip_tolerance) & (cone_defect <= impulse_tolerance)
    sliding = (slip > slip_tolerance) & (sliding_impulse_defect <= impulse_tolerance)
    friction_law_satisfied = (~coulomb) | sticking | sliding
    padding_defect = jnp.max(jnp.where(mask, 0.0, jnp.abs(impulse)), axis=-1, initial=0.0)
    physical_tangential_work = jnp.sum(
        jnp.where(coulomb[:, None], tangent_impulse * tangent_post, 0.0)
    )
    finite = (
        jnp.all(jnp.isfinite(impulse))
        & jnp.all(jnp.isfinite(post))
        & jnp.all(jnp.isfinite(law))
    )
    normal_impulse_valid = jnp.all(normal_impulse_defect <= impulse_tolerance)
    law_normal_valid = jnp.all(law_normal_defect <= law_tolerance)
    complementarity_valid = jnp.all(
        complementarity_work <= program.certificate.complementarity_work_tolerance
    )
    cone_valid = jnp.all(cone_defect <= impulse_tolerance)
    friction_law_valid = jnp.all(friction_law_satisfied)
    padding_valid = jnp.all(padding_defect <= impulse_tolerance)
    friction_work_valid = (
        physical_tangential_work <= program.certificate.friction_work_tolerance
    )
    numeric_inputs_valid = _numeric_inputs_valid(program)
    material_law_complete = jnp.all((~valid) | program.mechanical_available)
    successful = (
        finite
        & normal_impulse_valid
        & law_normal_valid
        & complementarity_valid
        & cone_valid
        & friction_law_valid
        & padding_valid
        & friction_work_valid
        & numeric_inputs_valid
        & material_law_complete
    )
    return ContactConeCertificate(
        post_velocity=post,
        law_velocity=law,
        normal_impulse_defect=normal_impulse_defect,
        law_normal_defect=law_normal_defect,
        complementarity_work=complementarity_work,
        cone_defect=cone_defect,
        law_slip_speed=slip,
        sliding_impulse_defect=sliding_impulse_defect,
        padding_defect=padding_defect,
        friction_law_satisfied=friction_law_satisfied,
        physical_tangential_work=physical_tangential_work,
        impulse_tolerance=impulse_tolerance,
        law_velocity_tolerance=law_tolerance,
        slip_speed_tolerance=slip_tolerance,
        normal_impulse_valid=normal_impulse_valid,
        law_normal_valid=law_normal_valid,
        complementarity_valid=complementarity_valid,
        cone_valid=cone_valid,
        friction_law_valid=friction_law_valid,
        padding_valid=padding_valid,
        friction_work_valid=friction_work_valid,
        numeric_inputs_valid=numeric_inputs_valid,
        material_law_complete=material_law_complete,
        finite=finite,
        successful=successful,
    )


def certify_contact_cone_impulse(
    program: ContactConeProgram, impulse: ConvertibleToArray, /
) -> ContactConeCertificate:
    """Evaluate every physical certificate of one impulse with current actions."""

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    value = jnp.asarray(impulse, dtype=program.free_velocity.dtype)
    if value.shape != program.free_velocity.shape:
        raise ValueError("impulse must have the program block shape.")
    return _certify(program, value, program.response_action(value))


def classify_contact_cone_branches(
    program: ContactConeProgram,
    impulse: ConvertibleToArray,
    certificate: ContactConeCertificate,
    /,
) -> tuple[Array, Array]:
    """Return branch labels and dimensionless margins in tolerance units.

    A margin is the distance of the deciding quantity beyond its tolerance
    band, divided by the corresponding family tolerance.  Normal-only blocks
    in contact are ``NORMAL_ACTIVE``.  Coulomb blocks with zero static and
    dynamic coefficients in contact are ``FRICTIONLESS_BOUNDARY``: their
    primal is the normal-only law, but they lie on the friction-parameter
    boundary.  Apex, stick/slip-boundary and invalid cases are
    ``UNCLASSIFIED`` with margin zero.
    """

    value = jnp.asarray(impulse, dtype=program.free_velocity.dtype)
    valid = program.valid
    coulomb = program.layout.law == int(ContactBlockLaw.COULOMB)
    impulse_tolerance = certificate.impulse_tolerance
    law_tolerance = certificate.law_velocity_tolerance
    slip_tolerance = certificate.slip_speed_tolerance
    normal = value[:, 0]
    law_normal = certificate.law_velocity[:, 0]
    tangent = jnp.where(program.layout.component_mask[:, 1:], value[:, 1:], 0.0)
    tangent_norm = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
    slip = certificate.law_slip_speed
    contacting = normal > impulse_tolerance
    separating = (~contacting) & (law_normal > law_tolerance)
    static_slack = program.static_friction * normal - tangent_norm
    frictional = coulomb & (
        (program.static_friction > 0.0) | (program.dynamic_friction > 0.0)
    )
    sticking = (
        contacting
        & frictional
        & (slip <= slip_tolerance)
        & (static_slack > impulse_tolerance)
    )
    sliding = contacting & frictional & (slip > slip_tolerance)
    normal_active = contacting & ~coulomb
    frictionless = contacting & coulomb & ~frictional
    branch = jnp.where(
        separating,
        int(ContactConeBranch.SEPARATING),
        jnp.where(
            sticking,
            int(ContactConeBranch.STICKING),
            jnp.where(
                sliding,
                int(ContactConeBranch.SLIDING),
                jnp.where(
                    normal_active,
                    int(ContactConeBranch.NORMAL_ACTIVE),
                    jnp.where(
                        frictionless,
                        int(ContactConeBranch.FRICTIONLESS_BOUNDARY),
                        int(ContactConeBranch.UNCLASSIFIED),
                    ),
                ),
            ),
        ),
    )
    branch = jnp.where(valid, branch, int(ContactConeBranch.UNCLASSIFIED)).astype(
        jnp.int8
    )
    contact_margin = normal / impulse_tolerance - 1.0
    margin = jnp.where(
        separating,
        law_normal / law_tolerance - 1.0,
        jnp.where(
            sticking,
            jnp.minimum(contact_margin, static_slack / impulse_tolerance - 1.0),
            jnp.where(
                sliding,
                jnp.minimum(contact_margin, slip / slip_tolerance - 1.0),
                jnp.where(normal_active | frictionless, contact_margin, 0.0),
            ),
        ),
    )
    margin = jnp.where(valid, margin, jnp.inf)
    return branch, margin


def _scaled_action(
    program: ContactConeProgram, scale: Array, /
) -> Callable[[Array], Array]:
    def action(value: Array) -> Array:
        impulse = scale[:, None] * value
        return scale[:, None] * (
            program.response.mv(impulse) + program.compliance * impulse
        )

    return action


def _projected_gradient(
    action: Callable[[Array], Array],
    bias: Array,
    initial: Array,
    project: Callable[[Array], Array],
    maximum_iterations: int,
    relaxation: float,
    maximum_reductions: int | Array,
    tolerance: Array,
    accept: Callable[[Array], Array] | None = None,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Monotone projected gradient on ``0.5 x·A x + b·x`` over the block cone.

    Iteration stops when the natural residual is within ``tolerance`` and the
    optional ``accept(value)`` predicate, evaluated only on such iterates,
    holds.  Returns iterate, convergence, iterations, natural residual, final
    step, total step reductions and budget exhaustion.
    """

    dtype = bias.dtype
    slack = jnp.sqrt(jnp.finfo(dtype).eps)
    tiny = jnp.finfo(dtype).tiny
    start = project(initial)
    start_action = action(start)
    gradient = start_action + bias
    descent = start - project(start - gradient)
    descent_action = action(descent)
    curvature = jnp.sum(descent * descent_action)
    squared = jnp.sum(descent * descent)
    initial_step = jnp.where(
        (squared > 0.0) & (curvature > tiny),
        squared / jnp.maximum(curvature, tiny),
        jnp.asarray(1.0, dtype=dtype),
    )
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    false = jnp.asarray(False)

    _State = tuple[Array, Array, Array, Array, Array, Array, Array, Array]

    def condition(state: _State) -> Array:
        index, _, _, _, converged, _, _, exhausted = state
        return (index < maximum_iterations) & ~converged & ~exhausted

    def body(state: _State) -> _State:
        index, value, value_action, step, _, _, reductions, _ = state
        gradient_ = value_action + bias
        natural = value - project(value - gradient_)
        norm = jnp.sqrt(jnp.sum(natural * natural))
        within = norm <= tolerance
        if accept is None:
            now = within
        else:

            def unchecked(_: Array) -> Array:
                return false

            now = jax.lax.cond(within, accept, unchecked, value)

        def advance(
            _: Array,
        ) -> tuple[Array, Array, Array, Array, Array]:
            def trial(step_: Array) -> tuple[Array, Array, Array]:
                difference = project(value - step_ * gradient_) - value
                difference_action = action(difference)
                accepted = jnp.sum(difference * difference_action) <= (
                    (1.0 + slack) * jnp.sum(difference * difference) / step_
                )
                return difference, difference_action, accepted

            first = trial(step)

            def search_condition(
                search: tuple[Array, Array, Array, Array, Array],
            ) -> Array:
                _, _, _, accepted, count = search
                return (~accepted) & (count < maximum_reductions)

            def search_body(
                search: tuple[Array, Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array, Array]:
                step_, _, _, _, count = search
                halved = 0.5 * step_
                difference, difference_action, accepted = trial(halved)
                return halved, difference, difference_action, accepted, count + 1

            final_step, difference, difference_action, accepted, count = (
                jax.lax.while_loop(
                    search_condition,
                    search_body,
                    (step, first[0], first[1], first[2], reductions),
                )
            )
            next_value = jnp.where(accepted, value + relaxation * difference, value)
            next_action = jnp.where(
                accepted,
                value_action + relaxation * difference_action,
                value_action,
            )
            return next_value, next_action, final_step, count, ~accepted

        def hold(_: Array) -> tuple[Array, Array, Array, Array, Array]:
            return value, value_action, step, reductions, false

        next_value, next_action, next_step, count, exhausted = jax.lax.cond(
            now, hold, advance, now
        )
        return (
            index + 1,
            next_value,
            next_action,
            next_step,
            now,
            norm,
            count,
            exhausted,
        )

    (
        iterations,
        value,
        _,
        step,
        converged,
        residual,
        reductions,
        exhausted,
    ) = jax.lax.while_loop(
        condition,
        body,
        (
            zero_int,
            start,
            start_action,
            initial_step,
            false,
            jnp.asarray(jnp.inf, dtype=dtype),
            zero_int,
            false,
        ),
    )
    return value, converged, iterations, residual, step, reductions, exhausted


def _solver_parameters(
    program: ContactConeProgram, solver: _ContactConeSolverControls, /
) -> Array:
    return jnp.asarray(
        (
            solver.maximum_iterations,
            solver.absolute_tolerance,
            solver.relative_tolerance,
            solver.relaxation,
        ),
        dtype=program.free_velocity.dtype,
    )


def _bound_numerics(program: ContactConeProgram, /) -> tuple[PyTree[Array], ...]:
    return (
        program.response,
        program.response_binding,
        program.certificate,
        program.free_velocity,
        program.normal_bias,
        program.compliance,
        program.static_friction,
        program.dynamic_friction,
        program.restitution,
        program.route_keys,
        program.valid,
        program.mechanical_available,
        program.layout.law,
        program.layout.identity,
        program.layout.component_mask,
        program.layout.unit_family,
    )


def contact_cone_numeric_revision(
    program: ContactConeProgram, solver: _ContactConeSolverControls, /
) -> ContactConeNumericRevision:
    """Snapshot every numerical input of a solve, including operator leaves."""

    return ContactConeNumericRevision(
        response=program.response,
        response_binding=program.response_binding,
        certificate=program.certificate,
        free_velocity=program.free_velocity,
        normal_bias=program.normal_bias,
        compliance=program.compliance,
        static_friction=program.static_friction,
        dynamic_friction=program.dynamic_friction,
        restitution=program.restitution,
        route_keys=program.route_keys,
        route_mask=program.valid,
        mechanical_available=program.mechanical_available,
        block_law=program.layout.law,
        block_identity=program.layout.identity,
        component_mask=program.layout.component_mask,
        unit_family=program.layout.unit_family,
        solver_parameters=_solver_parameters(program, solver),
        program_id=program.program_id,
        solver_id=solver.plan_id,
        layout_id=program.layout.layout_id,
    )


def contact_cone_numeric_revision_matches(
    program: ContactConeProgram, result: ContactConeResult, /
) -> Array:
    """Exact, JAX-safe equality of every bound numerical input of a result."""

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    if not isinstance(result, ContactConeResult):
        raise TypeError("result must be ContactConeResult.")
    revision = result.evidence.numeric_revision
    if (
        revision.program_id != program.program_id
        or result.evidence.program_id != program.program_id
        or revision.solver_id != result.evidence.solver_id
        or revision.layout_id != program.layout.layout_id
    ):
        return jnp.asarray(False)
    recorded = (
        revision.response,
        revision.response_binding,
        revision.certificate,
        revision.free_velocity,
        revision.normal_bias,
        revision.compliance,
        revision.static_friction,
        revision.dynamic_friction,
        revision.restitution,
        revision.route_keys,
        revision.route_mask,
        revision.mechanical_available,
        revision.block_law,
        revision.block_identity,
        revision.component_mask,
        revision.unit_family,
    )
    equal = eqx.tree_equal(_bound_numerics(program), recorded)
    if equal is False:
        return jnp.asarray(False)
    return (
        jnp.asarray(equal)
        & (revision.solver_parameters.shape == (4,))
        & jnp.all(jnp.isfinite(revision.solver_parameters))
    )


def contact_cone_result_is_certified(
    program: ContactConeProgram, result: ContactConeResult, /
) -> Array:
    """Re-certify a result with current actions rather than trusting evidence."""

    current = contact_cone_numeric_revision_matches(program, result)
    certificate = certify_contact_cone_impulse(program, result.impulse)
    velocity_tolerance = jnp.concatenate(
        (
            certificate.law_velocity_tolerance[:, None],
            jnp.broadcast_to(
                certificate.slip_speed_tolerance[:, None],
                (program.contact_count, program.tangent_dimension),
            ),
        ),
        axis=-1,
    )
    post_consistent = jnp.all(
        jnp.abs(certificate.post_velocity - result.post_relative_velocity)
        <= velocity_tolerance
    )
    law_consistent = jnp.all(
        jnp.abs(certificate.law_velocity - result.contact_law_velocity)
        <= velocity_tolerance
    )
    return (
        current
        & result.evidence.successful
        & certificate.successful
        & post_consistent
        & law_consistent
    )


def build_contact_cone_result(
    program: ContactConeProgram,
    solver: _ContactConeSolverControls,
    candidate: Array,
    /,
    *,
    converged: Array,
    iterations: Array,
    fixed_point_residual: Array,
    iteration_tolerance: Array,
    step_size: Array,
    step_reductions: Array,
    step_budget_exhausted: Array,
) -> ContactConeResult:
    """Certify a candidate and apply it only when every certificate holds."""

    certificate = certify_contact_cone_impulse(program, candidate)
    branch, _ = classify_contact_cone_branches(program, candidate, certificate)
    finite = certificate.finite & jnp.isfinite(fixed_point_residual)
    successful = (
        jnp.asarray(converged)
        & ~jnp.asarray(step_budget_exhausted)
        & finite
        & certificate.successful
    )
    accepted = jnp.where(successful, candidate, 0.0)
    accepted_post = jnp.where(
        successful, certificate.post_velocity, program.free_velocity
    )
    accepted_law = jnp.where(
        successful, certificate.law_velocity, program.law_free_velocity
    )
    evidence = ContactConeEvidence(
        converged=jnp.asarray(converged, dtype=jnp.bool_),
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        fixed_point_residual=jnp.asarray(fixed_point_residual),
        iteration_tolerance=jnp.asarray(iteration_tolerance),
        step_size=jnp.asarray(step_size),
        step_reductions=jnp.asarray(step_reductions, dtype=jnp.int32),
        step_budget_exhausted=jnp.asarray(step_budget_exhausted, dtype=jnp.bool_),
        certificate=certificate,
        finite=finite,
        successful=successful,
        numeric_revision=contact_cone_numeric_revision(program, solver),
        program_id=program.program_id,
        solver_id=solver.plan_id,
    )
    return ContactConeResult(
        accepted,
        accepted_post,
        evidence,
        candidate_impulse=candidate,
        candidate_post_relative_velocity=certificate.post_velocity,
        contact_law_velocity=accepted_law,
        candidate_contact_law_velocity=certificate.law_velocity,
        branch=jnp.where(successful, branch, int(ContactConeBranch.UNCLASSIFIED)),
    )


def contact_iteration_tolerance(
    program: ContactConeProgram,
    solver: _ContactConeSolverControls,
    scale: Array,
    /,
) -> Array:
    """Scaled fixed-point tolerance; an iteration control, not a certificate.

    The requested tolerance is floored at the attainable floating-point
    resolution of the scaled natural residual (sixteen units in the last place
    of its magnitude), so single-precision programs can converge without
    stopping well above the precision the physical certificate needs; the
    physical certificate is unaffected by this floor.
    """

    active = jnp.where(
        program.valid[:, None], scale[:, None] * program.law_free_velocity, 0.0
    )
    magnitude = jnp.maximum(1.0, jnp.sqrt(jnp.sum(active * active)))
    resolution = 16.0 * jnp.finfo(program.free_velocity.dtype).eps * magnitude
    return jnp.maximum(
        solver.absolute_tolerance + solver.relative_tolerance * magnitude,
        resolution,
    )


def solve_contact_cone(
    program: ContactConeProgram,
    /,
    *,
    solver: ContactConeSolverPlan | None = None,
    initial_impulse: ConvertibleToArray | None = None,
) -> ContactConeResult:
    """Solve the declared compliant Signorini--Coulomb law with operator actions.

    A static-friction stage determines sticking blocks; the selected-friction
    stage solves the declared static/dynamic law.  Iteration is performed in
    the congruently scaled variables ``S W S``; every certificate is physical.
    """

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    solver_ = ContactConeSolverPlan() if solver is None else solver
    if not isinstance(solver_, ContactConeSolverPlan):
        raise TypeError("solver must be ContactConeSolverPlan or None.")
    shape = program.free_velocity.shape
    dtype = program.free_velocity.dtype
    initial = (
        jnp.zeros(shape, dtype=dtype)
        if initial_impulse is None
        else jnp.asarray(initial_impulse, dtype=dtype)
    )
    if initial.shape != shape:
        raise ValueError("initial_impulse has invalid shape.")
    scale = block_scale(program)
    action = _scaled_action(program, scale)
    bias = scale[:, None] * program.law_free_velocity
    tolerance = contact_iteration_tolerance(program, solver_, scale)
    scaled_initial = initial / scale[:, None]

    def projector(friction: Array) -> Callable[[Array], Array]:
        def project(value: Array) -> Array:
            return project_contact_blocks(value, friction, program.layout, program.valid)

        return project

    static_scaled, _, static_iterations, _, _, static_reductions, _ = _projected_gradient(
        action,
        bias,
        scaled_initial,
        projector(program.static_friction),
        solver_.maximum_iterations,
        solver_.relaxation,
        solver_.maximum_step_reductions,
        tolerance,
    )
    static_candidate = scale[:, None] * static_scaled
    static_certificate = certify_contact_cone_impulse(program, static_candidate)
    sticking = (
        static_certificate.law_slip_speed <= static_certificate.slip_speed_tolerance
    ) & (static_certificate.cone_defect <= static_certificate.impulse_tolerance)
    selected_friction = jnp.where(
        sticking, program.static_friction, program.dynamic_friction
    )

    def certified(value: Array) -> Array:
        # Evaluated only once the residual is within tolerance, with the same
        # fresh response action the returned certificate uses.  Iteration
        # continues within its budget until the declared physical certificate
        # holds; certificates that further iterations cannot change (inputs,
        # materials, finiteness) end the iteration and are reported as refusals.
        impulse = scale[:, None] * value
        certificate = _certify(program, impulse, program.response_action(impulse))
        unrecoverable = ~(
            certificate.numeric_inputs_valid
            & certificate.material_law_complete
            & certificate.finite
        )
        return certificate.successful | unrecoverable

    (
        candidate_scaled,
        converged,
        iterations,
        residual,
        step,
        reductions,
        exhausted,
    ) = _projected_gradient(
        action,
        bias,
        static_scaled,
        projector(selected_friction),
        solver_.maximum_iterations,
        solver_.relaxation,
        jnp.maximum(solver_.maximum_step_reductions - static_reductions, 0),
        tolerance,
        certified,
    )
    return build_contact_cone_result(
        program,
        solver_,
        scale[:, None] * candidate_scaled,
        converged=converged,
        iterations=static_iterations + iterations,
        fixed_point_residual=residual,
        iteration_tolerance=tolerance,
        step_size=step,
        step_reductions=static_reductions + reductions,
        step_budget_exhausted=exhausted,
    )


def build_contact_cone_program(
    kinematics: ContactKinematicsEpoch,
    materials: ContactMaterialPairTable,
    response: AbstractLinearOperator,
    /,
    *,
    certificate: ContactCertificatePolicy,
    compliance: ConvertibleToArray = 0.0,
    layout: ContactBlockLayout | None = None,
    response_binding: PyTree[Array] = (),
) -> ContactConeProgram:
    """Lower fixed-route kinematics and materials to the operator cone program.

    Restitution enters only the normal law bias; the free velocity remains the
    physical recorded contact velocity.
    """

    if not isinstance(kinematics, ContactKinematicsEpoch):
        raise TypeError("kinematics must be ContactKinematicsEpoch.")
    if not isinstance(materials, ContactMaterialPairTable):
        raise TypeError("materials must be ContactMaterialPairTable.")
    if not kinematics.batches:
        raise ValueError("Contact cone program requires positive route capacity.")
    tangent_dimension = kinematics.batches[0].tangential_velocity.shape[-1]
    if any(
        batch.tangential_velocity.shape[-1] != tangent_dimension
        for batch in kinematics.batches
    ):
        raise ValueError("Contact cone tangent dimensions disagree.")
    normal_velocity = jnp.concatenate(
        tuple(batch.normal_velocity for batch in kinematics.batches)
    )
    tangential_velocity = jnp.concatenate(
        tuple(batch.tangential_velocity for batch in kinematics.batches)
    )
    left_material = jnp.concatenate(
        tuple(batch.left_material_ids for batch in kinematics.batches)
    )
    right_material = jnp.concatenate(
        tuple(batch.right_material_ids for batch in kinematics.batches)
    )
    route_keys = jnp.concatenate(tuple(batch.route_keys for batch in kinematics.batches))
    valid = jnp.concatenate(tuple(batch.valid for batch in kinematics.batches))
    parameters = materials.lookup(left_material, right_material)
    free = jnp.concatenate((normal_velocity[:, None], tangential_velocity), axis=-1)
    count = valid.size
    width = 1 + tangent_dimension
    layout_ = (
        ContactBlockLayout.coulomb(count, width, family_count=certificate.family_count)
        if layout is None
        else layout
    )
    if layout_.block_count != count or layout_.width != width:
        raise ValueError("Block layout does not match the kinematic routes.")
    normal_bias = parameters.restitution * jnp.minimum(normal_velocity, 0.0)
    program_id = canonical_fingerprint(
        {
            "kind": "contact-cone-program",
            "kinematics": kinematics.epoch_id,
            "materials": materials.table_id,
            "response": response.operator_id,
            "layout": layout_.layout_id,
            "certificate": certificate.policy_id,
        }
    )
    return ContactConeProgram(
        response,
        free,
        compliance,
        parameters.dynamic_friction,
        route_keys,
        valid,
        layout_,
        certificate,
        program_id,
        static_friction=parameters.static_friction,
        normal_bias=normal_bias,
        restitution=parameters.restitution,
        mechanical_available=parameters.mechanical_available,
        response_binding=response_binding,
    )


__all__ = [
    "ContactBlockIdentity",
    "ContactBlockLaw",
    "ContactBlockLayout",
    "ContactCertificatePolicy",
    "ContactConeBranch",
    "ContactConeCertificate",
    "ContactConeEvidence",
    "ContactConeNumericRevision",
    "ContactConeProgram",
    "ContactConeResult",
    "ContactConeSolverPlan",
    "block_scale",
    "build_contact_cone_program",
    "build_contact_cone_result",
    "certify_contact_cone_impulse",
    "classify_contact_cone_branches",
    "contact_cone_numeric_revision",
    "contact_cone_numeric_revision_matches",
    "contact_cone_result_is_certified",
    "contact_cone_space",
    "contact_iteration_tolerance",
    "dense_contact_response",
    "local_block_contact_response",
    "project_contact_blocks",
    "project_signorini_coulomb_product",
    "solve_contact_cone",
]
