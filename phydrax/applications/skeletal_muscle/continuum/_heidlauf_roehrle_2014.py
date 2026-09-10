#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Heidlauf–Röhrle (2014), Eqs. (1)–(2): prescribed active-stress continuum.

Independently derived from DOI 10.3389/fphys.2014.00498 (CC BY 4.0).
This is not GASAM, an implemented cellular coupling, or a physiological oracle.
The source pressure is compression-positive. No isochoric projection, passive
compression cutoff, or active force-length/velocity multiplier is introduced.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ...._fingerprint import canonical_fingerprint
from ...._identity import (
    callable_payload,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
)
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import (
    MixedFiniteElementConstraintPlan,
    PreparedMixedFiniteElementConstraint,
)
from ....ein import contract
from ....equations import CellResidualAction, FiniteElementForm
from ....units import convert_value, KILOPASCAL, ONE, PASCAL
from ._fiber import PreparedUniformFiberArchitecture


_SOURCE_ID = "doi:10.3389/fphys.2014.00498"
_SOURCE_SHA256 = "545ef19e9ee0298c667849461483a45f209ec5c019db127a1d2810138217db41"
_FORCE_OWNER = "heidlauf-roehrle-2014-continuum"


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty identity.")
    return value.strip()


def _real(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return result


def _scalar(value: ArrayLike, name: str, /) -> Array:
    result = _real(value, name)
    if result.shape != ():
        raise ValueError(f"{name} must be one scalar.")
    return result


def _deformation(value: ArrayLike, /) -> Array:
    result = _real(value, "deformation_gradient")
    if result.shape != (3, 3):
        raise ValueError("deformation_gradient must have shape (3, 3).")
    return result


class HeidlaufRoehrle2014Parameters(StrictModule):
    """Trainable Table 2 material parameters in Pa, except dimensionless d1."""

    c10_pa: Array
    c01_pa: Array
    b1_pa: Array
    d1: Array
    maximum_active_nominal_stress_pa: Array

    def __init__(self, c10_pa, c01_pa, b1_pa, d1, maximum_active_nominal_stress_pa, /):
        values = tuple(
            _scalar(value, name)
            for name, value in (
                ("c10_pa", c10_pa),
                ("c01_pa", c01_pa),
                ("b1_pa", b1_pa),
                ("d1", d1),
                ("maximum_active_nominal_stress_pa", maximum_active_nominal_stress_pa),
            )
        )
        if any(not bool(jnp.isfinite(value) & (value > 0.0)) for value in values):
            raise ValueError("2014 material parameters must be finite and positive.")
        (
            self.c10_pa,
            self.c01_pa,
            self.b1_pa,
            self.d1,
            self.maximum_active_nominal_stress_pa,
        ) = values

    @classmethod
    def published_table_2(cls) -> HeidlaufRoehrle2014Parameters:
        """Convert the four source stress values from kPa to Pa exactly once."""
        stresses = convert_value(
            jnp.asarray((6.352e-10, 3.627, 2.756e-5, 73.0)),
            source=KILOPASCAL,
            target=PASCAL,
        )
        return cls(stresses[0], stresses[1], stresses[2], 43.373, stresses[3])

    def values(self, /) -> Array:
        return jnp.stack(
            (
                self.c10_pa,
                self.c01_pa,
                self.b1_pa,
                self.d1,
                self.maximum_active_nominal_stress_pa,
            )
        )


class HeidlaufRoehrle2014StressInput(StrictModule, NonTrainableState):
    """A prescribed, already homogenized gamma-bar and its source-state receipt.

    ``source_state_token`` is the eight uint32 words of a caller's content digest
    of the supplying state/protocol sample, not a force or an activation counter.
    Gamma is finite and signed, with no [0, 1] clipping: Eq. (6) supplies neither
    such a bound nor a tensile-only correction. A source may fail independently.
    """

    normalized_active_stress: Array
    source_state_token: Array
    source_successful: Array
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        normalized_active_stress,
        source_state_token,
        source_id,
        /,
        *,
        source_successful=True,
    ):
        gamma = _scalar(normalized_active_stress, "normalized_active_stress")
        token = jnp.asarray(source_state_token)
        if token.shape != (8,) or token.dtype != jnp.uint32:
            raise ValueError("source_state_token must contain eight uint32 digest words.")
        successful = jnp.asarray(source_successful, dtype=jnp.bool_)
        if successful.shape != ():
            raise ValueError("source_successful must be scalar.")
        self.normalized_active_stress = gamma
        self.source_state_token = token
        self.source_successful = successful
        self.source_id = _identifier(source_id, "source_id")


class HeidlaufRoehrle2014InputEvidence(StrictModule, NonTrainableState):
    finite: Array
    source_successful: Array
    source_matches: Array
    valid: Array
    source_state_token: Array
    force_owner: str = eqx.field(static=True, default=_FORCE_OWNER)


class HeidlaufRoehrle2014MaterialState(StrictModule, NonTrainableState):
    normalized_active_stress: Array
    source_state_token: Array
    accepted_updates: Array
    evidence: HeidlaufRoehrle2014InputEvidence


class HeidlaufRoehrle2014MaterialCommit(StrictModule, NonTrainableState):
    state: HeidlaufRoehrle2014MaterialState
    previous: HeidlaufRoehrle2014MaterialState
    parameter_values: Array
    committed: Array
    rollback_applied: Array
    prepared_id: str = eqx.field(static=True)


class HeidlaufRoehrle2014MaterialCandidate(StrictModule, NonTrainableState):
    previous: HeidlaufRoehrle2014MaterialState
    proposed: HeidlaufRoehrle2014MaterialState
    parameter_values: Array
    evidence: HeidlaufRoehrle2014InputEvidence
    prepared_id: str = eqx.field(static=True)

    def commit(
        self, /, *, successful: ArrayLike = True
    ) -> HeidlaufRoehrle2014MaterialCommit:
        """Select every input/evidence/counter leaf atomically under the outer gate."""
        gate = jnp.asarray(successful, dtype=jnp.bool_)
        if gate.shape != ():
            raise ValueError("successful must be scalar.")
        accepted = self.evidence.valid & gate
        state = jax.tree_util.tree_map(
            lambda proposed, previous: jnp.where(accepted, proposed, previous),
            self.proposed,
            self.previous,
        )
        return HeidlaufRoehrle2014MaterialCommit(
            state,
            self.previous,
            self.parameter_values,
            accepted,
            ~accepted,
            self.prepared_id,
        )


class HeidlaufRoehrle2014PointEvidence(StrictModule, NonTrainableState):
    jacobian: Array
    fiber_stretch: Array
    finite: Array
    orientation_preserved: Array
    parameters_valid: Array
    input_valid: Array
    valid: Array
    force_owner: str = eqx.field(static=True, default=_FORCE_OWNER)


class HeidlaufRoehrle2014PointResponse(StrictModule):
    passive_energy_density_j_per_m3: Array
    passive_first_piola: Array
    active_first_piola: Array
    pressure_first_piola: Array
    first_piola: Array
    second_piola: Array
    cauchy_stress: Array
    constraint_residual: Array
    active_nominal_stress_pa: Array
    evidence: HeidlaufRoehrle2014PointEvidence


class HeidlaufRoehrle2014BlockTangent(StrictModule):
    deformation_deformation: Array
    deformation_pressure: Array
    deformation_active_stress: Array
    constraint_deformation: Array


class HeidlaufRoehrle2014ActiveStressField(StrictModule, NonTrainableState):
    """Identified quadrature transfer callback; never a second mechanical owner.

    ``evaluate(points, context)`` returns gamma-bar with shape ``points.shape[:-1]``.
    Dynamic input is ``context.user_args``; the callback must preserve its source
    success/state and coverage checks (invalid input returns NaN or raises).
    Provenance binds the complete fixed mapping, support, frames, and algorithm.
    No clipping or second force-length/velocity factor is applied by the owner.
    """

    evaluate: Callable
    provenance: SemanticProvenance
    revision: NumericRevision
    field_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluate: Callable,
        provenance: SemanticProvenance,
        revision: NumericRevision,
        /,
        *,
        source_id: str,
    ):
        if not callable(evaluate) or not isinstance(provenance, SemanticProvenance):
            raise TypeError(
                "An active stress field requires a callable and semantic provenance."
            )
        if (
            not isinstance(revision, NumericRevision)
            or revision.semantic_id != provenance.semantic_id
        ):
            raise ValueError(
                "The field numeric revision must belong to its semantic provenance."
            )
        identity = (
            callable_payload(evaluate)
            if isinstance(evaluate, StrictModule)
            else callable_payload(
                evaluate,
                semantic_id=provenance.semantic_id,
                numeric_id=revision.revision_id,
            )
        )
        self.evaluate = evaluate
        self.provenance = provenance
        self.revision = revision
        self.source_id = _identifier(source_id, "source_id")
        self.field_id = canonical_fingerprint(
            {
                "callable": identity,
                "source": self.source_id,
                "provenance": provenance.semantic_id,
                "revision": revision.revision_id,
            }
        )


class HeidlaufRoehrle2014Plan(StrictModule, NonTrainableState):
    """Fixed source, prescribed-input identity, reference fiber family and units."""

    material_id: str = eqx.field(static=True)
    active_stress_source_id: str = eqx.field(static=True)
    minimum_jacobian: float = eqx.field(static=True)
    provenance: SemanticProvenance
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_id: str,
        active_stress_source_id: str,
        /,
        *,
        minimum_jacobian: float = 1.0e-8,
    ):
        self.material_id = _identifier(material_id, "material_id")
        self.active_stress_source_id = _identifier(
            active_stress_source_id, "active_stress_source_id"
        )
        minimum = float(minimum_jacobian)
        if not isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_jacobian must be finite and positive.")
        self.minimum_jacobian = minimum
        self.provenance = SemanticProvenance(
            {
                "kind": _FORCE_OWNER,
                "material": self.material_id,
                "active_stress_source": self.active_stress_source_id,
                "equations": "2014:Eqs1-2;Table2;section2.2",
                "license": "CC-BY-4.0",
                "stress_unit": PASCAL,
                "normalized_stress_unit": ONE,
                "deformation_axes": ("spatial_i", "reference_J"),
                "pressure": "compression-positive;S_pressure=-p*C^-1",
                "constraint": "-log(det(F))=0;no-bulk-penalty",
                "passive": "unprojected-Mooney-Rivlin;no-compression-cutoff",
                "active": "Pmax*gamma_bar;nonconservative-source-input",
                "minimum_jacobian": minimum,
            },
            resource_ids={"paper": _SOURCE_ID, "article_sha256": _SOURCE_SHA256},
        )
        self.plan_id = self.provenance.semantic_id

    def prepare(
        self,
        parameters: HeidlaufRoehrle2014Parameters,
        architecture: PreparedUniformFiberArchitecture,
        active_stress: HeidlaufRoehrle2014StressInput,
        /,
    ) -> PreparedHeidlaufRoehrle2014Material:
        if not isinstance(parameters, HeidlaufRoehrle2014Parameters):
            raise TypeError("parameters must be HeidlaufRoehrle2014Parameters.")
        if not isinstance(architecture, PreparedUniformFiberArchitecture):
            raise TypeError("architecture must be PreparedUniformFiberArchitecture.")
        if not bool(architecture.evidence.valid):
            raise ValueError(
                "A valid supported reference fiber architecture is required."
            )
        evidence = _input_evidence(active_stress, self.active_stress_source_id)
        if not bool(evidence.valid):
            raise ValueError(
                "Initial normalized stress must be finite with matching successful source evidence."
            )
        signature = ExecutableSignature(
            shapes={
                "deformation": (3, 3),
                "pressure": (),
                "gamma_bar": (),
                "source_token": (8,),
            },
            dtypes={
                "material": parameters.c10_pa.dtype,
                "gamma_bar": active_stress.normalized_active_stress.dtype,
            },
            topology_ids={"architecture": architecture.prepared_id},
            algorithm_facts={
                "owner": _FORCE_OWNER,
                "residual": "exact-u-p-active-stress",
            },
        )
        prepared_id = canonical_fingerprint(
            {"plan": self.plan_id, "signature": signature.signature_id}
        )
        state = HeidlaufRoehrle2014MaterialState(
            active_stress.normalized_active_stress,
            active_stress.source_state_token,
            jnp.asarray(0, dtype=jnp.uint32),
            evidence,
        )
        return PreparedHeidlaufRoehrle2014Material(
            self, parameters, architecture, state, signature, prepared_id
        )


def _input_evidence(value, source_id, /):
    if not isinstance(value, HeidlaufRoehrle2014StressInput):
        raise TypeError("active_stress must be HeidlaufRoehrle2014StressInput.")
    finite = jnp.isfinite(value.normalized_active_stress)
    matches = jnp.asarray(value.source_id == source_id)
    return HeidlaufRoehrle2014InputEvidence(
        finite,
        value.source_successful,
        matches,
        finite & value.source_successful & matches,
        value.source_state_token,
    )


class PreparedHeidlaufRoehrle2014Material(StrictModule):
    """Sole passive/active/pressure stress owner with the native mixed FE residual.

    Only the passive material has stored energy. Although prescribed constant
    gamma admits a frozen-input primitive, it is deliberately not used as a
    coupled active energy: the cellular source may depend on history/velocity.
    """

    plan: HeidlaufRoehrle2014Plan
    parameters: HeidlaufRoehrle2014Parameters
    architecture: PreparedUniformFiberArchitecture
    state: HeidlaufRoehrle2014MaterialState
    signature: ExecutableSignature
    prepared_id: str = eqx.field(static=True)

    def numeric_revision(self, /) -> NumericRevision:
        """Host-side numeric content identity, separate from compilation identity."""
        return NumericRevision(
            self.plan.provenance, {"parameters": self.parameters, "state": self.state}
        )

    def propose_active_stress(
        self, active_stress: HeidlaufRoehrle2014StressInput, /
    ) -> HeidlaufRoehrle2014MaterialCandidate:
        evidence = _input_evidence(active_stress, self.plan.active_stress_source_id)
        if (
            active_stress.normalized_active_stress.dtype
            != self.state.normalized_active_stress.dtype
        ):
            raise ValueError(
                "A constitutive transaction cannot change the prepared input dtype."
            )
        proposed = HeidlaufRoehrle2014MaterialState(
            active_stress.normalized_active_stress,
            active_stress.source_state_token,
            self.state.accepted_updates + jnp.asarray(1, dtype=jnp.uint32),
            evidence,
        )
        return HeidlaufRoehrle2014MaterialCandidate(
            self.state,
            proposed,
            self.parameters.values(),
            evidence,
            self.prepared_id,
        )

    def with_commit(
        self, commit: HeidlaufRoehrle2014MaterialCommit, /
    ) -> PreparedHeidlaufRoehrle2014Material:
        if not isinstance(commit, HeidlaufRoehrle2014MaterialCommit):
            raise TypeError("commit must be HeidlaufRoehrle2014MaterialCommit.")
        if commit.prepared_id != self.prepared_id:
            raise ValueError("Commit belongs to a foreign prepared 2014 material.")
        equal_leaves = jax.tree_util.tree_map(
            lambda left, right: jnp.all(left == right),
            self.state,
            commit.previous,
        )
        matches = jnp.all(jnp.stack(jax.tree_util.tree_leaves(equal_leaves)))
        matches = matches & jnp.all(commit.parameter_values == self.parameters.values())
        checked = eqx.error_if(
            commit.state.normalized_active_stress,
            ~matches,
            "Commit belongs to a stale source state or changed parameter revision.",
        )
        state = eqx.tree_at(
            lambda value: value.normalized_active_stress, commit.state, checked
        )
        return eqx.tree_at(lambda value: value.state, self, state)

    def passive_energy_density(self, deformation_gradient: ArrayLike, /) -> Array:
        """Reference energy W=c10(I1-3)+c01(I2-3)+b1[(lambda^d1-1)/d1-log(lambda)]."""
        deformation = _deformation(deformation_gradient)
        c = deformation.T @ deformation
        first = jnp.trace(c)
        second = 0.5 * (first * first - jnp.sum(c * c.T))
        stretch = jnp.linalg.norm(deformation @ self.architecture.reference_direction)
        log_stretch = jnp.log(stretch)
        parameters = self.parameters
        anisotropic = parameters.b1_pa * (
            jnp.expm1(parameters.d1 * log_stretch) / parameters.d1 - log_stretch
        )
        return (
            parameters.c10_pa * (first - 3.0)
            + parameters.c01_pa * (second - 3.0)
            + anisotropic
        )

    def _stress_parts(self, deformation, pressure, gamma, /):
        c = deformation.T @ deformation
        direction = self.architecture.reference_direction
        current_fiber = deformation @ direction
        stretch = jnp.linalg.norm(current_fiber)
        parameters = self.parameters
        anisotropic = (
            parameters.b1_pa
            * jnp.expm1(parameters.d1 * jnp.log(stretch))
            / (stretch * stretch)
        )
        passive_second = (
            2.0 * parameters.c10_pa * jnp.eye(3, dtype=deformation.dtype)
            + 2.0
            * parameters.c01_pa
            * (jnp.trace(c) * jnp.eye(3, dtype=deformation.dtype) - c)
            + anisotropic * self.architecture.structural_tensor
        )
        nominal = parameters.maximum_active_nominal_stress_pa * gamma
        active = nominal * jnp.outer(current_fiber / stretch, direction)
        inverse_result = la.inverse_small_linear(
            la.SmallLinearSolvePlan(3),
            deformation,
        )
        inverse = eqx.error_if(
            inverse_result.value,
            ~inverse_result.successful,
            "Muscle deformation gradient must remain nonsingular.",
        )
        return (
            deformation @ passive_second,
            active,
            -pressure * inverse.T,
            stretch,
            nominal,
            inverse,
        )

    def first_piola(
        self, deformation_gradient, pressure_pa, normalized_active_stress=None, /
    ) -> Array:
        """Return P(F,p;gamma), with gamma held independent in mechanical tangents."""
        deformation = _deformation(deformation_gradient)
        pressure = _scalar(pressure_pa, "pressure_pa")
        gamma = (
            self.state.normalized_active_stress
            if normalized_active_stress is None
            else _scalar(normalized_active_stress, "normalized_active_stress")
        )
        passive, active, pressure_stress, _, _, _ = self._stress_parts(
            deformation, pressure, gamma
        )
        stress = passive + active + pressure_stress
        valid = (
            jnp.linalg.det(deformation) > self.plan.minimum_jacobian
        ) & self.state.evidence.valid
        valid = valid & jnp.all(
            jnp.isfinite(self.parameters.values()) & (self.parameters.values() > 0.0)
        )
        valid = valid & jnp.all(jnp.isfinite(stress))
        # Multiplicative NaN propagation keeps direct JVP/VJP paths invalid;
        # selecting a constant NaN result instead can have a zero derivative.
        return stress * jnp.where(valid, 1.0, jnp.nan)

    def constraint(self, deformation_gradient: ArrayLike, /) -> Array:
        """Exact incompressibility residual, conjugate to source pressure -p log J."""
        deformation = _deformation(deformation_gradient)
        jacobian = jnp.linalg.det(deformation)
        valid = jnp.isfinite(jacobian) & (jacobian > self.plan.minimum_jacobian)
        return -jnp.log(jacobian) * jnp.where(valid, 1.0, jnp.nan)

    def evaluate(
        self, deformation_gradient, pressure_pa, normalized_active_stress=None, /
    ) -> HeidlaufRoehrle2014PointResponse:
        deformation = _deformation(deformation_gradient)
        pressure = _scalar(pressure_pa, "pressure_pa")
        gamma = (
            self.state.normalized_active_stress
            if normalized_active_stress is None
            else _scalar(normalized_active_stress, "normalized_active_stress")
        )
        passive, active, pressure_stress, stretch, nominal, inverse = self._stress_parts(
            deformation, pressure, gamma
        )
        total = passive + active + pressure_stress
        energy = self.passive_energy_density(deformation)
        jacobian = jnp.linalg.det(deformation)
        finite = jnp.all(jnp.isfinite(total)) & jnp.isfinite(energy) & jnp.isfinite(gamma)
        oriented = jacobian > self.plan.minimum_jacobian
        parameters_valid = jnp.all(
            jnp.isfinite(self.parameters.values()) & (self.parameters.values() > 0.0)
        )
        valid = finite & oriented & parameters_valid & self.state.evidence.valid
        evidence = HeidlaufRoehrle2014PointEvidence(
            jacobian,
            stretch,
            finite,
            oriented,
            parameters_valid,
            self.state.evidence.valid,
            valid,
        )
        total = total * jnp.where(valid, 1.0, jnp.nan)
        return HeidlaufRoehrle2014PointResponse(
            energy,
            passive,
            active,
            pressure_stress,
            total,
            inverse @ total,
            total @ deformation.T / jacobian,
            self.constraint(deformation),
            nominal,
            evidence,
        )

    def block_tangent(
        self, deformation_gradient, pressure_pa, normalized_active_stress=None, /
    ) -> HeidlaufRoehrle2014BlockTangent:
        deformation = _deformation(deformation_gradient)
        pressure = _scalar(pressure_pa, "pressure_pa")
        gamma = (
            self.state.normalized_active_stress
            if normalized_active_stress is None
            else _scalar(normalized_active_stress, "normalized_active_stress")
        )
        ff, fp, fg = jax.jacfwd(self.first_piola, argnums=(0, 1, 2))(
            deformation, pressure, gamma
        )
        valid = self.evaluate(deformation, pressure, gamma).evidence.valid
        blocks = (ff, fp, fg, jax.grad(self.constraint)(deformation))
        # AD of a selected constant NaN can be zero: mask the derivative values
        # themselves so an invalid primal is never an admissible tangent.
        return HeidlaufRoehrle2014BlockTangent(
            *(jnp.where(valid, block, jnp.nan) for block in blocks)
        )

    def first_piola_points(
        self, deformation, pressure, normalized_active_stress, /
    ) -> Array:
        """Batched FE/coupling input axes (...,3,3), (...), (...), respectively."""
        deformation = _real(deformation, "deformation")
        pressure = _real(pressure, "pressure")
        gamma = _real(normalized_active_stress, "normalized_active_stress")
        if (
            deformation.shape[-2:] != (3, 3)
            or pressure.shape != deformation.shape[:-2]
            or gamma.shape != pressure.shape
        ):
            raise ValueError(
                "Pressure and gamma must exactly cover the deformation leading axes."
            )
        stress = jax.vmap(self.first_piola)(
            deformation.reshape((-1, 3, 3)), pressure.reshape(-1), gamma.reshape(-1)
        )
        return stress.reshape(deformation.shape)

    def form(
        self,
        displacement_field="u",
        pressure_field="p",
        /,
        *,
        active_stress_field: HeidlaufRoehrle2014ActiveStressField | None = None,
        pressure_origin_pa: ArrayLike = 0.0,
    ) -> FiniteElementForm:
        """Native total-Lagrangian residual: int P:grad(v), int -log(J)q.

        FE pressure is a compression-positive increment: source p = p_FE +
        pressure_origin_pa. This explicit offset permits core zero/pinned gauges
        about a physically chosen preload; at passive rest choose 2*c10+4*c01.
        A field callback consumes dynamic ``context.user_args`` and must return
        one gamma per quadrature point, without broadcasting partial coverage.
        """
        displacement_name = _identifier(displacement_field, "displacement_field")
        pressure_name = _identifier(pressure_field, "pressure_field")
        if displacement_name == pressure_name:
            raise ValueError("Displacement and pressure names must be distinct.")
        origin = _scalar(pressure_origin_pa, "pressure_origin_pa")
        if not bool(jnp.isfinite(origin)):
            raise ValueError("pressure_origin_pa must be finite.")
        if active_stress_field is not None and not isinstance(
            active_stress_field, HeidlaufRoehrle2014ActiveStressField
        ):
            raise TypeError(
                "active_stress_field must be an identified 2014 quadrature field."
            )
        if (
            active_stress_field is not None
            and active_stress_field.source_id != self.plan.active_stress_source_id
        ):
            raise ValueError(
                "Quadrature stress field belongs to a foreign active-stress source."
            )
        form_id = canonical_fingerprint(
            {
                "kind": "heidlauf-roehrle-2014-native-mixed-residual",
                "material": self.prepared_id,
                "numeric_revision": self.numeric_revision().revision_id,
                "fields": (displacement_name, pressure_name),
                "active_field": None
                if active_stress_field is None
                else active_stress_field.field_id,
                "pressure_origin_pa": float(origin).hex(),
            }
        )

        def displacement_kernel(
            values, gradients, points, weights, basis_values, basis_gradients, context
        ):
            del basis_values
            deformation = jnp.swapaxes(jnp.asarray(gradients[0]), -1, -2) + jnp.eye(3)
            pressure = jnp.asarray(values[1]) + origin
            gamma = (
                jnp.full(pressure.shape, self.state.normalized_active_stress)
                if active_stress_field is None
                else jnp.asarray(active_stress_field.evaluate(points, context))
            )
            stress = self.first_piola_points(deformation, pressure, gamma)
            return contract("cq,cqad,cqid->cia", weights, stress, basis_gradients)

        def pressure_kernel(
            values, gradients, points, weights, basis_values, basis_gradients, context
        ):
            del values, points, basis_gradients, context
            deformation = jnp.swapaxes(jnp.asarray(gradients[0]), -1, -2) + jnp.eye(3)
            residual = jax.vmap(self.constraint)(deformation.reshape((-1, 3, 3))).reshape(
                deformation.shape[:-2]
            )
            return contract("cq,cq,qi->ci", weights, residual, basis_values)

        return FiniteElementForm(
            form_id,
            (displacement_name, pressure_name),
            (
                CellResidualAction(
                    displacement_name,
                    (displacement_name, pressure_name),
                    displacement_kernel,
                    action_id=f"{form_id}:displacement",
                ),
                CellResidualAction(
                    pressure_name,
                    (displacement_name,),
                    pressure_kernel,
                    action_id=f"{form_id}:pressure",
                ),
            ),
        )

    def prepare_qualified_mixed(
        self,
        finite_element_plan: MixedFiniteElementConstraintPlan,
        /,
        *,
        initial_state=None,
        args=None,
        active_stress_field: HeidlaufRoehrle2014ActiveStressField | None = None,
        pressure_origin_pa: ArrayLike = 0.0,
    ) -> PreparedMixedFiniteElementConstraint:
        """Use the core Taylor–Hood/Q2–Q1, gauge and assembled inf-sup owner."""
        if not isinstance(finite_element_plan, MixedFiniteElementConstraintPlan):
            raise TypeError(
                "finite_element_plan must be MixedFiniteElementConstraintPlan."
            )
        if (
            finite_element_plan.formulation != "exact"
            or finite_element_plan.bulk_modulus is not None
        ):
            raise ValueError("The 2014 continuum requires exact incompressibility.")
        if finite_element_plan.mesh.ambient_dimension != 3:
            raise ValueError("The source continuum requires a three-dimensional mesh.")
        form = self.form(
            finite_element_plan.displacement_field,
            finite_element_plan.pressure_field,
            active_stress_field=active_stress_field,
            pressure_origin_pa=pressure_origin_pa,
        )
        prepared = finite_element_plan.prepare(
            form, initial_state=initial_state, args=args
        )
        state = (
            prepared.problem.state_space.zeros()
            if initial_state is None
            else initial_state
        )
        if not bool(prepared.evaluate(state, args).valid):
            raise ValueError(
                "2014 mixed preparation failed finite residual, gauge or inf-sup evidence."
            )
        return prepared


__all__ = [
    "HeidlaufRoehrle2014ActiveStressField",
    "HeidlaufRoehrle2014BlockTangent",
    "HeidlaufRoehrle2014InputEvidence",
    "HeidlaufRoehrle2014MaterialCandidate",
    "HeidlaufRoehrle2014MaterialCommit",
    "HeidlaufRoehrle2014MaterialState",
    "HeidlaufRoehrle2014Parameters",
    "HeidlaufRoehrle2014Plan",
    "HeidlaufRoehrle2014PointEvidence",
    "HeidlaufRoehrle2014PointResponse",
    "HeidlaufRoehrle2014StressInput",
    "PreparedHeidlaufRoehrle2014Material",
]
