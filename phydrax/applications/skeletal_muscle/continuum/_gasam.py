#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import (
    MixedFiniteElementConstraintPlan,
    PreparedMixedFiniteElementConstraint,
)
from ....equations import FiniteElementForm
from ....operators.mechanics import finite_strain_kinematics, VolumetricConstraint
from ...solid_mechanics import (
    mixed_hyperelastic_form,
    MixedHyperelasticBlockTangent,
    MixedHyperelasticLaw,
    MixedHyperelasticModel,
    MixedHyperelasticResponse,
    prepare_mixed_hyperelastic_problem,
)
from ._fiber import PreparedUniformFiberArchitecture


_SOURCE_ID = "doi:10.1002/cnm.70036"
_PARAMETER_SCHEMA_ID = canonical_fingerprint(
    {
        "kind": "engelhardt-gasam-2025-parameter-schema",
        "source": _SOURCE_ID,
        "fields": (
            "alpha",
            "beta",
            "stiffness_pa",
            "isotropic_weight",
            "minimum_active_stretch",
            "optimal_active_stretch",
            "peak_active_nominal_stress_pa",
        ),
    }
)


def _scalar(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.shape != ():
        raise ValueError(f"{name} must be one scalar.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be real-valued.")
    return result


def _positive(value: ArrayLike, name: str, /) -> Array:
    result = _scalar(value, name)
    if not bool(jnp.isfinite(result)) or bool(result <= 0.0):
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class EngelhardtGasam2025Parameters(StrictModule):
    """Dynamic parameters for the 2025 GASAM continuum law.

    Stress-valued leaves use Pa.  Dimensionless leaves retain the notation in
    Engelhardt et al. (2025), Eqs. (15), (16), (20), and (26).  The material is
    one coupled active/passive law; these fields do not define separable force
    routes.
    """

    alpha: Array
    beta: Array
    stiffness_pa: Array
    isotropic_weight: Array
    minimum_active_stretch: Array
    optimal_active_stretch: Array
    peak_active_nominal_stress_pa: Array

    def __init__(
        self,
        alpha: ArrayLike,
        beta: ArrayLike,
        stiffness_pa: ArrayLike,
        isotropic_weight: ArrayLike,
        minimum_active_stretch: ArrayLike,
        optimal_active_stretch: ArrayLike,
        peak_active_nominal_stress_pa: ArrayLike,
        /,
    ):
        alpha_ = _positive(alpha, "alpha")
        beta_ = _positive(beta, "beta")
        stiffness = _positive(stiffness_pa, "stiffness_pa")
        weight = _scalar(isotropic_weight, "isotropic_weight")
        minimum = _positive(minimum_active_stretch, "minimum_active_stretch")
        optimal = _positive(optimal_active_stretch, "optimal_active_stretch")
        peak = _scalar(
            peak_active_nominal_stress_pa, "peak_active_nominal_stress_pa"
        )
        if not bool(jnp.isfinite(weight)) or bool(
            (weight <= 0.0) | (weight >= 1.0)
        ):
            raise ValueError(
                "isotropic_weight must lie strictly between zero and one."
            )
        if bool(optimal <= minimum):
            raise ValueError(
                "optimal_active_stretch must exceed minimum_active_stretch."
            )
        if not bool(jnp.isfinite(peak)) or bool(peak < 0.0):
            raise ValueError(
                "peak_active_nominal_stress_pa must be finite and nonnegative."
            )
        self.alpha = alpha_
        self.beta = beta_
        self.stiffness_pa = stiffness
        self.isotropic_weight = weight
        self.minimum_active_stretch = minimum
        self.optimal_active_stretch = optimal
        self.peak_active_nominal_stress_pa = peak

    @property
    def schema_id(self) -> str:
        return _PARAMETER_SCHEMA_ID

    @classmethod
    def published_multiload_fit(cls) -> EngelhardtGasam2025Parameters:
        """Return Table 5's GASAM fit, converted from kPa to Pa."""
        return cls(2.3796, 0.5161, 27_107.2, 0.6388, 0.5680, 1.1806, 64_680.9)


class PrescribedActivationEvidence(StrictModule, NonTrainableState):
    """Closed-support evidence for the normalized prescribed activation."""

    activation: Array
    finite: Array
    in_support: Array
    valid: Array
    source_id: str = eqx.field(static=True)


class GasamMaterialPointEvidence(StrictModule, NonTrainableState):
    """Material-point domain and branch evidence for one GASAM evaluation."""

    fiber_stretch: Array
    passive_first_invariant: Array
    cofactor_invariant: Array
    force_length: Array
    integrated_force_length: Array
    activation_weight: Array
    activation_branch_smooth: Array
    mixed_valid: Array
    valid: Array
    source_id: str = eqx.field(static=True)


class GasamMaterialPointResponse(StrictModule):
    """Exact mixed material response plus source-level active invariants."""

    mixed: MixedHyperelasticResponse
    evidence: GasamMaterialPointEvidence

    @property
    def reference_energy_density(self) -> Array:
        return self.mixed.isochoric_energy

    @property
    def first_piola(self) -> Array:
        return self.mixed.first_piola


class GasamMaterialState(StrictModule, NonTrainableState):
    """Committed prescribed activation for one prepared material route."""

    activation: Array
    evidence: PrescribedActivationEvidence
    state_id: Array


class GasamMaterialCommit(StrictModule, NonTrainableState):
    """Atomic update result; rejection contains the complete rollback state."""

    state: GasamMaterialState
    committed: Array
    rollback_applied: Array
    source_state_id: Array
    source_activation: Array
    prepared_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)


class GasamMaterialCandidate(StrictModule, NonTrainableState):
    """Uncommitted activation update retaining complete prior state."""

    previous: GasamMaterialState
    proposed: GasamMaterialState
    evidence: PrescribedActivationEvidence
    source_state_id: Array
    source_activation: Array
    prepared_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def commit(self, /) -> GasamMaterialCommit:
        """Commit a valid update, otherwise atomically select ``previous``."""
        accepted = self.evidence.valid
        previous_evidence = self.previous.evidence
        proposed_evidence = self.proposed.evidence
        selected_evidence = PrescribedActivationEvidence(
            jnp.where(
                accepted,
                proposed_evidence.activation,
                previous_evidence.activation,
            ),
            jnp.where(accepted, proposed_evidence.finite, previous_evidence.finite),
            jnp.where(
                accepted,
                proposed_evidence.in_support,
                previous_evidence.in_support,
            ),
            jnp.where(accepted, proposed_evidence.valid, previous_evidence.valid),
            _SOURCE_ID,
        )
        selected = GasamMaterialState(
            jnp.where(
                accepted,
                self.proposed.activation,
                self.previous.activation,
            ),
            selected_evidence,
            jnp.where(
                accepted,
                self.proposed.state_id,
                self.previous.state_id,
            ),
        )
        return GasamMaterialCommit(
            selected,
            accepted,
            ~accepted,
            self.source_state_id,
            self.previous.activation,
            self.prepared_id,
            self.candidate_id,
        )


class ExactMixedGasamQualification(StrictModule, NonTrainableState):
    """Gauge, residual, LBB-pair, and assembled inf-sup evidence."""

    inf_sup_constant: Array
    gauge_valid: Array
    residual_finite: Array
    stable_pair: Array
    assembled_inf_sup_stable: Array
    locking_safe: Array
    valid: Array
    pair_names: tuple[str, ...] = eqx.field(static=True)
    gauge_mode: str = eqx.field(static=True)


class QualifiedExactMixedGasamProblem(StrictModule, NonTrainableState):
    """Exact u-p GASAM problem admitted only after mixed-FEM qualification."""

    prepared: PreparedMixedFiniteElementConstraint
    qualification: ExactMixedGasamQualification
    material_id: str = eqx.field(static=True)


class EngelhardtGasam2025Plan(StrictModule, NonTrainableState):
    """Static identity and admissibility plan for one source-complete GASAM route."""

    minimum_jacobian: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_id: str,
        /,
        *,
        minimum_jacobian: float = 1.0e-8,
    ):
        identifier = _identifier(material_id, "material_id")
        minimum = float(minimum_jacobian)
        if not isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_jacobian must be positive and finite.")
        self.minimum_jacobian = minimum
        self.material_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "engelhardt-gasam-2025-plan",
                "material_id": identifier,
                "minimum_jacobian": minimum.hex(),
                "source": _SOURCE_ID,
                "incompressibility": "exact-mixed",
            }
        )

    def prepare(
        self,
        parameters: EngelhardtGasam2025Parameters,
        architecture: PreparedUniformFiberArchitecture,
        prescribed_activation: ArrayLike,
        /,
    ) -> PreparedEngelhardtGasam2025Material:
        if not isinstance(parameters, EngelhardtGasam2025Parameters):
            raise TypeError("parameters must be EngelhardtGasam2025Parameters.")
        if not isinstance(architecture, PreparedUniformFiberArchitecture):
            raise TypeError("architecture must be PreparedUniformFiberArchitecture.")
        if not bool(architecture.evidence.valid):
            raise ValueError(
                "A GASAM material requires valid supported fiber architecture."
            )
        activation_evidence = _activation_evidence(prescribed_activation)
        if not bool(activation_evidence.valid):
            raise ValueError("prescribed_activation must be finite and in [0, 1].")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-engelhardt-gasam-2025",
                "plan": self.plan_id,
                "architecture": architecture.prepared_id,
                "parameter_schema": parameters.schema_id,
            }
        )
        state = _material_state(
            activation_evidence.activation,
            activation_evidence,
            jnp.asarray(0, dtype=jnp.uint32),
        )
        return PreparedEngelhardtGasam2025Material(
            self,
            parameters,
            architecture,
            state,
            prepared_id,
        )


class PreparedEngelhardtGasam2025Material(StrictModule):
    """Prepared exact-incompressible, prescribed-activation GASAM material.

    This is a terminal active/passive continuum fidelity.  It does not expose a
    passive substrate or accept external fiber tension.  Stress and tangent are
    derivatives of the complete Eq. (16)/(26) potential.  The source force-length
    switch at ``lambda_min`` is a hard constitutive branch, so AD claims are local
    to either open branch and never global across the switch.
    """

    plan: EngelhardtGasam2025Plan
    parameters: EngelhardtGasam2025Parameters
    architecture: PreparedUniformFiberArchitecture
    state: GasamMaterialState
    prepared_id: str = eqx.field(static=True)

    def propose_activation(
        self, prescribed_activation: ArrayLike, /
    ) -> GasamMaterialCandidate:
        evidence = _activation_evidence(prescribed_activation)
        proposed = _material_state(
            evidence.activation,
            evidence,
            self.state.state_id + jnp.asarray(1, dtype=self.state.state_id.dtype),
        )
        return GasamMaterialCandidate(
            self.state,
            proposed,
            evidence,
            self.state.state_id,
            self.state.activation,
            self.prepared_id,
            canonical_fingerprint(
                {
                    "kind": "gasam-prescribed-activation-candidate",
                    "prepared": self.prepared_id,
                }
            ),
        )

    def with_commit(
        self, commit: GasamMaterialCommit, /
    ) -> PreparedEngelhardtGasam2025Material:
        """Return a prepared material carrying the atomically selected state."""
        if not isinstance(commit, GasamMaterialCommit):
            raise TypeError("commit must be GasamMaterialCommit.")
        if commit.prepared_id != self.prepared_id:
            raise ValueError("GASAM commit belongs to a different prepared material.")
        source_mismatch = (
            (commit.source_state_id != self.state.state_id)
            | (commit.source_activation != self.state.activation)
        )
        checked_activation = eqx.error_if(
            commit.state.activation,
            source_mismatch,
            "GASAM commit belongs to a stale or different source state.",
        )
        checked_state = GasamMaterialState(
            checked_activation,
            commit.state.evidence,
            commit.state.state_id,
        )
        return PreparedEngelhardtGasam2025Material(
            self.plan,
            self.parameters,
            self.architecture,
            checked_state,
            self.prepared_id,
        )

    def _base_terms(
        self, deformation_gradient: ArrayLike, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape != (3, 3):
            raise ValueError(
                "A GASAM material-point deformation must have shape (3, 3)."
            )
        kinematics = finite_strain_kinematics(deformation)
        c = kinematics.right_cauchy_green
        inverse_c = ein.contract(
            "ik,jk->ij",
            kinematics.inverse_deformation_gradient,
            kinematics.inverse_deformation_gradient,
        )
        cofactor_c = (kinematics.jacobian * kinematics.jacobian) * inverse_c
        structural_tensor = self.architecture.structural_tensor
        weight = self.parameters.isotropic_weight
        structure = (weight / 3.0) * jnp.eye(3, dtype=c.dtype) + (
            1.0 - weight
        ) * structural_tensor
        passive_first = ein.contract("ij,ij->", c, structure)
        cofactor_invariant = ein.contract("ij,ij->", cofactor_c, structure)
        fiber_square = ein.contract("ij,ij->", c, structural_tensor)
        fiber_stretch = jnp.sqrt(fiber_square)
        force_length, integrated = _force_length_terms(
            fiber_stretch,
            self.parameters.minimum_active_stretch,
            self.parameters.optimal_active_stretch,
        )
        return (
            fiber_stretch,
            passive_first,
            cofactor_invariant,
            force_length,
            integrated,
        )

    def source_terms(
        self, deformation_gradient: ArrayLike, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Return lambda, I_p, J-tilde, f_xi, integral(f_xi), and omega_a."""
        (
            fiber_stretch,
            passive_first,
            cofactor_invariant,
            force_length,
            integrated,
        ) = self._base_terms(deformation_gradient)
        phi = 1.0 + (
            4.0
            * self.parameters.alpha
            / self.parameters.stiffness_pa
            * jnp.exp(self.parameters.alpha * (1.0 - passive_first))
            * self.parameters.peak_active_nominal_stress_pa
            * self.state.activation
            * integrated
        )
        fiber_square = fiber_stretch * fiber_stretch
        safe_square = jnp.where(fiber_square > 0.0, fiber_square, 1.0)
        activation_weight = jnp.log(phi) / (self.parameters.alpha * safe_square)
        activation_weight = jnp.where(
            fiber_square > 0.0, activation_weight, jnp.nan
        )
        return (
            fiber_stretch,
            passive_first,
            cofactor_invariant,
            force_length,
            integrated,
            activation_weight,
        )

    def reference_energy_density(self, deformation_gradient: ArrayLike, /) -> Array:
        """Evaluate the complete exact-incompressible GASAM reference energy."""
        (
            _fiber_stretch,
            passive_first,
            cofactor_invariant,
            _force_length,
            integrated,
        ) = self._base_terms(deformation_gradient)
        passive_energy = 0.25 * self.parameters.stiffness_pa * (
            jnp.expm1(self.parameters.alpha * (passive_first - 1.0))
            / self.parameters.alpha
            + jnp.expm1(self.parameters.beta * (cofactor_invariant - 1.0))
            / self.parameters.beta
        )
        # Substitution of Eq. (26) into Eq. (16) reduces the complete active
        # contribution exactly to P_opt * activation * integral(f_xi).
        energy = (
            passive_energy
            + self.parameters.peak_active_nominal_stress_pa
            * self.state.activation
            * integrated
        )
        return jnp.where(jnp.isfinite(energy), energy, jnp.nan)

    def mixed_law(self, /) -> MixedHyperelasticLaw:
        """Bind the complete potential to Phydrax's exact mixed owner."""
        return MixedHyperelasticLaw(
            self.reference_energy_density,
            VolumetricConstraint("jacobian").value,
            minimum_jacobian=self.plan.minimum_jacobian,
        )

    def mixed_model(self, /) -> MixedHyperelasticModel:
        return MixedHyperelasticModel(self.mixed_law())

    def evaluate(
        self, deformation_gradient: ArrayLike, pressure_pa: ArrayLike, /
    ) -> GasamMaterialPointResponse:
        mixed = self.mixed_law().evaluate(deformation_gradient, pressure_pa)
        terms = self.source_terms(mixed.isochoric_deformation_gradient)
        (
            stretch,
            passive_first,
            cofactor_invariant,
            force_length,
            integrated,
            weight,
        ) = terms
        finite = jnp.all(jnp.isfinite(jnp.stack(terms)))
        branch_distance = jnp.abs(
            stretch - self.parameters.minimum_active_stretch
        )
        smooth = (
            branch_distance > 8.0 * jnp.finfo(stretch.dtype).eps
        )
        evidence = GasamMaterialPointEvidence(
            stretch,
            passive_first,
            cofactor_invariant,
            force_length,
            integrated,
            weight,
            smooth,
            mixed.evidence.valid,
            finite & smooth & mixed.evidence.valid,
            _SOURCE_ID,
        )
        return GasamMaterialPointResponse(mixed, evidence)

    def block_tangent(
        self, deformation_gradient: ArrayLike, pressure_pa: ArrayLike, /
    ) -> MixedHyperelasticBlockTangent:
        """Return all exact u-p derivative blocks from the native mixed owner."""
        return self.mixed_law().block_tangent(deformation_gradient, pressure_pa)

    def form(
        self,
        displacement_field: str = "u",
        pressure_field: str = "p",
        /,
        *,
        form_id: str = "engelhardt-gasam-2025-exact-mixed-equilibrium",
    ) -> FiniteElementForm:
        return mixed_hyperelastic_form(
            displacement_field,
            pressure_field,
            self.mixed_model(),
            form_id=form_id,
        )

    def prepare_qualified_mixed(
        self,
        finite_element_plan: MixedFiniteElementConstraintPlan,
        /,
        *,
        initial_state: tuple[ArrayLike, ArrayLike] | None = None,
        args: object = None,
    ) -> QualifiedExactMixedGasamProblem:
        """Prepare and fail-close qualify the existing Taylor-Hood/Q2-Q1 owner."""
        if not isinstance(finite_element_plan, MixedFiniteElementConstraintPlan):
            raise TypeError(
                "finite_element_plan must be MixedFiniteElementConstraintPlan."
            )
        if (
            finite_element_plan.formulation != "exact"
            or finite_element_plan.bulk_modulus is not None
        ):
            raise ValueError("GASAM requires an exact mixed finite-element plan.")
        prepared = prepare_mixed_hyperelastic_problem(
            self.mixed_model(),
            finite_element_plan,
            initial_state=initial_state,
            args=args,
            form_id="engelhardt-gasam-2025-exact-mixed-equilibrium",
        )
        state = (
            prepared.problem.state_space.zeros()
            if initial_state is None
            else prepared.problem.state_space.validate(initial_state)
        )
        evaluation = prepared.evaluate(state, args)
        spaces = prepared.spaces
        stable_pair = jnp.asarray(
            spaces.displacement_degree == 2
            and spaces.pressure_degree == 1
            and spaces.lbb_conforming
            and spaces.stabilization_absent
        )
        qualification = ExactMixedGasamQualification(
            jnp.asarray(evaluation.inf_sup.inf_sup_constant),
            jnp.asarray(evaluation.gauge.valid),
            jnp.asarray(evaluation.finite),
            stable_pair,
            jnp.asarray(evaluation.inf_sup.stable),
            jnp.asarray(spaces.locking_safe and evaluation.inf_sup.locking_safe),
            jnp.asarray(evaluation.valid) & stable_pair,
            spaces.pair_names,
            prepared.gauge.mode,
        )
        if not bool(qualification.valid):
            raise ValueError(
                "GASAM mixed preparation failed gauge, finite-residual, LBB, inf-sup, "
                "or locking-safety qualification."
            )
        return QualifiedExactMixedGasamProblem(
            prepared,
            qualification,
            self.plan.material_id,
        )


def _activation_evidence(value: ArrayLike, /) -> PrescribedActivationEvidence:
    activation = _scalar(value, "prescribed_activation")
    finite = jnp.isfinite(activation)
    support = (activation >= 0.0) & (activation <= 1.0)
    return PrescribedActivationEvidence(
        activation, finite, support, finite & support, _SOURCE_ID
    )


def _material_state(
    activation: Array,
    evidence: PrescribedActivationEvidence,
    state_id: Array,
    /,
) -> GasamMaterialState:
    return GasamMaterialState(
        activation,
        evidence,
        state_id,
    )


def _force_length_terms(
    stretch: Array, minimum: Array, optimal: Array, /
) -> tuple[Array, Array]:
    """Evaluate Eq. (20) and its exact lower-limit integral used by Eq. (26)."""
    width = optimal - minimum
    distance = stretch - minimum
    exponent = 0.5 - 0.5 * (distance / width) ** 2
    active = stretch > minimum
    force_length = jnp.where(
        active, distance / width * jnp.exp(exponent), 0.0
    )
    integrated = jnp.where(
        active,
        width * (jnp.exp(jnp.asarray(0.5, dtype=stretch.dtype)) - jnp.exp(exponent)),
        0.0,
    )
    return force_length, integrated


__all__ = [
    "EngelhardtGasam2025Parameters",
    "EngelhardtGasam2025Plan",
    "ExactMixedGasamQualification",
    "GasamMaterialCandidate",
    "GasamMaterialCommit",
    "GasamMaterialPointEvidence",
    "GasamMaterialPointResponse",
    "GasamMaterialState",
    "PrescribedActivationEvidence",
    "PreparedEngelhardtGasam2025Material",
    "QualifiedExactMixedGasamProblem",
]
