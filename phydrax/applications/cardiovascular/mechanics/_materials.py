#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import (
    MixedFiniteElementConstraintPlan,
    PreparedMixedFiniteElementConstraint,
)
from ....equations import finite_element_form_from_functional, FiniteElementForm
from ....operators.mechanics import (
    finite_strain_kinematics,
    first_piola_to_cauchy,
    HyperelasticLaw,
    HyperelasticResponse,
    VolumetricConstraint,
)
from ....variational import FieldJetSpec, Functional, LocalIntegralTerm
from ...solid_mechanics import (
    mixed_hyperelastic_form,
    MixedHyperelasticBlockTangent,
    MixedHyperelasticLaw,
    MixedHyperelasticModel,
    MixedHyperelasticResponse,
    prepare_mixed_hyperelastic_problem,
)
from ..anatomy._microstructure import CardiacMaterialFrame


IsochoricCardiacEnergy = Callable[[Array], Array]
MaterialFrameInput = CardiacMaterialFrame | ArrayLike


def resolve_material_frame(
    material_frame: MaterialFrameInput,
    /,
    *,
    frame_id: str | None = None,
    cell_index: int | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[Array, str, int | None]:
    """Resolve an anatomy cell frame or a structural basis array plus identity."""
    if isinstance(material_frame, CardiacMaterialFrame):
        if frame_id is not None and str(frame_id) != material_frame.frame_id:
            raise ValueError("frame_id does not match the anatomy material frame.")
        if cell_index is None:
            raise ValueError(
                "cell_index is required for an anatomy CardiacMaterialFrame."
            )
        index = int(cell_index)
        if index < 0 or index >= material_frame.fiber.shape[0]:
            raise ValueError("cell_index lies outside the anatomy material frame.")
        if not bool(material_frame.valid[index]):
            raise ValueError("Selected anatomy material-frame cell is invalid.")
        frame = material_frame.matrix[index]
        identifier = material_frame.frame_id
        selected_index: int | None = index
    else:
        if frame_id is None:
            raise ValueError(
                "frame_id is required for a structural material-frame array."
            )
        if cell_index is not None:
            raise ValueError("cell_index is only valid with CardiacMaterialFrame.")
        frame = jnp.asarray(material_frame)
        identifier = _identifier(frame_id, "frame_id")
        selected_index = None
    validated = validate_material_frame(
        frame,
        frame_id=identifier,
        tolerance=tolerance,
    )
    return validated, identifier, selected_index


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _point_deformation(value: ArrayLike, /) -> Array:
    deformation = jnp.asarray(value)
    if deformation.shape != (3, 3):
        raise ValueError("Cardiac material-point deformation gradients must be 3x3.")
    if not jnp.issubdtype(deformation.dtype, jnp.inexact):
        deformation = deformation.astype(float)
    if jnp.issubdtype(deformation.dtype, jnp.complexfloating):
        raise TypeError("Cardiac deformation gradients must be real.")
    return deformation


class MaterialFrameEvidence(StrictModule, NonTrainableState):
    """Evidence for a reference-frame fiber/sheet/sheet-normal triad.

    Columns are, in order, the reference fiber, sheet, and sheet-normal vectors.
    The mechanics package deliberately does not define an anatomy frame record;
    anatomy-owned frames are passed here as their structural 3x3 array plus ID.
    """

    gram_error: Array
    determinant: Array
    finite: Array
    orthonormal: Array
    right_handed: Array
    valid: Array
    frame_id: str = eqx.field(static=True)


def material_frame_evidence(
    material_frame: ArrayLike,
    /,
    *,
    frame_id: str,
    tolerance: float = 1.0e-8,
) -> MaterialFrameEvidence:
    """Validate one right-handed orthonormal material frame in reference axes."""
    identifier = _identifier(frame_id, "frame_id")
    limit = float(tolerance)
    if not isfinite(limit) or limit <= 0.0:
        raise ValueError("Material-frame tolerance must be positive and finite.")
    frame = jnp.asarray(material_frame)
    if frame.shape != (3, 3):
        raise ValueError("A material frame must be one 3x3 array with basis columns.")
    if not jnp.issubdtype(frame.dtype, jnp.inexact):
        frame = frame.astype(float)
    if jnp.issubdtype(frame.dtype, jnp.complexfloating):
        raise TypeError("Material-frame vectors must be real.")
    gram = frame.T @ frame
    gram_error = jnp.max(jnp.abs(gram - jnp.eye(3, dtype=frame.dtype)))
    determinant = jnp.linalg.det(frame)
    finite = jnp.all(jnp.isfinite(frame)) & jnp.isfinite(determinant)
    orthonormal = gram_error <= limit
    right_handed = determinant > 0.0
    return MaterialFrameEvidence(
        gram_error,
        determinant,
        finite,
        orthonormal,
        right_handed,
        finite & orthonormal & right_handed,
        identifier,
    )


def validate_material_frame(
    material_frame: ArrayLike,
    /,
    *,
    frame_id: str,
    tolerance: float = 1.0e-8,
) -> Array:
    """Return a certified frame or fail before constitutive preparation."""
    evidence = material_frame_evidence(
        material_frame,
        frame_id=frame_id,
        tolerance=tolerance,
    )
    if not bool(evidence.valid):
        raise ValueError(
            "Material frame must be finite, orthonormal, and right-handed with "
            "columns (fiber, sheet, sheet-normal)."
        )
    return jnp.asarray(material_frame)


def material_green_lagrange_strain(
    deformation_gradient: ArrayLike,
    material_frame: ArrayLike,
    /,
) -> Array:
    """Return Green--Lagrange strain components in the reference material frame."""
    deformation = _point_deformation(deformation_gradient)
    frame = jnp.asarray(material_frame, dtype=deformation.dtype)
    if frame.shape != (3, 3):
        raise ValueError("material_frame must be a 3x3 basis-column array.")
    right_cauchy_green = deformation.T @ deformation
    strain = 0.5 * (right_cauchy_green - jnp.eye(3, dtype=deformation.dtype))
    return frame.T @ strain @ frame


def material_invariants(
    deformation_gradient: ArrayLike,
    material_frame: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Return I1, I4f, I4s, and I8fs in the reference material frame."""
    deformation = _point_deformation(deformation_gradient)
    frame = jnp.asarray(material_frame, dtype=deformation.dtype)
    if frame.shape != (3, 3):
        raise ValueError("material_frame must be a 3x3 basis-column array.")
    right_cauchy_green = deformation.T @ deformation
    material_tensor = frame.T @ right_cauchy_green @ frame
    return (
        jnp.trace(right_cauchy_green),
        material_tensor[0, 0],
        material_tensor[1, 1],
        material_tensor[0, 1],
    )


class FiniteBulkCardiacMaterial(HyperelasticLaw):
    """Displacement-only isochoric cardiac law with a finite bulk penalty.

    This is a distinct fidelity route from exact mixed incompressibility. The
    energy is ``W_iso(J^(-1/3) F) + K g(F)^2 / 2``. It makes no no-locking claim;
    large ``K`` in a low-order displacement space remains a locking risk.
    """

    isochoric_energy: IsochoricCardiacEnergy
    bulk_modulus: Array
    volumetric_constraint: VolumetricConstraint
    minimum_jacobian: float = eqx.field(static=True)
    energy_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        isochoric_energy: IsochoricCardiacEnergy,
        bulk_modulus: ArrayLike,
        /,
        *,
        energy_id: str,
        volumetric_constraint: VolumetricConstraint | None = None,
        minimum_jacobian: float = 1.0e-8,
        material_id: str | None = None,
    ):
        if not callable(isochoric_energy):
            raise TypeError("isochoric_energy must be callable.")
        bulk = jnp.asarray(bulk_modulus)
        if (
            bulk.shape != ()
            or not bool(jnp.isfinite(bulk))
            or bool(bulk <= 0.0)
            or jnp.issubdtype(bulk.dtype, jnp.complexfloating)
        ):
            raise ValueError("bulk_modulus must be one positive finite real scalar.")
        constraint = (
            VolumetricConstraint("jacobian")
            if volumetric_constraint is None
            else volumetric_constraint
        )
        if not isinstance(constraint, VolumetricConstraint):
            raise TypeError("volumetric_constraint must be VolumetricConstraint or None.")
        minimum = float(minimum_jacobian)
        if not isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_jacobian must be positive and finite.")
        energy_identifier = _identifier(energy_id, "energy_id")
        generated = canonical_fingerprint(
            {
                "kind": "finite-bulk-cardiac-material",
                "energy_id": energy_identifier,
                "bulk_modulus": float(bulk).hex(),
                "constraint": constraint.kind,
                "minimum_jacobian": minimum.hex(),
            }
        )
        identifier = (
            generated if material_id is None else _identifier(material_id, "material_id")
        )
        self.isochoric_energy = isochoric_energy
        self.bulk_modulus = bulk
        self.volumetric_constraint = constraint
        self.minimum_jacobian = minimum
        self.energy_id = energy_identifier
        self.material_id = identifier

    def reference_energy_density(self, deformation_gradient: ArrayLike, /) -> Array:
        deformation = _point_deformation(deformation_gradient)
        jacobian = jnp.linalg.det(deformation)
        valid_jacobian = jnp.isfinite(jacobian) & (jacobian > self.minimum_jacobian)
        safe_jacobian = jnp.where(valid_jacobian, jacobian, 1.0)
        isochoric_deformation = deformation * safe_jacobian ** (-1.0 / 3.0)
        isochoric = jnp.asarray(self.isochoric_energy(isochoric_deformation))
        if isochoric.shape != ():
            raise ValueError("Isochoric cardiac energy must return one scalar per point.")
        constraint = self.volumetric_constraint.value(deformation)
        energy = isochoric + 0.5 * self.bulk_modulus * constraint * constraint
        valid = valid_jacobian & jnp.isfinite(energy)
        return jnp.where(valid, energy, jnp.nan)

    def evaluate(self, deformation_gradient: ArrayLike, /) -> HyperelasticResponse:
        deformation = _point_deformation(deformation_gradient)
        energy_function = self.reference_energy_density
        energy, first_piola = jax.value_and_grad(energy_function)(deformation)
        tangent = jax.jacfwd(jax.grad(energy_function))(deformation)
        kinematics = finite_strain_kinematics(deformation)
        cauchy = first_piola_to_cauchy(kinematics, first_piola)
        material_admissible = (
            (kinematics.jacobian > self.minimum_jacobian)
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(first_piola))
            & jnp.all(jnp.isfinite(tangent))
        )
        admissible = kinematics.admissible & material_admissible
        return HyperelasticResponse(
            kinematics,
            energy,
            first_piola,
            cauchy,
            tangent,
            kinematics.admissible,
            material_admissible,
            admissible,
        )

    def energy_points(self, deformation_gradients: ArrayLike, /) -> Array:
        deformation = jnp.asarray(deformation_gradients)
        if deformation.ndim < 2 or deformation.shape[-2:] != (3, 3):
            raise ValueError("Cardiac deformation-gradient fields must end in 3x3.")
        flat = deformation.reshape((-1, 3, 3))
        values = jax.vmap(self.reference_energy_density)(flat)
        return values.reshape(deformation.shape[:-2])


class ExactMixedQualification(StrictModule, NonTrainableState):
    """Independent gauge, residual, LBB-pair, and assembled inf-sup evidence."""

    inf_sup_constant: Array
    gauge_valid: Array
    residual_finite: Array
    stable_pair: Array
    assembled_inf_sup_stable: Array
    locking_safe: Array
    valid: Array
    pair_names: tuple[str, ...] = eqx.field(static=True)
    gauge_mode: str = eqx.field(static=True)


class QualifiedExactIncompressibleProblem(StrictModule, NonTrainableState):
    """Prepared exact u-p problem admitted only after fail-closed qualification."""

    prepared: PreparedMixedFiniteElementConstraint
    qualification: ExactMixedQualification
    material_id: str = eqx.field(static=True)


class ExactIncompressibleCardiacMaterial(StrictModule, NonTrainableState):
    """Exact mixed u-p cardiac route using the generic LBB-qualified FEM substrate.

    No displacement-only P1 route exists here. Preparation is restricted by
    ``MixedFiniteElementConstraintPlan`` to Taylor--Hood P2/P1 or Q2/Q1 and an
    explicit pressure gauge; assembled inf-sup evidence must pass before the
    qualified problem is returned.
    """

    law: MixedHyperelasticLaw
    model: MixedHyperelasticModel
    energy_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        isochoric_energy: IsochoricCardiacEnergy,
        /,
        *,
        energy_id: str,
        volumetric_constraint: VolumetricConstraint | None = None,
        minimum_jacobian: float = 1.0e-8,
        material_id: str | None = None,
    ):
        if not callable(isochoric_energy):
            raise TypeError("isochoric_energy must be callable.")
        constraint = (
            VolumetricConstraint("jacobian")
            if volumetric_constraint is None
            else volumetric_constraint
        )
        if not isinstance(constraint, VolumetricConstraint):
            raise TypeError("volumetric_constraint must be VolumetricConstraint or None.")
        minimum = float(minimum_jacobian)
        if not isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_jacobian must be positive and finite.")
        energy_identifier = _identifier(energy_id, "energy_id")
        generated = canonical_fingerprint(
            {
                "kind": "exact-incompressible-cardiac-material",
                "energy_id": energy_identifier,
                "constraint": constraint.kind,
                "minimum_jacobian": minimum.hex(),
            }
        )
        identifier = (
            generated if material_id is None else _identifier(material_id, "material_id")
        )
        law = MixedHyperelasticLaw(
            isochoric_energy,
            constraint.value,
            minimum_jacobian=minimum,
        )
        self.law = law
        self.model = MixedHyperelasticModel(law)
        self.energy_id = energy_identifier
        self.material_id = identifier

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> MixedHyperelasticResponse:
        return self.law.evaluate(deformation_gradient, pressure)

    def block_tangent(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> MixedHyperelasticBlockTangent:
        return self.law.block_tangent(deformation_gradient, pressure)

    def form(
        self,
        displacement_field: str = "u",
        pressure_field: str = "p",
        /,
        *,
        form_id: str = "cardiac-exact-incompressible-equilibrium",
    ) -> FiniteElementForm:
        return mixed_hyperelastic_form(
            displacement_field,
            pressure_field,
            self.model,
            form_id=form_id,
        )

    def prepare_qualified(
        self,
        plan: MixedFiniteElementConstraintPlan,
        /,
        *,
        initial_state: tuple[ArrayLike, ArrayLike] | None = None,
        args: object = None,
        form_id: str = "cardiac-exact-incompressible-equilibrium",
    ) -> QualifiedExactIncompressibleProblem:
        if not isinstance(plan, MixedFiniteElementConstraintPlan):
            raise TypeError("plan must be MixedFiniteElementConstraintPlan.")
        if plan.formulation != "exact" or plan.bulk_modulus is not None:
            raise ValueError("Exact cardiac material requires an exact mixed FE plan.")
        prepared = prepare_mixed_hyperelastic_problem(
            self.model,
            plan,
            initial_state=initial_state,
            args=args,
            form_id=form_id,
        )
        spaces = prepared.spaces
        qualification_state = (
            prepared.problem.state_space.zeros()
            if initial_state is None
            else prepared.problem.state_space.validate(initial_state)
        )
        evaluation = prepared.evaluate(qualification_state, args)
        gauge_valid = jnp.asarray(evaluation.gauge.valid)
        residual_finite = jnp.asarray(evaluation.finite)
        stable_pair = jnp.asarray(
            spaces.displacement_degree == 2
            and spaces.pressure_degree == 1
            and spaces.lbb_conforming
            and spaces.stabilization_absent
        )
        assembled = jnp.asarray(evaluation.inf_sup.stable)
        locking_safe = jnp.asarray(
            spaces.locking_safe and evaluation.inf_sup.locking_safe
        )
        valid = gauge_valid & residual_finite & stable_pair & assembled & locking_safe
        qualification = ExactMixedQualification(
            jnp.asarray(evaluation.inf_sup.inf_sup_constant),
            gauge_valid,
            residual_finite,
            stable_pair,
            assembled,
            locking_safe,
            valid,
            spaces.pair_names,
            prepared.gauge.mode,
        )
        if not bool(qualification.valid):
            raise ValueError(
                "Exact cardiac mixed preparation failed gauge, finite-residual, "
                "LBB, inf-sup, or locking-safety qualification."
            )
        return QualifiedExactIncompressibleProblem(
            prepared,
            qualification,
            self.material_id,
        )


def cardiac_passive_functional(
    field_name: str,
    material: FiniteBulkCardiacMaterial,
    /,
    *,
    region: str = "myocardium",
    functional_id: str = "cardiac-passive-equilibrium",
) -> Functional:
    """Build a representation-independent finite-bulk passive energy functional."""
    field = _identifier(field_name, "field_name")
    region_ = _identifier(region, "region")
    identifier = _identifier(functional_id, "functional_id")
    if not isinstance(material, FiniteBulkCardiacMaterial):
        raise TypeError("material must be FiniteBulkCardiacMaterial.")

    def density(fields, geometry, context):
        del geometry, context
        gradient = fields[field].gradient
        if gradient is None:
            raise ValueError("Cardiac passive energy requires a displacement gradient.")
        gradient_ = jnp.asarray(gradient)
        if gradient_.shape[-2:] != (3, 3):
            raise ValueError("Cardiac displacement gradients must end in 3x3.")
        deformation = gradient_ + jnp.eye(3, dtype=gradient_.dtype)
        return material.energy_points(deformation)

    return Functional(
        identifier,
        (
            LocalIntegralTerm(
                "cardiac-passive-stored-energy",
                region=region_,
                fields=(FieldJetSpec(field, gradient=True),),
                density=density,
                density_id=canonical_fingerprint(
                    {
                        "kind": "cardiac-passive-stored-energy",
                        "material_id": material.material_id,
                    }
                ),
            ),
        ),
        variable_fields=(field,),
    )


def cardiac_passive_form(
    field_name: str,
    material: FiniteBulkCardiacMaterial,
    /,
    *,
    region: str = "myocardium",
    form_id: str = "cardiac-passive-equilibrium",
) -> FiniteElementForm:
    """Lower the passive energy to the generic finite-element form substrate."""
    functional = cardiac_passive_functional(
        field_name,
        material,
        region=region,
        functional_id=form_id,
    )
    return finite_element_form_from_functional(
        functional,
        {field_name: field_name},
        {region: None},
        form_id=form_id,
    )


__all__ = [
    "ExactIncompressibleCardiacMaterial",
    "ExactMixedQualification",
    "FiniteBulkCardiacMaterial",
    "IsochoricCardiacEnergy",
    "MaterialFrameInput",
    "MaterialFrameEvidence",
    "QualifiedExactIncompressibleProblem",
    "cardiac_passive_form",
    "cardiac_passive_functional",
    "material_frame_evidence",
    "material_green_lagrange_strain",
    "material_invariants",
    "resolve_material_frame",
    "validate_material_frame",
]
