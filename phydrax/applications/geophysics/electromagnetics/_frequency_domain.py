#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.linalg as la
from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import CellMesh, TetrahedralConnectivity
from ....discretization.fem._de_rham import FiniteElementDeRhamComplex
from ....exterior._complex import ComplexBoundary
from ....linalg._complexes import coordinate_operator, coordinate_space
from ....typing import checked


def _material_stiffness(
    complex: FiniteElementDeRhamComplex,
    coefficient: Array,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> la.AbstractLinearOperator:
    differential = coordinate_operator(
        complex.hilbert_complex(boundary=boundary).differential(1)
    )
    metric = complex.constitutive_operator(2, coefficient, boundary=boundary)
    return la.assemble_sparse(la.adjoint(differential) @ metric @ differential)


def _hx_preconditioner(
    complex: FiniteElementDeRhamComplex,
    surrogate: la.AbstractLinearOperator,
    /,
) -> la.AbstractPreconditioner:
    hilbert = complex.hilbert_complex(boundary="relative")
    interpolation = coordinate_operator(
        complex.vector_interpolation(1, boundary="relative")
    )
    scalar = la.assemble_sparse(la.stiffness_form(hilbert, 0) + la.mass_form(hilbert, 0))
    scalar_diagonal = la.assemble_diagonal(scalar)
    potential = la.DiagonalPreconditioner(
        scalar_diagonal,
        space=coordinate_space(hilbert.space(0)),
        positive_definite=True,
    )
    components = complex.dimension
    vector = la.DiagonalPreconditioner(
        jnp.repeat(scalar_diagonal, components),
        space=interpolation.source,
        positive_definite=True,
    )
    smoother = la.DiagonalPreconditioner(
        la.assemble_diagonal(surrogate),
        space=surrogate.source,
        positive_definite=True,
    )
    builder = la.hiptmair_xu_preconditioner_builder(
        hilbert,
        1,
        vector_interpolation=complex.vector_interpolation(1, boundary="relative"),
        vector_builder=vector,
        potential_builder=potential,
        smoother=smoother,
    )
    return builder.prepare(surrogate, materialization=la.MaterializationPolicy())


def _complex_action(operator: la.AbstractLinearOperator, values: ArrayLike, /) -> Array:
    field = jnp.asarray(values)
    dtype = coordinate_space(operator.source).dtype
    return operator.mv(jnp.real(field).astype(dtype)) + 1j * operator.mv(
        jnp.imag(field).astype(dtype)
    )


@final
class ConductiveEMMaterial(StrictModule):
    conductivity_S_m: Array
    permittivity_F_m: Array
    inverse_permeability_m_H: Array

    def __init__(
        self,
        conductivity_S_m: ArrayLike,
        permittivity_F_m: ArrayLike,
        inverse_permeability_m_H: ArrayLike,
        cell_count: int,
        /,
    ) -> None:
        self.conductivity_S_m = self._tensor(conductivity_S_m, cell_count, "conductivity")
        self.permittivity_F_m = self._tensor(permittivity_F_m, cell_count, "permittivity")
        self.inverse_permeability_m_H = self._tensor(
            inverse_permeability_m_H, cell_count, "inverse permeability"
        )

    @staticmethod
    def _tensor(value: ArrayLike, count: int, name: str) -> Array:
        raw = jnp.asarray(value)
        if raw.shape in ((), (count,)):
            scalar = jnp.broadcast_to(raw, (count,))
            tensor = scalar[:, None, None] * jnp.eye(3)
        elif raw.shape == (3, 3):
            tensor = jnp.broadcast_to(raw, (count, 3, 3))
        elif raw.shape == (count, 3, 3):
            tensor = raw
        else:
            raise ValueError(f"EM {name} must be scalar/cell scalar or 3x3 tensor.")
        hermitian = 0.5 * (tensor + jnp.swapaxes(tensor.conj(), -1, -2))
        eigenvalues = jnp.linalg.eigvalsh(hermitian)
        return eqx.error_if(
            hermitian,
            jnp.any(~jnp.isfinite(tensor))
            | jnp.any(jnp.abs(tensor - jnp.swapaxes(tensor.conj(), -1, -2)) > 1e-10)
            | jnp.any(eigenvalues <= 0),
            f"EM {name} must be finite Hermitian positive definite.",
        )


@final
class FrequencyDomainEMSurvey(StrictModule, NonTrainableState):
    electric_current_functionals: Array
    receiver_functionals: Array
    source_indices: Array
    survey_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_current_functionals: ArrayLike,
        receiver_functionals: ArrayLike,
        source_indices: ArrayLike,
        /,
    ) -> None:
        sources = jnp.asarray(electric_current_functionals)
        receivers = jnp.asarray(receiver_functionals)
        indices = np.asarray(source_indices)
        if sources.ndim != 2 or sources.shape[0] == 0:
            raise ValueError("Frequency EM sources must have shape (sources, edge dofs).")
        if (
            receivers.ndim != 2
            or receivers.shape[1] != sources.shape[1]
            or receivers.shape[0] == 0
        ):
            raise ValueError(
                "Frequency EM receivers must have shape (measurements, edge dofs)."
            )
        if indices.shape != (receivers.shape[0],) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("Frequency EM source indices have wrong shape/type.")
        sources = eqx.error_if(
            sources,
            jnp.any(~jnp.isfinite(sources))
            | jnp.any(~jnp.isfinite(receivers))
            | jnp.asarray(np.any(indices < 0) or np.any(indices >= sources.shape[0])),
            "Frequency EM survey functionals and source indices must be finite and valid.",
        )
        self.electric_current_functionals = sources
        self.receiver_functionals = receivers
        self.source_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.survey_id = canonical_fingerprint(
            {
                "kind": "frequency-domain-em-survey",
                "sources": np.asarray(sources),
                "receivers": np.asarray(receivers),
                "source_indices": indices,
            }
        )


@final
class FrequencyDomainEMResult(StrictModule):
    angular_frequencies: Array
    observations: Array
    electric_fields: Array
    residual_norms: Array
    dissipated_power_W: Array
    successful: Array


@final
class FrequencyDomainEMPlan(StrictModule, NonTrainableState):
    """3D tetrahedral H(curl) conductive Maxwell with PEC outer boundary.

    The exp(-i omega t) operator is curl(mu^-1 curl E)-omega^2 epsilon E
    -i omega sigma E. Source rows represent weak electric-current functionals;
    the operator right-hand side is i omega J. Primary-secondary solves use an
    explicitly supplied background material and primary field.
    """

    complex: FiniteElementDeRhamComplex
    survey: FrequencyDomainEMSurvey
    free_edges: Array
    reduced_space: la.ArraySpace
    policy: la.LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(self, mesh: CellMesh, survey: FrequencyDomainEMSurvey, /) -> None:
        complex = FiniteElementDeRhamComplex(
            mesh, family="trimmed", order=1, coefficient_dtype=jnp.complex128
        )
        if survey.electric_current_functionals.shape[1] != complex.cell_counts[1]:
            raise ValueError("Frequency-domain EM survey does not match H(curl) edges.")
        connectivity = mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("Frequency-domain EM requires tetrahedral connectivity.")
        boundary = np.asarray(connectivity.boundary_edges)
        sources = np.asarray(survey.electric_current_functionals)
        if np.any(sources[:, boundary] != 0):
            raise ValueError("PEC boundary edge source functionals must be zero.")
        free = np.flatnonzero(~boundary)
        if free.size == 0:
            raise ValueError("Frequency-domain EM mesh has no interior H(curl) edges.")
        self.complex, self.survey = complex, survey
        self.free_edges = jnp.asarray(free, dtype=jnp.int32)
        self.reduced_space = la.ArraySpace((free.size,), dtype=jnp.complex128)
        self.policy = la.LinearSolvePolicy(
            la.GMRES(restart=50, stagnation_iterations=50),
            tolerance=la.TolerancePolicy(relative=1e-8, absolute=1e-11, max_steps=2000),
            failure=la.FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "frequency-domain-em-plan",
                "hcurl": complex.realization_id,
                "survey": survey.survey_id,
                "boundary": "pec",
            }
        )

    def _operator(
        self, angular_frequency: Array, material: ConductiveEMMaterial
    ) -> tuple[la.FunctionLinearOperator, Callable[[ArrayLike], Array]]:
        omega = angular_frequency
        stiffness = _material_stiffness(self.complex, material.inverse_permeability_m_H)
        permittivity = self.complex.constitutive_operator(1, material.permittivity_F_m)
        conductivity = self.complex.constitutive_operator(1, material.conductivity_S_m)

        def full_action(field: ArrayLike) -> Array:
            return (
                _complex_action(stiffness, field)
                - omega**2 * _complex_action(permittivity, field)
                - 1j * omega * _complex_action(conductivity, field)
            )

        def reduced_action(values: Array) -> Array:
            full = (
                jnp.zeros((self.complex.cell_counts[1],), dtype=values.dtype)
                .at[self.free_edges]
                .set(values)
            )
            return full_action(full)[self.free_edges]

        return la.FunctionLinearOperator(
            reduced_action,
            source=self.reduced_space,
            target=self.reduced_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "conductive-frequency-em-operator",
                    "plan": self.plan_id,
                    "binding": "frequency",
                }
            ),
        ), full_action

    @checked
    def solve(
        self,
        angular_frequencies: ArrayLike,
        material: ConductiveEMMaterial,
        /,
        *,
        primary_electric: ArrayLike | None = None,
        background_material: ConductiveEMMaterial | None = None,
    ) -> FrequencyDomainEMResult:
        frequencies = jnp.asarray(angular_frequencies)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("Frequency-domain EM angular frequencies must be a vector.")
        frequencies = eqx.error_if(
            frequencies,
            jnp.any(~jnp.isfinite(frequencies)) | jnp.any(frequencies <= 0),
            "Frequency-domain EM frequencies must be finite and positive.",
        )
        if material.conductivity_S_m.shape[0] != self.complex.cell_counts[3]:
            raise ValueError("Frequency-domain EM material does not match mesh cells.")
        primary = None if primary_electric is None else jnp.asarray(primary_electric)
        if (primary is None) != (background_material is None):
            raise ValueError(
                "Primary-secondary EM requires both primary field and background material."
            )
        if primary is not None and primary.shape != (
            frequencies.size,
            self.survey.electric_current_functionals.shape[0],
            self.complex.cell_counts[1],
        ):
            raise ValueError("Primary EM fields have wrong frequency/source/edge shape.")
        observation_rows, field_rows, residual_rows, power_rows, success_rows = (
            [],
            [],
            [],
            [],
            [],
        )
        for frequency_index, omega in enumerate(frequencies):
            operator, full_action = self._operator(omega, material)
            stiffness = _material_stiffness(
                self.complex, material.inverse_permeability_m_H, boundary="relative"
            )
            epsilon = self.complex.constitutive_operator(
                1, material.permittivity_F_m, boundary="relative"
            )
            sigma = self.complex.constitutive_operator(1, material.conductivity_S_m)

            relative_sigma = self.complex.constitutive_operator(
                1, material.conductivity_S_m, boundary="relative"
            )
            surrogate_sum = stiffness + omega**2 * epsilon + omega * relative_sigma
            surrogate_sum = eqx.tree_at(
                lambda value: value.properties,
                surrogate_sum,
                la.OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={"positive_definite": "construction"},
                ),
            )
            surrogate = la.assemble_sparse(surrogate_sum)
            native_inverse = _hx_preconditioner(self.complex, surrogate)

            def complex_inverse(values: Array) -> Array:
                return native_inverse.apply(values)

            inverse = la.OperatorPreconditioner(
                la.FunctionLinearOperator(
                    complex_inverse,
                    source=self.reduced_space,
                    target=self.reduced_space,
                    operator_id=self.plan_id + ":complex-surrogate-inverse",
                )
            )
            # The plan policy declares no preconditioner; its absent ``None``
            # slot is the single leaf replaced by the per-frequency surrogate.
            policy = eqx.tree_at(
                lambda value: value.preconditioning,
                self.policy,
                la.PreconditioningPolicy(inverse),
                is_leaf=lambda node: node is None,
            )
            background_action = None
            if background_material is not None:
                _, background_action = self._operator(omega, background_material)
            fields, residuals, powers, successes = [], [], [], []
            for source_index, current in enumerate(
                self.survey.electric_current_functionals
            ):
                rhs_full = 1j * omega * current
                primary_field = None
                if primary is not None and background_action is not None:
                    primary_field = primary[frequency_index, source_index]
                    background_residual = background_action(primary_field) - rhs_full
                    background_scale = jnp.maximum(
                        jnp.sqrt(jnp.real(jnp.vdot(rhs_full, rhs_full))), 1.0
                    )
                    primary_field = eqx.error_if(
                        primary_field,
                        jnp.sqrt(
                            jnp.real(jnp.vdot(background_residual, background_residual))
                        )
                        > 1e-7 * background_scale,
                        "Primary EM field does not solve its declared background problem.",
                    )
                    rhs_full = background_action(primary_field) - full_action(
                        primary_field
                    )
                rhs = rhs_full[self.free_edges]
                result = la.solve(la.LinearSystem(operator), rhs, policy=policy)
                secondary = (
                    jnp.zeros((self.complex.cell_counts[1],), dtype=result.value.dtype)
                    .at[self.free_edges]
                    .set(result.value)
                )
                field = secondary if primary_field is None else primary_field + secondary
                residual = operator.mv(result.value) - rhs
                conduction = _complex_action(sigma, field)
                power = 0.5 * jnp.real(jnp.vdot(field, conduction))
                fields.append(field)
                residuals.append(jnp.sqrt(jnp.real(jnp.vdot(residual, residual))))
                powers.append(power)
                successes.append(result.successful & jnp.isfinite(power) & (power >= 0))
            field = jnp.stack(fields)
            measurements = ein.contract(
                "me,me->m",
                self.survey.receiver_functionals,
                field[self.survey.source_indices],
            )
            observation_rows.append(measurements)
            field_rows.append(field)
            residual_rows.append(jnp.stack(residuals))
            power_rows.append(jnp.stack(powers))
            success_rows.append(jnp.all(jnp.stack(successes)))
        return FrequencyDomainEMResult(
            frequencies,
            jnp.stack(observation_rows),
            jnp.stack(field_rows),
            jnp.stack(residual_rows),
            jnp.stack(power_rows),
            jnp.all(jnp.stack(success_rows)),
        )


__all__ = [
    "ConductiveEMMaterial",
    "FrequencyDomainEMPlan",
    "FrequencyDomainEMResult",
    "FrequencyDomainEMSurvey",
]
