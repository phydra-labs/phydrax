#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.spectral import (
    IncompressibleSpectralDiagnostics,
    PeriodicLerayProjector,
    PreparedPseudospectralMethod,
    PseudospectralMethodPlan,
    TensorSpectralDiscretization,
)
from ..linalg import ArraySpace, DiagonalLinearOperator
from ._ir import PDEField
from ._spectral_compile import SpectralStateLayout


class IncompressibleFlowProblem(StrictModule):
    """Newtonian incompressible velocity dynamics with an optional modal forcing."""

    viscosity: Array
    forcing: Any
    spatial_dimension: int = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_dimension: int,
        viscosity: ArrayLike,
        /,
        *,
        forcing: Any = None,
        forcing_id: str | None = None,
        problem_id: str | None = None,
    ):
        dimension = int(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError(
                "Incompressible flow requires spatial dimension two or three."
            )
        viscosity_ = jnp.asarray(viscosity, dtype=float)
        if viscosity_.shape != () or not bool(
            jnp.isfinite(viscosity_) & (viscosity_ >= 0.0)
        ):
            raise ValueError("viscosity must be one finite nonnegative scalar.")
        if forcing is not None and not callable(forcing):
            raise TypeError("forcing must be callable or None.")
        source_id = "none" if forcing is None else str(forcing_id or "modal-forcing")
        if not source_id:
            raise ValueError("forcing_id must be non-empty.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "incompressible-flow-problem-v1",
                    "dimension": dimension,
                    "viscosity": float(viscosity_),
                    "forcing": source_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.viscosity = viscosity_
        self.forcing = forcing
        self.spatial_dimension = dimension
        self.forcing_id = source_id
        self.problem_id = identifier


class _PeriodicRotationalDrift(StrictModule):
    problem: IncompressibleFlowProblem
    discretization: TensorSpectralDiscretization
    method: PreparedPseudospectralMethod
    projector: PeriodicLerayProjector

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        discretization: TensorSpectralDiscretization,
        method: PreparedPseudospectralMethod,
        projector: PeriodicLerayProjector,
        /,
    ):
        self.problem = problem
        self.discretization = discretization
        self.method = method
        self.projector = projector

    def _rotational_product(self, state: Array, /) -> Array:
        dealiasing = self.method.dealiasing
        evaluation = dealiasing.evaluation
        padded = dealiasing.embed(self.projector.zero_forbidden_modes(state))
        velocity = evaluation.reconstruct(padded)
        derivatives = tuple(
            evaluation.modal_derivative(padded, axis=axis)
            for axis in range(self.problem.spatial_dimension)
        )
        if self.problem.spatial_dimension == 2:
            vorticity_modal = derivatives[0][..., 1] - derivatives[1][..., 0]
            vorticity = evaluation.reconstruct(vorticity_modal)
            product = jnp.stack(
                (-vorticity * velocity[..., 1], vorticity * velocity[..., 0]),
                axis=-1,
            )
        else:
            vorticity_modal = jnp.stack(
                (
                    derivatives[1][..., 2] - derivatives[2][..., 1],
                    derivatives[2][..., 0] - derivatives[0][..., 2],
                    derivatives[0][..., 1] - derivatives[1][..., 0],
                ),
                axis=-1,
            )
            vorticity = evaluation.reconstruct(vorticity_modal)
            product = jnp.cross(vorticity, velocity, axis=-1)
        return dealiasing.project(product)

    def unconstrained_rhs(self, time: Array, state: Array, args: Any, /) -> Array:
        value = self.projector.validate_state(state)
        result = -self._rotational_product(value)
        if self.problem.forcing is not None:
            forcing = self.projector.validate_state(
                self.problem.forcing(time, value, args), owner="Modal forcing"
            )
            result = result + forcing
        return self.projector.zero_forbidden_modes(result)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.projector.project(self.unconstrained_rhs(time, state, args))


class CompiledIncompressibleSpectralDynamics(StrictModule):
    """Velocity-only periodic incompressible dynamics in full complex coefficients."""

    problem: IncompressibleFlowProblem
    discretization: TensorSpectralDiscretization
    spatial_method: PreparedPseudospectralMethod
    projector: PeriodicLerayProjector
    layout: SpectralStateLayout
    nonlinear_drift: _PeriodicRotationalDrift
    semilinear_drift: Any
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        discretization: TensorSpectralDiscretization,
        spatial_method: PreparedPseudospectralMethod,
        projector: PeriodicLerayProjector,
        layout: SpectralStateLayout,
        nonlinear_drift: _PeriodicRotationalDrift,
        semilinear_drift: Any,
        /,
        *,
        compilation_id: str,
    ):
        residual_key = DiscretizationKey(
            "periodic_incompressible_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                    precision_evidence_id=discretization.precision_evidence_id,
                    resource_evidence_id=discretization.resource_evidence_id,
                ),
                DiscretizationRecord(
                    residual_key,
                    "compiled-periodic-incompressible-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.problem = problem
        self.discretization = discretization
        self.spatial_method = spatial_method
        self.projector = projector
        self.layout = layout
        self.nonlinear_drift = nonlinear_drift
        self.semilinear_drift = semilinear_drift
        self.discretization_bundle = bundle
        self.compilation_id = str(compilation_id)
        self.source_hash = problem.problem_id
        self.resolved_method = "periodic-incompressible-rotational-diagonal"

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.projector.state_shape

    def project_state(self, values: ArrayLike, /) -> Array:
        physical = jnp.asarray(values)
        expected = self.discretization.physical_shape + (self.problem.spatial_dimension,)
        if physical.shape != expected:
            raise ValueError(
                f"Physical velocity must have shape {expected}; got {physical.shape}."
            )
        return self.projector.project(self.discretization.project(physical))

    def reconstruct_state(self, state: ArrayLike, /) -> Array:
        return self.discretization.reconstruct(self.projector.project(state))

    def physical_state(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> Array:
        del time, args
        return self.reconstruct_state(state)

    def pressure_coefficients(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> Array:
        raw = self.nonlinear_drift.unconstrained_rhs(
            jnp.asarray(time), self.projector.validate_state(state), args
        )
        return self.projector.pressure_from_unconstrained_rhs(raw)

    def diagnostics(self, state: ArrayLike, /) -> IncompressibleSpectralDiagnostics:
        value = self.projector.validate_state(state)
        admissible = self.projector.project(value)
        physical = self.discretization.reconstruct(admissible)
        speed_squared = jnp.sum(jnp.real(physical * jnp.conj(physical)), axis=-1)
        kinetic_energy = 0.5 * jnp.sum(
            self.discretization.quadrature_weights * speed_squared
        )
        gradient_squared = jnp.zeros(
            self.discretization.physical_shape, dtype=physical.real.dtype
        )
        for axis in range(self.problem.spatial_dimension):
            derivative = self.discretization.reconstruct(
                self.discretization.modal_derivative(admissible, axis=axis)
            )
            gradient_squared = gradient_squared + jnp.sum(
                jnp.real(derivative * jnp.conj(derivative)), axis=-1
            )
        dissipation = self.problem.viscosity * jnp.sum(
            self.discretization.quadrature_weights * gradient_squared
        )
        forbidden = value - self.projector.zero_forbidden_modes(value)
        return IncompressibleSpectralDiagnostics(
            kinetic_energy=kinetic_energy,
            dissipation=dissipation,
            divergence_norm=self.projector.divergence_norm(value),
            imaginary_leakage=self.discretization.imaginary_leakage(admissible),
            forbidden_mode_norm=jnp.linalg.norm(forbidden.reshape((-1,))),
            projector_id=self.projector.projector_id,
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.semilinear_drift(time, state, args)


def compile_periodic_incompressible_flow(
    problem: IncompressibleFlowProblem,
    discretization: TensorSpectralDiscretization,
    method: PseudospectralMethodPlan,
    /,
) -> CompiledIncompressibleSpectralDynamics:
    """Compile rotational incompressible velocity dynamics on a periodic tensor grid."""
    if not isinstance(problem, IncompressibleFlowProblem):
        raise TypeError("problem must be an IncompressibleFlowProblem.")
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be a TensorSpectralDiscretization.")
    if len(discretization.axes) != problem.spatial_dimension:
        raise ValueError("Problem dimension must match the tensor spectral rank.")
    if not isinstance(method, PseudospectralMethodPlan):
        raise TypeError("method must be a PseudospectralMethodPlan.")
    prepared_method = method.prepare(
        discretization,
        required_polynomial_degree=2,
        nonlinear=True,
    )
    projector = PeriodicLerayProjector(discretization)
    field = PDEField(
        "velocity",
        representation="vector",
        components=problem.spatial_dimension,
        coordinates=discretization.plan.axis_names,
        component_names=discretization.plan.axis_names,
    )
    layout = SpectralStateLayout((field,), discretization)
    if layout.state_shape != projector.state_shape:
        raise RuntimeError("Incompressible state and spectral field layouts disagree.")
    nonlinear = _PeriodicRotationalDrift(
        problem,
        discretization,
        prepared_method,
        projector,
    )
    state_space = ArraySpace(
        projector.state_shape,
        dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
    )
    diagonal = jnp.broadcast_to(
        (
            -problem.viscosity.astype(projector.wavenumber_squared.dtype)
            * projector.wavenumber_squared
        )[..., None],
        projector.state_shape,
    ).reshape((-1,))
    linear = DiagonalLinearOperator(
        diagonal,
        space=state_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "periodic-incompressible-viscosity",
                "problem": problem.problem_id,
                "projector": projector.projector_id,
            }
        ),
    )
    from ..solver._semilinear_drift import SemilinearDrift

    semilinear = SemilinearDrift(
        linear,
        nonlinear,
        state_shape=projector.state_shape,
        operator_id=linear.operator_id,
    )
    compilation_id = canonical_fingerprint(
        {
            "kind": "periodic-incompressible-compiler-v1",
            "problem": problem.problem_id,
            "discretization": discretization.prepared_id,
            "method": prepared_method.prepared_id,
            "projector": projector.projector_id,
        }
    )
    return CompiledIncompressibleSpectralDynamics(
        problem,
        discretization,
        prepared_method,
        projector,
        layout,
        nonlinear,
        semilinear,
        compilation_id=compilation_id,
    )


__all__ = [
    "CompiledIncompressibleSpectralDynamics",
    "IncompressibleFlowProblem",
    "compile_periodic_incompressible_flow",
]
