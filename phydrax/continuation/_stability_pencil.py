#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, PyTree

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractVectorSpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
)
from ..linalg.eigen import (
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenSolvePolicy,
    SpectralStabilityStatus,
)
from ._core import (
    AbstractStabilityAnalyzer,
    ContinuationCurveProblem,
    StabilityAnalysisStatus,
    StabilityEvidence,
)
from ._geometry import ContinuationGeometry, ContinuationRepresentationPolicy


class ContinuationStabilityPencil(StrictModule, NonTrainableState):
    """Declared square execution pencil, including projected rectangular closures."""

    provider: Callable[[Any, Any, Any], tuple[Any, Any | None]]
    stability_space: AbstractVectorSpace | None
    lift: Callable[[Array], Any] | None
    project: Callable[[Any], Array] | None
    pencil_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider: Callable[[Any, Any, Any], tuple[Any, Any | None]],
        stability_space: AbstractVectorSpace | None = None,
        /,
        *,
        lift: Callable[[Array], Any] | None = None,
        project: Callable[[Any], Array] | None = None,
        pencil_id: str,
    ):
        if not callable(provider):
            raise TypeError("pencil provider must be callable.")
        if stability_space is not None and not isinstance(
            stability_space, AbstractVectorSpace
        ):
            raise TypeError("stability_space must be an AbstractVectorSpace or None.")
        if (lift is None) != (project is None):
            raise ValueError("Projected pencils require both lift and project.")
        if lift is not None and (not callable(lift) or not callable(project)):
            raise TypeError("lift and project must be callable.")
        if not isinstance(pencil_id, str) or not pencil_id:
            raise ValueError("pencil_id must be non-empty.")
        self.provider = provider
        self.stability_space = stability_space
        self.lift = lift
        self.project = project
        self.pencil_id = canonical_fingerprint(
            {
                "kind": "continuation-stability-pencil",
                "user_id": pencil_id,
                "projected": lift is not None,
                "space": None if stability_space is None else stability_space.space_id,
            }
        )

    def matrices(
        self, state: Any, coordinate: Any, args: Any = None, /
    ) -> tuple[Array, Array | None]:
        operator, mass = self.provider(state, coordinate, args)
        operator_ = jnp.asarray(operator)
        mass_ = None if mass is None else jnp.asarray(mass)
        if operator_.ndim != 2 or operator_.shape[0] != operator_.shape[1]:
            raise ValueError("A stability pencil provider must return a square operator.")
        if mass_ is not None and mass_.shape != operator_.shape:
            raise ValueError("Generalized pencil mass and operator matrices must match.")
        if (
            self.stability_space is not None
            and self.stability_space.size != operator_.shape[0]
        ):
            raise ValueError(
                "Pencil matrix size does not match the declared stability space."
            )
        return operator_, mass_

    @classmethod
    def projected_residual(
        cls,
        residual: Callable[[Any, Any, Any], Any],
        lift: Callable[[Array], Any],
        project: Callable[[Any], Array],
        state_template: Any,
        stability_template: ArrayLike,
        /,
        *,
        mass_provider: Callable[[Any, Any, Any], Array] | None = None,
        pencil_id: str,
    ) -> "ContinuationStabilityPencil":
        """Build only an explicitly lifted/projected square residual Jacobian."""

        template = jnp.asarray(stability_template)
        if template.ndim != 1:
            raise ValueError("stability_template must be a rank-one coordinate vector.")

        def provider(state: Any, coordinate: Any, args: Any):
            basis = jnp.eye(template.size, dtype=template.dtype)
            columns = jax.vmap(
                lambda direction: project(
                    jax.jvp(
                        lambda value: residual(value, coordinate, args),
                        (state,),
                        (lift(direction),),
                    )[1]
                )
            )(basis)
            matrix = jnp.swapaxes(columns, 0, 1)
            mass = (
                None if mass_provider is None else mass_provider(state, coordinate, args)
            )
            return matrix, mass

        del state_template
        return cls(provider, lift=lift, project=project, pencil_id=pencil_id)

    @classmethod
    def dae(
        cls,
        residual: Callable[[Array, Array, Array, Any], Array],
        state_rate: ArrayLike,
        /,
        *,
        pencil_id: str,
    ) -> "ContinuationStabilityPencil":
        """Declare the regular DAE convention A=-F_y and B=F_ydot."""

        rate = jnp.asarray(state_rate)

        def provider(state: Array, coordinate: Any, args: Any):
            time = jnp.asarray(coordinate)
            a = -jax.jacfwd(lambda value: residual(time, value, rate, args))(state)
            b = jax.jacfwd(lambda value: residual(time, state, value, args))(rate)
            return a, b

        return cls(provider, pencil_id=pencil_id)


class GeneralizedPencilStabilityAnalyzer(AbstractStabilityAnalyzer):
    """Native general eigensolve routed from one declared square pencil."""

    pencil: ContinuationStabilityPencil
    policy: GeneralEigenSolvePolicy
    analyzer_id: str = eqx.field(static=True)
    zero_tolerance: float = eqx.field(static=True)
    pair_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        pencil: ContinuationStabilityPencil,
        policy: GeneralEigenSolvePolicy | None = None,
        /,
        *,
        zero_tolerance: float = 1.0e-8,
        pair_tolerance: float = 1.0e-7,
    ):
        if not isinstance(pencil, ContinuationStabilityPencil):
            raise TypeError("pencil must be a ContinuationStabilityPencil.")
        policy_ = GeneralEigenSolvePolicy() if policy is None else policy
        if not isinstance(policy_, GeneralEigenSolvePolicy):
            raise TypeError("policy must be a GeneralEigenSolvePolicy or None.")
        zero = float(zero_tolerance)
        pair = float(pair_tolerance)
        if not np.isfinite(zero) or zero <= 0 or not np.isfinite(pair) or pair <= 0:
            raise ValueError("stability tolerances must be positive and finite.")
        self.pencil = pencil
        self.policy = policy_
        self.zero_tolerance = zero
        self.pair_tolerance = pair
        self.analyzer_id = canonical_fingerprint(
            {
                "kind": "generalized-pencil-stability",
                "pencil": pencil.pencil_id,
                "zero_tolerance": zero,
                "pair_tolerance": pair,
            }
        )

    def analyze(
        self,
        problem: ContinuationCurveProblem,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
        *,
        geometry: ContinuationGeometry | None = None,
    ) -> StabilityEvidence:
        del problem, geometry
        matrix, mass = self.pencil.matrices(state, coordinate, args)
        operator = DenseLinearOperator(matrix, operator_id=f"{self.pencil.pencil_id}:A")
        mass_operator = (
            None
            if mass is None
            else DenseLinearOperator(
                mass,
                source=operator.source,
                target=operator.target,
                operator_id=f"{self.pencil.pencil_id}:B",
            )
        )
        result = general_eigensolve(
            GeneralEigenproblem(
                operator, mass_operator, problem_id=self.pencil.pencil_id
            ),
            policy=self.policy,
        )
        values = result.eigenvalues
        mask = result.finite_mask & result.diagnostics.mode_mask
        real_parts = jnp.where(mask, jnp.real(values), -jnp.inf)
        leading_index = jnp.argmax(real_parts)
        leading = values[leading_index]
        complex_mask = mask & (jnp.abs(jnp.imag(values)) > self.zero_tolerance)
        complex_real = jnp.where(complex_mask, jnp.real(values), -jnp.inf)
        complex_index = jnp.argmax(complex_real)
        leading_complex = values[complex_index]
        unstable_count = jnp.sum(mask & (jnp.real(values) > self.zero_tolerance))
        marginal_count = jnp.sum(
            mask & (jnp.abs(jnp.real(values)) <= self.zero_tolerance)
        )
        near_zero_count = jnp.sum(mask & (jnp.abs(values) <= self.zero_tolerance))
        pair_matrix = jnp.abs(values[:, None] - jnp.conj(values[None, :]))
        paired = complex_mask & jnp.any(
            pair_matrix <= self.pair_tolerance * (1 + jnp.abs(values[:, None])), axis=1
        )
        pair_count = jnp.sum(paired) // 2
        unpaired_count = jnp.sum(complex_mask & (~paired))
        finite = jnp.all(jnp.where(mask, jnp.isfinite(values), True))
        successful = result.successful & finite & jnp.any(mask)
        stability = jnp.where(
            unstable_count > 0,
            int(SpectralStabilityStatus.UNSTABLE),
            jnp.where(
                marginal_count > 0,
                int(SpectralStabilityStatus.MARGINAL),
                int(SpectralStabilityStatus.STABLE),
            ),
        )
        return StabilityEvidence(
            eigenvalues=values,
            mode_mask=mask,
            leading_eigenvalue=leading,
            leading_real_part=jnp.real(leading),
            leading_complex_eigenvalue=leading_complex,
            leading_complex_real_part=jnp.real(leading_complex),
            unstable_count=unstable_count,
            marginal_count=marginal_count,
            near_zero_count=near_zero_count,
            conjugate_pair_count=pair_count,
            unpaired_complex_count=unpaired_count,
            stability=jnp.where(
                successful, stability, int(SpectralStabilityStatus.UNKNOWN)
            ),
            status=jnp.where(
                successful,
                int(StabilityAnalysisStatus.SUCCESS),
                int(StabilityAnalysisStatus.SOURCE_FAILURE),
            ),
            source_status=result.status,
            analyzer_id=self.analyzer_id,
            full_spectrum=self.policy.selection.kind == "all",
            zero_tolerance=self.zero_tolerance,
            pair_tolerance=self.pair_tolerance,
        )


class _HopfCurveProblem(ContinuationCurveProblem):
    physical_residual: Callable[[Array, Any, Any], Array]
    pencil: ContinuationStabilityPencil
    parameter_plane: Callable[[Array, Array, Any], Any]
    reference_mode: Array
    coordinate_lower: float = eqx.field(static=True)
    coordinate_upper: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def residual(
        self,
        state: tuple[Array, Array, Array, Array, Array],
        coordinate: Any,
        args: Any = None,
        /,
    ):
        physical_state, parameter_one, mode_real, mode_imag, frequency = state
        parameters = self.parameter_plane(parameter_one, coordinate, args)
        physical = jnp.asarray(self.physical_residual(physical_state, parameters, args))
        matrix, mass = self.pencil.matrices(physical_state, parameters, args)
        mass_ = jnp.eye(matrix.shape[0], dtype=matrix.dtype) if mass is None else mass
        real_mode = ein.contract("ij,j->i", matrix, mode_real) + frequency * ein.contract(
            "ij,j->i", mass_, mode_imag
        )
        imag_mode = ein.contract("ij,j->i", matrix, mode_imag) - frequency * ein.contract(
            "ij,j->i", mass_, mode_real
        )
        normalization = (
            jnp.real(jnp.vdot(mode_real, mode_real) + jnp.vdot(mode_imag, mode_imag)) - 1
        )
        phase = jnp.real(jnp.vdot(self.reference_mode, mode_imag))
        return physical, real_mode, imag_mode, normalization, phase

    def parameters(self, coordinate: Any, args: Any = None, /):
        del args
        return jnp.asarray(coordinate)

    def declared_spaces(self, /):
        return None, None

    def representation_policy(self, /):
        return ContinuationRepresentationPolicy()

    def state_jacobian_action(self, state, coordinate, tangent, args: Any = None, /):
        return jax.jvp(
            lambda value: self.residual(value, coordinate, args), (state,), (tangent,)
        )[1]

    def coordinate_derivative(self, state, coordinate, args: Any = None, /):
        coordinate_ = jnp.asarray(coordinate)
        return jax.jvp(
            lambda value: self.residual(state, value, args),
            (coordinate_,),
            (jnp.ones_like(coordinate_),),
        )[1]


class HopfContinuationAdapter(StrictModule, NonTrainableState):
    """Lower a two-parameter real-block Hopf locus to one continuation curve."""

    problem: _HopfCurveProblem
    omega_min: float = eqx.field(static=True)
    spectral_isolation_tolerance: float = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_residual: Callable[[Array, Any, Any], Array],
        pencil: ContinuationStabilityPencil,
        parameter_plane: Callable[[Array, Array, Any], Any],
        reference_mode: ArrayLike,
        /,
        *,
        coordinate_lower: float,
        coordinate_upper: float,
        omega_min: float = 1.0e-8,
        spectral_isolation_tolerance: float = 1.0e-6,
        problem_id: str,
    ):
        if not callable(physical_residual) or not callable(parameter_plane):
            raise TypeError(
                "Hopf physical residual and parameter plane must be callable."
            )
        if not isinstance(pencil, ContinuationStabilityPencil):
            raise TypeError("pencil must be a ContinuationStabilityPencil.")
        lower, upper = float(coordinate_lower), float(coordinate_upper)
        omega, isolation = float(omega_min), float(spectral_isolation_tolerance)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("Hopf continuation coordinate bounds are invalid.")
        if (
            omega <= 0
            or isolation <= 0
            or not np.isfinite(omega)
            or not np.isfinite(isolation)
        ):
            raise ValueError(
                "Hopf frequency/isolation tolerances must be positive and finite."
            )
        reference = jnp.asarray(reference_mode)
        if reference.ndim != 1 or not reference.size:
            raise ValueError("reference_mode must be a nonempty rank-one vector.")
        curve = _HopfCurveProblem(
            physical_residual=physical_residual,
            pencil=pencil,
            parameter_plane=parameter_plane,
            reference_mode=reference,
            coordinate_lower=lower,
            coordinate_upper=upper,
            problem_id=problem_id,
        )
        self.problem = curve
        self.omega_min = omega
        self.spectral_isolation_tolerance = isolation
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "hopf-continuation-adapter",
                "problem": problem_id,
                "pencil": pencil.pencil_id,
                "omega_min": omega,
                "isolation": isolation,
            }
        )


class HopfPointEvidence(StrictModule, NonTrainableState):
    frequency: Array
    mode_real: Array
    mode_imag: Array
    normalization_residual: Array
    phase_residual: Array
    conjugate_pair_residual: Array
    spectral_isolation: Array
    transversality: Array
    augmented_jacobian_full_rank: Array
    valid: Array
    local_only: bool = eqx.field(static=True)
    normal_form_not_claimed: bool = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)


def hopf_point_evidence(
    adapter: HopfContinuationAdapter,
    state: tuple[Array, Array, Array, Array, Array],
    coordinate: ArrayLike,
    /,
    *,
    args: Any = None,
) -> HopfPointEvidence:
    """Audit simplicity, phase, frequency, isolation, and augmented rank locally."""

    if not isinstance(adapter, HopfContinuationAdapter):
        raise TypeError("adapter must be a HopfContinuationAdapter.")
    residual = adapter.problem.residual(state, coordinate, args)
    _, _, mode_real, mode_imag, frequency = state
    normalization = jnp.abs(residual[-2])
    phase = jnp.abs(residual[-1])
    matrix, mass = adapter.problem.pencil.matrices(
        state[0], adapter.problem.parameter_plane(state[1], coordinate, args), args
    )
    mass_ = jnp.eye(matrix.shape[0], dtype=matrix.dtype) if mass is None else mass
    complex_mode = mode_real + 1j * mode_imag
    pair_residual = jnp.sqrt(
        jnp.mean(
            jnp.square(
                jnp.abs(
                    ein.contract("ij,j->i", matrix, complex_mode)
                    - 1j * frequency * ein.contract("ij,j->i", mass_, complex_mode)
                )
            )
        )
    )
    eigen_result = general_eigensolve(
        GeneralEigenproblem(
            DenseLinearOperator(
                matrix,
                operator_id=f"{adapter.problem.pencil.pencil_id}:hopf:A",
            ),
            DenseLinearOperator(
                mass_,
                operator_id=f"{adapter.problem.pencil.pencil_id}:hopf:B",
            ),
            problem_id=f"{adapter.problem.pencil.pencil_id}:hopf",
        )
    )
    distances = jnp.where(
        eigen_result.finite_mask,
        jnp.abs(eigen_result.eigenvalues - 1j * frequency),
        jnp.inf,
    )
    ordered = jnp.sort(distances)
    isolation = ordered[1] if ordered.size > 1 else jnp.asarray(jnp.inf)
    flattened, unravel = ravel_pytree(state)
    jacobian = jax.jacfwd(
        lambda value: ravel_pytree(
            adapter.problem.residual(unravel(value), coordinate, args)
        )[0]
    )(flattened)
    singular_values = factorize(
        DenseLinearOperator(
            jacobian,
            operator_id=f"{adapter.adapter_id}:augmented-jacobian",
        ),
        FactorizationPolicy("svd"),
    ).singular_values()
    full_rank = jnp.min(singular_values) > adapter.spectral_isolation_tolerance
    coordinate_derivative = adapter.problem.coordinate_derivative(state, coordinate, args)
    transversality = jnp.sqrt(
        sum(
            jnp.sum(jnp.square(jnp.abs(value)))
            for value in jax.tree.leaves(coordinate_derivative)
        )
    )
    valid = (
        (frequency > adapter.omega_min)
        & (normalization <= adapter.spectral_isolation_tolerance)
        & (phase <= adapter.spectral_isolation_tolerance)
        & (pair_residual <= adapter.spectral_isolation_tolerance)
        & (isolation > adapter.spectral_isolation_tolerance)
        & (transversality > adapter.spectral_isolation_tolerance)
        & full_rank
    )
    return HopfPointEvidence(
        frequency,
        mode_real,
        mode_imag,
        normalization,
        phase,
        pair_residual,
        isolation,
        transversality,
        full_rank,
        valid,
        True,
        True,
        adapter.adapter_id,
    )


__all__ = [
    "ContinuationStabilityPencil",
    "GeneralizedPencilStabilityAnalyzer",
    "HopfContinuationAdapter",
    "HopfPointEvidence",
    "hopf_point_evidence",
]
