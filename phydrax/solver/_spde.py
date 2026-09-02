#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._precision import (
    precision_dtype_name,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
)
from .._strict import StrictModule
from ..discretization import (
    AbstractStrongFormDiscretization,
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.spectral import TensorSpectralDiscretization
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DiagonalLinearOperator,
    DiagonalPairing,
    FunctionLinearOperator,
    OperatorProperties,
    TransformDiagonalRepresentation,
)
from ..stochastic import (
    LevyAreaKind,
    SPDESolutionSpec,
    WienerRealization,
)
from ..stochastic._spatial_noise import SpatialNoiseBasis
from ._differential import (
    DifferentialInterpretation,
    DifferentialProblem,
    NoiseStructure,
    WienerTerm,
)
from ._semilinear_drift import SemilinearDrift


SpatialDiscretization = AbstractStrongFormDiscretization | TensorSpectralDiscretization


def _spatial_laplacian(
    discretization: SpatialDiscretization,
    state: Array,
    /,
) -> Array:
    return (
        discretization.modal_laplacian(state)
        if isinstance(discretization, TensorSpectralDiscretization)
        else discretization.laplacian(state)
    )


def _spatial_weights(
    discretization: SpatialDiscretization,
    /,
) -> Array:
    return (
        jnp.ones(
            discretization.modal_shape,
            dtype=jnp.dtype(discretization.plan.precision.physical_dtype),
        )
        if isinstance(discretization, TensorSpectralDiscretization)
        else jnp.asarray(discretization.quadrature_weights)
    )


def _spatial_id(discretization: SpatialDiscretization, /) -> str:
    return (
        discretization.prepared_id
        if isinstance(discretization, TensorSpectralDiscretization)
        else discretization.discretization_id
    )


_ZERO_SEMILINEAR_NONLINEAR_ID = canonical_fingerprint(
    {"kind": "semilinear-zero-nonlinear-drift-v1"}
)
_NO_REACTION_ID = canonical_fingerprint({"kind": "reaction-diffusion-no-reaction-v1"})


def _resolve_optional_callable_id(
    value: Callable[..., Any] | None,
    identifier: str | None,
    /,
    *,
    value_name: str,
    identifier_name: str,
    zero_identifier: str,
) -> str:
    if value is None:
        if identifier is not None:
            raise ValueError(f"{identifier_name} must be None when {value_name} is None.")
        return zero_identifier
    if not callable(value):
        raise TypeError(f"{value_name} must be callable or None.")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(
            f"{identifier_name} is required and must be non-empty when "
            f"{value_name} is callable."
        )
    return identifier


class _ValidatedVectorField(StrictModule):
    field: Callable[[Array, Array, Any], ArrayLike]
    output_shape: tuple[int, ...] = eqx.field(static=True)
    name: str = eqx.field(static=True)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        value = jnp.asarray(self.field(time, state, args))
        if tuple(value.shape) != self.output_shape:
            raise ValueError(
                f"{self.name} must return shape {self.output_shape}; got {value.shape}."
            )
        return value


class _ConstantBasisDiffusion(StrictModule):
    basis: SpatialNoiseBasis

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        del time, state, args
        return self.basis.diffusion


class _BasisAmplitudeDiffusion(StrictModule):
    amplitude: Callable[[Array, Array, Any], ArrayLike]
    basis: SpatialNoiseBasis

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        amplitude = jnp.asarray(self.amplitude(time, state, args))
        if amplitude.shape == ():
            return amplitude * self.basis.diffusion
        if tuple(amplitude.shape) == self.basis.state_shape:
            return amplitude[..., None] * self.basis.diffusion
        full_shape = self.basis.state_shape + self.basis.noise_shape
        if tuple(amplitude.shape) == full_shape:
            return amplitude
        raise ValueError(
            "Noise amplitude must be scalar, have exact state shape "
            f"{self.basis.state_shape}, or return the full diffusion shape "
            f"{full_shape}; got {amplitude.shape}."
        )


class _ReactionDiffusionDrift(StrictModule):
    discretization: SpatialDiscretization
    kappa: Any
    reaction: Callable[[Array, Array, Any], ArrayLike] | None
    reaction_id: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        state_array = jnp.asarray(state)
        coefficient = (
            self.kappa(time, state_array, args) if callable(self.kappa) else self.kappa
        )
        coefficient_array = jnp.asarray(coefficient)
        if coefficient_array.shape not in ((), self.state_shape):
            raise ValueError(
                "kappa must be scalar or have exact state shape "
                f"{self.state_shape}; got {coefficient_array.shape}."
            )
        reaction = (
            jnp.zeros_like(state_array)
            if self.reaction is None
            else jnp.asarray(self.reaction(time, state_array, args))
        )
        if tuple(reaction.shape) != self.state_shape:
            raise ValueError(
                f"reaction must return shape {self.state_shape}; got {reaction.shape}."
            )
        return (
            coefficient_array * _spatial_laplacian(self.discretization, state_array)
            + reaction
        )


class _ScaledLaplacianOperator(StrictModule):
    discretization: SpatialDiscretization
    coefficient: Array

    def __call__(self, state: Array) -> Array:
        return self.coefficient * _spatial_laplacian(self.discretization, state)


def _compatible_noise_eigenvalues(
    discretization: SpatialDiscretization,
    noise_basis: SpatialNoiseBasis | None,
    coefficient: Array,
    /,
) -> Array | None:
    if (
        noise_basis is None
        or coefficient.shape != ()
        or noise_basis.field_space_id != discretization.field_spaces[0].field_space_id
    ):
        return None
    if isinstance(discretization, TensorSpectralDiscretization):
        values, point_value_modes = discretization.eigenpairs(rank=noise_basis.rank)
        modes = discretization.project(point_value_modes)
        comparison_dtype = jnp.result_type(modes.dtype, noise_basis.modes.dtype)
        modes = modes.astype(comparison_dtype)
        basis_modes = noise_basis.modes.astype(comparison_dtype)
        if modes.shape != basis_modes.shape or not bool(
            jnp.allclose(modes, basis_modes, rtol=1e-6, atol=1e-7)
        ):
            return None
        return -coefficient * values
    laplacian_eigenvalues, modes = discretization.eigenpairs(rank=noise_basis.rank)
    if modes.shape != noise_basis.modes.shape or not bool(
        jnp.allclose(modes, noise_basis.modes, rtol=1e-6, atol=1e-7)
    ):
        return None
    return -coefficient * laplacian_eigenvalues


def _spectral_linear_representation(
    operator: AbstractLinearOperator,
    discretization: SpatialDiscretization,
    coefficient: Array,
    state_shape: tuple[int, ...],
    /,
) -> TransformDiagonalRepresentation | None:
    from ..discretization._tensor import EigenbasisDiscretization

    if not isinstance(discretization, EigenbasisDiscretization):
        return None
    if coefficient.shape != () or state_shape != discretization.state_shape:
        return None
    return TransformDiagonalRepresentation(
        operator,
        -coefficient * discretization.plan.eigenvalues,
        discretization.plan.analysis,
        discretization.plan.synthesis,
        representation_id=discretization.discretization_id,
    )


class SemidiscreteSPDE(StrictModule):
    """Finite-dimensional method-of-lines problem plus spatial/noise provenance."""

    problem: DifferentialProblem
    spatial_discretization: SpatialDiscretization
    noise_basis: SpatialNoiseBasis | None
    semilinear_drift: SemilinearDrift | None
    discretization_bundle: DiscretizationBundle
    solution_spec: SPDESolutionSpec = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)
    precision_evidence_id: str = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: DifferentialProblem,
        spatial_discretization: SpatialDiscretization,
        noise_basis: SpatialNoiseBasis | None,
        semilinear_drift: SemilinearDrift | None = None,
        solution_spec: SPDESolutionSpec | None = None,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        basis_id: str | None,
    ):
        if not isinstance(problem, DifferentialProblem):
            raise TypeError("problem must be a DifferentialProblem.")
        if not isinstance(
            spatial_discretization,
            (AbstractStrongFormDiscretization, TensorSpectralDiscretization),
        ):
            raise TypeError(
                "spatial_discretization must provide a strong-form or tensor-spectral "
                "state space."
            )
        if noise_basis is not None and not isinstance(noise_basis, SpatialNoiseBasis):
            raise TypeError("noise_basis must be a SpatialNoiseBasis or None.")
        state = tuple(int(size) for size in state_shape)
        noise = tuple(int(size) for size in noise_shape)
        if state != tuple(problem.initial_state.shape):
            raise ValueError("state_shape must match the differential problem state.")
        if noise != problem.noise_shape:
            raise ValueError("noise_shape must match the differential problem noise.")
        if basis_id != problem.noise_id:
            raise ValueError("basis_id must match the differential problem noise_id.")
        if noise_basis is not None and noise_basis.basis_id != basis_id:
            raise ValueError("noise_basis identity must match basis_id.")
        if semilinear_drift is not None:
            if not isinstance(semilinear_drift, SemilinearDrift):
                raise TypeError("semilinear_drift must be a SemilinearDrift or None.")
            if semilinear_drift.state_shape != state:
                raise ValueError("semilinear_drift state_shape must match the problem.")
            declared = semilinear_drift(problem.t0, problem.initial_state, problem.args)
            actual = problem.drift(problem.t0, problem.initial_state, problem.args)
            if not bool(jnp.allclose(declared, actual, rtol=1e-6, atol=1e-7)):
                raise ValueError(
                    "semilinear_drift must reproduce the differential problem drift."
                )
        if solution_spec is None:
            resolved_solution = SPDESolutionSpec(
                "strong",
                noise_regularization=("finite_rank" if problem.stochastic else "none"),
                cutoff_id=problem.noise_id,
            )
        else:
            if not isinstance(solution_spec, SPDESolutionSpec):
                raise TypeError("solution_spec must be an SPDESolutionSpec or None.")
            resolved_solution = solution_spec
        if problem.stochastic and resolved_solution.noise_regularization == "none":
            raise ValueError(
                "A stochastic semidiscrete problem cannot declare noise_regularization='none'."
            )
        if not problem.stochastic and resolved_solution.noise_regularization != "none":
            raise ValueError(
                "A deterministic semidiscrete problem must declare "
                "noise_regularization='none'."
            )
        if resolved_solution.rough_forcing and resolved_solution.cutoff_id is None:
            raise ValueError(
                "A finite-dimensional approximation of rough forcing requires cutoff_id."
            )
        self.problem = problem
        self.spatial_discretization = spatial_discretization
        self.noise_basis = noise_basis
        self.state_shape = state
        self.semilinear_drift = semilinear_drift
        self.solution_spec = resolved_solution
        self.noise_shape = noise
        self.discretization_id = _spatial_id(spatial_discretization)
        self.basis_id = basis_id
        spatial_precision = spatial_discretization.precision_evidence
        spatial_precision_id = (
            None if spatial_precision is None else spatial_precision.evidence_id
        )
        noise_precision = None if noise_basis is None else noise_basis.precision_evidence
        noise_precision_id = (
            None if noise_precision is None else noise_precision.evidence_id
        )
        state_dtype = precision_dtype_name(jnp.asarray(problem.initial_state).dtype)
        request = PrecisionRequest(
            "semidiscrete-spde",
            {
                "storage": state_dtype,
                "compute": state_dtype,
                "output": state_dtype,
            },
        )
        resolution = PrecisionResolution(
            request,
            "phydrax-spde",
            dict(request.requested),
        )
        children = {}
        if spatial_precision is not None:
            children["spatial"] = spatial_precision
        if noise_precision is not None:
            children["noise"] = noise_precision
        precision_evidence = PrecisionEvidenceEnvelope(
            resolution,
            dict(resolution.effective),
            children=children,
        )
        self.precision_evidence = precision_evidence
        self.precision_evidence_id = precision_evidence.evidence_id
        records = [
            DiscretizationRecord(
                spatial_discretization.key,
                type(spatial_discretization).__name__,
                spatial_discretization.prepared_id,
                numeric_version=spatial_discretization.numeric_version,
                precision_evidence_id=spatial_precision_id,
                resource_evidence_id=spatial_discretization.resource_evidence_id,
            )
        ]
        dependencies = [spatial_discretization.key.key_id]
        if noise_basis is not None:
            noise_key = DiscretizationKey(
                "spatial_noise",
                DiscretizationRole.DRIVER,
                domain_labels=spatial_discretization.key.domain_labels,
            )
            records.append(
                DiscretizationRecord(
                    noise_key,
                    "spatial-noise-basis",
                    noise_basis.basis_id,
                    dependency_key_ids=(spatial_discretization.key.key_id,),
                    precision_evidence_id=noise_precision_id,
                )
            )
            dependencies.append(noise_key.key_id)
        form_key = DiscretizationKey(
            "spde_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=spatial_discretization.key.domain_labels,
        )
        records.append(
            DiscretizationRecord(
                form_key,
                "semidiscrete-spde",
                canonical_fingerprint(
                    {
                        "kind": "semidiscrete-spde-form-v2",
                        "problem": problem.problem_id,
                        "spatial_discretization": spatial_discretization.prepared_id,
                        "noise_basis": basis_id,
                    }
                ),
                dependency_key_ids=tuple(dependencies),
                precision_evidence_id=self.precision_evidence_id,
            )
        )
        self.discretization_bundle = DiscretizationBundle(records)

    def wiener_realization(
        self,
        key: Key[Array, ""],
        /,
        *,
        support: tuple[float, float] | None = None,
        sample_shape: Sequence[int] = (),
        tolerance: float = 1e-3,
        levy_area: LevyAreaKind = "brownian",
        label: str | None = None,
        coupling_id: str | None = None,
    ) -> WienerRealization:
        """Create a global realization synchronized with the retained noise basis."""
        if not self.problem.stochastic:
            raise ValueError(
                "Deterministic semidiscrete problems have no Wiener realization."
            )
        resolved_support = (
            (float(self.problem.t0), float(self.problem.t1))
            if support is None
            else support
        )
        return WienerRealization(
            key,
            self.problem.noise_shape,
            support=resolved_support,
            sample_shape=sample_shape,
            tolerance=tolerance,
            levy_area=levy_area,
            noise_id=self.problem.noise_id,
            label=label,
            coupling_id=coupling_id,
        )


def semidiscretize_spde(
    drift: Callable[[Array, Array, Any], ArrayLike],
    initial_state: ArrayLike,
    spatial_discretization: SpatialDiscretization,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    args: Any = None,
    semilinear_drift: SemilinearDrift | None = None,
    diffusion: Callable[[Array, Array, Any], ArrayLike] | None = None,
    noise_basis: SpatialNoiseBasis | None = None,
    noise_shape: Sequence[int] | None = None,
    basis_id: str | None = None,
    noise_structure: NoiseStructure | None = None,
    interpretation: DifferentialInterpretation = "ito",
    solution_spec: SPDESolutionSpec | None = None,
) -> SemidiscreteSPDE:
    r"""Compose a validated finite-rank method-of-lines SPDE.

    With ``noise_basis``, omitting ``diffusion`` gives additive noise. Supplying it
    gives a scalar/pointwise amplitude :math:`G_h` (or a fully composed diffusion
    factor) multiplying the retained basis :math:`B`. Without a basis, stochastic
    problems must supply both ``diffusion`` and ``noise_shape``.

    A ``TensorSpectralDiscretization`` supplies a modal primary state, so its drift,
    diffusion, and initial state use modal representation.
    """
    if not callable(drift):
        raise TypeError("drift must be callable.")
    if not isinstance(
        spatial_discretization,
        (AbstractStrongFormDiscretization, TensorSpectralDiscretization),
    ):
        raise TypeError(
            "spatial_discretization must provide a strong-form or tensor-spectral "
            "state space."
        )
    state = jnp.asarray(initial_state)
    spatial_shape = spatial_discretization.state_shape
    spatial_rank = len(spatial_shape)
    if state.ndim < spatial_rank or tuple(state.shape[:spatial_rank]) != spatial_shape:
        raise ValueError(
            "initial_state must begin with spatial shape "
            f"{spatial_shape}; got {state.shape}."
        )
    state_shape = tuple(int(size) for size in state.shape)
    if semilinear_drift is not None:
        if not isinstance(semilinear_drift, SemilinearDrift):
            raise TypeError("semilinear_drift must be a SemilinearDrift or None.")
        if semilinear_drift.state_shape != state_shape:
            raise ValueError("semilinear_drift state_shape must match initial_state.")
    if noise_basis is not None:
        if not isinstance(noise_basis, SpatialNoiseBasis):
            raise TypeError("noise_basis must be a SpatialNoiseBasis or None.")
        if noise_basis.state_shape != state_shape:
            raise ValueError(
                "noise basis state shape must match initial_state exactly; "
                f"got {noise_basis.state_shape} and {state_shape}."
            )
        expected_field_space_id = spatial_discretization.field_spaces[0].field_space_id
        if noise_basis.field_space_id != expected_field_space_id:
            raise ValueError(
                "noise basis field_space_id must match the spatial state field space."
            )
        resolved_noise_shape = noise_basis.noise_shape
        if (
            noise_shape is not None
            and tuple(int(v) for v in noise_shape) != resolved_noise_shape
        ):
            raise ValueError("noise_shape must agree with noise_basis.rank.")
        if basis_id is not None and str(basis_id) != noise_basis.basis_id:
            raise ValueError("basis_id must agree with noise_basis.basis_id.")
        resolved_basis_id = noise_basis.basis_id
        effective_diffusion: Callable[[Array, Array, Any], ArrayLike] = (
            _ConstantBasisDiffusion(noise_basis)
            if diffusion is None
            else _BasisAmplitudeDiffusion(diffusion, noise_basis)
        )
        resolved_structure: NoiseStructure = (
            ("additive" if diffusion is None else "general")
            if noise_structure is None
            else noise_structure
        )
    elif diffusion is not None:
        if noise_shape is None:
            raise ValueError(
                "noise_shape is required for stochastic problems without a noise basis."
            )
        resolved_noise_shape = tuple(int(size) for size in noise_shape)
        if not resolved_noise_shape or any(size <= 0 for size in resolved_noise_shape):
            raise ValueError("noise_shape must contain positive dimensions.")
        resolved_basis_id = None if basis_id is None else str(basis_id)
        if resolved_basis_id == "":
            raise ValueError("basis_id must be non-empty or None.")
        effective_diffusion = diffusion
        resolved_structure = "general" if noise_structure is None else noise_structure
    else:
        if noise_shape is not None or basis_id is not None or noise_structure is not None:
            raise ValueError(
                "noise_shape, basis_id, and noise_structure are only valid for "
                "stochastic problems."
            )
        resolved_noise_shape = ()
        resolved_basis_id = None
        resolved_structure = "general"
        effective_diffusion = None  # type: ignore[assignment]

    validated_drift = _ValidatedVectorField(drift, state_shape, "drift")
    validated_diffusion = (
        None
        if effective_diffusion is None
        else _ValidatedVectorField(
            effective_diffusion,
            state_shape + resolved_noise_shape,
            "diffusion",
        )
    )
    # Fail before entering Diffrax, while callback shapes are still easy to diagnose.
    validated_drift(jnp.asarray(t0, dtype=float), state, args)
    if validated_diffusion is not None:
        validated_diffusion(jnp.asarray(t0, dtype=float), state, args)
    wiener_terms = (
        ()
        if validated_diffusion is None
        else (
            WienerTerm(
                "forcing",
                validated_diffusion,
                resolved_noise_shape,
                structure=resolved_structure,
                basis_id=resolved_basis_id,
            ),
        )
    )
    if semilinear_drift is not None:
        dynamics_identity = {"semilinear_drift": semilinear_drift.drift_id}
    elif isinstance(drift, _ReactionDiffusionDrift):
        dynamics_identity = {"reaction": drift.reaction_id}
    else:
        dynamics_identity = None
    problem_id = (
        None
        if dynamics_identity is None
        else (
            "semidiscrete-spde:"
            + canonical_fingerprint(
                {
                    "kind": "semidiscrete-spde-v1",
                    "dynamics": dynamics_identity,
                    "spatial_discretization": _spatial_id(spatial_discretization),
                    "state_shape": list(state_shape),
                    "state_dtype": str(state.dtype),
                    "noise_shape": list(resolved_noise_shape),
                    "noise_basis": resolved_basis_id,
                    "noise_structure": (
                        None if validated_diffusion is None else resolved_structure
                    ),
                    "interpretation": interpretation,
                }
            )
        )
    )
    problem = DifferentialProblem(
        validated_drift,
        state,
        t0=t0,
        t1=t1,
        args=args,
        wiener_terms=wiener_terms,
        interpretation=interpretation,
        problem_id=problem_id,
    )
    return SemidiscreteSPDE(
        problem=problem,
        spatial_discretization=spatial_discretization,
        noise_basis=noise_basis,
        semilinear_drift=semilinear_drift,
        state_shape=state_shape,
        noise_shape=problem.noise_shape,
        basis_id=problem.noise_id,
        solution_spec=solution_spec,
    )


def semidiscretize_semilinear_spde(
    linear_operator: AbstractLinearOperator,
    nonlinear_drift: Callable[[Array, Array, Any], ArrayLike] | None,
    initial_state: ArrayLike,
    spatial_discretization: SpatialDiscretization,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    operator_id: str,
    nonlinear_id: str | None = None,
    args: Any = None,
    diffusion: Callable[[Array, Array, Any], ArrayLike] | None = None,
    noise_basis: SpatialNoiseBasis | None = None,
    noise_shape: Sequence[int] | None = None,
    basis_id: str | None = None,
    noise_structure: NoiseStructure | None = None,
    interpretation: DifferentialInterpretation = "ito",
    solution_spec: SPDESolutionSpec | None = None,
    mass_self_adjoint: bool = False,
    mass_weights: ArrayLike | None = None,
    spectral_bounds: tuple[float, float] | None = None,
    spectral_representation: TransformDiagonalRepresentation | None = None,
    compatible_noise_eigenvalues: ArrayLike | None = None,
) -> SemidiscreteSPDE:
    """Semidiscretize an explicitly identified semilinear stochastic equation.

    ``nonlinear_id`` is required when ``nonlinear_drift`` is callable. When the
    nonlinear drift is ``None``, the API binds a canonical zero-drift identity.
    """
    resolved_nonlinear_id = _resolve_optional_callable_id(
        nonlinear_drift,
        nonlinear_id,
        value_name="nonlinear_drift",
        identifier_name="nonlinear_id",
        zero_identifier=_ZERO_SEMILINEAR_NONLINEAR_ID,
    )
    state_shape = tuple(int(size) for size in jnp.asarray(initial_state).shape)
    if not isinstance(linear_operator, AbstractLinearOperator):
        raise TypeError("linear_operator must be an AbstractLinearOperator.")
    compatible_basis_id = (
        None
        if compatible_noise_eigenvalues is None
        else (
            noise_basis.basis_id
            if noise_basis is not None
            else (None if basis_id is None else str(basis_id))
        )
    )
    semilinear = SemilinearDrift(
        linear_operator,
        nonlinear_drift,
        state_shape=state_shape,
        operator_id=operator_id,
        nonlinear_id=resolved_nonlinear_id,
        mass_self_adjoint=mass_self_adjoint,
        mass_weights=mass_weights,
        spectral_bounds=spectral_bounds,
        spectral_representation=spectral_representation,
        compatible_noise_eigenvalues=compatible_noise_eigenvalues,
        compatible_noise_basis_id=compatible_basis_id,
    )
    return semidiscretize_spde(
        semilinear,
        initial_state,
        spatial_discretization,
        t0=t0,
        t1=t1,
        args=args,
        semilinear_drift=semilinear,
        diffusion=diffusion,
        noise_basis=noise_basis,
        noise_shape=noise_shape,
        basis_id=basis_id,
        noise_structure=noise_structure,
        interpretation=interpretation,
        solution_spec=solution_spec,
    )


def semidiscretize_reaction_diffusion(
    initial_state: ArrayLike,
    spatial_discretization: SpatialDiscretization,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    kappa: ArrayLike | Callable[[Array, Array, Any], ArrayLike],
    reaction: Callable[[Array, Array, Any], ArrayLike] | None = None,
    reaction_id: str | None = None,
    args: Any = None,
    noise_basis: SpatialNoiseBasis | None = None,
    noise_amplitude: Callable[[Array, Array, Any], ArrayLike] | None = None,
    interpretation: DifferentialInterpretation = "ito",
    noise_structure: NoiseStructure | None = None,
    solution_spec: SPDESolutionSpec | None = None,
) -> SemidiscreteSPDE:
    r"""Semidiscretize stochastic reaction--diffusion dynamics.

    This constructs

    .. math::

        dU_t=[\kappa\Delta_hU_t+R(t,U_t,a)]dt+g(t,U_t,a)B\,dW_t.

    When a noise basis is supplied and ``noise_amplitude`` is omitted, the noise is
    additive. Tensor state shape is preserved.

    ``TensorSpectralDiscretization`` uses modal primary state. Its
    ``initial_state``, ``reaction`` result, and state-shaped noise amplitude are
    therefore modal. Evaluate physical nonlinearities through a prepared
    ``PreparedPseudospectralMethod.nonlinear_action`` so the declared dealiasing
    policy remains explicit.

    A callable ``reaction`` requires an explicit stable ``reaction_id``. Omitting
    the reaction binds a canonical no-reaction identity.
    """
    resolved_reaction_id = _resolve_optional_callable_id(
        reaction,
        reaction_id,
        value_name="reaction",
        identifier_name="reaction_id",
        zero_identifier=_NO_REACTION_ID,
    )
    state_shape = tuple(int(size) for size in jnp.asarray(initial_state).shape)
    semilinear: SemilinearDrift | None
    if callable(kappa):
        drift: Callable[[Array, Array, Any], ArrayLike] = _ReactionDiffusionDrift(
            spatial_discretization,
            kappa,
            reaction,
            resolved_reaction_id,
            state_shape,
        )
        semilinear = None
    else:
        initial = jnp.asarray(initial_state)
        state_dtype = initial.dtype
        coefficient_values = jnp.asarray(kappa)
        if jnp.iscomplexobj(coefficient_values):
            if not bool(jnp.all(jnp.imag(coefficient_values) == 0)):
                raise ValueError("kappa must be real-valued.")
            coefficient_values = jnp.real(coefficient_values)
        coefficient = coefficient_values.astype(initial.real.dtype)
        if coefficient.shape not in ((), state_shape):
            raise ValueError("kappa must be scalar or have exact initial-state shape.")
        linear_operator = _ScaledLaplacianOperator(
            spatial_discretization,
            coefficient,
        )
        compatible_eigenvalues = _compatible_noise_eigenvalues(
            spatial_discretization,
            noise_basis,
            coefficient,
        )
        operator_id = (
            f"{_spatial_id(spatial_discretization)}:scaled-laplacian:{coefficient!r}"
        )
        weights = _spatial_weights(spatial_discretization)
        expanded_weights = jnp.broadcast_to(
            weights.reshape(weights.shape + (1,) * (len(state_shape) - weights.ndim)),
            state_shape,
        ).astype(state_dtype)
        pairing = (
            DiagonalPairing(
                expanded_weights,
                pairing_id=f"{operator_id}:mass-pairing",
            )
            if coefficient.shape == ()
            else None
        )
        operator_space = ArraySpace(
            state_shape,
            dtype=state_dtype,
            pairing=pairing,
        )
        if (
            isinstance(spatial_discretization, TensorSpectralDiscretization)
            and coefficient.shape == ()
        ):
            eigenvalues = spatial_discretization.laplacian_eigenvalues()
            diagonal = -coefficient * jnp.broadcast_to(
                eigenvalues.reshape(
                    eigenvalues.shape + (1,) * (len(state_shape) - eigenvalues.ndim)
                ),
                state_shape,
            )
            canonical_operator = DiagonalLinearOperator(
                diagonal.reshape((-1,)),
                space=operator_space,
                properties=OperatorProperties(
                    diagonal=True,
                    self_adjoint=True,
                    evidence={
                        "diagonal": "construction",
                        "self_adjoint": "construction",
                    },
                ),
                operator_id=operator_id,
            )
            spectral = None
            lower = float(jnp.min(jnp.real(diagonal)))
            upper = float(jnp.max(jnp.real(diagonal)))
            bounds = (lower, upper) if lower < upper else None
        else:
            canonical_operator = FunctionLinearOperator(
                linear_operator,
                source=operator_space,
                target=operator_space,
                properties=OperatorProperties(
                    self_adjoint=coefficient.shape == (),
                    evidence=(
                        {"self_adjoint": "construction"}
                        if coefficient.shape == ()
                        else None
                    ),
                ),
                operator_id=operator_id,
            )
            spectral = _spectral_linear_representation(
                canonical_operator,
                spatial_discretization,
                coefficient,
                state_shape,
            )
            bounds = None
            if spectral is not None:
                lower = float(jnp.min(spectral.modal_values))
                upper = float(jnp.max(spectral.modal_values))
                if lower < upper:
                    bounds = (lower, upper)
        semilinear = SemilinearDrift(
            canonical_operator,
            reaction,
            state_shape=state_shape,
            operator_id=operator_id,
            nonlinear_id=resolved_reaction_id,
            mass_self_adjoint=coefficient.shape == (),
            mass_weights=weights,
            spectral_bounds=bounds,
            spectral_representation=spectral,
            compatible_noise_eigenvalues=compatible_eigenvalues,
            compatible_noise_basis_id=(
                None
                if compatible_eigenvalues is None or noise_basis is None
                else noise_basis.basis_id
            ),
        )
        drift = semilinear
    return semidiscretize_spde(
        drift,
        initial_state,
        spatial_discretization,
        t0=t0,
        t1=t1,
        args=args,
        semilinear_drift=semilinear,
        diffusion=noise_amplitude,
        noise_basis=noise_basis,
        noise_structure=noise_structure,
        interpretation=interpretation,
        solution_spec=solution_spec,
    )


__all__ = [
    "SemidiscreteSPDE",
    "semidiscretize_reaction_diffusion",
    "semidiscretize_semilinear_spde",
    "semidiscretize_spde",
]
