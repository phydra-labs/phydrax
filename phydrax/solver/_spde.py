#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ._differential import (
    DifferentialInterpretation,
    DifferentialProblem,
    LevyAreaKind,
    WienerDriver,
)
from ._noise import SpatialNoiseBasis
from ._spatial import AbstractSpatialDiscretization


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
    discretization: AbstractSpatialDiscretization
    kappa: Any
    reaction: Callable[[Array, Array, Any], ArrayLike] | None
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
        return coefficient_array * self.discretization.laplacian(state_array) + reaction


class SemidiscreteSPDE(StrictModule):
    """Finite-dimensional method-of-lines problem plus spatial/noise provenance."""

    problem: DifferentialProblem
    spatial_discretization: AbstractSpatialDiscretization
    noise_basis: SpatialNoiseBasis | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: DifferentialProblem,
        spatial_discretization: AbstractSpatialDiscretization,
        noise_basis: SpatialNoiseBasis | None,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        basis_id: str | None,
    ):
        if not isinstance(problem, DifferentialProblem):
            raise TypeError("problem must be a DifferentialProblem.")
        if not isinstance(spatial_discretization, AbstractSpatialDiscretization):
            raise TypeError(
                "spatial_discretization must implement AbstractSpatialDiscretization."
            )
        self.problem = problem
        self.spatial_discretization = spatial_discretization
        self.noise_basis = noise_basis
        self.state_shape = tuple(int(size) for size in state_shape)
        self.noise_shape = tuple(int(size) for size in noise_shape)
        self.discretization_id = spatial_discretization.discretization_id
        self.basis_id = basis_id

    def wiener_driver(
        self,
        key: Key[Array, ""],
        /,
        *,
        tolerance: float = 1e-3,
        levy_area: LevyAreaKind = "brownian",
        realization_id: str | int | None = None,
    ) -> WienerDriver:
        """Create a driver synchronized with this problem's retained noise basis."""
        if not self.problem.stochastic:
            raise ValueError("Deterministic semidiscrete problems have no Wiener driver.")
        return WienerDriver(
            key,
            self.noise_shape,
            tolerance=tolerance,
            levy_area=levy_area,
            basis_id=self.basis_id,
            realization_id=realization_id,
        )


def semidiscretize_spde(
    drift: Callable[[Array, Array, Any], ArrayLike],
    initial_state: ArrayLike,
    spatial_discretization: AbstractSpatialDiscretization,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    args: Any = None,
    diffusion: Callable[[Array, Array, Any], ArrayLike] | None = None,
    noise_basis: SpatialNoiseBasis | None = None,
    noise_shape: Sequence[int] | None = None,
    basis_id: str | None = None,
    interpretation: DifferentialInterpretation = "ito",
) -> SemidiscreteSPDE:
    r"""Compose a validated finite-rank method-of-lines SPDE.

    With ``noise_basis``, omitting ``diffusion`` gives additive noise. Supplying it
    gives a scalar/pointwise amplitude :math:`G_h` (or a fully composed diffusion
    factor) multiplying the retained basis :math:`B`. Without a basis, stochastic
    problems must supply both ``diffusion`` and ``noise_shape``.
    """
    if not callable(drift):
        raise TypeError("drift must be callable.")
    if not isinstance(spatial_discretization, AbstractSpatialDiscretization):
        raise TypeError(
            "spatial_discretization must implement AbstractSpatialDiscretization."
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
    if noise_basis is not None:
        if not isinstance(noise_basis, SpatialNoiseBasis):
            raise TypeError("noise_basis must be a SpatialNoiseBasis or None.")
        if noise_basis.state_shape != state_shape:
            raise ValueError(
                "noise basis state shape must match initial_state exactly; "
                f"got {noise_basis.state_shape} and {state_shape}."
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
    else:
        if noise_shape is not None or basis_id is not None:
            raise ValueError(
                "noise_shape and basis_id are only valid for stochastic problems."
            )
        resolved_noise_shape = ()
        resolved_basis_id = None
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
    problem = DifferentialProblem(
        validated_drift,
        state,
        t0=t0,
        t1=t1,
        args=args,
        diffusion=validated_diffusion,
        interpretation=interpretation,
    )
    return SemidiscreteSPDE(
        problem=problem,
        spatial_discretization=spatial_discretization,
        noise_basis=noise_basis,
        state_shape=state_shape,
        noise_shape=resolved_noise_shape,
        basis_id=resolved_basis_id,
    )


def semidiscretize_reaction_diffusion(
    initial_state: ArrayLike,
    spatial_discretization: AbstractSpatialDiscretization,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    kappa: ArrayLike | Callable[[Array, Array, Any], ArrayLike],
    reaction: Callable[[Array, Array, Any], ArrayLike] | None = None,
    args: Any = None,
    noise_basis: SpatialNoiseBasis | None = None,
    noise_amplitude: Callable[[Array, Array, Any], ArrayLike] | None = None,
    interpretation: DifferentialInterpretation = "ito",
) -> SemidiscreteSPDE:
    r"""Semidiscretize stochastic reaction--diffusion dynamics.

    This constructs

    .. math::

        dU_t=[\kappa\Delta_hU_t+R(t,U_t,a)]dt+g(t,U_t,a)B\,dW_t.

    When a noise basis is supplied and ``noise_amplitude`` is omitted, the noise is
    additive. Tensor state shape is preserved.
    """
    if reaction is not None and not callable(reaction):
        raise TypeError("reaction must be callable or None.")
    if callable(kappa) is False:
        coefficient = jnp.asarray(kappa)
        if coefficient.shape not in ((), tuple(jnp.asarray(initial_state).shape)):
            raise ValueError("kappa must be scalar or have exact initial-state shape.")
    if noise_amplitude is not None and noise_basis is None:
        raise ValueError("noise_amplitude requires a noise_basis.")
    state_shape = tuple(int(size) for size in jnp.asarray(initial_state).shape)
    drift = _ReactionDiffusionDrift(
        spatial_discretization,
        kappa,
        reaction,
        state_shape,
    )
    return semidiscretize_spde(
        drift,
        initial_state,
        spatial_discretization,
        t0=t0,
        t1=t1,
        args=args,
        diffusion=noise_amplitude,
        noise_basis=noise_basis,
        interpretation=interpretation,
    )


__all__ = [
    "SemidiscreteSPDE",
    "semidiscretize_reaction_diffusion",
    "semidiscretize_spde",
]
