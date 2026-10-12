#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import finite_real_scalar, positive_integer
from ...typing import checked, Float, parse, VariadicDim
from ._space import TensorSpectralDiscretization


SpectralViscosityKernel: TypeAlias = Literal["tadmor_step", "exponential_highpass"]


class SpectralViscosityModalDims(VariadicDim):
    """Retained tensor modal axes of the prepared viscosity operator."""


class SpectralVanishingViscosityPlan(StrictModule, NonTrainableState):
    """Explicit spectral viscosity scaling on an isotropic periodic box.

    The scaling epsilon_K = epsilon K^(1-2s), m_K = m0 K^theta follows
    Rohner--Mishra (2024), section 2 (spectral hyperviscosity approximation).
    The step profile has an exactly undamped low band. The exponential profile
    is practical high-pass hyperviscosity, not the cited theorem's operator.
    No operator alone certifies rough-data convergence without initialization
    hypotheses; see Lanthaler--Mishra (2019), spectral viscosity formulation.
    """

    strength: float = eqx.field(static=True)
    order: float = eqx.field(static=True)
    cutoff_scale: float = eqx.field(static=True)
    cutoff_exponent: float = eqx.field(static=True)
    kernel: SpectralViscosityKernel = eqx.field(static=True)
    highpass_exponent: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        strength: float,
        order: float,
        cutoff_scale: float,
        cutoff_exponent: float,
        kernel: SpectralViscosityKernel,
        highpass_exponent: int | None = None,
    ) -> None:
        epsilon = finite_real_scalar(strength, "strength")
        s = finite_real_scalar(order, "order")
        m0 = finite_real_scalar(cutoff_scale, "cutoff_scale")
        theta = finite_real_scalar(cutoff_exponent, "cutoff_exponent")
        if epsilon <= 0.0 or s < 1.0 or m0 <= 0.0 or not 0.0 < theta <= 1.0:
            raise ValueError(
                "Require strength > 0, order >= 1, cutoff_scale > 0, and cutoff_exponent in (0, 1]."
            )
        kernel_ = parse(kernel, SpectralViscosityKernel, "kernel")
        exponent = (
            None
            if highpass_exponent is None
            else positive_integer(highpass_exponent, "highpass_exponent")
        )
        match kernel_:
            case "tadmor_step":
                if exponent is not None:
                    raise ValueError("tadmor_step requires highpass_exponent=None.")
            case "exponential_highpass":
                if exponent is None:
                    raise ValueError(
                        "exponential_highpass requires an integer highpass_exponent >= 2."
                    )
                if exponent < 2:
                    raise ValueError(
                        "exponential_highpass requires highpass_exponent >= 2."
                    )
            case _:
                assert_never(kernel_)
        self.strength = epsilon
        self.order = s
        self.cutoff_scale = m0
        self.cutoff_exponent = theta
        self.kernel = kernel_
        self.highpass_exponent = exponent
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-vanishing-viscosity-plan",
                "strength": epsilon,
                "order": s,
                "cutoff_scale": m0,
                "cutoff_exponent": theta,
                "kernel": kernel_,
                "highpass_exponent": exponent,
            }
        )

    @checked
    def prepare(
        self, discretization: TensorSpectralDiscretization, /
    ) -> PreparedSpectralVanishingViscosity:
        """Prepare the immutable diagonal and its scientific evidence on the host."""
        return PreparedSpectralVanishingViscosity(self, discretization)


class PreparedSpectralVanishingViscosity(StrictModule, NonTrainableState):
    """Nonpositive modal decay rates and qualified operator-scaling evidence."""

    __strict_contract__ = True

    decay_rate: Float[SpectralViscosityModalDims]
    resolution: int = eqx.field(static=True)
    scaled_strength: float = eqx.field(static=True)
    cutoff_mode: float = eqx.field(static=True)
    maximum_decay_rate: float = eqx.field(static=True)
    active_mode_count: int = eqx.field(static=True)
    operator_scaling_admissible: bool = eqx.field(static=True)
    cited_convergence_supported: bool = eqx.field(static=True)
    support_reason: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: SpectralVanishingViscosityPlan,
        discretization: TensorSpectralDiscretization,
        /,
    ) -> None:
        axes = discretization.axes
        if any(axis.family != "fourier" for axis in axes):
            raise ValueError("Spectral viscosity requires all Fourier axes.")
        lengths = tuple(float(axis.length) for axis in axes)
        counts = tuple(axis.mode_count for axis in axes)
        if len(set(lengths)) != 1 or len(set(counts)) != 1:
            raise ValueError(
                "Spectral viscosity requires an isotropic box with equal axis point counts."
            )
        squared = np.zeros(discretization.modal_shape, dtype=np.float64)
        retained = np.ones(discretization.modal_shape, dtype=np.bool_)
        resolution = (counts[0] - 1) // 2
        if resolution < 1:
            raise ValueError("Spectral viscosity damps no retained mode.")
        for axis_index, axis in enumerate(axes):
            shape = [1] * len(axes)
            shape[axis_index] = axis.mode_count
            numbers = np.asarray(axis.modes.mode_numbers, dtype=np.float64).reshape(
                tuple(shape)
            )
            squared += numbers**2
            retained &= ~np.asarray(axis.modes.nyquist_mask, dtype=np.bool_).reshape(
                tuple(shape)
            )
        scaled = plan.strength * resolution ** (1.0 - 2.0 * plan.order)
        cutoff = plan.cutoff_scale * resolution**plan.cutoff_exponent
        ratio = np.sqrt(squared) / cutoff
        match plan.kernel:
            case "tadmor_step":
                profile = (ratio > 1.0).astype(np.float64)
            case "exponential_highpass":
                exponent = plan.highpass_exponent
                if exponent is None:
                    raise ValueError(
                        "Exponential profile requires its prepared exponent."
                    )
                profile = -np.expm1(-(ratio**exponent))
            case _:
                assert_never(plan.kernel)
        physical_squared = squared * (2.0 * np.pi / lengths[0]) ** 2
        decay = np.where(
            retained & (squared > 0.0),
            -scaled * physical_squared**plan.order * profile,
            0.0,
        )
        coefficient_dtype = np.dtype(discretization.plan.precision.coefficient_dtype)
        real_dtype = np.empty((), dtype=coefficient_dtype).real.dtype
        decay = decay.astype(real_dtype)
        if not np.all(np.isfinite(decay)):
            raise ValueError("Prepared spectral viscosity decay rates must be finite.")
        active = np.count_nonzero(decay < 0.0)
        if active == 0:
            raise ValueError("Spectral viscosity damps no retained mode.")
        self.decay_rate = jnp.asarray(decay, dtype=real_dtype)
        self.resolution = resolution
        self.scaled_strength = scaled
        self.cutoff_mode = cutoff
        self.maximum_decay_rate = float(np.max(-decay))
        self.active_mode_count = index(active)
        self.operator_scaling_admissible = (
            plan.kernel == "tadmor_step"
            and 0.0 < plan.cutoff_exponent < (2.0 * plan.order - 1.0) / (2.0 * plan.order)
        )
        self.cited_convergence_supported = False
        self.support_reason = "Operator scaling alone does not establish cited rough-data convergence: the initial-data mollification and full-method hypotheses are not certified."
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-vanishing-viscosity",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
            }
        )

    def rate(self, coefficients: ArrayLike, /) -> Array:
        """Apply the diagonal to modal axes followed by optional component axes."""
        value = jnp.asarray(coefficients)
        if value.shape[: self.decay_rate.ndim] != self.decay_rate.shape:
            raise ValueError("SVV coefficients must begin with the prepared modal shape.")
        shape = self.decay_rate.shape + (1,) * (value.ndim - self.decay_rate.ndim)
        return self.decay_rate.reshape(shape) * value

    def dissipation(self, coefficients: ArrayLike, /) -> Array:
        """Return -sum(lambda |c|^2), the nonnegative integrated energy loss."""
        value = jnp.asarray(coefficients)
        return -jnp.sum(jnp.real(jnp.conj(value) * self.rate(value)))


__all__ = [
    "SpectralViscosityKernel",
    "SpectralVanishingViscosityPlan",
    "PreparedSpectralVanishingViscosity",
]
