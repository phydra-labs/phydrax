#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._execution_resources import MemoryBudgetError
from .._fingerprint import canonical_fingerprint
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from .._validation import finite_real_scalar, positive_integer
from ..discretization.spectral._coordinates import HermitianSpectralCoordinates
from ..discretization.spectral._incompressible import PeriodicLerayProjector
from ..discretization.spectral._space import TensorSpectralDiscretization
from ..discretization.spectral._transfer import RetainedBandRestrictionPlan
from ..typing import checked, Complex, Dim, Float, parse, PRNGKey, UInt32, VariadicDim


FourierCoefficientLaw: TypeAlias = Literal["normal", "uniform"]
FourierSeriesBasis: TypeAlias = Literal["complex_exponential", "tensor_trigonometric"]


class _MemberDim(Dim):
    pass


class _FieldDims(VariadicDim):
    pass


class PowerLawSpectralAmplitude(StrictModule):
    """Mean-zero amplitude ``A(k)=scale*k**(-exponent)`` with a radial cutoff.

    Herde et al. (2024), Appendix B.2.2, Eq. (42), motivates power-law
    trigonometric initial fields. Here the exponent is explicit and real
    coordinates have unit variance, unlike that paper's U[-1,1] recipe.
    No continuum convergence is asserted for a finite periodic truncation.
    """

    exponent: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    maximum_wavenumber: float | None = eqx.field(static=True)
    amplitude_id: str = eqx.field(static=True)

    def __init__(
        self,
        exponent: float,
        /,
        *,
        scale: float = 1.0,
        maximum_wavenumber: float | None = None,
    ) -> None:
        exponent_ = finite_real_scalar(exponent, "exponent")
        scale_ = finite_real_scalar(scale, "scale")
        cutoff = (
            None
            if maximum_wavenumber is None
            else finite_real_scalar(maximum_wavenumber, "maximum_wavenumber")
        )
        if scale_ <= 0:
            raise ValueError("scale must be finite and positive.")
        if cutoff is not None and cutoff <= 0:
            raise ValueError("maximum_wavenumber must be finite and positive.")
        self.exponent = exponent_
        self.scale = scale_
        self.maximum_wavenumber = cutoff
        self.amplitude_id = canonical_fingerprint(
            {
                "kind": "power-law-spectral-amplitude",
                "exponent": exponent_,
                "scale": scale_,
                "maximum_wavenumber": cutoff,
            }
        )

    def __call__(self, wavenumber: ArrayLike, /) -> Array:
        """Evaluate the amplitude, exactly zero at the origin and beyond cutoff."""
        wave = jnp.asarray(wavenumber)
        active = wave > 0
        if self.maximum_wavenumber is not None:
            active = active & (wave <= self.maximum_wavenumber)
        safe = jnp.where(active, wave, jnp.ones_like(wave))
        return jnp.where(
            active, self.scale * safe ** (-self.exponent), jnp.zeros_like(wave)
        )


class PeriodicPowerLawFieldPlan(StrictModule):
    """Periodic real fields with iid unit-variance latent real coordinates.

    Without projection, every retained nonzero complex coefficient has expected
    squared magnitude ``V*A(|k|)**2`` per component. Complex conjugate pairs use
    ``(a+ib)/sqrt(2)``. Uniform latents are U[-sqrt(3),sqrt(3)], not Gaussian.

    Tensor-trigonometric synthesis multiplies the unnormalized product series
    by ``2**(r/2)`` modewise, where r counts nonzero frequency axes. This yields
    the same expected spectrum. Sine factors at zero vanish identically; even
    Nyquist frequencies (both sine and cosine) are excluded from the retained
    band. Consequently no Nyquist sine or cosine latent contributes.
    """

    amplitude: PowerLawSpectralAmplitude
    law: FourierCoefficientLaw = eqx.field(static=True)
    basis: FourierSeriesBasis = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    solenoidal: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        amplitude: PowerLawSpectralAmplitude,
        /,
        *,
        law: FourierCoefficientLaw = "normal",
        basis: FourierSeriesBasis = "complex_exponential",
        component_shape: tuple[int, ...] = (),
        solenoidal: bool = False,
    ) -> None:
        law_ = parse(law, FourierCoefficientLaw, "law")
        basis_ = parse(basis, FourierSeriesBasis, "basis")
        components = tuple(
            positive_integer(size, "component_shape dimension")
            for size in component_shape
        )
        if not isinstance(solenoidal, bool):
            raise TypeError("solenoidal must be a boolean.")
        self.amplitude, self.law, self.basis = amplitude, law_, basis_
        self.component_shape, self.solenoidal = components, solenoidal
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-power-law-field-plan",
                "amplitude": amplitude.amplitude_id,
                "law": law_,
                "basis": basis_,
                "component_shape": list(components),
                "solenoidal": solenoidal,
            }
        )

    @classmethod
    def fractional_gaussian(
        cls,
        hurst: float,
        /,
        *,
        spatial_dimension: int,
        component_shape: tuple[int, ...] = (),
        solenoidal: bool = False,
        scale: float = 1.0,
        maximum_wavenumber: float | None = None,
    ) -> PeriodicPowerLawFieldPlan:
        """Gaussian coefficients with exponent H+d/2, for 0<H<1.

        This is a periodic, finite-band fractional-Gaussian field, not a
        nonperiodic fractional Brownian process. The exponent follows from
        radial spectral scaling: k**(d-1) A(k)**2 = k**(-1-2H), hence continuum
        expected squared increments scale as r**(2H). Herde et al. (2024),
        Appendix B.2.2, Eq. (42), motivates the power-law field family, not
        this Gaussian/unit-variance normalization.
        """
        dimension = positive_integer(spatial_dimension, "spatial_dimension")
        h = finite_real_scalar(hurst, "hurst")
        if not 0 < h < 1:
            raise ValueError("Require spatial_dimension >= 1 and 0 < hurst < 1.")
        return cls(
            PowerLawSpectralAmplitude(
                h + dimension / 2, scale=scale, maximum_wavenumber=maximum_wavenumber
            ),
            component_shape=component_shape,
            solenoidal=solenoidal,
        )

    @checked
    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        member_chunk_size: int = 8,
        maximum_sample_bytes: int = 2**30,
    ) -> PreparedPeriodicPowerLawField:
        return PreparedPeriodicPowerLawField(
            self,
            discretization,
            member_chunk_size=member_chunk_size,
            maximum_sample_bytes=maximum_sample_bytes,
        )


class PeriodicPowerLawFieldRealization(StrictModule):
    """Addressed member coefficients and cumulative removed coefficient energy.

    Energy means sum |c|², twice the integrated kinetic energy. The baseline
    before projection is preserved, so restriction evidence is independent of
    the restriction path, including floating-point reduction order.
    """

    __strict_contract__ = True
    coefficients: Complex[_MemberDim, _FieldDims]
    member_indices: UInt32[_MemberDim]
    removed_coefficient_energy: Float[_MemberDim]
    baseline_coefficient_energy: Float[_MemberDim]
    discretization: TensorSpectralDiscretization
    field_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    coefficient_realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)


class PreparedPeriodicPowerLawField(StrictModule):
    """Prepared spectral weights and bounded, member-addressed execution.

    The field identity contains only science and the generation discretization;
    the prepared identity additionally contains execution caps. Sample keys
    depend on field identity and member number, never batch size or chunk size.
    """

    __strict_contract__ = True
    plan: PeriodicPowerLawFieldPlan
    discretization: TensorSpectralDiscretization
    coordinates: HermitianSpectralCoordinates
    projector: PeriodicLerayProjector | None
    modal_amplitude: Float[_FieldDims]
    address: SampleAddress
    trig_shape: tuple[int, ...] = eqx.field(static=True)
    member_chunk_size: int = eqx.field(static=True)
    maximum_sample_bytes: int = eqx.field(static=True)
    member_bytes: int = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: PeriodicPowerLawFieldPlan,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        member_chunk_size: int = 8,
        maximum_sample_bytes: int = 2**30,
    ) -> None:
        if any(axis.family != "fourier" for axis in discretization.axes):
            raise ValueError("Power-law fields require all-Fourier tensor spaces.")
        chunk = positive_integer(member_chunk_size, "member_chunk_size")
        cap = positive_integer(maximum_sample_bytes, "maximum_sample_bytes")
        dimension = len(discretization.axes)
        if plan.solenoidal and (
            dimension not in (2, 3) or plan.component_shape != (dimension,)
        ):
            raise ValueError(
                "Solenoidal fields require d in {2,3} and component_shape=(d,)."
            )
        dtype = jnp.dtype(discretization.plan.precision.coefficient_dtype)
        real_dtype = jnp.empty((), dtype=dtype).real.dtype
        state_size = prod(discretization.modal_shape + plan.component_shape)
        trig_shape = (
            tuple(
                size for n in discretization.modal_shape for size in ((n - 1) // 2 + 1, 2)
            )
            + plan.component_shape
        )
        # Admission covers padded output plus live coefficient/latent worksets.
        member_bytes = (
            3 * state_size * dtype.itemsize
            + prod(trig_shape) * real_dtype.itemsize
            + 3 * real_dtype.itemsize
            + np.dtype(np.uint32).itemsize
        )
        if chunk * member_bytes > cap:
            raise MemoryBudgetError("maximum_sample_bytes", chunk * member_bytes, cap)
        squared = np.zeros(discretization.modal_shape, dtype=np.float64)
        retained = np.ones(discretization.modal_shape, dtype=np.bool_)
        volume = 1.0
        for axis_index, axis in enumerate(discretization.axes):
            length = float(axis.length)
            volume *= length
            shape = [1] * dimension
            shape[axis_index] = axis.mode_count
            squared += (2 * np.pi * np.asarray(axis.modes.mode_numbers) / length).reshape(
                shape
            ) ** 2
            retained &= (~np.asarray(axis.modes.nyquist_mask)).reshape(shape)
        amplitudes = np.sqrt(volume) * plan.amplitude(
            jnp.asarray(np.sqrt(squared), dtype=real_dtype)
        )
        amplitudes = jnp.where(
            jnp.asarray(retained), amplitudes, jnp.zeros_like(amplitudes)
        )
        field_id = canonical_fingerprint(
            {
                "kind": "periodic-power-law-field",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
            }
        )
        self.plan, self.discretization = plan, discretization
        self.coordinates = HermitianSpectralCoordinates(
            discretization,
            component_shape=plan.component_shape,
            maximum_coordinate_size=state_size,
        )
        self.projector = (
            PeriodicLerayProjector(discretization) if plan.solenoidal else None
        )
        self.modal_amplitude = amplitudes
        self.address = SampleAddress(
            "periodic-power-law-field", "coefficients", target=field_id
        )
        self.trig_shape = trig_shape
        self.member_chunk_size, self.maximum_sample_bytes = chunk, cap
        self.member_bytes, self.field_id = member_bytes, field_id
        self.prepared_id = canonical_fingerprint(
            {"field": field_id, "member_chunk_size": chunk, "maximum_sample_bytes": cap}
        )

    @checked
    def sample(
        self, key: PRNGKey, member_indices: ArrayLike, /
    ) -> PeriodicPowerLawFieldRealization:
        """Sample nonempty uint32-addressable members; refuse resources before execution.

        Root key fingerprinting is host work done exactly once at this boundary.
        ``lax.map`` processes padded, explicitly masked fixed-size member lanes.
        """
        if str(jr.key_impl(key)) != "threefry2x32":
            raise ValueError(
                "Member-independent batched fields require threefry2x32 typed keys."
            )
        indices = jnp.asarray(member_indices)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("member_indices must be a nonempty rank-one array.")
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("member_indices must have an integer dtype.")
        host_indices = np.asarray(indices)
        if np.any(host_indices < 0) or np.any(host_indices >= 2**32):
            raise ValueError("member_indices must lie in [0, 2**32).")
        padded_count = (
            (indices.size + self.member_chunk_size - 1)
            // self.member_chunk_size
            * self.member_chunk_size
        )
        if padded_count * self.member_bytes > self.maximum_sample_bytes:
            raise MemoryBudgetError(
                "maximum_sample_bytes",
                padded_count * self.member_bytes,
                self.maximum_sample_bytes,
            )
        root_id = canonical_fingerprint(
            {
                "kind": "periodic-field-root-key",
                "implementation": str(jr.key_impl(key)),
                "words": np.asarray(jr.key_data(key)).tolist(),
            }
        )
        coupling_id = canonical_fingerprint({"field": self.field_id, "root": root_id})
        canonical_indices = indices.astype(jnp.uint32)
        padded = jnp.pad(canonical_indices, (0, padded_count - indices.size))
        valid = jnp.arange(padded_count, dtype=jnp.int64) < indices.size
        coefficients, baseline, removed = _sample_members(self, key, padded, valid)
        return PeriodicPowerLawFieldRealization(
            coefficients=coefficients[: indices.size],
            member_indices=canonical_indices,
            removed_coefficient_energy=removed[: indices.size],
            baseline_coefficient_energy=baseline[: indices.size],
            discretization=self.discretization,
            field_id=self.field_id,
            discretization_id=self.discretization.prepared_id,
            coefficient_realization_id=root_id,
            coupling_id=coupling_id,
        )

    @checked
    def restrict(
        self,
        realization: PeriodicPowerLawFieldRealization,
        target_discretization: TensorSpectralDiscretization,
        /,
    ) -> PeriodicPowerLawFieldRealization:
        """Pure canonical retained-band gather; preserve coupling and never reproject."""
        if realization.field_id != self.field_id:
            raise ValueError("Realization belongs to a different generation field.")
        restriction = RetainedBandRestrictionPlan(
            realization.discretization, target_discretization
        ).prepare()
        coefficients = jax.vmap(restriction)(realization.coefficients)
        axes = tuple(range(1, coefficients.ndim))
        removed = jnp.maximum(
            realization.baseline_coefficient_energy
            - jnp.sum(jnp.abs(coefficients) ** 2, axis=axes),
            0.0,
        )
        return PeriodicPowerLawFieldRealization(
            coefficients=coefficients,
            member_indices=realization.member_indices,
            removed_coefficient_energy=removed,
            baseline_coefficient_energy=realization.baseline_coefficient_energy,
            discretization=target_discretization,
            field_id=realization.field_id,
            discretization_id=target_discretization.prepared_id,
            coefficient_realization_id=realization.coefficient_realization_id,
            coupling_id=realization.coupling_id,
        )


def _draw(
    key: PRNGKey, shape: tuple[int, ...], law: FourierCoefficientLaw, dtype: np.dtype, /
) -> Array:
    match law:
        case "normal":
            return jr.normal(key, shape, dtype=dtype)
        case "uniform":
            bound = jnp.sqrt(jnp.asarray(3.0, dtype=dtype))
            return jr.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)
        case _:
            assert_never(law)


def _tensor_trigonometric_modal(alpha: Array, modal_shape: tuple[int, ...], /) -> Array:
    """Deterministic separable folds; each output location is written once."""
    result = alpha.astype(jnp.complex128 if alpha.dtype == jnp.float64 else jnp.complex64)
    for axis, size in enumerate(modal_shape):
        cosine = jnp.take(result, 0, axis=axis + 1)
        sine = jnp.take(result, 1, axis=axis + 1)
        zero = jnp.take(cosine, jnp.asarray([0], dtype=jnp.int32), axis=axis)
        positive_indices = jnp.arange(1, cosine.shape[axis], dtype=jnp.int32)
        positive = jnp.take(
            (cosine - 1j * sine) / jnp.sqrt(2.0), positive_indices, axis=axis
        )
        negative = jnp.flip(
            jnp.take((cosine + 1j * sine) / jnp.sqrt(2.0), positive_indices, axis=axis),
            axis=axis,
        )
        parts = (
            (zero, positive, negative)
            if size % 2
            else (zero, positive, jnp.zeros_like(zero), negative)
        )
        result = jnp.concatenate(parts, axis=axis)
    return result


@eqx.filter_jit
def _sample_members(
    prepared: PreparedPeriodicPowerLawField, key: PRNGKey, indices: Array, valid: Array, /
) -> tuple[Array, Array, Array]:
    real_dtype = prepared.modal_amplitude.dtype

    def one(lane: tuple[Array, Array]) -> tuple[Array, Array, Array]:
        member, lane_valid = lane
        member_key = derive_key(key, prepared.address, member)
        match prepared.plan.basis:
            case "complex_exponential":
                latent = _draw(
                    member_key,
                    (prepared.coordinates.coordinate_size,),
                    prepared.plan.law,
                    real_dtype,
                )
                unit = prepared.coordinates.from_real_coordinates(latent)
            case "tensor_trigonometric":
                alpha = _draw(
                    member_key, prepared.trig_shape, prepared.plan.law, real_dtype
                )
                unit = _tensor_trigonometric_modal(
                    alpha, prepared.discretization.modal_shape
                )
            case _:
                assert_never(prepared.plan.basis)
        weights = prepared.modal_amplitude.reshape(
            prepared.discretization.modal_shape
            + (1,) * len(prepared.plan.component_shape)
        )
        initial = unit * weights
        baseline = jnp.sum(jnp.abs(initial) ** 2)
        coefficients = (
            initial if prepared.projector is None else prepared.projector.project(initial)
        )
        removed = jnp.maximum(baseline - jnp.sum(jnp.abs(coefficients) ** 2), 0.0)
        return (
            jnp.where(lane_valid, coefficients, jnp.zeros_like(coefficients)),
            jnp.where(lane_valid, baseline, 0.0),
            jnp.where(lane_valid, removed, 0.0),
        )

    return jax.lax.map(one, (indices, valid), batch_size=prepared.member_chunk_size)


__all__ = [
    "FourierCoefficientLaw",
    "FourierSeriesBasis",
    "PowerLawSpectralAmplitude",
    "PeriodicPowerLawFieldPlan",
    "PreparedPeriodicPowerLawField",
    "PeriodicPowerLawFieldRealization",
]
