#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from math import prod
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.special import j0, j1

from ..._execution_resources import MemoryBudgetError
from ..._fingerprint import canonical_fingerprint
from ..._sampling._addressing import derive_key, SampleAddress
from ..._spectral._nonuniform_fourier import (
    NonuniformFourierPlan,
    PreparedNonuniformFourier,
)
from ..._strict import StrictModule
from ..._validation import finite_real_scalar, nonnegative_integer, positive_integer
from ...discretization._axis_domain import AxisDomain
from ...discretization.spectral._basis import FourierBasisPlan
from ...discretization.spectral._incompressible import PeriodicLerayProjector
from ...discretization.spectral._space import (
    TensorSpectralDiscretization,
    TensorSpectralPlan,
)
from ...discretization.spectral._transfer import (
    PreparedRetainedBandRestriction,
    RetainedBandRestrictionPlan,
)
from ...ein import contract
from ...stochastic._periodic_power_law_field import (
    PeriodicPowerLawFieldPlan,
    PreparedPeriodicPowerLawField,
)
from ...typing import checked, Complex, Dim, Float, parse, PRNGKey, UInt32, VariadicDim


ShearLayerProfile: TypeAlias = Literal["smooth", "discontinuous"]


class _MemberDim(Dim):
    pass


class _VelocityDims(VariadicDim):
    pass


def _range(value: tuple[float, float], name: str, /) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must have two endpoints.")
    lo, hi = (finite_real_scalar(v, name) for v in value)
    if lo > hi:
        raise ValueError(f"{name} must be ordered.")
    return lo, hi


def _periodic_lower(domain: AxisDomain, /) -> float:
    lower = domain.lower
    if lower is None:
        raise ValueError("Periodic initial measures require finite axis bounds.")
    return float(lower)


@final
class TaylorGreenFamily(StrictModule):
    """Classical Taylor--Green (Taylor & Green, 1937), normalized box angles.

    ``A(sin(x)cos(y),-cos(x)sin(y))`` in 2D; multiply both components by
    ``cos(z)`` and append zero in 3D. Equal box lengths are required.
    """

    spatial_dimension: int = eqx.field(static=True)
    amplitude: float = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(self, spatial_dimension: int, /, *, amplitude: float = 1.0) -> None:
        dimension = positive_integer(spatial_dimension, "spatial_dimension")
        if dimension not in (2, 3):
            raise ValueError("Taylor--Green requires dimension 2 or 3.")
        amplitude_ = finite_real_scalar(amplitude, "amplitude")
        self.spatial_dimension, self.amplitude = dimension, amplitude_
        self.family_id = canonical_fingerprint(
            {"family": "taylor-green", "dimension": dimension, "amplitude": amplitude_}
        )


@final
class ShearLayerFamily(StrictModule):
    """Displaced LMP layer profile with Poseidon's mode-weighted perturbation.

    ``P U(x,mod(y+sigma(x),1))`` uses the periodic extension of LMP §5.2
    Eq. (42)'s layer profile and Poseidon Appendix B.2.3 Eq. (45)'s
    ``sigma=xi+delta*sum(alpha_k*sin(2*pi*k*x-beta_k))``.
    This is not LMP Eq. (43), which uses a single frequency for every term.
    Smooth tanh thickness is positive in normalized box coordinates. Wrapping
    precedes both the branch comparison and tanh evaluation, so periodicity
    imposes no displacement bound: any finite shift and nonnegative perturbation
    amplitude are admitted. Nonnegativity specifies the declared amplitude law,
    not a periodicity constraint. Discontinuous layers refuse thickness.
    Oversampling is an explicit point-quadrature policy, not exactness.

    Poseidon Appendix B.2.3, Eqs. (44)--(45), uses q=7..12, delta=.025,
    xi in [-.0625,.0625], rho_P in [.08,.12]. Its tanh argument is
    ``2*pi*(y-.25)/rho_P`` without sigma inside: it is NOT equivalent to this
    displaced LMP family, even with thickness=rho_P/(2*pi).
    No Poseidon-equivalence claim is made.
    """

    profile: ShearLayerProfile = eqx.field(static=True)
    thickness_range: tuple[float, float] | None = eqx.field(static=True)
    perturbation_amplitude: float = eqx.field(static=True)
    perturbation_mode_range: tuple[int, int] = eqx.field(static=True)
    interface_shift_range: tuple[float, float] = eqx.field(static=True)
    oversampling: int = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        profile: ShearLayerProfile,
        thickness_range: tuple[float, float] | None,
        perturbation_amplitude: float,
        perturbation_mode_range: tuple[int, int],
        interface_shift_range: tuple[float, float],
        oversampling: int,
    ) -> None:
        profile_ = parse(profile, ShearLayerProfile, "profile")
        thickness = None
        match profile_:
            case "smooth":
                if thickness_range is None:
                    raise ValueError("Smooth layers require thickness_range.")
                thickness = _range(thickness_range, "thickness_range")
                if thickness[0] <= 0:
                    raise ValueError("Thickness must be positive.")
            case "discontinuous":
                if thickness_range is not None:
                    raise ValueError("Discontinuous layers refuse thickness.")
            case _:
                assert_never(profile_)
        if len(perturbation_mode_range) != 2:
            raise ValueError("perturbation_mode_range must have two endpoints.")
        modes = tuple(
            positive_integer(v, "perturbation_mode_range")
            for v in perturbation_mode_range
        )
        if modes[0] > modes[1]:
            raise ValueError("Mode range must be ordered.")
        if modes[1] >= 2**31 - 1:
            raise ValueError("Mode counts must be representable by the int32 draw.")
        shift = _range(interface_shift_range, "interface_shift_range")
        delta = finite_real_scalar(perturbation_amplitude, "perturbation_amplitude")
        if delta < 0:
            raise ValueError("perturbation_amplitude must be nonnegative.")
        oversample = positive_integer(oversampling, "oversampling")
        if oversample < 2:
            raise ValueError("oversampling must be at least 2.")
        self.profile, self.thickness_range = profile_, thickness
        self.perturbation_amplitude, self.perturbation_mode_range = (
            delta,
            (modes[0], modes[1]),
        )
        self.interface_shift_range, self.oversampling = shift, oversample
        self.family_id = canonical_fingerprint(
            {
                "family": "displaced-shear-layer",
                "profile_law": "lmp-periodic-displaced-layer",
                "perturbation_law": "poseidon-mode-weighted",
                "profile": profile_,
                "thickness": thickness,
                "delta": delta,
                "modes": modes,
                "shift": shift,
            }
        )


@final
class VortexPatchFamily(StrictModule):
    """Circular constant-vorticity patch PLUS a neutralizing uniform background.

    The disk Fourier integral is ``2*pi*r*J1(|k|r)/|k| * exp(-ik·(c-x0))``.
    Orthonormal modal coefficients divide that integral by sqrt(box volume).
    Radius/center are physical lengths/coordinates; strength is vorticity.
    This transform follows directly by integrating the disk in polar coordinates;
    no stochastic patch law or source-derived parameter defaults are imposed.
    """

    radius: float = eqx.field(static=True)
    center: tuple[float, float] = eqx.field(static=True)
    strength: float = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self, *, radius: float, center: tuple[float, float], strength: float
    ) -> None:
        radius_ = finite_real_scalar(radius, "radius")
        if radius_ <= 0 or len(center) != 2:
            raise ValueError("Require positive radius and a two-dimensional center.")
        center_ = tuple(finite_real_scalar(v, "center") for v in center)
        strength_ = finite_real_scalar(strength, "strength")
        self.radius, self.center, self.strength = (
            radius_,
            (center_[0], center_[1]),
            strength_,
        )
        self.family_id = canonical_fingerprint(
            {
                "family": "neutralized-vortex-patch",
                "radius": radius_,
                "center": center_,
                "strength": strength_,
            }
        )


@final
class PiecewiseConstantVorticityFamily(StrictModule):
    """Poseidon Appendix B.2.1, Eq. (41), with explicit mean neutralization.

    Uniform independent cell values on a uniform rectangular partition; exact
    box transforms, not transforms of point samples. The published recipe is
    partition=(10,10), value_range=(-1,1); no hidden scientific defaults.
    """

    partition: tuple[int, int] = eqx.field(static=True)
    value_range: tuple[float, float] = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self, *, partition: tuple[int, int], value_range: tuple[float, float]
    ) -> None:
        if len(partition) != 2:
            raise ValueError("partition must contain two positive cell counts.")
        cells = tuple(positive_integer(v, "partition") for v in partition)
        values = _range(value_range, "value_range")
        self.partition, self.value_range = (cells[0], cells[1]), values
        self.family_id = canonical_fingerprint(
            {
                "family": "neutralized-piecewise-vorticity",
                "partition": cells,
                "values": values,
            }
        )


@final
class SinusoidalVortexSheetFamily(StrictModule):
    """Poseidon Appendix B.2.4, Eqs. (46)--(50), neutralized arclength measure.

    Periodized radial cubic mollifier convolution. Coefficients are the exact
    discrete sum over trapezoid quadrature, NOT exact continuum coefficients.
    Q-versus-2Q residual is reported per member, not a certified error bound.
    Published recipe: rho=5/128 at N=128, d=.2, p=10, alpha in [0,.003125].
    Radius is physical; graph amplitudes/phase are normalized coordinates.
    ``quadrature_points`` is an explicit execution policy.
    """

    mollifier_radius: float = eqx.field(static=True)
    base_amplitude: float = eqx.field(static=True)
    perturbation_count: int = eqx.field(static=True)
    perturbation_amplitude_range: tuple[float, float] = eqx.field(static=True)
    quadrature_points: int = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mollifier_radius: float,
        base_amplitude: float,
        perturbation_count: int,
        perturbation_amplitude_range: tuple[float, float],
        quadrature_points: int,
    ) -> None:
        radius = finite_real_scalar(mollifier_radius, "mollifier_radius")
        amplitude = finite_real_scalar(base_amplitude, "base_amplitude")
        count = positive_integer(perturbation_count, "perturbation_count")
        bounds = _range(perturbation_amplitude_range, "perturbation_amplitude_range")
        points = positive_integer(quadrature_points, "quadrature_points")
        if radius <= 0 or points < 2:
            raise ValueError(
                "Require positive radius and at least two quadrature points."
            )
        self.mollifier_radius, self.base_amplitude = radius, amplitude
        self.perturbation_count, self.perturbation_amplitude_range = count, bounds
        self.quadrature_points = points
        self.family_id = canonical_fingerprint(
            {
                "family": "neutralized-sinusoidal-sheet",
                "radius": radius,
                "base": amplitude,
                "count": count,
                "bounds": bounds,
            }
        )


@final
class PowerLawVelocityFamily(StrictModule):
    """Wrap a solenoidal d-component periodic power-law field without resampling."""

    plan: PeriodicPowerLawFieldPlan
    family_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: PeriodicPowerLawFieldPlan, /) -> None:
        if not plan.solenoidal or plan.component_shape not in ((2,), (3,)):
            raise ValueError(
                "Require solenoidal=True and component_shape=(d,), d=2 or 3."
            )
        self.plan = plan
        self.family_id = canonical_fingerprint(
            {
                "family": "power-law-velocity",
                "plan": plan.plan_id,
            }
        )


PeriodicFlowFamily: TypeAlias = (
    TaylorGreenFamily
    | ShearLayerFamily
    | VortexPatchFamily
    | PiecewiseConstantVorticityFamily
    | SinusoidalVortexSheetFamily
    | PowerLawVelocityFamily
)


@final
class PeriodicFlowInitialState(StrictModule):
    """Coupled modal initial members and always-present removal evidence.

    Energies are sum|c|² (twice integrated kinetic energy). Baseline energy is
    retained to make cumulative restriction evidence bitwise path independent.
    Removed mean vorticity is the physical spatial average, not a modal scalar.
    Parameters are fixed-shape arrays with a leading member axis; deterministic
    families expose their declared parameters as well.
    """

    __strict_contract__ = True
    velocity: Complex[_MemberDim, _VelocityDims]
    member_indices: UInt32[_MemberDim]
    parameters: dict[str, Array]
    removed_coefficient_energy: Float[_MemberDim]
    projection_removed_energy: Float[_MemberDim]
    removed_mean_vorticity: Float[_MemberDim]
    quadrature_residual: Float[_MemberDim]
    baseline_coefficient_energy: Float[_MemberDim]
    discretization: TensorSpectralDiscretization
    measure_id: str = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    coefficient_realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)


@final
class PeriodicFlowInitialMeasure(StrictModule):
    family: PeriodicFlowFamily

    @checked
    def __init__(self, family: PeriodicFlowFamily, /) -> None:
        self.family = family

    @checked
    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        member_chunk_size: int = 8,
        maximum_bytes: int = 2**30,
    ) -> PreparedPeriodicFlowInitialMeasure:
        return PreparedPeriodicFlowInitialMeasure(
            self.family,
            discretization,
            member_chunk_size=member_chunk_size,
            maximum_bytes=maximum_bytes,
        )


@final
class PreparedPeriodicFlowInitialMeasure(StrictModule):
    """Byte-admitted generation of canonical modal initial states.

    RNG addresses contain the scientific measure identity and member number,
    never resource caps, chunk sizes, or quadrature/oversampling policies.
    Preparation and root fingerprinting are explicit host boundaries. Member
    generation is a stable jitted fixed-shape map without per-member host work.
    """

    family: PeriodicFlowFamily
    discretization: TensorSpectralDiscretization
    projector: PeriodicLerayProjector
    address: SampleAddress
    tables: tuple[Array, ...]
    oversampled: TensorSpectralDiscretization | None
    transfer: PreparedRetainedBandRestriction | None
    sheet_transform: PreparedNonuniformFourier | None
    power_law: PreparedPeriodicPowerLawField | None
    member_chunk_size: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    member_bytes: int = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    lengths: tuple[float, ...] = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    mollifier_normalization: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        family: PeriodicFlowFamily,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        member_chunk_size: int = 8,
        maximum_bytes: int = 2**30,
    ) -> None:
        chunk = positive_integer(member_chunk_size, "member_chunk_size")
        cap = positive_integer(maximum_bytes, "maximum_bytes")
        d = len(discretization.axes)
        if d not in (2, 3) or any(a.family != "fourier" for a in discretization.axes):
            raise ValueError("Initial measures require 2D or 3D periodic Fourier boxes.")
        lengths = tuple(float(a.length) for a in discretization.axes)
        volume = prod(lengths)
        dtype = np.dtype(discretization.plan.precision.coefficient_dtype)
        real_dtype = np.empty((), dtype=dtype).real.dtype
        work_size = 0
        match family:
            case TaylorGreenFamily():
                if family.spatial_dimension != d or any(l != lengths[0] for l in lengths):
                    raise ValueError(
                        "Taylor--Green requires matching dimension and isotropic lengths."
                    )
                if any(n < 3 for n in discretization.modal_shape):
                    raise ValueError("Taylor--Green requires retained modes ±1.")
            case PowerLawVelocityFamily():
                if family.plan.component_shape != (d,):
                    raise ValueError("Power-law component dimension must match the box.")
            case ShearLayerFamily():
                grid_shape = tuple(
                    n * family.oversampling for n in discretization.modal_shape
                )
                count = family.perturbation_mode_range[1]
                work_size = 8 * prod(grid_shape) + 4 * (grid_shape[0] + 1) * count
            case SinusoidalVortexSheetFamily():
                points = family.quadrature_points
                count = family.perturbation_count
                work_size = (
                    8 * points * (count + 1)
                    + 8 * count
                    + 4 * prod(discretization.modal_shape) * min(256, 2 * points)
                )
            case PiecewiseConstantVorticityFamily():
                work_size = 4 * prod(family.partition) + sum(
                    n * p
                    for n, p in zip(
                        discretization.modal_shape, family.partition, strict=True
                    )
                )
            case VortexPatchFamily():
                if family.radius >= min(lengths) / 2:
                    raise ValueError(
                        "Patch radius must be less than half the shortest box length."
                    )
            case _:
                assert_never(family)
        if not isinstance(family, (TaylorGreenFamily, PowerLawVelocityFamily)) and d != 2:
            raise ValueError("This initial family requires a 2D box.")
        member_bytes = (
            8 * prod(discretization.modal_shape) * d + work_size
        ) * dtype.itemsize + 64
        if chunk * member_bytes > cap:
            raise MemoryBudgetError("maximum_bytes", chunk * member_bytes, cap)
        projector = PeriodicLerayProjector(discretization)
        tables, oversampled, transfer, transform, power, normalization = _prepare_family(
            family, discretization, projector, chunk, cap, volume, real_dtype
        )
        measure_id = canonical_fingerprint(
            {
                "kind": "periodic-flow-initial-measure",
                "family": family.family_id,
                "generation": discretization.prepared_id,
            }
        )
        self.family, self.discretization, self.projector = (
            family,
            discretization,
            projector,
        )
        self.tables, self.oversampled, self.transfer = tables, oversampled, transfer
        self.sheet_transform, self.power_law = transform, power
        self.address = SampleAddress(
            "periodic-flow-initial-measure", "parameters", target=measure_id
        )
        self.member_chunk_size, self.maximum_bytes, self.member_bytes = (
            chunk,
            cap,
            member_bytes,
        )
        self.volume, self.lengths, self.measure_id = volume, lengths, measure_id
        self.mollifier_normalization = normalization
        self.prepared_id = canonical_fingerprint(
            {
                "measure": measure_id,
                "chunk": chunk,
                "cap": cap,
                "quadrature": family.quadrature_points
                if isinstance(family, SinusoidalVortexSheetFamily)
                else None,
                "oversampling": family.oversampling
                if isinstance(family, ShearLayerFamily)
                else None,
            }
        )

    @checked
    def realize(
        self, key: PRNGKey, member_indices: ArrayLike, /
    ) -> PeriodicFlowInitialState:
        if str(jr.key_impl(key)) != "threefry2x32":
            raise ValueError(
                "Member-independent initial measures require threefry2x32 typed keys."
            )
        indices = jnp.asarray(member_indices)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("member_indices must be nonempty rank one.")
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("member_indices must be integer.")
        host = np.asarray(indices)
        if np.any(host < 0) or np.any(host >= 2**32):
            raise ValueError("member_indices must lie in [0,2**32).")
        count = (
            (indices.size + self.member_chunk_size - 1)
            // self.member_chunk_size
            * self.member_chunk_size
        )
        if count * self.member_bytes > self.maximum_bytes:
            raise MemoryBudgetError(
                "maximum_bytes", count * self.member_bytes, self.maximum_bytes
            )
        indices = indices.astype(jnp.uint32)
        root_id = canonical_fingerprint(
            {
                "kind": "periodic-flow-root-key",
                "implementation": str(jr.key_impl(key)),
                "words": np.asarray(jr.key_data(key)).tolist(),
            }
        )
        if self.power_law is not None:
            realization = self.power_law.sample(derive_key(key, self.address), indices)
            velocity = realization.coefficients
            projection = realization.removed_coefficient_energy
            tail = jnp.zeros_like(projection)
            mean = tail
            residual = tail
            parameters: dict[str, Array] = {}
        else:
            velocity, parameters, tail, projection, mean, residual = _realize_members(
                self, key, jnp.pad(indices, (0, count - indices.size))
            )
            velocity = velocity[: indices.size]
            parameters = jax.tree.map(lambda a: a[: indices.size], parameters)
            tail, projection, mean, residual = (
                a[: indices.size] for a in (tail, projection, mean, residual)
            )
        baseline = _energy(velocity) + tail
        return PeriodicFlowInitialState(
            velocity=velocity,
            member_indices=indices,
            parameters=parameters,
            removed_coefficient_energy=tail,
            projection_removed_energy=projection,
            removed_mean_vorticity=mean,
            quadrature_residual=residual,
            baseline_coefficient_energy=baseline,
            discretization=self.discretization,
            measure_id=self.measure_id,
            family_id=self.family.family_id,
            discretization_id=self.discretization.prepared_id,
            coefficient_realization_id=root_id,
            coupling_id=canonical_fingerprint(
                {"measure": self.measure_id, "root": root_id}
            ),
        )


@checked
def restrict_periodic_flow_initial_state(
    state: PeriodicFlowInitialState,
    target_discretization: TensorSpectralDiscretization,
    /,
    *,
    maximum_bytes: int = 2**30,
) -> PeriodicFlowInitialState:
    """Restrict any initial state, including planar embeddings, without relabeling.

    Pure retained-band gather preserves measure, coupling, and root identities.
    Only the discretization identity changes; no reprojection is performed.
    """
    cap = positive_integer(maximum_bytes, "maximum_bytes")
    required_bytes = state.velocity.size * state.velocity.dtype.itemsize * 2
    if required_bytes > cap:
        raise MemoryBudgetError("maximum_bytes", required_bytes, cap)
    restriction = RetainedBandRestrictionPlan(
        state.discretization, target_discretization
    ).prepare()
    velocity = jax.vmap(restriction)(state.velocity)
    removed = jnp.maximum(state.baseline_coefficient_energy - _energy(velocity), 0.0)
    return PeriodicFlowInitialState(
        velocity=velocity,
        member_indices=state.member_indices,
        parameters=state.parameters,
        removed_coefficient_energy=removed,
        projection_removed_energy=state.projection_removed_energy,
        removed_mean_vorticity=state.removed_mean_vorticity,
        quadrature_residual=state.quadrature_residual,
        baseline_coefficient_energy=state.baseline_coefficient_energy,
        discretization=target_discretization,
        measure_id=state.measure_id,
        family_id=state.family_id,
        discretization_id=target_discretization.prepared_id,
        coefficient_realization_id=state.coefficient_realization_id,
        coupling_id=state.coupling_id,
    )


def _energy(velocity: Array, /) -> Array:
    return jnp.sum(jnp.abs(velocity) ** 2, axis=tuple(range(1, velocity.ndim)))


def _mollifier_hankel(
    wavenumbers: np.ndarray, radius: float, maximum_bytes: int, /
) -> tuple[np.ndarray, float]:
    """Host Hankel quadrature of Poseidon Eq. (49), split at r=1/2.

    A fixed Gauss--Legendre rule resolves the largest |k|*rho oscillation.
    Its order is an execution accuracy policy, not a publication parameter.
    The vectorized rule is checked against independent adaptive quadrature in
    tests; production preparation never starts one integrator per wavenumber.
    """
    radius_ = finite_real_scalar(radius, "radius")
    cap = positive_integer(maximum_bytes, "maximum_bytes")
    maximum_wave = finite_real_scalar(np.max(wavenumbers), "maximum_wavenumber")
    # Exact integer ratios avoid overflow before the finite byte-admission check.
    wave_num, wave_den = maximum_wave.as_integer_ratio()
    radius_num, radius_den = radius_.as_integer_ratio()
    numerator, denominator = wave_num * radius_num, 2 * wave_den * radius_den
    order = max(16, -(-numerator // denominator) + 16)
    # leggauss also constructs an order-squared host eigenproblem.
    required_bytes = (4 * wavenumbers.size * order + 4 * order**2 + 12 * order) * 8
    if required_bytes > cap:
        raise MemoryBudgetError("maximum_bytes", required_bytes, cap)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    r = np.concatenate(((nodes + 1) / 4, (nodes + 3) / 4))
    weights = np.tile(weights / 4, 2)
    # Algebraically reduce the published truncated powers inside support.
    psi = 80 / (7 * np.pi) * np.where(r <= 0.5, 0.5 - 3 * r**2 + 3 * r**3, (1 - r) ** 3)
    radial_weights = 2 * np.pi * r * psi * weights
    normalization = float(np.sum(radial_weights))
    if abs(normalization - 1.0) > 1e-10:
        raise ValueError("Published radial mollifier failed normalization.")
    transform = j0(radius_ * wavenumbers[:, None] * r[None, :]) @ radial_weights
    return transform, normalization


def _prepare_family(
    family: PeriodicFlowFamily,
    space: TensorSpectralDiscretization,
    projector: PeriodicLerayProjector,
    chunk: int,
    cap: int,
    volume: float,
    real_dtype: np.dtype,
    /,
) -> tuple[
    tuple[Array, ...],
    TensorSpectralDiscretization | None,
    PreparedRetainedBandRestriction | None,
    PreparedNonuniformFourier | None,
    PreparedPeriodicPowerLawField | None,
    float,
]:
    tables: tuple[Array, ...] = ()
    oversampled = None
    transfer = None
    transform = None
    power = None
    normalization = 1.0
    match family:
        case TaylorGreenFamily():
            modes = np.meshgrid(
                *(np.asarray(a.modes.mode_numbers) for a in space.axes), indexing="ij"
            )
            mask = np.logical_and.reduce(tuple(np.abs(n) == 1 for n in modes))
            c = np.zeros(
                space.modal_shape + (len(space.axes),),
                dtype=np.dtype(space.plan.precision.coefficient_dtype),
            )
            scale = family.amplitude * np.sqrt(volume) / 2 ** len(space.axes)
            c[..., 0] = np.where(mask, -1j * modes[0] * scale, 0.0)
            c[..., 1] = np.where(mask, 1j * modes[1] * scale, 0.0)
            tables = (jnp.asarray(c),)
        case VortexPatchFamily():
            magnitude = np.sqrt(np.asarray(projector.wavenumber_squared))
            safe = np.where(magnitude > 0, magnitude, 1.0)
            phase = sum(
                np.asarray(k) * (c - _periodic_lower(a.domain))
                for k, c, a in zip(
                    projector.wavenumbers, family.center, space.axes, strict=True
                )
            )
            omega = (
                family.strength
                * 2
                * np.pi
                * family.radius
                * j1(magnitude * family.radius)
                / safe
                * np.exp(-1j * phase)
                / np.sqrt(volume)
            )
            omega = np.where(magnitude > 0, omega, 0.0)
            tables = (jnp.asarray(omega, dtype=space.plan.precision.coefficient_dtype),)
        case PiecewiseConstantVorticityFamily():
            matrices = []
            for a, p in zip(space.axes, family.partition, strict=True):
                n = np.asarray(a.modes.mode_numbers)
                width = float(a.length) / p
                centers = (np.arange(p, dtype=np.float64) + 0.5) / p
                matrices.append(
                    jnp.asarray(
                        width
                        * np.sinc(n[:, None] / p)
                        * np.exp(-2j * np.pi * n[:, None] * centers),
                        dtype=space.plan.precision.coefficient_dtype,
                    )
                )
            tables = tuple(matrices)
        case ShearLayerFamily():
            oversampled = TensorSpectralPlan(
                tuple(
                    FourierBasisPlan(n * family.oversampling) for n in space.modal_shape
                ),
                axis_names=space.plan.axis_names,
                field_name=space.plan.field_name,
                precision=space.plan.precision,
            ).prepare(tuple(a.domain for a in space.axes))
            transfer = RetainedBandRestrictionPlan(oversampled, space).prepare()
            tables = tuple(
                jnp.arange(n, dtype=real_dtype) / n for n in oversampled.physical_shape
            )
        case SinusoidalVortexSheetFamily():
            magnitude = np.sqrt(np.asarray(projector.wavenumber_squared))
            unique, inverse = np.unique(magnitude, return_inverse=True)
            weights, normalization = _mollifier_hankel(
                unique, family.mollifier_radius, cap
            )
            weights = weights.astype(real_dtype)
            tables = (jnp.asarray(weights[inverse].reshape(space.modal_shape)),)
            transform = PreparedNonuniformFourier(
                NonuniformFourierPlan(
                    space.modal_shape,
                    1,
                    sign=-1,
                    route="chunked",
                    chunk_size=min(256, 2 * family.quadrature_points),
                ),
                dtype=real_dtype,
            )
        case PowerLawVelocityFamily():
            power = family.plan.prepare(
                space, member_chunk_size=chunk, maximum_sample_bytes=cap
            )
        case _:
            assert_never(family)
    return tables, oversampled, transfer, transform, power, normalization


@eqx.filter_jit
def _realize_members(
    prepared: PreparedPeriodicFlowInitialMeasure, key: PRNGKey, indices: Array, /
) -> tuple[Array, dict[str, Array], Array, Array, Array, Array]:
    def one(member: Array) -> tuple[Array, dict[str, Array], Array, Array, Array, Array]:
        return _realize_one(prepared, derive_key(key, prepared.address, member))

    return jax.lax.map(one, indices, batch_size=prepared.member_chunk_size)


def _uniform(
    key: PRNGKey, shape: tuple[int, ...], bounds: tuple[float, float], dtype: jnp.dtype, /
) -> Array:
    return jr.uniform(key, shape, dtype=dtype, minval=bounds[0], maxval=bounds[1])


def _realize_one(
    p: PreparedPeriodicFlowInitialMeasure, key: PRNGKey, /
) -> tuple[Array, dict[str, Array], Array, Array, Array, Array]:
    dtype = p.projector.wavenumber_squared.dtype
    zero = jnp.zeros((), dtype=dtype)
    keys = jr.split(key, 5)
    match p.family:
        case TaylorGreenFamily() as f:
            return (
                p.tables[0],
                {"amplitude": jnp.asarray(f.amplitude, dtype=dtype)},
                zero,
                zero,
                zero,
                zero,
            )
        case VortexPatchFamily() as f:
            omega = p.tables[0]
            mean = jnp.asarray(f.strength * np.pi * f.radius**2 / p.volume, dtype=dtype)
            parameters = {
                "radius": jnp.asarray(f.radius, dtype=dtype),
                "center": jnp.asarray(f.center, dtype=dtype),
                "strength": jnp.asarray(f.strength, dtype=dtype),
            }
        case PiecewiseConstantVorticityFamily() as f:
            values = _uniform(keys[0], f.partition, f.value_range, dtype)
            omega = contract("ai,bj,ij->ab", *p.tables, values) / np.sqrt(p.volume)
            mean = jnp.mean(values)
            parameters = {"cell_values": values}
        case ShearLayerFamily() as f:
            return _shear_member(p, f, keys)
        case SinusoidalVortexSheetFamily() as f:
            alpha = _uniform(
                keys[0], (f.perturbation_count,), f.perturbation_amplitude_range, dtype
            )
            beta = _uniform(keys[1], (f.perturbation_count,), (0.0, 1.0), dtype)
            coarse, mean = _sheet_coefficients(p, f, alpha, beta, f.quadrature_points)
            omega, _ = _sheet_coefficients(p, f, alpha, beta, 2 * f.quadrature_points)
            # The delivered field uses Q, while 2Q supplies independent evidence.
            residual = jnp.sqrt(jnp.sum(jnp.abs(omega - coarse) ** 2))
            velocity = p.projector.velocity_from_vorticity(
                coarse, tolerance=100 * float(jnp.finfo(dtype).eps)
            )
            return velocity, {"alpha": alpha, "beta": beta}, zero, zero, mean, residual
        case PowerLawVelocityFamily():
            raise ValueError(
                "Power-law sampling is owned by PreparedPeriodicPowerLawField."
            )
        case _:
            assert_never(p.family)
    omega = jnp.where(p.projector.wavenumber_squared > 0, omega, jnp.zeros_like(omega))
    velocity = p.projector.velocity_from_vorticity(
        omega, tolerance=100 * float(jnp.finfo(dtype).eps)
    )
    return velocity, parameters, zero, zero, mean, zero


def _shear_member(
    p: PreparedPeriodicFlowInitialMeasure, f: ShearLayerFamily, keys: Array, /
) -> tuple[Array, dict[str, Array], Array, Array, Array, Array]:
    if p.oversampled is None or p.transfer is None:
        raise ValueError("Shear preparation is missing its oversampled transform.")
    dtype = p.projector.wavenumber_squared.dtype
    zero = jnp.zeros((), dtype=dtype)
    q = jr.randint(
        keys[0],
        (),
        f.perturbation_mode_range[0],
        f.perturbation_mode_range[1] + 1,
        dtype=jnp.int32,
    )
    alpha = _uniform(keys[1], (f.perturbation_mode_range[1],), (0.0, 1.0), dtype)
    beta = _uniform(keys[2], alpha.shape, (0.0, 2 * np.pi), dtype)
    xi = _uniform(keys[3], (), f.interface_shift_range, dtype)
    modes = jnp.arange(1, alpha.size + 1, dtype=dtype)
    x, y = p.tables
    sigma = xi + f.perturbation_amplitude * contract(
        "k,xk->x",
        jnp.where(modes <= q, alpha, 0.0),
        jnp.sin(2 * jnp.pi * x[:, None] * modes - beta),
    )
    rho = (
        zero
        if f.thickness_range is None
        else _uniform(keys[4], (), f.thickness_range, dtype)
    )
    horizontal = _shear_profile(y[None, :] + sigma[:, None], rho, f.profile)
    raw = p.oversampled.project(
        jnp.stack((horizontal, jnp.zeros_like(horizontal)), axis=-1)
    )
    retained = p.transfer(raw)
    velocity = p.projector.project(retained)
    tail = jnp.maximum(jnp.sum(jnp.abs(raw) ** 2) - jnp.sum(jnp.abs(retained) ** 2), 0.0)
    projection = jnp.sum(jnp.abs(retained - velocity) ** 2)
    return (
        velocity,
        {
            "mode_count": q,
            "alpha": alpha,
            "beta": beta,
            "interface_shift": xi,
            "thickness": rho,
        },
        tail,
        projection,
        zero,
        zero,
    )


def _shear_profile(
    displaced_coordinate: Array, thickness: Array, profile: ShearLayerProfile, /
) -> Array:
    """Evaluate the normalized layer's periodic extension before branching."""
    shifted = jnp.mod(displaced_coordinate, 1.0)
    match profile:
        case "smooth":
            return jnp.where(
                shifted <= 0.5,
                jnp.tanh((shifted - 0.25) / thickness),
                jnp.tanh((0.75 - shifted) / thickness),
            )
        case "discontinuous":
            return jnp.where((shifted > 0.25) & (shifted <= 0.75), 1.0, -1.0)
        case _:
            assert_never(profile)


def _sheet_coefficients(
    p: PreparedPeriodicFlowInitialMeasure,
    f: SinusoidalVortexSheetFamily,
    alpha: Array,
    beta: Array,
    points: int,
    /,
) -> tuple[Array, Array]:
    if p.sheet_transform is None:
        raise ValueError("Sheet preparation is missing its NUFFT.")
    x = jnp.arange(points, dtype=alpha.dtype) / points
    angles = 2 * jnp.pi * (x[:, None] + beta)
    graph = (
        0.5
        + f.base_amplitude * jnp.sin(2 * jnp.pi * x)
        + contract("qi,i->q", jnp.sin(angles), alpha)
    )
    slope = (
        2
        * jnp.pi
        * (
            f.base_amplitude * jnp.cos(2 * jnp.pi * x)
            + contract("qi,i->q", jnp.cos(angles), alpha)
        )
    )
    weights = jnp.sqrt(p.lengths[0] ** 2 + (p.lengths[1] * slope) ** 2) / points
    coefficients = (
        p.sheet_transform.type1(
            2 * jnp.pi * jnp.stack((x, graph), axis=-1),
            weights.astype(p.discretization.plan.precision.coefficient_dtype),
        )
        * p.tables[0]
        / np.sqrt(p.volume)
    )
    coefficients = jnp.where(
        p.projector.wavenumber_squared > 0, coefficients, jnp.zeros_like(coefficients)
    )
    return coefficients, jnp.sum(weights) / p.volume


@checked
def embed_planar_velocity(
    state: PeriodicFlowInitialState,
    discretization_3d: TensorSpectralDiscretization,
    /,
    *,
    invariant_axis: int,
) -> PeriodicFlowInitialState:
    """Embed c² in n_invariant=0 with c³=sqrt(L_invariant)c².

    Integrated 3D energy is L_invariant times 2D energy; energy per unit
    invariant length and pointwise planar dynamics at t=0 are preserved.
    Measure/coupling identity is independent of target resolution and includes
    the invariant axis, its domain/length, and the orthonormal zero-mode policy.
    Use ``restrict_periodic_flow_initial_state`` for coupled embedded coarsening.
    """
    axis = nonnegative_integer(invariant_axis, "invariant_axis")
    if axis not in (0, 1, 2):
        raise ValueError("invariant_axis must be 0, 1, or 2.")
    if (
        len(state.discretization.axes) != 2
        or state.velocity.shape[-1] != 2
        or len(discretization_3d.axes) != 3
    ):
        raise ValueError("Embedding requires a planar state and a 3D target.")
    PeriodicLerayProjector(discretization_3d)
    planar = tuple(i for i in range(3) if i != axis)
    for source, i in zip(state.discretization.axes, planar, strict=True):
        target = discretization_3d.axes[i]
        if (
            source.domain.domain_id != target.domain.domain_id
            or not np.array_equal(
                np.asarray(source.modes.mode_numbers),
                np.asarray(target.modes.mode_numbers),
            )
            or not np.array_equal(
                np.asarray(source.modes.nyquist_mask),
                np.asarray(target.modes.nyquist_mask),
            )
        ):
            raise ValueError(
                "Embedding requires identical planar axis domains and retained modes."
            )
    invariant = discretization_3d.axes[axis]
    zero_indices = np.flatnonzero(np.asarray(invariant.modes.mode_numbers) == 0)
    if zero_indices.size != 1:
        raise ValueError("Invariant axis must contain one zero mode.")
    length = float(invariant.length)
    velocity = jnp.zeros(
        (state.member_indices.size,) + discretization_3d.modal_shape + (3,),
        dtype=state.velocity.dtype,
    )
    for component, target_component in enumerate(planar):
        selector: list[int | slice] = [slice(None)] * velocity.ndim
        selector[axis + 1] = int(zero_indices[0])
        selector[-1] = target_component
        velocity = velocity.at[tuple(selector)].set(
            np.sqrt(length) * state.velocity[..., component]
        )
    measure_id = canonical_fingerprint(
        {
            "kind": "planar-initial-embedding",
            "measure": state.measure_id,
            "invariant_axis": axis,
            "invariant_domain": invariant.domain.domain_id,
            "invariant_length": length,
            "policy": "orthonormal-zero-mode",
        }
    )
    return PeriodicFlowInitialState(
        velocity=velocity,
        member_indices=state.member_indices,
        parameters=state.parameters,
        removed_coefficient_energy=length * state.removed_coefficient_energy,
        projection_removed_energy=length * state.projection_removed_energy,
        removed_mean_vorticity=state.removed_mean_vorticity,
        quadrature_residual=np.sqrt(length) * state.quadrature_residual,
        baseline_coefficient_energy=length * state.baseline_coefficient_energy,
        discretization=discretization_3d,
        measure_id=measure_id,
        family_id=state.family_id,
        discretization_id=discretization_3d.prepared_id,
        coefficient_realization_id=state.coefficient_realization_id,
        coupling_id=canonical_fingerprint(
            {"measure": measure_id, "root": state.coefficient_realization_id}
        ),
    )


__all__ = [
    "TaylorGreenFamily",
    "ShearLayerProfile",
    "ShearLayerFamily",
    "VortexPatchFamily",
    "PiecewiseConstantVorticityFamily",
    "SinusoidalVortexSheetFamily",
    "PowerLawVelocityFamily",
    "PeriodicFlowFamily",
    "PeriodicFlowInitialMeasure",
    "PreparedPeriodicFlowInitialMeasure",
    "PeriodicFlowInitialState",
    "embed_planar_velocity",
    "restrict_periodic_flow_initial_state",
]
