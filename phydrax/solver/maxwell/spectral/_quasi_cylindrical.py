#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quasi-cylindrical PSATD/Galilean spectral Maxwell field solver for PIC.

Fields are azimuthal Fourier series ``F = Re Σ_{m=0}^{M} F_m(r, z) e^{imθ}`` on
the cell-centered radial grid of `SharedGridHankelPlan` and a periodic axial
grid (`QuasiCylindricalGrid`). The circular components
``F_± = (F_r ∓ iF_θ)/2`` of mode ``m`` are Hankel-transformed with orders
``m ∓ 1`` and ``F_z`` with order ``m`` on one shared k-grid per mode; the axial
direction is Fourier-transformed. On that spectrum the combinations

    F̃_x = i(F̂_+ − F̂_−),   F̃_y = −(F̂_+ + F̂_−),   F̃_z = F̂_z

are the Cartesian spectrum at the wavevector ``k̃ = (k_⊥, 0, k_z)`` (Lehe et
al., Comput. Phys. Commun. 203, 66, 2016), so every mode ``(m, k_⊥, k_z)`` is
advanced by the Cartesian P2 propagators of `._galilean` (standard, Galilean,
and averaged-Galilean PSATD with ``κ = k_z v_gal``) with the derivative symbol
``ik̃``; vacuum dispersion ``ω = c|k̃|`` is exact.

Particles carry Cartesian positions; `PreparedAzimuthalTransfer` deposits the
modal charge and the step-mean current ``q v`` at the path midpoint with
``e^{−imθ}`` weights (``×2`` for ``m > 0``) and the near-axis volume correction.
The solver's Gauss charge is the spectral charge ``ρ̂[m, k_⊥, k_z]``; charge
conservation is spectral (``"update-with-rho"`` or ``"spectral-correction"``),
so only the transverse part of the deposited current is used and the Gauss law
holds to roundoff. Radial content outside the synthesis range of order ``m ≥ 1``
(one radial direction per mode, reported by the Hankel evidence) is projected
out by the pseudoinverse analysis.

Capabilities: `PICSpectralSymbol`, `PICHuygensSampling` (closed cylinders,
standard variant, no antennas), `PICWindowShift` along ``z`` (followed by a
spectral Gauss/solenoidal projection, since the spectral divergence is
nonlocal), `PICGalileanGrid`, `PICRestartState`, and `PICGaussProjection`.
Optional radial damping absorbs the transverse fields in an outer layer
without touching their divergence. Sheet antennas (`QuasiCylindricalAntennaPlan`)
inject per-mode one-way currents ``K = s ẑ × H'``, ``K_m = −s ẑ × E'`` whose
declared electric and magnetic sheet charges are tracked in the field state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._dtype_names import RealPrecisionDType
from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._interpolation import (
    cubic_hermite_interpolate,
    linear_interpolate,
    local_cubic_slopes,
)
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.pic import (
    PICSpeciesPlan,
    PreparedAzimuthalTransfer,
    QuasiCylindricalGrid,
)
from ....discretization.spectral import PreparedSharedGridHankel, SharedGridHankelPlan
from ....linalg import FFTLinearTransform
from ....typing import parse
from ..._maxwell_antenna import AntennaEmissionDirection
from ..._maxwell_far_field import (
    _gather_map,
    _require_exterior,
    _require_huygens_acquisition,
    _SurfaceGeometry,
    HomogeneousMaxwellExterior,
    HuygensSurfacePhasors,
)
from ..._maxwell_observers import DFTObserverState, MaxwellSpectralAcquisition
from ..._pic_field_solver import (
    AbstractPreparedPICFieldSolver,
    PICFieldAdvance,
    PICFieldDeposit,
    PICGaussProjectionResult,
    PICRestartComponent,
    restart_component,
    restore_component,
)
from ._galilean import (
    apply_function,
    exact_interval,
    galilean_charge_current,
    gauss_following_increment,
    gauss_following_window,
    phi_functions,
    SpectralOperators,
    window_integral,
)
from ._psatd import (
    PreparedSpectralHuygensBox,
    SpectralChargeConservation,
    SpectralMaxwellVariant,
)


QuasiCylindricalAbsorber: TypeAlias = Literal["none", "radial-damping"]

_CIRCULAR_OFFSETS = (-1, 1, 0)
_UNRESOLVED_TOLERANCE = 1.0e-10


# -- declared plans -------------------------------------------------------------------


class RadialDampingPlan(StrictModule, NonTrainableState):
    """Graded damping of the transverse fields in the outer radial layer.

    ``σ(r) = σ_max ((r − r₀)/L)^p`` over the last ``thickness`` cells, with
    ``σ_max = −(p + 1) c ln(attenuation)/(2L)`` so that a wave crossing the layer
    and back is attenuated by ``attenuation`` in amplitude. The step multiplies
    the transverse (solenoidal) ``E`` and ``B`` by ``exp(−σΔt)`` in real space and
    re-projects the change onto solenoidal fields, so the Gauss laws are
    unchanged.
    """

    thickness: int = eqx.field(static=True)
    attenuation: float = eqx.field(static=True)
    profile_power: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thickness: int,
        /,
        *,
        attenuation: float = 1.0e-4,
        profile_power: float = 3.0,
    ) -> None:
        cells = int(thickness)
        target = float(attenuation)
        power = float(profile_power)
        if cells < 1:
            raise ValueError("Radial damping thickness must be at least one cell.")
        if not 0.0 < target < 1.0:
            raise ValueError("Radial damping attenuation must lie in (0, 1).")
        if not math.isfinite(power) or power < 1.0:
            raise ValueError("Radial damping profile power must be at least one.")
        self.thickness = cells
        self.attenuation = target
        self.profile_power = power
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quasi-cylindrical-radial-damping",
                "thickness": cells,
                "attenuation": target,
                "profile_power": power,
            }
        )

    def rate(self, grid: QuasiCylindricalGrid, speed: float, /) -> np.ndarray:
        """Host damping rate ``σ[N_r]`` at the radial nodes."""
        width = self.thickness * grid.radial_spacing
        start = grid.radius - width
        maximum = -(self.profile_power + 1.0) * speed * math.log(self.attenuation)
        maximum /= 2.0 * width
        depth = np.clip((grid.radial_coordinates - start) / width, 0.0, None)
        return maximum * depth**self.profile_power


class QuasiCylindricalAntennaPlan(StrictModule, NonTrainableState):
    """Stationary one-way sheet antenna on the plane ``z = plane_coordinate``.

    ``electric[x, y, t, (E_x, E_y)]`` is the complex envelope ``A`` of the
    tangential forward-wave field at the sheet on the tensor sample grid
    ``first_coordinates × second_coordinates`` (Cartesian ``x``, ``y``) and
    rest-frame ``times``; the physical field is ``Re[A e^{−iω₀t}]`` and samples
    outside the window are zero. ``magnetic`` holds ``(H_x, H_y)`` and defaults to
    the plane-wave relation ``H' = s ẑ × E'/η``. The same conventions as the
    Cartesian `SampledPlaneCurrentAntennaPlan`; the optics adapter
    ``pulse_envelope_quasi_cylindrical_antenna`` builds it from a
    `PulseEnvelopeField`.
    """

    first_coordinates: Array
    second_coordinates: Array
    times: Array
    electric: Array
    magnetic: Array
    plane_coordinate: float = eqx.field(static=True)
    carrier_angular_frequency: float = eqx.field(static=True)
    direction: AntennaEmissionDirection = eqx.field(static=True)
    medium: HomogeneousMaxwellExterior
    provenance_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plane_coordinate: float,
        first_coordinates: ArrayLike,
        second_coordinates: ArrayLike,
        times: ArrayLike,
        electric: ArrayLike,
        /,
        *,
        magnetic: ArrayLike | None = None,
        carrier_angular_frequency: float = 0.0,
        direction: AntennaEmissionDirection = "positive",
        medium: HomogeneousMaxwellExterior | None = None,
        provenance_id: str | None = None,
    ) -> None:
        position = float(plane_coordinate)
        carrier = float(carrier_angular_frequency)
        if not math.isfinite(position):
            raise ValueError("plane_coordinate must be finite.")
        if not math.isfinite(carrier) or carrier < 0.0:
            raise ValueError("carrier_angular_frequency must be finite and nonnegative.")
        emission = parse(direction, AntennaEmissionDirection, "direction")
        medium_ = HomogeneousMaxwellExterior() if medium is None else medium
        _require_exterior(medium_)
        axes = []
        for name, values in (
            ("first_coordinates", first_coordinates),
            ("second_coordinates", second_coordinates),
            ("times", times),
        ):
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or array.size < 2 or not np.all(np.diff(array) > 0.0):
                raise ValueError(f"{name} must be strictly increasing with >= 2 values.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite.")
            axes.append(array)
        shape = (axes[0].size, axes[1].size, axes[2].size, 2)
        field = np.asarray(electric, dtype=np.complex128)
        if field.shape != shape or not np.all(np.isfinite(field)):
            raise ValueError(f"electric must be finite with shape {shape}.")
        sign = 1.0 if emission == "positive" else -1.0
        if magnetic is None:
            response = (
                sign
                * np.stack((-field[..., 1], field[..., 0]), axis=-1)
                / medium_.impedance
            )
        else:
            response = np.asarray(magnetic, dtype=np.complex128)
            if response.shape != shape or not np.all(np.isfinite(response)):
                raise ValueError(f"magnetic must be finite with shape {shape}.")
        if provenance_id is not None and not str(provenance_id):
            raise ValueError("provenance_id must be nonempty when supplied.")
        self.first_coordinates = jnp.asarray(axes[0])
        self.second_coordinates = jnp.asarray(axes[1])
        self.times = jnp.asarray(axes[2])
        self.electric = jnp.asarray(field)
        self.magnetic = jnp.asarray(response)
        self.plane_coordinate = position
        self.carrier_angular_frequency = carrier
        self.direction = emission
        self.medium = medium_
        self.provenance_id = None if provenance_id is None else str(provenance_id)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quasi-cylindrical-antenna-plan",
                "plane_coordinate": position.hex(),
                "carrier_angular_frequency": carrier.hex(),
                "direction": emission,
                "medium": medium_.exterior_id,
                "provenance": self.provenance_id,
                "samples": array_tree_fingerprint(
                    (axes[0], axes[1], axes[2], field, response)
                ),
            }
        )

    @property
    def emission_sign(self) -> float:
        match self.direction:
            case "positive":
                return 1.0
            case "negative":
                return -1.0


class QuasiCylindricalHuygensPlan(StrictModule, NonTrainableState):
    """Closed Huygens cylinder of the quasi-cylindrical grid.

    The side wall lies at ``r = radial_face·Δr`` (midway between radial nodes
    ``radial_face − 1`` and ``radial_face``) from axial node ``lower`` to
    ``upper``; the end caps lie on those node planes. Each patch is sampled on
    ``azimuthal_samples`` uniform angles; fields are the azimuthal syntheses of
    the modes, taken at the cap nodes and interpolated to the side-wall cell
    centers by the four-point (fourth-order) midpoint rule in ``r`` and ``z``.
    Patches are labeled 0/1 (lower/upper cap) and 2 (side).
    """

    radial_face: int = eqx.field(static=True)
    lower: int = eqx.field(static=True)
    upper: int = eqx.field(static=True)
    azimuthal_samples: int = eqx.field(static=True)
    acquisition: MaxwellSpectralAcquisition
    exterior: HomogeneousMaxwellExterior
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radial_face: int,
        lower: int,
        upper: int,
        acquisition: MaxwellSpectralAcquisition,
        exterior: HomogeneousMaxwellExterior,
        /,
        *,
        azimuthal_samples: int = 32,
    ) -> None:
        face, low, high, samples = (
            int(radial_face),
            int(lower),
            int(upper),
            int(azimuthal_samples),
        )
        if face < 2 or low < 0 or high <= low or samples < 4:
            raise ValueError(
                "Huygens cylinder needs radial_face >= 2, 0 <= lower < upper, and at "
                "least four azimuthal samples."
            )
        self.radial_face = face
        self.lower = low
        self.upper = high
        self.azimuthal_samples = samples
        self.acquisition = _require_huygens_acquisition(acquisition)
        self.exterior = _require_exterior(exterior)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quasi-cylindrical-huygens-cylinder",
                "radial_face": face,
                "lower": low,
                "upper": high,
                "azimuthal_samples": samples,
                "acquisition": self.acquisition.acquisition_id,
                "exterior": self.exterior.exterior_id,
            }
        )


class QuasiCylindricalMaxwellPlan(StrictModule, NonTrainableState):
    """Declared quasi-cylindrical PSATD configuration.

    ``variant`` and ``charge_conservation`` use the Cartesian P2 vocabulary;
    ``"vay-deposition"`` is a Cartesian-stencil scheme and is refused. The
    time dependency is constant-J. ``galilean_velocity`` is the axial grid
    velocity of the Galilean variants.
    """

    grid: QuasiCylindricalGrid
    variant: SpectralMaxwellVariant = eqx.field(static=True)
    charge_conservation: SpectralChargeConservation = eqx.field(static=True)
    absorber: QuasiCylindricalAbsorber = eqx.field(static=True)
    damping: RadialDampingPlan | None
    galilean_velocity: float = eqx.field(static=True)
    antennas: tuple[QuasiCylindricalAntennaPlan, ...]
    observers: tuple[QuasiCylindricalHuygensPlan, ...]
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: QuasiCylindricalGrid,
        /,
        *,
        variant: SpectralMaxwellVariant = "standard",
        charge_conservation: SpectralChargeConservation = "update-with-rho",
        absorber: QuasiCylindricalAbsorber = "none",
        damping: RadialDampingPlan | None = None,
        galilean_velocity: float | None = None,
        antennas: Sequence[QuasiCylindricalAntennaPlan] = (),
        observers: Sequence[QuasiCylindricalHuygensPlan] = (),
        permittivity: float = 1.0,
        permeability: float = 1.0,
    ) -> None:
        if not isinstance(grid, QuasiCylindricalGrid):
            raise TypeError("grid must be a QuasiCylindricalGrid.")
        variant = parse(variant, SpectralMaxwellVariant, "variant")
        charge_conservation = parse(
            charge_conservation, SpectralChargeConservation, "charge_conservation"
        )
        absorber = parse(absorber, QuasiCylindricalAbsorber, "absorber")
        epsilon, mu = float(permittivity), float(permeability)
        if not (
            math.isfinite(epsilon) and epsilon > 0.0 and math.isfinite(mu) and mu > 0.0
        ):
            raise ValueError("permittivity and permeability must be finite and positive.")
        speed = 1.0 / math.sqrt(epsilon * mu)
        match charge_conservation:
            case "spectral-correction" | "update-with-rho":
                pass
            case "vay-deposition":
                raise ValueError(
                    "vay-deposition is a Cartesian-stencil deposition; quasi-"
                    "cylindrical PSATD uses update-with-rho or spectral-correction."
                )
            case _:
                raise ValueError("charge_conservation is invalid.")
        velocity = _axial_velocity(variant, galilean_velocity, speed)
        antenna_values = tuple(antennas)
        if any(
            not isinstance(value, QuasiCylindricalAntennaPlan) for value in antenna_values
        ):
            raise TypeError("antennas must be QuasiCylindricalAntennaPlan instances.")
        if any(
            value.medium.permittivity != epsilon or value.medium.permeability != mu
            for value in antenna_values
        ):
            raise ValueError("Antenna media must be the solver's vacuum medium.")
        observer_values = tuple(observers)
        if any(
            not isinstance(value, QuasiCylindricalHuygensPlan)
            for value in observer_values
        ):
            raise TypeError("observers must be QuasiCylindricalHuygensPlan instances.")
        match absorber:
            case "none":
                if damping is not None:
                    raise ValueError("A damping plan requires absorber='radial-damping'.")
                layer = 0
            case "radial-damping":
                if not isinstance(damping, RadialDampingPlan):
                    raise TypeError(
                        "absorber='radial-damping' requires a RadialDampingPlan."
                    )
                if damping.thickness >= grid.radial_count:
                    raise ValueError("The radial damping layer leaves no interior.")
                layer = damping.thickness
            case _:
                raise ValueError("absorber is invalid.")
        if variant == "averaged-galilean" and (absorber != "none" or antenna_values):
            raise ValueError(
                "averaged-galilean gathers time-averaged fields; radial damping and "
                "antennas are refused with it."
            )
        if observer_values:
            if variant != "standard":
                raise ValueError(
                    "Huygens sampling requires the standard (lab-frame) variant."
                )
            if antenna_values:
                raise ValueError(
                    "A spectral antenna sheet has current at every axial node, so a "
                    "Huygens surface cannot be current-free; Huygens sampling with "
                    "antennas is refused."
                )
        for box in observer_values:
            if box.radial_face > grid.radial_count - layer - 2:
                raise ValueError("Huygens cylinders must lie inside the damping layer.")
            if box.upper >= grid.axial_count:
                raise ValueError("Huygens cylinder caps must be axial grid nodes.")
            if box.exterior.permittivity != epsilon or box.exterior.permeability != mu:
                raise ValueError("Huygens exterior must be the solver's vacuum medium.")
        self.grid = grid
        self.variant = variant
        self.charge_conservation = charge_conservation
        self.absorber = absorber
        self.damping = damping
        self.galilean_velocity = velocity
        self.antennas = antenna_values
        self.observers = observer_values
        self.permittivity = epsilon
        self.permeability = mu
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quasi-cylindrical-maxwell-plan",
                "grid": grid.grid_id,
                "variant": variant,
                "charge_conservation": charge_conservation,
                "absorber": absorber,
                "damping": None if damping is None else damping.plan_id,
                "galilean_velocity": velocity,
                "antennas": [value.plan_id for value in antenna_values],
                "observers": [value.plan_id for value in observer_values],
                "permittivity": epsilon,
                "permeability": mu,
            }
        )

    @property
    def speed_of_light(self) -> float:
        return 1.0 / math.sqrt(self.permittivity * self.permeability)

    def prepare(
        self, transfers: Sequence[PreparedAzimuthalTransfer] = (), /
    ) -> PreparedQuasiCylindricalMaxwell:
        return PreparedQuasiCylindricalMaxwell(self, transfers)


def _axial_velocity(
    variant: SpectralMaxwellVariant, velocity: float | None, speed: float, /
) -> float:
    match variant:
        case "standard":
            if velocity is not None:
                raise ValueError("The standard variant takes no galilean_velocity.")
            return 0.0
        case "galilean" | "averaged-galilean":
            if velocity is None:
                raise ValueError("Galilean variants require galilean_velocity.")
            value = float(velocity)
            if not math.isfinite(value) or not 0.0 < abs(value) < speed:
                raise ValueError("galilean_velocity must be nonzero and subluminal.")
            return value
        case _:
            raise ValueError("variant is invalid.")


# -- state -------------------------------------------------------------------------


class QuasiCylindricalSource(StrictModule):
    """Deposited source of one step.

    ``current[m, j, k, (J_+, J_−, J_z)]`` is the step-mean modal current density
    in circular components on the real-space nodes; ``charge_change[m, j, k]``
    the real-space modal charge-density change over the step (analyzed onto
    the spectral Gauss charge by the solver). Both are additive over species.
    """

    current: Array
    charge_change: Array


class QuasiCylindricalMaxwellState(StrictModule):
    """Spectral fields and Gauss charges with their real-space modal samples.

    ``spectral_electric``/``spectral_magnetic`` are ``[M+1, N_k, N_z, 3]``
    Cartesian-equivalent spectra (``F̃_x, F̃_y, F̃_z``) and ``charge`` the
    spectral particle Gauss charge. ``electric``/``magnetic`` are the circular
    real-space modes ``[M+1, N_r, N_z, (F_+, F_−, F_z)]`` gathered by particles
    (``averaged_*`` for the averaged-Galilean variant). ``antenna_charge`` and
    ``antenna_magnetic_charge`` are the declared spectral sheet charges of the
    antennas (``None`` without antennas); ``window_offset`` is the cumulative
    moving-window translation along ``z``.
    """

    spectral_electric: Array
    spectral_magnetic: Array
    charge: Array
    electric: Array
    magnetic: Array
    averaged_electric: Array | None
    averaged_magnetic: Array | None
    antenna_charge: Array | None
    antenna_magnetic_charge: Array | None
    window_offset: Array
    observations: tuple[DFTObserverState, ...]


class QuasiCylindricalMaxwellDiagnostics(StrictModule):
    """Constraint, energy, absorber, and surface evidence of one field step.

    Constraints are the real-space max-norm residuals of ``∇·E − (ρ + ρ_a)/ε``
    and ``∇·B − ρ_m`` over every mode; ``absorbed_energy`` is removed by radial
    damping and ``surface_current`` is the largest deposited current sampled on
    a Huygens surface inside its window.
    """

    electric_constraint: Array
    magnetic_constraint: Array
    energy: Array
    absorbed_energy: Array
    surface_current: Array


# -- transforms ---------------------------------------------------------------------


class _ModalTransform(StrictModule, NonTrainableState):
    """Shared-grid Hankel plus mean-normalized axial Fourier transforms.

    Axial coefficients are ``f̂(k_z) = N_z⁻¹ Σ_k f(z_k) e^{−ik_z(z_k − z₀)}``
    (the native orthonormal FFT scaled by ``N_z^{−½}``), so a coefficient carries
    the amplitude of its axial harmonic.
    """

    hankel: PreparedSharedGridHankel
    axial: FFTLinearTransform
    scale: float = eqx.field(static=True)

    def _map(self, values: Array, inverse: bool, /) -> Array:
        moved = jnp.moveaxis(values.astype(jnp.complex128), 2, -1)
        lines = moved.reshape((-1, moved.shape[-1]))
        action = self.axial.synthesize if inverse else self.axial.analyze
        mapped = jax.vmap(action)(lines).reshape(moved.shape)
        factor = 1.0 / self.scale if inverse else self.scale
        return jnp.moveaxis(factor * mapped, -1, 2)

    def axial_forward(self, values: Array, /) -> Array:
        return self._map(values, False)

    def axial_inverse(self, coefficients: Array, /) -> Array:
        return self._map(coefficients, True)

    def scalar_forward(self, values: Array, /) -> Array:
        return self.hankel.forward(self.axial_forward(values), 0)

    def scalar_inverse(self, coefficients: Array, /) -> Array:
        return self.axial_inverse(self.hankel.inverse(coefficients, 0))

    def vector_forward(self, values: Array, /) -> Array:
        """Circular real-space modes ``[..., 3]`` → Cartesian-equivalent spectra."""
        axial = self.axial_forward(values)
        plus = self.hankel.forward(axial[..., 0], -1)
        minus = self.hankel.forward(axial[..., 1], 1)
        along = self.hankel.forward(axial[..., 2], 0)
        return jnp.stack((1j * (plus - minus), -(plus + minus), along), axis=-1)

    def vector_inverse(self, coefficients: Array, /) -> Array:
        first, second = coefficients[..., 0], coefficients[..., 1]
        plus = self.hankel.inverse(0.5 * (-1j * first - second), -1)
        minus = self.hankel.inverse(0.5 * (1j * first - second), 1)
        along = self.hankel.inverse(coefficients[..., 2], 0)
        return self.axial_inverse(jnp.stack((plus, minus, along), axis=-1))


def _axial_wavenumbers(grid: QuasiCylindricalGrid, /) -> np.ndarray:
    """Axial symbol ``k_z`` with the even-count Nyquist entry zeroed (real ``m = 0``)."""
    k = 2.0 * math.pi * np.fft.fftfreq(grid.axial_count, d=grid.axial_spacing)
    if grid.axial_count % 2 == 0:
        k[grid.axial_count // 2] = 0.0
    return k


def _operators(
    hankel: PreparedSharedGridHankel,
    grid: QuasiCylindricalGrid,
    velocity: float,
    speed: float,
    permittivity: float,
    /,
) -> SpectralOperators:
    radial = np.asarray(hankel.wavenumbers)[:, :, None]
    axial = _axial_wavenumbers(grid)[None, None, :]
    shape = (grid.mode_count, grid.radial_count, grid.axial_count)
    symbol = np.stack(
        (
            np.broadcast_to(radial, shape),
            np.zeros(shape),
            np.broadcast_to(axial, shape),
        ),
        axis=-1,
    )
    squared = np.sum(symbol**2, axis=-1)
    return SpectralOperators(
        plus=jnp.asarray(1j * symbol),
        minus=jnp.asarray(1j * symbol),
        squared=jnp.asarray(squared),
        advection=jnp.asarray(np.broadcast_to(axial * velocity, shape)),
        resolved=jnp.asarray(squared > 0.0),
        speed=speed,
        permittivity=permittivity,
    )


def _divergence(operators: SpectralOperators, value: Array, /) -> Array:
    return jnp.sum(operators.minus * value, axis=-1)


def _magnetic_charge_field(operators: SpectralOperators, charge: Array, /) -> Array:
    """Longitudinal ``B`` with ``ik̃·B = ρ_m`` on resolved modes."""
    return -operators.charge_current(charge)


# -- antennas ------------------------------------------------------------------------


class _PreparedAntenna(StrictModule, NonTrainableState):
    """Per-mode spectral sheet-current profiles of one antenna.

    ``electric[t, part, m, n, 3]``/``magnetic[...]`` are the Cartesian-equivalent
    spectra of ``K`` and ``K_m`` for the envelope (``part`` 0) and its conjugate
    (``part`` 1); the physical sheet current at time ``t`` is
    ``½(K₀(t)e^{−iω₀t} + K₁(t)e^{iω₀t})``.
    """

    times: Array
    electric: Array
    magnetic: Array
    electric_slopes: Array
    magnetic_slopes: Array
    plane_coordinate: float = eqx.field(static=True)
    carrier_angular_frequency: float = eqx.field(static=True)

    def sheets(self, time: Array, /) -> tuple[Array, Array]:
        """Spectral electric and magnetic sheet currents ``[M+1, N_k, 3]``."""
        carrier = jnp.exp(-1j * self.carrier_angular_frequency * time)
        electric = cubic_hermite_interpolate(
            self.times,
            self.electric,
            time,
            slopes=self.electric_slopes,
            bounds="fill",
        ).values
        magnetic = cubic_hermite_interpolate(
            self.times,
            self.magnetic,
            time,
            slopes=self.magnetic_slopes,
            bounds="fill",
        ).values
        return (
            0.5 * (electric[0] * carrier + electric[1] * jnp.conj(carrier)),
            0.5 * (magnetic[0] * carrier + magnetic[1] * jnp.conj(carrier)),
        )


def _modal_profiles(
    plan: QuasiCylindricalAntennaPlan,
    grid: QuasiCylindricalGrid,
    field: Array,
    /,
) -> np.ndarray:
    """Circular modal profiles ``[t, part, m, j, (F_+, F_−)]`` of sampled ``(F_x, F_y)``.

    Samples are bilinearly interpolated onto ``(r_j, θ_q)`` and Fourier-analyzed
    in ``θ`` with ``c_m/N_θ Σ_q e^{−imθ_q}``; ``part`` 1 analyzes the conjugate
    envelope.
    """
    samples = 4 * (grid.mode_count + 2)
    theta = 2.0 * math.pi * np.arange(samples) / samples
    radii = grid.radial_coordinates
    x = (radii[:, None] * np.cos(theta)[None, :]).reshape(-1)
    y = (radii[:, None] * np.sin(theta)[None, :]).reshape(-1)
    along_first = linear_interpolate(
        plan.first_coordinates, field, jnp.asarray(x), axis=0, bounds="fill"
    )

    def second(values: Array, query: Array) -> Array:
        return linear_interpolate(
            plan.second_coordinates, values, query, axis=0, bounds="fill"
        ).values

    points = np.asarray(jax.vmap(second)(along_first.values, jnp.asarray(y)))  # [P, t, 2]
    points = points.reshape((radii.size, samples, plan.times.shape[0], 2))
    profiles = []
    for part in (points, np.conj(points)):
        cos, sin = np.cos(theta)[None, :, None], np.sin(theta)[None, :, None]
        radial = part[..., 0] * cos + part[..., 1] * sin
        azimuthal = -part[..., 0] * sin + part[..., 1] * cos
        modes = []
        for mode in range(grid.mode_count):
            weight = (1.0 if mode == 0 else 2.0) / samples
            phase = np.exp(-1j * mode * theta)[None, :, None]
            r_m = weight * np.sum(radial * phase, axis=1)
            t_m = weight * np.sum(azimuthal * phase, axis=1)
            modes.append(np.stack((0.5 * (r_m - 1j * t_m), 0.5 * (r_m + 1j * t_m)), -1))
        profiles.append(np.stack(modes))  # [m, j, t, 2]
    return np.moveaxis(np.stack(profiles), 3, 0)  # [t, part, m, j, 2]


def _prepare_antenna(
    plan: QuasiCylindricalAntennaPlan,
    grid: QuasiCylindricalGrid,
    hankel: PreparedSharedGridHankel,
    /,
) -> _PreparedAntenna:
    sign = plan.emission_sign
    electric = _modal_profiles(plan, grid, plan.electric)
    magnetic = _modal_profiles(plan, grid, plan.magnetic)
    # K = s ẑ × H' and K_m = −s ẑ × E'; (ẑ × F)_± = ∓i F_±.
    sheet = np.stack((-1j * sign * magnetic[..., 0], 1j * sign * magnetic[..., 1]), -1)
    magnetic_sheet = np.stack(
        (1j * sign * electric[..., 0], -1j * sign * electric[..., 1]), -1
    )

    def spectrum(values: np.ndarray) -> Array:
        # values [t, part, m, j, 2] → Cartesian-equivalent [t, part, m, n, 3].
        moved = jnp.moveaxis(jnp.asarray(values), (2, 3), (0, 1))
        plus = hankel.forward(moved[..., 0], -1)
        minus = hankel.forward(moved[..., 1], 1)
        result = jnp.stack(
            (1j * (plus - minus), -(plus + minus), jnp.zeros_like(plus)), axis=-1
        )
        return jnp.moveaxis(result, (0, 1), (2, 3))

    electric_spectrum = spectrum(sheet)
    magnetic_spectrum = spectrum(magnetic_sheet)
    return _PreparedAntenna(
        times=plan.times,
        electric=electric_spectrum,
        magnetic=magnetic_spectrum,
        electric_slopes=local_cubic_slopes(plan.times, electric_spectrum),
        magnetic_slopes=local_cubic_slopes(plan.times, magnetic_spectrum),
        plane_coordinate=plan.plane_coordinate,
        carrier_angular_frequency=plan.carrier_angular_frequency,
    )


# -- Huygens cylinders ------------------------------------------------------------


class _SurfaceCell(NamedTuple):
    position: np.ndarray
    normal: np.ndarray
    measure: float
    patch: int
    nodes: tuple[tuple[int, int, float], ...]


def _cylinder_cells(
    plan: QuasiCylindricalHuygensPlan, grid: QuasiCylindricalGrid, /
) -> list[_SurfaceCell]:
    dr, dz = grid.radial_spacing, grid.axial_spacing
    samples = plan.azimuthal_samples
    dtheta = 2.0 * math.pi / samples
    theta = dtheta * (np.arange(samples) + 0.5)
    cells = []
    for side, node in enumerate((plan.lower, plan.upper)):
        z = grid.lower + node * dz
        for j in range(plan.radial_face):
            radius = (j + 0.5) * dr
            for angle in theta:
                cells.append(
                    _SurfaceCell(
                        np.asarray(
                            [radius * math.cos(angle), radius * math.sin(angle), z]
                        ),
                        np.asarray([0.0, 0.0, 1.0 if side else -1.0]),
                        radius * dr * dtheta,
                        side,
                        ((j, node, 1.0),),
                    )
                )
    radius = plan.radial_face * dr
    # Four-point midpoint interpolation weights (−1, 9, 9, −1)/16.
    stencil = ((-1, -1.0 / 16.0), (0, 9.0 / 16.0), (1, 9.0 / 16.0), (2, -1.0 / 16.0))
    for k in range(plan.lower, plan.upper):
        z = grid.lower + (k + 0.5) * dz
        for angle in theta:
            cells.append(
                _SurfaceCell(
                    np.asarray([radius * math.cos(angle), radius * math.sin(angle), z]),
                    np.asarray([math.cos(angle), math.sin(angle), 0.0]),
                    radius * dz * dtheta,
                    2,
                    tuple(
                        (
                            plan.radial_face - 1 + a,
                            (k + b) % grid.axial_count,
                            wa * wb,
                        )
                        for a, wa in stencil
                        for b, wb in stencil
                    ),
                )
            )
    return cells


def _cylinder_geometry(
    plan: QuasiCylindricalHuygensPlan, grid: QuasiCylindricalGrid, /
) -> _SurfaceGeometry:
    """Sparse maps from stacked ``[Re/Im, m, j, k, (F_+, F_−, F_z)]`` to Cartesian cells.

    With ``P = Σ F_+ e^{i(m−1)θ}``, ``Q = Σ F_− e^{i(m+1)θ}``, ``S = Σ F_z e^{imθ}``:
    ``F_x = Re(P + Q)``, ``F_y = −Im(P − Q)``, ``F_z = Re S``.
    """
    modes, nr, nz = grid.mode_count, grid.radial_count, grid.axial_count
    cells = _cylinder_cells(plan, grid)
    sources, targets, weights = [], [], []

    def flat(part: int, mode: int, j: int, k: int, component: int) -> int:
        return (((part * modes + mode) * nr + j) * nz + k) * 3 + component

    for index, cell in enumerate(cells):
        angle = math.atan2(cell.position[1], cell.position[0])
        for j, k, w in cell.nodes:
            for mode in range(modes):
                for component, offset in enumerate(_CIRCULAR_OFFSETS):
                    phase = (mode + offset) * angle
                    cosine, sine = math.cos(phase), math.sin(phase)
                    # Re(A e^{iφ}) = Re A cos φ − Im A sin φ; Im(A e^{iφ}) = Re A sin φ + Im A cos φ.
                    real = (cosine, -sine)
                    imaginary = (sine, cosine)
                    match component:
                        case 0 | 1:
                            sign = 1.0 if component == 0 else -1.0
                            rows = ((0, real, 1.0), (1, imaginary, -sign))
                        case _:
                            rows = ((2, real, 1.0),)
                    for axis, factors, scale in rows:
                        for part, factor in enumerate(factors):
                            coefficient = w * scale * factor
                            if coefficient != 0.0:
                                sources.append(flat(part, mode, j, k, component))
                                targets.append(3 * index + axis)
                                weights.append(coefficient)
    size = 2 * modes * nr * nz * 3
    geometry_id = canonical_fingerprint(
        {
            "kind": "quasi-cylindrical-huygens-geometry",
            "plan": plan.plan_id,
            "grid": grid.grid_id,
        }
    )
    source = np.asarray(sources, dtype=np.int32)
    target = np.asarray(targets, dtype=np.int32)
    coefficients = np.asarray(weights, dtype=np.float64)

    def routes(name: str) -> Any:
        return _gather_map(
            source,
            target,
            coefficients,
            source_size=size,
            target_count=len(cells),
            operator_id=f"{geometry_id}:{name}",
        )

    indices = jnp.asarray(np.unique(source))
    return _SurfaceGeometry(
        positions=jnp.asarray(np.stack(tuple(cell.position for cell in cells))),
        normals=jnp.asarray(np.stack(tuple(cell.normal for cell in cells))),
        measures=jnp.asarray(np.asarray(tuple(cell.measure for cell in cells))),
        patches=jnp.asarray(np.asarray(tuple(cell.patch for cell in cells), np.int32)),
        electric_gather=routes("electric"),
        magnetic_gather=routes("magnetic"),
        electric_indices=indices,
        magnetic_indices=indices,
        geometry_id=geometry_id,
    )


def _stacked(values: Array, /) -> Array:
    return jnp.stack((jnp.real(values), jnp.imag(values))).reshape((-1,))


# -- prepared solver ----------------------------------------------------------------


class PreparedQuasiCylindricalMaxwell(AbstractPreparedPICFieldSolver, NonTrainableState):
    """Prepared quasi-cylindrical PSATD solver bound to one PIC run's transfers."""

    plan: QuasiCylindricalMaxwellPlan
    transfers: tuple[PreparedAzimuthalTransfer, ...]
    transform: _ModalTransform
    operators: SpectralOperators
    antennas: tuple[_PreparedAntenna, ...]
    huygens: tuple[PreparedSpectralHuygensBox, ...]
    damping_rate: Array | None
    radial_volume: Array
    axial_wavenumbers: Array
    maximum_symbol: float = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    field_dtype: RealPrecisionDType = eqx.field(static=True)

    def __init__(
        self,
        plan: QuasiCylindricalMaxwellPlan,
        transfers: Sequence[PreparedAzimuthalTransfer],
        /,
    ) -> None:
        if not isinstance(plan, QuasiCylindricalMaxwellPlan):
            raise TypeError("plan must be a QuasiCylindricalMaxwellPlan.")
        transfer_values = tuple(transfers)
        if any(
            not isinstance(value, PreparedAzimuthalTransfer) for value in transfer_values
        ):
            raise TypeError("transfers must be PreparedAzimuthalTransfer instances.")
        grid = plan.grid
        if any(value.plan.grid.grid_id != grid.grid_id for value in transfer_values):
            raise ValueError("Every azimuthal transfer must use the solver's grid.")
        hankel = SharedGridHankelPlan(
            grid.radius, grid.radial_count, grid.mode_count
        ).prepare()
        if not bool(hankel.evidence.successful):
            raise ValueError(
                "The shared-grid Hankel pseudoinverses failed their rank or residual "
                "certification."
            )
        axial = FFTLinearTransform(grid.axial_count)
        speed = plan.speed_of_light
        operators = _operators(
            hankel, grid, plan.galilean_velocity, speed, plan.permittivity
        )
        huygens = []
        for box in plan.observers:
            geometry = _cylinder_geometry(box, grid)
            huygens.append(
                PreparedSpectralHuygensBox(
                    geometry=geometry,
                    current_gather=geometry.electric_gather,
                    acquisition=box.acquisition,
                    exterior=box.exterior,
                    prepared_id=canonical_fingerprint(
                        {
                            "kind": "prepared-quasi-cylindrical-huygens",
                            "plan": box.plan_id,
                            "geometry": geometry.geometry_id,
                        }
                    ),
                )
            )
        radial_max = float(np.max(np.asarray(hankel.wavenumbers)))
        axial_max = float(np.max(np.abs(_axial_wavenumbers(grid))))
        self.plan = plan
        self.transfers = transfer_values
        self.transform = _ModalTransform(hankel, axial, 1.0 / math.sqrt(grid.axial_count))
        self.operators = operators
        self.antennas = tuple(
            _prepare_antenna(value, grid, hankel) for value in plan.antennas
        )
        self.huygens = tuple(huygens)
        self.damping_rate = (
            None if plan.damping is None else jnp.asarray(plan.damping.rate(grid, speed))
        )
        self.radial_volume = jnp.asarray(
            grid.radial_coordinates * grid.radial_spacing * grid.axial_spacing
        )
        self.axial_wavenumbers = jnp.asarray(_axial_wavenumbers(grid))
        self.maximum_symbol = math.hypot(radial_max, axial_max)
        self.spatial_dimension = 3
        self.field_dtype = "float64"
        self.solver_id = canonical_fingerprint(
            {
                "kind": "prepared-quasi-cylindrical-maxwell",
                "plan": plan.plan_id,
                "hankel": hankel.prepared_id,
                "transfers": [value.prepared_id for value in transfer_values],
            }
        )

    # -- geometry and core protocol ------------------------------------------------

    @property
    def grid(self) -> QuasiCylindricalGrid:
        return self.plan.grid

    @property
    def hankel_evidence(self) -> Any:
        return self.transform.hankel.evidence

    @property
    def grid_velocity(self) -> tuple[float, ...]:
        """Axial Galilean grid velocity; particle positions are grid coordinates."""
        return (0.0, 0.0, self.plan.galilean_velocity)

    @property
    def stable_step(self) -> Array:
        """Largest step with unaliased vacuum dispersion, ``π/(c max|k̃|)``."""
        return jnp.asarray(math.pi / (self.plan.speed_of_light * self.maximum_symbol))

    @property
    def displacement_widths(self) -> Array:
        grid = self.grid
        return jnp.asarray([grid.radial_spacing, grid.radial_spacing, grid.axial_spacing])

    def validate_species(self, species: tuple[PICSpeciesPlan, ...], /) -> None:
        if len(species) != len(self.transfers):
            raise ValueError(
                "Quasi-cylindrical PIC requires one azimuthal transfer per species."
            )
        for value, transfer in zip(species, self.transfers, strict=True):
            if value.population.particles.prepared_id != transfer.particles_id:
                raise ValueError(
                    "Species population and azimuthal transfer use different particles."
                )

    def pairing_probe(self, species: int, capacity: int, /) -> tuple[Array, Array]:
        del species
        grid = self.grid
        slot = np.arange(capacity)
        radius = grid.radius * (0.2 + 0.3 * ((slot * 0.618) % 1.0))
        angle = 2.0 * math.pi * ((slot * 0.382) % 1.0)
        z = grid.lower + grid.length * (0.25 + 0.5 * ((slot * 0.707) % 1.0))
        start = np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=-1)
        step = np.asarray(
            [
                0.3 * grid.radial_spacing,
                -0.2 * grid.radial_spacing,
                0.4 * grid.axial_spacing,
            ]
        )
        return jnp.asarray(start), jnp.asarray(start + step)

    def _spectral_zeros(self) -> Array:
        grid = self.grid
        return jnp.zeros(
            (grid.mode_count, grid.radial_count, grid.axial_count, 3),
            dtype=jnp.complex128,
        )

    def _state(
        self,
        electric: Array,
        magnetic: Array,
        charge: Array,
        window_offset: Array,
        /,
    ) -> QuasiCylindricalMaxwellState:
        physical = self.transform.vector_inverse(jnp.stack((electric, magnetic), -2))
        averaged = self.plan.variant == "averaged-galilean"
        zero = jnp.zeros_like(charge) if self.antennas else None
        return QuasiCylindricalMaxwellState(
            spectral_electric=electric,
            spectral_magnetic=magnetic,
            charge=charge,
            electric=physical[..., 0, :],
            magnetic=physical[..., 1, :],
            averaged_electric=physical[..., 0, :] if averaged else None,
            averaged_magnetic=physical[..., 1, :] if averaged else None,
            antenna_charge=zero,
            antenna_magnetic_charge=zero,
            window_offset=window_offset,
            observations=tuple(box.initialize() for box in self.huygens),
        )

    def field_with_charge(self, charge: Array, /) -> QuasiCylindricalMaxwellState:
        return self._state(
            self._spectral_zeros(),
            self._spectral_zeros(),
            jnp.asarray(charge, dtype=jnp.complex128),
            jnp.zeros((), dtype=jnp.float64),
        )

    def initialize_field(
        self, charge: Array, /, *, magnetic: Any = None
    ) -> tuple[QuasiCylindricalMaxwellState, Array]:
        """Coulomb field ``Ẽ = −ik̃ρ̂/(ε|k̃|²)`` of the spectral charge.

        The radial Hankel basis vanishes at ``r = R`` (a grounded wall), so a
        non-neutral charge is admissible; only content in unresolved modes
        (``m ≥ 1``, ``k_⊥ = k_z = 0``) is refused. ``magnetic`` is an optional
        circular real-space modal field ``[M+1, N_r, N_z, 3]``.
        """
        rho = jnp.asarray(charge, dtype=jnp.complex128)
        electric = self.operators.coulomb_field(rho)
        if magnetic is None:
            field_b = self._spectral_zeros()
        else:
            value = jnp.asarray(magnetic, dtype=jnp.complex128)
            if value.shape != self._spectral_zeros().shape:
                raise ValueError("magnetic must have shape (M+1, N_r, N_z, 3).")
            field_b = self.transform.vector_forward(value)
        unresolved = jnp.max(
            jnp.where(self.operators.resolved, 0.0, jnp.abs(rho)), initial=0.0
        )
        state = self._state(electric, field_b, rho, jnp.zeros((), dtype=jnp.float64))
        return state, (unresolved <= _UNRESOLVED_TOLERANCE) & jnp.all(
            jnp.isfinite(state.electric)
        )

    def field_charge(self, field: QuasiCylindricalMaxwellState, /) -> Array:
        return field.charge

    def _energy(self, electric: Array, magnetic: Array, /) -> Array:
        """``½∫(ε|E|² + |B|²/μ)dV`` of circular modes; ``2π`` for ``m = 0``, else ``π``."""
        modes = self.grid.mode_count
        weight = jnp.asarray(np.where(np.arange(modes) == 0, 2.0 * math.pi, math.pi))

        def density(value: Array) -> Array:
            squared = jnp.abs(value) ** 2
            return 2.0 * (squared[..., 0] + squared[..., 1]) + squared[..., 2]

        total = (
            self.plan.permittivity * density(electric)
            + density(magnetic) / self.plan.permeability
        )
        return 0.5 * jnp.sum(
            weight[:, None, None] * self.radial_volume[None, :, None] * total
        )

    def field_energy(self, field: QuasiCylindricalMaxwellState, /) -> Array:
        return self._energy(field.electric, field.magnetic)

    # -- deposition -------------------------------------------------------------------

    def deposit_charge(
        self,
        species: int,
        position: Array,
        macrocharge: Array,
        active: Array,
        /,
    ) -> tuple[Array, Array]:
        density, successful = self.transfers[species].deposit(
            position, active, macrocharge[:, None], (0,)
        )
        return self.transform.scalar_forward(density)[..., 0], successful

    def deposit(
        self,
        species: int,
        start: Array,
        end: Array,
        velocity: Array,
        macrocharge: Array,
        active: Array,
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        """Endpoint spectral charges and the midpoint current ``q v`` of one species.

        ``velocity`` is the step-mean lab velocity, so the deposited current is
        the lab current in Galilean coordinates too; only its transverse part
        enters the field update.
        """
        transfer = self.transfers[species]
        first, first_ok = transfer.deposit(start, active, macrocharge[:, None], (0,))
        last, last_ok = transfer.deposit(end, active, macrocharge[:, None], (0,))
        spectral = self.transform.scalar_forward(jnp.concatenate((first, last), -1))
        vx, vy, vz = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        amplitude = macrocharge[:, None] * jnp.stack(
            (0.5 * (vx - 1j * vy), 0.5 * (vx + 1j * vy), vz + 0.0j), axis=-1
        )
        current, current_ok = transfer.deposit(
            0.5 * (start + end), active, amplitude, _CIRCULAR_OFFSETS
        )
        return PICFieldDeposit(
            QuasiCylindricalSource(current, (last - first)[..., 0]),
            spectral[..., 0],
            spectral[..., 1],
            jnp.zeros((), dtype=jnp.float64),
            first_ok & last_ok & current_ok,
            # Continuity holds by construction (zero defect); the scale is the
            # spectral charge magnitude moved over the step.
            jnp.max(jnp.abs(spectral), initial=0.0) / step_size,
        )

    # -- advance ----------------------------------------------------------------------

    def _antenna_sources(
        self, time: Array, window_offset: Array, /
    ) -> tuple[Array, Array]:
        """Spectral antenna current densities ``J``, ``M`` at ``time``.

        The sheet is the band-limited axial delta ``e^{−ik_z(z_a − z₀)}/L`` at the
        antenna's window coordinate; the unpaired even-count Nyquist entry is
        left empty so the ``m = 0`` source stays real.
        """
        grid = self.grid
        current = self._spectral_zeros()
        magnetic = self._spectral_zeros()
        kz = self.axial_wavenumbers
        live = np.ones(grid.axial_count, dtype=np.bool_)
        if grid.axial_count % 2 == 0:
            live[grid.axial_count // 2] = False
        for antenna in self.antennas:
            electric_sheet, magnetic_sheet = antenna.sheets(time)
            position = (
                antenna.plane_coordinate
                - self.plan.galilean_velocity * time
                - window_offset
            )
            delta = jnp.where(
                live, jnp.exp(-1j * kz * (position - grid.lower)) / grid.length, 0.0
            )
            current = current + electric_sheet[:, :, None, :] * delta[None, None, :, None]
            magnetic = magnetic + (
                magnetic_sheet[:, :, None, :] * delta[None, None, :, None]
            )
        return current, magnetic

    def _propagate(
        self,
        field: QuasiCylindricalMaxwellState,
        current: Array,
        charge_change: Array,
        antenna_current: Array,
        antenna_magnetic: Array,
        dt: Array,
        /,
    ) -> tuple[Array, Array, Array | None, Array | None, Array | None, Array | None]:
        operators = self.operators
        electric, magnetic = field.spectral_electric, field.spectral_magnetic
        start, end = field.charge, field.charge + charge_change
        match self.plan.charge_conservation:
            case "spectral-correction":
                driven = operators.transverse_current(current) + galilean_charge_current(
                    operators, start, end, dt
                )
            case "update-with-rho":
                driven = operators.transverse_current(current)
            case _:
                raise ValueError("charge_conservation is invalid.")
        driven = driven + antenna_current
        next_e, next_b = exact_interval(operators, electric, magnetic, driven, None, dt)
        first_order = tuple(phi_functions(z, 2)[1] for z in operators.arguments(dt))
        if self.antennas:
            source_e, source_b = apply_function(
                operators,
                (first_order[0], first_order[1], first_order[2]),
                jnp.zeros_like(antenna_magnetic),
                -antenna_magnetic,
            )
            next_e = next_e + dt * source_e
            next_b = next_b + dt * source_b
        if self.plan.charge_conservation == "update-with-rho":
            next_e = next_e + gauss_following_increment(operators, start, end, dt)
        averaged_e = averaged_b = None
        if self.plan.variant == "averaged-galilean":
            lower, upper = 0.5 * dt, 1.5 * dt
            window_e, window_b = window_integral(
                operators, electric, magnetic, driven, None, lower, upper
            )
            if self.plan.charge_conservation == "update-with-rho":
                window_e = window_e + gauss_following_window(
                    operators, start, end, dt, lower, upper
                )
            averaged_e, averaged_b = window_e / dt, window_b / dt
        if field.antenna_charge is None or field.antenna_magnetic_charge is None:
            return next_e, next_b, averaged_e, averaged_b, None, None
        # Declared sheet charges follow the exact longitudinal propagator:
        # ρ(h) = e^{iκh}ρ(0) − hφ₁(iκh) ik̃·J.
        phase = jnp.exp(1j * operators.advection * dt)
        antenna_charge = phase * field.antenna_charge - dt * first_order[0] * (
            _divergence(operators, antenna_current)
        )
        antenna_magnetic_charge = phase * field.antenna_magnetic_charge - dt * (
            first_order[0] * _divergence(operators, antenna_magnetic)
        )
        return (
            next_e,
            next_b,
            averaged_e,
            averaged_b,
            antenna_charge,
            antenna_magnetic_charge,
        )

    def _damp(
        self, electric: Array, magnetic: Array, dt: Array, /
    ) -> tuple[Array, Array]:
        """Damp the solenoidal fields in the radial layer; divergences unchanged."""
        if self.damping_rate is None:
            return electric, magnetic
        operators = self.operators
        transverse_e = electric - operators.longitudinal_electric(electric)
        transverse_b = magnetic - operators.longitudinal_magnetic(magnetic)
        physical = self.transform.vector_inverse(
            jnp.stack((transverse_e, transverse_b), -2)
        )
        factor = jnp.exp(-self.damping_rate * dt) - 1.0
        change = self.transform.vector_forward(
            factor[None, :, None, None, None] * physical
        )
        delta_e = change[..., 0, :]
        delta_b = change[..., 1, :]
        return (
            electric + delta_e - operators.longitudinal_electric(delta_e),
            magnetic + delta_b - operators.longitudinal_magnetic(delta_b),
        )

    def _residuals(
        self,
        electric: Array,
        magnetic: Array,
        charge: Array,
        antenna_charge: Array | None,
        antenna_magnetic_charge: Array | None,
        /,
    ) -> tuple[Array, Array]:
        """Real-space max-norm Gauss residuals of both constraints."""
        operators = self.operators
        total = charge if antenna_charge is None else charge + antenna_charge
        residual_e = (
            _divergence(operators, electric)
            - jnp.where(operators.resolved, total, 0.0) / self.plan.permittivity
        )
        residual_b = _divergence(operators, magnetic)
        if antenna_magnetic_charge is not None:
            residual_b = residual_b - jnp.where(
                operators.resolved, antenna_magnetic_charge, 0.0
            )
        physical = self.transform.scalar_inverse(
            jnp.stack((residual_e, residual_b), axis=-1)
        )
        return (
            jnp.max(jnp.abs(physical[..., 0]), initial=0.0),
            jnp.max(jnp.abs(physical[..., 1]), initial=0.0),
        )

    def advance(
        self,
        time: Array,
        field: QuasiCylindricalMaxwellState,
        current: QuasiCylindricalSource,
        step_size: Array,
        /,
    ) -> PICFieldAdvance:
        if not isinstance(current, QuasiCylindricalSource):
            raise TypeError("Quasi-cylindrical Maxwell advances with its own source.")
        if (
            current.current.shape != self._spectral_zeros().shape
            or current.charge_change.shape != self._spectral_zeros().shape[:-1]
        ):
            raise ValueError("The deposited source does not match the modal grid.")
        dt = jnp.asarray(step_size, dtype=jnp.float64).reshape(())
        start_time = jnp.asarray(time, dtype=jnp.float64).reshape(())
        spectral_current = self.transform.vector_forward(current.current)
        charge_change = self.transform.scalar_forward(current.charge_change[..., None])
        charge_change = charge_change[..., 0]
        antenna_current, antenna_magnetic = (
            self._antenna_sources(start_time + 0.5 * dt, field.window_offset)
            if self.antennas
            else (self._spectral_zeros(), self._spectral_zeros())
        )
        next_e, next_b, average_e, average_b, sheet, sheet_m = self._propagate(
            field,
            spectral_current,
            charge_change,
            antenna_current,
            antenna_magnetic,
            dt,
        )
        undamped = None
        if self.damping_rate is not None:
            undamped = (next_e, next_b)
            next_e, next_b = self._damp(next_e, next_b, dt)
        charge = field.charge + charge_change
        outputs = [next_e, next_b]
        if average_e is not None and average_b is not None:
            outputs += [average_e, average_b]
        if undamped is not None:
            outputs += list(undamped)
        physical = self.transform.vector_inverse(jnp.stack(outputs, axis=-2))
        new_e, new_b = physical[..., 0, :], physical[..., 1, :]
        averaged_e = averaged_b = None
        if average_e is not None:
            averaged_e, averaged_b = physical[..., 2, :], physical[..., 3, :]
        energy = self._energy(new_e, new_b)
        absorbed = jnp.zeros((), dtype=jnp.float64)
        if undamped is not None:
            absorbed = self._energy(physical[..., 2, :], physical[..., 3, :]) - energy
        electric_constraint, magnetic_constraint = self._residuals(
            next_e, next_b, charge, sheet, sheet_m
        )
        next_time = start_time + dt
        observations = []
        surface = jnp.zeros((), dtype=jnp.float64)
        flat_current = _stacked(current.current)
        for box, state in zip(self.huygens, field.observations, strict=True):
            observations.append(
                box.update(
                    next_time,
                    _stacked(new_e),
                    _stacked(new_b / self.plan.permeability),
                    state,
                )
            )
            active = box.acquisition.active(next_time) | box.acquisition.active(
                start_time
            )
            surface = jnp.maximum(
                surface, jnp.where(active, box.surface_current(flat_current), 0.0)
            )
        state = QuasiCylindricalMaxwellState(
            spectral_electric=next_e,
            spectral_magnetic=next_b,
            charge=charge,
            electric=new_e,
            magnetic=new_b,
            averaged_electric=averaged_e,
            averaged_magnetic=averaged_b,
            antenna_charge=sheet,
            antenna_magnetic_charge=sheet_m,
            window_offset=field.window_offset,
            observations=tuple(observations),
        )
        finite = jnp.all(jnp.isfinite(next_e)) & jnp.all(jnp.isfinite(next_b))
        diagnostics = QuasiCylindricalMaxwellDiagnostics(
            electric_constraint=electric_constraint,
            magnetic_constraint=magnetic_constraint,
            energy=energy,
            absorbed_energy=absorbed,
            surface_current=surface,
        )
        return PICFieldAdvance(
            state,
            charge,
            electric_constraint,
            magnetic_constraint,
            energy,
            diagnostics,
            finite & (surface == 0.0) & (dt > 0.0),
        )

    # -- gather -------------------------------------------------------------------------

    def gather_fields(
        self,
        species: int,
        position: Array,
        active: Array,
        field: QuasiCylindricalMaxwellState,
        /,
    ) -> tuple[Array, Array, Array]:
        electric, magnetic = field.electric, field.magnetic
        if field.averaged_electric is not None and field.averaged_magnetic is not None:
            electric, magnetic = field.averaged_electric, field.averaged_magnetic
        synthesis, support = self.transfers[species].gather(
            position,
            active,
            jnp.concatenate((electric, magnetic), axis=-1),
            _CIRCULAR_OFFSETS + _CIRCULAR_OFFSETS,
        )

        def cartesian(values: Array) -> Array:
            plus, minus, along = values[:, 0], values[:, 1], values[:, 2]
            return jnp.stack(
                (
                    jnp.real(plus + minus),
                    -jnp.imag(plus - minus),
                    jnp.real(along),
                ),
                axis=-1,
            )

        return cartesian(synthesis[:, :3]), cartesian(synthesis[:, 3:]), support

    # -- capabilities -------------------------------------------------------------------

    def dispersion_frequency(
        self, wavevector: ArrayLike, step_size: ArrayLike, /
    ) -> Array:
        """Vacuum ``ω = c|k|``: every ``(m, k_⊥, k_z)`` mode is integrated exactly.

        Galilean variants report the lab-frame frequency; stepped modes alias
        for ``c|k|Δt > π``.
        """
        dt = float(jnp.asarray(step_size))
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("step_size must be positive and finite.")
        k = np.asarray(wavevector, dtype=np.float64)
        if k.ndim != 2 or k.shape[1] != 3:
            raise ValueError("wavevector must have shape (K, 3).")
        return jnp.asarray(self.plan.speed_of_light * np.linalg.norm(k, axis=-1))

    def huygens_phasors(
        self, field: QuasiCylindricalMaxwellState, /
    ) -> tuple[HuygensSurfacePhasors, ...]:
        return tuple(
            box.surface_phasors(state)
            for box, state in zip(self.huygens, field.observations, strict=True)
        )

    def window_interval(self, axis: int, /) -> float:
        _require_axial(axis)
        return self.grid.axial_spacing

    def window_bounds(self, axis: int, /) -> tuple[float, float]:
        _require_axial(axis)
        return (self.grid.lower, self.grid.upper)

    def _physical_shift(self, values: Array, cells: int, /) -> Array:
        """Translate nodes ``[m, j, k, …]`` by ``cells`` toward lower ``z``, zero-filled."""
        count = self.grid.axial_count
        index = np.arange(count) + cells
        keep = jnp.asarray(index < count).reshape(
            (1, 1, count) + (1,) * (values.ndim - 3)
        )
        moved = jnp.take(values, jnp.asarray(np.minimum(index, count - 1)), axis=2)
        return jnp.where(keep, moved, 0.0)

    def _axial_shift(self, coefficients: Array, cells: int, /) -> Array:
        """Translate spectra through the mixed ``(k_⊥, z)`` representation."""
        return self.transform.axial_forward(
            self._physical_shift(self.transform.axial_inverse(coefficients), cells)
        )

    def shift_window(
        self, field: QuasiCylindricalMaxwellState, axis: int, cells: int, /
    ) -> QuasiCylindricalMaxwellState:
        """Integer-cell axial translation, then spectral Gauss/solenoidal projection.

        Zero-filling the leading cells truncates fields whose spectral divergence
        is nonlocal; the curl-free correction restores ``ik̃·E = (ρ + ρ_a)/ε`` and
        the longitudinal ``B`` is reset to the declared magnetic charge field.
        """
        _require_axial(axis)
        shift = int(cells)
        if shift <= 0 or shift >= self.grid.axial_count:
            raise ValueError("Window shifts must be positive and shorter than the grid.")
        operators = self.operators
        electric = self._axial_shift(field.spectral_electric, shift)
        magnetic = self._axial_shift(field.spectral_magnetic, shift)
        charge = self._axial_shift(field.charge[..., None], shift)[..., 0]
        antenna_charge = (
            None
            if field.antenna_charge is None
            else self._axial_shift(field.antenna_charge[..., None], shift)[..., 0]
        )
        antenna_magnetic = (
            None
            if field.antenna_magnetic_charge is None
            else self._axial_shift(field.antenna_magnetic_charge[..., None], shift)[
                ..., 0
            ]
        )
        total = charge if antenna_charge is None else charge + antenna_charge
        electric = electric + operators.coulomb_field(
            total - self.plan.permittivity * _divergence(operators, electric)
        )
        magnetic = magnetic - operators.longitudinal_magnetic(magnetic)
        if antenna_magnetic is not None:
            magnetic = magnetic + _magnetic_charge_field(operators, antenna_magnetic)
        averaged = (
            None
            if field.averaged_electric is None or field.averaged_magnetic is None
            else (
                self._physical_shift(field.averaged_electric, shift),
                self._physical_shift(field.averaged_magnetic, shift),
            )
        )
        physical = self.transform.vector_inverse(jnp.stack((electric, magnetic), -2))
        return QuasiCylindricalMaxwellState(
            spectral_electric=electric,
            spectral_magnetic=magnetic,
            charge=charge,
            electric=physical[..., 0, :],
            magnetic=physical[..., 1, :],
            averaged_electric=None if averaged is None else averaged[0],
            averaged_magnetic=None if averaged is None else averaged[1],
            antenna_charge=antenna_charge,
            antenna_magnetic_charge=antenna_magnetic,
            window_offset=field.window_offset + shift * self.grid.axial_spacing,
            observations=field.observations,
        )

    def restart_component(
        self, field: QuasiCylindricalMaxwellState, /
    ) -> PICRestartComponent:
        return restart_component("field", self.solver_id, field)

    def restore_component(
        self, component: PICRestartComponent, /
    ) -> QuasiCylindricalMaxwellState:
        template = self.field_with_charge(self._spectral_zeros()[..., 0])
        return restore_component(component, "field", self.solver_id, template)

    def project_gauss(
        self, field: QuasiCylindricalMaxwellState, charge: Array, /
    ) -> PICGaussProjectionResult:
        """Spectral Poisson projection ``Ẽ ← Ẽ + Ẽ_c`` onto the Gauss charge ``charge``.

        ``Ẽ_c = −ik̃(ρ + ρ_a − εik̃·Ẽ)/(ε|k̃|²)`` is curl free; ``B``, declared
        antenna charges, and observers are unchanged.
        """
        rho = jnp.asarray(charge, dtype=jnp.complex128)
        operators = self.operators
        before, _ = self._residuals(
            field.spectral_electric,
            field.spectral_magnetic,
            rho,
            field.antenna_charge,
            field.antenna_magnetic_charge,
        )
        total = rho if field.antenna_charge is None else rho + field.antenna_charge
        correction = operators.coulomb_field(
            total
            - self.plan.permittivity * _divergence(operators, field.spectral_electric)
        )
        electric = field.spectral_electric + correction
        after, _ = self._residuals(
            electric,
            field.spectral_magnetic,
            rho,
            field.antenna_charge,
            field.antenna_magnetic_charge,
        )
        physical = self.transform.vector_inverse(correction)
        projected = QuasiCylindricalMaxwellState(
            spectral_electric=electric,
            spectral_magnetic=field.spectral_magnetic,
            charge=rho,
            electric=field.electric + physical,
            magnetic=field.magnetic,
            averaged_electric=None
            if field.averaged_electric is None
            else field.averaged_electric + physical,
            averaged_magnetic=field.averaged_magnetic,
            antenna_charge=field.antenna_charge,
            antenna_magnetic_charge=field.antenna_magnetic_charge,
            window_offset=field.window_offset,
            observations=field.observations,
        )
        return PICGaussProjectionResult(
            projected,
            before,
            after,
            self.field_energy(projected) - self.field_energy(field),
            jnp.all(jnp.isfinite(electric)),
            "spectral-poisson",
        )

    # -- modal conveniences --------------------------------------------------------------

    def add_propagating_field(
        self,
        field: QuasiCylindricalMaxwellState,
        electric: ArrayLike,
        /,
        *,
        direction: AntennaEmissionDirection = "positive",
    ) -> QuasiCylindricalMaxwellState:
        """Superpose a vacuum wave with transverse field ``electric`` on ``field``.

        ``electric`` holds circular modes ``[M+1, N_r, N_z, 3]`` whose ``F_±`` are
        the wave's transverse ``E`` (its ``F_z`` is ignored). ``E_z`` follows from
        ``ik̃·E = 0`` and ``B = k̃ × E/ω`` with ``ω = s·sign(k_z)c|k̃|`` for emission
        direction ``s``, so every Fourier component travels along ``±z``
        (components with ``k_z = 0``, which have no propagation direction, keep
        only their divergence-free part ``F̃_y`` and carry no ``B``). The Gauss
        charge and constraints of ``field`` are unchanged.
        """
        values = jnp.asarray(electric, dtype=jnp.complex128)
        if values.shape != self._spectral_zeros().shape:
            raise ValueError("electric must have shape (M+1, N_r, N_z, 3).")
        sign = (
            1.0
            if parse(direction, AntennaEmissionDirection, "direction") == ("positive")
            else -1.0
        )
        spectral = self.transform.vector_forward(values.at[..., 2].set(0.0))
        symbol = jnp.imag(self.operators.plus)
        radial, axial = symbol[..., 0], symbol[..., 2]
        live = axial != 0.0
        safe = jnp.where(live, axial, 1.0)
        longitudinal = jnp.where(live, -radial * spectral[..., 0] / safe, 0.0)
        wave = jnp.stack(
            (jnp.where(live, spectral[..., 0], 0.0), spectral[..., 1], longitudinal),
            axis=-1,
        )
        frequency = (
            sign
            * jnp.sign(axial)
            * self.plan.speed_of_light
            * jnp.sqrt(self.operators.squared)
        )
        rotating = frequency != 0.0
        cross = jnp.cross(symbol, wave)
        magnetic = jnp.where(
            rotating[..., None],
            cross / jnp.where(rotating, frequency, 1.0)[..., None],
            0.0,
        )
        electric_k = field.spectral_electric + wave
        magnetic_k = field.spectral_magnetic + magnetic
        physical = self.transform.vector_inverse(jnp.stack((electric_k, magnetic_k), -2))
        averaged = field.averaged_electric is not None
        return QuasiCylindricalMaxwellState(
            spectral_electric=electric_k,
            spectral_magnetic=magnetic_k,
            charge=field.charge,
            electric=physical[..., 0, :],
            magnetic=physical[..., 1, :],
            averaged_electric=physical[..., 0, :] if averaged else None,
            averaged_magnetic=physical[..., 1, :] if averaged else None,
            antenna_charge=field.antenna_charge,
            antenna_magnetic_charge=field.antenna_magnetic_charge,
            window_offset=field.window_offset,
            observations=field.observations,
        )

    def circular_field(
        self,
        electric: ArrayLike,
        magnetic: ArrayLike,
        /,
        *,
        charge: ArrayLike | None = None,
    ) -> QuasiCylindricalMaxwellState:
        """Field state from circular real-space modes ``[M+1, N_r, N_z, 3]``.

        The spectral fields are the pseudoinverse analyses of the samples; the
        Gauss charge defaults to the one the analyzed ``E`` carries.
        """
        values = jnp.stack(
            (
                jnp.asarray(electric, dtype=jnp.complex128),
                jnp.asarray(magnetic, dtype=jnp.complex128),
            ),
            axis=-2,
        )
        if values.shape[:3] + values.shape[4:] != self._spectral_zeros().shape:
            raise ValueError("Circular fields must have shape (M+1, N_r, N_z, 3).")
        spectral = self.transform.vector_forward(values)
        electric_k, magnetic_k = spectral[..., 0, :], spectral[..., 1, :]
        rho = (
            self.plan.permittivity * _divergence(self.operators, electric_k)
            if charge is None
            else jnp.asarray(charge, dtype=jnp.complex128)
        )
        return self._state(electric_k, magnetic_k, rho, jnp.zeros((), jnp.float64))


def _require_axial(axis: int, /) -> None:
    if axis != 2:
        raise ValueError("Quasi-cylindrical windows translate along z (axis 2) only.")


__all__ = [
    "PreparedQuasiCylindricalMaxwell",
    "QuasiCylindricalAbsorber",
    "QuasiCylindricalAntennaPlan",
    "QuasiCylindricalHuygensPlan",
    "QuasiCylindricalMaxwellDiagnostics",
    "QuasiCylindricalMaxwellPlan",
    "QuasiCylindricalMaxwellState",
    "QuasiCylindricalSource",
    "RadialDampingPlan",
]
