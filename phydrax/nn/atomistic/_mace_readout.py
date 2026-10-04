#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""MACE readouts, atomic reference energies, head selection, and ZBL repulsion.

Source placement is preserved exactly. Per-node interaction energy is the sum of
the ZBL pair energy and every layer readout; ``"scale-shift"`` scaling applies the
selected head's ``scale * interaction + shift`` to that sum, and the atomic
reference energy ``E0`` is added afterwards, never scaled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._model import register_artifact_value
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_integer
from ...atomistic._types import AtomisticScaleContract
from ...typing import AnyShape, as_host_array, Dim, Float, HostFloat64, PRNGKey
from ...units import ANGSTROM, conversion_factor, ELECTRONVOLT
from ..operator.layers import O3IrrepLinear
from ..operator.representations import O3IrrepLayout
from ._radial import PolynomialCutoff
from ._symmetric_contraction import SymmetricContraction


MACEEnergyScaling: TypeAlias = Literal["unscaled", "scale-shift"]


class MACESpeciesDim(Dim):
    """Model species in declared order."""


class MACEHeadDim(Dim):
    """Declared readout heads."""


class MACEChannelDim(Dim):
    """Scalar model channels entering a readout."""


class MACEReadoutHiddenDim(Dim):
    """Head-major hidden width of the nonlinear readout."""


class _ZBLCoefficientDim(Dim):
    """Universal ZBL screening-function coefficients."""


class _ZBLScreeningDim(Dim):
    """ZBL screening-length exponent and prefactor."""


# Covalent radii in angstrom indexed by atomic number (index 0 is the dummy
# element). Values are Cordero et al., Dalton Trans. (2008) 2832, in the exact
# tabulation the MACE source family consumes, including 2.0 placeholders for
# Z >= 97. Agnesi transforms and ZBL envelopes of imported checkpoints depend
# on these exact values.
_COVALENT_RADII_ANGSTROM = (
    0.2, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41,
    1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6, 1.53, 1.39, 1.39,
    1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.2, 1.19, 1.2, 1.2, 1.16, 2.2, 1.95, 1.9,
    1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38,
    1.39, 1.4, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94,
    1.92, 1.92, 1.89, 1.9, 1.87, 1.87, 1.75, 1.7, 1.62, 1.51, 1.44, 1.41, 1.36,
    1.36, 1.32, 1.45, 1.46, 1.48, 1.4, 1.5, 1.5, 2.6, 2.21, 2.15, 2.06, 2.0, 1.96,
    1.9, 1.87, 1.8, 1.69, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
)  # fmt: skip

# Universal ZBL screening function (Ziegler, Biersack, Littmark 1985) and the
# Coulomb constant e^2 / (4 pi eps0) in eV angstrom, with the source's Bohr
# length 0.529 angstrom in the screening length.
_ZBL_COEFFICIENTS = (0.1818, 0.5099, 0.2802, 0.02817)
_ZBL_EXPONENTS = (3.2, 0.9423, 0.4028, 0.2016)
_ZBL_COULOMB_EV_ANGSTROM = 14.3996
_ZBL_BOHR_ANGSTROM = 0.529
_ZBL_SCREENING_EXPONENT = 0.300
_ZBL_SCREENING_PREFACTOR = 0.4543


def covalent_radii_angstrom(atomic_numbers: Sequence[int], /) -> np.ndarray:
    """Return the tabulated covalent radii (angstrom) of atomic numbers in order."""
    numbers = tuple(positive_integer(value, "atomic_numbers") for value in atomic_numbers)
    if any(value >= len(_COVALENT_RADII_ANGSTROM) for value in numbers):
        raise ValueError("Covalent radii are tabulated for atomic numbers 1..118.")
    return np.asarray(
        [_COVALENT_RADII_ANGSTROM[value] for value in numbers], dtype=np.float64
    )


@final
class MACELinearReadout(StrictModule):
    """Linear invariant readout of the scalar channels into every head."""

    __strict_contract__ = True

    weight: Float[MACEChannelDim, MACEHeadDim]

    def __init__(self, weight: ArrayLike, /) -> None:
        values = jnp.asarray(weight)
        if values.ndim != 2 or min(values.shape) <= 0:
            raise ValueError("A linear readout weight must have shape (channels, heads).")
        if values.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("Linear readout weights must be float32 or float64.")
        self.weight = values

    @classmethod
    def initialize(
        cls, channel_count: int, head_count: int, /, *, key: PRNGKey, dtype: DTypeLike
    ) -> MACELinearReadout:
        """Draw unit-normal weights, the original readout initialization."""
        return cls(jr.normal(key, (channel_count, head_count), dtype=dtype))

    def __call__(self, scalars: Array, head: int, /) -> Array:
        """Energy of one head; ``scalars[..., C] -> [...]``, scaled ``1/sqrt(C)``."""
        column = self.weight[:, head]
        fan_in = jnp.asarray(self.weight.shape[0], dtype=self.weight.dtype)
        return (scalars @ column) / jnp.sqrt(fan_in)


@final
class MACENonlinearReadout(StrictModule):
    """Two-layer gated invariant readout with head-major hidden blocks.

    The hidden width is ``heads * width``; every head reads only its own hidden
    block, exactly as the source head mask, while the second layer keeps the
    whole hidden fan-in normalization ``1 / sqrt(heads * width)``.
    """

    __strict_contract__ = True

    first: Float[MACEChannelDim, MACEReadoutHiddenDim]
    second: Float[MACEReadoutHiddenDim, MACEHeadDim]
    activation_scale: float = eqx.field(static=True)

    def __init__(
        self, first: ArrayLike, second: ArrayLike, /, *, activation_scale: float
    ) -> None:
        first_ = jnp.asarray(first)
        second_ = jnp.asarray(second)
        if first_.ndim != 2 or second_.ndim != 2:
            raise ValueError("Nonlinear readout weights must be rank-two.")
        if first_.dtype != second_.dtype or first_.dtype not in (
            jnp.dtype(jnp.float32),
            jnp.dtype(jnp.float64),
        ):
            raise TypeError("Nonlinear readout weights must share float32/float64.")
        hidden, heads = second_.shape
        if first_.shape[1] != hidden or heads <= 0 or hidden % heads != 0:
            raise ValueError("Nonlinear readout hidden width must be heads * width.")
        scale = float(activation_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("activation_scale must be finite and positive.")
        self.first = first_
        self.second = second_
        self.activation_scale = scale

    @classmethod
    def initialize(
        cls,
        channel_count: int,
        width: int,
        head_count: int,
        /,
        *,
        activation_scale: float,
        key: PRNGKey,
        dtype: DTypeLike,
    ) -> MACENonlinearReadout:
        first_key, second_key = jr.split(key)
        hidden = width * head_count
        return cls(
            jr.normal(first_key, (channel_count, hidden), dtype=dtype),
            jr.normal(second_key, (hidden, head_count), dtype=dtype),
            activation_scale=activation_scale,
        )

    @property
    def head_count(self) -> int:
        return self.second.shape[1]

    @property
    def width(self) -> int:
        return self.second.shape[0] // self.second.shape[1]

    def __call__(self, scalars: Array, head: int, /) -> Array:
        """Energy of one head; ``scalars[..., C] -> [...]``."""
        width = self.width
        dtype = self.first.dtype
        block = slice(head * width, (head + 1) * width)
        first = self.first[:, block] / jnp.sqrt(
            jnp.asarray(self.first.shape[0], dtype=dtype)
        )
        hidden = self.activation_scale * jax.nn.silu(scalars @ first)
        second = self.second[block, head] / jnp.sqrt(
            jnp.asarray(self.second.shape[0], dtype=dtype)
        )
        return hidden @ second


@final
class MACEInvariantReadoutProduct(StrictModule):
    """Separate one-interaction invariant product ahead of the nonlinear readout.

    The hidden irreps of the single product are reshaped per channel, contracted
    by their own species-conditioned symmetric product into one scalar per
    channel, mixed by the product linear, and only then read out. It has no
    self-connection.
    """

    contraction: SymmetricContraction
    linear: O3IrrepLinear
    channel_layout: O3IrrepLayout

    def __init__(
        self,
        contraction: SymmetricContraction,
        linear: O3IrrepLinear,
        channel_layout: O3IrrepLayout,
        /,
    ) -> None:
        if not isinstance(contraction, SymmetricContraction):
            raise TypeError("contraction must be a SymmetricContraction.")
        if not isinstance(linear, O3IrrepLinear):
            raise TypeError("linear must be an O3IrrepLinear.")
        if not isinstance(channel_layout, O3IrrepLayout):
            raise TypeError("channel_layout must be an O3IrrepLayout.")
        output = contraction.plan.output_layout
        if len(output.blocks) != 1 or (
            output.blocks[0].degree,
            output.blocks[0].parity,
        ) != (0, 1):
            raise ValueError(
                "The invariant readout product must output one scalar block."
            )
        if contraction.plan.input_layout.layout_id != channel_layout.layout_id:
            raise ValueError(
                "The invariant readout product input layout is inconsistent."
            )
        if (
            len(linear.in_layout.blocks) != 1
            or linear.in_layout.packed_size != contraction.plan.channel_count
            or linear.out_layout.layout_id != linear.in_layout.layout_id
        ):
            raise ValueError("The invariant product linear must map C x 0e to C x 0e.")
        self.contraction = contraction
        self.linear = linear
        self.channel_layout = channel_layout

    @property
    def correlation(self) -> int:
        return len(self.contraction.weights[0])

    def __call__(self, channel_features: Array, species: Array, /) -> Array:
        """``channel_features[N, C, D]`` and ``species[N]`` -> scalars ``[N, C]``."""
        contracted = self.contraction(channel_features, species)
        return self.linear(contracted[..., 0])


@final
class MACEEnergyReference(StrictModule, NonTrainableState):
    """Fixed atomic reference energies, head selection, and interaction scaling."""

    __strict_contract__ = True

    atomic_energies: Float[MACEHeadDim, MACESpeciesDim]
    scale: Float[MACEHeadDim]
    shift: Float[MACEHeadDim]
    heads: tuple[str, ...] = eqx.field(static=True)
    head: int = eqx.field(static=True)
    scaling: MACEEnergyScaling = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        atomic_energies: ArrayLike,
        /,
        *,
        heads: Sequence[str],
        head: str,
        scaling: MACEEnergyScaling,
        scale: ArrayLike | None,
        shift: ArrayLike | None,
        species_count: int,
        dtype: DTypeLike,
    ) -> None:
        names = tuple(canonical_identifier(value, "heads") for value in heads)
        if not names or len(set(names)) != len(names):
            raise ValueError("MACE heads must be distinct, non-empty identifiers.")
        selected = canonical_identifier(head, "head")
        if selected not in names:
            raise ValueError(f"MACE head {selected!r} is not a declared head.")
        energies = np.asarray(
            as_host_array(atomic_energies, HostFloat64[AnyShape], "atomic_energies")
        )
        if energies.ndim == 1:
            energies = energies[None, :]
        if energies.shape != (len(names), species_count):
            raise ValueError("atomic_energies must have shape (heads, species).")
        match scaling:
            case "unscaled":
                if scale is not None or shift is not None:
                    raise ValueError("Unscaled MACE energies take no scale or shift.")
                scale_ = np.ones((len(names),), dtype=np.float64)
                shift_ = np.zeros((len(names),), dtype=np.float64)
            case "scale-shift":
                if scale is None or shift is None:
                    raise ValueError("Scale-shift MACE energies require scale and shift.")
                scale_ = _head_values(scale, len(names), "energy_scale")
                shift_ = _head_values(shift, len(names), "energy_shift")
            case unreachable:
                assert_never(unreachable)
        if not (
            np.all(np.isfinite(energies))
            and np.all(np.isfinite(scale_))
            and np.all(np.isfinite(shift_))
        ):
            raise ValueError(
                "MACE reference energies, scales, and shifts must be finite."
            )
        dtype_ = np.dtype(dtype)
        self.atomic_energies = jnp.asarray(energies.astype(dtype_))
        self.scale = jnp.asarray(scale_.astype(dtype_))
        self.shift = jnp.asarray(shift_.astype(dtype_))
        self.heads = names
        self.head = names.index(selected)
        self.scaling = scaling
        self.reference_id = canonical_fingerprint(
            {
                "kind": "mace-energy-reference",
                "heads": list(names),
                "head": selected,
                "scaling": scaling,
                "atomic_energies": energies.astype(dtype_).tolist(),
                "scale": scale_.astype(dtype_).tolist(),
                "shift": shift_.astype(dtype_).tolist(),
            }
        )

    def atom_energy(self, species: Array, interaction: Array, /) -> Array:
        """``E0[head, s] + scale[head] * interaction + shift[head]`` per atom."""
        reference = self.atomic_energies[self.head][species]
        return reference + self.scale[self.head] * interaction + self.shift[self.head]


def _head_values(value: ArrayLike, count: int, name: str, /) -> np.ndarray:
    values = np.asarray(as_host_array(value, HostFloat64[AnyShape], name)).reshape((-1,))
    if values.shape == (1,) and count > 1:
        raise ValueError(f"{name} must declare one value per head.")
    if values.shape != (count,):
        raise ValueError(f"{name} must have one value per head.")
    return values


@final
class MACEPairRepulsion(StrictModule, NonTrainableState):
    """Universal ZBL pair repulsion with the source covalent-radius envelope.

    Each directed edge contributes half of ``k Z_u Z_v / r phi(r / a)`` times the
    polynomial envelope on ``r < R_u + R_v``; the two directions of one pair sum
    to the full pair energy. Constants are in eV and angstrom and converted once
    into the model's declared units. Screening coefficients, screening-length
    constants and radii are fixed numerical leaves: native construction uses the
    universal values, while a reconstructed source keeps its stored realization
    (for example float32-rounded buffers), admitted only within float32 rounding
    of the universal constants.
    """

    __strict_contract__ = True

    atomic_numbers: Float[MACESpeciesDim]
    covalent_radii: Float[MACESpeciesDim]
    coefficients: Float[_ZBLCoefficientDim]
    exponents: Float[_ZBLCoefficientDim]
    screening: Float[_ZBLScreeningDim]
    envelope: PolynomialCutoff
    length_to_angstrom: float = eqx.field(static=True)
    energy_from_electronvolt: float = eqx.field(static=True)
    constants_id: str = eqx.field(static=True)

    def __init__(
        self,
        atomic_numbers: Sequence[int],
        scale: AtomisticScaleContract,
        /,
        *,
        power: int,
        dtype: DTypeLike,
        covalent_radii: ArrayLike | None = None,
        coefficients: ArrayLike | None = None,
        screening: ArrayLike | None = None,
    ) -> None:
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        dtype_ = np.dtype(dtype)
        numbers = tuple(atomic_numbers)
        radii = _realized(
            covalent_radii, covalent_radii_angstrom(numbers), "covalent_radii"
        )
        realized_coefficients = _realized(
            coefficients, np.asarray(_ZBL_COEFFICIENTS), "coefficients"
        )
        realized_screening = _realized(
            screening,
            np.asarray((_ZBL_SCREENING_EXPONENT, _ZBL_SCREENING_PREFACTOR)),
            "screening",
        )
        self.atomic_numbers = jnp.asarray(np.asarray(numbers, dtype=dtype_))
        self.covalent_radii = jnp.asarray(radii.astype(dtype_))
        self.coefficients = jnp.asarray(realized_coefficients.astype(dtype_))
        self.exponents = jnp.asarray(np.asarray(_ZBL_EXPONENTS, dtype=dtype_))
        self.screening = jnp.asarray(realized_screening.astype(dtype_))
        self.envelope = PolynomialCutoff(1.0, power)
        self.length_to_angstrom = float(conversion_factor(scale.length_unit, ANGSTROM))
        self.energy_from_electronvolt = float(
            conversion_factor(ELECTRONVOLT, scale.energy_unit)
        )
        self.constants_id = canonical_fingerprint(
            {
                "kind": "mace-zbl-constants",
                "atomic_numbers": list(numbers),
                "covalent_radii": radii.astype(dtype_).tolist(),
                "coefficients": realized_coefficients.astype(dtype_).tolist(),
                "screening": realized_screening.astype(dtype_).tolist(),
                "power": power,
                "length_to_angstrom": self.length_to_angstrom,
                "energy_from_electronvolt": self.energy_from_electronvolt,
            }
        )

    def edge_energy(
        self, distance: Array, sender_species: Array, receiver_species: Array, /
    ) -> Array:
        """Receiver share of one directed edge's ZBL energy in model units."""
        radius = distance * self.length_to_angstrom
        sender_z = self.atomic_numbers[sender_species]
        receiver_z = self.atomic_numbers[receiver_species]
        exponent, prefactor = self.screening[0], self.screening[1]
        screening = (
            prefactor * _ZBL_BOHR_ANGSTROM / (sender_z**exponent + receiver_z**exponent)
        )
        reduced = radius / screening
        phi = jnp.sum(
            self.coefficients * jnp.exp(-self.exponents * reduced[..., None]), axis=-1
        )
        coulomb = _ZBL_COULOMB_EV_ANGSTROM * sender_z * receiver_z / radius * phi
        extent = (
            self.covalent_radii[sender_species] + self.covalent_radii[receiver_species]
        )
        energy = 0.5 * coulomb * self.envelope(radius / extent)
        return energy * self.energy_from_electronvolt


# Imported realizations of universal constants may carry float32 rounding.
_FLOAT32_ROUNDING = 1.0e-6


def _realized(value: ArrayLike | None, universal: np.ndarray, name: str, /) -> np.ndarray:
    if value is None:
        return universal.astype(np.float64)
    realized = np.asarray(as_host_array(value, HostFloat64[AnyShape], name))
    if realized.shape != universal.shape or not np.allclose(
        realized, universal, rtol=_FLOAT32_ROUNDING, atol=0.0
    ):
        raise ValueError(f"ZBL {name} differ from the universal ZBL constants.")
    return realized


__all__ = [
    "covalent_radii_angstrom",
    "MACEEnergyReference",
    "MACEEnergyScaling",
    "MACEInvariantReadoutProduct",
    "MACELinearReadout",
    "MACENonlinearReadout",
    "MACEPairRepulsion",
]

for _artifact in (
    MACEEnergyReference,
    MACEInvariantReadoutProduct,
    MACELinearReadout,
    MACENonlinearReadout,
    MACEPairRepulsion,
):
    register_artifact_value(
        f"phydrax.nn.atomistic.internal:{_artifact.__name__}", _artifact
    )
del _artifact
