#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed reconstruction of native MACE and MACELES potentials from source tensors.

The input is the exact source parameter inventory keyed by the source e3nn
(``mul_ir`` layout, non-cuEquivariance) state-dictionary names, plus the
explicit basis evidence of the source real-harmonic convention. Reconstruction
preserves the original parameterization: linear, radial, self-connection,
product, readout and embedding weights are the source leaves (only transposed,
reshaped or split into native blocks), the symmetric products keep their
original ``W`` with the source ``U`` transformed into the native basis, and
each coupling path records the sign relating the source coupling to the native
one (including the odd-sum paths of the source SO(3) convention). Every source
tensor is consumed exactly once; a missing or unadmitted name refuses.

Admitted families: Bessel, Gaussian and Chebyshev radial bases; the Agnesi and
Soft distance transforms (source covalent-radius tables indexed by species, in
angstrom); O(3) and SO(3) parity; uniform edge irreps (optionally also in the
first layer); the standard, attention-residual and residual-nonlinear
interactions; linear, nonlinear and nonlinear-bias readouts with optionally
kept last-layer irreps; the joint embedding and its embedding readout; and the
MACELES latent readouts. Source normalization constants are not re-derived
here; the caller verifies them against the provider before calling. This
module never imports a provider or deserializes anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, Literal

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._validation import canonical_identifier, positive_integer
from ...atomistic._latent_electrostatics import (
    LatentEwaldDefinition,
    LatentEwaldResources,
)
from ...atomistic._types import AtomisticPrecisionPolicy, AtomisticScaleContract
from ...ein import contract
from ...sparse._streamed import StreamedRelationPlan
from ...special import RealCartesianHarmonics
from ...typing import AnyShape, as_host_array, HostFloat64
from ...units import ANGSTROM, conversion_factor
from ..operator.layers import O3IrrepLinear, O3TensorProduct
from ..operator.representations import o3_real_coupling, O3IrrepBlock, O3IrrepLayout
from ._mace import (
    _compute_dtype,
    mace_gate,
    mace_radial_widths,
    mace_tensor_product_plan,
    MACEArchitecture,
    MACEPotential,
)
from ._mace_embedding import MACEJointEmbedding
from ._mace_interaction import (
    ATTENTION_PROJECTION_WIDTH,
    convolution_paths,
    gate_layout,
    interaction_family,
    interaction_traits,
    linear_paths,
    MACEAnyInteraction,
    MACEAttentionInteraction,
    MACEEdgeGeometry,
    MACEInteraction,
    MACELayer,
    MACENonlinearInteraction,
    MACEPlainLinear,
    MACEProduct,
    MACESelfConnection,
    NONLINEAR_DENSITY_WIDTH,
    self_connection_paths,
)
from ._mace_les import MACELESLatentReadout, MACELESPotential
from ._mace_readout import (
    covalent_radii_angstrom,
    MACEInvariantReadoutProduct,
    MACELinearReadout,
    MACENonlinearBiasReadout,
    MACENonlinearReadout,
    MACEPairRepulsion,
    MACEReadout,
)
from ._radial import (
    AgnesiTransform,
    BesselRadialBasis,
    ChebyshevRadialBasis,
    DistanceTransform,
    GaussianRadialBasis,
    LayerNormRadialMLP,
    PolynomialCutoff,
    RadialBasis,
    RadialEmbedding,
    RadialMLP,
    RadialPostprocess,
    SoftTransform,
)
from ._symmetric_contraction import (
    SymmetricContraction,
    SymmetricContractionBasis,
    SymmetricContractionPlan,
)


# Variance epsilon of every ``torch.nn.LayerNorm`` in the source
# ``mace.modules.radial.RadialMLP`` (the torch default the source never
# overrides). The converter verifies the provider modules against it.
MACE_SOURCE_LAYER_NORM_EPSILON = 1.0e-5

_ORTHOGONALITY_TOLERANCE = 1.0e-10
_COUPLING_TOLERANCE = 1.0e-6
_ZBL_SOURCE_CONSTANTS = (0.1818, 0.5099, 0.2802, 0.02817)
_ZBL_SOURCE_SCREENING = (0.300, 0.4543)
# Floor of the owner's default equivariance residual gate for exact float64 bases.
_BASIS_TOLERANCE_FLOOR = 1.0e-9


def _basis_tolerance(dtype: np.dtype, /) -> float:
    """Equivariance gate of a stored source ``U`` at its own rounding.

    A float32-stored basis is exact only to float32 rounding; the gate scales
    with the source epsilon so stored rounding is admitted while a wrong basis
    transform (an order-one residual) still refuses.
    """
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("Source U tensors must be floating point.")
    return max(_BASIS_TOLERANCE_FLOOR, 64.0 * float(np.finfo(dtype).eps))


class _SourceInventory:
    """Exact-inventory reader: every key is consumed once, extras refuse."""

    def __init__(self, tensors: Mapping[str, ArrayLike], /) -> None:
        if not isinstance(tensors, Mapping):
            raise TypeError("tensors must be a mapping of source tensor names.")
        self._tensors = dict(tensors)
        self._consumed: set[str] = set()

    def _take(self, name: str, /) -> np.ndarray:
        if name not in self._tensors:
            raise ValueError(f"Source tensor {name!r} is missing.")
        if name in self._consumed:
            raise RuntimeError(f"Source tensor {name!r} was consumed twice.")
        self._consumed.add(name)
        value = np.asarray(self._tensors[name])
        if value.dtype.kind not in "fiu":
            raise TypeError(f"Source tensor {name!r} is not numeric.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Source tensor {name!r} must be finite.")
        return value

    def array(self, name: str, shape: tuple[int, ...] | None = None, /) -> np.ndarray:
        """A flat source tensor (e3nn flattened weights) viewed in ``shape``."""
        value = self._take(name)
        if shape is not None:
            if value.size != int(np.prod(shape, dtype=np.int64)):
                raise ValueError(
                    f"Source tensor {name!r} has {value.size} values; expected {shape}."
                )
            value = value.reshape(shape)
        return value

    def exact(self, name: str, shape: tuple[int, ...], /) -> np.ndarray:
        """A shaped source tensor (torch ``Linear``/``LayerNorm``/``Embedding``)."""
        value = self._take(name)
        if value.shape != shape:
            raise ValueError(
                f"Source tensor {name!r} has shape {value.shape}; expected {shape}."
            )
        return value

    def scalar(self, name: str, /) -> np.ndarray:
        return self.array(name, ())

    def finish(self) -> None:
        extra = sorted(set(self._tensors) - self._consumed)
        if extra:
            raise ValueError(f"Unadmitted source tensors: {extra}.")


def _matches(
    source: np.ndarray,
    expected: float | Sequence[float] | Sequence[Sequence[float]],
    /,
) -> bool:
    """Equality of a declared value with a source value at the source precision."""
    return bool(np.array_equal(source, np.asarray(expected, dtype=source.dtype)))


def _transforms(
    values: Sequence[ArrayLike], maximum_degree: int, /
) -> tuple[np.ndarray, ...]:
    transforms = tuple(
        np.asarray(as_host_array(value, HostFloat64[AnyShape], "degree_transforms"))
        for value in values
    )
    if len(transforms) <= maximum_degree:
        raise ValueError(f"degree_transforms must cover degrees 0..{maximum_degree}.")
    for degree, matrix in enumerate(transforms):
        size = 2 * degree + 1
        if matrix.shape != (size, size):
            raise ValueError(f"degree_transforms[{degree}] must be {size} x {size}.")
        if np.max(np.abs(matrix @ matrix.T - np.eye(size))) > _ORTHOGONALITY_TOLERANCE:
            raise ValueError(f"degree_transforms[{degree}] must be orthogonal.")
    return transforms


def _block_transform(
    transforms: Sequence[np.ndarray], layout: O3IrrepLayout, /
) -> np.ndarray:
    size = layout.packed_size
    matrix = np.zeros((size, size), dtype=np.float64)
    start = 0
    for block in layout.blocks:
        stop = start + block.dimension
        matrix[start:stop, start:stop] = transforms[block.degree]
        start = stop
    return matrix


def _source_coupling(
    couplings: Mapping[tuple[int, int, int], ArrayLike],
    transforms: Sequence[np.ndarray],
    degrees: tuple[int, int, int],
    /,
) -> tuple[float, np.ndarray]:
    """Native-basis source coupling ``sqrt(2l3+1) w3j`` and its native sign.

    The table keeps the source realization (for example float32-rounded
    buffers); it is bound as the executed coefficients of its path.
    """
    left, right, output = degrees
    if degrees not in couplings:
        raise ValueError(f"couplings lacks the source coupling {degrees}.")
    source = np.asarray(
        as_host_array(couplings[degrees], HostFloat64[AnyShape], "couplings")
    )
    if source.shape != (2 * left + 1, 2 * right + 1, 2 * output + 1):
        raise ValueError(f"Source coupling {degrees} has shape {source.shape}.")
    effective = np.sqrt(2.0 * output + 1.0) * contract(
        "oc,ia,jb,abc->oij",
        transforms[output],
        transforms[left],
        transforms[right],
        source,
    )
    native = o3_real_coupling(left, right, output).dense()
    sign = 1.0 if float(np.sum(effective * native)) >= 0.0 else -1.0
    if np.max(np.abs(effective - sign * native)) > _COUPLING_TOLERANCE:
        raise ValueError(
            f"Source coupling {degrees} is not a signed native coupling in the "
            "declared basis; the basis evidence is inconsistent."
        )
    return sign, effective


def _linear_weights(
    flat: np.ndarray, in_layout: O3IrrepLayout, out_layout: O3IrrepLayout, /
) -> tuple[np.ndarray, ...]:
    """Split a source equivariant-linear vector into native per-output blocks.

    Source instructions run input-major over matching ``(l, p)`` pairs and store
    ``[mul_in, mul_out]``; native output blocks store ``[mul_out, fan_in]`` with
    inputs concatenated in input-layout order. Every output block must be
    matched; unmatched inputs carry no weights.
    """
    parts: list[list[np.ndarray]] = [[] for _ in out_layout.blocks]
    offset = 0
    for source in in_layout.blocks:
        for index, target in enumerate(out_layout.blocks):
            if (source.degree, source.parity) != (target.degree, target.parity):
                continue
            size = source.multiplicity * target.multiplicity
            block = flat[offset : offset + size].reshape(
                (source.multiplicity, target.multiplicity)
            )
            parts[index].append(block.T)
            offset += size
    if offset != flat.size or any(not blocks for blocks in parts):
        raise ValueError("Source linear weights do not match the declared irreps.")
    return tuple(np.concatenate(blocks, axis=1) for blocks in parts)


def _linear_count(in_layout: O3IrrepLayout, out_layout: O3IrrepLayout, /) -> int:
    return sum(
        source.multiplicity * target.multiplicity
        for source in in_layout.blocks
        for target in out_layout.blocks
        if (source.degree, source.parity) == (target.degree, target.parity)
    )


def _matched_outputs(
    in_layout: O3IrrepLayout, out_layout: O3IrrepLayout, /
) -> O3IrrepLayout:
    """Output blocks of an e3nn ``Linear`` that receive at least one input path."""
    matched = sorted({right for _, right in linear_paths(in_layout, out_layout)})
    if not matched:
        raise ValueError("A source linear map has no matching irreps.")
    return O3IrrepLayout(tuple(out_layout.blocks[index] for index in matched))


class _Reconstruction:
    """One source reconstruction: inventory, basis evidence, and target dtype."""

    def __init__(
        self,
        architecture: MACEArchitecture,
        scale: AtomisticScaleContract,
        inventory: _SourceInventory,
        transforms: tuple[np.ndarray, ...],
        couplings: Mapping[tuple[int, int, int], ArrayLike],
        source_id: str,
        dtype: np.dtype,
        maximum_entries: int,
        /,
    ) -> None:
        self.architecture = architecture
        self.scale = scale
        self.inventory = inventory
        self.transforms = transforms
        self.couplings = couplings
        self.source_id = source_id
        self.dtype = dtype
        self.maximum_entries = maximum_entries

    def cast(self, value: np.ndarray, /) -> np.ndarray:
        return np.ascontiguousarray(value).astype(self.dtype)

    def linear(
        self, name: str, in_layout: O3IrrepLayout, out_layout: O3IrrepLayout, /
    ) -> O3IrrepLinear:
        """An e3nn ``Linear`` whose every output block has a matching input."""
        flat = self.inventory.array(name, (_linear_count(in_layout, out_layout),))
        return O3IrrepLinear(
            in_layout,
            out_layout,
            weights=tuple(
                self.cast(block) for block in _linear_weights(flat, in_layout, out_layout)
            ),
            dtype=self.dtype,
        )

    def plain_linear(
        self, name: str, in_layout: O3IrrepLayout, out_layout: O3IrrepLayout, /
    ) -> MACEPlainLinear:
        """An e3nn ``Linear`` whose unmatched output blocks are absent (zero)."""
        matched = _matched_outputs(in_layout, out_layout)
        flat = self.inventory.array(name, (_linear_count(in_layout, out_layout),))
        return MACEPlainLinear(
            in_layout,
            out_layout,
            weights=tuple(
                self.cast(block) for block in _linear_weights(flat, in_layout, matched)
            ),
            dtype=self.dtype,
        )

    def radial_network(
        self,
        prefix: str,
        widths: tuple[int, ...],
        /,
        *,
        postprocess: RadialPostprocess = "none",
    ) -> RadialMLP:
        """An e3nn ``FullyConnectedNet`` (``layer{k}.weight`` stored ``[in, out]``)."""
        return RadialMLP(
            tuple(
                self.cast(
                    self.inventory.array(
                        f"{prefix}.layer{index}.weight", (fan_in, fan_out)
                    )
                )
                for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:]))
            ),
            activation_scale=self.architecture.radial_activation_scale,
            postprocess=postprocess,
        )

    def layer_norm_network(
        self,
        prefix: str,
        widths: tuple[int, ...],
        /,
        *,
        postprocess: RadialPostprocess = "none",
    ) -> LayerNormRadialMLP:
        """A source ``RadialMLP``: ``net.{3k}`` Linear, ``net.{3k+1}`` LayerNorm.

        Torch ``Linear.weight`` is ``[out, in]``; the native weight is its
        transpose. Hidden layers carry the LayerNorm gain and shift.
        """
        inventory = self.inventory
        weights, biases, scales, shifts = [], [], [], []
        count = len(widths) - 1
        for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
            position = 3 * index
            weights.append(
                self.cast(
                    inventory.exact(
                        f"{prefix}.net.{position}.weight", (fan_out, fan_in)
                    ).T
                )
            )
            biases.append(
                self.cast(inventory.exact(f"{prefix}.net.{position}.bias", (fan_out,)))
            )
            if index < count - 1:
                scales.append(
                    self.cast(
                        inventory.exact(f"{prefix}.net.{position + 1}.weight", (fan_out,))
                    )
                )
                shifts.append(
                    self.cast(
                        inventory.exact(f"{prefix}.net.{position + 1}.bias", (fan_out,))
                    )
                )
        return LayerNormRadialMLP(
            weights,
            biases,
            scales,
            shifts,
            epsilon=MACE_SOURCE_LAYER_NORM_EPSILON,
            postprocess=postprocess,
        )

    def radii(self, name: str, family: str, /) -> np.ndarray:
        """Source covalent radii of the model species, in the model length unit.

        The source tables are indexed by atomic number in angstrom; only the
        model species are ever read (tables of other elements differ between
        provider releases).
        """
        architecture = self.architecture
        table = self.inventory.array(name)
        if table.ndim != 1 or table.shape[0] <= max(architecture.species):
            raise ValueError(f"Source {family} radii do not cover the model species.")
        selected = table[np.asarray(architecture.species)].astype(np.float64)
        if np.any(selected <= 0.0):
            raise ValueError(
                f"Source {family} radii of the model species must be positive."
            )
        return selected * float(conversion_factor(ANGSTROM, self.scale.length_unit))

    def basis(self) -> RadialBasis:
        architecture = self.architecture
        inventory = self.inventory
        count = architecture.radial_basis_count
        prefix = "radial_embedding.bessel_fn"
        match architecture.radial_basis:
            case "bessel":
                if not _matches(inventory.scalar(f"{prefix}.r_max"), architecture.cutoff):
                    raise ValueError("Source Bessel radius differs from the declaration.")
                frequencies = inventory.array(f"{prefix}.bessel_weights", (count,))
                prefactor = inventory.scalar(f"{prefix}.prefactor")
                return BesselRadialBasis(
                    frequencies.astype(np.float64),
                    float(prefactor),
                    architecture.cutoff,
                    dtype=self.dtype,
                )
            case "gaussian":
                centers = inventory.array(f"{prefix}.gaussian_weights", (count,))
                return GaussianRadialBasis(
                    centers.astype(np.float64), architecture.cutoff, dtype=self.dtype
                )
            case "chebyshev":
                orders = inventory.exact(f"{prefix}.n", (1, count))
                if not _matches(orders, [[float(n) for n in range(1, count + 1)]]):
                    raise ValueError("Source Chebyshev orders are not 1..count.")
                return ChebyshevRadialBasis(architecture.cutoff, count, dtype=self.dtype)
            case unreachable:
                assert_never(unreachable)

    def transform(self) -> DistanceTransform | None:
        architecture = self.architecture
        inventory = self.inventory
        prefix = "radial_embedding.distance_transform"
        match architecture.distance_transform:
            case "none":
                return None
            case "agnesi":
                parameters = architecture.agnesi_parameters
                if parameters is None:
                    raise RuntimeError("The Agnesi transform lost its parameters.")
                values = tuple(
                    inventory.scalar(f"{prefix}.{name}") for name in ("a", "q", "p")
                )
                if not all(
                    _matches(value, declared)
                    for value, declared in zip(values, parameters, strict=True)
                ):
                    raise ValueError(
                        "Source Agnesi parameters differ from the declaration."
                    )
                a, q, p = parameters
                return AgnesiTransform(
                    self.radii(f"{prefix}.covalent_radii", "Agnesi"),
                    atomic_numbers=architecture.species,
                    a=a,
                    q=q,
                    p=p,
                    dtype=self.dtype,
                )
            case "soft":
                alpha = architecture.soft_alpha
                if alpha is None:
                    raise RuntimeError("The Soft transform lost its steepness.")
                if not _matches(inventory.scalar(f"{prefix}.alpha"), alpha):
                    raise ValueError(
                        "Source Soft steepness differs from the declaration."
                    )
                return SoftTransform(
                    self.radii(f"{prefix}.covalent_radii", "Soft"),
                    atomic_numbers=architecture.species,
                    alpha=alpha,
                    dtype=self.dtype,
                )
            case unreachable:
                assert_never(unreachable)

    def geometry(self) -> MACEEdgeGeometry:
        architecture = self.architecture
        inventory = self.inventory
        radii = (
            inventory.scalar("r_max"),
            inventory.scalar("radial_embedding.cutoff_fn.r_max"),
        )
        if not all(_matches(value, architecture.cutoff) for value in radii):
            raise ValueError("Source cutoff radius differs from the declaration.")
        if (
            int(inventory.scalar("radial_embedding.cutoff_fn.p"))
            != architecture.cutoff_power
        ):
            raise ValueError(
                "Source polynomial cutoff power differs from the declaration."
            )
        return MACEEdgeGeometry(
            RadialEmbedding(
                self.basis(),
                PolynomialCutoff(architecture.cutoff, architecture.cutoff_power),
                transform=self.transform(),
            ),
            RealCartesianHarmonics(
                architecture.edge_degree, normalization="fully_normalized"
            ),
            cutoff_placement=architecture.cutoff_placement,
        )

    def contraction(
        self,
        prefix: str,
        input_degree: int,
        output_layout: O3IrrepLayout,
        correlation: int,
        /,
    ) -> SymmetricContraction:
        """Original ``W`` and source ``U`` (in the native basis) of one product."""
        architecture = self.architecture
        inputs = architecture.layout(1, input_degree)
        input_transform = _block_transform(self.transforms, inputs)
        species = architecture.contraction_species_count
        channels = architecture.channel_count
        bases: list[tuple[SymmetricContractionBasis, ...]] = []
        weights: list[tuple[np.ndarray, ...]] = []
        for index, block in enumerate(output_layout.blocks):
            base = f"{prefix}.symmetric_contractions.contractions.{index}"
            output = O3IrrepBlock(block.name, block.degree, block.parity)
            block_bases = []
            for order in range(1, correlation + 1):
                tensor = self.inventory.array(f"{base}.U_matrix_{order}")
                if block.degree == 0 and tensor.ndim == order + 1:
                    tensor = tensor[None]
                block_bases.append(
                    SymmetricContractionBasis.from_source_tensor(
                        tensor,
                        inputs,
                        output,
                        order,
                        source_id=f"{self.source_id}:{base}:U{order}",
                        input_transform=input_transform,
                        output_transform=self.transforms[block.degree],
                        maximum_entries=self.maximum_entries,
                        equivariance_tolerance=_basis_tolerance(tensor.dtype),
                        dtype=self.dtype,
                    )
                )
            block_weights: list[np.ndarray] = []
            for order, basis in enumerate(block_bases, start=1):
                name = (
                    f"{base}.weights_max"
                    if order == correlation
                    else f"{base}.weights.{correlation - 1 - order}"
                )
                block_weights.append(
                    self.cast(
                        self.inventory.array(name, (species, basis.path_count, channels))
                    )
                )
            bases.append(tuple(block_bases))
            weights.append(tuple(block_weights))
        plan = SymmetricContractionPlan(
            inputs,
            architecture.layout(1, output_layout.blocks[-1].degree),
            bases,
            species_count=species,
            channel_count=channels,
        )
        return SymmetricContraction(plan, weights)

    def coupling(self, index: int, /) -> tuple[O3TensorProduct, O3IrrepLayout]:
        """Signed source coupling of one layer and its message layout."""
        architecture = self.architecture
        edges = architecture.edge_layout(index)
        harmonics = architecture.harmonics_layout()
        message, paths = convolution_paths(edges, harmonics, architecture.target_layout())
        degrees = {block.name: block.degree for block in edges.blocks}
        degrees.update({block.name: block.degree for block in harmonics.blocks})
        degrees.update({block.name: block.degree for block in message.blocks})
        bound = tuple(
            _source_coupling(
                self.couplings,
                self.transforms,
                (degrees[left], degrees[right], degrees[output]),
            )
            for left, right, output in paths
        )
        plan = mace_tensor_product_plan(
            architecture, index, path_scales=tuple(sign for sign, _ in bound)
        )
        coupling = O3TensorProduct(
            plan,
            internal_weights=False,
            dtype=self.dtype,
            coefficients=tuple(table for _, table in bound),
            coefficient_tolerance=_COUPLING_TOLERANCE,
        )
        return coupling, message

    def self_connection(
        self,
        name: str,
        connection_in: O3IrrepLayout,
        connection_out: O3IrrepLayout,
        /,
    ) -> MACESelfConnection:
        """The species-conditioned source ``skip_tp`` (``uvw`` paths)."""
        species = self.architecture.species_count
        connection_paths = self_connection_paths(connection_in, connection_out)
        flat = self.inventory.array(
            name,
            (
                sum(
                    connection_in.blocks[left].multiplicity
                    * species
                    * connection_out.blocks[right].multiplicity
                    for left, right in connection_paths
                ),
            ),
        )
        weights = []
        offset = 0
        for left, right in connection_paths:
            shape = (
                connection_in.blocks[left].multiplicity,
                species,
                connection_out.blocks[right].multiplicity,
            )
            size = int(np.prod(shape))
            weights.append(self.cast(flat[offset : offset + size].reshape(shape)))
            offset += size
        return MACESelfConnection(
            connection_in, connection_out, weights, species_count=species
        )

    def interaction(self, index: int, /) -> MACEAnyInteraction:
        architecture = self.architecture
        prefix = f"interactions.{index}"
        kind = architecture.interactions[index]
        inputs = architecture.input_layout(index)
        outputs = architecture.output_layout(index)
        edges = architecture.edge_layout(index)
        target = architecture.target_layout()
        widths = mace_radial_widths(architecture, index)
        linear_up = self.linear(f"{prefix}.linear_up.weight", inputs, edges)
        coupling, message = self.coupling(index)
        match interaction_family(kind):
            case "standard":
                residual, density_normalized = interaction_traits(kind)
                return MACEInteraction(
                    linear_up,
                    coupling,
                    self.radial_network(f"{prefix}.conv_tp_weights", widths),
                    self.linear(f"{prefix}.linear.weight", message, target),
                    self.self_connection(
                        f"{prefix}.skip_tp.weight",
                        inputs if residual else target,
                        outputs if residual else target,
                    ),
                    kind=kind,
                    average_neighbor_count=architecture.average_neighbor_count,
                    density=self.radial_network(
                        f"{prefix}.density_fn",
                        (architecture.radial_basis_count, 1),
                        postprocess="tanh-square",
                    )
                    if density_normalized
                    else None,
                )
            case "attention":
                return MACEAttentionInteraction(
                    linear_up,
                    self.linear(
                        f"{prefix}.linear_down.weight",
                        inputs,
                        architecture.layout(ATTENTION_PROJECTION_WIDTH, 0),
                    ),
                    coupling,
                    self.radial_network(f"{prefix}.conv_tp_weights", widths),
                    self.linear(f"{prefix}.linear.weight", message, target),
                    self.plain_linear(f"{prefix}.skip_linear.weight", inputs, outputs),
                    average_neighbor_count=architecture.average_neighbor_count,
                )
            case "nonlinear":
                nonlinear = gate_layout(target)
                embedding = (architecture.species_count, architecture.channel_count)
                return MACENonlinearInteraction(
                    linear_up,
                    self.cast(
                        self.inventory.array(
                            f"{prefix}.source_embedding.weight", embedding
                        )
                    ),
                    self.cast(
                        self.inventory.array(
                            f"{prefix}.target_embedding.weight", embedding
                        )
                    ),
                    coupling,
                    self.layer_norm_network(f"{prefix}.conv_tp_weights", widths),
                    self.layer_norm_network(
                        f"{prefix}.density_fn",
                        (widths[0], NONLINEAR_DENSITY_WIDTH, 1),
                        postprocess="tanh-square",
                    ),
                    self.linear(f"{prefix}.linear_1.weight", message, nonlinear),
                    self.plain_linear(f"{prefix}.linear_res.weight", edges, nonlinear),
                    mace_gate(architecture),
                    self.linear(f"{prefix}.linear_2.weight", target, target),
                    self.plain_linear(f"{prefix}.skip_tp.weight", inputs, outputs),
                    self.cast(self.inventory.scalar(f"{prefix}.alpha")),
                    self.cast(self.inventory.scalar(f"{prefix}.beta")),
                )
            case unreachable:
                assert_never(unreachable)

    def readout(
        self, prefix: str, kind: Literal["linear", "nonlinear"], /
    ) -> MACEReadout:
        """The readout block at ``prefix`` (``readouts.k`` or ``les_readouts.k``)."""
        architecture = self.architecture
        channels = architecture.channel_count
        heads = len(architecture.heads)
        inventory = self.inventory
        match kind:
            case "linear":
                return MACELinearReadout(
                    self.cast(
                        inventory.array(f"{prefix}.linear.weight", (channels, heads))
                    )
                )
            case "nonlinear":
                width = architecture.readout_width
                scale = architecture.readout_activation_scale
                if width is None or scale is None:
                    raise RuntimeError(
                        "A nonlinear readout lost its declared width or scale."
                    )
                hidden = width * heads
                # Only the scalar input channels carry ``linear_1`` paths, also
                # when the last layer keeps its higher irreps.
                first = self.cast(
                    inventory.array(f"{prefix}.linear_1.weight", (channels, hidden))
                )
                match architecture.readout_variant:
                    case "nonlinear":
                        return MACENonlinearReadout(
                            first,
                            self.cast(
                                inventory.array(
                                    f"{prefix}.linear_2.weight", (hidden, heads)
                                )
                            ),
                            activation_scale=scale,
                        )
                    case "nonlinear-bias":
                        return MACENonlinearBiasReadout(
                            first,
                            self.cast(
                                inventory.array(
                                    f"{prefix}.linear_mid.weight", (hidden, hidden)
                                )
                            ),
                            self.cast(
                                inventory.exact(f"{prefix}.linear_mid.bias", (hidden,))
                            ),
                            self.cast(
                                inventory.array(
                                    f"{prefix}.linear_2.weight", (hidden, heads)
                                )
                            ),
                            self.cast(
                                inventory.exact(f"{prefix}.linear_2.bias", (heads,))
                            ),
                            activation_scale=scale,
                        )
                    case unreachable:
                        assert_never(unreachable)
            case unreachable:
                assert_never(unreachable)

    def readout_product(self) -> MACEInvariantReadoutProduct:
        architecture = self.architecture
        correlation = architecture.readout_correlation
        if correlation is None:
            raise RuntimeError("The invariant readout product needs readout_correlation.")
        scalars = architecture.layout(architecture.channel_count, 0)
        return MACEInvariantReadoutProduct(
            self.contraction(
                "readout_products.0", architecture.hidden_degree, scalars, correlation
            ),
            self.linear("readout_products.0.linear.weight", scalars, scalars),
            architecture.layout(1, architecture.hidden_degree),
        )

    def pair_repulsion(self) -> MACEPairRepulsion:
        architecture = self.architecture
        inventory = self.inventory
        coefficients = inventory.array("pair_repulsion_fn.c", (4,))
        radii = inventory.array("pair_repulsion_fn.covalent_radii")
        exponent = inventory.scalar("pair_repulsion_fn.a_exp")
        prefactor = inventory.scalar("pair_repulsion_fn.a_prefactor")
        if radii.ndim != 1 or radii.shape[0] <= max(architecture.species):
            raise ValueError("Source ZBL radii do not cover the model species.")
        # Only the radii of the model species are ever read; source tables of
        # other elements differ between provider releases.
        species_radii = radii[np.asarray(architecture.species)]
        if (
            not _matches(coefficients, _ZBL_SOURCE_CONSTANTS)
            or not _matches(exponent, _ZBL_SOURCE_SCREENING[0])
            or not _matches(prefactor, _ZBL_SOURCE_SCREENING[1])
            or int(inventory.scalar("pair_repulsion_fn.p")) != architecture.cutoff_power
            or not _matches(
                species_radii, covalent_radii_angstrom(architecture.species).tolist()
            )
        ):
            raise ValueError(
                "Source ZBL constants differ from the admitted universal ZBL."
            )
        return MACEPairRepulsion(
            architecture.species,
            self.scale,
            power=architecture.cutoff_power,
            dtype=self.dtype,
            covalent_radii=species_radii.astype(np.float64),
            coefficients=coefficients.astype(np.float64),
            screening=np.asarray([exponent, prefactor], dtype=np.float64),
        )

    def readout_slots(self) -> tuple[Literal["linear", "nonlinear"] | None, ...]:
        """Readout kind of every layer (``None`` without one), in source order."""
        slots: list[Literal["linear", "nonlinear"] | None] = []
        for index in range(self.architecture.interaction_count):
            kind = self.architecture.readout_kind(index)
            slots.append(None if kind == "none" else kind)
        return tuple(slots)

    def layers(self) -> tuple[MACELayer, ...]:
        architecture = self.architecture
        layers = []
        position = 0
        for index, readout_kind in enumerate(self.readout_slots()):
            outputs = architecture.output_layout(index)
            residual, _ = interaction_traits(architecture.interactions[index])
            product = MACEProduct(
                self.contraction(
                    f"products.{index}",
                    architecture.edge_degree,
                    outputs,
                    architecture.correlations[index],
                ),
                self.linear(f"products.{index}.linear.weight", outputs, outputs),
                architecture.target_layout(),
                self_connection=index > 0 or residual,
            )
            readout = None
            if readout_kind is not None:
                readout = self.readout(f"readouts.{position}", readout_kind)
                position += 1
            layers.append(
                MACELayer(
                    self.interaction(index),
                    product,
                    readout=readout,
                    readout_product=self.readout_product()
                    if architecture.readout_correlation is not None
                    else None,
                    pair_repulsion=self.pair_repulsion()
                    if index == 0 and architecture.pair_repulsion
                    else None,
                )
            )
        return tuple(layers)

    def latent_readouts(self) -> tuple[MACELESLatentReadout | None, ...]:
        """The MACELES ``les_readouts.k`` copies, aligned with the readout slots."""
        architecture = self.architecture
        if architecture.readout_variant != "nonlinear":
            raise ValueError(
                "The source MACELES copies only linear and nonlinear readouts; a "
                "nonlinear-bias readout has no latent counterpart."
            )
        if architecture.readout_correlation is not None:
            raise ValueError("The source MACELES latent copy has no readout product.")
        latent: list[MACELESLatentReadout | None] = []
        position = 0
        for kind in self.readout_slots():
            if kind is None:
                latent.append(None)
                continue
            readout = self.readout(f"les_readouts.{position}", kind)
            if not isinstance(readout, (MACELinearReadout, MACENonlinearReadout)):
                raise RuntimeError("A latent readout must be linear or nonlinear.")
            latent.append(readout)
            position += 1
        return tuple(latent)

    def joint_embedding(self) -> MACEJointEmbedding | None:
        """The source ``GenericJointEmbedding`` in native (transposed) layout."""
        architecture = self.architecture
        features = architecture.embedding_features
        if not features:
            return None
        inventory = self.inventory
        groups: list[tuple[np.ndarray, ...]] = []
        for feature in features:
            base = f"joint_embedding.embedders.{feature.name}"
            width = feature.width
            match feature.kind:
                case "categorical":
                    classes = feature.classes
                    if classes is None:
                        raise RuntimeError("A categorical feature lost its classes.")
                    groups.append(
                        (self.cast(inventory.exact(f"{base}.weight", (classes, width))),)
                    )
                case "continuous":
                    first = self.cast(
                        inventory.exact(
                            f"{base}.0.weight", (width, feature.input_width)
                        ).T
                    )
                    second = self.cast(
                        inventory.exact(f"{base}.2.weight", (width, width)).T
                    )
                    if feature.bias:
                        groups.append(
                            (
                                first,
                                self.cast(inventory.exact(f"{base}.0.bias", (width,))),
                                second,
                                self.cast(inventory.exact(f"{base}.2.bias", (width,))),
                            )
                        )
                    else:
                        groups.append((first, second))
                case unreachable:
                    assert_never(unreachable)
        total = sum(feature.width for feature in features)
        projection = self.cast(
            inventory.exact(
                "joint_embedding.project.0.weight", (architecture.channel_count, total)
            ).T
        )
        return MACEJointEmbedding(features, groups, projection)

    def embedding_readout(self) -> MACELinearReadout | None:
        architecture = self.architecture
        if not architecture.embedding_readout:
            return None
        return MACELinearReadout(
            self.cast(
                self.inventory.array(
                    "embedding_readout.linear.weight",
                    (architecture.channel_count, len(architecture.heads)),
                )
            )
        )


def _reconstruction(
    scale: AtomisticScaleContract,
    architecture: MACEArchitecture,
    tensors: Mapping[str, ArrayLike],
    degree_transforms: Sequence[ArrayLike],
    couplings: Mapping[tuple[int, int, int], ArrayLike],
    source_id: str,
    precision: AtomisticPrecisionPolicy,
    maximum_source_entries: int,
    /,
) -> _Reconstruction:
    if not isinstance(scale, AtomisticScaleContract):
        raise TypeError("scale must be an AtomisticScaleContract.")
    if not isinstance(architecture, MACEArchitecture):
        raise TypeError("architecture must be a MACEArchitecture.")
    if not isinstance(couplings, Mapping):
        raise TypeError("couplings must map degree triples to source coupling tables.")
    inventory = _SourceInventory(tensors)
    numbers = inventory.array("atomic_numbers")
    if (
        numbers.ndim != 1
        or tuple(int(value) for value in numbers) != architecture.species
    ):
        raise ValueError("Source atomic numbers differ from the declared species order.")
    return _Reconstruction(
        architecture,
        scale,
        inventory,
        _transforms(
            degree_transforms, max(architecture.edge_degree, architecture.hidden_degree)
        ),
        couplings,
        canonical_identifier(source_id, "source_id"),
        _compute_dtype(precision),
        positive_integer(maximum_source_entries, "maximum_source_entries"),
    )


def _precision(precision: AtomisticPrecisionPolicy | None, /) -> AtomisticPrecisionPolicy:
    resolved = AtomisticPrecisionPolicy() if precision is None else precision
    if not isinstance(resolved, AtomisticPrecisionPolicy):
        raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
    return resolved


def _potential(
    reconstruction: _Reconstruction,
    precision: AtomisticPrecisionPolicy,
    streaming: StreamedRelationPlan | None,
    /,
) -> MACEPotential:
    """The native MACE potential of every MACE source tensor (inventory left open)."""
    architecture = reconstruction.architecture
    inventory = reconstruction.inventory
    geometry = reconstruction.geometry()
    layers = reconstruction.layers()
    embedding = reconstruction.cast(
        inventory.array(
            "node_embedding.linear.weight",
            (architecture.species_count, architecture.channel_count),
        )
    )
    joint = reconstruction.joint_embedding()
    embedding_readout = reconstruction.embedding_readout()
    atomic_energies = inventory.array("atomic_energies_fn.atomic_energies")
    energy_scale = energy_shift = None
    if architecture.energy_scaling == "scale-shift":
        energy_scale = inventory.array("scale_shift.scale")
        energy_shift = inventory.array("scale_shift.shift")
    return MACEPotential(
        reconstruction.scale,
        architecture,
        atomic_energies=atomic_energies.astype(np.float64),
        energy_scale=None if energy_scale is None else energy_scale.astype(np.float64),
        energy_shift=None if energy_shift is None else energy_shift.astype(np.float64),
        precision=precision,
        streaming=streaming,
        embedding=jnp.asarray(embedding),
        geometry=geometry,
        layers=layers,
        joint_embedding=joint,
        embedding_readout=embedding_readout,
    )


def mace_potential_from_source(
    scale: AtomisticScaleContract,
    architecture: MACEArchitecture,
    tensors: Mapping[str, ArrayLike],
    /,
    *,
    degree_transforms: Sequence[ArrayLike],
    couplings: Mapping[tuple[int, int, int], ArrayLike],
    source_id: str,
    precision: AtomisticPrecisionPolicy | None = None,
    streaming: StreamedRelationPlan | None = None,
    maximum_source_entries: int = 1 << 26,
) -> MACEPotential:
    """Reconstruct a native MACE potential from one exact source tensor inventory.

    ``tensors`` maps every admitted source tensor name to its value; a missing
    or unadmitted name refuses. ``degree_transforms[l]`` is the orthogonal
    ``T_l`` with ``x_native = T_l @ x_source`` for real degree-``l`` components.
    ``couplings[(l1, l2, l3)]`` is the source ``wigner_3j(l1, l2, l3)`` table of
    shape ``(2 l1 + 1, 2 l2 + 1, 2 l3 + 1)`` for every coupling path; it
    certifies each path's native sign. ``source_id`` names the trusted source
    (for example its conversion identity); it is recorded in every imported
    coupling basis.
    """
    precision_ = _precision(precision)
    reconstruction = _reconstruction(
        scale,
        architecture,
        tensors,
        degree_transforms,
        couplings,
        source_id,
        precision_,
        maximum_source_entries,
    )
    potential = _potential(reconstruction, precision_, streaming)
    reconstruction.inventory.finish()
    return potential


def maceles_potential_from_source(
    scale: AtomisticScaleContract,
    architecture: MACEArchitecture,
    tensors: Mapping[str, ArrayLike],
    /,
    *,
    les_arguments: Mapping[str, object],
    atom_capacity: int,
    resources: LatentEwaldResources,
    degree_transforms: Sequence[ArrayLike],
    couplings: Mapping[tuple[int, int, int], ArrayLike],
    source_id: str,
    precision: AtomisticPrecisionPolicy | None = None,
    streaming: StreamedRelationPlan | None = None,
    maximum_source_entries: int = 1 << 26,
) -> MACELESPotential:
    """Reconstruct a native MACELES potential from one exact source inventory.

    The inventory is a ``ScaleShiftMACE`` inventory (see
    `mace_potential_from_source`) plus one ``les_readouts.k`` copy per readout
    ``readouts.k``. ``les_arguments`` are the source ``Les`` arguments recorded
    from the module attributes; they define the latent Ewald energy through
    `LatentEwaldDefinition.from_source_arguments`, which is prepared once for
    ``atom_capacity`` atoms per case under the caller's explicit ``resources``.
    """
    if not isinstance(resources, LatentEwaldResources):
        raise TypeError("resources must be LatentEwaldResources.")
    precision_ = _precision(precision)
    reconstruction = _reconstruction(
        scale,
        architecture,
        tensors,
        degree_transforms,
        couplings,
        source_id,
        precision_,
        maximum_source_entries,
    )
    mace = _potential(reconstruction, precision_, streaming)
    latent = reconstruction.latent_readouts()
    reconstruction.inventory.finish()
    definition = LatentEwaldDefinition.from_source_arguments(les_arguments)
    prepared = definition.prepare_graph(
        scale, positive_integer(atom_capacity, "atom_capacity"), resources=resources
    )
    return MACELESPotential(mace, latent, prepared)


__all__ = [
    "MACE_SOURCE_LAYER_NORM_EPSILON",
    "mace_potential_from_source",
    "maceles_potential_from_source",
]
