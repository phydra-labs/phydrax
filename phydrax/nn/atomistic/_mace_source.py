#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed reconstruction of a native MACE potential from source tensors.

The input is the exact source parameter inventory keyed by the source
(non-cuEquivariance) state-dictionary names, plus the explicit basis evidence of
the source real-harmonic convention. Reconstruction preserves the original
parameterization: linear, radial, self-connection, product and readout weights
are the source leaves (only transposed or split into native blocks), the
symmetric products keep their original ``W`` with the source ``U`` transformed
into the native basis, and each coupling path records the sign relating the
source coupling to the native one. Source normalization constants are not
re-derived here; the caller verifies them against the provider before calling.
This module never imports a provider or deserializes anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, Literal

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._validation import canonical_identifier, positive_integer
from ...atomistic._types import AtomisticPrecisionPolicy, AtomisticScaleContract
from ...ein import contract
from ...sparse._streamed import StreamedRelationPlan
from ...special import RealCartesianHarmonics
from ...typing import AnyShape, as_host_array, HostFloat64
from ..operator.layers import O3IrrepLinear, O3TensorProduct
from ..operator.representations import o3_real_coupling, O3IrrepBlock, O3IrrepLayout
from ._mace import (
    _compute_dtype,
    mace_tensor_product_plan,
    MACEArchitecture,
    MACEPotential,
)
from ._mace_interaction import (
    convolution_paths,
    interaction_traits,
    mace_irrep_layout,
    MACEEdgeGeometry,
    MACEInteraction,
    MACELayer,
    MACEProduct,
    MACESelfConnection,
    self_connection_paths,
)
from ._mace_readout import (
    covalent_radii_angstrom,
    MACEInvariantReadoutProduct,
    MACELinearReadout,
    MACENonlinearReadout,
    MACEPairRepulsion,
)
from ._radial import (
    AgnesiTransform,
    BesselRadialBasis,
    PolynomialCutoff,
    RadialEmbedding,
    RadialMLP,
)
from ._symmetric_contraction import (
    SymmetricContraction,
    SymmetricContractionBasis,
    SymmetricContractionPlan,
)


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

    def array(self, name: str, shape: tuple[int, ...] | None = None, /) -> np.ndarray:
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
        if shape is not None:
            if value.size != int(np.prod(shape, dtype=np.int64)):
                raise ValueError(
                    f"Source tensor {name!r} has {value.size} values; expected {shape}."
                )
            value = value.reshape(shape)
        return value

    def scalar(self, name: str, /) -> np.ndarray:
        return self.array(name, ())

    def present(self, name: str, /) -> bool:
        return name in self._tensors

    def finish(self) -> None:
        extra = sorted(set(self._tensors) - self._consumed)
        if extra:
            raise ValueError(f"Unadmitted source tensors: {extra}.")


def _matches(source: np.ndarray, expected: float | Sequence[float], /) -> bool:
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
    inputs concatenated in input-layout order.
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
        return value.astype(self.dtype)

    def linear(
        self, name: str, in_layout: O3IrrepLayout, out_layout: O3IrrepLayout, /
    ) -> O3IrrepLinear:
        flat = self.inventory.array(name, (_linear_count(in_layout, out_layout),))
        return O3IrrepLinear(
            in_layout,
            out_layout,
            weights=tuple(
                self.cast(block) for block in _linear_weights(flat, in_layout, out_layout)
            ),
            dtype=self.dtype,
        )

    def radial_network(self, prefix: str, widths: tuple[int, ...], /) -> RadialMLP:
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
        )

    def geometry(self) -> MACEEdgeGeometry:
        architecture = self.architecture
        inventory = self.inventory
        radii = (
            inventory.scalar("r_max"),
            inventory.scalar("radial_embedding.cutoff_fn.r_max"),
            inventory.scalar("radial_embedding.bessel_fn.r_max"),
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
        frequencies = inventory.array(
            "radial_embedding.bessel_fn.bessel_weights",
            (architecture.radial_basis_count,),
        )
        prefactor = inventory.scalar("radial_embedding.bessel_fn.prefactor")
        basis = BesselRadialBasis(
            frequencies.astype(np.float64),
            float(prefactor),
            architecture.cutoff,
            dtype=self.dtype,
        )
        transform = None
        if architecture.agnesi_parameters is not None:
            values = tuple(
                inventory.scalar(f"radial_embedding.distance_transform.{name}")
                for name in ("a", "q", "p")
            )
            if not all(
                _matches(value, declared)
                for value, declared in zip(values, architecture.agnesi_parameters)
            ):
                raise ValueError("Source Agnesi parameters differ from the declaration.")
            radii = inventory.array("radial_embedding.distance_transform.covalent_radii")
            if radii.ndim != 1 or radii.shape[0] <= max(architecture.species):
                raise ValueError("Source Agnesi radii do not cover the model species.")
            a, q, p = architecture.agnesi_parameters
            transform = AgnesiTransform(
                radii[np.asarray(architecture.species)].astype(np.float64),
                atomic_numbers=architecture.species,
                a=a,
                q=q,
                p=p,
                dtype=self.dtype,
            )
        return MACEEdgeGeometry(
            RadialEmbedding(
                basis,
                PolynomialCutoff(architecture.cutoff, architecture.cutoff_power),
                transform=transform,
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
        inputs = mace_irrep_layout(1, input_degree)
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
            mace_irrep_layout(1, output_layout.blocks[-1].degree),
            bases,
            species_count=species,
            channel_count=channels,
        )
        return SymmetricContraction(plan, weights)

    def interaction(self, index: int, /) -> MACEInteraction:
        architecture = self.architecture
        prefix = f"interactions.{index}"
        kind = architecture.interactions[index]
        residual, density_normalized = interaction_traits(kind)
        inputs = architecture.input_layout(index)
        outputs = architecture.output_layout(index)
        target = architecture.target_layout()
        message, paths = convolution_paths(
            inputs, architecture.harmonics_layout(), target
        )
        degrees = {block.name: block.degree for block in inputs.blocks}
        degrees.update(
            {block.name: block.degree for block in architecture.harmonics_layout().blocks}
        )
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
        widths = (
            architecture.radial_basis_count,
            *architecture.radial_widths,
            plan.parameter_count,
        )
        connection_in = inputs if residual else target
        connection_out = outputs if residual else target
        species = architecture.species_count
        connection_paths = self_connection_paths(connection_in, connection_out)
        flat = self.inventory.array(
            f"{prefix}.skip_tp.weight",
            (
                sum(
                    connection_in.blocks[left].multiplicity
                    * species
                    * connection_out.blocks[right].multiplicity
                    for left, right in connection_paths
                ),
            ),
        )
        connection_weights = []
        offset = 0
        for left, right in connection_paths:
            shape = (
                connection_in.blocks[left].multiplicity,
                species,
                connection_out.blocks[right].multiplicity,
            )
            size = int(np.prod(shape))
            connection_weights.append(
                self.cast(flat[offset : offset + size].reshape(shape))
            )
            offset += size
        density = None
        if density_normalized:
            density = RadialMLP(
                (
                    self.cast(
                        self.inventory.array(
                            f"{prefix}.density_fn.layer0.weight",
                            (architecture.radial_basis_count, 1),
                        )
                    ),
                ),
                activation_scale=architecture.radial_activation_scale,
                postprocess="tanh-square",
            )
        return MACEInteraction(
            self.linear(f"{prefix}.linear_up.weight", inputs, inputs),
            coupling,
            self.radial_network(f"{prefix}.conv_tp_weights", widths),
            self.linear(f"{prefix}.linear.weight", message, target),
            MACESelfConnection(
                connection_in, connection_out, connection_weights, species_count=species
            ),
            kind=kind,
            average_neighbor_count=architecture.average_neighbor_count,
            density=density,
        )

    def readout(
        self, position: int, kind: Literal["linear", "nonlinear"], /
    ) -> MACELinearReadout | MACENonlinearReadout:
        architecture = self.architecture
        channels = architecture.channel_count
        heads = len(architecture.heads)
        prefix = f"readouts.{position}"
        match kind:
            case "linear":
                return MACELinearReadout(
                    self.cast(
                        self.inventory.array(f"{prefix}.linear.weight", (channels, heads))
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
                return MACENonlinearReadout(
                    self.cast(
                        self.inventory.array(
                            f"{prefix}.linear_1.weight", (channels, hidden)
                        )
                    ),
                    self.cast(
                        self.inventory.array(f"{prefix}.linear_2.weight", (hidden, heads))
                    ),
                    activation_scale=scale,
                )
            case unreachable:
                assert_never(unreachable)

    def readout_product(self) -> MACEInvariantReadoutProduct:
        architecture = self.architecture
        correlation = architecture.readout_correlation
        if correlation is None:
            raise RuntimeError("The invariant readout product needs readout_correlation.")
        scalars = mace_irrep_layout(architecture.channel_count, 0)
        return MACEInvariantReadoutProduct(
            self.contraction(
                "readout_products.0", architecture.hidden_degree, scalars, correlation
            ),
            self.linear("readout_products.0.linear.weight", scalars, scalars),
            mace_irrep_layout(1, architecture.hidden_degree),
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

    def layers(self) -> tuple[MACELayer, ...]:
        architecture = self.architecture
        layers = []
        readout_position = 0
        for index in range(architecture.interaction_count):
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
            readout_kind = architecture.readout_kind(index)
            readout = None
            if readout_kind != "none":
                readout = self.readout(readout_position, readout_kind)
                readout_position += 1
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
    (for example its digest); it is recorded in every imported coupling basis.
    """
    if not isinstance(scale, AtomisticScaleContract):
        raise TypeError("scale must be an AtomisticScaleContract.")
    if not isinstance(architecture, MACEArchitecture):
        raise TypeError("architecture must be a MACEArchitecture.")
    precision_ = AtomisticPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, AtomisticPrecisionPolicy):
        raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
    if not isinstance(couplings, Mapping):
        raise TypeError("couplings must map degree triples to source coupling tables.")
    dtype = _compute_dtype(precision_)
    inventory = _SourceInventory(tensors)
    numbers = inventory.array("atomic_numbers")
    if (
        numbers.ndim != 1
        or tuple(int(value) for value in numbers) != architecture.species
    ):
        raise ValueError("Source atomic numbers differ from the declared species order.")
    reconstruction = _Reconstruction(
        architecture,
        scale,
        inventory,
        _transforms(
            degree_transforms, max(architecture.edge_degree, architecture.hidden_degree)
        ),
        couplings,
        canonical_identifier(source_id, "source_id"),
        dtype,
        positive_integer(maximum_source_entries, "maximum_source_entries"),
    )
    geometry = reconstruction.geometry()
    layers = reconstruction.layers()
    embedding = reconstruction.cast(
        inventory.array(
            "node_embedding.linear.weight",
            (architecture.species_count, architecture.channel_count),
        )
    )
    atomic_energies = inventory.array("atomic_energies_fn.atomic_energies")
    energy_scale = energy_shift = None
    if architecture.energy_scaling == "scale-shift":
        energy_scale = inventory.array("scale_shift.scale")
        energy_shift = inventory.array("scale_shift.shift")
    inventory.finish()
    return MACEPotential(
        scale,
        architecture,
        atomic_energies=atomic_energies.astype(np.float64),
        energy_scale=None if energy_scale is None else energy_scale.astype(np.float64),
        energy_shift=None if energy_shift is None else energy_shift.astype(np.float64),
        precision=precision_,
        streaming=streaming,
        embedding=jnp.asarray(embedding),
        geometry=geometry,
        layers=layers,
    )


__all__ = ["mace_potential_from_source"]
