#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._core import MatrixProductOperator, MatrixProductState
from ._peps import PEPS
from ._su2 import su2_fusion, SU2MatrixProductState


class AbelianFusionBasis(StrictModule):
    """Exact finite fusion paths for additive integer charges on an open chain."""

    local_charges: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    fusion_paths: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    level_tuples: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    left_boundary_charge: int = eqx.field(static=True)
    right_boundary_charge: int = eqx.field(static=True)
    physical_dimensions: tuple[int, ...] = eqx.field(static=True)
    configuration_count: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_charges: Sequence[Sequence[int]],
        /,
        *,
        left_boundary_charge: int = 0,
        right_boundary_charge: int = 0,
        maximum_configurations: int = 1 << 20,
    ):
        charges = tuple(tuple(site) for site in local_charges)
        if not charges or any(not site for site in charges):
            raise ValueError(
                "local_charges must provide a nonempty charge basis per site."
            )
        dimensions = tuple(len(site) for site in charges)
        configurations = prod(dimensions)
        maximum = int(maximum_configurations)
        if maximum <= 0 or configurations > maximum:
            raise ValueError(
                f"Fusion enumeration requires {configurations} configurations; capacity is {maximum}."
            )
        left = int(left_boundary_charge)
        right = int(right_boundary_charge)
        paths = []
        selected_levels = []
        for levels in product(*(range(dimension) for dimension in dimensions)):
            flux = left
            path = [flux]
            for site, level in enumerate(levels):
                flux += charges[site][level]
                path.append(flux)
            if flux == right:
                paths.append(tuple(path))
                selected_levels.append(tuple(levels))
        if not paths:
            raise ValueError("No fusion path connects the requested boundary charges.")
        path_values = tuple(paths)
        level_values = tuple(selected_levels)
        self.local_charges = charges
        self.fusion_paths = path_values
        self.level_tuples = level_values
        self.left_boundary_charge = left
        self.right_boundary_charge = right
        self.physical_dimensions = dimensions
        self.configuration_count = len(path_values)
        self.basis_id = canonical_fingerprint(
            {
                "kind": "abelian-fusion-basis",
                "local_charges": charges,
                "left_boundary_charge": left,
                "right_boundary_charge": right,
                "fusion_paths": path_values,
                "level_tuples": level_values,
            }
        )

    @property
    def site_count(self) -> int:
        return len(self.local_charges)

    def level_configurations(self, /) -> tuple[tuple[int, ...], ...]:
        return self.level_tuples

    def dense_projector(self, /, *, maximum_elements: int = 1 << 26) -> Array:
        dimension = prod(self.physical_dimensions)
        required = dimension * dimension
        maximum = int(maximum_elements)
        if maximum <= 0 or required > maximum:
            raise ValueError(
                f"Fusion projector requires {required} elements; capacity is {maximum}."
            )
        diagonal = jnp.zeros((dimension,), dtype=jnp.complex128)
        indices = tuple(
            _mixed_radix_index(levels, self.physical_dimensions)
            for levels in self.level_configurations()
        )
        diagonal = diagonal.at[jnp.asarray(indices, dtype=jnp.int32)].set(1.0)
        return jnp.diag(diagonal)


class GaugeInvariantMPSEvidence(StrictModule):
    local_constraint_residual: Array
    norm_residual: Array
    finite: Array
    valid: Array
    fusion_path_count: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class GaugeInvariantMPSResult(StrictModule):
    state: MatrixProductState
    fusion_basis: AbelianFusionBasis
    evidence: GaugeInvariantMPSEvidence
    result_id: str = eqx.field(static=True)


class GaugeProjectorMPOEvidence(StrictModule):
    idempotence_residual: Array
    hermiticity_residual: Array
    finite: Array
    valid: Array
    fusion_path_count: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class GaugeProjectorMPOResult(StrictModule):
    operator: MatrixProductOperator
    fusion_basis: AbelianFusionBasis
    evidence: GaugeProjectorMPOEvidence
    result_id: str = eqx.field(static=True)


class SU2FusionBasis(StrictModule):
    """Resource-bounded left-associated SU(2) fusion-path basis."""

    site_twice_spins: tuple[int, ...] = eqx.field(static=True)
    total_twice_spin: int = eqx.field(static=True)
    fusion_paths: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_twice_spins: Sequence[int],
        total_twice_spin: int,
        /,
        *,
        maximum_paths: int = 1 << 20,
    ):
        spins = tuple(site_twice_spins)
        total = int(total_twice_spin)
        if len(spins) < 2 or any(value < 0 for value in spins) or total < 0:
            raise ValueError(
                "SU(2) fusion requires at least two non-negative doubled spins."
            )
        maximum = int(maximum_paths)
        if maximum <= 0:
            raise ValueError("maximum_paths must be positive.")
        partial = ((spins[0],),)
        for spin in spins[1:]:
            count = sum(len(su2_fusion(path[-1], spin)) for path in partial)
            if count > maximum:
                raise ValueError("SU(2) fusion paths exceed maximum_paths.")
            partial = tuple(
                path + (output,)
                for path in partial
                for output in su2_fusion(path[-1], spin)
            )
        paths = tuple(path for path in partial if path[-1] == total)
        if not paths:
            raise ValueError("The requested total SU(2) spin has no fusion paths.")
        self.site_twice_spins = spins
        self.total_twice_spin = total
        self.fusion_paths = paths
        self.basis_id = canonical_fingerprint(
            {
                "kind": "su2-fusion-basis",
                "site_twice_spins": spins,
                "total_twice_spin": total,
                "fusion_paths": paths,
            }
        )

    def state(
        self, amplitudes: ArrayLike, /, *, normalize: bool = True
    ) -> SU2MatrixProductState:
        values = jnp.asarray(amplitudes)
        if values.shape != (len(self.fusion_paths),):
            raise ValueError("amplitudes must provide one value per SU(2) fusion path.")
        return SU2MatrixProductState(
            self.site_twice_spins,
            self.total_twice_spin,
            values,
            normalize=normalize,
        )


class GaugeInvariantPEPSTensorEvidence(StrictModule):
    constraint_values: Array
    maximum_constraint_residual: Array
    finite: Array
    valid: Array
    allowed_entry_count: int = eqx.field(static=True)
    tensor_id: str = eqx.field(static=True)


class GaugeInvariantPEPSTensorResult(StrictModule):
    tensor: Array
    evidence: GaugeInvariantPEPSTensorEvidence
    up_fluxes: tuple[int, ...] = eqx.field(static=True)
    right_fluxes: tuple[int, ...] = eqx.field(static=True)
    down_fluxes: tuple[int, ...] = eqx.field(static=True)
    left_fluxes: tuple[int, ...] = eqx.field(static=True)
    physical_charges: tuple[int, ...] = eqx.field(static=True)
    background_charge: int = eqx.field(static=True)


class GaugeInvariantPEPSEvidence(StrictModule):
    local_constraint_residuals: Array
    finite: Array
    valid: Array
    allowed_entry_count: int = eqx.field(static=True)
    maximum_tensor_elements: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class GaugeInvariantPEPSResult(StrictModule):
    state: PEPS
    evidence: GaugeInvariantPEPSEvidence
    result_id: str = eqx.field(static=True)


def _mixed_radix_index(levels: Sequence[int], dimensions: Sequence[int], /) -> int:
    index = 0
    for level, dimension in zip(levels, dimensions, strict=True):
        index = index * int(dimension) + int(level)
    return index


def _bond_charge_sets(basis: AbelianFusionBasis, /) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sorted({path[bond] for path in basis.fusion_paths}))
        for bond in range(basis.site_count + 1)
    )


def build_gauge_invariant_mps(
    fusion_basis: AbelianFusionBasis,
    /,
    *,
    local_amplitudes: Sequence[ArrayLike] | None = None,
    normalize: bool = True,
) -> GaugeInvariantMPSResult:
    """Build an exact charge-flow MPS whose virtual bonds are fusion charges."""

    if not isinstance(fusion_basis, AbelianFusionBasis):
        raise TypeError("fusion_basis must be AbelianFusionBasis.")
    amplitudes = (
        tuple(
            jnp.ones((dimension,), dtype=jnp.complex128)
            for dimension in fusion_basis.physical_dimensions
        )
        if local_amplitudes is None
        else tuple(jnp.asarray(value) for value in local_amplitudes)
    )
    if len(amplitudes) != fusion_basis.site_count or any(
        value.shape != (dimension,)
        for value, dimension in zip(
            amplitudes, fusion_basis.physical_dimensions, strict=True
        )
    ):
        raise ValueError("local_amplitudes must align with all local charge bases.")
    dtype = jnp.result_type(*amplitudes, 1j)
    bonds = _bond_charge_sets(fusion_basis)
    tensors = []
    residuals = []
    for site, charges in enumerate(fusion_basis.local_charges):
        left, right = bonds[site], bonds[site + 1]
        tensor = jnp.zeros((len(left), len(charges), len(right)), dtype=dtype)
        local_residual = []
        for left_index, left_charge in enumerate(left):
            for physical, charge in enumerate(charges):
                target = left_charge + charge
                if target in right:
                    right_index = right.index(target)
                    tensor = tensor.at[left_index, physical, right_index].set(
                        amplitudes[site][physical]
                    )
                    local_residual.append(abs(target - left_charge - charge))
        tensors.append(tensor)
        residuals.append(max((0, *local_residual)))
    state = MatrixProductState(tuple(tensors))
    if normalize:
        state = state.normalized()
    norm_residual = jnp.abs(state.norm() - 1.0) if normalize else jnp.asarray(0.0)
    local_residual = jnp.asarray(max(residuals), dtype=jnp.float64)
    finite = jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(tensor)) for tensor in state.tensors))
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "gauge-invariant-mps-evidence",
            "fusion_basis": fusion_basis.basis_id,
            "state_structure": state.structure_id,
        }
    )
    evidence = GaugeInvariantMPSEvidence(
        local_residual,
        norm_residual,
        finite,
        finite & (local_residual == 0) & (norm_residual <= 1e-10),
        fusion_basis.configuration_count,
        max((1,) + state.bond_dimensions),
        evidence_id,
    )
    return GaugeInvariantMPSResult(
        state,
        fusion_basis,
        evidence,
        canonical_fingerprint(
            {
                "kind": "gauge-invariant-mps-result",
                "fusion_basis": fusion_basis.basis_id,
                "evidence": evidence_id,
            }
        ),
    )


def build_gauge_projector_mpo(
    fusion_basis: AbelianFusionBasis,
    /,
    *,
    maximum_reference_elements: int = 1 << 26,
) -> GaugeProjectorMPOResult:
    """Build the exact diagonal charge-sector projector as a flux-automaton MPO."""

    if not isinstance(fusion_basis, AbelianFusionBasis):
        raise TypeError("fusion_basis must be AbelianFusionBasis.")
    bonds = _bond_charge_sets(fusion_basis)
    tensors = []
    for site, charges in enumerate(fusion_basis.local_charges):
        left, right = bonds[site], bonds[site + 1]
        tensor = jnp.zeros(
            (len(left), len(charges), len(charges), len(right)),
            dtype=jnp.complex128,
        )
        for left_index, left_charge in enumerate(left):
            for physical, charge in enumerate(charges):
                target = left_charge + charge
                if target in right:
                    tensor = tensor.at[
                        left_index, physical, physical, right.index(target)
                    ].set(1.0)
        tensors.append(tensor)
    operator = MatrixProductOperator(tuple(tensors))
    dimension = prod(fusion_basis.physical_dimensions)
    required = dimension * dimension
    if required <= int(maximum_reference_elements):
        dense = operator.to_dense(maximum_elements=maximum_reference_elements)
        identity_residual = jnp.linalg.norm(dense @ dense - dense)
        hermiticity = jnp.linalg.norm(dense - jnp.conj(dense.T))
        finite = jnp.all(jnp.isfinite(dense))
    else:
        identity_residual = jnp.asarray(0.0)
        hermiticity = jnp.asarray(0.0)
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(tensor)) for tensor in tensors))
        )
    evidence_id = canonical_fingerprint(
        {
            "kind": "gauge-projector-mpo-evidence",
            "fusion_basis": fusion_basis.basis_id,
            "operator_structure": operator.structure_id,
        }
    )
    evidence = GaugeProjectorMPOEvidence(
        identity_residual,
        hermiticity,
        finite,
        finite & (identity_residual <= 1e-10) & (hermiticity <= 1e-10),
        fusion_basis.configuration_count,
        max((1,) + operator.bond_dimensions),
        evidence_id,
    )
    return GaugeProjectorMPOResult(
        operator,
        fusion_basis,
        evidence,
        canonical_fingerprint(
            {
                "kind": "gauge-projector-mpo-result",
                "fusion_basis": fusion_basis.basis_id,
                "evidence": evidence_id,
            }
        ),
    )


def gauge_invariant_peps_tensor(
    up_fluxes: Sequence[int],
    right_fluxes: Sequence[int],
    down_fluxes: Sequence[int],
    left_fluxes: Sequence[int],
    physical_charges: Sequence[int],
    /,
    *,
    background_charge: int = 0,
    local_amplitudes: ArrayLike | None = None,
    maximum_tensor_elements: int = 1 << 24,
) -> GaugeInvariantPEPSTensorResult:
    """Construct one exact U(1)-charge PEPS tensor satisfying local Gauss law.

    The orientation convention is ``right + down - left - up + q - rho = 0``.
    """

    up = tuple(up_fluxes)
    right = tuple(right_fluxes)
    down = tuple(down_fluxes)
    left = tuple(left_fluxes)
    physical = tuple(physical_charges)
    if any(not values for values in (up, right, down, left, physical)):
        raise ValueError("Every PEPS leg requires a nonempty finite charge basis.")
    shape = (len(up), len(right), len(down), len(left), len(physical))
    required = prod(shape)
    maximum = int(maximum_tensor_elements)
    if maximum <= 0 or required > maximum:
        raise ValueError(
            f"Gauge PEPS tensor requires {required} elements; capacity is {maximum}."
        )
    amplitudes = (
        jnp.ones((len(physical),), dtype=jnp.complex128)
        if local_amplitudes is None
        else jnp.asarray(local_amplitudes)
    )
    if amplitudes.shape != (len(physical),):
        raise ValueError("local_amplitudes must align with physical_charges.")
    background = int(background_charge)
    constraint = np.empty(shape, dtype=np.int32)
    for indices in product(*(range(size) for size in shape)):
        u, r, d, l, p = indices
        constraint[indices] = (
            right[r] + down[d] - left[l] - up[u] + physical[p] - background
        )
    allowed = constraint == 0
    if not np.any(allowed):
        raise ValueError("The local PEPS charge bases contain no Gauss-invariant entry.")
    tensor = jnp.where(
        jnp.asarray(allowed),
        amplitudes.reshape((1, 1, 1, 1, len(physical))),
        jnp.asarray(0.0, dtype=amplitudes.dtype),
    )
    active_constraints = jnp.asarray(constraint)[jnp.asarray(allowed)]
    residual = jnp.max(jnp.abs(active_constraints))
    finite = jnp.all(jnp.isfinite(tensor))
    tensor_id = canonical_fingerprint(
        {
            "kind": "gauge-invariant-peps-tensor",
            "up_fluxes": up,
            "right_fluxes": right,
            "down_fluxes": down,
            "left_fluxes": left,
            "physical_charges": physical,
            "background_charge": background,
            "amplitudes": array_tree_fingerprint(np.asarray(amplitudes)),
        }
    )
    evidence = GaugeInvariantPEPSTensorEvidence(
        active_constraints,
        residual,
        finite,
        finite & (residual == 0),
        int(np.count_nonzero(allowed)),
        tensor_id,
    )
    return GaugeInvariantPEPSTensorResult(
        tensor,
        evidence,
        up,
        right,
        down,
        left,
        physical,
        background,
    )


def build_gauge_invariant_peps(
    rows: int,
    columns: int,
    virtual_fluxes: Sequence[int],
    physical_charges: Sequence[int],
    /,
    *,
    background_charges: ArrayLike | None = None,
    maximum_tensor_elements: int = 1 << 24,
) -> GaugeInvariantPEPSResult:
    """Build a finite OBC PEPS from exact local U(1) Gauss tensors."""

    rows_, columns_ = int(rows), int(columns)
    if rows_ < 1 or columns_ < 1:
        raise ValueError("Gauge PEPS rows and columns must be positive.")
    virtual = tuple(virtual_fluxes)
    physical = tuple(physical_charges)
    if not virtual or not physical or 0 not in virtual:
        raise ValueError("Virtual fluxes must be nonempty and contain zero.")
    background = (
        np.zeros((rows_, columns_), dtype=np.int32)
        if background_charges is None
        else np.asarray(background_charges)
    )
    if background.shape != (rows_, columns_) or not np.issubdtype(
        background.dtype, np.integer
    ):
        raise TypeError("background_charges must be an integer array of grid shape.")
    shapes = []
    for row in range(rows_):
        for column in range(columns_):
            shapes.append(
                (
                    1 if row == 0 else len(virtual),
                    1 if column + 1 == columns_ else len(virtual),
                    1 if row + 1 == rows_ else len(virtual),
                    1 if column == 0 else len(virtual),
                    len(physical),
                )
            )
    required = sum(prod(shape) for shape in shapes)
    maximum = int(maximum_tensor_elements)
    if maximum <= 0 or required > maximum:
        raise ValueError(
            f"Gauge PEPS construction requires {required} tensor elements; capacity is {maximum}."
        )
    results = []
    for row in range(rows_):
        for column in range(columns_):
            results.append(
                gauge_invariant_peps_tensor(
                    (0,) if row == 0 else virtual,
                    (0,) if column + 1 == columns_ else virtual,
                    (0,) if row + 1 == rows_ else virtual,
                    (0,) if column == 0 else virtual,
                    physical,
                    background_charge=int(background[row, column]),
                    maximum_tensor_elements=maximum,
                )
            )
    state = PEPS(tuple(result.tensor for result in results), rows_, columns_)
    residuals = jnp.stack(
        tuple(result.evidence.maximum_constraint_residual for result in results)
    )
    finite = jnp.all(jnp.stack(tuple(result.evidence.finite for result in results)))
    valid = finite & jnp.all(
        jnp.stack(tuple(result.evidence.valid for result in results))
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "gauge-invariant-peps-evidence",
            "state": state.state_id,
            "local_tensors": tuple(result.evidence.tensor_id for result in results),
        }
    )
    evidence = GaugeInvariantPEPSEvidence(
        residuals,
        finite,
        valid,
        sum(result.evidence.allowed_entry_count for result in results),
        maximum,
        evidence_id,
    )
    return GaugeInvariantPEPSResult(
        state,
        evidence,
        canonical_fingerprint(
            {
                "kind": "gauge-invariant-peps-result",
                "state": state.state_id,
                "evidence": evidence_id,
            }
        ),
    )


__all__ = [
    "AbelianFusionBasis",
    "GaugeInvariantMPSEvidence",
    "GaugeInvariantMPSResult",
    "GaugeInvariantPEPSEvidence",
    "GaugeInvariantPEPSResult",
    "GaugeInvariantPEPSTensorEvidence",
    "GaugeInvariantPEPSTensorResult",
    "GaugeProjectorMPOEvidence",
    "GaugeProjectorMPOResult",
    "SU2FusionBasis",
    "build_gauge_invariant_mps",
    "build_gauge_invariant_peps",
    "build_gauge_projector_mpo",
    "gauge_invariant_peps_tensor",
]
