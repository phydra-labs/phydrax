#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .lattice._compile import (
    prepare_quantum_lattice,
    PreparedQuantumLattice,
    QuantumLatticeResourcePolicy,
)
from .lattice._model import (
    LocalOperatorPlan,
    LocalSpacePlan,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from .lattice._operator import QuantumSectorOperator
from .lattice._sector import (
    FixedSpinProjectionBasis,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)


class QuantumSpinModel(StrictModule):
    """Finite quantum-spin model backed only by the canonical lattice terms."""

    specification: QuantumLatticeSpecification
    twice_spins: tuple[int, ...] = eqx.field(static=True)
    site_ids: tuple[str, ...] = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    sign_convention: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)


class QuantumSpinObservablePlan(StrictModule):
    model: QuantumSpinModel
    components: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, model: QuantumSpinModel, components: Sequence[str], /):
        if not isinstance(model, QuantumSpinModel):
            raise TypeError("model must be QuantumSpinModel.")
        values = tuple(str(value) for value in components)
        if not values or any(value not in ("x", "y", "z") for value in values):
            raise ValueError("Spin observable components must be selected from x, y, z.")
        self.model = model
        self.components = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quantum-spin-observable-plan",
                "model": model.model_id,
                "components": values,
            }
        )


def spin_matrices(twice_spin: int, /) -> tuple[Array, Array, Array, Array, Array]:
    """Return Sx,Sy,Sz,S+,S- in ascending doubled-projection order."""

    doubled = int(twice_spin)
    if doubled < 1:
        raise ValueError("twice_spin must be positive.")
    spin = 0.5 * doubled
    projections = np.arange(-doubled, doubled + 1, 2, dtype=float) * 0.5
    dimension = doubled + 1
    raising = np.zeros((dimension, dimension), dtype=complex)
    for column, projection in enumerate(projections[:-1]):
        raising[column + 1, column] = np.sqrt(
            spin * (spin + 1.0) - projection * (projection + 1.0)
        )
    lowering = raising.conj().T
    sx = 0.5 * (raising + lowering)
    sy = (raising - lowering) / (2.0j)
    sz = np.diag(projections)
    return tuple(jnp.asarray(value) for value in (sx, sy, sz, raising, lowering))


def _spaces_and_operators(
    site_ids: Sequence[str], twice_spins: Sequence[int], /
) -> tuple[tuple[LocalSpacePlan, ...], tuple[dict[str, LocalOperatorPlan], ...]]:
    sites = tuple(str(value) for value in site_ids)
    spins = tuple(int(value) for value in twice_spins)
    if not sites or len(sites) != len(spins) or len(set(sites)) != len(sites):
        raise ValueError("Quantum spin sites must be unique and align with twice_spins.")
    spaces = tuple(
        LocalSpacePlan.spin(site, spin) for site, spin in zip(sites, spins, strict=True)
    )
    operators = []
    for space, spin in zip(spaces, spins, strict=True):
        _, _, sz, raising, lowering = spin_matrices(spin)
        operators.append(
            {
                "z": LocalOperatorPlan(space, "Sz", sz, (0,)),
                "+": LocalOperatorPlan(space, "S+", raising, (2,)),
                "-": LocalOperatorPlan(space, "S-", lowering, (-2,)),
            }
        )
    return spaces, tuple(operators)


def _base_model(
    site_ids: Sequence[str],
    twice_spins: Sequence[int],
    terms: Sequence[QuantumLatticeTerm],
    /,
    *,
    energy_unit: str,
    kind: str,
    spaces: tuple[LocalSpacePlan, ...],
) -> QuantumSpinModel:
    unit = str(energy_unit).strip()
    if not unit:
        raise ValueError("Quantum spin energy_unit must be explicit.")
    specification = QuantumLatticeSpecification(spaces, terms)
    spins = tuple(int(value) for value in twice_spins)
    sites = tuple(str(value) for value in site_ids)
    identifier = canonical_fingerprint(
        {
            "kind": kind,
            "specification": specification.specification_id,
            "twice_spins": spins,
            "energy_unit": unit,
        }
    )
    return QuantumSpinModel(
        specification,
        spins,
        sites,
        unit,
        "H=-sum_bond(coupling)-sum_site(field/anisotropy); each bond occurs once",
        identifier,
    )


def _bond_inputs(
    site_count: int, bonds: ArrayLike, values: ArrayLike, name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(bonds)
    coupling = np.asarray(values)
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or not np.issubdtype(edges.dtype, np.integer)
    ):
        raise TypeError("bonds must have shape (bond, 2) with integer site indices.")
    if coupling.shape != (edges.shape[0],) or np.any(~np.isfinite(coupling)):
        raise ValueError(f"{name} must contain one finite value per once-only bond.")
    if (
        np.any(edges < 0)
        or np.any(edges >= site_count)
        or np.any(edges[:, 0] >= edges[:, 1])
    ):
        raise ValueError(
            "Every quantum spin bond must appear once with sender < receiver."
        )
    if np.unique(edges, axis=0).shape[0] != edges.shape[0]:
        raise ValueError("Quantum spin bonds must be unique.")
    return edges.astype(np.int32, copy=False), coupling


def xxz_spin_model(
    site_ids: Sequence[str],
    twice_spins: Sequence[int],
    bonds: ArrayLike,
    transverse_exchange: ArrayLike,
    longitudinal_exchange: ArrayLike,
    /,
    *,
    energy_unit: str,
) -> QuantumSpinModel:
    """Build H=-sum[Jxy(SxSx+SySy)+Jz SzSz] through canonical terms."""

    spaces, operators = _spaces_and_operators(site_ids, twice_spins)
    edges, jxy = _bond_inputs(
        len(spaces), bonds, transverse_exchange, "transverse_exchange"
    )
    _, jz = _bond_inputs(
        len(spaces), bonds, longitudinal_exchange, "longitudinal_exchange"
    )
    terms: list[QuantumLatticeTerm] = []
    for bond, (left, right) in enumerate(edges):
        terms.append(
            QuantumLatticeTerm(
                (operators[left]["z"], operators[right]["z"]),
                coefficient=-jz[bond],
                label=f"xxz-z-{bond}",
            )
        )
        terms.append(
            QuantumLatticeTerm(
                (operators[left]["+"], operators[right]["-"]),
                coefficient=-0.5 * jxy[bond],
                add_adjoint=True,
                label=f"xxz-transverse-{bond}",
            )
        )
    return _base_model(
        site_ids,
        twice_spins,
        terms,
        energy_unit=energy_unit,
        kind="xxz-quantum-spin-model",
        spaces=spaces,
    )


def heisenberg_spin_model(
    site_ids: Sequence[str],
    twice_spins: Sequence[int],
    bonds: ArrayLike,
    exchange: ArrayLike,
    /,
    *,
    energy_unit: str,
) -> QuantumSpinModel:
    """Build the isotropic J>0 ferromagnetic Heisenberg convention."""

    return xxz_spin_model(
        site_ids,
        twice_spins,
        bonds,
        exchange,
        exchange,
        energy_unit=energy_unit,
    )


def transverse_field_ising_model(
    site_ids: Sequence[str],
    twice_spins: Sequence[int],
    bonds: ArrayLike,
    exchange: ArrayLike,
    transverse_fields: ArrayLike,
    /,
    *,
    energy_unit: str,
) -> QuantumSpinModel:
    """Build H=-sum J Sz_i Sz_j - sum h Sx_i without Pauli rescaling."""

    spaces, operators = _spaces_and_operators(site_ids, twice_spins)
    edges, coupling = _bond_inputs(len(spaces), bonds, exchange, "exchange")
    fields = np.asarray(transverse_fields)
    if fields.shape != (len(spaces),) or np.any(~np.isfinite(fields)):
        raise ValueError("transverse_fields must have one finite value per site.")
    terms: list[QuantumLatticeTerm] = []
    for bond, (left, right) in enumerate(edges):
        terms.append(
            QuantumLatticeTerm(
                (operators[left]["z"], operators[right]["z"]),
                coefficient=-coupling[bond],
                label=f"tfim-bond-{bond}",
            )
        )
    for site, field in enumerate(fields):
        terms.append(
            QuantumLatticeTerm(
                (operators[site]["+"],),
                coefficient=-0.5 * field,
                add_adjoint=True,
                label=f"tfim-field-{site}",
            )
        )
    return _base_model(
        site_ids,
        twice_spins,
        terms,
        energy_unit=energy_unit,
        kind="transverse-field-ising-quantum-spin-model",
        spaces=spaces,
    )


def dmi_spin_model(
    site_ids: Sequence[str],
    twice_spins: Sequence[int],
    bonds: ArrayLike,
    dmi_vectors: ArrayLike,
    /,
    *,
    energy_unit: str,
) -> QuantumSpinModel:
    """Build once-oriented ``-D·(S_i×S_j)`` for arbitrary supplied D vectors."""

    spaces, operators = _spaces_and_operators(site_ids, twice_spins)
    raw_edges = np.asarray(bonds)
    edge_count = raw_edges.shape[0] if raw_edges.ndim == 2 else 0
    edges, _ = _bond_inputs(
        len(spaces), bonds, np.zeros((edge_count,)), "dmi bond support"
    )
    dmi = np.asarray(dmi_vectors, dtype=float)
    if dmi.shape != (edges.shape[0], 3) or np.any(~np.isfinite(dmi)):
        raise ValueError("dmi_vectors must have shape (bond, 3) with finite values.")
    if not np.any(np.abs(dmi) > 0.0):
        raise ValueError("A DMI model requires at least one nonzero oriented D vector.")
    components = (
        (("+", 0.5), ("-", 0.5)),
        (("+", -0.5j), ("-", 0.5j)),
        (("z", 1.0),),
    )
    epsilon = np.zeros((3, 3, 3), dtype=int)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
    adjoint_label = {"+": "-", "-": "+", "z": "z"}
    terms: list[QuantumLatticeTerm] = []
    for bond, (left, right) in enumerate(edges):
        coefficients: dict[tuple[str, str], complex] = {}
        for left_axis in range(3):
            for right_axis in range(3):
                cartesian = -sum(
                    dmi[bond, axis] * epsilon[axis, left_axis, right_axis]
                    for axis in range(3)
                )
                for left_label, left_factor in components[left_axis]:
                    for right_label, right_factor in components[right_axis]:
                        key = (left_label, right_label)
                        coefficients[key] = (
                            coefficients.get(key, 0.0)
                            + cartesian * left_factor * right_factor
                        )
        visited: set[tuple[str, str]] = set()
        for key, coefficient in sorted(coefficients.items()):
            if key in visited or abs(coefficient) <= 1.0e-14:
                continue
            adjoint_key = (
                adjoint_label[key[0]],
                adjoint_label[key[1]],
            )
            adjoint_coefficient = coefficients.get(adjoint_key, 0.0)
            if not np.allclose(
                adjoint_coefficient,
                np.conj(coefficient),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("DMI ladder decomposition lost Hermitian pairing.")
            if key == adjoint_key:
                terms.append(
                    QuantumLatticeTerm(
                        (operators[left][key[0]], operators[right][key[1]]),
                        coefficient=float(np.real(coefficient)),
                        label=f"dmi-{bond}-{key[0]}-{key[1]}",
                    )
                )
            else:
                terms.append(
                    QuantumLatticeTerm(
                        (operators[left][key[0]], operators[right][key[1]]),
                        coefficient=coefficient,
                        add_adjoint=True,
                        label=f"dmi-{bond}-{key[0]}-{key[1]}",
                    )
                )
            visited.add(key)
            visited.add(adjoint_key)
    return _base_model(
        site_ids,
        twice_spins,
        terms,
        energy_unit=energy_unit,
        kind="dmi-quantum-spin-model",
        spaces=spaces,
    )


def dmi_z_spin_model(
    site_ids: Sequence[str],
    twice_spins: Sequence[int],
    bonds: ArrayLike,
    dmi_z: ArrayLike,
    /,
    *,
    energy_unit: str,
) -> QuantumSpinModel:
    """Build once-oriented ``-D_z (S_i×S_j)_z``; reversing a bond flips D."""

    values = np.asarray(dmi_z)
    if values.ndim != 1:
        raise ValueError("dmi_z must be a vector with one value per bond.")
    vectors = np.zeros((values.size, 3), dtype=np.result_type(values.dtype, float))
    vectors[:, 2] = values
    return dmi_spin_model(
        site_ids,
        twice_spins,
        bonds,
        vectors,
        energy_unit=energy_unit,
    )


def spin_one_anisotropy_model(
    site_ids: Sequence[str],
    anisotropy: ArrayLike,
    longitudinal_fields: ArrayLike,
    /,
    *,
    energy_unit: str,
) -> QuantumSpinModel:
    """Build spin-I onsite ``-K Sz²-h Sz`` with I=1 at every site."""

    spins = (2,) * len(tuple(site_ids))
    spaces, operators = _spaces_and_operators(site_ids, spins)
    k = np.asarray(anisotropy)
    fields = np.asarray(longitudinal_fields)
    if k.shape != (len(spaces),) or fields.shape != (len(spaces),):
        raise ValueError("Spin-one anisotropy and fields must match site count.")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(fields)):
        raise ValueError("Spin-one onsite coefficients must be finite.")
    terms = []
    for site in range(len(spaces)):
        sz_squared = LocalOperatorPlan(
            spaces[site],
            "Sz-squared",
            operators[site]["z"].matrix @ operators[site]["z"].matrix,
            (0,),
        )
        terms.append(
            QuantumLatticeTerm(
                (sz_squared,), coefficient=-k[site], label=f"spin-one-anisotropy-{site}"
            )
        )
        terms.append(
            QuantumLatticeTerm(
                (operators[site]["z"],),
                coefficient=-fields[site],
                label=f"spin-one-field-{site}",
            )
        )
    return _base_model(
        site_ids,
        spins,
        terms,
        energy_unit=energy_unit,
        kind="spin-one-anisotropy-quantum-spin-model",
        spaces=spaces,
    )


def prepare_quantum_spin_model(
    model: QuantumSpinModel, resources: QuantumLatticeResourcePolicy, /
) -> PreparedQuantumLattice:
    if not isinstance(model, QuantumSpinModel):
        raise TypeError("model must be QuantumSpinModel.")
    return prepare_quantum_lattice(model.specification, resources)


def quantum_spin_sector_operator(
    model: QuantumSpinModel,
    prepared: PreparedQuantumLattice,
    twice_projection: int,
    /,
    *,
    sector_resources: SectorBasisResourcePolicy,
) -> QuantumSectorOperator:
    if not isinstance(model, QuantumSpinModel) or not isinstance(
        prepared, PreparedQuantumLattice
    ):
        raise TypeError("Expected QuantumSpinModel and PreparedQuantumLattice.")
    if prepared.specification.specification_id != model.specification.specification_id:
        raise ValueError("Prepared lattice belongs to another quantum spin model.")
    basis = FixedSpinProjectionBasis(
        model.site_ids,
        model.twice_spins,
        twice_projection,
        resources=sector_resources,
    )
    return QuantumSectorOperator(prepared, SectorChargeMap(basis, basis, 0))


__all__ = [
    "QuantumSpinModel",
    "QuantumSpinObservablePlan",
    "dmi_spin_model",
    "dmi_z_spin_model",
    "heisenberg_spin_model",
    "prepare_quantum_spin_model",
    "quantum_spin_sector_operator",
    "spin_matrices",
    "spin_one_anisotropy_model",
    "transverse_field_ising_model",
    "xxz_spin_model",
]
