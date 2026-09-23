#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Uniform tensor-network values with explicit Abelian unit-cell charges."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..operators.quantum._abelian_charge import AbelianGroup
from ._uniform import UniformMatrixProductOperator, UniformMatrixProductState


class UniformAbelianMatrixProductState(StrictModule):
    state: UniformMatrixProductState
    group: AbelianGroup = eqx.field(static=True)
    charge_labels: tuple[str, ...] = eqx.field(static=True)
    physical_charges: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    unit_cell_charge: tuple[int, ...] = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: UniformMatrixProductState,
        group: AbelianGroup,
        charge_labels: Sequence[str],
        physical_charges: Sequence[Sequence[Sequence[int]]],
        unit_cell_charge: Sequence[int],
        /,
    ):
        if not isinstance(state, UniformMatrixProductState) or not isinstance(
            group, AbelianGroup
        ):
            raise TypeError("state and group have invalid types.")
        labels = tuple(str(value).strip() for value in charge_labels)
        if (
            len(labels) != len(group.components)
            or any(not value for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("charge_labels must uniquely name every Abelian component.")
        charges = tuple(
            tuple(group.normalize(value) for value in site) for site in physical_charges
        )
        cell_charge = group.normalize(unit_cell_charge)
        if len(charges) != state.unit_cell_size or any(
            len(site) != tensor.shape[1]
            for site, tensor in zip(charges, state.tensors, strict=True)
        ):
            raise ValueError("Uniform physical charges do not match the MPS unit cell.")
        self.state = state
        self.group = group
        self.charge_labels = labels
        self.physical_charges = charges
        self.unit_cell_charge = cell_charge
        self.state_id = canonical_fingerprint(
            {
                "kind": "uniform-abelian-mps",
                "state": state.structure_id,
                "group": group.group_id,
                "charge_labels": labels,
                "physical_charges": charges,
                "unit_cell_charge": cell_charge,
            }
        )


class UniformAbelianMatrixProductOperator(StrictModule):
    operator: UniformMatrixProductOperator
    group: AbelianGroup = eqx.field(static=True)
    charge_labels: tuple[str, ...] = eqx.field(static=True)
    physical_charges: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: UniformMatrixProductOperator,
        group: AbelianGroup,
        charge_labels: Sequence[str],
        physical_charges: Sequence[Sequence[Sequence[int]]],
        /,
    ):
        if not isinstance(operator, UniformMatrixProductOperator) or not isinstance(
            group, AbelianGroup
        ):
            raise TypeError("operator and group have invalid types.")
        labels = tuple(str(value).strip() for value in charge_labels)
        if (
            len(labels) != len(group.components)
            or any(not value for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("charge_labels must uniquely name every Abelian component.")
        charges = tuple(
            tuple(group.normalize(value) for value in site) for site in physical_charges
        )
        if len(charges) != operator.unit_cell_size or any(
            len(site) != tensor.shape[1] or tensor.shape[1] != tensor.shape[2]
            for site, tensor in zip(charges, operator.tensors, strict=True)
        ):
            raise ValueError("Uniform physical charges do not match the MPO unit cell.")
        self.operator = operator
        self.group = group
        self.charge_labels = labels
        self.physical_charges = charges
        self.operator_id = canonical_fingerprint(
            {
                "kind": "uniform-abelian-mpo",
                "operator": operator.structure_id,
                "group": group.group_id,
                "charge_labels": labels,
                "physical_charges": charges,
            }
        )


__all__ = [
    "UniformAbelianMatrixProductOperator",
    "UniformAbelianMatrixProductState",
]
