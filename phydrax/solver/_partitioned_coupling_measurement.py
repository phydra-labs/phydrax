#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscreteMeasure, FieldRepresentation, MeasureNormalization
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    FunctionLinearOperator,
)
from ..typing import checked, parse
from ..units import ONE, UnitDefinition


CouplingMeasurementRepresentation: TypeAlias = Literal[
    "density",
    "extensive",
    "functional",
]
_FieldStorage: TypeAlias = Literal["density", "extensive", "declared"]


def _identifier(value: str, role: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{role} must be a non-empty stripped string.")
    return value


def _component_ids(values: Sequence[str], /) -> tuple[str, ...]:
    components = tuple(_identifier(value, "Measurement component ID") for value in values)
    if not components or len(set(components)) != len(components):
        raise ValueError("Measurement component IDs must be unique and non-empty.")
    return components


def _space_dtype(space: AbstractVectorSpace, /) -> np.dtype:
    leaves = jax.tree.leaves(space.structure())
    return np.dtype(jnp.result_type(*(leaf.dtype for leaf in leaves)))


def field_storage(representation: FieldRepresentation, /) -> _FieldStorage:
    """Return whether stored coordinates are densities, extensive amounts, or neither.

    Density coordinates integrate against a physical measure; extensive coordinates
    already carry their amount and are summed without a second measure weight.
    Every other representation requires an explicitly declared functional.
    """
    match representation:
        case "point_value" | "cell_average" | "basis_coefficient":
            return "density"
        case "cell_integral" | "flux_moment" | "circulation_moment" | "cochain":
            return "extensive"
        case (
            "polynomial_moment"
            | "modal_coefficient"
            | "particle_value"
            | "functional"
            | "custom"
        ):
            return "declared"
        case _:
            assert_never(representation)


def _dual_rows(functional: AbstractLinearOperator, count: int, /) -> np.ndarray:
    """Host covectors of every inventory component via transposed actions."""
    target = functional.target
    dtype = _space_dtype(target)
    rows = []
    for component in range(count):
        basis = jnp.zeros((count,), dtype=dtype).at[component].set(1)
        covector = functional.transpose_mv(target.unflatten(basis))
        rows.append(np.asarray(functional.source.flatten(covector)))
    return np.stack(rows)


def _validate_structure(
    representation: CouplingMeasurementRepresentation,
    normalization: MeasureNormalization,
    unit: UnitDefinition,
    rows: np.ndarray,
    /,
) -> None:
    if not np.all(np.isfinite(rows)):
        raise ValueError("Measurement functional covectors must be finite.")
    match representation:
        case "density":
            if normalization != "physical":
                raise ValueError("A density measurement requires a physical measure.")
            if np.any(rows < 0) or np.any(np.max(rows, axis=1) <= 0):
                raise ValueError(
                    "A density measurement integrates against nonnegative physical "
                    "weights with positive mass per component."
                )
        case "extensive":
            if normalization != "counting":
                raise ValueError("An extensive measurement uses counting normalization.")
            if not unit.dimension.is_dimensionless or unit.scale_to_reference != 1:
                raise ValueError(
                    "Extensive content is not weighted by a dimensional measure again."
                )
            if not np.all((rows == 0) | (rows == 1)) or np.any(rows.sum(axis=0) > 1):
                raise ValueError(
                    "An extensive measurement sums each stored amount exactly once; "
                    "re-weighted coordinates double count the measure."
                )
        case "functional":
            pass
        case _:
            assert_never(representation)


class CouplingMeasurement(StrictModule, NonTrainableState):
    """Prepared linear inventory functional of one port's native coordinates.

    `inventory(value)[k]` is the physical amount of component `k`, in the port
    quantity's unit multiplied by `unit`. `density` coordinates are integrated
    against nonnegative physical weights, `extensive` coordinates are summed
    exactly once per component, and a `functional` owns an explicitly declared
    general linear action such as basis or flux-moment integrals. Conservation
    certificates and budgets use the same functional through its transposed
    action, never a dense transfer matrix. `covector_norms[k]` is the ℓ¹ norm of
    component `k`'s covector: the largest inventory of a signal whose coordinates
    are bounded by one, which scales ledger rounding bounds.
    """

    functional: AbstractLinearOperator
    unit: UnitDefinition
    component_ids: tuple[str, ...] = eqx.field(static=True)
    representation: CouplingMeasurementRepresentation = eqx.field(static=True)
    normalization: MeasureNormalization = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    covector_norms: tuple[float, ...] = eqx.field(static=True)
    measurement_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        functional: AbstractLinearOperator,
        unit: UnitDefinition,
        /,
        *,
        representation: CouplingMeasurementRepresentation,
        support_id: str,
        provenance_id: str,
        component_ids: Sequence[str] = ("scalar",),
        normalization: MeasureNormalization = "physical",
    ) -> None:
        representation = parse(
            representation, CouplingMeasurementRepresentation, "representation"
        )
        normalization = parse(normalization, MeasureNormalization, "normalization")
        components = _component_ids(component_ids)
        support = _identifier(support_id, "Measurement support_id")
        provenance = _identifier(provenance_id, "Measurement provenance_id")
        target = functional.target
        if not isinstance(target, ArraySpace) or target.shape != (len(components),):
            raise ValueError(
                "Measurement functional must map into one inventory entry per component."
            )
        if not np.issubdtype(target.dtype, np.floating):
            raise TypeError("Physical inventories must use a real floating dtype.")
        if not functional.capabilities.transpose:
            raise ValueError(
                "Measurement functional requires a transposed action for dual "
                "conservation certificates."
            )
        rows = _dual_rows(functional, len(components))
        _validate_structure(representation, normalization, unit, rows)
        self.functional = functional
        self.unit = unit
        self.component_ids = components
        self.representation = representation
        self.normalization = normalization
        self.support_id = support
        self.provenance_id = provenance
        self.covector_norms = tuple(float(norm) for norm in np.sum(np.abs(rows), axis=1))
        self.measurement_id = canonical_fingerprint(
            {
                "kind": "coupling-measurement",
                "representation": representation,
                "normalization": normalization,
                "unit": unit.to_dict(),
                "components": list(components),
                "support": support,
                "provenance": provenance,
                "source_space": functional.source.space_id,
                "covectors": array_tree_fingerprint(rows),
            }
        )

    @property
    def component_count(self) -> int:
        return len(self.component_ids)

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.functional.source

    @property
    def inventory_dtype(self) -> np.dtype:
        return _space_dtype(self.functional.target)

    def inventory(self, value: Any, /) -> Array:
        """Return component amounts of `value` in the quantity unit times `unit`."""
        return self.functional.mv(value)

    def covector(self, weights: Array, /) -> Any:
        """Pull inventory weights back to the port's native coordinates."""
        return self.functional.transpose_mv(weights)

    @classmethod
    @checked
    def from_measure(
        cls,
        measure: DiscreteMeasure,
        space: AbstractVectorSpace,
        unit: UnitDefinition,
        /,
        *,
        component_ids: Sequence[str] = ("scalar",),
    ) -> CouplingMeasurement:
        """Integrate entity-major density coordinates against one physical measure.

        Coordinates flatten as `(entity, component)` with components fastest, the
        layout of `DiscreteFieldSpace` coefficient events.
        """
        components = _component_ids(component_ids)
        entities = measure.weights.size
        width = len(components)
        if space.size != entities * width:
            raise ValueError(
                "A measured space requires one weight per entity for every component."
            )
        dtype = _space_dtype(space)
        weights = measure.masked_weights().astype(dtype)
        inventory = ArraySpace((width,), dtype=dtype)

        def integrate(value: Any) -> Array:
            return measure.integrate(space.flatten(value).reshape(entities, width))

        def pull_back(amount: Array) -> Any:
            return space.unflatten((weights[:, None] * amount[None, :]).reshape(-1))

        functional = FunctionLinearOperator(
            integrate,
            source=space,
            target=inventory,
            transpose_action=pull_back,
            operator_id=f"coupling-measurement:{measure.measure_id}:{space.space_id}",
        )
        return cls(
            functional,
            unit,
            representation="density",
            support_id=measure.support_id,
            provenance_id=measure.measure_id,
            component_ids=components,
            normalization=measure.normalization,
        )

    @classmethod
    @checked
    def extensive(
        cls,
        space: AbstractVectorSpace,
        support_id: str,
        /,
        *,
        provenance_id: str,
        component_ids: Sequence[str] = ("scalar",),
    ) -> CouplingMeasurement:
        """Sum entity-major extensive amounts once per component, without weights."""
        components = _component_ids(component_ids)
        width = len(components)
        if space.size % width:
            raise ValueError(
                "Extensive coordinates must hold every component per entity."
            )
        entities = space.size // width
        dtype = _space_dtype(space)
        inventory = ArraySpace((width,), dtype=dtype)

        def total(value: Any) -> Array:
            return jnp.sum(space.flatten(value).reshape(entities, width), axis=0)

        def spread(amount: Array) -> Any:
            return space.unflatten(
                jnp.broadcast_to(amount, (entities, width)).reshape(-1)
            )

        functional = FunctionLinearOperator(
            total,
            source=space,
            target=inventory,
            transpose_action=spread,
            operator_id=f"coupling-extensive-measurement:{space.space_id}",
        )
        return cls(
            functional,
            ONE,
            representation="extensive",
            support_id=support_id,
            provenance_id=provenance_id,
            component_ids=components,
            normalization="counting",
        )


__all__ = [
    "CouplingMeasurement",
    "CouplingMeasurementRepresentation",
    "field_storage",
]
