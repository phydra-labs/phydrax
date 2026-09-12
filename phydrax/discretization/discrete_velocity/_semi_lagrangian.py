#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import floor, isfinite, prod
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    DiagonalPairing,
    FunctionLinearOperator,
    TransposeLinearOperator,
)
from .._core import DiscretizationCapability, PreparationReport
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from .._transfer import FieldTransfer, TransferProperties
from ._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v37_off_lattice_quadrature,
)


if TYPE_CHECKING:
    from ._smooth_compressible import (
        SmoothCompressibleD2VKineticMethod,
        SmoothCompressibleKineticState,
    )


class DeclaredPopulationMomentMap(StrictModule, NonTrainableState):
    """Identified linear map from population integrals to declared moments."""

    coefficients: Array
    quadrature_id: str = eqx.field(static=True)
    moment_names: tuple[str, ...] = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        coefficients: ArrayLike,
        /,
        *,
        moment_names: Sequence[str],
        name: str,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        values = np.asarray(coefficients)
        names = tuple(str(value) for value in moment_names)
        name_ = str(name)
        if (
            values.ndim != 2
            or values.shape[1] != quadrature.population_count
            or values.shape[0] == 0
        ):
            raise ValueError(
                "Declared-moment coefficients must have non-empty shape (M, Q)."
            )
        if not np.issubdtype(values.dtype, np.number) or np.any(~np.isfinite(values)):
            raise ValueError(
                "Declared-moment coefficients must be finite numeric values."
            )
        if len(names) != values.shape[0] or any(not value for value in names):
            raise ValueError("moment_names must contain one non-empty name per map row.")
        if len(set(names)) != len(names):
            raise ValueError("Declared moment names must be unique.")
        if not name_:
            raise ValueError("Declared-moment map name must be non-empty.")
        coefficients_ = jnp.asarray(values, dtype=quadrature.velocities.dtype)
        self.coefficients = coefficients_
        self.quadrature_id = quadrature.quadrature_id
        self.moment_names = names
        self.map_id = canonical_fingerprint(
            {
                "kind": "declared-population-moment-map",
                "name": name_,
                "quadrature": quadrature.quadrature_id,
                "moment_names": list(names),
                "coefficients": np.asarray(coefficients_).tolist(),
            }
        )

    @classmethod
    def population_integrals(
        cls, quadrature: CertifiedDiscreteVelocityQuadrature, /
    ) -> "DeclaredPopulationMomentMap":
        return cls(
            quadrature,
            np.eye(quadrature.population_count),
            moment_names=tuple(
                f"population_{index}_integral"
                for index in range(quadrature.population_count)
            ),
            name="population-integrals",
        )

    def validate_quadrature(
        self, quadrature: CertifiedDiscreteVelocityQuadrature, /
    ) -> None:
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if quadrature.quadrature_id != self.quadrature_id:
            raise ValueError("Declared-moment map and transport quadrature do not match.")

    def evaluate(self, population_integrals: ArrayLike, /) -> Array:
        values = jnp.asarray(population_integrals)
        if values.shape != (self.coefficients.shape[1],):
            raise ValueError(
                "Population integrals must have the declared trailing population shape."
            )
        return ein.contract("mq,q->m", self.coefficients.astype(values.dtype), values)


class PeriodicUniformGridDepartureTransfer(StrictModule, NonTrainableState):
    """Prepared conservative positive bilinear pull on a periodic uniform 2-D grid."""

    field_transfer: FieldTransfer
    spatial_shape: tuple[int, int] = eqx.field(static=True)
    cell_spacing: tuple[float, float] = eqx.field(static=True)
    offset_in_cells: tuple[float, float] = eqx.field(static=True)
    integer_offset: tuple[int, int] = eqx.field(static=True)
    fractional_offset: tuple[float, float] = eqx.field(static=True)
    periodic_axes: tuple[bool, bool] = eqx.field(static=True)

    def __init__(
        self,
        spatial_shape: Sequence[int],
        cell_spacing: Sequence[float],
        offset_in_cells: Sequence[float],
        /,
        *,
        periodic_axes: Sequence[bool] = (True, True),
        dtype: object = jnp.float64,
    ):
        shape = tuple(int(value) for value in spatial_shape)
        spacing = tuple(float(value) for value in cell_spacing)
        offset = tuple(float(value) for value in offset_in_cells)
        periodic = tuple(bool(value) for value in periodic_axes)
        if len(shape) != 2 or any(value < 2 for value in shape):
            raise ValueError(
                "Periodic multilinear departure transfer requires a 2-D shape "
                "with at least two cells per axis."
            )
        if len(spacing) != 2 or any(
            not isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError("cell_spacing must contain two finite positive values.")
        if len(offset) != 2 or any(not isfinite(value) for value in offset):
            raise ValueError("offset_in_cells must contain two finite values.")
        if len(periodic) != 2 or not all(periodic):
            raise ValueError(
                "Periodic multilinear departure transfer requires periodic 2-D geometry."
            )
        integer = tuple(floor(value) for value in offset)
        fractional = tuple(
            value - base for value, base in zip(offset, integer, strict=True)
        )
        volume = prod(spacing)
        pairing = DiagonalPairing(
            jnp.full(shape, volume, dtype=dtype),
            pairing_id=canonical_fingerprint(
                {
                    "kind": "uniform-cell-volume-pairing",
                    "shape": list(shape),
                    "cell_spacing": list(spacing),
                }
            ),
        )
        vector_space = ArraySpace(shape, dtype=dtype, pairing=pairing)
        field_space = DiscreteFieldSpace(
            "periodic_uniform_cell_average",
            canonical_fingerprint(
                {
                    "kind": "periodic-uniform-grid",
                    "shape": list(shape),
                    "cell_spacing": list(spacing),
                }
            ),
            TensorDofLayout(("x", "y"), shape),
            vector_space,
            representation="cell_average",
            conformity="discontinuous",
        )
        shifts = (
            integer,
            (integer[0] + 1, integer[1]),
            (integer[0], integer[1] + 1),
            (integer[0] + 1, integer[1] + 1),
        )
        fx, fy = fractional
        weights = (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        )

        def pull(values: Array) -> Array:
            result = jnp.zeros_like(values)
            for weight, shift in zip(weights, shifts, strict=True):
                result = result + weight * jnp.roll(values, shift=shift, axis=(0, 1))
            return result

        def transpose_pull(values: Array) -> Array:
            result = jnp.zeros_like(values)
            for weight, shift in zip(weights, shifts, strict=True):
                reverse = (-shift[0], -shift[1])
                result = result + weight * jnp.roll(values, shift=reverse, axis=(0, 1))
            return result

        operator_id = canonical_fingerprint(
            {
                "kind": "periodic-uniform-grid-bilinear-departure",
                "field_space": field_space.field_space_id,
                "offset_in_cells": list(offset),
                "integer_offset": list(integer),
                "fractional_offset": list(fractional),
            }
        )
        operator = FunctionLinearOperator(
            pull,
            source=vector_space,
            target=vector_space,
            transpose_action=transpose_pull,
            operator_id=operator_id,
        )
        properties = TransferProperties(
            constant_preserving=True,
            conservative=True,
            positivity_preserving=True,
            differentiable_geometry=False,
            exact_on=(
                "periodic_constant",
                "periodic_cell_integral",
                "fixed_offset_bilinear_departure",
            ),
        )
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.FIELD_TRANSFER,
                DiscretizationCapability.MATRIX_FREE,
            ),
            diagnostics=(
                "all four bilinear weights are non-negative and sum to one",
                "periodic routes make the transfer matrix doubly stochastic",
                "offset geometry is fixed at preparation",
            ),
            resource_counts={"departure_corners": 4, "spatial_dimensions": 2},
        )
        self.field_transfer = FieldTransfer(
            field_space,
            field_space,
            operator,
            dual_pullback_operator=TransposeLinearOperator(operator),
            hilbert_adjoint_operator=TransposeLinearOperator(operator),
            properties=properties,
            preparation=preparation,
            transfer_id=canonical_fingerprint(
                {
                    "kind": "periodic-uniform-grid-departure-transfer",
                    "operator": operator_id,
                    "properties": {
                        "constant_preserving": True,
                        "conservative": True,
                        "positivity_preserving": True,
                        "integer_roll_exactness_claimed": False,
                    },
                }
            ),
        )
        self.spatial_shape = shape
        self.cell_spacing = spacing
        self.offset_in_cells = offset
        self.integer_offset = integer
        self.fractional_offset = fractional
        self.periodic_axes = periodic

    @property
    def source(self) -> DiscreteFieldSpace:
        return self.field_transfer.source

    @property
    def target(self) -> DiscreteFieldSpace:
        return self.field_transfer.target

    @property
    def primal_operator(self) -> FunctionLinearOperator:
        return self.field_transfer.primal_operator

    @property
    def properties(self) -> TransferProperties:
        return self.field_transfer.properties

    @property
    def transfer_id(self) -> str:
        return self.field_transfer.transfer_id


class SemiLagrangianTransferRequirements(StrictModule, NonTrainableState):
    """Capabilities required from every prepared departure-point transfer."""

    constant_preserving: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    positivity_preserving: bool = eqx.field(static=True)
    differentiable_geometry: bool = eqx.field(static=True)
    exact_on: tuple[str, ...] = eqx.field(static=True)
    requirement_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        constant_preserving: bool = True,
        conservative: bool = True,
        positivity_preserving: bool = True,
        differentiable_geometry: bool = False,
        exact_on: Sequence[str] = (),
    ):
        exact = tuple(str(value) for value in exact_on)
        if any(not value for value in exact) or len(set(exact)) != len(exact):
            raise ValueError("exact_on requirements must be unique non-empty strings.")
        self.constant_preserving = bool(constant_preserving)
        self.conservative = bool(conservative)
        self.positivity_preserving = bool(positivity_preserving)
        self.differentiable_geometry = bool(differentiable_geometry)
        self.exact_on = exact
        self.requirement_id = canonical_fingerprint(
            {
                "kind": "semi-lagrangian-transfer-requirements-v1",
                "constant_preserving": bool(constant_preserving),
                "conservative": bool(conservative),
                "positivity_preserving": bool(positivity_preserving),
                "differentiable_geometry": bool(differentiable_geometry),
                "exact_on": list(exact),
            }
        )

    def validate(
        self,
        transfer: FieldTransfer | PeriodicUniformGridDepartureTransfer,
        /,
    ) -> None:
        value = (
            transfer.field_transfer
            if isinstance(transfer, PeriodicUniformGridDepartureTransfer)
            else transfer
        )
        if not isinstance(value, FieldTransfer):
            raise TypeError(
                "Semi-Lagrangian population transfers must be FieldTransfer values."
            )
        transfer = value
        properties = transfer.properties
        missing = []
        for name in (
            "constant_preserving",
            "conservative",
            "positivity_preserving",
            "differentiable_geometry",
        ):
            if getattr(self, name) and not getattr(properties, name):
                missing.append(name)
        unavailable_exactness = tuple(
            name for name in self.exact_on if name not in properties.exact_on
        )
        if missing or unavailable_exactness:
            details = []
            if missing:
                details.append("properties=" + ",".join(missing))
            if unavailable_exactness:
                details.append("exact_on=" + ",".join(unavailable_exactness))
            raise ValueError(
                "FieldTransfer does not satisfy semi-Lagrangian requirements: "
                + "; ".join(details)
            )


class SemiLagrangianTransportEvidence(StrictModule):
    """Population, declared-moment, and positivity audit for one transfer."""

    source_population_integrals: Array
    target_population_integrals: Array
    population_conservation_residual: Array
    maximum_absolute_population_residual: Array
    source_moments: Array
    target_moments: Array
    conservation_residual: Array
    maximum_absolute_residual: Array
    minimum_source_population: Array
    minimum_target_population: Array
    source_nonnegative: Array
    target_nonnegative: Array
    positivity_preserved: Array
    declared_moment_map_id: str = eqx.field(static=True)
    declared_moment_names: tuple[str, ...] = eqx.field(static=True)


class PreparedOffLatticeSemiLagrangianDVM(StrictModule, NonTrainableState):
    """Fixed-step, off-lattice departure transfer for trailing-Q populations.

    Each velocity owns a prepared ``FieldTransfer``. No interpolation is built at
    execution time, and capability claims are checked before the plan can exist.
    """

    quadrature: CertifiedDiscreteVelocityQuadrature
    population_transfers: tuple[FieldTransfer, ...]
    departure_transfers: tuple[PeriodicUniformGridDepartureTransfer, ...]
    requirements: SemiLagrangianTransferRequirements
    declared_moments: DeclaredPopulationMomentMap
    time_step: float = eqx.field(static=True)
    source_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        population_transfers: Sequence[
            FieldTransfer | PeriodicUniformGridDepartureTransfer
        ],
        time_step: float,
        /,
        *,
        requirements: SemiLagrangianTransferRequirements | None = None,
        declared_moments: DeclaredPopulationMomentMap | None = None,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if quadrature.transport_kind != "off_lattice":
            raise ValueError(
                "PreparedOffLatticeSemiLagrangianDVM requires an explicitly off-lattice quadrature."
            )
        declared_transfers = tuple(population_transfers)
        if not all(
            isinstance(transfer, (FieldTransfer, PeriodicUniformGridDepartureTransfer))
            for transfer in declared_transfers
        ):
            raise TypeError("population_transfers must contain prepared field transfers.")
        transfers = tuple(
            transfer.field_transfer
            if isinstance(transfer, PeriodicUniformGridDepartureTransfer)
            else transfer
            for transfer in declared_transfers
        )
        if len(transfers) != quadrature.population_count:
            raise ValueError(
                "One prepared FieldTransfer is required per discrete velocity."
            )
        requirements_ = (
            SemiLagrangianTransferRequirements() if requirements is None else requirements
        )
        if not isinstance(requirements_, SemiLagrangianTransferRequirements):
            raise TypeError("requirements must be SemiLagrangianTransferRequirements.")
        step = float(time_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("Semi-Lagrangian time_step must be finite and positive.")
        moment_map = (
            DeclaredPopulationMomentMap.population_integrals(quadrature)
            if declared_moments is None
            else declared_moments
        )
        if not isinstance(moment_map, DeclaredPopulationMomentMap):
            raise TypeError("declared_moments must be a DeclaredPopulationMomentMap.")
        moment_map.validate_quadrature(quadrature)
        for transfer in transfers:
            requirements_.validate(transfer)
        source = transfers[0].source
        target = transfers[0].target
        if not isinstance(source.vector_space, ArraySpace) or not isinstance(
            target.vector_space, ArraySpace
        ):
            raise TypeError(
                "Semi-Lagrangian transfers require array-valued field spaces."
            )
        for transfer in transfers[1:]:
            if not transfer.source.vector_space.compatible(
                source.vector_space
            ) or not transfer.target.vector_space.compatible(target.vector_space):
                raise ValueError(
                    "Every population transfer must share exact source and target spaces."
                )
        self.quadrature = quadrature
        self.population_transfers = transfers
        self.departure_transfers = (
            tuple(
                transfer
                for transfer in declared_transfers
                if isinstance(transfer, PeriodicUniformGridDepartureTransfer)
            )
            if all(
                isinstance(transfer, PeriodicUniformGridDepartureTransfer)
                for transfer in declared_transfers
            )
            else ()
        )
        self.requirements = requirements_
        self.declared_moments = moment_map
        self.time_step = step
        self.source_shape = source.vector_space.shape
        self.target_shape = target.vector_space.shape
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-off-lattice-semi-lagrangian-dvm-v1",
                "quadrature": quadrature.quadrature_id,
                "transfers": [transfer.transfer_id for transfer in transfers],
                "requirements": requirements_.requirement_id,
                "declared_moments": moment_map.map_id,
                "time_step": step,
            }
        )

    def transport(self, populations: ArrayLike, /) -> Array:
        values = self.quadrature.validate_populations(populations)
        if values.shape != self.source_shape + (self.quadrature.population_count,):
            raise ValueError(
                "Semi-Lagrangian source populations must have shape "
                f"{self.source_shape + (self.quadrature.population_count,)}."
            )
        return jnp.stack(
            tuple(
                transfer.primal_operator.mv(values[..., population])
                for population, transfer in enumerate(self.population_transfers)
            ),
            axis=-1,
        )

    @staticmethod
    def _population_integrals(space: ArraySpace, populations: Array, /) -> Array:
        ones = jnp.ones(space.shape, dtype=space.dtype)
        return jnp.stack(
            tuple(
                space.inner(ones, populations[..., index])
                for index in range(populations.shape[-1])
            )
        )

    def transport_with_evidence(
        self,
        populations: ArrayLike,
        /,
        *,
        declared_moments: DeclaredPopulationMomentMap | None = None,
    ) -> tuple[Array, SemiLagrangianTransportEvidence]:
        values = self.quadrature.validate_populations(populations)
        transported = self.transport(values)
        source_space = self.population_transfers[0].source.vector_space
        target_space = self.population_transfers[0].target.vector_space
        if not isinstance(source_space, ArraySpace) or not isinstance(
            target_space, ArraySpace
        ):
            raise TypeError("Semi-Lagrangian evidence requires array field spaces.")
        moment_map = (
            self.declared_moments if declared_moments is None else declared_moments
        )
        if not isinstance(moment_map, DeclaredPopulationMomentMap):
            raise TypeError("declared_moments must be a DeclaredPopulationMomentMap.")
        moment_map.validate_quadrature(self.quadrature)
        source_integrals = self._population_integrals(source_space, values)
        target_integrals = self._population_integrals(target_space, transported)
        population_residual = target_integrals - source_integrals
        source_moments = moment_map.evaluate(source_integrals)
        target_moments = moment_map.evaluate(target_integrals)
        residual = target_moments - source_moments
        minimum_source = jnp.min(values)
        minimum_target = jnp.min(transported)
        source_nonnegative = minimum_source >= 0.0
        target_nonnegative = minimum_target >= 0.0
        return transported, SemiLagrangianTransportEvidence(
            source_population_integrals=source_integrals,
            target_population_integrals=target_integrals,
            population_conservation_residual=population_residual,
            maximum_absolute_population_residual=jnp.max(jnp.abs(population_residual)),
            source_moments=source_moments,
            target_moments=target_moments,
            conservation_residual=residual,
            maximum_absolute_residual=jnp.max(jnp.abs(residual)),
            minimum_source_population=minimum_source,
            minimum_target_population=minimum_target,
            source_nonnegative=source_nonnegative,
            target_nonnegative=target_nonnegative,
            positivity_preserved=(~source_nonnegative) | target_nonnegative,
            declared_moment_map_id=moment_map.map_id,
            declared_moment_names=moment_map.moment_names,
        )


class CoupledD2V37TransportStatus(IntEnum):
    SUCCESS = 0
    FIXED_STEP_MISMATCH = 1
    NONFINITE_POPULATIONS = 2
    NEGATIVE_POPULATIONS = 3
    CONSERVATION_FAILED = 4


class CoupledD2V37TransportEvidence(StrictModule):
    """Complete evidence for coupled f/g transport through one transfer tuple."""

    f: SemiLagrangianTransportEvidence
    g: SemiLagrangianTransportEvidence
    finite: Array
    populations_nonnegative: Array
    positivity_preserved: Array
    maximum_absolute_population_residual: Array
    maximum_absolute_declared_moment_residual: Array
    status: Array
    successful: Array
    transport_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class CoupledD2V37TransportResult(StrictModule):
    candidate_state: SmoothCompressibleKineticState
    evidence: CoupledD2V37TransportEvidence
    successful: Array
    status: Array
    transport_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedCoupledD2V37OffLatticeTransport(StrictModule, NonTrainableState):
    """Fixed-step periodic D2V37 transport for coupled f and g populations only."""

    quadrature: CertifiedDiscreteVelocityQuadrature
    population_transport: PreparedOffLatticeSemiLagrangianDVM
    f_declared_moments: DeclaredPopulationMomentMap
    g_declared_moments: DeclaredPopulationMomentMap
    method_id: str = eqx.field(static=True)
    cell_spacing: tuple[float, float] = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: SmoothCompressibleD2VKineticMethod,
        population_transport: PreparedOffLatticeSemiLagrangianDVM,
        cell_spacing: Sequence[float],
        /,
        *,
        conservation_tolerance: float = 1.0e-11,
    ):
        from ._smooth_compressible import SmoothCompressibleD2VKineticMethod

        if not isinstance(method, SmoothCompressibleD2VKineticMethod):
            raise TypeError("method must be a SmoothCompressibleD2VKineticMethod.")
        if not isinstance(population_transport, PreparedOffLatticeSemiLagrangianDVM):
            raise TypeError(
                "population_transport must be a PreparedOffLatticeSemiLagrangianDVM."
            )
        quadrature = method.quadrature
        expected = d2v37_off_lattice_quadrature(dtype=quadrature.velocities.dtype)
        if quadrature.quadrature_id != expected.quadrature_id:
            raise ValueError(
                "Coupled off-lattice transport requires the declared D2V37 quadrature identity."
            )
        if population_transport.quadrature.quadrature_id != quadrature.quadrature_id:
            raise ValueError(
                "Coupled transport method and prepared quadrature identities do not match."
            )
        spacing = tuple(float(value) for value in cell_spacing)
        if len(spacing) != 2 or any(
            not isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError("cell_spacing must contain two finite positive values.")
        tolerance = float(conservation_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be finite and positive.")
        transfers = population_transport.departure_transfers
        if len(transfers) != quadrature.population_count or not all(
            isinstance(transfer, PeriodicUniformGridDepartureTransfer)
            for transfer in transfers
        ):
            raise TypeError(
                "Coupled D2V37 transport requires prepared periodic multilinear transfers."
            )
        scaled = np.asarray(quadrature.velocities, dtype=float)
        scaled = scaled * population_transport.time_step / np.asarray(spacing)[None, :]
        for index, transfer in enumerate(transfers):
            if transfer.cell_spacing != spacing or not all(transfer.periodic_axes):
                raise ValueError(
                    "Every D2V37 transfer must use the declared periodic uniform geometry."
                )
            actual = np.asarray(transfer.offset_in_cells)
            expected_offset = scaled[index]
            scale = max(float(np.max(np.abs(expected_offset))), 1.0)
            offset_tolerance = 64.0 * np.finfo(float).eps * scale
            if float(np.max(np.abs(actual - expected_offset))) > offset_tolerance:
                raise ValueError(
                    "Prepared departure offsets must equal velocity*time_step/cell_spacing."
                )
        f_map = DeclaredPopulationMomentMap(
            quadrature,
            jnp.concatenate(
                (
                    jnp.ones(
                        (1, quadrature.population_count),
                        dtype=quadrature.velocities.dtype,
                    ),
                    quadrature.velocities.T,
                ),
                axis=0,
            ),
            moment_names=("mass", "momentum_x", "momentum_y"),
            name="coupled-d2v37-f-conserved",
        )
        g_map = DeclaredPopulationMomentMap(
            quadrature,
            jnp.ones(
                (1, quadrature.population_count),
                dtype=quadrature.velocities.dtype,
            ),
            moment_names=("total_energy",),
            name="coupled-d2v37-g-conserved",
        )
        transport_id = canonical_fingerprint(
            {
                "kind": "coupled-d2v37-periodic-multilinear-transport",
                "quadrature": quadrature.quadrature_id,
                "population_transport": population_transport.prepared_id,
                "f_declared_moments": f_map.map_id,
                "g_declared_moments": g_map.map_id,
                "cell_spacing": list(spacing),
            }
        )
        self.quadrature = quadrature
        self.population_transport = population_transport
        self.f_declared_moments = f_map
        self.g_declared_moments = g_map
        self.method_id = method.method_id
        self.cell_spacing = spacing
        self.conservation_tolerance = tolerance
        self.transport_id = transport_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-coupled-d2v37-off-lattice-transport",
                "method": method.method_id,
                "transport": transport_id,
                "conservation_tolerance": tolerance,
            }
        )

    @classmethod
    def prepare(
        cls,
        method: SmoothCompressibleD2VKineticMethod,
        spatial_shape: Sequence[int],
        cell_spacing: Sequence[float],
        time_step: float,
        /,
        *,
        periodic_axes: Sequence[bool] = (True, True),
        conservation_tolerance: float = 1.0e-11,
    ) -> "PreparedCoupledD2V37OffLatticeTransport":
        from ._smooth_compressible import SmoothCompressibleD2VKineticMethod

        if not isinstance(method, SmoothCompressibleD2VKineticMethod):
            raise TypeError("method must be a SmoothCompressibleD2VKineticMethod.")
        quadrature = method.quadrature
        expected = d2v37_off_lattice_quadrature(dtype=quadrature.velocities.dtype)
        if quadrature.quadrature_id != expected.quadrature_id:
            raise ValueError(
                "Coupled off-lattice transport requires the declared D2V37 quadrature identity."
            )
        spacing = tuple(float(value) for value in cell_spacing)
        step = float(time_step)
        if len(spacing) != 2 or any(
            not isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError("cell_spacing must contain two finite positive values.")
        if not isfinite(step) or step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        scaled = np.asarray(quadrature.velocities, dtype=float)
        scaled = scaled * step / np.asarray(spacing)[None, :]
        transfers = tuple(
            PeriodicUniformGridDepartureTransfer(
                spatial_shape,
                spacing,
                offset,
                periodic_axes=periodic_axes,
                dtype=quadrature.velocities.dtype,
            )
            for offset in scaled
        )
        population_transport = PreparedOffLatticeSemiLagrangianDVM(
            quadrature,
            transfers,
            step,
            declared_moments=DeclaredPopulationMomentMap.population_integrals(quadrature),
        )
        return cls(
            method,
            population_transport,
            spacing,
            conservation_tolerance=conservation_tolerance,
        )

    @property
    def required_step_size(self) -> float:
        return self.population_transport.time_step

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def _validate_state(self, state: SmoothCompressibleKineticState, /) -> None:
        from ._smooth_compressible import SmoothCompressibleKineticState

        if not isinstance(state, SmoothCompressibleKineticState):
            raise TypeError("state must be a SmoothCompressibleKineticState.")
        f = self.quadrature.validate_populations(state.particle_populations)
        g = self.quadrature.validate_populations(state.total_energy_populations)
        expected = self.population_transport.source_shape + (
            self.quadrature.population_count,
        )
        if f.shape != expected or g.shape != expected:
            raise ValueError(
                f"Coupled D2V37 populations must have shape {expected} for both f and g."
            )
        if f.dtype != g.dtype or f.dtype != self.quadrature.velocities.dtype:
            raise TypeError(
                "Coupled D2V37 populations must use the prepared quadrature dtype."
            )

    def _step_matches(self, time_step: Array, /) -> Array:
        tolerance = (
            8.0
            * jnp.finfo(time_step.dtype).eps
            * jnp.maximum(jnp.abs(time_step), self.required_step_size)
        )
        return jnp.isfinite(time_step) & (
            jnp.abs(time_step - self.required_step_size) <= tolerance
        )

    def transport_with_evidence(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        /,
    ) -> CoupledD2V37TransportResult:
        from ._smooth_compressible import SmoothCompressibleKineticState

        self._validate_state(state)
        step = jnp.asarray(time_step, dtype=state.particle_populations.dtype)
        if step.shape != ():
            raise ValueError("time_step must be a scalar.")
        f, f_evidence = self.population_transport.transport_with_evidence(
            state.particle_populations,
            declared_moments=self.f_declared_moments,
        )
        g, g_evidence = self.population_transport.transport_with_evidence(
            state.total_energy_populations,
            declared_moments=self.g_declared_moments,
        )
        candidate = SmoothCompressibleKineticState(f, g)
        finite = (
            jnp.all(jnp.isfinite(state.particle_populations))
            & jnp.all(jnp.isfinite(state.total_energy_populations))
            & jnp.all(jnp.isfinite(f))
            & jnp.all(jnp.isfinite(g))
        )
        populations_nonnegative = (
            f_evidence.source_nonnegative
            & g_evidence.source_nonnegative
            & f_evidence.target_nonnegative
            & g_evidence.target_nonnegative
        )
        positivity_preserved = (
            f_evidence.positivity_preserved & g_evidence.positivity_preserved
        )
        maximum_population_residual = jnp.maximum(
            f_evidence.maximum_absolute_population_residual,
            g_evidence.maximum_absolute_population_residual,
        )
        maximum_moment_residual = jnp.maximum(
            f_evidence.maximum_absolute_residual,
            g_evidence.maximum_absolute_residual,
        )
        scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(f_evidence.source_population_integrals)),
                jnp.max(jnp.abs(g_evidence.source_population_integrals)),
            ),
            1.0,
        )
        tolerance = (
            jnp.asarray(self.conservation_tolerance, dtype=step.dtype)
            + 512.0 * jnp.finfo(step.dtype).eps * scale
        )
        conservation_ok = (maximum_population_residual <= tolerance) & (
            maximum_moment_residual <= tolerance
        )
        step_matches = self._step_matches(step)
        successful = (
            step_matches
            & finite
            & populations_nonnegative
            & positivity_preserved
            & conservation_ok
        )
        status = jnp.asarray(int(CoupledD2V37TransportStatus.SUCCESS), dtype=jnp.int32)
        status = jnp.where(
            ~step_matches,
            int(CoupledD2V37TransportStatus.FIXED_STEP_MISMATCH),
            status,
        )
        status = jnp.where(
            step_matches & ~finite,
            int(CoupledD2V37TransportStatus.NONFINITE_POPULATIONS),
            status,
        )
        status = jnp.where(
            step_matches & finite & ~populations_nonnegative,
            int(CoupledD2V37TransportStatus.NEGATIVE_POPULATIONS),
            status,
        )
        status = jnp.where(
            step_matches
            & finite
            & populations_nonnegative
            & positivity_preserved
            & ~conservation_ok,
            int(CoupledD2V37TransportStatus.CONSERVATION_FAILED),
            status,
        ).astype(jnp.int32)
        evidence = CoupledD2V37TransportEvidence(
            f=f_evidence,
            g=g_evidence,
            finite=finite,
            populations_nonnegative=populations_nonnegative,
            positivity_preserved=positivity_preserved,
            maximum_absolute_population_residual=maximum_population_residual,
            maximum_absolute_declared_moment_residual=maximum_moment_residual,
            status=status,
            successful=successful,
            transport_id=self.transport_id,
            prepared_id=self.prepared_id,
        )
        return CoupledD2V37TransportResult(
            candidate_state=candidate,
            evidence=evidence,
            successful=successful,
            status=status,
            transport_id=self.transport_id,
            prepared_id=self.prepared_id,
        )

    def transport(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        /,
    ) -> SmoothCompressibleKineticState:
        from ._smooth_compressible import SmoothCompressibleKineticState

        self._validate_state(state)
        step = jnp.asarray(time_step, dtype=state.particle_populations.dtype)
        if step.shape != ():
            raise ValueError("time_step must be a scalar.")
        checked_particles = eqx.error_if(
            state.particle_populations,
            ~self._step_matches(step),
            "Coupled D2V37 transport refuses a time step other than its "
            "prepared fixed step.",
        )
        checked_state = SmoothCompressibleKineticState(
            checked_particles, state.total_energy_populations
        )
        return self.transport_with_evidence(checked_state, step).candidate_state


__all__ = [
    "CoupledD2V37TransportEvidence",
    "CoupledD2V37TransportResult",
    "CoupledD2V37TransportStatus",
    "DeclaredPopulationMomentMap",
    "PeriodicUniformGridDepartureTransfer",
    "PreparedCoupledD2V37OffLatticeTransport",
    "PreparedOffLatticeSemiLagrangianDVM",
    "SemiLagrangianTransferRequirements",
    "SemiLagrangianTransportEvidence",
]
