#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._structured_cochain import StructuredCochainBridge
from ..discretization.finite_volume._uct import (
    AbstractUCTElectromotivePlan,
    HLLUCTElectromotivePlan,
)


VectorPotentialGaugeKind: TypeAlias = Literal[
    "none",
    "weyl",
    "generalized_lorenz",
]


class GRMHDMagneticStateLayout(StrictModule, NonTrainableState):
    """Dimension-aware split between material cells and densitized magnetic faces."""

    dimension: int = eqx.field(static=True)
    reduced_component_indices: tuple[int, ...] = eqx.field(static=True)
    face_magnetic_indices: tuple[int, ...] = eqx.field(static=True)
    cell_magnetic_indices: tuple[int, ...] = eqx.field(static=True)
    magnetic_degree: int = eqx.field(static=True)
    electromotive_degree: int | None = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3):
            raise ValueError("GRMHD layout dimension must be one, two, or three.")
        face_owned = tuple(range(5, 5 + dimension_))
        cell_owned = tuple(range(5 + dimension_, 8))
        reduced = (0, 1, 2, 3, 4, *cell_owned)
        self.dimension = dimension_
        self.reduced_component_indices = reduced
        self.face_magnetic_indices = face_owned
        self.cell_magnetic_indices = cell_owned
        self.magnetic_degree = dimension_ - 1
        self.electromotive_degree = dimension_ - 2 if dimension_ >= 2 else None
        self.layout_id = canonical_fingerprint(
            {
                "kind": "grmhd-densitized-magnetic-layout",
                "dimension": dimension_,
                "reduced_components": list(reduced),
                "face_components": list(face_owned),
                "cell_components": list(cell_owned),
            }
        )

    @property
    def reduced_component_count(self) -> int:
        return len(self.reduced_component_indices)

    def reduce_full_state(self, full_state: ArrayLike, /) -> Array:
        full = jnp.asarray(full_state)
        if full.shape[-1:] != (8,):
            raise ValueError("A full Valencia state must have eight components.")
        return full[..., jnp.asarray(self.reduced_component_indices)]

    def expand_reduced_state(
        self,
        reduced_state: ArrayLike,
        face_owned_cell_field: ArrayLike,
        /,
    ) -> Array:
        reduced = jnp.asarray(reduced_state)
        magnetic = jnp.asarray(face_owned_cell_field)
        if reduced.shape[-1:] != (self.reduced_component_count,):
            raise ValueError("Reduced GRMHD component count does not match the layout.")
        if magnetic.shape != reduced.shape[:-1] + (self.dimension,):
            raise ValueError("Cell-centered densitized magnetic field shape is invalid.")
        full = jnp.zeros(reduced.shape[:-1] + (8,), dtype=reduced.dtype)
        full = full.at[..., jnp.asarray(self.reduced_component_indices)].set(reduced)
        return full.at[..., jnp.asarray(self.face_magnetic_indices)].set(magnetic)


class GRMHDVectorPotentialGauge(StrictModule, NonTrainableState):
    """Static discrete vector-potential choice used without changing JIT shapes."""

    kind: VectorPotentialGaugeKind = eqx.field(static=True)
    propagation_speed: float = eqx.field(static=True)
    damping_rate: float = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: VectorPotentialGaugeKind = "none",
        /,
        *,
        propagation_speed: float = 1.0,
        damping_rate: float = 0.0,
    ):
        if kind not in ("none", "weyl", "generalized_lorenz"):
            raise ValueError("Unknown GRMHD vector-potential gauge.")
        speed = float(propagation_speed)
        damping = float(damping_rate)
        if (
            not np.isfinite(speed)
            or speed <= 0.0
            or not np.isfinite(damping)
            or damping < 0.0
        ):
            raise ValueError("Vector-potential gauge controls are invalid.")
        self.kind = kind
        self.propagation_speed = speed
        self.damping_rate = damping
        self.gauge_id = canonical_fingerprint(
            {
                "kind": "grmhd-vector-potential-gauge",
                "gauge": kind,
                "propagation_speed": speed,
                "damping_rate": damping,
            }
        )

    @property
    def evolves_vector_potential(self) -> bool:
        return self.kind != "none"

    @property
    def evolves_scalar(self) -> bool:
        return self.kind == "generalized_lorenz"


class GRMHDCTState(StrictModule):
    """Packed cochains for densitized magnetic flux and its optional potential."""

    magnetic_flux: Array
    vector_potential: Array
    gauge_scalar: Array


class GRMHDCTRate(StrictModule):
    magnetic_rate: Array
    vector_potential_rate: Array
    gauge_scalar_rate: Array
    edge_electromotive_circulation: Array
    faraday_defect: Array
    vector_potential_defect: Array
    gauge_constraint: Array
    uct_consistency_defect: Array
    uct_maximum_dissipation: Array


class GRMHDCTDefectLedger(StrictModule):
    magnetic_flux_change: Array
    integrated_edge_electromotive: Array
    faraday_balance_defect: Array
    divergence_before: Array
    divergence_after: Array
    divergence_change: Array
    vector_potential_defect: Array
    gauge_constraint: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class GRMHDConstrainedTransportPlan(StrictModule, NonTrainableState):
    """Compatible cochain CT for face-integrated ``sqrt(gamma) B^i``.

    Material conservation and Faraday evolution share the same interface magnetic
    fluxes.  In vector-potential modes, the magnetic cochain is exactly ``d A``;
    the generalized Lorenz gauge adds a vertex scalar in three dimensions.
    """

    bridge: StructuredCochainBridge
    layout: GRMHDMagneticStateLayout
    gauge: GRMHDVectorPotentialGauge
    electromotive_plan: AbstractUCTElectromotivePlan
    divergence_tolerance: float = eqx.field(static=True)
    compatibility_tolerance: float = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        gauge: GRMHDVectorPotentialGauge | None = None,
        electromotive_plan: AbstractUCTElectromotivePlan | None = None,
        divergence_tolerance: float = 1.0e-10,
        compatibility_tolerance: float = 1.0e-10,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        gauge_ = GRMHDVectorPotentialGauge() if gauge is None else gauge
        electromotive = (
            HLLUCTElectromotivePlan()
            if electromotive_plan is None
            else electromotive_plan
        )
        if not isinstance(gauge_, GRMHDVectorPotentialGauge):
            raise TypeError("gauge must be GRMHDVectorPotentialGauge.")
        if not isinstance(electromotive, AbstractUCTElectromotivePlan):
            raise TypeError("electromotive_plan must implement the UCT contract.")
        if bridge.dimension == 1 and gauge_.evolves_vector_potential:
            raise ValueError("One-dimensional CT has no vector-potential cochain.")
        if bridge.dimension != 3 and gauge_.evolves_scalar:
            raise ValueError(
                "The discrete generalized Lorenz gauge requires three dimensions."
            )
        divergence = float(divergence_tolerance)
        compatibility = float(compatibility_tolerance)
        if (
            not np.isfinite(divergence)
            or divergence < 0.0
            or not np.isfinite(compatibility)
            or compatibility < 0.0
        ):
            raise ValueError("GRMHD CT tolerances must be finite and non-negative.")
        self.bridge = bridge
        self.layout = GRMHDMagneticStateLayout(bridge.dimension)
        self.gauge = gauge_
        self.electromotive_plan = electromotive
        self.divergence_tolerance = divergence
        self.compatibility_tolerance = compatibility
        self.cell_shape = tuple(int(value) for value in bridge.grid.shape)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "grmhd-compatible-constrained-transport",
                "bridge": bridge.bridge_id,
                "layout": self.layout.layout_id,
                "gauge": gauge_.gauge_id,
                "electromotive": electromotive.electromotive_id,
                "divergence_tolerance": divergence,
                "compatibility_tolerance": compatibility,
            }
        )

    @property
    def vector_potential_size(self) -> int:
        if not self.gauge.evolves_vector_potential:
            return 0
        return self.bridge.cochain.cell_counts[self.layout.magnetic_degree - 1]

    @property
    def gauge_scalar_size(self) -> int:
        return self.bridge.cochain.cell_counts[0] if self.gauge.evolves_scalar else 0

    def validate_magnetic_flux(self, magnetic_flux: ArrayLike, /) -> Array:
        value = jnp.asarray(magnetic_flux)
        expected = self.bridge.cochain.cell_counts[self.layout.magnetic_degree]
        if value.shape != (expected,):
            raise ValueError(f"Densitized magnetic flux must have shape ({expected},).")
        return value

    def validate_vector_potential(self, vector_potential: ArrayLike, /) -> Array:
        value = jnp.asarray(vector_potential)
        expected = self.vector_potential_size
        if value.shape != (expected,):
            raise ValueError(f"Vector potential must have shape ({expected},).")
        return value

    def validate_gauge_scalar(self, gauge_scalar: ArrayLike, /) -> Array:
        value = jnp.asarray(gauge_scalar)
        expected = self.gauge_scalar_size
        if value.shape != (expected,):
            raise ValueError(f"Gauge scalar must have shape ({expected},).")
        return value

    def pack_densitized_face_flux(
        self,
        components: tuple[ArrayLike, ...],
        /,
    ) -> Array:
        """Pack coordinate-face integrals of contravariant ``sqrt(gamma) B``."""
        return self.validate_magnetic_flux(self.bridge.pack_normal_flux(components))

    def pack_physical_face_flux(
        self,
        magnetic_components: tuple[ArrayLike, ...],
        face_volume_densities: tuple[ArrayLike, ...],
        /,
    ) -> Array:
        if (
            len(magnetic_components) != self.layout.dimension
            or len(face_volume_densities) != self.layout.dimension
        ):
            raise ValueError(
                "One magnetic field and volume density are required per axis."
            )
        densitized = tuple(
            jnp.asarray(field) * jnp.asarray(volume)
            for field, volume in zip(
                magnetic_components,
                face_volume_densities,
                strict=True,
            )
        )
        return self.pack_densitized_face_flux(densitized)

    def cell_densitized_magnetic_field(self, magnetic_flux: ArrayLike, /) -> Array:
        faces = self.bridge.unpack_normal_flux(self.validate_magnetic_flux(magnetic_flux))
        centered = tuple(
            0.5 * (face + jnp.roll(face, 1, axis=axis))
            if self.bridge.grid.structured_axes[axis].periodic
            else 0.5
            * (
                jnp.take(face, jnp.arange(face.shape[axis] - 1), axis=axis)
                + jnp.take(face, jnp.arange(1, face.shape[axis]), axis=axis)
            )
            for axis, face in enumerate(faces)
        )
        return jnp.stack(centered, axis=-1)

    def cell_physical_magnetic_field(
        self,
        magnetic_flux: ArrayLike,
        spatial_volume_density: ArrayLike,
        /,
    ) -> Array:
        volume = jnp.asarray(spatial_volume_density)
        if volume.shape != self.cell_shape:
            raise ValueError("Cell spatial volume density shape does not match the grid.")
        return self.cell_densitized_magnetic_field(magnetic_flux) / volume[..., None]

    def full_state(self, reduced_state: ArrayLike, magnetic_flux: ArrayLike, /) -> Array:
        reduced = jnp.asarray(reduced_state)
        expected = self.cell_shape + (self.layout.reduced_component_count,)
        if reduced.shape != expected:
            raise ValueError(f"Reduced GRMHD state must have shape {expected}.")
        return self.layout.expand_reduced_state(
            reduced,
            self.cell_densitized_magnetic_field(magnetic_flux),
        )

    def magnetic_from_vector_potential(self, vector_potential: ArrayLike, /) -> Array:
        if not self.gauge.evolves_vector_potential:
            raise ValueError("The selected CT plan does not evolve a vector potential.")
        potential = self.validate_vector_potential(vector_potential)
        return self.validate_magnetic_flux(
            self.bridge.exterior_derivative(
                self.layout.magnetic_degree - 1,
                potential,
            )
        )

    def magnetic_divergence(self, magnetic_flux: ArrayLike, /) -> Array:
        return self.bridge.exterior_derivative(
            self.layout.magnetic_degree,
            self.validate_magnetic_flux(magnetic_flux),
        )

    def initialize(
        self,
        magnetic_flux: ArrayLike | None = None,
        /,
        *,
        vector_potential: ArrayLike | None = None,
        gauge_scalar: ArrayLike | None = None,
    ) -> GRMHDCTState:
        dtype = self.bridge.cochain.primal_measures[self.layout.magnetic_degree].dtype
        if self.gauge.evolves_vector_potential:
            if vector_potential is None:
                raise ValueError("Vector-potential CT requires initial potential data.")
            potential = self.validate_vector_potential(vector_potential)
            magnetic_from_potential = self.magnetic_from_vector_potential(potential)
            if magnetic_flux is None:
                magnetic = magnetic_from_potential
            else:
                magnetic = self.validate_magnetic_flux(magnetic_flux)
                mismatch = jnp.max(
                    jnp.abs(magnetic - magnetic_from_potential), initial=0.0
                )
                magnetic_scale = jnp.maximum(
                    jnp.max(jnp.abs(magnetic_from_potential), initial=0.0),
                    1.0,
                )
                compatibility_tolerance = jnp.maximum(
                    self.compatibility_tolerance,
                    256.0 * jnp.finfo(magnetic.dtype).eps * magnetic_scale,
                )
                magnetic = eqx.error_if(
                    magnetic,
                    mismatch > compatibility_tolerance,
                    "Initial magnetic flux is not the discrete curl of the vector potential.",
                )
        else:
            if magnetic_flux is None:
                raise ValueError(
                    "Face-flux CT requires initial densitized magnetic flux."
                )
            if vector_potential is not None:
                raise ValueError("Vector potential is disabled by the selected gauge.")
            magnetic = self.validate_magnetic_flux(magnetic_flux)
            potential = jnp.zeros((0,), dtype=dtype)
        if self.gauge.evolves_scalar:
            scalar = self.validate_gauge_scalar(
                jnp.zeros((self.gauge_scalar_size,), dtype=potential.dtype)
                if gauge_scalar is None
                else gauge_scalar
            )
        else:
            if gauge_scalar is not None:
                raise ValueError("Gauge scalar is disabled by the selected gauge.")
            scalar = jnp.zeros((0,), dtype=potential.dtype)
        divergence = self.magnetic_divergence(magnetic)
        magnetic_scale = jnp.maximum(
            jnp.max(jnp.abs(magnetic), initial=0.0),
            1.0,
        )
        divergence_tolerance = jnp.maximum(
            self.divergence_tolerance,
            256.0 * jnp.finfo(magnetic.dtype).eps * magnetic_scale,
        )
        magnetic = eqx.error_if(
            magnetic,
            jnp.max(jnp.abs(divergence), initial=0.0) > divergence_tolerance,
            "Initial densitized magnetic flux violates the discrete divergence constraint.",
        )
        return GRMHDCTState(magnetic, potential, scalar)

    def _average_to_edges(self, value: Array, axis: int, /) -> Array:
        if self.bridge.grid.structured_axes[axis].periodic:
            return 0.5 * (value + jnp.roll(value, -1, axis=axis))
        lower = jnp.take(value, jnp.asarray([0]), axis=axis)
        upper = jnp.take(value, jnp.asarray([value.shape[axis] - 1]), axis=axis)
        interior = 0.5 * (
            jnp.take(value, jnp.arange(value.shape[axis] - 1), axis=axis)
            + jnp.take(value, jnp.arange(1, value.shape[axis]), axis=axis)
        )
        return jnp.concatenate((lower, interior, upper), axis=axis)

    def _bounded_electromotive_components(
        self,
        face_fluxes: tuple[Array, ...],
        /,
    ) -> tuple[Array, ...]:
        if self.layout.dimension == 1:
            return ()
        if self.layout.dimension == 2:
            flux_x, flux_y = face_fluxes
            return (
                0.5
                * (
                    -self._average_to_edges(flux_x[..., 6], 1)
                    + self._average_to_edges(flux_y[..., 5], 0)
                ),
            )
        flux_x, flux_y, flux_z = face_fluxes
        ex = 0.5 * (
            -self._average_to_edges(flux_y[..., 7], 2)
            + self._average_to_edges(flux_z[..., 6], 1)
        )
        ey = 0.5 * (
            self._average_to_edges(flux_x[..., 7], 2)
            - self._average_to_edges(flux_z[..., 5], 0)
        )
        ez = 0.5 * (
            -self._average_to_edges(flux_x[..., 6], 1)
            + self._average_to_edges(flux_y[..., 5], 0)
        )
        return ex, ey, ez

    def edge_electromotive(
        self,
        full_state: Array,
        face_fluxes: tuple[Array, ...],
        signal_speeds: tuple[Array, ...],
        /,
    ) -> tuple[Array, Array, Array]:
        if (
            len(face_fluxes) != self.layout.dimension
            or len(signal_speeds) != self.layout.dimension
        ):
            raise ValueError(
                "One GRMHD face flux and signal speed are required per axis."
            )
        if self.layout.dimension == 1:
            dtype = jnp.asarray(full_state).dtype
            return (
                jnp.zeros((0,), dtype=dtype),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(0.0, dtype=dtype),
            )
        bounded = any(not axis.periodic for axis in self.bridge.grid.structured_axes)
        if bounded:
            components = self._bounded_electromotive_components(face_fluxes)
            defect = jnp.asarray(0.0, dtype=jnp.asarray(full_state).dtype)
            dissipation = jnp.asarray(0.0, dtype=jnp.asarray(full_state).dtype)
        else:
            result = self.electromotive_plan.electromotive(
                jnp.asarray(full_state),
                face_fluxes,
                signal_speeds,
                self.layout.dimension,
            )
            components = result.components
            defect = result.one_dimensional_consistency_defect
            dissipation = result.maximum_dissipation
        return self.bridge.pack_electromotive(components), defect, dissipation

    def rate(
        self,
        state: GRMHDCTState,
        edge_electromotive_circulation: ArrayLike,
        /,
        *,
        uct_consistency_defect: ArrayLike = 0.0,
        uct_maximum_dissipation: ArrayLike = 0.0,
    ) -> GRMHDCTRate:
        if not isinstance(state, GRMHDCTState):
            raise TypeError("state must be GRMHDCTState.")
        magnetic = self.validate_magnetic_flux(state.magnetic_flux)
        if self.layout.dimension == 1:
            electromotive = jnp.asarray(edge_electromotive_circulation)
            if electromotive.shape != (0,):
                raise ValueError("One-dimensional GRMHD EMF storage must be empty.")
            magnetic_rate = jnp.zeros_like(magnetic)
        else:
            degree = self.layout.electromotive_degree
            if degree is None:
                raise RuntimeError("Multidimensional GRMHD requires an EMF degree.")
            electromotive = jnp.asarray(edge_electromotive_circulation)
            expected = self.bridge.cochain.cell_counts[degree]
            if electromotive.shape != (expected,):
                raise ValueError(f"Edge electromotive must have shape ({expected},).")
            magnetic_rate = -self.bridge.exterior_derivative(
                degree,
                electromotive,
            )
        if self.gauge.evolves_vector_potential:
            potential = self.validate_vector_potential(state.vector_potential)
            if self.gauge.evolves_scalar:
                scalar = self.validate_gauge_scalar(state.gauge_scalar)
                gradient = self.bridge.exterior_derivative(0, scalar)
                potential_rate = -electromotive - gradient
                gauge_constraint = self.bridge.cochain.codifferential(1, potential)
                scalar_rate = (
                    -(self.gauge.propagation_speed**2) * gauge_constraint
                    - self.gauge.damping_rate * scalar
                )
            else:
                potential_rate = -electromotive
                scalar_rate = jnp.zeros_like(state.gauge_scalar)
                gauge_constraint = jnp.zeros((0,), dtype=potential.dtype)
            curl_rate = self.bridge.exterior_derivative(
                self.layout.magnetic_degree - 1,
                potential_rate,
            )
            vector_defect = self.magnetic_from_vector_potential(potential) - magnetic
        else:
            potential_rate = jnp.zeros_like(state.vector_potential)
            scalar_rate = jnp.zeros_like(state.gauge_scalar)
            gauge_constraint = jnp.zeros((0,), dtype=magnetic.dtype)
            curl_rate = magnetic_rate
            vector_defect = jnp.zeros_like(magnetic)
        return GRMHDCTRate(
            magnetic_rate=magnetic_rate,
            vector_potential_rate=potential_rate,
            gauge_scalar_rate=scalar_rate,
            edge_electromotive_circulation=electromotive,
            faraday_defect=curl_rate - magnetic_rate,
            vector_potential_defect=vector_defect,
            gauge_constraint=gauge_constraint,
            uct_consistency_defect=jnp.asarray(uct_consistency_defect),
            uct_maximum_dissipation=jnp.asarray(uct_maximum_dissipation),
        )

    def defects(
        self,
        before: GRMHDCTState,
        after: GRMHDCTState,
        integrated_edge_electromotive: ArrayLike,
        /,
    ) -> GRMHDCTDefectLedger:
        edge = jnp.asarray(integrated_edge_electromotive)
        magnetic_change = after.magnetic_flux - before.magnetic_flux
        if self.layout.dimension == 1:
            expected_change = jnp.zeros_like(magnetic_change)
        else:
            degree = self.layout.electromotive_degree
            if degree is None:
                raise RuntimeError("Multidimensional GRMHD requires an EMF degree.")
            expected_change = -self.bridge.exterior_derivative(
                degree,
                edge,
            )
        faraday = magnetic_change - expected_change
        divergence_before = self.magnetic_divergence(before.magnetic_flux)
        divergence_after = self.magnetic_divergence(after.magnetic_flux)
        if self.gauge.evolves_vector_potential:
            vector_defect = after.magnetic_flux - self.magnetic_from_vector_potential(
                after.vector_potential
            )
        else:
            vector_defect = jnp.zeros_like(after.magnetic_flux)
        if self.gauge.evolves_scalar:
            gauge_constraint = self.bridge.cochain.codifferential(
                1,
                after.vector_potential,
            )
        else:
            gauge_constraint = jnp.zeros((0,), dtype=after.magnetic_flux.dtype)
        arrays = (
            magnetic_change,
            edge,
            faraday,
            divergence_before,
            divergence_after,
            vector_defect,
            gauge_constraint,
        )
        finite = jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(x)) for x in arrays)))
        max_divergence = jnp.max(jnp.abs(divergence_after), initial=0.0)
        max_faraday = jnp.max(jnp.abs(faraday), initial=0.0)
        max_vector = jnp.max(jnp.abs(vector_defect), initial=0.0)
        magnetic_scale = jnp.maximum(
            jnp.max(jnp.abs(after.magnetic_flux), initial=0.0),
            1.0,
        )
        roundoff = 256.0 * jnp.finfo(after.magnetic_flux.dtype).eps * magnetic_scale
        divergence_tolerance = jnp.maximum(self.divergence_tolerance, roundoff)
        compatibility_tolerance = jnp.maximum(self.compatibility_tolerance, roundoff)
        physically_valid = finite & (max_divergence <= divergence_tolerance)
        qualified = (
            physically_valid
            & (max_faraday <= compatibility_tolerance)
            & (max_vector <= compatibility_tolerance)
        )
        return GRMHDCTDefectLedger(
            magnetic_flux_change=magnetic_change,
            integrated_edge_electromotive=edge,
            faraday_balance_defect=faraday,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            divergence_change=divergence_after - divergence_before,
            vector_potential_defect=vector_defect,
            gauge_constraint=gauge_constraint,
            finite=finite,
            physically_valid=physically_valid,
            qualified=qualified,
            plan_id=self.plan_id,
        )


__all__ = [
    "GRMHDCTDefectLedger",
    "GRMHDCTRate",
    "GRMHDCTState",
    "GRMHDConstrainedTransportPlan",
    "GRMHDMagneticStateLayout",
    "GRMHDVectorPotentialGauge",
    "VectorPotentialGaugeKind",
]
