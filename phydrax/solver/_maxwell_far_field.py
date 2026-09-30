#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Huygens-surface acquisition and homogeneous-exterior far fields.

Phasors follow ``exp(-iωt)``: a Huygens sampler accumulates the transient
spectrum ``F̃(ω) = ∫ f(t) e^{+iωt} dt`` of the tangential fields on a closed
surface with outward normals ``n̂``. The far field of the equivalent currents
``J = n̂ × H̃`` and ``M = -n̂ × Ẽ`` in a homogeneous exterior ``(ε, μ)`` with
``k = ω√(εμ)`` and ``η = √(μ/ε)`` is ``Ẽ(r r̂) ≈ e^{ikr} F(r̂)/r`` with

    F_θ = (ik/4π)(L_φ + η N_θ),    F_φ = -(ik/4π)(L_θ − η N_φ),

where ``N = ∫ J e^{-ik r̂·r'} dS'`` and ``L = ∫ M e^{-ik r̂·r'} dS'`` are the
radiation vectors. The one-sided spectral energy is
``d²W/(dω dΩ) = εc |F|²/π`` and the surface Poynting spectrum is
``dW/dω = (1/π) Re ∫ (Ẽ × H̃*)·n̂ dS``; the Hertzian-dipole test fixes every sign.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, final, Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .. import ein
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge, tetrahedral_connectivity
from ..discretization._simplicial_locator import (
    PreparedSimplicialCellLocator,
    SimplicialLocationPolicy,
)
from ..discretization.fem import FiniteElementDeRhamComplex
from ..discretization.fem._cell_map import PreparedFiniteElementCellMap
from ..discretization.fem._simplicial_whitney_chains import (
    SimplicialWhitneyKernel,
    whitney_basis,
)
from ..exterior import form_to_vector, FormType, FormValueSpec
from ..sparse import EdgeRelation, SparseLinearMap
from ..typing import Complex128, Dim, Float64, Int32
from ._maxwell import (
    MaxwellCochainLayout,
    PreparedCompatibleMaxwell,
    PreparedDiagonalMaxwellConstitutive,
)
from ._maxwell_materials import PreparedConductiveMaxwellConstitutive
from ._maxwell_observers import (
    AbstractMaxwellObserverPlan,
    AbstractPreparedMaxwellObserver,
    DFTObserverState,
    MaxwellSpectralAcquisition,
)
from ._maxwell_sources import PreparedMaxwellSource
from ._maxwell_unstructured import (
    PreparedFiniteElementMaxwellConstitutive,
    PreparedUnstructuredMaxwell,
)
from ._pic_current_source import PreparedPICMaxwellCurrentSource


class _FrequencyDim(Dim, minimum=1):
    """Acquired angular frequencies."""


class _SurfaceDim(Dim, minimum=1):
    """Huygens surface quadrature cells."""


class _DirectionDim(Dim, minimum=1):
    """Far-field observation directions."""


class HomogeneousMaxwellExterior(StrictModule, NonTrainableState):
    """Declared lossless homogeneous medium outside a Huygens surface."""

    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    exterior_id: str = eqx.field(static=True)

    def __init__(self, *, permittivity: float = 1.0, permeability: float = 1.0) -> None:
        epsilon, mu = float(permittivity), float(permeability)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("Exterior permittivity must be finite and positive.")
        if not math.isfinite(mu) or mu <= 0.0:
            raise ValueError("Exterior permeability must be finite and positive.")
        self.permittivity = epsilon
        self.permeability = mu
        self.exterior_id = canonical_fingerprint(
            {
                "kind": "homogeneous-maxwell-exterior",
                "permittivity": epsilon,
                "permeability": mu,
            }
        )

    @property
    def wave_speed(self) -> float:
        return 1.0 / math.sqrt(self.permittivity * self.permeability)

    @property
    def impedance(self) -> float:
        return math.sqrt(self.permeability / self.permittivity)

    def wavenumber(self, angular_frequency: Array, /) -> Array:
        return angular_frequency * math.sqrt(self.permittivity * self.permeability)


class HuygensSurfacePhasors(StrictModule):
    """Tangential ``exp(-iωt)`` field phasors on one closed oriented surface.

    ``electric``/``magnetic`` are the transient spectra ``∫ f(t) e^{+iωt} dt`` of
    the tangential fields at the cell centers ``positions`` with outward unit
    ``normals`` and quadrature ``measures``; ``patches`` labels the surface
    patch each cell belongs to (box faces ``2·axis + side``).
    """

    __strict_contract__ = True

    angular_frequencies: Float64[_FrequencyDim]
    positions: Float64[_SurfaceDim, Literal[3]]
    normals: Float64[_SurfaceDim, Literal[3]]
    measures: Float64[_SurfaceDim]
    patches: Int32[_SurfaceDim]
    electric: Complex128[_FrequencyDim, _SurfaceDim, Literal[3]]
    magnetic: Complex128[_FrequencyDim, _SurfaceDim, Literal[3]]
    exterior: HomogeneousMaxwellExterior


class MaxwellHuygensSampler(Protocol):
    """Streaming Huygens capability: synchronized fields in, surface phasors out.

    The compatible cochain solver implements it through
    `MaxwellHuygensBoxPlan`; tetrahedral runtimes through
    `MaxwellHuygensSurfacePlan`. Spectral field solvers and frequency-domain
    solutions adapt to the same contract.
    """

    @property
    def acquisition(self) -> MaxwellSpectralAcquisition: ...

    @property
    def exterior(self) -> HomogeneousMaxwellExterior: ...

    def initialize(self, /) -> DFTObserverState: ...

    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
    ) -> DFTObserverState: ...

    def surface_phasors(self, state: Any, /) -> HuygensSurfacePhasors: ...


@final
class _SurfaceGeometry(StrictModule, NonTrainableState):
    """Host-prepared quadrature and sparse tangential-field gathers."""

    positions: Array
    normals: Array
    measures: Array
    patches: Array
    electric_gather: SparseLinearMap
    magnetic_gather: SparseLinearMap
    electric_indices: Array
    magnetic_indices: Array
    geometry_id: str = eqx.field(static=True)
    material_cells: Array | None = None


def _require_huygens_acquisition(
    acquisition: MaxwellSpectralAcquisition, /
) -> MaxwellSpectralAcquisition:
    if not isinstance(acquisition, MaxwellSpectralAcquisition):
        raise TypeError("acquisition must be a MaxwellSpectralAcquisition.")
    if acquisition.measure != "time-integral" or acquisition.sign != "positive":
        raise ValueError(
            "Huygens acquisition requires the time-integral measure with the "
            "positive Fourier exponent (exp(-iωt) phasors)."
        )
    return acquisition


def _require_exterior(
    exterior: HomogeneousMaxwellExterior, /
) -> HomogeneousMaxwellExterior:
    if not isinstance(exterior, HomogeneousMaxwellExterior):
        raise TypeError("exterior must be a HomogeneousMaxwellExterior.")
    return exterior


def _gather_map(
    source: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
    /,
    *,
    source_size: int,
    target_count: int,
    operator_id: str,
) -> SparseLinearMap:
    return SparseLinearMap(
        EdgeRelation(
            source.astype(np.int32),
            target.astype(np.int32),
            source_size=source_size,
            target_size=3 * target_count,
        ),
        coefficients.astype(np.float64),
        operator_id=operator_id,
    )


def _surface_material(
    prepared_constitutive: Any,
    electric_indices: np.ndarray,
    magnetic_indices: np.ndarray,
    permittivity: float,
    permeability: float,
    /,
    *,
    owner: str = "Huygens surface",
    cell_indices: np.ndarray | None = None,
) -> None:
    """Refuse constitutive laws that are not the declared medium on ``owner``'s
    entities (a Huygens surface or an antenna sheet)."""
    match prepared_constitutive:
        case PreparedDiagonalMaxwellConstitutive() as material:
            surface_permittivity = np.asarray(material.permittivity)[electric_indices]
            surface_permeability = np.asarray(material.permeability)[magnetic_indices]
        case PreparedConductiveMaxwellConstitutive() as material:
            surface_permittivity = np.asarray(material.permittivity)[electric_indices]
            surface_permeability = np.asarray(material.permeability)[magnetic_indices]
            electric_loss = np.asarray(material.electric_conductivity)[electric_indices]
            magnetic_loss = np.asarray(material.magnetic_conductivity)[magnetic_indices]
            if np.any(electric_loss != 0.0) or np.any(magnetic_loss != 0.0):
                raise ValueError(
                    f"{owner} entities carry nonzero conductivity; the "
                    "exterior must be lossless."
                )
        case PreparedFiniteElementMaxwellConstitutive() as material:
            if cell_indices is None:
                raise ValueError(
                    "FE Huygens material validation requires adjacent cell identities."
                )
            epsilon = np.asarray(material.permittivity)
            inverse_mu = np.asarray(material.inverse_permeability)
            surface_epsilon = epsilon if epsilon.ndim == 0 else epsilon[cell_indices]
            surface_inverse_mu = (
                inverse_mu if inverse_mu.ndim == 0 else inverse_mu[cell_indices]
            )
            epsilon_target = (
                permittivity * np.eye(3, dtype=np.float64)
                if surface_epsilon.ndim == 3
                else permittivity
            )
            inverse_mu_target = (
                np.eye(3, dtype=np.float64) / permeability
                if surface_inverse_mu.ndim == 3
                else 1.0 / permeability
            )
            if not np.allclose(
                surface_epsilon, epsilon_target, rtol=1e-12, atol=0.0
            ) or not np.allclose(
                surface_inverse_mu, inverse_mu_target, rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    f"{owner} cells do not carry the declared homogeneous exterior permittivity/permeability."
                )
            return
        case _:
            raise ValueError(
                f"{owner}s require a lossless diagonal, conductive, or FE Gram "
                f"constitutive law; got {type(prepared_constitutive).__name__}."
            )
    if not np.allclose(
        surface_permittivity, permittivity, rtol=1e-12, atol=0.0
    ) or not np.allclose(surface_permeability, permeability, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"{owner} entities do not carry the declared homogeneous "
            "exterior permittivity/permeability."
        )


def _surface_sources(
    sources: Sequence[Any],
    electric_indices: np.ndarray,
    magnetic_indices: np.ndarray,
    /,
) -> None:
    """Refuse sources whose static support reaches the surface, or dynamic ones."""
    # The antenna and prescribed-charge modules build on this one; resolve their
    # prepared types lazily.
    from ._maxwell_antenna import PreparedSampledPlaneCurrentAntenna
    from ._prescribed_charge_maxwell import PreparedPrescribedChargeCurrentSource

    electric_set = np.asarray(electric_indices)
    magnetic_set = np.asarray(magnetic_indices)
    for source in sources:
        match source:
            case PreparedSampledPlaneCurrentAntenna() as antenna:
                if np.any(np.isin(antenna.electric_support, electric_set)) or np.any(
                    np.isin(antenna.magnetic_support, magnetic_set)
                ):
                    raise ValueError(
                        "An antenna sheet can drive entities on the Huygens surface; "
                        "equivalent currents require J = M = 0 there."
                    )
            case PreparedMaxwellSource() as static:
                if np.any(
                    np.isin(np.asarray(static.electric_indices), electric_set)
                ) or np.any(np.isin(np.asarray(static.magnetic_indices), magnetic_set)):
                    raise ValueError(
                        "A Maxwell source drives entities on the Huygens surface; "
                        "equivalent currents require J = M = 0 there."
                    )
            case PreparedPrescribedChargeCurrentSource() as prescribed:
                if np.any(np.asarray(prescribed.electric_support)[electric_set]):
                    raise ValueError(
                        "A prescribed charge path drives entities on the Huygens "
                        "surface; equivalent currents require J = 0 there."
                    )
            case PreparedPICMaxwellCurrentSource():
                raise ValueError(
                    "Dynamic PIC currents cannot certify J = 0 on a Huygens surface."
                )
            case _:
                raise ValueError(
                    "Huygens surfaces accept only prepared static Maxwell sources; "
                    f"got {type(source).__name__}."
                )


def _box_geometry(
    bridge: StructuredCochainBridge,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    layout: MaxwellCochainLayout,
    /,
) -> _SurfaceGeometry:
    axes = bridge.grid.structured_axes
    points = tuple(np.asarray(axis.point_coordinates, dtype=np.float64) for axis in axes)
    centers = tuple(np.asarray(axis.interval_centers, dtype=np.float64) for axis in axes)
    widths = tuple(np.asarray(axis.interval_widths, dtype=np.float64) for axis in axes)
    edge_shapes = bridge.orientation_shapes[1]
    edge_offsets = bridge.orientation_offsets[1]
    face_orientations = bridge.orientations[2]
    face_shapes = bridge.orientation_shapes[2]
    face_offsets = bridge.orientation_offsets[2]

    def edge_index(axis: int, index: tuple[int, int, int]) -> int:
        return edge_offsets[axis] + int(np.ravel_multi_index(index, edge_shapes[axis]))

    def face_index(orientation: tuple[int, int], index: tuple[int, int, int]) -> int:
        position = face_orientations.index(orientation)
        return face_offsets[position] + int(
            np.ravel_multi_index(index, face_shapes[position])
        )

    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    measures: list[float] = []
    patches: list[int] = []
    electric_routes: list[tuple[int, int, float]] = []
    magnetic_routes: list[tuple[int, int, float]] = []
    cell = 0
    for normal_axis in range(3):
        tangent_b, tangent_c = tuple(axis for axis in range(3) if axis != normal_axis)
        for side, node in enumerate((lower[normal_axis], upper[normal_axis])):
            sign = -1.0 if side == 0 else 1.0
            for jb in range(lower[tangent_b], upper[tangent_b]):
                for jc in range(lower[tangent_c], upper[tangent_c]):
                    position = np.zeros(3)
                    position[normal_axis] = points[normal_axis][node]
                    position[tangent_b] = centers[tangent_b][jb]
                    position[tangent_c] = centers[tangent_c][jc]
                    normal = np.zeros(3)
                    normal[normal_axis] = sign
                    positions.append(position)
                    normals.append(normal)
                    measures.append(widths[tangent_b][jb] * widths[tangent_c][jc])
                    patches.append(2 * normal_axis + side)
                    base = {normal_axis: node, tangent_b: jb, tangent_c: jc}
                    for component, other in (
                        (tangent_b, tangent_c),
                        (tangent_c, tangent_b),
                    ):
                        # Tangential E: mean of the two parallel edges bounding the
                        # surface cell, each circulation divided by its length.
                        length = widths[component][base[component]]
                        for shift in (0, 1):
                            index = dict(base)
                            index[other] = base[other] + shift
                            electric_routes.append(
                                (
                                    edge_index(
                                        component,
                                        (index[0], index[1], index[2]),
                                    ),
                                    3 * cell + component,
                                    0.5 / length,
                                )
                            )
                        # Tangential H: mean over the four faces normal to the
                        # component that straddle the surface plane (normal
                        # intervals node - 1 and node) and bound the surface cell
                        # along the component (nodes j and j + 1), each flux divided
                        # by its area and signed by the Levi-Civita orientation of
                        # the packed face.
                        low, high = sorted((normal_axis, other))
                        # Packed (p, q) faces carry +B_r for cyclic (p, q, r) and
                        # -B_r for the anti-cyclic (0, 2, 1) orientation.
                        levi_civita = -1.0 if (low, high) == (0, 2) else 1.0
                        for normal_shift in (-1, 0):
                            for component_shift in (0, 1):
                                index = dict(base)
                                index[normal_axis] = node + normal_shift
                                index[component] = base[component] + component_shift
                                area = (
                                    widths[normal_axis][index[normal_axis]]
                                    * widths[other][base[other]]
                                )
                                magnetic_routes.append(
                                    (
                                        face_index(
                                            (low, high),
                                            (index[0], index[1], index[2]),
                                        ),
                                        3 * cell + component,
                                        levi_civita * 0.25 / area,
                                    )
                                )
                    cell += 1
    electric_source = np.asarray([route[0] for route in electric_routes], dtype=np.int32)
    electric_target = np.asarray([route[1] for route in electric_routes], dtype=np.int32)
    electric_weight = np.asarray([route[2] for route in electric_routes])
    magnetic_source = np.asarray([route[0] for route in magnetic_routes], dtype=np.int32)
    magnetic_target = np.asarray([route[1] for route in magnetic_routes], dtype=np.int32)
    magnetic_weight = np.asarray([route[2] for route in magnetic_routes])
    positions_ = np.stack(positions)
    normals_ = np.stack(normals)
    measures_ = np.asarray(measures)
    patches_ = np.asarray(patches, dtype=np.int32)
    geometry_id = canonical_fingerprint(
        {
            "kind": "maxwell-huygens-box-geometry",
            "bridge": bridge.bridge_id,
            "lower": list(lower),
            "upper": list(upper),
        }
    )
    return _SurfaceGeometry(
        positions=jnp.asarray(positions_),
        normals=jnp.asarray(normals_),
        measures=jnp.asarray(measures_),
        patches=jnp.asarray(patches_),
        electric_gather=_gather_map(
            electric_source,
            electric_target,
            electric_weight,
            source_size=layout.electric_count,
            target_count=cell,
            operator_id=f"{geometry_id}:electric",
        ),
        magnetic_gather=_gather_map(
            magnetic_source,
            magnetic_target,
            magnetic_weight,
            source_size=layout.magnetic_count,
            target_count=cell,
            operator_id=f"{geometry_id}:magnetic",
        ),
        electric_indices=jnp.asarray(np.unique(electric_source)),
        magnetic_indices=jnp.asarray(np.unique(magnetic_source)),
        geometry_id=geometry_id,
    )


def _phasors(
    geometry: _SurfaceGeometry,
    acquisition: MaxwellSpectralAcquisition,
    exterior: HomogeneousMaxwellExterior,
    state: Any,
    /,
) -> HuygensSurfacePhasors:
    if not isinstance(state, DFTObserverState):
        raise TypeError("Huygens sampling requires DFTObserverState.")
    value = acquisition.value(state)
    count = geometry.measures.shape[0]
    fields = value.reshape((value.shape[0], 2, count, 3))
    return HuygensSurfacePhasors(
        angular_frequencies=acquisition.angular_frequencies,
        positions=geometry.positions,
        normals=geometry.normals,
        measures=geometry.measures,
        patches=geometry.patches,
        electric=fields[:, 0],
        magnetic=fields[:, 1],
        exterior=exterior,
    )


def _payload(geometry: _SurfaceGeometry, electric: Array, magnetic: Array, /) -> Array:
    gathered_electric = geometry.electric_gather.mv(electric)
    gathered_magnetic = geometry.magnetic_gather.mv(magnetic)
    return jnp.concatenate((gathered_electric, gathered_magnetic))


class MaxwellHuygensBoxPlan(AbstractMaxwellObserverPlan):
    """Closed axis-aligned Huygens box on a full 3-D structured cochain bridge.

    The box faces lie on the node planes ``lower_nodes``/``upper_nodes`` and are
    sampled at surface-cell centers: tangential ``E`` is the mean of the two
    bounding edge circulations per unit length, tangential ``H`` the mean of the
    four straddling face fluxes per unit area. Preparation refuses CPML overlap,
    sources on the surface, and any material on the surface entities other than
    the declared lossless homogeneous exterior.
    """

    bridge: StructuredCochainBridge
    lower_nodes: tuple[int, int, int] = eqx.field(static=True)
    upper_nodes: tuple[int, int, int] = eqx.field(static=True)
    acquisition: MaxwellSpectralAcquisition
    exterior: HomogeneousMaxwellExterior
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        lower_nodes: Sequence[int],
        upper_nodes: Sequence[int],
        acquisition: MaxwellSpectralAcquisition,
        exterior: HomogeneousMaxwellExterior,
        /,
    ) -> None:
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be a StructuredCochainBridge.")
        if bridge.dimension != 3:
            raise ValueError("Huygens boxes require a three-dimensional bridge.")
        lower = tuple(int(value) for value in lower_nodes)
        upper = tuple(int(value) for value in upper_nodes)
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("lower_nodes and upper_nodes must have three entries.")
        for axis, structured_axis in enumerate(bridge.grid.structured_axes):
            count = structured_axis.point_coordinates.shape[0]
            if not 1 <= lower[axis] < upper[axis] <= count - 2:
                raise ValueError(
                    "Huygens box node planes must satisfy "
                    "1 <= lower < upper <= points - 2 on every axis so that both "
                    "staggered magnetic faces exist."
                )
        self.bridge = bridge
        self.lower_nodes = (lower[0], lower[1], lower[2])
        self.upper_nodes = (upper[0], upper[1], upper[2])
        self.acquisition = _require_huygens_acquisition(acquisition)
        self.exterior = _require_exterior(exterior)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-huygens-box-plan",
                "bridge": bridge.bridge_id,
                "lower": list(self.lower_nodes),
                "upper": list(self.upper_nodes),
                "acquisition": self.acquisition.acquisition_id,
                "exterior": self.exterior.exterior_id,
            }
        )

    def prepare(self, layout: Any, /) -> PreparedMaxwellHuygensBox:
        if not isinstance(layout, MaxwellCochainLayout):
            raise TypeError("Huygens box preparation requires a MaxwellCochainLayout.")
        if layout.polarization != "full_3d":
            raise ValueError("Huygens boxes require the full_3d Maxwell polarization.")
        if (
            layout.electric_count != self.bridge.cochain.cell_counts[1]
            or layout.magnetic_count != self.bridge.cochain.cell_counts[2]
        ):
            raise ValueError("Huygens box bridge does not match the runtime layout.")
        return PreparedMaxwellHuygensBox(
            self, _box_geometry(self.bridge, self.lower_nodes, self.upper_nodes, layout)
        )


class PreparedMaxwellHuygensBox(AbstractPreparedMaxwellObserver):
    """Prepared streaming Huygens box; implements `MaxwellHuygensSampler`."""

    geometry: _SurfaceGeometry
    acquisition: MaxwellSpectralAcquisition
    exterior: HomogeneousMaxwellExterior
    bridge_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: MaxwellHuygensBoxPlan, geometry: _SurfaceGeometry, /
    ) -> None:
        self.geometry = geometry
        self.acquisition = plan.acquisition
        self.exterior = plan.exterior
        self.bridge_id = plan.bridge.bridge_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-huygens-box",
                "plan": plan.plan_id,
                "geometry": geometry.geometry_id,
            }
        )

    @property
    def surface_count(self) -> int:
        return self.geometry.measures.shape[0]

    def validate_runtime(self, prepared: PreparedCompatibleMaxwell, /) -> None:
        bridge = prepared.plan.bridge
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("Huygens boxes require a structured cochain bridge.")
        if bridge.bridge_id != self.bridge_id:
            raise ValueError("Huygens box was built on a different cochain bridge.")
        electric_indices = np.asarray(self.geometry.electric_indices)
        magnetic_indices = np.asarray(self.geometry.magnetic_indices)
        if prepared.pml is not None:
            absorbed_electric = np.concatenate(
                tuple(np.asarray(term.indices) for term in prepared.pml.electric_terms)
                or (np.zeros((0,), dtype=np.int32),)
            )
            absorbed_magnetic = np.concatenate(
                tuple(np.asarray(term.indices) for term in prepared.pml.magnetic_terms)
                or (np.zeros((0,), dtype=np.int32),)
            )
            if np.any(np.isin(electric_indices, absorbed_electric)) or np.any(
                np.isin(magnetic_indices, absorbed_magnetic)
            ):
                raise ValueError("Huygens box intersects the CPML region.")
        _surface_material(
            prepared.constitutive,
            electric_indices,
            magnetic_indices,
            self.exterior.permittivity,
            self.exterior.permeability,
        )
        _surface_sources(prepared.sources, electric_indices, magnetic_indices)
        for boundary in prepared.boundaries:
            constrained = (
                np.asarray(boundary.magnetic_boundary)[magnetic_indices]
                if boundary.kind == "pmc"
                else np.asarray(boundary.electric_boundary)[electric_indices]
            )
            if np.any(constrained):
                raise ValueError(
                    f"A {boundary.kind} boundary constrains entities on the Huygens "
                    "surface; the exterior must be the declared homogeneous medium."
                )

    def initialize(self, /) -> DFTObserverState:
        return self.acquisition.initialize((6 * self.surface_count,))

    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
    ) -> DFTObserverState:
        if not isinstance(state, DFTObserverState):
            raise TypeError("Huygens box requires DFTObserverState.")
        payload = _payload(self.geometry, electric, magnetic)
        return self.acquisition.accumulate(state, jnp.asarray(time), payload)

    def value(self, state: Any, /) -> Array:
        """Return the phasors as ``[frequency, (E, H), surface cell, component]``."""
        if not isinstance(state, DFTObserverState):
            raise TypeError("Huygens box requires DFTObserverState.")
        value = self.acquisition.value(state)
        return value.reshape((value.shape[0], 2, self.surface_count, 3))

    def surface_phasors(self, state: Any, /) -> HuygensSurfacePhasors:
        return _phasors(self.geometry, self.acquisition, self.exterior, state)


def _surface_whitney_tables(
    complex_: FiniteElementDeRhamComplex, /
) -> tuple[Array, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Bind the canonical affine gradients and oriented global Whitney routes."""
    kernels = tuple(
        SimplicialWhitneyKernel(
            complex_,
            PreparedSimplicialCellLocator(
                PreparedFiniteElementCellMap(complex_.discretization, block_index),
                complex_.mesh.coordinates,
                SimplicialLocationPolicy(1, 1, 1),
            ),
        )
        for block_index in range(len(complex_.mesh.blocks))
    )
    gradients = jnp.concatenate([kernel.locator.affine_gradients() for kernel in kernels])
    entities = tuple(
        np.concatenate(
            [np.asarray(kernel.cell_routes[degree].indices) for kernel in kernels]
        )
        for degree in (1, 2)
    )
    signs = tuple(
        np.concatenate(
            [np.asarray(kernel.cell_routes[degree].signs) for kernel in kernels]
        )
        for degree in (1, 2)
    )
    return gradients, entities, signs


def _surface_cell_routes(
    barycentric: Array,
    gradients: Array,
    entities: tuple[np.ndarray, ...],
    signs: tuple[np.ndarray, ...],
    projector: np.ndarray,
    flux_spec: FormValueSpec,
    surface_index: int,
    /,
) -> tuple[list[tuple[int, int, float]], ...]:
    edge_values = whitney_basis(barycentric, gradients, 1)
    face_values = form_to_vector(whitney_basis(barycentric, gradients, 2), flux_spec)
    routes: tuple[list[tuple[int, int, float]], ...] = ([], [])
    for slot, values in enumerate((edge_values, face_values)):
        tangential = np.asarray(values) @ projector
        for local_index, global_index in enumerate(entities[slot]):
            for component in range(3):
                routes[slot].append(
                    (
                        int(global_index),
                        3 * surface_index + component,
                        0.5
                        * signs[slot][local_index]
                        * float(tangential[local_index, component]),
                    )
                )
    return routes


def _surface_geometry_from_faces(
    complex_: FiniteElementDeRhamComplex,
    faces: np.ndarray,
    /,
) -> _SurfaceGeometry:
    points = np.asarray(complex_.mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int32) for block in complex_.mesh.blocks]
    )
    connectivity = tetrahedral_connectivity(cells, points.shape[0])
    mesh_faces = np.asarray(connectivity.faces)
    face_cell_counts = np.asarray(connectivity.face_cell_counts)
    face_lookup = {
        tuple(int(vertex) for vertex in face): index
        for index, face in enumerate(mesh_faces)
    }
    face_cells: dict[int, list[int]] = {}
    for cell_index, row in enumerate(np.asarray(connectivity.cell_faces)):
        for face in row:
            face_cells.setdefault(int(face), []).append(cell_index)
    directed: set[tuple[int, int]] = set()
    for face in faces:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            key = (int(face[start]), int(face[end]))
            if key in directed:
                raise ValueError(
                    "Huygens face set is not consistently oriented: a directed edge appears twice."
                )
            directed.add(key)
    if any((end, start) not in directed for start, end in directed):
        raise ValueError("Huygens face set is not closed.")
    gradients, entities, signs = _surface_whitney_tables(complex_)
    flux_spec = FormValueSpec(FormType(3, 2, twist="untwisted"), proxy="flux")
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    measures: list[float] = []
    routes: tuple[list[tuple[int, int, float]], ...] = ([], [])
    material_cells: set[int] = set()
    for surface_index, face in enumerate(faces):
        canonical = tuple(sorted(int(vertex) for vertex in face))
        if canonical not in face_lookup:
            raise ValueError(
                "Huygens face set contains a triangle that is not a mesh face."
            )
        global_face = face_lookup[canonical]
        if face_cell_counts[global_face] != 2:
            raise ValueError(
                "Huygens surfaces must lie strictly inside the tetrahedral mesh."
            )
        material_cells.update(face_cells[global_face])
        corners = points[face]
        cross = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        area = 0.5 * float(np.linalg.norm(cross))
        normal = cross / (2.0 * area)
        positions.append(np.mean(corners, axis=0))
        normals.append(normal)
        measures.append(area)
        projector = np.eye(3, dtype=np.float64) - np.outer(normal, normal)
        for cell_index in face_cells[global_face]:
            barycentric = jnp.asarray(
                [1.0 / 3.0 if vertex in face else 0.0 for vertex in cells[cell_index]],
                dtype=jnp.float64,
            )
            local_routes = _surface_cell_routes(
                barycentric,
                gradients[cell_index],
                tuple(row[cell_index] for row in entities),
                tuple(row[cell_index] for row in signs),
                projector,
                flux_spec,
                surface_index,
            )
            routes[0].extend(local_routes[0])
            routes[1].extend(local_routes[1])
    geometry_id = canonical_fingerprint(
        {
            "kind": "maxwell-huygens-surface-geometry",
            "complex": complex_.realization_id,
            "faces": array_tree_fingerprint(faces),
        }
    )
    source = tuple(
        np.asarray([route[0] for route in values], dtype=np.int32) for values in routes
    )
    gather = tuple(
        _gather_map(
            source[slot],
            np.asarray([route[1] for route in values], dtype=np.int32),
            np.asarray([route[2] for route in values], dtype=np.float64),
            source_size=complex_.cell_counts[slot + 1],
            target_count=faces.shape[0],
            operator_id=f"{geometry_id}:{slot + 1}",
        )
        for slot, values in enumerate(routes)
    )
    return _SurfaceGeometry(
        positions=jnp.asarray(np.stack(positions)),
        normals=jnp.asarray(np.stack(normals)),
        measures=jnp.asarray(measures),
        patches=jnp.zeros((faces.shape[0],), dtype=jnp.int32),
        electric_gather=gather[0],
        magnetic_gather=gather[1],
        electric_indices=jnp.asarray(np.unique(source[0])),
        magnetic_indices=jnp.asarray(np.unique(source[1])),
        geometry_id=geometry_id,
        material_cells=jnp.asarray(sorted(material_cells), dtype=jnp.int32),
    )


@final
class MaxwellHuygensSurfacePlan(StrictModule):
    """Closed tetrahedral face surface reconstructing physical E and H.

    Metric pairings carry geometry only. The prepared runtime supplies its
    material-weighted magnetic field, so reconstruction never multiplies a
    material coefficient stored in a Hodge.
    """

    complex: FiniteElementDeRhamComplex
    faces: Array
    acquisition: MaxwellSpectralAcquisition
    exterior: HomogeneousMaxwellExterior
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: FiniteElementDeRhamComplex,
        faces: ArrayLike,
        acquisition: MaxwellSpectralAcquisition,
        exterior: HomogeneousMaxwellExterior,
        /,
    ) -> None:
        if not isinstance(complex, FiniteElementDeRhamComplex):
            raise TypeError("complex must be a FiniteElementDeRhamComplex.")
        if (
            complex.dimension != 3
            or complex.family != "trimmed"
            or complex.order != 1
            or any(block.cell_kind != "tetrahedron" for block in complex.mesh.blocks)
        ):
            raise ValueError(
                "Whitney Huygens sampling requires a trimmed order-one tetrahedral complex."
            )
        faces_ = np.asarray(faces)
        if faces_.ndim != 2 or faces_.shape[1] != 3 or faces_.shape[0] == 0:
            raise ValueError("faces must have shape (faces, 3) with at least one face.")
        if not np.issubdtype(faces_.dtype, np.integer):
            raise TypeError("faces must be integer vertex triples.")
        acquisition_ = _require_huygens_acquisition(acquisition)
        exterior_ = _require_exterior(exterior)
        identifier = canonical_fingerprint(
            {
                "kind": "maxwell-huygens-surface-plan",
                "complex": complex.realization_id,
                "faces": array_tree_fingerprint(faces_),
                "acquisition": acquisition_.acquisition_id,
                "exterior": exterior_.exterior_id,
            }
        )
        self.complex = complex
        self.faces = jnp.asarray(faces_, dtype=jnp.int32)
        self.acquisition = acquisition_
        self.exterior = exterior_
        self.plan_id = identifier

    def prepare(
        self, runtime: PreparedUnstructuredMaxwell, /
    ) -> PreparedMaxwellHuygensSurface:
        if not isinstance(runtime, PreparedUnstructuredMaxwell):
            raise TypeError("runtime must be a PreparedUnstructuredMaxwell.")
        if runtime.plan.cochain.realization_id != self.complex.realization_id:
            raise ValueError(
                "Huygens surface complex does not match the runtime complex."
            )
        geometry = _surface_geometry_from_faces(self.complex, np.asarray(self.faces))
        _surface_material(
            runtime.constitutive,
            np.asarray(geometry.electric_indices),
            np.asarray(geometry.magnetic_indices),
            self.exterior.permittivity,
            self.exterior.permeability,
            cell_indices=None
            if geometry.material_cells is None
            else np.asarray(geometry.material_cells),
        )
        return PreparedMaxwellHuygensSurface(self, geometry)


@final
class PreparedMaxwellHuygensSurface(StrictModule):
    """Prepared tetrahedral Huygens surface; implements `MaxwellHuygensSampler`."""

    geometry: _SurfaceGeometry
    acquisition: MaxwellSpectralAcquisition
    exterior: HomogeneousMaxwellExterior
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: MaxwellHuygensSurfacePlan, geometry: _SurfaceGeometry, /
    ) -> None:
        self.geometry = geometry
        self.acquisition = plan.acquisition
        self.exterior = plan.exterior
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-huygens-surface",
                "plan": plan.plan_id,
                "geometry": geometry.geometry_id,
            }
        )

    @property
    def surface_count(self) -> int:
        return self.geometry.measures.shape[0]

    def initialize(self, /) -> DFTObserverState:
        return self.acquisition.initialize((6 * self.surface_count,))

    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
        *,
        electric_current: ArrayLike | None = None,
    ) -> DFTObserverState:
        """Fold one synchronized sample; a supplied current must vanish on the surface."""
        if not isinstance(state, DFTObserverState):
            raise TypeError("Huygens surface requires DFTObserverState.")
        time_ = jnp.asarray(time)
        electric_ = jnp.asarray(electric)
        if electric_current is not None:
            current = jnp.asarray(electric_current)
            if current.shape != electric_.shape:
                raise ValueError("electric_current must be a degree-one cochain.")
            on_surface = jnp.any(current[self.geometry.electric_indices] != 0)
            electric_ = eqx.error_if(
                electric_,
                self.acquisition.active(time_) & on_surface,
                "Electric current drives the Huygens surface inside the window.",
            )
        payload = _payload(self.geometry, electric_, jnp.asarray(magnetic))
        return self.acquisition.accumulate(state, time_, payload)

    def surface_phasors(self, state: Any, /) -> HuygensSurfacePhasors:
        return _phasors(self.geometry, self.acquisition, self.exterior, state)


def spectral_poynting_energy(
    phasors: HuygensSurfacePhasors,
    face_mask: ArrayLike | None = None,
    /,
) -> Array:
    """One-sided spectral energy ``(1/π) Re ∫ (Ẽ × H̃*)·n̂ dS`` through the surface.

    ``face_mask`` selects surface cells; the total over a closed surface is the
    energy radiated outward per unit angular frequency.
    """
    if not isinstance(phasors, HuygensSurfacePhasors):
        raise TypeError("phasors must be HuygensSurfacePhasors.")
    weights = phasors.measures
    if face_mask is not None:
        mask = jnp.asarray(face_mask, dtype=jnp.bool_)
        if mask.shape != weights.shape:
            raise ValueError("face_mask must have one entry per surface cell.")
        weights = jnp.where(mask, weights, 0.0)
    flux = jnp.real(
        jnp.sum(
            jnp.cross(phasors.electric, jnp.conj(phasors.magnetic))
            * phasors.normals[None],
            axis=-1,
        )
    )
    return jnp.sum(weights[None] * flux, axis=1) / jnp.pi


class MaxwellFarFieldResult(StrictModule):
    """Far-field spectra of one Huygens surface in the ``(θ̂, φ̂)`` basis.

    ``field_spectrum[f, d]`` is ``r Ẽ e^{-ikr}`` resolved on ``(θ̂, φ̂)``;
    ``coherency`` is ``F_i F_j*``; ``stokes`` holds ``(I, Q, U, V)`` with
    ``U = 2 Re(F_θ F_φ*)`` and ``V = -2 Im(F_θ F_φ*)``; ``spectral_energy`` is
    ``εc |F|²/π``, the one-sided ``d²W/(dω dΩ)``.
    """

    __strict_contract__ = True

    angular_frequencies: Float64[_FrequencyDim]
    directions: Float64[_DirectionDim, Literal[3]]
    theta_basis: Float64[_DirectionDim, Literal[3]]
    phi_basis: Float64[_DirectionDim, Literal[3]]
    field_spectrum: Complex128[_FrequencyDim, _DirectionDim, Literal[2]]
    coherency: Complex128[_FrequencyDim, _DirectionDim, Literal[2], Literal[2]]
    stokes: Float64[_FrequencyDim, _DirectionDim, Literal[4]]
    spectral_energy: Float64[_FrequencyDim, _DirectionDim]


class MaxwellFarFieldPlan(StrictModule, NonTrainableState):
    """Radiation-vector far field of Huygens phasors in a homogeneous exterior.

    ``reference_axis`` is the polar axis of the polarization basis:
    ``φ̂ = â × r̂ / |â × r̂|`` and ``θ̂ = φ̂ × r̂``; directions parallel to the
    reference axis are refused. The transform maps over frequencies so that the
    working set is one ``directions × surface`` phase table.
    """

    directions: Array
    reference_axis: Array
    theta_basis: Array
    phi_basis: Array
    exterior: HomogeneousMaxwellExterior
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        directions: ArrayLike,
        reference_axis: ArrayLike,
        exterior: HomogeneousMaxwellExterior,
        /,
    ) -> None:
        directions_ = np.asarray(directions, dtype=np.float64)
        axis = np.asarray(reference_axis, dtype=np.float64)
        if (
            directions_.ndim != 2
            or directions_.shape[1] != 3
            or directions_.shape[0] == 0
        ):
            raise ValueError("directions must have shape (directions, 3).")
        if axis.shape != (3,):
            raise ValueError("reference_axis must have shape (3,).")
        if not np.all(np.isfinite(directions_)) or not np.all(np.isfinite(axis)):
            raise ValueError("Far-field directions and reference axis must be finite.")
        lengths = np.linalg.norm(directions_, axis=1)
        axis_length = np.linalg.norm(axis)
        if np.any(lengths <= 0.0) or axis_length <= 0.0:
            raise ValueError("Far-field directions and reference axis must be nonzero.")
        unit = directions_ / lengths[:, None]
        axis = axis / axis_length
        phi = np.cross(axis[None, :], unit)
        phi_length = np.linalg.norm(phi, axis=1)
        if np.any(phi_length <= 1e-12):
            raise ValueError(
                "Far-field directions parallel to the reference axis have no "
                "polarization basis."
            )
        phi = phi / phi_length[:, None]
        theta = np.cross(phi, unit)
        self.directions = jnp.asarray(unit)
        self.reference_axis = jnp.asarray(axis)
        self.theta_basis = jnp.asarray(theta)
        self.phi_basis = jnp.asarray(phi)
        self.exterior = _require_exterior(exterior)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-far-field-plan",
                "directions": array_tree_fingerprint(unit),
                "reference_axis": array_tree_fingerprint(axis),
                "exterior": self.exterior.exterior_id,
            }
        )

    def evaluate(self, phasors: HuygensSurfacePhasors, /) -> MaxwellFarFieldResult:
        if not isinstance(phasors, HuygensSurfacePhasors):
            raise TypeError("phasors must be HuygensSurfacePhasors.")
        if phasors.exterior.exterior_id != self.exterior.exterior_id:
            raise ValueError("Huygens phasors declare a different exterior medium.")
        impedance = self.exterior.impedance
        projection = self.directions @ phasors.positions.T
        electric_current = jnp.cross(phasors.normals[None], phasors.magnetic)
        magnetic_current = -jnp.cross(phasors.normals[None], phasors.electric)

        def one_frequency(operands: tuple[Array, Array, Array]) -> Array:
            omega, current_j, current_m = operands
            wavenumber = self.exterior.wavenumber(omega)
            phase = jnp.exp(-1j * wavenumber * projection) * phasors.measures[None]
            radiation_n = ein.contract("ds,sc->dc", phase, current_j)
            radiation_l = ein.contract("ds,sc->dc", phase, current_m)
            n_theta = jnp.sum(radiation_n * self.theta_basis, axis=-1)
            n_phi = jnp.sum(radiation_n * self.phi_basis, axis=-1)
            l_theta = jnp.sum(radiation_l * self.theta_basis, axis=-1)
            l_phi = jnp.sum(radiation_l * self.phi_basis, axis=-1)
            factor = 1j * wavenumber / (4.0 * jnp.pi)
            f_theta = factor * (l_phi + impedance * n_theta)
            f_phi = -factor * (l_theta - impedance * n_phi)
            return jnp.stack((f_theta, f_phi), axis=-1)

        spectrum = jax.lax.map(
            one_frequency,
            (phasors.angular_frequencies, electric_current, magnetic_current),
        )
        coherency = spectrum[..., :, None] * jnp.conj(spectrum[..., None, :])
        intensity_theta = jnp.real(coherency[..., 0, 0])
        intensity_phi = jnp.real(coherency[..., 1, 1])
        cross = coherency[..., 0, 1]
        stokes = jnp.stack(
            (
                intensity_theta + intensity_phi,
                intensity_theta - intensity_phi,
                2.0 * jnp.real(cross),
                -2.0 * jnp.imag(cross),
            ),
            axis=-1,
        )
        admittance = self.exterior.permittivity * self.exterior.wave_speed
        return MaxwellFarFieldResult(
            angular_frequencies=phasors.angular_frequencies,
            directions=self.directions,
            theta_basis=self.theta_basis,
            phi_basis=self.phi_basis,
            field_spectrum=spectrum,
            coherency=coherency,
            stokes=stokes,
            spectral_energy=admittance * stokes[..., 0] / jnp.pi,
        )


__all__ = [
    "HomogeneousMaxwellExterior",
    "HuygensSurfacePhasors",
    "MaxwellFarFieldPlan",
    "MaxwellFarFieldResult",
    "MaxwellHuygensBoxPlan",
    "MaxwellHuygensSampler",
    "MaxwellHuygensSurfacePlan",
    "PreparedMaxwellHuygensBox",
    "PreparedMaxwellHuygensSurface",
    "spectral_poynting_energy",
]
