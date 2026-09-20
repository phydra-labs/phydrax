#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import ceil, isfinite, log2, pi

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._topology import CellComplexTopology
from ...linalg._materialization import MaterializationPolicy
from ...operators.quantum._fermionic_fock import (
    fermion_ladder_matrix,
    FermionicFockBasis,
    FermionModeOrder,
)
from ...operators.quantum._gauge_constraints import (
    gauss_leakage,
    GaussConstraintNetwork,
    GaussSectorResourcePolicy,
    prepare_sparse_physical_sector,
    PreparedGaussPhysicalSector,
)
from ...operators.quantum._gauge_link_hilbert import TruncatedU1LinkHilbertSpace
from ...operators.quantum._operations import LocalUnitaryOperation, QuantumProgram
from ...operators.quantum._propagation import apply_local_unitary_to_state
from ...operators.quantum._register import HilbertRegisterLayout
from ...solver._local_hamiltonian import (
    LocalHamiltonian,
    LocalHamiltonianTerm,
    materialize_local_hamiltonian,
)
from ...tensor_network._core import MatrixProductOperator


class PeriodicSchwingerModel(StrictModule):
    """Periodic staggered Schwinger model with retained regulated U(1) links."""

    layout: HilbertRegisterLayout
    mode_order: FermionModeOrder = eqx.field(static=True)
    fermion_basis: FermionicFockBasis = eqx.field(static=True)
    link_space: TruncatedU1LinkHilbertSpace
    staggered_background: Array
    lattice_spacing: Array
    mass: Array
    gauge_coupling: Array
    theta_angle: Array
    valid: Array
    site_count: int = eqx.field(static=True)
    global_flux_sector: int = eqx.field(static=True)
    link_wire_ids: tuple[str, ...] = eqx.field(static=True)
    matter_wire_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)

    def __init__(
        self,
        site_count: int,
        /,
        *,
        maximum_flux: int,
        lattice_spacing: float,
        mass: float,
        gauge_coupling: float,
        theta_angle: float = 0.0,
        global_flux_sector: int = 0,
        maximum_link_dimension: int = 4096,
    ):
        sites = int(site_count)
        spacing = float(lattice_spacing)
        mass_ = float(mass)
        coupling = float(gauge_coupling)
        theta = float(theta_angle)
        sector = int(global_flux_sector)
        if sites < 2 or sites % 2:
            raise ValueError(
                "Periodic staggered Schwinger models require an even site count."
            )
        if not isfinite(spacing) or spacing <= 0.0:
            raise ValueError("lattice_spacing must be finite and positive.")
        if not all(isfinite(value) for value in (mass_, coupling, theta)):
            raise ValueError("Schwinger couplings and theta_angle must be finite.")
        if coupling < 0.0:
            raise ValueError("gauge_coupling must be non-negative.")
        link_space = TruncatedU1LinkHilbertSpace(int(maximum_flux), center_flux=sector)
        if link_space.dimension > int(maximum_link_dimension):
            raise ValueError("Periodic Schwinger link exceeds maximum_link_dimension.")
        matter_ids = tuple(f"matter:{site}" for site in range(sites))
        link_ids = tuple(f"link:{site}->{(site + 1) % sites}" for site in range(sites))
        layout = HilbertRegisterLayout(
            link_ids + matter_ids,
            (link_space.dimension,) * sites + (2,) * sites,
        )
        order = FermionModeOrder(matter_ids)
        background = jnp.asarray(
            tuple(1 if site % 2 == 0 else 0 for site in range(sites)),
            dtype=jnp.float64,
        )
        model_id = canonical_fingerprint(
            {
                "kind": "periodic-schwinger-model",
                "site_count": sites,
                "maximum_flux": int(maximum_flux),
                "global_flux_sector": sector,
                "lattice_spacing": spacing,
                "mass": mass_,
                "gauge_coupling": coupling,
                "theta_angle": theta,
                "encoding": "retained-u1-links-jordan-wigner-matter",
            }
        )
        self.layout = layout
        self.mode_order = order
        self.fermion_basis = FermionicFockBasis(order)
        self.link_space = link_space
        self.staggered_background = background
        self.lattice_spacing = jnp.asarray(spacing)
        self.mass = jnp.asarray(mass_)
        self.gauge_coupling = jnp.asarray(coupling)
        self.theta_angle = jnp.asarray(theta)
        self.valid = link_space.algebra.valid & jnp.asarray(True)
        self.site_count = sites
        self.global_flux_sector = sector
        self.link_wire_ids = link_ids
        self.matter_wire_ids = matter_ids
        self.model_id = model_id
        self.boundary = "periodic"

    @property
    def hopping_scale(self) -> Array:
        return 0.5 / self.lattice_spacing

    @property
    def electric_scale(self) -> Array:
        return 0.5 * self.lattice_spacing * self.gauge_coupling**2

    @property
    def theta_flux(self) -> Array:
        return self.theta_angle / (2.0 * pi)

    def charge_operators(self, /) -> tuple[Array, ...]:
        number = jnp.diag(jnp.asarray((0.0, 1.0), dtype=jnp.complex128))
        identity = jnp.eye(2, dtype=jnp.complex128)
        return tuple(
            number - self.staggered_background[site] * identity
            for site in range(self.site_count)
        )


class CompactU1GaugeModel(StrictModule):
    """Compact U(1) Kogut--Susskind Hamiltonian on a cell-complex two-skeleton."""

    topology: CellComplexTopology
    layout: HilbertRegisterLayout
    link_space: TruncatedU1LinkHilbertSpace
    vertex_link_incidence: Array
    plaquette_link_incidence: Array
    electric_coupling: Array
    magnetic_coupling: Array
    electric_flux_offset: Array
    valid: Array
    link_count: int = eqx.field(static=True)
    plaquette_count: int = eqx.field(static=True)
    link_wire_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        /,
        *,
        maximum_flux: int,
        electric_coupling: float,
        magnetic_coupling: float,
        electric_flux_offset: float = 0.0,
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be a CellComplexTopology.")
        if topology.dimension < 2:
            raise ValueError("Compact U(1) topology must have dimension at least two.")
        electric = float(electric_coupling)
        magnetic = float(magnetic_coupling)
        offset = float(electric_flux_offset)
        if not all(isfinite(value) for value in (electric, magnetic, offset)):
            raise ValueError(
                "Compact U(1) couplings and electric_flux_offset must be finite."
            )
        if electric < 0.0 or magnetic < 0.0:
            raise ValueError("Compact U(1) couplings must be non-negative.")
        link_space = TruncatedU1LinkHilbertSpace(int(maximum_flux))
        vertex_link = np.asarray(topology.incidences[0].scipy_boundary().toarray())
        plaquette_link = np.asarray(topology.incidences[1].scipy_boundary().toarray()).T
        if np.any(~np.isin(vertex_link, (-1, 0, 1))) or np.any(
            ~np.isin(plaquette_link, (-1, 0, 1))
        ):
            raise ValueError("Compact U(1) topology must have unit oriented incidence.")
        link_ids = tuple(
            f"link:{int(identifier)}" for identifier in topology.entities(1).entity_ids
        )
        layout = HilbertRegisterLayout(link_ids, (link_space.dimension,) * len(link_ids))
        model_id = canonical_fingerprint(
            {
                "kind": "compact-u1-gauge-model",
                "topology": topology.topology_id,
                "link_space": link_space.space_id,
                "electric_coupling": electric,
                "magnetic_coupling": magnetic,
                "electric_flux_offset": offset,
            }
        )
        chain_residual = vertex_link @ plaquette_link.T
        self.topology = topology
        self.layout = layout
        self.link_space = link_space
        self.vertex_link_incidence = jnp.asarray(vertex_link, dtype=jnp.int8)
        self.plaquette_link_incidence = jnp.asarray(plaquette_link, dtype=jnp.int8)
        self.electric_coupling = jnp.asarray(electric)
        self.magnetic_coupling = jnp.asarray(magnetic)
        self.electric_flux_offset = jnp.asarray(offset)
        self.valid = link_space.algebra.valid & jnp.asarray(
            np.count_nonzero(chain_residual) == 0
        )
        self.link_count = len(link_ids)
        self.plaquette_count = plaquette_link.shape[0]
        self.link_wire_ids = link_ids
        self.model_id = model_id


class GaugeHamiltonianMPOEvidence(StrictModule):
    dense_residual: Array
    hermiticity_residual: Array
    finite: Array
    valid: Array
    dense_elements: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class GaugeHamiltonianMPOResult(StrictModule):
    operator: MatrixProductOperator
    evidence: GaugeHamiltonianMPOEvidence
    hamiltonian_id: str = eqx.field(static=True)


class GaugeCompilationStatus(IntEnum):
    SUCCESS = 0
    NON_GAUGE_PRESERVING = 1
    NON_UNITARY = 2
    NUMERICAL_FAILURE = 3


class GaugeProgramExecutionStatus(IntEnum):
    SUCCESS = 0
    INVALID_COMPILATION = 1
    GAUGE_LEAKAGE = 2
    NUMERICAL_FAILURE = 3


class GaugeSimulationResourcePolicy(StrictModule):
    """Hard limits for gauge-preserving product-formula compilation."""

    maximum_dense_elements: int = eqx.field(static=True)
    maximum_program_operations: int = eqx.field(static=True)
    maximum_local_exponential_elements: int = eqx.field(static=True)
    gauge_tolerance: float = eqx.field(static=True)
    unitarity_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_dense_elements: int = 1 << 26,
        maximum_program_operations: int = 1_000_000,
        maximum_local_exponential_elements: int = 1 << 24,
        gauge_tolerance: float = 1e-9,
        unitarity_tolerance: float = 1e-9,
    ):
        dense = int(maximum_dense_elements)
        operations = int(maximum_program_operations)
        local = int(maximum_local_exponential_elements)
        gauge = float(gauge_tolerance)
        unitary = float(unitarity_tolerance)
        if dense <= 0 or operations <= 0 or local <= 0:
            raise ValueError("Gauge simulation resource limits must be positive.")
        if not all(isfinite(value) and value >= 0.0 for value in (gauge, unitary)):
            raise ValueError(
                "Gauge simulation tolerances must be finite and non-negative."
            )
        self.maximum_dense_elements = dense
        self.maximum_program_operations = operations
        self.maximum_local_exponential_elements = local
        self.gauge_tolerance = gauge
        self.unitarity_tolerance = unitary
        self.policy_id = canonical_fingerprint(
            {
                "kind": "gauge-simulation-resource-policy",
                "maximum_dense_elements": dense,
                "maximum_program_operations": operations,
                "maximum_local_exponential_elements": local,
                "gauge_tolerance": gauge,
                "unitarity_tolerance": unitary,
            }
        )


class GaugePreservingProductFormulaPlan(StrictModule):
    """Immutable operation schedule admitted before numerical exponentiation."""

    resources: GaugeSimulationResourcePolicy
    term_indices: tuple[int, ...] = eqx.field(static=True)
    term_scales: tuple[float, ...] = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    operation_count: int = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)
    network_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class GaugePreservingCompilationEvidence(StrictModule):
    term_gauss_commutator_residuals: Array
    maximum_gauss_commutator_residual: Array
    unitary_residuals: Array
    maximum_unitary_residual: Array
    finite: Array
    gauge_preserving: Array
    valid: Array
    status: Array
    evidence_id: str = eqx.field(static=True)


class PreparedGaugePreservingProgram(StrictModule):
    plan: GaugePreservingProductFormulaPlan
    program: QuantumProgram
    evidence: GaugePreservingCompilationEvidence
    prepared_id: str = eqx.field(static=True)


class GaugeProgramExecutionEvidence(StrictModule):
    initial_leakage: Array
    final_leakage: Array
    leakage_growth: Array
    norm_residual: Array
    finite: Array
    valid: Array
    status: Array
    prepared_id: str = eqx.field(static=True)


class GaugeProgramExecutionResult(StrictModule):
    final_state: Array
    evidence: GaugeProgramExecutionEvidence
    prepared_id: str = eqx.field(static=True)


class GaugeHamiltonianResourceEvidence(StrictModule):
    """Concrete product-formula counts and norm-based LCU/qubitization estimates."""

    lcu_one_norm: Array
    evolution_time: Array
    target_error: Array
    finite: Array
    valid: Array
    term_count: int = eqx.field(static=True)
    product_formula_order: int = eqx.field(static=True)
    product_formula_steps: int = eqx.field(static=True)
    product_formula_exponentials: int = eqx.field(static=True)
    system_qubits: int = eqx.field(static=True)
    lcu_select_qubits: int = eqx.field(static=True)
    lcu_prepare_amplitudes: int = eqx.field(static=True)
    lcu_block_encoding_qubits: int = eqx.field(static=True)
    qubitization_queries: int = eqx.field(static=True)
    signal_ancilla_qubits: int = eqx.field(static=True)
    estimate_kind: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def periodic_schwinger_gauss_network(
    model: PeriodicSchwingerModel, /
) -> GaussConstraintNetwork:
    if not isinstance(model, PeriodicSchwingerModel):
        raise TypeError("model must be PeriodicSchwingerModel.")
    incidence = np.zeros((model.site_count, model.site_count), dtype=np.int8)
    for site in range(model.site_count):
        incidence[site, site] = 1
        incidence[(site + 1) % model.site_count, site] = -1
    electric = model.link_space.electric_field[None, :, :]
    link_generators = tuple(electric for _ in range(model.site_count))
    number = jnp.diag(jnp.asarray((0.0, 1.0), dtype=jnp.complex128))
    matter = tuple((-number)[None, :, :] for _ in range(model.site_count))
    return GaussConstraintNetwork(
        incidence,
        link_generators,
        link_generators,
        matter_charge_generators=matter,
        background_charges=-model.staggered_background[:, None],
        link_wire_ids=model.link_wire_ids,
        matter_wire_ids=model.matter_wire_ids,
    )


def periodic_schwinger_hamiltonian(
    model: PeriodicSchwingerModel,
    /,
    *,
    maximum_fermion_matrix_elements: int = 1 << 24,
    maximum_term_elements: int = 1 << 26,
) -> LocalHamiltonian:
    """Lower the retained-link periodic Schwinger model to exact local terms."""

    if not isinstance(model, PeriodicSchwingerModel):
        raise TypeError("model must be PeriodicSchwingerModel.")
    hopping_dimension = model.fermion_basis.dimension * model.link_space.dimension
    required_hopping_elements = hopping_dimension * hopping_dimension
    if int(maximum_term_elements) <= 0 or required_hopping_elements > int(
        maximum_term_elements
    ):
        raise ValueError(
            f"Periodic Schwinger hopping terms require {required_hopping_elements} "
            f"elements; capacity is {int(maximum_term_elements)}."
        )
    number = jnp.diag(jnp.asarray((0.0, 1.0), dtype=jnp.complex128))
    identity_link = jnp.eye(model.link_space.dimension, dtype=jnp.complex128)
    shifted_electric = model.link_space.electric_field + model.theta_flux * identity_link
    terms: list[LocalHamiltonianTerm] = []
    for site, wire in enumerate(model.matter_wire_ids):
        terms.append(
            LocalHamiltonianTerm.from_product(
                ((-1.0) ** site * model.mass * number,),
                (wire,),
                term_id=f"{model.model_id}:mass:{site}",
            )
        )
    for link, wire in enumerate(model.link_wire_ids):
        terms.append(
            LocalHamiltonianTerm.from_product(
                (model.electric_scale * shifted_electric @ shifted_electric,),
                (wire,),
                term_id=f"{model.model_id}:electric:{link}",
            )
        )
    creation = tuple(
        fermion_ladder_matrix(
            model.fermion_basis,
            site,
            "create",
            maximum_elements=maximum_fermion_matrix_elements,
        )
        for site in range(model.site_count)
    )
    annihilation = tuple(jnp.conj(value.T) for value in creation)
    matter_targets = model.matter_wire_ids
    for site in range(model.site_count):
        neighbor = (site + 1) % model.site_count
        forward = creation[site] @ annihilation[neighbor]
        transported = jnp.kron(forward, model.link_space.link_operator)
        generator = -1.0j * model.hopping_scale * (transported - jnp.conj(transported.T))
        terms.append(
            LocalHamiltonianTerm(
                generator,
                matter_targets + (model.link_wire_ids[site],),
                term_id=f"{model.model_id}:gauge-hopping:{site}",
            )
        )
    return LocalHamiltonian(
        model.layout,
        terms,
        hamiltonian_id=f"{model.model_id}:hamiltonian",
    )


def periodic_schwinger_physical_sector(
    model: PeriodicSchwingerModel,
    /,
    *,
    resources: GaussSectorResourcePolicy | None = None,
) -> PreparedGaussPhysicalSector:
    """Enumerate an exact diagonal Gauss sector for a resource-bounded small ring."""

    return prepare_sparse_physical_sector(
        periodic_schwinger_gauss_network(model), resources=resources
    )


def periodic_schwinger_flux_values(
    model: PeriodicSchwingerModel, link_levels: ArrayLike, /
) -> Array:
    """Resolve local link-basis levels including the explicit theta displacement."""

    if not isinstance(model, PeriodicSchwingerModel):
        raise TypeError("model must be PeriodicSchwingerModel.")
    levels = jnp.asarray(link_levels, dtype=jnp.int32)
    if levels.shape[-1:] != (model.site_count,):
        raise ValueError("link_levels must have trailing site_count axis.")
    levels = eqx.error_if(
        levels,
        jnp.any((levels < 0) | (levels >= model.link_space.dimension)),
        "A periodic Schwinger link level is outside the truncation.",
    )
    return model.link_space.electric_levels[levels] + model.theta_flux


def compact_u1_gauss_network(model: CompactU1GaugeModel, /) -> GaussConstraintNetwork:
    if not isinstance(model, CompactU1GaugeModel):
        raise TypeError("model must be CompactU1GaugeModel.")
    electric = model.link_space.electric_field[None, :, :]
    link_generators = tuple(electric for _ in range(model.link_count))
    return GaussConstraintNetwork(
        model.vertex_link_incidence,
        link_generators,
        link_generators,
        link_wire_ids=model.link_wire_ids,
    )


def compact_u1_hamiltonian(
    model: CompactU1GaugeModel,
    /,
    *,
    maximum_term_elements: int = 1 << 26,
) -> LocalHamiltonian:
    """Build electric Casimirs and oriented magnetic plaquette terms."""

    if not isinstance(model, CompactU1GaugeModel):
        raise TypeError("model must be CompactU1GaugeModel.")
    maximum = int(maximum_term_elements)
    if maximum <= 0 or model.link_space.dimension**2 > maximum:
        raise ValueError("Compact U(1) electric term exceeds maximum_term_elements.")
    identity = jnp.eye(model.link_space.dimension, dtype=jnp.complex128)
    shifted = model.link_space.electric_field + model.electric_flux_offset * identity
    terms = [
        LocalHamiltonianTerm.from_product(
            (0.5 * model.electric_coupling * shifted @ shifted,),
            (wire,),
            term_id=f"{model.model_id}:electric:{link}",
        )
        for link, wire in enumerate(model.link_wire_ids)
    ]
    incidence = np.asarray(model.plaquette_link_incidence)
    for plaquette in range(model.plaquette_count):
        links = np.flatnonzero(incidence[plaquette])
        if links.size == 0:
            continue
        required = model.link_space.dimension ** (2 * links.size)
        if required > maximum:
            raise ValueError(
                f"Compact U(1) plaquette term requires {required} elements; capacity is {maximum}."
            )
        factors = tuple(
            model.link_space.link_operator
            if incidence[plaquette, link] == 1
            else model.link_space.link_adjoint
            for link in links
        )
        loop = jnp.asarray([[1.0]], dtype=jnp.complex128)
        for factor in factors:
            loop = jnp.kron(loop, factor)
        magnetic = -0.5 * model.magnetic_coupling * (loop + jnp.conj(loop.T))
        terms.append(
            LocalHamiltonianTerm(
                magnetic,
                tuple(model.link_wire_ids[int(link)] for link in links),
                term_id=f"{model.model_id}:magnetic:{plaquette}",
            )
        )
    return LocalHamiltonian(
        model.layout,
        terms,
        hamiltonian_id=f"{model.model_id}:hamiltonian",
    )


def gauge_hamiltonian_mpo(
    hamiltonian: LocalHamiltonian,
    /,
    *,
    maximum_dense_elements: int = 1 << 24,
    maximum_bond_dimension: int = 4096,
    singular_value_tolerance: float = 1e-12,
) -> GaugeHamiltonianMPOResult:
    """Factor an admitted small exact Hamiltonian into an exact heterogeneous MPO."""

    if not isinstance(hamiltonian, LocalHamiltonian):
        raise TypeError("hamiltonian must be LocalHamiltonian.")
    dimension = hamiltonian.layout.dimension
    required = dimension * dimension
    maximum = int(maximum_dense_elements)
    if maximum <= 0 or required > maximum:
        raise ValueError(
            f"Gauge Hamiltonian dense reference requires {required} elements; capacity is {maximum}."
        )
    tolerance = float(singular_value_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("singular_value_tolerance must be finite and non-negative.")
    dense = materialize_local_hamiltonian(
        hamiltonian,
        policy=MaterializationPolicy(
            max_entries=maximum,
            max_bytes=max(16 * maximum, 1),
        ),
    )
    dimensions = hamiltonian.layout.local_dimensions
    sites = len(dimensions)
    tensor = dense.reshape(dimensions + dimensions)
    axes = tuple(index for site in range(sites) for index in (site, sites + site))
    remainder = jnp.transpose(tensor, axes)
    tensors = []
    left_rank = 1
    for site, local_dimension in enumerate(dimensions[:-1]):
        matrix = remainder.reshape((left_rank * local_dimension**2, -1))
        left, singular, right = jnp.linalg.svd(matrix, full_matrices=False)
        rank = singular.shape[0]
        if rank > int(maximum_bond_dimension):
            raise ValueError(
                "Exact gauge Hamiltonian MPO exceeds maximum_bond_dimension."
            )
        core = left[:, :rank].reshape((left_rank, local_dimension, local_dimension, rank))
        tensors.append(core)
        remainder = singular[:rank, None] * right[:rank]
        left_rank = rank
    final_dimension = dimensions[-1]
    tensors.append(remainder.reshape((left_rank, final_dimension, final_dimension, 1)))
    operator = MatrixProductOperator(tuple(tensors))
    reconstructed = operator.to_dense(maximum_elements=maximum)
    dense_residual = jnp.linalg.norm(reconstructed - dense)
    hermiticity = jnp.linalg.norm(reconstructed - jnp.conj(reconstructed.T))
    finite = jnp.all(jnp.isfinite(reconstructed))
    evidence_id = canonical_fingerprint(
        {
            "kind": "gauge-hamiltonian-mpo-evidence",
            "hamiltonian": hamiltonian.hamiltonian_id,
            "operator_structure": operator.structure_id,
            "maximum_bond_dimension": int(maximum_bond_dimension),
            "singular_value_tolerance": tolerance,
        }
    )
    evidence = GaugeHamiltonianMPOEvidence(
        dense_residual,
        hermiticity,
        finite,
        finite & (dense_residual <= 10.0 * tolerance) & (hermiticity <= 10.0 * tolerance),
        required,
        max((1,) + operator.bond_dimensions),
        evidence_id,
    )
    return GaugeHamiltonianMPOResult(operator, evidence, hamiltonian.hamiltonian_id)


def plan_gauge_preserving_product_formula(
    hamiltonian: LocalHamiltonian,
    network: GaussConstraintNetwork,
    time_step: float,
    /,
    *,
    step_count: int,
    order: int = 2,
    resources: GaugeSimulationResourcePolicy | None = None,
) -> GaugePreservingProductFormulaPlan:
    """Admit a fixed product-formula schedule before numerical exponentiation."""

    if not isinstance(hamiltonian, LocalHamiltonian):
        raise TypeError("hamiltonian must be LocalHamiltonian.")
    if not isinstance(network, GaussConstraintNetwork):
        raise TypeError("network must be GaussConstraintNetwork.")
    if hamiltonian.layout.layout_id != network.layout.layout_id:
        raise ValueError("Hamiltonian and Gauss network layouts must agree.")
    if not bool(hamiltonian.valid) or not bool(network.valid):
        raise ValueError("Gauge compilation requires valid Hamiltonian and constraints.")
    step = float(time_step)
    count = int(step_count)
    order_ = int(order)
    if not isfinite(step) or step <= 0.0 or count <= 0:
        raise ValueError("time_step and step_count must be positive.")
    if order_ not in (1, 2):
        raise ValueError("Product-formula order must be one or two.")
    selected = GaugeSimulationResourcePolicy() if resources is None else resources
    if not isinstance(selected, GaugeSimulationResourcePolicy):
        raise TypeError("resources must be GaugeSimulationResourcePolicy or None.")
    term_count = len(hamiltonian.terms)
    if order_ == 1:
        one_step_indices = tuple(range(term_count))
        one_step_scales = (1.0,) * term_count
    else:
        one_step_indices = tuple(range(term_count)) + tuple(range(term_count - 1, -1, -1))
        one_step_scales = (0.5,) * (2 * term_count)
    operation_count = len(one_step_indices) * count
    if operation_count > selected.maximum_program_operations:
        raise ValueError("Product formula exceeds maximum_program_operations.")
    if any(
        term.generator.size > selected.maximum_local_exponential_elements
        for term in hamiltonian.terms
    ):
        raise ValueError(
            "A local exponential exceeds maximum_local_exponential_elements."
        )
    term_indices = one_step_indices * count
    term_scales = one_step_scales * count
    plan_id = canonical_fingerprint(
        {
            "kind": "gauge-preserving-product-formula-plan",
            "hamiltonian": hamiltonian.hamiltonian_id,
            "network": network.network_id,
            "time_step": step,
            "step_count": count,
            "order": order_,
            "term_indices": term_indices,
            "term_scales": term_scales,
            "resources": selected.policy_id,
        }
    )
    return GaugePreservingProductFormulaPlan(
        selected,
        term_indices,
        term_scales,
        step,
        count,
        order_,
        operation_count,
        hamiltonian.hamiltonian_id,
        network.network_id,
        plan_id,
    )


def _single_term_dense(
    hamiltonian: LocalHamiltonian,
    term: LocalHamiltonianTerm,
    maximum_elements: int,
    /,
) -> Array:
    local = LocalHamiltonian(
        hamiltonian.layout,
        (term,),
        hamiltonian_id=f"{hamiltonian.hamiltonian_id}:term:{term.term_id}",
    )
    return materialize_local_hamiltonian(
        local,
        policy=MaterializationPolicy(
            max_entries=maximum_elements,
            max_bytes=max(16 * maximum_elements, 1),
        ),
    )


def prepare_gauge_preserving_product_formula(
    plan: GaugePreservingProductFormulaPlan,
    hamiltonian: LocalHamiltonian,
    network: GaussConstraintNetwork,
    /,
) -> PreparedGaugePreservingProgram:
    """Bind exact local exponentials after certifying every factor commutes with Gauss."""

    if not isinstance(plan, GaugePreservingProductFormulaPlan):
        raise TypeError("plan must be GaugePreservingProductFormulaPlan.")
    if not isinstance(hamiltonian, LocalHamiltonian) or not isinstance(
        network, GaussConstraintNetwork
    ):
        raise TypeError("hamiltonian and network have invalid types.")
    if (
        plan.hamiltonian_id != hamiltonian.hamiltonian_id
        or plan.network_id != network.network_id
    ):
        raise ValueError("Gauge product-formula plan identities changed.")
    dimension = hamiltonian.layout.dimension
    if dimension * dimension > plan.resources.maximum_dense_elements:
        raise ValueError("Gauge commutator certification exceeds maximum_dense_elements.")
    generators = network.dense_generators(
        maximum_elements=plan.resources.maximum_dense_elements
    )
    commutator_residuals = []
    for term in hamiltonian.terms:
        dense = _single_term_dense(
            hamiltonian, term, plan.resources.maximum_dense_elements
        )
        commutators = generators @ dense - dense @ generators
        commutator_residuals.append(jnp.max(jnp.linalg.norm(commutators, axis=(-2, -1))))
    term_residuals = jnp.stack(commutator_residuals)
    maximum_gauss = jnp.max(term_residuals)
    operations = []
    unitarity_residuals = []
    for term_index, scale in zip(plan.term_indices, plan.term_scales, strict=True):
        term = hamiltonian.terms[term_index]
        unitary = jsp.linalg.expm(-1.0j * plan.time_step * scale * term.generator)
        identity = jnp.eye(unitary.shape[0], dtype=unitary.dtype)
        unitarity_residuals.append(
            jnp.linalg.norm(jnp.conj(unitary.T) @ unitary - identity)
        )
        operations.append(LocalUnitaryOperation(unitary, term.target_wire_ids))
    unitary_residual = jnp.stack(unitarity_residuals)
    maximum_unitary = jnp.max(unitary_residual)
    finite = jnp.all(jnp.isfinite(term_residuals)) & jnp.all(
        jnp.isfinite(unitary_residual)
    )
    gauge_preserving = maximum_gauss <= plan.resources.gauge_tolerance
    valid = (
        finite
        & gauge_preserving
        & (maximum_unitary <= plan.resources.unitarity_tolerance)
    )
    status = jnp.where(
        ~finite,
        int(GaugeCompilationStatus.NUMERICAL_FAILURE),
        jnp.where(
            ~gauge_preserving,
            int(GaugeCompilationStatus.NON_GAUGE_PRESERVING),
            jnp.where(
                maximum_unitary > plan.resources.unitarity_tolerance,
                int(GaugeCompilationStatus.NON_UNITARY),
                int(GaugeCompilationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    program = QuantumProgram(hamiltonian.layout, operations, state_kind="state-vector")
    evidence_id = canonical_fingerprint(
        {
            "kind": "gauge-preserving-compilation-evidence",
            "plan": plan.plan_id,
            "program": program.program_id,
        }
    )
    evidence = GaugePreservingCompilationEvidence(
        term_residuals,
        maximum_gauss,
        unitary_residual,
        maximum_unitary,
        finite,
        gauge_preserving,
        valid,
        status,
        evidence_id,
    )
    return PreparedGaugePreservingProgram(
        plan,
        program,
        evidence,
        canonical_fingerprint(
            {
                "kind": "prepared-gauge-preserving-program",
                "plan": plan.plan_id,
                "program": program.program_id,
            }
        ),
    )


def compile_gauge_preserving_product_formula(
    hamiltonian: LocalHamiltonian,
    network: GaussConstraintNetwork,
    time_step: float,
    /,
    *,
    step_count: int,
    order: int = 2,
    resources: GaugeSimulationResourcePolicy | None = None,
) -> PreparedGaugePreservingProgram:
    plan = plan_gauge_preserving_product_formula(
        hamiltonian,
        network,
        time_step,
        step_count=step_count,
        order=order,
        resources=resources,
    )
    return prepare_gauge_preserving_product_formula(plan, hamiltonian, network)


def execute_gauge_preserving_program(
    prepared: PreparedGaugePreservingProgram,
    network: GaussConstraintNetwork,
    initial_state: ArrayLike,
    /,
) -> GaugeProgramExecutionResult:
    """Execute the fixed program and report actual initial/final Gauss leakage."""

    if not isinstance(prepared, PreparedGaugePreservingProgram):
        raise TypeError("prepared must be PreparedGaugePreservingProgram.")
    if not isinstance(network, GaussConstraintNetwork):
        raise TypeError("network must be GaussConstraintNetwork.")
    if prepared.plan.network_id != network.network_id:
        raise ValueError("Prepared gauge program and network identities differ.")
    state = jnp.asarray(initial_state)
    if state.shape != (network.physical_dimension,):
        raise ValueError("initial_state has the wrong gauge Hilbert dimension.")
    if not jnp.issubdtype(state.dtype, jnp.complexfloating):
        state = state.astype(jnp.complex128)
    initial = gauss_leakage(
        network,
        state,
        maximum_elements=prepared.plan.resources.maximum_dense_elements,
    )
    final = state
    for operation in prepared.program.operations:
        final = apply_local_unitary_to_state(
            prepared.program.layout,
            operation.unitary,
            operation.target_wire_ids,
            final,
        )
    final_leakage = gauss_leakage(
        network,
        final,
        maximum_elements=prepared.plan.resources.maximum_dense_elements,
    )
    growth = jnp.maximum(final_leakage - initial, 0.0)
    norm_residual = jnp.abs(
        jnp.real(jnp.vdot(final, final)) - jnp.real(jnp.vdot(state, state))
    )
    finite = jnp.all(jnp.isfinite(final)) & jnp.all(
        jnp.isfinite(jnp.stack((initial, final_leakage, norm_residual)))
    )
    valid = (
        finite
        & prepared.evidence.valid
        & (growth <= prepared.plan.resources.gauge_tolerance)
        & (norm_residual <= prepared.plan.resources.unitarity_tolerance)
    )
    status = jnp.where(
        ~finite,
        int(GaugeProgramExecutionStatus.NUMERICAL_FAILURE),
        jnp.where(
            ~prepared.evidence.valid,
            int(GaugeProgramExecutionStatus.INVALID_COMPILATION),
            jnp.where(
                (growth > prepared.plan.resources.gauge_tolerance)
                | (norm_residual > prepared.plan.resources.unitarity_tolerance),
                int(GaugeProgramExecutionStatus.GAUGE_LEAKAGE),
                int(GaugeProgramExecutionStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = GaugeProgramExecutionEvidence(
        initial,
        final_leakage,
        growth,
        norm_residual,
        finite,
        valid,
        status,
        prepared.prepared_id,
    )
    return GaugeProgramExecutionResult(final, evidence, prepared.prepared_id)


def estimate_gauge_hamiltonian_resources(
    hamiltonian: LocalHamiltonian,
    evolution_time: float,
    target_error: float,
    /,
    *,
    product_formula_order: int = 2,
    product_formula_steps: int | None = None,
) -> GaugeHamiltonianResourceEvidence:
    """Report exact IR counts and explicit norm-only simulation estimates.

    LCU one-norm uses local spectral norms.  The qubitization query count records
    ``ceil(alpha*t + log(1/error))`` as a resource estimate, not an error theorem.
    """

    if not isinstance(hamiltonian, LocalHamiltonian):
        raise TypeError("hamiltonian must be LocalHamiltonian.")
    time = float(evolution_time)
    error = float(target_error)
    order = int(product_formula_order)
    if not isfinite(time) or time < 0.0:
        raise ValueError("evolution_time must be finite and non-negative.")
    if not isfinite(error) or not 0.0 < error < 1.0:
        raise ValueError("target_error must lie strictly between zero and one.")
    if order not in (1, 2):
        raise ValueError("product_formula_order must be one or two.")
    norms = jnp.stack(
        tuple(jnp.linalg.norm(term.generator, ord=2) for term in hamiltonian.terms)
    )
    alpha = jnp.sum(norms)
    alpha_host = float(alpha)
    if product_formula_steps is None:
        scale = max(alpha_host * time, 1e-15)
        steps = max(
            1,
            int(ceil(scale ** (1.0 + 1.0 / order) / error ** (1.0 / order))),
        )
    else:
        steps = int(product_formula_steps)
        if steps <= 0:
            raise ValueError("product_formula_steps must be positive.")
    term_count = len(hamiltonian.terms)
    per_step = term_count if order == 1 else 2 * term_count
    select_qubits = int(ceil(log2(term_count))) if term_count > 1 else 0
    system_qubits = int(ceil(log2(hamiltonian.layout.dimension)))
    queries = int(ceil(alpha_host * time + np.log(1.0 / error)))
    finite = jnp.isfinite(alpha) & jnp.asarray(isfinite(time))
    evidence_id = canonical_fingerprint(
        {
            "kind": "gauge-hamiltonian-resource-evidence",
            "hamiltonian": hamiltonian.hamiltonian_id,
            "evolution_time": time,
            "target_error": error,
            "product_formula_order": order,
            "product_formula_steps": steps,
            "estimate_kind": "norm-only-explicit-count-estimate",
        }
    )
    return GaugeHamiltonianResourceEvidence(
        alpha,
        jnp.asarray(time),
        jnp.asarray(error),
        finite,
        finite & (alpha >= 0.0),
        term_count,
        order,
        steps,
        per_step * steps,
        system_qubits,
        select_qubits,
        term_count,
        system_qubits + select_qubits + 1,
        queries,
        1,
        "norm-only-explicit-count-estimate",
        evidence_id,
    )


__all__ = [
    "CompactU1GaugeModel",
    "GaugeCompilationStatus",
    "GaugeHamiltonianMPOEvidence",
    "GaugeHamiltonianMPOResult",
    "GaugeHamiltonianResourceEvidence",
    "GaugePreservingCompilationEvidence",
    "GaugePreservingProductFormulaPlan",
    "GaugeProgramExecutionEvidence",
    "GaugeProgramExecutionResult",
    "GaugeProgramExecutionStatus",
    "GaugeSimulationResourcePolicy",
    "PeriodicSchwingerModel",
    "PreparedGaugePreservingProgram",
    "compact_u1_gauss_network",
    "compact_u1_hamiltonian",
    "compile_gauge_preserving_product_formula",
    "estimate_gauge_hamiltonian_resources",
    "execute_gauge_preserving_program",
    "gauge_hamiltonian_mpo",
    "periodic_schwinger_flux_values",
    "periodic_schwinger_gauss_network",
    "periodic_schwinger_hamiltonian",
    "periodic_schwinger_physical_sector",
    "plan_gauge_preserving_product_formula",
    "prepare_gauge_preserving_product_formula",
]
