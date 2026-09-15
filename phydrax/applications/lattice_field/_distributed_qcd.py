#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Partition-independent lattice QCD execution, preparation, and transitions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends.lattice import (
    LatticeKernelCapabilities,
    LatticeKernelProvider,
    NativeJaxLatticeProvider,
)
from ...discretization._lattice_distribution import LatticeDecompositionPlan
from ...graph._gauge_transport import GaugeCovariantShiftPlan, GaugeStaplePlan
from ...linalg._dense_pseudoinverse import (
    apply_pseudoinverse,
    DensePseudoinverseFactors,
    factor_pseudoinverse,
)
from ...linalg._local_blocks import (
    LocalBlockFactorization,
    prepare_local_block_factorization,
    solve_local_blocks,
)
from ...linalg._policies import RankPolicy
from ...linalg.krylov._decompositions import _orthonormalize_block
from ...metrix._complex_matrix_manifold import SpecialUnitaryGroup
from ...metrix._lie_group import LieGroupStateGeometry
from ...metrix._state_geometry import PointwiseStateGeometry
from ...operators.path_integral._lattice_fermion import (
    AbstractLatticeDiracOperator,
    CloverWilsonDiracOperator,
    WilsonDiracOperator,
)
from ...operators.path_integral._pseudofermion import PseudofermionTerm
from ...sampling._compact_group_hamiltonian import (
    CompactGeometricTarget,
    initialize_compact_group_hamiltonian_state,
    prepare_compact_group_hamiltonian_kernel,
    PreparedCompactGroupHamiltonianKernel,
    sample_compact_group_hamiltonian,
)
from ...sampling._rhmc import (
    initialize_rhmc_state,
    prepare_rhmc,
    PreparedRHMCKernel,
    rhmc_transition,
    RHMCChainState,
    RHMCPlan,
    RHMCSampleResult,
    RHMCTransitionResult,
    sample_rhmc,
    SeparableActionRegistry,
    SeparableActionTerm,
)


def _matrix_product(left: Array, right: Array, /) -> Array:
    return contract("...ij,...jk->...ik", left, right, backend="jax")


def _linear_product(matrix: Array, value: Array, /) -> Array:
    return contract("ij,j...->i...", matrix, value, backend="jax")


def _adjoint(value: Array, /) -> Array:
    return jnp.conj(jnp.swapaxes(value, -1, -2))


def _canonical_links(
    decomposition: LatticeDecompositionPlan,
    links: ArrayLike,
    /,
) -> Array:
    values = jnp.asarray(links)
    canonical_shape = (
        decomposition.site_count,
        decomposition.dimension,
    )
    if (
        values.ndim == 3
        and values.shape[0] == decomposition.site_count * decomposition.dimension
        and values.shape[-1] == values.shape[-2]
    ):
        values = values.reshape(canonical_shape + values.shape[-2:])
    elif (
        values.ndim != 4
        or values.shape[:2] != canonical_shape
        or values.shape[-1] != values.shape[-2]
    ):
        raise ValueError(
            "links must use canonical (site, direction, color, color) or flattened-link shape."
        )
    if not jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError("Distributed gauge links require a complex dtype.")
    return values


def _owned_site_payload(
    decomposition: LatticeDecompositionPlan,
    values: Array,
    /,
) -> Array:
    result = jnp.zeros(
        (decomposition.partition_count, decomposition.owned_capacity) + values.shape[1:],
        dtype=values.dtype,
    )
    for part in range(decomposition.partition_count):
        ids = decomposition.owned_global_ids[part]
        mask = decomposition.owned_valid[part].reshape(
            (decomposition.owned_capacity,) + (1,) * (values.ndim - 1)
        )
        result = result.at[part].set(jnp.where(mask, values[ids], 0))
    return result


class DistributedGaugeTheoryPlan(StrictModule, NonTrainableState):
    """Static Wilson-gauge execution contract over one exact decomposition."""

    decomposition: LatticeDecompositionPlan
    beta: Array
    maximum_gauge_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        decomposition: LatticeDecompositionPlan,
        beta: ArrayLike,
        /,
        *,
        maximum_gauge_bytes: int = 4_294_967_296,
    ) -> None:
        if not isinstance(decomposition, LatticeDecompositionPlan):
            raise TypeError("decomposition must be LatticeDecompositionPlan.")
        beta_ = np.asarray(beta)
        if beta_.shape != () or not np.isfinite(beta_) or beta_ < 0.0:
            raise ValueError("beta must be a finite non-negative scalar.")
        maximum = int(maximum_gauge_bytes)
        if maximum <= 0:
            raise ValueError("maximum_gauge_bytes must be positive.")
        self.decomposition = decomposition
        self.beta = jnp.asarray(beta_)
        self.maximum_gauge_bytes = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-wilson-gauge-plan",
                "decomposition": decomposition.plan_id,
                "beta": float(beta_),
                "maximum_gauge_bytes": maximum,
            }
        )

    def prepare(
        self,
        provider: LatticeKernelProvider | None = None,
        /,
    ) -> "PreparedDistributedGaugeTheory":
        return PreparedDistributedGaugeTheory(
            self,
            NativeJaxLatticeProvider() if provider is None else provider,
        )


class DistributedGaugeActionResult(StrictModule):
    value: Array
    local_values: Array
    local_face_counts: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)


class DistributedGaugeForceResult(StrictModule):
    value: Array
    local_values: Array
    staples: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)


class DistributedDslashResult(StrictModule):
    value: Array
    local_owned_values: Array
    finite: Array
    successful: Array
    block_width: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    decomposition_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)


class PreparedDistributedGaugeTheory(StrictModule, NonTrainableState):
    """Prepared provider binding; gauge fields remain runtime numeric values."""

    plan: DistributedGaugeTheoryPlan
    provider: LatticeKernelProvider = eqx.field(static=True)
    capabilities: LatticeKernelCapabilities
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DistributedGaugeTheoryPlan,
        provider: LatticeKernelProvider,
        /,
    ) -> None:
        if not isinstance(plan, DistributedGaugeTheoryPlan):
            raise TypeError("plan must be DistributedGaugeTheoryPlan.")
        if not isinstance(provider.capabilities, LatticeKernelCapabilities):
            raise TypeError("provider must expose LatticeKernelCapabilities.")
        self.plan = plan
        self.provider = provider
        self.capabilities = provider.capabilities
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-wilson-gauge",
                "plan": plan.plan_id,
                "provider": provider.provider_id,
                "capabilities": provider.capabilities.capabilities_id,
            }
        )

    def _links(self, links: ArrayLike, /) -> Array:
        values = _canonical_links(self.plan.decomposition, links)
        if values.nbytes > self.plan.maximum_gauge_bytes:
            raise ValueError("Gauge field exceeds maximum_gauge_bytes.")
        return values

    def _configuration(self, links: ArrayLike, /) -> Array:
        values = jnp.asarray(links)
        canonical = self._links(values)
        if values.shape not in (
            canonical.shape,
            (canonical.shape[0] * canonical.shape[1],) + canonical.shape[2:],
        ):
            raise ValueError("Gauge configuration shape is not canonical.")
        return values

    def plaquettes(self, links: ArrayLike, /) -> tuple[Array, Array]:
        values = self._links(links)
        decomposition = self.plan.decomposition
        axes = np.asarray(decomposition.ownership.face_axes)
        plaquettes = []
        valid = []
        for left, right in axes:
            mu, nu = int(left), int(right)
            plus_mu = decomposition.neighbor_ids[:, mu, 1]
            plus_nu = decomposition.neighbor_ids[:, nu, 1]
            u_mu = values[:, mu]
            u_nu_at_mu = values[plus_mu, nu]
            u_mu_at_nu = values[plus_nu, mu]
            u_nu = values[:, nu]
            product_1 = _matrix_product(u_mu, u_nu_at_mu)
            product_2 = _matrix_product(product_1, _adjoint(u_mu_at_nu))
            plaquettes.append(_matrix_product(product_2, _adjoint(u_nu)))
            valid.append(
                decomposition.neighbor_valid[:, mu, 1]
                & decomposition.neighbor_valid[:, nu, 1]
            )
        if not plaquettes:
            color = values.shape[-1]
            return (
                jnp.zeros(
                    (decomposition.site_count, 0, color, color), dtype=values.dtype
                ),
                jnp.zeros((decomposition.site_count, 0), dtype=bool),
            )
        return jnp.stack(plaquettes, axis=1), jnp.stack(valid, axis=1)

    def gauge_action(self, links: ArrayLike, /) -> DistributedGaugeActionResult:
        plaquettes, valid = self.plaquettes(links)
        self.capabilities.require("gauge.wilson_action", plaquettes.dtype)
        decomposition = self.plan.decomposition
        local_values = []
        local_counts = []
        for part in range(decomposition.partition_count):
            mask = decomposition.ownership.face_mask(part) & valid
            local_values.append(
                self.provider.gauge_action(plaquettes, self.plan.beta, mask)
            )
            local_counts.append(jnp.sum(mask, dtype=jnp.int32))
        local = jnp.stack(local_values)
        counts = jnp.stack(local_counts)
        value = jnp.sum(local)
        finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(local))
        expected_faces = jnp.sum(decomposition.ownership.face_valid, dtype=jnp.int32)
        successful = finite & (jnp.sum(counts) == expected_faces)
        return DistributedGaugeActionResult(
            value,
            local,
            counts,
            finite,
            successful,
            self.plan.plan_id,
            self.provider.provider_id,
        )

    def gauge_force(
        self,
        links: ArrayLike,
        staples: ArrayLike,
        /,
    ) -> DistributedGaugeForceResult:
        """Partition a force built from graph-owned, precomputed staple sums."""

        values = self._links(links)
        self.capabilities.require("gauge.wilson_force", values.dtype)
        staple_values = _canonical_links(self.plan.decomposition, staples)
        if staple_values.shape != values.shape:
            raise ValueError(
                "staples must match canonical links after graph GaugeStaplePlan evaluation."
            )
        decomposition = self.plan.decomposition
        local = []
        for part in range(decomposition.partition_count):
            local.append(
                self.provider.gauge_force(
                    values,
                    staple_values,
                    self.plan.beta,
                    decomposition.ownership.link_mask(part),
                )
            )
        local_values = jnp.stack(local)
        value = jnp.sum(local_values, axis=0)
        finite = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(staple_values))
        ownership_count = jnp.sum(
            jnp.stack(
                tuple(
                    decomposition.ownership.link_mask(part).astype(jnp.int32)
                    for part in range(decomposition.partition_count)
                )
            ),
            axis=0,
        )
        successful = finite & jnp.all(ownership_count == 1)
        return DistributedGaugeForceResult(
            value,
            local_values,
            staple_values,
            finite,
            successful,
            self.plan.plan_id,
            self.provider.provider_id,
        )

    def gauge_force_from_graph(
        self,
        transport: GaugeCovariantShiftPlan,
        staple_plan: GaugeStaplePlan,
        links: ArrayLike,
        /,
    ) -> DistributedGaugeForceResult:
        """Evaluate graph-owned staples and reduce each graph edge exactly once."""

        if not isinstance(transport, GaugeCovariantShiftPlan):
            raise TypeError("transport must be GaugeCovariantShiftPlan.")
        if not isinstance(staple_plan, GaugeStaplePlan):
            raise TypeError("staple_plan must be GaugeStaplePlan.")
        if (
            transport.link_space_id != staple_plan.link_space.link_space_id
            or transport.site_count != self.plan.decomposition.site_count
            or transport.dimension != self.plan.decomposition.dimension
        ):
            raise ValueError(
                "Gauge transport, staples, and decomposition must share one lattice."
            )
        values = jnp.asarray(links)
        if values.shape != staple_plan.link_space.configuration_shape:
            raise ValueError("links must match the graph gauge-link configuration.")
        if values.nbytes > self.plan.maximum_gauge_bytes:
            raise ValueError("Gauge field exceeds maximum_gauge_bytes.")
        self.capabilities.require("gauge.wilson_force", values.dtype)
        forward_edges = np.asarray(transport.forward_edges).reshape((-1,))
        edge_count = staple_plan.link_space.num_edges
        if forward_edges.size != edge_count or not np.array_equal(
            np.sort(forward_edges), np.arange(edge_count)
        ):
            raise ValueError(
                "Graph forward routes must bijectively identify canonical lattice links."
            )
        directed_owner = np.asarray(self.plan.decomposition.ownership.link_owner).reshape(
            (-1,)
        )
        edge_owner = np.zeros((edge_count,), dtype=np.int32)
        edge_owner[forward_edges] = directed_owner
        staple_values = staple_plan.staples(values)
        local_values = jnp.stack(
            tuple(
                self.provider.gauge_force(
                    values,
                    staple_values,
                    self.plan.beta,
                    jnp.asarray(edge_owner == part),
                )
                for part in range(self.plan.decomposition.partition_count)
            )
        )
        value = jnp.sum(local_values, axis=0)
        finite = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(staple_values))
        ownership_count = jnp.sum(
            jnp.stack(
                tuple(
                    jnp.asarray(edge_owner == part, dtype=jnp.int32)
                    for part in range(self.plan.decomposition.partition_count)
                )
            ),
            axis=0,
        )
        successful = finite & jnp.all(ownership_count == 1)
        return DistributedGaugeForceResult(
            value,
            local_values,
            staple_values,
            finite,
            successful,
            self.plan.plan_id,
            self.provider.provider_id,
        )

    def dslash(
        self,
        operator: AbstractLatticeDiracOperator,
        spinor: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> DistributedDslashResult:
        if not isinstance(operator, AbstractLatticeDiracOperator):
            raise TypeError("operator must implement AbstractLatticeDiracOperator.")
        if operator.lattice_shape != self.plan.decomposition.global_shape:
            raise ValueError("Dirac operator and decomposition lattice shapes disagree.")
        values = jnp.asarray(spinor)
        expected = (
            operator.site_count,
            operator.spin_components,
            operator.color_components,
        )
        is_block = values.ndim == 4
        if (
            values.shape[:3] != expected
            or (is_block and values.shape[3] < 1)
            or values.ndim not in (3, 4)
        ):
            raise ValueError(
                "spinor must have shape (site, spin, color) with an optional final RHS axis."
            )
        if is_block:
            self.capabilities.require("fermion.block_dslash", values.dtype)
        elif isinstance(operator, CloverWilsonDiracOperator):
            self.capabilities.require("fermion.clover_dslash", values.dtype)
        elif isinstance(operator, WilsonDiracOperator):
            self.capabilities.require("fermion.wilson_dslash", values.dtype)
        if is_block:
            result = self.provider.block_dslash(operator, values, adjoint=adjoint)
            block_width = int(values.shape[-1])
        elif isinstance(operator, CloverWilsonDiracOperator):
            result = self.provider.clover_dslash(operator, values, adjoint=adjoint)
            block_width = 1
        elif isinstance(operator, WilsonDiracOperator):
            result = self.provider.wilson_dslash(operator, values, adjoint=adjoint)
            block_width = 1
        else:
            raise TypeError(
                "Distributed Dslash supports Wilson and clover-Wilson operators."
            )
        finite = jnp.all(jnp.isfinite(result))
        return DistributedDslashResult(
            result,
            _owned_site_payload(self.plan.decomposition, result),
            finite,
            finite,
            block_width,
            operator.operator_id,
            self.plan.decomposition.plan_id,
            self.provider.provider_id,
        )


class DeflationPlan(StrictModule, NonTrainableState):
    mode_capacity: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    maximum_basis_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_capacity: int,
        /,
        *,
        rank_tolerance: float = 1.0e-8,
        maximum_basis_bytes: int = 1_073_741_824,
    ) -> None:
        modes = int(mode_capacity)
        tolerance = float(rank_tolerance)
        maximum = int(maximum_basis_bytes)
        if modes <= 0 or not np.isfinite(tolerance) or tolerance <= 0.0 or maximum <= 0:
            raise ValueError(
                "Deflation capacity, tolerance, and byte limit must be positive."
            )
        self.mode_capacity = modes
        self.rank_tolerance = tolerance
        self.maximum_basis_bytes = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-deflation-plan",
                "mode_capacity": modes,
                "rank_tolerance": tolerance,
                "maximum_basis_bytes": maximum,
            }
        )


class PreparedDeflation(StrictModule, NonTrainableState):
    plan: DeflationPlan
    basis: Array
    images: Array
    coarse_factors: DensePseudoinverseFactors
    rank: Array
    orthogonality_error: Array
    residual_norms: Array
    finite: Array
    field_shape: tuple[int, int, int] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def coarse_correction(self, right_hand_side: ArrayLike, /) -> Array:
        values = jnp.asarray(right_hand_side)
        original_shape = values.shape
        if values.shape[: len(self.field_shape)] == self.field_shape:
            trailing = values.shape[len(self.field_shape) :]
            coordinates = values.reshape((self.basis.shape[0],) + trailing)
        elif values.shape[:1] == (self.basis.shape[0],):
            coordinates = values
        else:
            raise ValueError(
                "Deflation right-hand side must use canonical field or coordinate shape."
            )
        vector = coordinates.ndim == 1
        columns = (
            coordinates[:, None]
            if vector
            else coordinates.reshape((coordinates.shape[0], -1))
        )
        projected = _matrix_product(_adjoint(self.basis), columns)
        coefficients = apply_pseudoinverse(self.coarse_factors, projected)
        correction = _matrix_product(self.basis, coefficients)
        restored = correction[:, 0] if vector else correction.reshape(coordinates.shape)
        return restored.reshape(original_shape)


def prepare_deflation(
    plan: DeflationPlan,
    operator: AbstractLatticeDiracOperator,
    candidates: ArrayLike,
    /,
) -> PreparedDeflation:
    if not isinstance(plan, DeflationPlan):
        raise TypeError("plan must be DeflationPlan.")
    if not isinstance(operator, AbstractLatticeDiracOperator):
        raise TypeError("operator must implement AbstractLatticeDiracOperator.")
    values = jnp.asarray(candidates)
    field_shape = (
        operator.site_count,
        operator.spin_components,
        operator.color_components,
        plan.mode_capacity,
    )
    if values.shape != field_shape:
        raise ValueError(f"candidates must have shape {field_shape}.")
    coordinates = values.reshape((operator.source.size, plan.mode_capacity))
    required_bytes = 3 * coordinates.size * coordinates.dtype.itemsize
    if required_bytes > plan.maximum_basis_bytes:
        raise ValueError("Deflation basis exceeds maximum_basis_bytes.")
    inner = lambda left, right: jnp.vdot(left, right)
    basis, _, rank = _orthonormalize_block(
        coordinates,
        inner,
        jnp.asarray(plan.rank_tolerance, dtype=coordinates.real.dtype),
    )
    basis_fields = basis.reshape(field_shape)
    image_fields = jax.vmap(operator.mv, in_axes=-1, out_axes=-1)(basis_fields)
    images = image_fields.reshape(coordinates.shape)
    coarse = _matrix_product(_adjoint(basis), images)
    coarse_factors = factor_pseudoinverse(
        coarse,
        RankPolicy(relative_cutoff=plan.rank_tolerance),
    )
    gram = _matrix_product(_adjoint(basis), basis)
    active = jnp.arange(plan.mode_capacity) < rank
    active_matrix = active[:, None] & active[None, :]
    orthogonality_error = jnp.max(
        jnp.abs(
            jnp.where(
                active_matrix,
                gram - jnp.eye(plan.mode_capacity, dtype=gram.dtype),
                0,
            )
        )
    )
    correction = _matrix_product(basis, coarse)
    residual_norms = jnp.sqrt(jnp.sum(jnp.abs(images - correction) ** 2, axis=0))
    finite = (
        jnp.all(jnp.isfinite(basis))
        & jnp.all(jnp.isfinite(images))
        & coarse_factors.finite
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-lattice-deflation",
            "plan": plan.plan_id,
            "operator": operator.operator_id,
            "candidate_signature": array_tree_fingerprint(np.asarray(values)),
        }
    )
    return PreparedDeflation(
        plan,
        basis,
        images,
        coarse_factors,
        rank,
        orthogonality_error,
        residual_norms,
        finite,
        tuple(int(value) for value in operator.source.shape),
        operator.operator_id,
        prepared_id,
    )


class SAPPlan(StrictModule, NonTrainableState):
    decomposition: LatticeDecompositionPlan
    sweep_count: int = eqx.field(static=True)
    maximum_factor_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        decomposition: LatticeDecompositionPlan,
        /,
        *,
        sweep_count: int = 2,
        maximum_factor_bytes: int = 2_147_483_648,
    ) -> None:
        if not isinstance(decomposition, LatticeDecompositionPlan):
            raise TypeError("decomposition must be LatticeDecompositionPlan.")
        sweeps = int(sweep_count)
        maximum = int(maximum_factor_bytes)
        if sweeps <= 0 or maximum <= 0:
            raise ValueError("SAP sweep count and factor byte limit must be positive.")
        self.decomposition = decomposition
        self.sweep_count = sweeps
        self.maximum_factor_bytes = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-sap-plan",
                "decomposition": decomposition.plan_id,
                "sweep_count": sweeps,
                "maximum_factor_bytes": maximum,
            }
        )


class PreparedSAP(StrictModule, NonTrainableState):
    plan: SAPPlan
    matrix: Array
    block_indices: Array
    block_valid: Array
    block_parity: Array
    factorization: LocalBlockFactorization
    finite: Array
    prepared_id: str = eqx.field(static=True)

    def apply(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        initial: ArrayLike | None = None,
    ) -> Array:
        rhs = jnp.asarray(right_hand_side)
        if rhs.shape[:1] != (self.matrix.shape[0],):
            raise ValueError("SAP right-hand side leading dimension is invalid.")
        result = jnp.zeros_like(rhs) if initial is None else jnp.asarray(initial)
        if result.shape != rhs.shape:
            raise ValueError("SAP initial value must match right_hand_side.")
        for _ in range(self.plan.sweep_count):
            for color in range(2):
                residual = rhs - _linear_product(self.matrix, result)
                local_rhs = residual[self.block_indices]
                correction, failed = solve_local_blocks(self.factorization, local_rhs)
                active = self.block_valid & (self.block_parity[:, None] == color)
                mask = active.reshape(active.shape + (1,) * (correction.ndim - 2))
                correction = jnp.where(mask, correction, 0)
                result = result.at[self.block_indices].add(correction)
                result = jnp.where(failed, jnp.zeros_like(result), result)
        return result


def prepare_sap(
    plan: SAPPlan,
    operator_matrix: ArrayLike,
    /,
) -> PreparedSAP:
    if not isinstance(plan, SAPPlan):
        raise TypeError("plan must be SAPPlan.")
    matrix = jnp.asarray(operator_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator_matrix must be square.")
    decomposition = plan.decomposition
    if matrix.shape[0] % decomposition.site_count:
        raise ValueError(
            "Operator dimension must contain an integral number of site DOFs."
        )
    dofs_per_site = matrix.shape[0] // decomposition.site_count
    block_size = decomposition.owned_capacity * dofs_per_site
    required_bytes = (
        decomposition.partition_count * block_size * block_size * matrix.dtype.itemsize
    )
    if required_bytes > plan.maximum_factor_bytes:
        raise ValueError("SAP local factors exceed maximum_factor_bytes.")
    indices = np.zeros((decomposition.partition_count, block_size), dtype=np.int32)
    valid = np.zeros_like(indices, dtype=bool)
    for part in range(decomposition.partition_count):
        sites = np.asarray(decomposition.owned_global_ids[part])[
            np.asarray(decomposition.owned_valid[part])
        ]
        coordinates = (
            sites[:, None] * dofs_per_site + np.arange(dofs_per_site)[None, :]
        ).reshape((-1,))
        indices[part, : coordinates.size] = coordinates
        valid[part, : coordinates.size] = True
    index_array = jnp.asarray(indices)
    valid_array = jnp.asarray(valid)
    blocks = matrix[index_array[:, :, None], index_array[:, None, :]]
    active_matrix = valid_array[:, :, None] & valid_array[:, None, :]
    blocks = jnp.where(active_matrix, blocks, 0)
    blocks = blocks + jnp.where(
        ~valid_array[:, :, None] & jnp.eye(block_size, dtype=bool)[None, :, :],
        jnp.ones((), dtype=matrix.dtype),
        jnp.zeros((), dtype=matrix.dtype),
    )
    factorization = prepare_local_block_factorization(blocks)
    partition_coordinates = np.asarray(
        tuple(np.ndindex(decomposition.partition_shape)), dtype=np.int32
    )
    block_parity = np.mod(np.sum(partition_coordinates, axis=1), 2).astype(np.int32)
    finite = jnp.all(jnp.isfinite(blocks)) & ~jnp.any(factorization.failed_blocks)
    return PreparedSAP(
        plan,
        matrix,
        index_array,
        valid_array,
        jnp.asarray(block_parity),
        factorization,
        finite,
        canonical_fingerprint(
            {
                "kind": "prepared-lattice-sap",
                "plan": plan.plan_id,
                "matrix_signature": array_tree_fingerprint(np.asarray(matrix)),
                "block_indices": indices,
            }
        ),
    )


class AdaptiveCoarseSpacePlan(StrictModule, NonTrainableState):
    deflation: DeflationPlan
    smoothing_steps: int = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        deflation: DeflationPlan,
        /,
        *,
        smoothing_steps: int = 4,
        relaxation: float = 0.1,
    ) -> None:
        if not isinstance(deflation, DeflationPlan):
            raise TypeError("deflation must be DeflationPlan.")
        steps = int(smoothing_steps)
        relaxation_ = float(relaxation)
        if steps <= 0 or not np.isfinite(relaxation_) or relaxation_ <= 0.0:
            raise ValueError("Adaptive coarse smoothing controls must be positive.")
        self.deflation = deflation
        self.smoothing_steps = steps
        self.relaxation = relaxation_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-lattice-coarse-space-plan",
                "deflation": deflation.plan_id,
                "smoothing_steps": steps,
                "relaxation": relaxation_,
            }
        )


class PreparedAdaptiveCoarseSpace(StrictModule, NonTrainableState):
    plan: AdaptiveCoarseSpacePlan
    deflation: PreparedDeflation
    smoothed_candidates: Array
    initial_normal_residual: Array
    final_normal_residual: Array
    improved: Array
    prepared_id: str = eqx.field(static=True)


def prepare_adaptive_coarse_space(
    plan: AdaptiveCoarseSpacePlan,
    operator: AbstractLatticeDiracOperator,
    candidates: ArrayLike,
    /,
    *,
    sap: PreparedSAP | None = None,
) -> PreparedAdaptiveCoarseSpace:
    if not isinstance(plan, AdaptiveCoarseSpacePlan):
        raise TypeError("plan must be AdaptiveCoarseSpacePlan.")
    if not isinstance(operator, AbstractLatticeDiracOperator):
        raise TypeError("operator must implement AbstractLatticeDiracOperator.")
    if sap is not None and not isinstance(sap, PreparedSAP):
        raise TypeError("sap must be PreparedSAP or None.")
    values = jnp.asarray(candidates)
    expected = (
        operator.site_count,
        operator.spin_components,
        operator.color_components,
        plan.deflation.mode_capacity,
    )
    if values.shape != expected:
        raise ValueError(f"candidates must have shape {expected}.")

    def normal_action(block: Array) -> Array:
        images = jax.vmap(operator.mv, in_axes=-1, out_axes=-1)(block)
        return jax.vmap(operator.adjoint_mv, in_axes=-1, out_axes=-1)(images)

    initial_normal = normal_action(values)
    initial_residual = jnp.sqrt(jnp.sum(jnp.abs(initial_normal) ** 2, axis=(0, 1, 2)))
    smoothed = values
    for _ in range(plan.smoothing_steps):
        normal = normal_action(smoothed)
        if sap is None:
            scale = jnp.maximum(
                jnp.max(jnp.sqrt(jnp.sum(jnp.abs(normal) ** 2, axis=(0, 1, 2)))),
                jnp.finfo(normal.real.dtype).tiny,
            )
            correction = plan.relaxation * normal / scale
        else:
            flat = normal.reshape((operator.source.size, plan.deflation.mode_capacity))
            correction = sap.apply(flat).reshape(expected)
        smoothed = smoothed - correction
    final_normal = normal_action(smoothed)
    final_residual = jnp.sqrt(jnp.sum(jnp.abs(final_normal) ** 2, axis=(0, 1, 2)))
    deflation = prepare_deflation(plan.deflation, operator, smoothed)
    improved = jnp.all(final_residual <= initial_residual) & deflation.finite
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-adaptive-lattice-coarse-space",
            "plan": plan.plan_id,
            "operator": operator.operator_id,
            "sap": None if sap is None else sap.prepared_id,
            "candidate_signature": array_tree_fingerprint(np.asarray(values)),
        }
    )
    return PreparedAdaptiveCoarseSpace(
        plan,
        deflation,
        smoothed,
        initial_residual,
        final_residual,
        improved,
        prepared_id,
    )


class DistributedHMCPlan(StrictModule, NonTrainableState):
    step_size: float = eqx.field(static=True)
    leapfrog_steps: int = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    maximum_links: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        step_size: float,
        leapfrog_steps: int,
        divergence_threshold: float = 1000.0,
        maximum_links: int = 67_108_864,
    ) -> None:
        step = float(step_size)
        steps = int(leapfrog_steps)
        threshold = float(divergence_threshold)
        maximum = int(maximum_links)
        if (
            not np.isfinite(step)
            or step <= 0.0
            or steps <= 0
            or not np.isfinite(threshold)
            or threshold <= 0.0
            or maximum <= 0
        ):
            raise ValueError("Distributed HMC controls must be finite and positive.")
        self.step_size = step
        self.leapfrog_steps = steps
        self.divergence_threshold = threshold
        self.maximum_links = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-lattice-hmc-plan",
                "step_size": step,
                "leapfrog_steps": steps,
                "divergence_threshold": threshold,
                "maximum_links": maximum,
            }
        )


class DistributedHMCTransitionResult(StrictModule):
    links: Array
    action: Array
    accepted: Array
    acceptance_probability: Array
    energy_error: Array
    divergent: Array
    nonfinite: Array
    membership_failure: Array
    leapfrog_steps: Array
    step_index: Array
    plan_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)


class PreparedDistributedHMC(StrictModule, NonTrainableState):
    plan: DistributedHMCPlan
    gauge: PreparedDistributedGaugeTheory
    kernel: PreparedCompactGroupHamiltonianKernel
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def transition(
        self,
        links: ArrayLike,
        /,
        *,
        key: Key[Array, ""],
        step_index: int | Array = 0,
    ) -> DistributedHMCTransitionResult:
        values = self.gauge._configuration(links)
        state = initialize_compact_group_hamiltonian_state(self.kernel, values[None, ...])
        state = eqx.tree_at(
            lambda node: node.step_index,
            state,
            jnp.asarray(step_index, dtype=jnp.uint32),
        )
        sampled = sample_compact_group_hamiltonian(
            self.kernel,
            state,
            key=key,
            num_draws=1,
        )
        return DistributedHMCTransitionResult(
            sampled.samples[0, 0],
            -sampled.log_target[0, 0],
            sampled.accepted[0, 0],
            sampled.acceptance_probability[0, 0],
            sampled.energy_error[0, 0],
            sampled.divergent[0, 0],
            sampled.nonfinite[0, 0],
            sampled.membership_failure[0, 0],
            sampled.leapfrog_steps[0, 0],
            sampled.final_state.step_index,
            self.plan.plan_id,
            self.kernel.target_id,
        )


def prepare_distributed_hmc(
    plan: DistributedHMCPlan,
    gauge: PreparedDistributedGaugeTheory,
    link_template: ArrayLike,
    /,
    *,
    additional_action: Callable[[Array], Array] | None = None,
    term_id: str = "gauge-only",
) -> PreparedDistributedHMC:
    if not isinstance(plan, DistributedHMCPlan):
        raise TypeError("plan must be DistributedHMCPlan.")
    if not isinstance(gauge, PreparedDistributedGaugeTheory):
        raise TypeError("gauge must be PreparedDistributedGaugeTheory.")
    links = gauge._configuration(link_template)
    canonical = gauge._links(links)
    if canonical.shape[0] * canonical.shape[1] > plan.maximum_links:
        raise ValueError("HMC link count exceeds maximum_links.")
    identifier = str(term_id).strip()
    if not identifier:
        raise ValueError("term_id must be non-empty.")
    group = SpecialUnitaryGroup(links.shape[-1])
    point_geometry = LieGroupStateGeometry(group)
    geometry = PointwiseStateGeometry(
        point_geometry,
        group.point_shape,
        local_shape=group.algebra_shape,
        tangent_shape=group.point_shape,
    )
    if additional_action is None:
        evaluate = lambda value: -gauge.gauge_action(value).value
    else:
        if not callable(additional_action):
            raise TypeError("additional_action must be callable or None.")
        evaluate = lambda value: (
            -(gauge.gauge_action(value).value + jnp.asarray(additional_action(value)))
        )
    target_id = canonical_fingerprint(
        {
            "kind": "distributed-lattice-hmc-target",
            "plan": plan.plan_id,
            "gauge": gauge.prepared_id,
            "term_id": identifier,
            "link_shape": links.shape,
            "dtype": links.dtype.name,
        }
    )
    target = CompactGeometricTarget(
        evaluate,
        geometry,
        configuration_shape=links.shape,
        local_coordinate_shape=links.shape[:-2] + group.algebra_shape,
        reference_measure="product-haar",
        target_id=target_id,
    )
    kernel = prepare_compact_group_hamiltonian_kernel(
        target,
        step_size=plan.step_size,
        leapfrog_steps=plan.leapfrog_steps,
        divergence_threshold=plan.divergence_threshold,
    )
    return PreparedDistributedHMC(
        plan,
        gauge,
        kernel,
        identifier,
        canonical_fingerprint(
            {
                "kind": "prepared-distributed-lattice-hmc",
                "plan": plan.plan_id,
                "gauge": gauge.prepared_id,
                "kernel": kernel.kernel_id,
                "term_id": identifier,
            }
        ),
    )


class DistributedRHMCPlan(StrictModule, NonTrainableState):
    """Bind a native RHMC trajectory plan to one distributed gauge ownership."""

    gauge_plan_id: str = eqx.field(static=True)
    rhmc: RHMCPlan = eqx.field(static=True)
    maximum_links: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gauge: DistributedGaugeTheoryPlan,
        rhmc: RHMCPlan,
        /,
        *,
        maximum_links: int = 67_108_864,
    ) -> None:
        if not isinstance(gauge, DistributedGaugeTheoryPlan):
            raise TypeError("gauge must be DistributedGaugeTheoryPlan.")
        if not isinstance(rhmc, RHMCPlan):
            raise TypeError("rhmc must be the native sampling RHMCPlan.")
        maximum = int(maximum_links)
        if maximum <= 0:
            raise ValueError("maximum_links must be positive.")
        self.gauge_plan_id = gauge.plan_id
        self.rhmc = rhmc
        self.maximum_links = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-lattice-rhmc-composition-plan",
                "gauge": gauge.plan_id,
                "rhmc": rhmc.plan_id,
                "maximum_links": maximum,
            }
        )


class PreparedDistributedRHMC(StrictModule, NonTrainableState):
    """Native RHMC kernel whose bosonic term uses exactly-owned gauge reduction."""

    plan: DistributedRHMCPlan
    gauge: PreparedDistributedGaugeTheory
    kernel: PreparedRHMCKernel
    prepared_id: str = eqx.field(static=True)

    def initialize(
        self,
        links: ArrayLike,
        /,
        *,
        key: Key[Array, ""],
    ) -> RHMCChainState:
        values = self.gauge._configuration(links)
        return initialize_rhmc_state(self.kernel, values, key=key)

    def transition(
        self,
        state: RHMCChainState,
        /,
    ) -> RHMCTransitionResult:
        if not isinstance(state, RHMCChainState):
            raise TypeError("state must be RHMCChainState.")
        return rhmc_transition(self.kernel, state)

    def sample(
        self,
        state: RHMCChainState,
        /,
        *,
        num_draws: int,
    ) -> RHMCSampleResult:
        if not isinstance(state, RHMCChainState):
            raise TypeError("state must be RHMCChainState.")
        return sample_rhmc(self.kernel, state, num_draws=num_draws)


def prepare_distributed_rhmc(
    plan: DistributedRHMCPlan,
    gauge: PreparedDistributedGaugeTheory,
    link_template: ArrayLike,
    /,
    *,
    pseudofermion_terms: Sequence[PseudofermionTerm] = (),
    additional_action_terms: Sequence[SeparableActionTerm] = (),
) -> PreparedDistributedRHMC:
    """Compose distributed gauge reduction with native pseudofermion RHMC."""

    if not isinstance(plan, DistributedRHMCPlan):
        raise TypeError("plan must be DistributedRHMCPlan.")
    if not isinstance(gauge, PreparedDistributedGaugeTheory):
        raise TypeError("gauge must be PreparedDistributedGaugeTheory.")
    if gauge.plan.plan_id != plan.gauge_plan_id:
        raise ValueError("Distributed RHMC and prepared gauge plans disagree.")
    links = gauge._configuration(link_template)
    canonical = gauge._links(links)
    if canonical.shape[0] * canonical.shape[1] > plan.maximum_links:
        raise ValueError("RHMC link count exceeds maximum_links.")
    extra = tuple(additional_action_terms)
    if any(not isinstance(term, SeparableActionTerm) for term in extra):
        raise TypeError(
            "additional_action_terms must contain SeparableActionTerm values."
        )
    gauge_term = SeparableActionTerm(
        lambda value: gauge.gauge_action(value).value,
        term_id=f"{gauge.plan.plan_id}:distributed-wilson-gauge-action",
    )
    registry = SeparableActionRegistry(
        (gauge_term,) + extra,
        tuple(pseudofermion_terms),
    )
    group = SpecialUnitaryGroup(canonical.shape[-1])
    geometry = PointwiseStateGeometry(
        LieGroupStateGeometry(group),
        group.point_shape,
        local_shape=group.algebra_shape,
        tangent_shape=group.point_shape,
    )
    kernel = prepare_rhmc(
        registry,
        plan.rhmc,
        links,
        geometry=geometry,
        local_coordinate_shape=links.shape[:-2] + group.algebra_shape,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-distributed-lattice-rhmc-composition",
            "plan": plan.plan_id,
            "gauge": gauge.prepared_id,
            "registry": registry.registry_id,
            "kernel": kernel.kernel_id,
        }
    )
    return PreparedDistributedRHMC(plan, gauge, kernel, prepared_id)


__all__ = [
    "AdaptiveCoarseSpacePlan",
    "DeflationPlan",
    "DistributedDslashResult",
    "DistributedGaugeActionResult",
    "DistributedGaugeForceResult",
    "DistributedGaugeTheoryPlan",
    "DistributedHMCPlan",
    "DistributedHMCTransitionResult",
    "DistributedRHMCPlan",
    "PreparedAdaptiveCoarseSpace",
    "PreparedDeflation",
    "PreparedDistributedGaugeTheory",
    "PreparedDistributedHMC",
    "PreparedDistributedRHMC",
    "PreparedSAP",
    "SAPPlan",
    "prepare_adaptive_coarse_space",
    "prepare_deflation",
    "prepare_distributed_hmc",
    "prepare_distributed_rhmc",
    "prepare_sap",
]
