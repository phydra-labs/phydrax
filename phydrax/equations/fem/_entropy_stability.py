#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.fem._reference import FiniteElementSpec
from ...discretization.finite_volume._riemann import (
    AbstractSymmetricTwoPointFluxPlan,
)
from .._entropy_pair import ConvexEntropyPair


EntropyDGFormulation = Literal[
    "tensor_sbp",
    "generalized_sbp",
    "skew_modal",
]


class PhysicalBoundaryEntropyContract(StrictModule, NonTrainableState):
    boundary_id: str = eqx.field(static=True)
    supply: Any = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(self, boundary_id: str, supply: Any, /):
        identifier = str(boundary_id)
        if not identifier or not callable(supply):
            raise ValueError("Boundary entropy contracts require ID and supply callable.")
        self.boundary_id = identifier
        self.supply = supply
        self.contract_id = canonical_fingerprint(
            {
                "kind": "physical-boundary-entropy-contract",
                "boundary": identifier,
            }
        )

    @classmethod
    def adiabatic_wall(cls, boundary_id: str, /) -> "PhysicalBoundaryEntropyContract":
        return cls(
            boundary_id,
            lambda time, state, coordinates, normal, numerical, entropy_pair, args: (
                jnp.zeros(numerical.shape, dtype=numerical.dtype)
            ),
        )

    @classmethod
    def transparent(cls, boundary_id: str, /) -> "PhysicalBoundaryEntropyContract":
        return cls(
            boundary_id,
            lambda time, state, coordinates, normal, numerical, entropy_pair, args: (
                numerical
            ),
        )

    def allowed_supply(
        self,
        time: Array,
        state: Array,
        coordinates: Array,
        normal: Array,
        numerical_entropy_flux: Array,
        entropy_pair: ConvexEntropyPair,
        args: Any,
        /,
    ) -> Array:
        value = jnp.asarray(
            self.supply(
                time,
                state,
                coordinates,
                normal,
                numerical_entropy_flux,
                entropy_pair,
                args,
            )
        )
        if value.shape != numerical_entropy_flux.shape:
            raise ValueError("Boundary entropy supply shape is incompatible.")
        return value


class EntropyStableDGPlan(StrictModule, NonTrainableState):
    volume_flux: Any
    entropy_pair: ConvexEntropyPair
    boundary_contracts: tuple[PhysicalBoundaryEntropyContract, ...]
    formulation: EntropyDGFormulation = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_flux: Any,
        entropy_pair: ConvexEntropyPair,
        /,
        *,
        formulation: EntropyDGFormulation = "generalized_sbp",
        tolerance: float = 1.0e-10,
        boundary_contracts: Sequence[PhysicalBoundaryEntropyContract] = (),
    ):
        formulation_ = str(formulation)
        tolerance_ = float(tolerance)
        if (
            formulation_ not in ("tensor_sbp", "generalized_sbp", "skew_modal")
            or not isinstance(entropy_pair, ConvexEntropyPair)
            or not isinstance(volume_flux, AbstractSymmetricTwoPointFluxPlan)
            or not math.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Entropy-stable DG plan inputs are invalid.")
        contracts = tuple(boundary_contracts)
        if any(
            not isinstance(value, PhysicalBoundaryEntropyContract) for value in contracts
        ) or len({value.boundary_id for value in contracts}) != len(contracts):
            raise ValueError("Boundary entropy contracts must be typed and unique.")
        self.volume_flux = volume_flux
        self.entropy_pair = entropy_pair
        self.boundary_contracts = contracts
        self.formulation = formulation_
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "entropy-stable-dg-plan",
                "volume_flux": volume_flux.flux_id,
                "entropy_pair": entropy_pair.entropy_id,
                "formulation": formulation_,
                "tolerance": tolerance_,
                "boundary_contracts": tuple(value.contract_id for value in contracts),
            }
        )

    def boundary_contract(self, boundary_id: str, /) -> PhysicalBoundaryEntropyContract:
        identifier = str(boundary_id)
        for contract in self.boundary_contracts:
            if contract.boundary_id == identifier:
                return contract
        raise ValueError(
            f"Missing entropy contract for physical boundary {identifier!r}."
        )


class EntropyReferenceOperator(StrictModule, NonTrainableState):
    mass_matrix: Array
    mass_inverse: Array
    weak_derivatives: Array
    boundary_matrices: Array
    sbp_defect: Array
    constant_defect: Array
    minimum_mass_eigenvalue: Array
    formal_sbp: bool = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def flux_differencing_dual(
        self,
        system: Any,
        state: ArrayLike,
        contravariant_cofactors: ArrayLike,
        plan: EntropyStableDGPlan,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        cofactors = jnp.asarray(contravariant_cofactors)
        if value.ndim < 3:
            raise ValueError("Flux differencing state requires cell and node axes.")
        left = value[:, :, None, :]
        right = value[:, None, :, :]
        result = jnp.zeros_like(value)
        for direction in range(self.weak_derivatives.shape[0]):
            physical_flux = jnp.stack(
                tuple(
                    plan.volume_flux.two_point_flux(
                        system,
                        left,
                        right,
                        axis,
                        args,
                    )
                    for axis in range(system.dimension)
                ),
                axis=-1,
            )
            metric_pair = 0.5 * (
                cofactors[:, :, None, direction, :] + cofactors[:, None, :, direction, :]
            )
            contravariant = ein.contract(
                "cijvd,cijd->cijv",
                physical_flux,
                metric_pair,
                backend="jax",
            )
            result = result + 2.0 * ein.contract(
                "ij,cijv->civ",
                self.weak_derivatives[direction],
                contravariant,
                backend="jax",
            )
        return result


class EntropyMortarEvidence(StrictModule, NonTrainableState):
    entropy_production: Array
    conservative_defect: Array
    compatible: Array
    evidence_id: str = eqx.field(static=True)


class BoundaryEntropyEvidence(StrictModule, NonTrainableState):
    numerical_entropy_flux: Array
    allowed_entropy_supply: Array
    defect: Array
    compatible: Array
    evidence_id: str = eqx.field(static=True)


def prepare_entropy_reference_operator(
    element: FiniteElementSpec,
    volume_rule: Any,
    facet_rules: Sequence[Any],
    /,
    *,
    tolerance: float = 1.0e-10,
) -> EntropyReferenceOperator:
    from ...integration._rules import reference_rule_data

    if not isinstance(element, FiniteElementSpec):
        raise TypeError("element must be FiniteElementSpec.")
    volume = reference_rule_data(volume_rule)
    values, gradients = element.tabulate(volume.points)
    mass = ein.contract("q,qi,qj->ij", volume.weights, values, values, backend="jax")
    weak = jnp.stack(
        tuple(
            ein.contract(
                "q,qi,qj->ij",
                volume.weights,
                values,
                gradients[..., direction],
                backend="jax",
            )
            for direction in range(element.topological_dimension)
        )
    )
    from ...discretization.fem._reference_operator import (
        _map_edge_rule,
        _map_face_rule,
    )

    boundary = jnp.zeros_like(weak)
    for facet, rule in enumerate(facet_rules):
        data = reference_rule_data(rule)
        points, weights, normals = (
            _map_edge_rule(element.cell_kind, facet, data)
            if element.topological_dimension == 2
            else _map_face_rule(element.cell_kind, facet, data)
        )
        trace = element.tabulate(points)[0]
        boundary = boundary + jnp.stack(
            tuple(
                ein.contract(
                    "q,q,qi,qj->ij",
                    weights,
                    normals[..., direction],
                    trace,
                    trace,
                    backend="jax",
                )
                for direction in range(element.topological_dimension)
            )
        )
    defect = weak + jnp.swapaxes(weak, -1, -2) - boundary
    constant = jnp.ones((element.local_dof_count,), dtype=mass.dtype)
    constant_defect = jnp.max(
        jnp.abs(ein.contract("dij,j->di", weak, constant, backend="jax"))
    )
    mass_host = np.asarray(mass)
    eigenvalues = np.linalg.eigvalsh(mass_host)
    minimum = float(np.min(eigenvalues))
    mass_inverse = np.linalg.solve(mass_host, np.eye(mass_host.shape[0]))
    maximum_defect = float(np.max(np.abs(np.asarray(defect))))
    formal = bool(
        minimum > 0.0
        and maximum_defect <= tolerance
        and float(np.asarray(constant_defect)) <= tolerance
    )
    operator_id = canonical_fingerprint(
        {
            "kind": "entropy-reference-operator",
            "element": element.element_id,
            "volume_rule": canonical_fingerprint(
                {
                    "cell": volume.cell,
                    "points": array_tree_fingerprint(np.asarray(volume.points)),
                    "weights": array_tree_fingerprint(np.asarray(volume.weights)),
                }
            ),
            "facet_rules": tuple(
                canonical_fingerprint(
                    {
                        "cell": reference_rule_data(rule).cell,
                        "points": array_tree_fingerprint(
                            np.asarray(reference_rule_data(rule).points)
                        ),
                        "weights": array_tree_fingerprint(
                            np.asarray(reference_rule_data(rule).weights)
                        ),
                    }
                )
                for rule in facet_rules
            ),
            "mass": array_tree_fingerprint(mass_host),
            "maximum_sbp_defect": maximum_defect,
            "formal_sbp": formal,
        }
    )
    return EntropyReferenceOperator(
        mass,
        jnp.asarray(mass_inverse),
        weak,
        boundary,
        jnp.asarray(maximum_defect),
        constant_defect,
        jnp.asarray(minimum),
        formal,
        operator_id,
    )


def entropy_mortar_evidence(
    entropy_pair: ConvexEntropyPair,
    left_state: ArrayLike,
    right_state: ArrayLike,
    normal_flux: ArrayLike,
    normal: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> EntropyMortarEvidence:
    left = jnp.asarray(left_state)
    right = jnp.asarray(right_state)
    flux = jnp.asarray(normal_flux)
    normal_ = jnp.asarray(normal)
    left_variables = entropy_pair.entropy_variables(left)
    right_variables = entropy_pair.entropy_variables(right)
    potential_jump = sum(
        normal_[..., direction]
        * (
            entropy_pair.entropy_potential(right, direction)
            - entropy_pair.entropy_potential(left, direction)
        )
        for direction in range(normal_.shape[-1])
    )
    production = (
        ein.contract(
            "...v,...v->...", right_variables - left_variables, flux, backend="jax"
        )
        - potential_jump
    )
    conservative_defect = jnp.max(jnp.abs(production))
    compatible = jnp.all(production <= tolerance)
    evidence_id = canonical_fingerprint(
        {
            "kind": "entropy-mortar-evidence",
            "entropy_pair": entropy_pair.entropy_id,
            "tolerance": float(tolerance),
        }
    )
    return EntropyMortarEvidence(
        production,
        conservative_defect,
        compatible,
        evidence_id,
    )


def boundary_entropy_evidence(
    entropy_pair: ConvexEntropyPair,
    interior_state: ArrayLike,
    normal_flux: ArrayLike,
    normal: ArrayLike,
    allowed_entropy_supply: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> BoundaryEntropyEvidence:
    state = jnp.asarray(interior_state)
    flux = jnp.asarray(normal_flux)
    normal_ = jnp.asarray(normal)
    variables = entropy_pair.entropy_variables(state)
    potential = sum(
        normal_[..., direction] * entropy_pair.entropy_potential(state, direction)
        for direction in range(normal_.shape[-1])
    )
    numerical = ein.contract("...v,...v->...", variables, flux, backend="jax") - potential
    supply = jnp.asarray(allowed_entropy_supply)
    defect = numerical - supply
    compatible = jnp.all(defect <= tolerance)
    evidence_id = canonical_fingerprint(
        {
            "kind": "boundary-entropy-evidence",
            "entropy_pair": entropy_pair.entropy_id,
            "tolerance": float(tolerance),
        }
    )
    return BoundaryEntropyEvidence(
        numerical,
        supply,
        defect,
        compatible,
        evidence_id,
    )


__all__ = [
    "BoundaryEntropyEvidence",
    "EntropyDGFormulation",
    "EntropyMortarEvidence",
    "EntropyReferenceOperator",
    "EntropyStableDGPlan",
    "PhysicalBoundaryEntropyContract",
    "boundary_entropy_evidence",
    "entropy_mortar_evidence",
    "prepare_entropy_reference_operator",
]
