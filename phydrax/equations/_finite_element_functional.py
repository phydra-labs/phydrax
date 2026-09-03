#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._numerics._compensated import compensated_sum
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._local_variational import AbstractPreparedLocalDiscretization
from ..discretization.fem import FiniteElementDiscretization, IntegrationDomain
from ._finite_element_variational import (
    _default_rule,
    _reference_rule_data,
    FiniteElementExecutionContext,
)
from ._variational import (
    _normalize_rules,
    _rule_id,
    IntegrationRule as ReferenceRule,
)


class FiniteElementFunctional(StrictModule, NonTrainableState):
    """One representation-bound scalar functional on a prepared local field."""

    functional_id: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    density: Callable[[Array, Array, Array, object], ArrayLike]
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, ReferenceRule], ...]

    def __init__(
        self,
        functional_id: str,
        field_name: str,
        density: Callable[[Array, Array, Array, object], ArrayLike],
        /,
        *,
        domain: IntegrationDomain | None = None,
        rules: Mapping[str, ReferenceRule] | Sequence[tuple[str, ReferenceRule]] = (),
    ):
        identifier = str(functional_id)
        field = str(field_name)
        if not identifier or not field or not callable(density):
            raise ValueError("Functional ID, field, and callable density are required.")
        if domain is not None and not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be IntegrationDomain or None.")
        if domain is not None and domain.kind != "cell":
            raise ValueError("FiniteElementFunctional currently requires a cell domain.")
        normalized_rules = _normalize_rules(rules)
        self.functional_id = canonical_fingerprint(
            {
                "kind": "finite-element-functional",
                "declared_id": identifier,
                "field_name": field,
                "domain": None if domain is None else domain.domain_id,
                "rules": [
                    [block_name, _rule_id(rule)] for block_name, rule in normalized_rules
                ],
            }
        )
        self.field_name = field
        self.density = density
        self.domain = domain
        self.rules = normalized_rules

    def evaluate(
        self,
        discretization: AbstractPreparedLocalDiscretization,
        state: ArrayLike,
        args: object = None,
        /,
    ) -> Array:
        field_index = discretization._field_index(self.field_name)
        values = discretization.field_spaces[field_index].vector_space.validate(state)
        binding = discretization.local_field_binding(self.field_name)
        execution_values = binding.flatten(values)
        context = (
            args
            if isinstance(args, FiniteElementExecutionContext)
            else FiniteElementExecutionContext(
                discretization.default_runtime,
                user_args=args,
            )
        )
        domain = (
            discretization.integration_domain("cell")
            if self.domain is None
            else self.domain
        )
        if domain.support_id != discretization.support.support_id:
            raise ValueError("Functional domain belongs to another support.")
        discretization.validate_local_runtime(context.runtime)
        if not self.rules or not isinstance(discretization, FiniteElementDiscretization):
            mode = (
                "dense"
                if isinstance(discretization, FiniteElementDiscretization)
                else "sum_factorized"
            )
            regions = discretization.prepare_local_regions(
                domain,
                field_names=(self.field_name,),
                maximum_derivative_order=1,
                kernel_mode=mode,
            )
            contributions = []
            for region in regions:
                reference = region.reference_actions[0].realize_reference_actions(
                    context.runtime
                )
                metric = region.geometry_actions.realize(context.runtime)
                local = execution_values[region.field_gathers[0]]
                field_values = reference.interpolate(context.runtime, local)
                reference_gradient = reference.reference_gradient(context.runtime, local)
                gradients = jnp.moveaxis(
                    metric.physical_gradient(reference_gradient), -1, 2
                )
                density = jnp.asarray(
                    self.density(field_values, gradients, metric.points, context)
                )
                if density.shape != metric.physical_weights.shape:
                    raise ValueError(
                        "Finite-element functional density must return one scalar "
                        "per local point."
                    )
                valid = jnp.asarray(region.valid) & jnp.asarray(metric.valid)
                contributions.append(
                    discretization.precision_policy.accumulation(
                        density * metric.physical_weights * valid[:, None]
                    ).reshape((-1,))
                )
            combined = jnp.concatenate(tuple(contributions))
            if discretization.precision_policy.compensated_accumulation:
                return discretization.precision_policy.output(compensated_sum(combined))
            return discretization.precision_policy.output(jnp.sum(combined))
        rules = dict(self.rules)
        contributions = []
        cell_offset = 0
        for block_index, (block, dofs) in enumerate(
            zip(
                discretization.mesh.blocks,
                discretization.dof_maps[field_index].cell_dofs,
                strict=True,
            )
        ):
            block_cells = jnp.arange(
                cell_offset,
                cell_offset + block.cell_count,
                dtype=jnp.int32,
            )
            cell_offset += block.cell_count
            selected = jnp.isin(block_cells, domain.entity_indices)
            rule = rules.get(block.name, _default_rule(block.cell_kind))
            data = _reference_rule_data(rule)
            if data.cell != block.cell_kind:
                raise ValueError("Functional rule does not match its cell block.")
            geometry = discretization.evaluate_block_geometry(
                self.field_name,
                block_index,
                context.runtime.coordinates,
                data.points,
                data.weights,
            )
            local = values[dofs]
            field_values = ein.contract(
                "qi,ci...->cq...",
                geometry.basis_values,
                local,
            )
            gradients = ein.contract(
                "cqid,ci...->cqd...",
                geometry.physical_gradients,
                local,
            )
            density = jnp.asarray(
                self.density(
                    field_values,
                    gradients,
                    geometry.physical_points,
                    context,
                )
            )
            expected = geometry.physical_weights.shape
            if density.shape != expected:
                raise ValueError(
                    "Finite-element functional density must return one scalar "
                    "per selected quadrature point."
                )
            contributions.append(
                discretization.precision_policy.accumulation(
                    density * geometry.physical_weights * selected[:, None]
                ).reshape((-1,))
            )
        if not contributions:
            return discretization.precision_policy.output(jnp.asarray(0.0))
        combined = jnp.concatenate(tuple(contributions))
        if discretization.precision_policy.compensated_accumulation:
            return discretization.precision_policy.output(compensated_sum(combined))
        return discretization.precision_policy.output(jnp.sum(combined))


__all__ = ["FiniteElementFunctional"]
