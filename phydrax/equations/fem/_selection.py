#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...discretization import IntegrationDomain
from ...discretization._local_variational import (
    AbstractPreparedLocalDiscretization,
    LocalVariationalRequest,
    LocalVariationalSelection,
)


def _jet_kinds(operators: tuple[tuple[str, str], ...], /) -> tuple[str, ...]:
    kinds = []
    for _, operator in operators:
        if operator == "value":
            kind = "value"
        elif operator in ("grad", "sym-grad", "div", "curl"):
            kind = "gradient"
        elif operator in (
            "normal-trace",
            "tangential-trace",
            "jump",
            "average",
        ):
            kind = "trace"
        else:
            kind = operator
        if kind not in kinds:
            kinds.append(kind)
    return tuple(kinds)


def _required_semantics(jet_kinds: tuple[str, ...], /) -> tuple[str, ...]:
    semantics = []
    if "value" in jet_kinds:
        semantics.append("exact_interpolation_transpose")
    if "gradient" in jet_kinds:
        semantics.append("exact_gradient_transpose")
    if "trace" in jet_kinds:
        semantics.append("exact_trace_transpose")
    return tuple(semantics)


def select_local_execution(
    discretization: AbstractPreparedLocalDiscretization,
    domain: IntegrationDomain,
    provider_action_kind: str,
    operators: tuple[str, ...],
    /,
    *,
    requested_kernel_mode: str,
    requested_operator_realization: str,
    explicit_rules: bool = False,
    provider_offers: tuple[str, ...] = ("prepared-local",),
) -> LocalVariationalSelection:
    """Select one declared local execution without method type gates."""
    if not isinstance(discretization, AbstractPreparedLocalDiscretization):
        raise TypeError("discretization must be AbstractPreparedLocalDiscretization.")
    if not isinstance(domain, IntegrationDomain):
        raise TypeError("domain must be IntegrationDomain.")
    operators_ = tuple(str(operator) for operator in operators)
    jets = _jet_kinds(tuple(("", operator) for operator in operators_))
    request = LocalVariationalRequest(
        str(provider_action_kind),
        domain.kind,
        operators_,
        jets,
        requested_kernel_mode=str(requested_kernel_mode),
        requested_operator_realization=str(requested_operator_realization),
        action_semantics=_required_semantics(jets),
        constraint_mode="external_map",
        material_mode="none",
        history_mode="none",
        explicit_rules=bool(explicit_rules),
    )
    selection = discretization.local_variational_capabilities().select(request)
    if selection.execution_kind not in provider_offers:
        raise ValueError(
            "Selected local execution is not admitted by the variational action."
        )
    return selection


def select_prepared_local_execution(
    action,
    discretization: AbstractPreparedLocalDiscretization,
    domain: IntegrationDomain,
    /,
    *,
    requested_kernel_mode: str,
    requested_operator_realization: str,
) -> LocalVariationalSelection:
    descriptor = action.descriptor
    return select_local_execution(
        discretization,
        domain,
        descriptor.provider_action_kind,
        tuple(operator for _, operator in descriptor.operators),
        requested_kernel_mode=requested_kernel_mode,
        requested_operator_realization=requested_operator_realization,
        explicit_rules=bool(action.rules),
        provider_offers=descriptor.provider_offers,
    )


__all__ = ["select_local_execution", "select_prepared_local_execution"]
