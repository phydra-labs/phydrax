#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from phydrax.domain import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    DomainFunction,
)


@dataclass(frozen=True, slots=True)
class DerivativeRequest:
    """One derivative of a named residual field requested by an operator."""

    field: str
    variable: str
    axes: tuple[int | None, ...]
    laplacian_count: int = 0

    @property
    def contracted_laplacian(self) -> bool:
        return self.laplacian_count > 0

    @property
    def order(self) -> int:
        return len(self.axes) + 2 * self.laplacian_count


class _RequestRecorderRule(DerivativeRule):
    def __init__(
        self,
        *,
        source: DomainFunction,
        field: str,
        requests: list[DerivativeRequest],
        prefix: tuple[int | None, ...] = (),
        prefix_laplacians: int = 0,
    ):
        self.source = source
        self.field = field
        self.requests = requests
        self.prefix = prefix
        self.prefix_laplacians = int(prefix_laplacians)

    def _result(
        self,
        *,
        prefix: tuple[int | None, ...],
        prefix_laplacians: int,
    ) -> DomainFunction:
        return DomainFunction(
            domain=self.source.domain,
            deps=self.source.deps,
            func=self.source.func,
            metadata=self.source.metadata,
            derivative_rule=_RequestRecorderRule(
                source=self.source,
                field=self.field,
                requests=self.requests,
                prefix=prefix,
                prefix_laplacians=prefix_laplacians,
            ),
        )

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, backend, basis, periodic
        axes = self.prefix + (axis,) * int(order)
        request = DerivativeRequest(
            field=self.field,
            variable=var,
            axes=axes,
            laplacian_count=self.prefix_laplacians,
        )
        self.requests.append(request)
        return self._result(
            prefix=axes,
            prefix_laplacians=self.prefix_laplacians,
        )

    def derive_laplacian(
        self,
        *,
        var: str,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, backend, basis, periodic
        request = DerivativeRequest(
            field=self.field,
            variable=var,
            axes=self.prefix,
            laplacian_count=self.prefix_laplacians + 1,
        )
        self.requests.append(request)
        return self._result(
            prefix=self.prefix,
            prefix_laplacians=self.prefix_laplacians + 1,
        )


def trace_derivative_requests(
    residual: Callable[[Mapping[str, DomainFunction]], DomainFunction],
    functions: Mapping[str, DomainFunction],
    /,
) -> tuple[DerivativeRequest, ...]:
    """Trace first/second derivative requirements without evaluating a batch."""

    recorded: list[DerivativeRequest] = []
    traced = {
        name: function.with_derivative_rule(
            _RequestRecorderRule(
                source=function,
                field=name,
                requests=recorded,
            )
        )
        for name, function in functions.items()
    }
    result = residual(traced)
    if not isinstance(result, DomainFunction):
        raise TypeError("A ResidualPenalty condition must return a DomainFunction.")

    unique: list[DerivativeRequest] = []
    seen: set[DerivativeRequest] = set()
    for request in recorded:
        if request.order > 2:
            raise ValueError(
                "KFAC supports residual derivatives through order two; "
                f"field {request.field!r} requested order {request.order}."
            )
        if request not in seen:
            seen.add(request)
            unique.append(request)
    return tuple(unique)


__all__ = ["DerivativeRequest", "trace_derivative_requests"]
