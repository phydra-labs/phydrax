#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider support findings before scientific preparation."""

from typing import final, Literal

import equinox as eqx

from ..._strict import StrictModule
from ...typing import parse


type SupportCheck = Literal[
    "geometry",
    "cell-families",
    "fill-strategy",
    "boundary-scope",
    "size-control",
    "region-identity",
    "region-scope",
    "disabled-region",
    "region-seed",
    "region-semantics",
    "duplicate-patch",
    "patch-dimension",
    "patch-adjacency",
    "exterior-patch",
    "exterior-regions",
    "patch-scope",
    "feature-scope",
    "feature-identity",
    "periodic-recovery",
    "layer-route",
]


@final
class SupportIssue(StrictModule):
    """A stable support check and its consumer-facing unmet requirement."""

    code: SupportCheck = eqx.field(static=True)
    detail: str = eqx.field(static=True)

    def __init__(self, code: SupportCheck, detail: str) -> None:
        code = parse(code, SupportCheck, "SupportIssue.code")
        self.code, self.detail = code, detail


def format_support_issues(issues: tuple[SupportIssue, ...], /) -> str:
    """Format support details in their validation order."""
    return "; ".join(issue.detail for issue in issues)
