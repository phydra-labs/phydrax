#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static, martingale, and transport-based financial arbitrage diagnostics."""

from ._martingale import (
    bind_option_marginals_to_martingale_bridge,
    bind_option_marginals_to_martingale_transport,
    FinanceMartingaleBridgeProblem,
    FinanceMartingaleBridgeResult,
    FinanceMartingaleTransportProblem,
    FinanceMartingaleTransportResult,
    IndependentBridgeValidation,
    NumeraireOptionMarginal,
    option_marginal_convex_order,
    semi_static_hedge_dual,
    SemiStaticHedgeDual,
    solve_finance_martingale_bridge,
    solve_finance_martingale_transport,
    validate_martingale_bridge_paths,
)
from ._static import (
    CalendarArbitrageEvidence,
    evaluate_calendar_arbitrage,
    evaluate_static_option_arbitrage,
    OptionCallSlice,
    StaticOptionArbitrageEvidence,
)


__all__ = [
    "CalendarArbitrageEvidence",
    "FinanceMartingaleBridgeProblem",
    "FinanceMartingaleBridgeResult",
    "FinanceMartingaleTransportProblem",
    "FinanceMartingaleTransportResult",
    "IndependentBridgeValidation",
    "NumeraireOptionMarginal",
    "OptionCallSlice",
    "SemiStaticHedgeDual",
    "StaticOptionArbitrageEvidence",
    "bind_option_marginals_to_martingale_bridge",
    "bind_option_marginals_to_martingale_transport",
    "evaluate_calendar_arbitrage",
    "evaluate_static_option_arbitrage",
    "option_marginal_convex_order",
    "semi_static_hedge_dual",
    "solve_finance_martingale_bridge",
    "solve_finance_martingale_transport",
    "validate_martingale_bridge_paths",
]
