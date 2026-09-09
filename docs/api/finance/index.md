# Financial mathematics API

`phydrax.finance` is divided by semantic ownership rather than by a universal pricing interface. Contracts define obligations, market snapshots own observed data, model objects describe dynamics without silently choosing a measure, and each computational route owns its definition → resolution → plan → preparation → result → replay lifecycle.

| Domain | Ownership |
| --- | --- |
| `core` | identifiers, currency, exact money, dates, calendars, laws, scenarios, evidence binding |
| `market` | lineage, reference data, quotes, fixings, risk-factor layouts, causal snapshots |
| `contracts` | contract resolution, cashflows, exercise, settlement, trades, positions |
| `curves` | typed curves, interpolation, bootstrapping, calibration residuals |
| `models` | measure-neutral rates and credit model structures |
| `calibration` | objectives, constraints, prepared calibration, evidence |
| `valuation` | analytic, Fourier, PDE, simulation, exercise, and BSDE routes |
| `econometrics` | historical datasets, estimators, filters, diagnostics, replay |
| `portfolio` | holdings, objectives, constraints, compilation, optimization |
| `risk` | sensitivities, scenarios, measures, allocation, backtesting |
| `exposure` | netting, collateral, exposure profiles, CVA/DVA/FVA/MVA/KVA |
| `execution` | order flow, impact, market making, controls, execution evidence |
| `arbitrage` | static-arbitrage diagnostics and martingale transport candidates |
| `qualification` | exact support tuples, evidence matrices, campaigns, archives |
| `interchange` | deterministic records/tables and the explicit FpML subset |

The package does not provide a universal `price()` function. Select a route whose input and output contracts match the mathematical claim being made.
