# Rheology

The [granular rheology guide](../guides_mu_i_granular_rheology.md) specifies SI
units, the physical-pressure domain, zero-rate status, exact Barker–Gray equations,
and diagnostic applicability.

`MuIParameters` holds scalar dynamic JAX leaves, so material coefficients remain
visible to `jit`, `vmap`, and first-order differentiation. `MuILaw` owns only
static curve structure; always pass parameters explicitly to `evaluate`.
`MuILawEvaluation` reports stress/friction separately from singular apparent
viscosity and exposes separation and neutral-root evidence.

`GeneralizedNewtonianLaw.viscosity` returns `GeneralizedNewtonianEvaluation`;
consume `viscosity_pa_s` with its domain/status fields. The old array-only return
and artificial shear-rate floor have been removed. Newtonian viscosity is the
existing power-law selector with `power_index=1`.

::: phydrax.rheology.MuIRegularizationKind

::: phydrax.rheology.MuIParameters

::: phydrax.rheology.MuILaw

::: phydrax.rheology.MuILawEvaluation

::: phydrax.rheology.MuIPackingFractionLaw

::: phydrax.rheology.GeneralizedNewtonianLaw

::: phydrax.rheology.GeneralizedNewtonianEvaluation
