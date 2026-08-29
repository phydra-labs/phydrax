# Conditions

Conditions declare scientific semantics: named fields, support, operators, and
targets. They do not sample, integrate, weight, scalarize, or choose soft versus
exact treatment.

The numerical layers are separate:

1. [`phydrax.conditions`](core.md) declares the requirement.
2. [`phydrax.integration`](../integration.md) owns measures, plans, and
   realizations.
3. [`phydrax.terms`](../terms.md) produces scalar penalties and signed terms.
4. [`phydrax.enforcement`](../enforcement.md) compiles selected conditions into
   exact field transforms.

Physical catalogs cover [boundary and initial](boundary.md), [CFD](cfd.md),
[solid mechanics](solid.md), [thermal physics](thermal.md),
[electromagnetics](em.md), [stochastic physics](stochastic.md), and
[conservation moments](conservation.md).

See [Conditions, integration, terms, and enforcement](../../guides_conditions.md)
for complete soft and exact examples.

## Free-boundary and interface conditions

Implicit-interface conditions use the negative-inside level-set convention and
remain ordinary `Residual` values. Pair them with
`phydrax.terms.implicit_interface_penalty` to apply the coarea surface measure;
the condition itself never chooses a diffuse width or integration plan.

::: phydrax.conditions.free_boundary.InterfaceValueJump

::: phydrax.conditions.free_boundary.InterfaceFluxJump

::: phydrax.conditions.free_boundary.InterfaceKinematic

::: phydrax.conditions.free_boundary.StefanBalance

::: phydrax.conditions.free_boundary.YoungLaplaceJump

::: phydrax.conditions.free_boundary.GibbsThomson

::: phydrax.conditions.free_boundary.InterfaceTractionJump

::: phydrax.conditions.free_boundary.LevelSetEikonal
