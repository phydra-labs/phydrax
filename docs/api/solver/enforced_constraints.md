# Enforced constraint pipelines

Pipelines compose and apply enforced-constraint ansätze.

For the underlying ansatz constructors (`enforce_dirichlet`, `enforce_neumann`, etc.), see
[Enforced constraint ansätze](../constraints/enforced.md).

!!! note
    Key notes:

    - An `EnforcedConstraintPipeline` stages overlays in the order boundary → initial/generic overlays → interior data.
    - `EnforcedConstraintPipelines` topologically orders multi-field dependencies (`co_vars`).
    - Graph value overlays built with `enforce_graph_values` are generic overlays and can be passed through `constraint_terms`.
    - For a detailed mathematical treatment of the PCI pipeline (including BVH-weighted boundary blending, boundary–initial gating, and the interior anchor/data stage), see [Appendix → Physics-Constrained Interpolation](../../appendix/physics_constrained_interpolation.md).
    - Geometry factors used in boundary/initial and interior-data preservation
      gates are dimensionless. `gate_method="auto"` selects the global CAD
      R-equivalence gate. Select `"compact"` on `EnforcedConstraintPipeline`,
      `EnforcedConstraintPipelines.build`, or `FunctionalSolver` to use the compact
      fallback; `gate_saturation_fraction` and `gate_linear_fraction` then configure
      its extent. Configure each Dirichlet boundary term on `enforce_dirichlet`.
      None of these settings alter the boundary-defining field used by derivative
      constraints.

::: phydrax.solver.SingleFieldEnforcedConstraint
    options:
        members:
            - __init__
            - co_vars

---

::: phydrax.solver.MultiFieldEnforcedConstraint
    options:
        members:
            - __init__

---

::: phydrax.solver.EnforcedInteriorData
    options:
        members:
            - __init__

---

::: phydrax.solver.EnforcedConstraintPipeline
    options:
        members:
            - __init__
            - apply

---

::: phydrax.solver.EnforcedConstraintPipelines
    options:
        members:
            - __init__
            - build
            - apply
