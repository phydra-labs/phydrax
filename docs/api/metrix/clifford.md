# Clifford algebra and fields

The Clifford API uses explicit immutable algebra and blade-layout metadata. Products are
prepared before traced evaluation and retain independent closure/resource evidence.
Differential-form, metric-isometry, equivariance, and Dirac trial-space evidence are
separate contracts.

::: phydrax.metrix.clifford.CliffordAlgebraSpec

::: phydrax.metrix.clifford.CliffordResourceBudget

::: phydrax.metrix.clifford.CliffordResourceEvidence

::: phydrax.metrix.clifford.CliffordBladeLayout

::: phydrax.metrix.clifford.CliffordProductPlan

::: phydrax.metrix.clifford.CliffordProductEvidence

::: phydrax.metrix.clifford.prepare_product

::: phydrax.metrix.clifford.basis_blade_product

::: phydrax.metrix.clifford.basis_blade

::: phydrax.metrix.clifford.embed_layout

::: phydrax.metrix.clifford.extract_layout

::: phydrax.metrix.clifford.project_grades

::: phydrax.metrix.clifford.grade_involution

::: phydrax.metrix.clifford.reverse

::: phydrax.metrix.clifford.clifford_conjugate

::: phydrax.metrix.clifford.scalar_part

::: phydrax.metrix.clifford.CliffordMetricBridge

::: phydrax.metrix.clifford.MetricIsometryAction

::: phydrax.metrix.clifford.MetricIsometryAuditSet

::: phydrax.metrix.clifford.FiniteMetricIsometryGroup

::: phydrax.metrix.clifford.lorentz_boost_action

::: phydrax.metrix.clifford.CliffordOutermorphismPlan

::: phydrax.metrix.clifford.CliffordActionAuditReport

::: phydrax.metrix.clifford.audit_clifford_action

::: phydrax.metrix.clifford.audit_clifford_actions

## Dirac and monogenic fields

::: phydrax.operators.clifford_dirac

::: phydrax.equations.MonogenicPolynomialBasis

::: phydrax.equations.LinearMonogenicField

## Neural representations and layers

::: phydrax.nn.operator.representations.CliffordGradeRepresentation

::: phydrax.nn.operator.representations.CliffordGradeFeatures

::: phydrax.nn.operator.layers.CliffordGradeLinear

::: phydrax.nn.operator.layers.CliffordGeometricProductLayer

::: phydrax.nn.operator.layers.clifford_gated_activation

::: phydrax.nn.operator.layers.CliffordEquivarianceCertificate

::: phydrax.nn.operator.layers.CliffordEquivarianceAuditReport

::: phydrax.nn.operator.layers.audit_clifford_equivariance
