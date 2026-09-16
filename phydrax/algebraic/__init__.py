#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse polynomial systems, exact grading evidence, and root lowering."""

from ._certification import (
    ExactRealRootInterval,
    isolate_univariate_real_roots,
    krawczyk_certificate,
    KrawczykCertificate,
    smale_alpha_certificate,
    SmaleAlphaCertificate,
)
from ._exact import (
    EliminateArguments,
    ExactCoefficientDomain,
    ExactPolynomialResult,
    ExactSparsePolynomialSystem,
    ExactSymbolicArguments,
    ExactSymbolicEvidence,
    ExactSymbolicOperation,
    ExactSymbolicPlan,
    ExactSymbolicResult,
    ExactSymbolicStatus,
    execute_exact_symbolic,
    GF,
    GroebnerBasisArguments,
    NormalFormArguments,
    plan_exact_symbolic,
    prepare_exact_symbolic,
    PreparedExactSymbolic,
    QQ,
    UnivariateDiscriminantArguments,
    UnivariateResultantArguments,
    ZZ,
)
from ._grading import (
    BezoutForecastKind,
    multihomogeneous_bezout_forecast,
    multihomogeneous_degrees,
    polynomial_degree_profile,
    PolynomialBezoutForecast,
    PolynomialDegreeProfile,
    PolynomialVariableGeometry,
    PolynomialVariableGroup,
    total_degree_bezout_forecast,
    total_degrees,
)
from ._isolated import *  # noqa: F403
from ._isolated import __all__ as _isolated_all
from ._nonlinear import (
    ComplexPolynomialRootLowering,
    lower_complex_polynomial_root,
)
from ._positive_dimensional import *  # noqa: F403
from ._positive_dimensional import __all__ as _positive_dimensional_all
from ._quotient import *  # noqa: F403
from ._quotient import __all__ as _quotient_all
from ._symmetry import (
    analyze_exponent_lattice_scaling,
    ExponentLatticeScalingEvidence,
)
from ._system import PolynomialScaling, SparsePolynomialSupport, SparsePolynomialSystem


__all__ = [
    "ExactRealRootInterval",
    "isolate_univariate_real_roots",
    "KrawczykCertificate",
    "krawczyk_certificate",
    "SmaleAlphaCertificate",
    "smale_alpha_certificate",
    "analyze_exponent_lattice_scaling",
    "BezoutForecastKind",
    "ComplexPolynomialRootLowering",
    "EliminateArguments",
    "ExactCoefficientDomain",
    "ExactPolynomialResult",
    "ExactSparsePolynomialSystem",
    "ExactSymbolicArguments",
    "ExactSymbolicEvidence",
    "ExactSymbolicOperation",
    "ExactSymbolicPlan",
    "ExactSymbolicResult",
    "ExactSymbolicStatus",
    "execute_exact_symbolic",
    "ExponentLatticeScalingEvidence",
    "GF",
    "GroebnerBasisArguments",
    "lower_complex_polynomial_root",
    "multihomogeneous_bezout_forecast",
    "multihomogeneous_degrees",
    "polynomial_degree_profile",
    "NormalFormArguments",
    "plan_exact_symbolic",
    "prepare_exact_symbolic",
    "PreparedExactSymbolic",
    "PolynomialBezoutForecast",
    "PolynomialDegreeProfile",
    "PolynomialScaling",
    "PolynomialVariableGeometry",
    "PolynomialVariableGroup",
    "QQ",
    "SparsePolynomialSupport",
    "SparsePolynomialSystem",
    "UnivariateDiscriminantArguments",
    "UnivariateResultantArguments",
    "ZZ",
    "total_degree_bezout_forecast",
    "total_degrees",
]
__all__ += [
    name
    for name in (*_positive_dimensional_all, *_quotient_all, *_isolated_all)
    if name not in __all__
]
