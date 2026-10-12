from typing import assert_type

from jax import Array

from phydrax.rheology import (
    GeneralizedNewtonianEvaluation,
    GeneralizedNewtonianLaw,
    MuILaw,
    MuILawEvaluation,
    MuIPackingFractionLaw,
    MuIParameters,
)


parameters = MuIParameters(0.383, 0.643, 0.279, 0.001, 2500.0)
law = MuILaw(packing_fraction_law=MuIPackingFractionLaw(0.6, 0.2))
assert_type(law.evaluate(20.0, 100.0, parameters), MuILawEvaluation)
assert_type(parameters.mu_s, Array)
assert_type(law.evaluate(20.0, 100.0, parameters).viscosity_defined, Array)
assert_type(law.evaluate(20.0, 100.0, parameters).packing_fraction, Array | None)
assert_type(
    GeneralizedNewtonianLaw("power-law", 1.0).viscosity(0.0),
    GeneralizedNewtonianEvaluation,
)
MuILaw(regularization="bingham-type")  # ty: ignore[invalid-argument-type]
MuIParameters(
    0.383, 0.643, 0.279, 0.001, 2500.0, barker_alpha=1.9, barker_mu_infinity=0.05
)
