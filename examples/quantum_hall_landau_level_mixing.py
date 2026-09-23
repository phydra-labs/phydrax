import jax.random as jr
import numpy as np

from phydrax.applications import quantum_hall as qh
from phydrax.operators.quantum import evaluate_local_operator
from phydrax.units import ELECTRONVOLT


sphere = qh.HaldaneSpherePlan(2, qh.MonopoleLandauLevel(3, 0, qh.SPIN_POLARIZED_ELECTRON), "fermion", qh.QuantumHallEnergyScale(
    ELECTRONVOLT,
    1.602_176_634e-19,
    "electronvolt",
), )
prepared = qh.prepare_landau_level_mixing_vmc(
    qh.LandauLevelMixingVMCPlan(
        sphere,
        0.5,
        chain_count=4,
        hidden_dimension=8,
        layer_count=1,
        determinant_count=2,
    ),
    jr.key(7),
)
local = evaluate_local_operator(
    prepared.model,
    prepared.operator,
    prepared.problem.initial_configurations,
)
print(
    {
        "local_energy": np.asarray(local.value).tolist(),
        "successful": np.asarray(local.successful).tolist(),
        "problem_id": prepared.problem.problem_id,
    }
)
