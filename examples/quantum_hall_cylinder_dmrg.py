import numpy as np

from phydrax.applications import quantum_hall as qh


prepared = qh.prepare_hall_cylinder_hamiltonian(
    qh.HallCylinderPlan(
        3,
        2,
        8.0,
        {1: 1.0},
        maximum_orbital_separation=2,
    )
)
result = qh.solve_hall_cylinder_dmrg(
    qh.HallCylinderDMRGPlan(
        prepared,
        (1, 0, 1),
        maximum_sweeps=1,
        maximum_bond_dimension=8,
        descent_step=0.02,
        residual_tolerance=1.0e-5,
    )
)
print(
    {
        "energy": float(np.asarray(result.evidence.energies[-1])),
        "residual": float(np.asarray(result.evidence.residual_norms[-1])),
        "charge_drift": float(np.asarray(result.evidence.charge_drifts[-1])),
        "successful": bool(np.asarray(result.successful)),
    }
)
