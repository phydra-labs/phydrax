"""Full-line harmonic oscillator with rational Chebyshev resolution evidence."""

import jax.numpy as jnp

import phydrax as phx


def solve(mode_count: int):
    domain = phx.discretization.AxisDomain.real_line()
    boundary = phx.discretization.SpectralBoundaryConditionPlan.decay()
    basis = phx.discretization.ConstrainedBasisPlan(
        phx.discretization.RationalChebyshevLineBasisPlan(mode_count, 4.0),
        boundary,
    )
    space = phx.discretization.TensorSpectralPlan(
        (basis,),
        axis_names=("x",),
        field_name="wavefunction",
    ).prepare((domain,))
    second_derivative = phx.discretization.spectral_derivative_operator(
        space,
        0,
        2,
    ).operator
    coordinates = space.axes[0].nodes
    potential = phx.linalg.FunctionLinearOperator(
        lambda coefficients: space.project(
            coordinates**2 * space.reconstruct(coefficients)
        ),
        source=space.modal_space.vector_space,
        target=space.modal_space.vector_space,
        operator_id=f"harmonic-potential:{space.prepared_id}",
    )
    hamiltonian = -second_derivative + potential
    result = phx.linalg.eigen.general_eigensolve(
        phx.linalg.eigen.GeneralEigenproblem(hamiltonian)
    )
    return space, result


def main() -> None:
    coarse_space, coarse = solve(20)
    fine_space, fine = solve(28)
    transfer = phx.discretization.prepare_spectral_modal_transfer(
        coarse_space,
        fine_space,
    )
    evidence = phx.discretization.compare_spectral_eigen_resolutions(
        coarse,
        fine,
        coarse_space,
        fine_space,
        transfer,
        policy=phx.discretization.SpectralEigenResolutionPolicy(
            phx.linalg.eigen.GeneralEigenResolutionPolicy(
                chordal_tolerance=1e-5,
                normalized_drift_tolerance=0.1,
            ),
            subspace_tolerance=1e-2,
        ),
    )
    finite = coarse.eigenvalues[coarse.finite_mask]
    ordered = jnp.sort(jnp.real(finite))
    print("lowest oscillator eigenvalues:", ordered[:3])
    print("trusted coarse modes at 1% subspace tolerance:", evidence.trusted_count)
    print(
        "maximum original residual:", jnp.max(coarse.diagnostics.right_relative_residuals)
    )


if __name__ == "__main__":
    main()
