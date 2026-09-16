import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import polymer_field_theory as pft, polymer_liquids as pl


def test_particle_form_factor_lowers_into_prism_and_observation_product():
    layout = phx.atomistic.PolymerChainLayoutPlan(
        [[0, 1], [2, 3]],
        [[True, True], [True, True]],
        maximum_frames=1,
    )
    positions = jnp.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]]
    )
    transform = pl.IsotropicRadialTransformPlan(4, np.pi / 0.2).prepare()
    wave = transform.wave_numbers
    form_factor = pl.trajectory_intramolecular_form_factor(
        wave,
        positions,
        layout,
        [0, 0, 0, 0],
        1,
        source_id="particle-trajectory",
    )
    prepared = pl.PRISMPlan(
        pl.PRISMClosurePlan(pl.PRISMClosureKind.HNC), maximum_iterations=8
    ).prepare(
        transform,
        pl.SiteMixturePlan(("A",), [0.1]),
        form_factor,
        pl.SitePairPotentialPlan(
            transform.radii,
            jnp.zeros((1, 1, 4)),
            source_id="ideal",
        ),
    )
    result = pl.solve_prism(prepared)
    vector = pl.prism_structure_theory_vector(
        result,
        ("A",),
        transform.wave_numbers,
        wave_number_unit_id="reduced-inverse-length",
        normalization_id="site-matrix",
    )
    assert result.successful
    assert vector.values.shape == (4,)


def test_particle_and_scft_scattering_adapters_preserve_product_identity():
    positions = jnp.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    debye = phx.atomistic.debye_scattering(
        phx.atomistic.DebyeScatteringPlan(
            [0.0, 1.0], maximum_frames=1, maximum_particles=2
        ),
        positions,
    )
    debye_vector = pl.debye_theory_vector(
        debye,
        wave_number_unit_id="reduced-inverse-length",
        intensity_normalization_id="sum-b-squared",
    )

    architecture = pft.PolymerContourArchitecturePlan(
        "AB",
        (
            pft.ContourBlockPlan("A", "left", "middle", 0, 0.5, 2),
            pft.ContourBlockPlan("B", "middle", "right", 1, 0.5, 2),
        ),
        root_node="left",
    )
    model = pft.IncompressibleGaussianMixturePlan(
        ("A", "B"),
        [1.0, 1.0],
        [[0.0, 0.0], [0.0, 0.0]],
        (pft.PolymerComponentPlan("AB", architecture, 1.0, 10.0),),
    )
    spectral = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(4),), axis_names=("x",)
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 4.0),))
    prepared = pft.SCFTPlan(model).prepare(spectral)
    evaluation = prepared.evaluate(jnp.zeros(prepared.field_shape))
    scft_vector = pl.scft_density_scattering_theory_vector(
        evaluation,
        spectral,
        [1.0, -1.0],
        reciprocal_unit_id="reduced-inverse-length",
        normalization_id="orthonormal-fourier",
    )

    assert debye_vector.product_id != scft_vector.product_id
    assert debye_vector.layout.size == 2
    assert scft_vector.layout.size == spectral.num_modes
