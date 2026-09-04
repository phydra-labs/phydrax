# De Groote–Fregly 2016 musculotendon

`phydrax.applications.skeletal_muscle.musculotendon` provides two concrete, separately named compliant-tendon runtimes. Both consume `independent_excitation`; neither consumes D1 force, common D1 drive, firing rates, or a Shorten state. The returned tendon force is the only force owned by this route and must not be multiplied by another force law.

## Scientific boundary

The source is De Groote, Kinney, Rao, and Fregly, *Annals of Biomedical Engineering* 44 (2016), 2922–2936, DOI [10.1007/s10439-016-1591-9](https://doi.org/10.1007/s10439-016-1591-9). The [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC5043004/) supplies activation dynamics (Eqs. 1–2), the Hill-type model and pennation geometry (Eqs. 3–7), and the four formulation definitions (Eqs. 12–17). The publisher’s [online supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs10439-016-1591-9/MediaObjects/10439_2016_1591_MOESM1_ESM.pdf) supplies the curve equations and Table 1 coefficients (S1–S4), explicit normalized tendon-force dynamics (S5–S19), and the implicit tendon-force path constraint (S24–S28).

`DeGrooteFregly2016Plan` implements explicit formulation 1: activation and normalized tendon force are states. `DeGrooteFregly2016ImplicitTendonForcePlan` implements formulation 3’s algebraic equation. The paper uses the scaled tendon-force rate as an optimization control; Phydrax additionally solves that algebraic equation with `phydrax.nonlinear.implicit_root_result` when a rate is requested. This root-solving policy and its implicit JVP/VJP are a Phydrax runtime construction, not a claim made by the paper.

The supplement’s smooth passive and tendon curves are not clipped. In particular, the source discusses the tendon curve’s mathematically negative slack extension; runtime states are nevertheless admitted only over the paper’s formulation-comparison bounds: activation `[0.01, 1]`, normalized tendon force `[0, 3]`, and normalized fiber length `[0.4, 1.6]`.

## Units and signs

| API quantity | Canonical quantity | Unit | Sign / axis |
|---|---|---|---|
| `excitation` | `independent_excitation` | 1 | `[0, 1]`, muscle axis |
| `musculotendon_length_m` | `musculotendon_length` | m | non-negative route length, muscle axis |
| `musculotendon_velocity_m_per_s` | `musculotendon_velocity` | m/s | positive lengthening, muscle axis |
| `activation` | `muscle_activation` | 1 | `[0.01, 1]`, muscle axis |
| `normalized_tendon_force` | `normalized_tendon_force` | 1 | positive tensile, muscle axis |
| `tendon_force_N` | `tendon_force` | N | positive tensile along the route, muscle axis |
| `fiber_length_m` | `muscle_fiber_length` | m | non-negative, muscle axis |
| `fiber_velocity_m_per_s` | `muscle_fiber_velocity` | m/s | positive lengthening, muscle axis |
| `pennation_angle_rad` | `pennation_angle` | rad | fiber-to-tendon angle, muscle axis |

`maximum_isometric_force_N` is physical peak isometric muscle force. Therefore `tendon_force_N = maximum_isometric_force_N * normalized_tendon_force`; no later physical-force scaling is permitted.

## Transaction and evidence

Numeric parameter fields and body-independent curve coefficients are JAX leaves. Muscle names, the fixed capacity, mask, formulation ID, and model ID are static. `prepare()` binds the state shape and dtype. `candidate()` uses a documented forward-Euler state update and retains both the complete source and proposed state. `commit()` accepts the whole proposal only when every enabled muscle passes; otherwise it returns the whole untouched source state.

Each constitutive evaluation reports:

- force equilibrium and both pennation/length closure residuals;
- tendon and passive-fiber energies, their rates, and force–velocity power residuals;
- series-path power balance `F_T v_MT = F_T v_T + F_M v_M`;
- finite, state-bound, geometry, residual, energy, and final success masks.

The energy functions are Phydrax antiderivatives of source Eqs. S1 and S3, referenced to normalized length 1. Their derivatives reproduce the source force curves; the paper does not assign these reference energy values.

## Minimal use

The following is API-shape pseudocode; values and imports are intentionally omitted. For executable code, run [`examples/skeletal_musculotendon_de_groote_fregly_2016.py`](https://github.com/phydra-labs/phydrax/blob/dev/examples/skeletal_musculotendon_de_groote_fregly_2016.py).

```text
parameters = DeGrooteFregly2016Parameters(
    maximum_isometric_force_N,
    optimal_fiber_length_m,
    tendon_slack_length_m,
    pennation_angle_at_optimum_rad,
    maximum_fiber_velocity_m_per_s,
)
model = DeGrooteFregly2016Plan(parameters, muscle_names).prepare(initial_state)
evaluation = model.evaluate(
    state,
    independent_excitation,
    musculotendon_length,
    musculotendon_velocity,
)
candidate = model.candidate(
    state,
    independent_excitation,
    musculotendon_length,
    musculotendon_velocity,
    time_step_s,
)
state = model.commit(candidate)
```

Run [`tools/qualify_de_groote_fregly_2016.py`](https://github.com/phydra-labs/phydrax/blob/dev/tools/qualify_de_groote_fregly_2016.py) for an independent Table 1 and equation audit.
