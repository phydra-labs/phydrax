# Physical relative-force calibration

`PhysicalRelativeForceCalibrationPlan` is an observation and personalization
model. It is not a quantity conversion. Its declared measurement equation is

$$
y_i[\mathrm N]
=s[\mathrm{N/relative\ force}]\,f_i[1]
+\sum_j Z_{ij}[1]\,\beta_j[\mathrm N]+\epsilon_i[\mathrm N].
$$

The explicit positive scale $s$ maps one specified relative-force route to
newtons under one named protocol and calibrated asset. Every nuisance column
has a static name; common examples are a load-cell zero offset or a
protocol-defined drift basis. The plan records immutable `protocol_id` and
`asset_id` values in its identity, state, evidence, and physical observations.

The separation of measurand, input quantities, and influence/nuisance
quantities follows JCGM 100:2008, clauses 4.1–4.2,
[*Evaluation of measurement data — Guide to the expression of uncertainty in
measurement*](https://www.bipm.org/en/committees/jc/jcgm/publications). That
reference supports the explicit measurement-model treatment, not this
skeletal-muscle design matrix or an identity between relative-force
fidelities.

## Identifiability and transaction

Evaluation uses Phydrax’s diagnosed weighted least-squares owner with weights
$1/u_i^2$, where each standard uncertainty $u_i$ is supplied in newtons.
Evidence includes:

- valid sample mask and count;
- singular values, design rank, and condition number;
- scale information remaining after weighted projection on every nuisance;
- nuisance-confounding fraction;
- residual and scale standard uncertainty;
- explicit `SCALE_NOT_IDENTIFIABLE`, `RANK_DEFICIENT`, and other failure bits.

A nuisance column proportional to relative force makes scale and nuisance gain
indistinguishable and must fail closed. A negative fitted physical scale,
insufficient samples, nonpositive uncertainty, solver failure, or excessive
condition number also rejects the candidate. Commit then preserves the entire
previous calibration. Successful commit advances one calibration epoch.

`observe` applies only the committed physical scale. Fitted nuisance
coefficients explain calibration observations and are deliberately not added
to subsequent physical force predictions. Calibrating one source route does
not authorize mixing it with D1 or any other terminal force owner.

Run the example and positive/negative-control qualification surface with:

```console
python examples/skeletal_force_calibration.py
python tools/skeletal_force_calibration_qualification.py
```
