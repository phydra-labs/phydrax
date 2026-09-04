# Liu–Brown–Yue 2002 fatigue and recovery

`LiuBrownYue2002Plan` implements the macroscopic model in Liu, Brown &
Yue, “A Dynamical Model of Muscle Activation, Fatigue, and Recovery,”
*Biophysical Journal* 82 (2002), 2344–2359,
[doi:10.1016/S0006-3495(02)75580-X](https://doi.org/10.1016/S0006-3495(02)75580-X).
The citation supports the three motor-unit groups and phenomenological rates
below. It does not support later target-load controllers or enhanced
intermittent-rest recovery multipliers.

The model partitions a conserved total $M_0$ into uncommitted $M_{uc}$,
active $M_A$, and fatigued $M_F$ fractions. Equations 1a–1b are

$$
\dot M_A=B M_{uc}-F M_A+R M_F,\qquad
\dot M_F=F M_A-R M_F,
$$

with $M_{uc}=M_0-M_A-M_F$. Brain effort $B$, fatigue rate $F$, and recovery
rate $R$ all have units $\mathrm{s}^{-1}$. State time and step duration use
seconds; compartment values are dimensionless fractions. The observable
active relative force is $M_A/M_0$, and available nonfatigued capacity is
$(M_{uc}+M_A)/M_0$.

The prepared runtime evaluates the exact solution for piecewise-constant $B$
over each step. It uses the stable repeated-rate limit when $B=F+R$, preserves
the compartment total, and rejects nonfinite, negative, or nonconservative
candidates. `commit_liu_brown_yue_2002` atomically retains either every
proposed compartment and time or the complete source state.

This fidelity is a standalone terminal macroscopic route. It does not modify,
wrap, attenuate, or add recovery to D1. The numerical values $F=0.0206\,
\mathrm{s}^{-1}$ and $R\approx0.0084\,\mathrm{s}^{-1}$ used in the example are
cohort-level sustained maximal handgrip fits reported by Liu et al.; they are
not universal defaults, so the API requires both explicitly.

## Intermittent-task source gate

No intermittent-task-specific fidelity is released here. The Liu–Brown–Yue
paper supplies the model above and analyzes constant brain effort, but it does
not supply the later enhanced-rest multiplier and a complete intermittent-task
parameter/data protocol. Adding such a multiplier under the 2002 name would
misattribute a later model. Use the source model directly for an explicit
piecewise-constant $B(t)$ study, or add a separately source-named fidelity only
when its primary equations, parameter protocol, and validation data are all
available.

Run the standalone example and qualification surface with:

```console
python examples/skeletal_fatigue_liu_2002.py
python tools/skeletal_fatigue_qualification.py
```
