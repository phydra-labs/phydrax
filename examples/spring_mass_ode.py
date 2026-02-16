#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import marimo


__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="Spring-Mass ODE in Phydrax")


@app.cell
def _():
    import time as time_mod

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import optax

    import phydrax as phx

    plt.style.use("seaborn-v0_8-whitegrid")
    return eqx, jax, jnp, jr, mo, optax, phx, plt, time_mod


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Coupled Spring-Mass ODE in Phydrax

    This notebook recreates the NVIDIA PhysicsNeMo spring-mass ODE benchmark:
    [PhysicsNeMo Spring-Mass Example](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo-sym/user_guide/foundational/ode_spring_mass.html).

    Matrix form:

    $$
    \ddot{x}(t) + Kx(t)=0,\quad
    x(t)=\begin{bmatrix}x_1(t)\\x_2(t)\\x_3(t)\end{bmatrix},\quad
    K=\begin{bmatrix}
    3 & -1 & 0\\
    -1 & 2 & -1\\
    0 & -1 & 3
    \end{bmatrix}.
    $$

    We solve the 3-DOF system on \(t\in[0,10]\):

    $$
    \begin{aligned}
    \ddot{x}_1 + 3x_1 - x_2 &= 0, \\
    \ddot{x}_2 + 2x_2 - x_1 - x_3 &= 0, \\
    \ddot{x}_3 + 3x_3 - x_2 &= 0.
    \end{aligned}
    $$

    Internally, training is performed on normalized time
    \(s=(t-t_{\min})/(t_{\max}-t_{\min})\in[0,1]\), with the residual scaled so
    the learned solution matches the same physical system in \(t\).

    Initial conditions:

    $$
    x_1(0)=1,\ x_2(0)=0,\ x_3(0)=0,\quad
    \dot{x}_1(0)=\dot{x}_2(0)=\dot{x}_3(0)=0.
    $$

    We enforce all six initial conditions exactly by construction, then train with
    a single matrix-form residual loss.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            ## Why this setup is strong

            1. ICs are enforced directly in the ansatz (no IC penalty balancing).
            2. Training objective focuses on residual physics only.
            3. Time derivatives use AD with JVP (`dt_n(..., order=2, ad_engine="jvp")`).
            """
        ),
        kind="success",
    )
    return


@app.cell
def _():
    # -------------------------------------------------------------------------
    # Configurations
    # -------------------------------------------------------------------------
    width_size = 20
    depth = 6
    num_iter = 500
    learning_rate = 1e-3
    num_t_interior = 10_000
    nt_plot = 2000
    seed = 0
    t_min = 0.0
    t_max = 10.0

    # Published PhysicsNeMo spring-mass baseline specs
    physicsnemo_steps = 50_000
    physicsnemo_interior_batch = 500
    physicsnemo_params = 1_315_843
    return (
        depth,
        learning_rate,
        nt_plot,
        num_iter,
        num_t_interior,
        physicsnemo_interior_batch,
        physicsnemo_params,
        physicsnemo_steps,
        seed,
        t_max,
        t_min,
        width_size,
    )


@app.cell
def _(eqx, jax, jnp, phx):
    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------
    def exact_states(t: jax.Array) -> jax.Array:
        c1 = jnp.cos(t)
        c3 = jnp.cos(jnp.sqrt(3.0) * t)
        c2 = jnp.cos(2.0 * t)
        x1 = (1.0 / 6.0) * c1 + 0.5 * c3 + (1.0 / 3.0) * c2
        x2 = (1.0 / 3.0) * c1 - (1.0 / 3.0) * c2
        x3 = (1.0 / 6.0) * c1 - 0.5 * c3 + (1.0 / 3.0) * c2
        return jnp.stack((x1, x2, x3), axis=-1)

    def exact_velocities(t: jax.Array) -> jax.Array:
        s1 = jnp.sin(t)
        s3 = jnp.sin(jnp.sqrt(3.0) * t)
        s2 = jnp.sin(2.0 * t)
        v1 = -(1.0 / 6.0) * s1 - 0.5 * jnp.sqrt(3.0) * s3 - (2.0 / 3.0) * s2
        v2 = -(1.0 / 3.0) * s1 + (2.0 / 3.0) * s2
        v3 = -(1.0 / 6.0) * s1 + 0.5 * jnp.sqrt(3.0) * s3 - (2.0 / 3.0) * s2
        return jnp.stack((v1, v2, v3), axis=-1)

    def total_energy(x: jax.Array, v: jax.Array) -> jax.Array:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        kinetic = 0.5 * jnp.sum(v**2, axis=1)
        potential = (x1**2) + 0.5 * (x2 - x1) ** 2 + 0.5 * (x3 - x2) ** 2 + (x3**2)
        return kinetic + potential

    def count_parameters(module) -> int:
        leaves = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_inexact_array))
        return sum(leaf.size for leaf in leaves)

    def make_solver(
        *,
        t_min: float,
        t_max: float,
        width_size: int,
        depth: int,
        num_t_interior: int,
        key: jax.Array,
    ) -> tuple[phx.solver.FunctionalSolver, int]:
        t_min_phys = t_min
        t_max_phys = t_max
        t_span = t_max_phys - t_min_phys
        if t_span <= 0.0:
            raise ValueError(
                f"Expected positive physical time span, got t_min={t_min_phys}, t_max={t_max_phys}."
            )

        # Train on normalized time s in [0, 1], while preserving physical ODE scaling.
        time_domain = phx.domain.TimeInterval(0.0, 1.0)
        structure_t = phx.domain.ProductStructure((("t",),))
        x_model = phx.nn.SeparableMLP(
            in_size="scalar",
            out_size=3,
            split_input=3,
            width_size=width_size,
            depth=depth,
            activation=phx.nn.Stan(width_size),
            key=key,
        )
        x_raw = time_domain.Model("t")(x_model)

        @time_domain.Function("t")
        def tau(t):
            if isinstance(t, tuple):
                if len(t) != 1:
                    raise ValueError(
                        f"Expected a single scalar-axis tuple input, got {len(t)} axes."
                    )
                t = t[0]
            return t

        @time_domain.Function()
        def x_anchor():
            return jnp.asarray([1.0, 0.0, 0.0], dtype=float)

        # Hard IC ansatz is intentionally used here: it yields better residual
        # conditioning for this ODE than the generic initial-overlay parameterization.
        tau2 = tau * tau
        x = x_anchor + tau2 * x_raw

        k_mat = jnp.asarray(
            [
                [3.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 3.0],
            ],
            dtype=float,
        )

        residual = phx.constraints.ContinuousPointwiseInteriorConstraint(
            "x",
            time_domain,
            operator=lambda x: (
                (1.0 / (t_span * t_span))
                * phx.operators.dt_n(x, var="t", order=2, ad_engine="jvp")
                + phx.operators.einsum("ij,...j->...i", k_mat, x)
            ),
            num_points=num_t_interior,
            structure=structure_t,
            reduction="mean",
            label="ode_matrix",
        )

        solver = phx.solver.FunctionalSolver(
            functions={"x": x},
            constraints=[residual],
        )

        model_params = count_parameters(x_model)
        return solver, model_params

    def evaluate_solver(
        solver: phx.solver.FunctionalSolver,
        *,
        t_min: float,
        t_max: float,
        nt: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        fields = solver.ansatz_functions()
        t_min_phys = t_min
        t_max_phys = t_max
        t_span = t_max_phys - t_min_phys
        if t_span <= 0.0:
            raise ValueError(
                f"Expected positive physical time span, got t_min={t_min_phys}, t_max={t_max_phys}."
            )

        t = jnp.linspace(t_min_phys, t_max_phys, nt)
        s = (t - t_min_phys) / t_span

        x_fun = fields["x"]
        x_pred = jax.vmap(lambda s_i: x_fun.func(s_i))(s)

        v_s_fun = phx.operators.dt(x_fun, var="t", ad_engine="jvp")
        v_s_pred = jax.vmap(lambda s_i: v_s_fun.func(s_i))(s)
        v_pred = v_s_pred / t_span

        x_true = exact_states(t)
        v_true = exact_velocities(t)
        err = x_pred - x_true
        return t, x_pred, x_true, err, v_pred, v_true

    def initial_condition_errors(
        solver: phx.solver.FunctionalSolver,
        *,
        t0_phys: float,
        t_min: float,
        t_max: float,
    ) -> dict[str, float]:
        t_min_phys = t_min
        t_max_phys = t_max
        t_span = t_max_phys - t_min_phys
        if t_span <= 0.0:
            raise ValueError(
                f"Expected positive physical time span, got t_min={t_min_phys}, t_max={t_max_phys}."
            )

        s0 = (t0_phys - t_min_phys) / t_span
        fields = solver.ansatz_functions()
        x_fun = fields["x"]
        v_s_fun = phx.operators.dt(x_fun, var="t", ad_engine="jvp")

        x0 = jnp.asarray(x_fun.func(s0))
        v0 = jnp.asarray(v_s_fun.func(s0)) / t_span

        return {
            "x1_0_error": float(jnp.abs(x0[0] - 1.0)),
            "x2_0_error": float(jnp.abs(x0[1] - 0.0)),
            "x3_0_error": float(jnp.abs(x0[2] - 0.0)),
            "v1_0_error": float(jnp.abs(v0[0] - 0.0)),
            "v2_0_error": float(jnp.abs(v0[1] - 0.0)),
            "v3_0_error": float(jnp.abs(v0[2] - 0.0)),
        }

    return (
        evaluate_solver,
        exact_states,
        initial_condition_errors,
        make_solver,
        total_energy,
    )


@app.cell
def _(
    depth,
    jr,
    learning_rate,
    make_solver,
    mo,
    num_iter,
    num_t_interior,
    optax,
    seed,
    t_max,
    t_min,
    time_mod,
    width_size,
):
    # -------------------------------------------------------------------------
    # Main execution path
    # -------------------------------------------------------------------------
    solver, our_params = make_solver(
        t_min=t_min,
        t_max=t_max,
        width_size=width_size,
        depth=depth,
        num_t_interior=num_t_interior,
        key=jr.key(seed),
    )

    t0 = time_mod.perf_counter()
    init_loss = float(solver.loss(key=jr.key(seed + 1)))
    trained_solver = solver.solve(
        num_iter=num_iter,
        optim=optax.rprop(learning_rate),
        seed=seed,
        jit=True,
        keep_best=True,
    )
    final_loss = float(trained_solver.loss(key=jr.key(seed + 2)))
    elapsed = time_mod.perf_counter() - t0

    train_stats = {
        "init_loss": init_loss,
        "final_loss": final_loss,
        "elapsed_s": elapsed,
        "s_per_iter": elapsed / max(num_iter, 1),
    }

    train_status = mo.callout(
        mo.md(
            f"""
            ✅ Training complete

            - initial loss: `{init_loss:.6e}`
            - final loss: `{final_loss:.6e}`
            - elapsed: `{elapsed:.2f}s` (`{train_stats["s_per_iter"]:.4f}s/iter`)
            """
        ),
        kind="success",
    )
    train_status
    return our_params, train_stats, trained_solver


@app.cell
def _(
    evaluate_solver,
    exact_states,
    initial_condition_errors,
    jnp,
    nt_plot,
    t_max,
    t_min,
    total_energy,
    trained_solver,
):
    t_diag, x_pred_diag, x_true_diag, x_err_diag, v_pred_diag, v_true_diag = (
        evaluate_solver(
            trained_solver,
            t_min=t_min,
            t_max=t_max,
            nt=nt_plot,
        )
    )

    ic_err = initial_condition_errors(
        trained_solver,
        t0_phys=t_min,
        t_min=t_min,
        t_max=t_max,
    )
    v_err_diag = v_pred_diag - v_true_diag

    e_pred_diag = total_energy(x_pred_diag, v_pred_diag)
    e_true_diag = total_energy(exact_states(t_diag), v_true_diag)
    e0_diag = jnp.maximum(jnp.abs(e_pred_diag[0]), 1e-12)

    diag_stats = {
        "state_l2": float(jnp.sqrt(jnp.mean(x_err_diag**2))),
        "state_linf": float(jnp.max(jnp.abs(x_err_diag))),
        "velocity_l2": float(jnp.sqrt(jnp.mean(v_err_diag**2))),
        "velocity_linf": float(jnp.max(jnp.abs(v_err_diag))),
        "energy_abs_drift": float(jnp.max(jnp.abs(e_pred_diag - e_pred_diag[0]))),
        "energy_rel_drift": float(
            jnp.max(jnp.abs(e_pred_diag - e_pred_diag[0])) / e0_diag
        ),
        "energy_true_abs_drift": float(jnp.max(jnp.abs(e_true_diag - e_true_diag[0]))),
        **ic_err,
    }

    plot_data = (t_diag, x_pred_diag, x_true_diag, x_err_diag, e_pred_diag)
    return diag_stats, plot_data


@app.cell(hide_code=True)
def _(diag_stats, mo, train_stats):
    diagnostics_panel = mo.callout("Diagnostics unavailable.", kind="warn")
    if diag_stats is not None:
        msg = f"""
        ## Diagnostics

        - State L2 error: `{diag_stats["state_l2"]:.3e}`
        - State Linf error: `{diag_stats["state_linf"]:.3e}`
        - Velocity L2 error: `{diag_stats["velocity_l2"]:.3e}`
        - Velocity Linf error: `{diag_stats["velocity_linf"]:.3e}`
        - Energy abs drift: `{diag_stats["energy_abs_drift"]:.3e}`
        - Energy rel drift: `{diag_stats["energy_rel_drift"]:.3e}`

        Initial-condition residuals:
        - `|x1(0)-1|`: `{diag_stats["x1_0_error"]:.3e}`
        - `|x2(0)-0|`: `{diag_stats["x2_0_error"]:.3e}`
        - `|x3(0)-0|`: `{diag_stats["x3_0_error"]:.3e}`
        - `|x1'(0)-0|`: `{diag_stats["v1_0_error"]:.3e}`
        - `|x2'(0)-0|`: `{diag_stats["v2_0_error"]:.3e}`
        - `|x3'(0)-0|`: `{diag_stats["v3_0_error"]:.3e}`
        """
        if train_stats is not None:
            msg += (
                f"\n- loss (init → final): `{train_stats['init_loss']:.3e}` → "
                f"`{train_stats['final_loss']:.3e}`"
            )
        diagnostics_panel = mo.md(msg)
    diagnostics_panel
    return


@app.cell(hide_code=True)
def _(jnp, mo, plot_data, plt):
    plot_panel = mo.callout("No plot data available.", kind="warn")
    if plot_data is not None:
        t_plot, x_pred_plot, x_true_plot, x_err_plot, e_pred_plot = plot_data
        max_abs_state_err = jnp.max(jnp.abs(x_err_plot), axis=1)
        abs_energy_drift = jnp.abs(e_pred_plot - e_pred_plot[0])

        fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), constrained_layout=True)
        ax0 = axes[0, 0]
        ax1 = axes[0, 1]
        ax2 = axes[1, 0]
        ax3 = axes[1, 1]

        ax0.plot(t_plot, x_pred_plot[:, 0], color="tab:blue", lw=2.0, label="Phydrax")
        ax0.plot(t_plot, x_true_plot[:, 0], color="black", lw=1.2, ls="--", label="Exact")
        ax0.set_title("x1(t)")
        ax0.set_xlabel("t")
        ax0.set_ylabel("displacement")
        ax0.legend(loc="upper right")

        ax1.plot(t_plot, x_pred_plot[:, 1], color="tab:orange", lw=2.0, label="Phydrax")
        ax1.plot(t_plot, x_true_plot[:, 1], color="black", lw=1.2, ls="--", label="Exact")
        ax1.set_title("x2(t)")
        ax1.set_xlabel("t")
        ax1.set_ylabel("displacement")
        ax1.legend(loc="upper right")

        ax2.plot(t_plot, x_pred_plot[:, 2], color="tab:green", lw=2.0, label="Phydrax")
        ax2.plot(t_plot, x_true_plot[:, 2], color="black", lw=1.2, ls="--", label="Exact")
        ax2.set_title("x3(t)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("displacement")
        ax2.legend(loc="upper right")

        ax3.semilogy(
            t_plot,
            jnp.asarray(max_abs_state_err) + 1e-15,
            color="tab:red",
            lw=2.0,
            label="max |state error|",
        )
        ax3.semilogy(
            t_plot,
            jnp.asarray(abs_energy_drift) + 1e-15,
            color="tab:purple",
            lw=1.7,
            label="|E(t)-E(0)|",
        )
        ax3.set_title("Error and Invariant Drift")
        ax3.set_xlabel("t")
        ax3.set_ylabel("log scale")
        ax3.legend(loc="upper right")

        plot_panel = mo.hstack([mo.md(""), fig, mo.md("")], widths=[1, 8, 1])
    plot_panel
    return


@app.cell(hide_code=True)
def _(
    diag_stats,
    jax,
    mo,
    num_iter,
    num_t_interior,
    our_params,
    physicsnemo_interior_batch,
    physicsnemo_params,
    physicsnemo_steps,
    train_stats,
):
    our_steps = num_iter
    our_points_per_step = num_t_interior
    physicsnemo_interior = physicsnemo_interior_batch
    batch_ratio = our_points_per_step / max(physicsnemo_interior, 1)

    step_reduction = 100.0 * ((physicsnemo_steps - our_steps) / physicsnemo_steps)
    param_reduction = 100.0 * ((physicsnemo_params - our_params) / physicsnemo_params)
    param_ratio = physicsnemo_params / max(our_params, 1)

    state_line = ""
    if diag_stats is not None:
        state_line = (
            f"\n- Current notebook state Linf error: `{diag_stats['state_linf']:.3e}`"
        )

    speed_line = ""
    if train_stats is not None:
        speed_line = (
            f"\n- This run time per iteration: `{train_stats['s_per_iter']:.3f}s/iter`"
        )

    devices = tuple(jax.devices())
    platform_counts: dict[str, int] = {}
    device_kinds: list[str] = []
    for dev in devices:
        platform = str(dev.platform)
        platform_counts[platform] = platform_counts.get(platform, 0) + 1
        device_kinds.append(str(dev.device_kind))
    platform_summary = ", ".join(
        f"{name}:{count}" for name, count in sorted(platform_counts.items())
    )
    if not platform_summary:
        platform_summary = "unknown"
    kinds_summary = ", ".join(sorted(set(device_kinds)))
    if not kinds_summary:
        kinds_summary = "unknown"

    comparison_panel = mo.md(
        f"""
    ## Comparison with PhysicsNeMo Spring-Mass Example

    From NVIDIA's published PhysicsNeMo spring-mass example:

    - Docs page: https://docs.nvidia.com/physicsnemo/25.11/physicsnemo-sym/user_guide/foundational/ode_spring_mass.html
    - Config: https://github.com/NVIDIA/physicsnemo-sym/blob/main/examples/ode_spring_mass/conf/config.yaml
    - Solver: https://github.com/NVIDIA/physicsnemo-sym/blob/main/examples/ode_spring_mass/spring_mass_solver.py
    - FC arch defaults: https://github.com/NVIDIA/physicsnemo-sym/blob/main/physicsnemo/sym/hydra/arch.py

    Training/config context:

    - PhysicsNeMo steps: `{physicsnemo_steps:,}`
    - This notebook steps: `{our_steps:,}` (**`{step_reduction:.1f}%` fewer**)

    - PhysicsNeMo interior points/step: `{physicsnemo_interior:,}`
    - This notebook interior points/step: `{our_points_per_step:,}` (**`{batch_ratio:.0f}x` larger**)

    - PhysicsNeMo parameter count (default FC): `{physicsnemo_params:,}`
    - This notebook parameter count: `{our_params:,}` (**`{param_ratio:.1f}x` smaller**, `{param_reduction:.1f}%` fewer)

    Conditioning and invariants:

    - ICs are enforced by construction in this notebook (no soft IC loss balancing).
    - Energy drift is explicitly monitored.
    {state_line}
    {speed_line}

    Hardware context for this run: MacBook Pro M1 Max

    **tl;dr**: On a laptop with **no dedicated GPU**, we are able to **locally** solve this example
    from the PhysicsNeMo documentation in a couple of minutes, with **20x** the batch size, 
    at **1% of the iterations**, with **99% fewer parameters**, while exactly satisfying ***multiple exact initial 
    conditions simultaneously***. All using out-of-the-box Phydrax components.

    *Interested in custom optimized software for your use case? Email us at partner@phydra.ai*
    """
    )
    comparison_panel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ---
    **Tips**

    - Run as a notebook editor: `marimo edit examples/spring_mass_ode.py`
    - Run as an app: `python examples/spring_mass_ode.py`
    """
    )
    return


if __name__ == "__main__":
    app.run()
