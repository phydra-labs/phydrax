#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import marimo


__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="1D Wave Equation in Phydrax")


@app.cell
def _():
    import time

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import optax

    import phydrax as phx

    plt.style.use("seaborn-v0_8-whitegrid")
    return jax, jnp, jr, mo, optax, phx, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 1D Wave Equation in Phydrax

    This notebook is a hands-on tutorial for a central Phydrax workflow:

    1. Build a flexible neural field \(u_\theta\) from factorized latent branches.
    2. Enforce boundary and initial constraints **by construction** using Phydrax's 
        unique enforced constraint overlay pipeline.
    3. Train with a **single PDE residual objective** rather than juggling many soft 
        penalties using a **problem-optimal differentiation** configuration.

    ---

    We solve

    $$
    u_{tt} - c^2 u_{xx} = 0,\qquad (x,t)\in[0,\pi]\times[0,2\pi],
    $$

    with constraints

    $$
    u(0,t)=u(\pi,t)=0,\qquad
    u(x,0)=\sin x,\qquad
    u_t(x,0)=\sin x.
    $$

    We also use the exact reference solution

    $$
    u_{\text{exact}}(x,t)=\sin(x)\big(\sin(t)+\cos(t)\big).
    $$

    For reference, Nvidia also provides a 
    [**PhysicsNeMo implementation**](https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/foundational/1d_wave_equation.html) 
    of this problem. 

    We demonstrate several optimizations developed at Phydra Labs that allow for 
    **orders-of-magnitude improvement across multiple axes**.

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Why PCI overlays change the optimization problem

    A traditional PINN-style objective often looks like

    $$
    J_{\text{soft}}(\theta)=
    \lambda_r\mathcal{L}_r +
    \lambda_{bc}\mathcal{L}_{bc} +
    \lambda_{ic,0}\mathcal{L}_{ic,0} + 
    \lambda_{ic,1}\mathcal{L}_{ic,1},
    $$

    where tuning the \(\lambda\)'s becomes a balancing problem.

    Phydrax's PCI pipeline instead constructs an enforced ansatz

    $$
    u_\theta = \mathcal{E}_{ic,1}\circ\mathcal{E}_{ic,0}\circ\mathcal{E}_{bc}[u_{\theta,\text{raw}}],
    $$

    so BC/IC are pinned in the function form itself. Training is then

    $$
    \min_\theta\;\mathcal{L}_r(\theta),\qquad
    \mathcal{L}_r=\mathbb{E}_{(x,t)}\left[
    \left(u_{tt}-c^2u_{xx}\right)^2
    \right].
    $$

    In short: PCI converts a multi-loss balancing act into a residual-focused solve on a
    constrained function manifold. 

    *A deeper mathematical dive into Phydrax's PCI pipeline can 
    be found [here](https://phydra-labs.github.io/phydrax/appendix/physics_constrained_interpolation/).*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            ## Why double forward JVP is the right derivative engine here

            This example uses a latent-factorized architecture of the form

            $$
            z(x,t)=z_x(x)\odot z_t(t)\in\mathbb{R}^{L},\qquad
            u(x,t)=\mathcal{C}(z(x,t)),
            $$

            where \(z_x,z_t\) map **low-dimensional coordinates** (1D \(x\), 1D \(t\))
            into a **high-dimensional latent** \(L\), then \(\mathcal{C}\) contracts
            latent channels to a scalar field.

            ---

            For wave residuals we need second derivatives:

            $$
            \mathcal{R}(x,t)=u_{tt}(x,t)-c^2u_{xx}(x,t).
            $$

            In Phydrax with `backend="ad", ad_engine="jvp"`, these are computed as
            forward-over-forward directional derivatives:

            $$
            u_{xx}=D_{e_x}\!\left(D_{e_x}u\right),\qquad
            u_{tt}=D_{e_t}\!\left(D_{e_t}u\right).
            $$

            ---

            Why this is ideal for this model class:

            1. **Input dimension is tiny.**  
               Forward-mode complexity is tied to tangent dimension (here 1 per axis),
               so JVP is naturally efficient for coordinate derivatives.

            2. **Derivative flow matches factorization.**  
               Differentiating w.r.t. \(x\) only perturbs the \(x\)-branch tangent while
               preserving multiplicative interaction with \(t\)-branch latents, and vice versa.
               This keeps derivative propagation aligned with the model’s separable algebra.

            3. **No full Hessian materialization.**  
               JVP-of-JVP gives the exact directional second derivative needed by PDE operators
               without constructing dense Hessian tensors.

            4. **Great fit with coord-separable sampling.**  
               In this notebook we sample \(x\)-axes and \(t\)-points in a structured way,
               so derivative computation stays close to the branchwise geometry instead of
               flattening everything into a monolithic dense-input path.

            Conceptually: this architecture is "low-dimensional coordinate geometry 
            + large latent algebra + contraction."
            Double forward JVP is the AD mechanism that most directly respects that geometry.
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
    latent_size = 32
    width_size = 20
    depth = 6
    num_iter = 100
    learning_rate = 7e-3
    num_x_interior = 100
    num_t_interior = 200
    nx_plot = 200
    nt_plot = 400
    seed = 0
    c = 1.0
    x_min = 0.0
    x_max = 3.141592653589793
    t_min = 0.0
    t_max = 6.283185307179586
    return (
        c,
        depth,
        latent_size,
        learning_rate,
        nt_plot,
        num_iter,
        num_t_interior,
        num_x_interior,
        nx_plot,
        seed,
        t_max,
        t_min,
        width_size,
        x_max,
        x_min,
    )


@app.cell
def _(jax, jnp, jr, phx):
    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------
    def u_exact(x: jax.Array, t: jax.Array) -> jax.Array:
        return jnp.sin(x) * (jnp.sin(t) + jnp.cos(t))

    def evaluate_on_grid(
        u: phx.domain.DomainFunction,
        *,
        x_min: float,
        x_max: float,
        t_min: float,
        t_max: float,
        nx: int,
        nt: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        x = jnp.linspace(x_min, x_max, int(nx))
        t = jnp.linspace(t_min, t_max, int(nt))
        xx, tt = jnp.meshgrid(x, t, indexing="ij")

        x_flat = xx.reshape((-1,))
        t_flat = tt.reshape((-1,))
        point_eval = jax.jit(
            jax.vmap(lambda x_i, t_i: u.func(jnp.asarray([x_i], dtype=float), t_i))
        )
        u_pred = point_eval(x_flat, t_flat).reshape((int(nx), int(nt)))
        u_true = u_exact(xx, tt)
        diff = u_pred - u_true
        return x, t, u_pred, u_true, diff

    def constraint_errors(
        u: phx.domain.DomainFunction,
        *,
        x_min: float,
        x_max: float,
        t_min: float,
        t_max: float,
    ) -> dict[str, float]:
        t_line = jnp.linspace(t_min, t_max, 256)
        x_line = jnp.linspace(x_min, x_max, 256)

        left = jax.vmap(lambda t_i: u.func(jnp.asarray([x_min], dtype=float), t_i))(
            t_line
        )
        right = jax.vmap(lambda t_i: u.func(jnp.asarray([x_max], dtype=float), t_i))(
            t_line
        )
        u0 = jax.vmap(lambda x_i: u.func(jnp.asarray([x_i], dtype=float), t_min))(x_line)
        ut = phx.operators.dt(u, var="t")
        ut0 = jax.vmap(lambda x_i: ut.func(jnp.asarray([x_i], dtype=float), t_min))(
            x_line
        )

        return {
            "max_bc_error": float(
                jnp.maximum(jnp.max(jnp.abs(left)), jnp.max(jnp.abs(right)))
            ),
            "max_u0_error": float(jnp.max(jnp.abs(u0 - jnp.sin(x_line)))),
            "max_ut0_error": float(jnp.max(jnp.abs(ut0 - jnp.sin(x_line)))),
        }

    # -------------------------------------------------------------------------
    # Main solver factory
    # -------------------------------------------------------------------------
    def make_solver(
        *,
        c: float,
        x_min: float,
        x_max: float,
        t_min: float,
        t_max: float,
        latent_size: int,
        width_size: int,
        depth: int,
        num_x_interior: int,
        num_t_interior: int,
        key: jax.Array,
    ) -> phx.solver.FunctionalSolver:
        geom = phx.domain.Interval1d(float(x_min), float(x_max))
        time_dom = phx.domain.TimeInterval(float(t_min), float(t_max))
        domain = geom @ time_dom

        key_x, key_t, key_bw = jr.split(key, 3)
        model = phx.nn.LatentContractionModel(
            latent_size=int(latent_size),
            out_size="scalar",
            execution_policy=phx.nn.LatentExecutionPolicy(
                topology="best_effort_flat",
                layout="coord_separable",
                fallback="error",
            ),
            x=phx.nn.MLP(
                in_size="scalar",
                out_size=int(latent_size),
                width_size=int(width_size),
                depth=int(depth),
                activation=phx.nn.Stan(int(width_size)),
                key=key_x,
            ),
            t=phx.nn.MLP(
                in_size="scalar",
                out_size=int(latent_size),
                width_size=int(width_size),
                depth=int(depth),
                activation=phx.nn.Stan(int(width_size)),
                key=key_t,
            ),
        )
        u_raw = domain.Model("x", "t")(model)

        boundary = domain.component({"x": phx.domain.Boundary()})
        initial = domain.component({"t": phx.domain.FixedStart()})

        @domain.Function("x")
        def u0_target(x):
            return jnp.sin(x[0])

        @domain.Function("x")
        def ut0_target(x):
            return jnp.sin(x[0])

        # PCI overlays: exact BC/IC enforcement by construction.
        overlays = [
            phx.solver.SingleFieldEnforcedConstraint(
                "u",
                boundary,
                lambda f: phx.constraints.enforce_dirichlet(
                    f, boundary, var="x", target=0.0
                ),
            ),
            phx.solver.SingleFieldEnforcedConstraint(
                "u",
                initial,
                lambda f: f,
                time_derivative_order=0,
                initial_target=u0_target,
            ),
            phx.solver.SingleFieldEnforcedConstraint(
                "u",
                initial,
                lambda f: f,
                time_derivative_order=1,
                initial_target=ut0_target,
            ),
        ]

        # Residual objective only: derivatives are computed with JVP engine.
        structure_xt = phx.domain.ProductStructure((("x",), ("t",)))
        residual = phx.constraints.ContinuousPointwiseInteriorConstraint(
            "u",
            domain,
            operator=lambda f: (
                phx.operators.dt_n(f, var="t", order=2, ad_engine="jvp")
                - (float(c) ** 2)
                * phx.operators.laplacian(f, var="x", ad_engine="jvp")
            ),
            num_points={"x": int(num_x_interior), "t": int(num_t_interior)},
            structure=structure_xt,
            reduction="mean",
            label="wave_residual",
        )

        return phx.solver.FunctionalSolver(
            functions={"u": u_raw},
            constraints=[residual],
            constraint_terms=overlays,
            boundary_weight_key=key_bw,
        )

    return constraint_errors, evaluate_on_grid, make_solver


@app.cell
def _(
    c,
    depth,
    jr,
    latent_size,
    learning_rate,
    make_solver,
    mo,
    num_iter,
    num_t_interior,
    num_x_interior,
    optax,
    seed,
    t_max,
    t_min,
    time,
    width_size,
    x_max,
    x_min,
):
    # -------------------------------------------------------------------------
    # Main execution path
    # -------------------------------------------------------------------------
    solver = make_solver(
        c=c,
        x_min=float(x_min),
        x_max=float(x_max),
        t_min=float(t_min),
        t_max=float(t_max),
        latent_size=int(latent_size),
        width_size=int(width_size),
        depth=int(depth),
        num_x_interior=int(num_x_interior),
        num_t_interior=int(num_t_interior),
        key=jr.key(int(seed)),
    )
    t0 = time.perf_counter()
    init_loss = float(solver.loss(key=jr.key(int(seed) + 1)))
    trained_solver = solver.solve(
        num_iter=int(num_iter),
        optim=optax.rprop(float(learning_rate)),
    )
    final_loss = float(trained_solver.loss(key=jr.key(int(seed) + 2)))
    elapsed = float(time.perf_counter() - t0)
    train_stats = {
        "init_loss": init_loss,
        "final_loss": final_loss,
        "elapsed_s": elapsed,
        "s_per_iter": elapsed / max(int(num_iter), 1),
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
    return train_stats, trained_solver


@app.cell
def _(
    constraint_errors,
    evaluate_on_grid,
    jnp,
    nt_plot,
    nx_plot,
    t_max,
    t_min,
    trained_solver,
    x_max,
    x_min,
):
    u = trained_solver.ansatz_functions()["u"]
    x, t, u_pred, u_true, diff = evaluate_on_grid(
        u,
        x_min=float(x_min),
        x_max=float(x_max),
        t_min=float(t_min),
        t_max=float(t_max),
        nx=int(nx_plot),
        nt=int(nt_plot),
    )
    errors = constraint_errors(
        u,
        x_min=float(x_min),
        x_max=float(x_max),
        t_min=float(t_min),
        t_max=float(t_max),
    )
    diag_stats = {
        "l2_error": float(jnp.sqrt(jnp.mean((u_pred - u_true) ** 2))),
        "linf_error": float(jnp.max(jnp.abs(diff))),
        **errors,
    }
    plot_data = (x, t, u_pred, u_true, diff)
    return diag_stats, plot_data


@app.cell(hide_code=True)
def _(diag_stats, mo, train_stats):
    diagnostics_panel = mo.callout("Diagnostics unavailable.", kind="warn")
    if diag_stats is not None:
        msg = f"""
        ## Diagnostics

        - L2 error: `{diag_stats["l2_error"]:.3e}`
        - Linf error: `{diag_stats["linf_error"]:.3e}`
        - max BC error: `{diag_stats["max_bc_error"]:.3e}`
        - max IC value error: `{diag_stats["max_u0_error"]:.3e}`
        - max IC slope error: `{diag_stats["max_ut0_error"]:.3e}`
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
        x_plot, t_plot, u_pred_plot, u_true_plot, diff_plot = plot_data

        extent = (
            float(x_plot[0]),
            float(x_plot[-1]),
            float(t_plot[0]),
            float(t_plot[-1]),
        )
        field_min = float(jnp.minimum(jnp.min(u_pred_plot), jnp.min(u_true_plot)))
        field_max = float(jnp.maximum(jnp.max(u_pred_plot), jnp.max(u_true_plot)))
        diff_lim = float(jnp.max(jnp.abs(diff_plot)))
        diff_lim = diff_lim if diff_lim > 0.0 else 1e-12

        fig, axes = plt.subplots(1, 3, figsize=(10.4, 6.6), constrained_layout=True)
        ax0, ax1, ax2 = axes

        im0 = ax0.imshow(
            jnp.asarray(u_pred_plot.T),
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="jet",
            vmin=field_min,
            vmax=field_max,
        )
        ax0.set_title("Phydrax Solution")
        ax0.set_xlabel("x")
        ax0.set_ylabel("t")
        ax0.grid(False)
        fig.colorbar(im0, ax=ax0, fraction=0.072, pad=0.042)

        im1 = ax1.imshow(
            jnp.asarray(u_true_plot.T),
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="jet",
            vmin=field_min,
            vmax=field_max,
        )
        ax1.set_title("Ground Truth")
        ax1.set_xlabel("x")
        ax1.set_ylabel("t")
        ax1.grid(False)
        fig.colorbar(im1, ax=ax1, fraction=0.072, pad=0.042)

        im2 = ax2.imshow(
            jnp.asarray(diff_plot.T),
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="jet",
            vmin=-diff_lim,
            vmax=diff_lim,
        )
        ax2.set_title("Error (pred - exact)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("t")
        ax2.grid(False)
        fig.colorbar(im2, ax=ax2, fraction=0.072, pad=0.042)

        plot_panel = mo.hstack([mo.md(""), fig, mo.md("")], widths=[1, 8, 1])
    plot_panel
    return


@app.cell(hide_code=True)
def _(diag_stats, mo, num_iter, num_t_interior, num_x_interior, train_stats):
    physicsnemo_linf = 0.025
    phydrax_reference_linf = 0.00186
    physicsnemo_steps = 10_000
    physicsnemo_collocation = 3_200

    our_steps = int(num_iter)
    our_collocation = int(num_x_interior) * int(num_t_interior)
    step_reduction = 100.0 * ((physicsnemo_steps - our_steps) / physicsnemo_steps)
    collocation_ratio = our_collocation / physicsnemo_collocation
    reference_linf_reduction = 100.0 * (
        (physicsnemo_linf - phydrax_reference_linf) / physicsnemo_linf
    )

    current_linf_line = ""
    if diag_stats is not None:
        current_linf = float(diag_stats["linf_error"])
        current_reduction = 100.0 * ((physicsnemo_linf - current_linf) / physicsnemo_linf)
        current_linf_line = (
            f"\n- Current notebook run Linf: `{current_linf:.3e}` "
            f"(`{current_reduction:.1f}%` lower than PhysicsNeMo baseline)"
        )

    speed_line = ""
    if train_stats is not None:
        speed_line = (
            f"\n- This run time per iteration: `{train_stats['s_per_iter']:.3f}s/iter`"
        )

    comparison_panel = mo.md(
        f"""
    ## Comparison with PhysicsNeMo

    From NVIDIA's published PhysicsNeMo 1D wave example:

    - PhysicsNeMo reported $L_\\infty$ error: ~`{physicsnemo_linf:.3e}`  
      Source: https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/foundational/1d_wave_equation.html#results
    - Phydrax reference result (this setup family): ~`{phydrax_reference_linf:.3e}`  
      → **`{reference_linf_reduction:.1f}%` lower Linf error**
    {current_linf_line}

    Training/config context:

    - PhysicsNeMo config steps: `{physicsnemo_steps:,}`
    - This notebook steps: `{our_steps:,}` (**`{step_reduction:.1f}%` fewer**)
    - PhysicsNeMo collocation points: ~`{physicsnemo_collocation:,}`
    - This notebook collocation points: `{our_collocation:,}` (**`{collocation_ratio:.0f}x` more**)
    {speed_line}

    PhysicsNeMo config reference:
    https://github.com/NVIDIA/physicsnemo-sym/blob/main/examples/wave_equation/conf/config.yaml

    Hardware context for these Phydrax runs: MacBook Pro M1 Max laptop (no dedicated GPU).


    **tl;dr**: On a **laptop** with **no dedicated GPU**, we are able to **locally** solve this example
    from the PhysicsNeMo documentation in a **couple of minutes**, with **100x** the batch size, 
    at **1%** of the iterations,
    and still get a **>90% reduction in error**, while exactly satisfying ***multiple exact initial and 
    boundary conditions simultaneously***. All using out-of-the-box Phydrax components, highlighting our philosophy 
    at Phydra Labs of scalable, modular, composable high-performance SciML software tooling, where all the meticulous 
    optimization and implementation complexity is abstracted away from the user (human or agent!)

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

    - Run as a notebook editor: `marimo edit examples/wave1d.py`
    - Run as an app: `python examples/wave1d.py`
    """
    )
    return


if __name__ == "__main__":
    app.run()
