"""
Visualization for Bayesian Optimization iterations.

This module provides plotting functions for:
- BO iteration plots (observations, GP fit, uncertainty, acquisition)
- Convergence plots
- Final summary visualizations

Each plot shows:
1. All observations (initial design + BO iterations)
2. GP posterior mean
3. GP uncertainty
4. Acquisition function (WEI)

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
from typing import Tuple, Optional

# Third-party
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Local modules
from acquisition import optimize_acquisition, weighted_expected_improvement
import config
from gaussian_process import GaussianProcessSurrogate, fit_gp_model

# =============================================================================
# PLOT STYLING CONFIGURATION
# =============================================================================

PLOT_STYLE = {
    'initial': {
        'color': 'green',
        'marker': 's',
        'size': 150,
        'label': 'Initial Design'
    },
    'bo': {
        'color': 'red',
        'marker': 'o',
        'size': 150,
        'label': 'BO Iterations'
    },
    'best': {
        'color': 'gold',
        'marker': '*',
        'size': 400,
        'label': 'Best So Far'
    },
    'next':{
        'color': 'red',
        'marker': '^',
        'size': 200,
        'label': 'Maximum (Next Point)'        
    },
    'edge': {
        'color': 'black',
        'width': 1.5
    }
}

def _add_uncertainty_bands(ax, X_plot, mu, sigma):
    """
    Add 1σ, 2σ, 3σ uncertainty bands to axis.
        
    Creates three-layer visualization of GP posterior uncertainty with
    progressively lighter shading for wider confidence intervals.
    
    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        X_plot (np.ndarray): X coordinates for plotting, shape (n_points,) or 
            (n_points, 1).
        mu (np.ndarray): GP posterior mean, shape (n_points,).
        sigma (np.ndarray): GP posterior standard deviation, shape (n_points,).
    
    Returns:
        None
    
    Note:
        Layers are ordered by z-order: ±3σ (back) → ±2σ (middle) → ±1σ (front)
        for proper visual hierarchy.
    """
    ax.fill_between(
        X_plot.flatten(),
        mu - 3*sigma,
        mu + 3*sigma,
        alpha=0.15,
        color='#B0E0E6',
        label='±3σ',
        zorder=1
    )
    ax.fill_between(
        X_plot.flatten(),
        mu - 2*sigma,
        mu + 2*sigma,
        alpha=0.25,
        color='#6495ED',
        label='±2σ',
        zorder=2
    )
    ax.fill_between(
        X_plot.flatten(),
        mu - sigma,
        mu + sigma,
        alpha=0.35,
        color='#1E90FF',
        label='±1σ',
        zorder=3
    )
    
# =============================================================================
# MAIN BO ITERATION PLOT
# =============================================================================

def plot_bo_iteration(
    iteration: int,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    bounds: Tuple[float, float],
    n_initial: int = config.N_INITIAL_DESIGN,
    w: float = config.WEI_WEIGHT,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive plot for one BO iteration.
    
    Generates a two-panel visualization showing:
        1. Top panel: GP posterior (mean + uncertainty bands) with all 
           observations and next evaluation point
        2. Bottom panel: WEI acquisition function landscape with maximum
    
    The function internally fits a GP to all observations and optimizes
    the acquisition function to determine the next evaluation point.
    
    Args:
        iteration (int): Current iteration number.
        X_observed (np.ndarray): All observed points so far in log-space,
            shape (n_obs, 1).
        y_observed (np.ndarray): All observed validation losses, 
            shape (n_obs,).
        bounds (Tuple[float, float]): Search space bounds in log-space
            (e.g., (-4, -1) for LR range [10^-4, 10^-1]).
        n_initial (int, optional): Number of initial design points for 
            visual distinction. Defaults to config.N_INITIAL_DESIGN.
        w (float, optional): WEI weight parameter. 
        save_path (Optional[str], optional): Path to save plot. If None,
            saves to 'plots/iteration_XX.pdf'. Defaults to None.
    
    Returns:
        None
    
    Raises:
        RuntimeError: If GP fitting or acquisition optimization fails.
    
    Note:
        The top panel includes a secondary x-axis showing actual learning
        rates for easier interpretation.
    """
    # Determine save path
    if save_path is None:
        save_path = (f"{config.PLOTS_DIR}/iteration_{iteration:02d}"
                     f".{config.PLOT_TYPE}.")
    
    # Create dense grid for smooth curves
    X_plot = np.linspace(bounds[0], bounds[1], 10000).reshape(-1, 1)
    
    try:
        # Fit GP on all observed data
        gp = fit_gp_model(X_observed, 
                          y_observed)
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to fit Gaussian Process on "
            f"{len(X_observed)} observations for plotting. "
            f"Error: {str(e)}"
        ) from e
    
    # Get GP predictions
    mu, sigma = gp.predict(X_plot, return_std=True)
    
    try:
        # Compute next proposed point via acquisition optimization
        x_next_scalar, _ = optimize_acquisition(
            gp=gp,
            y_best=np.min(y_observed),
            bounds=bounds,
            w=w,
            X_observed=X_observed
        )
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to optimize acquisition function. "
            f"Error: {str(e)}"
        ) from e
    
    # Compute acquisition function
    acq_values = weighted_expected_improvement(
        X=X_plot, 
        gp=gp, 
        y_best=np.min(y_observed), 
        w=w
    )
 
    # Create figure with two subplots
    fig = plt.figure(figsize=config.PLOT_FIGSIZE, constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3, figure=fig)
    
    # =========================================================================
    # SUBPLOT 1: GP FIT WITH OBSERVATIONS
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    
    # GP posterior mean
    ax1.plot(
        X_plot.flatten(),
        mu,
        'b-',
        linewidth=2,
        label='GP Posterior Mean',
        zorder=2
    )
    
    # Three-layer uncertainty visualization (like lecture slides)
    _add_uncertainty_bands(ax1, X_plot, mu, sigma)
    
    # Separate observations into initial design and BO iterations
    initial_mask = np.arange(len(X_observed)) < n_initial
    bo_mask = ~initial_mask
    
    # Plot initial design points
    if np.any(initial_mask):
        ax1.scatter(
            X_observed[initial_mask].flatten(),
            y_observed[initial_mask],
            c=PLOT_STYLE['initial']['color'],
            s=PLOT_STYLE['initial']['size'],
            marker=PLOT_STYLE['initial']['marker'],
            edgecolors=PLOT_STYLE['edge']['color'],
            linewidth=PLOT_STYLE['edge']['width'],
            label=PLOT_STYLE['initial']['label'],
            zorder=10
        )
    
    # Plot BO-selected points
    if np.any(bo_mask):
        ax1.scatter(
            X_observed[bo_mask].flatten(),
            y_observed[bo_mask],
            c=PLOT_STYLE['bo']['color'],
            s=PLOT_STYLE['bo']['size'],
            marker=PLOT_STYLE['bo']['marker'],
            edgecolors=PLOT_STYLE['edge']['color'],
            linewidth=PLOT_STYLE['edge']['width'],
            label=PLOT_STYLE['bo']['label'],
            zorder=10
        )
    
    # Mark best observation
    best_idx = np.argmin(y_observed)
    ax1.scatter(
        X_observed[best_idx],
        y_observed[best_idx],
        c=PLOT_STYLE['best']['color'],
        s=PLOT_STYLE['best']['size'],
        marker=PLOT_STYLE['best']['marker'],
        edgecolors=PLOT_STYLE['edge']['color'],
        linewidth=PLOT_STYLE['edge']['width'],
        label=PLOT_STYLE['best']['label'],
        zorder=15
    )
    
    # Mark next evaluation point (if provided)
    if x_next_scalar is not None:
        ax1.axvline(
            x=x_next_scalar,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label='Next Evaluation',
            zorder=5
        )
    
    # Formatting
    ax1.set_xlabel('log₁₀(Learning Rate)', 
                   fontsize=12, 
                   fontweight='bold')
    ax1.set_ylabel('Validation Loss', 
                   fontsize=12, 
                   fontweight='bold')
    ax1.set_title(
        f'Bayesian Optimization - Iteration {iteration}/{config.TOTAL_BUDGET}',
        fontsize=14,
        fontweight='bold'
    )
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add secondary x-axis with actual learning rates
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    
    # Create nice tick positions
    log_ticks = ax1.get_xticks()
    actual_lr_ticks = 10 ** log_ticks
    ax1_top.set_xticks(log_ticks)
    ax1_top.set_xticklabels([f'{lr:.4f}' for lr in actual_lr_ticks], 
                            fontsize=9)
    ax1_top.set_xlabel('Actual Learning Rate', 
                       fontsize=11, 
                       style='italic')
    
    # =========================================================================
    # SUBPLOT 2: ACQUISITION FUNCTION
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    
    # Plot acquisition function
    ax2.plot(
        X_plot.flatten(),
        acq_values,
        'g-',
        linewidth=2,
        label=f'WEI (w={w})',
        zorder=2
    )
    
    # Fill under curve
    ax2.fill_between(
        X_plot.flatten(),
        0,
        acq_values,
        alpha=0.3,
        color='green',
        zorder=1
    )
        
    # Mark maximum (next evaluation point)
    if x_next_scalar is not None:
        acq_at_next = weighted_expected_improvement(
            X=np.array([[x_next_scalar]]),
            gp=gp,
            y_best=np.min(y_observed),
            w=w
        )[0] 
        
        ax2.scatter(
            x_next_scalar,
            acq_at_next,
            c=PLOT_STYLE['next']['color'],
            s=PLOT_STYLE['next']['size'],
            marker=PLOT_STYLE['next']['marker'],
            edgecolors=PLOT_STYLE['edge']['color'],
            linewidth=PLOT_STYLE['edge']['width'],
            label=PLOT_STYLE['next']['label'],
            zorder=10
        )
        
        ax2.axvline(
            x=x_next_scalar,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            zorder=5
        )
    
    # Formatting
    ax2.set_xlabel('log₁₀(Learning Rate)', 
                   fontsize=12, 
                   fontweight='bold')
    ax2.set_ylabel('Acquisition Value', 
                   fontsize=12, 
                   fontweight='bold')
    ax2.set_title('Weighted Expected Improvement', 
                  fontsize=12, 
                  fontweight='bold')
    ax2.legend(loc='upper right', 
               fontsize=10)
    ax2.grid(True, 
             alpha=0.3, 
             linestyle='--')
    x_range = bounds[1] - bounds[0]
    padding = 0.01 * x_range
    ax1.set_xlim(bounds[0] - padding, bounds[1] + padding)
    ax2.set_xlim(bounds[0] - padding, bounds[1] + padding)
    
    # =========================================================================
    # SAVE FIGURE
    # =========================================================================
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"  Plot saved: {save_path}")


# =============================================================================
# CONVERGENCE PLOT
# =============================================================================

def plot_convergence(
    y_observed: np.ndarray,
    n_initial: int = config.N_INITIAL_DESIGN,
    save_path: Optional[str] = None
) -> None:
    """
    Plot convergence: best observed value vs iteration.
    
    Shows the incumbent as a step function over iterations.
    
    Args:
        y_observed (np.ndarray): All observed validation losses in order
            of evaluation, shape (n_obs,).
        n_initial (int, optional): Number of initial design points. Used
            to mark transition to BO phase. 
        save_path (Optional[str], optional): Path to save plot. If None,
            saves to 'plots/convergence.pdf'. Defaults to None.
    
    Returns:
        None
    
    """
    if save_path is None:
        save_path = f"{config.PLOTS_DIR}/convergence.{config.PLOT_TYPE}"
    
    # Compute incumbent (best value at each iteration)
    best_values = np.minimum.accumulate(y_observed)
    iterations = np.arange(1, len(y_observed) + 1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot as step function
    ax.step(
        iterations,
        best_values,
        'b-',
        linewidth=2.5,
        where='post',
        label='Best Validation Loss',
        zorder=2
    )
    
    # Mark transition from initial design to BO
    ax.axvline(
        x=n_initial + 0.5,
        color='red',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label='Initial Design → BO',
        zorder=1
    )
    
    # Formatting
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title(
        'Bayesian Optimization Convergence',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, len(y_observed) + 0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"  Convergence plot saved: {save_path}")


# =============================================================================
# OBSERVATIONS SCATTER PLOT
# =============================================================================

def plot_all_observations(
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    n_initial: int = config.N_INITIAL_DESIGN,
    save_path: Optional[str] = None
) -> None:
    """
    Create scatter plot of all observations.
    
    Visualizes all evaluated learning rates and their validation losses,
    with visual distinction between initial design (Sobol) and BO-selected
    points. Points are connected in evaluation order and annotated with
    iteration numbers.
    
    Args:
        X_observed (np.ndarray): All observed points in log-space,
            shape (n_obs, 1).
        y_observed (np.ndarray): All observed validation losses,
            shape (n_obs,).
        n_initial (int, optional): Number of initial design points for
            visual distinction.
        save_path (Optional[str], optional): Path to save plot. If None,
            saves to 'plots/all_observations.pdf'. Defaults to None.
    
    Returns:
        None
    
    Note:
        Uses logarithmic x-axis scale for learning rates to better visualize
        the search space structure.
    """
    if save_path is None:
        save_path = f"{config.PLOTS_DIR}/all_observations.{config.PLOT_TYPE}"
    
    # Transform X to actual learning rates
    lr_observed = 10 ** X_observed.flatten()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate initial design and BO points
    initial_mask = np.arange(len(X_observed)) < n_initial
    bo_mask = ~initial_mask
    
    # Plot initial design
    if np.any(initial_mask):
        ax.scatter(
            lr_observed[initial_mask],
            y_observed[initial_mask],
            c=PLOT_STYLE['initial']['color'],
            s=PLOT_STYLE['initial']['size'],
            marker=PLOT_STYLE['initial']['marker'],
            edgecolors=PLOT_STYLE['edge']['color'],
            linewidth=PLOT_STYLE['edge']['width'],
            label=PLOT_STYLE['initial']['label'],
            alpha=0.8
        )
    
    # Plot BO iterations
    if np.any(bo_mask):
        ax.scatter(
            lr_observed[bo_mask],
            y_observed[bo_mask],
            c=PLOT_STYLE['bo']['color'],
            s=PLOT_STYLE['bo']['size'],
            marker=PLOT_STYLE['bo']['marker'],
            edgecolors=PLOT_STYLE['edge']['color'],
            linewidth=PLOT_STYLE['edge']['width'],
            label=PLOT_STYLE['bo']['label'],
            alpha=0.8
        )
    
    # Mark best observation
    best_idx = np.argmin(y_observed)
    ax.scatter(
        lr_observed[best_idx],
        y_observed[best_idx],
        c=PLOT_STYLE['best']['color'],
        s=PLOT_STYLE['best']['size'],
        marker=PLOT_STYLE['best']['marker'],
        edgecolors=PLOT_STYLE['edge']['color'],
        linewidth=3,
        label='Best Configuration',
        zorder=10
    )
    
    # Connect points in order
    ax.plot(
        lr_observed,
        y_observed,
        'k--',
        alpha=0.3,
        linewidth=1,
        zorder=1
    )
    
    # Formatting
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('All Evaluations', fontsize=14, fontweight='bold')
    ax.set_xscale('log')  # Log scale for learning rate
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Annotate each point with iteration number
    for i, (lr, loss) in enumerate(zip(lr_observed, y_observed)):
        ax.annotate(
            str(i + 1),
            xy=(lr, loss),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.7
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"  Observations plot saved: {save_path}")


# =============================================================================
# SUMMARY VISUALIZATION
# =============================================================================

def plot_summary(
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    gp: GaussianProcessSurrogate,
    bounds: Tuple[float, float],
    n_initial: int = config.N_INITIAL_DESIGN,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive summary plot with 4 subplots.
    
    Generates a plot with four subplots providing complete overview of the 
    optimization:
        1. Top-left: All observations scatter (log-scale LR)
        2. Top-right: Final GP fit with uncertainty bands
        3. Bottom-left: Convergence curve (incumbent over iterations)
        4. Bottom-right: Learning rate distribution histogram
    
    Args:
        X_observed (np.ndarray): All observed points in log-space,
            shape (n_obs, 1).
        y_observed (np.ndarray): All observed validation losses,
            shape (n_obs,).
        gp (GaussianProcessSurrogate): Fitted GP model for final landscape
            visualization.
        bounds (Tuple[float, float]): Search space bounds in log-space.
        n_initial (int, optional): Number of initial design points for
            visual distinction. Defaults to config.N_INITIAL_DESIGN.
        save_path (Optional[str], optional): Path to save plot. If None,
            saves to 'plots/summary.pdf'.
    
    Returns:
        None
    """
    if save_path is None:
        save_path = f"{config.PLOTS_DIR}/summary.{config.PLOT_TYPE}"
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Transform to actual learning rates
    lr_observed = 10 ** X_observed.flatten()
    
    # =========================================================================
    # SUBPLOT 1: All observations
    # =========================================================================
    ax = axes[0, 0]
    
    initial_mask = np.arange(len(X_observed)) < n_initial
    bo_mask = ~initial_mask
    
    if np.any(initial_mask):
        ax.scatter(lr_observed[initial_mask], 
                   y_observed[initial_mask],
                   c=PLOT_STYLE['initial']['color'], 
                   s=PLOT_STYLE['initial']['size'], 
                   marker=PLOT_STYLE['initial']['marker'], 
                   edgecolors=PLOT_STYLE['edge']['color'],
                   linewidth=PLOT_STYLE['edge']['width'],
                   label=PLOT_STYLE['initial']['label']
        )
        
           
    
    if np.any(bo_mask):
        ax.scatter(lr_observed[bo_mask], 
                   y_observed[bo_mask],
                   c=PLOT_STYLE['bo']['color'], 
                   s=PLOT_STYLE['bo']['size'], 
                   marker=PLOT_STYLE['bo']['marker'], 
                   edgecolors=PLOT_STYLE['edge']['color'],
                   linewidth=PLOT_STYLE['edge']['width'],
                   label=PLOT_STYLE['bo']['label']
        )
    
    best_idx = np.argmin(y_observed)
    ax.scatter(lr_observed[best_idx], 
               y_observed[best_idx],
               c=PLOT_STYLE['best']['color'], 
               s=PLOT_STYLE['best']['size'], 
               marker=PLOT_STYLE['best']['marker'], 
               edgecolors=PLOT_STYLE['edge']['color'],
               linewidth=3, 
               label='Best', 
               zorder=10
    )
    
    ax.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
    ax.set_title('All Evaluations', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # SUBPLOT 2: Final GP fit
    # =========================================================================
    ax = axes[0, 1]
        
    X_plot = np.linspace(bounds[0], bounds[1], 10000).reshape(-1, 1)
    mu, sigma = gp.predict(X_plot, return_std=True)
    
    ax.plot(X_plot.flatten(), mu, 'b-', linewidth=2, label='GP Mean')

    _add_uncertainty_bands(ax, X_plot, mu, sigma)    
        
    # Plot initial design points
    if np.any(initial_mask):
        ax.scatter(
            X_observed[initial_mask].flatten(),
            y_observed[initial_mask],
            c=PLOT_STYLE['initial']['color'],
            s=100,
            marker=PLOT_STYLE['initial']['marker'],
            edgecolors=PLOT_STYLE['edge']['color'],
            linewidth=PLOT_STYLE['edge']['width'],
            label=PLOT_STYLE['initial']['label'],
            zorder=10
        )
    
    # Plot BO-selected points
    if np.any(bo_mask):
        ax.scatter(
            X_observed[bo_mask].flatten(),
            y_observed[bo_mask],
            c=PLOT_STYLE['bo']['color'],
            s=100,
            marker=PLOT_STYLE['bo']['marker'],
            edgecolors=PLOT_STYLE['edge']['color'],
            linewidth=PLOT_STYLE['edge']['width'],
            label=PLOT_STYLE['bo']['label'],
            zorder=10
        )
    
    # Mark best observation
    best_idx = np.argmin(y_observed)
    ax.scatter(
        X_observed[best_idx],
        y_observed[best_idx],
        c=PLOT_STYLE['best']['color'],
        s=150,
        marker=PLOT_STYLE['best']['marker'],
        edgecolors=PLOT_STYLE['edge']['color'],
        linewidth=PLOT_STYLE['edge']['width'],
        label=PLOT_STYLE['best']['label'],
        zorder=15
    )
    
    ax.set_xlabel('log₁₀(Learning Rate)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
    ax.set_title('Final GP Fit', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # SUBPLOT 3: Convergence
    # =========================================================================
    ax = axes[1, 0]

    best_values = np.minimum.accumulate(y_observed)
    iterations = np.arange(1, len(y_observed) + 1)
    
    # Step function
    ax.step(
        iterations,
        best_values,
        'b-',
        linewidth=2,
        where='post',
        label='Best Validation Loss'
    )
    
    ax.axvline(
        x=n_initial + 0.5,
        color='red',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label='Initial → BO'
    )
    
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Best Validation Loss', fontsize=11, fontweight='bold')
    ax.set_title('Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # SUBPLOT 4: Learning rate distribution
    # =========================================================================
    ax = axes[1, 1]
    
    ax.hist(lr_observed, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(lr_observed[best_idx], color='gold', linestyle='--',
               linewidth=3, label=f'Best LR: {lr_observed[best_idx]:.6f}')
    
    ax.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate Distribution', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # OVERALL TITLE
    # =========================================================================
    fig.suptitle(
        'Bayesian Optimization Summary',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"  Summary plot saved: {save_path}")

