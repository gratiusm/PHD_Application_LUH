"""
Acquisition functions for Bayesian Optimization.

This module implements Weighted Expected Improvement (WEI) and its optimization
for efficient exploration-exploitation trade-off in BO.

Weighted Expected Improvement (Sobester et al. 2005):
    WEI(x) = w * (y_min - μ) * Φ(Z) + (1-w) * σ * φ(Z)
    where Z = (y_min - μ) / σ
    
    - w=0: Pure exploration (maximize uncertainty)
    - w=1: Pure exploitation (minimize predicted mean)
    - w=0.5: Standard Expected Improvement (balanced)

Components:
    - weighted_expected_improvement(): Compute WEI for candidate points
    - optimize_acquisition(): Multi-start L-BFGS-B optimization with
      duplicate prevention

The optimization uses 1000 random restarts (Sobester et al. 2005)
to handle the highly multimodal WEI surface, with minimum-distance constraints
to prevent redundant evaluations.

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
from typing import Tuple

# Third-party
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Local modules
import config
from gaussian_process import GaussianProcessSurrogate

# =============================================================================
# WEIGHTED EXPECTED IMPROVEMENT (WEI)
# =============================================================================

def weighted_expected_improvement(
    X: np.ndarray,
    gp: GaussianProcessSurrogate,
    y_best: float,
    w: float = config.WEI_WEIGHT
) -> np.ndarray:
    """
    Compute Weighted Expected Improvement (WEI).
    
    WEI according to Sobester et al. (2005), Equation 13:
        WEI(x) = w * (y_best - μ(x)) * Φ(Z) + (1-w) * σ(x) * φ(Z)
    
    where:
        - μ(x): GP posterior mean at x
        - σ(x): GP posterior standard deviation at x
        - y_best: Best observed value so far (minimum for minimization)
        - Z = (y_best - μ(x)) / σ(x): Standardized improvement
        - Φ(Z): Standard normal CDF
        - φ(Z): Standard normal PDF
        - w ∈ [0,1]: Weight parameter (controls exploration vs exploitation)
    
    Interpretation:
        - w = 0: Pure exploration (focus on high uncertainty)
        - w = 1: Pure exploitation (focus on low predicted mean)
        - w = 0.5: Standard Expected Improvement (balanced)
    
    Args:
        X (np.ndarray): Candidate points of shape (n_candidates, n_features).
        gp (GaussianProcessRegressor): Fitted Gaussian Process surrogate model.
        y_best (float): Best observed value so far (minimum for minimization).
        w (float): Weight parameter in [0, 1]. 0.5 (Standard EI).
    
    Returns:
        Array of WEI values of shape (n_candidates,).
        Higher WEI = more promising candidate.
    
    Note:
        When σ = 0 (at observed points), WEI is defined as 0.
    
    References:
        Sobester et al. (2005): "On the Design of Optimization Strategies 
        Based on Global Response Surface Approximation Models"
    """
    
    # Validate y_best
    if not np.isfinite(y_best):
        raise ValueError(f"y_best must be finite, got {y_best}")
    
    # Get GP predictions
    mu, sigma = gp.predict(X, return_std=True)

    # Numerical stability threshold based on GP noise level
    # Points with sigma < 3*sqrt(alpha) are considered "certain"
    # (3-sigma rule: 99.7% confidence that we're at an observed point)
    sigma_threshold = 3.0 * np.sqrt(config.GP_ALPHA)
    
    # Identify points where GP prediction is certain (at or near observations)
    certain_points = sigma < sigma_threshold
    
    # Initialize WEI
    wei = np.zeros(len(X))
    
    # Handle uncertain points (standard case)
    if np.any(~certain_points):
        uncertain_mask = ~certain_points
        
        # Compute improvement
        improvement = y_best - mu[uncertain_mask]
        
        # Standardized improvement
        Z = improvement / sigma[uncertain_mask]
        
        # Standard normal CDF and PDF
        Phi = norm.cdf(Z)
        phi = norm.pdf(Z)
        
        # WEI formula (Sobester et al. 2005, Eq. 13)
        exploitation_term = improvement * Phi
        exploration_term = sigma[uncertain_mask] * phi
        
        wei[uncertain_mask] = (
            w * exploitation_term + (1 - w) * exploration_term
        )
    
    # Certain points (σ < threshold) get WEI = 0
    wei[certain_points] = 0.0
    
    return wei


# =============================================================================
# ACQUISITION FUNCTION OPTIMIZATION
# =============================================================================

def optimize_acquisition(
    gp: GaussianProcessSurrogate,
    y_best: float,
    bounds: Tuple[float, float],
    X_observed: np.ndarray,
    w: float = config.WEI_WEIGHT,
    n_restarts: int = config.ACQUISITION_RESTARTS,
    min_distance: float = config.MIN_DIST
) -> Tuple[np.ndarray, float]:
    """
    Optimize acquisition function using multi-start L-BFGS-B.
    
    Implements the method recommended in Sobester et al. (2005):
    "BFGS search with 1000 random restarts".
    
    The high number of restarts is necessary because the WEI surface
    is highly multimodal, especially in higher dimensions.
    
    Duplicate Prevention:
        Candidates within min_distance of already observed points are
        excluded to prevent redundant evaluations. This is critical
        because WEI can have local maxima at observed points where the
        GP has low uncertainty but good predicted performance.
    
    Args:
        gp (GaussianProcessRegressor): Fitted Gaussian Process surrogate model.
        y_best (float): Best observed value so far (minimum for minimization).
        bounds (tuple): Tuple of (min, max) for search space (log-space).
        X_observed (np.ndarray): Array of previously evaluated points to avoid.
                          If None or empty, no duplicate checking is performed.
        w (float): Weight parameter for WEI (0=exploration, 1=exploitation).
        n_restarts (int): Number of random restarts for L-BFGS-B.
        min_distance (float): Minimum distance to observed points in log-space.
                              Prevents duplicate evaluations.
    
    Returns:
        Tuple of (x_next, acq_value) where:
            x_next (np.ndarray): Next point to evaluate (in log-space).
            acq_value (float): WEI value at x_next.
    
    Raises:
        ValueError: If bounds are invalid (non-finite or min >= max).
        RuntimeError: If no valid candidate found after all restarts.
    
    References:
        Sobester et al. (2005): "On the Design of Optimization Strategies 
        Based on Global Response Surface Approximation Models".
    """
    # Reset seed for deterministic optimization
    np.random.seed(config.RANDOM_SEED)
    
    best_x = None
    best_acq = -np.inf
    
    # Define negative acquisition (scipy minimizes)
    def neg_acquisition(x):
        x_2d = np.atleast_2d(x).reshape(-1, 1)
        acq = weighted_expected_improvement(x_2d, gp, y_best, w)
        return -acq[0]  # Negative for minimization
    
    # Multi-start optimization
    # Validate bounds
    if not (np.isfinite(bounds[0]) and np.isfinite(bounds[1])):
        raise ValueError(f"Bounds must be finite, got {bounds}")

    if bounds[0] >= bounds[1]:
        raise ValueError(
            f"Lower bound must be less than upper bound, "
            f"got bounds={bounds}"
        )
    
    for _ in range(n_restarts):
        # Try to sample away from observed points
        if X_observed is not None and len(X_observed) > 0:
            max_attempts = 20  # Prevent infinite loop
            for attempt in range(max_attempts):
                x0 = np.random.uniform(bounds[0], bounds[1], size=1)
                distances = np.abs(x0[0] - X_observed.flatten())
                
                if np.min(distances) >= min_distance:
                    break  # Good initialization found

        else:
            x0 = np.random.uniform(bounds[0], bounds[1], size=1)
        
        # Optimize
        result = minimize(
            fun=neg_acquisition,
            x0=x0,
            method='L-BFGS-B',
            bounds=[bounds],
            options={
                'ftol': 1e-12,  
                'gtol': 1e-8,   
                'maxiter': 100  
            }
        )
        
        # Check if too close to observed points
        if X_observed is not None and len(X_observed) > 0:
            distances = np.abs(result.x[0] - X_observed.flatten())
            if np.min(distances) < min_distance:
                continue  
            
        # Check if better
        if -result.fun > best_acq:
            best_acq = -result.fun
            best_x = result.x
    
    if best_x is None:
        raise RuntimeError(
            f"No valid candidate found after {n_restarts} restarts. "
            f"Search space may be exhausted or min_distance too large. "
            f"Try reducing min_distance (current: {min_distance})"
        )
    
    return best_x[0], best_acq

