"""
Gaussian Process wrapper for Bayesian Optimization.

This module provides a clean interface to scikit-learn's GaussianProcessRegressor
optimized for hyperparameter optimization tasks.

Key Features:
    - Matern 5/2 kernel
    - Automatic hyperparameter optimization via MLE
    - Numerical stability features (Alpha, normalization)
    - Retry mechanism for failed Cholesky decompositions
    - Clean prediction interface for acquisition functions

Components:
    - create_kernel(): Factory for Matern 5/2 kernel with proper bounds
    - GaussianProcessSurrogate: Main GP wrapper class with fit/predict
    - fit_gp_model(): Convenience function for quick GP creation

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
import warnings
from typing import Tuple, Optional

# Third-party
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Local modules
import config

# =============================================================================
# KERNEL CREATION
# =============================================================================

def create_kernel() -> object:
    """
    Create Matern 5/2 kernel for Gaussian Process.
    
    Mathematical Definition:
        k(x, x') = σ² × Matern_{5/2}(r)
    
        where:
            r = ||x - x'|| / ℓ
            Matern_{5/2}(r) = (1 + √5·r + 5r²/3) × exp(-√5·r)
            σ² : signal variance (controlled by ConstantKernel)
            ℓ  : length scale (controlled by Matern kernel)
    
    Hyperparameter Bounds:
        Signal variance σ²: [1e-2, 10.0]
            - Covers expected val loss variance (0.01-10.0)
            - Lower bound prevents over-trusting noisy observations
            - Upper bound prevents modeling all variance as signal
            
        Length scale ℓ: [0.2, 20.0]
            - Scaled for log-LR space [-4, -2] with range = 2.0
            - Lower bound (0.2): 10% of range, captures fine details
            - Upper bound (20.0): 10× range, allows smooth functions
            - Based on Rasmussen & Williams (2006) guidelines
        
    Why Matern 5/2?
        1. Not infinitely smooth like RBF (more flexible)
        2. Recommended by Snoek et al. (2012) for BO
        3. Good balance: smooth enough but not too rigid
        
    Returns:
        sklearn.gaussian_process.kernels.Kernel: Product kernel 
            (ConstantKernel * Matern) ready for GP fitting with:
            - ConstantKernel: initial value 1.0, bounds [1e-2, 10.0]
            - Matern: nu=2.5 (5/2 smoothness), length_scale=1.0, 
              bounds [0.2, 20.0]
        
    Example:
        >>> kernel = create_kernel()
        >>> gp = GaussianProcessRegressor(kernel=kernel, 
                                          n_restarts_optimizer=10)
        >>> gp.fit(X_train, y_train)
    """
    kernel = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-2, 10.0)  # Signal variance
    ) * Matern(
        length_scale=1.0,
        length_scale_bounds=(0.2, 20.0),    # Length scale
        nu=config.GP_MATERN_NU  # 2.5 for Matern 5/2
    )
 
    return kernel


# =============================================================================
# GAUSSIAN PROCESS WRAPPER
# =============================================================================

class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model for Bayesian Optimization.
    
    This class wraps scikit-learn's GaussianProcessRegressor with:
        - Appropriate kernel (Matern 5/2)
        - Automatic hyperparameter optimization
        - Numerical stability features (Alpha, normalization)
        - Retry mechanism for failed fits
        - Clean prediction interface
    
    Attributes:
        gp (GaussianProcessRegressor): Scikit-learn GaussianProcessRegressor 
            instance.
        kernel (Kernel): Kernel used by the GP.
        alpha (float): Observation noise variance.
        n_restarts_optimizer (int): Number of restarts for hyperparameter 
            optimization.
        normalize_y (bool): Whether to normalize target values.
        X_train (np.ndarray): Training inputs.
        y_train (np.ndarray): Training targets.
        is_fitted (bool): Whether the GP has been fitted.
    """
    
    def __init__(
        self,
        alpha: float = config.GP_ALPHA,
        n_restarts_optimizer: int = config.GP_N_RESTARTS,
        normalize_y: bool = config.GP_NORMALIZE_Y
    ):
        """
        Initialize Gaussian Process surrogate.

        Args:
            alpha (float): Observation noise variance. Models the noise in 
                target values: y = f(x) + ε, where ε ~ N(0, α).
            n_restarts_optimizer (int): Number of random restarts for 
                hyperparameter optimization via L-BFGS-B.
            normalize_y (bool): Whether to normalize target values before 
                fitting. Recommended for BO to improve numerical stability.
        """
        self.kernel = create_kernel()
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        
        # Create GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=config.RANDOM_SEED
        )
        
        # Training data (stored for reference)
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_retries: int = config.GP_MAX_RETRIES
    ) -> None:
        """
        Fit Gaussian Process to observations.
        
        How GP Fitting Works:
            1. Hyperparameter Optimization:
               - Maximizes log marginal likelihood via L-BFGS-B
               - Multiple random restarts (n_restarts_optimizer)
               - Optimizes: signal variance σ² and length scale ℓ
            
            2. Retry Mechanism:
               - If Cholesky decomposition fails (singular matrix)
               - Increases alpha by GP_JITTER_MULTIPLIER (default 10x)
               - Up to max_retries attempts
            
            3. Computational Complexity:
               - Time: O(n³) for Cholesky decomposition
               - Space: O(n²) for kernel matrix K
               - Becomes slow for n > 1000
            
        Numerical Stability:
            Alpha is added to the diagonal of the kernel matrix: K + α·I
            
            This prevents:
                - Singular matrices from numerical precision issues
                - Overfitting when observations are noise-free
                - Cholesky decomposition failures
        
        Args:
            X (np.ndarray): Training inputs of shape (n_samples, 1)
            y (np.ndarray): Training targets of shape (n_samples,).
                Validation losses to minimize.
            max_retries (int): Maximum number of retries with increased alpha.
        
        Raises:
            ValueError: If input dimensions are invalid.
            RuntimeError: If GP fitting fails after all retries.
        """
        # Validate input
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        
        if X.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 samples to fit GP. Got {X.shape[0]}"
            )
        
        # Store training data
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Attempt to fit GP with retries
        current_alpha = self.alpha
        
        for attempt in range(max_retries):
            try:
                # Suppress sklearn warnings during fitting
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    # Update alpha if retrying
                    if attempt > 0:
                        current_alpha *= config.GP_JITTER_MULTIPLIER
                        self.gp.alpha = current_alpha
                        
                        if config.VERBOSE:
                            print(f"  Retry {attempt}/{max_retries}: "
                                  f"Increased noise to {current_alpha:.2e}")
                    
                    # Fit GP (includes hyperparameter optimization)
                    self.gp.fit(X, y)
                
                self.is_fitted = True
                
                if config.VERBOSE:
                    print(f"  GP fitted successfully with "
                          f"{X.shape[0]} observations")
                
                return  
            
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    raise RuntimeError(
                        f"GP fitting failed after {max_retries} attempts. "
                        f"Last error: {str(e)}"
                    )
                # Continue to next retry
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with fitted GP.
        
        Interpretation of Outputs:
            Mean μ(x):
                - Expected value of function at x
                - Best point estimate for prediction
                
            Standard deviation σ(x):
                - Uncertainty estimate at x
                - High σ: unexplored region (good for exploration)
                - Low σ: near observed data (confident prediction)
                - σ ≈ 0 at observed points (by construction)
            
            Confidence Intervals:
                - 68% CI: [μ - σ, μ + σ]
                - 95% CI: [μ - 2σ, μ + 2σ]
    
        Args:
            X (np.ndarray): Test inputs of shape (n_samples, n_features).
            return_std (bool): If True, return standard deviation. 
                If False, only return mean.
        
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                If return_std is True:
                    Tuple of (mean, std) where:
                        - mean (np.ndarray): Predictive mean
                        - std (np.ndarray): Predictive standard deviation
                If return_std is False:
                    mean (np.ndarray): Predictive mean only
        
        Raises:
            RuntimeError: If GP has not been fitted yet.
        
        """
        if not self.is_fitted:
            raise RuntimeError("GP must be fitted before making predictions")
        
        # Ensure 2D array
        X = np.atleast_2d(X)
        
        # Make predictions
        if return_std:
            mean, std = self.gp.predict(X, return_std=True)
            return mean, std
        else:
            mean = self.gp.predict(X, return_std=False)
            return mean
    
    
# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fit_gp_model(
    X: np.ndarray,
    y: np.ndarray
) -> GaussianProcessSurrogate:
    """
    Convenience function to create and fit GP model.
    
    Args:
        X: Training inputs of shape (n_samples, n_features).
        y: Training targets of shape (n_samples,).
    
    Returns:
        Fitted GaussianProcessSurrogate instance.
    
    Example:
        >>> X = np.array([[-3.5], [-2.8], [-2.1]]).reshape(-1, 1)
        >>> y = np.array([0.45, 0.38, 0.52])
        >>> gp = fit_gp_model(X, y)
        >>> mean, std = gp.predict(np.array([[-3.0]]))
    """
    gp = GaussianProcessSurrogate()
    gp.fit(X, y)
    return gp

