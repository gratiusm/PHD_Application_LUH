"""
Bayesian Optimization main algorithm.

This module orchestrates the complete BO procedure for ResNet learning rate
optimization on Fashion-MNIST.

Workflow:
    1. Initial Design: Sobol sequence sampling for space-filling initialization
    2. BO Loop: GP fitting → WEI acquisition optimization → objective 
       evaluation
    3. Model Checkpointing: Continuous tracking and saving of best model
    4. Visualization: Per-iteration plots of GP landscape (from iteration 4)
    5. Logging: Comprehensive CSV/JSON tracking of all observations

Components:
    - generate_sobol_samples(): Low-discrepancy sequence generation
    - bayesian_optimization(): Main BO loop orchestrating all phases

The algorithm uses Weighted Expected Improvement (WEI) with Gaussian Process
surrogate (Matern 5/2 kernel) to efficiently search the log-learning-rate space.

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
import time
import warnings
from typing import Tuple, Dict, Any

# Third-party
import numpy as np
from scipy.stats import qmc

# Local modules
from acquisition import optimize_acquisition
import config
from gaussian_process import fit_gp_model
from plotting import plot_bo_iteration
import utils

# =============================================================================
# SOBOL SEQUENCE GENERATION
# =============================================================================

def generate_sobol_samples(
    n_samples: int,
    bounds: Tuple[float, float],
    seed: int = config.RANDOM_SEED
) -> np.ndarray:
    """
    Generate quasi-random Sobol samples for initial design.
    
    Sobol sequences are low-discrepancy sequences that provide better
    space-filling properties than pure random sampling.
    
    Args:
        n_samples (int): Number of samples to generate (e.g., 3).
        bounds (Tuple[float, float]): Tuple of (min, max) for the search space.
        seed (int): Random seed for scrambling (reproducibility).
    
    Returns:
        np.ndarray: Array of shape (n_samples, 1) with Sobol samples in 
            [min, max].

    Note:
        Suppresses scipy's "balance properties" warning for sample sizes that 
        are not powers of 2.
    """
    # Create Sobol sampler
    # d=1 because 1D search space (learning rate)
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)

    # Suppress the power-of-2 warning for small initial designs
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*balance properties.*')
        # Generate samples in [0, 1]
        samples_unit = sampler.random(n_samples)
            
    # Scale to actual bounds
    l_bound = np.array([bounds[0]])
    u_bound = np.array([bounds[1]])
    samples_scaled = qmc.scale(samples_unit, l_bound, u_bound)
    
    if config.VERBOSE:
        print(f"  Generated {n_samples} Sobol samples")
        print(f"  Bounds: [{bounds[0]}, {bounds[1]}]")
        print(f"  Samples (log-space): {samples_scaled.flatten()}")
        print(f"  Samples (actual LR): {10**samples_scaled.flatten()}")
    
    return samples_scaled


# =============================================================================
# BAYESIAN OPTIMIZATION MAIN LOOP
# =============================================================================

def bayesian_optimization(
    objective_function: callable,
    bounds: Tuple[float, float],
    n_initial: int = config.N_INITIAL_DESIGN,
    n_iterations: int = config.N_BO_ITERATIONS,
    w: float = config.WEI_WEIGHT,
    verbose: bool = config.VERBOSE
) -> Dict[str, Any]:
    """
    Complete Bayesian Optimization procedure.
    
    This function implements the full BO algorithm:
        
    1. Phase 1: Initial Design (Sobol sampling)
        - Generate n_initial Sobol samples
        - Evaluate objective function at each point
        - Save best model encountered so far
        
    2. Phase 2: Bayesian Optimization Loop
       - Fit GP to all observations
       - Optimize acquisition function (WEI)
       - Evaluate objective function at selected point
       - Update best model if current result is better
       - Create visualization plot (starting from iteration 4)
       - Log all results
    
    Model Checkpointing:
        The best model (lowest validation loss) is continuously tracked
        and saved during optimization. Each time a better configuration
        is found, the model checkpoint at 'results/best_model.pth' is
        overwritten. This ensures the final checkpoint always contains
        the best model found during the entire BO procedure.
    
    Args:
        objective_function (callable): Black-box function to optimize.
            Signature: f(x: float) -> (float, dict) where x is in log-space and 
            returns (validation_loss, detailed_results_dict).
        bounds (Tuple[float, float]): Tuple of (min, max) for search space 
            in log-space.
        n_initial (int): Number of initial Sobol samples.
        n_iterations (int): Number of BO iterations after initial design.
        w (float): Weight parameter for WEI (0=exploration, 1=exploitation).
        verbose (bool): Whether to print detailed information.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'X_observed' (np.ndarray): All evaluated points (log-space), 
               shape (n, 1)
            - 'y_observed' (np.ndarray): All objective values, shape (n,)
            - 'best_x' (float): Best point found (log-space)
            - 'best_y' (float): Best objective value (validation loss)
            - 'best_val_acc' (float): Validation accuracy at best point
            - 'best_iteration' (int): Iteration where best was found 
            - 'best_learning_rate' (float): Best LR in real space (10^best_x)
            - 'total_time' (float): Total optimization time in seconds
            - 'detailed_results' (List[dict]): List of detailed results per 
               iteration
            - 'observation_logger' (ObservationLogger): ObservationLogger 
               instance with all logs
            - 'training_logger' (TrainingLogger): TrainingLogger instance with 
               epoch details
    
    Note:
        The best model checkpoint is saved at 'results/best_model.pth'
        and is loaded using utils.load_best_model() for final test set 
        evaluation.
    """
    
    start_time = time.time()
    
    # Initialize loggers
    obs_logger = utils.ObservationLogger()
    train_logger = utils.TrainingLogger()
    
    # Storage for observations
    X_observed = []
    y_observed = []
    detailed_results = []
    
    # Best model tracking (during optimization)
    current_best_val_loss = float('inf')
    
    # =========================================================================
    # PHASE 1: INITIAL DESIGN (SOBOL SAMPLING)
    # =========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("PHASE 1: INITIAL DESIGN (SOBOL SAMPLING)")
        print("="*70 + "\n")
    
    # Generate Sobol samples
    X_initial = generate_sobol_samples(n_initial, bounds)
    
    # Evaluate all initial points
    for i in range(n_initial):
        iteration = i + 1
        x_log = X_initial[i, 0]
        
        utils.print_iteration_header(iteration, 
                                     n_initial + n_iterations, 
                                     is_initial=True)
        
        # Evaluate objective function
        y, result_dict = objective_function(x_log)
        
        # Store observation
        X_observed.append(x_log)
        y_observed.append(y)
        detailed_results.append(result_dict)
        
        # Check if this is best so far
        is_best = (y == min(y_observed))
        
        # Log observation
        obs_logger.log_observation(
            iteration=iteration,
            learning_rate=10**x_log,
            log_learning_rate=x_log,
            validation_loss=y,
            validation_accuracy=result_dict['final_val_accuracy'],
            training_time=result_dict['training_time'],
            is_initial_design=True,
            is_best_so_far=is_best
        )
        
        # Save best model if this is best so far
        if y < current_best_val_loss:
            current_best_val_loss  = y
            
            if 'model_state_dict' in result_dict:
                utils.save_best_model(
                    model_state_dict=result_dict['model_state_dict'],
                    learning_rate=10**x_log,
                    validation_loss=y,
                    validation_accuracy=result_dict['final_val_accuracy'],
                    iteration=iteration,
                    epoch_metrics={
                        'train_losses': result_dict.get('epoch_train_losses'),
                        'val_losses': result_dict.get('epoch_val_losses'),
                        'val_accuracies': result_dict.get(
                            'epoch_val_accuracies'
                        )
                    }
                )

        # Log detailed training info
        if result_dict:
            train_logger.log_iteration(
                iteration=iteration,
                learning_rate=10**x_log,
                epoch_losses=result_dict['epoch_train_losses'],
                epoch_val_losses=result_dict['epoch_val_losses'],
                epoch_val_accs=result_dict['epoch_val_accuracies'],
                final_val_loss=result_dict['final_val_loss'],
                final_val_accuracy=result_dict['final_val_accuracy'],
                training_time=result_dict['training_time'],
                metadata={'phase': 'initial_design', 'method': 'sobol'}
            )
        
        # Print summary
        utils.print_iteration_summary(
            iteration=iteration,
            learning_rate=10**x_log,
            log_lr=x_log,
            val_loss=y,
            val_acc=result_dict['final_val_accuracy'],
            training_time=result_dict['training_time'],
            is_best=is_best
        )
    
    # Convert to numpy arrays
    X_observed_array = np.array(X_observed).reshape(-1, 1)
    y_observed_array = np.array(y_observed)
    
    # =========================================================================
    # PHASE 2: BAYESIAN OPTIMIZATION LOOP
    # =========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("PHASE 2: BAYESIAN OPTIMIZATION LOOP")
        print("="*70 + "\n")
    
    for bo_iter in range(n_iterations):
        iteration = n_initial + bo_iter + 1
        
        utils.print_iteration_header(iteration, 
                                     n_initial + n_iterations, 
                                     is_initial=False)
        
        # ---------------------------------------------------------------------
        # STEP 1: Fit Gaussian Process
        # ---------------------------------------------------------------------
        if verbose:
            print(f"Step 1: Fitting Gaussian Process on "
                  f"{len(X_observed)} observations")
        
        with utils.timer("GP fitting", verbose=verbose):
            gp = fit_gp_model(X_observed_array, 
                              y_observed_array
            )
        
        # ---------------------------------------------------------------------
        # STEP 2: Optimize Acquisition Function
        # ---------------------------------------------------------------------
        if verbose:
            print("\nStep 2: Optimizing acquisition function")
        
        y_best = np.min(y_observed_array)
        
        with utils.timer("Optimizing acquisition function", verbose=verbose):
            x_next_scalar, acq_value = optimize_acquisition(
                gp=gp,
                y_best=y_best,
                bounds=bounds,
                w=w,
                X_observed=X_observed_array
            )
        
        if verbose:
            print(f"  Next point: {x_next_scalar:.6f} (log-space)")
            print(f"  Next LR: {10**x_next_scalar:.6f}")
            print(f"  Acquisition value: {acq_value:.6f}")
        
        # ---------------------------------------------------------------------
        # STEP 3: Evaluate Objective Function
        # ---------------------------------------------------------------------
        if verbose:
            print("\nStep 3: Evaluating objective function")
        
        y_next, result_dict = objective_function(x_next_scalar)
        
        # ---------------------------------------------------------------------
        # STEP 4: Update Observations
        # ---------------------------------------------------------------------
        X_observed.append(x_next_scalar)
        y_observed.append(y_next)
        detailed_results.append(result_dict)
        
        X_observed_array = np.array(X_observed).reshape(-1, 1)
        y_observed_array = np.array(y_observed)
        
        # Check if this is best so far
        is_best = (y_next == np.min(y_observed_array))
        
        # Log observation
        obs_logger.log_observation(
            iteration=iteration,
            learning_rate=10**x_next_scalar,
            log_learning_rate=x_next_scalar,
            validation_loss=y_next,
            validation_accuracy=result_dict['final_val_accuracy'],
            training_time=result_dict['training_time'],
            is_initial_design=False,
            is_best_so_far=is_best
        )
        
        # Save best model if this is best so far
        if y_next < current_best_val_loss:
            current_best_val_loss  = y_next
            
            if 'model_state_dict' in result_dict:
                utils.save_best_model(
                    model_state_dict=result_dict['model_state_dict'],
                    learning_rate=10**x_next_scalar,
                    validation_loss=y_next,
                    validation_accuracy=result_dict['final_val_accuracy'],
                    iteration=iteration,
                    epoch_metrics={
                        'train_losses': result_dict.get('epoch_train_losses'),
                        'val_losses': result_dict.get('epoch_val_losses'),
                        'val_accuracies': result_dict.get(
                            'epoch_val_accuracies'
                        )
                    }
                )      
        
        # Log detailed training info
        if result_dict:
            train_logger.log_iteration(
                iteration=iteration,
                learning_rate=10**x_next_scalar,
                epoch_losses=result_dict['epoch_train_losses'],
                epoch_val_losses=result_dict['epoch_val_losses'],
                epoch_val_accs=result_dict['epoch_val_accuracies'],
                final_val_loss=result_dict['final_val_loss'],
                final_val_accuracy=result_dict['final_val_accuracy'],
                training_time=result_dict['training_time'],
                metadata={'phase': 'bo_loop', 
                          'acquisition': 'WEI', 
                          'acq_value': float(acq_value)}
            )
        
        # Print summary
        utils.print_iteration_summary(
            iteration=iteration,
            learning_rate=10**x_next_scalar,
            log_lr=x_next_scalar,
            val_loss=y_next,
            val_acc=result_dict['final_val_accuracy'],
            training_time=result_dict['training_time'],
            is_best=is_best
        )
        
        # ---------------------------------------------------------------------
        # STEP 5: Create Plot (starting from iteration 4)
        # ---------------------------------------------------------------------
        if iteration >= config.PLOT_START_ITERATION:
            if verbose:
                print(f"Step 4: Creating plot for iteration {iteration}")
            
            plot_bo_iteration(
                iteration=iteration,
                X_observed=X_observed_array,
                y_observed=y_observed_array,
                bounds=bounds,
                n_initial=n_initial,
            )
        
    
    # =========================================================================
    # OPTIMIZATION COMPLETE
    # =========================================================================
    
    total_time = time.time() - start_time
    
    # Find best result
    best_idx = np.argmin(y_observed_array)
    best_x = X_observed_array[best_idx, 0]
    best_y = y_observed_array[best_idx]
    best_iteration = best_idx + 1
    
    if verbose:
        print("\n" + "="*70)
        print("BAYESIAN OPTIMIZATION COMPLETE")
        print("="*70)

    # Return comprehensive results
    return {
        'X_observed': X_observed_array,
        'y_observed': y_observed_array,
        'best_x': best_x,
        'best_y': best_y,
        'best_val_acc': detailed_results[best_idx]['final_val_accuracy'],
        'best_iteration': best_iteration,
        'best_learning_rate': 10**best_x,
        'total_time': total_time,
        'detailed_results': detailed_results,
        'observation_logger': obs_logger,
        'training_logger': train_logger
    }
