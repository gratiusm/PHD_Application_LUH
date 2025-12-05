"""
Main entry point for Bayesian Optimization of ResNet Learning Rate.

This script orchestrates the complete BO experiment workflow:
    1. Initialization: Seeds, config validation, directory setup
    2. Data Setup: Fashion-MNIST loading and train/val/test split
    3. Bayesian Optimization: Sobol initialization + GP-guided iterations
    4. Test Evaluation: Load best model and evaluate on test set
    5. Visualization: Convergence, observations, and GP landscape plots
    6. Summary: Comprehensive text report with all results

The experiment optimizes ResNet-18 learning rate on Fashion-MNIST using
Weighted Expected Improvement acquisition with Gaussian Process surrogate.

Usage:
    python main.py

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
import time

# Local modules
import config
from bayesian_optimization import bayesian_optimization
from data import setup_data
from gaussian_process import fit_gp_model
from plotting import plot_convergence, plot_all_observations, plot_summary
from trainer import objective_function, evaluate_final_model
import utils

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    """
    Main function to run complete Bayesian Optimization experiment.
    
    This function orchestrates the entire Bayesian Optimization workflow
    for optimizing ResNet learning rate on Fashion-MNIST dataset.
    
   Workflow:
        1. Initialize environment (seed, config validation)
            - Set random seed for reproducibility
            - Validate configuration parameters
            - Print configuration summary
        
        2. Setup data (Fashion-MNIST)
            - Download and prepare Fashion-MNIST dataset
            - Create train/val/test data loaders
            - Verify data loading and preprocessing
        
        3. Run Bayesian Optimization
            - Initial design phase (Sobol sampling)
            - Iterative BO loop with GP surrogate and WEI acquisition
            - Track and save best model checkpoint continuously
            - Generate iteration-specific visualization plots
        
        4. Evaluate best configuration on test set
            - Load best model checkpoint from BO
            - Compute final test set accuracy and loss
            - Display comprehensive results
        
        5. Generate final plots and summary
            - Convergence plot (loss over iterations)
            - All observations plot (scatter in log-space)
            - Summary plot (GP landscape with observations)
            - Save observation logs and statistics
        
        6. Save final summary
            - Create comprehensive text summary
            - Save all observations to CSV
            - Document experiment configuration and results
    
    Returns:
        None: All results are saved to disk.
    """
    print("\n" + "="*70)
    print("BAYESIAN OPTIMIZATION FOR LEARNING RATE TUNING")
    print("ResNet on Fashion-MNIST")
    print("="*70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    experiment_start_time = time.time()
    
    # =========================================================================
    # STEP 1: INITIALIZATION
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: INITIALIZATION")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    utils.set_seed(config.RANDOM_SEED)
    
    # Validate configuration
    config.validate_config()
    
    # Print configuration
    config.print_config()
    
    # =========================================================================
    # STEP 2: DATA SETUP
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: DATA SETUP")
    print("="*70 + "\n")
    
    # Load Fashion-MNIST and create data loaders
    train_loader, val_loader, test_loader = setup_data()
    
    # Verify data loading
    print("\n  Data Loading Verification:")
    images, labels = next(iter(train_loader))
    print(f"    Batch shape: {images.shape}")
    print(f"    Labels shape: {labels.shape}")
    print(f"    Image dtype: {images.dtype}")
    print(f"    Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # =========================================================================
    # STEP 3: BAYESIAN OPTIMIZATION
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: BAYESIAN OPTIMIZATION")
    print("="*70)
    
    # Define objective function wrapper
    def objective_wrapper(log_lr: float):
        """
        Wrapper for objective function that includes data loaders.
        
        Args:
            log_lr: Learning rate in log10 space.
        
        Returns:
            Tuple of (validation_loss, detailed_results).
        """
        return objective_function(
            log_learning_rate=log_lr,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=config.N_EPOCHS,
            device=config.DEVICE
        )
    
    # Run Bayesian Optimization
    bo_results = bayesian_optimization(
        objective_function=objective_wrapper,
        bounds=(config.LOG_LR_MIN, config.LOG_LR_MAX),
        n_initial=config.N_INITIAL_DESIGN,
        n_iterations=config.N_BO_ITERATIONS,
        w=config.WEI_WEIGHT,
        verbose=config.VERBOSE
    )
    
    # Extract results
    X_observed = bo_results['X_observed']
    y_observed = bo_results['y_observed']
    best_x_log = bo_results['best_x']
    best_y = bo_results['best_y']
    best_lr = bo_results['best_learning_rate']
    best_iteration = bo_results['best_iteration']
    bo_time = bo_results['total_time']
    bo_val_acc = bo_results['best_val_acc']
    
    print("\n" + "="*70)
    print("BAYESIAN OPTIMIZATION RESULTS")
    print("="*70)
    print("Best Configuration Found:")
    print(f"  Learning Rate: {best_lr:.6f}")
    print(f"  Log10(LR): {best_x_log:.4f}")
    print(f"  Validation Loss: {best_y:.6f}")
    print(f"  Validation Accuracy: {bo_val_acc}%")
    print(f"  Found at Iteration: {best_iteration}/{config.TOTAL_BUDGET}")
    print(f"  BO Time: {bo_time:.2f}s ({bo_time/60:.2f} minutes)")
    print("="*70 + "\n")   
    
    # =========================================================================
    # STEP 4: FINAL EVALUATION ON TEST SET
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: FINAL EVALUATION ON TEST SET")
    print("="*70 + "\n")
    
    print("Loading best model from Bayesian Optimization...")
    print(f"Best Learning Rate: {best_lr:.6f}")
    
    # Load best model and evaluate on test set
    # All metadata (LR, validation metrics, etc.) is read from checkpoint
    test_loss, test_accuracy, model_info = evaluate_final_model(
        test_loader=test_loader,
        device=config.DEVICE
    )
    
    # Display comprehensive results
    print("\n" + "="*70)
    print("FINAL TEST SET RESULTS")
    print("="*70)
    print(f"Best Learning Rate: {model_info['learning_rate']:.6f}")
    print(f"Found at Iteration: {model_info['found_at_iteration']}")
    print(f"Validation Loss: {model_info['validation_loss']:.6f}")
    print(f"Validation Accuracy: {model_info['validation_accuracy']:.4f}%")
    print("---")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("="*70 + "\n")
    
    # =========================================================================
    # STEP 5: ADDITIONAL VISUALIZATIONS
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: CREATING ADDITIONAL VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Convergence plot
    print("Creating convergence plot...")
    plot_convergence(
        y_observed=y_observed,
        n_initial=config.N_INITIAL_DESIGN
    )
    
    # All observations plot
    print("Creating all observations plot...")
    plot_all_observations(
        X_observed=X_observed,
        y_observed=y_observed,
        n_initial=config.N_INITIAL_DESIGN
    )
    
    # Summary plot
    print("Creating summary plot...")
    final_gp = fit_gp_model(X_observed, 
                            y_observed)
    plot_summary(
        X_observed=X_observed,
        y_observed=y_observed,
        gp=final_gp,
        bounds=(config.LOG_LR_MIN, config.LOG_LR_MAX),
        n_initial=config.N_INITIAL_DESIGN
    )
    
    print("\n  All additional plots created")
    
    # =========================================================================
    # STEP 6: SAVE FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: SAVING FINAL SUMMARY")
    print("="*70 + "\n")
    
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    # Get all observations from logger
    obs_logger = bo_results['observation_logger']
    all_observations = obs_logger.get_all_observations()
    
    # Save final summary
    detailed_result = bo_results['detailed_results'][best_iteration - 1]
    best_val_acc = detailed_result['final_val_accuracy']
    utils.save_final_summary(
        best_learning_rate=best_lr,
        best_validation_loss=best_y,
        best_validation_accuracy=best_val_acc,
        best_iteration=best_iteration,
        all_observations=all_observations,
        total_time=total_experiment_time,
        test_accuracy=test_accuracy
    )
    
    # =========================================================================
    # EXPERIMENT COMPLETE
    # =========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Runtime: {total_experiment_time:.2f}s "
          f"({total_experiment_time/60:.2f} minutes)")
    print("\nFinal Results:")
    print(f"  Best Learning Rate: {best_lr:.6f}")
    print(f"  Best Validation Loss: {best_y:.6f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}%")
    print("\nOutputs:")
    print(f"  Plots: {config.PLOTS_DIR}/")
    print(f"  Results: {config.RESULTS_DIR}/")
    print("\nKey Files:")
    print(f"  - {config.OBSERVATIONS_CSV}")
    print(f"  - {config.TRAINING_LOGS_JSON}")
    print(f"  - {config.FINAL_SUMMARY_TXT}")
    print(f"  - {config.PLOTS_DIR}/iteration_*.png "
          f"({config.N_BO_ITERATIONS} plots)")
    print("="*70 + "\n")
    

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("EXPERIMENT INTERRUPTED BY USER")
        print("="*70)
        print("Partial results may be available in:")
        print(f"  - {config.RESULTS_DIR}/")
        print(f"  - {config.PLOTS_DIR}/")
        print("="*70 + "\n")
    except Exception as e:
        print("\n\n" + "="*70)
        print("EXPERIMENT FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")
        raise