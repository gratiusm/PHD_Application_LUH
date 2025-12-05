"""
Utility functions for Bayesian Optimization experiment.

This module provides essential helper functions for:
    - Reproducibility: Random seed setting for all libraries
    - Logging: CSV/JSON loggers for observations and training metrics
    - Model Management: Save/load best model checkpoints
    - Timing: Context manager for code block timing
    - Error Handling: NaN/Inf detection and CUDA OOM diagnostics
    - Pretty Printing: Formatted console output for iterations

Key Components:
    - set_seed(): Set random seeds for reproducibility
    - ObservationLogger: Track all BO evaluations (CSV)
    - TrainingLogger: Track per-epoch training metrics (JSON)
    - save_best_model() / load_best_model(): Model checkpoint management
    - timer(): Context manager for timing operations
    - is_nan_or_inf() / handle_cuda_oom(): Error diagnostics

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
import csv
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional

# Third-party
import numpy as np
import torch

# Local modules
import config

# =============================================================================
# REPRODUCIBILITY: SEED SETTING
# =============================================================================

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """
    Set random seed for all libraries to ensure reproducibility.
    
    This function sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA)
        - cuDNN backend (deterministic mode)
    
    Args:
        seed (int): Integer seed value for random number generators.
                    Default: config.RANDOM_SEED (42)
    
    Returns:
        None
        
    Note:
        Setting deterministic mode may reduce performance but ensures
        reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure deterministic behavior in cuDNN 
    # (disables non-deterministic algorithms)
    # Note: This reduces performance by ~10-30% but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if config.VERBOSE:
        print(f"  Random seed set to {seed} for reproducibility")


# =============================================================================
# LOGGING: OBSERVATION TRACKING
# =============================================================================

class ObservationLogger:
    """
    Logger for tracking all Bayesian Optimization observations.
    
    This class manages comprehensive logging of the BO optimization process,
    including learning rates evaluated, validation metrics, timing information,
    and metadata about each observation. All data is automatically saved to
    CSV format for easy post-hoc analysis and visualization.
    
    The logger tracks:
        - Learning rates (both real and log-space)
        - Validation losses and accuracies
        - Iteration numbers and timestamps
        - Training times per iteration
        - Phase information (initial design vs. BO iterations)
        - Best-so-far indicators

   Attributes:
        csv_path (str): Path to CSV file for saving observations.
        observations (List[Dict[str, Any]]): In-memory list of all logged
            observations as dictionaries.
    """
    
    def __init__(self, csv_path: str = config.OBSERVATIONS_CSV):
        """
        Initialize observation logger.
        
        Args:
            csv_path: Path to CSV file for saving observations.
        """
        self.csv_path = csv_path
        self.observations = []
        
        # Create CSV with header if it doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration',
                    'learning_rate',
                    'log_learning_rate',
                    'validation_loss',
                    'validation_accuracy',
                    'training_time_seconds',
                    'is_initial_design',
                    'is_best_so_far',
                    'timestamp'
                ])
    
    def log_observation(
        self,
        iteration: int,
        learning_rate: float,
        log_learning_rate: float,
        validation_loss: float,
        validation_accuracy: float,
        training_time: float,
        is_initial_design: bool,
        is_best_so_far: bool
    ) -> None:
        """
        Log a single observation to CSV.
        
        Args:
            iteration: Current iteration number (1-10)
            learning_rate: Evaluated learning rate
            log_learning_rate: log10(learning_rate)
            validation_loss: Validation loss achieved
            validation_accuracy: Validation accuracy achieved
            training_time: Time taken for training in seconds
            is_initial_design: Whether this is from Sobol initial design
            is_best_so_far: Whether this is the best result so far
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        observation = {
            'iteration': iteration,
            'learning_rate': learning_rate,
            'log_learning_rate': log_learning_rate,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_accuracy,
            'training_time_seconds': training_time,
            'is_initial_design': is_initial_design,
            'is_best_so_far': is_best_so_far,
            'timestamp': timestamp
        }
        
        self.observations.append(observation)
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                f"{learning_rate:.6f}",
                f"{log_learning_rate:.4f}",
                f"{validation_loss:.6f}",
                f"{validation_accuracy:.4f}",
                f"{training_time:.2f}",
                is_initial_design,
                is_best_so_far,
                timestamp
            ])
        
        if config.VERBOSE:
            print(f"  Logged observation for iteration {iteration}")
    
    def get_best_observation(self) -> Optional[Dict[str, Any]]:
        """
        Get the observation with the lowest validation loss.
        
        Returns:
            Dictionary containing best observation, or None if no observations.
        """
        if not self.observations:
            return None
        
        return min(self.observations, key=lambda x: x['validation_loss'])
    
    def get_all_observations(self) -> List[Dict[str, Any]]:
        """
        Get all logged observations.
        
        Returns:
            List of observation dictionaries.
        """
        return self.observations


# =============================================================================
# LOGGING: DETAILED TRAINING LOGS
# =============================================================================

class TrainingLogger:
    """
    Logger for detailed per-epoch training information.
    
    This class tracks comprehensive training dynamics across all BO
    iterations, recording per-epoch metrics for detailed post-hoc analysis.
    All data is automatically saved to JSON format with structured hierarchy.
    
    The logger tracks:
        - Per-epoch training losses
        - Per-epoch validation losses and accuracies
        - Final validation metrics per iteration
        - Learning rates used
        - Training times
        - Timestamps and optional metadata
    
    Attributes:
        json_path (str): Path to JSON file for saving detailed logs.
        logs (List[Dict[str, Any]]): In-memory list of all logged iterations
            with nested structure for training/validation metrics.
    
    JSON Structure:
        Each iteration entry contains:
            - iteration: Iteration number
            - learning_rate: LR used for this iteration
            - training: {epoch_losses, num_epochs}
            - validation: {epoch_losses, epoch_accuracies, final_loss, 
              final_accuracy}
            - training_time_seconds: Total training duration
            - timestamp: ISO format timestamp
            - metadata: Optional additional information

    """
    
    def __init__(self, json_path: str = config.TRAINING_LOGS_JSON):
        """
        Initialize training logger.
        
        Args:
            json_path: Path to JSON file for saving detailed logs.
        """
        self.json_path = json_path
        self.logs = []
    
    def log_iteration(
        self,
        iteration: int,
        learning_rate: float,
        epoch_losses: List[float],
        epoch_val_losses: List[float],
        epoch_val_accs: List[float],
        final_val_loss: float,
        final_val_accuracy: float,
        training_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log detailed information for one iteration.
        
        Args:
            iteration: Current iteration number
            learning_rate: Learning rate used
            epoch_losses: List of training losses per epoch
            epoch_val_losses: List of validation losses per epoch
            epoch_val_accuracies: List of validation accuracies per epoch
            final_val_loss: Final validation loss
            final_val_accuracy: Final validation accuracy
            training_time: Total training time in seconds
            metadata: Optional additional metadata
        """
        log_entry = {
            'iteration': iteration,
            'learning_rate': learning_rate,
            'training': {
                'epoch_losses': [float(loss) for loss in epoch_losses],
                'num_epochs': len(epoch_losses)
            },
            'validation': {
                'epoch_losses': [float(loss) for loss in epoch_val_losses],
                'epoch_accuracies': [float(acc) for acc in epoch_val_accs],
                'final_loss': float(final_val_loss),
                'final_accuracy': float(final_val_accuracy)
            },
            'training_time_seconds': float(training_time),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if metadata:
            log_entry['metadata'] = metadata
        
        self.logs.append(log_entry)
        
        # Save to JSON after each iteration
        self.save()
    
    def save(self) -> None:
        """
        Save all logs to JSON file.
        """
        with open(self.json_path, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        if config.VERBOSE:
            print(f"  Training logs saved to {self.json_path}")
    
    def load(self) -> List[Dict[str, Any]]:
        """
        Load logs from JSON file.
        
        Returns:
            List of log entries.
        """
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.logs = json.load(f)
        return self.logs


# =============================================================================
# BEST MODEL MANAGEMENT
# =============================================================================

def save_best_model(
    model_state_dict: Dict[str, torch.Tensor],
    learning_rate: float,
    validation_loss: float,
    validation_accuracy: float,
    iteration: int,
    epoch_metrics: Optional[Dict[str, List[float]]] = None
) -> str:
    """
    Save the best model found during Bayesian Optimization.
    
    This function saves the model with the lowest validation loss encountered
    during the hyperparameter search. The saved checkpoint includes all
    necessary information to reproduce the result and evaluate on test data.
    
    The checkpoint is saved as a single file that gets overwritten whenever
    a better model is found, ensuring only the truly best model is retained.
    
    Args:
        model_state_dict (Dict[str, torch.Tensor]): Model state dictionary 
            containing all weights and biases of the ResNet.
        learning_rate (float): Learning rate that achieved this result.
        validation_loss (float): Validation loss achieved (lower is better).
        validation_accuracy (float): Validation accuracy achieved (percentage).
        iteration (int): Iteration number where this best result was found.
        epoch_metrics (Optional[Dict[str, List[float]]], optional): Dictionary 
            containing per-epoch training metrics with keys: 'train_losses', 
            'val_losses', 'val_accuracies'. Each value is a list of floats with 
            length equal to number of epochs.
            
    Returns:
        str: Absolute path to the saved checkpoint file 
        
    Checkpoint Structure:
        The saved checkpoint contains:
            - model_state_dict: Complete model weights
            - best_learning_rate: Optimal LR found
            - validation_loss: Validation loss achieved
            - validation_accuracy: Validation accuracy (%)
            - found_at_iteration: When this was found
            - epoch_metrics: Per-epoch training history
            - timestamp: Save time in ISO format
            - config_snapshot: Reproducibility information
                (n_epochs, batch_size, random_seed, architecture, 
                dataset, device)
                
    Notes:
        - File is saved to config.RESULTS_DIR as 'best_model.pth'
        - Previous best model is automatically overwritten
        - Checkpoint includes reproducibility information (seed, config)
        - Uses PyTorch's standard checkpoint format (.pth)
    
    See Also:
        load_best_model: Function to load the saved checkpoint.
        torch.save: PyTorch's checkpoint saving function.
    """
    # Construct absolute path for best model checkpoint
    best_model_path = os.path.join(
        config.RESULTS_DIR,
        "best_model.pth"
    )
    
    # Create comprehensive checkpoint dictionary
    checkpoint = {
        # Model and optimizer states
        'model_state_dict': model_state_dict,
        
        # Best hyperparameter and performance metrics
        'best_learning_rate': learning_rate,
        'validation_loss': validation_loss,
        'validation_accuracy': validation_accuracy,
        'found_at_iteration': iteration,
        
        # Optional detailed training history
        'epoch_metrics': epoch_metrics,
        
        # Metadata for reproducibility
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config_snapshot': {
            'n_epochs': config.N_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'random_seed': config.RANDOM_SEED,
            'model_architecture': 'ResNet18',
            'dataset': 'Fashion-MNIST',
            'device': str(config.DEVICE)
        }
    }
    
    # Ensure results directory exists
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save checkpoint using PyTorch's save function
    torch.save(checkpoint, best_model_path)
        
    return os.path.abspath(best_model_path)


def load_best_model(
    model: torch.nn.Module
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load the best model checkpoint from disk.
    
    This function loads the saved best model checkpoint and restores the
    model weights. It is primarily used for final test set evaluation
    after Bayesian Optimization completes.
    
    The function performs validation to ensure the checkpoint exists and
    contains all required keys before attempting to load weights.
    
    Args:
        model (torch.nn.Module): Initialized ResNet model. The state dict will 
            be loaded into this model instance.
    
    Returns:                           
        Tuple[torch.nn.Module, Dict[str, Any]]: A tuple containing:
            - model (torch.nn.Module): PyTorch model with loaded weights,
              set to eval mode (dropout/batchnorm disabled).
            - checkpoint_dict (Dict[str, Any]): Full checkpoint dictionary
              containing all metadata and metrics:
                - model_state_dict: Model weights
                - best_learning_rate: Optimal LR found
                - validation_loss: Validation loss achieved
                - validation_accuracy: Validation accuracy (%)
                - found_at_iteration: BO iteration number
                - epoch_metrics: Per-epoch training history
                - timestamp: When checkpoint was saved
                - config_snapshot: Reproducibility information
                
    Raises:
        FileNotFoundError: If checkpoint file does not exist at the
            specified path. 
        ValueError: If checkpoint is missing required keys (model_state_dict,
            best_learning_rate, validation_loss, validation_accuracy).
        RuntimeError: If model architecture doesn't match the saved state
            dict (e.g., different number of classes or layers).
    
    Notes:
        - Model is automatically set to eval mode after loading
        - Uses map_location=config.DEVICE for GPU/CPU compatibility
        - Validates that checkpoint contains expected keys
    """

    best_model_path = os.path.join(
        config.RESULTS_DIR,
        "best_model.pth"
    )
    
    # Validate file existence
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model checkpoint not found at: {best_model_path}\n"
            f"Ensure Bayesian Optimization has completed and saved a model."
        )
    
    # Load checkpoint with full pickle support (weights_only=False)
    # Safe because checkpoint is created by this code (not external source)
    # Needed to load epoch_metrics and config_snapshot dictionaries
    checkpoint = torch.load(
        best_model_path,
        map_location=config.DEVICE,
        weights_only=False 
    )
    
    # Validate checkpoint structure
    required_keys = ['model_state_dict', 'best_learning_rate', 
                     'validation_loss', 'validation_accuracy']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}\n"
            f"Available keys: {list(checkpoint.keys())}"
        )
    
    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load state dict into model. "
            f"Architecture mismatch?\nOriginal error: {e}"
        )
    
    # Set model to evaluation mode (disables dropout, batchnorm in train mode)
    model.eval()
    
    return model, checkpoint


# =============================================================================
# RESULT SUMMARY
# =============================================================================

def save_final_summary(
    best_learning_rate: float,
    best_validation_loss: float,
    best_validation_accuracy: float,
    best_iteration: int,
    all_observations: List[Dict[str, Any]],
    total_time: float,
    test_accuracy: float
) -> None:
    """
    This creates a comprehensive, human-readable text file summarizing
    the entire BO experiment, including the best configuration found,
    a detailed table of all observations, runtime statistics, and
    experiment configuration parameters.
    
    The summary file is organized into four main sections:
        1. Best Configuration Found - optimal hyperparameters and metrics
        2. All Observations - complete table of every evaluation
        3. Statistics - timing and evaluation count summary
        4. Configuration - experiment setup and parameters
    
    Args:
        best_learning_rate (float): Best learning rate found
        best_validation_loss (float): Best validation loss achieved
        best_validation_accuracy (float): Best validation accuracy achieved
        best_iteration (int): Iteration where best result was found
        all_observations(List[Dict[str, Any]]): Complete list of all
            observations from ObservationLogger.get_all_observations().
            Each dict must contain: 'iteration', 'learning_rate',
            'validation_loss', 'validation_accuracy', 'training_time_seconds',
            'is_initial_design', 'is_best_so_far'.
        total_time (float): Total experiment runtime in seconds
        test_accuracy (float): test set accuracy with best LR
    
    Returns:
        None
    
    File Format:
        The generated text file contains:
            - Section headers with "===" and "---" dividers
            - Formatted table with columns: Iter, LR, Val Loss, Val Acc,
              Time(s), Type
            - Best observations marked with " *" suffix
            - All numeric values formatted for readability (fixed precision)
            - Timestamp of completion
    """
    summary_path = config.FINAL_SUMMARY_TXT
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BAYESIAN OPTIMIZATION - FINAL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Best result
        f.write("--- BEST CONFIGURATION FOUND ---\n")
        f.write(f"Learning Rate: {best_learning_rate:.6f}\n")
        f.write(f"Log10(LR): {np.log10(best_learning_rate):.4f}\n")
        f.write(f"Validation Loss: {best_validation_loss:.6f}\n")
        f.write(f"Validation Accuracy: {best_validation_accuracy:.4f}%\n")
        f.write(f"Found at Iteration: "
                f"{best_iteration}/{config.TOTAL_BUDGET}\n")
        
        if test_accuracy is not None:
            f.write(f"Test Accuracy: {test_accuracy:.4f}%\n")
        
        f.write("\n")
        
        # All observations
        f.write("--- ALL OBSERVATIONS ---\n")
        f.write(f"{'Iter':<6} {'LR':<12} {'Val Loss':<12} {'Val Acc':<12} "
                f"{'Time(s)':<10} {'Type':<10}\n")
        f.write("-" * 70 + "\n")
        
        for obs in all_observations:
            obs_type = "Initial" if obs['is_initial_design'] else "BO"
            marker = " *" if obs['is_best_so_far'] else ""
            
            f.write(
                f"{obs['iteration']:<6} "
                f"{obs['learning_rate']:<12.6f} "
                f"{obs['validation_loss']:<12.6f} "
                f"{obs['validation_accuracy']:<12.4f} "
                f"{obs['training_time_seconds']:<10.2f} "
                f"{obs_type:<10}{marker}\n"
            )
        
        f.write("\n")
        
        # Statistics
        f.write("--- STATISTICS ---\n")
        f.write(f"Total Evaluations: {len(all_observations)}\n")
        f.write(f"Initial Design: {config.N_INITIAL_DESIGN}\n")
        f.write(f"BO Iterations: {config.N_BO_ITERATIONS}\n")
        f.write(f"Total Runtime: {total_time:.2f} seconds "
                f"({total_time/60:.2f} minutes)\n")
        f.write(f"Average Time per Evaluation: "
                f"{total_time/len(all_observations):.2f} seconds\n")
        
        f.write("\n")
        
        # Configuration
        f.write("--- CONFIGURATION ---\n")
        f.write(f"Random Seed: {config.RANDOM_SEED}\n")
        f.write(f"Epochs per Evaluation: {config.N_EPOCHS}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"WEI Weight (w): {config.WEI_WEIGHT}\n")
        f.write(f"LR Search Space: [10^{config.LOG_LR_MIN}, "
                f"10^{config.LOG_LR_MAX}]\n")
        f.write(f"Device: {config.DEVICE}\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write(f"Experiment completed: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
    
    if config.VERBOSE:
        print(f"\n  Final summary saved to {summary_path}")


# =============================================================================
# UTILITY: Timing
# =============================================================================

@contextmanager
def timer(name: str, verbose: bool = True):
    """
    Context manager for timing code blocks.
    
    Measures elapsed time for any code block and optionally prints the
    result in a clean, formatted manner. Uses time.perf_counter() for
    high-resolution timing that is not affected by system clock adjustments.
    
    Args:
        name (str): Descriptive name for the timed operation. Will be
            displayed in the output if verbose=True.
        verbose (bool, optional): Whether to print timing information to
            stdout. If False, timing still occurs but no output is produced.
            Defaults to True.
    
    Yields:
        None: This context manager does not yield any value.
        
    Note:
        Timing is always measured but only printed if verbose=True.
        The timer does not interfere with exceptions - they propagate normally.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if verbose:
            elapsed = time.perf_counter() - start
            print(f"  {name}: {elapsed:.2f}s")


# =============================================================================
# UTILITY: ERROR HANDLING HELPERS
# =============================================================================

def is_nan_or_inf(tensor: torch.Tensor) -> bool:
    """
    This utility function performs a quick check for numerical instability
    by detecting Not-a-Number (NaN) or infinite (Inf) values in a tensor.
    It is commonly used for debugging gradient explosions, loss divergence,
    or other numerical issues during training.
    
    Args:
        tensor (torch.Tensor): PyTorch tensor to check. Can be of any shape
            and dtype (float16, float32, float64).
    
    Returns:
        bool: True if tensor contains at least one NaN or Inf value,
            False if all values are finite normal numbers.
    """
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def handle_cuda_oom(
    iteration: int, 
    epoch: int, 
    learning_rate: float, 
    batch_size: int
) -> None:
    """
    This function provides comprehensive debugging information when a CUDA
    out-of-memory (OOM) error occurs, including the execution context,
    current GPU memory usage, and actionable troubleshooting steps.
    
    The function prints a formatted diagnostic report but does not raise
    an exception itself - the caller is responsible for handling the error
    appropriately (e.g., by re-raising or attempting recovery).
    
    Args:
        iteration (int): Current BO iteration number (1-indexed) when
            the OOM error occurred.
        epoch (int): Current training epoch (1-indexed) when the OOM
            error occurred.
        learning_rate (float): Learning rate being used when the error
            occurred (helps identify if unusually high LR caused instability).
        batch_size (int): Batch size being used (primary cause of OOM errors).
    
    Returns:
        None
    """
    print("\n" + "="*70)
    print("CUDA OUT OF MEMORY ERROR")
    print("="*70)
    print(f"Iteration: {iteration}")
    print(f"Epoch: {epoch}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {config.DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: "
              f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: "
              f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("\nTroubleshooting:")
    print("- Reduce BATCH_SIZE in config.py")
    print("- Reduce model size (fewer ResNet layers)")
    print("- Use gradient checkpointing")
    print("="*70 + "\n")


# =============================================================================
# UTILITY: PRETTY PRINTING
# =============================================================================

def print_iteration_header(
    iteration: int, 
    total: int, 
    is_initial: bool = False
) -> None:
    """
    Print formatted header for each iteration.
    
    Creates a visually distinct section header to demarcate iterations
    in the console output, indicating the phase (Initial Design vs.
    Bayesian Optimization) and progress (current/total iterations).
    
    Args:
        iteration (int): Current iteration number, 1-indexed
            (e.g., 1 for first iteration).
        total (int): Total number of iterations in current phase or
            entire experiment (e.g., 10 for N_INITIAL_DESIGN + N_BO_ITERATIONS).
        is_initial (bool, optional): Whether this iteration is part of the
            initial design phase (Sobol sampling) or the BO phase (GP-guided).
            Defaults to False (BO phase).
    
    Returns:
        None
    """
    phase = "INITIAL DESIGN" if is_initial else "BAYESIAN OPTIMIZATION"
    
    print("\n" + "="*70)
    print(f"{phase} - Iteration {iteration}/{total}")
    print("="*70)


def print_iteration_summary(
    iteration: int,
    learning_rate: float,
    log_lr: float,
    val_loss: float,
    val_acc: float,
    training_time: float,
    is_best: bool
) -> None:
    """
    Print formatted summary after each iteration.
    
    Args:
        iteration: Current iteration number
        learning_rate: Evaluated learning rate
        log_lr: log10(learning_rate)
        val_loss: Validation loss achieved
        val_acc: Validation accuracy achieved
        training_time: Training time in seconds
        is_best: Whether this is the best result so far
    """
    best_marker = " --> NEW BEST!" if is_best else ""
    
    print(f"\n--- Iteration {iteration} Summary ---")
    print(f"Learning Rate: {learning_rate:.6f} (log10: {log_lr:.4f})")
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation Accuracy: {val_acc:.4f}%")
    print(f"Training Time: {training_time:.2f}s")
    print(best_marker)
    print()

