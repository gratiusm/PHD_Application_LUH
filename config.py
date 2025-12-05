"""
Configuration file for Bayesian Optimization of ResNet Learning Rate.

This module centralizes all hyperparameters, paths, and settings for the
Bayesian Optimization experiment optimizing ResNet-18 learning rate on
Fashion-MNIST.

Configuration Groups:
    - Random Seed: Reproducibility settings
    - BO Parameters: Budget, search space, WEI weight
    - Gaussian Process: Kernel, noise, optimization restarts
    - Acquisition: Multi-start optimization parameters
    - ResNet Training: Epochs, batch size, optimizer settings
    - Data: Train/val/test splits, normalization
    - Paths: Output directories for results and plots
    - Plotting: Visualization settings
    - Logging: Verbosity and progress tracking
    - Error Handling: NaN penalties, GP retry logic

Key Functions:
    - validate_config(): Comprehensive parameter validation
    - print_config(): Display current configuration

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
import os

# Third-party
import torch

# =============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42  # Fixed seed for reproducible results


# =============================================================================
# BAYESIAN OPTIMIZATION PARAMETERS
# =============================================================================

# Budget and Initial Design
N_INITIAL_DESIGN = 3 # Number of Sobol samples for initial design
N_BO_ITERATIONS = 7  # Number of BO iterations after initial design
TOTAL_BUDGET = N_INITIAL_DESIGN + N_BO_ITERATIONS  # Total: 10 evaluations

# Learning Rate Search Space (log-scale)
LOG_LR_MIN = -4  # log10(0.0001) = -4
LOG_LR_MAX = -1  # log10(0.1) = -1
LR_BOUNDS = [(LOG_LR_MIN, LOG_LR_MAX)]  # Bounds for optimization

# Weighted Expected Improvement Parameters
WEI_WEIGHT = 0.4  # w parameter: 0.5 corresponds to standard EI
MIN_DIST = 1e-4   # min distance to already evaluated points
                  # 0.0001 in log-space


# =============================================================================
# GAUSSIAN PROCESS PARAMETERS
# =============================================================================

# GP Kernel Configuration
GP_MATERN_NU = 2.5         # Nu parameter for Matern kernel (5/2)
GP_ALPHA = 1e-6            # Observation noise variance 
GP_NORMALIZE_Y = True      # Normalize targets before GP fitting
GP_N_RESTARTS = 5          # Number of restarts for hyperparameter optimization


# =============================================================================
# ACQUISITION FUNCTION PARAMETERS
# =============================================================================

ACQUISITION_RESTARTS = 1000   # 1000 is recommended by Sobester et al. (2005)


# =============================================================================
# RESNET TRAINING PARAMETERS
# =============================================================================

# Training Configuration
N_EPOCHS = 10           # Number of epochs per evaluation
BATCH_SIZE = 128        # Batch size for training
NUM_WORKERS = 4         # Number of workers for data loading

# Model Architecture
RESNET_LAYERS = [2, 2, 2, 2]  # ResNet-18 structure: [l1, l2, l3, l4]
NUM_CLASSES = 10              # Fashion-MNIST has 10 classes

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()


# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Fashion-MNIST Split
TRAIN_SIZE = 50000  # Training set size (from 60k)
VAL_SIZE = 10000    # Validation set size (from 60k)
TEST_SIZE = 10000   # Test set size (original test set)

# Data Normalization (PyTorch convention for MNIST datasets)
MEAN = 0.5  # Grayscale mean
STD = 0.5   # Grayscale std


# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================

# Base directory (current working directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Output directories
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Output file paths
OBSERVATIONS_CSV = os.path.join(RESULTS_DIR, "observations.csv")
TRAINING_LOGS_JSON = os.path.join(RESULTS_DIR, "training_logs.json")
FINAL_SUMMARY_TXT = os.path.join(RESULTS_DIR, "final_summary.txt")


# =============================================================================
# PLOTTING PARAMETERS
# =============================================================================

# Plot Configuration
PLOT_DPI = 300                    # High resolution for publication quality
PLOT_FIGSIZE = (12, 8)            # Figure size in inches
PLOT_START_ITERATION = 4          # Start plotting from iteration 4
PLOT_LR_LOG_SCALE = True          # Use log scale for learning rate axis
PLOT_TYPE = 'pdf'                 # used to define the file format of the plots
                                  # 'pdf', 'png'

# =============================================================================
# LOGGING AND VERBOSITY
# =============================================================================

# Verbosity Level
VERBOSE = True               # Print detailed information during execution
USE_TQDM = True              # Use progress bars for training epochs


# =============================================================================
# ERROR HANDLING
# =============================================================================

# NaN Detection
NAN_PENALTY = 1e6          # Penalty value for NaN losses

# GP Fitting Retry
GP_MAX_RETRIES = 3         # Max retries if GP fitting fails
GP_JITTER_MULTIPLIER = 10  # Multiply alpha on retry if fitting fails


# =============================================================================
# SCIENTIFIC REPRODUCIBILITY NOTES
# =============================================================================

"""
Reproducibility Settings:
- RANDOM_SEED: Set to 42 for all random number generators
- torch.backends.cudnn.deterministic: Set to True in main.py
- torch.backends.cudnn.benchmark: Set to False in main.py
- Sobol sequence: Scrambled for better space-filling properties

Hardware Information:
- Designed for: NVIDIA RTX 3090 (24GB VRAM)
- Tested on: CUDA 12.1, PyTorch 2.0+

"""


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """
    Validate configuration parameters for consistency.
    
    Raises:
        ValueError: If configuration parameters are inconsistent.
    """
    # Check budget
    assert TOTAL_BUDGET == N_INITIAL_DESIGN + N_BO_ITERATIONS, \
        "Total budget must equal initial design + BO iterations"
    
    assert N_INITIAL_DESIGN >= 1, \
        "N_INITIAL_DESIGN must be at least 1"
    
    assert N_BO_ITERATIONS >= 1, \
        "N_BO_ITERATIONS must be at least 1"
        
    # Check learning rate bounds
    assert LOG_LR_MIN < LOG_LR_MAX, \
        "LOG_LR_MIN must be less than LOG_LR_MAX"
    
    assert -10 <= LOG_LR_MIN <= 0, \
        "LOG_LR_MIN should be in reasonable range [-10, 0]"
    
    assert -10 <= LOG_LR_MAX <= 0, \
        "LOG_LR_MAX should be in reasonable range [-10, 0]"
    
    assert len(LR_BOUNDS) == 1 and len(LR_BOUNDS[0]) == 2, \
        "LR_BOUNDS must be a list with one tuple of (min, max)"
    
    # Check WEI weight
    assert 0 <= WEI_WEIGHT <= 1, \
        "WEI_WEIGHT must be in range [0, 1]"
    
    # Check GP Params
    assert GP_MATERN_NU in [0.5, 1.5, 2.5], \
        "GP_MATERN_NU must be 0.5, 1.5, or 2.5 for Matern kernel"
    
    assert GP_ALPHA > 0, \
        "GP_ALPHA must be positive"
        
    assert GP_N_RESTARTS >= 1, \
        "GP_N_RESTARTS must be at least 1"
    
    assert GP_N_RESTARTS <= 50, \
        "GP_N_RESTARTS > 50 may be unnecessarily slow"
    
    assert isinstance(GP_NORMALIZE_Y, bool), \
        "GP_NORMALIZE_Y must be a boolean"
    
    # Check Acquisition restarts
    assert ACQUISITION_RESTARTS > 0, \
        "ACQUISITION_RESTARTS must be positive"   
    
    # Check train params
    assert N_EPOCHS >= 1, \
        "N_EPOCHS must be at least 1"
    
    assert BATCH_SIZE > 0, \
        "BATCH_SIZE must be positive"
    
    assert BATCH_SIZE % 2 == 0, \
        "BATCH_SIZE should be a power of 2 for optimal GPU performance"
    
    assert NUM_WORKERS >= 0, \
        "NUM_WORKERS must be non-negative"
    
    assert len(RESNET_LAYERS) == 4, \
        "RESNET_LAYERS must have exactly 4 elements"
    
    assert all(layer > 0 for layer in RESNET_LAYERS), \
        "All RESNET_LAYERS must be positive integers"
        
    # Check data params
    assert TRAIN_SIZE > 0, \
        "TRAIN_SIZE must be positive"
        
    assert VAL_SIZE > 0, \
        "VAL_SIZE must be positive"
        
    assert TRAIN_SIZE + VAL_SIZE == 60000, \
        "Train + Val must equal 60000 (Fashion-MNIST training set)"
        
    assert TEST_SIZE == 10000, \
        "TEST_SIZE must be 10000 (Fashion-MNIST test set)"
    
    assert TRAIN_SIZE >= BATCH_SIZE, \
        "TRAIN_SIZE must be at least as large as BATCH_SIZE"
    
    assert VAL_SIZE >= BATCH_SIZE, \
        "VAL_SIZE must be at least as large as BATCH_SIZE"
        
    # Check normalization values
    assert isinstance(MEAN, (int, float)), \
        "MEAN must be a number"
    
    assert isinstance(STD, (int, float)), \
        "STD must be a number"
        
    assert STD > 0, \
        "STD must be positive"
        
    # Check plotting iteration
    assert PLOT_START_ITERATION >= N_INITIAL_DESIGN + 1, \
        "Plotting should start after initial design + at least 1 BO iteration"
        
    assert PLOT_DPI > 0, \
        "PLOT_DPI must be positive"
    
    assert len(PLOT_FIGSIZE) == 2, \
        "PLOT_FIGSIZE must be a tuple of (width, height)"
    
    assert all(s > 0 for s in PLOT_FIGSIZE), \
        "PLOT_FIGSIZE dimensions must be positive"
    
    assert isinstance(PLOT_LR_LOG_SCALE, bool), \
        "PLOT_LR_LOG_SCALE must be a boolean"
    
    # Check error handling
    assert NAN_PENALTY > 0, \
        "NAN_PENALTY must be positive"
    
    assert GP_MAX_RETRIES >= 1, \
        "GP_MAX_RETRIES must be at least 1"
    
    assert GP_JITTER_MULTIPLIER > 1, \
        "GP_JITTER_MULTIPLIER must be greater than 1"
    
    # Verbosity check
    assert isinstance(VERBOSE, bool), \
        "VERBOSE must be a boolean"
    
    assert isinstance(USE_TQDM, bool), \
        "USE_TQDM must be a boolean"
    
    # Path check
    assert os.path.isdir(DATA_DIR), \
        f"DATA_DIR does not exist: {DATA_DIR}"
    
    assert os.path.isdir(PLOTS_DIR), \
        f"PLOTS_DIR does not exist: {PLOTS_DIR}"
    
    assert os.path.isdir(RESULTS_DIR), \
        f"RESULTS_DIR does not exist: {RESULTS_DIR}"
        
    print("  Configuration validated successfully")


# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

def print_config():
    """
    Print current configuration for verification.
    """
    print("\n" + "="*55)
    print("BAYESIAN OPTIMIZATION CONFIGURATION")
    print("="*55)
    
    print("\n--- Optimization Settings ---")
    print(f"  Total Budget: {TOTAL_BUDGET} evaluations")
    print(f"  Initial Design: {N_INITIAL_DESIGN} Sobol samples")
    print(f"  BO Iterations: {N_BO_ITERATIONS}")
    print(f"  LR Search Space: [10^{LOG_LR_MIN}, 10^{LOG_LR_MAX}] = "
          f"  [{10**LOG_LR_MIN}, {10**LOG_LR_MAX}]")
    print(f"  WEI Weight (w): {WEI_WEIGHT}")
    
    print("\n--- GP Configuration ---")
    print(" Kernel: MATERN")
    print(f"  Matern Nu: {GP_MATERN_NU}")
    print(f"  Alpha (Jitter): {GP_ALPHA}")
    print(f"  N Restarts: {GP_N_RESTARTS}")
    
    print("\n--- Training Configuration ---")
    print(f"  Epochs per Evaluation: {N_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print("  Optimizer: Plain SGD")
    print(f"  Device: {DEVICE}")
    
    print("\n--- ResNet-18 for Fashion-MNIST ---")
    print(f"  Architecture: {RESNET_LAYERS} blocks per layer")
    print(f"  Output classes: {NUM_CLASSES}")
    print(f"  Device: {DEVICE}")
    
    print("\n--- Data Configuration ---")
    print(f"  Train: {TRAIN_SIZE}, Val: {VAL_SIZE}, Test: {TEST_SIZE}")
    
    print("\n--- Reproducibility ---")
    print(f"  Random Seed: {RANDOM_SEED}")
    
    print("\n--- Output Directories ---")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    
    # Check CUDA availability
    print("\n--- Device Information ---")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if USE_CUDA:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {gpu_mem / 1024**3:.2f} GB")
    print(f"  Using Device: {DEVICE}")


# =============================================================================
# RUN VALIDATION ON IMPORT
# =============================================================================

if __name__ == "__main__":
    validate_config()
    print_config()
    
    