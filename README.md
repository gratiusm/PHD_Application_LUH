# Bayesian Optimization for Learning Rate 

**PhD Application**  
Gottfried Wilhelm Leibniz Universität Hannover

---

## Overview

This repository implements Bayesian Optimization (BO) with Weighted Expected Improvement (WEI) to find the optimal learning rate for a ResNet-18 trained on Fashion-MNIST through minimising the loss.

**Key Components:**
- **Initial Design:** 3 Sobol quasi-random samples
- **Surrogate Model:** Gaussian Process with Matern 5/2 kernel
- **Acquisition Function:** Weighted Expected Improvement (Sobester et al., 2005)
- **Neural Network:** ResNet-18 (modified for 28×28 grayscale images)
- **Dataset:** Fashion-MNIST (60k train, 10k test)
- **Budget:** 10 function evaluations (3 initial + 7 BO iterations)

---

## Requirements

### System Requirements
- **Python:** 3.11+
- **GPU:** Recommended (CUDA-compatible), but CPU is supported
- **RAM:** 8GB minimum, 16GB recommended
- **Disk Space:** ~5GB (for data, checkpoints, results, Environment)

### Testet on
- **Python:** 3.11.6
- **Cuda:** 12.1
- **GPU:** Nvidea RTX 3090 24 GB
- **CPU:** i9-12900K (3.19 GHz)
- **RAM:** 64 GB
- **OS:** Windows 11 Pro
---

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Usage

### How to run

Simply run the main script:
```bash
python main.py
```

This will:
1. Download Fashion-MNIST (if not already present)
2. Run Bayesian Optimization (10 evaluations)
3. Generate 7 plots (iterations 4-10)
4. Save results to `results/` directory
5. Save the best model to `results/` directory

**Expected Runtime:**
- With GPU (RTX 3090): ~60 minutes
- With CPU (16 Cores, 3.19 Ghz): ~6 hours

### Configuration

All hyperparameters can be modified in `config.py`:
```python

# Bayesian Optimization
N_INITIAL_DESIGN = 3          # Sobol samples
N_BO_ITERATIONS = 7           # BO iterations

# Acquisition function
WEI_WEIGHT = 0.4              # w parameter for WEI
MIN_DIST = 1e-4               # min distance to already evaluated points

# Learning Rate Search Space
LOG_LR_MIN = -4               # log10(0.0001)
LOG_LR_MAX = -1               # log10(0.1)

# ResNet-18 Training
N_EPOCHS = 10                 # Epochs per evaluation
BATCH_SIZE = 128              # Batch size
NUM_WORKERS = 4               # Number of workers for data loading

# Random Seed
RANDOM_SEED = 42              # For reproducibility

# Fashion-MNIST Split
TRAIN_SIZE = 50000            # Training set size (from 60k)
VAL_SIZE = 10000              # Validation set size (from 60k)
```

---

## Outputs

### Generated Files

After running the experiment, the following outputs are created:

#### 1. Plots (`plots/` directory)

- `iteration_04.png` through `iteration_10.png`: BO iteration plots (7 total)
- `convergence.png`: Best value vs iteration
- `all_observations.png`: Scatter plot of all evaluations
- `summary.png`: 4-panel summary visualization

Each iteration plot contains:
- GP posterior mean and uncertainty
- All observations (initial design vs BO)
- Best point so far
- Acquisition function (WEI)
- Next point to evaluate

#### 2. Results (`results/` directory)

- `observations.csv`: All evaluations with metadata
- `training_logs.json`: Detailed per-epoch training information
- `final_summary.txt`: Human-readable summary report
- `best_model.pth`: Best model save (was to big for uploading)

---

## Methodology

### Bayesian Optimization Algorithm

1. **Initial Design (Iterations 1-3):**
   - Generate 3 quasi-random samples using Sobol sequence
   - Evaluate objective function at each point
   - Build initial dataset D = {(x₁, y₁), (x₂, y₂), (x₃, y₃)}

2. **BO Loop (Iterations 4-10):**
```
   For each iteration t = 4 to 10:
       a) Fit Gaussian Process on all observations D
       b) Optimize kernel hyperparameters via MLE
       c) Compute WEI acquisition function
       d) Find x_next = argmax WEI(x)
       e) Evaluate y_next = f(x_next)
       f) Update D ← D ∪ {(x_next, y_next)}
       g) Saving the best model so far
       h) Generate visualization plot
```

3. **Final Evaluation:**
   - Loading the model with best learning rate 
   - Evaluate on test set

### Weighted Expected Improvement (WEI)

Formula (Sobester et al., 2005):
```
WEI(x) = w × (y_min - μ(x)) × Φ(Z) + (1-w) × σ(x) × φ(Z)

where:
  Z = (y_min - μ(x)) / σ(x)
  μ(x) = GP posterior mean
  σ(x) = GP posterior standard deviation
  Φ(z) = Standard normal CDF
  φ(z) = Standard normal PDF
  w ∈ [0,1] = Weight parameter

and w = 1 is Pure exploration (focus on high uncertainty)
and w = 0 is Pure exploitation (focus on low predicted mean)
```

### ResNet Architecture

Modified ResNet-18 for Fashion-MNIST:
- **Input:** 1 channel (grayscale) instead of 3 (RGB)
- **First Conv:** 3×3 kernel, stride=1 (not 7×7, stride=2)
- **No initial MaxPooling** (preserves spatial information for 28×28 images)
- **Output:** 10 classes
- **Parameters:** ~11M trainable parameters

### Training Configuration

- **Optimizer:** Plain SGD 
- **Loss:** CrossEntropyLoss
- **Epochs:** 10 per evaluation
- **Batch Size:** 128
- **Data Split:** 50k train / 10k validation / 10k test

---

## Final Results

### final_summary.txt:

```
======================================================================
BAYESIAN OPTIMIZATION - FINAL SUMMARY
======================================================================

--- BEST CONFIGURATION FOUND ---
Learning Rate: 0.055553
Log10(LR): -1.2553
Validation Loss: 0.260240
Validation Accuracy: 92.2900%
Found at Iteration: 6/10
Test Accuracy: 91.2800%

--- ALL OBSERVATIONS ---
Iter   LR           Val Loss     Val Acc      Time(s)    Type      
----------------------------------------------------------------------
1      0.001964     0.378571     88.2100      355.65     Initial    *
2      0.008003     0.336234     90.8700      354.88     Initial    *
3      0.028319     0.303149     92.1400      354.63     Initial    *
4      0.100000     0.301255     88.9600      355.55     BO         *
5      0.000100     0.603998     80.2800      356.44     BO        
6      0.055553     0.260240     92.2900      359.17     BO         *
7      0.048002     0.262372     92.0200      356.68     BO        
8      0.000595     0.417396     86.0900      356.43     BO        
9      0.004251     0.359242     89.6900      356.20     BO        
10     0.015037     0.319149     91.7700      355.19     BO        

--- STATISTICS ---
Total Evaluations: 10
Initial Design: 3
BO Iterations: 7
Total Runtime: 3614.59 seconds (60.24 minutes)
Average Time per Evaluation: 361.46 seconds

--- CONFIGURATION ---
Random Seed: 42
Epochs per Evaluation: 10
Batch Size: 128
WEI Weight (w): 0.4
LR Search Space: [10^-4, 10^-1]
Device: cuda

```

### observations.csv:

```
iteration,learning_rate,log_learning_rate,validation_loss,validation_accuracy,training_time_seconds,is_initial_design,is_best_so_far
1,0.001964,-2.7069,0.378571,88.2100,355.65,True,True
2,0.008003,-2.0968,0.336234,90.8700,354.88,True,True
3,0.028319,-1.5479,0.303149,92.1400,354.63,True,True
4,0.100000,-1.0000,0.301255,88.9600,355.55,False,True
5,0.000100,-4.0000,0.603998,80.2800,356.44,False,False
6,0.055553,-1.2553,0.260240,92.2900,359.17,False,True
7,0.048002,-1.3187,0.262372,92.0200,356.68,False,False
8,0.000595,-3.2253,0.417396,86.0900,356.43,False,False
9,0.004251,-2.3715,0.359242,89.6900,356.20,False,False
10,0.015037,-1.8228,0.319149,91.7700,355.19,False,False

```

---
## Troubleshooting


#### Data Download Issues

**Error:** Fashion-MNIST download fails

**Solution:**
- Manually download from: https://github.com/zalandoresearch/fashion-mnist
- Place in `data/FashionMNIST/` directory

The dataset should look like this:
```
.
├── /data/FashionMNIST/raw/train-images-idx3-ubyte
├── /data/FashionMNIST/raw/train-images-idx3-ubyte.gz
├── /data/FashionMNIST/raw/train-labels-idx1-ubyte
├── /data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
├── /data/FashionMNIST/raw/t10k-images-idx3-ubyte
├── /data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
├── /data/FashionMNIST/raw/t10k-labels-idx1-ubyte
└── /data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
```

---

## Repository Structure
```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                      # Entry point
├── config.py                    # Configuration parameters
├── utils.py                     # Utility functions (logging, checkpoints)
├── data.py                      # Fashion-MNIST loading
├── model.py                     # ResNet architecture
├── trainer.py                   # Training and evaluation
├── gaussian_process.py          # GP surrogate model
├── acquisition.py               # WEI implementation
├── bayesian_optimization.py     # BO main algorithm
├── plotting.py                  # Visualization
├── plots/                       # Generated plots
└── results/                     # Logs and summaries
```

---
## License

This project is submitted as part of a PhD application and is intended for academic evaluation purposes.

