"""
Training and evaluation logic for ResNet on Fashion-MNIST.

This module implements the complete training pipeline:
    - Single-epoch training with NaN/Inf detection and CUDA OOM handling
    - Evaluation on validation/test sets
    - Full training procedure with metric tracking
    - Objective function for Bayesian Optimization (black-box function)
    - Final model evaluation on test set

Key Functions:
    - train_one_epoch(): Single epoch with progress tracking
    - evaluate(): Validation/test set evaluation
    - train_resnet(): Complete training for given learning rate
    - objective_function(): BO objective (trains model, returns val loss)
    - evaluate_final_model(): Load best checkpoint and evaluate on test

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
import math
import time
from typing import Tuple, Dict, Any

# Third-party
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local modules
import config
from model import create_resnet_fashion_mnist
import utils

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_tqdm: bool = config.USE_TQDM
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Performs complete training loop with forward/backward passes,
    NaN/Inf detection, optional progress bar, and CUDA OOM handling.
    
    Args:
        model (nn.Module): ResNet model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., SGD).
        device (torch.device): Device to train on.
        epoch (int): Current epoch number (for logging).
        use_tqdm (bool, optional): Show progress bar. 
            Defaults to config.USE_TQDM.
    
    Returns:
        Tuple[float, float]: (average_loss, accuracy_percentage)
    
    Raises:
        RuntimeError: If NaN/Inf loss or gradients detected, or CUDA OOM.
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create iterator
    if use_tqdm:
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.N_EPOCHS}",
            leave=False
        )
    else:
        iterator = train_loader
    
    try:
        for batch_idx, (images, labels) in enumerate(iterator):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Check for NaN loss
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                raise RuntimeError(
                    f"NaN or Inf loss detected at epoch {epoch}, "
                    f"batch {batch_idx}. "
                    f"Loss value: {loss.item()}"
                )
            
            # Backward pass
            loss.backward()
            
            # Comprehensive gradient check (for debugging)
            for name, param in model.named_parameters():
                if param.grad is not None and utils.is_nan_or_inf(param.grad):
                    raise RuntimeError(
                        f"NaN or Inf gradient detected in {name} "
                        f"at epoch {epoch}"
                    )
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            if use_tqdm:
                iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.0 * correct / total:.2f}%"
                })
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            # CUDA OOM - provide detailed error information
            utils.handle_cuda_oom(
                iteration=-1,  # Unknown iteration in this context
                epoch=epoch,
                learning_rate=optimizer.param_groups[0]['lr'],
                batch_size=config.BATCH_SIZE
            )
        raise e
    
    # Compute epoch statistics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on validation or test set.
    
    Args:
        model (nn.Module): Model to evaluate.
        data_loader (DataLoader): Validation or test data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.
    
    Returns:
        Tuple[float, float]: (average_loss, accuracy_percentage)
    
    Note:
        Uses @torch.no_grad() for memory efficiency during inference.
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in data_loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Compute statistics
    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# =============================================================================
# FULL TRAINING PROCEDURE
# =============================================================================

def train_resnet(
    learning_rate: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = config.N_EPOCHS,
    device: torch.device = config.DEVICE,
    verbose: bool = config.VERBOSE
) -> Dict[str, Any]:
    """
    Complete training procedure for ResNet with given learning rate.
    Creates fresh model, trains for n_epochs, and tracks all metrics.
    
    Args:
        learning_rate: Learning rate for SGD optimizer.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        n_epochs: Number of training epochs.
        device: Device to train on.
        verbose: Whether to print detailed information.
    
    Args:
        learning_rate (float): Learning rate for SGD optimizer.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        n_epochs (int, optional): Number of training epochs. 
            Defaults to config.N_EPOCHS.
        device (torch.device, optional): Device to train on. 
            Defaults to config.DEVICE.
        verbose (bool, optional): Print detailed information. 
            Defaults to config.VERBOSE.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - final_val_loss (float): Final validation loss
            - final_val_accuracy (float): Final validation accuracy (%)
            - epoch_train_losses (List[float]): Training losses per epoch
            - epoch_val_losses (List[float]): Validation losses per epoch
            - epoch_val_accuracies (List[float]): Validation accuracies per 
               epoch
            - training_time (float): Total training time in seconds
            - model_state_dict (Dict): Trained model state dictionary
            - optimizer_state_dict (Dict): Final optimizer state
    
    Raises:
        RuntimeError: If training fails (NaN loss, CUDA OOM, etc.).
    """
    start_time = time.time()
    
    # Create fresh model
    model = create_resnet_fashion_mnist(device=device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Plain SGD optimizer 
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )
    
    if verbose:
        print(f"\nTraining ResNet with LR={learning_rate:.6f}\n")
    
    # Tracking lists
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accuracies = []
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_tqdm=config.USE_TQDM
        )
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Store metrics
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_accuracies.append(val_acc)
        
        # Print epoch summary
        if verbose:
            print(
                f"Epoch {epoch}/{n_epochs}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
            )
    
    # Training complete
    training_time = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {training_time:.2f}s")
    
    # Return comprehensive results
    return {
        'final_val_loss': epoch_val_losses[-1],
        'final_val_accuracy': epoch_val_accuracies[-1],
        'epoch_train_losses': epoch_train_losses,
        'epoch_val_losses': epoch_val_losses,
        'epoch_val_accuracies': epoch_val_accuracies,
        'training_time': training_time,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }


# =============================================================================
# OBJECTIVE FUNCTION FOR BAYESIAN OPTIMIZATION
# =============================================================================

def objective_function(
    log_learning_rate: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = config.N_EPOCHS,
    device: torch.device = config.DEVICE
) -> Tuple[float, Dict[str, Any]]:
    """
    Objective function for Bayesian Optimization.
    
    The expensive black-box function that BO optimizes. Trains ResNet
    with given learning rate and returns validation loss to minimize.
    
    Workflow:
        1. Transform log_lr to actual LR: lr = 10^log_lr
        2. Train ResNet for n_epochs
        3. Return validation loss and detailed results
    
    Args:
        log_learning_rate (float): Learning rate in log10 space
            (e.g., -3.0 for lr=0.001).
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        n_epochs (int, optional): Number of training epochs. 
            Defaults to config.N_EPOCHS.
        device (torch.device, optional): Device to train on. 
            Defaults to config.DEVICE.
    
    Returns:
        Tuple[float, Optional[Dict[str, Any]]]: 
            - validation_loss: Final validation loss (objective to minimize)
            - detailed_results: Dict with model state and metrics, or None if 
              training failed
    """
    # Transform from log space to actual learning rate
    learning_rate = 10 ** log_learning_rate
        
    try:
        # Train model with this learning rate
        results = train_resnet(
            learning_rate=learning_rate,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            device=device,
            verbose=config.VERBOSE
        )
        
        # Extract validation loss (objective to minimize)
        validation_loss = results['final_val_loss']
        
        # Check for invalid loss values
        # Anything > 10.0 indicates training failure
        if math.isnan(validation_loss) or validation_loss > 10:
            if config.VERBOSE:
                print(f"WARNING: Invalid validation loss: {validation_loss}")
                print(f"Returning penalty value: {config.NAN_PENALTY}")
            return config.NAN_PENALTY, None
        
        return validation_loss, results
    
    except RuntimeError as e:
        # Handle training failures (NaN, OOM, etc.)
        print(f"\n{'='*70}")
        print(f"ERROR: Training failed for LR={learning_rate:.6f}")
        print(f"Error message: {str(e)}")
        print(f"Returning penalty value: {config.NAN_PENALTY}")
        print(f"{'='*70}\n")
        
        return config.NAN_PENALTY, None
    
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\n{'='*70}")
        print("UNEXPECTED ERROR during training")
        print(f"Learning Rate: {learning_rate:.6f}")
        print(f"Error: {str(e)}")
        print(f"Returning penalty value: {config.NAN_PENALTY}")
        print(f"{'='*70}\n")
        
        return config.NAN_PENALTY, None


# =============================================================================
# FINAL MODEL EVALUATION ON TEST SET
# =============================================================================

def evaluate_final_model(
    test_loader: DataLoader,
    device: torch.device = config.DEVICE
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evaluate final model on test set.
    
    Loads best model from Bayesian Optimization and evaluates on test set.
    All metadata (learning rate, validation metrics) is read from checkpoint.
    
    Args:
        test_loader (DataLoader): Test data loader.
        device (torch.device, optional): Device to evaluate on. 
            Defaults to config.DEVICE.
    
    Returns:
        Tuple[float, float, Dict[str, Any]]: 
            - test_loss: Test set loss
            - test_accuracy: Test set accuracy (%)
            - model_info: Dict with learning_rate, found_at_iteration, 
              validation_loss, validation_accuracy, timestamp, 
              config_snapshot, epoch_metrics
    
    Raises:
        FileNotFoundError: If best model checkpoint doesn't exist.
    """  
    
    # Create model architecture
    model = create_resnet_fashion_mnist(device=device)
    
    # Load best model 
    try:
        model, metadata = utils.load_best_model(model)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Best model not found. "
            f"Ensure Bayesian Optimization completed successfully.\n"
            f"Original error: {e}"
        )    
        
    # Extract metadata from checkpoint
    model_info = {
        'learning_rate': metadata['best_learning_rate'],
        'found_at_iteration': metadata['found_at_iteration'],
        'validation_loss': metadata['validation_loss'],
        'validation_accuracy': metadata['validation_accuracy'],
        'timestamp': metadata.get('timestamp', 'unknown'),
        'config_snapshot': metadata.get('config_snapshot', {}),
        'epoch_metrics': metadata.get('epoch_metrics')
    }
        
    if config.VERBOSE:
        print("  Best model loaded successfully")
        print(f"  Learning Rate: {model_info['learning_rate']:.6f}")
        print(f"  Found at iteration: {model_info['found_at_iteration']}")
        print(f"  Validation loss: {model_info['validation_loss']:.6f}")
        print(f"  Validation accuracy: "
              f"{model_info['validation_accuracy']:.2f}%")
    
    if config.VERBOSE:
        print("\nEvaluating on test set...")
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )
        
    return test_loss, test_accuracy, model_info

