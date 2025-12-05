"""
Data loading and preprocessing for Fashion-MNIST dataset.

This module handles the complete data pipeline:
    - Fashion-MNIST dataset loading with normalization
    - Stratified train/validation split (50k/10k)
    - DataLoader creation with GPU optimization

Key Functions:
    - setup_data(): Complete pipeline returning train/val/test loaders
    - load_fashion_mnist(): Load raw Fashion-MNIST datasets
    - create_train_val_split(): Stratified 50k/10k split
    - create_dataloaders(): Configure PyTorch DataLoaders


Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
from typing import Tuple

# Third-party
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Local modules
import config

# =============================================================================
# DATA TRANSFORMATIONS
# =============================================================================

def get_transforms() -> transforms.Compose:
    """
    Get data normalization for Fashion-MNIST.
    
    Applies standard PyTorch preprocessing:
        1. Convert PIL image to tensor (scales to [0, 1])
        2. Normalize with mean=0.5, std=0.5 (maps to [-1, 1])
    
    Returns:
        transforms.Compose: Composed torchvision transforms.
    
    Note:
        Uses mean=0.5, std=0.5 (PyTorch convention for MNIST-style datasets),
        not empirical dataset statistics.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(config.MEAN,), std=(config.STD,))
    ])
    
    return transform


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_fashion_mnist(
    data_dir: str = config.DATA_DIR
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load Fashion-MNIST dataset.
    
    Fashion-MNIST contains:
        - 60,000 training images, 10,000 test images
        - 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, 
          Sandal, Shirt, Sneaker, Bag, Ankle boot)
        - 28x28 grayscale images
    
    Args:
        data_dir (str, optional): Directory to store/load data. 
            Defaults to config.DATA_DIR.
    
    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, test_dataset) with 
            normalization transforms applied.
    """
    # Training set (60k images)
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms()
    )
    
    # Test set (10k images)
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms()
    )
    
    if config.VERBOSE:
        print(f"  Fashion-MNIST loaded from {data_dir}")
        print(f"  Training set: {len(train_dataset)} images")
        print(f"  Test set: {len(test_dataset)} images")
    
    return train_dataset, test_dataset


# =============================================================================
# TRAIN/VALIDATION SPLIT
# =============================================================================

def create_train_val_split(
    full_train_dataset: torch.utils.data.Dataset,
    train_size: int = config.TRAIN_SIZE,
    val_size: int = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED
) -> Tuple[Subset, Subset]:
    """
    Split Fashion-MNIST training set into train and validation subsets.
    
    Uses stratified sampling to preserve class distribution. Default split:
        - Training: 50k images
        - Validation: 10k images
    
    Args:
        full_train_dataset (Dataset): Full training dataset (60k images).
        train_size (int, optional): Training subset size. 
            Defaults to config.TRAIN_SIZE.
        val_size (int, optional): Validation subset size. 
            Defaults to config.VAL_SIZE.
        seed (int, optional): Random seed for reproducible split. 
            Defaults to config.RANDOM_SEED.
    
    Returns:
        Tuple[Subset, Subset]: (train_subset, val_subset)
    """
    total_size = len(full_train_dataset)

    # Extract labels for stratification
    labels = full_train_dataset.targets.numpy()
    indices = list(range(total_size))    

    # Stratified split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        stratify=labels,
        random_state=seed
    )
    
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)   
    
    if config.VERBOSE:
        print(f"  Train/Val split created with seed {seed}")
        print(f"  Training: {len(train_subset)} images")
        print(f"  Validation: {len(val_subset)} images")

        train_labels = [full_train_dataset[i][1] for i in train_indices]
        val_labels = [full_train_dataset[i][1] for i in val_indices]
        print(f"  Train class distribution: {np.bincount(train_labels)}")
        print(f"  Val class distribution: {np.bincount(val_labels)}")

    return train_subset, val_subset


# =============================================================================
# DATALOADER CREATION
# =============================================================================

def create_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Configures batching, shuffling (train only), parallel loading,
    and pinned memory for GPU transfer.
    
    Args:
        train_dataset (Dataset): Training dataset or subset.
        val_dataset (Dataset): Validation dataset or subset.
        test_dataset (Dataset): Test dataset.
        batch_size (int, optional): Batch size for all loaders. 
            Defaults to config.BATCH_SIZE.
        num_workers (int, optional): Worker processes for data loading. 
            Defaults to config.NUM_WORKERS.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 
            (train_loader, val_loader, test_loader)
    """
    # Training loader (with shuffling)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=num_workers,
        pin_memory=config.USE_CUDA, 
        drop_last=False  
    )
    
    # Validation loader (no shuffling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=config.USE_CUDA,
        drop_last=False
    )
    
    # Test loader (no shuffling)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=config.USE_CUDA,
        drop_last=False
    )
    
    if config.VERBOSE:
        print("  DataLoaders created")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def setup_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Complete data setup pipeline for Fashion-MNIST.
    
    Pipeline:
        1. Load Fashion-MNIST (train and test)
        2. Split training set into train/val (50k/10k, stratified)
        3. Create DataLoaders for all splits
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 
            (train_loader, val_loader, test_loader)
    """
    
    # Load datasets
    full_train_dataset, test_dataset = load_fashion_mnist()
    
    # Split train into train/val
    train_subset, val_subset = create_train_val_split(full_train_dataset)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_subset,
        val_subset,
        test_dataset
    )
        
    return train_loader, val_loader, test_loader
