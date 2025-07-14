"""
Training module for MedVision Classification
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pathlib import Path
from typing import Dict, Any, Optional

from .helpers import setup_logging, load_config, create_output_dirs


def setup_callbacks(config: Dict[str, Any]) -> list:
    """Setup training callbacks"""
    callbacks = []
    
    # Get callbacks from training config
    training_config = config.get("training", {})
    
    # Early stopping
    if "early_stopping" in training_config:
        es_config = training_config["early_stopping"]
        callbacks.append(EarlyStopping(
            monitor=es_config.get("monitor", "val/val_loss"),
            patience=es_config.get("patience", 10),
            mode=es_config.get("mode", "min"),
            verbose=True
        ))
    
    # Model checkpoint
    if "model_checkpoint" in training_config:
        mc_config = training_config["model_checkpoint"]
        checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
        callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=mc_config.get("monitor", "val/val_accuracy"),
            mode=mc_config.get("mode", "max"),
            save_top_k=mc_config.get("save_top_k", 3),
            filename=mc_config.get("filename", "epoch_{epoch:02d}-val_acc_{val/val_accuracy:.3f}"),
            verbose=True
        ))
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    
    return callbacks


def setup_logger(config: Dict[str, Any]):
    """Setup logger"""
    logging_config = config.get("logging", {})
    logger_type = logging_config.get("logger", "tensorboard")
    
    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=logging_config.get("save_dir", "outputs/logs"),
            name=logging_config.get("name", "medvision_cls"),
            version=logging_config.get("version", None)
        )
    elif logger_type == "wandb":
        wandb_config = logging_config.get("wandb", {})
        return WandbLogger(
            project=wandb_config.get("project", "medvision-classification"),
            entity=wandb_config.get("entity", None),
            tags=wandb_config.get("tags", []),
            save_dir=logging_config.get("save_dir", "outputs/logs"),
            name=logging_config.get("name", "medvision_cls"),
            version=logging_config.get("version", None)
        )
    else:
        return None


def train_model(
    config_file: str,
    resume_checkpoint: Optional[str] = None,
    debug: bool = False
):
    """
    Train a classification model
    
    Args:
        config_file: Path to configuration file
        resume_checkpoint: Path to checkpoint to resume from
        debug: Enable debug mode
    """
    # Import here to avoid circular imports
    from ..models import ClassificationLightningModule
    from ..datasets import get_datamodule
    
    # Load configuration
    config = load_config(config_file)
    
    # Setup logging
    setup_logging(debug=debug)
    
    # Set seed
    pl.seed_everything(config.get("seed", 42))
    
    # Create output directories
    create_output_dirs(config.get("paths", {}))
    
    # Setup data module
    data_config = config.get("data", {})
    data_module = get_datamodule(data_config)
    
    # Setup data module to get class info for training
    data_module.setup("fit")
    
    # Setup model
    model_config = config.get("model", {})
    network_config = model_config.get("network", {}).copy()
    
    # Extract specific parameters to avoid duplicate keyword arguments
    model_name = network_config.pop("name", "resnet50")
    pretrained = network_config.pop("pretrained", True)
    
    model = ClassificationLightningModule(
        model_name=model_name,
        num_classes=data_module.num_classes,
        pretrained=pretrained,
        loss_config=model_config.get("loss", {}),
        optimizer_config=model_config.get("optimizer", {}),
        scheduler_config=model_config.get("scheduler", {}),
        metrics_config=model_config.get("metrics", {}),
        **network_config
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Setup logger
    logger = setup_logger(config)
    
    # Setup trainer
    training_config = config.get("training", {})
    
    # Check if model is 3D to determine deterministic setting
    is_3d_model = "3d" in model_name.lower()
    
    trainer = pl.Trainer(
        max_epochs=training_config.get("max_epochs", 100),
        accelerator=training_config.get("accelerator", "gpu"),
        devices=training_config.get("devices", 1),
        precision=training_config.get("precision", 16),
        log_every_n_steps=config.get("logging", {}).get("log_every_n_steps", 10),
        val_check_interval=config.get("validation", {}).get("check_val_every_n_epoch", 1),
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=logger,
        deterministic=not is_3d_model,  # Disable deterministic for 3D models
    )
    
    # Start training
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)
    
    # Test best model if test data is available
    if data_module.test_dataset is not None:
        trainer.test(model, data_module, ckpt_path="best")
    
    return trainer, model
