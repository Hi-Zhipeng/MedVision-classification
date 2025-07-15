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
    create_output_dirs(config.get("outputs", {}))
    
    # Setup data module
    data_config = config.get("data", {})
    data_module = get_datamodule(data_config)
    
    # Setup data module to get class info for training
    data_module.setup("fit")
    
    # Setup model
    model_config = config.get("model", {})
    network_config = model_config.get("network", {}).copy()

    # Task-dim
    task_dim = config.get("task_dim")

    if task_dim is None:
        print("No task_dim specified")
        return
    
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
    
    # Handle devices configuration
    devices = training_config.get("devices", 1)
    if isinstance(devices, list):
        num_devices = len(devices)
    elif isinstance(devices, int):
        num_devices = devices
    else:
        num_devices = 1
    
    trainer = pl.Trainer(
        max_epochs=training_config.get("max_epochs", 100),
        accelerator=training_config.get("accelerator", "gpu"),
        devices=devices,
        precision=training_config.get("precision", 16),
        log_every_n_steps=config.get("logging", {}).get("log_every_n_steps", 10),
        check_val_every_n_epoch=config.get("validation", {}).get("check_val_every_n_epoch", 1),
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=logger,
        deterministic=not is_3d_model,  # Disable deterministic for 3D models
        # 跳过 sanity check 避免空样本问题
        num_sanity_val_steps=0,
        # 启用指标聚合
        enable_progress_bar=True
    )
    
    # Start training
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)

    # Save training results
    train_results = trainer.logged_metrics

    test_results = trainer.test(model, data_module, ckpt_path="best")

    save_metrics = config["training"].get("save_metrics", True)
    
    if save_metrics:
        import json

        # 提取 best checkpoint callback
        best_ckpt_cb = None
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_ckpt_cb = cb
                break

        # 提取 train/val/test 指标
        train_val_metrics = {
            k: float(v) for k, v in train_results.items()
            if isinstance(v, torch.Tensor) and (k.startswith("val/") or k.startswith("train/"))
        }

        test_metrics = {
            k: float(v) for k, v in test_results[0].items()
        } if test_results else {}

        # 汇总结果
        final_metrics = {
            "train_val_metrics": train_val_metrics,
            "test_metrics": test_metrics,
            "best_model_path": best_ckpt_cb.best_model_path if best_ckpt_cb else None,
            "best_model_score": float(best_ckpt_cb.best_model_score) if best_ckpt_cb and best_ckpt_cb.best_model_score is not None else None
        }

        # 保存 JSON 文件
        result_path = os.path.join(config.get("outputs")["output_dir"], "results.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(final_metrics, f, indent=4)

        print(f"✅ Final metrics saved to: {result_path}")

    return trainer, model
