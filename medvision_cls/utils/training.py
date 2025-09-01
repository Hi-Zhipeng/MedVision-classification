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
        # 基于output_dir拼接checkpoint目录
        output_dir = config.get("outputs", {}).get("output_dir", "outputs")
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        monitor_metric = mc_config.get("monitor", "val/accuracy")

        callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=monitor_metric,
            mode=mc_config.get("mode", "max"),
            save_top_k=mc_config.get("save_top_k", 1),
            filename=f"{config['training'].get('experiment_name')}",
            verbose=True
        ))   
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    
    return callbacks


def setup_logger(config: Dict[str, Any]):
    """Setup logger"""
    logging_config = config.get("logging", {})
    logger_type = logging_config.get("logger", "tensorboard")
    
    # 基于output_dir拼接log目录
    output_dir = config.get("outputs", {}).get("output_dir", "outputs")
    log_dir = os.path.join(output_dir, "logs")
    
    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=log_dir,
            name=logging_config.get("name", "medvision_cls"),
            version=logging_config.get("version", None)
        )
    elif logger_type == "wandb":
        wandb_config = logging_config.get("wandb", {})
        return WandbLogger(
            project=wandb_config.get("project", "medvision-classification"),
            entity=wandb_config.get("entity", None),
            tags=wandb_config.get("tags", []),
            save_dir=log_dir,
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
    # data_module.setup("fit")
    
    # Setup model
    model_config = config.get("model", {})

    model = ClassificationLightningModule(
        model_config=model_config,
        loss_config=model_config.get("loss", {}),
        optimizer_config=model_config.get("optimizer", {}),
        scheduler_config=model_config.get("scheduler", {}),
        metrics_config=model_config.get("metrics", {})
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Setup logger
    logger = setup_logger(config)
    
    # Setup trainer
    training_config = config.get("training", {})
    
    # Check if model is 3D to determine deterministic setting
    task_dim = config.get("task_dim", "")

    if task_dim == "":
        return "Error: task_dim is not set in the config file."

    # Handle devices configuration
    devices = training_config.get("devices", -1)
    
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
        num_sanity_val_steps=0,
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
            "best_model_score": float(best_ckpt_cb.best_model_score) if best_ckpt_cb and best_ckpt_cb.best_model_score is not None else None,
            "monitor": config.get("training", {}).get("model_checkpoint", {}).get("monitor", "val/accuracy"),
        }

    # ONNX Export after training - 转换所有保存的top-k模型
    convert_to_onnx = config.get("training", {}).get("export_onnx", True)
    converted_models = []
    onnx_dir = None
    
    if convert_to_onnx:
        print("\n🔄 Starting ONNX conversion for all saved models...")
        try:
            from .onnx_export import convert_models_to_onnx
            from ..models.lightning_module import ClassificationLightningModule
            
            # 找到ModelCheckpoint callback
            checkpoint_callback = None
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    checkpoint_callback = cb
                    break
            
            if checkpoint_callback:
                # 转换所有保存的模型
                converted_models, onnx_dir = convert_models_to_onnx(
                    checkpoint_callback=checkpoint_callback,
                    model_class=ClassificationLightningModule,
                    config=config,
                    datamodule=data_module
                )
                
                if converted_models:
                    print(f"✅ ONNX conversion completed: {len(converted_models)} models converted")
                    print(f"📁 ONNX models saved to: {onnx_dir}")
                else:
                    print("❌ No models were converted to ONNX")
            else:
                print("⚠️ No ModelCheckpoint callback found, skipping ONNX conversion")
                
        except Exception as e:
            print(f"❌ ONNX conversion error: {e}")
            import traceback
            traceback.print_exc()

    # 汇总并保存最终结果
    if save_metrics:
        # 添加ONNX转换信息
        if convert_to_onnx and converted_models:
            final_metrics["onnx_conversion"] = {
                "converted_count": len(converted_models),
                "onnx_directory": onnx_dir,
                "models": converted_models
            }

        # 保存 JSON 文件
        result_path = os.path.join(config.get("outputs")["output_dir"], "results.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(final_metrics, f, indent=4)

        print(f"✅ Final metrics saved to: {result_path}")

    return trainer, model
