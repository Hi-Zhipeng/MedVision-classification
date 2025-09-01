"""
Command line interface for MedVision Classification
"""

import click
from pathlib import Path

from ..utils import (
    train_model, 
    test_model, 
    run_inference_from_config,
    setup_logging
)


@click.group()
def cli():
    """MedVision Classification CLI"""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--resume", type=str, default=None, help="Resume from checkpoint")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def train(config_file: str, resume: str, debug: bool):
    """Train a classification model"""
    train_model(config_file, resume, debug)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--checkpoint", type=str, required=True, help="Checkpoint path")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--output-dir", type=str, default="outputs/test_results", help="Output directory for results")
def test(config_file: str, checkpoint: str, debug: bool, output_dir: str):
    """Test a classification model"""
    test_model(
        config_file=config_file,
        checkpoint_path=checkpoint,
        debug=debug,
        save_predictions=True,
        output_dir=output_dir
    )


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--input", type=str, required=True, help="Input path (file or directory)")
@click.option("--output", type=str, required=True, help="Output path for results")
@click.option("--checkpoint", type=str, required=True, help="Checkpoint path")
@click.option("--batch-size", type=int, default=32, help="Batch size for inference")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def predict(config_file: str, input: str, output: str, checkpoint: str, batch_size: int, debug: bool):
    """Run inference on images"""
    
    if debug:
        setup_logging(debug=True)
    
    results = run_inference_from_config(
        config_file=config_file,
        input_path=input,
        output_path=output,
        checkpoint_path=checkpoint,
        batch_size=batch_size
    )
    
    click.echo(f"Inference completed on {len(results)} images. Results saved to: {output}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--checkpoint", "-c", type=click.Path(exists=True), help="Model checkpoint path")
@click.option("--output", "-o", type=click.Path(), help="Output ONNX file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def export_onnx(config_file: str, checkpoint: str = None, output: str = None, verbose: bool = False):
    """Export trained model to ONNX format"""
    import torch
    import os
    from ..utils.onnx_export import convert_single_model_to_onnx
    from ..utils.helpers import load_config
    from ..models.lightning_module import ClassificationLightningModule
    
    # Load configuration
    config = load_config(config_file)
    
    # Determine output path
    if output is None:
        outputs_config = config.get("outputs", {})
        output_dir = outputs_config.get("output_dir", "outputs")
        onnx_dir = os.path.join(output_dir, "onnx_models")
        model_name = config.get("model", {}).get("network", {}).get("name", "model")
        num_classes = config.get("model", {}).get("num_classes", 2)
        output = os.path.join(onnx_dir, f"{model_name}_classes{num_classes}.onnx")
    
    # Create sample input based on config
    data_config = config.get("data", {})
    transforms_config = data_config.get("transforms", {})
    image_size = transforms_config.get("image_size", [224, 224])
    
    task_dim = config.get("task_dim", "2d")
    if task_dim.lower() == "3d" and len(image_size) == 3:
        sample_input = torch.randn(1, 1, *image_size)
    else:
        sample_input = torch.randn(1, 3, *image_size[-2:])
    
    # Convert to ONNX
    result = convert_single_model_to_onnx(
        checkpoint_path=checkpoint,
        model_class=ClassificationLightningModule,
        config=config,
        sample_input=sample_input,
        output_path=output,
        opset_version=config.get("onnx_export", {}).get("opset_version", 11),
        model_instance=None  # 从checkpoint加载
    )
    
    if result["success"]:
        click.echo(f"✅ ONNX export successful: {output}")
        if result.get("validation_passed"):
            click.echo("✅ ONNX model validation passed")
        else:
            click.echo(f"⚠️ ONNX validation: {result.get('validation_error', 'Unknown error')}")
    else:
        click.echo(f"❌ ONNX export failed: {result.get('error', 'Unknown error')}")
        exit(1)


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()
