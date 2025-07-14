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


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()
