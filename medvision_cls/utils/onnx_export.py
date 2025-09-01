"""
ONNX conversion utilities for MedVision.
"""

import os
import glob
import torch
from typing import Dict, Any, List, Tuple


def generate_triton_config(model_name: str, input_shape: List[int], num_classes: int, output_dir: str, onnx_path: str) -> str:
    """
    生成Triton配置文件和目录结构
    
    Args:
        model_name: 模型名称
        input_shape: 输入形状 [batch, channels, height, width] 或 [batch, channels, depth, height, width]
        num_classes: 类别数量
        output_dir: 输出目录（onnx_models目录）
        onnx_path: 原始ONNX文件路径
        
    Returns:
        str: 配置文件路径
    """
    # 确定输入维度（去掉batch维度）
    input_dims = input_shape[1:]
    
    # 生成Triton配置内容
    config_content = f'''name: "{model_name}"
backend: "onnxruntime_onnx"
max_batch_size: 8

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: {input_dims}
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [{num_classes}]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
'''
    
    # 创建Triton模型仓库结构: model_name/1/model.onnx
    model_dir = os.path.join(output_dir, model_name)
    version_dir = os.path.join(model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    
    # 生成Triton模型仓库的配置文件
    triton_config_path = os.path.join(model_dir, "config.pbtxt")
    with open(triton_config_path, 'w') as f:
        f.write(config_content)
    
    # 复制ONNX文件到版本目录中，命名为model.onnx
    import shutil
    triton_model_path = os.path.join(version_dir, "model.onnx")
    shutil.copy2(onnx_path, triton_model_path)
    
    return triton_config_path


def convert_models_to_onnx(
    checkpoint_callback, 
    model_class, 
    config: Dict[str, Any], 
    datamodule
) -> Tuple[List[Dict], str]:
    """
    将保存的top-k模型转换为ONNX格式
    
    Args:
        checkpoint_callback: ModelCheckpoint回调对象
        model_class: 模型类
        config: 配置字典
        datamodule: 数据模块
        
    Returns:
        Tuple[List[Dict], str]: 转换成功的模型列表和ONNX目录路径
    """
    # 配置参数
    opset_version = config.get("onnx_opset_version", 11)

    checkpoint_dir = checkpoint_callback.dirpath
    # 基于output_dir拼接onnx目录
    output_dir = config.get("outputs", {}).get("output_dir", "outputs")
    onnx_dir = os.path.join(output_dir, "onnx_models")
    os.makedirs(onnx_dir, exist_ok=True)
    
    # 获取所有检查点文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    # 获取示例输入
    datamodule.setup('fit')
    sample_batch = next(iter(datamodule.train_dataloader()))
    
    # 处理不同的batch格式
    if isinstance(sample_batch, dict):
        # 字典格式: {"image": tensor, "label": tensor}
        sample_input = sample_batch["image"][:1]  # 取一个样本
    elif isinstance(sample_batch, (tuple, list)) and len(sample_batch) >= 2:
        # 元组/列表格式: (images, labels)
        sample_input = sample_batch[0][:1]  # 取一个样本
    else:
        # 单张量格式或其他格式
        sample_input = sample_batch[:1] if hasattr(sample_batch, '__getitem__') else sample_batch
    
    converted_models = []
    
    print(f"Found {len(checkpoint_files)} checkpoint files to convert...")
    
    # 检查示例输入的设备
    print(f"Sample input device: {sample_input.device}")
    
    for ckpt_path in checkpoint_files:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\nConverting {ckpt_name}...")
        
        try:
            # 加载模型 - 直接从checkpoint加载，不需要传递config
            model = model_class.load_from_checkpoint(ckpt_path)
            model.eval()
            
            # 禁用训练相关功能
            torch.set_grad_enabled(False)
            
            # 检查模型参数的设备
            model_device = next(model.parameters()).device
            print(f"  Model loaded on device: {model_device}")
            
            # 将模型移动到CPU进行ONNX转换
            if model_device.type == 'cuda':
                print(f"  Moving model from {model_device} to CPU for ONNX export...")
                model = model.cpu()
            
            # 确保示例输入在CPU上
            sample_input_cpu = sample_input.cpu()
            print(f"  Using sample input on device: {sample_input_cpu.device}")
            print(f"  Input shape: {sample_input_cpu.shape}")
            
            # ONNX文件路径
            onnx_path = os.path.join(onnx_dir, f"{ckpt_name}.onnx")
            
            # 获取输入shape信息
            input_shape = sample_input_cpu.shape
            
            # 测试前向传播
            try:
                with torch.no_grad():
                    test_output = model(sample_input_cpu)
                    print(f"  Model forward test successful, output shape: {test_output.shape}")
            except Exception as e:
                print(f"  ⚠️ Model forward test failed: {e}")
                continue
            
            # 转换为ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,  # 使用完整的lightning模块
                    sample_input_cpu,
                    onnx_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    },
                    verbose=False
                )
            
            # 验证ONNX模型
            try:
                import onnx
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f"✓ ONNX model validation passed: {ckpt_name}")
            except ImportError:
                print(f"⚠ ONNX validation skipped (onnx package not installed): {ckpt_name}")
            except Exception as e:
                print(f"⚠ ONNX validation failed: {ckpt_name}, error: {e}")
            
            # 生成Triton配置文件和目录结构
            triton_config_path = None
            try:
                num_classes = config.get("model", {}).get("num_classes", 2)
                
                triton_config_path = generate_triton_config(
                    model_name=ckpt_name,
                    input_shape=list(input_shape),
                    num_classes=num_classes,
                    output_dir=onnx_dir,
                    onnx_path=onnx_path
                )
                print(f"✓ Triton config and model repository generated: {ckpt_name}")
            except Exception as e:
                print(f"⚠ Triton config generation failed for {ckpt_name}: {e}")
            
            converted_models.append({
                "checkpoint_path": ckpt_path,
                "onnx_path": onnx_path,
                "model_name": ckpt_name,
                "input_shape": list(input_shape),
                "triton_config_path": triton_config_path,
                "original_device": str(model_device) if 'model_device' in locals() else "unknown"
            })
            
            print(f"  ✓ Successfully converted {ckpt_name} to ONNX")
            
        except Exception as e:
            print(f"  ❌ Failed to convert {ckpt_name}: {str(e)}")
            import traceback
            print(f"  Full error traceback:")
            traceback.print_exc()
        finally:
            # 重新启用梯度
            torch.set_grad_enabled(True)
    
    return converted_models, onnx_dir


def convert_single_model_to_onnx(
    checkpoint_path: str,
    model_class,
    config: Dict[str, Any],
    sample_input: torch.Tensor,
    output_path: str,
    opset_version: int = 11,
    model_instance=None  # 添加可选的模型实例参数
) -> Dict[str, Any]:
    """
    将单个模型转换为ONNX格式
    
    Args:
        checkpoint_path: 模型检查点路径
        model_class: 模型类
        config: 模型配置字典
        sample_input: 示例输入张量
        output_path: 输出ONNX文件路径
        opset_version: ONNX opset版本
        model_instance: 可选的模型实例，如果提供则不从checkpoint加载
        
    Returns:
        Dict[str, Any]: 转换结果信息
    """
    try:
        # 加载模型
        if model_instance is not None:
            model = model_instance
        elif checkpoint_path and checkpoint_path != "current_state":
            # 直接从checkpoint加载，不需要传递config
            model = model_class.load_from_checkpoint(checkpoint_path)
        else:
            # 如果没有提供checkpoint或model_instance，创建新模型
            model_config = config.get("model", {})
            model = model_class(
                model_config=model_config,
                loss_config=config.get("loss", {}),
                optimizer_config=config.get("optimizer", {}),
                scheduler_config=config.get("scheduler", {}),
                metrics_config=config.get("metrics", {})
            )
        
        model.eval()
        
        # 检查模型参数的设备
        model_device = next(model.parameters()).device
        
        # 将模型移动到CPU进行ONNX转换
        if model_device.type == 'cuda':
            model = model.cpu()
        
        # 确保示例输入在CPU上
        sample_input_cpu = sample_input.cpu()
        
        # 获取输入shape信息
        input_shape = sample_input_cpu.shape
        
        # 转换为ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,  # 使用完整的lightning模块
                sample_input_cpu,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        # 验证ONNX模型
        validation_passed = False
        validation_error = None
        
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            validation_passed = True
        except ImportError:
            validation_error = "ONNX package not installed"
        except Exception as e:
            validation_error = str(e)
        
        return {
            "success": True,
            "checkpoint_path": checkpoint_path,
            "onnx_path": output_path,
            "input_shape": list(input_shape),
            "validation_passed": validation_passed,
            "validation_error": validation_error
        }
        
    except Exception as e:
        return {
            "success": False,
            "checkpoint_path": checkpoint_path,
            "onnx_path": output_path,
            "error": str(e)
        }


def validate_onnx_model(onnx_path: str) -> Tuple[bool, str]:
    """
    验证ONNX模型的有效性
    
    Args:
        onnx_path: ONNX模型文件路径
        
    Returns:
        Tuple[bool, str]: (是否验证通过, 错误信息)
    """
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        return True, "Validation passed"
    except ImportError:
        return False, "ONNX package not installed"
    except Exception as e:
        return False, str(e)


def get_onnx_model_info(onnx_path: str) -> Dict[str, Any]:
    """
    获取ONNX模型的信息
    
    Args:
        onnx_path: ONNX模型文件路径
        
    Returns:
        Dict[str, Any]: 模型信息字典
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        
        # 获取输入信息
        inputs = []
        for input_tensor in model.graph.input:
            input_info = {
                "name": input_tensor.name,
                "type": input_tensor.type.tensor_type.elem_type,
                "shape": [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            }
            inputs.append(input_info)
        
        # 获取输出信息
        outputs = []
        for output_tensor in model.graph.output:
            output_info = {
                "name": output_tensor.name,
                "type": output_tensor.type.tensor_type.elem_type,
                "shape": [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            }
            outputs.append(output_info)
        
        return {
            "file_path": onnx_path,
            "file_size": os.path.getsize(onnx_path),
            "opset_version": model.opset_import[0].version if model.opset_import else None,
            "inputs": inputs,
            "outputs": outputs,
            "node_count": len(model.graph.node)
        }
        
    except ImportError:
        return {"error": "ONNX package not installed"}
    except Exception as e:
        return {"error": str(e)}


# 兼容性函数，保持与原有代码的接口一致
def export_model_after_training(
    model, 
    config: Dict[str, Any], 
    checkpoint_path: str = None, 
    verbose: bool = True
) -> str:
    """
    训练后导出模型的兼容性包装函数
    """
    try:
        from ..models.lightning_module import ClassificationLightningModule
        
        # 获取ONNX输出路径 - 基于output_dir拼接
        output_dir = config.get("outputs", {}).get("output_dir", "outputs")
        onnx_dir = os.path.join(output_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        
        # 创建示例输入
        data_config = config.get("data", {})
        transforms_config = data_config.get("transforms", {})
        image_size = transforms_config.get("image_size", [224, 224])
        
        task_dim = config.get("task_dim", "2d")
        if task_dim.lower() == "3d" and len(image_size) == 3:
            sample_input = torch.randn(1, 1, *image_size)
        else:
            sample_input = torch.randn(1, 3, *image_size[-2:])
        
        # 生成输出文件名
        model_name = config.get("model", {}).get("network", {}).get("name", "model")
        num_classes = config.get("model", {}).get("num_classes", 2)
        onnx_filename = f"{model_name}_classes{num_classes}.onnx"
        onnx_path = os.path.join(onnx_dir, onnx_filename)
        
        # 转换
        result = convert_single_model_to_onnx(
            checkpoint_path=checkpoint_path,
            model_class=ClassificationLightningModule,
            config=config,
            sample_input=sample_input,
            output_path=onnx_path,
            opset_version=config.get("onnx_export", {}).get("opset_version", 11),
            model_instance=model if checkpoint_path is None else None  # 传递模型实例
        )
        
        if result["success"] and verbose:
            print(f"✅ ONNX export successful: {onnx_path}")
            if result.get("validation_passed"):
                print("✅ ONNX model validation passed")
            else:
                print(f"⚠️ ONNX validation: {result.get('validation_error', 'Unknown error')}")
        
        return onnx_path if result["success"] else None
        
    except Exception as e:
        if verbose:
            print(f"❌ ONNX export failed: {e}")
        return None
