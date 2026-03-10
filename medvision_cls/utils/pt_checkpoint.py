"""
Custom callback to save .pt format in Lightning
"""
import os
import torch
from pytorch_lightning.callbacks import Callback


class PTCheckpoint(Callback):
    """Save .pt format instead of .ckpt"""

    def __init__(self, dirpath, filename, monitor="val/accuracy", mode="max"):
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = None
        os.makedirs(dirpath, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        # Always save last epoch weights
        last_path = os.path.join(self.dirpath, f"{self.filename}_last.pt")
        torch.save(pl_module.model.state_dict(), last_path)

        is_best = (self.mode == 'min' and current < self.best_score) or \
                  (self.mode == 'max' and current > self.best_score)

        if is_best:
            self.best_score = current
            self.best_model_path = os.path.join(self.dirpath, f"{self.filename}.pt")
            torch.save(pl_module.model.state_dict(), self.best_model_path)


def load_from_pt(pt_path, model_name, num_classes, **kwargs):
    """
    从 backbone.pt 文件加载模型

    Args:
        pt_path: .pt 文件路径
        model_name: 模型名称，需与训练时一致（如 "resnet50"）
        num_classes: 分类数，需与训练时一致
        **kwargs: 其他传给 create_model 的参数
    """
    from ..models import create_model
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False, **kwargs)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    return model
