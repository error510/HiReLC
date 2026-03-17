"""Generic compression pipeline for arbitrary PyTorch models."""

from typing import Dict, List, Tuple, Optional
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from hirelc_package.config import ExperimentConfig, LayerConfig, KernelConfig
from hirelc_package.core import (
    AdvancedQuantizer,
    AdvancedPruner,
    build_sensitivity_estimator,
)


class GenericCompressionRunner:
    """
    Lightweight, model-agnostic compression pipeline.

    This is a pragmatic fallback for arbitrary user models (MLP/CNN/Transformer)
    when full HRL agents are not required.
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 test_loader,
                 config: ExperimentConfig,
                 logger=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = logger
        self.device = config.device

        self.model.to(self.device)
        self.model_blocks = self._get_model_blocks()

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)

    def _get_model_blocks(self) -> List[List[Tuple[str, nn.Module]]]:
        blocks: List[List[Tuple[str, nn.Module]]] = []
        current: List[Tuple[str, nn.Module]] = []

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                current.append((name, module))
                if len(current) >= 4:
                    blocks.append(current)
                    current = []

        if current:
            blocks.append(current)

        if not blocks:
            blocks = [[]]

        return blocks

    def evaluate(self, model: Optional[nn.Module] = None, max_batches: Optional[int] = None) -> float:
        model = model or self.model
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total if total > 0 else 0.0

    def compute_sensitivity(self, num_samples: int = 10) -> Dict[int, float]:
        estimator = build_sensitivity_estimator(
            self.model,
            self.train_loader,
            device=self.device,
            method=self.config.sensitivity_method,
            num_blocks=len(self.model_blocks),
            methods=self.config.sensitivity_methods,
        )
        scores = estimator.estimate_importance(num_samples=num_samples)

        if scores:
            min_score = min(scores.values())
            max_score = max(scores.values())
            if max_score > min_score:
                scores = {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

        return scores

    def build_layer_configs(self, sensitivity_scores: Dict[int, float]) -> Dict[int, LayerConfig]:
        configs: Dict[int, LayerConfig] = {}
        goal = self.config.compression_goal

        for block_idx, block_layers in enumerate(self.model_blocks):
            sensitivity = sensitivity_scores.get(block_idx, 0.5)

            min_bits = goal.min_layer_bits
            max_bits = goal.max_layer_bits
            bits = int(round(min_bits + sensitivity * (max_bits - min_bits)))

            min_prune = goal.min_layer_pruning
            max_prune = goal.max_layer_pruning
            prune_ratio = min_prune + (1.0 - sensitivity) * (max_prune - min_prune)
            keep_ratio = max(0.0, min(1.0, 1.0 - prune_ratio))

            qtype = self.config.quantization_type.lower()
            if qtype == 'float':
                quant_type = 'FLOAT'
            else:
                quant_type = 'INT'

            quant_mode = self.config.default_strategy or 'uniform'

            layer_config = LayerConfig(block_idx=block_idx)

            for k_idx, (name, module) in enumerate(block_layers):
                kc = KernelConfig(
                    name=f"kernel{k_idx+1}",
                    weight_bits=bits,
                    act_bits=bits,
                    quant_type=quant_type,
                    quant_mode=quant_mode,
                    pruning_ratio=keep_ratio,
                    shape=tuple(module.weight.shape) if hasattr(module, 'weight') else tuple()
                )
                layer_config.kernels[kc.name] = kc
                if k_idx == 0:
                    layer_config.kernel1_config = kc
                elif k_idx == 1:
                    layer_config.kernel2_config = kc
                elif k_idx == 2:
                    layer_config.kernel3_config = kc
                elif k_idx == 3:
                    layer_config.kernel4_config = kc

            layer_config.update_aggregates()
            configs[block_idx] = layer_config

        return configs

    def apply_configs(self, configs: Dict[int, LayerConfig]) -> None:
        for block_idx, layer_config in configs.items():
            if block_idx >= len(self.model_blocks):
                continue
            for k_idx, (name, module) in enumerate(self.model_blocks[block_idx]):
                if not hasattr(module, 'weight') or module.weight is None:
                    continue

                kernel_name = f"kernel{k_idx+1}"
                kc = layer_config.kernels.get(kernel_name)
                if kc is None:
                    continue

                weights = module.weight.data
                if kc.pruning_ratio < 1.0:
                    mask = AdvancedPruner.create_neuron_mask(
                        weights, kc.pruning_ratio, importance_method=kc.importance_method
                    )
                    weights = AdvancedPruner.apply_mask(weights, mask)

                if kc.weight_bits < 32:
                    weights = AdvancedQuantizer.quantize(
                        weights, kc.weight_bits, mode=kc.quant_mode, quant_type=kc.quant_type
                    )

                module.weight.data = weights

    def finetune(self) -> None:
        if not self.config.do_finetune:
            return

        self._log(f"Fine-tuning for {self.config.finetune_epochs} epochs...")
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config.finetune_epochs):
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(self.train_loader))
            self._log(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    def run(self) -> Dict[int, LayerConfig]:
        self._log("Running generic compression pipeline...")

        baseline_acc = self.evaluate(max_batches=20)
        self._log(f"Baseline accuracy: {baseline_acc:.2f}%")

        sensitivity_scores = self.compute_sensitivity(num_samples=10)
        self._log(f"Sensitivity computed for {len(sensitivity_scores)} blocks")

        configs = self.build_layer_configs(sensitivity_scores)
        self.apply_configs(configs)

        compressed_acc = self.evaluate(max_batches=20)
        self._log(f"Compressed accuracy: {compressed_acc:.2f}%")

        if self.config.do_finetune:
            self.finetune()
            finetuned_acc = self.evaluate(max_batches=20)
            self._log(f"Finetuned accuracy: {finetuned_acc:.2f}%")

        return configs
