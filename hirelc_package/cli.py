"""Command-line interface for HiReLC."""

import argparse
import importlib
from typing import Tuple, Any

from hirelc_package.config import ExperimentConfig
from hirelc_package.utils import ExperimentLogger, ReproducibilityManager
from hirelc_package.trainers import CNNCompressionTrainer, ViTCompressionTrainer
from hirelc_package.pipeline import GenericCompressionRunner


def _load_factory(factory_path: str):
    if ':' not in factory_path:
        raise ValueError("Factory path must be in module:function format")
    module_name, func_name = factory_path.split(':', 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    return factory


def _load_custom_components(factory_path: str) -> Tuple[Any, Any, Any, int]:
    factory = _load_factory(factory_path)
    result = factory()

    if isinstance(result, dict):
        model = result.get('model')
        train_loader = result.get('train_loader')
        test_loader = result.get('test_loader')
        num_classes = result.get('num_classes', 0)
    elif isinstance(result, (list, tuple)):
        if len(result) == 3:
            model, train_loader, test_loader = result
            num_classes = 0
        elif len(result) >= 4:
            model, train_loader, test_loader, num_classes = result[:4]
        else:
            raise ValueError("Factory must return (model, train_loader, test_loader[, num_classes])")
    else:
        raise ValueError("Factory must return dict or tuple")

    if model is None or train_loader is None or test_loader is None:
        raise ValueError("Factory did not return required components")

    return model, train_loader, test_loader, int(num_classes) if num_classes else 0


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.model_name:
        config.model_name = args.model_name
    if args.dataset:
        config.dataset = args.dataset
    if args.num_classes:
        config.num_classes = args.num_classes
    if args.device:
        config.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.no_finetune:
        config.do_finetune = False
    if args.sensitivity_method:
        config.sensitivity_method = args.sensitivity_method
    if args.sensitivity_methods:
        methods = [m.strip() for m in args.sensitivity_methods.split(',') if m.strip()]
        config.sensitivity_methods = methods or None
    if args.quantization_type:
        config.quantization_type = args.quantization_type
    if args.default_strategy:
        config.default_strategy = args.default_strategy
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hirelc", description="HiReLC compression CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run compression")
    run_p.add_argument("--trainer", choices=["cnn", "vit", "custom"], default="cnn")
    run_p.add_argument("--config", help="Path to JSON config file")
    run_p.add_argument("--model-name")
    run_p.add_argument("--dataset")
    run_p.add_argument("--num-classes", type=int)
    run_p.add_argument("--device")
    run_p.add_argument("--output-dir")
    run_p.add_argument("--experiment-name")
    run_p.add_argument("--no-finetune", action="store_true")
    run_p.add_argument("--sensitivity-method")
    run_p.add_argument("--sensitivity-methods", help="Comma-separated list for composite sensitivity")
    run_p.add_argument("--quantization-type")
    run_p.add_argument("--default-strategy")
    run_p.add_argument("--factory", help="module:function factory for custom model")

    cfg_p = sub.add_parser("write-config", help="Write a default config JSON")
    cfg_p.add_argument("output", help="Output path for config JSON")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "write-config":
        cfg = ExperimentConfig()
        cfg.to_json(args.output)
        print(f"Wrote config to {args.output}")
        return

    config = ExperimentConfig()
    if args.config:
        config = ExperimentConfig.from_json(args.config)
    config = _apply_overrides(config, args)

    ReproducibilityManager.set_seed(config.seed)
    logger = ExperimentLogger(config.experiment_name, output_dir=config.output_dir)

    if args.trainer == "cnn":
        trainer = CNNCompressionTrainer(config, logger)
        trainer.run_hierarchical_compression()
        return

    if args.trainer == "vit":
        trainer = ViTCompressionTrainer(config, logger)
        trainer.run_hierarchical_compression()
        return

    if args.trainer == "custom":
        if not args.factory:
            raise ValueError("--factory is required for custom trainer")
        model, train_loader, test_loader, num_classes = _load_custom_components(args.factory)
        if num_classes:
            config.num_classes = num_classes
        runner = GenericCompressionRunner(model, train_loader, test_loader, config, logger=logger)
        runner.run()
        return


if __name__ == "__main__":
    main()
