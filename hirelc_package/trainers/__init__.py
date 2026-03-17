"""Trainers module for HiReLC compression pipeline."""

from hirelc_package.trainers.base_trainer import BaseCompressionTrainer
from hirelc_package.trainers.cnn_trainer import CNNCompressionTrainer, create_cnn_trainer
from hirelc_package.trainers.vit_trainer import ViTCompressionTrainer, create_vit_trainer

__all__ = [
    'BaseCompressionTrainer',
    'CNNCompressionTrainer',
    'ViTCompressionTrainer',
    'create_cnn_trainer',
    'create_vit_trainer',
]
