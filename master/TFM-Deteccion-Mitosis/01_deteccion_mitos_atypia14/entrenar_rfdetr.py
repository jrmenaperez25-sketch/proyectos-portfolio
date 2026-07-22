#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

HYPERPARAMS = {
    'epochs': 30,
    'batch_size': 16,
    'grad_accum_steps': 2,
    'lr': 1e-4,
    'lr_encoder': 1e-5,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine',
    'warmup_epochs': 3.0,
    'clip_max_norm': 0.1,

    'multi_scale': False,
    'expanded_scales': False,
    'square_resize_div_64': True,

    'use_ema': True,
    'eval_interval': 1,

    'early_stopping': True,
    'early_stopping_patience': 8,
    'early_stopping_min_delta': 0.001,
    'early_stopping_use_ema': True,

    'eval_max_dets': 500,
    'log_per_class_metrics': True,

    'tensorboard': True,
    'wandb': False,
    'mlflow': False,

    'aug_config': {},
    'seed': SEED,

    'compute_val_loss': True,
    'compute_test_loss': False,
    'run_test': False,

    'num_workers': 4,
}

MODEL_REGISTRY = {
    'small': 'RFDETRSmall',
    'medium': 'RFDETRMedium',
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=Path, required=True,
                   help='Dataset ensamblado por assemble_split.py')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='Donde RF-DETR escribirá checkpoints, logs, args.yaml')
    p.add_argument('--model', choices=list(MODEL_REGISTRY), default='small')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override de HYPERPARAMS["epochs"]')
    p.add_argument('--lr-head', type=float, default=None,
                   help='Override del lr del head (lr). Default 1e-4.')
    p.add_argument('--lr-encoder', type=float, default=None,
                   help='Override del lr del backbone. Default 1e-5.')
    p.add_argument('--aug-preset', choices=['none', 'leve_hsv', 'full_geom'],
                   default='none',
                   help='Preset de Albumentations. none=ninguna, leve_hsv=brillo+HSV, full_geom=flips+rot+brillo+HSV')
    p.add_argument('--no-early-stopping', action='store_true',
                   help='Desactiva early_stopping (entrena las epochs completas)')
    p.add_argument('--early-stopping-patience', type=int, default=None,
                   help='Activa early stopping con esta paciencia (épocas sin mejora). '
                        'Tiene prioridad sobre --no-early-stopping.')
    p.add_argument('--dry-run', action='store_true',
                   help='Solo imprime la config y termina sin entrenar.')
    return p.parse_args()

AUG_LEVE_HSV = {
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5,
    },
    "HueSaturationValue": {
        "hue_shift_limit": 15, "sat_shift_limit": 25, "val_shift_limit": 20, "p": 0.4,
    },
}
AUG_FULL_GEOM = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.6},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5,
    },
    "HueSaturationValue": {
        "hue_shift_limit": 15, "sat_shift_limit": 25, "val_shift_limit": 20, "p": 0.4,
    },
}

def main():
    args = parse_args()

    if not args.dataset_dir.exists():
        raise SystemExit(f'Dataset no existe: {args.dataset_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.epochs is not None:
        HYPERPARAMS['epochs'] = args.epochs
    if args.lr_head is not None:
        HYPERPARAMS['lr'] = args.lr_head
    if args.lr_encoder is not None:
        HYPERPARAMS['lr_encoder'] = args.lr_encoder
    if args.aug_preset == 'leve_hsv':
        HYPERPARAMS['aug_config'] = AUG_LEVE_HSV
    elif args.aug_preset == 'full_geom':
        HYPERPARAMS['aug_config'] = AUG_FULL_GEOM
    if args.early_stopping_patience is not None:
        HYPERPARAMS['early_stopping'] = True
        HYPERPARAMS['early_stopping_patience'] = args.early_stopping_patience
    elif args.no_early_stopping:
        HYPERPARAMS['early_stopping'] = False

    print('=' * 70)
    print(f'TRAIN RF-DETR — modelo: {args.model.upper()}')
    print('=' * 70)
    print(f'  Dataset    : {args.dataset_dir}')
    print(f'  Output     : {args.output_dir}')
    print(f'  Seed       : {SEED}')
    print(f'  Torch      : {torch.__version__}')
    print(f'  CUDA       : {torch.version.cuda if torch.cuda.is_available() else "CPU"}')
    print(f'  GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—"}')
    print('\nHiperparámetros:')
    for k, v in HYPERPARAMS.items():
        print(f'  {k:30s}: {v}')
    print('=' * 70)

    cfg_path = args.dataset_dir / 'split_config.json'
    if cfg_path.exists():
        ds_cfg = json.loads(cfg_path.read_text())
        print('\nSplit config del dataset:')
        for split, items in ds_cfg.get('patients', {}).items():
            print(f'  {split}: {", ".join(items)}')

    try:
        import rfdetr
    except ImportError:
        raise SystemExit(
            'rfdetr no instalado. Hazlo con: pip install "rfdetr[train,loggers]"'
        )

    rfdetr_path = Path(rfdetr.__file__).resolve()
    print(f'\nrfdetr cargado desde: {rfdetr_path}')

    print('\nAplicando monkey-patch para desactivar RandomSizedCrop residual…')
    from rfdetr.datasets import coco as coco_ds

    _original_build_resize = coco_ds._build_train_resize_config

    def _patched_build_train_resize_config(scales, *, square, max_size=None):
        if square:
            return [
                {
                    "OneOf": {
                        "transforms": [
                            {"Resize": {"height": s, "width": s}} for s in scales
                        ],
                    }
                }
            ]
        cap = max_size or 1333
        size_param = scales[0] if len(scales) == 1 else scales
        return [
            {
                "Sequential": {
                    "transforms": [
                        {"SmallestMaxSize": {"max_size": size_param}},
                        {"LongestMaxSize": {"max_size": cap}},
                    ]
                }
            }
        ]

    coco_ds._build_train_resize_config = _patched_build_train_resize_config
    print('  ✓ _build_train_resize_config reemplazado (sin RandomSizedCrop)')

    cls_name = MODEL_REGISTRY[args.model]
    ModelCls = getattr(rfdetr, cls_name)

    model_kwargs = {'num_classes': 1}
    if args.model == 'small':
        model_kwargs['resolution'] = 512
    elif args.model == 'medium':
        model_kwargs['resolution'] = 576

    print(f'\nInstanciando {cls_name}({model_kwargs})…')
    model = ModelCls(**model_kwargs)

    if args.dry_run:
        print('\n--dry-run: configuración OK, no se entrena.')
        return

    exp_cfg = {
        'model_class': cls_name,
        'dataset_dir': str(args.dataset_dir),
        'output_dir': str(args.output_dir),
        'hyperparams': HYPERPARAMS,
        'seed': SEED,
        'torch_version': torch.__version__,
        'rfdetr_path': str(rfdetr_path),
        'cuda_version': str(torch.version.cuda) if torch.cuda.is_available() else None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'start_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (args.output_dir / 'experiment_config.json').write_text(
        json.dumps(exp_cfg, indent=2, default=str)
    )

    print('\nLanzando model.train(...)\n')
    t0 = time.time()
    model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        **HYPERPARAMS,
    )
    elapsed = time.time() - t0
    print(f'\n✓ Entrenamiento completado en {elapsed/60:.1f} min')

    exp_cfg['end_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    exp_cfg['elapsed_minutes'] = round(elapsed / 60, 1)
    (args.output_dir / 'experiment_config.json').write_text(
        json.dumps(exp_cfg, indent=2, default=str)
    )

if __name__ == '__main__':
    main()
