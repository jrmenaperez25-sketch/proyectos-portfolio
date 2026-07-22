#!/usr/bin/env python3
from __future__ import annotations

import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import argparse
from pathlib import Path

SEED = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True, help='dataset_yolo.yaml del fold')
    p.add_argument('--output-dir', type=Path, required=True, help='project (carpeta de runs)')
    p.add_argument('--name', required=True, help='nombre del run')
    p.add_argument('--model', default=f'{WORKSPACE}/yolo26s.pt', help='pesos de partida')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--no-mosaic', action='store_true',
                   help='Desactiva mosaic (variante del sweep)')
    return p.parse_args()

def main():
    args = parse_args()
    from ultralytics import YOLO, settings
    settings.update({'mlflow': False, 'wandb': False, 'comet': False, 'dvc': False, 'tensorboard': False})

    model = YOLO(args.model)
    print(f'YOLO26s | data={args.data} | epochs={args.epochs} | imgsz={args.imgsz} | '
          f'mosaic={"OFF" if args.no_mosaic else "default"}')

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=SEED,
        deterministic=True,
        patience=args.epochs + 1,
        mosaic=0.0 if args.no_mosaic else 1.0,
        project=str(args.output_dir),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    print(f'✓ YOLO entrenado. Resultados en {args.output_dir}/{args.name}')

if __name__ == '__main__':
    main()
