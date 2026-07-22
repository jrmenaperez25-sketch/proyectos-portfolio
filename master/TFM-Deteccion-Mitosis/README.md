# Detección automática de figuras mitóticas en histopatología de cáncer de mama

Código de los experimentos del Trabajo Fin de Máster: un sistema de **detección de mitosis en dos
fases** evaluado sobre **MITOS-ATYPIA-14** y **TUPAC16**.

- **Fase 1 — Detección:** un detector (RF-DETR *Small* o YOLO26 *Small*, con y sin *oversampling*)
  propone candidatos de mitosis.
- **Fase 2 — Clasificación:** un modelo fundacional (**Virchow** ViT-H/14) filtra falsos positivos,
  en dos variantes: *frozen* (regresión logística sobre el embedding congelado) y **LoRA** (adaptación
  de bajo rango).

Metodología **val-first**: la mejor configuración y los umbrales del clasificador se eligen en
**validación** y se aplican sin cambios en test.

## Estructura del repositorio

```
01_deteccion_mitos_atypia14/   Fase 1 sobre MITOS-ATYPIA-14 (LOO, 10 folds)
    parches_mitos.py             Generación de parches: coverage (val), oversampling (train), sliding (test)
    entrenar_rfdetr.py           Entrenamiento RF-DETR Small (res 512)
    entrenar_yolo.py             Entrenamiento YOLO26 Small
    evaluar_test_mitos.py        Evaluación en el test fijo A04 (sliding window)
    evaluar_validacion_mitos.py  Evaluación en validación (coverage)

02_deteccion_tupac16/          Fase 1 sobre TUPAC16 (73 pacientes, split único)
    parches_tupac.py             Generación de parches (multi-frame 1-23 + single-frame 24-73)
    split_tupac.py               Definición del split train/val/test por paciente
    entrenar_rfdetr.py           Entrenamiento RF-DETR Small
    entrenar_yolo.py             Entrenamiento YOLO26 Small
    evaluar_test_tupac.py        Evaluación en test (sliding window)
    evaluar_validacion_tupac.py  Evaluación en validación

03_clasificador_virchow/       Fase 2: clasificador fundacional (Virchow)
    virchow_nucleo.py               Núcleo Virchow: backbone, embeddings, LoRA, evaluación end-to-end
    config_mitos.py                 Configuración de la 2.ª fase en MITOS (rutas, folds, criterios)
    entrenar_clasificador_mitos.py  Entrenamiento del clasificador (frozen y LoRA) sobre MITOS
    evaluar_end2end_mitos.py        Evaluación end-to-end per-fold (umbral fijo de validación)
    pipeline_tupac.py               Pipeline completo de la 2.ª fase sobre TUPAC (entrenar + evaluar)

04_evaluacion_y_figuras/       Análisis y figuras
    evaluar_end2end_tupac.py        End-to-end TUPAC en test con umbral fijo de validación
    frozen_end2end_y_fp.py          End-to-end del frozen + desglose de FP (empty vs hard)
    cross_scanner.py                Generalización entre escáneres MITOS <-> TUPAC
    fig_oversampling_vs_rot90.py    Figura: oversampling offline vs rotación 90°
```

## Correspondencia con la memoria

| Capítulo del TFM | Carpeta |
|---|---|
| Fase 1 · Detección (MITOS-ATYPIA-14) | `01_deteccion_mitos_atypia14/` |
| Fase 1 · Detección (TUPAC16) | `02_deteccion_tupac16/` |
| Fase 2 · Clasificador fundacional (Virchow, frozen y LoRA) | `03_clasificador_virchow/` |
| Evaluación end-to-end, desglose de FP, generalización entre escáneres y figuras | `04_evaluacion_y_figuras/` |

## Entorno

Entrenado y evaluado en una **NVIDIA A100-PCIE-40 GB** (CUDA 12.1). Versiones principales en
[`requirements.txt`](requirements.txt): PyTorch 2.5.1, rfdetr 1.6.5, ultralytics 8.4.31, peft 0.19,
timm 1.0.27, transformers 5.9. Semilla fija **42** en todos los entrenamientos.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # instalar torch/torchvision con el build CUDA adecuado
```

## Configuración de rutas

Los scripts localizan datos, pesos y salidas bajo una raíz de trabajo configurable mediante la
variable de entorno **`TFM_WORKSPACE`** (por defecto `/workspace`):

```bash
export TFM_WORKSPACE=/ruta/a/tus/datos
```

Las importaciones entre módulos (p. ej. `import parches_mitos`, `import virchow_nucleo`) se resuelven
solas: cada script añade automáticamente al `PYTHONPATH` las carpetas hermanas del repositorio.

## Datos y pesos

Los conjuntos **MITOS-ATYPIA-14** y **TUPAC16** están sujetos a sus respectivas licencias y **no se
incluyen** en este repositorio; deben solicitarse a sus organizadores. Tampoco se versionan los pesos
entrenados ni los artefactos intermedios (checkpoints, embeddings, CSV de predicciones). El código se
publica como **referencia reproducible** de la metodología descrita en la memoria.

## Notas

- El backbone Virchow (`hf-hub:paige-ai/Virchow`) requiere acceso a su repositorio en Hugging Face.
- Criterio de emparejamiento por defecto: **distancia de centroide ≤ 7,5 µm** (estándar MIDOG); se
  reporta también IoU ≥ 0,5. Punto de operación del detector: confianza 0,3.
- No se incluyen aquí algunos scripts auxiliares de figuras/verificación ni el experimento paralelo
  sobre MIDOG++; pueden facilitarse aparte si se necesitan.
