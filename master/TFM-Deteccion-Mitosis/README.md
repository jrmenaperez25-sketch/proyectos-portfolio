# Detección y clasificación de figuras mitóticas en histopatología de cáncer de mama

Sistema de **detección de mitosis en dos fases** sobre imagen histopatológica de cáncer de mama:
un detector propone candidatos con alta sensibilidad y un **modelo fundacional de patología**
adaptado con **LoRA** filtra los falsos positivos.

Este repositorio contiene el código de mi **Trabajo Fin de Máster** (Máster Universitario en
Ingeniería Biomédica, Universitat Politècnica de València) y del artículo derivado, presentado en
**CASEIB 2026**.

- 📄 **TFM:** *Diseño, desarrollo y validación de un framework de detección y clasificación de figuras
  mitóticas en histopatología de cáncer de mama* — [PDF](docs/TFM_completo.pdf)
- 📝 **Artículo:** *Two-Phase Mitotic Figure Detection in Breast Cancer Histopathology: RF-DETR
  Candidates and a LoRA-Adapted Pathology Foundation Model as False-Positive Filter* —
  XLIV Congreso Anual de la Sociedad Española de Ingeniería Biomédica (CASEIB 2026), Valencia,
  11–13 de noviembre de 2026.
- 🏛️ Grupo **CVBLab**, instituto HUMAN-tech, UPV.

---

## El problema

El **recuento de mitosis** es uno de los tres componentes del **grado de Nottingham**, la escala con
la que se gradúa el cáncer de mama: mide cuánto está proliferando el tumor y condiciona el
tratamiento. Hacerlo a mano es lento y subjetivo —el patólogo recorre la preparación al microscopio
y cuenta figuras mitóticas una a una—, y automatizarlo es difícil por tres motivos:

1. Una mitosis adopta formas muy distintas según la fase en la que esté la célula.
2. Hay muchas células que **parecen** mitosis y no lo son: en los conjuntos usados aquí hay del
   orden de **cuatro imitaciones por cada mitosis**.
3. Cada escáner digitaliza con un color y un contraste distintos.

## La propuesta: dos fases

| Fase | Modelo | Qué hace |
|---|---|---|
| 1 · Detección | **RF-DETR Small** (transformer) o **YOLO26 Small** (convolucional) | Propone candidatos priorizando la sensibilidad: lo que no se detecta aquí ya no se recupera. |
| 2 · Clasificación | **Virchow** (ViT-H/14, fundacional de patología), *frozen* o adaptado con **LoRA** | Recorta cada candidato y decide si es mitosis, filtrando los falsos positivos. |

La clave de la segunda fase es **con qué se entrena**: los negativos no son parches cualesquiera,
sino **los falsos positivos que el detector comete de verdad**. El clasificador aprende así a
corregir los errores concretos de su detector.

Metodología **val-first**: la configuración y el umbral de decisión (τ) se eligen en **validación**
y se aplican sin cambios en test.

## Resultados

Detección considerada correcta si el centro predicho está a **≤ 7,5 µm** del centro anotado
(criterio estándar de los retos MITOS/TUPAC/MIDOG).

**MITOS-ATYPIA-14** (test: paciente A04)

| Sistema | Precisión | Sensibilidad | F₁ |
|---|---|---|---|
| Solo detector (RF-DETR) | 0,470 | **0,909** | 0,619 |
| + Virchow *frozen* (τ 0,30) | 0,734 | 0,619 | 0,671 |
| + Virchow **LoRA** (τ 0,20) | **0,752** | 0,835 | **0,791** |

**TUPAC16** (split por paciente)

| Sistema | Precisión | Sensibilidad | F₁ |
|---|---|---|---|
| Solo detector (RF-DETR) | 0,405 | **0,939** | 0,566 |
| + Virchow *frozen* (τ 0,25) | 0,632 | 0,759 | 0,689 |
| + Virchow **LoRA** (τ 0,85) | **0,786** | 0,736 | **0,760** |

**Conclusiones:**

- La segunda fase **casi duplica la precisión sin perder apenas mitosis**: el F₁ sube de 0,619 a
  0,791 en MITOS y de 0,566 a 0,760 en TUPAC.
- La adaptación con **LoRA marca la diferencia** frente al modelo congelado: el *frozen* filtra bien
  el fondo pero se lleva alrededor de un tercio de las mitosis reales; LoRA pierde apenas un 8 %.
- El patrón **se repite en dos conjuntos independientes**, con escáneres y protocolos distintos.
- **Generalización entre escáneres:** el detector transfiere razonablemente, pero el clasificador
  se especializa en la apariencia de su dominio. El sistema **no es portable a otro escáner sin
  readaptarlo**.

## Datos

- **MITOS-ATYPIA-14** — 16 pacientes, cada preparación digitalizada con dos escáneres (Aperio y
  Hamamatsu). Validación cruzada *leave-one-patient-out*.
- **TUPAC16** — 73 casos, ~1 900 mitosis frente a más de 5 000 imitaciones. Partición única por
  paciente.

Los dos son conjuntos públicos; el repositorio **no incluye las imágenes**, solo el código para
reproducir los experimentos.

## Estructura del repositorio

```
docs/
    TFM_completo.pdf              Memoria del TFM
    TFM_defensa.pdf               Presentación de la defensa

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

requirements.txt
```

## Cómo se generan los parches

- **Entrenamiento — *oversampling* offline:** cada mitosis se replica hasta 15 veces (5 ángulos × 3
  reflexiones), y **cada copia se recorta volviendo al tejido original**, así aparece sobre un
  contexto distinto y sin bordes rellenos artificialmente.
- **Validación — *coverage*:** un parche por grupo de anotaciones, de modo que cada mitosis aparece
  una sola vez y la métrica no queda inflada.
- **Test — ventana deslizante:** se recorre el frame completo y después se fusionan las detecciones
  repetidas.

## Reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TFM_WORKSPACE=/ruta/a/tus/datos
```

Todos los entrenamientos usan **semilla 42**. Entrenado en una **NVIDIA A100-PCIE de 40 GB**
(CUDA 12.1).

## Stack

`PyTorch 2.5` · `rfdetr` · `ultralytics` (YOLO26) · `timm` (Virchow ViT-H/14) · `peft` (LoRA) ·
`scikit-learn` · `Albumentations` · `supervision` · `OpenCV`

**Configuración LoRA:** `r=4`, `alpha=8`, `dropout=0.1` sobre las proyecciones **qkv** de los 32
bloques + cabeza lineal → **~0,66 M parámetros entrenables, un 0,1 % del modelo**.

## Cómo citar

```bibtex
@inproceedings{menaperez2026mitosis,
  title     = {Two-Phase Mitotic Figure Detection in Breast Cancer Histopathology:
               RF-DETR Candidates and a LoRA-Adapted Pathology Foundation Model
               as False-Positive Filter},
  author    = {Mena-P{\'e}rez, Jos{\'e} Ram{\'o}n and Golfe, Alejandro and
               Rodr{\'i}guez Albendea, V{\'i}ctor and Terradez, Liria and
               Colomer, Adri{\'a}n},
  booktitle = {XLIV Congreso Anual de la Sociedad Espa{\~n}ola de Ingenier{\'i}a Biom{\'e}dica
               (CASEIB)},
  address   = {Valencia, Spain},
  year      = {2026}
}
```

## Autor

**José Ramón Mena Pérez** — Máster en Ingeniería Biomédica, UPV.  
Dirigido por **Adrián Colomer** y **Alejandro Golfe San Martín** (CVBLab · HUMAN-tech · UPV).
