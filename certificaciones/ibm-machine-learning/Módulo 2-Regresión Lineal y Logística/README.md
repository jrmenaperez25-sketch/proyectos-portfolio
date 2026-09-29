# Módulo 2 — Regresión Lineal y Logística

## ¿Qué se estudia?

Este módulo introduce los modelos de regresión supervisada, tanto para predicción de valores continuos (regresión lineal) como para clasificación binaria (regresión logística). Se analiza el fundamento matemático de cada modelo y sus supuestos principales.

**Conceptos clave:**
- Regresión lineal simple y múltiple: mínimos cuadrados, interpretación de coeficientes, R²
- Regularización: Ridge (L2) y Lasso (L1) para evitar sobreajuste
- Regresión logística: función sigmoide, odds ratio, clasificación binaria
- Evaluación: MSE, RMSE, MAE para regresión; accuracy, precision, recall para clasificación

---

## Prácticas realizadas

| Notebook | Descripción |
|----------|-------------|
| `Regresión Lineal Simple.ipynb` | Ajuste de un modelo lineal simple, visualización de la recta de regresión y evaluación con métricas básicas |
| `Regresión Lineal Múltiple.ipynb` | Modelo con múltiples predictores, análisis de colinealidad y selección de variables |
| `Regresión Logística.ipynb` | Clasificación binaria con regresión logística, matriz de confusión y curva ROC |

---

## Material de apoyo

- `Aprendizaje_Automático_Tema_2.pdf` — Diapositivas del módulo
- `Hoja de trucos-Regresión lineal y regresión logística.pdf` — Resumen de fórmulas y conceptos clave

---

## Librerías utilizadas

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
```
