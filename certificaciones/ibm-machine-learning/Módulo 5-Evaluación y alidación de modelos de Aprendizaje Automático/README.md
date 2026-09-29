# Módulo 5 — Evaluación y Validación de Modelos de Aprendizaje Automático

## ¿Qué se estudia?

Saber construir un modelo no es suficiente: hay que saber evaluarlo correctamente. Este módulo profundiza en las métricas adecuadas para cada tipo de problema, las técnicas de validación para evitar sobreajuste, y las herramientas de scikit-learn para automatizar y optimizar flujos de ML.

**Conceptos clave:**
- Métricas de clasificación: accuracy, precision, recall, F1-score, AUC-ROC, matriz de confusión
- Métricas de regresión: MSE, RMSE, MAE, R²
- Métricas de clustering: índice de silueta, Davies-Bouldin, inercia
- Validación cruzada: K-Fold, Stratified K-Fold, Leave-One-Out
- Regularización en regresión: Ridge, Lasso, ElasticNet
- Pipelines: encadenamiento de preprocesado y modelo en un solo objeto
- Búsqueda de hiperparámetros: GridSearchCV y RandomizedSearchCV

---

## Prácticas realizadas

| Notebook | Descripción |
|----------|-------------|
| `Evaluación de Modelos de Clasificación.ipynb` | Cálculo e interpretación de métricas de clasificación, curva ROC y precision-recall |
| `Evaluación K-Means Clustering.ipynb` | Evaluación de la calidad del clustering con índice de silueta y método del codo |
| `Evaluación Random Forest.ipynb` | Validación cruzada, importancia de variables y ajuste fino de Random Forest |
| `Regularización en Regresión Lineal.ipynb` | Comparativa de Ridge, Lasso y ElasticNet; efecto del parámetro de regularización α |
| `ML-Pipelines-and-GridSearchCV.ipynb` | Construcción de pipelines con preprocesado + modelo y búsqueda exhaustiva de hiperparámetros |

---

## Material de apoyo

- `Aprendizaje_Automático_Tema_5.pdf` — Diapositivas del módulo
- `Hoja de Trucos Evaluación de Modelos.pdf` — Referencia rápida de métricas y validación
- `Modelos de Regresión ante distintas situaciones.png` — Comparativa visual de modelos

---

## Librerías utilizadas

```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
```
