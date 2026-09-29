# Módulo 3 — Construcción de Modelos de Aprendizaje Supervisado

## ¿Qué se estudia?

Este módulo cubre los principales algoritmos de aprendizaje supervisado más allá de la regresión. Se trabaja tanto con modelos basados en árboles como con métodos de instancias (KNN) y máquinas de soporte vectorial (SVM), haciendo énfasis en cuándo usar cada uno y cómo ajustar sus hiperparámetros.

**Conceptos clave:**
- Árboles de decisión y regresión: criterios de división (Gini, entropía), profundidad máxima, poda
- K-Vecinos más cercanos (KNN): distancias euclídea y Manhattan, elección de K
- Support Vector Machines (SVM): hiperplano de separación, kernel trick, margen blando
- Métodos ensemble: Bagging (Random Forest) y Boosting (XGBoost)
- Clasificación multiclase: estrategias One-vs-Rest y One-vs-One

---

## Prácticas realizadas

| Notebook | Descripción |
|----------|-------------|
| `Árboles de Decisión.ipynb` | Construcción y visualización de un árbol de decisión, análisis de importancia de variables |
| `Árboles de Regresión.ipynb` | Aplicación de árboles para predicción continua y comparación con regresión lineal |
| `Clasificador K-vecinos más cercanos.ipynb` | Implementación de KNN, efecto del valor de K y fronteras de decisión |
| `Clasificación Multiclase.ipynb` | Extensión de clasificadores binarios a problemas con múltiples clases |
| `Random Forest y XGBoost.ipynb` | Comparación de ambos métodos ensemble, tuning de hiperparámetros y feature importance |
| `Detección de fraudes con tarjetas de crédito mediante árboles de decisiones y SVM-Árbol.ipynb` | Caso práctico real: detección de fraude con datos desbalanceados usando Decision Tree y SVM |

---

## Material de apoyo

- `Aprendizaje_Automático_Tema_3.pdf` — Diapositivas del módulo
- `Hoja de trucos-Construcción de modelos de Aprendizaje supervisado.pdf` — Resumen de algoritmos y parámetros clave

---

## Librerías utilizadas

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
```
