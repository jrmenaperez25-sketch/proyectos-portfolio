# Módulo 4 — Construcción de Modelos de Aprendizaje No Supervisado

## ¿Qué se estudia?

El aprendizaje no supervisado trabaja con datos sin etiquetas. Este módulo cubre los principales enfoques: clustering (agrupamiento) y reducción de dimensionalidad. Se estudia cómo descubrir estructura oculta en los datos y cómo visualizar datasets de alta dimensión.

**Conceptos clave:**
- Clustering particional: K-Means, elección del número de clusters (método del codo, silueta)
- Clustering basado en densidad: DBSCAN y HDBSCAN, puntos núcleo, frontera y ruido
- Reducción de dimensionalidad: PCA (componentes principales), varianza explicada, biplots
- Visualización no lineal: t-SNE (preserva estructura local) y UMAP (preserva estructura global y local)

---

## Prácticas realizadas

| Notebook | Descripción |
|----------|-------------|
| `Algoritmo K-Means.ipynb` | Segmentación de datos con K-Means, método del codo y visualización de clusters |
| `Aplicaciones del Análisis de componentes principales (PCA).ipynb` | Reducción de dimensionalidad con PCA, varianza explicada y reconstrucción de datos |
| `Comparación de DBSCAN y HDBSCAN.ipynb` | Clustering basado en densidad: comparativa de ambos algoritmos en distintas geometrías de datos |
| `t-SNE y UMAP.ipynb` | Visualización 2D de datasets de alta dimensión, comparativa de ambas técnicas |

---

## Material de apoyo

- `Aprendizaje_Automático_Tema_4.pdf` — Diapositivas del módulo
- `Hoja de trucos-Construcción de modelos de Aprendizaje no supervisado.pdf` — Resumen de algoritmos

---

## Librerías utilizadas

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hdbscan
import umap
```
