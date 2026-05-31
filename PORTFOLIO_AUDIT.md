# Auditoria del portfolio

Revision realizada para dejar el repositorio mas presentable como portfolio academico/profesional.

## Cambios aplicados

- Eliminados archivos vacios sin contenido util:
  - `grado-matematicas/grasp-path-relinking-mdp/resultadosGRASPPR.txt`
  - `grado-matematicas/grasp-path-relinking-mdp/Pycode GRASPPR for MDP (for students)/pathfunctions/cosas.py`
  - `master/Mineria de Datos/.Rhistory`
- Ampliado `.gitignore` para evitar historiales de R, caches de notebooks, caches de Python, temporales de Office y auxiliares de LaTeX.
- Revisados posibles secretos o credenciales: no se han encontrado patrones evidentes de tokens, claves privadas, API keys o passwords.

## Observaciones

- El repositorio funciona como archivo academico amplio, pero no todos los proyectos tienen el mismo nivel de acabado.
- Hay archivos pesados que conviene mantener solo si son necesarios para reproducir o entender el trabajo. El mas relevante es `master/Mineria de Datos/miniweb-marocars/web/model-data.js`, de unos 46 MB.
- Algunas rutas contienen espacios, acentos y nombres largos. GitHub los maneja, pero para un portfolio profesional es mas robusto usar nombres cortos y consistentes en futuros proyectos.
- Hay materiales de cursos, PDFs, notebooks y codigo mezclados. Para lectura externa, los README por carpeta son importantes porque explican que debe mirar una persona reclutadora.

## Recomendacion

Mantener este repositorio como portfolio academico general y destacar desde la web personal solo los proyectos mas fuertes: series temporales, analisis multivariante, optimizacion GRASP/Path Relinking, TFG y deep learning aplicado.
