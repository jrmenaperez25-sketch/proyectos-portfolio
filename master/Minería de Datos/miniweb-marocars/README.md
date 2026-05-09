# SouqAuto Maroc

Miniweb en `HTML/CSS/JS` con backend en `R` para:

- estimar `Price` con el `Random Forest`
- calcular la probabilidad de `First.Owner` con la regresion logistica
- descargar un PDF con el resultado y los datos introducidos

## Estructura

- `api/train_models.R`: reentrena y guarda los modelos en `models/model_bundle.rds`
- `api/plumber.R`: expone la API con `/metadata`, `/predict` y `/health`
- `web/index.html`: interfaz principal
- `web/config.js`: configuracion local de la URL de la API
- `web/app.js`: logica del formulario, peticiones y PDF
- `web/styles.css`: estilos

## Dependencias en R

Necesitas tener instalados estos paquetes:

```r
install.packages(c("readxl", "caret", "randomForest", "plumber"))
```

## Paso 1. Datos locales

La miniweb usa su propia copia del dataset en:

- `data/datos_imputados_knn.xlsx`

No depende del `.Rmd` ni necesita leer el Excel desde fuera de `miniweb-marocars`.

## Paso 2. Guardar los modelos

Desde la carpeta `miniweb-marocars/api`:

```powershell
Rscript train_models.R
```

Si `Rscript` no esta en el PATH, puedes abrir R y ejecutar:

```r
setwd("C:/Users/joser/OneDrive/Desktop/Proyecto Final Minería de Datos/miniweb-marocars/api")
source("train_models.R")
```

## Paso 3. Levantar la API

Desde `miniweb-marocars/api`:

```powershell
Rscript -e "pr <- plumber::plumb('plumber.R'); pr$run(host='127.0.0.1', port=8000)"
```

O desde una sesion de R:

```r
setwd("C:/Users/joser/OneDrive/Desktop/Proyecto Final Minería de Datos/miniweb-marocars/api")
pr <- plumber::plumb("plumber.R")
pr$run(host = "127.0.0.1", port = 8000)
```

## Paso 4. Abrir la miniweb

Abre `web/index.html` en el navegador.

Antes, crea `web/config.js` a partir de `web/config.example.js`.

Si prefieres servirla localmente:

```powershell
cd "C:\Users\joser\OneDrive\Desktop\Proyecto Final Minería de Datos\miniweb-marocars\web"
python -m http.server 5500
```

Luego entra en `http://127.0.0.1:5500`.

## Para publicarla con enlace

GitHub por si solo no basta, porque la prediccion depende de una API en `R`.

Necesitas desplegar:

- el frontend estatico `web/`
- la API `plumber` de `api/`

La web ya esta preparada para eso: cuando publiques la API, solo tendras que poner su URL en `web/config.js`.

Ejemplo:

```js
window.APP_CONFIG = {
  API_BASE_URL: "https://tu-api-publica.onrender.com"
};
```

## Despliegue recomendado

### Backend en Render

El backend ya incluye:

- `Dockerfile`
- `api/start-api.R`

Pasos en Render:

1. Crea un `Web Service`.
2. Conecta tu repositorio de GitHub.
3. Usa estos valores:
   - `Root Directory`: `miniweb-marocars`
   - `Language`: `Docker`
4. Despliega.

Render indica que los `Web Services` deben escuchar en `0.0.0.0` y usar el puerto esperado por la plataforma. En este proyecto eso ya queda resuelto en `start-api.R`, que toma `PORT` y arranca `plumber` sobre `0.0.0.0`.[Render Web Services](https://render.com/docs/web-services) [Render Docker](https://render.com/docs/docker)

Cuando el servicio quede desplegado, prueba:

- `https://tu-servicio.onrender.com/health`
- `https://tu-servicio.onrender.com/metadata`

### Frontend en GitHub Pages

El repo ya incluye un workflow:

- `.github/workflows/miniweb-pages.yml`

Pasos en GitHub:

1. Ve a `Settings -> Pages`.
2. En `Source`, selecciona `GitHub Actions`.
3. Haz `push` al repo.

GitHub Pages publica sitios estaticos directamente desde los archivos del repositorio mediante un flujo de build/deploy, asi que encaja bien con `web/`.[GitHub Pages docs](https://docs.github.com/pages/getting-started-with-github-pages/what-is-github-pages)

## Orden recomendado de publicacion

1. Desplegar primero la API en Render.
2. Copiar la URL publica de Render.
3. Cambiar `web/config.js` para que apunte a esa URL.
4. Hacer `git push`.
5. Dejar que GitHub Pages publique la web.

## Nota sobre `config.js`

Para local:

```js
window.APP_CONFIG = {
  API_BASE_URL: "http://127.0.0.1:8000"
};
```

Para produccion:

```js
window.APP_CONFIG = {
  API_BASE_URL: "https://tu-servicio.onrender.com"
};
```

## Notas del comportamiento

- `Brand` y `Model` se piden para mejorar la experiencia y para incluirlos en el PDF, pero no entran en los modelos.
- `Mileage` es el unico campo libre.
- Si el kilometraje supera `700000`, la web muestra un aviso de anomalia.
- Si la marca pertenece al segmento de lujo, la web muestra un aviso de menor fiabilidad.
