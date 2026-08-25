# Marcadores conductuales objetivos (Behavior/Objective_Markers)

Modela cómo las dimensiones de la sonda MDES (`onoff`, `valence`, `selfother`,
`time`, `confidence`) predicen marcadores conductuales objetivos del SART:
tasa de omisión, tasa de comisión, errores totales y variabilidad de RT
(RTCV). Es el **Paper 2, Sección 1 (Comportamiento)**: establece la validez
de constructo de las dimensiones de sonda antes de las secciones de EEG.

Este documento tiene dos partes: **cómo usar el código** (para quien vaya a
correr o modificar el pipeline) y **qué se hizo y por qué** (para escribir o
revisar la sección de Métodos/Resultados del paper).

---

## Parte 1 — Cómo usar el código

### Qué hace cada script

| Archivo | Rol |
|---|---|
| `aggregate_objective_markers_by_probe.py` | Upstream. Convierte datos del SART a nivel de ensayo en una fila por sonda: `n_go`, `n_nogo`, `n_omissions`, `n_commissions`, `rt_mean`, `rt_sd`, `rtcv`. Genera los dos CSV de entrada (`full_segment` y `n10`). |
| `objective_markers_analysis.py` | Análisis descriptivo de los marcadores (distribuciones, comparaciones de grupo). Independiente del track LMM/GLMM. |
| `vtc_analysis.py`, `vtc_zone_analysis.py` | Variance Time Course: clasificación "in the zone" / "out of the zone" a partir de la variabilidad de RT. Independiente de este pipeline. |
| **`lmm_probe_dimensions.py`** | **Punto de entrada principal.** Ajusta los cuatro tracks de modelos, corre el análisis de moderación, y genera todas las figuras y tablas. |
| `glmm_config.yaml` | Todos los parámetros del GLMM: mapa marcador→familia, columnas de respuesta, transformaciones, ruta de Rscript, optimizador. Nada hardcodeado en Python. |
| `response_transforms.py` | Funciones puras — construcción de la respuesta binomial, logit empírico con corrección de Haldane, transformación log. Sin I/O, sin R. |
| `glmm_fit.R` | Ajusta un `lme4::glmer`, escribe coeficientes + diagnósticos + residuos como CSV. |
| `glmm_backend.py` | Frontera Python↔R. Devuelve resultados en el **mismo esquema tidy que `fit_lmm`**, así todo el código de plotting funciona sin modificarse. |
| `diagnostics.py` | Diagnósticos de residuos: QQ, residuos vs. ajustados, residuos binned, Breusch-Pagan. |

### Cómo correrlo

```bash
<PATH_TO_YOUR_PLOTS_CONDA_ENV>/bin/python \
  Behavior/Objective_Markers/lmm_probe_dimensions.py
```

**Usar el env `plots`, no `eeg`**: `eeg` tiene un par plotly/kaleido
incompatible (5.18 + 1.0) que rompe la exportación de imágenes estáticas.
Corre ambos datasets (`full_segment` y `n10`), los cuatro tracks, y genera
~150 figuras. Tarda varios minutos (112 ajustes `glmer` + ~50 gaussianos +
figuras).

Para regenerar solo las figuras sin reajustar los modelos (ej. tras cambiar
una etiqueta o color):

```bash
<PATH_TO_YOUR_PLOTS_CONDA_ENV>/bin/python \
  Behavior/Objective_Markers/lmm_probe_dimensions.py --plots-only
```

### Tests

```bash
<PATH_TO_YOUR_EEG_CONDA_ENV>/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -v -o addopts=""
```

`eeg` tiene pytest; `plots` no. 15 tests: transformaciones puras, ida y
vuelta Python↔R con un efecto log-odds conocido inyectado (recupera β=0.8),
consistencia de conteos en los datos reales, contrato de la moderación,
diagnósticos de residuos.

### Dependencia de R

`glmer` corre como **subproceso** `Rscript`, no vía `rpy2`. En esta red
`conda.anaconda.org` está bloqueado por el proxy institucional, `rpy2` no
compila contra el R de conda, y `lme4` tampoco compila ahí (falta
`SHLIB_LIBADD` en el `Makeconf` de conda-R). Pero `lme4` 2.0.1 ya está
instalado en el módulo `R/4.6.0`:

```
Rscript: <PATH_TO_YOUR_RSCRIPT_BINARY>
```

Verificado que este binario funciona con entorno limpio, sin necesitar
`module load`. La ruta absoluta está en `glmm_config.yaml`.

### Estructura de salida

```
results/Behavior/objective_markers/lmm_probe_dimensions/<dataset>/
├── glmm/                      ← PRIMARIO
├── sensitivity_olre/          ← corregido por sobredispersión
├── sensitivity_gaussian/      ← modelo original mal especificado
├── sensitivity_transformed/   ← respuesta log / logit empírico
├── diagnostics/               ← chequeos de residuos de todos los tracks
├── model_comparison.csv       ← los 4 tracks lado a lado
├── quadratic_orthogonalization.csv
└── used_config.yaml
```

Detalle completo de qué contiene cada archivo:
`results/Behavior/objective_markers/lmm_probe_dimensions/README.md`.

**Nota**: existe un directorio `full_model/` remanente de la versión anterior
del pipeline (antes del respec a GLMM). Ya no se escribe; queda ahí como
resultado histórico hasta que se decida borrarlo.

---

## Parte 2 — Qué se hizo y por qué (para el paper)

### 2.1 El problema de partida

El pipeline original ajustaba un **LMM gaussiano** (`statsmodels.mixedlm`)
sobre los cuatro marcadores y **nunca validaba ningún supuesto
distribucional**: no había extracción de residuos, ni QQ-plot, ni test de
normalidad u homocedasticidad en ningún lugar del módulo.

Las distribuciones reales (dataset `full_segment`, n=2460 sondas, 42
sujetos) violan los supuestos gaussianos de forma severa, no marginal:

| Marcador | Rango | Skew | % ceros |
|---|---|---|---|
| `omission_rate` | [0, 1] | 6.08 | 65.2 % |
| `commission_rate` | [0, 1] | 1.59 | 66.1 % |
| `rtcv` | (0.071, 2.715] | 6.65 | 0 % |

El problema central son los denominadores de las tasas de error:

| Dataset | `n_go` (denom. omisión) | `n_nogo` (denom. comisión) |
|---|---|---|
| `full_segment` | 14–37 (mediana 27) | **1–4 (mediana 3)** |
| `n10` | 9–10 | **0 o 1** |

`commission_rate` en `full_segment` es una proporción sobre 1 a 4 ensayos
(toma valores como 0, ⅓, ½, 1). En `n10` es **binaria**, y 847 de 2460
sondas la tienen indefinida (`n_nogo=0`). Un LMM gaussiano sobre un
resultado binario no es defendible: puede predecir valores fuera de [0,1] y
asume una estructura de varianza que los datos no tienen.

**Por qué la transformación sola no alcanza para las tasas.** Aplicar logit
o arcoseno a una proporción con denominador 2 requiere una constante de
continuidad arbitraria, y sobre una variable binaria no tiene sentido. La
transformación es apropiada únicamente para `rtcv` (continua, positiva,
skew 6.65). Las tasas de error necesitan un modelo que trate los conteos
como conteos.

Especificación completa de la decisión: [`docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md`](../../docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md).

### 2.2 Especificación de los modelos

| Marcador | Respuesta | Familia | Link |
|---|---|---|---|
| `omission_rate` | `cbind(n_omissions, n_go - n_omissions)` | binomial | logit |
| `commission_rate` | `cbind(n_commissions, n_nogo - n_commissions)` | binomial | logit |
| `total_errors` | `cbind(n_om + n_com, n_trials - n_om - n_com)` | binomial | logit |
| `rtcv` | `rtcv` (>0) | Gamma | log |

Efectos fijos y aleatorios sin cambios respecto del modelo original:

```
marker ~ onoff + valence + valence_sq + selfother + time + time_sq
         + confidence + time_on_task + (1|subject)
```

- Los predictores están z-scoreados (β en unidades de SD).
- Los términos cuadráticos (`valence_sq`, `time_sq`) están centrados en el
  punto medio de la escala y ortogonalizados globalmente contra su término
  lineal por OLS pooled (parametrización `poly(x,2)`), lo que no cambia el
  estimador/SE/p del término cuadrático pero reduce su colinealidad con el
  lineal.
- **Los coeficientes están en escala de link**: log-odds para los
  marcadores binomiales, log para `rtcv`. No son unidades de tasa.
- El intercepto de un GLMM con intercepto aleatorio es **condicional**
  (sujeto mediano), no la media marginal. Ejemplo verificado: para
  `omission_rate` el intercepto GLMM es −4.20 → 1.47%, que coincide con la
  *mediana* de tasas por sujeto (1.51%), no con la media pooled (3.83%).
  Esperado, no un bug.
- Optimización: máxima verosimilitud con BOBYQA. Tests de efectos fijos:
  Wald z (no t — ni `lme4::glmer` ni `statsmodels.mixedlm` reportan t;
  ambos usan aproximación normal).

**`total_errors` fue redefinido** como
`(n_omissions + n_commissions) / n_trials_window` — una proporción genuina
sobre todos los ensayos de la ventana. La definición anterior
(`omission_rate + commission_rate`) sumaba dos tasas con denominadores
distintos (~27 ensayos go vs ~3 no-go) y por lo tanto no era proporción de
nada bien definido. La definición vieja se conserva como
`total_errors_legacy`, ajustada únicamente en el track gaussiano, por
continuidad con resultados ya publicados.

### 2.3 Por qué cuatro tracks (y no solo el GLMM)

La elección de familia se fijó **antes** de ver los resultados. Tres
análisis de sensibilidad se especificaron de antemano y se reportan sin
importar el resultado — así no hay elección de especificación posterior a
mirar los datos (garden of forking paths):

1. **`glmm/` — primario.** Binomial/Gamma como se describe arriba.
2. **`sensitivity_olre/`** — el *mismo* modelo binomial más un efecto
   aleatorio a nivel de observación, `(1|subject) + (1|obs_id)`. No es otra
   familia: el binomial simple impone que la varianza sea exactamente
   `n·p·(1−p)`; cuando los datos varían más que eso, los errores estándar
   del binomial simple salen subestimados. El OLRE le da a cada sonda su
   propia desviación aleatoria para absorber ese exceso. Solo se ajusta
   para los tres marcadores binomiales — `rtcv` (Gamma) ya tiene su propio
   parámetro de dispersión.
3. **`sensitivity_gaussian/`** — el LMM gaussiano original, para que el
   efecto de la respecificación sea visible, no asumido. Único track que
   contiene `total_errors_legacy`.
4. **`sensitivity_transformed/`** — LMM gaussiano sobre `log(rtcv)` y logit
   empírico con corrección de Haldane
   (`log[(y+0.5)/(n-y+0.5)]`) para las tasas, que se mantiene finito incluso
   sin eventos observados.

### 2.4 Convergencia y ajustes singulares

Corrida completa verificada: **112 ajustes `glmer`, 0 fallos de
convergencia** (exit 0). Cada CSV de resultados guarda `converged`,
`singular` (`isSingular()` de lme4), `re_sd` (SD de cada efecto aleatorio),
`max_grad` y `dispersion` (Pearson χ²/df).

**Un hallazgo relevante**: el OLRE resultó **singular** para
`commission_rate` en ambos datasets (`obs_id` colapsa a varianza 0). Esto
es coherente con su dispersión (0.92 en `full_segment`, 0.88 en `n10`) —
por debajo de 1, o sea que no hay sobredispersión que absorber. Los
coeficientes del OLRE singular fueron **idénticos al binomial simple hasta
el cuarto decimal**. Esto no es un fallo del modelo: confirma que el
binomial simple ya es adecuado para ese marcador. No se excluye nada de
`commission_rate`; se reporta la singularidad como nota.

Para `omission_rate` y `total_errors`, en cambio, el OLRE sí absorbe
varianza real: la dispersión baja de 1.95→0.33 y 1.73→0.43
respectivamente, con SD de `obs_id` de 1.09 y 0.76. Ahí los coeficientes
casi no se mueven pero los SE se **inflan ~1.4–2.0×** — el binomial simple
estaba subestimando la incertidumbre.

### 2.5 Multiple comparisons

FDR Benjamini-Hochberg. En el modelo aditivo, corrección a través de los 8
predictores dentro de cada modelo. En el análisis de moderación, corrección
conjunta a través de todos los tests marcador × moderador. α = 0.05 en
ambos casos.

### 2.6 Diagnóstico de residuos

Se calcula para **todos los modelos, en los cuatro tracks** — esto es lo
que faltaba en el pipeline original y motivó todo el trabajo. Con n≈2460,
un test formal de normalidad (Shapiro-Wilk) rechaza ante desviaciones
triviales, así que la normalidad se reporta de forma descriptiva (skew,
kurtosis) más gráficos, no como test binario pasa/no-pasa.

- QQ-plot y residuos vs. ajustados para todos los modelos.
- Breusch-Pagan para homocedasticidad.
- **Residuos binned** (Gelman & Hill) para los modelos binomiales, donde
  los residuos crudos no son informativos contra los valores ajustados
  porque la respuesta es discreta. Sustituye a `DHARMa`, que no está
  disponible en esta red (canal de conda bloqueado).

**Resultado**: el respec mejora sustancialmente los residuos sin dejarlos
perfectos. Skew de residuos, gaussiano vs. GLMM (`full_segment`):

| Marcador | Skew gaussiano | Skew GLMM |
|---|---|---|
| `rtcv` | 5.15 | **2.13** |
| `omission_rate` | 4.37 | **2.64** |
| `total_errors` | 4.04 | **2.30** |
| `commission_rate` | 1.28 | 1.39 |

Breusch-Pagan p=0.0 en los cuatro marcadores del track gaussiano (
heterocedasticidad total); en el GLMM mejora pero sigue significativo en la
mayoría — se reporta así, sin sobrevender.

### 2.7 Bug de escala en las figuras (corregido)

Al principio el esquema tidy compartido entre gaussiano y GLMM hacía
"funcionar" el plotting existente sin tocarlo, pero eso ocultaba un error:
`marginal_lmm_line` construía el predictor lineal y lo dibujaba
directamente, sin aplicar la función de link inversa. Para `rtcv` la línea
quedaba en `log(0.25)=-1.39` mientras la nube de datos estaba en 0.25 — la
línea desaparecía del rango visible y el eje Y compartido aplastaba la
nube. Se corrigió aplicando `inverse_link` (logit→sigmoide, log→exp) antes
de graficar, y las etiquetas de los ejes ahora indican la escala real
(`β (log-odds per SD, 95% CI)`, `β (log per SD, 95% CI)`) en vez de un
genérico "SD units" que solo es correcto en los tracks gaussianos.

### 2.8 Qué sobrevive al cambio de especificación — resultados clave

Comparación de significancia (FDR) entre GLMM y gaussiano, en
`model_comparison.csv`:

- **Acuerdo GLMM vs. gaussiano**: 75% en `full_segment`, 84% en `n10`.
- **`rtcv ~ onoff` NO era significativo en el gaussiano y SÍ lo es en el
  GLMM** (β=−0.041, p_fdr=0.00027 en `full_segment`; β=−0.080,
  p_fdr=1.07e-06 en `n10`). El modelo mal especificado estaba *perdiendo*
  un efecto real.
- `onoff` es significativo para los 4 marcadores, en ambos datasets, en
  todos los tracks (incluido OLRE) — el hallazgo más robusto del análisis.
- `time_on_task → rtcv` replica en ambos datasets (β=+0.040 y +0.053).
- 5 efectos en `full_segment` pierden significancia bajo OLRE (`time`,
  `time_sq`, `time_on_task` sobre `omission_rate`; `time_sq`,
  `time_on_task` sobre `total_errors`) — no se reclaman como hallazgos,
  eran artefacto de SE subestimados por el binomial simple.
- Los términos cuadráticos (`time_sq`, `valence_sq`) no replican entre
  datasets — se tratan como exploratorios.
- `selfother` no es significativo para ningún marcador, en ningún dataset,
  en ningún track — nulo limpio.

### 2.9 Moderación

`marker ~ onoff × moderador + (1|subject)`, misma familia de error que el
modelo aditivo correspondiente. 28 modelos por dataset (4 marcadores × 7
moderadores), todos convergen.

- **`onoff × time_on_task` es la única interacción que replica en ambos
  datasets** (`omission_rate` p_fdr=1.27e-14 y 6.27e-05;
  `total_errors` p_fdr=3.30e-10 y 1.50e-03). β negativo: el efecto
  protector de estar on-task se debilita a medida que avanza la tarea.
- Las demás interacciones significativas (`valence_sq`, `time`, `time_sq`
  sobre `omission_rate`/`total_errors`; `valence` sobre `rtcv`) solo
  aparecen en `full_segment` — tratar como exploratorias, y notar que
  varias involucran predictores que ya no sobreviven al OLRE en el modelo
  aditivo.

### 2.10 Versiones y reproducibilidad

```
R 4.6.0, lme4 2.0.1 (módulo R/4.6.0, sin necesitar module load)
Python 3.12.9, statsmodels 0.14.6, numpy 2.4.3, scipy 1.15.3
random_state: 42
```

Cada corrida guarda un snapshot de configuración
(`<dataset>/used_config.yaml`) y la orthogonalización cuadrática aplicada
(`quadratic_orthogonalization.csv`), por transparencia y para poder
reproducir los coeficientes exactos.

### 2.11 Borrador de la sección de Métodos

Redacción lista para pegar en el paper (en inglés), con las notas de qué
reportar en Resultados: [`docs/paper_sections/statistical_analysis_behaviour.md`](../../docs/paper_sections/statistical_analysis_behaviour.md).
