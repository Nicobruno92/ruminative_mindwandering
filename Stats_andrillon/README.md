# Stats_andrillon — Pipeline de Cluster-Based Permutation Testing (CBPT) para EEG × Mind-Wandering

Implementación en Python de la metodología estadística de **Andrillon et al. (2020)**
(cluster-permutation sobre t-values de un LMM) aplicada a los marcadores EEG de este
proyecto. Es el pipeline detrás de la **Sección 2 (CBPT: firmas neurales específicas
por dimensión)** del Paper 2 — ver `CLAUDE.md` en la raíz del repo para la estructura
completa de los dos papers.

> **Pregunta que responde esta sección del paper**: ¿qué marcadores EEG muestran una
> correlación a nivel de grupo con cada dimensión de mind-wandering (onoff, valence,
> selfother, time, confidence), y con qué topografía espacial?

Este documento tiene dos partes:
1. **Guía de uso** — cómo correr, configurar y depurar el pipeline (§1–§6).
2. **Metodología en detalle** — descripción exhaustiva, a nivel de sección de Métodos
   de un paper, de cada decisión estadística tomada y su justificación (§7).

Todo lo que sigue está anclado al código tal como existe en este directorio y en
`Statistics/` (que este pipeline reutiliza) a fecha **2026-07-25**. Los números de
resultados en §8 son un *snapshot* del estado actual de `results/andrillon_cluster/`,
no resultados finales del paper — están marcados como tales.

---

## 1. Instalación / entorno

Este pipeline vive en el entorno conda **`eeg`** (ver `CLAUDE.md` raíz — "for anything
related to eeg analysis => eeg"). Los scripts SLURM ya activan `eeg` automáticamente.
Para trabajo interactivo:

```bash
conda activate eeg
```

Dependencias propias de este pipeline (encima de lo que ya usa `Statistics/`):
- `mne` (adyacencia espacial, montage, `_get_components`, topomaps)
- `joblib` (paralelización de permutaciones, backend `loky`)
- `statsmodels` (backend LMM por defecto)
- **Opcional**: `rpy2` + `r-lme4` + `r-lmertest` (backend `"r"`, requerido sólo para
  WLS-LMM exacto con `obs_weight_col`). Ya están instalados en `eeg` según el
  comentario en `config_andrillon.yaml`.

El pipeline **reutiliza directamente** módulos de `Statistics/` (no los reimplementa):
`reader.py`, `cluster_test.py`, `lmm_model.py`, `lmm_r_backend.py`, `helpers.py`,
`multiple_comparisons.py`, `apply_mcc_postprocessing.py`, `generate_summary_report.py`,
`plot_results.py`. Ver §7 para el porqué de esta reutilización.

---

## 2. Quickstart

```bash
conda activate eeg

# 1. Ver qué markers va a correr la config actual (y cuántos jobs de array necesitás)
python Stats_andrillon/get_marker_list.py --config Stats_andrillon/config_andrillon.yaml --format count

# 2a. Correr UN marker localmente (rápido, para debug)
python Stats_andrillon/run_andrillon_pipeline.py \
    --config Stats_andrillon/config_andrillon.yaml \
    --marker sleep/psd_relative_gamma

# 2b. Correr TODOS los markers de un predictor localmente (secuencial, lento)
bash Stats_andrillon/run_local_andrillon_pipeline.sh --predictor onoff

# 2c. Correr TODOS los markers × TODOS los predictores en SLURM (recomendado)
bash Stats_andrillon/submit_andrillon_predictor_loop.sh
```

La opción 2c es la que se usa en producción: lee `lmm.predictor_of_interest` (una
lista) del config y somete, por cada predictor, un array job (uno por marker) más un
job de reporte dependiente que corre automáticamente al terminar.

---

## 3. Configuración (`config_andrillon.yaml`)

Todo parámetro vive acá — **nada está hardcodeado en el código** (Regla crítica #2 del
`CLAUDE.md` raíz). Secciones principales:

| Sección | Qué controla |
|---|---|
| `project` | Rutas (features, montage, output), sujetos/tasks incluidos, filtro de variabilidad del predictor |
| `feature_families` / `selected_markers` | Qué marcadores se corren, agrupados por epoch type (`evoked` vs `sleep`) — ver §7.1 |
| `derived_ratios` | Marcadores "virtuales" calculados como numerador/denominador channel-wise |
| `preprocessing` | Normalización por sujeto, transform de respuesta, features cuadráticas ortogonalizadas |
| `lmm` | Fórmula, predictor(es) de interés, backend (`python`/`r`), pesos de observación (WLS) |
| `andrillon_clustering` | `cluster_alpha`, `n_permutations`, `montecarlo_alpha`, método de permutación, exclusión de canales de borde |
| `multiple_comparisons` | Unidad de corrección, método por familia (evoked/state/sleep), `alpha`, `require_complete_family` |
| `output` | Qué se guarda a disco |

Cada decisión no obvia está documentada **in situ** como comentario largo en el YAML
(por qué `n_jobs=16` y no `-1`, por qué `permutation_method: freedman_lane`, por qué
BH y no Bonferroni, etc.) — son la fuente de verdad; este README las resume pero el
YAML tiene el detalle completo y debe seguir siendo la referencia autoritativa si
diverge de lo que dice acá.

**Para cambiar qué se corre:**
- Un solo predictor → `lmm.predictor_of_interest: "onoff"` (string).
- Varios predictores (uno por directorio de salida) → lista de strings; usar
  `submit_andrillon_predictor_loop.sh`, no `run_andrillon_pipeline.py` directo (ese
  aborta con un mensaje explícito si recibe una lista de >1 elemento).
- Un predictor de interacción (ej. `"onoff:confidence"`) → agregar override en
  `lmm.per_predictor_formula` con la fórmula multiplicativa; `_resolve_formula_for_predictor`
  valida que el override efectivamente contenga el término de interacción.

---

## 4. Ejecución paso a paso

### 4.1 Local, un marker (debug)

```bash
python Stats_andrillon/run_andrillon_pipeline.py \
    --config Stats_andrillon/config_andrillon.yaml \
    --marker evoked/P3b \
    --predictor-of-interest onoff
```

Corre el LMM observado + `n_permutations` permutaciones **para ese marker solo**, en
el proceso actual (usar `andrillon_clustering.n_jobs` chico o `1` para tracebacks
legibles — `n_jobs=1` fuerza ejecución secuencial en `andrillon_pipeline.py`).

### 4.2 Local, todos los markers de un predictor

```bash
bash Stats_andrillon/run_local_andrillon_pipeline.sh --predictor onoff
```

Loop secuencial sobre todos los markers resueltos por `get_marker_list()`, seguido de
`generate_summary_report.py`. **No aplica la corrección MCC** — para eso hace falta
correr `apply_mcc_postprocessing.py` aparte (ver 4.4). Pensado para correr en un nodo
interactivo, no en el login node (cada marker fitea 23 LMMs × 5000 permutaciones).

### 4.3 SLURM — producción

```
submit_andrillon_predictor_loop.sh
        │  (lee lmm.predictor_of_interest; si es lista, itera)
        ▼
submit_parallel_andrillon_markers.sh --predictor <p>
        │  (cuenta markers vía get_marker_list, genera el array script,
        │   somete el array job, y somete un job de reporte con
        │   --dependency=afterok:<array_job_id>)
        ▼
run_andrillon_marker_array.sh   (uno por marker, SLURM array)
        │  cada task: run_andrillon_pipeline.py --marker-index $SLURM_ARRAY_TASK_ID
        ▼
run_andrillon_report_generation.sh   (sólo si TODOS los markers exit 0)
        │  1. Statistics/apply_mcc_postprocessing.py <model_dir>
        │  2. Statistics/generate_summary_report.py <model_dir>
```

Puntos importantes:
- El array job propaga el exit code de Python (`exit ${EXIT_CODE}`) — si no lo
  hiciera, SLURM registraría `COMPLETED 0:0` para un marker que en realidad murió,
  y el job de reporte (que depende de `afterok`) correría igual sobre datos
  incompletos.
- `#SBATCH --cpus-per-task` del array job **debe coincidir** con
  `andrillon_clustering.n_jobs`: `loky` (el backend de `joblib`) lee el conteo físico
  de cores del nodo, no la asignación de cgroup, así que un `n_jobs` más alto que lo
  asignado sobre-suscribe CPUs y ralentiza brutalmente la corrida (ver comentario en
  el YAML: 5000 permutaciones tardaron ~8h en 2 cores por este motivo antes del fix).
- Para re-generar sólo el reporte de un modelo ya corrido (sin re-someter el array):

  ```bash
  sbatch Stats_andrillon/run_andrillon_report.sh <MODEL_DIR> [CONFIG_FILE]
  ```

### 4.4 Corrección por comparaciones múltiples (MCC) — manual

Si corriste markers sueltos y necesitás aplicar/reaplicar la corrección sin volver a
correr todo el array:

```bash
python Statistics/apply_mcc_postprocessing.py \
    results/andrillon_cluster/<model_folder> \
    --config Stats_andrillon/config_andrillon.yaml
```

Esto: (1) carga todos los `results.pkl` del directorio, (2) descarta cualquier marker
que no esté declarado en `selected_markers`/`feature_families` del config actual
(evita que un marker viejo, de una corrida anterior, infle o contamine la familia),
(3) aplica la corrección configurada, (4) **persiste** los p-valores corregidos de
vuelta en los pickles (`results.pkl` y `<marker>_results.pkl`), (5) escribe
`multiple_comparisons_summary.csv` y `mcc_family_composition.csv`.

⚠️ **Se debe correr después de cada cambio en `feature_families`/`selected_markers`**,
incluso si no se re-corrió ningún marker: la familia de corrección (y por lo tanto
todos los p-valores corregidos) depende de qué markers *declara* el config, no de qué
carpetas hay en disco.

### 4.5 Reporte resumen (topoplots + tablas)

```bash
python Statistics/generate_summary_report.py results/andrillon_cluster/<model_folder>
```

Lee `results.pkl` de cada marker y genera, dentro del `model_folder`:
`SUMMARY_REPORT_<timestamp>.csv`, `SUMMARY_DETAILED_<timestamp>.xlsx`,
`SUMMARY_TOPOPLOTS_<timestamp>.pdf`. Se corre automáticamente al final del pipeline
SLURM (§4.3) y también al final de `run_local_andrillon_pipeline.sh`.

### 4.6 Diagnóstico de supuestos del LMM

```bash
sbatch Stats_andrillon/run_andrillon_assumption_diagnostics.sh
# o interactivo:
python Stats_andrillon/lmm_assumption_diagnostics.py \
    --config Stats_andrillon/config_andrillon.yaml \
    --predictor onoff --marker sleep/psd_relative_gamma   # opcional, restringe a 1×1
```

Re-fitea **sólo el modelo observado** (sin permutaciones — barato) para cada
combinación predictor × marker y escribe, bajo
`<output_path>/<model_folder>/assumption_diagnostics/`:
- `<marker>_channel_diagnostics.csv` — un renglón por canal (R² condicional,
  varianza residual, skew, exceso de curtosis, % outliers, p de Breusch-Pagan).
- `assumption_summary.csv` — un renglón por marker (promedios/máximos de lo anterior).
- `assumption_summary.png` — barras horizontales por marker, sólido si supera el
  umbral de alerta (`RESIDUAL_SKEW_FLAG=1.0`, `RESIDUAL_EXKURT_FLAG=3.0` en
  `Statistics/lmm_model.py`), hueco si no.

Ver §7.8 para qué significan estos diagnósticos y por qué **no** se usa un test de
normalidad formal (Shapiro-Wilk).

### 4.7 Test omnibus de familia

```bash
python Stats_andrillon/omnibus_test.py --config Stats_andrillon/config_andrillon.yaml
# o restringido a un solo target:
python Stats_andrillon/omnibus_test.py --config ... --model-dir results/andrillon_cluster/<model_folder>
```

⚠️ **Gotcha de uso**: el output (`results/andrillon_cluster/omnibus_test.csv`) es un
**único archivo** para todos los `model_dir`s, y cada invocación lo **sobreescribe
por completo** con sólo las filas de los `model_dir`s que procesó esa vez. Si corrés
con `--model-dir` apuntando a un solo target, el CSV en disco queda con las filas de
*ese* target únicamente — no se acumula entre corridas. Para tener la tabla completa
de los 6 targets hay que correrlo **sin** `--model-dir` (recorre todos los
subdirectorios de `output_path`). *(Estado actual en disco: el CSV sólo tiene las dos
filas de `target_valence_sq` — ver §8.4.)*

Requiere que cada marker tenga `null_cluster_stats` en su `results.pkl` (lo escribe
`andrillon_pipeline.py` desde que se agregó esta feature — commit `8e4acce`); markers
corridos con una versión anterior del pipeline no son usables acá y hay que
re-correrlos.

### 4.8 Figura resumen del Paper 2 (topomaps + heatmap)

```bash
conda activate plots
python3.12 Stats_andrillon/plot_cbpt_summary_figure.py
```

Genera `results/figures/cbpt_summary_figure.{png,pdf}` — Panel A: topomaps de 4
markers representativos (P3b, PE_gamma, PSD gamma, PSD alpha) × 4 dimensiones
principales; Panel B: heatmap de todos los markers con ≥1 cluster significativo ×
6 columnas (4 dimensiones + 2 interacciones con confidence). Ambos paneles leen los
`*_clusters.csv` en disco directamente (no los pickles) y resuelven qué markers
"están activos" leyendo `feature_families`/`selected_markers` del config actual — un
marker sacado del config desaparece automáticamente de la figura sin tocar este
script.

---

## 5. Estructura de salidas en disco

```
results/andrillon_cluster/
├── omnibus_test.csv                         # ver gotcha en §4.7
├── assumption_summary.csv                   # combinado, todos los predictores corridos
└── {fixed_effects}__target_{predictor}/     # un directorio por predictor de interés
    ├── evoked_P3b/
    │   ├── results.pkl                      # genérico, leído por generate_summary_report
    │   ├── evoked_P3b_results.pkl           # mismo contenido, nombre explícito
    │   ├── evoked_P3b_clusters.csv          # clusters SIN corregir
    │   ├── cluster_summary_corrected.csv    # sólo tras correr apply_mcc_postprocessing
    │   ├── evoked_P3b_convergence_summary.csv
    │   ├── evoked_P3b_perm_diagnostics.csv
    │   └── *.png                            # topomap, distribución nula, etc.
    ├── sleep_psd_relative_gamma/ ...
    ├── assumption_diagnostics/              # §4.6
    ├── mcc_family_composition.csv           # §4.4, auditoría de qué entró a la familia
    ├── multiple_comparisons_summary.csv     # §4.4, un renglón por marker
    ├── SUMMARY_REPORT_<ts>.csv
    ├── SUMMARY_DETAILED_<ts>.xlsx
    └── SUMMARY_TOPOPLOTS_<ts>.pdf
```

El nombre del directorio de modelo (`{fixed_effects}__target_{predictor}`) lo genera
`Statistics/helpers.py::get_model_folder_name()`: la parte `fixed_effects` es la
fórmula completa (todos los predictores del modelo, en orden), y `target_{predictor}`
identifica cuál de esos predictores es el que se testea con permutaciones. Esto
permite correr la misma fórmula con distintos predictores de interés sin pisar
resultados entre sí.

---

## 6. Arquitectura — qué reutiliza y qué es propio de `Stats_andrillon/`

```
Stats_andrillon/                              Statistics/  (reutilizado, NO reimplementado)
├── config_andrillon.yaml                     ├── reader.py            (carga BIDS/junifer → DataFrame)
├── andrillon_pipeline.py  ───────imports────>├── cluster_test.py       (adyacencia, find_clusters_from_pvalues)
│     get_marker_list()                       ├── lmm_model.py          (run_lmm_per_channel, Freedman-Lane, diagnósticos)
│     load_marker_matrix()                    ├── lmm_r_backend.py      (lme4::lmer vía rpy2, WLS)
│     run_marker_analysis()          ────────>├── helpers.py            (normalización, transforms, quadratic features)
│     save_results()                          ├── multiple_comparisons.py (BH/BY/Bonferroni, unit=marker)
│     run_andrillon_pipeline()       ────────>├── apply_mcc_postprocessing.py (orquesta la corrección + persiste)
├── run_andrillon_pipeline.py (CLI)           ├── generate_summary_report.py (CSV/XLSX/PDF)
├── get_marker_list.py (CLI helper)           └── plot_results.py       (topomaps)
├── lmm_assumption_diagnostics.py
├── omnibus_test.py
├── plot_cbpt_summary_figure.py
└── run_*.sh / submit_*.sh                    (orquestación SLURM)
```

**Por qué se reutiliza en vez de reimplementar**: el pipeline "clásico" en
`Statistics/` (usado para el Paper 1 / análisis exploratorio inicial) ya tenía
resueltos los problemas difíciles — alineación canal↔adyacencia con invariantes que
abortan la corrida si se rompen, manejo de NaN por canal fallido, Freedman-Lane
correcto, backend R para WLS-LMM. `Stats_andrillon/andrillon_pipeline.py` es
esencialmente una capa fina que: (1) resuelve la lista de markers desde
`feature_families`/`selected_markers`, (2) llama a `run_lmm_per_channel` para el dato
observado, (3) genera las permutaciones con el esquema Andrillon (permutar el
predictor, o Freedman-Lane) en lugar del t-threshold fijo del pipeline clásico, (4)
llama a `find_clusters_from_pvalues` (que sí es específica de Andrillon — clusters
candidatos definidos por **p-valor**, no por t-threshold) en `Statistics/cluster_test.py`.

Los módulos originales del plan de implementación de 2024
(`lmm_permutation.py`, `cluster_detection.py`) **fueron descartados** durante el
desarrollo a favor de esta reutilización — ya no existen en el directorio, aunque
`tests/test_permutation.py` y `tests/test_clustering.py` todavía los importan (ver
§9, deuda técnica).

---

## 7. Análisis en detalle — sección de Métodos

Esta sección describe, con el nivel de detalle esperable en la sección de Métodos de
un paper de Nature (o su material suplementario), cada decisión estadística del
pipeline, su justificación y su implementación exacta. Todo lo que sigue es
determinístico y controlado por `config_andrillon.yaml` — no hay ramas condicionales
ocultas en el código (regla "Determinism" del `CLAUDE.md` raíz).

### 7.1 Diseño y datos de entrada

- **Diseño**: within-subject, medidas repetidas. 42 sujetos nominales (IDs `"02"`–`"43"`
  en el config; el número efectivo de sujetos por marker depende de qué tan variable
  fue su respuesta en el predictor de interés — ver filtro de variabilidad más abajo),
  4 tareas SART por sujeto (`Sart1`–`Sart4`), hasta 15 probes MDES por tarea.
- **Unidad de observación**: una época pre-probe (ventana de distancia `-5` a `-1`
  respecto del probe, epoch type `"sleep"`) o una época evocada bloqueada al estímulo
  (epoch type `"evoked"`), por canal.
- **Predictores de interés testeados actualmente**: `onoff`, `valence`, `valence_sq`,
  `selfother`, `time`, `time_sq` (uno a la vez — cada uno define su propio
  `model_folder` y su propia serie de permutaciones, ya que el t-value que se permuta
  es siempre el del predictor de interés dentro de un modelo que incluye a todos los
  demás como covariables).
- **Familias de markers evaluadas** (declaradas en `feature_families` ×
  `selected_markers`, no descubiertas desde disco — ver §7.6):
  - `evoked` (m=4): ERPs `P1`, `N1`, `P3a`, `P3b`.
  - `sleep` (m=19): `spectral_relative` (delta/theta/alpha/beta/gamma relativos, 5),
    `connectivity` (wSMI theta/alpha/beta/gamma, 4), `information_theory` (PE
    theta/alpha/beta/gamma + Kolmogorov complexity, 5), `sleep` (slow-wave density,
    duration, frequency, PTP, slope, 5).
  - Se excluyeron deliberadamente del correction set: `state` (duplicaba
    exactamente las mismas cantidades espectrales/conectividad/IT ya computadas
    sobre las épocas `sleep`), `spectral` absoluto (duplicaba `spectral_relative`),
    `psd_bands_*` ratios (comentados, exploratorios), y `spindles` (matriz de diseño
    singular en todos los canales bajo `treatment*mind_state` por escasez de
    observaciones no-cero por celda — documentado en el YAML).
- **Filtro de variabilidad del predictor**: sujetos cuya varianza en el predictor de
  interés es menor a `min_predictor_variability` (30 puntos en escala 0–100 para
  predictores lineales, 15 para los cuadráticos — la mitad, porque `_sq` vive en
  escala 0–50) son excluidos **para ese predictor específico** — un sujeto puede
  entrar en el análisis de `onoff` y quedar afuera del de `time` si nunca varió su
  reporte de orientación temporal. Ver §7.1.1 para el detalle exacto (cuántos sujetos,
  cuáles).
- **QA**: `qa_summary_path` y `exclude_failed_qa: true` están **declarados en el
  config pero no tienen ningún efecto en `Stats_andrillon`** — ver §7.1.1, es una
  brecha real entre lo que dice la config y lo que hace el código.

#### 7.1.1 Exclusión de sujetos — qué se elimina y cuántos, en números reales

Esta subsección responde con precisión a "¿cuántos sujetos se pierden y por qué?".
Los números salen de correr **la lógica real del pipeline**
(`Statistics/reader.py::filter_subjects_by_variability`, la misma función que usa
`prepare_data_for_lmm`) sobre los datos crudos — no son una estimación. Verificados
además contra `n_subjects` en los `results.pkl` ya generados (coincide exactamente:
`onoff` → 35/42 en ambas fuentes).

**Paso 1 — filtro de QA: configurado, pero NO aplicado.** `config_andrillon.yaml`
declara `qa_summary_path` y `exclude_failed_qa: true`, y `Statistics/run_pipeline.py`
(el pipeline "clásico") sí los usa (`load_qa_summary` + `get_qa_exclusion_list`,
pasados a `load_all_probe_data` como `qa_exclusions`). **`Stats_andrillon/andrillon_pipeline.py`
nunca importa ni llama a ninguna de esas dos funciones, y nunca pasa `qa_exclusions`
a `load_all_probe_data`** (verificado por `grep` sobre todo el archivo — cero
coincidencias con `qa_summary`/`qa_exclusion`/`exclude_failed_qa`). Efecto práctico:
**0 de 42 sujetos son excluidos por QA en este pipeline**, sin importar cuántos hayan
fallado el control de calidad — el filtro que el config promete simplemente no
corre. Esto es una brecha real de código, no una decisión de diseño: está en §9 como
ítem de deuda técnica.

**Paso 2 — filtro de variabilidad del predictor: el único que efectivamente remueve
sujetos.** `range = max − min` del valor crudo del predictor a través de todas las
probes del sujeto; se excluye si `range` no supera el umbral (30 para predictores
lineales 0–100, 15 para los cuadráticos `_sq` en escala 0–50). Resultado, partiendo
de los 42 sujetos nominales (`"02"`–`"43"`), **idéntico entre markers de la familia
`evoked` y `sleep`** (el filtro opera sobre datos conductuales, no sobre el marcador
EEG en sí):

| Predictor | Umbral (range >) | N final | Excluidos | IDs excluidos |
|---|---|---|---|---|
| `onoff` | 30 | **35/42** | 7 | 02, 04, 07, 09, 13, 16, 20 |
| `valence` | 30 | **33/42** | 9 | 07, 09, 13, 16, 17, 19, 20, 33, 41 |
| `valence_sq` | 15 | **31/42** | 11 | 09, 13, 14, 16, 17, 19, 20, 22, 23, 33, 41 |
| `selfother` | 30 | **37/42** | 5 | 02, 09, 13, 16, 20 |
| `time` | 30 | **35/42** | 7 | 02, 13, 16, 17, 19, 20, 33 |
| `time_sq` | 15 | **34/42** | 8 | 02, 13, 16, 17, 19, 20, 24, 33 |

**Los excluidos no son un conjunto aleatorio distinto por predictor — hay un núcleo
de sujetos "planos" que se repite**: 13, 16 y 20 se excluyen de **las 6** dimensiones
(nunca variaron lo suficiente en ningún reporte MDES); 02 se excluye de 4/6
(`onoff`, `selfother`, `time`, `time_sq`, pero no de `valence`/`valence_sq`); 09 se
excluye de 4/6 (`onoff`, `valence`, `valence_sq`, `selfother`, pero no de
`time`/`time_sq`); 17, 19, 33 se excluyen juntos de `valence`, `valence_sq`, `time` y
`time_sq` (los 4 predictores que involucran valence/time) pero no de `onoff`/`selfother`.
Esto implica que **la muestra efectiva no es la misma entre análisis de distintas
dimensiones** — al comparar, por ejemplo, cuántos markers sobreviven para `onoff`
(19/23, §8) contra `valence` (0/23), parte de la diferencia podría, en principio,
reflejar la composición muestral distinta (n=35 vs n=33) y no sólo el tamaño del
efecto neural — algo a tener en cuenta si se reporta la comparación entre dimensiones
como si vinieran de la misma muestra.

### 7.2 Preprocesamiento

1. **Quadratic features ortogonalizadas** (`preprocessing.quadratic_features`): para
   `valence` y `time`, se computa `(x - 50)^2 / 50` y se **residualiza vía OLS**
   contra el término lineal `x`, produciendo un predictor cuadrático (forma de U)
   ortogonal al lineal. Esto evita que la colinealidad entre `x` y `x²` infle el
   error estándar del término lineal o produzca cancelación de varianza entre ambos.
2. **Transform de la variable de respuesta** (`preprocessing.response_transform`,
   actualmente `enabled: false` — ver §8 para qué implica dejarlo apagado):
   - Aplicado **antes** de la normalización por sujeto, porque el z-score es un mapa
     lineal y no puede remover asimetría ni exceso de curtosis.
   - **Regla de selección fijada de antemano** (no post-hoc): un marker se transforma
     si su exceso de curtosis residual medio (medido por
     `lmm_assumption_diagnostics.py` sobre el fit observado) supera
     `RESIDUAL_EXKURT_FLAG = 3.0`. Este criterio es ciego a la significancia del
     predictor — describe la forma de la distribución residual, no el tamaño de
     ningún efecto — así que aplicarlo por marker no constituye un "forking path".
   - Transform específico dictado por la escala de medición, no elegido por ensayo y
     error: `psd_relative_*` (proporciones en (0,1)) → `log` (empíricamente mejor
     comportado que `logit` acá); `slowwaves_PTP` (amplitud estrictamente positiva) →
     `log`; `slowwaves_Density` (conteo con ~44% de ceros) → `log1p`.
   - **Por qué sólo estos 3 de 23**: exceso de curtosis residual medio de 20.5, 12.0 y
     7.3 para estos tres, contra <5 para el resto y <3 para 17/23. Markers cuyo valor
     crudo se ve sesgado (PE_theta, wsmi_beta) tienen residuos bien comportados
     (exceso de curtosis 2.7 y 1.4) una vez que el modelo y el z-score por sujeto se
     aplican — transformarlos sería cosmético.
   - **Sensibilidad documentada**: correlación entre los t-maps observados
     sin-transformar y transformados: r=0.90 (psd_relative_gamma), r=0.99
     (slowwaves_PTP), r=0.96 (slowwaves_Density) — ninguna topografía se invierte.
   - **Caveat conocido, registrado deliberadamente**: transformar un subconjunto
     implica que los markers ya no están todos en la misma escala de respuesta. Esto
     es aceptable porque el estadístico de cluster es una suma de t-values (libre de
     escala por construcción) y porque la familia de comparaciones múltiples se
     define sobre markers, no sobre una escala compartida. **No** sería aceptable si
     se compararan tamaños de efecto en unidades crudas entre markers — no se hace.
3. **Normalización por sujeto** (`normalize_by_subject: true`, método `zscore`,
   no channel-wise): sin esto, diferencias de escala entre sujetos se propagan a la
   varianza residual del LMM y sesgan la pendiente del predictor hacia los sujetos de
   mayor amplitud, produciendo clusters espacialmente extendidos que reflejan
   topografías individuales de baseline en vez del efecto del predictor.

### 7.3 Especificación del modelo (LMM)

```
power ~ onoff + valence + valence_sq + selfother + time + time_sq + time_on_task + confidence + (1|subject)
```

- Modelo **aditivo** (sin interacciones) por defecto; ajustado **independientemente
  por canal** (64 canales del montage `CACS-64_REF.bvef`, menos los excluidos de
  clustering — ver 7.5).
- **Efecto aleatorio**: intercepto por sujeto, `(1|subject)` — captura el baseline
  individual sin asumir pendientes aleatorias (no hay suficiente potencia por sujeto
  para estimar 8 pendientes aleatorias correlacionadas de forma estable).
- **Backend**: `statsmodels.MixedLM` por defecto (`lmm.backend: "python"`,
  optimizador Powell, 200 iteraciones máx). Backend alternativo `lme4::lmer` vía
  `rpy2` (`backend: "r"`, optimizador bobyqa) — **requerido** cuando
  `obs_weight_col` no es `null`, porque sólo `lme4` implementa WLS-LMM exacto.
  Actualmente `obs_weight_col: null` (sin ponderación por confianza) en el config de
  producción.
- **Términos de interacción** (ej. `onoff:confidence`): se especifican vía
  `lmm.per_predictor_formula`, que sobrescribe la fórmula sólo para ese predictor de
  interés; `_resolve_formula_for_predictor()` valida en tiempo de carga que el
  override efectivamente contenga el término multiplicativo, para no testear
  silenciosamente el efecto principal cuando la intención era la interacción.
- **Determinismo del fit**: mismo `method`/`maxiter`/`random_state` para el fit
  observado y para **cada** fit permutado — usar un optimizador distinto entre
  observado y permutado produciría t-statistics sistemáticamente distintas y tasas de
  convergencia distintas, generando una null distribution asimétrica y sesgando la
  tasa de falsos positivos.

### 7.4 Esquema de inferencia por permutación

Dos esquemas disponibles vía `andrillon_clustering.permutation_method`:

| Método | Qué permuta | Cuándo es válido |
|---|---|---|
| `"simple"` (Andrillon 2020 original) | Las etiquetas del predictor de interés, dentro de cada sujeto (`permutation_within: ["subject"]`) | Sólo si el predictor de interés es **no correlacionado** con las demás covariables de la fórmula |
| `"freedman_lane"` (**usado en producción**) | Los residuos de un modelo reducido (fórmula sin el predictor de interés), reconstruyendo `Y* = Ŷ_reducido + R_permutado`, y re-ajustando el modelo **completo** sobre `Y*` con las etiquetas **originales** del predictor | Preserva la estructura de covarianza entre el predictor de interés y el resto de covariables bajo H0 |

**Por qué Freedman-Lane es la opción de producción**: con permutación simple, cuando
el predictor de interés está correlacionado con otras covariables del modelo (p.ej.
`time_on_task`, o `valence` con `valence_sq`), permutar sólo sus etiquetas destruye
esa colinealidad **únicamente en la null distribution** — la contrae artificialmente
y por lo tanto infla la tasa de falsos positivos del test observado. Freedman-Lane
evita esto ajustando el modelo reducido **una sola vez** (fuera del loop de
permutaciones) y reutilizando sus residuos/valores ajustados en cada draw.

**Detección de permutaciones degeneradas**: si más del 50% de los canales fallan a
converger en una permutación dada (`DEGENERATE_RATE_THRESHOLD = 0.5`), esa
permutación se marca `is_degenerate` pero **no aborta la corrida** — sus canales
fallidos propagan como NaN simétricamente a la null distribution. Si más del 1% de
las 5000 permutaciones resultan degeneradas (`DEGENERATE_RATE_WARN`), se emite un
warning recomendando cambiar a `freedman_lane` (si no se está usando ya). Además, al
final de cada corrida se compara explícitamente la tasa de convergencia observada
vs. permutada — una asimetría ahí (más canales NaN en el lado permutado) indicaría
que la null se está contrayendo y el FPR real es mayor al nominal.

**Semillas**: `base_seed = andrillon_clustering.seed` (42). El fit observado usa
`base_seed`. Cada permutación `k` usa `base_seed + 1 + k + n_permutations` — el
offset por `n_permutations` es deliberado, para que el espacio de semillas de las
permutaciones nunca colisione con el del fit observado (evitando correlación espuria
entre ambos vía cualquier consumidor de `np.random` dentro del stack del
optimizador). Ejecución paralela vía `joblib`/`loky`: cada proceso hijo tiene su
propio estado de RNG, así que no hay fuga de semilla global entre permutaciones.

### 7.5 Formación de clusters espaciales

Sigue la definición **exacta** de Andrillon et al. (2020), implementada en
`Statistics/cluster_test.py::find_clusters_from_pvalues`:

1. **Clusters candidatos**: electrodos vecinos (según matriz de adyacencia del
   montage) con `p < cluster_alpha` (0.025). La adyacencia se computa con
   `mne.channels.find_ch_adjacency()` sobre las posiciones reales del montage
   `CACS-64_REF.bvef` — no es un k-NN arbitrario.
2. **Signo separado**: clusters positivos y negativos se agrupan y evalúan por
   separado (`separate_signs=True`) — evita que un cluster con t-values de signo
   mixto vea su estadístico artificialmente reducido por cancelación de signos.
3. **Exclusión de canales de borde**: `FT9, TP9, FT10, TP10, Iz` (los más laterales,
   |θ|≥90°, más `Iz` en el borde posterior) se siguen testeando (tienen su t-value y
   p-value) pero **no pueden formar ni conectar** clusters — previene artefactos de
   borde donde canales con adyacencia parcial generan clusters espurios.
4. **Estadístico de cluster**: `stat_fun = "sum"` — suma de t-values de todos los
   canales del cluster (no la suma de F ni el máximo). Ésta es la definición textual
   del paper de Andrillon ("we computed the sum of the t-values for all the
   electrodes belonging to the cluster").
5. **Umbral de visualización** (`threshold_t`, sólo para topoplots): el valor crítico
   normal de dos colas correspondiente a `cluster_alpha` (`z ≈ 2.24` para 0.025) — **no**
   es un cuantil data-dependiente de la distribución observada de |t| (como en una
   versión anterior del pipeline), precisamente para que el umbral sea el mismo,
   comparable y reproducible entre markers.

### 7.6 P-valor de Monte Carlo

Para cada permutación se repite exactamente el mismo procedimiento de formación de
clusters sobre los t-values/p-values permutados, guardando el estadístico máximo (o
mínimo, para negativos) por permutación:

```
p_val(cluster positivo) = ( #{permutaciones con max(stat_perm_pos) ≥ stat_observado} + 1 ) / (n_perm + 1)
p_val(cluster negativo) = ( #{permutaciones con min(stat_perm_neg) ≤ stat_observado} + 1 ) / (n_perm + 1)
```

El `+1` en numerador y denominador es la **corrección de continuidad de Phipson &
Smyth (2010)**: sin ella, cuando ninguna permutación alcanza un estadístico tan
extremo como el observado, el p-valor reportaría exactamente 0 — lo cual subestima la
incertidumbre real (el p verdadero está acotado por `1/n_permutaciones`) y rompe
cualquier lógica downstream que trate `p=0` como caso especial.

`n_permutations = 5000` (config actual; el paper de Andrillon usa 1000 — se subió
para estabilidad del p-valor, ver advertencias de `cluster_test.py`:
"n_permutations < 5000 puede no ser suficiente para publicación").

### 7.7 Corrección por comparaciones múltiples (entre markers)

**Qué NO se corrige acá**: la multiplicidad entre electrodos. Ya está controlada por
el propio test de cluster-permutation, cuya null distribution se construye a partir
del estadístico máximo a través de todo el montage. Aplicar una segunda corrección
por electrodos penalizaría dos veces la misma multiplicidad.

**Qué SÍ se corrige acá**: la multiplicidad entre markers. La familia es exactamente
la que declaran `selected_markers`/`feature_families` en el config **vigente al
momento de correr la corrección** — no lo que haya en disco. Un marker declarado que
no produjo resultados (crash, no corrido) **aborta** la corrección
(`require_complete_family: true`) en vez de encoger silenciosamente el denominador
(un `m` más chico relaja la corrección — si el crash se ignorara, un marker faltante
volvería *más* significativos a los demás, exactamente al revés de lo deseable).

**Unidad de corrección — `unit: "marker"`**: cada marker aporta **un** p-valor
representativo (el de su cluster de estadístico máximo — el único calibrado con
precisión contra la null de máximo-estadístico; los clusters sub-máximos de un mismo
marker cargan p-valores conservadores por construcción). Un marker sin clusters
candidatos no aportó evidencia y se representa con p=1.0 (se mantiene en la familia:
fue testeado, y excluirlo encogería el tamaño de familia post-hoc). La significancia
a nivel de **cluster** se resuelve luego por **gatekeeping jerárquico**: un cluster es
significativo si (a) su marker sobrevivió la corrección de familia **y** (b) su
propio p-valor de permutación (sin corregir) es < α. Esto se expresa como
`p_cluster_corregido = max(p_marker_corregido, p_cluster_crudo)`.

**Método — Benjamini-Hochberg (`fdr_bh`), con caveat documentado**: BH es el estándar
de facto en la literatura EEG/neuroimagen y se usa acá por comparabilidad. Controla
FDR bajo independencia o dependencia de regresión positiva (PRDS). **El supuesto PRDS
sólo está claramente satisfecho por la familia `evoked`** — correlacionando los
t-maps de los markers (agrupados sobre todos los directorios de modelo):

| Familia | m | r media | % pares negativos | % r < −0.3 | PRDS |
|---|---|---|---|---|---|
| evoked | 4 | +0.55 | 0% | — | plausible |
| state | 16 | +0.01 | 52% | 19% | dudoso |
| sleep | 5* | +0.07 | 30% | 20% | dudoso |

*(la fila `state` fue posteriormente eliminada del análisis activo — ver 7.1 — pero
la correlación negativa documentada en el YAML sigue siendo la razón para dudar de
PRDS en familias con markers espectrales relativos, que son composicionales por
construcción: `psd_relative_beta` vs `psd_relative_delta` r=−0.72, y los ratios
comparten denominador con sus componentes, `psd_relative_beta` vs
`psd_relative_theta_beta_ratio` r=−0.81, esencialmente mecánico.)*

Bajo esa dependencia negativa, el FDR real de BH puede exceder el 0.05 nominal.
**Benjamini-Yekutieli (`fdr_by`) es válido bajo dependencia arbitraria** pero paga por
ello con una corrección más conservadora (penalización `C(m) = Σ 1/i`). Bonferroni se
descarta directamente: combinado con el piso de p-valor de la permutación
(`~1/(n_permutaciones+1) ≈ 0.0002`), deja la significancia inalcanzable en las
familias más grandes sin importar el tamaño del efecto.

**Estado actual**: `evoked`, `state`, `sleep` usan `fdr_bh` en el config de
producción. **El análisis de sensibilidad BH-vs-BY que el propio YAML recomienda
reportar junto a los resultados primarios (`"BECAUSE THE CHOICE IS DEBATABLE, report
the BY-vs-BH sensitivity analysis alongside the primary BH results"`) todavía no se
corrió** — no hay ningún directorio ni CSV con `fdr_by` en `results/andrillon_cluster/`
a la fecha de este documento. Ver §9.

### 7.8 Test omnibus a nivel de familia

Complementa (no reemplaza) la corrección marker-por-marker. Responde una pregunta
distinta: *¿hay señal en ALGUNA parte de la familia?*, en vez de *¿cuál marker
específico la tiene?*. Es relevante porque una dimensión cuya señal está distribuida
débilmente entre 6 markers moderados puede fallar cada test marker-por-marker
individualmente mientras es, colectivamente, muy lejos del azar — llamar "nula" a esa
dimensión basándose sólo en el test marker-por-marker sería una afirmación que los
datos no sostienen.

**Por qué un test de permutación y no un binomial**: contar markers con p≤α y
compararlo contra `Binomial(m, α)` asume independencia entre markers, que no se
cumple (ver tabla de correlación arriba). Acá la null se construye desde las
permutaciones mismas: la permutación `k` es el **mismo** shuffle de los datos
subyacentes para todos los markers de la familia, así que la estructura de
dependencia conjunta bajo H0 —incluida la que existe entre markers— se reproduce
exactamente, sin supuesto distribucional.

**Dos estadísticos reportados**:
- `count` — número de markers con p-valor propio ≤ α. Sensible a señal distribuida
  débilmente entre muchos markers.
- `min_p` (Tippett) — el p-valor mínimo de la familia. Su p-valor de permutación es
  una significancia FWER-corregida para el mejor marker individual, válida bajo
  dependencia arbitraria — alternativa libre de supuestos a BH.

Para cada permutación, su propio p-valor a nivel de marker se calcula **leave-one-out**
(excluyendo esa permutación de su propia null de referencia) — de lo contrario cada
permutación se compararía consigo misma incluida en el denominador, sesgándola
sistemáticamente hacia verse "menos extrema" que el dato observado (que sí se compara
contra el conjunto completo), descalibrando el p-valor omnibus resultante.

Requiere `null_cluster_stats` guardado en cada `results.pkl` — no reconstruible a
posteriori sin re-ajustar cada LMM permutado; markers corridos antes de que esto se
guardara (commit `8e4acce`) deben re-correrse para entrar al omnibus.

### 7.9 Verificación de supuestos del LMM

Se reporta **descriptivamente** (skew, exceso de curtosis, fracción de outliers), no
como test formal. Con ~1700–2000 observaciones por canal, Shapiro-Wilk rechaza ante
desvíos demasiado pequeños como para importar en un test basado en permutaciones —
devolvía "100% de los canales violan normalidad" para prácticamente cualquier
marker, lo cual no es accionable.

**Por qué esto no invalida el test**: los p-valores de cluster vienen de la
distribución de permutación, no de la distribución de referencia t — así que
residuos no-normales no invalidan la tasa de error tipo I. Los costos reales son
pérdida de potencia y t-statistics inestables, dominadas por un puñado de
observaciones extremas (lo que sí miden el exceso de curtosis y la fracción de
outliers).

**Heterocedasticidad es el hallazgo más consecuente**: Freedman-Lane permuta
residuos **dentro de sujeto** y por lo tanto asume que son intercambiables; el test
de Breusch-Pagan por canal (`breusch_pagan_p`) chequea justamente esto.

Umbrales de alerta (no de exclusión automática — son para inspección manual):
`RESIDUAL_SKEW_FLAG = 1.0`, `RESIDUAL_EXKURT_FLAG = 3.0`, `RESIDUAL_OUTLIER_SD = 4.0`
(definidos en `Statistics/lmm_model.py`, reutilizados sin duplicar acá).

### 7.10 Determinismo y reproducibilidad

- Semilla explícita en cada punto estocástico: `andrillon_clustering.seed=42` para el
  fit observado y como base del esquema de semillas por permutación (§7.4);
  `lmm.random_state=42` pasado también a `run_lmm_per_channel`.
- Sin `try/except` en el código científico (Regla crítica #6 del `CLAUDE.md` raíz) —
  errores de convergencia, archivos faltantes o familias incompletas abortan con un
  mensaje explícito en vez de degradarse silenciosamente.
- Invariantes explícitos (con `RuntimeError`, no warnings) verifican en cada corrida
  que el orden de canales entre t-stats/p-values, adyacencia y montage coincide
  exactamente antes de formar clusters — un desalineamiento ahí produciría clusters
  espacialmente sin sentido sin ningún error visible.

---

## 8. Resultados actuales — snapshot (2026-07-25)

⚠️ **Esto es un estado de `results/andrillon_cluster/` a la fecha de este documento,
no una afirmación confirmatoria del paper.** Reportado acá por transparencia y para
que quien retome el análisis sepa exactamente qué hay corrido. Etiquetar explícitamente
como *exploratorio* cualquier patrón que se cite antes de que la sensibilidad
BH-vs-BY (§7.7, pendiente) se haya corrido.

Familia de corrección: 4 markers `evoked` + 19 markers `sleep` = 23 hipótesis por
target, corregidas con BH separadamente dentro de cada familia, α=0.05. Tabla:
markers cuyo p-valor **corregido a nivel de marker** sobrevive α, según
`multiple_comparisons_summary.csv` de cada directorio de modelo. Columna `N` = tamaño
muestral efectivo de ese predictor tras el filtro de variabilidad (§7.1.1) — **no es
el mismo entre filas**, tenerlo presente al comparar targets entre sí.

| Target (predictor) | N (de 42) | Markers significativos (corregidos) / 23 | Cuáles |
|---|---|---|---|
| `onoff` | 35 | **19/23** | evoked: P3a, P3b · sleep: PE_alpha, PE_beta, PE_gamma, kolmogorov_complexity, psd_relative_{alpha,beta,delta,gamma,theta}, slowwaves_{Density,Duration,Frequency,PTP}, wsmi_{alpha,beta,gamma,theta} |
| `valence_sq` (cuadrático) | 31 | **7/23** (todos `sleep`) | PE_beta, PE_gamma, PE_theta, psd_relative_{alpha,beta,delta}, slowwaves_PTP |
| `valence` (lineal) | 33 | 0/23 | — |
| `selfother` | 37 | 0/23 | — |
| `time` (lineal) | 35 | 0/23 | — |
| `time_sq` (cuadrático) | 34 | 0/23 | — |

Test omnibus (`omnibus_test.py`) corrido hasta ahora sólo para `valence_sq` (ver
gotcha en §4.7 — el CSV combinado en disco no refleja los otros 5 targets):

| Familia | m | markers con p≤0.05 (sin corregir) | p omnibus (count) | mejor marker | p FWER (min-p) |
|---|---|---|---|---|---|
| evoked | 4 | 0 | 1.000 (no sig.) | P3a (p=0.073) | 0.356 |
| sleep | 19 | 10 | **0.0004** | PE_beta (p=0.0002) | **0.0002** |

Lectura (exploratoria): para `valence_sq`, la familia `sleep` muestra evidencia
colectiva clara de señal (10/19 markers con p crudo ≤0.05, muy por encima del ~1
esperado al azar bajo la null de permutación), aunque sólo 7 sobreviven la corrección
marker-por-marker — consistente con la motivación del test omnibus (§7.8): señal real
distribuida entre varios markers moderados puede quedar sub-representada por el test
marker-por-marker solo. La familia `evoked` no muestra evidencia de familia (p=1.0).

Esto es compatible con la narrativa del Paper 2 en `CLAUDE.md` ("emerges for valence
only via valence×confidence interaction / 48-electrode clusters" para la familia de
complejidad/arousal) en el sentido de que valence **lineal** no deja huella y sí lo
hace su componente **cuadrática** (forma de U) — pero la interacción valence×confidence
mencionada ahí es un modelo *distinto* (con `per_predictor_formula` de interacción,
no corrido en este snapshot) y no debe confundirse con este resultado de `valence_sq`.

---

## 9. Limitaciones conocidas / deuda técnica

Documentado explícitamente para que no se redescubra por sorpresa:

1. **El filtro de QA está en el config pero no en el código** (§7.1.1): `qa_summary_path`
   y `exclude_failed_qa: true` no tienen ningún efecto en `Stats_andrillon/andrillon_pipeline.py`
   — a diferencia de `Statistics/run_pipeline.py`, nunca llama a `load_qa_summary`/
   `get_qa_exclusion_list` ni pasa `qa_exclusions` a `load_all_probe_data`. Todas las
   sesiones marcadas como fallidas en `qa_summary.csv` entran igual al análisis. Si se
   quiere que el filtro de QA declarado en el config realmente aplique, hay que
   cablearlo en `load_marker_matrix`/`_load_ratio_marker_data` (mismo patrón que ya
   existe en `Statistics/run_pipeline.py`).
2. **Tests unitarios desactualizados**: `tests/test_permutation.py` y
   `tests/test_clustering.py` importan `lmm_permutation` y `cluster_detection` —
   módulos del plan de implementación original de 2024 que fueron reemplazados por
   la reutilización directa de `Statistics/lmm_model.py` y `Statistics/cluster_test.py`
   (§6), y ya no existen en el directorio. `python tests/run_tests.py` falla en el
   import antes de correr un solo test. La cobertura real de este pipeline hoy es
   indirecta: pasa por los tests de `Statistics/` (si existen) más las invariantes
   `RuntimeError` embebidas en el propio pipeline.
3. **`omnibus_test.csv` no acumula entre corridas** (§4.7) — cada invocación
   sobreescribe el archivo completo con sólo los `model_dir` que procesó. El estado
   en disco hoy sólo tiene las filas de `valence_sq`.
4. **No se persiste `used_config.yaml`** en los directorios de salida — a diferencia
   de lo que pide el checklist de automatización del `CLAUDE.md` raíz
   ("Results folder contains `used_config.yaml`"). Reconstruir qué config generó un
   resultado dado requiere `git log`/`git blame` sobre `config_andrillon.yaml`, no
   está auto-contenido en `results/`.
5. **Sensibilidad BY-vs-BH pendiente** (§7.7) — el propio config documenta que
   debería reportarse, pero no hay corrida con `fdr_by` en disco todavía.
6. **`response_transform.enabled: false`** en producción — el pipeline corre
   actualmente sobre variables sin transformar a pesar de que la regla de selección
   (§7.2) identifica 3 markers que la ameritarían. Si se activa, hay que re-correr
   los 6 targets y volver a generar §8.

---

## 10. Referencias

- Andrillon, T., et al. (2020) — metodología de cluster-permutation con LMM y
  Monte-Carlo p-values (líneas 102–115 citadas textualmente en el diseño original,
  ver historial de commits `andrillon lmem`, `fixes andrillon`).
- Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and
  MEG-data. *Journal of Neuroscience Methods*, 164(1), 177–190.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
  *JRSS-B*, 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in
  multiple testing under dependency. *Annals of Statistics*, 29(4), 1165–1188.
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero.
  *Statistical Applications in Genetics and Molecular Biology*, 9(1).
- Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
  significance levels. *Journal of Business & Economic Statistics*, 1(4), 292–298.
