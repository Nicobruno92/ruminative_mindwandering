# junifer_markers — Extracción de marcadores EEG por probe (Paper 2, Secciones 2–3)

Pipeline que convierte las épocas EEG ya preprocesadas (`BIDS/derivatives/`) en
marcadores cuantitativos por canal/ROI y por probe MDES, usando
[Junifer](https://juaml.github.io/junifer/) + la extensión propia `junifer_eeg`.
Es la fuente de datos de la que dependen tanto **Stats_andrillon** (CBPT,
Sección 2 del Paper 2) como **mw_classification_pipeline** (clasificación
individual, Sección 3) — ver `CLAUDE.md` en la raíz para la estructura completa
de los dos papers.

> **Qué responde este documento**: para cada marcador que termina en un CSV o
> en un modelo estadístico río abajo, ¿de qué señal salió, con qué parámetros
> exactos se computó, y qué transformaciones sufrió antes de llegar ahí?

Estructura del documento:
1. **Resumen y ejecución** — los 4 pasos del pipeline y cómo correrlos (§1–§3).
2. **Metodología en detalle** — qué se computa exactamente, a nivel de sección
   de Métodos de un paper (§4–§6).
3. **Salidas, límites conocidos y referencias** (§7–§9).

Todo lo que sigue está anclado a los archivos de config tal como existen en
este directorio a fecha **2026-07-27**. Los YAML son la fuente de verdad — si
algo diverge de lo que dice acá, el YAML manda.

---

## 1. Resumen del pipeline (4 pasos)

```
Paso 0                Paso 1                    Paso 2                  Paso 3
─────────────         ──────────────────        ──────────────────     ──────────────
Umbral PTP      ──►   Extracción de H5    ──►    Agregación por    ──► Topomaps
por sujeto            (Junifer, SLURM           probe (CSV largos      grupales
(sólo sleep/           array, por elemento       o anchos por           (QA)
ondas lentas)          subject×task×desc)        canal/ROI)
```

| Paso | Directorio | Qué hace | Depende de |
|---|---|---|---|
| 0 | `0.compute_sw_thresholds/` | Umbral de amplitud pico-a-pico (PTP) por sujeto/canal para detección de ondas lentas | — |
| 1 | `1.markers_h5_creation/` | Corre Junifer sobre cada `sub-XX_task-SartN_desc-{evoked,state,sleep}_epo.fif` y guarda un `.h5` con todos los marcadores de esa época | Paso 0 (sólo `sleep`, porque `ptp_thresholds_strict: true`) |
| 2 | `2.aggregate_probes/` | Lee los `.h5`, filtra épocas por criterios de ensayo, agrega en 2 niveles (época→canal, canal→ROI) y escribe un CSV por sujeto/tarea/probe | Paso 1 |
| 3 | `3.topomaps/` | Figuras de topomap a nivel de grupo (QA visual, no forma parte del análisis inferencial) | Paso 2 |

Orquestador end-to-end con dependencias SLURM: `run_full_pipeline.sh`
(`--skip-step0`, `--skip-topomaps`, `--dependency=JOB_ID`). Cada paso tiene su
propio `README.md`/`QUICK_START.md` con detalle operativo (SLURM, troubleshooting);
este documento se enfoca en **qué se computa y por qué**, no en cómo correrlo.

---

## 2. Datos de entrada — tres tipos de época

`junifer_markers` no epocha datos: consume `*_epo.fif` ya producidos por el
pipeline de preprocesamiento (`Preprocessing_pipeline_new/` + `ERPs_new/`), con
tres esquemas de epoching (`desc-{evoked,state,sleep}`) que responden a
preguntas distintas y por eso reciben marcadores distintos (§4):

| `desc` | Qué es | Bloqueo temporal | Baseline | Marcadores que recibe |
|---|---|---|---|---|
| `evoked` | Época por ensayo *go* individual, bloqueada al estímulo | Estímulo (dígito) | `[-0.2, 0]` s | Espectrales, conectividad, IT, **ERPs**, resúmenes PSD |
| `state` | Bins temporales de estado tónico pre-probe (`dt = -3 … -10` s antes del probe) | Ninguno (ventana continua) | Sin baseline | Espectrales, conectividad, IT, resúmenes PSD (sin ERPs — no hay estímulo que bloquear) |
| `sleep` | Una única época larga (~20 s) pre-probe por probe | Ninguno | Sin baseline | Espectrales, conectividad, IT, **ondas lentas y spindles** |

Los tres comparten el mismo montage de grabación (**64 canales, `CACS-64_REF.bvef`**,
`Preprocessing_pipeline_new/config.yaml`), la misma tasa de remuestreo
(`sr_target: 250` Hz) y la misma referencia de preprocesamiento (**promedio,
`avg_ref_projector: true`**). Algunos marcadores sobreescriben esa referencia
puntualmente (ver `reference: ["TP9","TP10"]` en §4.4 y §4.6) — es una decisión
explícita, no un descuido: ERPs y eventos de sueño se leen tradicionalmente en
referencia mastoidea/TP9-10, no en referencia promedio, porque la referencia
promedio puede distorsionar la topografía y la amplitud de eventos puntuales
que es justamente lo que estos marcadores miden.

Canales EOG (`VEOG`, `HEOG`) se excluyen explícitamente en la etapa de
agregación (Paso 2), no en la de extracción.

---

## 3. Ejecución (referencia rápida)

```bash
conda activate junifer   # ver junifer_eeg en site-packages (instalación editable)

# Pipeline completo, 4 pasos encadenados por SLURM
./run_full_pipeline.sh

# Saltar el paso 0 (si pooled_sw_thresholds.csv ya existe)
./run_full_pipeline.sh --skip-step0

# Un solo paso, manualmente:
cd 0.compute_sw_thresholds && sbatch slurm_compute_thresholds.sh
cd 1.markers_h5_creation   && CONFIG_TYPE=sleep sbatch --array=0-$(python discover_elements.py --desc sleep --count) slurm_array_junifer.sh
cd 2.aggregate_probes      && python aggregate_markers_by_probe.py --config config.yaml
cd 3.topomaps               && sbatch slurm_topomaps.sh
```

Detalle operativo completo (troubleshooting, variables de entorno, gestión de
`elements`) en el `README.md`/`QUICK_START.md` de cada subdirectorio.

---

## 4. Marcadores computados — sección de Métodos

Cada marcador corre **por canal y por época**; la agregación a nivel de probe
(episodio→canal→ROI) es un paso posterior y separado (§5). Todos los
parámetros abajo son literales de `1.markers_h5_creation/config_{evoked,state,sleep}.yaml`.

### 4.1 Potencia espectral (`SpectralPowerBands`)

Densidad espectral de potencia por método de Welch (`n_fft=1024`,
`n_per_seg=512` → ventanas de 2.048 s a 250 Hz, `n_overlap=256` → solape 50%),
integrada en bandas:

| Banda | Rango (Hz) | Presente en |
|---|---|---|
| delta | 1.0–4.0 | evoked, state, sleep |
| theta | 4.0–8.0 | evoked, state, sleep |
| alpha | 8.0–12.0 | evoked, state, sleep |
| beta | 12.0–25.0 | evoked, state, sleep |
| iota | 25.0–35.0 | **sólo state, sleep** |
| gamma | 35.0–45.0 (state/sleep) — **30.0–45.0 en evoked** | evoked, state, sleep |

> **Nota para el paper**: la definición de banda `gamma` difiere entre
> `evoked` (30–45 Hz, sin banda `iota` separada) y `state`/`sleep` (35–45 Hz +
> `iota` 25–35 Hz como banda propia). No es un error de transcripción — son
> archivos de config independientes que evolucionaron por separado — pero **no
> son comparables directamente entre epoch types** sin tenerlo en cuenta; si
> el paper compara potencia gamma entre `evoked` y `sleep`, hay que homogeneizar
> el rango primero o reportarlo como limitación.

Dos variantes por banda:
- **`psd_bands`** — potencia absoluta en dB (`dB: true`, `normalize: false`):
  `10·log10(potencia)`.
- **`psd_relative`** — potencia normalizada (`normalize: true`, `dB: false`):
  proporción de la potencia total en esa banda (suma de bandas = 1), sin
  logaritmo. Es la que efectivamente se usa en el LMM de `Stats_andrillon` para
  los marcadores `sleep` (ver ahí, §7.2, por qué se transforma con `log` antes
  del modelo para 3 de 23 marcadores).

### 4.2 Conectividad — Symbolic Mutual Information ponderada (wSMI)

`SymbolicMutualInformation`, implementación exacta del paquete NICE (King
et al., 2013, *Curr. Biol.*; Sitt et al., 2014, *Brain*), corrida una vez por
banda vía el parámetro `taus`:

| Marcador | `tau` (muestras @ 250 Hz) | Banda objetivo |
|---|---|---|
| `wsmi_theta` | 8 | θ |
| `wsmi_alpha` | 4 | α |
| `wsmi_beta` | 2 | β |
| `wsmi_gamma` | 1 | γ |

Parámetros comunes: `kernel=3` (patrones ordinales de longitud 3 → 6 símbolos
posibles), `weighted=true`, `csd=true`.

**Cómo se computa** (por par de canales, por época):
1. Cada señal se transforma en una secuencia de símbolos ordinales
   (permutación relativa de `kernel` muestras espaciadas `tau` entre sí —
   Bandt & Pompe, 2002).
2. Se estima la probabilidad conjunta de símbolos `p(x,y)` entre cada par de
   canales y la información mutua clásica
   `MI = Σ p(x,y) · [log p(x,y) − log p(x) − log p(y)]`.
3. **Ponderación (`weighted`)**: los términos correspondientes a símbolos
   idénticos o especularmente invertidos (misma forma, fase opuesta) se
   ponen a cero antes de sumar. Estos son los pares esperables bajo una
   fuente común/conducción de volumen con desfase 0° o 180° — down-weightearlos
   es lo que distingue "SMI" (sensible a señal compartida trivial) de "wSMI"
   (más específica a intercambio de información genuino entre generadores).
4. **CSD (`csd=true`)**: antes de simbolizar, se aplica una transformada de
   densidad de corriente superficial (Laplaciano de superficie,
   `mne.preprocessing.compute_current_source_density`) — reduce la
   sensibilidad a la referencia y a la conducción de volumen, paso estándar
   del paquete NICE.
5. Normalización final por `log(n_símbolos)` (máxima entropía posible con 6
   símbolos), acotando el valor a `[0, 1]`.

Los `tau` (8/4/2/1) y `kernel=3` son los valores canónicos de la literatura de
wSMI en estados de consciencia (King et al. 2013; Sitt et al. 2014) para datos
a 250 Hz, no un ajuste ad-hoc de este proyecto.

### 4.3 Teoría de la información

- **`PermutationEntropy`** (`PE_{theta,alpha,beta,gamma}`): entropía de Shannon
  normalizada de la distribución de patrones ordinales (Bandt & Pompe, 2002),
  mismos `kernel=3` / `taus` por banda que wSMI (§4.2) — es la versión
  univariada (dentro de un canal) del mismo formalismo simbólico usado para
  conectividad.
- **`KolmogorovComplexity`** (`kolmogorov_complexity`): aproximación por
  compresión — la señal se simboliza en `nbins=16` (default del marcador, no
  hay override en el config) y se comprime con `zlib`; la complejidad
  reportada es la razón de compresión (señales más regulares/predecibles
  comprimen más, señales más complejas/ricas en información comprimen menos).

Ambos corren `trial_method: null` (sin agregar dentro del canal — una fila por
época/canal, la agregación por probe ocurre en el Paso 2).

### 4.4 ERPs — topografías bloqueadas al estímulo (`TimeLockedTopography`, sólo `evoked`)

Amplitud promedio por ventana temporal, con corrección de línea de base
`[-0.2, 0]` s y **re-referencia a `TP9`+`TP10`** (mastoideo, override de la
referencia promedio de preprocesamiento — ver §2):

| Componente | Ventana (s) | Racional / literatura (comentarios de `config_evoked.yaml`) |
|---|---|---|
| P1 | 0.080–0.150 | Bozhilova, Broadway, Baird |
| N1 | 0.140–0.200 | Smallwood multimodal (misma topografía que P1) |
| P2 | 0.180–0.250 | P2 visual centroparietal |
| P3a | 0.250–0.450 | Fronto-central, relacionado con novedad — Henríquez, Arnett, Barroso (meta-análisis) |
| P3b | 0.300–0.600 | Centroparietal, actualización de memoria de trabajo — operacionalización preregistrada del "P3" de Smallwood (2008) / Kam (2011); Bozhilova, Smallwood, Cooper, Wang |

Nota: las ventanas en `config_general.yaml` (config legado, **no usado** por el
pipeline activo — ver `slurm_array_junifer.sh`, que sólo lee
`config_{evoked,state,sleep}.yaml` según `CONFIG_TYPE`) son ligeramente más
angostas para P1/P3a/P3b; se documenta acá para que no genere confusión si
alguien lo encuentra, pero **la especificación vigente es `config_evoked.yaml`**.

### 4.5 Resúmenes espectrales de banda ancha (`PowerSpectralDensitySummary`, `evoked`/`state`)

Adaptación del paquete NICE: percentiles de la densidad acumulada de potencia
entre 1–45 Hz (`fmin=1.0`, `fmax=45.0`), Welch con ventana corta
(`n_fft=4096`, `n_per_seg=128` ≈ 0.51 s @ 250 Hz, `n_overlap=100`), evaluado
sobre los primeros 600 ms de la época (`tmax=0.6`):

- **`msf`** (median spectral frequency, `percentile=0.5`): frecuencia por
  debajo de la cual cae el 50% de la potencia acumulada.
- **`sef90`** (`percentile=0.9`) / **`sef95`** (`percentile=0.95`): frecuencias
  de borde espectral al 90%/95% — sensibles a corrimientos hacia frecuencias
  altas de toda la distribución espectral, no sólo a una banda fija.

`channel_method: null` → sin agregación de canales en esta etapa (una columna
por canal, no por ROI).

### 4.6 Micro-eventos de sueño (`sleep` únicamente)

Sólo corren sobre las épocas `sleep` (ventana larga ~20 s pre-probe, sin
bloqueo a estímulo). Ambos detectores re-referencian a `TP9`+`TP10` antes de
detectar (mismo racional que §4.4).

#### Ondas lentas (`SlowWavesDetection`)

Adaptación de la metodología de **Andrillon et al. (2021)** / **Pinggal et al.
(2022)** para SART/mind-wandering (sin sesión placebo — ver Paso 0):

1. **Detección** (`detection_method: "custom"`, algoritmo de cruce por cero /
   Le Coz): filtro pasabanda `freq_sw: [1.0, 10.0]` Hz, umbral inicial de
   amplitud pico-a-pico `amp_ptp_initial: 15.0` µV para la detección bruta de
   candidatos.
2. **Filtros post-detección** (aplicados a cada onda candidata):
   - Frecuencia ≤ `freq_threshold: 7.0` Hz.
   - Pico positivo < `artifact_threshold: 75.0` µV (descarta artefactos).
   - **Sin** tope de PTP propio de la onda (`max_ptp_amplitude: 100000.0`
     desactiva efectivamente el cap de Junifer — Pinggal no define uno).
   - Descarta ondas dentro de `proximity_window: 1.0` s de cualquier muestra
     que exceda `proximity_amplitude: 150.0` µV (evita contaminación por
     artefactos de amplitud extrema cercanos).
3. **Umbral de amplitud "grande" (adaptativo, por sujeto y canal)**:
   percentil 90 (`ptp_percentile: 90.0`) de la distribución de PTP de las
   ondas ya filtradas, calculado **una vez por sujeto en el Paso 0** sobre el
   pool combinado de los 4 bloques SART (no por probe, no separado por
   condición — ver racional completo en `0.compute_sw_thresholds/README.md`),
   y aplicado acá como corte fijo vía
   `ptp_thresholds_path: .../pooled_sw_thresholds.csv`.
   `ptp_thresholds_strict: true` → si falta un sujeto en el CSV, el pipeline
   falla en vez de recalcular un percentil ad-hoc dentro de la época (evita
   sesgar el umbral hacia la condición que se está contrastando — ver Paso 0).
4. **Features reportadas** (una entrada Junifer por feature, mismo detector
   compartido vía cache singleton — todas las entradas *deben* compartir
   parámetros idénticos para que la caché y la semántica del umbral sean
   consistentes):

   | Marcador | Qué mide |
   |---|---|
   | `slowwaves_Duration` | Duración media de las ondas detectadas |
   | `slowwaves_PTP` | Amplitud pico-a-pico media |
   | `slowwaves_Frequency` | Frecuencia media de las ondas |
   | `slowwaves_Slope` | Pendiente media (semiperiodo positivo) |
   | `slowwaves_Density` | Nº de ondas por unidad de tiempo (conteo → 0 si no hay ondas, nunca NaN, ver §5.4) |

#### Spindles (`SpindlesDetection`, backend YASA — Vallat & Walker, 2021)

Banda sigma `freq_sp: [12, 15]` Hz sobre banda ancha `freq_broad: [1, 30]` Hz,
duración válida `[0.5, 2.5]` s, distancia mínima entre eventos `500` ms,
umbrales `thresh_rms: 1.5` (SD sobre la media) y `thresh_corr: 0.65`
(correlación banda sigma vs banda ancha). Features: `Duration`, `Amplitude`,
`Frequency`, `Density` (misma semántica de conteo→0 que `slowwaves_Density`).

### 4.7 Resumen — marcadores por tipo de época

| Tipo de época | Espectral | Conectividad | Teoría inf. | ERP | PSD summary | Sueño | Total entradas Junifer |
|---|---|---|---|---|---|---|---|
| `evoked` | 2 (10 cols) | 4 | 5 | 5 | 3 | — | 19 |
| `state` | 2 (12 cols) | 4 | 5 | — | 3 | — | 14 |
| `sleep` | 2 (12 cols) | 4 | 5 | — | — | 9 (5 SW + 4 spindles) | 20 |

("cols" = columnas expandidas tras separar por banda; el resto son
1 columna por entrada.)

---

## 5. Agregación por probe (Paso 2) — sección de Métodos

`2.aggregate_probes/aggregate_markers_by_probe.py` colapsa "muchas épocas por
probe" en "una fila por probe", en dos niveles configurables
(`2.aggregate_probes/config.yaml`).

### 5.1 Selección de ensayos (sólo `evoked`)

- `filter_go: true` — sólo ensayos *go* (no *no-go*/inhibición).
- `filter_correct: false` — **no** se filtra por corrección de respuesta en la
  configuración de producción actual (a pesar de que el comentario del YAML
  describe `filter_correct` como si filtrara; el valor efectivo es `false`).
  Si el paper reporta "sólo ensayos correctos", hay que verificar este flag en
  el momento de generar los números, porque no es lo que corre hoy.
- Ventana: los `evoked_distance_max=5` ensayos *go* más cercanos al probe
  (`distance_to_probe` 0–4, 0=el más cercano) — últimos ~5 ensayos antes del
  probe, alineado con la ventana de filtrado `-5` a `-1` que usa el resto del
  proyecto (regla #5 de `CLAUDE.md`).
- `min_required_distances: 3` — un probe se descarta si tiene menos de 3
  ensayos válidos en esa ventana (ver `qa_report`/summary para el conteo de
  probes descartados).

`state` y `sleep` no tienen selección de ensayo: se usan todos los bins
temporales / la única época, respectivamente.

### 5.2 Nivel 1 — época → canal

Para cada canal, se agregan los valores a través de las épocas seleccionadas
con `trial_methods: [mean]` (media aritmética simple; `trimmean` está
disponible pero comentado/desactivado en la config de producción).

### 5.3 Nivel 2 — canal → ROI (opcional, activo en producción: `channel_methods: [trimmean, std]`)

- **ROIs amplios** (9): frontal/central/posterior × izquierda/media/derecha,
  excluyendo electrodos temporales (`T7/T8/TP*/FT*`) y fronto-laterales
  extremos (`Fp1/2, F7/8, AF7/8`) y parieto-laterales (`P7/8`). Son el
  `default_rois` para espectrales/conectividad/IT.
- **ROIs estrictos para ERPs** (`marker_rois`, uno por componente,
  4 canales cada uno — más angostos que los ROIs amplios para maximizar SNR
  en el sitio canónico del componente en vez de diluir hacia topografías de
  componentes vecinos):

  | Componente | ROI | Canales |
  |---|---|---|
  | P1, N1 | lateral parieto-occipital | `O1,P7` / `O2,P8` (surrogate: el montage no tiene `PO7/PO8`) |
  | P2 | centroparietal medial | `Cz, CP1, CP2, Pz` |
  | P3a | frontocentral medial | `Fz, FC1, FC2, Cz` (surrogate: no hay `FCz`) |
  | P3b | centroparietal medial | `Cz, CP1, CP2, Pz` (surrogate: no hay `CPz`) |

  Estos surrogates son necesarios porque el montage de grabación **no**
  incluye `FCz`, `CPz`, `PO7`, `PO8` como canales activos independientes
  (ver la lista de canales en `2.aggregate_probes/config.yaml`) — reemplazan
  el sitio canónico por el vecino más cercano disponible.
- Agregación dentro de cada ROI: `trimmean` (10% de recorte por cola —
  `trimmean_percent: 10`) y `std`.

### 5.4 Manejo de NaN — sólo marcadores de sueño

`apply_nan_policy` aplica dos capas:

1. **Guard-rail siempre activo**: cualquier columna con `density` en el
   nombre rellena NaN con `0.0` — "0 ondas detectadas" es una observación
   válida, no un dato faltante.
2. **Política configurable** (`sleep_markers.nan_policy: "null"` en
   producción) para el resto de features de sueño (`PTP`, `Frequency`,
   `Duration`, `Slope`): se **mantiene el NaN** en vez de imputar. Rellenar
   con 0 sesgaría los LMM/CBPT río abajo tratando "sin ondas por encima del
   umbral" como "onda de amplitud/duración cero", inflando artificialmente la
   diferencia entre condiciones. Las alternativas (`zero`/`mean`/`median`)
   existen sólo para análisis de sensibilidad/ablación, no para producción.

### 5.5 Salida

Un CSV por `(sujeto, tarea, probe, epoch_type)`:
```
{features_root}/sub-XX/eeg/junifer_aggregated/
  sub-XX_task-SartN_probe-NNN_{onTask|offTask}_{evoked|state|sleep}.csv       # por canal
  sub-XX_task-SartN_probe-NNN_{onTask|offTask}_{evoked|state|sleep}_agg.csv   # por ROI (si channel_methods está activo)
```
más un resumen agregado (`junifer_probe_aggregation_summary.csv`) y un reporte
HTML de QA (`junifer_aggregation_qa_report.html`) con probes descartados,
mismatches de conteo de épocas y markers saltados.

**Etiqueta on/off-task del probe**: la etiqueta mayoritaria (moda) entre las
épocas de ese probe (formato harmonizado `ONTASK`/`OFFTASK`); si no hay
etiqueta harmonizada, cae a la mediana del rating legado 0–100 con
`onoff_threshold: 50` (`> 50` → `onTask`).

---

## 6. Paso 0 — umbral de ondas lentas por sujeto (racional completo)

Ver `0.compute_sw_thresholds/README.md` para el detalle completo; resumen:

Andrillon (2021)/Pinggal (2022) calibran el umbral de "onda grande" con una
sesión placebo, ajena a los brazos que se contrastan. Este proyecto no tiene
placebo — el contraste de interés (on-task vs. mind-wandering) está *dentro*
de la misma sesión. Calibrar el umbral con sólo probes on-task (o sólo
mind-wandering) sesgaría la detección hacia las estadísticas de amplitud de
esa condición. La solución adoptada: agrupar los 4 bloques SART completos
(todas las condiciones mezcladas) para estimar el percentil 90 por canal —
"ciego a condición por construcción" (Bernardi et al., 2015; Hung et al.,
2013; Vyazovskiy et al., 2011), maximizando además el N de ondas disponibles
para una estimación estable del percentil.

---

## 7. Qué consume esto río abajo

- **`Stats_andrillon/`** (CBPT + test omnibus, Sección 2 del Paper 2): opera
  directamente sobre los CSV agregados por ROI del Paso 2 (familias `evoked`
  m=4, `sleep` m=19 — ver `Stats_andrillon/README.md` §7.1).
- **`mw_classification_pipeline/`**: usa los mismos marcadores agregados como
  features de entrada al clasificador within-subject/LOSO (Sección 3).
- **`Behavior/Objective_Markers/`**: no consume esto — usa marcadores de
  comportamiento (RT, omisiones), no EEG.

---

## 8. Limitaciones conocidas / decisiones a documentar en el paper

- Definición de banda `gamma` (y ausencia de `iota`) distinta entre `evoked` y
  `state`/`sleep` (§4.1) — homogeneizar antes de comparar entre epoch types.
- `filter_correct: false` en la config de producción del Paso 2 (§5.1) — sólo
  se filtra por *go*, no por corrección de respuesta, pese a lo que sugiere el
  comentario del YAML.
- `config_general.yaml` (Paso 1) es legado y **no** lo usa el pipeline activo
  — no citarlo como fuente de parámetros.
- Surrogates de ROI para `FCz`/`CPz`/`PO7`/`PO8` (§5.3): el montage de
  grabación no tiene esos sitios; los ROIs "canónicos" de la literatura se
  aproximan con el vecino disponible más cercano — reportar como tal, no como
  el sitio exacto de la literatura citada.
- `slowwaves`/`spindles` sólo existen para `desc-sleep` (ventana larga
  pre-probe); no hay equivalente evocado ni de estado tónico corto para estos
  eventos, por construcción (requieren ventanas de varios segundos).

---

## 9. Referencias

- Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity
  measure for time series. *Physical Review Letters*, 88(17), 174102.
- King, J.-R., Sitt, J. D., Faugeras, F., Rohaut, B., El Karoui, I., Cohen,
  L., Naccache, L. (2013). Information sharing in the brain indexes
  consciousness in noncommunicative patients. *Current Biology*, 23(19),
  1914–1919.
- Sitt, J. D., et al. (2014). Large scale screening of neural signatures of
  consciousness in patients in a vegetative or minimally conscious state.
  *Brain*, 137(8), 2258–2270.
- Andrillon, T., Burns, A., Mackay, T., Windt, J., & Tsuchiya, N. (2021).
  Predicting lapses of attention with sleep-like slow waves. *Nature
  Communications*, 12, 3657. https://doi.org/10.1038/s41467-021-23890-7
- Pinggal, E., Dockree, P. M., O'Connell, R. G., & Andrillon, T. (2022).
  Pharmacological manipulations of physiological arousal and sleep-like slow
  waves modulate sustained attention. *Journal of Neuroscience*, 42(43),
  8113–8124.
- Bernardi, G., et al. (2015); Hung, C.-S., et al. (2013); Vyazovskiy, V. V.,
  et al. (2011) — justificación de la ventana de pooling condition-blind para
  el umbral PTP (ver `0.compute_sw_thresholds/README.md`).
- Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool
  for automated sleep staging. *eLife*, 10, e70092. (YASA — detección de
  spindles.)
- Smallwood, J., et al. (2008); Kam, J. W. Y., et al. (2011) — operacionalización
  ERP de mind-wandering (P3b) usada como referencia para la ventana `P3b`.
