# 🧠 Deterministic EEG Preprocessing Pipeline

Este pipeline automatizado está diseñado para pre-procesar datos de electroencefalografía (EEG) en formato BrainVision dentro de una estructura estandarizada BIDS. El sistema hace especial énfasis en la **reproducibilidad científica garantizada** (flujo de procesos determinista sin fallbacks aleatorios), control estricto de semillas (random seeds) y selección automática y rigurosa de hiper-parámetros (como umbrales de los artefactos en ICA y AutoReject).

## 📊 Arquitectura General y Pasos Realizados

El archivo central es `preprocessing_pipeline.py`, el cual ejecuta secuencialmente los siguientes pasos sobre una combinación de Sujeto y Tarea:

1. **Configuración Inicial e Inicialización BIDS**: 
   Carga del archivo `config.yaml` e inicialización del manejador de rutas I/O `BIDSCompliance`. Además, instancia un objeto *MNE Report* y un log en formato JSON para registrar métricas de Quality Assessment (QA).
2. **Carga de Datos (Raw)**: 
   Se cargan los archivos `.vhdr` crudos. Los canales "VEOG" y "HEOG" (si existen) se etiquetan forzosamente como tipo `eog` y extrae los eventos/anotaciones pre-armonizadas de la etapa anterior.
3. **Filtro de Segmentos de Descanso (Opcional pero Recomendado)**:
   A través de `steps_bad_rest.py`, se remueven de forma temprana los segmentos anotados como `BAD_rest` para evitar que contaminen de ruido fisiológico ajustes posteriores (como PyPREP e ICA).
4. **Filtraje inicial y Notch**: 
   Aplica un filtro FIR multipunto (frecuencias `notch_freqs` por defecto: 50, 100 y 150 Hz) para remover el ruido térmico y de la línea eléctrica. 
5. **Reconstrucción del Subespacio de Artefactos (ASR - Opcional)**:
   Si se activa en el config, `steps_asr.py` aplica ASR para limpiar artefactos ocasionales de alta amplitud antes de la detección espacial de canales.
6. **Detección Ciega de Canales Ruidosos (PyPREP)**:
   A través del script `steps_bads.py`, utiliza RANSAC para localizar canales atípicos o planos. Se puede aplicar un *"pre_ica"* rápido antes de esto para evitar que canales frontales sean etiquetados erróneamente como malos simplemente por el efecto del parpadeo natural de los ojos. 
7. **Interpolación y Montaje**:
   Los canales detectados como malos se interpolan esféricamente desde los vecinos sanos. Posteriormente se establece la configuración topológica provista por `CACS-64_REF.bvef` y se utiliza una re-referenciación del tipo *average reference* sólo para electrodos de EEG reales.
8. **División de Copias y Filtrado Frecuencial**: 
   El pipeline no subsamplea temporalmente (no hace downsample), manteniendo la frecuencia de resampleo intacta al crudo para evitar la pérdida de fidelidad en la señal. En su lugar, crea 2 copias virtuales:
   - *Copia ICA*: Se filtra (usualmente 1-30 Hz) exclusivamente para asegurar que la descomposición matemática sea óptima y no persiga derivaciones de baja frecuencia.
   - *Copia Analysis*: Se filtra (usualmente 0.1-35 Hz) para mantener todo el espectro biológico necesario para análisis biológicos (ej: deltas y evocados cognitivos tempranos).
9. **Descomposición ICA y Eliminación de Componentes Ciega**:
   Utilizando el algoritmo seleccionado (Picard, Infomax o FastICA) en modo estricto, el pipeline ajusta los componentes topológicos. Luego, usando `auto_select_ica_components` (sistema heurístico empírico ponderado por varianza), el sistema bloquea los artefactos musculares, cardíacos u oculares. Finalmente, esta matriz de desmezcla limpia se aplica sobre la *Copia Analysis*. 
10. **Densidad de Fuente de Corriente (CSD - Opcional)**:
    Una vez limpio el EEG de parpadeos espurios, y si el yaml lo indica, calcula el CSD eliminando el esparcimiento espacial del cráneo mediante derivadas gaussianas superficiales.
11. **Guardado de Raw Limpio (ICA-Clean)**:  
    El objeto continuo filtrado y sin componentes ICA `raw_clean` se almacena nativamente estructurado en el subdirectorio de pre-procesados de la lógica BIDS local (ej: `desc-icaClean_raw.fif`).
12. **Extracción y Segmentación de Épocas Evocadas**:
    Recorre el continuo y extrae fragmentos basándose en estímulos de tarea (por defecto todo lo etiquetado como marcador `Go` o `NoGo`). Extrae ventanas usando offsets dictados por el config (e.g. `tmin=-0.2` a `tmax=1.0`). Se aplica un rechazo pico-a-pico conservador (ej: >500uv) para descartar en primera instancia movimientos o desconexiones groseras de los electrodos.
13. **Rechazo Estadístico Local con AutoReject (AR)**:
    Un microscopio final para los artefactos. Usando el algoritmo cruzado AutoReject, encuentra el umbral bayesiano óptimo individual para cada paciente y descarta/interpola épocas y canales residuales que si bien sobrevivieron al ICA poseen picos anómalos o transitorios esporádicos.
14. **Reporte HTML interactivo (QA Metrics) y Guardado Final**:
    Finalmente, los datos en formato de época lista para estadística se salvan en el árbol BIDS como `_desc-evoked_epo.fif`. Luego, consolida figuras como comparativas de pre/post Power Spectral Density (PSD), gráficas de log de descartes, la topología ICA rechazada y la visualización del Average Evoked general (ERP) en un archivo HTML interactivo guardado en la carpeta de reportes particular del sujeto.

***

## ⚙️ Configuración del Pipeline (`config.yaml`)

El pipeline es agnóstico al código duro en sus rutinas paramétricas y es manejado enteramente por su configuración; siendo `config.yaml` su única fuente de la verdad para el ciclo de vida. Algunos de los módulos configurables principales abarcan:

*   **`project`**: Rutas absolutas a los datos, como `data_root` (input de origen crudo tipo BrainVision) y `derivatives_root` (para la salida organizada del pre-procesamiento). También define la lista restrictiva de sujetos (del `02` al `43`) y las tareas a transitar (`Sart1` a `Sart4`).
*   **`pyprep` / `pyprep_pre_ica`**: Controla la agressividad del detector automático de canales malos (RANSAC) y autoriza correr un ICA descartable muy rápido sólo y únicamente para evitar falsos positivos de desconexión en electrodos frontales debido al blink. 
*   **`filters`**: Maneja las dos líneas paralelas de filtrado (la de optimización de topología ICA y la señal real de análisis a retener).
*   **`ica` y `ica_selection`**: Método matemático (`picard` altamente recomendado) y constantes complejas como umbrales directos ICLabel, compensaciones por patrón (EOG real vs componente ICA inferido) y el ratio máximo permitido de varianza a extraer. Por regla predeterminada está configurado en umbrales muy garantistas (`max_excluded_variance_ratio: 0.90`) para evitar borrar señal neural genuina.
*   **`epochs_evoked`**: Rango temporal para cortar los segmentos vinculados al ensayo (e.g., entre `-0.2` a `1` s) y el parámetro paramétrizado del blanqueo de base de línea o reseteo de cero (`baseline`).
*   **`autoreject`**: Parámetros estadísticos demandantes computacionalmente para limpiar el trail final (N_Jobs, cantidad de iteraciones, n_consensus bayesiano) encargados de rescatar o finalmente matar epochs con picos locales estallados.
*   **`pass_criteria`**: Controla de manera auditable un conjunto estricto de banderas booleanas (flags) lógicas. Si un ensayo presenta defectos físicos aberrantes superando este límite (ej. porcentaje de épocas rechazadas superior al umbral del 35%), causa que pro-activamente el paciente de ese bloque se marque como `Failed` para evitar meter ruido invisible a la muestra final en procesos probabilísticos subsiguientes.

***

## 💻 Instrucciones de Uso y Workflow

Para garantizar el entorno, recuerda utilizar el manejador apropiado de Python instalado previamente (en el clúster usando `proxy` + virtual env, localmente cargando simplemente conda). Asumimos en este documento un entorno genéricamente nombrado `eeg`.

### Forma 1: Ejecución Manual para Un Único Sujeto (Modo Development/Testing Local)

Puedes ejecutar el pipeline a discreción sobre un sujeto puntual y de una sola tarea en tu terminal local para observar su comportamiento detallado por el STDOUT:

```bash
# Cambios relativos al origen de tu proyecto en tu máquina
cd <PATH_TO_YOUR_REPO_ROOT>/

# Requerido: Cargar entorno científico con dependencias resueltas (MNE, pyprep, autoreject, etc)
conda activate eeg 

# Lanzamiento apuntando específicamente la metadata de configuración 
python Preprocessing_pipeline_new/preprocessing_pipeline.py \
  --config Preprocessing_pipeline_new/config.yaml \
  --subject 04 \
  --task Sart1
```
*(Ideal para validar en local refactorizaciones del código fuente, agregar componentes al pipeline, o probar arreglos en el `config.yaml` viendo el plot interactivo MNE antes de escalar).*

### Forma 2: Arranque Completo y Concurrente en el Substrato SLURM (Clúster HPS) 🚀

Para procesar todos los sujetos y todos los bloques de una sentada aprovechando los recursos de súper-computación y colas, **ya cuentas con un script orquestador en BASH (`run_preprocessing_slurm.sh`)** que usa nativamente el motor de un *SLURM Job Array*.

El array despliega *168 tareas* paralelas simultáneamente (es decir, 42 sujetos x 4 bloques Sart cada uno). El script auto-calcula mediante división matemática simple en bash para trazar de manera autónoma iterativa a quien le corresponde preprocesar a qué nodo ciego computacional:
`subject_idx=$((SLURM_ARRAY_TASK_ID / 4))`  //  `task_idx=$((SLURM_ARRAY_TASK_ID % 4))`

Simplemente ejecuta el programador (scheduler) desde el login node:

```bash
# Desde el nodo cabecera del servidor de procesos principal:
cd /network/iss/.../tu_carpeta_del_proyecto
sbatch Preprocessing_pipeline_new/run_preprocessing_slurm.sh
```

**Beneficios Vitales del macro-script de SLURM adjunto:**
1. **Control Férreo de Hilos C o C++ Interno:** Apaga el afinamiento de multihilos automágicos invasivos pre-instalados de hardware en BLAS o compiladores OpenMP (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc) previniendo activamente "deadlocks" silenciosos de librerías cruzadas que ocurren con ONNXRuntime y Numpy al tratar de explotar un núcleo simultáneamente bajo una sub-rutina de MNE-Python u optimización de tensores.
2. Inhabilita predictivamente el multi-hilo o el uso directo de ICLabel defectuoso localmente (`DISABLE_ICLABEL=1`) ya que en clústers de memoria compartida sin GPUs exclusivas este módulo causa inconsistencias fatales que pausan o traban el nodo; forzando al mismo en su reemplazo a priorizar el patrón comparativo de topología seguro y efectivo ("template matching") que se encuentra programado de manera natural nativa en los parsers de MNE.
3. Almacenará los registros del terminal de estado independientes paralelos y errores del output nativo de cada trial en la sub-carpeta `logs/` con formato auditable y autoexplicativo: `preproc_auto_<JOB-ID>_<ARRAY-ID>.out`.
