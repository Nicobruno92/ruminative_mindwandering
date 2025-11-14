# Andrillon 2020 Cluster-Permutation Pipeline

## Objetivo

Implementar la metodología estadística de **Andrillon et al. (2020)** para nuestros análisis de EEG, manteniendo la misma estructura de datos y modelos del pipeline actual pero cambiando el enfoque estadístico.

---

## Metodología de Andrillon 2020

### Descripción del Paper

Según el paper (líneas 102-115):

> **Statistics.** A cluster-permutation approach (derived from Maris & Oostenveld 2007) was applied to identify significant clusters in topographical maps. Candidate clusters were defined as neighbouring electrodes with a p-value below a threshold (called cluster alpha) of 0.025. For each candidate cluster, we computed the **sum of the t-values** for all the electrodes belonging to the cluster (which we will refer to as the cluster statistics). We then created permuted datasets by **permuting the labels of the predictor within each subject, each task and each electrode** (N=1,000 permutations). For each of these permuted datasets, we also identified the candidate cluster with maximal absolute cluster statistics. The cluster statistics from permutations formed a null distribution, against which we compared the cluster statistics from the real dataset. Clusters (real and permuted) with positive and negative cluster statistics were compared separately. A Monte-Carlo p-value was derived from this comparison (p_cluster<0.05 means that a negative cluster has a cluster statistics below the 5th percentile of the permuted distribution and that a positive cluster has a cluster statistics above the 95th percentile of the permuted distribution). In cases where several cluster-permutations were performed in the same analysis (Fig. 5 and 6), we corrected the Monte-Carlo p-values of the real clusters with the Bonferroni method.

### Diferencias Clave con Nuestro Pipeline Actual

| Aspecto | Pipeline Actual (MNE-based) | Andrillon 2020 |
|---------|----------------------------|----------------|
| **Estadístico de cluster** | Suma de estadísticos F (de ANOVA) | **Suma de t-values** (de LMM) |
| **Permutación** | Freedman-Lane (permuta residuales) | **Permuta labels del predictor** dentro de subject/task/electrode |
| **Threshold** | t-threshold fijo (e.g., 3.0) | **p-value threshold** (cluster_alpha = 0.025) |
| **Cluster statistic** | Suma de F-values | **Suma de t-values** |
| **Corrección múltiple** | FDR sobre markers | **Bonferroni** sobre comparaciones |

### Ventajas del Enfoque de Andrillon

1. **Más simple y directo**: Permuta directamente las etiquetas del predictor
2. **Específico para LMM**: Usa t-values del predictor de interés
3. **Transparente**: Fácil de entender y replicar
4. **Conservador**: Bonferroni para múltiples comparaciones

---

## Estructura del Código de Andrillon

### Archivos Clave Analizados

1. **`lme_perm.m`** (43 líneas)
   - Función que ajusta LMM y genera permutaciones
   - **Permutación**: Dentro de cada subject × task, permuta el predictor
   - Output: t-value, t-stat, p-value para datos reales y permutados

2. **`get_clusterperm_lme.m`** (141 líneas)
   - Identifica clusters candidatos (p < cluster_alpha)
   - Calcula cluster statistics (suma de t-values)
   - Compara con distribución nula de permutaciones
   - Separa clusters positivos y negativos
   - Retorna clusters significativos (p_MC < 0.05)

3. **`wanderIM_LocalSleep_plotWavesPPties2_clusterPerm_rev1.m`** (430 líneas)
   - Pipeline completo de análisis
   - Loop sobre electrodos
   - Ajusta LMM por electrodo
   - Genera 1000 permutaciones
   - Aplica cluster-permutation
   - Visualiza resultados

---

## Plan de Implementación

### Fase 1: Estructura Base (Scripts Core)

**Objetivo**: Crear los módulos fundamentales en Python

#### 1.1 `lmm_permutation.py`
- **Función**: `fit_lmm_with_permutations()`
- **Input**: 
  - DataFrame con datos de un electrodo
  - Fórmula LMM
  - Predictor de interés
  - N permutaciones
- **Output**:
  - Real: [electrode_id, beta, t_value, p_value]
  - Permuted: [electrode_id, beta, t_value, p_value, perm_id] × N_perm
- **Lógica de permutación**:
  ```python
  # Para cada permutación:
  for subject in unique_subjects:
      for task in unique_tasks:
          # Permuta predictor dentro de subject × task
          idx = (data.subject == subject) & (data.task == task)
          data.loc[idx, predictor] = shuffle(data.loc[idx, predictor])
  ```

#### 1.2 `cluster_detection.py`
- **Función**: `find_clusters_andrillon()`
- **Input**:
  - T-values por electrodo (real y permutados)
  - P-values por electrodo (real y permutados)
  - Adjacency matrix (vecindad de electrodos)
  - cluster_alpha (threshold p-value, default=0.025)
- **Output**:
  - Lista de clusters con:
    - Tipo (positivo/negativo)
    - Electrodos miembros
    - Cluster statistic (suma t-values)
    - p-value Monte Carlo
- **Algoritmo**:
  1. Identificar electrodos candidatos: p < cluster_alpha
  2. Separar por signo (positivos vs negativos)
  3. Agrupar electrodos vecinos en clusters
  4. Calcular cluster statistic = sum(t_values)
  5. Repetir para cada permutación
  6. Comparar real vs distribución nula

#### 1.3 `andrillon_pipeline.py`
- **Función**: `run_andrillon_analysis()`
- **Workflow completo**:
  1. Cargar datos (reutilizar `reader.py` del pipeline actual)
  2. Loop sobre markers
  3. Loop sobre electrodos → ajustar LMM + permutaciones
  4. Detectar clusters
  5. Aplicar corrección Bonferroni (si múltiples comparaciones)
  6. Guardar resultados
  7. Generar visualizaciones

### Fase 2: Configuración

#### 2.1 `config_andrillon.yaml`
- Estructura similar a `config.yaml` actual
- Parámetros específicos de Andrillon:
  ```yaml
  andrillon_clustering:
    cluster_alpha: 0.025        # p-value threshold para clusters candidatos
    n_permutations: 1000        # N permutaciones (paper usa 1000)
    montecarlo_alpha: 0.05      # Threshold para p-value Monte Carlo
    stat_fun: "sum"             # Cluster statistic: "sum" de t-values
    bonferroni_correction: true # Aplicar Bonferroni si múltiples comparaciones
    permutation_within: ["subject", "task"]  # Permutar dentro de estos niveles
  ```

### Fase 3: Integración con Pipeline Actual

#### 3.1 Reutilizar Componentes Existentes
- **`reader.py`**: Cargar datos de markers (sin cambios)
- **`helpers.py`**: Funciones auxiliares (sin cambios)
- **Adjacency matrix**: Usar la misma del pipeline actual
- **Montage**: Mismo archivo BIDS

#### 3.2 Nuevos Componentes
- **Permutación**: Implementar estrategia de Andrillon (diferente a Freedman-Lane)
- **Cluster detection**: Algoritmo específico de Andrillon
- **Visualización**: Adaptar `plot_results.py` para mostrar clusters de Andrillon

### Fase 4: Scripts de Ejecución

#### 4.1 `run_andrillon_pipeline.py`
- Script principal para ejecución local
- Similar a `run_pipeline.py` actual

#### 4.2 `submit_andrillon_array.sh`
- SLURM array job para cluster
- Paralelizar por marker (como pipeline actual)

#### 4.3 `compare_pipelines.py`
- Script para comparar resultados:
  - Pipeline actual (MNE-based) vs Andrillon
  - Visualizar diferencias en clusters detectados
  - Análisis de concordancia

### Fase 5: Validación y Testing

#### 5.1 Tests Unitarios
- Test permutación: verificar que permuta correctamente
- Test clustering: verificar detección de clusters
- Test estadísticas: verificar cálculo de cluster statistics

#### 5.2 Validación con Datos Simulados
- Generar datos con efecto conocido
- Verificar que detecta clusters esperados
- Comparar con implementación de Andrillon (Matlab)

---

## Comparación Detallada: Algoritmos

### Pipeline Actual (MNE)
```
1. Ajustar LMM por electrodo → obtener F-statistics
2. Freedman-Lane permutation:
   - Ajustar modelo reducido (sin predictor de interés)
   - Permutar residuales
   - Re-ajustar modelo completo
3. Threshold: F > threshold_fixed (e.g., F=3.0)
4. Cluster statistic: sum(F-values) o max(F-values)
5. TFCE opcional
6. FDR correction sobre markers
```

### Andrillon 2020
```
1. Ajustar LMM por electrodo → obtener t-values del predictor
2. Simple permutation:
   - Permutar labels del predictor dentro de subject × task
   - Re-ajustar modelo completo
3. Threshold: p < cluster_alpha (e.g., p=0.025)
4. Cluster statistic: sum(t-values)
5. Separar clusters positivos/negativos
6. Bonferroni correction sobre comparaciones
```

---

## Archivos Implementados ✅

```
Stats_andrillon/
├── README.md                          # ✅ Este archivo
├── QUICKSTART.md                      # ✅ Guía rápida
├── NEXT_STEPS.md                      # ✅ Desarrollo futuro
├── STATUS.md                          # ✅ Estado actual
├── FINAL_CHECKLIST.md                 # ✅ Verificación final
├── config_andrillon.yaml              # ✅ Configuración
├── lmm_permutation.py                 # ✅ Módulo: LMM + permutaciones
├── cluster_detection.py               # ✅ Módulo: Detección de clusters
├── andrillon_pipeline.py              # ✅ Pipeline principal
├── plot_andrillon_results.py          # ✅ Visualización
├── compare_pipelines.py               # ✅ Comparación con pipeline actual
├── run_andrillon_pipeline.py          # ✅ Script ejecución local
├── submit_andrillon_array.sh          # ✅ SLURM array job
├── get_marker_list.py                 # ✅ Helper para markers
└── results/                           # Directorio de resultados (auto-generado)
```

---

## Parámetros Clave del Paper

- **cluster_alpha**: 0.025 (p-value threshold)
- **n_permutations**: 1000
- **montecarlo_alpha**: 0.05
- **Bonferroni**: Aplicado cuando múltiples comparaciones (Fig 5 y 6)
- **Permutación**: Dentro de subject × task × electrode

---

## Ventajas de Esta Implementación

1. **Modularidad**: Componentes independientes y testeables
2. **Compatibilidad**: Usa mismos datos que pipeline actual
3. **Comparabilidad**: Permite comparar ambos enfoques
4. **Reproducibilidad**: Implementación fiel al paper
5. **Escalabilidad**: Paralelizable en cluster (SLURM)

---

## Próximos Pasos

1. ✅ **Crear estructura de directorios**
2. ✅ **Implementar `lmm_permutation.py`**
3. ✅ **Implementar `cluster_detection.py`**
4. ✅ **Crear `config_andrillon.yaml`**
5. ✅ **Implementar `andrillon_pipeline.py`**
6. ✅ **Crear scripts de ejecución**
7. ⏳ **Validar con datos de prueba** (SIGUIENTE PASO)
8. ⏳ **Ejecutar análisis completo**
9. ⏳ **Comparar con pipeline actual**

---

## Referencias

- **Andrillon et al. (2020)**: Metodología de cluster-permutation con LMM
- **Maris & Oostenveld (2007)**: Cluster-based permutation testing original
- **Código Matlab**: `/Users/nicolas.bruno/wanderIM/paper/`
  - `lme_perm.m`
  - `get_clusterperm_lme.m`
  - `wanderIM_LocalSleep_plotWavesPPties2_clusterPerm_rev1.m`

---

**Autor**: Nicolas Bruno  
**Fecha**: 2024-11-12  
**Versión**: 1.0
