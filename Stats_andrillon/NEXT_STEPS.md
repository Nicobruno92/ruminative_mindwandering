# Próximos Pasos - Pipeline Andrillon 2020

## ✅ Trabajo Completado

### 1. Documentación
- **README.md**: Plan detallado con metodología de Andrillon 2020
  - Comparación con pipeline actual
  - Estructura de archivos
  - Referencias al código Matlab original

### 2. Módulos Core Implementados

#### `lmm_permutation.py` (458 líneas)
- ✅ Función `fit_lmm_with_permutations()`: Ajusta LMM y genera permutaciones
- ✅ Permutación estratificada: Dentro de subject × task (método Andrillon)
- ✅ Función `fit_lmm_per_electrode()`: Loop sobre todos los electrodos
- ✅ Función `format_results_for_clustering()`: Formatea resultados para clustering
- ✅ Ejemplo de uso y testing incluido

**Características clave**:
- Permuta directamente las etiquetas del predictor (no residuales como Freedman-Lane)
- Mantiene estructura within-subject/task
- Compatible con cualquier fórmula LMM de statsmodels

#### `cluster_detection.py` (462 líneas)
- ✅ Función `find_clusters_andrillon()`: Detección completa de clusters
- ✅ Identificación de clusters candidatos (p < cluster_alpha)
- ✅ Agrupación espacial de electrodos vecinos
- ✅ Cálculo de cluster statistic (suma de t-values)
- ✅ Distribución nula de permutaciones
- ✅ P-values Monte Carlo
- ✅ Corrección Bonferroni
- ✅ Clase `ClusterResult` para resultados
- ✅ Ejemplo de uso y testing incluido

**Características clave**:
- Separa clusters positivos y negativos
- Usa matriz de adyacencia para vecindad espacial
- Implementa exactamente el algoritmo del paper

#### `config_andrillon.yaml` (188 líneas)
- ✅ Configuración completa del pipeline
- ✅ Parámetros específicos de Andrillon:
  - `cluster_alpha: 0.025`
  - `n_permutations: 1000`
  - `montecarlo_alpha: 0.05`
  - `stat_fun: "sum"`
  - `bonferroni_correction: true`
- ✅ Documentación inline de cada parámetro
- ✅ Comparación con pipeline actual
- ✅ Reutiliza paths y filtros del pipeline existente

---

## 🔄 Trabajo Pendiente

### Fase 1: Pipeline Principal (ALTA PRIORIDAD)

#### 1.1 `andrillon_pipeline.py`
**Objetivo**: Workflow completo que integra todos los módulos

**Funciones a implementar**:
```python
def run_andrillon_analysis(config_path):
    """
    Pipeline completo:
    1. Cargar configuración
    2. Cargar datos (reutilizar reader.py del pipeline actual)
    3. Preprocesar (normalización)
    4. Loop sobre markers:
       a. Loop sobre electrodos
       b. Ajustar LMM + permutaciones
       c. Detectar clusters
       d. Aplicar corrección múltiple
    5. Guardar resultados
    6. Generar visualizaciones
    """
    pass

def load_marker_data(marker_name, config):
    """Cargar datos de un marker (reutilizar reader.py)"""
    pass

def preprocess_data(data, config):
    """Normalización por sujeto (reutilizar helpers.py)"""
    pass

def save_results(clusters, output_path, marker_name):
    """Guardar resultados en pickle/CSV"""
    pass
```

**Archivos a reutilizar del pipeline actual**:
- `Statistics/reader.py`: Cargar datos de markers
- `Statistics/helpers.py`: Funciones auxiliares, normalización
- `Statistics/lmm_model.py`: Puede servir de referencia (pero usaremos statsmodels directamente)

**Estimación**: ~300-400 líneas

---

### Fase 2: Visualización

#### 2.1 `plot_andrillon_results.py`
**Objetivo**: Generar figuras de resultados

**Funciones a implementar**:
```python
def plot_cluster_topography(clusters, montage, marker_name):
    """
    Topoplot mostrando:
    - T-values por electrodo
    - Clusters significativos marcados
    - Similar a las figuras del paper Andrillon
    """
    pass

def plot_cluster_statistics(clusters):
    """Gráfico de barras con cluster statistics"""
    pass

def create_summary_report(all_results, output_path):
    """Reporte HTML/PDF con todos los resultados"""
    pass
```

**Estimación**: ~200-300 líneas

---

### Fase 3: Scripts de Ejecución

#### 3.1 `run_andrillon_pipeline.py`
**Objetivo**: Script para ejecución local

```python
#!/usr/bin/env python
"""
Run Andrillon 2020 pipeline locally.

Usage:
    python run_andrillon_pipeline.py --config config_andrillon.yaml
    python run_andrillon_pipeline.py --config config_andrillon.yaml --marker alpha
"""
import argparse
from andrillon_pipeline import run_andrillon_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--marker", default=None)
    args = parser.parse_args()
    
    run_andrillon_analysis(args.config, marker=args.marker)
```

**Estimación**: ~50-100 líneas

#### 3.2 `submit_andrillon_array.sh`
**Objetivo**: SLURM array job para paralelizar por marker

```bash
#!/bin/bash
#SBATCH --job-name=andrillon_cluster
#SBATCH --array=0-N  # N = número de markers
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Activar environment
conda activate ML  # o el environment apropiado

# Get marker from array
MARKERS=(alpha beta delta gamma theta)
MARKER=${MARKERS[$SLURM_ARRAY_TASK_ID]}

# Run pipeline
python run_andrillon_pipeline.py \
    --config config_andrillon.yaml \
    --marker $MARKER
```

**Estimación**: ~100 líneas

---

### Fase 4: Validación y Testing

#### 4.1 Tests Unitarios
**Crear**: `tests/test_permutation.py`
```python
def test_permutation_preserves_groups():
    """Verificar que permutación mantiene estructura within-group"""
    pass

def test_permutation_changes_values():
    """Verificar que valores realmente se permutan"""
    pass
```

**Crear**: `tests/test_clustering.py`
```python
def test_cluster_detection_simple_case():
    """Test con datos sintéticos simples"""
    pass

def test_cluster_statistic_calculation():
    """Verificar cálculo de suma de t-values"""
    pass

def test_montecarlo_pvalue():
    """Verificar cálculo de p-values Monte Carlo"""
    pass
```

**Estimación**: ~200-300 líneas total

#### 4.2 Validación con Datos Reales
1. Ejecutar pipeline con 1 marker (e.g., alpha)
2. Verificar que:
   - Permutaciones se ejecutan correctamente
   - Clusters se detectan
   - Resultados tienen sentido
3. Comparar con pipeline actual

---

### Fase 5: Comparación con Pipeline Actual

#### 5.1 `compare_pipelines.py`
**Objetivo**: Comparar resultados de ambos pipelines

```python
def load_results(pipeline_type, marker):
    """Cargar resultados de pipeline actual o Andrillon"""
    pass

def compare_clusters(clusters_current, clusters_andrillon):
    """
    Comparar:
    - Número de clusters detectados
    - Electrodos en cada cluster
    - Overlap espacial
    - P-values
    """
    pass

def plot_comparison(clusters_current, clusters_andrillon):
    """Visualización lado a lado"""
    pass

def generate_comparison_report():
    """Reporte completo de diferencias"""
    pass
```

**Estimación**: ~300-400 líneas

---

## 📋 Checklist de Implementación

### Fase 1: Core Pipeline
- [ ] Implementar `andrillon_pipeline.py`
- [ ] Integrar con `reader.py` del pipeline actual
- [ ] Integrar con `helpers.py` para normalización
- [ ] Testear con 1 marker pequeño

### Fase 2: Visualización
- [ ] Implementar `plot_andrillon_results.py`
- [ ] Generar topoplots con clusters
- [ ] Crear reporte de resultados

### Fase 3: Ejecución
- [ ] Crear `run_andrillon_pipeline.py`
- [ ] Crear `submit_andrillon_array.sh`
- [ ] Testear ejecución local
- [ ] Testear ejecución en cluster

### Fase 4: Validación
- [ ] Crear tests unitarios
- [ ] Ejecutar con datos reales (1 marker)
- [ ] Verificar resultados
- [ ] Ejecutar pipeline completo (todos los markers)

### Fase 5: Comparación
- [ ] Implementar `compare_pipelines.py`
- [ ] Ejecutar ambos pipelines con mismos datos
- [ ] Generar reporte de comparación
- [ ] Analizar diferencias

---

## 🎯 Prioridades Inmediatas

### 1. **Implementar `andrillon_pipeline.py`** (URGENTE)
Este es el archivo que falta para tener un pipeline funcional end-to-end.

**Pasos**:
1. Copiar estructura de `Statistics/run_pipeline.py`
2. Adaptar para usar nuestros módulos (`lmm_permutation.py`, `cluster_detection.py`)
3. Reutilizar `reader.py` y `helpers.py` del pipeline actual

### 2. **Crear script de ejecución simple** (URGENTE)
Para poder testear rápidamente.

### 3. **Testear con 1 marker** (URGENTE)
Validar que todo funciona antes de escalar.

---

## 💡 Notas Importantes

### Reutilización de Código Existente

**Del pipeline actual (`Statistics/`)**, podemos reutilizar:
- ✅ `reader.py`: Cargar datos de markers
- ✅ `helpers.py`: Normalización, funciones auxiliares
- ✅ Matriz de adyacencia (ya calculada)
- ✅ Montage (CACS-64_REF.bvef)
- ❌ `cluster_test.py`: NO (usa MNE, diferente metodología)
- ❌ `lmm_model.py`: Referencia, pero usamos statsmodels directamente

### Diferencias Clave a Mantener

1. **Permutación**: Andrillon permuta labels, NO residuales
2. **Threshold**: P-value (0.025), NO t-value fijo
3. **Cluster stat**: Suma de t-values, NO F-values
4. **Corrección**: Bonferroni, NO FDR

### Estimación de Tiempo

- **Fase 1 (Pipeline)**: 4-6 horas
- **Fase 2 (Visualización)**: 2-3 horas
- **Fase 3 (Scripts)**: 1-2 horas
- **Fase 4 (Testing)**: 3-4 horas
- **Fase 5 (Comparación)**: 2-3 horas

**Total estimado**: 12-18 horas de trabajo

---

## 📚 Referencias

### Código Matlab Original (Andrillon 2020)
- `/Users/nicolas.bruno/wanderIM/paper/lme_perm.m`
- `/Users/nicolas.bruno/wanderIM/paper/get_clusterperm_lme.m`
- `/Users/nicolas.bruno/wanderIM/paper/wanderIM_LocalSleep_plotWavesPPties2_clusterPerm_rev1.m`

### Pipeline Actual
- `/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Statistics/`

### Paper
- Andrillon et al. (2020), líneas 102-115 (metodología estadística)

---

## 🚀 Comando para Empezar

```bash
# 1. Ir al directorio
cd /Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Stats_andrillon

# 2. Testear módulos existentes
python lmm_permutation.py  # Debería ejecutar el ejemplo
python cluster_detection.py  # Debería ejecutar el ejemplo

# 3. Siguiente: Implementar andrillon_pipeline.py
# (Ver estructura sugerida arriba)
```

---

**Última actualización**: 2024-11-12  
**Estado**: Módulos core completados, listo para integración
