# Quick Start Guide - Andrillon 2020 Pipeline

## Archivos Listos para Ejecutar

✅ **Módulos Core**
- `lmm_permutation.py` - LMM + permutaciones
- `cluster_detection.py` - Detección de clusters
- `andrillon_pipeline.py` - Pipeline completo

✅ **Configuración**
- `config_andrillon.yaml` - Parámetros del análisis

✅ **Scripts de Ejecución**
- `run_andrillon_pipeline.py` - Ejecución local
- `submit_andrillon_array.sh` - SLURM array job
- `get_marker_list.py` - Helper para listar markers

---

## Ejecución Local (Testing)

### 1. Testear con 1 Marker

```bash
cd /Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Stats_andrillon

# Activar environment
conda activate ML  # o el environment apropiado

# Ejecutar con 1 marker para testing
python run_andrillon_pipeline.py \
    --config config_andrillon.yaml \
    --marker EEG_psd_bands_spectralpower_alpha
```

### 2. Ejecutar Todos los Markers

```bash
python run_andrillon_pipeline.py --config config_andrillon.yaml
```

---

## Ejecución en Cluster (Producción)

### Paso 1: Obtener Lista de Markers

```bash
# Ver lista de markers
python get_marker_list.py --config config_andrillon.yaml

# Ver en formato bash (para copiar a submit script)
python get_marker_list.py --config config_andrillon.yaml --format bash

# Ver solo el número (para --array parameter)
python get_marker_list.py --config config_andrillon.yaml --format count
```

### Paso 2: Actualizar SLURM Script

Editar `submit_andrillon_array.sh`:

1. **Actualizar `--array` parameter** (línea 6):
   ```bash
   #SBATCH --array=0-N  # donde N = número de markers - 1
   ```

2. **Actualizar `MARKER_LIST`** (línea 38+):
   ```bash
   MARKER_LIST=(
       "marker1"
       "marker2"
       # ... copiar output de get_marker_list.py --format bash
   )
   ```

### Paso 3: Crear Directorio de Logs

```bash
mkdir -p logs
```

### Paso 4: Enviar Job

```bash
# Desde el cluster
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Stats_andrillon

# Cargar módulos necesarios
module load proxy

# Enviar array job
sbatch submit_andrillon_array.sh
```

### Paso 5: Monitorear Jobs

```bash
# Ver estado de jobs
squeue -u $USER

# Ver logs en tiempo real
tail -f logs/andrillon_*.out

# Ver errores
tail -f logs/andrillon_*.err
```

---

## Verificar Resultados

### Estructura de Output

```
results/andrillon_cluster/
└── onoff_time_on_task_valence_selfother_time/  # Carpeta del modelo
    ├── EEG_psd_bands_spectralpower_alpha/
    │   ├── EEG_psd_bands_spectralpower_alpha_results.pkl
    │   └── EEG_psd_bands_spectralpower_alpha_clusters.csv
    ├── EEG_psd_bands_spectralpower_beta/
    │   └── ...
    └── ...
```

### Revisar Resultados

```python
import pickle
import pandas as pd

# Cargar resultados
with open('results/andrillon_cluster/MODEL/MARKER/MARKER_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Ver clusters
for cluster in results['clusters']:
    print(cluster)

# Leer CSV
df = pd.read_csv('results/andrillon_cluster/MODEL/MARKER/MARKER_clusters.csv')
print(df)
```

---

## Troubleshooting

### Error: "Module not found: reader"

**Solución**: El pipeline busca `reader.py` en `../Statistics/`. Verifica que existe:
```bash
ls ../Statistics/reader.py
```

### Error: "Adjacency matrix not found"

**Solución**: Verifica que el montage existe:
```bash
ls ../Preprocessing_pipeline_new/CACS-64_REF.bvef
```

### Error: "No data found for marker"

**Solución**: Verifica que los datos existen:
```bash
ls /network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features/
```

### Jobs Fallan en Cluster

**Revisar logs**:
```bash
# Ver último error
tail -20 logs/andrillon_*_0.err

# Buscar errores específicos
grep -i error logs/andrillon_*.err
```

---

## Configuración Avanzada

### Cambiar Número de Permutaciones

Editar `config_andrillon.yaml`:
```yaml
andrillon_clustering:
  n_permutations: 100  # Reducir para testing rápido
  # n_permutations: 1000  # Valor del paper (producción)
```

### Cambiar Threshold de Clusters

```yaml
andrillon_clustering:
  cluster_alpha: 0.025  # Más estricto
  # cluster_alpha: 0.05  # Menos estricto
```

### Desactivar Bonferroni

```yaml
andrillon_clustering:
  bonferroni_correction: false
```

### Cambiar Fórmula LMM

```yaml
lmm:
  formula: "power ~ onoff + (1|subject)"  # Más simple
  predictor_of_interest: "onoff"
```

---

## Comparación con Pipeline Actual

Para comparar resultados con el pipeline actual:

```bash
# Ejecutar pipeline actual
cd ../Statistics
python run_pipeline.py

# Ejecutar pipeline Andrillon
cd ../Stats_andrillon
python run_andrillon_pipeline.py --config config_andrillon.yaml

# Comparar resultados (implementar compare_pipelines.py)
```

---

## Próximos Pasos

1. ✅ **Testing**: Ejecutar con 1 marker localmente
2. ✅ **Validación**: Verificar que clusters tienen sentido
3. ✅ **Producción**: Ejecutar todos los markers en cluster
4. ⏳ **Visualización**: Implementar `plot_andrillon_results.py`
5. ⏳ **Comparación**: Implementar `compare_pipelines.py`

---

## Contacto

Para problemas o preguntas:
- Revisar `README.md` para detalles metodológicos
- Revisar `NEXT_STEPS.md` para desarrollo futuro
- Revisar logs en `logs/` para errores específicos
